"""Dense exact-diagonalization QEI infimum (Layer 7, L5): the second implementation.

Why it exists
-------------
:mod:`vacuum.interacting.qei_exact` computes lambda_min(O_f) through three
approximations of its own (the Fock cutoff n_max, the Trotter step of the
operator-space TEBD, and the operator-MPO truncation).  PLAN.md's anomaly
protocol asks for a second, independent implementation before any of its
numbers is read as physics.  This module is that implementation, for chains
small enough to diagonalize: the smeared operator is built by DENSE
Heisenberg evolution -- the full eigendecomposition H = V diag(w) V^T of the
truncated-boson chain, in which

    <a| O_f |b> = h_ab F(w_a - w_b),      F(nu) = int dt f(t)^2 e^{i nu t},

so there is no Trotter step, no time grid (with the analytic Fourier
transform of f^2; the MPO route's trapezoid grid is available as a kernel so
that its quadrature can be matched exactly), and no operator truncation.  The
one approximation left is the Fock cutoff, shared with the MPO route by
construction (same ``local_boson_ops``, same basis frequency, same PSD h_x
of ``local_energy_operator_dense``), which is what makes the two routes
comparable at the same (L, n_max): their difference is the MPO route's
Trotter + truncation error and nothing else.

Because O_f = V M V^T with M = h_V o F (h_V = V^T h V, o elementwise),
lambda_min(O_f) = lambda_min(M) and nothing larger than M is ever formed; the
vacuum Omega is the lowest column of V and <Omega|O_f|Omega> = M_00 =
||f||^2 <Omega|h_x|Omega> exactly (an eigenstate of H).

Parity.  H, h_x and hence O_f commute with the global reflection phi -> -phi,
P = (-1)^{sum_i n_i} (every term is even in phi, and pi^2 is even too), so the
problem splits into two blocks of ~d^L/2 states.  Each is diagonalized
separately and the infimum is the smaller of the two sector minima -- both
are reported, because the extremal state of a smeared energy need not share
the vacuum's parity.  This halves the memory and cuts the eigh cost ~4x,
which is what makes n_max = 5 at L = 6 (23 328 states per sector) affordable.

Cost: eigh of each sector, O((d^L / 2)^3), plus two dense matmuls for h_V.
L = 6 up to n_max = 5, L = 7 up to n_max = 3 and L = 8 up to n_max = 3 fit
in tens of GB and minutes to tens of minutes; beyond that the MPO route is
the only one.  Pure numpy/scipy -- TeNPy is not needed: the truncated
Hamiltonian is assembled here straight from the model's definition (the
``phi4`` docstring), independently of the TeNPy ``H_bond`` terms, and
``tests/test_interacting_qei_ed.py`` checks that the two agree.

The matrix-free route (:func:`qei_exact_lanczos`) lifts the dense limit: O_f
is never formed; it is applied to vectors on the MPO route's own trapezoid
grid by Chebyshev time evolution of the parity block (exact to machine
precision, no Trotter step) and its lowest eigenvalue found by Lanczos, so
L = 6 at n_max = 6 (117 649 states) and L = 5 at n_max = 8 (59 049) are
minutes to an hour on two threads.  It agrees with the dense route at the
same grid to 1e-11 relative (tests; L = 6, n_max = 3: 4e-13 and 5e-12
against the archived rows), and a finer grid measures the quadrature.
"""

from __future__ import annotations

import math
import time
from typing import NamedTuple, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from scipy.special import jv

from vacuum.inequalities.qei import FHandle, _gauss_legendre_grid

from .phi4 import _resolve_omega, hartree_mass_squared, local_boson_ops
from .qei_exact import _time_grid, free_exact_infimum, local_energy_operator_dense

__all__ = [
    "dense_hamiltonian",
    "dense_local_energy",
    "parity_sectors",
    "time_kernel",
    "QEIDenseResult",
    "qei_exact_dense",
    "smeared_operator_dense",
    "ChebyshevPropagator",
    "SmearedOperatorMatvec",
    "QEILanczosResult",
    "qei_exact_lanczos",
]


def _embed(L, d, mats):
    """kron-embed {site: (d, d) matrix} into the d^L chain (sparse CSR)."""
    out = sp.identity(1, format="csr")
    for j in range(L):
        A = sp.csr_matrix(mats[j]) if j in mats else sp.identity(d, format="csr")
        out = sp.kron(out, A, format="csr")
    return out


def dense_hamiltonian(L, m, lam, n_max, omega, bc="dirichlet"):
    """H of the truncated chain as a sparse CSR matrix (d^L x d^L).

    H = sum_i [(1/2) Psq_i + ((1/2) m^2 + c_i) Xsq_i + (lam/24) X4_i]
        - sum_i X_i X_{i+1},

    with c_i = 1 in the interior (half of each adjacent bond's Xsq), and at
    the ends c = 1 for 'dirichlet' (the wall bond (phi_0 - 0)^2 / 2) or 1/2
    for 'open' -- the same projected operators and the same coefficients as
    the TeNPy model of ``phi4.phi4_model_class`` (tested against it).
    """
    L = int(L)
    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")
    if bc not in ("dirichlet", "open"):
        raise ValueError(f"bc must be 'dirichlet' or 'open', got {bc!r}")
    ops = local_boson_ops(n_max, omega)
    d = int(n_max) + 1
    m = float(m)
    lam = float(lam)
    c = np.ones(L)
    if bc == "open":
        c[0] = c[L - 1] = 0.5
    H = sp.csr_matrix((d**L, d**L))
    for i in range(L):
        on = 0.5 * ops["Psq"] + (0.5 * m * m + c[i]) * ops["Xsq"]
        if lam != 0.0:
            on = on + (lam / 24.0) * ops["X4"]
        H = H + _embed(L, d, {i: on})
    for i in range(L - 1):
        H = H - _embed(L, d, {i: ops["X"], i + 1: ops["X"]})
    H = 0.5 * (H + H.T)
    return H.tocsr()


def dense_local_energy(L, m, lam, n_max, omega, site, bc="dirichlet"):
    """h_site (``local_energy_operator_dense``) embedded in the chain (sparse CSR)."""
    d = int(n_max) + 1
    sites, H3 = local_energy_operator_dense(n_max, omega, m, lam, site, L, bc)
    k = len(sites)
    left = sp.identity(d ** sites[0], format="csr")
    right = sp.identity(d ** (int(L) - 1 - sites[-1]), format="csr")
    h = sp.kron(sp.kron(left, sp.csr_matrix(H3), format="csr"), right, format="csr")
    assert h.shape == (d ** int(L),) * 2 and k == len(sites)
    return h.tocsr()


def parity_sectors(L, d):
    """Index arrays (even, odd) of the global occupation parity (-1)^{sum n_i}.

    kron ordering: site 0 is the slowest index.
    """
    par = np.zeros(1, dtype=np.int64)
    for _ in range(int(L)):
        par = ((par[:, None] + np.arange(int(d))[None, :]) % 2).ravel()
    return np.flatnonzero(par == 0), np.flatnonzero(par == 1)


def time_kernel(f: FHandle, kind="analytic", dt=None, n_panels=24, nodes=8):
    """F(nu) = int dt f(t)^2 cos(nu t) as a vectorized callable.

    'analytic'  -- the handle's Fourier transform ``f2hat`` (exact, all t);
    'trapezoid' -- the MPO route's grid: t_k = k dt on [0, t_hi], weights
                   1/2 at the ends, doubled by t -> -t (the O_f = Q + Q^T of
                   ``qei_exact``), so only the Trotter error separates the two
                   routes at the same dt;
    'gauss'     -- composite Gauss-Legendre over the handle's support.
    Returns (F, meta).
    """
    if kind == "analytic":
        if getattr(f, "f2hat", None) is None:
            raise ValueError("the sampling handle has no f2hat; use kind='trapezoid' or 'gauss'")

        def F(nu):
            return np.real(f.f2hat(nu))

        return F, {"kernel": "analytic"}
    if kind == "trapezoid":
        if dt is None:
            raise ValueError("kind='trapezoid' needs dt")
        times, w, dt_eff = _time_grid(f, dt)
        f2w = 2.0 * w * np.asarray(f.f(times), float) ** 2

        def F(nu):
            nu = np.asarray(nu, float)
            out = np.zeros_like(nu)
            for tk, ck in zip(times, f2w):
                out += ck * np.cos(nu * tk)
            return out

        return F, {"kernel": "trapezoid", "dt": float(dt_eff), "n_steps": int(times.size - 1)}
    if kind == "gauss":
        ts, ws = _gauss_legendre_grid(f.t_lo, f.t_hi, n_panels, nodes)
        f2w = ws * np.asarray(f.f(ts), float) ** 2

        def F(nu):
            nu = np.asarray(nu, float)
            out = np.zeros_like(nu)
            for tk, ck in zip(ts, f2w):
                out += ck * np.cos(nu * tk)
            return out

        return F, {"kernel": "gauss", "n_panels": int(n_panels), "nodes": int(nodes)}
    raise ValueError(f"kind must be 'analytic', 'trapezoid' or 'gauss', got {kind!r}")


def _sector_eig(H, h, idx):
    """Eigendecompose one parity block of H and return (w, V, h_V)."""
    Hs = H[idx][:, idx].toarray()
    w, V = np.linalg.eigh(Hs)
    del Hs
    hs = h[idx][:, idx]
    hV = V.T @ (hs @ V)
    return w, V, hV


def _apply_kernel_inplace(hV, w, F, chunk=2048):
    """hV <- hV o F(w_a - w_b), row-chunked so the n^2 kernel is never stored twice."""
    n = w.size
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        hV[a:b] *= F(w[a:b, None] - w[None, :])
    return hV


def _lowest(M, k_dense=1200):
    """(lambda_min, eigenvector) of a dense symmetric matrix."""
    n = M.shape[0]
    if n <= k_dense:
        ev, U = np.linalg.eigh(M)
        return float(ev[0]), U[:, 0]
    ev, U = spla.eigsh(M, k=1, which="SA", tol=1e-13, maxiter=20000)
    return float(ev[0]), U[:, 0]


class QEIDenseResult(NamedTuple):
    E_min: float                 # lambda_min(O_f) - <Omega|O_f|Omega>
    E_star: float                # lambda_min(O_f)
    E_ref: float                 # <Omega|O_f|Omega> = ||f||^2 <Omega|h_x|Omega>
    E_star_by_sector: dict       # {0: even-sector minimum, 1: odd-sector minimum}
    sector_star: int             # parity of the extremal state
    E_vac: float                 # ground energy of H (even sector)
    h_vac: float                 # <Omega|h_x|Omega>
    E_free_exact: float          # Gaussian infimum, same f/m/L (Williamson)
    E_hartree_exact: float       # ... at the Hartree mass
    mu_H2: float
    psi_star: Optional[np.ndarray]   # extremal state in the SITE basis (None if not kept)
    meta: dict


def smeared_operator_dense(L, m, lam, n_max, site, f: FHandle, omega="auto", bc="dirichlet",
                           kernel="analytic", dt=None):
    """The full d^L x d^L O_f in the site basis (tests / tiny chains only).

    Assembled from the parity blocks; returns (O_f, H_dense, sectors).
    """
    L = int(L)
    d = int(n_max) + 1
    om = _resolve_omega(omega, L, m, lam, bc)
    H = dense_hamiltonian(L, m, lam, n_max, om, bc)
    h = dense_local_energy(L, m, lam, n_max, om, site, bc)
    F, _ = time_kernel(f, kernel, dt=dt)
    O = np.zeros((d**L, d**L))
    sectors = parity_sectors(L, d)
    for idx in sectors:
        w, V, hV = _sector_eig(H, h, idx)
        M = _apply_kernel_inplace(hV, w, F)
        O[np.ix_(idx, idx)] = V @ M @ V.T
    return 0.5 * (O + O.T), H.toarray(), sectors


def qei_exact_dense(L, m, lam, n_max, site, f: FHandle, *, omega="auto", bc="dirichlet",
                    kernel="analytic", dt=None, keep_state=False):
    """The exact lattice QEI infimum of the truncated chain by dense ED.

    L, m, lam, n_max, omega ('auto' | 'hartree' | float), bc: as in
    ``phi4.phi4_model``; site: the worldline; f: the sampling function;
    kernel / dt: see :func:`time_kernel` -- `kernel` may be a sequence of
    kinds, all evaluated from the ONE eigendecomposition (the first is the
    reported one, the others land in ``meta['by_kernel']``).  Returns
    :class:`QEIDenseResult`.  E_min is exact for the truncated chain up to
    floating point -- the Fock cutoff is the only approximation, and it is
    the same cutoff the MPO route uses.
    """
    t0 = time.time()
    L = int(L)
    x = int(site)
    m = float(m)
    lam = float(lam)
    d = int(n_max) + 1
    om = _resolve_omega(omega, L, m, lam, bc)
    if bc != "dirichlet":
        raise ValueError("the free Gaussian references exist for bc='dirichlet' only")
    H = dense_hamiltonian(L, m, lam, n_max, om, bc)
    h = dense_local_energy(L, m, lam, n_max, om, x, bc)
    kinds = (kernel,) if isinstance(kernel, str) else tuple(kernel)
    kernels = [time_kernel(f, k, dt=dt) for k in kinds]
    f2_norms = [float(F(np.zeros(1))[0]) for F, _ in kernels]
    t_build = time.time() - t0

    sectors = parity_sectors(L, d)
    E_star_by = {}          # (kernel index, parity) -> lambda_min of that block
    vec_by = {}
    E_vac = None
    h_vac = None
    t_eig = {}
    for p, idx in enumerate(sectors):
        ts = time.time()
        w, V, hV = _sector_eig(H, h, idx)
        if p == 0:
            E_vac = float(w[0])
            h_vac = float(hV[0, 0])
        for ki, (F, _km) in enumerate(kernels):
            M = _apply_kernel_inplace(hV.copy() if ki < len(kernels) - 1 else hV, w, F)
            e, u = _lowest(M)
            E_star_by[(ki, p)] = e
            if keep_state:
                full = np.zeros(d**L)
                full[idx] = V @ u
                vec_by[(ki, p)] = full
            del M
        del V, hV
        t_eig[p] = time.time() - ts
    E_free = free_exact_infimum(L, x, f, m, bc)
    mu_H2 = hartree_mass_squared(L, m, lam, bc)
    E_H = free_exact_infimum(L, x, f, math.sqrt(mu_H2), bc)
    by_kernel = {}
    for ki, ((_F, km), f2n) in enumerate(zip(kernels, f2_norms)):
        secs = {p: E_star_by[(ki, p)] for p in range(len(sectors))}
        p_star = min(secs, key=secs.get)
        by_kernel[kinds[ki]] = {"E_star": float(secs[p_star]), "E_ref": float(f2n * h_vac),
                                "E_min": float(secs[p_star] - f2n * h_vac),
                                "sector_star": int(p_star),
                                "E_star_by_sector": {int(k): float(v) for k, v in secs.items()},
                                "f2_norm": f2n, **km}
    main = by_kernel[kinds[0]]
    meta = {"L": L, "m": m, "lam": lam, "n_max": int(n_max), "omega": float(om), "bc": bc,
            "site": x, "f": f.label, "f2_norm": f2_norms[0], "dim": int(d**L),
            "dim_sectors": [int(idx.size) for idx in sectors],
            "time_build_s": t_build, "time_sector_s": t_eig, "time_total_s": time.time() - t0,
            "by_kernel": by_kernel, **kernels[0][1]}
    return QEIDenseResult(E_min=main["E_min"], E_star=main["E_star"], E_ref=main["E_ref"],
                          E_star_by_sector=main["E_star_by_sector"],
                          sector_star=main["sector_star"], E_vac=E_vac, h_vac=h_vac,
                          E_free_exact=float(E_free), E_hartree_exact=float(E_H),
                          mu_H2=float(mu_H2),
                          psi_star=(vec_by.get((0, main["sector_star"])) if keep_state else None),
                          meta=meta)


# --------------------------------------------------------------------------
# The matrix-free route: Lanczos on O_f applied by Chebyshev time evolution
# --------------------------------------------------------------------------
#
# The dense route above needs the full eigendecomposition of each parity block,
# O((d^L/2)^3) time and O((d^L/2)^2) memory, which ends at L = 6, n_max = 4-5.
# But lambda_min(O_f) only needs O_f *applied* to vectors.  With the MPO route's
# own trapezoid grid t_k = k dt (k = 0..n, n dt = t_hi) and c_k = w_k f(t_k)^2,
#
#     O_f = Q + Q^T,   Q = sum_k c_k e^{iHt_k} h e^{-iHt_k},
#
# and because H and h are real symmetric, Q^T = conj(Q), so for a REAL vector v
#
#     O_f v = 2 Re(Q v),   Q v = sum_k c_k e^{iHt_k} h v_k,   v_k = e^{-iHt_k} v.
#
# Q v costs 2n short-time propagations by dt: the forward chain v_k = e^{-iH dt}
# v_{k-1} (n steps, the v_k kept), then the Horner sum u <- e^{+iH dt} u + c_k h
# v_k from k = n down to 0 (n steps), so that u = sum_k c_k e^{iHt_k} h v_k.
# Each propagation is the Chebyshev expansion of e^{-iH tau} on the block's
# spectral interval [E_lo, E_hi] (Tal-Ezer & Kosloff 1984): with a = (E_hi -
# E_lo)/2, b = (E_hi + E_lo)/2 and Htilde = (H - b)/a,
#
#     e^{-iH tau} = e^{-ib tau} sum_{k>=0} (2 - delta_k0) (-i)^k J_k(a tau) T_k(Htilde),
#
# truncated where the Bessel factor falls below eps (super-exponentially past
# k = a tau), so every propagation is exact to machine precision: no Trotter
# step, no operator truncation, and only the Fock cutoff shared with the MPO
# route.  ARPACK's Lanczos ('SA') on the resulting real symmetric LinearOperator
# returns lambda_min of the block, one sector at a time, warm-started from the
# block's lowest H eigenvector; the extremal state is the Ritz vector.
# Validated against the dense route at the same grid (tests) to 1e-11 relative.
#
# Cost: (2n) x (a dt + ~30) sparse matvecs per O_f application, times the
# Lanczos count (~100-300); memory O(n d^L).  L = 6 at n_max = 6 (117 649 states,
# 58 825 per block) and L = 5 at n_max = 8 (59 049) are minutes to an hour on
# two threads.


class ChebyshevPropagator:
    """e^{-i H tau} on a real symmetric sparse H with spectrum inside [e_lo, e_hi].

    Chebyshev expansion (module comment) truncated where |J_k(a tau)| < eps;
    ``n_terms`` is the number of Chebyshev terms (n_terms - 1 matvecs per call)
    and ``matvecs`` counts the sparse products applied so far.  tau may be
    negative (the backward step).  H is converted to complex CSR once.
    """

    def __init__(self, H, e_lo, e_hi, tau, eps=1e-16):
        self.H = sp.csr_matrix(H, dtype=np.complex128)
        self.a = 0.5 * (float(e_hi) - float(e_lo))
        self.b = 0.5 * (float(e_hi) + float(e_lo))
        self.tau = float(tau)
        if self.a <= 0.0:
            raise ValueError("e_hi must exceed e_lo")
        z = self.a * self.tau
        K = int(abs(z)) + 60
        ks = np.arange(K + 1)
        J = jv(ks, z)
        keep = np.flatnonzero(np.abs(J) > float(eps))
        K = int(keep[-1]) if keep.size else 0
        ks = ks[:K + 1]
        self.c = np.where(ks == 0, 1.0, 2.0) * (-1j) ** ks * J[:K + 1] * np.exp(-1j * self.b * self.tau)
        self.n_terms = int(K + 1)
        self.matvecs = 0

    def _Ht(self, v):
        self.matvecs += 1
        return (self.H @ v - self.b * v) / self.a

    def __call__(self, v):
        v = np.asarray(v, dtype=np.complex128)
        t0 = v
        acc = self.c[0] * t0
        if self.n_terms == 1:
            return acc
        t1 = self._Ht(v)
        acc = acc + self.c[1] * t1
        for k in range(2, self.n_terms):
            t2 = 2.0 * self._Ht(t1) - t0
            acc += self.c[k] * t2
            t0, t1 = t1, t2
        return acc


class SmearedOperatorMatvec:
    """O_f = sum_k c_k [h(t_k) + h(-t_k)] applied to real vectors (module comment).

    H, h: real symmetric sparse matrices on one parity block; times / weights:
    the trapezoid grid of ``qei_exact._time_grid`` (t_k >= 0, weights halved at
    the ends); f2: f(t_k)^2.  ``matvec(v)`` returns O_f v for real v; ``count``
    is the number of O_f applications, ``propagator_matvecs`` the sparse
    products behind them.
    """

    def __init__(self, H, h, e_lo, e_hi, times, weights, f2, eps=1e-16):
        times = np.asarray(times, float)
        if times.size < 2:
            raise ValueError("need at least two grid points")
        dt = float(times[1] - times[0])
        if not np.allclose(np.diff(times), dt):
            raise ValueError("the time grid must be uniform")
        self.dt = dt
        self.coef = np.asarray(weights, float) * np.asarray(f2, float)
        self.n = int(times.size - 1)
        self.fwd = ChebyshevPropagator(H, e_lo, e_hi, +dt, eps)
        self.bwd = ChebyshevPropagator(H, e_lo, e_hi, -dt, eps)
        self.h = sp.csr_matrix(h, dtype=np.complex128)
        self.dim = int(self.h.shape[0])
        self.count = 0

    @property
    def n_terms(self):
        return self.fwd.n_terms

    @property
    def propagator_matvecs(self):
        return self.fwd.matvecs + self.bwd.matvecs

    def apply_complex(self, v):
        """Q v = sum_k c_k e^{iHt_k} h e^{-iHt_k} v (complex)."""
        vs = [np.asarray(v, dtype=np.complex128)]
        for _ in range(self.n):
            vs.append(self.fwd(vs[-1]))
        u = self.coef[self.n] * (self.h @ vs[self.n])
        for k in range(self.n - 1, -1, -1):
            u = self.bwd(u) + self.coef[k] * (self.h @ vs[k])
        return u

    def matvec(self, v):
        self.count += 1
        v = np.asarray(v, dtype=float).reshape(-1)
        return 2.0 * np.real(self.apply_complex(v))

    def f2_norm(self):
        """sum_k 2 c_k = the trapezoid ||f||^2 (F_trap(0))."""
        return float(2.0 * np.sum(self.coef))

    def as_linear_operator(self):
        return spla.LinearOperator((self.dim, self.dim), matvec=self.matvec, dtype=np.float64)


class QEILanczosResult(NamedTuple):
    E_min: float                 # lambda_min(O_f) - <Omega|O_f|Omega> on the grid (MPO route's quadrature)
    E_star: float                # lambda_min(O_f)
    E_ref: float                 # ||f||^2_trap <Omega|h_x|Omega>
    E_star_by_sector: dict       # {0: even-block minimum, 1: odd-block minimum}
    sector_star: int
    E_vac: float
    h_vac: float
    E_free_exact: float          # Gaussian infimum, same f/m/L (Williamson)
    E_hartree_exact: float       # ... at the Hartree mass
    mu_H2: float
    psi_star: Optional[np.ndarray]   # extremal state in the site basis (None if not kept)
    meta: dict


def _block(A, idx):
    return A[idx][:, idx].tocsr()


def qei_exact_lanczos(L, m, lam, n_max, site, f: FHandle, *, dt, dt_fine=None, omega="auto",
                      bc="dirichlet", tol=1e-12, ncv=None, maxiter=None, cheb_eps=1e-16,
                      bound_pad=0.02, sectors=(0, 1), keep_state=False):
    """The exact lattice QEI infimum of the truncated chain, matrix-free (module comment).

    Same (L, m, lam, n_max, omega, bc, site, f) conventions as
    :func:`qei_exact_dense`; dt: the trapezoid grid step (adjusted so that n dt
    = f.t_hi exactly, as the MPO route does), so the number is directly
    comparable with ``qei_exact_mps`` at the same dt up to its Trotter and
    truncation errors, and with ``qei_exact_dense(kernel='trapezoid', dt=dt)``
    up to floating point; dt_fine: if given, the winning block is re-solved on
    the finer grid, warm-started from the coarse extremal vector (the
    quadrature check, in ``meta['fine']``); tol / ncv / maxiter: ARPACK's;
    sectors: which parity blocks to solve (both by default -- the extremal
    state need not share the vacuum's parity).  Returns
    :class:`QEILanczosResult`; ``meta`` carries the block dimensions, the
    spectral bounds, the Chebyshev order, the matvec counts, the Lanczos
    residuals ||O_f u - E u|| and the wall times.
    """
    t_start = time.time()
    L = int(L)
    x = int(site)
    m = float(m)
    lam = float(lam)
    d = int(n_max) + 1
    om = _resolve_omega(omega, L, m, lam, bc)
    if bc != "dirichlet":
        raise ValueError("the free Gaussian references exist for bc='dirichlet' only")
    H = dense_hamiltonian(L, m, lam, n_max, om, bc)
    h = dense_local_energy(L, m, lam, n_max, om, x, bc)
    times, w, dt_eff = _time_grid(f, dt)
    f2 = np.asarray(f.f(times), float) ** 2
    t_build = time.time() - t_start
    blocks = parity_sectors(L, d)

    E_star_by = {}
    vec_by = {}
    info = {}
    # the vacuum lives in the even block whether or not that block's O_f is solved
    H0 = _block(H, blocks[0])
    e0, v0 = spla.eigsh(H0, k=1, which="SA", tol=1e-14, ncv=min(40, H0.shape[0] - 1))
    Omega = np.real(v0[:, 0])
    Omega /= np.linalg.norm(Omega)
    E_vac = float(e0[0])
    h_vac = float(Omega @ (_block(h, blocks[0]) @ Omega))
    del H0, v0
    for p in sectors:
        ts = time.time()
        idx = blocks[p]
        Hs = _block(H, idx)
        hs = _block(h, idx)
        # spectral bounds of the block (and its lowest eigenvector as the Lanczos start)
        if p == 0:
            e_lo, v_lo = E_vac, Omega
        else:
            e_lo, v_lo = spla.eigsh(Hs, k=1, which="SA", tol=1e-14, ncv=min(40, Hs.shape[0] - 1))
            e_lo = float(e_lo[0])
            v_lo = np.real(v_lo[:, 0])
            v_lo /= np.linalg.norm(v_lo)
        e_hi = float(spla.eigsh(Hs, k=1, which="LA", tol=1e-6, return_eigenvectors=False,
                                ncv=min(40, Hs.shape[0] - 1))[0])
        pad = float(bound_pad) * (e_hi - e_lo)
        op = SmearedOperatorMatvec(Hs, hs, e_lo - pad, e_hi + pad, times, w, f2, eps=cheb_eps)
        A = op.as_linear_operator()
        # Warm start from the block's H ground state, with a fixed-seed random
        # admixture: O_f may carry a symmetry the start vector is an eigenvector of
        # (the spatial reflection when the worldline is the chain's centre, L odd),
        # and Lanczos from a pure eigenvector of it never leaves that sector -- the
        # odd block at L = 5 came out 4.6e-4 high that way (tests).
        rng = np.random.default_rng(int(idx.size) + 7919 * p)
        v_start = v_lo + 0.1 * rng.standard_normal(idx.size)
        v_start /= np.linalg.norm(v_start)
        ev, U = spla.eigsh(A, k=1, which="SA", v0=v_start, tol=float(tol),
                           ncv=(None if ncv is None else int(ncv)), maxiter=maxiter)
        u = np.real(U[:, 0])
        u /= np.linalg.norm(u)
        resid = float(np.linalg.norm(op.matvec(u) - ev[0] * u))
        E_star_by[p] = float(ev[0])
        vec_by[p] = u
        info[p] = {"dim": int(idx.size), "e_lo": e_lo, "e_hi": e_hi, "n_cheb": op.n_terms,
                   "lanczos_matvecs": int(op.count), "sparse_matvecs": int(op.propagator_matvecs),
                   "residual": resid, "time_s": time.time() - ts}
    p_star = min(E_star_by, key=E_star_by.get)
    f2_norm = float(2.0 * np.sum(w * f2))
    E_ref = f2_norm * h_vac
    E_star = E_star_by[p_star]
    fine = None
    if dt_fine is not None:
        ts = time.time()
        idx = blocks[p_star]
        Hs = _block(H, idx)
        hs = _block(h, idx)
        tf, wf, dtf = _time_grid(f, float(dt_fine))
        f2f = np.asarray(f.f(tf), float) ** 2
        pad = float(bound_pad) * (info[p_star]["e_hi"] - info[p_star]["e_lo"])
        opf = SmearedOperatorMatvec(Hs, hs, info[p_star]["e_lo"] - pad, info[p_star]["e_hi"] + pad,
                                    tf, wf, f2f, eps=cheb_eps)
        evf, Uf = spla.eigsh(opf.as_linear_operator(), k=1, which="SA", v0=vec_by[p_star],
                             tol=float(tol), ncv=(None if ncv is None else int(ncv)), maxiter=maxiter)
        uf = np.real(Uf[:, 0])
        uf /= np.linalg.norm(uf)
        f2nf = float(2.0 * np.sum(wf * f2f))
        fine = {"dt": float(dtf), "n_steps": int(tf.size - 1), "E_star": float(evf[0]),
                "E_ref": f2nf * h_vac, "E_min": float(evf[0]) - f2nf * h_vac, "sector": int(p_star),
                "n_cheb": opf.n_terms, "lanczos_matvecs": int(opf.count),
                "residual": float(np.linalg.norm(opf.matvec(uf) - evf[0] * uf)),
                "time_s": time.time() - ts}
    E_free = free_exact_infimum(L, x, f, m, bc)
    mu_H2 = hartree_mass_squared(L, m, lam, bc)
    E_H = free_exact_infimum(L, x, f, math.sqrt(mu_H2), bc)
    psi = None
    if keep_state:
        psi = np.zeros(d ** L)
        psi[blocks[p_star]] = vec_by[p_star]
    meta = {"L": L, "m": m, "lam": lam, "n_max": int(n_max), "omega": float(om), "bc": bc, "site": x,
            "f": f.label, "kernel": "trapezoid", "dt": float(dt_eff), "n_steps": int(times.size - 1),
            "f2_norm": f2_norm, "dim": int(d ** L), "dim_sectors": [int(b.size) for b in blocks],
            "sectors_solved": [int(p) for p in sectors],
            "spectral_bounds": {int(p): [info[p]["e_lo"], info[p]["e_hi"]] for p in info},
            "n_cheb": {int(p): info[p]["n_cheb"] for p in info},
            "matvecs": {int(p): {"lanczos": info[p]["lanczos_matvecs"], "sparse": info[p]["sparse_matvecs"]}
                        for p in info},
            "lanczos_residuals": {int(p): info[p]["residual"] for p in info},
            "time_build_s": t_build, "time_sector_s": {int(p): info[p]["time_s"] for p in info},
            "time_total_s": time.time() - t_start, "tol": float(tol), "cheb_eps": float(cheb_eps),
            "bound_pad": float(bound_pad), "fine": fine}
    return QEILanczosResult(E_min=float(E_star - E_ref), E_star=float(E_star), E_ref=float(E_ref),
                            E_star_by_sector={int(k): float(v) for k, v in E_star_by.items()},
                            sector_star=int(p_star), E_vac=float(E_vac), h_vac=float(h_vac),
                            E_free_exact=float(E_free), E_hartree_exact=float(E_H), mu_H2=float(mu_H2),
                            psi_star=psi, meta=meta)
