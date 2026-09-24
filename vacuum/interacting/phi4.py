"""Lattice lambda-phi^4 chain as a truncated-boson TeNPy model (Layer 7, L5).

The interacting vacuum of PLAN.md's leap L5:

    H = (1/2) sum_i [ pi_i^2 + m^2 phi_i^2 + (phi_{i+1} - phi_i)^2 ]
        + (lambda / 4!) sum_i phi_i^4,

on a finite open chain with unit lattice spacing, hbar = 1, [phi_i, pi_j] =
i delta_ij. At lambda = 0 this is exactly ``vacuum.core.models.harmonic_chain_K``
(bc='dirichlet' walls phi_{-1} = phi_N = 0; bc='open' free ends), whose
Gaussian ground state ``vacuum.core.ground_state_cov`` is the lambda = 0
anchor for everything in this package. The dimensionless-lattice form and the
lambda/4! normalization are those of Milsted, Haegeman and Osborne, Phys. Rev.
D 88, 085030 (2013) [arXiv:1302.5582], Eq. (3) (their mu_0^2 is our m^2, their
lambda-tilde our lambda); the (1+1)d critical line, in *renormalized* units,
sits at lambda/mu_R^2 ~ 66 in this lambda/4! normalization (MHO 2013 quote
~66 at lambda-tilde = 0.1; the continuum value 10.913(56) of Kadoh et al.,
JHEP 05 (2019) 184 [arXiv:1811.12376], is in the lambda/4 normalization and
maps to 6 x 10.913 = 65.5 here). The transition requires a *negative* bare
mass squared: for m^2 > 0 and lambda > 0 the chain is in the symmetric phase
for every lambda (the tadpole only ever raises the mass), which is the regime
this module targets — see :func:`hartree_coupling_matrix` for the distance
from criticality, reported as lambda/mu_H^2 against 66.

Local basis and its frequency (the n_max truncation)
-----------------------------------------------------
Each site carries the harmonic-oscillator Fock basis |0>, ..., |n_max> of a
reference oscillator of frequency omega,

    phi = (b + b^dag) / sqrt(2 omega),   pi = i sqrt(omega/2) (b^dag - b),

and every local operator entering H or an observable is the *projection*
P O P of the exact infinite-basis operator onto that span (matrix elements of
phi^2, pi^2, phi^4 are evaluated in a basis extended by four levels and then
cut). H_trunc = P H P is therefore the Rayleigh-Ritz projection of the true
Hamiltonian: its ground energy is a rigorous **upper bound** on the true E_0
and is **non-increasing in n_max** (nested subspaces), which is the
convergence law the tests check. Truncating the *operators* b, b^dag first
and multiplying afterwards would lose both properties at the top level.

omega is a free choice of basis, not a physical parameter; the default
'auto' takes, from the free chain, omega* = sqrt(<pi^2>/<phi^2>) at the
middle site, the frequency whose thermal-like local state has the smallest
mean occupation: for a single-mode Gaussian state with diagonal covariance
(<phi^2>, <pi^2>) the mean number in the omega-oscillator basis is
nbar(omega) = (omega <phi^2> + <pi^2>/omega - 1)/2, minimized exactly at
omega* with nbar_min = nu_loc - 1/2 (nu_loc the local symplectic eigenvalue).
Convergence in n_max is then exponential with rate ~ ln(nbar/(1+nbar)).
'hartree' uses the self-consistent Hartree chain instead (heavier local
oscillator at lambda > 0). Both are exposed; the n_max convergence sweep is
what certifies a choice.

Perturbation theory (anchor 2; derivation)
-------------------------------------------
Write H = H_0 + lambda V, V = (1/4!) sum_i phi_i^4, and diagonalize
K = U diag(omega_k^2) U^T so that phi_i = sum_k G_ik (a_k + a_k^dag) with
G_ik = U_ik / sqrt(2 omega_k) and X_ii := <phi_i^2>_0 = sum_k G_ik^2
= (1/2)(K^{-1/2})_ii. Wick's theorem gives phi^4 = :phi^4: + 6 X :phi^2:
+ 3 X^2, hence

    E_1 = <0|V|0> = (1/8) sum_i X_ii^2.

Second order, E_2 = - sum_{n != 0} |<n|V|0>|^2 / (E_n - E_0), receives only
two- and four-particle intermediate states. For a symmetric rank-r tensor T,
|| sum T_{k1..kr} a^dag_{k1} .. a^dag_{kr} |0> ||^2 = r! sum_k T_k^2 (sum over
permutations of the r! Wick pairings), and every ordering of a given index
multiset has the same energy, so

    E_2 = - 2  sum_{kl}   M_kl^2   / (omega_k + omega_l)
          - 24 sum_{klmn} T_klmn^2 / (omega_k + omega_l + omega_m + omega_n),

    M_kl   = (1/4)  sum_i X_ii G_ik G_il        (tadpole-dressed 2-particle),
    T_klmn = (1/24) sum_i G_ik G_il G_im G_in   (4-particle).

Ground energy: E(lambda) = E_0 + lambda E_1 + lambda^2 E_2 + O(lambda^3).

Mass renormalization at first order: the two-particle amplitude of V equals
that of the quadratic perturbation V' = (1/2) sum_i dm_i^2 phi_i^2 with
dm_i^2 = X_ii / 2, i.e. delta m^2 = (lambda/2) <phi^2> (the tadpole /
Hartree self-energy). Since phi_i phi_j connects the vacuum only to
two-particle states, the O(lambda) correction to every equal-time correlator
is exactly that of the free chain with coupling matrix K + lambda diag(X/2),
so <phi_i phi_j>(lambda) = (1/2)[(K + lambda D)^{-1/2}]_ij + O(lambda^2),
<pi_i pi_j>(lambda) = (1/2)[(K + lambda D)^{1/2}]_ij + O(lambda^2),
D = diag(X_ii/2) (:func:`first_order_covariance`). Iterating the tadpole to
self-consistency gives the Hartree chain (:func:`hartree_coupling_matrix`),
which resums the leading mass renormalization but is not exact at O(lambda^2).

TeNPy is imported lazily (:func:`_tenpy`), mirroring ``vacuum.qet.dmrg``;
importing this module never requires the ``.[tensor]`` extra, and everything
that does not need an MPS (:func:`local_boson_ops`, :func:`perturbative_energy`,
:func:`first_order_covariance`, :func:`hartree_coupling_matrix`) is pure numpy.
Functions are pure over their MPS arguments (inputs are copied, never mutated).
"""

from __future__ import annotations

import functools
import math
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from vacuum.core import ground_state_cov, harmonic_chain_K

__all__ = [
    "CRITICAL_RATIO_LAMBDA_OVER_MU2",
    "free_coupling_matrix",
    "local_boson_ops",
    "oscillator_frequency",
    "phi4_model_class",
    "phi4_model",
    "GroundState",
    "phi4_ground_mps",
    "mps_energy",
    "perturbative_energy",
    "first_order_covariance",
    "hartree_coupling_matrix",
    "hartree_mass_squared",
]

#: (1+1)d phi^4 critical ratio lambda / mu_R^2 in the lambda/4! normalization:
#: 6 x 10.913(56) (Kadoh et al., JHEP 05 (2019) 184, lambda/4 normalization,
#: continuum limit), consistent with the ~66 quoted by Milsted-Haegeman-Osborne
#: PRD 88, 085030 (2013). Used only to *report* the distance from criticality.
CRITICAL_RATIO_LAMBDA_OVER_MU2 = 6.0 * 10.913

_EXTRA_LEVELS = 4  # phi^4 connects |n> to |n +- 4>: projection needs 4 spare levels


def _tenpy():
    """Lazy, guarded TeNPy import (the ``.[tensor]`` extra)."""
    try:
        import tenpy.linalg.np_conserved as npc
        from tenpy.algorithms import dmrg, tebd
        from tenpy.models.lattice import Chain
        from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
        from tenpy.networks.mps import MPS
        from tenpy.networks.site import BosonSite
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise ImportError(
            "vacuum.interacting needs TeNPy — install the '.[tensor]' extra "
            "(pip install physics-tenpy); the pure-numpy pieces "
            "(local_boson_ops, perturbative_energy, hartree_coupling_matrix) "
            "work without it"
        ) from exc
    return SimpleNamespace(npc=npc, dmrg=dmrg, tebd=tebd, Chain=Chain,
                           CouplingMPOModel=CouplingMPOModel,
                           NearestNeighborModel=NearestNeighborModel,
                           MPS=MPS, BosonSite=BosonSite)


def _check_bc(bc):
    if bc not in ("dirichlet", "open"):
        raise ValueError(f"bc must be 'dirichlet' or 'open', got {bc!r}")
    return bc


def free_coupling_matrix(N, m, bc="dirichlet"):
    """K of the lambda = 0 chain: H_0 = (1/2)(pi.pi + phi.K.phi).

    'dirichlet' delegates to ``vacuum.core.harmonic_chain_K`` (walls
    phi_{-1} = phi_N = 0, diagonal m^2 + 2 everywhere); 'open' drops the two
    wall bonds (end diagonals m^2 + 1), which is TeNPy's natural open chain.
    """
    _check_bc(bc)
    K = harmonic_chain_K(int(N), float(m), bc="dirichlet")
    if bc == "open" and N >= 1:
        K[0, 0] -= 1.0
        K[N - 1, N - 1] -= 1.0
    return K


def local_boson_ops(n_max, omega, extra=_EXTRA_LEVELS):
    """Projected local operators P O P in the omega-oscillator Fock basis.

    Returns dense real (n_max+1)^2 matrices:

    - 'X'    = phi                     (real symmetric; exact within the span)
    - 'Pim'  = -i pi = sqrt(omega/2)(b^dag - b)   (real antisymmetric)
    - 'Xsq'  = P phi^2 P,  'Psq' = P pi^2 P,  'X4' = P phi^4 P
    - 'XPim' = -i P (phi pi + pi phi)/2 P      (real antisymmetric)
    - 'Num'  = b^dag b

    Built in a basis extended by `extra` (>= 4) levels and cut back, so each
    entry is the exact matrix element of the operator between kept states
    (Rayleigh-Ritz projection; see the module docstring). Pure numpy.
    """
    n_max = int(n_max)
    if n_max < 1:
        raise ValueError(f"n_max must be >= 1, got {n_max}")
    omega = float(omega)
    if omega <= 0.0:
        raise ValueError(f"omega must be > 0, got {omega}")
    if extra < _EXTRA_LEVELS:
        raise ValueError(f"extra must be >= {_EXTRA_LEVELS} for an exact phi^4 projection")
    D = n_max + 1 + int(extra)
    b = np.diag(np.sqrt(np.arange(1, D, dtype=float)), 1)
    bd = b.T
    x = (b + bd) / math.sqrt(2.0 * omega)
    pim = math.sqrt(omega / 2.0) * (bd - b)
    d = n_max + 1
    cut = (slice(0, d), slice(0, d))
    xx = x @ x
    return {
        "X": x[cut].copy(),
        "Pim": pim[cut].copy(),
        "Xsq": xx[cut].copy(),
        "Psq": (-(pim @ pim))[cut].copy(),
        "X4": (xx @ xx)[cut].copy(),
        "XPim": (0.5 * (x @ pim + pim @ x))[cut].copy(),
        "Num": np.diag(np.arange(d, dtype=float)),
    }


def oscillator_frequency(N, m, bc="dirichlet", site=None, K=None):
    """omega* = sqrt(<pi^2>/<phi^2>) of the free (or given-K) chain at `site`.

    The basis frequency minimizing the local mean occupation (module
    docstring). Default site: the middle of the chain. Pass K to use another
    quadratic chain (e.g. the Hartree chain).
    """
    N = int(N)
    if K is None:
        K = free_coupling_matrix(N, m, bc)
    V = ground_state_cov(K)
    c = N // 2 if site is None else int(site)
    return float(math.sqrt(V[N + c, N + c] / V[c, c]))


@functools.lru_cache(maxsize=None)
def phi4_model_class():
    """The TeNPy model class (built lazily so importing never needs TeNPy).

    ``Phi4Chain(model_params)`` with keys L, m, lam, n_max, omega, bc
    ('dirichlet' | 'open'), bc_MPS ('finite'). A ``CouplingMPOModel`` (MPO
    for DMRG) that is also a ``NearestNeighborModel`` (bond Hamiltonians for
    TEBD). Terms: (1/2) Psq + [(1/2) m^2 + c_i] Xsq + (lam/24) X4 on site i,
    with c_i = 1 (interior; the two half-bond diagonals), c_end = 1
    ('dirichlet', wall bond) or 1/2 ('open'); and -X_i X_{i+1} on bonds.
    """
    t = _tenpy()

    class Phi4Chain(t.CouplingMPOModel, t.NearestNeighborModel):
        default_lattice = t.Chain
        force_default_lattice = True

        def init_sites(self, model_params):
            n_max = model_params.get("n_max", 6, int)
            omega = model_params.get("omega", 1.0, float)
            site = t.BosonSite(Nmax=int(n_max), conserve=None)
            for name, mat in local_boson_ops(n_max, omega).items():
                site.add_op(name, mat)
            return site

        def init_terms(self, model_params):
            m = model_params.get("m", 1.0, float)
            lam = model_params.get("lam", 0.0, float)
            bc = _check_bc(model_params.get("bc", "dirichlet", str))
            L = self.lat.N_sites
            c = np.ones(L)
            if bc == "open":
                c[0] = 0.5
                c[L - 1] = 0.5
            self.add_onsite(0.5, 0, "Psq")
            self.add_onsite(0.5 * m * m + c, 0, "Xsq")
            if lam != 0.0:
                self.add_onsite(lam / 24.0, 0, "X4")
            self.add_coupling(-1.0, 0, "X", 0, "X", 1)

    Phi4Chain.__module__ = __name__
    return Phi4Chain


def _resolve_omega(omega, L, m, lam, bc):
    if isinstance(omega, str):
        if omega == "auto":
            return oscillator_frequency(L, m, bc)
        if omega == "hartree":
            return oscillator_frequency(L, m, bc, K=hartree_coupling_matrix(L, m, lam, bc))
        raise ValueError(f"omega must be 'auto', 'hartree' or a positive float, got {omega!r}")
    omega = float(omega)
    if omega <= 0.0:
        raise ValueError(f"omega must be > 0, got {omega}")
    return omega


def phi4_model(L, m, lam, n_max=6, omega="auto", bc="dirichlet"):
    """Instantiate the TeNPy model (finite MPS boundary conditions)."""
    L = int(L)
    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")
    lam = float(lam)
    if lam < 0.0:
        raise ValueError(f"lambda must be >= 0 (H unbounded below otherwise), got {lam}")
    _check_bc(bc)
    omega_val = _resolve_omega(omega, L, m, lam, bc)
    cls = phi4_model_class()
    return cls({"L": L, "m": float(m), "lam": lam, "n_max": int(n_max),
                "omega": omega_val, "bc": bc, "bc_MPS": "finite"})


class GroundState(NamedTuple):
    """DMRG ground state: energy, MPS, the model, and the convergence record."""

    E0: float
    psi: object
    model: object
    info: dict


def mps_energy(psi, model):
    """<psi|H|psi> from the model's MPO (real part; the state is normalized)."""
    return float(np.real(model.H_MPO.expectation_value(psi)))


def phi4_ground_mps(L, m, lam, n_max=6, chi_max=32, omega="auto", bc="dirichlet",
                    max_sweeps=30, min_sweeps=4, max_E_err=1e-12, svd_min=1e-13,
                    mixer=False, psi0=None, lanczos_tight=True):
    """DMRG ground state of the phi^4 chain: ``GroundState(E0, psi, model, info)``.

    Parameters mirror ``vacuum.qet.dmrg.tfim_ground_mps``: chi_max caps the
    bond dimension, (max_sweeps, min_sweeps, max_E_err) control the
    TwoSiteDMRGEngine, svd_min the truncation, psi0 warm-starts (copied,
    never mutated; used for chi-doubling continuations), lanczos_tight
    pushes the site eigenproblem to machine precision. omega: 'auto',
    'hartree' or a float (module docstring). Initial state: every site in
    the local oscillator vacuum |0>.

    info keys: 'sweeps', 'delta_E', 'max_trunc_err', 'chi_max', 'chi'
    (largest bond dimension actually used), 'converged', 'n_max', 'omega',
    'L', 'm', 'lam', 'bc'.
    """
    t = _tenpy()
    model = phi4_model(L, m, lam, n_max=n_max, omega=omega, bc=bc)
    L = int(L)
    if psi0 is not None:
        psi = psi0.copy()
    else:
        psi = t.MPS.from_product_state(model.lat.mps_sites(), ["vac"] * L,
                                       bc="finite", unit_cell_width=L)
    options = {
        "trunc_params": {"chi_max": int(chi_max), "svd_min": float(svd_min)},
        "max_E_err": float(max_E_err),
        "max_S_err": 1e-8,
        "max_sweeps": int(max_sweeps),
        "min_sweeps": int(min_sweeps),
        "mixer": bool(mixer),
    }
    if lanczos_tight:
        options["lanczos_params"] = {"N_max": 60, "P_tol": 1e-16, "E_tol": 1e-16}
    eng = t.dmrg.TwoSiteDMRGEngine(psi, model, options)
    E0, psi = eng.run()

    delta_E = None
    max_trunc = 0.0
    stats = getattr(eng, "sweep_stats", None) or {}
    if stats.get("Delta_E"):
        delta_E = float(abs(stats["Delta_E"][-1]))
    elif len(stats.get("E", [])) >= 2:
        delta_E = float(abs(stats["E"][-1] - stats["E"][-2]))
    if stats.get("max_trunc_err"):
        max_trunc = float(np.max(stats["max_trunc_err"]))
    info = {
        "sweeps": int(getattr(eng, "sweeps", 0)),
        "delta_E": delta_E,
        "max_trunc_err": max_trunc,
        "chi_max": int(chi_max),
        "chi": int(max(psi.chi)) if len(psi.chi) else 1,
        "converged": (delta_E is not None and delta_E <= float(max_E_err)),
        "n_max": int(n_max),
        "omega": float(model.options["omega"]),
        "L": L, "m": float(m), "lam": float(lam), "bc": bc,
    }
    return GroundState(float(E0), psi, model, info)


# --------------------------------------------------------------------------
# Perturbation theory and the Hartree chain (pure numpy)
# --------------------------------------------------------------------------


def _modes(K):
    w2, U = np.linalg.eigh(0.5 * (K + K.T))
    if np.min(w2) <= 0.0:
        raise ValueError(f"K must be positive definite: min eig {np.min(w2):.3e}")
    w = np.sqrt(w2)
    G = U / np.sqrt(2.0 * w)[None, :]  # G_ik = U_ik / sqrt(2 omega_k)
    return w, G


def perturbative_energy(N, m, lam, bc="dirichlet", order=2, K=None):
    """Rayleigh-Schroedinger ground energy to O(lambda^2) (module docstring).

    Returns a dict with 'E0' (zero-point energy sum omega_k / 2), 'E1', 'E2'
    (coefficients of lambda, lambda^2) and 'total' = E0 + lam E1 + lam^2 E2
    (or through first order for order=1). The four-particle sum is O(N^4)
    dense — fine for the N <= 40 chains DMRG handles here.
    """
    if K is None:
        K = free_coupling_matrix(N, m, bc)
    w, G = _modes(K)
    X = np.sum(G * G, axis=1)  # <phi_i^2>_0 = (1/2)(K^{-1/2})_ii
    E0 = 0.5 * float(np.sum(w))
    E1 = 0.125 * float(np.sum(X * X))
    out = {"E0": E0, "E1": E1, "E2": 0.0}
    if order >= 2:
        # two-particle: M_kl = (1/4) sum_i X_i G_ik G_il
        M = 0.25 * (G.T * X[None, :]) @ G
        denom2 = w[:, None] + w[None, :]
        E2 = -2.0 * float(np.sum(M * M / denom2))
        # four-particle: T_klmn = (1/24) sum_i G_ik G_il G_im G_in
        T = np.einsum("ik,il,im,in->klmn", G, G, G, G) / 24.0
        denom4 = (w[:, None, None, None] + w[None, :, None, None]
                  + w[None, None, :, None] + w[None, None, None, :])
        E2 += -24.0 * float(np.sum(T * T / denom4))
        out["E2"] = E2
    elif order != 1:
        raise ValueError(f"order must be 1 or 2, got {order}")
    lam = float(lam)
    out["total"] = E0 + lam * E1 + lam * lam * out["E2"]
    out["lam"] = lam
    return out


def _tadpole_D(K):
    """D = diag(X_ii / 2) with X = <phi^2>_0 of the chain K (delta m^2 = lam X/2)."""
    N = K.shape[0]
    V = ground_state_cov(K)
    return np.diag(0.5 * np.diag(V[:N, :N]))


def first_order_covariance(N, m, lam, bc="dirichlet"):
    """Covariance of the free chain with the O(lambda) tadpole mass, K + lam D.

    Equal to the interacting ground-state covariance through O(lambda)
    (module docstring): <phi phi> = (1/2)(K + lam D)^{-1/2},
    <pi pi> = (1/2)(K + lam D)^{1/2}, D = diag(<phi_i^2>_0 / 2).
    """
    K = free_coupling_matrix(N, m, bc)
    return ground_state_cov(K + float(lam) * _tadpole_D(K))


def hartree_coupling_matrix(N, m, lam, bc="dirichlet", self_consistent=True,
                            tol=1e-13, max_iter=500):
    """K_H = K + lam diag(<phi_i^2>_H / 2), iterated to self-consistency.

    With self_consistent=False this is the first-order (one tadpole) chain
    K + lam D of :func:`first_order_covariance`. The self-consistent Hartree
    chain is the Gaussian variational ground state's coupling matrix — the
    resummed leading mass renormalization, used here for the 'hartree' basis
    frequency, the distance-from-criticality report, and the "mass
    renormalization only" comparison line of the QEI study.
    """
    K = free_coupling_matrix(N, m, bc)
    lam = float(lam)
    if lam == 0.0:
        return K.copy()
    KH = K + lam * _tadpole_D(K)
    if not self_consistent:
        return KH
    for _ in range(int(max_iter)):
        KH_new = K + lam * _tadpole_D(KH)
        if np.max(np.abs(KH_new - KH)) < tol:
            return KH_new
        KH = KH_new
    raise RuntimeError("Hartree iteration did not converge")


def hartree_mass_squared(N, m, lam, bc="dirichlet", site=None, **kw):
    """mu_H^2 = m^2 + lam <phi_site^2>_H / 2 at `site` (default: middle).

    The ratio lam / mu_H^2 against :data:`CRITICAL_RATIO_LAMBDA_OVER_MU2`
    (~65.5) is how far the run sits from the symmetric/broken transition.
    """
    N = int(N)
    KH = hartree_coupling_matrix(N, m, lam, bc, **kw)
    K = free_coupling_matrix(N, m, bc)
    c = N // 2 if site is None else int(site)
    return float(m) ** 2 + float(KH[c, c] - K[c, c])
