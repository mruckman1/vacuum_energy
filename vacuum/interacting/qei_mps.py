"""First interacting quantum-energy-inequality numbers (PLAN.md Layer 7, L5).

The quantity
------------
Along the static worldline of lattice site i, with a smooth sampling
function f (``vacuum.inequalities.qei.FHandle``, convention rho = f^2),

    E_f(psi) = int dt f(t)^2 [ <psi(t)| h_i |psi(t)> - <Omega| h_i |Omega> ],

|psi(t)> = e^{-iHt} |psi>, h_i the PSD local energy density of
:mod:`vacuum.interacting.observables` (interaction term included), and
|Omega> the *interacting* ground state, whose constant energy density is
subtracted — the lattice form of normal ordering with respect to the
interacting vacuum. At lambda = 0 the Heisenberg evolution is the linear
map h_i(t) = S(t)^T h_i S(t), and E_f(psi) is exactly the smeared
normal-ordered energy (1/2) Tr(O_f (V_psi - V_vac)) of
``vacuum.inequalities.qei`` (sampling operator O_f): that identity is the
anchor the tests measure, with the TEBD/Trotter/n_max errors as the
tolerance.

Why this is new: the Gaussian machinery of Layer 3 *solves* the free
infimum (one Williamson problem); no interacting analogue exists. Beyond
the (1+1)d integrable and conformal exceptions (Bostelmann-Cadamuro-Fewster
for the massive Ising model; Fewster-Hollands, Rev. Math. Phys. 17, 577
(2005) for unitary positive-energy 2d CFTs) there is no QEI theorem for
phi^4 and no numerical data. The number produced here is therefore an
**upper bound on the infimum** over a stated variational family, compared
with the free-theory exact infimum at the same f, m and lattice.

The variational family (real, local, time-reversal symmetric)
-------------------------------------------------------------
    |psi(theta)> = [ prod_{bonds b in W} S2_b(s_b) ] [ prod_{j in W} S1_j(r_j) ] |Omega>,

    S1_j(r) = exp( r (b_j^dag^2 - b_j^2) / 2 )          single-site squeezer,
    S2_b(s) = exp( s (b_j^dag b_{j+1}^dag - b_j b_{j+1}) )   two-mode squeezer,

on a window W of 2w+1 sites centred on the worldline (bonds inside W),
theta = (r_{-w..w}, s_{-w..w-1}), all real. In the truncated local basis
these are the exact exponentials of the projected generators (real
orthogonal matrices); on the free vacuum their Gaussian images are the
symplectics phi_j -> e^{r} phi_j, pi_j -> e^{-r} pi_j and
(phi_j, phi_k) -> cosh s (phi_j, phi_k) + sinh s (phi_k, phi_j),
(pi_j, pi_k) -> cosh s (pi_j, pi_k) - sinh s (pi_k, pi_j) (Serafini 2017,
Sec. 5.1), composed last-applied-leftmost: V -> M V M^T,
M = M_n ... M_1. Real states make <h(-t)> = <h(t)>, so only t >= 0 is
evolved (Simpson weights on the TEBD grid, the integrand doubled). The
family contains the free theory's extremal state only approximately (the
Williamson vacuum of O_f is a multimode squeezer with a longer-range
kernel), so even at lambda = 0 the family minimum sits above the exact
infimum; that gap is measured and reported (``family_quality``).

Search strategy (compute-bounded)
---------------------------------
Every interacting evaluation is a TEBD trajectory (cost ~ L d^3 chi^3 per
step); a full multi-parameter search in the interacting theory is out of
budget. Instead: (1) optimize theta over the *free* family (cheap: each
evaluation is one Gaussian smeared energy) at the Hartree mass of the
interacting chain, giving the shape theta*; (2) in the interacting theory
evaluate E_f along the amplitude line s theta* on a short grid, refine once
by the parabola through the best three points, and report the **minimum of
the evaluated energies** — a state that was actually constructed and
measured, hence a genuine upper bound on the infimum. Every number carries
chi, n_max, dt, and the TEBD truncation error, and the reported optimum is
re-evaluated at 2 chi (the chi-doubling gate).

Anomaly protocol: if E_var(lambda) ever falls below the free exact infimum
E_min^free at the same (f, m, lattice) — "the interacting theory beats the
free bound" — the PLAN.md protocol applies: suspect the numerics first
(Trotter step, chi, n_max, light-cone/wall reflections, the reference
subtraction), then the family, and only then the physics.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from vacuum.core import ground_state_cov
from vacuum.inequalities.qei import (
    FHandle,
    qei_bound_2d,
    qei_minimize,
    sampling_operator,
    smeared_energy,
)

from .observables import local_energy_density
from .phi4 import (
    GroundState,
    _tenpy,
    free_coupling_matrix,
    hartree_coupling_matrix,
    hartree_mass_squared,
)

__all__ = [
    "SqueezeFamily",
    "squeeze_family",
    "family_symplectic",
    "free_family_energy",
    "optimize_free_family",
    "apply_squeeze_family",
    "smeared_energy_mps",
    "qei_variational_mps",
    "QEIVariationalResult",
]


class SqueezeFamily(NamedTuple):
    """Window geometry of the variational family (module docstring)."""

    site: int
    sites: tuple      # window sites, ascending
    bonds: tuple      # (j, j+1) pairs inside the window
    n_params: int

    def split(self, theta):
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != self.n_params:
            raise ValueError(f"theta must have {self.n_params} entries, got {theta.size}")
        n_s = len(self.sites)
        return theta[:n_s], theta[n_s:]


def squeeze_family(site, half_width, L):
    """Family on the window site-half_width .. site+half_width (clipped to the chain)."""
    site = int(site)
    lo = max(0, site - int(half_width))
    hi = min(int(L) - 1, site + int(half_width))
    sites = tuple(range(lo, hi + 1))
    bonds = tuple((j, j + 1) for j in range(lo, hi))
    return SqueezeFamily(site=site, sites=sites, bonds=bonds,
                         n_params=len(sites) + len(bonds))


def family_symplectic(fam: SqueezeFamily, theta, N):
    """(2N, 2N) symplectic M of the family's Gaussian image, V -> M V M^T."""
    r, s = fam.split(theta)
    N = int(N)
    M = np.eye(2 * N)
    for j, rj in zip(fam.sites, r):
        Mj = np.eye(2 * N)
        Mj[j, j] = math.exp(rj)
        Mj[N + j, N + j] = math.exp(-rj)
        M = Mj @ M
    for (j, k), sb in zip(fam.bonds, s):
        c, sh = math.cosh(sb), math.sinh(sb)
        Mb = np.eye(2 * N)
        Mb[j, j] = Mb[k, k] = c
        Mb[j, k] = Mb[k, j] = sh
        Mb[N + j, N + j] = Mb[N + k, N + k] = c
        Mb[N + j, N + k] = Mb[N + k, N + j] = -sh
        M = Mb @ M
    return M


def free_family_energy(fam: SqueezeFamily, theta, V0, op):
    """Smeared normal-ordered energy of the family's Gaussian image of V0."""
    N = V0.shape[0] // 2
    M = family_symplectic(fam, theta, N)
    return smeared_energy(M @ V0 @ M.T, op)


def optimize_free_family(fam: SqueezeFamily, K, op, theta0=None, maxiter=400):
    """Minimize the free smeared energy over the family (BFGS, numerical grad).

    K: coupling matrix of the free chain the family acts on (the free chain
    or the Hartree chain); op: the sampling operator of the *free* chain
    the energy is measured with. Returns (theta*, E_var_free, result).
    """
    V0 = ground_state_cov(K)
    if theta0 is None:
        theta0 = np.zeros(fam.n_params)
        theta0[len(fam.sites) // 2] = -0.1  # seed: squeeze pi at the worldline
    res = minimize(lambda th: free_family_energy(fam, th, V0, op), np.asarray(theta0, float),
                   method="BFGS", options={"maxiter": int(maxiter), "gtol": 1e-10})
    return np.asarray(res.x, float), float(res.fun), res


def _squeezer_ops(site_obj, r_single=None, s_bond=None):
    """Truncated-basis squeezers as npc arrays (real orthogonal)."""
    t = _tenpy()
    d = site_obj.dim
    b = np.diag(np.sqrt(np.arange(1, d, dtype=float)), 1)
    bd = b.T
    leg = site_obj.leg
    if r_single is not None:
        G = 0.5 * float(r_single) * (bd @ bd - b @ b)
        U = expm(G)
        return t.npc.Array.from_ndarray(U, [leg, leg.conj()], labels=["p", "p*"])
    G = float(s_bond) * (np.kron(bd, bd) - np.kron(b, b))
    U = expm(G).reshape(d, d, d, d)
    return t.npc.Array.from_ndarray(U, [leg, leg, leg.conj(), leg.conj()],
                                    labels=["p0", "p1", "p0*", "p1*"])


def apply_squeeze_family(psi, fam: SqueezeFamily, theta):
    """Copy of `psi` with U(theta) applied (single-site first, then bonds)."""
    r, s = fam.split(theta)
    out = psi.copy()
    for j, rj in zip(fam.sites, r):
        if rj != 0.0:
            out.apply_local_op(int(j), _squeezer_ops(out.sites[j], r_single=rj), unitary=True)
    for (j, _k), sb in zip(fam.bonds, s):
        if sb != 0.0:
            out.apply_local_op(int(j), _squeezer_ops(out.sites[j], s_bond=sb), unitary=True)
    return out


def _time_grid(f: FHandle, dt):
    if abs(f.t_lo + f.t_hi) > 1e-12 * max(1.0, abs(f.t_hi)):
        raise ValueError("sampling function must be centred at t = 0 (t_lo = -t_hi)")
    n = int(math.ceil(f.t_hi / float(dt)))
    if n % 2:
        n += 1
    n = max(n, 2)
    t = dt * np.arange(n + 1)
    w = np.ones(n + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w *= dt / 3.0  # composite Simpson on [0, n dt]
    return t, w


def _evolve_trace(psi, model, site, f, m, lam, bc, dt, order, chi_max, svd_min):
    """<h_site>(t_k) along the TEBD trajectory of a copy of psi."""
    t = _tenpy()
    times, _ = _time_grid(f, dt)
    work = psi.copy()
    eng = t.tebd.TEBDEngine(work, model, {
        "order": int(order), "dt": float(dt), "N_steps": 1,
        "trunc_params": {"chi_max": int(chi_max), "svd_min": float(svd_min)},
    })
    h = np.empty(times.size)
    h[0] = local_energy_density(work, site, m, lam, bc)
    for k in range(1, times.size):
        eng.run()
        h[k] = local_energy_density(eng.psi, site, m, lam, bc)
    return times, h, float(eng.trunc_err.eps), int(max(eng.psi.chi))


def smeared_energy_mps(psi, model, site, f: FHandle, m, lam, bc="dirichlet", *,
                       dt=0.1, order=4, chi_max=None, svd_min=1e-12,
                       reference="evolved", psi_ref=None, ref_trace=None):
    """E_f(psi) by TEBD real-time evolution (module docstring).

    psi must be a real MPS (time-reversal shortcut); psi_ref is the ground
    state whose energy density is subtracted — 'static' uses its t = 0
    value, 'evolved' its own TEBD trace (cancels the common Trotter drift;
    pass ref_trace to reuse one across many evaluations). Returns a dict:
    'E_f', 'times', 'h', 'h_ref', 'trunc_err', 'chi', 'dt', 'order'.
    """
    if np.iscomplexobj(psi.get_B(0).to_ndarray()):
        raise ValueError("smeared_energy_mps assumes a real MPS (time-reversal symmetric family)")
    if chi_max is None:
        chi_max = int(max(psi.chi)) if len(psi.chi) else 8
    if psi_ref is None and ref_trace is None:
        raise ValueError("give psi_ref (the ground state) or ref_trace")
    times, w = _time_grid(f, dt)
    _, h, trunc, chi = _evolve_trace(psi, model, site, f, m, lam, bc, dt, order, chi_max, svd_min)
    if ref_trace is not None:
        h_ref = np.asarray(ref_trace, float)
        if h_ref.shape != h.shape:
            raise ValueError("ref_trace does not match the time grid")
    elif reference == "static":
        h_ref = np.full_like(h, local_energy_density(psi_ref, site, m, lam, bc))
    elif reference == "evolved":
        _, h_ref, _, _ = _evolve_trace(psi_ref, model, site, f, m, lam, bc, dt, order, chi_max, svd_min)
    else:
        raise ValueError(f"reference must be 'static' or 'evolved', got {reference!r}")
    f2 = np.asarray(f.f(times), float) ** 2
    E = 2.0 * float(np.sum(w * f2 * (h - h_ref)))
    return {"E_f": E, "times": times, "h": h, "h_ref": h_ref, "trunc_err": trunc,
            "chi": chi, "dt": float(dt), "order": int(order)}


class QEIVariationalResult(NamedTuple):
    E_var: float            # min over evaluated family states (upper bound on inf)
    s_star: float           # amplitude of the reported state
    theta_star: np.ndarray  # its parameters
    s_grid: np.ndarray      # amplitudes evaluated
    E_grid: np.ndarray      # their energies
    E_free_exact: float     # free exact infimum, same f/m/lattice (qei_minimize)
    E_free_family: float    # family minimum in the free theory at the bare mass
    E_hartree_exact: float  # free exact infimum at the Hartree mass (mass-only line)
    bound_2d: float         # Flanagan continuum bound for f
    mu_H2: float            # Hartree mass squared
    trunc_err: float        # worst TEBD truncation error among the evaluations
    chi: int
    dt: float
    meta: dict


def qei_variational_mps(gs: GroundState, site, f: FHandle, half_width=1, *,
                        s_grid=(0.5, 0.75, 1.0, 1.25, 1.5), dt=0.1, order=4,
                        chi_max=None, svd_min=1e-12, theta_star=None,
                        shape_mass="hartree", refine=True):
    """The interacting variational QEI number (module docstring, 'Search strategy').

    gs: the DMRG ground state; site: the worldline; f: sampling function;
    half_width: window w. shape_mass: 'hartree' (default) optimizes the
    free family on the Hartree chain of the interacting parameters,
    'bare' on the bare-mass free chain; pass theta_star to skip the shape
    step. Returns :class:`QEIVariationalResult`.
    """
    L, m, lam, bc = gs.info["L"], gs.info["m"], gs.info["lam"], gs.info["bc"]
    site = int(site)
    fam = squeeze_family(site, half_width, L)
    op = sampling_operator(L, site, f, m=m, bc=bc)
    E_free_exact = qei_minimize(op).e_min
    K_free = free_coupling_matrix(L, m, bc)
    _, E_free_family, _ = optimize_free_family(fam, K_free, op)
    KH = hartree_coupling_matrix(L, m, lam, bc)
    mu_H2 = hartree_mass_squared(L, m, lam, bc)
    op_H = sampling_operator(L, site, f, m=math.sqrt(mu_H2), bc=bc)
    E_hartree_exact = qei_minimize(op_H).e_min
    if theta_star is None:
        if shape_mass == "hartree":
            theta_star, _, _ = optimize_free_family(fam, KH, op_H)
        elif shape_mass == "bare":
            theta_star, _, _ = optimize_free_family(fam, K_free, op)
        else:
            raise ValueError("shape_mass must be 'hartree' or 'bare'")
    theta_star = np.asarray(theta_star, float)
    if chi_max is None:
        chi_max = int(gs.info["chi_max"])

    _, h_ref, trunc_ref, _ = _evolve_trace(gs.psi, gs.model, site, f, m, lam, bc,
                                           dt, order, chi_max, svd_min)
    s_vals = []
    E_vals = []
    worst_trunc = trunc_ref
    chi_seen = 0

    def evaluate(s):
        nonlocal worst_trunc, chi_seen
        psi_s = apply_squeeze_family(gs.psi, fam, s * theta_star)
        out = smeared_energy_mps(psi_s, gs.model, site, f, m, lam, bc, dt=dt, order=order,
                                 chi_max=chi_max, svd_min=svd_min, ref_trace=h_ref)
        worst_trunc = max(worst_trunc, out["trunc_err"])
        chi_seen = max(chi_seen, out["chi"])
        s_vals.append(float(s))
        E_vals.append(out["E_f"])
        return out["E_f"]

    for s in s_grid:
        evaluate(float(s))
    if refine and len(s_vals) >= 3:
        order_idx = np.argsort(E_vals)
        best = int(order_idx[0])
        # parabola through the best point and its grid neighbours
        i0 = min(max(best, 1), len(s_vals) - 2)
        xs = np.array(s_vals[i0 - 1:i0 + 2])
        ys = np.array(E_vals[i0 - 1:i0 + 2])
        a, b, _c = np.polyfit(xs, ys, 2)
        if a > 0:
            s_par = -b / (2.0 * a)
            if xs[0] < s_par < xs[-1] and min(abs(s_par - x) for x in s_vals) > 1e-3:
                evaluate(float(s_par))
    k = int(np.argmin(E_vals))
    return QEIVariationalResult(
        E_var=float(E_vals[k]), s_star=float(s_vals[k]), theta_star=theta_star,
        s_grid=np.asarray(s_vals), E_grid=np.asarray(E_vals),
        E_free_exact=float(E_free_exact), E_free_family=float(E_free_family),
        E_hartree_exact=float(E_hartree_exact), bound_2d=float(qei_bound_2d(f)),
        mu_H2=float(mu_H2), trunc_err=float(worst_trunc), chi=int(chi_seen),
        dt=float(dt),
        meta={"site": site, "half_width": int(half_width), "family": fam,
              "shape_mass": shape_mass, "order": int(order), "L": L, "m": m,
              "lam": lam, "bc": bc, "n_max": gs.info["n_max"], "chi_max": int(chi_max),
              "f": f.label},
    )
