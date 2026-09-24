"""Pump-enhanced harvesting: the Layer 5 x Layer 2 composite.

The question (PLAN.md, Layer 5): does Floquet-preparing the field —
modulating it before and during detector coupling — raise the entanglement
two harmonic (oscillator-UDW) detectors can harvest from it?

Protocol (one audited ``vacuum.protocol.run_protocol`` call per point):

    t in [0, t_prep):                 field pumped, detectors idle (lam = 0)
    t in [t_prep, t_prep + t_win):    detectors switched on with the cos^2
                                      window lam(t) = lam_max sin^2(pi u/t_win),
                                      u = t - t_prep; the pump stays on
                                      (``pump_during_coupling=True``) or is
                                      off (field back to K0)
    t = t_prep + t_win:               exit; the drive profile is at zero
                                      (whole periods of a 'sin' drive) and
                                      lam = 0, so the exit Hamiltonian is the
                                      undriven blockdiag(K0, Omega_d^2).

The initial state is the lam = 0 ground state of blockdiag(K0, Omega_d^2)
(``vacuum.core.ground_state_cov``) and is *claimed* as a ground state to the
runner, which runs the passivity audit on it; the generator is
``attach_detectors(K_field(t), sites, gaps, lam(t), coupling='xx')`` (the
Layer 2 nonperturbative stack, Brown-Martin-Martinez-Menicucci-Mann PRD 87,
084062 (2013)) spelled as CF4 static pieces (:func:`vacuum.floquet.floquet_steps`)
so the ledger closes to roundoff and the drive work — pump *and* switching —
is charged per step; the dt-halving gate doubles the substeps per period
until E_N and both detector occupations move by less than ``conv_tol``.
E_N is :func:`vacuum.detectors.harvested_log_negativity` (the mandatory
mp-margin rule, ``flagged`` carried in the row).

The unpumped baseline is the same protocol at depth 0 (or ``drive=None``).
A null enhancement is the informative outcome the plan anticipates (pivot
trigger, Layer 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from scipy.integrate import cumulative_simpson

from vacuum.core import Omega, ground_state_cov, reduce as reduce_cov, thermal_state_cov
from vacuum.protocol import GeneratorStep, run_protocol

from .drives import Drive
from .monodromy import cf4_pieces, propagator
from .runs import floquet_steps
from .spectra import mode_occupations, normal_modes

__all__ = [
    "PumpedHarvest",
    "pumped_harvest",
    "cos2_window",
    "detector_occupation",
    "HarvestSplit",
    "floquet_window_rows",
    "pair_integrals_on_grid",
    "pumped_harvest_split",
]


def _detectors():
    """Late import of the Layer 2 public API (edited concurrently; retried)."""
    import importlib
    import time

    last = None
    for _ in range(3):
        try:
            mod = importlib.import_module("vacuum.detectors.nonperturbative")
            return mod.attach_detectors, mod.harvested_log_negativity
        except Exception as exc:  # pragma: no cover - transient concurrent edit
            last = exc
            time.sleep(0.5)
    raise last


def cos2_window(t_on, t_win):
    """lam(t)/lam_max = sin^2(pi (t - t_on)/t_win) on [t_on, t_on + t_win], else 0."""
    t_on, t_win = float(t_on), float(t_win)

    def chi(t):
        u = float(t) - t_on
        if u <= 0.0 or u >= t_win:
            return 0.0
        return float(np.sin(np.pi * u / t_win) ** 2)

    return chi


def detector_occupation(V, mode, gap):
    """<n> = (Omega <x^2> + <p^2>/Omega - 1)/2 of one mode (mean-zero state)."""
    V = np.asarray(V, dtype=float)
    n = V.shape[0] // 2
    return 0.5 * (float(gap) * V[mode, mode] + V[n + mode, n + mode] / float(gap) - 1.0)


@dataclass
class PumpedHarvest:
    """One point of the pump-enhanced-harvesting map."""

    E_N: float
    flagged: bool
    n_d: np.ndarray
    work_total: float
    work_pump: float
    work_switching: float
    depth: float
    omega_mod: float
    n_prep: int
    n_win: int
    t_prep: float
    t_win: float
    pump_during_coupling: bool
    converged: bool
    n_substeps_per_period: int
    halvings: int
    movement: Optional[float]
    history: List[Dict[str, Any]]
    result: Any  # ProtocolResult (accepted level)
    V_det: np.ndarray
    field_occupations: np.ndarray
    field_omega: np.ndarray
    neg: Any  # NegativityResult
    passivity_passed: Optional[bool]
    ledger_defect: float
    all_audits_passed: bool


def pumped_harvest(
    K0,
    drive: Optional[Drive],
    sites,
    gaps,
    lam_max,
    n_prep,
    n_win,
    *,
    period=None,
    pump_during_coupling=True,
    n_substeps_per_period=16,
    method="cf4",
    conv_tol=1e-9,
    max_halvings=8,
    T_field=0.0,
    strict=True,
    require_convergence=True,
    floor_tol=1e-10,
    dps=50,
):
    """Floquet-prepare the field, then harvest (README API: ``pumped_harvest``).

    Parameters
    ----------
    K0 : (N, N) field coupling matrix (undriven).
    drive : Drive or None
        The pump (``modulated_K(K0, depth, omega_mod, 'sin')`` — a profile
        that vanishes at whole periods keeps the entry/exit Hamiltonians
        undriven).  None = unpumped baseline.
    sites, gaps : detector sites/profiles and gaps (``attach_detectors``).
    lam_max : float — peak x-x coupling.
    n_prep, n_win : int — pump periods before coupling / periods of the
        coupling window (its duration is n_win * period).
    period : float, optional — needed when ``drive`` is None (baseline
        timeline); defaults to ``drive.period``.
    pump_during_coupling : bool — keep the pump on during the window.
    n_substeps_per_period, method, conv_tol, max_halvings : the gate.
    T_field : float — field temperature of the initial state (0 = vacuum;
        the passivity claim is only made at T = 0).

    Returns
    -------
    PumpedHarvest
    """
    attach_detectors, harvested_log_negativity = _detectors()
    K0 = np.asarray(K0, dtype=float)
    N = K0.shape[0]
    gaps = np.asarray(gaps, dtype=float)
    n_det = gaps.size
    if drive is None:
        if period is None:
            raise ValueError("period is required for the unpumped baseline")
        T = float(period)
        depth = 0.0
        omega_mod = 2.0 * np.pi / T
    else:
        T = drive.period if period is None else float(period)
        depth = drive.depth
        omega_mod = drive.omega_mod
    n_prep, n_win = int(n_prep), int(n_win)
    t_prep = n_prep * T
    t_win = n_win * T
    chi = cos2_window(t_prep, t_win)
    lam_max = float(lam_max)

    def K_field(t):
        if drive is None or depth == 0.0:
            return K0
        if t < t_prep or pump_during_coupling:
            return drive(t)
        return K0

    class _Gen:
        """Callable t -> K_tot(t) carrying K0 for floquet_steps' envelope hook."""

        def __init__(self):
            self.K0 = attach_detectors(K0, sites, gaps, 0.0, coupling="xx")

        def __call__(self, t):
            return attach_detectors(K0 if False else K_field(t), sites, gaps,
                                    lam_max * chi(t), coupling="xx")

    gen = _Gen()
    K_tot0 = gen.K0
    if float(T_field) == 0.0:
        V0 = ground_state_cov(K_tot0)
        claim = True
    else:
        Vf = thermal_state_cov(K0, float(T_field))
        Vd = ground_state_cov(K_tot0[N:, N:])
        n_tot = N + n_det
        V0 = np.zeros((2 * n_tot, 2 * n_tot))
        i1 = np.r_[np.arange(N), n_tot + np.arange(N)]
        i2 = np.r_[N + np.arange(n_det), n_tot + N + np.arange(n_det)]
        V0[np.ix_(i1, i1)] = Vf
        V0[np.ix_(i2, i2)] = Vd
        claim = False
    det_modes = list(range(N, N + n_det))
    omega_f, U_f = normal_modes(K0)

    def build(nsub):
        steps: List[Any] = []
        t = 0.0
        for p in range(n_prep):
            steps.extend(floquet_steps(gen, t, T, n_substeps=nsub, method=method,
                                       label=f"prep{p}"))
            t += T
        for q in range(n_win):
            steps.extend(floquet_steps(gen, t, T, n_substeps=nsub, method=method,
                                       label=f"win{q}"))
            t += T
        steps.append(GeneratorStep(K_tot0, 0.0, label="exit"))
        return steps

    def observe(V):
        V_det = reduce_cov(V, det_modes)
        neg = harvested_log_negativity(V_det, floor_tol=floor_tol, dps=dps)
        n_d = np.array([detector_occupation(V, N + d, gaps[d]) for d in range(n_det)])
        return neg, n_d, V_det

    history: List[Dict[str, Any]] = []
    prev = None
    movement = None
    converged = False
    res = None
    nsub = int(n_substeps_per_period)
    for level in range(int(max_halvings) + 1):
        nsub = int(n_substeps_per_period) * 2**level
        res = run_protocol(V0, K_tot0, build(nsub), claim_ground_entry=claim,
                           report_modes=[det_modes], run_causality=False, strict=strict)
        neg, n_d, V_det = observe(res.V)
        obs = {"E_N": neg.E_N, **{f"n_d{d}": float(n_d[d]) for d in range(n_det)}}
        movement = None if prev is None else max(abs(obs[k] - prev[k]) for k in obs)
        history.append({"level": level, "n_substeps_per_period": nsub, "observables": obs,
                        "movement": movement, "work": res.work_total})
        if movement is not None and movement < conv_tol:
            converged = True
            break
        prev = obs
    if not converged and require_convergence:
        raise RuntimeError(
            f"pumped_harvest: E_N/n_d did not converge to {conv_tol:g} within "
            f"{max_halvings} halvings (last movement {movement})"
        )
    work_pump = sum(e.get("work", 0.0) for e in res.step_log if e["label"].startswith("prep"))
    # the zero-duration exit switch carries the telescoped remainder of the
    # last CF4 piece (drive work, vanishing with dt): it belongs to the window
    work_win = sum(e.get("work", 0.0) for e in res.step_log
                   if e["label"].startswith("win") or e["label"] == "exit")
    if n_win == 0:
        work_pump += work_win
        work_win = 0.0
    V_field = reduce_cov(res.V, list(range(N)))
    occ = mode_occupations(V_field, omega=omega_f, U=U_f)
    pas = res.audits.get("passivity_entry")
    return PumpedHarvest(
        E_N=float(neg.E_N), flagged=bool(neg.flagged), n_d=n_d,
        work_total=float(res.work_total), work_pump=float(work_pump),
        work_switching=float(work_win), depth=float(depth), omega_mod=float(omega_mod),
        n_prep=n_prep, n_win=n_win, t_prep=t_prep, t_win=t_win,
        pump_during_coupling=bool(pump_during_coupling), converged=converged,
        n_substeps_per_period=nsub, halvings=level, movement=movement,
        history=history, result=res, V_det=V_det, field_occupations=occ,
        field_omega=omega_f, neg=neg,
        passivity_passed=None if pas is None else bool(pas.passed),
        ledger_defect=float(abs(res.audits["ledger"].details["defect"])),
        all_audits_passed=bool(res.all_passed()),
    )


# --------------------------------------------------------------------------
# The M2.5 vacuum/communication split on the Floquet-prepared field
# --------------------------------------------------------------------------
#
# The nonperturbative engine above has no M_vac / M_comm of its own (Layer 2's
# derivative-coupling map faced the same limitation and took the split from
# the perturbative estimator on the thermal kernel).  Here the estimator of
# ``vacuum.detectors.communication`` (Tjoa & Martin-Martinez, PRD 104, 125005
# (2021), Eqs. (22)-(33): M = M^+ + M^- with M^+ from the anti-commutator
# (state-dependent) and M^- from the commutator (state-independent) half of
# the two-point function) is applied to the kernel the Floquet-prepared field
# actually presents to the detectors during the window.
#
# The kernel.  For the quadratic field with Heisenberg propagator S(t <- t_p)
# (the *driven* propagator of ``cf4_pieces`` when the pump stays on, the
# static exp((t - t_p) Omega H0) when it is off) and covariance V_f at the
# end of the preparation, R(t) = S(t) R(t_p) gives, for t, t' >= t_p,
#
#     W_ij(t, t') = <x_i(t) x_j(t')>
#                 = [S(t) V_f S(t')^T]_ij  +  (i/2) [S(t) Omega S(t')^T]_ij
#                 = (1/2)<{x_i(t), x_j(t')}>  +  (1/2)<[x_i(t), x_j(t')]>,
#
# because [R_a, R_b] = i Omega_ab (hbar = 1, block ordering).  The first
# term is the symmetrized (Hadamard) part, Re W in the stationary case; it
# carries the state (the pumped (k, -k) pairs).  The second is the
# Pauli-Jordan commutator: S(t) Omega S(t')^T = S(t <- t') Omega by
# symplecticity, so it is the retarded propagator between the two events,
# fixed by the canonical commutation relations and the drive alone, zero
# outside the (modulated) lattice light cone up to the leakage the Layer-0
# causality audit measures.  Neither is stationary (the state is not, and the
# drive is not), so the exponential-sum quadratures of the M2.3 engine do not
# apply; the M-integral is instead evaluated on the window's uniform grid,
# where the separable structure W(t, t') = a(t) C b(t')^T (a, b = rows of
# S(t)) turns the time-ordered triangle into a cumulative integral of a
# vector function (composite Simpson, ``scipy.integrate.cumulative_simpson``)
# and the same halving gate as the rest of the module.  At depth 0 the kernel
# is the stationary vacuum Wightman function and the split reproduces
# ``communication_split`` on ``LatticeWightman`` (tests/test_floquet_composite.py).
#
# The matrix elements are PKMM Eqs. (15)-(18) with the two-time kernel:
#
#   M   = -lam_A lam_B int dt int_{t_p}^{t} dt' [chi_A(t) chi_B(t') e^{i(O_A t + O_B t')} W_AB(t, t')
#                                                + chi_B(t) chi_A(t') e^{i(O_B t + O_A t')} W_BA(t, t')],
#   P_A = lam_A^2 int dt int dt' chi_A(t) chi_A(t') e^{-i O_A (t - t')} W_AA(t, t'),
#   C   = lam_A lam_B int dt int dt' chi_A(t) chi_B(t') e^{i O_A t} e^{-i O_B t'} W_AB(t', t),
#
# M_comm = M with W -> (i/2) S Omega S^T, M_vac = M with W -> S V_f S^T (the
# split is linear, so M = M_vac + M_comm identically).  These are the
# second-order matrix elements for *qubit* detectors; for the harmonic
# detectors of the composite the same integrals are the leading-order pair
# amplitude <a_A a_B> (the Dyson expansion is identical: e^{+-i Omega t} on
# sigma^+- or on a^dagger, a), so the fraction |M_comm|/|M| is the standard
# estimator for both, and the perturbative negativity is reported next to the
# nonperturbative E_N only as a regime check (lam_max = 0.4 is not small).


@dataclass
class HarvestSplit:
    """M = M_vac + M_comm on the Floquet-prepared kernel (one composite point).

    ``comm_fraction`` = |M_comm|/|M| and ``vac_fraction`` = |M_vac|/|M| (the
    M2.5 columns; they need not sum to 1), ``estimator`` = N^-/N of TMM21
    Eq. (39) on the perturbative negativities N (full), ``N_vac``, ``N_comm``
    [PKMM Eqs. (66)-(67)], ``E_N_perturbative`` = ln(1 + 2N).  ``M_comm_static``
    is the commutator term evaluated with the *undriven* window propagator —
    the baseline's M_comm — so ``comm_drive_shift`` = |M_comm - M_comm_static|
    isolates what the pump does to the signalling channel itself.
    """

    M: complex
    M_vac: complex
    M_comm: complex
    P_A: float
    P_B: float
    C: complex
    comm_fraction: float
    vac_fraction: float
    N: float
    N_vac: float
    N_comm: float
    estimator: float
    E_N_perturbative: float
    M_comm_static: complex
    comm_drive_shift: float
    depth: float
    omega_mod: float
    n_prep: int
    n_win: int
    t_prep: float
    t_win: float
    pump_during_coupling: bool
    spacelike: bool
    n_grid_per_period: int
    halvings: int
    movement: Optional[float]
    converged: bool
    history: List[Dict[str, Any]]
    field_n_total: float
    meta: Dict[str, Any] = field(default_factory=dict)


def _simpson_weights(n, h):
    """Composite Simpson weights on n + 1 uniform points (n even)."""
    if n % 2:
        raise ValueError("Simpson needs an even number of intervals")
    w = np.ones(n + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * (h / 3.0)


def floquet_window_rows(K0, drive, n_prep, n_win, sites, n_per, *, period=None,
                        pump_during_coupling=True, method="cf4"):
    """Propagator rows of the pumped field on the window grid, and V_f.

    Returns ``(t, X, V_f, S_F)``: ``t`` (n + 1,) the grid t_p + m h with
    h = T/n_per, n = n_win n_per; ``X`` (n + 1, len(sites), 2N) with
    ``X[m, i] = S(t_m <- t_p)[site_i, :]`` (the x-quadrature row); ``V_f``
    the field covariance at t_p = n_prep T (vacuum evolved through n_prep
    periods of the drive on the same grid); ``S_F`` the one-period map at
    this resolution.  The drive's CF4 pieces are built once per period and
    reused (the grid is commensurate with the period); ``drive=None`` or
    depth 0 is the static field (exact exponentials).
    """
    K0 = np.asarray(K0, dtype=float)
    N = K0.shape[0]
    sites = [int(s) for s in sites]
    if drive is None:
        if period is None:
            raise ValueError("period is required for the unpumped baseline")
        T, depth = float(period), 0.0
    else:
        T = drive.period if period is None else float(period)
        depth = drive.depth
    n_per = int(n_per)
    h = T / n_per
    static = propagator(K0, h)
    if depth == 0.0:
        S_F = np.linalg.matrix_power(static, n_per)
        period_props = None
    else:
        period_props = []
        for k in range(n_per):
            if method == "cf4":
                Pk = np.eye(2 * N)
                for Kp, tau in cf4_pieces(drive, k * h, h):
                    Pk = propagator(Kp, tau) @ Pk
            elif method == "midpoint":
                Pk = propagator(drive((k + 0.5) * h), h)
            else:
                raise ValueError("method must be 'cf4' or 'midpoint'")
            period_props.append(Pk)
        S_F = np.eye(2 * N)
        for Pk in period_props:
            S_F = Pk @ S_F
    V = ground_state_cov(K0)
    for _ in range(int(n_prep)):
        V = S_F @ V @ S_F.T
    V_f = 0.5 * (V + V.T)
    n = int(n_win) * n_per
    t = float(n_prep) * T + h * np.arange(n + 1)
    X = np.empty((n + 1, len(sites), 2 * N))
    S = np.eye(2 * N)
    X[0] = S[sites, :]
    for m in range(n):
        if period_props is not None and pump_during_coupling:
            S = period_props[m % n_per] @ S
        else:
            S = static @ S
        X[m + 1] = S[sites, :]
    return t, X, V_f, S_F


def pair_integrals_on_grid(t, XA, XB, C, chi_A, chi_B, gap_A, gap_B, lam_A, lam_B):
    """(M, P_A, P_B, C_AB) of the two-time kernel W_ij(t, t') = X_i(t) C X_j(t')^T.

    Composite Simpson on the uniform grid ``t`` (even number of intervals);
    the time-ordered inner integral of M is the cumulative Simpson integral
    of the vector function chi_B(t') e^{i O_B t'} X_B(t') (and A <-> B).
    ``chi_A``, ``chi_B`` are callables of t (vectorised).
    """
    t = np.asarray(t, dtype=float)
    n = t.size - 1
    h = (t[-1] - t[0]) / n
    w = _simpson_weights(n, h)
    cA = np.asarray(chi_A(t), dtype=float)
    cB = np.asarray(chi_B(t), dtype=float)
    eA = np.exp(1j * float(gap_A) * t)
    eB = np.exp(1j * float(gap_B) * t)
    C = np.asarray(C)
    # time-ordered pair term
    uB = (cB * eB)[:, None] * XB
    uA = (cA * eA)[:, None] * XA
    IB = cumulative_simpson(uB, x=t, axis=0, initial=0.0)
    IA = cumulative_simpson(uA, x=t, axis=0, initial=0.0)
    vA = w * cA * eA
    vB = w * cB * eB
    term1 = np.einsum("m,mi,ij,mj->", vA, XA, C, IB)
    term2 = np.einsum("m,mi,ij,mj->", vB, XB, C, IA)
    M = -float(lam_A) * float(lam_B) * (term1 + term2)
    # local terms P = lam^2 alpha C alpha^dagger, alpha = sum_m w chi e^{-i O t} X(t_m)
    alA = (w * cA * np.conj(eA)) @ XA
    alB = (w * cB * np.conj(eB)) @ XB
    P_A = float(lam_A) ** 2 * (alA @ C @ np.conj(alA))
    P_B = float(lam_B) ** 2 * (alB @ C @ np.conj(alB))
    # cross term [PKMM Eq. (15)/(17)]: C_AB = lam_A lam_B int dt int dt' chi_A(t) e^{i O_A t}
    # chi_B(t') e^{-i O_B t'} <Phi_B(t') Phi_A(t)>  (A's switching at t, B's at t'; the field
    # operator of B stands to the left), i.e. X_B(t') C X_A(t)^T on the two-time kernel.
    bB = (w * cB * np.conj(eB)) @ XB
    aA = (w * cA * eA) @ XA
    C_AB = float(lam_A) * float(lam_B) * (bB @ C @ aA)
    return complex(M), float(np.real(P_A)), float(np.real(P_B)), complex(C_AB)


def pumped_harvest_split(
    K0,
    drive: Optional[Drive],
    sites,
    gaps,
    lam_max,
    n_prep,
    n_win,
    *,
    period=None,
    pump_during_coupling=True,
    n_grid_per_period=32,
    rtol=1e-7,
    max_halvings=6,
    method="cf4",
    require_convergence=True,
):
    """The M2.5 split of the pumped harvest's pair term (see the section note).

    Same arguments and timeline as :func:`pumped_harvest` (two detectors,
    x-x coupling, cos^2 window of n_win periods after n_prep pump periods).
    The grid is halved (``n_grid_per_period`` doubled) until M, M_comm, P_A
    and P_B all move by less than ``rtol`` relative to max(|M|, |P_A|, |P_B|).

    Returns
    -------
    HarvestSplit
    """
    from vacuum.detectors.communication import communication_estimator
    from vacuum.detectors.udw import pair_log_negativity

    K0 = np.asarray(K0, dtype=float)
    N = K0.shape[0]
    sites = [int(s) for s in sites]
    gaps = np.asarray(gaps, dtype=float)
    if len(sites) != 2 or gaps.size != 2:
        raise ValueError("the split is a statement about one detector pair")
    if drive is None:
        if period is None:
            raise ValueError("period is required for the unpumped baseline")
        T, depth, omega_mod = float(period), 0.0, 2.0 * np.pi / float(period)
    else:
        T = drive.period if period is None else float(period)
        depth, omega_mod = drive.depth, drive.omega_mod
    n_prep, n_win = int(n_prep), int(n_win)
    t_prep, t_win = n_prep * T, n_win * T
    lam = float(lam_max)

    def chi(t):
        u = np.asarray(t, dtype=float) - t_prep
        inside = (u > 0.0) & (u < t_win)
        return np.where(inside, np.sin(np.pi * u / t_win) ** 2, 0.0)

    Om = Omega(N)
    C_comm = 0.5j * Om
    omega_f, U_f = normal_modes(K0)

    def level(n_per):
        t, X, V_f, _ = floquet_window_rows(K0, drive, n_prep, n_win, sites, n_per, period=T,
                                           pump_during_coupling=pump_during_coupling, method=method)
        XA, XB = X[:, 0, :], X[:, 1, :]
        C_full = V_f + C_comm
        M, P_A, P_B, C_AB = pair_integrals_on_grid(t, XA, XB, C_full, chi, chi, gaps[0], gaps[1], lam, lam)
        M_vac, _, _, _ = pair_integrals_on_grid(t, XA, XB, V_f.astype(complex), chi, chi, gaps[0], gaps[1], lam, lam)
        M_comm, _, _, _ = pair_integrals_on_grid(t, XA, XB, C_comm, chi, chi, gaps[0], gaps[1], lam, lam)
        # the undriven window propagator's commutator term (the baseline's M_comm)
        if depth > 0.0 and pump_during_coupling:
            _, X0, _, _ = floquet_window_rows(K0, None, n_prep, n_win, sites, n_per, period=T)
            M_comm_static, _, _, _ = pair_integrals_on_grid(t, X0[:, 0, :], X0[:, 1, :], C_comm, chi, chi,
                                                            gaps[0], gaps[1], lam, lam)
        else:
            M_comm_static = M_comm
        return dict(M=M, M_vac=M_vac, M_comm=M_comm, P_A=P_A, P_B=P_B, C=C_AB,
                    M_comm_static=M_comm_static, V_f=V_f)

    history: List[Dict[str, Any]] = []
    prev = None
    movement = None
    converged = False
    cur = None
    n_per = int(n_grid_per_period)
    if n_per % 2:
        n_per += 1
    for lvl in range(int(max_halvings) + 1):
        n_here = n_per * 2**lvl
        cur = level(n_here)
        scale = max(abs(cur["M"]), abs(cur["P_A"]), abs(cur["P_B"]), 1e-300)
        if prev is not None:
            movement = max(abs(cur[k] - prev[k]) for k in ("M", "M_comm", "P_A", "P_B")) / scale
        history.append({"level": lvl, "n_grid_per_period": n_here, "movement": movement,
                        "M": cur["M"], "M_comm": cur["M_comm"], "P_A": cur["P_A"], "P_B": cur["P_B"]})
        if movement is not None and movement < rtol:
            converged = True
            break
        prev = cur
    if not converged and require_convergence:
        raise RuntimeError(
            f"pumped_harvest_split: not converged to {rtol:g} in {max_halvings} halvings "
            f"(last movement {movement})"
        )
    M, M_vac, M_comm = cur["M"], cur["M_vac"], cur["M_comm"]
    est = communication_estimator(cur["P_A"], cur["P_B"], M_vac, M_comm)
    absM = abs(M)
    occ = mode_occupations(cur["V_f"], omega=omega_f, U=U_f)
    return HarvestSplit(
        M=M, M_vac=M_vac, M_comm=M_comm, P_A=cur["P_A"], P_B=cur["P_B"], C=cur["C"],
        comm_fraction=0.0 if absM == 0.0 else abs(M_comm) / absM,
        vac_fraction=0.0 if absM == 0.0 else abs(M_vac) / absM,
        N=est["N"], N_vac=est["N_vac"], N_comm=est["N_comm"], estimator=est["estimator"],
        E_N_perturbative=pair_log_negativity(cur["P_A"], cur["P_B"], M),
        M_comm_static=cur["M_comm_static"], comm_drive_shift=abs(M_comm - cur["M_comm_static"]),
        depth=float(depth), omega_mod=float(omega_mod), n_prep=n_prep, n_win=n_win,
        t_prep=t_prep, t_win=t_win, pump_during_coupling=bool(pump_during_coupling),
        spacelike=bool(t_win < abs(sites[1] - sites[0])),
        n_grid_per_period=n_here, halvings=lvl, movement=movement, converged=converged,
        history=history, field_n_total=float(np.sum(occ)),
        meta={"source": "Tjoa & Martin-Martinez, PRD 104, 125005 (2021), Eqs. (22)-(33) on the "
                        "two-time kernel S(t) (V_f + i Omega/2) S(t')^T", "method": method,
              "additivity_defect": abs(M - (M_vac + M_comm))},
    )
