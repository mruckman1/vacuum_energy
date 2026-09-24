"""Nonperturbative Gaussian detector stack (Layer 2, milestone M2.2).

The exact harvesting engine: detectors are extra harmonic-oscillator modes
appended after the N field modes, coupled through the quadratic form

    K_tot = [[ K,        lam F ],
             [ lam F^T,  Omega_d^2 ]]        (spec M2.2, x-x coupling)

so the whole interacting problem stays inside H = (1/2)(p.p + x^T K_tot x)
and the validated ``vacuum.core`` machinery applies unchanged.  This is the
harmonic-detector (oscillator UDW) model treated nonperturbatively via
Gaussian covariance evolution — the formalism of E. G. Brown,
E. Martin-Martinez, N. C. Menicucci, R. B. Mann, Phys. Rev. D 87, 084062
(2013) ("Detectors for probing relativistic quantum physics beyond
perturbation theory") — with every reported number passing through the
standing audits of ``vacuum.audits``.

Pieces (all pure functions over arrays — no hidden state, no input
mutation; pre-positioning for the Layer 6 JAX mirror):

- :func:`attach_detectors` — build the coupled quadratic form.
  ``coupling='xx'`` returns the (N_tot, N_tot) coupling matrix K_tot above;
  ``coupling='xp'`` is the **derivative coupling**: the detector position
  x_d couples to the field momentum pi = d_t phi (free field),
  H_int = lam(t) x_d F^T p_field — the lattice transcription of the
  d_t phi model of Teixido-Bonfill & Martin-Martinez, PRD 110, 105016
  (2024), arXiv:2406.14637, Eq. (2) (provenance and the perturbative side
  in ``vacuum.detectors.udw_derivative``).  That interaction lives off the
  block diagonal of H_mat = [[diag(K, Omega_d^2), C], [C^T, I]], so the
  'xp' path returns the full (2 N_tot, 2 N_tot) H_mat; the stepper,
  the drive-work telescoping and the ledger all take the general form
  (``vacuum.core.mean_energy`` has a (1/2) Tr(H_mat V) path).
- :func:`initial_state` — lam(0) = 0 initial covariances:
  ``ground_state_cov(blockdiag(K, Omega_d^2))`` at T = 0 (with the
  MANDATORY passivity audit on the claimed ground state, spec M2.2) or
  thermal field block (``thermal_state_cov``) tensor detector ground
  states at T > 0.
- :func:`protocol_evolve` / :func:`protocol_evolve_fixed` — exponential-
  midpoint symplectic stepping (second-order Magnus: S_k =
  ``symplectic_from_quadratic(H(t_k + dt/2), dt)``; Blanes-Casas-Oteo-Ros,
  Phys. Rep. 470, 151 (2009), Sec. 5.4) with the dt-halving convergence
  gate: halve dt until every reported observable moves < ``conv_tol``
  (default 1e-9) and record the dt at which the result converged.
  Gaussian channels interleave between steps in a *Strang* (symmetric)
  splitting (G. Strang, SIAM J. Numer. Anal. 5, 506 (1968)) — half-step
  channel, full midpoint step, half-step channel — so the open dynamics
  converges at second order under the same gate.  Ledger ``dissipate``
  entries are fed from before/after mean energies around every channel
  application (closure automatic, not estimated), and drive work is the
  telescoped discrete sum of <dH/dt> dt (see ``vacuum.protocol``:
  Alicki first-law split, J. Phys. A 12, L103 (1979)), which closes the
  books to roundoff at any dt.  An independent quadrature of
  <dH/dt> = (1/2) Tr(dH_mat/dt V(t)) from the analytic lam(t), Omega_d(t)
  derivatives is returned as ``work_deriv`` when ``dK_of_t`` is supplied
  (trapezoid in t: second-order, cross-checked against Delta E in tests).
- :func:`detector_block` — ``reduce`` onto the detector modes.
- :func:`harvested_log_negativity` — ``core.log_negativity`` semantics
  plus the MANDATORY mp-margin rule: when the partial-transpose minimum
  symplectic eigenvalue sits within 1e-10 of the vacuum floor 1/2, the
  margins nu - 1/2 are recomputed via
  ``vacuum.core.precision.symplectic_margins_mp`` and E_N is assembled
  from the margins (E_N = -log1p(2 m) per negative margin m).  Near-death
  negativity is where the physics lives and where float64 lies; the
  ``flagged`` field is carried into every stored result row.
- :func:`run_harvesting` — the audited end-to-end stack: initial state
  (passivity-checked at T = 0), gated evolution, closing
  :class:`vacuum.audits.EnergyLedger`, negativity with its flag.

Conventions per docs/API.md: hbar = 1, V_vac = I/2, nu >= 1/2, block
quadrature ordering R = (x_1..x_N, p_1..p_N), entropies/negativities in
nats.  E_N = sum_k max(0, -ln(2 nu~_k)) over partial-transpose symplectic
eigenvalues (Vidal-Werner, PRA 65, 032314 (2002)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vacuum.audits import AuditResult, EnergyLedger, passivity_audit
from vacuum.audits.ledger import DEFAULT_CLOSURE_ATOL
from vacuum.core import (
    ground_state_cov,
    partial_transpose,
    reduce as reduce_cov,
    symplectic_eigenvalues,
    symplectic_from_quadratic,
    symplectic_margins_mp,
    thermal_state_cov,
)
from vacuum.protocol import coupling_from_hamiltonian, hamiltonian_matrix

__all__ = [
    "attach_detectors",
    "profile_matrix",
    "detector_coupling_derivative",
    "initial_state",
    "InitialState",
    "protocol_evolve",
    "protocol_evolve_fixed",
    "EvolveResult",
    "FixedRunResult",
    "detector_block",
    "detector_occupations",
    "harvested_log_negativity",
    "NegativityResult",
    "run_harvesting",
    "HarvestResult",
]

VACUUM_NU = 0.5
#: mp-margin rule trigger (spec M2.2): |min nu~ - 1/2| below this escalates
#: the negativity computation to mpmath margins.
FLOOR_TOL = 1e-10


# --------------------------------------------------------------------------
# Coupled quadratic form
# --------------------------------------------------------------------------


def profile_matrix(N, sites_or_profiles):
    """Column matrix F (N, n_det) of detector spatial profiles.

    Each element of `sites_or_profiles` is either an integer lattice site
    (pointlike detector: one-hot column) or a length-N profile array — a
    :class:`vacuum.detectors.Smearing` works directly via its ``__array__``.
    Profiles are used as given (Smearing already normalizes sum F = 1);
    nothing is renormalized here.
    """
    N = int(N)
    cols = []
    for k, sp in enumerate(sites_or_profiles):
        if isinstance(sp, (int, np.integer)):
            j = int(sp)
            if not 0 <= j < N:
                raise ValueError(
                    f"detector {k}: site {j} outside the lattice 0..{N - 1}"
                )
            col = np.zeros(N)
            col[j] = 1.0
        else:
            col = np.asarray(sp, dtype=float)
            if col.shape != (N,):
                raise ValueError(
                    f"detector {k}: profile shape {col.shape} != ({N},)"
                )
        cols.append(col)
    if not cols:
        raise ValueError("need at least one detector site/profile")
    return np.stack(cols, axis=1)


def attach_detectors(K, sites_or_profiles, gaps, lambdas=0.0, coupling="xx"):
    """Append detector oscillators to the field coupling matrix (spec M2.2).

    Builds the coupled quadratic form

        K_tot = [[ K,          lam_d F_d ],
                 [ lam_d F_d^T, diag(Omega_d^2) ]]

    with one column per detector (harmonic-oscillator UDW detectors,
    Brown-Martin-Martinez-Menicucci-Mann, PRD 87, 084062 (2013)); the
    interaction lam_d F_d,i x_i x_d stays inside H = (1/2)(p.p + x^T K_tot x)
    so the entire validated core applies unchanged.

    Parameters
    ----------
    K : (N, N) ndarray
        Field coupling matrix (H_field = (1/2)(p.p + x.K.x)).
    sites_or_profiles : sequence
        One entry per detector: integer site (pointlike) or length-N
        smearing profile F (``vacuum.detectors.Smearing`` accepted).
    gaps : sequence of float
        Detector frequencies Omega_d > 0; the detector diagonal block is
        Omega_d^2.
    lambdas : float or sequence of float
        Coupling strength(s) lam_d; a scalar broadcasts over detectors.
        lam = 0 gives the exactly block-diagonal (uncoupled) form.
    coupling : {'xx', 'xp'}
        'xx': x_field-x_detector coupling, H stays of diag(K_tot, I) form
        and the (N_tot, N_tot) coupling matrix K_tot is returned.
        'xp': derivative coupling — the detector position couples to the
        field *momentum*, H_int = sum_d lam_d x_d sum_i F_d,i p_i (p_i =
        d_t phi_i for the free lattice field; Teixido-Bonfill &
        Martin-Martinez, PRD 110, 105016 (2024), Eq. (2), transcribed in
        ``vacuum.detectors.udw_derivative``).  In H = (1/2) R^T H_mat R with
        R = (x_1..x_N, x_d.., p_1..p_N, p_d..) that term sits in the x-p
        blocks, H_mat[p_i, x_d] = H_mat[x_d, p_i] = lam_d F_d,i, so the
        full (2 N_tot, 2 N_tot) quadratic form is returned:

            H_mat = [[ diag(K, Omega_d^2),  C ],
                     [ C^T,                 I ]],   C[p_i, x_d] = lam_d F_d,i

        with C zero everywhere else (no x_i-p_j or p-p cross terms).  At
        lam = 0 this is exactly ``hamiltonian_matrix(K_tot)`` of the 'xx'
        form, which is why :func:`initial_state` is coupling-independent.

    Returns
    -------
    K_tot : (N + n_det, N + n_det) ndarray, symmetric      [coupling='xx']
    H_mat : (2 (N + n_det), 2 (N + n_det)) ndarray, symmetric [coupling='xp']
    """
    if coupling not in ("xx", "xp"):
        raise ValueError(f"coupling must be 'xx' or 'xp', got {coupling!r}")
    K = np.asarray(K, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got shape {K.shape}")
    N = K.shape[0]
    F = profile_matrix(N, sites_or_profiles)
    n_det = F.shape[1]
    gaps = np.asarray(gaps, dtype=float)
    if gaps.shape != (n_det,):
        raise ValueError(
            f"gaps shape {gaps.shape} != ({n_det},) "
            f"(one gap per detector site/profile)"
        )
    if np.any(gaps <= 0.0):
        raise ValueError(f"detector gaps must be > 0, got {gaps}")
    lam = np.broadcast_to(np.asarray(lambdas, dtype=float), (n_det,))

    n_tot = N + n_det
    K_tot = np.zeros((n_tot, n_tot))
    K_tot[:N, :N] = 0.5 * (K + K.T)
    C = F * lam[None, :]
    K_tot[N:, N:] = np.diag(gaps**2)
    if coupling == "xx":
        K_tot[:N, N:] = C
        K_tot[N:, :N] = C.T
        return K_tot
    # 'xp': x_d p_i coupling in the off-block-diagonal of the full form.
    H = hamiltonian_matrix(K_tot)              # diag(blockdiag(K, Omega^2), I)
    H[n_tot : n_tot + N, N:n_tot] = C          # rows p_i, columns x_d
    H[N:n_tot, n_tot : n_tot + N] = C.T        # rows x_d, columns p_i
    return H


def detector_coupling_derivative(N_field, sites_or_profiles, dlambdas, coupling="xx"):
    """d(K_tot)/dt (or d H_mat/dt) for static gaps: coupling blocks only.

    With the coupled form of :func:`attach_detectors` and time entering
    through lam_d(t) alone, the derivative has dlam_d F_d in the
    field-detector coupling blocks and zeros elsewhere.  `dlambdas` is the
    analytic dlam/dt (scalar or per detector) — e.g.
    ``lam_max * Switching.derivative(t)`` from the M2.1 switching library.
    (A gap schedule Omega_d(t) would add d(Omega_d^2)/dt on the detector
    diagonal; static gaps here.)

    ``coupling='xx'`` returns the (N_tot, N_tot) dK_tot/dt; ``'xp'`` returns
    the (2 N_tot, 2 N_tot) dH_mat/dt with the derivative in the x_d-p_i
    blocks — the same placement as :func:`attach_detectors`.
    """
    if coupling not in ("xx", "xp"):
        raise ValueError(f"coupling must be 'xx' or 'xp', got {coupling!r}")
    F = profile_matrix(int(N_field), sites_or_profiles)
    N, n_det = F.shape
    dlam = np.broadcast_to(np.asarray(dlambdas, dtype=float), (n_det,))
    n_tot = N + n_det
    C = F * dlam[None, :]
    if coupling == "xx":
        dK = np.zeros((n_tot, n_tot))
        dK[:N, N:] = C
        dK[N:, :N] = C.T
        return dK
    dH = np.zeros((2 * n_tot, 2 * n_tot))
    dH[n_tot : n_tot + N, N:n_tot] = C
    dH[N:n_tot, n_tot : n_tot + N] = C.T
    return dH


# --------------------------------------------------------------------------
# Initial states (spec M2.2: lam(0) = 0)
# --------------------------------------------------------------------------


def _direct_sum_cov(V1, V2):
    """Direct sum of two block-ordered covariances (modes of V1 first)."""
    V1 = np.asarray(V1, dtype=float)
    V2 = np.asarray(V2, dtype=float)
    n1 = V1.shape[0] // 2
    n2 = V2.shape[0] // 2
    n = n1 + n2
    idx1 = np.concatenate([np.arange(n1), n + np.arange(n1)])
    idx2 = np.concatenate([n1 + np.arange(n2), n + n1 + np.arange(n2)])
    V = np.zeros((2 * n, 2 * n))
    V[np.ix_(idx1, idx1)] = V1
    V[np.ix_(idx2, idx2)] = V2
    return V


@dataclass
class InitialState:
    """Initial covariance for a detector protocol, with its audit trail.

    Attributes
    ----------
    V : (2 N_tot, 2 N_tot) ndarray
        Initial covariance (lam = 0: field state tensor detector grounds).
    K_tot : (N_tot, N_tot) ndarray
        The uncoupled quadratic form blockdiag(K, Omega_d^2).
    T : float
        Field temperature used.
    passivity : AuditResult or None
        The mandatory passivity audit result for the claimed-T=0 ground
        state (spec M2.2); None for T > 0 (a thermal state is passive but
        not *claimed to be a ground state*, so the ground-state audit does
        not apply).
    """

    V: np.ndarray
    K_tot: np.ndarray
    T: float
    passivity: Optional[AuditResult] = None


def initial_state(
    K,
    sites_or_profiles,
    gaps,
    T=0.0,
    *,
    strict=True,
    passivity_kwargs=None,
):
    """Initial covariance at lam(0) = 0 (spec M2.2), passivity-audited at T=0.

    T = 0: ``V = ground_state_cov(blockdiag(K, Omega_d^2))`` — vacuum
    tensor detector ground states — and the passivity audit
    (Pusz-Woronowicz, Commun. Math. Phys. 58, 273 (1978)) runs on it
    EVERY time: the state is claimed to be a ground state, and the claim
    is made checkable.  There is deliberately no bypass flag.

    T > 0: thermal field block from ``thermal_state_cov(K, T)`` tensor
    detector ground states (the spec's finite-temperature runs thermalize
    the field, not the detectors).

    Returns an :class:`InitialState`; in strict mode a passivity failure
    raises :class:`vacuum.audits.AuditViolation`.
    """
    K_tot0 = attach_detectors(K, sites_or_profiles, gaps, 0.0)
    T = float(T)
    if T == 0.0:
        V0 = ground_state_cov(K_tot0)
        audit = passivity_audit(
            K_tot0, V=V0, strict=strict, **(passivity_kwargs or {})
        )
        return InitialState(V=V0, K_tot=K_tot0, T=0.0, passivity=audit)
    K = np.asarray(K, dtype=float)
    n_det = K_tot0.shape[0] - K.shape[0]
    V_field = thermal_state_cov(K, T)
    V_det = ground_state_cov(K_tot0[K.shape[0]:, K.shape[0]:])
    V0 = _direct_sum_cov(V_field, V_det)
    return InitialState(V=V0, K_tot=K_tot0, T=T, passivity=None)


# --------------------------------------------------------------------------
# Evolution: fixed-step engine + dt-halving convergence gate
# --------------------------------------------------------------------------


def _generator_form(H, n, what):
    """Normalize a generator to a symmetric (2n, 2n) H_mat.

    (n, n) input is an x-x coupling matrix K -> diag(K, I) (the 'xx' path;
    ``vacuum.protocol.hamiltonian_matrix``); (2n, 2n) is a general
    quadratic form (the future 'xp' path rides through unchanged).
    """
    H = np.asarray(H, dtype=float)
    if H.shape == (n, n):
        return hamiltonian_matrix(H)
    if H.shape == (2 * n, 2 * n):
        return 0.5 * (H + H.T)
    raise ValueError(
        f"{what} must return shape ({n}, {n}) (coupling matrix K) or "
        f"({2 * n}, {2 * n}) (quadratic form H_mat), got {H.shape}"
    )


def _derivative_form(D, n, what):
    """Normalize a generator *derivative* to a symmetric (2n, 2n) form.

    An (n, n) dK/dt embeds as diag(dK, 0) — the p-p block of diag(K, I)
    is constant, so its derivative is zero (unlike ``hamiltonian_matrix``).
    """
    D = np.asarray(D, dtype=float)
    if D.shape == (n, n):
        out = np.zeros((2 * n, 2 * n))
        out[:n, :n] = 0.5 * (D + D.T)
        return out
    if D.shape == (2 * n, 2 * n):
        return 0.5 * (D + D.T)
    raise ValueError(
        f"{what} must return shape ({n}, {n}) (dK/dt) or "
        f"({2 * n}, {2 * n}) (dH_mat/dt), got {D.shape}"
    )


@dataclass
class FixedRunResult:
    """One fixed-resolution pass of the midpoint/Strang stepper.

    ``work`` is the telescoped drive work (discrete midpoint sum of
    <dH/dt> dt; exact for the simulated piecewise-constant drive, so the
    ledger closes at any resolution).  ``work_deriv`` is the independent
    trapezoid quadrature of (1/2) Tr(dH_mat/dt V(t)) from the analytic
    schedule derivatives (None unless ``dK_of_t`` was supplied).
    ``dissipated_by_channel[j]`` is the summed before/after mean-energy
    difference (positive = energy out) around every application of
    channel j.
    """

    V: np.ndarray
    work: float
    work_deriv: Optional[float]
    dissipated: float
    dissipated_by_channel: np.ndarray
    n_steps: int
    dt_max: float
    H_start: np.ndarray
    H_end: np.ndarray


def protocol_evolve_fixed(V, K_of_t, ts, splits=1, *, channels=(), dK_of_t=None):
    """Midpoint-rule symplectic evolution over the grid `ts` (no gate).

    Each interval [ts[i], ts[i+1]] is divided into `splits` equal substeps;
    each substep evolves with S = ``symplectic_from_quadratic``(H(t_mid), dt)
    — exactly symplectic, the only physics error being second-order Magnus
    truncation (Blanes-Casas-Oteo-Ros, Phys. Rep. 470, 151 (2009), Sec. 5.4).
    Channels interleave in Strang (symmetric) order — channel(dt/2), step,
    channel(dt/2) — keeping the open dynamics second order (Strang 1968).

    Parameters
    ----------
    V : (2n, 2n) ndarray
        Input covariance (never mutated).
    K_of_t : callable
        t -> (n, n) coupling matrix K_tot (the 'xx' path) or (2n, 2n)
        quadratic form H_mat (the 'xp' path of :func:`attach_detectors`).
    ts : (M,) array_like, strictly increasing
        Segment grid; may be nonuniform (refine ramps, coarsen holds).
    splits : int
        Substeps per grid interval (the halving gate passes 2**level).
    channels : sequence of callable(V, t, dt) -> V
        Gaussian channels applied around every substep with duration dt/2
        each side.  The callable owns its dt-scaling (e.g.
        ``lambda V, t, dt: loss(V, modes, eta=exp(-kappa*dt))``) so the
        Trotter limit converges; energy differences around each
        application feed the dissipation account (ledger ``dissipate``).
    dK_of_t : callable, optional
        t -> dK_tot/dt (or d H_mat/dt); enables the ``work_deriv``
        trapezoid quadrature of <dH/dt>.

    Returns
    -------
    FixedRunResult
    """
    V = np.array(V, dtype=float, copy=True)
    n = V.shape[0] // 2
    if V.shape != (2 * n, 2 * n):
        raise ValueError(f"V must be (2n, 2n), got {V.shape}")
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 1 or ts.size < 2 or np.any(np.diff(ts) <= 0.0):
        raise ValueError("ts must be a strictly increasing grid of >= 2 times")
    splits = int(splits)
    if splits < 1:
        raise ValueError(f"splits must be >= 1, got {splits}")
    channels = tuple(channels)
    n_ch = len(channels)

    H_prev = _generator_form(K_of_t(ts[0]), n, "K_of_t(t0)")
    H_start = H_prev
    work = 0.0
    wd = 0.0 if dK_of_t is not None else None
    diss = np.zeros(n_ch)
    n_steps = 0
    dt_max = 0.0
    S = None
    H_used = None
    dt_used = None

    for i in range(ts.size - 1):
        dt = (ts[i + 1] - ts[i]) / splits
        dt_max = max(dt_max, dt)
        for k in range(splits):
            t_a = ts[i] + k * dt
            t_b = t_a + dt
            H_mid = _generator_form(K_of_t(t_a + 0.5 * dt), n, "K_of_t")
            if dK_of_t is not None:  # trapezoid node at substep start
                g_a = 0.5 * float(
                    np.sum(_derivative_form(dK_of_t(t_a), n, "dK_of_t") * V)
                )
            # switch work: midpoint telescoping of W = int <dH/dt> dt
            work += 0.5 * float(np.sum((H_mid - H_prev) * V))
            for j, ch in enumerate(channels):  # leading Strang half
                E_b = 0.5 * float(np.sum(H_mid * V))
                V = np.asarray(ch(V, t_a, 0.5 * dt), dtype=float)
                diss[j] += E_b - 0.5 * float(np.sum(H_mid * V))
            if S is None or dt != dt_used or not np.array_equal(H_mid, H_used):
                S = symplectic_from_quadratic(H_mid, dt)
                H_used, dt_used = H_mid, dt
            V = S @ V @ S.T
            for j, ch in enumerate(channels):  # trailing Strang half
                E_b = 0.5 * float(np.sum(H_mid * V))
                V = np.asarray(ch(V, t_b, 0.5 * dt), dtype=float)
                diss[j] += E_b - 0.5 * float(np.sum(H_mid * V))
            if dK_of_t is not None:  # trapezoid node at substep end
                g_b = 0.5 * float(
                    np.sum(_derivative_form(dK_of_t(t_b), n, "dK_of_t") * V)
                )
                wd += 0.5 * dt * (g_a + g_b)
            H_prev = H_mid
            n_steps += 1

    H_end = _generator_form(K_of_t(ts[-1]), n, "K_of_t(t_end)")
    work += 0.5 * float(np.sum((H_end - H_prev) * V))
    return FixedRunResult(
        V=V,
        work=work,
        work_deriv=wd,
        dissipated=float(np.sum(diss)),
        dissipated_by_channel=diss,
        n_steps=n_steps,
        dt_max=dt_max,
        H_start=H_start,
        H_end=H_end,
    )


@dataclass
class EvolveResult:
    """Gated evolution result (spec M2.2: converged dt is recorded).

    ``dt_converged`` is the largest substep of the accepted run —
    the dt at which every reported observable moved < ``conv_tol`` under
    the last halving.  ``history`` carries one row per halving level
    (level, n_steps, dt_max, observables, movement): the convergence/
    provenance metadata every Layer 2/3 sweep row must keep.
    """

    V: np.ndarray
    t0: float
    t1: float
    converged: bool
    dt_converged: float
    halvings: int
    n_steps: int
    observables: Dict[str, float]
    movement: Optional[float]
    work: float
    work_deriv: Optional[float]
    dissipated: float
    dissipated_by_channel: np.ndarray
    history: List[Dict[str, Any]] = field(default_factory=list)


def protocol_evolve(
    V,
    K_of_t,
    ts,
    *,
    channels=(),
    observables=None,
    conv_tol=1e-9,
    max_halvings=14,
    dK_of_t=None,
    ledger=None,
    require_convergence=True,
):
    """Evolve V under K_of_t with the mandatory dt-halving gate (spec M2.2).

    Runs :func:`protocol_evolve_fixed` at splits = 1, 2, 4, ... and accepts
    the first level at which EVERY reported observable moved less than
    `conv_tol` relative to the previous level, recording the dt at which
    the result converged.  The accepted (finer) run's telescoped drive
    work and per-channel dissipation feed `ledger` — and only the accepted
    run's, so the books describe exactly the reported evolution.

    Parameters
    ----------
    V, K_of_t, ts, channels, dK_of_t
        As in :func:`protocol_evolve_fixed`.
    observables : dict[str, callable(V) -> float], optional
        The reported observables gating convergence.  Defaults to the exit
        mean energy (1/2) Tr(H(t1) V) — callers with physical reports
        (E_N, occupations, ...) pass them explicitly, as
        :func:`run_harvesting` does.
    conv_tol : float
        Gate tolerance (spec M2.2: 1e-9).
    max_halvings : int
        Maximum halving levels tried (splits up to 2**max_halvings).
    ledger : EnergyLedger, optional
        Fed from the accepted run: entry/exit snapshots under K(t0)/K(t1),
        one ``work`` entry (telescoped drive work), one ``dissipate`` entry
        per channel.  Closure is the caller's mandatory last step.
    require_convergence : bool
        If True (default), failure to converge raises RuntimeError; if
        False the last run returns with ``converged=False`` (its books
        still close — closure tests instrumentation, the gate tests
        physics).

    Returns
    -------
    EvolveResult
    """
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 1 or ts.size < 2 or np.any(np.diff(ts) <= 0.0):
        raise ValueError("ts must be a strictly increasing grid of >= 2 times")
    conv_tol = float(conv_tol)
    if conv_tol <= 0.0:
        raise ValueError(f"conv_tol must be > 0, got {conv_tol}")
    n = np.asarray(V).shape[0] // 2
    if observables is None:
        H_exit = _generator_form(K_of_t(ts[-1]), n, "K_of_t(t_end)")
        observables = {
            "mean_energy_exit": lambda Vv, _H=H_exit: 0.5 * float(np.sum(_H * Vv))
        }
    if not observables:
        raise ValueError(
            "observables must be non-empty: the dt-halving gate needs at "
            "least one reported observable to converge on (spec M2.2)"
        )

    history: List[Dict[str, Any]] = []
    prev_obs = None
    run = None
    obs = None
    movement = None
    converged = False
    level = 0
    for level in range(int(max_halvings) + 1):
        run = protocol_evolve_fixed(
            V, K_of_t, ts, splits=2**level, channels=channels, dK_of_t=dK_of_t
        )
        obs = {name: float(f(run.V)) for name, f in observables.items()}
        movement = (
            None
            if prev_obs is None
            else max(abs(obs[k] - prev_obs[k]) for k in obs)
        )
        history.append(
            {
                "level": level,
                "n_steps": run.n_steps,
                "dt_max": run.dt_max,
                "observables": dict(obs),
                "movement": movement,
            }
        )
        if movement is not None and movement < conv_tol:
            converged = True
            break
        prev_obs = obs
    if not converged and require_convergence:
        raise RuntimeError(
            f"protocol_evolve: observables did not converge to {conv_tol:g} "
            f"within {max_halvings} dt-halvings (last movement "
            f"{'n/a (need >= 2 levels)' if movement is None else f'{movement:.3e}'})"
        )

    if ledger is not None:
        # Boundary snapshots are booked under the boundary Hamiltonians
        # actually simulated: their coupling matrix when of diag(K, I)
        # form, otherwise the full quadratic form (vacuum.core.mean_energy
        # takes either; the general path is (1/2) Tr(H_mat V)).  A
        # derivative-coupled ('xp') run whose switching does not vanish at
        # a boundary lands on the general path — the books close either way.
        K_entry = coupling_from_hamiltonian(run.H_start)
        K_exit = coupling_from_hamiltonian(run.H_end)
        ledger.record(
            "entry",
            np.asarray(V, dtype=float),
            K=run.H_start if K_entry is None else K_entry,
        )
        # n_ops: the drive work telescopes over run.n_steps midpoint
        # accumulations and each channel column aggregates one application per
        # substep -- the closure criterion budgets roundoff for them
        ledger.work("drive", run.work, n_ops=int(run.n_steps))
        for j, dq in enumerate(run.dissipated_by_channel):
            ledger.dissipate(f"channel{j}", float(dq), n_ops=int(run.n_steps))
        ledger.record("exit", run.V, K=run.H_end if K_exit is None else K_exit)

    return EvolveResult(
        V=run.V,
        t0=float(ts[0]),
        t1=float(ts[-1]),
        converged=converged,
        dt_converged=run.dt_max,
        halvings=level,
        n_steps=run.n_steps,
        observables=obs,
        movement=movement,
        work=run.work,
        work_deriv=run.work_deriv,
        dissipated=run.dissipated,
        dissipated_by_channel=run.dissipated_by_channel,
        history=history,
    )


# --------------------------------------------------------------------------
# Detector observables
# --------------------------------------------------------------------------


def detector_block(V_out, N_field):
    """Reduced covariance of the detector modes (spec M2.2: ``reduce``).

    The detectors are the modes appended after the N_field field modes;
    for the standard two-detector stack this is the last two modes.
    """
    V_out = np.asarray(V_out, dtype=float)
    n_tot = V_out.shape[0] // 2
    N_field = int(N_field)
    if not 0 <= N_field < n_tot:
        raise ValueError(
            f"N_field={N_field} leaves no detector modes of {n_tot} total"
        )
    return reduce_cov(V_out, range(N_field, n_tot))


def detector_occupations(V, N_field, gaps):
    """Mean detector occupations <n_d> = <a_d^dag a_d> (mean-zero states).

    For a mode with H_d = (1/2)(p^2 + Omega^2 x^2), a = (Omega x + i p) /
    sqrt(2 Omega), so <n> = (Omega <x^2> + <p^2>/Omega - 1)/2 (hbar = 1;
    Serafini, "Quantum Continuous Variables", 2017, Sec. 3.2: single-mode
    moments).  Exactly 0 on the detector ground state.
    """
    V = np.asarray(V, dtype=float)
    n_tot = V.shape[0] // 2
    N_field = int(N_field)
    gaps = np.asarray(gaps, dtype=float)
    n_det = n_tot - N_field
    if gaps.shape != (n_det,):
        raise ValueError(f"gaps shape {gaps.shape} != ({n_det},)")
    out = np.empty(n_det)
    for d in range(n_det):
        i = N_field + d
        out[d] = 0.5 * (gaps[d] * V[i, i] + V[n_tot + i, n_tot + i] / gaps[d] - 1.0)
    return out


# --------------------------------------------------------------------------
# Harvested negativity with the mandatory mp-margin rule
# --------------------------------------------------------------------------


@dataclass
class NegativityResult:
    """Log-negativity with its precision provenance (spec M2.2).

    ``flagged`` is True when the partial-transpose minimum symplectic
    eigenvalue sat within `floor_tol` of the vacuum floor 1/2 in float64 —
    the mp-margin rule then recomputed E_N from mpmath margins
    (``regime='mpmath'``).  The flag is carried into every stored result
    row; near-floor negativities reported without it are inadmissible.
    """

    E_N: float
    flagged: bool
    regime: str
    min_nu_pt: float
    margin_min: float
    nu_pt: np.ndarray
    floor_tol: float
    dps: int


def harvested_log_negativity(
    V_AB, modes_A=None, modes_B=None, floor_tol=FLOOR_TOL, dps=50
):
    """E_N of the detector block, mp-escalated at the vacuum floor.

    E_N = sum_k max(0, -ln(2 nu~_k)) in nats over the symplectic spectrum
    nu~ of the partial transpose on `modes_B` (Vidal-Werner, PRA 65,
    032314 (2002)).  MANDATORY mp-margin rule (spec M2.2): when
    min(nu~) sits within `floor_tol` (1e-10) of 1/2, the margins
    nu~ - 1/2 are recomputed via
    :func:`vacuum.core.precision.symplectic_margins_mp` and E_N is
    assembled from the margins as sum over negative margins m of
    -log1p(2 m) — the difference nu~ - 1/2 is representable in float64
    far below the spacing of nu~ itself, which is exactly what near-death
    negativity needs.

    Parameters
    ----------
    V_AB : (2 n_AB, 2 n_AB) ndarray
        Detector-block covariance (from :func:`detector_block`).
    modes_A, modes_B : sequences of local mode indices, optional
        Bipartition of the block; defaults to mode 0 vs the rest (the
        two-detector A|B split).
    floor_tol : float
        mp-escalation trigger on |min nu~ - 1/2| (spec: 1e-10).
    dps : int
        mpmath working precision for the escalated rung.

    Returns
    -------
    NegativityResult
    """
    V_AB = np.asarray(V_AB, dtype=float)
    n_AB = V_AB.shape[0] // 2
    if V_AB.shape != (2 * n_AB, 2 * n_AB) or n_AB < 2:
        raise ValueError(
            f"V_AB must be a >= 2-mode block covariance, got {V_AB.shape}"
        )
    if modes_A is None and modes_B is None:
        modes_A = (0,)
        modes_B = tuple(range(1, n_AB))
    elif modes_A is None or modes_B is None:
        raise ValueError("pass both modes_A and modes_B, or neither")
    modes_A = tuple(int(m) for m in modes_A)
    modes_B = tuple(int(m) for m in modes_B)
    if set(modes_A) & set(modes_B):
        raise ValueError("modes_A and modes_B must be disjoint")
    if set(modes_A) | set(modes_B) != set(range(n_AB)):
        raise ValueError("modes_A + modes_B must cover the block's modes")

    V_pt = partial_transpose(V_AB, modes_B)
    nu_pt = symplectic_eigenvalues(V_pt)
    min_nu = float(np.min(nu_pt))
    flagged = abs(min_nu - VACUUM_NU) < float(floor_tol)
    if flagged:
        margins = symplectic_margins_mp(V_pt, dps=int(dps))
        E_N = float(
            np.sum(np.where(margins < 0.0, -np.log1p(2.0 * margins), 0.0))
        )
        return NegativityResult(
            E_N=E_N,
            flagged=True,
            regime="mpmath",
            min_nu_pt=float(VACUUM_NU + np.min(margins)),
            margin_min=float(np.min(margins)),
            nu_pt=VACUUM_NU + margins,
            floor_tol=float(floor_tol),
            dps=int(dps),
        )
    E_N = float(np.sum(np.maximum(0.0, -np.log(2.0 * nu_pt))))
    return NegativityResult(
        E_N=E_N,
        flagged=False,
        regime="float64",
        min_nu_pt=min_nu,
        margin_min=min_nu - VACUUM_NU,
        nu_pt=nu_pt,
        floor_tol=float(floor_tol),
        dps=int(dps),
    )


# --------------------------------------------------------------------------
# The end-to-end audited stack
# --------------------------------------------------------------------------


@dataclass
class HarvestResult:
    """One result row of the nonperturbative harvesting stack.

    Carries everything a Layer 2 sweep must store per row (spec M2.2 and
    the pre-positioning rules): the negativity with its precision flag,
    the dt at which the run converged, the full halving history, and the
    closed ledger.
    """

    E_N: float
    flagged: bool
    neg: NegativityResult
    n_d: np.ndarray
    V: np.ndarray
    V_AB: np.ndarray
    evolve: EvolveResult
    dt_converged: float
    converged: bool
    halvings: int
    work: float
    work_deriv: Optional[float]
    dissipated: float
    ledger: EnergyLedger
    ledger_result: AuditResult
    passivity: Optional[AuditResult]
    K_tot0: np.ndarray
    t0: float
    t1: float


def run_harvesting(
    K,
    sites_or_profiles,
    gaps,
    lambda_of_t,
    ts,
    *,
    dlambda_of_t=None,
    T=0.0,
    channels=(),
    conv_tol=1e-9,
    max_halvings=14,
    ledger_tol=DEFAULT_CLOSURE_ATOL,
    floor_tol=FLOOR_TOL,
    dps=50,
    strict=True,
    passivity_kwargs=None,
    require_convergence=True,
    coupling="xx",
):
    """The nonperturbative harvesting stack, end to end (spec M2.2).

    initial state (passivity-audited at T = 0, every time) -> gated
    midpoint/Strang evolution under K_tot(t) = ``attach_detectors``(K, F,
    gaps, lam(t)) -> detector block -> mp-safe log-negativity -> closing
    :class:`EnergyLedger`.  The result row carries E_N with its precision
    ``flagged`` field, the converged dt, occupations, work/dissipation and
    the ledger's closure verdict.

    Parameters
    ----------
    K, sites_or_profiles, gaps
        As in :func:`attach_detectors`.
    lambda_of_t : callable
        t -> lam (scalar, broadcast over detectors) or (n_det,) array —
        a :class:`vacuum.detectors.Switching` object (times a strength)
        drops in directly.
    ts : array_like
        Evolution grid, as in :func:`protocol_evolve` (nonuniform allowed:
        refine ramps, coarsen holds).
    dlambda_of_t : callable, optional
        Analytic dlam/dt (e.g. ``Switching.derivative``); enables the
        <dH/dt> quadrature cross-check ``work_deriv``.
    T : float
        Field temperature for the initial state.
    channels, conv_tol, max_halvings, require_convergence
        As in :func:`protocol_evolve`.
    ledger_tol : float
        Absolute *floor* of the closure criterion (the criterion scales with
        the booked energies and operation count; see
        :meth:`vacuum.audits.EnergyLedger.closure_tolerance`); strict
        closure raises on failure.
    floor_tol, dps
        mp-margin rule parameters of :func:`harvested_log_negativity`.
    strict : bool
        Strict mode for the passivity audit and ledger closure.
    passivity_kwargs : dict, optional
        Forwarded to the initial-state passivity audit.
    coupling : {'xx', 'xp'}
        Field-detector coupling (see :func:`attach_detectors`): 'xx' is the
        amplitude coupling, 'xp' the derivative coupling x_d F^T p_field.
        The initial state (lam = 0) and every audit are the same for both;
        only the generator handed to the stepper differs.

    Returns
    -------
    HarvestResult
    """
    if coupling not in ("xx", "xp"):
        raise ValueError(f"coupling must be 'xx' or 'xp', got {coupling!r}")
    init = initial_state(
        K,
        sites_or_profiles,
        gaps,
        T,
        strict=strict,
        passivity_kwargs=passivity_kwargs,
    )
    N_field = np.asarray(K).shape[0]
    gaps_arr = np.asarray(gaps, dtype=float)

    def K_of_t(t):
        return attach_detectors(
            K, sites_or_profiles, gaps, lambda_of_t(t), coupling=coupling
        )

    dK_of_t = None
    if dlambda_of_t is not None:

        def dK_of_t(t):
            return detector_coupling_derivative(
                N_field, sites_or_profiles, dlambda_of_t(t), coupling=coupling
            )

    observables: Dict[str, Callable[[np.ndarray], float]] = {
        "E_N": lambda Vv: harvested_log_negativity(
            detector_block(Vv, N_field), floor_tol=floor_tol, dps=dps
        ).E_N
    }
    for d in range(gaps_arr.size):
        observables[f"n_d{d}"] = lambda Vv, _d=d: float(
            detector_occupations(Vv, N_field, gaps_arr)[_d]
        )

    ledger = EnergyLedger(K=init.K_tot, tol=float(ledger_tol))
    ev = protocol_evolve(
        init.V,
        K_of_t,
        ts,
        channels=channels,
        observables=observables,
        conv_tol=conv_tol,
        max_halvings=max_halvings,
        dK_of_t=dK_of_t,
        ledger=ledger,
        require_convergence=require_convergence,
    )
    ledger_result = ledger.close(strict=strict)

    V_AB = detector_block(ev.V, N_field)
    neg = harvested_log_negativity(V_AB, floor_tol=floor_tol, dps=dps)
    n_d = detector_occupations(ev.V, N_field, gaps_arr)

    return HarvestResult(
        E_N=neg.E_N,
        flagged=neg.flagged,
        neg=neg,
        n_d=n_d,
        V=ev.V,
        V_AB=V_AB,
        evolve=ev,
        dt_converged=ev.dt_converged,
        converged=ev.converged,
        halvings=ev.halvings,
        work=ev.work,
        work_deriv=ev.work_deriv,
        dissipated=ev.dissipated,
        ledger=ledger,
        ledger_result=ledger_result,
        passivity=init.passivity,
        K_tot0=init.K_tot,
        t0=ev.t0,
        t1=ev.t1,
    )
