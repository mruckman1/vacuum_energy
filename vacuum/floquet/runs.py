"""Audited Floquet runs: ledgers, loss, pulse trains, amplification bounds.

Two code paths, one accounting convention:

* **Fast path** (:func:`floquet_run`, :func:`lossy_floquet`): the converged
  one-period map S_F of :func:`vacuum.floquet.floquet_map` is applied n
  times, V -> S_F V S_F^T, with an optional Gaussian loss channel
  (``vacuum.core.channels.loss``) interleaved once per period.  Because
  every substep inside S_F is an exact exponential of a *piecewise-constant*
  quadratic Hamiltonian, the drive work of a period is exactly the mean-
  energy change across it under the Hamiltonian in force at the period
  boundary (``vacuum.protocol.midpoint_evolve`` telescoping: each
  exponential conserves its own H(t_mid), so W = sum of switch works =
  Delta E).  The :class:`vacuum.audits.EnergyLedger` therefore charges
  per period ``work = E(after unitary) - E(before)`` and
  ``dissipate = E(before channel) - E(after channel)`` — the same entries
  ``run_protocol`` would write, without re-integrating — plus the sudden
  switch-on/off work (1/2) Tr((H_drive(0) - H_K0) V) at the boundaries when
  the profile does not start at K0.  Closure to 1e-9 is mandatory and
  ``tests/test_floquet_ledger.py`` pins the fast path against
  ``run_protocol`` on the same drive.

* **Protocol path** (:func:`floquet_steps`, :func:`pulse_train`,
  :func:`run_floquet_protocol`): the drive is spelled as
  ``vacuum.protocol`` steps — static ``GeneratorStep`` s for
  piecewise-constant profiles or CF4 pieces (:func:`cf4_pieces`), a
  callable ``GeneratorStep`` for the midpoint rule, ``ChannelStep`` s for
  loss — and run through :func:`vacuum.protocol.run_protocol`, so the
  standing audits (passivity of the undriven vacuum, precision, ledger
  closure) are structural.  Pulse trains with ramped envelopes only exist
  on this path.

Amplification bounds
--------------------
:func:`amplification_bound_scan` maps the numerically converged maximum
gain per period, G(depth, omega_mod, eta) = eta * max|eig S_F|^2 (the
asymptotic photon-number multiplier per period of a lossy drive: the
Floquet multiplier squared for the covariance, times the transmissivity),
over grids of depth, frequency and loss, refining omega_mod to the tongue
centre by golden-section search.  The loss threshold is G = 1:
eta_th = 1/max|eig S_F|^2 (:func:`loss_threshold`).  The comparison
values are the exact bang-bang (two-level) optimum
:func:`bang_bang_bound` — for omega^2(t) switching between
omega^2 (1 -/+ depth) the largest possible multiplier per cycle is
lambda = sqrt((1 + depth)/(1 - depth)), reached with quarter-period
dwell times pi/(2 omega_1), pi/(2 omega_2) (Meissner equation; the
trace 2 cos(omega_1 tau_1) cos(omega_2 tau_2) - (omega_1/omega_2 +
omega_2/omega_1) sin sin is maximised at both sines = +-1) — and the
small-depth Mathieu growth rate depth * omega / 4 at the principal
resonance (Landau-Lifshitz, Mechanics, Sec. 27, Eq. 27.10 with
h = depth).

Platform mapping of the loss axis (:func:`loss_per_period`): a mode of
frequency omega with quality factor Q loses energy at kappa = omega/Q, so
over a drive period T = 2 pi / omega_mod its transmissivity is
eta = exp(-2 pi omega / (omega_mod Q)); at the principal resonance
omega_mod = 2 omega this is exp(-pi/Q).

Pure-functional over arrays (the ledger is the designated accumulator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vacuum.audits import AuditResult, EnergyLedger, passivity_audit
from vacuum.audits.ledger import DEFAULT_CLOSURE_ATOL
from vacuum.core import loss as loss_channel
from vacuum.core import mean_energy
from vacuum.protocol import ChannelStep, GeneratorStep, run_protocol

from .drives import Drive, modulated_K
from .monodromy import (
    FloquetMap,
    cf4_pieces,
    floquet_map,
    mode_monodromy,
    multiplier,
    propagator,
)
from .spectra import mode_occupations, normal_modes

__all__ = [
    "FloquetRun",
    "floquet_run",
    "lossy_floquet",
    "matched_bath_cov",
    "field_loss",
    "floquet_steps",
    "run_floquet_protocol",
    "PulseTrainResult",
    "pulse_train",
    "envelope_cos2",
    "loss_per_period",
    "quality_factor_from_eta",
    "loss_threshold",
    "bang_bang_bound",
    "mathieu_small_depth_rate",
    "AmplificationScan",
    "amplification_bound_scan",
    "max_gain_over_frequency",
]


# --------------------------------------------------------------------------
# Fast path with the ledger
# --------------------------------------------------------------------------


@dataclass
class FloquetRun:
    """Result of :func:`floquet_run`.

    ``energy[n]`` is <H_K0> after n periods (under the undriven K0),
    ``work[n]`` / ``dissipated[n]`` the per-period ledger entries,
    ``occupations[n]`` the normal-mode <n_k>.  ``ledger_result`` is the
    mandatory closure audit; ``passivity`` the audit of the undriven vacuum
    when the run started from one.
    """

    V: np.ndarray
    n_periods: int
    energy: np.ndarray
    work: np.ndarray
    dissipated: np.ndarray
    occupations: np.ndarray
    work_total: float
    dissipated_total: float
    switch_on_work: float
    switch_off_work: float
    ledger: EnergyLedger
    ledger_result: AuditResult
    passivity: Optional[AuditResult]
    fmap: FloquetMap
    eta: float
    nbar: float
    trajectory: Optional[List[np.ndarray]] = None


def floquet_run(
    V0,
    drive: Drive,
    n_periods,
    *,
    fmap: Optional[FloquetMap] = None,
    eta=1.0,
    nbar=0.0,
    loss_modes=None,
    bath="matched",
    ledger_tol=DEFAULT_CLOSURE_ATOL,
    claim_ground_entry=True,
    strict=True,
    passivity_kwargs=None,
    keep_states=False,
    n_steps=64,
    method="cf4",
    conv_tol=1e-10,
    max_halvings=12,
):
    """Apply the drive for n periods with the standing audits (fast path).

    Parameters
    ----------
    V0 : (2N, 2N) covariance
        Initial state under the undriven K0 (the K0 vacuum by default —
        ``claim_ground_entry`` then runs the passivity audit on it).
    drive : Drive
        The periodic drive; its K0 is the snapshot Hamiltonian.
    n_periods : int
    fmap : FloquetMap, optional
        Pre-computed converged monodromy (else computed here with the gate
        parameters ``n_steps``, ``method``, ``conv_tol``, ``max_halvings``).
    eta, nbar : float
        Loss transmissivity per period and environment occupation; eta = 1
        is lossless.  ``bath='matched'`` (default) uses the K0-matched
        multimode bath of :func:`matched_bath_cov` on all modes;
        ``bath='site'`` the bare ``vacuum.core.channels.loss`` on
        ``loss_modes`` (default all sites).
    ledger_tol : float
        Absolute *floor* of the closure criterion (which scales with the
        booked energies and operation count; see
        :meth:`vacuum.audits.EnergyLedger.closure_tolerance`).
    claim_ground_entry : bool
        Run the passivity audit (Pusz-Woronowicz) on V0 under K0.

    Returns
    -------
    FloquetRun
    """
    V = np.array(V0, dtype=float, copy=True)
    N = drive.N
    if V.shape != (2 * N, 2 * N):
        raise ValueError(f"V0 shape {V.shape} incompatible with N={N}")
    n_periods = int(n_periods)
    if n_periods < 0:
        raise ValueError("n_periods must be >= 0")
    eta = float(eta)
    if fmap is None:
        fmap = floquet_map(drive, n_steps=n_steps, method=method,
                           conv_tol=conv_tol, max_halvings=max_halvings)
    S = fmap.S
    K0 = drive.K0
    modes = list(range(N)) if loss_modes is None else list(loss_modes)
    omega, U = normal_modes(K0)
    if bath == "matched":
        V_env = matched_bath_cov(K0, nbar)

        def channel(Vv):
            return field_loss(Vv, eta, V_env)
    elif bath == "site":
        def channel(Vv):
            return loss_channel(Vv, modes, eta, nbar)
    else:
        raise ValueError("bath must be 'matched' or 'site'")

    passivity = None
    if claim_ground_entry:
        passivity = passivity_audit(K0, V=V, strict=strict, **(passivity_kwargs or {}))

    ledger = EnergyLedger(K=K0, tol=float(ledger_tol))
    ledger.record("entry", V, K=K0)
    # the drive Hamiltonian in force at the period boundaries
    K_b = drive(fmap.t0)
    E_K0 = mean_energy(V, K0)
    E_b = mean_energy(V, K_b)
    w_on = E_b - E_K0  # (1/2) Tr((H_drive - H_K0) V): sudden switch-on work
    ledger.work("switch_on", w_on)

    energy = np.empty(n_periods + 1)
    work = np.zeros(n_periods)
    diss = np.zeros(n_periods)
    occ = np.empty((n_periods + 1, N))
    energy[0] = E_K0
    occ[0] = mode_occupations(V, omega=omega, U=U)
    traj = [V.copy()] if keep_states else None
    for n in range(n_periods):
        E0 = mean_energy(V, K_b)
        V = S @ V @ S.T
        E1 = mean_energy(V, K_b)
        work[n] = E1 - E0
        ledger.work(f"period{n}", work[n])
        if eta < 1.0:
            V_new = channel(V)
            E2 = mean_energy(V_new, K_b)
            diss[n] = E1 - E2
            ledger.dissipate(f"loss{n}", diss[n])
            V = V_new
        energy[n + 1] = mean_energy(V, K0)
        occ[n + 1] = mode_occupations(V, omega=omega, U=U)
        if keep_states:
            traj.append(V.copy())
    w_off = mean_energy(V, K0) - mean_energy(V, K_b)
    ledger.work("switch_off", w_off)
    ledger.record("exit", V, K=K0)
    result = ledger.close(strict=strict)
    return FloquetRun(
        V=V, n_periods=n_periods, energy=energy, work=work, dissipated=diss,
        occupations=occ, work_total=float(np.sum(work) + w_on + w_off),
        dissipated_total=float(np.sum(diss)), switch_on_work=w_on,
        switch_off_work=w_off, ledger=ledger, ledger_result=result,
        passivity=passivity, fmap=fmap, eta=eta, nbar=float(nbar), trajectory=traj,
    )


def matched_bath_cov(K0, nbar=0.0):
    """Environment covariance of a bath matched to the field's normal modes.

    Every normal mode k of K0 sits at occupation nbar: V_env = U diag((nbar +
    1/2)/omega_k) U^T (+) U diag((nbar + 1/2) omega_k) U^T — the K0 vacuum at
    nbar = 0 (``vacuum.core.ground_state_cov``).  This is the bath a lattice
    *field* loses photons into: the loss channel V -> eta V + (1 - eta) V_env
    is ``vacuum.core.channels.loss`` written in each mode's dimensionless
    quadratures (the frequency-rescaling rule of
    ``vacuum.detectors.imperfections``), so it is completely positive
    (Holevo-Werner: (1 - eta)(V_env + i Omega/2) >= 0) and leaves the
    undriven vacuum *exactly* invariant.  The bare site-basis channel
    relaxes toward I/2 instead — the vacuum of unit-frequency site
    oscillators, not of the chain — and heats the K0 vacuum; it is kept
    only as ``bath='site'``.
    """
    omega, U = normal_modes(K0)
    N = omega.size
    V = np.zeros((2 * N, 2 * N))
    V[:N, :N] = (U * ((float(nbar) + 0.5) / omega)) @ U.T
    V[N:, N:] = (U * ((float(nbar) + 0.5) * omega)) @ U.T
    return V


def field_loss(V, eta, V_env):
    """Matched-bath loss: V -> eta V + (1 - eta) V_env (all modes)."""
    eta = float(eta)
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta}")
    return eta * np.asarray(V, dtype=float) + (1.0 - eta) * np.asarray(V_env, dtype=float)


def lossy_floquet(V0, S_F, eta, nbar, n_periods, modes=None, *, K0=None, bath="matched"):
    """V after n periods of S_F with loss interleaved (README API).

    ``bath='matched'`` (default; needs ``K0``): the K0-matched multimode
    bath of :func:`matched_bath_cov` at occupation nbar, on all modes.
    ``bath='site'``: the bare ``vacuum.core.channels.loss`` on the listed
    site modes (relaxes toward I/2; see :func:`matched_bath_cov`).  Use
    :func:`floquet_run` for the audited version with the ledger.
    """
    V = np.array(V0, dtype=float, copy=True)
    S = np.asarray(S_F, dtype=float)
    N = V.shape[0] // 2
    if bath == "matched":
        if K0 is None:
            raise ValueError("bath='matched' needs K0")
        V_env = matched_bath_cov(K0, nbar)
        for _ in range(int(n_periods)):
            V = S @ V @ S.T
            if eta < 1.0:
                V = field_loss(V, eta, V_env)
        return V
    if bath != "site":
        raise ValueError("bath must be 'matched' or 'site'")
    modes = list(range(N)) if modes is None else list(modes)
    for _ in range(int(n_periods)):
        V = S @ V @ S.T
        if eta < 1.0 or nbar > 0.0:
            V = loss_channel(V, modes, eta, nbar)
    return V


# --------------------------------------------------------------------------
# Protocol path (vacuum.protocol steps)
# --------------------------------------------------------------------------


def floquet_steps(
    K_of_t,
    t_start,
    duration,
    *,
    n_substeps,
    method="cf4",
    label="floquet",
    envelope=None,
    K_rest=None,
):
    """Spell a driven interval as ``vacuum.protocol`` steps.

    ``method='cf4'`` emits 2 static ``GeneratorStep`` s per substep
    (:func:`vacuum.floquet.cf4_pieces`; fourth order) — the protocol
    literally simulated is piecewise constant, so the runner's ledger
    closes to roundoff and the drive work converges with the state;
    ``method='midpoint'`` emits one callable ``GeneratorStep`` (the
    runner's own second-order rule).  ``envelope(t)`` in [0, 1] scales the
    modulation *about K_rest* (default ``K_of_t.K0``):
    K_env(t) = K_rest + envelope(t) (K_of_t(t) - K_rest), the ramped
    pulses of :func:`pulse_train`.
    """
    n_substeps = int(n_substeps)
    if n_substeps < 1:
        raise ValueError("n_substeps must be >= 1")
    if envelope is not None:
        K_rest = np.asarray(K_of_t.K0 if K_rest is None and isinstance(K_of_t, Drive)
                            else K_rest, dtype=float)

        def K_env(t, _K=K_of_t, _e=envelope, _R=K_rest):
            return _R + float(_e(t)) * (np.asarray(_K(t), dtype=float) - _R)
    else:
        K_env = K_of_t
    if method == "midpoint":
        return [GeneratorStep(K_env, float(duration), n_substeps=n_substeps, label=label)]
    if method != "cf4":
        raise ValueError("method must be 'cf4' or 'midpoint'")
    dt = float(duration) / n_substeps
    steps = []
    for k in range(n_substeps):
        ta = float(t_start) + k * dt
        for p, (Kp, tau) in enumerate(cf4_pieces(K_env, ta, dt)):
            steps.append(GeneratorStep(Kp, tau, label=f"{label}[{k}.{p}]"))
    return steps


def run_floquet_protocol(
    V0,
    drive: Drive,
    n_periods,
    *,
    n_substeps_per_period=16,
    method="cf4",
    eta=1.0,
    nbar=0.0,
    loss_modes=None,
    bath="matched",
    claim_ground_entry=True,
    report_modes=(),
    ledger_tol=DEFAULT_CLOSURE_ATOL,
    strict=True,
    **runner_kwargs,
):
    """n periods of `drive` through :func:`vacuum.protocol.run_protocol`.

    Piecewise-constant profiles are emitted as exact static steps (one per
    piece per period); smooth profiles as CF4 pieces or a midpoint callable.
    Loss is a ``ChannelStep`` after every period (matched bath by default,
    as in :func:`floquet_run`).  ``report_modes=()`` skips the exit
    precision audit (the caller audits the blocks it reports); pass
    explicit blocks to audit them.
    """
    N = drive.N
    T = drive.period
    steps: List[Any] = []
    modes = list(range(N)) if loss_modes is None else list(loss_modes)
    if bath == "matched":
        V_env = matched_bath_cov(drive.K0, nbar)
        loss_step = (field_loss, (float(eta), V_env))
    elif bath == "site":
        loss_step = (loss_channel, (modes, float(eta), float(nbar)))
    else:
        raise ValueError("bath must be 'matched' or 'site'")
    t = 0.0
    for n in range(int(n_periods)):
        if drive.piecewise_constant:
            for value, frac in drive.pieces:
                tau = frac * T
                steps.append(GeneratorStep(drive(t + 0.5 * tau), tau, label=f"period{n}"))
                t += tau
        else:
            steps.extend(floquet_steps(drive, t, T, n_substeps=n_substeps_per_period,
                                       method=method, label=f"period{n}"))
            t += T
        if eta < 1.0:
            # book the channel under the drive Hamiltonian at the period
            # boundary (what the fast path does), not under the last CF4 piece
            steps.append(GeneratorStep(drive(t), 0.0, label=f"boundary{n}"))
            steps.append(ChannelStep(loss_step[0], loss_step[1], label=f"loss{n}"))
    steps.append(GeneratorStep(drive.K0, 0.0, label="switch_off"))
    return run_protocol(
        V0, drive.K0, steps, claim_ground_entry=claim_ground_entry,
        report_modes=list(report_modes), run_causality=False,
        ledger_tol=ledger_tol, strict=strict, **runner_kwargs,
    )


def envelope_cos2(t_on, t_ramp, t_flat):
    """cos^2-ramped pulse envelope: 0 -> 1 over t_ramp, flat t_flat, 1 -> 0."""
    t_on, t_ramp, t_flat = float(t_on), float(t_ramp), float(t_flat)

    def env(t):
        u = float(t) - t_on
        if u <= 0.0 or u >= 2.0 * t_ramp + t_flat:
            return 0.0
        if u < t_ramp:
            return np.sin(0.5 * np.pi * u / t_ramp) ** 2
        if u <= t_ramp + t_flat:
            return 1.0
        return np.cos(0.5 * np.pi * (u - t_ramp - t_flat) / t_ramp) ** 2

    return env


@dataclass
class PulseTrainResult:
    """Result of :func:`pulse_train` (protocol path, gated)."""

    V: np.ndarray
    result: Any  # ProtocolResult of the accepted level
    converged: bool
    n_substeps_per_period: int
    halvings: int
    movement: Optional[float]
    observables: Dict[str, float]
    history: List[Dict[str, Any]]
    t_final: float


def pulse_train(
    V0,
    K_of_t,
    n_pulses,
    gap,
    *,
    n_periods_flat=4,
    n_periods_ramp=2,
    n_substeps_per_period=16,
    method="cf4",
    observables=None,
    conv_tol=1e-9,
    max_halvings=8,
    eta=1.0,
    nbar=0.0,
    loss_modes=None,
    bath="matched",
    claim_ground_entry=True,
    report_modes=(),
    strict=True,
    require_convergence=True,
):
    """Finite train of ramped pulses through ``run_protocol`` (README API).

    Each pulse: cos^2 ramp-up over ``n_periods_ramp`` periods, flat top
    ``n_periods_flat`` periods, cos^2 ramp-down; pulses are separated by
    free evolution under K0 for ``gap`` time units.  Loss (if eta < 1) is
    applied once per period on ``loss_modes``.  The dt-halving gate doubles
    ``n_substeps_per_period`` until every observable (default: the mean
    energy under K0 and the total normal-mode occupation) moves by less
    than ``conv_tol``.
    """
    drive = K_of_t
    if not isinstance(drive, Drive):
        raise TypeError("pulse_train needs a Drive (it carries period and K0)")
    K0 = drive.K0
    N = drive.N
    T = drive.period
    t_pulse = (2 * n_periods_ramp + n_periods_flat) * T
    modes = list(range(N)) if loss_modes is None else list(loss_modes)
    if bath == "matched":
        loss_step = (field_loss, (float(eta), matched_bath_cov(K0, nbar)))
    elif bath == "site":
        loss_step = (loss_channel, (modes, float(eta), float(nbar)))
    else:
        raise ValueError("bath must be 'matched' or 'site'")
    if observables is None:
        omega, U = normal_modes(K0)
        observables = {
            "energy_K0": lambda V: float(mean_energy(V, K0)),
            "n_total": lambda V: float(np.sum(mode_occupations(V, omega=omega, U=U))),
        }

    def build(nsub):
        steps: List[Any] = []
        t = 0.0
        for p in range(int(n_pulses)):
            env = envelope_cos2(t, n_periods_ramp * T, n_periods_flat * T)
            n_per = 2 * n_periods_ramp + n_periods_flat
            for q in range(n_per):
                steps.extend(floquet_steps(drive, t, T, n_substeps=nsub, method=method,
                                           label=f"pulse{p}.{q}", envelope=env))
                t += T
                if eta < 1.0:
                    K_here = K0 + float(env(t)) * (drive(t) - K0)
                    steps.append(GeneratorStep(K_here, 0.0, label=f"boundary{p}.{q}"))
                    steps.append(ChannelStep(loss_step[0], loss_step[1], label=f"loss{p}.{q}"))
            # the envelope is exactly zero here: switch to K0 (telescoped
            # remainder of the last piece is booked under this label)
            steps.append(GeneratorStep(K0, 0.0, label=f"off{p}"))
            if gap > 0.0:
                steps.append(GeneratorStep(K0, float(gap), label=f"gap{p}"))
                t += float(gap)
        steps.append(GeneratorStep(K0, 0.0, label="end"))
        return steps, t

    history: List[Dict[str, Any]] = []
    prev = None
    movement = None
    converged = False
    res = None
    obs = None
    nsub = int(n_substeps_per_period)
    for level in range(int(max_halvings) + 1):
        nsub = int(n_substeps_per_period) * 2**level
        steps, t_end = build(nsub)
        res = run_protocol(V0, K0, steps, claim_ground_entry=claim_ground_entry,
                           report_modes=list(report_modes), run_causality=False,
                           strict=strict)
        obs = {k: float(f(res.V)) for k, f in observables.items()}
        movement = None if prev is None else max(abs(obs[k] - prev[k]) for k in obs)
        history.append({"level": level, "n_substeps_per_period": nsub,
                        "observables": dict(obs), "movement": movement,
                        "work": res.work_total, "dissipated": res.dissipated_total})
        if movement is not None and movement < conv_tol:
            converged = True
            break
        prev = obs
    if not converged and require_convergence:
        raise RuntimeError(
            f"pulse_train: observables did not converge to {conv_tol:g} within "
            f"{max_halvings} halvings (last movement {movement})"
        )
    return PulseTrainResult(V=res.V, result=res, converged=converged,
                            n_substeps_per_period=nsub, halvings=level,
                            movement=movement, observables=obs, history=history,
                            t_final=res.t_final)


# --------------------------------------------------------------------------
# Loss mapping, bounds and the amplification scan
# --------------------------------------------------------------------------


def loss_per_period(Q, omega, omega_mod):
    """Energy transmissivity per drive period of a mode with quality factor Q.

    eta = exp(-kappa T), kappa = omega/Q, T = 2 pi / omega_mod
    (exp(-pi/Q) at the principal resonance omega_mod = 2 omega).
    """
    return float(np.exp(-2.0 * np.pi * float(omega) / (float(omega_mod) * float(Q))))


def quality_factor_from_eta(eta, omega, omega_mod):
    """Inverse of :func:`loss_per_period`: Q = 2 pi omega / (omega_mod (-ln eta))."""
    eta = float(eta)
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must lie in (0, 1) to define a finite Q")
    return float(2.0 * np.pi * float(omega) / (float(omega_mod) * (-np.log(eta))))


def loss_threshold(lam):
    """Transmissivity per period at which eta lam^2 = 1 (net gain threshold).

    Below eta_th = 1/lam^2 the photon number decays despite the drive;
    above it the mode amplifies.  In Q language (principal resonance):
    Q_th = pi / (2 ln lam).
    """
    lam = np.asarray(lam, dtype=float)
    return 1.0 / np.maximum(lam, 1.0) ** 2


def bang_bang_bound(depth):
    """Exact two-level (bang-bang) optimum: largest multiplier per cycle.

    For omega^2(t) in {omega^2 (1 + d), omega^2 (1 - d)} the monodromy
    trace over one cycle is 2 cos(w1 t1) cos(w2 t2) - (w1/w2 + w2/w1)
    sin(w1 t1) sin(w2 t2); its maximum modulus (w1/w2 + w2/w1) is attained
    at w1 t1 = w2 t2 = pi/2 and gives lambda = w1/w2 = sqrt((1+d)/(1-d)),
    i.e. ln lambda = artanh(d): the largest amplitude gain any bounded
    two-level modulation of relative depth d can produce per cycle.  The
    cycle lasts T* = pi/(2 w1) + pi/(2 w2), so the growth *rate* is
    artanh(d)/T*.  Returns (lambda, T*/T_res) with T_res = pi/omega the
    principal-resonance period of the undriven mode (omega = 1 units).
    """
    d = np.asarray(depth, dtype=float)
    w1 = np.sqrt(1.0 + d)
    w2 = np.sqrt(1.0 - d)
    lam = w1 / w2
    T_star = 0.5 * np.pi / w1 + 0.5 * np.pi / w2
    return lam, T_star / np.pi


def mathieu_small_depth_rate(depth, omega=1.0):
    """Leading-order growth rate depth*omega/4 at the principal resonance.

    Landau-Lifshitz, Mechanics (3rd ed.), Sec. 27, Eq. (27.10): for
    x'' + omega^2 (1 + h cos(2 omega t)) x = 0 the amplitude grows as
    exp(s t) with s = h omega / 4.  Per period T = pi/omega that is
    ln lambda = pi h / 4.
    """
    return float(depth) * float(omega) / 4.0


def _golden_max(f, a, b, tol=1e-10, max_iter=200):
    """Golden-section maximisation of a unimodal f on [a, b]."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    it = 0
    while abs(b - a) > tol * max(1.0, abs(a) + abs(b)) and it < max_iter:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
        it += 1
    x = 0.5 * (a + b)
    return x, f(x), it


def _single_mode_multiplier(depth, omega_mod, profile, method, n_steps, conv_tol,
                            max_halvings, omega=1.0):
    drv = modulated_K(np.array([[omega**2]]), depth, omega_mod, profile)
    M, info = mode_monodromy(drv, np.array([omega**2]), n_steps=n_steps, method=method,
                             conv_tol=conv_tol, max_halvings=max_halvings)
    tr = M[0, 0, 0] + M[0, 1, 1]
    lam = 1.0 if abs(tr) <= 2.0 else 0.5 * (abs(tr) + np.sqrt(tr * tr - 4.0))
    return float(lam), info


def _scan_ln_multipliers(depth, omegas, profile, method, n_steps, conv_tol,
                         max_halvings, omega=1.0):
    """ln max|eig| over an omega_mod grid in one batched (gated) evaluation."""
    from .monodromy import stability_chart

    chart = stability_chart(np.array([[omega**2]]), [depth], np.asarray(omegas, dtype=float),
                            profile=profile, n_steps=n_steps, method=method,
                            conv_tol=conv_tol, max_halvings=max_halvings)
    return np.log(chart.multiplier[0])


def max_gain_over_frequency(depth, profile="cos", *, m=1, omega=1.0, method="cf4",
                            n_steps=32, conv_tol=1e-10, max_halvings=12,
                            omega_tol=1e-9, bracket=None, n_scan=97):
    """max over omega_mod of ln max|eig S_F| for the m-th tongue at `depth`.

    Golden-section search of the (unimodal within a tongue) Floquet
    multiplier over omega_mod, bracketed around 2 omega/m; returns a dict
    with the argmax frequency, ln lambda per period, the growth rate per
    unit time and the convergence metadata.
    """
    tip = 2.0 * float(omega) / int(m)
    if bracket is None:
        half = tip * min(0.9, 1.5 * float(depth) / int(m) + 0.02)
        bracket = (tip - half, tip + half)
    info_last = {}

    def f(w):
        lam, info = _single_mode_multiplier(depth, w, profile, method, n_steps,
                                            conv_tol, max_halvings, omega)
        info_last.update(info)
        return np.log(lam)

    # coarse scan first: outside the tongue the multiplier is exactly 1 (a
    # plateau of ln lambda = 0 on which golden-section probes are blind),
    # so locate the peak on a grid and refine within its neighbours.
    grid = np.linspace(bracket[0], bracket[1], int(n_scan))
    vals = _scan_ln_multipliers(depth, grid, profile, method, n_steps, conv_tol,
                                max_halvings, omega)
    i_best = int(np.argmax(vals))
    if vals[i_best] <= 0.0:
        return {
            "depth": float(depth), "omega_mod": float(grid[i_best]), "ln_multiplier": 0.0,
            "rate": 0.0, "period": 2.0 * np.pi / grid[i_best], "iterations": 0,
            "n_steps": info_last.get("n_steps"), "halvings": info_last.get("halvings"),
            "movement": info_last.get("movement"), "bracket": tuple(bracket),
            "m": int(m), "profile": profile,
        }
    lo = grid[max(i_best - 1, 0)]
    hi = grid[min(i_best + 1, grid.size - 1)]
    w_star, lnlam, iters = _golden_max(f, lo, hi, tol=omega_tol)
    return {
        "depth": float(depth), "omega_mod": float(w_star), "ln_multiplier": float(lnlam),
        "rate": float(lnlam * w_star / (2.0 * np.pi)), "period": 2.0 * np.pi / w_star,
        "iterations": iters, "n_steps": info_last.get("n_steps"),
        "halvings": info_last.get("halvings"), "movement": info_last.get("movement"),
        "bracket": tuple(bracket), "m": int(m), "profile": profile,
    }


@dataclass
class AmplificationScan:
    """Converged max-gain surface (README API: ``amplification_bound_scan``).

    ``ln_multiplier[i]`` is max over omega_mod of ln max|eig S_F| at
    ``depth_grid[i]`` (per period), ``omega_star[i]`` the maximising
    frequency, ``gain[i, j, l]`` = eta_grid[l] * lambda^2 at the *grid*
    frequency omega_grid[j] (the surface), ``eta_threshold[i]`` the loss
    threshold 1/lambda_max^2, and ``bang_bang`` / ``small_depth`` the two
    reference laws.  ``conjecture`` states the fitted bound.
    """

    depth_grid: np.ndarray
    omega_grid: np.ndarray
    eta_grid: np.ndarray
    multiplier_grid: np.ndarray  # (nd, nw)
    gain: np.ndarray  # (nd, nw, ne)
    ln_multiplier: np.ndarray  # (nd,)
    omega_star: np.ndarray  # (nd,)
    rate: np.ndarray  # (nd,) per unit time
    eta_threshold: np.ndarray  # (nd,)
    bang_bang_ln_multiplier: np.ndarray
    bang_bang_rate: np.ndarray
    small_depth_ln_multiplier: np.ndarray
    convergence: List[Dict[str, Any]]
    profile: str
    conjecture: Dict[str, Any] = field(default_factory=dict)


def amplification_bound_scan(depth_grid, omega_grid, eta_grid, *, profile="cos",
                             omega=1.0, method="cf4", n_steps=32, conv_tol=1e-10,
                             max_halvings=12, omega_tol=1e-9, m=1):
    """Numerically converged max-gain surface vs depth, frequency and loss.

    Single-mode (Mathieu-type) problem in units omega = 1: for uniform
    ('coupling') modulation every lattice mode reduces to it with
    omega_mod -> omega_mod/omega_k, so the surface is the lattice's too.
    """
    depth_grid = np.asarray(depth_grid, dtype=float).ravel()
    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    eta_grid = np.asarray(eta_grid, dtype=float).ravel()
    nd, nw, ne = depth_grid.size, omega_grid.size, eta_grid.size
    from .monodromy import stability_chart

    chart = stability_chart(np.array([[omega**2]]), depth_grid, omega_grid,
                            profile=profile, n_steps=n_steps, method=method,
                            conv_tol=conv_tol, max_halvings=max_halvings)
    mult = chart.multiplier
    gain = mult[:, :, None] ** 2 * eta_grid[None, None, :]
    lnlam = np.empty(nd)
    wstar = np.empty(nd)
    rate = np.empty(nd)
    conv = []
    for i, d in enumerate(depth_grid):
        r = max_gain_over_frequency(d, profile, m=m, omega=omega, method=method,
                                    n_steps=n_steps, conv_tol=conv_tol,
                                    max_halvings=max_halvings, omega_tol=omega_tol)
        lnlam[i] = r["ln_multiplier"]
        wstar[i] = r["omega_mod"]
        rate[i] = r["rate"]
        conv.append(r)
    bb_lam, bb_T = bang_bang_bound(depth_grid)
    bb_ln = np.log(bb_lam)
    bb_rate = bb_ln / (bb_T * np.pi / omega)
    sd = np.array([mathieu_small_depth_rate(d, omega) * np.pi / omega for d in depth_grid])
    scan = AmplificationScan(
        depth_grid=depth_grid, omega_grid=omega_grid, eta_grid=eta_grid,
        multiplier_grid=mult, gain=gain, ln_multiplier=lnlam, omega_star=wstar,
        rate=rate, eta_threshold=loss_threshold(np.exp(lnlam)),
        bang_bang_ln_multiplier=bb_ln, bang_bang_rate=bb_rate,
        small_depth_ln_multiplier=sd, convergence=conv, profile=profile,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_bb = np.where(bb_ln > 0, lnlam / bb_ln, np.nan)
        ratio_sd = np.where(sd > 0, lnlam / sd, np.nan)
        ratio_rate = np.where(bb_rate > 0, rate / bb_rate, np.nan)
    scan.conjecture = {
        "max_ratio_to_bang_bang_per_period": float(np.nanmax(ratio_bb)),
        "max_ratio_to_bang_bang_rate": float(np.nanmax(ratio_rate)),
        "ratio_to_small_depth_law": ratio_sd,
        "statement": (
            "ln max|eig S_F| <= artanh(depth) for every bounded modulation of "
            "relative depth `depth` (bang-bang optimum); the cosine drive "
            "reaches (pi/4) depth (1 + O(depth^2)) of it at the principal "
            "resonance"
        ),
    }
    return scan
