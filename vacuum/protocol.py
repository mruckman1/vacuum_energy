"""Protocol runner: the single audited code path for Gaussian protocols (M-I.4).

Every Layer 2/3 pipeline runs its covariance-matrix protocol through
:func:`run_protocol`, so the audit discipline is structural rather than
per-author (docs/PLAN_LAYERS_2_3.md, M-I.4): a protocol is an initial
covariance, a sequence of steps — Hamiltonian evolution pieces or Gaussian
channels — and an :class:`vacuum.audits.EnergyLedger`, and the standing
audits run at entry and exit:

- **passivity** on any state *claimed* to be a ground state (entry and/or
  exit, per the caller's explicit claim flags);
- **precision** on every reported reduced state at exit (the audit's
  ``flagged`` field is carried into the result — near-floor spectra are
  where float64 lies);
- **causality** once per (K, t_max) pair actually evolved under;
- **ledger closure** mandatory — it always runs, last, and a non-closing
  ledger raises :class:`vacuum.audits.AuditViolation` in strict mode.

Step API (rides unchanged under M2.2's ``protocol_evolve`` and Layer 5's
Floquet drives):

- :class:`GeneratorStep` — evolve under a quadratic Hamiltonian
  H = (1/2) R^T H_mat R for a duration. The generator may be a static
  (N, N) coupling matrix K (meaning H_mat = diag(K, I), docs/API.md), a
  static (2N, 2N) H_mat, or a **time-dependent callable t -> H_mat**
  (absolute protocol time; returns either shape). A Floquet drive is a
  periodic callable — nothing else changes.
- :class:`ChannelStep` — apply a Gaussian channel ``V -> channel(V, *args,
  **kwargs)`` (e.g. ``vacuum.core.channels.loss``), instantaneous in
  protocol time.

Energy bookkeeping follows the standard open-system first-law split
(R. Alicki, J. Phys. A 12, L103 (1979)): work is energy entering through a
*change of the Hamiltonian* at fixed state, W = ∫ <∂H/∂t> dt; heat/
dissipation is energy leaving through a *change of state* at fixed
Hamiltonian. Concretely:

- a generator step whose Hamiltonian differs from the current one charges
  the sudden-switch work (1/2) Tr((H_new - H_old) V) — the step-function
  limit of <∂H/∂t> dt — and time-dependent generator steps charge the
  midpoint-discretized drive work (see :func:`midpoint_evolve`); both feed
  the ledger's ``work`` entries;
- a channel step feeds a ``dissipate`` entry from the before/after
  mean-energy difference under the current Hamiltonian, which makes ledger
  closure automatic for channels rather than estimated (spec M2.2).

Discretization contract: time-dependent steps use the exponential midpoint
rule, S_k = exp(dt Ω H(t_k + dt/2)) — the second-order Magnus integrator
(S. Blanes, F. Casas, J. A. Oteo, J. Ros, Phys. Rep. 470, 151 (2009),
Sec. 5.4) — so every substep is *exactly* symplectic and the only error is
Magnus truncation. The drive work is charged by telescoping against the
piecewise-constant Hamiltonian the stepper actually simulates, so the
ledger closes to roundoff at ANY substep count: closure tests the
instrumentation, while dt-halving convergence of the reported observables
(the caller's gate, per spec M2.2) tests the physics. Roundoff is the *only*
error in that closure, and it grows with the booked energy scale and with the
substep count, so the closure criterion scales with both (the substep count is
declared to the ledger as the entry's ``n_ops``; see
:meth:`vacuum.audits.EnergyLedger.closure_tolerance`).

Conventions per docs/API.md: hbar = 1, V_vac = I/2, nu >= 1/2, block
quadrature ordering R = (x_1..x_N, p_1..p_N), energies in hbar = 1 units.
Pure-functional over arrays: no input is mutated; the ledger is the one
designated accumulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from vacuum.audits import (
    EnergyLedger,
    causality_audit,
    passivity_audit,
    precision_audit,
)
from vacuum.core import reduce as reduce_cov
from vacuum.core import symplectic_from_quadratic

__all__ = [
    "GeneratorStep",
    "ChannelStep",
    "ProtocolResult",
    "run_protocol",
    "midpoint_evolve",
    "hamiltonian_matrix",
    "coupling_from_hamiltonian",
    "quadratic_mean_energy",
]


# --------------------------------------------------------------------------
# Quadratic-form helpers (pure functions over arrays)
# --------------------------------------------------------------------------


def hamiltonian_matrix(K):
    """H_mat = diag(K, I) for H = (1/2)(p.p + x.K.x) (docs/API.md convention).

    Block quadrature ordering: the x-x block is the (symmetrized) coupling
    matrix K, the p-p block is the identity, no x-p cross terms.
    """
    K = np.asarray(K, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got shape {K.shape}")
    N = K.shape[0]
    H = np.zeros((2 * N, 2 * N))
    H[:N, :N] = 0.5 * (K + K.T)
    H[N:, N:] = np.eye(N)
    return H


def coupling_from_hamiltonian(H_mat, atol=1e-12):
    """Extract K from H_mat = diag(K, I); return None if not of that form.

    The ledger's snapshot convention (vacuum.core.mean_energy) is defined
    for H = (1/2)(p.p + x.K.x), i.e. H_mat with unit p-p block and no x-p
    coupling; this is the checkable inverse of :func:`hamiltonian_matrix`.
    """
    H_mat = np.asarray(H_mat, dtype=float)
    N = H_mat.shape[0] // 2
    if H_mat.shape != (2 * N, 2 * N):
        return None
    if not np.allclose(H_mat[N:, N:], np.eye(N), rtol=0.0, atol=atol):
        return None
    if not (
        np.allclose(H_mat[:N, N:], 0.0, rtol=0.0, atol=atol)
        and np.allclose(H_mat[N:, :N], 0.0, rtol=0.0, atol=atol)
    ):
        return None
    return H_mat[:N, :N].copy()


def quadratic_mean_energy(V, H_mat):
    """<H> = (1/2) Tr(H_mat V) for H = (1/2) R^T H_mat R (mean-zero state).

    Coincides with vacuum.core.mean_energy(V, K) when H_mat = diag(K, I),
    and with ``vacuum.core.mean_energy(V, H_mat)`` (its general-H_mat
    path) always.
    """
    V = np.asarray(V, dtype=float)
    H_mat = np.asarray(H_mat, dtype=float)
    return 0.5 * float(np.trace(H_mat @ V))


def _as_quadratic_form(H, N, what="generator"):
    """Normalize a generator matrix to a symmetric (2N, 2N) H_mat.

    Accepts an (N, N) coupling matrix K (-> diag(K, I)) or a (2N, 2N)
    quadratic form (symmetrized defensively). Always returns a fresh array
    (no aliasing of caller data — pure-functional discipline).
    """
    H = np.asarray(H, dtype=float)
    if H.shape == (N, N):
        return hamiltonian_matrix(H)
    if H.shape == (2 * N, 2 * N):
        return 0.5 * (H + H.T)
    raise ValueError(
        f"{what} matrix must have shape ({N}, {N}) (a coupling matrix K) or "
        f"({2 * N}, {2 * N}) (a quadratic form H_mat), got {H.shape}"
    )


def _switch_work(H_new, H_old, V):
    """Sudden-switch work (1/2) Tr((H_new - H_old) V) at fixed state V.

    Step-function limit of W = integral <dH/dt> dt (Alicki 1979 split)."""
    return 0.5 * float(np.trace((H_new - H_old) @ V))


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorStep:
    """Evolve under a quadratic Hamiltonian for `duration` (hbar = 1 time).

    Parameters
    ----------
    H : ndarray or callable
        Static (N, N) coupling matrix K, static (2N, 2N) quadratic form
        H_mat, or a callable ``t -> H_mat`` of *absolute protocol time*
        returning either shape (time-dependent generator; Floquet drives
        pass a periodic callable here).
    duration : float
        Step duration, >= 0. A zero-duration step with a new generator is a
        pure sudden quench: work is charged, nothing evolves.
    n_substeps : int, optional
        Number of exponential-midpoint substeps for a *callable* generator
        (mandatory there — the dt is an explicit, recorded convergence
        knob, never a silent default). Ignored for static generators, which
        use one exact exponential.
    label : str, optional
        Ledger/step-log label; defaults to ``step<i>``.
    """

    H: Union[np.ndarray, Callable[[float], np.ndarray]]
    duration: float
    n_substeps: Optional[int] = None
    label: Optional[str] = None


@dataclass(frozen=True)
class ChannelStep:
    """Apply a Gaussian channel ``V -> channel(V, *args, **kwargs)``.

    `channel` is any covariance-in/covariance-out callable, e.g.
    ``vacuum.core.channels.loss`` with ``args=(modes, eta, nbar)``.
    Channels are instantaneous in protocol time; their energy change is
    booked as a ledger ``dissipate`` entry (positive = energy out).
    """

    channel: Callable[..., np.ndarray]
    args: Tuple[Any, ...] = ()
    kwargs: Optional[Dict[str, Any]] = None
    label: Optional[str] = None


def _normalize_step(step, index):
    """Coerce a raw (generator, duration) / (channel, args) tuple to a step.

    Disambiguation is by the second element: a real number means a
    generator step; a tuple/list (positional args) or dict (keyword args)
    means a channel step. Explicit GeneratorStep/ChannelStep instances pass
    through untouched and are the recommended spelling.
    """
    if isinstance(step, (GeneratorStep, ChannelStep)):
        return step
    if isinstance(step, tuple) and 2 <= len(step) <= 3:
        second = step[1]
        if isinstance(second, Real) and not isinstance(second, bool):
            n_sub = step[2] if len(step) == 3 else None
            return GeneratorStep(H=step[0], duration=float(second), n_substeps=n_sub)
        if isinstance(second, (tuple, list)):
            kwargs = step[2] if len(step) == 3 else None
            return ChannelStep(channel=step[0], args=tuple(second), kwargs=kwargs)
        if isinstance(second, dict):
            return ChannelStep(channel=step[0], args=(), kwargs=dict(second))
    raise TypeError(
        f"step {index}: expected GeneratorStep, ChannelStep, "
        f"(generator, duration[, n_substeps]) or (channel, args[, kwargs]); "
        f"got {step!r}"
    )


# --------------------------------------------------------------------------
# Time-dependent evolution (pure function; M2.2's protocol_evolve rides it)
# --------------------------------------------------------------------------


def midpoint_evolve(V, H_of_t, t0, duration, n_substeps):
    """Exponential-midpoint evolution under a time-dependent quadratic H.

    Steps V with S_k = exp(dt Omega H(t_k + dt/2)) — the second-order
    Magnus integrator (Blanes-Casas-Oteo-Ros, Phys. Rep. 470, 151 (2009),
    Sec. 5.4). Every substep is exactly symplectic (it is built by
    vacuum.core.symplectic_from_quadratic), so the only physics error is
    Magnus truncation, controlled by dt-halving at the caller.

    The returned drive work telescopes against the piecewise-constant
    Hamiltonian actually simulated,

        W = sum over switches of (1/2) Tr((H_next - H_prev) V_at_switch),

    which is (a) the midpoint discretization of W = integral <dH/dt> dt
    (Alicki, J. Phys. A 12, L103 (1979)) and (b) *exactly* the energy
    change of the simulated dynamics — mean energy under each constant
    H(t_k + dt/2) is conserved by its own exponential — so ledger closure
    holds to roundoff at any n_substeps.

    Parameters
    ----------
    V : (2N, 2N) ndarray
        Input covariance (not mutated).
    H_of_t : callable
        ``t -> H_mat`` at absolute time t; returns (N, N) K or (2N, 2N)
        H_mat.
    t0, duration : float
        Absolute start time and step length.
    n_substeps : int
        Number of midpoint substeps, >= 1.

    Returns
    -------
    (V_out, work, H_start, H_end)
        Final covariance, drive work charged over [t0, t0 + duration], and
        the (2N, 2N) quadratic forms H(t0) and H(t0 + duration) — the
        caller charges any boundary switch against H_start and continues
        the protocol from H_end.
    """
    V = np.asarray(V, dtype=float)
    N = V.shape[0] // 2
    n_substeps = int(n_substeps)
    if n_substeps < 1:
        raise ValueError(f"n_substeps must be >= 1, got {n_substeps}")
    if duration < 0:
        raise ValueError(f"duration must be >= 0, got {duration}")

    H_start = _as_quadratic_form(H_of_t(t0), N, "H_of_t(t0)")
    if duration == 0.0:
        return V.copy(), 0.0, H_start, H_start

    dt = duration / n_substeps
    H_prev = H_start
    work = 0.0
    for k in range(n_substeps):
        H_mid = _as_quadratic_form(H_of_t(t0 + (k + 0.5) * dt), N, "H_of_t")
        work += _switch_work(H_mid, H_prev, V)
        S = symplectic_from_quadratic(H_mid, dt)
        V = S @ V @ S.T
        H_prev = H_mid
    H_end = _as_quadratic_form(H_of_t(t0 + duration), N, "H_of_t(t_end)")
    work += _switch_work(H_end, H_prev, V)
    return V, work, H_start, H_end


# --------------------------------------------------------------------------
# Result record
# --------------------------------------------------------------------------


@dataclass
class ProtocolResult:
    """Everything a downstream sweep stores per protocol row.

    Attributes
    ----------
    V : (2N, 2N) ndarray
        Final covariance.
    K : (N, N) ndarray or None
        Coupling matrix of the exit Hamiltonian when it is of diag(K, I)
        form; None when the exit Hamiltonian carries x-p terms (see
        ``H_mat``).
    t_final : float
        Total protocol time.
    ledger : EnergyLedger
        The (closed) ledger, table included.
    audits : dict
        ``passivity_entry`` / ``passivity_exit`` (AuditResult or None),
        ``causality`` (list of AuditResult, one per (K, t_max) pair),
        ``precision`` (list of {"modes": tuple, "result": AuditResult}),
        ``ledger`` (AuditResult from the mandatory close).
    step_log : list of dict
        Per-step record: kind, label, timing, n_substeps/dt, the work or
        dissipation charged, and whether the generator changed — the
        convergence/provenance metadata the Layer 2/3 sweeps must carry.
    work_total, dissipated_total : float
        Sums of the entries this run charged to the ledger.
    flagged : bool
        True if ANY reported reduced state tripped the precision audit's
        near-floor guard (spec M2.2: the flag travels with every row).
    trajectory : list of (t, V) or None
        Post-step snapshots when ``keep_states=True``.
    H_mat : (2N, 2N) ndarray
        The exit Hamiltonian as a full quadratic form (always set; equals
        ``hamiltonian_matrix(K)`` when ``K`` is not None).  The exit ledger
        snapshot is booked under it via ``vacuum.core.mean_energy``'s
        general-H_mat path when ``K`` is None.
    """

    V: np.ndarray
    K: Optional[np.ndarray]
    t_final: float
    ledger: EnergyLedger
    audits: Dict[str, Any]
    step_log: List[Dict[str, Any]]
    work_total: float
    dissipated_total: float
    flagged: bool
    trajectory: Optional[List[Tuple[float, np.ndarray]]] = None
    H_mat: Optional[np.ndarray] = None

    def all_passed(self) -> bool:
        """True iff every audit that ran passed."""
        checks = []
        for key in ("passivity_entry", "passivity_exit"):
            if self.audits.get(key) is not None:
                checks.append(self.audits[key].passed)
        checks.extend(r.passed for r in self.audits.get("causality", []))
        checks.extend(d["result"].passed for d in self.audits.get("precision", []))
        checks.append(self.audits["ledger"].passed)
        return all(checks)

    def __repr__(self) -> str:
        n_caus = len(self.audits.get("causality", []))
        n_prec = len(self.audits.get("precision", []))
        verdict = "all audits PASSED" if self.all_passed() else "audit FAILURE"
        return (
            f"ProtocolResult(t_final={self.t_final:g}, steps={len(self.step_log)}, "
            f"work={self.work_total:+.6e}, dissipated={self.dissipated_total:+.6e}, "
            f"causality x{n_caus}, precision x{n_prec}, flagged={self.flagged}, "
            f"{verdict})"
        )


# --------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------


def run_protocol(
    V0,
    K0,
    steps,
    ledger=None,
    *,
    claim_ground_entry=False,
    claim_ground_exit=False,
    report_modes=None,
    run_causality=True,
    causality_kwargs=None,
    passivity_kwargs=None,
    precision_kwargs=None,
    ledger_tol=None,
    strict=True,
    keep_states=False,
):
    """Run a Gaussian protocol through the standing audits (spec M-I.4).

    Parameters
    ----------
    V0 : (2N, 2N) ndarray
        Initial covariance (block ordering; not mutated).
    K0 : (N, N) ndarray
        Initial coupling matrix: the protocol starts under H = (1/2)(p.p +
        x.K0.x), and the entry ledger snapshot is taken with it.
    steps : sequence
        GeneratorStep / ChannelStep instances, or raw tuples
        ``(generator, duration[, n_substeps])`` / ``(channel, args[,
        kwargs])`` — see :func:`_normalize_step` for the disambiguation
        rule.
    ledger : EnergyLedger, optional
        Accounting object; a fresh ``EnergyLedger(K=K0, tol=1e-9)`` is
        created when omitted. Closure is mandatory and runs last.
    claim_ground_entry, claim_ground_exit : bool
        Explicit ground-state claims. A claimed state gets the passivity
        audit (Pusz-Woronowicz) under the Hamiltonian in force at that
        boundary; claiming an active state makes the audit fire — the
        claim is checkable, per spec M2.2.
    report_modes : None, [] or sequence of mode-index sequences
        Which reduced states of the final V are *reported* (and therefore
        precision-audited, spec M-I.4). None reports the full state —
        note that a pure final state sits exactly on the vacuum floor and
        escalates to mpmath, which is O(N^3) arbitrary-precision work:
        pass the physically reported blocks explicitly for large N. An
        empty list reports nothing (allowed only when a caller audits its
        reported states itself).
    run_causality : bool
        Run the causality audit once per (K, t_max) pair, where K ranges
        over the distinct static diag(K, I)-form generators evolved under
        and t_max is the total time evolved under that K (the propagator
        pieces compose; the largest time is the binding check). Disable
        for non-chain mode orderings (e.g. detector-augmented K_tot) where
        the 1D lattice-distance convention of the audit does not apply.
        Time-dependent steps are not causality-audited here (their
        instantaneous K is not a fixed chain); the M2.2 stack cross-checks
        them through the kernel route.
    causality_kwargs, passivity_kwargs, precision_kwargs : dict, optional
        Forwarded to the respective audits (tolerances, bc, buffer,
        samplers, dps, ...). ``strict`` is always supplied by the runner.
    ledger_tol : float, optional
        Override for the *absolute floor* of the closure criterion (the
        criterion itself scales with the booked energies and the booked
        operation count -- see :meth:`vacuum.audits.EnergyLedger.
        closure_tolerance`); defaults to the ledger's own
        (:data:`vacuum.audits.ledger.DEFAULT_CLOSURE_ATOL`).
    strict : bool
        If True (default) any failing audit raises AuditViolation at the
        point it runs (ledger closure last); if False, failed AuditResults
        are returned in ``audits`` for the caller to inspect.
    keep_states : bool
        Record (t, V) after every step in ``result.trajectory``.

    Returns
    -------
    ProtocolResult

    Raises
    ------
    AuditViolation
        In strict mode, from the first failing standing audit — a
        non-closing ledger raises at close.
    ValueError
        On malformed steps, or when ``claim_ground_exit`` is set while the
        exit Hamiltonian is not of diag(K, I) form (the passivity audit
        takes a coupling matrix).  A general exit H_mat — an 'xp'-coupled
        detector stack with lambda(t_end) != 0, say — is otherwise
        accepted: its snapshot is booked as (1/2) Tr(H_mat V).
    """
    K0 = np.asarray(K0, dtype=float)
    if K0.ndim != 2 or K0.shape[0] != K0.shape[1]:
        raise ValueError(f"K0 must be square, got shape {K0.shape}")
    N = K0.shape[0]
    V = np.array(V0, dtype=float, copy=True)
    if V.shape != (2 * N, 2 * N):
        raise ValueError(f"V0 shape {V.shape} incompatible with K0 shape {K0.shape}")

    if ledger is None:
        ledger = (
            EnergyLedger(K=K0)
            if ledger_tol is None
            else EnergyLedger(K=K0, tol=ledger_tol)
        )
    steps_n = [_normalize_step(s, i) for i, s in enumerate(steps)]

    audits: Dict[str, Any] = {
        "passivity_entry": None,
        "passivity_exit": None,
        "causality": [],
        "precision": [],
        "ledger": None,
    }

    # -- entry audits and snapshot --------------------------------------
    if claim_ground_entry:
        audits["passivity_entry"] = passivity_audit(
            K0, V=V, strict=strict, **(passivity_kwargs or {})
        )
    ledger.record("entry", V, K=K0)

    H_current = hamiltonian_matrix(K0)
    t = 0.0
    work_total = 0.0
    dissipated_total = 0.0
    step_log: List[Dict[str, Any]] = []
    trajectory: Optional[List[Tuple[float, np.ndarray]]] = None
    if keep_states:
        trajectory = [(0.0, V.copy())]
    # distinct static K pieces -> [K, total duration] for the causality audit
    causality_acc: Dict[bytes, List[Any]] = {}

    # -- the steps -------------------------------------------------------
    for i, step in enumerate(steps_n):
        label = step.label if step.label is not None else f"step{i}"
        if isinstance(step, GeneratorStep):
            duration = float(step.duration)
            if duration < 0:
                raise ValueError(f"step {i} ({label}): duration must be >= 0")
            entry: Dict[str, Any] = {
                "index": i,
                "kind": "generator",
                "label": label,
                "t_start": t,
                "duration": duration,
            }
            if callable(step.H):
                if step.n_substeps is None:
                    raise ValueError(
                        f"step {i} ({label}): a time-dependent generator needs an "
                        "explicit n_substeps (the dt is a recorded convergence knob)"
                    )
                V_in = V
                V, drive_work, H_start, H_end = midpoint_evolve(
                    V_in, step.H, t, duration, step.n_substeps
                )
                w = _switch_work(H_start, H_current, V_in) + drive_work
                # the entry telescopes over n_substeps float64 accumulations:
                # its roundoff budget in the closure criterion scales with them
                ledger.work(label, w, n_ops=int(step.n_substeps))
                work_total += w
                H_current = H_end
                entry.update(
                    time_dependent=True,
                    n_substeps=int(step.n_substeps),
                    dt=duration / int(step.n_substeps) if duration > 0 else 0.0,
                    work=w,
                    generator_changed=True,
                    causality_audited=False,
                )
            else:
                H_new = _as_quadratic_form(step.H, N, f"step {i} generator")
                changed = not np.array_equal(H_new, H_current)
                w = 0.0
                if changed:
                    w = _switch_work(H_new, H_current, V)
                    ledger.work(label, w)
                    work_total += w
                    H_current = H_new
                if duration > 0:
                    S = symplectic_from_quadratic(H_current, duration)
                    V = S @ V @ S.T
                K_step = coupling_from_hamiltonian(H_current)
                audited = run_causality and duration > 0 and K_step is not None
                if audited:
                    key = K_step.tobytes()
                    if key not in causality_acc:
                        causality_acc[key] = [K_step, 0.0]
                    causality_acc[key][1] += duration
                entry.update(
                    time_dependent=False,
                    n_substeps=1,
                    dt=duration,
                    work=w,
                    generator_changed=changed,
                    causality_audited=audited,
                )
            t += duration
            entry["t_end"] = t
            step_log.append(entry)
        else:  # ChannelStep
            E_before = quadratic_mean_energy(V, H_current)
            V_new = np.asarray(
                step.channel(V, *step.args, **(step.kwargs or {})), dtype=float
            )
            if V_new.shape != (2 * N, 2 * N):
                raise ValueError(
                    f"step {i} ({label}): channel returned shape {V_new.shape}, "
                    f"expected {(2 * N, 2 * N)}"
                )
            E_after = quadratic_mean_energy(V_new, H_current)
            q = E_before - E_after  # positive = energy out (ledger convention)
            ledger.dissipate(label, q)
            dissipated_total += q
            V = V_new
            step_log.append(
                {
                    "index": i,
                    "kind": "channel",
                    "label": label,
                    "t_start": t,
                    "t_end": t,
                    "duration": 0.0,
                    "dissipated": q,
                }
            )
        if keep_states:
            trajectory.append((t, V.copy()))

    # -- exit snapshot and audits ---------------------------------------
    # The exit snapshot is booked under the Hamiltonian actually in force:
    # its coupling matrix when it is of diag(K, I) form, otherwise the full
    # quadratic form — vacuum.core.mean_energy takes either, the general
    # path being (1/2) Tr(H_mat V).  The derivative-coupled ('xp') detector
    # stack ends on such a form whenever lambda(t_end) != 0.
    K_exit = coupling_from_hamiltonian(H_current)
    ledger.record("exit", V, K=H_current if K_exit is None else K_exit)

    if claim_ground_exit:
        if K_exit is None:
            raise ValueError(
                "claim_ground_exit needs an exit Hamiltonian of diag(K, I) "
                "form: the passivity audit takes a coupling matrix, and a "
                "state with x-p coupling switched on is not a claimable "
                "ground state of this stack"
            )
        audits["passivity_exit"] = passivity_audit(
            K_exit, V=V, strict=strict, **(passivity_kwargs or {})
        )

    if run_causality:
        for K_piece, t_max in causality_acc.values():
            audits["causality"].append(
                causality_audit(
                    K_piece, t_max, strict=strict, **(causality_kwargs or {})
                )
            )

    if report_modes is None:
        report_list: List[Sequence[int]] = [tuple(range(N))]
    else:
        report_list = [tuple(int(m) for m in modes) for modes in report_modes]
    for modes in report_list:
        V_red = reduce_cov(V, modes)
        res = precision_audit(V_red, strict=strict, **(precision_kwargs or {}))
        audits["precision"].append({"modes": tuple(modes), "result": res})
    flagged = any(
        bool(d["result"].details.get("flagged", False)) for d in audits["precision"]
    )

    # -- mandatory ledger closure (always last) --------------------------
    audits["ledger"] = ledger.close(tol=ledger_tol, strict=strict)

    return ProtocolResult(
        V=V,
        K=K_exit,
        t_final=t,
        ledger=ledger,
        audits=audits,
        step_log=step_log,
        work_total=work_total,
        dissipated_total=dissipated_total,
        flagged=flagged,
        trajectory=trajectory,
        H_mat=H_current,
    )
