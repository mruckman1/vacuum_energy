"""Energy ledger: mean-energy accounting that must close for every protocol.

PLAN.md (Layer 0): "total energy accounting must close; any protocol that
appears to net energy without an external drive or classical-information
input is flagged as a bug, because passivity of the ground state
(Pusz-Woronowicz, Commun. Math. Phys. 58, 273 (1978)) forbids it."

The ledger is first-law bookkeeping at the level of means, and Layer 3's
two-point-measurement work statistics are the promised refinement: snapshots
record <H> = mean_energy(V, K) at labeled protocol stages, explicit entries
record work injected by external drives and energy dissipated into
environments (Gaussian channels), and `refine_work` attaches the full work
*distribution* of a drive step (mean, variance, DeltaF, Jarzynski average
and defect, optional quantiles) from
`vacuum.inequalities.work_stats.counting_statistics`. A refinement whose TPM
mean disagrees with the ledger's own work entry is an audit failure, so the
distribution and the mean bookkeeping cannot drift apart.
`close()` verifies

    E_last - E_first = sum(work in) - sum(dissipated out)

within tolerance and returns the ledger table. A protocol whose books do not
close is either mis-instrumented or — after the anomaly protocol — an event.
The tolerance is *scaled*, not absolute: the identity telescopes exactly in
exact arithmetic, so the only error is float64 roundoff, whose size is set by
the energy magnitudes actually booked and by how many operations were booked
(see :meth:`EnergyLedger.closure_tolerance` for the derivation and its
calibration).

Conventions per docs/API.md: hbar = 1, block quadrature ordering; energies
are the quadratic means <H> = (1/2) Tr(V_pp) + (1/2) Tr(K V_xx).
"""

from __future__ import annotations

import numbers

import numpy as np

from vacuum.core import mean_energy

from .result import AuditResult, AuditViolation

__all__ = [
    "EnergyLedger",
    "MAX_OPS_PER_ENTRY",
    "energy_ledger",
    "WORK_STATS_REQUIRED_KEYS",
    "DEFAULT_CLOSURE_ATOL",
    "DEFAULT_ULP_BUDGET",
]

#: float64 unit roundoff.
_EPS = float(np.finfo(float).eps)

#: Absolute floor of the closure criterion: the smallest defect the audit will
#: ever call a violation, so a genuinely tiny-energy protocol is still checked
#: (a protocol at <H> ~ 1e-6 is held to 1e-6 in relative terms).  Three orders
#: below the 1e-9 absolute tolerance this replaced.
DEFAULT_CLOSURE_ATOL = 1e-12

#: Roundoff budget of the relative term, in ulps of the booked energy scale per
#: booked operation per sqrt(mode) — see :meth:`EnergyLedger.closure_tolerance`.
#: Calibrated as 31x the worst constant measured over protocol scans spanning
#: 8-160 modes, 34-18434 booked operations and <H> = 5.7-530 (worst 0.13,
#: median 0.03), and 100x the constant of the circuit-QED anchor that exposed
#: the absolute criterion (0.040 at 105 modes, 49152 operations, <H> = 227).
#: The headroom is deliberate: accumulated roundoff is a random quantity, and
#: an audit that fires on its own tail is the defect this criterion replaces.
DEFAULT_ULP_BUDGET = 4.0

#: Largest ``n_ops`` a single entry may declare.  ``n_ops`` is caller-declared
#: and enters the closure tolerance linearly, so an inflated declaration would
#: loosen the audit on the caller's own protocol; this bounds the damage.  It
#: is a sanity bound, not the defence — the defence is that every declaration
#: is stored on its row and reported in the closure result.  The largest
#: legitimate declaration in this repository is 131072 (the L4 composite's
#: finest dt-halving level) and the deepest reachable one is 2**20 (a 64-point
#: grid at ``protocol_evolve``'s 14 halvings), so the cap sits 16x above what
#: any integrator here can produce.
MAX_OPS_PER_ENTRY = 2 ** 24


def _validated_n_ops(n_ops, label):
    """A declared operation count, or a loud refusal.

    ``n_ops`` must be a positive integer (an integral float is accepted) no
    larger than :data:`MAX_OPS_PER_ENTRY`.  Silently coercing a bad value
    would let a caller tighten or loosen its own audit by accident, so the
    value must already *be* a real number: a numeric string is refused
    rather than parsed, since a string that survives here is a type
    coercion waiting to happen downstream.
    """
    if isinstance(n_ops, (bool, str, bytes)) or not isinstance(n_ops, numbers.Real):
        raise TypeError(
            f"n_ops for {label!r} must be a real number, got {type(n_ops).__name__} "
            f"{n_ops!r}"
        )
    try:
        value = float(n_ops)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"n_ops for {label!r} must be an integer, got {n_ops!r}") from exc
    if not np.isfinite(value) or value != int(value):
        raise ValueError(f"n_ops for {label!r} must be a whole number, got {n_ops!r}")
    value = int(value)
    if value < 1:
        raise ValueError(f"n_ops for {label!r} must be >= 1, got {value}")
    if value > MAX_OPS_PER_ENTRY:
        raise ValueError(
            f"n_ops for {label!r} is {value}, above the cap {MAX_OPS_PER_ENTRY}: "
            "an entry cannot claim more float64 accumulations than any integrator "
            "in this stack performs (see MAX_OPS_PER_ENTRY)"
        )
    return value


#: keys a counting-statistics dict must carry to be accepted by
#: :meth:`EnergyLedger.refine_work` (the contract published as
#: ``vacuum.inequalities.work_stats.COUNTING_STATISTICS_KEYS``; named here so
#: the audits package does not import the inequalities package).
WORK_STATS_REQUIRED_KEYS = (
    "mean",
    "variance",
    "delta_F",
    "jarzynski_average",
    "jarzynski_defect",
)


class EnergyLedger:
    """Accounting object for a protocol's mean-energy budget.

    Sign conventions: `work(label, w)` records energy *entering* the system
    from an external drive (w > 0 means energy in); `dissipate(label, q)`
    records energy *leaving* into an environment (q > 0 means energy out).
    Either may be negative to describe the reverse flow — the closure
    identity is what matters.

    Parameters
    ----------
    K : (N, N) ndarray, optional
        Default coupling matrix used by `record` when none is given
        (protocols on a fixed Hamiltonian set it once here).
    tol : float
        Absolute *floor* of the closure criterion (:data:`DEFAULT_CLOSURE_ATOL`);
        the criterion itself scales — see :meth:`closure_tolerance`.
    ulp_budget : float
        Roundoff budget of the relative term, in ulps of the booked energy
        scale per booked operation per sqrt(mode)
        (:data:`DEFAULT_ULP_BUDGET`).  Pass 0.0 for a purely absolute
        criterion.
    """

    def __init__(self, K=None, tol=DEFAULT_CLOSURE_ATOL, ulp_budget=DEFAULT_ULP_BUDGET):
        self.K = None if K is None else np.asarray(K, dtype=float)
        self.tol = float(tol)
        self.ulp_budget = float(ulp_budget)
        self._rows = []  # dicts: {"kind", "label", "value", "n_ops"[, "n_modes"]}

    # -- entries ---------------------------------------------------------

    def record(self, label, V, K=None):
        """Snapshot <H> for state V (under K, defaulting to the ledger's K).

        Returns the recorded energy so callers can compute work/dissipation
        from mean-energy differences without recomputing.
        """
        K_use = self.K if K is None else np.asarray(K, dtype=float)
        if K_use is None:
            raise ValueError("no coupling matrix: pass K here or at construction")
        V = np.asarray(V, dtype=float)
        E = float(mean_energy(V, K_use))
        self._rows.append(
            {
                "kind": "snapshot",
                "label": str(label),
                "value": E,
                "n_ops": 1,
                "n_modes": int(V.shape[0] // 2),
            }
        )
        return E

    def work(self, label, amount, n_ops=1):
        """Record external work put into the system (positive = energy in).

        `n_ops` is the number of float64 energy accumulations behind the
        entry — 1 for a single switch, `n_substeps` for a work term
        telescoped over an integrator's substeps.  It enters only the
        closure tolerance (:meth:`closure_tolerance`), never the books.  It
        is *caller-declared*: it must be a positive integer at or below
        :data:`MAX_OPS_PER_ENTRY` (anything else raises), it is stored on the
        row and reported in the closure result, and the trust it still
        requires is documented in :meth:`closure_tolerance`.
        """
        label = str(label)
        self._rows.append(
            {
                "kind": "work",
                "label": label,
                "value": float(amount),
                "n_ops": _validated_n_ops(n_ops, label),
            }
        )
        return float(amount)

    def dissipate(self, label, amount, n_ops=1):
        """Record energy dissipated to an environment (positive = energy out).

        `n_ops` counts the float64 accumulations behind the entry, as in
        :meth:`work` (a channel applied once per substep of a gated
        integrator aggregates `n_substeps` of them), with the same
        validation and the same declared-not-verified status.
        """
        label = str(label)
        self._rows.append(
            {
                "kind": "dissipation",
                "label": label,
                "value": float(amount),
                "n_ops": _validated_n_ops(n_ops, label),
            }
        )
        return float(amount)

    # -- beyond-mean refinement (Layer 3, M3.5) --------------------------

    def refine_work(self, label, stats, tol=None, record_work=True):
        """Attach two-point-measurement work statistics to a drive step.

        This is the refinement the module docstring promises: `stats` is the
        dict returned by
        :func:`vacuum.inequalities.work_stats.counting_statistics`, carrying
        the TPM mean and variance of the work, the exact free-energy
        difference, the Jarzynski average <e^{-beta W}> and its defect
        against e^{-beta DeltaF}, and (dense route) truncation diagnostics
        and work quantiles. The mean-level books are unchanged: a
        ``work_stats`` row never enters the closure sum.

        If a ``work`` row already carries `label`, its value is cross-checked
        against ``stats['mean']`` and a mismatch raises
        :class:`AuditViolation` — the distribution and the mean bookkeeping
        are not allowed to drift apart. Otherwise, with `record_work` true
        (the default), the TPM mean is recorded as that step's work entry, so
        a protocol instrumented purely through counting statistics still
        closes.

        Parameters
        ----------
        label : str
            Name of the drive step; matched against existing ``work`` rows.
        stats : mapping
            Counting statistics; must contain
            :data:`WORK_STATS_REQUIRED_KEYS`.
        tol : float, optional
            Consistency tolerance for the mean cross-check (default: the
            ledger's absolute floor, raised by the same roundoff budget as
            :meth:`closure_tolerance` against the size of the two numbers
            compared).
        record_work : bool
            Record ``stats['mean']`` as a ``work`` entry when no ``work`` row
            with this label exists yet.

        Returns
        -------
        dict
            A copy of the stored statistics.
        """
        label = str(label)
        tol_given = None if tol is None else float(tol)
        try:
            missing = [k for k in WORK_STATS_REQUIRED_KEYS if k not in stats]
        except TypeError as exc:  # not a mapping
            raise TypeError(
                f"stats must be a mapping of counting statistics, got {type(stats)!r}"
            ) from exc
        if missing:
            raise ValueError(
                f"counting statistics for {label!r} missing required keys: {missing}"
            )
        stored = dict(stats)
        mean = float(stored["mean"])

        existing = [
            r for r in self._rows if r["kind"] == "work" and r["label"] == label
        ]
        if existing:
            recorded = sum(r["value"] for r in existing)
            defect = recorded - mean
            n_modes = max(
                max((int(r.get("n_modes", 1)) for r in self._snapshots()), default=1), 1
            )
            n_ops = max(1, sum(int(r.get("n_ops", 1)) for r in existing))
            floor = self.tol if tol_given is None else tol_given
            tol = max(
                floor,
                self.ulp_budget
                * _EPS
                * float(np.sqrt(n_modes))
                * n_ops
                * max(abs(recorded), abs(mean)),
            )
            if abs(defect) > tol:
                result = AuditResult(
                    name="work_statistics_consistency",
                    passed=False,
                    worst_violation=abs(defect),
                    details={
                        "label": label,
                        "ledger_work": recorded,
                        "tpm_mean": mean,
                        "defect": defect,
                        "tol": tol,
                        "stats": stored,
                    },
                )
                raise AuditViolation(result)
        elif record_work:
            self.work(label, mean)

        self._rows.append(
            {
                "kind": "work_stats",
                "label": label,
                "value": mean,
                "n_ops": 0,  # a refinement, not an accumulation: no roundoff budget
                "stats": stored,
            }
        )
        return dict(stored)

    # -- bookkeeping views ----------------------------------------------

    @property
    def table(self):
        """The ledger table: list of {'kind', 'label', 'value'} rows, in order.

        Rows also carry ``'n_ops'`` (the float64 accumulations behind the
        entry, which only the closure tolerance reads) and, on snapshots,
        ``'n_modes'``.  ``work_stats`` rows carry an extra ``'stats'``
        mapping (deep-copied out, so the table is a snapshot and not a
        handle on the ledger).
        """
        out = []
        for row in self._rows:
            copy = dict(row)
            if "stats" in copy:
                copy["stats"] = dict(copy["stats"])
            out.append(copy)
        return out

    @property
    def work_statistics(self):
        """{label: stats} for every :meth:`refine_work` entry, last wins."""
        return {
            r["label"]: dict(r["stats"])
            for r in self._rows
            if r["kind"] == "work_stats"
        }

    def _snapshots(self):
        return [r for r in self._rows if r["kind"] == "snapshot"]

    # -- closure ---------------------------------------------------------

    def closure_tolerance(self, tol=None):
        """The scaled closure criterion, with the numbers that set it.

        **Why the criterion cannot be absolute.** Every booked entry is an
        energy *difference* taken between two float64 mean energies of the
        same trajectory, so the closure identity telescopes *exactly*: in
        exact arithmetic the defect is identically zero, and the only error
        source is roundoff.  Roundoff is not a fixed number of joules.  Two
        things set it:

        * **the energy scale being differenced.**  One mean-energy
          evaluation <H> = (1/2) Tr(V_pp) + (1/2) Tr(K V_xx) carries an
          absolute error ~ eps |E| (eps = 2.2e-16); the trace is a sum over
          the 2n entries of a quadratic form, so the cancellations inside it
          add a random-walk factor ~ sqrt(n) in the mode count.  A protocol
          therefore cannot resolve its own books below ~ eps sqrt(n) <H>,
          *however small its physics defect is*.
        * **how many accumulations were booked.**  Each substep of a
          symplectic integrator re-rounds V, and the work term booked at the
          next switch is evaluated on the rounded V, so the errors are
          correlated and accumulate at the recursive-summation rate — linear
          in the number of booked operations, not sqrt.

        Measured on this stack (protocol scans over 8-160 modes, 34-49152
        booked operations, <H> = 5.7-530): defect / (eps <H>) = 0.024 *
        n_modes^0.57 * n_ops^0.99, i.e. **linear in the operation count and
        sqrt in the mode count**, with an envelope constant of 0.13 (median
        0.03) against the sqrt/linear form.  The circuit-QED anchor that
        exposed the old absolute criterion sits at 0.040 (105 modes, 49152
        operations, <H> = 227, ~0.6 ulp per midpoint substep) — the same
        constant, which is why an absolute 1e-9 fired there on arithmetic
        and nowhere else.

        Hence

            tol = max(atol, ulp_budget * eps * sqrt(n_modes) * n_ops * scale)

        with

        * ``scale`` = max(largest |snapshot <H>|, sum |work|, sum
          |dissipation|) — the energy magnitudes the books actually carry,
          not the size of the net flows they cancel to;
        * ``n_ops`` = booked operations: one per snapshot, work and
          dissipation row, except that a row may declare more (a work term
          telescoped over ``n_substeps`` midpoint substeps declares them);
        * ``n_modes`` = largest snapshot mode count;
        * ``atol`` = :data:`DEFAULT_CLOSURE_ATOL`, the floor that keeps a
          low-energy protocol under a hard check;
        * ``ulp_budget`` = :data:`DEFAULT_ULP_BUDGET` = 4 ulps, 31x the
          worst measured constant.

        **The trade, stated plainly.**  This is not uniformly stricter.  It
        is ~1000x stricter at the floor (a small protocol is held to 1e-12
        instead of 1e-9) and stricter for every protocol whose roundoff
        floor is below 1e-9 — most of the repository — but it is *looser*
        exactly where the old criterion was firing on arithmetic: at the
        circuit-QED cutoff-bracket cells that motivated the change (105
        modes, 49152 operations, <H> = 227) the tolerance is 1.02e-7 against
        the old 1e-9, i.e. **102x looser there**.  That is the point — those
        cells' own roundoff floor is ~1e-9 — but it is a widening, not a
        tightening, and it is what buys the scale covariance: rescaling H
        (and so every booked energy) rescales the tolerance with it, so a
        physically identical protocol can no longer fire or not fire
        according to how energetic it is or how many substeps its
        convergence gate happened to reach.

        **What it catches and what it cannot.**  The criterion bounds the
        defect *relative to the protocol's own energy scale*.  So it catches
        any violation that is large relative to `scale` — a mis-booked
        drive, an unbooked channel, a phantom work term — at any system
        size, and it catches arbitrarily small ones in low-energy protocols
        through `atol`.  It cannot catch a violation that is small relative
        to `scale` but large relative to *its own entry*: on the <H> = 227,
        49152-operation cell above, a **10 % mis-booking of a 1e-6 work
        entry (1e-7) does not fire** against tol 1.02e-7, while a 100 % one
        does.  No criterion can do better there — the float64 floor of that
        protocol is itself ~1e-9 — but the limitation is real and is the
        price of instrumenting a large, energetic protocol: fine-grained
        book errors are only resolvable in protocols whose energy scale (or
        entry count) is small enough for `atol` to bind.

        **What it trusts.**  `n_ops` is *declared by the caller*, not
        measured by the ledger: an entry that over-declares its substep
        count loosens the tolerance of its own protocol linearly (and one
        that under-declares tightens it).  Three defences: declarations are
        validated as positive integers at or below
        :data:`MAX_OPS_PER_ENTRY`; the declared count sits on the row, so
        ``AuditResult.details['table']`` carries every one of them for
        after-the-fact audit; and the totals — ``n_ops`` against
        ``n_entries``, with ``n_ops_excess`` and ``max_n_ops`` — are
        reported here and in the closure result, so a protocol claiming more
        operations than it booked entries is visible without re-running it.
        The residual assumption is that a caller's declaration matches the
        integrator it actually ran; in this repository the three call sites
        (`vacuum.protocol.run_protocol`,
        `vacuum.detectors.nonperturbative.protocol_evolve` and the channel
        columns beside it) pass the same `n_substeps`/`n_steps` the stepper
        used, and the ledger cannot verify that from the outside.

        Returns
        -------
        dict
            ``tol`` (the criterion), ``atol``, ``relative`` (the scaled
            term), ``scale``, ``n_ops``, ``n_entries``, ``n_ops_excess``,
            ``max_n_ops``, ``n_modes``, ``ulp_budget``, ``eps``.
        """
        atol = self.tol if tol is None else float(tol)
        snaps = self._snapshots()
        scale = max(
            max((abs(r["value"]) for r in snaps), default=0.0),
            sum(abs(r["value"]) for r in self._rows if r["kind"] == "work"),
            sum(abs(r["value"]) for r in self._rows if r["kind"] == "dissipation"),
        )
        booked = [
            r for r in self._rows if r["kind"] in ("snapshot", "work", "dissipation")
        ]
        declared = [int(r.get("n_ops", 1)) for r in booked]
        n_entries = max(1, len(booked))
        n_ops = max(1, sum(declared))
        n_modes = max(max((int(r.get("n_modes", 1)) for r in snaps), default=1), 1)
        relative = self.ulp_budget * _EPS * float(np.sqrt(n_modes)) * n_ops * scale
        return {
            "tol": max(atol, relative),
            "atol": atol,
            "relative": relative,
            "scale": scale,
            "n_ops": n_ops,
            # auditability of the caller-declared counts (see the docstring):
            "n_entries": n_entries,
            "n_ops_excess": n_ops - n_entries,
            "max_n_ops": max(declared, default=1),
            "n_modes": n_modes,
            "ulp_budget": self.ulp_budget,
            "eps": _EPS,
        }

    def close(self, tol=None, strict=True):
        """Verify Delta E_system = sum(work in) - sum(dissipated out).

        Returns an :class:`AuditResult` whose details carry the full ledger
        ``table`` plus ``delta_E``, ``work_in``, ``dissipated`` and the signed
        closure ``defect`` = delta_E - (work_in - dissipated); in strict mode
        (default) a non-closing ledger raises :class:`AuditViolation`.

        The criterion is the scaled one of :meth:`closure_tolerance` (its
        inputs are reported in ``details['tolerance']``); `tol` overrides the
        *absolute floor*, not the criterion.
        """
        budget = self.closure_tolerance(tol)
        tol = budget["tol"]
        snaps = self._snapshots()
        if len(snaps) < 2:
            raise ValueError(
                f"ledger needs at least two snapshots to close, has {len(snaps)}"
            )
        delta_E = snaps[-1]["value"] - snaps[0]["value"]
        work_in = sum(r["value"] for r in self._rows if r["kind"] == "work")
        dissipated = sum(r["value"] for r in self._rows if r["kind"] == "dissipation")
        defect = delta_E - (work_in - dissipated)
        passed = abs(defect) <= tol
        result = AuditResult(
            name="energy_ledger",
            passed=passed,
            worst_violation=abs(defect) if not passed else 0.0,
            details={
                "table": self.table,
                "delta_E": delta_E,
                "work_in": work_in,
                "dissipated": dissipated,
                "defect": defect,
                "tol": tol,
                "tolerance": budget,
                "first_snapshot": snaps[0]["label"],
                "last_snapshot": snaps[-1]["label"],
                "work_statistics": self.work_statistics,
            },
        )
        if strict and not passed:
            raise AuditViolation(result)
        return result

    def __repr__(self) -> str:
        lines = [f"EnergyLedger({len(self._rows)} rows)"]
        for r in self._rows:
            line = f"  {r['kind']:<11s} {r['label']:<24s} {r['value']:+.12e}"
            if r["kind"] == "work_stats":
                line += f"  (std={r['stats'].get('std', float('nan')):.6e})"
            lines.append(line)
        return "\n".join(lines)


def energy_ledger(K=None, tol=DEFAULT_CLOSURE_ATOL, ulp_budget=DEFAULT_ULP_BUDGET):
    """Create an :class:`EnergyLedger` (factory named per docs/API.md)."""
    return EnergyLedger(K=K, tol=tol, ulp_budget=ulp_budget)
