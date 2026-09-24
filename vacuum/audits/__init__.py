"""vacuum.audits — the standing audits of the Vacuum Program (Layer 0).

Executable versions of the hard walls (PLAN.md): thermodynamics (energy
ledger), passivity (Pusz-Woronowicz), causality (Lieb-Robinson / vanishing
spacelike commutators), and precision (the nu -> 1/2 danger zone). Every
simulated protocol, in every layer, passes through these checks.

Each audit returns an :class:`AuditResult` (passed, worst_violation, details)
and raises :class:`AuditViolation` in strict mode — audits raise/flag loudly
and never silently pass. An audit failure on validated code is an *event*,
not an error: see the anomaly protocol in PLAN.md.
"""

from .result import AuditResult, AuditViolation
from .ledger import (
    DEFAULT_CLOSURE_ATOL,
    DEFAULT_ULP_BUDGET,
    MAX_OPS_PER_ENTRY,
    WORK_STATS_REQUIRED_KEYS,
    EnergyLedger,
    energy_ledger,
)
from .passivity import (
    default_local_symplectic_sampler,
    embed_local_symplectic,
    passivity_audit,
)
from .causality import causality_audit, chain_propagator
from .precision import precision_audit

__all__ = [
    "AuditResult",
    "AuditViolation",
    "energy_ledger",
    "EnergyLedger",
    # contract for EnergyLedger.refine_work — the M3.5 counting_statistics
    # payload (vacuum.inequalities.work_stats.COUNTING_STATISTICS_KEYS)
    "WORK_STATS_REQUIRED_KEYS",
    "DEFAULT_CLOSURE_ATOL",
    "DEFAULT_ULP_BUDGET",
    "MAX_OPS_PER_ENTRY",
    "passivity_audit",
    "default_local_symplectic_sampler",
    "embed_local_symplectic",
    "causality_audit",
    "chain_propagator",
    "precision_audit",
]
