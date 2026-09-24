"""Result object and exception shared by the standing audits (Layer 0).

Every audit returns an :class:`AuditResult` — a small record with a boolean
verdict, the magnitude of the worst violation found, and a details dict — and
raises :class:`AuditViolation` (carrying that same result) when it fails in
strict mode. Audits never silently pass: a failure either raises or comes back
with ``passed=False`` and a loud repr.

Per PLAN.md (Layer 0): an audit failure on validated code is an *event*, not an
error — the result object therefore carries enough detail (the violating
operation, the worst matrix element, the escalation regime) to start the
anomaly protocol rather than a debugging session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

__all__ = ["AuditResult", "AuditViolation"]


@dataclass
class AuditResult:
    """Outcome of one standing audit.

    Attributes
    ----------
    name : str
        Which audit produced this result (e.g. ``'passivity'``).
    passed : bool
        True if the audit's invariant held within its tolerance.
    worst_violation : float
        Magnitude of the worst violation found; 0.0 for a clean pass. Its
        units/meaning are audit-specific and documented by each audit.
    details : dict
        Audit-specific diagnostics (tolerances used, worst operation,
        precision regime, ledger table, ...).
    """

    name: str
    passed: bool
    worst_violation: float
    details: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.passed)

    def require(self) -> "AuditResult":
        """Raise :class:`AuditViolation` unless the audit passed; else return self."""
        if not self.passed:
            raise AuditViolation(self)
        return self

    def __repr__(self) -> str:  # loud, greppable verdict
        verdict = "PASSED" if self.passed else "FAILED"
        keys = ", ".join(sorted(self.details))
        return (
            f"AuditResult({self.name}: {verdict}, "
            f"worst_violation={self.worst_violation:.6e}, details=[{keys}])"
        )


class AuditViolation(AssertionError):
    """A standing audit failed.

    Subclasses AssertionError so an unhandled audit failure reads as a broken
    invariant. The offending :class:`AuditResult` is available as ``.result``.
    """

    def __init__(self, result: AuditResult):
        self.result = result
        super().__init__(
            f"standing audit '{result.name}' FAILED: "
            f"worst_violation={result.worst_violation:.6e}; details={result.details!r}"
        )
