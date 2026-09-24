"""Precision audit: symplectic spectra near the vacuum floor nu = 1/2.

PLAN.md (Layer 0): "symplectic eigenvalues near 1/2 are the danger zone; the
stack escalates float64 -> longdouble -> mpmath automatically when Williamson
spectra approach the vacuum floor." Entropy and modular-Hamiltonian formulas
contain ln(nu - 1/2), so what must be resolved is the *gap* nu - 1/2, which
float64 eigensolvers lose to cancellation exactly where the physics lives —
"the fastest route to embarrassment is announcing [a passivity/causality]
violation from a rounding error at nu = 1/2" (anomaly protocol).

`precision_audit(V, tol)` flags a state whose float64 spectrum sits within
`tol` of the floor, routes it through the vacuum.core.precision escalation
ladder (longdouble only where the platform's longdouble is wider than float64,
e.g. not on darwin/arm64; mpmath otherwise), reports which regime produced
the returned numbers, and — independently of flagging — verifies physicality:
min(nu) >= 1/2 - phys_tol (the Robertson-Schrodinger uncertainty bound
V + i Omega/2 >= 0 in symplectic-eigenvalue form; Simon-Mukunda-Dutta,
Phys. Rev. A 49, 1567 (1994)). Being *near* the floor is not a failure — it
is the trigger for escalation; being *below* it is.
"""

from __future__ import annotations

import numpy as np

from vacuum.core import entropy_from_nu, symplectic_eigenvalues
from vacuum.core.precision import entropy_mp, near_vacuum_floor, symplectic_eigenvalues_mp

from .result import AuditResult, AuditViolation

__all__ = ["precision_audit"]

VACUUM_NU = 0.5


def _longdouble_available():
    """True where longdouble is genuinely wider than float64 (not darwin/arm64)."""
    return np.finfo(np.longdouble).eps < np.finfo(np.float64).eps


def precision_audit(V, tol=1e-10, dps=50, phys_tol=1e-12, strict=True):
    """Flag near-floor symplectic spectra and escalate precision automatically.

    Parameters
    ----------
    V : (2N, 2N) ndarray
        Covariance matrix to audit.
    tol : float
        Danger-zone width: the audit flags (and escalates) when the float64
        spectrum has min(nu) - 1/2 < tol (vacuum.core.precision guard).
    dps : int
        mpmath working precision (decimal digits) for the escalated rung.
    phys_tol : float
        Physicality allowance: min(nu) >= 1/2 - phys_tol must hold for the
        most precise spectrum available (a genuine vacuum lands at the floor
        up to input roundoff; well below it the state is unphysical).
    strict : bool
        If True (default), raise :class:`AuditViolation` when the state is
        unphysical. Flagging alone never raises.

    Returns
    -------
    AuditResult
        ``passed`` is the physicality verdict; ``worst_violation`` =
        max(0, 1/2 - min(nu)) from the most precise regime used.
        ``details``:

        - ``regime``: 'float64' | 'longdouble' | 'mpmath' — which rung
          produced the returned spectrum/entropy;
        - ``flagged``: True if the float64 guard tripped (min gap < tol);
        - ``nu`` / ``min_nu`` / ``gap``: escalated spectrum, its minimum, and
          min(nu) - 1/2;
        - ``min_nu_float64``: the float64 minimum, for comparison;
        - ``entropy``: von Neumann entropy (nats) evaluated in the reported
          regime (mpmath end-to-end when escalated);
        - ``entropy_float64``: the unescalated value.
    """
    V = np.asarray(V, dtype=float)
    nu64 = symplectic_eigenvalues(V)
    S64 = entropy_from_nu(nu64)
    flagged = near_vacuum_floor(nu64, tol)

    regime = "float64"
    nu = nu64
    S = S64
    if flagged:
        if _longdouble_available():
            # Longdouble rung: recompute the SVD input products in extended
            # precision (mirrors vacuum.core.precision's ladder).
            from vacuum.core.precision import longdouble_refine

            nu_ld = longdouble_refine(V)
            if not near_vacuum_floor(nu_ld, tol):
                regime = "longdouble"
                nu = nu_ld
                S = entropy_from_nu(nu_ld)
        if regime == "float64":  # longdouble absent or still in the danger zone
            regime = "mpmath"
            nu = symplectic_eigenvalues_mp(V, dps=dps)
            S = entropy_mp(V, dps=dps)

    min_nu = float(np.min(nu))
    worst = max(0.0, VACUUM_NU - min_nu)
    passed = worst <= phys_tol
    result = AuditResult(
        name="precision",
        passed=passed,
        worst_violation=worst,
        details={
            "regime": regime,
            "flagged": bool(flagged),
            "nu": nu,
            "min_nu": min_nu,
            "gap": min_nu - VACUUM_NU,
            "min_nu_float64": float(np.min(nu64)),
            "entropy": float(S),
            "entropy_float64": float(S64),
            "tol": float(tol),
            "phys_tol": float(phys_tol),
            "dps": int(dps),
        },
    )
    if strict and not passed:
        raise AuditViolation(result)
    return result
