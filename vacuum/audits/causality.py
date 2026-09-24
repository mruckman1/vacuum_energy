"""Causality audit: field commutators vanish outside the lattice light cone.

For the quadratic lattice field H = (1/2)(p.p + x.K.x) the Heisenberg
operators evolve linearly, R(t) = S(t) R(0) with the symplectic propagator
S(t) = exp(t Omega H_mat), H_mat = diag(K, I). The unequal-time commutators
are then state-independent c-numbers,

    [R_i(t), R_j(0)] = i (S(t) Omega)_ij      (hbar = 1, block ordering),

so the phi-phi (x-x) block of S(t) Omega is the lattice Pauli-Jordan function.
For the harmonic chain K = m^2 I + Laplacian the dispersion is
omega(k) = sqrt(m^2 + 4 sin^2(k/2)) with group velocity
|d omega/d k| = |sin k| / omega(k) <= 1, so commutators at separations
|i - j| > t (lattice units) lie outside the light cone and must vanish up to
Lieb-Robinson tails, which decay super-exponentially in the distance beyond
the cone (E. H. Lieb and D. W. Robinson, Commun. Math. Phys. 28, 251 (1972);
free-lattice sharpness: sub-cone leakage ~ (t/d)^d from the Bessel-function
kernel). The audit also verifies that the propagator is symplectic,
S Omega S^T = Omega, which is exact for quadratic dynamics (Serafini,
"Quantum Continuous Variables", 2017, Sec. 3.1).

Propagator construction lives in vacuum.core.dynamics.chain_propagator
(promoted from this module, M-I item I.1); it is re-imported here so the
historical import path vacuum.audits.chain_propagator keeps working and
resolves to the same function object.
"""

from __future__ import annotations

import numpy as np

from vacuum.core import Omega
from vacuum.core.dynamics import chain_propagator  # re-import shim (I.1)

from .result import AuditResult, AuditViolation

__all__ = ["causality_audit", "chain_propagator"]


def _lattice_distance(N, bc):
    """Pairwise 1D lattice distances d(i, j); periodic wraps around the ring."""
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    d = np.abs(i - j)
    if bc == "periodic":
        d = np.minimum(d, N - d)
    elif bc != "open":
        raise ValueError(f"bc must be 'periodic' or 'open', got {bc!r}")
    return d


def causality_audit(
    K,
    t,
    v_max=1.0,
    buffer=15.0,
    tol_commutator=1e-8,
    tol_symplectic=1e-10,
    bc="periodic",
    method="spectral",
    strict=True,
):
    """Verify commutators vanish outside the light cone and S(t) is symplectic.

    Checks, for the chain Hamiltonian K evolved to time t:

    1. |[phi_i(t), phi_j(0)]| = |(S(t) Omega)_{ij}| < tol_commutator for every
       pair with lattice distance d(i, j) > v_max * t + buffer (the buffer
       absorbs super-exponential Lieb-Robinson tails);
    2. max|S(t) Omega S(t)^T - Omega| < tol_symplectic.

    The audit cannot pass vacuously: it raises ValueError when no pair lies
    outside the cone-plus-buffer (lattice too small for this t), and reports
    the largest inside-cone commutator in details so callers can confirm the
    dynamics actually propagated.

    Parameters
    ----------
    K : (N, N) ndarray
        Chain coupling matrix (vacuum.core.models.harmonic_chain_K or
        compatible 1D ordering: mode index = lattice site).
    t : float
        Evolution time (lattice units, hbar = 1).
    v_max : float
        Maximal group velocity; 1.0 for the harmonic chain dispersion
        omega(k) = sqrt(m^2 + 4 sin^2(k/2)).
    buffer : float
        Extra lattice sites beyond the cone before the vanishing check starts.
    bc : {'periodic', 'open'}
        Distance convention: periodic uses ring distance min(|i-j|, N-|i-j|).
    method : {'spectral', 'expm'}
        Propagator construction (see :func:`chain_propagator`).
    strict : bool
        If True (default), raise :class:`AuditViolation` on failure.

    Returns
    -------
    AuditResult
        ``worst_violation`` = max(0, max_out - tol_commutator,
        defect - tol_symplectic); ``details`` holds
        ``max_abs_commutator_outside``, ``max_abs_commutator_inside``,
        ``symplectic_defect``, ``cone_radius`` (= v_max*t + buffer),
        ``n_outside_pairs``, and the tolerances.
    """
    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    S = chain_propagator(K, t, method=method)
    Om = Omega(N)

    defect = float(np.max(np.abs(S @ Om @ S.T - Om)))

    # phi-phi commutator matrix: [x_i(t), x_j(0)] = i (S Omega)_{ij}, i,j < N.
    Cxx = (S @ Om)[:N, :N]
    d = _lattice_distance(N, bc)
    cone_radius = v_max * t + buffer
    outside = d > cone_radius
    inside = d <= v_max * t
    n_outside = int(np.count_nonzero(outside))
    if n_outside == 0:
        raise ValueError(
            f"no site pairs outside the cone: v_max*t + buffer = {cone_radius:.1f} "
            f"exceeds every lattice distance on N = {N}; audit would be vacuous"
        )
    max_out = float(np.max(np.abs(Cxx)[outside]))
    max_in = float(np.max(np.abs(Cxx)[inside])) if np.any(inside) else 0.0

    passed = (max_out < tol_commutator) and (defect < tol_symplectic)
    worst = max(0.0, max_out - tol_commutator, defect - tol_symplectic)
    result = AuditResult(
        name="causality",
        passed=passed,
        worst_violation=worst,
        details={
            "max_abs_commutator_outside": max_out,
            "max_abs_commutator_inside": max_in,
            "symplectic_defect": defect,
            "cone_radius": float(cone_radius),
            "n_outside_pairs": n_outside,
            "t": float(t),
            "v_max": float(v_max),
            "buffer": float(buffer),
            "tol_commutator": float(tol_commutator),
            "tol_symplectic": float(tol_symplectic),
            "method": method,
        },
    )
    if strict and not passed:
        raise AuditViolation(result)
    return result
