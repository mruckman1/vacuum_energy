"""Bosonic (Gaussian) modular Hamiltonians on top of vacuum.core.

region_modular          G of rho_A ~ exp(-(1/2) R^T G R) for a region of the
                        ground state of H = (1/2)(p.p + x^T K x), via the
                        Williamson construction of vacuum.core.entanglement_hamiltonian
                        (Botero-Reznik 2004; Eisert-Cramer-Plenio 2010). Modes at the
                        vacuum floor nu = 1/2 make G singular; the clamp inside the
                        core routine is *reported*, never silent: the returned info
                        lists the floor modes and, when any mode is within floor_tol of
                        1/2, re-evaluates the symplectic spectrum in mpmath
                        (vacuum.core.precision) so the caller can see the true margins.
modular_energy_profile  Spatial decomposition of <K> = (1/2) Tr(G V_A) into per-mode
                        contributions (1/2)[(G V_A)_{kk} + (G V_A)_{N+k,N+k}], which resum
                        exactly to vacuum.core.modular_energy.

The first-law machinery (delta S = delta <K>) is vacuum.modular.first_law.
"""

from __future__ import annotations

import numpy as np

from vacuum.core import (
    entanglement_hamiltonian,
    ground_state_cov,
    modular_energy,
    reduce,
    symplectic_eigenvalues,
)
from vacuum.core.precision import near_vacuum_floor, symplectic_margins_mp

__all__ = ["region_modular", "modular_energy_profile"]


def region_modular(V, K, modes, floor_tol=1e-10, dps=50, return_info=False):
    """Entanglement Hamiltonian G of the reduced state of `modes`.

    Parameters
    ----------
    V : (2N, 2N) ndarray or None
        Global covariance; if None it is the ground state of K.
    K : (N, N) ndarray or None
        Coupling matrix (used only when V is None).
    modes : sequence of int
        0-based mode indices of the region.
    floor_tol : float
        Precision-audit threshold on nu - 1/2 (docs/API.md): below it the
        symplectic margins are re-evaluated in mpmath at `dps` digits.
    return_info : bool
        If True return (G, info): info = {'nu', 'floor_mask', 'escalated',
        'margins_mp' (None unless escalated), 'n_floor'}.
    """
    if V is None:
        if K is None:
            raise ValueError("give V or K")
        V = ground_state_cov(np.asarray(K, dtype=float))
    V_A = reduce(V, modes)
    G = entanglement_hamiltonian(V_A)
    nu = symplectic_eigenvalues(V_A)
    escalated = near_vacuum_floor(nu, floor_tol)
    info = {
        "nu": nu,
        "floor_mask": (nu - 0.5) < floor_tol,
        "n_floor": int(np.sum((nu - 0.5) < floor_tol)),
        "escalated": bool(escalated),
        "margins_mp": symplectic_margins_mp(V_A, dps=dps) if escalated else None,
    }
    return (G, info) if return_info else G


def modular_energy_profile(G, V_A):
    """Per-mode decomposition of <K> = (1/2) Tr(G V_A) in block quadrature ordering.

    Returns (N,) with entries (1/2)[(G V_A)_{kk} + (G V_A)_{N+k, N+k}]; their sum
    equals vacuum.core.modular_energy(G, V_A) to roundoff.
    """
    G = np.asarray(G, dtype=float)
    V_A = np.asarray(V_A, dtype=float)
    N = G.shape[0] // 2
    d = np.diag(G @ V_A)
    prof = 0.5 * (d[:N] + d[N:])
    assert abs(float(np.sum(prof)) - modular_energy(G, V_A)) < 1e-9 * max(1.0, abs(modular_energy(G, V_A)))
    return prof
