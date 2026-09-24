"""Lattice-field dynamics: the exact symplectic propagator of the chain.

For the quadratic lattice field H = (1/2)(p.p + x.K.x) the Heisenberg
operators evolve linearly, R(t) = S(t) R(0) with the symplectic propagator
S(t) = exp(t Omega H_mat), H_mat = diag(K, I) (block quadrature ordering
R = (x_1..x_N, p_1..p_N), hbar = 1).

Propagator construction: 'spectral' (default) uses the eigendecomposition
K = U diag(w^2) U^T, giving the exact normal-mode solution

    S(t) = [[ cos(Wt),  W^{-1} sin(Wt) ],
            [ -W sin(Wt),      cos(Wt) ]],   W = sqrt(K),

symplectic to eigh roundoff (~1e-15); 'expm' cross-checks with
scipy.linalg.expm via vacuum.core.symplectic_from_quadratic
(Serafini, "Quantum Continuous Variables", 2017, Sec. 3.1).

History: promoted verbatim from vacuum.audits.causality (M-I item I.1) —
detectors, inequalities, and floquet all need it, and physics modules must
not import from audits. vacuum.audits.causality re-imports it, so both
import paths resolve to this one function object.
"""

from __future__ import annotations

import numpy as np

from .gaussian import symplectic_from_quadratic

__all__ = ["chain_propagator"]


def chain_propagator(K, t, method="spectral"):
    """Symplectic propagator S(t) = exp(t Omega diag(K, I)) of the lattice field.

    'spectral' evaluates the exact normal-mode form via eigh(K) (fast, and
    symplectic to machine precision); 'expm' calls the generic matrix
    exponential from vacuum.core (slower, useful as a cross-check).
    """
    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    if method == "spectral":
        w2, U = np.linalg.eigh(0.5 * (K + K.T))
        if np.min(w2) <= 0:
            raise ValueError(
                f"K must be positive definite for the propagator: min eig {np.min(w2):.3e}"
            )
        w = np.sqrt(w2)
        c = np.cos(w * t)
        s = np.sin(w * t)
        C = (U * c) @ U.T          # cos(sqrt(K) t)
        Sw = (U * (s / w)) @ U.T   # K^{-1/2} sin(sqrt(K) t)
        Ws = (U * (w * s)) @ U.T   # K^{1/2} sin(sqrt(K) t)
        return np.block([[C, Sw], [-Ws, C]])
    if method == "expm":
        H_mat = np.zeros((2 * N, 2 * N))
        H_mat[:N, :N] = 0.5 * (K + K.T)
        H_mat[N:, N:] = np.eye(N)
        return symplectic_from_quadratic(H_mat, t)
    raise ValueError(f"method must be 'spectral' or 'expm', got {method!r}")
