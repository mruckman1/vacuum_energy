"""Lattice models: coupling matrices K for H = (1/2)(p.p + x.K.x).

The Layer-1 workhorse is the 1D chain of coupled harmonic oscillators
discretizing the Klein-Gordon field (PLAN.md, Layer 1),

    H = (1/2) sum_i [ pi_i^2 + m^2 phi_i^2 + (phi_{i+1} - phi_i)^2 ],

whose coupling matrix is K = m^2 I + L with L the graph Laplacian of the
chain (diagonal 2, nearest-neighbour -1). This is the normalization used by
Srednicki, PRL 71, 666 (1993) (unit lattice spacing) and in the critical-chain
entropy anchors of Calabrese-Cardy, JSTAT P06002 (2004).

Boundary conditions:
- 'periodic':  phi_N = phi_0 (circulant L; massless case is singular via the
  zero mode, so ground_state_cov requires m > 0 here).
- 'dirichlet': phi_{-1} = phi_N = 0 (standard Dirichlet discrete Laplacian:
  diagonal 2 everywhere, no wraparound).
"""

from __future__ import annotations

import numpy as np

__all__ = ["harmonic_chain_K"]


def harmonic_chain_K(N, m, bc="periodic"):
    """Coupling matrix of the harmonic chain, K = m^2 I + chain Laplacian.

    Parameters
    ----------
    N : int
        Number of oscillators (modes).
    m : float
        Mass (in lattice units, hbar = 1).
    bc : {'periodic', 'dirichlet'}
        Boundary condition (see module docstring).

    Returns
    -------
    K : (N, N) ndarray, symmetric; positive definite for m > 0
        (and for m = 0 with Dirichlet boundaries).
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    K = np.zeros((N, N))
    np.fill_diagonal(K, m**2 + 2.0)
    idx = np.arange(N - 1)
    K[idx, idx + 1] = -1.0
    K[idx + 1, idx] = -1.0
    if bc == "periodic":
        if N > 1:
            K[0, N - 1] += -1.0
            K[N - 1, 0] += -1.0
        else:  # single site, phi_1 = phi_0: the gradient term vanishes
            K[0, 0] = m**2
    elif bc == "dirichlet":
        pass  # diagonal 2 everywhere, no wraparound
    else:
        raise ValueError(f"bc must be 'periodic' or 'dirichlet', got {bc!r}")
    return K
