"""Gaussian channels acting on covariance matrices, V -> X V X^T + Y.

Complete positivity of a Gaussian channel (X, Y) requires
    Y + (i/2) (Omega - X Omega X^T)  >=  0
as a Hermitian matrix (Holevo-Werner; A. Serafini, "Quantum Continuous
Variables", CRC 2017, Sec. 5.3). All channels here satisfy it by construction:

- loss: beamsplitter of transmissivity eta mixing the listed modes with a
  thermal environment of occupation nbar (X = sqrt(eta), Y = (1-eta)(nbar+1/2)I
  on those modes).
- thermal_noise: classical Gaussian noise addition (X = I, Y = nbar_added I on
  the listed modes) — CP for nbar_added >= 0.
- amplifier: phase-insensitive amplification (X = sqrt(gain),
  Y = (gain-1)(nbar+1/2) I on the listed modes), gain >= 1 (Caves 1982
  amplifier noise bound saturated at nbar = 0).

Conventions per docs/API.md: hbar = 1, block quadrature ordering
R = (x_1..x_N, p_1..p_N), vacuum covariance I/2, mode k -> rows (k, N+k).
"""

from __future__ import annotations

import numpy as np

from .gaussian import _quad_indices

__all__ = [
    "apply_channel",
    "loss",
    "thermal_noise",
    "amplifier",
    "loss_XY",
    "thermal_noise_XY",
    "amplifier_XY",
]


def apply_channel(V, X, Y):
    """Apply a Gaussian channel: V -> X V X^T + Y."""
    V = np.asarray(V, dtype=float)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    return X @ V @ X.T + Y


def loss_XY(N, modes, eta, nbar=0.0):
    """(X, Y) for the loss channel on `modes` of an N-mode system.

    X = sqrt(eta) on the mode quadratures (identity elsewhere),
    Y = (1 - eta)(nbar + 1/2) I on the mode quadratures.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta}")
    if nbar < 0.0:
        raise ValueError(f"nbar must be >= 0, got {nbar}")
    q = _quad_indices(N, modes)
    X = np.eye(2 * N)
    X[q, q] = np.sqrt(eta)
    Y = np.zeros((2 * N, 2 * N))
    Y[q, q] = (1.0 - eta) * (nbar + 0.5)
    return X, Y


def loss(V, modes, eta, nbar=0.0):
    """Beamsplitter loss of transmissivity eta into a thermal environment.

    The listed modes are mixed with independent thermal modes of occupation
    nbar; eta = 1 is the identity, eta = 0 replaces the modes by the
    environment state (nbar + 1/2) I.
    """
    N = np.asarray(V).shape[0] // 2
    X, Y = loss_XY(N, modes, eta, nbar)
    return apply_channel(V, X, Y)


def thermal_noise_XY(N, modes, nbar_added):
    """(X, Y) for classical Gaussian noise addition on `modes`."""
    if nbar_added < 0.0:
        raise ValueError(f"nbar_added must be >= 0, got {nbar_added}")
    q = _quad_indices(N, modes)
    X = np.eye(2 * N)
    Y = np.zeros((2 * N, 2 * N))
    Y[q, q] = nbar_added
    return X, Y


def thermal_noise(V, modes, nbar_added):
    """Add nbar_added of classical Gaussian noise to each quadrature variance."""
    N = np.asarray(V).shape[0] // 2
    X, Y = thermal_noise_XY(N, modes, nbar_added)
    return apply_channel(V, X, Y)


def amplifier_XY(N, modes, gain, nbar=0.0):
    """(X, Y) for the phase-insensitive amplifier on `modes`.

    X = sqrt(gain) on the mode quadratures, Y = (gain - 1)(nbar + 1/2) I —
    the minimal added noise at nbar = 0 (Caves, PRD 26, 1817 (1982)).
    """
    if gain < 1.0:
        raise ValueError(f"gain must be >= 1, got {gain}")
    if nbar < 0.0:
        raise ValueError(f"nbar must be >= 0, got {nbar}")
    q = _quad_indices(N, modes)
    X = np.eye(2 * N)
    X[q, q] = np.sqrt(gain)
    Y = np.zeros((2 * N, 2 * N))
    Y[q, q] = (gain - 1.0) * (nbar + 0.5)
    return X, Y


def amplifier(V, modes, gain, nbar=0.0):
    """Phase-insensitive amplification of the listed modes by `gain` >= 1."""
    N = np.asarray(V).shape[0] // 2
    X, Y = amplifier_XY(N, modes, gain, nbar)
    return apply_channel(V, X, Y)
