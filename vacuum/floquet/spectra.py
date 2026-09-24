"""Photon-pair spectra, squeezing spectra and pair negativity (Layer 5).

Observables of a driven lattice-field covariance V (block ordering,
hbar = 1, V_vac = I/2) read in the normal modes of the *undriven* K0:

* :func:`normal_modes` — K0 = U diag(omega_k^2) U^T, ascending omega_k;
* :func:`mode_occupations` — <n_k> = (omega_k <x_k^2> + <p_k^2>/omega_k - 1)/2
  in the normal-mode quadratures (Serafini 2017, Sec. 3.2: single-mode
  moments); zero on the K0 vacuum, so every photon counted here was
  produced by the drive (the ledger charges omega_k <n_k> to it);
* :func:`traveling_wave_basis` — degenerate standing-wave pairs (cos k x,
  sin k x) of a translation-invariant chain are recombined into the
  traveling modes a_{+-k} = (a_c -/+ i a_s)/sqrt2, a *real symplectic*
  change of quadratures
      x_k = (x_c - p_s)/sqrt2,  p_k = (p_c + x_s)/sqrt2,
      x_-k = (x_c + p_s)/sqrt2, p_-k = (p_c - x_s)/sqrt2,
  in which a uniformly modulated chain produces exactly the two-mode
  squeezed (k, -k) pairs of the dynamical Casimir effect / photonic time
  crystal (Lustig et al. 2023, Sec. 2; Wilson et al. 2011, "photons will be
  produced in pairs");
* :func:`squeezing_spectrum` — r_k = arcsinh(sqrt <n_k>), the squeezing
  parameter a two-mode squeezed vacuum with that occupation would have
  (<n> = sinh^2 r; Serafini 2017, Sec. 5.1.3 — also the single-mode
  squeezed-vacuum relation for self-conjugate modes k = 0, pi);
* :func:`pair_negativity` — E_N of the (k, -k) block in the traveling-wave
  quadratures.  For a pure two-mode squeezed vacuum the partial-transpose
  symplectic eigenvalues are e^{-/+2r}/2, hence E_N = 2 r exactly
  (Vidal-Werner PRA 65, 032314 (2002); Adesso-Serafini-Illuminati, PRA 70,
  022318 (2004), Eq. (20)) — the anchor
  ``E_N(k,-k) = 2 arcsinh(sqrt<n_k>)`` tested to 1e-8 in
  tests/test_floquet_pairs.py.  The mp-margin rule of the Layer 2 stack
  (partial-transpose spectrum within 1e-10 of the floor -> mpmath margins)
  is applied here too.

Pure-functional over arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from vacuum.core import (
    log_negativity,
    partial_transpose,
    reduce as reduce_cov,
    symplectic_eigenvalues,
    symplectic_margins_mp,
)

__all__ = [
    "normal_modes",
    "normal_mode_symplectic",
    "mode_occupations",
    "TravelingWaveBasis",
    "traveling_wave_basis",
    "to_traveling_waves",
    "pair_spectrum",
    "squeezing_spectrum",
    "squeezing_from_occupation",
    "pair_negativity",
    "EntanglementGrowth",
    "entanglement_growth",
    "power_iterate",
    "FLOOR_TOL",
]

VACUUM_NU = 0.5
FLOOR_TOL = 1e-10


# --------------------------------------------------------------------------
# Normal modes
# --------------------------------------------------------------------------


def normal_modes(K0):
    """(omega_k, U): K0 = U diag(omega_k^2) U^T with omega_k ascending."""
    K0 = np.asarray(K0, dtype=float)
    w2, U = np.linalg.eigh(0.5 * (K0 + K0.T))
    if np.min(w2) <= 0.0:
        raise ValueError(f"K0 must be positive definite: min eig {np.min(w2):.3e}")
    return np.sqrt(w2), U


def normal_mode_symplectic(U):
    """Block-diagonal symplectic blockdiag(U^T, U^T): (x, p) -> (U^T x, U^T p)."""
    U = np.asarray(U, dtype=float)
    N = U.shape[0]
    T = np.zeros((2 * N, 2 * N))
    T[:N, :N] = U.T
    T[N:, N:] = U.T
    return T


def mode_occupations(V, K0=None, *, omega=None, U=None):
    """<n_k> in the normal modes of K0 (ascending omega_k), mean-zero state."""
    V = np.asarray(V, dtype=float)
    if omega is None or U is None:
        omega, U = normal_modes(K0)
    N = omega.size
    Vxx = V[:N, :N]
    Vpp = V[N:, N:]
    xx = np.einsum("ik,ij,jk->k", U, Vxx, U)
    pp = np.einsum("ik,ij,jk->k", U, Vpp, U)
    return 0.5 * (omega * xx + pp / omega - 1.0)


def power_iterate(V0, S, n):
    """S^n V0 (S^n)^T by repeated application (n >= 0)."""
    V = np.array(V0, dtype=float, copy=True)
    S = np.asarray(S, dtype=float)
    for _ in range(int(n)):
        V = S @ V @ S.T
    return V


def pair_spectrum(V0, S_F, n_periods, K0):
    """<n_k> after n periods (README API: ``pair_spectrum``).

    Returns the occupations of the normal modes of K0 (ascending omega_k)
    of S_F^n V0 (S_F^n)^T.
    """
    V = power_iterate(V0, S_F, n_periods)
    return mode_occupations(V, K0)


# --------------------------------------------------------------------------
# Traveling-wave (k, -k) basis
# --------------------------------------------------------------------------


@dataclass
class TravelingWaveBasis:
    """Symplectic T to traveling-wave quadratures, with the pair structure.

    ``T`` maps block-ordered lattice quadratures to block-ordered mode
    quadratures (mode order = ascending omega; degenerate pairs are kept
    adjacent).  ``pairs`` lists (i, j) mode indices of (k, -k) partners,
    ``singles`` the self-conjugate modes.  ``omega`` is per mode.
    """

    T: np.ndarray
    omega: np.ndarray
    pairs: Tuple[Tuple[int, int], ...]
    singles: Tuple[int, ...]
    k: Optional[np.ndarray] = None

    @property
    def N(self) -> int:
        return int(self.omega.size)

    @property
    def pair_omega(self) -> np.ndarray:
        return np.array([self.omega[i] for i, _ in self.pairs])


def traveling_wave_basis(K0, rtol=1e-8, mass=None):
    """Recombine degenerate normal-mode pairs into (k, -k) traveling modes.

    Consecutive normal modes with |omega_i - omega_j| <= rtol * omega are
    treated as a degenerate standing-wave pair and rotated into the
    traveling combination (module docstring); other modes stay as they are.
    Any orthonormal basis of the degenerate subspace gives the same
    traveling modes up to a local phase, so eigh's arbitrary choice is
    harmless for occupations and negativities.
    """
    omega, U = normal_modes(K0)
    N = omega.size
    T = normal_mode_symplectic(U)
    pairs: List[Tuple[int, int]] = []
    singles: List[int] = []
    R = np.eye(2 * N)
    s = 1.0 / np.sqrt(2.0)
    i = 0
    while i < N:
        if i + 1 < N and abs(omega[i + 1] - omega[i]) <= rtol * omega[i]:
            a, b = i, i + 1
            w = float(omega[i])
            # In the mode's dimensionless quadratures (sqrt(w) x, p/sqrt(w)):
            #   x~_k = (x~_c - p~_s)/sqrt2,  p~_k = (p~_c + x~_s)/sqrt2,
            #   x~_-k = (x~_c + p~_s)/sqrt2, p~_-k = (p~_c - x~_s)/sqrt2,
            # i.e. in the raw quadratures x_k = (x_c - p_s/w)/sqrt2,
            # p_k = (p_c + w x_s)/sqrt2 (and the -k partner with the signs
            # flipped) — symplectic for every w, [x_k, p_k] = i.
            R[a, a] = s
            R[a, N + b] = -s / w
            R[a, b] = 0.0
            R[b, a] = s
            R[b, N + b] = s / w
            R[b, b] = 0.0
            R[N + a, N + a] = s
            R[N + a, b] = s * w
            R[N + a, N + b] = 0.0
            R[N + b, N + a] = s
            R[N + b, b] = -s * w
            R[N + b, N + b] = 0.0
            pairs.append((a, b))
            i += 2
        else:
            singles.append(i)
            i += 1
    k = None
    if mass is not None:
        arg = np.sqrt(np.clip((omega**2 - float(mass) ** 2) / 4.0, 0.0, 1.0))
        k = 2.0 * np.arcsin(arg)
    return TravelingWaveBasis(T=R @ T, omega=omega, pairs=tuple(pairs),
                              singles=tuple(singles), k=k)


def to_traveling_waves(V, basis: TravelingWaveBasis):
    """V in the traveling-wave quadratures: T V T^T."""
    V = np.asarray(V, dtype=float)
    return basis.T @ V @ basis.T.T


# --------------------------------------------------------------------------
# Squeezing spectrum and pair negativity
# --------------------------------------------------------------------------


def squeezing_from_occupation(n):
    """r = arcsinh(sqrt(n)) (two-mode squeezed vacuum: <n> = sinh^2 r)."""
    n = np.asarray(n, dtype=float)
    return np.arcsinh(np.sqrt(np.clip(n, 0.0, None)))


def squeezing_spectrum(V, K0=None, *, basis: Optional[TravelingWaveBasis] = None):
    """Per-mode squeezing parameters r_k (README API: ``squeezing_spectrum``).

    r_k = arcsinh(sqrt <n_k>) with <n_k> the normal-mode occupations of K0
    (or of ``basis`` if given).  For a (k, -k) two-mode squeezed vacuum both
    partners carry r, and :func:`pair_negativity` returns 2 r.
    """
    if basis is not None:
        Vt = to_traveling_waves(V, basis)
        N = basis.N
        n = 0.5 * (
            basis.omega * np.diag(Vt[:N, :N]) + np.diag(Vt[N:, N:]) / basis.omega - 1.0
        )
    else:
        n = mode_occupations(V, K0)
    return squeezing_from_occupation(n)


def _mp_safe_log_negativity(V2, floor_tol=FLOOR_TOL, dps=50):
    """E_N of a two-mode block (mode 0 | mode 1) with the mp-margin rule."""
    V_pt = partial_transpose(V2, [1])
    nu = symplectic_eigenvalues(V_pt)
    min_nu = float(np.min(nu))
    if abs(min_nu - VACUUM_NU) < floor_tol:
        margins = symplectic_margins_mp(V_pt, dps=dps)
        E = float(np.sum(np.where(margins < 0.0, -np.log1p(2.0 * margins), 0.0)))
        return E, True
    return float(np.sum(np.maximum(0.0, -np.log(2.0 * nu)))), False


def pair_negativity(V, k, K0=None, *, basis: Optional[TravelingWaveBasis] = None,
                    floor_tol=FLOOR_TOL, dps=50, with_flag=False):
    """E_N (nats) of the (k, -k) block (README API: ``pair_negativity``).

    ``k`` indexes ``basis.pairs`` (build the basis with
    :func:`traveling_wave_basis`; passing ``K0`` builds it here).  With
    ``with_flag=True`` returns ``(E_N, flagged)`` where ``flagged`` marks
    the mp-margin escalation.
    """
    if basis is None:
        if K0 is None:
            raise ValueError("pass K0 or a TravelingWaveBasis")
        basis = traveling_wave_basis(K0)
    i, j = basis.pairs[int(k)]
    Vt = to_traveling_waves(V, basis)
    V2 = reduce_cov(Vt, [i, j])
    E, flagged = _mp_safe_log_negativity(V2, floor_tol, dps)
    return (E, flagged) if with_flag else E


@dataclass
class EntanglementGrowth:
    """E_N and r per (k, -k) pair after n = 0..n_periods periods."""

    n: np.ndarray
    E_N: np.ndarray  # (n_periods + 1, n_pairs)
    r: np.ndarray  # (n_periods + 1, n_pairs)
    pair_omega: np.ndarray
    pairs: Tuple[Tuple[int, int], ...]
    flagged: np.ndarray  # (n_periods + 1, n_pairs) bool
    occupations: np.ndarray  # (n_periods + 1, N) all modes


def entanglement_growth(V0, S_F, n_periods, K0, *, basis=None, pairs=None,
                        floor_tol=FLOOR_TOL, dps=50):
    """E_N(k,-k) vs period count (README API: ``entanglement_growth``)."""
    if basis is None:
        basis = traveling_wave_basis(K0)
    pair_ids = list(range(len(basis.pairs))) if pairs is None else [int(p) for p in pairs]
    n_periods = int(n_periods)
    V = np.array(V0, dtype=float, copy=True)
    S_F = np.asarray(S_F, dtype=float)
    N = basis.N
    E = np.zeros((n_periods + 1, len(pair_ids)))
    R = np.zeros_like(E)
    F = np.zeros(E.shape, dtype=bool)
    occ = np.zeros((n_periods + 1, N))
    for step in range(n_periods + 1):
        if step > 0:
            V = S_F @ V @ S_F.T
        Vt = to_traveling_waves(V, basis)
        n_k = 0.5 * (basis.omega * np.diag(Vt[:N, :N]) + np.diag(Vt[N:, N:]) / basis.omega - 1.0)
        occ[step] = n_k
        for c, p in enumerate(pair_ids):
            i, j = basis.pairs[p]
            e, fl = _mp_safe_log_negativity(reduce_cov(Vt, [i, j]), floor_tol, dps)
            E[step, c] = e
            F[step, c] = fl
            R[step, c] = float(squeezing_from_occupation(0.5 * (n_k[i] + n_k[j])))
    return EntanglementGrowth(
        n=np.arange(n_periods + 1), E_N=E, r=R,
        pair_omega=np.array([basis.omega[basis.pairs[p][0]] for p in pair_ids]),
        pairs=tuple(basis.pairs[p] for p in pair_ids), flagged=F, occupations=occ,
    )
