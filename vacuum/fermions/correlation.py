"""Free-fermion correlation-matrix methods (Peschel) for the Vacuum Program.

For a Gaussian (free) fermionic state, the reduced density matrix of any set
of sites A is fully determined by the two-point correlation matrix restricted
to A, C_A with (C_A)_ij = <c_i^dag c_j> for i, j in A. Its eigenvalues
zeta_k in [0, 1] are the occupations of the "entanglement modes", and

    rho_A = exp(-sum_ij h_ij c_i^dag c_j) / Z,   h = ln((1 - C_A) / C_A),

so the entanglement (modular) Hamiltonian is itself a free-fermion operator
(Peschel 2003). All entropies are in NATS (repo convention, docs/API.md).

Anchor references
-----------------
- I. Peschel, J. Phys. A 36, L205 (2003): reduced density matrices of free
  fermions from the correlation matrix; h = ln((1 - C)/C).
- I. Peschel and V. Eisler, J. Phys. A 42, 504003 (2009): review of
  correlation-matrix / entanglement-Hamiltonian methods.
- G. Vidal, J. I. Latorre, E. Rico, A. Kitaev, PRL 90, 227902 (2003) and
  P. Calabrese, J. Cardy, J. Stat. Mech. P06002 (2004): S(l) of a block in a
  critical chain, S = (c/3) ln[(N/pi) sin(pi l/N)] + c1 with c = 1 for the
  free fermion / XX chain.
- P. Calabrese, F. H. L. Essler, M. Fagotti and Calabrese et al.,
  PRL 104, 095701 (2010): parity-oscillating finite-size corrections
  ~ cos(2 k_F l) in critical-chain block entropies.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "tight_binding_C",
    "block_entropy",
    "entanglement_hamiltonian",
]

# Occupations are clamped into [_ZETA_CLIP, 1 - _ZETA_CLIP] before taking
# logs in entanglement_hamiltonian. 1e-14 sits just above the float64
# eigensolver noise floor for these O(1)-norm matrices and biases the
# reconstructed entropy by < 1e-12 per extremal mode.
_ZETA_CLIP = 1e-14


def _hopping_matrix(N, bc):
    """Single-particle matrix h of H = -(1/2) sum_i (c_i^dag c_{i+1} + h.c.).

    h[i, i+1] = h[i+1, i] = -1/2 on the open bonds; the boundary bond
    (N-1, 0) is -1/2 for 'periodic', +1/2 for 'antiperiodic' (c_N = -c_0),
    and absent for 'open'.
    """
    if N < 2:
        raise ValueError(f"need N >= 2 sites, got N={N}")
    h = np.zeros((N, N))
    idx = np.arange(N - 1)
    h[idx, idx + 1] = -0.5
    h[idx + 1, idx] = -0.5
    if bc == "periodic":
        h[N - 1, 0] += -0.5
        h[0, N - 1] += -0.5
    elif bc == "antiperiodic":
        h[N - 1, 0] += 0.5
        h[0, N - 1] += 0.5
    elif bc != "open":
        raise ValueError(
            f"bc must be 'periodic', 'open' or 'antiperiodic', got {bc!r}"
        )
    return h


def tight_binding_C(N, filling=0.5, bc="periodic"):
    """Ground-state correlation matrix C_ij = <c_i^dag c_j> of a hopping chain.

    Hamiltonian: H = -(1/2) sum_i (c_i^dag c_{i+1} + h.c.) with the boundary
    condition `bc` ('periodic' | 'open' | 'antiperiodic'). The ground state
    fills the n_f = round(filling * N) lowest single-particle modes, so
    C = sum_{k <= n_f} phi_k phi_k^T over occupied eigenvectors (Peschel and
    Eisler 2009, Sec. 2).

    Fermi-level degeneracies are handled *deliberately*: if the single-particle
    spectrum is degenerate across the Fermi level the ground state is not
    unique and C is ill-defined, so a ValueError is raised. At half filling
    the standard cures are bc='antiperiodic' with N = 0 mod 4, or
    bc='periodic' with N = 2 mod 4, or any open chain (spectrum
    -cos(q pi/(N+1)) is non-degenerate and gapped at half filling).

    Parameters
    ----------
    N : int
        Number of lattice sites (>= 2).
    filling : float
        Fraction of filled modes; n_f = round(filling * N).
    bc : str
        'periodic', 'open' or 'antiperiodic'.

    Returns
    -------
    (N, N) ndarray
        Real symmetric correlation matrix with eigenvalues in {0, 1} up to
        roundoff and trace n_f.
    """
    h = _hopping_matrix(N, bc)
    n_f = int(round(filling * N))
    if not 0 <= n_f <= N:
        raise ValueError(f"filling={filling} gives n_f={n_f} outside [0, {N}]")
    eps, phi = np.linalg.eigh(h)  # ascending eigenvalues, orthonormal columns
    if 0 < n_f < N:
        gap = eps[n_f] - eps[n_f - 1]
        if gap < 1e-8:
            raise ValueError(
                f"Fermi level is degenerate (gap={gap:.3e}) for N={N}, "
                f"filling={filling}, bc={bc!r}: ground state not unique. "
                "Choose N/bc so the single-particle spectrum is gapped at the "
                "Fermi level (e.g. antiperiodic with N = 0 mod 4, periodic "
                "with N = 2 mod 4, or an open chain, at half filling)."
            )
    occ = phi[:, :n_f]
    C = occ @ occ.T
    return 0.5 * (C + C.T)


def block_entropy(C, sites):
    """Von Neumann entropy (nats) of the block `sites` of a free-fermion state.

    S = -sum_k [zeta_k ln zeta_k + (1 - zeta_k) ln(1 - zeta_k)] over the
    eigenvalues zeta_k of the site-block C[sites, sites] (Peschel 2003;
    Peschel-Eisler 2009, Eq. (12)). Terms with zeta at 0 or 1 contribute 0
    and are guarded explicitly.

    Parameters
    ----------
    C : (N, N) ndarray
        Correlation matrix C_ij = <c_i^dag c_j> (hermitian).
    sites : sequence of int
        0-based site indices of the block A.

    Returns
    -------
    float
        Entanglement entropy of the block, in nats.
    """
    sites = np.asarray(sites, dtype=int)
    C_A = np.asarray(C)[np.ix_(sites, sites)]
    zeta = np.linalg.eigvalsh(0.5 * (C_A + C_A.conj().T))
    zeta = np.clip(zeta, 0.0, 1.0)
    # x ln x with the 0 ln 0 = 0 guard, applied to both zeta and 1 - zeta.
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -np.where(zeta > 0.0, zeta * np.log(zeta), 0.0)
        s -= np.where(zeta < 1.0, (1.0 - zeta) * np.log1p(-zeta), 0.0)
    return float(np.sum(s))


def entanglement_hamiltonian(C_A):
    """Single-particle entanglement Hamiltonian h = ln((1 - C_A)/C_A).

    The reduced density matrix of a free-fermion state on block A is
    rho_A = exp(-sum_ij h_ij c_i^dag c_j)/Z with h = ln((1 - C_A)/C_A)
    (Peschel 2003). Computed via the eigendecomposition of the hermitian
    C_A = U diag(zeta) U^dag, with occupations clamped into
    [1e-14, 1 - 1e-14] so the log is finite: eigenvalues of C_A that are
    exponentially close to 0/1 (generic for large blocks) map to large but
    finite single-particle energies eps = ln((1 - zeta)/zeta).

    Parameters
    ----------
    C_A : (l, l) ndarray
        Hermitian block correlation matrix (eigenvalues in [0, 1]).

    Returns
    -------
    (l, l) ndarray
        Hermitian matrix h; its eigenvalues eps_k give the occupations
        zeta_k = 1/(e^{eps_k} + 1) and the block entropy via the
        free-fermion formula S = sum_k [ln(1 + e^{-eps_k})
        + eps_k/(e^{eps_k} + 1)].
    """
    C_A = np.asarray(C_A)
    C_A = 0.5 * (C_A + C_A.conj().T)
    zeta, U = np.linalg.eigh(C_A)
    zeta = np.clip(zeta, _ZETA_CLIP, 1.0 - _ZETA_CLIP)
    eps = np.log1p(-zeta) - np.log(zeta)  # ln((1 - zeta)/zeta)
    h = (U * eps) @ U.conj().T
    return 0.5 * (h + h.conj().T)
