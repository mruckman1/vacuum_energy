"""Exact free-fermion (Majorana) reference for the transverse-field Ising /
Kitaev chain: block entropies and mutual information of Gaussian fermionic
vacua, on the infinite chain and on finite (possibly disordered) chains.

Model
-----
    H = - sum_j J_j X_j X_{j+1} - sum_j g_j Z_j

Jordan-Wigner with a_{2j} = (prod_{l<j} Z_l) X_j, a_{2j+1} = (prod_{l<j} Z_l)
Y_j gives Z_j = -i a_{2j} a_{2j+1}, X_j X_{j+1} = -i a_{2j+1} a_{2j+2}, so

    H = (i/4) a^T A a,   A_{2j,2j+1} = 2 g_j,  A_{2j+1,2j+2} = 2 J_j  (antisym.)

The ground-state Majorana covariance Gamma_mn = (i/2)<[a_m, a_n]> is
Gamma = i sgn(iA) (all negative-energy modes filled). The fermionic state is
Gaussian: the entropy of any set of sites S follows from the eigenvalues
+-nu_k of i Gamma_S, S = sum_k H2((1 + nu_k)/2) in nats (Vidal, Latorre,
Rico, Kitaev, PRL 90, 227902 (2003), Eqs. 11-13; Peschel-Eisler 2009).

For a CONTIGUOUS block the spin and fermion entropies coincide (the
Jordan-Wigner string stays inside the block); for disjoint blocks the
fermionic mutual information computed here is that of the Kitaev-chain
Gaussian vacuum, the fermionic dual of the TFIM -- stated wherever used.

Infinite chain (translation invariant, g_j = g, J_j = 1): the only nonzero
correlators couple a_{2j} to a_{2l+1},

    G_r = <i a_{2j} a_{2(j+r)+1}> -> g_r
        = (1/pi) int_0^pi dphi [cos((r+1)phi) - g cos(r phi)] / sqrt(1 + g^2 - 2 g cos phi)

(VLRK 2003 Eq. 9 with gamma = 1, lambda = g, up to the sign/orientation
convention that leaves every entropy invariant). At criticality the
integral closes: g_r = -2 / (pi (2r + 1)); in particular <Z> = 2/pi
(Pfeuty 1970). Off criticality the correlations decay with correlation
length xi = 1/|ln g| (Pfeuty 1970).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "tfim_majorana_g",
    "tfim_block_G",
    "tfim_block_entropy",
    "tfim_mutual_information",
    "tfim_correlation_length",
    "kitaev_chain_gamma",
    "majorana_block_entropy",
    "majorana_mutual_information",
    "E0_TFIM",
]


def E0_TFIM(g):
    """Ground energy per site of H = -sum XX - g sum Z (Pfeuty 1970):
    -(1/pi) int_0^pi dk sqrt(1 + g^2 - 2 g cos k)."""
    from scipy.integrate import quad

    return -quad(lambda k: np.sqrt(1.0 + g * g - 2.0 * g * np.cos(k)), 0.0, np.pi)[0] / np.pi


def _h2(p):
    """Binary entropy in nats with the 0 ln 0 = 0 guard, vectorized."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    out = np.zeros_like(p)
    m = (p > 0.0) & (p < 1.0)
    out[m] = -p[m] * np.log(p[m]) - (1.0 - p[m]) * np.log1p(-p[m])
    return out


def tfim_majorana_g(rs, g=1.0, n_grid=2**15):
    """Infinite-chain Majorana correlator g_r for displacements rs (ints).

    Trapezoid rule on n_grid points over [0, pi]; the denominator is
    evaluated as sqrt((1 - g)^2 + 4 g sin^2(phi/2)) (no cancellation near
    g = 1). For |g - 1| < 1e-6 the critical closed form
    g_r = -2/(pi (2r + 1)) is used (the difference is O(|g - 1| ln|g - 1|)).
    Off criticality the integrand is analytic and periodic, so the rule
    converges geometrically once the peak of width |1 - g| at phi = 0 is
    resolved (n_grid = 2^15 resolves |1 - g| >~ 1e-3 to ~1e-8).
    """
    rs = np.atleast_1d(np.asarray(rs, dtype=int))
    if abs(g - 1.0) < 1e-6:
        return -2.0 / (np.pi * (2.0 * rs + 1.0))
    phi = np.linspace(0.0, np.pi, n_grid + 1)
    D = np.sqrt((1.0 - g) ** 2 + 4.0 * g * np.sin(0.5 * phi) ** 2)
    w = np.full_like(phi, np.pi / n_grid)
    w[0] *= 0.5
    w[-1] *= 0.5
    out = np.empty(rs.shape, dtype=float)
    for k, r in enumerate(rs):
        f = (np.cos((r + 1) * phi) - g * np.cos(r * phi)) / D
        out[k] = np.sum(w * f) / np.pi
    return out


def tfim_block_G(sites, g=1.0):
    """The matrix G[j, l] = g_{s_l - s_j} = <i a_{2 s_j} a_{2 s_l + 1}> of
    correlators between the even Majorana of site s_j and the odd Majorana
    of site s_l (orientation fixed against kitaev_chain_gamma; every
    entropy is invariant under G -> G^T, G -> -G)."""
    sites = np.asarray(sites, dtype=int)
    diffs = sites[None, :] - sites[:, None]   # G[j, l] = g_{s_l - s_j}
    uniq = np.unique(diffs)
    vals = tfim_majorana_g(uniq, g)
    lookup = dict(zip(uniq.tolist(), vals.tolist()))
    return np.vectorize(lookup.get)(diffs).astype(float)


def _entropy_from_G(G):
    nu = np.linalg.svd(G, compute_uv=False)
    nu = np.clip(nu, 0.0, 1.0)
    return float(np.sum(_h2(0.5 * (1.0 + nu))))


def tfim_block_entropy(sites, g=1.0):
    """Entropy (nats) of the sites ``sites`` of the infinite TFIM/Kitaev
    chain (contiguous or not; for contiguous blocks equals the spin-chain
    block entropy). i Gamma_S has eigenvalues +- the singular values of
    G_S."""
    return _entropy_from_G(tfim_block_G(sites, g))


def tfim_mutual_information(A, B, g=1.0):
    """Fermionic-Gaussian mutual information I(A:B) (nats) on the infinite
    chain."""
    A, B = list(A), list(B)
    if set(A) & set(B):
        raise ValueError("regions must be disjoint")
    return (tfim_block_entropy(A, g) + tfim_block_entropy(B, g)
            - tfim_block_entropy(A + B, g))


def tfim_correlation_length(g):
    """xi = 1/|ln g| (Pfeuty 1970); infinite at g = 1."""
    g = float(g)
    if abs(g - 1.0) < 1e-14:
        return np.inf
    return 1.0 / abs(np.log(g))


# ---------------------------------------------------------------------------
# finite chains (open), site-dependent couplings -- the disorder families
# ---------------------------------------------------------------------------


def kitaev_chain_gamma(N, g, J=1.0, bc="open"):
    """Ground-state Majorana covariance Gamma (2N x 2N, real antisymmetric)
    of H = -sum_j J_j X_j X_{j+1} - sum_j g_j Z_j; g and J may be arrays
    (J has N-1 entries for 'open', N for 'periodic' -- the periodic case
    uses the fermionic (Kitaev) boundary term, i.e. the even-parity sector
    is not enforced)."""
    N = int(N)
    g = np.broadcast_to(np.asarray(g, dtype=float), (N,))
    nb = N - 1 if bc == "open" else N
    J = np.broadcast_to(np.asarray(J, dtype=float), (nb,))
    A = np.zeros((2 * N, 2 * N))
    for j in range(N):
        A[2 * j, 2 * j + 1] = 2.0 * g[j]
        A[2 * j + 1, 2 * j] = -2.0 * g[j]
    for j in range(nb):
        a, b = 2 * j + 1, (2 * j + 2) % (2 * N)
        A[a, b] += 2.0 * J[j]
        A[b, a] += -2.0 * J[j]
    lam, U = np.linalg.eigh(1j * A)
    # (near-)zero modes -- e.g. Majorana end modes of a strongly disordered
    # critical chain -- are filled with a fixed sign convention (+), which
    # picks one of the near-degenerate ground states deterministically.
    sign = np.where(lam >= 0.0, 1.0, -1.0)
    sgn = (U * sign) @ U.conj().T
    Gamma = np.real(1j * sgn)
    return 0.5 * (Gamma - Gamma.T)


def majorana_block_entropy(Gamma, sites):
    """Entropy (nats) of the sites in ``sites`` from the Majorana covariance."""
    sites = np.asarray(sites, dtype=int)
    idx = np.concatenate([2 * sites, 2 * sites + 1])
    idx.sort()
    sub = Gamma[np.ix_(idx, idx)]
    ev = np.linalg.eigvalsh(1j * sub)
    nu = np.clip(np.abs(ev), 0.0, 1.0)
    # eigenvalues come in +-nu pairs: count each pair once
    return float(0.5 * np.sum(_h2(0.5 * (1.0 + nu))))


def majorana_mutual_information(Gamma, A, B):
    A, B = list(A), list(B)
    if set(A) & set(B):
        raise ValueError("regions must be disjoint")
    return (majorana_block_entropy(Gamma, A) + majorana_block_entropy(Gamma, B)
            - majorana_block_entropy(Gamma, A + B))
