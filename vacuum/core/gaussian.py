"""Gaussian-state linear algebra for the Vacuum Program (Layer 0/1).

Conventions (see docs/API.md — non-negotiable):

- hbar = 1. The single-mode vacuum covariance is V = I/2; physical states have
  symplectic eigenvalues nu >= 1/2 (nu = 1/2 is the vacuum floor).
- Quadrature ordering is *block* form: R = (x_1..x_N, p_1..p_N), so the
  symplectic form is Omega = [[0, I_N], [-I_N, 0]] and [R_i, R_j] = i Omega_ij.
- Covariance: V_ij = (1/2) <{R_i - <R_i>, R_j - <R_j>}>.
- All entropies and negativities are in NATS.
- Symplectic matrices satisfy S Omega S^T = Omega; states evolve V -> S V S^T.
- Mode index sets are sequences of 0-based *mode* indices; mode k maps to
  quadrature rows (k, N + k).

Anchor references
-----------------
- J. Williamson, Am. J. Math. 58, 141 (1936): normal form of positive-definite
  quadratic forms under symplectic transformations.
- R. Simon, S. Chaturvedi, V. Srinivasan, J. Math. Phys. 40, 3632 (1999):
  congruences and the Williamson normal form.
- G. Vidal and R. F. Werner, Phys. Rev. A 65, 032314 (2002); M. B. Plenio,
  Phys. Rev. Lett. 95, 090503 (2005): logarithmic negativity.
- A. Serafini, "Quantum Continuous Variables" (CRC Press, 2017): Gaussian
  state formulary (entropy of a Gaussian state, partial transposition as a
  momentum sign flip, symplectic evolution exp(t Omega H)).
- M. Srednicki, Phys. Rev. Lett. 71, 666 (1993): ground-state covariance of
  H = (1/2)(p.p + x.K.x) from the matrix square root of K.
- A. Botero and B. Reznik, Phys. Rev. A 70, 052329 (2004); J. Eisert,
  M. Cramer, M. B. Plenio, Rev. Mod. Phys. 82, 277 (2010): Gaussian
  entanglement (modular) Hamiltonians via the Williamson decomposition.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import schur

__all__ = [
    "Omega",
    "ground_state_cov",
    "thermal_state_cov",
    "symplectic_eigenvalues",
    "williamson",
    "entropy",
    "entropy_from_nu",
    "reduce",
    "partial_transpose",
    "log_negativity",
    "mutual_information",
    "entanglement_hamiltonian",
    "modular_energy",
    "symplectic_from_quadratic",
    "evolve",
    "mean_energy",
    "symplectic_inverse",
]

VACUUM_NU = 0.5  # symplectic eigenvalue of the vacuum, hbar = 1


def Omega(N):
    """Symplectic form for N modes in block ordering R = (x_1..x_N, p_1..p_N).

    Omega = [[0, I_N], [-I_N, 0]], so [R_i, R_j] = i Omega_ij with hbar = 1.
    """
    I = np.eye(N)
    Z = np.zeros((N, N))
    return np.block([[Z, I], [-I, Z]])


def _sym(V):
    """Symmetrize (defensive against roundoff asymmetry)."""
    V = np.asarray(V, dtype=float)
    return 0.5 * (V + V.T)


def _psd_sqrt_and_invsqrt(K, with_inverse=True):
    """Square root and inverse square root of a symmetric PSD matrix via eigh.

    eigh is preferred over scipy.linalg.sqrtm for stability on symmetric input.
    Tiny negative eigenvalues from roundoff are clipped to zero (and would make
    the inverse root blow up loudly, as they should for a singular K).

    The inverse root is computed only when asked for (``with_inverse=False``
    returns ``(root, None)``): it is the sole reason this helper needs K to be
    *definite* rather than merely semi-definite, and callers that never use it
    — :func:`symplectic_eigenvalues` above all — must stay usable on the
    PSD-singular quadratic forms that show up all over Layers 2-3 (a sampling
    operator with unsampled modes, a reduced covariance of a pure state at the
    vacuum floor).  Splitting the computation is the whole fix; nothing about
    the definite path changes.
    """
    K = _sym(K)
    w, U = np.linalg.eigh(K)
    if np.min(w) < -1e-10 * max(1.0, np.max(np.abs(w))):
        raise ValueError(f"matrix is not PSD: min eigenvalue {np.min(w):.3e}")
    w = np.clip(w, 0.0, None)
    sw = np.sqrt(w)
    root = _sym((U * sw) @ U.T)
    if not with_inverse:
        return root, None
    with np.errstate(divide="raise"):
        inv_root = (U * (1.0 / sw)) @ U.T
    return root, _sym(inv_root)


def ground_state_cov(K):
    """Ground-state covariance of H = (1/2)(p.p + x.K.x).

    V = blockdiag(K^{-1/2}/2, K^{1/2}/2) in block quadrature ordering
    (Srednicki, PRL 71, 666 (1993)). K must be symmetric positive definite.
    """
    root, inv_root = _psd_sqrt_and_invsqrt(K)
    N = root.shape[0]
    V = np.zeros((2 * N, 2 * N))
    V[:N, :N] = 0.5 * inv_root
    V[N:, N:] = 0.5 * root
    return V


def thermal_state_cov(K, T):
    """Thermal (Gibbs) covariance of H = (1/2)(p.p + x.K.x) at temperature T.

    With the eigendecomposition K = U diag(omega^2) U^T (omega > 0), the
    thermal state at temperature T (hbar = k_B = 1) has

        V_xx = (1/2) U diag(coth(omega/2T)/omega) U^T,
        V_pp = (1/2) U diag(omega * coth(omega/2T)) U^T,

    i.e. each normal mode carries symplectic eigenvalue
    nu = (1/2) coth(omega/2T) = 1/2 + nbar with the Bose occupation
    nbar = 1/(e^{omega/T} - 1) (Serafini, "Quantum Continuous Variables",
    2017, Sec. 3.3: thermal Gaussian states). T = 0 is accepted and taken
    via the coth limit coth(x) -> 1 as x -> +inf, reproducing
    :func:`ground_state_cov` exactly with no division by zero.

    Mean energy check: mean_energy(thermal_state_cov(K, T), K)
    = sum_k (omega_k/2) coth(omega_k/2T), the analytic Bose sum.
    """
    T = float(T)
    if T < 0.0:
        raise ValueError(f"temperature must be >= 0, got {T}")
    K = _sym(K)
    w2, U = np.linalg.eigh(K)
    if np.min(w2) <= 0:
        raise ValueError(
            f"K must be positive definite for a thermal state: min eig {np.min(w2):.3e}"
        )
    omega = np.sqrt(w2)
    if T == 0.0:
        coth = np.ones_like(omega)  # coth(omega/2T) -> 1 as T -> 0+
    else:
        # tanh saturates to 1.0 for large arguments, so this is overflow-safe
        # at any T > 0 (small-T limit smoothly matches the T = 0 branch).
        coth = 1.0 / np.tanh(omega / (2.0 * T))
    N = omega.size
    V = np.zeros((2 * N, 2 * N))
    V[:N, :N] = 0.5 * _sym((U * (coth / omega)) @ U.T)
    V[N:, N:] = 0.5 * _sym((U * (coth * omega)) @ U.T)
    return V


def symplectic_eigenvalues(V):
    """Symplectic eigenvalues of a symmetric positive-*semi*-definite form V.

    Returns the N values nu_k = |spec(i Omega V)|, descending. Physical states
    have nu_k >= 1/2; partial transposes of entangled states dip below.

    Computed stably as the singular values of the antisymmetric matrix
    A = V^{1/2} Omega V^{1/2} (each nu appears twice; A is similar to Omega V).
    Only the *forward* root is needed, so a singular V is fine and returns
    zeros for the unsupported directions — which is what a sampling operator
    with unsampled modes (Layer 3, M3.4) genuinely has.
    """
    V = _sym(V)
    N = V.shape[0] // 2
    root, _ = _psd_sqrt_and_invsqrt(V, with_inverse=False)
    A = root @ Omega(N) @ root
    s = np.linalg.svd(0.5 * (A - A.T), compute_uv=False)  # [nu1,nu1,nu2,nu2,..]
    return np.sort(s)[::-1][::2].copy()


def _interleave_permutation(N):
    """Pi with z_interleaved = Pi z_block: Pi[2k, k] = Pi[2k+1, N+k] = 1."""
    Pi = np.zeros((2 * N, 2 * N))
    for k in range(N):
        Pi[2 * k, k] = 1.0
        Pi[2 * k + 1, N + k] = 1.0
    return Pi


def williamson(V):
    """Williamson normal form of a symmetric positive-definite V.

    Returns (S, nu) with V = S diag(nu, nu) S^T and S Omega S^T = Omega,
    nu descending (Williamson 1936; algorithm via the real Schur form of
    the antisymmetric matrix V^{-1/2} Omega V^{-1/2}, cf. Simon-Chaturvedi-
    Srinivasan 1999 and Serafini 2017, Sec. 3.2.3).
    """
    V = _sym(V)
    N = V.shape[0] // 2
    root, inv_root = _psd_sqrt_and_invsqrt(V)

    A = inv_root @ Omega(N) @ inv_root
    A = 0.5 * (A - A.T)  # exact antisymmetry
    T, Q = schur(A, output="real")

    # A is normal, so the real Schur form is block diagonal with 2x2
    # antisymmetric blocks [[0, lam], [-lam, 0]]; lam_k = 1/nu_k.
    lam = np.empty(N)
    for k in range(N):
        i = 2 * k
        lam_k = 0.5 * (T[i, i + 1] - T[i + 1, i])
        if lam_k < 0:  # flip block orientation: swap the two Schur vectors
            Q[:, [i, i + 1]] = Q[:, [i + 1, i]]
            lam_k = -lam_k
        lam[k] = lam_k

    nu = 1.0 / lam
    order = np.argsort(-nu)  # descending nu
    nu = nu[order]
    lam = lam[order]

    cols = np.empty(2 * N, dtype=int)
    cols[0::2] = 2 * order
    cols[1::2] = 2 * order + 1
    Qs = Q[:, cols]

    # S = V^{1/2} Q diag(sqrt(lam) twice, interleaved) Pi  (block ordering out)
    Chalf = np.repeat(np.sqrt(lam), 2)
    Pi = _interleave_permutation(N)
    S = root @ (Qs * Chalf) @ Pi
    return S, nu


def symplectic_inverse(S):
    """Inverse of a symplectic matrix: S^{-1} = -Omega S^T Omega (exact form)."""
    S = np.asarray(S, dtype=float)
    N = S.shape[0] // 2
    Om = Omega(N)
    return -Om @ S.T @ Om


def entropy_from_nu(nu):
    """Von Neumann entropy (nats) from symplectic eigenvalues.

    S = sum_k [(nu+1/2) ln(nu+1/2) - (nu-1/2) ln(nu-1/2)], exactly 0 at
    nu = 1/2. Eigenvalues are clamped up to the vacuum floor first, so tiny
    roundoff below 1/2 cannot produce NaNs or negative entropy.
    """
    nu = np.maximum(np.asarray(nu, dtype=float), VACUUM_NU)
    x = nu + 0.5
    y = nu - 0.5
    out = x * np.log(x)
    pos = y > 0.0
    out[pos] -= y[pos] * np.log(y[pos])  # y*ln(y) -> 0 as y -> 0+
    return float(np.sum(out))


def entropy(V):
    """Von Neumann entropy (nats) of the Gaussian state with covariance V."""
    return entropy_from_nu(symplectic_eigenvalues(V))


def _quad_indices(N, modes):
    """Map 0-based mode indices to quadrature row indices (k, N+k)."""
    m = np.asarray(list(modes), dtype=int)
    if m.size and (m.min() < 0 or m.max() >= N):
        raise IndexError(f"mode index out of range for N={N}: {m}")
    return np.concatenate([m, m + N])


def reduce(V, modes):
    """Reduced covariance matrix of the listed modes (order preserved)."""
    V = np.asarray(V, dtype=float)
    N = V.shape[0] // 2
    idx = _quad_indices(N, modes)
    return V[np.ix_(idx, idx)].copy()


def partial_transpose(V, modes_B):
    """Partial transpose on modes_B: flip the sign of their momenta.

    Simon's criterion / Vidal-Werner: V -> P V P with P = diag(1,..,-1 on p_B).
    """
    V = np.asarray(V, dtype=float).copy()
    N = V.shape[0] // 2
    p = np.ones(2 * N)
    for m in modes_B:
        p[N + int(m)] = -1.0
    return (V * p).T * p  # P V P with diagonal P


def log_negativity(V, modes_A, modes_B):
    """Logarithmic negativity (nats) between mode sets A and B.

    E_N = sum_k max(0, -ln(2 nu~_k)) over the symplectic eigenvalues nu~ of
    the partial transpose (on B) of the A-union-B reduced state
    (Vidal-Werner PRA 65, 032314 (2002), natural logs).
    """
    modes_A, modes_B = list(modes_A), list(modes_B)
    if set(modes_A) & set(modes_B):
        raise ValueError("modes_A and modes_B must be disjoint")
    VAB = reduce(V, modes_A + modes_B)
    B_local = range(len(modes_A), len(modes_A) + len(modes_B))
    nu_t = symplectic_eigenvalues(partial_transpose(VAB, B_local))
    return float(np.sum(np.maximum(0.0, -np.log(2.0 * nu_t))))


def mutual_information(V, modes_A, modes_B):
    """Mutual information I(A:B) = S(A) + S(B) - S(AB), in nats."""
    modes_A, modes_B = list(modes_A), list(modes_B)
    if set(modes_A) & set(modes_B):
        raise ValueError("modes_A and modes_B must be disjoint")
    SA = entropy(reduce(V, modes_A))
    SB = entropy(reduce(V, modes_B))
    SAB = entropy(reduce(V, modes_A + modes_B))
    return SA + SB - SAB


def entanglement_hamiltonian(V_A, nu_floor=0.5 * (1.0 + 1e-14)):
    """Quadratic entanglement (modular) Hamiltonian of a Gaussian state.

    rho_A ~ exp(-(1/2) R^T G R) with, via Williamson V_A = S diag(nu,nu) S^T,
    G = S^{-T} diag(g(nu), g(nu)) S^{-1},  g(nu) = ln((nu+1/2)/(nu-1/2))
    (Botero-Reznik 2004; Eisert-Cramer-Plenio RMP 82, 277 (2010), Sec. IV).

    nu is clamped at nu_floor = (1/2)(1 + 1e-14) before the log: modes at the
    vacuum floor get a large-but-finite modular temperature; callers needing
    more should escalate via vacuum.core.precision.
    """
    S, nu = williamson(V_A)
    nu = np.maximum(nu, nu_floor)
    g = np.log((nu + 0.5) / (nu - 0.5))
    Sinv = symplectic_inverse(S)
    return Sinv.T @ np.diag(np.concatenate([g, g])) @ Sinv


def modular_energy(G, V_A):
    """<K> = (1/2) Tr(G V_A) for K = (1/2) R^T G R (mean-zero state).

    Used for the entanglement first law delta S = delta<K> = (1/2) Tr(G dV_A)
    (Faulkner-Guica-Hartman-Myers-Van Raamsdonk 2014, lattice form).
    """
    return 0.5 * float(np.trace(np.asarray(G) @ np.asarray(V_A)))


def symplectic_from_quadratic(H_mat, t):
    """Symplectic evolution S = exp(t Omega H) for H = (1/2) R^T H_mat R.

    H_mat symmetric implies S Omega S^T = Omega for all t (Serafini 2017,
    Sec. 3.1: Gaussian dynamics generated by quadratic Hamiltonians).
    """
    from scipy.linalg import expm

    H_mat = np.asarray(H_mat, dtype=float)
    N = H_mat.shape[0] // 2
    return expm(t * Omega(N) @ H_mat)


def evolve(V, S):
    """Evolve a covariance matrix: V -> S V S^T."""
    return S @ V @ S.T


def mean_energy(V, K):
    """Mean energy <H> of a mean-zero Gaussian state with covariance V.

    ``K`` of shape (N, N): <H> = (1/2) Tr(V_pp) + (1/2) Tr(K V_xx) for
    H = (1/2)(p.p + x.K.x) — on the ground state this equals
    Tr(K^{1/2})/2, the zero-point energy.

    ``K`` of shape (2N, 2N): the general quadratic form H = (1/2) R^T H_mat R
    (x-p cross terms allowed — e.g. the derivative-coupled detector stack,
    ``vacuum.detectors.attach_detectors(..., coupling='xp')``), for which
    <H> = (1/2) Tr(H_mat V).  The two paths coincide identically when
    H_mat = diag(K, I) (tests/test_xp_coupling.py pins them to 1e-12), so
    the energy ledger can book snapshots under either form.
    """
    V = np.asarray(V, dtype=float)
    K = np.asarray(K, dtype=float)
    N = V.shape[0] // 2
    if V.ndim != 2 or V.shape != (2 * N, 2 * N):
        raise ValueError(f"V must be a (2N, 2N) covariance, got {V.shape}")
    if K.shape == (N, N):
        return 0.5 * float(np.trace(V[N:, N:])) + 0.5 * float(np.trace(K @ V[:N, :N]))
    if K.shape == (2 * N, 2 * N):
        return 0.5 * float(np.trace(K @ V))
    raise ValueError(
        f"K must be an (N, N) coupling matrix or a (2N, 2N) quadratic form "
        f"for a {2 * N}-dimensional covariance, got {K.shape}"
    )
