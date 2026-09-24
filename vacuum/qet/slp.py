"""Strong local passivity (SLP) — M3.3.

A state rho of a multi-qubit system with Hamiltonian H is *strongly local
passive* with respect to a region A when no quantum operation confined to A
— any CPTP map, not just a unitary — can lower the global mean energy:

    Delta E(rho) = Tr[H rho] - Tr[H (G (x) I_B)(rho)] <= 0   for every CPTP G on A.

The notion, and the name, are from

    [FFH]  M. Frey, K. Funo, M. Hotta, "Strong local passivity in finite
           quantum systems", Phys. Rev. E 90, 012127 (2014)
           [arXiv:1404.5081]. Eq. (1) defines Delta E; Eqs. (17)-(18) give
           the single-qubit-region Kraus parametrization and its
           completeness constraints; the "local energy" Omega_o of the
           region is the maximum of Delta E over all local operations G,
           with Omega_o >= 0 and rho SLP iff Omega_o = 0.

The certificate implemented here (the *analytic rung*) is the exact iff
criterion that completes the FFH program for regions of any size:

    [ASRSM] A. M. Alhambra, G. Styliaris, N. A. Rodriguez-Briones,
            J. Sikora, E. Martin-Martinez, "Fundamental limitations to
            local energy extraction in quantum systems", Phys. Rev. Lett.
            123, 190601 (2019) [arXiv:1902.02357]. Verbatim:

    Eq. (1):  Delta E_{(A)B} := min_{E_A} Tr[H_AB (E_A (x) I_B) rho_AB]
                                - Tr[H_AB rho_AB],
    Eq. (3):  Tr[H_AB (E_A (x) I_B) rho_AB] = Tr[C_AA' E_AA'] with the
              cost operator  C_AA' = Tr_B[ rho_AB^{Gamma_A} H_A'B ]
              (Gamma_A = partial transpose on A; E_AA' the unnormalized
              Choi-Jamiolkowski operator of E_A, Tr_A' E_AA' = I_A),
    Thm. 1:   {rho_AB, H_AB} is CP-local passive if and only if
              C_AA' - Tr_A'[ d_A |Phi><Phi| C_AA' ] (x) I_A' >= 0,
    Sec. A.1: the robustness remark
              Tr(d_A |Phi><Phi| C_AA') - min_E Tr(E_AA' C_AA') <= eps d_A,
              i.e. Delta E_{(A)B} >= -eps d_A, when the operator of Thm. 1
              has smallest eigenvalue -eps.

Writing Lambda := Tr_A'[d_A |Phi><Phi| C] (a d_A x d_A operator), the
criterion is closed-form linear algebra: Lambda must be Hermitian *and*
C - Lambda (x) I must be PSD. Both halves are separately meaningful:

  * Lambda^T = Tr_B[H rho] exactly (proved in :func:`slp_certificate`), so
    its Hermiticity deviation equals max |Tr_B[[H, rho]]| — the drift of
    the reduced state of A, :func:`reduced_commutator`. A nonzero drift is
    already first-order extractable by a local *unitary*, so this half is
    the "no local energy current out of A" condition. States commuting
    with H (Gibbs states, eigenmixtures) satisfy it identically.
  * The PSD half is the genuinely CP-map condition, and its violation
    -eps = min eig gives the quantitative bound Delta E <= eps d_A.

For a single-qubit region this is the sharp (iff) form of the FFH
single-qubit-region analysis, and this module's anchors check it against
the closed forms FFH derive for their two-particle models:

    Eq. (15): H = kappa sx(x)sx + sz(x)I + I(x)sz, eigenenergies Eq. (16)
              (-m, -kappa, kappa, m), m = sqrt(kappa^2 + 4);
    Eq. (20): eta = (2/m) delta_0, xi = (kappa^2/m) delta_0 + kappa delta_1,
              delta_0 = p_0 - p_3, delta_1 = p_1 - p_2 for an eigenmixture
              rho = sum_k p_k |E_k><E_k|;
    Eq. (21): Omega_o = sqrt((1-eta^2+xi^2)/(1-eta^2)) - xi - eta
                        if |eta xi| < 1 - eta^2,
                        |xi| + |eta| - xi - eta  otherwise;
    Eq. (23): Gibbs-state differences delta_0 = (2/Z) sinh(m/T),
              delta_1 = (2/Z) sinh(kappa/T),
              Z = 2 (cosh(kappa/T) + cosh(m/T))   (k_B = 1);
    Eq. (33): the XXX pair H_X = sx(x)sx + sy(x)sy + sz(x)sz, whose Gibbs
              states are SLP at *every* temperature.

The *SDP rung* solves ASRSM Eqs. (1)+(3) directly as the Choi program

    minimize  Tr[J C]   over  J >= 0,  Tr_out J = I_A,

for 1-4-site regions of <= 8-qubit systems, behind the ``.[sdp]`` extra
(cvxpy is imported lazily inside the SDP functions only; the rest of the
module — the analytic certificate, the Kraus search, the maps — runs on
the default install). The solver's answer is never trusted on its own:
:func:`extractable_energy_sdp` repairs the returned primal into an exactly
feasible Choi matrix (giving an *achievable* upper bound on the post-map
energy) and repairs the dual into an exactly feasible Lambda (giving a
rigorous lower bound), so every call reports a certified bracket and its
width, and escalates solvers until the bracket is tight.

Conventions: hbar = 1 (docs/API.md); qubit site 0 is the leftmost (most
significant) tensor factor, matching vacuum.qet.hotta (qubit A = site 0)
and vacuum.qet.chains. The Choi composite index on A_in (x) A_out puts
A_in first: (i, o) flattens to i * d_A + o. All functions are pure over
their array arguments — no hidden state, no input mutation (Layer-6
JAX-mirror discipline).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "state_energy",
    "gibbs_state",
    "eigenmixture",
    "choi_cost_matrix",
    "choi_to_kraus",
    "embed_region_operator",
    "apply_local_channel",
    "reduced_commutator",
    "slp_certificate",
    "extractable_energy_kraus",
    "extractable_energy_sdp",
    "locc_extractable_energy",
    "slp_map",
    "slp_critical_temperature",
    "ffh_pair_hamiltonian",
    "ffh_pair_eigenenergies",
    "ffh_pair_eigenmixture",
    "ffh_pair_gibbs_populations",
    "ffh_pair_local_energy",
    "xxx_pair_hamiltonian",
    "xxx_pair_eigenbasis",
    "xxx_pair_eigenmixture",
]

# Pauli matrices, computational basis |0> = |sz=+1>, |1> = |sz=-1>.
_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


# ---------------------------------------------------------------------------
# validation and small helpers
# ---------------------------------------------------------------------------


def _herm(X):
    """Hermitian part (X + X^dag)/2."""
    return (X + X.conj().T) / 2.0


def _herm_dev(X):
    """Max-abs deviation of a square matrix from Hermiticity."""
    return float(np.max(np.abs(X - X.conj().T)))


def _check_qubit_operator(name, X):
    """Validate a Hermitian operator on n qubits; return (X complex, n)."""
    X = np.asarray(X, dtype=complex)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"{name} must be square, got shape {X.shape}")
    n = int(round(np.log2(X.shape[0])))
    if 2**n != X.shape[0]:
        raise ValueError(f"{name} dimension {X.shape[0]} is not a power of 2")
    dev = _herm_dev(X)
    scale = max(1.0, float(np.max(np.abs(X))))
    if dev > 1e-10 * scale:
        raise ValueError(f"{name} is not Hermitian (max dev {dev:.3e})")
    return X, n


def _check_region(region, n):
    """Normalize a region spec to a sorted tuple of distinct site indices."""
    if np.isscalar(region):
        region = (int(region),)
    region = tuple(sorted(int(q) for q in region))
    if len(set(region)) != len(region):
        raise ValueError(f"region has repeated sites: {region}")
    if not region:
        raise ValueError("region must contain at least one site")
    if region[0] < 0 or region[-1] >= n:
        raise ValueError(f"region {region} outside system of {n} qubits")
    return region


def _check_populations(populations, dim):
    p = np.asarray(populations, dtype=float)
    if p.shape != (dim,):
        raise ValueError(f"need {dim} populations, got shape {p.shape}")
    if np.any(p < -1e-12) or abs(float(np.sum(p)) - 1.0) > 1e-10:
        raise ValueError("populations must be a probability vector")
    return np.clip(p, 0.0, None)


def state_energy(rho, H):
    """Mean energy Re Tr(rho H) of a density matrix (real part strips roundoff)."""
    return float(np.real(np.trace(np.asarray(rho) @ np.asarray(H))))


def gibbs_state(H, T):
    """Gibbs state exp(-H/T)/Z (k_B = 1, hbar = 1) via dense eigh.

    T = np.inf returns the maximally mixed state; T must be > 0 (the T -> 0
    limit is the ground-state projector — pass a tiny T or build it directly).
    Populations are the Boltzmann weights p_k = exp(-E_k/T)/Z, computed with
    the ground energy shifted out so exp() never overflows.
    """
    H, _ = _check_qubit_operator("H", H)
    T = float(T)
    if not T > 0.0:
        raise ValueError(f"need T > 0 (np.inf allowed), got T={T}")
    w, U = np.linalg.eigh(H)
    if np.isinf(T):
        p = np.full(w.shape, 1.0 / w.size)
    else:
        x = -(w - float(np.min(w))) / T  # shift-stabilized Boltzmann weights
        p = np.exp(x)
        p = p / float(np.sum(p))
    return (U * p) @ U.conj().T


def eigenmixture(H, populations):
    """rho = sum_k p_k |E_k><E_k| in the ascending eigh eigenbasis of H.

    FFH's eigenmixture family (their Eq. (4) region of state space).
    Populations are indexed by ascending energy. For a degenerate H the eigh
    basis inside a degenerate level is arbitrary — use an explicit eigenbasis
    (e.g. :func:`xxx_pair_eigenmixture`) when the level split matters.
    """
    H, _ = _check_qubit_operator("H", H)
    p = _check_populations(populations, H.shape[0])
    _, U = np.linalg.eigh(H)
    return (U * p) @ U.conj().T


# ---------------------------------------------------------------------------
# region plumbing and the Choi-program cost operator (ASRSM Eq. (3))
# ---------------------------------------------------------------------------


def _as_region_blocks(op, region, n):
    """Reshape a 2^n x 2^n operator to (dA, dB, dA, dB) with the region first.

    Site 0 is the leftmost (most significant) tensor factor; the A factor is
    the tensor product of the region's sites in ascending site order. The
    result indexes as t[a, b, a', b'] = op_{(a b), (a' b')}.
    """
    rest = [q for q in range(n) if q not in region]
    order = list(region) + rest
    t = np.asarray(op).reshape((2,) * (2 * n))
    t = np.transpose(t, order + [n + q for q in order])
    dA = 2 ** len(region)
    dB = 2 ** (n - len(region))
    return t.reshape(dA, dB, dA, dB)


def embed_region_operator(K, region, n):
    """Embed an operator K on the region's factor as K (x) I on the full chain.

    K is dA x dA in the region's ascending-site-order basis; the result is
    2^n x 2^n in the site-ordered basis. Pure function; used to apply Kraus
    operators (from :func:`extractable_energy_kraus` or :func:`choi_to_kraus`)
    to full-system states via :func:`apply_local_channel`.
    """
    region = _check_region(region, n)
    dA = 2 ** len(region)
    K = np.asarray(K, dtype=complex)
    if K.shape != (dA, dA):
        raise ValueError(f"K must be {dA}x{dA} for region {region}, got {K.shape}")
    rest = [q for q in range(n) if q not in region]
    order = list(region) + rest
    inv = np.argsort(order)  # permuted axis -> site axis
    dB = 2 ** (n - len(region))
    t = np.kron(K, np.eye(dB, dtype=complex)).reshape((2,) * (2 * n))
    t = np.transpose(t, list(inv) + [n + q for q in inv])
    return t.reshape(2**n, 2**n)


def apply_local_channel(rho, kraus, region):
    """Apply a CPTP map on the region to a full-system state: sum_mu K_mu rho K_mu^dag.

    ``kraus`` is a sequence of (d_A, d_A) matrices in the region's basis; each
    is embedded with :func:`embed_region_operator`. Pure — ``rho`` is not
    mutated. No completeness check is made here (see
    :func:`extractable_energy_kraus`, whose Kraus sets are isometric by
    construction); ``Tr`` of the output is the diagnostic.
    """
    rho, n = _check_qubit_operator("rho", rho)
    region = _check_region(region, n)
    out = np.zeros_like(rho)
    for K in kraus:
        Kf = embed_region_operator(K, region, n)
        out = out + Kf @ rho @ Kf.conj().T
    return out


def reduced_commutator(H, rho, region):
    """Tr_B([H, rho]) on the region — the drift of the region's reduced state.

    Equals i d(rho_A)/dt under the global Heisenberg flow, and equals the
    Hermiticity deviation source of the ASRSM certificate: with Lambda as in
    :func:`slp_certificate`, ``Lambda - Lambda^dag = (Tr_B[H, rho])^T``.
    Nonzero drift means a local *unitary* already extracts energy at first
    order, so it is exactly the "no local energy current out of A" half of
    the criterion. Identically zero when [H, rho] = 0.

    Returns the (d_A, d_A) anti-Hermitian matrix.
    """
    H, n = _check_qubit_operator("H", H)
    rho, n_r = _check_qubit_operator("rho", rho)
    if n_r != n:
        raise ValueError(f"H is on {n} qubits but rho is on {n_r}")
    region = _check_region(region, n)
    comm = H @ rho - rho @ H
    return np.einsum("abcb->ac", _as_region_blocks(comm, region, n))


def choi_cost_matrix(H, rho, region):
    """The Hermitian cost operator C_AA' of ASRSM Eq. (3).

    C_AA' = Tr_B[ rho^{Gamma_A} H_A'B ]: for any CPTP map E on region A with
    unnormalized Choi operator J on A_in (x) A_out (J >= 0, Tr_out J = I),
    the post-map energy is Tr[H (E (x) I_B)(rho)] = Tr[J C]. Composite index
    (i, o) -> i * d_A + o (A_in first). In components,

        C_{(a' c), (a c')} = sum_{b b'} rho_{(a b), (a' b')} H_{(c b'), (c' b)}.

    Returns the d_A^2 x d_A^2 Hermitian ndarray (Hermitian analytically for
    Hermitian rho, H; the roundoff-level antisymmetric part is projected out).
    """
    H, n = _check_qubit_operator("H", H)
    rho, n_r = _check_qubit_operator("rho", rho)
    if n_r != n:
        raise ValueError(f"H is on {n} qubits but rho is on {n_r}")
    region = _check_region(region, n)
    r = _as_region_blocks(rho, region, n)  # r[a, b, a', b']
    h = _as_region_blocks(H, region, n)    # h[c, b', c', b]
    dA = 2 ** len(region)
    C = np.einsum("abpq,cqdb->pcad", r, h).reshape(dA * dA, dA * dA)
    return _herm(C)


def choi_to_kraus(J, tol=1e-10):
    """Kraus operators of the CP map with unnormalized Choi matrix J.

    J is (d_A^2, d_A^2) with composite index (in, out), in-major, i.e.
    J = sum_mu k_mu k_mu^dag with k_mu[(i, o)] = (K_mu)_{o i}. Eigenvectors of
    J with eigenvalue above ``tol`` (absolute) become Kraus operators scaled
    by sqrt(eigenvalue); the map is CPTP iff sum_mu K_mu^dag K_mu = I, which
    holds when Tr_out J = I.

    Returns a list of (d_A, d_A) complex ndarrays, descending by weight.
    """
    J = np.asarray(J, dtype=complex)
    d2 = J.shape[0]
    dA = int(round(np.sqrt(d2)))
    if dA * dA != d2 or J.shape != (d2, d2):
        raise ValueError(f"J must be (d_A^2, d_A^2), got shape {J.shape}")
    w, V = np.linalg.eigh(_herm(J))
    kraus = []
    for idx in np.argsort(w)[::-1]:
        if w[idx] <= float(tol):
            continue
        # column V[:, idx] is k_mu with index (i, o); K_mu = reshape(k)^T
        kraus.append(np.sqrt(w[idx]) * V[:, idx].reshape(dA, dA).T)
    return kraus


# ---------------------------------------------------------------------------
# analytic rung: the exact SLP certificate (ASRSM Thm. 1; FFH for one qubit)
# ---------------------------------------------------------------------------


def slp_certificate(H, rho, region, tol=1e-9):
    """Exact strong-local-passivity certificate for (rho, H) on a region.

    ASRSM Theorem 1: with C = :func:`choi_cost_matrix` and

        Lambda = Tr_A'[ d_A |Phi><Phi| C ],

    the pair {rho, H} is SLP (their "CP-local passive") w.r.t. the region iff
    Lambda is Hermitian and C - Lambda (x) I_A' >= 0. For a single-qubit
    region this is the sharp form of the FFH single-qubit-region criterion
    (FFH Eqs. (17)-(18) optimize the qubit Kraus parametrization directly;
    ASRSM Thm. 1 makes the condition necessary *and* sufficient).

    Implementation. With |Phi_u> = sum_a |a a> (unnormalized), Lambda is
    ``reshape(C |Phi_u>, (d_A, d_A))`` up to the adjoint that Hermitization
    removes. Why that is the whole story (strong duality of the Choi program,
    which is strictly feasible at J = I/d_A so Slater holds): the dual is
    max Tr[Lambda] over Hermitian Lambda with C - Lambda (x) I >= 0, and the
    identity channel J = |Phi_u><Phi_u| is primal optimal iff some dual
    optimum satisfies complementary slackness (C - Lambda (x) I)|Phi_u> = 0,
    which fixes Lambda = reshape(C|Phi_u>) uniquely. Two consequences used
    below:

      * Tr[Lambda] = <Phi_u|C|Phi_u> = Tr[rho H] exactly (the identity
        channel's energy), so it is real for any input;
      * Lambda^T = Tr_B[H rho], hence Lambda - Lambda^dag = (Tr_B[[H,rho]])^T
        (:func:`reduced_commutator`).

    Because Tr[Lambda] is real, ``herm(Lambda) - eps I`` is a *dual feasible*
    point for eps = max(0, -lambda_min) whatever Lambda's Hermiticity, and
    weak duality then gives the bound Delta E <= d_A * eps unconditionally —
    the ASRSM Sec. A.1 robustness remark, valid here even when the
    Hermiticity half of the criterion fails.

    Parameters
    ----------
    H, rho : (2^n, 2^n) Hermitian ndarrays.
    region : int or sequence of site indices (0-based; site 0 leftmost).
    tol : float
        Absolute decision tolerance (energy units of H) on both the
        Hermiticity deviation of Lambda and the minimum eigenvalue.

    Returns
    -------
    dict with keys
        'slp'         : bool verdict (both halves within ``tol``),
        'herm_dev'    : max-abs deviation of Lambda from Hermiticity
                        (= max |Tr_B[[H, rho]]|),
        'lambda_min'  : min eigenvalue of C - herm(Lambda) (x) I,
        'extractable_bound' : d_A * max(0, -lambda_min), a rigorous upper
                        bound on the locally extractable energy,
        'e_initial'   : Tr[rho H],
        'Lambda', 'C' : the certificate operators,
        'region', 'd_A', 'tol' : echoes.
    """
    H_c, n = _check_qubit_operator("H", H)
    region_t = _check_region(region, n)
    C = choi_cost_matrix(H_c, rho, region_t)
    dA = 2 ** len(region_t)
    phi = np.eye(dA, dtype=complex).reshape(dA * dA)  # |Phi_u> = sum_a |a a>
    Lambda = (C @ phi).reshape(dA, dA)
    herm_dev = _herm_dev(Lambda)
    Lambda_h = _herm(Lambda)
    Z = _herm(C - np.kron(Lambda_h, np.eye(dA, dtype=complex)))
    lambda_min = float(np.linalg.eigvalsh(Z)[0])
    tol = float(tol)
    slp = (herm_dev <= tol) and (lambda_min >= -tol)
    return {
        "slp": bool(slp),
        "herm_dev": herm_dev,
        "lambda_min": lambda_min,
        "extractable_bound": dA * max(0.0, -lambda_min),
        "e_initial": state_energy(rho, H_c),
        "Lambda": Lambda,
        "C": C,
        "region": region_t,
        "d_A": dA,
        "tol": tol,
    }


# ---------------------------------------------------------------------------
# constructive rung: direct maximization over local Kraus operations
# (core-only fallback for the SDP rung — no cvxpy)
# ---------------------------------------------------------------------------


def _kraus_stack_to_vecs(T, r, dA):
    """Vertically stacked Kraus [K_0; ...; K_{r-1}] -> Choi vectors k_mu.

    k_mu[(i, o)] = (K_mu)_{o i} so that J = sum_mu k_mu k_mu^dag is the
    unnormalized Choi operator with composite index (in, out), in-major.
    """
    K = T.reshape(r, dA, dA)
    return K.transpose(0, 2, 1).reshape(r, dA * dA)


def extractable_energy_kraus(H, rho, region, n_kraus=None, restarts=8, seed=0):
    """Lower-bound the locally extractable energy by direct Kraus search.

    Maximizes Delta E over CPTP maps on the region in Kraus form; for a qubit
    region this is exactly the FFH Eq. (17)-(18) parametrization, since at
    most d_A^2 Kraus operators are needed. The completeness constraint
    sum_mu K_mu^dag K_mu = I says the vertical stack T = [K_0; ...; K_{r-1}]
    is an isometry, T^dag T = I_{d_A}; the search runs over unconstrained
    matrices retracted onto the isometry manifold by polar decomposition,
    with L-BFGS restarts from random starts plus the identity map and the
    computational-basis reset maps.

    This is a *restart-certified lower bound* on the local energy Omega_o
    (it also exhibits the extracting operation); the certified value is the
    SDP rung's job (:func:`extractable_energy_sdp`), and the two are anchored
    against each other and against FFH Eq. (21) in the tests. Cost grows
    fast with region size (2 r d_A^2 real parameters per start) — for
    multi-site regions prefer the SDP rung.

    Returns
    -------
    dict with keys
        'extractable' : best Delta E found (>= 0; the identity map is a start),
        'kraus'       : list of (d_A, d_A) Kraus operators achieving it
                        (apply via :func:`apply_local_channel`),
        'e_min'       : the post-map energy achieved,
        'e_initial'   : Tr[rho H],
        'region', 'n_kraus', 'restarts' : echoes.
    """
    H_c, n = _check_qubit_operator("H", H)
    region_t = _check_region(region, n)
    C = choi_cost_matrix(H_c, rho, region_t)
    dA = 2 ** len(region_t)
    r = int(n_kraus) if n_kraus is not None else dA * dA
    if r < 1:
        raise ValueError(f"n_kraus must be >= 1, got {r}")
    e0 = state_energy(rho, H_c)
    rng = np.random.default_rng(seed)
    m_rows = r * dA

    def isometrize(A):
        U, _, Vh = np.linalg.svd(A, full_matrices=False)
        return U @ Vh

    def objective(x):
        A = (x[: m_rows * dA] + 1j * x[m_rows * dA:]).reshape(m_rows, dA)
        T = isometrize(A)
        k = _kraus_stack_to_vecs(T, r, dA)
        return float(np.real(np.einsum("mi,ij,mj->", k.conj(), C, k)))

    def pack(T):
        return np.concatenate([T.real.reshape(-1), T.imag.reshape(-1)])

    # deterministic starts: identity map, and reset-to-basis-state maps
    starts = []
    T_id = np.zeros((m_rows, dA), dtype=complex)
    T_id[:dA, :] = np.eye(dA)
    starts.append(pack(T_id))
    for b in range(min(dA, r)):
        # reset map with Kraus {|b><i|}_i : block i holds K_i = |b><i|
        T_reset = np.zeros((m_rows, dA), dtype=complex)
        for i in range(min(dA, r)):
            T_reset[i * dA + b, i] = 1.0
        starts.append(pack(T_reset))
    for _ in range(int(restarts)):
        A = rng.standard_normal((m_rows, dA)) + 1j * rng.standard_normal((m_rows, dA))
        starts.append(pack(A))

    best_val = np.inf
    best_x = starts[0]
    for x0 in starts:
        res = minimize(objective, x0, method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-12})
        if res.fun < best_val:
            best_val = float(res.fun)
            best_x = res.x
    A = (best_x[: m_rows * dA] + 1j * best_x[m_rows * dA:]).reshape(m_rows, dA)
    T = isometrize(A)
    kraus = [T.reshape(r, dA, dA)[mu] for mu in range(r)]
    best_val = min(best_val, e0)  # the identity map is always available
    return {
        "extractable": e0 - best_val,
        "kraus": kraus,
        "e_min": best_val,
        "e_initial": e0,
        "region": region_t,
        "n_kraus": r,
        "restarts": int(restarts),
    }


# ---------------------------------------------------------------------------
# SDP rung (guarded: cvxpy behind the `.[sdp]` extra)
# ---------------------------------------------------------------------------

# Solver order and settings, chosen by measurement rather than by taste, on
# the FFH Eq. (15) Gibbs grid (d_A = 2) and on transverse-field Ising Gibbs
# states (d_A = 4, 8), each judged by the certified bracket width below:
#
#   d_A   SCS (eps 1e-12)        CLARABEL (tol_gap 1e-13)
#     2   gap ~3e-12,  7 ms      gap ~9e-11,  10 ms
#     4   gap ~4e-12, 20 ms      gap ~6e-08,  60 ms
#     8   gap ~8e-12,  0.4 s     gap ~5e-07,   24 s
#
# so SCS leads and CLARABEL is the fallback when SCS errors or leaves a loose
# bracket. SCS is a first-order method, so its iteration budget is capped
# rather than left to run; the bracket, not the solver's own status, is what
# decides whether the answer is good enough.
_SDP_SOLVER_ORDER = ("SCS", "CLARABEL")
_SDP_SOLVER_DEFAULTS = {
    "SCS": {"eps_abs": 1e-12, "eps_rel": 1e-12, "max_iters": 20_000},
    "CLARABEL": {"tol_gap_abs": 1e-13, "tol_gap_rel": 1e-13,
                 "tol_feas": 1e-9, "max_iter": 1000},
}


def _load_cvxpy():
    try:
        import cvxpy  # noqa: PLC0415 — extra-gated import by design (spec)
    except ImportError as exc:  # pragma: no cover - exercised only sans extra
        raise ImportError(
            "the SLP SDP rung needs cvxpy — install the extra: "
            "pip install 'vacuum[sdp]'"
        ) from exc
    return cvxpy


def _partial_trace_out(J, dA):
    """Tr_out J for a Choi matrix with composite index (in, out), in-major."""
    return np.einsum("iojo->ij", J.reshape(dA, dA, dA, dA))


def _repair_primal(C, J, dA, e0):
    """Project a numerical Choi matrix onto the exact feasible set.

    Feasibility is J >= 0 and Tr_out J = I. PSD is restored by clipping the
    eigenvalues; the trace condition by the congruence
    J -> (M^{-1/2} (x) I) J (M^{-1/2} (x) I) with M = Tr_out J, which maps
    Tr_out J to M^{-1/2} M M^{-1/2} = I and preserves positivity. A tiny
    multiple of the identity is added first if M is near-singular. The
    resulting Tr[C J] is therefore an *achievable* post-map energy, hence a
    valid upper bound on the optimum; it is capped at e0 = Tr[rho H] because
    the identity channel is always feasible and attains exactly e0.
    """
    Jp = _herm(np.asarray(J, dtype=complex))
    w, U = np.linalg.eigh(Jp)
    Jp = (U * np.clip(w, 0.0, None)) @ U.conj().T
    M = _herm(_partial_trace_out(Jp, dA))
    wm = np.linalg.eigvalsh(M)
    floor = 1e-12 * max(1.0, float(np.max(np.abs(wm))))
    if float(wm[0]) < floor:
        # Tr_out(I_{d_A^2}) = d_A I_A, so this shifts M by (floor - wm[0]) I.
        Jp = Jp + ((floor - float(wm[0])) / dA) * np.eye(dA * dA, dtype=complex)
        M = _herm(_partial_trace_out(Jp, dA))
    wm, Um = np.linalg.eigh(M)
    Xm = (Um * (1.0 / np.sqrt(np.clip(wm, 1e-300, None)))) @ Um.conj().T
    Xf = np.kron(Xm, np.eye(dA, dtype=complex))
    Jf = _herm(Xf @ Jp @ Xf.conj().T)
    ub = float(np.real(np.trace(C @ Jf)))
    if ub > e0:
        return e0, None  # identity channel wins; its Choi is |Phi_u><Phi_u|
    return ub, Jf


def _dual_lower_bound(C, L, dA):
    """Weak-duality lower bound from an arbitrary Hermitian candidate L.

    For any Hermitian Lambda with C - Lambda (x) I >= 0 and any feasible J,
    Tr[JC] = Tr[J (C - Lambda (x) I)] + Tr[Lambda Tr_out J] >= Tr[Lambda].
    Shifting L by its smallest eigenvalue makes the PSD condition exact, so
    Tr[herm(L)] + d_A * min(0, lambda_min) is a rigorous lower bound for any
    input L whatsoever — including a solver dual whose sign convention is
    unknown.
    """
    Lh = _herm(np.asarray(L, dtype=complex))
    mu = float(np.linalg.eigvalsh(_herm(C - np.kron(Lh, np.eye(dA, dtype=complex))))[0])
    return float(np.real(np.trace(Lh))) + dA * min(mu, 0.0)


def extractable_energy_sdp(H, rho, region, tol=1e-9, solver=None,
                           solver_opts=None, gap_target=1e-10):
    """Certified locally-extractable energy via the Choi program (SDP rung).

    ASRSM Eqs. (1) + (3): minimize Tr[J C] over unnormalized Choi operators
    J >= 0 with Tr_out J = I_A (C from :func:`choi_cost_matrix`); the
    extractable energy is Tr[rho H] - min, and {rho, H} is SLP iff it is 0.

    The solver's own answer is never reported directly. The returned primal
    is repaired onto the exact feasible set (:func:`_repair_primal`), giving
    an *achieved* post-map energy ``e_min`` and hence a lower bound
    ``extractable``; the dual is repaired by an eigenvalue shift
    (:func:`_dual_lower_bound`) — as is the analytic certificate's Lambda,
    which is always available — giving a rigorous ``extractable_upper``.
    ``gap`` is the width of that bracket, and with ``solver=None`` the
    solvers in ``_SDP_SOLVER_ORDER`` are tried in turn until the gap falls
    below ``gap_target``, the tightest bracket being kept.

    Dimensions are kept honest per M3.3: 1-4-site regions of <= 8-qubit
    systems (the single-site case is the overlap with the analytic rung).
    Cost is set by d_A = 2^|region|, not by n: measured here, one call is
    ~10 ms at d_A = 2, ~30 ms at d_A = 4, and 0.3-25 s at d_A = 8, the slow
    end being near-SLP states whose optimum sits on the boundary and where
    the escalation therefore fires. ``gap_target=np.inf`` disables escalation
    (one solve, whatever bracket it gives) when charting many points.

    Parameters
    ----------
    H, rho : (2^n, 2^n) Hermitian ndarrays, n <= 8.
    region : int or sequence of at most 4 site indices.
    tol : float
        SLP decision tolerance on the certified upper bound.
    solver : str or None
        cvxpy solver name; None escalates over ``_SDP_SOLVER_ORDER``.
    solver_opts : dict or None
        Overrides for the per-solver defaults in ``_SDP_SOLVER_DEFAULTS``.
    gap_target : float
        Stop escalating once the certified bracket is this tight.

    Returns
    -------
    dict with keys
        'extractable'       : Tr[rho H] - e_min, achieved by 'choi' (>= 0),
        'extractable_upper' : rigorous upper bound from the repaired dual,
        'gap'               : extractable_upper - extractable (>= 0),
        'slp'               : extractable_upper <= tol (a certified verdict),
        'e_min'             : the achieved post-map energy,
        'e_initial'         : Tr[rho H],
        'choi'              : the feasible Choi matrix achieving e_min
                              (composite (in, out), in-major); the identity
                              channel's |Phi_u><Phi_u| when nothing beat it,
        'status', 'solver', 'region', 'd_A', 'tol' : diagnostics/echoes.
    """
    cp = _load_cvxpy()
    H_c, n = _check_qubit_operator("H", H)
    region_t = _check_region(region, n)
    if n > 8:
        raise ValueError(f"SDP rung capped at 8-qubit systems (M3.3), got n={n}")
    if len(region_t) > 4:
        raise ValueError(
            f"SDP rung capped at 4-site regions (M3.3), got {len(region_t)} sites"
        )
    C = choi_cost_matrix(H_c, rho, region_t)
    dA = 2 ** len(region_t)
    e0 = state_energy(rho, H_c)
    eye_dA = np.eye(dA, dtype=complex)
    phi_choi = np.outer(eye_dA.reshape(dA * dA), eye_dA.reshape(dA * dA).conj())

    # the analytic certificate's Lambda is a dual candidate that costs nothing
    lam_cert = (C @ eye_dA.reshape(dA * dA)).reshape(dA, dA)
    best_lb = _dual_lower_bound(C, lam_cert, dA)
    best_ub, best_choi, best_status, best_solver = e0, None, None, None

    if solver is None:
        candidates = [s for s in _SDP_SOLVER_ORDER if s in cp.installed_solvers()]
        if not candidates:
            raise RuntimeError(
                f"no supported SDP solver installed; have {cp.installed_solvers()}"
            )
    else:
        candidates = [str(solver)]

    failures = []
    for name in candidates:
        opts = dict(_SDP_SOLVER_DEFAULTS.get(name, {}))
        if solver_opts is not None:
            opts.update(solver_opts)
        J = cp.Variable((dA * dA, dA * dA), hermitian=True)
        con_tr = cp.partial_trace(J, (dA, dA), 1) == eye_dA  # Tr_out J = I_A
        problem = cp.Problem(cp.Minimize(cp.real(cp.trace(C @ J))), [J >> 0, con_tr])
        with warnings.catch_warnings():
            # 'Solution may be inaccurate' is expected at these tolerances and
            # is not a failure here: the bracket below measures the real error.
            warnings.filterwarnings("ignore", message=".*may be inaccurate.*")
            try:
                problem.solve(solver=name, **opts)
            except (cp.error.SolverError, cp.error.DCPError) as exc:
                failures.append(f"{name}: {exc}")
                continue
        if J.value is None:
            failures.append(f"{name}: status {problem.status!r}, no primal returned")
            continue
        ub, choi = _repair_primal(C, np.asarray(J.value), dA, e0)
        best_status, best_solver = problem.status, name
        if ub < best_ub:
            best_ub, best_choi = ub, choi
        if con_tr.dual_value is not None:
            for sign in (1.0, -1.0):
                best_lb = max(best_lb,
                              _dual_lower_bound(C, sign * np.asarray(con_tr.dual_value), dA))
        if best_ub - best_lb <= float(gap_target):
            break

    if best_solver is None:
        # Never degrade silently to "the identity map is optimal": that would
        # report SLP for anything. A dead solver is an error, not a verdict.
        raise RuntimeError(
            "the SLP Choi program produced no solution; tried "
            + "; ".join(failures or [str(c) for c in candidates])
        )

    extractable = e0 - best_ub                     # achieved, so >= 0 exactly
    # lb <= min <= ub holds in exact arithmetic; the eigenvalue solves inside
    # the two repairs can invert that by ~1e-15 ||C||, so keep the invariant.
    extractable_upper = max(e0 - best_lb, extractable)
    return {
        "extractable": extractable,
        "extractable_upper": extractable_upper,
        "gap": max(0.0, extractable_upper - extractable),
        "slp": bool(extractable_upper <= float(tol)),
        "e_min": best_ub,
        "e_initial": e0,
        "choi": phi_choi if best_choi is None else best_choi,
        "status": best_status,
        "solver": best_solver,
        "region": region_t,
        "d_A": dA,
        "tol": float(tol),
    }


# ---------------------------------------------------------------------------
# the LOCC layer: classical communication reopening extraction
# ---------------------------------------------------------------------------


def _extractable(H, rho, region, method, tol, kraus_kwargs):
    if method == "sdp":
        return extractable_energy_sdp(H, rho, region, tol=tol)["extractable"]
    if method == "kraus":
        return extractable_energy_kraus(H, rho, region, **(kraus_kwargs or {}))["extractable"]
    if method == "certificate":
        return slp_certificate(H, rho, region, tol=tol)["extractable_bound"]
    raise ValueError(f"method must be 'sdp', 'kraus' or 'certificate', got {method!r}")


def locc_extractable_energy(H, rho, region, projectors, method="sdp", tol=1e-9,
                            kraus_kwargs=None):
    """The two layers of the SLP boundary chart: local-forbidden, LOCC-allowed.

    A measurement outside the region (projectors summing to the identity)
    followed by a classical message and a region-local channel is strictly
    more than a region-local channel. This function computes both:

      * ``local``: the energy extractable from ``region`` by one CPTP map
        acting on the *post-measurement ensemble average*
        rho_meas = sum_mu P_mu rho P_mu — i.e. the measurement happened but
        the outcome never arrived. If rho_meas is SLP w.r.t. the region this
        is zero.
      * ``locc``: sum_mu p_mu * (energy extractable from ``region`` on the
        normalized branch rho_mu = P_mu rho P_mu / p_mu) — the outcome
        arrived, and the local channel is conditioned on it.

    ``locc - local > 0`` is classical communication reopening extraction on a
    state the local certificate forbids: exactly Hotta's QET, whose Bob-side
    optimum this reproduces when the projectors are Alice's sigma_x POVM (see
    :mod:`vacuum.qet.hotta`).

    Parameters
    ----------
    H, rho : (2^n, 2^n) Hermitian ndarrays.
    region : int or sequence of site indices — the *extracting* region; the
        projectors must act outside it for the LOCC reading to hold (this is
        checked only through the physics, not the operator support).
    projectors : sequence of (2^n, 2^n) orthogonal projectors summing to I.
    method : 'sdp' (default, certified), 'kraus' (core-only, no cvxpy) or
        'certificate' (the analytic *upper* bound, fast but not achieved).
    tol : decision/solver tolerance passed through.
    kraus_kwargs : dict passed to :func:`extractable_energy_kraus`.

    Returns
    -------
    dict with 'local', 'locc', 'gap' (= locc - local), 'rho_meas',
    'e_initial', 'e_meas', 'branches' (per-outcome dicts with p, rho,
    e_branch, extractable), and 'region', 'method' echoes.
    """
    H_c, n = _check_qubit_operator("H", H)
    rho_c, n_r = _check_qubit_operator("rho", rho)
    if n_r != n:
        raise ValueError(f"H is on {n} qubits but rho is on {n_r}")
    region_t = _check_region(region, n)
    projectors = [np.asarray(P, dtype=complex) for P in projectors]
    total = sum(projectors)
    dev = float(np.max(np.abs(total - np.eye(2**n))))
    if dev > 1e-10:
        raise ValueError(f"projectors must sum to the identity (max dev {dev:.3e})")

    rho_meas = np.zeros_like(rho_c)
    branches = []
    for P in projectors:
        branch = P @ rho_c @ P.conj().T
        p = float(np.real(np.trace(branch)))
        rho_meas = rho_meas + branch
        if p <= 1e-14:
            branches.append({"p": p, "rho": None, "e_branch": 0.0, "extractable": 0.0})
            continue
        rho_mu = _herm(branch / p)
        branches.append({
            "p": p,
            "rho": rho_mu,
            "e_branch": state_energy(rho_mu, H_c),
            "extractable": _extractable(H_c, rho_mu, region_t, method, tol, kraus_kwargs),
        })
    rho_meas = _herm(rho_meas)
    local = _extractable(H_c, rho_meas, region_t, method, tol, kraus_kwargs)
    locc = float(sum(b["p"] * b["extractable"] for b in branches))
    return {
        "local": float(local),
        "locc": locc,
        "gap": locc - float(local),
        "rho_meas": rho_meas,
        "e_initial": state_energy(rho_c, H_c),
        "e_meas": state_energy(rho_meas, H_c),
        "branches": branches,
        "region": region_t,
        "method": method,
    }


# ---------------------------------------------------------------------------
# the chart: slp_map over (state family, coupling, region)
# ---------------------------------------------------------------------------


def slp_map(hamiltonian_fn, state_fn, couplings, state_params, regions=None,
            tol=1e-9, locc_projectors_fn=None, locc_method="sdp"):
    """Chart the analytic SLP verdict over (coupling, state parameter, region).

    Parameters
    ----------
    hamiltonian_fn : callable(coupling) -> (2^n, 2^n) Hermitian H.
        n must be the same for every coupling, with 2 <= n <= 6 (the
        analytic rung's charted domain per M3.3; call
        :func:`slp_certificate` directly for anything bigger).
    state_fn : callable(H, state_param) -> (2^n, 2^n) density matrix
        (the state family: Gibbs temperatures, eigenmixture populations, ...).
    couplings, state_params : sequences of the two family parameters.
    regions : sequence of region specs (ints or site tuples); default all
        single-site regions.
    tol : decision tolerance handed to :func:`slp_certificate`.
    locc_projectors_fn : callable(H, rho, region) -> projector sequence, or
        None (default). When given, every point gains the second layer of the
        chart via :func:`locc_extractable_energy`: 'locc_local', 'locc',
        'locc_gap' and the boolean 'locc_reopens' (SLP locally, extractable
        with the classical bit). This costs one extraction solve per outcome
        per point, so it is opt-in.
    locc_method : method forwarded to :func:`locc_extractable_energy`.

    Returns
    -------
    dict with
        'verdicts'   : bool ndarray, shape (n_couplings, n_params, n_regions),
        'lambda_min' : float ndarray, same shape (certificate margins),
        'herm_dev'   : float ndarray, same shape,
        'bound'      : float ndarray, same shape (extractable_bound),
        'locc_gap'   : float ndarray, same shape, NaN unless
                       locc_projectors_fn was given,
        'rows'       : flat list of per-point dicts,
        'couplings', 'state_params', 'regions', 'tol' : echoes.
    """
    couplings = list(couplings)
    state_params = list(state_params)
    if not couplings or not state_params:
        raise ValueError("couplings and state_params must be non-empty")
    _, n = _check_qubit_operator("H", hamiltonian_fn(couplings[0]))
    if not 2 <= n <= 6:
        raise ValueError(
            f"slp_map charts 2-6 qubit systems (M3.3), got n={n}; "
            "call slp_certificate directly for larger systems"
        )
    if regions is None:
        regions = [(q,) for q in range(n)]
    regions = [_check_region(rg, n) for rg in regions]
    shape = (len(couplings), len(state_params), len(regions))
    verdicts = np.zeros(shape, dtype=bool)
    lam_min = np.zeros(shape, dtype=float)
    herm_dev = np.zeros(shape, dtype=float)
    bound = np.zeros(shape, dtype=float)
    locc_gap = np.full(shape, np.nan, dtype=float)
    rows = []
    for i, c in enumerate(couplings):
        H, n_i = _check_qubit_operator("H", hamiltonian_fn(c))
        if n_i != n:
            raise ValueError("hamiltonian_fn changed system size across couplings")
        for j, p in enumerate(state_params):
            rho = state_fn(H, p)
            for k, rg in enumerate(regions):
                cert = slp_certificate(H, rho, rg, tol=tol)
                verdicts[i, j, k] = cert["slp"]
                lam_min[i, j, k] = cert["lambda_min"]
                herm_dev[i, j, k] = cert["herm_dev"]
                bound[i, j, k] = cert["extractable_bound"]
                row = {
                    "coupling": c,
                    "state_param": p,
                    "region": rg,
                    "slp": cert["slp"],
                    "lambda_min": cert["lambda_min"],
                    "herm_dev": cert["herm_dev"],
                    "extractable_bound": cert["extractable_bound"],
                }
                if locc_projectors_fn is not None:
                    lo = locc_extractable_energy(
                        H, rho, rg, locc_projectors_fn(H, rho, rg),
                        method=locc_method, tol=tol,
                    )
                    locc_gap[i, j, k] = lo["gap"]
                    row.update({
                        "locc_local": lo["local"],
                        "locc": lo["locc"],
                        "locc_gap": lo["gap"],
                        "locc_reopens": bool(lo["local"] <= tol and lo["locc"] > tol),
                    })
                rows.append(row)
    return {
        "verdicts": verdicts,
        "lambda_min": lam_min,
        "herm_dev": herm_dev,
        "bound": bound,
        "locc_gap": locc_gap,
        "rows": rows,
        "couplings": couplings,
        "state_params": state_params,
        "regions": regions,
        "tol": float(tol),
    }


def slp_critical_temperature(H, region, t_lo=1e-3, t_hi=1e3, tol=1e-9,
                             iterations=60):
    """Bisect the temperature at which the Gibbs family stops being SLP.

    FFH's central finding for Gibbs states: "Gibbs states are SL passive with
    respect to a subsystem only at or below a critical, system-dependent
    temperature" (abstract of Phys. Rev. E 90, 012127). This locates that
    temperature by bisection on the :func:`slp_certificate` verdict, assuming
    the verdict is monotone in T on [t_lo, t_hi] (true for the FFH models;
    the returned bracket lets a caller check).

    Returns
    -------
    dict with 'T_c' (bracket midpoint), 'bracket' (T_lo, T_hi with
    SLP(T_lo) and not SLP(T_hi)), 'slp_lo', 'slp_hi', and 'region'. When the
    whole interval is SLP (e.g. the XXX pair) T_c is np.inf; when none of it
    is, T_c is 0.0.
    """
    H_c, n = _check_qubit_operator("H", H)
    region_t = _check_region(region, n)

    def verdict(T):
        return slp_certificate(H_c, gibbs_state(H_c, T), region_t, tol=tol)["slp"]

    lo, hi = float(t_lo), float(t_hi)
    if not 0.0 < lo < hi:
        raise ValueError(f"need 0 < t_lo < t_hi, got {lo}, {hi}")
    slp_lo, slp_hi = verdict(lo), verdict(hi)
    if slp_lo and slp_hi:
        return {"T_c": np.inf, "bracket": (lo, hi), "slp_lo": True, "slp_hi": True,
                "region": region_t}
    if not slp_lo:
        return {"T_c": 0.0, "bracket": (lo, hi), "slp_lo": False, "slp_hi": slp_hi,
                "region": region_t}
    for _ in range(int(iterations)):
        mid = np.sqrt(lo * hi)  # geometric bisection: T spans decades
        if verdict(mid):
            lo = mid
        else:
            hi = mid
    return {"T_c": 0.5 * (lo + hi), "bracket": (lo, hi), "slp_lo": True,
            "slp_hi": False, "region": region_t}


# ---------------------------------------------------------------------------
# FFH model transcriptions (the analytic-rung anchors)
# ---------------------------------------------------------------------------


def ffh_pair_hamiltonian(kappa):
    """FFH Eq. (15): H = kappa sx(x)sx + sz(x)I + I(x)sz (kappa > 0)."""
    kappa = float(kappa)
    if not kappa > 0.0:
        raise ValueError(f"need kappa > 0, got {kappa}")
    return (kappa * np.kron(_SX, _SX) + np.kron(_SZ, _I2) + np.kron(_I2, _SZ))


def ffh_pair_eigenenergies(kappa):
    """FFH Eq. (16): (E_0, E_1, E_2, E_3) = (-m, -kappa, kappa, m), m = sqrt(kappa^2+4)."""
    kappa = float(kappa)
    m = np.hypot(kappa, 2.0)
    return np.array([-m, -kappa, kappa, m])


def ffh_pair_eigenmixture(kappa, populations):
    """Eigenmixture of the Eq. (15) pair, populations ordered by ascending E_k.

    The Eq. (16) spectrum is nondegenerate for kappa > 0 (m > kappa always),
    so the eigh basis is unambiguous (phases drop out of |E_k><E_k|).
    """
    return eigenmixture(ffh_pair_hamiltonian(kappa), populations)


def ffh_pair_gibbs_populations(kappa, T):
    """Gibbs populations of the Eq. (15) pair at temperature T (k_B = 1).

    Boltzmann weights on the Eq. (16) energies. FFH Eq. (23) gives the
    equivalent closed form for the differences that enter Eq. (20),
    delta_0 = (2/Z) sinh(m/T), delta_1 = (2/Z) sinh(kappa/T),
    Z = 2(cosh(kappa/T) + cosh(m/T)) — cross-checked in the tests.
    """
    E = ffh_pair_eigenenergies(kappa)
    T = float(T)
    if not T > 0.0:
        raise ValueError(f"need T > 0 (np.inf allowed), got T={T}")
    if np.isinf(T):
        return np.full(4, 0.25)
    w = np.exp(-(E - E[0]) / T)
    return w / float(np.sum(w))


def ffh_pair_local_energy(kappa, populations):
    """Closed-form local energy Omega_o of one particle of the Eq. (15) pair.

    FFH Eqs. (20)-(21) for an eigenmixture with populations p_k (ascending
    energy order): with m = sqrt(kappa^2 + 4),

        delta_0 = p_0 - p_3,   delta_1 = p_1 - p_2,
        eta = (2/m) delta_0,   xi = (kappa^2/m) delta_0 + kappa delta_1,  [(20)]

        Omega_o = sqrt((1 - eta^2 + xi^2)/(1 - eta^2)) - xi - eta
                                             if |eta xi| < 1 - eta^2,   [(21)]
                  |xi| + |eta| - xi - eta     otherwise.

    Omega_o >= 0 always, and the eigenmixture is SLP iff Omega_o = 0. The two
    particles are equivalent (Eq. (15) is swap-symmetric). Note |eta| <= 2/m
    < 1, so the 1 - eta^2 denominator never vanishes. Sanity: the maximally
    mixed state has eta = xi = 0 and Omega_o = 1, which is right — resetting
    one spin to |sz = -1> against the sz(x)I term extracts exactly 1.
    """
    kappa = float(kappa)
    p = _check_populations(populations, 4)
    m = np.hypot(kappa, 2.0)
    delta0 = p[0] - p[3]
    delta1 = p[1] - p[2]
    eta = 2.0 * delta0 / m                              # Eq. (20)
    xi = (kappa * kappa / m) * delta0 + kappa * delta1  # Eq. (20)
    if abs(eta * xi) < 1.0 - eta * eta:                 # Eq. (21), first branch
        return float(np.sqrt((1.0 - eta * eta + xi * xi) / (1.0 - eta * eta))
                     - xi - eta)
    return float(abs(xi) + abs(eta) - xi - eta)         # Eq. (21), second branch


def xxx_pair_hamiltonian():
    """FFH Eq. (33): the Heisenberg XXX pair H_X = sx(x)sx + sy(x)sy + sz(x)sz."""
    return (np.kron(_SX, _SX) + np.kron(_SY, _SY) + np.kron(_SZ, _SZ))


def xxx_pair_eigenbasis():
    """Eigenenergies and eigenvectors of the XXX pair, singlet first.

    E_0 = -3 (singlet) and E_1 = E_2 = E_3 = +1 (triplet), with

        |E_0> = (|10> - |01>)/sqrt(2),   |E_1> = (|10> + |01>)/sqrt(2),
        |E_2> = |00>,                    |E_3> = |11>.

    Returns (energies (4,), vectors (4, 4) with column k = |E_k>). The triplet
    level is degenerate, so an explicit basis is needed whenever the level
    split matters — that is what :func:`xxx_pair_eigenmixture` uses.
    """
    energies = np.array([-3.0, 1.0, 1.0, 1.0])
    vectors = np.zeros((4, 4), dtype=complex)
    s = 1.0 / np.sqrt(2.0)
    vectors[2, 0], vectors[1, 0] = s, -s  # |E_0> = (|10> - |01>)/sqrt(2)
    vectors[2, 1], vectors[1, 1] = s, s   # |E_1> = (|10> + |01>)/sqrt(2)
    vectors[0, 2] = 1.0                   # |E_2> = |00>
    vectors[3, 3] = 1.0                   # |E_3> = |11>
    return energies, vectors


def xxx_pair_eigenmixture(populations):
    """rho = sum_k p_k |E_k><E_k| in the explicit XXX eigenbasis above."""
    p = _check_populations(populations, 4)
    _, V = xxx_pair_eigenbasis()
    return (V * p) @ V.conj().T
