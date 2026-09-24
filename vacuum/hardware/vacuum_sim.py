"""Qubit encoding of the lattice scalar-field vacuum (Layer 7, L6).

"Hardware as noisy field": the Layer-1 harmonic chain

    H = (1/2) sum_i [ p_i^2 + m^2 x_i^2 + (x_{i+1} - x_i)^2 ]
      = (1/2) p.p + (1/2) x^T K x,        K = harmonic_chain_K(N, m, bc)

is put on qubits so that its ground state — the lattice vacuum whose exact
covariance is ``vacuum.core.ground_state_cov(K)`` — can be prepared
variationally on a gate-model device and then used as the medium for the
minimal protocols of :mod:`vacuum.hardware.protocols`.

The encoding (stated once, used everywhere)
-------------------------------------------
Each oscillator i is expanded in the Fock basis of a *local* reference
oscillator of frequency omega_i (default omega_i = sqrt(K_ii), the on-site
frequency) and **truncated to its lowest d levels**, d = 2^q, so a site is
q qubits.  This is the harmonic-oscillator-basis digitization of Klco &
Savage, Phys. Rev. A 99, 052335 (2019) [arXiv:1808.10378], Sec. II
(oscillator eigenstates as the field-space basis) and the Fock-cutoff boson
representation of Macridin, Spentzouris, Amundson, Harnik, Phys. Rev.
Lett. 121, 110504 (2018) [arXiv:1802.07347].  The truncation is the
**Galerkin projection** H_d = P H P with P the projector onto the first d
Fock levels of every site: every operator entering H is projected *after*
being formed, so

    x_d   = P x P     = (a_d + a_d^dag) / sqrt(2 omega)
    x2_d  = P x^2 P   = (a_d^2 + a_d^dag^2 + 2 n_d + 1) / (2 omega)
    p2_d  = P p^2 P   = omega (2 n_d + 1 - a_d^2 - a_d^dag^2) / 2

with a_d the d x d truncated ladder operator.  (x2_d differs from (x_d)^2
only in the top diagonal entry; the Galerkin choice makes E_0(d) an
*upper* bound on the exact ground energy that decreases monotonically in d
— the Rayleigh–Ritz principle on nested subspaces.)  At omega_i =
sqrt(K_ii) the on-site term is exactly omega_i (n_i + 1/2) at every d, so
the truncation error lives entirely in the bond terms K_ij x_i x_j.

Consequences worth knowing:

* d = 2 (one qubit per site): x_d = sigma_x / sqrt(2 omega), p_d =
  sqrt(omega/2) sigma_y, n = (1 - sigma_z)/2, so the encoded chain is a
  transverse-field-Ising-type model  H_2 = sum_i omega_i (n_i + 1/2)
  + sum_{i<j} K_ij sigma_x^i sigma_x^j / (2 sqrt(omega_i omega_j)).
  For N = 2 this is *exactly* Hotta's minimal QET model
  (:mod:`vacuum.qet.hotta`) with h = omega/2 and k = -K_01/(4 omega) up
  to local frame flips — the closed forms of that module become an
  anchor for :mod:`vacuum.hardware.protocols`.
* The encoded vacuum approaches the Gaussian one exponentially in d:
  :func:`encoding_convergence` tabulates |E_0(d) - Tr(K^{1/2})/2| and the
  max-norm distance of the truncated covariance to ``ground_state_cov(K)``.

Basis ordering (binding for this package): site i occupies qubits
[q i, ..., q i + q - 1]; the Fock index of site i is n_i = sum_b 2^b
bit_{q i + b}; the dense basis index is sum_i n_i d^i, i.e. **site 0 is the
least significant digit** — identical to qiskit's little-endian statevector
index, so dense states and Aer statevectors/density matrices compare with
NO permutation.  Pauli labels follow qiskit (leftmost character = highest
qubit).

Variational preparation
-----------------------
:func:`vqe_ground_state` runs a hardware-efficient ansatz — layers of R_Y
rotations separated by a linear CNOT ladder, the real-amplitude member of
the family of Kandala et al., Nature 549, 242 (2017) (H_d is real
symmetric, so its ground state is real and R_Y rotations suffice) — with
the energy evaluated on **Aer's statevector simulator** (batched
``parameter_binds``, one job per gradient), exact parameter-shift
gradients, L-BFGS-B, and seeded random restarts.  A pure-numpy statevector
of the same circuit is kept as the arbiter (Aer must reproduce it to
1e-12) and as the fast evaluator for tests.  Every VQE result carries its
energy error against the exact-diagonalization ground energy of the SAME
truncated Hamiltonian, the covariance error against the ED state, and the
infidelity.

Expressibility versus optimization (the (3,4) / (4,4) rows)
-------------------------------------------------------------
Whether an unconverged VQE row is the optimizer's fault or the ansatz's is
settled by :func:`fidelity_ceiling`: with the ED state in hand, the state
fidelity |<psi_ED|psi(theta)>|^2 is a smooth objective with an exact
reverse-mode (adjoint) gradient (:func:`hea_overlap_and_grad`, two
statevector passes per gradient), maximised from many seeded starts.  The
best fidelity per depth is a lower bound on the ansatz's expressibility
ceiling at that depth; :func:`fidelity_depth_scan` grows the depth,
warm-starting each depth from the previous optimum through
:func:`hea_embed_layers` (new layers prepended with zero angles fix
|0...0> exactly, so the embedded start reproduces the previous fidelity).
:func:`vqe_ground_state` then accepts ``warm_start`` (the fidelity optimum,
a layerwise scan, or a given vector) and ``gradient='adjoint'`` (the numpy
arbiter's exact energy gradient, :func:`hea_energy_and_grad`) so the
energy-VQE at the minimal sufficient depth polishes to roundoff.

qiskit is imported only inside :func:`_qiskit` (the ``.[ibm]`` extra);
everything up to and including exact diagonalization is core-only.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.sparse.linalg import eigsh

from vacuum.core import ground_state_cov, harmonic_chain_K, mean_energy

__all__ = [
    "PAULIS",
    "TruncatedChain",
    "truncated_chain",
    "site_operators",
    "embed_site",
    "hamiltonian_dense",
    "touched_hamiltonian_dense",
    "exact_ground_state",
    "covariance_from_state",
    "gaussian_reference",
    "encoding_convergence",
    "PauliTerm",
    "pauli_decompose",
    "site_pauli_terms",
    "hamiltonian_pauli_terms",
    "touched_pauli_terms",
    "pauli_term_dense",
    "pauli_terms_dense",
    "pauli_label",
    "commuting_groups",
    "hea_parameter_count",
    "hea_statevector",
    "hea_overlap_and_grad",
    "hea_energy_and_grad",
    "hea_state_jacobian",
    "hea_embed_layers",
    "FidelityResult",
    "fidelity_ceiling",
    "fidelity_depth_scan",
    "ansatz_energy",
    "VQEResult",
    "vqe_ground_state",
    "state_fidelity",
]

#: single-qubit Paulis in the computational basis
PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _qiskit():
    """Lazy, guarded qiskit + qiskit-aer import (the ``.[ibm]`` extra)."""
    try:
        from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp, Statevector
        from qiskit_aer import AerSimulator
        import qiskit_aer.library  # noqa: F401 (save_* instructions)
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise ImportError(
            "vacuum.hardware needs qiskit and qiskit-aer — install the '.[ibm]' "
            "extra; the dense/ED path of vacuum.hardware.vacuum_sim is core-only"
        ) from exc
    return SimpleNamespace(
        QuantumCircuit=QuantumCircuit,
        QuantumRegister=QuantumRegister,
        ClassicalRegister=ClassicalRegister,
        transpile=transpile,
        ParameterVector=ParameterVector,
        SparsePauliOp=SparsePauliOp,
        Statevector=Statevector,
        AerSimulator=AerSimulator,
    )


# ---------------------------------------------------------------------------
# the truncated chain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TruncatedChain:
    """A harmonic chain truncated to d Fock levels per site (see module doc).

    Attributes
    ----------
    N, m, bc : the Layer-1 chain parameters (``harmonic_chain_K``).
    d : Fock levels per site (a power of two for the qubit encoding).
    K : (N, N) coupling matrix.
    omega : (N,) local reference frequencies (default sqrt(diag K)).
    q : qubits per site, log2(d).
    n_qubits : N q.
    dim : d^N.
    """

    N: int
    m: float
    bc: str
    d: int
    K: np.ndarray
    omega: np.ndarray

    @property
    def q(self) -> int:
        return int(round(math.log2(self.d)))

    @property
    def n_qubits(self) -> int:
        return self.N * self.q

    @property
    def dim(self) -> int:
        return self.d ** self.N

    def site_qubits(self, site) -> List[int]:
        """The q qubits of ``site`` (little-endian within the site)."""
        return [self.q * int(site) + b for b in range(self.q)]

    def neighbours(self, site) -> List[int]:
        """Sites j != site with K[site, j] != 0."""
        return [int(j) for j in range(self.N) if j != site and self.K[site, j] != 0.0]


def truncated_chain(N, m=1.0, d=2, bc="dirichlet", omega=None):
    """Build a :class:`TruncatedChain`.

    Parameters
    ----------
    N : int  (2-4 is the hardware regime; ED runs to ~12 qubits)
    m : float  mass in lattice units (Dirichlet allows m = 0).
    d : int  Fock levels per site; must be a power of two >= 2 for the qubit
        encoding (:func:`hamiltonian_dense` and the ED accept any d >= 2).
    bc : 'dirichlet' | 'periodic'
    omega : float or (N,) array, optional
        Local reference frequencies of the Fock bases; default sqrt(K_ii).
    """
    N = int(N)
    d = int(d)
    if d < 2:
        raise ValueError(f"need d >= 2 Fock levels, got {d}")
    K = harmonic_chain_K(N, float(m), bc=bc)
    if omega is None:
        om = np.sqrt(np.diag(K))
    else:
        om = np.broadcast_to(np.asarray(omega, dtype=float), (N,)).copy()
    if np.any(om <= 0.0):
        raise ValueError("reference frequencies must be positive")
    return TruncatedChain(N=N, m=float(m), bc=str(bc), d=d, K=K, omega=om)


def site_operators(d, omega):
    """Truncated single-site operators in the d-level Fock basis of frequency omega.

    Returns dict with ``a`` (ladder), ``n``, ``x`` (= P x P), ``p`` (= P p P),
    ``x2`` (= P x^2 P), ``p2`` (= P p^2 P), ``h_site`` (= P (p^2/2 +
    omega^2 x^2/2) P = omega (n + 1/2)), all (d, d) complex ndarrays.
    """
    d = int(d)
    omega = float(omega)
    a = np.zeros((d, d), dtype=complex)
    for n in range(d - 1):
        a[n, n + 1] = math.sqrt(n + 1.0)
    ad = a.conj().T
    n_op = ad @ a
    I = np.eye(d, dtype=complex)
    a2 = a @ a
    ad2 = ad @ ad
    x = (a + ad) / math.sqrt(2.0 * omega)
    p = 1.0j * math.sqrt(omega / 2.0) * (ad - a)
    x2 = (a2 + ad2 + 2.0 * n_op + I) / (2.0 * omega)
    p2 = omega * (2.0 * n_op + I - a2 - ad2) / 2.0
    return {
        "a": a, "n": n_op, "x": x, "p": p, "x2": x2, "p2": p2,
        "h_site": 0.5 * p2 + 0.5 * omega * omega * x2,
    }


def embed_site(op, site, N, d):
    """Embed a (d, d) site operator into the d^N dense space, site 0 least significant."""
    site = int(site)
    left = np.eye(d ** (N - 1 - site), dtype=complex)
    right = np.eye(d ** site, dtype=complex)
    return np.kron(np.kron(left, np.asarray(op, dtype=complex)), right)


def _two_site(opA, i, opB, j, N, d):
    """opA on site i times opB on site j (i != j), dense."""
    return embed_site(opA, i, N, d) @ embed_site(opB, j, N, d)


def hamiltonian_dense(chain: TruncatedChain):
    """The Galerkin-truncated Hamiltonian H_d = P H P as a dense (d^N, d^N) matrix."""
    N, d = chain.N, chain.d
    ops = [site_operators(d, chain.omega[i]) for i in range(N)]
    H = np.zeros((chain.dim, chain.dim), dtype=complex)
    for i in range(N):
        H += embed_site(0.5 * ops[i]["p2"] + 0.5 * chain.K[i, i] * ops[i]["x2"], i, N, d)
        for j in range(i + 1, N):
            if chain.K[i, j] != 0.0:
                H += chain.K[i, j] * _two_site(ops[i]["x"], i, ops[j]["x"], j, N, d)
    return H


def touched_hamiltonian_dense(chain: TruncatedChain, site):
    """The terms of H_d that act on ``site``: (1/2) p_s^2 + (1/2) K_ss x_s^2
    + sum_{j != s} K_sj x_s x_j (full bonds).  The change of <H_d> under any
    operation local to ``site`` equals the change of this operator."""
    N, d = chain.N, chain.d
    s = int(site)
    ops = [site_operators(d, chain.omega[i]) for i in range(N)]
    H = embed_site(0.5 * ops[s]["p2"] + 0.5 * chain.K[s, s] * ops[s]["x2"], s, N, d)
    for j in chain.neighbours(s):
        H += chain.K[s, j] * _two_site(ops[s]["x"], s, ops[j]["x"], j, N, d)
    return H


def exact_ground_state(chain: TruncatedChain, dense_max=4096, k=1):
    """Exact diagonalization of H_d: returns (E0, psi0) with psi0 real, sign-fixed.

    Dense ``eigh`` up to ``dense_max`` states, Lanczos (``eigsh``) above.
    The returned vector has its largest-magnitude component positive.
    """
    H = hamiltonian_dense(chain)
    if chain.dim <= dense_max:
        w, U = np.linalg.eigh(H)
        E0 = float(w[0])
        psi = U[:, 0]
    else:
        from scipy.sparse import csr_matrix

        w, U = eigsh(csr_matrix(H), k=max(k, 1), which="SA")
        i = int(np.argmin(w))
        E0 = float(w[i])
        psi = U[:, i]
    psi = np.asarray(psi, dtype=complex)
    j = int(np.argmax(np.abs(psi)))
    psi = psi * (np.abs(psi[j]) / psi[j])
    if np.max(np.abs(psi.imag)) < 1e-12:
        psi = psi.real.astype(complex)
    return E0, psi


def _expect(op, state):
    state = np.asarray(state, dtype=complex)
    if state.ndim == 1:
        return float(np.real(np.vdot(state, op @ state)))
    return float(np.real(np.trace(op @ state)))


def covariance_from_state(chain: TruncatedChain, state):
    """Truncated-model covariance V_ij = (1/2)<{R_i, R_j}> - <R_i><R_j>,
    R = (x_1..x_N, p_1..p_N), using the Galerkin operators: x2_d / p2_d on
    the diagonal, x_d x_d / p_d p_d and (x_d p_d + p_d x_d)/2 off it.
    ``state`` is a (dim,) vector or (dim, dim) density matrix.  Returns
    (V, means) with V of shape (2N, 2N) in docs/API.md block ordering."""
    N, d = chain.N, chain.d
    ops = [site_operators(d, chain.omega[i]) for i in range(N)]
    X = [embed_site(ops[i]["x"], i, N, d) for i in range(N)]
    P = [embed_site(ops[i]["p"], i, N, d) for i in range(N)]
    R = X + P
    means = np.array([_expect(r, state) for r in R])
    V = np.zeros((2 * N, 2 * N))
    for a in range(2 * N):
        for b in range(a, 2 * N):
            if a == b:
                i = a % N
                op2 = ops[i]["x2"] if a < N else ops[i]["p2"]
                val = _expect(embed_site(op2, i, N, d), state)
            else:
                val = 0.5 * _expect(R[a] @ R[b] + R[b] @ R[a], state)
            V[a, b] = V[b, a] = val - means[a] * means[b]
    return V, means


def gaussian_reference(chain: TruncatedChain):
    """The exact (untruncated) vacuum: ``ground_state_cov(K)`` and its energy
    Tr(K^{1/2})/2 from :mod:`vacuum.core`."""
    V = ground_state_cov(chain.K)
    return V, float(mean_energy(V, chain.K))


def encoding_convergence(N, m=1.0, ds=(2, 4, 8, 16), bc="dirichlet", omega=None,
                         dense_max=4096):
    """Convergence of the truncated vacuum to the Gaussian vacuum as d grows.

    Returns a list of rows (one per d): ``d``, ``n_qubits`` (None if d is not
    a power of two), ``E0_trunc``, ``E0_exact``, ``dE`` = E0_trunc -
    E0_exact (>= 0 by Rayleigh–Ritz), ``cov_err`` = max |V_trunc - V_exact|,
    ``cov_err_xx``, ``cov_err_pp``, ``dim``.
    """
    rows = []
    for d in ds:
        ch = truncated_chain(N, m, d, bc=bc, omega=omega)
        E0, psi = exact_ground_state(ch, dense_max=dense_max)
        V, _ = covariance_from_state(ch, psi)
        Vg, Eg = gaussian_reference(ch)
        diff = np.abs(V - Vg)
        rows.append({
            "d": int(d),
            "n_qubits": ch.n_qubits if 2 ** ch.q == d else None,
            "E0_trunc": E0,
            "E0_exact": Eg,
            "dE": E0 - Eg,
            "cov_err": float(diff.max()),
            "cov_err_xx": float(diff[:N, :N].max()),
            "cov_err_pp": float(diff[N:, N:].max()),
            "dim": ch.dim,
        })
    return rows


# ---------------------------------------------------------------------------
# Pauli-string algebra (qiskit-free)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PauliTerm:
    """coeff * prod_q P_q with ``paulis`` a tuple of (qubit, 'X'|'Y'|'Z')."""

    paulis: Tuple[Tuple[int, str], ...]
    coeff: complex

    @property
    def qubits(self):
        return tuple(q for q, _ in self.paulis)

    def is_identity(self):
        return len(self.paulis) == 0


def pauli_decompose(M, tol=1e-14):
    """Decompose a (2^q, 2^q) matrix into Pauli strings.

    Returns list of (label, coeff) with label a q-character string, leftmost
    character = highest qubit (qiskit convention); coefficients with
    |coeff| <= tol are dropped.
    """
    M = np.asarray(M, dtype=complex)
    dim = M.shape[0]
    q = int(round(math.log2(dim)))
    if 2 ** q != dim:
        raise ValueError(f"matrix dimension {dim} is not a power of two")
    out = []
    for labels in itertools.product("IXYZ", repeat=q):
        P = np.array([[1.0]], dtype=complex)
        for ch in labels:                      # leftmost = highest qubit
            P = np.kron(P, PAULIS[ch])
        c = np.trace(P.conj().T @ M) / dim
        if abs(c) > tol:
            out.append(("".join(labels), complex(c)))
    return out


def _site_terms(M, chain: TruncatedChain, site, tol=1e-14):
    """PauliTerms of a single-site operator M embedded on ``site``."""
    qubits = chain.site_qubits(site)
    q = chain.q
    terms = []
    for label, c in pauli_decompose(M, tol):
        paulis = tuple((qubits[q - 1 - k], ch) for k, ch in enumerate(label) if ch != "I")
        terms.append(PauliTerm(paulis=tuple(sorted(paulis)), coeff=c))
    return terms


def _multiply_disjoint(t1: PauliTerm, t2: PauliTerm):
    """Product of two PauliTerms on disjoint qubit sets."""
    return PauliTerm(paulis=tuple(sorted(t1.paulis + t2.paulis)), coeff=t1.coeff * t2.coeff)


def _merge(terms, tol=1e-15):
    acc: Dict[Tuple[Tuple[int, str], ...], complex] = {}
    for t in terms:
        acc[t.paulis] = acc.get(t.paulis, 0.0) + t.coeff
    return [PauliTerm(p, c) for p, c in acc.items() if abs(c) > tol]


def site_pauli_terms(chain: TruncatedChain, site, which):
    """PauliTerms of a truncated site operator (``which`` in x, p, x2, p2, n, h_site)."""
    ops = site_operators(chain.d, chain.omega[int(site)])
    return _site_terms(ops[which], chain, site)


def hamiltonian_pauli_terms(chain: TruncatedChain):
    """H_d as a list of :class:`PauliTerm` (merged, identity term included)."""
    N = chain.N
    terms = []
    for i in range(N):
        ops = site_operators(chain.d, chain.omega[i])
        terms += _site_terms(0.5 * ops["p2"] + 0.5 * chain.K[i, i] * ops["x2"], chain, i)
        xi = _site_terms(ops["x"], chain, i)
        for j in range(i + 1, N):
            if chain.K[i, j] != 0.0:
                xj = site_pauli_terms(chain, j, "x")
                for a in xi:
                    for b in xj:
                        terms.append(_multiply_disjoint(a, PauliTerm(b.paulis, b.coeff * chain.K[i, j])))
    return _merge(terms)


def touched_pauli_terms(chain: TruncatedChain, site, x_override=None):
    """PauliTerms of :func:`touched_hamiltonian_dense` (the terms acting on ``site``).

    ``x_override`` = {site_j: (d, d) matrix} replaces x_j on the listed
    neighbour sites — used to write Bob's touched operator in Alice's
    measurement frame, where x_A becomes its eigenvalue diagonal.
    """
    s = int(site)
    x_override = dict(x_override or {})
    ops = site_operators(chain.d, chain.omega[s])
    terms = _site_terms(0.5 * ops["p2"] + 0.5 * chain.K[s, s] * ops["x2"], chain, s)
    xs = _site_terms(ops["x"], chain, s)
    for j in chain.neighbours(s):
        if j in x_override:
            xj = _site_terms(np.asarray(x_override[j], dtype=complex), chain, j)
        else:
            xj = site_pauli_terms(chain, j, "x")
        for a in xs:
            for b in xj:
                terms.append(_multiply_disjoint(a, PauliTerm(b.paulis, b.coeff * chain.K[s, j])))
    return _merge(terms)


def pauli_label(term: PauliTerm, n_qubits):
    """qiskit label of a term: leftmost character = qubit n_qubits - 1."""
    chars = ["I"] * int(n_qubits)
    for qb, ch in term.paulis:
        chars[n_qubits - 1 - qb] = ch
    return "".join(chars)


def pauli_term_dense(term: PauliTerm, n_qubits):
    """Dense matrix of a term on n qubits, qubit 0 least significant."""
    M = np.array([[1.0]], dtype=complex)
    lab = pauli_label(term, n_qubits)
    for ch in lab:
        M = np.kron(M, PAULIS[ch])
    return term.coeff * M


def pauli_terms_dense(terms, n_qubits):
    dim = 2 ** int(n_qubits)
    H = np.zeros((dim, dim), dtype=complex)
    for t in terms:
        H += pauli_term_dense(t, n_qubits)
    return H


def _qubitwise_commute(t1: PauliTerm, t2: PauliTerm):
    d1 = dict(t1.paulis)
    for qb, ch in t2.paulis:
        if qb in d1 and d1[qb] != ch:
            return False
    return True


def commuting_groups(terms):
    """Greedy partition into qubit-wise commuting groups (measurable in one
    basis each; their exponentials multiply exactly)."""
    groups: List[List[PauliTerm]] = []
    for t in terms:
        if t.is_identity():
            continue
        for g in groups:
            if all(_qubitwise_commute(t, u) for u in g):
                g.append(t)
                break
        else:
            groups.append([t])
    return groups


# ---------------------------------------------------------------------------
# hardware-efficient ansatz, numpy arbiter
# ---------------------------------------------------------------------------


def hea_parameter_count(n_qubits, layers):
    return int(n_qubits) * (int(layers) + 1)


def _apply_1q(psi, U, qb, n):
    t = psi.reshape((2,) * n)
    ax = n - 1 - qb
    t = np.moveaxis(t, ax, 0)
    t = np.tensordot(U, t, axes=([1], [0]))
    t = np.moveaxis(t, 0, ax)
    return t.reshape(-1)


def _apply_cx(psi, c, tq, n):
    t = psi.reshape((2,) * n).copy()
    ac, at = n - 1 - c, n - 1 - tq
    idx = [slice(None)] * n
    idx[ac] = 1
    sub = t[tuple(idx)]
    sub = np.flip(sub, axis=at if at < ac else at - 1)
    t[tuple(idx)] = sub
    return t.reshape(-1)


def _ry(theta):
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def hea_statevector(params, n_qubits, layers):
    """numpy statevector of the R_Y / linear-CNOT ansatz (same gate order as
    :func:`hea_circuit`): for l = 0..layers: R_Y(params[l n + q]) on every
    qubit, then (l < layers) CX(q, q+1) for q = 0..n-2."""
    n = int(n_qubits)
    params = np.asarray(params, dtype=float)
    if params.size != hea_parameter_count(n, layers):
        raise ValueError(f"expected {hea_parameter_count(n, layers)} parameters, got {params.size}")
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    for layer in range(int(layers) + 1):
        for qb in range(n):
            psi = _apply_1q(psi, _ry(params[layer * n + qb]), qb, n)
        if layer < layers:
            for qb in range(n - 1):
                psi = _apply_cx(psi, qb, qb + 1, n)
    return psi


def hea_circuit(n_qubits, layers, name="hea"):
    """The ansatz as a parameterised ``QuantumCircuit`` (ParameterVector 'th')."""
    Q = _qiskit()
    n = int(n_qubits)
    th = Q.ParameterVector("th", hea_parameter_count(n, layers))
    qc = Q.QuantumCircuit(n, name=name)
    for layer in range(int(layers) + 1):
        for qb in range(n):
            qc.ry(th[layer * n + qb], qb)
        if layer < layers:
            for qb in range(n - 1):
                qc.cx(qb, qb + 1)
    return qc, th


def state_fidelity(psi, phi):
    """|<psi|phi>|^2 for unit vectors."""
    return float(abs(np.vdot(psi, phi)) ** 2)


# ---------------------------------------------------------------------------
# exact gradients (adjoint differentiation), layerwise embedding, the
# expressibility ceiling
# ---------------------------------------------------------------------------


def _gate_sequence(n, layers):
    """The ansatz as ('ry', qubit, parameter index) / ('cx', control, target)
    triples in circuit order (the order of :func:`hea_statevector`)."""
    n = int(n)
    layers = int(layers)
    seq = []
    for layer in range(layers + 1):
        for qb in range(n):
            seq.append(("ry", qb, layer * n + qb))
        if layer < layers:
            for qb in range(n - 1):
                seq.append(("cx", qb, qb + 1))
    return seq


def _adjoint_overlap_grad(params, n, layers, psi, phi):
    """d<phi|psi(theta)>/dtheta_k for every k by reverse-mode (adjoint)
    differentiation: walk the gates backwards, un-applying each from both
    |psi> and |lambda> = (later gates)^dag |phi>; for R_Y(theta) =
    exp(-i theta Y/2), dR/dtheta = -(i/2) Y R, so the k-th component is
    <lambda_k| (-(i/2) Y_q) |psi_k> with psi_k the state just after gate k.
    Two statevector passes per gradient instead of 2 n_params + 1 circuit
    evaluations; exact to roundoff."""
    grad = np.zeros(params.size, dtype=complex)
    lam = np.asarray(phi, dtype=complex).copy()
    psi = np.asarray(psi, dtype=complex)
    Y = PAULIS["Y"]
    for kind, a, b in reversed(_gate_sequence(n, layers)):
        if kind == "ry":
            grad[b] = -0.5j * np.vdot(lam, _apply_1q(psi, Y, a, n))
            Rinv = _ry(-params[b])
            psi = _apply_1q(psi, Rinv, a, n)
            lam = _apply_1q(lam, Rinv, a, n)
        else:
            psi = _apply_cx(psi, a, b, n)
            lam = _apply_cx(lam, a, b, n)
    return grad


def hea_overlap_and_grad(params, n_qubits, layers, phi):
    """<phi|psi(theta)> and its exact gradient (complex, shape (n_params,))
    by adjoint differentiation through :func:`hea_statevector`."""
    n = int(n_qubits)
    params = np.asarray(params, dtype=float)
    psi = hea_statevector(params, n, layers)
    phi = np.asarray(phi, dtype=complex)
    return complex(np.vdot(phi, psi)), _adjoint_overlap_grad(params, n, int(layers), psi, phi)


def hea_energy_and_grad(params, n_qubits, layers, H):
    """<psi(theta)|H|psi(theta)> and its exact gradient
    2 Re <H psi|dpsi/dtheta> (real, shape (n_params,)) — the numpy
    arbiter's analytic energy gradient (H Hermitian, dense)."""
    n = int(n_qubits)
    params = np.asarray(params, dtype=float)
    psi = hea_statevector(params, n, layers)
    Hpsi = np.asarray(H) @ psi
    E = float(np.real(np.vdot(psi, Hpsi)))
    g = _adjoint_overlap_grad(params, n, int(layers), psi, Hpsi)
    return E, 2.0 * np.real(g)


def hea_state_jacobian(params, n_qubits, layers):
    """d psi(theta) / d theta as a (2^n, n_params) complex matrix.  Column k is
    psi(theta + pi e_k) / 2 — the shift rule for the AMPLITUDE (not the
    expectation value): R_Y(theta + pi) = 2 dR_Y/dtheta and the other gates
    are unchanged.  Exact to roundoff; n_params statevector evaluations."""
    n = int(n_qubits)
    params = np.asarray(params, dtype=float)
    J = np.empty((2 ** n, params.size), dtype=complex)
    for k in range(params.size):
        shifted = params.copy()
        shifted[k] += math.pi
        J[:, k] = 0.5 * hea_statevector(shifted, n, layers)
    return J


def hea_embed_layers(params, n_qubits, layers_from, layers_to):
    """Embed the parameters of an ``layers_from``-layer ansatz into a
    ``layers_to``-layer one (>=) WITHOUT changing the prepared state: the
    new layers are prepended with zero angles — R_Y(0) = 1 and the CNOT
    ladder fixes |0...0>, so the old circuit then acts on |0...0> exactly as
    before.  (Appending layers would not be an embedding: the ladder does
    not fix a generic state.)  This is the layerwise-growing warm start of
    :func:`fidelity_depth_scan`."""
    n = int(n_qubits)
    lf, lt = int(layers_from), int(layers_to)
    params = np.asarray(params, dtype=float)
    if params.size != hea_parameter_count(n, lf):
        raise ValueError(f"expected {hea_parameter_count(n, lf)} parameters, got {params.size}")
    if lt < lf:
        raise ValueError(f"layers_to = {lt} < layers_from = {lf}")
    return np.concatenate([np.zeros(n * (lt - lf)), params])


@dataclass
class FidelityResult:
    """Outcome of :func:`fidelity_ceiling`.

    fidelity / infidelity : the best |<psi_ED|psi(theta)>|^2 over the starts
        and 1 minus it (a LOWER bound on the expressibility ceiling at this
        depth; an UPPER bound on the attainable infidelity).
    params / state : the best parameters and the prepared state (sign-aligned
        to ED).
    energy / E0_exact / energy_error / cov_error : the energy of that state
        (numpy), the ED energy, their difference, and the max-norm covariance
        error against the ED state.
    grad_norm : 2-norm of the infidelity gradient at the optimum.
    restarts / restarts_run : starts prepared / actually run (``stop_at``).
    best_restart / best_start : index and kind ('normal', 'uniform', or the
        label of a given start such as 'layerwise') of the winning start.
    restart_infidelities : the final infidelity of every start run.
    n_iterations / n_evaluations : L-BFGS-B iterations and objective
        evaluations summed over the starts.
    seed, wall_s, history (one dict per start: infidelity, nit, nfev,
        grad_max, success, message).
    """

    layers: int
    n_params: int
    fidelity: float
    infidelity: float
    params: np.ndarray
    state: np.ndarray
    energy: float
    E0_exact: float
    energy_error: float
    cov_error: float
    grad_norm: float
    restarts: int
    restarts_run: int
    best_restart: int
    best_start: str
    restart_infidelities: List[float]
    n_iterations: int
    n_evaluations: int
    seed: int
    wall_s: float
    history: List[Dict[str, Any]] = field(default_factory=list)


def _fidelity_starts(n_par, restarts, seed, init_scale, wide_fraction, init_params):
    rng = np.random.default_rng(int(seed))
    starts = []
    if init_params is not None:
        given = init_params if isinstance(init_params, dict) else {"given": init_params}
        for label, p in given.items():
            p = np.asarray(p, dtype=float).copy()
            if p.size != n_par:
                raise ValueError(f"start {label!r}: expected {n_par} parameters, got {p.size}")
            starts.append((str(label), p))
    restarts = int(restarts)
    n_wide = int(round(restarts * float(wide_fraction)))
    for r in range(restarts):
        if r < restarts - n_wide:
            starts.append(("normal", rng.normal(0.0, float(init_scale), size=n_par)))
        else:
            starts.append(("uniform", rng.uniform(-math.pi, math.pi, size=n_par)))
    return starts


def fidelity_ceiling(chain: TruncatedChain, layers, restarts=20, seed=0, maxiter=5000, gtol=1e-12,
                     init_scale=0.3, wide_fraction=0.5, init_params=None, stop_at=None, target=None,
                     method="lm", perturb_rounds=0, perturb_scale=0.2):
    """Expressibility ceiling of the hardware-efficient ansatz at ``layers``
    for the ED ground state of H_d: maximise the state fidelity
    |<psi_ED|psi(theta)>|^2 directly (no Hamiltonian evaluation at all)
    from seeded random starts and any given ones, by

    * ``method='lm'`` (default): Levenberg–Marquardt (``scipy.optimize
      .least_squares``) on the state residual psi(theta) - s psi_ED, s the
      sign of Re<psi_ED|psi>, with the exact Jacobian
      :func:`hea_state_jacobian` — a zero-residual least-squares problem
      whenever the depth suffices, on which Gauss–Newton steps converge
      quadratically; ``maxiter`` bounds the Jacobian evaluations;
    * ``method='lbfgs'``: L-BFGS-B on 1 - |<psi_ED|psi>|^2 with the exact
      adjoint gradient (:func:`hea_overlap_and_grad`); ``gtol`` applies.

    After the starts, ``perturb_rounds`` rounds of iterated local search:
    the incumbent plus N(0, perturb_scale^2) noise is re-optimised and
    replaces it when better (starts labelled 'perturb'); the landscape is
    glassy at depth, and this is what moves between neighbouring basins.

    Starts, in order: ``init_params`` (an array, or a dict label -> array,
    e.g. the layerwise embedding of a shallower optimum), then ``restarts``
    random points: the first restarts - round(wide_fraction x restarts) are
    N(0, init_scale^2) (near the Fock vacuum |0...0>, which dominates the
    ground state), the rest U(-pi, pi).  ``stop_at`` ends the loop after the
    first start reaching infidelity <= stop_at.  ``target`` = (E0, psi_ED)
    if already computed.

    What the result means.  The best fidelity over the starts is a lower
    bound on sup_theta F at this depth (an upper bound on the attainable
    infidelity).  A best infidelity at roundoff PROVES the depth suffices
    (the state is exhibited).  A plateau shared by every start, each at a
    converged stationary point (``history``: grad_max, message), is
    evidence — not a proof — that the depth does not: the objective is
    non-convex and a global search is not attempted.
    """
    if 2 ** chain.q != chain.d:
        raise ValueError(f"d = {chain.d} is not a power of two; no qubit encoding")
    t0 = time.time()
    n = chain.n_qubits
    L = int(layers)
    n_par = hea_parameter_count(n, L)
    H = hamiltonian_dense(chain)
    if target is None:
        E0, psi_ed = exact_ground_state(chain)
    else:
        E0, psi_ed = target
    psi_ed = np.asarray(psi_ed, dtype=complex)
    starts = _fidelity_starts(n_par, restarts, seed, init_scale, wide_fraction, init_params)
    evals = 0
    history = []
    best = None
    n_iter = 0

    if method not in ("lm", "lbfgs"):
        raise ValueError("method must be 'lm' or 'lbfgs'")

    def infid(p):
        return 1.0 - abs(np.vdot(psi_ed, hea_statevector(p, n, L))) ** 2

    def fg(p):
        nonlocal evals
        evals += 1
        o, g = hea_overlap_and_grad(p, n, L, psi_ed)
        return 1.0 - abs(o) ** 2, -2.0 * np.real(np.conj(o) * g)

    def resid(p):
        nonlocal evals
        evals += 1
        psi = hea_statevector(p, n, L)
        sgn = 1.0 if np.real(np.vdot(psi_ed, psi)) >= 0.0 else -1.0
        r = psi - sgn * psi_ed
        return np.concatenate([r.real, r.imag])

    def jac(p):
        J = hea_state_jacobian(p, n, L)
        return np.concatenate([J.real, J.imag])

    def one(kind, p0):
        if method == "lbfgs":
            res = minimize(fg, p0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "maxfun": 4 * int(maxiter),
                                    "gtol": float(gtol), "ftol": 0.0})
            fun, nit, nfev, gmax = float(res.fun), int(res.nit), int(res.nfev), float(np.max(np.abs(res.jac)))
            msg, ok = str(res.message), bool(res.success)
        else:
            res = least_squares(resid, p0, jac=jac, method="lm", max_nfev=int(maxiter),
                                ftol=1e-15, xtol=1e-15, gtol=1e-15)
            fun, nit, nfev = infid(res.x), int(res.njev), int(res.nfev)
            gmax = float(np.max(np.abs(res.grad)))
            msg, ok = str(res.message), bool(res.success)
        return fun, nit, nfev, gmax, msg, ok, np.asarray(res.x, dtype=float)

    rng_p = np.random.default_rng(int(seed) + 7919)
    done = False
    for r, (kind, p0) in enumerate(starts):
        fun, nit, nfev, gmax, msg, ok, x = one(kind, p0)
        n_iter += nit
        history.append({"restart": r, "start": kind, "infidelity": fun, "nit": nit, "nfev": nfev,
                        "grad_max": gmax, "success": ok, "message": msg})
        if best is None or fun < best[1]:
            best = (r, fun, x, kind, gmax)
        if stop_at is not None and fun <= float(stop_at):
            done = True
            break
    for k in range(int(perturb_rounds)):
        if done:
            break
        r = len(history)
        p0 = best[2] + rng_p.normal(0.0, float(perturb_scale), size=n_par)
        fun, nit, nfev, gmax, msg, ok, x = one("perturb", p0)
        n_iter += nit
        history.append({"restart": r, "start": "perturb", "infidelity": fun, "nit": nit, "nfev": nfev,
                        "grad_max": gmax, "success": ok, "message": msg})
        if fun < best[1]:
            best = (r, fun, x, "perturb", gmax)
        if stop_at is not None and fun <= float(stop_at):
            done = True
    r_best, _, x_best, kind, gmax = best
    params = np.asarray(x_best, dtype=float)
    psi = hea_statevector(params, n, L)
    ov = np.vdot(psi_ed, psi)
    if abs(ov) > 0:
        psi = psi * (ov.conj() / abs(ov))
    E = float(np.real(np.vdot(psi, H @ psi)))
    V, _ = covariance_from_state(chain, psi)
    V_ed, _ = covariance_from_state(chain, psi_ed)
    F = float(abs(ov) ** 2)
    return FidelityResult(
        layers=L, n_params=n_par, fidelity=F, infidelity=1.0 - F, params=params, state=psi,
        energy=E, E0_exact=float(E0), energy_error=E - float(E0),
        cov_error=float(np.max(np.abs(V - V_ed))),
        grad_norm=float(np.linalg.norm(-2.0 * np.real(np.conj(ov) * _adjoint_overlap_grad(params, n, L, psi, psi_ed)))),
        restarts=len(starts), restarts_run=len(history), best_restart=int(r_best), best_start=str(kind),
        restart_infidelities=[h["infidelity"] for h in history], n_iterations=n_iter,
        n_evaluations=int(evals), seed=int(seed), wall_s=time.time() - t0, history=history,
    )


def fidelity_depth_scan(chain: TruncatedChain, layers=(1, 2, 3, 4, 5, 6), restarts=20, seed=0,
                        layerwise=True, until=None, max_layers=None, target=None, **kw):
    """:func:`fidelity_ceiling` at every depth of ``layers`` (ascending; seed
    ``seed + 1000 L`` at depth L), each depth also started from the previous
    depth's optimum embedded by :func:`hea_embed_layers` (the layerwise
    growing start, labelled 'layerwise'; it reproduces the previous fidelity
    exactly, so the scan's best infidelity is non-increasing in depth).  If
    ``until`` and ``max_layers`` are given and the listed depths do not reach
    infidelity <= until, the scan continues one layer at a time up to
    ``max_layers``.  Extra keywords go to :func:`fidelity_ceiling`.  Returns
    the list of :class:`FidelityResult`."""
    if target is None:
        target = exact_ground_state(chain)
    depths = sorted({int(L) for L in layers})
    out: List[FidelityResult] = []

    def run(L):
        init = None
        if layerwise and out:
            prev = out[-1]
            init = {"layerwise": hea_embed_layers(prev.params, chain.n_qubits, prev.layers, L)}
        out.append(fidelity_ceiling(chain, L, restarts=restarts, seed=int(seed) + 1000 * int(L),
                                    init_params=init, target=target, **kw))

    for L in depths:
        run(L)
    if until is not None and max_layers is not None:
        while out and out[-1].infidelity > float(until) and out[-1].layers < int(max_layers):
            run(out[-1].layers + 1)
    return out


# ---------------------------------------------------------------------------
# VQE
# ---------------------------------------------------------------------------


@dataclass
class VQEResult:
    """Outcome of :func:`vqe_ground_state`.

    energy / E0_exact / energy_error : Aer (or numpy) energy at the optimum,
        the ED ground energy of the same H_d, their difference (>= 0).
    params : optimal ansatz parameters (the circuit is ``hea_circuit``).
    state : the prepared statevector (numpy arbiter), sign-aligned to ED.
    fidelity : |<psi_ED|psi_VQE>|^2.
    covariance / covariance_exact / cov_error : truncated-model covariances
        of the VQE and ED states and their max-norm difference.
    cov_error_gaussian : max |V_VQE - ground_state_cov(K)| (the encoding's
        own error plus the VQE's).
    aer_vs_numpy : max |E_aer - E_numpy| over the final parameter batch.
    n_evaluations, n_iterations, restarts, layers, seed, evaluator, history.
    warm_start / warm_infidelity / warm_energy_error : the warm start used
        (None, 'fidelity', 'layerwise', 'given') and the infidelity and
        energy error of the warm-start point BEFORE the energy optimisation.
    gradient : 'shift' (parameter-shift through the evaluator) or 'adjoint'.
    grad_norm : 2-norm of the energy gradient at the returned optimum.
    wall_s : wall time of the call (including any warm start).
    """

    energy: float
    E0_exact: float
    energy_error: float
    params: np.ndarray
    state: np.ndarray
    fidelity: float
    covariance: np.ndarray
    covariance_exact: np.ndarray
    cov_error: float
    cov_error_gaussian: float
    aer_vs_numpy: Optional[float]
    n_evaluations: int
    n_iterations: int
    restarts: int
    layers: int
    seed: int
    evaluator: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    warm_start: Optional[str] = None
    warm_infidelity: Optional[float] = None
    warm_energy_error: Optional[float] = None
    gradient: str = "shift"
    grad_norm: Optional[float] = None
    wall_s: Optional[float] = None


class _NumpyEvaluator:
    def __init__(self, H, n, layers):
        self.H = H
        self.n = n
        self.layers = layers
        self.calls = 0

    def __call__(self, batch):
        batch = np.atleast_2d(batch)
        self.calls += batch.shape[0]
        out = np.empty(batch.shape[0])
        for i, p in enumerate(batch):
            psi = hea_statevector(p, self.n, self.layers)
            out[i] = float(np.real(np.vdot(psi, self.H @ psi)))
        return out


class _AerEvaluator:
    """Batched energies from Aer's statevector method: one job per batch via
    ``parameter_binds`` and ``save_expectation_value``."""

    def __init__(self, terms, n, layers, seed=0):
        Q = _qiskit()
        self.n = n
        self.layers = layers
        self.calls = 0
        qc, th = hea_circuit(n, layers)
        self.th = th
        op = Q.SparsePauliOp.from_list(
            [(pauli_label(t, n), complex(t.coeff)) for t in terms]
        )
        qc.save_expectation_value(op, list(range(n)), label="energy")
        self.sim = Q.AerSimulator(method="statevector", seed_simulator=int(seed))
        self.qc = Q.transpile(qc, self.sim, optimization_level=0)

    def __call__(self, batch):
        batch = np.atleast_2d(np.asarray(batch, dtype=float))
        self.calls += batch.shape[0]
        binds = [{self.th[k]: batch[:, k].tolist() for k in range(batch.shape[1])}]
        res = self.sim.run(self.qc, parameter_binds=binds, shots=1).result()
        return np.array([float(res.data(i)["energy"]) for i in range(batch.shape[0])])


def _parameter_shift_batch(params):
    """Rows: params, then params +- pi/2 e_k for every k."""
    n = params.size
    batch = np.tile(params, (2 * n + 1, 1))
    for k in range(n):
        batch[1 + 2 * k, k] += math.pi / 2.0
        batch[2 + 2 * k, k] -= math.pi / 2.0
    return batch


def ansatz_energy(chain: TruncatedChain, params, layers, evaluator="aer", seed=0):
    """<psi(theta)|H_d|psi(theta)> at ``params`` by the named evaluator
    ('aer': Aer statevector + save_expectation_value; 'numpy': the arbiter) —
    the cross-check that a state optimised on one evaluator is the same
    energy on the other."""
    n = chain.n_qubits
    params = np.asarray(params, dtype=float)
    if evaluator == "aer":
        ev = _AerEvaluator(hamiltonian_pauli_terms(chain), n, layers, seed=seed)
    elif evaluator == "numpy":
        ev = _NumpyEvaluator(hamiltonian_dense(chain), n, layers)
    else:
        raise ValueError("evaluator must be 'aer' or 'numpy'")
    return float(ev(params[None, :])[0])


def vqe_ground_state(chain: TruncatedChain, layers=2, restarts=3, seed=0, maxiter=400,
                     gtol=1e-9, evaluator="aer", init_scale=0.3, init_params=None,
                     warm_start=None, warm_kwargs=None, gradient="shift"):
    """Variational ground state of H_d with the hardware-efficient ansatz.

    Parameters
    ----------
    chain : :class:`TruncatedChain` (d a power of two).
    layers : entangling layers of :func:`hea_circuit`.
    restarts : seeded random initial points (each N(0, init_scale^2));
        the best final energy wins.  ``init_params`` (if given) is tried
        first, as an extra restart.
    seed : RNG seed for the initial points and Aer.
    maxiter, gtol : L-BFGS-B settings; gradients are exact parameter-shift
        differences (or the exact adjoint gradient), so ``gtol`` is a true
        gradient-norm criterion.
    evaluator : 'aer' (Aer statevector; needs the extra) or 'numpy'.
    warm_start : None (default), or a start tried BEFORE ``init_params`` and
        the random restarts: 'fidelity' — the optimum of
        :func:`fidelity_ceiling` at this depth (``warm_kwargs`` are its
        keywords: restarts, seed, maxiter, ...); 'layerwise' — the last
        result of :func:`fidelity_depth_scan` over depths 1..layers
        (``warm_kwargs`` likewise); or a parameter vector ('given').  The
        result records the warm point's infidelity and energy error.
    gradient : 'shift' (default; parameter-shift batches through the
        evaluator, 2 n_params + 1 energies per gradient) or 'adjoint' (the
        numpy arbiter's exact reverse-mode gradient,
        :func:`hea_energy_and_grad`; requires ``evaluator='numpy'``).

    Returns
    -------
    VQEResult
    """
    if 2 ** chain.q != chain.d:
        raise ValueError(f"d = {chain.d} is not a power of two; no qubit encoding")
    if gradient not in ("shift", "adjoint"):
        raise ValueError("gradient must be 'shift' or 'adjoint'")
    if gradient == "adjoint" and evaluator != "numpy":
        raise ValueError("gradient='adjoint' is the numpy arbiter's; use evaluator='numpy'")
    t0 = time.time()
    n = chain.n_qubits
    terms = hamiltonian_pauli_terms(chain)
    H = hamiltonian_dense(chain)
    E0, psi_ed = exact_ground_state(chain)
    if evaluator == "aer":
        ev = _AerEvaluator(terms, n, layers, seed=seed)
    elif evaluator == "numpy":
        ev = _NumpyEvaluator(H, n, layers)
    else:
        raise ValueError("evaluator must be 'aer' or 'numpy'")
    rng = np.random.default_rng(int(seed))
    n_par = hea_parameter_count(n, layers)
    starts = []
    warm_label = None
    warm_info = {}
    if warm_start is not None:
        wk = dict(warm_kwargs or {})
        if isinstance(warm_start, str):
            if warm_start == "fidelity":
                fr = fidelity_ceiling(chain, layers, target=(E0, psi_ed), **wk)
            elif warm_start == "layerwise":
                fr = fidelity_depth_scan(chain, layers=range(1, int(layers) + 1),
                                         target=(E0, psi_ed), **wk)[-1]
            else:
                raise ValueError("warm_start must be None, 'fidelity', 'layerwise' or a parameter vector")
            warm = fr.params
            warm_label = warm_start
            warm_info = {"warm_infidelity": float(fr.infidelity), "warm_energy_error": float(fr.energy_error)}
        else:
            warm = np.asarray(warm_start, dtype=float).copy()
            if warm.size != n_par:
                raise ValueError(f"warm_start: expected {n_par} parameters, got {warm.size}")
            warm_label = "given"
            psi_w = hea_statevector(warm, n, layers)
            warm_info = {"warm_infidelity": 1.0 - state_fidelity(psi_ed, psi_w),
                         "warm_energy_error": float(np.real(np.vdot(psi_w, H @ psi_w))) - E0}
        starts.append(warm)
    if init_params is not None:
        starts.append(np.asarray(init_params, dtype=float).copy())
    for _ in range(int(restarts)):
        starts.append(rng.normal(0.0, float(init_scale), size=n_par))

    history = []
    best = None
    n_iter_total = 0
    n_adjoint = 0

    if gradient == "adjoint":
        def fg(p):
            nonlocal n_adjoint
            n_adjoint += 1
            return hea_energy_and_grad(p, n, layers, H)
    else:
        def fg(p):
            batch = _parameter_shift_batch(p)
            E = ev(batch)
            grad = 0.5 * (E[1::2] - E[2::2])
            return float(E[0]), grad

    for r, p0 in enumerate(starts):
        res = minimize(fg, p0, jac=True, method="L-BFGS-B",
                       options={"maxiter": int(maxiter), "maxfun": max(15000, 4 * int(maxiter)),
                                "gtol": float(gtol), "ftol": 0.0})
        n_iter_total += int(res.nit)
        history.append({"restart": r, "energy": float(res.fun), "nit": int(res.nit),
                        "success": bool(res.success), "message": str(res.message),
                        "grad_max": float(np.max(np.abs(res.jac)))})
        if best is None or res.fun < best.fun:
            best = res
    params = np.asarray(best.x, dtype=float)
    psi = hea_statevector(params, n, layers)
    E_np = float(np.real(np.vdot(psi, H @ psi)))
    aer_vs_numpy = None
    if evaluator == "aer":
        aer_vs_numpy = float(abs(ev(params[None, :])[0] - E_np))
    # sign-align with ED for the reported state
    ov = np.vdot(psi_ed, psi)
    if abs(ov) > 0:
        psi = psi * (ov.conj() / abs(ov))
    V, _ = covariance_from_state(chain, psi)
    V_ed, _ = covariance_from_state(chain, psi_ed)
    Vg, _ = gaussian_reference(chain)
    return VQEResult(
        energy=float(best.fun),
        E0_exact=E0,
        energy_error=float(best.fun) - E0,
        params=params,
        state=psi,
        fidelity=state_fidelity(psi_ed, psi),
        covariance=V,
        covariance_exact=V_ed,
        cov_error=float(np.max(np.abs(V - V_ed))),
        cov_error_gaussian=float(np.max(np.abs(V - Vg))),
        aer_vs_numpy=aer_vs_numpy,
        n_evaluations=int(ev.calls) if gradient == "shift" else int(n_adjoint),
        n_iterations=n_iter_total,
        restarts=len(starts),
        layers=int(layers),
        seed=int(seed),
        evaluator=evaluator,
        history=history,
        warm_start=warm_label,
        warm_infidelity=warm_info.get("warm_infidelity"),
        warm_energy_error=warm_info.get("warm_energy_error"),
        gradient=gradient,
        grad_norm=float(np.linalg.norm(best.jac)),
        wall_s=time.time() - t0,
    )
