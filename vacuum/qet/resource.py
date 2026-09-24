"""A resource theory of vacuum manipulation — Layer 7, L3 (small-system numerics).

SETTING
-------
A finite qubit system split into Alice's region A and Bob's region B (by
default B is the complement of A), Hamiltonian H, ground state |g>. The
energy accounting is Hotta's [M. Hotta, "Quantum energy teleportation: an
introductory review", arXiv:1101.3954, Sec. 3: Eq. (12), E_A = sum_alpha
<g|P_A(alpha) H_A P_A(alpha)|g> "localized at A", and the unnumbered line
after Eq. (13), E_B = -Tr[rho (H_B + V)]]: the interaction is booked on
Bob's side,

    H = H_A + H_B^loc,   H_A := (Tr_B H / d_B) (x) I_B,   H_B^loc := H - H_A,   (1)

H_A being the part of H supported on A alone (H_B^loc = H_B + V in Hotta's
notation, up to a constant). Operator-Schmidt decompose
H_B^loc = I_A (x) H_B + sum_j X_j (x) Y_j with traceless "bond operators"
X_j on A (:func:`bond_operators`).

FREE OPERATIONS  (Chitambar-Gour, arXiv:1806.06107, Sec. III.A Def. 1)
-----------------------------------------------------------------------
(O1) Alice: local instruments {K_mu} on A that are ENERGY-NON-SIGNALLING —
     the average channel Abar(.) = sum_mu K_mu (.) K_mu^dag leaves Bob's local
     energy operator invariant,

         (Abar^dag (x) id)(H_B^loc) = H_B^loc   <=>   Abar^dag(X_j) = X_j  for all j.   (2)

     Before her classical message arrives Alice cannot change <H_B^loc> on
     ANY state: the energetic form of no-signalling. Hotta's sigma_x
     projectors are the finest such instrument ("P_A(alpha) commutes with
     H_B and V", loc. cit.; chains: "P_A(mu) commutes with every bond
     term", vacuum.qet.chains). A sufficient condition is [K_mu, X_j] = 0
     for all mu, j — the COMMUTING-KRAUS class, which is what
     :func:`random_free_instrument` samples. It is NECESSARY exactly when
     the joint bond eigenvalue vectors x(a) are in convex position
     (THEOREM 2, :func:`free_class_extremality`): no unitality is assumed,
     superseding the earlier Bloch-ball-plus-[A. Arias, A. Gheondea,
     S. Gudder, J. Math. Phys. 43, 5872 (2002)] argument, which needed
     Abar(I) = I and so reached only a single-qubit A. Where some x(b) is a
     convex combination of the others,
     :func:`nonfree_energy_nonsignalling_instrument` builds an explicit
     NON-unital free instrument outside the commutant, and the certified
     SDP :func:`search_free_instrument` measures the whole slack. Every
     model this module ships (single-qubit A; any region of a TFIM chain,
     whose commuting sigma_x bonds have hypercube-vertex spectra) is in
     convex position, so Prop. A and Theorem 1 stand on all of them;
     :func:`star_hamiltonian` is a model that is not, and there the wider
     free class extracts strictly more than the closed form (4).
(O2) Classical communication A -> B: Bob's operation may depend on mu.
(O3) Bob: any CPTP map E on B. His operations are the cash-out; the energy
     leaving into his device is  w_B(E, rho) = Tr[H_B^loc rho]
     - Tr[H_B^loc (id (x) E)(rho)]  (:func:`bob_energy_out`).
(O4) Convex mixtures and relabelling of classical flags.
Composition preserves (2) (a free instrument followed by branch-dependent
free instruments has Abar^dag(X_j) = sum_mu K_mu^dag Abar_mu^dag(X_j) K_mu
= X_j), so (F, O) is a QRT in the sense of Def. 1 with the golden rule.

FREE STATES: rho with W_->(rho) = 0 (below). Ground states are NOT free —
the vacuum is the resource (Hotta).

MONOTONE 1 — the QET POTENTIAL, the one-way-LOCC-assisted extractable
energy of Bob's region:

    W_->(rho) := sup_{free {K_mu}, CPTP {E_mu} on B}  sum_mu {
                   Tr[H_B^loc (K_mu (x) I) rho (K_mu (x) I)^dag]
                 - Tr[H_B^loc (id (x) E_mu)((K_mu (x) I) rho (K_mu (x) I)^dag)] }.   (3)

Proposition A (closed form; :func:`qet_potential`). If B is the complement
of A and the bond operators commute, with joint eigenprojectors Pi_a of
{X_j} (eigenvalues x_j(a)) — the value of the COMMUTING-KRAUS class, hence
of the whole free class exactly under Theorem 2's convex-position
hypothesis —

    W_->(rho) = sum_a { Tr[H^(a) sigma_a] - p_a lambda_min(H^(a)) },
    sigma_a = Tr_A[(Pi_a (x) I) rho],  p_a = Tr sigma_a,  H^(a) = H_B + sum_j x_j(a) Y_j.   (4)

Proof. Commuting-Kraus operators are block-diagonal in the Pi_a; inside a
block every X_j is the scalar x_j(a), so each branch sees the same
effective Bob Hamiltonian H^(a), Bob's CPTP optimum is the reset to its
ground state (energy p lambda_min(H^(a))), and refining an instrument
inside a block leaves sum_branches {Tr[H^(a) sigma] - p lambda_min} —
which is linear in sigma — unchanged. The projective measurement {Pi_a} is
therefore optimal, and Bob's optimal map is state-independent. []

Theorem 1 (ledger monotonicity; :func:`monotonicity_audit`). For every
free instrument on A with branches rho_mu of probability p_mu (strong,
i.e. on-average, monotonicity — Chitambar-Gour Eq. (61)) and for every
CPTP map E on B,

    sum_mu p_mu W_->(rho_mu) <= W_->(rho),                                     (5a)
    W_->((id (x) E)(rho))    <= W_->(rho) - w_B(E, rho).                       (5b)

Proof. (5a): composing Alice's instrument with each branch's optimal free
instrument is a free instrument for rho, and the objective (3) is additive
over branches. (5b): take the optimal free instrument {K_mu} and Bob maps
{E_mu} of the output state; by (2), sum_mu Tr[H_B^loc (K_mu (x) E)(rho)]
= Tr[H_B^loc (id (x) E)(rho)]; using {K_mu} with the maps E_mu o E on rho
gives W_->(rho) >= Tr[H_B^loc rho] - Tr[H_B^loc (id (x) E) rho]
+ W_->((id (x) E) rho). []
So Bob's extraction w_B > 0 is PAID FROM W_->. In the minimal model
W_->(|g>) = E_B^max (Hotta Eq. (11); tested), Alice's measurement plus the
bit convert it into locally extractable energy on the branches, and Bob's
rotation spends it: the Hotta protocol is one free-operation sequence and
E_B is the resource being spent. Energy-injecting Bob maps (w_B < 0) can
raise W_-> by up to -w_B — they are the non-free operations with teeth in
the tests.

MONOTONE 2 — the NEGATIVITY N(rho) = (||rho^{Gamma_A}||_1 - 1)/2
[G. Vidal, R. F. Werner, Phys. Rev. A 65, 032314 (2002)], an entanglement
monotone (non-increasing on average) under all LOCC, hence under O, which
is a subset of LOCC; for Gaussian systems the log-negativity of
vacuum.core. It vanishes on the post-measurement branches although W_->
does not: entanglement is what lets W_->(|g>) exceed Bob's local
extractable energy W_loc(|g>) = 0 (strong local passivity, vacuum.qet.slp),
but W_->, not entanglement, is what Bob cashes.

The QET SURPLUS Q(rho) := W_->(rho) - W_loc(rho) is NOT a monotone: a
local unitary on A can lower W_loc (it changes the mean field Bob sees
through V) without touching W_-> (:func:`surplus_counterexample`, tested).
It survives as the ground-state diagnostic Q(|g>) = W_->(|g>).

ONE-SHOT BOUND  (smooth entropies; M. Tomamichel, "Quantum Information
Processing with Finite Resources", arXiv:1504.00233v5; natural logs here,
his log is generic)
------------------------------------------------------------------------
Notation: H_min(A)_tau = -ln lambda_max(tau) (Def. 6.2 with trivial
conditioning), purified distance P (Def. 3.15), epsilon-ball (Def. 6.8),
H_min^eps = max over the ball (Def. 6.9, Eq. (6.34)), Delta <= P
(Lemma 3.17, Eq. (3.55)), conditioning on a classical register
(Eq. (6.25): H_min(A|Y) = -ln sum_y p(y) e^{-H_min(A)_{tau_y}}), duality
H_max(A|B)_psi = -H_min(A|C)_psi for pure psi_ABC (Lemma 6.7; smooth:
Prop. 6.14, Eq. (6.51)), H_min(A|B) SDP (Lemma 6.3, Eq. (6.8)), smooth SDP
(Eq. (6.37)). Let R carry Bob's local energy operator h_R (R = B when B is
the complement; R = B plus the support of its bond terms for a single-site
Bob, the chains' local_hamiltonian), tau_mu the conditional state of R
given Alice's outcome mu (probability p_mu), and let Bob act by a UNITARY
on (a subsystem of) R — Hotta's rotation is one. With the Ky Fan floor

    kappa(h, s) := min { Tr[h tau] : 0 <= tau <= e^{-s} I, Tr tau = 1 }
                 = e^{-s} sum_{i <= m} lambda_i^up(h) + (1 - m e^{-s}) lambda_{m+1}^up(h),
                 m = floor(e^s)                                                          (6)

(a linear programme over the diagonal of tau in h's eigenbasis, solved by
greedy filling of the lowest levels; :func:`ky_fan_floor`), the chain is

    E_B  =  sum_mu p_mu ( Tr[h_R tau_mu] - Tr[h_R U_mu tau_mu U_mu^dag] )                   (i)
        <=  sum_mu p_mu ( Tr[h_R tau_mu] - kappa(h_R, H_min(R)_{tau_mu}) )                  (ii)
        <=  sum_mu p_mu ( Tr[h_R tau_mu] - kappa(h_R, H_min^eps(R)_{tau_mu})
                          + eps [lambda_max(h_R) - lambda_min(h_R)] )                       (iii)
        =   sum_mu p_mu ( Tr[h_R tau_mu] - kappa(h_R, -H_max^eps(R|E)_{Psi_mu}) + eps spread )   (iv)

(ii): U tau U^dag has the spectrum of tau, so it is feasible in (6) with
s = H_min(tau). (iii): for the smoothing optimizer tau~ (normalized ball —
a subset of Tomamichel's, so our H_min^eps <= his and the bound is valid
with either), Tr[h U tau U^dag] >= Tr[h U tau~ U^dag] - spread(h)
Delta(tau, tau~) >= kappa(h, H_min^eps) - eps spread(h). (iv): for a pure
global branch Psi_mu (E = everything outside R), H_min^eps(R)_{tau_mu}
= -H_max^eps(R|E)_{Psi_mu} — Bob's yield is capped by how negative the
conditional max-entropy of his region given the rest of the chain is.
Aggregated form (h_R the same for every mu; kappa is convex in e^{-s} as
an LP value in its right-hand side, so Jensen applies):

    E_B <= Tr[h_R rho_R] - kappa(h_R, H_min(R|X)_{rho_XR}),                                 (v)

rho_XR the classical-quantum state of Alice's outcome and Bob's region.
For Bob = complement and pure rho every tau_mu is pure, H_min = 0,
kappa = lambda_min, and (ii) collapses to (4): the minimal model SATURATES
the bound (ratio 1 to roundoff; tested). The chain protocol's single-site
rotation does not; the paper candidate charts the gap.

L3 CLOSE-OUT (the five results that retire the leap's open items)
-----------------------------------------------------------------
THEOREM 2 (free class, no unitality; :func:`free_class_extremality`).
Energy-non-signalling = commuting-Kraus iff every joint bond eigenvalue
vector is an extreme point of their convex hull. Proof, and the explicit
non-unital counterexample, in those functions' docstrings. Consequence,
measured: on :func:`star_hamiltonian` the wider free class yields 5.15x the
closed form (4), which is not even monotone there — so (4) is the
commuting-Kraus value and the hypothesis must be stated, as it now is.

THEOREM 4 (linearity and the exact ledger; :func:`qet_potential_operator`).
W_-> is a LINEAR functional, W_->(rho) = Tr[W_op rho] with W_op = H_B^loc
- sum_a lambda_min(H^(a)) Pi_a (x) I, whose non-Hotta part lives on A alone.
Hence Theorem 1's inequalities are EQUALITIES for COMMUTING-KRAUS
instruments on A and for arbitrary instruments on B, and the total energy any
finite-round protocol with TWO-WAY classical communication delivers to Bob is
exactly W_->(rho): two-way communication gives Bob nothing (Eq. (12)). By
Theorem 2 that scope is the whole free class exactly where the bond spectrum
is in convex position — every model shipped here. Where it is not, (12) is
FALSE and already fails one-way: on :func:`star_hamiltonian` a single
genuinely free (non-commuting-Kraus) instrument raises the closed form (4) by
6.1509x (Eq. (12'), pinned in the tests). Prop. A's value is the
commuting-Kraus potential W_->^CK, and off convex position it is neither the
true potential nor a monotone. The symmetric booking (9),
:func:`split_hamiltonian_symmetric`, has the same free class; when BOTH
devices cash out (:func:`two_way_potential`) the value does exceed W_-> and
grows with the round count, but the surplus is paid by the measurement
apparatus — the net of the injections is round-independent (measured).

PROPOSITION B (the tight bound, exactly; :func:`single_site_extractable`).
For a single-qubit Bob the one-way extractable energy under unitaries is the
orthogonal-Procrustes closed form (15), Tr T + ||T||_* with the determinant
fix, T the energy-correlation matrix of Bob's site. It reproduces Hotta's
Eq. (11) and the chain protocol to roundoff, so the "tight relaxation" the
marginal-fixed SDP was reaching (0.92) is replaced by an exact, solver-free
answer.

PROPOSITION C (the entropic form; :func:`one_shot_bound_entropic`).
Passivity of Bob's unconditional state plus Hoelder gives
E_B <= 2 c (exp(-H_min(X|R)) - 1/2): Bob extracts only what his region KNOWS
about Alice's outcome, and the bound vanishes exactly when her bit is
unguessable. It is honest but LOOSE (ratio ~0.008 on the chains against
Prop. B's 1.000), and the refinements measured alongside it — the b-active
restriction and the row-sum relaxation of (15) — are loose for the same
reason: Alice's outcome moves Bob's state by an O(1) trace distance in a
nearly energy-neutral direction, and the tight content is a cancellation
inside ||T||_* that no norm-based entropic relaxation retains. QET's E_B is
exactly the DAEMONIC ERGOTROPY of G. Francica, J. Goold, M. Paternostro,
F. Plastina, "Daemonic ergotropy: enhanced work extraction from quantum
correlations", npj Quantum Information 3, 12 (2017) [arXiv:1608.00124],
Eqs. (3) and (5), with the free class constraining the daemon and the
unconditional ergotropy vanishing by strong local passivity
(:func:`daemonic_gain`).

PROPOSITION E (smoothing; :func:`smooth_min_entropy`). Over Tomamichel's own
sub-normalized ball (Def. 6.8) the smooth min-entropy is UNCHANGED until the
normalized value saturates at ln d, where it adds exactly -ln(1 - eps^2).
The one-shot chain's inputs are near-pure, so the "normalized ball" caveat
cost nothing; the sub-normalized chain (iii') is implemented
(:func:`one_shot_bound` with ``subnormalized=True``) and never tightens a
bound. L = 12 is reached by :func:`chain_potential_sparse` and
:func:`chain_region_ledger`, which never build a dense 2^L.

GAUSSIAN MIRROR (2-4 modes, vacuum.core conventions). For the harmonic
chain H = (1/2)(p.p + x.K.x), Alice's free instrument is homodyne
detection of x_A (it commutes with the bond K_{A,A+1} x_A x_{A+1}); given
the outcome xi, Bob's local Hamiltonian acquires the linear term
xi K_{AB} . x_B and its ground energy drops by (1/2) xi^2 K_AB^T K_BB^{-1}
K_AB, while his conditional state is Gaussian with the covariance of the
Schur complement and a mean linear in xi. :func:`gaussian_qet_potential`
integrates the resulting closed form over xi ~ N(0, V_xx[A, A]).

Conventions: hbar = 1; qubit site 0 is the leftmost tensor factor
(vacuum.qet.hotta, .chains, .slp); Gaussian block ordering per
docs/API.md. All functions are pure over their array arguments. cvxpy is
imported only inside the SDP functions (the ``.[sdp]`` extra).
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
from scipy.linalg import eigh

from vacuum.audits import AuditResult, AuditViolation
from vacuum.qet import slp
from vacuum.qet.slp import (
    _as_region_blocks,
    _check_qubit_operator,
    _check_region,
    _herm,
    apply_local_channel,
    embed_region_operator,
    state_energy,
)

__all__ = [
    "split_hamiltonian",
    "bond_operators",
    "free_projectors",
    "energy_signalling_defect",
    "random_free_instrument",
    "random_instrument",
    "random_local_unitary",
    "apply_instrument",
    "bob_energy_out",
    "qet_potential",
    "local_extractable",
    "qet_surplus",
    "surplus_counterexample",
    "negativity",
    "random_state",
    "monotonicity_audit",
    "ky_fan_floor",
    "min_entropy",
    "smooth_min_entropy",
    "conditional_min_entropy_cq",
    "min_entropy_sdp",
    "max_entropy_sdp",
    "max_entropy_direct",
    "classical_min_max_entropy",
    "one_shot_bound",
    "one_shot_bound_marginal",
    "hotta_resource_ledger",
    "chain_resource_ledger",
    "gaussian_qet_potential",
    "gaussian_monotonicity_audit",
    # --- L3 close-out (two-way theory, free class, tight/entropic bounds, L = 12)
    "free_class_extremality",
    "star_hamiltonian",
    "nonfree_energy_nonsignalling_instrument",
    "search_free_instrument",
    "split_hamiltonian_symmetric",
    "qet_potential_operator",
    "optimal_local_extraction",
    "two_way_potential",
    "two_way_monotonicity_audit",
    "bob_correlation_matrix",
    "single_site_extractable",
    "unitary_energy_spread",
    "guessing_probability_cq",
    "one_shot_bound_entropic",
    "daemonic_gain",
    "chain_potential_sparse",
    "chain_region_ledger",
]

_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)


# ---------------------------------------------------------------------------
# bipartition plumbing
# ---------------------------------------------------------------------------


def _complement(region, n):
    return tuple(q for q in range(n) if q not in region)


def _partial_trace_B(op, region, n):
    """Tr_B of a 2^n operator over the complement of ``region`` -> (dA, dA)."""
    return np.einsum("abcb->ac", _as_region_blocks(op, region, n))


def _partial_trace_A(op, region, n):
    """Tr_A of a 2^n operator over ``region`` -> (dB, dB) in complement order."""
    return np.einsum("abad->bd", _as_region_blocks(op, region, n))


def split_hamiltonian(H, region_A):
    """Eq. (1): (H_A, H_B^loc) with H_A = (Tr_B H / d_B) (x) I_B, H_B^loc = H - H_A.

    H_A is the component of H supported on A alone (its constant part
    included); everything else — Bob's own terms and every A-B interaction
    term — is Bob's local energy operator, Hotta's H_B + V. Both are returned
    as full 2^n x 2^n operators in the site-ordered basis.
    """
    H, n = _check_qubit_operator("H", H)
    region_A = _check_region(region_A, n)
    dB = 2 ** (n - len(region_A))
    hA = _partial_trace_B(H, region_A, n) / dB
    H_A = embed_region_operator(hA, region_A, n)
    return H_A, H - H_A


def bond_operators(H, region_A, tol=1e-12):
    """Operator-Schmidt decomposition of H_B^loc = I (x) H_B + sum_j X_j (x) Y_j.

    Returns (H_B, [X_j], [Y_j]): H_B is (dB, dB) in complement site order,
    X_j are traceless Hermitian (dA, dA) operators on A (orthonormal in the
    Hilbert-Schmidt sense, Tr X_i X_j = delta_ij), Y_j are Hermitian
    (dB, dB). Terms with Schmidt weight below ``tol`` are dropped.
    """
    H, n = _check_qubit_operator("H", H)
    region_A = _check_region(region_A, n)
    _, H_loc = split_hamiltonian(H, region_A)
    dA = 2 ** len(region_A)
    dB = 2 ** (n - len(region_A))
    t = _as_region_blocks(H_loc, region_A, n)          # t[a, b, a', b']
    H_B = np.einsum("abad->bd", t) / dA                 # I (x) H_B component
    rest = t - np.einsum("ac,bd->abcd", np.eye(dA), H_B)
    M = rest.transpose(0, 2, 1, 3).reshape(dA * dA, dB * dB)  # (a a'), (b b')
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    Xs, Ys = [], []
    for k in range(s.size):
        if s[k] <= tol:
            continue
        X = U[:, k].reshape(dA, dA)
        Y = s[k] * Vh[k].reshape(dB, dB)
        # each Schmidt pair can be rotated to a Hermitian pair because the
        # total is Hermitian and the singular values are distinct generically;
        # symmetrize both and re-fit the weight so the sum is unchanged
        Xh = _herm(X)
        if np.max(np.abs(Xh)) < 1e-14:
            Xh = _herm(1j * X)
            Y = -1j * Y
        Xh = Xh / np.sqrt(np.real(np.trace(Xh @ Xh)))
        Yh = np.einsum("ac,abcd->bd", Xh.conj(), rest)  # <X_h, rest>_A
        Xs.append(Xh)
        Ys.append(_herm(Yh))
        rest = rest - np.einsum("ac,bd->abcd", Xh, Yh)
    return _herm(H_B), Xs, Ys


def free_projectors(H, region_A, tol=1e-9):
    """Joint eigenprojectors {Pi_a} of the bond operators — Alice's finest free measurement.

    Returns (projectors, eigenvalues): a list of (dA, dA) projectors on A
    summing to I_A, and an array (n_blocks, n_bonds) of the bond
    eigenvalues x_j(a). Raises ValueError if the bond operators do not
    commute (no finest commuting-Kraus instrument exists then).
    """
    H_B, Xs, Ys = bond_operators(H, region_A)
    dA = H.shape[0] // H_B.shape[0]
    if not Xs:
        return [np.eye(dA, dtype=complex)], np.zeros((1, 0))
    for X1, X2 in itertools.combinations(Xs, 2):
        if np.max(np.abs(X1 @ X2 - X2 @ X1)) > 1e-10:
            raise ValueError("bond operators on A do not commute")
    # simultaneous diagonalization via a generic real combination
    rng = np.random.default_rng(12345)
    coeffs = rng.standard_normal(len(Xs))
    _, U = eigh(sum(c * X for c, X in zip(coeffs, Xs)))
    diag = np.array([[np.real(np.vdot(U[:, i], X @ U[:, i])) for X in Xs]
                     for i in range(dA)])
    # group columns with equal bond eigenvalues
    blocks = []
    used = np.zeros(dA, dtype=bool)
    for i in range(dA):
        if used[i]:
            continue
        members = [j for j in range(dA) if not used[j]
                   and np.max(np.abs(diag[j] - diag[i])) < tol]
        for j in members:
            used[j] = True
        blocks.append(members)
    projectors = [U[:, m] @ U[:, m].conj().T for m in blocks]
    eigenvalues = np.array([diag[m[0]] for m in blocks])
    return projectors, eigenvalues


# ---------------------------------------------------------------------------
# free operations: check, samplers, application
# ---------------------------------------------------------------------------


def energy_signalling_defect(H, kraus, region_A):
    """max | (Abar^dag (x) id)(H_B^loc) - H_B^loc |  for the instrument ``kraus`` on A.

    Zero (to roundoff) iff the instrument is energy-non-signalling, Eq. (2).
    ``kraus`` is a sequence of (dA, dA) matrices in A's basis; completeness
    is checked to 1e-10.
    """
    H, n = _check_qubit_operator("H", H)
    region_A = _check_region(region_A, n)
    dA = 2 ** len(region_A)
    K = [np.asarray(k, dtype=complex) for k in kraus]
    comp = sum(k.conj().T @ k for k in K)
    if np.max(np.abs(comp - np.eye(dA))) > 1e-10:
        raise ValueError("Kraus operators are not complete")
    _, H_loc = split_hamiltonian(H, region_A)
    out = np.zeros_like(H_loc)
    for k in K:
        kf = embed_region_operator(k, region_A, n)
        out = out + kf.conj().T @ H_loc @ kf
    return float(np.max(np.abs(out - H_loc)))


def random_free_instrument(projectors, n_kraus, rng):
    """A random commuting-Kraus instrument on A: block-diagonal in the free projectors.

    For each block Pi_a (rank m_a) an independent random (n_kraus m_a, m_a)
    isometry supplies the blocks of K_1..K_{n_kraus}; sum K^dag K = I holds
    exactly and every K commutes with every bond operator. With n_kraus = 1
    this is a random block-diagonal unitary (a free unitary).
    """
    dA = projectors[0].shape[0]
    K = [np.zeros((dA, dA), dtype=complex) for _ in range(int(n_kraus))]
    for P in projectors:
        w, U = eigh(P)
        cols = U[:, w > 0.5]
        m = cols.shape[1]
        A = rng.standard_normal((n_kraus * m, m)) + 1j * rng.standard_normal((n_kraus * m, m))
        Q, _ = np.linalg.qr(A)
        for mu in range(int(n_kraus)):
            blk = Q[mu * m:(mu + 1) * m, :]
            K[mu] = K[mu] + cols @ blk @ cols.conj().T
    return K


def random_instrument(dA, n_kraus, rng):
    """A random (generally non-free) instrument on a dA-dimensional region."""
    A = rng.standard_normal((n_kraus * dA, dA)) + 1j * rng.standard_normal((n_kraus * dA, dA))
    Q, _ = np.linalg.qr(A)
    return [Q[mu * dA:(mu + 1) * dA, :] for mu in range(int(n_kraus))]


def random_local_unitary(d, rng):
    """Haar-random d x d unitary (QR of a Ginibre matrix with phase fix)."""
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def apply_instrument(rho, kraus, region):
    """Branches {(p_mu, rho_mu)} of an instrument on ``region`` (normalized states).

    Branches of probability below 1e-14 are dropped.
    """
    rho, n = _check_qubit_operator("rho", rho)
    region = _check_region(region, n)
    out = []
    for k in kraus:
        kf = embed_region_operator(k, region, n)
        br = kf @ rho @ kf.conj().T
        p = float(np.real(np.trace(br)))
        if p > 1e-14:
            out.append((p, _herm(br / p)))
    return out


def bob_energy_out(H, rho, kraus_B, region_A):
    """w_B = Tr[H_B^loc rho] - Tr[H_B^loc (id (x) E)(rho)] for a CPTP map E on B.

    Positive when Bob's device receives energy. Equal to the change of the
    total <H> because E commutes with H_A.
    """
    H, n = _check_qubit_operator("H", H)
    region_A = _check_region(region_A, n)
    region_B = _complement(region_A, n)
    _, H_loc = split_hamiltonian(H, region_A)
    after = apply_local_channel(rho, kraus_B, region_B)
    return state_energy(rho, H_loc) - state_energy(after, H_loc)


# ---------------------------------------------------------------------------
# monotone 1: the QET potential W_->  (Eq. (3), closed form Eq. (4))
# ---------------------------------------------------------------------------


def qet_potential(H, rho, region_A, region_B=None, method="closed", tol=1e-9,
                  kraus_kwargs=None):
    """The QET potential W_->(rho), Eq. (3) of the module docstring.

    ``region_B=None`` (Bob = complement of A) uses the closed form (4): Alice's
    finest free measurement {Pi_a} (:func:`free_projectors`), Bob's reset to
    the ground state of his effective Hamiltonian H^(a). Exact for the
    commuting-Kraus free class (Prop. A), machine-precision, no solver.

    A proper subset ``region_B`` (e.g. a single chain site) restricts Bob;
    each branch's extraction is then the certified Choi SDP
    (``method='sdp'``, cvxpy) or the Kraus search (``method='kraus'``,
    core-only) of vacuum.qet.slp on the full Hamiltonian. This equals W_->
    when every free block is rank one (single-qubit A); for larger A it is
    the value of the projective free instrument, a lower bound.

    Returns
    -------
    dict with 'W' (the potential), 'branches' (list of dicts: p, x (bond
    eigenvalues), sigma_B (conditional state of B, normalized, complement
    site order — closed form only), 'H_eff', 'lambda_min', 'extract' = Bob's
    per-branch extraction (normalized)), 'projectors', 'region_A',
    'region_B', 'method'.
    """
    H, n = _check_qubit_operator("H", H)
    rho, n_r = _check_qubit_operator("rho", rho)
    if n_r != n:
        raise ValueError(f"H is on {n} qubits but rho is on {n_r}")
    region_A = _check_region(region_A, n)
    comp = _complement(region_A, n)
    projectors, xs = free_projectors(H, region_A)
    H_B, Xs, Ys = bond_operators(H, region_A)
    branches = []
    W = 0.0
    if region_B is None or tuple(sorted(int(q) for q in region_B)) == comp:
        if method != "closed":
            raise ValueError("Bob = complement uses method='closed' (exact); "
                             f"got {method!r}")
        r = _as_region_blocks(rho, region_A, n)  # r[a, b, a', b']
        for a, P in enumerate(projectors):
            sigma = np.einsum("ac,abcd->bd", P.T, r)   # Tr_A[(P (x) I) rho]
            sigma = _herm(sigma)
            p = float(np.real(np.trace(sigma)))
            H_eff = H_B + sum(float(x) * Y for x, Y in zip(xs[a], Ys))
            H_eff = _herm(H_eff)
            lam = float(eigh(H_eff, eigvals_only=True)[0])
            e_branch = state_energy(sigma, H_eff)      # unnormalized
            W += e_branch - p * lam
            branches.append({
                "p": p, "x": np.array(xs[a]),
                "sigma_B": sigma / p if p > 1e-14 else None,
                "H_eff": H_eff, "lambda_min": lam,
                "extract": (e_branch / p - lam) if p > 1e-14 else 0.0,
            })
        region_B_t = comp
    else:
        region_B_t = _check_region(region_B, n)
        if set(region_B_t) & set(region_A):
            raise ValueError("region_B overlaps region_A")
        for a, P in enumerate(projectors):
            Pf = embed_region_operator(P, region_A, n)
            br = Pf @ rho @ Pf.conj().T
            p = float(np.real(np.trace(br)))
            if p <= 1e-14:
                branches.append({"p": p, "x": np.array(xs[a]), "extract": 0.0})
                continue
            rho_a = _herm(br / p)
            if method == "sdp":
                ext = slp.extractable_energy_sdp(H, rho_a, region_B_t, tol=tol)["extractable"]
            elif method == "kraus":
                ext = slp.extractable_energy_kraus(H, rho_a, region_B_t,
                                                   **(kraus_kwargs or {}))["extractable"]
            else:
                raise ValueError("restricted Bob needs method 'sdp' or 'kraus', "
                                 f"got {method!r}")
            W += p * float(ext)
            branches.append({"p": p, "x": np.array(xs[a]), "extract": float(ext),
                             "rho": rho_a})
    return {
        "W": float(W),
        "branches": branches,
        "projectors": projectors,
        "region_A": region_A,
        "region_B": region_B_t,
        "method": method,
    }


def local_extractable(H, rho, region_B, method="sdp", tol=1e-9, kraus_kwargs=None):
    """W_loc(rho): Bob's extractable energy with no message (vacuum.qet.slp).

    ``method``: 'sdp' (certified, cvxpy), 'kraus' (core-only lower bound) or
    'certificate' (ASRSM analytic upper bound). Zero iff strongly locally
    passive w.r.t. region_B.
    """
    return float(slp._extractable(np.asarray(H, dtype=complex),
                                  np.asarray(rho, dtype=complex),
                                  _check_region(region_B, int(round(np.log2(H.shape[0])))),
                                  method, tol, kraus_kwargs))


def qet_surplus(H, rho, region_A, method="sdp", **kw):
    """Q(rho) = W_->(rho) - W_loc(rho) with Bob = complement. NOT a monotone (see docstring)."""
    H, n = _check_qubit_operator("H", H)
    region_A = _check_region(region_A, n)
    W = qet_potential(H, rho, region_A)["W"]
    W_loc = local_extractable(H, rho, _complement(region_A, n), method=method, **kw)
    return {"Q": W - W_loc, "W_arrow": W, "W_loc": W_loc}


def surplus_counterexample(h, k):
    """Q = W_-> - W_loc is not an LOCC monotone: a local unitary on A raises it.

    In the minimal model with rho_1 = |+x><+x| (x) |0><0| Bob sees the mean
    field H_B + 2k sigma_x from the start, so W_loc = W_-> = h + sqrt(h^2+4k^2)
    and Q(rho_1) = 0. The local unitary U_A: |+x> -> |0> gives
    rho_2 = |0><0| (x) |0><0|: Bob's mean field vanishes, W_loc(rho_2) = 2h,
    while Alice's free sigma_x measurement restores the +-2k sigma_x mean
    field on each branch, W_->(rho_2) = h + sqrt(h^2+4k^2) unchanged, so
    Q(rho_2) = sqrt(h^2+4k^2) - h > 0. Under the FREE class Q is not raised
    by Alice (free unitaries leave both W_-> and W_loc invariant because
    they commute with the bond operator; free instruments only refine the
    same branches) and is never lowered by Bob (Q = m(rho) - Lambda with
    m(rho) = min_E Tr[H_B^loc (id (x) E) rho] non-decreasing under Bob's maps
    and Lambda fixed by Alice's outcome probabilities), so it is not what
    Bob's extraction consumes. Returns the closed forms and the numerically
    computed values (closed-form W_->, W_loc via the reset formula).
    """
    from vacuum.qet import hotta
    h, k = float(h), float(k)
    H = hotta.hamiltonian(h, k)
    ket0 = np.array([1.0, 0.0], dtype=complex)
    plus_x = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    rho_1 = np.kron(np.outer(plus_x, plus_x.conj()), np.outer(ket0, ket0.conj()))
    rho_2 = np.kron(np.outer(ket0, ket0.conj()), np.outer(ket0, ket0.conj()))
    W1 = qet_potential(H, rho_1, (0,))["W"]
    W2 = qet_potential(H, rho_2, (0,))["W"]
    # W_loc of a product state = Tr[H_eff rho_B] - lambda_min(H_eff), H_eff the
    # mean-field Bob Hamiltonian (Bob resets to its ground state)
    def w_loc_product(rho):
        H_B, Xs, Ys = bond_operators(H, (0,))
        rho_A = _partial_trace_B(rho, (0,), 2)
        rho_B = _partial_trace_A(rho, (0,), 2)
        H_eff = _herm(H_B + sum(np.real(np.trace(X @ rho_A)) * Y for X, Y in zip(Xs, Ys)))
        return state_energy(rho_B, H_eff) - float(eigh(H_eff, eigvals_only=True)[0])
    Wl1, Wl2 = w_loc_product(rho_1), w_loc_product(rho_2)
    return {
        "Q_before": W1 - Wl1, "Q_after": W2 - Wl2, "rise": (W2 - Wl2) - (W1 - Wl1),
        "rise_closed_form": np.sqrt(h * h + 4.0 * k * k) - h,
        "W_arrow_before": W1, "W_arrow_after": W2, "W_loc_before": Wl1, "W_loc_after": Wl2,
        "W_loc_after_closed_form": 2.0 * h,
        "W_arrow_closed_form": h + np.sqrt(h * h + 4.0 * k * k),
    }


# ---------------------------------------------------------------------------
# monotone 2: negativity
# ---------------------------------------------------------------------------


def negativity(rho, region_A):
    """N(rho) = (||rho^{Gamma_A}||_1 - 1)/2 = sum of |negative eigenvalues| of rho^{Gamma_A}.

    Vidal-Werner, Phys. Rev. A 65, 032314 (2002), Eq. (3) convention
    (N = 0 iff the partial transpose is positive; for two qubits N = 0 iff
    separable).
    """
    rho, n = _check_qubit_operator("rho", rho)
    region_A = _check_region(region_A, n)
    t = _as_region_blocks(rho, region_A, n)          # t[a, b, a', b']
    dA, dB = t.shape[0], t.shape[1]
    pt = t.transpose(2, 1, 0, 3).reshape(dA * dB, dA * dB)
    w = np.linalg.eigvalsh(_herm(pt))
    return float(np.sum(np.abs(w[w < 0.0])))


# ---------------------------------------------------------------------------
# the numerical monotonicity theorem
# ---------------------------------------------------------------------------


def random_state(n, rng, kind="mixed", rank=None, H=None):
    """Random n-qubit density matrix.

    kind: 'pure', 'mixed' (Hilbert-Schmidt-like, random rank), 'ground_mix'
    (convex mixture of the ground state of H with a random pure state) or
    'thermal' (Gibbs state of H at a random temperature in [0.05, 3]).
    """
    d = 2**n
    if kind == "pure":
        v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        v = v / np.linalg.norm(v)
        return np.outer(v, v.conj())
    if kind == "mixed":
        r = int(rank) if rank is not None else int(rng.integers(1, d + 1))
        A = rng.standard_normal((d, r)) + 1j * rng.standard_normal((d, r))
        R = A @ A.conj().T
        return R / float(np.real(np.trace(R)))
    if H is None:
        raise ValueError(f"kind={kind!r} needs H")
    if kind == "ground_mix":
        w, U = eigh(np.asarray(H, dtype=complex))
        g = np.outer(U[:, 0], U[:, 0].conj())
        lam = float(rng.uniform(0.0, 1.0))
        return lam * g + (1.0 - lam) * random_state(n, rng, "pure")
    if kind == "thermal":
        return slp.gibbs_state(H, float(rng.uniform(0.05, 3.0)))
    raise ValueError(f"unknown kind {kind!r}")


def monotonicity_audit(H, region_A, n_samples=1000, seed=0, tol=1e-9,
                       state_kinds=("pure", "mixed", "ground_mix", "thermal"),
                       n_kraus_max=3, strict=True):
    """Numerical proof of Theorem 1 and of the negativity monotone (Bob = complement).

    For each sample: a random state (cycling through ``state_kinds``) and a
    random free operation drawn from
      'free_instrument'  — commuting-Kraus instrument on A with 1..n_kraus_max
                            Kraus operators (1 = a free unitary), checked with
                            (5a) on average over branches;
      'bob_channel'      — random CPTP map on B (random Kraus, 1..3 ops),
                            checked with the ledger (5b): W(out) + w_B <= W(in);
      'bob_unitary'      — Haar unitary on B, same ledger check.
    Negativity is checked to be non-increasing on average under every one of
    them (it is an LOCC monotone). Violations are max(0, lhs - rhs); the
    audit passes if the worst is <= tol.

    TEETH: the same states are also hit with NON-free operations —
      'nonfree_instrument' — a random (non-commuting) instrument on A,
      'bob_injection'      — a Bob map with w_B < 0 (energy injected) —
    and the largest observed *increase* of W_-> is recorded in
    details['teeth']; a resource theory whose monotone nothing can raise
    would be vacuous, so the tests demand these be > 0.

    Returns an :class:`AuditResult` (name 'qet_monotonicity') whose
    ``details`` carry per-operation worst violations, the teeth, sample
    counts and the energy-signalling defect of every free instrument used
    (its max must be at roundoff).
    """
    H, n = _check_qubit_operator("H", H)
    region_A = _check_region(region_A, n)
    region_B = _complement(region_A, n)
    dA, dB = 2 ** len(region_A), 2 ** len(region_B)
    projectors, _ = free_projectors(H, region_A)
    rng = np.random.default_rng(seed)
    worst = {"free_instrument": 0.0, "bob_channel": 0.0, "bob_unitary": 0.0,
             "negativity": 0.0}
    teeth = {"nonfree_instrument": 0.0, "bob_injection": 0.0,
             "bob_injection_bound_defect": 0.0}
    counts = {k: 0 for k in list(worst) + list(teeth)}
    max_defect = 0.0
    ops = ("free_instrument", "bob_channel", "bob_unitary",
           "nonfree_instrument", "bob_injection")
    for i in range(int(n_samples)):
        kind = state_kinds[i % len(state_kinds)]
        rho = random_state(n, rng, kind, H=H)
        W0 = qet_potential(H, rho, region_A)["W"]
        N0 = negativity(rho, region_A)
        op = ops[i % len(ops)]
        counts[op] += 1
        if op == "free_instrument":
            K = random_free_instrument(projectors, int(rng.integers(1, n_kraus_max + 1)), rng)
            max_defect = max(max_defect, energy_signalling_defect(H, K, region_A))
            br = apply_instrument(rho, K, region_A)
            W1 = sum(p * qet_potential(H, r, region_A)["W"] for p, r in br)
            N1 = sum(p * negativity(r, region_A) for p, r in br)
            worst[op] = max(worst[op], W1 - W0)
            worst["negativity"] = max(worst["negativity"], N1 - N0)
        elif op in ("bob_channel", "bob_unitary", "bob_injection"):
            if op == "bob_unitary":
                K = [random_local_unitary(dB, rng)]
            else:
                K = random_instrument(dB, int(rng.integers(1, 4)), rng)
            wB = bob_energy_out(H, rho, K, region_A)
            out = _herm(apply_local_channel(rho, K, region_B))
            W1 = qet_potential(H, out, region_A)["W"]
            N1 = negativity(out, region_A)
            worst["negativity"] = max(worst["negativity"], N1 - N0)
            if op == "bob_injection":
                if wB < 0.0:
                    teeth[op] = max(teeth[op], W1 - W0)
                    teeth["bob_injection_bound_defect"] = max(
                        teeth["bob_injection_bound_defect"], (W1 - W0) - (-wB))
                else:  # it happened to extract: still a ledger sample
                    worst["bob_channel"] = max(worst["bob_channel"], W1 + wB - W0)
            else:
                worst[op] = max(worst[op], W1 + wB - W0)
        else:  # nonfree_instrument
            K = random_instrument(dA, int(rng.integers(1, 3)), rng)
            br = apply_instrument(rho, K, region_A)
            W1 = sum(p * qet_potential(H, r, region_A)["W"] for p, r in br)
            teeth[op] = max(teeth[op], W1 - W0)
    worst_all = max(max(worst.values()), 0.0)
    passed = worst_all <= float(tol) and max_defect <= 1e-10
    result = AuditResult(
        name="qet_monotonicity",
        passed=passed,
        worst_violation=worst_all,
        details={"worst": worst, "teeth": teeth, "counts": counts,
                 "max_signalling_defect": max_defect, "tol": float(tol),
                 "n_samples": int(n_samples), "seed": int(seed),
                 "region_A": region_A, "region_B": region_B},
    )
    if strict and not passed:
        raise AuditViolation(result)
    return result


# ---------------------------------------------------------------------------
# smooth entropies (Tomamichel, arXiv:1504.00233v5; natural logs)
# ---------------------------------------------------------------------------


def ky_fan_floor(h, s, trace=1.0):
    """kappa(h, s) = min{Tr[h tau] : 0 <= tau <= e^{-s} I, Tr tau = trace}, Eq. (6).

    ``trace`` < 1 is the SUB-NORMALIZED floor needed when the smoothing
    optimizer is taken from Tomamichel's ball S_.(A) (Def. 6.8) rather than
    from its normalized sub-ball; the greedy filling is unchanged, it just
    stops at total weight ``trace``.

    Greedy filling of the lowest levels of h with weight e^{-s} each: with
    m = floor(e^s), kappa = e^{-s} sum_{i<m} lambda_i + (1 - m e^{-s}) lambda_m
    (ascending eigenvalues, 0-based). s <= 0 gives lambda_min; s >= ln d
    gives the mean eigenvalue (only tau = I/d is feasible).
    """
    lam = np.sort(np.real(eigh(np.asarray(h, dtype=complex), eigvals_only=True)))
    d = lam.size
    s = float(s)
    t = float(trace)
    if not 0.0 < t <= 1.0:
        raise ValueError(f"need 0 < trace <= 1, got {t}")
    if s <= 0.0:
        return float(t * lam[0])
    c = float(np.exp(-s))
    if c * d <= t:                 # cap below t/d: infeasible, saturate at (t/d) I
        return float(t * np.mean(lam))
    m = int(np.floor(t / c + 1e-12))
    m = min(m, d - 1)
    return float(c * np.sum(lam[:m]) + (t - m * c) * lam[m])


def min_entropy(rho):
    """H_min(A)_rho = -ln lambda_max(rho) (Def. 6.2 with trivial conditioning), nats."""
    w = eigh(_herm(np.asarray(rho, dtype=complex)), eigvals_only=True)
    return float(-np.log(max(float(w[-1]), 1e-300)))


def _fidelity_capped(r, cap):
    """max sum_i sqrt(r_i s_i) over s_i in [0, cap], sum s_i = 1: s_i = min(cap, t r_i)."""
    r = np.asarray(r, float)
    lo, hi = 0.0, 1.0 / max(float(r.max()), 1e-300) * max(1.0, 1.0 / cap) * 4.0
    for _ in range(200):
        t = 0.5 * (lo + hi)
        tot = float(np.sum(np.minimum(cap, t * r)))
        if tot < 1.0:
            lo = t
        else:
            hi = t
    s = np.minimum(cap, hi * r)
    return float(np.sum(np.sqrt(r * s))) ** 2, s


def smooth_min_entropy(rho, eps, return_state=False, subnormalized=False):
    """H_min^eps(A)_rho over the purified-distance ball, dense (no solver).

    ``subnormalized=True`` uses Tomamichel's own ball (Def. 6.8: tau in
    S_.(A), Tr tau <= 1, P(tau, rho) <= eps) with the generalized fidelity
    F_*(rho, tau) = (Tr|sqrt(rho) sqrt(tau)| + sqrt((1-Tr rho)(1-Tr tau)))^2
    (Def. 3.12) and the purified distance P = sqrt(1 - F_*) (Def. 3.15).

    PROPOSITION E (where the sub-normalized ball actually differs). For a
    NORMALIZED rho the second term of F_* vanishes, so the ball is
    {tau : F(tau, rho) >= 1 - eps^2, Tr tau <= 1} and, F being increasing in
    every eigenvalue of tau, the best total weight at a fixed cap lambda is
    t = min(1, lambda d) — no search is needed. Two regimes follow:

      * lambda >= 1/d: t = 1 and the value is EXACTLY the normalized one;
      * lambda < 1/d: the cap binds everywhere, tau = lambda I, and
        F = lambda (sum_i sqrt(r_i))^2, so the smallest feasible cap is
        lambda = (1 - eps^2) / (sum_i sqrt(r_i))^2.

    Since (sum_i sqrt r_i)^2 <= d with equality iff rho is maximally mixed,
    the second regime is reachable only once the normalized value has
    SATURATED at ln d, and then it adds at most -ln(1 - eps^2). So
    sub-normalized smoothing is worth nothing on the near-pure conditional
    states of the QET protocols (H_min ~ 0), which is why the module's
    one-shot chain was already reporting Tomamichel's value on its inputs;
    the caveat "smoothing is over normalized states" is retired, not by
    changing the number, but by bounding what the enlargement could ever buy.
    Measured in tests/test_resource_theory.py and in data/s8_entropic.npz.

    Def. 6.9 with the ball of Def. 6.8 restricted to normalized states (a
    subset of Tomamichel's, so the value is <= his and the one-shot bound
    (iii) stays valid). The optimizer can be taken diagonal in rho's
    eigenbasis: pinching there fixes rho, contracts the purified distance
    (Eq. (3.49)) and does not raise lambda_max. The remaining problem — the
    smallest cap lambda on the eigenvalues s_i compatible with
    F(s, r) = (sum_i sqrt(s_i r_i))^2 >= 1 - eps^2 — is solved by bisection
    on lambda with the inner water-filling s_i = min(lambda, t r_i).
    eps = 0 returns :func:`min_entropy`. Nats.
    """
    rho = _herm(np.asarray(rho, dtype=complex))
    r, U = eigh(rho)
    r = np.clip(np.real(r), 0.0, None)
    tr_rho = float(np.sum(r))
    if not subnormalized:
        r = r / tr_rho
    eps = float(eps)
    d = r.size
    if subnormalized:
        return _smooth_min_entropy_subnorm(r, U, tr_rho, eps, return_state)
    if not 0.0 <= eps < 1.0:
        raise ValueError(f"need 0 <= eps < 1, got {eps}")
    target = 1.0 - eps * eps
    if eps == 0.0:
        cap = float(r.max())
        s = r
    else:
        f_uniform, _ = _fidelity_capped(r, 1.0 / d)
        if f_uniform >= target:
            cap, s = 1.0 / d, np.full(d, 1.0 / d)
        else:
            lo, hi = 1.0 / d, float(r.max())
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                f, _ = _fidelity_capped(r, mid)
                if f >= target:
                    hi = mid
                else:
                    lo = mid
            cap = hi
            _, s = _fidelity_capped(r, cap)
    value = float(-np.log(cap))
    if return_state:
        return value, (U * s) @ U.conj().T
    return value


def conditional_min_entropy_cq(probs, states):
    """H_min(A|Y) = -ln sum_y p(y) exp(-H_min(A)_{tau_y}) for a cq state (Eq. (6.25))."""
    tot = 0.0
    for p, tau in zip(probs, states):
        tot += float(p) * float(np.exp(-min_entropy(tau)))
    return float(-np.log(max(tot, 1e-300)))


def _purify(rho_AB):
    """|psi> on AB (x) C with C = rank(rho) columns of the eigenbasis."""
    w, U = eigh(_herm(np.asarray(rho_AB, dtype=complex)))
    keep = w > 1e-14
    w, U = w[keep], U[:, keep]
    dC = int(w.size)
    psi = (U * np.sqrt(w)).reshape(-1)       # index (ab, c) -> ab * dC + c
    return psi, dC


def min_entropy_sdp(rho_AB, dims, eps=0.0, solver=None):
    """H_min^eps(A|B)_rho by semidefinite programming (cvxpy, guarded), nats.

    eps = 0: Lemma 6.3, Eq. (6.8) — primal min Tr(sigma_B) s.t.
    I_A (x) sigma_B >= rho_AB, and its dual max Tr(rho X) s.t.
    Tr_A X <= I_B, X >= 0; both are solved and returned, so the bracket
    [-ln primal, -ln dual] certifies the value (strong duality holds).
    eps > 0: the smooth primal, Eq. (6.37), with the ball restricted to
    normalized states (Tr rho~ = 1) over a purification rho_ABC:
    min Tr(sigma_B) s.t. I_A (x) sigma_B >= Tr_C rho~, Tr(rho~ psi) >= 1 - eps^2.

    Returns dict: 'H_min' (-ln primal), 'H_min_dual' (eps = 0 only), 'gap',
    'status', 'solver', 'eps'.
    """
    cp = slp._load_cvxpy()
    dA, dB = int(dims[0]), int(dims[1])
    rho = _herm(np.asarray(rho_AB, dtype=complex))
    if rho.shape != (dA * dB, dA * dB):
        raise ValueError(f"rho must be {dA * dB} x {dA * dB} for dims {dims}")
    eps = float(eps)
    candidates = [solver] if solver else [s for s in ("SCS", "CLARABEL") if s in cp.installed_solvers()]
    opts = {"SCS": {"eps_abs": 1e-11, "eps_rel": 1e-11, "max_iters": 50_000},
            "CLARABEL": {"tol_gap_abs": 1e-12, "tol_gap_rel": 1e-12}}
    out = None
    for name in candidates:
        sigma = cp.Variable((dB, dB), hermitian=True)
        cons = [sigma >> 0]
        if eps == 0.0:
            cons.append(cp.kron(np.eye(dA), sigma) - rho >> 0)
        else:
            psi, dC = _purify(rho)
            Psi = np.outer(psi, psi.conj())
            rt = cp.Variable((dA * dB * dC, dA * dB * dC), hermitian=True)
            cons += [rt >> 0, cp.real(cp.trace(rt)) == 1.0,
                     cp.real(cp.trace(Psi @ rt)) >= 1.0 - eps * eps,
                     cp.kron(np.eye(dA), sigma) - cp.partial_trace(rt, (dA * dB, dC), 1) >> 0]
        prob = cp.Problem(cp.Minimize(cp.real(cp.trace(sigma))), cons)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*nested list.*")
                warnings.filterwarnings("ignore", message=".*may be inaccurate.*")
                prob.solve(solver=name, **opts.get(name, {}))
        except (cp.error.SolverError, cp.error.DCPError):
            continue
        if sigma.value is None:
            continue
        primal = float(np.real(np.trace(sigma.value)))
        # repair: I (x) sigma >= rho exactly by shifting sigma up by the
        # most negative eigenvalue of the slack (gives a feasible, hence
        # valid, upper bound on the primal -> lower bound on H_min)
        if eps == 0.0:
            slack = _herm(np.kron(np.eye(dA), sigma.value) - rho)
            mu = float(eigh(slack, eigvals_only=True)[0])
            primal = float(np.real(np.trace(sigma.value))) + dB * max(0.0, -mu)
        res = {"H_min": float(-np.log(primal)), "status": prob.status, "solver": name,
               "eps": eps, "gap": np.nan, "H_min_dual": np.nan}
        if eps == 0.0:
            X = cp.Variable((dA * dB, dA * dB), hermitian=True)
            dual = cp.Problem(cp.Maximize(cp.real(cp.trace(rho @ X))),
                              [X >> 0, np.eye(dB) - cp.partial_trace(X, (dA, dB), 0) >> 0])
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*nested list.*")
                    warnings.filterwarnings("ignore", message=".*may be inaccurate.*")
                    dual.solve(solver=name, **opts.get(name, {}))
                Xv = _herm(X.value)
                w, Uv = eigh(Xv)
                Xv = (Uv * np.clip(w, 0.0, None)) @ Uv.conj().T
                trA = np.einsum("abad->bd", Xv.reshape(dA, dB, dA, dB))
                scale = max(1.0, float(eigh(_herm(trA), eigvals_only=True)[-1]))
                dval = float(np.real(np.trace(rho @ Xv))) / scale  # feasible -> valid
                res["H_min_dual"] = float(-np.log(dval))
                res["gap"] = abs(res["H_min"] - res["H_min_dual"])
            except (cp.error.SolverError, cp.error.DCPError):
                pass
        out = res
        break
    if out is None:
        raise RuntimeError("no SDP solver produced a solution for H_min")
    return out


def max_entropy_sdp(rho_AB, dims, eps=0.0, solver=None):
    """H_max^eps(A|B)_rho = -H_min^eps(A|C)_psi over a purification (Lemma 6.7 / Prop. 6.14).

    Builds psi_ABC, traces out B, and calls :func:`min_entropy_sdp` on the
    AC marginal with the same eps (smooth duality, Eq. (6.51); the
    normalized-ball restriction is inherited). Returns the min-entropy
    result dict with 'H_max' added.
    """
    dA, dB = int(dims[0]), int(dims[1])
    psi, dC = _purify(rho_AB)
    t = psi.reshape(dA, dB, dC)
    rho_AC = np.einsum("abc,dbe->acde", t, t.conj()).reshape(dA * dC, dA * dC)
    res = min_entropy_sdp(rho_AC, (dA, dC), eps=eps, solver=solver)
    res["H_max"] = -res["H_min"]
    return res


def max_entropy_direct(rho_AB, dims, restarts=6, seed=0):
    """H_max(A|B) = max_sigma ln F(rho_AB, I (x) sigma_B), Def. 6.4 (core-only).

    F(rho, tau) = ||sqrt(rho) sqrt(tau)||_1^2 (Def. 3.9). The maximum over
    sigma_B is found by L-BFGS over a Cholesky factor (F is concave in
    sigma, so restarts only guard the parametrization). A restart-certified
    lower bound; the SDP route is the certified one.
    """
    from scipy.linalg import sqrtm
    from scipy.optimize import minimize
    dA, dB = int(dims[0]), int(dims[1])
    rho = _herm(np.asarray(rho_AB, dtype=complex))
    w, U = eigh(rho)
    sq_rho = (U * np.sqrt(np.clip(w, 0.0, None))) @ U.conj().T
    rng = np.random.default_rng(seed)

    def fid(x):
        L = (x[: dB * dB] + 1j * x[dB * dB:]).reshape(dB, dB)
        sig = L @ L.conj().T
        sig = sig / max(float(np.real(np.trace(sig))), 1e-300)
        tau = np.kron(np.eye(dA), sig)
        M = sq_rho @ tau @ sq_rho
        wm = eigh(_herm(M), eigvals_only=True)
        return float(np.sum(np.sqrt(np.clip(wm, 0.0, None)))) ** 2

    best = 0.0
    for _ in range(int(restarts)):
        x0 = rng.standard_normal(2 * dB * dB)
        res = minimize(lambda x: -fid(x), x0, method="L-BFGS-B",
                       options={"maxiter": 400, "ftol": 1e-15, "gtol": 1e-11})
        best = max(best, -float(res.fun))
    return float(np.log(best))


# ---------------------------------------------------------------------------
# the one-shot bound, Eqs. (i)-(v)
# ---------------------------------------------------------------------------


def one_shot_bound(h_R, branches, eps_grid=(0.0, 0.01, 0.03, 0.1, 0.2, 0.3),
                   subnormalized=False):
    """Upper bound on Bob's unitary extraction given the bit: chain (ii)-(iii), (v).

    ``subnormalized=True`` smooths over Tomamichel's own ball (Def. 6.8,
    S_.(A), Prop. E) instead of its normalized sub-ball. The chain must then
    be re-derived, because the optimizer tau~ carries trace t <= 1:

        Tr[h U tau U^dag] = Tr[h U tau~ U^dag] + Tr[h U (tau - tau~) U^dag]
                         >= kappa(h, H_min^eps, t) - ||h||_inf ||tau - tau~||_1
                         >= kappa(h, H_min^eps, t) - 2 eps ||h||_inf,          (iii')

    using the sub-normalized Ky Fan floor at trace t (:func:`ky_fan_floor`
    with ``trace``) and the generalized trace distance of Lemma 3.17,
    Delta <= P <= eps with ||tau - tau~||_1 <= 2 Delta. h is shifted to
    lambda_min = 0 first (the extraction itself is shift-invariant, both
    states having trace one, but kappa at trace t < 1 is not), so
    ||h||_inf = spread and the penalty is 2 eps spread against the normalized
    convention's eps spread. The enlarged ball therefore buys at most
    -ln(1 - eps^2) of entropy (Prop. E) and pays twice the penalty plus the
    trace-t floor: measured never to help on any row here.

    Parameters
    ----------
    h_R : (d, d) Hermitian, or a list of one per branch (Bob = complement,
        where the bond eigenvalue enters his effective Hamiltonian).
    branches : sequence of (p_mu, tau_mu), tau_mu normalized states on R.
    eps_grid : smoothing parameters to try; 0 must be included for (ii).

    Returns dict: 'bound_exact' (ii), 'bound_eps' (array over eps_grid,
    (iii)), 'bound' = min over the grid, 'eps_best', 'ceiling' (CPTP
    ceiling sum p (Tr h tau - lambda_min)), 'bound_cq' (v; NaN when h_R
    varies across branches), 'H_min' (per branch), 'H_min_cq', 'spread'.
    """
    branches = [(float(p), _herm(np.asarray(t, dtype=complex))) for p, t in branches]
    hs = list(h_R) if isinstance(h_R, (list, tuple)) else [h_R] * len(branches)
    hs = [_herm(np.asarray(h, dtype=complex)) for h in hs]
    eps_grid = [float(e) for e in eps_grid]
    if 0.0 not in eps_grid:
        eps_grid = [0.0] + eps_grid
    hmins = [min_entropy(t) for _, t in branches]
    energies = [state_energy(t, h) for (_, t), h in zip(branches, hs)]
    lam = [eigh(h, eigvals_only=True) for h in hs]
    spread = [float(w[-1] - w[0]) for w in lam]
    ceiling = sum(p * (e - float(w[0])) for (p, _), e, w in zip(branches, energies, lam))
    bound_eps = []
    for eps in eps_grid:
        tot = 0.0
        for (p, t), h, e, sp, w in zip(branches, hs, energies, spread, lam):
            if eps == 0.0:
                s = min_entropy(t)
                tot += p * (e - ky_fan_floor(h, s))
            elif not subnormalized:
                s = smooth_min_entropy(t, eps)
                tot += p * (e - ky_fan_floor(h, s) + eps * sp)
            else:
                s, tau_s = smooth_min_entropy(t, eps, return_state=True,
                                              subnormalized=True)
                weight = min(1.0, max(float(np.real(np.trace(tau_s))), 1e-12))
                h0 = h - float(w[0]) * np.eye(h.shape[0])       # lambda_min(h0) = 0
                e0 = e - float(w[0])
                tot += p * (e0 - ky_fan_floor(h0, s, trace=weight) + 2.0 * eps * sp)
        bound_eps.append(tot)
    bound_eps = np.array(bound_eps)
    i_best = int(np.argmin(bound_eps))
    same_h = all(np.max(np.abs(h - hs[0])) < 1e-12 for h in hs)
    H_cq = conditional_min_entropy_cq([p for p, _ in branches], [t for _, t in branches])
    if same_h:
        rho_R = sum(p * t for p, t in branches)
        bound_cq = state_energy(rho_R, hs[0]) - ky_fan_floor(hs[0], H_cq)
    else:
        bound_cq = np.nan
    return {
        "bound_exact": float(bound_eps[0]),
        "bound_eps": bound_eps, "eps_grid": np.array(eps_grid),
        "bound": float(bound_eps[i_best]), "eps_best": eps_grid[i_best],
        "ceiling": float(ceiling), "bound_cq": float(bound_cq),
        "H_min": np.array(hmins), "H_min_cq": H_cq, "spread": np.array(spread),
    }


def classical_min_max_entropy(p_ab):
    """Closed forms for a classical-classical state p(a, b) (rows a, columns b), nats.

    H_min(A|B) = -ln sum_b max_a p(a,b)  (Eq. (6.27), the guessing probability);
    H_max(A|B) =  ln sum_b (sum_a sqrt(p(a,b)))^2  (Def. 6.4 with diagonal
    sigma_B, the inner maximization solved by Cauchy-Schwarz).
    """
    p = np.asarray(p_ab, float)
    return (float(-np.log(np.sum(np.max(p, axis=0)))),
            float(np.log(np.sum(np.sum(np.sqrt(p), axis=0) ** 2))))


def one_shot_bound_marginal(h_R, branches, dims, sub_B):
    """Tighter (SDP) floor for a unitary on the sub-factor sub_B of R: cvxpy, guarded.

    A unitary on sub_B (x) id preserves both the spectrum of tau_mu and the
    marginal on the untouched factor, so
        kappa_marg := min { Tr[h tau'] : 0 <= tau' <= lambda_max(tau_mu) I,
                            Tr tau' = 1, Tr_{sub_B} tau' = Tr_{sub_B} tau_mu }
    is a valid relaxation of the unitary orbit and kappa_marg >= kappa of
    Eq. (6). ``dims`` = (d_sub_B, d_rest) with R ordered (sub_B, rest);
    ``sub_B`` is a documentation echo. Returns dict 'bound' =
    sum p (Tr[h tau] - kappa_marg), 'floors', 'status'.
    """
    cp = slp._load_cvxpy()
    d1, d2 = int(dims[0]), int(dims[1])
    h = _herm(np.asarray(h_R, dtype=complex))
    floors, total = [], 0.0
    status = None
    for p, tau in branches:
        tau = _herm(np.asarray(tau, dtype=complex))
        cap = float(eigh(tau, eigvals_only=True)[-1])
        marg = np.einsum("abad->bd", tau.reshape(d1, d2, d1, d2))
        T = cp.Variable((d1 * d2, d1 * d2), hermitian=True)
        cons = [T >> 0, cap * np.eye(d1 * d2) - T >> 0, cp.real(cp.trace(T)) == 1.0,
                cp.partial_trace(T, (d1, d2), 0) == marg]
        prob = cp.Problem(cp.Minimize(cp.real(cp.trace(h @ T))), cons)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*nested list.*")
            warnings.filterwarnings("ignore", message=".*may be inaccurate.*")
            prob.solve(solver="SCS", eps_abs=1e-10, eps_rel=1e-10, max_iters=50_000)
        status = prob.status
        floor = float(prob.value)
        floors.append(floor)
        total += float(p) * (state_energy(tau, h) - floor)
    return {"bound": total, "floors": np.array(floors), "status": status}


# ---------------------------------------------------------------------------
# ledgers: the theory on the two validated protocols
# ---------------------------------------------------------------------------


def hotta_resource_ledger(h, k, eps_grid=(0.0, 0.01, 0.03, 0.1, 0.2, 0.3)):
    """W_->, E_B, W_loc, negativity and the one-shot bound on the minimal model at (h, k).

    E_B is Hotta's closed form (arXiv:1101.3954 Eq. (11)) AND the coded
    protocol's numerical optimum (vacuum.qet.hotta.optimize_e_b); the bound
    uses Bob's conditional (pure) states and per-branch effective
    Hamiltonians. Returns a flat dict of floats plus 'H_min' (per branch).
    """
    from vacuum.qet import hotta
    h, k = float(h), float(k)
    H = hotta.hamiltonian(h, k)
    g = hotta.ground_state(h, k)
    rho = np.outer(g, g.conj())
    pot = qet_potential(H, rho, (0,))
    hs = [b["H_eff"] for b in pot["branches"]]
    br = [(b["p"], b["sigma_B"]) for b in pot["branches"]]
    osb = one_shot_bound(hs, br, eps_grid=eps_grid)
    e_b = hotta.e_b_optimal(h, k)
    _, e_b_num = hotta.optimize_e_b(h, k)
    e_a = hotta.e_a_closed(h, k)
    cert = slp.slp_certificate(H, rho, 1)
    return {
        "h": h, "k": k, "W_arrow": pot["W"], "E_B": float(e_b), "E_B_protocol": float(e_b_num),
        "E_A": float(e_a), "W_over_E_A": pot["W"] / e_a,
        "W_loc_bound": cert["extractable_bound"], "slp": bool(cert["slp"]),
        "negativity": negativity(rho, (0,)), "concurrence": hotta.concurrence(g),
        "bound_exact": osb["bound_exact"], "bound_best": osb["bound"],
        "eps_best": osb["eps_best"], "ceiling": osb["ceiling"],
        "ratio_exact": float(e_b) / osb["bound_exact"],
        "ratio_best": float(e_b) / osb["bound"],
        "H_min": osb["H_min"], "bound_eps": osb["bound_eps"],
    }


def chain_resource_ledger(L, g, A, B_list, J=1.0, method="kraus", kraus_kwargs=None,
                          eps_grid=(0.0, 0.01, 0.03, 0.1, 0.2, 0.3)):
    """The theory on the TFIM chain protocol (vacuum.qet.chains), one row per Bob site.

    Per B: the achieved E_B (single-site conditional rotation), the
    single-site potential W_site (Bob's CPTP optimum on site B, via
    ``method`` 'sdp' / 'kraus' / None to skip), the one-shot bound on the
    region R = {B-1, B, B+1} (h_R = the chain's local_hamiltonian, tau_mu
    its conditional states), the aggregated cq bound (v), the CPTP
    ceiling on R, and the A|B negativity of the ground state. The
    Bob = complement potential W_arrow_full (Alice-only quantity) and E_A
    are returned once.
    """
    from vacuum.qet import chains
    L = int(L)
    H = np.asarray(chains.tfim_hamiltonian(L, g, J).toarray(), dtype=complex)
    E0, psi = chains.tfim_ground_state(L, g, J)
    h_ops = chains.tfim_local_energy_ops(L, g, J)
    rho = np.outer(psi, psi.conj()).astype(complex)
    A = int(A)
    full = qet_potential(H, rho, (A,))
    rows = []
    E_A = None
    for B in B_list:
        B = int(B)
        led = chains.run_chain_qet(L, g, A, B, J=J, ground=(E0, psi), h_ops=h_ops,
                                   theta_method="closed")
        E_A = led["E_A"]
        R = tuple(q for q in (B - 1, B, B + 1) if 0 <= q < L)
        h_loc = np.asarray(chains.local_hamiltonian(h_ops, B).toarray(), dtype=complex)
        h_R = _partial_trace_B(h_loc, R, L) / 2 ** (L - len(R))
        branches = []
        for mu, o in led["outcomes"].items():
            tau = chains.reduced_density(o["state_meas"].astype(complex), L, list(R))
            branches.append((o["p"], tau))
        osb = one_shot_bound(h_R, branches, eps_grid=eps_grid)
        if method is None:
            W_site = np.nan
        else:
            W_site = qet_potential(H, rho, (A,), region_B=(B,), method=method,
                                   kraus_kwargs=kraus_kwargs)["W"]
        rho_AB = chains.reduced_density(psi.astype(complex), L, [A, B])
        rows.append({
            "B": B, "d": abs(B - A), "E_B": led["E_B"], "W_site": float(W_site),
            "R": R, "bound_exact": osb["bound_exact"], "bound_best": osb["bound"],
            "eps_best": osb["eps_best"], "bound_cq": osb["bound_cq"],
            "ceiling_R": osb["ceiling"], "H_min": osb["H_min"], "H_min_cq": osb["H_min_cq"],
            "bound_eps": osb["bound_eps"],
            "negativity_AB": negativity(rho_AB, (0,)),
            "ratio_exact": led["E_B"] / osb["bound_exact"],
            "ratio_best": led["E_B"] / osb["bound"],
            "ratio_cq": led["E_B"] / osb["bound_cq"],
            "ratio_full": led["E_B"] / full["W"],
            "ratio_site": (led["E_B"] / W_site) if method is not None else np.nan,
        })
    return {"L": L, "g": float(g), "J": float(J), "A": A, "E_A": E_A,
            "W_arrow_full": full["W"], "rows": rows}


# ---------------------------------------------------------------------------
# Gaussian mirror (vacuum.core conventions: block ordering, V_vac = I/2)
# ---------------------------------------------------------------------------


def _quad_idx(N, modes):
    m = np.asarray(list(modes), int)
    return np.concatenate([m, m + N])


def _bob_local_energy_gaussian(V, K, modes_A, modes_B):
    """<H_B^loc> = (1/2)Tr V_pp,B + (1/2)Tr(K_BB V_xx,B) + sum_{a in A, b in B} K_ab V_xx[a,b]."""
    N = K.shape[0]
    B = list(modes_B)
    A = list(modes_A)
    Vxx = V[:N, :N]
    Vpp = V[N:, N:]
    return (0.5 * float(np.trace(Vpp[np.ix_(B, B)]))
            + 0.5 * float(np.trace(K[np.ix_(B, B)] @ Vxx[np.ix_(B, B)]))
            + float(np.sum(K[np.ix_(A, B)] * Vxx[np.ix_(A, B)])))


def gaussian_qet_potential(K, modes_A, V=None):
    """W_-> for a mean-zero Gaussian state of H = (1/2)(p.p + x.K.x), Bob = complement.

    Alice's free instrument is homodyne detection of the bond quadratures
    x_a (a in A with K[a, B] != 0); given the outcome vector xi ~ N(0,
    Sigma = V_xx[bond, bond]) Bob's conditional state is Gaussian with
    mean G xi (G = V[B, xi] Sigma^{-1}) and covariance the Schur complement,
    and his effective Hamiltonian H^xi = (1/2)p_B.p_B + (1/2)x_B K_BB x_B
    + xi^T K_{bond,B} x_B has ground energy E_0(K_BB) - (1/2) xi^T K_{bond,B}
    K_BB^{-1} K_{bond,B}^T xi. Integrating Bob's reset yield over xi gives

        W = <H_B^loc>_V - E_0(K_BB) + (1/2) Tr[K_{bond,B} K_BB^{-1} K_{bond,B}^T Sigma],

    returned as 'W' (from the explicit conditional-state integral) and
    'W_identity' (from this formula); the two agree to roundoff. V defaults
    to the ground state, for which W_loc = 0 (passivity) and W is the
    Gaussian QET surplus. Also returns 'log_negativity' (vacuum.core).
    """
    from vacuum.core import ground_state_cov, log_negativity, mean_energy
    K = np.asarray(K, float)
    N = K.shape[0]
    modes_A = tuple(sorted(int(a) for a in modes_A))
    modes_B = tuple(b for b in range(N) if b not in modes_A)
    if V is None:
        V = ground_state_cov(K)
    V = np.asarray(V, float)
    K_BB = K[np.ix_(modes_B, modes_B)]
    bond = [a for a in modes_A if np.any(np.abs(K[a, list(modes_B)]) > 0.0)]
    if not bond:
        raise ValueError("A is decoupled from B")
    Kb = K[np.ix_(bond, modes_B)]                         # (n_bond, n_B)
    iB = _quad_idx(N, modes_B)
    ix = np.asarray(bond, int)                             # x quadratures of the bond modes
    Sigma = V[np.ix_(ix, ix)]
    Cov = V[np.ix_(iB, ix)]                                # (2 n_B, n_bond)
    G = Cov @ np.linalg.inv(Sigma)
    V_B = V[np.ix_(iB, iB)]
    V_c = V_B - G @ Cov.T
    nB = len(modes_B)
    E0 = mean_energy(ground_state_cov(K_BB), K_BB)
    Gx, Gp = G[:nB], G[nB:]
    M = (0.5 * Gp.T @ Gp + 0.5 * Gx.T @ K_BB @ Gx + Kb @ Gx
         + 0.5 * Kb @ np.linalg.solve(K_BB, Kb.T))
    W = mean_energy(V_c, K_BB) - E0 + float(np.trace(M @ Sigma))
    e_loc = _bob_local_energy_gaussian(V, K, modes_A, modes_B)
    W_id = e_loc - E0 + 0.5 * float(np.trace(Kb @ np.linalg.solve(K_BB, Kb.T) @ Sigma))
    return {"W": float(W), "W_identity": float(W_id), "E_B_loc": float(e_loc),
            "E0_B": float(E0), "bond_modes": tuple(bond),
            "log_negativity": float(log_negativity(V, list(modes_A), list(modes_B)))}


def _random_symplectic(N_modes, rng, zero_quads=(), scale=0.5):
    """exp(Omega M) for a random symmetric M with the listed quadrature rows/cols zeroed."""
    from vacuum.core import Omega
    from scipy.linalg import expm
    d = 2 * N_modes
    M = rng.standard_normal((d, d))
    M = scale * (M + M.T) / 2.0
    for q in zero_quads:
        M[q, :] = 0.0
        M[:, q] = 0.0
    return expm(Omega(N_modes) @ M)


def gaussian_monotonicity_audit(K, modes_A, n_samples=300, seed=0, tol=1e-9, strict=True):
    """Theorem 1 and the log-negativity monotone on Gaussian states (mean zero).

    States: the ground state evolved by a random global symplectic, or a
    thermal state. Free Alice operations: symplectics on A generated by
    quadratic forms with no p_bond dependence (they commute with every bond
    quadrature x_a, so they are energy-non-signalling). Bob operations:
    random symplectics on B (ledger (5b) with w_B) and pure loss on a Bob
    mode (ledger). Log-negativity must not increase under any of them.
    Teeth: a shear that moves x_bond (non-free) raises W.
    """
    from vacuum.core import ground_state_cov, log_negativity, loss, thermal_state_cov
    K = np.asarray(K, float)
    N = K.shape[0]
    modes_A = tuple(sorted(int(a) for a in modes_A))
    modes_B = tuple(b for b in range(N) if b not in modes_A)
    rng = np.random.default_rng(seed)
    Vg = ground_state_cov(K)
    iA, iB = _quad_idx(N, modes_A), _quad_idx(N, modes_B)
    nA, nB = len(modes_A), len(modes_B)
    bond = gaussian_qet_potential(K, modes_A)["bond_modes"]
    p_bond_local = [nA + modes_A.index(a) for a in bond]

    def embed(S_local, idx):
        S = np.eye(2 * N)
        S[np.ix_(idx, idx)] = S_local
        return S

    worst = {"free_alice": 0.0, "bob_symplectic": 0.0, "bob_loss": 0.0, "log_negativity": 0.0}
    teeth = {"nonfree_alice": 0.0}
    for i in range(int(n_samples)):
        if i % 2 == 0:
            S = _random_symplectic(N, rng, scale=0.3)
            V = S @ Vg @ S.T
        else:
            V = thermal_state_cov(K, float(rng.uniform(0.1, 2.0)))
        W0 = gaussian_qet_potential(K, modes_A, V)["W"]
        E0 = log_negativity(V, list(modes_A), list(modes_B))
        op = ("free_alice", "bob_symplectic", "bob_loss", "nonfree_alice")[i % 4]
        if op == "free_alice":
            S = embed(_random_symplectic(nA, rng, zero_quads=p_bond_local), iA)
            V1 = S @ V @ S.T
            worst[op] = max(worst[op], gaussian_qet_potential(K, modes_A, V1)["W"] - W0)
        elif op == "bob_symplectic":
            S = embed(_random_symplectic(nB, rng), iB)
            V1 = S @ V @ S.T
            wB = (_bob_local_energy_gaussian(V, K, modes_A, modes_B)
                  - _bob_local_energy_gaussian(V1, K, modes_A, modes_B))
            worst[op] = max(worst[op], gaussian_qet_potential(K, modes_A, V1)["W"] + wB - W0)
        elif op == "bob_loss":
            V1 = loss(V, [int(rng.choice(modes_B))], float(rng.uniform(0.2, 0.95)))
            wB = (_bob_local_energy_gaussian(V, K, modes_A, modes_B)
                  - _bob_local_energy_gaussian(V1, K, modes_A, modes_B))
            worst[op] = max(worst[op], gaussian_qet_potential(K, modes_A, V1)["W"] + wB - W0)
        else:
            S = embed(_random_symplectic(nA, rng), iA)   # generic: moves x_bond
            V1 = S @ V @ S.T
            teeth[op] = max(teeth[op], gaussian_qet_potential(K, modes_A, V1)["W"] - W0)
            continue
        worst["log_negativity"] = max(worst["log_negativity"],
                                      log_negativity(V1, list(modes_A), list(modes_B)) - E0)
    worst_all = max(worst.values())
    passed = worst_all <= float(tol)
    result = AuditResult(name="gaussian_qet_monotonicity", passed=passed,
                         worst_violation=max(0.0, worst_all),
                         details={"worst": worst, "teeth": teeth, "n_samples": int(n_samples),
                                  "seed": int(seed), "tol": float(tol), "modes_A": modes_A})
    if strict and not passed:
        raise AuditViolation(result)
    return result


# ---------------------------------------------------------------------------
# L3 close-out A: the free class beyond unitality (Theorem 2)
# ---------------------------------------------------------------------------


def _embed_pair(op_AB, region_A, n):
    """An operator written in (A, B) factor order -> the site-ordered 2^n basis."""
    region_A = tuple(sorted(int(q) for q in region_A))
    rest = [q for q in range(n) if q not in region_A]
    order = list(region_A) + rest
    inv = np.argsort(order)
    t = np.asarray(op_AB, dtype=complex).reshape((2,) * (2 * n))
    t = np.transpose(t, list(inv) + [n + q for q in inv])
    return t.reshape(2**n, 2**n)


def free_class_extremality(H, region_A, tol=1e-9):
    """Theorem 2's hypothesis: are the joint bond eigenvalue vectors in convex position?

    The bond operators {X_j} on A commute, with joint eigenprojectors {Pi_a}
    and eigenvalue vectors x(a) in R^m (:func:`free_projectors`). Write
    K^{ab} = Pi_a K Pi_b and M_ab = sum_mu (K_mu^{ab})^dag K_mu^{ab} >= 0.
    Completeness and Eq. (2) read, block-column by block-column,

        sum_a M_ab = Pi_b,     sum_a x(a) M_ab = x(b) Pi_b,                    (7)

    i.e. for every unit vector v in the range of Pi_b the numbers
    p_a = <v|M_ab|v> are a probability distribution with barycentre x(b).

    THEOREM 2. If x(b) is an EXTREME point of conv{x(a)}, then M_ab = 0 for
    a != b, i.e. every Kraus operator satisfies Pi_a K_mu Pi_b = 0. If every
    x(b) is extreme, every energy-non-signalling instrument on A is therefore
    commuting-Kraus — with NO unitality assumption, superseding the
    Arias-Gheondea-Gudder route of the module docstring. Conversely, if some
    x(b) lies in the convex hull of the others, x(b) = sum_a lambda_a x(a),
    then :func:`nonfree_energy_nonsignalling_instrument` builds an explicit
    NON-unital energy-non-signalling instrument that is not commuting-Kraus.
    So the commuting-Kraus class is exactly the free class iff the bond
    spectrum is in convex position.

    A single-qubit A always qualifies (two eigenvalues +-|x| of a traceless
    X), and so does every region of the TFIM chains here (the bond operators
    are commuting sigma_x's, whose joint eigenvalues are the vertices of a
    hypercube) — Prop. A and Theorem 1 are untouched on every model this
    module ships. :func:`star_hamiltonian` is a model that fails it.

    Returns dict: 'in_convex_position' (bool), 'points' (n_blocks, n_bonds),
    'extreme' (bool per block), 'weights' (per non-extreme block: the convex
    weights on the others, else None), 'ranks', 'region_A'.
    """
    from scipy.optimize import linprog
    H, n = _check_qubit_operator("H", H)
    region_A = _check_region(region_A, n)
    projectors, xs = free_projectors(H, region_A)
    pts = np.atleast_2d(np.asarray(xs, dtype=float))
    n_blocks = len(projectors)
    extreme, weights = [], []
    for i in range(n_blocks):
        if n_blocks == 1 or pts.shape[1] == 0:
            extreme.append(True)
            weights.append(None)
            continue
        others = np.delete(pts, i, axis=0)
        A_eq = np.vstack([others.T, np.ones(others.shape[0])])
        b_eq = np.concatenate([pts[i], [1.0]])
        res = linprog(np.zeros(others.shape[0]), A_eq=A_eq, b_eq=b_eq,
                      bounds=[(0.0, None)] * others.shape[0])
        ok = bool(res.success) and float(np.max(np.abs(A_eq @ res.x - b_eq))) < tol
        extreme.append(not ok)
        weights.append(np.asarray(res.x, float) if ok else None)
    return {
        "in_convex_position": bool(all(extreme)),
        "points": pts, "extreme": extreme, "weights": weights,
        "ranks": [int(round(float(np.real(np.trace(P))))) for P in projectors],
        "region_A": region_A,
    }


def star_hamiltonian(h, k, n_alice=2):
    """A model whose bond spectrum is NOT in convex position (Theorem 2's converse).

    n_alice Alice qubits 0..n_alice-1 and one Bob qubit, all in a transverse
    field, with EVERY Alice site coupled to Bob through the same bond:

        H = h sum_i sigma_z^i + k (sum_{i in A} sigma_x^i) (x) sigma_x^B.

    The single bond operator on A is proportional to sum_i sigma_x^i, whose
    spectrum {n_alice, n_alice-2, ..., -n_alice} has interior points as soon
    as n_alice >= 2 — the eigenvalue 0 of two Alice sites is the midpoint of
    +-2. Returns the (2^(n_alice+1), 2^(n_alice+1)) Hamiltonian, Bob last.
    """
    n_alice = int(n_alice)
    if n_alice < 1:
        raise ValueError(f"need at least one Alice qubit, got {n_alice}")
    n = n_alice + 1
    h, k = float(h), float(k)

    def site(op, i):
        out = np.array([[1.0 + 0.0j]])
        for q in range(n):
            out = np.kron(out, op if q == i else np.eye(2, dtype=complex))
        return out

    H = sum(h * site(np.diag([1.0, -1.0]).astype(complex), i) for i in range(n))
    for i in range(n_alice):
        blk = np.array([[1.0 + 0.0j]])
        for q in range(n):
            blk = np.kron(blk, _SX if q in (i, n - 1) else np.eye(2, dtype=complex))
        H = H + k * blk
    return _herm(H)


def nonfree_energy_nonsignalling_instrument(H, region_A, tol=1e-9):
    """A NON-unital energy-non-signalling instrument on A that is not commuting-Kraus.

    Exists exactly when :func:`free_class_extremality` reports a block b whose
    bond eigenvalue vector is a convex combination x(b) = sum_a lambda_a x(a)
    of the others (Theorem 2's converse). Picking a unit vector |b> in that
    block and unit vectors |a> in the others, the instrument

        K_0 = I - |b><b|,   K_a = sqrt(lambda_a) |a><b|                       (8)

    is complete (sum K^dag K = I - |b><b| + sum_a lambda_a |b><b| = I) and
    satisfies Eq. (2) for every bond: sum K^dag X K = X - x(b)|b><b|
    + sum_a lambda_a x(a) |b><b| = X. It moves population out of the block b
    into the blocks a, so Abar(I) != I (non-unital) and [K_a, X_j] != 0.

    Returns None when the bond spectrum is in convex position (no such
    instrument exists — Theorem 2), else dict: 'kraus', 'block', 'weights',
    'defect' (:func:`energy_signalling_defect`, at roundoff), 'commutator'
    (max_mu,j |[K_mu, X_j]|, > 0), 'unitality_defect' (|Abar(I) - I|, > 0).
    """
    H, n = _check_qubit_operator("H", H)
    region_A = _check_region(region_A, n)
    info = free_class_extremality(H, region_A, tol=tol)
    if info["in_convex_position"]:
        return None
    projectors, _ = free_projectors(H, region_A)
    _, Xs, _ = bond_operators(H, region_A)
    i_bad = int(np.argmin([bool(e) for e in info["extreme"]]))
    lam = info["weights"][i_bad]
    dA = projectors[0].shape[0]

    def unit_in(P):
        w, U = eigh(_herm(P))
        return U[:, w > 0.5][:, 0]

    v_b = unit_in(projectors[i_bad])
    others = [j for j in range(len(projectors)) if j != i_bad]
    kraus = [np.eye(dA, dtype=complex) - np.outer(v_b, v_b.conj())]
    for j, w_j in zip(others, lam):
        if w_j <= 1e-12:
            continue
        v_a = unit_in(projectors[j])
        kraus.append(np.sqrt(float(w_j)) * np.outer(v_a, v_b.conj()))
    abar_I = sum(K @ K.conj().T for K in kraus)
    return {
        "kraus": kraus, "block": i_bad, "weights": lam,
        "defect": energy_signalling_defect(H, kraus, region_A),
        "commutator": float(max(np.max(np.abs(K @ X - X @ K))
                                for K in kraus for X in Xs)),
        "unitality_defect": float(np.max(np.abs(abar_I - np.eye(dA)))),
    }


def search_free_instrument(H, region_A, solver=None, tol=1e-9):
    """Certified search (SDP) for a free instrument OUTSIDE the commutant of the bonds.

    Everything in play is linear in the average channel's Choi operator J
    (vacuum.qet.slp conventions: composite index (in, out), in-major,
    J = sum_mu k_mu k_mu^dag with k_mu[(i, o)] = (K_mu)_{oi}). Writing
    C_M(J) for the linear map that returns sum_mu K_mu^dag M K_mu, the free
    instruments are exactly

        J >= 0,   Tr_out J = I    (the instrument is complete),
        C_{X_j}(J) = X_j          for every bond  (Eq. (2)),

    and the Theorem 2 proof's block weights are M_ab = P_b C_{P_a}(J) P_b, so
    the total weight OUTSIDE the commutant,

        w_off(J) := sum_{a != b} Tr[P_b C_{P_a}(J)],                          (20)

    is linear too. Maximizing w_off over that spectrahedron is therefore an
    SDP whose optimum is ZERO if and only if every energy-non-signalling
    instrument on A is commuting-Kraus — a certificate, not a local search,
    and stronger than driving the defect down numerically. Theorem 2 predicts
    0 exactly when the bond spectrum is in convex position and a positive
    optimum otherwise.

    Returns dict: 'w_off' (the SDP optimum), 'is_commuting_kraus_only'
    (w_off <= tol), 'in_convex_position' (Theorem 2's analytic verdict; the
    two must agree), 'kraus' (Kraus operators of the maximizer, or None),
    'commutator' (their distance from the commutant), 'status', 'solver'.
    """
    cp = slp._load_cvxpy()
    H, n = _check_qubit_operator("H", H)
    region_A = _check_region(region_A, n)
    projectors, _ = free_projectors(H, region_A)
    _, Xs, _ = bond_operators(H, region_A)
    dA = 2 ** len(region_A)

    def C_of(J, M):
        """sum_mu K_mu^dag M K_mu as a linear expression in J."""
        Jt = cp.reshape(J, (dA, dA, dA, dA), order="C")      # [i, o, j, o']
        return cp.sum(cp.multiply(
            np.asarray(M, dtype=complex).conj()[None, :, None, :], Jt), axis=(1, 3))

    J = cp.Variable((dA * dA, dA * dA), hermitian=True)
    cons = [J >> 0, cp.partial_trace(J, (dA, dA), 1) == np.eye(dA)]
    for X in Xs:
        cons.append(C_of(J, X) == np.asarray(X, dtype=complex))
    w_off = sum(cp.real(cp.trace(np.asarray(Pb, dtype=complex) @ C_of(J, Pa)))
                for ia, Pa in enumerate(projectors)
                for ib, Pb in enumerate(projectors) if ia != ib)
    prob = cp.Problem(cp.Maximize(w_off), cons)
    names = [solver] if solver else [s for s in ("SCS", "CLARABEL") if s in cp.installed_solvers()]
    opts = {"SCS": {"eps_abs": 1e-10, "eps_rel": 1e-10, "max_iters": 50_000},
            "CLARABEL": {"tol_gap_abs": 1e-11, "tol_gap_rel": 1e-11}}
    value, status, used = None, None, None
    for name in names:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*nested list.*")
                warnings.filterwarnings("ignore", message=".*may be inaccurate.*")
                warnings.filterwarnings("ignore", message=".*CPP backend.*")
                prob.solve(solver=name, **opts.get(name, {}))
        except (cp.error.SolverError, cp.error.DCPError):
            continue
        if J.value is not None:
            value, status, used = float(prob.value), prob.status, name
            break
    if value is None:
        raise RuntimeError("no SDP solver produced a solution for the free-class search")
    kraus, comm = None, 0.0
    if value > tol and J.value is not None:
        kraus = slp.choi_to_kraus(_herm(J.value))
        if kraus and Xs:
            comm = float(max(np.max(np.abs(K @ X - X @ K)) for K in kraus for X in Xs))
    return {"w_off": value, "is_commuting_kraus_only": bool(value <= tol),
            "in_convex_position": free_class_extremality(H, region_A)["in_convex_position"],
            "kraus": kraus, "commutator": comm, "status": status, "solver": used}


# ---------------------------------------------------------------------------
# L3 close-out B: the symmetric (two-way) theory
# ---------------------------------------------------------------------------


def split_hamiltonian_symmetric(H, region_A):
    """The SYMMETRIC booking: H = E_A + E_B with the interaction shared, plus V.

    Operator-Schmidt the total: H = c I + a (x) I + I (x) b + V with a, b
    traceless on their factors and V traceless on BOTH. Hotta's booking (1)
    gives Bob H_B^loc = I (x) b + V + const: the whole interaction. The
    symmetric booking splits it,

        E_A^sym := a (x) I + V/2 + (c/2) I,   E_B^sym := I (x) b + V/2 + (c/2) I,   (9)

    so E_A^sym + E_B^sym = H exactly and the two parties are on the same
    footing (equivalently V is booked to a third 'bond' register that neither
    party owns; every statement below depends on V only through its bond
    operators, so the two conventions give the same free class).

    KEY POINT. An instrument on A is energy-non-signalling toward B in the
    symmetric booking iff (Abar^dag (x) id)(E_B^sym) = E_B^sym. Since
    Abar^dag(I) = I kills the I (x) b and constant parts, this is again
    Abar^dag(X_j) = X_j for the bond operators of V — IDENTICAL to Eq. (2).
    The free class of Alice is therefore unchanged by the booking; what
    changes is Bob, whose CPTP maps are free in Hotta's booking (O3) but must
    themselves be energy-non-signalling in the symmetric one. Bob's rotation
    is NOT free symmetrically, which is why the cash-out is booked as
    extraction, not as a free operation, in :func:`two_way_potential`.

    Returns (E_A_sym, E_B_sym, V), all full 2^n x 2^n operators.
    """
    H, n = _check_qubit_operator("H", H)
    region_A = _check_region(region_A, n)
    comp = _complement(region_A, n)
    dA = 2 ** len(region_A)
    dB = 2 ** (n - len(region_A))
    hA = _partial_trace_B(H, region_A, n) / dB          # c I_A + a
    hB = _partial_trace_A(H, region_A, n) / dA          # c I_B + b
    c = float(np.real(np.trace(H))) / (dA * dB)
    H_A = embed_region_operator(hA, region_A, n)
    H_B = embed_region_operator(hB, comp, n)
    V = H - H_A - H_B + c * np.eye(dA * dB, dtype=complex)
    half = 0.5 * (c * np.eye(dA * dB, dtype=complex))
    return _herm(H_A - half + V / 2.0), _herm(H_B - half + V / 2.0), _herm(V)


def qet_potential_operator(H, region_A):
    """W_op with W_->(rho) = Tr[W_op rho] for every rho — Theorem 4 (linearity).

    Prop. A's value (4) is built from sigma_a = Tr_A[(Pi_a (x) I) rho] and
    p_a = Tr sigma_a, both LINEAR in rho, and Alice's optimal measurement and
    Bob's optimal maps are state-independent. Hence W_-> is a linear
    functional,

        W_->(rho) = Tr[W_op rho],
        W_op = sum_a Pi_a (x) (H^(a) - lambda_min(H^(a)) I)
             = H_B^loc - sum_a lambda_min(H^(a)) Pi_a (x) I.                  (10)

    THEOREM 4 (exact ledger; the two-way corollary). G := W_op - H_B^loc
    = -sum_a lambda_min(H^(a)) Pi_a (x) I is supported on A ALONE and
    commutes with every Pi_a. Therefore

      (a) for EVERY instrument {L_nu} on B (not merely every channel), since
          each L_nu commutes with G,
              sum_nu p_nu W_->(rho_nu) = W_->(rho) - sum_nu p_nu w_nu,        (11a)
          an EQUALITY: the energy Bob's device receives is exactly what W_->
          loses, branch by branch, and a measurement he makes for free
          (w_nu = 0 on average) leaves the ledger untouched;
      (b) for every commuting-Kraus instrument {K_mu} on A, K_mu commutes
          with every Pi_a and sum_mu K_mu^dag H_B^loc K_mu = H_B^loc (Eq. (2)),
          so sum_mu p_mu W_->(rho_mu) = W_->(rho), again an EQUALITY.        (11b)

    SCOPE — READ (11b) LITERALLY. It is proved for COMMUTING-KRAUS
    instruments on A, which is exactly what it assumes: that K_mu commutes
    with every Pi_a. By Theorem 2 the commuting-Kraus class IS the whole free
    class precisely when the bond spectrum is in convex position, so on every
    model this module ships (single-qubit A; every region of a TFIM chain)
    (11b) covers all free instruments and, by induction over rounds, the
    TOTAL energy delivered to Bob's device by any finite-round protocol built
    from free instruments on either side and arbitrary maps on B — classical
    communication in BOTH directions, each party's action conditioned on the
    whole record — is exactly W_->(rho) - sum(final branch potentials):

        W_<->^B(rho) = W_->(rho):  TWO-WAY COMMUNICATION GIVES BOB NOTHING,   (12)
        for commuting-Kraus instruments on A, hence (Theorem 2) for the whole
        free class wherever the bond spectrum is in convex position.

    Back-communication B -> A cannot help there because Alice's free
    instruments leave the ledger invariant whatever she conditions on, and
    Bob's own instruments pay for every rise in W_-> with an equal injection.

    OFF CONVEX POSITION (12) IS FALSE, and the failure has nothing to do with
    two-way communication — ONE free instrument on A already breaks it. The
    quantity `qet_potential` computes is Prop. A's closed form, i.e. the
    COMMUTING-KRAUS value W_->^CK; where the free class is strictly larger
    (Theorem 2's converse) W_->^CK is not the true potential and not a
    monotone. On :func:`star_hamiltonian` (1.0, 0.5) with A = (0, 1), the
    witness of :func:`nonfree_energy_nonsignalling_instrument` — genuinely
    free, energy-signalling defect 9.2e-16 — raises it

        sum_a p_a W_->^CK(rho_a) = 0.225331  vs  W_->^CK(rho) = 0.036634,     (12')

    a factor 6.1509 from one-way communication alone (pinned in
    tests/test_resource_theory.py). What survives off convex position is the
    linearity of W_->^CK and part (a): Bob's own instruments still pay for
    every rise. What fails is part (b), and with it the ledger and (12).
    :func:`two_way_monotonicity_audit` samples commuting-Kraus instruments —
    the theorem's scope — and additionally evaluates the witness wherever
    convex position fails, reporting the breach as
    ``details['worst_violation_wider_class']`` instead of hiding it.

    Returns the (2^n, 2^n) Hermitian operator W_op.
    """
    H, n = _check_qubit_operator("H", H)
    region_A = _check_region(region_A, n)
    projectors, xs = free_projectors(H, region_A)
    H_B, _, Ys = bond_operators(H, region_A)
    dB = H_B.shape[0]
    out = np.zeros((H.shape[0], H.shape[0]), dtype=complex)
    for a, Pi in enumerate(projectors):
        H_eff = _herm(H_B + sum(float(x) * Y for x, Y in zip(xs[a], Ys)))
        lam = float(eigh(H_eff, eigvals_only=True)[0])
        out = out + _embed_pair(np.kron(Pi, H_eff - lam * np.eye(dB)), region_A, n)
    return _herm(out)


def _is_product(rho, region, n, tol=1e-10):
    """(is_product, rho_region, rho_rest) for the region : complement cut."""
    t = _as_region_blocks(rho, region, n)
    rho_R = _herm(np.einsum("abcb->ac", t))
    rho_O = _herm(np.einsum("abad->bd", t))
    prod = _embed_pair(np.kron(rho_R, rho_O), region, n)
    return float(np.max(np.abs(rho - prod))) <= tol, rho_R, rho_O


def _effective_hamiltonian(H, rho_rest, region, n):
    """Tr_rest[H (I (x) rho_rest)] — the mean-field Hamiltonian a party sees."""
    t = _as_region_blocks(H, region, n)          # t[q, o, q', o']
    return _herm(np.einsum("abcd,db->ac", t, rho_rest))


def optimal_local_extraction(H, rho, region, method="auto", tol=1e-9,
                             kraus_kwargs=None):
    """(extraction, post-state) for the best CPTP map on ``region``: the cash-out.

    ``method='auto'`` takes the exact closed form whenever the state is a
    product across region : complement (then Tr[H (E (x) id)(rho_reg (x)
    rho_rest)] = Tr[H_eff E(rho_reg)] + const with H_eff of
    :func:`_effective_hamiltonian`, so the optimum is the reset to H_eff's
    ground state) and otherwise falls back to the SDP ('sdp', certified) or
    the Kraus search ('kraus'). Used by :func:`two_way_potential`, where
    every cash-out that follows a rank-one free measurement is a product
    state and therefore exact and solver-free.
    """
    H, n = _check_qubit_operator("H", H)
    rho, _ = _check_qubit_operator("rho", rho)
    region = _check_region(region, n)
    prod, rho_R, rho_O = _is_product(rho, region, n)
    if prod and method in ("auto", "closed"):
        H_eff = _effective_hamiltonian(H, rho_O, region, n)
        w, U = eigh(H_eff)
        ext = state_energy(rho_R, H_eff) - float(w[0])
        gs = np.outer(U[:, 0], U[:, 0].conj())
        return max(float(ext), 0.0), _herm(_embed_pair(np.kron(gs, rho_O), region, n))
    if method in ("auto", "sdp"):
        try:
            res = slp.extractable_energy_sdp(H, rho, region, tol=tol)
            kr = slp.choi_to_kraus(res["choi"]) if res["choi"] is not None else [np.eye(2 ** len(region))]
            out = _herm(apply_local_channel(rho, kr, region))
            return float(res["extractable"]), out
        except (ImportError, ValueError, RuntimeError):
            if method == "sdp":
                raise
    res = slp.extractable_energy_kraus(H, rho, region, **(kraus_kwargs or {}))
    out = _herm(apply_local_channel(rho, res["kraus"], region))
    return float(res["extractable"]), out


def two_way_potential(H, rho, region_A, rounds=2, cash_both=True, method="auto",
                      kraus_kwargs=None, _depth=0):
    """W_<->^{(r)}: the finite-round two-way extractable energy (symmetric booking).

    A ROUND is: one party applies a free (energy-non-signalling, Eq. (2) for
    the bonds it faces) instrument on its own region; the outcome is
    broadcast; the RECEIVING party applies any CPTP map on its region, whose
    energy drop is booked as extraction. Both directions of classical
    communication are available and every action may depend on the whole
    record, so this is finite-round two-way LOCC restricted to free
    instruments (Chitambar-Gour arXiv:1806.06107 Sec. II: LOCC round number
    is itself a resource, "there exists a sequence of protocols, each
    increasing in round number, that converge to a map which cannot be
    implemented by either finite-round or unbounded-round LOCC" — so the
    finite-round value is the honest object).

        W_<->^{(0)} = 0,
        W_<->^{(r)}(rho) = max over the measuring party P of
            sum_mu p_mu [ e_Q(rho_mu) + W_<->^{(r-1)}(rho_mu') ],             (13)

    Q the other party, e_Q its optimal extraction and rho_mu' the
    post-cash-out branch. Alice's finest free measurement is optimal at each
    step because the branch value is convex in the state (it is a supremum of
    linear functionals) and refining a free instrument inside its blocks
    leaves the free class (Prop. A's argument).

    ``cash_both=False`` restricts the CASH-OUT to Bob (Hotta's booking, where
    only Bob's device is being filled) while leaving the communication
    two-way: either party may still be the one who measures and broadcasts,
    and Alice's instrument may be conditioned on Bob's outcome. By Theorem 4
    the value is then exactly W_->(rho) for EVERY number of rounds — two-way
    communication buys Bob nothing. (The Bob-measures-first branch needs
    Bob's full CPTP optimum on a state that is not yet a product; that is
    computed when his region has at most two sites — the minimal model, where
    both branches really are explored — and otherwise SKIPPED, with
    'b_first_skipped' set, because the full CPTP optimum on a larger region
    costs a Choi program per branch per round. Skipping cannot change the
    maximum: the Alice-first branch already attains the proven ceiling
    W_->(rho), which Theorem 4 shows no branch can exceed. The genuine
    two-way evidence on the chains is :func:`two_way_monotonicity_audit`,
    which runs real two-way sequences — Bob measures, Alice's free instrument
    is conditioned on his outcome, Bob cashes — and needs no CPTP optimum
    because the linear W_op evaluates the ledger directly.) With ``cash_both=True`` both devices are filled and the
    value grows without bound in the number of rounds, because each free
    measurement injects energy from the measuring party's apparatus: the
    surplus over W_-> is paid by the measurement devices, not by the state.
    'injected' and 'net' report that ledger honestly.

    Returns dict: 'W' (total energy delivered to the two devices), 'net'
    (W minus the energy the free instruments injected), 'injected', 'rounds',
    'per_round' (list of dicts: measuring party, extraction, injection),
    'measured_first' (which party's measurement the optimum starts with),
    'by_side' (the value of EACH first-mover branch that was evaluated — the
    two-sided search is what makes this two-way, so a regression test pins
    that both keys are present and that the B branch wins where it should),
    'b_first_skipped', and 'W_one_way' (Prop. A's W_-> for reference, top
    call only).
    """
    H, n = _check_qubit_operator("H", H)
    rho, _ = _check_qubit_operator("rho", rho)
    region_A = _check_region(region_A, n)
    region_B = _complement(region_A, n)
    rounds = int(rounds)
    if rounds <= 0:
        return {"W": 0.0, "net": 0.0, "injected": 0.0, "rounds": 0, "per_round": []}
    best = None
    skipped = False
    by_side = {}
    for meas in ("A", "B"):
        reg_m = region_A if meas == "A" else region_B
        # receiver cashes when both devices are being filled; otherwise Bob always does
        reg_c = region_B if (meas == "A" or not cash_both) else region_A
        if meas == "B" and not cash_both and len(reg_c) > 2:
            prod, _, _ = _is_product(rho, reg_c, n)
            if not prod:
                skipped = True
                continue
        projectors, _ = free_projectors(H, reg_m)
        e_before = state_energy(rho, H)
        tot = inj = inj_round = 0.0
        sub_rounds = []
        for p, branch in apply_instrument(rho, projectors, reg_m):
            inj_round += p * (state_energy(branch, H) - e_before)
            ext, post = optimal_local_extraction(H, branch, reg_c, method=method,
                                                 kraus_kwargs=kraus_kwargs)
            sub = two_way_potential(H, post, region_A, rounds - 1, cash_both=cash_both,
                                    method=method, kraus_kwargs=kraus_kwargs,
                                    _depth=_depth + 1)
            tot += p * (ext + sub["W"])
            inj += p * sub["injected"]
            sub_rounds.append({"p": p, "measured": meas, "extract": ext,
                               "tail": sub["per_round"]})
        inj += inj_round
        if meas == "B" and not cash_both:
            # Bob measured AND cashed: his own apparatus paid for that injection,
            # so it is not energy delivered to his device (Theorem 4 bounds the NET).
            tot -= inj_round
        by_side[meas] = float(tot)
        if best is None or tot > best["W"]:
            best = {"W": float(tot), "injected": float(inj), "net": float(tot - inj),
                    "rounds": rounds, "per_round": sub_rounds, "measured_first": meas}
    best["b_first_skipped"] = bool(skipped)
    best["by_side"] = by_side
    if _depth == 0:
        best["W_one_way"] = qet_potential(H, rho, region_A)["W"]
    return best


def two_way_monotonicity_audit(H, region_A, n_samples=200, seed=0, tol=1e-9,
                               rounds=3, state_kinds=("pure", "mixed", "ground_mix", "thermal"),
                               strict=True):
    """Theorem 4 numerically: random two-way free protocols never pay Bob more than W_->.

    Each sample runs a random FINITE-ROUND TWO-WAY protocol in Hotta's
    booking — a random sequence of (i) commuting-Kraus free instruments on A
    conditioned on the whole record so far, (ii) random instruments and
    channels on B (Bob may measure, and his outcomes may be sent back to
    Alice, who then acts on them), (iii) Bob's cash-outs — and accumulates
    the energy delivered to Bob's device NET of what his own operations
    injected. Eq. (12) demands

        sum over rounds of Bob's net take  <=  W_->(rho),                     (14)

    and Theorem 4 says more: the running total plus the current W_->(state)
    is CONSERVED, so 'worst_ledger_drift' (the largest absolute deviation
    from conservation, not merely from monotonicity) is the sharp statistic.

    SCOPE. The instruments sampled on A are commuting-Kraus — the scope of
    Theorem 4(b) — which by Theorem 2 is the whole free class exactly when
    the bond spectrum is in convex position. When it is NOT, the audit
    additionally applies the analytic witness of
    :func:`nonfree_energy_nonsignalling_instrument` (a genuinely free
    instrument outside the commutant) and records the resulting breach of
    (14) as 'worst_violation_wider_class'. That number is reported, never
    folded into ``passed``: it is a true statement about the wider class, not
    a failure of the theorem, and burying it is how the scope silently
    widened before.

    TEETH: the same protocols are run with a non-free instrument on A in the
    first round; 'teeth' records how far above W_->(rho) Bob's take then
    goes.

    Returns an :class:`AuditResult` (name 'qet_two_way') with details
    'worst_violation_of_(14)', 'worst_ledger_drift', 'teeth', 'counts',
    'free_class_is_commuting_kraus' and 'worst_violation_wider_class'.
    """
    H, n = _check_qubit_operator("H", H)
    region_A = _check_region(region_A, n)
    region_B = _complement(region_A, n)
    dA, dB = 2 ** len(region_A), 2 ** len(region_B)
    projectors, _ = free_projectors(H, region_A)
    proj_B, _ = free_projectors(H, region_B)
    W_op = qet_potential_operator(H, region_A)
    in_convex = free_class_extremality(H, region_A)["in_convex_position"]
    witness = None if in_convex else nonfree_energy_nonsignalling_instrument(H, region_A)
    worst_wider = 0.0
    rng = np.random.default_rng(int(seed))
    worst_violation = 0.0
    worst_drift = 0.0
    teeth = 0.0
    counts = {"protocols": 0, "rounds": 0, "teeth": 0}
    for i in range(int(n_samples)):
        rho0 = random_state(n, rng, state_kinds[i % len(state_kinds)], H=H)
        non_free = (i % 5 == 4)
        ensemble = [(1.0, rho0)]
        take = 0.0
        W0 = state_energy(rho0, W_op)
        counts["protocols"] += 1
        for r in range(int(rounds)):
            counts["rounds"] += 1
            move = 0 if (non_free and r == 0) else int(rng.integers(0, 4))
            new = []
            for p, rho in ensemble:
                if move == 0:                      # Alice: free instrument (or non-free tooth)
                    if non_free and r == 0:
                        K = random_instrument(dA, int(rng.integers(1, 3)), rng)
                    else:
                        K = random_free_instrument(projectors, int(rng.integers(1, 4)), rng)
                    for q, br in apply_instrument(rho, K, region_A):
                        new.append((p * q, br))
                elif move == 1:                    # Bob: free instrument on his side
                    K = random_free_instrument(proj_B, int(rng.integers(1, 3)), rng)
                    for q, br in apply_instrument(rho, K, region_B):
                        new.append((p * q, br))
                elif move == 2:                    # Bob: arbitrary instrument (may inject)
                    K = random_instrument(dB, int(rng.integers(1, 4)), rng)
                    for q, br in apply_instrument(rho, K, region_B):
                        new.append((p * q, br))
                else:                              # Bob: cash out (unitary or channel)
                    K = ([random_local_unitary(dB, rng)] if rng.random() < 0.5
                         else random_instrument(dB, int(rng.integers(1, 4)), rng))
                    out = _herm(apply_local_channel(rho, K, region_B))
                    take += p * bob_energy_out(H, rho, K, region_A)
                    new.append((p, out))
            if move in (1, 2):
                # Bob's instruments: book their energy change as (negative) take
                e_in = sum(p * state_energy(rho, H) for p, rho in ensemble)
                e_out = sum(p * state_energy(rho, H) for p, rho in new)
                take += e_in - e_out
            ensemble = [(p, r_) for p, r_ in new if p > 1e-14]
        W_end = sum(p * state_energy(rho, W_op) for p, rho in ensemble)
        if witness is not None:
            # a genuinely free instrument OUTSIDE the commutant, one-way
            wider = sum(q * state_energy(br, W_op)
                        for q, br in apply_instrument(rho0, witness["kraus"], region_A))
            worst_wider = max(worst_wider, wider - W0)
        if non_free:
            # Bob can still take W_->(final) later, so the closed ledger of a free
            # protocol is take + W_end == W0 exactly (Theorem 4); a non-free
            # instrument on A breaks it upward, and by that much.
            teeth = max(teeth, take + W_end - W0)
            counts["teeth"] += 1
        else:
            worst_violation = max(worst_violation, take - W0)
            worst_drift = max(worst_drift, abs(take + W_end - W0))
    passed = worst_violation <= float(tol) and worst_drift <= float(tol)
    result = AuditResult(
        name="qet_two_way", passed=passed,
        worst_violation=max(worst_violation, worst_drift, 0.0),
        details={"worst_violation_of_(14)": worst_violation,
                 "worst_ledger_drift": worst_drift, "teeth": teeth,
                 "free_class_is_commuting_kraus": bool(in_convex),
                 "worst_violation_wider_class": float(worst_wider),
                 "counts": counts, "tol": float(tol), "rounds": int(rounds),
                 "n_samples": int(n_samples), "seed": int(seed),
                 "region_A": region_A, "region_B": region_B},
    )
    if strict and not passed:
        raise AuditViolation(result)
    return result


# ---------------------------------------------------------------------------
# L3 close-out C: the tight bound for a single-site Bob, and its entropic form
# ---------------------------------------------------------------------------

_PAULI = (_SX, _SY, np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex))


def bob_correlation_matrix(h_R, tau, dims):
    """T_ij = Tr[(sigma_j^b (x) A_i) tau], the energy-correlation matrix of Bob's site.

    ``dims`` = (2, d_rest) with Bob's qubit b the FIRST factor of the region R.
    Decomposing h_R = I_b (x) A_0 + sum_i sigma_i^b (x) A_i (A_i = Tr_b[(sigma_i
    (x) I) h_R]/2), the 3x3 real matrix T carries everything a local operation
    on b can touch: row i is the Bloch vector of C_i = Tr_rest[(I (x) A_i) tau],
    and the row belonging to Bob's own field (A_i proportional to I) is that
    field times the Bloch vector of Bob's reduced state.
    """
    d1, d2 = int(dims[0]), int(dims[1])
    if d1 != 2:
        raise ValueError(f"single-qubit Bob expected, got d_b = {d1}")
    hb = _herm(np.asarray(h_R, dtype=complex)).reshape(d1, d2, d1, d2)
    tb = _herm(np.asarray(tau, dtype=complex)).reshape(d1, d2, d1, d2)
    T = np.zeros((3, 3))
    for i, si in enumerate(_PAULI):
        A_i = np.einsum("ca,abcd->bd", si, hb) / 2.0
        for j, sj in enumerate(_PAULI):
            T[i, j] = float(np.real(np.einsum("ca,abcd,db->", sj, tb, A_i)))
    return T


def single_site_extractable(h_R, tau, dims):
    """PROPOSITION B (exact, solver-free): Bob's unitary yield for a single-qubit Bob.

    A unitary U on b conjugates the Pauli basis by a rotation,
    U^dag sigma_i U = sum_j R_ij sigma_j with R in SO(3) (and every R in SO(3)
    arises), so with T of :func:`bob_correlation_matrix`

        Tr[h_R (U (x) I) tau (U (x) I)^dag] = Tr[A_0 tau_rest] + <R, T>_F,

    the A_0 term being untouchable. Bob's optimal yield is therefore the
    orthogonal-Procrustes value

        E_B^unit(tau) = Tr T + max_{R in SO(3)} (-<R, T>_F)
                      = Tr T + s_1 + s_2 + sign(det(-T)) s_3,                 (15)

    s_i the singular values of -T [the Kabsch construction: max_{R in SO(3)}
    <R, M> = sum s_i with the smallest flipped when det M < 0].

    This is EXACT — ratio 1 to roundoff against the chain protocol's
    conditional rotation and against Hotta's Eq. (11) — so for a single-site
    Bob it supersedes both the region-unitary entropic bound (ii) and the
    marginal-fixed SDP relaxation :func:`one_shot_bound_marginal`, which it
    also certifies. The tightness the marginal-fixed SDP was reaching (0.92)
    is the SDP's relaxation gap, not physics.

    Returns dict: 'value', 'T', 'R_opt' (the optimal rotation), 'det_flip'.
    """
    T = bob_correlation_matrix(h_R, tau, dims)
    U, s, Vh = np.linalg.svd(-T)
    flip = float(np.sign(np.linalg.det(U) * np.linalg.det(Vh)))
    D = np.diag([1.0, 1.0, flip])
    R_opt = U @ D @ Vh
    val = float(np.trace(T) + s[0] + s[1] + flip * s[2])
    return {"value": val, "T": T, "R_opt": R_opt, "det_flip": flip}


def unitary_energy_spread(h_R, dims, restarts=6, seed=0):
    """c_U = (1/2) max_{U on b} spread(h_R - (U^dag (x) I) h_R (U (x) I)).

    The sharp Hoelder constant for a traceless perturbation in Prop. C. Bounded
    analytically by 2 ||h_R^(1)||_inf, h_R^(1) = h_R - I_b (x) Tr_b h_R / d_b
    being the part of h_R that any operation on b can move (the CPTP-valid
    constant, since E^dag(I) = I leaves the b-inert part alone); the numerical
    maximum over SU(2) is returned as 'c_unitary' and the analytic one as
    'c_cptp'.
    """
    from scipy.optimize import minimize
    d1, d2 = int(dims[0]), int(dims[1])
    h = _herm(np.asarray(h_R, dtype=complex))
    A0 = np.einsum("abad->bd", h.reshape(d1, d2, d1, d2)) / d1
    h1 = _herm(h - np.kron(np.eye(d1), A0))
    c_cptp = 2.0 * float(np.max(np.abs(eigh(h1, eigvals_only=True))))
    rng = np.random.default_rng(int(seed))
    best = 0.0

    def neg(x):
        A = (x[: d1 * d1] + 1j * x[d1 * d1:]).reshape(d1, d1)
        q, r = np.linalg.qr(A)
        U = q * (np.diag(r) / np.abs(np.diag(r)))
        Uf = np.kron(U, np.eye(d2))
        w = eigh(_herm(h - Uf.conj().T @ h @ Uf), eigvals_only=True)
        return -float(w[-1] - w[0]) / 2.0

    for _ in range(int(restarts)):
        res = minimize(neg, rng.standard_normal(2 * d1 * d1), method="L-BFGS-B",
                       options={"maxiter": 300})
        best = max(best, -float(res.fun))
    return {"c_unitary": float(best), "c_cptp": float(c_cptp), "h_active": h1}


def guessing_probability_cq(probs, states):
    """p_guess = exp(-H_min(X|R)) for the cq state of Alice's outcome and Bob's region.

    Tomamichel Eq. (6.28)-(6.29) (Koenig-Renner-Schaffner): the conditional
    min-entropy of a classical register given a quantum one is minus the log
    of the optimal guessing probability over all measurements of R. For the
    BINARY, equiprobable outcome of the QET protocols here the Helstrom
    bound gives it in closed form,

        p_guess = 1/2 + (1/4) ||tau_+ - tau_-||_1,                            (16)

    and then sum_mu p_mu ||tau_mu - rho_R||_1 = 2 (p_guess - 1/2) exactly,
    with rho_R = sum_mu p_mu tau_mu the state Bob holds before the message
    (no-signalling: it is the unmeasured reduced state). Non-binary or
    non-uniform ensembles fall back to :func:`min_entropy_sdp` on the cq
    state when cvxpy is available, else to the two-outcome Helstrom value of
    the most probable pair (a lower bound, flagged).

    Returns dict: 'p_guess', 'H_min_XR', 'mean_trace_distance'
    (sum_mu p_mu ||tau_mu - rho_R||_1), 'exact'.
    """
    probs = [float(p) for p in probs]
    states = [_herm(np.asarray(t, dtype=complex)) for t in states]
    rho_R = sum(p * t for p, t in zip(probs, states))
    mtd = sum(p * float(np.sum(np.abs(eigh(_herm(t - rho_R), eigvals_only=True))))
              for p, t in zip(probs, states))
    binary_uniform = (len(states) == 2 and abs(probs[0] - 0.5) < 1e-12
                      and abs(probs[1] - 0.5) < 1e-12)
    if binary_uniform:
        diff = float(np.sum(np.abs(eigh(_herm(states[0] - states[1]), eigvals_only=True))))
        p_guess = 0.5 + 0.25 * diff
        exact = True
    else:
        dR = states[0].shape[0]
        cq = np.zeros((len(states) * dR, len(states) * dR), dtype=complex)
        for m, (p, t) in enumerate(zip(probs, states)):
            cq[m * dR:(m + 1) * dR, m * dR:(m + 1) * dR] = p * t
        try:
            p_guess = float(np.exp(-min_entropy_sdp(cq, (len(states), dR))["H_min"]))
            exact = True
        except (ImportError, RuntimeError):
            p_guess = max(probs) + 0.5 * mtd
            exact = False
    return {"p_guess": float(p_guess), "H_min_XR": float(-np.log(max(p_guess, 1e-300))),
            "mean_trace_distance": float(mtd), "exact": bool(exact)}


def one_shot_bound_entropic(h_R, branches, dims, restarts=6, seed=0):
    """PROPOSITION C: an entropic bound that vanishes with Bob's information.

    Let rho_R = sum_mu p_mu tau_mu be Bob's region BEFORE Alice's message —
    by no-signalling it is the unmeasured reduced state, and on a vacuum it
    is PASSIVE under every operation on Bob's site (strong local passivity,
    vacuum.qet.slp; certified per row and reported as 'passive'). Then for
    every CPTP map E on b, writing Y_E := h_R - (E^dag (x) id)(h_R),

        Tr[Y_E rho_R] <= 0                  (passivity)
        E_B = sum_mu p_mu max_E Tr[Y_E tau_mu]
            <= sum_mu p_mu max_E Tr[Y_E (tau_mu - rho_R)]
            <= c sum_mu p_mu ||tau_mu - rho_R||_1                             (17)
            =  2 c (exp(-H_min(X|R)) - 1/2)   (uniform binary outcome, Eq. (16))

    with c = c_U of :func:`unitary_energy_spread` for a unitary Bob (c_cptp
    for an arbitrary map), using |Tr[Y delta]| <= (spread(Y)/2) ||delta||_1
    for traceless delta. So BOB CAN EXTRACT ONLY WHAT HIS REGION KNOWS ABOUT
    ALICE'S OUTCOME: the bound is exactly zero when the bit is unguessable
    (p_guess = 1/2), which is the passivity/no-signalling statement, and it
    is the first entropic bound here that vanishes with the correlation
    rather than with the region's energy scale.

    Also returned, for the honest tightness comparison:
      'bound_field' — the purely entropic part, (2 exp(-H_min(b|X)) - 1) times
        Bob's own local field, where H_min(b|X) aggregates over Alice's
        outcome by Tomamichel Eq. (6.25); it is the exact yield when Bob's
        site decouples from the rest of R (the row of T that survives);
      'bound_rows' — the exact value (15) relaxed by the triangle inequality
        ||T||_* <= sum_i ||T_i||;
      'exact' — Prop. B's exact value (15).

    MEASURED (see papers/.../data/s8_entropic.npz): on the chains all three
    entropic relaxations are LOOSE by about two orders of magnitude, because
    Alice's outcome moves Bob's conditional state by an O(1) trace distance
    in a nearly energy-neutral direction; the tight content is a cancellation
    between the field row and the correlation rows of T, which no norm-based
    relaxation retains. Prop. B keeps it exactly.

    Returns dict: 'bound_guess', 'bound_guess_cptp', 'bound_field',
    'bound_rows', 'exact', 'p_guess', 'H_min_XR', 'H_min_bX',
    'mean_trace_distance', 'c_unitary', 'c_cptp'.
    """
    d1, d2 = int(dims[0]), int(dims[1])
    branches = [(float(p), _herm(np.asarray(t, dtype=complex))) for p, t in branches]
    h_R = _herm(np.asarray(h_R, dtype=complex))
    cs = unitary_energy_spread(h_R, dims, restarts=restarts, seed=seed)
    g = guessing_probability_cq([p for p, _ in branches], [t for _, t in branches])
    exact = sum(p * single_site_extractable(h_R, t, dims)["value"] for p, t in branches)
    rows = 0.0
    for p, t in branches:
        T = bob_correlation_matrix(h_R, t, dims)
        rows += p * float(sum(np.linalg.norm(T[i]) - T[i, i] for i in range(3)))
    # Bob's own field: the A_i proportional to the identity on the rest
    hb = h_R.reshape(d1, d2, d1, d2)
    field = np.zeros(3)
    for i, si in enumerate(_PAULI):
        A_i = np.einsum("ca,abcd->bd", si, hb) / 2.0
        field[i] = float(np.real(np.trace(A_i))) / d2
    f_norm = float(np.linalg.norm(field))
    taus_b = [np.einsum("abcb->ac", t.reshape(d1, d2, d1, d2)) for _, t in branches]
    H_min_bX = conditional_min_entropy_cq([p for p, _ in branches], taus_b)
    bound_field = f_norm * (2.0 * float(np.exp(-H_min_bX)) - 1.0)
    return {
        "bound_guess": float(cs["c_unitary"] * g["mean_trace_distance"]),
        "bound_guess_cptp": float(cs["c_cptp"] * g["mean_trace_distance"]),
        "bound_field": float(bound_field), "bound_rows": float(rows),
        "exact": float(exact), "p_guess": g["p_guess"], "H_min_XR": g["H_min_XR"],
        "H_min_bX": float(H_min_bX), "mean_trace_distance": g["mean_trace_distance"],
        "c_unitary": cs["c_unitary"], "c_cptp": cs["c_cptp"],
    }


def daemonic_gain(h_R, branches, dims=None, h_ref=None, rho_ref=None):
    """QET's E_B IS a daemonic gain (Francica et al., arXiv:1608.00124).

    Their Eq. (3): W_{Pi} = Tr[rho_S H_S] - sum_a p_a Tr[U_a rho_{S|a} U_a^dag
    H_S], the work extracted from S by unitaries conditioned on the outcome a
    of a measurement of the ancilla; their Eq. (5): the daemonic gain
    delta_W = max_{Pi} W_{Pi} - W, W the plain (unconditioned) ergotropy of
    rho_S = Tr_A rho_SA. Reading S = Bob's region, A = Alice's, and
    restricting the daemon's measurements to the FREE (energy-non-signalling)
    ones, Bob's one-way yield is exactly W_{Pi}, and since the vacuum is
    strongly locally passive its unconditioned ergotropy is W = 0, so

        E_B = W_{Pi} = delta_W:  QET's teleported energy is a daemonic gain,  (18)

    with the resource theory's free class as the constraint on the daemon.
    ``h_R`` may be one operator or one per branch (Bob's effective Hamiltonian
    depends on the bond eigenvalue when Bob is the whole complement); the
    unconditioned reference then needs its own ``h_ref`` — the MEAN-FIELD
    operator H_B + sum_j <X_j>_rho Y_j Bob faces with no message — and
    optionally its own ``rho_ref`` (default: the branch average). Passing the
    wrong reference is the one way to get a spurious nonzero W.

    Returns dict: 'W_conditional' (their W_{Pi}, = E_B), 'W_unconditional'
    (their W, = 0 on a passive reference), 'daemonic_gain' (their delta_W),
    'ergotropy_of_average'. With ``dims`` given, the conditional yields are
    Prop. B's exact values; otherwise the region's full unitary ergotropy is
    used (dense eigendecomposition, Allahverdyan et al. via their Eq. (2)).
    """
    branches = [(float(p), _herm(np.asarray(t, dtype=complex))) for p, t in branches]
    hs = list(h_R) if isinstance(h_R, (list, tuple)) else [h_R] * len(branches)
    hs = [_herm(np.asarray(h, dtype=complex)) for h in hs]
    rho_R = sum(p * t for p, t in branches) if rho_ref is None else _herm(np.asarray(rho_ref, complex))
    h_R = hs[0] if h_ref is None else _herm(np.asarray(h_ref, dtype=complex))

    def ergotropy(tau, h):
        if dims is not None:
            return single_site_extractable(h, tau, dims)["value"]
        lam = np.sort(np.real(eigh(_herm(tau), eigvals_only=True)))[::-1]   # descending
        eps = np.sort(np.real(eigh(h, eigvals_only=True)))                  # ascending
        return float(state_energy(tau, h) - float(np.dot(lam, eps)))

    w_cond = sum(p * ergotropy(t, h) for (p, t), h in zip(branches, hs))
    w_uncond = ergotropy(rho_R, h_R)
    return {"W_conditional": float(w_cond), "W_unconditional": float(w_uncond),
            "daemonic_gain": float(w_cond - w_uncond),
            "ergotropy_of_average": float(w_uncond)}


def _smooth_min_entropy_subnorm(r, U, tr_rho, eps, return_state):
    """H_min^eps over Tomamichel's sub-normalized ball (Def. 6.8), dense.

    For a normalized input this is Prop. E in closed form: the cap is the
    normalized water-filling value whenever that is >= 1/d, and otherwise
    (1 - eps^2)/(sum_i sqrt r_i)^2. A genuinely sub-normalized input keeps the
    second term of the generalized fidelity, and the total weight is then
    optimized numerically.
    """
    d = r.size
    target = 1.0 - eps * eps
    if eps == 0.0:
        cap = float(np.max(r))
        value = float(-np.log(max(cap, 1e-300)))
        return (value, (U * r) @ U.conj().T) if return_state else value
    if abs(tr_rho - 1.0) <= 1e-12:
        rn = r / tr_rho
        cap_norm = float(np.exp(-smooth_min_entropy((U * rn) @ U.conj().T, eps)))
        cap_flat = target / float(np.sum(np.sqrt(rn))) ** 2
        cap = min(cap_norm, cap_flat) if cap_flat < 1.0 / d else cap_norm
        s_opt = (np.full(d, cap) if cap < 1.0 / d
                 else _fidelity_capped(rn, cap)[1])
    else:                                   # sub-normalized input: optimize the weight
        best_cap, s_opt = float(np.max(r)), r
        for t in np.linspace(1e-6, 1.0, 201):
            lo, hi = 0.0, float(np.max(r)) + 1.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                th_lo, th_hi = 0.0, 1e12
                for _ in range(80):
                    th = 0.5 * (th_lo + th_hi)
                    if float(np.sum(np.minimum(mid, th * r))) < t:
                        th_lo = th
                    else:
                        th_hi = th
                ss = np.minimum(mid, th_hi * r)
                f = (float(np.sum(np.sqrt(r * ss)))
                     + np.sqrt(max(0.0, (1.0 - tr_rho) * (1.0 - float(np.sum(ss))))))
                if f * f >= target:
                    hi = mid
                else:
                    lo = mid
            if hi < best_cap:
                th_lo, th_hi = 0.0, 1e12
                for _ in range(80):
                    th = 0.5 * (th_lo + th_hi)
                    if float(np.sum(np.minimum(hi, th * r))) < t:
                        th_lo = th
                    else:
                        th_hi = th
                best_cap, s_opt = hi, np.minimum(hi, th_hi * r)
        cap = best_cap
    value = float(-np.log(max(cap, 1e-300)))
    if return_state:
        return value, (U * s_opt) @ U.conj().T
    return value


# ---------------------------------------------------------------------------
# L3 close-out D: chains beyond the dense wall (L = 12), sparse Prop. A
# ---------------------------------------------------------------------------


def chain_potential_sparse(L, g, A, J=1.0, ground=None):
    """W_->(|g>) for the TFIM chain by Prop. A, without ever building a dense 2^L.

    Alice sits on the single site A, Bob is the whole complement. The only
    A-B coupling is -J sigma_x^A (sigma_x^{A-1} + sigma_x^{A+1}), so the bond
    operator on A is sigma_x^A, Alice's finest free measurement is the
    protocol's own sigma_x projectors, and inside the block mu = +-1 Bob's
    effective Hamiltonian is the (L-1)-site sparse operator

        H^(mu) = -J sum_{bonds not touching A} sx sx - g sum_{i != A} sz_i
                 - J mu (sx_{A-1} + sx_{A+1}),                                (19)

    whose ground energy comes from sparse eigsh. Prop. A then gives
    W_-> = sum_mu p_mu (<H_B^loc>_mu - lambda_min(H^(mu))), every piece of
    which is a sparse matrix-vector product on 2^L or 2^(L-1) amplitudes.
    Agrees with the dense :func:`qet_potential` to roundoff for L <= 10
    (tested) and reaches L = 12 in seconds.

    Returns dict: 'W', 'branches' (p, mean energy, lambda_min per outcome),
    'L', 'g', 'J', 'A'.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    from vacuum.qet import chains
    L, A = int(L), int(A)
    g, J = float(g), float(J)
    if ground is None:
        ground = chains.tfim_ground_state(L, g, J)
    _, psi = ground
    h_ops = chains.tfim_local_energy_ops(L, g, J)
    H_full = chains.tfim_hamiltonian(L, g, J)
    H_A = -g * chains._kron_chain(L, {A: chains._SZ})          # Alice's own term
    H_loc = (H_full - H_A).tocsr()                             # Hotta's H_B + V
    sites_B = [q for q in range(L) if q != A]
    nB = L - 1
    idx = {q: i for i, q in enumerate(sites_B)}
    H_mu = {}
    for mu in (+1, -1):
        Hm = sp.csr_matrix((2**nB, 2**nB))
        for i in range(L - 1):
            if i == A or i + 1 == A:
                continue
            Hm = Hm - J * chains._kron_chain(nB, {idx[i]: chains._SX, idx[i + 1]: chains._SX})
        for q in sites_B:
            Hm = Hm - g * chains._kron_chain(nB, {idx[q]: chains._SZ})
        for nb in (A - 1, A + 1):
            if 0 <= nb < L:
                Hm = Hm - J * mu * chains._kron_chain(nB, {idx[nb]: chains._SX})
        H_mu[mu] = Hm.tocsr()
    W = 0.0
    rows = []
    for mu, br in chains.measure_site_x(psi, L, A).items():
        p, v = br["p"], br["state"]
        e = float(np.real(np.vdot(v, H_loc @ v)))
        v0 = np.full(2**nB, 1.0 / np.sqrt(2**nB))
        lam = float(eigsh(H_mu[mu], k=1, which="SA", v0=v0)[0][0])
        W += p * (e - lam)
        rows.append({"mu": int(mu), "p": float(p), "energy": e, "lambda_min": lam})
    return {"W": float(W), "branches": rows, "L": L, "g": g, "J": J, "A": A}


def chain_region_ledger(L, g, A, B_list, J=1.0, eps_grid=(0.0, 0.01, 0.1),
                        marginal=True, entropic=True, W_full=True):
    """The whole L3 ledger for a chain from REGION-SIZED objects only (L = 12 reachable).

    Every quantity Bob's single site can be judged by lives on R = {B-1, B,
    B+1}: because h_R = local_hamiltonian(B) contains every term of H that
    touches B, for any CPTP map E on B

        Tr[H (E (x) id) rho] - Tr[H rho] = Tr[h_R (E (x) id) rho_R] - Tr[h_R rho_R],

    so Bob's single-site CPTP potential, his exact unitary optimum (Prop. B),
    the region one-shot bound, the marginal-fixed SDP and the entropic bounds
    are all computable from the 8-dimensional rho_R — no dense 2^L anywhere.
    Verified against the dense :func:`chain_resource_ledger` for L <= 8.
    Combined with :func:`chain_potential_sparse` for the whole-complement
    ceiling, this takes the candidate from L <= 10 to L = 12.

    Returns dict: 'L', 'g', 'J', 'A', 'E_A', 'W_arrow_full', 'rows' (per B:
    E_B, W_site, exact Prop. B value, bounds, ratios, negativity, H_min's).
    """
    from vacuum.qet import chains
    L, A = int(L), int(A)
    g, J = float(g), float(J)
    ground = chains.tfim_ground_state(L, g, J)
    E0, psi = ground
    h_ops = chains.tfim_local_energy_ops(L, g, J)
    W_arrow = (chain_potential_sparse(L, g, A, J, ground=ground)["W"] if W_full else np.nan)
    rows = []
    E_A = None
    for B in B_list:
        B = int(B)
        led = chains.run_chain_qet(L, g, A, B, J=J, ground=ground, h_ops=h_ops,
                                   theta_method="closed")
        E_A = led["E_A"]
        Rr = tuple(q for q in (B - 1, B, B + 1) if 0 <= q < L)
        order = [B] + [q for q in Rr if q != B]            # Bob's site first
        k = len(order)
        h_loc = np.asarray(chains.local_hamiltonian(h_ops, B).toarray(), dtype=complex)
        h_R = _partial_trace_B(h_loc, tuple(sorted(order)), L) / 2 ** (L - k)
        perm = [sorted(order).index(q) for q in order]
        h_R = _herm(np.transpose(h_R.reshape((2,) * (2 * k)),
                                 perm + [k + i for i in perm]).reshape(2**k, 2**k))
        dims = (2, 2 ** (k - 1))
        branches = [(o["p"], chains.reduced_density(o["state_meas"].astype(complex), L, order))
                    for o in led["outcomes"].values()]
        rho_R = sum(p * t for p, t in branches)
        osb = one_shot_bound(h_R, branches, eps_grid=eps_grid)
        exact = sum(p * single_site_extractable(h_R, t, dims)["value"] for p, t in branches)
        w_site = sum(p * slp.extractable_energy_kraus(h_R, t, (0,), restarts=3)["extractable"]
                     for p, t in branches)
        cert = slp.slp_certificate(h_R, rho_R, (0,))
        ent = one_shot_bound_entropic(h_R, branches, dims) if entropic else {}
        bm = np.nan
        if marginal:
            try:
                bm = one_shot_bound_marginal(h_R, branches, dims, (B,))["bound"]
            except (ImportError, RuntimeError):
                bm = np.nan
        rho_AB = chains.reduced_density(psi.astype(complex), L, [A, B])
        rows.append({
            "B": B, "d": abs(B - A), "E_B": led["E_B"], "W_site": float(w_site),
            "exact_unitary": float(exact), "R": Rr,
            "bound_exact": osb["bound_exact"], "bound_best": osb["bound"],
            "eps_best": osb["eps_best"], "bound_cq": osb["bound_cq"],
            "bound_marginal": float(bm), "ceiling_R": osb["ceiling"],
            "bound_guess": ent.get("bound_guess", np.nan),
            "bound_field": ent.get("bound_field", np.nan),
            "bound_rows": ent.get("bound_rows", np.nan),
            "p_guess": ent.get("p_guess", np.nan), "H_min_XR": ent.get("H_min_XR", np.nan),
            "H_min_bX": ent.get("H_min_bX", np.nan),
            "unconditionally_passive": bool(cert["slp"]),
            "negativity_AB": negativity(rho_AB, (0,)),
            "ratio_exact": led["E_B"] / osb["bound_exact"],
            "ratio_marginal": led["E_B"] / bm if np.isfinite(bm) else np.nan,
            "ratio_closed_form": led["E_B"] / float(exact) if exact > 0 else np.nan,
            "ratio_guess": led["E_B"] / ent["bound_guess"] if entropic else np.nan,
            "ratio_full": led["E_B"] / W_arrow if W_full else np.nan,
        })
    return {"L": L, "g": g, "J": J, "A": A, "E_A": E_A, "W_arrow_full": W_arrow,
            "rows": rows}
