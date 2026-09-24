"""Correlation harvesting beyond negativity: MI, discord, coherence (Layer 2 pivot).

PLAN.md, Layer 2 pivot trigger: when harvested negativity under honest noise
models sits below detectability for every realistic parameter, "the layer
pivots to mutual-information and coherence harvesting (measurable
classical-quantum correlations rather than distillable entanglement)".  The
``circuit-qed-noise-thresholds`` candidate fired that trigger for E_N and
carried I(A:B) as a single extra column.  This module supplies the
correlation measures the pivot needs for BOTH detector models of the
harvesting engine, and a resolvability model that turns the tomography floor
already derived for E_N into floors for the other measures.

Everything is a pure function over arrays (no hidden state, no input
mutation), in docs/API.md conventions: hbar = 1, V_vac = I/2, block
quadrature ordering R = (x_1..x_N, p_1..p_N), entropies in NATS.

The measures
------------
Two-detector Gaussian block (oscillator detectors,
``vacuum.detectors.nonperturbative``; the block from ``detector_block``):

- **mutual information** I(A:B) = S(A) + S(B) - S(AB), a thin wrapper on
  ``vacuum.core.mutual_information``;
- **Gaussian quantum discord** D^{<-}(A|B) — the discord with the conditional
  entropy restricted to (generalised) Gaussian measurements on B — in the
  closed form of G. Adesso and A. Datta, Phys. Rev. Lett. 105, 030501 (2010),
  arXiv:1003.4979, Eqs. (3)-(4): with the local symplectic invariants
  A = det alpha, B = det beta, C = det gamma, D = det sigma of the covariance
  sigma = [[alpha, gamma], [gamma^T, beta]] in THEIR normalisation
  (vacuum = identity, i.e. sigma = 2V here) and

      f(x) = ((x+1)/2) ln((x+1)/2) - ((x-1)/2) ln((x-1)/2),

      D^{<-} = f(sqrt B) - f(nu_-) - f(nu_+) + f(sqrt E^min),        [Eq. (3)]

  where nu_± are the symplectic eigenvalues of sigma and E^min is the minimum
  over Gaussian measurements of det(eps), eps = alpha - gamma (beta +
  sigma_0)^{-1} gamma^T being the post-measurement covariance of A [Eq. (4)]:

      E^min = [2C^2 + (B-1)(D-A) + 2|C| sqrt(C^2 + (B-1)(D-A))] / (B-1)^2
                                  if (D - AB)^2 <= (1+B) C^2 (A+D),
            = [AB - C^2 + D - sqrt(C^4 + (D-AB)^2 - 2C^2(AB+D))] / (2B)
                                  otherwise,

  the first branch attained by projecting B onto a squeezed state of finite
  unbalanced variances, the second by homodyne detection (infinite
  squeezing).  The same closed form was obtained independently by P. Giorda
  and M. G. F. Paris, Phys. Rev. Lett. 105, 020503 (2010), arXiv:1003.3207,
  over Gaussian POVMs (their Eq. (2) for squeezed thermal states, where the
  optimum is the heterodyne/coherent-state POVM).  Both directions are
  provided (``measured='B'`` / ``'A'``), and :func:`brute_force_gaussian_discord`
  re-derives the minimum by direct optimisation over pure single-mode
  Gaussian seeds (squeezing r, angle phi) for the test suite.

  **The Gaussian-measurement caveat, carried with every reported number.**
  S. Pirandola, G. Spedalieri, S. L. Braunstein, N. J. Cerf, S. Lloyd,
  Phys. Rev. Lett. 113, 140405 (2014), arXiv:1309.2215, prove that Gaussian
  measurements are optimal for the discord of a large family of two-mode
  Gaussian states (all two-mode squeezed thermal states among them).  A
  harvested two-detector state is a general two-mode Gaussian state, not
  necessarily in that family, so the Gaussian discord reported here is an
  UPPER bound on the unrestricted discord; it is exact whenever the state is
  in the Pirandola et al. family.
- **classical correlation** J^{<-} = I - D^{<-} = f(sqrt A) - f(sqrt E^min):
  the one-way classical correlation (Henderson-Vedral / Ollivier-Zurek) in
  the same Gaussian-measurement restriction, so I = J + D holds by
  construction and is pinned against the independent ``vacuum.core`` route.
- **coherence**: the relative entropy of coherence C_r(rho) = S(Delta rho) -
  S(rho) of T. Baumgratz, M. Cramer, M. B. Plenio, Phys. Rev. Lett. 113,
  140401 (2014), arXiv:1311.0275, in the product of the LOCAL ENERGY (Fock)
  eigenbases of the two detector oscillators H_d = (1/2)(p_d^2 + Omega_d^2
  x_d^2) — Delta is the dephasing map onto that basis.  S(rho) is exact from
  the symplectic spectrum; S(Delta rho) is the Shannon entropy of the joint
  photon-number distribution, which has no Gaussian closed form and is
  computed from the truncated Fock representation of the state
  (:func:`fock_density_matrix`: rho = exp(-(1/2) R^T G R)/Z with G the
  Gaussian modular Hamiltonian of ``vacuum.core.entanglement_hamiltonian``,
  in the detector's dimensionless quadratures) under a cutoff-doubling
  convergence check.  The photon-number generating function
  Tr[rho z_A^{N_A} z_B^{N_B}] = prod_k (1-z_k)^{-1} det(V + nu_z)^{-1/2}
  (Gaussian overlap formula) is a closed form and serves as the independent
  route to the diagonal in :func:`photon_number_distribution_gf`.
  Alongside C_r, the **correlated coherence** C_cc = C_r(AB) - C_r(A) -
  C_r(B) of K. C. Tan, H. Kwon, C.-Y. Park, H. Jeong, Phys. Rev. A 94,
  022329 (2016), arXiv:1603.01958, is reported: it vanishes on product states
  (local squeezing is coherence but not correlation) and is the coherence
  measure the pivot's "product states give zero" requirement refers to.

The closest published anchor for these measures on THIS detector model is
E. G. Brown, "Thermal amplification of field-correlation harvesting",
Phys. Rev. A 88, 062336 (2013), arXiv:1309.1425: a pair of oscillator
detectors harvesting from a hot scalar field in a periodic cavity, whose
entanglement decays rapidly to zero with temperature while the Gaussian
quantum discord and the mutual information can *grow* by orders of magnitude
("thermal amplification").  The map built on this module reproduces that
ordering on the circuit-QED digital twin (thermal MI and discord rising
across the E_N sudden-death wall) and asks the question Brown's setting did
not: which of them a real tomography could see.

Qubit pair (perturbative UDW detectors, ``vacuum.detectors.udw``): the same
five quantities from the 4x4 density matrix — MI from the eigenvalues,
discord by numerical minimisation over projective (rank-1) measurements of
the measured qubit (dense Bloch-sphere grid, then Nelder-Mead refinement),
coherence in the product energy basis {|g>, |e>}^2.  Projective measurements
are the standard two-qubit choice (S. Luo, Phys. Rev. A 77, 042303 (2008),
Bell-diagonal closed form used as a test anchor; M. Ali, A. R. P. Rau,
G. Alber, Phys. Rev. A 81, 042105 (2010), arXiv:1002.3429, X states); general
POVMs can lower two-qubit discord slightly, so this too is an upper bound on
the unrestricted discord.  The second-order PKMM state (``UDWPairState.rho``)
is positive only to O(lambda^2): its {|gg>, |ee>} block has a negative
eigenvalue of order |M|^2, beyond the order of the expansion; eigenvalues
are clipped at zero and the clipped weight is reported with every number.

Communication split of the mutual information
--------------------------------------------
A. Pozas-Kerstjens and E. Martin-Martinez, Phys. Rev. D 92, 064042 (2015),
arXiv:1506.03081, Sec. IV ("Harvesting of mutual information"), Eqs. (76)-(77):

    I(rho_AB) = L_+ ln L_+ + L_- ln L_- - L_AA ln L_AA - L_BB ln L_BB + O(lambda^4),
    L_± = (1/2)[L_AA + L_BB ± sqrt((L_AA - L_BB)^2 + 4 |L_AB|^2)],

so to leading order the MI depends on the cross term L_AB (= ``C`` of
``vacuum.detectors.udw``) and NOT on the pair term M, and — as PKMM note —
"harvesting of mutual information is possible for the whole spacetime, even
for large temporal and spatial separations ... long after entanglement
harvesting is no longer possible".  L_AB is linear in the Wightman kernel,
so the Tjoa-Martin-Martinez split (E. Tjoa, E. Martin-Martinez, Phys. Rev. D
104, 125005 (2021); ``vacuum.detectors.communication``) passes through it
exactly as it does through M: L_AB = L_AB^+ (anti-commutator, state-dependent,
harvested) + L_AB^- (commutator, state-independent, field-mediated
signalling, zero at spacelike separation).  :func:`mutual_information_split`
runs the cross-term quadrature with the commutator kernel and reports the
leading-order MI with and without the signalling part.  This is this
module's construction, following TMM21's logic for M; it is not a published
MI estimator, and it is perturbative — the nonperturbative rows of a sweep
get their causal control from the spacelike-window rows instead.

Resolvability model at the tomography floor
-------------------------------------------
The E_N floor of ``experiments/circuit_qed_harvesting/teixido-bonfill_2026_
table1.json`` (``negativity_resolution_nats``, 1e-3 nats) is the concurrence
floor of the Waterloo group's own MLE-tomography simulation (S. Ren, MSc
thesis, Waterloo 2022, Sec. 4.3 / Fig. 4.5; ``ren_2022_sec4-3.json``).  The
other measures respond to reconstruction noise very differently from a
concurrence (MI is quadratic in a small coherence, and non-negative measures
pick up a positive bias from noise on an uncorrelated state), so a single
nats value cannot be transferred.  :func:`monte_carlo_floors` therefore
re-derives, per state, what the SAME tomography resolves, under these
stated assumptions:

  (A1) shot statistics and readout are Ren 2022's: N shots per prerotation
       setting and a single-shot readout fidelity F, both read from the
       parameter file at run time (:func:`tomography_precision_from_experiment`);
  (A2) each Pauli expectation <sigma_i x sigma_j> is an independent Gaussian
       estimate with standard error 1/(v sqrt N) for single-qubit and
       1/(v^2 sqrt N) for two-qubit Paulis, v = 2F - 1 the readout visibility
       (the binomial standard error of an N-shot average, inflated by the
       visibility loss of an F-fidelity discriminator); correlations between
       the 15 estimates are ignored;
  (A3) the reconstruction is the nearest physical state in 2-norm — the
       fast maximum-likelihood estimate for additive Gaussian noise of
       J. A. Smolin, J. M. Gambetta, G. Smith, Phys. Rev. Lett. 108, 070502
       (2012), arXiv:1106.5458 — in place of Ren's iterative MLE;
  (A4) a measure Q is *resolvable* on a state rho when Q(rho) exceeds the
       q-quantile (default 0.95) of Q over reconstructions of the
       uncorrelated reference rho_A x rho_B (the false-positive floor); the
       reconstruction distribution of Q(rho_hat) itself is reported too;
  (A5) for oscillator detectors the two-qubit state entering the model is
       the {0, 1}^2 Fock block of the Gaussian state, renormalised (weight
       recorded), i.e. the qubits the oscillators stand in for.

Two calibration checks are computed from the same model (and stored by the
map): the spurious concurrence on Ren's separable input |eg><eg| against the
plot-read axis extent of his Fig. 4.5(b) (``qst_spurious_concurrence_
separable_max``), and the reconstructed-concurrence spread on his printed
optimal-point X state against the transcribed range [0.001, 0.01] with
"nearly half" zeros.  A transparent analytic companion,
:func:`correlation_floors`, evaluates the measures on the X state that
carries a single |eg>-|ge> coherence at the tomography precision
delta = E_floor/2 (the concurrence-to-coherence relation C = 2|rho_23| of
Ren's near-pure X-state geometry) at the row's own populations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from vacuum.core import (
    entanglement_hamiltonian,
    entropy,
    entropy_from_nu,
    mutual_information as core_mutual_information,
    reduce as reduce_cov,
    symplectic_eigenvalues,
)
from vacuum.experiments_io import load_experiment

from .communication import commutator_kernel
from .udw import _pair_params, cross_term, pair_kernels

__all__ = [
    # Gaussian two-mode block
    "TwoModeInvariants",
    "two_mode_invariants",
    "f_entropy",
    "mutual_information",
    "minimum_conditional_determinant",
    "gaussian_discord",
    "classical_correlation",
    "GaussianCorrelations",
    "gaussian_correlations",
    "measurement_covariance",
    "conditional_determinant",
    "brute_force_gaussian_discord",
    "random_two_mode_state",
    # Fock representation and coherence
    "fock_ladder",
    "fock_quadratures",
    "rescale_to_dimensionless",
    "fock_density_matrix",
    "photon_number_distribution",
    "photon_number_distribution_gf",
    "shannon_entropy",
    "CoherenceResult",
    "gaussian_coherence",
    "qubit_proxy_from_fock",
    # qubit pair
    "PAULI",
    "qubit_pair_from_udw",
    "qubit_entropy",
    "qubit_marginals",
    "qubit_mutual_information",
    "qubit_discord",
    "qubit_coherence",
    "qubit_correlations",
    "concurrence",
    "qubit_log_negativity",
    "pkmm_mutual_information",
    "mutual_information_split",
    # resolvability model
    "TomographyPrecision",
    "tomography_precision",
    "tomography_precision_from_experiment",
    "coherence_precision_from_negativity_floor",
    "pauli_expectations",
    "state_from_pauli",
    "project_to_physical",
    "noisy_reconstructions",
    "batch_qubit_measures",
    "monte_carlo_floors",
    "floor_state",
    "correlation_floors",
]

LN2 = math.log(2.0)
_SQRT2 = math.sqrt(2.0)

# ==========================================================================
# Two-mode Gaussian block: invariants, MI, discord, classical correlation
# ==========================================================================


@dataclass(frozen=True)
class TwoModeInvariants:
    """Local symplectic invariants of a two-mode covariance (Adesso-Datta).

    In the sigma = 2V normalisation (vacuum = identity): A = det alpha,
    B = det beta, C = det gamma, D = det sigma, and the symplectic
    eigenvalues nu_- <= nu_+ of sigma (= 2 x those of V).
    """

    A: float
    B: float
    C: float
    D: float
    nu_minus: float
    nu_plus: float

    @property
    def Delta(self) -> float:
        """Seralian invariant Delta = A + B + 2C (= nu_-^2 + nu_+^2)."""
        return self.A + self.B + 2.0 * self.C


def _two_mode_sigma(V, modes_A, modes_B):
    """(sigma, alpha, beta, gamma) of the A|B block in Adesso-Datta normalisation."""
    modes_A, modes_B = list(modes_A), list(modes_B)
    if len(modes_A) != 1 or len(modes_B) != 1:
        raise ValueError(
            "the two-mode closed forms need exactly one mode per side, got "
            f"{modes_A} | {modes_B}"
        )
    if modes_A[0] == modes_B[0]:
        raise ValueError("modes_A and modes_B must be disjoint")
    VAB = reduce_cov(V, modes_A + modes_B)  # ordering (x_A, x_B, p_A, p_B)
    sigma = 2.0 * VAB
    iA, iB = [0, 2], [1, 3]
    return sigma, sigma[np.ix_(iA, iA)], sigma[np.ix_(iB, iB)], sigma[np.ix_(iA, iB)]


def two_mode_invariants(V, modes_A=(0,), modes_B=(1,)) -> TwoModeInvariants:
    """A, B, C, D and nu_± of the A|B two-mode block (sigma = 2V)."""
    sigma, alpha, beta, gamma = _two_mode_sigma(V, modes_A, modes_B)
    nu = 2.0 * symplectic_eigenvalues(0.5 * sigma)
    return TwoModeInvariants(
        A=float(np.linalg.det(alpha)),
        B=float(np.linalg.det(beta)),
        C=float(np.linalg.det(gamma)),
        D=float(np.linalg.det(sigma)),
        nu_minus=float(np.min(nu)),
        nu_plus=float(np.max(nu)),
    )


def f_entropy(x) -> float:
    """f(x) of Adesso-Datta: the entropy of a mode with sigma-eigenvalue x.

    f(x) = ((x+1)/2) ln((x+1)/2) - ((x-1)/2) ln((x-1)/2), i.e. the entropy
    of a thermal mode with symplectic eigenvalue nu = x/2 in this repo's
    normalisation — evaluated through ``vacuum.core.entropy_from_nu`` so the
    vacuum floor is handled identically everywhere (x < 1 from roundoff is
    clamped to the floor, never a NaN).
    """
    return float(entropy_from_nu(np.atleast_1d(np.asarray(x, dtype=float)) / 2.0))


def mutual_information(V, modes_A=(0,), modes_B=(1,)) -> float:
    """I(A:B) in nats — ``vacuum.core.mutual_information``, wrapped verbatim."""
    return float(core_mutual_information(V, list(modes_A), list(modes_B)))


def minimum_conditional_determinant(A, B, C, D, tol=1e-12) -> Tuple[float, str]:
    """E^min of Adesso-Datta Eq. (4), with the branch that attained it.

    Returns ``(E_min, branch)`` with branch in {'uncorrelated', 'squeezed',
    'homodyne'}.  'uncorrelated' (gamma = 0, so C = 0 and D = AB) is the
    limit where no measurement on B tells anything about A and E^min = A;
    both analytic branches reduce to A there, but the first has (B-1)^2 in
    its denominator and is 0/0 for a pure B mode, so the case is taken
    explicitly.  ``tol`` is relative to max(1, AB).
    """
    A, B, C, D = float(A), float(B), float(C), float(D)
    scale = max(1.0, A * B)
    if abs(C) <= tol * scale and abs(D - A * B) <= tol * scale:
        return A, "uncorrelated"
    rad2 = C**4 + (D - A * B) ** 2 - 2.0 * C**2 * (A * B + D)
    E_hom = (A * B - C**2 + D - math.sqrt(max(0.0, rad2))) / (2.0 * B)
    if (D - A * B) ** 2 <= (1.0 + B) * C**2 * (A + D) and B - 1.0 > tol:
        rad1 = C**2 + (B - 1.0) * (D - A)
        E_sq = (
            2.0 * C**2 + (B - 1.0) * (D - A) + 2.0 * abs(C) * math.sqrt(max(0.0, rad1))
        ) / (B - 1.0) ** 2
        return E_sq, "squeezed"
    return E_hom, "homodyne"


def _oriented_invariants(V, modes_A, modes_B, measured):
    """Invariants with the *measured* mode in the 'B' slot of Eq. (4)."""
    if measured == "B":
        return two_mode_invariants(V, modes_A, modes_B)
    if measured == "A":
        inv = two_mode_invariants(V, modes_B, modes_A)  # swap roles
        return inv
    raise ValueError(f"measured must be 'A' or 'B', got {measured!r}")


def gaussian_discord(V, modes_A=(0,), modes_B=(1,), measured="B") -> float:
    """Gaussian quantum discord (nats), Adesso-Datta PRL 105, 030501, Eq. (3).

    ``measured='B'`` is D^{<-}(A|B) (Gaussian measurement on B, conditional
    entropy of A); ``measured='A'`` swaps the roles.  Restricted to Gaussian
    measurements — an upper bound on the unrestricted discord outside the
    Pirandola et al. (2014) family; see the module docstring.  Clamped at 0
    against roundoff (the closed form is >= 0 for physical states).
    """
    inv = _oriented_invariants(V, modes_A, modes_B, measured)
    E_min, _branch = minimum_conditional_determinant(inv.A, inv.B, inv.C, inv.D)
    D = (
        f_entropy(math.sqrt(inv.B))
        - f_entropy(inv.nu_minus)
        - f_entropy(inv.nu_plus)
        + f_entropy(math.sqrt(max(E_min, 0.0)))
    )
    return max(0.0, float(D))


def classical_correlation(V, modes_A=(0,), modes_B=(1,), measured="B") -> float:
    """One-way classical correlation J (nats) under Gaussian measurements.

    J^{<-} = f(sqrt A) - f(sqrt E^min): the entropy of the unmeasured mode
    minus its minimum conditional entropy after the optimal Gaussian
    measurement on the other (Adesso-Datta, text before Eq. (3)).
    """
    inv = _oriented_invariants(V, modes_A, modes_B, measured)
    E_min, _branch = minimum_conditional_determinant(inv.A, inv.B, inv.C, inv.D)
    return max(0.0, float(f_entropy(math.sqrt(inv.A)) - f_entropy(math.sqrt(max(E_min, 0.0)))))


@dataclass(frozen=True)
class GaussianCorrelations:
    """All Gaussian correlation measures of one two-mode block, in nats.

    ``I`` from the core route; ``D_B``/``J_B`` with the Gaussian measurement
    on B, ``D_A``/``J_A`` with it on A; ``I_invariants`` = J + D assembled
    from the invariants (the decomposition-identity check); the branch that
    attained E^min in each direction; the invariants themselves.
    """

    I: float
    D_B: float
    J_B: float
    D_A: float
    J_A: float
    I_invariants: float
    branch_B: str
    branch_A: str
    invariants: TwoModeInvariants
    gaussian_measurement_caveat: str = (
        "Gaussian discord: conditional entropy minimised over Gaussian "
        "measurements only (Adesso-Datta 2010 / Giorda-Paris 2010); optimal "
        "for the Pirandola et al. 2014 family, otherwise an upper bound on the "
        "unrestricted discord"
    )

    def as_row(self, prefix="") -> Dict[str, Any]:
        return {
            f"{prefix}MI": self.I,
            f"{prefix}D_B": self.D_B,
            f"{prefix}J_B": self.J_B,
            f"{prefix}D_A": self.D_A,
            f"{prefix}J_A": self.J_A,
            f"{prefix}MI_invariants": self.I_invariants,
            f"{prefix}discord_branch_B": self.branch_B,
            f"{prefix}discord_branch_A": self.branch_A,
        }


def gaussian_correlations(V, modes_A=(0,), modes_B=(1,)) -> GaussianCorrelations:
    """MI, both discords and both classical correlations of the A|B block."""
    inv_B = two_mode_invariants(V, modes_A, modes_B)
    inv_A = two_mode_invariants(V, modes_B, modes_A)
    E_B, br_B = minimum_conditional_determinant(inv_B.A, inv_B.B, inv_B.C, inv_B.D)
    E_A, br_A = minimum_conditional_determinant(inv_A.A, inv_A.B, inv_A.C, inv_A.D)
    fA, fB = f_entropy(math.sqrt(inv_B.A)), f_entropy(math.sqrt(inv_B.B))
    fnu = f_entropy(inv_B.nu_minus) + f_entropy(inv_B.nu_plus)
    fE_B = f_entropy(math.sqrt(max(E_B, 0.0)))
    fE_A = f_entropy(math.sqrt(max(E_A, 0.0)))
    D_B = max(0.0, fB - fnu + fE_B)
    J_B = max(0.0, fA - fE_B)
    D_A = max(0.0, fA - fnu + fE_A)
    J_A = max(0.0, fB - fE_A)
    return GaussianCorrelations(
        I=mutual_information(V, modes_A, modes_B),
        D_B=D_B,
        J_B=J_B,
        D_A=D_A,
        J_A=J_A,
        I_invariants=float(fA + fB - fnu),
        branch_B=br_B,
        branch_A=br_A,
        invariants=inv_B,
    )


# --- brute force over pure single-mode Gaussian seeds (the test's referee) --


def measurement_covariance(r, phi):
    """Covariance (sigma normalisation) of the pure squeezed seed |r, phi>.

    sigma_0 = R(phi) diag(e^{2r}, e^{-2r}) R(phi)^T; r = 0 is the heterodyne
    (coherent-state) POVM, |r| -> infinity the homodyne limit.
    """
    c, s = math.cos(phi), math.sin(phi)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag([math.exp(2.0 * r), math.exp(-2.0 * r)]) @ R.T


def conditional_determinant(alpha, beta, gamma, sigma_0=None, *, r=None, phi=None) -> float:
    """det eps, eps = alpha - gamma (beta + sigma_0)^{-1} gamma^T.

    Pass either a general seed covariance ``sigma_0`` or the pure-seed
    parameters ``(r, phi)``.  The latter path inverts beta + sigma_0 in the
    seed's own frame by the closed 2x2 formula, whose determinant
    (b11 + s)(b22 + 1/s) - b12^2 has no cancelling large terms, so the
    homodyne limit |r| -> infinity is reached without the conditioning loss
    of a generic solve at s = e^{2r} ~ 1e10 (which is what limited a first
    version of the brute-force referee to ~1e-8).
    """
    if sigma_0 is not None:
        eps = alpha - gamma @ np.linalg.solve(beta + sigma_0, gamma.T)
        return float(np.linalg.det(eps))
    c, sn = math.cos(float(phi)), math.sin(float(phi))
    R = np.array([[c, -sn], [sn, c]])
    b = R.T @ beta @ R
    g = gamma @ R
    s = math.exp(2.0 * float(r))
    m11, m22, m12 = b[0, 0] + s, b[1, 1] + 1.0 / s, b[0, 1]
    det = m11 * m22 - m12 * m12
    inv = np.array([[m22, -m12], [-m12, m11]]) / det
    eps = alpha - g @ inv @ g.T
    return float(np.linalg.det(eps))


def brute_force_gaussian_discord(
    V, modes_A=(0,), modes_B=(1,), measured="B", n_r=41, n_phi=48, r_max=12.0
) -> Dict[str, float]:
    """Discord by direct minimisation over pure Gaussian seeds (r, phi).

    Dense grid over r in [-r_max, r_max] x phi in [0, pi), then Nelder-Mead
    refinement from the best grid points; r is clamped to ±r_max inside the
    objective, where the homodyne limit is reached to O(e^{-2 r_max}).
    Returns ``E_min``, ``D``, ``J`` and the optimal ``(r, phi)``.  Test
    referee for :func:`gaussian_discord`; not for production sweeps.
    """
    if measured == "B":
        sigma, alpha, beta, gamma = _two_mode_sigma(V, modes_A, modes_B)
    elif measured == "A":
        sigma, beta, alpha, gamma_t = _two_mode_sigma(V, modes_A, modes_B)
        gamma = gamma_t.T
    else:
        raise ValueError(f"measured must be 'A' or 'B', got {measured!r}")
    nu = 2.0 * symplectic_eigenvalues(0.5 * sigma)
    fA = f_entropy(math.sqrt(float(np.linalg.det(alpha))))
    fB = f_entropy(math.sqrt(float(np.linalg.det(beta))))
    fnu = f_entropy(nu[0]) + f_entropy(nu[1])

    def objective(x):
        r = min(max(float(x[0]), -r_max), r_max)
        return f_entropy(
            math.sqrt(max(conditional_determinant(alpha, beta, gamma, r=r, phi=x[1]), 0.0))
        )

    rs = np.linspace(-r_max, r_max, int(n_r))
    phis = np.linspace(0.0, math.pi, int(n_phi), endpoint=False)
    best = []
    for r in rs:
        for phi in phis:
            best.append((objective((r, phi)), r, phi))
    best.sort(key=lambda t: t[0])
    val, r_opt, phi_opt = best[0]
    for v0, r0, phi0 in best[:6]:
        res = minimize(objective, x0=(r0, phi0), method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 4000})
        if res.fun < val:
            val, r_opt, phi_opt = float(res.fun), float(res.x[0]), float(res.x[1])
    r_opt = min(max(r_opt, -r_max), r_max)
    E_min = conditional_determinant(alpha, beta, gamma, r=r_opt, phi=phi_opt)
    return {
        "E_min": float(E_min),
        "f_E_min": float(val),
        "D": max(0.0, fB - fnu + val),
        "J": max(0.0, fA - val),
        "r": r_opt,
        "phi": phi_opt,
    }


def random_two_mode_state(rng, nu_max=2.0, strength=0.6):
    """A random physical two-mode covariance V = S diag(nu, nu) S^T (V_vac = I/2).

    S = exp(Omega H) for a random symmetric H (scaled by ``strength``), nu
    drawn uniformly in [1/2, nu_max/2]; generic (mixed, correlated, possibly
    entangled) states for the test suite.
    """
    from vacuum.core import Omega, symplectic_from_quadratic

    H = rng.normal(size=(4, 4))
    H = strength * (H + H.T) / 2.0
    S = symplectic_from_quadratic(H, 1.0)
    nu = rng.uniform(0.5, 0.5 * nu_max, size=2)
    return S @ np.diag(np.concatenate([nu, nu])) @ S.T


# ==========================================================================
# Fock representation of the detector block; coherence
# ==========================================================================


def fock_ladder(cutoff):
    """Annihilation operator a truncated to the lowest ``cutoff`` Fock levels."""
    cutoff = int(cutoff)
    if cutoff < 2:
        raise ValueError(f"cutoff must be >= 2, got {cutoff}")
    a = np.zeros((cutoff, cutoff))
    n = np.arange(1, cutoff)
    a[n - 1, n] = np.sqrt(n)
    return a


def fock_quadratures(cutoff, n_modes):
    """Block-ordered quadratures (x_1..x_n, p_1..p_n) in the truncated Fock space.

    x = (a + a^dag)/sqrt2, p = -i (a - a^dag)/sqrt2 per mode (dimensionless
    quadratures, [x, p] = i), tensored so that the basis index of |n_1 ...
    n_k> is sum_j n_j cutoff^{k-1-j} (first mode most significant).
    """
    a = fock_ladder(cutoff)
    x = (a + a.T) / _SQRT2
    p = -1j * (a - a.T) / _SQRT2
    eye = np.eye(cutoff)
    ops = []
    for op in (x, p):
        for k in range(n_modes):
            full = np.array([[1.0 + 0.0j]])
            for j in range(n_modes):
                full = np.kron(full, op if j == k else eye)
            ops.append(full)
    return ops


def rescale_to_dimensionless(VAB, gaps):
    """Covariance in the detectors' dimensionless quadratures x~ = sqrt(Omega) x, p~ = p/sqrt(Omega).

    The Fock basis of H_d = (1/2)(p^2 + Omega^2 x^2) is that of a = (x~ + i
    p~)/sqrt2; ``gaps=None`` means every Omega = 1 (already dimensionless).
    The rescaling is a local symplectic, so entropies and discords are
    unchanged by it — only the coherence, which names a basis, sees it.
    """
    VAB = np.asarray(VAB, dtype=float)
    n = VAB.shape[0] // 2
    if gaps is None:
        return VAB.copy()
    g = np.atleast_1d(np.asarray(gaps, dtype=float))
    if g.shape == (1,):
        g = np.full(n, float(g[0]))
    if g.shape != (n,) or np.any(g <= 0.0):
        raise ValueError(f"gaps must be {n} positive frequencies, got {g}")
    s = np.concatenate([np.sqrt(g), 1.0 / np.sqrt(g)])
    return (s[:, None] * VAB) * s[None, :]


def fock_density_matrix(VAB, gaps=None, cutoff=8):
    """Density matrix of the Gaussian state in the truncated local Fock basis.

    rho = exp(-(1/2) R^T G R) / Z with G the Gaussian modular Hamiltonian
    (``vacuum.core.entanglement_hamiltonian``, Botero-Reznik 2004) of the
    covariance in dimensionless quadratures, R the truncated quadratures of
    :func:`fock_quadratures`.  The truncation acts on the exponent, so the
    low-lying block converges to the exact matrix elements as the cutoff
    grows; :func:`gaussian_coherence` doubles the cutoff until it has.
    Hermitian, unit trace, ``(cutoff^n, cutoff^n)`` complex.
    """
    Vt = rescale_to_dimensionless(VAB, gaps)
    n = Vt.shape[0] // 2
    G = entanglement_hamiltonian(Vt)
    cutoff = int(cutoff)
    # Normal-ordered form of (1/2) R^T G R.  With v = (a_1..a_n, a_1^dag..
    # a_n^dag) and R = T v (x = (a + a^dag)/sqrt2, p = -i (a - a^dag)/sqrt2),
    # (1/2) R^T G R = (1/2) v^T M v, M = T^T G T; the a_k a_l^dag terms are
    # re-ordered as a_l^dag a_k + delta_kl.  Every normal-ordered quadratic
    # monomial, evaluated as a product of truncated ladder matrices, is
    # EXACTLY the projection P (.) P of the exact operator onto the retained
    # levels (a lowers, a^dag raises: the intermediate state never leaves
    # the block unless the result does), whereas the truncated quadratures
    # themselves are not (their x^2 + p^2 under-weights the top level).
    T = np.zeros((2 * n, 2 * n), dtype=complex)
    for k in range(n):
        T[k, k] = T[k, n + k] = 1.0 / _SQRT2
        T[n + k, k] = -1j / _SQRT2
        T[n + k, n + k] = 1j / _SQRT2
    M = T.T @ G @ T
    a_ops = []
    a1 = fock_ladder(cutoff)
    eye = np.eye(cutoff)
    for k in range(n):
        full = np.array([[1.0]])
        for j in range(n):
            full = np.kron(full, a1 if j == k else eye)
        a_ops.append(full.astype(complex))
    dim = cutoff**n
    H = np.zeros((dim, dim), dtype=complex)
    for k in range(n):
        for l in range(n):
            ak, al = a_ops[k], a_ops[l]
            akd, ald = ak.conj().T, al.conj().T
            H += 0.5 * M[k, l] * (ak @ al)                 # a_k a_l
            H += 0.5 * M[n + k, n + l] * (akd @ ald)       # a_k^dag a_l^dag
            H += 0.5 * M[n + k, l] * (akd @ al)            # a_k^dag a_l
            H += 0.5 * M[k, n + l] * (ald @ ak)            # a_k a_l^dag -> a_l^dag a_k (+ const)
    H = 0.5 * (H + H.conj().T)
    rho = expm(-H)
    rho = 0.5 * (rho + rho.conj().T)
    return rho / float(np.trace(rho).real)


def photon_number_distribution(VAB, gaps=None, cutoff=8):
    """Joint photon-number distribution p(n_1, ..., n_k) from the Fock matrix."""
    VAB = np.asarray(VAB, dtype=float)
    n = VAB.shape[0] // 2
    rho = fock_density_matrix(VAB, gaps, cutoff)
    p = np.real(np.diag(rho)).reshape((int(cutoff),) * n)
    return p


def photon_number_distribution_gf(VAB, gaps=None, cutoff=8, radius=0.5, n_fft=None):
    """Joint photon-number distribution from the closed-form generating function.

    For a zero-mean Gaussian state with covariance V (V_vac = I/2),

        G(z_1..z_k) = Tr[rho prod_j z_j^{N_j}]
                    = prod_j (1 - z_j)^{-1} * det(V + diag(nu_{z_j}) (+) diag(nu_{z_j}))^{-1/2},
        nu_z = (1 + z) / (2 (1 - z)),

    because z^{N} = (1-z)^{-1} rho_th(z) with rho_th(z) the thermal state of
    symplectic eigenvalue nu_z, and Tr[rho_1 rho_2] = det(V_1 + V_2)^{-1/2}
    for zero-mean Gaussian states.  p(n) are G's Taylor coefficients,
    recovered by an FFT on the polydisc of radius ``radius`` (< 1: G is
    analytic on the closed unit polydisc, its singularities lying outside);
    the square root's branch is fixed by phase unwrapping from z = radius
    (real, det > 0).  Independent of the operator construction — the
    diagonal's referee in the test suite.  Coefficients beyond ~cutoff are
    increasingly roundoff-amplified by radius^{-n}; keep cutoff modest.
    """
    Vt = rescale_to_dimensionless(VAB, gaps)
    n = Vt.shape[0] // 2
    if n not in (1, 2):
        raise ValueError("generating-function route implemented for 1 or 2 modes")
    M = int(n_fft) if n_fft is not None else max(4 * int(cutoff), 32)
    theta = 2.0 * math.pi * np.arange(M) / M
    z = radius * np.exp(1j * theta)
    nu_z = (1.0 + z) / (2.0 * (1.0 - z))
    if n == 1:
        dets = np.empty(M, dtype=complex)
        for i in range(M):
            dets[i] = np.linalg.det(Vt + nu_z[i] * np.eye(2))
        phase = np.unwrap(np.angle(dets))
        G = 1.0 / (1.0 - z) / (np.sqrt(np.abs(dets)) * np.exp(0.5j * phase))
        coeff = np.fft.fft(G) / M
        p = np.real(coeff[: int(cutoff)] * radius ** (-np.arange(int(cutoff))))
        return p
    dets = np.empty((M, M), dtype=complex)
    for i in range(M):
        for j in range(M):
            dets[i, j] = np.linalg.det(Vt + np.diag([nu_z[i], nu_z[j], nu_z[i], nu_z[j]]))
    phase = np.unwrap(np.unwrap(np.angle(dets), axis=1), axis=0)
    sqrt_det = np.sqrt(np.abs(dets)) * np.exp(0.5j * phase)
    G = 1.0 / ((1.0 - z)[:, None] * (1.0 - z)[None, :]) / sqrt_det
    coeff = np.fft.fft2(G) / (M * M)
    c = int(cutoff)
    scale = radius ** (-np.arange(c))
    return np.real(coeff[:c, :c]) * scale[:, None] * scale[None, :]


def shannon_entropy(p) -> float:
    """H(p) = -sum p ln p in nats (zeros and tiny negatives dropped)."""
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p)))


@dataclass
class CoherenceResult:
    """Coherence of a Gaussian block in its local Fock basis (nats).

    ``C_r``: relative entropy of coherence S(Delta rho) - S(rho), S(rho)
    exact from the symplectic spectrum; ``C_cc``: correlated coherence
    C_r(AB) - C_r(A) - C_r(B); ``C_r_A``, ``C_r_B``: the local terms;
    ``S_diag``: Shannon entropy of the joint photon-number distribution;
    ``S_exact``/``S_trunc``: the Gaussian entropy and the truncated matrix's
    own entropy (their difference is a truncation diagnostic); ``cutoff``:
    Fock cutoff per mode at acceptance; ``converged``, ``movement``,
    ``tail_weight`` (population of the highest retained level), ``gf_defect``
    (max |p - p_generating_function| over the compared levels) and
    ``criterion`` (which of the two acceptance tests fired); ``p_joint``
    the distribution; ``rho`` the truncated density matrix.
    """

    C_r: float
    C_cc: float
    C_r_A: float
    C_r_B: float
    S_diag: float
    S_exact: float
    S_trunc: float
    cutoff: int
    converged: bool
    movement: float
    tail_weight: float
    gf_defect: float
    criterion: str
    p_joint: np.ndarray
    rho: np.ndarray
    history: list = field(default_factory=list)

    def as_row(self, prefix="") -> Dict[str, Any]:
        return {
            f"{prefix}C_r": self.C_r,
            f"{prefix}C_cc": self.C_cc,
            f"{prefix}C_r_A": self.C_r_A,
            f"{prefix}C_r_B": self.C_r_B,
            f"{prefix}coherence_cutoff": float(self.cutoff),
            f"{prefix}coherence_converged": bool(self.converged),
            f"{prefix}coherence_movement": self.movement,
            f"{prefix}coherence_tail_weight": self.tail_weight,
            f"{prefix}coherence_gf_defect": self.gf_defect,
            f"{prefix}coherence_criterion": self.criterion,
            f"{prefix}coherence_entropy_defect": abs(self.S_trunc - self.S_exact),
        }


def _trunc_entropy(rho) -> float:
    w = np.linalg.eigvalsh(rho)
    w = w[w > 0.0]
    return float(-np.sum(w * np.log(w)))


def gaussian_coherence(
    VAB, gaps=None, cutoff0=4, tol=1e-10, max_cutoff=32, require_convergence=True,
    gf_check=True, gf_tol=1e-11, gf_levels=10, gf_radius=0.5,
) -> CoherenceResult:
    """C_r and C_cc of a two-mode Gaussian block with cutoff-doubling convergence.

    Doubles the per-mode Fock cutoff from ``cutoff0`` until, across one
    doubling, C_r moves by less than ``tol`` AND the population of the
    highest retained level of either mode is below ``tol``; the entropy of
    the truncated matrix against the exact Gaussian entropy is recorded as
    a third diagnostic.  Raises if ``max_cutoff`` is reached without
    convergence (unless ``require_convergence=False``) — silence is not an
    option for a truncation.

    Two independent acceptance criteria, either of which converges the
    cutoff (both also require the tail weight below ``tol``):

    1. the doubling movement of C_r falls below ``tol``;
    2. the accepted level's photon-number diagonal agrees with the CLOSED-FORM
       generating function (:func:`photon_number_distribution_gf`, which has no
       operator truncation at all) to ``gf_tol`` over its first ``gf_levels``
       levels per mode.

    The second exists because the movement is a pessimistic error estimate:
    it is dominated by the COARSER level (the convergence is geometric), so a
    state can sit at movement ~1e-9 while the accepted level is already exact
    to 1e-15 — measured on the map's hottest dephasing rows, where the
    doubling 16 -> 32 moves C_r by 8.9e-10 and the further doubling to 64 moves
    it by 2.9e-15.  Criterion 2 catches that without paying for cutoff 64
    (a 4096-dimensional ``expm``).  ``gf_defect`` is recorded either way.

    Convergence rate, measured: for a squeezed mode the low-lying matrix
    elements of the projected exponential converge as ~tanh(r)^cutoff — the
    ground state of the PROJECTED operator differs from the projection of the
    exact ground state at first order in the amplitude at the cut, not at
    second (a single-mode squeezed vacuum with r = 0.3 has p_0 errors 2e-4,
    2e-8, 4e-15 at cutoffs 8, 16, 32).  The movement across a doubling is
    therefore dominated by the coarser level's error, and the accepted
    cutoff's own error is below the recorded ``movement``.  Near-vacuum
    detector blocks (n_d <~ 1e-2) converge at cutoff 8-16; strongly
    squeezed or hot states need the cap raised.
    """
    VAB = np.asarray(VAB, dtype=float)
    if VAB.shape != (4, 4):
        raise ValueError(f"gaussian_coherence takes a two-mode (4, 4) block, got {VAB.shape}")
    Vt = rescale_to_dimensionless(VAB, gaps)
    S_exact = entropy(Vt)
    S_A = entropy(reduce_cov(Vt, [0]))
    S_B = entropy(reduce_cov(Vt, [1]))
    history = []
    prev = None
    cutoff = int(cutoff0)
    result = None
    converged = False
    p_gf = None
    if gf_check:
        p_gf = photon_number_distribution_gf(Vt, None, int(gf_levels), radius=float(gf_radius))
    while True:
        rho = fock_density_matrix(Vt, None, cutoff)
        p = np.real(np.diag(rho)).reshape(cutoff, cutoff)
        p = np.clip(p, 0.0, None)
        S_diag = shannon_entropy(p)
        C_r = S_diag - S_exact
        C_r_A = shannon_entropy(p.sum(axis=1)) - S_A
        C_r_B = shannon_entropy(p.sum(axis=0)) - S_B
        tail = float(p[-1, :].sum() + p[:, -1].sum())
        movement = float("nan") if prev is None else abs(C_r - prev)
        n_gf = min(int(gf_levels), cutoff)
        gf_defect = (float("nan") if p_gf is None
                     else float(np.max(np.abs(p[:n_gf, :n_gf] - p_gf[:n_gf, :n_gf]))))
        history.append({"cutoff": cutoff, "C_r": C_r, "tail_weight": tail,
                        "movement": movement, "gf_defect": gf_defect})
        result = CoherenceResult(
            C_r=float(C_r), C_cc=float(C_r - C_r_A - C_r_B), C_r_A=float(C_r_A),
            C_r_B=float(C_r_B), S_diag=float(S_diag), S_exact=float(S_exact),
            S_trunc=_trunc_entropy(rho), cutoff=cutoff, converged=False,
            movement=movement, tail_weight=tail, gf_defect=gf_defect,
            criterion="none", p_joint=p, rho=rho, history=history,
        )
        if tail < tol:
            if prev is not None and movement < tol:
                converged, result.criterion = True, "movement"
                break
            if p_gf is not None and gf_defect < gf_tol:
                converged, result.criterion = True, "generating_function"
                break
        prev = C_r
        if 2 * cutoff > int(max_cutoff):
            break
        cutoff *= 2
    result.converged = converged
    if not converged and require_convergence:
        raise RuntimeError(
            f"gaussian_coherence: Fock cutoff did not converge by cutoff {cutoff} "
            f"(movement {result.movement:.3e} vs tol {tol:g}; generating-function "
            f"defect {result.gf_defect:.3e} vs {gf_tol:g}; tail {result.tail_weight:.3e})"
        )
    return result


def qubit_proxy_from_fock(rho, cutoff) -> Tuple[np.ndarray, float]:
    """The {0,1}^2 Fock block of a two-mode Fock matrix, renormalised.

    Returns ``(rho_qubits, weight)`` with rho_qubits in A x B order
    (index 2 n_A + n_B) and ``weight`` the trace of the block before
    renormalisation — assumption (A5) of the resolvability model.
    """
    cutoff = int(cutoff)
    idx = [0 * cutoff + 0, 0 * cutoff + 1, 1 * cutoff + 0, 1 * cutoff + 1]
    block = np.asarray(rho)[np.ix_(idx, idx)]
    w = float(np.trace(block).real)
    return block / w, w


# ==========================================================================
# Qubit pair: MI, discord (projective), coherence, PKMM Eq. (76), MI split
# ==========================================================================

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
#: single-qubit Paulis (I, X, Y, Z)
PAULI = (_I2, _X, _Y, _Z)


def qubit_pair_from_udw(rho_or_state):
    """PKMM Eq. (14) basis {gg, eg, ge, ee} -> A x B order {gg, ge, eg, ee}.

    Accepts a ``UDWPairState`` (its ``.rho``) or a 4x4 matrix in PKMM order;
    returns the same state with index 2 a + b (a, b in {0 = g, 1 = e}), the
    ordering every function below uses.
    """
    rho = getattr(rho_or_state, "rho", rho_or_state)
    rho = np.asarray(rho, dtype=complex)
    perm = [0, 2, 1, 3]
    return rho[np.ix_(perm, perm)]


def _clip_spectrum(rho):
    """Eigenvalues clipped at 0 and the clipped (negative) weight."""
    w = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    neg = float(-np.sum(w[w < 0.0]))
    w = np.clip(w, 0.0, None)
    return w, neg


def qubit_entropy(rho) -> float:
    """S(rho) in nats from the clipped spectrum (clip weight discarded here)."""
    w, _ = _clip_spectrum(np.asarray(rho, dtype=complex))
    w = w[w > 0.0]
    return float(-np.sum(w * np.log(w)))


def qubit_marginals(rho):
    """(rho_A, rho_B) of a 4x4 state in A x B order."""
    r = np.asarray(rho, dtype=complex).reshape(2, 2, 2, 2)
    return np.einsum("abcb->ac", r), np.einsum("abad->bd", r)


def qubit_mutual_information(rho) -> float:
    """I(A:B) = S(A) + S(B) - S(AB), nats."""
    rA, rB = qubit_marginals(rho)
    return float(qubit_entropy(rA) + qubit_entropy(rB) - qubit_entropy(rho))


def _bloch_projectors(theta, phi):
    """Rank-1 projectors (K, 2, 2, 2): [k, ±, :, :] = (I ± n_k.sigma)/2."""
    theta = np.asarray(theta, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    nx, ny, nz = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
    n_sigma = nx[:, None, None] * _X + ny[:, None, None] * _Y + nz[:, None, None] * _Z
    return np.stack([0.5 * (_I2 + n_sigma), 0.5 * (_I2 - n_sigma)], axis=1)


def _entropy_2x2_batch(m):
    """Entropies of a batch (..., 2, 2) of unnormalised PSD matrices, with their traces."""
    tr = np.real(m[..., 0, 0] + m[..., 1, 1])
    det = np.real(m[..., 0, 0] * m[..., 1, 1]) - np.abs(m[..., 0, 1]) ** 2
    disc = np.sqrt(np.clip(tr * tr - 4.0 * det, 0.0, None))
    lam = np.stack([0.5 * (tr + disc), 0.5 * (tr - disc)], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = lam / np.where(tr[..., None] > 0.0, tr[..., None], 1.0)
        q = np.clip(q, 0.0, 1.0)
        h = -np.where(q > 0.0, q * np.log(np.where(q > 0.0, q, 1.0)), 0.0).sum(axis=-1)
    return h, tr


def _conditional_entropy_batch(rho4, projectors):
    """S(A | {Pi_k}) for a batch of measurements on B.

    ``rho4``: (..., 2, 2, 2, 2) states rho[a, b, a', b']; ``projectors``:
    (K, 2, 2, 2).  Returns (..., K).
    """
    # unnormalised conditional states of A: sum_{b b'} rho[a b a' b'] Pi[b' b]
    cond = np.einsum("...abcd,kpdb->...kpac", rho4, projectors)
    h, p = _entropy_2x2_batch(cond)
    return np.sum(p * h, axis=-1)


def qubit_discord(rho, measured="B", n_theta=24, n_phi=48, refine=True) -> Dict[str, Any]:
    """Discord of a two-qubit state under projective measurements (nats).

    Minimises the conditional entropy S(A|{Pi}) over rank-1 projective
    measurements of the measured qubit on a (theta, phi) grid, then refines
    the best grid point by Nelder-Mead.  Returns ``D`` (clamped at 0), ``J``,
    ``I``, the minimum conditional entropy, the optimal Bloch angles and the
    clipped weight of the input spectrum.
    """
    rho = np.asarray(rho, dtype=complex)
    if measured == "A":
        rho_m = rho.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
    elif measured == "B":
        rho_m = rho
    else:
        raise ValueError(f"measured must be 'A' or 'B', got {measured!r}")
    rho4 = rho_m.reshape(2, 2, 2, 2)
    rA, rB = qubit_marginals(rho_m)
    S_A = qubit_entropy(rA)
    I = float(S_A + qubit_entropy(rB) - qubit_entropy(rho_m))
    _w, clipped = _clip_spectrum(rho_m)

    th = np.linspace(0.0, math.pi, int(n_theta))
    ph = np.linspace(0.0, 2.0 * math.pi, int(n_phi), endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    S_cond = _conditional_entropy_batch(rho4, _bloch_projectors(TH.ravel(), PH.ravel()))
    k = int(np.argmin(S_cond))
    best = float(S_cond[k])
    theta_opt, phi_opt = float(TH.ravel()[k]), float(PH.ravel()[k])
    if refine:
        def obj(x):
            return float(_conditional_entropy_batch(rho4, _bloch_projectors([x[0]], [x[1]]))[0])
        res = minimize(obj, x0=(theta_opt, phi_opt), method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 2000})
        if res.fun < best:
            best, theta_opt, phi_opt = float(res.fun), float(res.x[0]), float(res.x[1])
    J = S_A - best
    return {
        "D": max(0.0, I - J),
        "D_raw": I - J,
        "J": J,
        "I": I,
        "S_cond_min": best,
        "theta": theta_opt,
        "phi": phi_opt,
        "clipped_weight": clipped,
        "measured": measured,
    }


def qubit_coherence(rho) -> Dict[str, float]:
    """Relative entropy of coherence in the product energy basis, and C_cc.

    C_r = S(diag rho) - S(rho); the local terms use the marginals' diagonals;
    C_cc = C_r(AB) - C_r(A) - C_r(B) (Tan et al. 2016).
    """
    rho = np.asarray(rho, dtype=complex)
    rA, rB = qubit_marginals(rho)
    C_r = shannon_entropy(np.clip(np.real(np.diag(rho)), 0.0, None)) - qubit_entropy(rho)
    C_A = shannon_entropy(np.clip(np.real(np.diag(rA)), 0.0, None)) - qubit_entropy(rA)
    C_B = shannon_entropy(np.clip(np.real(np.diag(rB)), 0.0, None)) - qubit_entropy(rB)
    return {"C_r": float(C_r), "C_r_A": float(C_A), "C_r_B": float(C_B),
            "C_cc": float(C_r - C_A - C_B)}


def concurrence(rho) -> float:
    """Wootters concurrence of a two-qubit state (A x B order)."""
    rho = np.asarray(rho, dtype=complex)
    YY = np.kron(_Y, _Y)
    R = rho @ YY @ rho.conj() @ YY
    w = np.sort(np.sqrt(np.clip(np.real(np.linalg.eigvals(R)), 0.0, None)))[::-1]
    return float(max(0.0, w[0] - w[1] - w[2] - w[3]))


def qubit_log_negativity(rho) -> float:
    """E_N = ln ||rho^{T_B}||_1 (nats) of a two-qubit state."""
    r = np.asarray(rho, dtype=complex).reshape(2, 2, 2, 2)
    pt = r.transpose(0, 3, 2, 1).reshape(4, 4)
    w = np.linalg.eigvalsh(0.5 * (pt + pt.conj().T))
    return float(math.log(np.sum(np.abs(w))))


def qubit_correlations(rho, refine=True) -> Dict[str, Any]:
    """All qubit-pair measures of a 4x4 state in A x B order (nats)."""
    dB = qubit_discord(rho, "B", refine=refine)
    dA = qubit_discord(rho, "A", refine=refine)
    coh = qubit_coherence(rho)
    return {
        "MI": dB["I"],
        "D_B": dB["D"], "J_B": dB["J"],
        "D_A": dA["D"], "J_A": dA["J"],
        "C_r": coh["C_r"], "C_cc": coh["C_cc"], "C_r_A": coh["C_r_A"], "C_r_B": coh["C_r_B"],
        "E_N": qubit_log_negativity(rho),
        "concurrence": concurrence(rho),
        "clipped_weight": dB["clipped_weight"],
    }


def pkmm_mutual_information(P_A, P_B, C) -> float:
    """Leading-order two-detector MI, PKMM Eqs. (76)-(77) (nats).

    I = L_+ ln L_+ + L_- ln L_- - L_AA ln L_AA - L_BB ln L_BB with
    L_± = (1/2)[L_AA + L_BB ± sqrt((L_AA - L_BB)^2 + 4 |L_AB|^2)];
    independent of the pair term M.  Zero when C = 0 or a population is 0.
    NaN when |C|^2 > L_AA L_BB: the {|eg>, |ge>} block is then not positive
    (L_- < 0) and the leading-order expression is not the MI of a state —
    which happens to the anti-commutator part alone at causal contact (see
    :func:`mutual_information_split`); the full C never violates it (W is a
    positive kernel).
    """
    P_A, P_B, c = float(P_A), float(P_B), abs(complex(C))
    if P_A <= 0.0 or P_B <= 0.0:
        return 0.0
    if c * c > P_A * P_B * (1.0 + 1e-12):
        return float("nan")
    disc = math.hypot(P_A - P_B, 2.0 * c)
    Lp, Lm = 0.5 * (P_A + P_B + disc), 0.5 * (P_A + P_B - disc)
    out = Lp * math.log(Lp) if Lp > 0.0 else 0.0
    out += Lm * math.log(Lm) if Lm > 0.0 else 0.0
    return float(max(0.0, out - P_A * math.log(P_A) - P_B * math.log(P_B)))


def mutual_information_split(
    kernel, det_A, det_B, lam=1.0, gap=0.0, coupling="xx", C=None, P_A=None, P_B=None,
    comm_atol=None, **quad_kwargs
) -> Dict[str, Any]:
    """Split the leading-order MI into harvested and signalling parts (TMM21 logic on L_AB).

    Runs the cross term with the commutator kernel i Im W (state-independent,
    light-cone supported) to get ``C_comm``; ``C_vac = C - C_comm``.  Then
    ``MI_pkmm`` = PKMM Eq. (76) with the full C, ``MI_vac`` the same with
    C_vac, ``MI_comm = MI_pkmm - MI_vac``, and ``comm_fraction_C`` =
    |C_comm|/|C|.  ``P_A``, ``P_B`` and the full ``C`` may be supplied from
    an existing ``UDWPairState`` (they are not recomputed then).  For
    spacelike-separated compact switchings C_comm is analytically zero, so
    the commutator run gets the absolute tolerance ``comm_atol`` (default
    max(atol, rtol |C|)) exactly as ``communication_split`` does for M.
    Perturbative by construction (O(lambda^2) matrix elements).

    Two facts the numbers carry.  (i) For identical detectors both halves
    of C are REAL (the commutator kernel is imaginary and antisymmetric,
    hence Hermitian, so its quadratic form is real) and can have opposite
    signs: |C| < |C_vac| is possible and occurs at causal contact — the
    signalling part interferes destructively with the harvested part
    (TMM21 call the contributions sub-additive).  (ii) The full W is a
    positive kernel, so |C|^2 <= L_AA L_BB always; the anti-commutator
    part alone is bounded only by the anti-commutator populations, which
    exceed L_AA, so |C_vac| can exceed sqrt(L_AA L_BB) and then the
    leading-order "harvested-only" MI is not the MI of a state: ``MI_vac``
    is NaN, ``vac_part_physical`` False, and ``C_vac_over_population`` =
    |C_vac|/sqrt(L_AA L_BB) says by how much.  At spacelike separation
    C_vac = C and the split is trivially physical.
    """
    lam_A, lam_B, gap_A, gap_B = _pair_params(lam, gap)
    ker_AA, ker_BB, ker_AB = pair_kernels(kernel, det_A, det_B, coupling=coupling)
    rtol = float(quad_kwargs.get("rtol", 1e-10))
    atol = float(quad_kwargs.get("atol", 1e-18))
    meta: Dict[str, Any] = {"coupling": coupling, "backend": type(kernel).__name__}
    if C is None:
        C, meta_C = cross_term(ker_AB, det_A.chi, det_B.chi, gap_A, gap_B, lam_A, lam_B,
                               return_meta=True, **quad_kwargs)
        meta["full"] = meta_C
    C = complex(C)
    if P_A is None or P_B is None:
        from .udw import transition_probability
        resp = {k: v for k, v in quad_kwargs.items()
                if k in ("rtol", "atol", "n_gl", "max_doublings", "refine_kernel")}
        P_A = transition_probability(ker_AA, det_A.chi, gap_A, lam_A, **resp)
        P_B = transition_probability(ker_BB, det_B.chi, gap_B, lam_B, **resp)
    if comm_atol is None:
        comm_atol = max(atol, rtol * abs(C))
    kw = dict(quad_kwargs)
    kw["atol"] = float(comm_atol)
    C_comm, meta_comm = cross_term(commutator_kernel(ker_AB), det_A.chi, det_B.chi, gap_A,
                                   gap_B, lam_A, lam_B, return_meta=True, **kw)
    meta["commutator"] = meta_comm
    meta["comm_atol"] = float(comm_atol)
    C_vac = C - C_comm
    MI_full = pkmm_mutual_information(P_A, P_B, C)
    MI_vac = pkmm_mutual_information(P_A, P_B, C_vac)
    pop = math.sqrt(max(float(P_A) * float(P_B), 0.0))
    ratio = (abs(C_vac) / pop) if pop > 0.0 else float("inf")
    physical = bool(ratio <= 1.0 + 1e-12)
    MI_comm = (MI_full - MI_vac) if physical else float("nan")
    return {
        "P_A": float(P_A), "P_B": float(P_B),
        "C": C, "C_vac": C_vac, "C_comm": C_comm,
        "comm_fraction_C": (abs(C_comm) / abs(C)) if abs(C) > 0.0 else 0.0,
        "C_vac_over_population": float(ratio),
        "vac_part_physical": physical,
        "MI_pkmm": MI_full, "MI_vac": MI_vac if physical else float("nan"),
        "MI_comm": MI_comm,
        "comm_fraction_MI": ((MI_comm / MI_full) if (physical and MI_full > 0.0)
                             else (0.0 if physical else float("nan"))),
        "meta": meta,
    }


# ==========================================================================
# Resolvability model at the tomography floor
# ==========================================================================


@dataclass(frozen=True)
class TomographyPrecision:
    """Pauli-expectation precision of an N-shot, fidelity-F tomography.

    ``sigma_single`` = 1/(v sqrt N), ``sigma_double`` = 1/(v^2 sqrt N),
    v = 2F - 1 (assumptions (A1)-(A2) of the module docstring).
    """

    n_shots: float
    readout_fidelity: float
    n_settings: int
    sigma_single: float
    sigma_double: float
    provenance: str = "chosen"


def tomography_precision(n_shots, readout_fidelity, n_settings=9, provenance="chosen"):
    """Standard errors of the Pauli expectations (assumption (A2))."""
    N = float(n_shots)
    F = float(readout_fidelity)
    if N <= 0.0 or not (0.5 < F <= 1.0):
        raise ValueError(f"need n_shots > 0 and 0.5 < F <= 1, got {N}, {F}")
    v = 2.0 * F - 1.0
    return TomographyPrecision(
        n_shots=N, readout_fidelity=F, n_settings=int(n_settings),
        sigma_single=1.0 / (v * math.sqrt(N)), sigma_double=1.0 / (v * v * math.sqrt(N)),
        provenance=str(provenance),
    )


def tomography_precision_from_experiment(
    experiment="circuit_qed_harvesting/ren_2022_sec4-3", root=None
) -> TomographyPrecision:
    """The precision of Ren 2022's simulated tomography, read from its parameter file.

    Quantities ``qst_shots_per_prerotation``, ``single_shot_readout_fidelity``
    (percent) and ``qst_prerotations`` of
    ``experiments/circuit_qed_harvesting/ren_2022_sec4-3.json`` (all
    transcribed; the fidelity is an *assumed* value of the thesis's own
    simulation, "roughly in line with our setup").
    """
    exp = experiment if hasattr(experiment, "quantities") else load_experiment(experiment, root=root)
    q = exp.quantities
    N = float(q["qst_shots_per_prerotation"].value)
    F = float(q["single_shot_readout_fidelity"].value)
    if q["single_shot_readout_fidelity"].units in ("percent", "%"):
        F /= 100.0
    n_set = int(q["qst_prerotations"].value)
    return tomography_precision(N, F, n_set, provenance=f"{exp.name}: {exp.source[:60]}...")


def coherence_precision_from_negativity_floor(E_floor) -> float:
    """delta = E_floor / 2: the coherence a concurrence floor resolves.

    For Ren's near-pure X state (rho_44 ~ 0) C = 2 |rho_23| and the derived
    E_N floor is E_N ~ C, so a floor of E_floor nats corresponds to resolving
    an |eg>-|ge> coherence of E_floor/2.
    """
    return 0.5 * float(E_floor)


_PAULI_BASIS = [np.kron(PAULI[i], PAULI[j]) for i in range(4) for j in range(4)]
_PAULI_IS_SINGLE = np.array([(i == 0) != (j == 0) for i in range(4) for j in range(4)])


def pauli_expectations(rho):
    """The 16 expectations Tr[rho sigma_i x sigma_j] (index 4 i + j; entry 0 is 1)."""
    rho = np.asarray(rho, dtype=complex)
    return np.array([np.real(np.trace(P @ rho)) for P in _PAULI_BASIS])


def state_from_pauli(expect):
    """rho = (1/4) sum e_ij sigma_i x sigma_j from (..., 16) expectations."""
    e = np.asarray(expect, dtype=float)
    basis = np.array(_PAULI_BASIS)  # (16, 4, 4)
    return 0.25 * np.einsum("...k,kab->...ab", e, basis)


def project_to_physical(rho):
    """Nearest unit-trace PSD matrix in 2-norm (Smolin-Gambetta-Smith 2012).

    Eigen-decompose, then project the spectrum onto the probability simplex
    (largest k such that lambda_k - (sum_{i<=k} lambda_i - 1)/k > 0; subtract
    that offset; clip).  Works on a batch (..., d, d).
    """
    rho = np.asarray(rho, dtype=complex)
    rho = 0.5 * (rho + np.swapaxes(rho, -1, -2).conj())
    w, U = np.linalg.eigh(rho)
    w = w[..., ::-1]
    U = U[..., ::-1]
    d = w.shape[-1]
    cs = np.cumsum(w, axis=-1)
    k = np.arange(1, d + 1)
    mu = (cs - 1.0) / k
    ok = (w - mu) > 0.0
    kstar = np.sum(ok, axis=-1)  # ok is monotone (true for the first k*)
    kstar = np.maximum(kstar, 1)
    mu_star = np.take_along_axis(mu, (kstar - 1)[..., None], axis=-1)
    w_new = np.clip(w - mu_star, 0.0, None)
    return np.einsum("...ab,...b,...cb->...ac", U, w_new, U.conj())


def noisy_reconstructions(rho, precision: TomographyPrecision, n_samples=500, rng=None):
    """``n_samples`` physical reconstructions of rho under assumptions (A2)-(A3)."""
    rng = np.random.default_rng(rng)
    e = pauli_expectations(rho)
    sig = np.where(_PAULI_IS_SINGLE, precision.sigma_single, precision.sigma_double)
    sig[0] = 0.0
    noisy = e[None, :] + rng.normal(size=(int(n_samples), 16)) * sig[None, :]
    return project_to_physical(state_from_pauli(noisy))


def _batch_entropy(rho):
    w = np.clip(np.linalg.eigvalsh(rho), 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(w > 0.0, w * np.log(np.where(w > 0.0, w, 1.0)), 0.0)
    return -np.sum(t, axis=-1)


def _batch_marginals(rho):
    r = rho.reshape(rho.shape[:-2] + (2, 2, 2, 2))
    return np.einsum("...abcb->...ac", r), np.einsum("...abad->...bd", r)


def batch_qubit_measures(rho, n_theta=16, n_phi=32) -> Dict[str, np.ndarray]:
    """MI, discords (grid-only), C_r, C_cc, concurrence, E_N for a batch (n, 4, 4).

    The discord minimisation uses the Bloch grid without refinement: an
    upper bound on each sample's projective discord (conservative when the
    batch is the noise reference of a floor).
    """
    rho = np.asarray(rho, dtype=complex)
    rA, rB = _batch_marginals(rho)
    S_AB, S_A, S_B = _batch_entropy(rho), _batch_entropy(rA), _batch_entropy(rB)
    I = S_A + S_B - S_AB
    th = np.linspace(0.0, math.pi, int(n_theta))
    ph = np.linspace(0.0, 2.0 * math.pi, int(n_phi), endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    proj = _bloch_projectors(TH.ravel(), PH.ravel())
    rho4 = rho.reshape(rho.shape[:-2] + (2, 2, 2, 2))
    S_cond_B = np.min(_conditional_entropy_batch(rho4, proj), axis=-1)
    rho4_swapped = rho4.transpose(0, 2, 1, 4, 3)
    S_cond_A = np.min(_conditional_entropy_batch(rho4_swapped, proj), axis=-1)
    D_B = np.clip(I - (S_A - S_cond_B), 0.0, None)
    D_A = np.clip(I - (S_B - S_cond_A), 0.0, None)
    diag = np.clip(np.real(np.diagonal(rho, axis1=-2, axis2=-1)), 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = lambda p: -np.sum(np.where(p > 0.0, p * np.log(np.where(p > 0.0, p, 1.0)), 0.0), axis=-1)
    C_r = H(diag) - S_AB
    C_A = H(np.clip(np.real(np.diagonal(rA, axis1=-2, axis2=-1)), 0.0, None)) - S_A
    C_B = H(np.clip(np.real(np.diagonal(rB, axis1=-2, axis2=-1)), 0.0, None)) - S_B
    # concurrence and E_N
    YY = np.kron(_Y, _Y)
    R = rho @ YY @ rho.conj() @ YY
    lam = np.sort(np.sqrt(np.clip(np.real(np.linalg.eigvals(R)), 0.0, None)), axis=-1)[..., ::-1]
    conc = np.clip(lam[..., 0] - lam[..., 1] - lam[..., 2] - lam[..., 3], 0.0, None)
    pt = rho4.transpose(0, 1, 4, 3, 2).reshape(rho.shape)
    E_N = np.log(np.sum(np.abs(np.linalg.eigvalsh(0.5 * (pt + np.swapaxes(pt, -1, -2).conj()))), axis=-1))
    return {"MI": I, "D_B": D_B, "D_A": D_A, "C_r": C_r, "C_cc": C_r - C_A - C_B,
            "concurrence": conc, "E_N": E_N}


#: the measures the resolvability model floors
FLOOR_MEASURES = ("MI", "D_B", "D_A", "C_r", "C_cc", "E_N", "concurrence")


def monte_carlo_floors(
    rho, precision: TomographyPrecision, n_samples=500, seed=0, quantile=0.95,
    n_theta=16, n_phi=32,
) -> Dict[str, Any]:
    """Per-state resolvability of every measure at the tomography floor (A1)-(A5).

    For the target ``rho`` (A x B order) and its uncorrelated reference
    rho_A x rho_B, draws ``n_samples`` reconstructions each and returns, per
    measure: the true value, the ``quantile`` of the reference's spurious
    value (the FLOOR), the reference median, the target reconstruction's
    median and (1 - quantile) lower quantile, ``resolvable`` (true value >
    floor) and ``detectable`` (target lower quantile > floor).
    """
    rho = np.asarray(rho, dtype=complex)
    rng = np.random.default_rng(seed)
    rA, rB = qubit_marginals(rho)
    ref = np.kron(rA, rB)
    true = batch_qubit_measures(rho[None, :, :], n_theta, n_phi)
    rec_t = batch_qubit_measures(noisy_reconstructions(rho, precision, n_samples, rng), n_theta, n_phi)
    rec_r = batch_qubit_measures(noisy_reconstructions(ref, precision, n_samples, rng), n_theta, n_phi)
    out: Dict[str, Any] = {"n_samples": int(n_samples), "quantile": float(quantile),
                           "sigma_single": precision.sigma_single,
                           "sigma_double": precision.sigma_double, "measures": {}}
    for m in FLOOR_MEASURES:
        floor = float(np.quantile(rec_r[m], quantile))
        lower = float(np.quantile(rec_t[m], 1.0 - quantile))
        tv = float(true[m][0])
        nonzero = rec_t[m][rec_t[m] > 0.0]
        out["measures"][m] = {
            "true": tv, "floor": floor,
            "reference_median": float(np.median(rec_r[m])),
            "reference_max": float(np.max(rec_r[m])),
            "target_median": float(np.median(rec_t[m])),
            "target_lower": lower,
            "target_upper": float(np.quantile(rec_t[m], quantile)),
            "target_nonzero_range": ([float(np.min(nonzero)), float(np.max(nonzero))]
                                     if nonzero.size else [0.0, 0.0]),
            "target_zero_fraction": float(np.mean(rec_t[m] <= 0.0)),
            "resolvable": bool(tv > floor),
            "detectable": bool(lower > floor),
        }
    return out


def floor_state(P_A, P_B, delta):
    """The product of the marginals diag(1-P, P) plus ONE |ge>-|eg> coherence.

    Populations ((1-P_A)(1-P_B), (1-P_A) P_B, P_A (1-P_B), P_A P_B) on
    (gg, ge, eg, ee) — exactly the uncorrelated reference, so every measure
    vanishes at delta = 0 — and the coherence min(delta, sqrt(rho_ge rho_eg))
    at (1, 2) of the A x B matrix (positivity of the {|ge>, |eg>} block caps
    it).  A first version used (1-P_A-P_B, P_A, P_B, 0), which is not a
    product at zero coherence and carried a spurious classical MI ~ P^2.
    """
    P_A, P_B, delta = float(P_A), float(P_B), float(delta)
    rho_ge, rho_eg = (1.0 - P_A) * P_B, P_A * (1.0 - P_B)
    c = min(delta, math.sqrt(max(rho_ge * rho_eg, 0.0)))
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = (1.0 - P_A) * (1.0 - P_B)
    rho[1, 1] = rho_ge  # |g e>: index 2*0 + 1
    rho[2, 2] = rho_eg  # |e g>
    rho[3, 3] = P_A * P_B
    rho[1, 2] = c
    rho[2, 1] = c
    return rho


def correlation_floors(P_A, P_B, delta, refine=True) -> Dict[str, float]:
    """Analytic-companion floors: the measures of :func:`floor_state`.

    What each measure reads on a state whose only correlation is a single
    coherence at the tomography precision ``delta`` between the detectors'
    one-excitation levels, on top of the product of the marginals at
    populations (P_A, P_B).  For delta << P the MI is ~ delta^2 / P
    (quadratic response), far below the E_N floor; for P <~ delta it
    saturates at the positivity cap, MI -> 2 P ln 2 for equal P.
    """
    rho = floor_state(P_A, P_B, delta)
    q = qubit_correlations(rho, refine=refine)
    return {k: float(q[k]) for k in ("MI", "D_B", "D_A", "J_B", "J_A", "C_r", "C_cc", "E_N", "concurrence")}
