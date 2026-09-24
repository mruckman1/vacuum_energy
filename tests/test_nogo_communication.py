"""Milestone M2.5 anchors: the Simidzija null and the communication split.

Two permanent rows of the plan's test matrix live here:

    | Simidzija null   | N < 1e-14 |
    | Spacelike M_comm | < 1e-12   |

The first is the suite's *designated exact null* -- the test that proves the
engine can output zero.  It is anchored twice over: against the closed form
of Simidzija & Martin-Martinez, PRD 96, 065008 (2017), arXiv:1707.00016
(``vacuum/detectors/nogo.py``), and against a brute-force truncated-Fock
evaluation of that paper's *definition* Eq. (57), which shares no algebra
with the closed form at all.

The second doubles as a physics-level causality audit on the whole detector
stack: with strictly compact ``cos2`` switchings and a separation outside
the light cone, the commutator (signalling) half of the pair term must
vanish.

Every assertion message carries the measured number (docs/API.md).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from vacuum.core.dynamics import chain_propagator
from vacuum.core.gaussian import Omega
from vacuum.core.models import harmonic_chain_K
from vacuum.detectors import (
    UDWDetector,
    smearing,
    switching,
    wightman_continuum_3p1,
    wightman_lattice,
)
from vacuum.detectors.communication import (
    commutator_kernel,
    communication_columns,
    communication_estimator,
    communication_split,
    hadamard_kernel,
    pair_state_with_split,
)
from vacuum.detectors.nogo import (
    coherent_shift,
    delta_coupled_negativity,
    delta_coupled_overlaps,
    delta_coupled_pair,
    delta_coupled_rho,
    lattice_coherent_shifts,
    partial_transpose_A,
    pt_eigenvalues,
    single_detector_eigenvalues,
    single_detector_rho,
    wightman_triple,
)
from vacuum.detectors.udw import pair_negativity, pair_term

# Spec tolerances (docs/PLAN_LAYERS_2_3.md, M2.5 and the test matrix).
NULL_TOL = 1e-14          # Simidzija negativity null
SPACELIKE_TOL = 1e-12     # |M_comm| for spacelike-separated switchings

N_DRAWS = 200
SEED = 20260901


# ---------------------------------------------------------------------------
# Shared, expensive-to-build fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chain():
    """A massive periodic harmonic chain (positive definite, no zero mode)."""
    return harmonic_chain_K(24, 0.35, bc="periodic")


@pytest.fixture(scope="module")
def lattice_ker(chain):
    return wightman_lattice(chain)


@pytest.fixture(scope="module")
def profiles():
    F_A = np.asarray(smearing("gaussian", 6.0, 1.2, 24), dtype=float)
    F_B = np.asarray(smearing("gaussian", 17.0, 1.2, 24), dtype=float)
    return F_A, F_B


@pytest.fixture(scope="module")
def continuum_ker():
    """Continuum 3+1 cross kernel, sigma = 0.25, detector separation d = 4."""
    return wightman_continuum_3p1(0.25, distance=4.0)


# ---------------------------------------------------------------------------
# Brute-force truncated-Fock evaluation of SMM17 Eq. (57)
# ---------------------------------------------------------------------------

def _fock_pieces(K, F_A, F_B, lam, eta, times, n_cut=22):
    """cosh/sinh of Y_A, Y_B and the field operators, in a truncated Fock space.

    Y_nu = -i lam_nu eta_nu Phi_nu(t_nu) [SMM17 Eqs. (21), (49)] with
    Phi_nu(t) = sum_i F_nu,i x_i(t) the smeared lattice field in the
    interaction picture,
    x_i(t) = sum_k U_ik (a_k e^{-i w_k t} + a_k^dag e^{i w_k t}) / sqrt(2 w_k).

    Deliberately independent of ``vacuum.detectors.nogo``: only the
    definition U_2 = exp(mu_B Y_B) exp(mu_A Y_A) [Eq. (47)] and
    m_nu^2 = 1 are used, never the paper's BCH algebra.
    """
    K = np.asarray(K, dtype=float)
    n_mode = K.shape[0]
    w2, U = np.linalg.eigh(K)
    w = np.sqrt(w2)
    a1 = np.diag(np.sqrt(np.arange(1, n_cut + 1)), 1)
    idn = np.eye(n_cut + 1)

    def kron_at(k):
        out = np.array([[1.0 + 0.0j]])
        for i in range(n_mode):
            out = np.kron(out, a1 if i == k else idn)
        return out

    A = [kron_at(k) for k in range(n_mode)]
    dim = (n_cut + 1) ** n_mode

    def Phi(F, t):
        out = np.zeros((dim, dim), dtype=complex)
        for k in range(n_mode):
            coef = float(np.dot(F, U[:, k])) / math.sqrt(2.0 * w[k])
            out = out + coef * (
                A[k] * np.exp(-1j * w[k] * t) + A[k].conj().T * np.exp(1j * w[k] * t)
            )
        return out

    lam_A, lam_B = lam
    eta_A, eta_B = eta
    t_A, t_B = times
    Phi_A = Phi(F_A, t_A)
    Phi_B = Phi(F_B, t_B)
    Y_A = -1j * lam_A * eta_A * Phi_A
    Y_B = -1j * lam_B * eta_B * Phi_B
    # Y is anti-Hermitian, so e^{-Y} = (e^{Y})^dag exactly -- one expm each.
    eA = expm(Y_A)
    eB = expm(Y_B)
    return {
        "cosh_A": 0.5 * (eA + eA.conj().T),
        "sinh_A": 0.5 * (eA - eA.conj().T),
        "cosh_B": 0.5 * (eB + eB.conj().T),
        "sinh_B": 0.5 * (eB - eB.conj().T),
        "Phi_A": Phi_A,
        "Phi_B": Phi_B,
        "A": A,
        "dim": dim,
    }


def _fock_rho(pieces, psi, gaps=(0.0, 0.0), times=(0.0, 0.0), tilde=True):
    """rho_AB from the definition, by tracing out a truncated Fock space.

    The four field operators multiplying the detector basis states are
    X_(++) = cosh Y_B cosh Y_A, X_(+-) = cosh Y_B sinh Y_A,
    X_(-+) = sinh Y_B cosh Y_A, X_(--) = sinh Y_B sinh Y_A -- which is
    SMM17 Eq. (51) rewritten, obtained here directly by applying
    exp(mu_B Y_B) exp(mu_A Y_A) to |g_A g_B> (mu_nu^2 = 1).

    ``tilde=True`` returns the state in the gap-free basis of Eq. (52);
    ``tilde=False`` keeps the bare {|g>, |e>} basis, where the gap phases
    e^{i Omega_nu t_nu} appear explicitly -- used to test that the spectrum
    is gap-independent.
    """
    X = {
        (+1, +1): pieces["cosh_B"] @ pieces["cosh_A"],
        (+1, -1): pieces["cosh_B"] @ pieces["sinh_A"],
        (-1, +1): pieces["sinh_B"] @ pieces["cosh_A"],
        (-1, -1): pieces["sinh_B"] @ pieces["sinh_A"],
    }
    order = ((+1, +1), (-1, +1), (+1, -1), (-1, -1))   # matches SIMIDZIJA_BASIS
    phase = {(+1, +1): 1.0, (-1, +1): 1.0, (+1, -1): 1.0, (-1, -1): 1.0}
    if not tilde:
        gA, gB = gaps
        tA, tB = times
        pa = np.exp(1j * gA * tA)
        pb = np.exp(1j * gB * tB)
        phase = {
            (+1, +1): 1.0,
            (-1, +1): pb,       # B excited
            (+1, -1): pa,       # A excited
            (-1, -1): pa * pb,
        }
    vecs = [phase[jk] * (X[jk] @ psi) for jk in order]
    return np.array([[np.vdot(v2, v1) for v2 in vecs] for v1 in vecs])


# ---------------------------------------------------------------------------
# 1. The closed form is the paper's definition
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fock_setup():
    """Two-mode lattice small enough for a dense Fock-space cross-check."""
    K = np.array([[2.2, -0.7], [-0.7, 1.8]])
    F_A = np.array([1.0, 0.2])
    F_B = np.array([0.1, 0.9])
    lam = (0.35, 0.28)
    eta = (0.9, 1.1)
    times = (0.0, 1.3)
    pieces = _fock_pieces(K, F_A, F_B, lam, eta, times, n_cut=22)
    return dict(K=K, F_A=F_A, F_B=F_B, lam=lam, eta=eta, times=times, pieces=pieces)


def _closed_form(setup, C=(0.0, 0.0)):
    ker = wightman_lattice(setup["K"])
    return delta_coupled_pair(
        ker, setup["F_A"], setup["F_B"], setup["lam"], setup["eta"],
        setup["times"], C,
    )


def test_closed_form_matches_fock_bruteforce_vacuum(fock_setup):
    """SMM17 Eq. (60) == the definition Eq. (57), vacuum field state."""
    dim = fock_setup["pieces"]["dim"]
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    rho_fock = _fock_rho(fock_setup["pieces"], psi)
    rho_closed = _closed_form(fock_setup).rho
    dev = float(np.max(np.abs(rho_fock - rho_closed)))
    assert dev < 1e-12, (
        f"delta-coupled rho_AB: closed form (SMM17 Eq. (60)) vs truncated-Fock "
        f"Eq. (57) differ by {dev:.3e}"
    )
    assert abs(np.trace(rho_fock).real - 1.0) < 1e-12


def test_closed_form_matches_fock_bruteforce_coherent(fock_setup):
    """Same, with the field in a coherent state -- and C_nu = -lam eta <Phi>."""
    pieces = fock_setup["pieces"]
    dim = pieces["dim"]
    vac = np.zeros(dim, dtype=complex)
    vac[0] = 1.0
    alphas = (0.6 + 0.3j, -0.4 + 0.5j)
    gen = sum(
        al * pieces["A"][k].conj().T - np.conj(al) * pieces["A"][k]
        for k, al in enumerate(alphas)
    )
    psi = expm(gen) @ vac                       # |alpha(k)> = D|0>, SMM17 Eq. (3)
    assert abs(np.vdot(psi, psi).real - 1.0) < 1e-10

    lam_A, lam_B = fock_setup["lam"]
    eta_A, eta_B = fock_setup["eta"]
    # SMM17 Eq. (32): C_nu = -lam_nu eta_nu <Phi_nu(t_nu)>, measured *in the
    # Fock space* -- nothing from nogo.py enters this number but the formula.
    C = (
        coherent_shift(lam_A, eta_A, np.vdot(psi, pieces["Phi_A"] @ psi).real),
        coherent_shift(lam_B, eta_B, np.vdot(psi, pieces["Phi_B"] @ psi).real),
    )
    rho_fock = _fock_rho(pieces, psi)
    rho_closed = _closed_form(fock_setup, C).rho
    dev = float(np.max(np.abs(rho_fock - rho_closed)))
    assert dev < 1e-12, (
        f"coherent-state rho_AB: closed form vs truncated-Fock differ by {dev:.3e} "
        f"(C_A, C_B = {C[0]:.6f}, {C[1]:.6f})"
    )


def test_delta_coupled_state_is_gap_independent(fock_setup):
    """SMM17 Sec. IV: with delta switching the spectrum cannot see Omega_nu."""
    pieces = fock_setup["pieces"]
    dim = pieces["dim"]
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    specs = []
    for gaps in [(0.0, 0.0), (1.7, -0.9), (13.0, 40.0)]:
        rho = _fock_rho(pieces, psi, gaps=gaps, times=fock_setup["times"], tilde=False)
        specs.append(np.sort(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))))
        pt = np.sort(np.linalg.eigvalsh(partial_transpose_A(rho)))
        specs.append(pt)
    ref_rho, ref_pt = specs[0], specs[1]
    worst = 0.0
    for i in range(0, len(specs), 2):
        worst = max(
            worst,
            float(np.max(np.abs(specs[i] - ref_rho))),
            float(np.max(np.abs(specs[i + 1] - ref_pt))),
        )
    assert worst < 1e-13, (
        f"delta-coupled spectra moved by {worst:.3e} when the detector gaps "
        "changed; SMM17 Sec. IV says they cannot"
    )


# ---------------------------------------------------------------------------
# 2. The Wightman identifications behind the closed form
# ---------------------------------------------------------------------------

def test_overlaps_match_mode_sums(chain, lattice_ker, profiles):
    """f_nu, omega, theta from M2.1 kernels == the SMM17 beta-integrals."""
    F_A, F_B = profiles
    lam = (0.4, 0.7)
    eta = (1.1, 0.8)
    t_A, t_B = 0.0, 1.4
    w2, U = np.linalg.eigh(chain)
    w = np.sqrt(w2)
    a = U.T @ F_A
    b = U.T @ F_B
    W_AA0 = float(np.sum(a * a / (2.0 * w)))
    W_BB0 = float(np.sum(b * b / (2.0 * w)))
    W_AB = complex(np.sum(a * b * np.exp(-1j * w * (t_A - t_B)) / (2.0 * w)))

    st = delta_coupled_pair(lattice_ker, F_A, F_B, lam, eta, (t_A, t_B))
    a_A, a_B, omega, theta = delta_coupled_overlaps(W_AA0, W_BB0, W_AB, lam, eta)
    for name, got, want in [
        ("a_A", st.a_A, a_A),
        ("a_B", st.a_B, a_B),
        ("omega", st.omega, omega),
        ("theta", st.theta, theta),
    ]:
        assert abs(got - want) <= 1e-14 * max(1.0, abs(want)), (
            f"{name}: engine {got!r} vs direct mode sum {want!r}"
        )
    # f_nu = exp(-2 lam^2 eta^2 W_nunu(0)) [Eqs. (37), (39)]
    assert abs(st.f_A - math.exp(-2.0 * lam[0] ** 2 * eta[0] ** 2 * W_AA0)) < 1e-15
    assert abs(st.f_B - math.exp(-2.0 * lam[1] ** 2 * eta[1] ** 2 * W_BB0)) < 1e-15


def test_continuum_coincidence_limit_closed_form(continuum_ker):
    """W(0) of the Gaussian-smeared massless 3+1 kernel is 1/(4 pi^2 sigma^2)."""
    auto, _, _ = wightman_triple(wightman_continuum_3p1(0.25, distance=0.0))
    got = float(np.real(auto(0.0)))
    want = 1.0 / (4.0 * math.pi**2 * 0.25**2)
    assert abs(got - want) <= 1e-12 * want, f"W_AA(0) = {got:.15e} vs {want:.15e}"


def test_cauchy_schwarz_is_the_perturbative_null(chain, lattice_ker, profiles):
    """|W_AB(dt)|^2 <= W_AA(0) W_BB(0): the O(lam^4) shadow of Theorem 3.

    Expanding SMM17 Eq. (87) to leading order turns "N > 0" into
    |W_AB|^2 > W_AA(0) W_BB(0), i.e. the harvesting condition |M| > P of
    the perturbative engine.  Cauchy-Schwarz on the two-point function
    forbids it -- which is exactly why the null is marginal at weak
    coupling and needs the stabilized evaluation.
    """
    F_A, F_B = profiles
    ker_AA, ker_BB, ker_AB = wightman_triple(lattice_ker, F_A, F_B)
    W_AA0 = float(np.real(ker_AA(0.0)))
    W_BB0 = float(np.real(ker_BB(0.0)))
    worst = -np.inf
    for dt in np.linspace(-8.0, 8.0, 81):
        lhs = abs(complex(ker_AB(dt))) ** 2
        worst = max(worst, lhs - W_AA0 * W_BB0)
    assert worst <= 1e-14 * W_AA0 * W_BB0, (
        f"Cauchy-Schwarz violated by {worst:.3e} on the smeared Wightman kernel"
    )


# ---------------------------------------------------------------------------
# 3. THE NULL: N = 0 to 1e-14 across seeded random draws
# ---------------------------------------------------------------------------

def _lattice_draws(rng, chain, lattice_ker, profiles, coherent, n=N_DRAWS):
    F_A, F_B = profiles
    for _ in range(n):
        lam = rng.uniform(0.02, 1.5, 2)
        eta = rng.uniform(0.05, 2.0, 2)
        times = rng.uniform(-4.0, 4.0, 2)
        if coherent:
            x0 = rng.normal(size=chain.shape[0]) * 0.7
            p0 = rng.normal(size=chain.shape[0]) * 0.7
            C = lattice_coherent_shifts(chain, (F_A, F_B), x0, p0, times, lam, eta)
        else:
            C = (0.0, 0.0)
        yield delta_coupled_pair(lattice_ker, F_A, F_B, lam, eta, times, C)


def test_simidzija_null_vacuum_lattice(chain, lattice_ker, profiles):
    """SMM17 Theorem 3, lattice backend, vacuum: N = 0 to 1e-14."""
    rng = np.random.default_rng(SEED)
    worst = 0.0
    worst_state = None
    for st in _lattice_draws(rng, chain, lattice_ker, profiles, coherent=False):
        if st.negativity > worst:
            worst, worst_state = st.negativity, st
    assert worst < NULL_TOL, (
        f"Simidzija null (vacuum, lattice): worst negativity {worst:.3e} over "
        f"{N_DRAWS} draws exceeds {NULL_TOL:.0e}; worst draw "
        f"(a_A, a_B, omega, theta) = "
        f"{None if worst_state is None else (worst_state.a_A, worst_state.a_B, worst_state.omega, worst_state.theta)}"
    )


def test_simidzija_null_coherent_states_lattice(chain, lattice_ker, profiles):
    """SMM17 Theorems 2-3: the same null for the paper's coherent-state class.

    The coherent shifts come from an actual displaced-vacuum (Gaussian)
    field state -- a random classical solution of the chain, propagated to
    each detector's switching instant -- not from arbitrary numbers.
    """
    rng = np.random.default_rng(SEED + 1)
    worst = 0.0
    worst_C = None
    for st in _lattice_draws(rng, chain, lattice_ker, profiles, coherent=True):
        if st.negativity > worst:
            worst, worst_C = st.negativity, (st.C_A, st.C_B)
    assert worst < NULL_TOL, (
        f"Simidzija null (coherent states, lattice): worst negativity "
        f"{worst:.3e} over {N_DRAWS} draws exceeds {NULL_TOL:.0e} "
        f"(worst C_A, C_B = {worst_C})"
    )


def test_simidzija_null_continuum():
    """SMM17 Theorem 3, continuum 3+1 backend, vacuum and coherent."""
    rng = np.random.default_rng(SEED + 2)
    worst = 0.0
    for _ in range(N_DRAWS):
        sigma = rng.uniform(0.15, 0.8)
        d = rng.uniform(0.3, 6.0)
        ker = wightman_continuum_3p1(sigma, distance=d)
        lam = rng.uniform(0.02, 1.2, 2)
        eta = rng.uniform(0.05, 1.5, 2)
        times = rng.uniform(-4.0, 4.0, 2)
        C = rng.normal(size=2) * 1.5
        st = delta_coupled_pair(ker, lam=lam, eta=eta, times=times, coherent=C)
        worst = max(worst, st.negativity)
    assert worst < NULL_TOL, (
        f"Simidzija null (continuum 3+1): worst negativity {worst:.3e} over "
        f"{N_DRAWS} draws exceeds {NULL_TOL:.0e}"
    )


def test_simidzija_null_deep_perturbative(chain, lattice_ker, profiles):
    """The marginal corner: lam down to 1e-8, where E_2 is O(lam^4).

    This is where a naive transcription of Eqs. (86)-(89) fails (the
    radicand goes negative in float64); the stabilized evaluation must hold
    the null, and the mpmath escalation must return an exact zero.
    """
    F_A, F_B = profiles
    rng = np.random.default_rng(SEED + 3)
    worst = 0.0
    worst_args = None
    for scale in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8):
        for _ in range(40):
            lam = np.array([scale, scale * rng.uniform(0.5, 2.0)])
            eta = rng.uniform(0.2, 2.0, 2)
            times = rng.uniform(-4.0, 4.0, 2)
            st = delta_coupled_pair(lattice_ker, F_A, F_B, lam, eta, times)
            if st.negativity > worst:
                worst = st.negativity
                worst_args = (st.a_A, st.a_B, st.omega, st.theta)
    assert worst < NULL_TOL, (
        f"Simidzija null (deep perturbative): worst negativity {worst:.3e} "
        f"exceeds {NULL_TOL:.0e} at (a_A, a_B, omega, theta) = {worst_args}"
    )
    if worst_args is not None:
        exact = delta_coupled_negativity(*worst_args, dps=80)
        assert exact == 0.0, (
            f"mpmath(dps=80) re-evaluation of the worst draw gives N = {exact:.3e}, "
            "not exactly zero"
        )


def test_pt_eigenvalues_match_numerical_diagonalization(chain, lattice_ker, profiles):
    """Analytic Eqs. (86)-(89) == eigvalsh of the partial transpose of Eq. (60)."""
    rng = np.random.default_rng(SEED + 4)
    worst_pt = 0.0
    worst_rho = 0.0
    worst_herm = 0.0
    for st in _lattice_draws(rng, chain, lattice_ker, profiles, coherent=True, n=60):
        rho = st.rho
        worst_herm = max(worst_herm, float(np.max(np.abs(rho - rho.conj().T))))
        assert abs(np.trace(rho).real - 1.0) < 1e-13
        analytic = np.sort(np.asarray(st.pt_eigenvalues))
        num_pt = np.sort(np.linalg.eigvalsh(partial_transpose_A(rho)))
        num_rho = np.sort(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)))
        worst_pt = max(worst_pt, float(np.max(np.abs(analytic - num_pt))))
        # SMM17: Eqs. (86)-(89) are invariant under omega -> -omega, hence they
        # are also the spectrum of rho_AB itself -- which is why every one of
        # them is non-negative.
        worst_rho = max(worst_rho, float(np.max(np.abs(analytic - num_rho))))
    assert worst_herm < 1e-13, f"rho_AB non-Hermitian by {worst_herm:.3e}"
    assert worst_pt < 1e-13, (
        f"analytic PT eigenvalues vs numerics differ by {worst_pt:.3e}"
    )
    assert worst_rho < 1e-13, (
        f"spec(rho_AB) != spec(rho_AB^{{t_A}}) by {worst_rho:.3e}"
    )


def test_theorem_2_spectra_independent_of_coherent_amplitude(lattice_ker, profiles):
    """SMM17 Theorem 2: C_A, C_B move rho_AB but never any of its spectra."""
    F_A, F_B = profiles
    rng = np.random.default_rng(SEED + 5)
    lam, eta, times = (0.6, 0.45), (1.3, 0.9), (0.0, 2.1)
    ref = delta_coupled_pair(lattice_ker, F_A, F_B, lam, eta, times, (0.0, 0.0))
    ref_rho = np.sort(np.linalg.eigvalsh(0.5 * (ref.rho + ref.rho.conj().T)))
    ref_pt = np.sort(np.linalg.eigvalsh(partial_transpose_A(ref.rho)))
    worst_spec = 0.0
    moved = 0.0
    for _ in range(25):
        C = tuple(rng.normal(size=2) * 2.0)
        st = delta_coupled_pair(lattice_ker, F_A, F_B, lam, eta, times, C)
        moved = max(moved, float(np.max(np.abs(st.rho - ref.rho))))
        worst_spec = max(
            worst_spec,
            float(np.max(np.abs(np.sort(np.linalg.eigvalsh(0.5 * (st.rho + st.rho.conj().T))) - ref_rho))),
            float(np.max(np.abs(np.sort(np.linalg.eigvalsh(partial_transpose_A(st.rho))) - ref_pt))),
        )
    assert moved > 1e-3, (
        f"the coherent amplitude barely moved rho_AB ({moved:.3e}); the "
        "Theorem 2 test would be vacuous"
    )
    assert worst_spec < NULL_TOL, (
        f"Theorem 2 violated: spectra moved by {worst_spec:.3e} under a change "
        f"of coherent amplitude (rho itself moved by {moved:.3e})"
    )


def test_theorem_1_single_detector(lattice_ker, profiles):
    """SMM17 Eqs. (40)-(42): one detector's spectrum is (1 +/- f_A)/2."""
    F_A, F_B = profiles
    st = delta_coupled_pair(lattice_ker, F_A, F_B, (0.5, 0.5), (1.0, 1.0), (0.0, 1.0))
    rng = np.random.default_rng(SEED + 6)
    want = np.sort(np.asarray(single_detector_eigenvalues(st.f_A)))
    worst = 0.0
    for _ in range(20):
        C = float(rng.normal() * 3.0)
        rho = single_detector_rho(st.f_A, C)
        assert np.max(np.abs(rho - rho.conj().T)) < 1e-15
        assert abs(np.trace(rho).real - 1.0) < 1e-15
        got = np.sort(np.linalg.eigvalsh(rho))
        worst = max(worst, float(np.max(np.abs(got - want))))
    assert worst < 1e-14, (
        f"Theorem 1 violated: single-detector spectrum moved by {worst:.3e}"
    )


def test_swapped_switching_order_is_a_relabelling(lattice_ker, profiles):
    """t_B < t_A: Eq. (47) puts B first; the physics must not notice."""
    F_A, F_B = profiles
    lam, eta = (0.55, 0.4), (1.2, 0.85)
    fwd = delta_coupled_pair(lattice_ker, F_A, F_B, lam, eta, (0.0, 1.7), (0.3, -0.6))
    bwd = delta_coupled_pair(
        lattice_ker, F_B, F_A, (lam[1], lam[0]), (eta[1], eta[0]), (1.7, 0.0), (-0.6, 0.3)
    )
    assert fwd.swapped is False and bwd.swapped is True
    # bwd is fwd with the two qubits relabelled: same spectra, and rho related
    # by the SWAP permutation.
    p = np.array([0, 2, 1, 3])
    dev = float(np.max(np.abs(bwd.rho[np.ix_(p, p)] - fwd.rho)))
    assert dev < 1e-14, f"relabelled rho differs by {dev:.3e}"
    assert bwd.negativity < NULL_TOL and fwd.negativity < NULL_TOL


def test_pt_eigenvalue_input_validation():
    with pytest.raises(ValueError):
        pt_eigenvalues(-0.1, 0.2, 0.0, 0.0)
    with pytest.raises(ValueError):
        delta_coupled_overlaps(-1.0, 1.0, 0.0)
    # a legitimate zero-coupling state is the pure |gg><gg| product state
    E = pt_eigenvalues(0.0, 0.0, 0.0, 0.0)
    assert abs(sorted(E)[-1] - 1.0) < 1e-15 and abs(min(E)) < 1e-15


def test_delta_coupled_rho_direct_entry_point():
    """delta_coupled_rho(f_A, f_B, omega, theta, C_A, C_B) is usable alone."""
    rho = delta_coupled_rho(0.9, 0.8, -0.05, 0.02, 0.3, -0.1)
    assert rho.shape == (4, 4)
    assert np.max(np.abs(rho - rho.conj().T)) < 1e-15
    assert abs(np.trace(rho).real - 1.0) < 1e-14


# ---------------------------------------------------------------------------
# 4. The communication split
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lattice_detectors(profiles):
    F_A, F_B = profiles
    chi = switching("cos2", 1.0, 0.0)
    return UDWDetector(chi, F_A), UDWDetector(chi, F_B)


def test_kernel_parts_reconstruct_the_wightman_function(lattice_ker, profiles,
                                                        continuum_ker):
    """commutator_kernel = i Im W, hadamard_kernel = Re W, and they sum to W."""
    F_A, F_B = profiles
    _, _, ker_AB = wightman_triple(lattice_ker, F_A, F_B)
    worst = 0.0
    for ker in (ker_AB, continuum_ker):
        comm = commutator_kernel(ker)
        had = hadamard_kernel(ker)
        for dt in np.linspace(-6.0, 6.0, 25):
            W = complex(ker(dt))
            c = complex(comm(dt))
            h = complex(had(dt))
            scale = max(1.0, abs(W))
            worst = max(
                worst,
                abs(c - 1j * W.imag) / scale,
                abs(h - W.real) / scale,
                abs(c + h - W) / scale,
            )
    assert worst < 1e-13, (
        f"W = Re W + i Im W decomposition (TMM21 Eq. (22)) off by {worst:.3e}"
    )


def test_commutator_kernel_is_the_pauli_jordan_function(chain, lattice_ker, profiles):
    """2 Im W_AB(dt) = F_A^T (S(dt) Omega)_xx F_B -- the causality-audit object.

    The commutator half of the Wightman kernel is state-independent
    spectral data; on the lattice it is literally the propagator the
    causality audit checks (``vacuum.core.dynamics.chain_propagator``).
    This ties the M2.5 split to Layer 0's causality machinery.
    """
    F_A, F_B = profiles
    N = chain.shape[0]
    Om = Omega(N)
    _, _, ker_AB = wightman_triple(lattice_ker, F_A, F_B)
    comm = commutator_kernel(ker_AB)
    worst = 0.0
    for dt in np.linspace(-5.0, 5.0, 21):
        S = chain_propagator(chain, dt)
        pj = float(F_A @ (S @ Om)[:N, :N] @ F_B)      # [x_i(dt), x_j(0)] = i (S Omega)_ij
        got = 2.0 * complex(comm(dt)).imag             # comm(dt) = i Im W
        worst = max(worst, abs(got - pj))
    assert worst < 1e-12, (
        f"commutator kernel vs Pauli-Jordan propagator differ by {worst:.3e}"
    )


def test_split_is_an_exact_decomposition(lattice_ker, lattice_detectors):
    """M = M_vac + M_comm identically (the split is a rewrite, not a fit)."""
    det_A, det_B = lattice_detectors
    sp = communication_split(lattice_ker, det_A, det_B, 0.05, 1.0)
    dev = abs((sp.M_vac + sp.M_comm) - sp.M)
    assert dev <= 1e-15 * max(1.0, abs(sp.M)), (
        f"M_vac + M_comm - M = {dev:.3e} (|M| = {abs(sp.M):.3e})"
    )
    assert 0.0 <= sp.comm_fraction
    mv, mc = sp                     # spec signature: -> (M_vac, M_comm)
    assert mv == sp.M_vac and mc == sp.M_comm


def test_hadamard_route_agrees_with_the_subtraction(lattice_ker, lattice_detectors):
    """Integrating Re W directly reproduces M - M_comm (TMM21 Eqs. (31)-(33))."""
    det_A, det_B = lattice_detectors
    sp = communication_split(
        lattice_ker, det_A, det_B, 0.05, 1.0, rtol=1e-11, cross_check=True
    )
    resid = sp.meta["cross_check"]
    assert resid <= 1e-9 * max(abs(sp.M), 1e-30), (
        f"direct Hadamard integral differs from M - M_comm by {resid:.3e} "
        f"(|M| = {abs(sp.M):.3e})"
    )


def test_commutator_part_is_state_independent(chain, lattice_ker, lattice_detectors,
                                               profiles):
    """TMM21's crux: C^- does not see the field state; C^+ does.

    The thermal lattice Wightman kernel is built here by hand,
    W_T(dt) = sum_k c_k [(n_k + 1) e^{-i w_k dt} + n_k e^{+i w_k dt}] with
    n_k = 1/(e^{w_k/T} - 1), which is an exponential sum on the doubled
    frequency set and therefore drops into the M2.3 engine unchanged.
    """
    from vacuum.detectors.kernels import ExpSumKernel

    F_A, F_B = profiles
    det_A, det_B = lattice_detectors
    w2, U = np.linalg.eigh(chain)
    w = np.sqrt(w2)
    T = 0.8
    n = 1.0 / np.expm1(w / T)

    def thermal(Fx, Fy):
        c = (U.T @ Fx) * (U.T @ Fy) / (2.0 * w)
        return ExpSumKernel(
            np.concatenate([w, -w]),
            np.concatenate([c * (n + 1.0), c * n]).astype(complex),
        )

    triple = (thermal(F_A, F_A), thermal(F_B, F_B), thermal(F_A, F_B))
    vac = communication_split(lattice_ker, det_A, det_B, 0.05, 1.0, rtol=1e-11)
    thm = communication_split(triple, det_A, det_B, 0.05, 1.0, rtol=1e-11)

    d_comm = abs(thm.M_comm - vac.M_comm)
    d_vac = abs(thm.M_vac - vac.M_vac)
    assert d_comm <= 1e-12 * max(1.0, abs(vac.M)), (
        f"M_comm moved by {d_comm:.3e} between vacuum and T = {T} -- the "
        "commutator contribution must be state-independent (TMM21 Eq. (24))"
    )
    assert d_vac > 1e3 * max(d_comm, 1e-18), (
        f"M_vac barely moved ({d_vac:.3e}) between vacuum and T = {T}; the "
        "state-dependence test is vacuous"
    )


def test_spacelike_communication_vanishes(continuum_ker):
    """ANCHOR: |M_comm| < 1e-12 for strictly spacelike compact switchings.

    Continuum 3+1, Gaussian smearing sigma = 0.25, detector separation
    d = 4.  Both switchings are ``cos2`` of half-width 0.5 centred at
    t = 0, so every pair (t, t') in the double integral obeys
    |t - t'| <= 1 < 4 = d: the whole interaction region is spacelike, up
    to the Gaussian smearing tail, whose leak is O(e^{-(d - 1)^2/2 sigma^2})
    = O(1e-31).  This doubles as a physics-level causality audit of the
    whole M2.1 + M2.3 stack.
    """
    chi = switching("cos2", 0.5, 0.0)
    det_A = UDWDetector(chi)
    det_B = UDWDetector(chi)
    sp = communication_split(
        continuum_ker, det_A, det_B, 1.0, 1.0, comm_atol=1e-16
    )
    assert abs(sp.M) > 1e-8, (
        f"|M| = {abs(sp.M):.3e} is too small for the spacelike test to have "
        "any content"
    )
    assert abs(sp.M_comm) < SPACELIKE_TOL, (
        f"spacelike |M_comm| = {abs(sp.M_comm):.3e} exceeds {SPACELIKE_TOL:.0e} "
        f"(|M| = {abs(sp.M):.3e}, |M_comm|/|M| = {sp.comm_fraction:.3e})"
    )
    # and the whole pair term is then genuine harvesting
    assert abs(sp.M_vac - sp.M) < SPACELIKE_TOL
    est = communication_estimator(1e-3, 1e-3, sp.M_vac, sp.M_comm)
    assert est["N_comm"] == 0.0


def test_causally_connected_communication_dominates():
    """TMM21's headline, qualitatively: at light contact M^- takes over.

    Continuum 3+1, sigma = 0.3, separation d = 1.5, compact ``cos2``
    switchings of half-width 0.4 at t_A = 0 and t_B = 1.5 -- detector B
    switches exactly on detector A's light cone.  TMM21 Sec. V report the
    estimator approaching 1 there (massless field, 3+1, strong Huygens):
    essentially all of the bipartite correlation is signalling, none of it
    harvested.  The contrast with the spacelike configuration above
    (|M_comm|/|M| ~ 4e-16) is the whole point of the split.
    """
    ker = wightman_continuum_3p1(0.3, distance=1.5)
    chi_A = switching("cos2", 0.4, 0.0)
    chi_B = switching("cos2", 0.4, 1.5)
    sp = communication_split(
        ker, UDWDetector(chi_A), UDWDetector(chi_B), 1.0, 1.0, rtol=1e-6
    )
    assert sp.comm_fraction > 0.9, (
        f"|M_comm|/|M| = {sp.comm_fraction:.4f} at light contact; TMM21 report "
        "the commutator part dominating there"
    )
    assert sp.vac_fraction < 0.1, (
        f"|M_vac|/|M| = {sp.vac_fraction:.4f} at light contact"
    )


def test_results_row_carries_the_m25_columns(lattice_ker, lattice_detectors):
    """The permanent schema: (E_N, M_vac, M_comm, |M_comm|/|M|) on every row."""
    from vacuum.detectors.udw import udw_pair_state

    det_A, det_B = lattice_detectors
    plain = udw_pair_state(lattice_ker, det_A, det_B, 0.05, 1.0)
    row = plain.row
    for col in ("E_N", "M_vac", "M_comm", "comm_fraction"):
        assert col in row, f"results row is missing the M2.5 column {col!r}"
    assert plain.comm is None
    assert row["M_vac"] is None and row["M_comm"] is None
    assert row["comm_fraction"] is None
    assert row["E_N"] == pytest.approx(plain.log_negativity, abs=0.0, rel=0.0)

    full = pair_state_with_split(lattice_ker, det_A, det_B, 0.05, 1.0)
    row = full.row
    assert full.comm is not None
    assert row["M_vac"] is not None and row["M_comm"] is not None
    assert row["comm_fraction"] == pytest.approx(full.comm.comm_fraction)
    # the split must not perturb the state it decorates
    assert full.M == plain.M
    assert abs(row["M_vac"] + row["M_comm"] - row["M"]) < 1e-18
    cols = communication_columns(full)
    assert set(cols) == {"E_N", "M_vac", "M_comm", "comm_fraction"}
    assert communication_columns(plain)["M_comm"] is None


def test_communication_estimator_reduces_to_tjoa_form(lattice_ker, lattice_detectors):
    """N^{+/-} = max(0, |M^{+/-}| - L_jj) for identical detectors [TMM21 Eq. (39)]."""
    det_A, det_B = lattice_detectors
    st = pair_state_with_split(lattice_ker, det_A, det_B, 0.05, 1.0)
    P = 0.5 * (st.P_A + st.P_B)
    est = communication_estimator(P, P, st.comm.M_vac, st.comm.M_comm)
    assert est["N"] == pytest.approx(max(0.0, abs(st.comm.M) - P), abs=1e-18)
    assert est["N_vac"] == pytest.approx(max(0.0, abs(st.comm.M_vac) - P), abs=1e-18)
    assert est["N_comm"] == pytest.approx(max(0.0, abs(st.comm.M_comm) - P), abs=1e-18)
    assert est["estimator"] == (est["N_comm"] / est["N"] if est["N"] > 0 else 0.0)
    # cross-check the total against the engine's own negativity
    assert pair_negativity(st.P_A, st.P_B, st.comm.M) == pytest.approx(
        st.negativity, abs=1e-18
    )


def test_commutator_kernel_rejects_bad_part():
    from vacuum.detectors.communication import WightmanPart

    _, _, ker = wightman_triple(wightman_continuum_3p1(0.4, distance=1.0))
    with pytest.raises(ValueError):
        WightmanPart(ker, "anticommutator")


def test_commutator_term_matches_split(lattice_ker, lattice_detectors, profiles):
    """The standalone M^- entry point agrees with the packaged split."""
    from vacuum.detectors.communication import commutator_term

    F_A, F_B = profiles
    det_A, det_B = lattice_detectors
    _, _, ker_AB = wightman_triple(lattice_ker, F_A, F_B)
    direct = commutator_term(
        ker_AB, det_A.chi, det_B.chi, 1.0, 1.0, 0.05, 0.05, atol=1e-16
    )
    sp = communication_split(
        lattice_ker, det_A, det_B, 0.05, 1.0, comm_atol=1e-16
    )
    assert abs(direct - sp.M_comm) <= 1e-15 * max(1.0, abs(sp.M)), (
        f"commutator_term {direct!r} vs communication_split {sp.M_comm!r}"
    )


def test_pair_term_is_linear_in_the_kernel(lattice_ker, lattice_detectors, profiles):
    """The split's whole justification: M is linear in the Wightman kernel."""
    from vacuum.detectors.kernels import ExpSumKernel

    F_A, F_B = profiles
    det_A, det_B = lattice_detectors
    _, _, ker = wightman_triple(lattice_ker, F_A, F_B)
    scaled = ExpSumKernel(ker.freqs, 3.5 * np.asarray(ker.amps))
    m1 = pair_term(ker, det_A.chi, det_B.chi, 1.0, 1.0, 0.05, 0.05)
    m2 = pair_term(scaled, det_A.chi, det_B.chi, 1.0, 1.0, 0.05, 0.05)
    assert abs(m2 - 3.5 * m1) <= 1e-14 * abs(m2), (
        f"pair_term is not linear in the kernel: {m2!r} vs {3.5 * m1!r}"
    )
