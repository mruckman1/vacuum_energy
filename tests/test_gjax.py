"""Layer-6 anchor: vacuum.opt.gjax mirrors vacuum.core and is differentiable.

Four groups (PLAN.md Layer 6, vacuum/opt/README.md validation anchors):

1. AGREEMENT — every mirrored function reproduces the numpy core to 1e-12
   (relative to max(1, |value|)) on the Layer-1 anchor inputs: the critical
   chain block entropies (tests/test_cft_anchor.py), the negativity-death
   curve (tests/test_negativity_death.py), the first-law delta S = delta <K>
   (tests/test_first_law.py), the ground energy Tr sqrt(K)/2, TMSV E_N = 2r,
   and the channel outputs.  Measured: <= 3e-15 on every well-conditioned
   quantity; negativity curve worst 9e-15; first-law dK 5e-14 relative.

   Critical-chain caveat (measured, not assumed): on the N = 200, m = 1e-6
   chain the reduced covariances have condition numbers ~1e10 and dozens
   of symplectic eigenvalues within 1e-14 of the vacuum floor, and float64
   itself — the numpy core included — resolves their block entropies only to
   ~1e-10 (numpy vs 25-digit mpmath: 2.4e-11 at l = 10, 1.6e-10 at l = 20,
   5.5e-10 at l = 40; numpy vs the scipy LAPACK build: up to 2e-9).  So for
   l <= 30 the two implementations are compared at 1e-12 outright (measured
   <= 9e-14), and for every block additionally against numpy's OWN
   sensitivity to a one-ulp input perturbation (the attainable float64
   accuracy on that input; JAX-vs-numpy is <= 0.2 of it everywhere).  The
   symplectic eigenvalues and the entropy map are each verified at 1e-12 on
   identical inputs, and the anchor itself (c = 1) is reproduced with the
   anchor's tolerance, |c_jax - c_np| ~ 1e-10.

2. GRADIENTS — jax.grad against Richardson-extrapolated central finite
   differences of the numpy core, relative error < 1e-6 (measured worst
   2.2e-9): d entropy / d K entries (periodic chain: degenerate K spectrum),
   d E_N / d squeezing (analytic 2), d <H> / d symplectic generator through
   expm, d E_N / d mass through ground_state_cov (N = 200 periodic,
   degenerate), through a quench (spectral and expm propagators), the fully
   degenerate symplectic spectrum V = nu0 I (analytic gradient), thermal
   states, the entanglement Hamiltonian, and channel parameters.

3. JIT — jitted calls equal eager calls to 1e-14 (measured 0 or 2e-16).

4. VACUUM FLOOR — no NaN gradients: entropy_from_nu at exactly nu = 1/2 has
   gradient exactly 0; entropy at V = I/2 and of the global ground state has
   finite gradient; a dead negativity has gradient exactly 0 (subgradient
   convention), its surrogate pt_min_symplectic_eigenvalue does not.

Conventions: hbar = 1, block quadrature ordering, nats (docs/API.md).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vacuum import core as C  # noqa: E402
from vacuum.modular import first_law_check  # noqa: E402
from vacuum.opt import gjax as G  # noqa: E402

AGREE = 1e-12      # implementation agreement, relative to max(1, |value|)
GRAD_TOL = 1e-6    # gradient vs finite differences, relative
JIT_TOL = 1e-14    # jitted vs eager


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def dev(a, b):
    """max |a - b| / max(1, max |b|)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(1.0, np.max(np.abs(b))))


def assert_agree(name, a, b, tol=AGREE):
    d = dev(a, b)
    assert d <= tol, f"{name}: jax vs numpy deviation {d:.3e} > {tol:.0e}"
    return d


def fd(f, h):
    """d f(e)/de at e = 0: central differences at h, h/2, Richardson O(h^4)."""
    c = lambda e: (f(e) - f(-e)) / (2.0 * e)
    return (4.0 * c(h / 2.0) - c(h)) / 3.0


def rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


def assert_grad(name, g, ref, tol=GRAD_TOL):
    r = rel(g, ref)
    assert r < tol, f"{name}: jax grad {float(g)!r} vs FD {float(ref)!r} (rel {r:.2e})"
    return r


def symdir(rs, n):
    D = rs.randn(n, n)
    D = 0.5 * (D + D.T)
    return D / np.abs(D).max()


def tmsv_np(r):
    c, s = np.cosh(2 * r), np.sinh(2 * r)
    V = np.zeros((4, 4))
    V[:2, :2] = 0.5 * np.array([[c, s], [s, c]])
    V[2:, 2:] = 0.5 * np.array([[c, -s], [-s, c]])
    return V


def tmsv_jax(r):
    c, s = jnp.cosh(2 * r), jnp.sinh(2 * r)
    Vxx = 0.5 * jnp.array([[c, s], [s, c]])
    Vpp = 0.5 * jnp.array([[c, -s], [-s, c]])
    Z = jnp.zeros((2, 2))
    return jnp.block([[Vxx, Z], [Z, Vpp]])


# ----------------------------------------------------------------------------
# fixtures (module-scoped lattice builds)
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def massive_chain():
    K = C.harmonic_chain_K(10, m=0.7, bc="periodic")
    return K, C.ground_state_cov(K)


@pytest.fixture(scope="module")
def random_chain():
    rs = np.random.RandomState(1234)
    B = rs.randn(8, 8)
    K = B @ B.T + 0.5 * np.eye(8)
    return K, C.ground_state_cov(K)


@pytest.fixture(scope="module")
def critical_chain():
    """tests/test_cft_anchor.py parameters."""
    K = C.harmonic_chain_K(200, m=1e-6, bc="periodic")
    return K, C.ground_state_cov(K)


@pytest.fixture(scope="module")
def death_chain():
    """tests/test_negativity_death.py parameters."""
    K = C.harmonic_chain_K(200, m=3.0, bc="periodic")
    return K, C.ground_state_cov(K)


@pytest.fixture(scope="module")
def first_law_chain():
    """tests/test_first_law.py parameters."""
    return C.harmonic_chain_K(120, m=0.005, bc="periodic")


# ============================================================================
# 1. agreement with the numpy core
# ============================================================================

def test_float64_enabled():
    assert jax.config.jax_enable_x64, "gjax must enable float64"
    assert G.Omega(2).dtype == jnp.float64


def test_structural_functions_exact(massive_chain):
    K, V = massive_chain
    assert_agree("Omega", G.Omega(3), C.Omega(3), 0.0)
    for bc in ("periodic", "dirichlet"):
        assert_agree(f"harmonic_chain_K({bc})", G.harmonic_chain_K(7, 0.3, bc),
                     C.harmonic_chain_K(7, 0.3, bc), 0.0)
    assert_agree("harmonic_chain_K(N=1)", G.harmonic_chain_K(1, 0.3),
                 C.harmonic_chain_K(1, 0.3), 0.0)
    assert_agree("reduce", G.reduce(V, [1, 4, 7]), C.reduce(V, [1, 4, 7]), 0.0)
    assert_agree("partial_transpose", G.partial_transpose(V, [1, 4]),
                 C.partial_transpose(V, [1, 4]), 0.0)
    S = C.chain_propagator(K, 0.4)
    assert_agree("symplectic_inverse", G.symplectic_inverse(S), C.symplectic_inverse(S), 1e-14)
    with pytest.raises(IndexError):
        G.reduce(V, [10])
    with pytest.raises(ValueError):
        G.log_negativity(V, [0, 1], [1, 2])


def test_ground_state_and_energy_agree(massive_chain, random_chain):
    for name, (K, V) in (("massive", massive_chain), ("random", random_chain)):
        Vj = G.ground_state_cov(K)
        assert_agree(f"ground_state_cov[{name}]", Vj, V)
        E = G.mean_energy(Vj, K)
        E_exact = 0.5 * np.sum(np.sqrt(np.linalg.eigvalsh(K)))
        assert_agree(f"mean_energy[{name}] vs numpy", E, C.mean_energy(V, K))
        assert_agree(f"ground energy[{name}] vs Tr sqrt(K)/2", E, E_exact)
        N = K.shape[0]
        H = np.zeros((2 * N, 2 * N))
        H[:N, :N] = K
        H[N:, N:] = np.eye(N)
        assert_agree(f"mean_energy[{name}] (2N form)", G.mean_energy(V, H), C.mean_energy(V, H))
    with pytest.raises(ValueError):
        G.ground_state_cov(np.diag([1.0, 0.0, 4.0]))


def test_symplectic_spectrum_and_entropies_agree(massive_chain, random_chain):
    K, V = massive_chain
    nu = G.symplectic_eigenvalues(V)
    assert_agree("vacuum nu", nu, C.symplectic_eigenvalues(V))
    assert float(np.max(np.abs(np.asarray(nu) - 0.5))) < 1e-12
    for A in ([2, 3], [0, 1, 2, 3], [1, 5, 8], list(range(5))):
        VA = C.reduce(V, A)
        assert_agree(f"nu(reduce {A})", G.symplectic_eigenvalues(VA), C.symplectic_eigenvalues(VA))
        assert_agree(f"entropy(reduce {A})", G.entropy(VA), C.entropy(VA))
    Kr, Vr = random_chain
    assert_agree("entropy random region", G.entropy(G.reduce(Vr, [0, 3, 5])),
                 C.entropy(C.reduce(Vr, [0, 3, 5])))
    # purity: complementary partitions
    SA = float(G.entropy(G.reduce(V, [0, 1, 2, 3])))
    SB = float(G.entropy(G.reduce(V, [4, 5, 6, 7, 8, 9])))
    assert abs(SA - SB) < 1e-9, f"S_A = {SA!r} vs S_B = {SB!r}"
    # thermal single mode, analytic
    nbar = 1.7
    S = float(G.entropy((nbar + 0.5) * np.eye(2)))
    S_exact = (nbar + 1) * np.log(nbar + 1) - nbar * np.log(nbar)
    assert abs(S - S_exact) < 1e-12, f"thermal entropy {S!r} vs {S_exact!r}"
    # singular PSD forms are accepted, as in the core
    cs = np.array([2.5, 0.0, 0.75, 0.0])
    O = np.diag(np.concatenate([cs, cs]))
    assert_agree("singular PSD nu", G.symplectic_eigenvalues(O), C.symplectic_eigenvalues(O), 1e-14)
    with pytest.raises(ValueError):
        G.symplectic_eigenvalues(np.diag([1.0, -1.0, 1.0, 1.0]))


def test_tmsv_negativity_and_mutual_information():
    r = 0.8
    V = tmsv_np(r)
    EN = G.log_negativity(V, [0], [1])
    assert_agree("TMSV E_N vs numpy", EN, C.log_negativity(V, [0], [1]))
    assert_agree("TMSV E_N vs 2r", EN, 2 * r)
    MI = G.mutual_information(V, [0], [1])
    assert_agree("TMSV MI vs numpy", MI, C.mutual_information(V, [0], [1]))
    S1 = G.entropy(G.reduce(V, [0]))
    assert_agree("TMSV MI = 2 S_single", MI, 2 * float(S1), 1e-9)
    assert abs(float(G.entropy(V))) < 1e-9


def test_thermal_state_cov_agree(massive_chain):
    K, V = massive_chain
    assert_agree("thermal T=0", G.thermal_state_cov(K, 0.0), C.thermal_state_cov(K, 0.0))
    assert_agree("thermal T=0 = ground", G.thermal_state_cov(K, 0.0), V)
    for T in (0.3, 0.9, 2.5):
        Vt = G.thermal_state_cov(K, T)
        assert_agree(f"thermal T={T}", Vt, C.thermal_state_cov(K, T))
        w = np.sqrt(np.linalg.eigvalsh(K))
        E_bose = np.sum(0.5 * w / np.tanh(w / (2 * T)))
        assert_agree(f"thermal energy T={T}", G.mean_energy(Vt, K), E_bose)
    with pytest.raises(ValueError):
        G.thermal_state_cov(K, -1.0)


def test_dynamics_agree(massive_chain):
    K, V = massive_chain
    N = K.shape[0]
    H = np.zeros((2 * N, 2 * N))
    H[:N, :N] = K
    H[N:, N:] = np.eye(N)
    t = 0.7
    S = G.symplectic_from_quadratic(H, t)
    assert_agree("symplectic_from_quadratic", S, C.symplectic_from_quadratic(H, t))
    Om = C.Omega(N)
    assert_agree("S Omega S^T = Omega", np.asarray(S) @ Om @ np.asarray(S).T, Om, 1e-9)
    assert_agree("chain_propagator spectral", G.chain_propagator(K, t), C.chain_propagator(K, t))
    assert_agree("chain_propagator expm", G.chain_propagator(K, t, "expm"),
                 C.chain_propagator(K, t, "expm"))
    Vt = G.evolve(V, S)
    assert_agree("evolve", Vt, C.evolve(V, np.asarray(S)))
    assert_agree("ground state stationary", Vt, V, 1e-9)
    rs = np.random.RandomState(7)
    Hr = symdir(rs, 2 * N)
    assert_agree("expm random generator", G.symplectic_from_quadratic(Hr, 1.3),
                 C.symplectic_from_quadratic(Hr, 1.3))
    with pytest.raises(ValueError):
        G.chain_propagator(K, t, "bogus")


def test_channels_agree(massive_chain):
    K, V = massive_chain
    N = K.shape[0]
    assert_agree("loss", G.loss(V, [1, 3], 0.6, 0.3), C.loss(V, [1, 3], 0.6, 0.3), 1e-14)
    assert_agree("loss eta=1", G.loss(V, [1, 3], 1.0, 0.4), V, 1e-14)
    assert_agree("loss eta=0", G.loss(V, list(range(N)), 0.0, 0.9),
                 (0.9 + 0.5) * np.eye(2 * N), 1e-14)
    assert_agree("thermal_noise", G.thermal_noise(V, [0, 2], 0.7),
                 C.thermal_noise(V, [0, 2], 0.7), 1e-14)
    assert_agree("amplifier", G.amplifier(V, [2], 1.8, 0.2), C.amplifier(V, [2], 1.8, 0.2), 1e-14)
    for name, (Xj, Yj), (Xn, Yn) in (
        ("loss_XY", G.loss_XY(4, [0, 2], 0.6, 0.3), C.loss_XY(4, [0, 2], 0.6, 0.3)),
        ("thermal_noise_XY", G.thermal_noise_XY(4, [0, 1], 0.7), C.thermal_noise_XY(4, [0, 1], 0.7)),
        ("amplifier_XY", G.amplifier_XY(4, [2, 3], 1.8, 0.2), C.amplifier_XY(4, [2, 3], 1.8, 0.2)),
    ):
        assert_agree(f"{name} X", Xj, Xn, 1e-14)
        assert_agree(f"{name} Y", Yj, Yn, 1e-14)
        assert_agree(f"{name} apply", G.apply_channel(V[:8, :8], Xj, Yj),
                     C.apply_channel(V[:8, :8], Xn, Yn), 1e-14)
    out = G.amplifier(0.5 * np.eye(4), [0], 2.0, 0.0)
    assert abs(float(out[0, 0]) - 1.5) < 1e-12
    # concrete-value validation mirrors the core (skipped only under tracing)
    with pytest.raises(ValueError):
        G.loss(V, [0], 1.5)
    with pytest.raises(ValueError):
        G.amplifier(V, [0], 0.5)
    with pytest.raises(ValueError):
        G.thermal_noise(V, [0], -0.1)


def test_entanglement_hamiltonian_agree(massive_chain):
    K, V = massive_chain
    VA = C.reduce(V, [2, 3])
    Gj = G.entanglement_hamiltonian(VA)
    Gn = C.entanglement_hamiltonian(VA)
    assert_agree("G 2-site region", Gj, Gn)
    assert_agree("modular_energy", G.modular_energy(Gj, VA), C.modular_energy(Gn, VA))
    nbar = 0.8
    nu0 = nbar + 0.5
    g = np.log((nu0 + 0.5) / (nu0 - 0.5))
    assert_agree("G thermal mode", G.entanglement_hamiltonian(nu0 * np.eye(2)), g * np.eye(2))
    # random Williamson state: symplectic spectrum of G is g(nu)
    rs = np.random.RandomState(7)
    Sr = C.symplectic_from_quadratic(0.5 * symdir(rs, 12), 1.0)
    nur = 0.5 + 1.5 * rs.rand(6)
    Vr = Sr @ np.diag(np.concatenate([nur, nur])) @ Sr.T
    Gj = G.entanglement_hamiltonian(Vr)
    assert_agree("G random Williamson state", Gj, C.entanglement_hamiltonian(Vr))
    gs = np.sort(C.symplectic_eigenvalues(np.asarray(Gj)))
    assert_agree("symplectic spectrum of G = g(nu)", gs,
                 np.sort(np.log((nur + 0.5) / (nur - 0.5))), 1e-12)
    with pytest.raises(ValueError):
        G.entanglement_hamiltonian(np.diag([1.0, 0.0, 1.0, 1.0]))


# --- anchor: critical chain block entropies (Calabrese-Cardy) ---------------

CFT_BLOCKS = np.arange(10, 101, 5)  # tests/test_cft_anchor.py FIT_BLOCKS


def test_critical_chain_block_entropies(critical_chain):
    _, V = critical_chain
    N = 200
    eps = np.finfo(float).eps
    rs = np.random.RandomState(0)
    S_jax, S_np = [], []
    worst_small, worst_ratio, worst_nu_rel, worst_map = 0.0, 0.0, 0.0, 0.0
    for l in CFT_BLOCKS:
        VA = C.reduce(V, range(l))
        nu_n = C.symplectic_eigenvalues(VA)
        nu_j = np.asarray(G.symplectic_eigenvalues(VA))
        s_n = C.entropy(VA)
        s_j = float(G.entropy(VA))
        S_jax.append(s_j)
        S_np.append(s_n)
        # (a) the scalar entropy map on identical spectra
        worst_map = max(worst_map, abs(float(G.entropy_from_nu(nu_n)) - C.entropy_from_nu(nu_n)))
        # (b) the spectra themselves, relative
        worst_nu_rel = max(worst_nu_rel, float(np.max(np.abs(nu_j - nu_n) / np.maximum(nu_n, 1.0))))
        # (c) attainable float64 accuracy: numpy's own sensitivity to a
        #     one-ulp (eps * max|V_A|) symmetric input perturbation
        scale = eps * np.abs(VA).max()
        sens = 0.0
        for _ in range(3):
            sens = max(sens, abs(C.entropy(VA + scale * symdir(rs, 2 * l)) - s_n))
        ratio = abs(s_j - s_n) / max(sens, AGREE)
        worst_ratio = max(worst_ratio, ratio)
        assert abs(s_j - s_n) <= max(AGREE, sens), (
            f"S({l}): jax {s_j!r} vs numpy {s_n!r}, |diff| = {abs(s_j - s_n):.2e} exceeds "
            f"numpy's own one-ulp sensitivity {sens:.2e}"
        )
        if l <= 30:
            worst_small = max(worst_small, abs(s_j - s_n))
            assert abs(s_j - s_n) <= AGREE, (
                f"S({l}): jax {s_j!r} vs numpy {s_n!r} (|diff| {abs(s_j - s_n):.2e} > 1e-12)"
            )
    assert worst_map <= AGREE, f"entropy_from_nu on identical nu: worst {worst_map:.2e}"
    assert worst_nu_rel <= 1e-10, f"symplectic eigenvalues: worst relative dev {worst_nu_rel:.2e}"
    assert worst_ratio <= 1.0
    # THE anchor, through the JAX route: c = 3 * slope vs the chord log
    S_jax, S_np = np.array(S_jax), np.array(S_np)
    chord = np.log((N / np.pi) * np.sin(np.pi * CFT_BLOCKS / N))
    c_jax = 3.0 * np.polyfit(chord, S_jax, 1)[0]
    c_np = 3.0 * np.polyfit(chord, S_np, 1)[0]
    assert abs(c_jax - 1.0) <= 0.05, f"central charge (jax) = {c_jax:.6f}"
    assert abs(c_jax - c_np) <= 1e-8, f"c_jax = {c_jax!r} vs c_np = {c_np!r}"
    print(
        f"\nCFT anchor via gjax: c = {c_jax:.7f} (numpy {c_np:.7f}, diff {c_jax - c_np:+.1e}); "
        f"|S_jax - S_np| <= {worst_small:.1e} for l <= 30, <= {worst_ratio:.2f} x numpy's "
        f"one-ulp sensitivity for all blocks; worst nu rel dev {worst_nu_rel:.1e}"
    )


# --- anchor: negativity death curve ------------------------------------------

def test_negativity_death_curve(death_chain):
    _, V = death_chain
    worst = 0.0
    curve = []
    for d in range(0, 31):
        A = list(range(10))
        B = list(range(10 + d, 20 + d))
        e_j = float(G.log_negativity(V, A, B))
        e_n = C.log_negativity(V, A, B)
        worst = max(worst, abs(e_j - e_n))
        curve.append(e_j)
        assert abs(e_j - e_n) <= AGREE, f"E_N(d={d}): jax {e_j!r} vs numpy {e_n!r}"
    assert curve[0] > 0.01, f"adjacent blocks: E_N(0) = {curve[0]:.6f}"
    assert curve[1] < curve[0]
    # beyond the exact-arithmetic death point (d = 14) float64 gives only
    # roundoff-level junk, in both implementations
    assert max(curve[14:]) < 1e-12, f"dead tail not dead: {max(curve[14:]):.3e}"
    print(f"\nnegativity-death curve: worst |E_jax - E_np| = {worst:.2e}; E_N(0) = {curve[0]:.6e}")


# --- anchor: entanglement first law -------------------------------------------

FL_REGION = list(range(48, 72))


def _first_law_jax(K, perturb, eps):
    """dS (entropy route) and d<K> (modular route) via gjax, Richardson as the core."""
    V_A = G.reduce(G.ground_state_cov(K), FL_REGION)
    Gm = G.entanglement_hamiltonian(V_A)

    def central(e):
        Vp = G.reduce(G.ground_state_cov(perturb(K, +e)), FL_REGION)
        Vm = G.reduce(G.ground_state_cov(perturb(K, -e)), FL_REGION)
        dS = (G.entropy(Vp) - G.entropy(Vm)) / (2.0 * e)
        dK = G.modular_energy(Gm, (Vp - Vm) / (2.0 * e))
        return float(dS), float(dK)

    e_c, e_f = eps, 0.5 * eps
    dS_c, dK_c = central(e_c)
    dS_f, dK_f = central(e_f)
    wc, wf = e_c ** 2, e_f ** 2
    return (wc * dS_f - wf * dS_c) / (wc - wf), (wc * dK_f - wf * dK_c) / (wc - wf)


def _site_bump(j):
    def perturb(K, e):
        Kp = np.array(K, copy=True)
        Kp[j, j] += e
        return Kp
    return perturb


@pytest.mark.parametrize(
    "name, perturb, eps",
    [
        ("global mass shift", lambda K, e: K + e * np.eye(120), 1e-7),
        ("site bump outside", _site_bump(28), 1e-5),
        ("site bump inside", _site_bump(60), 1e-5),
    ],
)
def test_first_law_anchor(first_law_chain, name, perturb, eps):
    K = first_law_chain
    dS, dK = _first_law_jax(K, perturb, eps)
    ref = first_law_check(K, FL_REGION, perturb, eps)
    rel_err = abs(dS - dK) / abs(dS)
    assert rel_err < 1e-3, f"[{name}] first law (jax): dS={dS!r} dK={dK!r} rel_err={rel_err:.3e}"
    assert rel_err < 1e-6, f"[{name}] achieved rel_err {rel_err:.3e} regressed far above ~1e-10"
    assert_agree(f"[{name}] d<K> jax vs numpy", dK, ref["dK_modular"])
    assert rel(dS, ref["dS"]) < 1e-9, f"[{name}] dS jax {dS!r} vs numpy {ref['dS']!r}"
    print(f"\nfirst law via gjax [{name}]: rel_err = {rel_err:.2e} (numpy {ref['rel_err']:.2e})")


def test_first_law_autodiff_matches_finite_differences(first_law_chain):
    """jax.grad straight through ground_state_cov -> reduce -> entropy on the
    floor-heavy 24-site interval (a dozen nu within 1e-14 of 1/2): the
    near-degenerate floor cluster limits autodiff to ~1e-8 relative here
    (measured 8e-9), well inside the 1e-6 gradient tolerance."""
    K = first_law_chain
    ref = first_law_check(K, FL_REGION, lambda K, e: K + e * np.eye(120), 1e-7)
    g = jax.grad(lambda e: G.entropy(G.reduce(G.ground_state_cov(K + e * jnp.eye(120)), FL_REGION)))(0.0)
    assert_grad("dS/d(m^2) through the first-law interval", g, ref["dS"])


# ============================================================================
# 2. gradients vs finite differences
# ============================================================================

def test_grad_entropy_wrt_coupling_matrix(massive_chain, random_chain):
    rs = np.random.RandomState(0)
    A = [2, 3]
    f_np = lambda K: C.entropy(C.reduce(C.ground_state_cov(K), A))
    f_j = lambda K: G.entropy(G.reduce(G.ground_state_cov(K), A))
    # periodic chain: K has a DEGENERATE spectrum (k <-> -k)
    K, _ = massive_chain
    N = K.shape[0]
    w = np.linalg.eigvalsh(K)
    assert np.min(np.diff(w)) < 1e-12, "fixture must have a degenerate K spectrum"
    g = np.asarray(jax.grad(f_j)(jnp.asarray(K)))
    assert np.all(np.isfinite(g))
    D = symdir(rs, N)
    assert_grad("dS/dK directional (degenerate periodic K)", np.sum(g * D),
                fd(lambda e: f_np(K + e * D), 1e-4))
    for (i, j) in [(0, 0), (2, 3), (0, 5)]:
        E = np.zeros((N, N))
        E[i, j] = E[j, i] = 1.0
        assert_grad(f"dS/dK_{i}{j}", np.sum(g * E), fd(lambda e: f_np(K + e * E), 1e-4))
    Kr, _ = random_chain
    g = np.asarray(jax.grad(f_j)(jnp.asarray(Kr)))
    D = symdir(rs, Kr.shape[0])
    assert_grad("dS/dK directional (random SPD)", np.sum(g * D), fd(lambda e: f_np(Kr + e * D), 1e-4))


def test_grad_log_negativity_wrt_squeezing():
    f = lambda r: G.log_negativity(tmsv_jax(r), [0], [1])
    g = jax.grad(f)(0.8)
    assert_grad("dE_N/dr (TMSV, analytic 2)", g, 2.0)
    assert_grad("dE_N/dr vs FD", g, fd(lambda e: C.log_negativity(tmsv_np(0.8 + e), [0], [1]), 1e-4))
    g0 = float(jax.grad(f)(0.0))
    assert np.isfinite(g0) and g0 == 0.0, f"kink at r = 0: gradient {g0!r} (convention: 0)"


def test_grad_mean_energy_wrt_symplectic_generator():
    rs = np.random.RandomState(11)
    N = 4
    K = C.harmonic_chain_K(N, 0.5, "dirichlet")
    V = C.ground_state_cov(K)
    H0 = symdir(rs, 2 * N)
    t = 0.7
    f_np = lambda H, t: C.mean_energy(C.evolve(V, C.symplectic_from_quadratic(H, t)), K)
    f_j = lambda H, t: G.mean_energy(G.evolve(V, G.symplectic_from_quadratic(H, t)), K)
    gH, gt = jax.grad(f_j, argnums=(0, 1))(jnp.asarray(H0), t)
    D = symdir(rs, 2 * N)
    assert_grad("d<H>/dH_mat directional (expm)", np.sum(np.asarray(gH) * D),
                fd(lambda e: f_np(H0 + e * D, t), 1e-4))
    assert_grad("d<H>/dt (expm)", gt, fd(lambda e: f_np(H0, t + e), 1e-4))


def test_grad_negativity_wrt_mass_through_ground_state():
    """N = 200 periodic chain (degenerate K), the negativity-death geometry."""
    N = 200
    A = list(range(10))
    for d in (0, 1):
        B = list(range(10 + d, 20 + d))
        f_np = lambda m: C.log_negativity(C.ground_state_cov(C.harmonic_chain_K(N, m)), A, B)
        f_j = lambda m: G.log_negativity(G.ground_state_cov(G.harmonic_chain_K(N, m)), A, B)
        g = jax.grad(f_j)(3.0)
        assert_grad(f"dE_N/dm through ground_state_cov (d={d})", g, fd(lambda e: f_np(3.0 + e), 1e-3))


def test_grad_through_time_evolution(massive_chain):
    """Quench: vacuum of K0 evolved with K1 (periodic, degenerate spectrum)."""
    rs = np.random.RandomState(5)
    K0, V0 = massive_chain
    N = K0.shape[0]
    A = [2, 3, 4]
    tq = 0.9

    def q_np(m1, t, method):
        S = C.chain_propagator(C.harmonic_chain_K(N, m1), t, method)
        return C.entropy(C.reduce(C.evolve(V0, S), A))

    def q_j(m1, t, method):
        S = G.chain_propagator(G.harmonic_chain_K(N, m1), t, method)
        return G.entropy(G.reduce(G.evolve(V0, S), A))

    for method in ("spectral", "expm"):
        gm, gt = jax.grad(q_j, argnums=(0, 1))(1.2, tq, method)
        assert_grad(f"quench dS/dm1 ({method})", gm, fd(lambda e: q_np(1.2 + e, tq, method), 1e-3))
        assert_grad(f"quench dS/dt ({method})", gt, fd(lambda e: q_np(1.2, tq + e, method), 1e-3))
    K1 = C.harmonic_chain_K(N, 1.2)
    D = symdir(rs, N)
    gK = jax.grad(lambda K: G.entropy(G.reduce(G.evolve(V0, G.chain_propagator(K, tq)), A)))(jnp.asarray(K1))
    assert_grad("quench dS/dK directional (spectral)", np.sum(np.asarray(gK) * D),
                fd(lambda e: C.entropy(C.reduce(C.evolve(V0, C.chain_propagator(K1 + e * D, tq)), A)), 1e-3))
    gE = jax.grad(lambda t: G.mean_energy(G.evolve(V0, G.chain_propagator(K1, t)), K0))(tq)
    assert_grad("quench d<H>/dt (spectral)", gE,
                fd(lambda e: C.mean_energy(C.evolve(V0, C.chain_propagator(K1, tq + e)), K0), 1e-3))


def test_grad_degenerate_symplectic_spectrum():
    """V = nu0 I: every symplectic eigenvalue equal; dS/dV = (1/2) ln((nu0+1/2)/(nu0-1/2)) I."""
    rs = np.random.RandomState(2)
    nu0 = 1.3
    V = nu0 * np.eye(6)
    g = np.asarray(jax.grad(G.entropy)(jnp.asarray(V)))
    ga = 0.5 * np.log((nu0 + 0.5) / (nu0 - 0.5)) * np.eye(6)
    assert np.max(np.abs(g - ga)) < 1e-12, f"analytic gradient at V = nu0 I: max dev {np.max(np.abs(g - ga)):.2e}"
    D = symdir(rs, 6)
    assert_grad("dS/dV directional at V = nu0 I", np.sum(g * D), fd(lambda e: C.entropy(V + e * D), 1e-4))
    Vp = np.diag([1.3, 1.3, 0.9, 1.3, 1.3, 0.9])  # two equal modes + one distinct
    g = np.asarray(jax.grad(G.entropy)(jnp.asarray(Vp)))
    assert_grad("dS/dV directional, partially degenerate", np.sum(g * D), fd(lambda e: C.entropy(Vp + e * D), 1e-4))


def test_grad_thermal_state():
    rs = np.random.RandomState(9)
    T = 0.8
    for bc in ("dirichlet", "periodic"):
        K = C.harmonic_chain_K(6, 0.4, bc)
        w = np.sqrt(np.linalg.eigvalsh(K))
        E_exact = lambda T: np.sum(0.5 * w / np.tanh(w / (2 * T)))
        gT = jax.grad(lambda T: G.mean_energy(G.thermal_state_cov(K, T), K))(T)
        assert_grad(f"thermal d<H>/dT ({bc})", gT, fd(lambda e: E_exact(T + e), 1e-4))
        D = symdir(rs, 6)
        gK = jax.grad(lambda K: G.entropy(G.reduce(G.thermal_state_cov(K, T), [1, 2])))(jnp.asarray(K))
        assert_grad(f"thermal dS_A/dK directional ({bc})", np.sum(np.asarray(gK) * D),
                    fd(lambda e: C.entropy(C.reduce(C.thermal_state_cov(K + e * D, T), [1, 2])), 1e-4))
    g0 = float(jax.grad(lambda T: G.mean_energy(G.thermal_state_cov(K, T), K))(0.0))
    assert g0 == 0.0, f"d<H>/dT at T = 0 must be 0 (masked coth limit), got {g0!r}"


def test_grad_entanglement_hamiltonian(massive_chain):
    rs = np.random.RandomState(3)
    K, V = massive_chain
    cases = [
        ("2-site region", C.reduce(V, [2, 3])),
        ("3-mode thermal region", C.reduce(C.thermal_state_cov(K, 0.5), [1, 2, 3])),
        ("degenerate nu", np.diag([1.3, 1.3, 1.3, 1.3])),
    ]
    for name, VA in cases:
        n = VA.shape[0]
        W = symdir(rs, n)
        D = symdir(rs, n)
        gV = jax.grad(lambda V: G.modular_energy(G.entanglement_hamiltonian(V), W))(jnp.asarray(VA))
        assert_grad(f"d Tr(G W)/dV directional ({name})", np.sum(np.asarray(gV) * D),
                    fd(lambda e: C.modular_energy(C.entanglement_hamiltonian(VA + e * D), W), 1e-4))


def test_grad_channels(massive_chain):
    K, V = massive_chain
    Vt = tmsv_np(0.8)
    ge = jax.grad(lambda eta: G.entropy(G.loss(V, [1, 3], eta, 0.3)))(0.6)
    assert_grad("dS/d eta (loss)", ge, fd(lambda e: C.entropy(C.loss(V, [1, 3], 0.6 + e, 0.3)), 1e-4))
    gg = jax.grad(lambda gain: G.log_negativity(G.amplifier(Vt, [0], gain, 0.1), [0], [1]))(1.5)
    assert_grad("dE_N/d gain (amplifier)", gg,
                fd(lambda e: C.log_negativity(C.amplifier(Vt, [0], 1.5 + e, 0.1), [0], [1]), 1e-4))
    gn = jax.grad(lambda nb: G.mutual_information(G.thermal_noise(Vt, [1], nb), [0], [1]))(0.4)
    assert_grad("dMI/d nbar (thermal noise)", gn,
                fd(lambda e: C.mutual_information(C.thermal_noise(Vt, [1], 0.4 + e), [0], [1]), 1e-4))


# ============================================================================
# 3. jit
# ============================================================================

def test_jit_matches_eager(massive_chain, death_chain):
    K, V = massive_chain
    N = K.shape[0]
    VA = C.reduce(V, [2, 3])
    Vt = tmsv_np(0.8)
    rs = np.random.RandomState(7)
    H0 = symdir(rs, 2 * N)
    _, V200 = death_chain
    checks = [
        ("entropy", jax.jit(G.entropy)(VA), G.entropy(VA)),
        ("log_negativity TMSV", jax.jit(G.log_negativity, static_argnums=(1, 2))(Vt, (0,), (1,)),
         G.log_negativity(Vt, [0], [1])),
        ("log_negativity chain blocks",
         jax.jit(G.log_negativity, static_argnums=(1, 2))(V200, tuple(range(10)), tuple(range(10, 20))),
         G.log_negativity(V200, list(range(10)), list(range(10, 20)))),
        ("mutual_information", jax.jit(G.mutual_information, static_argnums=(1, 2))(V, (0, 1), (5, 6)),
         G.mutual_information(V, [0, 1], [5, 6])),
        ("ground_state_cov", jax.jit(G.ground_state_cov)(K), G.ground_state_cov(K)),
        ("thermal_state_cov", jax.jit(G.thermal_state_cov)(K, 0.9), G.thermal_state_cov(K, 0.9)),
        ("symplectic_from_quadratic", jax.jit(G.symplectic_from_quadratic)(H0, 0.7),
         G.symplectic_from_quadratic(H0, 0.7)),
        ("chain_propagator", jax.jit(G.chain_propagator, static_argnums=2)(K, 0.7, "spectral"),
         G.chain_propagator(K, 0.7)),
        ("entanglement_hamiltonian", jax.jit(G.entanglement_hamiltonian)(VA), G.entanglement_hamiltonian(VA)),
        ("loss", jax.jit(G.loss, static_argnums=1)(V, (1, 3), 0.6, 0.3), G.loss(V, [1, 3], 0.6, 0.3)),
        ("amplifier", jax.jit(G.amplifier, static_argnums=1)(V, (2,), 1.8, 0.2), G.amplifier(V, [2], 1.8, 0.2)),
        ("mean_energy", jax.jit(G.mean_energy)(V, K), G.mean_energy(V, K)),
        ("harmonic_chain_K", jax.jit(G.harmonic_chain_K, static_argnums=(0, 2))(N, 0.7, "periodic"),
         G.harmonic_chain_K(N, 0.7)),
    ]
    for name, a, b in checks:
        d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        assert d <= JIT_TOL, f"jit vs eager [{name}]: {d:.2e}"
    # jit of a gradient through the whole stack
    f = lambda m: G.log_negativity(G.ground_state_cov(G.harmonic_chain_K(200, m)), list(range(10)), list(range(10, 20)))
    gj = float(jax.jit(jax.grad(f))(3.0))
    ge = float(jax.grad(f)(3.0))
    assert abs(gj - ge) <= JIT_TOL, f"jit(grad) vs grad: {abs(gj - ge):.2e}"


# ============================================================================
# 4. vacuum floor: finite gradients, never NaN
# ============================================================================

def test_no_nan_gradients_at_vacuum_floor():
    # exact floor input: gradient exactly zero
    g = np.asarray(jax.grad(G.entropy_from_nu)(jnp.array([0.5, 0.5, 0.5])))
    assert np.array_equal(g, np.zeros(3)), f"grad entropy_from_nu at nu = 1/2: {g}"
    assert float(G.entropy_from_nu(jnp.array([0.5, 0.5]))) == 0.0
    # the exact vacuum covariance (all nu = 1/2 up to one ulp of roundoff)
    g = np.asarray(jax.grad(G.entropy)(0.5 * jnp.eye(6)))
    assert np.all(np.isfinite(g)), "NaN/inf gradient of entropy at V = I/2"
    # the global ground state of a chain, differentiated through K: pure, so
    # dS/dK vanishes up to roundoff, and must be finite
    K = C.harmonic_chain_K(6, 0.4)
    g = np.asarray(jax.grad(lambda K: G.entropy(G.ground_state_cov(K)))(jnp.asarray(K)))
    assert np.all(np.isfinite(g)) and np.max(np.abs(g)) < 1e-12, f"global vacuum dS/dK: max |g| = {np.max(np.abs(g)):.2e}"
    gm = float(jax.grad(lambda m: G.entropy(G.ground_state_cov(G.harmonic_chain_K(6, m))))(0.4))
    assert np.isfinite(gm) and abs(gm) < 1e-12, f"global vacuum dS/dm = {gm!r}"
    # negativity at the vacuum and in the dead region: exactly 0 gradient
    g = np.asarray(jax.grad(lambda V: G.log_negativity(V, [0], [1]))(0.5 * jnp.eye(4)))
    assert np.all(np.isfinite(g)) and np.all(g == 0.0)
    A, B = [0, 1, 2], [15, 16, 17]
    fdead = lambda m: G.log_negativity(G.ground_state_cov(G.harmonic_chain_K(30, m)), A, B)
    assert float(fdead(3.0)) == 0.0, "fixture must be in the sudden-death region"
    gd = float(jax.grad(fdead)(3.0))
    assert gd == 0.0, f"dead-region E_N gradient must be exactly 0, got {gd!r}"
    # ... while the surrogate keeps a finite, nonzero gradient there
    gs = float(jax.grad(lambda m: G.pt_min_symplectic_eigenvalue(
        G.ground_state_cov(G.harmonic_chain_K(30, m)), A, B))(3.0))
    assert np.isfinite(gs) and gs != 0.0, f"pt_min surrogate gradient {gs!r}"
    # singular PSD form (nu = 0 branch point): finite by convention
    O = jnp.diag(jnp.array([2.5, 0.0, 0.75, 0.0, 2.5, 0.0, 0.75, 0.0]))
    g = np.asarray(jax.grad(lambda O: jnp.sum(G.symplectic_eigenvalues(O)))(O))
    assert np.all(np.isfinite(g))
