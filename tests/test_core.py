"""Unit anchors for vacuum.core (Layer 0/1).

Every assertion records the measured number in its message (API.md testing
policy). Conventions: hbar = 1, vacuum nu = 1/2, block quadrature ordering,
nats everywhere.
"""

import math

import numpy as np
import pytest

from vacuum.core import (
    Omega,
    amplifier,
    amplifier_XY,
    apply_channel,
    entanglement_hamiltonian,
    entropy,
    entropy_mp,
    entropy_precise,
    ground_state_cov,
    harmonic_chain_K,
    log_negativity,
    loss,
    loss_XY,
    mean_energy,
    modular_energy,
    mutual_information,
    near_vacuum_floor,
    partial_transpose,
    reduce,
    symplectic_eigenvalues,
    symplectic_eigenvalues_mp,
    symplectic_from_quadratic,
    thermal_noise_XY,
    williamson,
)


# ----------------------------------------------------------------------------
# module-scoped fixtures (cache the lattice builds)
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def random_chain():
    """Random symmetric positive-definite coupling matrix K and its vacuum."""
    rs = np.random.RandomState(1234)
    N = 8
    B = rs.randn(N, N)
    K = B @ B.T + 0.5 * np.eye(N)  # SPD, modest conditioning
    V = ground_state_cov(K)
    return K, V


@pytest.fixture(scope="module")
def massive_chain():
    """Periodic massive harmonic chain (Srednicki normalization) and vacuum."""
    N = 10
    K = harmonic_chain_K(N, m=0.7, bc="periodic")
    V = ground_state_cov(K)
    return K, V


def two_mode_squeezed(r):
    """Explicit TMSV covariance in block ordering (x1, x2, p1, p2).

    V_xx = (1/2)[[c, s], [s, c]], V_pp = (1/2)[[c, -s], [-s, c]],
    c = cosh 2r, s = sinh 2r; PT symplectic eigenvalue e^{-2r}/2 gives
    E_N = 2r in nats.
    """
    c, s = np.cosh(2 * r), np.sinh(2 * r)
    V = np.zeros((4, 4))
    V[:2, :2] = 0.5 * np.array([[c, s], [s, c]])
    V[2:, 2:] = 0.5 * np.array([[c, -s], [-s, c]])
    return V


# ----------------------------------------------------------------------------
# 1. vacuum of a random chain: nu = 1/2, entropy = 0
# ----------------------------------------------------------------------------

def test_vacuum_symplectic_spectrum_and_entropy(random_chain):
    K, V = random_chain
    nu = symplectic_eigenvalues(V)
    dev = np.max(np.abs(nu - 0.5))
    assert dev < 1e-12, f"vacuum nu deviates from 1/2 by {dev:.3e}"
    S = entropy(V)
    assert abs(S) < 1e-12, f"vacuum entropy = {S:.3e} nats (expected 0)"


# ----------------------------------------------------------------------------
# 2. single-mode thermal entropy
# ----------------------------------------------------------------------------

def test_thermal_entropy():
    nbar = 1.7
    V = (nbar + 0.5) * np.eye(2)
    S = entropy(V)
    S_exact = (nbar + 1) * np.log(nbar + 1) - nbar * np.log(nbar)
    assert abs(S - S_exact) < 1e-12, (
        f"thermal entropy {S!r} vs exact {S_exact!r} (nbar={nbar})"
    )


# ----------------------------------------------------------------------------
# 3. two-mode squeezed state: E_N = 2r, MI = 2 S_single (pure state)
# ----------------------------------------------------------------------------

def test_tmsv_log_negativity_and_mutual_information():
    r = 0.8
    V = two_mode_squeezed(r)

    EN = log_negativity(V, [0], [1])
    assert abs(EN - 2 * r) < 1e-9, f"E_N = {EN!r} nats, expected 2r = {2 * r}"

    S1 = entropy(reduce(V, [0]))
    MI = mutual_information(V, [0], [1])
    assert abs(MI - 2 * S1) < 1e-9, (
        f"MI = {MI!r} vs 2 * S_single = {2 * S1!r} (pure TMSV)"
    )
    # sanity: the global state is pure
    Sg = entropy(V)
    assert abs(Sg) < 1e-9, f"global TMSV entropy = {Sg:.3e} (expected 0)"


# ----------------------------------------------------------------------------
# 4. Williamson decomposition of a random 6-mode physical state
# ----------------------------------------------------------------------------

def test_williamson_random_physical_state():
    rs = np.random.RandomState(7)
    N = 6
    H = rs.randn(2 * N, 2 * N)
    H = 0.25 * (H + H.T)
    S_r = symplectic_from_quadratic(H, 1.0)
    nu_r = 0.5 + 1.5 * rs.rand(N)
    V = S_r @ np.diag(np.concatenate([nu_r, nu_r])) @ S_r.T

    S, nu = williamson(V)
    Om = Omega(N)

    nu_err = np.max(np.abs(nu - np.sort(nu_r)[::-1]))
    assert nu_err < 1e-9, f"Williamson nu error {nu_err:.3e}"

    recon = S @ np.diag(np.concatenate([nu, nu])) @ S.T
    rec_err = np.max(np.abs(recon - V))
    assert rec_err < 1e-9, f"Williamson reconstruction error {rec_err:.3e}"

    symp_err = np.max(np.abs(S @ Om @ S.T - Om))
    assert symp_err < 1e-9, f"S Omega S^T - Omega deviation {symp_err:.3e}"


# ----------------------------------------------------------------------------
# 5. ground-state energy = Tr sqrt(K) / 2
# ----------------------------------------------------------------------------

def test_ground_state_energy(massive_chain):
    K, V = massive_chain
    E = mean_energy(V, K)
    w = np.linalg.eigvalsh(K)
    E_exact = 0.5 * np.sum(np.sqrt(w))  # Tr(sqrtm(K)) / 2
    assert abs(E - E_exact) < 1e-9, (
        f"ground energy {E!r} vs Tr(sqrt K)/2 = {E_exact!r}"
    )


# ----------------------------------------------------------------------------
# 6. purity: S_A = S_B for complementary partitions of the global vacuum
# ----------------------------------------------------------------------------

def test_complementary_entropies_equal(massive_chain):
    K, V = massive_chain
    N = K.shape[0]
    A = [0, 1, 2, 3]
    B = [m for m in range(N) if m not in A]
    SA = entropy(reduce(V, A))
    SB = entropy(reduce(V, B))
    assert abs(SA - SB) < 1e-9, (
        f"S_A = {SA!r} vs S_B = {SB!r} on a global pure state"
    )
    assert SA > 1e-3, f"S_A = {SA!r}: partition entropy suspiciously small"


# ----------------------------------------------------------------------------
# 7. channels: identity limit, full-loss limit, complete positivity
# ----------------------------------------------------------------------------

def test_loss_eta_one_is_identity(massive_chain):
    K, V = massive_chain
    out = loss(V, [1, 3], eta=1.0, nbar=0.4)
    dev = np.max(np.abs(out - V))
    assert dev < 1e-14, f"eta=1 loss changed V by {dev:.3e}"


def test_loss_eta_zero_yields_environment_state(massive_chain):
    K, V = massive_chain
    N = K.shape[0]
    nbar = 0.9
    out = loss(V, list(range(N)), eta=0.0, nbar=nbar)
    dev = np.max(np.abs(out - (nbar + 0.5) * np.eye(2 * N)))
    assert dev < 1e-14, (
        f"eta=0 loss on vacuum deviates from thermal env by {dev:.3e}"
    )


@pytest.mark.parametrize(
    "XY",
    [
        lambda N: loss_XY(N, [0, 2], eta=0.6, nbar=0.3),
        lambda N: loss_XY(N, [1], eta=0.05, nbar=0.0),
        lambda N: thermal_noise_XY(N, [0, 1, 2, 3], nbar_added=0.7),
        lambda N: amplifier_XY(N, [2, 3], gain=1.8, nbar=0.2),
        lambda N: amplifier_XY(N, [0], gain=3.0, nbar=0.0),
    ],
    ids=["loss", "deep-loss", "thermal", "amp", "strong-amp"],
)
def test_channel_complete_positivity(XY):
    N = 4
    X, Y = XY(N)
    Om = Omega(N)
    cp = Y.astype(complex) + 0.5j * (Om - X @ Om @ X.T)
    ev = np.linalg.eigvalsh(cp)
    assert np.min(ev) >= -1e-12, (
        f"CP matrix has negative eigenvalue {np.min(ev):.3e}"
    )
    # applying the channel to the vacuum must keep the state physical
    V = 0.5 * np.eye(2 * N)
    nu = symplectic_eigenvalues(apply_channel(V, X, Y))
    assert np.min(nu) >= 0.5 - 1e-12, (
        f"channel output unphysical: min nu = {np.min(nu)!r}"
    )


def test_amplifier_scales_variances():
    N = 2
    V = 0.5 * np.eye(2 * N)
    out = amplifier(V, [0], gain=2.0, nbar=0.0)
    # quantum-limited amp on vacuum: variance -> gain*1/2 + (gain-1)*1/2
    assert abs(out[0, 0] - 1.5) < 1e-12, f"amp output variance {out[0, 0]!r}"
    assert abs(out[1, 1] - 0.5) < 1e-12, "untouched mode was modified"


# ----------------------------------------------------------------------------
# 8. entanglement Hamiltonian: thermal mode, and the first law dS = d<K>
# ----------------------------------------------------------------------------

def test_entanglement_hamiltonian_thermal_mode():
    nbar = 0.8
    nu0 = nbar + 0.5
    V = nu0 * np.eye(2)
    G = entanglement_hamiltonian(V)
    g = np.log((nu0 + 0.5) / (nu0 - 0.5))
    dev = np.max(np.abs(G - g * np.eye(2)))
    assert dev < 1e-9, f"thermal-mode G deviates from g(nu) I by {dev:.3e}"


def test_modular_energy_first_law(massive_chain):
    # First law dS = d<K> = (1/2) Tr(G dV) on a two-site region whose reduced
    # spectrum sits away from the vacuum floor (min nu - 1/2 ~ 1e-2); larger
    # regions have nu - 1/2 ~ 1e-5 where the entropy's higher derivatives
    # (~ 1/(nu-1/2)^2) defeat any float64 finite difference — that regime is
    # exactly what vacuum.core.precision escalates for.
    K, V = massive_chain
    A = [2, 3]
    VA = reduce(V, A)
    G = entanglement_hamiltonian(VA)

    rs = np.random.RandomState(99)
    dV = rs.randn(2 * len(A), 2 * len(A))
    dV = 0.5 * (dV + dV.T)
    dV /= np.max(np.abs(dV))

    def central(eps):
        return (entropy(VA + eps * dV) - entropy(VA - eps * dV)) / (2 * eps)

    eps = 2e-5
    dS_num = (4.0 * central(eps / 2) - central(eps)) / 3.0  # Richardson, O(eps^4)
    dK = modular_energy(G, dV)  # (1/2) Tr(G dV)
    assert abs(dS_num - dK) < 1e-8 * max(1.0, abs(dK)), (
        f"first law: dS = {dS_num!r} vs d<K> = {dK!r}"
    )


# ----------------------------------------------------------------------------
# 9. ground state is stationary under its own quadratic Hamiltonian
# ----------------------------------------------------------------------------

def test_ground_state_stationary(massive_chain):
    K, V = massive_chain
    N = K.shape[0]
    H_mat = np.zeros((2 * N, 2 * N))
    H_mat[:N, :N] = K
    H_mat[N:, N:] = np.eye(N)
    S = symplectic_from_quadratic(H_mat, t=0.7)

    Om = Omega(N)
    symp_err = np.max(np.abs(S @ Om @ S.T - Om))
    assert symp_err < 1e-9, f"exp(t Omega H) not symplectic: {symp_err:.3e}"

    Vt = S @ V @ S.T
    dev = np.max(np.abs(Vt - V))
    assert dev < 1e-9, f"ground state moved under its own H by {dev:.3e}"


# ----------------------------------------------------------------------------
# precision ladder
# ----------------------------------------------------------------------------

def test_precision_guard_and_mp(random_chain):
    K, V = random_chain
    nu = symplectic_eigenvalues(V)
    assert near_vacuum_floor(nu), "guard must trip on the vacuum"
    assert not near_vacuum_floor(np.array([0.5 + 1e-3, 1.2])), (
        "guard must not trip away from the floor"
    )

    # small system through the full mpmath route
    Ks = harmonic_chain_K(4, m=1.0, bc="dirichlet")
    Vs = ground_state_cov(Ks)
    nu_mp = symplectic_eigenvalues_mp(Vs, dps=40)
    dev = np.max(np.abs(nu_mp - 0.5))
    assert dev < 1e-14, f"mpmath vacuum nu deviates by {dev:.3e}"
    S_mp = entropy_mp(Vs, dps=40)
    assert abs(S_mp) < 1e-14, f"mpmath vacuum entropy = {S_mp:.3e}"
    S_prec = entropy_precise(Vs)
    assert abs(S_prec) < 1e-14, f"escalated vacuum entropy = {S_prec:.3e}"

    # thermal state: mpmath agrees with float64 route away from the floor
    nbar = 0.6
    Vt = (nbar + 0.5) * np.eye(4)
    S64 = entropy(Vt)
    Smp = entropy_mp(Vt, dps=40)
    assert abs(S64 - Smp) < 1e-12, f"mp vs float64 entropy: {S64!r} vs {Smp!r}"


# ----------------------------------------------------------------------------
# structural sanity: partial transpose, Omega
# ----------------------------------------------------------------------------

def test_omega_and_partial_transpose_structure():
    N = 3
    Om = Omega(N)
    assert np.array_equal(Om, -Om.T), "Omega must be antisymmetric"
    assert np.array_equal(Om @ Om, -np.eye(2 * N)), "Omega^2 must be -I"

    rs = np.random.RandomState(5)
    V = rs.randn(2 * N, 2 * N)
    V = V @ V.T + np.eye(2 * N)
    Vpt = partial_transpose(V, [1])
    P = np.diag([1, 1, 1, 1, -1, 1])
    assert np.allclose(Vpt, P @ V @ P, atol=1e-14), "PT must equal P V P"
    assert np.allclose(partial_transpose(Vpt, [1]), V, atol=1e-14), (
        "PT must be an involution"
    )


# ----------------------------------------------------------------------------
# symplectic_eigenvalues on PSD-*singular* quadratic forms
#
# Regression for a core-API gap found by the Layer 3 QEI milestone (M3.4):
# the shared eigh helper used to build an inverse square root that
# symplectic_eigenvalues never uses, so a singular-but-PSD input raised
# FloatingPointError instead of returning its spectrum. The Layer 2/3
# sampling operators are generically singular (unsampled modes have exactly
# zero weight), so the function has to work there. `ground_state_cov` and
# `williamson` still need the inverse root and still demand definiteness.
# ----------------------------------------------------------------------------

def test_symplectic_eigenvalues_accepts_singular_psd_forms():
    """A rank-deficient PSD form has a spectrum, and it is the right one."""
    N = 4
    # O = sum over a SUBSET of modes of (x_k^2 + p_k^2) * c_k: symplectic
    # eigenvalues are c_k on the supported modes and 0 on the rest.
    cs = np.array([2.5, 0.0, 0.75, 0.0])
    O = np.diag(np.concatenate([cs, cs]))
    nu = symplectic_eigenvalues(O)
    assert np.all(np.isfinite(nu)), f"non-finite spectrum: {nu}"
    dev = float(np.max(np.abs(np.sort(nu) - np.sort(cs))))
    assert dev < 1e-14, (
        f"singular-PSD symplectic spectrum {np.sort(nu)} vs expected "
        f"{np.sort(cs)} (max dev {dev:.3e})"
    )

    # a dense, non-diagonal singular case: rank 4 of 8, so exactly two
    # symplectic eigenvalues can be nonzero and the other two are exact nulls.
    rs = np.random.RandomState(11)
    B = rs.randn(2 * N, 4)
    O2 = B @ B.T
    nu2 = np.sort(symplectic_eigenvalues(O2))
    assert np.all(np.isfinite(nu2)) and np.all(nu2 >= -1e-14), (
        f"rank-4 form gave {nu2}"
    )
    ref = np.sort(np.abs(np.linalg.eigvals(1j * Omega(N) @ O2)))[::2]
    scale = float(np.max(np.abs(O2)))
    # The two genuine eigenvalues must agree to float64 precision; the two
    # exact nulls only to the sqrt(eps)*||O|| floor that ANY square-root
    # route has on a singular form (both routes take a root, so both carry
    # it — this is the reason `qei` regularizes rather than trusting them).
    null_floor = math.sqrt(np.finfo(float).eps) * scale
    dev_top = float(np.max(np.abs(nu2[2:] - np.sort(ref)[2:])))
    assert dev_top < 1e-10 * scale, (
        f"rank-4 nonzero spectrum {nu2[2:]} vs |eig(i Omega O)| "
        f"{np.sort(ref)[2:]} (max dev {dev_top:.3e}, scale {scale:.3e})"
    )
    assert float(np.max(nu2[:2])) < null_floor, (
        f"rank-4 null directions came back as {nu2[:2]} "
        f"(sqrt(eps)*||O|| floor {null_floor:.3e})"
    )


def test_singular_input_still_refused_where_the_inverse_root_is_needed():
    """The definite-only entry points must NOT have been loosened with it."""
    K_singular = np.diag([1.0, 0.0, 4.0])
    with pytest.raises((FloatingPointError, ValueError, np.linalg.LinAlgError)):
        ground_state_cov(K_singular)
    V_singular = np.diag([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    with pytest.raises((FloatingPointError, ValueError, np.linalg.LinAlgError)):
        williamson(V_singular)


def test_definite_path_is_bit_for_bit_unchanged_by_the_lazy_inverse():
    """The vacuum spectrum still lands exactly on the floor."""
    K = harmonic_chain_K(6, m=0.4, bc="periodic")
    nu = symplectic_eigenvalues(ground_state_cov(K))
    dev = float(np.max(np.abs(nu - 0.5)))
    assert dev < 1e-14, f"vacuum nu deviates by {dev:.3e} (floor is exactly 1/2)"
