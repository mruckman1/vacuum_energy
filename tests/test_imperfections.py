"""Tests for detector imperfections and the circuit-QED parameter files (M2.6).

Spec anchors pinned here (docs/PLAN_LAYERS_2_3.md, M2.6):

- every channel this module builds is complete-positive by the
  Holevo-Werner condition Y + (i/2)(Omega - X Omega X^T) >= 0 (Holevo-Werner,
  PRA 63, 032312 (2001), Eq. (3.4)) — including the frequency-rescaled
  forms, whose CP follows from a real symplectic congruence;
- ``with_imperfections`` equals the manual composition of the underlying
  ``vacuum.core.channels`` maps to roundoff, and equals
  ``apply_channel(V, *imperfection_XY(...))``;
- the chosen circuit-QED dephasing mapping (phase-covariant additive
  Gaussian noise; module docstring item 2) is CP for every strength and
  every anisotropy option, and the loss channel at eta -> 0 relaxes a
  detector of gap Omega_d exactly to ``thermal_state_cov([[Omega_d^2]], T)``
  — the field-temperature route of the same milestone;
- the new ``experiments/circuit_qed_harvesting/`` files load through the
  provenance-enforcing loader, and the synthetic ones say so;
- a smoke run of the nonperturbative stack with imperfections, both
  through ``vacuum.protocol.run_protocol`` and through the Trotterized
  ``run_harvesting`` channel interface, closes its ledger to 1e-9.

Conventions per docs/API.md: hbar = k_B = 1, V_vac = I/2, block ordering
R = (x_1..x_N, p_1..p_N), nats.  Measured numbers ride in the assertion
messages (repo testing convention).
"""

import math

import numpy as np
import pytest

from vacuum import experiments_io as xio
from vacuum.audits import AuditViolation
from vacuum.core import (
    Omega,
    apply_channel,
    ground_state_cov,
    harmonic_chain_K,
    loss,
    mean_energy,
    symplectic_eigenvalues,
    thermal_noise,
    thermal_state_cov,
)
from vacuum.detectors import (
    DEPHASING_AXES,
    Imperfections,
    attach_detectors,
    bath_occupation,
    circuit_qed_noise,
    dephasing,
    dephasing_XY,
    detector_block,
    detector_loss,
    field_bath_occupations,
    field_thermal_state,
    harvested_log_negativity,
    holevo_werner_margin,
    holevo_werner_matrix,
    imperfection_XY,
    imperfection_channel,
    imperfection_step,
    imperfections_from_rates,
    initial_state,
    is_completely_positive,
    list_experiments,
    load_experiment,
    loss_channel_XY,
    mode_scaling,
    run_harvesting,
    scale_channel,
    switching,
    with_imperfections,
)
from vacuum.protocol import GeneratorStep, run_protocol

# --- the harvesting rig (matches tests/test_nonperturbative.py's) ----------
N_F = 8
SITES = (1, 5)
GAPS = (1.0, 1.0)
DET = (N_F, N_F + 1)
LAM = 0.6
LEDGER_TOL = 1e-9  # spec test matrix: "Ledger closure, all protocols 1e-9"
CP_TOL = 1e-12

NEW_FILES = (
    "circuit_qed_harvesting/janzen_2023_fig6",
    "circuit_qed_harvesting/janzen_2023_fig6_variant-noise-sweep",
    "circuit_qed_harvesting/teixido-bonfill_2026_table1_variant-switching",
)
SYNTHETIC_NEW_FILES = NEW_FILES[1:]


@pytest.fixture(scope="module")
def K_field():
    return harmonic_chain_K(N_F, 1.0, bc="periodic")


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20260901)


@pytest.fixture(scope="module")
def random_state(rng):
    """A physical 3-mode Gaussian state: a squeezed, correlated vacuum."""
    K = np.array([[2.0, 0.4, 0.1], [0.4, 1.3, 0.25], [0.1, 0.25, 0.8]])
    return ground_state_cov(K)


# ==========================================================================
# 0. Package consolidation: vacuum.detectors is the single import surface
# ==========================================================================


def test_detectors_package_reexports_every_submodule():
    """Every submodule's public API is reachable from ``vacuum.detectors``.

    M2.6 consolidates the package's exports; this test is what keeps a new
    submodule from quietly landing outside the single import surface.
    """
    import importlib

    import vacuum.detectors as pkg

    submodules = (
        "kernels",
        "switching",
        "smearing",
        "nonperturbative",
        "udw",
        "nogo",
        "communication",
        "imperfections",
    )
    for name in submodules:
        mod = importlib.import_module(f"vacuum.detectors.{name}")
        missing = [n for n in getattr(mod, "__all__", ()) if n not in pkg.__all__]
        assert not missing, f"vacuum.detectors.{name} exports not consolidated: {missing}"
    absent = [n for n in pkg.__all__ if not hasattr(pkg, n)]
    assert not absent, f"names in __all__ that do not exist: {absent}"
    # the loader spelling promised by experiments/README.md
    assert pkg.load_experiment is xio.load_experiment


# ==========================================================================
# 1. Complete positivity (Holevo-Werner), the milestone's stated CP test
# ==========================================================================

_CP_CASES = {
    "loss": lambda: loss_channel_XY(3, (1, 2), 0.6, 0.3),
    "loss-deep": lambda: loss_channel_XY(3, (0,), 0.0, 1.7),
    "loss-scaled": lambda: loss_channel_XY(3, (1, 2), 0.45, 0.9, gaps=(2.3, 0.4)),
    "deph-iso": lambda: dephasing_XY(3, (1, 2), 0.31),
    "deph-p": lambda: dephasing_XY(3, (1, 2), 0.31, axis="p"),
    "deph-x": lambda: dephasing_XY(3, (0,), 1.5, axis="x"),
    "deph-scaled": lambda: dephasing_XY(3, (1, 2), 0.31, gaps=(2.3, 0.4)),
    "deph-zero": lambda: dephasing_XY(3, (1,), 0.0),
    "combined": lambda: imperfection_XY(3, (1, 2), 0.7, 0.4, 0.05),
    "combined-scaled": lambda: imperfection_XY(
        3, (1, 2), 0.7, 0.4, 0.05, axis="p", gaps=(2.3, 0.4)
    ),
    "combined-identity": lambda: imperfection_XY(3, (1, 2), 1.0, 0.0, 0.0),
}


@pytest.mark.parametrize("name", sorted(_CP_CASES))
def test_channels_are_completely_positive(name, random_state):
    """Holevo-Werner: Y + (i/2)(Omega - X Omega X^T) >= 0 for every channel."""
    X, Y = _CP_CASES[name]()
    margin = holevo_werner_margin(X, Y)
    assert margin >= -CP_TOL, (
        f"{name}: Holevo-Werner margin {margin:.3e} < -{CP_TOL:g} (channel not CP)"
    )
    assert is_completely_positive(X, Y, tol=CP_TOL)
    # and the physical consequence: a physical input stays physical
    nu = symplectic_eigenvalues(apply_channel(random_state, X, Y))
    assert np.min(nu) >= 0.5 - CP_TOL, (
        f"{name}: output min nu = {np.min(nu):.15f} < 1/2 (unphysical state)"
    )


def test_holevo_werner_matrix_is_hermitian_and_identity_is_marginal():
    """The identity channel sits exactly on the CP boundary (margin 0)."""
    X, Y = np.eye(4), np.zeros((4, 4))
    M = holevo_werner_matrix(X, Y)
    assert np.allclose(M, M.conj().T, atol=0.0), "CP matrix not Hermitian"
    assert holevo_werner_margin(X, Y) == 0.0


def test_holevo_werner_catches_a_deliberately_invalid_channel():
    """Canary: a lossy X with no compensating noise must FAIL the CP test."""
    N = 2
    X = math.sqrt(0.5) * np.eye(2 * N)
    Y = np.zeros((2 * N, 2 * N))  # missing the (1-eta)/2 vacuum noise
    margin = holevo_werner_margin(X, Y)
    assert margin < -0.2, (
        f"noiseless attenuator passed CP with margin {margin:.3e} — the "
        "Holevo-Werner check is not discriminating"
    )
    assert not is_completely_positive(X, Y)


def test_dephasing_is_additive_noise_so_CP_for_every_strength(rng):
    """X = I makes the CP condition read Y >= 0 — true for all D >= 0."""
    for D in (0.0, 1e-12, 0.7, 25.0):
        for axis in DEPHASING_AXES:
            X, Y = dephasing_XY(2, (0,), D, axis=axis, gaps=(1.9,))
            assert np.allclose(X, np.eye(4), atol=0.0), "dephasing must have X = I"
            assert holevo_werner_margin(X, Y) >= -CP_TOL, f"D={D}, axis={axis}"
    with pytest.raises(ValueError, match="must be >= 0"):
        dephasing_XY(2, (0,), -1e-9)


def test_frequency_rescaling_preserves_cp_by_congruence(rng):
    """Sigma is symplectic, so the CP matrix transforms by a real congruence."""
    N, modes, gaps = 3, (0, 2), (3.1, 0.27)
    X, Y = loss_channel_XY(N, modes, 0.35, 0.8)
    s = mode_scaling(N, modes, gaps)
    Xs, Ys = scale_channel(X, Y, s)
    # Sigma is symplectic: Sigma Omega Sigma^T = Omega
    Sig = np.diag(s)
    dev = np.max(np.abs(Sig @ Omega(N) @ Sig.T - Omega(N)))
    assert dev < 1e-14, f"mode_scaling is not symplectic: deviation {dev:.3e}"
    # congruence identity on the CP matrix
    M = holevo_werner_matrix(X, Y)
    Ms = holevo_werner_matrix(Xs, Ys)
    inv = np.diag(1.0 / s)
    dev2 = np.max(np.abs(Ms - inv @ M @ inv))
    assert dev2 < 1e-13, f"CP matrix congruence violated by {dev2:.3e}"
    assert holevo_werner_margin(Xs, Ys) >= -CP_TOL


# ==========================================================================
# 2. Composition: with_imperfections == manual channel application
# ==========================================================================


def test_composition_matches_manual_channel_application(random_state):
    """The milestone's composition test, in all three spellings."""
    V = random_state
    N, modes = 3, (1, 2)
    eta, nbar, D = 0.73, 0.41, 0.06
    got = with_imperfections(V, modes, eta, nbar, D)
    manual = dephasing(detector_loss(V, modes, eta, nbar), modes, D)
    via_XY = apply_channel(V, *imperfection_XY(N, modes, eta, nbar, D))
    assert np.max(np.abs(got - manual)) < 1e-15, (
        f"composed map differs from manual loss->dephasing by "
        f"{np.max(np.abs(got - manual)):.3e}"
    )
    assert np.max(np.abs(got - via_XY)) < 1e-15, (
        f"composed map differs from apply_channel(imperfection_XY) by "
        f"{np.max(np.abs(got - via_XY)):.3e}"
    )


def test_gaps_none_is_exactly_the_bare_core_channels(random_state):
    """With Omega_d = 1 the module reduces to vacuum.core.channels (1 ulp).

    The only difference is the order of the final float additions
    (Y_loss + Y_deph is summed once here, applied twice there), so the
    tolerance is a rounding unit, not a physics tolerance.
    """
    V = random_state
    modes = (0, 2)
    eta, nbar, D = 0.55, 0.2, 0.13
    core_route = thermal_noise(loss(V, modes, eta, nbar), modes, D)
    ours = with_imperfections(V, modes, eta, nbar, D)  # gaps=None
    dev = np.max(np.abs(ours - core_route))
    assert dev < 1e-15, (
        f"gaps=None path deviates from core.channels by {dev:.3e} "
        f"(expected roundoff only)"
    )
    # unit gaps given explicitly must agree to roundoff too
    unit_gaps = with_imperfections(V, modes, eta, nbar, D, gaps=(1.0, 1.0))
    assert np.max(np.abs(unit_gaps - core_route)) < 1e-15


def test_composition_order_is_the_documented_one(random_state):
    """Loss-then-dephasing; the other order differs by exactly eta*Y_deph."""
    V, N, modes = random_state, 3, (1,)
    eta, nbar, D = 0.4, 0.0, 0.5
    Xl, Yl = loss_channel_XY(N, modes, eta, nbar)
    Xd, Yd = dephasing_XY(N, modes, D)
    X_ours, Y_ours = imperfection_XY(N, modes, eta, nbar, D)
    assert np.max(np.abs(Y_ours - (Yl + Yd))) < 1e-15, "Y != Y_loss + Y_deph"
    # dephasing first would give X_l Y_d X_l^T + Y_l = eta Y_d + Y_l
    Y_other = Xl @ Yd @ Xl.T + Yl
    gap = np.max(np.abs(Y_ours - Y_other))
    assert gap == pytest.approx((1.0 - eta) * D, rel=1e-12), (
        f"order-dependence {gap:.6e} != (1-eta)D = {(1.0 - eta) * D:.6e}"
    )
    assert np.max(np.abs(X_ours - Xl)) < 1e-15


def test_identity_limits(random_state):
    """eta = 1, D = 0 is the identity map, bit for bit."""
    V = random_state
    out = with_imperfections(V, (0, 1, 2), eta=1.0, nbar=0.9, dephasing=0.0)
    assert np.max(np.abs(out - V)) == 0.0, "eta=1, D=0 perturbed the state"
    inert = imperfection_channel((0, 1), kappa=0.0, gamma_phi=0.0)
    assert np.max(np.abs(inert(V, 0.3, 0.25) - V)) == 0.0, (
        "a zero-rate imperfection channel is not inert"
    )
    assert Imperfections().is_identity()


def test_input_is_never_mutated(random_state):
    V = np.array(random_state, copy=True)
    ref = V.copy()
    with_imperfections(V, (0,), 0.3, 0.5, 0.2, gaps=(1.4,))
    dephasing(V, (1,), 0.4)
    detector_loss(V, (2,), 0.1, 0.2)
    assert np.array_equal(V, ref), "an imperfection map mutated its input"


# ==========================================================================
# 3. Physics: the bath, the field temperature, and the loss/dephasing action
# ==========================================================================


@pytest.mark.parametrize("gap", [0.35, 1.0, 2.75])
@pytest.mark.parametrize("T", [0.0, 0.4, 3.0])
def test_full_loss_relaxes_a_detector_to_its_thermal_state(gap, T):
    """eta -> 0 with the Bose bath gives exactly thermal_state_cov (M2.6).

    This is the statement that ties the detector bath to the field-side
    temperature route: both are the same coth(omega/2T) formula.
    """
    V = ground_state_cov(np.array([[gap**2]]))
    nbar = bath_occupation(gap, T)
    out = detector_loss(V, (0,), eta=0.0, nbar=nbar, gaps=(gap,))
    want = thermal_state_cov(np.array([[gap**2]]), T)
    dev = np.max(np.abs(out - want))
    assert dev < 1e-14, (
        f"gap={gap}, T={T}: full loss gave a state {dev:.3e} away from "
        f"thermal_state_cov (nbar = {nbar:.6e})"
    )


def test_unscaled_loss_would_relax_to_the_wrong_state():
    """Why the frequency rescaling exists: gaps=None is wrong for Omega != 1."""
    gap = 2.75
    V = ground_state_cov(np.array([[gap**2]]))
    wrong = detector_loss(V, (0,), eta=0.0, nbar=0.0)  # gaps=None -> I/2
    right = detector_loss(V, (0,), eta=0.0, nbar=0.0, gaps=(gap,))
    assert np.max(np.abs(wrong - 0.5 * np.eye(2))) < 1e-15
    assert np.max(np.abs(right - V)) < 1e-14, "scaled loss moved the ground state"
    assert np.max(np.abs(wrong - right)) > 0.5, (
        "the frequency rescaling made no difference at Omega_d = 2.75 — "
        "the test rig is not exercising it"
    )


def test_zero_temperature_loss_is_the_detector_ground_state():
    gap = 1.6
    V = 4.0 * ground_state_cov(np.array([[gap**2]]))  # a hot detector
    out = detector_loss(V, (0,), eta=0.0, nbar=bath_occupation(gap, 0.0), gaps=(gap,))
    want = ground_state_cov(np.array([[gap**2]]))
    assert np.max(np.abs(out - want)) < 1e-14


def test_dephasing_adds_the_stated_variance_and_energy():
    """Isotropic Y~ = D I_2, single-axis Y~ = 2D e_p e_p^T: equal added energy."""
    gap, D = 1.7, 0.23
    K = np.array([[gap**2]])
    V = ground_state_cov(K)
    iso = dephasing(V, (0,), D, axis="isotropic", gaps=(gap,))
    pon = dephasing(V, (0,), D, axis="p", gaps=(gap,))
    xon = dephasing(V, (0,), D, axis="x", gaps=(gap,))
    # isotropic in the RESCALED quadratures: xx grows by D/gap, pp by D*gap
    assert iso[0, 0] - V[0, 0] == pytest.approx(D / gap, rel=1e-12)
    assert iso[1, 1] - V[1, 1] == pytest.approx(D * gap, rel=1e-12)
    assert pon[0, 0] == pytest.approx(V[0, 0], rel=1e-12)
    assert pon[1, 1] - V[1, 1] == pytest.approx(2.0 * D * gap, rel=1e-12)
    # equal added energy <H> = (1/2)(V_pp + gap^2 V_xx)
    E0 = mean_energy(V, K)
    dE = [mean_energy(W, K) - E0 for W in (iso, pon, xon)]
    assert dE[0] == pytest.approx(dE[1], rel=1e-12), f"added energies {dE}"
    assert dE[0] == pytest.approx(dE[2], rel=1e-12), f"added energies {dE}"
    assert dE[0] > 0.0, "dephasing did not heat the detector"


def test_pure_loss_degrades_without_killing(clean_harvest):
    """Vacuum-bath loss (nbar = 0) thins the PT margin but never zeroes it.

    Measured on this rig: margin -2.8395e-3 at eta = 1, thinning to
    -2.7849e-3 / -2.3176e-3 / -7.5902e-4 / -4.5993e-5 / -1.9520e-8 at
    eta = 0.99 / 0.9 / 0.5 / 0.1 / 1e-4 — closer to eta^2 than to eta,
    because the added (1-eta)/2 noise is isotropic while the near-floor
    direction of the partial transpose is strongly squeezed.  Two
    properties are asserted: the margin never crosses zero at eta > 0 (no
    sudden death under vacuum-bath loss, the backbone of the noise map),
    and it thins at least in proportion to eta.
    """
    V_AB = detector_block(clean_harvest.V, N_F)
    neg0 = harvested_log_negativity(V_AB)
    assert neg0.margin_min < 0.0, "clean harvest is not entangled; rig broken"
    prev = neg0.E_N
    for eta in (0.99, 0.9, 0.5, 0.1, 1e-4):
        neg = harvested_log_negativity(
            with_imperfections(V_AB, (0, 1), eta=eta, nbar=0.0, gaps=GAPS)
        )
        assert 0.0 < neg.E_N < prev, (
            f"eta={eta}: E_N {neg.E_N:.6e} not strictly between 0 and the "
            f"previous {prev:.6e} — pure loss must degrade, not kill"
        )
        assert neg.margin_min < 0.0, (
            f"eta={eta}: PT margin {neg.margin_min:.6e} crossed the vacuum "
            f"floor under vacuum-bath loss"
        )
        ratio = neg.margin_min / neg0.margin_min / eta
        assert ratio <= 1.0 + 1e-9, (
            f"eta={eta}: margin thinned more slowly than eta "
            f"(margin/(eta*margin0) = {ratio:.6f} > 1)"
        )
        prev = neg.E_N


def test_hot_bath_and_dephasing_kill_negativity_beyond_a_threshold(clean_harvest):
    """Sudden death in both noise channels — the go/no-go boundary of M2.7.

    Measured on this rig (clean E_N 5.6952e-3): a bath of nbar = 1e-3 at
    eta = 0.9 leaves 4.4442e-3, nbar = 3e-2 kills it outright; dephasing
    D = 1e-3 leaves 3.6752e-3 and D = 3e-3 kills it.
    """
    V_AB = detector_block(clean_harvest.V, N_F)
    E_bath = [
        harvested_log_negativity(
            with_imperfections(V_AB, (0, 1), eta=0.9, nbar=nb, gaps=GAPS)
        ).E_N
        for nb in (0.0, 1e-3, 3e-2)
    ]
    assert E_bath[0] > E_bath[1] > 0.0, f"E_N not degrading with bath: {E_bath}"
    assert E_bath[2] == 0.0, (
        f"E_N {E_bath[2]:.3e} survived nbar = 3e-2 (expected sudden death)"
    )
    E_deph = [
        harvested_log_negativity(
            with_imperfections(V_AB, (0, 1), eta=1.0, dephasing=D, gaps=GAPS)
        ).E_N
        for D in (0.0, 1e-3, 3e-3)
    ]
    assert E_deph[0] > E_deph[1] > 0.0, f"E_N not degrading with D: {E_deph}"
    assert E_deph[2] == 0.0, (
        f"E_N {E_deph[2]:.3e} survived D = 3e-3 (expected sudden death)"
    )


def test_field_thermal_state_is_the_core_route(K_field):
    """Field temperature enters ONLY through thermal_state_cov (spec M2.6)."""
    assert np.max(np.abs(field_thermal_state(K_field, 0.0) - ground_state_cov(K_field))) == 0.0
    for T in (0.2, 1.5):
        V = field_thermal_state(K_field, T)
        assert np.max(np.abs(V - thermal_state_cov(K_field, T))) == 0.0
        omega, nbar = field_bath_occupations(K_field, T)
        E_bose = float(np.sum(omega * (nbar + 0.5)))
        E_cov = mean_energy(V, K_field)
        assert E_cov == pytest.approx(E_bose, rel=1e-12), (
            f"T={T}: <H> {E_cov:.12e} != analytic Bose sum {E_bose:.12e}"
        )
    with pytest.raises(ValueError, match="must be >= 0"):
        field_thermal_state(K_field, -0.1)


def test_bath_occupation_limits():
    assert bath_occupation(1.0, 0.0) == 0.0
    assert bath_occupation(2.0, 1e-6) == 0.0  # exp-overflow guard, exactly 0
    # high-T limit nbar -> T/omega - 1/2
    assert bath_occupation(1.0, 1e4) == pytest.approx(1e4 - 0.5, rel=1e-6)


# ==========================================================================
# 4. Rates -> channel parameters, and the semigroup property
# ==========================================================================


def test_imperfections_from_rates_matches_the_channel_propagator(random_state):
    """The window form and the rate form are the same map, identically."""
    modes, gaps, tau = (1, 2), (1.3, 0.8), 0.7
    G1, Gphi, T = 0.35, 0.12, 0.9
    imp = imperfections_from_rates(gaps, tau, T=T, Gamma_1=G1, Gamma_phi=Gphi)
    ch = imperfection_channel(
        modes, kappa=G1, nbar=imp.nbar, gamma_phi=Gphi, gaps=gaps
    )
    a = imp.apply(random_state, modes)
    b = ch(random_state, 0.0, tau)
    assert np.max(np.abs(a - b)) < 1e-15, (
        f"rate form and window form differ by {np.max(np.abs(a - b)):.3e}"
    )
    assert imp.eta == pytest.approx(math.exp(-G1 * tau), rel=1e-15)
    assert imp.dephasing == pytest.approx(
        Gphi * (1.0 - math.exp(-G1 * tau)) / G1, rel=1e-14
    )
    assert imp.nbar == pytest.approx(bath_occupation(float(np.mean(gaps)), T))


def test_imperfection_channel_is_a_semigroup(random_state):
    """C_{dt/2} o C_{dt/2} = C_{dt} — why the Strang splitting stays 2nd order."""
    modes, gaps = (0, 1), (1.4, 0.9)
    ch = imperfection_channel(modes, kappa=0.6, nbar=0.3, gamma_phi=0.25, gaps=gaps)
    dt = 0.4
    once = ch(random_state, 0.0, dt)
    twice = ch(ch(random_state, 0.0, 0.5 * dt), 0.5 * dt, 0.5 * dt)
    dev = np.max(np.abs(once - twice))
    assert dev < 1e-14, (
        f"half-steps do not compose into the full step: {dev:.3e} — the "
        "combined Lindbladian was not integrated exactly"
    )


def test_zero_relaxation_limit_is_pure_dephasing_diffusion():
    """kappa -> 0: D -> gamma_phi * dt, computed stably through expm1."""
    imp0 = imperfections_from_rates(1.0, 2.0, Gamma_1=0.0, Gamma_phi=0.3)
    assert imp0.eta == 1.0
    assert imp0.dephasing == pytest.approx(0.6, rel=1e-15)
    tiny = imperfections_from_rates(1.0, 2.0, Gamma_1=1e-14, Gamma_phi=0.3)
    assert tiny.dephasing == pytest.approx(0.6, rel=1e-12)


def test_rate_and_gap_validation():
    with pytest.raises(ValueError, match="rates must be >= 0"):
        imperfections_from_rates(1.0, 1.0, Gamma_1=-1.0)
    with pytest.raises(ValueError, match="gaps must be > 0"):
        imperfections_from_rates(-1.0, 1.0)
    with pytest.raises(ValueError, match="kappa must be >= 0"):
        imperfection_channel((0,), kappa=-1.0)
    with pytest.raises(ValueError, match="axis"):
        dephasing_XY(2, (0,), 0.1, axis="theta")
    with pytest.raises(IndexError):
        mode_scaling(2, (5,), 1.0)


# ==========================================================================
# 5. The parameter files: provenance, honesty, and the loader bridge
# ==========================================================================


@pytest.fixture(scope="module")
def shipped():
    return {name: load_experiment(name) for name in list_experiments()}


def test_new_parameter_files_ship_and_load(shipped):
    for want in NEW_FILES:
        assert want in shipped, f"missing new parameter file {want}"
    # the sibling fixtures must keep loading with the new files present
    for name, exp in shipped.items():
        assert exp.source.strip() and exp.location.strip(), name
        assert exp.retrieved.strip() and exp.transcribed_by.strip(), name
        assert len(exp.quantities) > 0, name
        for qname, q in exp.quantities.items():
            assert xio.unit_kind(q.units), f"{name}:{qname}"


def test_synthetic_files_declare_themselves(shipped):
    for name in SYNTHETIC_NEW_FILES:
        exp = shipped[name]
        assert exp.synthetic is True, f"{name} is a declared variant but not synthetic"
        assert exp.variant_of and exp.variant_of in shipped, (
            f"{name}: variant_of {exp.variant_of!r} is not a shipped file"
        )
        assert exp.changes, f"{name}: empty changes dict"
        assert "SYNTHETIC" in exp.comment.upper(), f"{name}: comment hides the status"
        # every changed/added key must be named in changes
        for qname in exp.quantities:
            if qname in exp.changes:
                continue
            note = (exp.quantities[qname].note or "") + (
                exp.quantities[qname].location or ""
            )
            assert "parent" in note.lower() or "synthetic" in note.lower(), (
                f"{name}:{qname} is neither declared in 'changes' nor marked "
                f"as carried from the parent"
            )
    assert shipped[NEW_FILES[0]].synthetic is False


def test_transcription_spot_values(shipped):
    """Verbatim values from arXiv:2208.05571v2 (PRResearch 5, 033155 (2023))."""
    j = shipped["circuit_qed_harvesting/janzen_2023_fig6"]
    q = j.quantities
    assert q["coupling_alpha_min"].value == pytest.approx(6.2e-5)
    assert q["coupling_alpha_max"].value == pytest.approx(2.19e-2)
    assert q["relaxation_rate_gamma1_min"].value == pytest.approx(8.7)
    assert q["relaxation_rate_gamma1_min"].units == "MHz"
    assert q["relaxation_rate_gamma1_max"].value == pytest.approx(1.85)
    assert q["relaxation_rate_gamma1_max"].units == "GHz"
    assert q["transmission_line_temperature"].value == pytest.approx(50)
    assert q["transmission_line_temperature"].units == "mK"
    assert q["fridge_base_temperature"].value == pytest.approx(30)
    assert q["qubit_gap_frequency"].value == pytest.approx(5.7)
    assert q["qubit_gap_frequency"].units == "GHz"
    # cross-check: the six junction critical currents must agree with the
    # value transcribed independently from footnote 6 of arXiv:2505.01516
    tb = shipped["circuit_qed_harvesting/teixido-bonfill_2026_table1"]
    for k in range(1, 7):
        key = f"critical_current_{k}"
        assert j.quantities[key].value == pytest.approx(tb.quantities[key].value), (
            f"{key}: device paper {j.quantities[key].value} vs harvesting paper "
            f"{tb.quantities[key].value} — one transcription is wrong"
        )
        assert j.quantities[key].units == tb.quantities[key].units == "uA"


def test_plot_digitized_values_are_flagged(shipped):
    """A digitized number must say so in its own note, not only in the header."""
    j = shipped["circuit_qed_harvesting/janzen_2023_fig6"]
    listed = j.extras.get("plot_digitized")
    assert listed, "the transcription does not declare its plot-digitized entries"
    for qname in listed:
        assert qname in j.quantities, qname
        note = j.quantities[qname].note or ""
        assert "PLOT-DIGITIZED" in note, (
            f"{qname} is listed as digitized but its note does not say so"
        )
    for qname, qty in j.quantities.items():
        if "PLOT-DIGITIZED" in (qty.note or ""):
            assert qname in listed, f"{qname} is digitized but not listed"


def test_unit_conversion_of_the_transcription(shipped):
    """The loader is the one place SI values become lattice units."""
    j = shipped["circuit_qed_harvesting/janzen_2023_fig6"]
    omega_ref = 2.0 * math.pi * 5.7e9  # lattice energy unit = the device gap
    lat = j.lattice(omega_ref)
    assert lat["qubit_gap_frequency"] == pytest.approx(1.0, rel=1e-12)
    assert lat["relaxation_rate_gamma1_max"] == pytest.approx(1.85 / 5.7, rel=1e-12)
    # 50 mK in units of h * 5.7 GHz
    want_T = (xio.K_B_SI * 50e-3 / xio.HBAR_SI) / omega_ref
    assert lat["transmission_line_temperature"] == pytest.approx(want_T, rel=1e-12)
    assert want_T == pytest.approx(0.18278, rel=1e-4), (
        f"50 mK / (h*5.7 GHz) = {want_T:.6f}"
    )
    # labels and ohms carry no hbar=1 conversion and are skipped, not guessed
    assert "flux_noise_amplitude_eps" not in lat
    with pytest.raises(ValueError):
        j.lattice(omega_ref, strict=True)


def test_circuit_qed_noise_builds_a_noise_model_from_the_file(shipped):
    omega_ref = 2.0 * math.pi * 5.7e9
    tau = 0.5
    imp = circuit_qed_noise(
        "circuit_qed_harvesting/janzen_2023_fig6", omega_ref, tau=tau
    )
    lat = shipped["circuit_qed_harvesting/janzen_2023_fig6"].lattice(omega_ref)
    assert imp.eta == pytest.approx(
        math.exp(-lat["relaxation_rate_gamma1_max"] * tau), rel=1e-14
    )
    assert imp.nbar == pytest.approx(
        bath_occupation(lat["qubit_gap_frequency"], lat["transmission_line_temperature"])
    )
    assert 0.0 < imp.eta < 1.0 and imp.dephasing > 0.0 and imp.nbar > 0.0
    assert "janzen_2023_fig6" in imp.provenance, "the row lost its provenance"
    # the file's own record object is accepted too
    same = circuit_qed_noise(
        shipped["circuit_qed_harvesting/janzen_2023_fig6"], omega_ref, tau=tau
    )
    assert same.eta == imp.eta and same.dephasing == imp.dephasing
    # a file without the expected quantities is refused, not guessed at
    with pytest.raises(KeyError):
        circuit_qed_noise("dce_microwave/wilson_2011_table1", omega_ref, tau=tau)


def test_synthetic_sweep_grids_bracket_the_measured_values(shipped):
    """The declared variant's grids must actually contain the parent's numbers."""
    parent = shipped["circuit_qed_harvesting/janzen_2023_fig6"].quantities
    var = shipped["circuit_qed_harvesting/janzen_2023_fig6_variant-noise-sweep"]
    T_grid = var.quantities["waveguide_temperature_grid"].value
    assert parent["transmission_line_temperature"].value in T_grid
    assert parent["fridge_base_temperature"].value in T_grid
    rates = var.quantities["relaxation_rate_grid"].value  # MHz
    assert min(rates) < parent["relaxation_rate_gamma1_min"].value
    assert max(rates) > 1e3 * parent["relaxation_rate_gamma1_max"].value
    deph = var.quantities["pure_dephasing_rate_grid"].value  # Hz
    assert min(deph) < parent["pure_dephasing_rate_min"].value
    assert max(deph) > parent["pure_dephasing_rate_max"].value
    alpha = var.quantities["coupling_alpha_grid"].value
    assert min(alpha) == pytest.approx(parent["coupling_alpha_min"].value)
    assert max(alpha) == pytest.approx(parent["coupling_alpha_max"].value)


def test_switching_variant_is_causally_self_consistent(shipped):
    """t_d = L/v vs the switching window T, computed from the file itself."""
    var = shipped["circuit_qed_harvesting/teixido-bonfill_2026_table1_variant-switching"]
    v = var.quantities["waveguide_speed"].value  # m/s
    for tag, expect in (("spacelike", 1), ("causal", -1), ("marginal", 0)):
        T_ns = var.quantities[f"switching_duration_{tag}"].value
        L_mm = var.quantities[f"detector_separation_{tag}"].value
        t_d_ns = (L_mm * 1e-3 / v) * 1e9
        if expect > 0:
            assert t_d_ns > T_ns, f"{tag}: t_d {t_d_ns:.3f} ns <= T {T_ns} ns"
        elif expect < 0:
            assert t_d_ns < 0.5 * T_ns, f"{tag}: t_d {t_d_ns:.3f} ns not << T {T_ns}"
        else:
            assert t_d_ns == pytest.approx(T_ns, rel=1e-9), (
                f"marginal point is not on the light cone: t_d {t_d_ns:.4f} ns "
                f"vs T {T_ns} ns"
            )


# ==========================================================================
# 6. Smoke runs: the nonperturbative stack with imperfections, ledger closing
# ==========================================================================

CHI = switching("cos2", 3.0, t0=3.0)  # compact bump on [0, 6]
TS = np.linspace(0.0, 6.0, 25)


def _lambda_of_t(t):
    return LAM * CHI(t)


def _K_of_t(K_field, t):
    return attach_detectors(K_field, SITES, GAPS, _lambda_of_t(t))


@pytest.fixture(scope="module")
def clean_harvest(K_field):
    """Noiseless reference run (matches the M2.2 rig): E_N ~ 5.695e-3."""
    return run_harvesting(K_field, SITES, GAPS, _lambda_of_t, TS)


@pytest.fixture(scope="module")
def noisy_harvest(K_field):
    """The same run with the imperfection channel Trotterized into it."""
    ch = imperfection_channel(DET, kappa=0.02, nbar=1e-4, gamma_phi=1e-4, gaps=GAPS)
    return run_harvesting(K_field, SITES, GAPS, _lambda_of_t, TS, channels=(ch,))


def test_smoke_run_harvesting_with_imperfections_closes_its_ledger(
    clean_harvest, noisy_harvest
):
    """Trotterized imperfections: gate converges, books close to 1e-9.

    Measured on this rig (N=8, m=1, sites (1,5), gaps 1, lam 0.6, cos2 T=3,
    kappa=0.02, nbar=1e-4, gamma_phi=1e-4): E_N 5.6952e-3 -> 3.7118e-3.
    """
    res = noisy_harvest
    assert res.converged, (
        f"open run failed the dt-halving gate after {res.halvings} halvings"
    )
    defect = res.ledger_result.details["defect"]
    assert abs(defect) <= LEDGER_TOL, (
        f"ledger defect {defect:.3e} > {LEDGER_TOL:g} with imperfections on"
    )
    assert res.ledger_result.passed
    kinds = [r["kind"] for r in res.ledger.table]
    assert kinds.count("dissipation") == 1 and kinds.count("work") == 1
    assert res.dissipated != 0.0, "the imperfection channel booked zero energy flow"
    assert res.dissipated < 0.0, (
        f"dissipation {res.dissipated:.3e} > 0: a heating channel must book "
        "energy IN (negative dissipation)"
    )
    assert 0.0 < res.E_N < clean_harvest.E_N, (
        f"noisy E_N {res.E_N:.6e} not strictly between 0 and the clean "
        f"{clean_harvest.E_N:.6e}"
    )
    assert np.all(res.n_d > clean_harvest.n_d), (
        f"detectors not heated: <n_d> {res.n_d} vs clean {clean_harvest.n_d}"
    )
    # the row carries its provenance metadata (pre-positioning rule)
    assert res.dt_converged > 0.0 and len(res.evolve.history) == res.halvings + 1


def test_smoke_run_protocol_with_imperfections(K_field):
    """The same physics through vacuum/protocol.py, the audited code path."""
    init = initial_state(K_field, SITES, GAPS, 0.0)
    assert init.passivity.passed, "initial ground state failed the passivity audit"

    def build(noise_step):
        steps = [
            GeneratorStep(
                H=lambda t: _K_of_t(K_field, t),
                duration=3.0,
                n_substeps=96,
                label="ramp_on",
            )
        ]
        if noise_step is not None:
            steps.append(noise_step)
        steps.append(
            GeneratorStep(
                H=lambda t: _K_of_t(K_field, t),
                duration=3.0,
                n_substeps=96,
                label="ramp_off",
            )
        )
        return run_protocol(
            init.V,
            init.K_tot,
            steps,
            claim_ground_entry=True,
            run_causality=False,  # detector-augmented ordering is not a chain
            report_modes=[DET],
            ledger_tol=LEDGER_TOL,
        )

    clean = build(None)
    noisy = build(
        imperfection_step(DET, eta=0.97, nbar=1e-3, dephasing=2e-4, gaps=GAPS)
    )
    for tag, res in (("clean", clean), ("noisy", noisy)):
        defect = res.audits["ledger"].details["defect"]
        assert abs(defect) <= LEDGER_TOL, (
            f"{tag}: protocol ledger defect {defect:.3e} > {LEDGER_TOL:g}"
        )
        assert res.all_passed(), f"{tag}: a standing audit failed"
    E_clean = harvested_log_negativity(detector_block(clean.V, N_F)).E_N
    E_noisy = harvested_log_negativity(detector_block(noisy.V, N_F)).E_N
    assert 0.0 < E_noisy < E_clean, (
        f"E_N with imperfections {E_noisy:.6e} not strictly between 0 and the "
        f"clean {E_clean:.6e}"
    )
    # the noise step booked exactly one dissipation entry, and it heated
    diss = [r for r in noisy.step_log if r["kind"] == "channel"]
    assert len(diss) == 1 and diss[0]["dissipated"] < 0.0, (
        f"imperfection step booked {diss}"
    )
    assert clean.dissipated_total == 0.0


def test_protocol_ledger_still_closes_with_a_hot_field(K_field):
    """Field temperature (thermal_state_cov) + detector imperfections."""
    T = 0.3
    init = initial_state(K_field, SITES, GAPS, T)
    assert init.passivity is None, "a thermal state must not be claimed as ground"
    assert np.max(np.abs(init.V[:N_F, :N_F] - field_thermal_state(K_field, T)[:N_F, :N_F])) < 1e-14
    steps = [
        GeneratorStep(
            H=lambda t: _K_of_t(K_field, t), duration=6.0, n_substeps=192, label="drive"
        ),
        imperfection_step(DET, eta=0.9, nbar=bath_occupation(1.0, T), gaps=GAPS),
    ]
    res = run_protocol(
        init.V,
        init.K_tot,
        steps,
        claim_ground_entry=False,
        run_causality=False,
        report_modes=[DET],
        ledger_tol=LEDGER_TOL,
    )
    defect = res.audits["ledger"].details["defect"]
    assert abs(defect) <= LEDGER_TOL, f"hot-field ledger defect {defect:.3e}"
    assert res.all_passed()
    E = harvested_log_negativity(detector_block(res.V, N_F)).E_N
    assert E >= 0.0, f"negative E_N {E:.3e}"


def test_a_broken_ledger_would_be_caught(K_field):
    """Canary: the closure audit is load-bearing, not decorative."""
    init = initial_state(K_field, SITES, GAPS, 0.0)

    def leaky(V, modes, **kw):
        # a map that changes the energy without any bookkeeping hook would
        # still be booked by the runner; instead corrupt the ledger directly
        return with_imperfections(V, modes, **kw)

    res = run_protocol(
        init.V,
        init.K_tot,
        [
            GeneratorStep(
                H=lambda t: _K_of_t(K_field, t), duration=6.0, n_substeps=64
            ),
            imperfection_step(DET, eta=0.8, nbar=0.1, dephasing=0.05, gaps=GAPS),
        ],
        claim_ground_entry=True,
        run_causality=False,
        report_modes=[DET],
        ledger_tol=LEDGER_TOL,
    )
    assert res.audits["ledger"].passed
    res.ledger.work("phantom", 1.0)  # inject an unphysical entry
    with pytest.raises(AuditViolation):
        res.ledger.close(tol=LEDGER_TOL, strict=True)
