"""Pins for the blind re-derivation in docs/coupling_rederivation/.

Every identity of derivation.md that can be checked numerically is checked
here, and the headline numbers are pinned.  The module under test reads
only 'transcribed' parameter-file entries.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from vacuum.core.gaussian import evolve, ground_state_cov, symplectic_from_quadratic
from vacuum.core.models import harmonic_chain_K
from vacuum.detectors.nonperturbative import attach_detectors
from vacuum.experiments_io import HBAR_SI, load_experiment

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "rederive", _REPO / "docs" / "coupling_rederivation" / "rederive.py"
)
rd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rd)


@pytest.fixture(scope="module")
def exp():
    return load_experiment(rd.EXPERIMENT)


# ------------------------------------------------- Eq. (66) and its checks --


def test_eq66_coefficient_from_SI_exact_constants(exp):
    Z0 = rd.transcribed(exp, "characteristic_impedance").value
    coef = rd.lambda_tb_per_gamma(Z0)
    assert coef == pytest.approx(-4.5322, abs=5e-4), f"-sqrt(R_K/(8 pi Z0)) = {coef}"
    printed = rd.transcribed(exp, "lambda_per_gamma").value
    assert coef == pytest.approx(printed, rel=1e-3), f"{coef} vs printed {printed}"


def test_eq66_velocity_cancels(exp):
    """-(phi_0/(hbar v l_0)) sqrt(hbar Z_0) with l_0 = Z_0/v is v-independent."""
    Z0 = rd.transcribed(exp, "characteristic_impedance").value
    for v in (1.2e8, 1.334e8, 3.0e8):
        assert rd.lambda_tb_per_gamma_explicit(Z0, v) == pytest.approx(
            rd.lambda_tb_per_gamma(Z0), rel=1e-12
        )


def test_lambda_squared_is_pi_alpha(exp):
    """TB26 after Eq. (66): lambda^2 = pi alpha; Eq. (33)/(A6) coefficient 6.54."""
    Z0 = rd.transcribed(exp, "characteristic_impedance").value
    assert rd.lambda_tb_per_gamma(Z0) ** 2 == pytest.approx(
        math.pi * rd.alpha_per_gamma2(Z0), rel=1e-12
    )
    printed = rd.transcribed(exp, "spin_boson_alpha_coefficient").value
    assert rd.alpha_per_gamma2(Z0) == pytest.approx(printed, rel=1e-3)


def test_flux_quantum_identity():
    """phi_0^2/(hbar Z_0) = R_K/(8 pi Z_0): the J23 / FD17 form equals the TB26 form."""
    Z0 = 50.0
    assert rd.PHI0_SI**2 / (HBAR_SI * Z0) == pytest.approx(
        rd.R_K_SI / (8.0 * math.pi * Z0), rel=1e-12
    )


def test_table1_internal_consistency(exp):
    lam = rd.transcribed(exp, "table1_scenario_lambda").value
    gam = rd.transcribed(exp, "table1_scenario_gamma").value
    dOm = rd.transcribed(exp, "table1_scenario_gap_variation").value
    alp = rd.transcribed(exp, "table1_scenario_alpha").value
    Z0 = rd.transcribed(exp, "characteristic_impedance").value
    per_lam = rd.transcribed(exp, "gap_variation_per_lambda").value
    for i in range(1, 6):
        # Table I prints lambda to 1-2 significant figures (-0.1, -0.3, -0.65, -1, 0.1)
        # from -4.53 gamma = -0.0906, -0.317, -0.634, -0.997, +0.0906
        pred = rd.lambda_tb_from_gamma(gam[i], Z0)
        assert abs(lam[i] - pred) <= 0.02 + 0.03 * abs(pred), (i, lam[i], pred)
        assert dOm[i] == pytest.approx(per_lam * lam[i], abs=0.05)
        assert alp[i] == pytest.approx(lam[i] ** 2 / math.pi, rel=0.4)
    # scenario 2 is the one with visible rounding: -0.0906 printed as -0.1
    assert rd.lambda_tb_from_gamma(gam[1], Z0) == pytest.approx(-0.0906, abs=5e-4)


def test_refuses_derived_quantities(exp):
    for name, q in exp.quantities.items():
        if q.extra.get("provenance") == "derived":
            with pytest.raises(ValueError):
                rd.transcribed(exp, name)


# ----------------------------------------- lattice / continuum identities --


def test_lattice_field_is_the_continuum_dimensionless_field():
    """<dphi_0 dphi_r> = (1/2)(K^{1/2})_{0r} = 2/(pi(1-4r^2)) -> -1/(2 pi r^2):
    the continuum <d_x phi d_x phi> with a = 1 and NO stray normalization factor."""
    N = 512
    K = harmonic_chain_K(N, 0.0, bc="periodic")
    w, U = np.linalg.eigh(K)
    w[w < 0] = 0.0
    Khalf = (U * np.sqrt(w)) @ U.T
    k = 2 * np.pi * np.arange(N) / N
    for r in (1, 2, 3, 5, 8, 13):
        lat = 0.5 * Khalf[0, r]
        # exact finite-N mode sum (1/N) sum_k omega_k cos(k r) / 2
        finite_N = 0.5 * np.mean(2.0 * np.abs(np.sin(k / 2)) * np.cos(k * r))
        assert lat == pytest.approx(finite_N, abs=1e-10), (r, lat, finite_N)
        # infinite-chain closed form: the finite-N remainder is a constant absolute
        # offset, +2.0e-6 at N = 512 and proportional to 1/N^2 (1.25e-7 at N = 2048);
        # an absolute pin of 5e-6 is 0.5 % of the r = 13 value and excludes any stray
        # factor of a, 2 or pi in the field normalization
        exact_lattice = 2.0 / (math.pi * (1.0 - 4.0 * r * r))
        assert lat == pytest.approx(exact_lattice, abs=5e-6), (r, lat, exact_lattice)
        continuum = -1.0 / (2.0 * math.pi * r * r)
        assert lat == pytest.approx(continuum * 4 * r * r / (4 * r * r - 1.0), abs=5e-6)


def test_momentum_equals_discrete_derivative_on_massless_chain():
    """V_pp = D V_xx D^T: 'xp' and the paper's d_x coupling see the same field statistics."""
    N = 128
    m = 1e-5
    K = harmonic_chain_K(N, m, bc="periodic")
    V = ground_state_cov(K)
    Vxx, Vpp = V[:N, :N], V[N:, N:]
    D = np.roll(np.eye(N), -1, axis=1) - np.eye(N)  # (D phi)_i = phi_{i+1} - phi_i
    Vdd = D @ Vxx @ D.T
    assert np.allclose(Vdd, Vpp, atol=1e-6, rtol=1e-6), np.abs(Vdd - Vpp).max()
    # and mode by mode: |e^{ik} - 1|^2 = omega_k^2 = 4 sin^2(k/2)
    k = 2 * np.pi * np.arange(N) / N
    assert np.allclose(np.abs(np.exp(1j * k) - 1) ** 2, 4 * np.sin(k / 2) ** 2)


def test_oscillator_matrix_element():
    """<0|x_d^2|0> = |<1|x_d|0>|^2 = 1/(2 Omega) in the repo's H_d = (p^2 + Omega^2 x^2)/2."""
    for Om in (0.146, 1.0, 1.7):
        V = ground_state_cov(np.array([[Om**2]]))
        assert V[0, 0] == pytest.approx(1.0 / (2.0 * Om), rel=1e-12)


def _qubit_excitation(coupling, lam_q, Om, w_f, T, n_max=10):
    """Exact two-level + single-Fock-mode dynamics from |g,0>, sudden switching over [0, T].

    H = Om s+s- + w_f a^dag a + lam_q sigma_x O_f, with O_f = x_f = (a+a^dag)/sqrt(2 w_f)
    for 'xx' and O_f = p_f = i sqrt(w_f/2) (a^dag - a) for 'xp'.
    """
    n = np.arange(n_max)
    a = np.diag(np.sqrt(n[1:]), 1)
    ad = a.T
    if coupling == "xx":
        Of = (a + ad) / math.sqrt(2.0 * w_f)
    else:
        Of = 1j * math.sqrt(w_f / 2.0) * (ad - a)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sp_sm = np.array([[0, 0], [0, 1]], dtype=complex)  # |e><e|, basis (g, e)
    I2, If = np.eye(2), np.eye(n_max)
    H = Om * np.kron(sp_sm, If) + w_f * np.kron(I2, np.diag(n)) + lam_q * np.kron(sx, Of)
    psi0 = np.zeros(2 * n_max, dtype=complex)
    psi0[0] = 1.0
    psi = expm(-1j * H * T) @ psi0
    return float(np.sum(np.abs(psi[n_max:]) ** 2))


def _oscillator_excitation(coupling, lam_osc, Om, w_f, T):
    """<b^dag b> of the repo's oscillator detector attached to a one-mode 'field' (K = w_f^2)
    through attach_detectors, Gaussian evolution from the lam = 0 ground state."""
    K = np.array([[w_f**2]])
    H = attach_detectors(K, [0], [Om], lam_osc, coupling=coupling)
    if coupling == "xx":
        H = np.block([[H, np.zeros((2, 2))], [np.zeros((2, 2)), np.eye(2)]])
    V0 = ground_state_cov(np.diag([w_f**2, Om**2]))
    S = symplectic_from_quadratic(H, T)
    V = evolve(V0, S)
    # R = (x_f, x_d, p_f, p_d): detector quadratures at indices 1 and 3
    return 0.5 * (Om * V[1, 1] + V[3, 3] / Om) - 0.5


@pytest.mark.parametrize("coupling", ["xx", "xp"])
def test_two_level_to_oscillator_factor_is_sqrt_2Omega(coupling):
    """P_e(qubit, lam_q) == <n_d>(oscillator, lam_osc) at O(lam^2) iff lam_osc = lam_q sqrt(2 Omega).

    The two detectors differ at O(lam^4) (the oscillator's higher levels), so the
    ratio is 1 + c lam^2: we check the value at two couplings AND that the residual
    scales as lam^2 (measured: ratio - 1 = 4.3e-4 at lam_q = 0.0025, x4 per doubling).
    """
    Om, w_f, T = 1.7, 1.3, 4.0
    resid = {}
    for lam_q in (0.005, 0.0025):
        pe = _qubit_excitation(coupling, lam_q, Om, w_f, T)
        assert pe > 1e-8
        nd_right = _oscillator_excitation(coupling, rd.lambda_osc_xp(lam_q, Om), Om, w_f, T)
        resid[lam_q] = nd_right / pe - 1.0
        assert abs(resid[lam_q]) < 4e-3 * (lam_q / 0.005) ** 2 * 1.5, (lam_q, pe, nd_right)
        nd_naive = _oscillator_excitation(coupling, lam_q, Om, w_f, T)
        assert nd_naive / pe == pytest.approx(1.0 / (2.0 * Om), rel=1e-2), (pe, nd_naive)
    assert resid[0.005] / resid[0.0025] == pytest.approx(4.0, rel=0.15), resid
    # second-order closed form for the sudden quench, as a sanity anchor
    me2 = 1.0 / (2.0 * w_f) if coupling == "xx" else w_f / 2.0
    p2 = 0.0025**2 * me2 * (2.0 - 2.0 * math.cos((Om + w_f) * T)) / (Om + w_f) ** 2
    assert _qubit_excitation(coupling, 0.0025, Om, w_f, T) == pytest.approx(p2, rel=1e-3)


# ----------------------------------------------------------- the numbers --


def test_pinned_numbers_scenario2(exp):
    r = rd.rederive(exp, 2, "gap", "free")
    assert r["lambda_tb_table"] == -0.1
    assert r["Omega_lat"] == pytest.approx(1.0)
    assert r["a_m"] == pytest.approx(1.2e8 / (2 * math.pi * 7.3e9), rel=1e-12)  # 2.62 mm
    assert r["lambda_osc_xp"] == pytest.approx(0.141421, abs=1e-6), r["lambda_osc_xp"]
    assert r["lambda_osc_xx"] == pytest.approx(0.141421, abs=1e-6)
    assert r["lambda_osc_xp_lo"] == pytest.approx(0.128192, abs=1e-5)
    assert r["invariant"] == pytest.approx(0.01, rel=1e-12)

    c = rd.rederive(exp, 2, "cutoff", "free")
    assert c["Omega_lat"] == pytest.approx(0.146, rel=1e-12)
    assert c["a_m"] == pytest.approx(0.382e-3, rel=2e-3)  # TB26 footnote 2: 0.38 mm
    assert c["lambda_osc_xp"] == pytest.approx(0.054037, abs=1e-6), c["lambda_osc_xp"]
    assert c["lambda_osc_xx"] == pytest.approx(0.0078894, abs=1e-7), c["lambda_osc_xx"]
    assert c["lambda_osc_xp_lo"] == pytest.approx(0.048982, abs=1e-5)
    assert c["invariant"] == pytest.approx(0.01, rel=1e-12)

    g = rd.rederive(exp, 2, "gap", "coupled")
    assert g["Omega_lat"] == pytest.approx(6.8 / 7.3, rel=1e-12)
    assert g["lambda_osc_xp"] == pytest.approx(0.1 * math.sqrt(2 * 6.8 / 7.3), rel=1e-12)


def test_waveguide_speed_cancels(exp):
    base = rd.rederive(exp, 2, "cutoff", "free")
    # Same file, other v: rederive reads v only to report a_m.
    from vacuum.experiments_io import Experiment, Quantity

    q = dict(exp.quantities)
    q["waveguide_speed"] = Quantity(
        value=1.334e8, units="m/s", extra=dict(exp.quantities["waveguide_speed"].extra)
    )
    exp2 = Experiment(
        name=exp.name, path=exp.path, source=exp.source, location=exp.location,
        retrieved=exp.retrieved, transcribed_by=exp.transcribed_by, quantities=q,
    )
    other = rd.rederive(exp2, 2, "cutoff", "free")
    assert other["lambda_osc_xp"] == base["lambda_osc_xp"]
    assert other["lambda_osc_xx"] == base["lambda_osc_xx"]
    assert other["a_m"] != base["a_m"]


def test_all_scenarios_scale_linearly(exp):
    lam = rd.transcribed(exp, "table1_scenario_lambda").value
    for sc in range(2, 7):
        r = rd.rederive(exp, sc, "gap", "free")
        assert r["lambda_osc_xp"] == pytest.approx(math.sqrt(2) * abs(lam[sc - 1]), rel=1e-12)
