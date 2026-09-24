"""Tests for vacuum.detectors.circuit_mapping -- the circuit -> lattice coupling map.

What is checked, in the order of the module docstring's derivation:

* the paper's own coefficients (TB26 Eqs. 33 and 66) reproduce from SI-exact
  constants, and lambda^2 = pi alpha is an identity;
* the dimensional-analysis identity behind Eq. (66): the interaction energy
  scale (phi_0 / l_0) gamma sqrt(hbar Z_0) (omega / v) equals hbar omega |lambda_TB|
  for every omega, v and Z_0 (so no Z_0 / v factor survives the map);
* the field-normalisation claim: the lattice spectral density closed form
  integrates to the harmonic chain's own ground-state <phi_j^2>, and tends
  to the continuum 1 / omega;
* MAP-1 (Gate A, tests/test_gate_crossvalidation.py) is reproduced exactly by
  the derivative transcription, and the 'xx' map differs from the archive's
  identification by exactly the factor Omega_lat -- unity at the operating
  point, not elsewhere (the canary);
* every matching criterion satisfies its Golden-rule identity to roundoff, and
  every map inverts;
* the enumerated alternatives sit where the docstring says (0.980, 0.924,
  0.907, 0.845, 0.906 of the default; band [0.118, 0.148]);
* the uncertainty propagates a file's ``error`` fields (on a FABRICATED
  fixture, clearly marked) and otherwise equals the reading spread;
* the parameter file's derived entries equal the code's values;
* the candidate notebook's anchored and derived modes agree numerically, tag
  the coupling as recorded, and the device-point rows go through the very
  ``run_point`` the surfaces used.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import importlib.util
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from vacuum.core import harmonic_chain_K
from vacuum.detectors import circuit_mapping as cm
from vacuum.experiments_io import HBAR_SI, H_PLANCK_SI, load_experiment

PROPOSAL = "circuit_qed_harvesting/teixido-bonfill_2026_table1"
REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "circuit-qed-noise-thresholds" / "notebook.py"

#: the candidate's synthetic IR mass (lattice units) and the scenario-2 gap shift
FIELD_MASS = 0.2
SHIFTED_GAP = 1.0 - 0.46 / 7.3


@pytest.fixture(scope="module")
def tb():
    return load_experiment(PROPOSAL)


@pytest.fixture(scope="module")
def nb():
    spec = importlib.util.spec_from_file_location("m27_notebook_for_circuit_mapping", NOTEBOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------ the paper's own numbers --


def test_eq66_and_eq33_coefficients_reproduce_from_si_exact_constants(tb):
    """-4.53 (Eq. 66) and 6.54 (Eq. 33) at Z_0 = 50 Ohm, from R_K = h / e^2."""
    Z0 = float(tb.quantities["characteristic_impedance"].value)
    assert Z0 == 50.0
    assert cm.R_K_SI == pytest.approx(25812.807, abs=1e-3)
    k_lam = float(tb.quantities["lambda_per_gamma"].value)  # -4.53 printed
    k_alp = float(tb.quantities["spin_boson_alpha_coefficient"].value)  # 6.54 printed
    assert cm.lambda_tb_from_gamma(1.0, Z0) == pytest.approx(k_lam, abs=0.005), (
        f"Eq. (66) coefficient {cm.lambda_tb_from_gamma(1.0, Z0):.4f} vs printed {k_lam}"
    )
    assert cm.alpha_from_gamma(1.0, Z0) == pytest.approx(k_alp, abs=0.005)
    # lambda^2 = pi alpha is an identity of the two formulas, not a fit
    for g in (0.02, 0.07, 0.14, 0.22, -0.02):
        lam = cm.lambda_tb_from_gamma(g, Z0)
        assert lam**2 / math.pi == pytest.approx(cm.alpha_from_gamma(g, Z0), rel=1e-14)
        assert cm.alpha_from_lambda_tb(lam) == pytest.approx(cm.alpha_from_gamma(g, Z0), rel=1e-14)
        assert cm.lambda_tb_from_alpha(cm.alpha_from_gamma(g, Z0)) == pytest.approx(abs(lam), rel=1e-14)
        assert cm.gamma_from_lambda_tb(lam, Z0) == pytest.approx(g, rel=1e-14)
    # the sign convention: lambda has the opposite sign to gamma (Eq. 66)
    assert cm.lambda_tb_from_gamma(0.02, Z0) < 0.0 < cm.lambda_tb_from_gamma(-0.02, Z0)


def test_scenario_2_readings_are_the_three_published_roundings(tb):
    r = cm.scenario_lambda_readings(tb, 1)
    assert r.scenario == 2
    assert r.table1 == -0.1 and r.gamma == 0.02 and r.alpha_value == 0.003
    assert r.eq66 == pytest.approx(-0.0906448, abs=1e-6)
    assert r.alpha == pytest.approx(-0.0970813, abs=1e-6)
    assert r.mean == pytest.approx(-0.0959087, abs=1e-6)
    assert r.sample_std == pytest.approx(0.0047866, abs=1e-6)
    assert r.sigma_propagated == 0.0  # no error fields on gamma / Z0 in the file
    assert r.sigma == r.sample_std
    # the weak-coupling scenario cannot be mapped
    with pytest.raises(ValueError, match="zero"):
        cm.lambda_osc_from_circuit(tb, scenario_index=0)


# ------------------------------------------ dimensional-analysis identity --


def test_coupling_energy_identity_for_any_omega_v_and_Z0():
    """(phi_0 / l_0) gamma sqrt(hbar Z_0) (omega / v) == hbar omega |lambda_TB|, l_0 = Z_0 / v."""
    for gamma, Z0, v, f in itertools.product(
        (0.02, 0.22), (30.0, 50.0, 80.0), (1.0e8, 1.2e8, 3.0e8), (2.0e9, 7.3e9, 20.0e9)
    ):
        om = 2.0 * math.pi * f
        E = cm.coupling_energy_SI(gamma, Z0, v, om)
        assert E == pytest.approx(HBAR_SI * om * abs(cm.lambda_tb_from_gamma(gamma, Z0)), rel=1e-12)
    # the physical scale at the operating point: 0.66 GHz, 9 % of the 7.3 GHz gap
    E = cm.coupling_energy_SI(0.02, 50.0, 1.2e8, 2.0 * math.pi * 7.3e9)
    assert E / H_PLANCK_SI / 1e9 == pytest.approx(0.0906448 * 7.3, rel=1e-5)


# ------------------------------------------- field normalisation / S(omega) --


@pytest.mark.parametrize("m", [0.2, 0.5])
def test_lattice_spectral_density_integrates_to_the_chain_ground_state_variance(m):
    """int domega S_phi^lat(omega) / 2 pi == <phi_j^2> of the periodic chain of harmonic_chain_K.

    This is the statement that the lattice field IS the canonically
    normalised field of TB26 Eq. (36) (docstring step 3): the closed form
    1 / |sin kappa| is the density of states of exactly that field.
    """
    wmax = math.sqrt(m * m + 4.0)
    val, err = quad(lambda w: cm.amplitude_spectral_density(w, m) / (2.0 * math.pi),
                    m, wmax, limit=400)
    N = 1201
    w2, U = np.linalg.eigh(harmonic_chain_K(N, m, bc="periodic"))
    var = float((U[0] ** 2 / (2.0 * np.sqrt(w2))).sum())
    assert val == pytest.approx(var, rel=2e-6), f"int S/2pi = {val} vs <phi^2> = {var}"


def test_spectral_densities_have_the_right_limits_and_ratios():
    # continuum: S_phi = 1 / omega, S_dphi = omega
    for w in (0.3, 1.0, 1.7):
        assert cm.amplitude_spectral_density(w) == pytest.approx(1.0 / w)
        assert cm.derivative_spectral_density(w) == pytest.approx(w)
        assert cm.derivative_spectral_density(w, operator="finite-difference") == pytest.approx(w)
    # lattice, m = 0: 1 / |sin kappa| -> 1 / omega as omega -> 0 (kappa -> omega)
    assert cm.amplitude_spectral_density(0.01, 0.0) == pytest.approx(100.0, rel=1e-4)
    # the numbers of docstring step 5 at the operating point
    assert cm.amplitude_spectral_density(1.0, FIELD_MASS) == pytest.approx(1.1707323, abs=1e-6)
    assert cm.lattice_wavenumber(1.0, 0.0) == pytest.approx(math.pi / 3.0)
    # 'xp' vs finite-difference weights on the lattice: omega^2 vs omega^2 - m^2
    S = cm.amplitude_spectral_density(1.0, FIELD_MASS)
    assert cm.derivative_spectral_density(1.0, FIELD_MASS, "xp") == pytest.approx(S)
    assert cm.derivative_spectral_density(1.0, FIELD_MASS, "finite-difference") == pytest.approx(
        (1.0 - FIELD_MASS**2) * S
    )
    with pytest.raises(ValueError):
        cm.amplitude_spectral_density(0.1, FIELD_MASS)  # below the mass gap
    with pytest.raises(ValueError):
        cm.amplitude_spectral_density(3.0, FIELD_MASS)  # above the band


# ---------------------------------------------------- MAP-1 and the map --


def test_derivative_transcription_is_map1_exactly():
    for lam, W in itertools.product((-0.1, 0.0906, 1.0), (0.5, 1.0, 2.3)):
        xp = cm.lambda_xp_osc_from_tb(lam, W)
        assert cm.lambda_udw_from_osc(xp, W) == pytest.approx(abs(lam), rel=1e-15)
        assert cm.lambda_osc_from_udw(cm.lambda_udw_from_osc(0.37, W), W) == pytest.approx(0.37, rel=1e-15)
        assert xp == pytest.approx(abs(lam) * math.sqrt(2.0 * W), rel=1e-15)


def test_xx_map_is_the_archive_identification_times_omega_lat():
    """lambda_osc('xx', 'xp' criterion) = |lambda_TB| Omega sqrt(2 Omega): the
    identification |lambda_TB| sqrt(2 Omega) misses exactly one factor Omega_lat,
    which is 1 at the operating point (omega_ref = Omega^0) and nowhere else."""
    for lam in (-0.1, 0.3):
        for W in (0.5, 0.937, 1.0, 2.0):
            derived = cm.lambda_osc_from_tb(lam, W, "xp")
            identification = cm.lambda_osc_from_udw(abs(lam), W)
            assert derived / identification == pytest.approx(W, rel=1e-14)
        assert cm.lambda_osc_from_tb(lam, 1.0, "xp") == pytest.approx(abs(lam) * math.sqrt(2.0), rel=1e-15)
    # the canary: at Omega_lat = 0.5 the two differ by a factor 2
    assert cm.lambda_osc_from_tb(0.1, 0.5, "xp") == pytest.approx(0.5 * cm.lambda_osc_from_udw(0.1, 0.5))


@pytest.mark.parametrize("criterion", cm.CRITERIA)
@pytest.mark.parametrize("W", [0.6, 1.0, 1.4])
def test_golden_rule_identity_for_every_criterion(criterion, W):
    """lambda_osc^2 S_phi(W) / 2 W == the criterion's reference rate, to roundoff."""
    lam = -0.1
    m = FIELD_MASS
    lo = cm.lambda_osc_from_tb(lam, W, criterion, m)
    S = cm.amplitude_spectral_density(W, m)
    rate_xx = cm.golden_rule_rate(lo, W, S)
    if criterion == "xp":
        ref = cm.golden_rule_rate(cm.lambda_xp_osc_from_tb(lam, W), W,
                                  cm.derivative_spectral_density(W, m, "xp"))
        # and identically the continuum-massless physical rate lambda^2 W
        assert cm.golden_rule_rate(lo, W, cm.amplitude_spectral_density(W)) == pytest.approx(
            lam**2 * W, rel=1e-13
        )
    elif criterion == "finite-difference":
        ref = cm.golden_rule_rate(lam, W, cm.derivative_spectral_density(W, m, "finite-difference"),
                                  two_level=True)
    else:  # physical-rate: the device's Gamma_1 = lambda^2 Omega (= pi alpha Omega)
        ref = cm.golden_rule_rate(lam, W, W, two_level=True)
        assert ref == pytest.approx(math.pi * cm.alpha_from_lambda_tb(lam) * W, rel=1e-14)
    assert rate_xx == pytest.approx(ref, rel=1e-13), criterion
    # the inverse map
    assert cm.lambda_tb_from_osc(lo, W, criterion, m) == pytest.approx(abs(lam), rel=1e-13)
    inv = cm.circuit_from_lambda_osc(lo, W, 50.0, criterion, m)
    assert inv["lambda_tb_abs"] == pytest.approx(abs(lam), rel=1e-13)
    assert inv["gamma_abs"] == pytest.approx(abs(cm.gamma_from_lambda_tb(lam, 50.0)), rel=1e-13)
    assert inv["alpha"] == pytest.approx(cm.alpha_from_lambda_tb(lam), rel=1e-13)


def test_lattice_criteria_need_the_mass_and_reject_a_gap_below_it():
    with pytest.raises(ValueError, match="field_mass"):
        cm.mapping_factor(1.0, "physical-rate")
    with pytest.raises(ValueError):
        cm.mapping_factor(0.1, "finite-difference", FIELD_MASS)
    with pytest.raises(ValueError, match="criterion"):
        cm.mapping_factor(1.0, "nonsense")
    assert cm.mapping_factor(1.0, "xp", FIELD_MASS) == cm.mapping_factor(1.0, "xp")


def test_alternatives_sit_where_the_derivation_says(tb):
    """The enumerated criteria land at the ratios the derivation predicts, and
    the uncertainty is an independently computed number -- not the code's own
    expression fed back to it."""
    r = cm.lambda_osc_from_circuit(tb, gap_lat=1.0, field_mass=FIELD_MASS, shifted_gap_lat=SHIFTED_GAP)
    assert r.criterion == "xp" and r.gap_lat == 1.0 and r.lambda_tb == -0.1
    assert r.lambda_osc == pytest.approx(math.sqrt(2.0) * 0.1, rel=1e-15)
    # NOT sigma == M * sample_std (that restates the code): an independently
    # computed number, and the invariant that linear propagation preserves the
    # RELATIVE uncertainty.  std([-0.1, -0.0906448, -0.0970813], ddof=1) x sqrt(2)
    assert r.sigma == pytest.approx(0.00676926, abs=1e-8)
    assert r.sigma / r.lambda_osc == pytest.approx(r.sigma_tb / abs(r.lambda_tb), rel=1e-14)
    assert r.lambda_udw == pytest.approx(0.1, rel=1e-15)
    a = r.alternatives
    base = r.lambda_osc
    assert a["identification_map1_only"] == pytest.approx(base, rel=1e-15)
    assert a["xp@free_gap"] == pytest.approx(base, rel=1e-15)
    assert a["finite-difference@free_gap"] / base == pytest.approx(0.980, abs=1e-3)
    assert a["physical-rate@free_gap"] / base == pytest.approx(0.924, abs=1e-3)
    assert a["xp@shifted_gap"] / base == pytest.approx(0.907, abs=1e-3)
    assert a["physical-rate@shifted_gap"] / base == pytest.approx(0.845, abs=1e-3)
    assert a["reading_eq66@default"] / base == pytest.approx(0.906, abs=1e-3)
    assert a["reading_sqrt_pi_alpha@default"] / base == pytest.approx(0.971, abs=1e-3)
    assert a["lower_envelope(eq66_reading,physical-rate)"] == pytest.approx(0.1185, abs=1e-3)
    lo, hi = r.band
    assert lo == pytest.approx(0.1185, abs=1e-3) and hi == pytest.approx(0.1482, abs=1e-3)
    d = r.as_dict()
    assert d["formula"] and d["inputs"]["Z0"]["quantity"] == "characteristic_impedance"
    # without a mass only the mass-free alternatives are offered
    r0 = cm.lambda_osc_from_circuit(tb)
    assert "physical-rate@free_gap" not in r0.alternatives and r0.lambda_osc == r.lambda_osc


def test_uncertainty_propagates_error_fields_when_a_file_carries_them(tb):
    """FABRICATED fixture: the shipped file with error fields ADDED to gamma and Z_0.

    No published error bar exists for either (the file quotes both with '~');
    the numbers below are chosen to make the propagation hand-checkable and
    describe no experiment.
    """
    q = {k: {"value": v.value, "units": v.units, "error": v.error} for k, v in tb.quantities.items()}
    fx = copy.deepcopy(q)
    fx["table1_scenario_gamma"]["error"] = [0.0, 0.002, 0.0, 0.0, 0.0, 0.0]
    fx["characteristic_impedance"]["error"] = 5.0
    r = cm.scenario_lambda_readings(fx, 1)
    coef = math.sqrt(cm.R_K_SI / (8.0 * math.pi * 50.0))
    expect = math.hypot(coef * 0.002, abs(r.eq66) / 100.0 * 5.0)
    assert r.sigma_propagated == pytest.approx(expect, rel=1e-12)
    assert r.sigma == pytest.approx(math.hypot(r.sample_std, expect), rel=1e-12)
    res = cm.lambda_osc_from_circuit(fx, gap_lat=1.0)
    assert res.sigma == pytest.approx(math.sqrt(2.0) * r.sigma, rel=1e-12)
    # explicit overrides win
    res2 = cm.lambda_osc_from_circuit(tb, gap_lat=1.0, lambda_tb=-0.0906, sigma_tb=0.01)
    assert res2.lambda_osc == pytest.approx(0.0906 * math.sqrt(2.0)) and res2.sigma == pytest.approx(0.01 * math.sqrt(2.0))


def test_parameter_file_derived_entries_equal_the_code(tb):
    q = tb.quantities
    lt = q["scenario2_coupling_lambda_tb"]
    lo = q["oscillator_coupling_lambda_osc_scenario2"]
    assert lt.extra["provenance"] == "derived" and lo.extra["provenance"] == "derived"
    r = cm.scenario_lambda_readings(tb, 1)
    assert float(lt.value) == r.table1
    assert float(lt.error) == pytest.approx(r.sigma, rel=1e-12)
    res = cm.lambda_osc_from_circuit(tb, gap_lat=1.0)
    assert float(lo.value) == pytest.approx(res.lambda_osc, rel=1e-14)
    assert float(lo.error) == pytest.approx(res.sigma, rel=1e-12)
    for name in ("scenario2_coupling_lambda_tb", "oscillator_coupling_lambda_osc_scenario2"):
        assert name in tb.extras["derived_quantities"]
        for ref in q[name].extra["derived_from"].split(","):
            assert ref.strip() in q, ref
    assert "circuit_mapping" in lo.note and "Eq. (66)" in lt.note


# ------------------------------------------------- the candidate notebook --


def test_notebook_anchored_and_derived_modes_agree_and_are_tagged_as_recorded(nb):
    ia = nb.load_inputs()
    idd = nb.load_inputs(coupling="derived")
    assert ia.point.coupling_mode == "anchored" and idd.point.coupling_mode == "derived"
    # the same number (Omega_lat = 1), a different status
    assert ia.point.coupling_lambda == pytest.approx(idd.point.coupling_lambda, rel=1e-15)
    # the formula at the published inputs (a change-detector), AND an
    # independent round-trip: the alpha recovered from lam_osc must reproduce
    # Table I's OWN alpha column, which never enters the forward map
    assert idd.point.coupling_lambda == pytest.approx(math.sqrt(2.0) * 0.1, rel=1e-15)
    back = cm.circuit_from_lambda_osc(idd.point.coupling_lambda, 1.0)
    tb_alpha = float(idd.experiments[nb.PROPOSAL].quantities["table1_scenario_alpha"].value[1])
    assert back["alpha"] == pytest.approx(tb_alpha, rel=0.07), (back["alpha"], tb_alpha)
    assert ia.point.coupling_sigma == 0.0
    assert idd.point.coupling_sigma == pytest.approx(math.sqrt(2.0) * 0.0047866, abs=1e-6)
    assert ia.provenance["coupling_lambda"][1] == "unsourced"  # the archive's record
    assert ia.provenance["coupling_lambda_derived"][1] == "derived"
    assert idd.provenance["coupling_lambda"][1] == "derived"
    assert idd.provenance["coupling_lambda"][0].endswith("teixido-bonfill_2026_table1")
    assert "coupling_lambda" in ia.synthetic_inputs
    assert "coupling_lambda" not in idd.synthetic_inputs
    assert set(idd.synthetic_inputs) == set(ia.synthetic_inputs) - {"coupling_lambda"}
    assert idd.coupling_map["lambda_osc"] == idd.point.coupling_lambda
    assert idd.anchors["coupling_lambda"] == {"derived": idd.point.coupling_lambda}
    assert "not a derivation" in nb.COUPLING_STATUS and "derived" in nb.COUPLING_DERIVED_STATUS
    with pytest.raises(ValueError, match="coupling"):
        nb.load_inputs(coupling="guessed")


def test_device_point_specs_cover_couplings_temperatures_and_anchors(nb):
    inputs = nb.load_inputs(coupling="derived")
    couplings = nb.device_point_couplings(inputs)
    anchors = nb.device_point_anchors(inputs)
    assert set(couplings) == set(nb.DEVICE_POINT_COUPLINGS)
    assert couplings["c0"] == inputs.point.coupling_lambda
    assert couplings["c-1s"] < couplings["c0"] < couplings["c+1s"]
    assert couplings["cLE"] < couplings["c66"] < couplings["c0"]
    assert couplings["cLE"] < couplings["cPR"] < couplings["c0"]
    assert set(anchors) == {"Gphi_min", "Gphi_max", "G1_min", "G1_max", "d19.2", "d60", "d120",
                            "G1min+Gphimin", "G1max+Gphimax"}
    assert anchors["d19.2"][1] == {"separation_mm": 19.2}
    assert anchors["G1max+Gphimax"][1] == {"gamma1_MHz": 1850.0, "gammaphi_Hz": 2e8}
    specs = nb.device_point_specs(inputs, couplings, anchors)
    assert len(specs) == len(couplings) * len(nb.DEVICE_POINT_T_MK) * len(anchors)
    labels = [s.label for s in specs]
    assert len(set(labels)) == len(labels) and max(len(l) for l in labels) <= 64
    assert all(s.sweep == "device_point" for s in specs)
    assert {s.coupling_lambda for s in specs} == set(couplings.values())
    assert {s.temperature_mK for s in specs} == set(nb.DEVICE_POINT_T_MK)
    # every spec keeps the operating point's gap schedule and window
    assert all(s.gap_shift_gamma == inputs.point.gap_shift_gamma for s in specs)
    assert all(s.window_ns == inputs.point.window_ns for s in specs)


def test_device_point_row_goes_through_run_point_and_summarises(nb):
    """One device-point row at the quick gate settings, through the surfaces' run_point."""
    inputs = nb.load_inputs(coupling="derived")
    couplings = {"c0": inputs.point.coupling_lambda}
    anchors = {"G1_min": nb.device_point_anchors(inputs)["G1_min"]}
    specs = nb.device_point_specs(inputs, couplings, anchors, T_values=(30.0,), **nb.QUICK_COMMON)
    assert len(specs) == 1 and specs[0].label == "dp-c0-T30-G1_min"
    row = nb.run_point(specs[0], inputs)
    assert set(row) == set(nb.ROW_COLUMNS)
    assert row["coupling_lambda"] == pytest.approx(math.sqrt(2.0) * 0.1, rel=1e-15)
    assert row["lambda_udw"] == pytest.approx(0.1, rel=1e-12)
    # independent: the row's coupling maps back onto Table I's published alpha
    assert cm.circuit_from_lambda_osc(row["coupling_lambda"], row["gap_lat"])["alpha"] == (
        pytest.approx(0.003, rel=0.07))
    assert row["gamma1_MHz"] == 8.7 and row["temperature_mK"] == 30.0
    assert row["all_audits_passed"] and row["converged"]
    summary = nb.device_point_summary([row], inputs, couplings, anchors, archive_rows=None,
                                      wall_clock_s=0.0)
    e = summary["per_coupling"]["c0"]
    assert e["n_rows"] == 1 and e["rows"][0]["anchor"] == "G1_min"
    assert e["device_box_30_50mK"]["max_E_N"] == row["E_N"]
    for k, v in inputs.resolution.levels.items():
        assert e["rows"][0]["verdict"][k] == ("GO" if row["E_N"] > v else "NO-GO")
    assert summary["archive_cross_check"] == {"available": False}
    assert summary["operating_point"]["coupling_mode"] == "derived"
    assert summary["verdict"]["derived_coupling_device_box"]["n_rows"] == 1
    assert "coupling_lambda" not in summary["synthetic_inputs_remaining"]
    assert {"field_mass", "boundary_buffer"} <= set(summary["synthetic_inputs_remaining"])


# --------------------------------------- the derivative ('xp') coupling --


def test_coupling_schedule_tracks_the_gap(nb):
    inputs = nb.load_inputs(coupling="derived")
    u = inputs.units
    W = u.time(inputs.point.window_ns * 1e-9)
    chi = nb.switching_profile("cos2", W)
    shift = u.energy(inputs.point.gap_shift_GHz * 1e9)
    gaps = nb._gaps_of_t(1.0, shift, chi)
    lam = inputs.point.coupling_lambda
    f0 = nb.coupling_schedule(lam, chi, gaps, 1.0, 0.0)
    f5 = nb.coupling_schedule(lam, chi, gaps, 1.0, 0.5)
    f15 = nb.coupling_schedule(lam, chi, gaps, 1.0, 1.5)
    for t in (0.0, 0.25 * W, 0.5 * W, W):
        assert f0(t) == pytest.approx(lam * chi(t), rel=1e-15)
        ratio = gaps(t)[0] / 1.0
        assert f5(t) == pytest.approx(lam * chi(t) * ratio**0.5, rel=1e-14)
        assert f15(t) == pytest.approx(lam * chi(t) * ratio**1.5, rel=1e-14)
    # at full switch-on the tracked 'xp' coupling is 0.968x, the 'xx' one 0.907x
    assert f5(0.5 * W) / f0(0.5 * W) == pytest.approx((1.0 - 0.46 / 7.3) ** 0.5, rel=1e-12)
    assert f15(0.5 * W) / f0(0.5 * W) == pytest.approx((1.0 - 0.46 / 7.3) ** 1.5, rel=1e-12)
    assert nb.GAP_TRACKING_EXPONENT == {"xx": 1.5, "xp": 0.5}


def test_xp_device_point_row_and_two_driver_check(nb):
    """An 'xp' device-point row through run_point, and the static-gap 'xp' point
    through both drivers (run_protocol vs run_harvesting) with the derivative
    perturbative companion populated."""
    inputs = nb.load_inputs(coupling="derived")
    lam = inputs.point.coupling_lambda
    anchors = {"G1_min": nb.device_point_anchors(inputs)["G1_min"]}
    spec = nb.device_point_specs(inputs, {"c0": lam}, anchors, T_values=(30.0,), coupling="xp",
                                 **nb.QUICK_COMMON)[0]
    assert spec.coupling == "xp" and spec.gap_tracking_exponent == 0.0
    row = nb.run_point(spec, inputs)
    assert set(row) == set(nb.ROW_COLUMNS)
    assert row["coupling"] == "xp" and row["gap_tracking_exponent"] == 0.0
    assert row["all_audits_passed"] and row["converged"] and abs(row["ledger_defect"]) < 1e-9
    static = nb.base_spec(inputs, "xp-static", "split", coupling="xp", gap_shift_gamma=0.0,
                          **nb.QUICK_COMMON)
    r = nb.run_point(static, inputs)
    assert r["has_perturbative_row"] and r["perturbative_ok"]
    assert 0.0 < r["E_N"] and 0.0 < r["E_N_pert"]
    assert r["comm_fraction"] > 0.9  # the derivative model at the light-cone edge is communication-dominated
    x = nb.cross_check_run_harvesting(static, inputs)
    assert x["coupling"] == "xp"
    assert abs(x["E_N_run_harvesting"] - r["E_N"]) < 1e-6, (x["E_N_run_harvesting"], r["E_N"])
    # the amplitude row at the same point is the archive's model: different number
    r_xx = nb.run_point(replace(static, coupling="xx", label="xx-static"), inputs)
    assert r_xx["E_N"] > 2.0 * r["E_N"]


def test_xp_map_and_gap_tracking_specs_structure(nb):
    inputs = nb.load_inputs(coupling="derived")
    couplings = nb.device_point_couplings(inputs)
    anchors = nb.device_point_anchors(inputs)
    specs = nb.xp_map_specs(inputs, couplings, anchors, grid=12)
    n_surface = 3 * len(nb.T_GRID_12_MK) * len(nb.GPHI_GRID_12_HZ)
    n_dp = 3 * len(nb.DEVICE_POINT_T_MK) * len(anchors)
    assert len(specs) == n_surface + n_dp + len(nb.SEP_GRID_SITES) + 3
    assert all(s.coupling == "xp" for s in specs)
    labels = [s.label for s in specs]
    assert len(set(labels)) == len(labels) and max(len(l) for l in labels) <= 64
    assert {s.coupling_lambda for s in specs if s.sweep == "dephasing"} == {
        couplings[c] for c in nb.XP_MAP_COUPLINGS}
    assert all(s.coupling_lambda == couplings["c0"] for s in specs if s.sweep in ("split", "control"))
    # the reduced axes keep every published anchor
    for T in inputs.anchors["temperature_mK"].values():
        assert T in nb.T_GRID_12_MK
    for g in (inputs.anchors["gammaphi_Hz"]["digitized_min"], inputs.anchors["gammaphi_Hz"]["digitized_max"]):
        assert g in nb.GPHI_GRID_12_HZ
    full = nb.xp_map_specs(inputs, couplings, anchors, grid=0)
    assert len(full) == 3 * 256 + n_dp + len(nb.SEP_GRID_SITES) + 3
    with pytest.raises(ValueError, match="grid"):
        nb.xp_map_specs(inputs, couplings, anchors, grid=7)
    gt = nb.gap_tracking_specs(inputs, anchors, couplings["c0"])
    assert len(gt) == 2 * len(nb.DEVICE_POINT_T_MK) * len(anchors)
    assert {(s.coupling, s.gap_tracking_exponent) for s in gt} == {("xx", 1.5), ("xp", 0.5)}
    assert all(s.coupling_lambda == couplings["c0"] for s in gt)


def test_ir_control_specs_and_summary_structure(nb):
    """The IR-mass control sweeps the declared regulator for both operators."""
    inputs = nb.load_inputs(coupling="derived")
    lam = nb.device_point_couplings(inputs)["c0"]
    anchors = {"G1_min": nb.device_point_anchors(inputs)["G1_min"],
               "Gphi_min": nb.device_point_anchors(inputs)["Gphi_min"]}
    assert nb.FIELD_MASS in nb.IR_MASS_GRID, "the declared mass must be on the control grid"
    specs = nb.ir_control_specs(inputs, anchors, lam, masses=(0.1, 0.2), T_values=(30.0,))
    assert len(specs) == 2 * 2 * 1 * len(anchors)
    assert {s.field_mass for s in specs} == {0.1, 0.2}
    assert {s.coupling for s in specs} == {"xx", "xp"}
    assert all(s.coupling_lambda == lam for s in specs)
    rows = [nb.run_point(replace(s, **nb.QUICK_COMMON), inputs) for s in specs]
    summary = nb.ir_control_summary(rows, inputs, anchors, lam, wall_clock_s=0.0,
                                    masses=(0.1, 0.2))
    assert summary["declared_mass"] == nb.FIELD_MASS
    for coupling in ("xx", "xp"):
        e = summary["per_coupling"][coupling]
        assert set(e["by_mass"]) == {"0.1", "0.2"}
        assert set(e["verdicts_independent_of_mass"]) == set(inputs.resolution.levels)
        # the dephasing anchor is the one the device statement rests on
        assert e["dephasing_anchors_all_zero_at_every_mass"] is True
    # every stored row carries the mass it was run at
    for s_, r in zip(specs, rows):
        assert r["field_mass"] == s_.field_mass and r["coupling"] == s_.coupling


# ------------------------------------------- the lattice UV-cutoff bracket --


def test_lattice_at_spacing_is_the_identity_at_a_equals_one(nb):
    """a = 1 must reproduce the archives' construction exactly, or the whole
    cutoff bracket is a different model rather than a control."""
    K1, rescale, cut = nb.lattice_at_spacing(24, FIELD_MASS, 1.0)
    K0 = harmonic_chain_K(24, FIELD_MASS, bc="dirichlet")
    assert np.array_equal(K1, K0), "a = 1 must give harmonic_chain_K bit-for-bit"
    assert rescale == 1.0
    assert cut == pytest.approx(math.sqrt(FIELD_MASS**2 + 4.0))
    assert nb.geometry(7, 10.09, 6, 1.0) == nb.geometry(7, 10.09, 6)
    with pytest.raises(ValueError):
        nb.lattice_at_spacing(8, FIELD_MASS, 0.0)


@pytest.mark.parametrize("a", [1.0, 0.6, 0.4, 0.29209, 0.2])
def test_brillouin_edge_is_the_model_cutoff(nb, a):
    """The chain's largest eigenfrequency is sqrt(m^2 + 4/a^2) — the quantity
    the bracket calls the UV cutoff — and it lands on 50 GHz at a = 0.292."""
    N = 64  # even, periodic: kappa = pi is on the grid, so the edge is attained
    K, _r, cut = nb.lattice_at_spacing(N, FIELD_MASS, a, bc="periodic")
    omega_max = float(np.sqrt(np.linalg.eigvalsh(K).max()))
    assert omega_max == pytest.approx(cut, rel=1e-12)
    assert cut == pytest.approx(math.sqrt(FIELD_MASS**2 + 4.0 / a**2), rel=1e-14)
    if a == 0.29209:
        assert cut * 7.3 == pytest.approx(50.0, abs=0.05)  # the device's line cutoff


def test_field_renormalisation_reproduces_the_continuum_correlator():
    """√a is the right field normalisation: the PHYSICAL correlator
    ⟨φ(0)φ(r)⟩ = ⟨x_i x_j⟩ / a at fixed physical r must converge, as a → 0, to
    the 1+1 massive continuum value K_0(m r) / 2π.

    This is the one check that the cutoff bracket varies the resolution and
    not the physics: get the √a wrong and these numbers diverge or vanish.
    """
    from scipy.special import k0
    m, r, L = 0.5, 2.0, 32.0
    exact = float(k0(m * r) / (2.0 * math.pi))
    vals = []
    for a in (0.5, 0.25, 0.125):
        N = int(round(L / a))
        Kd = harmonic_chain_K(N, m * a, bc="periodic")   # K_a = Kd / a^2
        w, U = np.linalg.eigh(Kd)
        inv_sqrt = (U * (w ** -0.25)) @ (U * (w ** -0.25)).T   # Kd^{-1/2}
        # ⟨x_i x_j⟩ = (a/2)(Kd^{-1/2})_ij, and φ = x/√a ⇒ ⟨φφ⟩ = (1/2)(Kd^{-1/2})_ij
        n = int(round(r / a))
        vals.append(0.5 * float(inv_sqrt[0, n]))
    errs = [abs(v - exact) / exact for v in vals]
    assert errs[0] < 0.2, f"a = 0.5 already off by {errs[0]:.3f} (values {vals}, exact {exact})"
    assert errs[-1] < errs[0], f"not converging: {errs}"
    assert errs[-1] < 0.02, f"a = 0.125 still {errs[-1]:.4f} from the continuum {exact}"


def test_geometry_holds_the_physical_buffer_fixed(nb):
    """Halving the spacing must double the site counts, leaving the physical
    separation, buffer and chain length unchanged."""
    W, buf = 10.09, 6
    required = 0.5 * W + buf  # causal half-window + the declared buffer, in physical units
    excess = []
    for a in (1.0, 0.5, 0.25, 0.125):
        d_sites = max(1, int(round(7.0 / a)))
        N, (sA, sB) = nb.geometry(d_sites, W, buf, a)
        sep, buffer_left, length = (sB - sA) * a, sA * a, N * a
        assert sep == pytest.approx(7.0, abs=0.5 * a)          # the sourced 19.2 mm
        assert length == pytest.approx(sep + 2.0 * buffer_left, rel=1e-12)
        # never below the causal requirement, never more than one site above it:
        # the geometry is the same physical box at every resolution
        assert required <= buffer_left < required + a, (a, buffer_left, required)
        excess.append(buffer_left - required)
    assert excess == sorted(excess, reverse=True), f"excess should shrink with a: {excess}"
    assert nb.geometry(7, W, buf, 1.0)[0] == 31  # the archives' chain


def test_cutoff_specs_bracket_the_device_and_run_point_honours_the_spacing(nb):
    inputs = nb.load_inputs(coupling="derived")
    lam = nb.device_point_couplings(inputs)["c0"]
    anchors = nb.device_point_anchors(inputs)
    cuts = [math.sqrt(FIELD_MASS**2 + 4.0 / a**2) * 7.3 for a in nb.CUTOFF_SPACINGS.values()]
    assert min(cuts) < 14.7 + 0.1 and max(cuts) > 70.0
    assert min(cuts) <= 50.0 <= max(cuts), "the bracket must contain the device's 50 GHz"
    assert set(nb.DEPHASING_ANCHOR_KEYS) <= set(nb.CUTOFF_ANCHOR_KEYS)
    assert all("Gphi" in k for k in nb.DEPHASING_ANCHOR_KEYS)
    specs = nb.cutoff_specs(inputs, lam, anchors, spacings={"a1.0": 1.0, "a0.6": 0.6})
    assert {s.lattice_spacing for s in specs} == {1.0, 0.6}
    assert all(s.coupling == "xp" for s in specs)
    n_anchor = 2 * len(nb.CUTOFF_T_MK) * len(nb.CUTOFF_ANCHOR_KEYS)
    assert sum(s.sweep == "device_point" for s in specs) == n_anchor
    # a finer lattice really is a finer lattice, at the same physical point
    base = [s for s in specs if s.label == "dp-a1.0-T30-Gphi_min"][0]
    fine = replace(base, lattice_spacing=0.5, **nb.QUICK_COMMON)
    r1 = nb.run_point(replace(base, **nb.QUICK_COMMON), inputs)
    r2 = nb.run_point(fine, inputs)
    assert r2["n_field_sites"] > 1.8 * r1["n_field_sites"]
    assert r2["uv_cutoff_GHz"] == pytest.approx(2.0 * r1["uv_cutoff_GHz"], rel=0.02)
    assert r1["separation_mm"] == r2["separation_mm"]      # same physical point
    assert r1["gammaphi_Hz"] == r2["gammaphi_Hz"]          # same physical rate
    assert r1["E_N"] == 0.0 and r2["E_N"] == 0.0           # the negative statement


def test_a_equals_one_reproduces_the_fourth_pass_archive_exactly(nb):
    """The fifth pass ADDED the lattice-spacing machinery; this asserts it is
    inert at a = 1, which is why the fourth-pass archives were not regenerated.

    Skips if the archive is absent (a fresh checkout); when it is there, the
    agreement must be EXACT — a single ulp of drift would mean the cutoff
    bracket changed the model rather than its resolution.
    """
    path = nb.DATA_DIR / "rows_xp.npz"
    if not path.exists():
        pytest.skip(f"archive {path} not present")
    archive = {r["label"]: r for r in nb.load_rows(path)}
    inputs = nb.load_inputs(coupling="derived")
    lam = nb.device_point_couplings(inputs)["c0"]
    anchors = nb.device_point_anchors(inputs)
    checked = 0
    for key in ("Gphi_min", "G1_min", "d19.2"):
        spec = nb.device_point_specs(inputs, {"c0": lam}, {key: anchors[key]},
                                     T_values=(30.0,), coupling="xp")[0]
        ref = archive.get(spec.label)
        if ref is None:
            continue
        assert spec.lattice_spacing == 1.0
        row = nb.run_point(spec, inputs)
        assert row["E_N"] == ref["E_N"], f"{spec.label}: {row['E_N']!r} != {ref['E_N']!r}"
        assert row["MI"] == ref["MI"], spec.label
        assert row["n_field_sites"] == ref["n_field_sites"] == 31.0
        assert row["uv_cutoff_GHz"] == pytest.approx(14.67, abs=0.01)
        checked += 1
    assert checked >= 2, f"only {checked} archive rows matched by label"
