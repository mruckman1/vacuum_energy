"""Tests for vacuum.experiments_io — the experiments loader (M-I, item I.3).

Covers: (1) round-trip of every shipped experiments/ file, (2) load-time
ValueError refusal on every missing mandatory provenance field, and
(3) the hbar = 1 unit-conversion factors, unit-tested on a FABRICATED
fixture (clearly marked synthetic) whose round numbers make every factor
hand-checkable, plus one cross-check against a published number (the
thermal occupation quoted by Wilson et al., Nature 479, 376 (2011)), and
(4) provenance honesty of the circuit_qed_harvesting/ platform: every
quantity is provenance-tagged, the set of synthetic-tagged quantities in
every file equals exactly the documented list, derived quantities name
real inputs and reproduce under their stated arithmetic, and the
negativity-resolution field the go/no-go boundary is built on loads and
traces back to its sources.
"""

import copy
import json
import math

import pytest

from vacuum import experiments_io as xio

# The three platform fixture files pinned by docs/PLAN_LAYERS_2_3.md (M-I I.3).
CANONICAL_FILES = (
    "circuit_qed_harvesting/teixido-bonfill_2026_table1",
    "dce_microwave/wilson_2011_table1",
    "ibm_qet/ikeda_2023_fig2",
)

# Every circuit_qed_harvesting/ file and the EXACT set of quantities each one is
# allowed to carry as synthetic (gap-closure pass, 2026-09-01).  Adding a synthetic
# number anywhere else must fail here until it is documented.
CIRCUIT_QED_FILES = {
    "circuit_qed_harvesting/teixido-bonfill_2026_table1": frozenset(),
    "circuit_qed_harvesting/teixido-bonfill_2026_table1_variant-switching": frozenset(),
    "circuit_qed_harvesting/teixido-bonfill_2025_thesis-outlook": frozenset(),
    "circuit_qed_harvesting/janzen_2023_fig6": frozenset(),
    "circuit_qed_harvesting/janzen_2023_fig6_variant-noise-sweep": frozenset(
        {"waveguide_temperature_grid", "relaxation_rate_grid", "pure_dephasing_rate_grid"}
    ),
    "circuit_qed_harvesting/ren_2022_sec4-3": frozenset(),
    "circuit_qed_harvesting/orgiazzi_2016_fig3": frozenset(),
    "circuit_qed_harvesting/dicarlo_2009_fig3": frozenset(),
}
PROVENANCE_TAGS = frozenset(
    {"transcribed", "plot_digitized", "selection", "derived", "synthetic"}
)
PROPOSAL = "circuit_qed_harvesting/teixido-bonfill_2026_table1"
VARIANT_SWITCH = "circuit_qed_harvesting/teixido-bonfill_2026_table1_variant-switching"
DEVICE = "circuit_qed_harvesting/janzen_2023_fig6"
VARIANT_NOISE = "circuit_qed_harvesting/janzen_2023_fig6_variant-noise-sweep"
REN = "circuit_qed_harvesting/ren_2022_sec4-3"
DICARLO = "circuit_qed_harvesting/dicarlo_2009_fig3"

# FABRICATED synthetic fixture — round numbers, corresponds to no experiment.
SYNTHETIC_FIXTURE = {
    "source": "SYNTHETIC — fabricated inside tests/test_experiments_io.py, not from any publication",
    "location": "tests/test_experiments_io.py (fabricated fixture)",
    "retrieved": "2026-08-31",
    "transcribed_by": "vacuum-program build agent (test fixture)",
    "synthetic": True,
    "comment": "FABRICATED fixture with round values chosen so every hbar=1 conversion factor is hand-checkable; the numbers describe no experiment.",
    "quantities": {
        "freq": {"value": 1.0, "units": "GHz"},
        "temp": {"value": 50.0, "units": "mK", "error": 5.0},
        "time": {"value": 100.0, "units": "ns"},
        "length": {"value": 3.0, "units": "mm"},
        "speed": {"value": 1.2e8, "units": "m/s"},
        "fraction": {"value": 10.0, "units": "percent"},
        "plain": {"value": 0.05, "units": "dimensionless"},
        "band": {"value": [4.0, 6.0], "units": "GHz"},
        "resistance": {"value": 218.0, "units": "ohm"},
    },
}

OMEGA_REF = 2.0 * math.pi * 1e9  # lattice energy unit = h * 1 GHz (angular)
V_WAVE = 1.2e8  # m/s, fabricated propagation speed


def write_fixture(tmp_path, data, name="synthetic_fixture.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


@pytest.fixture(scope="module")
def shipped():
    """All shipped experiments/ files, loaded once."""
    names = xio.list_experiments()
    return {name: xio.load_experiment(name) for name in names}


# ------------------------------------------------------------- round-trip --


def test_all_three_canonical_files_ship():
    names = xio.list_experiments()
    for want in CANONICAL_FILES:
        assert want in names, f"missing canonical fixture file {want} (found: {names})"


def test_roundtrip_all_shipped_files(shipped):
    assert len(shipped) >= 3
    for name, exp in shipped.items():
        assert exp.source.strip(), name
        assert exp.location.strip(), name
        assert exp.retrieved.strip(), name
        assert exp.transcribed_by.strip(), name
        assert len(exp.quantities) > 0, name
        for qname, q in exp.quantities.items():
            assert isinstance(q.units, str) and q.units, f"{name}:{qname}"
            assert xio.unit_kind(q.units), f"{name}:{qname}"


def test_shipped_spot_values(shipped):
    wilson = shipped["dce_microwave/wilson_2011_table1"]
    assert wilson.quantities["pump_frequency_fig2c"].value == pytest.approx(11.30)
    assert wilson.quantities["pump_frequency_fig2c"].units == "GHz"
    assert wilson.quantities["sample_temperature"].value == pytest.approx(50)
    assert wilson.quantities["sample_temperature"].units == "mK"

    ikeda = shipped["ibm_qet/ikeda_2023_fig2"]
    assert ikeda.quantities["h"].value == 1
    assert ikeda.quantities["k"].value == 1
    assert ikeda.quantities["shots"].value == 100000
    assert ikeda.quantities["analytical_E1"].value == pytest.approx(-0.1147)
    assert ikeda.extras["device"]["backend_fig2B"] == "ibm_cairo"

    waterloo = shipped["circuit_qed_harvesting/teixido-bonfill_2026_table1"]
    assert waterloo.quantities["qubit_gap_frequency"].value == pytest.approx(7.3)
    assert waterloo.quantities["qubit_gap_frequency"].units == "GHz"
    assert waterloo.quantities["waveguide_speed"].value == pytest.approx(1.2e8)
    assert waterloo.quantities["waveguide_speed"].units == "m/s"
    # Table I and the Sec. VI.A fixed parameters (transcribed 2026-09-01)
    for col in ("lambda", "gamma", "alpha", "gap_variation"):
        assert len(waterloo.quantities[f"table1_scenario_{col}"].value) == 6, col
    assert waterloo.quantities["signaling_time_td"].value == 1
    assert waterloo.quantities["signaling_time_td"].units == "ns"
    assert waterloo.quantities["detector_separation_d"].value == 12
    assert waterloo.quantities["detector_separation_d"].units == "cm"


def test_load_by_name_suffix_and_path(shipped):
    a = xio.load_experiment("dce_microwave/wilson_2011_table1")
    b = xio.load_experiment("dce_microwave/wilson_2011_table1.json")
    c = xio.load_experiment(a.path)  # absolute path
    assert a.quantities["sample_temperature"].value == b.quantities[
        "sample_temperature"
    ].value
    assert a.source == b.source == c.source


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        xio.load_experiment("dce_microwave/does_not_exist_2011_table9")


# ------------------------------------------------- provenance enforcement --


@pytest.mark.parametrize(
    "field", ["source", "location", "retrieved", "transcribed_by", "quantities"]
)
def test_missing_mandatory_field_refused(tmp_path, field):
    data = copy.deepcopy(SYNTHETIC_FIXTURE)
    del data[field]
    p = write_fixture(tmp_path, data, f"missing_{field}.json")
    with pytest.raises(ValueError, match=field):
        xio.load_experiment(p)


@pytest.mark.parametrize("field", ["source", "location", "retrieved", "transcribed_by"])
def test_empty_mandatory_field_refused(tmp_path, field):
    data = copy.deepcopy(SYNTHETIC_FIXTURE)
    data[field] = "   "
    p = write_fixture(tmp_path, data, f"empty_{field}.json")
    with pytest.raises(ValueError, match=field):
        xio.load_experiment(p)


def test_empty_quantities_refused(tmp_path):
    data = copy.deepcopy(SYNTHETIC_FIXTURE)
    data["quantities"] = {}
    p = write_fixture(tmp_path, data)
    with pytest.raises(ValueError, match="quantities"):
        xio.load_experiment(p)


def test_quantity_missing_units_refused(tmp_path):
    data = copy.deepcopy(SYNTHETIC_FIXTURE)
    del data["quantities"]["temp"]["units"]
    p = write_fixture(tmp_path, data)
    with pytest.raises(ValueError, match="'temp'.*units"):
        xio.load_experiment(p)


def test_quantity_missing_value_refused(tmp_path):
    data = copy.deepcopy(SYNTHETIC_FIXTURE)
    del data["quantities"]["freq"]["value"]
    p = write_fixture(tmp_path, data)
    with pytest.raises(ValueError, match="'freq'.*value"):
        xio.load_experiment(p)


def test_unknown_units_refused(tmp_path):
    data = copy.deepcopy(SYNTHETIC_FIXTURE)
    data["quantities"]["freq"]["units"] = "furlongs"
    p = write_fixture(tmp_path, data)
    with pytest.raises(ValueError, match="furlongs"):
        xio.load_experiment(p)


def test_variant_filename_requires_declaration(tmp_path):
    data = copy.deepcopy(SYNTHETIC_FIXTURE)
    p = write_fixture(tmp_path, data, "synthetic_2026_table1_variant-hot.json")
    with pytest.raises(ValueError, match="variant_of"):
        xio.load_experiment(p)


def test_variant_requires_changes(tmp_path):
    data = copy.deepcopy(SYNTHETIC_FIXTURE)
    data["variant_of"] = "synthetic_fixture.json"
    p = write_fixture(tmp_path, data, "synthetic_2026_table1_variant-hot.json")
    with pytest.raises(ValueError, match="changes"):
        xio.load_experiment(p)


def test_valid_variant_loads(tmp_path):
    data = copy.deepcopy(SYNTHETIC_FIXTURE)
    data["variant_of"] = "synthetic_fixture.json"
    data["changes"] = {"temp": "raised 50 -> 100 mK to probe the thermal threshold"}
    data["quantities"]["temp"]["value"] = 100.0
    p = write_fixture(tmp_path, data, "synthetic_2026_table1_variant-hot.json")
    exp = xio.load_experiment(p)
    assert exp.variant_of == "synthetic_fixture.json"
    assert "temp" in exp.changes


def test_synthetic_requires_comment(tmp_path):
    data = copy.deepcopy(SYNTHETIC_FIXTURE)
    del data["comment"]
    p = write_fixture(tmp_path, data)
    with pytest.raises(ValueError, match="comment"):
        xio.load_experiment(p)


# ------------------------------------ provenance honesty (circuit QED) --


def _tag(q):
    return q.extra.get("provenance")


def test_circuit_qed_files_ship():
    names = xio.list_experiments()
    for want in CIRCUIT_QED_FILES:
        assert want in names, f"missing circuit-QED parameter file {want}"


@pytest.mark.parametrize("name", sorted(CIRCUIT_QED_FILES))
def test_every_circuit_qed_quantity_is_provenance_tagged(shipped, name):
    exp = shipped[name]
    for qname, q in exp.quantities.items():
        assert _tag(q) in PROVENANCE_TAGS, f"{name}:{qname} has provenance {_tag(q)!r}"
    for key in ("synthetic_quantities", "derived_quantities"):
        assert isinstance(exp.extras.get(key), list), f"{name} lacks top-level {key}"


@pytest.mark.parametrize("name", sorted(CIRCUIT_QED_FILES))
def test_synthetic_flags_are_honest(shipped, name):
    """The synthetic-tagged set == the file's own list == the documented list,
    and the file-level flag is true exactly when that set is non-empty."""
    exp = shipped[name]
    tagged = {n for n, q in exp.quantities.items() if _tag(q) == "synthetic"}
    listed = set(exp.extras["synthetic_quantities"])
    assert tagged == listed, f"{name}: tagged {tagged} != listed {listed}"
    assert tagged == set(CIRCUIT_QED_FILES[name]), (
        f"{name}: synthetic set {tagged} != documented {set(CIRCUIT_QED_FILES[name])}"
    )
    if tagged:
        assert exp.synthetic is True, f"{name}: synthetic numbers but flag False"
    elif exp.synthetic:
        # allowed only for a declared variant whose comment says the flag covers
        # the SELECTION of published operating points, not any number
        assert exp.variant_of, f"{name}: flagged synthetic with no synthetic number"
        c = (exp.comment or "").upper()
        assert "SYNTHETIC AT FILE LEVEL ONLY" in c and "SELECTION" in c, name
    derived = {n for n, q in exp.quantities.items() if _tag(q) == "derived"}
    assert derived == set(exp.extras["derived_quantities"]), name
    digitized = {n for n, q in exp.quantities.items() if _tag(q) == "plot_digitized"}
    assert digitized == set(exp.extras.get("plot_digitized", [])), name
    # every synthetic quantity's note labels the synthetic entries
    for n in tagged:
        assert "SYNTHETIC" in (exp.quantities[n].note or ""), f"{name}:{n}"


def test_platform_wide_synthetic_set_is_exactly_documented(shipped):
    union = {
        (name, n)
        for name in CIRCUIT_QED_FILES
        for n, q in shipped[name].quantities.items()
        if _tag(q) == "synthetic"
    }
    assert union == {
        (VARIANT_NOISE, "waveguide_temperature_grid"),
        (VARIANT_NOISE, "relaxation_rate_grid"),
        (VARIANT_NOISE, "pure_dephasing_rate_grid"),
    }


@pytest.mark.parametrize("name", sorted(CIRCUIT_QED_FILES))
def test_derived_quantities_name_real_inputs(shipped, name):
    exp = shipped[name]
    for qname, q in exp.quantities.items():
        if _tag(q) != "derived":
            continue
        refs = q.extra.get("derived_from")
        assert isinstance(refs, str) and refs.strip(), f"{name}:{qname} lacks derived_from"
        for ref in (r.strip() for r in refs.split(",")):
            if ":" in ref:
                fname, inp = ref.split(":", 1)
                assert fname in shipped, f"{name}:{qname} -> unknown file {fname}"
                assert inp in shipped[fname].quantities, f"{name}:{qname} -> {ref}"
            else:
                assert ref in exp.quantities, f"{name}:{qname} -> {ref}"


def test_table1_is_internally_consistent(shipped):
    """Transcription sanity: Table I's columns obey the paper's own laws
    (Eqs. 30, 33, 66; caption Delta-Omega/2pi = 5.2 lambda GHz) to rounding."""
    tb = shipped[PROPOSAL].quantities
    lam = tb["table1_scenario_lambda"].value
    gam = tb["table1_scenario_gamma"].value
    alp = tb["table1_scenario_alpha"].value
    dom = tb["table1_scenario_gap_variation"].value
    k_lam = tb["lambda_per_gamma"].value  # -4.53
    k_dom = tb["gap_variation_per_lambda"].value  # 5.2 GHz
    k_alp = tb["spin_boson_alpha_coefficient"].value  # 6.54
    assert lam[0] == gam[0] == alp[0] == dom[0] == 0  # the '-> 0' scenario
    for i in range(1, 6):
        assert lam[i] == pytest.approx(k_lam * gam[i], rel=0.15), i  # lambda to 1 s.f.
        assert dom[i] == pytest.approx(k_dom * lam[i], rel=0.10), i
        assert alp[i] == pytest.approx(k_alp * gam[i] ** 2, rel=0.30), i  # 1 s.f.
    assert min(gam) == tb["coupling_range_gamma_x"].value[0]
    assert max(gam) == tb["coupling_range_gamma_x"].value[1]


def test_switching_variant_causal_structure(shipped):
    """The three operating points are what their names say, and the derived
    separations reproduce d = v t_d (including the published d = 12 cm)."""
    v = shipped[VARIANT_SWITCH]
    q = v.quantities
    assert v.variant_of == PROPOSAL and not v.extras["synthetic_quantities"]
    nat = lambda name: xio.to_natural(q[name].value, q[name].units)[0]
    speed = nat("waveguide_speed")
    for tag, rel in (("spacelike", "gt"), ("marginal", "eq"), ("causal", "lt")):
        td, T = nat(f"signaling_time_{tag}"), nat(f"switching_duration_{tag}")
        if rel == "gt":
            assert td > T
        elif rel == "eq":
            assert td == pytest.approx(T)
        else:
            assert td < T
        L = nat(f"detector_separation_{tag}")
        assert L == pytest.approx(speed * td, rel=1e-9), tag
    assert q["detector_separation_spacelike"].units == "mm"
    assert "12 cm" in q["detector_separation_spacelike"].extra["printed"]
    # 'causal' means deep contact, t_d < T/2 (the M2.6 self-consistency rule)
    assert nat("signaling_time_causal") < 0.5 * nat("switching_duration_causal")
    assert nat("switching_delay_spacelike") == pytest.approx(
        nat("signaling_time_spacelike") - nat("switching_duration_spacelike")
    )
    assert nat("switching_delay_marginal") == 0.0
    # gamma schedule is Table I's column
    assert list(q["gap_schedule_gamma"].value) == list(
        shipped[PROPOSAL].quantities["table1_scenario_gamma"].value
    )


def test_noise_sweep_grids_anchor_on_published_numbers(shipped):
    tb = shipped[PROPOSAL].quantities
    dev = shipped[DEVICE].quantities
    ns = shipped[VARIANT_NOISE].quantities
    # separations are v * t_d for the three published delays
    v = ns["waveguide_speed"].value
    seps_m = [x * 1e-3 for x in ns["detector_separation_grid"].value]
    delays_s = [x * 1e-9 for x in ns["signaling_time_grid"].value]
    assert seps_m == pytest.approx([v * t for t in delays_s], rel=1e-9)
    assert seps_m[-1] == pytest.approx(0.12)  # the published d = 12 cm
    # every published delay and duration is one the proposal prints
    assert set(ns["signaling_time_grid"].value) == {
        tb["signaling_time_td_fig23c"].value,
        tb["signaling_time_td_fig15"].value,
        tb["signaling_time_td"].value,
    }
    published_T = (
        set(tb["switching_duration_gaussian_by_scenario"].value)
        | set(tb["switching_duration_cosine_ramps_by_scenario"].value)
        | set(tb["switching_duration_trapezoid_by_scenario"].value)
        | set(tb["spacelike_duration_thresholds"].value)
    )
    assert set(ns["switching_duration_grid"].value) <= published_T
    # measured endpoints appear verbatim in the (synthetic) relaxation grid
    g1 = ns["relaxation_rate_grid"].value
    assert dev["relaxation_rate_gamma1_min"].value in g1
    assert dev["relaxation_rate_gamma1_max"].value * 1e3 in g1  # 1.85 GHz -> 1850 MHz
    # gap grid: amplifier band edges + the two published operating gaps
    assert set(ns["detector_gap_grid"].value) == set(dev["amplifier_band"].value) | {
        dev["qubit_gap_frequency"].value,
        tb["qubit_gap_frequency"].value,
    }
    # alpha grid: measured endpoints, Table I scenario-2 interior
    a = ns["coupling_alpha_grid"].value
    assert a[0] == dev["coupling_alpha_min"].value
    assert a[-1] == dev["coupling_alpha_max"].value
    assert a[1] == tb["table1_scenario_alpha"].value[1]
    # temperature grid: the four published anchors lead, extensions follow
    T = ns["waveguide_temperature_grid"].value
    assert T[:4] == [20, 30, 40, 50] and T == sorted(T)


def _x_state_negativity(p11, p44, coh):
    return max(0.0, math.hypot((p11 - p44) / 2.0, coh) - (p11 + p44) / 2.0)


def test_negativity_resolution_field_loads_and_traces_to_sources(shipped):
    """The field the go/no-go boundary is built on: present, dimensionless,
    derived, and equal to the documented conversion of its named source."""
    tb = shipped[PROPOSAL].quantities
    ren = shipped[REN].quantities
    dc = shipped[DICARLO].quantities
    q = tb["negativity_resolution_nats"]
    assert q.units == "dimensionless" and _tag(q) == "derived"
    assert q.extra["derived_from"] == f"{REN}:qst_concurrence_floor"
    # E_N = ln(2N+1) <= ln(1+C) ~ C: the stored floor IS the concurrence floor
    c_floor = ren["qst_concurrence_floor"].value
    assert q.value == pytest.approx(c_floor)
    assert math.log(1.0 + c_floor) == pytest.approx(q.value, rel=1e-3)
    lo, hi = q.extra["plausible_range_nats"]
    assert lo == pytest.approx(ren["optimal_point_log_negativity_nats"].value, rel=0.02)
    assert hi == pytest.approx(max(dc["concurrence"].error))
    # the measured transmon cross-check is the smallest reported concurrence s.d.
    cc = tb["negativity_resolution_nats_transmon_crosscheck"]
    assert _tag(cc) == "derived" and cc.value == pytest.approx(min(dc["concurrence"].error))
    # the Ren-state negativity reproduces from the printed density matrix
    p11, p22, p33, p44 = ren["optimal_point_populations"].value
    coh = ren["optimal_point_coherence_magnitude"].value
    N = _x_state_negativity(p11, p44, coh)
    assert ren["optimal_point_negativity"].value == pytest.approx(N, rel=0.02)
    assert ren["optimal_point_log_negativity_nats"].value == pytest.approx(
        math.log(1.0 + 2.0 * N), rel=0.02
    )
    C = 2.0 * (coh - math.sqrt(p11 * p44))
    assert ren["optimal_point_concurrence"].value == pytest.approx(C, rel=0.03)
    # the floor sits inside its own plausible range, and the old 1e-4 assumption
    # sits at the bottom of it
    assert lo <= q.value <= hi and lo <= 1.0e-4 <= hi


def test_new_files_convert_to_lattice(shipped):
    for name in CIRCUIT_QED_FILES:
        lat = shipped[name].lattice(OMEGA_REF, v=V_WAVE)
        assert len(lat) > 0, name
    org = shipped["circuit_qed_harvesting/orgiazzi_2016_fig3"]
    # a printed decay rate 0.77 us^-1 stored in s^-1: no 2*pi under the loader
    lat = org.lattice(OMEGA_REF)
    assert lat["pure_dephasing_rate_ramsey_qubit1"] == pytest.approx(
        7.7e5 / OMEGA_REF, rel=1e-12
    )
    assert org.quantities["pure_dephasing_rate_ramsey_qubit1"].extra["printed"]
    sw = shipped[VARIANT_SWITCH].lattice(OMEGA_REF, v=V_WAVE)
    assert sw["detector_separation_spacelike"] == pytest.approx(
        0.12 / V_WAVE * OMEGA_REF, rel=1e-12
    )


# ------------------------------------------- unit conversion (fabricated) --


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    p = write_fixture(tmp_path_factory.mktemp("experiments"), SYNTHETIC_FIXTURE)
    return xio.load_experiment(p)


def test_synthetic_fixture_is_marked(synthetic):
    assert synthetic.synthetic is True
    assert "FABRICATED" in synthetic.comment


def test_frequency_conversion(synthetic):
    # 1 GHz -> 2*pi*1e9 rad/s; with omega_ref = 2*pi*1e9 exactly 1 lattice unit
    nat, kind = xio.to_natural(1.0, "GHz")
    assert kind == "energy"
    assert nat == pytest.approx(2.0 * math.pi * 1e9, rel=1e-14)
    lat = synthetic.lattice(OMEGA_REF, v=V_WAVE)
    assert lat["freq"] == pytest.approx(1.0, rel=1e-12), f"freq -> {lat['freq']}"


def test_temperature_conversion(synthetic):
    # 50 mK -> k_B*T/hbar; hand value: k_B/hbar = 1.3092034e11 rad/(s*K)
    nat, kind = xio.to_natural(50.0, "mK")
    assert kind == "energy"
    assert nat == pytest.approx(xio.K_B_SI * 0.050 / xio.HBAR_SI, rel=1e-14)
    assert nat == pytest.approx(6.546017e9, rel=1e-5), f"50 mK -> {nat} rad/s"
    lat = synthetic.lattice(OMEGA_REF)
    assert lat["temp"] == pytest.approx(1.0418288, rel=1e-5), f"temp -> {lat['temp']}"
    # published errors convert with the same linear factor: 5 mK is temp/10
    assert lat["temp_error"] == pytest.approx(lat["temp"] / 10.0, rel=1e-12)


def test_time_conversion(synthetic):
    # 100 ns * (2*pi*1e9 rad/s) = 200*pi
    lat = synthetic.lattice(OMEGA_REF)
    assert lat["time"] == pytest.approx(200.0 * math.pi, rel=1e-12), (
        f"100 ns -> {lat['time']}"
    )


def test_length_and_velocity_conversion(synthetic):
    # 3 mm / (1.2e8 m/s) = 25 ps; * omega_ref = pi/20 light-crossing units
    lat = synthetic.lattice(OMEGA_REF, v=V_WAVE)
    assert lat["length"] == pytest.approx(math.pi / 20.0, rel=1e-12), (
        f"3 mm -> {lat['length']}"
    )
    # the waveguide speed itself maps to 1 by construction
    assert lat["speed"] == pytest.approx(1.0, rel=1e-14)


def test_dimensionless_and_percent(synthetic):
    lat = synthetic.lattice(OMEGA_REF)
    assert lat["fraction"] == pytest.approx(0.10, rel=1e-14)
    assert lat["plain"] == pytest.approx(0.05, rel=1e-14)


def test_list_values_convert_elementwise(synthetic):
    lat = synthetic.lattice(OMEGA_REF)
    assert lat["band"] == pytest.approx((4.0, 6.0), rel=1e-12)


def test_unconvertible_units_skipped_or_strict(synthetic):
    lat = synthetic.lattice(OMEGA_REF, v=V_WAVE)
    assert "resistance" not in lat  # ohms carry no hbar=1 conversion
    with pytest.raises(ValueError, match="resistance"):
        synthetic.lattice(OMEGA_REF, v=V_WAVE, strict=True)
    with pytest.raises(ValueError, match="ohm"):
        xio.quantity_to_lattice(218.0, "ohm", OMEGA_REF)


def test_length_without_speed_raises():
    with pytest.raises(ValueError, match="propagation speed"):
        xio.quantity_to_lattice(3.0, "mm", OMEGA_REF)


def test_bad_omega_ref_raises():
    with pytest.raises(ValueError, match="omega_ref"):
        xio.quantity_to_lattice(1.0, "GHz", 0.0)


# ----------------------------------------- published-number cross-checks --


def test_wilson_thermal_occupation_crosscheck(shipped):
    """The full conversion chain against a published number.

    Wilson et al., Nature 479, 376 (2011), main text: at the 50 mK sample
    temperature 'this corresponds to a thermal photon occupation number of
    n = 0.008 at 5 GHz'.  Convert the file's own temperature and reference
    frequency through the loader and check bose_occupation against the
    published (1-significant-digit) value.
    """
    wilson = shipped["dce_microwave/wilson_2011_table1"]
    T_nat, _ = xio.to_natural(
        wilson.quantities["sample_temperature"].value,
        wilson.quantities["sample_temperature"].units,
    )
    w_nat, _ = xio.to_natural(
        wilson.quantities["thermal_occupation_reference_frequency"].value,
        wilson.quantities["thermal_occupation_reference_frequency"].units,
    )
    n = xio.bose_occupation(w_nat, T_nat)
    published = wilson.quantities["thermal_occupation_at_reference"].value
    assert abs(n - published) < 1.5e-3, (
        f"bose_occupation(5 GHz, 50 mK) = {n:.6f}, paper quotes {published}"
    )


def test_bose_occupation_edges():
    assert xio.bose_occupation(1.0, 0.0) == 0.0
    assert xio.bose_occupation(1e6, 1.0) == 0.0  # overflow-guarded tail
    assert xio.bose_occupation(1.0, 1.0) == pytest.approx(1.0 / (math.e - 1.0))
    with pytest.raises(ValueError):
        xio.bose_occupation(0.0, 1.0)


# ----------------------------------------------------------------- purity --


def test_loading_is_pure_and_results_read_only(shipped):
    a = xio.load_experiment("ibm_qet/ikeda_2023_fig2")
    b = xio.load_experiment("ibm_qet/ikeda_2023_fig2")
    assert a.quantities["shots"].value == b.quantities["shots"].value
    with pytest.raises(TypeError):
        a.quantities["shots"] = None  # MappingProxyType: no mutation
    # conversion never touches the stored verbatim values
    before = a.quantities["qubit0_frequency"].value
    _ = a.lattice(OMEGA_REF)
    assert a.quantities["qubit0_frequency"].value == before
