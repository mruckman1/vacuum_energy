"""Smoke test for the M2.7 papers candidate: circuit-QED noise-threshold surfaces.

The deliverable lives in ``papers/circuit-qed-noise-thresholds/``; this file
is its schema guard and the tautology-proof check of its boundary finder.

It runs the notebook's ``--quick`` configuration through exactly the code
path the full run uses -- ``notebook.run_specs`` -> ``notebook.run_point``
-> :func:`vacuum.protocol.run_protocol` -> the dt-halving gate ->
:func:`vacuum.detectors.harvested_log_negativity` and
:func:`vacuum.core.mutual_information` -- and asserts that every row carries
the complete mandatory schema (spec M2.7: audit flags, converged dt, the
mp-margin ``flagged`` field, the M_vac/M_comm split where the perturbative
row exists, plus this candidate's MI column and the resolvability flags at
the derived floor and at both edges of its plausible band).

The quick configuration is a 4 x 4 (temperature x dephasing) mini-surface
at the sourced operating point plus the branches it does not reach (loss
channel, static-gap perturbative companion, truncated-gaussian switching),
at a coarse gate and a small buffer so the whole file is seconds -- the
physics tolerances live in the sweep itself, not here.

Conventions per docs/API.md throughout.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from vacuum.experiments_io import quantity_to_lattice

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "circuit-qed-noise-thresholds" / "notebook.py"

#: Ledger closure tolerance, spec test matrix ("all protocols", 1e-9).
LEDGER_TOL = 1e-9

#: The three grids experiments/README.md documents as the platform's ONLY
#: synthetic quantities.
FILE_SYNTHETIC_GRIDS = {
    "waveguide_temperature_grid",
    "relaxation_rate_grid",
    "pure_dephasing_rate_grid",
}


def _load_notebook():
    """Import the candidate's notebook.py as a module (it is not a package)."""
    if not NOTEBOOK_PATH.exists():  # pragma: no cover - guards a moved file
        pytest.skip(f"notebook not found at {NOTEBOOK_PATH}")
    spec = importlib.util.spec_from_file_location(
        "m27_noise_threshold_notebook", NOTEBOOK_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def nb():
    return _load_notebook()


@pytest.fixture(scope="module")
def inputs(nb):
    """The parameter files, loaded through the provenance-enforcing loader."""
    return nb.load_inputs()


@pytest.fixture(scope="module")
def quick_rows(nb, inputs):
    """The --quick configuration, run sequentially (cached; the expensive bit)."""
    specs = nb.quick_specs(inputs)
    assert len(specs) == 16 + 3, f"quick configuration must be 4x4 + 3, got {len(specs)}"
    return nb.run_specs(specs, inputs, verbose=False)


# --------------------------------------------------------------------------
# Inputs and provenance
# --------------------------------------------------------------------------


def test_all_parameter_files_load(nb, inputs):
    """Every file the candidate consumes passes the M-I.3 provenance rules."""
    assert set(inputs.experiments) == set(nb.PARAM_FILES)
    for name, exp in inputs.experiments.items():
        assert exp.source and exp.location and exp.retrieved and exp.transcribed_by
        assert exp.quantities, f"{name} has no quantities"


def test_file_level_synthetic_content_is_exactly_the_three_grids(inputs):
    """Across the platform's files, only the three documented grids are synthetic.

    Read off the per-quantity ``provenance`` tags (the loader's view), not
    off memory: this is the input to the admissibility verdict.
    """
    synthetic = set()
    for exp in inputs.experiments.values():
        for qname, q in exp.quantities.items():
            if q.extra.get("provenance") == "synthetic":
                synthetic.add(qname)
    assert synthetic == FILE_SYNTHETIC_GRIDS, sorted(synthetic)


def test_registered_provenance_is_explicit(nb, inputs):
    """Every input the notebook uses carries a known tag; the unsourced set is named.

    Sourced (not synthetic): everything that fixes the operating point and
    the resolution.  Synthetic/unsourced: the three file grids, this
    candidate's axis refinements, the lattice regulators, and the coupling.
    """
    for key, (_file, tag, note) in inputs.provenance.items():
        assert tag in nb.PROVENANCE_TAGS, f"{key}: unknown tag {tag!r}"
        assert note, f"{key}: empty provenance note"
    syn = set(inputs.synthetic_inputs)
    assert syn == {
        "coupling_lambda", "field_mass", "boundary_buffer",
        "waveguide_temperature_grid", "relaxation_rate_grid", "pure_dephasing_rate_grid",
        "temperature_axis", "dephasing_axis", "relaxation_axis", "separation_axis",
        "coupling_axis",
    }, sorted(syn)
    for key in (
        "omega_ref", "waveguide_speed", "scenario_lambda", "scenario_gamma",
        "scenario_alpha", "gap_variation_coefficient", "switching_window_ns",
        "signaling_time_ns", "separation_mm", "resolution_nats", "resolution_band_nats",
        "T_fridge_base", "T_transmission_line", "G1_min", "G1_max", "Gphi_min", "Gphi_max",
    ):
        assert key in inputs.provenance, key
        assert key not in syn, f"{key} is sourced, not synthetic"
    # the plot-digitised device numbers are tagged as such, not as transcribed
    assert inputs.provenance["Gphi_min"][1] == "plot_digitized"
    assert inputs.provenance["Gphi_max"][1] == "plot_digitized"
    # and the coupling is the ONE unsourced input
    assert [k for k, (_f, t, _n) in inputs.provenance.items() if t == "unsourced"] == [
        "coupling_lambda"
    ]


def test_operating_point_is_the_table1_scenario_closest_to_the_device(nb, inputs):
    """Scenario 2 is the only Table I scenario inside the measured alpha range."""
    tb = inputs.experiments[nb.PROPOSAL].quantities
    dev = inputs.experiments[nb.DEVICE].quantities
    alphas = [float(a) for a in tb["table1_scenario_alpha"].value]
    a_min = float(dev["coupling_alpha_min"].value)
    a_max = float(dev["coupling_alpha_max"].value)
    inside = [i + 1 for i, a in enumerate(alphas) if a_min <= a <= a_max]
    assert inside == [2, 6], inside  # scenarios 2 and 6 share alpha = 0.003
    p = inputs.point
    assert p.scenario == 2 and nb.SCENARIO_INDEX == 1
    assert p.alpha_tb == alphas[1]
    assert p.lambda_tb == float(tb["table1_scenario_lambda"].value[1])
    assert p.gap_shift_gamma == float(tb["table1_scenario_gamma"].value[1])
    coef = float(tb["gap_variation_coefficient"].value)
    assert p.gap_shift_GHz == pytest.approx(coef * p.gap_shift_gamma)
    assert p.gap_shift_GHz < 0.0  # Eq. (30): the coupler shifts the gap DOWN
    assert p.window_ns == float(tb["switching_duration_cosine_ramps_by_scenario"].value[1])
    assert p.switching_kind == "cos2"  # cosine ramps with S_f = 0
    assert float(tb["switching_flat_fraction_cosine_ramps"].value) == 0.0
    v = float(tb["waveguide_speed"].value)
    assert p.separation_mm == pytest.approx(v * p.signaling_time_ns * 1e-9 * 1e3)
    assert p.gap_GHz == float(tb["qubit_gap_frequency"].value)


def test_coupling_is_anchored_but_declared_unsourced(nb, inputs, quick_rows):
    """lam_osc = |lambda_TB| sqrt(2 Omega) by MAP-1, and every row says it is unsourced."""
    p = inputs.point
    gap_lat = inputs.units.energy(p.gap_GHz * 1e9)
    assert p.coupling_lambda == pytest.approx(abs(p.lambda_tb) * math.sqrt(2.0 * gap_lat))
    assert inputs.provenance["coupling_lambda"][1] == "unsourced"
    assert "not a derivation" in nb.COUPLING_STATUS
    for row in quick_rows:
        assert row["inputs_all_sourced"] is False
        assert row["lambda_udw"] == pytest.approx(
            row["coupling_lambda"] / math.sqrt(2.0 * row["gap_lat"]), rel=1e-12
        )


def test_resolution_comes_from_the_parameter_file(nb, inputs):
    """The floor and its band are read from the derived quantity, not typed in."""
    q = inputs.experiments[nb.PROPOSAL].quantities["negativity_resolution_nats"]
    r = inputs.resolution
    assert r.derived == float(q.value)
    assert [r.band_low, r.band_high] == [float(x) for x in q.extra["plausible_range_nats"]]
    assert r.band_low < r.derived < r.band_high
    assert q.extra["provenance"] == "derived"
    ren = inputs.experiments[nb.REN].quantities["optimal_point_log_negativity_nats"]
    assert r.band_low == float(ren.value)
    xq = inputs.experiments[nb.PROPOSAL].quantities[
        "negativity_resolution_nats_transmon_crosscheck"
    ]
    assert r.crosscheck == float(xq.value)
    assert r.levels == {"derived": r.derived, "band_low": r.band_low, "band_high": r.band_high}


def test_units_match_experiments_io(nb, inputs):
    """The notebook's named conversions ARE ``experiments_io``'s rules."""
    u = inputs.units
    w, v = u.omega_ref, u.v
    assert u.energy(5.7e9) == pytest.approx(quantity_to_lattice(5.7, "GHz", w), rel=1e-15)
    assert u.temperature(50e-3) == pytest.approx(quantity_to_lattice(50.0, "mK", w), rel=1e-15)
    assert u.time(0.3e-9) == pytest.approx(quantity_to_lattice(0.3, "ns", w), rel=1e-15)
    assert u.length(6e-3) == pytest.approx(quantity_to_lattice(6.0, "mm", w, v=v), rel=1e-15)
    assert w == pytest.approx(2.0 * math.pi * 7.3e9, rel=1e-15)
    assert u.site_mm == pytest.approx(v / w * 1e3, rel=1e-15)


def test_anchors_lie_on_their_axes(nb, inputs):
    """Every device anchor is a grid point, so verdicts are reads, not interpolations."""
    for ax, anchors in inputs.anchors.items():
        axis = np.asarray(inputs.axes[ax])
        for name, val in anchors.items():
            if name == "orgiazzi_ramsey_floor_Hz_equiv":
                continue  # a comparison floor, deliberately not a grid point
            assert np.any(np.isclose(axis, val, rtol=1e-9, atol=0.0)), (ax, name, val)
    # the axes are supersets of the file grids' entries inside their range
    for key, ax in (("waveguide_temperature_grid", "temperature_mK"),
                    ("relaxation_rate_grid", "gamma1_MHz"),
                    ("pure_dephasing_rate_grid", "gammaphi_Hz")):
        file_grid = inputs.experiments[nb.VARIANT_NOISE].quantities[key].value
        axis = np.asarray(inputs.axes[ax])
        for x in file_grid:
            if axis.min() <= x <= axis.max():
                assert np.any(np.isclose(axis, x, rtol=1e-9, atol=0.0)), (key, x)
    for ax, vals in inputs.axes.items():
        assert len(vals) >= 12, f"{ax}: {len(vals)} < 12 points"
        assert list(vals) == sorted(vals) and len(set(vals)) == len(vals), ax


# --------------------------------------------------------------------------
# Schema completeness -- the point of this file
# --------------------------------------------------------------------------


def test_schema_columns_are_present_on_every_row(nb, quick_rows):
    """Every mandatory column exists on every row, with nothing extra."""
    cols = set(nb.ROW_COLUMNS)
    assert len(nb.ROW_COLUMNS) == len(cols), "duplicate name in ROW_COLUMNS"
    for must in ("E_N", "MI", "work", "E_N_per_work", "MI_per_work", "dt_converged",
                 "mp_margin_flagged", "resolvable", "resolvable_band_low",
                 "resolvable_band_high", "MI_resolvable", "MI_resolvable_band_low",
                 "MI_resolvable_band_high", "resolution_derived_nats",
                 "resolution_band_low_nats", "resolution_band_high_nats",
                 "M_vac_abs", "M_comm_abs", "comm_fraction", "grid_iT", "grid_iX"):
        assert must in cols, must
    for row in quick_rows:
        missing = cols - set(row)
        extra = set(row) - cols
        assert not missing, f"row {row['label']!r} missing columns {sorted(missing)}"
        assert not extra, f"row {row['label']!r} has undeclared columns {sorted(extra)}"


def test_mandatory_column_groups_are_populated(nb, inputs, quick_rows):
    """Audit flags, converged dt, the mp-margin field, MI, and the band flags."""
    audit_flags = (
        "mp_margin_flagged", "precision_flagged", "passivity_entry_passed",
        "passivity_initial_passed", "ledger_passed", "causality_audited",
        "all_audits_passed",
    )
    r_ = inputs.resolution
    for row in quick_rows:
        for flag in audit_flags:
            assert isinstance(row[flag], bool), f"{flag} must be a bool"
        assert isinstance(row["ledger_defect"], float)
        assert row["converged"] is True, f"{row['label']} did not converge"
        assert row["dt_converged"] > 0.0
        assert row["n_substeps"] == row["n_segments"] * nb.SUBSTEPS_PER_SEGMENT
        assert row["dt_converged"] == pytest.approx(
            row["window_lat"] / row["n_substeps"], rel=1e-12
        )
        assert row["movement"] < row["window_lat"]
        assert np.isfinite(row["work_movement"])
        assert row["negativity_regime"] in ("float64", "mpmath")
        assert row["mp_margin_flagged"] == (row["negativity_regime"] == "mpmath")
        assert row["E_N"] >= 0.0 and np.isfinite(row["E_N"])
        assert np.isfinite(row["MI"]) and row["MI"] > -1e-12  # subadditivity
        assert row["min_nu_pt"] > 0.0
        assert np.isfinite(row["work"]) and np.isfinite(row["dissipated"])
        assert row["n_d_A"] >= 0.0 and row["n_d_B"] >= 0.0
        # the resolvability flags are exactly E_N / MI against the stored floors
        assert (row["resolution_derived_nats"], row["resolution_band_low_nats"],
                row["resolution_band_high_nats"]) == (r_.derived, r_.band_low, r_.band_high)
        assert row["resolvable"] == (row["E_N"] > r_.derived)
        assert row["resolvable_band_low"] == (row["E_N"] > r_.band_low)
        assert row["resolvable_band_high"] == (row["E_N"] > r_.band_high)
        assert row["MI_resolvable"] == (row["MI"] > r_.derived)
        assert row["MI_resolvable_band_low"] == (row["MI"] > r_.band_low)
        assert row["MI_resolvable_band_high"] == (row["MI"] > r_.band_high)
        # band monotonicity: resolvable at a higher floor implies at a lower one
        assert (not row["resolvable_band_high"]) or row["resolvable"]
        assert (not row["resolvable"]) or row["resolvable_band_low"]
        if row["work"] > 0.0:
            assert row["E_N_per_work"] == pytest.approx(row["E_N"] / row["work"])
            assert row["MI_per_work"] == pytest.approx(row["MI"] / row["work"])


def test_audits_pass_and_ledger_closes(quick_rows):
    """Standing audits on the exact runs that produced these rows."""
    for row in quick_rows:
        assert row["all_audits_passed"], f"{row['label']}: an audit failed"
        assert row["passivity_initial_passed"], f"{row['label']}: initial passivity failed"
        assert abs(row["ledger_defect"]) <= LEDGER_TOL, (
            f"{row['label']}: ledger defect {row['ledger_defect']:.3e} > {LEDGER_TOL}"
        )


def test_communication_split_present_exactly_where_it_exists(nb, quick_rows):
    """M_vac / M_comm are filled on the static-gap companion and NaN elsewhere.

    The perturbative engine is a vacuum, noiseless, STATIC-gap, compact-
    support object; the operating point carries the scenario-2 gap shift,
    so the surface rows must NOT carry a split, and the ``split`` companion
    (gap_shift_gamma = 0) must.
    """
    split = [r for r in quick_rows if r["sweep"] == "split"]
    assert len(split) == 1
    ok = split[0]
    assert ok["gap_shift_gamma"] == 0.0
    assert ok["has_perturbative_row"] is True and not ok["perturbative_quadrature_failed"]
    for col in ("P_A", "P_B", "M_abs", "M_vac_abs", "M_comm_abs", "comm_fraction",
                "E_N_pert", "perturbative_scale"):
        assert np.isfinite(ok[col]), f"perturbative column {col} is not finite"
    assert ok["P_A"] > 0.0 and ok["P_B"] > 0.0
    assert 0.0 <= ok["comm_fraction"] <= 2.0
    for r in quick_rows:
        if r["sweep"] == "split":
            continue
        assert r["has_perturbative_row"] is False, r["label"]
        for col in ("M_vac_abs", "M_comm_abs", "comm_fraction", "E_N_pert"):
            assert math.isnan(r[col]), f"{r['label']}.{col} must be NaN"


def test_branch_coverage_of_the_quick_configuration(nb, quick_rows):
    """The quick rows exercise every branch the surfaces do."""
    by = {r["label"]: r for r in quick_rows}
    surf = [r for r in quick_rows if r["sweep"] == nb.MAIN_SURFACE]
    assert len(surf) == 16
    assert {r["temperature_mK"] for r in surf} == set(nb.QUICK_T_MK)
    assert {r["gammaphi_Hz"] for r in surf} == set(nb.QUICK_GPHI_HZ)
    assert all(r["gap_shift_gamma"] > 0.0 and r["gap_shift_GHz"] < 0.0 for r in surf)
    assert all(r["gap_min_lat"] > 0.0 for r in surf)
    assert by["quick-relaxation"]["gamma1_MHz"] > 0.0
    assert by["quick-relaxation"]["dissipated"] != 0.0, (
        "a loss channel that books zero energy flow is not exercising the channel path"
    )
    assert by["quick-gaussian"]["switching_kind"] == "gaussian"
    assert any(r["temperature_mK"] == 0.0 for r in surf)
    assert any(r["temperature_mK"] > 0.0 for r in surf)
    for r in quick_rows:
        assert r["passivity_entry_passed"] is True


def test_mp_margin_rule_is_live_on_this_code_path(nb, inputs):
    """The mp-margin escalation actually fires here, on a floor state.

    Across the surfaces the rule may report ``flagged = False`` on every
    row (harvesting dies suddenly, so the partial-transpose spectrum is
    never parked at the vacuum floor).  This test drives the SAME
    ``run_point`` path to a configuration whose exit state is exactly the
    uncoupled vacuum -- lambda = 0 -- where min(nu~) sits on the floor 1/2
    and the M2.2 rule MUST escalate to mpmath.
    """
    spec = nb.base_spec(
        inputs, "Q-floor", "quick", coupling_lambda=0.0, gap_shift_gamma=0.0,
        **nb.QUICK_COMMON,
    )
    row = nb.run_point(spec, inputs, perturbative=False)
    assert row["mp_margin_flagged"] is True, (
        f"lambda = 0 leaves the state on the vacuum floor "
        f"(min nu~ = {row['min_nu_pt']!r}) but the mp-margin rule did not fire"
    )
    assert row["negativity_regime"] == "mpmath"
    assert row["precision_flagged"] is True
    assert row["E_N"] < 1e-12 and abs(row["margin_min"]) < 1e-13
    assert abs(row["MI"]) < 1e-12
    assert row["work"] == 0.0 and row["all_audits_passed"]
    assert math.isnan(row["E_N_per_work"]) and math.isnan(row["MI_per_work"])


# --------------------------------------------------------------------------
# The surface machinery and the boundary finder
# --------------------------------------------------------------------------


def test_quick_surface_reconstructs_and_contours(nb, inputs, quick_rows):
    """The 4x4 mini-surface round-trips to arrays; contours and anchors compute."""
    T, X, Z = nb.surface_arrays(quick_rows, nb.MAIN_SURFACE, "E_N")
    assert Z.shape == (4, 4) and list(T) == list(nb.QUICK_T_MK)
    assert list(X) == list(nb.QUICK_GPHI_HZ)
    _T, _X, M = nb.surface_arrays(quick_rows, nb.MAIN_SURFACE, "MI")
    assert M.shape == (4, 4)
    for level in inputs.resolution.levels.values():
        c = nb.surface_contour(T, X, Z, level, log_x=True)
        assert c["n_points"] == 16 and 0.0 <= c["go_fraction"] <= 1.0
        assert len(c["rows"]) == 4 and len(c["cols"]) == 4
        assert c["n_go"] == int((Z > level).sum())
    anchors = nb.anchor_verdicts(quick_rows, nb.MAIN_SURFACE, inputs)
    # T in {20, 30, 50} x Gamma_phi in {1e7, 2e8} are on the quick grid
    assert len(anchors) == 6, [(a["temperature_anchor"], a["x_anchor"]) for a in anchors]
    for a in anchors:
        assert set(a["verdict"]) == {"derived", "band_low", "band_high"}
        assert all(v in ("GO", "NO-GO") for v in a["verdict"].values())
    piv = nb.pivot_verdict(quick_rows, inputs)
    ent = piv["per_surface"][nb.MAIN_SURFACE]
    for k in ("derived", "band_low", "band_high"):
        assert 0.0 <= ent[k]["MI_only_fraction"] <= ent[k]["MI_resolvable_fraction"] <= 1.0
    assert piv["device_box"]["n_anchor_points"] >= 2


def test_contour_finder_recovers_a_known_boundary_exactly(nb):
    """Tautology-proof check of the boundary finder on synthetic surfaces.

    (a) A plane Z = max(0, 1 - x - y): the level-f contour is the straight
    line x + y = 1 - f, and linear interpolation is exact on a linear
    surface, so every row and column crossing must land on it to roundoff
    (and the finder must not report sudden death where both sides are
    positive).  (b) A log-axis surface linear in log10(x).  (c) A step
    (sudden death): the bracket is exact and flagged.  (d) A non-monotone
    row: two crossings with opposite directions.  (e) No crossing: empty.
    """
    # (a) the plane
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 11)
    Z = np.maximum(0.0, 1.0 - x[None, :] - y[:, None])
    f = 0.25
    c = nb.surface_contour(y, x, Z, f)
    n_checked = 0
    for row in c["rows"]:
        yy = row["temperature_mK"]
        for cr in row["crossings"]:
            if cr["z_lo"] > 0.0 and cr["z_hi"] > 0.0:
                assert cr["x_interp"] == pytest.approx(1.0 - f - yy, abs=1e-12)
                assert cr["direction"] == "go->nogo" and cr["sudden_death"] is False
                n_checked += 1
    for col in c["cols"]:
        xx = col["x"]
        for cr in col["crossings"]:
            if cr["z_lo"] > 0.0 and cr["z_hi"] > 0.0:
                assert cr["x_interp"] == pytest.approx(1.0 - f - xx, abs=1e-12)
                n_checked += 1
    assert n_checked >= 10, "the plane test checked too few crossings to mean anything"
    assert c["n_go"] == int((Z > f).sum()) and c["max_z"] == 1.0
    # (b) a log axis: z = 2 - log10(x), level 0.9 -> x* = 10**1.1 exactly
    xl = 10.0 ** np.arange(0.0, 3.01, 0.25)
    zl = 2.0 - np.log10(xl)
    crs = nb.axis_crossings(xl, zl, 0.9, log_x=True)
    assert len(crs) == 1
    assert crs[0]["x_interp"] == pytest.approx(10.0**1.1, rel=1e-12)
    assert crs[0]["direction"] == "go->nogo"
    # (c) sudden death: exact bracket, flagged
    xs = np.array([0.0, 10.0, 20.0, 30.0])
    zs = np.array([1e-2, 1e-2, 0.0, 0.0])
    crs = nb.axis_crossings(xs, zs, 1e-3)
    assert len(crs) == 1
    assert (crs[0]["x_lo"], crs[0]["x_hi"]) == (10.0, 20.0)
    assert crs[0]["sudden_death"] is True and crs[0]["direction"] == "go->nogo"
    assert crs[0]["z_lo"] == 1e-2 and crs[0]["z_hi"] == 0.0
    # (d) non-monotone: leaves GO then re-enters
    crs = nb.axis_crossings(np.arange(5.0), np.array([1.0, 0.1, 0.1, 1.0, 1.0]), 0.5)
    assert [cr["direction"] for cr in crs] == ["go->nogo", "nogo->go"]
    assert crs[0]["x_interp"] == pytest.approx(0.5 / 0.9) and crs[1]["x_interp"] == pytest.approx(2.0 + 0.4 / 0.9)
    # (e) nothing to find, and exactly-at-level counts as NO-GO (strict >)
    assert nb.axis_crossings(np.arange(3.0), np.array([1.0, 1.0, 1.0]), 0.5) == []
    assert nb.axis_crossings(np.arange(2.0), np.array([0.5, 0.5]), 0.5) == []
    with pytest.raises(ValueError):
        nb.axis_crossings(np.array([0.0, 2.0, 1.0]), np.zeros(3), 0.1)


def test_best_pockets_ranks_by_negativity_per_switching_work(nb):
    """The 'per unit switching energy' column is E_N / ledger work."""
    keys = ("gammaphi_Hz", "gamma1_MHz", "separation_mm", "separation_sites",
            "coupling_lambda", "temperature_mK", "MI", "MI_per_work")
    base = {k: 0.0 for k in keys}
    rows = [
        dict(base, label="cheap", sweep="dephasing", E_N=1e-3, work=1e-3, E_N_per_work=1.0),
        dict(base, label="big", sweep="coupling", E_N=1e-1, work=1.0, E_N_per_work=0.1),
        dict(base, label="dead", sweep="coupling", E_N=0.0, work=1.0, E_N_per_work=0.0),
        dict(base, label="control", sweep="control", E_N=1.0, work=1e-3, E_N_per_work=1e3),
    ]
    top = nb.best_pockets(rows, 1e-4, n=4)
    assert [r["label"] for r in top] == ["cheap", "big"], (
        "dead rows and non-surface rows must not enter the pocket ranking"
    )
    assert nb.best_pockets(rows, 5e-2, n=4)[0]["label"] == "big"
    assert "work" in top[0] and "E_N_per_work" in top[0]


# --------------------------------------------------------------------------
# Persistence and the helpers
# --------------------------------------------------------------------------


def test_rows_round_trip_through_npz(nb, quick_rows, tmp_path):
    """data/*.npz keeps the whole schema and identifies its generator and build."""
    path = nb.save_rows(tmp_path / "rows_quick.npz", quick_rows, sweep="test")
    with np.load(path, allow_pickle=False) as z:
        stored = [str(c) for c in z["__columns__"]]
        assert tuple(stored) == nb.ROW_COLUMNS
        build = str(z["__build__"])
        for key in ("generator", "generator_sha256", "git_head", "vacuum_version", "numpy"):
            assert key in build, f"build info missing {key}"
    back = nb.load_rows(path)
    assert len(back) == len(quick_rows)
    for orig, rt in zip(quick_rows, back):
        assert orig["label"] == rt["label"]
        assert orig["E_N"] == pytest.approx(rt["E_N"], rel=0.0, abs=0.0)
        assert orig["MI"] == pytest.approx(rt["MI"], rel=0.0, abs=0.0)
        assert orig["mp_margin_flagged"] == bool(rt["mp_margin_flagged"])
        assert orig["resolvable_band_low"] == bool(rt["resolvable_band_low"])
    # the dense-surface archive carries the same numbers
    inputs = nb.load_inputs()
    spath = nb.save_surfaces(tmp_path / "surfaces.npz", quick_rows, inputs)
    with np.load(spath, allow_pickle=False) as z:
        T, X, Z = nb.surface_arrays(quick_rows, nb.MAIN_SURFACE, "E_N")
        assert np.array_equal(z[f"{nb.MAIN_SURFACE}/E_N"], Z)
        assert np.array_equal(z[f"{nb.MAIN_SURFACE}/T_mK"], T)
        assert list(z["resolution_levels_nats"]) == [
            inputs.resolution.derived, inputs.resolution.band_low, inputs.resolution.band_high
        ]


def test_geometry_is_causally_safe(nb):
    """No boundary reflection can reach a detector inside the window."""
    for window in (4.5, 10.1, 40.0):
        for d in (1, 2, 7, 46):
            N, (a, b) = nb.geometry(d, window, nb.BOUNDARY_BUFFER)
            assert b - a == d
            assert 2 * min(a, N - 1 - b) > window, (
                f"window {window} reaches a boundary at d={d}, N={N}"
            )


def test_gate_schedule_only_skips_ahead_never_across_a_halving(nb):
    """The accepted movement is always a single-halving comparison."""
    tol = 1e-9

    def movement_at(level):
        return 4.0e-4 * 4.0 ** (-(level - 1))

    visited = {}
    for m0_scale in (1.0, 1e-2, 1e-4):
        sch = nb._level_schedule(13)
        seen, accepted = [], None
        for level in sch:
            seen.append(level)
            if len(seen) >= 2 and seen[-2] == level - 1:
                if movement_at(level) * m0_scale < tol:
                    accepted = level
                    break
                sch.jump(level, movement_at(level) * m0_scale, tol)
            if len(seen) > 30:  # pragma: no cover - runaway guard
                pytest.fail(f"schedule did not terminate: {seen}")
        assert accepted is not None, f"scale {m0_scale}: never accepted ({seen})"
        assert seen[0] == 0 and seen[1] == 1, seen
        assert seen == sorted(seen) and len(set(seen)) == len(seen), seen
        assert max(seen) <= 13
        assert seen[-2] == seen[-1] - 1, seen
        visited[m0_scale] = list(seen)
    realistic = visited[1.0]
    assert max(realistic) > len(realistic) - 1, f"no level was skipped: {realistic}"
    sch = nb._level_schedule(13)
    seen, accepted = [], None
    for level in sch:
        seen.append(level)
        if len(seen) >= 2 and seen[-2] == level - 1:
            if movement_at(level) * 1e4 < tol:
                accepted = level
                break
            sch.jump(level, movement_at(level) * 1e4, tol)
    assert accepted is None and max(seen) == 13, seen


def test_gate_schedule_jump_is_inert_when_it_should_be(nb):
    """No prediction from the two coarsest levels, none once converged."""
    sch = nb._level_schedule(13)
    it = iter(sch)
    assert next(it) == 0
    sch.jump(0, 1.0, 1e-9)
    assert next(it) == 1
    sch.jump(1, 1.0, 1e-9)
    assert next(it) == 2
    sch.jump(2, 1e-12, 1e-9)
    assert next(it) == 3
    sch.jump(3, 0.0, 1e-9)
    assert next(it) == 4
    sch.jump(4, 1.0, 1e-9)
    assert next(it) > 5
    capped = nb._level_schedule(6)
    it2 = iter(capped)
    for _ in range(5):
        next(it2)
    capped.jump(4, 1e30, 1e-9)
    assert next(it2) == 5
    assert list(it2) == [6]


def test_switching_profiles_share_their_window(nb):
    """cos2 is exactly compact on [0, W] (the S_f = 0 cosine ramps); the
    gaussian is truncated there at a stated 3.4e-4 of its peak."""
    W = 10.0
    c = nb.switching_profile("cos2", W)
    g = nb.switching_profile("gaussian", W)
    assert c.support == pytest.approx((0.0, W))
    assert c(0.0) == pytest.approx(0.0, abs=1e-15)
    assert c(0.5 * W) == pytest.approx(1.0)
    # sin^2(pi t / W) == cos2 profile: the transcribed shape, pointwise
    for t in (0.1 * W, 0.3 * W, 0.7 * W):
        assert c(t) == pytest.approx(math.sin(math.pi * t / W) ** 2, rel=1e-12)
    assert g(0.5 * W) == pytest.approx(1.0)
    edge = g(0.0)
    assert edge == pytest.approx(math.exp(-0.5 * nb.GAUSSIAN_HALF_WIDTHS**2), rel=1e-12)
    assert edge < 1e-3
    with pytest.raises(ValueError):
        nb.switching_profile("tophat", W)
