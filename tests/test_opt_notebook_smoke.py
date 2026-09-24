"""Smoke test for the Layer-6 papers candidate: papers/differentiable-vacuum.

Runs ``notebook.main(['--quick', ...])`` — the same code path as the full
run at smoke size (one noise floor, one harmonic, a declared 1e-6 gate) —
and asserts the archive schema: build provenance, the recomputed anchors
within their tolerances, the PKMM/QET/QEI sections, a round-tripped
waveform row carrying its audits/convergence/check flags, exchange rates
and the inverse-design showcase.  Physics tolerances live in the anchor
test files; this guards the deliverable's shape.  Requires JAX; ~45 s.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "papers" / "differentiable-vacuum" / "notebook.py"


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("l6_differentiable_vacuum_notebook", NOTEBOOK)
    nb = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = nb
    spec.loader.exec_module(nb)
    out = tmp_path_factory.mktemp("nb")
    assert nb.main(["--quick", "--out", str(out)]) == 0
    s = json.loads((out / "summary_quick.json").read_text())
    arrays = np.load(out / "waveforms_quick.npz")
    return s, arrays


def test_build_and_anchors(summary):
    s, arrays = summary
    assert s["quick"] is True and len(s["build"]["generator_sha256"]) == 64 and s["build"]["build"]
    assert "__build__" in arrays.files
    a = s["anchors"]
    assert a["forward_max_dev_V"] < 1e-10 and a["forward_dev_E_N"] < 1e-10 and a["forward_dev_work"] < 1e-10
    assert a["grad_rel_err_lam"] < 1e-6 and a["grad_rel_err_gap"] < 1e-6
    assert a["distill_coded_vs_recurrence"] < 1e-12
    assert a["inverse_design_recovery_residual"] < 1e-8
    assert abs(s["pkmm"]["alpha_star"] - s["pkmm"]["alpha_star_bracketed"]) < 1e-4
    assert s["pkmm"]["death_line_rel_dev"] < 0.02
    assert max(r["abs_dev"] for r in s["qet"]) < 1e-8 and all(r["audits_passed"] for r in s["qet"])
    q = s["qei"]
    assert abs(q["tau0_star"] - q["tau0_numpy"]) < 5e-2 and abs(q["ratio_star"] - q["ratio_numpy"]) < 1e-5


def test_waveform_rows_schema(summary):
    s, arrays = summary
    assert len(s["waveforms"]) == 1
    res = s["waveforms"][0]
    for key in ("floor", "W_budget", "twin_work_bias", "comm_bound", "n_h", "splits", "maxiter", "max_halvings", "conv_tol", "rows"):
        assert key in res
    assert abs(res["twin_work_bias"] - 1.0) < 1e-3
    rows = res["rows"]
    assert set(rows) == {"baseline", "constrained"}
    for name, row in rows.items():
        for key in ("theta", "E_N", "work", "comm_fraction", "converged", "halvings", "dt_converged", "flagged", "regime",
                    "audits", "audits_passed", "E_N_jax", "work_jax", "comm_proxy", "min_nu_pt"):
            assert key in row, (name, key)
        assert row["audits"]["ledger"] and row["audits"]["precision"]
        assert row["converged"], (name, row["halvings"])
        assert abs(row["comm_proxy"] - row["comm_fraction"]) < 1e-6  # JAX split vs numpy split
    c = rows["constrained"]
    assert "gain" in c and "check" in c and "admissible" in c
    assert abs(c["work"] - res["W_budget"]) < 1e-5 * res["W_budget"]  # equal work by projection
    assert f"{res['floor']}__constrained__theta" in arrays.files
    # the baseline reproduces the noise-threshold candidate's archived base row at 30 mK (gated E_N)
    assert abs(rows["baseline"]["E_N"] - 0.010093115183475387) < 5e-7
    h = s["headline"]
    assert "pivot_verdict" in h and "signaling_bound" in h and "constrained_all_floors" in h


def test_exchange_rates_and_inverse_design(summary):
    s, _ = summary
    xr = s["exchange_rates"]
    assert {r["protocol"] for r in xr} == {"DEJMPS", "BBPSSW"}
    for r in xr:
        assert r["fidelity"] >= 0.99 or r["bell_pairs_per_run"] == 0.0
        assert r["rate"] >= 0.0 and r["n_rounds"] >= 0
    dej = [r for r in xr if r["protocol"] == "DEJMPS" and r["variant"] == "baseline"][0]
    bb = [r for r in xr if r["protocol"] == "BBPSSW" and r["variant"] == "baseline"][0]
    assert dej["rate"] > bb["rate"]
    inv = s["inverse_design"]
    assert inv["designs"] and abs(inv["designs"][0]["achieved"] - inv["designs"][0]["target"]) < 1e-4
    assert inv["designs"][0]["min_eigenvalue"] > 0 and inv["designs"][0]["audits"]["passivity"]
