"""Tests for vacuum.hardware.real_device — the L6 free-tier batch on a real
device, rehearsed end to end on a fake backend and then run once on
``ibm_fez`` (PLAN.md Layer 7, L6; anchor 7 below is that run's record).

Needs the ``.[ibm]`` extra, which carries ``qiskit-ibm-runtime`` (its local testing
mode and ``fake_provider``); skipped without them.  Nothing here contacts
a device: the "run" is ``qiskit_ibm_runtime.SamplerV2(mode=FakeSherbrooke())``,
whose calibration snapshot is a real, dated one.

Anchors (every measured number is printed in the assertion message):

1. The batch is the spec.  ``FREE_TIER_SPEC`` equals the archived
   ``s6_free_tier.json``; the built batch has 11 circuits (9 measurement +
   2 calibration) on 3 qubits, native depth exactly the archived 36 and
   CNOTs exactly the archived 8; the working point's dense E_B_info is the
   archived ideal value; the QPU-time estimate reproduces the archived
   15.24 s / 387 s.
2. The transpiled budget holds.  On FakeSherbrooke the ISA circuits have
   <= 8 two-qubit gates, exactly the native CNOT count each (no routing on
   the chosen path), and read the chosen physical qubits; the noise-selected
   path is a coupling-map path and beats a fixed alternative; off-spec
   budgets raise.
3. The dry run agrees with the shot-free survival analysis at that
   calibration: same verdict at 20 000 shots (z ~ 7 shot-free), sampled
   E_B_info / E_A / E_B_ctrl within 4 sigma of the density-matrix values,
   every audit passed, provenance persisted (backend, calibration date,
   job id, shots, per-circuit depth / two-qubit counts).
4. Noiseless, the analysis is statevector-exact: E_A, E_B_ctrl, E_B_info,
   E_B_raw, the full ledger energies and the closure agree with the dense
   arbiter to 1e-10 and the audits pass.
5. The record survives a save / load round trip, its sha256 guards it, and
   structural corruption is rejected.
6. Corrupted counts fail the audits: Bob's bit flipped in the control
   family fails the ledger closure, the no-signaling marginals and the
   standing hardware_qet audit (AuditViolation in strict mode); Alice's bit
   flipped in the ground family fails E_A >= E_B; a corrupted *wrong*
   family — which the other three audits cannot see, since it enters
   neither the closure nor the marginal test — fails ``signal_vs_ideal``
   (E_B_info more than 3 sigma above its noiseless value) and so returns
   claim_holds = False, and the ideal it is measured against is recomputed
   from the record's own VQE parameters (editing the record's stated
   prediction does not buy the check off).
7. The real run.  ``data/real_device/ibm_fez_dadkfvl1ierc738l0ch0.json`` —
   the first real-device fixture in the repository (ibm_fez, 156 qubits,
   cz, calibration 2026-09-04 14:37 PDT, physical qubits [136, 143, 142],
   9 600 shots per circuit, IBM usage meter 30 s) — validates against its
   sha256, its dense ledger is exactly the one its VQE parameters give, and
   the audited analysis re-derives the archived verdict bit for bit:
   E_B_info = +0.04853 +- 0.00779 (z = 6.23) at 5 sigma, all four audits
   passed, E_A = +0.84999 +- 0.00193, E_B_ctrl = -0.01381 +- 0.00755 and
   E_B_raw = -0.02897 +- 0.00569 (teleportation proper lost, as the survival
   map predicted), the unmitigated E_B_info within 0.2 sigma of the
   mitigated one, and the stored ``.analysis.json`` equal to a fresh one.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ibm_runtime")

from vacuum.audits import AuditViolation  # noqa: E402
from vacuum.hardware import protocols as pr  # noqa: E402
from vacuum.hardware import real_device as rd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S6 = ROOT / "papers" / "hardware-as-noisy-field" / "data" / "s6_free_tier.json"
REAL = ROOT / "papers" / "hardware-as-noisy-field" / "data" / "real_device" / "ibm_fez_dadkfvl1ierc738l0ch0.json"
DRY_SHOTS = 20000

#: the archived verdict of the real run (papers/hardware-as-noisy-field/MANIFEST.md,
#: "Real-device run"); the analysis must re-derive every number from the counts
REAL_RUN = {
    "sha256": "a3499fbaf74a98ee0bbdf2ae78e025623efdc52fa821f6de43415765b9f535f1",
    "n_sigma": 5.0, "shots": 9600, "physical_qubits": [136, 143, 142], "two_qubit_gate": "cz",
    "calibration_last_update": "2026-09-04T14:37:12-07:00",
    "E_B_info": (0.04853, 0.00779), "z_E_B_info": 6.23,
    "E_A": (0.84999, 0.00193), "E_B_ctrl": (-0.01381, 0.00755), "E_B_raw": (-0.02897, 0.00569),
    "raw_E_B_info": (0.04715, 0.00765),
    "closure_defect": (-0.0031, 0.0087), "signal_vs_ideal_pull": 0.90,
    "no_signaling_ratio_at_3sigma": 0.48, "no_signaling_ratio_at_5sigma": 0.21,
    "pull_vs_sourced_noise": 1.08, "ideal": 0.04152, "sourced_prediction": 0.04010,
    "usage_s": 30,
}


@pytest.fixture(scope="module")
def s6():
    with open(S6) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def job():
    return rd.build_free_tier_job()


@pytest.fixture(scope="module")
def fake():
    return rd.fake_backend("FakeSherbrooke")


@pytest.fixture(scope="module")
def isa_job(job, fake):
    rd.transpile_for_backend(job, fake)
    return job


@pytest.fixture(scope="module")
def prediction(isa_job, fake):
    return rd.predict_on_backend(isa_job, fake, DRY_SHOTS)


@pytest.fixture(scope="module")
def dry(isa_job, fake, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("real_device_dry")
    return rd.run_batch(fake, DRY_SHOTS, seed=1, out_dir=out_dir, job=isa_job)


# ---------------------------------------------------------------------------
# 1. the batch is the spec
# ---------------------------------------------------------------------------


class TestSpec:
    def test_spec_constants_are_the_archived_file(self, s6):
        q = s6["qet_information_signal"]
        spec = rd.FREE_TIER_SPEC
        assert spec["qubits"] == q["qubits"] == 3
        assert spec["circuits"] == q["circuits"] == 11
        assert spec["depth_native"] == q["depth_native"]
        assert spec["cx_count_max"] == q["cx_count_max"]
        assert spec["shots_per_circuit_for_3sigma"] == q["shots_per_circuit_for_3sigma_at_sourced_noise"]
        assert spec["predicted_value_at_sourced_noise"] == q["predicted_value_at_sourced_noise"]
        assert spec["predicted_std_error_at_sourced_noise"] == q["predicted_std_error"]
        assert spec["predicted_std_error_shots"] == q["shots_per_circuit_used"]
        assert spec["ideal_value"] == q["ideal_value"]
        assert spec["open_plan_budget_s"] == s6["open_plan_budget_s"]
        est = rd.qpu_time_estimate(spec["circuits"], spec["shots_per_circuit_for_3sigma"])
        assert abs(est["seconds"] - q["qpu_time_s_at_3sigma_shots"]) < 1e-9, est
        assert est["fits_open_plan_budget"] is True
        est_used = rd.qpu_time_estimate(spec["circuits"], q["shots_per_circuit_used"])
        assert abs(est_used["seconds"] - q["qpu_time_s_at_used_shots"]) < 1e-9, est_used

    def test_batch_matches_spec(self, job, s6):
        q = s6["qet_information_signal"]
        kinds = [l["kind"] for l in job.labels]
        assert job.n_circuits == 11 and kinds.count("measurement") == 9 and kinds.count("calibration") == 2
        assert job.n_qubits == 3 and all(qc.num_qubits == 3 for qc in job.circuits)
        assert job.native_depth == q["depth_native"], f"native depth {job.native_depth} vs archived {q['depth_native']}"
        assert job.native_cx == q["cx_count_max"], f"native CNOTs {job.native_cx} vs archived {q['cx_count_max']}"
        assert all(s["cx_count"] <= 8 for s in job.native_stats)
        fams = [l["family"] for l in job.labels if l["kind"] == "measurement"]
        assert fams == ["ground"] * 3 + ["control"] * 2 + ["protocol"] * 2 + ["wrong"] * 2, fams
        assert [l["operator"] for l in job.labels[:3]] == ["D_A", "H_B", "H_B"]
        assert all(qc.cregs[-1].name == rd.READOUT_REGISTER for qc in job.circuits)
        assert [l["prepared"] for l in job.labels if l["kind"] == "calibration"] == ["000", "111"]
        assert abs(job.dense["E_B_info"] - q["ideal_value"]) < 1e-9, (job.dense["E_B_info"], q["ideal_value"])
        assert rd.assert_spec(job) is True
        with pytest.raises(rd.BudgetError):
            rd.assert_spec(job, spec={**rd.FREE_TIER_SPEC, "cx_count_max": 7})
        with pytest.raises(rd.BudgetError):
            rd.assert_spec(job, spec={**rd.FREE_TIER_SPEC, "depth_native": 35})
        with pytest.raises(ValueError):
            rd.working_point({"N": 2})


# ---------------------------------------------------------------------------
# 2. the transpiled budget on a real coupling map / basis
# ---------------------------------------------------------------------------


class TestTranspiled:
    def test_isa_budget_holds(self, isa_job, fake):
        job = isa_job
        assert job.backend_name == fake.name and job.backend_kind == "fake"
        assert job.two_qubit_gate in rd.TWO_QUBIT_GATES and job.two_qubit_gate in fake.target.operation_names
        assert job.isa_two_qubit <= rd.FREE_TIER_SPEC["cx_count_max"], job.isa_two_qubit
        for lab, st, ns in zip(job.labels, job.isa_stats, job.native_stats):
            assert st["two_qubit_count"] == ns["cx_count"], (lab["name"], st, ns)
            assert st["ops"].get("cx", 0) == 0, "the ISA circuit must use the backend's native two-qubit gate"
        assert rd._is_path(fake, job.physical_qubits)
        for lab, tqc in zip(job.labels, job.isa):
            assert pr._readout_qubits(tqc) == [job.physical_qubits[k] for k in lab["measured"]]
            assert tqc.num_qubits == fake.num_qubits
        assert job.isa_depth >= job.native_depth - 10  # ECR/CZ decompositions do not shrink the depth much

    def test_layout_selection(self, fake):
        path, score = rd.select_physical_qubits(fake, 3)
        assert len(path) == 3 and rd._is_path(fake, path) and score["complete"]
        assert score["n_candidates"] > 100
        assert abs(rd._layout_score(fake, path)["score"] - score["score"]) < 1e-15
        alt = rd._layout_score(fake, [0, 1, 2])
        assert alt["score"] >= score["score"], (alt["score"], score["score"])
        assert all(v is not None and v > 0 for v in score["edge_errors"].values())
        with pytest.raises(ValueError):
            rd.transpile_for_backend(rd.build_free_tier_job(), fake, physical_qubits=[0, 2, 4])

    def test_calibration_snapshot(self, isa_job, fake):
        snap = rd.calibration_snapshot(fake, isa_job.physical_qubits)
        assert snap["last_update_date"].startswith("2025-02-26"), snap["last_update_date"]
        assert snap["source"].startswith("backend.properties()")
        assert len(snap["properties_sha256"]) == 64
        for q in isa_job.physical_qubits:
            row = snap["qubits"][str(q)]
            assert row["t1_us"] > 0 and row["t2_us"] > 0 and row["readout_error"] is not None
            assert row["sx_error"] is not None and row["properties_prob_meas1_prep0"] is not None
        assert len(snap["edges"]) >= 2 and all(e["gate"] == isa_job.two_qubit_gate for e in snap["edges"].values())


# ---------------------------------------------------------------------------
# 3. dry run vs the shot-free survival analysis at that calibration
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_shot_free_prediction_is_audited(self, prediction, job):
        sv = prediction["survival"]
        assert sv["shots"] == DRY_SHOTS and sv["alive"] and sv["z"] > 5.0, sv
        assert 0 < sv["shots_for_n_sigma"] < DRY_SHOTS
        assert prediction["audits"]["all_passed"]
        # the from_backend noise model is local (no crosstalk, no idle decay):
        # shot-free, the closure and the no-signaling marginals are exact
        assert abs(prediction["closure"]["defect"]) < 1e-9, prediction["closure"]
        assert prediction["no_signaling"]["worst_ratio"] < 1e-9
        assert abs(prediction["E_A"] - job.dense["E_A"]) < 0.05 * job.dense["E_A"]
        assert 0.0 < prediction["E_B_info"] < job.dense["E_B_info"] + 0.01

    def test_verdict_agrees_with_prediction(self, dry, prediction):
        an = dry["analysis"]
        sv = prediction["survival"]
        assert an["verdict"]["positive_at_n_sigma"] == sv["alive"], (an["verdict"], sv)
        assert an["verdict"]["audits_passed"] and an["verdict"]["claim_holds"]
        for key in ("E_B_info", "E_A", "E_B_ctrl"):
            pull = (an[key] - prediction[key]) / an[key + "_std_error"]
            assert abs(pull) < 4.0, f"{key}: sampled {an[key]:.5f} vs shot-free {prediction[key]:.5f}, pull {pull:.2f}"
        assert an["z"]["E_B_info"] > 3.0
        assert rd.verdict(an) == an["verdict"]

    def test_provenance_persisted(self, dry, fake):
        rec = dry["record"]
        path = dry["path"]
        assert path.exists() and path.name == f"{fake.name}_{rec['job']['job_id']}.json"
        assert dry["submission_path"].exists() and rd.load_submission(dry["submission_path"])["job_id"] == rec["job"]["job_id"]
        assert rec["format"] == rd.RECORD_FORMAT and rec["backend"]["name"] == fake.name
        assert rec["backend"]["kind"] == "fake" and rec["backend"]["simulated"] is True
        assert rec["backend"]["calibration"]["last_update_date"].startswith("2025-02-26")
        assert rec["job"]["job_id"] and rec["job"]["shots"] == DRY_SHOTS and rec["job"]["submitted_utc"]
        assert rec["job"]["sampler"].endswith("SamplerV2") and rec["job"]["seed"] == 1
        assert rec["environment"]["qiskit_ibm_runtime"] and rec["environment"]["qiskit"]
        assert len(rec["circuits"]) == 11
        for c in rec["circuits"]:
            assert sum(c["counts"].values()) == DRY_SHOTS
            assert c["native"]["depth"] > 0 and c["isa"]["two_qubit_count"] <= 8
        assert rec["working_point"]["theta"] == dry["job"].theta
        with pytest.raises(ValueError, match="seed 0"):
            rd.submit(dry["job"], fake, 100, seed=0)   # silently non-reproducible in local mode
        assert rec["spec"] == rd.FREE_TIER_SPEC


# ---------------------------------------------------------------------------
# 4. noiseless: exact against the dense arbiter
# ---------------------------------------------------------------------------


class TestNoiseless:
    def test_analysis_is_statevector_exact(self, isa_job, fake):
        job = isa_job
        shots = rd.FREE_TIER_SPEC["shots_per_circuit_for_3sigma"]
        counts = rd.exact_counts(job, shots)
        info = rd.submission_info(job, rd.backend_info(fake, job.physical_qubits), shots, "noiseless", "Statevector")
        rec = rd.build_record(info, counts, {"shot_free": True, "noiseless": True})
        an = rd.analyze(rec, strict=True)
        d = job.dense
        for key, dkey in (("E_A", "E_A"), ("E_B_ctrl", "E_B"), ("E_B_info", "E_B_info"), ("E_B_raw", "E_B_raw"),
                          ("HB_ground", "HB_ground"), ("HB_meas", "HB_meas"), ("HB_final", "HB_final"),
                          ("HB_wrong", "HB_wrong"), ("E_B_cut", "E_B_cut")):
            assert abs(an[key] - d[dkey]) < 1e-10, (key, an[key], d[dkey])
        for key, dkey in (("ground", "E_ground"), ("meas", "E_meas"), ("final", "E_final")):
            assert abs(an["full_energies"][key]["value"] - d[dkey]) < 1e-10, (key, an["full_energies"][key], d[dkey])
        assert abs(an["closure"]["defect"]) < 1e-10 and an["no_signaling"]["worst_ratio"] < 1e-12
        assert all(np.allclose(m, np.eye(2), atol=1e-12) for m in an["readout_calibration"])
        assert an["audits"]["all_passed"] and an["verdict"]["positive_at_n_sigma"]
        assert an["ledger"]["raw"]["E_B_info"] == pytest.approx(an["ledger"]["mitigated"]["E_B_info"], abs=1e-12)


# ---------------------------------------------------------------------------
# 5. record round trip and validation
# ---------------------------------------------------------------------------


class TestRecord:
    def test_roundtrip_and_hash(self, dry, tmp_path):
        rec = dry["record"]
        loaded = rd.load_record(dry["path"])
        assert loaded == rec and loaded["sha256"] == rd.record_sha256(loaded)
        again = rd.analyze(loaded)
        assert again["E_B_info"] == dry["analysis"]["E_B_info"]
        tampered = copy.deepcopy(rec)
        tampered["job"]["shots"] = DRY_SHOTS + 1
        with pytest.raises(ValueError, match="sha256"):
            rd.validate_record(tampered)
        p = rd.save_record(tampered, tmp_path / "t.json")
        with pytest.raises(ValueError):
            rd.load_record(p)
        short = copy.deepcopy(rec)
        short["circuits"] = short["circuits"][:10]
        short["sha256"] = rd.record_sha256(short)
        with pytest.raises(ValueError, match="circuits"):
            rd.validate_record(short)
        nokey = copy.deepcopy(rec)
        del nokey["backend"]["calibration"]
        nokey["sha256"] = rd.record_sha256(nokey)
        with pytest.raises(ValueError, match="calibration"):
            rd.validate_record(nokey)


# ---------------------------------------------------------------------------
# 6. corrupted counts fail the audits
# ---------------------------------------------------------------------------


def _flip_bit(counts, bit):
    """Flip classical bit ``bit`` (0 = rightmost) in every key."""
    out = {}
    for k, v in counts.items():
        i = len(k) - 1 - bit
        kk = k[:i] + ("1" if k[i] == "0" else "0") + k[i + 1:]
        out[kk] = out.get(kk, 0) + v
    return out


class TestCorruption:
    def test_bobs_bit_flipped_in_control_fails_everything(self, dry):
        bad = copy.deepcopy(dry["record"])
        for c in bad["circuits"]:
            if c["kind"] == "measurement" and c["family"] == "control" and c["group"] == 0:
                c["counts"] = _flip_bit(c["counts"], 1)
        with pytest.raises(ValueError, match="sha256"):
            rd.validate_record(bad)
        bad["sha256"] = rd.record_sha256(bad)
        with pytest.raises(AuditViolation):
            rd.analyze(bad, strict=True)
        an = rd.analyze(bad, strict=False)
        assert not an["audits"]["all_passed"]
        assert not an["audits"]["energy_ledger"]["passed"], an["closure"]
        assert not an["audits"]["no_signaling_marginals"]["passed"], an["no_signaling"]["worst_ratio"]
        assert not an["audits"]["hardware_qet"]["passed"]
        assert set(an["audits"]) == {"energy_ledger", "no_signaling_marginals", "signal_vs_ideal",
                                     "hardware_qet", "all_passed"}
        assert abs(an["closure"]["defect"]) > 3.0 * an["closure"]["sigma"]
        assert not an["verdict"]["claim_holds"]

    def test_corrupted_wrong_family_fails_signal_vs_ideal(self, dry):
        """The gap the other three audits cannot see: the 'wrong' family
        enters neither the ledger closure nor the marginal no-signaling
        test, so only the one-sided ideal bound stops a mis-shaped
        E_B_info from being reported as a verdict."""
        clean = dry["analysis"]
        assert clean["audits"]["signal_vs_ideal"]["passed"]
        assert abs(clean["prediction"]["pull_vs_ideal_audited"]) < 3.0, clean["prediction"]
        bad = copy.deepcopy(dry["record"])
        for c in bad["circuits"]:
            if c["kind"] == "measurement" and c["family"] == "wrong":
                c["counts"] = _flip_bit(c["counts"], 1)          # Bob's bit
        bad["sha256"] = rd.record_sha256(bad)
        an = rd.analyze(bad, strict=False)
        aud = an["audits"]
        assert an["E_B_info"] > an["dense"]["E_B_info"], an["E_B_info"]
        assert an["prediction"]["pull_vs_ideal_audited"] > 3.0, an["prediction"]
        assert not aud["signal_vs_ideal"]["passed"], aud["signal_vs_ideal"]["details"]
        assert not aud["all_passed"] and not an["verdict"]["audits_passed"]
        assert an["verdict"]["positive_at_n_sigma"], "the corrupted signal is positive — that is the point"
        assert an["verdict"]["claim_holds"] is False
        assert "signal_vs_ideal" in an["verdict"]["statement"]
        # the other three are blind to it (why this audit had to exist)
        assert aud["energy_ledger"]["passed"] and aud["no_signaling_marginals"]["passed"]
        with pytest.raises(AuditViolation) as exc:
            rd.analyze(bad, strict=True)
        assert exc.value.result.name == "signal_vs_ideal"
        # the ideal is recomputed from the record's own VQE parameters
        spoof = copy.deepcopy(bad)
        spoof["spec"] = {**spoof["spec"], "ideal_value": 10.0, "predicted_value_at_sourced_noise": 10.0}
        spoof["sha256"] = rd.record_sha256(spoof)
        an2 = rd.analyze(spoof, strict=False)
        assert not an2["audits"]["signal_vs_ideal"]["passed"], "the spec block must not buy the audit off"
        assert an2["audits"]["signal_vs_ideal"]["details"]["ideal"] == pytest.approx(an["dense"]["E_B_info"])

    def test_protocol_and_wrong_swapped_is_not_a_claim(self, dry):
        bad = copy.deepcopy(dry["record"])
        pc = {c["group"]: c["counts"] for c in bad["circuits"] if c.get("family") == "protocol"}
        wc = {c["group"]: c["counts"] for c in bad["circuits"] if c.get("family") == "wrong"}
        for c in bad["circuits"]:
            if c.get("family") == "protocol":
                c["counts"] = wc[c["group"]]
            elif c.get("family") == "wrong":
                c["counts"] = pc[c["group"]]
        bad["sha256"] = rd.record_sha256(bad)
        an = rd.analyze(bad, strict=False)
        assert an["E_B_info"] < 0, an["E_B_info"]
        assert not an["verdict"]["positive_at_n_sigma"] and an["verdict"]["claim_holds"] is False

    def test_alices_bit_flipped_in_ground_fails_E_A_ge_E_B(self, dry):
        bad = copy.deepcopy(dry["record"])
        for c in bad["circuits"]:
            if c["kind"] == "measurement" and c["family"] == "ground" and c["operator"] == "D_A":
                c["counts"] = _flip_bit(c["counts"], 0)
        bad["sha256"] = rd.record_sha256(bad)
        an = rd.analyze(bad, strict=False)
        assert an["E_A"] < 0, an["E_A"]
        chk = an["audits"]["hardware_qet"]["details"]["checks"]["E_A_ge_E_B"]
        assert chk["value"] > chk["tol"], chk
        assert not an["audits"]["hardware_qet"]["passed"] and not an["verdict"]["audits_passed"]
        with pytest.raises(AuditViolation):
            rd.analyze(bad, strict=True)


# ---------------------------------------------------------------------------
# 7. the real run: the persisted ibm_fez record is a fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_record():
    if not REAL.exists():
        pytest.skip(f"real-device record {REAL.name} not present")
    return rd.load_record(REAL)


class TestRealDeviceRecord:
    def test_record_is_the_committed_one(self, real_record):
        rec = real_record
        assert rec["sha256"] == REAL_RUN["sha256"] == rd.record_sha256(rec)
        b = rec["backend"]
        assert b["name"] == "ibm_fez" and b["kind"] == "ibm_runtime" and b["simulated"] is False
        assert b["physical_qubits"] == REAL_RUN["physical_qubits"] and b["two_qubit_gate"] == REAL_RUN["two_qubit_gate"]
        cal = b["calibration"]
        assert cal["last_update_date"] == REAL_RUN["calibration_last_update"]
        assert cal["source"].startswith("backend.properties()") and len(cal["properties_sha256"]) == 64
        for q in REAL_RUN["physical_qubits"]:
            row = cal["qubits"][str(q)]
            assert 100 < row["t1_us"] < 200 and row["readout_error"] < 0.01 and row["sx_error"] < 1e-3
        assert all(e["gate"] == "cz" and e["error"] < 3e-3 for e in cal["edges"].values())
        j = rec["job"]
        assert j["job_id"] == "dadkfvl1ierc738l0ch0" and j["shots"] == REAL_RUN["shots"] and j["seed"] is None
        assert j["sampler"].endswith("SamplerV2") and j["execution"]["status"] == "DONE"
        assert j["execution"]["usage"] == REAL_RUN["usage_s"]
        assert j["sampler_options"]["environment"]["job_tags"] == ["vacuum-program", "L6-free-tier"]
        assert all(v == "Unset" for v in j["sampler_options"]["twirling"].values()), "no server-side twirling"
        assert j["sampler_options"]["dynamical_decoupling"]["enable"] == "Unset", "no dynamical decoupling"
        assert len(rec["circuits"]) == 11 and all(sum(c["counts"].values()) == REAL_RUN["shots"] for c in rec["circuits"])
        assert max(c["isa"]["two_qubit_count"] for c in rec["circuits"]) == 8
        assert max(c["isa"]["depth"] for c in rec["circuits"]) == 40
        assert rec["spec"] == rd.FREE_TIER_SPEC

    def test_analysis_rederives_the_archived_verdict(self, real_record):
        an = rd.analyze(real_record, n_sigma=REAL_RUN["n_sigma"], strict=True)
        assert an["simulated"] is False and an["dense_consistency"] == 0.0
        for key in ("E_B_info", "E_A", "E_B_ctrl", "E_B_raw"):
            v, s_ = REAL_RUN[key]
            assert abs(an[key] - v) < 5e-6 and abs(an[key + "_std_error"] - s_) < 5e-6, (key, an[key], an[key + "_std_error"])
        assert abs(an["z"]["E_B_info"] - REAL_RUN["z_E_B_info"]) < 5e-3, an["z"]
        v, s_ = REAL_RUN["raw_E_B_info"]
        assert abs(an["verdict"]["raw"]["E_B_info"] - v) < 5e-6 and abs(an["verdict"]["raw"]["std_error"] - s_) < 5e-6
        # not a mitigation artifact: raw and mitigated agree well inside one sigma
        assert abs(an["E_B_info"] - an["verdict"]["raw"]["E_B_info"]) < 0.2 * an["E_B_info_std_error"]
        assert an["verdict"]["raw"]["E_B_info"] / an["verdict"]["raw"]["std_error"] > 5.0
        d, sd = REAL_RUN["closure_defect"]
        assert abs(an["closure"]["defect"] - d) < 5e-5 and abs(an["closure"]["sigma"] - sd) < 5e-5, an["closure"]
        assert abs(an["prediction"]["pull_vs_ideal_audited"] - REAL_RUN["signal_vs_ideal_pull"]) < 5e-3
        assert abs(an["prediction"]["pull_vs_sourced_noise"] - REAL_RUN["pull_vs_sourced_noise"]) < 5e-3
        assert abs(an["no_signaling"]["worst_ratio"] - REAL_RUN["no_signaling_ratio_at_5sigma"]) < 5e-3
        assert an["audits"]["all_passed"] and all(
            an["audits"][k]["passed"] for k in ("energy_ledger", "no_signaling_marginals", "signal_vs_ideal", "hardware_qet"))
        v = an["verdict"]
        assert v["positive_at_n_sigma"] and v["audits_passed"] and v["claim_holds"] and v["n_sigma"] == 5.0
        assert v["backend"] == "ibm_fez" and v["simulated"] is False
        # the survival map's honest negatives hold on the device
        assert an["E_B_ctrl"] < 0 and an["E_B_raw"] < 0
        assert an["E_B_ctrl"] + 3 * an["E_B_ctrl_std_error"] > 0 - 0.03, "E_B_ctrl is lost, not wildly negative"
        assert abs(an["E_B_ctrl"] - (-0.0112)) < 3 * an["E_B_ctrl_std_error"], "within the cairo-noise prediction"
        assert abs(an["E_B_raw"] - (-0.0264)) < 3 * an["E_B_raw_std_error"]
        # the 3-sigma reading of the same counts
        an3 = rd.analyze(real_record, n_sigma=3.0)
        assert abs(an3["no_signaling"]["worst_ratio"] - REAL_RUN["no_signaling_ratio_at_3sigma"]) < 5e-3
        assert an3["verdict"]["claim_holds"] and an3["E_B_info"] == an["E_B_info"]

    def test_stored_analysis_matches_a_fresh_one(self, real_record):
        path = rd.analysis_path(REAL)
        if not path.exists():
            pytest.skip("no stored analysis file")
        stored = rd.load_analysis(path)
        assert stored["record_sha256"] == real_record["sha256"]
        fresh = rd.analyze(real_record, n_sigma=stored["n_sigma"])
        for key in ("E_B_info", "E_B_info_std_error", "E_A", "E_A_std_error", "E_B_ctrl", "E_B_raw",
                    "HB_ground", "HB_meas", "HB_final", "HB_wrong", "E_B_cut"):
            assert fresh[key] == stored[key], key
        assert fresh["closure"] == stored["closure"] and fresh["verdict"] == stored["verdict"]
        assert fresh["audits"] == stored["audits"]

    def test_script_analyze_exit_code(self, real_record):
        import importlib.util

        spec = importlib.util.spec_from_file_location("run_ibm_free_tier", ROOT / "scripts" / "run_ibm_free_tier.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.main(["--analyze", str(REAL), "--n-sigma", "5"]) == 0
