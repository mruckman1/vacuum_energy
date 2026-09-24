#!/usr/bin/env python
"""Run the L6 free-tier batch — the QET information signal, 3 qubits, 11
circuits, 3 439 shots each — on an IBM Quantum device, persist the raw
counts with provenance, and print the audited verdict.

    # the real thing (needs a free IBM Quantum account and token):
    export IBM_QUANTUM_TOKEN=...            # or a saved account
    .venv/bin/python scripts/run_ibm_free_tier.py

    # rehearsal on a fake backend with a real calibration snapshot:
    .venv/bin/python scripts/run_ibm_free_tier.py --dry-run

    # just the budget:
    .venv/bin/python scripts/run_ibm_free_tier.py --estimate-only

    # submit now, collect later (the Open Plan queue can be hours):
    .venv/bin/python scripts/run_ibm_free_tier.py --no-wait
    .venv/bin/python scripts/run_ibm_free_tier.py --collect <job id>

    # re-analyse a persisted record:
    .venv/bin/python scripts/run_ibm_free_tier.py --analyze papers/hardware-as-noisy-field/data/real_device/<file>.json

Records go to ``papers/hardware-as-noisy-field/data/real_device/<backend>_<job
id>.json`` (dry runs: ``.../real_device/dry_run/``).  Exit status: 0 the
claim holds (E_B_info > 0 at n sigma and every audit passed), 1 the signal
is not positive at n sigma, 3 an audit failed (an event — see PLAN.md's
anomaly protocol), 2 no service / bad arguments.  Everything numerical is
:mod:`vacuum.hardware.real_device`; this file only parses arguments,
chooses the backend and prints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vacuum.hardware import real_device as rd  # noqa: E402

DATA_DIR = ROOT / "papers" / "hardware-as-noisy-field" / "data" / "real_device"

TOKEN_HELP = """\
No IBM Quantum service available.  To run on a real device:
  1. create a free IBM Quantum account (https://quantum.cloud.ibm.com), Open Plan
     (10 min of QPU time per 28-day window), and copy the API token;
  2. .venv/bin/pip install -e '.[ibm]'      (qiskit + qiskit-aer + qiskit-ibm-runtime)
  3. export IBM_QUANTUM_TOKEN=<token>      (optionally IBM_QUANTUM_INSTANCE=<CRN>)
     or save it once:  python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; \\
                        S.save_account(channel='ibm_quantum_platform', token='<token>')"
  4. .venv/bin/python scripts/run_ibm_free_tier.py
Rehearse without an account:  .venv/bin/python scripts/run_ibm_free_tier.py --dry-run
"""


def _fmt(v, s):
    return f"{v:+.5f} +- {s:.5f}"


def print_batch(job):
    print(f"batch: {job.n_circuits} circuits ({rd.FREE_TIER_SPEC['measurement_circuits']} measurement + "
          f"{rd.FREE_TIER_SPEC['calibration_circuits']} calibration) on {job.n_qubits} qubits; "
          f"native ({'/'.join(rd.pr.IBM_BASIS_GATES)}) depth {job.native_depth}, CNOTs {job.native_cx} "
          f"(spec: depth {rd.FREE_TIER_SPEC['depth_native']}, <= {rd.FREE_TIER_SPEC['cx_count_max']} CNOTs)")
    if job.isa_stats is not None:
        print(f"ISA for {job.backend_name}: physical qubits {job.physical_qubits} "
              f"(Alice, Bob, spectator), two-qubit gate {job.two_qubit_gate}, depth {job.isa_depth}, "
              f"two-qubit gates {job.isa_two_qubit} per circuit (max); layout score {job.layout_score['score']:.4f}")


def print_analysis(an, pred=None):
    b = an["backend"]
    print(f"\n=== {b} ({'SIMULATED (fake backend, real calibration snapshot)' if an['simulated'] else 'REAL DEVICE'}), "
          f"{an['shots']} shots/circuit, mitigation {an['mitigation']} ===")
    print(f"E_A        = {_fmt(an['E_A'], an['E_A_std_error'])}   (dense {an['dense']['E_A']:+.5f})")
    print(f"E_B_ctrl   = {_fmt(an['E_B_ctrl'], an['E_B_ctrl_std_error'])}   (dense E_B {an['dense']['E_B']:+.5f}; teleportation proper)")
    print(f"E_B_raw    = {_fmt(an['E_B_raw'], an['E_B_raw_std_error'])}   (Ikeda's raw definition)")
    print(f"E_B_info   = {_fmt(an['E_B_info'], an['E_B_info_std_error'])}   z = {an['z']['E_B_info']:.2f}  "
          f"(ideal {an['prediction']['ideal_value']:+.5f}; sourced-noise prediction {an['prediction']['sourced_noise_value']:+.5f}, "
          f"pull {an['prediction']['pull_vs_sourced_noise']:+.2f})")
    print(f"  raw (unmitigated) E_B_info = {_fmt(an['verdict']['raw']['E_B_info'], an['verdict']['raw']['std_error'])}")
    c = an["closure"]
    print(f"audits: energy ledger defect {c['defect']:+.4f} (sigma {c['sigma']:.4f}) -> {'PASS' if c['passed'] else 'FAIL'}; "
          f"no-signaling marginals worst chi2/crit {an['no_signaling']['worst_ratio']:.2f} -> "
          f"{'PASS' if an['no_signaling']['passed'] else 'FAIL'}; "
          f"hardware_qet (closure, E_A >= E_B, no-signaling) -> {'PASS' if an['audits']['hardware_qet']['passed'] else 'FAIL'}; "
          f"signal vs noiseless ideal (one-sided) pull {an['prediction']['pull_vs_ideal_audited']:+.2f} -> "
          f"{'PASS' if an['audits']['signal_vs_ideal']['passed'] else 'FAIL'}")
    if pred is not None:
        sv = pred["survival"]
        print(f"shot-free prediction at this calibration: E_B_info = {_fmt(sv['signal'], sv['sigma'])}, z = {sv['z']:.2f}, "
              f"{'alive' if sv['alive'] else 'dead'} at {sv['n_sigma']:g} sigma; shots for {sv['n_sigma']:g} sigma: {sv['shots_for_n_sigma']}; "
              f"sampled pull vs prediction {(an['E_B_info'] - sv['signal']) / an['E_B_info_std_error']:+.2f}")
    print("\nVERDICT:", an["verdict"]["statement"])


def preflight(job, backend, shots, n_sigma):
    """Shot-free prediction at the backend's live calibration (Aer from
    ``backend.target``, ~5 s locally): the spec's 3 439 shots were sized on
    the sourced ibm_cairo calibration, so say what THIS calibration needs."""
    try:
        pred = rd.predict_on_backend(job, backend, shots, n_sigma=n_sigma)
    except Exception as exc:  # pragma: no cover - backend-specific
        print(f"(no shot-free pre-flight prediction: {type(exc).__name__}: {exc})")
        return None
    sv = pred["survival"]
    need3 = sv["shots_for_n_sigma"]
    need5 = rd.nz_shots(sv["signal"], sv["sigma"], shots, 5.0)
    print(f"pre-flight at this calibration (shot-free Aer on the ISA circuits): E_B_info = {sv['signal']:+.4f}, "
          f"sigma at {shots} shots = {sv['sigma']:.4f}, z = {sv['z']:.2f} -> {'alive' if sv['alive'] else 'NOT alive'} "
          f"at {n_sigma:g} sigma; shots per circuit for {n_sigma:g} sigma: {need3}, for 5 sigma: {need5} "
          f"(QPU time {rd.qpu_time_estimate(job.n_circuits, need5)['seconds']:.0f} s)")
    if not sv["alive"]:
        print(f"  -> consider --shots {need5} (still inside the Open Plan budget: "
              f"{rd.qpu_time_estimate(job.n_circuits, need5)['fits_open_plan_budget']})")
    return pred


def exit_code(an):
    v = an["verdict"]
    if not v["audits_passed"]:
        return 3
    return 0 if v["positive_at_n_sigma"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="run on a fake backend (local Aer with its calibration)")
    ap.add_argument("--fake", default="FakeSherbrooke", help="fake backend class for --dry-run (default FakeSherbrooke)")
    ap.add_argument("--backend", default=None, help="IBM backend name (default: least busy operational device)")
    ap.add_argument("--instance", default=None, help="IBM Cloud instance CRN (or IBM_QUANTUM_INSTANCE)")
    ap.add_argument("--shots", type=int, default=rd.FREE_TIER_SPEC["shots_per_circuit_for_3sigma"],
                    help="shots per circuit (default: the spec's 3439)")
    ap.add_argument("--seed", type=int, default=None,
                    help="simulator seed (dry runs only; default 1; 0 is refused — the local sampler ignores it)")
    ap.add_argument("--physical-qubits", default=None, help="comma-separated path of 3 physical qubits (default: lowest-noise path)")
    ap.add_argument("--out", default=None, help="output directory (default: the paper's data/real_device[/dry_run])")
    ap.add_argument("--no-wait", action="store_true", help="submit and exit; collect later with --collect")
    ap.add_argument("--collect", metavar="JOB_ID", default=None, help="fetch a submitted job and analyse it")
    ap.add_argument("--submission", metavar="FILE", default=None, help="the .submitted.json sidecar for --collect")
    ap.add_argument("--analyze", metavar="RECORD", default=None, help="re-analyse a persisted record")
    ap.add_argument("--estimate-only", action="store_true", help="print the batch and the QPU-time estimate, submit nothing")
    ap.add_argument("--n-sigma", type=float, default=3.0)
    ap.add_argument("--json", action="store_true", help="also print the analysis as JSON")
    ap.add_argument("--save-analysis", action="store_true",
                    help="write the analysis next to the record as <record>.analysis.json")
    args = ap.parse_args(argv)

    if args.analyze:
        an = rd.analyze(rd.load_record(args.analyze), n_sigma=args.n_sigma)
        print_analysis(an)
        if args.json:
            print(json.dumps(an, indent=1))
        if args.save_analysis:
            out = rd.save_analysis(an, rd.analysis_path(args.analyze))
            print(f"analysis -> {out}")
        return exit_code(an)

    physical = None if args.physical_qubits is None else [int(x) for x in args.physical_qubits.split(",")]
    est = rd.qpu_time_estimate(rd.FREE_TIER_SPEC["circuits"], args.shots)
    print(f"QPU-time estimate: {est['seconds']:.1f} s for {est['circuits']} x {est['shots_per_circuit']} executions "
          f"({est['formula']}); Open Plan budget {est['open_plan_budget_s']:.0f} s per 28 days -> "
          f"{'fits' if est['fits_open_plan_budget'] else 'DOES NOT FIT'}")

    if args.estimate_only:
        job = rd.build_free_tier_job()
        print_batch(job)
        print(f"number tested: {rd.FREE_TIER_SPEC['number_tested']} — predicted {rd.FREE_TIER_SPEC['predicted_value_at_sourced_noise']:+.4f} "
              f"at the sourced (ibm_cairo) noise, ideal {rd.FREE_TIER_SPEC['ideal_value']:+.4f}")
        return 0

    if args.dry_run:
        backend = rd.fake_backend(args.fake)
        out_dir = Path(args.out) if args.out else DATA_DIR / "dry_run"
        seed = 1 if args.seed is None else args.seed
        cal = backend.properties()
        print(f"DRY RUN on {backend.name} (fake backend; calibration snapshot of {getattr(cal, 'last_update_date', None)}), "
              f"local qiskit-ibm-runtime SamplerV2 on Aer, seed {seed}")
        job = rd.build_free_tier_job(backend=backend, physical_qubits=physical)
        print_batch(job)
        out = rd.run_batch(backend, args.shots, seed=seed, out_dir=out_dir, job=job, n_sigma=args.n_sigma)
        pred = rd.predict_on_backend(job, backend, args.shots, n_sigma=args.n_sigma)
        print(f"job id {out['record']['job']['job_id']}; record -> {out['path']}")
        print_analysis(out["analysis"], pred)
        if args.json:
            print(json.dumps(out["analysis"], indent=1))
        return exit_code(out["analysis"])

    # ---- real service ----
    try:
        service = rd.runtime_service(instance=args.instance)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}\n\n{TOKEN_HELP}", file=sys.stderr)
        return 2
    out_dir = Path(args.out) if args.out else DATA_DIR

    if args.collect:
        sub_path = Path(args.submission) if args.submission else None
        if sub_path is None:
            hits = sorted(out_dir.glob(f"*_{args.collect}.submitted.json"))
            if len(hits) != 1:
                print(f"need --submission: found {len(hits)} sidecar(s) for job {args.collect} in {out_dir}", file=sys.stderr)
                return 2
            sub_path = hits[0]
        info = rd.load_submission(sub_path)
        record = rd.collect(args.collect, info, service=service)
        path = rd.save_record(record, rd.default_record_path(out_dir, record))
        print(f"record -> {path}")
        an = rd.analyze(record, n_sigma=args.n_sigma)
        print_analysis(an)
        if args.json:
            print(json.dumps(an, indent=1))
        if args.save_analysis:
            print(f"analysis -> {rd.save_analysis(an, rd.analysis_path(path))}")
        return exit_code(an)

    backend = service.backend(args.backend) if args.backend else rd.least_busy_backend(service)
    cal = None
    try:
        cal = backend.properties()
    except Exception:
        pass
    print(f"backend {backend.name} ({backend.num_qubits} qubits, {rd.two_qubit_gate_name(backend)}), "
          f"calibration {getattr(cal, 'last_update_date', 'n/a')}")
    job = rd.build_free_tier_job(backend=backend, physical_qubits=physical)
    print_batch(job)
    pre = preflight(job, backend, args.shots, args.n_sigma)
    sub = rd.submit(job, backend, args.shots, tags=["vacuum-program", "L6-free-tier"])
    if pre is not None:   # lesson of the ibm_fez run: the pre-flight was printed, not persisted
        sub.info["preflight"] = {"survival": pre["survival"], "E_A": pre["E_A"], "E_B_ctrl": pre["E_B_ctrl"],
                                 "E_B_raw": pre["E_B_raw"], "E_B_info": pre["E_B_info"],
                                 "E_B_info_std_error": pre["E_B_info_std_error"],
                                 "calibration_last_update": pre["record"]["backend"]["calibration"].get("last_update_date"),
                                 "method": pre["record"]["job"]["sampler"]}
    sub_path = rd.save_submission(sub, rd.default_record_path(out_dir, sub, ".submitted.json"))
    print(f"submitted job {sub.job_id} on {backend.name}; sidecar -> {sub_path}")
    if args.no_wait:
        print(f"collect later with:\n  .venv/bin/python scripts/run_ibm_free_tier.py --collect {sub.job_id}")
        return 0
    print("waiting for the result (the Open Plan queue can take a while; Ctrl-C and --collect later is safe) ...")
    record = rd.collect(sub.runtime_job, sub)
    path = rd.save_record(record, rd.default_record_path(out_dir, record))
    print(f"record -> {path}")
    an = rd.analyze(record, n_sigma=args.n_sigma)
    print_analysis(an)
    if args.json:
        print(json.dumps(an, indent=1))
    if args.save_analysis:
        print(f"analysis -> {rd.save_analysis(an, rd.analysis_path(path))}")
    return exit_code(an)


if __name__ == "__main__":
    sys.exit(main())
