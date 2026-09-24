#!/usr/bin/env python
"""Hardware as noisy field, section S2b — the (3,4) and (4,4) VQE rows.

The main notebook's ``s2`` left two rows unconverged: (N, d) = (3, 4) and
(4, 4) with the 3-layer hardware-efficient ansatz (energy error 2.3e-4 /
1.4e-3, infidelity 6.6e-5 / 3.2e-4; ``data/s2_vqe.json``).  This section
settles whether that was the optimizer or the ansatz, and prepares the
states to the standard of the other four rows:

1. **Expressibility ceiling by depth** (``vacuum_sim.fidelity_ceiling``):
   with the ED ground state in hand, maximise |<psi_ED|psi(theta)>|^2
   directly — Levenberg–Marquardt on the state residual with the exact
   Jacobian — from many seeded starts (near the Fock vacuum and uniform in
   (-pi, pi)), at every depth of a scan; plus a **layerwise-growing** scan
   (``fidelity_depth_scan``) that warm-starts each depth from the previous
   optimum embedded by ``hea_embed_layers``.  The best infidelity per depth
   is an upper bound on what the ansatz can reach there: at roundoff, the
   depth suffices (the state is exhibited); a plateau shared by every
   converged start is evidence, not proof, that it does not.
2. **Sufficient depth** L_suff: the smallest depth whose ENVELOPE (min over
   depths <= L, attainable there because the layer embedding is exact)
   reaches ACCEPT_INFIDELITY, after every depth has had a CONTINUATION from
   its own optimum at a larger Jacobian budget.  It is an upper bound, not a
   proved minimum: round 1 of this section published a depth one layer too
   deep because the depth below it had exited on ``maxiter`` while still
   descending.  The falsifiable half of the row is the **below probe**: at
   L_suff - 1, the incumbent plus fresh seeded starts and iterated local
   search at a stated large budget, archived with what it reached.
3. **Energy-VQE at L_suff** (``vqe_ground_state`` with ``warm_start`` = the
   fidelity optimum, tighter gtol, more iterations) on BOTH evaluators —
   Aer statevector with parameter-shift gradients, and the numpy arbiter
   with the exact adjoint gradient — each optimum cross-evaluated on the
   other evaluator (``ansatz_energy``).  Also the energy-VQE at the
   pinned depth L = 3 from the fidelity optimum there: the energy floor of
   the 3-layer ansatz with the optimizer failure removed.
4. **Downstream**: the protocol rows the main notebook computes at (3, 4)
   on ED states — the QET dense ledgers of ``s3a`` (A/B = 0/1 and 0/2) and
   the Gaussian-limit harvesting rows of ``s4b`` (N = 3, d = 4) — repeated
   on the VQE state, with the differences.  Nothing downstream runs at
   (4, 4); the file says so.

Long jobs are idempotent: every (case, depth) ceiling, every layerwise scan,
every VQE and every downstream row is one spec JSON and one result JSON
under ``data/jobs/`` (``s2b_<tag>.spec.json`` / ``s2b_<tag>.json``); a rerun
skips finished jobs, so a crash resumes instead of restarting.  Workers are
subprocesses of this file (``--job spec --job-out result``), ``--workers``
at a time, one BLAS/OpenMP thread each.

    .venv/bin/python papers/hardware-as-noisy-field/notebook_vqe_deep.py            # full; resumable
    .venv/bin/python papers/hardware-as-noisy-field/notebook_vqe_deep.py --quick    # smoke, temp dir
    .venv/bin/python papers/hardware-as-noisy-field/notebook_vqe_deep.py --out DIR  # elsewhere

Writes ``data/s2b_vqe_deep.json`` with ``__build__`` (UTC date, git HEAD,
generator sha256, wall time).  Nothing in ``data/s2_vqe.json`` or the other
``s*.json`` files is touched: the pinned rows stay as recorded.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

from vacuum.hardware import protocols as pr  # noqa: E402
from vacuum.hardware import vacuum_sim as vs  # noqa: E402

# --------------------------------------------------------------------------
# knobs (stated once; every row carries the ones it used)
# --------------------------------------------------------------------------

#: the standard the other four rows meet (MANIFEST numbers table): a depth is
#: "sufficient" when the fidelity maximisation reaches this infidelity
ACCEPT_INFIDELITY = 1e-12
#: the depth scan per unconverged case, in blocks: depths, random restarts per
#: depth, LM Jacobian-evaluation budget per start.  A generic real n-qubit
#: state has 2^n - 1 real degrees of freedom and the R_Y ladder ignores the
#: parity / reflection symmetries of the target, so the counting bound for
#: exact representability is n (L + 1) >= 2^n - 1: L >= 10 at (3,4)
#: (6 qubits, 63 dof) and L >= 31 at (4,4) (8 qubits, 255 dof).  The (4,4)
#: scan therefore covers the requested 3..12 and a sparse tail around 31.
CEILING_CASES = {
    (3, 4): [{"layers": list(range(1, 13)), "restarts": 8, "maxiter": 400}],
    (4, 4): [{"layers": list(range(3, 13)), "restarts": 8, "maxiter": 500},
             {"layers": [16, 20, 24, 28, 30, 31, 32, 36], "restarts": 3, "maxiter": 300}],
}
#: CONTINUATION jobs.  Round 1 of this unit published a sufficient depth one
#: layer too deep because the cold LM starts at the depth below it exited on
#: ``maxiter`` while still descending: the per-depth number was an
#: under-converged search, not the ansatz.  So EVERY scanned depth whose best
#: point has not reached ``CEILING_STOP_AT`` is re-optimised from its own
#: optimum with a larger Jacobian budget (tag ``..._cont``; a second stage
#: ``..._cont2`` for the named depths), and the archive records whether the
#: continuation converged (``xtol``/``ftol``) or exhausted its budget.
#: {(N, d): {"depths": "all" | [L, ...], "maxiter": {"default": M, L: M},
#:           "stage2": {L: M}}}
CEILING_CONTINUE = {
    (3, 4): {"depths": "all", "maxiter": {"default": 4000}, "stage2": {}},
    (4, 4): {"depths": "all", "maxiter": {"default": 1000, 16: 500, 20: 500, 24: 500, 28: 500, 30: 1200},
             "stage2": {30: 1000}},
}
#: BELOW PROBE: the falsifiable half of a "sufficient depth" row.  At the
#: largest scanned depth strictly below the sufficient one, a hard search --
#: the incumbent optimum plus fresh seeded starts and iterated local search at
#: a stated large budget -- is run and archived.  "L - 1 did not reach 1e-12"
#: then means "did not reach it at THESE knobs", a number the next reader can
#: rerun, instead of a silent budget shortfall.  {(N, d): knobs}
BELOW_PROBE = {
    (3, 4): {"restarts": 6, "perturb_rounds": 4, "maxiter": 2500},
    # 8 qubits, 248 parameters: one Jacobian evaluation costs ~1.2 s, so the
    # probe here is the incumbent plus iterated local search, not fresh starts.
    (4, 4): {"restarts": 0, "perturb_rounds": 1, "maxiter": 400},
}
#: every ceiling job stops early once a start reaches this (roundoff), and
#: adds this many iterated-local-search rounds when none does
CEILING_STOP_AT = 1e-13
CEILING_PERTURB_ROUNDS = 2
#: the four converged rows, at every depth up to the pinned one (context: the
#: same method reports roundoff where the energy-VQE converged)
CONTEXT_CASES = {
    (2, 2): [{"layers": [1], "restarts": 8, "maxiter": 400}],
    (3, 2): [{"layers": [1, 2], "restarts": 8, "maxiter": 400}],
    (4, 2): [{"layers": [1, 2, 3], "restarts": 8, "maxiter": 400}],
    (2, 4): [{"layers": [1, 2], "restarts": 8, "maxiter": 400}],
}
#: layerwise-growing scan: random restarts per depth in addition to the
#: embedded previous optimum; grows until ACCEPT_INFIDELITY or max depth
LAYERWISE = {(3, 4): {"restarts": 1, "max_layers": 12, "maxiter": 400},
             (4, 4): {"restarts": 1, "max_layers": 12, "maxiter": 300}}
#: energy-VQE at the minimal depth: cold restarts BEYOND the warm start (0: the
#: warm start only -- a cold L-BFGS-B run at 256 parameters through parameter-shift
#: batches costs ~5e5 circuit evaluations and cannot beat a start that is already at
#: roundoff; the cold-start behaviour of this ansatz is the s2 row and the
#: fidelity-ceiling restart statistics), iterations, gtol
VQE_KW = {"restarts": 0, "maxiter": 1000, "gtol": 1e-12}
VQE_EVALUATORS = (("aer", "shift"), ("numpy", "adjoint"))
#: downstream (3, 4) rows: the s3a QET cases and the s4b harvesting couplings
QET_CASES = [(0, 1), (0, 2)]
HARVEST_LAMS = (0.1, 0.4, 1.0)
HARVEST_SPEC = {"sites": (0, 1), "gap": 1.5, "width": 1.5, "n_steps": 6}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def build_info():
    src = Path(__file__).resolve()
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=HERE, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        head = None
    import scipy
    import vacuum
    try:
        import qiskit
        import qiskit_aer
        qv, av = qiskit.__version__, qiskit_aer.__version__
    except ImportError:  # pragma: no cover
        qv = av = None
    return {
        "generator": "papers/hardware-as-noisy-field/notebook_vqe_deep.py",
        "generator_sha256": sha,
        "git_head": head,
        "produced_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "qiskit": qv, "qiskit_aer": av,
        "vacuum": vacuum.__version__,
        "vacuum_sim_sha256": hashlib.sha256((Path(vs.__file__)).read_bytes()).hexdigest(),
    }


def counting_bound_layers(n_qubits):
    """Smallest L with n (L + 1) >= 2^n - 1, the real dimension of the sphere
    of real n-qubit states.  The R_Y/CX ansatz started from |0...0> prepares
    REAL amplitudes, so its image is a subset of that sphere of dimension at
    most n (L + 1); below this L the image cannot contain a generic real
    state.  It is a NECESSARY condition for covering the sphere, not for
    containing one particular state: a heuristic for our target, not a
    theorem about it."""
    n = int(n_qubits)
    return -(-(2 ** n - 1) // n) - 1


def symmetric_sector_dof(chain):
    """Real degrees of freedom of the sector the ground state lives in: even
    total Fock parity (H_d is even under x -> -x) and reflection symmetry
    (site i <-> N-1-i; the Dirichlet chain is mirror symmetric), minus one
    for normalisation.  A parameter-counting heuristic only: n_params below
    this cannot reach a generic state of the sector, n_params above it need
    not reach this one."""
    N, d = chain.N, chain.d
    idx = np.arange(chain.dim)
    digits = np.array([(idx // d ** i) % d for i in range(N)]).T
    even = digits.sum(axis=1) % 2 == 0
    refl = (digits[:, ::-1] * (d ** np.arange(N))).sum(axis=1)
    self_sym = int(np.sum(even & (refl == idx)))
    pairs = (int(np.sum(even)) - self_sym) // 2
    return self_sym + pairs - 1


# --------------------------------------------------------------------------
# worker: one job per subprocess
# --------------------------------------------------------------------------


def _fr_row(fr, N, d, chain, **extra):
    return {"N": N, "d": d, "n_qubits": chain.n_qubits, "layers": fr.layers, "n_params": fr.n_params,
            "sector_dof": symmetric_sector_dof(chain),
            "infidelity": fr.infidelity, "fidelity": fr.fidelity, "energy": fr.energy,
            "E0_exact_truncated": fr.E0_exact, "energy_error": fr.energy_error, "cov_error_vs_ED": fr.cov_error,
            "grad_norm": fr.grad_norm, "restarts": fr.restarts, "restarts_run": fr.restarts_run,
            "best_restart": fr.best_restart, "best_start": fr.best_start,
            "restart_infidelities": fr.restart_infidelities, "n_iterations": fr.n_iterations,
            "n_evaluations": fr.n_evaluations, "seed": fr.seed, "wall_s": fr.wall_s,
            "params": fr.params.tolist(), "history": fr.history, **extra}


def run_job(spec):
    t0 = time.time()
    kind = spec["kind"]
    N, d = int(spec["N"]), int(spec["d"])
    chain = vs.truncated_chain(N, 1.0, d)
    if kind == "ceiling":
        init = spec.get("init_params")
        if init is not None:
            init = {k: np.asarray(v, dtype=float) for k, v in init.items()}
        fr = vs.fidelity_ceiling(chain, int(spec["layers"]), restarts=int(spec["restarts"]), seed=int(spec["seed"]),
                                 maxiter=int(spec["maxiter"]), method=spec.get("method", "lm"),
                                 init_params=init, stop_at=spec.get("stop_at"),
                                 perturb_rounds=int(spec.get("perturb_rounds", 0)))
        row = _fr_row(fr, N, d, chain, kind="ceiling", method=spec.get("method", "lm"), maxiter=int(spec["maxiter"]),
                      stop_at=spec.get("stop_at"), perturb_rounds=int(spec.get("perturb_rounds", 0)))
        # how the BEST start ended: a budget exhaustion means the number is an
        # upper bound on an unfinished descent, not this depth's ceiling.
        bh = min(fr.history, key=lambda h: h["infidelity"])
        row.update({"best_message": bh["message"], "best_nfev": bh["nfev"],
                    "budget_exhausted": "maximum number of function evaluations" in bh["message"].lower(),
                    "stage": spec.get("stage", "scan"),
                    "continued_from": spec.get("continued_from")})
    elif kind == "layerwise":
        scan = vs.fidelity_depth_scan(chain, layers=[1], restarts=int(spec["restarts"]), seed=int(spec["seed"]),
                                      until=ACCEPT_INFIDELITY, max_layers=int(spec["max_layers"]),
                                      maxiter=int(spec["maxiter"]), method=spec.get("method", "lm"),
                                      stop_at=spec.get("stop_at"))
        row = {"N": N, "d": d, "kind": "layerwise", "restarts_per_depth": int(spec["restarts"]),
               "max_layers": int(spec["max_layers"]), "seed": int(spec["seed"]),
               "depths": [_fr_row(fr, N, d, chain, kind="layerwise") for fr in scan]}
    elif kind == "vqe":
        warm = np.asarray(spec["warm_params"], dtype=float)
        r = vs.vqe_ground_state(chain, layers=int(spec["layers"]), restarts=int(spec["restarts"]), seed=int(spec["seed"]),
                                maxiter=int(spec["maxiter"]), gtol=float(spec["gtol"]), evaluator=spec["evaluator"],
                                gradient=spec["gradient"], warm_start=warm)
        other = "numpy" if spec["evaluator"] == "aer" else "aer"
        E_other = vs.ansatz_energy(chain, r.params, r.layers, evaluator=other)
        row = {"N": N, "d": d, "n_qubits": chain.n_qubits, "layers": r.layers,
               "n_params": vs.hea_parameter_count(chain.n_qubits, r.layers), "kind": "vqe",
               "evaluator": r.evaluator, "gradient": r.gradient, "warm_start": spec.get("warm_label", "fidelity"),
               "warm_infidelity": r.warm_infidelity, "warm_energy_error": r.warm_energy_error,
               "energy": r.energy, "E0_exact_truncated": r.E0_exact, "energy_error": r.energy_error,
               "infidelity": 1.0 - r.fidelity, "cov_error_vs_ED": r.cov_error,
               "cov_error_vs_gaussian": r.cov_error_gaussian, "aer_vs_numpy": r.aer_vs_numpy,
               "cross_evaluator": other, "cross_energy": E_other, "cross_vs_energy": abs(E_other - r.energy),
               "grad_norm": r.grad_norm, "n_evaluations": r.n_evaluations, "n_iterations": r.n_iterations,
               "restarts": r.restarts, "restart_energies": [h["energy"] for h in r.history],
               "restart_grad_max": [h["grad_max"] for h in r.history], "maxiter": int(spec["maxiter"]),
               "gtol": float(spec["gtol"]), "seed": int(spec["seed"]), "runtime_s": r.wall_s,
               "params": r.params.tolist(), "history": r.history}
    elif kind == "downstream_qet":
        params = np.asarray(spec["params"], dtype=float)
        L = int(spec["layers"])
        psi_v = vs.hea_statevector(params, chain.n_qubits, L)
        E0, psi_e = vs.exact_ground_state(chain)
        ov = np.vdot(psi_e, psi_v)
        psi_v = psi_v * (ov.conj() / abs(ov))
        cases = []
        for A, B in spec["cases"]:
            th_e, EB_e = pr.qet_optimize_theta(chain, psi_e, A, B)
            led_e = pr.qet_dense(chain, psi_e, A, B, th_e)
            th_v, EB_v = pr.qet_optimize_theta(chain, psi_v, A, B)
            led_v = pr.qet_dense(chain, psi_v, A, B, th_v)
            aud = pr.qet_audits(led_v, tol=1e-12, ledger_tol=1e-12, signaling_tol=1e-12)
            keys = ("E_ground", "E_A", "E_B", "E_B_wrong", "E_B_info", "E_B_cut", "ledger_defect", "no_signaling")
            cases.append({"site_A": A, "site_B": B, "theta_ED": th_e, "theta_VQE": th_v,
                          "ED": {k: led_e[k] for k in keys}, "VQE": {k: led_v[k] for k in keys},
                          "abs_diff": {k: abs(led_v[k] - led_e[k]) for k in keys},
                          "theta_abs_diff": abs(th_v - th_e), "audit_passed_VQE": aud.passed,
                          "max_abs_diff": max(abs(led_v[k] - led_e[k]) for k in keys)})
        row = {"N": N, "d": d, "kind": "downstream_qet", "layers": L, "infidelity": 1.0 - vs.state_fidelity(psi_e, psi_v),
               "cases": cases, "max_abs_diff": max(c["max_abs_diff"] for c in cases)}
    elif kind == "downstream_harvest":
        params = np.asarray(spec["params"], dtype=float)
        L = int(spec["layers"])
        psi_v = vs.hea_statevector(params, chain.n_qubits, L)
        E0, psi_e = vs.exact_ground_state(chain)
        ov = np.vdot(psi_e, psi_v)
        psi_v = psi_v * (ov.conj() / abs(ov))
        hs = spec["spec"]
        rows = []
        for lam in spec["lams"]:
            sp = pr.harvest_spec(tuple(hs["sites"]), hs["gap"], lam, width=hs["width"], n_steps=hs["n_steps"])
            ex_e = pr.harvest_dense(chain, psi_e, sp, method="exact", exact_min_steps=128, exact_tol=1e-9)
            ex_v = pr.harvest_dense(chain, psi_v, sp, method="exact", exact_min_steps=128, exact_tol=1e-9)
            rows.append({"lam": lam, "E_N_ED": ex_e["E_N"], "E_N_VQE": ex_v["E_N"],
                         "lambda_min_ED": ex_e["lambda_min"], "lambda_min_VQE": ex_v["lambda_min"],
                         "abs_diff_E_N": abs(ex_v["E_N"] - ex_e["E_N"]),
                         "abs_diff_lambda_min": abs(ex_v["lambda_min"] - ex_e["lambda_min"]),
                         "rho_AB_max_abs_diff": float(np.max(np.abs(ex_v["rho_AB"] - ex_e["rho_AB"]))),
                         "converged_ED": ex_e["converged"], "converged_VQE": ex_v["converged"],
                         "causal_contact": sp.causal_contact()})
        row = {"N": N, "d": d, "kind": "downstream_harvest", "layers": L,
               "infidelity": 1.0 - vs.state_fidelity(psi_e, psi_v), "spec": hs, "rows": rows,
               "max_abs_diff_E_N": max(r["abs_diff_E_N"] for r in rows),
               "max_rho_AB_abs_diff": max(r["rho_AB_max_abs_diff"] for r in rows)}
    else:
        raise ValueError(f"unknown job kind {kind!r}")
    row["threads"] = os.environ.get("OMP_NUM_THREADS")
    row["time_s"] = time.time() - t0
    return row


def run_jobs(specs, workers, jobs_dir, label="s2b"):
    """Run the specs as subprocesses of this file, ``workers`` at a time, one
    BLAS/OpenMP thread each; every result is written to
    ``jobs_dir/<label>_<tag>.json`` as it lands and reused on a rerun.
    Returns the rows in the specs' order."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(specs)), key=lambda i: -specs[i].get("cost", 1.0))

    def _one(i):
        spec = specs[i]
        key = str(spec["tag"]).replace("/", "_")
        sp_path = jobs_dir / f"{label}_{key}.spec.json"
        out_path = jobs_dir / f"{label}_{key}.json"
        if out_path.exists():
            with open(out_path) as fh:
                row = json.load(fh)["row"]
            # a result cached by an older generator carries no stage label;
            # the spec that asked for it does.
            row.setdefault("stage", spec.get("stage", "scan"))
            row.setdefault("continued_from", spec.get("continued_from"))
            return i, row, "cached"
        with open(sp_path, "w") as fh:
            json.dump(spec, fh, default=_json_default)
        env = dict(os.environ)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[k] = "1"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--job", str(sp_path), "--job-out", str(out_path)]
        tj = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            return i, {"error": proc.stderr[-2000:], "tag": spec["tag"], "time_s": time.time() - tj}, "failed"
        with open(out_path) as fh:
            return i, json.load(fh)["row"], "done"

    rows = [None] * len(specs)
    with cf.ThreadPoolExecutor(max_workers=int(workers)) as ex:
        for i, row, status in ex.map(_one, order):
            rows[i] = row
            tag = specs[i]["tag"]
            if "error" in row:
                print(f"  [{label}] {tag}: FAILED {row['error'][:300]}", flush=True)
            elif "infidelity" in row:
                print(f"  [{label}] {tag}: infidelity {row['infidelity']:.3e} [{row['time_s']:.0f}s, {status}]", flush=True)
            else:
                print(f"  [{label}] {tag}: done [{row['time_s']:.0f}s, {status}]", flush=True)
    failed = [specs[i]["tag"] for i, r in enumerate(rows) if "error" in r]
    if failed:
        raise RuntimeError(f"jobs failed: {failed}")
    return rows


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv=None):
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke configuration (~1 min, temp dir)")
    ap.add_argument("--out", default=None, help="output directory (default data/; a temp dir for --quick)")
    ap.add_argument("--workers", type=int, default=3, help="parallel worker subprocesses (1 thread each)")
    ap.add_argument("--job", default=None, help="(worker) JSON job spec to evaluate")
    ap.add_argument("--job-out", default=None, help="(worker) where to write the result JSON")
    args = ap.parse_args(argv)
    if args.job:
        with open(args.job) as fh:
            spec = json.load(fh)
        row = run_job(spec)
        with open(args.job_out, "w") as fh:
            json.dump({"spec": spec, "row": row}, fh, indent=1, default=_json_default)
        return row
    quick = args.quick
    if args.out:
        DATA = Path(args.out)
    elif quick:
        DATA = Path(tempfile.mkdtemp(prefix="hardware-as-noisy-field-s2b-quick-"))
    DATA.mkdir(parents=True, exist_ok=True)
    jobs_dir = DATA / "jobs"
    T0 = time.time()
    info = build_info()
    info["quick"] = quick

    ceiling_cases = dict(CEILING_CASES)
    context_cases = dict(CONTEXT_CASES)
    layerwise = dict(LAYERWISE)
    continue_cases = dict(CEILING_CONTINUE)
    below_probe = dict(BELOW_PROBE)
    accept = ACCEPT_INFIDELITY
    vqe_kw = dict(VQE_KW)
    evaluators = VQE_EVALUATORS
    lams = HARVEST_LAMS
    if quick:
        ceiling_cases = {(3, 4): [{"layers": [3, 4, 5], "restarts": 2, "maxiter": 60}]}
        context_cases = {(2, 4): [{"layers": [2], "restarts": 2, "maxiter": 60}]}
        layerwise = {(3, 4): {"restarts": 1, "max_layers": 5, "maxiter": 60}}
        continue_cases = {(3, 4): {"depths": "all", "maxiter": {"default": 120}, "stage2": {4: 150}}}
        below_probe = {(3, 4): {"restarts": 1, "perturb_rounds": 1, "maxiter": 120}}
        accept = 1e-6  # quick only: so the sufficient-depth and probe phases are exercised
        vqe_kw = {"restarts": 0, "maxiter": 200, "gtol": 1e-12}
        evaluators = (("numpy", "adjoint"),)
        lams = (1.0,)

    # phase 1: ceilings by depth (parallel over (case, depth)) and the layerwise scans
    print("S2b phase 1: expressibility ceilings", flush=True)
    specs = []
    for (N, d), blocks in list(ceiling_cases.items()) + list(context_cases.items()):
        for c in blocks:
            for L in c["layers"]:
                specs.append({"kind": "ceiling", "tag": f"ceiling_N{N}_d{d}_L{L}", "N": N, "d": d, "layers": L,
                              "restarts": c["restarts"], "maxiter": c["maxiter"], "seed": 1000 * L, "method": "lm",
                              "stop_at": CEILING_STOP_AT, "perturb_rounds": CEILING_PERTURB_ROUNDS,
                              "cost": (2 ** (N * int(np.log2(d)))) * (L + 1) ** 2 * c["restarts"] * c["maxiter"]})
    for (N, d), c in layerwise.items():
        specs.append({"kind": "layerwise", "tag": f"layerwise_N{N}_d{d}", "N": N, "d": d, "restarts": c["restarts"],
                      "max_layers": c["max_layers"], "maxiter": c["maxiter"], "seed": 7, "method": "lm",
                      "stop_at": CEILING_STOP_AT,
                      "cost": (2 ** (N * int(np.log2(d)))) * c["max_layers"] ** 3 * (c["restarts"] + 1) * c["maxiter"]})
    rows1 = run_jobs(specs, args.workers, jobs_dir)
    ceiling_rows = [r for r in rows1 if r["kind"] == "ceiling"]
    layerwise_rows = [r for r in rows1 if r["kind"] == "layerwise"]

    # ----------------------------------------------------------------------
    # phase 1b: CONTINUATIONS.  Every scanned depth whose best point has not
    # reached CEILING_STOP_AT is re-optimised from its own optimum with a
    # larger Jacobian budget (and, for named depths, a second stage), so that
    # "this depth did not reach the acceptance" is never an artefact of
    # maxiter.  Round 1 of this unit published a sufficient depth one layer
    # too deep for exactly that reason.
    # ----------------------------------------------------------------------
    #: a depth's representative row: lowest infidelity, and on a tie the one
    #: whose search went furthest (a continuation or a probe over the raw
    #: scan), so the archived table says how the number was actually earned.
    _STAGE_RANK = {"scan": 0, "cont1": 1, "cont2": 2, "below_probe": 3}

    def _key(r):
        return (r["infidelity"], -_STAGE_RANK.get(r.get("stage", "scan"), 0))

    def _by_depth(rows, lw_rows, N, d):
        cands = [r for r in rows if r["N"] == N and r["d"] == d]
        for lw in lw_rows:
            if lw["N"] == N and lw["d"] == d:
                cands += lw["depths"]
        bd = {}
        for r in cands:
            if r["layers"] not in bd or _key(r) < _key(bd[r["layers"]]):
                bd[r["layers"]] = r
        return bd

    def _cost(N, d, L, mi, starts=1):
        return (2 ** (N * int(np.log2(d)))) * (L + 1) ** 2 * int(mi) * starts

    cont_rows = []
    for stage in (1, 2):
        cont_specs = []
        for (N, d), c in continue_cases.items():
            bd = _by_depth(ceiling_rows + cont_rows, layerwise_rows, N, d)
            mi_map = c.get("maxiter", {})
            if stage == 1:
                depths = sorted(bd) if c.get("depths") == "all" else [L for L in c.get("depths", []) if L in bd]
            else:
                depths = [L for L in sorted(c.get("stage2", {})) if L in bd]
            for L in depths:
                src = bd[L]
                if src["infidelity"] <= CEILING_STOP_AT:
                    continue
                mi = c["stage2"][L] if stage == 2 else mi_map.get(L, mi_map.get("default"))
                cont_specs.append({
                    "kind": "ceiling", "tag": f"ceiling_N{N}_d{d}_L{L}_cont" + ("" if stage == 1 else str(stage)),
                    "N": N, "d": d, "layers": L, "restarts": 0, "maxiter": int(mi), "seed": 1000 * L,
                    "method": "lm", "stop_at": CEILING_STOP_AT, "perturb_rounds": 0, "stage": f"cont{stage}",
                    "init_params": {"continue": src["params"]},
                    "continued_from": {"stage": src.get("stage", "scan"), "infidelity": src["infidelity"],
                                       "maxiter": src.get("maxiter"), "budget_exhausted": src.get("budget_exhausted")},
                    "cost": _cost(N, d, L, mi)})
        if cont_specs:
            print(f"S2b phase 1b: continuations, stage {stage} ({len(cont_specs)} depths)", flush=True)
            cont_rows = cont_rows + run_jobs(cont_specs, args.workers, jobs_dir)
    ceiling_rows = ceiling_rows + cont_rows

    # ----------------------------------------------------------------------
    # phase 1c / 2: the depth table, its monotone envelope, the SUFFICIENT
    # depth, and the hard probe one layer below it.
    #
    # Monotone envelope.  hea_embed_layers maps an optimum at depth K into any
    # depth L >= K with the state UNCHANGED (exact, pinned by a test), so
    # whatever infidelity was reached at any depth <= L is attainable at L.
    # The raw per-depth search is NOT monotone at large depth -- the LM lands
    # in local minima of an over-determined residual problem from cold starts
    # -- so the envelope, not the raw row, is the upper bound on the
    # attainable infidelity of the ansatz at that depth.
    # ----------------------------------------------------------------------
    def _summarize(N, d, rows):
        bd = _by_depth(rows, layerwise_rows, N, d)
        table = [{"layers": L, "n_params": bd[L]["n_params"], "best_infidelity": bd[L]["infidelity"],
                  "best_source": bd[L]["kind"], "best_stage": bd[L].get("stage", "scan"),
                  "best_start": bd[L]["best_start"], "maxiter": bd[L].get("maxiter"),
                  "restarts_run": bd[L].get("restarts_run"), "grad_norm": bd[L].get("grad_norm"),
                  "best_message": bd[L].get("best_message"),
                  "budget_exhausted": bd[L].get("budget_exhausted")} for L in sorted(bd)]
        run = float("inf")
        for t in table:
            run = min(run, t["best_infidelity"])
            t["envelope_infidelity"] = run
        non_monotone = [t["layers"] for t in table if t["best_infidelity"] > t["envelope_infidelity"] + 1e-15]
        suff = [t["layers"] for t in table if t["envelope_infidelity"] <= accept]
        return bd, table, (min(suff) if suff else None), non_monotone

    print("S2b phase 1c: the probe one layer below the sufficient depth", flush=True)
    probe_rows, probe_done = [], set()
    for (N, d), kn in below_probe.items():
        for _round in range(2):
            bd, _tab, L_suff, _nm = _summarize(N, d, ceiling_rows + probe_rows)
            below = [L for L in sorted(bd) if L_suff is not None and L < L_suff]
            if not below:
                break
            Lp = below[-1]
            tag = f"probe_N{N}_d{d}_L{Lp}"
            if tag in probe_done:
                break
            probe_done.add(tag)
            r = run_jobs([{"kind": "ceiling", "tag": tag, "N": N, "d": d, "layers": Lp,
                           "restarts": kn["restarts"], "maxiter": kn["maxiter"], "seed": 500000 + Lp,
                           "method": "lm", "stop_at": CEILING_STOP_AT, "stage": "below_probe",
                           "perturb_rounds": kn["perturb_rounds"], "init_params": {"incumbent": bd[Lp]["params"]},
                           "continued_from": {"stage": bd[Lp].get("stage", "scan"),
                                              "infidelity": bd[Lp]["infidelity"]},
                           "cost": _cost(N, d, Lp, kn["maxiter"], kn["restarts"] + kn["perturb_rounds"] + 1)}],
                         args.workers, jobs_dir)[0]
            probe_rows.append(r)
            if r["infidelity"] > accept:
                break
    ceiling_rows = ceiling_rows + probe_rows

    print("S2b phase 2: the sufficient depth per case", flush=True)
    minimal = {}
    for (N, d) in ceiling_cases:
        by_depth, table, L_min, non_monotone = _summarize(N, d, ceiling_rows)
        L_use = L_min if L_min is not None else min(by_depth, key=lambda L: by_depth[L]["infidelity"])
        n_q = 0
        while 2 ** n_q < d:
            n_q += 1
        n_q *= N
        L_count = counting_bound_layers(n_q)
        tab = {t["layers"]: t for t in table}
        pr = next((r for r in probe_rows if r["N"] == N and r["d"] == d), None)
        below_scanned = [L for L in sorted(tab) if L_min is not None and L < L_min]
        probe = None
        if pr is not None:
            probe = {"layers": pr["layers"], "n_params": pr["n_params"], "infidelity": pr["infidelity"],
                     "restarts": pr["restarts"], "restarts_run": pr["restarts_run"],
                     "perturb_rounds": pr.get("perturb_rounds"), "maxiter": pr.get("maxiter"),
                     "n_evaluations": pr["n_evaluations"], "grad_norm": pr["grad_norm"],
                     "best_start": pr["best_start"], "best_message": pr.get("best_message"),
                     "budget_exhausted": pr.get("budget_exhausted"), "wall_s": pr["wall_s"],
                     "reached_acceptance": pr["infidelity"] <= accept,
                     "verdict": (f"depth {pr['layers']} ({pr['n_params']} parameters) did NOT reach "
                                 f"{accept:g} at these knobs: best infidelity "
                                 f"{pr['infidelity']:.6e} from {pr['restarts_run']} starts "
                                 f"(incumbent + {pr['restarts']} seeded + iterated local search) at "
                                 f"maxiter {pr.get('maxiter')} Jacobian evaluations each, best start ended "
                                 f"'{pr.get('best_message')}'.  Evidence at a stated budget, not a proof.")}
        gaps = [L for L in range(1, L_min) if L not in tab] if L_min is not None else []
        minimal[f"N{N}_d{d}"] = {"N": N, "d": d, "accept_infidelity": accept,
                                 "sufficient_layers": L_min, "reached": L_min is not None,
                                 "is_proved_minimal": False, "layers_used_for_vqe": L_use,
                                 "n_params_at_sufficient": by_depth[L_use]["n_params"],
                                 "sector_dof": by_depth[L_use]["sector_dof"],
                                 "best_infidelity_at_sufficient": by_depth[L_use]["infidelity"],
                                 "counting_bound_layers": L_count,
                                 "counting_bound_params": n_q * (L_count + 1),
                                 "layers_below_counting_bound": (None if L_min is None else L_count - L_min),
                                 "real_state_dof": 2 ** n_q - 1,
                                 "equals_counting_bound": (L_min == L_count),
                                 "depths_scanned": sorted(by_depth),
                                 "depths_not_scanned_below": gaps,
                                 "depths_below_not_excluded": below_scanned,
                                 "below_probe": probe,
                                 "budget_exhausted_depths": [t["layers"] for t in table if t.get("budget_exhausted")],
                                 "non_monotone_depths": non_monotone,
                                 "verdict": (
                                     f"SUFFICIENT depth reaching infidelity <= {accept:g} among the "
                                     f"depths scanned {sorted(by_depth)}: L = {L_min} "
                                     f"({by_depth[L_use]['n_params']} parameters for {2 ** n_q - 1} real dof of a "
                                     f"real {n_q}-qubit state; the parameter-counting depth is {L_count}). "
                                     f"NOT proved minimal: each depth's number is the best over finitely many "
                                     f"seeded starts at these knobs -- an upper bound on the attainable "
                                     f"infidelity, never a proven supremum -- so the depths below are not "
                                     f"excluded. What IS recorded is the probe one layer below: "
                                     + (probe["verdict"] if probe else "none run at this case.")),
                                 "best_by_depth": table, "_params": by_depth[L_use]["params"],
                                 "_params_L3": by_depth[3]["params"] if 3 in by_depth else None}
        print(f"  N={N} d={d}: sufficient depth {L_min} (infidelity <= {accept:g}); "
              f"probe below: {probe['infidelity']:.3e} at L={probe['layers']}" if probe else
              f"  N={N} d={d}: sufficient depth {L_min}", flush=True)

    # phase 3: energy-VQE at the sufficient depth (both evaluators) and at the pinned depth 3
    print("S2b phase 3: energy-VQE at the sufficient depth", flush=True)
    specs = []
    for key, m in minimal.items():
        N, d = m["N"], m["d"]
        for ev, grad in evaluators:
            specs.append({"kind": "vqe", "tag": f"vqe_N{N}_d{d}_L{m['layers_used_for_vqe']}_{ev}", "N": N, "d": d,
                          "layers": m["layers_used_for_vqe"], "evaluator": ev, "gradient": grad,
                          "warm_params": m["_params"], "warm_label": "fidelity", "seed": 0, **vqe_kw,
                          "cost": 10 * (2 ** (N * int(np.log2(d)))) * (1 + (ev == "aer"))})
        if m["_params_L3"] is not None and m["layers_used_for_vqe"] != 3:
            specs.append({"kind": "vqe", "tag": f"vqe_N{N}_d{d}_L3_numpy", "N": N, "d": d, "layers": 3,
                          "evaluator": "numpy", "gradient": "adjoint", "warm_params": m["_params_L3"],
                          "warm_label": "fidelity", "seed": 0, **vqe_kw, "cost": 2 ** (N * int(np.log2(d)))})
    vqe_rows = run_jobs(specs, args.workers, jobs_dir)
    for r in vqe_rows:
        print(f"  VQE N={r['N']} d={r['d']} L={r['layers']} {r['evaluator']}/{r['gradient']}: dE={r['energy_error']:.2e} "
              f"infid={r['infidelity']:.2e} cov={r['cov_error_vs_ED']:.2e} cross={r['cross_vs_energy']:.1e} "
              f"aer_vs_numpy={r['aer_vs_numpy']} ({r['runtime_s']:.0f}s)", flush=True)

    # phase 4: downstream rows at (3, 4) on the VQE state (the aer optimum at the minimal depth)
    print("S2b phase 4: downstream comparison rows", flush=True)
    downstream = {}
    specs = []
    for key, m in minimal.items():
        N, d = m["N"], m["d"]
        if (N, d) != (3, 4):
            downstream[key] = {"N": N, "d": d, "note": "no downstream protocol row runs at this (N, d) "
                                                      "in notebook.py (s3a, s3b, s4a-s4d, s5, s6): nothing to compare"}
            continue
        pick = [r for r in vqe_rows if r["N"] == N and r["d"] == d and r["layers"] == m["layers_used_for_vqe"]]
        pick.sort(key=lambda r: (r["evaluator"] != "aer", r["energy_error"]))
        best = pick[0]
        base = {"N": N, "d": d, "layers": best["layers"], "params": best["params"],
                "state_from": f"vqe_N{N}_d{d}_L{best['layers']}_{best['evaluator']}"}
        specs.append({"kind": "downstream_qet", "tag": f"downstream_qet_N{N}_d{d}_L{best['layers']}", "cases": QET_CASES, "cost": 1, **base})
        specs.append({"kind": "downstream_harvest", "tag": f"downstream_harvest_N{N}_d{d}_L{best['layers']}", "lams": list(lams),
                      "spec": HARVEST_SPEC, "cost": 5, **base})
        downstream[key] = {"N": N, "d": d, "state_from": base["state_from"]}
    if specs:
        rows4 = run_jobs(specs, args.workers, jobs_dir)
        for r in rows4:
            key = f"N{r['N']}_d{r['d']}"
            downstream[key][r["kind"]] = r
            if r["kind"] == "downstream_qet":
                print(f"  QET (3,4) on VQE vs ED: max |diff| {r['max_abs_diff']:.2e} over {len(r['cases'])} cases", flush=True)
            else:
                print(f"  harvest (3,4) on VQE vs ED: max |dE_N| {r['max_abs_diff_E_N']:.2e}, rho_AB {r['max_rho_AB_abs_diff']:.2e}", flush=True)

    for m in minimal.values():
        m.pop("_params")
        m.pop("_params_L3")
    def _job_wall(rows):
        tot = 0.0
        for r in rows:
            tot += float(r.get("time_s") or 0.0)
            for sub in r.get("depths", []) or []:
                tot += float(sub.get("wall_s") or 0.0)
        return tot

    out = {
        "__build__": info,
        "wall_clock_s": time.time() - T0,
        "jobs_wall_clock_s": _job_wall(rows1 + cont_rows + probe_rows + vqe_rows + (rows4 if specs else [])),
        "wall_clock_note": "wall_clock_s is THIS invocation; jobs_wall_clock_s is the summed single-thread wall "
                           "of every job behind the archive, whether run now or resumed from data/jobs/",
        "acceptance": {"infidelity": accept,
                       "note": "a depth is sufficient when the best fidelity maximisation reaches this infidelity; "
                               "the energy-VQE rows report what they reached at their knobs (VQE_KW)"},
        "knobs": {"ceiling_cases": {f"N{N}_d{d}": c for (N, d), c in ceiling_cases.items()},
                  "context_cases": {f"N{N}_d{d}": c for (N, d), c in context_cases.items()},
                  "layerwise": {f"N{N}_d{d}": c for (N, d), c in layerwise.items()},
                  "vqe": vqe_kw, "evaluators": [list(e) for e in evaluators],
                  "fidelity_method": "lm", "ceiling_stop_at": CEILING_STOP_AT,
                  "ceiling_perturb_rounds": CEILING_PERTURB_ROUNDS,
                  "continue_cases": {f"N{N}_d{d}": c for (N, d), c in continue_cases.items()},
                  "below_probe": {f"N{N}_d{d}": c for (N, d), c in below_probe.items()},
                  "counting_bound": {"note": "smallest L with n (L + 1) >= 2^n - 1, the real dof of a generic "
                                             "real n-qubit state; NECESSARY for the ansatz image to cover the "
                                             "sphere of real states, a heuristic (not a theorem) for one target",
                                     "N3_d4": counting_bound_layers(6), "N4_d4": counting_bound_layers(8)},
                  "envelope": "best_by_depth carries best_infidelity (this depth's own search, after its "
                              "continuation) and envelope_infidelity = min over depths <= L (attainable at L "
                              "by exact layer embedding); sufficient_layers is read off the envelope",
                  "workers": args.workers, "threads_per_worker": 1},
        "ceiling": [{k: v for k, v in r.items() if k not in ("params", "history")} for r in ceiling_rows],
        "layerwise": [{**{k: v for k, v in lw.items() if k != "depths"},
                       "depths": [{k: v for k, v in r.items() if k not in ("params", "history")} for r in lw["depths"]]}
                      for lw in layerwise_rows],
        "sufficient_depth": minimal,
        "vqe": [{k: v for k, v in r.items() if k != "history"} for r in vqe_rows],
        "downstream": downstream,
    }
    with open(DATA / "s2b_vqe_deep.json", "w") as fh:
        json.dump(out, fh, indent=1, default=_json_default)
    print(f"  wrote {DATA / 's2b_vqe_deep.json'}")
    print(f"done in {time.time() - T0:.1f} s")
    return out


if __name__ == "__main__":
    main()
