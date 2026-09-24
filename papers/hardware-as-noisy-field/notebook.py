#!/usr/bin/env python
"""Hardware as noisy field — the survival map of the two minimal protocols.

Re-runnable end to end (needs the ``.[ibm]`` extra):

    .venv/bin/python papers/hardware-as-noisy-field/notebook.py          # full, ~5-10 min
    .venv/bin/python papers/hardware-as-noisy-field/notebook.py --quick  # smoke, ~1 min

THE CLAIM (one paragraph)
=========================
A lattice scalar-field vacuum (the Layer-1 harmonic chain) can be put on a
handful of qubits by a Fock truncation whose error is exponentially small
in the levels per site, prepared variationally to the exact truncated
ground state, and used as the medium for the two minimal vacuum protocols
of the program: quantum energy teleportation (Alice measures the sign of
the field amplitude, Bob applies a conditional displacement) and
entanglement harvesting by two ancilla Unruh-DeWitt qubits.  Under the ONE
sourced device calibration in the repository (Ikeda 2023, Table III,
ibm_cairo) the *teleportation* statements E_B_raw > 0 and E_B_ctrl > 0 are
lost — Bob's two controlled rotations heat his site by more than the 1.2 %
of E_A that the encoded field lets him extract — while the equal-depth
*information* statement E_B_info > 0 (the classical bit lowers Bob's
energy relative to the wrong bit) survives to tens of times the sourced
gate and readout errors; the 6-step harvesting circuit (52 CNOTs) is dead
at the sourced point and revives only when every error rate is reduced by
a factor recorded in ``data/summary.json``.  The map says which of the
two protocols an experimentalist could read off a free-tier device today
(the information signal, at a stated shot budget), and what number that
run would test.  Every number lives in ``data/``; none is restated here.

WHAT GOES THROUGH WHAT
======================
* encoding + ED + VQE: :mod:`vacuum.hardware.vacuum_sim` (Aer statevector
  evaluator, numpy arbiter, ED of the same truncated H, Gaussian reference
  from ``vacuum.core.ground_state_cov``);
* protocols: :mod:`vacuum.hardware.protocols` — dense arbiters
  (``qet_dense``, ``harvest_dense``) and the Aer circuit families
  (``qet_exact_aer``, ``harvest_exact_aer``: density-matrix, shot-free,
  analytic readout flips, tensored calibration-matrix mitigation);
  ``harvest_gaussian_reference`` runs ``vacuum.detectors.run_harvesting``
  (oscillator detectors, MAP-1) on the same K / sites / switching;
* noise: :mod:`vacuum.hardware.noise` — the calibration point read from
  ``vacuum.qet.ikeda_qiskit.noise_model_from_params`` (every number cited
  to Table III), the labelled multiplier sweeps (declared SYNTHETIC in
  ``experiments/ibm_qet/ikeda_2023_fig2_variant-noise-sweep.json``), the
  3-sigma survival bisection;
* audits on every reported QET run: ledger closure (locality of the
  bookkeeping), E_A >= E_B, no-signaling (``qet_audits``); every harvesting
  row carries its Trotter error against the RK4-exact evolution.

Model choices, stated once: chain N = 3, m = 1, Dirichlet, d = 2 (one
qubit per site; the d = 4 rows quantify what the truncation costs);
Alice/Bob on adjacent sites 0/1 (the distance-2 rows show the signal
falls by 10^2); harvesting detectors on sites 0/1 with gap 1.5, coupling
1.0, cos^2 switching of half-width 1.5 (window 3 > separation 1: the
detectors are in causal contact, flagged in every row), 6 Trotter steps.
Shots per circuit: 10^5 for QET (the paper's budget), 10^4 per tomography
setting for harvesting.  All energies in lattice units (hbar = 1),
negativities in nats.
"""

from __future__ import annotations

import argparse
import tempfile
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DRAFT = HERE / "draft"

from vacuum import experiments_io as xio  # noqa: E402
from vacuum.hardware import noise as nz  # noqa: E402
from vacuum.hardware import protocols as pr  # noqa: E402
from vacuum.hardware import vacuum_sim as vs  # noqa: E402
from vacuum.qet.hotta import e_a_closed, e_b_optimal  # noqa: E402

QET_SHOTS = 100000
HARV_SHOTS = 10000
N_BOOT = 200


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


def dump(name, obj):
    DATA.mkdir(exist_ok=True)
    with open(DATA / name, "w") as fh:
        json.dump(obj, fh, indent=1, default=_json_default)
    print(f"  wrote {DATA.name}/{name}")


def build_info():
    src = HERE / "notebook.py"
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=HERE, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        head = None
    import qiskit
    import qiskit_aer
    import scipy
    import vacuum
    return {
        "generator": "papers/hardware-as-noisy-field/notebook.py",
        "generator_sha256": sha,
        "git_head": head,
        "produced_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
        "vacuum": vacuum.__version__,
    }


# ---------------------------------------------------------------------------
# S1 — encoding convergence
# ---------------------------------------------------------------------------


def s1_encoding(quick):
    rows = []
    cases = [(2, 1.0, (2, 3, 4, 6, 8, 12, 16)), (3, 1.0, (2, 3, 4, 6, 8)), (3, 0.3, (2, 4, 8)),
             (3, 0.0, (2, 4, 8)), (4, 1.0, (2, 3, 4, 6))]
    if quick:
        cases = [(2, 1.0, (2, 4, 8)), (3, 1.0, (2, 4))]
    for N, m, ds in cases:
        for r in vs.encoding_convergence(N, m, ds=ds):
            rows.append({"N": N, "m": m, **r})
    dump("s1_encoding_convergence.json", rows)
    return rows


# ---------------------------------------------------------------------------
# S2 — VQE on Aer
# ---------------------------------------------------------------------------


def s2_vqe(quick):
    cases = [(2, 2, 1), (3, 2, 2), (4, 2, 3), (2, 4, 2), (3, 4, 3)]
    if not quick:
        cases.append((4, 4, 3))
    else:
        cases = [(3, 2, 2), (2, 4, 2)]
    rows = []
    results = {}
    for N, d, L in cases:
        ch = vs.truncated_chain(N, 1.0, d)
        t = time.time()
        r = vs.vqe_ground_state(ch, layers=L, restarts=3, seed=0, evaluator="aer",
                                maxiter=(200 if quick else 400))
        results[(N, d)] = (ch, r)
        rows.append({"N": N, "d": d, "n_qubits": ch.n_qubits, "layers": L,
                     "n_params": vs.hea_parameter_count(ch.n_qubits, L),
                     "energy": r.energy, "E0_exact_truncated": r.E0_exact,
                     "energy_error": r.energy_error, "infidelity": 1.0 - r.fidelity,
                     "cov_error_vs_ED": r.cov_error, "cov_error_vs_gaussian": r.cov_error_gaussian,
                     "aer_vs_numpy": r.aer_vs_numpy, "n_evaluations": r.n_evaluations,
                     "n_iterations": r.n_iterations, "restart_energies": [h["energy"] for h in r.history],
                     "runtime_s": time.time() - t})
        print(f"  VQE N={N} d={d} L={L}: dE={r.energy_error:.2e} cov={r.cov_error:.2e} "
              f"covG={r.cov_error_gaussian:.2e} ({time.time() - t:.1f}s)")
    dump("s2_vqe.json", rows)
    return rows, results


# ---------------------------------------------------------------------------
# S3 — QET
# ---------------------------------------------------------------------------


def qet_signal_fn(ch, r, theta, dense, key, mitigation, shots=QET_SHOTS):
    cmap = pr.linear_coupling_map(ch.n_qubits)

    def f(c):
        nm, d = nz.noise_model_from_calibration(c, ch.n_qubits, cmap)
        led = pr.qet_exact_aer(ch, r.params, r.layers, 0, 1, theta, noise_model=nm,
                               readout_flips=d["readout_error"], shots=shots, ideal=dense,
                               mitigation=mitigation)
        return {"signal": led[key], "sigma": led[key + "_std_error"],
                "E_A": led["E_A"], "E_B_raw": led["E_B_raw"], "E_B_ctrl": led["E_B_ctrl"],
                "E_B_info": led["E_B_info"], "E_B_cut": led["E_B_cut"],
                "no_signaling_energy": led["no_signaling_energy"], "no_signaling": led["no_signaling"]}
    return f


def s3_qet(quick, vqe):
    out = {}
    # dense ledgers, incl. the Hotta anchor and the distance dependence
    dense_rows = []
    cases = [(2, 2, 0, 1), (3, 2, 0, 1), (3, 2, 0, 2), (2, 4, 0, 1), (3, 4, 0, 1), (3, 4, 0, 2), (4, 2, 0, 1), (4, 2, 0, 3)]
    if quick:
        cases = [(2, 2, 0, 1), (3, 2, 0, 1), (3, 2, 0, 2)]
    for N, d, A, B in cases:
        ch = vs.truncated_chain(N, 1.0, d)
        E0, psi = vs.exact_ground_state(ch)
        theta, E_B = pr.qet_optimize_theta(ch, psi, A, B)
        led = pr.qet_dense(ch, psi, A, B, theta)
        audit = pr.qet_audits(led, tol=1e-12, ledger_tol=1e-12, signaling_tol=1e-12)
        row = {"N": N, "d": d, "site_A": A, "site_B": B, "theta": theta,
               **{k: led[k] for k in ("E_ground", "E_A", "E_B", "E_B_wrong", "E_B_info", "E_B_cut",
                                      "ledger_defect", "no_signaling")},
               "E_B_over_E_A": led["E_B"] / led["E_A"], "audit_passed": audit.passed}
        if N == 2 and d == 2:
            om = ch.omega[0]
            h, k = om / 2.0, 1.0 / (4.0 * om)
            row["hotta_h"], row["hotta_k"] = h, k
            row["hotta_E_A_residual"] = abs(led["E_A"] - e_a_closed(h, k))
            row["hotta_E_B_residual"] = abs(E_B - e_b_optimal(h, k))
        dense_rows.append(row)
    dump("s3a_qet_dense.json", dense_rows)
    out["dense"] = dense_rows

    # the working point: N = 3, d = 2, VQE state, Aer
    ch, r = vqe[(3, 2)]
    theta, _ = pr.qet_optimize_theta(ch, r.state, 0, 1)
    dense = pr.qet_dense(ch, r.state, 0, 1, theta)
    ideal = pr.qet_exact_aer(ch, r.params, r.layers, 0, 1, theta, ideal=dense, shots=QET_SHOTS)
    calib = nz.calibration_point()
    cmap = pr.linear_coupling_map(ch.n_qubits)
    nm, dsc = nz.noise_model_from_calibration(calib, ch.n_qubits, cmap)
    sourced = {}
    for mit in ("none", "tensored"):
        led = pr.qet_exact_aer(ch, r.params, r.layers, 0, 1, theta, noise_model=nm,
                               readout_flips=dsc["readout_error"], shots=QET_SHOTS, ideal=dense,
                               mitigation=mit)
        sourced[mit] = {k: led[k] for k in ("E_A", "E_A_std_error", "HB_ground", "HB_meas", "HB_final",
                                            "HB_wrong", "E_B_raw", "E_B_raw_std_error", "E_B_ctrl",
                                            "E_B_ctrl_std_error", "E_B_info", "E_B_info_std_error",
                                            "E_B_cut", "no_signaling_energy", "no_signaling",
                                            "depth", "cx_count", "n_circuits")}
        for key in ("E_B_raw", "E_B_ctrl", "E_B_info"):
            sourced[mit][key + "_z"] = led[key] / led[key + "_std_error"]
            sourced[mit][key + "_shots_for_3sigma"] = nz.shots_for_n_sigma(
                led[key], led[key + "_std_error"], QET_SHOTS)
    working = {
        "N": 3, "d": 2, "site_A": 0, "site_B": 1, "theta": theta, "shots_per_circuit": QET_SHOTS,
        "dense": {k: dense[k] for k in ("E_A", "E_B", "E_B_wrong", "E_B_info", "E_B_cut", "ledger_defect", "no_signaling")},
        "ideal_aer": {k: ideal[k] for k in ("E_A", "E_B_ctrl", "E_B_raw", "E_B_info", "E_B_cut",
                                            "E_B_ctrl_std_error", "E_B_info_std_error", "no_signaling",
                                            "depth", "cx_count", "n_circuits")},
        "ideal_aer_vs_dense": max(abs(ideal[k] - dense[k]) for k, _ in
                                  (("E_A", 0), ("E_B_info", 0), ("E_B_cut", 0))) ,
        "ideal_aer_E_B_vs_dense": abs(ideal["E_B_ctrl"] - dense["E_B"]),
        "ideal_z": {k: ideal[k] / ideal[k + "_std_error"] for k in ("E_B_ctrl", "E_B_info")},
        "sourced_point": sourced,
        "noise_description": {k: v for k, v in dsc.items() if k not in ("coupling_map",)},
        "audit_ideal_aer": pr.qet_audits(ideal, tol=1e-10, ledger_tol=1e-10, signaling_tol=1e-10).passed,
    }
    dump("s3b_qet_working_point.json", working)
    out["working"] = working

    # survival: three signals x four axes (tensored mitigation on), plus the declared grids
    thresholds = []
    grids = []
    axes = nz.SWEEP_AXES if not quick else ("gate", "all")
    for key in ("E_B_info", "E_B_ctrl", "E_B_raw"):
        fn = qet_signal_fn(ch, r, theta, dense, key, "tensored")
        for axis in axes:
            t = time.time()
            res = nz.survival_threshold(fn, calib, axis, n_iter=(3 if quick else 8))
            thresholds.append({"protocol": "qet", "signal": key, "mitigation": "tensored",
                               "axis": axis, "nu_star": res["nu_star"], "physical": res["physical"],
                               "alive_at_source": res["alive_at_source"], "note": res.get("note"),
                               "bracket": res.get("bracket"), "n_evaluations": len(res["history"]),
                               "history": res["history"], "runtime_s": time.time() - t})
            print(f"  QET {key} {axis}: nu*={res['nu_star']} ({res.get('note')})")
        if key == "E_B_info" and not quick:
            for axis in ("gate", "readout", "t1", "all"):
                grid, _ = nz.sweep_grid(axis)
                grids += [{"protocol": "qet", "signal": key, "mitigation": "tensored", **row}
                          for row in nz.survival_sweep(fn, calib, axis, grid)]
    # unmitigated reference for the information signal on the gate axis
    fn = qet_signal_fn(ch, r, theta, dense, "E_B_info", "none")
    res = nz.survival_threshold(fn, calib, "gate", n_iter=(3 if quick else 8))
    thresholds.append({"protocol": "qet", "signal": "E_B_info", "mitigation": "none", "axis": "gate",
                       "nu_star": res["nu_star"], "physical": res["physical"],
                       "alive_at_source": res["alive_at_source"], "note": res.get("note"),
                       "bracket": res.get("bracket"), "n_evaluations": len(res["history"]),
                       "history": res["history"]})
    dump("s3c_qet_survival.json", {"thresholds": thresholds, "grid_rows": grids})
    out["thresholds"] = thresholds
    out["grids"] = grids
    out["state"] = (ch, r, theta, dense)
    return out


# ---------------------------------------------------------------------------
# S4 — harvesting
# ---------------------------------------------------------------------------


def harvest_signal_fn(ch, r, spec, mitigation, shots=HARV_SHOTS, n_boot=N_BOOT):
    cmap = pr.harvest_coupling_map(ch, spec)

    def f(c):
        nm, d = nz.noise_model_from_calibration(c, ch.n_qubits + 2, cmap)
        out = pr.harvest_exact_aer(ch, r.params, r.layers, spec, noise_model=nm,
                                   readout_flips=d["readout_error"], shots=shots, n_boot=n_boot,
                                   mitigation=mitigation, coupling_map=cmap)
        return {"signal": -out["lambda_min"], "sigma": out["lambda_min_std_error"], "E_N": out["E_N"],
                "lambda_min": out["lambda_min"], "cx_count": out["cx_count"], "depth": out["depth"]}
    return f


def s4_harvest(quick, vqe):
    out = {}
    ch, r = vqe[(3, 2)]
    base = pr.harvest_spec((0, 1), 1.5, 1.0, width=1.5, n_steps=6)

    # Trotter convergence against the RK4-exact evolution
    exact = pr.harvest_dense(ch, r.state, base, method="exact", exact_min_steps=64, exact_tol=1e-10)
    trot = []
    for n in ((3, 4, 6, 8, 12) if quick else (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)):
        sp = pr.harvest_spec(base.sites, base.gap, base.lam, width=base.width, n_steps=n)
        tr = pr.harvest_dense(ch, r.state, sp, method="trotter")
        trot.append({"n_steps": n, "dt": sp.dt, "E_N": tr["E_N"], "lambda_min": tr["lambda_min"],
                     "trotter_error_E_N": tr["E_N"] - exact["E_N"], "p_exc": tr["p_exc"]})
    dump("s4a_harvest_trotter.json", {"exact": {"E_N": exact["E_N"], "lambda_min": exact["lambda_min"],
                                                "rk4_steps": exact["n_steps"], "converged": exact["converged"],
                                                "movement": exact["movement"], "p_exc": exact["p_exc"],
                                                "causal_contact": exact["causal_contact"]},
                                      "spec": base.__dict__, "trotter": trot})
    out["trotter"] = trot
    out["exact"] = exact

    # Gaussian limit: oscillator detectors on the exact Gaussian field vs qubit detectors on the truncated field
    grows = []
    gcases = [(2, (2, 4, 8), (0.05, 0.1, 0.2, 0.4, 1.0)), (3, (2, 4), (0.1, 0.4, 1.0))]
    if quick:
        gcases = [(2, (2, 4), (0.1, 1.0))]
    for N, ds, lams in gcases:
        chN = vs.truncated_chain(N, 1.0, 2)
        for lam in lams:
            sp = pr.harvest_spec(base.sites, base.gap, lam, width=base.width, n_steps=base.n_steps)
            t = time.time()
            g = pr.harvest_gaussian_reference(chN, sp)
            row = {"N": N, "lam": lam, "lam_osc": g["lam_osc"], "E_N_gaussian": g["E_N"],
                   "gaussian_flagged": g["flagged"], "gaussian_runtime_s": time.time() - t,
                   "E_N_qubit": {}, "relative_deviation": {}}
            for d in ds:
                chd = vs.truncated_chain(N, 1.0, d)
                E0, psi = vs.exact_ground_state(chd)
                ex = pr.harvest_dense(chd, psi, sp, method="exact", exact_min_steps=128, exact_tol=1e-9)
                row["E_N_qubit"][str(d)] = ex["E_N"]
                row["relative_deviation"][str(d)] = (ex["E_N"] - g["E_N"]) / g["E_N"] if g["E_N"] > 0 else None
            grows.append(row)
            print(f"  Gaussian N={N} lam={lam}: gauss={g['E_N']:.5f} qubit={row['E_N_qubit']}")
    dump("s4b_harvest_gaussian_limit.json", grows)
    out["gaussian"] = grows

    # working point on Aer: ideal, sourced (raw / mitigated), per n_steps
    calib = nz.calibration_point()
    ideal = pr.harvest_exact_aer(ch, r.params, r.layers, base, shots=HARV_SHOTS, n_boot=N_BOOT)
    tr6 = pr.harvest_dense(ch, r.state, base, method="trotter")
    working = {"spec": base.__dict__, "shots_per_setting": HARV_SHOTS, "n_settings": 9,
               "ideal": {k: ideal[k] for k in ("E_N", "lambda_min", "lambda_min_std_error", "z_score",
                                               "depth", "cx_count", "n_circuits")},
               "ideal_aer_vs_dense_trotter": float(np.max(np.abs(ideal["rho_AB"] - tr6["rho_AB"]))),
               "trotter_error_E_N": tr6["E_N"] - exact["E_N"],
               "causal_contact": base.causal_contact(), "sourced_point": {}}
    steps_list = (4, 6) if quick else (3, 4, 6, 8)
    for n in steps_list:
        sp = pr.harvest_spec(base.sites, base.gap, base.lam, width=base.width, n_steps=n)
        for mit in ("none", "tensored"):
            fn = harvest_signal_fn(ch, r, sp, mit)
            s = fn(calib)
            tr = pr.harvest_dense(ch, r.state, sp, method="trotter")
            working["sourced_point"][f"n{n}_{mit}"] = {**s, "z": s["signal"] / s["sigma"], "n_steps": n,
                                                       "ideal_E_N": tr["E_N"],
                                                       "shots_for_3sigma": nz.shots_for_n_sigma(s["signal"], s["sigma"], HARV_SHOTS)}
            print(f"  harvest sourced n={n} {mit}: E_N={s['E_N']:.4f} z={s['signal'] / s['sigma']:.1f} cx={s['cx_count']}")
    dump("s4c_harvest_working_point.json", working)
    out["working"] = working

    # survival thresholds
    thresholds = []
    grids = []
    axes = nz.SWEEP_AXES if not quick else ("all",)
    for n in ((6,) if quick else (4, 6)):
        sp = pr.harvest_spec(base.sites, base.gap, base.lam, width=base.width, n_steps=n)
        for mit in (("tensored",) if quick else ("none", "tensored")):
            fn = harvest_signal_fn(ch, r, sp, mit, n_boot=(60 if quick else 120))
            for axis in axes:
                t = time.time()
                res = nz.survival_threshold(fn, calib, axis, n_iter=(3 if quick else 8))
                thresholds.append({"protocol": "harvest", "signal": "-lambda_min", "n_steps": n,
                                   "mitigation": mit, "axis": axis, "nu_star": res["nu_star"],
                                   "physical": res["physical"], "alive_at_source": res["alive_at_source"],
                                   "note": res.get("note"), "bracket": res.get("bracket"),
                                   "n_evaluations": len(res["history"]), "history": res["history"],
                                   "runtime_s": time.time() - t})
                print(f"  harvest n={n} {mit} {axis}: nu*={res['nu_star']} ({res.get('note')})")
            if n == 6 and mit == "tensored" and not quick:
                for axis in ("gate", "readout", "t1", "all"):
                    grid, _ = nz.sweep_grid(axis)
                    grids += [{"protocol": "harvest", "n_steps": n, "mitigation": mit, **row}
                              for row in nz.survival_sweep(fn, calib, axis, grid)]
    dump("s4d_harvest_survival.json", {"thresholds": thresholds, "grid_rows": grids})
    out["thresholds"] = thresholds
    out["grids"] = grids
    out["base"] = base
    return out


# ---------------------------------------------------------------------------
# S5 — sampled cross-checks (Aer with the noise model, actual shots)
# ---------------------------------------------------------------------------


def s5_sampled(quick, qet, harv):
    from qiskit_aer import AerSimulator

    ch, r, theta, dense = qet["state"]
    calib = nz.calibration_point()
    rows = []
    nm, d = nz.noise_model_from_calibration(calib, ch.n_qubits, pr.linear_coupling_map(ch.n_qubits))
    exact = pr.qet_exact_aer(ch, r.params, r.layers, 0, 1, theta, noise_model=nm,
                             readout_flips=d["readout_error"], shots=QET_SHOTS, ideal=dense, mitigation="tensored")
    sm = pr.qet_sampled(ch, r.params, r.layers, 0, 1, theta, AerSimulator(noise_model=nm), shots=QET_SHOTS,
                        seed=11, ideal=dense, mitigation="tensored")
    rows.append({"protocol": "qet", "point": "sourced", "shots": QET_SHOTS,
                 "exact": {k: exact[k] for k in ("E_A", "E_B_ctrl", "E_B_info", "E_B_info_std_error", "E_B_ctrl_std_error")},
                 "sampled": {k: sm[k] for k in ("E_A", "E_B_ctrl", "E_B_info", "E_B_info_std_error", "E_B_ctrl_std_error")},
                 "pull_E_B_info": (sm["E_B_info"] - exact["E_B_info"]) / exact["E_B_info_std_error"],
                 "pull_E_B_ctrl": (sm["E_B_ctrl"] - exact["E_B_ctrl"]) / exact["E_B_ctrl_std_error"]})
    base = harv["base"]
    for nu in ((0.125,) if quick else (0.0625, 0.125)):
        c = nz.scale_calibration(calib, gate=nu, readout=nu, t1=nu)
        cmap = pr.harvest_coupling_map(ch, base)
        nmh, dh = nz.noise_model_from_calibration(c, ch.n_qubits + 2, cmap)
        ex = pr.harvest_exact_aer(ch, r.params, r.layers, base, noise_model=nmh, readout_flips=dh["readout_error"],
                                  shots=HARV_SHOTS, n_boot=N_BOOT, mitigation="tensored", coupling_map=cmap)
        sm = pr.harvest_sampled(ch, r.params, r.layers, base, AerSimulator(noise_model=nmh), shots=HARV_SHOTS,
                                seed=13, n_boot=N_BOOT, mitigation="tensored")
        rows.append({"protocol": "harvest", "point": f"all x {nu}", "shots": HARV_SHOTS,
                     "exact": {k: ex[k] for k in ("E_N", "lambda_min", "lambda_min_std_error", "z_score")},
                     "sampled": {k: sm[k] for k in ("E_N", "lambda_min", "lambda_min_std_error", "z_score")},
                     "pull_lambda_min": (sm["lambda_min"] - ex["lambda_min"]) / ex["lambda_min_std_error"]})
    dump("s5_sampled_crosschecks.json", rows)
    return rows


# ---------------------------------------------------------------------------
# S6 — what a free-tier run would need
# ---------------------------------------------------------------------------


def s6_free_tier(qet, harv):
    sweep = xio.load_experiment(nz.SWEEP_FILE)
    budget = float(sweep.quantities["open_plan_qpu_budget"].value)
    t0 = float(sweep.quantities["qpu_time_overhead"].value)
    per = float(sweep.quantities["qpu_time_per_execution"].value)
    w = qet["working"]
    src = w["sourced_point"]["tensored"]
    n_circ = src["n_circuits"] + 2          # + the two calibration circuits
    shots_info = src["E_B_info_shots_for_3sigma"]
    entries = {
        "qpu_time_formula": "t = 2 s + 0.00035 s x (circuits x shots)  [IBM 'Workload usage', transcribed in the sweep file]",
        "open_plan_budget_s": budget,
        "qet_information_signal": {
            "qubits": 3, "circuits": n_circ, "depth_native": src["depth"], "cx_count_max": src["cx_count"],
            "shots_per_circuit_for_3sigma_at_sourced_noise": shots_info,
            "shots_per_circuit_used": QET_SHOTS,
            "qpu_time_s_at_used_shots": t0 + per * n_circ * QET_SHOTS,
            "qpu_time_s_at_3sigma_shots": (t0 + per * n_circ * shots_info) if math.isfinite(shots_info) else None,
            "fits_open_plan_budget": (t0 + per * n_circ * QET_SHOTS) <= budget,
            "predicted_value_at_sourced_noise": src["E_B_info"],
            "predicted_std_error": src["E_B_info_std_error"],
            "ideal_value": w["dense"]["E_B_info"],
            "number_tested": "E_B_info = <H_B>_wrong-bit - <H_B>_right-bit > 0 at 3 sigma: the classical bit is worth "
                             "a positive amount of Bob's local energy; predicted value and error above",
        },
        "qet_teleportation_signal": {
            "predicted_E_B_ctrl_at_sourced_noise": src["E_B_ctrl"],
            "predicted_E_B_raw_at_sourced_noise": src["E_B_raw"],
            "verdict": "negative at the sourced point: not testable on this calibration (see thresholds)",
        },
        "harvesting": {
            "qubits": 5, "circuits": 9 + 2, "depth_native": harv["working"]["ideal"]["depth"],
            "cx_count": harv["working"]["ideal"]["cx_count"],
            "shots_per_setting_used": HARV_SHOTS,
            "qpu_time_s_at_used_shots": t0 + per * 11 * HARV_SHOTS,
            "predicted_E_N_at_sourced_noise": harv["working"]["sourced_point"]["n6_tensored"]["E_N"],
            "verdict": "dead at the sourced point (lambda_min > 0); revives only at the 'all' multiplier in s4d",
        },
    }
    dump("s6_free_tier.json", entries)
    return entries


# ---------------------------------------------------------------------------
# figure (pure SVG, no plotting dependency) — the survival map
# ---------------------------------------------------------------------------


def survival_map_svg(qet_grids, harv_grids, path):
    axes = ["gate", "readout", "t1", "all"]
    W, H = 900, 640
    pad_l, pad_r, pad_t, pad_b = 70, 20, 40, 50
    pw, ph = (W - pad_l - pad_r) / 2 - 20, (H - pad_t - pad_b) / 2 - 30
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif" font-size="12">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{W/2}" y="22" text-anchor="middle" font-size="15">Survival map: z = signal / sigma vs noise multiplier nu (nu = 1: Ikeda 2023 Table III, ibm_cairo)</text>']
    zmax = 40.0
    for i, axis in enumerate(axes):
        x0 = pad_l + (i % 2) * (pw + 40)
        y0 = pad_t + (i // 2) * (ph + 40)
        lines.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#333"/>')
        lines.append(f'<text x="{x0 + pw/2}" y="{y0 - 6}" text-anchor="middle">axis: {axis}</text>')
        nus = sorted({row["nu"] for row in qet_grids + harv_grids if row["axis"] == axis})
        if not nus:
            continue
        lo, hi = math.log2(min(nus)), math.log2(max(nus))

        def X(nu):
            return x0 + (math.log2(nu) - lo) / (hi - lo) * pw

        def Y(z):
            z = max(min(z, zmax), -5.0)
            return y0 + ph - (z + 5.0) / (zmax + 5.0) * ph

        lines.append(f'<line x1="{x0}" y1="{Y(3.0)}" x2="{x0 + pw}" y2="{Y(3.0)}" stroke="#999" stroke-dasharray="4,3"/>')
        lines.append(f'<text x="{x0 + pw - 4}" y="{Y(3.0) - 3}" text-anchor="end" fill="#666">z = 3</text>')
        lines.append(f'<line x1="{X(1.0)}" y1="{y0}" x2="{X(1.0)}" y2="{y0 + ph}" stroke="#c33" stroke-dasharray="2,3"/>')
        lines.append(f'<text x="{X(1.0) + 3}" y="{y0 + 12}" fill="#c33">sourced</text>')
        for nu in nus:
            lines.append(f'<text x="{X(nu)}" y="{y0 + ph + 14}" text-anchor="middle" font-size="10">{nu:g}</text>')
        for z in (0, 10, 20, 30, 40):
            lines.append(f'<text x="{x0 - 6}" y="{Y(z) + 4}" text-anchor="end" font-size="10">{z}</text>')
        for rows, color, label in ((qet_grids, "#1f77b4", "QET E_B_info (3 qubits, 1e5 shots)"),
                                   (harv_grids, "#2ca02c", "harvesting -lambda_min (5 qubits, 6 steps, 1e4 shots)")):
            pts = sorted([(row["nu"], row["z"]) for row in rows if row["axis"] == axis])
            if not pts:
                continue
            path_d = " ".join(f"{'M' if k == 0 else 'L'} {X(nu):.1f} {Y(z):.1f}" for k, (nu, z) in enumerate(pts))
            lines.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2"/>')
            for nu, z in pts:
                lines.append(f'<circle cx="{X(nu):.1f}" cy="{Y(z):.1f}" r="3" fill="{color}"/>')
            if i == 0:
                ly = y0 + 24 + (0 if color == "#1f77b4" else 16)
                lines.append(f'<line x1="{x0 + 8}" y1="{ly}" x2="{x0 + 28}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
                lines.append(f'<text x="{x0 + 32}" y="{ly + 4}">{label}</text>')
    lines.append(f'<text x="{W/2}" y="{H - 14}" text-anchor="middle" fill="#444">nu multiplies the gate errors / readout flips, divides T1 and T2, or all three ("all"); z clipped to [-5, 40]</text>')
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines))
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    global DATA, DRAFT
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke configuration (~20 s)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: data/ with the figure in draft/; a fresh temporary "
                         "directory for --quick, so a smoke run never overwrites the archive)")
    args = ap.parse_args(argv)
    quick = args.quick
    if args.out:
        DATA = Path(args.out)
        DRAFT = DATA / "draft"
    elif quick:
        DATA = Path(tempfile.mkdtemp(prefix="hardware-as-noisy-field-quick-"))
        DRAFT = DATA / "draft"
    T0 = time.time()
    info = build_info()
    info["quick"] = quick
    print("S1 encoding convergence")
    s1 = s1_encoding(quick)
    print("S2 VQE on Aer")
    s2, vqe = s2_vqe(quick)
    print("S3 QET")
    qet = s3_qet(quick, vqe)
    print("S4 harvesting")
    harv = s4_harvest(quick, vqe)
    print("S5 sampled cross-checks")
    s5 = s5_sampled(quick, qet, harv)
    print("S6 free tier")
    s6 = s6_free_tier(qet, harv)
    if not quick:
        DRAFT.mkdir(exist_ok=True)
        survival_map_svg(qet["grids"], harv["grids"], DRAFT / "survival_map.svg")
    summary = {
        "__build__": info,
        "wall_clock_s": time.time() - T0,
        "encoding": {f"N{r['N']}_m{r['m']}_d{r['d']}": {"dE": r["dE"], "cov_err": r["cov_err"]} for r in s1},
        "vqe": {f"N{r['N']}_d{r['d']}": {"energy_error": r["energy_error"], "cov_error_vs_ED": r["cov_error_vs_ED"],
                                          "cov_error_vs_gaussian": r["cov_error_vs_gaussian"], "infidelity": r["infidelity"]}
                for r in s2},
        "qet_dense": {f"N{r['N']}_d{r['d']}_A{r['site_A']}B{r['site_B']}": {"E_A": r["E_A"], "E_B": r["E_B"], "E_B_info": r["E_B_info"],
                                                                            "E_B_over_E_A": r["E_B_over_E_A"], "audit": r["audit_passed"]}
                      for r in qet["dense"]},
        "qet_working_point": {"dense": qet["working"]["dense"], "ideal_aer_E_B_vs_dense": qet["working"]["ideal_aer_E_B_vs_dense"],
                              "sourced_point": {m: {k: v for k, v in d.items() if k.startswith("E_B") or k in ("E_A", "depth", "cx_count", "n_circuits")}
                                                for m, d in qet["working"]["sourced_point"].items()}},
        "qet_thresholds": [{k: t[k] for k in ("signal", "mitigation", "axis", "nu_star", "physical", "alive_at_source", "note")}
                           for t in qet["thresholds"]],
        "harvest_working_point": {"ideal": harv["working"]["ideal"], "trotter_error_E_N": harv["working"]["trotter_error_E_N"],
                                  "exact_E_N": harv["exact"]["E_N"], "ideal_aer_vs_dense_trotter": harv["working"]["ideal_aer_vs_dense_trotter"],
                                  "causal_contact": harv["working"]["causal_contact"],
                                  "sourced_point": {k: {kk: vv for kk, vv in v.items() if kk in ("E_N", "z", "cx_count", "ideal_E_N", "shots_for_3sigma")}
                                                    for k, v in harv["working"]["sourced_point"].items()}},
        "harvest_gaussian_limit": [{k: r[k] for k in ("N", "lam", "E_N_gaussian", "E_N_qubit", "relative_deviation")} for r in harv["gaussian"]],
        "harvest_thresholds": [{k: t[k] for k in ("n_steps", "mitigation", "axis", "nu_star", "physical", "alive_at_source", "note")}
                               for t in harv["thresholds"]],
        "sampled_crosschecks": [{k: v for k, v in r.items() if k.startswith("pull") or k in ("protocol", "point")} for r in s5],
        "free_tier": s6,
    }
    dump("summary.json" if not quick else "summary_quick.json", summary)
    print(f"done in {time.time() - T0:.1f} s")
    return summary


if __name__ == "__main__":
    main()
