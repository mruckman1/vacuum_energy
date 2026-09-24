#!/usr/bin/env python
"""The differentiable vacuum (Layer 6): optimized harvesting waveforms, QET/QEI
optima, the vacuum-to-Bell-pair exchange rate, and an inverse-design showcase.

Re-runnable end to end:

    .venv/bin/python papers/differentiable-vacuum/notebook.py              # full (<= 45 min; --workers 4 default)
    .venv/bin/python papers/differentiable-vacuum/notebook.py --quick      # smoke size (~2 min)
    .venv/bin/python papers/differentiable-vacuum/notebook.py --lambda-osc X   # re-derived scenario-2 coupling

THE CLAIM (one paragraph)
=========================
At the circuit-QED operating point of papers/circuit-qed-noise-thresholds
(gap 5.7 GHz, separation 6 mm, 0.3 ns window, coupling lambda = 0.3, IR
mass 0.2, boundary buffer 6; omega_ref = 2 pi x 7.3 GHz, v = 1.2e8 m/s),
gradient-based optimization of the switching waveform lam(t) = lam_max
chi(t)[a_0 + sum_k a_k cos(k pi u/T) + b_k sin(k pi u/T)] (cos^2 envelope,
same compact window) under (i) a FIXED ENERGY BUDGET (the ledger's drive
work of the published cos^2 protocol at the same noise floor), (ii) a FIXED
NOISE FLOOR (the same field temperature and detector loss/dephasing
channels) and (iii) a FIXED SIGNALING BOUND (the communication fraction
|M_comm|/|M| of the pair term, Tjoa-Martin-Martinez split, may not exceed
the published protocol's) gives a harvested log-negativity gain factor that
this file computes and states in ``data/summary.json`` under
``headline.gain_constrained``, together with the SAME optimization run
without the signaling bound (``gain_unconstrained``), whose waveforms buy
negativity with field-mediated signaling and are therefore inadmissible
under (iii).  The pivot trigger of PLAN.md Layer 6 is applied to the
constrained, round-trip-surviving numbers: if they fall below 2x the
candidate's contribution is that it STANDARDIZES the baselines.  Every
optimized theta* is re-run through the plain numpy stack (mandatory
passivity audit, dt-halving gate, closing ledger, mp-margin rule,
precision audit, the M2.5 split) before it is reported; the JAX numbers
are never reported.  The same file records the QET angle and QEI
sampling-function optima, the DEJMPS/BBPSSW exchange rate (distilled Bell
pairs per unit switching work) of the baseline and optimized protocols,
and the inverse-design showcase.

No number is restated in this docstring; the file's sha256 rides in every
archive it writes.

THE THREE CLOSURES OF THE LAYER-6 OPEN ITEMS (sections [5b]-[5d])
==================================================================
[5b] GLOBAL SEARCH.  The archived optima were the best of two starts.  At
each archived floor ``vacuum.opt.multistart`` runs 32 starts in the k <= 5
harmonic family (published, the archived optimum embedded, cross-entropy
seeded, random at three scales), clusters the projected waveforms into
basins, polishes the top basins, and round-trips every basin representative
whose twin negativity exceeds the archived round-tripped value, with the
dt gate cap raised from 16 to 18 and the gate movement recorded per row.
The archive states per floor whether the archived optimum stands or is
beaten (and archives the new waveform with its exchange rate if so).
[5c] THE SCENARIO-2 TWIN.  A second twin at the SHIPPED dense-map base point
of papers/circuit-qed-noise-thresholds (Table I scenario 2: 7.3 GHz gap
shifted by -0.46 GHz along the envelope per Eq. (30), 0.22 ns cosine ramps,
19.2 mm, the derived oscillator coupling), every input re-read from
experiments/ at run time, optimized under the same three constraints at
the device conditions of the derived resolution floor (30 mK, measured
Gamma_1, plot-digitized Gamma_phi) where the map's verdict is NO-GO.  The
verdict is whether an admissible waveform takes E_N across the derived floor.
[5d] THE PROXY.  The in-loop signaling proxy is second order in the
coupling.  On every final waveform the exact field-state /
detector-zero-point decomposition of the pair correlation
(``vacuum.opt.multistart.nonperturbative_split``, which reduces to the
TMM21 split as lambda -> 0) is compared with the proxy; the archive carries
the one-number consistency check ``proxy_consistency.summary.max_delta_final``.

WHAT GOES THROUGH WHAT
======================
vacuum.opt.objectives.make_objective('negativity') -> the JAX twin
(vacuum.opt.gjax_detectors: initial_state, midpoint/Strang protocol_evolve,
detector block, log-negativity + dead-region surrogate, the fixed-grid
second-order communication split) -> vacuum.opt.optimize.optimize (L-BFGS-B
on the JAX gradient with the energy/signaling penalties) ->
project_energy_budget (exact equal work on the twin) -> Objective.roundtrip
(vacuum.detectors: initial_state, protocol_evolve gated at 1e-9,
harvested_log_negativity, EnergyLedger.close; vacuum.audits: precision;
vacuum.detectors.communication: pair_state_with_split on the same kernel,
shape and smearing) -> Constraints.check.  The published baseline is the
cos^2 protocol at theta0 run through the identical round trip; its E_N and
work reproduce the archived rows of papers/circuit-qed-noise-thresholds
(checked below).

Inputs and provenance: the operating point and the noise anchors are read
from the experiments/ files through vacuum.experiments_io exactly as the
noise-threshold candidate does (its synthetic-input caveats apply verbatim:
window, separation and coupling are DECLARED simulation choices; 30 mK,
8.7 MHz and 1e7 Hz are the device paper's fridge base, minimum Gamma_1 and
plot-digitized minimum Gamma_phi).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
REPO = HERE.parent.parent

PROPOSAL = "circuit_qed_harvesting/teixido-bonfill_2026_table1"
VARIANT_SWITCH = "circuit_qed_harvesting/teixido-bonfill_2026_table1_variant-switching"
DEVICE = "circuit_qed_harvesting/janzen_2023_fig6"
#: the shipped dense map's archive (READ ONLY here: cross-check rows for the scenario-2 twin)
NOISE_MAP_SUMMARY = REPO / "papers" / "circuit-qed-noise-thresholds" / "data" / "summary.json"
#: the scenario-2 twin: Table I scenario index (0-based), the map's declared IR mass and buffer, and the parameter-file
#: key of the derived oscillator coupling (read at run time; absent -> the map's identification)
SCENARIO2 = dict(scenario_index=1, field_mass=0.2, boundary_buffer=6, coupling_key="oscillator_coupling_lambda_osc_scenario2")
#: the dt-halving gate cap of the global search (the archived rows sat at 15-16 of a cap of 16)
GATE_CAP = 18

#: the noise-threshold candidate's base operating point (its ``base_spec``)
BASE = dict(gap_GHz=5.7, separation_mm=6.0, window_ns=0.3, coupling_lambda=0.3, field_mass=0.2, boundary_buffer=6)
CONV_TOL = 1.0e-9
LEDGER_TOL = 1.0e-9
#: relative tolerance of the equal-work check (the work column's gate uncertainty; see optimize_waveform)
WORK_CHECK_RTOL = 1.0e-4


def build_info() -> Dict[str, str]:
    import jax
    import scipy

    src = Path(__file__).resolve()
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip() != ""
    except Exception:  # pragma: no cover
        head, dirty = "unknown", True
    return {
        "generator": str(src), "generator_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "build": head, "tree_dirty": str(dirty), "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vacuum_version": "0.1.0", "numpy": np.__version__, "scipy": scipy.__version__, "jax": jax.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
    }


# --------------------------------------------------------------------------
# The operating point, in lattice units, from the parameter files
# --------------------------------------------------------------------------


def operating_point() -> Dict[str, Any]:
    from vacuum.experiments_io import HBAR_SI, K_B_SI, load_experiment

    tb = load_experiment(PROPOSAL)
    dev = load_experiment(DEVICE)
    omega_ref = 2.0 * math.pi * float(tb.quantities["qubit_gap_frequency"].value) * 1e9
    v = float(tb.quantities["waveguide_speed"].value)
    energy = lambda hz: 2.0 * math.pi * hz / omega_ref
    window = BASE["window_ns"] * 1e-9 * omega_ref
    gap = energy(BASE["gap_GHz"] * 1e9)
    T_mK = float(dev.quantities["fridge_base_temperature"].value)
    T30 = (K_B_SI * T_mK * 1e-3 / HBAR_SI) / omega_ref
    g1_MHz = float(dev.quantities["relaxation_rate_gamma1_min"].value)
    gphi_Hz = float(dev.quantities["pure_dephasing_rate_min"].value)
    sep = max(1, int(round(BASE["separation_mm"] * 1e-3 / v * omega_ref)))
    Bf = int(math.ceil(0.5 * window)) + BASE["boundary_buffer"]
    return dict(omega_ref=omega_ref, v=v, window=window, gap=gap, T30=T30, T30_mK=T_mK,
                kappa=energy(g1_MHz * 1e6), gamma1_MHz=g1_MHz, gamma_phi=energy(gphi_Hz), gammaphi_Hz=gphi_Hz,
                sep_sites=sep, N=sep + 2 * Bf, sites=(Bf, Bf + sep), mass=BASE["field_mass"], lam=BASE["coupling_lambda"],
                provenance={"omega_ref": f"{PROPOSAL}: qubit_gap_frequency (Eq. 30)", "v": f"{PROPOSAL}: waveguide_speed",
                            "T30": f"{DEVICE}: fridge_base_temperature", "Gamma_1": f"{DEVICE}: relaxation_rate_gamma1_min",
                            "Gamma_phi": f"{DEVICE}: pure_dephasing_rate_min (plot-digitized)",
                            "window/separation/lambda/mass/buffer": "SYNTHETIC (noise-threshold candidate base_spec)"})


def floors(op, quick):
    from vacuum.detectors import bath_occupation

    out = [("30mK", op["T30"], None)]
    if not quick:  # order matters: each floor warm-starts from the previous one's constrained solution
        out = [("30mK", op["T30"], None), ("vacuum", 0.0, None),
               ("30mK+G1+Gphi", op["T30"], dict(kappa=op["kappa"], nbar=bath_occupation(op["gap"], op["T30"]), gamma_phi=op["gamma_phi"]))]
    return out


# --------------------------------------------------------------------------
# Waveform optimization at the operating point
# --------------------------------------------------------------------------


def objective_kwargs(op, n_h, T, noise, splits, max_halvings, free=("lam", "lam_max"), shifted=False):
    """Plain-data kwargs of ``make_objective('negativity', ...)`` at an operating point (picklable for the process pool).

    ``shifted`` adds the operating point's gap schedule Omega_d(t) = Omega^0 + Delta-Omega chi(t)
    (Teixido-Bonfill Eq. (30) along the envelope, ``op['gap_shift']``) as a fixed cos^2 gap block.
    """
    kw = dict(N=int(op["N"]), mass=float(op["mass"]), bc="dirichlet", detectors=[{"site": int(op["sites"][0])}, {"site": int(op["sites"][1])}],
              gaps=(float(op["gap"]), float(op["gap"])), window=(0.5 * float(op["window"]), 0.5 * float(op["window"])), basis="fourier",
              n_coeff=1 + 2 * int(n_h), lam_max=float(op["lam"]), free=tuple(free), T=float(T), noise=None if noise is None else dict(noise),
              splits=int(splits), quantity="surrogate", conv_tol=CONV_TOL, max_halvings=int(max_halvings), ledger_tol=LEDGER_TOL)
    if shifted and float(op.get("gap_shift", 0.0)) != 0.0:
        kw.update(gap_basis="cos2", n_gap_coeff=1, gap_coeffs=[float(op["gap_shift"]) / float(op["gap"])])
    return kw


def make_neg_objective(op, n_h, T, noise, splits, max_halvings, free=("lam", "lam_max"), shifted=False):
    from vacuum.opt.objectives import make_objective

    return make_objective("negativity", **objective_kwargs(op, n_h, T, noise, splits, max_halvings, free=free, shifted=shifted))


def _audit_flags(rt):
    a = rt["audits"]
    return {k: (bool(v.passed) if hasattr(v, "passed") else None) for k, v in a.items()}


def optimize_waveform(op, floor, quick, comm_bound, warm=None, verbose=True):
    """Constrained + unconstrained optimization at one noise floor; every reported row round-tripped.

    The penalized landscape is multimodal (measured: from the published cos^2
    start the vacuum floor stalls at ~1.3x while the 30 mK solution
    transplanted to T = 0 gives ~2.7x), so the constrained problem is run
    from TWO starts — the published protocol (theta0) and ``warm`` (the best
    constrained waveform of the previously processed floor, or a seeded
    perturbation of theta0 for the first floor) — and the start with the
    lower twin loss among those whose twin communication fraction respects
    the bound is round-tripped.  The in-loop signaling bound carries a
    declared 0.1% margin so the soft penalty lands under the hard check.
    """
    import jax.numpy as jnp

    from vacuum.opt.optimize import constraints, optimize, project_energy_budget

    name, T, noise = floor
    n_h = 1 if quick else 3
    splits = 256 if quick else 512
    maxiter = 15 if quick else 80
    restarts = 0 if quick else 2
    max_halvings = 10 if quick else 16
    conv_tol = 1e-6 if quick else CONV_TOL  # quick: a declared coarse gate (recorded per row) so the smoke row converges
    margin = 1e-3
    obj = make_neg_objective(op, n_h, T, noise, splits, max_halvings)
    obj0 = make_neg_objective(op, n_h, 0.0, None, 64, 8)  # T = 0 objective for the perturbative split only
    t_start = time.time()
    q0, aux0 = obj.quantity_aux(jnp.asarray(obj.theta0))
    rows = {}
    base = obj.roundtrip(obj.theta0, communication=False, conv_tol=conv_tol)
    base_comm = obj0.roundtrip(obj0.theta0, evolve=False)
    # the ENERGY BUDGET is the published protocol's ROUND-TRIP (gated) drive
    # work; the twin carries a resolution bias, so the equal-work projection
    # targets budget * (W_twin/W_gated) of the baseline.
    W_budget = float(base["work"])
    twin_bias = float(aux0["work"]) / W_budget
    rows["baseline"] = dict(theta=obj.theta0.tolist(), E_N=base["E_N"], work=base["work"], comm_fraction=base_comm["comm_fraction"],
                            E_N_pert=base_comm["E_N_pert"], converged=base["converged"], halvings=base["halvings"],
                            dt_converged=base["dt_converged"], flagged=base["flagged"], regime=base["regime"],
                            audits=_audit_flags(base), audits_passed=base["audits_passed"], n_d=base["n_d"].tolist(),
                            E_N_jax=float(aux0["E_N"]), work_jax=float(aux0["work"]), comm_proxy=float(aux0["comm_fraction"]),
                            V_AB=base["V_AB"], min_nu_pt=base["min_nu_pt"])
    rng = np.random.default_rng(0)
    seeded = obj.theta0 + np.concatenate([[0.0], 0.1 * rng.standard_normal(len(obj.theta0) - 2), [0.0]])
    starts = {"published": obj.theta0}
    if not quick:
        starts["warm"] = np.asarray(warm, dtype=float) if warm is not None else seeded
    variants = [("constrained", constraints(energy_budget=W_budget * twin_bias, signaling_bound=comm_bound * (1.0 - margin),
                                             weight=1e4, signaling_weight=1e3), starts)]
    if not quick:
        variants.append(("unconstrained", constraints(energy_budget=W_budget * twin_bias, weight=1e4), {"published": obj.theta0}))
    for vname, C, start_set in variants:
        t0 = time.time()
        trials = []
        for sname, th0 in start_set.items():
            theta, hist = optimize(obj, C, th0, method="lbfgs", maxiter=maxiter, restarts=restarts)
            theta = project_energy_budget(obj, theta, W_budget * twin_bias, index=len(obj.theta0) - 1)
            q, aux = obj.quantity_aux(jnp.asarray(theta))
            trials.append(dict(start=sname, theta=theta, E_N_jax=float(aux["E_N"]), comm_proxy=float(aux["comm_fraction"]),
                               n_evals=len(hist), message=str(hist[-1].get("message")),
                               ok=(vname == "unconstrained") or float(aux["comm_fraction"]) <= comm_bound))
        pool = [t for t in trials if t["ok"]] or trials
        best = max(pool, key=lambda t: t["E_N_jax"])
        theta = best["theta"]
        t_opt = time.time() - t0
        # Equal work is finished on the GATED ledger: the twin's discretization
        # bias depends on the waveform (an optimized shape is not the cos^2 the
        # baseline bias was measured on), so after the round trip lam_max is
        # rescaled by sqrt(W_budget / W_gated) (W ~ lam_max^2 at leading order)
        # and the round trip repeated — at most twice — until the gated work
        # sits within 1e-6 relative of the budget.  The reported E_N always
        # belongs to the final theta.
        rt = obj.roundtrip(theta, communication=False, conv_tol=conv_tol)
        reprojections = 0
        while abs(rt["work"] / W_budget - 1.0) > 1e-6 and reprojections < 2:
            theta = np.asarray(theta, dtype=float).copy()
            theta[-1] *= math.sqrt(W_budget / rt["work"])
            rt = obj.roundtrip(theta, communication=False, conv_tol=conv_tol)
            reprojections += 1
        q, aux = obj.quantity_aux(jnp.asarray(theta))
        comm = obj0.roundtrip(theta, evolve=False)
        Cfull = constraints(energy_budget=W_budget, signaling_bound=comm_bound)
        # The energy check's tolerance is the gate uncertainty of the WORK
        # column: the dt-halving gate converges E_N and <n_d> to 1e-9, while
        # the telescoped drive work of an optimized (harmonic-rich) waveform is
        # converged only to ~1e-5 relative at the accepted level (measured:
        # re-projected rows land 1e-5 off the budget).  1e-4 relative on the
        # work is four orders below any reported gain.
        chk = Cfull.check({"work": rt["work"], "comm_fraction": comm["comm_fraction"], "audits_passed": rt["audits_passed"]},
                          work_rtol=WORK_CHECK_RTOL)
        rows[vname] = dict(theta=theta.tolist(), E_N=rt["E_N"], work=rt["work"], comm_fraction=comm["comm_fraction"],
                           E_N_pert=comm["E_N_pert"], converged=rt["converged"], halvings=rt["halvings"],
                           dt_converged=rt["dt_converged"], flagged=rt["flagged"], regime=rt["regime"],
                           audits=_audit_flags(rt), audits_passed=rt["audits_passed"], n_d=rt["n_d"].tolist(),
                           E_N_jax=float(aux["E_N"]), work_jax=float(aux["work"]), comm_proxy=float(aux["comm_fraction"]),
                           gain=rt["E_N"] / base["E_N"] if base["E_N"] > 0 else float("inf"),
                           gain_per_work=(rt["E_N"] / rt["work"]) / (base["E_N"] / base["work"]) if base["E_N"] > 0 else float("inf"),
                           check=chk, admissible=bool(chk["admissible"] and rt["converged"]), start=best["start"],
                           gated_reprojections=reprojections, work_over_budget=rt["work"] / W_budget - 1.0,
                           trials=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in t.items()} for t in trials],
                           runtime_s=t_opt, V_AB=rt["V_AB"], min_nu_pt=rt["min_nu_pt"])
        if verbose:
            print(f"  [{name}/{vname}] E_N {base['E_N']:.5e} -> {rt['E_N']:.5e} (x{rows[vname]['gain']:.3f}) at W {rt['work']:.5e} "
                  f"(budget {W_budget:.5e}); comm {base_comm['comm_fraction']:.4f} -> {comm['comm_fraction']:.4f} "
                  f"(bound {comm_bound:.4f}); converged={rt['converged']} halvings={rt['halvings']} audits={rt['audits_passed']} "
                  f"admissible={rows[vname]['admissible']} start={best['start']} ({t_opt:.0f}s opt, "
                  f"{[t['n_evals'] for t in trials]} evals)", flush=True)
    return dict(floor=name, T=T, noise=noise, W_budget=W_budget, twin_work_bias=twin_bias, comm_bound=comm_bound,
                signaling_margin=margin, n_h=n_h, splits=splits, maxiter=maxiter, restarts=restarts, max_halvings=max_halvings,
                conv_tol=conv_tol, rows=rows, runtime_s=time.time() - t_start,
                cfg={k: v for k, v in obj.cfg.items() if k not in ("detectors", "ts")})


# --------------------------------------------------------------------------
# Anchors (recomputed here so the archive carries them), QET, QEI, distillation, inverse design
# --------------------------------------------------------------------------


def section_anchors(quick) -> Dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from vacuum.core import harmonic_chain_K
    from vacuum.detectors import attach_detectors, detector_block, harvested_log_negativity, initial_state, protocol_evolve_fixed, switching
    from vacuum.opt import distill as Ds
    from vacuum.opt import gjax_detectors as GD
    from vacuum.opt.inverse_design import inverse_design
    from vacuum.core import ground_state_cov

    out: Dict[str, Any] = {}
    # forward agreement, Gate-A setup A
    N, K = 20, harmonic_chain_K(20, 0.6, bc="dirichlet")
    chi = switching("cos2", 2.0, 0.0)
    init = initial_state(K, [8, 9], (2.0, 2.0), 0.0)
    r = protocol_evolve_fixed(init.V, lambda t: attach_detectors(K, [8, 9], (2.0, 2.0), 0.24 * chi(t)), np.array([-2.0, 2.0]), splits=128)
    F = GD.profile_matrix(N, [8, 9])
    cj = lambda t: 0.24 * jnp.where(jnp.abs(t) <= 2.0, jnp.cos(0.25 * jnp.pi * t) ** 2, 0.0) * jnp.ones(2)
    o = GD.run_harvesting(K, F, lambda t: jnp.array([2.0, 2.0]), cj, np.array([-2.0, 2.0]), splits=128)
    out["forward_max_dev_V"] = float(np.max(np.abs(np.asarray(o.V) - r.V)))
    out["forward_dev_work"] = abs(float(o.work) - r.work)
    out["forward_dev_E_N"] = abs(float(o.E_N) - harvested_log_negativity(detector_block(r.V, N)).E_N)
    # gradient vs Richardson FD: coupling amplitude and gap
    def np_E(lam, gap):
        ini = initial_state(K, [8, 9], (gap, gap), 0.0)
        rr = protocol_evolve_fixed(ini.V, lambda t: attach_detectors(K, [8, 9], (gap, gap), lam * chi(t)), np.array([-2.0, 2.0]), splits=128)
        return harvested_log_negativity(detector_block(rr.V, N)).E_N
    def jx_E(th):
        return GD.run_harvesting(K, F, lambda t: jnp.array([th[1], th[1]]),
                                 lambda t: th[0] * jnp.where(jnp.abs(t) <= 2.0, jnp.cos(0.25 * jnp.pi * t) ** 2, 0.0) * jnp.ones(2),
                                 np.array([-2.0, 2.0]), splits=128).E_N
    g = np.asarray(jax.grad(jx_E)(jnp.array([0.24, 2.0])))
    fd = lambda f, h: (4 * (f(h / 2) - f(-h / 2)) / h - (f(h) - f(-h)) / (2 * h)) / 3
    r_lam = fd(lambda e: np_E(0.24 + e, 2.0), 1e-3)
    r_gap = fd(lambda e: np_E(0.24, 2.0 + e), 1e-3)
    out["grad_rel_err_lam"] = abs(g[0] - r_lam) / abs(r_lam)
    out["grad_rel_err_gap"] = abs(g[1] - r_gap) / abs(r_gap)
    # distillation vs literature
    worst = 0.0
    for Fw in (0.55, 0.7, 0.9):
        _, F2, p = Ds.bbpssw_coded_round(Ds.werner_state(Fw, "phi+"))
        Fr, pr = Ds.bbpssw_recurrence(Fw)
        worst = max(worst, abs(F2 - Fr), abs(p - pr))
    rs = np.random.RandomState(0)
    for _ in range(3):
        c = rs.dirichlet([3, 1, 1, 1])
        _, cc, p = Ds.dejmps_coded_round(Ds.bell_diagonal_state(c))
        A, B, Cc, Dd, Nn = Ds.dejmps_recurrence(*c)
        worst = max(worst, float(np.max(np.abs(np.array(cc) - [A, B, Cc, Dd]))), abs(p - Nn))
    out["distill_coded_vs_recurrence"] = worst
    # inverse design recovery
    Nd = 32 if quick else 64
    Kd = harmonic_chain_K(Nd, 1e-6, bc="dirichlet")
    Vd = ground_state_cov(Kd)
    res = inverse_design({"kind": "corr_xx", "V_xx": Vd[:Nd, :Nd]}, ansatz="banded", band=1, N=Nd, audit=False)
    out["inverse_design_recovery_residual"] = res.residual
    out["inverse_design_recovery_max_dK"] = float(np.max(np.abs(res.K - Kd)))
    out["inverse_design_recovery_N"] = Nd
    return out


def section_qet() -> List[Dict[str, Any]]:
    from vacuum.opt.objectives import make_objective
    from vacuum.opt.optimize import optimize
    from vacuum.qet import hotta

    rows = []
    for h, k in ((1.0, 0.5), (0.3, 2.5), (5.0, 0.1), (2.0, 2.0)):
        obj = make_objective("qet_energy", h=h, k=k)
        th, hist = optimize(obj, None, np.array([0.6]), method="lbfgs", maxiter=200)
        rt = obj.roundtrip(th)
        rows.append(dict(h=h, k=k, theta_star=float(th[0]), theta_closed=float(hotta.theta_optimal(h, k)),
                         abs_dev=abs(float(th[0]) - float(hotta.theta_optimal(h, k))), E_B=rt["E_B"], E_A=rt["E_A"],
                         E_B_closed=float(hotta.e_b_optimal(h, k)), audits_passed=rt["audits_passed"], n_evals=len(hist)))
    return rows


def section_pkmm() -> Dict[str, Any]:
    from scipy.optimize import brentq, minimize_scalar

    from vacuum.detectors import pkmm_negativity_estimator
    from vacuum.opt.objectives import make_objective
    from vacuum.opt.optimize import optimize

    obj = make_objective("pkmm_negativity", free=("alpha",), beta=6.0, gamma=3.0, delta=0.02)
    th, hist = optimize(obj, None, np.array([9.0]), method="nelder-mead", bounds=obj.bounds, xatol=1e-6, fatol=1e-14)
    ref = minimize_scalar(lambda a: -pkmm_negativity_estimator(a, 6.0, 3.0, 0.02), bounds=(3.0, 12.0), method="bounded", options={"xatol": 1e-8})
    obj2 = make_objective("pkmm_negativity", free=("beta",), alpha=9.1, gamma=3.0, delta=0.02)
    th2, _ = optimize(lambda t: obj2.quantity(t) ** 2, None, np.array([5.0]), method="nelder-mead", bounds=((0.05, 25.0),), xatol=1e-7, fatol=1e-30)
    root = brentq(lambda b: pkmm_negativity_estimator(9.1, b, 3.0, 0.02), 5.0, 15.0, xtol=1e-9)
    return dict(alpha_star=float(th[0]), alpha_star_bracketed=float(ref.x), N2_at_alpha_star=float(obj.quantity(th)), beta_fixed=6.0,
                death_line_beta_star=float(th2[0]), death_line_brentq=float(root), death_line_digitized=9.803,
                death_line_rel_dev=abs(float(th2[0]) / 9.803 - 1.0), n_evals=len(hist))


def section_qei(quick) -> Dict[str, Any]:
    from vacuum.inequalities.qei import gaussian_family, optimize_sampling
    from vacuum.opt.objectives import make_objective
    from vacuum.opt.optimize import optimize

    N, site = (81, 40) if quick else (161, 80)
    bounds = (2.0, 4.0) if quick else (2.5, 7.0)
    obj = make_objective("qei_ratio", N=N, site=site, family="gaussian", bounds=bounds)
    th0 = np.array([bounds[0] + 0.2])
    th, hist = optimize(obj, None, th0, method="lbfgs", bounds=obj.bounds, maxiter=100)
    rt = obj.roundtrip(th, weight_floor=0.0)
    rt0 = obj.roundtrip(th0, weight_floor=0.0)
    ref = optimize_sampling(N, site, gaussian_family(), th0, bounds=[bounds], m=0.0, bc="dirichlet")
    return dict(N=N, site=site, bounds=bounds, tau0_start=float(th0[0]), ratio_start=rt0["ratio"], tau0_star=float(th[0]),
                ratio_star=rt["ratio"], tau0_numpy=float(ref.theta[0]), ratio_numpy=float(ref.ratio),
                audits_passed=rt["audits_passed"], margin_star=rt["margin_star"], n_evals=len(hist))


def exchange_rates(op, results) -> List[Dict[str, Any]]:
    from vacuum.detectors import attach_detectors
    from vacuum.core import harmonic_chain_K
    from vacuum.opt.distill import vacuum_to_bell_rate

    K = harmonic_chain_K(op["N"], op["mass"], bc="dirichlet")
    K_tot0 = attach_detectors(K, list(op["sites"]), (op["gap"], op["gap"]), 0.0)
    rows = []
    for res in results:
        for vname, row in res["rows"].items():
            h = SimpleNamespace(V_AB=row["V_AB"], work=row["work"], E_N=row["E_N"], K_tot0=K_tot0)
            for protocol in ("DEJMPS", "BBPSSW"):
                xr = vacuum_to_bell_rate(h, protocol=protocol, target_fidelity=0.99)
                rows.append(dict(floor=res["floor"], variant=vname, protocol=protocol, rate=xr.rate, bell_pairs_per_run=xr.bell_pairs_per_run,
                                 fidelity=xr.fidelity, n_rounds=xr.n_rounds, work=xr.work, E_N=xr.E_N,
                                 phi_plus_fidelity_in=xr.bell_coefficients[0], bell_coefficients=list(xr.bell_coefficients),
                                 admissible=bool(row.get("admissible", True))))
    return rows


def section_inverse_design(quick) -> Dict[str, Any]:
    from vacuum.core import ground_state_cov, harmonic_chain_K, log_negativity
    from vacuum.opt.inverse_design import inverse_design

    N = 16
    K0 = harmonic_chain_K(N, 0.5, bc="dirichlet")
    A, B = [3, 4, 5], [7, 8, 9]
    e0 = log_negativity(ground_state_cov(K0), A, B)
    out = {"N": N, "blocks": (A, B), "E_N_free": e0, "designs": []}
    for factor in ((2.0,) if quick else (1.5, 2.0, 3.0)):
        res = inverse_design({"kind": "negativity", "pairs": [(A, B, factor * e0)]}, ansatz="banded", band=1, K_ref=K0, reg=1e-3, maxiter=500)
        V = ground_state_cov(res.K)
        profile = [log_negativity(V, [i, i + 1], [i + 3, i + 4]) for i in range(0, N - 5)]
        out["designs"].append(dict(factor=factor, target=factor * e0, achieved=res.achieved["E_N"][0], residual=res.residual,
                                   min_eigenvalue=res.min_eigenvalue, max_dK=float(np.max(np.abs(res.K - K0))),
                                   v_max_bound=res.v_max_bound, audits={k: (bool(v.passed) if hasattr(v, "passed") else None) for k, v in res.audits.items()},
                                   K_diag=np.diag(res.K).tolist(), K_offdiag=np.diagonal(res.K, 1).tolist(),
                                   negativity_profile_pairs=profile, n_evals=len(res.history)))
    out["negativity_profile_free"] = [log_negativity(ground_state_cov(K0), [i, i + 1], [i + 3, i + 4]) for i in range(0, N - 5)]
    return out


# --------------------------------------------------------------------------
# Global search (Layer 6 'Open': best of two starts), the scenario-2 twin and
# the nonperturbative check of the signaling proxy
# --------------------------------------------------------------------------


def embed_theta(theta, n_h_from, n_h_to):
    """Embed a k <= n_h_from Fourier waveform (theta = [a_0, cos.., sin.., lam_max]) in the k <= n_h_to family."""
    theta = np.asarray(theta, dtype=float)
    if n_h_to < n_h_from:
        raise ValueError("cannot embed into a smaller harmonic family")
    pad = np.zeros(n_h_to - n_h_from)
    a0, cos_, sin_, lm = theta[:1], theta[1:1 + n_h_from], theta[1 + n_h_from:1 + 2 * n_h_from], theta[-1:]
    return np.concatenate([a0, cos_, pad, sin_, pad, lm])


def floor_spec(T, noise):
    """The noise floor as a mapping the numpy round trip is CHECKED against.

    ``Constraints.check`` compares these against the channel rates and field
    temperature the gated numpy run actually used (``roundtrip['noise']``),
    so the floor is re-evaluated on the numpy side like the other two
    constraints rather than being assumed from how the objective was built.
    """
    out = {"T": float(T)}
    for key in ("kappa", "nbar", "gamma_phi"):
        out[key] = float((noise or {}).get(key, 0.0))
    return out


def search_settings(quick):
    """The global-search configuration (quick: a smoke of the same code path)."""
    if quick:
        return dict(n_h=1, splits=256, gate_cap=10, conv_tol=1e-6, n_starts=3, seeds=(0, 1), n_cem=0, cem_pop=8, cem_iter=2,
                    maxiter=6, restarts=0, n_polish=0, polish_maxiter=0, polish_restarts=0, time_limit=None, max_roundtrips=2,
                    twin_n_starts=2, twin_max_roundtrips=1)
    return dict(n_h=5, splits=512, gate_cap=GATE_CAP, conv_tol=CONV_TOL, n_starts=32, seeds=(0, 1), n_cem=2, cem_pop=32, cem_iter=6,
                maxiter=30, restarts=1, n_polish=3, polish_maxiter=80, polish_restarts=2, time_limit=720.0, max_roundtrips=6,
                twin_n_starts=16, twin_max_roundtrips=4)


def global_search_specs(op, results, comm_bound, quick):
    """One plain-data spec per archived floor: the k <= n_h family, 32 starts, the archived optimum embedded as a start."""
    st = search_settings(quick)
    specs = []
    for res in results:
        base, arch = res["rows"]["baseline"], res["rows"]["constrained"]
        extra = {"archived": embed_theta(np.asarray(arch["theta"]), int(res["n_h"]), st["n_h"]).tolist()}
        specs.append(dict(
            label=res["floor"],
            objective=objective_kwargs(op, st["n_h"], res["T"], res["noise"], st["splits"], st["gate_cap"]),
            split=objective_kwargs(op, st["n_h"], 0.0, None, 64, 8),
            constraints=dict(energy_budget=res["W_budget"] * res["twin_work_bias"], signaling_bound=comm_bound * (1.0 - res["signaling_margin"]),
                             weight=1e4, signaling_weight=1e3),
            n_starts=st["n_starts"], seeds=st["seeds"],
            options=dict(budget=res["W_budget"], comm_bound=comm_bound, baseline_E_N=base["E_N"], extra_starts=extra, noise_floor=floor_spec(res["T"], res["noise"]), n_cem=st["n_cem"],
                         cem_pop=st["cem_pop"], cem_iter=st["cem_iter"], maxiter=st["maxiter"], restarts=st["restarts"], n_polish=st["n_polish"],
                         polish_maxiter=st["polish_maxiter"], polish_restarts=st["polish_restarts"],
                         current_best=arch["E_N"] if arch["admissible"] else None, conv_tol=st["conv_tol"], max_halvings=st["gate_cap"],
                         work_rtol=WORK_CHECK_RTOL, time_limit=st["time_limit"], max_roundtrips=st["max_roundtrips"], verbose=True)))
    return specs


def _bell_rates(op, row, K_tot0):
    from vacuum.opt.distill import vacuum_to_bell_rate

    out = []
    for protocol in ("DEJMPS", "BBPSSW"):
        h = SimpleNamespace(V_AB=np.asarray(row["V_AB"]), work=row["work"], E_N=row["E_N"], K_tot0=K_tot0)
        xr = vacuum_to_bell_rate(h, protocol=protocol, target_fidelity=0.99)
        out.append(dict(protocol=protocol, rate=xr.rate, bell_pairs_per_run=xr.bell_pairs_per_run, fidelity=xr.fidelity, n_rounds=xr.n_rounds,
                        work=xr.work, E_N=xr.E_N, phi_plus_fidelity_in=xr.bell_coefficients[0]))
    return out


def postprocess_global_search(op, results, outs):
    """Per floor: the archived optimum against the global search (verdict, stats, the exchange rate of a new optimum)."""
    from vacuum.core import harmonic_chain_K
    from vacuum.detectors import attach_detectors

    K = harmonic_chain_K(op["N"], op["mass"], bc="dirichlet")
    K_tot0 = attach_detectors(K, list(op["sites"]), (op["gap"], op["gap"]), 0.0)
    gs = {}
    for res in results:
        d = outs[res["floor"]]
        arch, base = res["rows"]["constrained"], res["rows"]["baseline"]
        best = d["best"]
        beats = bool(d["beats_current_best"])
        if best is not None:
            best["exchange_rates"] = _bell_rates(op, best, K_tot0)
        top_rt = max(d["roundtrips"], key=lambda r: r["E_N"]) if d["roundtrips"] else None
        if beats:
            verdict = (f"BEATEN: an admissible k <= {d['config']['objective_cfg']['n_coeff'] // 2} waveform reaches E_N {best['E_N']:.5e} "
                       f"(archived {arch['E_N']:.5e}, +{100 * (best['E_N'] / arch['E_N'] - 1):.2f} %, above the {WORK_CHECK_RTOL:g} equal-work tolerance) "
                       f"in basin {best['basin']} ({best['start']})")
        elif best is not None:
            verdict = (f"archived optimum STANDS: the best admissible round trip {best['E_N']:.5e} is within the {WORK_CHECK_RTOL:g} equal-work "
                       f"tolerance of the archived {arch['E_N']:.5e} (+{100 * (best['E_N'] / arch['E_N'] - 1):.4f} %)")
        else:
            verdict = (f"archived optimum STANDS: no round-tripped candidate is admissible above it (top round trip "
                       f"{'none' if top_rt is None else f'{top_rt['E_N']:.5e}, admissible={top_rt['admissible']}'})")
        gs[res["floor"]] = dict(d, archived=dict(E_N=arch["E_N"], gain=arch["gain"], theta=arch["theta"], halvings=arch["halvings"],
                                                 comm_fraction=arch["comm_fraction"]),
                                baseline_E_N=base["E_N"], verdict=verdict, beats_archived=beats,
                                gain_best=None if best is None else best["gain"])
        print(f"  [{res['floor']}] {verdict}; {d['stats']['n_basins']} basins among {d['stats']['n_ok']} bound-respecting trials "
              f"({d['stats']['n_trials']} trials, {d['starts_run']} starts run, time-limited={d['time_limited']}); twin gains "
              f"best/median/spread {d['stats']['twin_gains']['best']}/{d['stats']['twin_gains']['median']}/{d['stats']['twin_gains']['spread']}; "
              f"gate max level {d['gate']['max_level_used']} movement {d['gate']['max_movement']}; {d['runtime_s']:.0f} s", flush=True)
    return gs


# ---- the scenario-2 twin ------------------------------------------------------


def scenario2_operating_point(lambda_osc=None) -> Dict[str, Any]:
    """The shipped Table-I-scenario-2 dense-map base point, from the parameter files at run time.

    Everything the noise-threshold candidate derived from a publication is
    re-derived here from the same files (gap and its Eq. (30) shift along
    the envelope, the cosine-ramps window, the 19.2 mm separation, the
    device's 30 mK / minimum Gamma_1 / plot-digitized minimum Gamma_phi, the
    derived resolution floor and its band); the IR mass and the boundary
    buffer are that candidate's declared choices (``SCENARIO2``).  The
    oscillator coupling is read from the parameter file when it carries the
    derived quantity ``SCENARIO2['coupling_key']`` (the coupling-derivation
    pass), else from the identification the map used; ``lambda_osc``
    overrides both (recorded as such), so a re-derived coupling propagates
    without touching this file.
    """
    from vacuum.experiments_io import HBAR_SI, K_B_SI, load_experiment

    tb = load_experiment(PROPOSAL)
    sw = load_experiment(VARIANT_SWITCH)
    dev = load_experiment(DEVICE)
    i = int(SCENARIO2["scenario_index"])
    q = lambda e, k: e.quantities[k].value
    f_gap = float(q(tb, "qubit_gap_frequency"))
    v = float(q(tb, "waveguide_speed"))
    omega_ref = 2.0 * math.pi * f_gap * 1e9
    energy = lambda hz: 2.0 * math.pi * hz / omega_ref
    lam_tb = float(q(tb, "table1_scenario_lambda")[i])
    gamma = float(q(tb, "table1_scenario_gamma")[i])
    coef = float(q(tb, "gap_variation_coefficient"))
    window_ns = float(q(tb, "switching_duration_cosine_ramps_by_scenario")[i])
    sep_mm = float(q(sw, "detector_separation_marginal"))
    t_d = float(q(sw, "signaling_time_marginal"))
    gap = energy(f_gap * 1e9)
    shift = energy(coef * gamma * 1e9)
    window = window_ns * 1e-9 * omega_ref
    key = SCENARIO2["coupling_key"]
    if lambda_osc is not None:
        lam, lam_err, lam_source = float(lambda_osc), None, "command-line override (--lambda-osc)"
    elif key in tb.quantities:
        qq = tb.quantities[key]
        lam, lam_err = float(qq.value), (None if qq.error is None else float(qq.error))
        lam_source = f"{PROPOSAL}:{key} ({qq.extra.get('provenance', 'derived')}; derived_from {qq.extra.get('derived_from')})"
    else:
        lam, lam_err = abs(lam_tb) * math.sqrt(2.0 * gap), None
        lam_source = "identification lam_osc = |lambda_TB| sqrt(2 Omega_lat) (the map's convention; unsourced)"
    rq = tb.quantities["negativity_resolution_nats"]
    resolution = float(rq.value)
    band = tuple(float(x) for x in rq.extra["plausible_range_nats"])
    T_mK = float(q(dev, "fridge_base_temperature"))
    T30 = (K_B_SI * T_mK * 1e-3 / HBAR_SI) / omega_ref
    g1_MHz = float(q(dev, "relaxation_rate_gamma1_min"))
    gphi_Hz = float(q(dev, "pure_dephasing_rate_min"))
    sep = max(1, int(round(sep_mm * 1e-3 / v * omega_ref)))
    Bf = int(math.ceil(0.5 * window)) + int(SCENARIO2["boundary_buffer"])
    return dict(omega_ref=omega_ref, v=v, window=window, window_ns=window_ns, gap=gap, gap_GHz=f_gap, gap_shift=shift, gap_shift_GHz=coef * gamma,
                gap_shift_gamma=gamma, T30=T30, T30_mK=T_mK, kappa=energy(g1_MHz * 1e6), gamma1_MHz=g1_MHz, gamma_phi=energy(gphi_Hz),
                gammaphi_Hz=gphi_Hz, sep_sites=sep, separation_mm=sep_mm, signaling_time_ns=t_d, N=sep + 2 * Bf, sites=(Bf, Bf + sep),
                mass=float(SCENARIO2["field_mass"]), boundary_buffer=int(SCENARIO2["boundary_buffer"]), lam=lam, lam_error=lam_err,
                lam_source=lam_source, lambda_tb=lam_tb, scenario=i + 1, resolution_derived=resolution, resolution_band=band,
                provenance={"omega_ref": f"{PROPOSAL}: qubit_gap_frequency (Eq. 30)", "v": f"{PROPOSAL}: waveguide_speed",
                            "gap_shift": f"{PROPOSAL}: gap_variation_coefficient x table1_scenario_gamma[{i}] (Eq. 30)",
                            "window": f"{PROPOSAL}: switching_duration_cosine_ramps_by_scenario[{i}] (Fig. {17 + i} caption)",
                            "separation": f"{VARIANT_SWITCH}: detector_separation_marginal (= v x signaling_time_marginal)",
                            "coupling": lam_source, "resolution": f"{PROPOSAL}: negativity_resolution_nats (+ plausible_range_nats)",
                            "T30": f"{DEVICE}: fridge_base_temperature", "Gamma_1": f"{DEVICE}: relaxation_rate_gamma1_min",
                            "Gamma_phi": f"{DEVICE}: pure_dephasing_rate_min (plot-digitized)",
                            "mass/buffer": "SYNTHETIC (the noise-threshold candidate's declared choices)"})


def _read_noise_map_rows(op2):
    """Read-only cross-check rows of the shipped dense map (its finite-size control and static-gap split at this separation)."""
    out = {"available": False, "path": str(NOISE_MAP_SUMMARY)}
    if not NOISE_MAP_SUMMARY.exists():
        return out
    try:
        d = json.loads(NOISE_MAP_SUMMARY.read_text())
    except Exception as exc:  # pragma: no cover
        out["error"] = str(exc)
        return out
    out["available"] = True
    fsc = d.get("finite_size_control", {})
    buffers = [int(b) for b in fsc.get("boundary_buffers", [])]
    if op2["boundary_buffer"] in buffers:
        j = buffers.index(op2["boundary_buffer"])
        out["finite_size_row"] = dict(E_N=fsc["E_N"][j], work=fsc["work"][j], n_field_sites=fsc["n_field_sites"][j])
    for r in d.get("communication_split_rows", []):
        if int(r.get("separation_sites", -1)) == op2["sep_sites"]:
            out["split_row"] = {k: r.get(k) for k in ("label", "E_N", "work", "comm_fraction", "E_N_pert", "M_abs", "M_vac_abs", "M_comm_abs")}
    opm = d.get("operating_point", {})
    out["map_operating_point"] = {k: opm.get(k) for k in ("gap_GHz", "gap_shift_GHz", "window_ns", "separation_mm", "separation_sites",
                                                          "coupling_lambda_osc", "window_lat")}
    return out


def scenario2_baseline(op2, quick):
    """The published protocol at the shipped base point: the device point (NO-GO anchor) and the T = 0 cross-check rows."""
    import jax.numpy as jnp

    from vacuum.detectors import bath_occupation
    from vacuum.opt.objectives import make_objective

    st = search_settings(quick)
    noise = dict(kappa=op2["kappa"], nbar=bath_occupation(op2["gap"], op2["T30"]), gamma_phi=op2["gamma_phi"])
    dev_kw = objective_kwargs(op2, st["n_h"], op2["T30"], noise, st["splits"], st["gate_cap"], shifted=True)
    static_kw = objective_kwargs(op2, st["n_h"], 0.0, None, 64, 8, shifted=False)
    obj_dev = make_objective("negativity", **dev_kw)
    obj_static = make_objective("negativity", **static_kw)
    theta0 = obj_dev.theta0
    base = obj_dev.roundtrip(theta0, communication=False, conv_tol=st["conv_tol"])
    split0 = obj_static.roundtrip(theta0, evolve=False)
    _, aux = obj_dev.quantity_aux(jnp.asarray(theta0))
    W_budget = float(base["work"])
    twin_bias = float(aux["work"]) / W_budget
    rows = {"baseline": dict(theta=theta0.tolist(), E_N=base["E_N"], work=base["work"], comm_fraction=split0["comm_fraction"],
                             E_N_pert=split0["E_N_pert"], converged=base["converged"], halvings=base["halvings"], movement=base["movement"],
                             dt_converged=base["dt_converged"], flagged=base["flagged"], regime=base["regime"], min_nu_pt=base["min_nu_pt"],
                             surrogate=base["surrogate"], audits=_audit_flags(base), audits_passed=base["audits_passed"], n_d=base["n_d"].tolist(),
                             E_N_jax=float(aux["E_N"]), work_jax=float(aux["work"]), V_AB=base["V_AB"], noise=base["noise"])}
    # cross-checks against the shipped map: T = 0 with the shift (its finite-size control row) and the static gap (its split row)
    obj_T0 = make_objective("negativity", **objective_kwargs(op2, st["n_h"], 0.0, None, st["splits"], st["gate_cap"], shifted=True))
    rt = obj_T0.roundtrip(theta0, communication=False, conv_tol=st["conv_tol"])
    rows["T0_shifted"] = dict(E_N=rt["E_N"], work=rt["work"], halvings=rt["halvings"], movement=rt["movement"], converged=rt["converged"])
    obj_st = make_objective("negativity", **objective_kwargs(op2, st["n_h"], 0.0, None, st["splits"], st["gate_cap"], shifted=False))
    rt = obj_st.roundtrip(theta0, conv_tol=st["conv_tol"])
    rows["T0_static"] = dict(E_N=rt["E_N"], work=rt["work"], comm_fraction=rt["comm_fraction"], E_N_pert=rt["E_N_pert"], halvings=rt["halvings"],
                             movement=rt["movement"], converged=rt["converged"])
    xc = _read_noise_map_rows(op2)
    if xc.get("finite_size_row"):
        xc["T0_shifted_rel_dev"] = abs(rows["T0_shifted"]["E_N"] / xc["finite_size_row"]["E_N"] - 1.0)
    if xc.get("split_row"):
        xc["T0_static_rel_dev"] = abs(rows["T0_static"]["E_N"] / xc["split_row"]["E_N"] - 1.0)
        xc["T0_static_comm_dev"] = abs(rows["T0_static"]["comm_fraction"] - xc["split_row"]["comm_fraction"])
    if xc.get("map_operating_point", {}).get("coupling_lambda_osc") is not None:
        xc["coupling_matches_map"] = abs(op2["lam"] / xc["map_operating_point"]["coupling_lambda_osc"] - 1.0) < 1e-12
    return dict(rows=rows, crosscheck=xc, W_budget=W_budget, twin_work_bias=twin_bias, comm_bound=float(split0["comm_fraction"]),
                dev_kw=dev_kw, static_kw=static_kw, noise=noise, settings=st, signaling_margin=1e-3)


def scenario2_spec(op2, pre):
    """The twin's multistart spec: shifted-gap device floor, static-gap sibling as proxy and numpy split."""
    st = pre["settings"]
    return dict(label="scenario2", objective=pre["dev_kw"], split=pre["static_kw"], proxy=pre["static_kw"],
                constraints=dict(energy_budget=pre["W_budget"] * pre["twin_work_bias"], signaling_bound=pre["comm_bound"] * (1.0 - pre["signaling_margin"]),
                                 weight=1e4, signaling_weight=1e3),
                n_starts=st["twin_n_starts"], seeds=st["seeds"],
                options=dict(budget=pre["W_budget"], comm_bound=pre["comm_bound"], baseline_E_N=pre["rows"]["baseline"]["E_N"], extra_starts={},
                             noise_floor=floor_spec(float(op2["T30"]), pre["noise"]),
                             n_cem=st["n_cem"], cem_pop=st["cem_pop"], cem_iter=st["cem_iter"], maxiter=st["maxiter"], restarts=st["restarts"],
                             n_polish=st["n_polish"], polish_maxiter=st["polish_maxiter"], polish_restarts=st["polish_restarts"], current_best=0.0,
                             conv_tol=st["conv_tol"], max_halvings=st["gate_cap"], work_rtol=WORK_CHECK_RTOL, time_limit=st["time_limit"],
                             max_roundtrips=st["twin_max_roundtrips"], verbose=True))


def _split_row(d):
    """JSON-friendly view of a nonperturbative_split dict (moduli and phases instead of complex numbers)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, complex):
            out[k + "_abs"] = abs(v)
            out[k + "_phase"] = math.atan2(v.imag, v.real)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def postprocess_twin(op2, pre, out, quick):
    """The scenario-2 verdict: does an admissible waveform take the device point across the derived floor?"""
    from vacuum.core import harmonic_chain_K
    from vacuum.opt.multistart import split_from_objective
    from vacuum.opt.objectives import make_objective

    st = pre["settings"]
    base = pre["rows"]["baseline"]
    best = out["best"]
    floor, band = op2["resolution_derived"], op2["resolution_band"]
    K = harmonic_chain_K(op2["N"], op2["mass"], bc="dirichlet")
    verdict_rows = {}
    if best is not None:
        E_after = best["E_N"]
        go = bool(E_after > floor)
        verdict = (f"{'GO' if go else 'NO-GO'} at the derived floor {floor:g} nats: the published waveform gives E_N = {base['E_N']:.3e} "
                   f"(NO-GO, nu~_min {base['min_nu_pt']:.6f}); the admissible optimized waveform gives E_N = {E_after:.4e} at equal work "
                   f"{best['work']:.4e} with static-gap communication fraction {best['comm_fraction']:.4f} (bound {pre['comm_bound']:.4f}); "
                   f"band-low {'GO' if E_after > band[0] else 'NO-GO'}, band-high {'GO' if E_after > band[1] else 'NO-GO'}")
        obj_dev = make_objective("negativity", **pre["dev_kw"])
        obj_st = make_objective("negativity", **pre["static_kw"])
        level = int(best["halvings"])
        theta = np.asarray(best["theta"])
        verdict_rows["np_split_device"] = _split_row(split_from_objective(obj_dev, theta, K, splits=2 ** level))
        verdict_rows["np_split_static_T0"] = _split_row(split_from_objective(obj_st, theta, K, splits=2 ** level))
        verdict_rows["np_split_baseline_device"] = _split_row(split_from_objective(obj_dev, obj_dev.theta0, K, splits=2 ** max(level, int(base["halvings"]))))
        verdict_rows["np_split_baseline_static_T0"] = _split_row(split_from_objective(obj_st, obj_st.theta0, K, splits=2 ** max(level, int(base["halvings"]))))
    else:
        go = False
        E_after = None
        top = max(out["roundtrips"], key=lambda r: r["E_N"]) if out["roundtrips"] else None
        if top is not None and top["admissible"]:
            verdict = (f"NO-GO: every admissible round-tripped candidate stays dead (top round trip E_N {top['E_N']:.3e}, nu~_min {top['min_nu_pt']:.6f}, "
                       f"work {top['work']:.4e}, static-gap communication fraction {top['comm_fraction']:.4f} <= bound {pre['comm_bound']:.4f}); "
                       f"the device point stays at E_N = {base['E_N']:.3e} (nu~_min {base['min_nu_pt']:.6f})")
        else:
            verdict = (f"NO-GO: no round-tripped candidate is admissible (top round trip "
                       f"{'none' if top is None else f'E_N {top['E_N']:.3e}, check {top['check']}'}); the device point stays at E_N = {base['E_N']:.3e}")
    print(f"  [scenario2] {verdict}", flush=True)
    return dict(operating_point=op2, base_point_source="papers/circuit-qed-noise-thresholds (Table I scenario 2 dense map), inputs re-read from experiments/",
                settings=st, rows=pre["rows"], crosscheck=pre["crosscheck"], W_budget=pre["W_budget"], twin_work_bias=pre["twin_work_bias"],
                comm_bound=pre["comm_bound"], signaling_margin=pre["signaling_margin"],
                signaling_note="the perturbative split has static gaps only: the in-loop proxy and the numpy check are taken on the static-gap, "
                               "T = 0 sibling of the same waveform (baseline and candidate alike); the shifted-gap device rows carry the "
                               "nonperturbative decomposition instead",
                search=out, best=best, E_N_before=base["E_N"], E_N_after=E_after, go=go, resolution_derived=floor, resolution_band=list(band),
                verdict=verdict, **verdict_rows)


# ---- the signaling proxy against the nonperturbative decomposition ------------


def _spacelike_op(op):
    """The same operating point with the detectors moved to spacelike separation (separation > full window)."""
    sep = int(math.ceil(op["window"])) + 1
    Bf = int(math.ceil(0.5 * op["window"])) + BASE["boundary_buffer"]
    return dict(op, sep_sites=sep, N=sep + 2 * Bf, sites=(Bf, Bf + sep))


def section_proxy_consistency(op, results, gs, quick):
    """f_np (nonperturbative decomposition) against the second-order proxy on the final waveforms of every floor."""
    from vacuum.core import harmonic_chain_K
    from vacuum.opt.multistart import split_from_objective

    K = harmonic_chain_K(op["N"], op["mass"], bc="dirichlet")
    rows = []
    extra = {"small_lambda": [], "spacelike": []}
    for res in results:
        floor = res["floor"]
        g = gs.get(floor)
        base = res["rows"]["baseline"]
        if g is not None and g["beats_archived"]:
            final, source = g["best"], "global"
        else:
            final, source = res["rows"]["constrained"], "archived"
        theta = np.asarray(final["theta"], dtype=float)
        n_h = (theta.size - 2) // 2
        level = int(final["halvings"])
        obj_floor = make_neg_objective(op, n_h, res["T"], res["noise"], 64, 8)
        obj_T0 = make_neg_objective(op, n_h, 0.0, None, 64, 8)
        theta_b = embed_theta(np.asarray(base["theta"]), int(res["n_h"]), n_h)
        per = {}
        for name, th, proxy in (("baseline", theta_b, base["comm_fraction"]), ("final", theta, final["comm_fraction"])):
            d_T0 = split_from_objective(obj_T0, th, K, splits=2 ** level)
            d_fl = d_T0 if (res["T"] == 0.0 and res["noise"] is None) else split_from_objective(obj_floor, th, K, splits=2 ** level)
            per[name] = dict(f_proxy=proxy, f_np_T0=d_T0["f_np"], f_field_T0=d_T0["f_field"], f_np_floor=d_fl["f_np"], f_np_det_floor=d_fl["f_np_det"],
                             f_channel_floor=(abs(d_fl["M_signal_channel"]) / abs(d_fl["M_full"])) if abs(d_fl["M_full"]) > 0 else 0.0,
                             M_full_abs_T0=abs(d_T0["M_full"]), M_field_abs_T0=abs(d_T0["M_field"]), M_signal_abs_T0=abs(d_T0["M_signal"]),
                             E_N_np_floor=d_fl["E_N"], additivity=max(d_T0["additivity"], d_fl["additivity"]),
                             delta_T0=abs(d_T0["f_np"] - proxy), splits=2 ** level)
        rows.append(dict(floor=floor, source=source, E_N_final=final["E_N"], E_N_baseline=base["E_N"], baseline=per["baseline"], final=per["final"],
                         np_bound_holds=bool(per["final"]["f_np_floor"] <= per["baseline"]["f_np_floor"]),
                         proxy_bound_holds=bool(final["comm_fraction"] <= base["comm_fraction"]),
                         proxy_ordering_matches=bool((per["final"]["f_np_T0"] <= per["baseline"]["f_np_T0"]) == (final["comm_fraction"] <= base["comm_fraction"]))))
        print(f"  [{floor}] {source} waveform: proxy {final['comm_fraction']:.4f} vs f_np(T=0) {per['final']['f_np_T0']:.4f} "
              f"(baseline {base['comm_fraction']:.4f} vs {per['baseline']['f_np_T0']:.4f}); at the floor f_np {per['final']['f_np_floor']:.4f} "
              f"vs baseline {per['baseline']['f_np_floor']:.4f}; nonperturbative bound holds={rows[-1]['np_bound_holds']}", flush=True)
        if quick:
            continue
        if res["T"] == 0.0 and res["noise"] is None:  # the lambda -> 0 limit of the decomposition on the final shape (T = 0)
            for scale in (0.3, 0.1, 0.03):
                th = theta.copy()
                th[-1] *= scale
                d = split_from_objective(obj_T0, th, K, splits=2 ** level)
                extra["small_lambda"].append(dict(floor=floor, lam_scale=scale, lam_max=float(th[-1]), f_np=d["f_np"], f_proxy=final["comm_fraction"],
                                                  delta=abs(d["f_np"] - final["comm_fraction"]), f_field=d["f_field"], M_full_abs=abs(d["M_full"])))
        ops = _spacelike_op(op)
        obj_s = make_neg_objective(ops, n_h, 0.0, None, 64, 8)
        Ks = harmonic_chain_K(ops["N"], ops["mass"], bc="dirichlet")
        d = split_from_objective(obj_s, theta, Ks, splits=2 ** level)
        rt_s = obj_s.roundtrip(theta, evolve=False)
        extra["spacelike"].append(dict(floor=floor, sep_sites=ops["sep_sites"], N=ops["N"], window=op["window"], f_np=d["f_np"], E_N_np=d["E_N"],
                                       M_full_abs=abs(d["M_full"]), M_signal_abs=abs(d["M_signal"]), comm_fraction_pert=rt_s.get("comm_fraction"),
                                       perturbative_error=rt_s.get("perturbative_error")))
    deltas = {r["floor"]: r["final"]["delta_T0"] for r in rows}
    summary = dict(max_delta_final=max(deltas.values()), delta_final=deltas, delta_baseline={r["floor"]: r["baseline"]["delta_T0"] for r in rows},
                   np_bound_holds_all=all(r["np_bound_holds"] for r in rows), proxy_ordering_matches_all=all(r["proxy_ordering_matches"] for r in rows),
                   definition="delta = |f_np(T=0, same waveform, same gate level) - |M_comm|/|M| (second-order proxy)|; f_np from the exact "
                              "field-state / detector-zero-point decomposition of <a_A a_B> (vacuum.opt.multistart.nonperturbative_split)")
    return dict(rows=rows, summary=summary, **extra)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if k != "V_AB"}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="smoke-size run (one floor, one harmonic, coarse gate)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: data/; a fresh temporary directory for "
                         "--quick, so a smoke run never overwrites the archive)")
    ap.add_argument("--workers", type=int, default=4, help="parallel processes for the global search and the twin (quick: 1)")
    ap.add_argument("--lambda-osc", type=float, default=None, help="override the scenario-2 oscillator coupling (recorded as an override)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out) if args.out else (
        Path(tempfile.mkdtemp(prefix="differentiable-vacuum-quick-")) if args.quick else DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    quick = bool(args.quick)
    info = build_info()
    print(f"build {info['build']} (dirty={info['tree_dirty']}), quick={quick}", flush=True)

    op = operating_point()
    print(f"operating point: N={op['N']} sites={op['sites']} window={op['window']:.4f} gap={op['gap']:.5f} T30={op['T30']:.5f} "
          f"kappa={op['kappa']:.3e} gamma_phi={op['gamma_phi']:.3e}", flush=True)

    print("[1] anchors", flush=True)
    anchors = section_anchors(quick)
    print("    " + json.dumps(_jsonable(anchors)), flush=True)
    print("[2] PKMM re-discovery", flush=True)
    pk = section_pkmm()
    print("    " + json.dumps(_jsonable(pk)), flush=True)
    print("[3] QET angle", flush=True)
    qet = section_qet()
    print("    worst |theta* - closed| = " + f"{max(r['abs_dev'] for r in qet):.2e}", flush=True)
    print("[4] QEI ascent", flush=True)
    qei = section_qei(quick)
    print("    " + json.dumps(_jsonable(qei)), flush=True)

    print("[5] waveform optimization at the circuit-QED operating point", flush=True)
    # the signaling bound: the published protocol's own communication fraction (T = 0 split)
    obj_probe = make_neg_objective(op, 1, 0.0, None, 64, 8)
    comm_bound = obj_probe.roundtrip(obj_probe.theta0, evolve=False)["comm_fraction"]
    print(f"    signaling bound = baseline comm fraction {comm_bound:.6f}", flush=True)
    results = []
    warm = None
    for fl in floors(op, quick):
        res = optimize_waveform(op, fl, quick, comm_bound, warm=warm)
        results.append(res)
        rc = res["rows"]["constrained"]
        if rc["converged"] and rc["audits_passed"]:  # warm-start the next floor from a physically sound row
            warm = np.asarray(rc["theta"])

    print("[5b] global search at every archived floor + [5c] the scenario-2 twin", flush=True)
    from vacuum.opt.multistart import run_specs

    op2 = scenario2_operating_point(args.lambda_osc)
    print(f"    scenario-2 base point: N={op2['N']} sites={op2['sites']} window={op2['window']:.4f} gap={op2['gap']:.5f} shift={op2['gap_shift']:+.5f} "
          f"lam_osc={op2['lam']:.5f} ({op2['lam_source']}) floor={op2['resolution_derived']:g} nats", flush=True)
    twin_pre = scenario2_baseline(op2, quick)
    xc = twin_pre["crosscheck"]
    print(f"    twin baseline: device E_N={twin_pre['rows']['baseline']['E_N']:.3e} (nu~ {twin_pre['rows']['baseline']['min_nu_pt']:.6f}), "
          f"T=0 shifted E_N={twin_pre['rows']['T0_shifted']['E_N']:.6e} (map rel dev {xc.get('T0_shifted_rel_dev')}), "
          f"T=0 static E_N={twin_pre['rows']['T0_static']['E_N']:.6e} comm={twin_pre['rows']['T0_static']['comm_fraction']:.5f} "
          f"(map rel dev {xc.get('T0_static_rel_dev')}, comm dev {xc.get('T0_static_comm_dev')})", flush=True)
    specs = global_search_specs(op, results, comm_bound, quick) + [scenario2_spec(op2, twin_pre)]
    t_gs = time.time()
    outs = dict(run_specs(specs, workers=1 if quick else max(1, int(args.workers))))
    gs_wall = time.time() - t_gs
    print(f"    global search + twin wall clock {gs_wall:.0f} s ({'sequential' if quick else f'{args.workers} workers'})", flush=True)
    gs = postprocess_global_search(op, results, outs)
    twin = postprocess_twin(op2, twin_pre, outs["scenario2"], quick)

    print("[5d] the signaling proxy against the nonperturbative decomposition", flush=True)
    proxy = section_proxy_consistency(op, results, gs, quick)
    print(f"    max |f_np - proxy| on the final waveforms: {proxy['summary']['max_delta_final']:.4f}; nonperturbative bound holds on every floor: "
          f"{proxy['summary']['np_bound_holds_all']}", flush=True)

    print("[6] exchange rates", flush=True)
    xr = exchange_rates(op, results)
    for r in xr:
        print(f"    {r['floor']:>14s} {r['variant']:>13s} {r['protocol']}: {r['rate']:.3e} pairs/work ({r['bell_pairs_per_run']:.3e}/run, "
              f"{r['n_rounds']} rounds, F={r['fidelity']:.4f})", flush=True)
    print("[7] inverse design", flush=True)
    inv = section_inverse_design(quick)
    for d in inv["designs"]:
        print(f"    x{d['factor']}: E_N {inv['E_N_free']:.5f} -> {d['achieved']:.5f} (target {d['target']:.5f}), residual {d['residual']:.1e}, "
              f"audits {d['audits']}", flush=True)

    # ---- headline and pivot verdict ------------------------------------
    cons = [(res["floor"], res["rows"]["constrained"]) for res in results]
    adm = [(f, r) for f, r in cons if r["admissible"]]
    best_c = max(adm, key=lambda fr: fr[1]["gain"]) if adm else None
    unc = [(res["floor"], res["rows"]["unconstrained"]) for res in results if "unconstrained" in res["rows"]]
    best_u = max(unc, key=lambda fr: (fr[1]["gain"] if math.isfinite(fr[1]["gain"]) else 1e9)) if unc else None
    headline = {
        "signaling_bound": comm_bound,
        "gain_constrained": None if best_c is None else best_c[1]["gain"],
        "gain_constrained_floor": None if best_c is None else best_c[0],
        "gain_per_work_constrained": None if best_c is None else best_c[1]["gain_per_work"],
        "constrained_all_floors": {f: dict(gain=r["gain"], E_N=r["E_N"], E_N_base=res["rows"]["baseline"]["E_N"], work=r["work"],
                                           comm=r["comm_fraction"], comm_base=res["rows"]["baseline"]["comm_fraction"],
                                           admissible=r["admissible"], converged=r["converged"], check=r["check"])
                                   for res in results for f, r in [(res["floor"], res["rows"]["constrained"])]},
        "gain_unconstrained": None if best_u is None else best_u[1]["gain"],
        "gain_unconstrained_floor": None if best_u is None else best_u[0],
        "unconstrained_all_floors": {res["floor"]: dict(gain=res["rows"]["unconstrained"]["gain"], E_N=res["rows"]["unconstrained"]["E_N"],
                                                       comm=res["rows"]["unconstrained"]["comm_fraction"], admissible=res["rows"]["unconstrained"]["admissible"],
                                                       check=res["rows"]["unconstrained"]["check"]) for res in results if "unconstrained" in res["rows"]},
    }
    g = headline["gain_constrained"]
    if g is None:
        verdict = "NO ADMISSIBLE CONSTRAINED CANDIDATE (round trip or checks failed): the result standardizes the baselines"
    elif g < 2.0:
        verdict = (f"PIVOT TRIGGER APPLIES: the largest admissible gain under fixed energy budget, noise floor and signaling bound is "
                   f"{g:.3f}x (< 2x). The result STANDARDIZES the published baseline; effort shifts to inverse design.")
    else:
        verdict = f"gain {g:.3f}x at floor {headline['gain_constrained_floor']} under fixed energy budget, noise floor and signaling bound (>= 2x)."
    headline["pivot_verdict"] = verdict
    headline["global_search"] = {
        f: dict(archived_E_N=g["archived"]["E_N"], archived_gain=g["archived"]["gain"], best_E_N=None if g["best"] is None else g["best"]["E_N"],
                best_gain=g["gain_best"], beats_archived=g["beats_archived"], n_basins=g["stats"]["n_basins"], n_basins_all=g["stats"]["n_basins_all"],
                n_trials=g["stats"]["n_trials"], starts_run=g["starts_run"], time_limited=g["time_limited"], twin_gains=g["stats"]["twin_gains"],
                roundtrip_gains=g["stats"]["roundtrip_gains"], n_roundtrips=g["stats"]["n_roundtrips"], n_admissible=g["stats"]["n_admissible"],
                gate=g["gate"], verdict=g["verdict"]) for f, g in gs.items()}
    headline["scenario2_twin"] = dict(E_N_before=twin["E_N_before"], E_N_after=twin["E_N_after"], go=twin["go"], resolution_derived=twin["resolution_derived"],
                                      resolution_band=twin["resolution_band"], comm_bound=twin["comm_bound"],
                                      comm_after=None if twin["best"] is None else twin["best"]["comm_fraction"], lam_osc=op2["lam"], lam_source=op2["lam_source"],
                                      verdict=twin["verdict"])
    headline["proxy_consistency"] = proxy["summary"]
    print("\nHEADLINE: " + verdict, flush=True)
    for f, g in gs.items():
        print(f"GLOBAL SEARCH [{f}]: {g['verdict']}", flush=True)
    print("SCENARIO-2 TWIN: " + twin["verdict"], flush=True)
    print(f"PROXY: max |f_np - proxy| = {proxy['summary']['max_delta_final']:.4f} on the final waveforms", flush=True)
    if best_u is not None:
        print(f"(unconstrained/communication-assisted: {headline['gain_unconstrained']:.3f}x at {headline['gain_unconstrained_floor']}, "
              f"INADMISSIBLE under the signaling bound)", flush=True)

    summary = dict(build=info, quick=quick, operating_point=_jsonable({k: v for k, v in op.items()}), base_spec=BASE,
                   anchors=anchors, pkmm=pk, qet=qet, qei=qei, waveforms=_jsonable(results), exchange_rates=_jsonable(xr),
                   inverse_design=_jsonable(inv), global_search=_jsonable(gs), scenario2_twin=_jsonable(twin), proxy_consistency=_jsonable(proxy),
                   search_wall_clock_s=gs_wall, workers=1 if quick else int(args.workers), headline=_jsonable(headline),
                   wall_clock_s=time.time() - t_all)
    tag = "_quick" if quick else ""
    (out_dir / f"summary{tag}.json").write_text(json.dumps(summary, indent=1))
    arrays = {"__build__": np.array(json.dumps(info, indent=1), dtype="U8192")}
    for res in results:
        for vname, row in res["rows"].items():
            key = f"{res['floor']}__{vname}"
            arrays[f"{key}__theta"] = np.asarray(row["theta"])
            arrays[f"{key}__V_AB"] = np.asarray(row["V_AB"])
            arrays[f"{key}__scalars"] = np.array([row["E_N"], row["work"], row["comm_fraction"], float(row.get("gain", 1.0)), float(row["converged"]),
                                                  float(row["halvings"]), float(row["audits_passed"])])
    for f, g in gs.items():  # the global search's best round-tripped waveform per floor (admissible or not: its check is in the summary)
        row = g["best"] if g["best"] is not None else (max(g["roundtrips"], key=lambda r: r["E_N"]) if g["roundtrips"] else None)
        if row is not None:
            arrays[f"{f}__global__theta"] = np.asarray(row["theta"])
            arrays[f"{f}__global__V_AB"] = np.asarray(row["V_AB"])
            arrays[f"{f}__global__scalars"] = np.array([row["E_N"], row["work"], row["comm_fraction"], float(row["gain"] if row["gain"] is not None else 1.0),
                                                        float(row["converged"]), float(row["halvings"]), float(row["admissible"])])
    arrays["scenario2__baseline__theta"] = np.asarray(twin["rows"]["baseline"]["theta"])
    arrays["scenario2__baseline__V_AB"] = np.asarray(twin["rows"]["baseline"]["V_AB"])
    if twin["best"] is not None:
        arrays["scenario2__constrained__theta"] = np.asarray(twin["best"]["theta"])
        arrays["scenario2__constrained__V_AB"] = np.asarray(twin["best"]["V_AB"])
        arrays["scenario2__constrained__scalars"] = np.array([twin["best"]["E_N"], twin["best"]["work"], twin["best"]["comm_fraction"], float(twin["go"]),
                                                              float(twin["best"]["converged"]), float(twin["best"]["halvings"]), float(twin["best"]["admissible"])])
    np.savez(out_dir / f"waveforms{tag}.npz", **arrays)
    print(f"\nwrote {out_dir / f'summary{tag}.json'} and {out_dir / f'waveforms{tag}.npz'} ({time.time() - t_all:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
