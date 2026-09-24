#!/usr/bin/env python
"""Time-modulated vacua: amplification bounds, loss thresholds, pumped harvesting.

Re-runnable end to end:

    .venv/bin/python papers/time-modulated-vacua/notebook.py            # full (~10 min)
    .venv/bin/python papers/time-modulated-vacua/notebook.py --quick    # smoke (~30 s)
    .venv/bin/python papers/time-modulated-vacua/notebook.py --sections bounds,composite

THE CLAIMS (three, each a separate data file)
=============================================
(a) AMPLIFICATION BOUND.  For a quadratic lattice field whose coupling
    matrix is modulated uniformly, K(t) = K0 (1 + d f(omega_mod t)), every
    normal mode is a Mathieu-type oscillator and the largest Floquet
    multiplier per period, lambda(d, omega_mod), is a single-mode function
    of (d, omega_mod/omega_k).  The candidate's numerically converged
    max-gain curves ln lambda_max(d) = max over omega_mod, for cosine,
    triangular and 50 % square profiles, are all bounded by the EXACT
    two-level (bang-bang) optimum ln lambda <= artanh(d) — attained by a
    two-piece drive with quarter-period dwell times pi/(2 omega_1),
    pi/(2 omega_2) — and the cosine drive reaches (pi/4) d (1 + c2 d^2 + ...)
    of it at the principal resonance (Landau-Lifshitz Sec. 27 at leading
    order).  The conjecture stated in ``summary.json['bounds']['conjecture']``
    is that artanh(d) bounds the per-cycle gain of ANY bounded modulation of
    relative depth d; the numbers here are its converged evidence for three
    profile families, with the convergence (dt-halving levels, omega
    refinement, tolerance sweeps) recorded per point.
(b) LOSS THRESHOLD.  With a matched loss channel of transmissivity eta per
    period the photon-number multiplier is eta lambda^2, so the threshold is
    eta_th(d) = 1/lambda_max(d)^2; for a mode of quality factor Q at the
    principal resonance eta = exp(-pi/Q), hence Q_th(d) = pi/(2 ln lambda_max)
    and d_th(Q) ~ 2/Q for the cosine drive.  The mapping to platforms uses
    only transcribed numbers: Wilson et al. 2011 report parasitic line
    resonances with Q ~ 30-50 and a 10 % SQUID-inductance modulation
    (arXiv:1105.4714, p. 7); Wang et al., Nature Photonics 18 (2024),
    arXiv:2310.02786, state that non-resonant optical PTCs need
    Delta n/n ~ 1 at ~2 omega, that low-loss materials saturate at Delta n/n
    < 1 %, that their resonant design amplifies if gamma/omega_r < 0.05,
    and that silicon has gamma/omega_r ~ 1e-5 in the near infrared.
(c) PUMP-ENHANCED HARVESTING.  Two harmonic (oscillator-UDW) detectors,
    x-x coupled to a Dirichlet chain with a cos^2 window, harvest E_N from
    the vacuum (baseline) and from the same field Floquet-prepared by a
    'sin' coupling modulation before (n_prep periods) and optionally during
    the window; every point is one ``vacuum.protocol.run_protocol`` call
    (passivity claim on the entry vacuum, precision audit on the detector
    block, ledger closure at 1e-9 with the pump AND switching work charged),
    gated at 1e-9 on E_N and both detector occupations.  The map reports
    E_N / E_N(baseline), the detector occupations (local noise), the pump
    and switching work and E_N per unit work, in causally connected and
    spacelike (window shorter than the separation) configurations.

Every number backing those statements is computed here and written under
``data/``; none is restated in this docstring, so that the file's sha256 —
recorded in the archives — identifies the code that produced the numbers.

WHAT GOES THROUGH WHAT
======================
* bounds / threshold: ``vacuum.floquet.max_gain_over_frequency`` (batched
  gated 2 x 2 monodromies + golden-section refinement), ``bang_bang_bound``,
  ``stability_chart`` for the multimode lattice cross-check,
  ``mathieu_tongue_edges`` (scipy characteristic values) for gap widths.
* DCE regime: ``wilson_regime`` (experiments file via ``vacuum.experiments_io``),
  ``dce_chain_K`` + ``modulated_K(kind='boundary')``, ``floquet_map`` (CF4,
  gate 1e-8), ``dce_spectrum`` / ``parabola_fit``.
* composite: ``vacuum.floquet.pumped_harvest`` -> ``vacuum.protocol.run_protocol``
  with ``vacuum.detectors.attach_detectors(coupling='xx')`` and
  ``harvested_log_negativity`` (mp-margin rule, ``flagged`` carried).

Conventions per docs/API.md: hbar = 1, V_vac = I/2, block ordering, nats.
Pure-functional over arrays: every sweep function takes its configuration
and returns rows; the only side effects are the archives written by main().
"""

from __future__ import annotations

import argparse
import tempfile
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import vacuum
from vacuum.core import ground_state_cov, harmonic_chain_K, thermal_state_cov
from vacuum.experiments_io import bose_occupation
from vacuum.floquet import (
    bang_bang_bound,
    dce_chain_K,
    dce_spectrum,
    floquet_map,
    loss_threshold,
    mathieu_tongue_edges,
    max_gain_over_frequency,
    modulated_K,
    normal_modes,
    parabola_fit,
    power_iterate,
    pumped_harvest,
    stability_chart,
    wilson_flux_density,
    wilson_regime,
)

from vacuum.floquet.bounds import (
    adversarial_search,
    multimode_check,
    operator_norm_depth,
    optimize_control,
    random_controls,
    rate_optimum,
)
from vacuum.floquet.composite import pumped_harvest_split

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# ---------------------------------------------------------------- config --

DEPTHS_FULL = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                        0.8, 0.9, 0.95])
DEPTHS_QUICK = np.array([0.05, 0.3, 0.8])
PROFILES = ("cos", "square", "triangle")
GATE_TOL = 1e-10       # monodromy dt-halving gate (relative movement of S_F)
OMEGA_TOL = 1e-9       # golden-section refinement of omega_mod
COMPOSITE_TOL = 1e-9   # gate on E_N and n_d (spec M2.2 value)
DCE_TOL = 1e-8         # monodromy gate for the N = 300 boundary drive

# transcribed platform anchors (see docstring; numbers verbatim from the sources)
PLATFORMS = {
    "wilson2011_parasitic_Q_low": {"Q": 30.0, "source": "Wilson et al. 2011, p. 7: 'Q ~ 30-50'"},
    "wilson2011_parasitic_Q_high": {"Q": 50.0, "source": "Wilson et al. 2011, p. 7: 'Q ~ 30-50'"},
    "wang2024_resonant_loss_threshold": {"Q": 20.0, "source": "Wang et al. 2024 Sec. I.3: amplification remains significant if gamma/omega_r < 0.05"},
    "wang2024_silicon_NIR": {"Q": 1.0e5, "source": "Wang et al. 2024 Sec. I.3: silicon gamma/omega_r ~ 1e-5"},
}
WILSON_DEPTH = 0.10  # 'a 10% modulation of the SQUID inductance' (Wilson 2011, p. 3)
WANG_LOWLOSS_DEPTH = 0.01  # 'relative change ... saturate at less than 1%' (Wang 2024, Intro)


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _build_info(t_start):
    src = Path(__file__).read_bytes()
    return {
        "generator": str(Path(__file__).relative_to(HERE.parent.parent)),
        "generator_sha256": hashlib.sha256(src).hexdigest(),
        "git_head": _git_head(),
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package": f"vacuum {vacuum.__version__}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "wall_clock_s": None,
        "_t_start": t_start,
    }


# --------------------------------------------------------------- bounds --


def section_bounds(quick=False):
    depths = DEPTHS_QUICK if quick else DEPTHS_FULL
    n_scan = 31 if quick else 61
    rows = []
    t0 = time.perf_counter()
    for prof in PROFILES:
        for d in depths:
            r = max_gain_over_frequency(float(d), prof, n_steps=32, conv_tol=GATE_TOL,
                                        omega_tol=OMEGA_TOL, n_scan=n_scan)
            rows.append(dict(profile=prof, m=1, **{k: r[k] for k in (
                "depth", "omega_mod", "ln_multiplier", "rate", "period", "iterations",
                "n_steps", "halvings", "movement")}))
    # second tongue of the cosine drive (subdominant)
    for d in depths[depths >= 0.2]:
        r = max_gain_over_frequency(float(d), "cos", m=2, n_steps=32, conv_tol=GATE_TOL,
                                    omega_tol=OMEGA_TOL, n_scan=n_scan)
        rows.append(dict(profile="cos", m=2, **{k: r[k] for k in (
            "depth", "omega_mod", "ln_multiplier", "rate", "period", "iterations",
            "n_steps", "halvings", "movement")}))
    lam_bb, T_rel = bang_bang_bound(depths)
    bb = {"depth": depths, "ln_multiplier": np.log(lam_bb), "rate": np.log(lam_bb) / (T_rel * np.pi)}

    # convergence demonstration: tolerance sweeps at three depths
    conv = []
    for d in ([0.3] if quick else [0.1, 0.5, 0.9]):
        for gate in ([1e-8, 1e-10] if quick else [1e-6, 1e-8, 1e-10, 1e-12]):
            for wtol in ([1e-7, 1e-9] if quick else [1e-5, 1e-7, 1e-9]):
                r = max_gain_over_frequency(d, "cos", n_steps=32, conv_tol=gate, omega_tol=wtol,
                                            max_halvings=14, n_scan=n_scan)
                conv.append(dict(depth=d, gate_tol=gate, omega_tol=wtol,
                                 ln_multiplier=r["ln_multiplier"], omega_mod=r["omega_mod"],
                                 n_steps=r["n_steps"], halvings=r["halvings"]))
    # fitted correction law for the cosine drive
    cos_rows = [r for r in rows if r["profile"] == "cos" and r["m"] == 1]
    dd = np.array([r["depth"] for r in cos_rows])
    ratio = np.array([r["ln_multiplier"] for r in cos_rows]) / (np.pi * dd / 4.0)
    A = np.stack([dd**2, dd**4], axis=1)
    coef, *_ = np.linalg.lstsq(A, ratio - 1.0, rcond=None)
    fit_resid = float(np.max(np.abs(ratio - 1.0 - A @ coef)))
    # tongue (momentum gap) widths of the principal resonance vs depth
    widths = []
    for d in depths:
        lo, hi = mathieu_tongue_edges(1.0, float(d), 1)
        widths.append(dict(depth=float(d), omega_lo=lo, omega_hi=hi,
                           relative_width=(hi - lo) / 2.0, ratio_to_d_over_2=(hi - lo) / 2.0 / (d / 2.0)))
    # multimode lattice cross-check: chain chart equals single-mode law at omega_mod/omega_k
    N = 24
    K0 = harmonic_chain_K(N, 0.5)
    omega_k, _ = normal_modes(K0)
    dgrid = np.array([0.1, 0.3])
    wgrid = np.linspace(0.8, 4.5, 60 if quick else 200)
    chart = stability_chart(K0, dgrid, wgrid, n_steps=32, conv_tol=GATE_TOL, mass=0.5)
    single = stability_chart(np.array([[1.0]]), dgrid,
                             np.unique(np.round((wgrid[None, :] / omega_k[:, None]).ravel(), 12)),
                             n_steps=32, conv_tol=GATE_TOL)
    worst = 0.0
    for i in range(dgrid.size):
        for j, w in enumerate(wgrid):
            pred = max(
                float(np.interp(w / wk, single.omega_grid, single.multiplier[i])) for wk in omega_k
            )
            worst = max(worst, abs(chart.multiplier[i, j] - pred))
    lattice_check = {"N": N, "mass": 0.5, "worst_abs_diff_multiplier": worst,
                     "chart_info": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                    for k, v in chart.info.items()}}
    conjecture = {
        "statement": ("ln max|eig S_F| <= artanh(d) per cycle for every bounded modulation of "
                      "relative depth d (bang-bang two-level optimum, attained); the cosine "
                      "drive reaches (pi/4) d (1 + c2 d^2 + c4 d^4) at the principal resonance"),
        "c2": float(coef[0]), "c4": float(coef[1]), "fit_max_residual": fit_resid,
        "max_ratio_cos_to_bangbang_per_period": float(np.max(
            np.array([r["ln_multiplier"] for r in cos_rows]) / np.arctanh(dd))),
        "max_ratio_any_profile_to_bangbang_per_period": float(np.max([
            r["ln_multiplier"] / np.arctanh(r["depth"]) for r in rows])),
        "second_tongue_max_ratio_to_first": float(max(
            [r2["ln_multiplier"] / r1["ln_multiplier"]
             for r2 in rows if r2["profile"] == "cos" and r2["m"] == 2
             for r1 in cos_rows if r1["depth"] == r2["depth"]] or [0.0])),
        "second_tongue_max_rate_ratio_to_first": float(max(
            [r2["rate"] / r1["rate"]
             for r2 in rows if r2["profile"] == "cos" and r2["m"] == 2
             for r1 in cos_rows if r1["depth"] == r2["depth"]] or [0.0])),
    }
    return {"rows": rows, "bang_bang": bb, "convergence": conv, "conjecture": conjecture,
            "tongue_widths": widths, "lattice_check": lattice_check,
            "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------ threshold --


def section_threshold(bounds):
    cos_rows = sorted([r for r in bounds["rows"] if r["profile"] == "cos" and r["m"] == 1],
                      key=lambda r: r["depth"])
    d = np.array([r["depth"] for r in cos_rows])
    lnlam = np.array([r["ln_multiplier"] for r in cos_rows])
    eta_th = loss_threshold(np.exp(lnlam))
    Q_th = np.pi / (2.0 * lnlam)  # principal resonance: eta = exp(-pi/Q)
    # d_th(Q) by inverting the monotone table (small-d law below the table)
    Q_grid = np.logspace(0, 7, 71)
    ln_needed = np.pi / (2.0 * Q_grid)
    d_th = np.where(ln_needed < lnlam[0], ln_needed / (np.pi / 4.0),
                    np.interp(ln_needed, lnlam, d, right=np.nan))
    bb_ln = np.arctanh(d)
    platforms = {}
    for name, p in PLATFORMS.items():
        need = np.pi / (2.0 * p["Q"])
        dth = float(need / (np.pi / 4.0)) if need < lnlam[0] else float(np.interp(need, lnlam, d, right=np.nan))
        platforms[name] = dict(Q=p["Q"], source=p["source"], eta_per_period=float(np.exp(-np.pi / p["Q"])),
                               d_threshold_cos=dth, d_threshold_bangbang=float(np.tanh(need)),
                               wilson_10pct_above_threshold=bool(WILSON_DEPTH > dth),
                               lowloss_1pct_above_threshold=bool(WANG_LOWLOSS_DEPTH > dth))
    return {"depth": d, "ln_multiplier_cos": lnlam, "eta_threshold": eta_th, "Q_threshold": Q_th,
            "Q_grid": Q_grid, "d_threshold_cos": d_th, "d_threshold_bangbang": np.tanh(ln_needed),
            "platforms": platforms,
            "mapping": "eta = exp(-2 pi omega/(omega_mod Q)); at omega_mod = 2 omega, eta = exp(-pi/Q); "
                       "threshold eta lambda_max^2 = 1  <=>  Q_th = pi/(2 ln lambda_max(d))"}


# ------------------------------------------------------------------ dce --


def section_dce(quick=False):
    reg = wilson_regime(0.5)
    N = 120 if quick else 300
    wd, dL = reg.omega_d_lattice, reg.dL_over_v_lattice
    out = {"regime": {k: (v if isinstance(v, (str, float, int, tuple)) else float(v))
                      for k, v in reg.__dict__.items()}, "N": N, "rows": []}
    t0 = time.perf_counter()
    K0 = dce_chain_K(N, 0.5)
    V0 = ground_state_cov(K0)
    cache = {}
    for L0, scale, T_lat in ([(0.5, 1.0, 0.0), (0.5, 0.5, 0.0), (0.5, 1.0, reg.temperature_lattice)]
                             if quick else
                             [(0.5, 1.0, 0.0), (0.5, 0.5, 0.0), (0.5, 1.5, 0.0), (0.3, 1.0, 0.0),
                              (1.0, 1.0, 0.0), (2.5, 1.0, 0.0), (0.5, 1.0, reg.temperature_lattice)]):
        K = K0 if L0 == 0.5 else dce_chain_K(N, L0)
        depth = min(scale * dL / L0, 0.5)
        dL_eff = depth * L0
        drv = modulated_K(K, depth, wd, "cos", kind="boundary", L0=L0)
        key = (L0, depth)
        if key not in cache:
            cache[key] = floquet_map(drv, n_steps=32, conv_tol=DCE_TOL, max_halvings=8)
        fm = cache[key]
        T = drv.period
        n_max = int(1.9 * N / T)
        n1 = n_max // 3
        Vstart = ground_state_cov(K) if T_lat == 0.0 else thermal_state_cov(K, T_lat)
        V1 = power_iterate(Vstart, fm.S, n1)
        V2 = power_iterate(V1, fm.S, n_max - n1)
        sp = dce_spectrum(V2, K, (n_max - n1) * T, wd, dL_eff, n_bins=10, V_ref=V1)
        fit = parabola_fit(sp)
        row = dict(L0=L0, depth_scale=scale, dL_eff=dL_eff, kL0=wd * L0, T_lattice=T_lat,
                   epsilon=0.5 * dL_eff * wd, n_periods=n_max - n1, t=(n_max - n1) * T,
                   multiplier=fm.multiplier, map_n_steps=fm.n_steps, map_movement=fm.movement,
                   A_ratio=fit["A_ratio"], relative_rms_residual=fit["relative_rms_residual"],
                   R2=fit["R2"], total_flux=sp.total_flux,
                   total_flux_prediction=sp.total_flux_prediction,
                   flux_ratio=sp.total_flux / sp.total_flux_prediction,
                   omega_centers=sp.omega_centers, n_out=sp.n_out, prediction=sp.prediction)
        if T_lat > 0.0:
            n_in = np.vectorize(lambda w, _T=T_lat: bose_occupation(w, _T) if w > 0 else 0.0)
            stim = wilson_flux_density(sp.omega_centers, wd, dL_eff, n_in=n_in) - n_in(sp.omega_centers)
            row["stimulated_prediction"] = stim
            row["ratio_to_stimulated"] = sp.n_out / stim
        out["rows"].append(row)
    r1 = [r for r in out["rows"] if r["L0"] == 0.5 and r["T_lattice"] == 0.0]
    fl = {r["depth_scale"]: r["total_flux"] for r in r1}
    if 0.5 in fl and 1.0 in fl:
        out["depth_exponent_0.5_to_1"] = float(np.log(fl[1.0] / fl[0.5]) / np.log(2.0))
    if 1.5 in fl and 1.0 in fl:
        out["depth_exponent_1_to_1.5"] = float(np.log(fl[1.5] / fl[1.0]) / np.log(1.5))
    out["seconds"] = time.perf_counter() - t0
    return out


# ------------------------------------------------------------ composite --


def section_composite(quick=False):
    N_f, m, sites, gap, lam = 16, 0.5, (4, 11), 1.2, 0.4
    K0 = harmonic_chain_K(N_f, m, bc="dirichlet")
    omega_f, _ = normal_modes(K0)
    omegas = np.array([2.4]) if quick else np.array([1.6, 2.0, 2.4, 2.8, 3.2])
    depths = np.array([0.05, 0.2]) if quick else np.array([0.02, 0.05, 0.1, 0.2])
    preps = [2] if quick else [2, 4]
    configs = [("connected", 3)] + ([] if quick else [("spacelike", 2)])
    rows = []
    t0 = time.perf_counter()

    def run(drive, wmod, n_prep, n_win, during, label):
        ts = time.perf_counter()
        p = pumped_harvest(K0, drive, sites, (gap, gap), lam, n_prep, n_win,
                           period=2 * np.pi / wmod, pump_during_coupling=during,
                           n_substeps_per_period=8, conv_tol=COMPOSITE_TOL, max_halvings=8)
        j = int(np.argmin(np.abs(omega_f - gap)))
        return dict(label=label, omega_mod=float(wmod), depth=p.depth, n_prep=n_prep, n_win=n_win,
                    pump_during_coupling=during, t_prep=p.t_prep, t_win=p.t_win,
                    separation=sites[1] - sites[0], spacelike=bool(p.t_win < sites[1] - sites[0]),
                    E_N=p.E_N, flagged=p.flagged, n_d0=float(p.n_d[0]), n_d1=float(p.n_d[1]),
                    work_total=p.work_total, work_pump=p.work_pump, work_switching=p.work_switching,
                    E_N_per_work=p.E_N / p.work_total if p.work_total > 0 else np.nan,
                    field_n_at_gap=float(p.field_occupations[j]),
                    field_n_total=float(np.sum(p.field_occupations)),
                    converged=p.converged, n_substeps_per_period=p.n_substeps_per_period,
                    halvings=p.halvings, movement=p.movement, ledger_defect=p.ledger_defect,
                    passivity_passed=bool(p.passivity_passed), all_audits_passed=p.all_audits_passed,
                    runtime_s=time.perf_counter() - ts)

    for cname, n_win in configs:
        for wmod in omegas:
            base = run(None, wmod, 0, n_win, True, f"{cname}-baseline")
            rows.append(dict(base, config=cname, enhancement=1.0, E_N_baseline=base["E_N"]))
            for d in depths:
                drive = modulated_K(K0, float(d), float(wmod), "sin")
                for n_prep in preps:
                    for during in ((True,) if cname == "spacelike" else (True, False)):
                        if wmod != 2.4 and (n_prep == 4 or not during) and not quick:
                            continue  # off-resonant pumps: one representative timeline
                        r = run(drive, wmod, n_prep, n_win, during, f"{cname}-pumped")
                        # a baseline that died suddenly (E_N = 0) has no finite enhancement
                        enh = r["E_N"] / base["E_N"] if base["E_N"] > 0.0 else np.nan
                        rows.append(dict(r, config=cname, enhancement=enh,
                                         E_N_baseline=base["E_N"]))
    pumped = [r for r in rows if r["depth"] > 0]
    finite = [r for r in pumped if np.isfinite(r["enhancement"])]
    best = max(finite, key=lambda r: r["enhancement"])
    best_eff = max(pumped, key=lambda r: r["E_N_per_work"])
    zero_base = sorted({(r["config"], r["omega_mod"]) for r in rows if r["depth"] == 0 and r["E_N"] == 0.0})
    revived = [r for r in pumped if not np.isfinite(r["enhancement"]) and r["E_N"] > 0.0]
    base_eff = {r["omega_mod"]: r["E_N_per_work"] for r in rows if r["depth"] == 0 and r["config"] == "connected"}
    verdict = {
        "max_enhancement": best["enhancement"], "at": {k: best[k] for k in (
            "omega_mod", "depth", "n_prep", "n_win", "pump_during_coupling", "config")},
        "max_enhancement_spacelike": max([r["enhancement"] for r in finite if r["config"] == "spacelike"] or [np.nan]),
        "max_E_N_spacelike": max([r["E_N"] for r in pumped if r["config"] == "spacelike"] or [np.nan]),
        "baselines_with_zero_E_N": [list(z) for z in zero_base],
        "n_pumped_points_reviving_a_dead_baseline": len(revived),
        "max_E_N_over_dead_baselines": max([r["E_N"] for r in revived] or [np.nan]),
        "best_E_N_per_work_pumped": best_eff["E_N_per_work"],
        "baseline_E_N_per_work": base_eff,
        "efficiency_ratio_best_pumped_to_baseline": best_eff["E_N_per_work"] / base_eff[best_eff["omega_mod"]]
        if best_eff["omega_mod"] in base_eff else np.nan,
        "n_points": len(rows), "all_converged": all(r["converged"] for r in rows),
        "all_audits_passed": all(r["all_audits_passed"] for r in rows),
        "worst_ledger_defect": max(r["ledger_defect"] for r in rows),
        "any_flagged": any(r["flagged"] for r in rows),
        "worst_movement": max(r["movement"] for r in rows),
    }
    return {"rows": rows, "verdict": verdict, "field_omega": omega_f,
            "setup": dict(N_field=N_f, mass=m, sites=sites, gap=gap, lam_max=lam,
                          profile="sin", switching="cos^2 window of n_win periods"),
            "seconds": time.perf_counter() - t0}


# ------------------------------------------------------- bound theorem --


def section_bound_theorem(quick=False):
    """Theorem A/B of vacuum/floquet/bounds.py: the adversarial battery, the rate
    optimum table and the multimode checks (coupling: exact block structure;
    pattern: the conjectured operator-norm rate bound under CEM attack)."""
    t0 = time.perf_counter()
    depths = (0.1, 0.5) if quick else (0.05, 0.1, 0.3, 0.5, 0.8, 0.95)
    adv = adversarial_search(depths=depths, seed=2026, n_random=200 if quick else 2000,
                             M=32 if quick else 128, n_iter=80 if quick else 300,
                             n_restarts=2 if quick else 8,
                             periods_over_Tstar=(1.0, 2.0) if quick else (0.7, 0.85, 1.0, 1.15, 1.5, 2.0, 2.4, 3.0))
    rate_rows = []
    for d in DEPTHS_FULL:
        r = rate_optimum(float(d))
        rate_rows.append(dict(depth=float(d), lambda_star=r.lambda_star, quarter_period_rate=r.quarter_period_rate,
                              excess=r.excess, theta_up=r.theta_up, tau_high=r.tau_high, tau_low=r.tau_low,
                              period=r.period, ln_growth_per_cycle=r.ln_growth_per_cycle,
                              artanh=float(np.arctanh(d)),
                              tau_high_over_quarter=r.tau_high / (0.5 * np.pi / np.sqrt(1 + d)),
                              tau_low_over_quarter=r.tau_low / (0.5 * np.pi / np.sqrt(1 - d))))
    # multimode: coupling kind (exact block structure) and pattern kind (conjectured)
    rng = np.random.default_rng(7)
    N_mm = 6 if quick else 8
    K0 = harmonic_chain_K(N_mm, 0.5, bc="dirichlet")
    coupling_rows, pattern_rows = [], []
    for i in range(4 if quick else 24):
        fam = ("clipped_fourier", "sign_fourier", "piecewise")[i % 3]
        s = random_controls(rng, 1, 32, fam)[0]
        T = rng.uniform(0.5, 3.5) * np.pi / float(np.max(np.sqrt(np.linalg.eigvalsh(K0))))
        d = rng.uniform(0.1, 0.8)
        c = multimode_check(K0, s, T, d, kind="coupling")
        coupling_rows.append(dict(depth=d, period=T, family=fam, ln_full=c["ln_full"],
                                  max_mode_ratio=c["max_ratio"], block_defect=c["block_defect"],
                                  symplectic_defect=c["symplectic_defect"]))
    n_pat = 8 if quick else 60
    patterns = []
    for i in range(n_pat):
        P = rng.normal(size=(N_mm, N_mm))
        P = 0.5 * (P + P.T)
        d_eff = rng.uniform(0.1, 0.7)
        d = d_eff / operator_norm_depth(K0, P, 1.0)
        s = random_controls(rng, 1, 24, ("sign_fourier", "clipped_fourier")[i % 2])[0]
        T = rng.uniform(1.0, 5.0)
        c = multimode_check(K0, s, T, d, kind="pattern", pattern=P)
        patterns.append(P)
        pattern_rows.append(dict(depth=d, d_eff=c["d_eff"], period=T, ln_full=c["ln_full"],
                                 rate_ratio=c["rate_ratio"], cycle_count_ratio=c["cycle_count_ratio"],
                                 omega_max=c["omega_max"], attacked=False))
    # CEM attack on the pattern bound: the worst random patterns, control optimised (16 pieces)
    worst_idx = np.argsort([-r["rate_ratio"] for r in pattern_rows])[: (1 if quick else 3)]
    rng_attack = np.random.default_rng(99)
    for idx in worst_idx:
        row = pattern_rows[int(idx)]
        P = patterns[int(idx)]
        d = row["depth"]
        T = row["period"]

        def objective(sb, _P=P, _d=d, _T=T):
            return np.array([multimode_check(K0, s, _T, _d, kind="pattern", pattern=_P)["ln_full"] for s in sb])

        r = optimize_control(d, T, 16, method="cem", rng=rng_attack, n_iter=8 if quick else 40,
                             batch=32 if quick else 64, objective=objective)
        c = multimode_check(K0, r["control"], T, d, kind="pattern", pattern=P)
        pattern_rows.append(dict(depth=d, d_eff=c["d_eff"], period=T, ln_full=c["ln_full"],
                                 rate_ratio=c["rate_ratio"], cycle_count_ratio=c["cycle_count_ratio"],
                                 omega_max=c["omega_max"], attacked=True))
    return {
        "adversarial": {k: v for k, v in adv.items() if k != "rows"},
        "adversarial_rows": adv["rows"],
        "rate_optimum": rate_rows,
        "multimode_coupling": coupling_rows,
        "multimode_pattern": pattern_rows,
        "verdict": {
            "max_ratio_any_control": adv["max_ratio"],
            "worst_row": adv["worst_row"],
            "bang_bang_attainment_dev": adv["bang_bang_ratio_max_dev"],
            "coupling_worst_block_defect": max(r["block_defect"] for r in coupling_rows),
            "coupling_worst_mode_ratio": max(r["max_mode_ratio"] for r in coupling_rows),
            "pattern_max_rate_ratio": max(r["rate_ratio"] for r in pattern_rows),
            "pattern_max_rate_ratio_attacked": max(r["rate_ratio"] for r in pattern_rows if r["attacked"]),
            "statement_proved": (
                "ln rho(S_F) <= Z artanh(d) for every measurable |s| <= 1, Z = zeros per period of the "
                "unstable Floquet solution (Z = 1 on the principal tongue); equality iff s = sgn(sin 2 theta) "
                "(quarter-period bang-bang).  Maximal Lyapunov exponent = omega0 lambda*(d), root of "
                "artanh(d/(1+lambda^2)) = lambda tau(lambda); the quarter-period drive is not rate-optimal."
            ),
            "statement_conjectured": (
                "non-commuting pattern modulation K0 + d s(t) P with (1 -/+ d_eff) K0 bracketing K(t): "
                "ln rho(S_F)/T <= omega_max lambda*(d_eff) (tested on the battery above, not proved)"
            ),
        },
        "seconds": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------- split --


def _split_row(K0, row, sites, gap, lam):
    wmod = row["omega_mod"]
    drive = None if row["depth"] == 0.0 else modulated_K(K0, float(row["depth"]), float(wmod), "sin")
    sp = pumped_harvest_split(K0, drive, sites, (gap, gap), lam, int(row["n_prep"]), int(row["n_win"]),
                              period=2 * np.pi / wmod, pump_during_coupling=bool(row["pump_during_coupling"]),
                              n_grid_per_period=32, rtol=1e-7, max_halvings=6)
    return dict(M_abs=abs(sp.M), M_vac_abs=abs(sp.M_vac), M_comm_abs=abs(sp.M_comm),
                comm_fraction=sp.comm_fraction, vac_fraction=sp.vac_fraction,
                comm_estimator=sp.estimator, N_pert=sp.N, N_vac_pert=sp.N_vac, N_comm_pert=sp.N_comm,
                E_N_perturbative=sp.E_N_perturbative, P_A=sp.P_A, P_B=sp.P_B, C_abs=abs(sp.C),
                M_comm_static_abs=abs(sp.M_comm_static), comm_drive_shift=sp.comm_drive_shift,
                split_converged=sp.converged, split_n_grid_per_period=sp.n_grid_per_period,
                split_movement=sp.movement, field_n_total_at_prep=sp.field_n_total)


def section_split(quick=False):
    """The M2.5 vacuum/communication split of every composite row (the same
    grid as section_composite, re-run so E_N and the split share one build)."""
    t0 = time.perf_counter()
    comp = section_composite(quick)
    setup = comp["setup"]
    K0 = harmonic_chain_K(setup["N_field"], setup["mass"], bc="dirichlet")
    rows = []
    for r in comp["rows"]:
        rows.append(dict(r, **_split_row(K0, r, setup["sites"], setup["gap"], setup["lam_max"])))
    pumped = [r for r in rows if r["depth"] > 0]
    base = {(r["config"], r["omega_mod"]): r for r in rows if r["depth"] == 0}
    for r in pumped:
        b = base[(r["config"], r["omega_mod"])]
        r["comm_fraction_baseline"] = b["comm_fraction"]
        r["M_abs_baseline"] = b["M_abs"]
    connected = [r for r in pumped if not r["spacelike"]]
    spacelike = [r for r in pumped if r["spacelike"]]
    live = [r for r in pumped if r["E_N"] > 0]
    verdict = {
        "n_rows": len(rows),
        "all_split_converged": all(r["split_converged"] for r in rows),
        "baseline_comm_fraction": {f"{k[0]}@{k[1]}": v["comm_fraction"] for k, v in base.items()},
        "pumped_comm_fraction_max": max(r["comm_fraction"] for r in pumped),
        "pumped_comm_fraction_max_connected": max([r["comm_fraction"] for r in connected] or [np.nan]),
        "pumped_comm_fraction_max_spacelike": max([r["comm_fraction"] for r in spacelike] or [np.nan]),
        "pumped_comm_fraction_max_where_E_N_positive": max([r["comm_fraction"] for r in live] or [np.nan]),
        "pumped_vac_fraction_min_where_E_N_positive": min([r["vac_fraction"] for r in live] or [np.nan]),
        "comm_estimator_max_pumped": max(r["comm_estimator"] for r in pumped),
        "headline_rows": [
            {k: r[k] for k in ("config", "omega_mod", "depth", "n_prep", "n_win", "pump_during_coupling",
                               "spacelike", "t_win", "E_N", "E_N_baseline", "enhancement", "comm_fraction",
                               "comm_fraction_baseline", "vac_fraction", "M_abs", "M_abs_baseline",
                               "comm_drive_shift", "E_N_perturbative", "comm_estimator")}
            for r in pumped if r["depth"] == 0.1 and r["n_prep"] == 2 and r["pump_during_coupling"]
        ],
        "perturbative_over_exact_E_N": {
            "min": min(r["E_N_perturbative"] / r["E_N"] for r in live),
            "max": max(r["E_N_perturbative"] / r["E_N"] for r in live),
        },
        "mislabel_note": (
            "the 'spacelike' config at omega_mod = 1.6 has t_win = 2T = 7.85 > separation 7 (flag False): "
            "the 4.75x of the earlier MANIFEST is a connected window; the truly spacelike rows all have "
            "E_N(baseline) = 0"
        ),
    }
    return {"rows": rows, "verdict": verdict, "setup": setup, "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------- finite size --


def section_finite_size(quick=False):
    """N doubling at fixed detector geometry (separation 7, centred, walls receding)."""
    t0 = time.perf_counter()
    m, gap, lam, d, n_prep = 0.5, 1.2, 0.4, 0.1, 2
    sizes = (16, 32) if quick else (16, 32, 64)
    configs = [("connected", 3, (2.0,) if quick else (1.6, 2.0, 2.4, 2.8))]
    if not quick:
        configs.append(("spacelike", 2, (1.6, 2.0)))
    rows = []
    for N_f in sizes:
        K0 = harmonic_chain_K(N_f, m, bc="dirichlet")
        omega_f, _ = normal_modes(K0)
        a = (N_f - 7) // 2
        sites = (a, a + 7)
        for cname, n_win, omegas in configs:
            for wmod in omegas:
                out = []
                for depth in (0.0, d):
                    ts = time.perf_counter()
                    drive = None if depth == 0.0 else modulated_K(K0, depth, float(wmod), "sin")
                    p = pumped_harvest(K0, drive, sites, (gap, gap), lam, 0 if depth == 0.0 else n_prep, n_win,
                                       period=2 * np.pi / wmod, pump_during_coupling=True,
                                       n_substeps_per_period=8, conv_tol=COMPOSITE_TOL, max_halvings=8)
                    j = int(np.argmin(np.abs(omega_f - gap)))
                    row = dict(N_field=N_f, sites=list(sites), config=cname, omega_mod=float(wmod), depth=float(depth),
                               n_prep=p.n_prep, n_win=n_win, t_win=p.t_win, separation=7,
                               spacelike=bool(p.t_win < 7), E_N=p.E_N, flagged=p.flagged,
                               n_d0=float(p.n_d[0]), n_d1=float(p.n_d[1]), work_total=p.work_total,
                               work_pump=p.work_pump, field_n_at_gap=float(p.field_occupations[j]),
                               field_n_total=float(np.sum(p.field_occupations)),
                               nearest_mode_gap_offset=float(omega_f[j] - gap), converged=p.converged,
                               n_substeps_per_period=p.n_substeps_per_period, halvings=p.halvings,
                               movement=p.movement, ledger_defect=p.ledger_defect,
                               all_audits_passed=p.all_audits_passed, runtime_s=time.perf_counter() - ts)
                    row.update(_split_row(K0, dict(row, pump_during_coupling=True), sites, gap, lam))
                    out.append(row)
                base, pumped = out
                enh = pumped["E_N"] / base["E_N"] if base["E_N"] > 0 else np.nan
                rows.append(dict(base, enhancement=1.0, E_N_baseline=base["E_N"]))
                rows.append(dict(pumped, enhancement=enh, E_N_baseline=base["E_N"]))
    # convergence table: per (config, omega_mod), E_N / enhancement / comm fraction vs N
    table = {}
    for cname, n_win, omegas in configs:
        for wmod in omegas:
            key = f"{cname}@{wmod}"
            seq = [r for r in rows if r["config"] == cname and r["omega_mod"] == wmod]
            ent = {}
            for r in seq:
                tag = "pumped" if r["depth"] > 0 else "baseline"
                ent.setdefault(tag, {})[str(r["N_field"])] = dict(
                    E_N=r["E_N"], comm_fraction=r["comm_fraction"], field_n_at_gap=r["field_n_at_gap"],
                    nearest_mode_gap_offset=r["nearest_mode_gap_offset"], n_d0=r["n_d0"])
            enh = {str(r["N_field"]): r["enhancement"] for r in seq if r["depth"] > 0}
            ent["enhancement"] = enh
            Ns = sorted(int(k) for k in enh)
            if len(Ns) >= 2:
                e1, e2 = enh[str(Ns[-2])], enh[str(Ns[-1])]
                ent["enhancement_rel_change_last_doubling"] = abs(e2 - e1) / abs(e2) if np.isfinite(e1) and np.isfinite(e2) and e2 != 0 else np.nan
                p1 = [r for r in seq if r["depth"] > 0 and r["N_field"] == Ns[-2]][0]["E_N"]
                p2 = [r for r in seq if r["depth"] > 0 and r["N_field"] == Ns[-1]][0]["E_N"]
                ent["pumped_E_N_rel_change_last_doubling"] = abs(p2 - p1) / abs(p2) if p2 != 0 else np.nan
            table[key] = ent
    finite_changes = [v["enhancement_rel_change_last_doubling"] for v in table.values()
                      if "enhancement_rel_change_last_doubling" in v and np.isfinite(v["enhancement_rel_change_last_doubling"])]
    verdict = {
        "sizes": list(sizes),
        "all_converged": all(r["converged"] for r in rows),
        "all_audits_passed": all(r["all_audits_passed"] for r in rows),
        "worst_ledger_defect": max(r["ledger_defect"] for r in rows),
        "enhancement_rel_change_last_doubling_max": max(finite_changes) if finite_changes else np.nan,
        "enhancement_rel_change_last_doubling": {k: v.get("enhancement_rel_change_last_doubling") for k, v in table.items()},
        "pumped_E_N_rel_change_last_doubling": {k: v.get("pumped_E_N_rel_change_last_doubling") for k, v in table.items()},
    }
    return {"rows": rows, "table": table, "verdict": verdict, "seconds": time.perf_counter() - t0,
            "setup": dict(mass=m, gap=gap, lam_max=lam, depth=d, n_prep=n_prep, separation=7,
                          geometry="detectors centred, separation 7 sites, Dirichlet walls receding with N")}


# ----------------------------------------------------------------- io --


def _rows_to_npz(path, rows, build):
    keys = sorted({k for r in rows for k in r if not isinstance(r[k], (np.ndarray, list, tuple, dict))})
    cols = {k: np.array([r.get(k, np.nan) for r in rows], dtype=object) for k in keys}
    arrays = {}
    for k, v in cols.items():
        try:
            arrays[k] = v.astype(float)
        except (TypeError, ValueError):
            arrays[k] = v.astype(str)
    np.savez(path, __columns__=np.array(keys), __build__=np.array(json.dumps(build)), **arrays)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def main(argv=None):
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke configuration (~30 s)")
    ap.add_argument("--sections", default="bounds,dce,composite,bound_theorem,split,finite_size")
    ap.add_argument("--out", default=None,
                    help="output directory (default: data/; a fresh temporary directory for --quick, "
                         "so a smoke run never overwrites the archive)")
    args = ap.parse_args(argv)
    sections = args.sections.split(",")
    if args.out:
        DATA = Path(args.out)
    elif args.quick:
        DATA = Path(tempfile.mkdtemp(prefix="time-modulated-vacua-quick-"))
    t_start = time.perf_counter()
    build = _build_info(t_start)
    DATA.mkdir(exist_ok=True)
    suffix = "_quick" if args.quick else ""
    summary = {"__build__": None, "quick": args.quick}
    ALL = ("bounds", "dce", "composite", "bound_theorem", "split", "finite_size")
    partial = set(sections) != set(ALL)
    prior = DATA / f"summary{suffix}.json"
    if partial and prior.exists():
        # a subset run updates its own sections in place and records its build per section;
        # the untouched sections keep the producing build recorded in __build__ / __build_by_section__
        with open(prior) as fh:
            summary = json.load(fh)
        summary.setdefault("__build_by_section__", {})
    if "bounds" in sections:
        b = section_bounds(args.quick)
        th = section_threshold(b)
        _rows_to_npz(DATA / f"bounds{suffix}.npz", b["rows"], build)
        np.savez(DATA / f"loss_threshold{suffix}.npz", __build__=np.array(json.dumps(build)),
                 **{k: np.asarray(v) for k, v in th.items() if isinstance(v, np.ndarray)})
        summary["bounds"] = {k: _jsonable(v) for k, v in b.items() if k != "rows"}
        summary["bounds"]["rows"] = _jsonable(b["rows"])
        summary["loss_threshold"] = _jsonable({k: v for k, v in th.items()})
        print(f"[bounds] {b['seconds']:.1f} s; conjecture {json.dumps(_jsonable(b['conjecture']))}")
    if "dce" in sections:
        d = section_dce(args.quick)
        np.savez(DATA / f"dce{suffix}.npz", __build__=np.array(json.dumps(build)),
                 **{f"row{i}_{k}": np.asarray(v) for i, r in enumerate(d["rows"]) for k, v in r.items()
                    if isinstance(v, np.ndarray)})
        summary["dce"] = _jsonable(d)
        print(f"[dce] {d['seconds']:.1f} s; rows: " + "; ".join(
            f"L0={r['L0']} x{r['depth_scale']} T={r['T_lattice']:.3f}: A_ratio={r['A_ratio']:.4f} "
            f"rms={r['relative_rms_residual']:.4f} flux={r['flux_ratio']:.4f}" for r in d["rows"]))
    if "composite" in sections:
        c = section_composite(args.quick)
        _rows_to_npz(DATA / f"composite{suffix}.npz", c["rows"], build)
        summary["composite"] = _jsonable(c)
        print(f"[composite] {c['seconds']:.1f} s; verdict {json.dumps(_jsonable(c['verdict']))}")
    if "bound_theorem" in sections:
        bt = section_bound_theorem(args.quick)
        _rows_to_npz(DATA / f"bound_adversarial{suffix}.npz", bt["adversarial_rows"], build)
        summary["bound_theorem"] = _jsonable(bt)
        print(f"[bound_theorem] {bt['seconds']:.1f} s; verdict {json.dumps(_jsonable(bt['verdict']))}")
    if "split" in sections:
        s = section_split(args.quick)
        _rows_to_npz(DATA / f"composite_split{suffix}.npz", s["rows"], build)
        summary["composite_split"] = _jsonable(s)
        print(f"[split] {s['seconds']:.1f} s; verdict {json.dumps(_jsonable(s['verdict']))}")
    if "finite_size" in sections:
        f = section_finite_size(args.quick)
        _rows_to_npz(DATA / f"composite_finite_size{suffix}.npz", f["rows"], build)
        summary["composite_finite_size"] = _jsonable(f)
        print(f"[finite_size] {f['seconds']:.1f} s; verdict {json.dumps(_jsonable(f['verdict']))}")
    build["wall_clock_s"] = time.perf_counter() - t_start
    build.pop("_t_start", None)
    build["sections"] = list(sections)
    if partial and prior.exists():
        for sec in sections:
            summary["__build_by_section__"][sec] = build
        summary["__build__"]["last_partial_run"] = build
    else:
        summary["__build__"] = build
    with open(DATA / f"summary{suffix}.json", "w") as fh:
        json.dump(_jsonable(summary), fh, indent=1)
    print(f"wall clock {build['wall_clock_s']:.1f} s -> {DATA}")


if __name__ == "__main__":
    main()
