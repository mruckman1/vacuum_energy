"""papers/interacting-vacua-first-numbers — the EXACT lattice QEI infimum (L5 closure).

The open item this closes
-------------------------
``notebook.py`` S3 reports a **variational upper bound** on the interacting
QEI infimum: the smeared energy minimized over a 9-parameter family of
locally squeezed MPS.  Its gap to the true infimum was unknown -- the family
reaches only 0.72 of the exact free infimum at lambda = 0, so "the free bound
survives, and the whole reduction is mass renormalization" rested on a
comparison of two upper bounds.

This notebook computes the infimum itself.  The smeared energy
O_f = int dt f(t)^2 h_x(t) is a Hermitian operator on the chain; its infimum
over ALL states is its lowest eigenvalue.  h_x(t) = U(t)^dag h_x U(t) is
built as an MPO by operator-space (Heisenberg) TEBD under the phi^4
Hamiltonian and integrated in t by quadrature
(``vacuum.interacting.qei_exact``); DMRG on that MPO returns lambda_min, and
its ground state IS the extremal state.  At lambda = 0 the answer must be the
Williamson infimum of ``vacuum.inequalities.qei`` -- that is the anchor.

What is measured, and what limits it
------------------------------------
Three approximations, all reported per row, with OPPOSITE signs for the two
that matter:

* **Fock cutoff n_max** -- a Rayleigh-Ritz restriction, so it RAISES E_min
  (a smaller Hilbert space cannot reach as low).  Sign: positive deviation.
* **Operator MPO truncation (chi_op)** -- discards operator weight, and
  LOWERS E_min (it can manufacture states below the true infimum).  Sign:
  negative deviation.  This is the new dominant uncertainty and it replaces
  the old "unknown variational gap".
* Trotter step dt and the trapezoid quadrature in t: measured by dt-halving,
  and negligible next to the two above (order-4 Suzuki-Trotter; the Gaussian
  weight makes the trapezoid rule spectrally accurate).

Because the two dominant errors have opposite signs, the lambda = 0 anchor --
where the true answer is known exactly -- measures the total systematic AT THAT
CUTOFF.  It is used here as the anomaly threshold (S5d).  It is deliberately NOT
propagated as an error bar onto the lambda > 0 ratios: those drift with n_max and
chi_op in a way the lambda = 0 residual does not capture (both refinements push
the interacting ratios up while the lambda = 0 residual stays put), so the honest
uncertainty on a lambda > 0 ratio is the measured drift of S5e, not this band.
The occupation-weighted truncation norm (``weight_decay``) is what makes the
whole study affordable: at fixed chi_op it cuts the lambda = 0 error by ~4x
(S5c), because the plain Frobenius norm spends bond dimension on
high-occupation matrix elements the low-energy sector never sees.

Sections
--------
S5a  lambda = 0 anchor at the paper point: the exact-MPO infimum vs the
     Gaussian (Williamson) infimum, with the chi_op and weight_decay ladders
     that show what the residual is made of
S5b  the exact infimum vs lambda at the archived point (m = 0.5, tau0 = 0.75),
     beside the variational number re-run at the SAME cutoff (the measured
     family gap) and the free bound at the Hartree mass
S5c  breadth: a second smearing width (tau0 = 0.5) and a second mass
     (m = 0.8), so the conclusion is not a one-point statement
S5d  audits: the extremal state's physicality, the MPO variance, and the
     independent Schroedinger-picture re-measurement of every reported number

Run
---
    .venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py
    .venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py --quick

``--quick`` runs a ~2 min miniature to a scratch directory.  Rows are written
to ``data/`` incrementally as they are computed, so a partial run is still a
readable record.  Nothing here modifies any ``vacuum.*`` module.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="unit_cell_width")
warnings.filterwarnings("ignore", message="unused options")
logging.getLogger("tenpy").setLevel(logging.ERROR)

from vacuum.inequalities.qei import assert_physical, gaussian_f  # noqa: E402
import vacuum  # noqa: E402
from vacuum.interacting import (  # noqa: E402
    covariance_from_mps,
    phi4_ground_mps,
    qei_variational_mps,
    relative_change,
    smeared_energy_mps,
)
from vacuum.interacting.qei_exact import free_exact_infimum, qei_exact_mps  # noqa: E402

_ap = argparse.ArgumentParser(description="interacting-vacua: the exact QEI infimum")
_ap.add_argument("--quick", action="store_true", help="~2 min smoke configuration, scratch output")
_ap.add_argument("--out", default=None, help="output directory (default: data/)")
_ap.add_argument("--diagnostics", action="store_true",
                 help="run the S5f-S5i diagnostics (parallel subprocesses) INSTEAD of S5a-S5e")
_ap.add_argument("--workers", type=int, default=8, help="parallel workers for --diagnostics")
_ap.add_argument("--only", default=None,
                 help="comma-separated subset of diagnostic sections (f,g,h,i) to run")
_ap.add_argument("--no-new-jobs", action="store_true",
                 help="(--diagnostics) summarize the finished jobs in data/jobs/ only; never launch one")
_ap.add_argument("--s8-parts", default=None,
                 help="(--diagnostics --only s8) comma-separated subset of the S8 parts "
                      "(lam01,tau0,plane,mpo_anchor,ed) to run; default: all five")
_ap.add_argument("--plan", action="store_true",
                 help="(--diagnostics --only s8) print the S8 job table with its wall estimate and exit")
_ap.add_argument("--job", default=None, help="(worker) JSON job spec to evaluate")
_ap.add_argument("--job-out", default=None, help="(worker) where to write the result JSON")
_args, _ = _ap.parse_known_args()

HERE = Path(__file__).resolve().parent
SMOKE = os.environ.get("NOTEBOOK_SMOKE", "") == "1" or bool(_args.quick)
if _args.out:
    DATA = Path(_args.out)
elif os.environ.get("NOTEBOOK_DATA_DIR"):
    DATA = Path(os.environ["NOTEBOOK_DATA_DIR"])
elif SMOKE:
    DATA = Path(tempfile.mkdtemp(prefix="interacting-vacua-exact-quick-"))
else:
    DATA = HERE / "data"
DATA.mkdir(exist_ok=True, parents=True)

# ---- parameters (the archived S3 point, plus the exact-route knobs) --------
MASS = 0.5
QEI_L = 16            # same chain, mass, smearing and lattice as notebook.py S3
TAU0 = 0.75
NMAX = 6              # n_max = 8 is the archived variational cutoff; the exact route
                      # pays d^2 per site in the doubled operator space, so the ladder
                      # runs at 6 and the residual Fock error is measured at lambda = 0
CHI_OP = 16           # operator-MPO bond dimension (the dominant systematic)
CHI_ACC = 32
CHI_DMRG = 12
WDECAY = 0.25         # occupation-weighted truncation norm (S5c ladder measures it)
DT = 0.375            # order-4 Suzuki-Trotter; 8 steps to t = 3 = 4 tau0
ORDER = 4
PAD = 2
SVD_REAL = 1e-6
CHI_MPO = 64          # cap on the final O_f MPO bond dimension: the DMRG cost is linear
                      # in it and it was the wall-clock bottleneck (chi_mpo 32/64/123 ->
                      # 60/110/300 s of DMRG).  Its discarded weight is trunc_real, and
                      # the chi_op = 24 row of S5a re-measures E_min with the cap raised.
LAMS = [0.0, 1.0, 4.0]   # the ends and the middle of the archived sweep: the exact
                         # route costs ~d^2 per site in the doubled operator space, so
                         # the lambda grid is coarser than notebook.py's six points
BREADTH_TAU0 = 0.5
BREADTH_MASS = 0.8
BREADTH_LAMS = [0.0, 4.0]
if SMOKE:
    QEI_L, TAU0, NMAX, CHI_OP, CHI_ACC, CHI_DMRG = 8, 0.3, 4, 12, 24, 10
    DT, LAMS = 0.15, [0.0, 1.0]
    BREADTH_TAU0, BREADTH_MASS, BREADTH_LAMS = 0.2, 0.8, [1.0]

EXACT_KW = dict(dt=DT, order=ORDER, chi_op=CHI_OP, svd_min_op=1e-12, chi_acc=CHI_ACC,
                svd_min_acc=1e-12, window_pad=PAD, svd_min_real=SVD_REAL,
                chi_mpo=CHI_MPO, weight_decay=WDECAY, chi_dmrg=CHI_DMRG,
                max_sweeps=8, max_E_err=1e-10)

T_START = time.time()
TIMES = {}


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _jdefault(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not serializable: {type(o)}")


def _dump(name, obj):
    with open(DATA / name, "w") as fh:
        json.dump(obj, fh, indent=1, default=_jdefault)


def _csv(name, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(DATA / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _section(label):
    print(f"\n=== {label} ===", flush=True)


def _row(res, **extra):
    """The reported record of one exact-infimum evaluation: every knob and error."""
    op = res.op
    return {
        "E_min": res.E_min, "E_star": res.E_star, "E_ref": res.E_ref,
        "E_ref_static": res.E_ref_static, "ref_consistency": abs(res.E_ref - res.E_ref_static),
        "E_free_exact": res.E_free_exact, "E_hartree_exact": res.E_hartree_exact,
        "mu_H2": res.mu_H2, "bound_2d_flanagan": res.bound_2d,
        "ratio_exact_over_free_exact": res.E_min / res.E_free_exact,
        "ratio_exact_over_hartree_exact": res.E_min / res.E_hartree_exact,
        "ratio_exact_over_bound2d": res.E_min / res.bound_2d,
        "n_max": res.meta["n_max"], "L": res.meta["L"], "m": res.meta["m"],
        "lam": res.meta["lam"], "tau0": extra.pop("tau0", None),
        "dt": op.meta["dt"], "order": op.meta["order"], "n_steps": op.meta["n_steps"],
        "chi_op_allowed": op.meta["chi_op_max_allowed"], "chi_op": op.chi_op,
        "chi_acc": op.chi_acc, "chi_mpo": op.chi_mpo, "weight_decay": op.meta["weight_decay"],
        "window_pad": op.meta["window_pad"],
        "trunc_op_weighted": op.trunc_op, "trunc_acc": op.trunc_acc,
        "trunc_real": op.trunc_real, "norm_ratio": op.norm_ratio,
        "h_ref_drift": float(np.ptp(op.h_ref_trace)) if op.h_ref_trace.size else float("nan"),
        "dmrg_sweeps": res.dmrg["sweeps"], "dmrg_delta_E": res.dmrg["delta_E"],
        "dmrg_chi": res.dmrg["chi"], "dmrg_trunc": res.dmrg["max_trunc_err"],
        "dmrg_real": res.dmrg["real"], "mpo_variance": res.variance,
        "time_operator_s": res.dmrg["time_operator_s"], "time_dmrg_s": res.dmrg["time_dmrg_s"],
        **extra,
    }


# ---------------------------------------------------------------------------
# Diagnostics worker: one evaluation per subprocess (--job spec.json --job-out r.json)
# ---------------------------------------------------------------------------
def run_job(spec):
    """One diagnostic evaluation: the MPO route (kind='mpo') or the dense ED route
    (kind='ed'), at the spec's (L, m, lam, n_max, tau0) with EXACT_KW overridden by
    spec['kw'].  Returns the reported row (every knob and error, as in _row)."""
    t0 = time.time()
    kind = spec["kind"]
    L, m, lam, n_max, tau0 = (int(spec["L"]), float(spec["m"]), float(spec["lam"]),
                              int(spec["n_max"]), float(spec["tau0"]))
    x = L // 2
    f = gaussian_f(tau0, n_sigmas=4.0)
    extra = {k: spec[k] for k in ("tag", "axis", "section") if k in spec}
    if kind == "mpo":
        gs = phi4_ground_mps(L, m, lam, n_max=n_max, chi_max=int(spec.get("chi_gs", 16)))
        t_gs = time.time() - t0
        kw = dict(EXACT_KW, **spec.get("kw", {}))
        if kw.get("svd_dense_limit") == "inf":
            kw["svd_dense_limit"] = float("inf")
        if "dt" in spec.get("kw", {}) and kw["dt"] is None:
            kw["dt"] = DT * (tau0 / TAU0)
        res = qei_exact_mps(gs, x, f, **kw)
        row = _row(res, tau0=tau0, route="mpo", E_vac=gs.E0, gs_chi=gs.info["chi"],
                   svd_dense_limit=res.op.meta.get("svd_dense_limit"),
                   time_gs_s=t_gs, time_s=time.time() - t0, **extra)
    elif kind == "ed":
        from vacuum.interacting.qei_ed import qei_exact_dense
        res = qei_exact_dense(L, m, lam, n_max, x, f,
                              kernel=spec.get("kernels", spec.get("kernel", "analytic")),
                              dt=spec.get("dt"))
        alt = {f"E_min_{k}": v["E_min"] for k, v in res.meta["by_kernel"].items()}
        row = {"E_min": res.E_min, "E_star": res.E_star, "E_ref": res.E_ref, **alt,
               "E_star_even": res.E_star_by_sector[0], "E_star_odd": res.E_star_by_sector[1],
               "sector_star": res.sector_star, "E_vac": res.E_vac, "h_vac": res.h_vac,
               "E_free_exact": res.E_free_exact, "E_hartree_exact": res.E_hartree_exact,
               "mu_H2": res.mu_H2,
               "ratio_exact_over_free_exact": res.E_min / res.E_free_exact,
               "ratio_exact_over_hartree_exact": res.E_min / res.E_hartree_exact,
               "n_max": n_max, "L": L, "m": m, "lam": lam, "tau0": tau0, "route": "ed",
               "kernel": res.meta["kernel"], "dt": res.meta.get("dt"), "dim": res.meta["dim"],
               "dim_sectors": res.meta["dim_sectors"], "omega": res.meta["omega"],
               "time_s": time.time() - t0, **extra}
    elif kind == "ed_lanczos":
        # the matrix-free ED route (S8): Lanczos on O_f applied by Chebyshev time
        # evolution on the parity blocks -- no dense matrix, no Trotter step, no
        # operator truncation; the MPO route's own trapezoid grid at spec['dt'],
        # re-evaluated at spec['dt_fine'] in the winning sector (the quadrature check)
        from vacuum.interacting.qei_ed import qei_exact_lanczos
        res = qei_exact_lanczos(L, m, lam, n_max, x, f, dt=float(spec.get("dt", DT)),
                                dt_fine=spec.get("dt_fine"), tol=float(spec.get("tol", 1e-12)),
                                ncv=spec.get("ncv"))
        fine = res.meta.get("fine") or {}
        row = {"E_min": res.E_min, "E_star": res.E_star, "E_ref": res.E_ref,
               "E_min_trapezoid": res.E_min,
               "E_min_fine": fine.get("E_min"), "dt_fine": fine.get("dt"),
               "rel_quadrature_fine_minus_grid": ((fine["E_min"] - res.E_min) / abs(res.E_min)
                                                  if fine.get("E_min") is not None else None),
               "E_star_even": res.E_star_by_sector[0], "E_star_odd": res.E_star_by_sector[1],
               "sector_star": res.sector_star, "E_vac": res.E_vac, "h_vac": res.h_vac,
               "E_free_exact": res.E_free_exact, "E_hartree_exact": res.E_hartree_exact,
               "mu_H2": res.mu_H2,
               "ratio_exact_over_free_exact": res.E_min / res.E_free_exact,
               "ratio_exact_over_hartree_exact": res.E_min / res.E_hartree_exact,
               "n_max": n_max, "L": L, "m": m, "lam": lam, "tau0": tau0, "route": "ed_lanczos",
               "kernel": "trapezoid", "dt": res.meta["dt"], "n_steps": res.meta["n_steps"],
               "dim": res.meta["dim"], "dim_sectors": res.meta["dim_sectors"], "omega": res.meta["omega"],
               "spectral_bounds": res.meta["spectral_bounds"], "n_cheb": res.meta["n_cheb"],
               "matvecs": res.meta["matvecs"], "lanczos_residuals": res.meta["lanczos_residuals"],
               "time_sector_s": res.meta["time_sector_s"], "time_s": time.time() - t0, **extra}
    elif kind == "keff":
        # the measured-dispersion reference E_free(K_eff) at this lambda (S8): the s7a
        # pipeline of notebook_qei_reference.py verbatim (n_max 6 ground state at chi 16,
        # TEBD correlators to T = 10 at dt 0.1 and chi 32, harmonic inversion, non-negative
        # cosine fit to r_max 4, the free Williamson infimum of K_eff with the same f), so
        # the cutoff-free factor E_keff / E_H exists at every lambda of the plane (s7a
        # measured lambda 0 / 1 / 4 only)
        import math as _m
        import vacuum.interacting.qei_reference as qr
        from vacuum.interacting.phi4 import hartree_mass_squared
        gs = phi4_ground_mps(L, m, lam, n_max=n_max, chi_max=int(spec.get("chi_gs", 16)))
        d = qr.interacting_dispersion_tebd(gs, T=float(spec.get("T", 10.0)), dt=float(spec.get("disp_dt", 0.1)),
                                           order=4, chi_max=int(spec.get("disp_chi", 32)))
        gap, ginfo = qr.excitation_gap_dmrg(gs, chi_max=16)
        mu2, J, info = qr.fit_dispersion_couplings(d.k, np.asarray(d.omega) ** 2, r_max=int(spec.get("r_max", 4)),
                                                   weights=d.weight)
        kres, _op = qr.free_infimum_keff(L, x, f, mu2, J)
        E_free = free_exact_infimum(L, x, f, m)
        mu_H2 = float(hartree_mass_squared(L, m, lam))
        E_H = free_exact_infimum(L, x, f, _m.sqrt(mu_H2))
        row = {"route": "keff", "L": L, "m": m, "lam": lam, "n_max": n_max, "tau0": tau0,
               "E_keff": float(kres.e_min), "E_hartree_exact": float(E_H), "E_free_exact": float(E_free),
               "mu_H2": mu_H2, "E_keff_over_E_hartree": float(kres.e_min) / float(E_H),
               "mu2_eff": float(mu2), "J_eff": [float(v) for v in J],
               "keff_misfit_max": info["misfit_max"], "keff_misfit_rms": info["misfit_rms"],
               "k": d.k, "omega": d.omega, "omega_hartree": d.omega_hartree, "omega_free": d.omega_free,
               "gamma": d.gamma, "weight": d.weight,
               "weight_min": float(np.min(d.weight)), "gamma_over_gap_max": float(np.max(np.abs(d.gamma)) / d.omega[0]),
               "gap_dmrg": float(gap), "gap_check_dev": float(d.omega[0] - gap),
               "max_dev_from_hartree": float(np.max(np.abs(d.omega - d.omega_hartree))),
               "max_dev_from_free": float(np.max(np.abs(d.omega - d.omega_free))),
               "disp_T": d.meta["T"], "disp_dt": d.meta["dt"], "disp_chi_max": d.meta["chi_max"],
               "disp_chi": d.chi, "disp_trunc_err": d.trunc_err, "E0": d.E0,
               "time_s": time.time() - t0, **extra}
    else:
        raise ValueError(f"unknown job kind {kind!r}")
    row["threads"] = os.environ.get("OMP_NUM_THREADS")
    return row


if _args.job:
    with open(_args.job) as fh:
        _spec = json.load(fh)
    _result = run_job(_spec)
    with open(_args.job_out, "w") as fh:
        json.dump({"spec": _spec, "row": _result}, fh, indent=1, default=_jdefault)
    raise SystemExit(0)


def run_jobs(specs, workers, label):
    """Run the specs as subprocesses of this file (--job), `workers` at a time, each
    pinned to spec['threads'] BLAS threads; longest-first ordering; every result is
    written to DATA/jobs/<label>_<tag>.json as it lands and reused on a re-run (a
    finished job is never recomputed).  Returns the rows in the specs' order."""
    import concurrent.futures as cf
    import sys
    jobs_dir = DATA / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    # highest 'priority' first (S8 uses it to land the verdict-relevant pairs before the
    # multi-day corners of the plane), longest-first within a priority (makespan)
    order = sorted(range(len(specs)),
                   key=lambda i: (-float(specs[i].get("priority", 0.0)), -specs[i].get("cost", 1.0)))

    def _one(i):
        spec = specs[i]
        key = str(spec.get("tag", i)).replace("/", "_")
        sp_path = jobs_dir / f"{label}_{key}.spec.json"
        out_path = jobs_dir / f"{label}_{key}.json"
        if out_path.exists() and not spec.get("force", False):
            try:
                with open(out_path) as fh:
                    return i, json.load(fh)["row"], "cached"
            except (OSError, ValueError, KeyError) as exc:
                # mid-write by the other runner, or truncated by a crash: a summarizing run
                # skips it; a computing run falls through and recomputes it
                if _args.no_new_jobs:
                    return i, {"error": f"result file unreadable ({exc})", "tag": spec.get("tag"),
                               "time_s": 0.0}, "skipped"
        if _args.no_new_jobs:
            return i, {"error": "not run (--no-new-jobs)", "tag": spec.get("tag"), "time_s": 0.0}, "skipped"
        with open(sp_path, "w") as fh:
            json.dump(spec, fh, default=_jdefault)
        env = dict(os.environ)
        nt = str(int(spec.get("threads", 2)))
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[k] = nt
        env["NOTEBOOK_DATA_DIR"] = str(DATA)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--job", str(sp_path),
               "--job-out", str(out_path)]
        tj = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0 and int(spec.get("retries", 1)) > 0:
            # one retry after a pause: a transient failure (the shared tree mid-edit, a
            # memory spike from another agent's run) must not lose a multi-hour slot
            time.sleep(60)
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            return i, {"error": proc.stderr[-2000:], "tag": spec.get("tag"),
                       "time_s": time.time() - tj}, "failed"
        with open(out_path) as fh:
            return i, json.load(fh)["row"], "done"

    rows = [None] * len(specs)
    with cf.ThreadPoolExecutor(max_workers=int(workers)) as ex:
        for i, row, status in ex.map(_one, order):
            rows[i] = row
            tag = specs[i].get("tag", i)
            if "error" in row:
                print(f"  [{label}] {tag}: {'SKIPPED' if status == 'skipped' else 'FAILED'} {row['error'][:200]}", flush=True)
            else:
                # a keff row carries E_keff / E_keff_over_E_hartree instead of E_min / the ratio
                e_ = row.get("E_min", row.get("E_keff", float("nan")))
                r_ = row.get("ratio_exact_over_hartree_exact", row.get("E_keff_over_E_hartree", float("nan")))
                print(f"  [{label}] {tag}: E_min {e_:.6e} exact/hartree {r_:.5f} "
                      f"[{row['time_s']:.0f}s, {status}]", flush=True)
    return rows


# ---------------------------------------------------------------------------
# S5f-S5i  DIAGNOSTICS (--diagnostics): an outside reviewer's plan, in its order
# ---------------------------------------------------------------------------
# The reviewer's premise: "Fock truncation is itself a suppression mechanism --
# the extremal state of a smeared energy is far more excited than the ground
# state, so the n_max = 8 convergence verified on the lambda = 0 ground state does
# not transfer, and the truncation raises the infimum in exactly the direction of
# the residual."  Four things are measured, each in its own section and each as
# a set of parallel subprocesses of this file (--job), so that a partial run is a
# readable record (data/jobs/ keeps every evaluation):
#
#   S5f  the n_max ladder ON THE INFIMUM at lambda = 4 (6 ... 12, archived knobs)
#   S5g  the lambda = 0 calibration band decomposed one knob at a time (Fock
#        cutoff, operator truncation chi_op x weight x exact-vs-sketched SVD, the
#        final MPO cap, the accumulator, Trotter dt/order, DMRG chi/sweeps, the
#        light-cone window), the same sensitivities at lambda = 4, and combined
#        "best" settings to see how far the band can be driven
#   S5h  a 7-point lambda sweep at the best-converged settings, the residual
#        1 - exact/Hartree with its band, the small-lambda exponent, and the
#        O(lambda) anchor: the exact Hellmann-Feynman slope of the infimum in the
#        free extremal state (qei_exact.first_order_infimum_slope)
#   S5i  the second implementation (anomaly protocol): the dense-ED infimum of
#        vacuum.interacting.qei_ed at L = 6-8, checked against the MPO-DMRG route
#        at the same (L, n_max), plus the ED n_max ladder where Rayleigh-Ritz can
#        be tested with no MPO truncation in the way
#
# Run:  notebook_qei_exact.py --diagnostics [--workers 8] [--only f,g,h,i]
if _args.diagnostics:
    import sys
    from vacuum.interacting.qei_exact import first_order_infimum_slope

    ONLY = set((_args.only or "f,g,h,i").split(","))
    W = int(_args.workers)
    DIAG_T0 = time.time()
    f_main = gaussian_f(TAU0, n_sigmas=4.0)
    E_gauss = free_exact_infimum(QEI_L, QEI_L // 2, f_main, MASS)
    LAMS7 = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
    # the settings S5h uses; chosen from S5g (see s5g_calibration.json 'best')
    BEST_NMAX = 8
    BEST_KW = dict(chi_op=32, chi_acc=64, chi_mpo=128, weight_decay=0.5, chi_dmrg=16)

    def _mpo(lam, n_max, tag, section, axis=None, L=QEI_L, threads=2, chi_gs=16, **kw):
        chi = kw.get("chi_op", CHI_OP)
        dtf = DT / kw["dt"] if kw.get("dt") else 1.0
        cost = (n_max + 1) ** 6 * (chi / CHI_OP) ** 2 * dtf * (L / QEI_L)
        if kw.get("svd_dense_limit") == "inf":
            cost *= 3.0
        spec = {"kind": "mpo", "L": int(L), "m": MASS, "tau0": TAU0, "lam": float(lam),
                "n_max": int(n_max), "tag": tag, "section": section, "threads": threads,
                "chi_gs": int(chi_gs), "cost": cost, "kw": kw}
        if axis:
            spec["axis"] = axis
        return spec

    def _ed(lam, n_max, L, tag, section, threads=6, kernels=("analytic", "trapezoid")):
        d = n_max + 1
        return {"kind": "ed", "L": int(L), "m": MASS, "tau0": TAU0, "lam": float(lam),
                "n_max": int(n_max), "tag": tag, "section": section, "threads": threads,
                "cost": (d ** L / 2.0) ** 3 / 1e6, "kernels": list(kernels), "dt": DT}

    def _ok(rows):
        return [r for r in rows if r is not None and "error" not in r]

    _section("S5f-S5i provenance")
    try:
        build = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception as exc:  # pragma: no cover
        build = f"git unavailable: {exc}"
    import tenpy  # noqa: E402
    diag_build = {"_source": "notebook_qei_exact.py::S5f-S5i", "date_utc": _now(),
                  "build_git_head": build, "python": platform.python_version(),
                  "platform": platform.platform(), "numpy": np.__version__,
                  "tenpy": tenpy.__version__, "vacuum_version": vacuum.__version__,
                  "workers": W, "sections": sorted(ONLY),
                  "archived_knobs": {k: v for k, v in EXACT_KW.items()},
                  "best_settings": {"n_max": BEST_NMAX, **BEST_KW},
                  "lambda_grid": LAMS7, "E_gaussian_exact": E_gauss}
    # an --only s8 run must not overwrite the archived S5f-S5i records; and a partial-parts
    # S8 runner (--s8-parts) must not clobber the full plan / build-info / summary records
    # either (the launch's pool-B runner rewrote s8_plan.json as its 16-job 'ed' plan): its
    # records carry the parts in their names (s8_plan_ed.json, s8_diagnostics_build_info_ed.json)
    S8_ONLY = ONLY <= {"s8"}
    # S8 TRIAGE 2026-09-12: "plane" is RETIRED (see MANIFEST "S8 triage") and is no
    # longer in the default set -- it is still buildable with an explicit
    # --s8-parts plane, but nothing quotes it.  "plane2" replaces it and "wt" is the
    # ED-anchored weight_decay ladder that decided the replacement.
    S8_ALL_PARTS = ("lam01", "tau0", "plane2", "wt", "mpo_anchor", "ed", "keff")
    S8_RETIRED_PARTS = ("plane",)
    _s8_parts_arg = tuple(sorted(set((_args.s8_parts or ",".join(S8_ALL_PARTS)).split(","))))
    S8_SUFFIX = "" if set(_s8_parts_arg) >= set(S8_ALL_PARTS) else "_" + "_".join(_s8_parts_arg)
    _dump(f"s8_diagnostics_build_info{S8_SUFFIX}.json" if S8_ONLY else "s5f_diagnostics_build_info.json", diag_build)
    print(json.dumps({k: diag_build[k] for k in ("workers", "sections", "best_settings")}), flush=True)

    # ---------------- S5f: the n_max ladder on the infimum at lambda = 4 --------
    if "f" in ONLY:
        _section("S5f n_max ladder ON THE INFIMUM at lambda = 4 (archived knobs: chi_op 16, decay 0.25)")
        t0 = time.time()
        specs = [_mpo(4.0, n, f"lam4_nmax{n}", "S5f", axis="fock", threads=(4 if n >= 10 else 2))
                 for n in range(6, 13)]
        rows = _ok(run_jobs(specs, W, "s5f"))
        rows.sort(key=lambda r: r["n_max"])
        _csv("s5f_nmax_ladder_lam4.csv", rows)
        emins = [r["E_min"] for r in rows]
        ratios = [r["ratio_exact_over_hartree_exact"] for r in rows]
        ladder_f = {"n_max": [r["n_max"] for r in rows], "E_min": emins,
                    "ratio_hartree": ratios,
                    "steps_E_min": np.diff(emins).tolist(), "steps_ratio": np.diff(ratios).tolist(),
                    "wall_s": [r["time_s"] for r in rows],
                    "trunc_op_weighted": [r["trunc_op_weighted"] for r in rows],
                    "norm_ratio": [r["norm_ratio"] for r in rows],
                    "ref_consistency": [r["ref_consistency"] for r in rows],
                    "E_hartree_exact": rows[0]["E_hartree_exact"] if rows else None,
                    # Rayleigh-Ritz on the infimum would read: E_min non-increasing in n_max
                    "monotone_E_min_nonincreasing": bool(np.all(np.diff(emins) <= 1e-12)),
                    "max_up_step_E_min": float(max([0.0] + np.diff(emins).tolist())),
                    "lower_bound_ratio_hartree": (max(ratios) if ratios else None),
                    "last_ratio_hartree": (ratios[-1] if ratios else None)}
        _dump("s5f_nmax_ladder_lam4.json", {"_source": "notebook_qei_exact.py::S5f", **ladder_f})
        print("  n_max  E_min          exact/Hartree  step        wall s")
        for i, r in enumerate(rows):
            st = "" if i == 0 else f"{ratios[i] - ratios[i - 1]:+.4f}"
            print(f"  {r['n_max']:5d}  {r['E_min']:.6e}  {ratios[i]:.4f}      {st:>8}  {r['time_s']:6.0f}")
        print(f"  E_min non-increasing in n_max (Rayleigh-Ritz reading): "
              f"{ladder_f['monotone_E_min_nonincreasing']}", flush=True)
        TIMES["S5f"] = time.time() - t0

    # ---------------- S5g: the calibration band, decomposed ---------------------
    if "g" in ONLY:
        _section("S5g the lambda = 0 calibration band decomposed one knob at a time")
        t0 = time.time()
        grid = [_mpo(0.0, NMAX, "lam0_base", "S5g", axis="base"),
                _mpo(4.0, NMAX, "lam4_base", "S5g", axis="base")]
        for n in range(7, 13):
            grid.append(_mpo(0.0, n, f"lam0_fock_nmax{n}", "S5g", axis="fock",
                             threads=(4 if n >= 10 else 2)))
        for chi in (16, 32, 64):
            for wd in (0.25, 0.5, 1.0):
                if chi == 16 and wd == 0.25:
                    continue
                grid.append(_mpo(0.0, NMAX, f"lam0_chiop{chi}_wd{wd:g}", "S5g", axis="mpo",
                                 chi_op=chi, chi_acc=2 * chi, chi_mpo=4 * chi, weight_decay=wd))
        for chi, wd in ((16, 0.25), (16, 0.5), (32, 0.5), (32, 1.0)):
            grid.append(_mpo(0.0, NMAX, f"lam0_exactsvd_chiop{chi}_wd{wd:g}", "S5g", axis="svd",
                             chi_op=chi, chi_acc=2 * chi, chi_mpo=4 * chi, weight_decay=wd,
                             svd_dense_limit="inf"))
        grid += [_mpo(0.0, NMAX, "lam0_chimpo_uncapped", "S5g", axis="cap", chi_mpo=None,
                      svd_min_real=1e-8),
                 _mpo(0.0, NMAX, "lam0_chiacc64", "S5g", axis="acc", chi_acc=64),
                 _mpo(0.0, NMAX, "lam0_dt0.1875", "S5g", axis="trotter", dt=0.1875),
                 _mpo(0.0, NMAX, "lam0_dt0.125", "S5g", axis="trotter", dt=0.125),
                 _mpo(0.0, NMAX, "lam0_order2", "S5g", axis="trotter", order=2),
                 _mpo(0.0, NMAX, "lam0_chidmrg24", "S5g", axis="dmrg", chi_dmrg=24),
                 _mpo(0.0, NMAX, "lam0_sweeps16", "S5g", axis="dmrg", max_sweeps=16, max_E_err=1e-12),
                 _mpo(0.0, NMAX, "lam0_pad4", "S5g", axis="window", window_pad=4),
                 _mpo(0.0, NMAX, "lam0_chigs32", "S5g", axis="gs", chi_gs=32)]
        # the same sensitivities at lambda = 4 (no exact reference: the shift IS the band there)
        for chi, wd in ((32, 0.5), (64, 0.5), (32, 1.0), (64, 1.0)):
            grid.append(_mpo(4.0, NMAX, f"lam4_chiop{chi}_wd{wd:g}", "S5g", axis="mpo",
                             chi_op=chi, chi_acc=2 * chi, chi_mpo=4 * chi, weight_decay=wd))
        grid += [_mpo(4.0, NMAX, "lam4_exactsvd_chiop32_wd0.5", "S5g", axis="svd", chi_op=32,
                      chi_acc=64, chi_mpo=128, weight_decay=0.5, svd_dense_limit="inf"),
                 _mpo(4.0, NMAX, "lam4_dt0.1875", "S5g", axis="trotter", dt=0.1875),
                 _mpo(4.0, NMAX, "lam4_chidmrg24", "S5g", axis="dmrg", chi_dmrg=24),
                 _mpo(4.0, NMAX, "lam4_pad4", "S5g", axis="window", window_pad=4)]
        # combined candidates at n_max = 8: the weight bias and chi_op at the sweep's
        # cutoff (the sweep's own lambda = 0 point is the (8, 32, 0.5) row)
        for n, chi, wd in ((8, 32, 1.0), (8, 64, 0.5)):
            grid.append(_mpo(0.0, n, f"lam0_best_nmax{n}_chiop{chi}_wd{wd:g}", "S5g", axis="combined",
                             threads=4, chi_op=chi, chi_acc=2 * chi, chi_mpo=4 * chi,
                             weight_decay=wd, chi_dmrg=16))
        rows = run_jobs(grid, W, "s5g")
        base0 = next((r for r in rows if r and r.get("tag") == "lam0_base" and "error" not in r), None)
        base4 = next((r for r in rows if r and r.get("tag") == "lam4_base" and "error" not in r), None)
        out_rows = []
        for r in _ok(rows):
            base = base0 if r["lam"] == 0.0 else base4
            r = dict(r)
            if r["lam"] == 0.0:
                r["dev_vs_gaussian"] = r["E_min"] - E_gauss
                r["rel_dev_vs_gaussian"] = r["dev_vs_gaussian"] / abs(E_gauss)
            if base is not None:
                r["shift_vs_base"] = r["E_min"] - base["E_min"]
                r["rel_shift_vs_base"] = r["shift_vs_base"] / abs(base["E_min"])
            out_rows.append(r)
        _csv("s5g_calibration.csv", out_rows)
        per_axis = {}
        for r in out_rows:
            key = f"lam={r['lam']:g}/{r.get('axis')}"
            per_axis.setdefault(key, []).append(
                {"tag": r["tag"], "E_min": r["E_min"], "rel_dev_vs_gaussian": r.get("rel_dev_vs_gaussian"),
                 "rel_shift_vs_base": r.get("rel_shift_vs_base"), "time_s": r["time_s"],
                 "trunc_op_weighted": r["trunc_op_weighted"], "chi_mpo": r["chi_mpo"]})
        lam0 = [r for r in out_rows if r["lam"] == 0.0]
        best = min(lam0, key=lambda r: abs(r["rel_dev_vs_gaussian"])) if lam0 else None
        calib = {"_source": "notebook_qei_exact.py::S5g", "E_gaussian_exact": E_gauss,
                 "base_lam0": (None if base0 is None else
                               {"E_min": base0["E_min"], "rel_dev_vs_gaussian": (base0["E_min"] - E_gauss) / abs(E_gauss)}),
                 "per_axis": per_axis,
                 "best_lam0": (None if best is None else
                               {"tag": best["tag"], "rel_dev_vs_gaussian": best["rel_dev_vs_gaussian"],
                                "n_max": best["n_max"], "chi_op": best["chi_op_allowed"],
                                "weight_decay": best["weight_decay"], "svd_dense_limit": best.get("svd_dense_limit"),
                                "time_s": best["time_s"]})}
        _dump("s5g_calibration.json", calib)
        print("  lambda = 0: rel dev vs Gaussian per row (negative = MPO truncation manufacturing states)")
        for r in sorted(lam0, key=lambda r: (r.get("axis") or "", r["tag"])):
            print(f"    {r['tag']:<36} {r['rel_dev_vs_gaussian']:+.2e}  (shift vs base {r.get('rel_shift_vs_base', float('nan')):+.2e}) "
                  f"trunc(w) {r['trunc_op_weighted']:.1e} [{r['time_s']:.0f}s]")
        print("  lambda = 4: shift vs base per row")
        for r in sorted([r for r in out_rows if r["lam"] == 4.0], key=lambda r: (r.get("axis") or "", r["tag"])):
            print(f"    {r['tag']:<36} exact/H {r['ratio_exact_over_hartree_exact']:.4f} shift vs base "
                  f"{r.get('rel_shift_vs_base', float('nan')):+.2e} [{r['time_s']:.0f}s]")
        if best:
            print(f"  best lambda = 0 band reached: {best['rel_dev_vs_gaussian']:+.2e} at {best['tag']} "
                  f"({best['time_s']:.0f} s)", flush=True)
        TIMES["S5g"] = time.time() - t0

    # ---------------- S5h: the 7-point sweep and the O(lambda) anchor -----------
    if "h" in ONLY:
        _section(f"S5h 7-point lambda sweep at the best-converged settings "
                 f"(n_max {BEST_NMAX}, {BEST_KW}) and the O(lambda) anchor")
        t0 = time.time()
        specs = [_mpo(lam, BEST_NMAX, f"sweep_lam{lam:g}", "S5h", axis="sweep", threads=4, **BEST_KW)
                 for lam in LAMS7]
        rows = _ok(run_jobs(specs, W, "s5h"))
        rows.sort(key=lambda r: r["lam"])
        ctrl = next((r for r in rows if r["lam"] == 0.0), None)
        band = abs(ctrl["ratio_exact_over_hartree_exact"] - 1.0) if ctrl else float("nan")
        for r in rows:
            r["residual"] = 1.0 - r["ratio_exact_over_hartree_exact"]
            r["residual_ctrl_normalized"] = (1.0 - r["ratio_exact_over_hartree_exact"]
                                             / ctrl["ratio_exact_over_hartree_exact"]) if ctrl else None
            r["band"] = band
        _csv("s5h_lambda_sweep.csv", rows)
        # the O(lambda) anchor: exact slope of the infimum at lambda = 0
        slope = first_order_infimum_slope(QEI_L, QEI_L // 2, f_main, MASS)
        # small-lambda exponent from the control-normalized residual on lambda <= 1
        small = [r for r in rows if 0.0 < r["lam"] <= 1.0 and r["residual_ctrl_normalized"] is not None]
        fit = {}
        if len(small) >= 2:
            x = np.log([r["lam"] for r in small])
            y = np.array([r["residual_ctrl_normalized"] for r in small])
            if np.all(y > 0):
                pfit = np.polyfit(x, np.log(y), 1)
                fit["exponent"] = float(pfit[0])
                fit["prefactor"] = float(np.exp(pfit[1]))
                # the band moves each point by +- band: the exponent's reach across the
                # extreme placements of the two end points
                lo, hi = small[0], small[-1]
                ends = []
                for s1 in (-1, 1):
                    for s2 in (-1, 1):
                        r1 = lo["residual_ctrl_normalized"] + s1 * band
                        r2 = hi["residual_ctrl_normalized"] + s2 * band
                        if r1 > 0 and r2 > 0:
                            ends.append(float(np.log(r2 / r1) / np.log(hi["lam"] / lo["lam"])))
                fit["exponent_range_within_band"] = [min(ends), max(ends)] if ends else None
            fit["points"] = [(r["lam"], r["residual_ctrl_normalized"]) for r in small]
        pt = {"dR_dlambda_at_0": slope["dR"], "dE_min": slope["dE_min"], "dE_hartree": slope["dE_hartree"],
              "dE_star_quartic": slope["dE_star_quartic"], "dE_star_dynamic": slope["dE_star_dynamic"],
              "dE_ref": slope["dE_ref"], "E_free": slope["E_free"],
              "predicted_residual_O1": {f"lam={r['lam']:g}": slope["dR"] * r["lam"] for r in rows},
              "measured_minus_predicted": {f"lam={r['lam']:g}": (r["residual_ctrl_normalized"] - slope["dR"] * r["lam"])
                                           for r in rows if r["residual_ctrl_normalized"] is not None}}
        sweep = {"_source": "notebook_qei_exact.py::S5h", "settings": {"n_max": BEST_NMAX, **BEST_KW},
                 "band_lam0": band, "rows": [{k: r[k] for k in (
                     "lam", "E_min", "E_hartree_exact", "E_free_exact", "ratio_exact_over_hartree_exact",
                     "ratio_exact_over_free_exact", "residual", "residual_ctrl_normalized", "trunc_op_weighted",
                     "ref_consistency", "time_s")} for r in rows],
                 "small_lambda_fit": fit, "pt_anchor": pt}
        _dump("s5h_lambda_sweep.json", sweep)
        print(f"  band (|control - 1| at lambda = 0): {band:.2e}")
        print("  lam    E_min          exact/H   residual   ctrl-norm   O(lam) PT    time")
        for r in rows:
            print(f"  {r['lam']:<5g} {r['E_min']:.6e}  {r['ratio_exact_over_hartree_exact']:.4f}   "
                  f"{r['residual']:+.4f}    {r['residual_ctrl_normalized']:+.4f}     "
                  f"{slope['dR'] * r['lam']:+.4f}   {r['time_s']:5.0f}s")
        print(f"  O(lambda) slope of the residual (exact, free extremal state): dR/dlam = {slope['dR']:+.4e}"
              f"  [dE_min {slope['dE_min']:+.4e} = quartic {slope['dE_star_quartic']:+.4e} + dynamic "
              f"{slope['dE_star_dynamic']:+.4e} - ref {slope['dE_ref']:+.4e}; dE_hartree {slope['dE_hartree']:+.4e}]")
        if fit.get("exponent") is not None:
            print(f"  small-lambda exponent (ctrl-normalized residual, lambda <= 1): {fit['exponent']:.2f}, "
                  f"range within band {fit.get('exponent_range_within_band')}", flush=True)
        TIMES["S5h"] = time.time() - t0

    # ---------------- S5i: the second implementation (dense ED) -----------------
    if "i" in ONLY:
        _section("S5i second implementation: dense-ED infimum at L = 6-8 vs the MPO-DMRG route")
        t0 = time.time()
        ed = []
        for lam in (0.0, 4.0):
            for n in (1, 2, 3, 4, 5):
                ed.append(_ed(lam, n, 6, f"ed_L6_nmax{n}_lam{lam:g}", "S5i"))
            for n in (2, 3):
                ed.append(_ed(lam, n, 7, f"ed_L7_nmax{n}_lam{lam:g}", "S5i"))
            ed.append(_ed(lam, 2, 8, f"ed_L8_nmax2_lam{lam:g}", "S5i"))
        ed.append(_ed(1.0, 4, 6, "ed_L6_nmax4_lam1", "S5i"))
        # the small-lambda anchor with no MPO in the way: the ED residual at L = 6
        # against the exact O(lambda) slope at L = 6 (first_order_infimum_slope)
        for lam in (0.1, 0.25, 0.5):
            for n in (3, 4):
                ed.append(_ed(lam, n, 6, f"ed_L6_nmax{n}_lam{lam:g}", "S5i"))
        # L = 8 at n_max = 3 (32 768 states per parity sector) was started and
        # abandoned: two single-threaded eigh's of ~35 min each on an idle core did
        # not fit the diagnostics budget on a shared machine; L = 8 is at n_max = 2
        GENEROUS = dict(chi_op=64, chi_acc=128, chi_mpo=None, svd_min_real=1e-10, weight_decay=1.0,
                        svd_dense_limit="inf", chi_dmrg=32, window_pad=None, max_sweeps=16,
                        max_E_err=1e-12)
        mp = []
        for lam in (0.0, 4.0):
            for n in (2, 3, 4, 5):
                mp.append(_mpo(lam, n, f"mpo_L6_nmax{n}_lam{lam:g}_archived", "S5i", axis="archived", L=6))
                mp.append(_mpo(lam, n, f"mpo_L6_nmax{n}_lam{lam:g}_generous", "S5i", axis="generous", L=6,
                               threads=4, **GENEROUS))
            mp.append(_mpo(lam, 3, f"mpo_L7_nmax3_lam{lam:g}_archived", "S5i", axis="archived", L=7))
            mp.append(_mpo(lam, 3, f"mpo_L7_nmax3_lam{lam:g}_generous", "S5i", axis="generous", L=7,
                           threads=4, **GENEROUS))
        for lam in (0.0, 4.0):
            mp.append(_mpo(lam, 2, f"mpo_L8_nmax2_lam{lam:g}_archived", "S5i", axis="archived", L=8))
            mp.append(_mpo(lam, 2, f"mpo_L8_nmax2_lam{lam:g}_generous", "S5i", axis="generous", L=8, threads=4, **GENEROUS))
        mp.append(_mpo(1.0, 4, "mpo_L6_nmax4_lam1_generous", "S5i", axis="generous", L=6, threads=4, **GENEROUS))
        rows = _ok(run_jobs(ed + mp, W, "s5i"))
        ed_rows = [r for r in rows if r["route"] == "ed"]
        mp_rows = [r for r in rows if r["route"] == "mpo"]
        _csv("s5i_ed_control.csv", ed_rows + mp_rows,
             fields=sorted({k for r in ed_rows + mp_rows for k in r.keys()}))
        pairs = []
        for r in mp_rows:
            e = next((q for q in ed_rows if (q["L"], q["n_max"], q["lam"]) == (r["L"], r["n_max"], r["lam"])), None)
            if e is None:
                continue
            e_trap = e.get("E_min_trapezoid", e["E_min"])
            pairs.append({"L": r["L"], "n_max": r["n_max"], "lam": r["lam"], "knobs": r.get("axis"),
                          "E_min_mpo": r["E_min"], "E_min_ed": e["E_min"], "E_min_ed_trapezoid": e_trap,
                          "dev_mpo_vs_ed": r["E_min"] - e["E_min"],
                          "rel_dev_mpo_vs_ed": (r["E_min"] - e["E_min"]) / abs(e["E_min"]),
                          "rel_dev_mpo_vs_ed_same_quadrature": (r["E_min"] - e_trap) / abs(e_trap),
                          "rel_quadrature_only": (e_trap - e["E_min"]) / abs(e["E_min"]),
                          "ratio_hartree_mpo": r["ratio_exact_over_hartree_exact"],
                          "ratio_hartree_ed": e["ratio_exact_over_hartree_exact"],
                          "trunc_op_weighted": r["trunc_op_weighted"], "chi_mpo": r["chi_mpo"],
                          "time_mpo_s": r["time_s"], "time_ed_s": e["time_s"]})
        ladders = {}
        for L_ in (6, 7, 8):
            for lam in (0.0, 0.1, 0.25, 0.5, 1.0, 4.0):
                seq = sorted([q for q in ed_rows if q["L"] == L_ and q["lam"] == lam], key=lambda q: q["n_max"])
                if len(seq) >= 2:
                    em = [q["E_min"] for q in seq]
                    ladders[f"L={L_}/lam={lam:g}"] = {
                        "n_max": [q["n_max"] for q in seq], "E_min": em,
                        "ratio_hartree": [q["ratio_exact_over_hartree_exact"] for q in seq],
                        "steps_E_min": np.diff(em).tolist(),
                        "monotone_E_min_nonincreasing": bool(np.all(np.diff(em) <= 1e-12)),
                        "sector_star": [q["sector_star"] for q in seq],
                        "dev_vs_gaussian_rel": ([(q["E_min"] - q["E_free_exact"]) / abs(q["E_free_exact"]) for q in seq]
                                                if lam == 0.0 else None),
                        "wall_s": [q["time_s"] for q in seq]}
        # the ED small-lambda anchor: control-normalized residual at L = 6 vs the exact slope
        slope6 = first_order_infimum_slope(6, 3, f_main, MASS)
        anchor6 = {}
        for n in (3, 4, 5):
            r0 = next((q for q in ed_rows if (q["L"], q["n_max"], q["lam"]) == (6, n, 0.0)), None)
            if r0 is None:
                continue
            for lam in (0.1, 0.25, 0.5, 1.0, 4.0):
                q = next((q for q in ed_rows if (q["L"], q["n_max"], q["lam"]) == (6, n, lam)), None)
                if q is None:
                    continue
                anchor6[f"n_max={n}/lam={lam:g}"] = {
                    "residual": 1.0 - q["ratio_exact_over_hartree_exact"],
                    "residual_ctrl_normalized": 1.0 - q["ratio_exact_over_hartree_exact"] / r0["ratio_exact_over_hartree_exact"],
                    "predicted_O1": slope6["dR"] * lam}
        control = {"_source": "notebook_qei_exact.py::S5i", "pairs": pairs, "ed_ladders": ladders,
                   "slope_L6": {k: slope6[k] for k in ("dR", "dE_min", "dE_hartree", "E_free")},
                   "ed_small_lambda_anchor_L6": anchor6,
                   "worst_rel_dev_generous": (max((abs(p["rel_dev_mpo_vs_ed_same_quadrature"]) for p in pairs
                                                  if p["knobs"] == "generous"), default=None)),
                   "worst_rel_dev_archived": (max((abs(p["rel_dev_mpo_vs_ed_same_quadrature"]) for p in pairs
                                                  if p["knobs"] == "archived"), default=None))}
        _dump("s5i_ed_control.json", control)
        print("  ED ladders (E_min, non-increasing in n_max?):")
        for k, v in ladders.items():
            print(f"    {k}: n_max {v['n_max']} E_min {['%.6e' % e for e in v['E_min']]} "
                  f"ratio_H {['%.4f' % x for x in v['ratio_hartree']]} monotone {v['monotone_E_min_nonincreasing']} "
                  f"sectors {v['sector_star']} wall {['%.0f' % w for w in v['wall_s']]}")
        if anchor6:
            print(f"  ED small-lambda anchor at L = 6 (exact dR/dlam = {slope6['dR']:+.4f}):")
            for k, v in anchor6.items():
                print(f"    {k}: residual {v['residual']:+.4f}, ctrl-normalized {v['residual_ctrl_normalized']:+.4f}, "
                      f"O(lambda) prediction {v['predicted_O1']:+.4f}")
        print("  MPO vs ED at the same (L, n_max, lambda):")
        for p_ in sorted(pairs, key=lambda p: (p["L"], p["lam"], p["n_max"], p["knobs"])):
            print(f"    L={p_['L']} n_max={p_['n_max']} lam={p_['lam']:g} [{p_['knobs']:8s}] E_mpo {p_['E_min_mpo']:.6e} "
                  f"E_ed {p_['E_min_ed']:.6e}  rel dev {p_['rel_dev_mpo_vs_ed']:+.2e} "
                  f"(same quadrature {p_['rel_dev_mpo_vs_ed_same_quadrature']:+.2e}; quadrature-only "
                  f"{p_['rel_quadrature_only']:+.1e}) trunc(w) {p_['trunc_op_weighted']:.1e}", flush=True)
        TIMES["S5i"] = time.time() - t0

    # ---------------- S8: the lambda >= 2 magnitude, the tau0 >= 1 rows, lambda = 0.1,
    #                  and the ED anchor beyond n_max 4 (MANIFEST items 4-5, Status) ----
    # Five parts, one job pool (data/jobs/s8_<tag>.json, idempotent), ordered by
    # 'priority' so that the pairs a verdict needs land first:
    #   lam01       lambda = 0.1 at the sweep's settings (the 'next decisive cheap point')
    #   tau0        tau0 = 1.0, 1.5 at lambda = 4 with lambda = 0 controls, chi_op 32 and 48
    #               (n_max 8, weight 0.5, sketched SVD, dt 0.375 as in s7d), and the a = 1/2
    #               point realized as (L 32, m 0.25, lambda 1, tau0 1.5) with its control
    #               (phi4.py has no lattice-spacing knob: unit spacing; the rescaling is s7e's)
    #   plane       the (n_max, chi_op) plane at lambda 0 / 4 / 2 (and 3 at two settings):
    #               n_max {8, 10, 12} x chi_op {32, 64}, weight 0.5, EXACT SVD, paper L/tau0/m
    #   mpo_anchor  the MPO route at L = 5-6 where the ED route is exact: the sweep's knobs,
    #               the plane's knobs, and the generous knobs, all at the same (L, n_max, lam)
    #   ed          the matrix-free ED infimum (qei_ed.qei_exact_lanczos) at L = 6 n_max 3-6
    #               and L = 5 n_max 6-8, lambda 0 and 4 (n_max 3-4 double as the validation
    #               against the archived dense rows), plus the dense route at L = 6 n_max 5
    # Wall model (2 BLAS threads, idle machine; logged per job as est_wall_s, calibrated on
    # 2026-09-11: the s5h (8, 32, sketched) point ~ 900 s idle; a complex SVD of a 2592^2
    # block 12.3 s; 11 Trotter layers per order-4 step; ~3/4 of the bond updates at full
    # chi_op): base = 900 s ((n_max+1)/9)^4 (chi_op/32)^1.5 (n_steps/8) (L/16), plus for
    # the exact SVD 0.75 n_updates 12.3 s (chi_op (n_max+1)^2 / 2592)^3.
    if "s8" in ONLY:
        import math as _math
        _section("S8 the convergence plane, the tau0 >= 1 rows, lambda = 0.1 and the ED anchor")
        t0 = time.time()
        PARTS = set((_args.s8_parts or ",".join(S8_ALL_PARTS)).split(","))
        SWEEP_KW = dict(BEST_KW)   # n_max 8, chi_op 32, chi_acc 64, chi_mpo 128, weight 0.5, chi_dmrg 16, sketched SVD
        GENEROUS8 = dict(chi_op=64, chi_acc=128, chi_mpo=None, svd_min_real=1e-10, weight_decay=1.0,
                         svd_dense_limit="inf", chi_dmrg=32, window_pad=None, max_sweeps=16,
                         max_E_err=1e-12)
        FINE_A = dict(L=32, m=0.25, lam=1.0, tau0=1.5)   # a = 1/2 at fixed bare physics (s7e)

        def _plane_kw(chi):
            return dict(chi_op=chi, chi_acc=2 * chi, chi_mpo=4 * chi, weight_decay=0.5, chi_dmrg=16,
                        svd_dense_limit="inf")

        def _n_updates(L, x, tau0, dt, pad):
            """Bond updates of one operator-TEBD trajectory (11 layers per order-4 step)."""
            n = max(2, int(_math.ceil(4.0 * tau0 / dt - 1e-9)))
            dt_eff = 4.0 * tau0 / n
            tot = 0
            for k in range(1, n + 1):
                if pad is None:
                    lo, hi = 0, L - 1
                else:
                    R = int(_math.ceil(k * dt_eff - 1e-9)) + int(pad)
                    lo, hi = max(0, x - R), min(L - 1, x + R)
                tot += 11 * (hi - lo) / 2.0
            return int(tot), n

        def _est_mpo(spec):
            kw = spec["kw"]
            n_max, L, tau0 = spec["n_max"], spec["L"], spec["tau0"]
            chi = kw.get("chi_op", CHI_OP)
            dt = kw.get("dt") or DT
            n_upd, n_steps = _n_updates(L, L // 2, tau0, dt, kw.get("window_pad", PAD))
            base = 900.0 * ((n_max + 1) / 9.0) ** 4 * (chi / 32.0) ** 1.5 * (n_steps / 8.0) * (L / 16.0)
            if kw.get("svd_dense_limit") == "inf":
                base += 0.75 * n_upd * 12.3 * (chi * (n_max + 1) ** 2 / 2592.0) ** 3
            return float(base)

        def _est_ed(spec):
            d = spec["n_max"] + 1
            dim = d ** spec["L"] / 2.0
            if spec["kind"] == "ed":
                return float((dim / 23328.0) ** 3 * 2 * 2400.0)    # eigh of each sector, ~40 min at 23k
            # Lanczos: ~250 O_f matvecs per sector, each 2 n_steps Chebyshev steps of ~45 matvecs
            return float(2 * 250 * 2 * 8 * 45 * (dim / 30000.0) * 0.005 * (2.0 if spec.get("dt_fine") else 1.0))

        SPECS8 = []

        def _mpo8(lam, n_max, tag, *, part, priority, L=QEI_L, m=MASS, tau0=TAU0, axis=None,
                  threads=2, chi_gs=16, **kw):
            spec = {"kind": "mpo", "L": int(L), "m": float(m), "tau0": float(tau0), "lam": float(lam),
                    "n_max": int(n_max), "tag": tag, "section": "S8", "part": part, "threads": threads,
                    "chi_gs": int(chi_gs), "kw": kw, "priority": float(priority)}
            if axis:
                spec["axis"] = axis
            spec["est_wall_s"] = _est_mpo(spec)
            spec["cost"] = spec["est_wall_s"]
            if part in PARTS:
                SPECS8.append(spec)
            return spec

        def _ed8(kind, lam, n_max, L, tag, *, part, priority, threads=2):
            spec = {"kind": kind, "L": int(L), "m": MASS, "tau0": TAU0, "lam": float(lam),
                    "n_max": int(n_max), "tag": tag, "section": "S8", "part": part, "threads": threads,
                    "dt": DT, "priority": float(priority)}
            if kind == "ed":
                spec["kernels"] = ["analytic", "trapezoid"]
            else:
                spec["dt_fine"] = DT / 3.0
                # ARPACK: ncv 80 halves the Lanczos count of the default basis (measured
                # at L = 6, n_max = 3: 842 -> 442 O_f applications), tol 1e-10 on the
                # residual keeps the eigenvalue at 6e-12 of the dense row (its error is
                # quadratic in the residual)
                spec["tol"] = 1e-10
                spec["ncv"] = 80
            spec["est_wall_s"] = _est_ed(spec)
            spec["cost"] = spec["est_wall_s"]
            if part in PARTS:
                SPECS8.append(spec)
            return spec

        # -- lam01: lambda = 0.1 at the sweep's settings (control: the s5h lambda = 0 row) --
        _mpo8(0.1, BEST_NMAX, "sweep_lam0.1", part="lam01", priority=100, axis="sweep", **SWEEP_KW)
        # -- tau0 rows and the a = 1/2 point --
        for chi, pr in ((32, 85), (48, 75)):
            kwt = dict(chi_op=chi, chi_acc=2 * chi, chi_mpo=4 * chi, weight_decay=0.5, chi_dmrg=16)
            for tau0 in (1.0, 1.5):
                for lam in (4.0, 0.0):
                    _mpo8(lam, 8, f"tau0_{tau0:g}_chiop{chi}_lam{lam:g}", part="tau0", priority=pr,
                          axis="tau0", tau0=tau0, **kwt)
            for lam in (FINE_A["lam"], 0.0):
                _mpo8(lam, 8, f"ahalf_chiop{chi}_lam{lam:g}", part="tau0", priority=pr - 17, axis="a_half",
                      L=FINE_A["L"], m=FINE_A["m"], tau0=FINE_A["tau0"], **kwt)
        # -- the plane: cheap settings first, lambda 0 / 4 pairs before lambda 2, lambda 3 last --
        PLANE = [(8, 32, 90), (10, 32, 80), (8, 64, 70), (12, 32, 65), (10, 64, 55), (12, 64, 40)]
        for n, chi, pr in PLANE:
            for lam in (4.0, 0.0):
                _mpo8(lam, n, f"plane_nmax{n}_chiop{chi}_lam{lam:g}", part="plane", priority=pr, axis="plane",
                      **_plane_kw(chi))
            _mpo8(2.0, n, f"plane_nmax{n}_chiop{chi}_lam2", part="plane", priority=pr - 30, axis="plane",
                  **_plane_kw(chi))
        for n, chi in ((10, 32), (8, 64)):      # ASSUMPTION: 'two best settings' = the two mid-cost
            _mpo8(3.0, n, f"plane_nmax{n}_chiop{chi}_lam3", part="plane", priority=48, axis="plane",  # ones spanning both axes
                  **_plane_kw(chi))
        # -- plane2 (S8 TRIAGE, 2026-09-12): the plane that can be quoted ----------------
        # Three measurements retired the plane above (all in the MANIFEST "S8 triage"
        # section, all archived in data/s8_weight_triage.json):
        #  (i)   its exact-SVD knob (svd_dense_limit "inf") costs 3.0-22x at fixed
        #        (L, n_max, chi_op, weight) and buys NO accuracy: at L = 16, n_max 8,
        #        chi_op 32, weight 0.5 it takes the lambda = 0 control -- the only
        #        exactly-known number at the paper's own L -- from exact/free 1.00192
        #        (sketched) to 1.02261, twelve times worse, for 3.4x the wall; at
        #        L = 5, n_max 8, lambda = 4 it takes the deviation from the exact ED
        #        value from +39.43 % to +42.25 %, for 18x the wall.  plane2 is
        #        sketched-SVD throughout, and that is where its cost saving comes from.
        #  (ii)  weight_decay, not chi_op, sets the lambda >= 2 error at L <= 6, but at
        #        L = 16 lambda = 0 the ordering REVERSES (weight 1 is 8.2 % off there
        #        against weight 0.5's 0.19 %).  So weight is an explicit plane AXIS with
        #        both values run and anchored -- never a value chosen off one point.
        #  (iii) every (n_max, chi_op, weight) column of plane2 has its own exact-ED
        #        anchor at L <= 6 at the SAME knobs (part "wt"), because the harvest's
        #        5 % reference-consistency gate is anti-correlated with the true error.
        PLANE2_W = (0.5, 1.0)

        def _plane2_kw(chi, w):
            return dict(chi_op=chi, chi_acc=2 * chi, chi_mpo=4 * chi, weight_decay=w, chi_dmrg=16)

        for n, chi, pr in ((8, 32, 92), (8, 64, 88), (10, 32, 84), (10, 64, 68),
                           (12, 32, 64), (12, 64, 44)):
            for w in PLANE2_W:
                for lam, dp in ((4.0, 0.0), (0.0, 0.0), (2.0, -2.0)):
                    _mpo8(lam, n, f"plane2_nmax{n}_chiop{chi}_w{w:g}_lam{lam:g}", part="plane2",
                          priority=pr + dp, axis="plane2", **_plane2_kw(chi, w))
        # -- wt: the exact-ED-anchored weight ladder that every plane2 column rests on ---
        # L = 5 n_max 8 is the largest Fock cutoff where a matrix-free ED value exists, and
        # is the plane's own cutoff; L = 6 n_max 5 and L = 5 n_max 6 are the second and
        # third anchors, so no statement rests on one point.  lambda = 2 is in because
        # item 4 is about lambda >= 2, not about lambda = 4.
        for (L_, n_, pr_) in ((5, 8, 168), (6, 5, 164), (5, 6, 160)):
            for chi in (64, 32):
                for w in (1.0, 0.5, 0.75, 0.25):
                    for lam in (4.0, 2.0, 0.0):
                        _mpo8(lam, n_, f"wt_L{L_}_n{n_}_chiop{chi}_w{w:g}_lam{lam:g}", part="wt",
                              priority=pr_ - (2.0 if chi == 32 else 0.0) - (0.0 if lam == 4.0 else 1.0),
                              axis="weight", L=L_, threads=1, **_plane2_kw(chi, w))
        # -- mpo_anchor: the MPO route where ED is exact --
        for (L_, n) in ((5, 6), (5, 8), (6, 5), (6, 6)):
            for lam in (0.0, 4.0):
                _mpo8(lam, n, f"anchor_sweepknobs_L{L_}_nmax{n}_lam{lam:g}", part="mpo_anchor", priority=78,
                      axis="sweep_knobs", L=L_, **SWEEP_KW)
        for lam in (0.0, 4.0):
            _mpo8(lam, 8, f"anchor_planeknobs32_L5_nmax8_lam{lam:g}", part="mpo_anchor", priority=77,
                  axis="plane_knobs", L=5, **_plane_kw(32))
            # S8 TRIAGE: these two anchor the RETIRED plane's chi_op 64 exact-SVD knobs, so
            # they move to the retired part with it (7.6 h that nothing quotes)
            _mpo8(lam, 8, f"anchor_planeknobs64_L5_nmax8_lam{lam:g}", part="plane", priority=52,
                  axis="plane_knobs", L=5, **_plane_kw(64))
            _mpo8(lam, 6, f"anchor_generous_L6_nmax6_lam{lam:g}", part="mpo_anchor", priority=74,
                  axis="generous", L=6, **GENEROUS8)
            for n, pr in ((6, 73), (7, 60), (8, 45)):
                _mpo8(lam, n, f"anchor_generous_L5_nmax{n}_lam{lam:g}", part="mpo_anchor", priority=pr,
                      axis="generous", L=5, **GENEROUS8)
        # -- ed: the matrix-free route (validation rungs first), then the dense n_max 5 rows --
        for n, pr in ((3, 99), (4, 98), (5, 97)):
            for lam in (0.0, 4.0):
                _ed8("ed_lanczos", lam, n, 6, f"edlz_L6_nmax{n}_lam{lam:g}", part="ed", priority=pr)
        for n, pr in ((6, 96), (7, 94), (8, 92)):
            for lam in (0.0, 4.0):
                _ed8("ed_lanczos", lam, n, 5, f"edlz_L5_nmax{n}_lam{lam:g}", part="ed", priority=pr)
        # S8 TRIAGE: item 4 is about lambda >= 2, so the weight ladder needs exact values
        # at lambda = 2 at the three anchor points, not only at lambda = 0 and 4
        for L_, n_ in ((5, 6), (6, 5), (5, 8)):
            _ed8("ed_lanczos", 2.0, n_, L_, f"edlz_L{L_}_nmax{n_}_lam2", part="ed", priority=93)
        for lam in (0.0, 4.0):
            _ed8("ed_lanczos", lam, 6, 6, f"edlz_L6_nmax6_lam{lam:g}", part="ed", priority=90)
            _ed8("ed", lam, 5, 6, f"ed_L6_nmax5_lam{lam:g}", part="ed", priority=80)

        # -- keff: the measured-dispersion reference at lambda = 2 (s7a has 0 / 1 / 4), s7a's knobs --
        for lam in (2.0,):
            spec = {"kind": "keff", "L": int(QEI_L), "m": MASS, "tau0": TAU0, "lam": float(lam), "n_max": 6,
                    "tag": f"keff_lam{lam:g}", "section": "S8", "part": "keff", "threads": 2, "chi_gs": 16,
                    "T": 10.0, "disp_dt": 0.1, "disp_chi": 32, "r_max": 4, "axis": "keff",
                    "priority": 95.0, "est_wall_s": 1000.0, "cost": 1000.0}
            if "keff" in PARTS:
                SPECS8.append(spec)

        est_total = sum(s["est_wall_s"] for s in SPECS8)
        by_part = {}
        for s in SPECS8:
            by_part.setdefault(s["part"], [0, 0.0])
            by_part[s["part"]][0] += 1
            by_part[s["part"]][1] += s["est_wall_s"]
        plan = {"_source": "notebook_qei_exact.py::S8 plan", "date_utc": _now(), "build_git_head": build,
                "parts": sorted(PARTS), "workers": W, "threads_per_job": 2,
                "n_jobs": len(SPECS8), "est_total_job_wall_h": est_total / 3600.0,
                "est_by_part": {k: {"n_jobs": v[0], "est_wall_h": v[1] / 3600.0} for k, v in by_part.items()},
                "wall_model": ("900 s ((n_max+1)/9)^4 (chi_op/32)^1.5 (n_steps/8) (L/16) [+ exact SVD: "
                               "0.75 n_updates 12.3 s (chi_op (n_max+1)^2/2592)^3]; 2 BLAS threads, idle"),
                "jobs": [{k: s[k] for k in ("tag", "part", "kind", "L", "m", "lam", "n_max", "tau0",
                                             "priority", "est_wall_s")} | {"kw": s.get("kw", {})}
                         for s in sorted(SPECS8, key=lambda s: (-s["priority"], -s["cost"]))]}
        _dump(f"s8_plan{S8_SUFFIX}.json", plan)
        print(f"  S8 parts {sorted(PARTS)}: {len(SPECS8)} jobs, estimated {est_total / 3600:.1f} h of "
              f"2-thread job wall ({', '.join(f'{k} {v[0]} jobs {v[1] / 3600:.1f} h' for k, v in by_part.items())})")
        print("  priority  est_h   tag")
        for s in plan["jobs"]:
            print(f"  {s['priority']:8.0f}  {s['est_wall_s'] / 3600:5.1f}   {s['tag']}")
        if _args.plan:
            raise SystemExit(0)

        rows8 = run_jobs(SPECS8, W, "s8")

        def _load_jobs(prefix):
            out = []
            for pth in sorted((DATA / "jobs").glob(f"{prefix}_*.json")):
                if pth.name.endswith(".spec.json"):
                    continue
                try:
                    with open(pth) as fh:
                        rec = json.load(fh)
                    out.append((rec["spec"], rec["row"]))
                except Exception:
                    continue
            return out

        # every finished S8 job on disk, whichever runner produced it (two runners share
        # the pool: the MPO parts and the ED part), so each summary below is complete
        # and is written only by a run that owns its part
        ok8 = [row for _spec, row in _load_jobs("s8") if "error" not in row]
        ok8_tags = {r.get("tag") for r in ok8}
        n_fin = sum(1 for s in SPECS8 if s["tag"] in ok8_tags)
        print(f"  S8: {n_fin} of this run's {len(SPECS8)} jobs finished; {len(ok8)} S8 rows on disk", flush=True)

        # E_keff / E_H at (L 16, tau0 0.75): the cutoff-free factor of s7b (lambda 1 and 4)
        # and the S8 keff rows (lambda 2), so every plane row also reads against K_eff
        keff_factor = {}
        pth = DATA / "s7b_reference.csv"
        if pth.exists():
            with open(pth) as fh:
                for r_ in csv.DictReader(fh):
                    if abs(float(r_["tau0"]) - TAU0) < 1e-9 and int(float(r_["L"])) == QEI_L:
                        keff_factor[float(r_["lam"])] = float(r_["E_keff_over_E_hartree"])
        for r_ in ok8:
            if r_.get("route") == "keff" and abs(r_["tau0"] - TAU0) < 1e-9 and r_["L"] == QEI_L:
                keff_factor[float(r_["lam"])] = float(r_["E_keff_over_E_hartree"])
        band_ref = None
        pth = DATA / "s5h_lambda_sweep.json"
        if pth.exists():
            with open(pth) as fh:
                band_ref = json.load(fh).get("band_lam0")
        band_ref = float(band_ref) if band_ref is not None else 0.0019

        # ---- plane: control-normalized residual per knob setting ----
        plane_rows = [r for r in ok8 if r.get("axis") == "plane"] if "plane" in PARTS else []
        settings = {}
        for r in plane_rows:
            settings.setdefault((r["n_max"], r["chi_op_allowed"]), {})[r["lam"]] = r
        plane_out = []
        for (n, chi), by_lam in sorted(settings.items()):
            ctrl = by_lam.get(0.0)
            for lam, r in sorted(by_lam.items()):
                q = {"n_max": n, "chi_op": chi, "lam": lam, "E_min": r["E_min"],
                     "E_hartree_exact": r["E_hartree_exact"], "E_free_exact": r["E_free_exact"],
                     "ratio_exact_over_hartree_exact": r["ratio_exact_over_hartree_exact"],
                     "ratio_exact_over_free_exact": r["ratio_exact_over_free_exact"],
                     "ref_consistency": r["ref_consistency"], "norm_ratio": r["norm_ratio"],
                     # the same reference-inconsistency gate the tau0 block applies (fix round 1):
                     # |E_ref - E_ref_static| must be <= 5 % of |E_min|, or the row's E_min is not
                     # a resolved number whatever its control does
                     "refcons_over_E_min": abs(r["ref_consistency"]) / abs(r["E_min"]),
                     "refcons_ok": bool(abs(r["ref_consistency"]) / abs(r["E_min"]) <= 0.05),
                     "trunc_op_weighted": r["trunc_op_weighted"], "chi_mpo": r["chi_mpo"],
                     "dmrg_chi": r["dmrg_chi"], "mpo_variance": r["mpo_variance"], "time_s": r["time_s"],
                     "control_lam0": (ctrl["ratio_exact_over_hartree_exact"] if ctrl else None),
                     "band_lam0": (abs(ctrl["ratio_exact_over_hartree_exact"] - 1.0) if ctrl else None),
                     "residual": 1.0 - r["ratio_exact_over_hartree_exact"],
                     "residual_ctrl_normalized": ((1.0 - r["ratio_exact_over_hartree_exact"]
                                                   / ctrl["ratio_exact_over_hartree_exact"]) if ctrl else None)}
                kf = keff_factor.get(float(lam))
                q["E_keff_over_E_hartree"] = kf
                q["ratio_exact_over_keff"] = (q["ratio_exact_over_hartree_exact"] / kf) if kf else None
                q["residual_keff_ctrl_normalized"] = ((1.0 - q["ratio_exact_over_hartree_exact"] / kf
                                                       / ctrl["ratio_exact_over_hartree_exact"])
                                                      if (kf and ctrl) else None)
                q["est_wall_s"] = next((s["est_wall_s"] for s in SPECS8 if s["tag"] == r.get("tag")), None)
                plane_out.append(q)
        if "plane" in PARTS:
            _csv("s8_convergence_plane.csv", plane_out)
        per_lam = {}
        for q in plane_out:
            if q["lam"] > 0 and q["residual_ctrl_normalized"] is not None:
                per_lam.setdefault(f"lam={q['lam']:g}", []).append(
                    {"n_max": q["n_max"], "chi_op": q["chi_op"], "ratio_hartree": q["ratio_exact_over_hartree_exact"],
                     "control": q["control_lam0"], "residual_ctrl_normalized": q["residual_ctrl_normalized"],
                     "refcons_over_E_min": q["refcons_over_E_min"], "refcons_ok": q["refcons_ok"],
                     "band_lam0": q["band_lam0"]})
        spread = {k: {"min": min(v_["residual_ctrl_normalized"] for v_ in v),
                      "max": max(v_["residual_ctrl_normalized"] for v_ in v), "n_settings": len(v)}
                  for k, v in per_lam.items()}
        # The verdict logic of the harvest (MANIFEST item 4).  At each lambda > 0 the
        # systematic is the spread of the control-normalized residual (i) across the two
        # largest n_max at chi_op 32 and (ii) between chi_op 32 and 64 at every n_max where
        # both exist; a magnitude is quoted only when both spreads exist and the larger is
        # within 3 lambda = 0 bands (3 x 0.19 % of s5h), otherwise the residual is BRACKETED
        # by the settings run and not resolved.  Measured, not assumed: the plane's own
        # lambda = 0 control at each setting is carried in band_lam0.
        verdict = {}
        for key, rows_ in per_lam.items():
            by = {(int(v["n_max"]), int(v["chi_op"])): v for v in rows_}
            nm32 = sorted({n for (n, c) in by if c == 32})
            top2 = nm32[-2:] if len(nm32) >= 2 else nm32
            vals_top = [by[(n, 32)]["residual_ctrl_normalized"] for n in top2]
            spread_nmax = (max(vals_top) - min(vals_top)) if len(vals_top) == 2 else None
            chi_pairs = [(n, by[(n, 32)]["residual_ctrl_normalized"], by[(n, 64)]["residual_ctrl_normalized"])
                         for n in nm32 if (n, 64) in by]
            spread_chi = max(abs(a - b) for _, a, b in chi_pairs) if chi_pairs else None
            spreads = [s_ for s_ in (spread_nmax, spread_chi) if s_ is not None]
            systematic = max(spreads) if spreads else None
            all_vals = [v["residual_ctrl_normalized"] for v in rows_]
            all_keff = [q["residual_keff_ctrl_normalized"] for q in plane_out
                        if f"lam={q['lam']:g}" == key and q["residual_keff_ctrl_normalized"] is not None]
            verdict[key] = {"n_max_at_chi32": nm32, "two_largest_n_max_at_chi32": top2,
                            "residual_at_two_largest_n_max": vals_top, "spread_two_largest_n_max": spread_nmax,
                            "chi_pairs_n_max": [n for n, _, _ in chi_pairs], "spread_chi32_vs_64": spread_chi,
                            "systematic": systematic, "band_lam0_sweep": band_ref,
                            "worst_control_band_in_plane": max(v["band_lam0"] for v in rows_),
                            "criterion": "resolved iff both spreads exist and max(spread) <= 3 x band_lam0_sweep",
                            "rows_failing_refcons_gate": [{"n_max": v["n_max"], "chi_op": v["chi_op"],
                                                           "refcons_over_E_min": v["refcons_over_E_min"]}
                                                          for v in rows_ if not v["refcons_ok"]],
                            "refcons_gate": 0.05,
                            "resolved": bool(systematic is not None and len(spreads) == 2
                                             and systematic <= 3.0 * band_ref),
                            "bracket_hartree": [min(all_vals), max(all_vals)],
                            "bracket_keff": ([min(all_keff), max(all_keff)] if all_keff else None),
                            "n_settings": len(rows_)}
        # measured vs estimated walls of every finished S8 MPO job (the plan's wall model),
        # and what the unfinished plane jobs would cost at the measured factor
        wall_cal = []
        for spec_, row_ in _load_jobs("s8"):
            if "error" in row_ or spec_.get("kind") != "mpo" or not spec_.get("est_wall_s"):
                continue
            wall_cal.append({"tag": spec_["tag"], "part": spec_.get("part"), "n_max": spec_["n_max"],
                             "chi_op": spec_["kw"].get("chi_op"), "L": spec_["L"], "tau0": spec_["tau0"],
                             "est_wall_s": spec_["est_wall_s"], "time_s": row_["time_s"],
                             "measured_over_estimated": row_["time_s"] / spec_["est_wall_s"],
                             "threads": row_.get("threads")})
        # the wall model is stated at 2 BLAS threads, so only 2-thread jobs calibrate it
        # (fix round 1: the chi_op-16 isolation rows ran at 1 thread and were skewing the median)
        cal2 = [w_ for w_ in wall_cal if str(w_.get("threads")) == "2"]
        ratios = sorted(w_["measured_over_estimated"] for w_ in cal2 if w_["part"] == "plane") or \
            sorted(w_["measured_over_estimated"] for w_ in cal2)
        med = ratios[len(ratios) // 2] if ratios else None
        remaining = {s_["tag"]: {"est_wall_h": s_["est_wall_s"] / 3600.0,
                                 "projected_wall_h": (s_["est_wall_s"] * med / 3600.0 if med else None)}
                     for s_ in SPECS8 if s_["part"] == "plane" and s_["tag"] not in ok8_tags}
        if "plane" in PARTS:
            _dump("s8_convergence_plane.json", {"_source": "notebook_qei_exact.py::S8 plane", "date_utc": _now(),
                                            "build_git_head": build, "knobs": "weight 0.5, exact SVD, chi_acc 2 chi_op, "
                                            "chi_mpo 4 chi_op, chi_dmrg 16, dt 0.375 order 4, pad 2, L 16, tau0 0.75, m 0.5",
                                            "rows": plane_out, "per_lambda": per_lam,
                                            "spread_residual_ctrl_normalized": spread,
                                            "verdict": verdict, "band_lam0_sweep": band_ref,
                                            "keff_factor_by_lambda": {f"lam={k_:g}": v_ for k_, v_ in keff_factor.items()},
                                            "wall_calibration": wall_cal,
                                            "wall_measured_over_estimated_median": med,
                                            "wall_calibration_n_2thread": len(ratios),
                                            "wall_calibration_note": ("median over 2-BLAS-thread jobs only (the model's "
                                                                      "stated condition), plane jobs if any have finished, "
                                                                      "else every 2-thread S8 MPO job"),
                                            "plane_jobs_unfinished": remaining,
                                            "n_finished": len(plane_out),
                                            "n_specified": sum(1 for s in SPECS8 if s["part"] == "plane")})
        if plane_out:
            print("  plane (weight 0.5, exact SVD): n_max chi_op lam  exact/H  control  ctrl-norm residual  refcons  norm  time  [exact/K_eff]")
            for q in plane_out:
                rn = q["residual_ctrl_normalized"]
                print(f"    {q['n_max']:2d} {q['chi_op']:3d} {q['lam']:<4g} {q['ratio_exact_over_hartree_exact']:.4f}  "
                      f"{(q['control_lam0'] if q['control_lam0'] is not None else float('nan')):.4f}  "
                      f"{(rn if rn is not None else float('nan')):+.4f}  {q['ref_consistency']:.1e}  "
                      f"{q['norm_ratio']:.3f}  {q['time_s']:.0f}s  "
                      f"[{(q['ratio_exact_over_keff'] if q['ratio_exact_over_keff'] is not None else float('nan')):.4f}]", flush=True)

        # the exact O(lambda) slope of the residual and the lambda <= 1 residuals, read from
        # the archived sweep so nothing here is a literal
        PT_SLOPE, p2_small_pts = None, []
        pth = DATA / "s5h_lambda_sweep.json"
        if pth.exists():
            with open(pth) as fh:
                _s5h = json.load(fh)
            PT_SLOPE = _s5h.get("pt_anchor", {}).get("dR_dlambda_at_0")
            PT_SLOPE = float(PT_SLOPE) if PT_SLOPE is not None else None
            p2_small_pts = [(float(z_["lam"]), float(z_["residual_ctrl_normalized"]))
                            for z_ in _s5h.get("rows", [])
                            if 0.0 < float(z_["lam"]) <= 1.0
                            and z_.get("residual_ctrl_normalized") is not None]
        pth = DATA / "s8_lam0.1.json"
        if pth.exists():
            with open(pth) as fh:
                _l01 = json.load(fh)
            if _l01.get("residual_ctrl_normalized") is not None:
                p2_small_pts.append((float(_l01["row"]["lam"]),
                                     float(_l01["residual_ctrl_normalized"])))
        p2_small_pts.sort()
        # ---- plane2 (S8 TRIAGE unit 2, 2026-09-12): the item-4 verdict -----------------
        # The retired "plane" part fixed weight 0.5 and the exact SVD.  plane2 walks
        # (n_max, chi_op, WEIGHT) at the sketched SVD, and every (chi_op, weight, lambda)
        # column has exact-ED partners in the L <= 6 grid.
        #
        # THE VERDICT RULE (MANIFEST item 4), applied to whatever has landed:
        #   systematic(lambda) = the LARGEST spread of the control-normalized residual over
        #     the axes actually walked -- weight at fixed (n_max, chi_op), chi_op at fixed
        #     (n_max, weight), n_max at fixed (chi_op, weight) -- together with the route's
        #     own same-spec reproducibility wherever the identical spec was run twice.
        #   A MAGNITUDE IS QUOTED only if systematic <= 3 x the lambda = 0 band of the sweep
        #     AND at least two settings pass the ED-anchor gate.  Otherwise the residual is
        #     BRACKETED by min/max over the landed settings, and labelled bracketed.
        #   The ED-anchor gate is MEASURED, not assumed: for each (chi_op, weight, lambda) it
        #     is max |rel_dev_vs_ed| over EVERY anchored point at that setting and the SAME
        #     SVD mode (L 5 n_max 8 and L 6 n_max 5/6, every archived thread count), so the
        #     gate compares like knobs with like.  A setting with no anchored point at that
        #     (chi_op, weight, lambda, svd) is NOT qualified: the gate is evidence of accuracy,
        #     and absence of evidence does not pass it.
        # ASSUMPTION (stated in the MANIFEST, not proved here): the (chi_op, weight) operator
        #     truncation error measured against exact ED at L <= 6, n_max <= 8 transfers to
        #     L = 16, n_max 8-10.  No exact answer exists at L = 16 at any lambda > 0, so this
        #     is the only accuracy evidence there is, and it is why the verdict is a bracket.
        P2_ANCHOR_GATE = 0.02           # 2 % of |E_min|; ~10 lambda = 0 bands
        P2_RESOLVE_BANDS = 3.0

        def _p2_svd(r_):
            v = r_.get("svd_dense_limit")
            return "exact" if (v is None or not np.isfinite(float(v))) else "sketched"

        # exact-ED partners, built from the job files directly so that a
        # "--s8-parts plane2 --no-new-jobs" harvest is complete on its own
        p2_ed = [r_ for r_ in ok8 if str(r_.get("route", "")).startswith("ed")]
        p2_ed += [row for spec, row in _load_jobs("s5i")
                  if spec["kind"] in ("ed", "ed_lanczos") and "error" not in row]
        p2_ed.sort(key=lambda q_: 0 if q_["route"] == "ed_lanczos" else 1)
        p2_anchor = {}
        for r_ in ok8:
            if r_.get("route") != "mpo" or r_["L"] > 6:
                continue
            e_ = next((q_ for q_ in p2_ed if (q_["L"], q_["n_max"], q_["lam"], q_["tau0"])
                       == (r_["L"], r_["n_max"], r_["lam"], r_["tau0"])), None)
            if e_ is None:
                continue
            ex_ = e_.get("E_min_trapezoid", e_["E_min"])
            p2_anchor.setdefault((int(r_["chi_op_allowed"]), float(r_["weight_decay"]),
                                  float(r_["lam"]), _p2_svd(r_)), []).append(
                {"tag": r_.get("tag"), "L": r_["L"], "n_max": r_["n_max"], "svd": _p2_svd(r_),
                 "threads": r_.get("threads"), "rel_dev_vs_ed": (r_["E_min"] - ex_) / abs(ex_)})

        # every spec run more than once, so the route's own floor enters the systematic
        def _p2_speckey(r_):
            return (r_["L"], r_["n_max"], r_["lam"], r_["tau0"], r_["m"], r_["chi_op_allowed"],
                    r_["chi_acc"], r_["chi_mpo"], r_["weight_decay"], r_["dt"], r_["order"],
                    r_["n_steps"], r_["window_pad"], r_["dmrg_chi"], _p2_svd(r_))
        p2_rep = {}
        for r_ in [q_ for _sp, q_ in (_load_jobs("s5g") + _load_jobs("s5h") + _load_jobs("s8"))
                   if "error" not in q_]:
            if r_.get("route") == "mpo":
                p2_rep.setdefault(_p2_speckey(r_), []).append(float(r_["E_min"]))

        p2_rows_raw = [r for r in ok8 if r.get("axis") == "plane2"] if "plane2" in PARTS else []
        p2_set = {}
        for r in p2_rows_raw:
            p2_set.setdefault((int(r["n_max"]), int(r["chi_op_allowed"]),
                               float(r["weight_decay"])), {})[float(r["lam"])] = r
        p2_out = []
        for (n, chi, w), by_lam in sorted(p2_set.items()):
            ctrl = by_lam.get(0.0)
            for lam, r in sorted(by_lam.items()):
                rh = r["ratio_exact_over_hartree_exact"]
                c0 = ctrl["ratio_exact_over_hartree_exact"] if ctrl else None
                kf = keff_factor.get(float(lam))
                pts = p2_anchor.get((chi, w, float(lam), _p2_svd(r)), [])
                amax = max(abs(p_["rel_dev_vs_ed"]) for p_ in pts) if pts else None
                es = p2_rep.get(_p2_speckey(r), [])
                d_es = (max(es) - min(es)) if len(es) >= 2 else None
                p2_out.append({
                    "n_max": n, "chi_op": chi, "weight_decay": w, "lam": lam, "tag": r.get("tag"),
                    "threads": r.get("threads"), "E_min": r["E_min"],
                    "E_hartree_exact": r["E_hartree_exact"], "E_free_exact": r["E_free_exact"],
                    "ratio_exact_over_hartree_exact": rh,
                    "ratio_exact_over_free_exact": r["ratio_exact_over_free_exact"],
                    "E_keff_over_E_hartree": kf,
                    "ratio_exact_over_keff": (rh / kf) if kf else None,
                    "control_lam0": c0, "band_lam0": (abs(c0 - 1.0) if c0 else None),
                    "residual": 1.0 - rh,
                    "residual_ctrl_normalized": ((1.0 - rh / c0) if c0 else None),
                    "residual_keff_ctrl_normalized": ((1.0 - rh / kf / c0) if (kf and c0) else None),
                    "refcons_over_E_min": abs(r["ref_consistency"]) / abs(r["E_min"]),
                    "refcons_ok": bool(abs(r["ref_consistency"]) / abs(r["E_min"]) <= 0.05),
                    "anchor_max_abs_rel_dev_vs_ed": amax,
                    "anchor_n_points": len(pts),
                    "anchor_ok": bool(pts and amax <= P2_ANCHOR_GATE),
                    "anchor_points": sorted(pts, key=lambda z_: (z_["L"], z_["n_max"], str(z_["threads"]))),
                    "same_spec_n_runs": len(es),
                    "same_spec_spread_E_min": d_es,
                    "same_spec_spread_in_residual": ((d_es / abs(r["E_hartree_exact"]) / c0)
                                                     if (d_es is not None and c0) else None),
                    "trunc_op_weighted": r["trunc_op_weighted"], "norm_ratio": r["norm_ratio"],
                    "mpo_variance": r["mpo_variance"], "chi_mpo": r["chi_mpo"],
                    "dmrg_chi": r["dmrg_chi"], "time_s": r["time_s"]})
        # the verdict, as a function so the same rule can be applied to the live rows and to
        # the FROZEN snapshot the MANIFEST quotes (the plane fills while the prose is written;
        # freezing the quoted tag set is what keeps a quoted number reproducible)
        def _p2_make_verdict(rows_in_):
            out_ = {}
            for lam in sorted({q["lam"] for q in rows_in_ if q["lam"] > 0}):
                rs = [q for q in rows_in_ if q["lam"] == lam and q["residual_ctrl_normalized"] is not None]
                if not rs:
                    continue
                by = {(q["n_max"], q["chi_op"], q["weight_decay"]): q for q in rs}
                def _sp(pairs):
                    got = [(k1, k2, abs(by[k1]["residual_ctrl_normalized"] - by[k2]["residual_ctrl_normalized"]))
                           for k1, k2 in pairs if k1 in by and k2 in by]
                    return max(got, key=lambda z_: z_[2]) if got else None
                sp_w = _sp([((n_, c_, 0.5), (n_, c_, 1.0)) for (n_, c_, _w) in by])
                sp_c = _sp([((n_, 32, w_), (n_, 64, w_)) for (n_, _c, w_) in by])
                sp_n = _sp([((8, c_, w_), (10, c_, w_)) for (_n, c_, w_) in by])
                sp_r = max((q["same_spec_spread_in_residual"] for q in rs
                            if q["same_spec_spread_in_residual"] is not None), default=None)
                parts_ = {"weight_0.5_vs_1": sp_w, "chi_op_32_vs_64": sp_c, "n_max_8_vs_10": sp_n}
                vals_ = [z_[2] for z_ in (sp_w, sp_c, sp_n) if z_ is not None]
                if sp_r is not None:
                    vals_.append(sp_r)
                systematic = max(vals_) if vals_ else None
                allh = [q["residual_ctrl_normalized"] for q in rs]
                allk = [q["residual_keff_ctrl_normalized"] for q in rs
                        if q["residual_keff_ctrl_normalized"] is not None]
                qual = [q for q in rs if q["anchor_ok"]]
                qh = [q["residual_ctrl_normalized"] for q in qual]
                qk = [q["residual_keff_ctrl_normalized"] for q in qual
                      if q["residual_keff_ctrl_normalized"] is not None]
                dR = PT_SLOPE
                out_[f"lam={lam:g}"] = {
                    "n_settings": len(rs), "settings": [[q["n_max"], q["chi_op"], q["weight_decay"]] for q in rs],
                    "band_lam0_sweep": band_ref,
                    "spreads": {k_: ({"pair": [list(v_[0]), list(v_[1])], "spread": v_[2]} if v_ else None)
                                for k_, v_ in parts_.items()},
                    "same_spec_spread_in_residual": sp_r,
                    "systematic": systematic,
                    "systematic_in_bands": (systematic / band_ref) if systematic else None,
                    "criterion": ("QUOTED iff systematic <= %g x band_lam0_sweep AND at least two "
                                  "settings pass the ED-anchor gate |rel_dev_vs_ed| <= %g at the same "
                                  "(chi_op, weight, lambda); else BRACKETED" % (P2_RESOLVE_BANDS, P2_ANCHOR_GATE)),
                    "anchor_gate": P2_ANCHOR_GATE,
                    # the SIGN of the residual is a weaker statement than its magnitude and it
                    # survives more of the plane, so it is recorded separately
                    "n_positive_hartree": sum(1 for v_ in allh if v_ > 0),
                    "n_positive_keff": sum(1 for v_ in allk if v_ > 0),
                    "n_keff": len(allk),
                    "negative_keff_settings": [{"n_max": q["n_max"], "chi_op": q["chi_op"],
                                                "weight_decay": q["weight_decay"],
                                                "residual_keff_ctrl_normalized": q["residual_keff_ctrl_normalized"],
                                                "anchor_max_abs_rel_dev_vs_ed": q["anchor_max_abs_rel_dev_vs_ed"],
                                                "refcons_over_E_min": q["refcons_over_E_min"]}
                                               for q in rs if q["residual_keff_ctrl_normalized"] is not None
                                               and q["residual_keff_ctrl_normalized"] <= 0],
                    "anchor_qualified": [{"n_max": q["n_max"], "chi_op": q["chi_op"], "weight_decay": q["weight_decay"],
                                          "anchor_max_abs_rel_dev_vs_ed": q["anchor_max_abs_rel_dev_vs_ed"],
                                          "anchor_n_points": q["anchor_n_points"],
                                          "residual_ctrl_normalized": q["residual_ctrl_normalized"],
                                          "residual_keff_ctrl_normalized": q["residual_keff_ctrl_normalized"]}
                                         for q in qual],
                    "n_anchor_qualified": len(qual),
                    "resolved": bool(systematic is not None and systematic <= P2_RESOLVE_BANDS * band_ref
                                     and len(qual) >= 2),
                    "bracket_hartree": [min(allh), max(allh)],
                    "bracket_keff": ([min(allk), max(allk)] if allk else None),
                    "bracket_hartree_anchor_qualified": ([min(qh), max(qh)] if qh else None),
                    "bracket_keff_anchor_qualified": ([min(qk), max(qk)] if qk else None),
                    # the gate is stated in advance, so its VALUE must not be what carries the
                    # verdict: the same bracket is recomputed at four gates.  (At lambda = 2 the
                    # (chi_op 64, weight 1) column misses the 2 % gate by 0.14 pp, so this is the
                    # one place the gate value could have been tuned to an answer.  It was not.)
                    "anchor_gate_sensitivity": [
                        {"gate": g_,
                         "n_qualified": sum(1 for q in rs if q["anchor_max_abs_rel_dev_vs_ed"] is not None
                                            and q["anchor_max_abs_rel_dev_vs_ed"] <= g_),
                         "settings": [[q["n_max"], q["chi_op"], q["weight_decay"]] for q in rs
                                      if q["anchor_max_abs_rel_dev_vs_ed"] is not None
                                      and q["anchor_max_abs_rel_dev_vs_ed"] <= g_],
                         "bracket_hartree": ([min(v_), max(v_)] if v_ else None),
                         "spread": ((max(v_) - min(v_)) if len(v_) >= 2 else None),
                         "spread_in_bands": (((max(v_) - min(v_)) / band_ref) if len(v_) >= 2 else None),
                         "resolved_at_this_gate": bool(len(v_) >= 2 and (max(v_) - min(v_)) <= P2_RESOLVE_BANDS * band_ref)}
                        for g_, v_ in ((g_, [q["residual_ctrl_normalized"] for q in rs
                                             if q["anchor_max_abs_rel_dev_vs_ed"] is not None
                                             and q["anchor_max_abs_rel_dev_vs_ed"] <= g_])
                                       for g_ in (0.01, 0.02, 0.025, 0.03, 0.05))],
                    "rows_failing_refcons_gate": [{"n_max": q["n_max"], "chi_op": q["chi_op"],
                                                   "weight_decay": q["weight_decay"],
                                                   "refcons_over_E_min": q["refcons_over_E_min"]}
                                                  for q in rs if not q["refcons_ok"]],
                    "refcons_gate": 0.05,
                    # confrontation with the exact first-order slope of the free extremal state
                    "predicted_residual_O1": (dR * lam) if dR is not None else None,
                    "implied_second_order_coefficient": ({
                        "rule": "c2 = (R_ctrl - dR_dlambda_at_0 * lam) / lam**2",
                        "dR_dlambda_at_0": dR,
                        "bracket": [min((v_ - dR * lam) / lam ** 2 for v_ in allh),
                                    max((v_ - dR * lam) / lam ** 2 for v_ in allh)],
                        "anchor_qualified": ([min((v_ - dR * lam) / lam ** 2 for v_ in qh),
                                              max((v_ - dR * lam) / lam ** 2 for v_ in qh)] if qh else None)}
                        if dR is not None else None)}
            return out_

        p2_verdict = _p2_make_verdict(p2_out)
        # UNIT 2 SNAPSHOT, frozen 2026-09-13: the 15 plane2 rows that had landed when the
        # MANIFEST's item 4 paragraph and its numbers-table rows were written.  The live
        # verdict above moves as the pool lands rows; this one does not, so every number the
        # prose quotes stays reproducible and the tests that pin it never go stale.  Adding a
        # tag here is a deliberate act that must be made together with the prose.
        # FIX ROUND 1 (2026-09-13): plane2_nmax10_chiop32_w0.5_lam2 was NOT in the first
        # freeze although it was on disk when the freeze commit was made and although its own
        # lambda = 0 control and its lambda = 4 sibling both were.  That omission is what left
        # spreads["n_max_8_vs_10"] null at lambda = 2 and made "n_max is the smallest axis at
        # both lambda" sayable.  It is added here: at lambda = 2 the landed n_max 8 -> 10 pair
        # moves the residual by 5.471 pp (28.45 bands), which is SECOND, not smallest, and the
        # MANIFEST/STATUS ranking is restated to match.  The lambda = 2 systematic, its bracket
        # and its c2 bracket are unchanged by the addition (the weight axis still carries it).
        P2_UNIT2_TAGS = (
            "plane2_nmax8_chiop32_w0.5_lam0", "plane2_nmax8_chiop32_w0.5_lam2",
            "plane2_nmax8_chiop32_w0.5_lam4", "plane2_nmax8_chiop32_w1_lam0",
            "plane2_nmax8_chiop32_w1_lam2", "plane2_nmax8_chiop32_w1_lam4",
            "plane2_nmax8_chiop64_w0.5_lam0", "plane2_nmax8_chiop64_w0.5_lam2",
            "plane2_nmax8_chiop64_w0.5_lam4", "plane2_nmax8_chiop64_w1_lam0",
            "plane2_nmax8_chiop64_w1_lam2", "plane2_nmax8_chiop64_w1_lam4",
            "plane2_nmax10_chiop32_w0.5_lam0", "plane2_nmax10_chiop32_w0.5_lam2",
            "plane2_nmax10_chiop32_w0.5_lam4")
        # a frozen row must also have its OWN lambda = 0 control inside the frozen set, or the
        # snapshot would silently move when a control lands outside it (the (n_max 10, chi_op 32,
        # weight 1) column did exactly that while this section was being written)
        _p2_frozen_cols = {(q["n_max"], q["chi_op"], q["weight_decay"]) for q in p2_out
                           if q.get("tag") in P2_UNIT2_TAGS and q["lam"] == 0.0}
        p2_frozen_rows = [q for q in p2_out if q.get("tag") in P2_UNIT2_TAGS
                          and (q["n_max"], q["chi_op"], q["weight_decay"]) in _p2_frozen_cols]
        _p2_all_plane2_tags = {s_["tag"] for s_ in SPECS8 if s_["part"] == "plane2"}
        p2_unit2 = {"what": ("the item 4 verdict on the 15 rows that had landed when the MANIFEST "
                             "prose was written (2026-09-13); frozen so the quoted numbers stay "
                             "reproducible while the pool keeps landing plane2 rows"),
                    "tags": list(P2_UNIT2_TAGS),
                    "n_tags_landed": len(p2_frozen_rows),
                    "verdict": _p2_make_verdict(p2_frozen_rows)}
        # WHAT WOULD CLOSE IT, AND WHAT IT WOULD COST.  The criterion needs a second
        # anchor-qualified setting at each lambda and an n_max ladder inside one.  The
        # cheapest such ladder is the (chi_op, weight) column of a setting that already
        # clears the ED gate, run at every n_max of the plane2 grid; this costs it out at
        # the plane2 wall calibration MEASURED on this tree (median time_s / est_wall_s over
        # the landed 2-thread plane2 jobs), so the hours are not a wall-model literal.
        def _p2_closure(verdict_, landed_):
            # the calibration is measured over the jobs in THIS landed set, so the frozen
            # snapshot's hours are frozen too (the live median drifts as the pool lands rows)
            p2_cal = sorted(r_["time_s"] / s_["est_wall_s"]
                            for s_, r_ in _load_jobs("s8")
                            if "error" not in r_ and s_.get("part") == "plane2"
                            and s_.get("est_wall_s") and str(r_.get("threads")) == "2"
                            and s_["tag"] in landed_)
            p2_cal_med = p2_cal[len(p2_cal) // 2] if p2_cal else None
            out_ = {}
            for k_, v_ in sorted(verdict_.items()):
              for qq in v_["anchor_qualified"]:
                col = (int(qq["chi_op"]), float(qq["weight_decay"]))
                if col in out_:
                    continue
                todo = [s_ for s_ in SPECS8 if s_["part"] == "plane2"
                        and int(s_["kw"]["chi_op"]) == col[0]
                        and float(s_["kw"]["weight_decay"]) == col[1]
                        and s_["tag"] not in landed_]
                nom_h = sum(s_["est_wall_s"] for s_ in todo) / 3600.0
                out_[f"chi_op{col[0]}_w{col[1]:g}"] = {
                    "qualified_at": k_, "n_jobs_unlanded": len(todo),
                    "tags": sorted(s_["tag"] for s_ in todo),
                    "n_max_covered": sorted({s_["n_max"] for s_ in todo}),
                    "nominal_h": nom_h,
                    "plane2_calibration_measured": p2_cal_med,
                    "n_calibration_jobs": len(p2_cal),
                    "calibrated_h": (nom_h * p2_cal_med) if p2_cal_med else None,
                    "caveat": ("finishing this column TESTS resolution, it does not deliver it: "
                               "the criterion also needs a SECOND (chi_op, weight) pair to clear "
                               "the ED-anchor gate at that lambda, which no other landed setting "
                               "does, and qualifying one is not on the queue")}
            return out_

        p2_closure = _p2_closure(p2_verdict, ok8_tags)
        p2_unit2["closure_cost"] = _p2_closure(p2_unit2["verdict"], set(P2_UNIT2_TAGS))
        # the same c2 read off every lambda <= 1 point, so "about -0.10 from the small-lambda
        # points" is quoted with its own spread instead of as a single number
        p2_c2_small = ({f"lam={l_:g}": (R_ - PT_SLOPE * l_) / l_ ** 2 for l_, R_ in p2_small_pts}
                       if PT_SLOPE is not None else {})
        if "plane2" in PARTS:
            _csv("s8_convergence_plane2.csv", [{k_: v_ for k_, v_ in q.items()
                                                if k_ != "anchor_points"} for q in p2_out])
            _dump("s8_convergence_plane2.json", {
                "_source": "notebook_qei_exact.py::S8 plane2 (item 4 verdict)", "date_utc": _now(),
                "build_git_head": build,
                "knobs": ("L 16, tau0 0.75, m 0.5, chi_acc 2 chi_op, chi_mpo 4 chi_op, chi_dmrg 16, "
                          "dt 0.375 order 4 n_steps 8, window_pad 2, SKETCHED SVD "
                          "(svd_dense_limit 600); the walked axes are n_max, chi_op and weight_decay"),
                "band_lam0_sweep": band_ref,
                "keff_factor_by_lambda": {f"lam={k_:g}": v_ for k_, v_ in sorted(keff_factor.items())},
                "anchor_gate": P2_ANCHOR_GATE, "resolve_bands": P2_RESOLVE_BANDS,
                "anchor_table": {f"chi_op{k_[0]}_w{k_[1]:g}_lam{k_[2]:g}_{k_[3]}":
                                 {"max_abs_rel_dev_vs_ed": max(abs(p_["rel_dev_vs_ed"]) for p_ in v_),
                                  "n_points": len(v_),
                                  "points": sorted(v_, key=lambda z_: (z_["L"], z_["n_max"], str(z_["threads"])))}
                                 for k_, v_ in sorted(p2_anchor.items())},
                "rows": p2_out, "verdict": p2_verdict, "unit2_snapshot": p2_unit2,
                "pt_slope_dR_dlambda_at_0": PT_SLOPE,
                "small_lambda_points": [{"lam": l_, "residual_ctrl_normalized": R_} for l_, R_ in p2_small_pts],
                "implied_second_order_coefficient_small_lambda": p2_c2_small,
                "closure_cost": p2_closure,
                "n_finished": len(p2_out),
                "n_specified": sum(1 for s in SPECS8 if s["part"] == "plane2"),
                "unfinished": sorted(s["tag"] for s in SPECS8
                                     if s["part"] == "plane2" and s["tag"] not in ok8_tags)})
        if p2_out:
            print("  plane2 (sketched SVD, L 16): n_max chi_op w  lam  exact/H  exact/K_eff  ctrl  "
                  "ctrl-norm R  ED-anchor  refcons  time")
            for q in p2_out:
                rn = q["residual_ctrl_normalized"]
                am = q["anchor_max_abs_rel_dev_vs_ed"]
                print(f"    {q['n_max']:2d} {q['chi_op']:3d} {q['weight_decay']:<4g} {q['lam']:<4g} "
                      f"{q['ratio_exact_over_hartree_exact']:.4f}  "
                      f"{(q['ratio_exact_over_keff'] if q['ratio_exact_over_keff'] is not None else float('nan')):.4f}  "
                      f"{(q['control_lam0'] if q['control_lam0'] is not None else float('nan')):.4f}  "
                      f"{(rn if rn is not None else float('nan')):+.5f}  "
                      f"{(f'{am:.2e}' if am is not None else '   --   '):>8s}"
                      f"{'*' if q['anchor_ok'] else ' '} {q['refcons_over_E_min']:.1e}  {q['time_s']:.0f}s",
                      flush=True)
            for k_, v_ in sorted(p2_verdict.items()):
                lab = "RESOLVED" if v_["resolved"] else "BRACKETED"
                print(f"    [item 4] {k_}: {lab}  R_ctrl in "
                      f"[{v_['bracket_hartree'][0]:+.5f}, {v_['bracket_hartree'][1]:+.5f}] vs E_H"
                      + (f", [{v_['bracket_keff'][0]:+.5f}, {v_['bracket_keff'][1]:+.5f}] vs K_eff"
                         if v_["bracket_keff"] else "")
                      + f"; systematic {v_['systematic']:.5f} = {v_['systematic_in_bands']:.1f} bands; "
                      f"{v_['n_anchor_qualified']} of {v_['n_settings']} settings pass the ED-anchor gate",
                      flush=True)

        # ---- tau0 rows and the a = 1/2 point ----
        tau_rows = [r for r in ok8 if r.get("axis") in ("tau0", "a_half")] if "tau0" in PARTS else []
        tset = {}
        for r in tau_rows:
            tset.setdefault((r["axis"], r["tau0"], r["chi_op_allowed"]), {})[r["lam"]] = r
        tau_out = []
        for (ax, tau0, chi), by_lam in sorted(tset.items()):
            ctrl = by_lam.get(0.0)
            for lam, r in sorted(by_lam.items()):
                tau_out.append({"axis": ax, "tau0": tau0, "chi_op": chi, "lam": lam, "L": r["L"], "m": r["m"],
                                "n_steps": r["n_steps"], "dt": r["dt"], "E_min": r["E_min"],
                                "E_free_exact": r["E_free_exact"], "E_hartree_exact": r["E_hartree_exact"],
                                "ratio_exact_over_free_exact": r["ratio_exact_over_free_exact"],
                                "ratio_exact_over_hartree_exact": r["ratio_exact_over_hartree_exact"],
                                "control_lam0": (ctrl["ratio_exact_over_hartree_exact"] if ctrl else None),
                                "control_refcons_over_E_min": (abs(ctrl["ref_consistency"]) / abs(ctrl["E_min"]) if ctrl else None),
                                "refcons_over_E_min": abs(r["ref_consistency"]) / abs(r["E_min"]),
                                "ratio_hartree_normalized": ((r["ratio_exact_over_hartree_exact"]
                                                              / ctrl["ratio_exact_over_hartree_exact"]) if ctrl else None),
                                "residual_hartree_normalized": ((1.0 - r["ratio_exact_over_hartree_exact"]
                                                                 / ctrl["ratio_exact_over_hartree_exact"]) if ctrl else None),
                                "ref_consistency": r["ref_consistency"], "norm_ratio": r["norm_ratio"],
                                "trunc_op_weighted": r["trunc_op_weighted"], "time_s": r["time_s"]})
        # The scaling refit (MANIFEST item 5).  A tau0 row is usable only if its lambda = 0
        # control reads 1 within the s7s gate (|control - 1| < 0.05); the archived three
        # (tau0 0.375 / 0.5 / 0.75, chi_op 16, n_max 6, weight 0.25; s7d) are refitted with
        # every usable S8 row (chi_op 32 / 48, n_max 8, weight 0.5) -- MIXED knobs, labelled
        # so -- against E_H and against K_eff (the s7b factor per tau0 at lambda 4), the
        # same log r vs log(1/tau0) power law as s7s.
        CONTROL_GATE = 0.05
        # ... and only if the row's and its control's reference inconsistency |E_ref - E_ref_static|
        # is within the lambda <= 1 level of the sweep (s5h: 1.3-5.1 % of |E_min|); the tau0 = 1.0
        # lambda = 4 row at chi_op 32 passes the control gate with a 38 % inconsistency, which is
        # not a resolved E_min at all
        REFCONS_GATE = 0.05
        arch = []
        pth = DATA / "s7b_reference.csv"
        if pth.exists():
            with open(pth) as fh:
                for r_ in csv.DictReader(fh):
                    if float(r_["lam"]) == 4.0 and r_.get("control_ok", "").strip().lower() == "true":
                        arch.append({"tau0": float(r_["tau0"]), "chi_op": 16, "source": "s7d",
                                     "control_lam0": float(r_["control_lam0"]),
                                     "residual_hartree_normalized": float(r_["residual_hartree_normalized"]),
                                     "residual_keff_normalized": float(r_["residual_keff_normalized"]),
                                     "E_keff_over_E_hartree": float(r_["E_keff_over_E_hartree"])})
        keff_tau = {}
        if pth.exists():
            with open(pth) as fh:
                for r_ in csv.DictReader(fh):
                    if float(r_["lam"]) == 4.0:
                        keff_tau[round(float(r_["tau0"]), 6)] = float(r_["E_keff_over_E_hartree"])
        refit = {"control_gate": CONTROL_GATE, "archived_rows_chiop16": arch, "by_chi_op": {}}
        for chi in sorted({int(q["chi_op"]) for q in tau_out if q["axis"] == "tau0"}):
            new = []
            for q in tau_out:
                if q["axis"] != "tau0" or int(q["chi_op"]) != chi or q["lam"] != 4.0:
                    continue
                kf = keff_tau.get(round(float(q["tau0"]), 6))
                ctrl_ok = q["control_lam0"] is not None and abs(q["control_lam0"] - 1.0) < CONTROL_GATE
                refcons_ok = (q["refcons_over_E_min"] <= REFCONS_GATE
                              and q["control_refcons_over_E_min"] is not None
                              and q["control_refcons_over_E_min"] <= REFCONS_GATE)
                ok = ctrl_ok and refcons_ok
                new.append({"tau0": q["tau0"], "chi_op": chi, "source": "s8", "control_lam0": q["control_lam0"],
                            "control_ok": bool(ctrl_ok), "refcons_ok": bool(refcons_ok), "usable": bool(ok),
                            "control_refcons_over_E_min": q["control_refcons_over_E_min"],
                            "ratio_exact_over_hartree": q["ratio_exact_over_hartree_exact"],
                            "residual_hartree_normalized": q["residual_hartree_normalized"],
                            "E_keff_over_E_hartree": kf,
                            "residual_keff_normalized": ((1.0 - q["ratio_exact_over_hartree_exact"] / kf / q["control_lam0"])
                                                         if (kf and q["control_lam0"]) else None),
                            "ref_consistency_over_E_min": abs(q["ref_consistency"]) / abs(q["E_min"]),
                            "norm_ratio": q["norm_ratio"]})
            usable = [n_ for n_ in new if n_["usable"]]
            entry = {"rows": new, "n_usable": len(usable), "usable_tau0": [n_["tau0"] for n_ in usable],
                     "failed_tau0_controls": {f"{n_['tau0']:g}": n_["control_lam0"] for n_ in new if not n_["control_ok"]},
                     "failed_tau0_refcons": {f"{n_['tau0']:g}": n_["ref_consistency_over_E_min"] for n_ in new if not n_["refcons_ok"]},
                     "gates": {"control": CONTROL_GATE, "refcons_over_E_min": REFCONS_GATE}}
            for key in ("residual_hartree_normalized", "residual_keff_normalized"):
                pts = [(a_["tau0"], a_[key]) for a_ in arch] + [(u_["tau0"], u_[key]) for u_ in usable if u_[key] is not None]
                # a power law is fitted to positive residuals only: a sign change is not a
                # power law, and is reported as such
                pos = [(t_, v_) for t_, v_ in pts if v_ is not None and v_ > 0]
                fit = {"tau0": [t_ for t_, _ in pts], "value": [v_ for _, v_ in pts], "knobs": "mixed (s7d chi_op 16 + s8 usable rows)",
                       "sign_changed_tau0": [t_ for t_, v_ in pts if v_ is not None and v_ <= 0]}
                if len(pos) >= 3 and len(usable) > 0 and any(u_[key] is not None and u_[key] > 0 for u_ in usable):
                    x_ = np.log(1.0 / np.array([t_ for t_, _ in pos]))
                    y_ = np.log(np.array([v_ for _, v_ in pos]))
                    p_, cov_ = np.polyfit(x_, y_, 1, cov=True)
                    fit["exponent_of_inverse_tau0"] = float(p_[0])
                    fit["exponent_stderr"] = float(np.sqrt(cov_[0, 0]))
                    fit["n_points_used"] = len(pos)
                else:
                    fit["exponent_of_inverse_tau0"] = None
                    fit["note"] = ("no usable positive-residual S8 row at this chi_op: the exponent statement stays "
                                   "restricted to the archived tau0 0.375-0.75 range (s7_summary.json)")
                entry[key] = fit
            refit["by_chi_op"][f"chi_op={chi}"] = entry
        ahalf = [q for q in tau_out if q["axis"] == "a_half"]
        refit["a_half"] = {"rows": [{"chi_op": q["chi_op"], "lam": q["lam"], "ratio_exact_over_hartree": q["ratio_exact_over_hartree_exact"],
                                     "control_lam0": q["control_lam0"], "ratio_hartree_normalized": q["ratio_hartree_normalized"],
                                     "control_ok": (q["control_lam0"] is not None and abs(q["control_lam0"] - 1.0) < CONTROL_GATE),
                                     "ref_consistency_over_E_min": abs(q["ref_consistency"]) / abs(q["E_min"]),
                                     "control_refcons_over_E_min": q["control_refcons_over_E_min"],
                                     "usable": (q["control_lam0"] is not None and abs(q["control_lam0"] - 1.0) < CONTROL_GATE
                                                and q["refcons_over_E_min"] <= REFCONS_GATE
                                                and q["control_refcons_over_E_min"] is not None
                                                and q["control_refcons_over_E_min"] <= REFCONS_GATE),
                                     "norm_ratio": q["norm_ratio"]} for q in ahalf if q["lam"] > 0],
                           "archived_control_chiop16": 1.173, "archived_ratio_hartree_chiop16": 1.548,
                           "_archived_knobs": ("s7e: n_max 6, chi_op 16, chi_acc 32, chi_mpo 64, chi_dmrg 12, "
                                               "weight 0.25 -- NOT the S8 knobs (n_max 8, weight 0.5, chi_dmrg 16), "
                                               "so a difference is not a chi_op effect; cf. chiop16_isolation")}
        # ---- the chi_op axis, isolated (fix round 1) ----
        # The archived tau0 >= 1 controls (1.085 / 1.410) are s7d rows at n_max 6, chi_op 16,
        # chi_acc 32, chi_mpo 64, chi_dmrg 12, weight 0.25; the S8 rows above are at n_max 8,
        # chi_op 32, chi_acc 64, chi_mpo 128, chi_dmrg 16, weight 0.5.  Six knobs apart, so the
        # improvement cannot be attributed to chi_op.  The `tau0_iso` rows run the S8 recipe AT
        # chi_op 16 (n_max 8, weight 0.5, chi_acc 32, chi_mpo 64, chi_dmrg 16), which splits the
        # gap into (s7d -> iso) = the (n_max, weight, chi_dmrg) step and (iso -> chi_op 32) = the
        # operator-bond step.  Controls only (lambda = 0): the control is what the claim is about.
        iso_rows = [r for r in ok8 if r.get("axis") == "tau0_iso"]
        iso_out = [{"tau0": r["tau0"], "chi_op": r["chi_op_allowed"], "lam": r["lam"], "L": r["L"],
                    "m": r["m"], "n_max": r["n_max"], "weight_decay": r["weight_decay"],
                    "chi_acc": r["chi_acc"], "chi_mpo": r["chi_mpo"], "dmrg_chi": r["dmrg_chi"],
                    "n_steps": r["n_steps"], "dt": r["dt"], "E_min": r["E_min"],
                    "E_hartree_exact": r["E_hartree_exact"],
                    "control_lam0": r["ratio_exact_over_hartree_exact"],
                    "refcons_over_E_min": abs(r["ref_consistency"]) / abs(r["E_min"]),
                    "norm_ratio": r["norm_ratio"], "trunc_op_weighted": r["trunc_op_weighted"],
                    "time_s": r["time_s"], "tag": r.get("tag")}
                   for r in sorted(iso_rows, key=lambda r_: (r_["tau0"], r_["lam"]))]
        ARCH_S7D = {1.0: 1.085, 1.5: 1.410}
        iso_split = {}
        for q in iso_out:
            if q["lam"] != 0.0:
                continue
            t_ = round(float(q["tau0"]), 6)
            s32 = next((z for z in tau_out if z["axis"] == "tau0" and z["lam"] == 0.0
                        and abs(z["tau0"] - t_) < 1e-9 and int(z["chi_op"]) == 32), None)
            a_ = ARCH_S7D.get(t_)
            iso_split[f"tau0={t_:g}"] = {
                "s7d_chiop16_nmax6_w0.25_chidmrg12": a_,
                "iso_chiop16_nmax8_w0.5_chidmrg16": q["control_lam0"],
                "s8_chiop32_nmax8_w0.5_chidmrg16": (s32["ratio_exact_over_hartree_exact"] if s32 else None),
                "step_nmax_weight_chidmrg": (q["control_lam0"] - a_) if a_ else None,
                "step_chi_op_16_to_32": ((s32["ratio_exact_over_hartree_exact"] - q["control_lam0"])
                                         if s32 else None),
                "iso_norm_ratio": q["norm_ratio"],
                "note": ("control (lambda = 0, must read 1); the two steps sum to the total "
                         "s7d -> S8 change, and say which knob carried it")}
        # FIX ROUND 2.  The control's chi_op sequence at FIXED n_max 8 / weight 0.5 /
        # chi_dmrg 16 (the iso row at chi_op 16, the pool rows at 32 and 48).  The MANIFEST
        # quotes it, so it is a field of the archive rather than a number reassembled by hand;
        # `toward_1` false means chi_op moves the control AWAY from its required value.
        chiop_seq = {}
        for t_ in sorted({round(float(q["tau0"]), 6) for q in iso_out if q["lam"] == 0.0}):
            pts = []
            q16 = next((z for z in iso_out if z["lam"] == 0.0 and abs(z["tau0"] - t_) < 1e-9), None)
            if q16 is not None:
                pts.append({"chi_op": 16, "control_lam0": q16["control_lam0"],
                            "refcons_over_E_min": q16["refcons_over_E_min"], "source": "tau0_iso"})
            for c_ in (32, 48, 64):
                z = next((z for z in tau_out if z["axis"] == "tau0" and z["lam"] == 0.0
                          and abs(z["tau0"] - t_) < 1e-9 and int(z["chi_op"]) == c_), None)
                if z is not None:
                    pts.append({"chi_op": c_, "control_lam0": z["ratio_exact_over_hartree_exact"],
                                "refcons_over_E_min": z["refcons_over_E_min"], "source": "tau0"})
            if len(pts) >= 2:
                chiop_seq[f"tau0={t_:g}"] = {
                    "chi_op": [q_["chi_op"] for q_ in pts],
                    "control_lam0": [q_["control_lam0"] for q_ in pts],
                    "refcons_over_E_min": [q_["refcons_over_E_min"] for q_ in pts],
                    "steps": [pts[i + 1]["control_lam0"] - pts[i]["control_lam0"] for i in range(len(pts) - 1)],
                    "toward_1": all(abs(pts[i + 1]["control_lam0"] - 1.0) < abs(pts[i]["control_lam0"] - 1.0)
                                    for i in range(len(pts) - 1)),
                    "refcons_gate": 0.05,
                    "refcons_ok": [bool(q_["refcons_over_E_min"] <= 0.05) for q_ in pts],
                    "note": ("lambda = 0 control at fixed n_max 8 / weight 0.5 / chi_dmrg 16; "
                             "chi_op 16 is the tau0_iso row, 32/48 are the pool rows")}
        if "tau0" in PARTS:
            _csv("s8_tau0_rows.csv", tau_out)
            _dump("s8_tau0_rows.json", {"_source": "notebook_qei_exact.py::S8 tau0", "date_utc": _now(),
                                    "build_git_head": build, "rows": tau_out,
                                    "knobs": "n_max 8, weight 0.5, sketched SVD, chi_acc 2 chi_op, chi_mpo 4 chi_op, "
                                             "chi_dmrg 16, dt 0.375 (s7d: DT for tau0 >= 0.75); a = 1/2 as s7e",
                                    "archived_controls_chiop16": {"tau0=1.0": 1.085, "tau0=1.5": 1.410, "a_half": 1.173,
                                                                  "_knobs": "s7d: n_max 6, chi_op 16, chi_acc 32, chi_mpo 64, "
                                                                            "chi_dmrg 12, weight 0.25 (NOT the S8 knobs)"},
                                    "chiop16_isolation": {"rows": iso_out, "split": iso_split,
                                                          "knobs": "S8 recipe at chi_op 16: n_max 8, weight 0.5, chi_acc 32, "
                                                                   "chi_mpo 64, chi_dmrg 16, sketched SVD, dt as s7d"},
                                    "control_chi_op_sequence": chiop_seq,
                                    "scaling_refit": refit,
                                    "n_finished": len(tau_out),
                                    "n_specified": sum(1 for s in SPECS8 if s["part"] == "tau0")})
        for q in iso_out:
            print(f"  [tau0_iso] tau0 {q['tau0']:<4g} chi_op {q['chi_op']:2d} lam {q['lam']:<4g} "
                  f"(S8 recipe at chi_op 16): control {q['control_lam0']:.4f} refcons/|E_min| "
                  f"{q['refcons_over_E_min']:.3f} norm {q['norm_ratio']:.3f} [{q['time_s']:.0f}s]", flush=True)
        for k_, v_ in iso_split.items():
            print(f"  [tau0_iso] {k_}: s7d {v_['s7d_chiop16_nmax6_w0.25_chidmrg12']} -> iso "
                  f"{v_['iso_chiop16_nmax8_w0.5_chidmrg16']:.5f} -> chi_op 32 "
                  f"{(v_['s8_chiop32_nmax8_w0.5_chidmrg16'] if v_['s8_chiop32_nmax8_w0.5_chidmrg16'] else float('nan')):.5f}"
                  f"  steps: (n_max,w,chi_dmrg) {(v_['step_nmax_weight_chidmrg'] if v_['step_nmax_weight_chidmrg'] else float('nan')):+.4f}"
                  f"  (chi_op) {(v_['step_chi_op_16_to_32'] if v_['step_chi_op_16_to_32'] else float('nan')):+.4f}", flush=True)
        for k_, v_ in chiop_seq.items():
            print(f"  [tau0_chiop] {k_} control vs chi_op {v_['chi_op']}: "
                  f"{['%.5f' % x for x in v_['control_lam0']]} steps "
                  f"{['%+.5f' % x for x in v_['steps']]} toward_1 {v_['toward_1']} "
                  f"refcons/|E_min| {['%.4f' % x for x in v_['refcons_over_E_min']]} "
                  f"(gate 0.05: {v_['refcons_ok']})", flush=True)
        for q in tau_out:
            print(f"  [{q['axis']}] tau0 {q['tau0']:<4g} chi_op {q['chi_op']:2d} lam {q['lam']:<4g}: exact/H "
                  f"{q['ratio_exact_over_hartree_exact']:.4f} control "
                  f"{(q['control_lam0'] if q['control_lam0'] is not None else float('nan')):.4f} normalized "
                  f"{(q['ratio_hartree_normalized'] if q['ratio_hartree_normalized'] is not None else float('nan')):.4f} "
                  f"[{q['time_s']:.0f}s]", flush=True)

        # ---- lambda = 0.1 at the sweep's settings ----
        r01 = next((r for r in ok8 if r.get("tag") == "sweep_lam0.1"), None) if "lam01" in PARTS else None
        ctrl_h = None
        pth = DATA / "jobs" / "s5h_sweep_lam0.json"
        if pth.exists():
            with open(pth) as fh:
                ctrl_h = json.load(fh)["row"]
        dR = None
        pth = DATA / "s5h_lambda_sweep.json"
        if pth.exists():
            with open(pth) as fh:
                dR = json.load(fh).get("pt_anchor", {}).get("dR_dlambda_at_0")
        lam01 = {"_source": "notebook_qei_exact.py::S8 lam01", "date_utc": _now(), "build_git_head": build,
                 "settings": {"n_max": BEST_NMAX, **SWEEP_KW}, "row": r01,
                 "control_lam0_s5h": (ctrl_h["ratio_exact_over_hartree_exact"] if ctrl_h else None),
                 "band_lam0_s5h": (abs(ctrl_h["ratio_exact_over_hartree_exact"] - 1.0) if ctrl_h else None),
                 "dR_dlambda_at_0": dR, "predicted_residual_O1": (0.1 * dR if dR is not None else None)}
        if r01 and ctrl_h:
            lam01["residual"] = 1.0 - r01["ratio_exact_over_hartree_exact"]
            lam01["residual_ctrl_normalized"] = 1.0 - r01["ratio_exact_over_hartree_exact"] / ctrl_h["ratio_exact_over_hartree_exact"]
            if dR is not None:
                lam01["measured_minus_predicted"] = lam01["residual_ctrl_normalized"] - 0.1 * dR
                lam01["bands_from_prediction"] = lam01["measured_minus_predicted"] / lam01["band_lam0_s5h"]
            print(f"  lambda = 0.1 (sweep settings): exact/H {r01['ratio_exact_over_hartree_exact']:.5f}, "
                  f"ctrl-normalized residual {lam01['residual_ctrl_normalized']:+.4f} vs O(lambda) prediction "
                  f"{(0.1 * dR if dR is not None else float('nan')):+.4f} (band {lam01['band_lam0_s5h']:.2e}) "
                  f"[{r01['time_s']:.0f}s]", flush=True)
        if "lam01" in PARTS:
            _dump("s8_lam0.1.json", lam01)

        # ---- the ED anchor: matrix-free rows, their validation, and MPO-vs-ED pairs ----
        ed_lz = [r for r in ok8 if r.get("route") == "ed_lanczos"] if "ed" in PARTS else []
        ed_dense = [r for r in ok8 if r.get("route") == "ed"]
        ed_dense += [row for spec, row in _load_jobs("s5i") if spec["kind"] == "ed" and "error" not in row]
        validation = []
        for r in ed_lz:
            e = next((q for q in ed_dense if (q["L"], q["n_max"], q["lam"], q["tau0"]) == (r["L"], r["n_max"], r["lam"], r["tau0"])), None)
            if e is None:
                continue
            e_trap = e.get("E_min_trapezoid", e["E_min"])
            validation.append({"L": r["L"], "n_max": r["n_max"], "lam": r["lam"], "E_min_lanczos": r["E_min"],
                               "E_min_dense_trapezoid": e_trap, "E_min_dense_analytic": e.get("E_min_analytic", e["E_min"]),
                               "abs_dev_same_grid": r["E_min"] - e_trap,
                               "rel_dev_same_grid": (r["E_min"] - e_trap) / abs(e_trap),
                               "rel_dev_vs_analytic": (r["E_min"] - e.get("E_min_analytic", e["E_min"])) / abs(e["E_min"]),
                               "E_min_lanczos_fine": r.get("E_min_fine"),
                               "rel_dev_fine_vs_analytic": ((r["E_min_fine"] - e.get("E_min_analytic", e["E_min"])) / abs(e["E_min"])
                                                            if r.get("E_min_fine") is not None else None),
                               "sector_star_lanczos": r["sector_star"], "sector_star_dense": e["sector_star"],
                               "time_lanczos_s": r["time_s"], "time_dense_s": e["time_s"]})
        ed_all = ed_lz + [e for e in ed_dense if not any((r["L"], r["n_max"], r["lam"]) == (e["L"], e["n_max"], e["lam"]) for r in ed_lz)]
        mpo_rows = [r for r in ok8 if r.get("route") == "mpo" and r["L"] <= 8]
        mpo_rows += [row for spec, row in _load_jobs("s5i") if spec["kind"] == "mpo" and "error" not in row]
        pairs8 = []
        for r in mpo_rows:
            e = next((q for q in ed_lz if (q["L"], q["n_max"], q["lam"], q["tau0"]) == (r["L"], r["n_max"], r["lam"], r["tau0"])), None)
            src = "ed_lanczos"
            if e is None:
                e = next((q for q in ed_dense if (q["L"], q["n_max"], q["lam"], q["tau0"]) == (r["L"], r["n_max"], r["lam"], r["tau0"])), None)
                src = "ed_dense"
            if e is None:
                continue
            e_trap = e.get("E_min_trapezoid", e["E_min"])
            pairs8.append({"L": r["L"], "n_max": r["n_max"], "lam": r["lam"], "knobs": r.get("axis"), "tag": r.get("tag"),
                           "chi_op": r["chi_op_allowed"], "weight_decay": r["weight_decay"],
                           "svd_dense_limit": r.get("svd_dense_limit"), "ed_source": src,
                           "E_min_mpo": r["E_min"], "E_min_ed_same_grid": e_trap,
                           "rel_dev_mpo_vs_ed_same_grid": (r["E_min"] - e_trap) / abs(e_trap),
                           "ratio_hartree_mpo": r["ratio_exact_over_hartree_exact"],
                           "ratio_hartree_ed": e_trap / e["E_hartree_exact"],
                           "ref_consistency_mpo": r["ref_consistency"], "norm_ratio_mpo": r["norm_ratio"],
                           "trunc_op_weighted": r["trunc_op_weighted"], "time_mpo_s": r["time_s"], "time_ed_s": e["time_s"]})
        pairs8.sort(key=lambda p: (p["L"], p["n_max"], p["lam"], str(p["knobs"])))
        worst = {}
        for p_ in pairs8:
            k = str(p_["knobs"])
            worst[k] = max(worst.get(k, 0.0), abs(p_["rel_dev_mpo_vs_ed_same_grid"]))
        xval = {}
        for p_ in pairs8:
            k = str(p_["knobs"])
            e_ = xval.setdefault(k, {"pairs": [], "worst_abs_rel_dev": 0.0})
            e_["pairs"].append({"L": p_["L"], "n_max": p_["n_max"], "lam": p_["lam"],
                                "rel_dev": p_["rel_dev_mpo_vs_ed_same_grid"], "chi_op": p_["chi_op"]})
            e_["worst_abs_rel_dev"] = max(e_["worst_abs_rel_dev"], abs(p_["rel_dev_mpo_vs_ed_same_grid"]))
        for k, e_ in xval.items():
            e_["n_max_range"] = [min(p_["n_max"] for p_ in e_["pairs"]), max(p_["n_max"] for p_ in e_["pairs"])]
            e_["L_values"] = sorted({p_["L"] for p_ in e_["pairs"]})
            e_["n_pairs"] = len(e_["pairs"])
            e_["worst_by_n_max"] = {str(n_): max(abs(p_["rel_dev"]) for p_ in e_["pairs"] if p_["n_max"] == n_)
                                    for n_ in sorted({p_["n_max"] for p_ in e_["pairs"]})}
        ed_ladders = {}
        for L_ in (5, 6):
            for lam in (0.0, 4.0):
                seq = sorted([q for q in ed_all if q["L"] == L_ and q["lam"] == lam], key=lambda q: q["n_max"])
                if len(seq) >= 2:
                    ed_ladders[f"L={L_}/lam={lam:g}"] = {
                        "n_max": [q["n_max"] for q in seq],
                        "E_min_same_grid": [q.get("E_min_trapezoid", q["E_min"]) for q in seq],
                        "ratio_hartree": [q.get("E_min_trapezoid", q["E_min"]) / q["E_hartree_exact"] for q in seq],
                        "route": [q["route"] for q in seq], "sector_star": [q["sector_star"] for q in seq]}
        if "ed" in PARTS:
            _dump("s8_ed_anchor.json", {"_source": "notebook_qei_exact.py::S8 ed", "date_utc": _now(), "build_git_head": build,
                                    "python": platform.python_version(), "numpy": np.__version__,
                                    "quadrature": f"trapezoid, dt {DT} (the MPO route's grid), fine dt {DT / 3:g}",
                                    "ed_lanczos_rows": ed_lz, "validation_vs_dense": validation,
                                    "worst_abs_rel_dev_validation": (max(abs(v["rel_dev_same_grid"]) for v in validation)
                                                                     if validation else None),
                                    "pairs_mpo_vs_ed": pairs8, "worst_rel_dev_by_knobs": worst,
                                    "cross_validation_by_knobs": xval,
                                    "ed_ladders": ed_ladders,
                                    "n_ed_lanczos_finished": len(ed_lz),
                                    "n_ed_specified": sum(1 for s in SPECS8 if s["part"] == "ed")})
        if validation:
            print("  ED matrix-free vs dense (same trapezoid grid):")
            for v in validation:
                print(f"    L={v['L']} n_max={v['n_max']} lam={v['lam']:g}: lanczos {v['E_min_lanczos']:.12e} dense "
                      f"{v['E_min_dense_trapezoid']:.12e} rel dev {v['rel_dev_same_grid']:+.2e} "
                      f"(fine grid vs analytic {v['rel_dev_fine_vs_analytic']}) [{v['time_lanczos_s']:.0f}s vs {v['time_dense_s']:.0f}s]")
        for k, v in ed_ladders.items():
            print(f"  ED ladder {k}: n_max {v['n_max']} ratio_H {['%.4f' % x for x in v['ratio_hartree']]} routes {v['route']}")
        if pairs8:
            # FIX ROUND 2: print the lambda = 4 error as a function of n_max per knob set.  At
            # fixed chi_op 32 it GROWS with the Fock cutoff; round 1's harvest printed only the
            # flat list and the MANIFEST called the effect a bounded "3-12 %".
            for k_, e_ in sorted(xval.items()):
                l4 = sorted([p_ for p_ in e_["pairs"] if p_["lam"] != 0.0], key=lambda z: z["n_max"])
                l0 = sorted([p_ for p_ in e_["pairs"] if p_["lam"] == 0.0], key=lambda z: z["n_max"])
                if l4:
                    print(f"  [xval {k_}] worst |rel dev| by n_max: {e_['worst_by_n_max']}; "
                          f"lam4 {[('n%d %+.2e' % (z['n_max'], z['rel_dev'])) for z in l4]}; "
                          f"lam0 {[('n%d %+.2e' % (z['n_max'], z['rel_dev'])) for z in l0]}", flush=True)
            print("  MPO vs ED at the same (L, n_max, lambda), same grid:")
            for p_ in pairs8:
                print(f"    L={p_['L']} n_max={p_['n_max']} lam={p_['lam']:g} [{str(p_['knobs']):12s}] rel dev "
                      f"{p_['rel_dev_mpo_vs_ed_same_grid']:+.2e}  exact/H mpo {p_['ratio_hartree_mpo']:.4f} ed "
                      f"{p_['ratio_hartree_ed']:.4f} ({p_['ed_source']})", flush=True)
        # ---- the weight_decay triage (S8 TRIAGE, 2026-09-12) ------------------------
        # Accuracy at lambda >= 2 is only ever MEASURED against an exact ED value at the
        # same (L, n_max, lam, trapezoid grid).  This block groups every MPO row at
        # L <= 6 that has such a partner, orders it in weight_decay at fixed chi_op and
        # SVD mode, and records the L = 16 cost pairs that the cost-per-accuracy
        # statement rests on.  It writes data/s8_weight_triage.json.
        if "wt" in PARTS:
            # built from the job files directly (not from ed_lz, which is gated on the "ed"
            # part) so that a --s8-parts wt harvest is complete on its own
            ed_ref = [r_ for r_ in ok8 if str(r_.get("route", "")).startswith("ed")]
            ed_ref += [row for spec, row in _load_jobs("s5i")
                       if spec["kind"] in ("ed", "ed_lanczos") and "error" not in row]
            ed_ref.sort(key=lambda q: 0 if q["route"] == "ed_lanczos" else 1)

            def _knobs_but(r_, *drop):
                """Everything that is not the axis being compared, so a 'pair' is a pair."""
                k = {"L": r_["L"], "n_max": r_["n_max"], "lam": r_["lam"], "chi_op": r_["chi_op_allowed"],
                     "chi_acc": r_["chi_acc"], "chi_mpo": r_["chi_mpo"], "weight": r_["weight_decay"],
                     "dt": r_["dt"], "order": r_["order"], "chi_dmrg": r_["dmrg_chi"],
                     "window_pad": r_["window_pad"], "svd": _svd_mode(r_),
                     "threads": r_.get("threads")}
                for d_ in drop:
                    k.pop(d_)
                return tuple(sorted(k.items()))

            def _svd_mode(r_):
                v = r_.get("svd_dense_limit")
                return "exact" if (v is None or not np.isfinite(float(v))) else "sketched"

            wt_rows = []
            for r_ in mpo_rows:
                e_ = next((q for q in ed_ref if (q["L"], q["n_max"], q["lam"], q["tau0"])
                           == (r_["L"], r_["n_max"], r_["lam"], r_["tau0"])), None)
                if e_ is None:
                    continue
                ex_ = e_.get("E_min_trapezoid", e_["E_min"])
                wt_rows.append({"tag": r_.get("tag"), "L": r_["L"], "n_max": r_["n_max"], "lam": r_["lam"],
                                "chi_op_requested": r_["chi_op_allowed"], "chi_op_reached": r_["chi_op"],
                                "weight_decay": r_["weight_decay"], "chi_dmrg": r_["dmrg_chi"],
                                "svd": _svd_mode(r_), "E_min": r_["E_min"], "E_min_exact_ed": ex_,
                                "rel_dev_vs_ed": (r_["E_min"] - ex_) / abs(ex_),
                                "refcons_over_E_min": r_["ref_consistency"] / abs(r_["E_min"]),
                                "trunc_op_weighted": r_["trunc_op_weighted"], "norm_ratio": r_["norm_ratio"],
                                "chi_mpo_reached": r_["chi_mpo"], "threads": r_.get("threads"),
                                "time_s": r_["time_s"]})
            wt_rows.sort(key=lambda z: (z["L"], z["n_max"], z["lam"], z["svd"], z["chi_op_requested"],
                                        z["weight_decay"]))
            # the ladders: at one (L, n_max, lam, chi_op, svd), the deviation vs weight
            ladders = {}
            for z in wt_rows:
                k = (f"L{z['L']}/n{z['n_max']}/lam{z['lam']:g}/chi_op{z['chi_op_requested']}"
                     f"/chi_dmrg{z['chi_dmrg']}/{z['svd']}/thr{z['threads']}")
                ladders.setdefault(k, []).append([z["weight_decay"], z["rel_dev_vs_ed"], z["time_s"],
                                                  z["chi_op_reached"], z["refcons_over_E_min"]])
            for k in ladders:
                ladders[k].sort()
            multi = {k: v for k, v in ladders.items() if len(v) >= 2}
            best_w = {}
            for k, v in multi.items():
                w_best, d_best = min(((row[0], abs(row[1])) for row in v), key=lambda t: t[1])
                best_w[k] = {"weight_best": w_best, "abs_rel_dev_best": d_best,
                             "abs_rel_dev_at_0.5": next((abs(r_[1]) for r_ in v if r_[0] == 0.5), None),
                             "abs_rel_dev_at_1.0": next((abs(r_[1]) for r_ in v if r_[0] == 1.0), None)}
            # the L = 16 cost of the weight axis: pairs at identical (n_max, chi_op, lam, svd)
            l16 = [r_ for _sp, r_ in (_load_jobs("s5g") + _load_jobs("s5h") + _load_jobs("s8"))
                   if "error" not in r_ and r_.get("route") == "mpo" and r_["L"] == QEI_L
                   and abs(r_.get("tau0", TAU0) - TAU0) < 1e-9]
            # REPRODUCIBILITY.  Two runs of the SAME spec that differ only in the BLAS
            # thread count of the worker: at a fixed seed the randomized sketch is fixed,
            # but the reduction order is not, and the truncation decisions amplify it.
            rep = {}
            for r_ in mpo_rows + l16:
                key = (r_["L"], r_["n_max"], r_["lam"], r_["tau0"], r_["chi_op_allowed"], r_["chi_acc"],
                       r_["chi_mpo"] if False else None, r_["weight_decay"], r_["dt"], r_["order"],
                       r_["dmrg_chi"], r_["window_pad"], _svd_mode(r_))
                rep.setdefault(key, []).append(r_)
            repeats = []
            for key, group in rep.items():
                if len(group) < 2:
                    continue
                group = sorted(group, key=lambda q: str(q.get("threads")))
                e = [q["E_min"] for q in group]
                spread = (max(e) - min(e)) / abs(min(e, key=abs))
                ex_ = next((q for q in ed_ref if (q["L"], q["n_max"], q["lam"], q["tau0"])
                            == (group[0]["L"], group[0]["n_max"], group[0]["lam"], group[0]["tau0"])), None)
                ee = ex_.get("E_min_trapezoid", ex_["E_min"]) if ex_ else None
                repeats.append({"L": key[0], "n_max": key[1], "lam": key[2], "chi_op_requested": key[4],
                                "weight_decay": key[7], "chi_dmrg": key[10], "svd": key[12],
                                "runs": [{"tag": q.get("tag"), "threads": q.get("threads"), "E_min": q["E_min"],
                                          "chi_mpo_reached": q["chi_mpo"], "trunc_op_weighted": q["trunc_op_weighted"],
                                          "ref_consistency": q["ref_consistency"], "time_s": q["time_s"],
                                          "rel_dev_vs_ed": ((q["E_min"] - ee) / abs(ee)) if ee else None}
                                         for q in group],
                                "rel_spread_E_min": spread, "E_min_exact_ed": ee})
            repeats.sort(key=lambda z: -z["rel_spread_E_min"])

            # the L = 16 cost of the weight axis: pairs at identical (n_max, chi_op, lam, svd)
            cost_pairs = []
            for a in l16:
                for b in l16:
                    if _knobs_but(a, "weight") != _knobs_but(b, "weight"):
                        continue
                    if not (a["weight_decay"] < b["weight_decay"]):
                        continue
                    cost_pairs.append({"tag_lo": a.get("tag"), "tag_hi": b.get("tag"),
                                       "threads": a.get("threads"),
                                       "n_max": a["n_max"], "chi_op_requested": a["chi_op_allowed"],
                                       "lam": a["lam"], "svd": _svd_mode(a),
                                       "weight_lo": a["weight_decay"], "weight_hi": b["weight_decay"],
                                       "chi_op_reached_lo": a["chi_op"], "chi_op_reached_hi": b["chi_op"],
                                       "time_lo_s": a["time_s"], "time_hi_s": b["time_s"],
                                       "cost_ratio_hi_over_lo": b["time_s"] / a["time_s"],
                                       "trunc_op_lo": a["trunc_op_weighted"], "trunc_op_hi": b["trunc_op_weighted"],
                                       "norm_ratio_lo": a["norm_ratio"], "norm_ratio_hi": b["norm_ratio"],
                                       "ratio_free_lo": a["ratio_exact_over_free_exact"],
                                       "ratio_free_hi": b["ratio_exact_over_free_exact"]})
            cost_pairs.sort(key=lambda z: (z["n_max"], z["chi_op_requested"], z["lam"], z["svd"]))
            # the SVD-mode pairs: identical knobs, sketched vs exact SVD
            svd_pairs = []
            for a in l16 + [r_ for r_ in mpo_rows if r_["L"] <= 6]:
                for b in l16 + [r_ for r_ in mpo_rows if r_["L"] <= 6]:
                    if _knobs_but(a, "svd") != _knobs_but(b, "svd"):
                        continue
                    if _svd_mode(a) != "sketched" or _svd_mode(b) != "exact":
                        continue
                    e_ = next((q for q in ed_ref if (q["L"], q["n_max"], q["lam"], q["tau0"])
                               == (a["L"], a["n_max"], a["lam"], a["tau0"])), None)
                    ex_ = e_.get("E_min_trapezoid", e_["E_min"]) if e_ else None
                    svd_pairs.append({"tag_sketched": a.get("tag"), "tag_exact_svd": b.get("tag"),
                                      "threads": a.get("threads"),
                                      "L": a["L"], "n_max": a["n_max"], "lam": a["lam"],
                                      "chi_op_requested": a["chi_op_allowed"], "weight_decay": a["weight_decay"],
                                      "E_sketched": a["E_min"], "E_exact_svd": b["E_min"],
                                      "time_sketched_s": a["time_s"], "time_exact_svd_s": b["time_s"],
                                      "cost_ratio": b["time_s"] / a["time_s"],
                                      "E_min_exact_ed": ex_,
                                      "rel_dev_sketched": ((a["E_min"] - ex_) / abs(ex_)) if ex_ else None,
                                      "rel_dev_exact_svd": ((b["E_min"] - ex_) / abs(ex_)) if ex_ else None,
                                      "ratio_free_sketched": a["ratio_exact_over_free_exact"],
                                      "ratio_free_exact_svd": b["ratio_exact_over_free_exact"]})
            svd_pairs.sort(key=lambda z: (z["L"], z["n_max"], z["lam"], z["chi_op_requested"], z["weight_decay"]))
            # ---- FIX ROUND 1: band-aware penalty factors -------------------------------
            # At L = 16, lambda = 0 the truth is known exactly (exact/E_free = 1), so a knob
            # penalty is naturally quoted as a ratio of two distances from 1.  That ratio is
            # meaningless unless BOTH distances clear the route's own same-spec scatter: the
            # first triage pass quoted 328x and 1190x on a denominator (6.9e-5) that is 28x
            # BELOW the 0.199 % same-spec spread archived at that very spec, and re-running
            # the denominator leg at another thread count moved the headline by an order of
            # magnitude.  Every factor below is therefore the CONSERVATIVE one: the worse
            # leg's smallest distance over the better leg's largest, taken across every
            # archived thread count of each spec, with the same-spec band reported beside it.
            def _legs(r_):
                k = _knobs_but(r_, "threads")
                return [q for q in l16 if _knobs_but(q, "threads") == k]

            def _spread(rows_):
                e_ = [q["E_min"] for q in rows_]
                return ((max(e_) - min(e_)) / abs(min(e_, key=abs))) if len(e_) > 1 else None

            def _banded(a_, b_, axis, axis_better, axis_worse):
                la, lb = _legs(a_), _legs(b_)
                ea = {str(q.get("threads")): abs(q["ratio_exact_over_free_exact"] - 1.0) for q in la}
                eb = {str(q.get("threads")): abs(q["ratio_exact_over_free_exact"] - 1.0) for q in lb}
                better, worse = (a_, b_) if min(ea.values()) <= min(eb.values()) else (b_, a_)
                e_bet, e_wor = (ea, eb) if better is a_ else (eb, ea)
                l_bet, l_wor = (la, lb) if better is a_ else (lb, la)
                v_bet, v_wor = ((axis_better, axis_worse) if better is a_
                                else (axis_worse, axis_better))
                band_bet, band_wor = _spread(l_bet), _spread(l_wor)
                return {"axis": axis, "L": a_["L"], "n_max": a_["n_max"], "lam": a_["lam"],
                        "chi_op_requested": a_["chi_op_allowed"],
                        "tag_better": better.get("tag"), "tag_worse": worse.get("tag"),
                        "value_better": v_bet, "value_worse": v_wor,
                        "err_from_1_better_by_threads": e_bet,
                        "err_from_1_worse_by_threads": e_wor,
                        "err_better_max": max(e_bet.values()), "err_worse_min": min(e_wor.values()),
                        "same_spec_band_better": band_bet, "same_spec_band_worse": band_wor,
                        "n_threads_archived_better": len(e_bet), "n_threads_archived_worse": len(e_wor),
                        # the number that may be quoted
                        "factor_conservative": min(e_wor.values()) / max(e_bet.values()),
                        # what the first pass quoted: one thread-matched pair, no band
                        "factor_at_matched_threads": (e_wor[str(worse.get("threads"))]
                                                      / e_bet[str(better.get("threads"))]),
                        # the better leg is BELOW its own noise floor -> its distance from 1
                        # is not resolved and no ratio built on it alone may be quoted
                        # None where only one thread count is archived, so "no band
                        # measured" is never reported as "not resolved"
                        "better_leg_resolved_above_its_band":
                            (None if band_bet is None else bool(max(e_bet.values()) > band_bet)),
                        "worse_leg_resolved_above_its_band":
                            (None if band_wor is None else bool(min(e_wor.values()) > band_wor))}

            banded = []
            _by_tag = {q.get("tag"): q for q in l16}
            for z in svd_pairs:
                if z["L"] != QEI_L or abs(z["lam"]) > 1e-12:
                    continue
                a_, b_ = _by_tag.get(z["tag_sketched"]), _by_tag.get(z["tag_exact_svd"])
                if a_ is not None and b_ is not None:
                    banded.append(_banded(a_, b_, "svd", "sketched", "exact_svd"))
            for z in cost_pairs:
                if abs(z["lam"]) > 1e-12:
                    continue
                a_, b_ = _by_tag.get(z["tag_lo"]), _by_tag.get(z["tag_hi"])
                if a_ is not None and b_ is not None:
                    banded.append(_banded(a_, b_, "weight_decay", z["weight_lo"], z["weight_hi"]))
            banded.sort(key=lambda z: (z["axis"], z["n_max"], z["chi_op_requested"],
                                       str(z["tag_better"])))

            # ---- FIX ROUND 1: the wall model's calibration, per part ------------------
            # The first pass calibrated the RETIRED plane (2.3x) and quoted its REPLACEMENT
            # at nominal, which flatters the saving.  Here every landed job is divided by
            # its own plan estimate, so both sides carry the same calibration.
            wall_cal = {"note": ("measured time_s over the job's OWN est_wall_s (both carried "
                                 "in the archived spec, so retired parts are covered too); "
                                 "'plane' is the retired exact-SVD part and 'plane2' its "
                                 "sketched replacement on the same (n_max, chi_op) grid"),
                        "per_part": {}, "per_job": {}}
            _seen_cal = {}
            for _sp, _r in (_load_jobs("s8") + _load_jobs("s5g") + _load_jobs("s5h")):
                _t, _e = _r.get("time_s"), _sp.get("est_wall_s")
                if "error" in _r or not _t or not _e:
                    continue
                _seen_cal[_sp.get("tag")] = (_sp.get("part", "?"), _t, _e, _t / _e)
            for _tg, (_part, _t, _e, _ra) in sorted(_seen_cal.items()):
                wall_cal["per_part"].setdefault(_part, []).append(_ra)
                if _part in ("plane", "plane2"):
                    wall_cal["per_job"][_tg] = {"part": _part, "time_s": _t,
                                                "est_wall_s": _e, "ratio": _ra}
            wall_cal["per_part"] = {k: {"n": len(v), "median": float(np.median(v)),
                                        "min": min(v), "max": max(v)}
                                    for k, v in sorted(wall_cal["per_part"].items())}
            _p2m = wall_cal["per_part"].get("plane2", {}).get("median")
            _plm = wall_cal["per_part"].get("plane", {}).get("median")
            try:
                _plan_all = json.loads((DATA / "s8_plan.json").read_text())
                _p2_nom_h = _plan_all["est_by_part"].get("plane2", {}).get("est_wall_h")
                _p2_extrap_h = sum(j["est_wall_s"] for j in _plan_all["jobs"]
                                   if j["part"] == "plane2" and j["n_max"] > 8) / 3600.0
            except Exception:
                _p2_nom_h = _p2_extrap_h = None
            wall_cal["plane2_nominal_h"] = _p2_nom_h
            wall_cal["plane2_calibration"] = _p2m
            wall_cal["plane2_calibrated_h"] = (_p2_nom_h * _p2m) if (_p2_nom_h and _p2m) else None
            wall_cal["plane2_hours_that_are_wall_model_extrapolation"] = _p2_extrap_h
            wall_cal["plane_calibration"] = _plm
            wall_cal["plane_retired_remaining_nominal_h"] = 368.0
            wall_cal["plane_retired_remaining_calibrated_h"] = (368.0 * _plm) if _plm else None
            wall_cal["saving_factor_like_for_like"] = (
                (368.0 * _plm) / (_p2_nom_h * _p2m) if (_plm and _p2m and _p2_nom_h) else None)
            wall_cal["saving_factor_nominal"] = (368.0 / _p2_nom_h) if _p2_nom_h else None
            # the exact-SVD-specific excess, once the plane class's own sketched
            # under-estimate is divided out
            wall_cal["exact_svd_excess_over_sketched_plane_class"] = (
                (_plm / _p2m) if (_plm and _p2m) else None)
            # FROZEN at fix round 1.  The live medians above keep moving as the pool lands
            # more plane2 rows; the quoted comparison must not, so it is recomputed here
            # over an explicit tag set and is what the test pins.
            _FIX1 = {"plane": ("plane_nmax8_chiop32_lam0", "plane_nmax8_chiop32_lam4"),
                     "plane2": ("plane2_nmax8_chiop32_w0.5_lam0", "plane2_nmax8_chiop32_w0.5_lam2",
                                "plane2_nmax8_chiop32_w0.5_lam4", "plane2_nmax8_chiop32_w1_lam0",
                                "plane2_nmax8_chiop32_w1_lam2", "plane2_nmax8_chiop32_w1_lam4",
                                "plane2_nmax8_chiop64_w0.5_lam0")}
            _f1 = {"tags": {k: list(v) for k, v in _FIX1.items()}}
            for _k, _tags in _FIX1.items():
                _rr = [_seen_cal[t][3] for t in _tags if t in _seen_cal]
                _f1[_k] = ({"n": len(_rr), "median": float(np.median(_rr)),
                            "min": min(_rr), "max": max(_rr)} if _rr else None)
            _m2 = (_f1["plane2"] or {}).get("median")
            _m1 = (_f1["plane"] or {}).get("median")
            _f1["plane2_nominal_h"] = _p2_nom_h
            _f1["plane2_calibrated_h"] = (_p2_nom_h * _m2) if (_p2_nom_h and _m2) else None
            _f1["plane_retired_remaining_nominal_h"] = 368.0
            _f1["plane_retired_remaining_calibrated_h"] = (368.0 * _m1) if _m1 else None
            _f1["exact_svd_excess_over_sketched_plane_class"] = (_m1 / _m2) if (_m1 and _m2) else None
            _f1["saving_factor_like_for_like"] = (
                (368.0 * _m1) / (_p2_nom_h * _m2) if (_m1 and _m2 and _p2_nom_h) else None)
            _f1["saving_factor_nominal"] = (368.0 / _p2_nom_h) if _p2_nom_h else None
            _f1["plane2_hours_that_are_wall_model_extrapolation"] = _p2_extrap_h
            _f1["note"] = ("frozen tag set, so the quoted saving does not drift as the pool "
                           "lands more plane2 rows; every listed plane2 tag is n_max 8, the "
                           "largest Fock cutoff any sketched L = 16 job has reached")
            wall_cal["fix_round_1"] = _f1
            _dump("s8_weight_triage.json",
                  {"_source": "notebook_qei_exact.py::S8 wt (triage)", "date_utc": _now(),
                   "build_git_head": build, "python": platform.python_version(), "numpy": np.__version__,
                   "what": ("every MPO row at L <= 6 with an exact-ED partner at the same "
                            "(L, n_max, lam, trapezoid grid), ordered in weight_decay at fixed "
                            "chi_op and SVD mode; plus the L = 16 weight and SVD-mode cost pairs"),
                   "rows": wt_rows, "weight_ladders": ladders, "weight_ladders_multi": multi,
                   "best_weight_by_point": best_w, "l16_weight_cost_pairs": cost_pairs,
                   "repeat_runs_same_spec": repeats,
                   "worst_rel_spread_same_spec": (max(z["rel_spread_E_min"] for z in repeats)
                                                  if repeats else None),
                   "svd_mode_pairs": svd_pairs,
                   "banded_penalties_l16_lam0": banded,
                   "wall_model_calibration": wall_cal, "n_rows": len(wt_rows)})
            print(f"  weight triage: {len(wt_rows)} ED-anchored MPO rows, {len(multi)} weight ladders, "
                  f"{len(repeats)} same-spec repeat groups")
            for z in repeats:
                print(f"    REPEAT L{z['L']} n{z['n_max']} lam{z['lam']:g} chi_op{z['chi_op_requested']} "
                      f"w{z['weight_decay']:g} {z['svd']}: rel spread {z['rel_spread_E_min']:.3%}  " +
                      "  ".join(f"[{r_['threads']} thr] {r_['E_min']:.7e}" for r_ in z["runs"]), flush=True)
            for k, v in sorted(multi.items()):
                print(f"    {k}: " + "  ".join(f"w{r_[0]:g} {r_[1]:+.3%}" for r_ in v), flush=True)
        TIMES["S8"] = time.time() - t0

    _dump((f"s8_diagnostics_summary{S8_SUFFIX}.json" if S8_ONLY else "s5f_diagnostics_summary.json"),
          {"_source": "notebook_qei_exact.py::S5f-S5i" + ("+S8" if "s8" in ONLY else ""),
                                           "wall_time_s": time.time() - DIAG_T0,
                                           "section_times_s": TIMES, "sections": sorted(ONLY),
                                           "files": ["s5f_nmax_ladder_lam4.csv/json", "s5g_calibration.csv/json",
                                                     "s5h_lambda_sweep.csv/json", "s5i_ed_control.csv/json"]})
    print(f"\nDiagnostics wall time: {(time.time() - DIAG_T0) / 60:.1f} min", flush=True)
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# S0  provenance
# ---------------------------------------------------------------------------
_section("S0 provenance")
try:
    build = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
except Exception as exc:  # pragma: no cover
    build = f"git unavailable: {exc}"
import tenpy  # noqa: E402

build_info = {
    "_source": "notebook_qei_exact.py::S0", "smoke_mode": SMOKE, "date_utc": _now(),
    "build_git_head": build, "python": platform.python_version(), "platform": platform.platform(),
    "numpy": np.__version__, "tenpy": tenpy.__version__, "vacuum_version": vacuum.__version__,
    "parameters": {"MASS": MASS, "L": QEI_L, "TAU0": TAU0, "NMAX": NMAX, "CHI_OP": CHI_OP,
                   "CHI_ACC": CHI_ACC, "CHI_DMRG": CHI_DMRG, "WEIGHT_DECAY": WDECAY, "DT": DT,
                   "ORDER": ORDER, "PAD": PAD, "CHI_MPO": CHI_MPO, "SVD_REAL": SVD_REAL,
                   "LAMS": LAMS, "BREADTH_TAU0": BREADTH_TAU0,
                   "BREADTH_MASS": BREADTH_MASS, "BREADTH_LAMS": BREADTH_LAMS},
}
_dump("s5_build_info.json", build_info)
print(json.dumps(build_info["parameters"], indent=1), flush=True)

# ---------------------------------------------------------------------------
# S5a  the lambda = 0 anchor and what its residual is made of
# ---------------------------------------------------------------------------
_section("S5a lambda = 0 anchor: exact-MPO infimum vs the Gaussian (Williamson) infimum")
t0 = time.time()
f_main = gaussian_f(TAU0, n_sigmas=4.0)
E_free_main = free_exact_infimum(QEI_L, QEI_L // 2, f_main, MASS)
print(f"Gaussian exact infimum (Williamson, qei_minimize): {E_free_main:.10e}", flush=True)

anchor_rows = []
gs0 = phi4_ground_mps(QEI_L, MASS, 0.0, n_max=NMAX, chi_max=16)
# (chi_op, weight_decay, chi_mpo): the reference row, the UNWEIGHTED control at the
# same chi_op (what the occupation-weighted norm buys), and the chi_op ladder step
# (what is left to buy).  The chi_mpo cap is held fixed across the ladder so the rows
# differ in one knob at a time; row 1 was also run uncapped during development and
# moved by 2.5e-4 relative, so the cap is not what sets the band.
# (chi_op, weight_decay, chi_mpo).  Two rows: the reference, and the UNWEIGHTED
# control at the same chi_op -- which is what isolates the occupation-weighted norm's
# contribution to the band.  The chi_op ladder itself (8 / 16 / 24 / 32 at both
# weights) was run during development and is summarized in the qei_exact module
# docstring; it is not re-run here because a single chi_op = 24 row costs more than
# the whole rest of this notebook.
ladder = [(CHI_OP, WDECAY, CHI_MPO), (CHI_OP, 1.0, CHI_MPO)]
for chi_op, wd, chi_mpo in ladder:
    kw = dict(EXACT_KW, chi_op=chi_op, chi_acc=2 * chi_op, weight_decay=wd, chi_mpo=chi_mpo)
    res = qei_exact_mps(gs0, QEI_L // 2, f_main, **kw)
    dev = res.E_min - E_free_main
    r = _row(res, tau0=TAU0, dev_vs_gaussian=dev, rel_dev_vs_gaussian=dev / abs(E_free_main),
             E_gaussian_exact=E_free_main)
    anchor_rows.append(r)
    _csv("s5a_free_anchor_exact.csv", anchor_rows)
    print(f"  chi_op={chi_op:3d} wdecay={wd:<5} chi_mpo<={chi_mpo}: E_min {res.E_min:.8e}  dev {dev:+.3e} "
          f"({dev / abs(E_free_main):+.2e} rel) | trunc_op(w) {res.op.trunc_op:.1e} "
          f"norm_ratio {res.op.norm_ratio:.5f} chi_mpo {res.op.chi_mpo} var {res.variance:.1e} "
          f"[{res.dmrg['time_operator_s']:.0f}s op + {res.dmrg['time_dmrg_s']:.0f}s dmrg]", flush=True)

# The Trotter step and the trapezoid quadrature are NOT re-measured here: the
# regression test ``test_smeared_operator_equals_the_dense_heisenberg_integral``
# pins the assembled MPO against a dense exact-diagonalization build of the same
# integral (Gauss-Legendre in t, exact e^{-iHt}) to < 1e-6 relative, and the
# dt-ladder there moves E_min by < 1e-8 -- both errors are far below the MPO
# truncation this section is measuring.
ref = next(r for r in anchor_rows
           if r["chi_op_allowed"] == CHI_OP and r["weight_decay"] == WDECAY)
TIMES["S5a"] = time.time() - t0

# ---------------------------------------------------------------------------
# S5b  the exact infimum vs lambda, beside the variational number
# ---------------------------------------------------------------------------
_section("S5b the exact infimum vs lambda (m = 0.5, tau0 = 0.75) and the measured family gap")
t0 = time.time()
sweep_rows = []
for lam in LAMS:
    tl = time.time()
    gs = gs0 if lam == 0.0 else phi4_ground_mps(QEI_L, MASS, lam, n_max=NMAX, chi_max=16)
    res = qei_exact_mps(gs, QEI_L // 2, f_main, **EXACT_KW)
    # the variational family re-run at the SAME cutoff, so the gap is like-for-like
    var = qei_variational_mps(gs, QEI_L // 2, f_main, half_width=2,
                              s_grid=(0.75, 1.0, 1.25), dt=0.25, order=4,
                              chi_max=16, shape_mass="hartree", refine=False)
    # independent re-measurement of the extremal state in the Schroedinger picture
    chk = smeared_energy_mps(res.psi_star, gs.model, QEI_L // 2, f_main, MASS, lam,
                             dt=DT, order=4, chi_max=24, psi_ref=gs.psi,
                             reference="evolved")
    V = covariance_from_mps(res.psi_star)
    try:
        margin = assert_physical(V, tol=1e-5)
        physical = True
    except Exception as exc:                      # pragma: no cover - audit event
        margin, physical = float("nan"), False
        print(f"  !! physicality audit fired at lam={lam}: {exc}", flush=True)
    row = _row(res, tau0=TAU0, E_var=var.E_var, E_var_s_star=var.s_star,
               family_gap=var.E_var - res.E_min,
               family_quality=var.E_var / res.E_min,
               E_schroedinger_recheck=chk["E_f"],
               rel_dev_schroedinger=relative_change(res.E_min, chk["E_f"]),
               min_symplectic_margin=margin, physical=physical,
               time_s=time.time() - tl)
    sweep_rows.append(row)
    _csv("s5b_exact_vs_lambda.csv", sweep_rows)
    print(f"  lam={lam:<4}: E_exact {res.E_min:.6e}  E_var {var.E_var:.6e}  "
          f"gap {row['family_gap']:+.3e} (family reaches {row['family_quality']:.3f}) | "
          f"exact/free {row['ratio_exact_over_free_exact']:.4f}  "
          f"exact/hartree {row['ratio_exact_over_hartree_exact']:.4f} | "
          f"trunc(w) {res.op.trunc_op:.1e} recheck {row['rel_dev_schroedinger']:.1e} "
          f"[{time.time() - tl:.0f}s]", flush=True)
TIMES["S5b"] = time.time() - t0

# ---------------------------------------------------------------------------
# S5c  breadth: a second smearing width and a second mass
# ---------------------------------------------------------------------------
_section("S5c breadth: a second smearing width and a second mass")
t0 = time.time()
breadth_rows = []
for label, mass, tau0, lams in (("tau0", MASS, BREADTH_TAU0, BREADTH_LAMS),
                                ("mass", BREADTH_MASS, TAU0, BREADTH_LAMS[-1:])):
    f_b = gaussian_f(tau0, n_sigmas=4.0)
    dt_b = DT * (tau0 / TAU0)
    for lam in lams:
        tl = time.time()
        gs = phi4_ground_mps(QEI_L, mass, lam, n_max=NMAX, chi_max=16)
        res = qei_exact_mps(gs, QEI_L // 2, f_b, **dict(EXACT_KW, dt=dt_b))
        var = qei_variational_mps(gs, QEI_L // 2, f_b, half_width=2,
                                  s_grid=(0.75, 1.0, 1.25), dt=dt_b, order=4,
                                  chi_max=16, shape_mass="hartree", refine=False)
        row = _row(res, tau0=tau0, axis=label, E_var=var.E_var,
                   family_gap=var.E_var - res.E_min, family_quality=var.E_var / res.E_min,
                   time_s=time.time() - tl)
        breadth_rows.append(row)
        _csv("s5c_breadth.csv", breadth_rows)
        print(f"  [{label}] m={mass} tau0={tau0} lam={lam}: E_exact {res.E_min:.6e} "
              f"free {res.E_free_exact:.6e} hartree {res.E_hartree_exact:.6e} | "
              f"exact/free {row['ratio_exact_over_free_exact']:.4f} "
              f"exact/hartree {row['ratio_exact_over_hartree_exact']:.4f} "
              f"E_var {var.E_var:.6e} | trunc(w) {res.op.trunc_op:.1e} [{time.time() - tl:.0f}s]",
              flush=True)
TIMES["S5c"] = time.time() - t0

# ---------------------------------------------------------------------------
# S5e  IS THE HEADLINE CONVERGED?  the n_max x chi_op ladder
# ---------------------------------------------------------------------------
# S5a establishes the lambda = 0 residual at ONE cutoff.  That is not enough to
# quote a ratio at lambda > 0: both refinements move the interacting numbers, and
# they move them the SAME way (upward, toward the free-at-Hartree bound), so a
# single-cutoff ratio is biased low and its error is not the lambda = 0 band.
# This section measures that drift directly, at lambda = 0 (where the exact answer
# is known, so the ladder can be checked against it) and at lambda = 4 (where the
# headline ratio lives and the drift is largest), and Aitken-extrapolates the
# n_max sequence.  Without it the reported ratio would be an unconverged number
# presented as a converged one.
_section("S5e convergence of the headline ratio in n_max and chi_op")
t0 = time.time()
conv_rows = []
# n_max runs to 7, not 6.  Measured at lambda = 4: 0.8483, 0.8727, 0.8729, 0.9053
# for n_max = 4, 5, 6, 7 -- the 5 -> 6 step moves 2e-4 and the 6 -> 7 step 3e-2, so
# stopping at 6 (as an earlier draft did) reads as converged and is not.  That
# lambda = 4 sequence is strictly INCREASING, so the 5 -> 6 near-plateau is a bad
# stopping point, not a monotonicity violation.  The lambda = 0 control is the
# genuinely non-monotone one (0.9993 / 1.0127 / 1.0088): the exact dense infimum is
# monotone in n_max (Rayleigh-Ritz), so the untruncated ratio must be too, and that
# fall is what pins the wobble on the MPO truncation's n_max dependence -- which we
# have not characterized.  WHY either ladder is irregular is NOT explained, and it
# is not a phi^4 parity/shell effect (the phi_i phi_{i+1} bond breaks occupation
# parity, and lambda = 0 has no phi^4 term at all).  The operational point needs no
# mechanism: a near-plateau here cannot be read as convergence.
CONV_GRID = [(0.0, 5, 16), (0.0, 6, 16), (0.0, 7, 16), (0.0, 6, 24),
             (4.0, 4, 16), (4.0, 5, 16), (4.0, 6, 16), (4.0, 7, 16), (4.0, 6, 24)]
if SMOKE:
    CONV_GRID = [(0.0, 3, 8), (0.0, 4, 8), (1.0, 3, 8), (1.0, 4, 8)]
for lam_c, n_c, chi_c in CONV_GRID:
    tl = time.time()
    gs_c = phi4_ground_mps(QEI_L, MASS, lam_c, n_max=n_c, chi_max=16)
    res_c = qei_exact_mps(gs_c, QEI_L // 2, f_main,
                          **dict(EXACT_KW, chi_op=chi_c, chi_acc=2 * chi_c,
                                 chi_mpo=(CHI_MPO if chi_c <= CHI_OP else 96)))
    conv_rows.append(_row(res_c, tau0=TAU0, axis="convergence", time_s=time.time() - tl))
    _csv("s5e_convergence.csv", conv_rows)
    print(f"  lam={lam_c:<4} n_max={n_c} chi_op={chi_c:2d}: E_min {res_c.E_min:.6e} "
          f"exact/free {res_c.E_min / res_c.E_free_exact:.4f} "
          f"exact/hartree {res_c.E_min / res_c.E_hartree_exact:.4f} "
          f"trunc(w) {res_c.op.trunc_op:.1e} [{time.time() - tl:.0f}s]", flush=True)


def _aitken(seq):
    """Aitken delta-squared limit of three successive values (None if degenerate)."""
    x0, x1, x2 = seq
    den = (x2 - x1) - (x1 - x0)
    if abs(den) < 1e-15:
        return None
    return x2 - (x2 - x1) ** 2 / den


conv = {}
for lam_c in sorted({r["lam"] for r in conv_rows}):
    seq = [r for r in conv_rows if r["lam"] == lam_c and r["chi_op_allowed"] == CHI_OP]
    seq.sort(key=lambda r: r["n_max"])
    ratios = [r["ratio_exact_over_hartree_exact"] for r in seq]
    # the chi_op comparison must be made at the SAME n_max, or it silently mixes
    # the two axes (the n_max = 7 row is not a chi_op control for the n_max = 6 one)
    hi = [r for r in conv_rows if r["lam"] == lam_c and r["chi_op_allowed"] > CHI_OP]
    base_for_hi = {r["n_max"]: r["ratio_exact_over_hartree_exact"]
                   for r in conv_rows
                   if r["lam"] == lam_c and r["chi_op_allowed"] == CHI_OP}
    entry = {"n_max": [r["n_max"] for r in seq], "ratio_hartree_vs_n_max": ratios,
             "drift_n_max": (ratios[-1] - ratios[0]) if len(ratios) >= 2 else None,
             # reported for completeness only: the n_max sequences are irregular (a
             # near-plateau at lambda = 4, genuinely non-monotone at lambda = 0; cause
             # unexplained -- see the CONV_GRID note), so Aitken, which assumes a smooth
             # geometric rate, has no footing and must not be quoted as the converged
             # value (it returns ~0.873 on a sequence whose next term is 0.905).  The
             # honest summary is the last value as a LOWER BOUND plus the drift.
             "aitken_ratio_hartree_unreliable": _aitken(ratios[-3:]) if len(ratios) >= 3 else None,
             "lower_bound_ratio_hartree": ratios[-1] if ratios else None,
             "ratio_hartree_at_2chi_op": (hi[0]["ratio_exact_over_hartree_exact"] if hi else None),
             "chi_op_control_n_max": (hi[0]["n_max"] if hi else None),
             "drift_chi_op": ((hi[0]["ratio_exact_over_hartree_exact"]
                               - base_for_hi[hi[0]["n_max"]])
                              if hi and hi[0]["n_max"] in base_for_hi else None)}
    conv[f"lam={lam_c:g}"] = entry
    print(f"  lam={lam_c:g}: ratio vs n_max {['%.4f' % x for x in ratios]} "
          f"(drift {entry['drift_n_max'] if entry['drift_n_max'] is None else '%+.4f' % entry['drift_n_max']}), "
          f"lower bound "
          f"{entry['lower_bound_ratio_hartree'] if entry['lower_bound_ratio_hartree'] is None else '%.4f' % entry['lower_bound_ratio_hartree']}, "
          f"chi_op drift (at n_max={entry['chi_op_control_n_max']}) "
          f"{entry['drift_chi_op'] if entry['drift_chi_op'] is None else '%+.4f' % entry['drift_chi_op']}",
          flush=True)
_dump("s5e_convergence.json", {"_source": "notebook_qei_exact.py::S5e", "per_lambda": conv,
                               "grid": CONV_GRID})
TIMES["S5e"] = time.time() - t0

# ---------------------------------------------------------------------------
# S5d  summary and the anomaly-protocol verdict
# ---------------------------------------------------------------------------
_section("S5d summary")
anchor_ref = ref
band = abs(anchor_ref["rel_dev_vs_gaussian"])
events = []
for r in sweep_rows + breadth_rows:
    # an infimum below the free exact one at the same f/m/lattice would be the
    # headline event; here it must clear the lambda = 0 calibration band first
    if r["E_min"] < r["E_free_exact"] * (1.0 + band) - 1e-12 and r["lam"] > 0:
        events.append({"lam": r["lam"], "tau0": r["tau0"], "E_min": r["E_min"],
                       "E_free_exact": r["E_free_exact"],
                       "event": "exact infimum below the free exact infimum by more than "
                                "the lambda = 0 calibration band",
                       "status": "ANOMALY CANDIDATE - suspect chi_op first (PLAN.md protocol)"})
    if r.get("E_var") is not None and r["E_min"] > r["E_var"] + 1e-12:
        events.append({"lam": r["lam"], "tau0": r["tau0"],
                       "event": "the 'infimum' exceeds a variational upper bound",
                       "E_min": r["E_min"], "E_var": r["E_var"],
                       "status": "ANOMALY CANDIDATE - the exact route is not an infimum"})

summary = {
    "_source": "notebook_qei_exact.py::S5d",
    "build_git_head": build,
    "wall_time_s": time.time() - T_START,
    "section_times_s": TIMES,
    "gaussian_exact_infimum": E_free_main,
    "anchor_calibration_band_relative": band,
    "anchor": anchor_rows,
    "sweep": {f"lam={r['lam']:g}": {k: r[k] for k in (
        "E_min", "E_var", "family_gap", "family_quality", "E_free_exact", "E_hartree_exact",
        "ratio_exact_over_free_exact", "ratio_exact_over_hartree_exact",
        "trunc_op_weighted", "rel_dev_schroedinger", "min_symplectic_margin",
        "mpo_variance")} for r in sweep_rows},
    "breadth": breadth_rows,
    "convergence": conv,
    "convergence_rows": conv_rows,
    "anomaly_events": events,
    "headline_is_converged": False,
    "headline_note": (
        "exact/Hartree at lambda > 0 is NOT converged: it drifts upward under both "
        "n_max and chi_op refinement (see 'convergence'). Quote it as a lower bound "
        "with the drift and the Aitken extrapolation, never as a converged value."),
}
_dump("s5_summary.json", summary)
print(f"\nlambda = 0 calibration band: {band:.2e} relative "
      f"(an anomaly THRESHOLD, not an error bar propagated onto the ratios -- "
      f"the lambda > 0 systematic is the S5e drift, which is larger)")
print(f"anomaly events: {len(events)}")
print(f"Total wall time: {summary['wall_time_s']:.1f} s ({summary['wall_time_s'] / 60:.1f} min)")
