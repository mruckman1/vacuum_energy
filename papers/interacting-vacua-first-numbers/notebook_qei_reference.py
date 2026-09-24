"""papers/interacting-vacua-first-numbers — the sharpest free reference (S7).

The question
------------
``notebook_qei_exact.py`` compares the exact lattice infimum E_min(lambda) of
the smeared energy with the free chain's Williamson infimum at the Hartree
mass, E_H.  (Its archived reading at the time this notebook was written --
"exact/E_H >= 0.905 at lambda = 4, residual <= ~10 %, not resolved from zero"
-- is RETRACTED by diagnostics A, outline.md section 6.7 item 7: the infimum is
not monotone in n_max and the archived knobs used here inflate the lambda = 4
residual, so no lambda = 4 value or bound is quoted; what stands is a
first-order residual, resolved at lambda <= 1.  Every number this notebook
produces is at those archived knobs; see the supersession banner of
draft/section_reference.md for what survives.)  The Hartree line
matches the interacting theory at one point of the band (the gap).  A
reviewer's objection: interactions renormalize the whole dispersion omega(k),
and the free infimum depends on all of it, so the reference is wrong, not
just the mass.  This notebook builds the sharpest quadratic reference and
decomposes the extremal state against it (``vacuum.interacting.qei_reference``).

Sections
--------
S7a  the interacting dispersion omega(k) at lambda = 0 (calibration), 1, 4 from
     TEBD correlators + harmonic inversion on the chain's standing waves; the
     quasiparticle peak's spectral-weight fraction, decay rate and windowed
     FWHM per k; the gap cross-checked by excited-state DMRG
S7b  K_eff from omega(k)^2 (non-negative cosine series, PSD local density) and
     the free Williamson infimum at K_eff with the identical sampling, beside
     the Hartree reference: exact/E_free(K_eff) next to exact/E_H, with the
     lambda = 0 control at the same cutoff
S7c  the four normal-ordered pieces of the smeared energy on the extremal
     state (kinetic, gradient, mass, quartic), in the Heisenberg picture (the
     audit: they sum to the infimum up to the operator truncation) and in the
     Schroedinger picture; the sign of the quartic term
S7d  the tau0 sweep at lambda = 4 with lambda = 0 controls: residual fraction
     against both references vs tau0, its power-law exponent against the
     reviewer's 1/tau0^2 prediction, and the quartic fraction vs tau0
S7e  controls: an L = 24 chain (the extremal state's support against the
     boundary) and a second lattice spacing a = 1/2 at fixed physical smearing
     (L = 32, m = 0.25, lambda = 1, tau0 = 1.5), with its own dispersion
S7s  summary (reads the section files; recomputes the K_eff references)

Run
---
    .venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_reference.py
    .venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_reference.py --quick
    ... --only a      (any subset of a,b,c,d,e,s; sections are independent
                       except that s reads the others' files)

Rows are written to ``data/`` incrementally (s7 prefix).  Nothing here
modifies any ``vacuum.*`` module.
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
warnings.filterwarnings("ignore", message=".*orthogonal_to.*")
logging.getLogger("tenpy").setLevel(logging.ERROR)

import vacuum  # noqa: E402
from vacuum.inequalities.qei import gaussian_f  # noqa: E402
from vacuum.interacting import hartree_mass_squared, phi4_ground_mps  # noqa: E402
from vacuum.interacting.qei_exact import free_exact_infimum, qei_exact_mps  # noqa: E402
import vacuum.interacting.qei_reference as qr  # noqa: E402

_ap = argparse.ArgumentParser(description="interacting-vacua: the sharpest free reference")
_ap.add_argument("--quick", action="store_true", help="~3 min smoke configuration, scratch output")
_ap.add_argument("--out", default=None, help="output directory (default: data/)")
_ap.add_argument("--only", default="abcdes", help="sections to run (subset of abcdes)")
_ap.add_argument("--lams", default=None,
                 help="S7a only: comma-separated subset of the lambda grid (parallel runs write tagged files)")
_args, _ = _ap.parse_known_args()

HERE = Path(__file__).resolve().parent
SMOKE = os.environ.get("NOTEBOOK_SMOKE", "") == "1" or bool(_args.quick)
if _args.out:
    DATA = Path(_args.out)
elif os.environ.get("NOTEBOOK_DATA_DIR"):
    DATA = Path(os.environ["NOTEBOOK_DATA_DIR"])
elif SMOKE:
    DATA = Path(tempfile.mkdtemp(prefix="interacting-vacua-reference-quick-"))
else:
    DATA = HERE / "data"
DATA.mkdir(exist_ok=True, parents=True)
ONLY = set(_args.only)

# ---- the archived point and the archived exact-route knobs (notebook_qei_exact.py) ----
MASS = 0.5
QEI_L = 16
TAU0 = 0.75
NMAX = 6
CHI_OP, CHI_ACC, CHI_DMRG, CHI_MPO = 16, 32, 12, 64
WDECAY, DT, ORDER, PAD, SVD_REAL = 0.25, 0.375, 4, 2, 1e-6
LAMS = [0.0, 1.0, 4.0]
# dispersion knobs
DISP_T, DISP_DT, DISP_CHI = 10.0, 0.1, 32   # chi 32: phi_x|Omega> stays low-entangled (trunc_err reported)
DISP_T_FINE = 16.0            # a = 1/2: frequencies halve in lattice units, so the trajectory lengthens
R_MAX = 4                     # range of the non-negative cosine fit of omega^2(k)
# tau0 sweep (lambda = 4 with lambda = 0 controls); TAU0 itself comes from S7b
TAU0_SWEEP = [0.375, 0.5, 1.0, 1.5]
# controls
L_CONTROL = 24
FINE = dict(L=32, m=0.25, lam=1.0, tau0=1.5)   # a = 1/2: m a, lambda a^2, tau0 / a
if SMOKE:
    QEI_L, TAU0, NMAX, CHI_OP, CHI_ACC, CHI_DMRG, CHI_MPO = 8, 0.3, 4, 12, 24, 10, 48
    DT, LAMS = 0.15, [0.0, 1.0]
    DISP_T, DISP_DT, DISP_CHI = 6.0, 0.1, 24
    DISP_T_FINE = 6.0
    TAU0_SWEEP = [0.2, 0.45]
    L_CONTROL = 10
    FINE = dict(L=12, m=0.25, lam=0.25, tau0=0.6)

OP_KW = dict(dt=DT, order=ORDER, chi_op=CHI_OP, svd_min_op=1e-12, chi_acc=CHI_ACC,
             svd_min_acc=1e-12, window_pad=PAD, svd_min_real=SVD_REAL, chi_mpo=CHI_MPO,
             weight_decay=WDECAY)
EXACT_KW = dict(OP_KW, chi_dmrg=CHI_DMRG, max_sweeps=8, max_E_err=1e-10)

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
    if isinstance(o, complex):
        return [o.real, o.imag]
    raise TypeError(f"not serializable: {type(o)}")


def _dump(name, obj):
    with open(DATA / name, "w") as fh:
        json.dump(obj, fh, indent=1, default=_jdefault)


def _load(name):
    with open(DATA / name) as fh:
        return json.load(fh)


def _csv(name, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(DATA / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _read_csv(name):
    with open(DATA / name) as fh:
        return list(csv.DictReader(fh))


def _section(label):
    print(f"\n=== {label} ===", flush=True)


def _op_kw(dt):
    return dict(OP_KW, dt=dt)


def _exact_kw(dt):
    return dict(EXACT_KW, dt=dt)


def _dt_for(tau0):
    """The archived step for tau0 <= TAU0 scales with tau0 (8 steps); larger tau0 keeps DT."""
    return DT * min(1.0, tau0 / TAU0)


def _ground(L, m, lam, n_max=NMAX):
    return phi4_ground_mps(L, m, lam, n_max=n_max, chi_max=16)


def _save_mps(name, psi, extra=None):
    """Archive an MPS as its B tensors and Schmidt values (plain npz)."""
    arrays = {f"B{i}": psi.get_B(i, form="B").to_ndarray() for i in range(psi.L)}
    for i in range(psi.L + 1):
        arrays[f"S{i}"] = np.asarray(psi.get_SL(i) if i < psi.L else psi.get_SR(psi.L - 1))
    arrays["norm"] = np.asarray(psi.norm)
    for k, v in (extra or {}).items():
        arrays[k] = np.asarray(v)
    np.savez(DATA / name, **arrays)


def _disp_row(d, lam, gap=None, gap_info=None, label=""):
    return {"label": label, "lam": lam, "L": d.meta["L"], "m": d.meta["m"], "n_max": d.meta["n_max"],
            "site": d.meta["site"], "T": d.meta["T"], "dt": d.meta["dt"], "chi_max": d.meta["chi_max"],
            "chi": d.chi, "trunc_err": d.trunc_err, "E0": d.E0,
            "gap_dmrg": gap, "gap_dmrg_overlap_with_ground": (None if gap_info is None else gap_info["overlap_with_ground"]),
            "omega_1": d.omega[0], "gap_check_dev": (None if gap is None else d.omega[0] - gap),
            "weight_min": float(np.min(d.weight)), "weight_at_k1": d.weight[0],
            "gamma_max_abs": float(np.max(np.abs(d.gamma))), "gamma_over_gap_max": float(np.max(np.abs(d.gamma)) / d.omega[0]),
            "fwhm_ratio_max": float(np.max(d.fwhm / d.fwhm_window)),
            "fwhm_over_gap_max": float(np.max(d.fwhm) / d.omega[0]),
            "resid_max": float(np.max(d.resid)),
            "max_dev_from_free": float(np.max(np.abs(d.omega - d.omega_free))),
            "max_dev_from_hartree": float(np.max(np.abs(d.omega - d.omega_hartree))),
            "rms_dev_from_hartree": float(np.sqrt(np.mean((d.omega - d.omega_hartree) ** 2)))}


def _disp_json(d, lam, gap, gap_info):
    return {"lam": lam, "k": d.k, "omega": d.omega, "gamma": d.gamma, "weight": d.weight,
            "amplitude": d.amplitude, "n_poles": d.n_poles, "resid": d.resid, "fwhm": d.fwhm,
            "fwhm_window": d.fwhm_window, "omega_free": d.omega_free, "omega_hartree": d.omega_hartree,
            "poles": [{k: [[complex(z).real, complex(z).imag] for z in p[k]] for k in ("omega", "gamma", "A")}
                      for p in d.poles],
            "gap_dmrg": gap, "gap_info": gap_info, "meta": d.meta, "E0": d.E0, "trunc_err": d.trunc_err,
            "chi": d.chi}


def _keff_reference(L, site, f, k, omega, weight, r_max=R_MAX):
    """(mu2, J, fit info, E_keff, QEIResult) from a measured dispersion."""
    mu2, J, info = qr.fit_dispersion_couplings(k, np.asarray(omega) ** 2, r_max=r_max, weights=weight)
    res, _op = qr.free_infimum_keff(L, site, f, mu2, J)
    mu2_u, J_u = qr.cosine_coefficients(k, np.asarray(omega) ** 2)
    return {"mu2_eff": mu2, "J_eff": J, "misfit_max": info["misfit_max"], "misfit_rms": info["misfit_rms"],
            "mu2_unconstrained": mu2_u, "J_unconstrained": J_u,
            "J_unconstrained_min": float(np.min(J_u)), "E_keff": res.e_min, "keff_gap": res.gap}


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
    "_source": "notebook_qei_reference.py::S0", "smoke_mode": SMOKE, "date_utc": _now(),
    "sections": sorted(ONLY),
    "build_git_head": build, "python": platform.python_version(), "platform": platform.platform(),
    "numpy": np.__version__, "tenpy": tenpy.__version__, "vacuum_version": vacuum.__version__,
    "parameters": {"MASS": MASS, "L": QEI_L, "TAU0": TAU0, "NMAX": NMAX, "CHI_OP": CHI_OP,
                   "CHI_ACC": CHI_ACC, "CHI_DMRG": CHI_DMRG, "WEIGHT_DECAY": WDECAY, "DT": DT,
                   "ORDER": ORDER, "PAD": PAD, "CHI_MPO": CHI_MPO, "SVD_REAL": SVD_REAL,
                   "LAMS": LAMS, "DISP_T": DISP_T, "DISP_DT": DISP_DT, "DISP_CHI": DISP_CHI,
                   "DISP_T_FINE": DISP_T_FINE, "R_MAX": R_MAX, "TAU0_SWEEP": TAU0_SWEEP,
                   "L_CONTROL": L_CONTROL, "FINE": FINE},
}
_dump(f"s7_build_info_{''.join(sorted(ONLY))}.json", build_info)
print(json.dumps(build_info["parameters"], indent=1), flush=True)

# ---------------------------------------------------------------------------
# S7a  the interacting dispersion
# ---------------------------------------------------------------------------
if "a" in ONLY:
    _section("S7a the interacting dispersion omega(k) by TEBD correlators + harmonic inversion")
    t0 = time.time()
    rows, dump = [], {}
    lams_a = LAMS if _args.lams is None else [float(v) for v in _args.lams.split(",")]
    tag = "" if _args.lams is None else "_lam" + "_".join(f"{v:g}" for v in lams_a)
    for lam in lams_a:
        tl = time.time()
        gs = _ground(QEI_L, MASS, lam)
        d = qr.interacting_dispersion_tebd(gs, T=DISP_T, dt=DISP_DT, order=4, chi_max=DISP_CHI)
        gap, ginfo = qr.excitation_gap_dmrg(gs, chi_max=16)
        rows.append(_disp_row(d, lam, gap, ginfo, label="archived_lattice"))
        rows[-1]["time_s"] = time.time() - tl
        dump[f"lam={lam:g}"] = _disp_json(d, lam, gap, ginfo)
        _csv(f"s7a_dispersion{tag}.csv", rows)
        _dump(f"s7a_dispersion{tag}.json", {"_source": "notebook_qei_reference.py::S7a", "per_lambda": dump})
        print(f"  lam={lam:<4}: omega(k) {np.array2string(d.omega, precision=4)}", flush=True)
        print(f"            hartree  {np.array2string(d.omega_hartree, precision=4)}", flush=True)
        print(f"            weight   {np.array2string(d.weight, precision=3)}  gamma/gap max {rows[-1]['gamma_over_gap_max']:.1e}  "
              f"fwhm ratio max {rows[-1]['fwhm_ratio_max']:.3f}", flush=True)
        print(f"            gap DMRG {gap:.6f} vs omega_1 {d.omega[0]:.6f} (dev {d.omega[0] - gap:+.1e}); "
              f"max dev from free {rows[-1]['max_dev_from_free']:.1e}, from Hartree {rows[-1]['max_dev_from_hartree']:.1e} "
              f"[{time.time() - tl:.0f}s, chi {d.chi}, trunc {d.trunc_err:.1e}]", flush=True)
    TIMES["S7a"] = time.time() - t0

# ---------------------------------------------------------------------------
# S7b  the exact infimum beside both references (the archived point)
# ---------------------------------------------------------------------------
if "b" in ONLY:
    _section("S7b the exact infimum at the archived point (regenerated) beside the Hartree reference")
    t0 = time.time()
    f_main = gaussian_f(TAU0, n_sigmas=4.0)
    rows = []
    for lam in LAMS:
        tl = time.time()
        gs = _ground(QEI_L, MASS, lam)
        res = qei_exact_mps(gs, QEI_L // 2, f_main, **EXACT_KW)
        rows.append({"lam": lam, "L": QEI_L, "m": MASS, "tau0": TAU0, "n_max": NMAX, "chi_op": CHI_OP,
                     "E_min": res.E_min, "E_free_exact": res.E_free_exact, "E_hartree_exact": res.E_hartree_exact,
                     "mu_H2": res.mu_H2, "ratio_exact_over_free": res.E_min / res.E_free_exact,
                     "ratio_exact_over_hartree": res.E_min / res.E_hartree_exact,
                     "trunc_op_weighted": res.op.trunc_op, "mpo_variance": res.variance,
                     "dmrg_sweeps": res.dmrg["sweeps"], "time_s": time.time() - tl})
        _csv("s7b_exact_regenerated.csv", rows)
        _save_mps(f"s7_extremal_mps_lam{lam:g}.npz", res.psi_star,
                  extra={"E_min": res.E_min, "E_star": res.E_star, "E_ref": res.E_ref, "lam": lam, "m": MASS,
                         "tau0": TAU0, "n_max": NMAX, "L": QEI_L, "site": QEI_L // 2})
        _save_mps(f"s7_vacuum_mps_lam{lam:g}.npz", gs.psi, extra={"E0": gs.E0})
        print(f"  lam={lam:<4}: E_min {res.E_min:.6e}  exact/free {rows[-1]['ratio_exact_over_free']:.4f}  "
              f"exact/hartree {rows[-1]['ratio_exact_over_hartree']:.4f}  [{time.time() - tl:.0f}s]", flush=True)
    TIMES["S7b"] = time.time() - t0

# ---------------------------------------------------------------------------
# S7c  the decomposition on the extremal state
# ---------------------------------------------------------------------------
if "c" in ONLY:
    _section("S7c the four normal-ordered pieces on the extremal state (Heisenberg + Schroedinger)")
    t0 = time.time()
    f_main = gaussian_f(TAU0, n_sigmas=4.0)
    rows = []
    for lam in LAMS:
        tl = time.time()
        gs = _ground(QEI_L, MASS, lam)
        res = qei_exact_mps(gs, QEI_L // 2, f_main, **EXACT_KW)
        dec = qr.decompose_smeared_energy(res, gs, f_main, **OP_KW)
        sch = qr.schroedinger_pieces(res.psi_star, gs.model, QEI_L // 2, f_main, gs.psi,
                                     dt=DT, order=4, chi_max=24)
        row = {"lam": lam, "tau0": TAU0, "L": QEI_L, "E_min": res.E_min,
               **{f"H_{k}": v for k, v in dec["pieces"].items()},
               "H_sum": dec["sum"], "H_audit": dec["audit"], "H_audit_rel": dec["audit_rel"],
               **{f"H_frac_{k}": v / res.E_min for k, v in dec["pieces"].items()},
               **{f"S_{k}": sch[k] for k in qr.PIECES}, "S_sum": sch["sum"],
               "S_sum_rel_dev": (sch["sum"] - res.E_min) / abs(res.E_min),
               **{f"S_frac_{k}": sch[k] / res.E_min for k in qr.PIECES},
               "quartic_sign": ("+" if dec["pieces"]["quartic"] > 0 else "-") if lam else "n/a",
               "phi2_star": dec["phi2_star"], "phi2_vac": dec["phi2_vac"],
               "pi2_star": dec["pi2_star"], "pi2_vac": dec["pi2_vac"],
               "phi4_star": dec["phi4_star"], "phi4_vac": dec["phi4_vac"],
               **{f"trunc_op_{k}": v for k, v in dec["trunc_op"].items()},
               "S_trunc_err": sch["trunc_err"], "time_s": time.time() - tl}
        rows.append(row)
        _csv("s7c_decomposition.csv", rows)
        print(f"  lam={lam:<4}: E_min {res.E_min:.6e} | kin {dec['pieces']['kinetic']:+.4e} grad {dec['pieces']['gradient']:+.4e} "
              f"mass {dec['pieces']['mass']:+.4e} quart {dec['pieces']['quartic']:+.4e} | sum {dec['sum']:.6e} "
              f"(audit {dec['audit_rel']:+.2e}) | Schroedinger sum {sch['sum']:.6e} ({row['S_sum_rel_dev']:+.2e}) | "
              f"<phi^2> {dec['phi2_star']:.4f} vs vac {dec['phi2_vac']:.4f} [{time.time() - tl:.0f}s]", flush=True)
    TIMES["S7c"] = time.time() - t0

# ---------------------------------------------------------------------------
# S7d  the tau0 sweep at lambda = 4 with lambda = 0 controls
# ---------------------------------------------------------------------------
if "d" in ONLY:
    _section("S7d the tau0 sweep: residual fraction and the quartic fraction vs tau0")
    t0 = time.time()
    rows = []
    lam_hi = LAMS[-1]
    gs4 = _ground(QEI_L, MASS, lam_hi)
    gs0 = _ground(QEI_L, MASS, 0.0)
    for tau0 in TAU0_SWEEP:
        tl = time.time()
        f_t = gaussian_f(tau0, n_sigmas=4.0)
        dt_t = _dt_for(tau0)
        res4 = qei_exact_mps(gs4, QEI_L // 2, f_t, **_exact_kw(dt_t))
        res0 = qei_exact_mps(gs0, QEI_L // 2, f_t, **_exact_kw(dt_t))
        dec = qr.decompose_smeared_energy(res4, gs4, f_t, **_op_kw(dt_t))
        control = res0.E_min / res0.E_free_exact
        row = {"lam": lam_hi, "tau0": tau0, "dt": dt_t, "n_steps": res4.op.meta["n_steps"], "L": QEI_L,
               "E_min": res4.E_min, "E_min_lam0": res0.E_min, "E_free_exact": res4.E_free_exact,
               "E_hartree_exact": res4.E_hartree_exact,
               "ratio_exact_over_free": res4.E_min / res4.E_free_exact,
               "ratio_exact_over_hartree": res4.E_min / res4.E_hartree_exact,
               "control_lam0": control,
               "ratio_hartree_normalized": res4.E_min / res4.E_hartree_exact / control,
               "residual_hartree_normalized": 1.0 - res4.E_min / res4.E_hartree_exact / control,
               **{f"H_{k}": v for k, v in dec["pieces"].items()},
               "H_sum": dec["sum"], "H_audit_rel": dec["audit_rel"],
               **{f"H_frac_{k}": v / res4.E_min for k, v in dec["pieces"].items()},
               "phi2_star": dec["phi2_star"], "phi2_vac": dec["phi2_vac"],
               "trunc_op_weighted": res4.op.trunc_op, "trunc_op_weighted_lam0": res0.op.trunc_op,
               "norm_ratio": res4.op.norm_ratio, "time_s": time.time() - tl}
        rows.append(row)
        _csv("s7d_tau0_sweep.csv", rows)
        print(f"  tau0={tau0:<5}: E_min {res4.E_min:.5e} (lam0 {res0.E_min:.5e}, control {control:.4f}) | "
              f"exact/H {row['ratio_exact_over_hartree']:.4f} -> normalized {row['ratio_hartree_normalized']:.4f} "
              f"(residual {row['residual_hartree_normalized']:+.4f}) | quartic frac {row['H_frac_quartic']:+.4f} "
              f"(audit {dec['audit_rel']:+.1e}) [{time.time() - tl:.0f}s]", flush=True)
    TIMES["S7d"] = time.time() - t0

# ---------------------------------------------------------------------------
# S7e  controls: L = 24 and a second lattice spacing
# ---------------------------------------------------------------------------
if "e" in ONLY:
    _section("S7e controls: the extremal state's support (L sweep) and a second lattice spacing")
    t0 = time.time()
    f_main = gaussian_f(TAU0, n_sigmas=4.0)
    lam_hi = LAMS[-1]
    rows = []
    # (i) L sweep: the extremal state's footprint against the boundary
    for L in (QEI_L, L_CONTROL):
        tl = time.time()
        gs = _ground(L, MASS, lam_hi)
        res = qei_exact_mps(gs, L // 2, f_main, **EXACT_KW)
        gs0 = _ground(L, MASS, 0.0)
        res0 = qei_exact_mps(gs0, L // 2, f_main, **EXACT_KW)
        dx = np.real(np.asarray(res.psi_star.expectation_value("Xsq"))) - np.real(np.asarray(gs.psi.expectation_value("Xsq")))
        dp = np.real(np.asarray(res.psi_star.expectation_value("Psq"))) - np.real(np.asarray(gs.psi.expectation_value("Psq")))
        dS = np.asarray(res.psi_star.entanglement_entropy()) - np.asarray(gs.psi.entanglement_entropy())
        foot = np.abs(dx) + np.abs(dp)
        # The DMRG ground state of the compressed O_f is defined only on the sampled
        # light cone: outside it O_f is the identity up to the 1e-6 compression
        # residue, so those directions are flat and the sweeps leave arbitrary content
        # there (the archived extremal MPS carries O(1) excitations at the cone edge).
        # The physical footprint is therefore read on the inner cone |j - x| <= 2 and
        # the boundary question is settled by the eigenvalue's L-dependence, not by
        # the tail of the state.
        j = np.arange(L)
        inner = np.abs(j - L // 2) <= 2
        rows.append({"control": "L_sweep", "L": L, "m": MASS, "lam": lam_hi, "tau0": TAU0, "site": L // 2,
                     "footprint_inner_max": float(np.max(foot[inner])),
                     "dphi2_inner": json.dumps([float(v) for v in dx[inner]]),
                     "dpi2_inner": json.dumps([float(v) for v in dp[inner]]),
                     "E_min": res.E_min, "E_min_lam0": res0.E_min, "E_free_exact": res.E_free_exact,
                     "E_hartree_exact": res.E_hartree_exact,
                     "ratio_exact_over_hartree": res.E_min / res.E_hartree_exact,
                     "control_lam0": res0.E_min / res0.E_free_exact,
                     "ratio_hartree_normalized": res.E_min / res.E_hartree_exact / (res0.E_min / res0.E_free_exact),
                     "footprint_center": float(foot[L // 2]),
                     "footprint_edge_max": float(max(foot[0], foot[-1])),
                     "footprint_edge_over_center": float(max(foot[0], foot[-1]) / foot[L // 2]),
                     "footprint_profile": json.dumps([float(v) for v in foot]),
                     "dS_edge_max": float(max(abs(dS[0]), abs(dS[-1]))), "dS_max": float(np.max(np.abs(dS))),
                     "trunc_op_weighted": res.op.trunc_op, "time_s": time.time() - tl})
        _csv("s7e_controls.csv", rows)
        print(f"  L={L}: E_min {res.E_min:.6e} exact/H {rows[-1]['ratio_exact_over_hartree']:.4f} "
              f"(normalized {rows[-1]['ratio_hartree_normalized']:.4f}) | footprint edge/center "
              f"{rows[-1]['footprint_edge_over_center']:.1e} dS edge {rows[-1]['dS_edge_max']:.1e} [{time.time() - tl:.0f}s]",
              flush=True)
    # (ii) a = 1/2 at fixed physical smearing: m a, lambda a^2, tau0 / a, L / a
    tl = time.time()
    Lf, mf, lamf, tauf = FINE["L"], FINE["m"], FINE["lam"], FINE["tau0"]
    f_f = gaussian_f(tauf, n_sigmas=4.0)
    dt_f = DT
    gsf = _ground(Lf, mf, lamf)
    resf = qei_exact_mps(gsf, Lf // 2, f_f, **_exact_kw(dt_f))
    gsf0 = _ground(Lf, mf, 0.0)
    resf0 = qei_exact_mps(gsf0, Lf // 2, f_f, **_exact_kw(dt_f))
    decf = qr.decompose_smeared_energy(resf, gsf, f_f, **_op_kw(dt_f))
    controlf = resf0.E_min / resf0.E_free_exact
    rowf = {"control": "lattice_spacing_half", "L": Lf, "m": mf, "lam": lamf, "tau0": tauf, "site": Lf // 2,
            "dt": dt_f, "E_min": resf.E_min, "E_min_lam0": resf0.E_min, "E_free_exact": resf.E_free_exact,
            "E_hartree_exact": resf.E_hartree_exact, "mu_H2": resf.mu_H2,
            "ratio_exact_over_free": resf.E_min / resf.E_free_exact,
            "ratio_exact_over_hartree": resf.E_min / resf.E_hartree_exact, "control_lam0": controlf,
            "ratio_hartree_normalized": resf.E_min / resf.E_hartree_exact / controlf,
            **{f"H_{k}": v for k, v in decf["pieces"].items()}, "H_audit_rel": decf["audit_rel"],
            **{f"H_frac_{k}": v / resf.E_min for k, v in decf["pieces"].items()},
            "trunc_op_weighted": resf.op.trunc_op, "time_s": time.time() - tl}
    rows.append(rowf)
    _csv("s7e_controls.csv", rows)
    print(f"  a=1/2 (L={Lf}, m={mf}, lam={lamf}, tau0={tauf}): E_min {resf.E_min:.6e} exact/H "
          f"{rowf['ratio_exact_over_hartree']:.4f} (control {controlf:.4f}, normalized {rowf['ratio_hartree_normalized']:.4f}) "
          f"quartic frac {rowf['H_frac_quartic']:+.4f} [{time.time() - tl:.0f}s]", flush=True)
    # its own dispersion, for the K_eff reference at this spacing
    tl = time.time()
    df = qr.interacting_dispersion_tebd(gsf, T=DISP_T_FINE, dt=DISP_DT, order=4, chi_max=min(DISP_CHI, 40))
    gapf, ginfof = qr.excitation_gap_dmrg(gsf, chi_max=16)
    drow = _disp_row(df, lamf, gapf, ginfof, label="lattice_spacing_half")
    drow["time_s"] = time.time() - tl
    _csv("s7e_dispersion_fine.csv", [drow])
    _dump("s7e_dispersion_fine.json", {"_source": "notebook_qei_reference.py::S7e",
                                       "dispersion": _disp_json(df, lamf, gapf, ginfof)})
    keff = _keff_reference(Lf, Lf // 2, f_f, df.k, df.omega, df.weight)
    rowf.update({"E_keff": keff["E_keff"], "mu2_eff": keff["mu2_eff"], "J_eff": json.dumps(list(keff["J_eff"])),
                 "keff_misfit_max": keff["misfit_max"], "ratio_exact_over_keff": resf.E_min / keff["E_keff"],
                 "ratio_keff_normalized": resf.E_min / keff["E_keff"] / controlf,
                 "weight_min": drow["weight_min"], "gap_check_dev": drow["gap_check_dev"]})
    _csv("s7e_controls.csv", rows)
    print(f"  a=1/2 dispersion: weight min {drow['weight_min']:.3f}, gap dev {drow['gap_check_dev']:+.1e}; "
          f"K_eff mu2 {keff['mu2_eff']:.4f} J {np.array2string(keff['J_eff'], precision=4)} | "
          f"exact/K_eff {rowf['ratio_exact_over_keff']:.4f} (normalized {rowf['ratio_keff_normalized']:.4f}) "
          f"[{time.time() - tl:.0f}s]", flush=True)
    TIMES["S7e"] = time.time() - t0

# ---------------------------------------------------------------------------
# S7s  summary: K_eff references from the measured dispersion, ratios, fits
# ---------------------------------------------------------------------------
if "s" in ONLY:
    _section("S7s summary: the K_eff reference beside the Hartree one, and the tau0 exponent")
    t0 = time.time()
    summary = {"_source": "notebook_qei_reference.py::S7s", "build_git_head": build, "date_utc": _now()}
    disp = {}
    for pth in sorted(DATA.glob("s7a_dispersion*.json")):   # one file, or one per parallel --lams run
        disp.update(json.load(open(pth))["per_lambda"])
    if disp and not (DATA / "s7a_dispersion.json").exists():
        _dump("s7a_dispersion.json", {"_source": "notebook_qei_reference.py::S7s (merged)", "per_lambda": disp})
        merged = []
        for pth in sorted(DATA.glob("s7a_dispersion_lam*.csv")):
            merged += _read_csv(pth.name)
        merged.sort(key=lambda r: float(r["lam"]))
        _csv("s7a_dispersion.csv", merged)
    summary["dispersion"] = {}
    for key, d in disp.items():
        w = np.asarray(d["weight"])
        om = np.asarray(d["omega"])
        summary["dispersion"][key] = {
            "omega": om.tolist(), "omega_hartree": d["omega_hartree"], "weight_min": float(np.min(w)),
            "weight_at_k1": float(w[0]), "gamma_over_gap_max": float(np.max(np.abs(d["gamma"])) / om[0]),
            "fwhm_ratio_max": float(np.max(np.asarray(d["fwhm"]) / np.asarray(d["fwhm_window"]))),
            "fwhm_over_gap_max": float(np.max(d["fwhm"]) / om[0]),
            "gap_dmrg": d["gap_dmrg"], "gap_check_dev": float(om[0] - d["gap_dmrg"]),
            "max_dev_from_hartree": float(np.max(np.abs(om - np.asarray(d["omega_hartree"])))),
            "max_dev_from_free": float(np.max(np.abs(om - np.asarray(d["omega_free"])))),
            "sharp_peak": bool(np.min(w) > 0.9 and np.max(np.abs(d["gamma"])) * d["meta"]["T"] < 0.5)}
    # the K_eff reference at every (lambda, tau0) with an exact number on file
    ref_rows = []
    if disp:
        control_by_tau = {}
        exact_rows = []
        quartic_by_lam = {}
        if (DATA / "s7c_decomposition.csv").exists():
            for r in _read_csv("s7c_decomposition.csv"):
                quartic_by_lam[float(r["lam"])] = float(r["H_frac_quartic"])
        if (DATA / "s7b_exact_regenerated.csv").exists():
            for r in _read_csv("s7b_exact_regenerated.csv"):
                row = {k: float(v) for k, v in r.items()}
                row["H_frac_quartic"] = quartic_by_lam.get(row["lam"])
                row["from"] = "S7b"
                exact_rows.append(row)
        if (DATA / "s7d_tau0_sweep.csv").exists():
            for r in _read_csv("s7d_tau0_sweep.csv"):
                exact_rows.append({"lam": float(r["lam"]), "tau0": float(r["tau0"]), "L": float(r["L"]),
                                   "E_min": float(r["E_min"]), "E_free_exact": float(r["E_free_exact"]),
                                   "E_hartree_exact": float(r["E_hartree_exact"]),
                                   "ratio_exact_over_hartree": float(r["ratio_exact_over_hartree"]),
                                   "E_min_lam0": float(r["E_min_lam0"]), "control_lam0": float(r["control_lam0"]),
                                   "H_frac_quartic": float(r["H_frac_quartic"]), "H_audit_rel": float(r["H_audit_rel"]),
                                   "from": "S7d"})
        # lambda = 0 controls per tau0 (S7b has the lambda = 0 row at TAU0; S7d carries its own)
        for r in exact_rows:
            if r["lam"] == 0.0:
                control_by_tau[round(r["tau0"], 6)] = r["E_min"] / r["E_free_exact"]
        for r in exact_rows:
            if r["lam"] == 0.0:
                continue
            key = f"lam={r['lam']:g}"
            if key not in disp:
                continue
            d = disp[key]
            L = int(r["L"])
            f_r = gaussian_f(r["tau0"], n_sigmas=4.0)
            keff = _keff_reference(L, L // 2, f_r, np.asarray(d["k"]), np.asarray(d["omega"]), np.asarray(d["weight"]))
            control = r.get("control_lam0", control_by_tau.get(round(r["tau0"], 6)))
            # sanity: the same pipeline at the uniform Hartree mass reproduces E_hartree_exact
            # (the omega_hartree curve of S7a is the self-consistent chain's, whose mass is
            # site-dependent near the walls, so it is a reference curve, not this check)
            E_H_check, _ = qr.free_infimum_keff(L, L // 2, f_r, hartree_mass_squared(L, MASS, r["lam"]), [1.0])
            row = {"lam": r["lam"], "tau0": r["tau0"], "L": L, "E_min": r["E_min"],
                   "E_free_exact": r["E_free_exact"], "E_hartree_exact": r["E_hartree_exact"],
                   "E_hartree_by_keff_pipeline": E_H_check.e_min,
                   "E_keff": keff["E_keff"], "mu2_eff": keff["mu2_eff"], "J_eff": json.dumps(list(keff["J_eff"])),
                   "keff_misfit_max": keff["misfit_max"], "J_unconstrained_min": keff["J_unconstrained_min"],
                   "ratio_exact_over_hartree": r["E_min"] / r["E_hartree_exact"],
                   "ratio_exact_over_keff": r["E_min"] / keff["E_keff"],
                   "control_lam0": control,
                   "ratio_hartree_normalized": (r["E_min"] / r["E_hartree_exact"] / control) if control else None,
                   "ratio_keff_normalized": (r["E_min"] / keff["E_keff"] / control) if control else None,
                   "residual_hartree_normalized": (1 - r["E_min"] / r["E_hartree_exact"] / control) if control else None,
                   "residual_keff_normalized": (1 - r["E_min"] / keff["E_keff"] / control) if control else None,
                   "E_keff_over_E_hartree": keff["E_keff"] / r["E_hartree_exact"],
                   "H_frac_quartic": r.get("H_frac_quartic"), "H_audit_rel": r.get("H_audit_rel"),
                   # the operator truncation is calibrated by the lambda = 0 control: a control
                   # off by more than 5 % (tau0 >= 1 at chi_op = 16) means the MPO route has
                   # failed at that trajectory length and the row carries no usable ratio
                   "control_ok": bool(control is not None and abs(control - 1.0) < 0.05)}
            ref_rows.append(row)
            print(f"  lam={r['lam']:g} tau0={r['tau0']:g}: E_H {r['E_hartree_exact']:.6e} E_keff {keff['E_keff']:.6e} "
                  f"(K_eff/H {row['E_keff_over_E_hartree']:.4f}) | exact/H {row['ratio_exact_over_hartree']:.4f} "
                  f"exact/K_eff {row['ratio_exact_over_keff']:.4f} | normalized H {row['ratio_hartree_normalized']} "
                  f"K_eff {row['ratio_keff_normalized']} | mu2_eff {keff['mu2_eff']:.4f} J {np.array2string(keff['J_eff'], precision=4)} "
                  f"misfit {keff['misfit_max']:.1e}", flush=True)
        ref_rows.sort(key=lambda r: (r["lam"], r["tau0"]))
        _csv("s7b_reference.csv", ref_rows)
    summary["reference"] = ref_rows
    # the tau0 exponent at the largest lambda, against 1/tau0^2
    fits = {}
    lam_hi = max((r["lam"] for r in ref_rows), default=None)
    if lam_hi is not None:
        sub = [r for r in ref_rows if r["lam"] == lam_hi and r["control_ok"]]
        fits["rows_excluded_control_failed"] = [r["tau0"] for r in ref_rows if r["lam"] == lam_hi and not r["control_ok"]]
        for key in ("residual_hartree_normalized", "residual_keff_normalized", "H_frac_quartic"):
            pts = [(r["tau0"], r[key]) for r in sub if r.get(key) is not None]
            # the quartic piece is positive (it opposes the negative energy): its fraction of
            # E_min < 0 is negative, so the power law is fitted to |fraction|
            pos = [(t, abs(v)) for t, v in pts if v is not None and v != 0]
            entry = {"tau0": [t for t, _ in pts], "value": [v for _, v in pts]}
            if len(pos) >= 3:
                x = np.log(1.0 / np.array([t for t, _ in pos]))
                y = np.log(np.array([v for _, v in pos]))
                p, cov = np.polyfit(x, y, 1, cov=True) if len(pos) > 2 else (np.polyfit(x, y, 1), None)
                entry["exponent_of_inverse_tau0"] = float(p[0])
                entry["exponent_stderr"] = float(np.sqrt(cov[0, 0])) if cov is not None else None
                entry["n_points_used"] = len(pos)
            else:
                entry["exponent_of_inverse_tau0"] = None
                entry["note"] = "fewer than 3 positive values: no power law can be fitted"
            fits[key] = entry
            print(f"  {key}: {[(t, None if v is None else round(v, 5)) for t, v in pts]} -> exponent "
                  f"{entry.get('exponent_of_inverse_tau0')} (reviewer's prediction: 2)", flush=True)
    summary["tau0_fits"] = fits
    summary["reviewer_prediction_exponent"] = 2.0
    for name in ("s7c_decomposition.csv", "s7d_tau0_sweep.csv", "s7e_controls.csv", "s7a_dispersion.csv"):
        if (DATA / name).exists():
            summary[name] = _read_csv(name)
    summary["section_times_s"] = TIMES
    summary["wall_time_s"] = time.time() - T_START
    _dump("s7_summary.json", summary)
    TIMES["S7s"] = time.time() - t0

print(f"\nsections {sorted(ONLY)} done; wall time {time.time() - T_START:.1f} s "
      f"({(time.time() - T_START) / 60:.1f} min); section times {json.dumps({k: round(v) for k, v in TIMES.items()})}")
