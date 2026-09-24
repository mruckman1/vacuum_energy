#!/usr/bin/env python
"""Toy Jacobson in 2+1 (Layer 7, unit 2 of 2): the square-lattice extension.

    .venv/bin/python papers/toy-jacobson/notebook_2p1.py            # full archive
    .venv/bin/python papers/toy-jacobson/notebook_2p1.py --quick    # smoke (~30 s)

Every section writes one result file under data/ (or --out DIR) and is SKIPPED
when that file already exists, so a crashed run resumes instead of restarting
(--force regenerates).

WHAT IS MEASURED
================
J1  r(R, N) for disks of radius R in an N x N Dirichlet box, for the wave
    lattice (omega = |k|) and the z = 2 biharmonic Lifshitz lattice, in the
    reflection-odd (dipole) and reflection-even sectors, with the CHM ansatz
    K_loc = 2 pi sum_B beta_i h_i built on the lattice's own PSD bond-split
    energy density -- with and without the conformal improvement term.
J2  The improvement coefficient.  With the CANONICAL energy density the 2+1
    residual is O(0.4) and does not decay with R: dS exceeds d<K_loc> by ~55 %.
    The deficit is a phi^2 (non-derivative) term uniform over the disk: fitting
    dS = d<K_can> + 2 pi xi (2/R) sum_B d<phi^2> gives xi, to be compared with
    the free scalar's conformal improvement (d-2)/(4(d-1)) = 1/8 in d = 3.
J3  The decisive test of the 1+1 even-sector question (item (a)): in 2+1 the
    massless scalar has no IR zero mode, so if the 1+1 even-sector obstruction
    is that zero mode, the 2+1 even sector must be consistent with the odd one.
J4  The weight set free over a radial Chebyshev family: is the CHM parabola
    recovered in 2+1 as it was in 1+1, and at what normalisation?
J5  Controls: the exact first law by finite differences (both parities, both
    dynamics), the N-convergence of the residual at fixed R, the two
    discretisations of the improvement term, and the floor diagnostics.
J6  The quadrature (xx / pp) split of the J1 canonical pairing -- the stated
    MECHANISM of J2 -- archived rather than left as a live diagnostic, together
    with the pp weight of the disk's exact modular Hamiltonian against 2 pi
    beta under two explicitly named statistics (the diagonal and the row sum).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from vacuum.geometry.jacobson import _mode_contributions, squeeze_variation
from vacuum.geometry.jacobson2d import (
    XI_IMPROVE_3D,
    chm_beta_2d,
    disk_ball,
    equilibrium_residual_2d,
    first_law_audit_2d,
    free_weight_residual_2d,
    local_modular_form_2d,
    sector_split_2d,
    sine_basis,
    square_lattice_K,
    squeeze_modes_2d,
    vacuum_block_2d,
)
from vacuum.geometry.jacobson2d import _halo_sites

DATA = Path(__file__).resolve().parent / "data"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def build_info(mode, wall, sha):
    return {
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "generator": "papers/toy-jacobson/notebook_2p1.py",
        "generator_sha256": sha,
        "mode": mode, "wall_clock_s": float(wall),
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(),
    }


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return x


def _done(name, force):
    """True when this section's result file is already on disk."""
    p = DATA / name
    return p.exists() and not force


def _save(name, payload, sha):
    payload = dict(payload)
    payload["__build__"] = sha
    with open(DATA / name, "w") as fh:
        json.dump(jsonable(payload), fh, indent=1)
    log(f"wrote {DATA / name}")


# ---------------------------------------------------------------------------
# J1 -- the residual map in 2+1
# ---------------------------------------------------------------------------


def section_J1(cfg):
    """r(R) for (dynamics) x (parity) x (improvement) x (beta discretisation)."""
    name = "J_2p1_residual.json"
    if _done(name, cfg["force"]):
        log(f"J1: {name} exists, skipping")
        return json.load(open(DATA / name))
    N = cfg["N"]
    out = {"N": N, "radii": list(cfg["radii"]), "ratios": list(cfg["ratios"]),
           "xi_improve": XI_IMPROVE_3D, "cells": {}}
    for z, label in ((1, "wave"), (2, "lifshitz_z2")):
        K = square_lattice_K(N, z=z)
        for xi in (0.0, XI_IMPROVE_3D):
            for parity in ("odd", "even"):
                for beta_at in cfg["beta_ats"]:
                    det = equilibrium_residual_2d(
                        N, radii=cfg["radii"], ratios=cfg["ratios"], parity=parity,
                        z=z, K=K, beta_at=beta_at, improve=xi, return_details=True)
                    key = f"{label}_{parity}_xi{xi:g}_{beta_at}"
                    out["cells"][key] = {
                        "residual": det["residual"], "rms": det["rms"],
                        "per_R": {str(int(k)): v for k, v in det["per_R"].items()},
                        "rows": det["rows"],
                    }
                    log(f"J1 {key}: r = {det['residual']:.4e}  per_R "
                        f"{ {int(k): float(f'{v:.3g}') for k, v in det['per_R'].items()} }")
    _save(name, out, cfg["sha"])
    return out


# ---------------------------------------------------------------------------
# J2 -- the conformal improvement coefficient, fitted
# ---------------------------------------------------------------------------


def section_J2(cfg):
    """Fit xi in dS = d<K_can> + 2 pi xi (2/R) sum_B d<phi^2>, cell by cell."""
    name = "J_2p1_improvement.json"
    if _done(name, cfg["force"]):
        log(f"J2: {name} exists, skipping")
        return json.load(open(DATA / name))
    N = cfg["N"]
    basis = sine_basis(N)
    rows = []
    for z, label in ((1, "wave"), (2, "lifshitz_z2")):
        K = square_lattice_K(N, z=z)
        for R in cfg["radii"]:
            disk = disk_ball(N, R)
            ball = disk["sites"]
            sites, _ = _halo_sites(K, ball)
            V_B = vacuum_block_2d(N, ball, z=z, basis=basis)
            G_can = local_modular_form_2d(K, disk, N, sites=sites, beta_at="site",
                                          improve=0.0)
            for parity in ("odd", "even"):
                for mode in squeeze_modes_2d(N, R, ratios=cfg["ratios"], parity=parity,
                                             z=z, basis=basis):
                    dV_B = squeeze_variation(mode["u"][ball], mode["omega"])
                    dV_s = squeeze_variation(mode["u"][sites], mode["omega"])
                    dS, info = _mode_contributions(V_B, dV_B, 1e-10, 1e-2, 40)
                    dK = 0.5 * float(np.trace(G_can @ dV_s))
                    phi2_site = mode["u"][ball] ** 2 / mode["omega"]
                    phi2 = float(np.sum(phi2_site))
                    base = 2.0 * np.pi * (2.0 / R) * phi2
                    # IDENTIFIABILITY: the same deficit fitted with three WRONG
                    # operator shapes.  Any single-operator fit is exact cell by
                    # cell; what identifies the shape is the SCATTER of the fitted
                    # coefficient across (R, wavelength).
                    beta_i = chm_beta_2d(np.asarray(disk["r"]), R)
                    rim = np.asarray(disk["r"]) >= (R - 1.0)
                    alt = {
                        "weight_gradsq_beta": base,
                        "beta_weighted": 2.0 * np.pi * float(beta_i @ phi2_site) / R,
                        "rim_only": 2.0 * np.pi * float(np.sum(phi2_site[rim])) / R,
                        "field_laplacian": float(
                            0.5 * np.trace((local_modular_form_2d(
                                K, disk, N, sites=sites, beta_at="site", improve=1.0,
                                improve_form="laplacian")
                                - G_can) @ dV_s)),
                    }
                    rows.append({
                        "alt_coefficients": {k: float((dS - dK) / v) if v != 0.0 else np.nan
                                             for k, v in alt.items()},
                        "dynamics": label, "parity": parity, "R": float(R),
                        "ratio": mode["ratio"], "wavelength": mode["wavelength"],
                        "dS": float(dS), "dK_canonical": dK,
                        "deficit": float(dS - dK), "phi2_response": phi2,
                        "improvement_base": float(base),
                        "xi_fit": float((dS - dK) / base),
                        "dK_over_dS": float(dK / dS),
                        "floor_share": info["floor_share"],
                    })
    wave = [r for r in rows if r["dynamics"] == "wave"]
    sel = [r for r in wave if r["R"] >= 4 and r["parity"] == "odd"]
    sel_e = [r for r in wave if r["R"] >= 4 and r["parity"] == "even"]
    out = {
        "N": N, "xi_exact": XI_IMPROVE_3D, "rows": rows,
        "xi_fit_odd_R_ge_4": {"mean": float(np.mean([r["xi_fit"] for r in sel])),
                              "sd": float(np.std([r["xi_fit"] for r in sel])),
                              "min": float(np.min([r["xi_fit"] for r in sel])),
                              "max": float(np.max([r["xi_fit"] for r in sel])),
                              "n": len(sel)},
        "xi_fit_even_R_ge_4": {"mean": float(np.mean([r["xi_fit"] for r in sel_e])),
                               "sd": float(np.std([r["xi_fit"] for r in sel_e])),
                               "min": float(np.min([r["xi_fit"] for r in sel_e])),
                               "max": float(np.max([r["xi_fit"] for r in sel_e])),
                               "n": len(sel_e)},
        "shape_identifiability": {
            shape: {
                "mean": float(np.mean([r["alt_coefficients"][shape] for r in sel])),
                "sd": float(np.std([r["alt_coefficients"][shape] for r in sel])),
                "relative_scatter": float(
                    np.std([r["alt_coefficients"][shape] for r in sel])
                    / abs(np.mean([r["alt_coefficients"][shape] for r in sel]))),
            }
            for shape in ("weight_gradsq_beta", "beta_weighted", "rim_only", "field_laplacian")
        },
        "dK_over_dS_canonical_odd": {
            "min": float(np.min([r["dK_over_dS"] for r in wave if r["parity"] == "odd"])),
            "max": float(np.max([r["dK_over_dS"] for r in wave if r["parity"] == "odd"]))},
    }
    log(f"J2: xi_fit (wave, odd, R>=4) = {out['xi_fit_odd_R_ge_4']['mean']:.5f} "
        f"+- {out['xi_fit_odd_R_ge_4']['sd']:.5f} "
        f"[{out['xi_fit_odd_R_ge_4']['min']:.5f}, {out['xi_fit_odd_R_ge_4']['max']:.5f}] "
        f"vs exact {XI_IMPROVE_3D}")
    log(f"J2: xi_fit (wave, even, R>=4) = {out['xi_fit_even_R_ge_4']['mean']:.5f} "
        f"+- {out['xi_fit_even_R_ge_4']['sd']:.5f}")
    for shape, v in out["shape_identifiability"].items():
        log(f"J2 shape {shape:22s}: coeff {v['mean']:+.5f} +- {v['sd']:.5f} "
            f"(relative scatter {v['relative_scatter']:.4f})")
    _save(name, out, cfg["sha"])
    return out


# ---------------------------------------------------------------------------
# J3 -- the decisive even/odd test
# ---------------------------------------------------------------------------


def section_J3(cfg):
    """Even vs odd in 2+1, where the 1+1 IR zero mode does not exist.

    Also: projecting the disk's top Williamson mode out of the variation (the
    1+1 zero-mode surgery) should do almost nothing in 2+1.
    """
    name = "J_2p1_even_odd.json"
    if _done(name, cfg["force"]):
        log(f"J3: {name} exists, skipping")
        return json.load(open(DATA / name))
    N = cfg["N"]
    K = square_lattice_K(N, z=1)
    xi = XI_IMPROVE_3D
    ratios = {}
    for R in cfg["radii"]:
        ro = equilibrium_residual_2d(N, radii=(R,), ratios=cfg["ratios"], parity="odd",
                                     K=K, beta_at="site", improve=xi, return_details=True)
        re = equilibrium_residual_2d(N, radii=(R,), ratios=cfg["ratios"], parity="even",
                                     K=K, beta_at="site", improve=xi, return_details=True)
        rp = equilibrium_residual_2d(N, radii=(R,), ratios=cfg["ratios"], parity="even",
                                     K=K, beta_at="site", improve=xi, return_details=True,
                                     n_project=1)
        ratios[str(int(R))] = {
            "r_odd": ro["residual"], "r_even": re["residual"],
            "r_even_projected": rp["residual"],
            "even_over_odd": re["residual"] / ro["residual"],
            "abs_defect_odd": float(np.max([r["abs_defect"] for r in ro["rows"]])),
            "abs_defect_even": float(np.max([r["abs_defect"] for r in re["rows"]])),
            "top_mode_share_odd": float(np.max([abs(r["top_mode_share"]) for r in ro["rows"]])),
            "top_mode_share_even": float(np.max([abs(r["top_mode_share"]) for r in re["rows"]])),
            "nu_top": float(ro["rows"][0]["nu_top"]),
            "cancellation_even": float(np.max([r["cancellation"] for r in re["rows"]])),
        }
        log(f"J3 R={R}: odd {ro['residual']:.4e}  even {re['residual']:.4e}  "
            f"even/odd {ratios[str(int(R))]['even_over_odd']:.2f}  "
            f"even|projected {rp['residual']:.4e}  nu_top {ratios[str(int(R))]['nu_top']:.4f}")
    out = {"N": N, "xi": xi, "per_R": ratios,
           "even_over_odd_max": float(max(v["even_over_odd"] for v in ratios.values())),
           "even_over_odd_min": float(min(v["even_over_odd"] for v in ratios.values())),
           "nu_top_max": float(max(v["nu_top"] for v in ratios.values()))}
    _save(name, out, cfg["sha"])
    return out


# ---------------------------------------------------------------------------
# J4 -- the weight set free
# ---------------------------------------------------------------------------


def section_J4(cfg):
    """Free radial (even Chebyshev) weight, with and without the improvement."""
    name = "J_2p1_free_weight.json"
    if _done(name, cfg["force"]):
        log(f"J4: {name} exists, skipping")
        return json.load(open(DATA / name))
    N = cfg["N"]
    out = {"N": N, "radii": list(cfg["radii_free"]), "ratios": list(cfg["ratios_free"])}
    for z, label in ((1, "wave"), (2, "lifshitz_z2")):
        K = square_lattice_K(N, z=z)
        for xi, tag in ((0.0, "canonical"), (XI_IMPROVE_3D, "improved")):
            res = free_weight_residual_2d(
                N, radii=cfg["radii_free"], ratios=cfg["ratios_free"], parity="odd",
                z=z, beta_at="site", n_modes=cfg["n_modes"], K=K, improve=xi,
                nonneg=True)
            out[f"{label}_{tag}"] = {str(int(R)): v for R, v in res.items()}
            for R, v in res.items():
                two = v["per_n_modes"].get(2)
                log(f"J4 {label}/{tag} R={R:.0f}: r_chm {v['r_chm']:.3e}  "
                    f"n=2 r_free {two['r_free']:.3e} coeffs {[round(c, 4) for c in two['coefficients_over_piR2']]} "
                    f"overlap {two['overlap_with_chm']:.5f} norm {two['normalisation']:.4f} "
                    f"rank {two['rank_profile']}")
    _save(name, out, cfg["sha"])
    return out


# ---------------------------------------------------------------------------
# J5 -- controls
# ---------------------------------------------------------------------------


def section_J5(cfg):
    """First-law audits, N-convergence, and the two improvement discretisations."""
    name = "J_2p1_controls.json"
    if _done(name, cfg["force"]):
        log(f"J5: {name} exists, skipping")
        return json.load(open(DATA / name))
    audits = []
    for z in (1, 2):
        for parity in ("odd", "even"):
            a = first_law_audit_2d(cfg["N_audit"], cfg["R_audit"], ratio=4, parity=parity, z=z)
            audits.append(a)
            log(f"J5 audit z={z} {parity}: dS_fd {a['dS_fd']:.10e} vs dS_mod "
                f"{a['dS_modular']:.10e}  rel_err {a['rel_err']:.2e}")
    conv = {}
    for N in cfg["N_convergence"]:
        det = equilibrium_residual_2d(N, radii=(cfg["R_conv"],), ratios=cfg["ratios"],
                                      parity="odd", z=1, beta_at="site",
                                      improve=XI_IMPROVE_3D, return_details=True)
        conv[str(N)] = {"residual": det["residual"],
                        "rows": [{"ratio": r["ratio"], "wavelength": r["wavelength"],
                                  "dS": r["dS"], "dK": r["dK"], "r": r["r"]} for r in det["rows"]]}
        log(f"J5 N={N} (R={cfg['R_conv']}, odd, improved): r = {det['residual']:.4e}")
    forms = {}
    K = square_lattice_K(cfg["N"], z=1)
    for form in ("weight", "laplacian"):
        det = equilibrium_residual_2d(cfg["N"], radii=cfg["radii"], ratios=cfg["ratios"],
                                      parity="odd", z=1, K=K, beta_at="site",
                                      improve=XI_IMPROVE_3D, improve_form=form,
                                      return_details=True)
        forms[form] = {"residual": det["residual"],
                       "per_R": {str(int(k)): v for k, v in det["per_R"].items()}}
        log(f"J5 improve_form={form}: r = {det['residual']:.4e}")
    out = {"audits": audits, "N_convergence": conv, "improve_forms": forms,
           "audit_rel_err_max": float(max(a["rel_err"] for a in audits))}
    _save(name, out, cfg["sha"])
    return out


# ---------------------------------------------------------------------------
# J6 -- the quadrature split of the pairing, and the pp weight of the exact K_B
# ---------------------------------------------------------------------------


def section_J6(cfg):
    """Archive the xx/pp split of the canonical pairing and the pp-weight statistic.

    This is the mechanism sentence of J2 ("a non-derivative operator is missing").
    It was reported as a live diagnostic in the first pass and is archived here so
    that every number in it is reproducible and pinned. The pp-weight block also
    replaces the retracted claim that the exact modular Hamiltonian's pp DIAGONAL
    equals 2 pi beta to 3 %: the diagonal drifts with R (its off-diagonal mass
    grows), and it is the ROW SUM that stays near 1.
    """
    name = "J_2p1_sector_split.json"
    if _done(name, cfg["force"]):
        log(f"J6: {name} exists, skipping")
        return json.load(open(DATA / name))
    N = cfg["N"]
    K = square_lattice_K(N, z=1)
    out = {"N": N, "radii": list(cfg["radii"]), "ratios": list(cfg["ratios"]),
           "interior_margin": 1.0, "cells": {}}
    for parity in ("odd", "even"):
        det = sector_split_2d(N, radii=cfg["radii"], ratios=cfg["ratios"], parity=parity,
                              z=1, K=K, beta_at="site", improve=0.0, interior_margin=1.0)
        out["cells"][f"wave_{parity}_xi0_site"] = det
        log(f"J6 wave_{parity}: pp {det['pp_ratio_range'][0]:.4f}-{det['pp_ratio_range'][1]:.4f}  "
            f"xx {det['xx_ratio_range'][0]:.4f}-{det['xx_ratio_range'][1]:.4f}")
    odd = out["cells"]["wave_odd_xi0_site"]
    big = [R for R in cfg["radii"] if R >= 4]
    rows_big = [r for r in odd["rows"] if r["R"] >= 4.0]
    if rows_big:
        out["odd_R_ge_4"] = {
            "pp_ratio_range": [min(r["pp_ratio"] for r in rows_big),
                               max(r["pp_ratio"] for r in rows_big)],
            "xx_ratio_range": [min(r["xx_ratio"] for r in rows_big),
                               max(r["xx_ratio"] for r in rows_big)],
            "dS_pp_over_dS_range": [min(r["dS_pp_over_dS"] for r in rows_big),
                                    max(r["dS_pp_over_dS"] for r in rows_big)],
            "radii": [float(R) for R in big],
        }
    out["pp_weight_diag_slope"] = {str(int(R)): odd["per_R"][str(int(R))]["diag"]["slope_lsq"]
                                   for R in cfg["radii"]}
    out["pp_weight_rowsum_slope"] = {str(int(R)): odd["per_R"][str(int(R))]["rowsum"]["slope_lsq"]
                                     for R in cfg["radii"]}
    out["pp_offdiag_over_diag"] = {str(int(R)): odd["per_R"][str(int(R))]["offdiag_over_diag"]
                                   for R in cfg["radii"]}
    log("J6 pp-weight diag slope  " + "  ".join(f"R={R}:{out['pp_weight_diag_slope'][str(int(R))]:.4f}"
                                                for R in cfg["radii"]))
    log("J6 pp-weight rowsum slope " + "  ".join(f"R={R}:{out['pp_weight_rowsum_slope'][str(int(R))]:.4f}"
                                                 for R in cfg["radii"]))
    _save(name, out, cfg["sha"])
    return out


# ---------------------------------------------------------------------------


def main(argv=None):
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke configuration (~30 s)")
    ap.add_argument("--out", default=None, help="output directory (default data/; a fresh "
                                                "temporary directory for --quick)")
    ap.add_argument("--force", action="store_true", help="regenerate sections whose file exists")
    args = ap.parse_args(argv)
    quick = args.quick
    if args.out:
        DATA = Path(args.out)
    elif quick:
        DATA = Path(tempfile.mkdtemp(prefix="toy-jacobson-2p1-quick-"))
    DATA.mkdir(exist_ok=True, parents=True)
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cfg = {
        "quick": quick, "sha": sha, "force": args.force,
        "N": 48 if quick else 128,
        "radii": (3, 4) if quick else (3, 4, 5, 6, 8, 10),
        "ratios": (2, 4) if quick else (4, 8, 16),
        "beta_ats": ("site",) if quick else ("site", "bond"),
        "radii_free": (3, 4) if quick else (4, 6, 8),
        "ratios_free": (2, 3, 4) if quick else (3, 4, 5, 6, 8, 10, 12, 16),
        "n_modes": (1, 2, 3) if quick else (1, 2, 3, 4),
        "N_audit": 48 if quick else 96, "R_audit": 3 if quick else 5,
        "N_convergence": (32, 48) if quick else (48, 64, 96, 128),
        "R_conv": 3 if quick else 5,
    }
    t0 = time.time()
    J1 = section_J1(cfg)
    J2 = section_J2(cfg)
    J3 = section_J3(cfg)
    J4 = section_J4(cfg)
    J5 = section_J5(cfg)
    J6 = section_J6(cfg)
    wall = time.time() - t0
    summary = {
        "mode": "quick" if quick else "full",
        "N": cfg["N"], "radii": list(cfg["radii"]), "ratios": list(cfg["ratios"]),
        "xi_exact": XI_IMPROVE_3D,
        "J1_residual": {k: {"residual": v["residual"], "per_R": v["per_R"]}
                        for k, v in J1["cells"].items()},
        "J2_xi_fit_odd_R_ge_4": J2["xi_fit_odd_R_ge_4"],
        "J2_xi_fit_even_R_ge_4": J2["xi_fit_even_R_ge_4"],
        "J2_dK_over_dS_canonical_odd": J2["dK_over_dS_canonical_odd"],
        "J2_shape_identifiability": J2["shape_identifiability"],
        "J3_even_odd": J3,
        "J4_free_weight_wave_improved": J4.get("wave_improved", {}),
        "J5_audit_rel_err_max": J5["audit_rel_err_max"],
        "J5_improve_forms": J5["improve_forms"],
        "J6_sector_split": {
            "pp_ratio_range_all_R": J6["cells"]["wave_odd_xi0_site"]["pp_ratio_range"],
            "xx_ratio_range_all_R": J6["cells"]["wave_odd_xi0_site"]["xx_ratio_range"],
            "odd_R_ge_4": J6.get("odd_R_ge_4", {}),
            "pp_weight_diag_slope": J6["pp_weight_diag_slope"],
            "pp_weight_rowsum_slope": J6["pp_weight_rowsum_slope"],
            "pp_offdiag_over_diag": J6["pp_offdiag_over_diag"],
        },
        "wall_clock_s": wall,
    }
    _save("J_2p1_summary.json", summary, sha)
    radii = [float(R) for R in cfg["radii"]]
    imp_rows = [r for r in J2["rows"] if r["dynamics"] == "wave" and r["parity"] == "odd"]
    np.savez(
        DATA / "J_2p1_summary.npz",
        radii=np.array(radii), N=cfg["N"], xi_exact=XI_IMPROVE_3D,
        r_wave_odd_canonical=np.array([J1["cells"]["wave_odd_xi0_site"]["per_R"][str(int(R))]
                                       for R in radii]),
        r_wave_odd_improved=np.array([J1["cells"][f"wave_odd_xi{XI_IMPROVE_3D:g}_site"]["per_R"][str(int(R))]
                                      for R in radii]),
        r_wave_even_improved=np.array([J1["cells"][f"wave_even_xi{XI_IMPROVE_3D:g}_site"]["per_R"][str(int(R))]
                                       for R in radii]),
        r_lifshitz_odd_canonical=np.array([J1["cells"]["lifshitz_z2_odd_xi0_site"]["per_R"][str(int(R))]
                                           for R in radii]),
        r_lifshitz_odd_improved=np.array([J1["cells"][f"lifshitz_z2_odd_xi{XI_IMPROVE_3D:g}_site"]["per_R"][str(int(R))]
                                          for R in radii]),
        even_over_odd=np.array([J3["per_R"][str(int(R))]["even_over_odd"] for R in radii]),
        nu_top=np.array([J3["per_R"][str(int(R))]["nu_top"] for R in radii]),
        xi_fit_R=np.array([r["R"] for r in imp_rows]),
        xi_fit_ratio=np.array([r["ratio"] for r in imp_rows]),
        xi_fit=np.array([r["xi_fit"] for r in imp_rows]),
        dK_over_dS_canonical=np.array([r["dK_over_dS"] for r in imp_rows]),
        N_convergence=np.array(sorted(int(k) for k in J5["N_convergence"])),
        r_convergence=np.array([J5["N_convergence"][str(k)]["residual"]
                                for k in sorted(int(k) for k in J5["N_convergence"])]),
        sector_pp_ratio=np.array([[r["pp_ratio"] for r in J6["cells"]["wave_odd_xi0_site"]["rows"]
                                   if r["R"] == R] for R in radii]),
        sector_xx_ratio=np.array([[r["xx_ratio"] for r in J6["cells"]["wave_odd_xi0_site"]["rows"]
                                   if r["R"] == R] for R in radii]),
        sector_dS_pp_over_dS=np.array([[r["dS_pp_over_dS"] for r in J6["cells"]["wave_odd_xi0_site"]["rows"]
                                        if r["R"] == R] for R in radii]),
        pp_weight_diag_slope=np.array([J6["pp_weight_diag_slope"][str(int(R))] for R in radii]),
        pp_weight_rowsum_slope=np.array([J6["pp_weight_rowsum_slope"][str(int(R))] for R in radii]),
        pp_offdiag_over_diag=np.array([J6["pp_offdiag_over_diag"][str(int(R))] for R in radii]),
        __build__=sha)
    log(f"wrote {DATA / 'J_2p1_summary.npz'}")
    with open(DATA / "J_2p1_build_info.json", "w") as fh:
        json.dump(build_info("quick" if quick else "full", wall, sha), fh, indent=1)
    log(f"done in {wall:.0f} s; data in {DATA}")


if __name__ == "__main__":
    main()
