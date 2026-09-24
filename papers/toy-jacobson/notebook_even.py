#!/usr/bin/env python
"""Toy Jacobson, Layer 7 unit 2, item (a): what the 1+1 even sector actually is.

    .venv/bin/python papers/toy-jacobson/notebook_even.py            # writes data/I_even_sector.*
    .venv/bin/python papers/toy-jacobson/notebook_even.py --quick    # smoke (~30 s)

THE QUESTION
============
MANIFEST anchor 8 left the reflection-even sector "still inconsistent": with the
ball's top Williamson mode projected out of the variation the even residual falls
from 2.93/2.23/2.07 to 0.0444/0.0510/0.167 (L = 8/16/32) but stays 53x/194x/508x
above the odd sector and does not decay with L.  Three candidate causes:

  (i)   physics: the 1+1 massless scalar's zero mode (Casini-Huerta, J. Phys. A
        42 (2009) 504007) is an O(1) piece of the ball's modular Hamiltonian with
        no local-energy counterpart;
  (ii)  numerics: dS - d<K_loc> is a cancellation of two O(1) numbers and the
        float64 subtraction loses the answer;
  (iii) the variation family: the even variations excite the zero mode and the
        odd ones do not.

WHAT IS MEASURED HERE
=====================
I1  The defect against the zero-mode contribution, cell by cell: dS, d<K_loc>,
    and the entropy contribution g(nu_1) w_1 of the ball's TOP (reflection-even)
    Williamson mode -- the lattice's phi_bar.  The ratio is archived at ALL
    THREE wavelengths: the identification is a convergence in lambda, not a
    uniform property of the family (fix round 2).
I2  The projection ladder: project the top n Williamson modes out of the
    variation (n = 0..6) and record the ABSOLUTE defect |dS - dK|, |dS|, the
    RELATIVE residual and -- the only sector-comparable statistic -- the defect
    PER UNIT VARIATION |dS - dK| / ||dV_B||_F, separately.  The absolute defect
    alone cannot be compared with the odd sector's, because projecting shrinks
    the variation itself by ~1e5 (fix round 2).
I3  Precision controls that decide (ii): the projected dS by an independent
    finite-difference route through the exact entropy; the term-by-term
    assembly of dS - d<K_loc> in the ball's Williamson frame (no big
    subtraction); the per-mode cancellation ratio sum|c_k| / |dS|; and the
    number of decimal digits the subtraction actually costs.
I4  An (L, m) scan: a mass gaps the zero mode, but it also destroys conformality,
    so the scan is reported with both residuals and the diagnostic nu_1.
I5  The cross-reference to 2+1 (data/J_2p1_even_odd.json), where the massless
    scalar has no IR zero mode -- the decisive test, produced by notebook_2p1.py.
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

from vacuum.core import entropy, symplectic_inverse, williamson
from vacuum.core.precision import entropy_precise, symplectic_margins_mp
from vacuum.geometry.jacobson import (
    _g_of_nu,
    _halo,
    _mode_contributions,
    _vacuum_block,
    centred_ball,
    coupling_chain_K,
    equilibrium_residual,
    first_law_audit,
    local_modular_form,
    squeeze_modes,
    squeeze_variation,
)

DATA = Path(__file__).resolve().parent / "data"
T0 = time.time()
RATIOS = (4, 8, 16)


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


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


# ---------------------------------------------------------------------------


def _setup(K, eig, L):
    """Ball, halo, vacuum block, local form, Williamson frame."""
    N = K.shape[0]
    ball = centred_ball(N, L)
    sites, bpos = _halo(K, ball)
    V_B = _vacuum_block(eig, ball)
    G_loc = local_modular_form(K, ball, split="bond", sites=sites, beta_at="bond")
    S_w, nu = williamson(V_B)
    return ball, sites, np.asarray(bpos, int), V_B, G_loc, S_w, symplectic_inverse(S_w), nu


def _cell(K, eig, L, parity, n_project, ratios=RATIOS):
    """One (L, parity, n_project) block of cells with every diagnostic."""
    ball, sites, bidx, V_B, G_loc, S_w, Si_w, nu = _setup(K, eig, L)
    n_s = len(sites)
    sel = np.concatenate([bidx, bidx + n_s])
    keep = np.ones(2 * L)
    for k in range(int(n_project)):
        keep[k] = keep[L + k] = 0.0
    g = _g_of_nu(nu)
    rows = []
    for mode in squeeze_modes(eig, L, ratios=ratios, parity=parity):
        u, w = mode["u"], mode["omega"]
        dV_B0 = squeeze_variation(u[ball], w)
        dV_s = squeeze_variation(u[sites], w)
        W0 = Si_w @ dV_B0 @ Si_w.T
        d0 = np.diag(W0)
        contrib_full = g * 0.5 * (d0[:L] + d0[L:])
        if n_project:
            dV_B = S_w @ (W0 * keep[:, None] * keep[None, :]) @ S_w.T
            dV_s = dV_s.copy()
            dV_s[np.ix_(sel, sel)] = dV_B
        else:
            dV_B = dV_B0
        dS, info = _mode_contributions(V_B, dV_B, 1e-10, 1e-3, 40)
        dK = 0.5 * float(np.trace(G_loc @ dV_s))
        # term-by-term assembly of dS - dK in the Williamson frame (no big subtraction)
        A = S_w.T @ G_loc[np.ix_(sel, sel)] @ S_w
        W = Si_w @ dV_B @ Si_w.T
        c = info["contributions"]
        dA = np.diag(A)
        dW = np.diag(W)
        per_mode = c - 0.5 * (dA[:L] * dW[:L] + dA[L:] * dW[L:])
        off = 0.5 * (float(np.sum(A * W.T)) - float(np.sum(dA * dW)))
        defect_terms = float(np.sum(per_mode) - off)
        # SCALE-INVARIANT normalisation of the defect (fix round 2).  |dS - dK|
        # alone is not comparable across sectors: the projected even variation
        # is ~1e5 times SMALLER than the odd one, so a smaller absolute defect
        # carries no information about the local form's accuracy.  The defect
        # per unit variation, |dS - dK| / ||dV_B||_F, is comparable.
        dV_B_fro = float(np.linalg.norm(dV_B, "fro"))
        dV_s_fro = float(np.linalg.norm(dV_s, "fro"))
        rows.append({
            "L": int(L), "parity": parity, "ratio": mode["ratio"],
            "wavelength": mode["wavelength"], "n_project": int(n_project),
            "dS": float(dS), "dK": float(dK), "abs_defect": float(abs(dS - dK)),
            "r": float(abs(dS - dK) / abs(dS)),
            "dV_B_fro": dV_B_fro, "dV_s_fro": dV_s_fro,
            "defect_per_dV": float(abs(dS - dK) / dV_B_fro),
            "dS_per_dV": float(abs(dS) / dV_B_fro),
            "defect_term_by_term": defect_terms,
            "term_by_term_rel_gap": float(abs(defect_terms - (dS - dK))
                                          / max(abs(dS - dK), 1e-300)),
            "zero_mode_contribution": float(contrib_full[0]),
            "defect_over_zero_mode": float((dS - dK) / contrib_full[0])
            if contrib_full[0] != 0.0 else np.inf,
            "top_mode_share": info["top_mode_share"],
            "cancellation": float(np.sum(np.abs(c)) / abs(dS)),
            "digits_lost": float(np.log10(max(abs(dS), abs(dK)) / abs(dS - dK))),
            "floor_share": info["floor_share"], "nu_top": float(nu[0]),
            "nu_min_margin": float(nu.min() - 0.5),
        })
    return rows, (ball, V_B, S_w, Si_w, nu)


def section_I1(cfg):
    """The defect against the zero-mode contribution, both parities."""
    N = cfg["N"]
    K = coupling_chain_K(N, [1.0])
    eig = np.linalg.eigh(K)
    rows = []
    for L in cfg["sizes"]:
        for parity in ("odd", "even"):
            rs, _ = _cell(K, eig, L, parity, 0)
            rows.extend(rs)
            worst = max(r["r"] for r in rs)
            log(f"I1 L={L} {parity}: r = {worst:.4e}; defect/zero-mode at lambda=16L = "
                f"{[round(r['defect_over_zero_mode'], 4) for r in rs if r['ratio'] == 16][0]}")
    # the explicit zero-mode quantity: nu_1(N) at fixed L.  For the 1+1 massless
    # scalar the ball's top Williamson eigenvalue grows without bound with the
    # chain length (the IR zero mode, Casini-Huerta 2009); in 2+1 it saturates.
    nscan = []
    for Nn in cfg["N_scan"]:
        Kn = coupling_chain_K(Nn, [1.0])
        eign = np.linalg.eigh(Kn)
        for L in cfg["sizes_nscan"]:
            rs, (_b, _V, _S, _Si, nu) = _cell(Kn, eign, L, "even", 0)
            ro, _ = _cell(Kn, eign, L, "odd", 0)
            r16 = [r for r in rs if r["ratio"] == 16][0]
            nscan.append({"N": int(Nn), "L": int(L), "nu_top": float(nu[0]),
                          "g_nu_top": float(_g_of_nu(nu[:1])[0]),
                          "r_even": float(max(r["r"] for r in rs)),
                          "r_odd": float(max(r["r"] for r in ro)),
                          "abs_defect_even": float(max(r["abs_defect"] for r in rs)),
                          "zero_mode_contribution": r16["zero_mode_contribution"],
                          "defect_over_zero_mode": r16["defect_over_zero_mode"]})
            log(f"I1b N={Nn} L={L}: nu_top {nu[0]:.5f}  r_even {nscan[-1]['r_even']:.3e}  "
                f"r_odd {nscan[-1]['r_odd']:.3e}  defect/zm(16L) {r16['defect_over_zero_mode']:.4f}")
    ev16 = [r for r in rows if r["parity"] == "even" and r["ratio"] == 16]
    ev = [r for r in rows if r["parity"] == "even"]
    # fix round 2: the identification defect = zero-mode contribution is a
    # CONVERGENCE IN THE WAVELENGTH, not a uniform fact about the family.  All
    # three wavelengths are summarised here, not only the longest.
    by_ratio = {}
    dev_by_ratio = {}
    for r in ev:
        by_ratio.setdefault(str(r["L"]), {})[str(int(r["ratio"]))] = r["defect_over_zero_mode"]
        dev_by_ratio.setdefault(str(r["L"]), {})[str(int(r["ratio"]))] = \
            abs(r["defect_over_zero_mode"] - 1.0)
    return {"N": N, "rows": rows, "nu_top_vs_N": nscan,
            "defect_over_zero_mode_longest_wavelength":
                {str(r["L"]): r["defect_over_zero_mode"] for r in ev16},
            "defect_over_zero_mode_by_ratio": by_ratio,
            "defect_over_zero_mode_dev_from_one_by_ratio": dev_by_ratio,
            "top_mode_share_odd_max":
                float(max(abs(r["top_mode_share"]) for r in rows if r["parity"] == "odd")),
            "top_mode_share_even_min_lambda16L":
                float(min(abs(r["top_mode_share"]) for r in ev if r["ratio"] == 16)),
            "top_mode_share_even_min_all_ratios":
                float(min(abs(r["top_mode_share"]) for r in ev))}


def section_I2(cfg):
    """The projection ladder: absolute defect vs relative residual."""
    N = cfg["N"]
    K = coupling_chain_K(N, [1.0])
    eig = np.linalg.eigh(K)
    ladder = {}
    for L in cfg["sizes"]:
        for n_project in cfg["n_projects"]:
            rs, _ = _cell(K, eig, L, "even", n_project)
            ladder[f"L{L}_n{n_project}"] = {
                "r": float(max(r["r"] for r in rs)),
                "abs_defect_max": float(max(r["abs_defect"] for r in rs)),
                "abs_dS_min": float(min(abs(r["dS"]) for r in rs)),
                "cancellation_max": float(max(r["cancellation"] for r in rs)),
                "defect_per_dV_max": float(max(r["defect_per_dV"] for r in rs)),
                "defect_per_dV_min": float(min(r["defect_per_dV"] for r in rs)),
                "dV_B_fro_max": float(max(r["dV_B_fro"] for r in rs)),
                "rows": rs,
            }
        odd, _ = _cell(K, eig, L, "odd", 0)
        ladder[f"L{L}_odd"] = {
            "r": float(max(r["r"] for r in odd)),
            "abs_defect_max": float(max(r["abs_defect"] for r in odd)),
            "abs_dS_min": float(min(abs(r["dS"]) for r in odd)),
            "defect_per_dV_max": float(max(r["defect_per_dV"] for r in odd)),
            "defect_per_dV_min": float(min(r["defect_per_dV"] for r in odd)),
            "dV_B_fro_max": float(max(r["dV_B_fro"] for r in odd)),
            "rows": odd,
        }
        # fix round 2: the scale-invariant comparison the absolute defect cannot
        # make.  Per unit variation the projected even sector stays WORSE than
        # the odd one, by this factor, at every rung of the ladder.
        odd_by_ratio = {int(r["ratio"]): r["defect_per_dV"] for r in odd}
        for n_project in cfg["n_projects"]:
            cell = ladder[f"L{L}_n{n_project}"]
            cell["even_over_odd_per_dV"] = float(
                cell["defect_per_dV_max"] / ladder[f"L{L}_odd"]["defect_per_dV_max"])
            cell["even_over_odd_per_dV_min"] = float(
                cell["defect_per_dV_min"] / ladder[f"L{L}_odd"]["defect_per_dV_max"])
            # matched cell by cell (same L, same wavelength) -- the comparison
            # the absolute defect pretended to make
            cell["even_over_odd_per_dV_by_ratio"] = {
                str(int(r["ratio"])): float(r["defect_per_dV"] / odd_by_ratio[int(r["ratio"])])
                for r in cell["rows"]}
        log(f"I2 L={L}: " + "  ".join(
            f"n={n}: r={ladder[f'L{L}_n{n}']['r']:.3e} |d|={ladder[f'L{L}_n{n}']['abs_defect_max']:.2e}"
            for n in cfg["n_projects"])
            + f"  | odd: r={ladder[f'L{L}_odd']['r']:.3e} "
              f"|d|={ladder[f'L{L}_odd']['abs_defect_max']:.2e}")
        log(f"I2 L={L} per unit variation |d|/||dV_B||_F: " + "  ".join(
            f"n={n}: {ladder[f'L{L}_n{n}']['defect_per_dV_max']:.3e} "
            f"({ladder[f'L{L}_n{n}']['even_over_odd_per_dV']:.1f}x odd)"
            for n in cfg["n_projects"])
            + f"  | odd: {ladder[f'L{L}_odd']['defect_per_dV_max']:.3e}")
    n_top = max(cfg["n_projects"])
    matched = [v for L in cfg["sizes"]
               for v in ladder[f"L{L}_n{n_top}"]["even_over_odd_per_dV_by_ratio"].values()]
    matched0 = [v for L in cfg["sizes"]
                for v in ladder[f"L{L}_n0"]["even_over_odd_per_dV_by_ratio"].values()]
    ladder["summary_per_unit_variation"] = {
        "n_top": int(n_top),
        "even_over_odd_matched_min": float(min(matched)),
        "even_over_odd_matched_max": float(max(matched)),
        "even_over_odd_matched_min_unprojected": float(min(matched0)),
        "even_over_odd_matched_max_unprojected": float(max(matched0)),
        "note": ("per unit variation |dS - d<K_loc>| / ||dV_B||_F, matched cell by "
                 "cell (same L, same wavelength): the projected even sector is NEVER "
                 "better than the odd sector.  The absolute defect is not comparable "
                 "across sectors because the projection shrinks ||dV_B||_F itself by "
                 "up to 1e6."),
    }
    log(f"I2 summary: per unit variation the projected even sector is "
        f"{ladder['summary_per_unit_variation']['even_over_odd_matched_min']:.1f}x-"
        f"{ladder['summary_per_unit_variation']['even_over_odd_matched_max']:.1f}x the odd "
        f"sector (unprojected "
        f"{ladder['summary_per_unit_variation']['even_over_odd_matched_min_unprojected']:.1f}x-"
        f"{ladder['summary_per_unit_variation']['even_over_odd_matched_max_unprojected']:.1f}x)")
    return ladder


def section_I3(cfg):
    """Precision controls: does the subtraction lose the answer? (cause (ii))"""
    N = cfg["N"]
    K = coupling_chain_K(N, [1.0])
    eig = np.linalg.eigh(K)
    out = {"rows": []}
    for L in cfg["sizes_precise"]:
        ball, sites, bidx, V_B, G_loc, S_w, Si_w, nu = _setup(K, eig, L)
        n_s = len(sites)
        sel = np.concatenate([bidx, bidx + n_s])
        keep = np.ones(2 * L)
        keep[0] = keep[L] = 0.0
        for mode in squeeze_modes(eig, L, ratios=RATIOS, parity="even"):
            u, w = mode["u"], mode["omega"]
            dV_B0 = squeeze_variation(u[ball], w)
            W0 = Si_w @ dV_B0 @ Si_w.T
            dV_B = S_w @ (W0 * keep[:, None] * keep[None, :]) @ S_w.T
            dS, info = _mode_contributions(V_B, dV_B, 1e-10, 1e-3, 40)
            # independent route: finite differences of the EXACT entropy along dV_proj
            eps = cfg["eps_fd"]
            scale = float(np.abs(dV_B).max())

            def central(e):
                return (entropy_precise(V_B + (e / scale) * dV_B)
                        - entropy_precise(V_B - (e / scale) * dV_B)) / (2.0 * e / scale)

            D_c, D_f = central(eps), central(0.5 * eps)
            dS_fd = (4.0 * D_f - D_c) / 3.0
            margins = symplectic_margins_mp(V_B, dps=cfg["dps"])
            g_mp = np.log((1.0 + margins) / np.maximum(margins, 1e-300))
            d = np.diag(W0 * keep[:, None] * keep[None, :])
            dS_mp = float(np.sum(g_mp * 0.5 * (d[:L] + d[L:])))
            out["rows"].append({
                "L": int(L), "ratio": mode["ratio"], "dS_williamson": float(dS),
                "dS_finite_difference": float(dS_fd),
                "dS_mpmath_margins": dS_mp,
                "rel_gap_fd": float(abs(dS_fd - dS) / abs(dS)),
                "rel_gap_mp": float(abs(dS_mp - dS) / abs(dS)),
                "cancellation": float(np.sum(np.abs(info["contributions"])) / abs(dS)),
                "nu_min_margin_mp": float(margins.min()),
            })
            log(f"I3 L={L} ratio={mode['ratio']:.0f}: dS {dS:+.10e}  fd gap "
                f"{out['rows'][-1]['rel_gap_fd']:.2e}  mp gap {out['rows'][-1]['rel_gap_mp']:.2e}  "
                f"cancellation {out['rows'][-1]['cancellation']:.4f}")
    out["max_rel_gap_fd"] = float(max(r["rel_gap_fd"] for r in out["rows"]))
    out["max_rel_gap_mp"] = float(max(r["rel_gap_mp"] for r in out["rows"]))
    out["max_cancellation"] = float(max(r["cancellation"] for r in out["rows"]))
    return out


def section_I4(cfg):
    """(L, m) scan. A mass gaps the zero mode AND destroys conformality."""
    rows = []
    N = cfg["N_mass"]
    for m in cfg["masses"]:
        K = coupling_chain_K(N, [1.0], m=m)
        eig = np.linalg.eigh(K)
        for L in cfg["sizes_mass"]:
            ball = centred_ball(N, L)
            _, nu = williamson(_vacuum_block(eig, ball))
            kw = dict(sizes=(L,), ratios=RATIOS, eig=eig, return_details=True,
                      escalate_above=1e-3, beta_at="bond")
            ro = equilibrium_residual(K, parity="odd", zero_mode="parity", **kw)
            re = equilibrium_residual(K, parity="even", zero_mode="parity", **kw)
            rows.append({
                "m": float(m), "L": int(L), "mL": float(m * L),
                "nu_top": float(nu[0]),
                "r_odd": ro["residual"], "r_even": re["residual"],
                "even_over_odd": re["residual"] / ro["residual"],
                "top_mode_share_even":
                    float(max(abs(r["top_mode_share"]) for r in re["rows"])),
            })
            log(f"I4 m={m} L={L} (mL={m * L:.2f}): nu_top {nu[0]:.5f}  r_odd {ro['residual']:.3e}  "
                f"r_even {re['residual']:.3e}  ratio {rows[-1]['even_over_odd']:.3g}  "
                f"top_share_even {rows[-1]['top_mode_share_even']:.4f}")
    return {"N": N, "rows": rows}


def section_I5(cfg):
    """Cross-reference to 2+1 (produced by notebook_2p1.py)."""
    p = DATA / "J_2p1_even_odd.json"
    if not p.exists():
        log(f"I5: {p} not present; run notebook_2p1.py first")
        return {"available": False}
    J = json.load(open(p))
    out = {"available": True, "N_2p1": J["N"],
           "even_over_odd_2p1": {k: v["even_over_odd"] for k, v in J["per_R"].items()},
           "even_over_odd_2p1_min": J["even_over_odd_min"],
           "even_over_odd_2p1_max": J["even_over_odd_max"]}
    log(f"I5: 2+1 even/odd = {out['even_over_odd_2p1']}")
    return out


def main(argv=None):
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    quick = args.quick
    if args.out:
        DATA = Path(args.out)
    elif quick:
        DATA = Path(tempfile.mkdtemp(prefix="toy-jacobson-even-quick-"))
    DATA.mkdir(exist_ok=True, parents=True)
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cfg = {
        "quick": quick, "sha": sha,
        "N": 800 if quick else 1600,
        "sizes": (8, 16) if quick else (8, 16, 32),
        "n_projects": (0, 1, 2, 3) if quick else (0, 1, 2, 3, 4, 5, 6),
        "sizes_precise": (8,) if quick else (8, 16),
        "eps_fd": 5e-4, "dps": 30 if quick else 40,
        "N_mass": 400 if quick else 800,
        "masses": (0.0, 0.1, 1.0) if quick else (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
        "sizes_mass": (8, 16) if quick else (8, 16, 32),
        "N_scan": (200, 400, 800) if quick else (200, 400, 800, 1600, 3200),
        "sizes_nscan": (8,) if quick else (8, 16),
    }
    t0 = time.time()
    I1 = section_I1(cfg)
    I2 = section_I2(cfg)
    I3 = section_I3(cfg)
    I4 = section_I4(cfg)
    I5 = section_I5(cfg)
    wall = time.time() - t0
    build = {
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
        "generator": "papers/toy-jacobson/notebook_even.py", "generator_sha256": sha,
        "mode": "quick" if quick else "full", "wall_clock_s": float(wall),
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(),
    }
    payload = {"I1_zero_mode_decomposition": I1, "I2_projection_ladder": I2,
               "I3_precision_controls": I3, "I4_mass_scan": I4,
               "I5_cross_reference_2p1": I5, "__build__": build}
    with open(DATA / "I_even_sector.json", "w") as fh:
        json.dump(jsonable(payload), fh, indent=1)
    lad = I2
    ls = cfg["sizes"]
    np.savez(
        DATA / "I_even_sector.npz",
        sizes=np.array(ls),
        n_projects=np.array(cfg["n_projects"]),
        r_even_ladder=np.array([[lad[f"L{L}_n{n}"]["r"] for n in cfg["n_projects"]] for L in ls]),
        abs_defect_ladder=np.array([[lad[f"L{L}_n{n}"]["abs_defect_max"]
                                     for n in cfg["n_projects"]] for L in ls]),
        abs_dS_ladder=np.array([[lad[f"L{L}_n{n}"]["abs_dS_min"]
                                 for n in cfg["n_projects"]] for L in ls]),
        r_odd=np.array([lad[f"L{L}_odd"]["r"] for L in ls]),
        abs_defect_odd=np.array([lad[f"L{L}_odd"]["abs_defect_max"] for L in ls]),
        defect_over_zero_mode=np.array(
            [I1["defect_over_zero_mode_longest_wavelength"][str(L)] for L in ls]),
        # fix round 2: the wavelength dependence of the identification, and the
        # scale-invariant defect per unit variation.
        ratios=np.array(RATIOS),
        defect_over_zero_mode_by_ratio=np.array(
            [[I1["defect_over_zero_mode_by_ratio"][str(L)][str(int(rr))] for rr in RATIOS]
             for L in ls]),
        defect_per_dV_ladder=np.array([[lad[f"L{L}_n{n}"]["defect_per_dV_max"]
                                        for n in cfg["n_projects"]] for L in ls]),
        defect_per_dV_odd=np.array([lad[f"L{L}_odd"]["defect_per_dV_max"] for L in ls]),
        even_over_odd_per_dV=np.array([[lad[f"L{L}_n{n}"]["even_over_odd_per_dV"]
                                        for n in cfg["n_projects"]] for L in ls]),
        dV_B_fro_ladder=np.array([[lad[f"L{L}_n{n}"]["dV_B_fro_max"]
                                   for n in cfg["n_projects"]] for L in ls]),
        dV_B_fro_odd=np.array([lad[f"L{L}_odd"]["dV_B_fro_max"] for L in ls]),
        even_over_odd_per_dV_matched=np.array(
            [[lad[f"L{L}_n{max(cfg['n_projects'])}"]["even_over_odd_per_dV_by_ratio"][str(int(rr))]
              for rr in RATIOS] for L in ls]),
        mass_mL=np.array([r["mL"] for r in I4["rows"]]),
        mass_r_odd=np.array([r["r_odd"] for r in I4["rows"]]),
        mass_r_even=np.array([r["r_even"] for r in I4["rows"]]),
        mass_nu_top=np.array([r["nu_top"] for r in I4["rows"]]),
        nscan_N=np.array([r["N"] for r in I1["nu_top_vs_N"]]),
        nscan_L=np.array([r["L"] for r in I1["nu_top_vs_N"]]),
        nscan_nu_top=np.array([r["nu_top"] for r in I1["nu_top_vs_N"]]),
        nscan_r_even=np.array([r["r_even"] for r in I1["nu_top_vs_N"]]),
        nscan_r_odd=np.array([r["r_odd"] for r in I1["nu_top_vs_N"]]),
        nscan_defect_over_zm=np.array([r["defect_over_zero_mode"] for r in I1["nu_top_vs_N"]]),
        max_rel_gap_fd=I3["max_rel_gap_fd"], max_rel_gap_mp=I3["max_rel_gap_mp"],
        max_cancellation=I3["max_cancellation"],
        N=cfg["N"], __build__=sha)
    log(f"done in {wall:.0f} s; data in {DATA}")


if __name__ == "__main__":
    main()
