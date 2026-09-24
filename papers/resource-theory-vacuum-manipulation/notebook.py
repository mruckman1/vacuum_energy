#!/usr/bin/env python
"""A resource theory of vacuum manipulation — numerical theorems and charts (L3).

Re-runnable end to end:

    .venv/bin/python papers/resource-theory-vacuum-manipulation/notebook.py          # full (< 15 min)
    .venv/bin/python papers/resource-theory-vacuum-manipulation/notebook.py --quick  # smoke size

THE CLAIM (one paragraph)
=========================
With free operations = energy-non-signalling local instruments on Alice's
region + classical communication + arbitrary local maps on Bob's region
(vacuum.qet.resource, module docstring), the one-way-LOCC extractable
energy of Bob's region W_-> is a ledger monotone: it never increases on
average under free operations and decreases by at least the energy Bob's
device receives. Hotta's minimal protocol is one free-operation sequence
whose W_->(|g>) equals its E_B^max exactly, Bob's rotation saturates the
smooth-entropy one-shot bound (ratio 1), and the QET surplus
W_-> - W_loc is NOT a monotone. On critical Ising chains the single-site
protocol saturates its own single-site potential but sits far below both
the region-unitary one-shot bound and the whole-complement ceiling; the
gap is charted against distance. Every number lives in data/, none is
restated here.

Sections (each writes its arrays under data/):
  s1  monotonicity theorems: thousands of random free operations, worst violation + teeth
  s2  the minimal model over (h, k): W_->, E_B, bound, negativity (heatmaps)
  s3  chains L = 4..10 vs distance: E_B, W_site, W_full, one-shot bounds (curves)
  s4  the surplus: LOCC counter-example and its behaviour under free operations
  s5  Gaussian mirror: W_-> of harmonic chains vs mass and size
  s6  the two-way theory: W_<-> vs W_->, the exact ledger, the round ratchet
  s7  the free class: convex position, the non-unital witness, the SDP certificate
  s8  the exact single-site closed form, the entropic bounds, smoothing, L = 12
Figures are written as plain SVG (no plotting dependency) under draft/.
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

from vacuum.core import harmonic_chain_K
from vacuum.qet import chains, hotta, slp
from vacuum.qet import resource as R

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DRAFT = HERE / "draft"


def build_info():
    src = Path(__file__).resolve()
    info = {
        "generator": str(src),
        "generator_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    try:
        info["git_head"] = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                                          capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        info["git_head"] = f"unavailable: {exc}"
    try:
        import cvxpy
        info["cvxpy"] = cvxpy.__version__
    except ImportError:
        info["cvxpy"] = None
    return info


def _save(name, **arrays):
    DATA.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload["__build__"] = np.array(json.dumps(build_info()), dtype="U4096")
    np.savez_compressed(DATA / name, **payload)


def _json(name, obj):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / name).write_text(json.dumps(obj, indent=2, default=_jsonable))


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (tuple, set)):
        return list(x)
    return str(x)


# --------------------------------------------------------------------------
# minimal SVG writer (line chart with log y; heatmap)
# --------------------------------------------------------------------------


def _svg_lines(path, series, title, xlabel, ylabel, logy=True, hlines=()):
    W, Hh, ml, mr, mt, mb = 640, 420, 70, 20, 40, 50
    xs_all = np.concatenate([np.asarray(s["x"], float) for s in series])
    ys_all = np.concatenate([np.asarray(s["y"], float) for s in series] + [np.array([v for v, _ in hlines])])
    ys_all = ys_all[np.isfinite(ys_all) & (ys_all > 0 if logy else True)]
    x0, x1 = xs_all.min(), xs_all.max()
    y0, y1 = ys_all.min(), ys_all.max()
    if logy:
        y0, y1 = np.log10(y0), np.log10(y1)
    y0, y1 = y0 - 0.05 * (y1 - y0 + 1e-9), y1 + 0.05 * (y1 - y0 + 1e-9)

    def px(x):
        return ml + (x - x0) / max(x1 - x0, 1e-12) * (W - ml - mr)

    def py(y):
        yy = np.log10(y) if logy else y
        return Hh - mb - (yy - y0) / max(y1 - y0, 1e-12) * (Hh - mt - mb)

    colors = ["#1f4e79", "#c0392b", "#2e8b57", "#8e44ad", "#e67e22", "#16a085", "#7f8c8d"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Helvetica, Arial" font-size="12">',
           f'<rect width="{W}" height="{Hh}" fill="white"/>',
           f'<text x="{W/2}" y="22" text-anchor="middle" font-size="14">{title}</text>',
           f'<line x1="{ml}" y1="{Hh-mb}" x2="{W-mr}" y2="{Hh-mb}" stroke="black"/>',
           f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{Hh-mb}" stroke="black"/>',
           f'<text x="{(ml+W-mr)/2}" y="{Hh-12}" text-anchor="middle">{xlabel}</text>',
           f'<text x="16" y="{(mt+Hh-mb)/2}" text-anchor="middle" transform="rotate(-90 16 {(mt+Hh-mb)/2})">{ylabel}</text>']
    for xv in np.unique(xs_all):
        out.append(f'<text x="{px(xv):.1f}" y="{Hh-mb+16}" text-anchor="middle">{xv:g}</text>')
    ticks = np.arange(np.ceil(y0), np.floor(y1) + 1) if logy else np.linspace(y0, y1, 5)
    for tv in ticks:
        yv = 10.0 ** tv if logy else tv
        out.append(f'<line x1="{ml-4}" y1="{py(yv):.1f}" x2="{ml}" y2="{py(yv):.1f}" stroke="black"/>')
        out.append(f'<text x="{ml-6}" y="{py(yv)+4:.1f}" text-anchor="end">{"1e%d" % tv if logy else "%.3g" % tv}</text>')
    for i, s in enumerate(series):
        c = colors[i % len(colors)]
        pts = [(px(x), py(y)) for x, y in zip(s["x"], s["y"]) if np.isfinite(y) and (y > 0 or not logy)]
        if len(pts) > 1:
            out.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>'
                       % (c, " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)))
        for a, b in pts:
            out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3" fill="{c}"/>')
        out.append(f'<text x="{W-mr-150}" y="{mt+16+14*i}" fill="{c}">{s["label"]}</text>')
    for j, (v, lab) in enumerate(hlines):
        out.append(f'<line x1="{ml}" y1="{py(v):.1f}" x2="{W-mr}" y2="{py(v):.1f}" stroke="#444" stroke-dasharray="6,4"/>')
        out.append(f'<text x="{ml+6}" y="{py(v)-4:.1f}" fill="#444">{lab}</text>')
    out.append("</svg>")
    Path(path).write_text("\n".join(out))


def _svg_heatmap(path, Z, xs, ys, title, xlabel, ylabel, log=False):
    W, Hh, ml, mr, mt, mb = 560, 460, 70, 90, 40, 50
    Z = np.asarray(Z, float)
    V = np.log10(Z) if log else Z
    vmin, vmax = np.nanmin(V), np.nanmax(V)
    nx, ny = len(xs), len(ys)
    cw, ch = (W - ml - mr) / nx, (Hh - mt - mb) / ny
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Helvetica, Arial" font-size="12">',
           f'<rect width="{W}" height="{Hh}" fill="white"/>',
           f'<text x="{W/2}" y="22" text-anchor="middle" font-size="14">{title}</text>']
    for i in range(ny):
        for j in range(nx):
            t = (V[i, j] - vmin) / max(vmax - vmin, 1e-12)
            r, g, b = int(255 * min(1, 2 * t)), int(255 * (1 - abs(2 * t - 1))), int(255 * max(0, 1 - 2 * t))
            out.append(f'<rect x="{ml+j*cw:.1f}" y="{Hh-mb-(i+1)*ch:.1f}" width="{cw:.1f}" height="{ch:.1f}" fill="rgb({r},{g},{b})"/>')
    for j in range(0, nx, max(1, nx // 6)):
        out.append(f'<text x="{ml+(j+0.5)*cw:.1f}" y="{Hh-mb+16}" text-anchor="middle">{xs[j]:.2g}</text>')
    for i in range(0, ny, max(1, ny // 6)):
        out.append(f'<text x="{ml-6}" y="{Hh-mb-(i+0.5)*ch+4:.1f}" text-anchor="end">{ys[i]:.2g}</text>')
    out.append(f'<text x="{(ml+W-mr)/2}" y="{Hh-12}" text-anchor="middle">{xlabel}</text>')
    out.append(f'<text x="16" y="{(mt+Hh-mb)/2}" text-anchor="middle" transform="rotate(-90 16 {(mt+Hh-mb)/2})">{ylabel}</text>')
    for k in range(6):
        t = k / 5
        v = vmin + t * (vmax - vmin)
        out.append(f'<text x="{W-mr+8}" y="{Hh-mb-t*(Hh-mt-mb)+4:.1f}">{("1e%.1f" % v) if log else ("%.3g" % v)}</text>')
    out.append("</svg>")
    Path(path).write_text("\n".join(out))


# --------------------------------------------------------------------------
# s1: monotonicity theorems
# --------------------------------------------------------------------------


def s1_monotonicity(quick):
    n = 300 if quick else 3000
    cases = [
        ("hotta h=1 k=0.5", hotta.hamiltonian(1.0, 0.5), (0,), n),
        ("hotta h=0.3 k=2", hotta.hamiltonian(0.3, 2.0), (0,), n),
        ("hotta h=2 k=2 (A=qubit 1)", hotta.hamiltonian(2.0, 2.0), (1,), n // 2),
        ("tfim L=4 g=1 A=0", chains.tfim_hamiltonian(4, 1.0).toarray().astype(complex), (0,), n // 2),
        ("tfim L=5 g=1 A=2", chains.tfim_hamiltonian(5, 1.0).toarray().astype(complex), (2,), n // 4),
        ("tfim L=5 g=0.6 A=(0,1)", chains.tfim_hamiltonian(5, 0.6).toarray().astype(complex), (0, 1), n // 6),
        ("tfim L=6 g=1 A=(2,3)", chains.tfim_hamiltonian(6, 1.0).toarray().astype(complex), (2, 3), n // 10),
    ]
    rows = []
    for label, H, region, ns in cases:
        t0 = time.time()
        res = R.monotonicity_audit(H, region, n_samples=ns, seed=2026, tol=1e-9, strict=False)
        rows.append({"case": label, "n_samples": ns, "passed": res.passed,
                     "worst_violation": res.worst_violation, **{"worst_" + k: v for k, v in res.details["worst"].items()},
                     **{"teeth_" + k: v for k, v in res.details["teeth"].items()},
                     "max_signalling_defect": res.details["max_signalling_defect"],
                     "runtime_s": time.time() - t0})
        print(f"  s1 {label}: {ns} samples, worst {res.worst_violation:.2e}, teeth "
              f"{res.details['teeth']['nonfree_instrument']:.3f}/{res.details['teeth']['bob_injection']:.3f}")
    gauss = []
    for N, m, A, ns in [(2, 0.7, (0,), n // 5), (3, 0.5, (0,), n // 5), (4, 0.5, (0, 1), n // 8), (4, 0.3, (1,), n // 8)]:
        res = R.gaussian_monotonicity_audit(harmonic_chain_K(N, m, "dirichlet"), A, n_samples=ns,
                                            seed=7, tol=1e-9, strict=False)
        gauss.append({"N": N, "m": m, "A": A, "n_samples": ns, "passed": res.passed,
                      "worst_violation": res.worst_violation, **res.details["worst"], **res.details["teeth"]})
    _json("s1_monotonicity.json", {"qubit_cases": rows, "gaussian_cases": gauss,
                                   "total_samples": int(sum(r["n_samples"] for r in rows) + sum(g["n_samples"] for g in gauss)),
                                   "worst_violation_all": float(max(max(r["worst_violation"] for r in rows),
                                                                    max(g["worst_violation"] for g in gauss)))})
    return rows, gauss


# --------------------------------------------------------------------------
# s2: the minimal model over (h, k)
# --------------------------------------------------------------------------


def s2_hotta(quick):
    n = 9 if quick else 33
    hs = np.geomspace(0.05, 5.0, n)
    ks = np.geomspace(0.1, 4.0, n)
    keys = ["W_arrow", "E_B", "E_A", "W_over_E_A", "negativity", "bound_exact", "bound_best",
            "ratio_exact", "ratio_best", "ceiling", "W_loc_bound"]
    grids = {k: np.zeros((n, n)) for k in keys}
    for i, h in enumerate(hs):
        for j, k in enumerate(ks):
            led = R.hotta_resource_ledger(h, k)
            for key in keys:
                grids[key][i, j] = led[key]
    _save("s2_hotta_grid.npz", h=hs, k=ks, **grids)
    _svg_heatmap(DRAFT / "fig_hotta_W_arrow.svg", grids["W_arrow"], ks, hs,
                 "Minimal model: W_->(|g>) = E_B^max = one-shot bound (log10)", "k", "h", log=True)
    _svg_heatmap(DRAFT / "fig_hotta_W_over_EA.svg", grids["W_over_E_A"], ks, hs,
                 "Minimal model: exchange rate W_-> / E_A", "k", "h")
    summ = {"n_grid": n * n, "max_abs_ratio_minus_1": float(np.max(np.abs(grids["ratio_exact"] - 1.0))),
            "max_abs_W_minus_EB": float(np.max(np.abs(grids["W_arrow"] - grids["E_B"]))),
            "max_W_loc_bound": float(np.max(grids["W_loc_bound"])),
            "max_W_over_E_A": float(np.max(grids["W_over_E_A"])),
            "argmax_W_over_E_A_hk": [float(hs[np.unravel_index(np.argmax(grids["W_over_E_A"]), (n, n))[0]]),
                                     float(ks[np.unravel_index(np.argmax(grids["W_over_E_A"]), (n, n))[1]])],
            "eps_best_always_zero": bool(np.all(grids["bound_best"] == grids["bound_exact"]))}
    print(f"  s2 hotta grid {n}x{n}: max|ratio-1| {summ['max_abs_ratio_minus_1']:.2e}, "
          f"max W/E_A {summ['max_W_over_E_A']:.4f} at (h,k)={summ['argmax_W_over_E_A_hk']}")
    return summ


# --------------------------------------------------------------------------
# s3: chains vs distance
# --------------------------------------------------------------------------


def _marginal_bound_for_row(L, g, A, B, row):
    """Region-R SDP floor with Bob's untouched marginal fixed (cvxpy), R ordered (B, rest)."""
    try:
        import cvxpy  # noqa: F401
    except ImportError:
        return np.nan
    E0, psi = chains.tfim_ground_state(L, g)
    h_ops = chains.tfim_local_energy_ops(L, g)
    led = chains.run_chain_qet(L, g, A, B, ground=(E0, psi), h_ops=h_ops, theta_method="closed")
    Rr = row["R"]
    order = [B] + [q for q in Rr if q != B]
    k = len(order)
    h_loc = chains.local_hamiltonian(h_ops, B).toarray().astype(complex)
    h_R = R._partial_trace_B(h_loc, tuple(sorted(order)), L) / 2 ** (L - k)
    perm = [sorted(order).index(q) for q in order]
    t = np.transpose(h_R.reshape((2,) * (2 * k)), perm + [k + i for i in perm]).reshape(2**k, 2**k)
    br = [(o["p"], chains.reduced_density(o["state_meas"].astype(complex), L, order))
          for o in led["outcomes"].values()]
    return R.one_shot_bound_marginal(t, br, (2, 2 ** (k - 1)), (B,))["bound"]


def s3_chains(quick):
    Ls = [4, 6] if quick else [4, 5, 6, 7, 8, 9, 10]
    gs = [1.0] if quick else [0.5, 1.0, 2.0]
    rows = []
    for g in gs:
        for L in Ls:
            t0 = time.time()
            A = 0
            led = R.chain_resource_ledger(L, g, A, list(range(1, L)), method="kraus",
                                          kraus_kwargs={"restarts": 2})
            for row in led["rows"]:
                bm = _marginal_bound_for_row(L, g, A, row["B"], row) if L <= 8 else np.nan
                rows.append({"L": L, "g": g, "A": A, "B": row["B"], "d": row["d"], "E_A": led["E_A"],
                             "E_B": row["E_B"], "W_site": row["W_site"], "W_full": led["W_arrow_full"],
                             "bound_exact": row["bound_exact"], "bound_best": row["bound_best"],
                             "eps_best": row["eps_best"], "bound_cq": row["bound_cq"],
                             "bound_marginal": bm, "ceiling_R": row["ceiling_R"],
                             "H_min_cq": row["H_min_cq"], "negativity_AB": row["negativity_AB"],
                             "ratio_exact": row["ratio_exact"], "ratio_best": row["ratio_best"],
                             "ratio_marginal": row["E_B"] / bm if np.isfinite(bm) else np.nan,
                             "ratio_site": row["ratio_site"], "ratio_full": row["ratio_full"]})
            print(f"  s3 L={L} g={g}: {time.time()-t0:.1f}s, W_full {led['W_arrow_full']:.4e}, "
                  f"E_B(d=1) {led['rows'][0]['E_B']:.4e}, ratio_exact(d=1) {led['rows'][0]['ratio_exact']:.3f}")
    cols = sorted({k for r in rows for k in r})
    _save("s3_chain_rows.npz", **{c: np.array([r[c] for r in rows], float) for c in cols})
    # chart: L = max, g = 1
    Lmax = max(Ls)
    sel = [r for r in rows if r["L"] == Lmax and r["g"] == 1.0]
    d = [r["d"] for r in sel]
    series = [{"x": d, "y": [r["E_B"] for r in sel], "label": "E_B achieved (1-site rotation)"},
              {"x": d, "y": [r["W_site"] for r in sel], "label": "W_site (1-site CPTP potential)"},
              {"x": d, "y": [r["bound_exact"] for r in sel], "label": "one-shot bound, region R (eps=0)"},
              {"x": d, "y": [r["bound_marginal"] for r in sel], "label": "one-shot bound, marginal-fixed SDP"},
              {"x": d, "y": [r["negativity_AB"] for r in sel], "label": "negativity N(A:B)"}]
    _svg_lines(DRAFT / f"fig_chain_L{Lmax}_g1.svg", series,
               f"Critical TFIM L={Lmax}: achieved vs the theory's ceilings", "distance |A-B|", "energy (J units) / N",
               logy=True, hlines=[(sel[0]["W_full"], "W_-> (Bob = whole complement)"), (sel[0]["E_A"], "E_A")])
    summ = {"n_rows": len(rows),
            "max_E_B_over_bound_exact": float(np.nanmax([r["ratio_exact"] for r in rows])),
            "min_E_B_over_bound_exact": float(np.nanmin([r["ratio_exact"] for r in rows])),
            "max_E_B_over_bound_marginal": float(np.nanmax([r["ratio_marginal"] for r in rows])),
            "max_abs_ratio_site_minus_1": float(np.nanmax([abs(r["ratio_site"] - 1.0) for r in rows])),
            "max_E_B_over_W_full": float(np.nanmax([r["ratio_full"] for r in rows])),
            "any_bound_violation": bool(any(r["E_B"] > r["bound_exact"] + 1e-12 or r["E_B"] > r["bound_cq"] + 1e-12
                                            or (np.isfinite(r["bound_marginal"]) and r["E_B"] > r["bound_marginal"] + 1e-9)
                                            for r in rows)),
            "eps_best_nonzero_count": int(sum(1 for r in rows if r["eps_best"] > 0)),
            "W_full_le_E_A": bool(all(r["W_full"] <= r["E_A"] + 1e-12 for r in rows))}
    print(f"  s3 summary: {summ}")
    return summ


# --------------------------------------------------------------------------
# s4: the surplus
# --------------------------------------------------------------------------


def s4_surplus(quick):
    ce = [R.surplus_counterexample(h, k) for h, k in [(1.0, 0.5), (0.3, 2.0), (2.0, 2.0)]]
    # Q under free operations and under Bob's maps, 2 qubits, W_loc by the Kraus search
    n = 20 if quick else 120
    rng = np.random.default_rng(99)
    H = hotta.hamiltonian(1.0, 0.7)
    P, _ = R.free_projectors(H, (0,))
    rise_free, rise_bob, drop_bob = -np.inf, -np.inf, np.inf
    for i in range(n):
        rho = R.random_state(2, rng, ("pure", "mixed", "ground_mix")[i % 3], H=H)
        Q0 = R.qet_surplus(H, rho, (0,), method="kraus", kraus_kwargs={"restarts": 2})["Q"]
        if i % 2 == 0:
            K = R.random_free_instrument(P, int(rng.integers(1, 4)), rng)
            Q1 = sum(p * R.qet_surplus(H, r, (0,), method="kraus", kraus_kwargs={"restarts": 2})["Q"]
                     for p, r in R.apply_instrument(rho, K, (0,)))
            rise_free = max(rise_free, Q1 - Q0)
        else:
            K = R.random_instrument(2, int(rng.integers(1, 4)), rng)
            out = slp.apply_local_channel(rho, K, (1,))
            Q1 = R.qet_surplus(H, out, (0,), method="kraus", kraus_kwargs={"restarts": 2})["Q"]
            rise_bob = max(rise_bob, Q1 - Q0)
            drop_bob = min(drop_bob, Q1 - Q0)
    summ = {"counterexamples": ce, "n_samples": n,
            "max_rise_of_Q_under_free_instruments": rise_free,
            "max_rise_of_Q_under_bob_maps": rise_bob,
            "min_change_of_Q_under_bob_maps": drop_bob}
    _json("s4_surplus.json", summ)
    print(f"  s4 surplus: LOCC rise {ce[0]['rise']:.4f}; under free instruments max rise {rise_free:.2e}; "
          f"under Bob maps rise up to {rise_bob:.3f}, min change {drop_bob:.2e}")
    return summ


# --------------------------------------------------------------------------
# s5: Gaussian mirror
# --------------------------------------------------------------------------


def s5_gaussian(quick):
    ms = np.geomspace(0.05, 3.0, 6 if quick else 25)
    rows = []
    for N in (2, 3, 4):
        for m in ms:
            res = R.gaussian_qet_potential(harmonic_chain_K(N, m, "dirichlet"), (0,))
            rows.append({"N": N, "m": m, "W": res["W"], "W_identity": res["W_identity"],
                         "log_negativity": res["log_negativity"], "E_B_loc": res["E_B_loc"]})
    _save("s5_gaussian.npz", **{c: np.array([r[c] for r in rows], float) for c in rows[0]})
    series = [{"x": [r["m"] for r in rows if r["N"] == N], "y": [r["W"] for r in rows if r["N"] == N],
               "label": f"W_-> N={N}"} for N in (2, 3, 4)]
    series += [{"x": [r["m"] for r in rows if r["N"] == N], "y": [r["log_negativity"] for r in rows if r["N"] == N],
                "label": f"E_N N={N}"} for N in (2, 4)]
    _svg_lines(DRAFT / "fig_gaussian_W_vs_mass.svg", series, "Harmonic chain vacuum: Gaussian QET potential vs mass",
               "mass m (lattice units)", "W_-> / E_N (nats)", logy=True)
    summ = {"max_identity_defect": float(max(abs(r["W"] - r["W_identity"]) for r in rows)),
            "W_range": [float(min(r["W"] for r in rows)), float(max(r["W"] for r in rows))]}
    print(f"  s5 gaussian: identity defect {summ['max_identity_defect']:.2e}, W range {summ['W_range']}")
    return summ



# --------------------------------------------------------------------------
# s6: the two-way theory (Theorem 4)
# --------------------------------------------------------------------------


def s6_two_way(quick):
    rows = []
    cases = [("hotta h=1 k=0.5", hotta.hamiltonian(1.0, 0.5), (0,), 2),
             ("hotta h=0.3 k=2", hotta.hamiltonian(0.3, 2.0), (0,), 2),
             ("hotta h=2 k=2", hotta.hamiltonian(2.0, 2.0), (0,), 2)]
    if not quick:
        cases += [(f"tfim L={L} g=1 A=0",
                   np.asarray(chains.tfim_hamiltonian(L, 1.0).toarray(), complex), (0,), L)
                  for L in (5, 6, 7, 8)]
    for label, H, region, n in cases:
        w, U = np.linalg.eigh(H)
        rho = np.outer(U[:, 0], U[:, 0].conj())
        W_one = R.qet_potential(H, rho, region)["W"]
        n_rounds = 2 if (quick or n >= 7) else 3
        vals, nets, injs = [], [], []
        for r in range(1, n_rounds + 1):
            res = R.two_way_potential(H, rho, region, rounds=r)
            vals.append(res["W"]); nets.append(res["net"]); injs.append(res["injected"])
        bob_only = R.two_way_potential(H, rho, region, rounds=n_rounds,
                                       cash_both=False)["W"]
        rows.append({"case": label, "n": n, "W_one_way": W_one,
                     "W_two_way": vals, "net": nets, "injected": injs,
                     "W_two_way_bob_only": bob_only,
                     "bob_only_minus_one_way": bob_only - W_one,
                     "rounds": list(range(1, n_rounds + 1))})
        print(f"  s6 {label}: W_-> {W_one:.6f}, two-way {['%.4f' % v for v in vals]}, "
              f"Bob-only {bob_only:.6f} (delta {bob_only - W_one:.2e}), net {nets[0]:.4f}")
    audits = []
    for label, H, region, ns in [("hotta h=1 k=0.5", hotta.hamiltonian(1.0, 0.5), (0,), 60 if quick else 400),
                                 ("hotta h=0.3 k=2", hotta.hamiltonian(0.3, 2.0), (0,), 40 if quick else 300),
                                 ("tfim L=4 g=1 A=0", np.asarray(chains.tfim_hamiltonian(4, 1.0).toarray(), complex), (0,), 40 if quick else 250),
                                 ("tfim L=5 g=1 A=2", np.asarray(chains.tfim_hamiltonian(5, 1.0).toarray(), complex), (2,), 30 if quick else 150),
                                 ("tfim L=6 g=1 A=(2,3)", np.asarray(chains.tfim_hamiltonian(6, 1.0).toarray(), complex), (2, 3), 20 if quick else 80)]:
        a = R.two_way_monotonicity_audit(H, region, n_samples=ns, seed=2026, rounds=3,
                                         tol=1e-9, strict=False)
        audits.append({"case": label, "n_samples": ns, "passed": a.passed, **a.details})
        print(f"  s6 audit {label}: drift {a.details['worst_ledger_drift']:.2e}, "
              f"teeth {a.details['teeth']:.3f}")
    summ = {"rows": rows, "audits": audits,
            "max_bob_only_minus_one_way": float(max(abs(r["bob_only_minus_one_way"]) for r in rows)),
            "worst_ledger_drift": float(max(a["worst_ledger_drift"] for a in audits)),
            "min_teeth": float(min(a["teeth"] for a in audits)),
            "max_two_way_over_one_way": float(max(r["W_two_way"][-1] / r["W_one_way"] for r in rows)),
            "net_is_round_independent": bool(all(max(r["net"]) - min(r["net"]) < 1e-9 for r in rows)),
            "total_protocols": int(sum(a["counts"]["protocols"] for a in audits))}
    _json("s6_two_way.json", summ)
    return summ


# --------------------------------------------------------------------------
# s7: the free class (Theorem 2)
# --------------------------------------------------------------------------


def s7_free_class(quick):
    rows = []
    cases = [("hotta A=q0", hotta.hamiltonian(1.0, 0.5), (0,)),
             ("hotta A=q1", hotta.hamiltonian(0.3, 2.0), (1,)),
             ("tfim L=5 A=0", np.asarray(chains.tfim_hamiltonian(5, 1.0).toarray(), complex), (0,)),
             ("tfim L=5 A=(1,2)", np.asarray(chains.tfim_hamiltonian(5, 1.0).toarray(), complex), (1, 2)),
             ("tfim L=6 A=(2,3)", np.asarray(chains.tfim_hamiltonian(6, 1.0).toarray(), complex), (2, 3)),
             ("star n_alice=2", R.star_hamiltonian(1.0, 0.5), (0, 1)),
             ("star n_alice=3", R.star_hamiltonian(1.0, 0.5, n_alice=3), (0, 1, 2))]
    for label, H, region in cases:
        info = R.free_class_extremality(H, region)
        w = R.nonfree_energy_nonsignalling_instrument(H, region)
        row = {"case": label, "in_convex_position": info["in_convex_position"],
               "n_blocks": len(info["ranks"]), "ranks": info["ranks"],
               "n_non_extreme": int(sum(1 for e in info["extreme"] if not e))}
        if w is not None:
            row.update({"witness_defect": w["defect"], "witness_commutator": w["commutator"],
                        "witness_unitality_defect": w["unitality_defect"]})
            # what the wider free class buys over the commuting-Kraus closed form
            n = int(round(np.log2(H.shape[0])))
            ev, U = np.linalg.eigh(H)
            rho = np.outer(U[:, 0], U[:, 0].conj())
            reg_B = tuple(q for q in range(n) if q not in region)
            W_closed = R.qet_potential(H, rho, region)["W"]
            y = sum(p * slp.extractable_energy_kraus(H, r, reg_B, restarts=4)["extractable"]
                    for p, r in R.apply_instrument(rho, w["kraus"], region))
            row.update({"W_closed_form": W_closed, "yield_wider_class": y,
                        "gain_factor": y / W_closed,
                        "monotonicity_violation": sum(
                            p * R.qet_potential(H, r, region)["W"]
                            for p, r in R.apply_instrument(rho, w["kraus"], region)) - W_closed})
        try:
            sdp = R.search_free_instrument(H, region)
            row.update({"sdp_w_off": sdp["w_off"],
                        "sdp_commuting_only": sdp["is_commuting_kraus_only"],
                        "sdp_agrees": sdp["is_commuting_kraus_only"] == info["in_convex_position"]})
        except (ImportError, RuntimeError) as exc:
            row["sdp_w_off"] = None
            row["sdp_agrees"] = None
        rows.append(row)
        print(f"  s7 {label}: convex position {info['in_convex_position']}, "
              f"SDP w_off {row.get('sdp_w_off')}")
    summ = {"rows": rows,
            "sdp_agrees_everywhere": bool(all(r["sdp_agrees"] for r in rows
                                              if r["sdp_agrees"] is not None)),
            "max_gain_factor": float(max([r["gain_factor"] for r in rows
                                          if "gain_factor" in r] or [np.nan])),
            "max_witness_defect": float(max([r["witness_defect"] for r in rows
                                             if "witness_defect" in r] or [np.nan]))}
    _json("s7_free_class.json", summ)
    return summ


# --------------------------------------------------------------------------
# s8: the tight closed form, the entropic bounds, and smoothing
# --------------------------------------------------------------------------


def s8_entropic(quick):
    Ls = [6, 8] if quick else [6, 8, 10, 12]
    gs = [1.0] if quick else [0.5, 1.0, 2.0]
    rows = []
    for g in gs:
        for L in Ls:
            led = R.chain_region_ledger(L, g, 0, list(range(1, L)),
                                        marginal=(L <= 8), entropic=True)
            for row in led["rows"]:
                rows.append({"L": L, "g": g, "d": row["d"], "E_A": led["E_A"],
                             "E_B": row["E_B"], "W_site": row["W_site"],
                             "exact_unitary": row["exact_unitary"],
                             "W_full": led["W_arrow_full"],
                             "bound_exact": row["bound_exact"],
                             "bound_marginal": row["bound_marginal"],
                             "bound_guess": row["bound_guess"],
                             "bound_field": row["bound_field"],
                             "bound_rows": row["bound_rows"],
                             "p_guess": row["p_guess"], "H_min_XR": row["H_min_XR"],
                             "H_min_bX": row["H_min_bX"],
                             "passive": float(row["unconditionally_passive"]),
                             "ratio_exact": row["ratio_exact"],
                             "ratio_marginal": row["ratio_marginal"],
                             "ratio_closed_form": row["ratio_closed_form"],
                             "ratio_guess": row["ratio_guess"],
                             "negativity_AB": row["negativity_AB"]})
            print(f"  s8 L={L} g={g}: W_full {led['W_arrow_full']:.4e}, "
                  f"closed-form ratio {rows[-1]['ratio_closed_form']:.9f}")
    cols = sorted({k for r in rows for k in r})
    _save("s8_entropic.npz", **{c: np.array([r[c] for r in rows], float) for c in cols})
    # smoothing: normalized vs Tomamichel's sub-normalized ball (Prop. E)
    grid = (0.0, 0.01, 0.03, 0.1, 0.2, 0.3)
    smooth = []
    rng = np.random.default_rng(17)
    for n in (1, 2, 3):
        for kind in ("pure", "mixed"):
            rho = R.random_state(n, rng, kind)
            for eps in (0.05, 0.2, 0.4):
                smooth.append({"n": n, "kind": kind, "eps": eps,
                               "H_norm": R.smooth_min_entropy(rho, eps),
                               "H_sub": R.smooth_min_entropy(rho, eps, subnormalized=True)})
    for n in (1, 2):
        rho = np.eye(2**n) / 2**n
        for eps in (0.2, 0.4):
            smooth.append({"n": n, "kind": "maximally_mixed", "eps": eps,
                           "H_norm": R.smooth_min_entropy(rho, eps),
                           "H_sub": R.smooth_min_entropy(rho, eps, subnormalized=True)})
    sub_rows = []
    for h, k in [(1.0, 0.5), (0.3, 2.0)]:
        H = hotta.hamiltonian(h, k)
        g0 = hotta.ground_state(h, k)
        pot = R.qet_potential(H, np.outer(g0, g0.conj()), (0,))
        br = [(b["p"], b["sigma_B"]) for b in pot["branches"]]
        hs = [b["H_eff"] for b in pot["branches"]]
        a = R.one_shot_bound(hs, br, eps_grid=grid)
        b = R.one_shot_bound(hs, br, eps_grid=grid, subnormalized=True)
        sub_rows.append({"case": f"hotta h={h} k={k}", "E_B": hotta.e_b_optimal(h, k),
                         "norm_best": a["bound"], "norm_eps": a["eps_best"],
                         "sub_best": b["bound"], "sub_eps": b["eps_best"],
                         "sub_valid": bool(np.all(hotta.e_b_optimal(h, k) <= b["bound_eps"] + 1e-12))})
    finite = [r for r in rows if np.isfinite(r["ratio_marginal"])]
    summ = {"n_rows": len(rows), "L_max": int(max(r["L"] for r in rows)),
            "max_ratio_closed_form": float(np.nanmax([r["ratio_closed_form"] for r in rows])),
            "min_ratio_closed_form": float(np.nanmin([r["ratio_closed_form"] for r in rows])),
            "max_ratio_marginal": float(np.nanmax([r["ratio_marginal"] for r in finite])) if finite else None,
            "max_ratio_region": float(np.nanmax([r["ratio_exact"] for r in rows])),
            "max_ratio_guess": float(np.nanmax([r["ratio_guess"] for r in rows])),
            "all_rows_passive": bool(all(r["passive"] > 0.5 for r in rows)),
            "any_bound_violation": bool(any(
                r["E_B"] > r["bound_exact"] + 1e-12 or r["E_B"] > r["bound_guess"] + 1e-12
                or r["E_B"] > r["exact_unitary"] + 1e-10 for r in rows)),
            "smoothing": smooth, "subnormalized_bound": sub_rows,
            "max_subnorm_gain_generic": float(max(
                s["H_sub"] - s["H_norm"] for s in smooth if s["kind"] != "maximally_mixed")),
            "subnorm_gain_at_saturation": [
                {"eps": s["eps"], "gain": s["H_sub"] - s["H_norm"],
                 "predicted": float(-np.log(1 - s["eps"] ** 2))}
                for s in smooth if s["kind"] == "maximally_mixed"],
            "subnormalized_ever_helps": bool(any(r["sub_best"] < r["norm_best"] - 1e-12
                                                 for r in sub_rows))}
    _json("s8_entropic_summary.json", summ)
    sel = [r for r in rows if r["L"] == max(r2["L"] for r2 in rows) and r["g"] == 1.0]
    d = [r["d"] for r in sel]
    series = [{"x": d, "y": [r["E_B"] for r in sel], "label": "E_B achieved"},
              {"x": d, "y": [r["exact_unitary"] for r in sel], "label": "Prop. B closed form (exact)"},
              {"x": d, "y": [r["bound_exact"] for r in sel], "label": "region-unitary bound"},
              {"x": d, "y": [r["bound_guess"] for r in sel], "label": "entropic bound, Prop. C"}]
    _svg_lines(DRAFT / f"fig_chain_L{max(r['L'] for r in rows)}_bounds.svg", series,
               f"Critical TFIM L={max(r['L'] for r in rows)}: exact vs entropic",
               "distance |A-B|", "energy (J units)", logy=True,
               hlines=[(sel[0]["W_full"], "W_-> (Bob = complement)")])
    print(f"  s8 summary: closed-form ratio in [{summ['min_ratio_closed_form']:.9f}, "
          f"{summ['max_ratio_closed_form']:.9f}], entropic max {summ['max_ratio_guess']:.4f}")
    return summ


def main(argv=None):
    global DATA, DRAFT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="smoke configuration (~40 s)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: data/ with figures in draft/; a fresh temporary "
                         "directory for --quick, so a smoke run never overwrites the archive)")
    args = ap.parse_args(argv)
    if args.out:
        DATA = Path(args.out)
        DRAFT = DATA / "draft"
    elif args.quick:
        DATA = Path(tempfile.mkdtemp(prefix="resource-theory-quick-"))
        DRAFT = DATA / "draft"
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    DRAFT.mkdir(parents=True, exist_ok=True)
    summary = {"build": build_info(), "quick": bool(args.quick)}
    rows, gauss = s1_monotonicity(args.quick)
    summary["s1"] = {"worst_violation_all": float(max(max(r["worst_violation"] for r in rows),
                                                      max(g["worst_violation"] for g in gauss))),
                     "total_samples": int(sum(r["n_samples"] for r in rows) + sum(g["n_samples"] for g in gauss)),
                     "min_teeth_nonfree": float(min(r["teeth_nonfree_instrument"] for r in rows)),
                     "min_teeth_injection": float(min(r["teeth_bob_injection"] for r in rows)),
                     "max_signalling_defect": float(max(r["max_signalling_defect"] for r in rows)),
                     "all_passed": bool(all(r["passed"] for r in rows) and all(g["passed"] for g in gauss))}
    summary["s2"] = s2_hotta(args.quick)
    summary["s3"] = s3_chains(args.quick)
    summary["s4"] = s4_surplus(args.quick)
    summary["s5"] = s5_gaussian(args.quick)
    summary["s6"] = s6_two_way(args.quick)
    summary["s7"] = s7_free_class(args.quick)
    summary["s8"] = s8_entropic(args.quick)
    summary["wall_clock_s"] = time.time() - t0
    _json("summary.json", summary)
    print(f"done in {summary['wall_clock_s']:.1f}s -> {DATA/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
