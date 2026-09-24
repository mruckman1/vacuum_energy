#!/usr/bin/env python
"""Toy Jacobson: entanglement equilibrium as a constraint on lattice dynamics
(PLAN.md Layer 4 discovery target, Layer 7 leap L2).

Re-runnable end to end, headless, no hidden state:

    .venv/bin/python papers/toy-jacobson/notebook.py          # full, ~15 min
    .venv/bin/python papers/toy-jacobson/notebook.py --quick  # smoke, ~2 min

THE CLAIM (one paragraph)
=========================
On a 1+1 harmonic chain with no gravity anywhere in the definition, impose
Jacobson's entanglement-equilibrium condition (PRL 116, 201101 (2016)) in
its only non-vacuous lattice form: for every small centred ball and every
long-wavelength (reflection-odd) squeeze of the vacuum, the exact first-law
entropy change delta S must equal delta<K_loc> with K_loc the LOCAL
Casini-Huerta-Myers form 2 pi sum_i beta_i h_i built from the lattice's own
energy density. The residual r(K) = max |delta S - delta<K_loc>|/|delta S|
vanishes for the massless nearest-neighbour wave chain as the ball grows
(a measured power of 1/L, continuum extrapolation consistent with zero),
stays at order one for every non-relativistic dispersion (Lifshitz
omega ~ k^2, omega ~ sqrt(k), power-law couplings 1/n^2), equals the
velocity mismatch |1 - v| exactly, grows with the mass as (mL)^~1, and
across a family of lattice dynamics tracks the departure of the bulk
dispersion from omega = k at the ball's own scale k = pi/L up to an
ansatz-discretisation floor ~ 1/L^2. Minimising r(K) over the coupling
family K = m^2 + c_1 L_1 + c_2 L_2 from generic starts recovers the wave
equation's velocity (v^2 = c_1 + 4 c_2 -> 1) and masslessness (m -> 0)
sharply, but NOT the nearest-neighbour Laplacian: the coupling range c_2 is
a flat valley of the residual within |c_2| <~ 0.3 at ball sizes L <= 16 --
the constraint pins the low-energy dispersion, not the lattice
discretisation. Two structural facts limit the toy and are measured here:
reflection-EVEN variations have an O(1) residual for every dynamics on the
massless chain (the 2d scalar's zero mode, whose entropy no local energy
density can see), and the residual has a secondary dip at masses m ~ 4
lattice units where the ball is nearly a product state (excluded by
Jacobson's premise that the UV entanglement is that of the gapless vacuum,
imposed as a mass bound in the fit).

All of that ASSUMES the CHM weight, which is derived from conformal
symmetry -- so r -> 0 for the wave chain and r = O(1) for Lifshitz is a
consistency check on CHM, not Lorentz invariance emerging from
entanglement (an outside reviewer's objection, quoted in the MANIFEST).
Section G frees the weight: K_loc = sum_k w_k O_k over the lattice's own
local energy pieces with w >= 0 from a low-parameter smooth family (or one
weight per site), minimised by linear programming over a variation family
that includes two-mode squeezes and whose numerical rank is reported next
to the parameter count (Jacobson's regime, ball << wavelength, is
low-rank; a per-site weight fits any dynamics and says nothing). With the
weight free the wave chain's minimiser IS the CHM parabola -- found, not
assumed -- and its residual drops below the CHM one; the Lifshitz chain
is distinguished by the COMPLEXITY of the admissible weight, not by its
existence: a scale-covariant nonnegative profile fits it too, but only at
>= 11-12 modes of a bumpy shape (below 1e-2 by 11-13 modes and below 1e-3
by 12-14, z-dependent; 5e-4 only by 20 Chebyshev modes; 0.09 at 8 modes),
against 3 Chebyshev modes (the parabola) for the wave chain. The caveat is
part of the claim: with enough modes EVERY tested dynamics fits, so the toy
tests the parsimony of the modular weight -- how many modes delta S =
delta<K_loc> requires -- and does not establish that Lorentz-invariant
dynamics are singled out by the existence of a local modular Hamiltonian.
The velocity is exactly absorbed (a unit: the vacuum of K and of
v^2 K is the same state), the mass is absorbed per ball but not by one
profile shared across ball sizes, and the constrained fit with the weight
free pins m/v -> 0 and the coupling range only through that shared
profile. What the toy does NOT say: nothing about an area term, the
Einstein equation, d > 2, or balls off the chain centre. None of the
numbers is restated in this docstring, so the file's sha256 (recorded in
every archive) identifies the code that produced them.

WHAT GOES THROUGH WHAT
======================
Every modular Hamiltonian is the exact Gaussian one of vacuum.core
(Williamson); delta S is its first-law pairing (1/2) Tr[G_B delta V_B] and is
checked by independent finite differences of the entropy of the exactly
squeezed ball state (vacuum.geometry.jacobson.first_law_audit, precision
ladder float64 -> mpmath of vacuum.core.precision). The local ansatz,
variations, residual and constrained fit are vacuum.geometry.jacobson (its
docstring states the definitions and the references); the locality_score
cross-check is vacuum.modular. Conventions per docs/API.md: hbar = 1,
vacuum nu = 1/2, block quadrature ordering, entropies in nats.
"""

from __future__ import annotations

import argparse
import tempfile
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import mpmath
import numpy as np
import scipy
from scipy.optimize import minimize_scalar

import vacuum
from vacuum.core import entropy
from vacuum.geometry.jacobson import (
    _vacuum_block,
    bulk_dispersion,
    bw_weights,
    centred_ball,
    coupling_chain_K,
    dispersion_departure,
    dispersion_power_K,
    eh_profiles,
    equilibrium_residual,
    first_law_audit,
    fit_dynamics,
    free_weight_residual,
    pair_variation,
    random_symmetric_chain_K,
    residual_scan,
    smooth_variations,
    universal_weight_residual,
)

RATIO_LABEL = {4: "lam=4L", 8: "lam=8L", 16: "lam=16L"}

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RATIOS = (4, 8, 16)


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=HERE, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def build_info(mode, wall):
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "generator": "papers/toy-jacobson/notebook.py",
        "generator_sha256": sha,
        "mode": mode,
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": wall,
        "package": f"vacuum {getattr(vacuum, '__version__', '0.1.0')}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mpmath": mpmath.__version__,
        "platform": platform.platform(),
    }


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def rows_to_arrays(rows, prefix=""):
    keys = ["L", "ratio", "index", "wavelength", "parity", "dS", "dK", "r",
            "top_mode_share", "floor_share", "n_floor", "escalated", "nu_min_margin"]
    return {prefix + k: np.array([row[k] for row in rows]) for k in keys}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Section A -- the massless wave chain: anchor 1, the zero-mode obstruction,
# the exact modular Hamiltonian's profiles, the constrained velocity
# ---------------------------------------------------------------------------


def section_A(cfg):
    N = cfg["N_A"]
    sizes = cfg["sizes_A"]
    log(f"A: massless chain N={N}, sizes={sizes}")
    K = coupling_chain_K(N, [1.0])
    eig = np.linalg.eigh(K)
    odd = equilibrium_residual(K, sizes=sizes, ratios=RATIOS, parity="odd", eig=eig,
                               return_details=True)
    even = equilibrium_residual(K, sizes=cfg["sizes_even"], ratios=RATIOS, parity="even",
                                eig=eig, return_details=True)
    Ls = np.array(sizes, dtype=float)
    r = np.array([odd["per_size"][L] for L in sizes])
    sel = Ls >= 8
    slope, logA = np.polyfit(np.log(Ls[sel]), np.log(r[sel]), 1)
    fit2 = np.linalg.lstsq(np.vstack([np.ones(sel.sum()), Ls[sel] ** -2.0]).T, r[sel], rcond=None)[0]
    fitp = np.linalg.lstsq(np.vstack([np.ones(sel.sum()), Ls[sel] ** slope]).T, r[sel], rcond=None)[0]
    per_ratio = {}
    for m in RATIOS:
        rr = np.array([row["r"] for row in odd["rows"] if row["ratio"] == m])
        LL = np.array([row["L"] for row in odd["rows"] if row["ratio"] == m], dtype=float)
        s = LL >= 8
        per_ratio[m] = {"L": LL, "r": rr, "slope": float(np.polyfit(np.log(LL[s]), np.log(rr[s]), 1)[0])}
    # exact-EH profiles (Di Giulio-Tonni combination of all diagonals) and locality
    profiles = {}
    for L in cfg["profile_sizes"]:
        p = eh_profiles(K, L, eig=eig)
        profiles[L] = p
    # constrained velocity: minimise r over K = v^2 L at fixed ball size
    lam, U = eig
    vstar = {}
    for L in cfg["vstar_sizes"]:
        f = lambda v, L=L: equilibrium_residual(v * v * K, sizes=(L,), ratios=RATIOS,  # noqa: E731
                                                eig=(v * v * lam, U), escalate_above=1e-3)
        res = minimize_scalar(f, bounds=(0.8, 1.2), method="bounded", options={"xatol": 1e-5})
        vstar[L] = {"v_star": float(res.x), "r_star": float(res.fun), "r_v1": float(f(1.0))}
    out = {
        "N": N, "sizes": np.array(sizes), "per_size_r": r,
        "slope_L_ge_8": float(slope), "amplitude": float(np.exp(logA)),
        "r_inf_fit_inverse_L2": float(fit2[0]), "A_fit_inverse_L2": float(fit2[1]),
        "r_inf_fit_power": float(fitp[0]),
        "per_ratio": per_ratio,
        "even_sizes": np.array(cfg["sizes_even"]),
        "even_per_size_r": np.array([even["per_size"][L] for L in cfg["sizes_even"]]),
        "odd_top_mode_share_max": float(max(abs(row["top_mode_share"]) for row in odd["rows"])),
        "profiles": profiles, "vstar": vstar,
        "odd_rows": odd["rows"], "even_rows": even["rows"],
    }
    log(f"A: r(L) = {dict(zip(sizes, [float(x) for x in np.round(r, 5)]))}; slope {slope:.3f}; "
        f"r_inf {fit2[0]:.2e} (1/L^2), {fitp[0]:.2e} (L^slope); even {out['even_per_size_r']}")
    log(f"A: v*(L) = { {L: round(v['v_star'], 5) for L, v in vstar.items()} }")
    np.savez(DATA / "A_massless_chain.npz", **{
        "N": N, "sizes": out["sizes"], "per_size_r": r, "slope_L_ge_8": slope,
        "r_inf_fit_inverse_L2": fit2[0], "A_fit_inverse_L2": fit2[1], "r_inf_fit_power": fitp[0],
        "even_sizes": out["even_sizes"], "even_per_size_r": out["even_per_size_r"],
        "vstar_L": np.array(list(vstar)), "vstar_v": np.array([v["v_star"] for v in vstar.values()]),
        "vstar_r": np.array([v["r_star"] for v in vstar.values()]),
        "vstar_r_v1": np.array([v["r_v1"] for v in vstar.values()]),
        **rows_to_arrays(odd["rows"], "odd_"), **rows_to_arrays(even["rows"], "even_"),
        **{f"profile_L{L}_{k}": v for L, p in profiles.items() for k, v in p.items() if k != "locality"},
        **{f"profile_L{L}_locality_bw{b}": s for L, p in profiles.items() for b, s in p["locality"].items()},
        "__build__": cfg["sha"],
    })
    return out


# ---------------------------------------------------------------------------
# the dynamics family (shared by sections B and C)
# ---------------------------------------------------------------------------


def long_range_c(alpha, n_max=40):
    n = np.arange(1, n_max + 1)
    c = n ** (-float(alpha))
    return c / np.sum(c * n ** 2)


def make_family(N, quick):
    """name -> (K, symbol) with symbol = ('coupling', c, m) | ('power', s) | None."""
    fam = {}
    fam["wave"] = (coupling_chain_K(N, [1.0]), ("coupling", [1.0], 0.0))
    masses = [0.05, 0.2, 1.0, 4.0] if quick else [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    for m in masses:
        fam[f"mass_{m:g}"] = (coupling_chain_K(N, [1.0], m=m), ("coupling", [1.0], m))
    vels = [0.5, 0.8, 1.25, 2.0] if quick else [0.5, 0.8, 0.9, 1.1, 1.25, 2.0]
    for v in vels:
        fam[f"vel_{v:g}"] = (coupling_chain_K(N, [v * v]), ("coupling", [v * v], 0.0))
    c2s = [-1.0, -1 / 12, 0.1] if quick else [-3.0, -1.0, -0.3, -0.2, -0.15, -1 / 12, -0.04, 0.05, 0.1, 0.2]
    for c2 in c2s:
        c = [1.0 - 4.0 * c2, c2]  # v = 1; k^4 coefficient of omega^2 is -1/12 - c2
        fam[f"nnn_{c2:+.4f}"] = (coupling_chain_K(N, c), ("coupling", c, 0.0))
    exps = [0.5, 2.0] if quick else [0.5, 0.75, 1.25, 1.5, 2.0]
    for s in exps:
        fam[f"z_{s:g}"] = (dispersion_power_K(N, s), ("power", s))
    alphas = [2.0, 4.0] if quick else [2.0, 3.0, 4.0, 6.0]
    for a in alphas:
        c = long_range_c(a)
        fam[f"longrange_{a:g}"] = (coupling_chain_K(N, c), ("coupling", c, 0.0))
    Ws = [0.3] if quick else [0.1, 0.3, 0.6]
    seeds = [0] if quick else [0, 1, 2]
    for W in Ws:
        for seed in seeds:
            fam[f"random_W{W:g}_s{seed}"] = (random_symmetric_chain_K(N, W, seed=seed), None)
    return fam


def departure(symbol, L, kind):
    if symbol is None:
        return np.nan
    if symbol[0] == "coupling":
        return dispersion_departure(symbol[1], symbol[2], L, kind=kind)
    s = symbol[1]
    kL = np.pi / L
    omega = lambda k: (2.0 * np.sin(0.5 * k)) ** s  # noqa: E731
    if kind == "v1":
        return float(abs(omega(kL) / kL - 1.0))
    k = kL * (np.arange(1, 65) / 64.0)
    w = omega(k)
    v = float(np.dot(w, k) / np.dot(k, k))
    return float(np.linalg.norm(w - v * k) / np.linalg.norm(w))


# ---------------------------------------------------------------------------
# Section B -- anchor 2: the exact first law by independent routes
# ---------------------------------------------------------------------------


def section_B(cfg):
    N = cfg["N_B"]
    log(f"B: first-law audits N={N}, sizes={cfg['sizes_B']}")
    fam = {
        "wave": coupling_chain_K(N, [1.0]),
        "mass_0.2": coupling_chain_K(N, [1.0], m=0.2),
        "nnn_+0.1": coupling_chain_K(N, [0.6, 0.1]),
        "improved": coupling_chain_K(N, [4 / 3, -1 / 12]),
        "nnn_-1": coupling_chain_K(N, [5.0, -1.0]),
        "lifshitz": dispersion_power_K(N, 2.0) / 4.0,
        "z_0.5": dispersion_power_K(N, 0.5),
        "random_W0.5": random_symmetric_chain_K(N, 0.5, seed=1),
        "longrange_2": coupling_chain_K(N, long_range_c(2.0)),
    }
    rows = []
    for name, K in fam.items():
        eig = np.linalg.eigh(K)
        for L in cfg["sizes_B"]:
            for precise in (False, True):
                a = first_law_audit(K, L, ratio=4, eig=eig, use_precise=precise)
                rows.append({"name": name, "L": L, "precise": precise, **a})
        worst = max(r["rel_err"] for r in rows if r["name"] == name and r["precise"])
        worst_f = max(r["rel_err"] for r in rows if r["name"] == name and not r["precise"])
        log(f"B: {name:12s} max rel_err ladder {worst:.1e} | float64 {worst_f:.1e}")
    out = {"rows": rows,
           "max_rel_err_ladder": float(max(r["rel_err"] for r in rows if r["precise"])),
           "max_rel_err_float64": float(max(r["rel_err"] for r in rows if not r["precise"])),
           "lifshitz_float64": float(max(r["rel_err"] for r in rows if r["name"] == "lifshitz" and not r["precise"])),
           "lifshitz_ladder": float(max(r["rel_err"] for r in rows if r["name"] == "lifshitz" and r["precise"]))}
    np.savez(DATA / "B_first_law_audits.npz",
             name=np.array([r["name"] for r in rows]), L=np.array([r["L"] for r in rows]),
             precise=np.array([r["precise"] for r in rows]),
             dS_fd=np.array([r["dS_fd"] for r in rows]), dS_modular=np.array([r["dS_modular"] for r in rows]),
             rel_err=np.array([r["rel_err"] for r in rows]), residual=np.array([r["residual"] for r in rows]),
             nu_min_margin=np.array([r["nu_min_margin"] for r in rows]),
             N=N, __build__=cfg["sha"])
    return out


# ---------------------------------------------------------------------------
# Section C -- the map: r(K) versus departure from Lorentz invariance
# ---------------------------------------------------------------------------


def section_C(cfg):
    N = cfg["N_C"]
    sizes = cfg["sizes_C"]
    fam = make_family(N, cfg["quick"])
    log(f"C: residual map N={N}, sizes={sizes}, {len(fam)} dynamics")
    records = []
    all_rows = []
    for name, (K, symbol) in fam.items():
        eig = np.linalg.eigh(K)
        det = equilibrium_residual(K, sizes=sizes, ratios=RATIOS, eig=eig, return_details=True,
                                   escalate_above=1e-4)
        prof = eh_profiles(K, 16, eig=eig)
        for L in sizes:
            records.append({
                "name": name, "L": L, "r": det["per_size"][L],
                "r_rms": float(np.sqrt(np.mean([row["r"] ** 2 for row in det["rows"] if row["L"] == L]))),
                "dep_v1": departure(symbol, L, "v1"), "dep_shape": departure(symbol, L, "shape"),
                "locality_bw1_L16": prof["locality"][1],
                "pp_rowsum_centre_L16": float(prof["pp_rowsum_over_beta"][8]),
                "family": name.split("_")[0],
            })
        for row in det["rows"]:
            all_rows.append({"name": name, **row})
        log(f"C: {name:18s} r = { {L: f'{det['per_size'][L]:.3e}' for L in sizes} } "
            f"dep_v1(L=16) = {departure(symbol, 16, 'v1'):.3e} locality {prof['locality'][1]:.3f}")
    # the map fit r = A / L^2 + C * dep_v1 over the analytic translation-invariant
    # members in the small-departure regime (ball smaller than every other length:
    # dep_v1 <= 0.6, which keeps |1 - v| <= 0.6 and mL <~ 1.5), weighted by 1/r so that
    # relative deviations are minimised
    fit_recs = [rec for rec in records if rec["family"] in ("wave", "mass", "vel", "nnn", "longrange")
                and np.isfinite(rec["dep_v1"]) and rec["dep_v1"] <= 0.6 and rec["r"] <= 0.6]
    X = np.array([[rec["L"] ** -2.0, rec["dep_v1"]] for rec in fit_recs])
    y = np.array([rec["r"] for rec in fit_recs])
    wts = 1.0 / y
    A_C = np.linalg.lstsq(X * wts[:, None], y * wts, rcond=None)[0]
    pred = X @ A_C
    rel_scatter = float(np.sqrt(np.mean(((pred - y) / y) ** 2)))
    # ansatz floor from the most linear dispersion (improved chain, c2 = -1/12)
    floor = {rec["L"]: rec["r"] for rec in records if rec["name"] == "nnn_-0.0833"}
    # per-family tracking coefficient (r - A/L^2) / dep_v1 where the departure exceeds the floor
    tracking = {}
    for famname in ("vel", "mass", "nnn", "longrange", "z"):
        vals = [(rec["r"] - A_C[0] * rec["L"] ** -2.0) / rec["dep_v1"] for rec in records
                if rec["family"] == famname and np.isfinite(rec["dep_v1"])
                and rec["dep_v1"] > 2.0 * A_C[0] * rec["L"] ** -2.0 and rec["dep_v1"] <= 0.6]
        tracking[famname] = {"median": float(np.median(vals)) if vals else np.nan,
                             "min": float(np.min(vals)) if vals else np.nan,
                             "max": float(np.max(vals)) if vals else np.nan, "n": len(vals)}
    out = {"records": records, "rows": all_rows, "fit_A": float(A_C[0]), "fit_C": float(A_C[1]),
           "fit_rel_scatter": rel_scatter, "n_fit": len(fit_recs), "ansatz_floor": floor,
           "tracking": tracking}
    log(f"C: fit r = {A_C[0]:.3f}/L^2 + {A_C[1]:.3f} * dep_v1 over {len(fit_recs)} points, "
        f"rms relative scatter {rel_scatter:.2f}; ansatz floor {floor}; tracking {tracking}")
    np.savez(DATA / "C_residual_map.npz",
             name=np.array([rec["name"] for rec in records]), family=np.array([rec["family"] for rec in records]),
             L=np.array([rec["L"] for rec in records]), r=np.array([rec["r"] for rec in records]),
             r_rms=np.array([rec["r_rms"] for rec in records]),
             dep_v1=np.array([rec["dep_v1"] for rec in records]), dep_shape=np.array([rec["dep_shape"] for rec in records]),
             locality_bw1_L16=np.array([rec["locality_bw1_L16"] for rec in records]),
             pp_rowsum_centre_L16=np.array([rec["pp_rowsum_centre_L16"] for rec in records]),
             fit_A=A_C[0], fit_C=A_C[1], fit_rel_scatter=rel_scatter,
             row_name=np.array([row["name"] for row in all_rows]),
             **rows_to_arrays(all_rows, "row_"), N=N, __build__=cfg["sha"])
    with open(DATA / "C_residual_map.json", "w") as fh:
        json.dump(jsonable({"records": records, "fit": {"A": A_C[0], "C": A_C[1], "rel_scatter": rel_scatter}}), fh, indent=1)
    return out


# ---------------------------------------------------------------------------
# Section D -- constrained dynamics: where the minimiser lands
# ---------------------------------------------------------------------------


def section_D(cfg):
    N = cfg["N_D"]
    sizes = cfg["sizes_D"]
    log(f"D: constrained dynamics N={N}, sizes={sizes}")
    K_of = lambda th: coupling_chain_K(N, [th[0], th[1]], m=abs(th[2]))  # noqa: E731
    bounds = [(0.05, 4.0), (-0.5, 0.5), (0.0, 1.0)]  # m <= 1: Jacobson's UV-entanglement premise
    r_wave = equilibrium_residual(coupling_chain_K(N, [1.0]), sizes=sizes, ratios=RATIOS, escalate_above=1e-3)
    landings = []
    for th0 in cfg["starts"]:
        t = time.time()
        th, info = fit_dynamics(K_of, th0, sizes=sizes, ratios=RATIOS, bounds=bounds,
                                maxfev=cfg["maxfev"], restarts=cfg["restarts"])
        c1, c2, m = float(th[0]), float(th[1]), float(abs(th[2]))
        rec = {"start": list(map(float, th0)), "c1": c1, "c2": c2, "m": m, "v2": c1 + 4 * c2,
               "k4_coeff": -c1 / 12.0 - 4.0 * c2 / 3.0, "residual": info["residual"],
               "nfev": info["nfev"], "seconds": time.time() - t,
               "per_size": info["details"]["per_size"]}
        landings.append(rec)
        log(f"D: start {th0} -> c1={c1:+.4f} c2={c2:+.4f} m={m:.2e} v^2={rec['v2']:.4f} "
            f"r={rec['residual']:.3e} (wave {r_wave:.3e}) nfev={rec['nfev']}")
    best = min(landings, key=lambda rec: rec["residual"])
    theta_star = np.array([best["c1"], best["c2"], best["m"]])
    kw = dict(sizes=sizes, ratios=RATIOS, escalate_above=1e-3)
    scans = {}
    for axis, name, deltas in ((0, "c1", np.linspace(-0.15, 0.15, 61)),
                               (1, "c2", np.linspace(-0.4, 0.4, 41)),
                               (2, "m", np.linspace(0.0, 0.5, 51) - theta_star[2])):
        d, rr = residual_scan(K_of, theta_star, axis, deltas, **kw)
        scans[name] = {"delta": d, "r": rr}
    # the valley in c2 at v = 1 exactly (c1 = 1 - 4 c2)
    c2_grid = np.linspace(-0.4, 0.22, 32)  # c1 = 1 - 4 c2 > 0 needs c2 < 1/4
    valley = []
    for c2 in c2_grid:
        try:
            valley.append(equilibrium_residual(coupling_chain_K(N, [1 - 4 * c2, c2]), **kw))
        except ValueError:
            valley.append(np.nan)
    valley = np.array(valley)
    # the gapped branch at v = 1
    m_grid = np.array(cfg["m_grid"])
    gapped, S8 = [], []
    for m in m_grid:
        K = coupling_chain_K(N, [1.0], m=m)
        eig = np.linalg.eigh(K)
        gapped.append(equilibrium_residual(K, sizes=cfg["sizes_gap"], ratios=RATIOS, eig=eig, escalate_above=1e-3))
        S8.append(entropy(_vacuum_block(eig, centred_ball(N, 8))))
    gapped, S8 = np.array(gapped), np.array(S8)
    # resolution: width of the region around the minimum where r < 2 r_min, with the
    # crossings located by linear interpolation between grid points
    def width(delta, rr):
        delta, rr = np.asarray(delta, dtype=float), np.asarray(rr, dtype=float)
        ok = np.isfinite(rr)
        delta, rr = delta[ok], rr[ok]
        if rr.size < 3:
            return np.nan
        i0 = int(np.argmin(rr))
        thr = 2.0 * rr[i0]
        lo, hi = delta[0], delta[-1]
        for i in range(i0, 0, -1):
            if rr[i - 1] >= thr:
                lo = delta[i - 1] + (thr - rr[i - 1]) * (delta[i] - delta[i - 1]) / (rr[i] - rr[i - 1])
                break
        for i in range(i0, rr.size - 1):
            if rr[i + 1] >= thr:
                hi = delta[i] + (thr - rr[i]) * (delta[i + 1] - delta[i]) / (rr[i + 1] - rr[i])
                break
        return float(hi - lo)
    out = {"r_wave": float(r_wave), "landings": landings, "best": best, "scans": scans,
           "c2_valley_grid": c2_grid, "c2_valley_r": valley,
           "c2_valley_width_2rmin": width(c2_grid, valley),
           "c1_width_2rmin": width(scans["c1"]["delta"], scans["c1"]["r"]),
           "m_width_2rmin": width(scans["m"]["delta"] + theta_star[2], scans["m"]["r"]),
           "m_grid": m_grid, "gapped_r": gapped, "gapped_S8": S8,
           "gapped_dip_m": float(m_grid[np.argmin(np.where(m_grid >= 2.0, gapped, np.inf))]),
           "gapped_dip_r": float(np.min(np.where(m_grid >= 2.0, gapped, np.inf)))}
    log(f"D: best landing {best}; c2 valley (v=1) r in [{valley.min():.3e}, {valley.max():.3e}] "
        f"width(2 r_min) {out['c2_valley_width_2rmin']:.2f}; gapped dip m={out['gapped_dip_m']} r={out['gapped_dip_r']:.3e}")
    np.savez(DATA / "D_constrained_dynamics.npz",
             r_wave=r_wave, start=np.array([l["start"] for l in landings]),
             landing=np.array([[l["c1"], l["c2"], l["m"]] for l in landings]),
             landing_v2=np.array([l["v2"] for l in landings]), landing_r=np.array([l["residual"] for l in landings]),
             landing_nfev=np.array([l["nfev"] for l in landings]),
             theta_star=theta_star,
             **{f"scan_{k}_{kk}": v for k, sc in scans.items() for kk, v in sc.items()},
             c2_valley_grid=c2_grid, c2_valley_r=valley,
             m_grid=m_grid, gapped_r=gapped, gapped_S8=S8, N=N, __build__=cfg["sha"])
    return out



# ---------------------------------------------------------------------------
# Section E -- the ansatz discretisation floor and the k^4 coefficient
# ---------------------------------------------------------------------------


def section_E(cfg):
    """Separate the local ansatz's own discretisation from the k^4 dispersion.

    (E1) Floor. The CHM weight can be attached to the lattice energy density in
    more than one way; all agree in the continuum and differ at O(1/L^2). Three
    are compared on the massless nearest-neighbour chain: beta at the site
    centres ('site', the original ansatz), beta at the bond midpoints ('bond',
    the midpoint rule for the gradient energy -- which also makes the bonds
    straddling dB drop out, since their midpoint sits exactly on the boundary),
    and the Richardson-style convex combinations of the two.

    (E2) k^4. Within the one-parameter family K = (1 - 4c_2) L_1 + c_2 L_2 at
    v = 1 the k^4 coefficient A_4 = sum_n c_n n^4 = 1 + 12 c_2 is an affine
    function of c_2, so *every* response linear in the couplings is degenerate
    with the dispersion and no k^4 coefficient can be extracted. Adding the
    third-neighbour coupling breaks the degeneracy: with c_1 + 4c_2 + 9c_3 = 1,
        A_4 = 1 + 12 c_2 + 72 c_3,  omega(k)/k - 1 = -A_4 k^2/24,
    so a response that is a function of the dispersion alone must have
    d rho/d c_3 = 6 d rho/d c_2 -- a falsifiable prediction. The signed residual
    rho = (dS - dK)/dS is fitted over the (c_2, c_3) grid as
    rho = rho_0 + l_2 c_2 + l_3 c_3, the ratio l_3/l_2 is compared with 6, and
    the sensitivity is quoted as C = coefficient of (omega(k_L)/k_L - 1) at the
    ball scale k_L = pi/L.
    """
    N = cfg["N_E"]
    sizes = cfg["sizes_E"]
    log(f"E: ansatz floor and k^4, N={N}, sizes={sizes}")
    K = coupling_chain_K(N, [1.0])
    eig = np.linalg.eigh(K)
    floor = {}
    for ba, mix in cfg["ansatz_variants"]:
        det = equilibrium_residual(K, sizes=sizes, ratios=RATIOS, eig=eig, return_details=True,
                                   escalate_above=1e-3, beta_at=ba, mix=mix)
        key = ba if ba != "mix" else f"mix{mix:g}"
        floor[key] = {"per_size": {int(L): float(v) for L, v in det["per_size"].items()},
                      "A_of_L2": {int(L): float(v) * L ** 2 for L, v in det["per_size"].items()},
                      "rho": {f"{r['L']}_{r['ratio']:.0f}": float(r["rho"]) for r in det["rows"]}}
        log(f"E1: beta_at={key:8s} r(L) = { {L: f'{v:.3e}' for L, v in floor[key]['per_size'].items()} } "
            f"A = { {L: round(v, 4) for L, v in floor[key]['A_of_L2'].items()} }")
    grid = [(c2, c3) for c2 in cfg["c2_grid"] for c3 in cfg["c3_grid"]]
    used, rho = [], {}
    for c2, c3 in grid:
        c1 = 1.0 - 4.0 * c2 - 9.0 * c3
        try:
            Kg = coupling_chain_K(N, [c1, c2, c3])
            eg = np.linalg.eigh(Kg)
            if eg[0][0] <= 1e-9:
                raise ValueError("not positive definite")
        except (ValueError, np.linalg.LinAlgError):
            continue
        used.append((c2, c3))
        for ba in ("site", "bond"):
            det = equilibrium_residual(Kg, sizes=sizes, ratios=RATIOS, eig=eg, return_details=True,
                                       escalate_above=1e-3, beta_at=ba)
            for r in det["rows"]:
                rho.setdefault((ba, r["L"], int(r["ratio"])), []).append(r["rho"])
    c2v = np.array([g[0] for g in used])
    c3v = np.array([g[1] for g in used])
    A4 = 1.0 + 12.0 * c2v + 72.0 * c3v
    k4 = {}
    for key, vals in sorted(rho.items()):
        ba, L, rat = key
        y = np.array(vals)
        X = np.vstack([np.ones_like(c2v), c2v, c3v]).T
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        rms3 = float(np.sqrt(np.mean((X @ c - y) ** 2)))
        XA = np.vstack([np.ones_like(A4), A4]).T
        cA = np.linalg.lstsq(XA, y, rcond=None)[0]
        rmsA = float(np.sqrt(np.mean((XA @ cA - y) ** 2)))
        kL = np.pi / L
        k4[f"{ba}_L{L}_r{rat}"] = {
            "l2": float(c[1]), "l3": float(c[2]), "l3_over_l2": float(c[2] / c[1]),
            "rms_3par": rms3, "rms_A4_only": rmsA,
            "C_from_c2": float(-c[1] * 24.0 / (12.0 * kL ** 2)),
            "C_from_c3": float(-c[2] * 24.0 / (72.0 * kL ** 2)),
            "C_A4_fit": float(-cA[1] * 24.0 / kL ** 2),
            "floor_at_A4_0": float(c[0] - c[1] / 12.0),
        }
        log(f"E2: {ba:5s} L={L:2d} {RATIO_LABEL[rat]:7s} l3/l2={c[2] / c[1]:+6.3f} (dispersion: 6) "
            f"C={-cA[1] * 24.0 / kL ** 2:+.3f} rms(A4 only)={rmsA:.1e} rms(3 par)={rms3:.1e}")
    per_ratio = {}
    for rat in RATIOS:
        for ba in ("site", "bond"):
            v = [k4[f"{ba}_L{L}_r{rat}"] for L in sizes]
            per_ratio[f"{ba}_r{rat}"] = {
                "l3_over_l2_mean": float(np.mean([q["l3_over_l2"] for q in v])),
                "l3_over_l2_sd": float(np.std([q["l3_over_l2"] for q in v])),
                "C_mean": float(np.mean([q["C_A4_fit"] for q in v])),
                "C_sd": float(np.std([q["C_A4_fit"] for q in v])),
            }
    out = {"N": N, "sizes": list(sizes), "floor": floor, "k4": k4, "per_ratio": per_ratio,
           "grid_c2": c2v, "grid_c3": c3v, "n_grid": len(used)}
    np.savez(DATA / "E_ansatz_and_k4.npz",
             sizes=np.array(sizes), grid_c2=c2v, grid_c3=c3v, A4=A4,
             variants=np.array(list(floor)),
             floor_r=np.array([[floor[k]["per_size"][L] for L in sizes] for k in floor]),
             k4_keys=np.array(list(k4)),
             k4_values=np.array([[k4[k][f] for f in ("l2", "l3", "l3_over_l2", "rms_3par", "rms_A4_only",
                                                     "C_from_c2", "C_from_c3", "C_A4_fit", "floor_at_A4_0")]
                                 for k in k4]),
             k4_value_columns=np.array(["l2", "l3", "l3_over_l2", "rms_3par", "rms_A4_only",
                                        "C_from_c2", "C_from_c3", "C_A4_fit", "floor_at_A4_0"]),
             N=N, __build__=cfg["sha"])
    return out


# ---------------------------------------------------------------------------
# Section F -- a second ball position, and the even sector with the zero mode
#              projected out rather than excluded by parity
# ---------------------------------------------------------------------------


def section_F(cfg):
    """(F1) The zero mode, projected instead of avoided. The 1+1 massless
    scalar's zero mode is removed here by projecting the ball's top Williamson
    mode out of the VARIATION (both quadratures, in the ball's Williamson
    frame), so that dS and d<K_loc> are the same linear functional of the same
    dV and no parity argument is used. Reported for reflection-odd, -even and
    parity-blind ('any') variations, and for the Lifshitz chain as a canary.

    (F2) A second ball position. A ball ATTACHED to the Dirichlet wall has one
    entangling point and the exact boundary-CFT weight is beta(y) =
    (B^2 - y^2)/(2B) with y measured from the wall; a DETACHED ball near the
    wall also carries a bilocal term to its mirror image (Eisler-Tonni-Peschel
    2022) which no local weight can represent. Both are measured, together
    with the bulk parabola used at the wrong position as a control.
    """
    N = cfg["N_F"]
    log(f"F: second position and projected zero mode, N={N}")
    K = coupling_chain_K(N, [1.0])
    eig = np.linalg.eigh(K)
    Kl = dispersion_power_K(N, 2.0) / 4.0
    eigl = np.linalg.eigh(Kl)
    sizes = cfg["sizes_F"]
    proj = {}
    for name, KK, ee in (("wave", K, eig), ("lifshitz", Kl, eigl)):
        for parity in ("odd", "even", "any"):
            for zm in ("parity", "project"):
                det = equilibrium_residual(KK, sizes=sizes, ratios=RATIOS, eig=ee, return_details=True,
                                           escalate_above=1e-3, beta_at="bond", parity=parity, zero_mode=zm)
                proj[f"{name}_{parity}_{zm}"] = {int(L): float(v) for L, v in det["per_size"].items()}
        log(f"F1: {name}: even/parity {proj[f'{name}_even_parity']} -> even/project {proj[f'{name}_even_project']}; "
            f"odd/project {proj[f'{name}_odd_project']}")
    pos = {}
    for L in sizes:
        c0 = centred_ball(N, L)[0]
        for a, label in ((0, "attached"), (2, "detached_d3"), (8, "detached_d9"), (32, "detached_d33"), (c0, "centred")):
            ball = list(range(a, a + L))
            kw = dict(balls=[ball], ratios=RATIOS, eig=eig, return_details=True, escalate_above=1e-3,
                      beta_at="bond", parity="any", zero_mode="project")
            det_w = equilibrium_residual(K, wall=-1.0, **kw)
            det_p = equilibrium_residual(K, wall=None, **kw)
            pos[f"L{L}_{label}"] = {
                "boundary_adapted": float(det_w["residual"]),
                "boundary_adapted_rows": {int(r["ratio"]): float(r["r"]) for r in det_w["rows"]},
                "bulk_parabola": float(det_p["residual"]),
            }
        log(f"F2: L={L}: attached {pos[f'L{L}_attached']['boundary_adapted']:.3e} "
            f"(bulk parabola there {pos[f'L{L}_attached']['bulk_parabola']:.3e}); "
            f"centred {pos[f'L{L}_centred']['boundary_adapted']:.3e}; "
            f"detached d=9 {pos[f'L{L}_detached_d9']['boundary_adapted']:.3e}")
    out = {"N": N, "sizes": list(sizes), "projection": proj, "positions": pos}
    np.savez(DATA / "F_second_position_even.npz",
             sizes=np.array(sizes),
             proj_keys=np.array(list(proj)),
             proj_values=np.array([[proj[k][L] for L in sizes] for k in proj]),
             pos_keys=np.array(list(pos)),
             pos_values=np.array([[pos[k]["boundary_adapted"], pos[k]["bulk_parabola"]] for k in pos]),
             pos_value_columns=np.array(["boundary_adapted", "bulk_parabola"]),
             N=N, __build__=cfg["sha"])
    return out


# ---------------------------------------------------------------------------
# Section G -- the weight set free: is there ANY local modular Hamiltonian?
# ---------------------------------------------------------------------------


def section_G(cfg):
    """The reviewer's version of the test (MANIFEST, "The motivating objection").

    Sections A-F pair delta S with ONE local form, the CHM parabola, which is
    derived from conformal symmetry; that it works for the wave chain and
    fails for Lifshitz is a consistency check on CHM. Here the weight is a
    free variable: K_loc = sum_k w_k O_k over the lattice's local energy
    pieces (site terms and bonds of the PSD bond split), w >= 0 from a
    low-parameter smooth family (even Chebyshev profiles at the sites or at
    the bond midpoints -- the two discretisations of Sec. 5b) or one weight
    per site, and r_free(K) = min over the family of the max relative
    first-law residual (a linear programme) over the long-wavelength odd
    single- and two-mode squeezes of wavelength >= 4L.

    (G1) r versus r_free across the dynamics family, with the rank of the
         variation family next to the parameter count (Jacobson's regime is
         low-rank: a per-site weight fits ANY dynamics).
    (G2) The wave chain's minimising weight against the CHM parabola:
         Chebyshev coefficients in units of pi R/2 (CHM = (1, -1, 0)),
         overlap and normalisation -- a result now, not an input.
    (G3) Lifshitz: the best nonnegative smooth weight and the best signed
         one; the universal test (one profile R^z f(x/R) for L = 8, 16, 32,
         z in {1, 2}) which over-determines even an 8-mode profile.
    (G4) constrain_dynamics with the weight free, from the same generic
         starts as Sec. 5, under the per-ball and the universal residual,
         with scans along v, m and c_2.
    (G5) The two-mode squeezes obey the exact first law (finite differences
         of the exactly transformed state against (1/2) Tr[G_B delta V_B]).
    (G6) The mode ladder of the shared profile: the number of modes of a
         nonnegative R^z f(x/R) that each dynamics needs, in three bases and
         for z = 1, 2, 3 -- the parsimony statement that replaces the
         withdrawn existence statement (adversarial audit, 2026-09-04): with
         enough modes every tested dynamics fits, and the family is then as
         large as the variation family's conditions at that level.
    """
    N = cfg["N_G"]
    sizes = cfg["sizes_G"]
    log(f"G: free-weight residual N={N}, sizes={sizes}")
    Lap = coupling_chain_K(N, [1.0])
    fam = {
        "wave": Lap,
        "vel_0.5": 0.25 * Lap,
        "mass_0.05": coupling_chain_K(N, [1.0], m=0.05),
        "mass_0.2": coupling_chain_K(N, [1.0], m=0.2),
        "mass_1": coupling_chain_K(N, [1.0], m=1.0),
        "nnn_-0.0833": coupling_chain_K(N, [4 / 3, -1 / 12]),
        "nnn_+0.1": coupling_chain_K(N, [0.6, 0.1]),
        "lifshitz": Lap @ Lap / 4.0,
        "z_0.5": dispersion_power_K(N, 0.5),
        "z_1.5": dispersion_power_K(N, 1.5),
        "longrange_2": coupling_chain_K(N, long_range_c(2.0)),
        "longrange_3": coupling_chain_K(N, long_range_c(3.0)),
        "random_W0.3_s0": random_symmetric_chain_K(N, 0.3, seed=0),
    }
    if cfg["quick"]:
        for k in ("mass_0.05", "mass_1", "nnn_-0.0833", "z_1.5", "longrange_3"):
            fam.pop(k)
    families = ("cheb1", "cheb2", "cheb3", "cheb4", "site")
    ufam = ("cheb3", "cheb4", "cheb6", "cheb8")
    table, weights, universal = {}, {}, {}
    for name, K in fam.items():
        t = time.time()
        eig = np.linalg.eigh(K)
        arch = equilibrium_residual(K, sizes=sizes, ratios=RATIOS, eig=eig, return_details=True,
                                    escalate_above=1e-4)
        d = free_weight_residual(K, sizes=sizes, eig=eig, families=families)
        u = universal_weight_residual(K, sizes=sizes, design=d["design"], families=ufam,
                                      exponents=(1.0, 2.0))
        rec = {"r_archived": {int(L): float(arch["per_size"][L]) for L in sizes}, "per_size": {}}
        for L in sizes:
            r = d[L]
            rec["per_size"][int(L)] = {
                "r_chm": {k: float(v) for k, v in r["r_chm"].items()},
                "r_free": {f"{f}/{dc}": float(v) for (f, dc), v in r["r_free"].items()},
                "r_signed": {f"{f}/{dc}": float(v) for (f, dc), v in r["r_signed"].items()},
                "n_params": {f"{f}/{dc}": int(v) for (f, dc), v in r["n_params"].items()},
                "rank": {f"{t:g}": int(v) for t, v in r["rank"].items()},
                "rank_family": {f"{f}/{dc}": {f"{t:g}": int(v) for t, v in rk.items()}
                                for (f, dc), rk in r["rank_family"].items()},
                "headline": float(d["per_size"][L]), "best": "/".join(r["best"]),
                "n_variations": int(r["n_variations"]), "n_modes": int(r["n_modes"]),
                "n_pieces": int(r["n_pieces"]), "n_floor": int(r["n_floor"]),
                "escalated": bool(r["escalated"]),
                "overlap": {f"{f}/{dc}": float(v) for (f, dc), v in r["overlap"].items()},
                "scale": {f"{f}/{dc}": float(v) for (f, dc), v in r["scale"].items()},
                "theta_over_piR2": {f"{f}/{dc}": (v / (np.pi * L / 4.0)).tolist()
                                    for (f, dc), v in r["theta"].items() if f.startswith("cheb")},
            }
            weights[(name, int(L))] = {
                "beta_chm": bw_weights(centred_ball(N, L)),
                **{f"beta_{f}_{dc}": r["beta_sites"][(f, dc)] for (f, dc) in r["beta_sites"]},
            }
        rec["universal"] = {
            "best": "/".join(str(k) for k in u["best"]), "residual": float(u["residual"]),
            "n_conditions": int(u["n_conditions"]),
            "per_ball_rank": {int(L): int(v[1e-6]) for L, v in u["per_ball_rank"].items()},
            "keys": {f"{key[0]}/{key[1]}/z{key[2]:g}": {
                         "r": float(v["r"]), "r_signed": float(v["r_signed"]), "n_params": int(v["n_params"]),
                         "per_size": {int(L): float(x) for L, x in v["per_size"].items()},
                         "overlap": {int(L): float(x) for L, x in v["overlap"].items()},
                         "scale": {int(L): float(x) for L, x in v["scale"].items()},
                         "theta": np.asarray(v["theta"]).tolist()}
                     for key, v in u.items() if isinstance(key, tuple)},
        }
        table[name] = rec
        head = {L: f"{rec['per_size'][L]['headline']:.2e}" for L in sizes}
        chm = {L: f"{min(rec['per_size'][L]['r_chm'].values()):.2e}" for L in sizes}
        log(f"G1: {name:14s} r_chm(best disc) {chm} r_free(cheb3) {head} "
            f"rank(1e-6) {[rec['per_size'][L]['rank']['1e-06'] for L in sizes]} "
            f"universal best {rec['universal']['best']} r={rec['universal']['residual']:.2e} [{time.time() - t:.0f}s]")
    # G2: the recovered weight of the wave chain
    wave = table["wave"]
    for L in sizes:
        th = wave["per_size"][L]["theta_over_piR2"]["cheb3/bond"]
        log(f"G2: wave L={L}: theta/(pi R/2) = {np.round(th, 4).tolist()} (CHM = [1, -1, 0]); "
            f"overlap {wave['per_size'][L]['overlap']['cheb3/bond']:.6f} scale {wave['per_size'][L]['scale']['cheb3/bond']:.4f}; "
            f"cheb2: {np.round(wave['per_size'][L]['theta_over_piR2']['cheb2/bond'], 4).tolist()}")
    lif = table["lifshitz"]
    for L in sizes:
        ps = lif["per_size"][L]
        log(f"G3: lifshitz L={L}: r_free(cheb3) {ps['headline']:.3e} (signed {ps['r_signed']['cheb3/bond']:.2e}), "
            f"cheb4 {ps['r_free']['cheb4/bond']:.2e}, per-site+boundary {ps['r_free']['site/bond']:.2e} "
            f"(p={ps['n_params']['site/bond']}, rank(1e-6)={ps['rank']['1e-06']}); scale {ps['scale']['cheb3/bond']:.1f}")
    log(f"G3: lifshitz universal: best {lif['universal']['best']} r={lif['universal']['residual']:.3e} "
        f"(wave: {wave['universal']['best']} r={wave['universal']['residual']:.3e})")
    # G4: constrained dynamics with the weight free
    n = cfg["N_D"]
    K_of = lambda th: coupling_chain_K(n, [th[0], th[1]], m=abs(th[2]))  # noqa: E731
    bounds = [(0.05, 4.0), (-0.5, 0.5), (0.0, 1.0)]
    fits = {}
    for mode in ("free", "universal"):
        fk = {"escalate_above": 1e-3}
        if mode == "universal":
            fk["exponents"] = (1.0, 2.0)
        fits[mode] = []
        for th0 in cfg["starts_G"]:
            t = time.time()
            th, info = fit_dynamics(K_of, th0, sizes=(8, 16), bounds=bounds, maxfev=cfg["maxfev_G"],
                                    restarts=cfg["restarts_G"], residual=mode, free_kwargs=fk)
            c1, c2, m = float(th[0]), float(th[1]), float(abs(th[2]))
            v2 = c1 + 4.0 * c2
            rec = {"start": list(map(float, th0)), "c1": c1, "c2": c2, "m": m, "v2": v2,
                   "m_over_v": m / np.sqrt(max(v2, 1e-12)), "c2_over_c1": c2 / c1,
                   "residual": float(info["residual"]), "nfev": int(info["nfev"]), "seconds": time.time() - t}
            fits[mode].append(rec)
            log(f"G4 [{mode}]: start {th0} -> c1={c1:+.4f} c2={c2:+.4f} m={m:.2e} v^2={v2:.3f} "
                f"m/v={rec['m_over_v']:.3e} c2/c1={rec['c2_over_c1']:+.4f} r={rec['residual']:.3e} "
                f"nfev={rec['nfev']} [{rec['seconds']:.0f}s]")
    scans = {}
    K_scan = {"v": [(f"{v:g}", v * v * coupling_chain_K(n, [1.0])) for v in (0.5, 0.8, 1.0, 1.25, 2.0)],
              "m": [(f"{m:g}", coupling_chain_K(n, [1.0], m=m)) for m in cfg["m_scan_G"]],
              "c2": [(f"{c2:+.4f}", coupling_chain_K(n, [1.0 - 4.0 * c2, c2])) for c2 in cfg["c2_scan_G"]]}
    for axis, items in K_scan.items():
        scans[axis] = {}
        for label, K in items:
            eig = np.linalg.eigh(K)
            d = free_weight_residual(K, sizes=(8, 16), eig=eig, families=(), signed=False, escalate_above=1e-3)
            u = universal_weight_residual(K, sizes=(8, 16), design=d["design"], families=("cheb3",),
                                          exponents=(1.0, 2.0), signed=False)
            scans[axis][label] = {"free": float(d["residual"]), "universal": float(u["residual"]),
                                  "universal_best": "/".join(str(k) for k in u["best"])}
        log(f"G4 scan {axis}: " + ", ".join(f"{k}: free {v['free']:.1e} / universal {v['universal']:.1e}"
                                              for k, v in scans[axis].items()))
    # G5: the two-mode squeezes obey the exact first law
    from scipy.linalg import expm
    from vacuum.core import Omega, ground_state_cov, reduce, entanglement_hamiltonian, modular_energy
    n5 = 200
    K5 = coupling_chain_K(n5, [1.0])
    eig5 = np.linalg.eigh(K5)
    V0 = ground_state_cov(K5)
    ball = centred_ball(n5, 8)
    var = smooth_variations(eig5, 8, ratio_min=4.0, pairs=True)
    audits = []
    for (a, b) in [var["pairs"][i] for i in (0, 1, len(var["pairs"]) // 2, len(var["pairs"]) - 1)]:
        ua, ub = eig5[1][:, a], eig5[1][:, b]
        wa, wb = np.sqrt(eig5[0][a]), np.sqrt(eig5[0][b])
        A = 0.5 * (np.outer(ua, ub) + np.outer(ub, ua))
        M = np.zeros((2 * n5, 2 * n5))
        M[:n5, n5:] = A
        M[n5:, :n5] = A
        X = Omega(n5) @ M
        dV_pred = pair_variation(ua[ball], wa, ub[ball], wb)
        V_B = reduce(V0, ball)
        G_B = entanglement_hamiltonian(V_B)
        dS_mod = modular_energy(G_B, dV_pred)

        def S_of(e):
            S = expm(e * X)
            return entropy(reduce(S @ V0 @ S.T, ball))

        eps = 4e-3
        D_c = (S_of(eps) - S_of(-eps)) / (2 * eps)
        D_f = (S_of(0.5 * eps) - S_of(-0.5 * eps)) / eps
        dS_fd = (4.0 * D_f - D_c) / 3.0
        S1 = expm(1e-6 * X)
        dV_num = (reduce(S1 @ V0 @ S1.T, ball) - V_B) / 1e-6
        audits.append({"a": int(a), "b": int(b), "dS_fd": float(dS_fd), "dS_modular": float(dS_mod),
                       "rel_err": float(abs(dS_fd - dS_mod) / abs(dS_mod)),
                       "dV_err": float(np.abs(dV_num - dV_pred).max() / np.abs(dV_pred).max())})
        log(f"G5: pair ({a}, {b}): dS_fd {dS_fd:.9e} dS_modular {dS_mod:.9e} rel_err {audits[-1]['rel_err']:.1e} "
            f"dV(first order) err {audits[-1]['dV_err']:.1e}")
    # G6: the mode ladder of the shared profile -- the parsimony statement. How many modes of a
    # nonnegative profile R^z f(x/R), shared across the balls, does each dynamics need? Two
    # bases of different kinds (Chebyshev, hat functions) and the even monomials, z = 1, 2, 3;
    # the per-ball ranks at 1e-2 / 1e-3 say when the family is as large as the conditions.
    ns = cfg["ladder_modes"]
    ladder = {"modes": list(ns)}
    for name in ("wave", "lifshitz"):
        K = fam[name]
        eig = np.linalg.eigh(K)
        d = free_weight_residual(K, sizes=sizes, eig=eig, families=(), signed=False,
                                 rank_tols=(1e-2, 1e-3, 1e-4, 1e-6))
        rec = {"per_ball_rank": {int(L): {f"{t:g}": int(v) for t, v in d[L]["rank"].items()} for L in sizes},
               "n_conditions": {f"{t:g}": int(sum(d[L]["rank"][t] for L in sizes)) for t in (1e-2, 1e-3, 1e-4, 1e-6)},
               "r": {}, "first_below": {}}
        for basis in ("cheb", "hat", "mono"):
            fams = tuple(f"{basis}{k}" for k in ns if not (basis == "mono" and k > 12))
            u = universal_weight_residual(K, sizes=sizes, design=d["design"], families=fams,
                                          exponents=(1.0, 2.0, 3.0), discretisations=("bond",), signed=False)
            for z in (1.0, 2.0, 3.0):
                row = {int(k): float(u[(f"{basis}{k}", "bond", z)]["r"]) for k in ns if (f"{basis}{k}", "bond", z) in u}
                rec["r"][f"{basis}_z{z:g}"] = row
                rec["first_below"][f"{basis}_z{z:g}"] = {
                    f"{lev:g}": int(min([k for k, v in row.items() if v < lev], default=-1)) for lev in (1e-2, 1e-3)}
                log(f"G6: {name:8s} {basis:4s} z={z:g}  " + " ".join(f"{k}:{v:.1e}" for k, v in row.items()))
        log(f"G6: {name}: conditions at 1e-2 / 1e-3 / 1e-6: {rec['n_conditions']}; first below 1e-2 / 1e-3: "
            f"{ {k: (v['0.01'], v['0.001']) for k, v in rec['first_below'].items()} }")
        ladder[name] = rec
    out = {"N": N, "sizes": list(sizes), "table": table, "weights": weights, "fits": fits, "scans": scans,
           "audits": audits, "families": list(families), "universal_families": list(ufam), "ladder": ladder}
    np.savez(DATA / "G_free_weight.npz",
             sizes=np.array(sizes), names=np.array(list(table)),
             r_archived=np.array([[table[k]["r_archived"][L] for L in sizes] for k in table]),
             r_chm_site=np.array([[table[k]["per_size"][L]["r_chm"]["site"] for L in sizes] for k in table]),
             r_chm_bond=np.array([[table[k]["per_size"][L]["r_chm"]["bond"] for L in sizes] for k in table]),
             r_free=np.array([[table[k]["per_size"][L]["headline"] for L in sizes] for k in table]),
             r_free_signed=np.array([[table[k]["per_size"][L]["r_signed"]["cheb3/bond"] for L in sizes] for k in table]),
             r_free_keys=np.array([f"{f}/{dc}" for f in families for dc in ("site", "bond")]),
             r_free_all=np.array([[[table[k]["per_size"][L]["r_free"][f"{f}/{dc}"] for f in families for dc in ("site", "bond")]
                                   for L in sizes] for k in table]),
             rank_1e6=np.array([[table[k]["per_size"][L]["rank"]["1e-06"] for L in sizes] for k in table]),
             overlap_cheb3_bond=np.array([[table[k]["per_size"][L]["overlap"]["cheb3/bond"] for L in sizes] for k in table]),
             scale_cheb3_bond=np.array([[table[k]["per_size"][L]["scale"]["cheb3/bond"] for L in sizes] for k in table]),
             theta_cheb3_bond=np.array([[table[k]["per_size"][L]["theta_over_piR2"]["cheb3/bond"] for L in sizes] for k in table]),
             universal_residual=np.array([table[k]["universal"]["residual"] for k in table]),
             universal_best=np.array([table[k]["universal"]["best"] for k in table]),
             universal_keys=np.array(list(table["wave"]["universal"]["keys"])),
             universal_all=np.array([[table[k]["universal"]["keys"][q]["r"] for q in table["wave"]["universal"]["keys"]] for k in table]),
             **{f"weight_{name}_L{L}_{q}": v for (name, L), w in weights.items() for q, v in w.items()
                if name in ("wave", "lifshitz", "mass_0.2")},
             fit_modes=np.array(list(fits)),
             fit_start=np.array([[f["start"] for f in fits[m]] for m in fits]),
             fit_landing=np.array([[[f["c1"], f["c2"], f["m"]] for f in fits[m]] for m in fits]),
             fit_residual=np.array([[f["residual"] for f in fits[m]] for m in fits]),
             fit_invariants=np.array([[[f["m_over_v"], f["c2_over_c1"]] for f in fits[m]] for m in fits]),
             **{f"scan_{ax}_labels": np.array(list(v)) for ax, v in scans.items()},
             **{f"scan_{ax}_{q}": np.array([w[q] for w in v.values()]) for ax, v in scans.items() for q in ("free", "universal")},
             audit_rel_err=np.array([a["rel_err"] for a in audits]), audit_dV_err=np.array([a["dV_err"] for a in audits]),
             ladder_modes=np.array(ns),
             **{f"ladder_{nm}_{key}": np.array([ladder[nm]["r"][key].get(int(k), np.nan) for k in ns])
                for nm in ("wave", "lifshitz") for key in ladder[nm]["r"]},
             **{f"ladder_{nm}_conditions": np.array([ladder[nm]["n_conditions"][t] for t in ("0.01", "0.001", "0.0001", "1e-06")])
                for nm in ("wave", "lifshitz")},
             N=N, __build__=cfg["sha"])
    return out


# ---------------------------------------------------------------------------


def main(argv=None):
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke configuration (~25 s)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: data/; a fresh temporary directory for --quick, "
                         "so a smoke run never overwrites the archive)")
    args = ap.parse_args(argv)
    quick = args.quick
    if args.out:
        DATA = Path(args.out)
    elif quick:
        DATA = Path(tempfile.mkdtemp(prefix="toy-jacobson-quick-"))
    DATA.mkdir(exist_ok=True, parents=True)
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cfg = {
        "quick": quick, "sha": sha,
        "N_A": 1200 if quick else 2400,
        "sizes_A": (4, 6, 8, 12, 16, 24) if quick else (4, 6, 8, 12, 16, 24, 32, 48),
        "sizes_even": (8, 16) if quick else (8, 16, 32),
        "profile_sizes": (16,) if quick else (16, 32),
        "vstar_sizes": (8, 16) if quick else (4, 8, 16, 32, 64),
        "N_B": 800, "sizes_B": (4, 8) if quick else (4, 8, 16),
        "N_C": 800 if quick else 1600, "sizes_C": (8, 16) if quick else (8, 16, 32),
        "N_D": 512, "sizes_D": (8, 16),
        "starts": [(2.0, -0.3, 0.2), (1.5, 0.1, 0.05)] if quick
        else [(0.5, 0.25, 0.4), (2.0, -0.3, 0.2), (0.3, 0.4, 0.8), (1.5, 0.1, 0.05)],
        "maxfev": 150 if quick else 300, "restarts": 1 if quick else 2,
        "m_grid": [0.0, 1.0, 4.0, 16.0] if quick else [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 16.0, 32.0],
        "sizes_gap": (4, 8, 16),
        "N_E": 1200 if quick else 2400,
        "sizes_E": (8, 16) if quick else (8, 16, 32),
        "ansatz_variants": [("site", 0.5), ("bond", 0.5)] if quick
        else [("site", 0.5), ("bond", 0.5), ("mix", 0.5), ("mix", 0.1)],
        "c2_grid": (-0.5, 0.0, 0.5) if quick else (-1.0, -0.5, -0.15, 0.0, 0.2, 0.5),
        "c3_grid": (-0.1, 0.0, 0.1) if quick else (-0.2, -0.1, -0.03, 0.0, 0.03, 0.1, 0.2),
        "N_F": 800 if quick else 1600,
        "sizes_F": (8, 16) if quick else (8, 16, 32),
        "N_G": 800 if quick else 1600,
        "sizes_G": (8, 16) if quick else (8, 16, 32),
        "starts_G": [(1.5, 0.1, 0.05)] if quick else [(2.0, -0.3, 0.2), (1.5, 0.1, 0.05)],
        "maxfev_G": 60 if quick else 300, "restarts_G": 0 if quick else 2,
        "m_scan_G": [0.0, 0.05, 0.2] if quick else [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        "c2_scan_G": [-1 / 12, 0.0, 0.1] if quick else [-0.4, -0.2, -1 / 12, -0.04, 0.0, 0.05, 0.1, 0.2],
        "ladder_modes": (3, 4, 8, 12) if quick else (3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 16, 20),
    }
    t0 = time.time()
    A = section_A(cfg)
    B = section_B(cfg)
    C = section_C(cfg)
    D = section_D(cfg)
    E = section_E(cfg)
    F = section_F(cfg)
    G = section_G(cfg)
    wall = time.time() - t0
    summary = {
        "mode": "quick" if quick else "full",
        "anchor1": {
            "N": A["N"], "sizes": A["sizes"], "per_size_r": A["per_size_r"],
            "slope_L_ge_8": A["slope_L_ge_8"], "per_ratio_slopes": {m: v["slope"] for m, v in A["per_ratio"].items()},
            "r_inf_fit_inverse_L2": A["r_inf_fit_inverse_L2"], "A_fit_inverse_L2": A["A_fit_inverse_L2"],
            "r_inf_fit_power": A["r_inf_fit_power"],
            "even_sizes": A["even_sizes"], "even_per_size_r": A["even_per_size_r"],
            "odd_top_mode_share_max": A["odd_top_mode_share_max"],
            "vstar": A["vstar"],
            "profiles": {L: {"locality": p["locality"],
                             "pp_rowsum_over_beta_centre": float(p["pp_rowsum_over_beta"][L // 2]),
                             "pp_rowsum_over_beta_edge": float(p["pp_rowsum_over_beta"][0]),
                             "xx_resummed_over_beta_centre": float(p["xx_resummed_over_beta"][L // 2]),
                             "xx_rowsum_centre": float(p["xx_rowsum"][L // 2]),
                             "nu_top": float(p["nu"][0])} for L, p in A["profiles"].items()},
        },
        "anchor2": {k: v for k, v in B.items() if k != "rows"},
        "anchor3_map": {
            "fit_A": C["fit_A"], "fit_C": C["fit_C"], "fit_rel_scatter": C["fit_rel_scatter"], "n_fit": C["n_fit"],
            "ansatz_floor_improved_chain": C["ansatz_floor"], "tracking": C["tracking"],
            "per_dynamics": {rec["name"]: {} for rec in C["records"]},
        },
        "anchor4": {k: v for k, v in D.items() if k not in ("scans",)},
        "anchor5_ansatz_floor_and_k4": {k: v for k, v in E.items() if k not in ("grid_c2", "grid_c3")},
        "anchor6_position_and_zero_mode": F,
        "anchor7_free_weight": {k: v for k, v in G.items() if k != "weights"},
        "wall_clock_s": wall,
    }
    for rec in C["records"]:
        summary["anchor3_map"]["per_dynamics"][rec["name"]][f"L{rec['L']}"] = {
            "r": rec["r"], "dep_v1": rec["dep_v1"], "dep_shape": rec["dep_shape"],
            "locality_bw1_L16": rec["locality_bw1_L16"]}
    with open(DATA / "summary.json", "w") as fh:
        json.dump(jsonable(summary), fh, indent=1)
    with open(DATA / "build_info.json", "w") as fh:
        json.dump(build_info("quick" if quick else "full", wall), fh, indent=1)
    log(f"done in {wall:.0f} s; data in {DATA}")


if __name__ == "__main__":
    main()
