#!/usr/bin/env python
"""Multi-interval modular Hamiltonians on free-fermion chains (Layer 4, instrument 1).

Re-runnable end to end, headless, no hidden state:

    .venv/bin/python papers/multi-interval-modular-hamiltonians/notebook.py          # full, a few minutes
    .venv/bin/python papers/multi-interval-modular-hamiltonians/notebook.py --quick  # smoke, ~1 min

THE CLAIM (one paragraph)
=========================
On the half-filled hopping chain, the exact entanglement Hamiltonian
h = ln((1 - C_A)/C_A) of unions of intervals, computed in arbitrary precision
from closed-form correlation matrices, reproduces the Casini-Huerta (2009)
continuum modular Hamiltonian - the local Bisognano-Wichmann weight
beta(x) = 2 pi/z'(x) AND the nonlocal coupling 2 pi/((x - x_c) z'(x_c)) between
conjugate points - after the lattice dictionary stated in
vacuum/modular/casini_huerta.py, with deviations that fall as 1/L^2; the
modular flow moves a right-moving packet along the CH trajectories and
"teleports" the weight sin^2 theta(tau) of CH Eq. (59) into the other interval.
Those are CONSISTENCY CHECKS (CH 2009; Eisler-Peschel 2017; Eisler-Tonni-Peschel
2022, whose Eqs. (31)-(32) coincide with the dictionary derived here). What is
pushed beyond the published closed forms - the discovery targets - is measured
by this file and written to data/: (i) three and four intervals, equal and
unequal, at criticality: every pair of intervals is coupled through its CH
conjugate points, the lattice alternating row sums follow the n-interval CH
weight 2 pi K(x - x_l)/z'(x_l), and the coupling of an outer pair across an
intermediate interval is quantified against the same pair with the middle
interval removed (screening ratio); the inter-interval coupling norm decays as
a power of the separation with a fitted exponent; (ii) gapped (staggered-mass)
chains: the BW profile deforms from the parabola into a triangle with a
renormalized edge slope s(m) 2 pi, linear over ~xi, and the nonlocal couplings
between intervals decay exponentially in the separation with a fitted length
xi_h compared with the correlation length xi = 1/asinh(m). None of the numbers
is restated in this docstring, so the file's sha256 (recorded in every archive)
identifies the code that produced them.

WHAT GOES THROUGH WHAT
======================
Every modular Hamiltonian goes through vacuum.modular.interval_modular at a
precision raised until the entanglement spectrum is resolved with 15 digits to
spare (vacuum.modular.scan.modular_hamiltonian_auto; a ModularPrecisionWarning
would be raised otherwise - none is). Correlation matrices are the closed
forms of vacuum.modular.lattice (infinite chain, staggered mass). Continuum
predictions are vacuum.modular.casini_huerta. Conventions per docs/API.md and
dictionary D1-D6 of casini_huerta.py: hopping 1/2, k_F = pi/2, v_F = 1, site j at
x = j + 1/2, entropies in nats.
"""

from __future__ import annotations

import argparse
import tempfile
import hashlib
import json
import math
import platform
import subprocess
import time
import warnings
from pathlib import Path

import mpmath
import numpy as np
import scipy

import vacuum
from vacuum.fermions import block_entropy
from vacuum.modular import (
    ModularPrecisionWarning,
    bilocal_weight,
    ctm_level_spacing,
    ctm_slope,
    ctm_slope_expansion,
    bw_profile,
    ch_beta,
    ch_bilinear,
    ch_conjugate_points,
    ch_mixing_angle,
    ch_mutual_information,
    ch_nonlocal_weight,
    ch_single_interval_flow,
    ch_trajectory,
    chiral_bilinear,
    correlation_length,
    eisler_peschel_eq26,
    eisler_peschel_eq27,
    eisler_peschel_eq54,
    flow_wavefunction,
    infinite_chain_C,
    intervals_from_parts,
    lattice_boost,
    massive_chain_C,
    modular_hamiltonian_auto,
    multi_interval_scan,
    part_centroids,
    part_weights,
    sites_of_config,
    to_float,
    wavepacket,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

warnings.simplefilter("error", ModularPrecisionWarning)  # insufficient precision is a failure here


# --------------------------------------------------------------------------- #
# S0 provenance
# --------------------------------------------------------------------------- #
def s0_provenance():
    gen = ROOT / "notebook.py"
    sha = hashlib.sha256(gen.read_bytes()).hexdigest()
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip() != ""
    except Exception:  # pragma: no cover
        git, dirty = "unavailable", None
    return {
        "_source": "notebook.py::s0_provenance",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mpmath": mpmath.__version__,
        "vacuum_package": str(Path(vacuum.__file__).parent),
        "generator_sha256": sha,
        "git_head": git,
        "git_dirty": dirty,
    }


def _save(name, build, **arrays):
    DATA.mkdir(exist_ok=True)
    np.savez(DATA / name, __build__=json.dumps(build), **arrays)


def _critical(sites, dps):
    return infinite_chain_C(sites, dps=dps)


# --------------------------------------------------------------------------- #
# S1 anchors reproduced (consistency checks)
# --------------------------------------------------------------------------- #
def s1_single_interval(build, quick):
    Ls = (20, 40) if quick else (20, 40, 60, 80)
    rows = []
    profiles = {}
    for L in Ls:
        sites = list(range(L))
        h, info, dps = modular_hamiltonian_auto(_critical, sites, 0.0, "auto")
        x, b_nn = bw_profile(h, mode="nn")
        _, b_res = bw_profile(h, mode="resummed")
        par = 2 * np.pi * x * (L - x) / L
        ep27 = np.array([L * eisler_peschel_eq27(xx / L) for xx in x])
        dev27 = b_nn / (2 * ep27) - 1
        dev_res = b_res / par - 1
        h3 = np.max(np.abs(np.diag(h, 3))) / L
        rows.append(
            [L, dps, info["eps_max"], eisler_peschel_eq54(L), np.max(np.abs(dev27)), dev27[L // 2 - 1],
             np.max(np.abs(dev_res)), dev_res[L // 2 - 1], b_nn[0] / (2 * np.pi) - 1, h3, eisler_peschel_eq26(0.5, 1)]
        )
        if L == 40:
            profiles = {"x40": x, "beta_nn40": b_nn, "beta_res40": b_res, "parabola40": par, "ep27_40": 2 * ep27}
    rows = np.array(rows)
    _save("s1_single_interval_bw.npz", build,
          columns=np.array(["L", "dps", "eps_max", "ep54", "dev27_max", "dev27_centre", "dev_res_max", "dev_res_centre",
                            "edge_bond1_dev_from_2pi", "h3_peak_over_L", "ep26_p1_peak"]),
          rows=rows, **profiles)
    return rows


def _two_interval_config(s):
    L1 = L2 = 10 * s
    d = 5 * s
    A1 = list(range(L1))
    A2 = list(range(L1 + d, L1 + d + L2))
    return A1, A2


def s1_two_intervals(build, quick):
    scales = (1, 2) if quick else (1, 2, 4)
    rows = []
    for s in scales:
        A1, A2 = _two_interval_config(s)
        sites = A1 + A2
        n1, n2 = len(A1), len(A2)
        iv = intervals_from_parts([A1, A2])
        h, info, dps = modular_hamiltonian_auto(_critical, sites, 0.0, "auto")
        # local part
        xb1, b_res1 = bw_profile(h[:n1, :n1], mode="resummed", offset=A1[0])
        _, b_nn1 = bw_profile(h[:n1, :n1], mode="nn", offset=A1[0])
        bch1 = ch_beta(xb1, iv)
        dev_res = np.abs(b_res1 / bch1 - 1)
        dev_nn = b_nn1 / bch1 - 1
        # nonlocal: row sums (ETP Eq. 32) both directions, interior
        rows_idx, cols_idx = range(n1), range(n1, n1 + n2)
        S12 = bilocal_weight(h, sites, rows_idx, cols_idx)
        S21 = bilocal_weight(h, sites, cols_idx, rows_idx)
        x1 = np.array(A1) + 0.5
        x2 = np.array(A2) + 0.5
        W12 = np.array([ch_nonlocal_weight(x, iv, 1) for x in x1])
        W21 = np.array([ch_nonlocal_weight(x, iv, 0) for x in x2])
        in1 = (x1 > iv[0][0] + 0.2 * n1) & (x1 < iv[0][1] - 0.2 * n1)
        in2 = (x2 > iv[1][0] + 0.2 * n2) & (x2 < iv[1][1] - 0.2 * n2)
        dev_row12 = np.max(np.abs(S12 / W12 - 1)[in1])
        dev_row21 = np.max(np.abs(S21 / W21 - 1)[in2])
        # smeared sin^2 envelopes
        g = lambda x: np.sin(np.pi * (x - iv[0][0]) / n1) ** 2
        f = lambda y: np.sin(np.pi * (y - iv[1][0]) / n2) ** 2
        M_lat = chiral_bilinear(h, sites, g, f, rows_idx, cols_idx)
        M_ch = ch_bilinear(iv, 0, 1, g, f)
        ratio_sin2 = abs(M_lat) / abs(M_ch)
        # 4x4 grid of Gaussian envelopes
        sig = n1 / 8
        cx = np.linspace(iv[0][0] + 0.2 * n1, iv[0][1] - 0.2 * n1, 4)
        cy = np.linspace(iv[1][0] + 0.2 * n2, iv[1][1] - 0.2 * n2, 4)
        Ml = np.zeros((4, 4), complex)
        Mc = np.zeros((4, 4), complex)
        for a, x0 in enumerate(cx):
            for b, y0 in enumerate(cy):
                ga = lambda x, x0=x0: np.exp(-((x - x0) ** 2) / (2 * sig ** 2))
                fb = lambda y, y0=y0: np.exp(-((y - y0) ** 2) / (2 * sig ** 2))
                Ml[a, b] = chiral_bilinear(h, sites, ga, fb, rows_idx, cols_idx)
                Mc[a, b] = ch_bilinear(iv, 0, 1, ga, fb)
        grid_dev = np.linalg.norm(Ml - Mc) / np.linalg.norm(Mc)
        # mutual information vs CH cross ratio
        Cf = to_float(_critical(sites, None))
        MI = block_entropy(Cf, range(n1)) + block_entropy(Cf, range(n1, n1 + n2)) - block_entropy(Cf, range(n1 + n2))
        MIch = ch_mutual_information(iv[0], iv[1])
        rows.append([s, n1 + n2, dps, info["eps_max"], np.max(dev_res), dev_res[n1 // 2 - 1], dev_nn[n1 // 2 - 1],
                     dev_row12, dev_row21, ratio_sin2 - 1, grid_dev, MI, MIch, MI - MIch])
        if s == 2:
            two = {"x_bond": xb1, "beta_res": b_res1, "beta_nn": b_nn1, "beta_ch": bch1, "x1": x1, "S12": S12, "W12": W12,
                   "x2": x2, "S21": S21, "W21": W21, "grid_lat": np.abs(Ml), "grid_ch": np.abs(Mc)}
    rows = np.array(rows)
    _save("s1_two_interval_ch.npz", build,
          columns=np.array(["scale", "n_sites", "dps", "eps_max", "dev_res_max", "dev_res_centre", "dev_nn_centre",
                            "dev_rowsum_12_interior", "dev_rowsum_21_interior", "sin2_ratio_minus_1", "grid_frobenius_dev",
                            "MI_lattice", "MI_ch", "MI_diff"]),
          rows=rows, **two)
    return rows


def s1_modular_flow(build, quick):
    # two intervals, scale 2: teleportation vs CH Eq. (59)
    A1, A2 = _two_interval_config(2)
    sites = A1 + A2
    n1 = len(A1)
    iv = intervals_from_parts([A1, A2])
    h, info, dps = modular_hamiltonian_auto(_critical, sites, 0.0, "auto")
    x0, sig = iv[0][0] + 0.5 * n1, 2.0
    psi0 = wavepacket(sites, x0, sig, chirality=+1)
    taus = np.linspace(-0.3, 0.5, 9) if quick else np.linspace(-0.3, 0.5, 33)
    tele = []
    for tau in taus:
        psi = flow_wavefunction(h, psi0, tau)
        P = part_weights(psi, [range(n1), range(n1, len(sites))])
        c = part_centroids(psi, sites, [range(n1), range(n1, len(sites))])
        th = ch_mixing_angle(x0, tau, iv[0], iv[1])
        tele.append([tau, P[1], math.sin(th) ** 2, c[0], ch_trajectory(x0, tau, iv, 0), c[1], ch_trajectory(x0, tau, iv, 1)])
    tele = np.array(tele)
    # single interval L = 40: trajectory vs CH Eq. (53)
    L = 40
    sites1 = list(range(L))
    h1, _, _ = modular_hamiltonian_auto(_critical, sites1, 0.0, "auto")
    x0s = 12.0
    psi0 = wavepacket(sites1, x0s, 2.0, chirality=+1)
    single = []
    for tau in (np.linspace(-0.15, 0.3, 7) if quick else np.linspace(-0.15, 0.3, 19)):
        psi = flow_wavefunction(h1, psi0, tau)
        c = part_centroids(psi, sites1, [range(L)])[0]
        single.append([tau, c, ch_single_interval_flow(x0s, tau, 0.0, float(L))])
    single = np.array(single)
    _save("s1_modular_flow.npz", build,
          tele_columns=np.array(["tau", "P2_lattice", "sin2theta_ch59", "centroid1", "ch_x1", "centroid2", "ch_x2"]),
          tele=tele, single_columns=np.array(["tau", "centroid", "ch_eq53"]), single=single)
    return tele, single


# --------------------------------------------------------------------------- #
# S2 three and four intervals at criticality (discovery target)
# --------------------------------------------------------------------------- #
def s2_multi_intervals(build, quick):
    L3, L4 = 12, 10
    gaps3 = (6, 24) if quick else (3, 6, 12, 24, 48)
    gaps4 = (6,) if quick else (3, 6, 12, 24)
    cfgs = []
    for d in gaps3:
        cfgs.append({"label": f"three_equal_d{d}", "intervals": [(0, L3), (L3 + d, L3), (2 * L3 + 2 * d, L3)]})
        cfgs.append({"label": f"two_outer_d{d}", "intervals": [(0, L3), (2 * L3 + 2 * d, L3)]})  # middle removed
    for d in gaps4:
        cfgs.append({"label": f"four_equal_d{d}", "intervals": [(0, L4), (L4 + d, L4), (2 * L4 + 2 * d, L4), (3 * L4 + 3 * d, L4)]})
    cfgs.append({"label": "three_unequal_a", "intervals": [(0, 8), (14, 16), (40, 12)]})
    cfgs.append({"label": "three_unequal_b", "intervals": [(0, 16), (20, 8), (34, 12)]})
    reps = multi_interval_scan(cfgs, include_h=True)
    by = {r["label"]: r for r in reps}
    pair_rows = []
    labels = []
    for r in reps:
        for p in r["nonlocal"]["pairs"]:
            pair_rows.append([r["n_sites"], r["dps"], r["eps_max"], p["p"], p["q"], p["gap"], p["fro"], p["max_abs"], p["sum_abs"],
                              p["dev_interior_pq"], p["dev_interior_qp"], float(np.sum(np.abs(p["ch_weight_pq"]))),
                              float(np.sum(np.abs(p["row_weight_pq"]))), r["locality_nn"], r["nonlocal"]["offblock_fraction"]])
            labels.append(r["label"])
    pair_rows = np.array(pair_rows)
    # screening: outer pair (0,2) in the three-interval configs vs the same pair alone
    screening = []
    for d in gaps3:
        r3 = by[f"three_equal_d{d}"]
        r2 = by[f"two_outer_d{d}"]
        p02 = [p for p in r3["nonlocal"]["pairs"] if (p["p"], p["q"]) == (0, 2)][0]
        p2 = r2["nonlocal"]["pairs"][0]
        screening.append([d, p02["fro"], p2["fro"], p02["fro"] / p2["fro"],
                          float(np.sum(np.abs(p02["row_weight_pq"]))), float(np.sum(np.abs(p2["row_weight_pq"]))),
                          float(np.sum(np.abs(p02["ch_weight_pq"]))), float(np.sum(np.abs(p2["ch_weight_pq"]))),
                          p02["dev_interior_pq"], p2["dev_interior_pq"]])
    screening = np.array(screening)
    # decay of the adjacent-pair coupling with the gap (three-interval configs)
    dec = np.array([[d, [p for p in by[f"three_equal_d{d}"]["nonlocal"]["pairs"] if (p["p"], p["q"]) == (0, 1)][0]["fro"],
                     float(np.sum(np.abs([p for p in by[f"three_equal_d{d}"]["nonlocal"]["pairs"] if (p["p"], p["q"]) == (0, 1)][0]["ch_weight_pq"])))]
                    for d in gaps3])
    fit_lat = np.polyfit(np.log(dec[:, 0]), np.log(dec[:, 1]), 1)[0] if len(gaps3) >= 3 else np.nan
    fit_ch = np.polyfit(np.log(dec[:, 0]), np.log(dec[:, 2]), 1)[0] if len(gaps3) >= 3 else np.nan
    # store one full four-interval h and its CH partners for the draft figure
    r4 = by[f"four_equal_d{gaps4[0]}"]
    _save("s2_three_four_intervals.npz", build,
          pair_columns=np.array(["n_sites", "dps", "eps_max", "p", "q", "gap", "fro", "max_abs", "sum_abs", "dev_interior_pq",
                                 "dev_interior_qp", "ch_weight_sum_pq", "lattice_rowweight_sum_pq", "locality_nn", "offblock_fraction"]),
          pair_rows=pair_rows, pair_labels=np.array(labels),
          screening_columns=np.array(["gap", "fro_02_with_middle", "fro_02_alone", "screening_ratio_fro", "S02_with_middle",
                                      "S02_alone", "CH02_with_middle", "CH02_alone", "dev_02_with_middle", "dev_02_alone"]),
          screening=screening, decay=dec, decay_exponent_lattice=fit_lat, decay_exponent_ch=fit_ch,
          h4=r4["h"], sites4=np.array(r4["sites"]),
          h3_partners=np.array([p["ch_partner_pq"] for p in by[f"three_equal_d{gaps3[0]}"]["nonlocal"]["pairs"]], dtype=object))
    return pair_rows, labels, screening, dec, fit_lat, fit_ch


# --------------------------------------------------------------------------- #
# S3 off-critical chains (discovery target)
# --------------------------------------------------------------------------- #
def s3_offcritical_single(build, quick):
    L = 40
    masses = (0.1, 0.5) if quick else (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    rows = []
    prof = {}
    for m in masses:
        sites = list(range(L))
        h, info, dps = modular_hamiltonian_auto(lambda s, d, m=m: __import__("vacuum.modular", fromlist=["massive_chain_C"]).massive_chain_C(s, m, dps=d), sites, m, "auto")
        x, b_nn = bw_profile(h, mode="nn")
        _, b_res = bw_profile(h, mode="resummed")
        slope = b_nn[0] / (2 * np.pi)  # beta(1)/(2 pi 1): edge slope in units of the BW slope
        slope3 = np.polyfit(x[:3], b_nn[:3], 1)[0] / (2 * np.pi)
        lin = np.abs(b_nn / (2 * np.pi * slope * x) - 1) < 0.02
        extent = int(np.argmin(lin)) if not lin.all() else len(lin)  # first bond leaving the linear law
        plateau = float(np.max(b_nn))
        xi = correlation_length(m)
        rows.append([m, xi, dps, info["eps_max"], slope, slope3, extent, plateau, plateau / (2 * np.pi * L / 4),
                     float(np.max(np.abs(np.diag(h)))), float(np.max(np.abs(np.diag(h, 2)))), float(np.max(np.abs(np.diag(h, 3))))])
        prof[f"beta_nn_m{m}"] = b_nn
        prof[f"beta_res_m{m}"] = b_res
    rows = np.array(rows)
    _save("s3_offcritical_single.npz", build, x=np.arange(1, L, dtype=float),
          columns=np.array(["m", "xi", "dps", "eps_max", "edge_slope_over_2pi", "edge_slope3_over_2pi", "linear_extent_bonds",
                            "plateau", "plateau_over_2piL4", "diag_max", "h2_max", "h3_max"]),
          rows=rows, **prof)
    return rows


def s3_offcritical_two(build, quick):
    L = 12
    masses = (0.2,) if quick else (0.05, 0.1, 0.2, 0.5)
    gaps = (2, 8, 32) if quick else (2, 4, 8, 16, 32)
    cfgs = [{"label": f"m{m}_d{d}", "intervals": [(0, L), (L + d, L)], "mass": m} for m in masses for d in gaps]
    cfgs += [{"label": f"m0_d{d}", "intervals": [(0, L), (L + d, L)], "mass": 0.0} for d in gaps]
    reps = multi_interval_scan(cfgs)
    rows = []
    for r in reps:
        p = r["nonlocal"]["pairs"][0]
        rows.append([r["mass"], r["xi"], p["gap"], p["fro"], p["max_abs"], float(np.max(np.abs(p["row_weight_pq"]))),
                     r["nonlocal"]["offblock_fraction"], r["dps"], r["eps_max"]])
    rows = np.array(rows)
    fits = []
    for m in masses:
        sel = (rows[:, 0] == m) & (rows[:, 2] >= 4)
        if np.sum(sel) >= 2:
            slope = np.polyfit(rows[sel, 2], np.log(rows[sel, 3]), 1)[0]
            xi_h = -1.0 / slope
            fits.append([m, correlation_length(m), xi_h, xi_h / correlation_length(m)])
    fits = np.array(fits) if fits else np.zeros((0, 4))
    sel0 = rows[:, 0] == 0.0
    crit_exp = np.polyfit(np.log(rows[sel0, 2]), np.log(rows[sel0, 3]), 1)[0] if np.sum(sel0) >= 3 else np.nan
    # three intervals, gapped
    cfg3 = [{"label": "three_m0.2", "intervals": [(0, 8), (12, 8), (24, 8)], "mass": 0.2}]
    r3 = multi_interval_scan(cfg3)[0]
    three = np.array([[p["p"], p["q"], p["gap"], p["fro"], p["max_abs"]] for p in r3["nonlocal"]["pairs"]])
    _save("s3_offcritical_two.npz", build,
          columns=np.array(["m", "xi", "gap", "fro", "max_abs", "max_rowweight", "offblock_fraction", "dps", "eps_max"]),
          rows=rows, fit_columns=np.array(["m", "xi", "xi_h_from_fro", "xi_h_over_xi"]), fits=fits,
          critical_decay_exponent=crit_exp, three_columns=np.array(["p", "q", "gap", "fro", "max_abs"]), three=three)
    return rows, fits, crit_exp, three



# --------------------------------------------------------------------------- #
# S4 n >= 3 intervals at larger L: the 1/L^2 trend and its extrapolation
# --------------------------------------------------------------------------- #
def s4_multi_large_L(build, quick):
    """Deviation from the n-interval CH formula at fixed gap/L, for growing L.

    Two observables per interval pair: the maximum interior deviation of the
    alternating row sums from CH Eq. (48) (row by row) and the deviation of
    their SUM over the interval (the total bilocal weight, which is what the
    screening and decay results use). Both are fitted against 1/L^2 and
    extrapolated to L -> infinity.
    """
    families = (
        ("three_gap_L", 3, 1.0, (12, 16, 24) if quick else (12, 16, 24, 32)),
        ("three_gap_L_over_2", 3, 0.5, (16, 24) if quick else (16, 24, 32)),
        ("four_gap_L", 4, 1.0, (10, 16) if quick else (10, 16, 24)),
    )
    rows, fits = [], []
    for label, n, frac, Ls in families:
        for L in Ls:
            d = int(round(frac * L))
            iv = [(k * (L + d), L) for k in range(n)]
            rep = multi_interval_scan([{"label": f"{label}_L{L}", "intervals": iv}])[0]
            dev_row = max(max(p["dev_interior_pq"], p["dev_interior_qp"]) for p in rep["nonlocal"]["pairs"])
            dev_sum = max(abs(float(np.sum(np.abs(p["row_weight_pq"]))) / float(np.sum(np.abs(p["ch_weight_pq"]))) - 1.0)
                          for p in rep["nonlocal"]["pairs"])
            dev_loc = max(float(np.max(np.abs(b["dev_resummed"][2:-2]))) for b in rep["bw"])
            rows.append([n, frac, L, d, rep["n_sites"], rep["dps"], rep["eps_max"], dev_row, dev_sum, dev_loc])
        sel = np.array([r for r in rows if r[0] == n and r[1] == frac], dtype=float)
        Lv = sel[:, 2]
        for col, name in ((8, "summed_weight"), (7, "rowsum_max")):
            A = np.vstack([np.ones(Lv.size), Lv ** -2.0]).T
            c, *_ = np.linalg.lstsq(A, sel[:, col], rcond=None)
            fits.append([n, frac, {"summed_weight": 0.0, "rowsum_max": 1.0}[name], float(c[0]), float(c[1]),
                         float(np.max(np.abs(A @ c - sel[:, col])))])
    rows = np.array(rows)
    fits = np.array(fits)
    _save("s4_multi_large_L.npz", build,
          columns=np.array(["n_intervals", "gap_over_L", "L", "gap", "n_sites", "dps", "eps_max",
                            "dev_rowsum_interior_max", "dev_summed_weight_max", "dev_local_resummed_max"]),
          rows=rows,
          fit_columns=np.array(["n_intervals", "gap_over_L", "observable(0=summed,1=rowsum)",
                                "extrapolated_L_inf", "amplitude_A_in_A_over_L2", "max_abs_fit_residual"]),
          fits=fits)
    return rows, fits


# --------------------------------------------------------------------------- #
# S5 xi_h/xi asymptotics: gapped chains at xi = 20 and 40, gaps out to 16 xi
# --------------------------------------------------------------------------- #
def s5_xi_h_asymptotics(build, quick):
    """Is xi_h/xi a constant, and which?

    The inter-interval couplings decay as W(d) ~ d^{-p} e^{-d/xi_h}. Fitting a
    single exponential over a window of gaps returns an *effective* length that
    still carries the algebraic prefactor; the asymptotic statement needs the
    LOCAL decay rate at the largest gaps,
        xi_h^loc(d) = (d_{k+1} - d_k) / ln(W(d_k)/W(d_{k+1})),
    which is quoted here at d ~ 10-16 xi for xi = 5, 10, 20, 40. The observable
    is the alternating row-sum weight of the lattice dictionary (D2), the
    quantity that converges to the CH bilocal weight at criticality; the
    Frobenius norm of the whole off-diagonal block is reported alongside.
    """
    plan = ((0.1, 12, (16, 32, 64, 128)),) if quick else (
        (0.2, 12, (8, 16, 32, 48, 64, 80)),
        (0.1, 12, (8, 16, 32, 64, 96, 128, 160)),
        (0.1, 24, (16, 32, 64, 96, 128, 160)),
        (0.05, 12, (20, 40, 80, 160, 320)),
        (0.05, 24, (20, 40, 80, 160, 240, 320)),
        (0.025, 48, (40, 80, 160, 240, 320, 480)),
    )
    rows, locs = [], []
    for m, L, gaps in plan:
        xi = correlation_length(m)
        reps = multi_interval_scan([{"label": f"m{m}_L{L}_d{d}", "intervals": [(0, L), (L + d, L)], "mass": m}
                                    for d in gaps])
        W = np.array([float(np.max(np.abs(r["nonlocal"]["pairs"][0]["row_weight_pq"]))) for r in reps])
        F = np.array([r["nonlocal"]["pairs"][0]["fro"] for r in reps])
        d = np.array(gaps, dtype=float)
        for k in range(d.size):
            rows.append([m, xi, L, d[k], d[k] / xi, W[k], F[k], reps[k]["dps"], reps[k]["eps_max"]])
        for name, Y in ((0.0, W), (1.0, F)):
            loc = (d[1:] - d[:-1]) / np.log(Y[:-1] / Y[1:])
            for k in range(loc.size):
                locs.append([m, xi, L, name, 0.5 * (d[k] + d[k + 1]) / xi, loc[k] / xi])
    rows = np.array(rows)
    locs = np.array(locs)
    tail = locs[(locs[:, 3] == 0.0) & (locs[:, 4] >= 8.0), 5]
    rows_out = {"asymptotic_xi_h_over_xi_mean": float(np.mean(tail)) if tail.size else float("nan"),
                "asymptotic_xi_h_over_xi_sd": float(np.std(tail)) if tail.size else float("nan"),
                "n_tail": int(tail.size)}
    _save("s5_xi_h_asymptotics.npz", build,
          columns=np.array(["m", "xi", "L", "gap", "gap_over_xi", "rowweight_max", "fro", "dps", "eps_max"]),
          rows=rows,
          loc_columns=np.array(["m", "xi", "L", "observable(0=rowweight,1=fro)", "gap_mid_over_xi", "xi_h_local_over_xi"]),
          local=locs, **{k: v for k, v in rows_out.items()})
    return rows, locs, rows_out


# --------------------------------------------------------------------------- #
# S6 the gapped BW slope s(m) in closed form (corner transfer matrices)
# --------------------------------------------------------------------------- #
def s6_ctm_slope(build, quick):
    """s(m) = (2/pi) k I(k'), k = 1/sqrt(1+m^2): the CTM closed form vs the fit.

    For each mass, the entanglement Hamiltonian of an L-site block of the
    staggered-mass chain is compared with the half-infinite CTM operator
    2 pi s(m) sum_x x h_x (vacuum.modular.lattice_boost): the edge slope taken
    from the first nearest-neighbour bond, the edge slope taken from the first
    on-site term, the whole first-min(L/2, 6 xi)-site block, and the lowest
    single-particle level against eps = pi I(k')/I(k) with its (2l+1) ladder.
    L is chosen at fixed L/xi where possible so that the two edges do not talk.
    """
    cases = ((1.0, 40), (0.2, 40)) if quick else ((2.0, 30), (1.0, 40), (0.5, 48), (0.3, 60), (0.2, 60), (0.1, 40), (0.05, 40), (0.02, 40))
    rows = []
    prof = {}
    for m, L in cases:
        sites = list(range(L))
        h, info, dps = modular_hamiltonian_auto(lambda s_, d_, mm=m: massive_chain_C(s_, mm, dps=d_), sites, m, "auto")
        xi = correlation_length(m)
        s_ctm = ctm_slope(m)
        eps_ctm = ctm_level_spacing(m)
        s_bond = -2.0 * h[0, 1] / (2.0 * math.pi)          # beta_nn(x=1)/(2 pi x)
        s_site = h[0, 0] / m / (2.0 * math.pi * 0.5)       # on-site term at x = 1/2
        ev = np.sort(np.abs(np.linalg.eigvalsh(h)))
        nb = max(2, min(L // 2, int(round(6 * xi))))
        hb = lattice_boost(L, m)
        blk = float(np.max(np.abs(h[:nb, :nb] - hb[:nb, :nb])) / np.max(np.abs(hb[:nb, :nb])))
        rows.append([m, xi, L, dps, s_ctm, s_bond, s_site, ctm_slope_expansion(xi), eps_ctm, ev[0],
                     ev[2] / ev[0], ev[4] / ev[0], nb, blk, 1.0 - 1.0 / L])
        prof[f"beta_nn_m{m}"] = -2.0 * np.diag(h, 1)
    rows = np.array(rows)
    _save("s6_ctm_slope.npz", build,
          columns=np.array(["m", "xi", "L", "dps", "s_ctm", "s_edge_bond", "s_edge_onsite", "s_scaling_expansion",
                            "eps_ctm", "eps_0_lattice", "level_ratio_3", "level_ratio_5", "n_block", "block_rel_dev",
                            "critical_edge_slope_1_minus_1_over_L"]),
          rows=rows, **prof)
    return rows


# --------------------------------------------------------------------------- #
def main(argv=None):
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke configuration (~15 s)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: data/; a fresh temporary directory for --quick, "
                         "so a smoke run never overwrites the archive)")
    args = ap.parse_args(argv)
    if args.out:
        DATA = Path(args.out)
    elif args.quick:
        DATA = Path(tempfile.mkdtemp(prefix="multi-interval-modular-quick-"))
    t0 = time.time()
    build = s0_provenance()
    build["quick"] = bool(args.quick)
    DATA.mkdir(exist_ok=True, parents=True)
    (DATA / "build_info.json").write_text(json.dumps(build, indent=2))

    s1a = s1_single_interval(build, args.quick)
    s1b = s1_two_intervals(build, args.quick)
    tele, single = s1_modular_flow(build, args.quick)
    pair_rows, labels, screening, dec, fit_lat, fit_ch = s2_multi_intervals(build, args.quick)
    s3a = s3_offcritical_single(build, args.quick)
    s3b, fits, crit_exp, three = s3_offcritical_two(build, args.quick)
    s4rows, s4fits = s4_multi_large_L(build, args.quick)
    s5rows, s5loc, s5sum = s5_xi_h_asymptotics(build, args.quick)
    s6rows = s6_ctm_slope(build, args.quick)

    tele_in = (np.abs(tele[:, 0]) <= 0.3) & (np.abs(tele[:, 0]) >= 0.05)  # exclude tau ~ 0 (0/0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tele_rel = np.abs(tele[tele_in, 1] / tele[tele_in, 2] - 1)
    tele_rel = tele_rel[np.isfinite(tele_rel)]

    def _cfg_gap(label):
        return int(label.split("_d")[-1]) if "_d" in label else None

    def _max_dev(pred):
        vals = [max(row[9], row[10]) for row, lab in zip(pair_rows, labels)
                if (lab.startswith("three_equal") or lab.startswith("four_equal")) and pred(_cfg_gap(lab))]
        return float(np.nanmax(vals)) if vals else None

    S02_ratio = (screening[:, 4] / screening[:, 5]).tolist()
    lat_weight_exp = float(np.polyfit(np.log(dec[:, 0]), np.log(np.array(
        [[row[12] for row, lab in zip(pair_rows, labels) if lab == f"three_equal_d{int(d)}" and (row[3], row[4]) == (0, 1)][0]
         for d in dec[:, 0]])), 1)[0]) if len(dec) >= 3 else float("nan")
    summary = {
        "_source": "notebook.py::main",
        "wall_clock_s": None,
        "anchor_1_single_interval": {
            "L": s1a[:, 0].tolist(),
            "nn_vs_EP27_max_rel_dev": s1a[:, 4].tolist(),
            "resummed_vs_parabola_max_rel_dev": s1a[:, 6].tolist(),
            "eps_max_minus_EP54": (s1a[:, 2] - s1a[:, 3]).tolist(),
            "edge_bond1_beta_over_2pi_minus_1": s1a[:, 8].tolist(),
            "h3_peak_over_L_vs_EP26": (s1a[:, 9] / s1a[:, 10] - 1).tolist(),
        },
        "anchor_2_two_intervals_vs_casini_huerta": {
            "scale": s1b[:, 0].tolist(),
            "local_resummed_max_rel_dev": s1b[:, 4].tolist(),
            "local_nn_centre_lattice_correction": s1b[:, 6].tolist(),
            "nonlocal_rowsum_interior_max_rel_dev_12": s1b[:, 7].tolist(),
            "nonlocal_rowsum_interior_max_rel_dev_21": s1b[:, 8].tolist(),
            "nonlocal_sin2_bilinear_ratio_minus_1": s1b[:, 9].tolist(),
            "nonlocal_grid_frobenius_rel_dev": s1b[:, 10].tolist(),
            "mutual_information_lattice_minus_ch": s1b[:, 13].tolist(),
            "teleportation_P2_vs_sin2theta_max_rel_dev_|tau|<=0.3": float(np.max(tele_rel)) if tele_rel.size else None,
            "teleportation_centroid1_max_abs_dev_|tau|<=0.3": float(np.max(np.abs(tele[tele_in, 3] - tele[tele_in, 4]))),
            "single_interval_flow_centroid_max_abs_dev": float(np.max(np.abs(single[:, 1] - single[:, 2]))),
        },
        "discovery_three_four_intervals": {
            "configs": sorted(set(labels)),
            "max_interior_rowsum_dev_all_pairs_configs_d>=12": _max_dev(lambda d: d is not None and d >= 12),
            "max_interior_rowsum_dev_all_pairs_configs_d=6": _max_dev(lambda d: d == 6),
            "max_interior_rowsum_dev_all_pairs_configs_d=3": _max_dev(lambda d: d == 3),
            "unequal_configs_max_interior_rowsum_dev": float(np.nanmax(
                [max(row[9], row[10]) for row, lab in zip(pair_rows, labels) if lab.startswith("three_unequal")])),
            "screening_gap": screening[:, 0].tolist(),
            "screening_ratio_summed_rowweight_02_with_middle_over_alone_lattice": S02_ratio,
            "screening_ratio_summed_weight_02_with_middle_over_alone_CH": (screening[:, 6] / screening[:, 7]).tolist(),
            "screening_ratio_fro_02_with_middle_over_alone": screening[:, 3].tolist(),
            "adjacent_pair_fro_decay_exponent_lattice": float(fit_lat),
            "adjacent_pair_summed_rowweight_decay_exponent_lattice": lat_weight_exp,
            "adjacent_pair_CH_weight_decay_exponent": float(fit_ch),
            "offblock_fraction_range": [float(np.min(pair_rows[:, 14])), float(np.max(pair_rows[:, 14]))],
        },
        "discovery_n_ge_3_large_L": {
            "families": ["three_gap_L", "three_gap_L_over_2", "four_gap_L"],
            "L": s4rows[:, 2].tolist(),
            "n_intervals": s4rows[:, 0].tolist(),
            "gap_over_L": s4rows[:, 1].tolist(),
            "dev_rowsum_interior_max": s4rows[:, 7].tolist(),
            "dev_summed_weight_max": s4rows[:, 8].tolist(),
            "fit_columns": ["n", "gap/L", "observable(0=summed,1=rowsum)", "extrapolated_L_inf", "A in A/L^2", "max_fit_residual"],
            "fits": s4fits.tolist(),
        },
        "discovery_xi_h_asymptotics": {
            "asymptotic_xi_h_over_xi_mean": s5sum["asymptotic_xi_h_over_xi_mean"],
            "asymptotic_xi_h_over_xi_sd": s5sum["asymptotic_xi_h_over_xi_sd"],
            "n_local_rates_beyond_8xi": s5sum["n_tail"],
            "xi_values": sorted(set(np.round(s5rows[:, 1], 3).tolist())),
            "max_gap_over_xi": float(np.max(s5rows[:, 4])),
            "local_rate_table_columns": ["m", "xi", "L", "observable(0=rowweight,1=fro)", "gap_mid/xi", "xi_h_local/xi"],
            "local_rate_table": s5loc.tolist(),
        },
        "discovery_ctm_slope": {
            "columns": ["m", "xi", "L", "dps", "s_ctm", "s_edge_bond", "s_edge_onsite", "s_scaling_expansion",
                        "eps_ctm", "eps_0_lattice", "level_ratio_3", "level_ratio_5", "n_block", "block_rel_dev",
                        "critical_edge_slope_1_minus_1_over_L"],
            "rows": s6rows.tolist(),
            "max_rel_dev_s_onsite_vs_ctm": float(np.max(np.abs(s6rows[:, 6] / s6rows[:, 4] - 1.0))),
            "max_rel_dev_eps0_vs_ctm": float(np.max(np.abs(s6rows[:, 9] / s6rows[:, 8] - 1.0))),
        },
        "discovery_off_critical": {
            "m": s3a[:, 0].tolist(),
            "xi": s3a[:, 1].tolist(),
            "edge_slope_over_2pi": s3a[:, 4].tolist(),
            "linear_extent_bonds": s3a[:, 6].tolist(),
            "plateau_over_critical_parabola_max": s3a[:, 8].tolist(),
            "two_interval_xi_h_over_xi": fits.tolist(),
            "critical_two_interval_fro_decay_exponent": float(crit_exp),
            "three_intervals_m0.2_pair_fro": three.tolist(),
        },
    }
    summary["wall_clock_s"] = time.time() - t0
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wall clock {summary['wall_clock_s']:.1f} s")


if __name__ == "__main__":
    main()
