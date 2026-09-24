"""Layer 5, time-modulated-vacua: the non-commuting multimode bound — verdict step.

Writes ``data/s3_multimode_verdict.json`` (``--quick``: a temporary directory or ``--out DIR``).

Sections
  A. archived battery re-check: the 60 random patterns + 3 CEM-attacked rows of
     ``section_bound_theorem`` (``summary.json['bound_theorem']['multimode_pattern']``) are
     regenerated from the same RNG sequence (seed 7, same call order), their ln rho verified
     against the archive, and every row is scored against the PROVED bound of Theorem M
     (``multimode_bound_proved``) and the refined conjecture next to the conjectured one;
  B. two-mode sum-frequency table: exact vs first-order (Theorem F, Eq. (5)) growth rate;
  C. the attack: CEM + Nelder-Mead over (omega, A, s, durations) at N = 2, 3 maximising the
     conjectured ratio (``multimode_attack``), several d_eff and seeds;
  D. detuned near-degenerate pairs polished from rate-optimum bang-bang starts;
  E. first-order Schur-multiplier search (can ||C o Shat|| exceed (2/pi)||C||?);
  F. the verdict: the largest ratios found against the conjectured, refined and proved bounds.

Usage:
    .venv/bin/python papers/time-modulated-vacua/notebook_multimode.py            # full (data/)
    .venv/bin/python papers/time-modulated-vacua/notebook_multimode.py --quick    # smoke
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
from scipy.optimize import minimize

import vacuum
from vacuum.core.models import harmonic_chain_K
from vacuum.floquet import (
    multimode_attack,
    multimode_bound_proved,
    multimode_check,
    multimode_ratios,
    operator_norm_depth,
    random_controls,
    rate_optimum,
    rate_optimum_control,
    rwa_growth_rate,
    sum_frequency_first_order,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


def _build_info():
    return {
        "generator": str(Path(__file__).resolve().relative_to(HERE.parent.parent)),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_head": _git_head(),
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package": f"vacuum {vacuum.__version__}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ------------------------------------------------------- A: archived battery --


def section_archive(quick=False):
    """Regenerate the archived multimode_pattern rows (seed 7, notebook.py call order) and score
    them against the proved bound.  The coupling rows are drawn first because they consume the
    same RNG stream; only the pattern rows are scored."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(7)
    N_mm = 8
    K0 = harmonic_chain_K(N_mm, 0.5, bc="dirichlet")
    w_max = float(np.max(np.sqrt(np.linalg.eigvalsh(K0))))
    for i in range(24):  # the coupling rows of section_bound_theorem (RNG consumption only)
        fam = ("clipped_fourier", "sign_fourier", "piecewise")[i % 3]
        random_controls(rng, 1, 32, fam)
        rng.uniform(0.5, 3.5)
        rng.uniform(0.1, 0.8)
    rows, patterns = [], []
    for i in range(60):
        P = rng.normal(size=(N_mm, N_mm))
        P = 0.5 * (P + P.T)
        d_eff = rng.uniform(0.1, 0.7)
        d = d_eff / operator_norm_depth(K0, P, 1.0)
        s = random_controls(rng, 1, 24, ("sign_fourier", "clipped_fourier")[i % 2])[0]
        T = rng.uniform(1.0, 5.0)
        c = multimode_check(K0, s, T, d, kind="pattern", pattern=P)
        patterns.append(P)
        rows.append(dict(index=i, depth=d, d_eff=c["d_eff"], period=T, ln_full=c["ln_full"],
                         rate_ratio=c["rate_ratio"], proved_ratio=c["proved_ratio"],
                         refined_ratio=c["refined_ratio"], proved_bound=c["proved_bound"],
                         conjectured_bound=w_max * rate_optimum(c["d_eff"]).lambda_star,
                         norm_C_over_wmax_normA=c["norm_C"] / (w_max * c["d_eff"] / d), attacked=False))
    archive = None
    try:
        with open(DATA / "summary.json") as fh:
            archive = json.load(fh)["bound_theorem"]["multimode_pattern"]
    except Exception:
        archive = None
    match = None
    if archive is not None:
        arch_random = [r for r in archive if not r["attacked"]]
        match = {"n_archived": len(archive), "n_random_archived": len(arch_random),
                 "max_abs_dln_full": max(abs(a["ln_full"] - r["ln_full"]) for a, r in zip(arch_random, rows)),
                 "max_abs_d_eff": max(abs(a["d_eff"] - r["d_eff"]) for a, r in zip(arch_random, rows))}
        # the 3 attacked rows: pattern = one of the 60 (the worst rate ratios), ln_full from the archive
        worst_idx = np.argsort([-r["rate_ratio"] for r in rows])[:3]
        for a, idx in zip([r for r in archive if r["attacked"]], worst_idx):
            P = patterns[int(idx)]
            mb = multimode_bound_proved(K0, P, a["depth"])
            rows.append(dict(index=int(idx), depth=a["depth"], d_eff=a["d_eff"], period=a["period"],
                             ln_full=a["ln_full"], rate_ratio=a["rate_ratio"],
                             proved_ratio=a["ln_full"] / (a["period"] * mb["rate_bound"]),
                             refined_ratio=a["ln_full"] / (a["period"] * mb["refined_conjecture"]),
                             proved_bound=mb["rate_bound"], conjectured_bound=mb["conjectured"],
                             norm_C_over_wmax_normA=mb["norm_C"] / (mb["omega_max"] * mb["norm_A"]),
                             attacked=True, archived_d_eff_match=abs(mb["d_eff"] - a["d_eff"])))
    unstable = [r for r in rows if r["ln_full"] > 0]
    return {"rows": rows, "archive_match": match, "n_rows": len(rows), "n_unstable": len(unstable),
            "max_rate_ratio": max(r["rate_ratio"] for r in rows),
            "max_proved_ratio": max(r["proved_ratio"] for r in rows),
            "max_refined_ratio": max(r["refined_ratio"] for r in rows),
            "proved_below_conjectured_count": sum(r["proved_bound"] < r["conjectured_bound"] for r in rows),
            "proved_over_conjectured_min": min(r["proved_bound"] / r["conjectured_bound"] for r in rows),
            "proved_over_conjectured_max": max(r["proved_bound"] / r["conjectured_bound"] for r in rows),
            "omega_max": w_max, "seconds": time.perf_counter() - t0}


# ------------------------------------------------------ B: sum frequency --


def section_sum_frequency(quick=False):
    """Two modes (omega1, 1), pure coupling A = [[0,1],[1,0]], square wave at omega1 + 1: exact ln rho
    vs the first-order rate (d_eff/pi) sqrt(omega1); ratios to the conjectured bound."""
    t0 = time.perf_counter()
    rows = []
    depths = (0.02, 0.2) if quick else (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
    w1s = (0.3, 0.9) if quick else (0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99)
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    for d in depths:
        for w1 in w1s:
            om = np.array([w1, 1.0])
            T = 2.0 * np.pi / (w1 + 1.0)
            s, taus = np.array([1.0, -1.0]), np.array([T / 2, T / 2])
            r = multimode_ratios(om, A, d, s, taus)
            K0, P = np.diag(om ** 2), A * np.outer(om, om)
            fo = sum_frequency_first_order(w1, 1.0, d)
            rows.append(dict(d_eff=d, omega1=w1, period=T, exact_rate=r["rate"], first_order_rate=fo["rate"],
                             rwa_rate=rwa_growth_rate(K0, P, d, s, taus),
                             exact_over_first_order=r["rate"] / fo["rate"],
                             ratio_conjectured=r["ratio_conjectured"], first_order_ratio=fo["ratio"],
                             ratio_refined=r["ratio_refined"], ratio_proved=r["ratio_proved"]))
    small = [r for r in rows if r["d_eff"] <= 0.05]
    return {"rows": rows, "max_ratio_conjectured": max(r["ratio_conjectured"] for r in rows),
            "max_abs_dev_first_order_small_d": max(abs(r["exact_over_first_order"] - 1.0) for r in small),
            "max_abs_dev_first_order_all": max(abs(r["exact_over_first_order"] - 1.0) for r in rows),
            "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------ C: attack --


def section_attack(quick=False):
    t0 = time.perf_counter()
    rows = []
    configs = [(2, 4, 0.3, 0)] if quick else [(N, M, d, seed) for d in (0.05, 0.3, 0.8) for N in (2, 3)
                                                for M in (4, 8) for seed in range(3)]
    for N, M, d, seed in configs:
        rng = np.random.default_rng(10007 * seed + 100 * N + M)
        r = multimode_attack(N, M, d, rng=rng, n_iter=25 if quick else 120, batch=48 if quick else 96,
                             polish_iter=400 if quick else 6000)
        rows.append(dict(N=N, M=M, d_eff=d, seed=seed, **{k: r[k] for k in (
            "ratio_conjectured", "ratio_refined", "ratio_proved", "omega", "A", "s", "taus", "period",
            "ln_rho", "cem_value", "polish_nfev")}))
    return {"rows": rows, "max_ratio_conjectured": max(r["ratio_conjectured"] for r in rows),
            "max_ratio_refined": max(r["ratio_refined"] for r in rows),
            "max_ratio_proved": max(r["ratio_proved"] for r in rows), "n": len(rows),
            "seconds": time.perf_counter() - t0}


# ----------------------------------------------------------- D: detuned --


def section_detuned(quick=False):
    """omega = (1 - delta, 1), A = R(phi) diag(1, q) R(phi)^T, n cycles of the rate-optimum bang-bang
    with a period pre-scan, then Nelder-Mead over (omega, A, s, durations)."""
    from vacuum.floquet.bounds_multimode import _pack, _unpack
    t0 = time.perf_counter()
    rows = []
    rng = np.random.default_rng(3)
    grid = [(0.3, 0.03, 0.8, 2)] if quick else [(d, delta, phi, n) for d in (0.05, 0.3, 0.7)
                                                  for delta in (0.002, 0.03, 0.2) for phi in (0.5, 1.2) for n in (1, 2)]
    for d, delta, phi, ncyc in grid:
        v, tau = rate_optimum_control(d, 1.0)
        q = float(rng.uniform(-1.0, 1.0))
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        A = R @ np.diag([1.0, q]) @ R.T
        om = np.array([1.0 - delta, 1.0])
        s = np.tile(v, ncyc)
        taus0 = np.tile(tau, ncyc)
        best, bs = -1.0, 1.0
        for sc in np.linspace(0.8, 1.25, 19):
            val = multimode_ratios(om, A, d, s, taus0 * sc)["ratio_conjectured"]
            if val > best:
                best, bs = val, sc
        N, M = 2, 2 * ncyc

        def f(theta):
            o, a, ss, tt = _unpack(theta, N, M)
            try:
                return -multimode_ratios(o, a, d, ss, tt)["ratio_conjectured"]
            except (ValueError, np.linalg.LinAlgError):
                return 0.0

        res = minimize(f, _pack(om, A, 0.999 * s, taus0 * bs), method="Nelder-Mead",
                       options={"maxiter": 500 if quick else 5000, "adaptive": True, "xatol": 1e-9, "fatol": 1e-12})
        o, a, ss, tt = _unpack(res.x, N, M)
        r = multimode_ratios(o, a, d, ss, tt)
        rows.append(dict(d_eff=d, delta=delta, phi=phi, q=q, ncyc=ncyc, start_ratio=best,
                         ratio_conjectured=r["ratio_conjectured"], ratio_refined=r["ratio_refined"],
                         ratio_proved=r["ratio_proved"], omega=o, period=r["period"]))
    return {"rows": rows, "max_ratio_conjectured": max(r["ratio_conjectured"] for r in rows),
            "max_ratio_refined": max(r["ratio_refined"] for r in rows), "n": len(rows),
            "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------- E: Schur --


def section_schur(quick=False):
    """First order: omega_i = m_i/2 (one parity), s 2pi-periodic on M pieces; maximise
    sigma_max(C o Shat)/||C|| over (s, C).  The conjecture holds at first order iff this never
    exceeds 2/pi.  The best of each set is re-checked with the exact monodromy at d_eff = 0.01."""
    t0 = time.perf_counter()
    M = 24 if quick else 48
    edges = np.linspace(0.0, 2.0 * np.pi, M + 1)
    two_over_pi = 2.0 / np.pi
    # the discretised square wave's own |shat|: the attainable maximum on this grid
    sq = np.sign(np.cos(0.5 * (edges[1:] + edges[:-1])))
    shat_sq = abs(np.sum(sq * (np.exp(1j * edges[1:]) - np.exp(1j * edges[:-1]))) / (1j * 2.0 * np.pi))

    def shat(s, W):
        return np.sum(s * (np.exp(1j * W * edges[1:]) - np.exp(1j * W * edges[:-1]))) / (1j * W * 2.0 * np.pi)

    def split(x, N):
        s = np.tanh(x[:M])
        C = np.zeros((N, N))
        iu = np.triu_indices(N)
        C[iu] = x[M:M + N * (N + 1) // 2]
        return s, C + np.triu(C, 1).T

    def objective(x, N, om):
        s, C = split(x, N)
        nrm = np.linalg.norm(C, 2)
        if nrm == 0.0:
            return 0.0
        S = np.array([[shat(s, om[i] + om[j]) for j in range(N)] for i in range(N)])
        return float(np.linalg.norm(C * S, 2) / nrm)

    rng = np.random.default_rng(11)
    msets = [(2, (1, 3)), (2, (1, 5))] if quick else [(2, (1, 3)), (2, (1, 5)), (2, (2, 4)), (2, (3, 5)),
                                                        (3, (1, 3, 5)), (3, (2, 4, 6)), (3, (1, 3, 7)),
                                                        (4, (1, 3, 5, 7)), (4, (2, 4, 6, 8))]
    rows = []
    for N, m in msets:
        om = np.asarray(m, dtype=float) / 2.0
        dim = M + N * (N + 1) // 2
        best_x, best_v = None, -1.0
        for trial in range(1 if quick else 3):
            mu = 0.5 * rng.normal(size=dim)
            sig = np.ones(dim)
            bx, bv = mu.copy(), objective(mu, N, om)
            for _ in range(20 if quick else 120):
                pop = mu[None] + sig[None] * rng.normal(size=(64 if quick else 96, dim))
                pop[0] = bx
                vals = np.array([objective(p, N, om) for p in pop])
                idx = np.argsort(vals)[::-1][:10]
                if vals[idx[0]] > bv:
                    bv, bx = float(vals[idx[0]]), pop[idx[0]].copy()
                mu = 0.7 * pop[idx].mean(0) + 0.3 * mu
                sig = np.maximum(0.7 * pop[idx].std(0) + 0.3 * sig, 1e-6)
            res = minimize(lambda x: -objective(x, N, om), bx, method="Nelder-Mead",
                           options={"maxiter": 300 if quick else 4000, "adaptive": True})
            x = res.x if -res.fun > bv else bx
            v = objective(x, N, om)
            if v > best_v:
                best_v, best_x = v, x
        s, C = split(best_x, N)
        A = C / np.outer(np.sqrt(om), np.sqrt(om))
        sc = float(om.max())
        r = multimode_ratios(om / sc, A, 0.01, s, np.full(M, 2.0 * np.pi / M) * sc)
        rows.append(dict(N=N, m=list(m), schur_ratio=best_v, over_two_over_pi=best_v / two_over_pi,
                         over_square_wave_grid_max=best_v / shat_sq,
                         exact_ratio_refined_d001=r["ratio_refined"], exact_ratio_conjectured_d001=r["ratio_conjectured"]))
    return {"rows": rows, "M": M, "square_wave_shat_on_grid": shat_sq, "two_over_pi": two_over_pi,
            "max_schur_ratio": max(r["schur_ratio"] for r in rows),
            "max_over_two_over_pi": max(r["over_two_over_pi"] for r in rows),
            "max_exact_refined_ratio_d001": max(r["exact_ratio_refined_d001"] for r in rows),
            "seconds": time.perf_counter() - t0}


# --------------------------------------------------------------- main --


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else (Path(tempfile.mkdtemp(prefix="tmv-multimode-")) if args.quick else DATA)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    build = _build_info()
    res = {"__build__": build, "quick": bool(args.quick)}
    res["archive"] = section_archive(args.quick)
    print(f"[archive] {res['archive']['seconds']:.1f} s; match {res['archive']['archive_match']}; "
          f"max rate_ratio {res['archive']['max_rate_ratio']:.4f} proved {res['archive']['max_proved_ratio']:.4f} "
          f"refined {res['archive']['max_refined_ratio']:.4f}", flush=True)
    res["sum_frequency"] = section_sum_frequency(args.quick)
    print(f"[sum_frequency] {res['sum_frequency']['seconds']:.1f} s; max dev first order (d<=0.05) "
          f"{res['sum_frequency']['max_abs_dev_first_order_small_d']:.2e}", flush=True)
    res["attack"] = section_attack(args.quick)
    print(f"[attack] {res['attack']['seconds']:.1f} s; max ratios conj {res['attack']['max_ratio_conjectured']:.6f} "
          f"refined {res['attack']['max_ratio_refined']:.6f} proved {res['attack']['max_ratio_proved']:.4f}", flush=True)
    res["detuned"] = section_detuned(args.quick)
    print(f"[detuned] {res['detuned']['seconds']:.1f} s; max conj {res['detuned']['max_ratio_conjectured']:.6f}", flush=True)
    res["schur"] = section_schur(args.quick)
    print(f"[schur] {res['schur']['seconds']:.1f} s; max Schur ratio / (2/pi) {res['schur']['max_over_two_over_pi']:.5f}", flush=True)
    all_conj = [res["attack"]["max_ratio_conjectured"], res["detuned"]["max_ratio_conjectured"],
                res["sum_frequency"]["max_ratio_conjectured"], res["archive"]["max_rate_ratio"]]
    all_refined = [res["attack"]["max_ratio_refined"], res["detuned"]["max_ratio_refined"], res["archive"]["max_refined_ratio"]]
    all_proved = [res["attack"]["max_ratio_proved"], res["archive"]["max_proved_ratio"]]
    res["verdict"] = {
        "max_ratio_conjectured": max(all_conj), "max_ratio_refined": max(all_refined), "max_ratio_proved": max(all_proved),
        "conjecture_refuted": bool(max(all_conj) > 1.0 + 1e-6),
        "n_polished_optima": res["attack"]["n"] + res["detuned"]["n"] + len(res["sum_frequency"]["rows"]),
        "statement_proved": "Theorem M: ln rho(S_F)/T <= (d/2)||K0^{-1/4} P K0^{-1/4}|| <= (d/2)||P K0^{-1/2}|| "
                            "<= omega_max d_eff/2 for every measurable |s| <= 1 (number-norm Gronwall).",
        "statement_first_order": "Theorem F: two-mode sum-frequency rate (d_eff/pi) sqrt(omega1 omega2) under the square "
                                 "wave; <= sqrt(omega1 omega2)/omega_max of the conjectured bound.",
        "statement_open": "omega_max lambda*(d_eff) (and the refined (||C||/||A||) lambda*(d_eff)) beyond first order.",
    }
    build["wall_clock_s"] = time.perf_counter() - t_start
    with open(out / "s3_multimode_verdict.json", "w") as fh:
        json.dump(_jsonable(res), fh, indent=1)
    print(f"verdict {json.dumps(_jsonable(res['verdict']))}\nwall clock {build['wall_clock_s']:.1f} s -> {out}")


if __name__ == "__main__":
    main()
