#!/usr/bin/env python
"""S8 -- the certified enclosure of C_4d(gaussian)/C_FE.

Regenerates ``data/s8_certified_enclosure.json``: the computer-assisted
certificate behind ``draft/proof_4d_gaussian_constant.md``.

    .venv/bin/python papers/qei-2d-sharp-constant/notebook_certified.py
    .venv/bin/python papers/qei-2d-sharp-constant/notebook_certified.py --quick
    .venv/bin/python papers/qei-2d-sharp-constant/notebook_certified.py --out DIR

``--quick`` runs the same code path on a short basis ladder and writes outside
``data/`` (a temporary directory, or ``--out DIR``).  The full run takes a few
minutes; it writes the result file incrementally after every ladder rung, so a
crash resumes rather than restarts (rungs already present in the target file
are skipped unless ``--force``).

Sections
--------
S8a  the phi-basis matrix elements against a direct 2D quadrature of the
     kernel that ``vacuum.inequalities.qei`` itself uses (formulation check);
S8b  the certified ladder: exhibited state, interval arithmetic, lower bound;
S8c  the two deliberate mutations (kernel scaled, pair kernel scaled);
S8d  the enclosure, next to the archived momentum-space value;
S8e  a NON-CERTIFIED diagnostic (the Powers-Stormer chain on the Nystrom
     matrices) showing how much room a certified upper bound would have.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mpmath as mp                                              # noqa: E402
from vacuum.inequalities.qei import fhat_of, gaussian_f          # noqa: E402
from vacuum.inequalities.qei_certified import (                  # noqa: E402
    CERTIFIED_PINNED_RATIO,
    FEWSTER_EVESON_RATIO_UPPER_BOUND,
    certified_lower_bound,
    phi_kernel_elements,
)

FULL_LADDER = (16, 24, 32, 40, 48)
QUICK_LADDER = (10, 14, 18)


def _git(*args):
    try:
        return subprocess.check_output(("git",) + args, cwd=_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def build_info(wall):
    return {
        "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain"),
        "python": sys.version.split()[0],
        "mpmath": mp.__version__,
        "numpy": np.__version__,
        "wall_seconds": round(wall, 2),
    }


def s8a_element_crosscheck(n_check=6, dps=80, n_nodes=4000, omega_max=40.0):
    """The series elements vs a 2D Gauss-Legendre quadrature of the code kernel."""
    f = gaussian_f(1.0)
    fh = fhat_of(f)
    x, w = leggauss(int(n_nodes))
    om = 0.5 * omega_max * (x + 1.0)
    wo = 0.5 * omega_max * w
    base = np.outer(om, om) ** 1.5
    Kp = base * np.real(fh(om[:, None] - om[None, :]))
    Km = base * np.real(fh(om[:, None] + om[None, :]))
    ke = phi_kernel_elements(n_check, dps=dps)
    worst_a = worst_b = 0.0
    for m in range(n_check):
        pm = wo * om ** (m + 1.5) * np.exp(-om ** 2 / 4)
        for n in range(n_check):
            pn = wo * om ** (n + 1.5) * np.exp(-om ** 2 / 4)
            qa, qb = float(pm @ Kp @ pn), float(pm @ Km @ pn)
            sa = float(mp.mpf(ke.a_elem[m][n].a))
            sb = float(mp.mpf(ke.b_elem[m][n].a))
            worst_a = max(worst_a, abs(qa / sa - 1.0))
            worst_b = max(worst_b, abs(qb / sb - 1.0))
    return {
        "n_check": int(n_check), "n_nodes": int(n_nodes),
        "omega_max": float(omega_max),
        "max_rel_disagreement_A": worst_a,
        "max_rel_disagreement_B": worst_b,
        "note": ("<phi_m, K phi_n> from the Gamma series against a direct 2D "
                 "quadrature of the kernel built by qei.fhat_of(gaussian_f(1.0)) "
                 "-- the formulation check of draft section 2"),
    }


def s8b_ladder(ladder, out_path, existing=None, force=False):
    rows = list(existing or [])
    done = {r["n_basis"] for r in rows} if not force else set()
    for N in ladder:
        if N in done:
            continue
        t0 = time.time()
        r = certified_lower_bound(N)
        rows.append({
            "n_basis": r.n_basis, "dps": r.dps, "n_terms": r.n_terms,
            "ratio_lower_bound": r.lo,
            "energy_interval": list(r.energy_interval),
            "energy_upper_bound": r.energy_upper,
            "delta": r.delta,
            "gram_cond_log10": r.gram_cond_log10,
            "max_element_rel_width": r.max_element_rel_width,
            "energy_width_log10": r.energy_width_log10,
            "element_rel_width_log10": r.element_rel_width_log10,
            "gap_to_archived": CERTIFIED_PINNED_RATIO - r.lo,
            "seconds": round(time.time() - t0, 2),
        })
        rows.sort(key=lambda d: d["n_basis"])
        _dump(out_path, {"_partial_rows": rows})
        print("  N=%3d dps=%4d lo=%.12f  gap=%.3e  %.1fs"
              % (N, r.dps, r.lo, rows[-1]["gap_to_archived"], rows[-1]["seconds"]),
              flush=True)
    return rows


def s8c_mutations(n_basis, dps=None):
    dps = int(dps) if dps else int(60 + 9.6 * int(n_basis))
    ke = phi_kernel_elements(n_basis, dps=dps)
    out = []
    for label, kw in (("kernel_scale=1+1e-4", {"kernel_scale": 1.0001}),
                      ("b_scale=1+1e-4", {"b_scale": 1.0001}),
                      ("b_scale=1-1e-4", {"b_scale": 0.9999})):
        r = certified_lower_bound(n_basis, dps=dps, elements=ke, **kw)
        out.append({
            "mutation": label, "n_basis": r.n_basis, "dps": r.dps,
            "ratio_lower_bound": r.lo,
            "excludes_archived": bool(r.lo > CERTIFIED_PINNED_RATIO),
        })
    return out


def s8e_powers_stormer_estimate(n=400, omega_max=14.0):
    """NON-CERTIFIED diagnostic: the Powers-Stormer chain on the Nystrom matrices.

    E_Q >= -1/2 ||Ae^{1/2} - Ao^{1/2}||_HS^2 with Ae = (K + K_-)/2,
    Ao = (K - K_-)/2 is a genuine inequality (|Tr M| <= ||M||_1 applied to
    M = Ae^{1/2} Ao^{1/2}), but its right-hand side involves operator square
    roots of unbounded operators.  Evaluating it in float64 on the n = 400
    Nystrom discretisation is NOT a certified bound; it is recorded only to
    show how much room a certified version of the OTHER half of the enclosure
    would have.  See draft/proof_4d_gaussian_constant.md section 7.3.
    """
    from vacuum.inequalities.qei import _bogoliubov_ground_energy  # noqa: PLC0415
    from vacuum.inequalities.qei import _worldline_kernels, momentum_grid

    f = gaussian_f(1.0)
    om, wom = momentum_grid(omega_max, n, power=2)
    A, B = _worldline_kernels(om, wom, fhat_of(f), 3)
    A, B = np.real(A), np.real(B)
    e_q, _ = _bogoliubov_ground_energy(A, B)

    def _sqrtm(M):
        w, V = np.linalg.eigh(M)
        return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T

    D = _sqrtm(0.5 * (A + B)) - _sqrtm(0.5 * (A - B))
    e_ps = -0.5 * float(np.sum(D * D))
    to_ratio = -32.0 / (3.0 * float(np.sqrt(np.pi)))
    ev_b = np.linalg.eigvalsh(B)
    return {
        "STATUS": "ESTIMATE -- NOT CERTIFIED (float64, no quadrature remainder)",
        "n": int(n), "omega_max": float(omega_max),
        "e_Q_nystrom": e_q,
        "ratio_nystrom": to_ratio * e_q,
        "powers_stormer_energy_lower_estimate": e_ps,
        "powers_stormer_ratio_upper_estimate": to_ratio * e_ps,
        "trace_norm_K_minus_measured": float(np.abs(ev_b).sum()),
        "trace_norm_K_minus_proved_bound": float(3 * np.sqrt(np.pi) * np.e ** 2 / 32),
        "coarsened_rigorous_ratio_upper_bound_from_trace_norm":
            float(16 * (3 * np.sqrt(np.pi) * np.e ** 2 / 32) / (3 * np.sqrt(np.pi))),
        "note": ("the last row is RIGOROUS but weaker than Fewster-Eveson's 1; "
                 "the Powers-Stormer row is the estimate discussed in draft "
                 "section 7.3 and must not be quoted as a bound"),
    }


def _dump(path, payload):
    if path is None:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ladder", default=None,
                    help="comma-separated basis sizes (overrides the default)")
    a = ap.parse_args(argv)

    if a.out:
        out_dir = a.out
        os.makedirs(out_dir, exist_ok=True)
    elif a.quick:
        out_dir = tempfile.mkdtemp(prefix="qei_s8_quick_")
    else:
        out_dir = os.path.join(_HERE, "data")
    out_path = os.path.join(out_dir, "s8_certified_enclosure.json")
    ladder = ([int(x) for x in a.ladder.split(",")] if a.ladder
              else (QUICK_LADDER if a.quick else FULL_LADDER))

    existing = None
    if os.path.exists(out_path) and not a.force:
        try:
            existing = json.load(open(out_path)).get("_partial_rows") or \
                json.load(open(out_path)).get("ladder")
        except Exception:
            existing = None

    t0 = time.time()
    print("S8a  matrix elements vs 2D quadrature of the code kernel", flush=True)
    cross = s8a_element_crosscheck(n_check=5 if a.quick else 6,
                                   n_nodes=1500 if a.quick else 4000)
    print("     max rel disagreement: A %.2e  B %.2e"
          % (cross["max_rel_disagreement_A"], cross["max_rel_disagreement_B"]),
          flush=True)

    print("S8b  certified ladder", flush=True)
    rows = s8b_ladder(ladder, out_path, existing=existing, force=a.force)

    print("S8c  mutations", flush=True)
    mut_N = ladder[len(ladder) // 2]
    muts = s8c_mutations(mut_N)
    for m in muts:
        print("     %-22s lo=%.12f  excludes archived: %s"
              % (m["mutation"], m["ratio_lower_bound"], m["excludes_archived"]),
              flush=True)

    print("S8e  non-certified Powers-Stormer diagnostic", flush=True)
    ps = s8e_powers_stormer_estimate()
    print("     ESTIMATE (not a bound): R <= %.6f ; ||K_-||_1 measured %.4f, proved <= %.4f"
          % (ps["powers_stormer_ratio_upper_estimate"],
             ps["trace_norm_K_minus_measured"],
             ps["trace_norm_K_minus_proved_bound"]), flush=True)

    best = max(rows, key=lambda r: r["ratio_lower_bound"])
    payload = {
        "_source": "notebook_certified.py",
        "claim": ("C_4d(gaussian)/C_FE lies in [lo, hi]; lo is a computer-assisted "
                  "theorem (exhibited squeezed state + interval arithmetic), hi is "
                  "Fewster & Eveson, PRD 58, 084010 (1998), Eq. (40)."),
        "lo": best["ratio_lower_bound"],
        "hi": FEWSTER_EVESON_RATIO_UPPER_BOUND,
        "lo_method": ("exhibited %d-mode Gaussian state in the Cholesky-orthonormalised "
                      "phi basis, energy evaluated in mpmath interval arithmetic at "
                      "dps=%d with a proved Gamma-series tail bound"
                      % (best["n_basis"], best["dps"])),
        "hi_method": "Fewster & Eveson, PRD 58, 084010 (1998), Eq. (40): C_sharp(f) <= C_FE",
        "width": FEWSTER_EVESON_RATIO_UPPER_BOUND - best["ratio_lower_bound"],
        "archived_momentum_value": CERTIFIED_PINNED_RATIO,
        "archived_manifest_headline": 0.4574904,
        "gap_lo_to_archived": CERTIFIED_PINNED_RATIO - best["ratio_lower_bound"],
        "ladder": rows,
        "element_crosscheck": cross,
        "mutations": muts,
        "powers_stormer_estimate_NOT_CERTIFIED": ps,
        "quick": bool(a.quick),
        "build": build_info(time.time() - t0),
    }
    _dump(out_path, payload)
    print("\nS8d  enclosure  [%.12f, %.6f]" % (payload["lo"], payload["hi"]))
    print("     archived momentum value %.12f  (gap to lo: %.3e)"
          % (CERTIFIED_PINNED_RATIO, payload["gap_lo_to_archived"]))
    print("     wrote", out_path)
    return payload


if __name__ == "__main__":
    main()
