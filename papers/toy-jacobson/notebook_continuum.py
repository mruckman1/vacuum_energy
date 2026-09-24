"""papers/toy-jacobson -- the continuum limit of anchor 1, certified.

Companion to ``notebook.py`` (which is left untouched, so its recorded
generator sha256 still matches ``data/``).  This script regenerates
``data/H_continuum_check.json``, the numerical half of
``draft/proof_continuum_limit.md``:

  H1  the continuum first law dS = d<K> for the 1+1 massless/chiral field on
      an interval, with the exact CHM modular Hamiltonian, for an explicit
      Gaussian (squeeze) variation -- in mpmath, with no lattice code -- plus
      the mutation runs (a weight scaled by 1.001, a weight with a wrong
      shape) that show the check has teeth;
  H2  the conformal-Killing (shape) test: the weight must satisfy w''' = 0;
  H3  the O(a^2) weight-placement prediction A(kappa) for
      rho_site - rho_bond, confronted with the lattice residual measured by
      ``vacuum.geometry.jacobson.equilibrium_residual`` at the same
      (L, wavelength).

Run:
    .venv/bin/python papers/toy-jacobson/notebook_continuum.py           # ~2 min, writes data/
    .venv/bin/python papers/toy-jacobson/notebook_continuum.py --quick   # ~20 s, writes a temp dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import vacuum  # noqa: E402
from vacuum.geometry.jacobson import (coupling_chain_K, equilibrium_residual)  # noqa: E402
from vacuum.geometry.jacobson_continuum import (  # noqa: E402
    bond_split_discretization_prediction,
    bond_split_weight_coefficient,
    chm_beta,
    conformal_killing_defect,
    continuum_first_law_defect,
    entropy_weight,
    modular_symplectic_eigenvalue,
)

HERE = Path(__file__).resolve().parent
RATIOS = (4, 8, 16)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_info(mode, wall):
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=HERE).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, cwd=HERE).stdout.strip())
    except Exception:  # pragma: no cover
        head, dirty = "unknown", True
    return {
        "generator": "papers/toy-jacobson/notebook_continuum.py",
        "generator_sha256": sha, "mode": mode,
        "git_head": head, "git_dirty": dirty,
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": wall, "package": f"vacuum {vacuum.__version__}",
        "python": platform.python_version(), "numpy": np.__version__,
        "mpmath": mp.__version__, "platform": platform.platform(),
    }


def _f(x):
    """mpmath -> float with its mpmath string kept alongside."""
    return float(x)


def section_H1(dps):
    """The continuum first law with the exact CHM modular Hamiltonian."""
    R = mp.mpf(1)
    beta = lambda x: chm_beta(x, R)          # noqa: E731
    betap = lambda x: -x / R                 # noqa: E731
    runs = {}
    for label, z0, sigma, s0 in (("centred", 0.0, 2.0, 4.0),
                                 ("offcentre", 1.5, 2.0, 4.0),
                                 ("narrow", 0.0, 1.0, 8.0)):
        d = continuum_first_law_defect(R=1.0, s0=s0, sigma=sigma, z0=z0, weight=beta,
                                       weight_prime=betap, dps=dps)
        runs[label] = {"z0": z0, "sigma": sigma, "s0": s0,
                       "dS": _f(d["dS"]), "dK": _f(d["dK"]),
                       "rel_defect": _f(d["rel_defect"]),
                       "abs_defect": _f(abs(d["defect"])),
                       "norm_s": _f(d["norm_s"]), "norm_x": _f(d["norm_x"]),
                       "im_dK": _f(d["im_dK"]), "neg_s_tail": _f(d["neg_s_tail"])}
        log(f"H1 {label}: dS = {runs[label]['dS']:.15f}, rel defect {runs[label]['rel_defect']:.2e}")
    # mutations: a wrong weight must break the identity
    eps = mp.mpf("0.001")
    mut = {}
    m = continuum_first_law_defect(R=1.0, weight=lambda x: (1 + eps) * beta(x),
                                   weight_prime=lambda x: (1 + eps) * betap(x), dps=dps)
    mut["scale_1.001"] = _f(m["rel_defect"])
    c = mp.mpf("0.05")
    wsh = lambda x: beta(x) * (1 + c * x ** 2)          # noqa: E731
    m2 = continuum_first_law_defect(R=1.0, weight=wsh,
                                    weight_prime=lambda x: mp.diff(wsh, x), dps=dps)
    mut["shape_beta_times_1_plus_0.05x2"] = _f(m2["rel_defect"])
    log(f"H1 mutations: {mut}")
    # the spectral identity g(nu(s)) = 2 pi s that the entropy route uses
    spec = {}
    for sv in ("0.25", "1", "4", "12"):
        # nu(s) - 1/2 ~ e^{-2 pi s}: resolving it needs ~3s extra digits
        with mp.workdps(dps + int(3 * float(sv)) + 10):
            spec[sv] = _f(abs(entropy_weight(modular_symplectic_eigenvalue(mp.mpf(sv)))
                              - 2 * mp.pi * mp.mpf(sv)))
    return {"runs": runs, "mutations": mut, "spectral_identity_abs_err": spec, "dps": dps}


def section_H2(dps):
    """The conformal-Killing (shape) test: 2 pi int w xi''' = 0 iff w''' = 0."""
    R = mp.mpf(1)
    beta = lambda x: chm_beta(x, R)                     # noqa: E731
    out = {
        "chm": _f(conformal_killing_defect(R=1.0, weight=beta, dps=dps)["rel_defect"]),
        "scaled_1.001": _f(conformal_killing_defect(
            R=1.0, weight=lambda x: mp.mpf("1.001") * beta(x), dps=dps)["rel_defect"]),
        "cubic_plus_0.1x3": _f(conformal_killing_defect(
            R=1.0, weight=lambda x: beta(x) + mp.mpf("0.1") * x ** 3, dps=dps)["rel_defect"]),
        "quartic_plus_0.1x4": _f(conformal_killing_defect(
            R=1.0, weight=lambda x: beta(x) + mp.mpf("0.1") * x ** 4, dps=dps)["rel_defect"]),
    }
    log(f"H2 shape test: {out}")
    return out


def section_H3(N, sizes):
    """The O(a^2) weight-placement prediction against the lattice residual."""
    K = coupling_chain_K(N, [1.0])
    eig = np.linalg.eigh(K)
    det = {ba: equilibrium_residual(K, sizes=sizes, ratios=RATIOS, parity="odd", eig=eig,
                                    return_details=True, escalate_above=1e-3, beta_at=ba)
           for ba in ("site", "bond")}
    rows = []
    for rs, rb in zip(det["site"]["rows"], det["bond"]["rows"]):
        assert rs["L"] == rb["L"] and rs["ratio"] == rb["ratio"]
        L, lam = int(rs["L"]), float(rs["wavelength"])
        p = bond_split_discretization_prediction(L, lam, rho_bond=rb["rho"])
        meas = (rs["rho"] - rb["rho"]) * L ** 2
        rows.append({
            "L": L, "ratio": float(rs["ratio"]), "wavelength": lam, "kappa": p["kappa"],
            "rho_site": float(rs["rho"]), "rho_bond": float(rb["rho"]),
            "A_site_measured": float(rs["rho"]) * L ** 2,
            "A_bond_measured": float(rb["rho"]) * L ** 2,
            "A_measured": float(meas),
            "A_continuum": p["A_continuum"], "A_discrete": p["A_discrete"],
            "A_exact": p["A_exact"], "ratio_exact_over_measured": p["A_exact"] / meas,
            "ratio_discrete_over_measured": p["A_discrete"] / meas,
            "ratio_continuum_over_measured": p["A_continuum"] / meas,
            # what the derivation accounts for is the DIFFERENCE rho_site -
            # rho_bond.  The site coefficient on its own is that difference
            # plus the bond floor, which is NOT derived (draft Sec. 4.5), so
            # A_site is accounted for only to within the bond floor's share.
            "bond_share_of_site": abs(float(rb["rho"]) / float(rs["rho"])),
            "rel_err_site_only": abs(p["A_exact"] / (float(rs["rho"]) * L ** 2) - 1.0),
            # the two approximations of the DIFFERENCE, kept apart on purpose:
            # rel_err_discrete drops the (1 - rho_bond) factor only (the
            # "0.1 %" figure); rel_err_continuum is the full step to the
            # closed form A(kappa) (sums -> integrals, omega -> k, beta_edge
            # -> a/2), which is ASSUMPTION A4 and is 40x larger.
            "rel_err_discrete": abs(p["A_discrete"] / meas - 1.0),
            "rel_err_continuum": abs(p["A_continuum"] / meas - 1.0),
        })
        log(f"H3 L={L:3d} lam={lam:8.2f} kappa={p['kappa']:.4f}: measured {meas:+.4f}, "
            f"discrete {p['A_discrete']:+.4f} ({p['A_discrete'] / meas:.4f}), "
            f"continuum {p['A_continuum']:+.4f}")
    exact = [abs(r["ratio_exact_over_measured"] - 1.0) for r in rows]
    disc = [abs(r["ratio_discrete_over_measured"] - 1.0) for r in rows]
    cont = [abs(r["ratio_continuum_over_measured"] - 1.0) for r in rows]
    return {
        "N": N, "sizes": list(sizes), "ratios": list(RATIOS), "rows": rows,
        # kappa = 1e-4: small enough for the limit, large enough that the
        # 1/kappa^3 cancellation in D(kappa) is still resolved in float64
        "A_long_wavelength_limit": float(bond_split_weight_coefficient(1e-4)),
        "A_long_wavelength_limit_kappa": 1e-4,
        "max_rel_err_exact": float(max(exact)),
        "max_rel_err_discrete": float(max(disc)),
        "max_rel_err_continuum": float(max(cont)),
        # the other end of the A4 window: the best cell (largest L, longest
        # wavelength).  Quoted in the draft's status table next to the max, so
        # that neither end can be replaced by max_rel_err_discrete.
        "min_rel_err_continuum": float(min(cont)),
        # the honest accuracy with which the SITE-centred coefficient itself
        # is derived: the difference is exact, the bond floor is not derived
        "max_rel_err_site_only": float(max(r["rel_err_site_only"] for r in rows)),
        "min_rel_err_site_only": float(min(r["rel_err_site_only"] for r in rows)),
    }


def main(argv=None):
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke configuration (~20 s)")
    ap.add_argument("--out", default=None, help="output directory (default: data/)")
    args = ap.parse_args(argv)
    quick = args.quick
    out_dir = Path(args.out) if args.out else (
        Path(tempfile.mkdtemp(prefix="toyjac_continuum_")) if quick else HERE / "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    dps = 25 if quick else 40
    N = 800 if quick else 2400
    sizes = (8, 16) if quick else (8, 16, 32)
    log(f"mode={'quick' if quick else 'full'}  out={out_dir}  dps={dps}  N={N}  sizes={sizes}")
    payload = {
        "H1_continuum_first_law": section_H1(dps),
        "H2_conformal_killing_shape": section_H2(dps),
        "H3_bond_split_prediction": section_H3(N, sizes),
    }
    payload["__build__"] = build_info("quick" if quick else "full", time.time() - t0)
    path = out_dir / "H_continuum_check.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    log(f"wrote {path} ({path.stat().st_size} bytes) in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    raise SystemExit(main())
