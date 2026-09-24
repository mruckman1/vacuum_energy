"""Does the derivative coupling harvest on a genuinely SPACELIKE window?  A scan.

papers/harvest-then-teleport -- companion generator of MANIFEST.md (L4 open
item: the 'xp' coupling "with the spacelike-window rows now able to carry
E_N > 0 where 'xx' could not").  The map (notebook.py) has six 'xp'
separation rows and a delay axis; this scan asks the question on a wider
grid, at the map's gate, and archives the answer:

    .venv/bin/python papers/harvest-then-teleport/scan_spacelike.py   # ~5 min, data/xp_spacelike_scan.json

Grid: N = 61 Dirichlet chain (walls >= 19 sites from either detector on
every row, so no reflection returns inside the window), field mass m in
{0, 0.2}, gap Omega in {0.5, 1, 2, 3}, cos^2 half-width T_sw in {2, 4},
both detectors switched simultaneously, separation d in {2 T_sw + 2,
2 T_sw + 6}: every (t, t') pair of the two supports has |t - t'| <=
2 T_sw < d, so the whole interaction region lies OUTSIDE the nominal
unit-velocity light cone.

"Outside the cone" is not "strictly spacelike" on a lattice, and this
scan does not claim it is.  The lattice cone is fuzzy: the free-field
commutator [R_A(t), R_B(0)] = i (S(t) Omega)_AB leaks past it with a
Lieb-Robinson tail.  It is measured and recorded per row as
``commutator_leakage``: the max over the protocol window and over the
phi-phi, phi-pi and pi-pi blocks at the detector pair (:func:`cone_leakage`),
the pi-pi block being the one that binds for the derivative coupling,
whose interaction is x_d F^T p_field.  On the CLOSE rows here
(d - 2 T_sw = 2 sites) it reaches 2.76e-02 over the window; on the FAR
rows (d - 2 T_sw = 6) it falls to 1.85e-05.  Those two figures are the
ones this run archives (``max_commutator_leakage`` and
``max_commutator_leakage_far_rows`` in data/xp_spacelike_scan.json), and
they are what the MANIFEST, the draft outline, README and docs/STATUS.md
quote -- there is one definition of the leakage here, not two.

That matters for how the result reads.  The result is a NULL -- E_N is
exactly 0 on every row -- and residual leakage can only ADD correlation
between the detectors, never remove it.  So the leakage could only have
manufactured a false POSITIVE, and did not: the null is CONSERVATIVE,
not an artifact of residual signalling.  A reader who wants the
strictly-spacelike reading should take the far rows, where the leakage
sits six orders below the negativity scale this stack resolves.  The
coupling strength is fixed at the STRONGER of the map's two values,
lam = 0.6 (weak coupling harvests strictly less, and the map's own
lam = 0.3 spacelike rows are in data/rows.npz), which keeps the scan
affordable at the map's own 1e-9 gate: 32 points.  Every point runs the audited
``run_harvesting`` stack (passivity at T = 0, dt-halving gate at 1e-9 on
E_N / margin / occupations, mandatory ledger closure) and records E_N with
the partial-transpose margin min(nu~) - 1/2 (positive = separable by that
margin; the mp-margin rule of ``harvested_log_negativity`` guards the
floor).  The result is a statement about THIS lattice model, not about
the continuum: Teixido-Bonfill & Martin-Martinez (PRD 110, 105016 (2024))
place the derivative coupling's advantage at light contact, and the
anticommutator W^+ is the only contribution to spacelike harvesting for
either coupling.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import vacuum
from vacuum.core import harmonic_chain_K
from vacuum.core.dynamics import chain_propagator
from vacuum.detectors import run_harvesting, switching

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

N_FIELD = 61
MASSES = (0.0, 0.2)
LAMS = (0.6,)
GAPS = (0.5, 1.0, 2.0, 3.0)
T_SWS = (2.0, 4.0)
CONV_TOL = 1e-9
MAX_HALVINGS = 12


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def cone_leakage(K, site_A, site_B, duration, n_t=201):
    """max_t |[R_A(t), R_B(0)]| at the detector pair: the measured cone leakage.

    [R_i(t), R_j(0)] = i (S(t) Omega)_ij for quadratic dynamics
    (``vacuum.audits.causality``), S = :func:`chain_propagator`.  With
    S = [[C, Sw], [-Ws, C]] and Omega = [[0, I], [-I, 0]],
    S Omega = [[-Sw, C], [-C, -Ws]]: phi-phi block -Sw (the lattice
    Pauli-Jordan function), phi-pi block C, pi-pi block -Ws.  The maximum
    over the three blocks and over the protocol window is returned; the
    pi-pi block binds for the derivative coupling.
    """
    N = np.asarray(K).shape[0]
    worst = 0.0
    for t in np.linspace(0.0, float(duration), int(n_t)):
        S = chain_propagator(K, float(t))
        C, Sw, Ws = S[:N, :N], S[:N, N:], S[N:, :N]
        worst = max(worst, abs(Sw[site_A, site_B]), abs(C[site_A, site_B]),
                    abs(Ws[site_A, site_B]))
    return float(worst)


def grid():
    for m in MASSES:
        for lam in LAMS:
            for gap in GAPS:
                for T_sw in T_SWS:
                    for sep in (int(2 * T_sw) + 2, int(2 * T_sw) + 6):
                        yield m, lam, gap, T_sw, sep


def run_point(m, lam, gap, T_sw, sep, coupling="xp"):
    K = harmonic_chain_K(N_FIELD, m, bc="dirichlet")
    a = (N_FIELD - 1 - sep) // 2
    chi = switching("cos2", T_sw, T_sw)
    ts = np.linspace(0.0, 2 * T_sw, 65)
    t0 = time.time()
    h = run_harvesting(K, [a, a + sep], [gap, gap], lambda t: lam * chi(t), ts,
                       conv_tol=CONV_TOL, max_halvings=MAX_HALVINGS, coupling=coupling)
    leakage = cone_leakage(K, a, a + sep, 2 * T_sw)
    return {
        "mass": m, "lam": lam, "gap": gap, "T_sw": T_sw, "separation": sep, "coupling": coupling,
        "site_A": a, "site_B": a + sep, "duration": 2 * T_sw,
        "outside_cone": 2 * T_sw < sep, "cone_margin_sites": sep - 2 * T_sw,
        "commutator_leakage": leakage,
        "E_N": float(h.E_N), "pt_margin": float(h.neg.margin_min), "flagged": bool(h.flagged),
        "W_in": float(h.work), "n_d_A": float(h.n_d[0]), "n_d_B": float(h.n_d[1]),
        "halvings": int(h.halvings), "converged": bool(h.converged),
        "ledger_passed": bool(h.ledger_result.passed),
        "ledger_defect": float(h.ledger_result.details["defect"]),
        "passivity_passed": bool(h.passivity.passed) if h.passivity is not None else None,
        "elapsed_s": time.time() - t0,
    }


def main(argv=None) -> int:
    out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    print(f"outside-the-cone 'xp' scan: N={N_FIELD}, gate {CONV_TOL:g}")
    for m, lam, gap, T_sw, sep in grid():
        r = run_point(m, lam, gap, T_sw, sep)
        rows.append(r)
        print(f"  m={m} lam={lam} gap={gap} T_sw={T_sw} sep={sep}: E_N={r['E_N']:.3e} "
              f"margin={r['pt_margin']:.3e} leak={r['commutator_leakage']:.2e} "
              f"W={r['W_in']:.3e} halv={r['halvings']} "
              f"ledger={'ok' if r['ledger_passed'] else 'FAIL'} ({r['elapsed_s']:.1f}s)", flush=True)
    summary = {
        "n_points": len(rows),
        "all_outside_cone": bool(all(r["outside_cone"] for r in rows)),
        "max_commutator_leakage": max(r["commutator_leakage"] for r in rows),
        "max_commutator_leakage_far_rows": max(
            (r["commutator_leakage"] for r in rows if r["cone_margin_sites"] >= 6), default=None),
        "leakage_note": ("the lattice cone is fuzzy; residual leakage can only ADD "
                         "detector correlation, so a null (no negativity) is conservative"),
        "n_with_negativity": int(sum(r["E_N"] > 0.0 for r in rows)),
        "min_pt_margin": min(r["pt_margin"] for r in rows),
        "all_converged": bool(all(r["converged"] for r in rows)),
        "all_ledgers_closed": bool(all(r["ledger_passed"] for r in rows)),
        "max_ledger_defect": max(abs(r["ledger_defect"]) for r in rows),
        "flagged_rows": [i for i, r in enumerate(rows) if r["flagged"]],
        "wall_s": time.time() - t0,
        "grid": dict(N_FIELD=N_FIELD, MASSES=list(MASSES), LAMS=list(LAMS), GAPS=list(GAPS),
                     T_SWS=list(T_SWS), separations="2 T_sw + 2, 2 T_sw + 6", CONV_TOL=CONV_TOL),
        "build": {
            "generator": "papers/harvest-then-teleport/scan_spacelike.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "produced_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_head": _git_head(),
            "vacuum_version": getattr(vacuum, "__version__", "0.1.0"),
            "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "rows": rows,
    }
    (out_dir / "xp_spacelike_scan.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n{len(rows)} points, {summary['n_with_negativity']} with E_N > 0, "
          f"min PT margin {summary['min_pt_margin']:.3e}, max cone leakage "
          f"{summary['max_commutator_leakage']:.2e} (far rows "
          f"{summary['max_commutator_leakage_far_rows']:.2e}), {summary['wall_s']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
