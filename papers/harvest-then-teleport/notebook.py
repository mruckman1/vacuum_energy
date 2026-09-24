"""Harvest-then-teleport: joules of switching work in, joules remotely available out.

papers/harvest-then-teleport -- the analysis behind MANIFEST.md (Layer 7, L4).
Re-runnable top to bottom, headless, no hidden state:

    .venv/bin/python papers/harvest-then-teleport/notebook.py            # full map (~10 min)
    .venv/bin/python papers/harvest-then-teleport/notebook.py --quick    # 3-point smoke (~10 s)
    .venv/bin/python papers/harvest-then-teleport/notebook.py --sweeps coupling,gap

The question (PLAN.md, L4): close the loop -- harvest entanglement from the
vacuum, then spend it as the channel for energy teleportation -- and map
the composite exchange rate between vacuum correlation and energy access.

The honest form of the second half (vacuum/composite/harvest_teleport.py,
module docstring, read it first): a freshly harvested detector pair is
locally ACTIVE, so "QET on the harvested state" would report Bob's own
local ergotropy as teleported energy.  What the harvested correlations
buy is the CLASSICAL-COMMUNICATION-ASSISTED SURPLUS -- the daemonic gain
(Francica-Goold-Plastina-Paternostro 2017) of Bob's detector under his
own oscillator Hamiltonian, computed in closed form for Gaussian
measurements on Alice (Kua-Serafini-Genoni 2025) and maximised over the
pure Gaussian measurement family.  A true-QET reference through a weak
Hotta-type x_A x_B coupling of the same pair (the joint ground state as
the locally passive reference) is reported as the second column; it uses
the ground-state correlations of that coupling, not the harvest.

The map.  One pointlike detector pair on a 1+1 lattice field (Dirichlet
chain; the working IR regulator is the synthetic mass m = 0.2 of
papers/circuit-qed-noise-thresholds, and the IR sweeps below take m -> 0
and N -> infinity on the Dirichlet chain, where m = 0 is admissible --
no zero mode -- so the map carries an IR-limit statement rather than an
arbitrary mass), amplitude ('xx') coupling, both detectors switched
simultaneously with a compact cos^2 profile, swept one axis at a time
about a base point that harvests (E_N > 0): coupling strength lam,
switching duration, separation, gap, and field temperature.  Three
further families of rows: the derivative ('xp', x_d d_t phi) coupling
along the separation and coupling axes; a DELAY axis at separation 12
for both couplings (Bob's switching delayed by 0 ... 24 sites, through
full light contact at delay = 12, where the derivative coupling harvests
genuinely in causal contact, TB-MM24, and 'xx' does not); and the IR
sequences (mass 0.2 -> 0 at N = 41, then N = 41 -> 121 massless, both
couplings; the delay axis stops at 18 because the causality audit's
15-site buffer must fit inside the N = 41 chain).  Every row also carries the NON-GAUSSIAN measurement columns
(vacuum/composite/fock_conditioning.py): photon-number-resolved and
on/off conditioning on Alice computed from the Fock representation of the
harvested state (cutoff-convergence-checked), Bob's general ergotropy,
and the PNR gain in the best squeezed number basis, next to the Gaussian
optimum, so the Gaussian lower bound can be compared with the
non-Gaussian family on every state of the map.  Per point, three ledgers
and their ratios:

  * W_in            the switching work (ledger ``work`` entries; closes to 1e-9)
  * E_N, ebits      log-negativity (nats, bits), the hashing lower bound and
                    the symmetric-state EoF -- ebits per unit work
  * surplus         E_daemonic - E_local at Bob (optimal Gaussian measurement),
                    the heterodyne and homodyne rows, the Landauer bound
                    k T_B I(A:B), and the exchange rate surplus / W_in;
                    the all-in rate surplus / (W_in + E_A^meas) at the
                    heterodyne, whose measurement cost is exactly Omega_A.

Every point runs THROUGH ``vacuum.protocol.run_protocol``: passivity audit
on the claimed T = 0 vacuum (twice: ``initial_state`` and the runner),
precision audit on the reported detector block, mandatory ledger closure,
the M2.2 dt-halving gate at 1e-9 on (E_N, PT margin, occupations, nu_B,
nu_{B|A}), the field's causality audit over the window, the no-signaling
audit on Bob's unconditional state, the passivity audit on the Hotta
reference ground state and the E_B <= E_A check.

Provenance of the inputs -- READ THIS.  The configuration is a MODEL
configuration in lattice units, chosen so that the base point harvests a
resolvable negativity inside a laptop budget.  None of the ``experiments/``
files is consumed; no number here is a prediction for a device.  The
oscillator coupling lam has no established mapping to a published device
coupling (papers/circuit-qed-noise-thresholds/MANIFEST.md).

Conventions per docs/API.md (hbar = k_B = 1, lattice units, nats unless
``_bits``).  Pure functions over frozen configs; the only side effects are
the files written under data/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy

import vacuum
from vacuum.composite import HarvestConfig, composite_ledger
from vacuum.composite.harvest_teleport import ROW_COLUMNS

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# --------------------------------------------------------------------------
# The configuration (model units; see the provenance paragraph above)
# --------------------------------------------------------------------------

N_FIELD = 41          #: Dirichlet chain; causality audit needs duration + 15 < N - 1
FIELD_MASS = 0.2      #: synthetic IR regulator (see docstring)
BASE = dict(
    separation=2,     #: sites; with T_sw = 3 the pair is in full light contact
    gap=1.0,
    lam=0.3,
    switching="cos2",
    T_sw=3.0,         #: cos^2 half-width; window [0, 6]
    T=0.0,
)
NSEG0 = 16
SUBSTEPS = 4
CONV_TOL = 1e-9       #: spec M2.2 gate tolerance
MAX_HALVINGS = 12
LEDGER_TOL = 1e-9
G_FRACTION = 0.1      #: Hotta coupling g = 0.1 Omega_A Omega_B (a tenth of the stability limit)
#: squeezed-number-basis grid of the PNR search (fock_conditioning.optimal_fock_squeezing)
FOCK_SQ_KWARGS = dict(s_grid=(1.0, 2.0, 4.0), n_theta=4)

#: the axes of the map: name -> (axis, values, fixed overrides, couplings).
#: The first five are the 'xx' map about BASE; then the 'xp' rows along
#: separation and coupling; the delay axis at separation 12 for both
#: couplings (light contact at delay = 12); the IR sequences at the base
#: point (mass -> 0 at N = 41; N -> 121 massless) for both couplings.
SWEEPS: Dict[str, Tuple[str, Sequence[float], Dict[str, Any], Tuple[str, ...]]] = {
    "coupling": ("lam", (0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0), {}, ("xx",)),
    "duration": ("T_sw", (1.0, 1.5, 2.0, 3.0, 4.0, 5.0), {}, ("xx",)),
    "separation": ("separation", (2, 4, 6, 8, 12, 16), {}, ("xx",)),
    "gap": ("gap", (0.5, 0.75, 1.0, 1.5, 2.0, 3.0), {}, ("xx",)),
    "temperature": ("T", (0.0, 0.05, 0.1, 0.2, 0.4, 0.8), {}, ("xx",)),
    "xp_separation": ("separation", (2, 4, 6, 8, 12, 16), {}, ("xp",)),
    "xp_coupling": ("lam", (0.3, 0.6, 1.0), {}, ("xp",)),
    #: delay 6 ... 18 is the range over which the light cone t' - t = 12 lies inside the
    #: two cos^2 supports (|t' - t - delay| <= 6); 18 is the last admissible value for the
    #: causality audit's buffer at N = 41 (window + 15 < N - 1)
    "delay": ("delay", (0.0, 6.0, 9.0, 12.0, 15.0, 18.0), {"separation": 12}, ("xx", "xp")),
    "ir_mass": ("mass", (0.2, 0.1, 0.05, 0.02, 0.01, 0.0), {}, ("xx", "xp")),
    "ir_size": ("N_field", (41, 61, 81, 121), {"mass": 0.0}, ("xx", "xp")),
}
#: the 'xx' simultaneous-switching map of the original claim (headline numbers)
XX_MAP_SWEEPS = ("base", "coupling", "duration", "separation", "gap", "temperature")

EXTRA_COLUMNS = ("label", "sweep", "axis", "value")


def base_config(**overrides) -> HarvestConfig:
    kw = dict(N_field=N_FIELD, mass=FIELD_MASS, bc="dirichlet", n_segments0=NSEG0,
              substeps=SUBSTEPS, conv_tol=CONV_TOL, max_halvings=MAX_HALVINGS,
              ledger_tol=LEDGER_TOL)
    kw.update(BASE)
    kw.update(overrides)
    return HarvestConfig(**kw)


def full_specs() -> List[Tuple[str, str, str, float, HarvestConfig]]:
    """(label, sweep, axis, value, cfg); every distinct configuration once.

    Labels: ``<sweep>-<axis><value>`` for 'xx' rows, with ``-xp`` appended
    for derivative-coupled rows; the 'xx' base point appears once, under
    'base', and the 'xp' base point under 'xp_separation-separation2-xp'.
    """
    specs = [("base", "base", "-", float("nan"), base_config())]
    seen = {base_config()}
    for sweep, (axis, values, fixed, couplings) in SWEEPS.items():
        for coupling in couplings:
            for v in values:
                cfg = base_config(**fixed, **{axis: v}, coupling=coupling)
                if cfg in seen:
                    continue
                seen.add(cfg)
                tag = "" if coupling == "xx" else f"-{coupling}"
                specs.append((f"{sweep}-{axis}{v:g}{tag}", sweep, axis, float(v), cfg))
    return specs


def smoke_specs() -> List[Tuple[str, str, str, float, HarvestConfig]]:
    """Four points at a relaxed gate on a smaller chain: the schema check."""
    small = dict(N_field=25, n_segments0=8, conv_tol=1e-7, max_halvings=8)
    return [
        ("smoke-base", "smoke", "-", float("nan"), replace(base_config(), **small)),
        ("smoke-spacelike", "smoke", "separation", 12.0,
         replace(base_config(separation=12, T_sw=2.0), **small)),
        ("smoke-thermal", "smoke", "T", 0.2, replace(base_config(T=0.2), **small)),
        ("smoke-xp-light-contact", "smoke", "delay", 4.0,
         replace(base_config(separation=4, T_sw=2.0, delay=4.0, coupling="xp", mass=0.0), **small)),
    ]


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def run_point(label, sweep, axis, value, cfg) -> Dict[str, Any]:
    t0 = time.time()
    row = composite_ledger(cfg, g=G_FRACTION * cfg.gaps[0] * cfg.gaps[1],
                           fock_sq_kwargs=FOCK_SQ_KWARGS).as_dict()
    row.update({"label": label, "sweep": sweep, "axis": axis, "value": value})
    row["elapsed_s"] = time.time() - t0
    return row


def run_specs(specs, verbose=True) -> List[Dict[str, Any]]:
    rows = []
    for label, sweep, axis, value, cfg in specs:
        row = run_point(label, sweep, axis, value, cfg)
        rows.append(row)
        if verbose:
            print(
                f"  {label:<30s} W_in={row['W_in']:.3e} E_N={row['E_N']:.3e} "
                f"I={row['I_AB']:.3e} surplus={row['surplus']:.3e} rate={row['exchange_rate']:.3e} "
                f"ebits/W={row['ebits_per_work']:.3e} landauer={row['landauer_ratio']:.2f} "
                f"pnr/gauss={row['fock_sq_over_gaussian']:.3f} halv={row['halvings']} "
                f"ledger={row['ledger_defect']:.1e} "
                f"audits={'ok' if row['all_audits_passed'] else 'FAIL'} ({row['elapsed_s']:.1f}s)",
                flush=True,
            )
    return rows


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def build_info() -> Dict[str, Any]:
    return {
        "generator": "papers/harvest-then-teleport/notebook.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "produced_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "vacuum_version": getattr(vacuum, "__version__", "0.1.0"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "config": dict(N_FIELD=N_FIELD, FIELD_MASS=FIELD_MASS, BASE=BASE, NSEG0=NSEG0,
                       SUBSTEPS=SUBSTEPS, CONV_TOL=CONV_TOL, MAX_HALVINGS=MAX_HALVINGS,
                       LEDGER_TOL=LEDGER_TOL, G_FRACTION=G_FRACTION,
                       FOCK_SQ_KWARGS={k: list(v) if isinstance(v, tuple) else v
                                       for k, v in FOCK_SQ_KWARGS.items()},
                       SWEEPS={k: [v[0], list(v[1]), dict(v[2]), list(v[3])]
                               for k, v in SWEEPS.items()}),
        "inputs_provenance": "model configuration in lattice units; no experiments/ file consumed",
    }


COLUMNS = EXTRA_COLUMNS + ROW_COLUMNS


def save_rows(path: Path, rows: Sequence[Dict[str, Any]], sweep: str) -> None:
    cols: Dict[str, Any] = {}
    for c in COLUMNS:
        vals = [r[c] for r in rows]
        if isinstance(vals[0], str):
            cols[c] = np.array(vals)
        elif any(v is None for v in vals):
            cols[c] = np.array([float("nan") if v is None else float(v) for v in vals])
        else:
            cols[c] = np.array(vals)
    meta = build_info()
    meta["sweep"] = sweep
    np.savez(path, __build__=json.dumps(meta, indent=2), __columns__=np.array(COLUMNS), **cols)


def load_rows(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    z = np.load(path, allow_pickle=False)
    cols = list(z["__columns__"])
    n = len(z[cols[0]])
    rows = [{c: z[c][i].item() if hasattr(z[c][i], "item") else z[c][i] for c in cols}
            for i in range(n)]
    return rows, json.loads(str(z["__build__"]))


def _argbest(rows, key, reverse=True):
    good = [r for r in rows if np.isfinite(r[key])]
    if not good:
        return None
    r = max(good, key=lambda r: r[key]) if reverse else min(good, key=lambda r: r[key])
    return {k: r[k] for k in ("label", "sweep", "axis", "value", key, "W_in", "E_N", "surplus",
                              "exchange_rate", "ebits_per_work", "work_per_ebit", "landauer_ratio",
                              "measurement", "coupling", "gain_fock_pnr", "gain_fock_pnr_sq",
                              "fock_sq_over_gaussian")}


_CFG_FIELDS = ("N_field", "mass", "separation", "gap", "lam", "switching", "T_sw", "T", "delay")


def _in_sweep(r, sweep):
    """Does the row lie on the sweep's axis?  Decided from its configuration.

    A configuration shared by two sweeps (the base point, the massless
    N = 41 row of both IR sequences, delay = 0 and separation 12, ...) is
    run once and reported under each axis it lies on.
    """
    axis, values, fixed, couplings = SWEEPS[sweep]
    if r["coupling"] not in couplings:
        return False
    ref = base_config(**fixed)
    return (all(r[f] == getattr(ref, f) for f in _CFG_FIELDS if f != axis)
            and any(r[axis] == v for v in values))


def _axis_profile(rows, sweep, keys, coupling=None):
    axis = SWEEPS[sweep][0]
    sel = [r for r in rows if _in_sweep(r, sweep) and (coupling is None or r["coupling"] == coupling)]
    sel = sorted(sel, key=lambda r: r[axis])
    return [{"value": r[axis], **{k: r[k] for k in keys}} for r in sel]


def _loglinear_fit(xs, ys):
    """Least-squares y = a + b ln x; returns (a, b, max |residual|)."""
    X = np.log(np.asarray(xs, dtype=float))
    Y = np.asarray(ys, dtype=float)
    A = np.stack([np.ones_like(X), X], axis=1)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ coef
    return float(coef[0]), float(coef[1]), float(np.max(np.abs(resid)))


def _fock_summary(rows):
    fin = [r for r in rows if np.isfinite(r["fock_sq_over_gaussian"])]
    if not fin:
        return {}
    worst = max(fin, key=lambda r: r["fock_sq_over_gaussian"])
    return {
        "n_rows": len(rows),
        "all_converged": bool(all(r["fock_converged"] for r in rows)),
        "all_local_consistent": bool(all(r["fock_local_consistent"] for r in rows)),
        "cutoff_range": [min(r["fock_cutoff"] for r in rows), max(r["fock_cutoff"] for r in rows)],
        "worst_movement": max(r["fock_movement"] for r in rows),
        "worst_trace_deficit": max(r["fock_trace_deficit"] for r in rows),
        "rows_where_non_gaussian_wins": [r["label"] for r in rows if r["best_family"] != "gaussian"],
        "max_pnr_over_gaussian": max(r["fock_over_gaussian"] for r in fin),
        "max_pnr_sq_over_gaussian": worst["fock_sq_over_gaussian"],
        "max_onoff_over_gaussian": max(r["gain_fock_onoff"] / r["surplus"] for r in fin if r["surplus"] > 0),
        "argmax_pnr_sq_over_gaussian": {k: worst[k] for k in ("label", "coupling", "E_N", "nu_B", "surplus",
                                                               "gain_fock_pnr", "gain_fock_pnr_sq",
                                                               "gain_fock_onoff", "fock_sq_s",
                                                               "fock_sq_theta", "fock_cutoff")},
        "min_pnr_over_gaussian": min(r["fock_over_gaussian"] for r in fin),
        "onoff_le_pnr_all": bool(all(r["gain_fock_onoff"] <= r["gain_fock_pnr"] + 1e-12 for r in rows)),
    }


def _xp_summary(rows, keys):
    sl = {c: [r for r in rows if r["coupling"] == c and r["spacelike"]] for c in ("xx", "xp")}
    lc = {c: [r for r in rows if r["coupling"] == c and r["light_contact"]] for c in ("xx", "xp")}
    out = {
        "spacelike_rows": {c: len(v) for c, v in sl.items()},
        "spacelike_rows_with_negativity": {c: [r["label"] for r in v if r["E_N"] > 0.0]
                                           for c, v in sl.items()},
        "spacelike_exchange_rate_range": {
            c: ([min(r["exchange_rate"] for r in v), max(r["exchange_rate"] for r in v)] if v else None)
            for c, v in sl.items()},
        "spacelike_all_positive_surplus": {c: bool(all(r["surplus"] > 0.0 for r in v)) for c, v in sl.items()},
        "light_contact_rows": {c: [{k: r[k] for k in ("label", "E_N", "W_in", "surplus", "exchange_rate",
                                                     "ebits_per_work", "I_AB")} for r in v]
                               for c, v in lc.items()},
        "profiles": {
            "xp_separation": _axis_profile(rows, "xp_separation", keys),
            "xp_coupling": _axis_profile(rows, "xp_coupling", keys),
            "delay": {c: _axis_profile(rows, "delay", keys, coupling=c) for c in ("xx", "xp")},
        },
    }
    xp_rows = [r for r in rows if r["coupling"] == "xp"]
    xp_h = [r for r in xp_rows if r["E_N"] > 0.0]
    out["xp_best_exchange_rate"] = _argbest(xp_rows, "exchange_rate")
    out["xp_best_exchange_rate_harvesting"] = _argbest(xp_h, "exchange_rate")
    out["xp_best_ebits_per_work"] = _argbest(xp_rows, "ebits_per_work")
    xp_base = [r for r in xp_rows if r["sweep"] == "xp_separation" and r["separation"] == BASE["separation"]]
    out["xp_base"] = {k: xp_base[0][k] for k in keys} if xp_base else None
    return out


def _ir_summary(rows, keys):
    out: Dict[str, Any] = {"mass": {}, "size": {}}
    for c in ("xx", "xp"):
        prof_m = _axis_profile(rows, "ir_mass", keys + ("mass", "N_field"), coupling=c)
        prof_N = _axis_profile(rows, "ir_size", keys + ("mass", "N_field"), coupling=c)
        out["mass"][c] = prof_m
        out["size"][c] = prof_N
        if prof_m:
            m0 = [p for p in prof_m if p["value"] == 0.0]
            mref = [p for p in prof_m if p["value"] == FIELD_MASS]
            if m0 and mref:
                out[f"massless_over_regulated_{c}"] = {
                    k: (m0[0][k] / mref[0][k] if mref[0][k] else None)
                    for k in ("W_in", "E_N", "surplus", "exchange_rate")}
                # movement over the last two masses of the sequence (m = 0.01 -> 0)
                last = sorted(prof_m, key=lambda p: p["value"])[:2]
                out[f"mass_limit_movement_{c}"] = {
                    k: abs(last[0][k] - last[1][k]) for k in ("W_in", "E_N", "surplus", "exchange_rate")}
        if len(prof_N) >= 3:
            Ns = [p["value"] for p in prof_N]
            fits = {}
            for k in ("W_in", "surplus", "E_N", "exchange_rate"):
                a, b, res = _loglinear_fit(Ns, [p[k] for p in prof_N])
                fits[k] = {"intercept": a, "slope_per_lnN": b, "max_residual": res,
                           "relative_change_first_to_last": (prof_N[-1][k] - prof_N[0][k]) / prof_N[0][k]
                           if prof_N[0][k] else None}
            fits["asymptotic_rate_slope_ratio"] = (
                fits["surplus"]["slope_per_lnN"] / fits["W_in"]["slope_per_lnN"]
                if fits["W_in"]["slope_per_lnN"] else None)
            out[f"size_fit_{c}"] = fits
    return out


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ("W_in", "E_N", "E_N_bits", "I_AB", "surplus", "surplus_het", "surplus_hom",
            "exchange_rate", "ebits_per_work", "work_per_ebit", "all_in_rate_het",
            "landauer_ratio", "E_local", "E_daemonic", "nu_B", "spacelike", "light_contact",
            "measurement", "hg_surplus", "ref_E_B", "ref_ratio", "coupling",
            "gain_fock_pnr", "gain_fock_onoff", "gain_fock_pnr_sq", "fock_over_gaussian",
            "fock_sq_over_gaussian", "surplus_best", "best_family", "exchange_rate_best")
    base = next(r for r in rows if r["sweep"] == "base")
    xx_map = [r for r in rows if r["sweep"] in XX_MAP_SWEEPS]
    harvesting = [r for r in xx_map if r["E_N"] > 0.0]
    dead_but_paid = [r for r in xx_map if r["E_N"] == 0.0 and r["surplus"] > 0.0]
    profiles = {s: _axis_profile(rows, s, keys, coupling="xx") for s in XX_MAP_SWEEPS if s != "base"}

    def saturation(sweep, key):
        prof = profiles[sweep]
        vals = [p[key] for p in prof if np.isfinite(p[key])]
        if len(vals) < 2:
            return None
        i = int(np.argmax(vals))
        return {"argmax": prof[i]["value"], "max": vals[i],
                "last_two_ratio": vals[-1] / vals[-2] if vals[-2] else None,
                "monotone_increasing": bool(all(np.diff(vals) >= -1e-15))}

    return {
        "n_points": len(rows),
        "n_points_xx_map": len(xx_map),
        "n_harvesting": len(harvesting),
        "n_dead_negativity_positive_surplus": len(dead_but_paid),
        "n_dead_negativity_positive_surplus_all_rows": len([r for r in rows if r["E_N"] == 0.0 and r["surplus"] > 0.0]),
        "base": {k: base[k] for k in keys},
        "best_exchange_rate": _argbest(xx_map, "exchange_rate"),
        "best_exchange_rate_harvesting": _argbest(harvesting, "exchange_rate"),
        "best_exchange_rate_all_rows": _argbest(rows, "exchange_rate"),
        "best_exchange_rate_best_family_all_rows": _argbest(rows, "exchange_rate_best"),
        "best_ebits_per_work": _argbest(xx_map, "ebits_per_work"),
        "best_ebits_per_work_all_rows": _argbest(rows, "ebits_per_work"),
        "best_work_per_ebit": _argbest(xx_map, "work_per_ebit"),
        "best_all_in_rate_het": _argbest(xx_map, "all_in_rate_het"),
        "best_landauer_ratio": _argbest(xx_map, "landauer_ratio"),
        "saturation": {s: {"exchange_rate": saturation(s, "exchange_rate"),
                           "ebits_per_work": saturation(s, "ebits_per_work")} for s in profiles},
        "profiles": profiles,
        "hotta_reference_base": {k: base[k] for k in ("g", "ref_E_A", "ref_E_B", "ref_E_B_corr",
                                                       "ref_E_B_inj", "ref_ratio", "ref_E_local",
                                                       "hg_E_local", "hg_E_daemonic", "hg_surplus",
                                                       "W_couple")},
        "fock": _fock_summary(rows),
        "xp": _xp_summary(rows, keys),
        "ir": _ir_summary(rows, keys),
        "audits": {
            "all_audits_passed": bool(all(r["all_audits_passed"] for r in rows)),
            "all_converged": bool(all(r["converged"] for r in rows)),
            "max_ledger_defect": max(abs(r["ledger_defect"]) for r in rows),
            "max_no_signaling_defect": max(r["no_signaling_defect"] for r in rows),
            "max_causality_outside": max(r["causality_max_outside"] for r in rows),
            "worst_movement": max(r["movement"] for r in rows),
            "worst_work_movement": max(r["work_movement"] for r in rows),
            "halvings_range": [min(r["halvings"] for r in rows), max(r["halvings"] for r in rows)],
            "surplus_le_landauer_all": bool(all(r["surplus_le_landauer"] for r in rows)),
            "surplus_le_full_ergotropy_all": bool(all(r["surplus_le_full_ergotropy"] for r in rows)),
            "hotta_inequality_all": bool(all(r["hotta_inequality"] for r in rows)),
            "precision_flagged_rows": [r["label"] for r in rows if r["precision_flagged"]],
            "mp_margin_flagged_rows": [r["label"] for r in rows if r["mp_margin_flagged"]],
            "fock_all_converged": bool(all(r["fock_converged"] for r in rows)),
            "fock_all_local_consistent": bool(all(r["fock_local_consistent"] for r in rows)),
        },
        "measurements_chosen": sorted({r["measurement"].split("(")[0] for r in rows}),
        "best_families_chosen": sorted({r["best_family"] for r in rows}),
        "total_elapsed_s": sum(r["elapsed_s"] for r in rows),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="3-point smoke configuration")
    ap.add_argument("--sweeps", default=None, help="comma-separated subset of the sweep names")
    ap.add_argument("--out", default=None,
                    help="output directory (default: data/; a fresh temporary directory for "
                         "--quick, so a smoke run never overwrites the archive)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out) if args.out else (
        Path(tempfile.mkdtemp(prefix="harvest-then-teleport-quick-")) if args.quick else DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Vacuum Program / Layer 7 L4 -- harvest-then-teleport composite ledger")
    print(f"  N={N_FIELD} Dirichlet m={FIELD_MASS} (IR sweeps: m -> 0, N -> 121), base {BASE}, "
          f"gate {CONV_TOL:g}, Hotta g = {G_FRACTION} Omega_A Omega_B, couplings xx and xp, "
          f"Fock (PNR / on-off) columns on every row")
    t0 = time.time()
    if args.quick:
        rows = run_specs(smoke_specs())
        save_rows(out_dir / "rows_quick.npz", rows, sweep="quick")
        print(f"\nquick: {len(rows)} rows in {time.time() - t0:.1f}s")
        return 0
    specs = full_specs()
    if args.sweeps:
        want = set(args.sweeps.split(","))
        specs = [s for s in specs if s[1] in want or s[1] == "base"]
    rows = run_specs(specs)
    name = "rows.npz" if not args.sweeps else f"rows_{args.sweeps.replace(',', '_')}.npz"
    save_rows(out_dir / name, rows, sweep="full" if not args.sweeps else args.sweeps)
    summary = summarize(rows)
    summary["wall_s"] = time.time() - t0
    summary["build"] = build_info()
    if not args.sweeps:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nsummary (headline):")
    for k in ("n_points", "n_harvesting", "n_dead_negativity_positive_surplus", "best_exchange_rate",
              "best_ebits_per_work", "best_work_per_ebit", "best_landauer_ratio", "fock", "audits"):
        print(f"  {k}: {json.dumps(summary[k], default=float)}")
    for k in ("spacelike_rows_with_negativity", "light_contact_rows", "xp_base"):
        print(f"  xp.{k}: {json.dumps(summary['xp'][k], default=float)}")
    for k in sorted(summary["ir"]):
        if k.startswith(("massless_over", "mass_limit", "size_fit")):
            print(f"  ir.{k}: {json.dumps(summary['ir'][k], default=float)}")
    print(f"\nfull map: {len(rows)} rows in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
