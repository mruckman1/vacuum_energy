#!/usr/bin/env python
"""The Layer 2 pivot map: MI, discord and coherence harvesting at the sourced operating point.

Re-runnable end to end:

    .venv/bin/python papers/mi-coherence-harvesting/notebook.py             # full 12x12 x 4 run
    .venv/bin/python papers/mi-coherence-harvesting/notebook.py --quick     # 2x2 + 2 rows smoke
    .venv/bin/python papers/mi-coherence-harvesting/notebook.py --workers 8

THE QUESTION (PLAN.md, Layer 2 pivot trigger)
============================================
"If harvested negativity under honest noise models sits orders of magnitude
below detectability for all realistic parameters, the layer pivots to
mutual-information and coherence harvesting (measurable classical-quantum
correlations rather than distillable entanglement)."  The
``circuit-qed-noise-thresholds`` candidate fired that trigger: at the
Waterloo proposal's Table I scenario 2 the device box has E_N = 0 exactly
and a mutual information that survives but sits below the (E_N-derived)
tomography floor.  This candidate walks the pivot: on the SAME four
surfaces — (temperature) x (dephasing rate), (temperature) x (relaxation
rate), (temperature) x (separation), (temperature) x (oscillator coupling)
— rebuilt at 12 x 12 through the same audited code path, every point now
carries, next to E_N and I(A:B), the Gaussian quantum discord in both
directions (Adesso-Datta / Giorda-Paris closed form, Gaussian-measurement
caveat stated), the classical correlations J, the relative entropy of
coherence in the detectors' local energy bases and the correlated
coherence; the go/no-go contour of EACH measure at its floor; where the
device point sits for each; and the vacuum/communication split of the MI
on the perturbative companions with the spacelike-window rows as the
nonperturbative causal control.

The floors are of two kinds, and both are reported: (i) the flat E_N-derived
levels of the noise-threshold candidate (1e-3 nats and its band), applied to
every measure as that candidate did for MI — comparable, but a proxy; (ii) a
per-point resolvability model that re-derives, from the SAME tomography
simulation the E_N floor came from (Ren 2022: shots, readout fidelity),
what each measure would read on an uncorrelated reference under
reconstruction noise (``vacuum.detectors.correlations.monte_carlo_floors``,
assumptions (A1)-(A5) stated there), plus an analytic companion floor.

NO number backing the claim is restated here: everything is computed by
this file and written to ``data/summary.json`` / ``data/rows.npz`` with this
file's sha256 embedded.

WHAT GOES THROUGH WHAT
======================
This notebook does NOT re-implement the digital twin.  It imports the
producing notebook of the noise-threshold candidate
(``papers/circuit-qed-noise-thresholds/notebook.py``, sha256 recorded in
every archive) and reuses, unchanged, its parameter loading (every published
value through ``vacuum.experiments_io.load_experiment`` at run time — the
base point is read from the loaded parameter files, never copied), its
operating point, its geometry and switching, its one-level runner
``_run_at_level`` (every point THROUGH :func:`vacuum.protocol.run_protocol`
with the ledger, passivity, precision and mp-margin audits) and its
dt-halving gate on E_N, <n_d> and I(A:B).  What is new per point is computed
on the accepted level's detector block: the correlation measures, their
movement across the accepted halving (recorded, not gating), the Fock-cutoff
convergence of the coherence, the two-qubit proxy and the Monte-Carlo
floors.  The perturbative companions use the sibling's M split
(``pair_state_with_split``) and this candidate's cross-term split
(``mutual_information_split``) on the same lattice kernel.

Inherited provenance (MANIFEST.md): the coupling identification lam_osc =
|lambda_TB| sqrt(2 Omega) is UNSOURCED (the blocker of any device claim),
the IR mass, buffer and the four surface axes are synthetic, the device's
Gamma_phi is plot-digitised.  Nothing here changes that; the map is a
claim about the MODEL.

Pure-functional style over arrays: no module-level mutable state, no input
mutation; every sweep function takes its configuration and returns rows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vacuum.detectors import (
    UDWDetector,
    bath_occupation,
    detector_block,
    initial_state,
    pair_state_with_split,
    wightman_lattice,
)
from vacuum.detectors import correlations as corr
from vacuum.experiments_io import load_experiment

# --------------------------------------------------------------------------
# Locations and the producing sibling
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DRAFT_DIR = HERE / "draft"
SIBLING_PATH = HERE.parent / "circuit-qed-noise-thresholds" / "notebook.py"

#: Ren 2022: the tomography-precision source (assumption (A1) of the model).
REN = "circuit_qed_harvesting/ren_2022_sec4-3"


def load_sibling(path=None):
    """Import the noise-threshold notebook as a module (its code path is reused).

    ``path`` selects a *snapshot* of it (see :func:`snapshot_sibling`): a full
    run pins one byte-for-byte copy for its whole duration, so a concurrent
    edit of the repository file cannot split the archive across two code
    states.  With ``path=None`` the repository file is imported directly.
    """
    src = Path(path) if path is not None else SIBLING_PATH
    if not src.exists():
        raise FileNotFoundError(f"producing sibling not found at {src}")
    spec = importlib.util.spec_from_file_location("m27_noise_threshold_sibling", src)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sibling_sha256(path=None) -> str:
    return hashlib.sha256((Path(path) if path is not None else SIBLING_PATH).read_bytes()).hexdigest()


def snapshot_sibling() -> Path:
    """Freeze the sibling notebook for the duration of a run.

    Returns the path of a byte-for-byte copy in a temporary directory.  The
    copy is what the parent and every worker import, and its sha256 is what
    the archive records; the repository file's sha256 is recorded again at the
    end of the run, and any difference is reported (not fatal — the run used
    the snapshot throughout).
    """
    d = Path(tempfile.mkdtemp(prefix="mi-coherence-sibling-"))
    dst = d / SIBLING_PATH.name
    dst.write_bytes(SIBLING_PATH.read_bytes())
    return dst


def load_sibling_inputs(nb, coupling: str = "derived"):
    """The sibling's parameter loading, preferring its DERIVED coupling.

    ``load_inputs(coupling='derived')`` takes lam_osc and its 1 sigma from the
    parameter file's derived ``oscillator_coupling_lambda_osc_scenario2``
    (Teixido-Bonfill et al. 2026 Eqs. (33)/(66) via
    ``vacuum.detectors.circuit_mapping``) instead of the archive's convention
    identification; the two coincide numerically at this operating point
    (Omega_lat = 1) and differ only in provenance — ``derived`` rather than
    ``unsourced``.  Older sibling revisions have no such keyword; the map then
    runs on the anchored value and says so in ``coupling_mode``.
    """
    try:
        return nb.load_inputs(coupling=coupling)
    except TypeError:
        return nb.load_inputs()


# --------------------------------------------------------------------------
# The 12-point axes: subsets of the sibling's 16-point axes, anchors kept
# --------------------------------------------------------------------------

T_MK_12: Tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 22.5, 25.0, 27.5, 30.0, 40.0, 50.0, 80.0)
GPHI_HZ_12: Tuple[float, ...] = (1e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 1e8, 2e8)
G1_MHZ_12: Tuple[float, ...] = (1.0, 5.0, 8.7, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 1850.0, 3000.0)
SEP_SITES_12: Tuple[int, ...] = (1, 2, 3, 4, 5, 7, 9, 11, 12, 14, 23, 46)
LAMBDA_12: Tuple[Optional[float], ...] = (
    0.05, 0.1, 0.125, None, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6,
)  # None -> the anchored coupling (read at load time)

#: Monte-Carlo samples per state of the resolvability model (Ren 2022 repeated
#: its tomography simulation 500 times per target state; matched).
MC_SAMPLES = 500
#: quantile of the reference (uncorrelated) reconstruction that defines a floor
MC_QUANTILE = 0.95

#: The correlation measures every deliverable is computed for.
MEASURES: Tuple[str, ...] = ("E_N", "MI", "D_B", "D_A", "C_cc", "C_r")
#: Those with a Monte-Carlo floor (concurrence added for the calibration).
MC_MEASURES: Tuple[str, ...] = ("MI", "D_B", "D_A", "C_r", "C_cc", "E_N", "concurrence")
#: Those with an analytic-companion floor (the correlation/coherence measures;
#: E_N's floor IS the derived one, by construction of the companion).
X_MEASURES: Tuple[str, ...] = ("MI", "D_B", "D_A", "C_r", "C_cc")
LEVELS: Tuple[str, ...] = ("derived", "band_low", "band_high")


def axes_12(nb, inputs) -> Dict[str, Tuple[float, ...]]:
    """The 12-point axes, checked to be subsets of the sibling's axes."""
    lam = tuple(inputs.point.coupling_lambda if x is None else float(x) for x in LAMBDA_12)
    axes = {
        "temperature_mK": T_MK_12,
        "gammaphi_Hz": GPHI_HZ_12,
        "gamma1_MHz": G1_MHZ_12,
        "separation_sites": tuple(float(s) for s in SEP_SITES_12),
        "coupling_lambda": lam,
    }
    for name, vals in axes.items():
        parent = inputs.axes[name]
        for v in vals:
            if not any(abs(v - p) <= 1e-12 * max(1.0, abs(v)) for p in parent):
                raise ValueError(f"axis {name}: {v} is not on the sibling's axis")
        if len(vals) != 12:
            raise ValueError(f"axis {name} must have 12 values, got {len(vals)}")
    return axes


# --------------------------------------------------------------------------
# Row schema
# --------------------------------------------------------------------------

_PARAM_COLUMNS: Tuple[str, ...] = (
    "label", "sweep", "grid_iT", "grid_iX",
    "gap_GHz", "separation_mm", "window_ns", "switching_kind", "coupling_lambda",
    "temperature_mK", "gamma1_MHz", "gammaphi_Hz", "gap_shift_gamma", "gap_shift_GHz",
    "gap_lat", "temperature_lat", "window_lat", "separation_sites", "n_field_sites",
    "site_A", "site_B", "boundary_buffer", "field_mass", "nbar_detector", "lambda_udw",
    "spacelike_window",
)
_OBS_COLUMNS: Tuple[str, ...] = (
    "E_N", "MI", "D_B", "D_A", "J_B", "J_A", "MI_invariants", "C_r", "C_cc", "C_r_A", "C_r_B",
    "n_d_A", "n_d_B", "work", "dissipated", "min_nu_pt", "negativity_regime",
    "discord_branch_B", "discord_branch_A",
    "coherence_cutoff", "coherence_converged", "coherence_movement",
    "coherence_tail_weight", "coherence_entropy_defect", "coherence_gf_defect",
    "coherence_criterion",
    "D_B_movement", "D_A_movement", "C_r_movement", "C_cc_movement",
    "proxy_weight", "proxy_MI", "proxy_D_B", "proxy_D_A", "proxy_C_r", "proxy_C_cc",
    "proxy_E_N", "proxy_concurrence",
)
_FLOOR_FLAT_COLUMNS: Tuple[str, ...] = (
    "resolution_derived_nats", "resolution_band_low_nats", "resolution_band_high_nats",
) + tuple(f"{m}_resolvable_{lvl}" for m in MEASURES for lvl in LEVELS)
_FLOOR_MC_COLUMNS: Tuple[str, ...] = (
    "mc_samples", "mc_quantile", "mc_sigma_single", "mc_sigma_double",
) + tuple(
    f"{m}_{k}" for m in MC_MEASURES
    for k in ("floor_mc", "floor_mc_median", "target_median_mc", "target_lower_mc",
              "resolvable_mc", "detectable_mc")
)
_FLOOR_X_COLUMNS: Tuple[str, ...] = ("coherence_precision_delta",) + tuple(
    f"{m}_{k}" for m in MEASURES for k in ("floor_x", "resolvable_x")
)
_CONV_COLUMNS: Tuple[str, ...] = (
    "converged", "dt_converged", "halvings", "gate_levels_run", "movement",
    "work_movement", "n_segments", "n_substeps",
)
_AUDIT_COLUMNS: Tuple[str, ...] = (
    "mp_margin_flagged", "precision_flagged", "passivity_entry_passed",
    "passivity_initial_passed", "ledger_passed", "ledger_defect",
    "causality_audited", "all_audits_passed",
)
_PERT_COLUMNS: Tuple[str, ...] = (
    "has_perturbative_row", "perturbative_quadrature_failed",
    "P_A", "P_B", "M_abs", "M_vac_abs", "M_comm_abs", "comm_fraction_M",
    "C_abs", "C_vac_abs", "C_comm_abs", "comm_fraction_C", "C_vac_over_population",
    "vac_part_physical", "MI_pkmm", "MI_pkmm_vac", "MI_pkmm_comm", "comm_fraction_MI",
    "E_N_pert", "pert_MI", "pert_D_B", "pert_D_A", "pert_C_r", "pert_C_cc",
    "pert_clipped_weight", "perturbative_scale", "perturbative_ok",
    "pert_MI_floor_mc", "pert_MI_resolvable_mc", "pert_D_B_floor_mc", "pert_D_B_resolvable_mc",
    "pert_C_cc_floor_mc", "pert_C_cc_resolvable_mc",
)
ROW_COLUMNS: Tuple[str, ...] = (
    _PARAM_COLUMNS + _OBS_COLUMNS + _FLOOR_FLAT_COLUMNS + _FLOOR_MC_COLUMNS
    + _FLOOR_X_COLUMNS + _CONV_COLUMNS + _AUDIT_COLUMNS + _PERT_COLUMNS
    + ("inputs_all_sourced", "runtime_s")
)
_STRING_COLUMNS = frozenset({
    "label", "sweep", "switching_kind", "negativity_regime", "discord_branch_B",
    "discord_branch_A", "coherence_criterion",
})
_BOOL_COLUMNS = frozenset(
    c for c in ROW_COLUMNS
    if c.endswith(("_resolvable_derived", "_resolvable_band_low", "_resolvable_band_high",
                   "_resolvable_mc", "_detectable_mc", "_resolvable_x"))
) | frozenset({
    "spacelike_window", "coherence_converged", "converged", "mp_margin_flagged",
    "precision_flagged", "passivity_entry_passed", "passivity_initial_passed",
    "ledger_passed", "causality_audited", "all_audits_passed", "has_perturbative_row",
    "perturbative_quadrature_failed", "perturbative_ok", "inputs_all_sourced",
    "vac_part_physical",
})


def _nan_dict(keys) -> Dict[str, Any]:
    return {k: (False if k in _BOOL_COLUMNS else float("nan")) for k in keys}


# --------------------------------------------------------------------------
# One sweep point
# --------------------------------------------------------------------------


def _seed(label: str) -> int:
    return int(zlib.crc32(label.encode("utf-8")) & 0xFFFFFFFF)


def gaussian_measures(V_AB, gaps) -> Tuple[corr.GaussianCorrelations, corr.CoherenceResult]:
    gc = corr.gaussian_correlations(V_AB, (0,), (1,))
    coh = corr.gaussian_coherence(V_AB, gaps, cutoff0=4, tol=1e-10, max_cutoff=32)
    return gc, coh


def perturbative_split_columns(nb, K_field, sites, chi, gap_lat, lam_osc, prec,
                               mc_samples) -> Dict[str, Any]:
    """M split (sibling) + C split and 4x4 measures (this candidate) on one static-gap row."""
    N = int(np.asarray(K_field).shape[0])
    kernel = wightman_lattice(K_field)
    F = []
    for s in sites:
        col = np.zeros(N)
        col[int(s)] = 1.0
        F.append(col)
    det_A, det_B = UDWDetector(chi, F[0]), UDWDetector(chi, F[1])
    lam_udw = float(lam_osc) / math.sqrt(2.0 * float(gap_lat))
    state = pair_state_with_split(kernel, det_A, det_B, lam_udw, float(gap_lat),
                                  **nb.PERTURBATIVE_QUAD)
    split = corr.mutual_information_split(
        kernel, det_A, det_B, lam_udw, float(gap_lat), C=state.C, P_A=state.P_A,
        P_B=state.P_B, **nb.PERTURBATIVE_QUAD,
    )
    rho = corr.qubit_pair_from_udw(state)
    q = corr.qubit_correlations(rho)
    mc = corr.monte_carlo_floors(rho, prec, n_samples=mc_samples, seed=_seed("pert"),
                                 quantile=MC_QUANTILE)
    return {
        "has_perturbative_row": True,
        "perturbative_quadrature_failed": False,
        "P_A": float(state.P_A), "P_B": float(state.P_B),
        "M_abs": float(abs(state.M)),
        "M_vac_abs": float(abs(state.comm.M_vac)),
        "M_comm_abs": float(abs(state.comm.M_comm)),
        "comm_fraction_M": float(state.comm.comm_fraction),
        "C_abs": float(abs(split["C"])),
        "C_vac_abs": float(abs(split["C_vac"])),
        "C_comm_abs": float(abs(split["C_comm"])),
        "comm_fraction_C": float(split["comm_fraction_C"]),
        "C_vac_over_population": float(split["C_vac_over_population"]),
        "vac_part_physical": bool(split["vac_part_physical"]),
        "MI_pkmm": float(split["MI_pkmm"]),
        "MI_pkmm_vac": float(split["MI_vac"]),
        "MI_pkmm_comm": float(split["MI_comm"]),
        "comm_fraction_MI": float(split["comm_fraction_MI"]),
        "E_N_pert": float(state.log_negativity),
        "pert_MI": float(q["MI"]), "pert_D_B": float(q["D_B"]), "pert_D_A": float(q["D_A"]),
        "pert_C_r": float(q["C_r"]), "pert_C_cc": float(q["C_cc"]),
        "pert_clipped_weight": float(q["clipped_weight"]),
        "perturbative_scale": float(state.meta["perturbative_scale"]),
        "perturbative_ok": bool(state.meta["perturbative_ok"]),
        "pert_MI_floor_mc": float(mc["measures"]["MI"]["floor"]),
        "pert_MI_resolvable_mc": bool(mc["measures"]["MI"]["resolvable"]),
        "pert_D_B_floor_mc": float(mc["measures"]["D_B"]["floor"]),
        "pert_D_B_resolvable_mc": bool(mc["measures"]["D_B"]["resolvable"]),
        "pert_C_cc_floor_mc": float(mc["measures"]["C_cc"]["floor"]),
        "pert_C_cc_resolvable_mc": bool(mc["measures"]["C_cc"]["resolvable"]),
    }


def run_point(spec, inputs, nb, prec, *, perturbative: bool = True,
              mc_samples: int = MC_SAMPLES) -> Dict[str, Any]:
    """Run ONE point through the sibling's audited code path and add the pivot's measures.

    The gate loop is the sibling's (``_run_at_level`` at ``nseg0 * 2**L``
    segments through :func:`vacuum.protocol.run_protocol`, accepted when
    E_N, <n_d> and I(A:B) all move < ``spec.conv_tol`` across one halving);
    the accepted level's detector block then carries the Gaussian discords,
    classical correlations, coherences (Fock-cutoff converged), the
    two-qubit proxy and the Monte-Carlo floors.  The previous level's block
    is kept so the movement of the new measures across the accepted halving
    is recorded (it does not gate — the gate discipline stays the sibling's).
    """
    t_start = time.time()
    u = inputs.units
    res_ = inputs.resolution

    window_lat = u.time(spec.window_ns * 1e-9)
    gap_lat = u.energy(spec.gap_GHz * 1e9)
    coef = nb._val(inputs.experiments, nb.PROPOSAL, "gap_variation_coefficient")
    shift_GHz = coef * float(spec.gap_shift_gamma)
    shift_lat = u.energy(shift_GHz * 1e9)
    if gap_lat + min(0.0, shift_lat) <= 0.0:
        raise ValueError(f"gap schedule drives Omega_d non-positive at {spec.label!r}")
    temperature_lat = u.temperature(spec.temperature_mK * 1e-3)
    gamma1_lat = u.energy(spec.gamma1_MHz * 1e6)
    gammaphi_lat = u.energy(spec.gammaphi_Hz)

    delay_lat = u.length(spec.separation_mm * 1e-3)
    sep_sites = max(1, int(round(delay_lat)))
    n_field, sites = nb.geometry(sep_sites, window_lat, spec.boundary_buffer)
    K_field = nb.harmonic_chain_K(n_field, spec.field_mass, bc="dirichlet")
    chi = nb.switching_profile(spec.switching_kind, window_lat)
    gaps_of_t = nb._gaps_of_t(gap_lat, shift_lat, chi)

    def lam_of_t(t):
        return float(spec.coupling_lambda) * chi(t)

    init = initial_state(K_field, sites, gaps_of_t(0.0), temperature_lat, strict=True)

    prev: Optional[Dict[str, float]] = None
    prev_work: Optional[float] = None
    prev_VAB: Optional[np.ndarray] = None
    movement: Optional[float] = None
    work_movement: Optional[float] = None
    converged = False
    level = 0
    levels_run: List[int] = []
    res = neg = n_d = obs = None
    mi = float("nan")
    n_segments = spec.nseg0
    schedule = nb._level_schedule(int(spec.max_halvings))
    V_AB = None
    for level in schedule:
        n_segments = int(spec.nseg0) * 2**level
        levels_run.append(level)
        res, neg, mi, n_d, obs = nb._run_at_level(
            K_field, sites, gaps_of_t, lam_of_t, window_lat, n_segments,
            temperature_lat=temperature_lat, gamma1_lat=gamma1_lat,
            gammaphi_lat=gammaphi_lat, V0=init.V, K_tot0=init.K_tot,
        )
        V_AB_new = detector_block(res.V, n_field)
        if prev is not None and levels_run[-2] == level - 1:
            movement = max(abs(obs[k] - prev[k]) for k in obs)
            work_movement = abs(res.work_total - prev_work)
            if movement < float(spec.conv_tol):
                converged = True
                V_AB = V_AB_new
                break
            schedule.jump(level, movement, float(spec.conv_tol))
        prev, prev_work, prev_VAB = dict(obs), res.work_total, V_AB_new
        V_AB = V_AB_new
    if not converged:
        raise RuntimeError(
            f"point {spec.label!r}: dt-halving gate did not reach {spec.conv_tol:g} "
            f"within {spec.max_halvings} halvings (levels {levels_run}, last "
            f"movement {movement!r})"
        )
    dt_converged = window_lat / (n_segments * nb.SUBSTEPS_PER_SEGMENT)
    work = float(res.work_total)
    ledger_res = res.audits["ledger"]
    gaps_end = gaps_of_t(float(window_lat))

    # ---- the pivot's measures on the accepted block ----------------------
    gc, coh = gaussian_measures(V_AB, gaps_end)
    gc_prev, coh_prev = gaussian_measures(prev_VAB, gaps_end)
    rho_q, weight = corr.qubit_proxy_from_fock(coh.rho, coh.cutoff)
    q = corr.qubit_correlations(rho_q)
    mc = corr.monte_carlo_floors(rho_q, prec, n_samples=int(mc_samples), seed=_seed(spec.label),
                                 quantile=MC_QUANTILE)
    delta = corr.coherence_precision_from_negativity_floor(res_.derived)
    xfl = corr.correlation_floors(float(n_d[0]), float(n_d[1]), delta)

    E_N = float(neg.E_N)
    values = {
        "E_N": E_N, "MI": float(mi), "D_B": gc.D_B, "D_A": gc.D_A,
        "C_cc": coh.C_cc, "C_r": coh.C_r,
    }
    row: Dict[str, Any] = {
        "label": spec.label, "sweep": spec.sweep,
        "grid_iT": float(spec.grid_iT), "grid_iX": float(spec.grid_iX),
        "gap_GHz": float(spec.gap_GHz), "separation_mm": float(spec.separation_mm),
        "window_ns": float(spec.window_ns), "switching_kind": spec.switching_kind,
        "coupling_lambda": float(spec.coupling_lambda),
        "temperature_mK": float(spec.temperature_mK), "gamma1_MHz": float(spec.gamma1_MHz),
        "gammaphi_Hz": float(spec.gammaphi_Hz), "gap_shift_gamma": float(spec.gap_shift_gamma),
        "gap_shift_GHz": float(shift_GHz), "gap_lat": float(gap_lat),
        "temperature_lat": float(temperature_lat), "window_lat": float(window_lat),
        "separation_sites": float(sep_sites), "n_field_sites": float(n_field),
        "site_A": float(sites[0]), "site_B": float(sites[1]),
        "boundary_buffer": float(spec.boundary_buffer), "field_mass": float(spec.field_mass),
        "nbar_detector": float(bath_occupation(gap_lat, temperature_lat)),
        "lambda_udw": float(spec.coupling_lambda) / math.sqrt(2.0 * gap_lat),
        "spacelike_window": bool(sep_sites > window_lat),
        # observables
        "E_N": E_N, "MI": float(mi), "D_B": gc.D_B, "D_A": gc.D_A, "J_B": gc.J_B,
        "J_A": gc.J_A, "MI_invariants": gc.I_invariants,
        "C_r": coh.C_r, "C_cc": coh.C_cc, "C_r_A": coh.C_r_A, "C_r_B": coh.C_r_B,
        "n_d_A": float(n_d[0]), "n_d_B": float(n_d[1]), "work": work,
        "dissipated": float(res.dissipated_total), "min_nu_pt": float(neg.min_nu_pt),
        "negativity_regime": neg.regime,
        "discord_branch_B": gc.branch_B, "discord_branch_A": gc.branch_A,
        "coherence_cutoff": float(coh.cutoff), "coherence_converged": bool(coh.converged),
        "coherence_movement": float(coh.movement),
        "coherence_tail_weight": float(coh.tail_weight),
        "coherence_gf_defect": float(coh.gf_defect),
        "coherence_criterion": str(coh.criterion),
        "coherence_entropy_defect": float(abs(coh.S_trunc - coh.S_exact)),
        "D_B_movement": float(abs(gc.D_B - gc_prev.D_B)),
        "D_A_movement": float(abs(gc.D_A - gc_prev.D_A)),
        "C_r_movement": float(abs(coh.C_r - coh_prev.C_r)),
        "C_cc_movement": float(abs(coh.C_cc - coh_prev.C_cc)),
        "proxy_weight": float(weight), "proxy_MI": float(q["MI"]),
        "proxy_D_B": float(q["D_B"]), "proxy_D_A": float(q["D_A"]),
        "proxy_C_r": float(q["C_r"]), "proxy_C_cc": float(q["C_cc"]),
        "proxy_E_N": float(q["E_N"]), "proxy_concurrence": float(q["concurrence"]),
        # flat floors
        "resolution_derived_nats": float(res_.derived),
        "resolution_band_low_nats": float(res_.band_low),
        "resolution_band_high_nats": float(res_.band_high),
        # MC floors
        "mc_samples": float(mc_samples), "mc_quantile": float(MC_QUANTILE),
        "mc_sigma_single": float(prec.sigma_single), "mc_sigma_double": float(prec.sigma_double),
        "coherence_precision_delta": float(delta),
        # convergence
        "converged": bool(converged), "dt_converged": float(dt_converged),
        "halvings": float(level), "gate_levels_run": float(len(levels_run)),
        "movement": float(movement), "work_movement": float(work_movement),
        "n_segments": float(n_segments),
        "n_substeps": float(n_segments * nb.SUBSTEPS_PER_SEGMENT),
        # audits
        "mp_margin_flagged": bool(neg.flagged), "precision_flagged": bool(res.flagged),
        "passivity_entry_passed": bool(
            res.audits["passivity_entry"].passed
            if res.audits["passivity_entry"] is not None else True),
        "passivity_initial_passed": bool(
            init.passivity.passed if init.passivity is not None else True),
        "ledger_passed": bool(ledger_res.passed),
        "ledger_defect": float(ledger_res.details["defect"]),
        "causality_audited": bool(len(res.audits["causality"]) > 0),
        "all_audits_passed": bool(res.all_passed()),
        "inputs_all_sourced": False,  # the coupling is unsourced on every row
    }
    for m in MEASURES:
        for lvl, lv in res_.levels.items():
            row[f"{m}_resolvable_{lvl}"] = bool(values[m] > lv)
    for m in MC_MEASURES:
        e = mc["measures"][m]
        row[f"{m}_floor_mc"] = float(e["floor"])
        row[f"{m}_floor_mc_median"] = float(e["reference_median"])
        row[f"{m}_target_median_mc"] = float(e["target_median"])
        row[f"{m}_target_lower_mc"] = float(e["target_lower"])
        # resolvable: the ROW's full (Gaussian) value against the proxy's floor
        v_full = values.get(m, float(q["concurrence"]) if m == "concurrence" else float("nan"))
        row[f"{m}_resolvable_mc"] = bool(v_full > e["floor"])
        row[f"{m}_detectable_mc"] = bool(e["detectable"])
    for m in X_MEASURES:
        row[f"{m}_floor_x"] = float(xfl[m])
        row[f"{m}_resolvable_x"] = bool(values[m] > xfl[m])
    for m in MEASURES:  # E_N has no analytic-companion floor: its floor is the derived one
        row.setdefault(f"{m}_floor_x", float("nan"))
        row.setdefault(f"{m}_resolvable_x", False)

    has_pert = (
        perturbative and spec.switching_kind == "cos2" and spec.temperature_mK == 0.0
        and spec.gamma1_MHz == 0.0 and spec.gammaphi_Hz == 0.0
        and spec.gap_shift_gamma == 0.0
    )
    pert = _nan_dict(_PERT_COLUMNS)
    pert["has_perturbative_row"] = False
    pert["perturbative_quadrature_failed"] = False
    pert["perturbative_ok"] = False
    if has_pert:
        try:
            pert = perturbative_split_columns(nb, K_field, sites, chi, gap_lat,
                                              spec.coupling_lambda, prec, int(mc_samples))
        except RuntimeError as exc:
            pert["perturbative_quadrature_failed"] = True
            print(f"    ! {spec.label}: perturbative quadrature refused ({exc}); "
                  "row carries has_perturbative_row=False", flush=True)
    row.update(pert)
    row["runtime_s"] = float(time.time() - t_start)
    missing = [c for c in ROW_COLUMNS if c not in row]
    extra = [c for c in row if c not in ROW_COLUMNS]
    if missing or extra:
        raise AssertionError(f"row schema violation: missing {missing}, extra {extra}")
    return row


# --------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------


def surface_specs_12(nb, inputs, name: str, **common) -> List[Any]:
    axes = axes_12(nb, inputs)
    x_axis, _log = nb.SURFACES[name]
    return nb.surface_specs(inputs, name, T_values=axes["temperature_mK"],
                            X_values=axes[x_axis], **common)


def full_specs(nb, inputs, surfaces: Sequence[str]) -> List[Any]:
    specs: List[Any] = []
    for name in surfaces:
        specs.extend(surface_specs_12(nb, inputs, name))
    specs.extend(nb.split_specs(inputs, X_values=SEP_SITES_12))
    specs.extend(nb.control_specs(inputs))
    return specs


QUICK_T_MK = (0.0, 30.0)
QUICK_GPHI_HZ = (1e5, 1e7)


def quick_specs(nb, inputs) -> List[Any]:
    """2 x 2 main surface + one static-gap split row + one strong-coupling row."""
    out = nb.surface_specs(inputs, nb.MAIN_SURFACE, T_values=QUICK_T_MK,
                           X_values=QUICK_GPHI_HZ, **nb.QUICK_COMMON)
    out.extend(nb.split_specs(inputs, X_values=(7,)))
    out[-1] = replace(out[-1], **nb.QUICK_COMMON)
    out.append(nb.base_spec(inputs, "quick-coupling", "coupling", coupling_lambda=0.4,
                            temperature_mK=30.0, grid_iT=0, grid_iX=0, **nb.QUICK_COMMON))
    return out


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

_WORKER: Dict[str, Any] = {}


def _worker_init(mc_samples: int, sibling_path: Optional[str] = None) -> None:
    nb = load_sibling(sibling_path)
    _WORKER["nb"] = nb
    _WORKER["inputs"] = load_sibling_inputs(nb)
    _WORKER["prec"] = corr.tomography_precision_from_experiment(REN)
    _WORKER["mc"] = int(mc_samples)


def _worker_run(job):
    i, spec = job
    try:
        return i, run_point(spec, _WORKER["inputs"], _WORKER["nb"], _WORKER["prec"],
                            mc_samples=_WORKER["mc"]), None
    except Exception as exc:
        return i, None, f"{spec.label}: {type(exc).__name__}: {exc}"


def run_specs(specs, inputs, nb, prec, *, verbose=True, n_workers=1,
              mc_samples=MC_SAMPLES, sibling_path=None) -> List[Dict[str, Any]]:
    def report(i, row):
        if verbose:
            print(
                f"  [{i + 1:4d}/{len(specs)}] {row['label']:<28s} E_N={row['E_N']:.3e} "
                f"MI={row['MI']:.3e} D_B={row['D_B']:.3e} C_cc={row['C_cc']:.3e} "
                f"MIfl={row['MI_floor_mc']:.1e} cut={int(row['coherence_cutoff'])} "
                f"audits={'PASS' if row['all_audits_passed'] else 'FAIL'} "
                f"({row['runtime_s']:.1f}s)", flush=True,
            )

    if n_workers <= 1:
        rows = []
        for i, spec in enumerate(specs):
            row = run_point(spec, inputs, nb, prec, mc_samples=mc_samples)
            report(i, row)
            rows.append(row)
        return rows
    results: List[Optional[Dict[str, Any]]] = [None] * len(specs)
    errors: List[str] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(int(n_workers), initializer=_worker_init,
                  initargs=(int(mc_samples), None if sibling_path is None else str(sibling_path))) as pool:
        for i, row, err in pool.imap_unordered(_worker_run, list(enumerate(specs)), chunksize=1):
            if err is not None:
                errors.append(err)
                print(f"  ! FAILED {err}", flush=True)
                continue
            results[i] = row
            report(i, row)
    if errors:
        raise RuntimeError(f"{len(errors)} point(s) failed:\n  " + "\n  ".join(errors))
    return [r for r in results if r is not None]


# --------------------------------------------------------------------------
# Deliverables
# --------------------------------------------------------------------------


def surface_arrays(nb, rows, name, column):
    return nb.surface_arrays(rows, name, column)


def measure_contours(nb, rows, name: str, inputs) -> Dict[str, Any]:
    """Go/no-go contours of every measure at the flat levels and at the MC floor."""
    x_axis, log_x = nb.SURFACES[name]
    T, X, _ = surface_arrays(nb, rows, name, "E_N")
    out: Dict[str, Any] = {"x_axis": x_axis, "log_x": log_x,
                           "T_mK": [float(t) for t in T], "X": [float(x) for x in X],
                           "contours": {}}
    for m in MEASURES:
        _T, _X, Z = surface_arrays(nb, rows, name, m)
        ent = {lvl: nb.surface_contour(T, X, Z, lv, log_x=log_x)
               for lvl, lv in inputs.resolution.levels.items()}
        if m in MC_MEASURES:
            _T, _X, Fl = surface_arrays(nb, rows, name, f"{m}_floor_mc")
            ent["mc_floor"] = nb.surface_contour(T, X, Z - Fl, 0.0, log_x=log_x)
            ent["mc_floor"]["note"] = "contour of (measure - per-point Monte-Carlo floor) = 0"
        if m in X_MEASURES:
            _T, _X, Fx = surface_arrays(nb, rows, name, f"{m}_floor_x")
            ent["x_floor"] = nb.surface_contour(T, X, Z - Fx, 0.0, log_x=log_x)
        out["contours"][m] = ent
    return out


def anchor_verdicts(nb, rows, name: str, inputs) -> List[Dict[str, Any]]:
    """Every measure's verdict at the published device anchors on this surface."""
    x_axis, _log = nb.SURFACES[name]
    by = {(int(r["grid_iT"]), int(r["grid_iX"])): r for r in rows if r["sweep"] == name}
    T, X, _Z = surface_arrays(nb, rows, name, "E_N")
    out = []
    levels = inputs.resolution.levels
    for tname, tval in inputs.anchors["temperature_mK"].items():
        t_hits = np.where(np.isclose(T, tval, rtol=1e-9, atol=0.0))[0]
        if t_hits.size == 0:
            continue
        iT = int(t_hits[0])
        for xname, xval in inputs.anchors[x_axis].items():
            x_hits = np.where(np.isclose(X, xval, rtol=1e-9, atol=0.0))[0]
            if x_hits.size == 0:
                continue
            r = by[(iT, int(x_hits[0]))]
            ent = {"temperature_anchor": tname, "temperature_mK": float(T[iT]),
                   "x_anchor": xname, x_axis: float(X[int(x_hits[0])]), "label": r["label"],
                   "values": {m: float(r[m]) for m in MEASURES},
                   "verdict": {}, "mc_floor": {}, "x_floor": {}}
            for m in MEASURES:
                ent["verdict"][m] = {k: ("GO" if r[m] > v else "NO-GO") for k, v in levels.items()}
                if m in MC_MEASURES:
                    ent["verdict"][m]["mc_floor"] = "GO" if r[f"{m}_resolvable_mc"] else "NO-GO"
                    ent["mc_floor"][m] = float(r[f"{m}_floor_mc"])
                if m in X_MEASURES:
                    ent["verdict"][m]["x_floor"] = "GO" if r[f"{m}_resolvable_x"] else "NO-GO"
                    ent["x_floor"][m] = float(r[f"{m}_floor_x"])
            out.append(ent)
    return out


def _fraction(rows, key) -> float:
    return float(np.mean([bool(r[key]) for r in rows])) if rows else float("nan")


def pivot_verdict(nb, rows, inputs) -> Dict[str, Any]:
    """The pivot's question, answered per measure: device box, surfaces, where the region opens."""
    levels = inputs.resolution.levels
    out: Dict[str, Any] = {"per_surface": {}, "device_box": {}, "opening": {}}
    for name in nb.SURFACES:
        sel = [r for r in rows if r["sweep"] == name]
        if not sel:
            continue
        ent: Dict[str, Any] = {"n_points": len(sel), "max": {m: float(max(r[m] for r in sel)) for m in MEASURES}}
        for lvl in levels:
            ent[lvl] = {m: _fraction(sel, f"{m}_resolvable_{lvl}") for m in MEASURES}
            ent[lvl]["MI_only"] = float(np.mean(
                [r[f"MI_resolvable_{lvl}"] and not r[f"E_N_resolvable_{lvl}"] for r in sel]))
            ent[lvl]["D_B_only"] = float(np.mean(
                [r[f"D_B_resolvable_{lvl}"] and not r[f"E_N_resolvable_{lvl}"] for r in sel]))
        ent["mc_floor"] = {m: _fraction(sel, f"{m}_resolvable_mc") for m in MC_MEASURES}
        ent["mc_floor"]["MI_only"] = float(np.mean(
            [r["MI_resolvable_mc"] and not r["E_N_resolvable_mc"] for r in sel]))
        ent["mc_floor"]["D_B_only"] = float(np.mean(
            [r["D_B_resolvable_mc"] and not r["E_N_resolvable_mc"] for r in sel]))
        ent["x_floor"] = {m: _fraction(sel, f"{m}_resolvable_x") for m in X_MEASURES}
        out["per_surface"][name] = ent
    box = []
    for name in nb.SURFACES:
        if any(r["sweep"] == name for r in rows):
            box.extend(anchor_verdicts(nb, rows, name, inputs))
    dev = [a for a in box if a["temperature_anchor"] in ("fridge_base_30mK", "transmission_line_50mK")]
    if dev:
        by_label = {r["label"]: r for r in rows}
        ent = {"n_anchor_points": len(dev), "max": {}, "over_derived_floor": {},
               "resolvable_anywhere": {}}
        for m in MEASURES:
            vmax = max(a["values"][m] for a in dev)
            ent["max"][m] = float(vmax)
            ent["over_derived_floor"][m] = float(vmax / inputs.resolution.derived)
            ent["resolvable_anywhere"][m] = {k: bool(vmax > v) for k, v in levels.items()}
            if m in MC_MEASURES:
                ent["resolvable_anywhere"][m]["mc_floor"] = bool(any(
                    by_label[a["label"]][f"{m}_resolvable_mc"] for a in dev))
                ent["resolvable_anywhere"][m]["mc_floor_points"] = [
                    a["label"] for a in dev if by_label[a["label"]][f"{m}_resolvable_mc"]]
            if m in X_MEASURES:
                ent["resolvable_anywhere"][m]["x_floor"] = bool(any(
                    by_label[a["label"]][f"{m}_resolvable_x"] for a in dev))
        # the best anchor point per measure with its floors
        ent["best_points"] = {}
        for m in MEASURES:
            a = max(dev, key=lambda a: a["values"][m])
            r = by_label[a["label"]]
            ent["best_points"][m] = {
                "label": a["label"], "value": float(r[m]),
                "mc_floor": float(r.get(f"{m}_floor_mc", float("nan"))),
                "x_floor": float(r.get(f"{m}_floor_x", float("nan"))),
                "n_d_A": float(r["n_d_A"]),
            }
        out["device_box"] = ent
    # where the measurable region opens: on the coupling and dephasing surfaces,
    # per temperature, the first resolvable X (coupling: smallest lambda;
    # dephasing: largest Gamma_phi) for each measure and floor kind.
    for name, pick in (("coupling", "min"), ("dephasing", "max"), ("relaxation", "max"),
                       ("separation", "max")):
        sel = [r for r in rows if r["sweep"] == name]
        if not sel:
            continue
        x_axis, _log = nb.SURFACES[name]
        Ts = sorted({r["temperature_mK"] for r in sel})
        ent = {}
        for m in MEASURES:
            kinds = list(levels) + (["mc_floor"] if m in MC_MEASURES else []) + (["x_floor"] if m in X_MEASURES else [])
            ent[m] = {}
            for kind in kinds:
                key = (f"{m}_resolvable_{kind}" if kind in levels
                       else (f"{m}_resolvable_mc" if kind == "mc_floor" else f"{m}_resolvable_x"))
                per_T = {}
                for T in Ts:
                    xs = [r[x_axis] for r in sel if r["temperature_mK"] == T and r[key]]
                    per_T[f"{T:g}"] = (None if not xs else float(min(xs) if pick == "min" else max(xs)))
                ent[m][kind] = per_T
        out["opening"][name] = {"x_axis": x_axis, "rule": f"first resolvable X per T ({pick})",
                                "per_measure": ent}
    return out


def split_rows(rows) -> List[Dict[str, Any]]:
    keep = ("label", "separation_mm", "separation_sites", "window_lat", "spacelike_window",
            "E_N", "MI", "D_B", "D_A", "C_cc", "C_r", "E_N_pert", "P_A", "M_abs",
            "M_vac_abs", "M_comm_abs", "comm_fraction_M", "C_abs", "C_vac_abs",
            "C_comm_abs", "comm_fraction_C", "C_vac_over_population", "vac_part_physical",
            "MI_pkmm", "MI_pkmm_vac", "MI_pkmm_comm",
            "comm_fraction_MI", "pert_MI", "pert_D_B", "pert_D_A", "pert_C_cc", "pert_C_r",
            "pert_clipped_weight", "perturbative_ok", "perturbative_quadrature_failed",
            "has_perturbative_row", "pert_MI_floor_mc", "pert_MI_resolvable_mc",
            "pert_D_B_floor_mc", "pert_D_B_resolvable_mc", "pert_C_cc_floor_mc",
            "pert_C_cc_resolvable_mc", "MI_floor_mc", "D_B_floor_mc", "C_cc_floor_mc")
    return [{k: r[k] for k in keep}
            for r in sorted((r for r in rows if r["sweep"] == "split"),
                            key=lambda r: r["separation_sites"])]


def spacelike_control(rows, inputs) -> Dict[str, Any]:
    """The nonperturbative causal control: rows outside the switching light cone."""
    sel = [r for r in rows if r["sweep"] == "separation"]
    if not sel:
        return {}
    inside = [r for r in sel if not r["spacelike_window"]]
    outside = [r for r in sel if r["spacelike_window"]]
    def summary(group):
        if not group:
            return {}
        return {"n_points": len(group),
                "max": {m: float(max(r[m] for r in group)) for m in MEASURES},
                "resolvable_mc_fraction": {m: _fraction(group, f"{m}_resolvable_mc") for m in MC_MEASURES},
                "resolvable_derived_fraction": {m: _fraction(group, f"{m}_resolvable_derived") for m in MEASURES}}
    by_sep = {}
    for r in sorted(outside, key=lambda r: (r["separation_sites"], r["temperature_mK"])):
        by_sep.setdefault(f"{int(r['separation_sites'])}", []).append(
            {"temperature_mK": r["temperature_mK"], **{m: float(r[m]) for m in MEASURES},
             "MI_floor_mc": r["MI_floor_mc"], "MI_resolvable_mc": bool(r["MI_resolvable_mc"])})
    return {
        "window_lat": float(sel[0]["window_lat"]),
        "light_cone_sites": int(math.ceil(sel[0]["window_lat"])),
        "inside_cone": summary(inside),
        "outside_cone": summary(outside),
        "outside_cone_rows": by_sep,
        "note": "outside the cone the field commutator between the detectors vanishes "
                "(up to the lattice's exponential Lieb-Robinson leakage measured by "
                "|M_comm|/|M| and |C_comm|/|C| on the split rows at the same separations), "
                "so every correlation there is harvested, not signalled",
    }


#: the sibling's own archive (READ-ONLY): its 16 x 16 surfaces at the same
#: operating point, used as an independent check that this 12 x 12 rebuild
#: reproduces E_N and I(A:B) on every grid point the two runs share.
SIBLING_ROWS = HERE.parent / "circuit-qed-noise-thresholds" / "data" / "rows.npz"


def sibling_crosscheck(nb, rows) -> Dict[str, Any]:
    """Reproduce-the-sibling control: E_N and MI on the shared grid points.

    Matches this candidate's surface rows against
    ``papers/circuit-qed-noise-thresholds/data/rows.npz`` (read-only) on the
    full parameter key, and reports the worst absolute and relative
    differences.  A non-zero difference is a code difference, not a physics
    one: both runs take the same protocol through the same runner at the same
    gate tolerance, so this should be at roundoff except where the archive's
    anchored coupling and this run's derived coupling differ (they coincide
    at this operating point).
    """
    if not SIBLING_ROWS.exists():
        return {"available": False, "path": str(SIBLING_ROWS)}
    with np.load(SIBLING_ROWS, allow_pickle=False) as z:
        cols = [str(c) for c in z["__columns__"]]
        n = len(z[cols[0]])
        sib = [{c: z[c][i].item() for c in cols} for i in range(n)]
        meta = json.loads(str(z["__build__"]))
    key = ("temperature_mK", "gammaphi_Hz", "gamma1_MHz", "separation_mm", "coupling_lambda",
           "gap_shift_gamma", "boundary_buffer")
    def k(r):
        return tuple(round(float(r[c]), 12) for c in key)
    by = {k(r): r for r in sib if r["sweep"] in tuple(nb.SURFACES) + ("split", "control")}
    matched, dE, dM, rel = 0, 0.0, 0.0, 0.0
    worst = None
    for r in rows:
        s_ = by.get(k(r))
        if s_ is None:
            continue
        matched += 1
        de, dm = abs(r["E_N"] - s_["E_N"]), abs(r["MI"] - s_["MI"])
        if dm > dM:
            worst = {"label": r["label"], "E_N": r["E_N"], "E_N_sibling": s_["E_N"],
                     "MI": r["MI"], "MI_sibling": s_["MI"]}
        dE, dM = max(dE, de), max(dM, dm)
        if s_["MI"] > 0:
            rel = max(rel, dm / s_["MI"])
    return {
        "available": True, "path": str(SIBLING_ROWS),
        "sibling_build": {kk: meta.get(kk) for kk in ("git_head", "produced_utc", "generator_sha256")},
        "n_sibling_rows": len(sib), "n_matched": matched,
        "max_abs_E_N_difference": dE, "max_abs_MI_difference": dM,
        "max_rel_MI_difference": rel, "worst_row": worst,
    }


def brown_ordering(nb, rows) -> Dict[str, Any]:
    """Thermal amplification (Brown 2013) check: along T at the lowest-noise column."""
    out = {}
    for name in ("dephasing", "relaxation"):
        sel = [r for r in rows if r["sweep"] == name and r["grid_iX"] == 0]
        if len(sel) < 3:
            continue
        sel.sort(key=lambda r: r["temperature_mK"])
        ent = {"temperature_mK": [r["temperature_mK"] for r in sel]}
        for m in MEASURES:
            v = [float(r[m]) for r in sel]
            d = np.diff(v)
            ent[m] = {"values": v, "increasing_fraction": float(np.mean(d > 0)) if d.size else float("nan"),
                      "ratio_max_over_T0": float(max(v) / v[0]) if v[0] > 0 else float("inf")}
        out[name] = ent
    return out


def calibration(prec, inputs, mc_samples=MC_SAMPLES) -> Dict[str, Any]:
    """The model's own cross-checks against Ren 2022's reported numbers."""
    ren = load_experiment(REN)
    q = ren.quantities
    spurious_read = float(q["qst_spurious_concurrence_separable_max"].value)
    rng_range = [float(x) for x in q["qst_reconstructed_concurrence_range_target"].value]
    C_true = float(q["optimal_point_concurrence"].value)
    pops = [float(x) for x in q["optimal_point_populations"].value]
    coh = float(q["optimal_point_coherence_magnitude"].value)
    # separable input |e g>
    eg = np.zeros((4, 4), dtype=complex)
    eg[2, 2] = 1.0
    mc_eg = corr.monte_carlo_floors(eg, prec, n_samples=mc_samples, seed=11, quantile=MC_QUANTILE)
    # Ren's optimal-point X state: rho_11 = 0.63 and rho_44 = 6e-7 pair against
    # the |rho_23| = 5e-3 coherence (file note); realised as the A x B matrix
    # with populations (0.63, 0.37, 0, 6e-7) on (gg, ge, eg, ee) and the
    # coherence between |ge> and |eg>.  The concurrence must come out 0.0087.
    rho_printed = np.diag([pops[0], pops[1], pops[2], pops[3]]).astype(complex)
    rho_printed[1, 2] = 1j * coh
    rho_printed[2, 1] = -1j * coh
    # The printed matrix rounds one population to 0 next to a 5e-3 coherence
    # and is therefore not positive semidefinite (its {|ge>, |eg>} block has
    # determinant -|rho_23|^2); the nearest physical state (the same
    # projection the model's reconstructions use) is what is measured.
    rho = corr.project_to_physical(rho_printed)
    conc_model = corr.concurrence(rho)
    mc_x = corr.monte_carlo_floors(rho, prec, n_samples=mc_samples, seed=12, quantile=MC_QUANTILE)
    return {
        "precision": {"n_shots": prec.n_shots, "readout_fidelity": prec.readout_fidelity,
                      "n_settings": prec.n_settings, "sigma_single": prec.sigma_single,
                      "sigma_double": prec.sigma_double, "provenance": prec.provenance},
        "separable_input_spurious_concurrence": {
            "model_max_over_samples": mc_eg["measures"]["concurrence"]["reference_max"],
            "model_q95": mc_eg["measures"]["concurrence"]["floor"],
            "model_E_N_q95": mc_eg["measures"]["E_N"]["floor"],
            "ren_fig4_5b_axis_extent_plot_read": spurious_read,
            "ratio_model_max_over_read": mc_eg["measures"]["concurrence"]["reference_max"] / spurious_read,
        },
        "optimal_point_state": {
            "printed_matrix_min_eigenvalue": float(np.min(np.linalg.eigvalsh(rho_printed))),
            "concurrence_of_nearest_physical_state": conc_model,
            "concurrence_stated": C_true,
            "reconstructed_concurrence_median": mc_x["measures"]["concurrence"]["target_median"],
            "reconstructed_concurrence_lower_q05": mc_x["measures"]["concurrence"]["target_lower"],
            "reconstructed_concurrence_upper_q95": mc_x["measures"]["concurrence"]["target_upper"],
            "reconstructed_concurrence_nonzero_range": mc_x["measures"]["concurrence"]["target_nonzero_range"],
            "reconstructed_zero_fraction": mc_x["measures"]["concurrence"]["target_zero_fraction"],
            "ren_reported_nonzero_range": rng_range,
            "ren_reported_zero_fraction": "nearly half",
            "E_N_of_state": corr.qubit_log_negativity(rho),
            "E_N_floor_mc_on_its_reference": mc_x["measures"]["E_N"]["floor"],
            "MI_of_state": mc_x["measures"]["MI"]["true"],
            "MI_floor_mc": mc_x["measures"]["MI"]["floor"],
        },
        "derived_E_N_floor_nats": inputs.resolution.derived,
        "coherence_precision_delta": corr.coherence_precision_from_negativity_floor(inputs.resolution.derived),
    }


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def _git(args):
    try:
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except Exception:  # pragma: no cover
        return "unavailable"


def build_info(sibling_path=None) -> Dict[str, str]:
    src = Path(__file__).resolve()
    import scipy
    import vacuum
    dirty = _git(["status", "--porcelain"])
    return {
        "generator": str(src),
        "generator_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "generator_entry_point": "notebook.main()",
        "sibling_generator": str(SIBLING_PATH),
        "sibling_sha256": sibling_sha256(sibling_path),
        "sibling_snapshot": "none" if sibling_path is None else str(sibling_path),
        "correlations_module_sha256": hashlib.sha256(Path(corr.__file__).read_bytes()).hexdigest(),
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_worktree_dirty": "unavailable" if dirty == "unavailable" else str(bool(dirty)),
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vacuum_version": getattr(vacuum, "__version__", "0.1.0"),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def rows_to_arrays(rows) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for col in ROW_COLUMNS:
        vals = [r[col] for r in rows]
        if col in _STRING_COLUMNS:
            out[col] = np.array([str(v) for v in vals], dtype="U64")
        elif col in _BOOL_COLUMNS:
            out[col] = np.array([bool(v) for v in vals], dtype=bool)
        else:
            out[col] = np.array([float(v) for v in vals], dtype=float)
    return out


def save_rows(path: Path, rows, sibling_path=None, **extra) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = rows_to_arrays(rows)
    meta = build_info(sibling_path)
    meta.update({k: str(v) for k, v in extra.items()})
    payload["__build__"] = np.array(json.dumps(meta, indent=2), dtype="U8192")
    payload["__columns__"] = np.array(ROW_COLUMNS, dtype="U64")
    np.savez_compressed(path, **payload)
    return path


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        cols = [str(c) for c in z["__columns__"]]
        n = len(z[cols[0]])
        return [{c: z[c][i].item() for c in cols} for i in range(n)]


def save_surfaces(nb, path: Path, rows, inputs, sibling_path=None) -> Path:
    payload: Dict[str, np.ndarray] = {}
    cols = list(MEASURES) + ["J_B", "J_A", "n_d_A", "work"] + [f"{m}_floor_mc" for m in MC_MEASURES] \
        + [f"{m}_floor_x" for m in X_MEASURES] + ["coherence_cutoff", "all_audits_passed", "dt_converged"]
    for name in nb.SURFACES:
        if not any(r["sweep"] == name for r in rows):
            continue
        for col in cols:
            T, X, Z = surface_arrays(nb, rows, name, col)
            payload[f"{name}/T_mK"] = T
            payload[f"{name}/{nb.SURFACES[name][0]}"] = X
            payload[f"{name}/{col}"] = Z
    payload["resolution_levels_nats"] = np.array(
        [inputs.resolution.derived, inputs.resolution.band_low, inputs.resolution.band_high])
    payload["__build__"] = np.array(json.dumps(build_info(sibling_path), indent=2), dtype="U8192")
    np.savez_compressed(path, **payload)
    return path


def summarize(nb, rows, inputs, prec, *, wall_clock_s: float, mc_samples=MC_SAMPLES,
              sibling_path=None) -> Dict[str, Any]:
    surfaces_present = [n for n in nb.SURFACES if any(r["sweep"] == n for r in rows)]
    go = {name: {**measure_contours(nb, rows, name, inputs),
                 "device_anchors": anchor_verdicts(nb, rows, name, inputs)}
          for name in surfaces_present}
    ctrl = sorted((r for r in rows if r["sweep"] == "control" and r["switching_kind"] == "cos2"),
                  key=lambda r: r["boundary_buffer"])
    control: Dict[str, Any] = {}
    if ctrl:
        control = {"boundary_buffers": [r["boundary_buffer"] for r in ctrl],
                   "n_field_sites": [r["n_field_sites"] for r in ctrl]}
        for m in MEASURES:
            v = [float(r[m]) for r in ctrl]
            control[m] = v
            control[f"{m}_relative_spread"] = float((max(v) - min(v)) / max(v)) if max(v) > 0 else float("nan")
    p = inputs.point
    return {
        "build": build_info(sibling_path),
        "operating_point": {
            "table1_scenario": p.scenario, "gap_GHz": p.gap_GHz,
            "gap_shift_gamma": p.gap_shift_gamma, "gap_shift_GHz": p.gap_shift_GHz,
            "window_ns": p.window_ns, "switching_kind": p.switching_kind,
            "separation_mm": p.separation_mm, "signaling_time_ns": p.signaling_time_ns,
            "coupling_lambda_osc": p.coupling_lambda,
            "coupling_sigma": float(getattr(p, "coupling_sigma", 0.0)),
            "coupling_mode": str(getattr(p, "coupling_mode", "anchored")),
            "coupling_provenance": list(inputs.provenance["coupling_lambda"]),
            "site_mm": inputs.units.site_mm, "window_lat": inputs.units.time(p.window_ns * 1e-9),
            "read_from_parameter_files_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "resolution": {
            "derived_nats": inputs.resolution.derived,
            "band_nats": [inputs.resolution.band_low, inputs.resolution.band_high],
            "source": inputs.provenance["resolution_nats"][2],
            "flat_levels_note": "the E_N-derived levels applied to every measure as a proxy "
                                "(the noise-threshold candidate's convention); the model floors "
                                "below are the candidate's own derivation",
            "model_floors": {
                "assumptions": "(A1)-(A5) of vacuum.detectors.correlations; Ren 2022 shots and "
                               "readout fidelity read from experiments/ at run time; Gaussian "
                               "Pauli noise; Smolin-Gambetta-Smith projection; floor = "
                               f"{MC_QUANTILE:g}-quantile of the measure on reconstructions of "
                               "the uncorrelated reference; two-qubit proxy = {0,1}^2 Fock block",
                "mc_samples": mc_samples, "quantile": MC_QUANTILE,
                "sigma_single": prec.sigma_single, "sigma_double": prec.sigma_double,
            },
        },
        "axes": {k: [float(x) for x in v] for k, v in axes_12(nb, inputs).items()},
        "anchors": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in inputs.anchors.items()},
        "measures": list(MEASURES),
        "gaussian_measurement_caveat": corr.GaussianCorrelations.gaussian_measurement_caveat,
        "n_rows": len(rows),
        "n_surface_points": {n: sum(r["sweep"] == n for r in rows) for n in surfaces_present},
        "wall_clock_s": float(wall_clock_s),
        "cpu_seconds_summed": float(sum(r["runtime_s"] for r in rows)),
        "slowest_point_s": float(max(r["runtime_s"] for r in rows)),
        "all_audits_passed": bool(all(r["all_audits_passed"] for r in rows)),
        "all_converged": bool(all(r["converged"] for r in rows)),
        "all_coherence_converged": bool(all(r["coherence_converged"] for r in rows)),
        "worst_ledger_defect": float(max(abs(r["ledger_defect"]) for r in rows)),
        "worst_accepted_movement": float(max(r["movement"] for r in rows)),
        "worst_discord_movement": float(max(max(r["D_B_movement"], r["D_A_movement"]) for r in rows)),
        "worst_coherence_movement_dt": float(max(max(r["C_r_movement"], r["C_cc_movement"]) for r in rows)),
        "worst_coherence_cutoff_movement": float(max(r["coherence_movement"] for r in rows)),
        "worst_coherence_entropy_defect": float(max(r["coherence_entropy_defect"] for r in rows)),
        "worst_coherence_gf_defect": float(max(r["coherence_gf_defect"] for r in rows)),
        "coherence_criteria": {c: int(sum(r["coherence_criterion"] == c for r in rows))
                               for c in ("movement", "generating_function")},
        "max_coherence_cutoff": float(max(r["coherence_cutoff"] for r in rows)),
        "worst_MI_identity_defect": float(max(abs(r["MI"] - r["MI_invariants"]) for r in rows)),
        "min_proxy_weight": float(min(r["proxy_weight"] for r in rows)),
        "worst_proxy_MI_defect": float(max(abs(r["proxy_MI"] - r["MI"]) for r in rows)),
        "worst_proxy_D_B_defect": float(max(abs(r["proxy_D_B"] - r["D_B"]) for r in rows)),
        "mp_margin_flagged_rows": int(sum(r["mp_margin_flagged"] for r in rows)),
        "discord_branches": {b: int(sum(r["discord_branch_B"] == b for r in rows))
                             for b in ("squeezed", "homodyne", "uncorrelated")},
        "go_no_go": go,
        "pivot": pivot_verdict(nb, rows, inputs),
        "communication_split_rows": split_rows(rows),
        "spacelike_control": spacelike_control(rows, inputs),
        "thermal_amplification": brown_ordering(nb, rows),
        "calibration": calibration(prec, inputs, mc_samples=mc_samples),
        "sibling_crosscheck": sibling_crosscheck(nb, rows),
        "finite_size_control": control,
        "provenance": {k: {"file": f, "tag": t, "note": n}
                       for k, (f, t, n) in sorted(inputs.provenance.items())},
        "synthetic_or_unsourced_inputs": list(inputs.synthetic_inputs),
        "sibling_sha256": sibling_sha256(sibling_path),
    }


def _print_report(summary: Dict[str, Any]) -> None:
    print("=" * 72)
    print(f"rows: {summary['n_rows']}   wall clock: {summary['wall_clock_s']:.1f}s   "
          f"cpu-s summed: {summary['cpu_seconds_summed']:.0f}   slowest: {summary['slowest_point_s']:.1f}s")
    print(f"audits all passed: {summary['all_audits_passed']}   dt converged: {summary['all_converged']}   "
          f"coherence converged: {summary['all_coherence_converged']}   worst ledger defect: "
          f"{summary['worst_ledger_defect']:.2e}   worst MI identity defect: {summary['worst_MI_identity_defect']:.2e}")
    print(f"worst discord dt-movement: {summary['worst_discord_movement']:.2e}   worst coherence "
          f"dt-movement: {summary['worst_coherence_movement_dt']:.2e}   worst cutoff movement: "
          f"{summary['worst_coherence_cutoff_movement']:.2e}   worst gf defect: "
          f"{summary['worst_coherence_gf_defect']:.2e}   max cutoff: "
          f"{summary['max_coherence_cutoff']:.0f}   criteria: {summary['coherence_criteria']}")
    print(f"proxy: min weight {summary['min_proxy_weight']:.6f}   worst |proxy MI - MI| "
          f"{summary['worst_proxy_MI_defect']:.2e}   worst |proxy D_B - D_B| {summary['worst_proxy_D_B_defect']:.2e}")
    for name, b in summary["go_no_go"].items():
        for m in MEASURES:
            c = b["contours"][m]
            parts = [f"{lvl}:{c[lvl]['n_go']}/{c[lvl]['n_points']}" for lvl in LEVELS]
            if "mc_floor" in c:
                parts.append(f"mc:{c['mc_floor']['n_go']}")
            if "x_floor" in c:
                parts.append(f"x:{c['x_floor']['n_go']}")
            print(f"  [{name}] {m:<4s} max={c['derived']['max_z']:.3e}  GO " + "  ".join(parts))
    box = summary["pivot"].get("device_box", {})
    if box:
        print("  device box:")
        for m in MEASURES:
            bp = box["best_points"][m]
            print(f"    {m:<4s} max={bp['value']:.3e} ({box['over_derived_floor'][m]:.2f}x derived floor)"
                  f"  mc floor={bp['mc_floor']:.2e}  x floor={bp['x_floor']:.2e}  "
                  f"resolvable: {box['resolvable_anywhere'][m]}")
    xc = summary.get("sibling_crosscheck", {})
    if xc.get("available"):
        print(f"  sibling cross-check: {xc['n_matched']} shared grid points, max |dE_N| "
              f"{xc['max_abs_E_N_difference']:.2e}, max |dMI| {xc['max_abs_MI_difference']:.2e} "
              f"(rel {xc['max_rel_MI_difference']:.2e})")
    cal = summary["calibration"]["separable_input_spurious_concurrence"]
    print(f"  calibration: spurious concurrence max {cal['model_max_over_samples']:.2e} vs Ren plot read "
          f"{cal['ren_fig4_5b_axis_extent_plot_read']:.2e} (ratio {cal['ratio_model_max_over_read']:.2f})")
    sc = summary.get("spacelike_control", {})
    if sc.get("outside_cone"):
        print(f"  spacelike control: outside-cone max MI {sc['outside_cone']['max']['MI']:.3e}, "
              f"D_B {sc['outside_cone']['max']['D_B']:.3e}, C_cc {sc['outside_cone']['max']['C_cc']:.3e}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--quick", action="store_true", help="2x2 + 2 rows smoke configuration")
    ap.add_argument("--surfaces", default="", help="comma-separated subset of surface names")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--mc-samples", type=int, default=MC_SAMPLES)
    ap.add_argument("--out", default="",
                    help="output directory (default: data/ for a full run, a temporary "
                         "directory for --quick, per papers/README.md)")
    ap.add_argument("--from-rows", default="",
                    help="rebuild surfaces.npz and summary.json from an existing rows.npz "
                         "(the notebook's own post-processing, no protocol re-run)")
    args = ap.parse_args(argv)

    if args.out:
        out_dir = Path(args.out)
    elif args.quick:
        out_dir = Path(tempfile.mkdtemp(prefix="mi-coherence-quick-"))
    else:
        out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_sibling()
    nb = load_sibling(snapshot)
    sha_start = sibling_sha256(snapshot)
    inputs = load_sibling_inputs(nb)
    prec = corr.tomography_precision_from_experiment(REN)
    p = inputs.point
    print("Vacuum Program / Layer 2 pivot -- MI, discord and coherence harvesting surfaces")
    print(f"  sibling notebook sha256 {sha_start[:16]}...; operating point Table I scenario "
          f"{p.scenario}, lam_osc={p.coupling_lambda:.4f} "
          f"[{inputs.provenance['coupling_lambda'][1]}, mode "
          f"{getattr(p, 'coupling_mode', 'anchored')}, 1sigma "
          f"{getattr(p, 'coupling_sigma', 0.0):.4f}], "
          f"gap {p.gap_GHz} GHz {p.gap_shift_GHz:+.2f} GHz, window {p.window_ns} ns, d={p.separation_mm} mm")
    print(f"  flat floors: derived {inputs.resolution.derived:g} nats, band "
          f"[{inputs.resolution.band_low:g}, {inputs.resolution.band_high:g}]; tomography model: "
          f"N={prec.n_shots:g} shots, F={prec.readout_fidelity:g} -> sigma_1={prec.sigma_single:.2e}, "
          f"sigma_2={prec.sigma_double:.2e} ({prec.provenance[:40]}...)")
    t0 = time.time()
    if args.from_rows:
        rows = load_rows(Path(args.from_rows))
        with np.load(Path(args.from_rows), allow_pickle=False) as z:
            meta = json.loads(str(z["__build__"]))
        print(f"  rebuilding deliverables from {args.from_rows} ({len(rows)} rows, produced "
              f"{meta.get('produced_utc')} by generator {meta.get('generator_sha256', '')[:16]}...)")
        save_surfaces(nb, out_dir / "surfaces.npz", rows, inputs, sibling_path=snapshot)
        summary = summarize(nb, rows, inputs, prec, wall_clock_s=float("nan"),
                            mc_samples=int(float(rows[0]["mc_samples"])), sibling_path=snapshot)
        summary["rebuilt_from_rows"] = {"path": str(args.from_rows), "rows_build": meta}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        _print_report(summary)
        return 0
    if args.quick:
        specs = quick_specs(nb, inputs)
        rows = run_specs(specs, inputs, nb, prec, mc_samples=min(args.mc_samples, 200),
                         sibling_path=snapshot)
        save_rows(out_dir / "rows_quick.npz", rows, sibling_path=snapshot, sweep="quick")
        summary = summarize(nb, rows, inputs, prec, wall_clock_s=time.time() - t0,
                            mc_samples=min(args.mc_samples, 200), sibling_path=snapshot)
        (out_dir / "summary_quick.json").write_text(json.dumps(summary, indent=2, default=float))
        _print_report(summary)
        print(f"\nquick run: {len(rows)} rows in {time.time() - t0:.1f}s -> {out_dir / 'rows_quick.npz'}")
        return 0
    wanted = [s.strip() for s in args.surfaces.split(",") if s.strip()] or list(nb.SURFACES)
    for w in wanted:
        if w not in nb.SURFACES:
            raise SystemExit(f"unknown surface {w!r}; choose from {list(nb.SURFACES)}")
    specs = full_specs(nb, inputs, wanted)
    print(f"{len(specs)} points on {args.workers} worker(s)")
    for var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    rows = run_specs(specs, inputs, nb, prec, n_workers=args.workers,
                     mc_samples=args.mc_samples, sibling_path=snapshot)
    wall = time.time() - t0
    sha_repo_end = sibling_sha256()
    if sha_repo_end != sha_start:
        print(f"  NOTE: the repository's sibling notebook changed during the run "
              f"({sha_start[:16]}... -> {sha_repo_end[:16]}...); every point used the "
              f"pinned snapshot {snapshot}, whose sha256 is what the archive records")
    save_rows(out_dir / "rows.npz", rows, sibling_path=snapshot, sweep="full",
              sibling_repo_sha256_at_end=sha_repo_end)
    save_surfaces(nb, out_dir / "surfaces.npz", rows, inputs, sibling_path=snapshot)
    summary = summarize(nb, rows, inputs, prec, wall_clock_s=wall, mc_samples=args.mc_samples,
                        sibling_path=snapshot)
    summary["sibling_repo_sha256_at_end"] = sha_repo_end
    summary["sibling_unchanged_during_run"] = bool(sha_repo_end == sha_start)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    _print_report(summary)
    print(f"  wrote {out_dir / 'rows.npz'}, {out_dir / 'surfaces.npz'}, {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
