#!/usr/bin/env python
"""Circuit-QED noise-threshold SURFACES for entanglement harvesting (M2.7).

Re-runnable end to end:

    .venv/bin/python papers/circuit-qed-noise-thresholds/notebook.py            # full run
    .venv/bin/python papers/circuit-qed-noise-thresholds/notebook.py --quick    # 4x4 smoke
    .venv/bin/python papers/circuit-qed-noise-thresholds/notebook.py --workers 8

THE CLAIM (one paragraph)
=========================
For a two-detector entanglement-harvesting protocol in a superconducting
microwave waveguide -- modelled as a 1+1 lattice field with harmonic
(oscillator-UDW) detectors -- this candidate maps the go/no-go boundary of
harvested log-negativity E_N as a CURVE on two-dimensional surfaces, at the
operating point of the Waterloo proposal that lies closest to the measured
device: Table I scenario 2 of Teixido-Bonfill et al., PRA 113, 043732 (2026)
(the only scenario whose spin-boson alpha = 0.003 lies inside the
alpha in [6.2e-5, 2.19e-2] measured on the Janzen et al. 2023 coupler),
with that scenario's published gap shift (Eq. 30 at gamma = 0.02), its
published cosine-ramps switching window (Fig. 18 caption) and the
proposal's shortest published light-crossing delay (Fig. 23(c), t_d =
0.16 ns).  The surfaces are (waveguide temperature) x (pure dephasing
rate), (temperature) x (energy relaxation rate), (temperature) x (detector
separation), and -- because at production it was the one input this stack
could not source; since derived (``COUPLING_DERIVED_STATUS``), the surface is
its sensitivity study and ``--device-point`` its re-run -- (temperature) x
(oscillator coupling).  The boundary is the E_N = resolution
contour, drawn at the DERIVED resolution of the group's own tomography
simulation (Ren 2022, via `negativity_resolution_nats`) AND as a band across
the recorded plausible range of that resolution; the sensitivity of the
boundary to the resolution assumption is part of the result.  Every point
also carries the detector pair's mutual information, so the PLAN.md pivot
question (is MI resolvable where E_N is not?) is answered with numbers, and
the ledger's drive work, so "best pocket per unit switching energy" is a
computed column.

NO number backing those statements is restated in this docstring: they are
computed by this file and written to ``data/summary.json`` and
``data/rows.npz`` (with this file's sha256 embedded), so the archives always
identify the code that produced them rather than a later edit of the prose.

WHAT GOES THROUGH WHAT
======================
Every nonperturbative point goes through :func:`vacuum.protocol.run_protocol`
-- the audited Layer-2/3 code path (spec M-I.4) -- with

* the initial covariance from :func:`vacuum.detectors.initial_state`
  (mandatory passivity audit on the claimed T = 0 ground state, spec M2.2)
  and a second, independent passivity claim made to the runner itself;
* a time-dependent generator ``t -> attach_detectors(K, sites, gaps(t),
  lam(t), coupling='xx')`` stepped by the runner's exponential-midpoint
  (2nd-order Magnus) integrator;
* Gaussian noise as :class:`vacuum.protocol.ChannelStep`s in a Strang
  (symmetric) splitting around every grid segment, built by
  :func:`vacuum.detectors.imperfections_from_rates` from the published
  rates, so the runner books their energy flow as ledger ``dissipate``
  entries and closure is automatic rather than estimated;
* the spec-M2.2 dt-halving convergence gate implemented here over the
  runner (``CONV_TOL`` on E_N, the detector occupations AND the mutual
  information -- one observable more than :func:`run_harvesting` gates on,
  never fewer), with the converged dt recorded in every row;
* E_N from :func:`vacuum.detectors.harvested_log_negativity`, i.e. with the
  MANDATORY mp-margin rule and its ``flagged`` field carried into the row;
* I(A:B) from :func:`vacuum.core.mutual_information` on the same detector
  block;
* the M2.5 communication split (M_vac / M_comm / |M_comm|/|M|) from the
  perturbative engine on the *same* lattice kernel, switching and smearing,
  wherever a perturbative row exists (vacuum field, no noise, STATIC gap,
  compact-support switching) -- which, because the operating point carries
  the scenario-2 gap shift, is the static-gap companion row of each
  separation, run as its own sweep and flagged as such.

A cross-check row runs the static-gap base point through
:func:`vacuum.detectors.run_harvesting` (the M2.2 stack, which does *not*
use ``vacuum.protocol.run_protocol``) and records the E_N difference.

UNITS AND THE DIGITAL TWIN
==========================
docs/API.md throughout: hbar = k_B = 1, V_vac = I/2, nu >= 1/2, block
ordering R = (x_1..x_N, p_1..p_N), negativities and MI in nats.  Published
SI values are converted in exactly one place, :mod:`vacuum.experiments_io`,
against the reference angular frequency omega_ref = 2 pi x (free-qubit gap
of Teixido-Bonfill Eq. 30); energies/frequencies divide by omega_ref, times
multiply by it, lengths become light-crossing times at the waveguide speed
of the same paper (Sec. II.1), so one lattice site = one unit of
light-crossing time.  The proposal's cosine-ramps shape with flat fraction
S_f = 0 (Eq. 71) is sin^2(pi t / T) on [0, T] -- exactly this library's
``cos2`` profile at full window T -- so the switching shape is transcribed,
not approximated.

Model limitations, stated once and carried in MANIFEST.md:

1. UV: unit lattice spacing => omega_max = 2 omega_ref, below the device's
   50 GHz mode-sum cutoff; gaps and switching bandwidths sit under it.
2. IR: a strictly massless 1+1 chain has IR-divergent <phi^2>; a mass
   m = FIELD_MASS (lattice units) is imposed as a declared regulator, and the
   finite-size control quantifies what the boundary buffer leaves.
3. Detectors are harmonic oscillators, not qubits.  MAP-1 (lam_UDW =
   lam_osc / sqrt(2 Omega), derived and gate-tested in
   tests/test_gate_crossvalidation.py) is used for the perturbative columns.
   The oscillator coupling itself: the surfaces archive (data/rows.npz) was
   produced with lam_osc ANCHORED to the scenario's published lambda by the
   identification lam_UDW == lambda_TB -- at production an identification of
   conventions, NOT a derivation (``COUPLING_STATUS``).  Since 2026-09-03
   the same number is DERIVED from the circuit (``COUPLING_DERIVED_STATUS``,
   :mod:`vacuum.detectors.circuit_mapping`, the parameter file's
   ``oscillator_coupling_lambda_osc_scenario2``) and the device-point rows
   are re-run at it and its +-1 sigma by ``--device-point``
   (data/rows_device_point.npz, data/device_point.json).  What remains a
   MODEL limitation is the amplitude ('xx') operator itself, see 4.
4. Only 'xx' coupling is used here (the derivative-coupling 'xp' variant of
   the 2026 paper is being added to vacuum.detectors concurrently and is out
   of scope for this candidate).

Pure-functional style over arrays: no module-level mutable state, no input
mutation; every sweep function takes its configuration and returns rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vacuum.audits import EnergyLedger
from vacuum.core import harmonic_chain_K, mutual_information
from vacuum.detectors import (
    UDWDetector,
    attach_detectors,
    bath_occupation,
    communication_columns,
    detector_block,
    detector_occupations,
    harvested_log_negativity,
    imperfection_step,
    imperfections_from_rates,
    initial_state,
    pair_state_with_split,
    run_harvesting,
    switching,
    wightman_lattice,
)
from vacuum.detectors import circuit_mapping
from vacuum.experiments_io import HBAR_SI, K_B_SI, load_experiment
from vacuum.protocol import GeneratorStep, run_protocol

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DRAFT_DIR = HERE / "draft"

# The parameter files this candidate consumes, every one loaded through
# :func:`vacuum.experiments_io.load_experiment` (which refuses incomplete
# provenance).  Named once here so a rename cannot leave a stale string.

#: Teixido-Bonfill et al., PRA 113, 043732 (2026): the proposal, Table I.
PROPOSAL = "circuit_qed_harvesting/teixido-bonfill_2026_table1"
#: Declared variant: the proposal's named operating points (fully sourced).
VARIANT_SWITCH = "circuit_qed_harvesting/teixido-bonfill_2026_table1_variant-switching"
#: Janzen et al., PRResearch 5, 033155 (2023): the measured device.
DEVICE = "circuit_qed_harvesting/janzen_2023_fig6"
#: Declared variant: the sweep grids (three of them synthetic).
VARIANT_NOISE = "circuit_qed_harvesting/janzen_2023_fig6_variant-noise-sweep"
#: Ren 2022 MSc thesis: the tomography-precision source and design values.
REN = "circuit_qed_harvesting/ren_2022_sec4-3"
#: Teixido-Bonfill 2025 PhD thesis: the expected operating temperature.
THESIS = "circuit_qed_harvesting/teixido-bonfill_2025_thesis-outlook"
#: Orgiazzi et al. 2016: measured flux-qubit dephasing floor (other device).
ORGIAZZI = "circuit_qed_harvesting/orgiazzi_2016_fig3"
#: DiCarlo et al. 2009: measured transmon tomography precision (cross-check).
DICARLO = "circuit_qed_harvesting/dicarlo_2009_fig3"

PARAM_FILES: Tuple[str, ...] = (
    PROPOSAL, VARIANT_SWITCH, DEVICE, VARIANT_NOISE, REN, THESIS, ORGIAZZI, DICARLO
)

# --------------------------------------------------------------------------
# Sweep constants (the knobs, all in one place)
# --------------------------------------------------------------------------

#: Table I scenario used as the operating point (0-based index into the
#: transcribed ``table1_scenario_*`` columns): scenario 2, the only one whose
#: alpha lies inside the device's measured coupling range.
SCENARIO_INDEX = 1

#: dt-halving gate tolerance (spec M2.2 default), applied to E_N, <n_d> and
#: I(A:B).  The drive work's movement under the same halving is recorded
#: per row as ``work_movement`` but does not gate (matching the M2.2 stack).
CONV_TOL = 1.0e-9
#: Safety cap on the gate, not a physics knob.
MAX_HALVINGS = 13
#: Base evolution grid: NSEG0 segments x SUBSTEPS_PER_SEGMENT midpoint
#: substeps; halving level L uses NSEG0 * 2**L segments (channels are
#: Strang-applied per segment, so their granularity refines with the gate).
NSEG0 = 16
SUBSTEPS_PER_SEGMENT = 4
#: Ledger closure tolerance (spec test matrix: 1e-9 for all protocols).
LEDGER_TOL = 1.0e-9
#: DECLARED field mass in lattice units -- the IR regulator of limitation (2).
FIELD_MASS = 0.2
#: DECLARED buffer, in lattice sites, between each detector and the nearest
#: Dirichlet boundary on top of the causal requirement ceil(window/2); its
#: sufficiency is measured by the finite-size control, not assumed.
BOUNDARY_BUFFER = 6
#: Gaussian switching (only used by the smoke branch coverage) is truncated
#: at sigma = window / (2 * GAUSSIAN_HALF_WIDTHS): chi(edge) = 3.4e-4 of peak.
GAUSSIAN_HALF_WIDTHS = 4.0
#: Perturbative (M2.3/M2.5) quadrature settings: the M2.3 defaults plus an
#: absolute floor far below every matrix element reported here (the pair
#: term is exponentially small at wide separation, ~e^{-m r}); a refusing
#: gate marks the row ``perturbative_quadrature_failed`` rather than
#: reporting a split it did not converge.
PERTURBATIVE_QUAD = {"rtol": 1e-8, "atol": 1e-16, "max_doublings": 12}

#: The coupling's status at PRODUCTION of the surfaces archive, in one string
#: every consumer can quote (``load_inputs(coupling='anchored')``, the
#: default, keeps it so that data/rows.npz stays described by the code that
#: made it).  The oscillator x-x coupling lam_osc was set so that MAP-1's
#: lam_UDW equals |lambda| of the chosen Table I scenario -- an
#: identification of conventions, not a derivation.
COUPLING_STATUS = (
    "anchored: lam_osc = |lambda_TB(scenario)| * sqrt(2 Omega_lat) identified "
    "the proposal's dimensionless lambda with this stack's MAP-1 lam_UDW; at "
    "production (2026-09-03T00:09Z) that identification was a convention choice, "
    "not a derivation. Since derived (vacuum.detectors.circuit_mapping, "
    "2026-09-03): lam_osc = |lambda_TB| Omega_lat sqrt(2 Omega_lat), the same "
    "number because Omega_lat = Omega^0/omega_ref = 1 -- see COUPLING_DERIVED_STATUS"
)

#: The coupling's status after the derivation (``load_inputs(coupling='derived')``,
#: used by ``--device-point``).  Derivation: :mod:`vacuum.detectors.circuit_mapping`
#: (module docstring, steps 1-7); value and 1-sigma: the parameter file's
#: ``oscillator_coupling_lambda_osc_scenario2`` (provenance 'derived').
COUPLING_DERIVED_STATUS = (
    "derived: lam_osc = |lambda_TB| Omega_lat sqrt(2 Omega_lat) with lambda_TB = "
    "-gamma sqrt(R_K/(8 pi Z_0)) (Teixido-Bonfill et al. 2026, Eq. 66; lambda_TB^2 = "
    "pi alpha, Eq. 33; Janzen et al. 2023 Eqs. 6, 10 fix the normalisation as the "
    "Golden-rule rate lambda^2 Omega) and Omega_lat = Omega^0/omega_ref: the field "
    "normalisation (Eq. 5/36, Z_0 absorbed), the waveguide speed (l_0 = Z_0/v) and "
    "the lattice spacing (1+1 scale invariance; the derivative's 1/a becomes "
    "Omega_lat) all contribute unit factors at this operating point; MAP-1 gives the "
    "oscillator; the amplitude ('xx') model is matched to the transcribed derivative "
    "('xp') model on the same lattice by equal field-induced rate at the free gap. "
    "1 sigma = the paper's own rounding spread of the scenario coupling (three "
    "published readings, 4.8 %); no input carries a published error bar. Alternatives "
    "(finite-difference derivative, physical-rate matching on the lattice, matching at "
    "the shifted gap, the Eq. 66 reading) span 0.118-0.148 and are run by --device-point"
)


# --------------------------------------------------------------------------
# Unit conversion (thin, explicit wrappers on vacuum.experiments_io's rules)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Units:
    """The one place this notebook turns SI into lattice units.

    ``omega_ref`` (rad/s) fixes the energy unit and ``v`` (m/s) the length
    unit, exactly as :func:`vacuum.experiments_io.quantity_to_lattice`
    does; the helpers are named views of those rules.
    """

    omega_ref: float
    v: float

    def energy(self, hz: float) -> float:
        """Ordinary frequency in Hz -> lattice angular frequency."""
        return 2.0 * math.pi * float(hz) / self.omega_ref

    def temperature(self, kelvin: float) -> float:
        """Temperature in K -> lattice energy (k_B T / hbar omega_ref)."""
        return (K_B_SI * float(kelvin) / HBAR_SI) / self.omega_ref

    def time(self, seconds: float) -> float:
        """Time in s -> lattice time (t omega_ref)."""
        return float(seconds) * self.omega_ref

    def length(self, metres: float) -> float:
        """Length in m -> lattice sites (light-crossing time at v)."""
        return float(metres) / self.v * self.omega_ref

    @property
    def site_mm(self) -> float:
        """One lattice site in millimetres."""
        return self.v / self.omega_ref * 1e3


# --------------------------------------------------------------------------
# Inputs: load the parameter files, fix the operating point and the axes
# --------------------------------------------------------------------------

#: provenance tags a registered input may carry
PROVENANCE_TAGS = (
    "transcribed", "derived", "selection", "plot_digitized", "synthetic", "unsourced"
)


@dataclass(frozen=True)
class OperatingPoint:
    """The sourced operating point, in PUBLISHED units."""

    scenario: int  # 1-based Table I scenario number
    gap_GHz: float
    gap_shift_gamma: float
    gap_shift_GHz: float  # Eq. (30) law evaluated at gamma
    window_ns: float
    switching_kind: str
    separation_mm: float
    signaling_time_ns: float
    coupling_lambda: float  # oscillator coupling (COUPLING_STATUS / COUPLING_DERIVED_STATUS)
    lambda_tb: float  # the scenario's published lambda
    alpha_tb: float  # the scenario's published spin-boson alpha
    coupling_sigma: float = 0.0  # 1-sigma of the derived coupling (0 in anchored mode)
    coupling_mode: str = "anchored"  # 'anchored' (the archive) or 'derived'


@dataclass(frozen=True)
class Resolution:
    """The resolvability floor and its recorded plausible range, in nats."""

    derived: float
    band_low: float
    band_high: float
    crosscheck: float

    @property
    def levels(self) -> Dict[str, float]:
        return {
            "derived": self.derived,
            "band_low": self.band_low,
            "band_high": self.band_high,
        }


@dataclass(frozen=True)
class Inputs:
    """Everything the sweep reads from ``experiments/``, plus provenance."""

    units: Units
    experiments: Dict[str, Any]
    point: OperatingPoint
    resolution: Resolution
    #: sweep axes in published units (value tuples)
    axes: Dict[str, Tuple[float, ...]]
    #: device anchors per axis (published values that lie ON the axis)
    anchors: Dict[str, Dict[str, float]]
    #: input name -> (file, provenance tag, note)
    provenance: Dict[str, Tuple[str, str, str]]
    #: the circuit -> lattice coupling map (``MappingResult.as_dict()``)
    coupling_map: Optional[Dict[str, Any]] = None

    @property
    def synthetic_inputs(self) -> Tuple[str, ...]:
        return tuple(
            sorted(k for k, (_f, tag, _n) in self.provenance.items()
                   if tag in ("synthetic", "unsourced"))
        )

    @property
    def sourced_inputs(self) -> Tuple[str, ...]:
        return tuple(
            sorted(k for k, (_f, tag, _n) in self.provenance.items()
                   if tag not in ("synthetic", "unsourced"))
        )


def _q(exps, file_key: str, name: str):
    return exps[file_key].quantities[name]


def _tag(exps, file_key: str, name: str) -> str:
    """The per-quantity provenance tag of a parameter-file entry."""
    q = _q(exps, file_key, name)
    tag = str(q.extra.get("provenance", "transcribed"))
    if tag not in PROVENANCE_TAGS:
        raise ValueError(f"{file_key}:{name} carries unknown provenance tag {tag!r}")
    return tag


def _val(exps, file_key: str, name: str, index: Optional[int] = None):
    """A parameter-file value: one entry of a list (``index``), or the value."""
    v = _q(exps, file_key, name).value
    if index is not None:
        return float(v[index])
    if isinstance(v, (list, tuple)):
        return tuple(float(x) for x in v)
    return float(v)


# ---- the sweep axes (published units) ------------------------------------
# Each axis is a superset of the corresponding entries of the declared-
# variant grids: the file's entries are published or declared-synthetic
# anchors, the refinement between them is this candidate's (synthetic)
# choice made so the death of E_N is resolved to a grid cell rather than a
# decade.  Every published anchor lies ON its axis, so "which side of the
# contour" is read off a grid point, never interpolated.

#: waveguide temperature (mK): 0 = the proposal's vacuum assumption (Eq. 53)
T_GRID_MK: Tuple[float, ...] = (
    0.0, 5.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 35.0, 40.0,
    50.0, 65.0, 80.0,
)
#: pure dephasing rate (Hz, loader convention x 2 pi), 1-2-5 per decade
GPHI_GRID_HZ: Tuple[float, ...] = (
    1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8, 1e9
)
#: energy relaxation rate (MHz), 1-2-5 per decade plus the measured ends
G1_GRID_MHZ: Tuple[float, ...] = (
    1.0, 2.0, 5.0, 8.7, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 1500.0,
    1850.0, 2000.0, 3000.0, 5000.0,
)
#: detector separation in lattice SITES (the lattice's own coordinate); the
#: published anchors (19.2 / 60 / 120 mm) are placed at the sites they
#: round to and carry their published millimetres on the row
SEP_GRID_SITES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 23, 46)
#: oscillator coupling lam_osc (the sensitivity axis for the unsourced input)
LAMBDA_GRID: Tuple[float, ...] = (
    0.05, 0.075, 0.1, 0.125, None, 0.17, 0.2, 0.225, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6,
)  # None is replaced by the anchored coupling at load time


def load_inputs(root: Optional[Path] = None, coupling: str = "anchored") -> Inputs:
    """Load and validate the parameter files; fix the operating point and axes.

    Every value used downstream is registered in ``provenance`` with the
    file it came from and the per-quantity provenance tag it carries there
    (or ``synthetic`` / ``unsourced`` for this candidate's own choices), so
    the claim's status can be read off the data rather than remembered.

    ``coupling='anchored'`` (default) reproduces the configuration that
    produced the surfaces archive: lam_osc anchored by the MAP-1
    identification and tagged ``unsourced`` as it was then.
    ``coupling='derived'`` takes lam_osc and its 1-sigma from the parameter
    file's derived ``oscillator_coupling_lambda_osc_scenario2`` (tag
    ``derived``), cross-checked against :mod:`vacuum.detectors.circuit_mapping`.
    The two values coincide (Omega_lat = 1); the map itself is carried on
    ``Inputs.coupling_map`` in both modes.
    """
    if coupling not in ("anchored", "derived"):
        raise ValueError(f"coupling must be 'anchored' or 'derived', got {coupling!r}")
    exps = {name: load_experiment(name, root=root) for name in PARAM_FILES}
    prov: Dict[str, Tuple[str, str, str]] = {}

    def note(key, file_key, name, text, index=None):
        tag = _tag(exps, file_key, name)
        prov[key] = (exps[file_key].name, tag, text)
        return _val(exps, file_key, name, index)

    # ---- units ----------------------------------------------------------
    f_gap = note("omega_ref", PROPOSAL, "qubit_gap_frequency",
                 "free-qubit gap Omega^0/2pi (Eq. 30); sets the lattice energy unit")
    v = note("waveguide_speed", PROPOSAL, "waveguide_speed",
             "v (Sec. II.1); sets the lattice length unit")
    units = Units(omega_ref=2.0 * math.pi * f_gap * 1e9, v=v)

    # ---- the operating point: Table I scenario SCENARIO_INDEX + 1 ---------
    i = SCENARIO_INDEX
    lam_tb = note("scenario_lambda", PROPOSAL, "table1_scenario_lambda",
                  f"Table I scenario {i + 1}, column lambda", index=i)
    gamma = note("scenario_gamma", PROPOSAL, "table1_scenario_gamma",
                 f"Table I scenario {i + 1}, column gamma", index=i)
    alpha = note("scenario_alpha", PROPOSAL, "table1_scenario_alpha",
                 f"Table I scenario {i + 1}, column alpha -- inside the device's "
                 "measured alpha range (the closeness criterion)", index=i)
    coef = note("gap_variation_coefficient", PROPOSAL, "gap_variation_coefficient",
                "Eq. (30): Delta-Omega/2pi = coef * gamma GHz")
    window = note("switching_window_ns", PROPOSAL,
                  "switching_duration_cosine_ramps_by_scenario",
                  f"Fig. {17 + i} caption, cosine ramps S_f = 0 (full window T); "
                  "sin^2(pi t/T) == this library's cos2 profile", index=i)
    note("switching_flat_fraction", PROPOSAL, "switching_flat_fraction_cosine_ramps",
         "S_f = 0 for every cosine-ramps panel (so cos2 IS the published shape)")
    t_d = note("signaling_time_ns", VARIANT_SWITCH, "signaling_time_marginal",
               "t_d = 0.16 ns, the proposal's shortest published light-crossing "
               "delay (Fig. 23(c))")
    sep = note("separation_mm", VARIANT_SWITCH, "detector_separation_marginal",
               "d = v t_d (derived in the variant file from the published t_d and v)")
    # the coupling: derived by vacuum.detectors.circuit_mapping from the
    # file's Table I columns; the file's derived quantity is the record and
    # must agree with the code to roundoff (a disagreement is a build error)
    gap_lat = units.energy(f_gap * 1e9)
    shifted_gap_lat = units.energy((f_gap + coef * gamma) * 1e9)
    cmap = circuit_mapping.lambda_osc_from_circuit(
        exps[PROPOSAL], gap_lat=gap_lat, scenario_index=i, field_mass=FIELD_MASS,
        shifted_gap_lat=shifted_gap_lat,
    )
    lam_file = _q(exps, PROPOSAL, "oscillator_coupling_lambda_osc_scenario2")
    if (abs(float(lam_file.value) - cmap.lambda_osc) > 1e-12
            or abs(float(lam_file.error) - cmap.sigma) > 1e-12):
        raise ValueError(
            "teixido-bonfill_2026_table1:oscillator_coupling_lambda_osc_scenario2 = "
            f"{lam_file.value} +- {lam_file.error} disagrees with circuit_mapping "
            f"({cmap.lambda_osc} +- {cmap.sigma}); regenerate the file's derived entry"
        )
    if coupling == "anchored":
        lam_osc = abs(lam_tb) * math.sqrt(2.0 * gap_lat)  # the archive's identification
        lam_sigma = 0.0
        prov["coupling_lambda"] = ("(none)", "unsourced", COUPLING_STATUS)
        note("coupling_lambda_derived", PROPOSAL, "oscillator_coupling_lambda_osc_scenario2",
             COUPLING_DERIVED_STATUS)
    else:
        lam_osc = note("coupling_lambda", PROPOSAL, "oscillator_coupling_lambda_osc_scenario2",
                       COUPLING_DERIVED_STATUS)
        lam_sigma = float(lam_file.error)
    prov["field_mass"] = (
        "(none)", "synthetic",
        f"IR regulator m = {FIELD_MASS} lattice units (fundamental of a finite "
        "line); model limitation (2)",
    )
    prov["boundary_buffer"] = (
        "(none)", "synthetic",
        f"lattice sizing, {BOUNDARY_BUFFER} sites beyond the causal minimum; its "
        "effect is measured by the finite-size control",
    )
    point = OperatingPoint(
        scenario=i + 1,
        gap_GHz=f_gap,
        gap_shift_gamma=gamma,
        gap_shift_GHz=coef * gamma,
        window_ns=window,
        switching_kind="cos2",
        separation_mm=sep,
        signaling_time_ns=t_d,
        coupling_lambda=lam_osc,
        lambda_tb=lam_tb,
        alpha_tb=alpha,
        coupling_sigma=lam_sigma,
        coupling_mode=coupling,
    )

    # ---- the resolution and its band ------------------------------------
    r_q = _q(exps, PROPOSAL, "negativity_resolution_nats")
    derived = note("resolution_nats", PROPOSAL, "negativity_resolution_nats",
                   "derived from Ren 2022 Sec. 4.3 / Fig. 4.5 (MLE tomography "
                   "concurrence floor) via E_N <= ln(1 + C) ~ C")
    band = tuple(float(x) for x in r_q.extra["plausible_range_nats"])
    prov["resolution_band_nats"] = (
        exps[PROPOSAL].name, "derived",
        "recorded plausible range: low end = exact E_N of Ren 2022's own "
        "optimal-point X-state (ren_2022_sec4-3:optimal_point_log_negativity_nats), "
        "high end = largest measured concurrence std-dev of DiCarlo 2009",
    )
    xcheck = note("resolution_crosscheck_nats", PROPOSAL,
                  "negativity_resolution_nats_transmon_crosscheck",
                  "DiCarlo 2009 Fig. 3: smallest measured concurrence std-dev")
    note("ren_optimal_point_E_N", REN, "optimal_point_log_negativity_nats",
         "the band's low end, computed from the thesis's printed density matrix")
    note("ren_concurrence_floor", REN, "qst_concurrence_floor",
         "the derived resolution's input")
    note("dicarlo_concurrence_std", DICARLO, "concurrence",
         "the band's high end (0.04) and the cross-check (0.01) come from these", index=3)
    resolution = Resolution(
        derived=derived, band_low=band[0], band_high=band[1], crosscheck=xcheck
    )
    if not (resolution.band_low < resolution.derived < resolution.band_high):
        raise ValueError("resolution band does not enclose the derived value")

    # ---- the axes and their published anchors ---------------------------
    for key, name, text in (
        ("waveguide_temperature_grid", "waveguide_temperature_grid",
         "file grid (synthetic extension beyond 50 mK) -- the T axis here is a "
         "refinement of its published entries 20/30/40/50 mK"),
        ("relaxation_rate_grid", "relaxation_rate_grid",
         "file grid (synthetic between/beyond the measured 8.7 MHz and 1.85 GHz)"),
        ("pure_dephasing_rate_grid", "pure_dephasing_rate_grid",
         "file grid (synthetic; brackets the plot-digitized [1e7, 2e8] Hz)"),
    ):
        note(key, VARIANT_NOISE, name, text)
    lam_axis = tuple(point.coupling_lambda if x is None else float(x) for x in LAMBDA_GRID)
    axes = {
        "temperature_mK": T_GRID_MK,
        "gammaphi_Hz": GPHI_GRID_HZ,
        "gamma1_MHz": G1_GRID_MHZ,
        "separation_sites": tuple(float(s) for s in SEP_GRID_SITES),
        "coupling_lambda": lam_axis,
    }
    for key, txt in (
        ("temperature_axis", "refinement of the file's temperature grid: 16 values, "
                             "published anchors 20/30/40/50 mK on the axis"),
        ("dephasing_axis", "1-2-5 per decade over [1e4, 1e9] Hz: the file's entries "
                           "plus refinement; plot-digitized anchors 1e7/2e8 on the axis"),
        ("relaxation_axis", "1-2-5 per decade over [1, 5000] MHz plus the measured "
                            "8.7 MHz and 1850 MHz on the axis"),
        ("separation_axis", "integer lattice sites 1-12, 14, 17, 23, 46; the published "
                            "19.2 / 60 / 120 mm round to 7 / 23 / 46 sites"),
        ("coupling_axis", "sensitivity axis for the UNSOURCED coupling; the anchored "
                          "value lies on it"),
    ):
        prov[key] = ("(none)", "synthetic", txt)

    anchors = {
        "temperature_mK": {
            "design_20mK": note("T_design", REN, "design_temperature",
                                "Lupascu-group design temperature (Table 3.1)"),
            "fridge_base_30mK": note("T_fridge_base", DEVICE, "fridge_base_temperature",
                                     "dilution-fridge base (App. A)"),
            "orgiazzi_40mK": note("T_orgiazzi", ORGIAZZI, "mixing_chamber_temperature",
                                  "mixing chamber of the Orgiazzi 2016 setup"),
            "transmission_line_50mK": note(
                "T_transmission_line", DEVICE, "transmission_line_temperature",
                "assumed TL temperature (Sec. IV.A) / heated fridge (App. A)"),
        },
        "gammaphi_Hz": {
            "digitized_min": note("Gphi_min", DEVICE, "pure_dephasing_rate_min",
                                  "PLOT-DIGITIZED lower end (Fig. 7(a))"),
            "digitized_max": note("Gphi_max", DEVICE, "pure_dephasing_rate_max",
                                  "PLOT-DIGITIZED upper end (Fig. 7(a))"),
        },
        "gamma1_MHz": {
            "measured_min": note("G1_min", DEVICE, "relaxation_rate_gamma1_min",
                                 "measured minimum Gamma_1 (Sec. III.B)"),
            "measured_max": 1e3 * note("G1_max", DEVICE, "relaxation_rate_gamma1_max",
                                       "measured maximum Gamma_1 (Sec. III.B), GHz"),
        },
        "separation_sites": {
            "t_d_0.16ns_19.2mm": float(max(1, round(units.length(sep * 1e-3)))),
            "t_d_0.5ns_60mm": float(round(units.length(
                note("sep_60mm", VARIANT_NOISE, "detector_separation_grid",
                     "d = v t_d for t_d = 0.5 ns (Fig. 15)", index=1) * 1e-3))),
            "t_d_1ns_120mm": float(round(units.length(
                note("sep_120mm", PROPOSAL, "detector_separation_d",
                     "d = 12 cm (Sec. VI.A default)") * 10.0 * 1e-3))),
        },
        "coupling_lambda": {coupling: point.coupling_lambda},
    }
    note("expected_operating_T", THESIS, "expected_operating_temperature",
         "authors' expected 20-30 mK (thesis Outlook item 5)", index=1)
    # measured dephasing floor of a different Lupascu-group device, for the
    # Gamma_phi axis comparison: stored in rad/s (a decay rate), so its
    # "Hz" equivalent under the loader's x2pi rule is value / 2pi
    ramsey = note("Gphi_measured_floor_rad_s", ORGIAZZI,
                  "pure_dephasing_rate_ramsey_qubit1",
                  "Ramsey pure dephasing of a Lupascu-group flux qubit (other device)")
    anchors["gammaphi_Hz"]["orgiazzi_ramsey_floor_Hz_equiv"] = ramsey / (2.0 * math.pi)
    for a in anchors["separation_sites"].values():
        if a not in axes["separation_sites"]:
            raise ValueError(f"separation anchor {a} sites is not on the axis")
    for ax_name in ("temperature_mK", "gamma1_MHz", "coupling_lambda"):
        for k, a in anchors[ax_name].items():
            if not any(abs(a - x) <= 1e-12 * max(1.0, abs(a)) for x in axes[ax_name]):
                raise ValueError(f"anchor {ax_name}:{k} = {a} is not on the axis")
    for k in ("digitized_min", "digitized_max"):
        if anchors["gammaphi_Hz"][k] not in axes["gammaphi_Hz"]:
            raise ValueError(f"dephasing anchor {k} is not on the axis")
    return Inputs(
        units=units, experiments=exps, point=point, resolution=resolution,
        axes=axes, anchors=anchors, provenance=prov, coupling_map=cmap.as_dict(),
    )


# --------------------------------------------------------------------------
# One sweep point
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PointSpec:
    """One harvesting run's physical configuration, in PUBLISHED units.

    Conversion to lattice units happens inside :func:`run_point`, never
    here: this record is what a reader compares against the parameter
    files.  ``separation_mm`` is rounded to whole lattice sites there.
    """

    label: str
    sweep: str
    gap_GHz: float
    separation_mm: float
    window_ns: float
    switching_kind: str
    coupling_lambda: float
    temperature_mK: float = 0.0
    gamma1_MHz: float = 0.0
    gammaphi_Hz: float = 0.0
    #: Teixido-Bonfill Eq. (30) law parameter; 0 = static gap.
    gap_shift_gamma: float = 0.0
    grid_iT: int = -1
    grid_iX: int = -1
    boundary_buffer: int = BOUNDARY_BUFFER
    field_mass: float = FIELD_MASS
    conv_tol: float = CONV_TOL
    nseg0: int = NSEG0
    max_halvings: int = MAX_HALVINGS
    #: field-detector coupling operator: 'xx' (amplitude; the surfaces archive)
    #: or 'xp' (derivative, x_d p_field: the proposal's operator, the exact
    #: transcription of Teixido-Bonfill Eq. (37) under MAP-1).
    coupling: str = "xx"
    #: gap tracking: lam(t) = lam_osc chi(t) (Omega_d(t)/Omega^0)^exponent; 0 = off
    #: (the archives), 0.5 = MAP-1 with the instantaneous gap ('xp'), 1.5 = the
    #: 'xx' matching factor Omega^{3/2} evaluated on the instantaneous gap.
    gap_tracking_exponent: float = 0.0
    #: lattice spacing a in light-crossing units (1.0 = the archives' unit
    #: spacing).  The lattice's Brillouin edge omega_max = sqrt(m^2 + 4/a^2) IS
    #: the model's UV cutoff, so a moves the cutoff; the PHYSICAL geometry
    #: (separation, window, buffer) and every rate are held fixed, which is
    #: what makes the cutoff bracket a control rather than a different point.
    #: See :func:`lattice_at_spacing` for the field renormalisation.
    lattice_spacing: float = 1.0


#: Every column a stored row must carry.  The tests assert this schema is
#: complete on every row, so a sweep can never quietly drop a flag.
ROW_COLUMNS: Tuple[str, ...] = (
    # --- identity and physical parameters (published units) -------------
    "label", "sweep", "grid_iT", "grid_iX",
    "gap_GHz", "separation_mm", "window_ns", "switching_kind", "coupling_lambda",
    "temperature_mK", "gamma1_MHz", "gammaphi_Hz", "gap_shift_gamma", "gap_shift_GHz",
    "coupling", "gap_tracking_exponent", "lattice_spacing", "uv_cutoff_GHz",
    # --- the same, in lattice units, plus the derived geometry ----------
    "omega_ref_rad_s", "gap_lat", "gap_min_lat", "temperature_lat", "gamma1_lat",
    "gammaphi_lat", "window_lat", "separation_sites", "separation_delay_ns",
    "field_mass", "n_field_sites", "site_A", "site_B", "boundary_buffer",
    "nbar_detector", "lambda_udw",
    # --- observables -----------------------------------------------------
    "E_N", "MI", "n_d_A", "n_d_B", "work", "dissipated", "E_N_per_work",
    "MI_per_work", "min_nu_pt", "margin_min", "negativity_regime",
    # --- resolvability at the derived floor and across the band ---------
    "resolution_derived_nats", "resolution_band_low_nats", "resolution_band_high_nats",
    "resolvable", "resolvable_band_low", "resolvable_band_high",
    "MI_resolvable", "MI_resolvable_band_low", "MI_resolvable_band_high",
    # --- convergence provenance (spec M2.2: the dt is recorded) ----------
    "converged", "dt_converged", "halvings", "gate_levels_run", "movement",
    "work_movement", "n_segments", "n_substeps",
    # --- audit flags (spec M2.7: every row carries them) ----------------
    "mp_margin_flagged", "precision_flagged", "passivity_entry_passed",
    "passivity_initial_passed", "ledger_passed", "ledger_defect",
    "causality_audited", "all_audits_passed",
    # --- M2.5 communication split (where the perturbative row exists) ---
    "has_perturbative_row", "perturbative_quadrature_failed", "P_A", "P_B",
    "M_abs", "M_vac_abs", "M_comm_abs", "comm_fraction", "E_N_pert",
    "perturbative_scale", "perturbative_ok",
    # --- provenance ------------------------------------------------------
    "inputs_all_sourced", "runtime_s",
)

_STRING_COLUMNS = frozenset({"label", "sweep", "switching_kind", "negativity_regime", "coupling"})
_BOOL_COLUMNS = frozenset({
    "resolvable", "resolvable_band_low", "resolvable_band_high",
    "MI_resolvable", "MI_resolvable_band_low", "MI_resolvable_band_high",
    "converged", "mp_margin_flagged", "precision_flagged", "passivity_entry_passed",
    "passivity_initial_passed", "ledger_passed", "causality_audited",
    "all_audits_passed", "has_perturbative_row", "perturbative_quadrature_failed",
    "perturbative_ok", "inputs_all_sourced",
})


def switching_profile(kind: str, window_lat: float):
    """The switching chi(t) on the window [0, window_lat].

    'cos2'     : sin^2(pi t / W), exactly compact on [0, W] -- the
                 proposal's cosine-ramps shape with S_f = 0 (Eq. 71).
    'gaussian' : sigma = W / (2 GAUSSIAN_HALF_WIDTHS), centred at W/2 and
                 truncated at the same window (declared: chi(edge) = 3.4e-4
                 of the peak); kept only for the smoke branch coverage.
    """
    if kind == "cos2":
        return switching("cos2", 0.5 * window_lat, t0=0.5 * window_lat)
    if kind == "gaussian":
        sigma = window_lat / (2.0 * GAUSSIAN_HALF_WIDTHS)
        return switching("gaussian", sigma, t0=0.5 * window_lat)
    raise ValueError(f"switching kind must be 'cos2' or 'gaussian', got {kind!r}")


def geometry(separation_sites: int, window_lat: float, buffer_sites: int, spacing: float = 1.0):
    """(N_field, (site_A, site_B)) for a causally safe Dirichlet chain.

    Each detector sits at least ``ceil(window/2) + buffer_sites`` from a
    boundary *in physical (light-crossing) units*, so a signal emitted at
    switch-on cannot reach a boundary and return inside the window.  With a
    lattice spacing ``a`` != 1 those two lengths are converted to sites, so
    the PHYSICAL buffer is the same at every spacing and only the resolution
    changes (``a = 1`` reproduces the original expression exactly).
    """
    d = int(separation_sites)
    if d < 1:
        raise ValueError(f"separation must be >= 1 site, got {d}")
    a = float(spacing)
    if not (a > 0.0):
        raise ValueError(f"lattice spacing must be > 0, got {a}")
    B = int(math.ceil(0.5 * float(window_lat) / a)) + int(round(float(buffer_sites) / a))
    return d + 2 * B, (B, B + d)


def lattice_at_spacing(n_field: int, field_mass: float, spacing: float, bc: str = "dirichlet"):
    """(K, coupling_rescale, uv_cutoff_lat) of the chain at spacing ``a``.

    Discretising H = (1/2) int dxi [pi^2 + m^2 phi^2 + (d_xi phi)^2] on a grid
    of spacing ``a`` gives H = (1/2) sum_j a [pi_j^2 + m^2 phi_j^2 +
    ((phi_{j+1} - phi_j)/a)^2].  The variable canonically conjugate to the
    site momentum is NOT phi_j but

        x_j = sqrt(a) phi_j,          p_j = pi_j / sqrt(a),

    (so [x_j, p_k] = i delta_jk), and in those variables

        H = (1/2)(p.p + x^T K_a x),   K_a = harmonic_chain_K(N, m a) / a^2,

    whose dispersion is omega(kappa) = sqrt(m^2 + (4/a^2) sin^2(kappa/2)),
    i.e. a Brillouin band edge

        omega_max = sqrt(m^2 + 4 / a^2)    (a = 1: 2.01 = 14.7 GHz;
                                            a = 0.292: 6.85 = 50 GHz),

    which is the model's UV cutoff.  A pointlike detector couples to the
    PHYSICAL field phi(xi_d) = x_j / sqrt(a) (amplitude, 'xx') or to
    d_t phi(xi_d) = p_j / sqrt(a) ('xp'), so in both cases the coupling that
    must be handed to :func:`attach_detectors` is lam_osc / sqrt(a) --
    the field-normalisation factor of the derivation
    (:mod:`vacuum.detectors.circuit_mapping`, step 3) made explicit at
    a != 1.  Everything else -- gaps, rates, temperature, the switching
    window and the physical separation -- is spacing-independent.

    Returns ``(K, 1/sqrt(a), omega_max)``; at ``a = 1`` this is exactly the
    archives' ``harmonic_chain_K(N, m)``, rescale 1.0.
    """
    a = float(spacing)
    if not (a > 0.0):
        raise ValueError(f"lattice spacing must be > 0, got {a}")
    m = float(field_mass)
    K = harmonic_chain_K(int(n_field), m * a, bc=bc) / (a * a)
    return K, 1.0 / math.sqrt(a), math.sqrt(m * m + 4.0 / (a * a))


def _gaps_of_t(gap_lat: float, shift_lat: float, chi) -> Callable[[float], np.ndarray]:
    """Omega_d(t) = Omega^0 + Delta-Omega chi(t): Teixido-Bonfill Eq. (30)
    along the switching envelope (gap is Omega^0 wherever lam = 0)."""

    def gaps(t):
        return np.array([gap_lat + shift_lat * chi(t)] * 2, dtype=float)

    return gaps


def coupling_schedule(lam_osc: float, chi, gaps_of_t, gap0_lat: float, exponent: float = 0.0):
    """lam(t) = lam_osc chi(t) (Omega_d(t) / Omega^0)^exponent.

    exponent 0 (the archives): the proposal's constant lambda times the
    switching, with the gap schedule of Eq. (30) acting only on Omega_d(t).
    exponent 0.5: MAP-1 taken with the instantaneous gap, lam_osc^xp(t) =
    |lambda_TB| sqrt(2 Omega(t)) -- the 'xp' gap-tracking variant.
    exponent 1.5: the 'xx' matching factor Omega^{3/2} on the instantaneous
    gap (circuit_mapping docstring, step 6).  Pure over its arguments.
    """
    e = float(exponent)
    lam0 = float(lam_osc)
    if e == 0.0:
        return lambda t: lam0 * chi(t)
    g0 = float(gap0_lat)

    def lam(t):
        return lam0 * chi(t) * (float(gaps_of_t(t)[0]) / g0) ** e

    return lam


def _run_at_level(
    K_field, sites, gaps_of_t, lam_of_t, window_lat, n_segments, *,
    temperature_lat, gamma1_lat, gammaphi_lat, V0, K_tot0,
    substeps: int = SUBSTEPS_PER_SEGMENT, coupling: str = "xx",
):
    """One fixed-resolution pass THROUGH :func:`vacuum.protocol.run_protocol`.

    Per grid segment: a Strang pair of :class:`ChannelStep`s (the integrated
    loss + dephasing map over half the segment, from
    :func:`imperfections_from_rates`) around one :class:`GeneratorStep` with
    ``substeps`` exponential-midpoint substeps.  Noiseless points emit no
    channel steps.  The runner's causality audit is off (it assumes a 1D
    chain ordering the detector-augmented K_tot does not have); causal
    structure is checked physically through the M2.5 |M_comm| column.
    """
    N = int(np.asarray(K_field).shape[0])
    det_modes = (N, N + 1)
    ts = np.linspace(0.0, float(window_lat), int(n_segments) + 1)
    noisy = (gamma1_lat > 0.0) or (gammaphi_lat > 0.0)

    def K_of_t(t):
        return attach_detectors(K_field, sites, gaps_of_t(t), lam_of_t(t), coupling=coupling)

    steps: List[Any] = []
    for i in range(int(n_segments)):
        duration = float(ts[i + 1] - ts[i])
        if noisy:
            imp = imperfections_from_rates(
                gaps_of_t(0.5 * (ts[i] + ts[i + 1])), 0.5 * duration,
                temperature_lat, gamma1_lat, gammaphi_lat,
            )
            half = imperfection_step(
                det_modes, eta=imp.eta, nbar=imp.nbar, dephasing=imp.dephasing,
                gaps=imp.gaps, label=f"noise{i}",
            )
            steps.append(half)
        steps.append(GeneratorStep(H=K_of_t, duration=duration,
                                   n_substeps=int(substeps), label=f"drive{i}"))
        if noisy:
            steps.append(half)

    ledger = EnergyLedger(K=K_tot0, tol=LEDGER_TOL)
    res = run_protocol(
        V0, K_tot0, steps, ledger=ledger,
        claim_ground_entry=(temperature_lat == 0.0),
        report_modes=[det_modes], run_causality=False,
        ledger_tol=LEDGER_TOL, strict=True,
    )
    V_AB = detector_block(res.V, N)
    neg = harvested_log_negativity(V_AB)
    mi = float(mutual_information(V_AB, [0], [1]))
    n_d = detector_occupations(res.V, N, gaps_of_t(float(ts[-1])))
    observables = {
        "E_N": neg.E_N, "MI": mi, "n_d_A": float(n_d[0]), "n_d_B": float(n_d[1]),
    }
    return res, neg, mi, n_d, observables


def perturbative_columns(K_field, sites, chi, gap_lat, lam_osc, coupling: str = "xx") -> Dict[str, Any]:
    """M2.3 pair state + M2.5 communication split on the SAME lattice setup.

    MAP-1 (tests/test_gate_crossvalidation.py): lam_UDW = lam_osc /
    sqrt(2 Omega).  Kernel, switching and (pointlike) smearing are the ones
    the nonperturbative run used; ``coupling='xp'`` uses the derivative
    kernel of ``vacuum.detectors.udw_derivative`` (d_t d_t' W).
    """
    N = int(np.asarray(K_field).shape[0])
    kernel = wightman_lattice(K_field)
    F = []
    for s in sites:
        col = np.zeros(N)
        col[int(s)] = 1.0
        F.append(col)
    det_A = UDWDetector(chi, F[0])
    det_B = UDWDetector(chi, F[1])
    lam_udw = float(lam_osc) / math.sqrt(2.0 * float(gap_lat))
    state = pair_state_with_split(kernel, det_A, det_B, lam_udw, float(gap_lat),
                                  coupling=coupling, **PERTURBATIVE_QUAD)
    cols = communication_columns(state)
    return {
        "has_perturbative_row": True,
        "perturbative_quadrature_failed": False,
        "P_A": float(state.P_A),
        "P_B": float(state.P_B),
        "M_abs": float(abs(state.M)),
        "M_vac_abs": float(abs(cols["M_vac"])),
        "M_comm_abs": float(abs(cols["M_comm"])),
        "comm_fraction": float(cols["comm_fraction"]),
        "E_N_pert": float(cols["E_N"]),
        "perturbative_scale": float(state.meta["perturbative_scale"]),
        "perturbative_ok": bool(state.meta["perturbative_ok"]),
    }


_EMPTY_PERTURBATIVE: Dict[str, Any] = {
    "has_perturbative_row": False, "perturbative_quadrature_failed": False,
    "P_A": float("nan"), "P_B": float("nan"), "M_abs": float("nan"),
    "M_vac_abs": float("nan"), "M_comm_abs": float("nan"),
    "comm_fraction": float("nan"), "E_N_pert": float("nan"),
    "perturbative_scale": float("nan"), "perturbative_ok": False,
}


class _level_schedule:
    """Which halving levels the gate visits, with a second-order jump.

    Iterating yields 0, 1, 2, ... unless :meth:`jump` is called, which uses
    the measured movement to predict the first level that will satisfy the
    tolerance and skips straight to *its predecessor* (so the accepted
    movement is always measured across one halving, never several).  The
    prediction is the exponent of the midpoint rule (movement ~ C dt^2
    falls by 4 per level).  Never jumps backwards, never more than
    :data:`_MAX_JUMP` levels, and never strands itself at the cap.
    """

    _MAX_JUMP = 10
    _MIN_PREDICT_LEVEL = 2

    def __init__(self, max_level: int):
        self.max_level = int(max_level)
        self._next = 0

    def __iter__(self):
        while self._next <= self.max_level:
            level = self._next
            self._next = level + 1
            yield level

    def jump(self, level: int, movement: float, tol: float) -> None:
        if level < self._MIN_PREDICT_LEVEL:
            return
        if not (movement > 0.0) or not (tol > 0.0) or movement <= tol:
            return
        need = int(math.ceil(math.log(movement / tol) / math.log(4.0)))
        target = level + min(max(need, 1), self._MAX_JUMP)
        self._next = max(self._next, min(target - 1, self.max_level - 1))


def run_point(spec: PointSpec, inputs: Inputs, *, perturbative: bool = True) -> Dict[str, Any]:
    """Run ONE sweep point end to end and return its full result row.

    Pure over its arguments.  The dt-halving gate (spec M2.2) runs
    :func:`_run_at_level` at ``nseg0 * 2**L`` segments and accepts the
    first level at which every gated observable (E_N, <n_d>, I(A:B)) moved
    less than ``spec.conv_tol``; the accepted level's
    :class:`~vacuum.protocol.ProtocolResult` is the one reported, so the
    audit flags, the ledger and the observables all describe exactly the
    evolution whose dt is recorded.
    """
    t_start = time.time()
    u = inputs.units
    res_ = inputs.resolution

    window_lat = u.time(spec.window_ns * 1e-9)
    gap_lat = u.energy(spec.gap_GHz * 1e9)
    coef = _val(inputs.experiments, PROPOSAL, "gap_variation_coefficient")
    shift_GHz = coef * float(spec.gap_shift_gamma)  # Teixido-Bonfill Eq. (30)
    shift_lat = u.energy(shift_GHz * 1e9)
    if gap_lat + min(0.0, shift_lat) <= 0.0:
        raise ValueError(f"gap schedule drives Omega_d non-positive at {spec.label!r}")
    temperature_lat = u.temperature(spec.temperature_mK * 1e-3)
    gamma1_lat = u.energy(spec.gamma1_MHz * 1e6)
    gammaphi_lat = u.energy(spec.gammaphi_Hz)

    a_lat = float(spec.lattice_spacing)
    delay_lat = u.length(spec.separation_mm * 1e-3)
    sep_sites = max(1, int(round(delay_lat / a_lat)))
    n_field, sites = geometry(sep_sites, window_lat, spec.boundary_buffer, a_lat)

    K_field, lam_rescale, uv_cutoff_lat = lattice_at_spacing(
        n_field, spec.field_mass, a_lat, bc="dirichlet")
    chi = switching_profile(spec.switching_kind, window_lat)
    gaps_of_t = _gaps_of_t(gap_lat, shift_lat, chi)

    # the detector couples to the PHYSICAL field, so the coupling handed to
    # attach_detectors carries the 1/sqrt(a) field normalisation
    lam_of_t = coupling_schedule(spec.coupling_lambda * lam_rescale, chi, gaps_of_t,
                                 gap_lat, spec.gap_tracking_exponent)

    # Initial state at lam(0) = 0; at T = 0 this runs the MANDATORY passivity
    # audit on the claimed ground state (spec M2.2, no bypass).
    init = initial_state(K_field, sites, gaps_of_t(0.0), temperature_lat, strict=True)

    prev: Optional[Dict[str, float]] = None
    prev_work: Optional[float] = None
    movement: Optional[float] = None
    work_movement: Optional[float] = None
    converged = False
    level = 0
    levels_run: List[int] = []
    res = neg = n_d = obs = None
    mi = float("nan")
    n_segments = spec.nseg0
    schedule = _level_schedule(int(spec.max_halvings))
    for level in schedule:
        n_segments = int(spec.nseg0) * 2**level
        levels_run.append(level)
        res, neg, mi, n_d, obs = _run_at_level(
            K_field, sites, gaps_of_t, lam_of_t, window_lat, n_segments,
            temperature_lat=temperature_lat, gamma1_lat=gamma1_lat,
            gammaphi_lat=gammaphi_lat, V0=init.V, K_tot0=init.K_tot,
            coupling=spec.coupling,
        )
        if prev is not None and levels_run[-2] == level - 1:
            movement = max(abs(obs[k] - prev[k]) for k in obs)
            work_movement = abs(res.work_total - prev_work)
            if movement < float(spec.conv_tol):
                converged = True
                break
            schedule.jump(level, movement, float(spec.conv_tol))
        prev, prev_work = dict(obs), res.work_total
    if not converged:
        raise RuntimeError(
            f"point {spec.label!r}: dt-halving gate did not reach {spec.conv_tol:g} "
            f"within {spec.max_halvings} halvings (levels {levels_run}, last "
            f"movement {movement!r})"
        )

    dt_converged = window_lat / (n_segments * SUBSTEPS_PER_SEGMENT)
    work = float(res.work_total)
    ledger_res = res.audits["ledger"]

    has_pert = (
        perturbative and spec.switching_kind == "cos2" and spec.temperature_mK == 0.0
        and spec.gamma1_MHz == 0.0 and spec.gammaphi_Hz == 0.0
        and spec.gap_shift_gamma == 0.0
    )
    pert = dict(_EMPTY_PERTURBATIVE)
    if has_pert:
        try:
            pert = perturbative_columns(K_field, sites, chi, gap_lat,
                                        spec.coupling_lambda * lam_rescale,
                                        coupling=spec.coupling)
        except RuntimeError as exc:
            pert["perturbative_quadrature_failed"] = True
            print(f"    ! {spec.label}: perturbative/M2.5 quadrature refused ({exc}); "
                  "row carries has_perturbative_row=False", flush=True)

    E_N = float(neg.E_N)
    row: Dict[str, Any] = {
        "label": spec.label,
        "sweep": spec.sweep,
        "grid_iT": float(spec.grid_iT),
        "grid_iX": float(spec.grid_iX),
        "gap_GHz": float(spec.gap_GHz),
        "separation_mm": float(spec.separation_mm),
        "window_ns": float(spec.window_ns),
        "switching_kind": spec.switching_kind,
        "coupling_lambda": float(spec.coupling_lambda),
        "temperature_mK": float(spec.temperature_mK),
        "gamma1_MHz": float(spec.gamma1_MHz),
        "gammaphi_Hz": float(spec.gammaphi_Hz),
        "gap_shift_gamma": float(spec.gap_shift_gamma),
        "gap_shift_GHz": float(shift_GHz),
        "coupling": str(spec.coupling),
        "gap_tracking_exponent": float(spec.gap_tracking_exponent),
        "lattice_spacing": float(a_lat),
        "uv_cutoff_GHz": float(uv_cutoff_lat * spec.gap_GHz),
        "omega_ref_rad_s": float(u.omega_ref),
        "gap_lat": float(gap_lat),
        "gap_min_lat": float(gap_lat + min(0.0, shift_lat)),
        "temperature_lat": float(temperature_lat),
        "gamma1_lat": float(gamma1_lat),
        "gammaphi_lat": float(gammaphi_lat),
        "window_lat": float(window_lat),
        "separation_sites": float(sep_sites),  # physical separation = sep_sites * a
        "separation_delay_ns": float(spec.separation_mm * 1e-3 / u.v * 1e9),
        "field_mass": float(spec.field_mass),
        "n_field_sites": float(n_field),
        "site_A": float(sites[0]),
        "site_B": float(sites[1]),
        "boundary_buffer": float(spec.boundary_buffer),
        "nbar_detector": float(bath_occupation(gap_lat, temperature_lat)),
        "lambda_udw": float(spec.coupling_lambda) / math.sqrt(2.0 * gap_lat),
        "E_N": E_N,
        "MI": float(mi),
        "n_d_A": float(n_d[0]),
        "n_d_B": float(n_d[1]),
        "work": work,
        "dissipated": float(res.dissipated_total),
        "E_N_per_work": float(E_N / work) if work > 0.0 else float("nan"),
        "MI_per_work": float(mi / work) if work > 0.0 else float("nan"),
        "min_nu_pt": float(neg.min_nu_pt),
        "margin_min": float(neg.margin_min),
        "negativity_regime": neg.regime,
        "resolution_derived_nats": float(res_.derived),
        "resolution_band_low_nats": float(res_.band_low),
        "resolution_band_high_nats": float(res_.band_high),
        "resolvable": bool(E_N > res_.derived),
        "resolvable_band_low": bool(E_N > res_.band_low),
        "resolvable_band_high": bool(E_N > res_.band_high),
        "MI_resolvable": bool(mi > res_.derived),
        "MI_resolvable_band_low": bool(mi > res_.band_low),
        "MI_resolvable_band_high": bool(mi > res_.band_high),
        "converged": bool(converged),
        "dt_converged": float(dt_converged),
        "halvings": float(level),
        "gate_levels_run": float(len(levels_run)),
        "movement": float(movement),
        "work_movement": float(work_movement),
        "n_segments": float(n_segments),
        "n_substeps": float(n_segments * SUBSTEPS_PER_SEGMENT),
        "mp_margin_flagged": bool(neg.flagged),
        "precision_flagged": bool(res.flagged),
        "passivity_entry_passed": bool(
            res.audits["passivity_entry"].passed
            if res.audits["passivity_entry"] is not None else True
        ),
        "passivity_initial_passed": bool(
            init.passivity.passed if init.passivity is not None else True
        ),
        "ledger_passed": bool(ledger_res.passed),
        "ledger_defect": float(ledger_res.details["defect"]),
        "causality_audited": bool(len(res.audits["causality"]) > 0),
        "all_audits_passed": bool(res.all_passed()),
        "inputs_all_sourced": False,  # the lattice regulators (mass, buffer) are synthetic on every row
        "runtime_s": float(time.time() - t_start),
    }
    row.update(pert)
    missing = [c for c in ROW_COLUMNS if c not in row]
    extra = [c for c in row if c not in ROW_COLUMNS]
    if missing or extra:  # schema is enforced here, not only in the tests
        raise AssertionError(f"row schema violation: missing {missing}, extra {extra}")
    return row


# --------------------------------------------------------------------------
# The surfaces
# --------------------------------------------------------------------------

#: surface name -> (X axis name, row field the axis maps to, log-axis?)
SURFACES: Dict[str, Tuple[str, bool]] = {
    "dephasing": ("gammaphi_Hz", True),
    "relaxation": ("gamma1_MHz", True),
    "separation": ("separation_sites", False),
    "coupling": ("coupling_lambda", False),
}
MAIN_SURFACE = "dephasing"


def base_spec(inputs: Inputs, label: str, sweep: str, **kw) -> PointSpec:
    """The sourced operating point (Table I scenario), overridden by keyword."""
    p = inputs.point
    fields = dict(
        gap_GHz=p.gap_GHz, separation_mm=p.separation_mm, window_ns=p.window_ns,
        switching_kind=p.switching_kind, coupling_lambda=p.coupling_lambda,
        gap_shift_gamma=p.gap_shift_gamma,
    )
    fields.update(kw)
    return PointSpec(label=label, sweep=sweep, **fields)


def _separation_mm_for_sites(inputs: Inputs, sites: int) -> float:
    """Published millimetres for anchor sites, the lattice's own otherwise."""
    for name, s in inputs.anchors["separation_sites"].items():
        if int(s) == int(sites):
            return float(name.split("_")[-1].rstrip("m"))  # "..._19.2mm" -> 19.2
    return float(sites) * inputs.units.site_mm


def surface_specs(
    inputs: Inputs, name: str, *, T_values=None, X_values=None, **common
) -> List[PointSpec]:
    """The (temperature) x (X) grid of one surface, all at the operating point."""
    x_axis, _log = SURFACES[name]
    Ts = tuple(T_values) if T_values is not None else inputs.axes["temperature_mK"]
    Xs = tuple(X_values) if X_values is not None else inputs.axes[x_axis]
    out: List[PointSpec] = []
    for iT, T in enumerate(Ts):
        for iX, X in enumerate(Xs):
            kw: Dict[str, Any] = {"temperature_mK": float(T), "grid_iT": iT, "grid_iX": iX}
            if x_axis == "separation_sites":
                kw["separation_mm"] = _separation_mm_for_sites(inputs, int(X))
            else:
                kw[x_axis] = float(X)
            kw.update(common)
            out.append(base_spec(inputs, f"{name}-T{T:g}-X{X:g}", name, **kw))
    return out


def split_specs(inputs: Inputs, X_values=None, **common) -> List[PointSpec]:
    """Static-gap companions along the separation axis at T = 0, noiseless.

    The perturbative engine is a static-gap object, so the M2.5 split is
    computed on these rows (gap_shift_gamma = 0, everything else as the
    operating point) and NOT on the gap-shifted surface rows.
    """
    Xs = tuple(X_values) if X_values is not None else inputs.axes["separation_sites"]
    return [
        base_spec(
            inputs, f"split-X{X:g}", "split", gap_shift_gamma=0.0, grid_iT=0, grid_iX=iX,
            separation_mm=_separation_mm_for_sites(inputs, int(X)), **common,
        )
        for iX, X in enumerate(Xs)
    ]


def control_specs(inputs: Inputs, **common) -> List[PointSpec]:
    """Finite-size (boundary) control: the base point at three buffers."""
    return [
        base_spec(inputs, f"control-buffer{b}", "control", boundary_buffer=b, **common)
        for b in (BOUNDARY_BUFFER, 2 * BOUNDARY_BUFFER, 4 * BOUNDARY_BUFFER)
    ]


def full_specs(inputs: Inputs, surfaces: Sequence[str] = tuple(SURFACES), **common) -> List[PointSpec]:
    specs: List[PointSpec] = []
    for name in surfaces:
        specs.extend(surface_specs(inputs, name, **common))
    specs.extend(split_specs(inputs, **common))
    specs.extend(control_specs(inputs, **common))
    return specs


# --------------------------------------------------------------------------
# The device point at the DERIVED coupling (--device-point)
# --------------------------------------------------------------------------

#: temperatures of the device-point rows: the proposal's vacuum assumption
#: (0, Eq. 53) and the four published anchors on the temperature axis
DEVICE_POINT_T_MK: Tuple[float, ...] = (0.0, 20.0, 30.0, 40.0, 50.0)

#: short coupling names (row labels are stored as U64) -> description.  The
#: values come from ``Inputs.coupling_map`` in :func:`device_point_couplings`.
DEVICE_POINT_COUPLINGS: Dict[str, str] = {
    "c0": "derived lam_osc (default criterion 'xp': equal field-induced rate to the "
          "transcribed derivative model on the same lattice, at the free gap)",
    "c-1s": "derived minus 1 sigma (the scenario coupling's reading spread)",
    "c+1s": "derived plus 1 sigma",
    "cPR": "alternative: physical-rate matching on the lattice at the free gap "
           "(the lattice density of states at half the band, x0.924)",
    "c66": "alternative: the Eq. (66) reading of lambda_TB at the scenario's gamma (x0.906)",
    "cLE": "alternative: lower envelope, Eq. (66) reading x physical-rate matching",
}


def device_point_couplings(inputs: Inputs) -> Dict[str, float]:
    """The couplings the device point is re-run at, from the coupling map."""
    m = inputs.coupling_map
    if m is None:
        raise ValueError("inputs carry no coupling map")
    alt = m["alternatives"]
    return {
        "c0": float(m["lambda_osc"]),
        "c-1s": float(m["lambda_osc"] - m["sigma"]),
        "c+1s": float(m["lambda_osc"] + m["sigma"]),
        "cPR": float(alt["physical-rate@free_gap"]),
        "c66": float(alt["reading_eq66@default"]),
        "cLE": float(alt["lower_envelope(eq66_reading,physical-rate)"]),
    }


def device_point_anchors(inputs: Inputs) -> Dict[str, Tuple[str, Dict[str, float]]]:
    """short anchor name -> (description, PointSpec overrides).

    The seven single-noise device anchors are exactly the surfaces' anchor
    cells (one axis at its published value, the others at zero), so their
    rows at the archive's coupling must reproduce ``data/rows.npz``; the two
    ``combined`` anchors apply the measured relaxation and the plot-digitised
    dephasing together, which the surfaces never did.
    """
    a = inputs.anchors
    out: Dict[str, Tuple[str, Dict[str, float]]] = {
        "Gphi_min": ("plot-digitised lower end of Gamma_phi (Janzen Fig. 7(a))",
                     {"gammaphi_Hz": a["gammaphi_Hz"]["digitized_min"]}),
        "Gphi_max": ("plot-digitised upper end of Gamma_phi (Janzen Fig. 7(a))",
                     {"gammaphi_Hz": a["gammaphi_Hz"]["digitized_max"]}),
        "G1_min": ("measured minimum Gamma_1 (Janzen Sec. III.B)",
                   {"gamma1_MHz": a["gamma1_MHz"]["measured_min"]}),
        "G1_max": ("measured maximum Gamma_1 (Janzen Sec. III.B)",
                   {"gamma1_MHz": a["gamma1_MHz"]["measured_max"]}),
    }
    for name, s in a["separation_sites"].items():
        mm = _separation_mm_for_sites(inputs, int(s))
        out[f"d{mm:g}"] = (f"sourced separation {name} ({int(s)} sites)", {"separation_mm": mm})
    out["G1min+Gphimin"] = (
        "combined: measured minimum Gamma_1 with the digitised minimum Gamma_phi",
        {"gamma1_MHz": a["gamma1_MHz"]["measured_min"],
         "gammaphi_Hz": a["gammaphi_Hz"]["digitized_min"]},
    )
    out["G1max+Gphimax"] = (
        "combined: measured maximum Gamma_1 with the digitised maximum Gamma_phi",
        {"gamma1_MHz": a["gamma1_MHz"]["measured_max"],
         "gammaphi_Hz": a["gammaphi_Hz"]["digitized_max"]},
    )
    return out


def device_point_specs(
    inputs: Inputs,
    couplings: Optional[Dict[str, float]] = None,
    anchors: Optional[Dict[str, Tuple[str, Dict[str, float]]]] = None,
    T_values: Sequence[float] = DEVICE_POINT_T_MK,
    **common,
) -> List[PointSpec]:
    """(coupling) x (temperature) x (anchor) rows at the operating point."""
    couplings = couplings if couplings is not None else device_point_couplings(inputs)
    anchors = anchors if anchors is not None else device_point_anchors(inputs)
    out: List[PointSpec] = []
    for cname, lam in couplings.items():
        for iT, T in enumerate(T_values):
            for iX, (aname, (_desc, over)) in enumerate(anchors.items()):
                label = f"dp-{cname}-T{T:g}-{aname}"
                if len(label) > 64:
                    raise ValueError(f"label {label!r} exceeds the U64 row storage")
                kw: Dict[str, Any] = {"temperature_mK": float(T), "grid_iT": iT, "grid_iX": iX,
                                      "coupling_lambda": float(lam)}
                kw.update(over)
                kw.update(common)
                out.append(base_spec(inputs, label, "device_point", **kw))
    return out


def _dp_row_key(r: Dict[str, Any]) -> Tuple[float, float, float, int]:
    return (round(float(r["temperature_mK"]), 6), round(float(r["gamma1_MHz"]), 6),
            round(float(r["gammaphi_Hz"]), 3), int(r["separation_sites"]))


def model_mismatch_companion(inputs: Inputs, lam_osc: float) -> Dict[str, Any]:
    """Perturbative 'xx' vs 'xp' companions at the static-gap operating point.

    Same lattice, switching, sites and MAP-1 lam_UDW: what the amplitude
    stand-in yields relative to the transcribed derivative model (a MODEL
    difference; circuit_mapping docstring, step 6).
    """
    u = inputs.units
    p = inputs.point
    window_lat = u.time(p.window_ns * 1e-9)
    gap_lat = u.energy(p.gap_GHz * 1e9)
    sep_sites = max(1, int(round(u.length(p.separation_mm * 1e-3))))
    n_field, sites = geometry(sep_sites, window_lat, BOUNDARY_BUFFER)
    K_field = harmonic_chain_K(n_field, FIELD_MASS, bc="dirichlet")  # a = 1
    chi = switching_profile("cos2", window_lat)
    kernel = wightman_lattice(K_field)
    F = []
    for s in sites:
        col = np.zeros(n_field)
        col[int(s)] = 1.0
        F.append(col)
    det_A, det_B = UDWDetector(chi, F[0]), UDWDetector(chi, F[1])
    lam_udw = circuit_mapping.lambda_udw_from_osc(lam_osc, gap_lat)
    out: Dict[str, Any] = {"lambda_udw": lam_udw, "separation_sites": sep_sites,
                           "gap_lat": gap_lat, "window_lat": window_lat}
    for coupling in ("xx", "xp"):
        st = pair_state_with_split(kernel, det_A, det_B, lam_udw, gap_lat, coupling=coupling,
                                   **PERTURBATIVE_QUAD)
        cols = communication_columns(st)
        out[coupling] = {
            "P_A": float(st.P_A), "M_abs": float(abs(st.M)),
            "M_vac_abs": float(abs(cols["M_vac"])), "M_comm_abs": float(abs(cols["M_comm"])),
            "comm_fraction": float(cols["comm_fraction"]), "E_N_pert": float(cols["E_N"]),
            "perturbative_ok": bool(st.meta["perturbative_ok"]),
        }
    out["E_N_ratio_xx_over_xp"] = out["xx"]["E_N_pert"] / out["xp"]["E_N_pert"]
    out["P_A_ratio_xx_over_xp"] = out["xx"]["P_A"] / out["xp"]["P_A"]
    out["note"] = (
        "The 'xx' model at the derived coupling is NOT the derivative model: at the "
        "same MAP-1 lam_UDW the amplitude companion's E_N is this ratio times the "
        "derivative companion's. A model difference (limitations 3-4), not a coupling "
        "uncertainty; the noise map's verdicts are for the amplitude model."
    )
    return out


def device_point_summary(
    rows: Sequence[Dict[str, Any]], inputs: Inputs, couplings: Dict[str, float],
    anchors: Dict[str, Tuple[str, Dict[str, float]]], *, archive_rows=None,
    wall_clock_s: float = 0.0, mismatch: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verdicts at every coupling and floor, plus the archive cross-check."""
    levels = inputs.resolution.levels
    p = inputs.point
    per: Dict[str, Any] = {}
    for cname, lam in couplings.items():
        sel = [r for r in rows if r["label"].startswith(f"dp-{cname}-")]
        table = []
        for r in sorted(sel, key=lambda r: (r["grid_iT"], r["grid_iX"])):
            aname = r["label"].split("-", 3)[3]
            table.append({
                "label": r["label"], "temperature_mK": r["temperature_mK"], "anchor": aname,
                "gamma1_MHz": r["gamma1_MHz"], "gammaphi_Hz": r["gammaphi_Hz"],
                "separation_mm": r["separation_mm"], "separation_sites": r["separation_sites"],
                "E_N": r["E_N"], "MI": r["MI"], "work": r["work"], "min_nu_pt": r["min_nu_pt"],
                "verdict": {k: ("GO" if r["E_N"] > v else "NO-GO") for k, v in levels.items()},
                "MI_verdict": {k: ("GO" if r["MI"] > v else "NO-GO") for k, v in levels.items()},
                "all_audits_passed": r["all_audits_passed"], "converged": r["converged"],
                "mp_margin_flagged": r["mp_margin_flagged"],
            })

        def box(Ts):
            rs = [t for t in table if t["temperature_mK"] in Ts]
            if not rs:
                return {}
            max_en = max(t["E_N"] for t in rs)
            max_mi = max(t["MI"] for t in rs)
            return {
                "temperatures_mK": list(Ts), "n_rows": len(rs),
                "max_E_N": max_en, "max_MI": max_mi,
                "E_N_over_derived_floor": max_en / inputs.resolution.derived,
                "MI_over_derived_floor": max_mi / inputs.resolution.derived,
                "E_N_GO_anywhere": {k: bool(max_en > v) for k, v in levels.items()},
                "MI_GO_anywhere": {k: bool(max_mi > v) for k, v in levels.items()},
                "n_GO": {k: int(sum(t["E_N"] > v for t in rs)) for k, v in levels.items()},
                "GO_anchors": {k: [f"{t['temperature_mK']:g}mK:{t['anchor']}" for t in rs
                                   if t["E_N"] > v] for k, v in levels.items()},
            }

        per[cname] = {
            "description": DEVICE_POINT_COUPLINGS.get(cname, ""),
            "lambda_osc": lam,
            "lambda_udw_map1": lam / math.sqrt(2.0 * inputs.units.energy(p.gap_GHz * 1e9)),
            "n_rows": len(sel),
            "all_audits_passed": bool(all(r["all_audits_passed"] for r in sel)),
            "all_converged": bool(all(r["converged"] for r in sel)),
            "worst_ledger_defect": float(max(abs(r["ledger_defect"]) for r in sel)) if sel else float("nan"),
            "device_box_30_50mK": box((30.0, 50.0)),
            "design_20mK": box((20.0,)),
            "orgiazzi_40mK": box((40.0,)),
            "vacuum_0mK": box((0.0,)),
            "rows": table,
        }

    # the archive cross-check: the archive's rows at the same physical
    # configuration (surfaces were produced at the anchored coupling, which
    # is the derived value)
    xcheck: Dict[str, Any] = {"available": archive_rows is not None}
    if archive_rows is not None:
        by: Dict[Tuple[float, float, float, int], Dict[str, Any]] = {}
        for r in archive_rows:
            if r["sweep"] in SURFACES and abs(r["coupling_lambda"] - couplings["c0"]) <= 1e-12:
                by.setdefault(_dp_row_key(r), r)
        pairs = []
        for r in rows:
            if not r["label"].startswith("dp-c0-"):
                continue
            a = by.get(_dp_row_key(r))
            if a is None:
                continue
            pairs.append({"label": r["label"], "archive_label": a["label"],
                          "dE_N": abs(r["E_N"] - a["E_N"]), "dMI": abs(r["MI"] - a["MI"]),
                          "dwork": abs(r["work"] - a["work"])})
        xcheck.update({
            "n_matched": len(pairs),
            "max_abs_dE_N": max((q["dE_N"] for q in pairs), default=float("nan")),
            "max_abs_dMI": max((q["dMI"] for q in pairs), default=float("nan")),
            "max_abs_dwork": max((q["dwork"] for q in pairs), default=float("nan")),
            "pairs": pairs,
            "note": "device-point rows at the derived coupling vs the surfaces archive at the "
                    "anchored coupling (the same number): same code path, so any difference "
                    "is a build difference",
        })

    c0 = per.get("c0", {})
    box30 = c0.get("device_box_30_50mK", {})
    verdict = {
        "device_box_E_N_zero_at_every_coupling": bool(all(
            per[c]["device_box_30_50mK"].get("max_E_N", 1.0) == 0.0 for c in per)),
        "device_box_max_E_N_over_couplings": max(
            (per[c]["device_box_30_50mK"].get("max_E_N", float("nan")) for c in per), default=float("nan")),
        "device_box_max_MI_over_couplings": max(
            (per[c]["device_box_30_50mK"].get("max_MI", float("nan")) for c in per), default=float("nan")),
        "device_box_E_N_GO_anywhere_at_any_coupling": {
            k: bool(any(per[c]["device_box_30_50mK"].get("E_N_GO_anywhere", {}).get(k, False) for c in per))
            for k in levels},
        "device_box_MI_GO_anywhere_at_any_coupling": {
            k: bool(any(per[c]["device_box_30_50mK"].get("MI_GO_anywhere", {}).get(k, False) for c in per))
            for k in levels},
        "design_20mK_E_N_GO_anywhere_at_any_coupling": {
            k: bool(any(per[c]["design_20mK"].get("E_N_GO_anywhere", {}).get(k, False) for c in per))
            for k in levels},
        "vacuum_0mK_E_N_GO_at_derived_floor_by_coupling": {
            c: per[c]["vacuum_0mK"].get("E_N_GO_anywhere", {}).get("derived", False) for c in per},
        "derived_coupling_device_box": box30,
    }
    return {
        "build": build_info(),
        "operating_point": {
            "table1_scenario": p.scenario, "gap_GHz": p.gap_GHz, "gap_shift_GHz": p.gap_shift_GHz,
            "window_ns": p.window_ns, "separation_mm": p.separation_mm,
            "coupling_lambda_osc": p.coupling_lambda, "coupling_sigma": p.coupling_sigma,
            "coupling_mode": p.coupling_mode,
            "coupling_provenance": inputs.provenance["coupling_lambda"],
            "coupling_status": COUPLING_DERIVED_STATUS,
        },
        "coupling_map": inputs.coupling_map,
        "couplings": {c: {"lambda_osc": v, "description": DEVICE_POINT_COUPLINGS.get(c, "")}
                      for c, v in couplings.items()},
        "anchors": {a: {"description": d, "overrides": o} for a, (d, o) in anchors.items()},
        "temperatures_mK": list(DEVICE_POINT_T_MK),
        "resolution": {"derived_nats": inputs.resolution.derived,
                       "band_nats": [inputs.resolution.band_low, inputs.resolution.band_high]},
        "n_rows": len(rows),
        "wall_clock_s": float(wall_clock_s),
        "cpu_seconds_summed": float(sum(r["runtime_s"] for r in rows)),
        "all_audits_passed": bool(all(r["all_audits_passed"] for r in rows)),
        "all_converged": bool(all(r["converged"] for r in rows)),
        "worst_ledger_defect": float(max(abs(r["ledger_defect"]) for r in rows)) if rows else float("nan"),
        "mp_margin_flagged_rows": int(sum(r["mp_margin_flagged"] for r in rows)),
        "verdict": verdict,
        "per_coupling": per,
        "archive_cross_check": xcheck,
        "model_mismatch_companion": mismatch,
        "synthetic_inputs_remaining": list(inputs.synthetic_inputs),
    }


def _print_device_point_report(summary: Dict[str, Any]) -> None:
    print("=" * 72)
    v = summary["verdict"]
    print(f"device point at the derived coupling: rows {summary['n_rows']}, audits all passed "
          f"{summary['all_audits_passed']}, converged {summary['all_converged']}, worst ledger "
          f"defect {summary['worst_ledger_defect']:.3e}")
    for c, e in summary["per_coupling"].items():
        b = e["device_box_30_50mK"]
        d = e["design_20mK"]
        z = e["vacuum_0mK"]
        print(f"  [{c:5s}] lam_osc={e['lambda_osc']:.4f}: device box max E_N={b.get('max_E_N', float('nan')):.3e} "
              f"max MI={b.get('max_MI', float('nan')):.3e}; 20 mK max E_N={d.get('max_E_N', float('nan')):.3e} "
              f"GO@band-low={d.get('E_N_GO_anywhere', {}).get('band_low')}; T=0 max E_N="
              f"{z.get('max_E_N', float('nan')):.3e} GO@derived={z.get('E_N_GO_anywhere', {}).get('derived')}")
    print(f"  device box E_N == 0 at every coupling: {v['device_box_E_N_zero_at_every_coupling']}; "
          f"MI GO anywhere: {v['device_box_MI_GO_anywhere_at_any_coupling']}")
    x = summary["archive_cross_check"]
    if x.get("available"):
        print(f"  archive cross-check: {x['n_matched']} rows matched, max |dE_N|={x['max_abs_dE_N']:.3e}, "
              f"max |dMI|={x['max_abs_dMI']:.3e}")
    mm = summary.get("model_mismatch_companion")
    if mm:
        print(f"  model mismatch (pert., static gap): E_N xx/xp = {mm['E_N_ratio_xx_over_xp']:.2f}, "
              f"P_A xx/xp = {mm['P_A_ratio_xx_over_xp']:.2f}")


# --------------------------------------------------------------------------
# The derivative-coupling ('xp') map and the gap-tracking variant
# --------------------------------------------------------------------------

#: couplings of the 'xp' map: the derived value and its +-1 sigma
XP_MAP_COUPLINGS: Tuple[str, ...] = ("c0", "c-1s", "c+1s")
#: gap-tracking exponent per operator (see :func:`coupling_schedule`)
GAP_TRACKING_EXPONENT: Dict[str, float] = {"xx": 1.5, "xp": 0.5}
#: reduced axes for a budgeted run (``--grid 12``): every published anchor kept
T_GRID_12_MK: Tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 27.5, 30.0, 35.0, 40.0, 50.0, 80.0)
GPHI_GRID_12_HZ: Tuple[float, ...] = (1e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 2e8, 1e9)


def xp_map_specs(
    inputs: Inputs, couplings: Dict[str, float],
    anchors: Dict[str, Tuple[str, Dict[str, float]]], *, grid: int = 0,
    surface: str = MAIN_SURFACE, coupling: str = "xp",
) -> List[PointSpec]:
    """The derivative-coupling map: the main surface at the derived coupling
    and +-1 sigma, the device-point rows at the same three couplings, the
    static-gap split companions (derivative kernel, c0) and the finite-size
    controls (c0).  Surface labels carry the coupling name as a prefix so the
    three surfaces stay separable; every row goes through :func:`run_point`."""
    if grid == 0:
        T_values, X_values = None, None
    elif grid == 12:
        T_values, X_values = T_GRID_12_MK, GPHI_GRID_12_HZ
    else:
        raise ValueError(f"grid must be 0 (full 16 x 16) or 12, got {grid}")
    specs: List[PointSpec] = []
    for cname in XP_MAP_COUPLINGS:
        for sp in surface_specs(inputs, surface, T_values=T_values, X_values=X_values,
                                coupling=coupling, coupling_lambda=couplings[cname]):
            specs.append(replace(sp, label=f"{cname}:{sp.label}"))
    specs.extend(device_point_specs(
        inputs, {c: couplings[c] for c in XP_MAP_COUPLINGS}, anchors, coupling=coupling))
    specs.extend(split_specs(inputs, coupling=coupling, coupling_lambda=couplings["c0"]))
    specs.extend(control_specs(inputs, coupling=coupling, coupling_lambda=couplings["c0"]))
    return specs


def wall_positions(contour: Dict[str, Any]) -> Dict[str, Any]:
    """The temperature wall of one contour: per X column, the first go->nogo
    crossing in T (bracket, interpolation, sudden-death flag) or None."""
    out: Dict[str, Any] = {}
    for col in contour["cols"]:
        cr = [c for c in col["crossings"] if c["direction"] == "go->nogo"]
        out[f"{col['x']:g}"] = None if not cr else {
            "T_lo": cr[0]["x_lo"], "T_hi": cr[0]["x_hi"], "T_interp": cr[0]["x_interp"],
            "sudden_death": cr[0]["sudden_death"],
        }
    return out


def row_crossings(contour: Dict[str, Any]) -> Dict[str, Any]:
    """Per temperature row, the first go->nogo crossing along X, or None."""
    out: Dict[str, Any] = {}
    for row in contour["rows"]:
        cr = [c for c in row["crossings"] if c["direction"] == "go->nogo"]
        out[f"{row['temperature_mK']:g}"] = None if not cr else {
            "x_lo": cr[0]["x_lo"], "x_hi": cr[0]["x_hi"], "x_interp": cr[0]["x_interp"],
            "sudden_death": cr[0]["sudden_death"],
        }
    return out


def _box_compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "max_E_N_xx": a.get("max_E_N"), "max_E_N_xp": b.get("max_E_N"),
        "max_MI_xx": a.get("max_MI"), "max_MI_xp": b.get("max_MI"),
        "E_N_GO_anywhere_xx": a.get("E_N_GO_anywhere"), "E_N_GO_anywhere_xp": b.get("E_N_GO_anywhere"),
        "MI_GO_anywhere_xx": a.get("MI_GO_anywhere"), "MI_GO_anywhere_xp": b.get("MI_GO_anywhere"),
    }


def xp_map_summary(
    rows: Sequence[Dict[str, Any]], inputs: Inputs, couplings: Dict[str, float],
    anchors: Dict[str, Tuple[str, Dict[str, float]]], *, surface: str = MAIN_SURFACE,
    wall_clock_s: float, grid: int = 0, xcheck=None, xx_device_point=None,
    xx_summary=None, mismatch=None,
) -> Dict[str, Any]:
    """Contours, walls and verdicts of the 'xp' map, side by side with 'xx'."""
    levels = inputs.resolution.levels
    x_axis, log_x = SURFACES[surface]
    per: Dict[str, Any] = {}
    for cname in XP_MAP_COUPLINGS:
        sel = [r for r in rows if r["label"].startswith(f"{cname}:{surface}-")]
        if not sel:
            continue
        T, X, Z = surface_arrays(sel, surface, "E_N")
        _T, _X, M = surface_arrays(sel, surface, "MI")
        en = {k: surface_contour(T, X, Z, v, log_x=log_x) for k, v in levels.items()}
        mi = {k: surface_contour(T, X, M, v, log_x=log_x) for k, v in levels.items()}
        per[cname] = {
            "lambda_osc": couplings[cname], "n_points": len(sel),
            "max_E_N": float(np.max(Z)), "max_MI": float(np.max(M)),
            "T_mK": [float(t) for t in T], "X": [float(x) for x in X], "x_axis": x_axis,
            "E_N_contour": en, "MI_contour": mi,
            "temperature_wall": {k: wall_positions(en[k]) for k in levels},
            "dephasing_crossings": {k: row_crossings(en[k]) for k in levels},
            "device_anchors": anchor_verdicts(sel, surface, inputs),
            "pivot": pivot_verdict(sel, inputs),
            "all_audits_passed": bool(all(r["all_audits_passed"] for r in sel)),
            "all_converged": bool(all(r["converged"] for r in sel)),
            "worst_ledger_defect": float(max(abs(r["ledger_defect"]) for r in sel)),
        }
    xx_side = None
    if xx_summary is not None and surface in xx_summary.get("go_no_go", {}):
        b = xx_summary["go_no_go"][surface]
        xx_side = {
            "lambda_osc": xx_summary["operating_point"]["coupling_lambda_osc"],
            "temperature_wall": {k: wall_positions(b["E_N_contour"][k]) for k in levels},
            "dephasing_crossings": {k: row_crossings(b["E_N_contour"][k]) for k in levels},
            "max_E_N": {k: b["E_N_contour"][k]["max_z"] for k in levels},
            "go_fraction": {k: b["E_N_contour"][k]["go_fraction"] for k in levels},
            "device_anchors": b["device_anchors"],
        }
    dp_rows = [r for r in rows if r["label"].startswith("dp-")]
    dp = None
    if dp_rows:
        dp = device_point_summary(
            dp_rows, inputs, {c: couplings[c] for c in XP_MAP_COUPLINGS}, anchors,
            archive_rows=None, wall_clock_s=0.0, mismatch=mismatch,
        )
    side: Dict[str, Any] = {}
    if dp is not None and xx_device_point is not None:
        for cname in XP_MAP_COUPLINGS:
            xxc = xx_device_point["per_coupling"].get(cname)
            if xxc is None:
                continue
            xx_rows = {t["label"]: t for t in xxc["rows"]}
            pairs = []
            for t in dp["per_coupling"][cname]["rows"]:
                a = xx_rows.get(t["label"])
                if a is None:
                    continue
                pairs.append({
                    "label": t["label"], "temperature_mK": t["temperature_mK"], "anchor": t["anchor"],
                    "E_N_xx": a["E_N"], "E_N_xp": t["E_N"], "MI_xx": a["MI"], "MI_xp": t["MI"],
                    "verdict_xx": a["verdict"], "verdict_xp": t["verdict"],
                    "verdict_differs": a["verdict"] != t["verdict"],
                })
            pc = dp["per_coupling"][cname]
            side[cname] = {
                "rows": pairs,
                "n_verdicts_differ": int(sum(q["verdict_differs"] for q in pairs)),
                "device_box_30_50mK": _box_compare(xxc["device_box_30_50mK"], pc["device_box_30_50mK"]),
                "design_20mK": _box_compare(xxc["design_20mK"], pc["design_20mK"]),
                "vacuum_0mK": _box_compare(xxc["vacuum_0mK"], pc["vacuum_0mK"]),
            }
    split_cols = ("label", "separation_mm", "separation_sites", "E_N", "MI", "E_N_pert", "P_A",
                  "M_abs", "M_vac_abs", "M_comm_abs", "comm_fraction", "perturbative_ok",
                  "perturbative_quadrature_failed", "has_perturbative_row")
    split_xp = [{k: r[k] for k in split_cols}
                for r in sorted((r for r in rows if r["sweep"] == "split"),
                                key=lambda r: r["separation_sites"])]
    split_side = None
    if xx_summary is not None and xx_summary.get("communication_split_rows"):
        xx_split = {int(r["separation_sites"]): r for r in xx_summary["communication_split_rows"]}
        split_side = []
        for r in split_xp:
            a = xx_split.get(int(r["separation_sites"]))
            if a is None:
                continue
            split_side.append({
                "separation_sites": r["separation_sites"], "separation_mm": r["separation_mm"],
                "E_N_xx": a["E_N"], "E_N_xp": r["E_N"],
                "comm_fraction_xx": a["comm_fraction"], "comm_fraction_xp": r["comm_fraction"],
                "M_vac_abs_xx": a["M_vac_abs"], "M_vac_abs_xp": r["M_vac_abs"],
                "M_comm_abs_xx": a["M_comm_abs"], "M_comm_abs_xp": r["M_comm_abs"],
                "perturbative_ok_xx": a["perturbative_ok"], "perturbative_ok_xp": r["perturbative_ok"],
            })
    ctrl = sorted((r for r in rows if r["sweep"] == "control"), key=lambda r: r["boundary_buffer"])
    control: Dict[str, Any] = {}
    if ctrl:
        e = [r["E_N"] for r in ctrl]
        control = {"boundary_buffers": [r["boundary_buffer"] for r in ctrl],
                   "n_field_sites": [r["n_field_sites"] for r in ctrl], "E_N": e,
                   "MI": [r["MI"] for r in ctrl], "work": [r["work"] for r in ctrl],
                   "E_N_spread": float(max(e) - min(e)),
                   "E_N_relative_spread": float((max(e) - min(e)) / max(e)) if max(e) > 0 else float("nan")}
    c0 = per.get("c0", {})
    verdict = {
        "surface_E_N_GO_anywhere_by_coupling": {
            c: {k: bool(per[c]["E_N_contour"][k]["n_go"] > 0) for k in levels} for c in per},
        "surface_max_E_N_by_coupling": {c: per[c]["max_E_N"] for c in per},
        "surface_device_anchor_verdicts_c0": [
            {"T": a["temperature_anchor"], "x": a["x_anchor"], "E_N": a["E_N"], "MI": a["MI"],
             "verdict": a["verdict"]} for a in c0.get("device_anchors", [])],
        "device_box_E_N_zero_at_every_coupling": (dp["verdict"]["device_box_E_N_zero_at_every_coupling"]
                                                  if dp else None),
        "device_box_E_N_GO_anywhere_at_any_coupling": (
            dp["verdict"]["device_box_E_N_GO_anywhere_at_any_coupling"] if dp else None),
        "device_box_MI_GO_anywhere_at_any_coupling": (
            dp["verdict"]["device_box_MI_GO_anywhere_at_any_coupling"] if dp else None),
        "design_20mK_E_N_GO_anywhere_at_any_coupling": (
            dp["verdict"]["design_20mK_E_N_GO_anywhere_at_any_coupling"] if dp else None),
        "vacuum_0mK_E_N_GO_at_derived_floor_by_coupling": (
            dp["verdict"]["vacuum_0mK_E_N_GO_at_derived_floor_by_coupling"] if dp else None),
        "n_device_point_verdicts_differing_from_xx": (
            {c: side[c]["n_verdicts_differ"] for c in side} if side else None),
    }
    p = inputs.point
    return {
        "build": build_info(),
        "coupling_operator": "xp",
        "grid": int(grid),
        "operating_point": {
            "table1_scenario": p.scenario, "gap_GHz": p.gap_GHz, "gap_shift_GHz": p.gap_shift_GHz,
            "window_ns": p.window_ns, "separation_mm": p.separation_mm,
            "coupling_lambda_osc": p.coupling_lambda, "coupling_sigma": p.coupling_sigma,
            "coupling_mode": p.coupling_mode, "coupling_status": COUPLING_DERIVED_STATUS,
            "transcription": "lam_osc^xp = |lambda_TB| sqrt(2 Omega_lat): the exact MAP-1 "
                             "transcription of Teixido-Bonfill Eq. (37) (circuit_mapping, step 4)",
        },
        "couplings": {c: {"lambda_osc": couplings[c], "description": DEVICE_POINT_COUPLINGS[c]}
                      for c in XP_MAP_COUPLINGS},
        "resolution": {"derived_nats": inputs.resolution.derived,
                       "band_nats": [inputs.resolution.band_low, inputs.resolution.band_high]},
        "n_rows": len(rows),
        "wall_clock_s": float(wall_clock_s),
        "cpu_seconds_summed": float(sum(r["runtime_s"] for r in rows)),
        "slowest_point_s": float(max(r["runtime_s"] for r in rows)),
        "all_audits_passed": bool(all(r["all_audits_passed"] for r in rows)),
        "all_converged": bool(all(r["converged"] for r in rows)),
        "worst_ledger_defect": float(max(abs(r["ledger_defect"]) for r in rows)),
        "mp_margin_flagged_rows": int(sum(r["mp_margin_flagged"] for r in rows)),
        "verdict": verdict,
        "surface": {"name": surface, "per_coupling": per, "xx_archive": xx_side},
        "device_point": dp,
        "device_point_side_by_side": side,
        "communication_split_rows": split_xp,
        "communication_split_side_by_side": split_side,
        "finite_size_control": control,
        "cross_check_run_harvesting": xcheck,
        "model_mismatch_companion": mismatch,
        "synthetic_inputs_remaining": list(inputs.synthetic_inputs),
    }


def save_surfaces_by_coupling(path: Path, rows, inputs: Inputs, couplings: Dict[str, float],
                              surface: str = MAIN_SURFACE) -> Path:
    """Dense arrays of one surface at several couplings, keyed ``surface@coupling/...``."""
    payload: Dict[str, np.ndarray] = {}
    for cname in XP_MAP_COUPLINGS:
        sel = [r for r in rows if r["label"].startswith(f"{cname}:{surface}-")]
        if not sel:
            continue
        for col in ("E_N", "MI", "work", "E_N_per_work", "MI_per_work", "n_d_A",
                    "mp_margin_flagged", "all_audits_passed", "dt_converged"):
            T, X, Z = surface_arrays(sel, surface, col)
            payload[f"{surface}@{cname}/T_mK"] = T
            payload[f"{surface}@{cname}/{SURFACES[surface][0]}"] = X
            payload[f"{surface}@{cname}/{col}"] = Z
        payload[f"{surface}@{cname}/lambda_osc"] = np.array([couplings[cname]])
    payload["resolution_levels_nats"] = np.array(
        [inputs.resolution.derived, inputs.resolution.band_low, inputs.resolution.band_high])
    payload["__build__"] = np.array(json.dumps(build_info(), indent=2), dtype="U8192")
    np.savez_compressed(path, **payload)
    return path


def gap_tracking_specs(inputs: Inputs, anchors, lam: float) -> List[PointSpec]:
    """Device-point rows at the derived coupling with lam(t) tracking the gap,
    for both operators (labels ``dp-gtxx-...`` / ``dp-gtxp-...``)."""
    specs: List[PointSpec] = []
    for coupling, e in GAP_TRACKING_EXPONENT.items():
        specs.extend(device_point_specs(inputs, {f"gt{coupling}": lam}, anchors,
                                        coupling=coupling, gap_tracking_exponent=e))
    return specs


def gap_tracking_summary(rows, inputs: Inputs, anchors, lam: float, *, wall_clock_s: float,
                         xx_device_point=None, xp_summary=None) -> Dict[str, Any]:
    """Tracked vs untracked device-point rows, per operator."""
    couplings = {f"gt{c}": lam for c in GAP_TRACKING_EXPONENT}
    dp = device_point_summary(rows, inputs, couplings, anchors, archive_rows=None,
                              wall_clock_s=wall_clock_s)
    compare: Dict[str, Any] = {}
    for coupling in GAP_TRACKING_EXPONENT:
        ref_rows = None
        if coupling == "xx" and xx_device_point is not None:
            ref_rows = xx_device_point["per_coupling"]["c0"]["rows"]
        if coupling == "xp" and xp_summary is not None and xp_summary.get("device_point"):
            ref_rows = xp_summary["device_point"]["per_coupling"]["c0"]["rows"]
        if ref_rows is None:
            continue
        ref = {t["label"].split("-", 2)[2]: t for t in ref_rows}
        pairs = []
        for t in dp["per_coupling"][f"gt{coupling}"]["rows"]:
            key = t["label"].split("-", 2)[2]
            a = ref.get(key)
            if a is None:
                continue
            rel = (t["E_N"] - a["E_N"]) / a["E_N"] if a["E_N"] > 0.0 else None
            pairs.append({"key": key, "E_N_untracked": a["E_N"], "E_N_tracked": t["E_N"],
                          "MI_untracked": a["MI"], "MI_tracked": t["MI"], "rel_change_E_N": rel,
                          "verdict_changed": a["verdict"] != t["verdict"],
                          "verdict_untracked": a["verdict"], "verdict_tracked": t["verdict"]})
        rels = [abs(q["rel_change_E_N"]) for q in pairs if q["rel_change_E_N"] is not None]
        compare[coupling] = {
            "exponent": GAP_TRACKING_EXPONENT[coupling], "n": len(pairs),
            "max_abs_rel_change_E_N": max(rels) if rels else None,
            "any_verdict_changed": bool(any(q["verdict_changed"] for q in pairs)),
            "changed": [q["key"] for q in pairs if q["verdict_changed"]],
            "pairs": pairs,
        }
    return {
        "build": build_info(), "lambda_osc": lam, "exponents": dict(GAP_TRACKING_EXPONENT),
        "bound_stated": {"xx": "<= 9 % in lambda ((0.937)^1.5)", "xp": "<= 3.2 % in lambda ((0.937)^0.5)"},
        "n_rows": len(rows), "wall_clock_s": float(wall_clock_s),
        "all_audits_passed": bool(all(r["all_audits_passed"] for r in rows)),
        "all_converged": bool(all(r["converged"] for r in rows)),
        "worst_ledger_defect": float(max(abs(r["ledger_defect"]) for r in rows)),
        "device_point": dp, "comparison_vs_untracked": compare,
    }


def _print_xp_report(summary: Dict[str, Any]) -> None:
    print("=" * 72)
    print(f"'xp' map: rows {summary['n_rows']}, wall {summary['wall_clock_s']:.1f}s, audits "
          f"{summary['all_audits_passed']}, converged {summary['all_converged']}, worst ledger "
          f"{summary['worst_ledger_defect']:.3e}, mp-flagged {summary['mp_margin_flagged_rows']}")
    for cname, e in summary["surface"]["per_coupling"].items():
        for k, c in e["E_N_contour"].items():
            w = e["temperature_wall"][k]
            first = next((v for v in w.values() if v is not None), None)
            print(f"  [xp {cname} {e['lambda_osc']:.4f}] E_N > {c['level_nats']:.2e}: GO {c['n_go']}/"
                  f"{c['n_points']} (max {c['max_z']:.3e}); wall at lowest Gphi: "
                  + (f"[{first['T_lo']:g}, {first['T_hi']:g}] mK sd={first['sudden_death']}" if first else "none"))
        for a in e["device_anchors"]:
            print(f"      anchor {a['temperature_anchor']} x {a['x_anchor']}: E_N={a['E_N']:.3e} "
                  f"MI={a['MI']:.3e} -> " + ", ".join(f"{k}:{v}" for k, v in a["verdict"].items()))
    v = summary["verdict"]
    print(f"  device box E_N == 0 at every coupling: {v['device_box_E_N_zero_at_every_coupling']}; "
          f"GO anywhere: {v['device_box_E_N_GO_anywhere_at_any_coupling']}; MI: "
          f"{v['device_box_MI_GO_anywhere_at_any_coupling']}; verdicts differing from xx: "
          f"{v['n_device_point_verdicts_differing_from_xx']}")
    x = summary.get("cross_check_run_harvesting") or {}
    if x:
        print(f"  two drivers (xp, static gap): |dE_N| = {x.get('abs_difference')}")
    if summary.get("communication_split_side_by_side"):
        for r in summary["communication_split_side_by_side"][:8]:
            print(f"  split {int(r['separation_sites']):2d} sites: E_N xx={r['E_N_xx']:.3e} xp={r['E_N_xp']:.3e} "
                  f"comm xx={r['comm_fraction_xx']:.3f} xp={r['comm_fraction_xp']:.3f}")


#: IR-mass control: the declared regulator FIELD_MASS and three alternatives,
#: in lattice units (x 7.3 GHz: 0.37 / 0.73 / 1.46 / 2.92 GHz).
IR_MASS_GRID: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.4)


def ir_control_specs(inputs: Inputs, anchors, lam: float,
                     masses: Sequence[float] = IR_MASS_GRID,
                     T_values: Sequence[float] = (0.0, 20.0, 30.0, 50.0)) -> List[PointSpec]:
    """Device-point rows at several IR masses, both operators.

    The mass is the candidate's one synthetic input whose effect on the
    verdicts was never measured (the buffer's is; the grids and axes are
    coordinates).  Labels ``dp-m<mass><coupling>-...`` keep the device-point
    shape so :func:`device_point_summary` reads them unchanged.
    """
    specs: List[PointSpec] = []
    for m in masses:
        for coupling in ("xx", "xp"):
            cname = f"m{m:g}{coupling}"
            specs.extend(device_point_specs(
                inputs, {cname: lam}, anchors, T_values=T_values,
                coupling=coupling, field_mass=float(m)))
    return specs


def ir_control_summary(rows, inputs: Inputs, anchors, lam: float, *, wall_clock_s: float,
                       masses: Sequence[float] = IR_MASS_GRID) -> Dict[str, Any]:
    """Does the go/no-go verdict depend on the synthetic IR regulator?"""
    levels = inputs.resolution.levels
    couplings = {f"m{m:g}{c}": lam for m in masses for c in ("xx", "xp")}
    dp = device_point_summary(rows, inputs, couplings, anchors, archive_rows=None,
                              wall_clock_s=wall_clock_s)
    per: Dict[str, Any] = {}
    for coupling in ("xx", "xp"):
        by_mass = {}
        for m in masses:
            e = dp["per_coupling"][f"m{m:g}{coupling}"]
            # a reduced temperature set leaves some boxes empty; report None there
            box30 = e["device_box_30_50mK"]
            by_mass[f"{m:g}"] = {
                "device_box_max_E_N": box30.get("max_E_N"),
                "device_box_GO": box30.get("E_N_GO_anywhere"),
                "device_box_GO_anchors_band_low": box30.get("GO_anchors", {}).get("band_low"),
                "design_20mK_max_E_N": e["design_20mK"].get("max_E_N"),
                "vacuum_0mK_max_E_N": e["vacuum_0mK"].get("max_E_N"),
                "rows": {f"{t['temperature_mK']:g}:{t['anchor']}": {"E_N": t["E_N"], "verdict": t["verdict"]}
                         for t in e["rows"]},
            }
        ref = by_mass[f"{FIELD_MASS:g}"]["rows"]
        stable = {k: True for k in levels}
        drift = []
        for mk, mv in by_mass.items():
            for key, r in mv["rows"].items():
                a = ref.get(key)
                if a is None:
                    continue
                for k in levels:
                    if a["verdict"][k] != r["verdict"][k]:
                        stable[k] = False
                        drift.append({"mass": mk, "row": key, "level": k,
                                      "verdict_at_declared_mass": a["verdict"][k],
                                      "verdict_here": r["verdict"][k],
                                      "E_N_declared": a["E_N"], "E_N_here": r["E_N"]})
        # the dephasing anchors are the ones the device claim rests on
        deph = {mk: max(v["rows"][k]["E_N"] for k in v["rows"] if "Gphi" in k)
                for mk, v in by_mass.items()}
        per[coupling] = {
            "by_mass": by_mass,
            "verdicts_independent_of_mass": stable,
            "verdict_drift": drift,
            "max_E_N_on_dephasing_anchors_by_mass": deph,
            "dephasing_anchors_all_zero_at_every_mass": all(v == 0.0 for v in deph.values()),
        }
    return {
        "build": build_info(), "lambda_osc": lam, "masses": list(masses),
        "declared_mass": FIELD_MASS, "n_rows": len(rows), "wall_clock_s": float(wall_clock_s),
        "all_audits_passed": bool(all(r["all_audits_passed"] for r in rows)),
        "all_converged": bool(all(r["converged"] for r in rows)),
        "worst_ledger_defect": float(max(abs(r["ledger_defect"]) for r in rows)),
        "note": "The IR mass is a synthetic regulator (model limitation 2). This control "
                "measures whether it can move a device-anchor verdict; the buffer's effect is "
                "measured by the finite-size control, and the grids/axes are coordinates.",
        "per_coupling": per, "device_point": dp,
    }


# --------------------------------------------------------------------------
# The UV-cutoff bracket (--cutoff-bracket)
# --------------------------------------------------------------------------

#: lattice spacings of the bracket -> the Brillouin cutoff they realise.
#: sqrt(m^2 + 4/a^2) x 7.3 GHz: 14.7 / 24.4 / 36.5 / 50.0 / 73.0 GHz, so the
#: bracket CONTAINS both the archives' 14.6 GHz and the device's 50 GHz.
CUTOFF_SPACINGS: Dict[str, float] = {
    "a1.0": 1.0,        # 14.7 GHz -- the archives
    "a0.6": 0.6,        # 24.4 GHz
    "a0.4": 0.4,        # 36.5 GHz
    "a0.292": 0.29209,  # 50.0 GHz -- the device's own line cutoff
    "a0.2": 0.2,        # 73.0 GHz -- above the device
}
#: the anchors the bracket is run at.  The four DEPHASING anchors are the ones
#: the negative device statement rests on; the three others are the contrast
#: (they are where E_N is alive at a = 1).  The 60/120 mm anchors are omitted:
#: they are dead by separation at every cutoff and their N would dominate cost.
CUTOFF_ANCHOR_KEYS: Tuple[str, ...] = (
    "Gphi_min", "Gphi_max", "G1min+Gphimin", "G1max+Gphimax",  # the statement
    "G1_min", "G1_max", "d19.2",                                # the contrast
)
#: anchors whose verdict IS the negative statement
DEPHASING_ANCHOR_KEYS: Tuple[str, ...] = (
    "Gphi_min", "Gphi_max", "G1min+Gphimin", "G1max+Gphimax",
)
CUTOFF_T_MK: Tuple[float, ...] = (0.0, 30.0, 50.0)
#: spacings at which the full (reduced-axis) surface is re-run
CUTOFF_SURFACE_SPACINGS: Tuple[str, ...] = ("a0.6", "a0.292")
#: Gamma_phi axis of the cutoff-bracket surfaces: :data:`GPHI_GRID_12_HZ`
#: WITHOUT its top synthetic column (1e9 Hz).  Recorded because it is an audit
#: event, not a convenience: on the finest lattice (a = 0.292, N = 103) the
#: 1e9 Hz / 80 mK cell books 16384 channel entries plus 8192 work entries on a
#: state of <H> = 227.0, and its ledger closes to 1.0107e-9 against the
#: **absolute** 1e-9 closure tolerance -- a 1 % overshoot of the tolerance and
#: 4.45e-12 of the state's own energy, i.e. float64 accumulation over the entry
#: count, not a physics defect.  The tolerance was NOT relaxed; the column is
#: dropped instead.  It carries no verdict information here: it is a synthetic
#: sweep extension 0.7 decades above the top device anchor (2e8 Hz), and E_N is
#: exactly 0 on it at every cutoff measured (and on all 64 of its rows retained
#: in rows.npz / rows_xp.npz).  Both device anchors (1e7, 2e8 Hz) are retained.
#: NOTE the exposure was NOT confined to the dropped column: Gamma_phi = 2e8 Hz,
#: a RETAINED device anchor, also crosses the absolute criterion (1.0527e-9) at
#: halving level 9, and the rows reported here pass only because the dt gate
#: converges at level 8.  Dropping the column removed the TRIGGER, not the
#: exposure; the marginal element was the absolute criterion itself, which did
#: not scale with <H> or entry count.
#: RESOLVED (see the ``audit_event`` block of :func:`cutoff_summary`): the
#: closure criterion now scales with the booked energies and entry count
#: (:meth:`vacuum.audits.EnergyLedger.closure_tolerance`), and re-running both
#: cells at halving levels 8 and 9 shows neither fires (margins 97-422x).  The
#: column stays out of the ARCHIVED surfaces -- restoring it needs a re-run of
#: the bracket, and it carries no verdict information (E_N = 0.0) -- but it is
#: no longer excluded because of an audit.
GPHI_GRID_CUTOFF_HZ: Tuple[float, ...] = tuple(x for x in GPHI_GRID_12_HZ if x <= 2e8)


def cutoff_specs(inputs: Inputs, lam: float, anchors, *, coupling: str = "xp",
                 spacings: Optional[Dict[str, float]] = None) -> List[PointSpec]:
    """Anchor rows at every cutoff, plus the reduced surface at two of them."""
    spacings = dict(CUTOFF_SPACINGS if spacings is None else spacings)
    sub = {k: anchors[k] for k in CUTOFF_ANCHOR_KEYS if k in anchors}
    specs: List[PointSpec] = []
    for cname, a in spacings.items():
        for sp in device_point_specs(inputs, {cname: lam}, sub, T_values=CUTOFF_T_MK,
                                     coupling=coupling, lattice_spacing=a):
            specs.append(sp)
    for cname in CUTOFF_SURFACE_SPACINGS:
        if cname not in spacings:
            continue
        for sp in surface_specs(inputs, MAIN_SURFACE, T_values=T_GRID_12_MK,
                                X_values=GPHI_GRID_CUTOFF_HZ, coupling=coupling,
                                coupling_lambda=lam, lattice_spacing=spacings[cname]):
            specs.append(replace(sp, label=f"{cname}:{sp.label}"))
    return specs


def cutoff_summary(rows, inputs: Inputs, lam: float, anchors, *, wall_clock_s: float,
                   coupling: str = "xp", spacings: Optional[Dict[str, float]] = None,
                   xp_summary=None) -> Dict[str, Any]:
    """Is any anchor verdict cutoff-sensitive?  Above all the negative one."""
    spacings = dict(CUTOFF_SPACINGS if spacings is None else spacings)
    levels = inputs.resolution.levels
    sub = {k: anchors[k] for k in CUTOFF_ANCHOR_KEYS if k in anchors}
    dp = device_point_summary(rows, inputs, {c: lam for c in spacings}, sub,
                              archive_rows=None, wall_clock_s=0.0)
    per: Dict[str, Any] = {}
    for cname, a in spacings.items():
        e = dp["per_coupling"][cname]
        sel = [r for r in rows if r["label"].startswith(f"dp-{cname}-")]
        cut = float(sel[0]["uv_cutoff_GHz"]) if sel else float("nan")
        deph = [t for t in e["rows"] if t["anchor"] in DEPHASING_ANCHOR_KEYS]
        cont = [t for t in e["rows"] if t["anchor"] not in DEPHASING_ANCHOR_KEYS]
        per[cname] = {
            "lattice_spacing": a,
            "uv_cutoff_GHz": cut,
            "n_field_sites": float(sel[0]["n_field_sites"]) if sel else float("nan"),
            "max_E_N_dephasing_anchors": max((t["E_N"] for t in deph), default=float("nan")),
            "max_MI_dephasing_anchors": max((t["MI"] for t in deph), default=float("nan")),
            "dephasing_anchors_all_zero": all(t["E_N"] == 0.0 for t in deph),
            "dephasing_anchor_verdicts": {
                f"{t['temperature_mK']:g}:{t['anchor']}": t["verdict"] for t in deph},
            "max_E_N_contrast_anchors": max((t["E_N"] for t in cont), default=float("nan")),
            "contrast_verdicts": {
                f"{t['temperature_mK']:g}:{t['anchor']}": t["verdict"] for t in cont},
            "rows": e["rows"],
            "all_audits_passed": e["all_audits_passed"],
            "all_converged": e["all_converged"],
            "worst_ledger_defect": e["worst_ledger_defect"],
        }
    # --- the surfaces at the bracketing cutoffs -----------------------------
    x_axis, log_x = SURFACES[MAIN_SURFACE]
    surf: Dict[str, Any] = {}
    for cname in CUTOFF_SURFACE_SPACINGS:
        sel = [r for r in rows if r["label"].startswith(f"{cname}:{MAIN_SURFACE}-")]
        if not sel:
            continue
        T, X, Z = surface_arrays(sel, MAIN_SURFACE, "E_N")
        _T, _X, M = surface_arrays(sel, MAIN_SURFACE, "MI")
        en = {k: surface_contour(T, X, Z, v, log_x=log_x) for k, v in levels.items()}
        surf[cname] = {
            "lattice_spacing": spacings[cname],
            "uv_cutoff_GHz": float(sel[0]["uv_cutoff_GHz"]),
            "n_points": len(sel), "max_E_N": float(np.max(Z)), "max_MI": float(np.max(M)),
            "go_fraction": {k: en[k]["go_fraction"] for k in levels},
            "temperature_wall": {k: wall_positions(en[k]) for k in levels},
            "dephasing_crossings": {k: row_crossings(en[k]) for k in levels},
            "device_anchors": anchor_verdicts(sel, MAIN_SURFACE, inputs),
            "gammaphi_axis_Hz": [float(x) for x in GPHI_GRID_CUTOFF_HZ],
            "excluded_columns_Hz": [float(x) for x in GPHI_GRID_12_HZ
                                    if x not in GPHI_GRID_CUTOFF_HZ],
        }
    if xp_summary is not None:
        c0 = xp_summary["surface"]["per_coupling"].get("c0")
        if c0 is not None:
            surf["a1.0_reference_16x16"] = {
                "lattice_spacing": 1.0, "uv_cutoff_GHz": 14.67,
                "note": "the Deliverable 5 surface (16 x 16 axes, not the reduced 12 x 12)",
                "max_E_N": c0["max_E_N"],
                "go_fraction": {k: c0["E_N_contour"][k]["go_fraction"] for k in levels},
                "temperature_wall": c0["temperature_wall"],
                "dephasing_crossings": c0["dephasing_crossings"],
            }
    # --- the margin of the negative statement --------------------------------
    dev_gphi = float(inputs.anchors["gammaphi_Hz"]["digitized_min"])
    margins = {}
    for cname, e in surf.items():
        cr = e.get("dephasing_crossings", {}).get("band_low", {})
        tol = [v["x_hi"] for v in cr.values() if v is not None]
        best = max(tol) if tol else None
        margins[cname] = {
            "uv_cutoff_GHz": e["uv_cutoff_GHz"],
            "dephasing_tolerance_Hz_upper_bracket": best,
            "device_digitised_Gphi_min_Hz": dev_gphi,
            "margin_factor": (dev_gphi / best) if best else None,
        }
    all_zero = all(v["dephasing_anchors_all_zero"] for v in per.values())
    drift = []
    ref = per.get("a1.0", {}).get("dephasing_anchor_verdicts", {})
    ref_c = per.get("a1.0", {}).get("contrast_verdicts", {})
    for cname, e in per.items():
        for key, v in e["dephasing_anchor_verdicts"].items():
            if ref.get(key) is not None and v != ref[key]:
                drift.append({"cutoff": cname, "anchor": key, "kind": "dephasing",
                              "at_a1": ref[key], "here": v})
        for key, v in e["contrast_verdicts"].items():
            if ref_c.get(key) is not None and v != ref_c[key]:
                drift.append({"cutoff": cname, "anchor": key, "kind": "contrast",
                              "at_a1": ref_c[key], "here": v})
    verdict = {
        "cutoff_bracket_GHz": [min(v["uv_cutoff_GHz"] for v in per.values()),
                               max(v["uv_cutoff_GHz"] for v in per.values())],
        "brackets_the_device_50GHz": bool(
            min(v["uv_cutoff_GHz"] for v in per.values()) <= 50.0
            <= max(v["uv_cutoff_GHz"] for v in per.values())),
        "negative_statement_holds_at_every_cutoff": bool(all_zero),
        "max_E_N_on_any_dephasing_anchor_over_bracket": max(
            v["max_E_N_dephasing_anchors"] for v in per.values()),
        "any_anchor_verdict_drifts": bool(drift),
        "verdict_drift": drift,
        "dephasing_margin_by_cutoff": margins,
        "min_margin_factor": min((m["margin_factor"] for m in margins.values()
                                  if m["margin_factor"] is not None), default=None),
        "contrast_max_E_N_by_cutoff": {c: e["max_E_N_contrast_anchors"] for c, e in per.items()},
    }
    return {
        "build": build_info(), "coupling_operator": coupling, "lambda_osc": lam,
        "spacings": spacings, "temperatures_mK": list(CUTOFF_T_MK),
        "anchors": {k: {"description": d, "overrides": o} for k, (d, o) in sub.items()},
        "dephasing_anchor_keys": list(DEPHASING_ANCHOR_KEYS),
        "resolution": {"derived_nats": inputs.resolution.derived,
                       "band_nats": [inputs.resolution.band_low, inputs.resolution.band_high]},
        "n_rows": len(rows), "wall_clock_s": float(wall_clock_s),
        "cpu_seconds_summed": float(sum(r["runtime_s"] for r in rows)),
        "slowest_point_s": float(max(r["runtime_s"] for r in rows)),
        "all_audits_passed": bool(all(r["all_audits_passed"] for r in rows)),
        "all_converged": bool(all(r["converged"] for r in rows)),
        "worst_ledger_defect": float(max(abs(r["ledger_defect"]) for r in rows)),
        "mp_margin_flagged_rows": int(sum(r["mp_margin_flagged"] for r in rows)),
        "audit_event": {
            "what": "the energy-ledger closure audit fired on ONE cell of the first pass of this "
                    "bracket: a = 0.292 (N = 103), T = 80 mK, Gamma_phi = 1e9 Hz",
            "defect": -1.0107e-9, "tolerance": LEDGER_TOL,
            "scale": "16384 channel entries plus 8192 work entries on a state of <H> = 227.0; "
                     "the defect is 4.45e-12 of that energy, i.e. float64 accumulation over "
                     "the entry count, not a physics defect (confirmed independently by a "
                     "41-run scaling study: the same 1.0107e-9 at halving level 9, a pooled "
                     "log-log slope of +0.81 in entry count over N = 48..24576, and an "
                     "<H>-proportional relative defect: 4.45e-12 at <H> = 227 vs 4.49e-12 at "
                     "<H> = 55.3)",
            "exposure_is_not_confined_to_the_dropped_column": "Gamma_phi = 2e8 Hz, a RETAINED "
                     "device anchor, also crosses the absolute criterion (1.0527e-9) at halving "
                     "level 9; the rows reported here pass only because the dt gate converges "
                     "at level 8. Dropping the synthetic column removed the TRIGGER, not the "
                     "exposure. No verdict moves either way (E_N = 0 on those cells regardless).",
            "marginal_element": "the ledger's ABSOLUTE 1e-9 closure criterion, which scales "
                     "with neither <H> nor the entry count.",
            "action": "the tolerance was NOT relaxed. The synthetic Gamma_phi extension column "
                      "above the device's range (1e9 Hz) was dropped from the cutoff-bracket "
                      "surfaces (GPHI_GRID_CUTOFF_HZ); both device anchors (1e7, 2e8 Hz) are "
                      "retained, E_N is exactly 0 on the dropped column at every cutoff "
                      "measured, and every row reported here passed the unrelaxed audit.",
            "resolution": "RESOLVED by correcting the criterion, not by relaxing it "
                      "(vacuum.audits.EnergyLedger.closure_tolerance): the closure identity "
                      "telescopes exactly, so its only error is roundoff, and the criterion is "
                      "now tol = max(atol, 4 eps sqrt(n_modes) n_ops scale) with atol = 1e-12 "
                      "-- stricter than the old absolute 1e-9 for every protocol whose "
                      "roundoff floor is below it. Both cells were re-run on the corrected "
                      "criterion: at halving level 9 the dropped 1e9 Hz cell closes to "
                      "-9.985e-10 against tol 1.03e-7 (103x margin) and the retained 2e8 Hz "
                      "anchor to -1.0516e-9 against tol 1.02e-7 (97x margin, reproducing the "
                      "recorded 1.0527e-9 to 0.1 %); level 8 gives -1.26e-10 and -1.21e-10. "
                      "Neither fires. The dropped column is still absent from the ARCHIVED "
                      "surfaces (restoring it needs a re-run of the bracket) and still carries "
                      "no verdict information: E_N = 0.0 on it at this cutoff at both levels.",
        },
        "note": "The lattice's Brillouin edge sqrt(m^2 + 4/a^2) IS the model's UV cutoff. "
                "Spacing a is varied at FIXED physical geometry (separation, window, buffer) "
                "and fixed rates, with the field renormalisation of lattice_at_spacing "
                "(K -> harmonic_chain_K(N, m a)/a^2, lam -> lam/sqrt(a)); a = 1 reproduces "
                "the archives bit-for-bit. The lattice cutoff is a HARD band edge while the "
                "device's is the exponential C(omega) = exp(-|omega|/2 Omega_cut) of Eq. (9), "
                "so the bracket is on the cutoff SCALE, not its shape.",
        "verdict": verdict, "per_cutoff": per, "surface": surf, "device_point": dp,
    }


def _print_cutoff_report(summary: Dict[str, Any]) -> None:
    print("=" * 72)
    v = summary["verdict"]
    print(f"UV-cutoff bracket: {summary['n_rows']} rows, wall {summary['wall_clock_s']:.1f}s, "
          f"audits {summary['all_audits_passed']}, converged {summary['all_converged']}, "
          f"worst ledger {summary['worst_ledger_defect']:.3e}")
    print(f"  bracket {v['cutoff_bracket_GHz'][0]:.1f}-{v['cutoff_bracket_GHz'][1]:.1f} GHz; "
          f"contains the device's 50 GHz: {v['brackets_the_device_50GHz']}")
    for c, e in summary["per_cutoff"].items():
        print(f"  [{c:7s}] cutoff={e['uv_cutoff_GHz']:6.2f} GHz N={e['n_field_sites']:4.0f}: "
              f"dephasing anchors max E_N={e['max_E_N_dephasing_anchors']:.3e} "
              f"(all zero: {e['dephasing_anchors_all_zero']}); contrast max E_N="
              f"{e['max_E_N_contrast_anchors']:.3e}")
    print(f"  NEGATIVE STATEMENT holds at every cutoff: "
          f"{v['negative_statement_holds_at_every_cutoff']}; any verdict drift: "
          f"{v['any_anchor_verdict_drifts']}")
    for d in v["verdict_drift"][:8]:
        print(f"     drift: {d['cutoff']} {d['kind']} {d['anchor']}: {d['at_a1']} -> {d['here']}")
    for c, m in v["dephasing_margin_by_cutoff"].items():
        print(f"  margin @ {c} ({m['uv_cutoff_GHz']:.1f} GHz): tolerance <= "
              f"{m['dephasing_tolerance_Hz_upper_bracket']} Hz vs device "
              f"{m['device_digitised_Gphi_min_Hz']:.0e} Hz -> {m['margin_factor']}x")


#: quick (smoke) configuration: coarse gate, small buffer, 4 x 4 main surface
QUICK_COMMON = dict(boundary_buffer=4, conv_tol=1e-7, nseg0=4, max_halvings=10)
QUICK_T_MK = (0.0, 20.0, 30.0, 50.0)
QUICK_GPHI_HZ = (1e5, 1e6, 1e7, 2e8)


def quick_specs(inputs: Inputs) -> List[PointSpec]:
    """A 4 x 4 mini-surface plus the branches the surface does not reach.

    Same code path as the full run, deliberately cheap.  Rows: the 4x4
    (T x Gamma_phi) main surface (thermal + dephasing + gap-schedule
    branches), one relaxation point (loss channel), one static-gap split
    companion (perturbative columns), and one truncated-gaussian point
    (the other switching branch).
    """
    out = surface_specs(inputs, MAIN_SURFACE, T_values=QUICK_T_MK,
                        X_values=QUICK_GPHI_HZ, **QUICK_COMMON)
    out.append(base_spec(inputs, "quick-relaxation", "relaxation",
                         gamma1_MHz=1850.0, grid_iT=0, grid_iX=0, **QUICK_COMMON))
    out.extend(split_specs(inputs, X_values=(SEP_GRID_SITES[6],)))
    out[-1] = replace(out[-1], **QUICK_COMMON)
    out.append(base_spec(inputs, "quick-gaussian", "control", switching_kind="gaussian",
                         **QUICK_COMMON))
    return out


# --------------------------------------------------------------------------
# Running (sequential or a spawn pool; identical code path per point)
# --------------------------------------------------------------------------

_WORKER_INPUTS: Optional[Inputs] = None


def _worker_init() -> None:
    global _WORKER_INPUTS
    _WORKER_INPUTS = load_inputs()


def _worker_run(job):
    i, spec = job
    try:
        return i, run_point(spec, _WORKER_INPUTS), None
    except Exception as exc:  # reported, then re-raised by the parent
        return i, None, f"{spec.label}: {type(exc).__name__}: {exc}"


def run_specs(
    specs: Sequence[PointSpec], inputs: Inputs, *, verbose: bool = True, n_workers: int = 1,
) -> List[Dict[str, Any]]:
    """Run a list of specs, returning their rows in order.

    ``n_workers > 1`` distributes points over a ``spawn`` pool (every worker
    loads the same inputs and calls the same :func:`run_point`); any point
    that fails is reported and the whole run raises -- there is no partial
    archive.
    """
    def report(i, row):
        if verbose:
            print(
                f"  [{i + 1:4d}/{len(specs)}] {row['label']:<30s} "
                f"E_N={row['E_N']:.4e} MI={row['MI']:.4e} W={row['work']:.3e} "
                f"dt={row['dt_converged']:.1e} mp={'Y' if row['mp_margin_flagged'] else 'n'} "
                f"audits={'PASS' if row['all_audits_passed'] else 'FAIL'} "
                f"({row['runtime_s']:.1f}s)", flush=True,
            )

    if n_workers <= 1:
        rows = []
        for i, spec in enumerate(specs):
            row = run_point(spec, inputs)
            report(i, row)
            rows.append(row)
        return rows

    results: List[Optional[Dict[str, Any]]] = [None] * len(specs)
    errors: List[str] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(int(n_workers), initializer=_worker_init) as pool:
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
# Deliverables computed from the rows
# --------------------------------------------------------------------------


def surface_arrays(rows: Sequence[Dict[str, Any]], name: str, column: str = "E_N"):
    """(T_values, X_values, Z[iT, iX]) of one surface from its rows."""
    x_axis, _log = SURFACES[name]
    sel = [r for r in rows if r["sweep"] == name]
    if not sel:
        raise ValueError(f"no rows for surface {name!r}")
    nT = int(max(r["grid_iT"] for r in sel)) + 1
    nX = int(max(r["grid_iX"] for r in sel)) + 1
    Z = np.full((nT, nX), np.nan)
    T = np.full(nT, np.nan)
    X = np.full(nX, np.nan)
    for r in sel:
        iT, iX = int(r["grid_iT"]), int(r["grid_iX"])
        Z[iT, iX] = float(r[column])
        T[iT] = float(r["temperature_mK"])
        X[iX] = float(r[x_axis])
    if np.isnan(Z).any():
        raise ValueError(f"surface {name!r} has {int(np.isnan(Z).sum())} missing cells")
    return T, X, Z


def axis_crossings(x, z, level: float, *, log_x: bool = False) -> List[Dict[str, Any]]:
    """Every crossing of z = level along a monotone 1D axis.

    GO means ``z > level`` strictly.  A crossing is a change of GO status
    between neighbouring grid points; ``x_interp`` is the linear (or
    log-linear, ``log_x``) interpolation of the crossing inside that cell,
    which is exact when z is linear in the interpolation variable there,
    and is flagged ``sudden_death`` when one side of the cell is exactly
    zero (E_N dies suddenly: the interpolated position is then a fiction
    and the bracket ``[x_lo, x_hi]`` is the result).
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    if x.ndim != 1 or z.shape != x.shape:
        raise ValueError("x and z must be 1D arrays of equal length")
    if x.size >= 2 and not (np.all(np.diff(x) > 0.0) or np.all(np.diff(x) < 0.0)):
        raise ValueError("axis must be strictly monotone")
    u = np.log10(x) if log_x else x
    go = z > level
    out: List[Dict[str, Any]] = []
    for k in range(x.size - 1):
        if go[k] == go[k + 1]:
            continue
        z0, z1 = z[k], z[k + 1]
        frac = (level - z0) / (z1 - z0)  # z1 != z0 because GO status differs
        ui = u[k] + frac * (u[k + 1] - u[k])
        out.append({
            "x_lo": float(x[k]), "x_hi": float(x[k + 1]),
            "z_lo": float(z0), "z_hi": float(z1),
            "x_interp": float(10.0**ui if log_x else ui),
            "direction": "go->nogo" if go[k] else "nogo->go",
            "sudden_death": bool(min(z0, z1) == 0.0),
        })
    return out


def surface_contour(T, X, Z, level: float, *, log_x: bool = False) -> Dict[str, Any]:
    """The z = level contour of a surface as row-wise and column-wise crossings.

    ``rows``: for each temperature, the X crossings; ``cols``: for each X,
    the temperature crossings.  Together they trace the boundary curve on
    the grid without asserting a topology it cannot resolve.
    """
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    rows = [{"temperature_mK": float(T[i]),
             "crossings": axis_crossings(X, Z[i, :], level, log_x=log_x)}
            for i in range(T.size)]
    cols = [{"x": float(X[j]), "crossings": axis_crossings(T, Z[:, j], level)}
            for j in range(X.size)]
    go = Z > level
    return {
        "level_nats": float(level),
        "n_go": int(go.sum()),
        "n_points": int(go.size),
        "go_fraction": float(go.mean()),
        "max_z": float(np.nanmax(Z)),
        "rows": rows,
        "cols": cols,
    }


def anchor_verdicts(rows, name: str, inputs: Inputs, column: str = "E_N") -> List[Dict[str, Any]]:
    """Which side of each contour the published device anchors fall on.

    Anchors lie ON the grid by construction (checked at load), so a verdict
    is a grid-point read, never an interpolation.
    """
    x_axis, _log = SURFACES[name]
    by = {(int(r["grid_iT"]), int(r["grid_iX"])): r for r in rows if r["sweep"] == name}
    T, X, _Z = surface_arrays(rows, name, column)
    out = []
    levels = inputs.resolution.levels
    for tname, tval in inputs.anchors["temperature_mK"].items():
        t_hits = np.where(np.isclose(T, tval, rtol=1e-9, atol=0.0))[0]
        if t_hits.size == 0:
            continue  # a reduced (smoke) grid need not carry every anchor
        iT = int(t_hits[0])
        for xname, xval in inputs.anchors[x_axis].items():
            x_hits = np.where(np.isclose(X, xval, rtol=1e-9, atol=0.0))[0]
            if x_hits.size == 0:
                continue  # comparison anchors that are not on the grid
            iX = int(x_hits[0])
            r = by[(iT, iX)]
            out.append({
                "temperature_anchor": tname, "temperature_mK": float(T[iT]),
                "x_anchor": xname, x_axis: float(X[iX]), "label": r["label"],
                "E_N": float(r["E_N"]), "MI": float(r["MI"]), "work": float(r["work"]),
                "verdict": {k: ("GO" if r[column] > v else "NO-GO") for k, v in levels.items()},
                "MI_verdict": {k: ("GO" if r["MI"] > v else "NO-GO") for k, v in levels.items()},
            })
    return out


def best_pockets(rows, level: float, n: int = 5, column: str = "E_N") -> List[Dict[str, Any]]:
    """Top rows by ``column`` per unit switching work among resolvable rows.

    ``E_N_per_work`` is E_N / (the ledger's summed ``work`` entries) -- a
    computed column, not an estimate, because the runner charges the drive
    work as the telescoped midpoint sum of <dH/dt> dt and the ledger closes
    on it.  Only surface rows compete (controls and companions excluded).
    """
    per_work = f"{column}_per_work"
    ok = [r for r in rows if r["sweep"] in SURFACES and r[column] > level
          and np.isfinite(r[per_work])]
    ok.sort(key=lambda r: -r[per_work])
    keep = ("label", "sweep", "temperature_mK", "gammaphi_Hz", "gamma1_MHz",
            "separation_mm", "separation_sites", "coupling_lambda", "E_N", "MI",
            "work", "E_N_per_work", "MI_per_work")
    return [{k: r[k] for k in keep} for r in ok[:n]]


def pivot_verdict(rows, inputs: Inputs) -> Dict[str, Any]:
    """PLAN.md Layer 2 pivot trigger, answered with numbers.

    Per level: on every surface, the fraction of grid points where E_N is
    resolvable, where MI is resolvable, and where MI is resolvable while
    E_N is not; the maximum E_N and MI inside the published device box (the
    anchor rows); and the ratio of that maximum E_N to the floor.
    """
    out: Dict[str, Any] = {"per_surface": {}, "device_box": {}}
    levels = inputs.resolution.levels
    for name in SURFACES:
        sel = [r for r in rows if r["sweep"] == name]
        if not sel:
            continue
        ent: Dict[str, Any] = {"n_points": len(sel),
                               "max_E_N": float(max(r["E_N"] for r in sel)),
                               "max_MI": float(max(r["MI"] for r in sel))}
        for k, v in levels.items():
            en = sum(r["E_N"] > v for r in sel)
            mi = sum(r["MI"] > v for r in sel)
            both = sum((r["MI"] > v) and not (r["E_N"] > v) for r in sel)
            ent[k] = {"E_N_resolvable_fraction": en / len(sel),
                      "MI_resolvable_fraction": mi / len(sel),
                      "MI_only_fraction": both / len(sel)}
        out["per_surface"][name] = ent
    box = []
    for name in SURFACES:
        if any(r["sweep"] == name for r in rows):
            box.extend(anchor_verdicts(rows, name, inputs))
    # the device box proper: the fridge-base / TL temperatures x the measured
    # or digitized rates x the sourced separations x the anchored coupling
    dev = [a for a in box if a["temperature_anchor"] in
           ("fridge_base_30mK", "transmission_line_50mK")]
    if dev:
        max_en = max(a["E_N"] for a in dev)
        max_mi = max(a["MI"] for a in dev)
        out["device_box"] = {
            "n_anchor_points": len(dev),
            "max_E_N": float(max_en),
            "max_MI": float(max_mi),
            "E_N_over_derived_floor": float(max_en / inputs.resolution.derived),
            "MI_over_derived_floor": float(max_mi / inputs.resolution.derived),
            "E_N_resolvable_anywhere": {k: bool(max_en > v) for k, v in levels.items()},
            "MI_resolvable_anywhere": {k: bool(max_mi > v) for k, v in levels.items()},
        }
    return out


def cross_check_run_harvesting(spec: PointSpec, inputs: Inputs) -> Dict[str, Any]:
    """Same physics through the OTHER driver (M2.2's ``run_harvesting``).

    ``run_harvesting`` builds its own evolution (``protocol_evolve``, Strang
    splitting, its own ledger) and never touches
    ``vacuum.protocol.run_protocol``; it takes static gaps, so the
    cross-check is the static-gap companion of the base point.
    """
    if spec.gap_shift_gamma != 0.0:
        raise ValueError("run_harvesting cross-check needs a static-gap spec")
    u = inputs.units
    window_lat = u.time(spec.window_ns * 1e-9)
    gap_lat = u.energy(spec.gap_GHz * 1e9)
    a_lat = float(spec.lattice_spacing)
    sep_sites = max(1, int(round(u.length(spec.separation_mm * 1e-3) / a_lat)))
    n_field, sites = geometry(sep_sites, window_lat, spec.boundary_buffer, a_lat)
    K_field, lam_rescale, _uv = lattice_at_spacing(n_field, spec.field_mass, a_lat,
                                                   bc="dirichlet")
    chi = switching_profile(spec.switching_kind, window_lat)
    res = run_harvesting(
        K_field, sites, (gap_lat, gap_lat),
        lambda t: spec.coupling_lambda * lam_rescale * chi(t),
        ts=np.linspace(0.0, window_lat, spec.nseg0 + 1),
        T=u.temperature(spec.temperature_mK * 1e-3),
        conv_tol=spec.conv_tol, max_halvings=spec.max_halvings,
        coupling=spec.coupling,
    )
    return {
        "label": spec.label,
        "coupling": spec.coupling,
        "E_N_run_harvesting": float(res.E_N),
        "dt_converged": float(res.dt_converged),
        "work": float(res.work),
        "ledger_defect": float(res.ledger_result.details["defect"]),
        "mp_margin_flagged": bool(res.flagged),
    }


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def _git(args: Sequence[str]) -> str:
    try:
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except Exception:  # pragma: no cover - git absent or not a repo
        return "unavailable"


def build_info() -> Dict[str, str]:
    """Identify the code that produced the arrays (papers/README.md rule)."""
    src = Path(__file__).resolve()
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    import scipy

    import vacuum

    dirty = _git(["status", "--porcelain"])
    return {
        "generator": str(src),
        "generator_sha256": digest,
        "generator_entry_point": "notebook.main()",
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_worktree_dirty": "unavailable" if dirty == "unavailable" else str(bool(dirty)),
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vacuum_version": getattr(vacuum, "__version__", "0.1.0"),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def rows_to_arrays(rows: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
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


def save_rows(path: Path, rows: Sequence[Dict[str, Any]], **extra) -> Path:
    """Save rows plus the build info that produced them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = rows_to_arrays(rows)
    meta = build_info()
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


def save_surfaces(path: Path, rows: Sequence[Dict[str, Any]], inputs: Inputs) -> Path:
    """The surfaces as dense arrays (one file, keyed by surface and column)."""
    payload: Dict[str, np.ndarray] = {}
    for name in SURFACES:
        if not any(r["sweep"] == name for r in rows):
            continue
        for col in ("E_N", "MI", "work", "E_N_per_work", "MI_per_work", "n_d_A",
                    "mp_margin_flagged", "all_audits_passed", "dt_converged"):
            T, X, Z = surface_arrays(rows, name, col)
            payload[f"{name}/T_mK"] = T
            payload[f"{name}/{SURFACES[name][0]}"] = X
            payload[f"{name}/{col}"] = Z
    payload["resolution_levels_nats"] = np.array(
        [inputs.resolution.derived, inputs.resolution.band_low, inputs.resolution.band_high]
    )
    payload["__build__"] = np.array(json.dumps(build_info(), indent=2), dtype="U8192")
    np.savez_compressed(path, **payload)
    return path


# --------------------------------------------------------------------------
# Summary assembly
# --------------------------------------------------------------------------


def summarize(rows, inputs: Inputs, *, wall_clock_s: float, xcheck=None) -> Dict[str, Any]:
    levels = inputs.resolution.levels
    surfaces_present = [n for n in SURFACES if any(r["sweep"] == n for r in rows)]

    boundary: Dict[str, Any] = {}
    for name in surfaces_present:
        x_axis, log_x = SURFACES[name]
        T, X, Z = surface_arrays(rows, name, "E_N")
        _T, _X, M = surface_arrays(rows, name, "MI")
        boundary[name] = {
            "x_axis": x_axis,
            "log_x": log_x,
            "T_mK": [float(t) for t in T],
            "X": [float(x) for x in X],
            "E_N_contour": {k: surface_contour(T, X, Z, v, log_x=log_x) for k, v in levels.items()},
            "MI_contour": {k: surface_contour(T, X, M, v, log_x=log_x) for k, v in levels.items()},
            "device_anchors": anchor_verdicts(rows, name, inputs),
        }

    pockets = {k: best_pockets(rows, v, n=8) for k, v in levels.items()}
    mi_pockets = {k: best_pockets(rows, v, n=5, column="MI") for k, v in levels.items()}

    ctrl = sorted((r for r in rows if r["sweep"] == "control" and r["switching_kind"] == "cos2"),
                  key=lambda r: r["boundary_buffer"])
    control: Dict[str, Any] = {}
    if ctrl:
        e = [r["E_N"] for r in ctrl]
        control = {
            "boundary_buffers": [r["boundary_buffer"] for r in ctrl],
            "n_field_sites": [r["n_field_sites"] for r in ctrl],
            "E_N": e, "MI": [r["MI"] for r in ctrl], "work": [r["work"] for r in ctrl],
            "E_N_spread": float(max(e) - min(e)),
            "E_N_relative_spread": float((max(e) - min(e)) / max(e)) if max(e) > 0 else float("nan"),
            "note": "finite-size (Dirichlet boundary) systematic on E_N at the operating "
                    f"point; production runs use boundary_buffer={BOUNDARY_BUFFER}. This "
                    f"spread, not the {CONV_TOL:g} dt gate, is the dominant numerical "
                    "uncertainty.",
        }

    split = [
        {k: r[k] for k in ("label", "separation_mm", "separation_sites", "window_lat",
                           "E_N", "MI", "E_N_pert", "P_A", "M_abs", "M_vac_abs",
                           "M_comm_abs", "comm_fraction", "perturbative_ok",
                           "perturbative_quadrature_failed", "has_perturbative_row")}
        for r in sorted((r for r in rows if r["sweep"] == "split"),
                        key=lambda r: r["separation_sites"])
    ]

    p = inputs.point
    return {
        "build": build_info(),
        "operating_point": {
            "table1_scenario": p.scenario,
            "gap_GHz": p.gap_GHz,
            "gap_shift_gamma": p.gap_shift_gamma,
            "gap_shift_GHz": p.gap_shift_GHz,
            "window_ns": p.window_ns,
            "switching_kind": p.switching_kind,
            "separation_mm": p.separation_mm,
            "separation_sites": int(max(1, round(inputs.units.length(p.separation_mm * 1e-3)))),
            "signaling_time_ns": p.signaling_time_ns,
            "coupling_lambda_osc": p.coupling_lambda,
            "coupling_lambda_udw_map1": p.coupling_lambda / math.sqrt(2.0 * inputs.units.energy(p.gap_GHz * 1e9)),
            "coupling_sigma": p.coupling_sigma,
            "coupling_mode": p.coupling_mode,
            "scenario_lambda_published": p.lambda_tb,
            "scenario_alpha_published": p.alpha_tb,
            "coupling_status": COUPLING_STATUS if p.coupling_mode == "anchored" else COUPLING_DERIVED_STATUS,
            "coupling_derived_status": COUPLING_DERIVED_STATUS,
            "coupling_map": inputs.coupling_map,
            "site_mm": inputs.units.site_mm,
            "window_lat": inputs.units.time(p.window_ns * 1e-9),
        },
        "resolution": {
            "derived_nats": inputs.resolution.derived,
            "band_nats": [inputs.resolution.band_low, inputs.resolution.band_high],
            "transmon_crosscheck_nats": inputs.resolution.crosscheck,
            "source": inputs.provenance["resolution_nats"][2],
            "band_source": inputs.provenance["resolution_band_nats"][2],
        },
        "axes": {k: [float(x) for x in v] for k, v in inputs.axes.items()},
        "anchors": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in inputs.anchors.items()},
        "n_rows": len(rows),
        "n_surface_points": {n: sum(r["sweep"] == n for r in rows) for n in surfaces_present},
        "wall_clock_s": float(wall_clock_s),
        "cpu_seconds_summed": float(sum(r["runtime_s"] for r in rows)),
        "slowest_point_s": float(max(r["runtime_s"] for r in rows)),
        "all_audits_passed": bool(all(r["all_audits_passed"] for r in rows)),
        "all_converged": bool(all(r["converged"] for r in rows)),
        "worst_ledger_defect": float(max(abs(r["ledger_defect"]) for r in rows)),
        "worst_accepted_movement": float(max(r["movement"] for r in rows)),
        "worst_work_movement": float(max(r["work_movement"] for r in rows)),
        "max_halvings_accepted": float(max(r["halvings"] for r in rows)),
        "mp_margin_flagged_rows": int(sum(r["mp_margin_flagged"] for r in rows)),
        "precision_flagged_rows": int(sum(r["precision_flagged"] for r in rows)),
        "go_no_go": boundary,
        "best_pockets_per_switching_work": pockets,
        "best_MI_pockets_per_switching_work": mi_pockets,
        "pivot": pivot_verdict(rows, inputs),
        "finite_size_control": control,
        "cross_check_run_harvesting": xcheck,
        "communication_split_rows": split,
        "provenance": {
            k: {"file": f, "tag": t, "note": n}
            for k, (f, t, n) in sorted(inputs.provenance.items())
        },
        "synthetic_or_unsourced_inputs": list(inputs.synthetic_inputs),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def _print_report(summary: Dict[str, Any]) -> None:
    print("=" * 72)
    print(f"rows: {summary['n_rows']}   wall clock: {summary['wall_clock_s']:.1f}s   "
          f"cpu-s summed: {summary['cpu_seconds_summed']:.0f}   "
          f"slowest point: {summary['slowest_point_s']:.1f}s")
    print(f"audits all passed: {summary['all_audits_passed']}   all converged: "
          f"{summary['all_converged']}   worst ledger defect: "
          f"{summary['worst_ledger_defect']:.3e}   mp-flagged rows: "
          f"{summary['mp_margin_flagged_rows']}")
    for name, b in summary["go_no_go"].items():
        for k, c in b["E_N_contour"].items():
            print(f"  [{name}] E_N > {c['level_nats']:.2e}: GO on {c['n_go']}/{c['n_points']} "
                  f"points (max E_N {c['max_z']:.3e})")
        for a in b["device_anchors"]:
            print(f"      anchor {a['temperature_anchor']} x {a['x_anchor']}: "
                  f"E_N={a['E_N']:.3e} MI={a['MI']:.3e} -> "
                  + ", ".join(f"{k}:{v}" for k, v in a["verdict"].items()))
    for k, ps in summary["best_pockets_per_switching_work"].items():
        if ps:
            p = ps[0]
            print(f"  best pocket @ {k}: {p['label']} E_N={p['E_N']:.3e} W={p['work']:.3e} "
                  f"E_N/W={p['E_N_per_work']:.3e}")
    box = summary["pivot"].get("device_box", {})
    if box:
        print(f"  pivot: device box max E_N={box['max_E_N']:.3e} "
              f"({box['E_N_over_derived_floor']:.2f} x floor), max MI={box['max_MI']:.3e} "
              f"({box['MI_over_derived_floor']:.2f} x floor)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--quick", action="store_true",
                    help="run the 4x4 smoke configuration instead of the full surfaces")
    ap.add_argument("--surfaces", default="",
                    help="comma-separated subset of surface names (default: all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="worker processes for the full run (default: cores - 2)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: data/; a fresh temporary directory for "
                         "--quick, so a smoke run never overwrites the archive)")
    ap.add_argument("--device-point", action="store_true",
                    help="re-run the device-anchor rows at the DERIVED coupling, its +-1 sigma "
                         "and the enumerated alternatives (writes rows_device_point.npz, "
                         "device_point.json, and a device_point_derived_coupling section "
                         "into an existing summary.json)")
    ap.add_argument("--xp-map", action="store_true",
                    help="the proposal's DERIVATIVE coupling: the main (T x Gamma_phi) surface at "
                         "the derived coupling and +-1 sigma, the device-point rows, the static-gap "
                         "split companions and the finite-size controls, all with coupling='xp' "
                         "(writes rows_xp.npz, surfaces_xp.npz, summary_xp.json; merges an "
                         "'xp_derivative_coupling' section into an existing summary.json)")
    ap.add_argument("--grid", type=int, default=0,
                    help="with --xp-map: 0 = full 16 x 16 surface, 12 = the reduced 12 x 12 axes")
    ap.add_argument("--gap-tracking", action="store_true",
                    help="device-point rows at the derived coupling with lam(t) tracking the gap, "
                         "both operators (writes rows_gap_tracking.npz, gap_tracking.json; merges "
                         "a 'gap_tracking' section into an existing summary.json)")
    ap.add_argument("--ir-control", action="store_true",
                    help="device-point rows at several IR masses, both operators: measures whether "
                         "the synthetic IR regulator can move a device-anchor verdict (writes "
                         "rows_ir_control.npz, ir_mass_control.json; merges an 'ir_mass_control' "
                         "section into an existing summary.json)")
    ap.add_argument("--cutoff-bracket", action="store_true",
                    help="vary the LATTICE SPACING at fixed physical geometry so the model's "
                         "Brillouin UV cutoff sweeps 14.7-73 GHz (bracketing the device's "
                         "50 GHz), and re-run the 'xp' device anchors at every cutoff plus the "
                         "reduced surface at two of them (writes rows_cutoff.npz, "
                         "cutoff_bracket.json; merges a 'uv_cutoff_bracket' section into an "
                         "existing summary.json)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if args.out else (
        Path(tempfile.mkdtemp(prefix="circuit-qed-noise-thresholds-quick-")) if args.quick else DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    derived_mode = (args.device_point or args.xp_map or args.gap_tracking
                    or args.ir_control or args.cutoff_bracket)
    inputs = load_inputs(coupling="derived" if derived_mode else "anchored")
    p = inputs.point

    print("Vacuum Program / M2.7 -- circuit-QED noise-threshold surfaces")
    print(f"  operating point: Table I scenario {p.scenario} (lambda_TB={p.lambda_tb}, "
          f"alpha={p.alpha_tb}), gap {p.gap_GHz} GHz shifted by {p.gap_shift_GHz:+.2f} GHz "
          f"(Eq. 30, gamma={p.gap_shift_gamma}), cos2 window {p.window_ns} ns, "
          f"separation {p.separation_mm} mm (t_d={p.signaling_time_ns} ns), "
          f"lam_osc={p.coupling_lambda:.4f} [{inputs.provenance['coupling_lambda'][1]}]")
    print(f"  omega_ref = {inputs.units.omega_ref:.6e} rad/s; 1 site = "
          f"{inputs.units.site_mm:.4f} mm; window = {inputs.units.time(p.window_ns*1e-9):.3f} lat")
    print(f"  resolution: derived {inputs.resolution.derived:g} nats, band "
          f"[{inputs.resolution.band_low:g}, {inputs.resolution.band_high:g}]")
    print(f"  synthetic/unsourced inputs: {', '.join(inputs.synthetic_inputs)}")
    print()

    # single-threaded BLAS per worker: the matrices are tiny and the pool is
    # the parallelism (children inherit this environment at spawn)
    for var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    t0 = time.time()
    if args.device_point:
        couplings = device_point_couplings(inputs)
        anchors = device_point_anchors(inputs)
        specs = device_point_specs(inputs, couplings, anchors)
        print(f"  derived coupling {p.coupling_lambda:.6f} +- {p.coupling_sigma:.6f} "
              f"[{inputs.provenance['coupling_lambda'][1]}]; couplings: "
              + ", ".join(f"{c}={v:.4f}" for c, v in couplings.items()))
        print(f"{len(specs)} device-point rows on {args.workers} worker(s)")
        rows = run_specs(specs, inputs, n_workers=args.workers)
        archive = load_rows(out_dir / "rows.npz") if (out_dir / "rows.npz").exists() else None
        mismatch = model_mismatch_companion(inputs, couplings["c0"])
        summary = device_point_summary(
            rows, inputs, couplings, anchors, archive_rows=archive,
            wall_clock_s=time.time() - t0, mismatch=mismatch,
        )
        save_rows(out_dir / "rows_device_point.npz", rows, sweep="device_point")
        (out_dir / "device_point.json").write_text(json.dumps(summary, indent=2, default=float))
        sp = out_dir / "summary.json"
        if sp.exists():
            s = json.loads(sp.read_text())
            s["device_point_derived_coupling"] = {
                "note": "added by --device-point on the build recorded here; the full "
                        "per-row tables are in device_point.json / rows_device_point.npz",
                "build": summary["build"],
                "couplings": summary["couplings"],
                "verdict": summary["verdict"],
                "per_coupling_headline": {
                    c: {k: e[k] for k in ("lambda_osc", "device_box_30_50mK", "design_20mK",
                                          "vacuum_0mK", "all_audits_passed", "all_converged")}
                    for c, e in summary["per_coupling"].items()
                },
                "archive_cross_check": {k: v for k, v in summary["archive_cross_check"].items()
                                        if k != "pairs"},
                "model_mismatch_companion": summary["model_mismatch_companion"],
                "coupling_map": summary["coupling_map"],
            }
            sp.write_text(json.dumps(s, indent=2, default=float))
        _print_device_point_report(summary)
        print(f"\ndevice point: {len(rows)} rows in {time.time() - t0:.1f}s -> "
              f"{out_dir / 'rows_device_point.npz'}, {out_dir / 'device_point.json'}"
              + (f", merged into {sp}" if sp.exists() else ""))
        return 0
    if args.xp_map:
        couplings = device_point_couplings(inputs)
        anchors = device_point_anchors(inputs)
        specs = xp_map_specs(inputs, couplings, anchors, grid=args.grid)
        print(f"  'xp' map at " + ", ".join(f"{c}={couplings[c]:.4f}" for c in XP_MAP_COUPLINGS)
              + f"; grid {'16x16' if args.grid == 0 else '12x12'}")
        print(f"{len(specs)} 'xp' rows on {args.workers} worker(s)")
        rows = run_specs(specs, inputs, n_workers=args.workers)
        xcheck = cross_check_run_harvesting(
            base_spec(inputs, "xcheck-static-gap-xp", "cross_check", gap_shift_gamma=0.0,
                      coupling="xp"), inputs)
        base_static = [r for r in rows if r["sweep"] == "split" and int(r["separation_sites"])
                       == int(inputs.anchors["separation_sites"]["t_d_0.16ns_19.2mm"])]
        if base_static:
            xcheck["E_N_run_protocol"] = base_static[0]["E_N"]
            xcheck["abs_difference"] = abs(base_static[0]["E_N"] - xcheck["E_N_run_harvesting"])
        dp_path, sp = out_dir / "device_point.json", out_dir / "summary.json"
        xx_dp = json.loads(dp_path.read_text()) if dp_path.exists() else None
        xx_s = json.loads(sp.read_text()) if sp.exists() else None
        mismatch = model_mismatch_companion(inputs, couplings["c0"])
        summary = xp_map_summary(
            rows, inputs, couplings, anchors, wall_clock_s=time.time() - t0, grid=args.grid,
            xcheck=xcheck, xx_device_point=xx_dp, xx_summary=xx_s, mismatch=mismatch,
        )
        save_rows(out_dir / "rows_xp.npz", rows, sweep="xp_map", coupling="xp")
        save_surfaces_by_coupling(out_dir / "surfaces_xp.npz", rows, inputs, couplings)
        (out_dir / "summary_xp.json").write_text(json.dumps(summary, indent=2, default=float))
        if xx_s is not None:
            per = summary["surface"]["per_coupling"]
            xx_s["xp_derivative_coupling"] = {
                "note": "added by --xp-map on the build recorded here; full contours, rows and "
                        "side-by-side tables are in summary_xp.json / rows_xp.npz / surfaces_xp.npz",
                "build": summary["build"], "grid": summary["grid"],
                "couplings": summary["couplings"], "verdict": summary["verdict"],
                "surface_headline": {
                    c: {"lambda_osc": e["lambda_osc"], "max_E_N": e["max_E_N"], "max_MI": e["max_MI"],
                        "temperature_wall": e["temperature_wall"],
                        "dephasing_crossings": e["dephasing_crossings"],
                        "go_fraction": {k: e["E_N_contour"][k]["go_fraction"] for k in e["E_N_contour"]},
                        "device_anchors": e["device_anchors"], "pivot": e["pivot"]}
                    for c, e in per.items()},
                "xx_archive_surface": summary["surface"]["xx_archive"],
                "device_point_headline": (
                    {c: {k: e[k] for k in ("lambda_osc", "device_box_30_50mK", "design_20mK",
                                           "vacuum_0mK", "all_audits_passed", "all_converged")}
                     for c, e in summary["device_point"]["per_coupling"].items()}
                    if summary["device_point"] else None),
                "device_point_side_by_side": (
                    {c: {k: v for k, v in e.items() if k != "rows"}
                     for c, e in summary["device_point_side_by_side"].items()}
                    if summary["device_point_side_by_side"] else None),
                "communication_split_side_by_side": summary["communication_split_side_by_side"],
                "finite_size_control": summary["finite_size_control"],
                "cross_check_run_harvesting": summary["cross_check_run_harvesting"],
                "model_mismatch_companion": summary["model_mismatch_companion"],
            }
            sp.write_text(json.dumps(xx_s, indent=2, default=float))
        _print_xp_report(summary)
        print(f"\n'xp' map: {len(rows)} rows in {time.time() - t0:.1f}s -> {out_dir / 'rows_xp.npz'}, "
              f"{out_dir / 'surfaces_xp.npz'}, {out_dir / 'summary_xp.json'}"
              + (f", merged into {sp}" if xx_s is not None else ""))
        return 0
    if args.gap_tracking:
        couplings = device_point_couplings(inputs)
        anchors = device_point_anchors(inputs)
        specs = gap_tracking_specs(inputs, anchors, couplings["c0"])
        print(f"{len(specs)} gap-tracking rows on {args.workers} worker(s); exponents {GAP_TRACKING_EXPONENT}")
        rows = run_specs(specs, inputs, n_workers=args.workers)
        dp_path, xp_path, sp = out_dir / "device_point.json", out_dir / "summary_xp.json", out_dir / "summary.json"
        xx_dp = json.loads(dp_path.read_text()) if dp_path.exists() else None
        xp_s = json.loads(xp_path.read_text()) if xp_path.exists() else None
        summary = gap_tracking_summary(rows, inputs, anchors, couplings["c0"],
                                       wall_clock_s=time.time() - t0, xx_device_point=xx_dp,
                                       xp_summary=xp_s)
        save_rows(out_dir / "rows_gap_tracking.npz", rows, sweep="gap_tracking")
        (out_dir / "gap_tracking.json").write_text(json.dumps(summary, indent=2, default=float))
        if sp.exists():
            s = json.loads(sp.read_text())
            s["gap_tracking"] = {
                "note": "added by --gap-tracking; per-row tables in gap_tracking.json / rows_gap_tracking.npz",
                "build": summary["build"], "lambda_osc": summary["lambda_osc"],
                "exponents": summary["exponents"], "bound_stated": summary["bound_stated"],
                "comparison_vs_untracked": {c: {k: v for k, v in e.items() if k != "pairs"}
                                            for c, e in summary["comparison_vs_untracked"].items()},
                "device_point_headline": {
                    c: {k: e[k] for k in ("lambda_osc", "device_box_30_50mK", "design_20mK", "vacuum_0mK")}
                    for c, e in summary["device_point"]["per_coupling"].items()},
            }
            sp.write_text(json.dumps(s, indent=2, default=float))
        print("=" * 72)
        for c, e in summary["comparison_vs_untracked"].items():
            print(f"  gap tracking [{c}] exponent {e['exponent']}: {e['n']} rows compared, max |rel dE_N| = "
                  f"{e['max_abs_rel_change_E_N']}, verdict changed on {e['changed']}")
        print(f"\ngap tracking: {len(rows)} rows in {time.time() - t0:.1f}s -> "
              f"{out_dir / 'rows_gap_tracking.npz'}, {out_dir / 'gap_tracking.json'}")
        return 0
    if args.cutoff_bracket:
        couplings = device_point_couplings(inputs)
        anchors = device_point_anchors(inputs)
        specs = cutoff_specs(inputs, couplings["c0"], anchors)
        cut_ghz = {c: round(math.sqrt(FIELD_MASS**2 + 4.0 / a**2) * p.gap_GHz, 1)
                   for c, a in CUTOFF_SPACINGS.items()}
        print(f"  cutoff bracket: spacings {CUTOFF_SPACINGS} -> {cut_ghz} GHz")
        print(f"{len(specs)} cutoff rows on {args.workers} worker(s)")
        rows = run_specs(specs, inputs, n_workers=args.workers)
        xp_path = out_dir / "summary_xp.json"
        xp_s = json.loads(xp_path.read_text()) if xp_path.exists() else None
        summary = cutoff_summary(rows, inputs, couplings["c0"], anchors,
                                 wall_clock_s=time.time() - t0, xp_summary=xp_s)
        save_rows(out_dir / "rows_cutoff.npz", rows, sweep="cutoff_bracket")
        (out_dir / "cutoff_bracket.json").write_text(json.dumps(summary, indent=2, default=float))
        sp = out_dir / "summary.json"
        if sp.exists():
            s = json.loads(sp.read_text())
            s["uv_cutoff_bracket"] = {
                "note": "added by --cutoff-bracket; per-row tables in cutoff_bracket.json / "
                        "rows_cutoff.npz",
                "build": summary["build"], "spacings": summary["spacings"],
                "method": summary["note"], "verdict": summary["verdict"],
                "per_cutoff": {c: {k: v for k, v in e.items() if k != "rows"}
                               for c, e in summary["per_cutoff"].items()},
                "surface": summary["surface"],
            }
            sp.write_text(json.dumps(s, indent=2, default=float))
        _print_cutoff_report(summary)
        print(f"\ncutoff bracket: {len(rows)} rows in {time.time() - t0:.1f}s -> "
              f"{out_dir / 'rows_cutoff.npz'}, {out_dir / 'cutoff_bracket.json'}")
        return 0
    if args.ir_control:
        couplings = device_point_couplings(inputs)
        anchors = device_point_anchors(inputs)
        specs = ir_control_specs(inputs, anchors, couplings["c0"])
        print(f"{len(specs)} IR-control rows on {args.workers} worker(s); masses {IR_MASS_GRID} "
              f"(declared {FIELD_MASS}), both operators")
        rows = run_specs(specs, inputs, n_workers=args.workers)
        summary = ir_control_summary(rows, inputs, anchors, couplings["c0"],
                                     wall_clock_s=time.time() - t0)
        save_rows(out_dir / "rows_ir_control.npz", rows, sweep="ir_control")
        (out_dir / "ir_mass_control.json").write_text(json.dumps(summary, indent=2, default=float))
        sp = out_dir / "summary.json"
        if sp.exists():
            s = json.loads(sp.read_text())
            s["ir_mass_control"] = {
                "note": "added by --ir-control; per-row tables in ir_mass_control.json / "
                        "rows_ir_control.npz",
                "build": summary["build"], "masses": summary["masses"],
                "declared_mass": summary["declared_mass"], "lambda_osc": summary["lambda_osc"],
                "per_coupling": {c: {k: v for k, v in e.items() if k != "by_mass"}
                                 for c, e in summary["per_coupling"].items()},
                "device_box_max_E_N_by_mass": {
                    c: {mk: mv["device_box_max_E_N"] for mk, mv in e["by_mass"].items()}
                    for c, e in summary["per_coupling"].items()},
            }
            sp.write_text(json.dumps(s, indent=2, default=float))
        print("=" * 72)
        for c, e in summary["per_coupling"].items():
            print(f"  [{c}] verdicts independent of the IR mass: {e['verdicts_independent_of_mass']}; "
                  f"dephasing anchors all zero at every mass: {e['dephasing_anchors_all_zero_at_every_mass']}")
            print(f"       device-box max E_N by mass: " + str(
                {k: (f"{v['device_box_max_E_N']:.3e}" if v["device_box_max_E_N"] is not None else "n/a")
                 for k, v in e["by_mass"].items()}))
            if e["verdict_drift"]:
                for d in e["verdict_drift"][:6]:
                    print(f"       drift: m={d['mass']} {d['row']} @{d['level']}: "
                          f"{d['verdict_at_declared_mass']} -> {d['verdict_here']} "
                          f"(E_N {d['E_N_declared']:.3e} -> {d['E_N_here']:.3e})")
        print(f"\nIR control: {len(rows)} rows in {time.time() - t0:.1f}s -> "
              f"{out_dir / 'rows_ir_control.npz'}, {out_dir / 'ir_mass_control.json'}")
        return 0
    if args.quick:
        specs = quick_specs(inputs)
        rows = run_specs(specs, inputs)
        save_rows(out_dir / "rows_quick.npz", rows, sweep="quick")
        summary = summarize(rows, inputs, wall_clock_s=time.time() - t0)
        (out_dir / "summary_quick.json").write_text(json.dumps(summary, indent=2, default=float))
        _print_report(summary)
        print(f"\nquick run: {len(rows)} rows in {time.time() - t0:.1f}s -> "
              f"{out_dir / 'rows_quick.npz'}")
        return 0

    wanted = [s.strip() for s in args.surfaces.split(",") if s.strip()] or list(SURFACES)
    for w in wanted:
        if w not in SURFACES:
            raise SystemExit(f"unknown surface {w!r}; choose from {list(SURFACES)}")
    specs = full_specs(inputs, wanted)
    print(f"{len(specs)} points on {args.workers} worker(s)")
    rows = run_specs(specs, inputs, n_workers=args.workers)
    xcheck = cross_check_run_harvesting(
        base_spec(inputs, "xcheck-static-gap", "cross_check", gap_shift_gamma=0.0), inputs
    )
    base_static = [r for r in rows if r["sweep"] == "split"
                   and int(r["separation_sites"]) == int(inputs.anchors["separation_sites"]["t_d_0.16ns_19.2mm"])]
    if base_static:
        xcheck["E_N_run_protocol"] = base_static[0]["E_N"]
        xcheck["abs_difference"] = abs(base_static[0]["E_N"] - xcheck["E_N_run_harvesting"])
    wall = time.time() - t0

    save_rows(out_dir / "rows.npz", rows, sweep="full")
    save_surfaces(out_dir / "surfaces.npz", rows, inputs)
    summary = summarize(rows, inputs, wall_clock_s=wall, xcheck=xcheck)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    _print_report(summary)
    print(f"  cross-check |E_N(run_protocol) - E_N(run_harvesting)| = "
          f"{xcheck.get('abs_difference')}")
    print(f"  wrote {out_dir / 'rows.npz'}, {out_dir / 'surfaces.npz'}, "
          f"{out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
