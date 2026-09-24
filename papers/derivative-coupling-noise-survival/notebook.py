"""Does derivative-coupling harvesting survive realistic noise?  (Layer 2 target)

papers/derivative-coupling-noise-survival -- the analysis behind MANIFEST.md.
Re-runnable top to bottom, headless, no hidden state:

    .venv/bin/python papers/derivative-coupling-noise-survival/notebook.py            # full map
    .venv/bin/python papers/derivative-coupling-noise-survival/notebook.py --quick    # 4-point smoke

The question (PLAN.md, Layer 2 discovery targets): the derivative-coupling
result of Teixido-Bonfill & Martin-Martinez, PRD 110, 105016 (2024)
[arXiv:2406.14637; restated for circuit QED in arXiv:2505.01516] -- that
even CAUSALLY CONNECTED detectors draw their entanglement primarily from
the field's correlations rather than from communication -- would relax the
brutal requirement of strict spacelike separation.  Does it survive
realistic noise?

The map.  One causally connected pair on the 1+1 massless lattice field
(the native spacetime of a microwave waveguide), derivative-coupled
(``attach_detectors(..., coupling='xp')``: x_d couples to the field
momentum pi = d_t phi), compact cos^2 switchings meeting on the light
cone (t_B - t_A = d), swept over three imperfections from
``vacuum.detectors.imperfections``:

  * field (waveguide) temperature T  -- ``thermal_state_cov`` initial field;
  * detector energy relaxation kappa -- loss into a bath at the SAME T
    (nbar = 1/(e^{Omega/T} - 1)), Strang-interleaved with the drive;
  * pure dephasing gamma_phi          -- the isotropic Gaussian dephasing
    channel (module docstring of imperfections.py, item 2).

Every point runs THROUGH ``vacuum.protocol.run_protocol`` (M-I.4): the
passivity audit on the claimed T = 0 ground state, precision audit on the
reported detector block, mandatory ledger closure, the spec-M2.2
dt-halving gate (on E_N, the partial-transpose margin and both
occupations -- the margin is continuous, so a dead point cannot "converge"
trivially at E_N = 0), and the mp-margin rule on E_N.

Two quantities per point:

  * E_N        -- the exact (nonperturbative) harvested log-negativity;
  * the M_vac / M_comm split of the pair term at that field temperature --
    the M2.5 estimator (Tjoa & Martin-Martinez, PRD 104, 125005 (2021))
    on the THERMAL pi-pi kernel: the commutator part is state-independent,
    the anti-commutator part grows with T, so |M_vac|/|M| is a function of
    T alone.  Detector-side loss and dephasing act on the pair state after
    (and, Trotterized, during) the interaction as Gaussian channels that
    rescale the pair moment <a_A a_B> without re-weighting where it came
    from: at leading order the split fraction is unchanged by them, which
    the nonperturbative |m| column lets the reader check point by point.

Provenance of the inputs -- READ THIS.  The configuration (lattice size,
separation, switching width, gap, coupling strength, and the noise grid)
is a MODEL configuration chosen in lattice units so that (i) the resonant
field modes sit on the linear part of the lattice dispersion (group
velocity 0.97, so the lattice light cone is the continuum one), (ii) the
noiseless point harvests with a healthy margin |M| - P, and (iii) the
whole map fits in a laptop budget.  It is NOT transcribed from a device
parameter file: none of the ``experiments/`` files is consumed here, and
no number below is a prediction for a specific device.  What this map
claims is a statement about the model.  Turning it into a device
prediction means re-running it on ``experiments/circuit_qed_harvesting``
inputs through the loader, exactly as papers/circuit-qed-noise-thresholds
does -- and inheriting that candidate's caveats about synthetic inputs.

Conventions per docs/API.md throughout (hbar = k_B = 1, lattice units,
nats).  Pure functions over frozen specs; the only side effects are the
files written under data/.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import vacuum
from vacuum.audits import EnergyLedger
from vacuum.core import harmonic_chain_K
from vacuum.detectors import (
    ExpSumKernel,
    UDWDetector,
    attach_detectors,
    communication_split,
    detector_block,
    detector_occupations,
    harvested_log_negativity,
    imperfection_step,
    imperfections_from_rates,
    initial_state,
    switching,
    transition_probability,
    wightman_lattice,
)
from vacuum.detectors.udw import pair_kernels, pair_negativity
from vacuum.protocol import GeneratorStep, run_protocol

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# --------------------------------------------------------------------------
# The configuration (model units; see the provenance paragraph above)
# --------------------------------------------------------------------------

N_FIELD = 60          #: massless Dirichlet chain; walls 24 sites from either detector
SITE_A = 24
SEPARATION = 12       #: d, in sites (= light-crossing time in lattice units)
SWITCH_T = 4.0        #: cos^2 half-width; supports [-4, 4] and [8, 16]
GAP = 0.5             #: detector gap Omega (resonant modes at k ~ 0.5: v_g = cos(k/2) = 0.97)
LAMBDA = 0.1          #: oscillator coupling; lam_UDW = lam/sqrt(2 Omega) = 0.1
FIELD_MASS = 0.0
WINDOW: Tuple[float, float] = (-SWITCH_T, SEPARATION + SWITCH_T)   # [-4, 16]

NSEG0 = 40            #: base grid segments over the window (dt0 = 0.5 / SUBSTEPS)
SUBSTEPS = 4          #: exponential-midpoint substeps per GeneratorStep
CONV_TOL = 1e-9       #: spec M2.2 gate tolerance
MAX_HALVINGS = 10
LEDGER_TOL = 1e-9     #: spec: all protocols
E_N_RESOLVABLE = 1e-4 #: stated measurement floor, as in papers/circuit-qed-noise-thresholds

#: The noise grid.  Temperatures in units of the lattice energy (Omega = 0.5,
#: so T/Omega runs 0 .. 0.2; the perturbative thermal death of this point is
#: between T = 0.10 and 0.12); rates per unit lattice time over a 20-unit
#: window (kappa = 0.02 -> total transmissivity e^{-0.4}; gamma_phi = 1e-5
#: -> integrated dephasing 2e-4, against a T = 0 margin |M| - P = 3.4e-4).
TEMPERATURES = (0.0, 0.02, 0.04, 0.06, 0.08, 0.1)
GAMMA_PHIS = (0.0, 3e-6, 1e-5, 3e-5)
KAPPAS = (0.0, 0.02)

#: quadrature settings of the perturbative split (rtol relaxed: these are
#: fractions, not 1e-12 anchors)
PERTURBATIVE_QUAD = dict(rtol=1e-9, max_doublings=14)

ROW_COLUMNS = (
    "label", "sweep", "coupling", "temperature", "kappa", "gamma_phi",
    "eta_total", "nbar_bath", "dephasing_total",
    "E_N", "resolvable", "survives", "flagged", "regime", "margin_min",
    "n_d_A", "n_d_B", "m_abs", "c_abs",
    "converged", "dt_converged", "halvings", "n_segments", "levels_run",
    "work", "dissipated", "ledger_defect", "ledger_passed", "passivity_passed",
    "precision_passed", "all_audits_passed",
    "P_T", "M_abs_T", "M_vac_abs_T", "M_comm_abs_T", "vac_fraction_T",
    "comm_fraction_T", "N2_T", "E_N_pert_T", "m_over_M",
    "xx_vac_fraction_T", "xx_comm_fraction_T", "xx_N2_T",
    "elapsed_s",
)


@dataclass(frozen=True)
class PointSpec:
    """One map point.  Frozen data; ``coupling`` is 'xp' except for controls."""

    label: str
    sweep: str
    temperature: float
    kappa: float = 0.0
    gamma_phi: float = 0.0
    coupling: str = "xp"
    conv_tol: float = CONV_TOL
    max_halvings: int = MAX_HALVINGS
    nseg0: int = NSEG0


# --------------------------------------------------------------------------
# Geometry, kernels, switching (pure, cached)
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def field_K():
    return harmonic_chain_K(N_FIELD, FIELD_MASS, bc="dirichlet")


def sites() -> Tuple[int, int]:
    return (SITE_A, SITE_A + SEPARATION)


def profiles() -> Tuple[np.ndarray, np.ndarray]:
    F_A = np.zeros(N_FIELD); F_A[SITE_A] = 1.0
    F_B = np.zeros(N_FIELD); F_B[SITE_A + SEPARATION] = 1.0
    return F_A, F_B


def switchings():
    """cos^2 of half-width SWITCH_T at t_A = 0 and t_B = d: full light contact."""
    return switching("cos2", SWITCH_T, 0.0), switching("cos2", SWITCH_T, float(SEPARATION))


def lambda_of_t(t: float) -> np.ndarray:
    chi_A, chi_B = switchings()
    return np.array([LAMBDA * chi_A(t), LAMBDA * chi_B(t)])


# --------------------------------------------------------------------------
# The perturbative split at field temperature T (M2.5 columns)
# --------------------------------------------------------------------------


def thermal_amplitude_triple(T: float):
    """(W_AA, W_BB, W_AB) of the field at temperature T, as exponential sums.

    W_T(dt) = sum_k c_k [(n_k + 1) e^{-i w_k dt} + n_k e^{+i w_k dt}],
    n_k = 1/(e^{w_k/T} - 1) -- the KMS thermal Wightman function of the
    lattice field on the doubled frequency set (as in
    tests/test_nogo_communication.py).  T = 0 returns the vacuum kernels.
    ``pair_kernels(..., coupling='xp')`` turns the triple into the pi-pi
    (derivative) kernels by the c_k -> c_k w_k^2 rule.
    """
    ker = wightman_lattice(field_K())
    F_A, F_B = profiles()
    if T <= 0.0:
        return (ker.smeared(F_A), ker.smeared(F_B), ker.smeared(F_A, F_B))
    w = np.asarray(ker.omega)
    U = np.asarray(ker.U)
    n = 1.0 / np.expm1(w / float(T))

    def thermal(Fx, Fy):
        c = (U.T @ Fx) * (U.T @ Fy) / (2.0 * w)
        return ExpSumKernel(
            np.concatenate([w, -w]),
            np.concatenate([c * (n + 1.0), c * n]).astype(complex),
        )

    return (thermal(F_A, F_A), thermal(F_B, F_B), thermal(F_A, F_B))


@functools.lru_cache(maxsize=None)
def split_columns(T: float, coupling: str) -> Dict[str, float]:
    """M2.5 split of the pair term at field temperature T, for one coupling.

    lam_UDW = LAMBDA / sqrt(2 GAP) is (MAP-1) of tests/test_gate_crossvalidation.py,
    gate-tested for 'xp' in tests/test_gate_xp.py.  Cached: a pure function
    of (T, coupling).
    """
    chi_A, chi_B = switchings()
    F_A, F_B = profiles()
    det_A, det_B = UDWDetector(chi_A, F_A), UDWDetector(chi_B, F_B)
    triple = thermal_amplitude_triple(float(T))
    lam_udw = LAMBDA / math.sqrt(2.0 * GAP)
    ker_AA, _, _ = pair_kernels(triple, det_A, det_B, coupling=coupling)
    P = transition_probability(ker_AA, chi_A, GAP, lam_udw, rtol=PERTURBATIVE_QUAD["rtol"],
                               max_doublings=PERTURBATIVE_QUAD["max_doublings"])
    sp = communication_split(
        triple, det_A, det_B, lam_udw, GAP, coupling=coupling, **PERTURBATIVE_QUAD
    )
    N2 = math.hypot(abs(sp.M), 0.0) - P          # identical detectors: |M| - P
    N = pair_negativity(P, P, sp.M)
    return {
        "P": float(P),
        "M_abs": float(abs(sp.M)),
        "M_vac_abs": float(abs(sp.M_vac)),
        "M_comm_abs": float(abs(sp.M_comm)),
        "vac_fraction": float(sp.vac_fraction),
        "comm_fraction": float(sp.comm_fraction),
        "N2": float(N2),
        "E_N_pert": float(math.log1p(2.0 * N)),
    }


# --------------------------------------------------------------------------
# One fixed-resolution pass THROUGH vacuum.protocol.run_protocol
# --------------------------------------------------------------------------


def _pair_moments(V_AB: np.ndarray, gap: float) -> Tuple[float, float]:
    """|<a_A a_B>| and |<a_A^dag a_B>| of the detector block (Gate A helper form)."""
    D = np.diag([math.sqrt(gap), math.sqrt(gap), 1.0 / math.sqrt(gap), 1.0 / math.sqrt(gap)])
    W = D @ V_AB @ D
    xx, pp, xp = W[:2, :2], W[2:, 2:], W[:2, 2:]
    m = complex(0.5 * (xx[0, 1] - pp[0, 1]), 0.5 * (xp[0, 1] + xp[1, 0]))
    c = complex(0.5 * (xx[0, 1] + pp[0, 1]), 0.5 * (xp[0, 1] - xp[1, 0]))
    return abs(m), abs(c)


def _run_at_level(spec: PointSpec, n_segments: int, V0, K_tot0):
    """run_protocol over n_segments Strang (channel, drive, channel) triples."""
    K = field_K()
    det_modes = (N_FIELD, N_FIELD + 1)
    gaps = (GAP, GAP)
    # run_protocol's clock starts at 0; the physical window starts at
    # WINDOW[0] = -SWITCH_T.  The generator is therefore evaluated at the
    # shifted absolute time t + WINDOW[0] (a callable GeneratorStep receives
    # protocol time, vacuum/protocol.py).
    ts = np.linspace(0.0, WINDOW[1] - WINDOW[0], int(n_segments) + 1)
    noisy = spec.kappa > 0.0 or spec.gamma_phi > 0.0

    def H_of_t(t):
        return attach_detectors(
            K, sites(), gaps, lambda_of_t(t + WINDOW[0]), coupling=spec.coupling
        )

    steps: List[Any] = []
    for i in range(int(n_segments)):
        duration = float(ts[i + 1] - ts[i])
        half = None
        if noisy:
            imp = imperfections_from_rates(
                gaps, 0.5 * duration, spec.temperature, spec.kappa, spec.gamma_phi
            )
            half = imperfection_step(
                det_modes, eta=imp.eta, nbar=imp.nbar, dephasing=imp.dephasing,
                gaps=imp.gaps, label=f"noise{i}",
            )
            steps.append(half)
        steps.append(GeneratorStep(H=H_of_t, duration=duration, n_substeps=SUBSTEPS,
                                   label=f"drive{i}"))
        if noisy:
            steps.append(half)

    ledger = EnergyLedger(K=K_tot0, tol=LEDGER_TOL)
    res = run_protocol(
        V0, K_tot0, steps, ledger=ledger,
        claim_ground_entry=(spec.temperature == 0.0),
        report_modes=[det_modes],
        run_causality=False,        # detector-augmented ordering; causality is the M_comm column
        ledger_tol=LEDGER_TOL,
        strict=True,
    )
    V_AB = detector_block(res.V, N_FIELD)
    neg = harvested_log_negativity(V_AB)
    n_d = detector_occupations(res.V, N_FIELD, np.asarray(gaps))
    observables = {
        "E_N": neg.E_N,
        "margin": float(neg.margin_min),
        "n_d_A": float(n_d[0]),
        "n_d_B": float(n_d[1]),
    }
    return res, neg, n_d, observables, V_AB


class _LevelSchedule:
    """Halving levels with a second-order jump (as in the M2.7 candidate)."""

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


def run_point(spec: PointSpec) -> Dict[str, Any]:
    """Run one map point end to end; returns the ROW_COLUMNS row (pure)."""
    t_start = time.time()
    K = field_K()
    init = initial_state(K, sites(), (GAP, GAP), spec.temperature, strict=True)

    prev: Optional[Dict[str, float]] = None
    movement: Optional[float] = None
    converged = False
    levels_run: List[int] = []
    res = neg = n_d = obs = V_AB = None
    n_segments = spec.nseg0
    schedule = _LevelSchedule(int(spec.max_halvings))
    for level in schedule:
        n_segments = int(spec.nseg0) * 2**level
        levels_run.append(level)
        res, neg, n_d, obs, V_AB = _run_at_level(spec, n_segments, init.V, init.K_tot)
        if prev is not None and levels_run[-2] == level - 1:
            movement = max(abs(obs[k] - prev[k]) for k in obs)
            if movement < float(spec.conv_tol):
                converged = True
                break
            schedule.jump(level, movement, float(spec.conv_tol))
        prev = dict(obs)
    if not converged:
        raise RuntimeError(
            f"point {spec.label!r}: dt-halving gate did not reach {spec.conv_tol:g} "
            f"within {spec.max_halvings} halvings (levels {levels_run}, last movement {movement!r})"
        )

    window = WINDOW[1] - WINDOW[0]
    imp_total = imperfections_from_rates(
        (GAP, GAP), window, spec.temperature, spec.kappa, spec.gamma_phi
    )
    m_abs, c_abs = _pair_moments(V_AB, GAP)
    sp = split_columns(float(spec.temperature), spec.coupling)
    sp_xx = split_columns(float(spec.temperature), "xx")
    ledger_res = res.audits["ledger"]
    precision_ok = all(d["result"].passed for d in res.audits["precision"])
    row: Dict[str, Any] = {
        "label": spec.label,
        "sweep": spec.sweep,
        "coupling": spec.coupling,
        "temperature": float(spec.temperature),
        "kappa": float(spec.kappa),
        "gamma_phi": float(spec.gamma_phi),
        "eta_total": float(imp_total.eta),
        "nbar_bath": float(imp_total.nbar),
        "dephasing_total": float(imp_total.dephasing),
        "E_N": float(neg.E_N),
        "resolvable": bool(neg.E_N > E_N_RESOLVABLE),
        "survives": bool(neg.E_N > 0.0),
        "flagged": bool(neg.flagged),
        "regime": neg.regime,
        "margin_min": float(neg.margin_min),
        "n_d_A": float(n_d[0]),
        "n_d_B": float(n_d[1]),
        "m_abs": float(m_abs),
        "c_abs": float(c_abs),
        "converged": bool(converged),
        "dt_converged": float(window / (n_segments * SUBSTEPS)),
        "halvings": int(levels_run[-1]),
        "n_segments": int(n_segments),
        "levels_run": list(levels_run),
        "work": float(res.work_total),
        "dissipated": float(res.dissipated_total),
        "ledger_defect": float(ledger_res.details["defect"]),
        "ledger_passed": bool(ledger_res.passed),
        "passivity_passed": (
            None if res.audits["passivity_entry"] is None
            else bool(res.audits["passivity_entry"].passed)
        ),
        "precision_passed": bool(precision_ok),
        "all_audits_passed": bool(res.all_passed()),
        "P_T": sp["P"],
        "M_abs_T": sp["M_abs"],
        "M_vac_abs_T": sp["M_vac_abs"],
        "M_comm_abs_T": sp["M_comm_abs"],
        "vac_fraction_T": sp["vac_fraction"],
        "comm_fraction_T": sp["comm_fraction"],
        "N2_T": sp["N2"],
        "E_N_pert_T": sp["E_N_pert"],
        "m_over_M": float(m_abs / sp["M_abs"]) if sp["M_abs"] > 0 else float("nan"),
        "xx_vac_fraction_T": sp_xx["vac_fraction"],
        "xx_comm_fraction_T": sp_xx["comm_fraction"],
        "xx_N2_T": sp_xx["N2"],
        "elapsed_s": float(time.time() - t_start),
    }
    return row


def run_specs(specs: Sequence[PointSpec], verbose: bool = True) -> List[Dict[str, Any]]:
    rows = []
    for spec in specs:
        row = run_point(spec)
        rows.append(row)
        if verbose:
            print(
                f"  {row['label']:<22s} T={row['temperature']:.2f} kappa={row['kappa']:.3g} "
                f"gphi={row['gamma_phi']:.1e}  E_N={row['E_N']:.3e} "
                f"vac={row['vac_fraction_T']:.3f} comm={row['comm_fraction_T']:.3f} "
                f"|m|/|M|={row['m_over_M']:.3f} halv={row['halvings']} "
                f"dt={row['dt_converged']:.2e} ledger={row['ledger_defect']:.1e} "
                f"({row['elapsed_s']:.1f}s)",
                flush=True,
            )
    return rows


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------


def full_specs() -> List[PointSpec]:
    specs: List[PointSpec] = []
    for T in TEMPERATURES:
        for kappa in KAPPAS:
            for gphi in GAMMA_PHIS:
                specs.append(PointSpec(
                    label=f"xp-T{T:g}-k{kappa:g}-g{gphi:g}", sweep="map",
                    temperature=T, kappa=kappa, gamma_phi=gphi,
                ))
    # the amplitude-coupling control at the same light contact, vacuum field
    specs.append(PointSpec(label="xx-control-T0", sweep="control", temperature=0.0,
                           coupling="xx"))
    return specs


def smoke_specs() -> List[PointSpec]:
    """Four points covering the branches: vacuum/thermal x clean/noisy."""
    return [
        PointSpec("smoke-clean", "smoke", 0.0, conv_tol=1e-7, max_halvings=6),
        PointSpec("smoke-thermal", "smoke", 0.1, conv_tol=1e-7, max_halvings=6),
        PointSpec("smoke-noisy", "smoke", 0.0, kappa=0.02, gamma_phi=1e-5,
                  conv_tol=1e-7, max_halvings=6),
        PointSpec("smoke-hot-noisy", "smoke", 0.15, kappa=0.02, gamma_phi=3e-5,
                  conv_tol=1e-7, max_halvings=6),
    ]


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


def save_rows(path: Path, rows: Sequence[Dict[str, Any]], sweep: str) -> None:
    cols: Dict[str, Any] = {}
    for c in ROW_COLUMNS:
        vals = [r[c] for r in rows]
        if c == "levels_run":
            cols[c] = np.array([json.dumps(v) for v in vals])
        elif isinstance(vals[0], str):
            cols[c] = np.array(vals)
        elif vals[0] is None or any(v is None for v in vals):
            cols[c] = np.array([float("nan") if v is None else float(v) for v in vals])
        else:
            cols[c] = np.array(vals)
    meta = {
        "sweep": sweep,
        "generator": "papers/derivative-coupling-noise-survival/notebook.py",
        "vacuum_version": getattr(vacuum, "__version__", "0.1.0"),
        "git_head": _git_head(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "config": dict(N_FIELD=N_FIELD, SITE_A=SITE_A, SEPARATION=SEPARATION,
                       SWITCH_T=SWITCH_T, GAP=GAP, LAMBDA=LAMBDA, FIELD_MASS=FIELD_MASS,
                       WINDOW=list(WINDOW), NSEG0=NSEG0, SUBSTEPS=SUBSTEPS,
                       CONV_TOL=CONV_TOL, LEDGER_TOL=LEDGER_TOL,
                       E_N_RESOLVABLE=E_N_RESOLVABLE,
                       TEMPERATURES=list(TEMPERATURES), GAMMA_PHIS=list(GAMMA_PHIS),
                       KAPPAS=list(KAPPAS)),
        "inputs_provenance": "model configuration in lattice units; no experiments/ file consumed",
    }
    np.savez(path, meta=json.dumps(meta), **cols)


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The headline numbers (written to data/summary.json)."""
    xp = [r for r in rows if r["coupling"] == "xp" and r["sweep"] == "map"]
    def sel(**kw):
        return [r for r in xp if all(abs(r[k] - v) < 1e-12 for k, v in kw.items())]

    def bracket(rs, key):
        """(last surviving, first dead) along a sorted axis; None if no death."""
        rs = sorted(rs, key=lambda r: r[key])
        alive = [r[key] for r in rs if r["survives"]]
        dead = [r[key] for r in rs if not r["survives"]]
        return {
            "last_surviving": (max(alive) if alive else None),
            "first_dead": (min(dead) if dead else None),
        }

    clean = sel(kappa=0.0, gamma_phi=0.0)
    surv = [r for r in xp if r["survives"]]
    out = {
        "n_points": len(xp),
        "n_surviving": len(surv),
        "n_resolvable": sum(1 for r in xp if r["resolvable"]),
        "E_N_noiseless": next((r["E_N"] for r in clean if r["temperature"] == 0.0), None),
        "vac_fraction_T0": next((r["vac_fraction_T"] for r in clean if r["temperature"] == 0.0), None),
        "comm_fraction_T0": next((r["comm_fraction_T"] for r in clean if r["temperature"] == 0.0), None),
        "xx_vac_fraction_T0": next((r["xx_vac_fraction_T"] for r in clean if r["temperature"] == 0.0), None),
        "xx_comm_fraction_T0": next((r["xx_comm_fraction_T"] for r in clean if r["temperature"] == 0.0), None),
        "temperature_death_clean": bracket(clean, "temperature"),
        "temperature_death_lossy": bracket(sel(kappa=max(KAPPAS), gamma_phi=0.0), "temperature"),
        "dephasing_death_T0": bracket(sel(temperature=0.0, kappa=0.0), "gamma_phi"),
        "dephasing_death_T0_lossy": bracket(sel(temperature=0.0, kappa=max(KAPPAS)), "gamma_phi"),
        "vac_fraction_range_surviving": (
            [min(r["vac_fraction_T"] for r in surv), max(r["vac_fraction_T"] for r in surv)]
            if surv else None
        ),
        "comm_fraction_range_surviving": (
            [min(r["comm_fraction_T"] for r in surv), max(r["comm_fraction_T"] for r in surv)]
            if surv else None
        ),
        "vac_fraction_by_T": {str(T): next((r["vac_fraction_T"] for r in clean if r["temperature"] == T), None)
                              for T in TEMPERATURES},
        "m_over_M_range_surviving": (
            [min(r["m_over_M"] for r in surv), max(r["m_over_M"] for r in surv)] if surv else None
        ),
        "all_audits_passed": all(r["all_audits_passed"] for r in rows),
        "all_converged": all(r["converged"] for r in rows),
        "max_ledger_defect": max(abs(r["ledger_defect"]) for r in rows),
        "controls": [
            {k: r[k] for k in ("label", "coupling", "E_N", "xx_vac_fraction_T",
                               "xx_comm_fraction_T", "xx_N2_T")}
            for r in rows if r["sweep"] == "control"
        ],
        "total_elapsed_s": sum(r["elapsed_s"] for r in rows),
    }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="4-point smoke configuration")
    ap.add_argument("--out", default=None,
                    help="output directory (default: data/; a fresh temporary directory for "
                         "--quick, so a smoke run never overwrites the archive)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out) if args.out else (
        Path(tempfile.mkdtemp(prefix="derivative-coupling-noise-survival-quick-")) if args.quick else DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Vacuum Program / Layer 2 -- derivative-coupling noise survival")
    print(f"  N={N_FIELD} massless Dirichlet, sites {sites()}, cos2 T={SWITCH_T} at light "
          f"contact (d={SEPARATION}), Omega={GAP}, lambda={LAMBDA}")
    print(f"  grid: T={TEMPERATURES}, gamma_phi={GAMMA_PHIS}, kappa={KAPPAS}")
    t0 = time.time()
    if args.quick:
        rows = run_specs(smoke_specs())
        save_rows(out_dir / "rows_quick.npz", rows, sweep="quick")
        print(f"\nquick: {len(rows)} rows in {time.time() - t0:.1f}s")
        return 0
    rows = run_specs(full_specs())
    save_rows(out_dir / "rows.npz", rows, sweep="full")
    summary = summarize(rows)
    summary["wall_s"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nsummary:")
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nfull map: {len(rows)} rows in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
