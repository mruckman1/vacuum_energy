"""Global search for the differentiable vacuum (Layer 6): multistart, basins,
the raised convergence gate, the round trip on every candidate above the
current best, and the nonperturbative communication decomposition.

The penalized waveform landscape of ``vacuum.opt.objectives`` ('negativity',
cos^2 envelope times a Fourier series) is multimodal: the archived optima of
papers/differentiable-vacuum were the best of TWO starts, and scratch runs
with k <= 5 harmonics reached higher twin-level values that were never
round-tripped.  :func:`multistart` closes that item:

1. STARTS.  ``n_starts`` starts combine the published protocol (theta0),
   caller-supplied extra starts (the archived optimum embedded in the
   larger harmonic family), cross-entropy-seeded starts
   (:func:`vacuum.opt.rl.cross_entropy_method` on the SAME penalized twin
   loss, best sample and fitted mean of each CEM run) and random
   perturbations of the published shape at several scales.  Every start
   is polished by L-BFGS-B on the JAX gradient (:func:`vacuum.opt.optimize.optimize`)
   and projected to the energy budget on the twin
   (:func:`~vacuum.opt.optimize.project_energy_budget`).
2. BASINS.  Projected waveforms lam(t) are sampled on the window,
   canonicalized in sign (the 'xx' model is invariant under lam -> -lam,
   a symplectic sign flip of the detector quadratures) and clustered by
   relative L2 distance (:func:`cluster_basins`, greedy in descending
   twin negativity, tolerance ``basin_tol``).  The number of clusters is
   the reported number of distinct basins; the top ``n_polish`` basin
   representatives get a longer L-BFGS-B polish.
3. GATE.  Every round trip runs the dt-halving gate of the numpy stack at
   the declared ``conv_tol`` (1e-9, never weakened) with the cap raised to
   ``max_halvings`` = 18 (the archived rows sat at levels 15-16 of a cap of
   16); the accepted level and the gate movement at acceptance are recorded
   per row (``halvings``, ``movement``) so "converged" is a number, not a flag.
4. ROUND TRIP ON EVERY CANDIDATE ABOVE THE CURRENT BEST.  Basin
   representatives are round-tripped in descending twin negativity while
   their twin value exceeds the running best ROUND-TRIPPED admissible
   negativity (seeded with ``current_best``, the archived value); equal
   work is finished on the gated ledger (lam_max rescaled by
   sqrt(W_budget/W_gated), at most ``max_reprojections`` times), the
   communication fraction comes from the numpy M2.5 split of
   ``split_objective`` and the checks are
   :meth:`~vacuum.opt.optimize.Constraints.check` with the work tolerance
   ``work_rtol``.  Nothing is reported from the twin alone.

The nonperturbative communication decomposition
----------------------------------------------
The signaling bound of the archive is the second-order TMM21 split
|M_comm|/|M| (Tjoa & Martin-Martinez, PRD 104, 125005 (2021), Eqs. (31)-(33))
from the perturbative engine -- a ratio of O(lambda^2) matrix elements.
The Gaussian evolution offers a nonperturbative counterpart that needs no
perturbative state at all.  The 'xx' protocol is an affine map
V_out = S V_in S^T + Y on covariances (symplectic steps, Gaussian channels),
and the initial state is a product of the field state and the detector
ground states, so the detectors' output cross-covariance splits EXACTLY into

    C_AB = [S V_field S^T]_AB  +  [S V_det S^T]_AB ,

the field-state part (the field's own correlations, Hadamard/anti-commutator
content) and the detector-zero-point part (a detector's ground-state
fluctuation carried to the other detector by the field's retarded
propagator, the commutator content -- what TMM21 call the entanglement a
field state "with no correlations at all" would produce just as well).
On the pair correlation m = <a_A a_B> of the detector block
(``vacuum.opt.distill.oscillator_moments``; m equals PKMM's pair term M at
second order with lam_UDW = lam_osc/sqrt(2 Omega), the MAP-1 identification
of the objective layer) this defines

    M_full = m[V],  M_field = m[V_field (+) 0],  M_signal = m[0 (+) V_det],
    f_np = |M_signal| / |M_full|             (:func:`nonperturbative_split`),

where on noisy floors a fourth run from V = 0 isolates the channel term
Y_tot -- noise injected by a channel into one detector and carried to the
other by the propagator, booked as ``M_signal_channel`` inside M_signal.  Anchors (tests/test_opt_multistart.py): M_field -> M_vac and
M_signal -> M_comm as lambda -> 0 with |f_np - |M_comm|/|M|| = O(lambda^2)
(measured 2.6e-4, 2.9e-5, 2.3e-6, 6.4e-8 at lam_osc = 0.3, 0.1, 0.03, 0.01
on the Gate-A lattice), additivity M_full = M_field + M_signal to 1e-12,
and M_signal = 0 to lattice leakage in a spacelike layout.  At the archived
operating point (lam_osc = 0.3, a 13.8-unit window) the O(lambda^4) terms
are order one: |M_full| is 1.9x the second-order |M| and f_np of the cos^2
baseline is 0.72 against the proxy's 0.216 -- the one-number consistency
check the Layer-6 status asked for, computed by the notebook on every final
waveform (``proxy_consistency``).

Everything is pure over its arguments; seeds are explicit; the process-pool
entry point :func:`multistart_from_spec` builds objectives from plain data so
floors can run in parallel without pickling closures.
"""

from __future__ import annotations

import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError("vacuum.opt.multistart needs JAX (install the '.[diff]' extra)") from exc

from vacuum.opt.objectives import Objective, basis_functions, make_objective
from vacuum.opt.optimize import Constraints, constraints, make_value, optimize, project_energy_budget
from vacuum.opt.rl import cross_entropy_method

jax.config.update("jax_enable_x64", True)

__all__ = [
    "MultistartResult",
    "multistart",
    "multistart_from_spec",
    "run_specs",
    "make_starts",
    "waveform_samples",
    "cluster_basins",
    "with_signaling_sibling",
    "roundtrip_equal_work",
    "nonperturbative_split",
    "split_from_objective",
    "lam_max_index",
]

#: gate defaults: the archive's tolerance (never weakened) and the raised cap
CONV_TOL = 1.0e-9
MAX_HALVINGS = 18
#: relative tolerance of the equal-work check (the work column's gate uncertainty)
WORK_CHECK_RTOL = 1.0e-4


# --------------------------------------------------------------------------
# Theta layout helpers
# --------------------------------------------------------------------------


def lam_max_index(objective) -> int:
    """Index of the coupling amplitude ``lam_max`` in theta (-1 if absent)."""
    names = tuple(objective.names)
    return names.index("lam_max") if "lam_max" in names else -1


def _lam_indices(objective) -> np.ndarray:
    """Indices of the waveform coefficients (the 'lam' block) in theta."""
    return np.array([i for i, n in enumerate(objective.names) if n == "lam" or n.startswith("lam[")], dtype=int)


def _shape_indices(objective) -> np.ndarray:
    """The 'lam' block without its leading coefficient a_0 (the cos^2 weight)."""
    idx = _lam_indices(objective)
    return idx[1:] if idx.size > 1 else idx


# --------------------------------------------------------------------------
# Signaling sibling: an in-loop proxy from a second objective
# --------------------------------------------------------------------------


def with_signaling_sibling(objective, sibling):
    """Objective whose quantity/work are ``objective``'s and whose ``comm_fraction`` is ``sibling``'s.

    For configurations where the main objective carries no differentiable
    communication proxy (a gap schedule: the perturbative split has static
    gaps only) the penalty is evaluated on a static-gap, T = 0, noiseless
    sibling with the SAME theta layout -- the declared approximation is
    that the split is taken at the static gap, for the baseline and the
    candidate alike.  The round trip stays the main objective's; the numpy
    split is the sibling's (``evolve=False``).
    """
    if tuple(objective.names) != tuple(sibling.names):
        raise ValueError("the signaling sibling must share the main objective's theta layout")

    def quantity_aux(theta):
        q, aux = objective.quantity_aux(theta)
        _, aux_s = sibling.quantity_aux(theta)
        out = dict(aux)
        for key in ("comm_fraction", "M_abs_proxy", "M_vac_abs_proxy", "M_comm_abs_proxy"):
            if key in aux_s:
                out[key] = aux_s[key]
        return q, out

    cfg = dict(objective.cfg)
    cfg["signaling_proxy"] = "sibling"
    cfg["sibling_cfg"] = {k: v for k, v in sibling.cfg.items() if k not in ("detectors", "ts")}
    return Objective(kind=objective.kind, names=tuple(objective.names), theta0=np.asarray(objective.theta0, dtype=float),
                     bounds=objective.bounds, quantity_aux=quantity_aux, roundtrip=objective.roundtrip,
                     unpack=objective.unpack, cfg=cfg, differentiable=bool(objective.differentiable))


# --------------------------------------------------------------------------
# Waveforms and basins
# --------------------------------------------------------------------------


def _waveform_factory(objective, theta):
    """(lam_of_t, gaps_of_t, profiles) reconstructed from the objective's cfg and theta."""
    cfg = objective.cfg
    p = objective.unpack(np.asarray(theta, dtype=float))
    basis, n_coeff = cfg["basis"], int(cfg["n_coeff"])
    t0, Tw = float(cfg["window"][0]), float(cfg["window"][1])
    n_det = len(cfg["detectors"])
    n_gap = int(cfg["n_gap_coeff"])

    def lam_of_t(t):
        return np.array([float(p["lam_max"][0]) * float(basis_functions(float(t) - float(p["delay"][d]), basis, n_coeff, t0, Tw, np) @ p["lam"])
                         for d in range(n_det)])

    def gaps_of_t(t):
        if n_gap == 0:
            return np.asarray(p["gap"], dtype=float)
        B = basis_functions(float(t), cfg["gap_basis"], n_gap, t0, Tw, np)
        return np.asarray(p["gap"], dtype=float) * (1.0 + float(B @ p["gap_mod"]))

    return lam_of_t, gaps_of_t, p


def waveform_samples(objective, theta, n=256, detector=0):
    """lam(t) of one detector sampled on its window, sign-canonicalized (module docstring, basins)."""
    cfg = objective.cfg
    t0, Tw = float(cfg["window"][0]), float(cfg["window"][1])
    lam_of_t, _, p = _waveform_factory(objective, theta)
    delay = float(p["delay"][int(detector)])
    ts = np.linspace(t0 - Tw + delay, t0 + Tw + delay, int(n))
    w = np.array([lam_of_t(t)[int(detector)] for t in ts])
    chi = np.array([basis_functions(t - delay, "cos2", 1, t0, Tw, np)[0] for t in ts])
    if float(np.dot(w, chi)) < 0.0:
        w = -w
    return w


def cluster_basins(waveforms, values, tol=0.05):
    """Greedy clustering of sampled waveforms by relative L2 distance; labels ordered by descending value.

    Two waveforms belong to one basin when ||w_i - w_j|| <= tol * max(||w_i||, ||w_j||).
    Returns an int array of labels (0 = the basin holding the highest value).
    """
    W = [np.asarray(w, dtype=float) for w in waveforms]
    vals = np.asarray(values, dtype=float)
    order = np.argsort(-vals, kind="stable")
    labels = -np.ones(len(W), dtype=int)
    reps: List[np.ndarray] = []
    for i in order:
        for k, r in enumerate(reps):
            scale = max(float(np.linalg.norm(W[i])), float(np.linalg.norm(r)), 1e-300)
            if float(np.linalg.norm(W[i] - r)) <= float(tol) * scale:
                labels[i] = k
                break
        if labels[i] < 0:
            labels[i] = len(reps)
            reps.append(W[i])
    return labels


# --------------------------------------------------------------------------
# Starts
# --------------------------------------------------------------------------


def make_starts(objective, n_starts, seeds, *, value_fn=None, extra_starts=(), n_cem=2, cem_pop=32, cem_iter=6,
                cem_std=0.2, random_scales=(0.1, 0.2, 0.4), lam_max_fixed=True):
    """The start set (module docstring, item 1): list of (name, theta, meta).

    ``value_fn`` (``theta -> (loss, info)``, :func:`vacuum.opt.optimize.make_value`)
    is what the cross-entropy runs maximize (as -loss) over the waveform
    coefficients a_1.., b_1.. with a_0 and lam_max held at theta0; without
    it no CEM starts are made.  ``extra_starts`` is a sequence of
    (name, theta).  Random starts perturb the same coefficients by
    N(0, scale^2) with the scales cycled; ``seeds`` seed the random starts
    (seeds[0]) and the CEM runs (seeds[k], extended by seeds[0] + k).
    """
    seeds = tuple(int(s) for s in (seeds if isinstance(seeds, (list, tuple)) else (seeds,)))
    if not seeds:
        seeds = (0,)
    theta0 = np.asarray(objective.theta0, dtype=float)
    idx = _shape_indices(objective)
    starts: List[Tuple[str, np.ndarray, Dict[str, Any]]] = [("published", theta0.copy(), {"kind": "published"})]
    for name, th in extra_starts:
        th = np.asarray(th, dtype=float)
        if th.shape != theta0.shape:
            raise ValueError(f"extra start {name!r} has shape {th.shape}, expected {theta0.shape}")
        starts.append((str(name), th.copy(), {"kind": "extra"}))
    n_evals = 0
    if value_fn is not None and int(n_cem) > 0 and idx.size > 0 and len(starts) < int(n_starts):
        for k in range(int(n_cem)):
            seed = seeds[k] if k < len(seeds) else seeds[0] + k
            base = theta0.copy()

            def f(x, base=base):
                th = base.copy()
                th[idx] = x
                val, _ = value_fn(th)
                return -val if math.isfinite(val) else -1e30

            bx, bf, mean, std, hist = cross_entropy_method(f, theta0[idx], float(cem_std), n_iter=int(cem_iter), pop=int(cem_pop), rng=seed)
            n_evals += int(hist[-1]["n_evals"]) if hist else 0
            th_best = theta0.copy()
            th_best[idx] = bx
            th_mean = theta0.copy()
            th_mean[idx] = mean
            starts.append((f"cem{seed}:best", th_best, {"kind": "cem", "seed": seed, "cem_best_loss": -bf, "cem_evals": int(hist[-1]["n_evals"]) if hist else 0}))
            if len(starts) < int(n_starts):
                starts.append((f"cem{seed}:mean", th_mean, {"kind": "cem", "seed": seed, "cem_best_loss": -bf}))
            if len(starts) >= int(n_starts):
                break
    rng = np.random.default_rng(seeds[0])
    scales = tuple(float(s) for s in random_scales) or (0.1,)
    k = 0
    while len(starts) < int(n_starts):
        scale = scales[k % len(scales)]
        th = theta0.copy()
        if idx.size > 0:
            th[idx] = th[idx] + scale * rng.standard_normal(idx.size)
        starts.append((f"random{k}:s{scale:g}", th, {"kind": "random", "scale": scale}))
        k += 1
    return starts[: int(n_starts)], n_evals


# --------------------------------------------------------------------------
# Round trip at equal work
# --------------------------------------------------------------------------


def _audit_flags(rt):
    a = rt.get("audits", {})
    return {k: (bool(v.passed) if hasattr(v, "passed") else None) for k, v in a.items()}


def roundtrip_equal_work(objective, theta, budget, *, split_objective=None, comm_bound=None, conv_tol=CONV_TOL,
                         max_halvings=MAX_HALVINGS, work_rtol=WORK_CHECK_RTOL, index=-1, max_reprojections=2,
                         baseline_E_N=None, noise_floor=None):
    """Gated numpy round trip of theta with equal work finished on the ledger (module docstring, item 4).

    Returns the archive row: E_N, work, the communication fraction (numpy
    split of ``split_objective`` at theta, or the objective's own when it
    computes one), gate level/movement, audits, the twin diagnostics
    (``*_jax``, ``comm_proxy``), the :meth:`Constraints.check` verdicts
    and ``admissible`` (checks passed AND the gate converged).
    """
    theta = np.asarray(theta, dtype=float).copy()
    budget = float(budget)
    rt = objective.roundtrip(theta, communication=(split_objective is None), conv_tol=conv_tol, max_halvings=max_halvings)
    reprojections = 0
    while abs(rt["work"] / budget - 1.0) > 1e-6 and reprojections < int(max_reprojections) and rt["work"] > 0.0:
        theta[index] *= math.sqrt(budget / rt["work"])
        rt = objective.roundtrip(theta, communication=(split_objective is None), conv_tol=conv_tol, max_halvings=max_halvings)
        reprojections += 1
    q, aux = objective.quantity_aux(jnp.asarray(theta))
    if split_objective is not None:
        comm = split_objective.roundtrip(theta, evolve=False)
    else:
        comm = {"comm_fraction": rt.get("comm_fraction", float("nan")), "E_N_pert": rt.get("E_N_pert", float("nan")),
                "M_vac_abs": rt.get("M_vac_abs", float("nan")), "M_comm_abs": rt.get("M_comm_abs", float("nan"))}
    chk = Constraints(energy_budget=budget, signaling_bound=comm_bound, noise_floor=noise_floor).check(
        {"work": rt["work"], "comm_fraction": comm["comm_fraction"], "audits_passed": rt["audits_passed"], "noise": rt.get("noise")},
        work_rtol=work_rtol)
    gain = None
    if baseline_E_N is not None:
        gain = rt["E_N"] / baseline_E_N if baseline_E_N > 0 else float("inf")
    return dict(theta=theta, E_N=rt["E_N"], work=rt["work"], comm_fraction=comm["comm_fraction"], E_N_pert=comm["E_N_pert"],
                M_vac_abs=comm.get("M_vac_abs", float("nan")), M_comm_abs=comm.get("M_comm_abs", float("nan")),
                converged=bool(rt["converged"]), halvings=int(rt["halvings"]), movement=rt["movement"], dt_converged=rt["dt_converged"],
                conv_tol=float(conv_tol), max_halvings=int(max_halvings), flagged=bool(rt["flagged"]), regime=rt["regime"],
                min_nu_pt=rt["min_nu_pt"], surrogate=rt["surrogate"], audits=_audit_flags(rt), audits_passed=bool(rt["audits_passed"]),
                n_d=np.asarray(rt["n_d"]), dissipated=rt["dissipated"], E_N_jax=float(aux["E_N"]), surrogate_jax=float(aux["surrogate"]),
                work_jax=float(aux["work"]), comm_proxy=float(aux["comm_fraction"]) if "comm_fraction" in aux else float("nan"),
                gain=gain, check=chk, admissible=bool(chk["admissible"] and rt["converged"]),
                gated_reprojections=reprojections, work_over_budget=rt["work"] / budget - 1.0, V_AB=rt["V_AB"])


# --------------------------------------------------------------------------
# The multistart
# --------------------------------------------------------------------------


@dataclass
class MultistartResult:
    """Everything a global search produced (module docstring)."""

    trials: List[Dict[str, Any]]
    basins: List[Dict[str, Any]]
    roundtrips: List[Dict[str, Any]]
    best: Optional[Dict[str, Any]]
    stats: Dict[str, Any]
    gate: Dict[str, Any]
    config: Dict[str, Any]
    n_evals: int
    runtime_s: float
    current_best: Optional[float]
    beats_current_best: bool
    time_limited: bool = False
    starts_run: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trials": [dict(t) for t in self.trials], "basins": [dict(b) for b in self.basins],
            "roundtrips": [dict(r) for r in self.roundtrips], "best": None if self.best is None else dict(self.best),
            "stats": dict(self.stats), "gate": dict(self.gate), "config": dict(self.config), "n_evals": int(self.n_evals),
            "runtime_s": float(self.runtime_s), "current_best": self.current_best, "beats_current_best": bool(self.beats_current_best),
            "time_limited": bool(self.time_limited), "starts_run": int(self.starts_run),
        }


def _stats(values):
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "best": None, "median": None, "min": None, "max": None, "std": None, "spread": None}
    return {"n": int(v.size), "best": float(v.max()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
            "std": float(v.std()), "spread": float(v.max() - v.min())}


def multistart(objective, C, n_starts=32, seeds=(0,), *, budget, comm_bound=None, baseline_E_N=None, split_objective=None,
               extra_starts=(), n_cem=2, cem_pop=32, cem_iter=6, cem_std=0.2, random_scales=(0.1, 0.2, 0.4), maxiter=30,
               restarts=1, n_polish=3, polish_maxiter=80, polish_restarts=2, basin_tol=0.05, current_best=None,
               conv_tol=CONV_TOL, max_halvings=MAX_HALVINGS, work_rtol=WORK_CHECK_RTOL, max_reprojections=2,
               min_roundtrips=1, max_roundtrips=None, time_limit=None, noise_floor=None, verbose=False):
    """Global search over ``objective`` under the in-loop constraints ``C`` (module docstring).

    Parameters
    ----------
    objective : Objective ('negativity'; theta = waveform coefficients + lam_max)
    C : Constraints
        In-loop penalties: ``energy_budget`` is the TWIN-scaled budget
        (budget x twin bias), ``signaling_bound`` the bound with its margin.
    n_starts, seeds : the start set (:func:`make_starts`)
    budget : float
        The gated ledger work budget every round trip is projected to.
    comm_bound : float or None
        Hard bound on the numpy communication fraction (the check).
    baseline_E_N : float or None
        Published protocol's round-trip E_N (gains are stated against it).
    split_objective : Objective or None
        Static-gap T = 0 sibling whose ``roundtrip(theta, evolve=False)``
        supplies the numpy split; None uses the objective's own columns.
    current_best : float or None
        The archived round-tripped admissible E_N; candidates above it are
        round-tripped (``min_roundtrips`` always are).
    time_limit : float or None
        Wall-clock budget (s) for the start loop; when exceeded the
        remaining starts are skipped and ``time_limited`` is set.

    Returns
    -------
    MultistartResult
    """
    t_all = time.time()
    theta0 = np.asarray(objective.theta0, dtype=float)
    idx_lm = lam_max_index(objective)
    value_fn = make_value(objective, C)
    starts, n_evals = make_starts(objective, n_starts, seeds, value_fn=value_fn, extra_starts=extra_starts, n_cem=n_cem,
                                  cem_pop=cem_pop, cem_iter=cem_iter, cem_std=cem_std, random_scales=random_scales)
    budget_twin = float(C.energy_budget) if C.energy_budget is not None else float(budget)

    def finish(theta, hist, name, meta, polished=False):
        nonlocal n_evals
        n_evals += len(hist)
        message = str(hist[-1].get("message", "")) if hist else ""
        try:
            theta = project_energy_budget(objective, theta, budget_twin, index=idx_lm)
            projected = True
        except RuntimeError:
            projected = False
        q, aux = objective.quantity_aux(jnp.asarray(theta))
        E_N = float(aux["E_N"])
        comm = float(aux["comm_fraction"]) if "comm_fraction" in aux else float("nan")
        ok = bool(np.isfinite(E_N) and projected and (comm_bound is None or not math.isfinite(comm) or comm <= float(comm_bound)))
        return dict(start=name, theta=np.asarray(theta, dtype=float), E_N_jax=E_N, surrogate_jax=float(aux["surrogate"]),
                    work_jax=float(aux["work"]), comm_proxy=comm, loss=float(hist[-1]["loss"]) if hist else float("nan"),
                    n_evals=len(hist), message=message, ok=ok, projected=projected, polished=polished, **meta)

    trials: List[Dict[str, Any]] = []
    time_limited = False
    for k, (name, th0, meta) in enumerate(starts):
        if time_limit is not None and time.time() - t_all > float(time_limit) and k >= 1:
            time_limited = True
            break
        t0 = time.time()
        theta, hist = optimize(objective, C, th0, method="lbfgs", maxiter=int(maxiter), restarts=int(restarts))
        row = finish(theta, hist, name, meta)
        row["runtime_s"] = time.time() - t0
        trials.append(row)
        if verbose:
            print(f"    start {name:>16s}: twin E_N {row['E_N_jax']:.5e} comm {row['comm_proxy']:.4f} ok={row['ok']} "
                  f"({row['n_evals']} evals, {row['runtime_s']:.0f}s)", flush=True)
    starts_run = len(trials)

    def label_basins():
        W = [waveform_samples(objective, t["theta"]) for t in trials]
        vals = [t["E_N_jax"] if t["ok"] else -np.inf for t in trials]
        labels = cluster_basins(W, vals, tol=basin_tol)
        for t, lab in zip(trials, labels):
            t["basin"] = int(lab)
        return labels

    labels = label_basins()
    # polish the top basins' representatives (by twin negativity, ok trials first)
    reps = {}
    for t in sorted(trials, key=lambda t: (-int(t["ok"]), -t["E_N_jax"])):
        reps.setdefault(t["basin"], t)
    top = [reps[b] for b in sorted(reps, key=lambda b: (-int(reps[b]["ok"]), -reps[b]["E_N_jax"]))][: int(n_polish)]
    for rep in top:
        if int(polish_maxiter) <= 0:
            break
        t0 = time.time()
        theta, hist = optimize(objective, C, rep["theta"], method="lbfgs", maxiter=int(polish_maxiter), restarts=int(polish_restarts))
        row = finish(theta, hist, rep["start"] + "+polish", {k: v for k, v in rep.items() if k in ("kind", "seed", "scale")}, polished=True)
        row["runtime_s"] = time.time() - t0
        row["polished_from"] = rep["start"]
        trials.append(row)
    labels = label_basins()
    basins: List[Dict[str, Any]] = []
    for b in sorted(set(int(x) for x in labels)):
        members = [t for t in trials if t["basin"] == b]
        ok_members = [t for t in members if t["ok"]]
        best_t = max(ok_members or members, key=lambda t: t["E_N_jax"])
        basins.append(dict(label=b, size=len(members), n_ok=len(ok_members), best_start=best_t["start"], E_N_jax=best_t["E_N_jax"],
                           comm_proxy=best_t["comm_proxy"], work_jax=best_t["work_jax"], theta=best_t["theta"], ok=bool(ok_members),
                           starts=[t["start"] for t in members]))
    # round trips: descending twin E_N while above the running best round-tripped admissible E_N
    best_rt = -math.inf if current_best is None else float(current_best)
    best_row = None
    roundtrips: List[Dict[str, Any]] = []
    order = sorted([b for b in basins if b["ok"]], key=lambda b: -b["E_N_jax"]) or sorted(basins, key=lambda b: -b["E_N_jax"])
    for b in order:
        above = b["E_N_jax"] > best_rt
        if not above and len(roundtrips) >= int(min_roundtrips):
            break
        if max_roundtrips is not None and len(roundtrips) >= int(max_roundtrips):
            break
        t0 = time.time()
        row = roundtrip_equal_work(objective, b["theta"], budget, split_objective=split_objective, comm_bound=comm_bound,
                                   conv_tol=conv_tol, max_halvings=max_halvings, work_rtol=work_rtol, index=idx_lm,
                                   max_reprojections=max_reprojections, baseline_E_N=baseline_E_N, noise_floor=noise_floor)
        row.update(basin=b["label"], start=b["best_start"], runtime_s=time.time() - t0, twin_above_current_best=bool(above))
        roundtrips.append(row)
        if verbose:
            print(f"    roundtrip basin {b['label']} ({b['best_start']}): E_N {row['E_N']:.5e} (twin {row['E_N_jax']:.5e}) work {row['work']:.5e} "
                  f"comm {row['comm_fraction']:.4f} level {row['halvings']} movement {row['movement']:.2e} admissible={row['admissible']}", flush=True)
        if row["admissible"] and row["E_N"] > best_rt:
            best_rt = row["E_N"]
            best_row = row
    ok_trials = [t for t in trials if t["ok"]]
    if baseline_E_N is not None and baseline_E_N > 0:
        gains_twin = [t["E_N_jax"] / baseline_E_N for t in ok_trials]
        gains_rt = [r["E_N"] / baseline_E_N for r in roundtrips if r["admissible"]]
    else:
        gains_twin = [t["E_N_jax"] for t in ok_trials]
        gains_rt = [r["E_N"] for r in roundtrips if r["admissible"]]
    top_rt = max(roundtrips, key=lambda r: r["E_N"]) if roundtrips else None
    stats = dict(top_roundtrip_E_N=None if top_rt is None else top_rt["E_N"], top_roundtrip_admissible=None if top_rt is None else bool(top_rt["admissible"]),
                 improvement_over_current_best=None if (best_row is None or current_best is None or current_best <= 0) else best_row["E_N"] / float(current_best) - 1.0,
                 n_trials=len(trials), n_ok=len(ok_trials), n_basins=len(set(t["basin"] for t in ok_trials)),
                 n_basins_all=len(basins), twin_gains=_stats(gains_twin), roundtrip_gains=_stats(gains_rt),
                 gains_are_factors=bool(baseline_E_N is not None and baseline_E_N > 0),
                 n_roundtrips=len(roundtrips), n_admissible=sum(int(r["admissible"]) for r in roundtrips))
    gate = dict(conv_tol=float(conv_tol), max_halvings=int(max_halvings),
                max_level_used=max([r["halvings"] for r in roundtrips], default=None),
                max_movement=max([r["movement"] for r in roundtrips if r["movement"] is not None], default=None),
                all_converged=all(r["converged"] for r in roundtrips) if roundtrips else None)
    config = dict(n_starts=int(n_starts), seeds=tuple(int(s) for s in (seeds if isinstance(seeds, (list, tuple)) else (seeds,))),
                  n_cem=int(n_cem), cem_pop=int(cem_pop), cem_iter=int(cem_iter), cem_std=float(cem_std),
                  random_scales=tuple(float(s) for s in random_scales), maxiter=int(maxiter), restarts=int(restarts),
                  n_polish=int(n_polish), polish_maxiter=int(polish_maxiter), polish_restarts=int(polish_restarts),
                  basin_tol=float(basin_tol), budget=float(budget), budget_twin=budget_twin, comm_bound=comm_bound,
                  baseline_E_N=baseline_E_N, work_rtol=float(work_rtol), time_limit=time_limit,
                  in_loop=dict(energy_budget=C.energy_budget, signaling_bound=C.signaling_bound, weight=C.weight,
                               signaling_weight=C.signaling_weight),
                  objective_cfg={k: v for k, v in objective.cfg.items() if k not in ("detectors", "ts")})
    # "beats" only when the improvement exceeds the equal-work check tolerance: two rows whose
    # work agrees with the budget to work_rtol are indistinguishable below that relative margin
    beats = bool(best_row is not None and current_best is not None and best_row["E_N"] > float(current_best) * (1.0 + float(work_rtol)))
    return MultistartResult(trials=trials, basins=basins, roundtrips=roundtrips, best=best_row, stats=stats, gate=gate,
                            config=config, n_evals=int(n_evals), runtime_s=time.time() - t_all, current_best=current_best,
                            beats_current_best=beats, time_limited=time_limited, starts_run=starts_run)


# --------------------------------------------------------------------------
# Process-pool entry points (plain-data specs)
# --------------------------------------------------------------------------


def multistart_from_spec(spec):
    """Build the objectives from a plain-data spec and run :func:`multistart`; returns (label, result dict).

    spec keys: ``objective`` (kwargs of ``make_objective('negativity', ...)``),
    optional ``split`` and ``proxy`` (same form; the proxy becomes the
    signaling sibling), ``constraints`` (kwargs of :func:`constraints`),
    ``n_starts``, ``seeds``, ``options`` (keyword arguments of
    :func:`multistart`; ``extra_starts`` there is a mapping name -> list).
    """
    obj = make_objective("negativity", **spec["objective"])
    split = make_objective("negativity", **spec["split"]) if spec.get("split") else None
    if spec.get("proxy"):
        obj = with_signaling_sibling(obj, make_objective("negativity", **spec["proxy"]))
    C = constraints(**spec.get("constraints", {}))
    opts = dict(spec.get("options", {}))
    extra = opts.pop("extra_starts", {}) or {}
    extra_starts = [(name, np.asarray(th, dtype=float)) for name, th in (extra.items() if isinstance(extra, dict) else extra)]
    res = multistart(obj, C, int(spec.get("n_starts", 32)), tuple(spec.get("seeds", (0,))), split_objective=split,
                     extra_starts=extra_starts, **opts)
    return spec.get("label"), res.as_dict()


def run_specs(specs, workers=1):
    """Run several specs, in parallel processes when ``workers`` > 1 (spawn context; JAX per process)."""
    specs = list(specs)
    if int(workers) <= 1 or len(specs) <= 1:
        return [multistart_from_spec(s) for s in specs]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(int(workers), len(specs)), mp_context=ctx) as ex:
        return list(ex.map(multistart_from_spec, specs))


# --------------------------------------------------------------------------
# The nonperturbative communication decomposition
# --------------------------------------------------------------------------


def nonperturbative_split(K, sites_or_profiles, gaps_of_t, lam_of_t, ts, splits, *, T=0.0, channels=(), moments_gaps=None):
    """Field-state / detector-zero-point decomposition of the pair correlation (module docstring).

    Runs the validated fixed-resolution stepper
    (``vacuum.detectors.protocol_evolve_fixed``) from the full initial
    state, from the field block alone, from the detector block alone and
    (with channels) from V = 0, and reads m = <a_A a_B> of the detector
    block from each.  Two detectors only.  Returns a dict with the complex
    ``M_full``, ``M_field``, ``M_signal``, the fraction ``f_np``, the
    additivity defect, ``E_N`` of the full run and the moments' gaps.
    """
    from vacuum.detectors import attach_detectors, detector_block, harvested_log_negativity, initial_state, protocol_evolve_fixed
    from vacuum.opt.distill import oscillator_moments

    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    ts = np.asarray(ts, dtype=float)
    dets = list(sites_or_profiles)
    if len(dets) != 2:
        raise ValueError("the pair-term decomposition needs exactly two detectors")
    gaps0 = np.asarray(gaps_of_t(ts[0]), dtype=float).reshape(2)
    init = initial_state(K, dets, tuple(float(g) for g in gaps0), float(T), strict=False)
    V = np.asarray(init.V, dtype=float)
    n = V.shape[0] // 2
    fi = np.concatenate([np.arange(N), n + np.arange(N)])
    di = np.concatenate([np.arange(N, n), n + np.arange(N, n)])
    V_field = np.zeros_like(V)
    V_det = np.zeros_like(V)
    V_field[np.ix_(fi, fi)] = V[np.ix_(fi, fi)]
    V_det[np.ix_(di, di)] = V[np.ix_(di, di)]
    channels = tuple(channels)

    def K_of_t(t):
        return attach_detectors(K, dets, gaps_of_t(t), lam_of_t(t))

    def run(V0):
        return protocol_evolve_fixed(V0, K_of_t, ts, splits=int(splits), channels=channels).V

    gm = np.asarray(gaps_of_t(ts[-1]) if moments_gaps is None else moments_gaps, dtype=float).reshape(2)
    out_full = run(V)
    out_field = run(V_field)
    out_det = run(V_det)
    out_zero = run(np.zeros_like(V)) if channels else None
    m = lambda Vo: oscillator_moments(detector_block(Vo, N), gm)["m"]
    # With channels the map is affine, V_out = S V S^T + Y_tot, and Y_tot has a
    # cross block of its own: noise injected into one detector by its channel
    # is carried to the other by the field's propagator.  That is a third,
    # field-state-independent (signaling-type) term, booked separately as
    # M_signal_channel; M_signal = detector-zero-point + channel-injected.
    m_channel = m(out_zero) if out_zero is not None else 0.0
    M_full = m(out_full)
    M_field = m(out_field) - m_channel
    M_signal_det = m(out_det) - m_channel
    M_signal = M_signal_det + m_channel
    denom = abs(M_full)
    frac = lambda x: (abs(x) / denom) if denom > 0 else 0.0
    return dict(M_full=M_full, M_field=M_field, M_signal=M_signal, M_signal_det=M_signal_det, M_signal_channel=m_channel,
                f_np=frac(M_signal), f_np_det=frac(M_signal_det), f_field=frac(M_field),
                additivity=abs(M_full - M_field - M_signal) / denom if denom > 0 else abs(M_full - M_field - M_signal),
                E_N=float(harvested_log_negativity(detector_block(out_full, N)).E_N), moments_gaps=gm, splits=int(splits), T=float(T),
                n_channels=len(channels))


def split_from_objective(objective, theta, K, *, splits):
    """:func:`nonperturbative_split` at theta with the objective's own layout, schedule, floor and channels."""
    from vacuum.detectors import imperfection_channel, smearing

    cfg = objective.cfg
    N = int(cfg["N"])
    K = np.asarray(K, dtype=float)
    if K.shape != (N, N):
        raise ValueError(f"K must be ({N}, {N}) for this objective, got {K.shape}")
    lam_of_t, gaps_of_t, p = _waveform_factory(objective, theta)
    profiles = []
    for d, det in enumerate(cfg["detectors"]):
        if "profile" in det:
            profiles.append(smearing(det["profile"], float(p["center"][d]), float(p["width"][d]), N, bc=det.get("bc", "open")))
        else:
            profiles.append(int(det["site"]))
    ts = np.asarray(cfg["ts"], dtype=float)
    channels = ()
    noise = cfg.get("noise")
    if noise:
        gaps0 = tuple(float(g) for g in np.asarray(gaps_of_t(ts[0])).reshape(-1))
        channels = (imperfection_channel(tuple(cfg["det_modes"]), kappa=float(noise.get("kappa", 0.0)), nbar=float(noise.get("nbar", 0.0)),
                                         gamma_phi=float(noise.get("gamma_phi", 0.0)), axis=noise.get("axis", "isotropic"), gaps=gaps0),)
    return nonperturbative_split(K, profiles, gaps_of_t, lam_of_t, ts, splits, T=float(cfg["T"]), channels=channels)
