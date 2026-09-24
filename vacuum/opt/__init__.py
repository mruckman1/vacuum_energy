"""vacuum.opt — the differentiable vacuum (Layer 6).

``vacuum.opt.gjax`` is the JAX mirror of ``vacuum.core`` (jit/grad-safe twins
with identical conventions: hbar = 1, V_vac = I/2, nu >= 1/2, block
quadrature ordering, nats); ``vacuum.opt.gjax_detectors`` mirrors the
'xx' detector stack of ``vacuum.detectors`` on top of it.  JAX is an
optional dependency (``.[diff]``): importing ``vacuum.opt`` never imports
it; ``HAS_JAX`` says whether the JAX-backed modules will import.  The
numpy-only modules — ``distill`` (BBPSSW/DEJMPS back end) and ``rl``
(cross-entropy search) — always import.

Layer-6 API (vacuum/opt/README.md), resolved lazily so the JAX-backed
names are attributes of the package without importing JAX eagerly:

    make_objective(kind, **cfg) -> Objective      # 'negativity' | 'qet_energy' | 'qei_ratio' | 'pkmm_negativity'
    constraints(energy_budget, noise_floor, signaling_bound) -> Constraints
    optimize(f, C, theta0, method='adam'|'lbfgs'|'nelder-mead') -> (theta_star, history)
    rl_search(env, cfg) -> Policy                  # CEM over a gym-like env (numpy)
    distill_yield(rho_or_V_AB, protocol, n_rounds) -> (bell_pairs_per_run, fidelity)
    vacuum_to_bell_rate(protocol_cfg) -> ExchangeRate
    inverse_design(target, ansatz='banded', reg) -> DesignResult (K > 0 by construction)
    design_residual(K, target) -> float

Optimized parameters are candidates, not results: each theta* is re-run
through the plain numpy stack and the full audit suite
(``Objective.roundtrip``), and only round-trip-surviving numbers are
admissible (papers/README.md).
"""

from __future__ import annotations

import importlib
import importlib.util

try:
    HAS_JAX = importlib.util.find_spec("jax") is not None
except (ImportError, ValueError):  # an import block (blocking finder / ``sys.modules`` entry): absent
    HAS_JAX = False

from .distill import (  # noqa: E402  (numpy only)
    BELL_ORDER,
    DistillResult,
    ExchangeRate,
    bbpssw_coded_round,
    bbpssw_recurrence,
    bell_diagonal_coefficients,
    bell_diagonal_from_covariance,
    bell_diagonal_from_pair_state,
    bell_diagonal_state,
    bell_states,
    dejmps_coded_round,
    dejmps_recurrence,
    distill_yield,
    oscillator_moments,
    twirl_to_bell_diagonal,
    vacuum_to_bell_rate,
    werner_state,
    x_state_from_moments,
)
from .rl import HarvestingEnv, Policy, cross_entropy_method, random_search, rl_search  # noqa: E402

_LAZY = {
    # objectives
    "make_objective": ("vacuum.opt.objectives", "make_objective"),
    "Objective": ("vacuum.opt.objectives", "Objective"),
    "ShapeSwitching": ("vacuum.opt.objectives", "ShapeSwitching"),
    "basis_functions": ("vacuum.opt.objectives", "basis_functions"),
    "NEGATIVITY_BLOCKS": ("vacuum.opt.objectives", "NEGATIVITY_BLOCKS"),
    # optimize
    "optimize": ("vacuum.opt.optimize", "optimize"),
    "constraints": ("vacuum.opt.optimize", "constraints"),
    "Constraints": ("vacuum.opt.optimize", "Constraints"),
    "project_energy_budget": ("vacuum.opt.optimize", "project_energy_budget"),
    "adam": ("vacuum.opt.optimize", "adam"),
    "lbfgs": ("vacuum.opt.optimize", "lbfgs"),
    "nelder_mead": ("vacuum.opt.optimize", "nelder_mead"),
    "fd_gradient": ("vacuum.opt.optimize", "fd_gradient"),
    # inverse design
    "inverse_design": ("vacuum.opt.inverse_design", "inverse_design"),
    "design_residual": ("vacuum.opt.inverse_design", "design_residual"),
    "DesignResult": ("vacuum.opt.inverse_design", "DesignResult"),
    "ANSATZE": ("vacuum.opt.inverse_design", "ANSATZE"),
    "TARGET_KINDS": ("vacuum.opt.inverse_design", "TARGET_KINDS"),
    "banded_indices": ("vacuum.opt.inverse_design", "banded_indices"),
    "coupling_from_theta": ("vacuum.opt.inverse_design", "coupling_from_theta"),
    "theta_from_coupling": ("vacuum.opt.inverse_design", "theta_from_coupling"),
    "lieb_robinson_velocity_bound": ("vacuum.opt.inverse_design", "lieb_robinson_velocity_bound"),
    "MultistartResult": ("vacuum.opt.multistart", "MultistartResult"),
    "multistart": ("vacuum.opt.multistart", "multistart"),
    "multistart_from_spec": ("vacuum.opt.multistart", "multistart_from_spec"),
    "run_specs": ("vacuum.opt.multistart", "run_specs"),
    "make_starts": ("vacuum.opt.multistart", "make_starts"),
    "waveform_samples": ("vacuum.opt.multistart", "waveform_samples"),
    "cluster_basins": ("vacuum.opt.multistart", "cluster_basins"),
    "with_signaling_sibling": ("vacuum.opt.multistart", "with_signaling_sibling"),
    "roundtrip_equal_work": ("vacuum.opt.multistart", "roundtrip_equal_work"),
    "nonperturbative_split": ("vacuum.opt.multistart", "nonperturbative_split"),
    "split_from_objective": ("vacuum.opt.multistart", "split_from_objective"),
    "lam_max_index": ("vacuum.opt.multistart", "lam_max_index"),
    "make_eval": ("vacuum.opt.optimize", "make_eval"),
    "make_value": ("vacuum.opt.optimize", "make_value"),
}

#: The two JAX mirrors are namespaces, not flat re-exports: ``vacuum.opt.gjax``
#: carries the SAME names as ``vacuum.core`` (``ground_state_cov``, ``entropy``,
#: ...) and ``vacuum.opt.gjax_detectors`` the same names as ``vacuum.detectors``
#: (``attach_detectors``, ``run_harvesting``, ...), by design — flattening them
#: into this package would shadow nothing here but would invite
#: ``from vacuum.opt import ground_state_cov`` next to the numpy one.  They are
#: served lazily as module attributes so ``vacuum.opt.gjax.entropy`` works
#: without importing JAX at package import.
_LAZY_MODULES = frozenset({"gjax", "gjax_detectors"})


def __getattr__(name):
    if name in _LAZY:
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    if name in _LAZY_MODULES:
        return importlib.import_module(f"vacuum.opt.{name}")
    raise AttributeError(f"module 'vacuum.opt' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | _LAZY_MODULES)


__all__ = [
    "HAS_JAX",
    # the JAX mirrors, as namespaces (lazy)
    "gjax",
    "gjax_detectors",
    # distillation back end (numpy)
    "distill_yield",
    "vacuum_to_bell_rate",
    "DistillResult",
    "ExchangeRate",
    "bbpssw_recurrence",
    "dejmps_recurrence",
    "bbpssw_coded_round",
    "dejmps_coded_round",
    "bell_diagonal_from_covariance",
    "bell_diagonal_from_pair_state",
    "bell_diagonal_coefficients",
    "bell_diagonal_state",
    "bell_states",
    "BELL_ORDER",
    "twirl_to_bell_diagonal",
    "werner_state",
    "oscillator_moments",
    "x_state_from_moments",
    # rugged-landscape search (numpy)
    "rl_search",
    "HarvestingEnv",
    "Policy",
    "cross_entropy_method",
    "random_search",
    # JAX-backed (lazy)
    "make_objective",
    "Objective",
    "ShapeSwitching",
    "basis_functions",
    "NEGATIVITY_BLOCKS",
    "optimize",
    "constraints",
    "Constraints",
    "project_energy_budget",
    "adam",
    "lbfgs",
    "nelder_mead",
    "fd_gradient",
    "inverse_design",
    "design_residual",
    "DesignResult",
    "ANSATZE",
    "TARGET_KINDS",
    "banded_indices",
    "coupling_from_theta",
    "theta_from_coupling",
    "lieb_robinson_velocity_bound",
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
    "make_eval",
    "make_value",
]
