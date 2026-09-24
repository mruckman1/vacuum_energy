"""Dependency-free search for rugged landscapes (Layer 6): CEM over a gym-like env.

Where gradients are useless — a harvested negativity that is identically
zero in the sudden-death region, a hard "coupler must be off" dead zone in
the switching window, discrete layout choices — the differentiable stack
hands over to a derivative-free search.  This module provides the minimal
version: the cross-entropy method (Rubinstein, Eur. J. Oper. Res. 99, 89
(1997); de Boer, Kroese, Mannor, Rubinstein, Ann. Oper. Res. 134, 19
(2005)) over a Gaussian policy on the action space of a gym-like
environment wrapping the plain-numpy simulator, plus a random-search
baseline it must beat.  No third-party RL dependency; seeds are explicit.

Environment contract (``gym``-like, single-step episodes):

    obs = env.reset()
    obs, reward, done, info = env.step(action)     # action: (action_dim,) float array
    env.action_dim, env.action_bounds               # (lo, hi) arrays

:class:`HarvestingEnv` is the reference environment: a small lattice field
with two oscillator detectors, the action being the switching-waveform
SAMPLES on interior knots of the window (hat-function basis, zero at the
edges), the reward the harvested log-negativity from the numpy stack
(``protocol_evolve_fixed`` at a declared resolution and
``harvested_log_negativity`` with its mp-margin rule), scaled by
``reward_scale``.  A HARD dead zone — a set of knots on which the coupler
must be off — turns the landscape rugged: an action violating it is not
simulated at all and receives ``-dead_zone_penalty * violation``, so the
feasible set is a lower-dimensional slice the search has to find and stay
on, and inside it the reward is flat-zero wherever the detectors do not
harvest.  ``rl_search(env, cfg)`` returns the fitted policy (mean, std) and
the best action/reward found, with the per-iteration history.

The demonstration test (tests/test_opt_rl.py) runs CEM and random search
at an equal evaluation budget over several seeds and requires CEM to win
on every seed — the honest version of "beats random search".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "HarvestingEnv",
    "cross_entropy_method",
    "random_search",
    "Policy",
    "rl_search",
]


# --------------------------------------------------------------------------
# Reference environment
# --------------------------------------------------------------------------


class HarvestingEnv:
    """Two-detector harvesting on a small chain; action = waveform samples.

    Parameters
    ----------
    N, mass, bc : lattice field (``harmonic_chain_K``)
    sites : (site_A, site_B)
    gaps : (Omega_A, Omega_B)
    window : (t0, T) — the switching window [t0 - T, t0 + T]
    n_knots : number of interior knots (action dimension)
    dead_zone : sequence of knot indices on which |action| must be <= dead_tol
    lam_max : coupling amplitude multiplying the samples
    splits : substeps of the fixed-resolution numpy evolution
    reward_scale : multiplies E_N (nats) in the reward
    dead_zone_penalty : reward = -penalty * sum |violations| for infeasible actions
    action_bounds : (lo, hi) scalars for every sample
    """

    def __init__(self, N=12, mass=0.6, bc="dirichlet", sites=(5, 6), gaps=(2.0, 2.0), window=(0.0, 2.0),
                 n_knots=9, dead_zone=(4, 5), lam_max=0.3, splits=32, reward_scale=1.0e3,
                 dead_zone_penalty=10.0, dead_tol=0.05, action_bounds=(-1.0, 1.0)):
        from vacuum.core import harmonic_chain_K
        from vacuum.detectors import initial_state

        self.N = int(N)
        self.K = harmonic_chain_K(self.N, float(mass), bc=bc)
        self.sites = tuple(int(s) for s in sites)
        self.gaps = tuple(float(g) for g in gaps)
        self.t0, self.T = float(window[0]), float(window[1])
        self.n_knots = int(n_knots)
        self.knots = np.linspace(self.t0 - self.T, self.t0 + self.T, self.n_knots + 2)
        self.dead_zone = tuple(int(i) for i in dead_zone)
        self.lam_max = float(lam_max)
        self.splits = int(splits)
        self.reward_scale = float(reward_scale)
        self.dead_zone_penalty = float(dead_zone_penalty)
        self.dead_tol = float(dead_tol)
        self.action_dim = self.n_knots
        lo, hi = action_bounds
        self.action_bounds = (np.full(self.n_knots, float(lo)), np.full(self.n_knots, float(hi)))
        self.ts = np.array([self.t0 - self.T, self.t0 + self.T])
        self._init = initial_state(self.K, self.sites, self.gaps, 0.0)
        self.n_evaluations = 0

    def reset(self):
        return np.zeros(0)

    def feasible(self, action):
        """(is_feasible, total violation) of the dead-zone constraint."""
        a = np.asarray(action, dtype=float)
        viol = float(np.sum(np.maximum(0.0, np.abs(a[list(self.dead_zone)]) - self.dead_tol))) if self.dead_zone else 0.0
        return viol <= 0.0, viol

    def lambda_of_t(self, action):
        samples = np.concatenate([[0.0], np.asarray(action, dtype=float), [0.0]])

        def lam(t):
            return self.lam_max * float(np.interp(t, self.knots, samples))

        return lam

    def simulate(self, action):
        """E_N (nats) and drive work of the fixed-resolution numpy protocol."""
        from vacuum.detectors import attach_detectors, detector_block, harvested_log_negativity, protocol_evolve_fixed

        lam = self.lambda_of_t(action)

        def K_of_t(t):
            return attach_detectors(self.K, self.sites, self.gaps, lam(t))

        r = protocol_evolve_fixed(self._init.V, K_of_t, self.ts, splits=self.splits)
        neg = harvested_log_negativity(detector_block(r.V, self.N))
        return float(neg.E_N), float(r.work)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), self.action_bounds[0], self.action_bounds[1])
        self.n_evaluations += 1
        ok, viol = self.feasible(a)
        if not ok:
            return np.zeros(0), -self.dead_zone_penalty * viol, True, {"feasible": False, "violation": viol}
        E_N, work = self.simulate(a)
        return np.zeros(0), self.reward_scale * E_N, True, {"feasible": True, "E_N": E_N, "work": work}


# --------------------------------------------------------------------------
# Searches
# --------------------------------------------------------------------------


def random_search(f, bounds, n_evals, rng):
    """Uniform random search in the box; returns (best_x, best_f, history of best_f)."""
    lo, hi = (np.asarray(b, dtype=float) for b in bounds)
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    best_x, best_f = None, -math.inf
    hist = []
    for _ in range(int(n_evals)):
        x = lo + (hi - lo) * rng.random(lo.size)
        v = float(f(x))
        if v > best_f:
            best_f, best_x = v, x.copy()
        hist.append(best_f)
    return best_x, best_f, hist


def cross_entropy_method(f, mean0, std0, *, n_iter=25, pop=40, elite_frac=0.2, rng=0, bounds=None,
                         smoothing=0.7, std_floor=1e-4):
    """Maximize f by the cross-entropy method with a diagonal Gaussian policy.

    Each iteration samples ``pop`` actions from N(mean, diag(std^2)),
    clipped to ``bounds``, evaluates them, and refits (mean, std) to the top
    ``elite_frac`` fraction with exponential smoothing (de Boer et al.
    2005, Algorithm 2.1 with the smoothed update of Sec. 3.1).  Returns
    ``(best_x, best_f, mean, std, history)``; the history has one row per
    iteration (mean/std snapshots, elite threshold, best so far, number of
    evaluations).
    """
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    mean = np.asarray(mean0, dtype=float).copy()
    std = np.broadcast_to(np.asarray(std0, dtype=float), mean.shape).copy()
    n_elite = max(1, int(round(elite_frac * pop)))
    best_x, best_f = mean.copy(), -math.inf
    hist: List[Dict[str, Any]] = []
    n_evals = 0
    for it in range(int(n_iter)):
        X = mean[None, :] + std[None, :] * rng.standard_normal((int(pop), mean.size))
        if bounds is not None:
            lo, hi = (np.asarray(b, dtype=float) for b in bounds)
            X = np.clip(X, lo, hi)
        vals = np.array([float(f(x)) for x in X])
        n_evals += int(pop)
        order = np.argsort(vals)[::-1]
        elite = X[order[:n_elite]]
        if vals[order[0]] > best_f:
            best_f, best_x = float(vals[order[0]]), X[order[0]].copy()
        new_mean = elite.mean(axis=0)
        new_std = np.maximum(elite.std(axis=0), std_floor)
        mean = smoothing * new_mean + (1.0 - smoothing) * mean
        std = smoothing * new_std + (1.0 - smoothing) * std
        hist.append({"iter": it, "best_f": best_f, "elite_threshold": float(vals[order[n_elite - 1]]),
                     "mean": mean.copy(), "std": std.copy(), "n_evals": n_evals})
    return best_x, best_f, mean, std, hist


@dataclass
class Policy:
    """A fitted Gaussian policy over the action space, with the best action seen."""

    mean: np.ndarray
    std: np.ndarray
    best_action: np.ndarray
    best_reward: float
    history: List[Dict[str, Any]] = field(default_factory=list)
    n_evaluations: int = 0
    seed: Optional[int] = None

    def act(self, rng=None):
        """Sample an action (the mean when rng is None)."""
        if rng is None:
            return self.mean.copy()
        rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
        return self.mean + self.std * rng.standard_normal(self.mean.size)


def rl_search(env, cfg=None):
    """Cross-entropy search over ``env``'s action space; returns a :class:`Policy`.

    cfg keys (defaults): ``n_iter`` 25, ``pop`` 40, ``elite_frac`` 0.2,
    ``seed`` 0, ``std0`` 0.5, ``mean0`` zeros, ``smoothing`` 0.7.  The
    policy's ``best_action``/``best_reward`` are the best single evaluation
    (the env is deterministic); ``n_evaluations`` the budget spent.
    """
    cfg = dict(cfg or {})
    seed = cfg.get("seed", 0)
    mean0 = np.asarray(cfg.get("mean0", np.zeros(env.action_dim)), dtype=float)
    std0 = cfg.get("std0", 0.5)

    def f(x):
        _, r, _, _ = env.step(x)
        return r

    n0 = getattr(env, "n_evaluations", 0)
    best_x, best_f, mean, std, hist = cross_entropy_method(
        f, mean0, std0, n_iter=int(cfg.get("n_iter", 25)), pop=int(cfg.get("pop", 40)),
        elite_frac=float(cfg.get("elite_frac", 0.2)), rng=seed, bounds=env.action_bounds,
        smoothing=float(cfg.get("smoothing", 0.7)),
    )
    return Policy(mean=mean, std=std, best_action=best_x, best_reward=best_f, history=hist,
                  n_evaluations=getattr(env, "n_evaluations", 0) - n0, seed=seed)
