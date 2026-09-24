"""Layer-6: the derivative-free search for rugged landscapes (vacuum.opt.rl).

The reference environment's landscape is rugged on purpose: a hard dead
zone in the switching window (two knots on which the coupler must be off,
|a| <= 0.05, else the action is not simulated and is penalized), inside
which the harvested negativity is flat-zero over most of the box and
positive only on a thin feasible ridge.  At an EQUAL evaluation budget
(1000 numpy simulations), the cross-entropy method must beat uniform
random search on every seed — random search almost never lands in the
feasible slice, CEM shrinks onto it.

Pure numpy; ~5 s.
"""

from __future__ import annotations

import numpy as np
import pytest

from vacuum.opt.rl import HarvestingEnv, cross_entropy_method, random_search, rl_search

SEEDS = (0, 1, 2)


def test_env_contract_and_dead_zone():
    env = HarvestingEnv()
    assert env.action_dim == 9 and env.reset().shape == (0,)
    a = np.array([0.3, 0.6, 0.8, 0.9, 0.0, 0.0, 0.9, 0.6, 0.3])
    obs, r, done, info = env.step(a)
    assert done and info["feasible"] and r >= 0.0 and info["work"] > 0.0
    bad = np.ones(9)
    _, r_bad, _, info_bad = env.step(bad)
    assert not info_bad["feasible"] and r_bad < 0.0
    assert abs(r_bad + env.dead_zone_penalty * info_bad["violation"]) < 1e-14
    # the dead zone is exactly the declared knots
    ok, viol = env.feasible(np.array([1, 1, 1, 1, 0.04, -0.05, 1, 1, 1]))
    assert ok and viol == 0.0


def test_cem_beats_random_search_on_every_seed():
    rows = []
    for seed in SEEDS:
        env_c = HarvestingEnv()
        pol = rl_search(env_c, dict(seed=seed, n_iter=25, pop=40))
        env_r = HarvestingEnv()
        _, best_r, _ = random_search(lambda x: env_r.step(x)[1], env_r.action_bounds, pol.n_evaluations,
                                     np.random.default_rng(seed))
        assert env_r.n_evaluations == pol.n_evaluations == 1000
        rows.append((seed, pol.best_reward, best_r))
        assert pol.best_reward > best_r, f"seed {seed}: CEM {pol.best_reward:.4f} <= random {best_r:.4f}"
        assert pol.best_reward > 1.0, f"seed {seed}: CEM found no harvesting ridge ({pol.best_reward:.4f})"
        ok, _ = env_c.feasible(pol.best_action)
        assert ok
        assert len(pol.history) == 25 and pol.history[-1]["n_evals"] == 1000
    print("\nCEM vs random (reward = 1e3 E_N): " + "; ".join(f"seed {s}: {c:.3f} vs {r:.3f}" for s, c, r in rows))


def test_cem_on_a_smooth_function_is_reproducible():
    f = lambda x: -float(np.sum((x - 0.3) ** 2))
    bx1, bf1, m1, s1, _ = cross_entropy_method(f, np.zeros(3), 0.5, n_iter=40, pop=30, rng=7)
    bx2, bf2, m2, s2, _ = cross_entropy_method(f, np.zeros(3), 0.5, n_iter=40, pop=30, rng=7)
    assert np.array_equal(bx1, bx2) and bf1 == bf2 and np.array_equal(m1, m2)
    assert np.max(np.abs(m1 - 0.3)) < 5e-2 and bf1 > -1e-2
