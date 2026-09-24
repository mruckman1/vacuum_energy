"""Toy-Jacobson hooks (Layer 4 discovery target / leap L2) -- the exact
first-law anchor only.

entanglement_equilibrium_residual(V, K, balls, mode='exact') evaluates
max_B |delta S(B) - (1/2) Tr[G_B delta V_B]| with G_B the exact modular
Hamiltonian of the vacuum of K. By the entanglement first law
(Blanco-Casini-Hung-Myers 2013; Faulkner et al. 2014) it must vanish at
first order for the true dynamics: here the residual is O(eps^2) under a
local perturbation of strength eps (ratio ~ 4 per halving) and is a small
fraction of |delta S| itself. The Bisognano-Wichmann mode and
constrain_dynamics are exercised for shape only -- their physics belongs to
the later agent.
"""

import numpy as np

from vacuum.core import ground_state_cov, harmonic_chain_K
from vacuum.geometry import (
    bw_weights,
    constrain_dynamics,
    entanglement_equilibrium_residual,
    local_energy_forms,
)

N = 40
BALLS = [list(range(10, 15)), list(range(17, 23)), list(range(18, 22)), list(range(5, 30))]


def _perturb(K, e):
    Kp = K.copy()
    Kp[N // 2, N // 2] += e
    return Kp


def test_local_energy_forms_sum_to_hamiltonian():
    K = harmonic_chain_K(N, 0.3, bc="dirichlet")
    H = sum(local_energy_forms(K))
    expected = np.block([[K, np.zeros((N, N))], [np.zeros((N, N)), np.eye(N)]])
    assert np.allclose(H, expected)
    beta = bw_weights(range(4, 9))
    assert np.allclose(beta, beta[::-1]) and beta.max() == beta[2]


def test_exact_first_law_residual_is_second_order():
    K = harmonic_chain_K(N, 0.3, bc="dirichlet")
    res, ratio = {}, {}
    for eps in (2e-3, 1e-3, 5e-4):
        V = ground_state_cov(_perturb(K, eps))
        d = entanglement_equilibrium_residual(V, K, BALLS, mode="exact", return_details=True)
        res[eps] = d["residual"]
        ratio[eps] = d["residual"] / np.max(np.abs(d["dS"]))
    assert ratio[1e-3] < 0.05, f"residual/|dS| = {ratio[1e-3]:.3e} at eps=1e-3"
    r1 = res[2e-3] / res[1e-3]
    r2 = res[1e-3] / res[5e-4]
    assert 3.4 < r1 < 4.6 and 3.4 < r2 < 4.6, \
        f"residual halving ratios {r1:.3f}, {r2:.3f} (expected ~4: O(eps^2))"
    assert entanglement_equilibrium_residual(ground_state_cov(K), K, BALLS) < 1e-12


def test_bw_mode_and_constrain_dynamics_shape():
    K = harmonic_chain_K(N, 0.3, bc="dirichlet")
    V = ground_state_cov(_perturb(K, 1e-2))
    r = entanglement_equilibrium_residual(V, K, BALLS, mode="bw")
    assert np.isfinite(r) and r >= 0.0
    L = harmonic_chain_K(N, 0.0, bc="dirichlet")
    fam = {"wave": 0.05**2 * np.eye(N) + L, "lifshitz": 0.05**2 * np.eye(N) + L @ L / 4}
    K_star, table = constrain_dynamics(fam, BALLS[:3], _perturb, eps=1e-2)
    assert table[0]["name"] in fam and len(table) == 2
    assert table[0]["normalized"] <= table[1]["normalized"]
    assert K_star is fam[table[0]["name"]]
