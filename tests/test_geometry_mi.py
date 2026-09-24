"""Mutual-information geometry (Layer 4, second instrument) -- anchors.

1. Exact Ising/Kitaev reference (vacuum.geometry.ising_exact): finite-chain
   Majorana covariance against the qet.chains exact diagonalization (block
   entropies to 1e-12), the infinite-chain closed form g_r = -2/(pi(2r+1))
   giving <Z> = 2/pi and e0 = -4/pi (Pfeuty 1970), and the c = 1/2
   Calabrese-Cardy block entropy S = (c/3) ln l + const (fit over
   l = 2..256).
2. Swingle (2012, Sec. V): on the critical chain the mutual-information
   distance grows as d = -ln I ~ 4 Delta_min ln r, with Delta_min = 1/2 for
   the Majorana fermion -> slope 2 (measured 2.06); the hyperbolic
   embedding beats the flat one, with an O(1) curvature radius; off
   criticality (xi ~ 3) the fitted curvature radius diverges and the
   linear law wins (Swingle Sec. IV: the geometry ends at the
   factorization scale).
3. Gromov delta on synthetic metrics: a tree metric has delta = 0, the
   boundary-of-H2 metric 2 ln|x - y| has delta <= (ln 2)/2 for any number
   of points (four-point computation), a random metric is far from
   tree-like.
4. failure_map runs on every family with the required diagnostics.
"""

import numpy as np
import pytest

from vacuum.core import ground_state_cov, harmonic_chain_K
from vacuum.geometry import (
    MODEL_FAMILIES,
    boson_correlation_length,
    distance_law,
    failure_map,
    geometry_report,
    hyperbolicity,
    kitaev_chain_gamma,
    majorana_block_entropy,
    mi_distance,
    mi_matrix,
    reconstruct_metric,
    tfim_block_entropy,
    tfim_correlation_length,
    tfim_majorana_g,
    tfim_mutual_information,
)
from vacuum.geometry.ising_exact import E0_TFIM
from vacuum.qet.chains import reduced_density, tfim_ground_state


# ---------------------------------------------------------------------------
# 1. the exact reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g", [1.0, 1.5, 0.6])
def test_majorana_matches_exact_diagonalization(g):
    L = 10
    E0, psi = tfim_ground_state(L, g)
    G = kitaev_chain_gamma(L, g)
    E_maj = -sum(-G[2 * j + 1, 2 * j + 2] for j in range(L - 1)) \
        - g * sum(-G[2 * j, 2 * j + 1] for j in range(L))
    assert abs(E_maj - E0) < 1e-10, f"E0 majorana {E_maj} vs ED {E0}"
    for sites in ([0], [4], [0, 1, 2], [3, 4, 5, 6], [2, 3, 4, 5, 6, 7]):
        rho = reduced_density(psi, L, sites)
        ev = np.linalg.eigvalsh(rho)
        ev = ev[ev > 1e-14]
        S_ed = -float(np.sum(ev * np.log(ev)))
        S_maj = majorana_block_entropy(G, sites)
        assert abs(S_ed - S_maj) < 1e-12, f"block {sites}: ED {S_ed} vs Majorana {S_maj}"


def test_infinite_chain_closed_forms():
    assert abs(-tfim_majorana_g([0], 1.0)[0] - 2 / np.pi) < 1e-14
    assert abs(E0_TFIM(1.0) + 4 / np.pi) < 1e-10
    # the quadrature branch is continuous with the closed form as g -> 1
    # (the shift is O(|1-g| ln|1-g|) ~ 5e-3 at |1-g| = 2e-3)
    q = tfim_majorana_g([0, 1, 3], 1.0 + 2e-3)
    assert np.allclose(q, -2 / (np.pi * np.array([1, 3, 7])), atol=1e-2)
    # quadrature branch vs the finite-chain ground state off criticality
    Gf = kitaev_chain_gamma(400, 0.8)
    assert abs(Gf[400, 403] - tfim_majorana_g([1], 0.8)[0]) < 1e-6
    # large open chain centre approaches the infinite-chain value
    N = 600
    G = kitaev_chain_gamma(N, 1.0)
    c = N // 2
    assert abs(G[2 * c, 2 * (c + 1) + 1] - tfim_majorana_g([1], 1.0)[0]) < 2e-3


def test_central_charge_one_half_block_entropy():
    ls = np.array([2, 4, 8, 16, 32, 64, 128, 256])
    S = np.array([tfim_block_entropy(range(l), 1.0) for l in ls])
    A = np.vstack([np.log(ls), np.ones_like(ls)]).T
    a, b = np.linalg.lstsq(A, S, rcond=None)[0]
    c = 3 * a
    assert abs(c - 0.5) < 3e-3, f"c = {c:.5f} from S(l) = (c/3) ln l + {b:.4f}"
    assert abs(tfim_block_entropy([0], 1.0) - 0.4739) < 2e-3


def test_correlation_length_off_criticality():
    for g in (0.9, 0.8):
        xi = tfim_correlation_length(g)
        rs = np.array([8, 16, 32, 48])
        I = np.array([tfim_mutual_information([0], [r], g) for r in rs])
        rate = -np.polyfit(rs, np.log(I), 1)[0] / 2.0
        # I ~ exp(-2 r / xi) times a power law: the fitted rate is within 50% of 1/xi
        assert 0.5 / xi < rate < 2.0 / xi, f"g={g}: MI decay rate {rate:.3f} vs 1/xi = {1/xi:.3f}"


# ---------------------------------------------------------------------------
# 2. Swingle's picture on the critical chain
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def critical_report():
    n, sp = 24, 2
    regs = [[k * sp] for k in range(n)]
    pos = [k * sp for k in range(n)]
    I = mi_matrix(None, regs, "tfim_inf", g=1.0)
    return geometry_report(I, pos), I


def test_critical_log_distance_law(critical_report):
    rep, I = critical_report
    assert abs(rep["slope_log"] - 2.0) < 0.15, f"slope {rep['slope_log']:.3f} vs 4 Delta_psi = 2"
    assert rep["r2_log"] > 0.998 and rep["r2_log"] > rep["r2_lin"] + 0.1, \
        f"R2 log {rep['r2_log']:.4f}, lin {rep['r2_lin']:.4f}"
    assert rep["floor_fraction"] == 0.0 and rep["scatter"] < 1e-12


def test_critical_hyperbolic_beats_flat(critical_report):
    rep, I = critical_report
    assert rep["stress_hyp"] < rep["stress_flat"] - 0.02, \
        f"stress hyp {rep['stress_hyp']:.4f} vs flat {rep['stress_flat']:.4f}"
    assert rep["distortion_hyp"] < rep["distortion_flat"]
    assert 0.5 < rep["curvature_radius"] < 5.0, rep["curvature_radius"]
    assert rep["stress_hyp"] < 0.08


def test_off_critical_flattens():
    n, sp = 24, 2
    regs = [[k * sp] for k in range(n)]
    pos = [k * sp for k in range(n)]
    rep = geometry_report(mi_matrix(None, regs, "tfim_inf", g=0.7), pos)
    assert rep["r2_lin"] > rep["r2_log"], (rep["r2_lin"], rep["r2_log"])
    assert rep["curvature_radius"] > 50, f"R = {rep['curvature_radius']:.1f} should diverge at xi ~ 3"
    assert rep["hyperbolic_advantage"] < 0.01


def test_massive_boson_flat():
    n, sp = 20, 2
    regs = [[k * sp] for k in range(n)]
    pos = [k * sp for k in range(n)]
    V = ground_state_cov(harmonic_chain_K(160, 0.2))
    rep = geometry_report(mi_matrix(V, regs, "gaussian"), pos)
    assert rep["r2_lin"] > 0.99 and rep["r2_lin"] > rep["r2_log"]
    assert abs(rep["slope_lin"] - 2 / boson_correlation_length(0.2)) < 0.15, \
        f"slope {rep['slope_lin']:.3f} vs 2/xi = {2/boson_correlation_length(0.2):.3f}"


# ---------------------------------------------------------------------------
# 3. embeddings and Gromov delta on synthetic metrics
# ---------------------------------------------------------------------------


def test_mi_distance_conventions():
    I = np.array([[0, 1.0, 0.1, 0.0], [1.0, 0, 0.2, np.nan],
                  [0.1, 0.2, 0, 0.5], [0.0, np.nan, 0.5, 0]])
    D, info = mi_distance(I, floor=1e-3)
    assert D[0, 1] == 0.0 and abs(D[0, 2] - np.log(10)) < 1e-12
    assert np.isnan(D[1, 3]) and abs(D[0, 3] - np.log(1e3)) < 1e-12
    assert info["I_max"] == 1.0 and abs(info["floor_fraction"] - 1 / 5) < 1e-12


def test_gromov_delta_tree_and_h2_boundary():
    x = np.arange(1, 13, dtype=float)
    D_line = np.abs(x[:, None] - x[None, :])
    assert hyperbolicity(D_line)[0] < 1e-12
    xs = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233], dtype=float)
    D_log = 2 * np.log(np.abs(xs[:, None] - xs[None, :]) + np.eye(12))
    np.fill_diagonal(D_log, 0.0)
    d, drel = hyperbolicity(D_log)
    assert d <= np.log(2) + 1e-9, d      # 2 ln|x-y| is (ln 2)-hyperbolic
    rng = np.random.default_rng(0)
    R = rng.uniform(1, 2, (12, 12))
    R = np.triu(R, 1) + np.triu(R, 1).T
    assert hyperbolicity(R)[1] > 0.1


def test_embeddings_recover_planar_points():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((15, 2))
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    Y, info = reconstruct_metric(D, dim=2, method="mds")
    assert info["stress"] < 1e-6 and info["distortion"] < 1 + 1e-4
    Yh, hinfo = reconstruct_metric(D, dim=2, method="hyperbolic")
    assert hinfo["stress"] < 0.02


# ---------------------------------------------------------------------------
# 4. failure_map smoke on every family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family,grid", [
    ("massive_boson", [0.05, 0.5]),
    ("disordered_boson", [0.0, 0.5]),
    ("kitaev_tfim", [1.0, 0.8]),
    ("disordered_kitaev", [0.0, 0.5]),
])
def test_failure_map_runs(family, grid):
    rows = failure_map(family, grid, n_points=12, spacing=2, N=120, seeds=(0, 1))
    assert len(rows) == len(grid)
    for r in rows:
        for k in ("stress_flat", "stress_hyp", "curvature_radius", "delta_rel",
                  "r2_log", "r2_lin", "scatter", "floor_fraction", "xi"):
            assert k in r
        assert np.isfinite(r["stress_flat"]) and np.isfinite(r["delta_rel"])
    assert set(MODEL_FAMILIES) == {"massive_boson", "disordered_boson",
                                   "kitaev_tfim", "disordered_kitaev"}
