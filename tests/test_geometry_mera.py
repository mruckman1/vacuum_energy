"""Scale-invariant ternary MERA for the critical Ising chain -- Layer 4 anchors.

Model: H = -sum XX - sum Z (g = J = 1), exact e0 = -4/pi (Pfeuty 1970).
Construction and optimization: Evenbly-Vidal, PRB 79, 144108 (2009)
(ternary MERA, Eqs. 11-13, 34, 40, 63-67, 70-72); conformal data:
Pfeifer-Evenbly-Vidal, PRA 79, 040301(R) (2009) (Eqs. 9, 16, Table I).

The optimized tensors are loaded from vacuum/geometry/data (chi = 8 and
chi = 6, Z2-symmetric, produced by ``mera_ising(chi, cache='refresh')`` with
the sweep counts recorded in the JSON sidecars); the tests below never
re-optimize beyond a short smoke run.

Anchors and the tolerances stated for chi = 8 (literature comparators:
Evenbly-Vidal Fig. 24, chi = 4 -> 5 digits in the energy after a long
optimization; Pfeifer-Evenbly-Vidal Table I, chi = 22 -> Delta_sigma =
0.124997 (0.002%), Delta_epsilon = 1.0001 (0.01%), c = 0.5007 (0.14%)):

1. energy density vs -4/pi within 5e-5 relative (measured value in the
   assertion message);
2. scaling dimensions from the parity-resolved two-site scaling
   superoperator: Delta_sigma = 1/8 within 4%, Delta_epsilon = 1 within 3%,
   the descendants (9/8 twice, 2 four times) within 8%;
3. central charge (PEV Eq. 16) within 3% of 1/2, and the exact one- and
   two-site block entropies of the critical chain (0.4739, 0.5934 nats)
   within 1%;
4. geometry: graph geodesics grow as 6 log_3 r bonds (2 layers of 3 bonds),
   the bond-count minimal cut of a block of 3^k sites crosses 4k - 1 bonds
   (four per layer), Vidal's ln-chi bound holds, and the
   minimal-cut entropy increment per tripling equals (c_MERA/3) ln 3, i.e.
   the (c/3) ln l law with c = 1/2 to within the c accuracy above;
5. the diagram algebra: unital ascending, trace-preserving descending,
   ascend/descend duality and environment/energy consistency to 1e-12.
"""

import numpy as np
import pytest

from vacuum.geometry import (
    E0_ISING_EXACT,
    block_entropies_level0,
    central_charge_estimate,
    energy_density,
    fixed_point_density,
    load_mera,
    mera_geodesic,
    mera_ising,
    minimal_cut,
    minimal_cut_entropy,
    optimize_mera,
    random_mera,
    scaling_dimensions,
    tfim_block_entropy,
)
from vacuum.geometry.mera import (
    DATA_DIR,
    ascend,
    ascend_avg,
    descend,
    environment,
    local_init_mera,
    scaling_dimensions_by_parity,
)

S1_EXACT = 0.47390  # single-spin entropy of the critical chain, <Z> = 2/pi
S2_EXACT = 0.59341  # two-site block entropy (free-fermion exact)


@pytest.fixture(scope="module")
def mera8():
    path = DATA_DIR / "mera_ising_chi8.npz"
    if not path.exists():
        pytest.skip("chi = 8 cache missing: run mera_ising(8, cache='refresh')")
    return load_mera(path)


@pytest.fixture(scope="module")
def mera6():
    path = DATA_DIR / "mera_ising_chi6.npz"
    if not path.exists():
        pytest.skip("chi = 6 cache missing")
    return load_mera(path)


@pytest.fixture(scope="module")
def rho8(mera8):
    rho, it, delta = fixed_point_density(mera8.u_s, mera8.w_s, n_iter=400)
    assert delta < 1e-9, f"fixed point not converged: {delta}"
    return rho


# ---------------------------------------------------------------------------
# 5. diagram algebra
# ---------------------------------------------------------------------------


def test_diagram_algebra():
    rng = np.random.default_rng(3)
    m = random_mera(4, seed=1, u_identity=False)
    u, w = m.u_s, m.w_s
    chi = 4
    I2 = np.eye(chi * chi).reshape(chi, chi, chi, chi)
    R = rng.standard_normal((chi * chi, chi * chi))
    rho = (R @ R.T)
    rho = (rho / np.trace(rho)).reshape(chi, chi, chi, chi)
    O = rng.standard_normal((chi * chi, chi * chi))
    o = (O + O.T).reshape(chi, chi, chi, chi)
    for p in "LCR":
        assert np.abs(ascend(I2, u, w, p) - I2).max() < 1e-12, p       # unital
        D = descend(rho, u, w, p)
        assert abs(np.einsum("abab", D) - 1.0) < 1e-12, p              # trace
        lhs = np.einsum("STst,stST", o, D)
        rhs = np.einsum("STst,stST", ascend(o, u, w, p), rho)
        assert abs(lhs - rhs) < 1e-12, (p, lhs, rhs)                   # duality
    e = np.einsum("STst,stST", ascend_avg(o, u, w), rho)
    assert abs(np.sum(environment("u", o, rho, u, w) * u) - e) < 1e-12
    assert abs(np.sum(environment("w", o, rho, u, w) * w) / 2 - e) < 1e-12


def test_optimizer_lowers_energy_smoke():
    m = local_init_mera(4)
    e_init = energy_density(m)
    m2, hist = optimize_mera(m, n_sweeps=40, n_hbar=3)
    e_fin = energy_density(m2)
    err_init, err_fin = e_init - E0_ISING_EXACT, e_fin - E0_ISING_EXACT
    assert err_init > 0 and err_fin > 0, "variational bound"
    assert err_fin < 0.5 * err_init, f"energy error {err_init:.3e} -> {err_fin:.3e} in 40 sweeps"
    assert np.all(np.diff(hist["energy"][5:]) < 1e-3)   # monotone after warm-up
    # isometry / unitarity preserved
    w = m2.w_s.reshape(-1, m2.chi)
    assert np.abs(w.T @ w - np.eye(m2.chi)).max() < 1e-12
    u = m2.u_s.reshape(m2.chi**2, -1)
    assert np.abs(u.T @ u - np.eye(m2.chi**2)).max() < 1e-12


def test_mera_ising_loads_cache(mera8):
    m = mera_ising(chi=8, cache="auto")
    assert m.chi == 8 and m.n_transitional == 2
    assert np.allclose(m.w_s, mera8.w_s)
    assert m.parities is not None and m.meta.get("n_sweeps_done", 0) >= 1000


# ---------------------------------------------------------------------------
# 1. energy
# ---------------------------------------------------------------------------


def test_energy_density_chi8(mera8, rho8):
    e = energy_density(mera8, rho_hat=rho8)
    rel = (e - E0_ISING_EXACT) / abs(E0_ISING_EXACT)
    assert rel > -1e-12, f"variational bound violated: e = {e:.10f} < -4/pi"
    assert rel < 5e-5, f"chi=8 energy density {e:.8f}, exact {E0_ISING_EXACT:.8f}, rel err {rel:.3e}"


def test_energy_improves_with_chi(mera6, mera8):
    r6 = (energy_density(mera6) - E0_ISING_EXACT) / abs(E0_ISING_EXACT)
    r8 = (energy_density(mera8) - E0_ISING_EXACT) / abs(E0_ISING_EXACT)
    assert 0 < r6 < 3e-5, f"chi=6 rel err {r6:.3e}"
    assert r8 < r6 * 1.5, f"chi=8 rel err {r8:.3e} not better than chi=6 {r6:.3e}"


# ---------------------------------------------------------------------------
# 2. scaling dimensions
# ---------------------------------------------------------------------------


def test_scaling_dimensions_chi8(mera8):
    sp = scaling_dimensions_by_parity(mera8, n=6, which="two-site")
    assert sp["leakage"] < 1e-12
    even, odd = sp["even"], sp["odd"]
    assert abs(even[0]) < 1e-8, "identity must have Delta = 0"
    d_sigma, d_eps = odd[0], even[1]
    assert abs(d_sigma - 0.125) / 0.125 < 0.04, \
        f"Delta_sigma = {d_sigma:.5f} (exact 1/8; PEV chi=22: 0.124997)"
    assert abs(d_eps - 1.0) < 0.03, \
        f"Delta_epsilon = {d_eps:.5f} (exact 1; PEV chi=22: 1.0001)"
    # descendants: d sigma, dbar sigma at 9/8; four operators at 2
    assert np.all(np.abs(odd[1:3] - 1.125) < 0.09), f"odd spectrum {np.round(odd, 4)}"
    assert np.all(np.abs(even[2:6] - 2.0) < 0.16), f"even spectrum {np.round(even, 4)}"
    # unlabeled full spectrum contains the same leading values
    d_all, _ = scaling_dimensions(mera8, n=4, which="two-site")
    assert abs(d_all[1] - d_sigma) < 1e-6 and abs(d_all[2] - d_eps) < 1e-6


def test_one_site_superoperator_consistent(mera8):
    sp1 = scaling_dimensions_by_parity(mera8, n=3, which="one-site")
    assert abs(sp1["odd"][0] - 0.125) < 0.02, f"one-site Delta_sigma {sp1['odd'][0]:.4f}"
    assert abs(sp1["even"][1] - 1.0) < 0.08, f"one-site Delta_epsilon {sp1['even'][1]:.4f}"


# ---------------------------------------------------------------------------
# 3. central charge and block entropies
# ---------------------------------------------------------------------------


def test_central_charge_and_entropies_chi8(mera8, rho8):
    c, S2, S1 = central_charge_estimate(mera8, rho_hat=rho8)
    assert abs(c - 0.5) < 0.015, f"c_MERA = {c:.4f} (PEV Eq. 16; exact 1/2, PEV chi=22: 0.5007)"
    ent = block_entropies_level0(mera8, rho_hat=rho8)
    S2v = np.array(list(ent["S2"].values()))
    S1v = np.array([v for pair in ent["S1"].values() for v in pair])
    assert np.all(np.abs(S2v - S2_EXACT) / S2_EXACT < 0.01), f"S(2) = {S2v} vs exact {S2_EXACT}"
    assert np.all(np.abs(S1v - S1_EXACT) / S1_EXACT < 0.01), f"S(1) = {S1v} vs exact {S1_EXACT}"
    assert abs(tfim_block_entropy([0, 1], 1.0) - S2_EXACT) < 1e-4


# ---------------------------------------------------------------------------
# 4. geometry
# ---------------------------------------------------------------------------


def test_geodesic_and_minimal_cut_scaling(mera8):
    seps = np.array([3, 9, 27, 81])
    geo = np.array([mera_geodesic(mera8, 0, int(r), n_layers=5) for r in seps])
    assert np.all(np.diff(geo) > 0)
    assert geo[-1] - geo[-2] == 6, f"one more layer costs 6 bonds, got {geo}"
    A = np.vstack([np.log(seps) / np.log(3), np.ones_like(seps)]).T
    a, b = np.linalg.lstsq(A, geo, rcond=None)[0]
    assert abs(a - 6.0) < 1.0, f"geodesic slope {a:.3f} bonds per log_3 r (expected 6)"
    for k in range(1, 6):
        bound, cut = minimal_cut(mera8, range(3**k), n_layers=6, weights="count")
        assert len(cut) == 4 * k - 1, (k, len(cut))
        bound_v, _ = minimal_cut(mera8, range(3**k), n_layers=6, weights="lnchi")
        assert bound >= bound_v >= tfim_block_entropy(range(3**k), 1.0), "Vidal bound violated"


def test_minimal_cut_entropy_reproduces_c_over_3_log(mera8, rho8):
    c, _, _ = central_charge_estimate(mera8, rho_hat=rho8)
    est = {l: minimal_cut_entropy(mera8, range(l), n_layers=6, rho_hat=rho8)[0]
           for l in (3, 9, 27, 81, 243)}
    exact = {l: tfim_block_entropy(range(l), 1.0) for l in est}
    for l1, l2 in ((3, 9), (9, 27), (27, 81), (81, 243)):
        d_cut = est[l2] - est[l1]
        d_ex = exact[l2] - exact[l1]
        assert abs(d_cut - d_ex) / d_ex < 0.03, \
            f"S_cut({l2})-S_cut({l1}) = {d_cut:.4f} vs exact {d_ex:.4f} = (1/6) ln 3; c_MERA = {c:.4f}"
    d_cut = est[243] - est[3]
    assert abs(d_cut - (0.5 / 3) * np.log(81)) / ((0.5 / 3) * np.log(81)) < 0.03
