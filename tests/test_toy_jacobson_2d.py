"""Regression tests for the 2+1 toy Jacobson (papers/toy-jacobson, section J).

The 1+1 toy (tests/test_toy_jacobson.py) pairs delta S with the local CHM form
built from the lattice's own energy density and finds r -> 0 for the wave chain.
This file pins the 2+1 extension, `vacuum.geometry.jacobson2d`, and the one
structural fact it uncovered:

  * with the CANONICAL energy density the 2+1 residual is 0.36-0.45 and does not
    decay with the disk radius (dS exceeds d<K_loc> by 48-82 %);
  * the whole deficit is the free scalar's CONFORMAL IMPROVEMENT term, which
    vanishes identically in d = 2 (xi = (d-2)/(4(d-1)) = 0) and is 1/8 in d = 3;
    fitting it gives xi = 0.1278 +- 0.0040;
  * with xi = 1/8 the residual falls to 0.005-0.075, the free radial weight
    recovers the CHM parabola at normalisation ~ 1, the z = 2 Lifshitz lattice
    stays obstructed, and -- the point of Layer 7 unit 2 item (a) -- the
    reflection-EVEN sector agrees with the odd one to within a factor 1.8-24,
    against ~2500-4500 in 1+1, where the IR zero mode lives.

Live checks run at N <= 64 and R <= 5 (the whole file is ~30 s); the archived
values come from papers/toy-jacobson/data/J_2p1_*.json (N = 128, R = 3..10).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from vacuum.core import entanglement_hamiltonian
from vacuum.geometry.jacobson import squeeze_variation
from vacuum.geometry.jacobson2d import (
    XI_IMPROVE_3D,
    chm_beta_2d,
    disk_ball,
    equilibrium_residual_2d,
    first_law_audit_2d,
    free_weight_residual_2d,
    lattice_coords,
    local_modular_form_2d,
    sine_basis,
    square_lattice_K,
    squeeze_modes_2d,
    vacuum_block_2d,
)

DATA = Path(__file__).resolve().parents[1] / "papers" / "toy-jacobson" / "data"


def _load(name):
    p = DATA / name
    if not p.exists():
        pytest.skip(f"{p} not present (run notebook_2p1.py)")
    return json.load(open(p))


# ---------------------------------------------------------------------------
# the machinery
# ---------------------------------------------------------------------------


def test_analytic_product_modes_match_the_dense_lattice():
    """The sine basis diagonalises K, and the disk covariance from the analytic
    modes equals the one from a dense eigendecomposition."""
    N = 16
    S, lam1 = sine_basis(N)
    Kd = square_lattice_K(N, z=1, dense=True)
    lam = np.linalg.eigvalsh(Kd)
    assert np.abs(lam - np.sort((lam1[:, None] + lam1[None, :]).ravel())).max() < 1e-12
    u = np.outer(S[:, 3], S[:, 1]).ravel()
    assert abs(u @ u - 1.0) < 1e-12
    assert np.abs(Kd @ u - (lam1[3] + lam1[1]) * u).max() < 1e-12
    w, U = np.linalg.eigh(Kd)
    Vxx = 0.5 * (U * w ** -0.25) @ (U * w ** -0.25).T
    Vpp = 0.5 * (U * w ** 0.25) @ (U * w ** 0.25).T
    disk = disk_ball(N, 3)
    sites = disk["sites"]
    V = vacuum_block_2d(N, sites, z=1)
    n = len(sites)
    assert np.abs(V[:n, :n] - Vxx[np.ix_(sites, sites)]).max() < 1e-12
    assert np.abs(V[n:, n:] - Vpp[np.ix_(sites, sites)]).max() < 1e-12


def test_the_local_energy_density_sums_to_the_hamiltonian_sector_by_sector():
    """With weight 1 on every site, K_loc IS H: the xx and pp sectors of the
    response to a normal-mode squeeze are +omega/2 and -omega/2 separately."""
    N = 16
    K = square_lattice_K(N, z=1)
    S, lam1 = sine_basis(N)
    u = np.outer(S[:, 5], S[:, 0]).ravel()
    w = float(np.sqrt(lam1[5] + lam1[0]))
    allsites = list(range(N * N))
    whole = {"sites": allsites, "r": np.zeros(N * N), "R": 1.0, "centre": (0.0, 0.0)}
    G = local_modular_form_2d(K, whole, N, sites=allsites, beta_at="site",
                              weight_fn=lambda r: np.ones_like(r))
    dV = squeeze_variation(u, w)
    M = N * N
    assert 0.5 * np.trace(G[:M, :M] @ dV[:M, :M]) == pytest.approx(0.5 * w, rel=1e-12)
    assert 0.5 * np.trace(G[M:, M:] @ dV[M:, M:]) == pytest.approx(-0.5 * w, rel=1e-12)
    assert abs(0.5 * np.trace(G @ dV)) < 1e-14


def test_the_disk_variations_have_the_stated_reflection_parity():
    N, R = 32, 4
    for parity, sign in (("odd", -1.0), ("even", +1.0)):
        for mode in squeeze_modes_2d(N, R, ratios=(4,), parity=parity):
            u = mode["u"].reshape(N, N)
            assert np.sum(u * u[::-1, :]) / np.sum(u * u) == pytest.approx(sign, abs=1e-12)


def test_first_law_holds_in_2p1_by_finite_differences():
    """dS from the Williamson decomposition against the entropy of the exactly
    squeezed disk state, both parities, wave and Lifshitz."""
    for z in (1, 2):
        for parity in ("odd", "even"):
            a = first_law_audit_2d(48, 3, ratio=4, parity=parity, z=z)
            assert a["rel_err"] < 1e-7, (z, parity, a)
    arch = _load("J_2p1_controls.json")
    assert arch["audit_rel_err_max"] < 1e-7
    assert arch["audit_rel_err_max"] == pytest.approx(2.4366568e-08, rel=1e-3)


# ---------------------------------------------------------------------------
# the canonical ansatz fails in 2+1, and the deficit is the improvement term
# ---------------------------------------------------------------------------


def test_canonical_chm_ansatz_fails_in_2p1_and_does_not_improve_with_R():
    """Archived: r = 0.357-0.451 over R = 3..10 at N = 128, with no decay."""
    J = _load("J_2p1_summary.json")
    per_R = J["J1_residual"]["wave_odd_xi0_site"]["per_R"]
    vals = {int(k): v for k, v in per_R.items()}
    assert vals[3] == pytest.approx(0.45118, rel=2e-3)
    assert vals[5] == pytest.approx(0.37047, rel=2e-3)
    assert vals[10] == pytest.approx(0.38172, rel=2e-3)
    assert min(vals.values()) > 0.30 and max(vals.values()) < 0.50
    # no decay: the largest R is not better than the smallest by more than 25 %
    assert vals[10] > 0.75 * vals[3]
    d = J["J2_dK_over_dS_canonical_odd"]
    assert d["min"] == pytest.approx(0.5488214, rel=1e-4)
    assert d["max"] == pytest.approx(0.6760269, rel=1e-4)
    # live, small: the failure is not an artefact of the archive's knobs
    r = equilibrium_residual_2d(64, radii=(4,), ratios=(4, 8), parity="odd",
                               beta_at="site", improve=0.0)
    assert r > 0.30


def test_the_deficit_is_the_conformal_improvement_and_xi_is_recovered():
    """Fitting dS = d<K_can> + 2 pi xi (2/R) sum_B d<phi^2> gives the free
    scalar's xi = (d-2)/(4(d-1)) = 1/8 in d = 3, in BOTH parity sectors."""
    J = _load("J_2p1_summary.json")
    odd, even = J["J2_xi_fit_odd_R_ge_4"], J["J2_xi_fit_even_R_ge_4"]
    assert XI_IMPROVE_3D == 0.125
    assert odd["mean"] == pytest.approx(0.1278140, rel=1e-4)
    assert odd["sd"] == pytest.approx(0.0040032, rel=1e-3)
    assert even["mean"] == pytest.approx(0.1322030, rel=1e-4)
    # the fitted coefficient is 1/8 to better than 6 %, cell by cell to 12 %
    assert abs(odd["mean"] - 0.125) < 0.008
    assert abs(even["mean"] - 0.125) < 0.010
    assert 0.115 < odd["min"] and odd["max"] < 0.140
    assert odd["n"] >= 15
    # live: one cell, independently of the archive
    imp = _load("J_2p1_improvement.json")
    rows = [r for r in imp["rows"]
            if r["dynamics"] == "wave" and r["parity"] == "odd" and r["R"] == 5.0]
    assert rows and all(abs(r["xi_fit"] - 0.125) < 0.02 for r in rows)


def test_the_improved_ansatz_lowers_the_residual_by_more_than_an_order():
    J = _load("J_2p1_summary.json")
    can = {int(k): v for k, v in J["J1_residual"]["wave_odd_xi0_site"]["per_R"].items()}
    imp = {int(k): v for k, v in J["J1_residual"]["wave_odd_xi0.125_site"]["per_R"].items()}
    assert imp[3] == pytest.approx(0.074869, rel=2e-3)
    assert imp[5] == pytest.approx(0.010194, rel=2e-3)
    assert imp[10] == pytest.approx(0.0051832, rel=2e-3)
    assert max(imp[R] for R in (4, 5, 6, 8, 10)) < 0.030
    for R in can:
        assert imp[R] < can[R] / 5.0
    # the bond-centred discretisation of beta gives the same answer to 70 %
    bond = {int(k): v for k, v in J["J1_residual"]["wave_odd_xi0.125_bond"]["per_R"].items()}
    for R in (4, 5, 6, 8, 10):
        assert 0.3 < bond[R] / imp[R] < 3.0
    # live
    r = equilibrium_residual_2d(64, radii=(4,), ratios=(4, 8), parity="odd",
                               beta_at="site", improve=XI_IMPROVE_3D)
    assert r < 0.05


def test_the_improvement_discretisation_is_not_free():
    """Putting the improvement's Laplacian on the FIELD (the discrete
    -xi sum_B beta (grad^2 phi^2)) keeps an entangling-circle contact term and
    does NOT reproduce the first law; putting it on the WEIGHT does."""
    J = _load("J_2p1_summary.json")
    f = J["J5_improve_forms"]
    assert f["weight"]["residual"] == pytest.approx(0.0748691, rel=1e-3)
    assert f["laplacian"]["residual"] == pytest.approx(0.7255893, rel=1e-3)
    assert f["laplacian"]["residual"] > 5.0 * f["weight"]["residual"]


# ---------------------------------------------------------------------------
# the contrast with non-relativistic dynamics, and the free weight
# ---------------------------------------------------------------------------


def test_lifshitz_z2_stays_obstructed_in_2p1():
    J = _load("J_2p1_summary.json")
    can = {int(k): v for k, v in J["J1_residual"]["lifshitz_z2_odd_xi0_site"]["per_R"].items()}
    imp = {int(k): v for k, v in J["J1_residual"]["lifshitz_z2_odd_xi0.125_site"]["per_R"].items()}
    wav = {int(k): v for k, v in J["J1_residual"]["wave_odd_xi0.125_site"]["per_R"].items()}
    assert can[3] == pytest.approx(0.98633, rel=2e-3)
    assert can[10] == pytest.approx(0.95478, rel=2e-3)
    assert min(can.values()) > 0.90
    # with the CFT ansatz (xi = 1/8) the Lifshitz residual GROWS with the disk
    assert imp[10] == pytest.approx(1.26392, rel=2e-3)
    assert imp[10] > 4.0 * imp[4]
    # the contrast at the two largest disks
    assert imp[8] / wav[8] > 30.0
    assert imp[10] / wav[10] > 200.0


def test_free_radial_weight_recovers_the_chm_parabola_in_2p1():
    """Two even Chebyshev modes in r/R: the CHM shape is recovered either way,
    but only WITH the improvement term is its normalisation ~ 1."""
    F = _load("J_2p1_free_weight.json")
    for R, expect in (("4", 1.6732), ("6", 1.6017), ("8", 1.6628)):
        c = F["wave_canonical"][R]["per_n_modes"]["2"]
        assert c["overlap_with_chm"] > 0.99
        assert c["normalisation"] == pytest.approx(expect, rel=3e-3)
        assert c["normalisation"] > 1.55
    for R, expect, rfree in (("4", 0.7792, 4.179e-04), ("6", 0.8041, 5.030e-04),
                             ("8", 0.9866, 1.858e-04)):
        c = F["wave_improved"][R]["per_n_modes"]["2"]
        assert c["overlap_with_chm"] > 0.99
        assert c["normalisation"] == pytest.approx(expect, rel=3e-3)
        assert c["r_free"] == pytest.approx(rfree, rel=5e-3)
        assert c["r_free"] < 1e-3
        assert 0.70 < c["normalisation"] < 1.05
        # two modes is BELOW the variation family's numerical rank, so the fit
        # is not a parameter count artefact
        assert c["rank_profile"]["1e-06"] >= 2
    # live, small: the recovered coefficients at N = 64
    res = free_weight_residual_2d(64, radii=(4,), ratios=(3, 4, 6, 8), parity="odd",
                                  n_modes=(2,), improve=XI_IMPROVE_3D)
    c = res[4.0]["per_n_modes"][2]
    assert c["overlap_with_chm"] > 0.98
    assert 0.6 < c["normalisation"] < 1.2


# ---------------------------------------------------------------------------
# the decisive test of the 1+1 even-sector question (Layer 7 unit 2, item (a))
# ---------------------------------------------------------------------------


def test_the_2p1_even_sector_is_consistent_with_the_odd_one():
    """In 2+1 the massless scalar has no IR zero mode.  If the 1+1 even-sector
    obstruction is that zero mode, the 2+1 even sector must behave like the odd
    one -- and it does, to within a factor 1.8-24, against ~2500-4500 in 1+1."""
    J = _load("J_2p1_even_odd.json")
    per_R = J["per_R"]
    assert per_R["4"]["even_over_odd"] == pytest.approx(1.7618, rel=3e-3)
    assert per_R["8"]["even_over_odd"] == pytest.approx(2.9783, rel=3e-3)
    assert per_R["10"]["even_over_odd"] == pytest.approx(24.230, rel=3e-3)
    ratios = [v["even_over_odd"] for v in per_R.values()]
    assert max(ratios) < 30.0
    assert float(np.median(ratios)) < 10.0
    # both sectors are small in absolute terms
    assert max(v["r_even"] for v in per_R.values()) < 0.20
    assert max(v["r_odd"] for v in per_R.values()) < 0.08
    # the disk's top Williamson eigenvalue does NOT run away with the box:
    # 0.666-0.747 over R = 3..10, against 0.934 -> 1.199 for a 1+1 ball of
    # fixed L = 8 as the chain grows from N = 200 to 3200 (data/I_even_sector)
    assert J["nu_top_max"] < 0.80
    # the 1+1 contrast, from the companion archive
    I = json.load(open(DATA / "I_even_sector.json"))
    lad = I["I2_projection_ladder"]
    for L in (8, 16, 32):
        r_even = lad[f"L{L}_n0"]["r"]
        r_odd = lad[f"L{L}_odd"]["r"]
        assert r_even / r_odd > 2000.0
    assert max(ratios) < 1e-2 * min(lad[f"L{L}_n0"]["r"] / lad[f"L{L}_odd"]["r"]
                                    for L in (8, 16, 32))


def test_the_2p1_residual_is_converged_in_the_box_size():
    J = _load("J_2p1_controls.json")
    conv = {int(k): v["residual"] for k, v in J["N_convergence"].items()}
    assert conv[48] == pytest.approx(0.0134553, rel=3e-3)
    assert conv[128] == pytest.approx(0.0101938, rel=3e-3)
    assert conv[48] > conv[64] > conv[96] > conv[128]
    assert abs(conv[128] - conv[96]) / conv[128] < 0.05


def test_the_archives_carry_build_information():
    for name in ("J_2p1_summary.json", "J_2p1_residual.json", "J_2p1_improvement.json",
                 "J_2p1_even_odd.json", "J_2p1_free_weight.json", "J_2p1_controls.json"):
        assert "__build__" in _load(name)
    b = _load("J_2p1_build_info.json")
    for key in ("produced_utc", "git_head", "generator", "generator_sha256", "mode"):
        assert key in b
    assert b["generator"] == "papers/toy-jacobson/notebook_2p1.py"
    gen = Path(__file__).resolve().parents[1] / "papers" / "toy-jacobson" / "notebook_2p1.py"
    import hashlib
    assert b["generator_sha256"] == hashlib.sha256(gen.read_bytes()).hexdigest(), (
        "data/J_2p1_* was produced by a different notebook_2p1.py than the one in the tree")


def test_the_improvement_operator_shape_is_identified_not_just_its_coefficient():
    """A one-parameter fit is exact cell by cell whatever operator you use; what
    identifies the improvement is that only the grad^2(beta) shape gives a
    coefficient that is the SAME across R = 4..10 and three wavelengths."""
    J = _load("J_2p1_summary.json")
    sh = J["J2_shape_identifiability"]
    right = sh["weight_gradsq_beta"]
    assert right["mean"] == pytest.approx(0.1278140, rel=1e-4)
    assert right["relative_scatter"] == pytest.approx(0.031320, rel=5e-3)
    assert right["relative_scatter"] < 0.05
    for name, factor in (("beta_weighted", 7.0), ("rim_only", 5.0), ("field_laplacian", 2.0)):
        alt = sh[name]
        assert alt["relative_scatter"] > factor * right["relative_scatter"], (
            f"{name}: scatter {alt['relative_scatter']:.4f} vs "
            f"{right['relative_scatter']:.4f} for the grad^2(beta) shape")
    assert sh["beta_weighted"]["relative_scatter"] == pytest.approx(0.328636, rel=5e-3)
    assert sh["rim_only"]["relative_scatter"] == pytest.approx(0.228804, rel=5e-3)
    # the field-side discretisation would need a NEGATIVE xi, excluded for a
    # free scalar, on top of its 2.5x larger scatter
    assert sh["field_laplacian"]["mean"] < 0.0
    assert sh["field_laplacian"]["mean"] == pytest.approx(-0.140494, rel=5e-3)


# ---------------------------------------------------------------------------
# the quadrature split (the MECHANISM sentence of section J) and the pp weight
# ---------------------------------------------------------------------------


def test_the_quadrature_split_of_the_pairing_is_archived_and_exact():
    """The xx/pp split of the 2+1 pairing, archived (it was a live diagnostic in
    the first pass and the MANIFEST quoted it without a data file or a test).

    Both G_B and G_loc are block diagonal for this vacuum and so is a single-mode
    squeeze, so dS = dS_xx + dS_pp and d<K> = dK_xx + dK_pp exactly; this test
    checks the identity live and pins the archived ratios INCLUDING R = 3, which
    the MANIFEST's original range silently dropped."""
    J = _load("J_2p1_sector_split.json")
    odd = J["cells"]["wave_odd_xi0_site"]
    assert J["N"] == 128 and J["interior_margin"] == 1.0
    # every radius, R = 3 included
    assert odd["pp_ratio_range"][0] == pytest.approx(0.6997479, rel=1e-4)
    assert odd["pp_ratio_range"][1] == pytest.approx(1.0185997, rel=1e-4)
    assert odd["xx_ratio_range"][0] == pytest.approx(0.5605118, rel=1e-4)
    assert odd["xx_ratio_range"][1] == pytest.approx(0.6801329, rel=1e-4)
    # the R >= 4 window the MANIFEST now quotes
    big = J["odd_R_ge_4"]
    assert big["pp_ratio_range"][0] == pytest.approx(0.9067056, rel=1e-4)
    assert big["pp_ratio_range"][1] == pytest.approx(1.0185997, rel=1e-4)
    assert big["xx_ratio_range"][0] == pytest.approx(0.6356274, rel=1e-4)
    assert big["xx_ratio_range"][1] == pytest.approx(0.6801329, rel=1e-4)
    # the pp sector pairs to within 10 % for R >= 4 while the xx sector is 35 % short
    assert min(abs(x - 1.0) for x in big["pp_ratio_range"]) < 0.02
    assert max(big["xx_ratio_range"]) < 0.70
    # ... but pp carries at most 13 % of dS, so the split must be read with this weight
    assert big["dS_pp_over_dS_range"][0] == pytest.approx(-0.1332785, rel=1e-4)
    assert big["dS_pp_over_dS_range"][1] == pytest.approx(-0.0058102, rel=1e-4)
    for R in ("3", "4", "5", "6", "8", "10"):
        row = odd["per_R"][R]
        assert row["pp_ratio_min"] <= row["pp_ratio_max"]
    assert odd["per_R"]["3"]["pp_ratio_max"] == pytest.approx(0.7039690, rel=1e-4)
    assert odd["per_R"]["6"]["pp_ratio_max"] == pytest.approx(1.0185997, rel=1e-4)
    # live: the split is exact and reproduces the same residual as equilibrium_residual_2d
    from vacuum.geometry.jacobson2d import sector_split_2d

    det = sector_split_2d(48, radii=(4,), ratios=(4, 8), parity="odd", z=1)
    ref = equilibrium_residual_2d(48, radii=(4,), ratios=(4, 8), parity="odd", z=1,
                                  beta_at="site", improve=0.0, return_details=True)
    for row, rr in zip(det["rows"], ref["rows"]):
        assert row["dS_xx"] + row["dS_pp"] == pytest.approx(row["dS"], rel=1e-12)
        assert row["dK_xx"] + row["dK_pp"] == pytest.approx(row["dK"], rel=1e-12)
        assert row["dS"] == pytest.approx(rr["dS"], rel=1e-9)
        assert row["dK"] == pytest.approx(rr["dK"], rel=1e-9)
        assert 0.85 < row["pp_ratio"] < 0.95 and 0.60 < row["xx_ratio"] < 0.70


def test_the_pp_weight_of_the_exact_modular_hamiltonian_drifts_with_R():
    """RETRACTION PIN (Layer 7 unit 2 fix round 1).

    The MANIFEST claimed the exact modular Hamiltonian's "pp diagonal equals
    2 pi beta_i to 3 %".  It named no statistic and no radius, and it is false as
    a statement about the disk's pp DIAGONAL: the least-squares slope of
    diag(G_B^pp) on 2 pi beta drifts from 1.106 at R = 3 to 0.743 at R = 10,
    because G_B^pp is not diagonal and its off-diagonal mass grows from 0.15 to
    0.74 of the diagonal's.  The k -> 0 pp weight -- the ROW SUM, the combination
    jacobson.eh_profiles uses in 1+1 -- is the statistic that stays near 1
    (slope 1.021-1.138).  This test pins both, so the retraction cannot silently
    come back."""
    J = _load("J_2p1_sector_split.json")
    diag = J["pp_weight_diag_slope"]
    rows = J["pp_weight_rowsum_slope"]
    off = J["pp_offdiag_over_diag"]
    expect_diag = {"3": 1.1059466, "4": 1.0487569, "5": 1.0024680,
                   "6": 0.9443271, "8": 0.8640931, "10": 0.7426881}
    expect_rows = {"3": 1.1382836, "4": 1.0591960, "5": 1.0459088,
                   "6": 1.0214335, "8": 1.0809414, "10": 1.0625613}
    expect_off = {"3": 0.1488729, "4": 0.1694135, "5": 0.2488824,
                  "6": 0.3481211, "8": 0.5247341, "10": 0.7394685}
    for R in expect_diag:
        assert diag[R] == pytest.approx(expect_diag[R], rel=1e-4)
        assert rows[R] == pytest.approx(expect_rows[R], rel=1e-4)
        assert off[R] == pytest.approx(expect_off[R], rel=1e-4)
    # the retracted claim: no "to 3 %" statement survives across R for the diagonal
    assert max(abs(diag[R] - 1.0) for R in diag) > 0.25
    assert abs(diag["10"] - 1.0) > 0.20
    # the diagonal statistic drifts DOWNWARD monotonically in R
    seq = [diag[R] for R in ("3", "4", "5", "6", "8", "10")]
    assert all(b < a for a, b in zip(seq, seq[1:]))
    # the off-diagonal pp mass, which is why, grows monotonically
    seq_off = [off[R] for R in ("3", "4", "5", "6", "8", "10")]
    assert all(b > a for a, b in zip(seq_off, seq_off[1:]))
    # the row-sum statistic is the one that is flat, and it is 2-14 % high, not 3 %
    assert max(abs(rows[R] - 1.0) for R in rows) < 0.14
    assert max(abs(rows[R] - 1.0) for R in rows if R != "3") < 0.09
    assert min(abs(rows[R] - 1.0) for R in rows) > 0.02
    # the summary carries the same block
    S = _load("J_2p1_summary.json")
    assert S["J6_sector_split"]["pp_weight_diag_slope"]["10"] == pytest.approx(
        expect_diag["10"], rel=1e-4)
