"""The continuum limit behind papers/toy-jacobson anchor 1, certified.

``draft/proof_continuum_limit.md`` states the continuum theorem the lattice
residual r(K) approximates; two of its steps are computational and are pinned
here.  Everything in this file is mpmath or closed form -- the only lattice
call is the one confrontation in the last test.

A. The continuum first law with the exact CHM modular Hamiltonian.  For the
   interval (-R, R) in the 1+1 massless/chiral vacuum the one-particle
   modular generator is -2 pi i (beta d/dx + beta'/2) with the CHM parabola
   beta = (R^2 - x^2)/(2R); the vacuum is thermal at modular temperature
   1/(2 pi), so the symplectic eigenvalue at modular frequency s is
   nu(s) = coth(pi s)/2 and the bosonic entropy weight is
   g(nu(s)) = ln((nu + 1/2)/(nu - 1/2)) = 2 pi s.  For an explicit Gaussian
   squeeze wave packet the entropy route and the position-space CHM route
   agree to 6e-27 (dps = 25) / 1e-30 (dps = 40) -- far inside the 1e-12 the
   proof needs.  MUTATION: a weight scaled by 1.001 gives a defect of exactly
   1.0e-03, and beta(1 + 0.05 x^2) gives 1.4e-02.

B. The conformal-Killing (shape) test.  A diffeomorphism supported strictly
   inside the interval is a unitary of the interval's algebra, so dS = 0 and
   the first law forces 2 pi int w dT_00 = -(1/12) int w xi''' = 0 for every
   such xi, i.e. w''' = 0.  Measured: 5e-37 for the CHM parabola, 2.6e-04 for
   beta + 0.1 x^3, 2.6e-04 for beta + 0.1 x^4, 6.0e-04 for beta cosh x.  This
   test is blind to the normalisation of w by construction (it is linear in
   w); test A is what fixes the normalisation.

C. The O(a^2) weight-placement error of the lattice bond split.  The site-
   centred (trapezoid) and bond-centred (midpoint) discretisations of
   int beta T_00 differ by exactly a^2 beta''/8 = -1/(8R) per interior bond
   plus beta_edge/2 on each straddling bond, which predicts the DIFFERENCE
   rho_site - rho_bond = A(kappa)/L^2 with kappa = pi L/lambda and
   A(0+) = -3/2.  With the measured rho_bond the relation is an EXACT lattice
   identity (draft Prop. 4.3), confirmed to 6.3e-12; dropping the
   (1 - rho_bond) factor costs 8.3e-04 (the '0.1 %' figure,
   max_rel_err_discrete); and taking the L -> infinity limit -- the step
   to A(kappa), ASSUMPTION A4 -- costs 3.6e-02 in the worst archived cell
   (L = 8, kappa = 0.785) and 8.4e-04 in the best (L = 32, kappa = 0.209),
   i.e. 43x the previous figure.  These three are separate archive keys and
   must never be quoted for one another.
   WHAT IT DOES NOT DO: A(kappa) is the coefficient of the DIFFERENCE, not of
   the site-centred floor.  The bond-centred floor is NOT derived (it is
   predicted to vanish at this order), so the site coefficient itself is
   accounted for only up to the bond floor's share of it -- measured at
   3.8 % (L = 8) to 13.6 % (L = 32) at the dominant wavelength and 116 % in
   the worst cell (L = 32, lambda = 4L, N = 2400), pinned below as
   rel_err_site_only.
"""

from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

from vacuum.geometry import (
    bond_split_discretization_prediction,
    bond_split_weight_coefficient,
    chm_beta,
    conformal_killing_defect,
    continuum_first_law_defect,
    entropy_weight,
    modular_coordinate,
    modular_coordinate_inverse,
    modular_symplectic_eigenvalue,
)
from vacuum.geometry.jacobson import coupling_chain_K, equilibrium_residual

DATA = Path(__file__).resolve().parents[1] / "papers" / "toy-jacobson" / "data"
RATIOS = (4, 8, 16)


def _beta(R):
    return lambda x: chm_beta(x, R)


# --- A: the continuum first law -------------------------------------------


def test_modular_coordinate_is_the_conformal_killing_flow():
    """dz/dx = 1/beta and x(z) = R tanh(z/2) invert each other."""
    with mp.workdps(30):
        R = mp.mpf(1)
        for x in ("-0.9", "-0.3", "0", "0.55", "0.97"):
            x = mp.mpf(x)
            z = modular_coordinate(x, R)
            assert abs(modular_coordinate_inverse(z, R) - x) < mp.mpf(10) ** -25
            dz = mp.diff(lambda y: modular_coordinate(y, R), x)
            assert abs(dz - 1 / chm_beta(x, R)) < mp.mpf(10) ** -25, f"dz/dx != 1/beta at {x}"


@pytest.mark.parametrize("s", ["0.25", "1", "4", "12"])
def test_entropy_weight_of_the_modular_spectrum_is_two_pi_s(s):
    """g(nu(s)) = 2 pi s: the thermality of the vacuum at modular temperature
    1/(2 pi) is what turns the first law into the modular-energy identity."""
    with mp.workdps(40 + int(3 * float(s))):
        sv = mp.mpf(s)
        err = abs(entropy_weight(modular_symplectic_eigenvalue(sv)) - 2 * mp.pi * sv)
        assert err < mp.mpf(10) ** -30, f"g(nu({s})) - 2 pi s = {mp.nstr(err, 5)}"


@pytest.mark.parametrize("z0,sigma,s0", [(0.0, 2.0, 4.0), (1.5, 2.0, 4.0), (0.0, 1.0, 8.0)])
def test_continuum_first_law_holds_with_the_chm_weight(z0, sigma, s0):
    """dS - d<K> = 0 to 1e-12 (measured <= 6e-27 at dps = 25), for the exact
    continuum modular Hamiltonian and an explicit Gaussian squeeze."""
    R = mp.mpf(1)
    d = continuum_first_law_defect(R=1.0, s0=s0, sigma=sigma, z0=z0, weight=_beta(R),
                                   weight_prime=lambda x: -x, dps=25)
    assert abs(d["norm_s"] - 1) < mp.mpf(10) ** -20, "the modular content is not normalized"
    assert abs(d["norm_x"] - 1) < mp.mpf(10) ** -20, "the position packet is not normalized"
    assert d["im_dK"] < mp.mpf(10) ** -20, "d<K> must be real (the generator is self-adjoint)"
    with mp.workdps(30):  # the reference needs the same precision as the check
        ref = 2 * mp.pi * mp.mpf(s0)
    assert abs(d["dS"] - ref) < mp.mpf(10) ** -20, (
        f"dS = {mp.nstr(d['dS'], 20)}, expected 2 pi s0 = {mp.nstr(ref, 20)}")
    assert d["rel_defect"] < mp.mpf(10) ** -12, (
        f"first-law defect {mp.nstr(d['rel_defect'], 5)} (measured 6e-27)")


@pytest.mark.parametrize("scale,expect", [("1.001", 1.0e-3), ("0.995", 5.0e-3)])
def test_mutation_a_rescaled_weight_breaks_the_first_law(scale, expect):
    """MUTATION: beta -> (1 + eps) beta moves the CHM route only; the defect
    is exactly |eps| (the identity is linear in the weight)."""
    R = mp.mpf(1)
    f = mp.mpf(scale)
    d = continuum_first_law_defect(R=1.0, weight=lambda x: f * chm_beta(x, R),
                                   weight_prime=lambda x: -f * x, dps=25)
    got = float(d["rel_defect"])
    assert got > 1e-4, f"a weight scaled by {scale} must be caught, defect {got:.3e}"
    assert abs(got - expect) < 1e-9 * max(1.0, expect / 1e-3), (
        f"defect {got:.6e}, expected |1 - {scale}| = {expect:.6e}")


def test_mutation_a_wrong_shape_breaks_the_first_law():
    """MUTATION: beta(1 + 0.05 x^2) is still nonnegative and still vanishes at
    the endpoints, and is still caught at the percent level."""
    R = mp.mpf(1)
    w = lambda x: chm_beta(x, R) * (1 + mp.mpf("0.05") * x ** 2)  # noqa: E731
    d = continuum_first_law_defect(R=1.0, weight=w, weight_prime=lambda x: mp.diff(w, x), dps=25)
    got = float(d["rel_defect"])
    assert 1e-3 < got < 1e-1, f"shape mutation defect {got:.3e} (measured 1.37e-2)"


# --- B: the conformal-Killing shape test ----------------------------------


def test_conformal_killing_shape_test_and_its_mutations():
    """int_A w xi''' = 0 for every xi supported inside A iff w''' = 0."""
    R = mp.mpf(1)
    chm = float(conformal_killing_defect(R=1.0, weight=_beta(R), dps=30)["rel_defect"])
    assert chm < 1e-25, f"the CHM parabola must pass exactly, got {chm:.3e} (measured 5e-37)"
    scaled = float(conformal_killing_defect(
        R=1.0, weight=lambda x: mp.mpf("1.001") * chm_beta(x, R), dps=30)["rel_defect"])
    assert scaled < 1e-25, (
        f"this test is linear in w and MUST be blind to normalisation, got {scaled:.3e}")
    for label, w in (("cubic", lambda x: chm_beta(x, R) + mp.mpf("0.1") * x ** 3),
                     ("quartic", lambda x: chm_beta(x, R) + mp.mpf("0.1") * x ** 4),
                     ("cosh", lambda x: chm_beta(x, R) * mp.cosh(x))):
        got = float(conformal_killing_defect(R=1.0, weight=w, dps=30)["rel_defect"])
        assert got > 1e-5, f"{label}: w''' != 0 must be caught, got {got:.3e}"


# --- C: the O(a^2) weight-placement prediction ----------------------------


def test_bond_split_coefficient_long_wavelength_limit_is_minus_three_halves():
    """A(kappa -> 0) = -3/2, where A is the coefficient of the DIFFERENCE
    rho_site - rho_bond (not of the site-centred floor, which also contains
    the underived bond floor)."""
    # D(kappa) cancels terms of size 1/kappa^3, so below kappa ~ 1e-4 the
    # float64 evaluation is noise-limited; 1e-2..1e-4 is the usable window.
    for kappa, tol in ((1e-2, 2e-4), (1e-3, 3e-6), (1e-4, 1e-7)):
        A = float(bond_split_weight_coefficient(kappa))
        assert abs(A + 1.5) < tol, f"A({kappa}) = {A:.10f}, expected -1.5"
    # and it is strongly wavelength dependent at the wavelengths measured
    A4, A8, A16 = (float(bond_split_weight_coefficient(np.pi / m)) for m in (4, 8, 16))
    assert abs(A4 + 0.3521) < 2e-3 and abs(A8 + 1.2083) < 2e-3 and abs(A16 + 1.4268) < 2e-3, (
        f"A(pi/4, pi/8, pi/16) = {A4:.4f}, {A8:.4f}, {A16:.4f}")


# measured at N = 800, L = 8/16, ratios 4/8/16, reflection-odd single-mode
# squeezes: (L, ratio, wavelength, A_site, A_bond).  These are the VALUES the
# identity is checked against, pinned so that a change of N, of the mode
# index, or a regression in equilibrium_residual cannot hide inside a
# self-consistent relative agreement.
_LIVE_ROWS = (
    (8, 4.0, 32.04, -0.31533095, +0.052767837),
    (8, 8.0, 61.61538461538461, -1.1492272, +0.056218334),
    (8, 16.0, 114.42857142857143, -1.3731674, +0.057036334),
    (16, 4.0, 61.61538461538461, -0.14907557, +0.11785875),
    (16, 8.0, 114.42857142857143, -1.021937, +0.11812309),
    (16, 16.0, 267.0, -1.3195969, +0.1182048),
)


def test_bond_split_prediction_reproduces_the_lattice_residual():
    """The derived A(kappa) against the measured rho_site - rho_bond at
    N = 800, L = 8, 16: the discrete form to 0.1 %, the closed form to 4 %.
    The measured coefficients themselves are pinned to 1e-4 relative."""
    N = 800
    K = coupling_chain_K(N, [1.0])
    eig = np.linalg.eigh(K)
    det = {ba: equilibrium_residual(K, sizes=(8, 16), ratios=RATIOS, parity="odd", eig=eig,
                                    return_details=True, escalate_above=1e-3, beta_at=ba)
           for ba in ("site", "bond")}
    worst_e, worst_d, worst_c = 0.0, 0.0, 0.0
    rows = list(zip(det["site"]["rows"], det["bond"]["rows"]))
    assert len(rows) == len(_LIVE_ROWS)
    for (rs, rb), want in zip(rows, _LIVE_ROWS):
        L, lam = int(rs["L"]), float(rs["wavelength"])
        wL, wratio, wlam, wAs, wAb = want
        assert (L, float(rs["ratio"])) == (wL, wratio)
        assert abs(lam / wlam - 1.0) < 1e-9, f"L={L}: wavelength {lam} != {wlam} (wrong N?)"
        A_site, A_bond = float(rs["rho"]) * L ** 2, float(rb["rho"]) * L ** 2
        assert abs(A_site / wAs - 1.0) < 1e-4, f"L={L} lam={lam}: A_site {A_site} != {wAs}"
        assert abs(A_bond / wAb - 1.0) < 1e-4, f"L={L} lam={lam}: A_bond {A_bond} != {wAb}"
        meas = A_site - A_bond
        p = bond_split_discretization_prediction(L, lam, rho_bond=rb["rho"])
        assert meas < 0, f"L={L}: the site ansatz must overweight the bonds, rho diff {meas}"
        worst_e = max(worst_e, abs(p["A_exact"] / meas - 1.0))
        worst_d = max(worst_d, abs(p["A_discrete"] / meas - 1.0))
        worst_c = max(worst_c, abs(p["A_continuum"] / meas - 1.0))
    assert worst_e < 1e-8, (
        f"Prop. 4.3 is an EXACT identity; it is off by {worst_e:.3e} (measured < 1e-9)")
    assert worst_d < 5e-3, f"discrete prediction off by {worst_d:.2%} (measured 0.1 %)"
    assert worst_c < 0.06, f"closed-form prediction off by {worst_c:.2%} (measured 3.7 %)"


def test_the_site_floor_is_only_derived_up_to_the_underived_bond_floor():
    """HONESTY PIN.  What Prop. 4.3 predicts is rho_site - rho_bond.  The bond
    floor is not derived, so the SITE coefficient is accounted for only to
    |A_bond/A_site| -- which at N = 800 is 4 % in the best cell and 79 % in
    the worst.  Any text that says "the site floor is derived to 0.1 %" is
    contradicted by this test."""
    # _LIVE_ROWS is itself checked against a live run by the test above
    shares = [abs(Ab / As) for (_, _, _, As, Ab) in _LIVE_ROWS]
    assert abs(min(shares) - 0.0415) < 5e-3, f"best-cell share {min(shares):.4f}"
    assert abs(max(shares) - 0.7906) < 5e-2, f"worst-cell share {max(shares):.4f}"
    assert max(shares) > 0.05, (
        "the bond floor is NOT a negligible correction to the site floor")


# --- the archived data file -----------------------------------------------


@pytest.mark.skipif(not (DATA / "H_continuum_check.json").exists(),
                    reason="run papers/toy-jacobson/notebook_continuum.py")
def test_archived_continuum_check_matches_the_claims():
    """data/H_continuum_check.json carries the numbers quoted in the MANIFEST."""
    d = json.loads((DATA / "H_continuum_check.json").read_text())
    h1 = d["H1_continuum_first_law"]
    for name, run in h1["runs"].items():
        assert run["rel_defect"] < 1e-12, f"{name}: archived defect {run['rel_defect']:.2e}"
        assert abs(run["norm_x"] - 1) < 1e-20 and abs(run["norm_s"] - 1) < 1e-20
    assert abs(h1["mutations"]["scale_1.001"] - 1e-3) < 1e-9
    assert h1["mutations"]["shape_beta_times_1_plus_0.05x2"] > 1e-3
    h2 = d["H2_conformal_killing_shape"]
    assert h2["chm"] < 1e-25 and h2["scaled_1.001"] < 1e-25
    assert h2["cubic_plus_0.1x3"] > 1e-5 and h2["quartic_plus_0.1x4"] > 1e-5
    h3 = d["H3_bond_split_prediction"]
    assert abs(h3["A_long_wavelength_limit"] + 1.5) < 1e-4
    assert h3["max_rel_err_exact"] < 1e-8, (
        f"archived exact identity off by {h3['max_rel_err_exact']:.3e}")
    assert h3["max_rel_err_discrete"] < 5e-3, (
        f"archived discrete prediction off by {h3['max_rel_err_discrete']:.2%}")
    # measured 3.608e-02; a tolerance a wrong number would not pass
    assert h3["max_rel_err_continuum"] < 0.05, (
        f"archived closed form off by {h3['max_rel_err_continuum']:.2%} (measured 3.6 %)")
    assert (h3["N"], tuple(h3["sizes"]), tuple(h3["ratios"])) == (2400, (8, 16, 32), (4, 8, 16))
    assert "generator_sha256" in d["__build__"] and "produced_utc" in d["__build__"]


# the nine archived cells the MANIFEST and the draft quote, N = 2400:
# (L, ratio, wavelength, kappa, A_site, A_bond, A_measured, A_continuum)
_ARCHIVED_ROWS = (
    (8, 4.0, 32.013333333333335, 0.7850710505, -0.31756432, +0.048679254, -0.36624358, -0.3530285),
    (8, 8.0, 63.18421052631579, 0.3977693322, -1.1685635, +0.052425674, -1.2209892, -1.2007927),
    (8, 16.0, 126.36842105263158, 0.1988846661, -1.3935795, +0.05329785, -1.4468773, -1.424931),
    (16, 4.0, 63.18421052631579, 0.7955386645, -0.27314612, +0.053058188, -0.32620431, -0.32301445),
    (16, 8.0, 126.36842105263158, 0.3977693322, -1.1486609, +0.057099586, -1.2057605, -1.2007927),
    (16, 16.0, 240.1, 0.2093522801, -1.3642412, +0.057970585, -1.4222118, -1.4168315),
    (32, 4.0, 126.36842105263158, 0.7955386645, -0.1500658, +0.1737055, -0.3237713, -0.32301445),
    (32, 8.0, 240.1, 0.4187045602, -0.99881269, +0.17092465, -1.1697373, -1.1686445),
    (32, 16.0, 480.2, 0.2093522801, -1.2478051, +0.17021202, -1.4180171, -1.4168315),
)


@pytest.mark.skipif(not (DATA / "H_continuum_check.json").exists(),
                    reason="run papers/toy-jacobson/notebook_continuum.py")
def test_archived_bond_split_rows_are_pinned():
    """Every archived cell quoted in MANIFEST.md and in the draft's table is
    pinned to 1e-4 relative on the measured coefficients and 1e-6 on the mode
    (a change of N, of the mode index, or a regression in equilibrium_residual
    moves these by percents, not by 1e-4)."""
    rows = json.loads((DATA / "H_continuum_check.json").read_text())[
        "H3_bond_split_prediction"]["rows"]
    assert len(rows) == len(_ARCHIVED_ROWS)
    for r, want in zip(rows, _ARCHIVED_ROWS):
        L, ratio, lam, kap, As, Ab, Ad, Ac = want
        assert (r["L"], r["ratio"]) == (L, ratio)
        for key, w, tol in (("wavelength", lam, 1e-6), ("kappa", kap, 1e-6),
                            ("A_site_measured", As, 1e-4), ("A_bond_measured", Ab, 1e-4),
                            ("A_measured", Ad, 1e-4), ("A_continuum", Ac, 1e-4),
                            ("A_exact", Ad, 1e-4)):
            assert abs(r[key] / w - 1.0) < tol, (
                f"L={L} ratio={ratio}: {key} = {r[key]!r}, pinned {w!r} (tol {tol:.0e})")


@pytest.mark.skipif(not (DATA / "H_continuum_check.json").exists(),
                    reason="run papers/toy-jacobson/notebook_continuum.py")
def test_archived_site_floor_is_not_derived_to_one_part_in_a_thousand():
    """HONESTY PIN on the archive.  0.1 % (max_rel_err_discrete) is how well
    the DIFFERENCE rho_site - rho_bond is predicted.  The site coefficient
    itself is accounted for only to rel_err_site_only = |A_bond/A_site|,
    which is 3.8 % in the best cell and 116 % in the worst (L = 32,
    lambda = 4L, where the N-limited bond floor exceeds the site floor)."""
    h3 = json.loads((DATA / "H_continuum_check.json").read_text())["H3_bond_split_prediction"]
    assert h3["max_rel_err_discrete"] < 1.1e-3, "the DIFFERENCE is the 0.1 % claim"
    assert abs(h3["min_rel_err_site_only"] - 0.03825) < 1e-3, (
        f"best cell {h3['min_rel_err_site_only']:.5f}")
    assert abs(h3["max_rel_err_site_only"] - 1.1575) < 1e-2, (
        f"worst cell {h3['max_rel_err_site_only']:.5f}")
    assert h3["min_rel_err_site_only"] > 30 * h3["max_rel_err_discrete"], (
        "the site floor is NOT derived to the accuracy of the difference")
    # the dominant-wavelength cells (ratio 16), which anchor 5 and anchor 1 quote
    dom = {r["L"]: r["rel_err_site_only"] for r in h3["rows"] if r["ratio"] == 16.0}
    assert abs(dom[8] - 0.03825) < 1e-3 and abs(dom[32] - 0.13641) < 2e-3, dom


@pytest.mark.skipif(not (DATA / "H_continuum_check.json").exists(),
                    reason="run papers/toy-jacobson/notebook_continuum.py")
def test_the_closed_form_remainder_is_not_the_discrete_one():
    """HONESTY PIN, round 2.  Three DIFFERENT approximations of the same
    difference rho_site - rho_bond are archived, and they must never be
    quoted for one another:

      exact identity (4.5) with the measured rho_bond   6.3e-12
      discrete form, (1 - rho_bond) factor dropped      8.3e-04  ("0.1 %")
      closed form A(kappa)/L^2, (4.6) = ASSUMPTION A4   3.6e-02  (worst cell)

    The remainder of Cor. 4.4 is the LAST of these, 43x the middle one.  This
    test pins both ends of the A4 window and the ratio, so no text can call
    A4's remainder 0.1 % again.
    """
    h3 = json.loads((DATA / "H_continuum_check.json").read_text())["H3_bond_split_prediction"]
    # measured 3.608275e-02 (L = 8, kappa = 0.785) and 8.36159e-04 (L = 32, kappa = 0.209)
    assert abs(h3["max_rel_err_continuum"] - 3.608275e-02) < 1e-5, h3["max_rel_err_continuum"]
    assert abs(h3["min_rel_err_continuum"] - 8.36159e-04) < 1e-6, h3["min_rel_err_continuum"]
    assert h3["max_rel_err_continuum"] > 30 * h3["max_rel_err_discrete"], (
        "A4's remainder and the discrete form's remainder are the same size -- "
        "one of the two is wrong")
    # per-cell, and consistent with the archived ratios
    per = {(r["L"], r["ratio"]): r for r in h3["rows"]}
    for key, want in (((8, 4.0), 3.608275e-02), ((8, 16.0), 1.516804e-02),
                      ((16, 4.0), 9.778712e-03), ((32, 4.0), 2.337607e-03),
                      ((32, 16.0), 8.36159e-04)):
        r = per[key]
        assert abs(r["rel_err_continuum"] / want - 1.0) < 1e-4, (key, r["rel_err_continuum"])
        assert abs(r["rel_err_continuum"] - abs(r["ratio_continuum_over_measured"] - 1.0)) < 1e-12
        assert abs(r["rel_err_discrete"] - abs(r["ratio_discrete_over_measured"] - 1.0)) < 1e-12
    # at fixed kappa the closed form improves by ~4x per doubling of L (observed,
    # not bounded -- ASSUMPTION A4); it must at least be monotone in L
    assert (per[(8, 4.0)]["rel_err_continuum"] > per[(16, 4.0)]["rel_err_continuum"]
            > per[(32, 4.0)]["rel_err_continuum"])


_DRAFT = Path(__file__).resolve().parents[1] / "papers/toy-jacobson/draft/proof_continuum_limit.md"


@pytest.mark.skipif(not (_DRAFT.exists() and (DATA / "H_continuum_check.json").exists()),
                    reason="draft or archive missing")
def test_the_draft_status_table_quotes_the_measured_remainders():
    """The status table is the one place the draft tells readers to quote from,
    and in round 1 it kept a 0.1 % figure that belongs to a different
    approximation.  This test reads the two rows and requires the percentages
    in them to be the ones in the data file."""
    h3 = json.loads((DATA / "H_continuum_check.json").read_text())["H3_bond_split_prediction"]
    lines = _DRAFT.read_text().splitlines()
    cor = [ln for ln in lines if ln.startswith("| §4, Cor. 4.4 |")]
    sec = [ln for ln in lines if ln.startswith("| §4.4 |")]
    assert len(cor) == 1 and len(sec) == 1, (len(cor), len(sec))
    row = cor[0]
    assert f"{h3['max_rel_err_continuum'] * 100:.1f} %" in row, row
    assert f"{h3['min_rel_err_continuum'] * 100:.3f} %" in row, row
    assert "ASSUMPTION A4" in row and "not bounded" in row
    for banned in ("measured at 0.1 %", "measured to 0.1 %", "measured at 0.1%"):
        assert banned not in row, f"Cor. 4.4's remainder is not 0.1 %: {row}"
    assert "predicts to 0.1 %" not in sec[0], sec[0]
    assert "exactly" in sec[0] and "6.3e-12" in sec[0], sec[0]
