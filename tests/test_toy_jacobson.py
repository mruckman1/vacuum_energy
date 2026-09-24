"""Toy Jacobson (PLAN.md Layer 4 discovery target / leap L2): the four anchors.

The entanglement first law delta S = delta<K> is an identity for any
dynamics (Layers 1 and 4 verify it at 1e-10); Jacobson's entanglement-
equilibrium argument (PRL 116, 201101 (2016)) has content only because the
ball's modular Hamiltonian is LOCAL -- the Casini-Huerta-Myers weight
2 pi (R^2 - r^2)/(2R) times the energy density -- which holds for
Lorentz-invariant dynamics and fails for generic lattice dynamics. The
residual r(K) of vacuum.geometry.jacobson measures that failure with the
lattice's own energy density and long-wavelength (reflection-odd) squeeze
variations; the module docstring states the definitions.

Anchors (all numbers measured on this build and recorded in the assertion
messages; N = 800 Dirichlet chains unless stated):

1. Massless wave chain: r decreases with ball size, 9.1e-2 -> 1.8e-3 over
   L = 4..24, with lattice-correction exponent -2.2 (L >= 8) and both
   continuum extrapolations |r_inf| <= 3.6e-4 -- consistent with 0 at the
   5e-4 level set by the scatter of the extrapolations.
2. The exact first law delta S = (1/2) Tr[G_B delta V_B] holds at 1e-8
   (measured <= 5e-9) for EVERY dynamics of the family, by independent
   routes (finite differences of the entropy of the exactly squeezed ball
   state vs the Williamson modular Hamiltonian), so the residual measures
   locality, not numerics. The precision ladder is needed: the Lifshitz
   chain's badly scaled ball covariance gives 1.3e-7 in float64.
3. Non-relativistic dynamics stay away from 0 as the ball grows: Lifshitz
   omega ~ k^2 has r = 0.97..0.99 at L = 8..32 (45x the wave chain at
   L = 8, 190x at L = 16); omega ~ sqrt(k) gives r > 3.
4. constrain_dynamics over (c_1, c_2, m) from a generic start lands at
   v^2 = c_1 + 4 c_2 within 5 % of 1 and m < 0.02 with a residual below the
   nearest-neighbour chain's; the coupling range c_2 is NOT recovered --
   it is a flat valley within |c_2| < 0.35 (the family's resolution at
   L <= 16), reported honestly.
5. The ansatz's discretisation floor is a CHOICE, not a bound: putting beta at
   the bond midpoints instead of the site centres (the midpoint rule for the
   gradient energy, which also makes the bonds straddling dB drop out) lowers
   the massless chain's floor from 1.38/L^2 to 0.054/L^2, a factor >= 15.
6. The k^4 coefficient. Within the (c_1, c_2) family at v = 1 the k^4 dispersion
   coefficient A_4 = 1 + 12 c_2 is affine in c_2, so the dispersion and any
   range-dependent discretisation error are degenerate. Sweeping the third-
   neighbour coupling too breaks it: a pure-dispersion response requires
   d rho/d c_3 = 6 d rho/d c_2. The site-centred ansatz obeys it (6.2 +- 0.1)
   and gives C = -0.93 +- 0.05 at lam = 4L; the bond-centred one does NOT
   (2.3), so its smaller floor is bought with a range-dependent artefact.
7. A second ball position: a ball ATTACHED to the Dirichlet wall has one
   entangling point and the exact boundary-CFT weight (B^2 - y^2)/(2B); with it
   r matches the centred ball, while the bulk parabola used at that position
   gives O(1). A DETACHED ball near the wall stays O(1): its modular
   Hamiltonian has a bilocal term to the mirror image.
8. The zero mode, projected instead of avoided: projecting the ball's top
   Williamson mode out of the VARIATION (so dS and dK are the same functional
   of the same dV) drops the reflection-even residual from ~2 to ~0.05 -- 40x,
   but still ~100x the odd sector, and it does not decay with L.
9. The weight set free (the reviewer's version of the test; anchors 1-8
   assume the CHM weight, which is derived from conformal symmetry). With
   K_loc = sum_k w_k O_k over the lattice's local energy pieces and w >= 0
   from a smooth family, minimised over a long-wavelength variation family
   that includes two-mode squeezes, the wave chain's minimiser is the CHM
   parabola to 0.1 % with r_free 20-50x below the CHM residual, in every
   basis tried. The Lifshitz verdict is a PARSIMONY statement, not an
   existence statement (two adversarial audits, 2026-09-04): per ball, four
   nonnegative modes fit a Lifshitz ball; shared across L = 8, 16, 32, a
   nonnegative profile R^z f(x/R) fits it too once >= 11-12 modes are
   allowed (below 1e-2 by 11-13 modes and below 1e-3 by 12-14, z-dependent;
   5e-4 only by 20 Chebyshev modes; beyond ~13 modes the family exceeds the
   variation family's 8-9 conditions at 1e-3), while the wave chain
   needs 3 Chebyshev modes (4 hats) for 1.5e-4. The 3-mode (0.27-0.68) and
   8-mode (0.09) Lifshitz floors are family properties. The velocity is
   exactly absorbed by the free weight (a unit); the mass is absorbed per
   ball but not by a shared profile; the constrained fit with the shared
   profile pins m/v -> 0. A per-site weight has as many parameters as the
   family has conditions and fits anything (no teeth).

Each test file stays well under 90 s (this one ~75 s).
"""

import numpy as np
import pytest
from scipy.linalg import expm

from vacuum.core import (Omega, ground_state_cov, harmonic_chain_K, reduce,
                         symplectic_inverse, williamson)
import vacuum.geometry.jacobson as jac
from vacuum.geometry.jacobson import (
    bulk_dispersion,
    bw_weights,
    centred_ball,
    constrain_dynamics,
    coupling_chain_K,
    dispersion_departure,
    dispersion_power_K,
    eh_profiles,
    equilibrium_residual,
    first_law_audit,
    free_weight_residual,
    k4_sensitivity,
    local_energy_forms,
    local_modular_form,
    neighbour_laplacian,
    pair_variation,
    random_symmetric_chain_K,
    smooth_variations,
    squeeze_modes,
    squeeze_variation,
    squeezed_ball_covariance,
    universal_weight_residual,
    weight_family,
)

N = 800
RATIOS = (4, 8, 16)


def _long_range_K(N, alpha, n_max=40):
    c = np.array([1.0 / n ** alpha for n in range(1, n_max + 1)])
    return coupling_chain_K(N, c / np.sum(c * np.arange(1, n_max + 1) ** 2))


@pytest.fixture(scope="module")
def family():
    """The dynamics family with cached eigendecompositions."""
    Lap = coupling_chain_K(N, [1.0])
    fam = {
        "wave": Lap,
        "massive_0.2": coupling_chain_K(N, [1.0], m=0.2),
        "nnn_0.1": coupling_chain_K(N, [0.6, 0.1]),
        "improved": coupling_chain_K(N, [4.0 / 3.0, -1.0 / 12.0]),
        "k4_1": Lap + Lap @ Lap,
        "lifshitz": dispersion_power_K(N, 2.0) / 4.0,
        "z_0.5": dispersion_power_K(N, 0.5),
        "random_0.5": random_symmetric_chain_K(N, 0.5, seed=1),
        "longrange_2": _long_range_K(N, 2.0),
    }
    return {name: (K, np.linalg.eigh(K)) for name, K in fam.items()}


# --- models, local forms, variations --------------------------------------


def test_models_and_local_energy_forms():
    assert np.allclose(coupling_chain_K(N, [1.0]), harmonic_chain_K(N, 0.0, bc="dirichlet"))
    K = coupling_chain_K(40, [0.7, 0.2, 0.1], m=0.3)
    target = np.block([[K, np.zeros((40, 40))], [np.zeros((40, 40)), np.eye(40)]])
    for split in ("bond", "site"):
        assert np.allclose(sum(local_energy_forms(K, split)), target), split
    assert min(np.linalg.eigvalsh(h).min() for h in local_energy_forms(K, "bond")) > -1e-12
    Kr = random_symmetric_chain_K(200, 0.5, seed=3)
    assert np.allclose(Kr, Kr[::-1, ::-1]) and np.linalg.eigvalsh(Kr).min() > 0
    # bulk symbol: v^2 = sum c_n n^2 and the k^2/24 lattice correction of the wave chain
    k = 1e-3
    assert abs(bulk_dispersion([0.7, 0.2, 0.1], 0.0, k) / k - np.sqrt(0.7 + 0.8 + 0.9)) < 1e-5
    dep = dispersion_departure([1.0], 0.0, 8)
    assert abs(dep - (np.pi / 8) ** 2 / 24) < 2e-4, f"NN departure at L=8: {dep:.3e}"
    assert dispersion_departure([4 / 3, -1 / 12], 0.0, 8) < 0.05 * dep


def test_squeeze_update_is_exact_and_modes_have_parity():
    n = 200
    K = coupling_chain_K(n, [1.0])
    eig = np.linalg.eigh(K)
    V0 = ground_state_cov(K)
    ball = centred_ball(n, 8)
    for parity, sign in (("odd", -1.0), ("even", 1.0)):
        for mode in squeeze_modes(eig, 8, ratios=RATIOS, parity=parity):
            assert np.sign(mode["parity"]) == sign
            assert 0.5 < mode["wavelength"] / mode["wavelength_target"] < 2.0
    mode = squeeze_modes(eig, 8, ratios=(4,), parity="odd")[0]
    u, w, e = mode["u"], mode["omega"], 3e-3
    M = np.zeros((2 * n, 2 * n))
    M[:n, n:] = np.outer(u, u)
    M[n:, :n] = np.outer(u, u)
    S = expm(e * Omega(n) @ M)
    exact = reduce(S @ V0 @ S.T, ball)
    rank1 = squeezed_ball_covariance(reduce(V0, ball), u[ball], w, e)
    assert np.abs(exact - rank1).max() < 1e-13


def test_odd_variations_are_blind_to_the_zero_mode_even_ones_are_not(family):
    K, eig = family["wave"]
    det = equilibrium_residual(K, sizes=(8, 16), ratios=RATIOS, parity="odd", eig=eig,
                               return_details=True)
    shares = [abs(row["top_mode_share"]) for row in det["rows"]]
    assert max(shares) < 1e-12, f"odd variations see the top (zero) mode: {max(shares):.1e}"
    assert det["residual"] < 0.03
    even = equilibrium_residual(K, sizes=(8, 16), ratios=RATIOS, parity="even", eig=eig,
                                return_details=True)
    assert min(even["per_size"].values()) > 0.3, (
        f"even (antinode) variations should show the zero-mode obstruction: {even['per_size']}")


def test_floor_escalation_path(family):
    K, eig = family["wave"]
    base = equilibrium_residual(K, sizes=(8,), ratios=(4,), eig=eig, return_details=True)
    forced = equilibrium_residual(K, sizes=(8,), ratios=(4,), eig=eig, return_details=True,
                                  escalate_above=0.0)
    assert forced["rows"][0]["escalated"] and not base["rows"][0]["escalated"]
    assert abs(forced["residual"] - base["residual"]) < 1e-6 * max(1.0, base["residual"])
    assert abs(forced["rows"][0]["floor_share"]) < 1e-6


# --- anchor 1 ---------------------------------------------------------------


def test_anchor1_wave_chain_residual_vanishes_with_ball_size(family):
    K, eig = family["wave"]
    sizes = (4, 6, 8, 12, 16, 24)
    det = equilibrium_residual(K, sizes=sizes, ratios=RATIOS, eig=eig, return_details=True)
    r = np.array([det["per_size"][L] for L in sizes])
    assert np.all(np.diff(r) < 0), f"r(L) not decreasing: {dict(zip(sizes, r))}"
    Ls = np.array(sizes, dtype=float)
    sel = Ls >= 8
    slope = np.polyfit(np.log(Ls[sel]), np.log(r[sel]), 1)[0]
    assert -3.0 < slope < -1.5, f"lattice-correction exponent {slope:.2f} (measured -2.2)"
    A = np.vstack([np.ones(sel.sum()), Ls[sel] ** -2.0]).T
    r_inf2 = np.linalg.lstsq(A, r[sel], rcond=None)[0][0]
    A = np.vstack([np.ones(sel.sum()), Ls[sel] ** slope]).T
    r_infp = np.linalg.lstsq(A, r[sel], rcond=None)[0][0]
    assert abs(r_inf2) < 5e-4 and abs(r_infp) < 5e-4, (
        f"continuum extrapolation not consistent with 0: {r_inf2:.2e} (1/L^2), "
        f"{r_infp:.2e} (L^{slope:.2f}); r = {dict(zip(sizes, np.round(r, 5)))}")
    assert r[-1] < 3e-3, f"r(L=24) = {r[-1]:.3e}"


# --- anchor 2 ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["wave", "massive_0.2", "nnn_0.1", "improved", "k4_1",
                                  "lifshitz", "z_0.5", "random_0.5", "longrange_2"])
def test_anchor2_exact_first_law_every_dynamics(family, name):
    K, eig = family[name]
    for L in (4, 8):
        a = first_law_audit(K, L, ratio=4, eig=eig, use_precise=True)
        assert a["rel_err"] < 1e-8, (
            f"[{name}, L={L}] first law: dS_fd={a['dS_fd']:.9e} dS_modular={a['dS_modular']:.9e} "
            f"rel_err={a['rel_err']:.2e} (margin {a['nu_min_margin']:.1e})")


def test_anchor2_precision_ladder_is_needed_for_lifshitz(family):
    K, eig = family["lifshitz"]
    a_f = first_law_audit(K, 8, eig=eig, use_precise=False)
    a_p = first_law_audit(K, 8, eig=eig, use_precise=True)
    assert a_f["nu_min_margin"] < 1e-10  # float64 margins at/below the floor
    assert a_p["rel_err"] < 1e-8 < a_f["rel_err"], (
        f"float64 {a_f['rel_err']:.1e} vs ladder {a_p['rel_err']:.1e}")


# --- anchor 3 ---------------------------------------------------------------


def test_anchor3_nonlocal_dynamics_bounded_away(family):
    sizes = (8, 16, 32)
    wave = equilibrium_residual(family["wave"][0], sizes=sizes, ratios=RATIOS,
                                eig=family["wave"][1], return_details=True)["per_size"]
    lif = equilibrium_residual(family["lifshitz"][0], sizes=sizes, ratios=RATIOS,
                               eig=family["lifshitz"][1], return_details=True,
                               escalate_above=1e-3)["per_size"]
    assert min(lif.values()) > 0.9, f"Lifshitz r(L) = {lif}"
    assert lif[32] >= lif[8], f"Lifshitz residual should not decay: {lif}"
    assert min(lif[L] / wave[L] for L in sizes) > 20, f"teeth: {lif} vs {wave}"
    z = equilibrium_residual(family["z_0.5"][0], sizes=sizes, ratios=RATIOS,
                             eig=family["z_0.5"][1], return_details=True,
                             escalate_above=1e-3)["per_size"]
    assert min(z.values()) > 1.0, f"omega ~ sqrt(k): {z}"
    lr = equilibrium_residual(family["longrange_2"][0], sizes=sizes, ratios=RATIOS,
                              eig=family["longrange_2"][1], return_details=True,
                              escalate_above=1e-3)["per_size"]
    assert min(lr.values()) > 0.1, f"long-range alpha=2: {lr}"


# --- what r tracks -----------------------------------------------------------


def test_residual_tracks_departure_from_lorentz_invariance(family):
    K, eig = family["wave"]
    lam, U = eig
    for v in (0.5, 0.8, 1.25):
        r = equilibrium_residual(v ** 2 * K, sizes=(8, 16), ratios=RATIOS, eig=(v ** 2 * lam, U),
                                 escalate_above=1e-3)
        assert abs(r - abs(1.0 - v)) < 0.03, f"v={v}: r={r:.4f} vs |1-v|={abs(1 - v):.3f}"
    rm = [equilibrium_residual(coupling_chain_K(N, [1.0], m=m), sizes=(16,), ratios=RATIOS,
                               escalate_above=1e-3) for m in (0.0, 0.05, 0.2)]
    assert rm[0] < rm[1] < rm[2], f"mass: {rm}"
    rk = [equilibrium_residual(K + a * K @ K, sizes=(8,), ratios=RATIOS, escalate_above=1e-3)
          for a in (0.0, 0.5, 2.0)]
    assert rk[0] < rk[1] < rk[2], f"k^4 deformation: {rk}"
    deps = [dispersion_departure([1 + 4 * a, -a], 0.0, 8) for a in (0.0, 0.5, 2.0)]
    assert deps[0] < deps[1] < deps[2]


def test_eh_profiles_cross_check(family):
    pw = eh_profiles(family["wave"][0], 16, eig=family["wave"][1])
    pl = eh_profiles(family["lifshitz"][0], 16, eig=family["lifshitz"][1])
    assert pw["locality"][1] > 0.98 and pl["locality"][1] < 0.6, (pw["locality"], pl["locality"])
    interior = slice(4, 12)
    assert np.all((pw["pp_rowsum_over_beta"][interior] > 0.9) & (pw["pp_rowsum_over_beta"][interior] < 1.3))
    assert np.all(pl["pp_rowsum_over_beta"][interior] > 5.0)
    assert np.all(np.abs(pw["xx_resummed_over_beta"][interior] - 1.0) < 0.25)


# --- anchor 4 ---------------------------------------------------------------


def test_anchor4_constrain_dynamics_from_generic_start():
    n = 512
    K_of = lambda th: coupling_chain_K(n, [th[0], th[1]], m=abs(th[2]))  # noqa: E731
    theta0 = (2.0, -0.3, 0.2)  # v^2 = 0.8, massive, wrong-sign NNN coupling
    theta, info = constrain_dynamics(K_of, theta0=theta0, sizes=(8, 16), ratios=RATIOS,
                                     bounds=[(0.05, 4.0), (-0.5, 0.5), (0.0, 1.0)],
                                     maxfev=250, restarts=1)
    c1, c2, m = theta[0], theta[1], abs(theta[2])
    v2 = c1 + 4.0 * c2
    r_wave = equilibrium_residual(coupling_chain_K(n, [1.0]), sizes=(8, 16), ratios=RATIOS,
                                  escalate_above=1e-3)
    msg = (f"landing: c1={c1:.4f} c2={c2:.4f} m={m:.2e} v^2={v2:.4f} r={info['residual']:.3e} "
           f"(wave chain {r_wave:.3e}), nfev={info['nfev']}")
    assert abs(v2 - 1.0) < 0.05, msg
    assert m < 0.02, msg
    assert abs(c2) < 0.35, msg  # the flat valley: coupling range unresolved at L <= 16
    assert info["residual"] <= 1.05 * r_wave, msg


# --- anchor 5: the ansatz floor is a choice ---------------------------------


def test_anchor5_bond_centred_beta_lowers_the_discretisation_floor():
    """beta at the bond midpoints instead of the site centres: floor 1.38/L^2 -> 0.054/L^2."""
    n = 1600
    K = coupling_chain_K(n, [1.0])
    eig = np.linalg.eigh(K)
    sizes = (8, 16)
    r = {}
    for ba in ("site", "bond"):
        det = equilibrium_residual(K, sizes=sizes, ratios=RATIOS, eig=eig, return_details=True,
                                   escalate_above=1e-3, beta_at=ba)
        r[ba] = {L: det["per_size"][L] for L in sizes}
    A = {ba: {L: r[ba][L] * L**2 for L in sizes} for ba in r}
    assert all(1.2 < A["site"][L] < 1.6 for L in sizes), f"site-centred floor A = {A['site']} (measured 1.39, 1.35)"
    assert all(A["bond"][L] < 0.10 for L in sizes), f"bond-centred floor A = {A['bond']} (measured 0.054, 0.067)"
    for L in sizes:
        assert r["site"][L] / r["bond"][L] > 15.0, (
            f"L={L}: floor ratio site/bond = {r['site'][L] / r['bond'][L]:.1f} (measured 26, 20)")
    # the two agree in the continuum: both are 2 pi int beta T_00 to O(1/L^2)
    G_s = local_modular_form(K, centred_ball(n, 16), beta_at="site")
    G_b = local_modular_form(K, centred_ball(n, 16), beta_at="bond")
    assert abs(np.trace(G_s) / np.trace(G_b) - 1.0) < 0.02, (
        f"the two discretisations must agree at O(1/L^2): traces {np.trace(G_s):.3f}, {np.trace(G_b):.3f}")


# --- anchor 6: the k^4 coefficient and its degeneracy -----------------------


def test_anchor6_k4_response_needs_the_third_neighbour_coupling():
    """A pure-dispersion response requires d rho/d c_3 = 6 d rho/d c_2 (A_4 = 1 + 12c_2 + 72c_3).

    Which assertion carries what. The SHAPE assertions (l3/l2 against 6, and rms(A_4 only)
    against rms(free two-parameter fit)) are meaningful only once the coupling response is
    resolved above the fit's own scatter: a null ansatz, whose rho does not depend on the
    couplings at all, has l_2 = l_3 = 0 to roundoff and would satisfy both by numerical luck
    (measured on this grid: l3/l2 = 4.5, rms ratio 0.67). The RESOLUTION assertions below --
    |l_2| > 5 rms and |l_3| > 10 rms, measured 15-45 and 34-279 for both real ansatze and 0.5
    for the null -- are therefore checked first and give the shape assertions their teeth. The
    C window is the quantitative claim (the k^4 coefficient itself) and rejects a null ansatz
    on its own.
    """
    out = {ba: k4_sensitivity(1200, sizes=(8, 16), ratios=(4,), beta_at=ba, escalate_above=1e-3)
           for ba in ("site", "bond")}
    for L in (8, 16):
        v_s = out["site"]["rows"][(L, 4.0)]
        v_b = out["bond"]["rows"][(L, 4.0)]
        for name, v in (("site", v_s), ("bond", v_b)):  # resolution: the response is not noise
            assert abs(v["l2"]) > 5.0 * v["rms_3par"] and abs(v["l3"]) > 10.0 * v["rms_3par"], (
                f"L={L} [{name}]: the coupling response is not resolved above the fit scatter, so "
                f"l3/l2 and the rms ratio are meaningless: |l2|/rms = {abs(v['l2']) / v['rms_3par']:.1f} "
                f"(measured 15.1-45.2), |l3|/rms = {abs(v['l3']) / v['rms_3par']:.1f} (measured 33.8-278.8; "
                f"a null ansatz gives 0.5)")
        assert abs(v_s["l3_over_l2"] - 6.0) < 0.8, (
            f"L={L}: the site-centred ansatz's coupling response must be a function of A_4 alone: "
            f"l3/l2 = {v_s['l3_over_l2']:.3f} vs 6 (measured 6.33, 6.17)")
        assert v_s["rms_A4_only"] / v_s["rms_3par"] < 1.2, (
            f"L={L}: fitting in A_4 alone must be as good as the free (c_2, c_3) fit: "
            f"{v_s['rms_A4_only']:.2e} vs {v_s['rms_3par']:.2e}")
        assert -1.15 < v_s["C"] < -0.75, (
            f"L={L}: k^4 sensitivity at lambda = 4L is C = {v_s['C']:.3f} (measured -0.889, -0.975)")
        assert v_b["l3_over_l2"] < 3.5, (
            f"L={L}: the bond-centred ansatz must FAIL the pure-dispersion test (it buys its low "
            f"floor with a range-dependent error): l3/l2 = {v_b['l3_over_l2']:.3f} (measured 2.23, 2.35)")
        assert v_b["rms_A4_only"] / v_b["rms_3par"] > 2.0, (
            f"L={L}: A_4 alone must NOT explain the bond ansatz's response: "
            f"{v_b['rms_A4_only']:.2e} vs {v_b['rms_3par']:.2e} (measured ratios 2.3, 6.1)")


# --- anchor 7: a second ball position ---------------------------------------


def test_anchor7_ball_attached_to_the_wall_needs_the_boundary_weight():
    n = 800
    K = coupling_chain_K(n, [1.0])
    eig = np.linalg.eigh(K)
    kw = dict(ratios=RATIOS, eig=eig, return_details=True, escalate_above=1e-3,
              beta_at="bond", parity="any", zero_mode="project")
    for L, tol in ((8, 0.03), (16, 0.012)):
        attached = list(range(L))
        centred = centred_ball(n, L)
        det_w = equilibrium_residual(K, balls=[attached], wall=-1.0, **kw)
        det_p = equilibrium_residual(K, balls=[attached], wall=None, **kw)
        det_c = equilibrium_residual(K, balls=[centred], wall=None, **kw)
        assert det_w["residual"] < tol, (
            f"L={L}: attached ball with the boundary-CFT weight r = {det_w['residual']:.3e} "
            f"(measured 2.26e-2, 7.59e-3)")
        assert det_p["residual"] / det_w["residual"] > 30.0, (
            f"L={L}: the bulk parabola at the wall must fail: {det_p['residual']:.3e} vs "
            f"{det_w['residual']:.3e} (measured 0.884 vs 2.26e-2, 3.33 vs 7.59e-3)")
        long_wave = min(row["r"] for row in det_w["rows"])
        assert long_wave < max(det_c["per_size"].values()), (
            f"L={L}: the attached ball's longest-wavelength residual {long_wave:.2e} should be no worse "
            f"than the centred ball's {max(det_c['per_size'].values()):.2e}")
        # detached: the image bilocal term has no local counterpart
        det_d = equilibrium_residual(K, balls=[list(range(8, 8 + L))], wall=-1.0, **kw)
        assert det_d["residual"] > 0.3, (
            f"L={L}: a ball 9 sites from the wall must stay obstructed (bilocal image term): "
            f"{det_d['residual']:.3e} (measured 1.25, 1.85)")
    # the weight itself: attached is the (B^2 - y^2)/(2B) of the half line, not the parabola
    w = bw_weights(list(range(8)), wall=-1.0)
    B = 8.5
    assert np.max(np.abs(w - (B**2 - (np.arange(8) + 1.0) ** 2) / (2 * B))) < 1e-12
    assert np.max(np.abs(bw_weights(list(range(400, 408)), wall=-1.0) / bw_weights(list(range(400, 408))) - 1.0)) < 1e-3


# --- anchor 8: the zero mode projected out rather than excluded by parity ----


def test_anchor8_projecting_the_zero_mode_out_of_the_variation(family):
    n = 800
    K = coupling_chain_K(n, [1.0])
    eig = np.linalg.eigh(K)
    sizes = (8, 16)
    res = {}
    for parity in ("odd", "even"):
        for zm in ("parity", "project"):
            res[(parity, zm)] = equilibrium_residual(K, sizes=sizes, ratios=RATIOS, eig=eig,
                                                     return_details=True, escalate_above=1e-3,
                                                     beta_at="bond", parity=parity, zero_mode=zm)["per_size"]
    for L in sizes:
        assert abs(res[("odd", "parity")][L] / res[("odd", "project")][L] - 1.0) < 1e-9, (
            "odd variations are already blind to the zero mode: projecting must change nothing")
        assert res[("even", "parity")][L] > 1.5, f"L={L}: unprojected even residual {res[('even', 'parity')][L]:.3f}"
        assert res[("even", "project")][L] < 0.10, (
            f"L={L}: projecting the zero mode out of the variation must remove the O(1) obstruction: "
            f"{res[('even', 'parity')][L]:.3f} -> {res[('even', 'project')][L]:.3e} (measured 2.25 -> 5.1e-2, 2.07 -> 5.9e-2)")
        assert res[("even", "project")][L] / res[("odd", "project")][L] > 20.0, (
            f"L={L}: ...but the even sector does NOT become consistent with the odd one: "
            f"{res[('even', 'project')][L]:.3e} vs {res[('odd', 'project')][L]:.3e} (measured 58x, 128x)")
    assert res[("even", "project")][16] > 0.5 * res[("even", "project")][8], (
        f"the projected even residual does not decay with L: {res[('even', 'project')]}")
    # canary: the projection does not rescue a non-relativistic dynamics
    Kl, eigl = family["lifshitz"]
    lif = equilibrium_residual(Kl, sizes=(8,), ratios=RATIOS, eig=eigl, return_details=True,
                               escalate_above=1e-3, beta_at="bond", parity="odd",
                               zero_mode="project")["per_size"]
    assert lif[8] > 0.9, f"Lifshitz with the projection: {lif} (measured 1.19)"


# --- anchor 9: the weight set free ------------------------------------------


def test_anchor9_free_weight_recovers_chm_on_the_wave_chain(family):
    """With the weight free, the wave chain's minimiser is the CHM parabola (found, not
    assumed) and r_free <= r for every family that contains CHM."""
    K, eig = family["wave"]
    sizes = (8, 16)
    d = free_weight_residual(K, sizes=sizes, eig=eig, families=("cheb2", "cheb3", "site"))
    for L in sizes:
        r = d[L]
        R = 0.5 * L
        for key, rf in r["r_free"].items():
            assert rf <= r["r_chm"][key[1]] * (1 + 1e-9), (
                f"L={L} {key}: r_free {rf:.3e} > r_chm {r['r_chm'][key[1]]:.3e} (CHM is in the family)")
        assert d["per_size"][L] < 1e-4, f"L={L}: r_free(cheb3) = {d['per_size'][L]:.3e} (measured 3.9e-5, 9.7e-6)"
        assert d["per_size"][L] < 0.05 * r["r_chm"]["bond"], (
            f"L={L}: r_free {d['per_size'][L]:.2e} should be well below the CHM residual {r['r_chm']['bond']:.2e} "
            f"(measured ratios 0.041, 0.020)")
        th = r["theta"][("cheb3", "bond")] / (np.pi * R / 2.0)  # CHM = (pi R/2)(T_0 - T_2) -> (1, -1, 0)
        assert abs(th[0] - 1.0) < 0.01 and abs(th[1] + 1.0) < 0.01 and abs(th[2]) < 0.01, (
            f"L={L}: recovered Chebyshev coefficients / (pi R/2) = {np.round(th, 4)} vs CHM (1, -1, 0) "
            f"(measured (1.0010, -1.0007, -0.0003), (1.0005, -1.0005, -0.0000))")
        assert r["overlap"][("cheb3", "bond")] > 0.9999, f"L={L}: overlap with CHM {r['overlap'][('cheb3', 'bond')]:.6f}"
        assert abs(r["scale"][("cheb3", "bond")] - 1.0) < 0.01, (
            f"L={L}: the recovered normalisation must be CHM's 2 pi (R^2 - x^2)/(2R): scale {r['scale'][('cheb3', 'bond')]:.4f}")
        th2 = r["theta"][("cheb2", "bond")] / (np.pi * R / 2.0)
        assert abs(th2[0] - 1.0) < 0.01 and abs(th2[1] + 1.0) < 0.01, f"L={L}: 2-parameter family {np.round(th2, 4)}"
    # the piece-weight map reproduces the CHM forms of local_modular_form exactly (site and bond)
    from vacuum.geometry.jacobson import local_pieces
    pieces = local_pieces(K, centred_ball(N, 8))
    for disc in ("site", "bond"):
        w = weight_family("chm", pieces, disc)[0][:, 0]
        assert np.all(w >= 0) and abs(w[pieces["kind"] == 0].max() / (2 * np.pi) - bw_weights(centred_ball(N, 8)).max()) < 1e-12


def test_anchor9_variation_family_is_low_rank_so_a_per_site_weight_has_no_teeth(family):
    """Jacobson's regime (ball << wavelength) imposes only ~5-6 independent conditions at 1e-6:
    a per-site weight (L/2 + 1 parameters) fits the wave chain AND the Lifshitz chain."""
    for name in ("wave", "lifshitz"):
        K, eig = family[name]
        d = free_weight_residual(K, sizes=(8, 16), eig=eig, families=("site",), signed=False)
        for L in (8, 16):
            r = d[L]
            assert r["rank"][1e-6] <= 8, f"[{name}] L={L}: rank of the variation family at 1e-6 is {r['rank']} (measured 5-6)"
            assert r["n_params"][("site", "bond")] >= r["rank"][1e-6] - 1
            assert r["r_free"][("site", "bond")] < 1e-4, (
                f"[{name}] L={L}: per-site weight r_free = {r['r_free'][('site', 'bond')]:.2e} "
                f"(measured wave 2.1e-6, 4.5e-7; lifshitz 4.5e-5, 1.9e-6): the per-site test fits anything")
        assert d[8]["n_variations"] > 100 and d[8]["n_variations"] > 10 * d[8]["n_modes"]  # pairs are included


def test_anchor9_lifshitz_three_and_eight_mode_floors_are_family_properties(family):
    """Recorded floors, NOT existence claims (two adversarial audits, 2026-09-04): a nonnegative
    FOUR-mode weight fits a single Lifshitz ball (6.9e-4 at L = 8, degrading 8x to 5.6e-3 at
    L = 16), so "no local weight exists" is false at fixed L; the three-mode per-ball numbers
    0.27, 0.48 and the eight-mode shared-profile floor 0.09 are properties of those families
    (the parsimony test below shows the shared floor falls to 5e-4 by 12-14 modes). This test
    keeps the numbers pinned as what they are."""
    Kw, eigw = family["wave"]
    Kl, eigl = family["lifshitz"]
    dw = free_weight_residual(Kw, sizes=(8, 16), eig=eigw, families=("cheb3", "cheb4"))
    dl = free_weight_residual(Kl, sizes=(8, 16), eig=eigl, families=("cheb3", "cheb4"))
    # per ball, four nonnegative modes DO fit Lifshitz -- the old "no local weight" claim is dead
    c4 = {L: min(dl[L]["r_free"][("cheb4", dc)] for dc in ("site", "bond")) for L in (8, 16)}
    assert c4[8] < 2e-3 and c4[16] < 2e-2, (
        f"Lifshitz cheb4 (nonnegative) r_free = {c4} (measured 6.9e-04, 5.6e-03): a per-ball local weight exists")
    assert all(dl[L]["n_params"][("cheb4", "bond")] < dl[L]["rank"][1e-6] for L in (8, 16)), "cheb4 is below the rank"
    assert c4[16] > 3.0 * c4[8], f"...and the per-ball fit degrades with L: {c4} (measured 8.1x)"
    # the three-mode family's numbers, recorded as a family property
    assert dl["per_size"][8] > 0.2 and dl["per_size"][16] > 0.4, (
        f"Lifshitz r_free(cheb3) = {dl['per_size']} (measured 0.270, 0.482; a three-mode-family property)")
    for L in (8, 16):
        assert min(dl[L]["r_chm"].values()) > 0.9  # the CHM weight itself, on the same family
        sgn = dl[L]["r_signed"][("cheb3", "bond")]
        assert sgn < 5e-3, f"L={L}: signed (unphysical) minimum {sgn:.2e} (measured 1.1e-03, 8.6e-04) -- recorded, not claimed"
        assert dw["per_size"][L] < 1e-4
    # one profile R^z f(x/R) for L = 8, 16, 32: over-determined even for 8 modes
    n = 1600
    Kw2 = coupling_chain_K(n, [1.0])
    Kl2 = Kw2 @ Kw2 / 4.0
    uw = universal_weight_residual(Kw2, sizes=(8, 16, 32), families=("cheb3", "cheb8"), exponents=(1.0, 2.0), signed=False)
    ul = universal_weight_residual(Kl2, sizes=(8, 16, 32), families=("cheb3", "cheb8"), exponents=(1.0, 2.0), signed=False)
    assert ul["n_conditions"] > 8, f"the universal test must be over-determined: {ul['n_conditions']} conditions (measured 15)"
    assert ul["residual"] > 0.05, (
        f"Lifshitz universal best {ul['best']} r = {ul['residual']:.3e} (measured 0.091): the EIGHT-mode "
        f"shared-profile floor, a family property (see test_anchor9_parsimony_gap_is_basis_independent)")
    assert uw["residual"] < 1e-4, f"wave universal best {uw['best']} r = {uw['residual']:.3e} (measured 2.6e-5)"
    key = ("cheb3", "bond", 1.0)
    assert uw[key]["r"] < 3e-4 and all(v > 0.999 for v in uw[key]["overlap"].values()), (
        f"wave: one CHM parabola scaling as R serves L = 8, 16, 32: r = {uw[key]['r']:.2e}, overlaps {uw[key]['overlap']}")
    assert uw[("cheb3", "bond", 2.0)]["r"] > 0.3, "the wave chain's profile does NOT scale as R^2"


def test_anchor9_two_mode_squeezes_obey_the_exact_first_law():
    """pair_variation is the first-order change of the exactly transformed state, and its
    pairing with the exact modular Hamiltonian is the entropy derivative."""
    from scipy.linalg import expm
    from vacuum.core import entanglement_hamiltonian, entropy, modular_energy
    n = 200
    K = coupling_chain_K(n, [1.0])
    eig = np.linalg.eigh(K)
    V0 = ground_state_cov(K)
    ball = centred_ball(n, 8)
    var = smooth_variations(eig, 8, ratio_min=4.0, pairs=True)
    assert len(var["pairs"]) == len(var["modes"]) * (len(var["modes"]) + 1) // 2
    a, b = var["pairs"][1]
    assert a != b
    ua, ub = eig[1][:, a], eig[1][:, b]
    wa, wb = np.sqrt(eig[0][a]), np.sqrt(eig[0][b])
    A = 0.5 * (np.outer(ua, ub) + np.outer(ub, ua))
    M = np.zeros((2 * n, 2 * n))
    M[:n, n:] = A
    M[n:, :n] = A
    X = Omega(n) @ M
    dV = pair_variation(ua[ball], wa, ub[ball], wb)
    assert np.allclose(pair_variation(ua[ball], wa, ua[ball], wa), np.block(
        [[np.outer(ua[ball], ua[ball]) / wa, np.zeros((8, 8))], [np.zeros((8, 8)), -wa * np.outer(ua[ball], ua[ball])]]))
    S1 = expm(1e-6 * X)
    dV_num = (reduce(S1 @ V0 @ S1.T, ball) - reduce(V0, ball)) / 1e-6
    assert np.abs(dV_num - dV).max() < 1e-6 * np.abs(dV).max()
    G_B = entanglement_hamiltonian(reduce(V0, ball))
    dS_mod = modular_energy(G_B, dV)

    def S_of(e):
        S = expm(e * X)
        return entropy(reduce(S @ V0 @ S.T, ball))

    eps = 4e-3
    D_c = (S_of(eps) - S_of(-eps)) / (2 * eps)
    D_f = (S_of(0.5 * eps) - S_of(-0.5 * eps)) / eps
    dS_fd = (4.0 * D_f - D_c) / 3.0
    assert abs(dS_fd - dS_mod) < 1e-7 * abs(dS_mod), f"two-mode first law: {dS_fd:.10e} vs {dS_mod:.10e}"


def test_anchor9_free_weight_absorbs_velocity_and_per_ball_mass_but_the_shared_profile_pins_m():
    """v is a unit (the vacuum of v^2 K is that of K): r_free is exactly flat in v. The
    per-ball 3-mode weight also absorbs a mass (a rescaled parabola fits a massive ball),
    and even a massive, wrong-velocity landing of the per-ball fit lies BELOW the wave
    chain's own r_free; only one profile shared across ball sizes sees the mass, and the
    constrained fit with it lands at m/v = 0 with the coupling range near nearest-neighbour."""
    n = 512
    sizes = (8, 16)

    def both(K):
        eig = np.linalg.eigh(K)
        d = free_weight_residual(K, sizes=sizes, eig=eig, families=(), signed=False, escalate_above=1e-3)
        u = universal_weight_residual(K, sizes=sizes, design=d["design"], families=("cheb3",), exponents=(1.0, 2.0), signed=False)
        return d["residual"], u["residual"]

    f1, u1 = both(coupling_chain_K(n, [1.0]))
    for v in (0.5, 2.0):
        fv, uv = both(v * v * coupling_chain_K(n, [1.0]))
        assert abs(fv / f1 - 1) < 1e-6 and abs(uv / u1 - 1) < 1e-6, f"v={v}: {fv:.3e} vs {f1:.3e}; {uv:.3e} vs {u1:.3e}"
    fm, um = both(coupling_chain_K(n, [1.0], m=0.05))
    assert fm < 3.0 * f1, f"per ball, m = 0.05 is absorbed: r_free {fm:.2e} vs massless {f1:.2e} (measured 1.8e-5 vs 3.7e-5)"
    assert um > 20.0 * u1, f"the shared profile sees m = 0.05: {um:.2e} vs massless {u1:.2e} (measured 2.3e-2 vs 6.3e-5)"
    f_land, _ = both(coupling_chain_K(n, [1.24, -0.11], m=0.43))  # a landing of the per-ball free fit
    assert f_land < f1, f"the per-ball free fit's massive landing has r_free {f_land:.2e} < wave chain {f1:.2e}: nothing is pinned"
    K_of = lambda th: coupling_chain_K(n, [th[0], th[1]], m=abs(th[2]))  # noqa: E731
    theta, info = constrain_dynamics(K_of, theta0=(1.5, 0.1, 0.05), sizes=sizes,
                                     bounds=[(0.05, 4.0), (-0.5, 0.5), (0.0, 1.0)], maxfev=100, restarts=0,
                                     residual="universal", free_kwargs={"exponents": (1.0, 2.0), "escalate_above": 1e-3})
    c1, c2, m = theta[0], theta[1], abs(theta[2])
    v = np.sqrt(c1 + 4 * c2)
    msg = f"universal landing: c1={c1:.4f} c2={c2:.4f} m={m:.2e}: m/v={m / v:.3e} c2/c1={c2 / c1:+.4f} r={info['residual']:.2e}"
    assert m / v < 0.05, msg + " (measured m/v = 2.8e-3)"
    assert abs(c2 / c1) < 0.1, msg + " (measured c2/c1 = -0.013)"
    assert info["residual"] < 3e-4, msg + " (measured 2.9e-5)"


def test_anchor9_parsimony_gap_is_basis_independent():
    """The surviving Lifshitz statement is about the COMPLEXITY of the admissible weight, not
    its existence: with one nonnegative profile R^z f(x/R) shared across L = 8, 16, 32, the wave
    chain is at 1.5e-4 with 3 Chebyshev modes (the parabola; 4 hats) and only with z = 1, while
    Lifshitz stays above 1e-2 up to 10 modes and above 1e-3 up to 11 modes for every z in
    {1, 2, 3} in BOTH bases -- and then does fit (below 1e-2 by 11-13 modes and below 1e-3 by
    12-14, z-dependent; 5e-4 only by 20 Chebyshev modes; beyond ~13 modes the family exceeds the
    variation family's 8-9 conditions at 1e-3). Both halves are asserted, so neither "no local
    weight for Lifshitz" nor "Lifshitz fits as easily as the wave chain" can come back; and the
    two bases are asserted to DIFFER (hat9 = 0.99 vs cheb9 = 0.033: the hat rows alternate with
    the node set, a genuine feature), so 'basis-independent' cannot be one basis twice."""
    n = 1600
    sizes = (8, 16, 32)
    Kw = coupling_chain_K(n, [1.0])
    Kl = Kw @ Kw / 4.0
    ns = (3, 4, 6, 8, 9, 10, 11, 12, 14)
    zs = (1.0, 2.0, 3.0)
    r = {}
    cond = {}
    for name, K in (("wave", Kw), ("lifshitz", Kl)):
        d = free_weight_residual(K, sizes=sizes, families=(), signed=False, rank_tols=(1e-3, 1e-6))
        cond[name] = sum(d[L]["rank"][1e-3] for L in sizes)
        for basis in ("cheb", "hat"):
            u = universal_weight_residual(K, sizes=sizes, design=d["design"], families=tuple(f"{basis}{k}" for k in ns),
                                          exponents=zs, discretisations=("bond",), signed=False)
            r[(name, basis)] = {(k, z): u[(f"{basis}{k}", "bond", z)]["r"] for k in ns for z in zs}
    w, l = r[("wave", "cheb")], r[("lifshitz", "cheb")]
    assert w[(3, 1.0)] < 2e-4 and r[("wave", "hat")][(4, 1.0)] < 2e-4, (
        f"wave: 3 Chebyshev modes {w[(3, 1.0)]:.2e} / 4 hats {r[('wave', 'hat')][(4, 1.0)]:.2e} (measured 1.5e-4, 1.5e-4)")
    assert min(w[(3, 2.0)], w[(3, 3.0)], w[(14, 2.0)], w[(14, 3.0)]) > 0.1, "the wave chain's profile scales as R and nothing else"
    for basis in ("cheb", "hat"):
        for z in zs:
            for k in ns:
                v = r[("lifshitz", basis)][(k, z)]
                if k <= 10:
                    assert v > 1e-2, f"Lifshitz {basis}{k} z={z:g}: {v:.2e} (measured >= 2.3e-2 up to 10 modes)"
                if k <= 11:
                    assert v > 1e-3, f"Lifshitz {basis}{k} z={z:g}: {v:.2e} (measured >= 5.5e-3 up to 11 modes)"
            assert min(r[("lifshitz", basis)][(12, z)], r[("lifshitz", basis)][(14, z)]) < 3e-3, (
                f"Lifshitz {basis} z={z:g}: a scale-covariant nonnegative weight EXISTS by 12-14 modes: "
                f"{r[('lifshitz', basis)][(12, z)]:.2e}, {r[('lifshitz', basis)][(14, z)]:.2e} (measured 5.3e-4 to 2.0e-3)")
    assert l[(8, 2.0)] > 0.05, f"the 8-mode floor 0.09 is a family property: {l[(8, 2.0)]:.2e}"
    # the two bases are genuinely different bases (an aliased hat basis would pass everything above)
    h = r[("lifshitz", "hat")]
    assert abs(h[(9, 2.0)] - l[(9, 2.0)]) > 0.5, (
        f"hat9 {h[(9, 2.0)]:.2e} vs cheb9 {l[(9, 2.0)]:.2e} at z = 2 (measured 0.99 vs 0.033): the bases must differ")
    assert max(abs(h[key] - l[key]) for key in l) > 0.1 and max(abs(r[('wave', 'hat')][key] - w[key]) for key in w) > 5e-3
    assert 6 <= cond["lifshitz"] <= 14 and 6 <= cond["wave"] <= 14, (
        f"conditions at 1e-3 across the three balls: {cond} -- the ladder's floor at >= 13 modes is the trivial regime")


# ===========================================================================
# Layer 7 unit 2, item (a): what the reflection-EVEN sector actually is.
#
# MANIFEST anchor 8 left the even sector "still inconsistent": with the ball's
# top Williamson mode projected out of the variation the even residual falls
# from 2.93/2.23/2.07 to 0.0444/0.0510/0.167 but stays 53x/194x/508x above the
# odd sector.  data/I_even_sector.{json,npz} (notebook_even.py, N = 1600)
# decides between the three candidate causes:
#   (i)   physics -- the 1+1 massless scalar's IR zero mode;
#   (ii)  numerics -- the subtraction dS - d<K_loc> losing the answer;
#   (iii) the variation family.
# The tests below pin (i), refute (ii), and pin the honest negative result that
# the (L, m) scan cannot isolate the zero mode because a mass also destroys
# conformality.  The decisive cross-check is in 2+1 and lives in
# tests/test_toy_jacobson_2d.py.
# ===========================================================================

import json as _json
from pathlib import Path as _Path

_EVEN_DATA = _Path(__file__).resolve().parents[1] / "papers" / "toy-jacobson" / "data"


def _even_archive():
    p = _EVEN_DATA / "I_even_sector.json"
    if not p.exists():
        pytest.skip(f"{p} not present (run notebook_even.py)")
    return _json.load(open(p))


def test_the_even_sector_defect_is_the_balls_top_williamson_mode():
    """dS - d<K_loc> approaches the entropy contribution of the ball's top
    (reflection-even) Williamson mode -- the lattice's zero mode phi_bar -- AS
    THE VARIATION'S WAVELENGTH GROWS, reaching 14 % at lambda = 16L uniformly
    in L.  It is NOT one number over the family: at lambda = 4L the same ratio
    is negative at every L.  Fix round 2 pins all three wavelengths, because
    quoting only lambda = 16L made a convergence look like a uniform fact."""
    I = _even_archive()["I1_zero_mode_decomposition"]
    d = I["defect_over_zero_mode_longest_wavelength"]
    assert float(d["8"]) == pytest.approx(1.14431, rel=2e-3)
    assert float(d["16"]) == pytest.approx(1.13879, rel=2e-3)
    assert float(d["32"]) == pytest.approx(1.15753, rel=2e-3)
    assert all(1.10 < float(v) < 1.20 for v in d.values()), (
        f"at lambda = 16L the defect is the zero mode to 10-20 %: {d}")
    # the whole family, all three wavelengths (fix round 2)
    by = I["defect_over_zero_mode_by_ratio"]
    expect = {"8": (-0.75156, 1.98687, 1.14431),
              "16": (-1.28794, 1.59609, 1.13879),
              "32": (-3.73774, 1.49491, 1.15753)}
    for L, (r4, r8, r16) in expect.items():
        got = [float(by[L][k]) for k in ("4", "8", "16")]
        for g, e in zip(got, (r4, r8, r16)):
            assert g == pytest.approx(e, rel=2e-3), (L, got)
        assert got[0] < 0, (
            f"L={L}: at lambda = 4L the ratio is NEGATIVE, not ~1: {got}")
    # the honest statement: |ratio - 1| falls MONOTONICALLY with the wavelength
    # at every L -- a convergence in lambda, not a uniform property.
    dev = I["defect_over_zero_mode_dev_from_one_by_ratio"]
    for L in ("8", "16", "32"):
        seq = [float(dev[L][k]) for k in ("4", "8", "16")]
        assert seq[0] > seq[1] > seq[2], f"L={L}: |ratio-1| must fall with lambda: {seq}"
        assert seq[2] < 0.17 and seq[0] > 1.7 * seq[2], f"L={L}: {seq}"
    assert [round(float(dev[L]["4"]), 2) for L in ("8", "16", "32")] == [1.75, 2.29, 4.74]
    # the odd sector is exactly blind to the top mode (parity, proof Lemma 2.2)
    assert I["top_mode_share_odd_max"] < 1e-12
    # RE-PINNED in fix round 2: the old field "top_mode_share_even_min" was a
    # minimum over the lambda = 16L rows only and was pinned as if it were the
    # minimum over the even sector.  Both are now named and pinned.
    assert "top_mode_share_even_min" not in I, (
        "the mislabelled field must stay removed")
    assert I["top_mode_share_even_min_lambda16L"] == pytest.approx(1.00960, rel=1e-3)
    assert I["top_mode_share_even_min_all_ratios"] == pytest.approx(0.21179, rel=1e-3)
    assert I["top_mode_share_even_min_all_ratios"] > 0.2, (
        "the even sector always puts a finite share on the top mode, but at "
        "lambda = 4L, L = 32 that share is 0.21, not ~1")


def test_the_zero_mode_and_the_even_residual_grow_with_the_chain_length():
    """The zero-mode signature: at FIXED ball size the ball's top Williamson
    eigenvalue grows without bound with the IR cutoff N, and so does the even
    residual, while the odd residual does not.  A UV/discretisation defect
    could not do this."""
    ns = [r for r in _even_archive()["I1_zero_mode_decomposition"]["nu_top_vs_N"]
          if r["L"] == 8]
    ns.sort(key=lambda r: r["N"])
    assert [r["N"] for r in ns] == [200, 400, 800, 1600, 3200]
    nu = [r["nu_top"] for r in ns]
    assert nu[0] == pytest.approx(0.934080, rel=1e-4)
    assert nu[-1] == pytest.approx(1.199290, rel=1e-4)
    assert all(nu[i + 1] > nu[i] for i in range(len(nu) - 1)), nu
    # logarithmic, not saturating: the increment per doubling stays within 20 %
    inc = [nu[i + 1] - nu[i] for i in range(len(nu) - 1)]
    assert min(inc) > 0.8 * max(inc), f"nu_1 grows like ln N: increments {inc}"
    re = [r["r_even"] for r in ns]
    ro = [r["r_odd"] for r in ns]
    assert re[0] == pytest.approx(2.03884, rel=2e-3)
    assert re[-1] == pytest.approx(3.60072, rel=2e-3)
    assert re[-1] > 1.5 * re[0], f"the even residual grows with the IR cutoff: {re}"
    assert max(ro) < 2e-3 and ro[-1] < ro[0], f"the odd residual does not: {ro}"


def test_projecting_the_soft_even_modes_kills_the_absolute_defect_not_the_ratio():
    """Removing the ball's softest EVEN Williamson modes from the variation
    drives |dS - d<K_loc>| down by seven orders of magnitude while the RATIO
    |dS - dK|/|dS| saturates at the percent level, because those same modes
    carry all of the even sector's dS.

    FIX ROUND 2: the absolute defect is NOT comparable with the odd sector's --
    projecting shrinks the variation itself by up to 1e6, so a smaller |dS - dK|
    says nothing about the local form's accuracy.  The two assertions that made
    that comparison (got[2] < 3 * odd_abs, got[3] < 0.1 * odd_abs) are replaced
    by the scale-invariant pins in
    test_per_unit_variation_the_projected_even_sector_stays_worse_than_the_odd,
    which measure the opposite conclusion."""
    lad = _even_archive()["I2_projection_ladder"]
    expect_abs = {8: (1.3403e-02, 6.7213e-05, 2.8394e-07, 5.2903e-10),
                  16: (2.9216e-02, 2.2711e-04, 2.6457e-06, 1.5053e-08),
                  32: (6.2558e-02, 6.6989e-04, 2.1558e-05, 2.5672e-07)}
    for L, (a0, a1, a3, a5) in expect_abs.items():
        got = [lad[f"L{L}_n{n}"]["abs_defect_max"] for n in (0, 1, 3, 5)]
        for g, e in zip(got, (a0, a1, a3, a5)):
            assert g == pytest.approx(e, rel=5e-3), (L, got)
        assert got[0] / got[3] > 1e4, f"L={L}: absolute defect collapses {got}"
        # the collapse is accompanied by an equally large collapse of the
        # variation itself -- which is why it is not a statement about accuracy
        fro = [lad[f"L{L}_n{n}"]["dV_B_fro_max"] for n in (0, 1, 3, 5)]
        assert fro[0] / fro[3] > 1e3, (
            f"L={L}: projecting shrinks ||dV_B||_F by {fro[0] / fro[3]:.1e}, so the "
            f"absolute defect cannot be compared across sectors: {fro}")
        rs = [lad[f"L{L}_n{n}"]["r"] for n in (1, 3, 5)]
        assert all(1e-2 < v < 2e-1 for v in rs), f"L={L}: the ratio saturates: {rs}"
    # the numbers anchor 8 of the MANIFEST quotes, unchanged
    assert lad["L8_n1"]["r"] == pytest.approx(0.044420, rel=5e-3)
    assert lad["L16_n1"]["r"] == pytest.approx(0.051002, rel=5e-3)
    assert lad["L32_n1"]["r"] == pytest.approx(0.166757, rel=5e-3)


def test_per_unit_variation_the_projected_even_sector_stays_worse_than_the_odd():
    """FIX ROUND 2, replacing the retracted "better than the odd sector in
    absolute terms" comparison.

    |dS - d<K_loc>| is not scale invariant, and projecting the soft even modes
    shrinks ||dV_B||_F by up to 1e6, so the projected even sector's tiny
    absolute defect carried no information.  The comparable statistic is the
    defect PER UNIT VARIATION, |dS - d<K_loc>| / ||dV_B||_F.  Matched cell by
    cell (same L, same wavelength) the projected even sector is 1.9x-659x WORSE
    than the odd sector and is never better -- the same direction the relative
    residual already said (0.048 vs 8.4e-4 at L = 8)."""
    lad = _even_archive()["I2_projection_ladder"]
    summ = lad["summary_per_unit_variation"]
    assert summ["even_over_odd_matched_min"] == pytest.approx(1.89077, rel=5e-3)
    assert summ["even_over_odd_matched_max"] == pytest.approx(659.009, rel=5e-3)
    assert summ["even_over_odd_matched_min_unprojected"] == pytest.approx(96.6057, rel=5e-3)
    assert summ["even_over_odd_matched_max_unprojected"] == pytest.approx(369.382, rel=5e-3)
    assert summ["even_over_odd_matched_min"] > 1.0, (
        "per unit variation the projected even sector is NEVER better than the odd "
        "sector -- the retracted claim said the opposite")
    # cell by cell, at the top of the ladder (n = 5 and n = 6 coincide: mode 6 is odd)
    expect = {"8": (22.5163, 96.6780, 228.6138),
              "16": (1.89077, 330.541, 533.854),
              "32": (69.8860, 463.199, 659.009)}
    for L, exp in expect.items():
        by = lad[f"L{L}_n5"]["even_over_odd_per_dV_by_ratio"]
        got = [float(by[k]) for k in ("4", "8", "16")]
        for g, e in zip(got, exp):
            assert g == pytest.approx(e, rel=5e-3), (L, got)
        assert min(got) > 1.0, f"L={L}: {got}"
    # the odd sector's own defect per unit variation, for scale
    odd = [lad[f"L{L}_odd"]["defect_per_dV_max"] for L in (8, 16, 32)]
    for o, e in zip(odd, (6.5591e-04, 1.0235e-04, 6.1020e-05)):
        assert o == pytest.approx(e, rel=5e-3), odd
    # and the projected even sector's, which does NOT fall below it
    ev = [lad[f"L{L}_n5"]["defect_per_dV_max"] for L in (8, 16, 32)]
    for e_, x in zip(ev, (1.4995e-01, 5.4639e-02, 4.0213e-02)):
        assert e_ == pytest.approx(x, rel=5e-3), ev
    assert all(a > b for a, b in zip(ev, odd))


def test_the_even_sector_residual_is_not_a_precision_artefact():
    """Cause (ii), refuted four ways: an independent finite-difference route to
    the projected dS, the mpmath rung, the per-mode cancellation ratio, and the
    term-by-term assembly of dS - d<K_loc> that never forms the difference of
    two O(1) numbers."""
    p = _even_archive()["I3_precision_controls"]
    assert p["max_rel_gap_fd"] == pytest.approx(1.4501e-06, rel=5e-2)
    assert p["max_rel_gap_mp"] == pytest.approx(2.1010e-11, rel=5e-2)
    assert p["max_rel_gap_fd"] < 1e-4 and p["max_rel_gap_mp"] < 1e-8, (
        "the projected dS is right to far better than the 4e-2 residual it is "
        "being blamed for")
    assert p["max_cancellation"] == pytest.approx(1.0000140, rel=1e-5)
    assert p["max_cancellation"] < 1.01, (
        "the projected dS is NOT a near-cancellation among Williamson modes: the "
        "per-mode contributions all carry the same sign")
    # live: assemble dS - dK term by term in the ball's Williamson frame at a
    # smaller N and check it against the naive subtraction
    N, L = 400, 8
    K = coupling_chain_K(N, [1.0])
    eig = np.linalg.eigh(K)
    ball = centred_ball(N, L)
    sites, bpos = jac._halo(K, ball)
    V_B = jac._vacuum_block(eig, ball)
    G_loc = local_modular_form(K, ball, split="bond", sites=sites, beta_at="bond")
    S_w, nu = williamson(V_B)
    Si_w = symplectic_inverse(S_w)
    sel = np.concatenate([np.asarray(bpos, int), np.asarray(bpos, int) + len(sites)])
    keep = np.ones(2 * L)
    keep[0] = keep[L] = 0.0
    A = S_w.T @ G_loc[np.ix_(sel, sel)] @ S_w
    for mode in squeeze_modes(eig, L, ratios=(4, 8, 16), parity="even"):
        dV_B0 = squeeze_variation(mode["u"][ball], mode["omega"])
        W0 = Si_w @ dV_B0 @ Si_w.T
        dV_B = S_w @ (W0 * keep[:, None] * keep[None, :]) @ S_w.T
        dV_s = squeeze_variation(mode["u"][sites], mode["omega"])
        dV_s[np.ix_(sel, sel)] = dV_B
        dS, info = jac._mode_contributions(V_B, dV_B, 1e-10, 1e-3, 40)
        dK = 0.5 * float(np.trace(G_loc @ dV_s))
        W = Si_w @ dV_B @ Si_w.T
        dA, dW = np.diag(A), np.diag(W)
        per_mode = info["contributions"] - 0.5 * (dA[:L] * dW[:L] + dA[L:] * dW[L:])
        off = 0.5 * (float(np.sum(A * W.T)) - float(np.sum(dA * dW)))
        assert float(np.sum(per_mode) - off) == pytest.approx(dS - dK, rel=1e-8), (
            "the term-by-term assembly reproduces the subtraction exactly, so the "
            "subtraction is not where the residual comes from")
        assert np.log10(max(abs(dS), abs(dK)) / abs(dS - dK)) < 3.0, (
            "the subtraction costs fewer than three decimal digits of the sixteen "
            "float64 carries")


def test_the_mass_scan_cannot_isolate_the_zero_mode():
    """Honest negative result.  A mass gaps the zero mode (nu_1 -> 1/2) but also
    destroys conformality, so the CHM form is wrong in BOTH sectors at mL >> 1:
    r_odd rises from 4.6e-4 to 0.33 while r_even stays O(2).  The mass scan
    therefore decides nothing; the decisive test is 2+1
    (tests/test_toy_jacobson_2d.py)."""
    rows = _even_archive()["I4_mass_scan"]["rows"]
    massless = [r for r in rows if r["m"] == 0.0]
    heavy = [r for r in rows if r["m"] == 3.0]
    assert max(r["nu_top"] for r in massless) > 1.0
    assert max(r["nu_top"] for r in heavy) < 0.51, "a mass gaps the ball's top mode"
    assert max(r["r_odd"] for r in massless) < 2e-3
    assert min(r["r_odd"] for r in [r for r in rows if r["m"] == 1.0]) > 0.29, (
        "the odd sector is ALSO destroyed by the mass -- the scan is confounded")
    assert min(r["r_even"] for r in rows) > 1.6, (
        "r_even stays O(2) at every mass, including mL = 96")


def test_the_even_sector_archive_carries_build_information():
    import hashlib
    I = _even_archive()
    b = I["__build__"]
    for key in ("produced_utc", "git_head", "generator", "generator_sha256", "mode"):
        assert key in b
    assert b["generator"] == "papers/toy-jacobson/notebook_even.py"
    gen = _EVEN_DATA.parent / "notebook_even.py"
    assert b["generator_sha256"] == hashlib.sha256(gen.read_bytes()).hexdigest(), (
        "data/I_even_sector.* was produced by a different notebook_even.py")
    npz = np.load(_EVEN_DATA / "I_even_sector.npz")
    assert int(npz["N"]) == 1600
    assert npz["r_odd"].shape == (3,)
    assert float(npz["max_rel_gap_mp"]) < 1e-8
