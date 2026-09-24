"""Layer 7 / L5: the dense exact-diagonalization QEI infimum -- the second implementation.

``vacuum.interacting.qei_ed`` computes lambda_min(O_f) - <Omega|O_f|Omega> for the
truncated-boson chain with NO Trotter step and NO operator truncation: the full
eigendecomposition of H, in which <a|O_f|b> = h_ab F(w_a - w_b) with F the Fourier
transform of f^2.  It is the anomaly-protocol control on the MPO route of
``qei_exact`` (PLAN.md: a second implementation before a number is read as
physics), and it shares exactly one approximation with it -- the Fock cutoff --
so the two routes are comparable at the same (L, n_max).

Anchors:

1. **Construction.** The Hamiltonian assembled here from the model's definition
   equals the TeNPy model's bond terms, and its ground energy the DMRG one.
2. **Parity.** O_f is block-diagonal in the global phi -> -phi parity, symmetric,
   and the sector split reproduces the full-matrix operator.
3. **lambda = 0 vs the Gaussian core.** The ED infimum converges to the exact
   Williamson infimum exponentially in n_max -- from above and monotonically at
   one point (L = 3).  At another (L = 4, tau0 = 0.5) the same ladder sits BELOW
   the Gaussian infimum on some rungs and changes sign from rung to rung: the
   smeared infimum of the truncated chain is not a Rayleigh-Ritz quantity, and a
   non-monotone n_max ladder is not evidence of MPO trouble.
4. **Kernels.** The analytic Fourier kernel, Gauss-Legendre on the handle's
   support, and the MPO route's trapezoid grid agree to the tail/aliasing level.
5. **The two implementations agree.** At L = 5, n_max = 2 the ED infimum with the
   MPO route's own time grid equals the operator-TEBD + DMRG number to 1e-7.
6. **The O(lambda) anchor is right.** ``qei_exact.first_order_infimum_slope`` --
   the Hellmann-Feynman slope of the infimum in the free extremal state, closed
   by Wick's theorem -- equals the finite-difference slope of the ED infimum.
7. **The matrix-free route is the dense route.** ``qei_exact_lanczos`` (Chebyshev
   time evolution on the parity blocks + Lanczos; no dense matrix) reproduces
   the dense trapezoid-kernel infimum, both block minima and the reference to
   1e-11 relative at toy size, its Chebyshev step is exact to 1e-15, and the
   S8 archive (``data/s8_ed_anchor.json``) pins its L = 6 rungs against the
   archived dense rows and the MPO route.

Sizes keep the whole file near ~30 s.  TeNPy is importorskip'd where the MPO
route is needed; the ED module itself is numpy/scipy only.
"""

import json
import logging
import pathlib
import warnings

import numpy as np
import pytest

from vacuum.inequalities.qei import gaussian_f
from vacuum.interacting.phi4 import local_boson_ops, oscillator_frequency
from vacuum.interacting.qei_ed import (
    ChebyshevPropagator,
    dense_hamiltonian,
    dense_local_energy,
    parity_sectors,
    qei_exact_dense,
    qei_exact_lanczos,
    smeared_operator_dense,
    time_kernel,
)
from vacuum.interacting.qei_exact import (
    first_order_infimum_slope,
    free_exact_infimum,
    local_energy_infimum_dense,
)

MASS = 0.5


def _vi():
    pytest.importorskip("tenpy")
    logging.getLogger("tenpy").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="unit_cell_width")
    import vacuum.interacting as vi
    return vi


def _dense_from_model(model):
    """Dense H from the TeNPy model's bond terms (the operator the TEBD applies)."""
    L = model.lat.N_sites
    d = model.lat.mps_sites()[0].dim
    H = np.zeros((d**L, d**L))
    for i in range(1, L):
        Hb = model.H_bond[i].copy()
        Hb.itranspose(["p0", "p1", "p0*", "p1*"])
        arr = np.real(Hb.to_ndarray()).reshape(d * d, d * d)
        H += np.kron(np.kron(np.eye(d ** (i - 1)), arr), np.eye(d ** (L - i - 1)))
    return 0.5 * (H + H.T)


# --------------------------------------------------------------------------
# 1. Construction: the same truncated Hamiltonian as the TeNPy model
# --------------------------------------------------------------------------


def test_dense_hamiltonian_matches_the_tenpy_model():
    vi = _vi()
    L, n_max, lam = 4, 3, 1.0
    gs = vi.phi4_ground_mps(L, MASS, lam, n_max=n_max, chi_max=32)
    H_model = _dense_from_model(gs.model)
    H_ed = dense_hamiltonian(L, MASS, lam, n_max, gs.info["omega"]).toarray()
    assert np.max(np.abs(H_ed - H_model)) < 1e-12, "the two assemblies of H differ"
    E0 = np.linalg.eigvalsh(H_ed)[0]
    assert abs(E0 - gs.E0) < 1e-8, f"ED ground energy {E0} != DMRG {gs.E0}"
    # the local energy densities sum to H (the bond split of vacuum.inequalities.qei)
    h_sum = sum(dense_local_energy(L, MASS, lam, n_max, gs.info["omega"], x).toarray() for x in range(L))
    assert np.max(np.abs(h_sum - H_ed)) < 1e-12


# --------------------------------------------------------------------------
# 2. Parity: block diagonal, symmetric, and the sectors reproduce the whole
# --------------------------------------------------------------------------


def test_parity_blocks_reproduce_the_full_operator():
    L, n_max, lam = 4, 2, 1.0
    x = L // 2
    f = gaussian_f(0.3, n_sigmas=4.0)
    O, H, (even, odd) = smeared_operator_dense(L, MASS, lam, n_max, x, f)
    d = n_max + 1
    assert even.size + odd.size == d**L
    assert np.max(np.abs(O - O.T)) < 1e-12
    assert np.max(np.abs(O[np.ix_(even, odd)])) < 1e-12, "O_f must not couple the parity sectors"
    # the parity operator commutes with H and O_f
    P = np.ones(d**L)
    P[odd] = -1.0
    assert np.max(np.abs(P[:, None] * H - H * P[None, :])) < 1e-12
    assert np.max(np.abs(P[:, None] * O - O * P[None, :])) < 1e-12
    # the sector route reports the same infimum as the full matrix
    res = qei_exact_dense(L, MASS, lam, n_max, x, f)
    w, V = np.linalg.eigh(H)
    Om = V[:, 0]
    E_full = np.linalg.eigvalsh(O)[0] - Om @ O @ Om
    assert abs(res.E_min - E_full) < 1e-10
    assert abs(res.E_star - min(res.E_star_by_sector.values())) < 1e-14
    assert set(res.E_star_by_sector) == {0, 1}
    # <Omega|O_f|Omega> = ||f||^2 <Omega|h|Omega> for an eigenstate of H
    assert abs(res.E_ref - res.meta["f2_norm"] * res.h_vac) < 1e-12


# --------------------------------------------------------------------------
# 3. lambda = 0: monotone, exponential convergence to the Gaussian infimum
# --------------------------------------------------------------------------


def test_free_fock_ladder_converges_monotonically_to_the_gaussian_infimum():
    """No MPO in the way: what the Fock cutoff alone does to the infimum, at a
    point where it happens to be one-signed and monotone (L = 3, m = 1, tau0 = 0.4).

    Positive, shrinking exponentially, non-increasing.  This is a MEASUREMENT at
    this point, not a theorem: e^{-iH_trunc t} is not the projection of e^{-iHt},
    so the smeared infimum of the truncated chain is not a Rayleigh-Ritz quantity
    -- the next test exhibits a point where the same ladder changes sign.
    """
    L, m, tau0 = 3, 1.0, 0.4
    x = L // 2
    f = gaussian_f(tau0, n_sigmas=4.0)
    E_gauss = free_exact_infimum(L, x, f, m)
    devs = []
    for n_max in (2, 4, 6, 8, 10):
        r = qei_exact_dense(L, m, 0.0, n_max, x, f)
        devs.append(r.E_min - E_gauss)
    print(f"\nlambda = 0 ED ladder (L = {L}, m = {m}, tau0 = {tau0}): Gaussian {E_gauss:.8e}; "
          "devs " + " ".join(f"{d:+.2e}" for d in devs))
    assert all(d > 0.0 for d in devs), f"the Fock cutoff must raise the infimum: {devs}"
    assert all(a > b for a, b in zip(devs, devs[1:])), f"not monotone: {devs}"
    rates = [a / b for a, b in zip(devs, devs[1:])]
    assert min(rates) > 3.0, f"convergence is not exponential enough: {rates}"
    assert devs[-1] < 2e-4 * abs(E_gauss)


def test_fock_ladder_of_the_smeared_infimum_is_not_rayleigh_ritz():
    """The lambda = 0 Fock ladder of the SMEARED infimum can sit below the exact
    Gaussian infimum and change sign from rung to rung -- with no MPO anywhere.

    Only the t = 0 operator h_x is a Rayleigh-Ritz projection (its lambda_min is
    monotone in n_max, ``test_interacting_qei_exact``); the smeared operator of
    the truncated chain is int f^2 e^{iH_trunc t} h_trunc e^{-iH_trunc t} with
    H_trunc = P H P, a different dynamical system whose infimum is not bounded by
    the exact one on either side.  Measured at L = 4, m = 1, tau0 = 0.5 (relative
    deviation from the Gaussian infimum): n_max = 4: +1.9e-3, 5: -2.0e-3,
    6: -6.6e-3, 7: +5.9e-4, 8: -1.7e-3, 9: +9.7e-5 -- converging, sign-alternating,
    non-monotone, and the extremal state's parity sector flips along the way.
    Consequence for the paper: a non-monotone lambda = 0 control of the MPO route
    is NOT by itself evidence of MPO-truncation trouble, and a plateau or a
    reversal in an n_max ladder of the infimum cannot be read either way without
    the operator-truncation axis held converged.
    """
    L, m, tau0 = 4, 1.0, 0.5
    x = L // 2
    f = gaussian_f(tau0, n_sigmas=4.0)
    E_gauss = free_exact_infimum(L, x, f, m)
    devs = {}
    sectors = {}
    for n_max in (4, 5, 6, 7):
        r = qei_exact_dense(L, m, 0.0, n_max, x, f)
        devs[n_max] = (r.E_min - E_gauss) / abs(E_gauss)
        sectors[n_max] = r.sector_star
    print(f"\nlambda = 0 ED ladder (L = {L}, m = {m}, tau0 = {tau0}): rel devs {devs}, sectors {sectors}")
    assert devs[4] > 0.0 and devs[5] < 0.0 and devs[6] < 0.0 and devs[7] > 0.0, (
        f"expected the sign-alternating ladder measured at this point, got {devs}")
    assert abs(devs[6]) > abs(devs[5]), "the ladder is not monotone in |dev| either"
    assert abs(devs[7]) < 1e-3 and abs(devs[6]) < 1e-2, "it must still converge"
    assert len(set(sectors.values())) == 2, "the extremal parity sector flips along the ladder"


# --------------------------------------------------------------------------
# 4. Kernels: analytic, Gauss-Legendre, and the MPO route's trapezoid grid
# --------------------------------------------------------------------------


def test_time_kernels_agree():
    L, n_max, lam, tau0 = 3, 6, 1.0, 0.4
    x = L // 2
    f = gaussian_f(tau0, n_sigmas=4.0)
    r = qei_exact_dense(L, MASS, lam, n_max, x, f, kernel=("analytic", "gauss", "trapezoid"), dt=0.2)
    by = r.meta["by_kernel"]
    ea, eg, et = by["analytic"]["E_min"], by["gauss"]["E_min"], by["trapezoid"]["E_min"]
    print(f"\nkernels: analytic {ea:.10e} gauss {eg:.10e} trapezoid(dt=0.2) {et:.10e}")
    # the support is +-4 tau0: the dropped tail of f^2 is e^{-16} ~ 1e-7 relative
    assert abs(eg - ea) < 1e-6 * abs(ea)
    assert abs(et - ea) < 1e-6 * abs(ea)
    F, _ = time_kernel(f, "analytic")
    assert abs(F(np.zeros(1))[0] - tau0 * np.sqrt(np.pi)) < 1e-14


# --------------------------------------------------------------------------
# 5. The two implementations agree (anomaly-protocol control at toy size)
# --------------------------------------------------------------------------


def test_dense_infimum_matches_the_mpo_route():
    """Same (L, n_max, f, dt): the ED infimum on the MPO route's own time grid
    equals the operator-TEBD + DMRG number to the Trotter error of the latter,
    which at dt = 0.1 and order 4 is < 1e-7."""
    vi = _vi()
    import vacuum.interacting.qei_exact as qx
    L, n_max, lam, tau0 = 5, 2, 1.0, 0.3
    x = L // 2
    f = gaussian_f(tau0, n_sigmas=4.0)
    gs = vi.phi4_ground_mps(L, MASS, lam, n_max=n_max, chi_max=32)
    ed = qei_exact_dense(L, MASS, lam, n_max, x, f, kernel=("trapezoid", "analytic"), dt=0.1)
    res = qx.qei_exact_mps(gs, x, f, dt=0.1, order=4, chi_op=48, svd_min_op=1e-13,
                           chi_acc=96, svd_min_acc=1e-13, window_pad=None,
                           svd_min_real=1e-13, weight_decay=1.0, chi_dmrg=32)
    dev = res.E_min - ed.E_min
    dev_an = res.E_min - ed.meta["by_kernel"]["analytic"]["E_min"]
    print(f"\nMPO {res.E_min:.10e} vs ED (same grid) {ed.E_min:.10e}: dev {dev:+.2e}; "
          f"vs ED (analytic) dev {dev_an:+.2e}; ED omega {ed.meta['omega']:.6f} = model {gs.info['omega']:.6f}")
    assert abs(ed.meta["omega"] - gs.info["omega"]) < 1e-12, "the two routes must share the basis"
    assert abs(dev) < 1e-7, f"the two implementations disagree by {dev:.2e}"
    assert abs(dev_an) < 1e-5 * abs(ed.E_min)
    assert abs(res.E_star - ed.E_star) < 1e-7 and abs(res.E_ref - ed.E_ref) < 1e-6


# --------------------------------------------------------------------------
# 6. The O(lambda) anchor: Hellmann-Feynman + Wick vs finite differences
# --------------------------------------------------------------------------


def test_first_order_slope_matches_finite_differences():
    """d E_min / d lambda at 0 from the free extremal state (qei_exact) equals the
    central difference of the ED infimum; the Fock cutoff at n_max = 8, m = 1 is
    far below the tolerance (the lambda = 0 point itself sits < 1e-4 off the
    Gaussian infimum)."""
    L, m, tau0, n_max = 4, 1.0, 0.5, 9
    x = L // 2
    f = gaussian_f(tau0, n_sigmas=4.0)
    sl = first_order_infimum_slope(L, x, f, m)
    eps = 0.02
    e_p = qei_exact_dense(L, m, +eps, n_max, x, f).E_min
    e_m = qei_exact_dense(L, m, -eps, n_max, x, f).E_min
    e_0 = qei_exact_dense(L, m, 0.0, n_max, x, f).E_min
    fd = (e_p - e_m) / (2.0 * eps)
    print(f"\nslope: formula {sl['dE_min']:+.8e} (quartic {sl['dE_star_quartic']:+.3e}, dynamic "
          f"{sl['dE_star_dynamic']:+.3e}, ref {sl['dE_ref']:+.3e}); FD {fd:+.8e}; "
          f"E_min(0) ED {e_0:.8e} vs Gaussian {sl['E_free']:.8e}; dR/dlam {sl['dR']:+.4e}")
    # n_max = 9: the lambda = 0 point sits 1e-4 off the Gaussian infimum and the
    # FD slope 4e-4 off the formula (measured); n_max = 8 is a rung where the
    # truncated chain sits BELOW the Gaussian value (see the ladder test below)
    # and its slope is 2e-2 off -- the Fock cutoff's error is not smooth in n_max
    assert abs(e_0 - sl["E_free"]) < 5e-4 * abs(sl["E_free"])
    assert abs(fd - sl["dE_min"]) < 3e-3 * abs(sl["dE_min"]), (
        f"the O(lambda) slope {sl['dE_min']} disagrees with finite differences {fd}")
    # structure: the vacuum's energy density rises with lambda, and so does the
    # extremal eigenvalue; the residual slope is the difference of two O(1) numbers
    assert sl["dE_ref"] > 0.0 and sl["dE_star"] > 0.0
    assert abs(sl["extremal_gap"]) < 1e-8


# --------------------------------------------------------------------------
# 7. The matrix-free route (Chebyshev + Lanczos) is the dense route
# --------------------------------------------------------------------------


def test_chebyshev_propagator_is_the_exact_short_time_evolution():
    """One Chebyshev step equals scipy's expm on the same sparse H to 1e-14, in both
    directions, with the Bessel truncation the module chooses."""
    import scipy.linalg as sla
    L, n_max, lam = 3, 3, 1.0
    om = oscillator_frequency(L, MASS)
    H = dense_hamiltonian(L, MASS, lam, n_max, om)
    w = np.linalg.eigvalsh(H.toarray())
    v = np.random.default_rng(1).standard_normal(H.shape[0])
    v /= np.linalg.norm(v)
    pad = 0.02 * (w[-1] - w[0])
    for tau in (0.375, -0.375, 0.125):
        P = ChebyshevPropagator(H, w[0] - pad, w[-1] + pad, tau)
        exact = sla.expm(-1j * tau * H.toarray()) @ v
        err = np.linalg.norm(P(v) - exact)
        assert err < 1e-14, f"tau={tau}: Chebyshev step off by {err:.2e} with {P.n_terms} terms"
        assert abs(np.linalg.norm(P(v)) - 1.0) < 1e-14, "the step must be unitary"
        assert P.n_terms < 40, f"the Bessel truncation kept {P.n_terms} terms for a tau = {tau}"


@pytest.mark.parametrize("L,n_max,lam,tau0,dt", [(4, 3, 1.0, 0.3, 0.1), (5, 2, 4.0, 0.75, 0.375)])
def test_lanczos_route_reproduces_the_dense_route_on_the_same_grid(L, n_max, lam, tau0, dt):
    """Same (L, n_max, lambda, f, dt): the matrix-free infimum equals the dense
    trapezoid-kernel one to 1e-11 relative -- both block minima, the reference and
    the winning sector -- and its finer grid lands within the dropped-tail level
    (e^{-16} ~ 1e-7) of the analytic kernel.  Measured: 7e-15 and 8e-14 relative at
    these two points (2026-09-11)."""
    x = L // 2
    f = gaussian_f(tau0, n_sigmas=4.0)
    ed = qei_exact_dense(L, MASS, lam, n_max, x, f, kernel=("trapezoid", "analytic"), dt=dt)
    lz = qei_exact_lanczos(L, MASS, lam, n_max, x, f, dt=dt, dt_fine=dt / 3.0)
    by = ed.meta["by_kernel"]
    rel = (lz.E_min - ed.E_min) / abs(ed.E_min)
    print(f"\nL={L} n_max={n_max} lam={lam}: dense {ed.E_min:.14e} lanczos {lz.E_min:.14e} rel dev {rel:+.2e}; "
          f"fine {lz.meta['fine']['E_min']:.12e} vs analytic {by['analytic']['E_min']:.12e}; "
          f"matvecs {lz.meta['matvecs']}, n_cheb {lz.meta['n_cheb']}")
    assert abs(rel) < 1e-11, f"the two implementations of the same grid differ by {rel:.2e}"
    assert abs(lz.E_ref - ed.E_ref) < 1e-13
    assert lz.sector_star == by["trapezoid"]["sector_star"]
    for p in (0, 1):
        assert abs(lz.E_star_by_sector[p] - by["trapezoid"]["E_star_by_sector"][p]) < 1e-11 * abs(ed.E_star)
    assert abs(lz.meta["dt"] - by["trapezoid"]["dt"]) < 1e-14 and lz.meta["n_steps"] == by["trapezoid"]["n_steps"]
    rel_fine = (lz.meta["fine"]["E_min"] - by["analytic"]["E_min"]) / abs(ed.E_min)
    assert abs(rel_fine) < 5e-6, f"the fine grid sits {rel_fine:.2e} from the analytic kernel"
    assert abs(rel_fine) < abs((ed.E_min - by["analytic"]["E_min"]) / abs(ed.E_min)) + 1e-9, \
        "refining the grid must not move away from the analytic kernel"
    assert max(lz.meta["lanczos_residuals"].values()) < 1e-11
    assert lz.meta["dim"] == (n_max + 1) ** L and sum(lz.meta["dim_sectors"]) == lz.meta["dim"]


# --------------------------------------------------------------------------
# 8. The S8 ED anchor archive: the rungs beyond n_max 4 and the MPO-vs-ED pairs
# --------------------------------------------------------------------------

_S8 = (pathlib.Path(__file__).resolve().parents[1] / "papers" / "interacting-vacua-first-numbers"
       / "data" / "s8_ed_anchor.json")

#: (L, n_max, lam) -> E_min on the MPO route's grid (dt 0.375), matrix-free route, 2026-09-11.
#: Tolerance 1e-9 relative: the route reproduces the dense one to 5e-12 and a rerun at the
#: same ARPACK tolerance moves the value by < 1e-11; a wrong sector or an unconverged
#: Lanczos moves it by >= 1e-4 (the L = 5 odd-block miss was 4.6e-4).
_S8_PINS = {
    # L = 6 (the four rungs validated against the archived dense rows: 1.6e-13 to 5.6e-12)
    (6, 3, 0.0): (-0.018331855972679945, 1e-9),
    (6, 3, 4.0): (-0.0034496771785621494, 1e-9),
    (6, 4, 0.0): (-0.020239658919991244, 1e-9),
    (6, 4, 4.0): (-0.0059720244591821015, 1e-9),
    # the rungs beyond the dense route (fix round 1: quoted in the MANIFEST, now pinned)
    (6, 5, 0.0): (-0.020019521042360378, 1e-9),
    (6, 5, 4.0): (-0.006148967937020577, 1e-9),
    (5, 6, 0.0): (-0.018475413248251527, 1e-9),
    (5, 7, 0.0): (-0.01844541052979043, 1e-9),
    (5, 8, 0.0): (-0.018462671543312714, 1e-9),
    (5, 7, 4.0): (-0.015554288974701347, 1e-9),
}

# Two rungs are pinned separately: their fine-grid quadrature check lands further than 1e-4
# from the dt 0.375 grid, so they do not satisfy the quadrature assertion of
# test_s8_lanczos_rungs_are_pinned.  Both are L = 5, lambda = 4 -- the family whose E_min is
# still moving fast with the cutoff.
# UNIT 2 OF THE S8 TRIAGE, 2026-09-13: a THIRD coarse-quadrature rung landed from the ED
# pool -- (L 6, n_max 6, lambda 4), rel_quadrature_fine_minus_grid 8.3906e-04 -- which made
# the "every other rung is within 1e-4" assertion below false on a file nobody had edited.
# It is pinned here with its measured deviation (the pin is ADDED, the assertion is not
# weakened), and the reason is recorded in the candidate's MANIFEST.  The pattern is now
# explicit: every coarse rung is a lambda = 4 rung at the largest n_max reached for its L.
_S8_PIN_COARSE_QUADRATURE = {(5, 6, 4.0): (-0.009439313955727169, 1e-9, 1.3171005e-03),
                             (5, 8, 4.0): (-0.021561126716973744, 1e-9, 4.9612016e-04),
                             (6, 6, 4.0): (-0.008353380718243208, 1e-9, 8.3906021e-04)}


def _s8():
    if not _S8.exists():
        pytest.skip("the S8 ED anchor archive is not on this tree")
    with open(_S8) as fh:
        return json.load(fh)


def test_s8_lanczos_rungs_reproduce_the_archived_dense_rows():
    """Every S8 matrix-free rung with an archived dense row at the same (L, n_max, lambda)
    -- L = 6, n_max 3 and 4, both lambdas, and n_max 5 once the dense rows exist --
    agrees with it on the same trapezoid grid to 1e-10 relative (measured 4e-13 to 5e-12)."""
    rec = _s8()
    val = rec["validation_vs_dense"]
    assert len(val) >= 4, f"expected at least the four L = 6 n_max 3-4 validation rows, got {len(val)}"
    for v in val:
        assert abs(v["rel_dev_same_grid"]) < 1e-10, f"{v['L']}/{v['n_max']}/{v['lam']}: {v['rel_dev_same_grid']:.2e}"
        assert v["sector_star_lanczos"] == v["sector_star_dense"]
        assert abs(v["rel_dev_fine_vs_analytic"]) < 1e-5, "the fine grid must land on the analytic kernel"
    assert rec["worst_abs_rel_dev_validation"] < 1e-10


def test_s8_lanczos_rungs_are_pinned():
    rec = _s8()
    rows = {(r["L"], r["n_max"], r["lam"]): r for r in rec["ed_lanczos_rows"]}
    for key, (e_min, tol) in _S8_PINS.items():
        assert key in rows, f"rung {key} missing from the archive"
        r = rows[key]
        assert abs(r["E_min"] - e_min) <= tol * abs(e_min), f"{key}: {r['E_min']!r} vs pinned {e_min!r}"
        assert max(r["lanczos_residuals"].values()) < 1e-9
        assert abs(r["rel_quadrature_fine_minus_grid"]) < 1e-4


# --------------------------------------------------------------------------
# The tiny-tau limit, on the ED route (no dynamics to get wrong)
# --------------------------------------------------------------------------


def test_s8_L6_lambda4_ed_ladder_is_still_rising_at_nmax_5():
    """FIX ROUND 1 (item 4).  The exact L = 6 ladder of exact/E_H at lambda = 4 reads
    0.4931 / 0.8536 / 0.8789 at n_max 3 / 4 / 5: monotone and still rising by +0.0253 at the
    top rung, while the lambda = 0 ladder has already turned over (1.0173 -> 1.0062, -0.0111).
    The exact lambda = 4 answer is therefore not converged in n_max at the largest cutoff any
    exact route has reached -- one of the reasons item 4 stays open."""
    rec = _s8()
    lad = rec["ed_ladders"].get("L=6/lam=4")
    if lad is None or 5 not in lad["n_max"]:
        pytest.skip("the L = 6 lambda = 4 ladder does not reach n_max 5")
    by = dict(zip(lad["n_max"], lad["ratio_hartree"]))
    assert abs(by[3] - 0.4930595) < 1e-6 and abs(by[4] - 0.8535768) < 1e-6 and abs(by[5] - 0.8788669) < 1e-6
    assert by[3] < by[4] < by[5], "the lambda = 4 ladder is monotone up to n_max 5"
    lad0 = rec["ed_ladders"].get("L=6/lam=0")
    by0 = dict(zip(lad0["n_max"], lad0["ratio_hartree"]))
    assert by0[5] < by0[4], "the lambda = 0 ladder has turned over by n_max 5"
    assert (by[5] - by[4]) > 0.02 > abs(by0[5] - by0[4])


def test_s8_L5_ladders_converge_at_lambda_0_and_do_not_at_lambda_4():
    """FIX ROUND 1 (item 4).  At L = 5 the EXACT lambda = 0 ratio converges on 1 --
    1.000739 / 0.999114 / 1.000049 at n_max 6 / 7 / 8, i.e. 4.9e-5 from 1 at the top rung --
    while at the same L and the same grid the exact lambda = 4 infimum falls 64.8 % between
    n_max 6 and 7 (E_min -9.4393e-03 -> -1.5554e-02, ratio 1.4292 -> 2.3551), with Lanczos
    residuals of 5e-11 and 8e-11: both rungs are exact eigenvalues, the LADDER is what is not
    converged.  So at lambda = 4 no cutoff reached by any route -- exact or MPO -- resolves
    the magnitude, which is why item 4 is open and why no lambda >= 2 number is quoted."""
    rec = _s8()
    lads = rec["ed_ladders"]
    l0 = lads.get("L=5/lam=0")
    if l0 is None or 8 not in l0["n_max"]:
        pytest.skip("the L = 5 lambda = 0 ladder does not reach n_max 8")
    by0 = dict(zip(l0["n_max"], l0["ratio_hartree"]))
    assert abs(by0[6] - 1.0007389) < 1e-6 and abs(by0[7] - 0.9991138) < 1e-6 and abs(by0[8] - 1.0000487) < 1e-6
    assert abs(by0[8] - 1.0) < 1e-4, "the exact lambda = 0 control converges on 1"
    assert abs(by0[8] - 1.0) < abs(by0[6] - 1.0) and abs(by0[8] - 1.0) < abs(by0[7] - 1.0)
    l4 = lads.get("L=5/lam=4")
    if l4 is None or 7 not in l4["n_max"]:
        pytest.skip("the L = 5 lambda = 4 ladder does not reach n_max 7")
    e4 = dict(zip(l4["n_max"], l4["E_min_same_grid"]))
    r4 = dict(zip(l4["n_max"], l4["ratio_hartree"]))
    assert e4[7] < e4[6] < 0.0, "Rayleigh-Ritz: a larger Fock space can only lower the infimum"
    drop = (e4[7] - e4[6]) / abs(e4[6])
    assert 0.60 < abs(drop) < 0.70, f"the n_max 6 -> 7 drop at lambda = 4 is {drop:.3f}"
    assert abs(r4[6] - 1.4292273) < 1e-6 and abs(r4[7] - 2.3551092) < 1e-6
    if 8 in e4:
        # the n_max 8 rung (landed 13:49 PDT): still falling, 38.6 % in the next rung
        assert abs(e4[8] + 0.021561127) < 1e-8 and abs(r4[8] - 3.2646177) < 1e-6
        assert e4[8] < e4[7]
        drop78 = (e4[8] - e4[7]) / abs(e4[7])
        assert 0.35 < abs(drop78) < 0.42, f"the n_max 7 -> 8 drop at lambda = 4 is {drop78:.3f}"
        assert abs(drop78) < abs(drop), "the fall is slowing but has not stopped"
        assert r4[8] > 3.0, "exact/E_H at L = 5, lambda = 4 is 3.3 and still growing with the cutoff"
    # and the lambda = 0 ladder at the same rungs moves by < 0.2 %
    assert abs((by0[7] - by0[6]) / by0[6]) < 2e-3 < abs(drop)


def test_s8_lambda4_L5_nmax6_rung_is_pinned_with_its_coarse_quadrature():
    """FIX ROUND 1.  The L = 5, lambda = 4 rungs at n_max 6 and 8 are exact (Lanczos residuals
    4.8e-11 and 3.6e-11) but their trapezoid grid is not converged: re-solving on the dt 0.125
    grid moves E_min by +1.32e-3 and +4.96e-4 relative, against <= 8.7e-5 for every other rung.
    They are pinned here with those deviations, and any MPO-vs-ED comparison at those points
    inherits them."""
    rec = _s8()
    rows = {(r["L"], r["n_max"], r["lam"]): r for r in rec["ed_lanczos_rows"]}
    for key, (e_min, tol, quad) in _S8_PIN_COARSE_QUADRATURE.items():
        if key not in rows:
            pytest.skip(f"rung {key} has not landed")
        r = rows[key]
        assert abs(r["E_min"] - e_min) <= tol * abs(e_min), f"{key}: {r['E_min']!r}"
        assert max(r["lanczos_residuals"].values()) < 1e-9
        assert abs(r["rel_quadrature_fine_minus_grid"] - quad) < 1e-6
    # and they really are the outliers: every rung outside this dict is within 1e-4
    others = [abs(q["rel_quadrature_fine_minus_grid"]) for k, q in rows.items()
              if k not in _S8_PIN_COARSE_QUADRATURE]
    assert others and max(others) < 1e-4
    for key in _S8_PIN_COARSE_QUADRATURE:
        if key in rows:
            assert abs(rows[key]["rel_quadrature_fine_minus_grid"]) > 1e-4


def test_tiny_tau_limit_is_the_local_energy_density_infimum():
    L, n_max, lam = 4, 3, 1.0
    x = L // 2
    omega = oscillator_frequency(L, MASS)
    assert abs(local_boson_ops(n_max, omega)["X"][0, 1]) > 0.0
    lam_min = local_energy_infimum_dense(n_max, omega, MASS, lam, x, L)
    devs = []
    for tau0 in (0.2, 0.1, 0.05):
        f = gaussian_f(tau0, n_sigmas=4.0)
        r = qei_exact_dense(L, MASS, lam, n_max, x, f)
        density = r.E_min / r.meta["f2_norm"]
        devs.append(density - (lam_min - r.h_vac))
    assert all(d > 0.0 for d in devs), "smearing can only raise the pointwise infimum"
    assert devs[0] > devs[1] > devs[2]
    assert 2.0 < devs[0] / devs[1] < 8.0, "the approach should be ~second order in tau0"
