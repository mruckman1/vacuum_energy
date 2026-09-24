"""M3.4 anchors — the QEI stress-tester core (vacuum.inequalities.qei).

Spec: docs/PLAN_LAYERS_2_3.md, section M3.4. Anchors covered here:

(i)   Ford-Roman-type 2d bound satisfied by every engineered squeezed-pocket
      state, and the unphysical-V canary (nu < 1/2) caught by the raw
      symplectic-margin gate (no entropy clamps);
(ii)  the 2d massless sharp constant: qei_ratio -> 1 under Richardson
      extrapolation in (a/tau0, m*tau0), targeting Flanagan's optimal bound
      (PRD 56, 4922 (1997), Eq. (8)); within 1 percent extrapolated;
(iii) support-independence of the mode-support regularization.

The heavy 4d Srednicki scan stays OUT of this suite per spec M3.4; only the
machinery (radial T_00 assembly from low-l partial waves, ball smearing) is
smoke-tested at tiny sizes.
"""

import math

import numpy as np
import pytest

from vacuum.core import (
    ground_state_cov,
    harmonic_chain_K,
    reduce as reduce_cov,
    symplectic_eigenvalues,
)
from vacuum.core.gaussian import entropy_from_nu
from vacuum.core.radial import srednicki_K
from vacuum.inequalities import (
    FEWSTER_EVESON_CONSTANT_2D,
    SHARP_CONSTANT_2D,
    QEIUnphysicalError,
    chain_energy_density_form,
    cos2_f,
    embed_extremal,
    fprime_sq_integral,
    gaussian_f,
    gaussian_family,
    min_symplectic_margin,
    optimize_sampling,
    qei_audit,
    qei_bound_2d,
    qei_minimize,
    qei_ratio,
    radial_ball_qei,
    radial_ball_weights,
    radial_energy_density_form,
    sampling_operator,
    smeared_energy,
    squeezed_pocket,
)
from vacuum.inequalities.qei import _gauss_legendre_grid, _symplectic_spectrum_psd


def _chain_run(tau0, mu=0.0, g=80, **kw):
    """One (tau0, m*tau0) point: ratio and full results at box ratio g."""
    m = mu / tau0
    N = int(g * tau0)
    if N % 2 == 0:
        N += 1
    site = N // 2
    f = gaussian_f(tau0)
    op = sampling_operator(N, site, f, m=m, bc="dirichlet", **kw)
    res = qei_minimize(op)
    return res.e_min / qei_bound_2d(f, constant="sharp"), res, op, f


# --------------------------------------------------------------------------
# module-scoped fixtures: the Richardson table and a shared mid-size setup
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def richardson_table():
    """ratio(tau0) at m = 0, box ratio g = 80 — the a/tau0 axis."""
    taus = np.array([6.0, 8.0, 10.0, 13.0, 16.0])
    ratios = np.array([_chain_run(t)[0] for t in taus])
    return taus, ratios


@pytest.fixture(scope="module")
def battery_setup():
    """Shared tau0 = 5 chain (N = 301, m = 0) with operator and minimum."""
    tau0, N = 5.0, 301
    site = N // 2
    f = gaussian_f(tau0)
    K = harmonic_chain_K(N, 0.0, "dirichlet")
    op = sampling_operator(N, site, f, m=0.0, bc="dirichlet")
    res = qei_minimize(op)
    return {"tau0": tau0, "N": N, "site": site, "f": f, "K": K, "op": op,
            "res": res, "V_vac": ground_state_cov(K)}


# --------------------------------------------------------------------------
# construction: PSD local energy density, partition of energy, bounds
# --------------------------------------------------------------------------


def test_density_form_partitions_hamiltonian():
    """sum_i h_i = diag(K, I) exactly, and each h_i is PSD (both bcs)."""
    N, m = 12, 0.3
    for bc in ("dirichlet", "periodic"):
        K = harmonic_chain_K(N, m, bc)
        ref = np.zeros((2 * N, 2 * N))
        ref[:N, :N] = K
        ref[N:, N:] = np.eye(N)
        total = np.zeros_like(ref)
        for i in range(N):
            h = chain_energy_density_form(N, i, m=m, bc=bc)
            ev_min = np.min(np.linalg.eigvalsh(h))
            assert ev_min >= -1e-12, f"h_{i} ({bc}) not PSD: min eig {ev_min:.2e}"
            total += h
        dev = np.max(np.abs(total - ref))
        assert dev < 1e-12, f"sum_i h_i != diag(K, I) for {bc}: max dev {dev:.2e}"


def test_radial_density_form_partitions_hamiltonian():
    """Radial: w == 1 reproduces diag(srednicki_K, I) exactly (l = 0, 1, 4)."""
    N = 30
    for l in (0, 1, 4):
        h = radial_energy_density_form(N, l, np.ones(N))
        ref = np.zeros((2 * N, 2 * N))
        ref[:N, :N] = srednicki_K(N, l)
        ref[N:, N:] = np.eye(N)
        dev = np.max(np.abs(h - ref))
        assert dev < 1e-12, f"radial density (l={l}) partition dev {dev:.2e}"
        ev_min = np.min(np.linalg.eigvalsh(h))
        assert ev_min >= -1e-12, f"radial h (l={l}) not PSD: {ev_min:.2e}"


def test_bound_constants_and_integrals():
    """Flanagan sharp = 1/(6 pi), FE = 1/(4 pi) (ratio exactly 3/2); analytic
    int f'^2 values match quadrature for both families."""
    assert SHARP_CONSTANT_2D == pytest.approx(1.0 / (6.0 * math.pi), rel=0, abs=0)
    assert FEWSTER_EVESON_CONSTANT_2D == pytest.approx(1.0 / (4.0 * math.pi), rel=0, abs=0)
    assert FEWSTER_EVESON_CONSTANT_2D / SHARP_CONSTANT_2D == pytest.approx(1.5, rel=1e-15)

    # wide window (6.5 sigma) so the tail truncation (~e^-42) is below the
    # quadrature comparison tolerance
    fg = gaussian_f(3.7, t0=0.4, n_sigmas=6.5)
    quad = fprime_sq_integral(fg, use_analytic=False)
    assert fg.fp2 == pytest.approx(math.sqrt(math.pi) / (2 * 3.7), rel=1e-14)
    assert quad == pytest.approx(fg.fp2, rel=1e-10), (
        f"gaussian int f'^2: quad {quad:.12e} vs analytic {fg.fp2:.12e}")

    fc = cos2_f(5.0, t0=-1.0)
    quad_c = fprime_sq_integral(fc, use_analytic=False)
    assert fc.fp2 == pytest.approx(math.pi**2 / 20.0, rel=1e-14)
    assert quad_c == pytest.approx(fc.fp2, rel=1e-10), (
        f"cos2 int f'^2: quad {quad_c:.12e} vs analytic {fc.fp2:.12e}")

    for constant in ("sharp", "fewster_eveson"):
        assert qei_bound_2d(fg, constant=constant) < 0.0
    with pytest.raises(ValueError):
        qei_bound_2d(fg, constant="nonsense")


def test_fhandle_analytic_derivatives():
    """f', f'' and the family parameter derivative agree with central FD.

    The cos2 grid stays strictly inside the compact support: FD stencils
    straddling the C^1 edge see the f'' jump and are not a fair check there;
    the edge itself is checked by its exact limits instead.
    """
    h = 1e-6
    ts_g = np.linspace(-9.0, 9.0, 41)
    ts_c = np.linspace(-5.9, 5.3, 37)  # interior of [-6.3, 5.7]
    for fh, ts in ((gaussian_f(2.3, t0=0.5), ts_g), (cos2_f(6.0, t0=-0.3), ts_c)):
        fd1 = (fh.f(ts + h) - fh.f(ts - h)) / (2 * h)
        fd2 = (fh.df(ts + h) - fh.df(ts - h)) / (2 * h)
        dev1 = np.max(np.abs(fh.df(ts) - fd1))
        dev2 = np.max(np.abs(fh.d2f(ts) - fd2))
        assert dev1 < 1e-8, f"{fh.label}: f' vs FD dev {dev1:.2e}"
        assert dev2 < 1e-7, f"{fh.label}: f'' vs FD dev {dev2:.2e}"
    # compact support edges: f and f' vanish continuously
    fc = cos2_f(6.0, t0=-0.3)
    edges = np.array([-6.3, 5.7])
    assert np.max(np.abs(fc.f(edges))) < 1e-30
    assert np.max(np.abs(fc.df(edges))) < 1e-30
    fam = gaussian_family()
    tau = 2.3
    fd_th = (gaussian_f(tau + h).f(ts_g) - gaussian_f(tau - h).f(ts_g)) / (2 * h)
    dev = np.max(np.abs(fam.df_dtheta([tau], ts_g)[0] - fd_th))
    assert dev < 1e-8, f"gaussian d f/d tau0 vs FD dev {dev:.2e}"


def test_local_psd_spectrum_matches_core_on_pd():
    """_symplectic_spectrum_psd == core.symplectic_eigenvalues on PD input."""
    rng = np.random.default_rng(3)
    n = 6
    M = rng.normal(size=(2 * n, 2 * n))
    V = M @ M.T + 0.6 * np.eye(2 * n)  # PD symmetric
    dev = np.max(np.abs(_symplectic_spectrum_psd(V) - symplectic_eigenvalues(V)))
    assert dev < 1e-10, f"local PSD spectrum vs core dev {dev:.2e}"


# --------------------------------------------------------------------------
# minimization: E_min, extremal state exhibited
# --------------------------------------------------------------------------


def test_vacuum_zero_and_emin_negative(battery_setup):
    """E_f(vacuum) = 0 by normal ordering; E_min < 0; spectrum consistent."""
    op, res = battery_setup["op"], battery_setup["res"]
    e_vac = smeared_energy(battery_setup["V_vac"], op)
    scale = abs(res.trace_vac)
    assert abs(e_vac) < 1e-10 * scale, f"E_f(vacuum) = {e_vac:.3e}, not 0"
    assert res.e_min < 0.0, f"E_min = {res.e_min:.3e}, expected negative"
    assert np.all(res.sigma >= -1e-12), "negative symplectic eigenvalue in O_f"
    recomputed = 0.5 * float(np.sum(res.sigma)) - res.trace_vac
    assert res.e_min == pytest.approx(recomputed, rel=0, abs=1e-14)


def test_extremal_state_exhibited(battery_setup):
    """The exhibited V_star is pure, physical, and achieves E_min (small gap)."""
    op, res = battery_setup["op"], battery_setup["res"]
    nu_star = symplectic_eigenvalues(res.V_star)
    purity_dev = float(np.max(np.abs(nu_star - 0.5)))
    assert purity_dev < 1e-6, (
        f"V_star should be a pure Gaussian state: max |nu - 1/2| = {purity_dev:.2e}")
    assert res.margin_star > -1e-6, f"V_star margin {res.margin_star:.2e}"
    assert res.gap >= -1e-12, f"negative optimality gap {res.gap:.2e}"
    rel_gap = res.gap / abs(res.e_min)
    assert rel_gap < 1e-4, (
        f"extremal state misses E_min by {rel_gap:.2e} rel (gap {res.gap:.2e})")
    # full-lattice embedding: physical, and reproduces `achieved` exactly
    V_emb = embed_extremal(res, op)
    e_emb = smeared_energy(V_emb, op)
    assert e_emb == pytest.approx(res.achieved, rel=0, abs=1e-12), (
        f"embedded extremal energy {e_emb:.9e} vs achieved {res.achieved:.9e}")
    m_emb = float(np.min(symplectic_eigenvalues(reduce_cov(V_emb, op.support))) - 0.5)
    assert m_emb > -1e-6, f"embedded extremal margin {m_emb:.2e}"


# --------------------------------------------------------------------------
# anchor (i): FR-type bound on every engineered state + unphysical canary
# --------------------------------------------------------------------------


def test_fr_bound_satisfied_by_engineered_states(battery_setup):
    """Every engineered squeezed-pocket state satisfies the 2d bounds (sharp
    Flanagan and the weaker Fewster-Eveson); a violation is an audit event."""
    op, res, f, K = (battery_setup["op"], battery_setup["res"],
                     battery_setup["f"], battery_setup["K"])
    site = battery_setup["site"]
    V_vac = battery_setup["V_vac"]
    V_star_full = embed_extremal(res, op)

    states = {"vacuum": V_vac, "extremal": V_star_full,
              "mix_0.3": 0.7 * V_vac + 0.3 * V_star_full,
              "squeeze_site": squeezed_pocket(K, [site], 1.0, 0.7),
              "squeeze_pair": squeezed_pocket(K, [site - 2, site + 2], 0.8, 1.1),
              "squeeze_region": squeezed_pocket(K, range(site - 3, site + 4), 0.4, 0.3)}
    rng = np.random.default_rng(7)
    for i in range(3):
        modes = np.unique(rng.integers(site - 8, site + 9, size=3))
        states[f"random_{i}"] = squeezed_pocket(
            K, modes, rng.uniform(0.1, 1.2), rng.uniform(0.0, math.pi))

    worst = math.inf
    for name, V in states.items():
        for constant in ("sharp", "fewster_eveson"):
            audit = qei_audit(V, op, f, constant=constant, tol=1e-6)
            assert audit["passed"], (
                f"{name} violates the {constant} bound: E_f {audit['e_f']:.6e} "
                f"< bound {audit['bound']:.6e}")
            worst = min(worst, audit["margin"])
        e_f = smeared_energy(V, op)
        assert e_f >= res.e_min - 1e-10, (
            f"{name}: E_f {e_f:.6e} below the global minimum {res.e_min:.6e}")
    assert worst > 0.0, f"smallest FR margin over the battery: {worst:.3e}"


def test_unphysical_canary_caught():
    """The nu < 1/2 canary: raw symplectic_margins_mp catches what the
    entropy clamp would hide (spec M3.4: no clamps in the stress-tester)."""
    N = 6
    K = harmonic_chain_K(N, 0.5, "periodic")
    f = cos2_f(3.0)
    op = sampling_operator(N, 3, f, m=0.5, bc="periodic", weight_floor=1e-12)
    V_bad = 0.9 * ground_state_cov(K)  # every nu = 0.45 < 1/2

    # (a) the gated path refuses to audit it
    with pytest.raises(QEIUnphysicalError):
        qei_audit(V_bad, op, f)

    # (b) the raw mp margin is exactly -0.05 (escalation path exercised)
    margin = min_symplectic_margin(V_bad)
    assert margin == pytest.approx(-0.05, abs=1e-10), (
        f"raw mp margin {margin:.6e}, expected -0.05")

    # (c) ungated, the unphysical state *would* "violate" the QEI — this is
    # exactly what the gate exists to catch
    e_bad = smeared_energy(V_bad, op)
    bound = qei_bound_2d(f, constant="sharp")
    assert e_bad < bound, (
        f"canary should undercut the bound: E_f {e_bad:.4e} vs {bound:.4e}")

    # (d) the entropy clamp is the WRONG detector: it silently reports zero
    # entropy for nu = 0.45 instead of failing — hence raw margins here
    nu_bad = symplectic_eigenvalues(V_bad)
    assert float(np.min(nu_bad)) < 0.5 - 1e-3
    assert entropy_from_nu(nu_bad) == 0.0  # clamped: no signal of trouble

    # (e) and a genuinely physical state passes without incident
    ok = min_symplectic_margin(reduce_cov(ground_state_cov(K), [1, 2, 3]))
    assert ok > -1e-12, f"vacuum reduced-state margin {ok:.2e}"


# --------------------------------------------------------------------------
# anchor (ii): the 2d sharp constant under Richardson extrapolation
# --------------------------------------------------------------------------


def test_sharp_constant_richardson_a_axis(richardson_table):
    """qei_ratio -> Flanagan's optimal 2d constant (ratio -> 1) under
    Richardson extrapolation in a/tau0 at m = 0, with the residual box-size
    bias measured and consistent. Spec tolerance: within 1 percent."""
    taus, ratios = richardson_table
    assert np.all(np.diff(ratios) > 0), (
        f"ratio not monotone in tau0: {ratios}")
    assert np.all(ratios < 1.0), f"lattice ratio above 1: {ratios}"

    # Richardson in h = 1/tau0^2 (leading lattice-dispersion correction is
    # O(a^2); measured c ~ -0.175): r = r_inf + c h + d h^2
    h = 1.0 / taus**2
    A = np.vstack([np.ones_like(h), h, h * h]).T
    coef, *_ = np.linalg.lstsq(A, ratios, rcond=None)
    r_inf = float(coef[0])
    resid = float(np.max(np.abs(A @ coef - ratios)))
    assert resid < 5e-5, f"Richardson fit residual {resid:.2e} (bad model)"

    # the spec anchor: within 1 percent of the sharp constant
    assert abs(r_inf - 1.0) < 0.01, (
        f"extrapolated ratio {r_inf:.6f} misses Flanagan's constant by "
        f"{abs(r_inf - 1) * 100:.3f}% (> 1%)")
    # measured margin is much better; pin it so regressions are loud
    assert abs(r_inf - 1.0) < 2e-3, (
        f"extrapolated ratio {r_inf:.6f} (historically ~0.99924 at g=80, "
        "the remainder being the measured finite-box bias)")

    # box-size axis: r(g) = r_box + A_box/g^2 at tau0 = 6; correcting the
    # g = 80 extrapolation by the fitted bias lands on 1 even tighter
    gs = np.array([40.0, 56.0, 80.0, 120.0])
    rg = np.array([_chain_run(6.0, g=int(g))[0] for g in gs])
    Ag = np.vstack([np.ones_like(gs), 1.0 / gs**2]).T
    cg, *_ = np.linalg.lstsq(Ag, rg, rcond=None)
    box_bias_at_80 = float(cg[1]) / 80.0**2  # negative
    r_corrected = r_inf - box_bias_at_80
    assert abs(r_corrected - 1.0) < 1.5e-3, (
        f"box-corrected extrapolated ratio {r_corrected:.6f} "
        f"(r_inf {r_inf:.6f}, box bias {box_bias_at_80:+.2e})")


def test_sharp_constant_mass_axis():
    """The m*tau0 axis: r(mu) -> r(0) smoothly; geometric Richardson in mu
    reproduces the direct m = 0 value. Together with the a-axis test this is
    the (a/tau0, m*tau0) -> (0, 0) joint extrapolation of spec anchor (ii)."""
    tau0 = 6.0
    r0_direct = _chain_run(tau0, mu=0.0)[0]
    mus = np.array([0.02, 0.01, 0.005])
    rv = np.array([_chain_run(tau0, mu=mu)[0] for mu in mus])
    assert np.all(np.diff(rv) > 0), f"ratio not monotone in mu: {rv}"
    assert np.all(rv < r0_direct), "massive ratio should sit below massless"

    d1 = rv[1] - rv[0]  # r(0.01) - r(0.02)
    d2 = rv[2] - rv[1]  # r(0.005) - r(0.01)
    p = math.log2(d1 / d2)  # effective power of the mu-correction
    assert 1.2 < p < 2.5, f"mass-correction exponent {p:.2f} out of range"
    r0_extrap = float(rv[2] + d2 / (2.0**p - 1.0))
    dev = abs(r0_extrap - r0_direct)
    assert dev < 3e-4, (
        f"mu -> 0 Richardson {r0_extrap:.6f} vs direct m=0 {r0_direct:.6f}: "
        f"dev {dev:.2e} (exponent {p:.2f})")


def test_cos2_family_ratio_converges():
    """Sharpness is shape-independent (Flanagan's bound is optimal for every
    smearing): the compact cos2 family also climbs toward 1 from below."""
    ratios = []
    for T in (8.0, 16.0, 32.0):
        N = int(10 * T) + 1
        f = cos2_f(T)
        op = sampling_operator(N, N // 2, f, m=0.0, bc="dirichlet")
        ratios.append(qei_minimize(op).e_min / qei_bound_2d(f))
    ratios = np.array(ratios)
    assert np.all(np.diff(ratios) > 0), f"cos2 ratios not monotone: {ratios}"
    assert np.all(ratios < 1.0), f"cos2 ratio above 1: {ratios}"
    assert ratios[-1] > 0.985, (
        f"cos2 ratio at T=32 is {ratios[-1]:.6f} (historically ~0.9910)")


# --------------------------------------------------------------------------
# anchor (iii): support-independence of the regularization
# --------------------------------------------------------------------------


def test_support_independence():
    """E_min is independent of the regularization knobs at measured levels:
    weight_floor, pad, time-window, quadrature refinement, eps_reg."""
    tau0 = 6.0
    r0, res0, op0, f0 = _chain_run(tau0)

    r_wf7 = _chain_run(tau0, weight_floor=1e-7)[0]
    assert abs(r_wf7 - r0) < 1e-4, (
        f"weight_floor 1e-9 -> 1e-7 moved ratio by {r_wf7 - r0:+.2e}")
    r_wf11 = _chain_run(tau0, weight_floor=1e-11)[0]
    assert abs(r_wf11 - r0) < 1e-6, (
        f"weight_floor 1e-9 -> 1e-11 moved ratio by {r_wf11 - r0:+.2e}")

    r_pad = _chain_run(tau0, pad=14)[0]
    assert abs(r_pad - r0) < 1e-8, f"pad 8 -> 14 moved ratio by {r_pad - r0:+.2e}"

    f_wide = gaussian_f(tau0, n_sigmas=5.5)
    N = int(80 * tau0) + 1
    op_w = sampling_operator(N, N // 2, f_wide, m=0.0, bc="dirichlet")
    r_wide = qei_minimize(op_w).e_min / qei_bound_2d(f_wide)
    assert abs(r_wide - r0) < 1e-6, (
        f"time window 4.5 -> 5.5 sigma moved ratio by {r_wide - r0:+.2e}")

    r_quad = _chain_run(tau0, panel_width=0.6, nodes_per_panel=12)[0]
    assert abs(r_quad - r0) < 1e-10, (
        f"quadrature doubling moved ratio by {r_quad - r0:+.2e}")

    # e_min must not depend on the exhibit regularizer at all
    e_a = qei_minimize(op0, eps_reg=1e-6).e_min
    e_b = qei_minimize(op0, eps_reg=1e-10).e_min
    assert e_a == pytest.approx(e_b, rel=1e-12), (
        f"e_min depends on eps_reg: {e_a:.12e} vs {e_b:.12e}")


# --------------------------------------------------------------------------
# smeared_energy plumbing, squeezed pockets, qei_ratio convenience
# --------------------------------------------------------------------------


def test_smeared_energy_shapes(battery_setup):
    """Full-lattice and support-sized covariances give identical energies;
    wrong dimensions fail loudly."""
    op = battery_setup["op"]
    V = squeezed_pocket(battery_setup["K"], [battery_setup["site"]], 0.6, 0.2)
    e_full = smeared_energy(V, op)
    e_sub = smeared_energy(reduce_cov(V, op.support), op)
    assert e_full == pytest.approx(e_sub, rel=0, abs=1e-12), (
        f"full {e_full:.9e} vs support-reduced {e_sub:.9e}")
    with pytest.raises(ValueError):
        smeared_energy(np.eye(10), op)


def test_squeezed_pocket_properties():
    """r = 0 is the vacuum; squeezing is symplectic (still physical) and
    raises the local smeared energy above the vacuum's zero."""
    N = 61
    K = harmonic_chain_K(N, 0.0, "dirichlet")
    V0 = ground_state_cov(K)
    assert np.max(np.abs(squeezed_pocket(K, [30], 0.0) - V0)) < 1e-14
    V = squeezed_pocket(K, [28, 32], [0.7, 0.4], [0.3, 1.2])
    margin = float(np.min(symplectic_eigenvalues(reduce_cov(V, range(20, 41)))) - 0.5)
    assert margin > -1e-10, f"squeezed pocket unphysical: margin {margin:.2e}"
    f = gaussian_f(3.0)
    op = sampling_operator(N, 30, f, m=0.0, bc="dirichlet")
    e = smeared_energy(V, op)
    assert e > 0.0, f"local squeezing should cost energy: E_f = {e:.3e}"


def test_qei_ratio_convenience(battery_setup):
    """qei_ratio == e_min / bound for both constants, in (0, 1) here."""
    op, res, f = battery_setup["op"], battery_setup["res"], battery_setup["f"]
    for constant in ("sharp", "fewster_eveson"):
        r = qei_ratio(op, f, constant=constant)
        expected = res.e_min / qei_bound_2d(f, constant=constant)
        assert r == pytest.approx(expected, rel=1e-12)
        assert 0.0 < r < 1.0, f"{constant} ratio {r:.6f} out of (0, 1)"


# --------------------------------------------------------------------------
# optimize_sampling: smooth f-families with analytic envelope gradients
# --------------------------------------------------------------------------


def test_optimize_sampling_gradient_is_analytic():
    """The envelope gradient (Hellmann-Feynman at the exhibited state) plus
    the integrated-by-parts bound gradient match central finite differences."""
    from vacuum.inequalities.qei import SHARP_CONSTANT_2D as C

    fam = gaussian_family()
    N, site, tau = 161, 80, 4.0
    sup = np.arange(40, 121)
    t_lo, t_hi = -31.5, 31.5

    def ratio_at(tv):
        fh = gaussian_f(tv)
        o = sampling_operator(N, site, fh, m=0.0, bc="dirichlet", t_lo=t_lo,
                              t_hi=t_hi, support=sup)
        return qei_minimize(o).e_min / qei_bound_2d(fh)

    step = 1e-4
    fd = (ratio_at(tau + step) - ratio_at(tau - step)) / (2 * step)

    fh = gaussian_f(tau)
    op = sampling_operator(N, site, fh, m=0.0, bc="dirichlet", t_lo=t_lo,
                           t_hi=t_hi, support=sup,
                           df_dtheta=lambda ts: fam.df_dtheta([tau], ts))
    res = qei_minimize(op, eps_reg=1e-10)
    bts, bws = _gauss_legendre_grid(t_lo, t_hi, 64, 20)
    dfp = fh.df(bts)
    B = -C * float(np.sum(bws * dfp * dfp))
    dB = 2.0 * C * float((fam.df_dtheta([tau], bts)[0] * bws * fh.d2f(bts)).sum())
    dE = 0.5 * float(np.einsum("ij,ji->", op.dO[0], res.V_star - op.V_vac_sub))
    grad = (dE * B - res.e_min * dB) / (B * B)
    rel = abs(grad - fd) / abs(fd)
    assert rel < 1e-4, (
        f"analytic ratio gradient {grad:+.6e} vs FD {fd:+.6e}: rel dev {rel:.2e}")


def test_optimize_sampling_finds_interior_optimum():
    """On a fixed finite lattice the Gaussian-family ratio has an interior
    maximum (wider f: fewer lattice artifacts, more box bias); the optimizer
    must find it and beat the starting point."""
    fam = gaussian_family()
    out = optimize_sampling(161, 80, fam, [3.0], bounds=[(2.5, 7.0)],
                            m=0.0, bc="dirichlet")
    assert out.converged, f"optimizer did not converge: {out.message}"
    r_start = _fixed_grid_ratio(3.0)
    assert out.ratio > r_start, (
        f"optimum {out.ratio:.6f} not above start {r_start:.6f}")
    assert 2.5 + 1e-3 < out.theta[0] < 7.0 - 1e-3, (
        f"optimum tau0* = {out.theta[0]:.3f} stuck at a bound "
        "(historically ~5.5 interior)")
    assert 0.98 < out.ratio < 1.0, f"optimal ratio {out.ratio:.6f}"


def _fixed_grid_ratio(tau):
    """Ratio at tau on the same fixed grid optimize_sampling uses."""
    fh = gaussian_f(tau)
    sup_lo = max(0, 80 - (int(math.ceil(7.0 * 4.5)) + 8))
    sup_hi = min(161 - 1, 80 + int(math.ceil(7.0 * 4.5)) + 8)
    op = sampling_operator(161, 80, fh, m=0.0, bc="dirichlet",
                           t_lo=-7.0 * 4.5, t_hi=7.0 * 4.5,
                           support=np.arange(sup_lo, sup_hi + 1))
    return qei_minimize(op).e_min / qei_bound_2d(fh)


# --------------------------------------------------------------------------
# 4d machinery smoke tests (the scan itself stays in papers/, per spec)
# --------------------------------------------------------------------------


def test_radial_ball_machinery_smoke():
    """Ball-smeared radial T_00: weights validate, per-sector minima are
    negative, and the (2l+1)-weighted terms decay with l."""
    N = 40
    w_u = radial_ball_weights(N, 3, profile="uniform")
    assert w_u.sum() == pytest.approx(3.0) and np.all(w_u >= 0)
    w_c = radial_ball_weights(N, 5, profile="cos2")
    assert np.all(w_c >= 0) and w_c[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        radial_ball_weights(N, 0)
    with pytest.raises(ValueError):
        radial_ball_weights(N, N)

    out = radial_ball_qei(N, 3, cos2_f(3.0), l_max=6, tol=1e-7)
    terms = out["terms"]
    assert out["e_min_total"] < 0.0, f"ball E_min {out['e_min_total']:.3e}"
    assert np.all(terms < 0.0), f"positive sector term: {terms}"
    mags = np.abs(terms)
    assert np.all(np.diff(mags) < 0.0), f"|terms| not decaying in l: {terms}"
    assert mags[-1] < 0.02 * mags[0], (
        f"l-decay too slow for the adaptive sum: {terms}")
    for res in out["results"]:
        assert res.e_min <= 0.0
        assert res.gap >= -1e-12


def test_cone_support_periodic_wrap():
    """Periodic cone support wraps around the chain."""
    from vacuum.inequalities.qei import _cone_support

    sup = _cone_support(20, 1, 3, "periodic")
    assert set(sup.tolist()) == {18, 19, 0, 1, 2, 3, 4}
    sup_d = _cone_support(20, 1, 3, "dirichlet")
    assert set(sup_d.tolist()) == {0, 1, 2, 3, 4}


# --------------------------------------------------------------------------
# cancellation-free sector minimizer (mode space): regression against the
# Williamson route where that route is reliable, digits recovered where it
# is not, the exhibited state, and the l-sum entry point
# --------------------------------------------------------------------------

from vacuum.inequalities.qei import (  # noqa: E402  (new surface, not re-exported)
    chain_worldline_minimize,
    mode_space_extremal_state,
    naive_volume,
    qei_minimize_mp,
    radial_sampling_operator,
    radial_sector_minimize,
    shell_volume,
)


def _old_route_noise(op, e_min):
    """Absolute float64 floor of the Williamson route, eps*lambda_max*dim,
    relative to the sector energy (the scale at which it stops being
    reliable)."""
    lam = float(np.linalg.eigvalsh(op.O)[-1])
    return 2.2e-16 * lam * op.O.shape[0] / abs(e_min)


def test_mode_space_matches_williamson_chain():
    """Same variational problem on the untruncated chain: e_min to 1e-10,
    the vacuum reference (1/2)Tr(O V_vac) to 1e-12, the leading symplectic
    eigenvalues to 1e-8 (the state's Z is re-solved with a 1e-10 weight
    shift for the chain, whose light-cone-edge packets are numerically null)."""
    N, site, tau = 61, 30, 3.0
    f = gaussian_f(tau)
    op = sampling_operator(N, site, f, m=0.0, bc="dirichlet",
                           support=np.arange(N), weight_floor=0.0)
    res = qei_minimize(op)
    ms = chain_worldline_minimize(N, site, f, fhat="quadrature")
    rel = abs(ms.e_min - res.e_min) / abs(res.e_min)
    assert rel < 1e-10, (
        f"mode-space e_min {ms.e_min:.15e} vs Williamson {res.e_min:.15e}: "
        f"rel dev {rel:.2e} (historically 7e-13)")
    assert abs(ms.e_vac - res.trace_vac) < 1e-12 * res.trace_vac, (
        f"vacuum reference {ms.e_vac:.15e} vs {res.trace_vac:.15e}")
    dev = float(np.max(np.abs(ms.sigma[:5] - res.sigma[:5])))
    assert dev < 1e-8, f"top-5 symplectic eigenvalues differ by {dev:.2e}"
    assert ms.residual < 1e-12, f"Riccati residual {ms.residual:.2e}"
    assert ms.e_dropped <= 0.0 and abs(ms.e_dropped) < 1e-10 * abs(ms.e_min)


def test_mode_space_matches_williamson_radial_low_l():
    """Radial sectors l = 0, 1, 2 (N = 40, n_ball = 2, tau = 3): agreement to
    1e-10 at l = 0, 1 and to 1e-9 at l = 2, where the Williamson route's own
    floor eps*lambda_max*dim is already ~1e-8 of the sector energy."""
    N, nb, tau = 40, 2, 3.0
    f = gaussian_f(tau)
    w = radial_ball_weights(N, nb)
    for l, tol in ((0, 1e-10), (1, 1e-10), (2, 1e-9)):
        op = radial_sampling_operator(N, l, w, f, support=np.arange(N), weight_floor=0.0)
        res = qei_minimize(op)
        ms = radial_sector_minimize(N, l, w, f, fhat="quadrature")
        rel = abs(ms.e_min - res.e_min) / abs(res.e_min)
        assert rel < tol, (
            f"l={l}: mode-space {ms.e_min:.13e} vs Williamson {res.e_min:.13e}, "
            f"rel dev {rel:.2e} > {tol:.0e} (Williamson floor est. "
            f"{_old_route_noise(op, res.e_min):.1e}; historically 4e-12, 2e-11, 3e-10)")
        assert abs(ms.e_vac - res.trace_vac) < 1e-12 * res.trace_vac
        assert ms.e_min < 0.0 and ms.e_kept < 0.0 and ms.e_dropped <= 0.0
        # per-quasimode decomposition: every term non-positive, sum = e_kept
        assert np.all(ms.mode_terms <= 1e-30), f"l={l}: positive mode term {ms.mode_terms.max():.2e}"
        sdev = abs(float(np.sum(ms.mode_terms)) - ms.e_kept) / abs(ms.e_kept)
        assert sdev < 1e-5, f"l={l}: sum of mode terms misses e_kept by {sdev:.2e}"


def test_mode_space_high_l_recovers_digits():
    """High-l sectors (N = 48, n_ball = 2, tau = 4, l = 4..9): the Williamson
    route's terms stop decaying and change sign at its float64 floor; the
    mode-space terms fall super-exponentially. At l = 7 the mpmath Williamson
    cross-check (vacuum.core.precision, 30 digits, exact for the operator as
    assembled) sits with the mode-space value, not the float64 one — the
    digits recovered are printed in the assertion message."""
    N, nb, tau = 48, 2, 4.0
    f = gaussian_f(tau)
    w = radial_ball_weights(N, nb)
    ls = list(range(4, 10))
    old, new, ops = [], [], {}
    for l in ls:
        op = radial_sampling_operator(N, l, w, f, support=np.arange(N), weight_floor=0.0)
        ops[l] = op
        old.append(qei_minimize(op).e_min)
        new.append(radial_sector_minimize(N, l, w, f, fhat="quadrature").e_min)
    old, new = np.array(old), np.array(new)
    assert np.all(new < 0.0), f"mode-space terms not all negative: {new}"
    ratios = np.abs(new[:-1]) / np.abs(new[1:])
    assert np.all(ratios > 10.0), (
        f"mode-space terms do not fall super-exponentially: ratios {ratios}")
    old_ok = np.all(old < 0.0) and np.all(np.abs(old[:-1]) > np.abs(old[1:]))
    assert not old_ok, (
        f"Williamson terms unexpectedly clean at this size: {old} (floor est "
        f"{2.2e-16 * np.linalg.eigvalsh(ops[9].O)[-1] * 96:.1e})")

    l_probe = 7
    mp_res = qei_minimize_mp(ops[l_probe], dps=30)
    e_mp = mp_res["e_min"]
    i = ls.index(l_probe)
    err_old = abs(old[i] - e_mp)
    err_new = abs(new[i] - e_mp)
    digits = math.log10(err_old / max(err_new, 1e-300))
    assert err_new < 0.05 * abs(e_mp), (
        f"l={l_probe}: mode-space {new[i]:.6e} vs mp {e_mp:.6e} ({err_new / abs(e_mp):.1%})")
    assert err_new < 0.1 * err_old, (
        f"l={l_probe}: |old-mp| {err_old:.2e}, |new-mp| {err_new:.2e}: only "
        f"{digits:.1f} digits recovered (historically 1.8 at this size; the "
        "cross-check itself is limited by the float64 assembly of O_f)")


def test_mode_space_extremal_state():
    """The exhibited state is pure and physical, achieves e_min to < 1e-5
    (exactly when no weight shift was needed), and the float64 trace of the
    untruncated O_f against it reproduces the exact variational energy."""
    N, nb, tau = 40, 2, 3.0
    f = gaussian_f(tau)
    w = radial_ball_weights(N, nb)
    for l in (0, 1):
        ms = radial_sector_minimize(N, l, w, f, fhat="quadrature")
        V, info = mode_space_extremal_state(ms)
        nu = symplectic_eigenvalues(V)
        assert float(np.max(np.abs(nu - 0.5))) < 1e-8, (
            f"l={l}: state not pure: max|nu-1/2| = {np.max(np.abs(nu - 0.5)):.2e}")
        assert float(np.min(nu)) > 0.5 - 1e-8, f"l={l}: unphysical: min nu {np.min(nu):.9f}"
        assert info["gap"] >= -1e-12 * abs(ms.e_min), f"l={l}: negative gap {info['gap']:.2e}"
        assert info["gap"] < 1e-5 * abs(ms.e_min), (
            f"l={l}: exhibited state misses e_min by {info['gap'] / abs(ms.e_min):.2e} "
            f"(state_reg {ms.meta['state_reg']:.0e}; historically 0 and 1.4e-7)")
        op = radial_sampling_operator(N, l, w, f, support=np.arange(N), weight_floor=0.0)
        e_tr = smeared_energy(V, op)
        assert abs(e_tr - info["achieved"]) < 1e-8 * abs(ms.e_min), (
            f"l={l}: float64 trace {e_tr:.12e} vs exact variational {info['achieved']:.12e}")
        assert np.all(info["squeeze_s"] < 1.0)


def test_radial_ball_qei_mode_space_and_volumes():
    """radial_ball_qei(method='mode_space') runs the l-sum to genuine
    convergence (the legacy method never does at tol -> 0), agrees with the
    legacy cone-support total to 1e-3, and its analytic-Fourier and
    windowed-quadrature samplings agree to 1e-6. The shell measure is the
    site sum of continuum shell volumes and the naive measure converges to
    it as 1 - 3/(2 n_ball) + O(1/n_ball^2)."""
    f = gaussian_f(3.0)
    legacy = radial_ball_qei(40, 3, f, l_max=12, tol=1e-30)
    assert not legacy["converged"]
    out = radial_ball_qei(40, 3, f, l_max=20, tol=0.0, method="mode_space")
    assert out["converged"] and out["l_stop"] <= 14, out["l_stop"]
    terms = out["terms"]
    assert np.all(terms < 0.0), f"positive sector term: {terms}"
    assert abs(terms[-1]) < 1e-8 * abs(terms[0]), f"tail not converged: {terms}"
    rel = abs(out["e_min_total"] - legacy["e_min_total"]) / abs(out["e_min_total"])
    assert rel < 1e-3, f"mode-space total vs legacy (cone-support) total: {rel:.2e}"
    outq = radial_ball_qei(40, 3, f, l_max=20, tol=0.0, method="mode_space", fhat="quadrature")
    relq = abs(out["e_min_total"] - outq["e_min_total"]) / abs(out["e_min_total"])
    assert relq < 1e-6, f"analytic vs windowed-quadrature sampling: {relq:.2e}"

    for n in (1, 2, 3, 5, 8):
        assert shell_volume(n) == pytest.approx(
            (4 * math.pi / 3) * ((n + 0.5) ** 3 - 0.125), rel=1e-14)
        assert shell_volume(n) == pytest.approx(
            sum(4 * math.pi * (j * j + 1.0 / 12.0) for j in range(1, n + 1)), rel=1e-14)
        r = naive_volume(n) / shell_volume(n) - 1.0
        assert abs(r + 1.5 / n) < 1.5 / n**2, (
            f"n={n}: V_naive/V_shell - 1 = {r:+.4f}, leading -3/(2n) = {-1.5 / n:+.4f}")


# --------------------------------------------------------------------------
# continuum momentum-space solution (no lattice): the 2d Flanagan control, the
# 4d worldline constant against the archived lattice band, the derived R/tau
# expansion, and the sampling-family optimizations (Layer 7, leap L1)
# --------------------------------------------------------------------------

from vacuum.inequalities.qei import (  # noqa: E402  (new surface, not re-exported)
    FEWSTER_EVESON_CONSTANT_4D,
    ball_momentum_minimize,
    bump_f,
    fhat_of,
    gaussian_poly_f,
    fsecond_sq_integral,
    lorentzian_f,
    optimize_sampling_momentum,
    rho2_coefficient,
    sqrt_lorentzian_f,
    worldline_momentum_minimize,
)

#: The archived lattice verdict (papers/qei-2d-sharp-constant/data/
#: s4e_extrapolation.json, build 2026-09-02): C_4d/C_FE = 0.4591 +- 0.0027, and
#: its six order-1 C_rho (a/tau -> 0 at fixed rho = R/tau, quadratic and cubic
#: fits in a/tau), transcribed to 7 digits.
ARCHIVED_C4D, ARCHIVED_C4D_UNC = 0.45907, 0.00267
ARCHIVED_RHO = np.array([0.5, 0.4, 1.0 / 3.0, 0.25, 1.0 / 6.0, 0.125])
ARCHIVED_C_RHO = np.array([0.4024669, 0.4209934, 0.4317096, 0.4428625, 0.4511539, 0.4541232])
ARCHIVED_C_RHO_CUBIC = np.array([0.4020459, 0.4205323, 0.4312280, 0.4423609, 0.4506388, 0.4536038])
#: pins of the continuum solver (this build): worldline constant and derived d2
CONTINUUM_C4D = 0.4574907
DERIVED_D2 = -0.24932


def _even_fit(rho, c, orders=(2, 4)):
    X = np.vstack([np.ones_like(rho)] + [rho**k for k in orders]).T
    coef, *_ = np.linalg.lstsq(X, c, rcond=None)
    return coef, c - X @ coef


def test_momentum_space_2d_flanagan_control():
    """The continuum solver reproduces Flanagan's sharp bound for every
    family (ratio -> 1): Gaussian to 1e-6 and scale-free, algebraic-tail
    Lorentzian to 1e-6, the cusped Ford-Roman weight to 2e-4 (algebraic
    quadrature convergence), the compact cos^2 to 1e-4 on a wider grid."""
    r = worldline_momentum_minimize(gaussian_f(1.0), d=2, n=400, omega_max=16.0)
    assert abs(r.ratio - 1.0) < 1e-6, f"Gaussian 2d ratio {r.ratio:.9f} (historically 0.9999999967)"
    assert r.residual < 1e-12 and abs(r.e_dropped) < 1e-8 * abs(r.e_min)
    r3 = worldline_momentum_minimize(gaussian_f(3.0), d=2, n=400, omega_max=16.0 / 3.0)
    assert abs(r3.ratio - r.ratio) < 1e-9, "the ratio is not scale invariant"
    assert abs(worldline_momentum_minimize(lorentzian_f(1.0), d=2, n=400, omega_max=30.0).ratio - 1.0) < 1e-6
    assert abs(worldline_momentum_minimize(sqrt_lorentzian_f(1.0), d=2, n=400, omega_max=30.0).ratio - 1.0) < 2e-4
    rc = worldline_momentum_minimize(cos2_f(1.0), d=2, n=800, omega_max=100.0)
    assert abs(rc.ratio - 1.0) < 1e-4, f"cos^2 2d ratio {rc.ratio:.7f} (historically 0.99995)"


def test_momentum_space_4d_worldline_constant():
    """The 4d Gaussian sharp constant from the continuum, C_4d/C_FE = 0.45749,
    converged to 5e-6 between grids, below the FE bound, and inside the
    archived lattice band 0.4591 +- 0.0027; l = 0 and the three l = 1 copies
    carry equal shares (the (1 + k.k') structure of T_00)."""
    a = worldline_momentum_minimize(gaussian_f(1.0), d=4, n=300, omega_max=12.0)
    b = worldline_momentum_minimize(gaussian_f(1.0), d=4, n=600, omega_max=20.0)
    assert abs(b.ratio - CONTINUUM_C4D) < 2e-6, f"C_4d/C_FE = {b.ratio:.8f} (pin {CONTINUUM_C4D})"
    assert abs(a.ratio - b.ratio) < 5e-6, f"grid dependence {abs(a.ratio - b.ratio):.1e}"
    assert 0.0 < b.ratio < 1.0
    assert abs(b.ratio - ARCHIVED_C4D) < ARCHIVED_C4D_UNC, (
        f"continuum {b.ratio:.5f} outside the archived lattice band {ARCHIVED_C4D} +- {ARCHIVED_C4D_UNC}")
    (l0, c0, e0), (l1, c1, e1) = b.sectors
    assert (l0, c0, l1, c1) == (0, 1, 1, 3) and abs(c1 * e1 - e0) < 1e-15 * abs(e0)
    assert b.bound == pytest.approx(-FEWSTER_EVESON_CONSTANT_4D * 3.0 * math.sqrt(math.pi) / 4.0, rel=1e-14)
    assert b.ratio == pytest.approx(worldline_momentum_minimize(gaussian_f(2.5), d=4, n=600, omega_max=8.0).ratio, abs=1e-9)


def test_rho_expansion_derived_form():
    """The R/tau expansion: even powers with the derived leading coefficient.
    (i) The ball solver reaches the worldline limit (R = 0.01 tau within
    d2 rho^2 ~ 2.5e-5 of it); (ii) the Hellmann-Feynman first-order shifts of
    the l = 0, 1 sectors match finite differences to 1e-6; (iii) the derived
    d2 = d2(HF) + d2(l = 2) = -0.2493 reproduces the small-rho slope of the
    ball solver (rho = 0.05: d4 rho^2 ~ 3e-4 away); (iv) the ball data at
    rho <= 0.2 fit C_4d + d2 rho^2 + d4 rho^4 with residual < 5e-7 and a
    free rho^3 coefficient consistent with zero."""
    f = gaussian_f(1.0)
    n, om = 300, 12.0
    wl = worldline_momentum_minimize(f, d=4, n=n, omega_max=om)
    k = rho2_coefficient(f, n=n, omega_max=om)
    assert abs(k["ratio"] - wl.ratio) < 1e-10  # two independent Riccati solves, roundoff
    for l in (0, 1):
        p = k["pieces"][l]
        rel = abs(p["de_dR2"] - p["de_dR2_fd"]) / abs(p["de_dR2"])
        assert rel < 1e-5, f"l={l}: Hellmann-Feynman {p['de_dR2']:.9e} vs FD {p['de_dR2_fd']:.9e} ({rel:.1e})"
    assert k["k2_hf"] < 0 < k["k2_l2"], "the l = 2 sector must open with negative energy (positive d2 share)"
    assert abs(k["k2"] - DERIVED_D2) < 2e-4, f"derived d2 = {k['k2']:.6f} (pin {DERIVED_D2})"
    b_small = ball_momentum_minimize(f, 0.01, n=n, omega_max=om)
    assert abs(b_small["ratio_FE"] - wl.ratio) < 3e-5
    assert abs(b_small["ratio_FE"] - (wl.ratio + k["k2"] * 1e-4)) < 3e-7
    rhos = np.array([0.05, 0.08, 0.1, 0.125, 0.15, 0.2])
    cs = np.array([ball_momentum_minimize(f, r, n=n, omega_max=om)["ratio_FE"] for r in rhos])
    slope = (cs[0] - wl.ratio) / rhos[0] ** 2
    assert abs(slope - k["k2"]) < 5e-4, f"small-rho slope {slope:.5f} vs derived d2 {k['k2']:.5f}"
    coef, res = _even_fit(rhos, cs, (2, 4))
    assert abs(coef[0] - wl.ratio) < 2e-6 and abs(coef[1] - k["k2"]) < 2e-3, (coef, wl.ratio, k["k2"])
    assert np.max(np.abs(res)) < 5e-7, f"even fit residual {np.max(np.abs(res)):.1e}"
    coef3, res3 = _even_fit(rhos, cs, (2, 3, 4))
    assert abs(coef3[2]) < 0.02 * abs(coef3[1]), f"rho^3 coefficient {coef3[2]:.4f} not negligible vs rho^2 {coef3[1]:.4f}"


def test_derived_form_reproduces_archived_lattice_value():
    """Re-extrapolating the archived lattice C_rho table with the DERIVED even
    form lands inside the archived band 0.4591 +- 0.0027, within 1e-3 of the
    continuum constant, with a fitted d2 within 0.01 of the derived one; the
    archived cubic-in-a/tau table lands within 3e-4 of the continuum, and the
    archived model with a linear rho term (0.4602) is the outlier the
    derivation excludes."""
    coef, res = _even_fit(ARCHIVED_RHO, ARCHIVED_C_RHO, (2, 4))
    assert abs(coef[0] - ARCHIVED_C4D) < ARCHIVED_C4D_UNC, coef[0]
    assert abs(coef[0] - CONTINUUM_C4D) < 1e-3, f"even-form lattice extrapolation {coef[0]:.6f} vs continuum {CONTINUUM_C4D}"
    assert abs(coef[1] - DERIVED_D2) < 0.01, f"fitted d2 {coef[1]:.4f} vs derived {DERIVED_D2}"
    assert np.max(np.abs(res)) < 5e-5
    coef_c, _ = _even_fit(ARCHIVED_RHO, ARCHIVED_C_RHO_CUBIC, (2, 4, 6))
    assert abs(coef_c[0] - CONTINUUM_C4D) < 3e-4, f"cubic-a/tau table -> {coef_c[0]:.6f}"
    X = np.vstack([np.ones_like(ARCHIVED_RHO), ARCHIVED_RHO, ARCHIVED_RHO**2]).T
    lin, *_ = np.linalg.lstsq(X, ARCHIVED_C_RHO, rcond=None)
    assert lin[0] - CONTINUUM_C4D > 2e-3, "the excluded linear-in-rho model should overshoot"


def test_d2_pinned_extrapolation_is_not_continuum_independent():
    """Audit guard (2026-09-04): an extrapolation that PINS d2 to the derived
    (continuum) value may not be averaged into a lattice number whose purpose
    is to be independent of the continuum solve, nor into the lattice-vs-
    continuum agreement. On the archived C_rho table the pinned fit sits
    ~5e-5 away from the free-d2 fits -- comparable to the whole quoted lattice
    uncertainty -- so including it materially moves both the central value and
    the apparent agreement. `notebook.py::s6c_rho_expansion` therefore reports
    `lattice_only_values_independent` (d2 free) separately from
    `lattice_only_values_d2_pinned_NOT_INDEPENDENT`, and sums only the former."""
    free = [_even_fit(ARCHIVED_RHO, ARCHIVED_C_RHO, o)[0][0] for o in ((2, 4), (2, 4, 6))]
    X = np.vstack([np.ones_like(ARCHIVED_RHO), ARCHIVED_RHO**4]).T
    pinned, *_ = np.linalg.lstsq(X, ARCHIVED_C_RHO - DERIVED_D2 * ARCHIVED_RHO**2, rcond=None)
    spread_free = abs(free[0] - free[1])
    shift = abs(pinned[0] - float(np.mean(free)))
    assert shift > spread_free, (
        f"pinned {pinned[0]:.6f} vs free {free} : shift {shift:.1e} <= free spread "
        f"{spread_free:.1e} -- if this ever holds, revisit whether the exclusion is needed")
    assert 1e-5 < shift < 5e-4, f"unexpected pinned-vs-free shift {shift:.2e}"


def test_momentum_space_4d_families_and_transforms():
    """Sampling families in 4d, C_sharp(f)/C_FE: the Ford-Roman weight (0.283)
    < Lorentzian (0.331) < Gaussian (0.4575) < compact bump p = 3 (0.55, at
    finite omega_max) — Gaussians are not optimal; every ratio < 1 (the FE
    bound holds); analytic transforms and int f''^2 agree with quadrature."""
    vals = {}
    for name, h, om in (("sqrt_lorentzian", sqrt_lorentzian_f(1.0), 30.0),
                        ("lorentzian", lorentzian_f(1.0), 30.0),
                        ("gaussian", gaussian_f(1.0), 14.0),
                        ("bump3", bump_f(1.0, 3.0), 60.0)):
        vals[name] = worldline_momentum_minimize(h, d=4, n=400, omega_max=om).ratio
        assert 0.0 < vals[name] < 1.0
    assert vals["sqrt_lorentzian"] < vals["lorentzian"] < vals["gaussian"] < vals["bump3"], vals
    assert abs(vals["sqrt_lorentzian"] - 0.28325) < 2e-3 and abs(vals["lorentzian"] - 0.33133) < 2e-3
    assert vals["bump3"] > 0.53
    # Ford-Roman weight in Fewster-Ford-Roman 2012 units: x0 = ratio * 27/128 >= their
    # moment estimate 0.0236 (PRD 85, 125038, Eq. (69), a lower estimate by their caveat)
    assert vals["sqrt_lorentzian"] * 27.0 / 128.0 > 0.0236
    nu = np.array([0.0, 0.7, 3.7, 12.0])
    # (handle, quadrature panels, tolerance): compact handles are exact to
    # roundoff; the algebraic-tail handles are limited by the truncation of
    # f^2 at the handle's support, int_{n_taus}^inf ds (1+s^2)^{-k} — 1/(3 n^3)
    # of Fhat(0) for the Lorentzian (k = 2) and 1/n for the Ford-Roman weight
    # (k = 1), i.e. 1e-6 and 5e-3 at n_taus = 60. That truncation is exactly
    # why the momentum-space route wants the analytic `f2hat`.
    for h, panels, tol in ((cos2_f(1.0), 40, 1e-9), (bump_f(1.0, 2.0), 40, 1e-9),
                           (bump_f(1.0, 2.5), 40, 1e-9),
                           (lorentzian_f(1.0, n_taus=60.0), 2000, 1e-5),
                           (sqrt_lorentzian_f(1.0, n_taus=60.0), 2000, 2e-2)):
        assert fsecond_sq_integral(h) == pytest.approx(fsecond_sq_integral(h, use_analytic=False), rel=1e-5), h.label
        assert fprime_sq_integral(h) == pytest.approx(fprime_sq_integral(h, use_analytic=False), rel=1e-5), h.label
        fq = fhat_of(h._replace(f2hat=None), n_panels=panels)(nu)
        dev = np.max(np.abs(fq - h.f2hat(nu))) / abs(h.f2hat(np.array([0.0]))[0])
        assert dev < tol, f"{h.label}: quadrature vs analytic f2hat {dev:.2e}"
        if h.t_hi > 2.0:  # the deviation must BE the tail, not a formula error
            assert dev > 1e-9, f"{h.label}: suspiciously exact ({dev:.1e}) for a truncated algebraic tail"
    assert bump_f(1.0, 2.0).fpp2 == pytest.approx(128.0 / 5.0) and bump_f(1.0, 2.0).fp2 == pytest.approx(256.0 / 105.0)
    with pytest.raises(ValueError):
        bump_f(1.0, 1.0)


def _modgauss_shape_family():
    """f = e^{-s^2/2}(1 + c s^2), tau = 1, theta = [c] — the packaged
    `gaussian_poly_f` handle (analytic Hermite transform, closed-form
    int f'^2 and int f''^2)."""
    from vacuum.inequalities.qei import SamplingFamily as _SF

    return _SF(make=lambda th: gaussian_poly_f(1.0, [1.0, float(np.asarray(th).ravel()[0])]),
               df_dtheta=None, n_params=1, label="modgauss_shape")


def test_gaussian_poly_handle():
    """gaussian_poly_f: P = [1] is exactly gaussian_f; its analytic int f'^2,
    int f''^2 and Hermite transform agree with quadrature for a quartic P."""
    g, gp = gaussian_f(1.7), gaussian_poly_f(1.7, [1.0])
    ts = np.linspace(-6.0, 6.0, 41)
    nu = np.array([0.0, 0.3, 1.1, 4.0])
    for x, y in ((g.f(ts), gp.f(ts)), (g.df(ts), gp.df(ts)), (g.d2f(ts), gp.d2f(ts))):
        assert np.max(np.abs(x - y)) < 1e-15
    assert gp.fp2 == pytest.approx(g.fp2, rel=1e-15) and gp.fpp2 == pytest.approx(g.fpp2, rel=1e-15)
    assert np.max(np.abs(gp.f2hat(nu) - g.f2hat(nu))) < 1e-15
    h = gaussian_poly_f(1.0, [1.0, 0.31, 0.065])
    assert h.fp2 == pytest.approx(fprime_sq_integral(h, use_analytic=False), rel=1e-9)
    assert h.fpp2 == pytest.approx(fsecond_sq_integral(h, use_analytic=False), rel=1e-9)
    fq = fhat_of(h._replace(f2hat=None), n_panels=200)(nu)
    assert np.max(np.abs(fq - h.f2hat(nu))) < 1e-9 * abs(h.f2hat(np.array([0.0]))[0])
    with pytest.raises(ValueError):
        gaussian_poly_f(1.0, [0.0, 0.0])


def test_optimize_sampling_momentum_2d_control_and_4d_optimum():
    """The family optimizer: in 2d the landscape is flat at Flanagan's value
    (ratio = 1 to 1e-6 at the start, the end and in between: the control);
    in 4d the same family has an interior optimum c* ~ 0.29 with ratio 0.488
    > the Gaussian's 0.4575 — Gaussians are not optimal in 4d."""
    fam = _modgauss_shape_family()
    for c in (0.0, 0.3, 0.8):
        r2 = worldline_momentum_minimize(fam.make([c]), d=2, n=400, omega_max=16.0).ratio
        assert abs(r2 - 1.0) < 1e-6, f"2d ratio at c={c}: {r2:.8f}"
    out2 = optimize_sampling_momentum(fam, [0.3], bounds=[(0.0, 1.0)], d=2, n=400, omega_max=16.0, maxiter=10)
    assert abs(out2.ratio - 1.0) < 1e-6
    out4 = optimize_sampling_momentum(fam, [0.1], bounds=[(0.0, 1.0)], d=4, n=400, omega_max=16.0)
    g4 = worldline_momentum_minimize(gaussian_f(1.0), d=4, n=400, omega_max=16.0).ratio
    assert out4.ratio > g4 + 0.02, f"4d optimum {out4.ratio:.6f} not above the Gaussian {g4:.6f}"
    assert 0.2 < out4.theta[0] < 0.4, f"c* = {out4.theta[0]:.4f} (historically 0.287)"
    assert abs(out4.ratio - 0.4884) < 2e-3, f"optimal ratio {out4.ratio:.6f} (historically 0.48843)"
    assert out4.ratio < 1.0


# --------------------------------------------------------------------------
# the published 4d values for the Lorentzian weight: the 2012 moment estimate of
# Fewster, Ford & Roman, PRD 85, 125038 (2012), arXiv:1204.3570, and the 2018
# cavity diagonalization of Schiappacasse, Fewster & Ford (SFF2018_CSHIFT below)
# --------------------------------------------------------------------------

#: FFR 2012, Eq. (68)-(69): y_inf(phidot^2) = 0.02361 +- 1e-5 for the Lorentzian
#: weight f(t) = tau/(pi (t^2 + tau^2)) (their f is our f^2), in the units
#: x = (4 pi tau^2)^2 rho; they take it as the estimate of the optimal bound
#: x0 for phidot^2 and, via their Eq. (57), for the energy density rho_S. Their
#: rigorous statement is Eq. (64), y_inf <= x0. In the same units the
#: Fewster-Eveson bound is 27/128 (their Sec. IV) and Ford-Roman's is 3/2.
FFR2012_Y_INF, FFR2012_Y_INF_ERR = 0.02361, 1e-5
FFR2012_X0_FE = 27.0 / 128.0
#: Schiappacasse, Fewster & Ford, PRD 97, 025013 (2018), arXiv:1711.09477, Table I:
#: the same operator (normal-ordered phidot^2 of the 4d massless scalar) and the same
#: Lorentzian weight, diagonalized by a Bogoliubov transformation in a finite
#: spherical cavity; its lowest eigenvalue, in FFR's units (they set it beside
#: FFR's -0.0236 and -27/128 themselves). Missed by this repository until 2026-09-23.
SFF2018_CSHIFT = 0.0593338
#: this build's exact infimum for the same weight, same units (continuum
#: momentum-space route, n = 1200, omega_max = 40/tau; cusp-limited to ~1e-5; the
#: test runs n = 600 and its 5e-5 budget covers the 1e-5 grid shift)
OURS_X0_FORD_ROMAN_WEIGHT, OURS_X0_BUDGET = 0.05976, 5e-5


def test_ffr2012_lorentzian_acceptance():
    """The acceptance test against the published 4d values for the Lorentzian weight.

    Our infimum of the Lorentzian-weighted energy density in FFR's units is
    x0 = 0.0598 -- NOT within 0.02361 +- 1e-5 (+ our 5e-5 budget): the ratio
    is 2.53. It does satisfy every rigorous statement in that paper: their
    Eq. (64) y_inf <= x0 -- strictly, which is what their Sec. IV says
    happens if the moment problem is Hamburger-indeterminate ("y_inf(A) =
    x_F(A) < x0(A)"). On that question their abstract states non-uniqueness
    as a caveat ("all of our results are subject to the caveat that these
    distributions are not uniquely determined by the moments") while the
    body leaves it open: the Hamburger and Stieltjes sufficient conditions
    fail, "this does not prove that the distribution is nonunique ... we
    have not been able to resolve the question of uniqueness" (Sec. I), and
    "on balance our expectation is that the problem is indeed indeterminate"
    (Sec. VIII). The exhibited state of `test_box_wavepacket_state_beats_ffr`
    settles it on their own reasoning: a physical state sits below -y_inf,
    so y_inf < x0 and the test converged to the Friedrichs edge. The
    Fewster-Eveson bound x0 <= 27/128 also holds. The identity their
    Eq. (57) infers by treating the components as independent, x0(rho_S) =
    x0(phidot^2), holds EXACTLY here: phidot^2 (l = 0), (grad phi)^2 (l = 1)
    and T_00 share one infimum E_Q/(2 pi^2). So the gap is the gap between
    the Friedrichs edge of an indeterminate moment problem and the true
    spectral edge.

    NOT A FINDING -- A REPRODUCTION (corrected 2026-09-23). Until then this
    docstring called the 2.53x "a finding for expert eyes". It had been
    published: Schiappacasse, Fewster & Ford, PRD 97, 025013 (2018) -- two of
    the three 2012 authors -- diagonalized the same operator for the same
    weight in a finite cavity and found C_shift = -0.0593338, setting it beside
    the 2012 value themselves. x0 here reproduces it (+0.71 % at n = 600;
    infinite space against their cavity), pinned in (vi) so neither comparison
    can silently drift. What is this repository's own is the mechanism behind
    (iii): the 2012 number reproduced by the 2012 method, and why it misses."""
    f = sqrt_lorentzian_f(1.0)
    vals = {op: worldline_momentum_minimize(f, d=4, n=600, omega_max=40.0, operator=op)
            for op in ("T00", "phidot2", "gradphi2")}
    x0 = {op: r.ratio * FFR2012_X0_FE for op, r in vals.items()}
    # (i) the three infima coincide exactly (one squeezing state serves all)
    assert abs(x0["phidot2"] - x0["T00"]) < 1e-12 and abs(x0["gradphi2"] - x0["T00"]) < 1e-12, x0
    assert vals["phidot2"].sectors[0][0] == 0 and vals["gradphi2"].sectors[0][0] == 1
    # (ii) our value, pinned (n = 600 sits ~1e-5 below the n = 1200 value 0.05976)
    assert abs(x0["T00"] - OURS_X0_FORD_ROMAN_WEIGHT) < OURS_X0_BUDGET, x0["T00"]
    # (iii) the published estimate is NOT reproduced: outside their error + our budget
    tol = FFR2012_Y_INF_ERR + OURS_X0_BUDGET
    assert abs(x0["T00"] - FFR2012_Y_INF) > tol, (
        f"x0 = {x0['T00']:.5f} now agrees with FFR's 0.02361 -- the indeterminacy "
        "reading in draft/outline.md section 6b would be wrong; revisit")
    assert 2.4 < x0["T00"] / FFR2012_Y_INF < 2.7, x0["T00"] / FFR2012_Y_INF
    # (iv) every rigorous statement in the literature is respected
    assert x0["T00"] > FFR2012_Y_INF + FFR2012_Y_INF_ERR, "their Eq. (64) y_inf <= x0 would be violated"
    assert x0["T00"] < FFR2012_X0_FE, "the Fewster-Eveson bound would be violated"
    assert x0["T00"] < 1.5, "the Ford-Roman bound would be violated"
    # (v) the same handle through the 2d kernel is Flanagan's sharp bound
    assert abs(worldline_momentum_minimize(f, d=2, n=600, omega_max=40.0).ratio - 1.0) < 2e-4
    # (vi) the 2018 diagonalization of the same operator and weight (finite cavity) is
    # REPRODUCED: +0.71 % at n = 600, +0.72 % at the archived n = 1200 (infinite space
    # here). The band is calibrated on those two values; a drift of ~0.2 % either way,
    # or agreement with FFR 2012 instead, fails.
    dev_sff = (x0["T00"] - SFF2018_CSHIFT) / SFF2018_CSHIFT
    assert 0.005 < dev_sff < 0.009, dev_sff


def test_box_wavepacket_state_beats_ffr():
    """PROOF-LEVEL version of the FFR comparison: an EXHIBITED physical state.

    Orthonormal box wavepackets (`box_wavepacket_state`) restrict the
    Lorentzian-weighted phidot^2 form to a finite normalizable mode set; the
    Bogoliubov vacuum of that restriction, tensored with the vacuum on the
    complement, is an explicit physical state whose expectation value of the
    normal-ordered operator is computed EXACTLY (variational energy, no
    dropped-mode estimate, |Z| < 1 checked). Therefore, rigorously,
        x0  >=  <x>_state  =  0.05975  (300 boxes; 0.05976 at 1000),
    which exceeds Fewster-Ford-Roman's y_inf + error = 0.02362 by 0.036 --
    a factor 2.53 -- with no appeal to solver convergence. The only numerics
    are Gauss-Legendre integrals of analytic pieces, shown node-independent
    to 1e-7 here. The Gaussian control: the same construction sits BELOW the
    solver's infimum (as a variational bound must) by 7e-5."""
    from vacuum.inequalities.qei import box_wavepacket_state

    f = sqrt_lorentzian_f(1.0)
    st = box_wavepacket_state(f, n_boxes=300, nodes=12, omega_max=40.0)
    st24 = box_wavepacket_state(f, n_boxes=300, nodes=24, omega_max=40.0)
    x_state = st["ratio_FE"] * FFR2012_X0_FE
    assert st["z_norm_used"] < 1.0 and not st["clipped"], "state must be normalizable as constructed"
    assert st["riccati_residual"] < 1e-12 and abs(st["e_state"] - st["e_riccati"]) < 1e-12 * abs(st["e_state"])
    assert abs(st24["ratio_FE"] - st["ratio_FE"]) * FFR2012_X0_FE < 1e-7, "quadrature-dependent matrix elements"
    # the rigorous statement, with a stated margin
    margin = 0.036
    assert x_state > FFR2012_Y_INF + FFR2012_Y_INF_ERR + margin, (
        f"exhibited state <x> = {x_state:.6f} does not beat y_inf + err + {margin}")
    assert x_state / FFR2012_Y_INF > 2.5
    assert abs(x_state - 0.059747) < 2e-5, x_state
    # variational: the state cannot go below the solver's infimum estimate
    solver = worldline_momentum_minimize(f, d=4, n=600, omega_max=40.0).ratio * FFR2012_X0_FE
    assert x_state <= solver + 1e-6 and solver - x_state < 5e-5, (x_state, solver)
    # T00 and (grad phi)^2 states carry the same value (proportional kernels, same Z)
    for op in ("T00", "gradphi2"):
        assert abs(box_wavepacket_state(f, n_boxes=300, nodes=12, omega_max=40.0, operator=op)["ratio_FE"]
                   - st["ratio_FE"]) < 1e-12
    # Gaussian control: below the solver value by a small, positive amount
    g = gaussian_f(1.0)
    sg = box_wavepacket_state(g, n_boxes=300, nodes=12, omega_max=14.0)
    ref = worldline_momentum_minimize(g, d=4, n=400, omega_max=14.0).ratio
    assert 0.0 < ref - sg["ratio_FE"] < 2e-4, (sg["ratio_FE"], ref)


# --------------------------------------------------------------------------
# Fewster-Ford-Roman 2012 reproduced BY THEIR METHOD on our kernel
# (vacuum.inequalities.ffr_moments): Table I moments, Table II Stieltjes
# edges, the Eq. (68) fit, the Carleman diagnostic, and the exhibited state
# placed above the reproduced y_inf
# --------------------------------------------------------------------------

import vacuum.inequalities.ffr_moments as ffr  # noqa: E402


@pytest.fixture(scope="module")
def ffr_moments_65():
    a3, C3, K3 = ffr.ffr_moment_sequence(3, 65)
    a1, C1, K1 = ffr.ffr_moment_sequence(1, 65)
    return {"a3": a3, "C3": C3, "K3": K3, "a1": a1, "C1": C1, "K1": K1}


def test_ffr_table_I_moments_reproduced(ffr_moments_65):
    """Their Appendix A on our transcription: the chain recurrence (A6), the
    run-structure derivation (A8) evaluated as the flow d/ds of K_1(s)^2/2,
    and M = e^W. Exact: a_2 = 9/2, a_3 = 1890, a_4 = 10206243/4 (phidot^2),
    a_2..a_4 = 2, 48, 1740 (phi^2); (A11)-(A14) reproduced symbolically
    (as exact rationals at the K_n values); every printed Table I entry
    (44 of them, 5 s.f.) to the rounding, 5e-5."""
    from fractions import Fraction as Fr
    m = ffr_moments_65
    a3, C3, K, a1 = m["a3"], m["C3"], m["K3"], m["a1"]
    assert a3[2] == Fr(9, 2) and a3[3] == 1890 and a3[4] == Fr(10206243, 4)
    assert (a1[2], a1[3], a1[4]) == (2, 48, 1740)
    assert C3[4] == 8**4 * (K[3] * K[1] + K[2]**2 + K[1]**4)
    assert C3[5] == 8**5 * (K[4] * K[1] + 3 * K[3] * K[2] + 8 * K[2] * K[1]**3)
    assert C3[6] == 8**6 * (K[5] * K[1] + 3 * K[3]**2 + 4 * K[4] * K[2] + 13 * K[3] * K[1]**3
                            + 31 * K[2]**2 * K[1]**2 + 8 * K[1]**6)
    assert C3[7] == 8**7 * (K[6] * K[1] + 10 * K[4] * K[3] + 5 * K[5] * K[2] + 19 * K[4] * K[1]**3
                            + 66 * K[2]**3 * K[1] + 123 * K[3] * K[1]**2 * K[2] + 136 * K[2] * K[1]**5)
    import mpmath as mp
    for p, a in ((3, a3), (1, a1)):
        for n, ref in ffr.FFR2012_TABLE_I[p].items():
            val = float(mp.mpf(a[n].numerator) / a[n].denominator)
            assert abs(val / ref - 1) < 5e-5, f"p={p} a_{n} = {val:.6e} vs Table I {ref:.5e}"


def test_ffr_kernel_is_their_kernel():
    """Our operator for the Ford-Roman weight has a rank-one anomalous block,
    so its necklace traces are FFR's chain integrals K_n = v^T A'^{n-1} v:
    Nystrom values on three grids converge monotonically to the exact (A6)
    recurrence values (K_1 to roundoff, the rest with the cusp's h^2 rate,
    Richardson to < 2e-5 for n <= 16), and the identity (A4) that turns the
    chain into the recurrence holds to 1e-15 in direct quadrature."""
    for q, kap in ((3, 0.7), (7, 2.5), (12, 9.0)):
        d, c, r = ffr.ffr_identity_A4(q, kap)
        assert r < 1e-15, (q, kap, d, c, r)
    kc = ffr.kernel_chain_integrals(sqrt_lorentzian_f(1.0), 3, 16, grids=((800, 60.0), (1600, 60.0), (3200, 60.0)))
    assert abs(kc["chain"][0]["rel_dev_finest"]) < 1e-12, "K_1 involves no A and must be exact"
    assert kc["all_monotone"], "Nystrom values must approach the exact K_n as h -> 0"
    assert kc["worst_rel_dev_richardson"] < 2e-5, kc["worst_rel_dev_richardson"]
    assert kc["worst_rel_dev_finest"] < 2e-3


def test_ffr_table_II_and_y_inf_reproduced(ffr_moments_65):
    """Their Sec. IV Stieltjes test on our moments: y_N = largest root of
    det(a_{m+n+1} + y a_{m+n}) = 0, here as minus the smallest Gauss node
    from the exact Chebyshev recurrence. Every Table II entry (31 x 2, 11
    digits) to 1e-11; their Eq. (68) fit on 21 <= N <= 33 returns
    y_inf(phidot^2) = 0.0236173 against their 0.0236175, inside their stated
    0.02361 +- 1e-5 -- THEIR NUMBER, BY THEIR METHOD, ON OUR KERNEL; and the
    phi^2 control lands on 1/6 exactly as it did for them (0.166666666057)."""
    a3, a1 = ffr_moments_65["a3"], ffr_moments_65["a1"]
    y3 = ffr.stieltjes_sequence(a3, 33, 2, dps=60)
    y1 = ffr.stieltjes_sequence(a1, 33, 2, dps=60)
    for p, y in ((3, y3), (1, y1)):
        for N, ref in ffr.FFR2012_TABLE_II[p].items():
            assert abs(y[N] - ref) < 1e-11, f"p={p} y_{N} = {y[N]:.12f} vs Table II {ref:.11f}"
    fit = ffr.ffr_fit_y_inf(y3, 21, 33)
    assert abs(fit["y_inf"] - ffr.FFR2012_Y_INF) < ffr.FFR2012_Y_INF_ERR, fit
    assert abs(fit["y_inf"] - 0.0236174942666) < 5e-7, fit["y_inf"]
    assert fit["max_resid"] < 1e-9
    Ns = np.arange(21, 34, dtype=float)
    ys = np.array([y1[int(N)] for N in Ns])
    X = np.vstack([np.ones_like(Ns), 1 / Ns, 1 / Ns**2]).T
    c, *_ = np.linalg.lstsq(X, ys, rcond=None)
    assert abs(c[0] - 0.166666666057) < 2e-11, c[0]
    assert all(y3[N + 1] > y3[N] for N in range(2, 33)), "y_N must increase with N (their Eq. 64)"


def test_ffr_carleman_and_blindness(ffr_moments_65):
    """The moment sequence grows as (3n-3)! x 3.36^n (their Appendix B class);
    both Carleman series converge (terms ~ n^{-3/2}), so the sufficient
    criteria for determinacy are inconclusive -- their Sec. I statement --
    and the growth class is Stieltjes' own indeterminate example. Shown, not
    asserted: a point mass at -0.059761 (our exhibited state's value) with
    weight 1e-3, 1e-2, even 1e-1 moves y_32 only to 0.02134, 0.02168,
    0.02515 -- the 65-moment test cannot see a support edge there."""
    from fractions import Fraction as Fr
    a3 = ffr_moments_65["a3"]
    car = ffr.carleman_diagnostics(a3, 3, shift=1)
    assert 3.2 < car["geometric_factor_per_n"] < 3.5, car["geometric_factor_per_n"]
    assert car["stieltjes_partial_sum"] < 5.0 and 1.4 < car["stieltjes_term_decay_exponent"] < 2.0, car
    assert car["hamburger_partial_sum"] < 1.0 and car["hamburger_term_decay_exponent"] > 1.5, car
    y32 = ffr.stieltjes_edge(a3, 32, 60)
    for w, expect in ((Fr(1, 1000), 0.02134), (Fr(1, 100), 0.02168), (Fr(1, 10), 0.02515)):
        ap = ffr.point_mass_perturbation(a3, Fr(59761, 10**6), w)
        y = ffr.stieltjes_edge(ap, 32, 60)
        assert abs(y - expect) < 3e-5, (float(w), y, expect)
        assert y < 0.03 < 0.059761, "the test would have to reach 0.0598 to see the added edge"
    assert abs(y32 - 0.02129828002) < 1e-11


def test_exhibited_state_above_reproduced_y_inf(ffr_moments_65):
    """The comparison at the same f: the reproduced y_inf (their method, our
    kernel) against our exhibited state's <x> = 0.05975 (300 boxes) --
    rigorous x0 >= 0.05975 > y_inf + 1e-5 by 0.036 -- and the same weight's
    C_sharp/C_FE = 0.283 is the Lorentzian point of the family curve next to
    the Gaussian 0.457 and (1-u^2)^2 0.611 of section 6b."""
    from vacuum.inequalities.qei import box_wavepacket_state
    y3 = {N: ffr.stieltjes_edge(ffr_moments_65["a3"], N, 60) for N in range(21, 34)}
    y_inf = ffr.ffr_fit_y_inf(y3, 21, 33)["y_inf"]
    st = box_wavepacket_state(sqrt_lorentzian_f(1.0), n_boxes=300, nodes=12, omega_max=40.0)
    x_state = st["ratio_FE"] * 27.0 / 128.0
    assert x_state > y_inf + ffr.FFR2012_Y_INF_ERR + 0.036, (x_state, y_inf)
    assert 0.282 < st["ratio_FE"] < 0.284


def test_ffr_chain_check_uses_the_solver_kernel_code_path():
    """The K_n of the n <= 16 kernel check come from the SAME code path as the
    solver and the exhibited state: `kernel_chain_integrals` builds its
    matrices with `qei._worldline_kernels(momentum_grid, fhat_of(f), p)` --
    bitwise the matrices `worldline_momentum_minimize` assembles -- and the
    exhibited state integrates the same `fhat_of(f)` and (ww')^{3/2}
    weighting over its boxes. Checked: (i) the chain module's matrices equal
    `_worldline_kernels` output exactly; (ii) B is rank one to 1e-13 with
    v read off its diagonal; (iii) K_n from the box state's own returned
    matrices converge to the same exact (A6) values; (iv) the accelerated
    sequence carries FFR's Table III labels (10 entries to 1e-11)."""
    from vacuum.inequalities.qei import _worldline_kernels, box_wavepacket_state, fhat_of, momentum_grid
    f = sqrt_lorentzian_f(1.0)
    kc = ffr.kernel_chain_integrals(f, 3, 12, grids=((600, 60.0), (1200, 60.0)))
    assert kc["code_path"].startswith("qei._worldline_kernels")
    assert kc["rank_one_residual"] < 1e-13, kc["rank_one_residual"]
    omg, wom = momentum_grid(60.0, 600, power=1)
    A_wk, B_wk = _worldline_kernels(omg, wom, fhat_of(f), 3)
    v = np.sqrt(np.diag(B_wk).real / math.pi)
    x, K = v.copy(), []
    for n in range(1, 13):
        K.append(float(v @ x) / math.pi ** (n - 1))
        x = A_wk.real @ x
    assert np.allclose(K, kc["per_grid"][0]["K"], rtol=0, atol=0), "chain check must use the solver's kernel bitwise"
    # (iii) the exhibited state's own matrices: box-integrated same kernel; K_n = (2 pi)^{n-1} u^T A_op^{n-1} u
    st = box_wavepacket_state(f, n_boxes=300, nodes=12, omega_max=40.0)
    A_op, B_op = st["A"], st["B"]                      # per-copy phidot^2 form: A_op = A'/(pi) ... scaled 2 A_0
    u = np.sqrt(np.diag(B_op).real / (2.0 * math.pi ** 2) * (2.0 * math.pi ** 2) / (2.0 * math.pi ** 2))
    # B_op = 2 A_0-block = 2 (pi u u^T)/(4 pi^2)/(...): read the rank-one vector from the diagonal directly
    u = np.sqrt(np.clip(np.diag(B_op).real, 0.0, None))
    scale = 1.0 / np.max(np.abs(B_op))
    assert np.max(np.abs(B_op - np.outer(u, u))) * scale < 1e-12, "box-integrated anomalous block must be rank one"
    Kex = ffr.chain_integrals_exact(3, 8)
    # A_op = (2/(4 pi^2)) * pi * A'_box = A'_box/(2 pi); B_op = u u^T with u = v_box/sqrt(2 pi)  =>  K_n = (2 pi)^n u^T A_op^{n-1} u
    x, Kbox = u.copy(), []
    for n in range(1, 9):
        Kbox.append(float(u @ x) * (2.0 * math.pi) ** n)
        x = A_op @ x
    worst = max(abs(Kbox[n - 1] / float(Kex[n]) - 1) for n in range(1, 9))
    assert worst < 5e-3, f"box-state kernel chain vs exact: {worst:.1e}"   # piecewise-constant boxes: O(h^2)
    st2 = box_wavepacket_state(f, n_boxes=600, nodes=12, omega_max=40.0)
    u2 = np.sqrt(np.clip(np.diag(st2["B"]).real, 0.0, None))
    x, Kbox2 = u2.copy(), []
    for n in range(1, 9):
        Kbox2.append(float(u2 @ x) * (2.0 * math.pi) ** n)
        x = st2["A"] @ x
    worst2 = max(abs(Kbox2[n - 1] / float(Kex[n]) - 1) for n in range(1, 9))
    assert worst2 < worst, "refining the boxes must move the box-state chain toward the exact K_n"
    a3, _, _ = ffr.ffr_moment_sequence(3, 65)
    y3 = ffr.stieltjes_sequence(a3, 33, 2, dps=60)
    acc = ffr.accelerated_sequence(y3)
    for N, ref in ffr.FFR2012_TABLE_III.items():
        assert abs(acc[N] - ref) < 1e-11, (N, acc[N], ref)


def test_lorentzian_three_routes_and_ir_bias():
    """The Ford-Roman weight through the lattice-free momentum route, the
    exhibited state, and the Srednicki lattice, in FFR units. (i) The
    continuum worldline solve (no box, no lattice: no infrared bias can
    enter) gives x0 = 0.05976; (ii) the exhibited state sits within 2e-5 of
    it from below-or-equal in energy (a lower bound on x0); (iii) one lattice
    point (rho = 1/4, n_ball = 3, g = 16) corrected with the box law measured
    for THIS weight in S6e (g^-2.50, B = -2.3706) lands within the (a/R)^2
    term of the continuum ball value at the same rho, whereas the Gaussian
    -10.04/g^4 law leaves it ~1% low -- the infrared bias is a property of the
    weight, real on the lattice, and absent from the number we quote."""
    from vacuum.inequalities.qei import ball_momentum_minimize, box_wavepacket_state, radial_ball_qei, shell_volume
    U = 27.0 / 128.0
    f = sqrt_lorentzian_f(1.0)
    x0_cont = worldline_momentum_minimize(f, d=4, n=800, omega_max=40.0).ratio * U
    assert abs(x0_cont - 0.05976) < 3e-5, x0_cont
    x_state = box_wavepacket_state(f, n_boxes=600, nodes=12, omega_max=40.0)["ratio_FE"] * U
    assert 0.05974 < x_state < x0_cont + 2e-5, (x_state, x0_cont)
    assert x_state > 0.02361 + 1e-5 + 0.036
    C_cont = ball_momentum_minimize(f, 0.25, n=500, omega_max=40.0, l_max=20)["ratio_FE"]
    assert abs(C_cont - 0.2706) < 5e-4, C_cont
    nb, rho, g = 3, 0.25, 16.0
    tau = (nb + 0.5) / rho
    N = int(nb + math.ceil(g * tau) + 8)
    out = radial_ball_qei(N, nb, sqrt_lorentzian_f(tau), l_max=80, tol=0.0, method="mode_space", rel_tol=1e-12)
    C_raw = -out["e_min_total"] / (shell_volume(nb) * fsecond_sq_integral(sqrt_lorentzian_f(tau))) / FEWSTER_EVESON_CONSTANT_4D
    gg = N / tau
    C_this = C_raw + 2.3706 * gg ** -2.50            # measured law for this weight (S6e)
    C_gauss = C_raw + 10.04 / gg ** 4                # the Gaussian law, wrongly applied
    aR2 = 1.0 / (nb + 0.5) ** 2                      # the remaining (a/R)^2 lattice term, ~ -0.03 aR^2 here
    assert abs(C_this - C_cont) < 0.05 * aR2, (C_this, C_cont)            # only the (a/R)^2 term is left
    shortfall = (C_cont - C_gauss) - abs(C_cont - C_this)                   # what the Gaussian law fails to remove
    assert shortfall > 1.2e-3, f"the Gaussian box law must visibly fail for this weight ({shortfall:.1e})"
    assert abs(C_raw - 0.26587) < 3e-4, C_raw       # the archived S6e point
