"""Anchors for the perturbative UDW engine (Layer 2, milestone M2.3).

What is pinned here, in the order the spec asks for it:

1.  **Conventions.**  PKMM's Gaussian switching Eq. (22) vs this repo's
    ``switching('gaussian', T)``; the Eq. (13)/(14) matrix layout; the
    Eq. (66)-(67) negativity against a brute-force partial transpose.

2.  **Single-detector P_nu against the closed form to 1e-8** (spec M2.3).
    Three independent routes agree: PKMM Eq. (29), the M2.1 closed form
    ``kernels.response_gaussian_closed_form`` (derived from scratch in
    that module, not from the paper), and the quadrature engine.  On the
    lattice the engine is pinned against the *exact* Gaussian mode sum.

3.  **The engine against PKMM's closed forms.**  ``udw_pair_state`` on the
    continuum 3+1 backend reproduces Eq. (29) [P_A, P_B], Eq. (30) [C] and
    Eq. (31) [|M|] -- for positive, zero and negative gap.  This is the
    strongest available statement about the transcription: it is not a
    2%-digitization statement, it is agreement with the paper's own
    analytic expressions at the 1e-12 level.

4.  **PKMM harvested-negativity structure across separation / gap /
    smearing.**  Spot values digitized out of the *vector artwork* of
    PKMM Fig. 2 (see :data:`PKMM_FIG2_B1_POINTLIKE` and
    :data:`PKMM_FIG2_A1_SMEARED` for the full provenance) are reproduced
    to <= 2%; the qualitative structure (entanglement death with
    separation, the direction of the gap dependence, the direction of the
    smearing dependence) is asserted exactly.

5.  **Both kernel backends.**  ``udw_pair_state`` runs on the continuum
    3+1 kernel (PKMM anchors) and on ``LatticeWightman`` (the circuit-QED
    digital twin's native spacetime), and the lattice pair term M is
    cross-validated against ``kernels.triangle_rule``, an independently
    written quadrature for the same time-ordered triangle.

Cost discipline: the continuum engine costs seconds per point, so all of
its evaluations live in module-scoped fixtures and are reused.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from vacuum.core.models import harmonic_chain_K
from vacuum.detectors import (
    smearing,
    switching,
    wightman_continuum_3p1,
    wightman_lattice,
)
from vacuum.detectors.kernels import (
    gl_panels,
    response_gaussian_closed_form,
    triangle_rule,
)
from vacuum.detectors.udw import (
    PKMM_NONOVERLAP_GAMMA,
    PKMM_SWITCHING_WIDTH_FACTOR,
    UDWDetector,
    UDWPairState,
    cross_term,
    pair_kernels,
    pair_log_negativity,
    pair_negativity,
    pair_term,
    pkmm_L_AA,
    pkmm_L_AB,
    pkmm_M_abs,
    pkmm_M_abs_coincident_pointlike,
    pkmm_M_abs_nonoverlap,
    pkmm_negativity_estimator,
    pkmm_pair_state,
    transition_probability,
    udw_pair_state,
)

# ---------------------------------------------------------------------------
# Digitized PKMM Fig. 2 anchors -- provenance
# ---------------------------------------------------------------------------
#
# Source: the arXiv source tarball of arXiv:1506.03081 (fetched
# 2026-09-01), files ``3DGP-gamma-fig.pdf`` (Fig. 2 panel b.1, pointlike
# detectors) and ``3DGS-gamma-fig.pdf`` (Fig. 2 panel a.1, Gaussian
# smearing sigma/T = 1).  Both panels are 3+1D with Gaussian switching
# [Eq. (22)] and a fixed switching delay Delta/T = 3 (stated in the panel
# title and in the Fig. 2 caption); they show in dark red the (d/T, Omega T)
# region where entanglement harvesting is possible, i.e. where
# N^(2) = |M| - L_AA > 0 [Eq. (68)].
#
# Digitization: each panel's plot area is a single 2001 x 2040 DeviceRGB
# image (PDF object /Subtype /Image, FlateDecode) with exactly two colors,
# (170, 0, 0) "harvesting" and (196, 196, 196) "no harvesting", placed by
# the content stream at "434.957787 0 0 435.2 7.521106 7.4 cm", which is
# exactly the span of the 21 axis tick marks the same stream draws.  The
# rows below are the largest d/T that is still red, at nine evenly spaced
# heights.
#
# Horizontal calibration is *independent and exact*: the same content
# stream draws a dashed vertical line at x = 353.301, which maps to
# d/T = (353.301 - 7.521106)/434.957787 * 10 = 7.94975, and the Fig. 2
# caption puts the light-cone boundary of these panels at
# Delta = d - 7T/sqrt(2), i.e. d/T = 3 + 7/sqrt(2) = 7.94975.  So the
# horizontal axis spans d/T in [0, 10] to five decimals.
#
# Vertical calibration is *not* stated unambiguously by the artwork: the
# panels carry 21 evenly spaced tick marks per edge but the axis labels
# (2, 4, 6, 8, 10, with no 0) are typeset separately, so the range has to
# be recovered.  It was fitted from panel b.1 alone -- the pointlike case,
# which has no smearing parameter left to absorb an error: inverting the
# nine digitized d* values through the pointlike model gives an affine map
# from image height fraction f to Omega T with endpoints 1.02 (f = 0) and
# 9.98 (f = 1), i.e. the vertical axis spans Omega T in [1, 10] and
# Omega T = 1 + 9 f.  Eight free digitized points against two calibration
# parameters is heavily over-determined, and the residuals are <= 0.15%.
#
# That single calibration is then applied *unchanged* to panel a.1, whose
# physics differs (sigma/T = 1), where it predicts a different curve
# correctly to <= 0.25% -- see ``test_pkmm_fig2_a1_smeared_death_line``.
# The last a.1 point is saturated at the plot edge (the red region runs
# off the right of the frame) and is recorded as a lower bound.

#: (Omega T, largest d/T with N^(2) > 0) from PKMM Fig. 2 panel b.1
#: (3+1D, Gaussian switching, pointlike detectors, Delta/T = 3).
PKMM_FIG2_B1_POINTLIKE = (
    (1.9, 3.671),
    (2.8, 4.685),
    (3.7, 5.380),
    (4.6, 6.009),
    (5.5, 6.674),
    (6.4, 7.404),
    (7.3, 8.178),
    (8.2, 8.983),
    (9.1, 9.803),
)

#: Same, from PKMM Fig. 2 panel a.1 (Gaussian smearing sigma/T = 1).  The
#: Omega T = 9.1 row saturates at the frame edge (digitized 9.998), so it
#: is excluded here and checked separately as a lower bound.
PKMM_FIG2_A1_SMEARED = (
    (1.9, 3.591),
    (2.8, 4.835),
    (3.7, 5.685),
    (4.6, 6.394),
    (5.5, 7.064),
    (6.4, 7.744),
    (7.3, 8.468),
    (8.2, 9.233),
)

#: Plot-digitization tolerance (spec M2.3: "spot values to plot-digitization
#: tolerance (2%)").
DIGITIZATION_RTOL = 0.02

#: PKMM Fig. 2 panels a.1/b.1 fix the switching delay at Delta/T = 3.
FIG2_GAMMA = 3.0

#: sigma/T used in place of the pointlike limit of panel b.1.  Eq. (31) is
#: only conditionally convergent at delta = 0; delta = 0.02 sits on the
#: pointlike plateau (checked in ``test_pointlike_plateau``).
POINTLIKE_DELTA = 0.02


def _death_separation(alpha, gamma, delta, hi=25.0):
    """Largest d/T with N^(2) > 0 -- the harvesting "death line" of Fig. 2."""
    f = lambda beta: pkmm_negativity_estimator(alpha, beta, gamma, delta)  # noqa: E731
    lo = 0.05
    assert f(lo) > 0.0, f"no harvesting even at d/T = {lo} (alpha={alpha})"
    return brentq(f, lo, hi, xtol=1e-9, rtol=1e-12)


# ---------------------------------------------------------------------------
# Fixtures: the expensive continuum engine runs, computed once
# ---------------------------------------------------------------------------

#: (alpha, beta, gamma, delta) points at which the full engine is run.
#: Chosen to exercise coincident (gamma = 0) and delayed switchings, and
#: positive / negative detector gap.
ENGINE_POINTS = (
    (1.0, 2.0, 0.0, 1.0),
    (2.0, 1.5, 1.0, 1.0),
    (-1.0, 2.0, 1.0, 1.0),
)


@pytest.fixture(scope="module")
def engine_states():
    """PKMM-setup engine runs, keyed by (alpha, beta, gamma, delta)."""
    return {p: pkmm_pair_state(*p) for p in ENGINE_POINTS}


@pytest.fixture(scope="module")
def lattice_setup():
    """A small 1+1 lattice field with two Gaussian-smeared detectors."""
    N = 24
    K = harmonic_chain_K(N, 0.3, bc="dirichlet")
    kernel = wightman_lattice(K)
    F_A = smearing("gaussian", 7.0, 1.2, N)
    F_B = smearing("gaussian", 15.0, 1.2, N)
    T = 0.8
    det_A = UDWDetector(switching("gaussian", T, 0.0), F_A)
    det_B = UDWDetector(switching("gaussian", T, 0.5), F_B)
    return kernel, det_A, det_B


# ===========================================================================
# 1. Conventions
# ===========================================================================

def test_pkmm_switching_convention():
    """PKMM Eq. (22) chi = e^{-(t-t0)^2/T^2} == switching('gaussian', T/sqrt2)."""
    chi = switching("gaussian", PKMM_SWITCHING_WIDTH_FACTOR, 0.0)
    t = np.linspace(-4.0, 4.0, 129)
    want = np.exp(-(t**2))  # PKMM Eq. (22) at T = 1
    err = float(np.max(np.abs(chi(t) - want)))
    assert err < 1e-15, f"PKMM Eq. (22) switching map off by {err:.3e}"
    assert PKMM_SWITCHING_WIDTH_FACTOR == pytest.approx(1.0 / math.sqrt(2.0), rel=0, abs=0)


def test_rho_layout_matches_pkmm_eq13():
    """rho has PKMM Eq. (13)'s zeros, trace 1 and Hermiticity."""
    st = UDWPairState(P_A=0.03, P_B=0.02, C=0.004 + 0.001j, M=-0.005 + 0.002j,
                      lam_A=1.0, lam_B=1.0, gap_A=1.0, gap_B=1.0)
    r = st.rho
    assert np.allclose(r, r.conj().T, atol=0.0), "rho is not Hermitian"
    assert abs(np.trace(r).real - 1.0) < 1e-15, f"trace {np.trace(r)}"
    assert abs(np.trace(r).imag) < 1e-18
    # Eq. (13): only the X-shape is populated, and rho_44 = 0 at this order.
    assert r[3, 3] == 0.0
    for i, j in [(0, 1), (0, 2), (1, 0), (1, 3), (2, 0), (2, 3), (3, 1), (3, 2)]:
        assert r[i, j] == 0.0, f"rho[{i},{j}] should vanish in Eq. (13)"
    assert r[1, 1] == st.P_A and r[2, 2] == st.P_B     # |e_A g_B>, |g_A e_B>
    assert r[1, 2] == st.C and r[3, 0] == st.M         # L_AB and M slots


def _pt_negativity_bruteforce(rho):
    """(negativity, log-negativity) of a two-qubit rho by explicit PT.

    Basis order is PKMM Eq. (14), {gg, eg, ge, ee}, i.e. index
    i = i_A + 2 i_B, so the partial transpose on B swaps the i_B labels.
    """
    r = rho.reshape(2, 2, 2, 2)              # [i_A, i_B, j_A, j_B]
    pt = np.transpose(r, (0, 3, 2, 1)).reshape(4, 4)
    ev = np.linalg.eigvalsh(0.5 * (pt + pt.conj().T))
    neg = float(-np.sum(ev[ev < 0.0]))
    return neg, float(math.log(float(np.sum(np.abs(ev)))))


def test_negativity_matches_partial_transpose_exactly():
    """With L_AB = 0 the only negative PT eigenvalue is E_1 [PKMM Eq. (66)].

    Then N = max(0, -E_1) [Eq. (67)] and E_N = ln||rho^Gamma||_1 = ln(1+2N)
    hold *exactly*, so this pins the negativity algebra with no O(lam^4)
    slack at all.
    """
    for P_A, P_B, M in [
        (0.02, 0.02, 0.03 + 0.0j),        # harvesting, identical detectors
        (0.03, 0.01, -0.02 + 0.01j),      # harvesting, asymmetric
        (0.05, 0.04, 0.004 - 0.002j),     # no harvesting (N clamps to 0)
    ]:
        st = UDWPairState(P_A=P_A, P_B=P_B, C=0.0 + 0.0j, M=M,
                          lam_A=1.0, lam_B=1.0, gap_A=1.0, gap_B=1.0)
        neg_ref, en_ref = _pt_negativity_bruteforce(st.rho)
        assert st.negativity == pytest.approx(neg_ref, abs=1e-15), (
            f"N {st.negativity} vs PT {neg_ref}"
        )
        assert st.log_negativity == pytest.approx(en_ref, abs=1e-15), (
            f"E_N {st.log_negativity} vs ln||rho^Gamma||_1 {en_ref}"
        )
        # Eq. (68) for identical detectors
        if P_A == P_B:
            assert st.negativity_estimator == pytest.approx(abs(M) - P_A, rel=1e-14)
        assert pair_negativity(P_A, P_B, M) == pytest.approx(st.negativity, abs=0.0)
        assert pair_log_negativity(P_A, P_B, M) == pytest.approx(
            st.log_negativity, abs=0.0
        )


def test_LAB_only_perturbs_negativity_at_fourth_order():
    """The extra PT eigenvalue from L_AB is O(lam^4) [PKMM text below Eq. (66)].

    PKMM point out that the naive fourth eigenvalue E_2 = -|L_AB|^2 is
    O(lam^4) and must not be counted at second order.  Here: scale the
    whole state by lam^2 and check the brute-force PT negativity minus the
    Eq. (67) value falls off like lam^4 (exponent >= 3.9 by a log-log fit).
    """
    base = dict(P_A=0.02, P_B=0.015, C=0.008 - 0.003j, M=0.03 + 0.01j)
    lams = np.array([1.0, 0.5, 0.25, 0.125])
    devs = []
    for lam in lams:
        s = lam**2
        st = UDWPairState(P_A=base["P_A"] * s, P_B=base["P_B"] * s,
                          C=base["C"] * s, M=base["M"] * s,
                          lam_A=lam, lam_B=lam, gap_A=1.0, gap_B=1.0)
        neg_ref, _ = _pt_negativity_bruteforce(st.rho)
        devs.append(abs(neg_ref - st.negativity))
    devs = np.array(devs)
    slope = np.polyfit(np.log(lams), np.log(devs), 1)[0]
    assert slope >= 3.9, f"L_AB contamination scales as lam^{slope:.2f}, expected 4"


# ===========================================================================
# 2. Single-detector transition probability vs the closed form (1e-8)
# ===========================================================================

@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0, 4.0, -1.0])
@pytest.mark.parametrize("delta", [0.25, 1.0, 2.0])
def test_pkmm_LAA_equals_m21_closed_form(alpha, delta):
    """PKMM Eq. (29) == kernels.response_gaussian_closed_form (independent derivations).

    The M2.1 closed form is derived from the mode integral in
    vacuum/detectors/kernels.py without reference to the paper; PKMM
    Eq. (29) is transcribed.  Under the switching map
    T_switching = T_PKMM/sqrt(2) they must be the same function.
    """
    got = pkmm_L_AA(alpha, delta)
    ref = response_gaussian_closed_form(
        alpha, PKMM_SWITCHING_WIDTH_FACTOR, sigma=delta
    )
    assert got == pytest.approx(ref, rel=1e-12), (
        f"Eq. (29) {got:.16e} vs M2.1 closed form {ref:.16e} "
        f"(rel {abs(got / ref - 1):.3e})"
    )


@pytest.mark.parametrize(
    "alpha,delta",
    [(0.5, 1.0), (2.0, 1.0), (1.0, 0.5), (-1.0, 1.0), (1.0, 0.1), (4.0, 1.0)],
)
def test_transition_probability_continuum_vs_closed_form(alpha, delta):
    """P_nu from the engine vs PKMM Eq. (29): spec M2.3 asks for 1e-8.

    delta = 0.1 is included on purpose: the kernel's top frequency is
    sqrt(120)/sigma ~ 110 there, an order of magnitude above the scale the
    M2.1 hazard rule sizes the time panels from, so this checks that the
    halving gate -- not the starting panel -- is what carries the accuracy.
    """
    ker = wightman_continuum_3p1(delta, distance=0.0)
    chi = switching("gaussian", PKMM_SWITCHING_WIDTH_FACTOR, 0.0)
    got = transition_probability(ker, chi, alpha, lam=1.0)
    ref = pkmm_L_AA(alpha, delta)
    rel = abs(got / ref - 1.0)
    assert rel < 1e-8, f"P_nu {got:.16e} vs Eq. (29) {ref:.16e}, rel {rel:.3e}"


def test_transition_probability_lam_squared_scaling():
    """P_nu is exactly lam^2 times the lam = 1 value [Eq. (17)]."""
    ker = wightman_continuum_3p1(1.0, distance=0.0)
    chi = switching("gaussian", PKMM_SWITCHING_WIDTH_FACTOR, 0.0)
    p1 = transition_probability(ker, chi, 1.0, lam=1.0)
    p2 = transition_probability(ker, chi, 1.0, lam=0.3)
    assert p2 == pytest.approx(0.09 * p1, rel=1e-14)


def test_transition_probability_lattice_exact_mode_sum(lattice_setup):
    """Lattice P_nu and C against the exact Gaussian mode sums.

    For Gaussian switching the time integrals of Eqs. (17)/(15) are exact:
    int chi(t) e^{i u t} dt = sqrt(2 pi) T e^{-T^2 u^2/2} e^{i u t0}, so
    P_nu = lam^2 sum_k c_k |chihat(Omega + w_k)|^2 and
    C = lam_A lam_B sum_k c_k chihat_A conj(chihat_B) with no quadrature
    anywhere.  This is a fully independent check of the lattice path.
    """
    kernel, det_A, det_B = lattice_setup
    gap, lam = 1.0, 0.05
    st = udw_pair_state(kernel, det_A, det_B, lam, gap)
    k_AA, k_BB, k_AB = pair_kernels(kernel, det_A, det_B)

    def chihat(u, chi):
        T = float(chi.width)
        return math.sqrt(2.0 * math.pi) * T * np.exp(
            -0.5 * T**2 * np.asarray(u) ** 2
        ) * np.exp(1j * np.asarray(u) * float(chi.t0))

    P_A_ref = lam**2 * float(
        np.real(np.sum(k_AA.amps * np.abs(chihat(gap + k_AA.freqs, det_A.chi)) ** 2))
    )
    P_B_ref = lam**2 * float(
        np.real(np.sum(k_BB.amps * np.abs(chihat(gap + k_BB.freqs, det_B.chi)) ** 2))
    )
    C_ref = lam**2 * complex(
        np.sum(
            k_AB.amps
            * chihat(gap + k_AB.freqs, det_A.chi)
            * np.conj(chihat(gap + k_AB.freqs, det_B.chi))
        )
    )
    assert st.P_A == pytest.approx(P_A_ref, rel=1e-12), f"{st.P_A} vs {P_A_ref}"
    assert st.P_B == pytest.approx(P_B_ref, rel=1e-12), f"{st.P_B} vs {P_B_ref}"
    assert abs(st.C - C_ref) <= 1e-12 * abs(C_ref), f"{st.C} vs {C_ref}"


# ===========================================================================
# 3. The engine against PKMM's closed forms (continuum 3+1)
# ===========================================================================

@pytest.mark.parametrize("point", ENGINE_POINTS)
def test_engine_matches_pkmm_closed_forms(engine_states, point):
    """udw_pair_state reproduces PKMM Eqs. (29), (30), (31) on their own setup."""
    alpha, beta, gamma, delta = point
    st = engine_states[point]

    P_ref = pkmm_L_AA(alpha, delta)
    for name, val in (("P_A", st.P_A), ("P_B", st.P_B)):
        rel = abs(val / P_ref - 1.0)
        assert rel < 1e-9, f"{name} {val:.16e} vs Eq. (29) {P_ref:.16e}, rel {rel:.3e}"

    C_ref = pkmm_L_AB(alpha, beta, gamma, delta)
    rel_C = abs(st.C - C_ref) / abs(C_ref)
    assert rel_C < 1e-9, f"C {st.C} vs Eq. (30) {C_ref}, rel {rel_C:.3e}"

    M_ref = pkmm_M_abs(alpha, beta, gamma, delta)
    rel_M = abs(abs(st.M) / M_ref - 1.0)
    assert rel_M < 1e-9, f"|M| {abs(st.M):.16e} vs Eq. (31) {M_ref:.16e}, rel {rel_M:.3e}"

    # Eq. (68) for these identical detectors, straight off the engine.
    n2_ref = M_ref - P_ref
    assert st.negativity_estimator == pytest.approx(n2_ref, rel=1e-8, abs=1e-18)
    assert st.meta["backend"] == "ContinuumWightman3p1"
    assert st.meta["perturbative_ok"] is True
    assert st.meta["pair"]["kernel_refinements"] >= 1
    assert st.meta["cross"]["halvings"] >= 1


def test_engine_negativity_and_log_negativity(engine_states):
    """N and E_N off the engine agree with a brute-force PT of its own rho."""
    st = engine_states[(1.0, 2.0, 0.0, 1.0)]
    neg_ref, en_ref = _pt_negativity_bruteforce(st.rho)
    # |C|^2 is the O(lam^4) contamination PKMM warn about below Eq. (66).
    slack = 4.0 * abs(st.C) ** 2
    assert abs(st.negativity - neg_ref) <= slack, (
        f"N {st.negativity:.6e} vs PT {neg_ref:.6e} (slack {slack:.3e})"
    )
    assert abs(st.log_negativity - en_ref) <= 3.0 * slack
    assert st.log_negativity > 0.0, "this point should harvest entanglement"


# ===========================================================================
# 4. PKMM closed-form self-consistency (the paper's own limits)
# ===========================================================================

@pytest.mark.parametrize("gamma", [6.0, 8.0, 10.0])
def test_M_nonoverlap_closed_form(gamma):
    """Eq. (31) -> Eq. (32) once the switchings stop overlapping."""
    alpha, beta, delta = 1.0, 2.0, 1.0
    quad = pkmm_M_abs(alpha, beta, gamma, delta)
    closed = pkmm_M_abs_nonoverlap(alpha, beta, gamma, delta)
    rel = abs(quad / closed - 1.0)
    assert rel < 1e-8, f"Eq. (31) {quad:.12e} vs Eq. (32) {closed:.12e}, rel {rel:.3e}"


def test_M_nonoverlap_rejects_overlapping_switchings():
    with pytest.raises(ValueError, match="non-overlapping"):
        pkmm_M_abs_nonoverlap(1.0, 2.0, 1.0, 1.0)
    assert PKMM_NONOVERLAP_GAMMA == pytest.approx(7.0 / math.sqrt(2.0), rel=1e-15)


def test_M_coincident_pointlike_limit():
    """Eq. (31) -> Eq. (37) as delta -> 0 at gamma = 0, at the expected rate."""
    alpha, beta = 1.0, 2.0
    closed = pkmm_M_abs_coincident_pointlike(alpha, beta)
    errs = []
    for delta in (0.08, 0.04, 0.02):
        errs.append(abs(pkmm_M_abs(alpha, beta, 0.0, delta) / closed - 1.0))
    assert errs[-1] < 1e-4, f"Eq. (31) at delta=0.02 vs Eq. (37): rel {errs[-1]:.3e}"
    # O(delta^2) approach: halving delta should quarter the error.
    for a, b in zip(errs[:-1], errs[1:]):
        assert 3.0 < a / b < 5.0, f"delta^2 convergence broken: ratio {a / b:.2f}"


def test_M_coincident_pointlike_no_overflow():
    """Eq. (37)'s erfi(beta/sqrt2) overflows past beta ~ 37; the product must not."""
    val = pkmm_M_abs_coincident_pointlike(1.0, 60.0)
    assert np.isfinite(val) and val > 0.0, f"|M_coinc| at beta=60 came out {val}"
    # e^{-beta^2/2} erfi(beta/sqrt2) -> sqrt(2/pi)/beta for large beta, so
    # |M_coinc| -> lam^2 e^{-alpha^2/2} / (4 pi beta^2).
    asymptote = math.exp(-0.5) / (4.0 * math.pi * 60.0**2)
    assert val == pytest.approx(asymptote, rel=2e-3), f"{val} vs {asymptote}"


def test_M_and_L_lambda_scaling():
    """Every element of Eq. (13) is exactly quadratic in the coupling."""
    args = (1.0, 2.0, 1.0, 1.0)
    assert pkmm_L_AA(1.0, 1.0, lam=0.2) == pytest.approx(
        0.04 * pkmm_L_AA(1.0, 1.0), rel=1e-14
    )
    assert pkmm_M_abs(*args, lam=0.2) == pytest.approx(0.04 * pkmm_M_abs(*args), rel=1e-12)
    assert pkmm_L_AB(*args, lam=0.2) == pytest.approx(0.04 * pkmm_L_AB(*args), rel=1e-14)


def test_pkmm_M_abs_rejects_pointlike():
    """delta = 0 makes Eq. (31) only conditionally convergent -- refuse it."""
    with pytest.raises(ValueError, match="conditional convergence"):
        pkmm_M_abs(1.0, 2.0, 0.0, 0.0)


def test_pointlike_plateau():
    """N^(2) is flat below sigma/T ~ 0.05, so delta = 0.02 stands in for pointlike.

    PKMM Fig. 3: "the behaviour of the detector quickly goes to the
    pointlike limit when sigma/T << 1".
    """
    ref = pkmm_negativity_estimator(5.5, 4.0, FIG2_GAMMA, POINTLIKE_DELTA)
    finer = pkmm_negativity_estimator(5.5, 4.0, FIG2_GAMMA, 0.01)
    assert abs(ref / finer - 1.0) < 2e-4, (
        f"pointlike plateau not reached: {ref:.6e} vs {finer:.6e}"
    )


# ===========================================================================
# 5. PKMM Fig. 2 anchors: death lines vs separation and gap (2%)
# ===========================================================================

def test_fig2_lightcone_calibration():
    """The dashed light-cone marker of Fig. 2 a.1/b.1 fixes the d/T axis.

    Measured from the panel content stream: the dashed line sits at
    d/T = 7.94975 (see the provenance note at the top of this file); the
    caption places it at Delta = d - 7T/sqrt(2) with Delta/T = 3.
    """
    measured = (353.301 - 7.521106) / 434.957787 * 10.0
    expected = FIG2_GAMMA + PKMM_NONOVERLAP_GAMMA
    assert measured == pytest.approx(expected, abs=1e-4), (
        f"digitized light-cone marker {measured:.5f} vs 3 + 7/sqrt(2) = {expected:.5f}"
    )


def test_pkmm_fig2_b1_pointlike_death_line():
    """Harvesting boundary d*(Omega T) of PKMM Fig. 2 b.1, to 2%.

    Pointlike detectors, Gaussian switching, Delta/T = 3, 3+1D.
    """
    worst = 0.0
    for alpha, d_digitized in PKMM_FIG2_B1_POINTLIKE:
        d_model = _death_separation(alpha, FIG2_GAMMA, POINTLIKE_DELTA)
        rel = abs(d_model / d_digitized - 1.0)
        worst = max(worst, rel)
        assert rel < DIGITIZATION_RTOL, (
            f"Fig. 2 b.1 at Omega T={alpha}: digitized d/T={d_digitized}, "
            f"model {d_model:.4f}, rel {rel:.4f}"
        )
    assert worst < DIGITIZATION_RTOL, f"worst Fig. 2 b.1 deviation {worst:.4f}"


def test_pkmm_fig2_a1_smeared_death_line():
    """Same for PKMM Fig. 2 a.1, sigma/T = 1, using the b.1 axis calibration.

    The vertical calibration is carried over unchanged from the pointlike
    panel, so this is a genuine prediction of a different curve, not a
    refit.
    """
    worst = 0.0
    for alpha, d_digitized in PKMM_FIG2_A1_SMEARED:
        d_model = _death_separation(alpha, FIG2_GAMMA, 1.0)
        rel = abs(d_model / d_digitized - 1.0)
        worst = max(worst, rel)
        assert rel < DIGITIZATION_RTOL, (
            f"Fig. 2 a.1 at Omega T={alpha}: digitized d/T={d_digitized}, "
            f"model {d_model:.4f}, rel {rel:.4f}"
        )
    assert worst < DIGITIZATION_RTOL, f"worst Fig. 2 a.1 deviation {worst:.4f}"
    # The Omega T = 9.1 row runs off the right edge of the frame (digitized
    # 9.998 out of a 10-wide axis), so it only bounds the model from below.
    assert _death_separation(9.1, FIG2_GAMMA, 1.0) > 9.99


def test_pkmm_fig2_smeared_below_pointlike_at_small_gap():
    """Panels a.1 and b.1 cross: the anchors pin the ordering, not just the size.

    Digitized: at Omega T = 1.9 the sigma/T = 1 region is *smaller* than
    the pointlike one (3.591 < 3.671), while at Omega T = 8.2 it is
    *larger* (9.233 > 8.983).  A convention slip that rescaled the
    smearing would not reproduce a sign change.
    """
    small_smeared = _death_separation(1.9, FIG2_GAMMA, 1.0)
    small_point = _death_separation(1.9, FIG2_GAMMA, POINTLIKE_DELTA)
    big_smeared = _death_separation(8.2, FIG2_GAMMA, 1.0)
    big_point = _death_separation(8.2, FIG2_GAMMA, POINTLIKE_DELTA)
    assert small_smeared < small_point, f"{small_smeared} !< {small_point}"
    assert big_smeared > big_point, f"{big_smeared} !> {big_point}"


# ===========================================================================
# 6. Structure asserted exactly (separation / gap / smearing)
# ===========================================================================

def test_entanglement_death_with_separation():
    """N^(2) falls monotonically with d/T and dies; N and E_N clamp to 0.

    PKMM Fig. 2: the harvesting region is bounded in d/T at fixed gap.
    """
    alpha, gamma, delta = 5.5, FIG2_GAMMA, 1.0
    betas = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0])
    n2 = np.array([pkmm_negativity_estimator(alpha, b, gamma, delta) for b in betas])
    assert np.all(np.diff(n2) < 0.0), f"N^(2) not monotone in d/T: {n2}"
    assert n2[0] > 0.0 and n2[-1] < 0.0, f"no death line in this range: {n2}"
    # Past the death line the clamped negativity and E_N are exactly zero.
    P = pkmm_L_AA(alpha, delta)
    M_dead = pkmm_M_abs(alpha, betas[-1], gamma, delta)
    assert pair_negativity(P, P, M_dead) == 0.0
    assert pair_log_negativity(P, P, M_dead) == 0.0


def test_gap_dependence_direction():
    """Bigger gap => bigger harvesting region but less harvested entanglement.

    PKMM Sec. IV.A: "for Gaussianly switched detectors, when the energy gap
    increases the total amount of entanglement decreases rapidly, but in
    exchange the region of nonzero negativity increases, eventually
    'leaking' arbitrarily much into the spacelike-separation region."
    """
    gamma, delta, beta_probe = FIG2_GAMMA, 1.0, 1.0
    alphas = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    d_star = [_death_separation(a, gamma, delta) for a in alphas]
    peak = [pkmm_negativity_estimator(a, beta_probe, gamma, delta) for a in alphas]
    assert all(b > a for a, b in zip(d_star[:-1], d_star[1:])), (
        f"harvesting region not growing with the gap: {d_star}"
    )
    assert all(b < a for a, b in zip(peak[:-1], peak[1:])), (
        f"harvested N^(2) not shrinking with the gap: {peak}"
    )
    # ... and the region eventually leaks past the light cone at
    # d/T = Delta/T + 7/sqrt(2) (Fig. 2 caption's spacelike boundary).
    spacelike = gamma + PKMM_NONOVERLAP_GAMMA
    assert d_star[0] < spacelike < d_star[-1], (
        f"spacelike harvesting onset not bracketed: {d_star[0]}, {spacelike}, {d_star[-1]}"
    )
    assert pkmm_negativity_estimator(7.0, spacelike + 0.2, gamma, delta) > 0.0
    assert pkmm_negativity_estimator(2.0, spacelike + 0.2, gamma, delta) < 0.0


def test_smearing_dependence_direction():
    """Bigger detectors harvest less (PKMM Fig. 3), with a pointlike plateau.

    PKMM Fig. 3 plots N^(2) against sigma/T at Omega T = 3.8, 4.8 and 5.8
    (read off the panel labels of ``detecsizeomega{3,4,5}-fig.pdf``), and
    Sec. IV.A states "as the sizes of the detectors grow, they become less
    efficient to harvest vacuum entanglement".  *No spot value is taken
    from that figure*: neither its caption nor the surrounding text gives
    the d/T and Delta/T it was made at, so its ordinate is not
    reproducible and reading a number off it would be unverifiable.  What
    is testable exactly -- and is tested here -- is the structure: a flat
    pointlike plateau for sigma/T << 1, then a monotone decrease.  The
    *quantitative* smearing anchor is the pair of Fig. 2 death lines
    above, which do fix all their parameters.
    """
    alpha, beta, gamma = 5.8, 2.0, FIG2_GAMMA   # Omega T = 5.8 is Fig. 3 panel c
    deltas = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    n2 = [pkmm_negativity_estimator(alpha, beta, gamma, d) for d in deltas]
    assert all(b < a for a, b in zip(n2[:-1], n2[1:])), (
        f"N^(2) not decreasing with detector size: {n2}"
    )
    # Fig. 3's flat left half: nothing moves between sigma/T = 0.05 and 0.1.
    assert abs(n2[1] / n2[0] - 1.0) < 5e-3, f"plateau too steep: {n2[0]}, {n2[1]}"
    # ... and the fall is substantial by sigma/T = 1.
    assert n2[-1] < 0.85 * n2[0], f"no visible size penalty: {n2[0]} -> {n2[-1]}"


# ===========================================================================
# 7. Lattice backend and the time-ordered triangle
# ===========================================================================

def test_lattice_pair_term_vs_triangle_rule(lattice_setup):
    """M from ``pair_term`` vs a direct sum on ``kernels.triangle_rule``.

    ``pair_term`` uses a cumulative composite inner rule; ``triangle_rule``
    builds an independent tensor-product rule on the same triangle.  Both
    discretize PKMM Eq. (18); agreement to 1e-9 relative rules out an
    error in the time-ordering, the phases or the A<->B sum.
    """
    kernel, det_A, det_B = lattice_setup
    _, _, k_AB = pair_kernels(kernel, det_A, det_B)
    gap_A, gap_B = 1.0, 1.3
    chi_A, chi_B = det_A.chi, det_B.chi
    a = min(chi_A.support[0], chi_B.support[0])
    b = max(chi_A.support[1], chi_B.support[1])

    def reference(max_panel):
        t, tp, w = triangle_rule(a, b, max_panel, n_gl=16)
        W = k_AB(t - tp)
        integrand = W * (
            chi_A(t) * chi_B(tp) * np.exp(1j * (gap_A * t + gap_B * tp))
            + chi_B(t) * chi_A(tp) * np.exp(1j * (gap_B * t + gap_A * tp))
        )
        return -complex(np.sum(w * integrand))

    ref = reference(0.85)
    ref_fine = reference(0.45)
    assert abs(ref_fine - ref) <= 1e-11 * abs(ref), "the reference rule is not converged"

    got = pair_term(k_AB, chi_A, chi_B, gap_A, gap_B)
    rel = abs(got - ref_fine) / abs(ref_fine)
    assert rel < 1e-9, f"M {got} vs triangle_rule {ref_fine}, rel {rel:.3e}"


def test_pair_term_quadrature_settings_are_converged(lattice_setup):
    """The default rule is at its limit: forcing finer settings moves nothing.

    Three knobs are pushed independently -- a 4x finer starting panel
    (``panel0``, which shortcuts the halving gate), a higher outer GL order
    and a higher inner order for the cumulative rule.  If any of the
    defaults were under-resolved these would disagree.
    """
    kernel, det_A, det_B = lattice_setup
    _, _, k_AB = pair_kernels(kernel, det_A, det_B)
    chi_A, chi_B = det_A.chi, det_B.chi
    base = pair_term(k_AB, chi_A, chi_B, 1.0, 1.3)
    for label, kwargs in [
        ("panel0", dict(panel0=0.02)),
        ("n_gl", dict(n_gl=24)),
        ("n_gl_inner", dict(n_gl_inner=8)),
    ]:
        got = pair_term(k_AB, chi_A, chi_B, 1.0, 1.3, **kwargs)
        rel = abs(got - base) / abs(base)
        assert rel < 1e-11, f"M moved by {rel:.3e} when tightening {label}"


def test_lattice_cross_term_vs_double_quadrature(lattice_setup):
    """C from ``cross_term`` vs a plain product-rule double integral of Eq. (15)."""
    kernel, det_A, det_B = lattice_setup
    _, _, k_AB = pair_kernels(kernel, det_A, det_B)
    gap_A, gap_B = 1.0, 1.3
    chi_A, chi_B = det_A.chi, det_B.chi
    tA, wA = gl_panels(chi_A.support[0], chi_A.support[1], 0.4, n_gl=16)
    tB, wB = gl_panels(chi_B.support[0], chi_B.support[1], 0.4, n_gl=16)
    W = k_AB(tB[:, None] - tA[None, :])                 # W_AB(t' - t)
    fA = wA * chi_A(tA) * np.exp(1j * gap_A * tA)
    fB = wB * chi_B(tB) * np.exp(-1j * gap_B * tB)
    ref = complex(fB @ W @ fA)

    got = cross_term(k_AB, chi_A, chi_B, gap_A, gap_B)
    rel = abs(got - ref) / abs(ref)
    assert rel < 1e-10, f"C {got} vs double quadrature {ref}, rel {rel:.3e}"


def test_lattice_pair_state_runs_and_scales(lattice_setup):
    """Both backends go through ``udw_pair_state``; couplings scale exactly."""
    kernel, det_A, det_B = lattice_setup
    st1 = udw_pair_state(kernel, det_A, det_B, lam=0.05, gap=1.0)
    st2 = udw_pair_state(kernel, det_A, det_B, lam=(0.05, 0.10), gap=(1.0, 1.0))
    assert st1.meta["backend"] == "LatticeWightman"
    assert st1.meta["pair"]["kernel_refinements"] == 0, (
        "the lattice normal-mode sum is exact -- nothing to refine"
    )
    assert st2.P_A == pytest.approx(st1.P_A, rel=1e-13)
    assert st2.P_B == pytest.approx(4.0 * st1.P_B, rel=1e-13)
    assert abs(st2.M - 2.0 * st1.M) <= 1e-12 * abs(st1.M)
    assert abs(st2.C - 2.0 * st1.C) <= 1e-12 * abs(st1.C)
    assert st1.meta["perturbative_ok"] is True
    r = st1.rho
    assert abs(np.trace(r).real - 1.0) < 1e-14


def test_lattice_ab_relabelling_symmetry(lattice_setup):
    """Swapping (A, B) swaps P_A/P_B, conjugates C and leaves |M| alone."""
    kernel, det_A, det_B = lattice_setup
    fwd = udw_pair_state(kernel, det_A, det_B, lam=0.05, gap=(1.0, 1.3))
    rev = udw_pair_state(kernel, det_B, det_A, lam=0.05, gap=(1.3, 1.0))
    assert rev.P_A == pytest.approx(fwd.P_B, rel=1e-12)
    assert rev.P_B == pytest.approx(fwd.P_A, rel=1e-12)
    assert abs(rev.C - np.conj(fwd.C)) <= 1e-12 * abs(fwd.C)
    # Eq. (18) is symmetric under A<->B by construction (the two summands swap).
    assert abs(rev.M - fwd.M) <= 1e-11 * abs(fwd.M)
    assert rev.negativity == pytest.approx(fwd.negativity, rel=1e-11, abs=1e-20)


def test_lattice_spacelike_pair_smaller_than_contact(lattice_setup):
    """Pushing the detectors apart on the chain kills the pair term."""
    kernel, det_A, _ = lattice_setup
    N = 24
    near = UDWDetector(switching("gaussian", 0.8, 0.0), smearing("gaussian", 9.0, 1.2, N))
    far = UDWDetector(switching("gaussian", 0.8, 0.0), smearing("gaussian", 20.0, 1.2, N))
    m_near = abs(udw_pair_state(kernel, det_A, near, 0.05, 1.0).M)
    m_far = abs(udw_pair_state(kernel, det_A, far, 0.05, 1.0).M)
    assert m_far < 0.2 * m_near, f"|M| near {m_near:.3e}, far {m_far:.3e}"


# ===========================================================================
# 8. Dispatch, validation, immutability
# ===========================================================================

def test_pair_kernels_dispatch_errors(lattice_setup):
    kernel, det_A, det_B = lattice_setup
    bare = UDWDetector(switching("gaussian", 0.8, 0.0))
    with pytest.raises(ValueError, match="smearing profile"):
        pair_kernels(kernel, bare, det_B)
    cont = wightman_continuum_3p1(1.0, distance=2.0)
    with pytest.raises(ValueError, match="must not carry"):
        pair_kernels(cont, det_A, det_B)
    with pytest.raises(TypeError, match="kernel must be"):
        pair_kernels("not a kernel", bare, bare)
    # The (W_AA, W_BB, W_AB) escape hatch (used by M2.5's communication split).
    triple = pair_kernels(kernel, det_A, det_B)
    assert pair_kernels(triple, det_A, det_B) == triple


def test_udw_pair_state_rejects_unknown_options(lattice_setup):
    kernel, det_A, det_B = lattice_setup
    with pytest.raises(TypeError, match="unknown quadrature options"):
        udw_pair_state(kernel, det_A, det_B, 0.05, 1.0, nonsense=1)


def test_udw_detector_is_frozen():
    det = UDWDetector(switching("gaussian", 1.0, 0.0))
    with pytest.raises(AttributeError):
        det.chi = None
    with pytest.raises(TypeError, match="Switching-like"):
        UDWDetector(lambda t: t)


def test_pair_params_validation(lattice_setup):
    kernel, det_A, det_B = lattice_setup
    with pytest.raises(ValueError, match="lam must be"):
        udw_pair_state(kernel, det_A, det_B, [0.1, 0.2, 0.3], 1.0)
    with pytest.raises(ValueError, match="gap must be"):
        udw_pair_state(kernel, det_A, det_B, 0.1, [1.0, 2.0, 3.0])


def test_pkmm_closed_form_domain_guards():
    with pytest.raises(ValueError, match="beta must be"):
        pkmm_L_AB(1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="beta must be"):
        pkmm_M_abs(1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="beta must be"):
        pkmm_M_abs_coincident_pointlike(1.0, 0.0)
    with pytest.raises(ValueError, match="delta must be >= 0"):
        pkmm_L_AA(1.0, -0.5)


# ---------------------------------------------------------------------------
# The quadrature gate's failure diagnostic must actually carry information
#
# Found by the M2.7 sweep: _doubling_gate assigned `prev = val` at the end of
# every non-returning iteration, so on exhaustion `prev is val` and the
# RuntimeError always reported "last change 0.000e+00" — the one number a
# caller needs in order to tell "nearly converged, raise max_doublings" from
# "diverging, fix the setup" from "analytically zero, pass atol".
# ---------------------------------------------------------------------------

def test_doubling_gate_failure_reports_the_real_last_change():
    from vacuum.detectors.udw import _doubling_gate

    # An evaluator that keeps moving by a fixed, known amount per halving.
    step = 0.125
    calls = {"n": 0}

    def never_converges(h):
        calls["n"] += 1
        return 1.0 + step * calls["n"]

    with pytest.raises(RuntimeError) as exc:
        _doubling_gate(never_converges, 1.0, rtol=1e-12, atol=0.0,
                       max_doublings=4, what="canary")
    msg = str(exc.value)
    assert "canary" in msg and "4 halvings" in msg, msg
    # the message must quote the true last movement, not 0
    assert f"{step:.3e}" in msg, (
        f"diagnostic lost the last change (expected {step:.3e}): {msg}"
    )
    assert "0.000e+00" not in msg, f"stale zero-change diagnostic: {msg}"


def test_doubling_gate_failure_on_an_analytically_zero_integral_is_legible():
    """The M2.5 hazard: a value that IS zero can never meet a relative test."""
    from vacuum.detectors.udw import _doubling_gate

    seq = iter([1e-19, 3e-20, 1e-20, 4e-21, 1e-21, 3e-22])

    with pytest.raises(RuntimeError) as exc:
        _doubling_gate(lambda h: next(seq), 1.0, rtol=1e-8, atol=1e-30,
                       max_doublings=6, what="commutator_term")
    msg = str(exc.value)
    assert "pass atol" in msg, f"no guidance toward an absolute scale: {msg}"
    assert "0.000e+00" not in msg, f"stale zero-change diagnostic: {msg}"


def test_doubling_gate_still_converges_and_reports_its_level():
    from vacuum.detectors.udw import _doubling_gate

    def converging(h):
        return 2.0 + h ** 2

    val, h, n = _doubling_gate(converging, 1.0, rtol=1e-9, atol=0.0,
                               max_doublings=40, what="ok")
    assert n >= 1 and h == pytest.approx(0.5 ** n)
    assert val == pytest.approx(2.0 + h * h, rel=1e-15), (
        f"converged value {val!r} at h = {h!r} after {n} halvings"
    )
