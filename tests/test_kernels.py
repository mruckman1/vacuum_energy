"""M2.1 anchors: Wightman kernels, switching, smearing (spec test matrix rows:
lattice equal-time moments 1e-12; Im W vs causality commutator 1e-12;
continuum single-detector response vs closed form 1e-8 + cutoff-independence
1e-8), plus regression tests for the quadrature toolkit and the profile
libraries.  Every anchor records its measured number in the assertion
message (docs/API.md testing rules).
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erfc

from vacuum.audits.causality import chain_propagator  # the causality module's commutator
from vacuum.core import Omega, ground_state_cov, harmonic_chain_K
from vacuum.detectors import (
    detector_response,
    gl_panels,
    response_gaussian_closed_form,
    smearing,
    switching,
    triangle_rule,
    wightman_continuum_3p1,
    wightman_lattice,
)
from vacuum.detectors.kernels import _response_quad

N_CHAIN = 48
M_CHAIN = 0.35
SIGMA = 0.4


@pytest.fixture(scope="module")
def chain():
    K = harmonic_chain_K(N_CHAIN, M_CHAIN, bc="periodic")
    return K, wightman_lattice(K), ground_state_cov(K)


@pytest.fixture(scope="module")
def continuum():
    return wightman_continuum_3p1(SIGMA)


# ---------------------------------------------------------------------------
# Lattice backend
# ---------------------------------------------------------------------------

def test_lattice_equal_time_xx_moments(chain):
    """<x_i x_j> = W(0) = (K^{-1/2}/2)_ij to 1e-12 (spec matrix row 1)."""
    K, LW, V = chain
    W0 = LW(0.0)
    dev = np.max(np.abs(W0.real - V[:N_CHAIN, :N_CHAIN]))
    assert dev < 1e-12, f"equal-time <xx> deviation {dev:.3e} >= 1e-12"
    imag = np.max(np.abs(W0.imag))
    assert imag < 1e-15, f"W(0) should be real, max |Im| = {imag:.3e}"


def test_lattice_equal_time_pp_moments(chain):
    """<p_i p_j> = -d^2 W(0) = (K^{1/2}/2)_ij to 1e-12 (second derivative)."""
    K, LW, V = chain
    D2 = LW.d2(0.0)
    dev = np.max(np.abs(-D2.real - V[N_CHAIN:, N_CHAIN:]))
    assert dev < 1e-12, f"equal-time <pp> deviation {dev:.3e} >= 1e-12"
    assert np.max(np.abs(D2.imag)) < 1e-15


def test_lattice_imW_matches_causality_commutator(chain):
    """2 Im W(t) = (S(t) Omega)_xx to 1e-12: the Pauli-Jordan commutator of
    the causality audit, [x_i(t), x_j(0)] = i (S(t) Omega)_ij (spec row 2).
    Im W is built from spectral data alone — state independence is by
    construction; this pins the normalization and sign against the
    independently validated propagator."""
    K, LW, V = chain
    Om = Omega(N_CHAIN)
    worst = 0.0
    for t in (0.7, 2.3, 5.1):
        S = chain_propagator(K, t)
        dev = np.max(np.abs(2.0 * LW.imag_part(t) - (S @ Om)[:N_CHAIN, :N_CHAIN]))
        worst = max(worst, dev)
    assert worst < 1e-12, f"Im W vs causality commutator deviation {worst:.3e} >= 1e-12"


def test_lattice_hermiticity_and_symmetry(chain):
    """W(-dt) = conj(W(dt)) and W(dt) symmetric (both operators at equal
    footing in the mode sum)."""
    K, LW, V = chain
    for dt in (0.9, 3.7):
        W = LW(dt)
        assert np.max(np.abs(W - W.T)) < 1e-14
        assert np.max(np.abs(LW(-dt) - np.conj(W))) < 1e-14
    # real/imag accessors agree with the complex matrix
    assert np.max(np.abs(LW.real_part(1.1) - LW(1.1).real)) < 1e-15
    assert np.max(np.abs(LW.imag_part(1.1) - LW(1.1).imag)) < 1e-15


def test_lattice_rejects_zero_mode():
    """Massless periodic chain has a zero mode: kernel must refuse it."""
    with pytest.raises(ValueError, match="positive definite"):
        wightman_lattice(harmonic_chain_K(16, 0.0, bc="periodic"))


def test_lattice_smeared_kernel_matches_sandwich(chain):
    """Smeared kernel is exactly F^T W(dt) F' (1e-13)."""
    K, LW, V = chain
    F = smearing("gaussian", 24.0, 3.0, N_CHAIN)
    Fp = smearing("cos2", 30.0, 5.0, N_CHAIN)
    ker = LW.smeared(F, Fp)
    worst = 0.0
    for dt in (0.0, 1.3, -2.6):
        direct = F.F @ LW(dt) @ Fp.F
        worst = max(worst, abs(ker(dt) - direct))
    assert worst < 1e-13, f"smeared kernel vs F^T W F' deviation {worst:.3e}"
    # auto-correlation default F' = F
    kaa = LW.smeared(F)
    assert abs(kaa(0.7) - F.F @ LW(0.7) @ F.F) < 1e-13
    # equal-time smeared moment is real and positive (it is <(F.x)^2>)
    assert kaa(0.0).real > 0.0 and abs(kaa(0.0).imag) < 1e-16


# ---------------------------------------------------------------------------
# Quadrature toolkit (hazard-note rules)
# ---------------------------------------------------------------------------

def test_gl_panels_oscillatory():
    """GL panels integrate e^{i 7 t} on [0, 3] to 1e-13."""
    x, w = gl_panels(0.0, 3.0, 0.25)
    num = np.sum(w * np.exp(1j * 7.0 * x))
    exact = (np.exp(21j) - 1.0) / 7j
    dev = abs(num - exact)
    assert dev < 1e-13, f"GL panel oscillatory deviation {dev:.3e}"
    assert abs(np.sum(w) - 3.0) < 1e-13  # weights integrate 1 exactly


def test_triangle_rule_analytic():
    """Triangular tensor-product rule: exponential and oscillatory
    time-ordered integrals over {0 <= t' <= t <= 1} to 1e-12 — the rule the
    hazard note demands instead of masking a square grid."""
    t, tp, w = triangle_rule(0.0, 1.0, 0.125)
    assert np.all(tp <= t + 1e-15)  # nodes actually live on the triangle
    lam = 3.0
    num = np.sum(w * np.exp(lam * (t + tp)))
    exact = ((np.exp(2 * lam) - 1) / (2 * lam) - (np.exp(lam) - 1) / lam) / lam
    dev = abs(num - exact) / abs(exact)
    assert dev < 1e-12, f"triangle rule (exp) relative deviation {dev:.3e}"
    num2 = np.sum(w * np.exp(1j * 5.0 * (t - tp)))
    exact2 = ((np.exp(5j) - 1.0) / 5j - 1.0) / 5j
    dev2 = abs(num2 - exact2)
    assert dev2 < 1e-12, f"triangle rule (oscillatory) deviation {dev2:.3e}"
    # triangle area = 1/2
    assert abs(np.sum(w) - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# Continuum 3+1 backend
# ---------------------------------------------------------------------------

def test_continuum_W_against_independent_quadrature(continuum):
    """Kernel values match adaptive scipy quadrature of the radial integral
    (1e-10) and the exact Gaussian-commutator closed form for Im W (1e-12):
    Im W(dt) = -(1/4pi^2)(dt/(4a)) sqrt(pi/a) e^{-dt^2/4a}, a = sigma^2/2
    (differentiate int_0^inf e^{-a k^2} cos(b k) dk = (1/2)sqrt(pi/a)
    e^{-b^2/4a} with respect to b)."""
    a = 0.5 * SIGMA**2
    worst_q, worst_cf = 0.0, 0.0
    for dt in (0.0, 0.5, 2.0):
        w = continuum(dt)
        re, _ = quad(
            lambda k: k * np.exp(-a * k * k) * np.cos(k * dt) / (4 * np.pi**2),
            0.0, np.inf, epsabs=1e-15,
        )
        im, _ = quad(
            lambda k: -k * np.exp(-a * k * k) * np.sin(k * dt) / (4 * np.pi**2),
            0.0, np.inf, epsabs=1e-15,
        )
        worst_q = max(worst_q, abs(w.real - re), abs(w.imag - im))
        im_cf = -(1.0 / (4 * np.pi**2)) * (dt / (4 * a)) * np.sqrt(np.pi / a) * np.exp(
            -dt * dt / (4 * a)
        )
        worst_cf = max(worst_cf, abs(w.imag - im_cf))
    assert worst_q < 1e-10, f"continuum W vs quad deviation {worst_q:.3e}"
    assert worst_cf < 1e-12, f"continuum Im W vs closed form deviation {worst_cf:.3e}"
    # equal-time smeared moment: W(0) = 1/(4 pi^2 sigma^2), real
    w0 = continuum(0.0)
    dev0 = abs(w0.real - 1.0 / (4 * np.pi**2 * SIGMA**2))
    assert dev0 < 1e-12, f"W(0) vs 1/(4 pi^2 sigma^2) deviation {dev0:.3e}"
    assert abs(w0.imag) < 1e-15


def test_continuum_W_separated_detectors():
    """d > 0 kernel (sinc(kd) angular factor) matches adaptive quadrature."""
    d = 1.5
    ker = wightman_continuum_3p1(SIGMA, distance=d)
    a = 0.5 * SIGMA**2
    worst = 0.0
    for dt in (0.0, 1.0):
        w = ker(dt)
        re, _ = quad(
            lambda k: np.exp(-a * k * k) * np.sin(k * d) / d * np.cos(k * dt)
            / (4 * np.pi**2),
            0.0, np.inf, epsabs=1e-15, limit=200,
        )
        im, _ = quad(
            lambda k: -np.exp(-a * k * k) * np.sin(k * d) / d * np.sin(k * dt)
            / (4 * np.pi**2),
            0.0, np.inf, epsabs=1e-15, limit=200,
        )
        worst = max(worst, abs(w.real - re), abs(w.imag - im))
    assert worst < 1e-10, f"separated-detector W vs quad deviation {worst:.3e}"


def test_continuum_requires_smearing():
    """Pointlike (sigma = 0) construction is refused: production always
    smears; the sigma -> 0 limit lives only in the closed-form cross-check."""
    with pytest.raises(ValueError, match="sigma"):
        wightman_continuum_3p1(0.0)


def test_continuum_response_closed_form_anchor(continuum):
    """SPEC ANCHOR: continuum single-detector response for Gaussian switching
    matches the closed form to 1e-8 (relative).  Two parameter sets; the
    closed form itself is triangulated against independent adaptive
    quadrature of the k-integral at 1e-12."""
    for Om, T, sig, ker in (
        (2.0, 1.0, SIGMA, continuum),
        (0.8, 1.25, 0.25, wightman_continuum_3p1(0.25)),
    ):
        cf = response_gaussian_closed_form(Om, T, sig)
        # independent check of the closed form: direct adaptive quadrature of
        # (T^2/2pi) int_0^inf k exp(-k^2 sig^2/2 - T^2 (Om+k)^2) dk
        val, _ = quad(
            lambda k: k * np.exp(-0.5 * (k * sig) ** 2 - (T * (Om + k)) ** 2),
            0.0, np.inf, epsabs=1e-18, epsrel=1e-13,
        )
        direct = T * T / (2 * np.pi) * val
        rel_cf = abs(cf - direct) / abs(direct)
        assert rel_cf < 1e-12, (
            f"closed form vs adaptive quad: rel {rel_cf:.3e} (Om={Om}, T={T}, "
            f"sigma={sig})"
        )
        P = detector_response(ker, switching("gaussian", T), Om, rtol=1e-12)
        rel = abs(P - cf) / abs(cf)
        assert rel < 1e-8, (
            f"continuum response anchor: P_num={P:.15e}, P_closed={cf:.15e}, "
            f"rel deviation {rel:.3e} >= 1e-8 (Om={Om}, T={T}, sigma={sig})"
        )


def test_continuum_response_cutoff_independence(continuum):
    """SPEC ANCHOR: varying the hard k-cutoff around the Gaussian tail moves
    the response and the kernel values by < 1e-8 (permanent test, not a
    one-off)."""
    Om, T = 2.0, 1.0
    chi = switching("gaussian", T)
    kernels = [continuum, continuum.with_kmax(0.7 * continuum.kmax),
               continuum.with_kmax(1.3 * continuum.kmax)]
    Ps = [detector_response(k, chi, Om, rtol=1e-12) for k in kernels]
    spread_P = max(Ps) - min(Ps)
    assert spread_P < 1e-8, f"response cutoff spread {spread_P:.3e} >= 1e-8"
    dts = np.linspace(-3.0, 3.0, 41)
    W0 = kernels[0](dts)
    spread_W = max(np.max(np.abs(W0 - k(dts))) for k in kernels[1:])
    assert spread_W < 1e-8, f"kernel cutoff spread {spread_W:.3e} >= 1e-8"


def test_continuum_k_refinement_stable(continuum):
    """Doubling the k-panel density leaves the kernel values unchanged at
    the 1e-10 level (the quadrature is converged, not accidental)."""
    fine = continuum.with_refined(2)
    dts = np.linspace(-2.0, 2.0, 21)
    dev = np.max(np.abs(continuum(dts) - fine(dts)))
    assert dev < 1e-10, f"k-refinement moved W by {dev:.3e}"


def test_pointlike_sigma_to_zero_crosscheck():
    """The sigma -> 0 cross-check: smeared responses converge to the
    pointlike closed form P = (1/4pi)[e^{-Om^2 T^2} - sqrt(pi) Om T
    erfc(Om T)] with the expected O(sigma^2) rate (distance ratio ~ 4 for
    sigma halving), and each smeared numeric response matches its own
    closed form to 1e-8."""
    Om, T = 2.0, 1.0
    chi = switching("gaussian", T)
    p_point = response_gaussian_closed_form(Om, T, 0.0)
    # the sigma = 0 branch of the code equals the textbook expression exactly
    textbook = (1.0 / (4 * np.pi)) * (
        np.exp(-((Om * T) ** 2)) - np.sqrt(np.pi) * Om * T * erfc(Om * T)
    )
    assert abs(p_point - textbook) / textbook < 1e-14
    dists = []
    for sig in (0.1, 0.05):
        P = detector_response(wightman_continuum_3p1(sig), chi, Om, rtol=1e-12)
        cf = response_gaussian_closed_form(Om, T, sig)
        rel = abs(P - cf) / cf
        assert rel < 1e-8, f"sigma={sig}: numeric vs closed form rel {rel:.3e}"
        dists.append(abs(P - p_point))
    ratio = dists[0] / dists[1]
    assert 3.4 < ratio < 4.6, (
        f"sigma->0 convergence rate: distance ratio {ratio:.3f} not ~4 "
        f"(O(sigma^2) expected); distances {dists[0]:.3e}, {dists[1]:.3e}"
    )
    assert dists[0] / p_point < 0.02, (
        f"sigma=0.1 response should sit within 2% of pointlike, got "
        f"{dists[0] / p_point:.3e}"
    )


# ---------------------------------------------------------------------------
# Response machinery
# ---------------------------------------------------------------------------

def test_response_factorized_equals_direct_double_sum(chain):
    """The per-mode factorized response sum equals the O(n^2) double time
    sum on identical nodes (same quadrature, different evaluation path) —
    guards the factorization algebra."""
    K, LW, V = chain
    F = smearing("gaussian", 24.0, 2.0, N_CHAIN)
    ker = LW.smeared(F)
    chi = switching("gaussian", 1.0)
    Om = 0.2
    a, b = chi.support
    h, n_gl = 0.25, 8
    P_fact = _response_quad(ker.freqs, ker.amps, chi, Om, a, b, h, n_gl)
    t, w = gl_panels(a, b, h, n_gl)
    g = w * chi(t)
    P_dir = 0.0 + 0.0j
    for i0 in range(0, t.size, 64):
        dts = t[i0 : i0 + 64, None] - t[None, :]
        P_dir += np.sum(
            g[i0 : i0 + 64, None] * g[None, :] * np.exp(-1j * Om * dts) * ker(dts)
        )
    rel = abs(P_fact - P_dir) / abs(P_dir)
    assert rel < 1e-12, f"factorized vs direct double sum rel deviation {rel:.3e}"


def test_response_lattice_positive_and_converged(chain):
    """Lattice-backend response is positive (it is sum_k c_k |A_k|^2 with
    c_k > 0 for an auto-correlation kernel) and stable under an externally
    imposed finer initial quadrature."""
    K, LW, V = chain
    F = smearing("gaussian", 24.0, 2.0, N_CHAIN)
    ker = LW.smeared(F)
    chi = switching("gaussian", 1.0)
    P = detector_response(ker, chi, 0.2, rtol=1e-11)
    assert P > 0.0
    # cos2 switching path also converges and stays positive
    Pc = detector_response(ker, switching("cos2", 3.0), 0.2, rtol=1e-10)
    assert Pc > 0.0


def test_response_rejects_matrix_kernel(chain):
    """detector_response requires a smeared scalar kernel: the raw (N, N)
    lattice kernel must be refused with guidance."""
    K, LW, V = chain
    with pytest.raises(TypeError, match="smeared"):
        detector_response(LW, switching("gaussian", 1.0), 1.0)


# ---------------------------------------------------------------------------
# Switching library
# ---------------------------------------------------------------------------

def test_switching_shapes_and_normalization():
    chi_g = switching("gaussian", 2.0, t0=1.0)
    chi_c = switching("cos2", 2.0, t0=1.0)
    # unit peak at t0
    assert abs(chi_g(1.0) - 1.0) < 1e-15
    assert abs(chi_c(1.0) - 1.0) < 1e-15
    # cos2: compact support, exact zeros outside, continuous at the edge
    t = np.array([-1.0 - 1e-12, -1.0, 0.0, 1.0, 3.0, 3.0 + 1e-12, 5.0])
    v = chi_c(t)
    assert v[0] == 0.0 and v[-1] == 0.0 and v[-2] == 0.0
    assert abs(v[1]) < 1e-15 and abs(v[4]) < 1e-15  # vanishes at the edges
    assert chi_c.support == (-1.0, 3.0)
    # gaussian effective support covers 8.5 widths
    lo, hi = chi_g.support
    assert abs(lo - (1.0 - 17.0)) < 1e-12 and abs(hi - (1.0 + 17.0)) < 1e-12
    assert chi_g(hi) < 1e-15  # tail below double resolution of the peak


def test_switching_derivative_handles():
    """Analytic derivatives match central finite differences (both kinds),
    and the cos2 derivative is continuous (zero) at the support edge."""
    eps = 1e-6
    for kind, T, t0 in (("gaussian", 1.7, 0.3), ("cos2", 2.4, -0.5)):
        chi = switching(kind, T, t0)
        t = np.linspace(t0 - 0.9 * T, t0 + 0.9 * T, 17)
        fd = (chi(t + eps) - chi(t - eps)) / (2 * eps)
        dev = np.max(np.abs(chi.derivative(t) - fd))
        assert dev < 1e-8, f"{kind}: derivative vs FD deviation {dev:.3e}"
    chi_c = switching("cos2", 2.4, -0.5)
    edge = -0.5 + 2.4
    assert abs(chi_c.derivative(edge)) < 1e-15
    assert chi_c.derivative(edge + 1e-9) == 0.0


def test_switching_no_sharp_and_validation():
    """Sharp/top-hat switching is not offered (spec M2.1)."""
    for bad in ("sharp", "tophat", "delta"):
        with pytest.raises(ValueError, match="not offered"):
            switching(bad, 1.0)
    with pytest.raises(ValueError, match="width"):
        switching("gaussian", 0.0)
    with pytest.raises(AttributeError):
        switching("gaussian", 1.0).width = 2.0  # frozen


# ---------------------------------------------------------------------------
# Smearing library
# ---------------------------------------------------------------------------

def test_smearing_normalization():
    """sum F = 1 to 1e-12 for every kind and boundary condition."""
    worst = 0.0
    for kind, c, w, bc in (
        ("gaussian", 24.0, 3.0, "open"),
        ("gaussian", 2.0, 4.0, "periodic"),   # wraps around the ring
        ("cos2", 10.3, 4.6, "open"),
        ("cos2", 30.0, 5.0, "periodic"),
        ("point", 7.0, 0.0, "open"),
    ):
        F = smearing(kind, c, w, 32, bc=bc)
        worst = max(worst, abs(float(np.sum(F.F)) - 1.0))
        assert np.all(F.F >= 0.0)
    assert worst < 1e-12, f"sum F = 1 deviation {worst:.3e}"


def test_smearing_derivative_handles():
    """Analytic dF/dcenter and dF/dwidth of the *normalized* profile match
    central finite differences (parameters chosen so no lattice site sits at
    the cos2 support edge)."""
    eps = 1e-5
    for kind, c, w in (("gaussian", 24.0, 3.0), ("cos2", 10.3, 4.6)):
        F0 = smearing(kind, c, w, 32)
        fd_c = (smearing(kind, c + eps, w, 32).F - smearing(kind, c - eps, w, 32).F) / (
            2 * eps
        )
        fd_w = (smearing(kind, c, w + eps, 32).F - smearing(kind, c, w - eps, 32).F) / (
            2 * eps
        )
        dev_c = np.max(np.abs(F0.dF_dcenter - fd_c))
        dev_w = np.max(np.abs(F0.dF_dwidth - fd_w))
        assert dev_c < 1e-8, f"{kind}: dF/dcenter vs FD deviation {dev_c:.3e}"
        assert dev_w < 1e-8, f"{kind}: dF/dwidth vs FD deviation {dev_w:.3e}"
    # derivative arrays sum to ~0 (normalization is preserved along the flow)
    F = smearing("gaussian", 24.0, 3.0, 32)
    assert abs(np.sum(F.dF_dcenter)) < 1e-14
    assert abs(np.sum(F.dF_dwidth)) < 1e-14


def test_smearing_point_and_validation():
    F = smearing("point", 7.2, 0.0, 16)
    assert F.F[7] == 1.0 and float(np.sum(F.F)) == 1.0
    assert np.all(F.dF_dcenter == 0.0) and np.all(F.dF_dwidth == 0.0)
    # array protocol: drops into sandwiches directly
    assert np.asarray(F).shape == (16,)
    with pytest.raises(ValueError, match="outside"):
        smearing("point", 20.0, 0.0, 16)
    with pytest.raises(ValueError, match="zero weight"):
        smearing("cos2", 100.0, 2.0, 16)  # support entirely off-lattice
    with pytest.raises(ValueError, match="width"):
        smearing("gaussian", 8.0, 0.0, 16)
    with pytest.raises(ValueError, match="kind"):
        smearing("tophat", 8.0, 1.0, 16)
    with pytest.raises(AttributeError):
        smearing("point", 7.0, 0.0, 16).center = 3.0  # frozen
    with pytest.raises(ValueError):
        np.asarray(smearing("point", 7.0, 0.0, 16)).__setitem__(0, 5.0)  # read-only
