"""Derivative coupling ('xp'): anchors for items 2-5 of the M2.5 close-out.

What is pinned here (docs/PLAN_LAYERS_2_3.md M2.2 "v2 flag", M2.5
"derivative-coupling claim"; PLAN.md Layer 2 discovery target):

- the general-H_mat energy path: ``vacuum.core.mean_energy(V, H_mat)``
  equals the diag(K, I) special case to 1e-12, and the ledger / protocol
  runner book an 'xp' run whose switching does NOT vanish at the exit
  under that path, closing to 1e-9 (spec: all protocols);
- ``attach_detectors(..., coupling='xp')`` IS the quadratic form
  H = (1/2)(p.p + x^T K_tot x) + sum_d lam_d x_d F_d^T p_field (checked
  against the definition on random phase-space vectors), and its analytic
  time derivative matches a finite difference;
- the pulled-back pi-pi kernel d_t d_t' W (``derivative_kernel``): equal-
  time <p_i p_j> = (K^{1/2}/2)_ij to 1e-12, the smeared form equals
  F^T momentum(dt) F', and the single-detector response of the derivative-
  coupled detector agrees with an EXACT mode sum (analytic switching
  transforms, no quadrature) to 1e-12 on both backends;
- the M2.5 reproduction: for CAUSALLY CONNECTED detectors on the 1+1
  lattice (compact cos2 supports meeting on the light cone), the derivative
  coupling draws its pair term overwhelmingly from the field's
  correlations, |M_vac|/|M| ~ 1 and |M_comm|/|M| << 1, while the amplitude
  coupling's communication share is an order of magnitude larger — the
  qualitative content of Teixido-Bonfill & Martin-Martinez, PRD 110, 105016
  (2024) [TB-MM24], restated in arXiv:2505.01516.  Measured fractions ride
  in every assertion message;
- a 4-point smoke test of the noise-survival map's code path
  (papers/derivative-coupling-noise-survival/notebook.py).

Conventions per docs/API.md: hbar = 1, V_vac = I/2, block ordering, nats.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from vacuum.audits import EnergyLedger
from vacuum.core import Omega as Omega_form
from vacuum.core import ground_state_cov, harmonic_chain_K, mean_energy
from vacuum.detectors import (
    DerivativeKernel,
    RefinableDerivativeKernel,
    UDWDetector,
    attach_detectors,
    communication_split,
    derivative_kernel,
    detector_block,
    detector_coupling_derivative,
    detector_response,
    gl_panels,
    harvested_log_negativity,
    imperfection_channel,
    initial_state,
    pair_state_with_split_derivative,
    protocol_evolve,
    protocol_evolve_fixed,
    response_mode_sum,
    run_harvesting,
    spatial_derivative_profile,
    switching,
    switching_fourier,
    udw_pair_state,
    wightman_continuum_3p1,
    wightman_lattice,
)
from vacuum.protocol import (
    GeneratorStep,
    hamiltonian_matrix,
    quadratic_mean_energy,
    run_protocol,
)

LEDGER_TOL = 1e-9   # spec test matrix: ledger closure, all protocols
N_F = 8
SITES = (1, 5)
GAPS = (1.5, 2.0)
REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "derivative-coupling-noise-survival" / "notebook.py"


@pytest.fixture(scope="module")
def K_field():
    return harmonic_chain_K(N_F, m=1.0)


def _random_physical_cov(n, rng):
    """S (I/2) S^T for a random symplectic S = exp(Omega H): a pure Gaussian state."""
    H = rng.normal(size=(2 * n, 2 * n))
    H = 0.5 * (H + H.T)
    S = expm(Omega_form(n) @ H)
    return 0.5 * S @ S.T


# --------------------------------------------------------------------------
# 1. General-H_mat mean energy == diag(K, I) special case (1e-12)
# --------------------------------------------------------------------------


def test_mean_energy_general_path_matches_diag_special_case():
    rng = np.random.default_rng(20260901)
    worst = 0.0
    for n in (1, 3, 7):
        for _ in range(5):
            V = _random_physical_cov(n, rng)
            A = rng.normal(size=(n, n))
            K = A @ A.T + n * np.eye(n)
            E_diag = mean_energy(V, K)
            E_gen = mean_energy(V, hamiltonian_matrix(K))
            E_proto = quadratic_mean_energy(V, hamiltonian_matrix(K))
            worst = max(
                worst,
                abs(E_gen - E_diag) / abs(E_diag),
                abs(E_proto - E_diag) / abs(E_diag),
            )
    assert worst < 1e-12, (
        f"mean_energy(V, H_mat) vs mean_energy(V, K) on diag(K, I): worst relative "
        f"difference {worst:.3e} >= 1e-12"
    )
    # the general path really is (1/2) Tr(H V) for a form WITH x-p terms
    V = _random_physical_cov(2, rng)
    H = rng.normal(size=(4, 4))
    H = 0.5 * (H + H.T)
    assert mean_energy(V, H) == pytest.approx(0.5 * np.trace(H @ V), rel=1e-14)
    with pytest.raises(ValueError, match="quadratic form"):
        mean_energy(V, np.eye(3))


# --------------------------------------------------------------------------
# 2. attach_detectors('xp') is the stated quadratic form; its derivative
# --------------------------------------------------------------------------


def test_xp_form_is_x_d_times_field_momentum(K_field):
    """(1/2) R^T H_mat R == (1/2)(p.p + x^T K_tot x) + sum_d lam_d x_d F_d^T p."""
    rng = np.random.default_rng(7)
    lam = (0.3, -0.45)
    K_tot0 = attach_detectors(K_field, SITES, GAPS, 0.0)
    H = attach_detectors(K_field, SITES, GAPS, lam, coupling="xp")
    n_tot = N_F + 2
    worst = 0.0
    for _ in range(20):
        R = rng.normal(size=2 * n_tot)
        x, p = R[:n_tot], R[n_tot:]
        want = 0.5 * (p @ p + x @ K_tot0 @ x)
        for d, (site, l) in enumerate(zip(SITES, lam)):
            want += l * x[N_F + d] * p[site]          # F_d = e_site (pointlike)
        got = 0.5 * R @ H @ R
        worst = max(worst, abs(got - want) / abs(want))
    assert worst < 1e-14, f"'xp' quadratic form differs from its definition by {worst:.3e}"
    with pytest.raises(ValueError, match="coupling"):
        attach_detectors(K_field, SITES, GAPS, lam, coupling="pp")


def test_xp_derivative_matches_finite_difference(K_field):
    chi = switching("cos2", 2.0, t0=1.0)
    lam_max = np.array([0.4, 0.25])

    def H_of_t(t):
        return attach_detectors(K_field, SITES, GAPS, lam_max * chi(t), coupling="xp")

    t, h = 0.3, 1e-5
    fd = (H_of_t(t + h) - H_of_t(t - h)) / (2.0 * h)
    an = detector_coupling_derivative(
        N_F, SITES, lam_max * chi.derivative(t), coupling="xp"
    )
    assert an.shape == fd.shape == (2 * (N_F + 2), 2 * (N_F + 2))
    assert np.max(np.abs(fd - an)) < 1e-8, (
        f"dH_mat/dt (xp) vs finite difference: {np.max(np.abs(fd - an)):.3e}"
    )
    # 'xx' derivative is unchanged in shape and content
    dK = detector_coupling_derivative(N_F, SITES, lam_max * chi.derivative(t))
    assert dK.shape == (N_F + 2, N_F + 2)
    # ... and the 'xp' p_i-x_d block carries exactly the 'xx' field-detector block
    n_tot = N_F + 2
    assert np.array_equal(an[n_tot : n_tot + N_F, N_F:n_tot], dK[:N_F, N_F:])


# --------------------------------------------------------------------------
# 3. The pi-pi kernel: moments, smeared form, exact-mode-sum response (1e-12)
# --------------------------------------------------------------------------


def test_momentum_wightman_equal_time_and_smeared_form():
    K = harmonic_chain_K(12, 0.7, bc="dirichlet")
    ker = wightman_lattice(K)
    V = ground_state_cov(K)
    Vpp = V[12:, 12:]                                 # = K^{1/2}/2
    dev = np.max(np.abs(np.real(ker.momentum(0.0)) - Vpp))
    assert dev < 1e-12, f"Re <p_i p_j>(0) vs K^(1/2)/2 differ by {dev:.3e}"
    # d_t d_t' W = -W'' pointwise, and the smeared derivative kernel is F^T (.) F'
    F = np.zeros(12); F[3] = 1.0
    G = np.zeros(12); G[8] = 1.0
    base = ker.smeared(F, G)
    dk = derivative_kernel(base)
    assert isinstance(dk, DerivativeKernel) and not hasattr(dk, "with_refined")
    assert np.allclose(dk.amps, np.asarray(base.amps) * np.asarray(base.freqs) ** 2)
    worst = 0.0
    for dt in (-3.1, -0.4, 0.0, 0.7, 2.5):
        want = complex(F @ ker.momentum(dt) @ G)
        got = complex(dk(dt))
        worst = max(worst, abs(got - want), abs(complex(base.d2(dt)) + want))
    assert worst < 1e-13, f"derivative kernel vs F^T momentum(dt) F' differ by {worst:.3e}"
    # the continuum backend gets the refinable flavour, forwarding its gates
    ck = derivative_kernel(wightman_continuum_3p1(0.4, distance=1.0))
    assert isinstance(ck, RefinableDerivativeKernel)
    assert isinstance(ck.with_refined(2), RefinableDerivativeKernel)
    assert ck.with_refined(2).freqs.size > ck.freqs.size
    assert np.allclose(ck.with_kmax(5.0).base.kmax, 5.0)
    with pytest.raises(TypeError):
        derivative_kernel(K)


@pytest.mark.parametrize("kind", ["cos2", "gaussian"])
def test_switching_fourier_matches_quadrature(kind):
    chi = switching(kind, 1.7, t0=0.35)
    a, b = chi.support
    t, w = gl_panels(a, b, 0.02, n_gl=16)
    worst = 0.0
    for u in (-3.3, -1.0, 0.0, math.pi / 1.7, 0.9, 2.2, 5.0):
        num = complex(np.sum(w * chi(t) * np.exp(-1j * u * t)))
        an = switching_fourier(chi, u)
        worst = max(worst, abs(num - an))
    assert worst < 1e-12, f"[{kind}] analytic switching transform vs quadrature: {worst:.3e}"


@pytest.mark.parametrize("kind", ["cos2", "gaussian"])
def test_derivative_response_matches_exact_mode_sum_lattice(kind):
    """P/lam^2 of the derivative-coupled detector: gated quadrature vs exact sum (1e-12)."""
    K = harmonic_chain_K(14, 0.5, bc="periodic")
    ker = wightman_lattice(K)
    F = np.zeros(14); F[5] = 1.0
    chi = switching(kind, 1.5, t0=0.0)
    for Omega in (0.7, 2.0):
        for kernel in (derivative_kernel(ker.smeared(F)), ker.smeared(F)):
            exact = response_mode_sum(kernel, chi, Omega)
            quad = detector_response(kernel, chi, Omega, rtol=1e-13, atol=1e-30,
                                     max_doublings=16)
            rel = abs(quad - exact) / exact
            assert rel < 1e-12, (
                f"[{kind}, Omega={Omega}, {type(kernel).__name__}] gated response "
                f"{quad:.15e} vs exact mode sum {exact:.15e}: relative {rel:.3e}"
            )


def test_derivative_response_matches_exact_mode_sum_continuum():
    """Same identity on the 3+1 backend at its own k-discretization (k^2 factor)."""
    kernel = derivative_kernel(wightman_continuum_3p1(0.3))
    chi = switching("gaussian", 0.8, t0=0.0)
    exact = response_mode_sum(kernel, chi, 1.5)
    quad = detector_response(
        kernel, chi, 1.5, rtol=1e-13, atol=1e-30, max_doublings=16, refine_kernel=False
    )
    rel = abs(quad - exact) / exact
    assert rel < 1e-12, f"continuum derivative response {quad:.15e} vs {exact:.15e}: {rel:.3e}"
    # ... and the k^2 factor is what distinguishes it from the amplitude response
    base = response_mode_sum(wightman_continuum_3p1(0.3), chi, 1.5)
    assert exact != pytest.approx(base, rel=1e-3)


# --------------------------------------------------------------------------
# 4. The nonperturbative 'xp' stack: inertness, ledger, drive work, channels
# --------------------------------------------------------------------------


def test_xp_lambda_zero_inert(K_field):
    res = run_harvesting(
        K_field, SITES, GAPS, lambda t: 0.0, ts=[0.0, 1.0, 2.5], coupling="xp"
    )
    V0 = ground_state_cov(attach_detectors(K_field, SITES, GAPS, 0.0))
    drift = float(np.max(np.abs(res.V - V0)))
    assert drift < 1e-14, f"xp lambda = 0 drift {drift:.3e}"
    assert res.work == 0.0 and res.dissipated == 0.0
    assert abs(res.ledger_result.details["defect"]) <= LEDGER_TOL


XP_LAM = 0.5
XP_CHI = switching("gaussian", 1.2, t0=3.0)     # NOT compact: lam(t_end) != 0 -> general H_mat at exit
XP_TS = np.linspace(0.0, 4.5, 19)


def _xp_H(K):
    def H_of_t(t):
        return attach_detectors(K, SITES, GAPS, XP_LAM * XP_CHI(t), coupling="xp")
    return H_of_t


def _xp_dH():
    def dH_of_t(t):
        return detector_coupling_derivative(
            N_F, SITES, XP_LAM * XP_CHI.derivative(t), coupling="xp"
        )
    return dH_of_t


@pytest.fixture(scope="module")
def xp_fixed_runs(K_field):
    V0 = ground_state_cov(attach_detectors(K_field, SITES, GAPS, 0.0))
    runs = {
        s: protocol_evolve_fixed(V0, _xp_H(K_field), XP_TS, splits=s, dK_of_t=_xp_dH())
        for s in (16, 32)
    }
    return V0, runs


def test_xp_telescoped_work_closes_under_general_hamiltonian(xp_fixed_runs, K_field):
    """W (telescoped) == Delta E with the exit energy on the general-H_mat path."""
    V0, runs = xp_fixed_runs
    H0 = _xp_H(K_field)(XP_TS[0])
    H1 = _xp_H(K_field)(XP_TS[-1])
    assert np.count_nonzero(H1[N_F + 2 :, : N_F + 2]) > 0, "exit form lost its x-p terms"
    for s, run in runs.items():
        dE = mean_energy(run.V, H1) - mean_energy(V0, H0)
        defect = run.work - dE
        assert abs(defect) <= LEDGER_TOL, (
            f"splits={s}: xp telescoped work {run.work:.12e} vs Delta E {dE:.12e}, "
            f"defect {defect:.3e}"
        )
        # The sign is physical: switching an x-p coupling ON and leaving it on
        # lowers <H> (the coupled ground state lies below the product ground
        # state at second order), so the drive *extracts* energy — W < 0 —
        # and the books still close.  Only the magnitude is a sanity check.
        assert abs(run.work) > 1e-4, f"xp drive work {run.work:.3e} suspiciously small"


def test_xp_derivative_drive_work_is_second_order(xp_fixed_runs, K_field):
    V0, runs = xp_fixed_runs
    H0 = _xp_H(K_field)(XP_TS[0])
    H1 = _xp_H(K_field)(XP_TS[-1])
    errs = {
        s: abs(run.work_deriv - (mean_energy(run.V, H1) - mean_energy(V0, H0)))
        for s, run in runs.items()
    }
    ratio = errs[16] / errs[32]
    assert 3.3 < ratio < 4.7, (
        f"xp <dH/dt> quadrature not O(dt^2): {errs[16]:.3e} -> {errs[32]:.3e} (ratio {ratio:.2f})"
    )


def test_xp_protocol_evolve_books_general_exit_snapshot(K_field):
    """protocol_evolve's ledger path accepts a general H_mat at the boundaries."""
    V0 = ground_state_cov(attach_detectors(K_field, SITES, GAPS, 0.0))
    ledger = EnergyLedger(K=attach_detectors(K_field, SITES, GAPS, 0.0), tol=LEDGER_TOL)
    ev = protocol_evolve(
        V0, _xp_H(K_field), XP_TS, ledger=ledger, conv_tol=1e-9, max_halvings=12
    )
    res = ledger.close()
    assert res.passed, f"xp ledger defect {res.details['defect']:.3e}"
    snaps = [r["value"] for r in ledger.table if r["kind"] == "snapshot"]
    H1 = _xp_H(K_field)(XP_TS[-1])
    assert snaps[-1] == pytest.approx(mean_energy(ev.V, H1), rel=1e-13)
    assert ev.converged and ev.work != 0.0


def test_run_protocol_accepts_xp_generator_with_general_exit(K_field):
    """The M-I.4 runner: a callable 'xp' generator ending off diag(K, I) form."""
    K_tot0 = attach_detectors(K_field, SITES, GAPS, 0.0)
    V0 = ground_state_cov(K_tot0)
    res = run_protocol(
        V0,
        K_tot0,
        [GeneratorStep(_xp_H(K_field), float(XP_TS[-1]), n_substeps=256, label="xp-drive")],
        claim_ground_entry=True,
        report_modes=[[N_F, N_F + 1]],
        run_causality=False,
        strict=True,
    )
    assert res.K is None, "exit Hamiltonian with x-p terms must not masquerade as diag(K, I)"
    assert res.H_mat is not None and np.count_nonzero(res.H_mat[N_F + 2 :, : N_F + 2]) > 0
    assert res.audits["ledger"].passed and res.all_passed()
    snaps = [r["value"] for r in res.ledger.table if r["kind"] == "snapshot"]
    assert snaps[-1] == pytest.approx(mean_energy(res.V, res.H_mat), rel=1e-13)
    assert res.work_total != 0.0
    # a ground-state claim at such an exit is refused, not silently audited
    with pytest.raises(ValueError, match="claim_ground_exit"):
        run_protocol(
            V0,
            K_tot0,
            [GeneratorStep(_xp_H(K_field), float(XP_TS[-1]), n_substeps=8)],
            claim_ground_exit=True,
            report_modes=[],
            run_causality=False,
        )


def test_xp_channels_interleave_and_close(K_field):
    chi = switching("cos2", 2.0, t0=2.0)
    ch = imperfection_channel([N_F, N_F + 1], kappa=0.2, nbar=0.05, gamma_phi=0.05, gaps=GAPS)
    res = run_harvesting(
        K_field, SITES, GAPS, lambda t: 0.6 * chi(t), ts=np.linspace(0.0, 4.0, 17),
        channels=(ch,), coupling="xp",
    )
    assert res.converged
    assert abs(res.ledger_result.details["defect"]) <= LEDGER_TOL, (
        f"open xp run ledger defect {res.ledger_result.details['defect']:.3e}"
    )
    assert res.dissipated != 0.0
    kinds = [r["kind"] for r in res.ledger.table]
    assert kinds.count("dissipation") == 1 and kinds.count("work") == 1


# --------------------------------------------------------------------------
# 5. Perturbative plumbing: coupling='xp' == the derivative kernel triple
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def light_contact():
    """1+1 massless (Dirichlet) lattice, two pointlike detectors at light contact.

    Sites 24 and 36 (d = 12) on N = 60: both are 24 sites from the nearest
    wall, so no reflection returns inside the protocol (round trip 48 >
    2T + d = 20).  cos2 switchings of half-width T = 4 centred at t_A = 0
    and t_B = d: every pair (t, t') with t - t' = d — the light cone — lies
    inside the double integral (causal contact), the configuration TB-MM24
    Fig. 2(a) marks as full light contact.  Gap Omega = 0.5 keeps the
    resonant field modes (omega ~ Omega) on the linear part of the lattice
    dispersion (group velocity 0.97), so the lattice light cone is the
    continuum one to 3%.
    """
    N, jA, d, T, Om = 60, 24, 12, 4.0, 0.5
    K = harmonic_chain_K(N, 0.0, bc="dirichlet")
    ker = wightman_lattice(K)
    chi_A = switching("cos2", T, 0.0)
    chi_B = switching("cos2", T, float(d))
    F_A = np.zeros(N); F_A[jA] = 1.0
    F_B = np.zeros(N); F_B[jA + d] = 1.0
    return dict(N=N, jA=jA, d=d, T=T, Om=Om, K=K, ker=ker, chi_A=chi_A, chi_B=chi_B,
                F_A=F_A, F_B=F_B)


def test_coupling_xp_equals_derivative_triple(light_contact):
    lc = light_contact
    det_A = UDWDetector(lc["chi_A"], lc["F_A"])
    det_B = UDWDetector(lc["chi_B"], lc["F_B"])
    st_xp = udw_pair_state(lc["ker"], det_A, det_B, 0.05, lc["Om"], coupling="xp")
    triple = tuple(
        derivative_kernel(k)
        for k in (lc["ker"].smeared(lc["F_A"]), lc["ker"].smeared(lc["F_B"]),
                  lc["ker"].smeared(lc["F_A"], lc["F_B"]))
    )
    st_tr = udw_pair_state(triple, det_A, det_B, 0.05, lc["Om"])
    assert st_xp.meta["coupling"] == "xp" and st_tr.meta["coupling"] == "xx"
    for name in ("P_A", "P_B", "C", "M"):
        assert getattr(st_xp, name) == getattr(st_tr, name), f"{name} differs"
    sp = communication_split(lc["ker"], det_A, det_B, 0.05, lc["Om"], M=st_xp.M, coupling="xp")
    assert sp.meta["coupling"] == "xp"
    assert abs(sp.M_vac + sp.M_comm - st_xp.M) <= 1e-15 * abs(st_xp.M)
    full = pair_state_with_split_derivative(lc["ker"], det_A, det_B, 0.05, lc["Om"])
    assert full.M == st_xp.M and full.row["comm_fraction"] == pytest.approx(sp.comm_fraction)
    with pytest.raises(ValueError, match="coupling"):
        udw_pair_state(lc["ker"], det_A, det_B, 0.05, lc["Om"], coupling="px")


def test_xp_commutator_part_is_causal_on_the_lattice(light_contact):
    """Spacelike xp pair: |M_comm| < 1e-12 (the spec's spacelike tolerance).

    Same massless lattice, detectors 20 sites apart, both switched on
    together with cos2 half-width 2, so |t - t'| <= 4 << 20: the whole
    double integral is outside the lattice light cone up to the Lieb-
    Robinson tail (measured |M_comm| = 7.5e-17 against |M| = 1.2e-3 at
    lambda = 1).  At (d = 12, T = 2) the tail is already 3e-14 — the cone
    on a lattice is fuzzy at the Bessel-tail level, which is why the
    separation here is generous.
    """
    lc = light_contact
    N = lc["N"]
    chi = switching("cos2", 2.0, 0.0)
    F_A = np.zeros(N); F_A[20] = 1.0
    F_B = np.zeros(N); F_B[40] = 1.0
    det_A = UDWDetector(chi, F_A)
    det_B = UDWDetector(chi, F_B)
    sp = communication_split(lc["ker"], det_A, det_B, 1.0, lc["Om"], coupling="xp",
                             comm_atol=1e-15)
    assert abs(sp.M) > 1e-4, f"|M| = {abs(sp.M):.3e}: spacelike xp test has no content"
    assert abs(sp.M_comm) < 1e-12, (
        f"spacelike xp |M_comm| = {abs(sp.M_comm):.3e} >= 1e-12 (|M| = {abs(sp.M):.3e}, "
        f"fraction {sp.comm_fraction:.3e}); the pi-pi commutator leaked outside the "
        "lattice light cone"
    )
    assert sp.comm_fraction < 1e-10, f"|M_comm|/|M| = {sp.comm_fraction:.3e}"


# --------------------------------------------------------------------------
# 6. M2.5: the derivative-coupling claim, qualitatively, at light contact
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def light_contact_splits(light_contact):
    lc = light_contact
    out = {}
    for name, coupling, F_A, F_B in (
        ("xx", "xx", lc["F_A"], lc["F_B"]),
        ("xp", "xp", lc["F_A"], lc["F_B"]),
        ("dx", "xx",
         spatial_derivative_profile(lc["N"], lc["jA"]),
         spatial_derivative_profile(lc["N"], lc["jA"] + lc["d"])),
    ):
        det_A = UDWDetector(lc["chi_A"], F_A)
        det_B = UDWDetector(lc["chi_B"], F_B)
        st = pair_state_with_split_derivative(lc["ker"], det_A, det_B, 0.05, lc["Om"]) \
            if coupling == "xp" else None
        if st is None:
            from vacuum.detectors import pair_state_with_split
            st = pair_state_with_split(lc["ker"], det_A, det_B, 0.05, lc["Om"])
        out[name] = st
    return out


def test_derivative_coupling_entanglement_is_field_correlations_not_communication(
    light_contact_splits,
):
    """TB-MM24 / arXiv:2505.01516, qualitatively, on the 1+1 lattice.

    For causally connected detectors the derivative-coupled pair term is
    dominated by the anti-commutator (field-correlation) part and it does
    harvest (|M| > P), whereas the amplitude coupling's communication share
    is an order of magnitude larger.  Measured on this rig (N = 60 massless
    Dirichlet, d = 12, cos2 T = 4 at light contact, Omega = 0.5):
    xp |M_vac|/|M| = 0.999, |M_comm|/|M| = 0.034, |M| - P > 0;
    xx |M_vac|/|M| = 0.931, |M_comm|/|M| = 0.365, |M| - P < 0.
    """
    xp = light_contact_splits["xp"]
    xx = light_contact_splits["xx"]
    dx = light_contact_splits["dx"]
    msg = (
        f"xp: |M_vac|/|M| = {xp.comm.vac_fraction:.4f}, |M_comm|/|M| = "
        f"{xp.comm.comm_fraction:.4f}, |M| = {abs(xp.M):.3e}, P = {xp.P_A:.3e}, "
        f"N = {xp.negativity:.3e}; xx: |M_vac|/|M| = {xx.comm.vac_fraction:.4f}, "
        f"|M_comm|/|M| = {xx.comm.comm_fraction:.4f}, |M| = {abs(xx.M):.3e}, "
        f"P = {xx.P_A:.3e}, N^(2) = {xx.negativity_estimator:.3e}; "
        f"d_x-stencil (2505.01516 Eq. (31) model): |M_vac|/|M| = "
        f"{dx.comm.vac_fraction:.4f}, |M_comm|/|M| = {dx.comm.comm_fraction:.4f}, "
        f"N^(2) = {dx.negativity_estimator:.3e}"
    )
    # the claim: correlation-part dominance under derivative coupling ...
    assert xp.comm.vac_fraction > 0.95 and xp.comm.comm_fraction < 0.1, msg
    assert xp.comm.vac_fraction > xp.comm.comm_fraction, msg
    # ... for detectors that DO end up entangled while in causal contact
    assert xp.negativity > 0.0 and xp.log_negativity > 0.0, msg
    # ... and it is a property of the coupling: the amplitude coupling at the
    # same light contact signals an order of magnitude more
    assert xx.comm.comm_fraction > 5.0 * xp.comm.comm_fraction, msg
    assert xx.comm.vac_fraction < xp.comm.vac_fraction, msg
    # the spatial-derivative (circuit-QED, arXiv:2505.01516 Eq. (31)) model
    # sits with the time-derivative one, not with the amplitude coupling
    assert dx.comm.vac_fraction > 0.95 and dx.comm.comm_fraction < 0.1, msg
    assert dx.negativity > 0.0, msg


def test_light_contact_rows_carry_the_m25_columns(light_contact_splits):
    for name, st in light_contact_splits.items():
        row = st.row
        for col in ("E_N", "M_vac", "M_comm", "comm_fraction"):
            assert row[col] is not None, f"[{name}] column {col} missing"
        assert st.meta["perturbative_ok"], f"[{name}] outside the second-order regime"
        assert abs(row["M_vac"] + row["M_comm"] - row["M"]) <= 1e-15 * abs(row["M"])


# --------------------------------------------------------------------------
# 7. The map's code path: a 4-point smoke test of the papers candidate
# --------------------------------------------------------------------------


def _load_notebook():
    if not NOTEBOOK_PATH.exists():  # pragma: no cover
        pytest.skip(f"notebook not found at {NOTEBOOK_PATH}")
    spec = importlib.util.spec_from_file_location("xp_noise_survival_notebook", NOTEBOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def nb():
    return _load_notebook()


@pytest.fixture(scope="module")
def smoke_rows(nb):
    specs = nb.smoke_specs()
    assert len(specs) == 4, f"smoke sweep must be 4 points, got {len(specs)}"
    return nb.run_specs(specs, verbose=False)


def test_map_smoke_rows_carry_the_schema(nb, smoke_rows):
    assert len(smoke_rows) == 4
    for row in smoke_rows:
        for col in nb.ROW_COLUMNS:
            assert col in row, f"row {row.get('label')!r} lacks column {col!r}"
        assert row["ledger_passed"] and abs(row["ledger_defect"]) <= LEDGER_TOL, row["label"]
        assert row["converged"] and row["dt_converged"] > 0.0, row["label"]
        assert row["coupling"] == "xp"
        assert isinstance(row["flagged"], (bool, np.bool_))
        assert 0.0 <= row["vac_fraction_T"] and row["E_N"] >= 0.0


def test_map_smoke_covers_the_branches(nb, smoke_rows):
    """vacuum vs thermal field, noiseless vs loss+dephasing, and the split at T."""
    temps = {row["temperature"] for row in smoke_rows}
    assert 0.0 in temps and any(t > 0.0 for t in temps)
    assert any(row["kappa"] > 0.0 or row["gamma_phi"] > 0.0 for row in smoke_rows)
    assert any(row["kappa"] == 0.0 and row["gamma_phi"] == 0.0 for row in smoke_rows)
    base = [r for r in smoke_rows if r["temperature"] == 0.0 and r["kappa"] == 0.0
            and r["gamma_phi"] == 0.0]
    assert base and base[0]["E_N"] > 0.0, "noiseless xp point must harvest"
    # the split fraction at T = 0 is the perturbative light-contact value
    assert base[0]["vac_fraction_T"] > 0.95, base[0]
