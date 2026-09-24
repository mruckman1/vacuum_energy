"""Layer 5 runs: matched loss bath, loss threshold, pulse trains, bounds.

Pinned:
- the K0-matched multimode bath is completely positive (Holevo-Werner) and
  leaves the undriven vacuum invariant (zero work, zero dissipation, ledger
  closed) — the site-basis bath does not, which is why it is not the default;
- the net-gain threshold eta lambda^2 = 1: occupations grow above it and
  decay below it, and the Q mapping round-trips;
- ramped pulse trains through ``run_protocol`` converge under the gate,
  close the ledger, do no work during the free gaps, and produce fewer
  photons than the same number of unramped periods;
- the bang-bang optimum artanh(depth) is *attained* by the two-piece drive
  with quarter-period dwell times and is never exceeded by cosine or 50 %
  square drives on a frequency scan;
- the amplification scan is self-consistent (threshold = 1/lambda^2, cosine
  ratio to (pi/4) depth within [1, 1.02] for depth <= 0.9).
"""

import numpy as np
import pytest

from vacuum.core import Omega, ground_state_cov, harmonic_chain_K, mean_energy
from vacuum.floquet import (
    amplification_bound_scan,
    bang_bang_bound,
    field_loss,
    floquet_map,
    floquet_run,
    loss_per_period,
    loss_threshold,
    lossy_floquet,
    matched_bath_cov,
    max_gain_over_frequency,
    modulated_K,
    normal_modes,
    pulse_train,
    quality_factor_from_eta,
    run_floquet_protocol,
    stability_chart,
)

N, M = 8, 0.5
OMEGA_RES = 1.5


@pytest.fixture(scope="module")
def chain():
    K0 = harmonic_chain_K(N, M)
    return K0, ground_state_cov(K0)


def _holevo_werner_margin(X, Y):
    n = X.shape[0] // 2
    Om = Omega(n)
    Mhw = Y.astype(complex) + 0.5j * (Om - X @ Om @ X.T)
    return float(np.min(np.linalg.eigvalsh(Mhw)))


def test_matched_bath_is_cp_and_fixes_the_vacuum(chain):
    K0, V0 = chain
    for nbar in (0.0, 0.3):
        V_env = matched_bath_cov(K0, nbar)
        eta = 0.8
        X = np.sqrt(eta) * np.eye(2 * N)
        Y = (1 - eta) * V_env
        margin = _holevo_werner_margin(X, Y)
        assert margin > -1e-12, f"matched bath nbar={nbar}: Holevo-Werner margin {margin:.2e}"
    assert np.max(np.abs(matched_bath_cov(K0, 0.0) - V0)) < 1e-13
    assert np.max(np.abs(field_loss(V0, 0.3, matched_bath_cov(K0, 0.0)) - V0)) < 1e-14
    # undriven, lossy vacuum: nothing happens, books close on zeros
    drive = modulated_K(K0, 0.0, 2 * OMEGA_RES, "sin")
    run = floquet_run(V0, drive, 5, eta=0.7, n_steps=8, conv_tol=1e-10)
    assert run.ledger_result.passed
    assert abs(run.work_total) < 1e-12 and abs(run.dissipated_total) < 1e-12
    assert np.max(np.abs(run.V - V0)) < 1e-12
    # the site bath heats the vacuum (negative 'dissipation'): documented, not default
    run_site = floquet_run(V0, drive, 5, eta=0.7, bath="site", n_steps=8, conv_tol=1e-10)
    assert run_site.ledger_result.passed and run_site.dissipated_total < -1e-3


def test_loss_threshold_separates_growth_from_decay(chain):
    K0, V0 = chain
    drive = modulated_K(K0, 0.1, 2 * OMEGA_RES, "cos")
    fm = floquet_map(drive, n_steps=16, conv_tol=1e-10)
    lam = fm.multiplier
    eta_th = float(loss_threshold(lam))
    assert abs(eta_th - 1.0 / lam**2) < 1e-15
    omega, _ = normal_modes(K0)
    j = int(np.argmin(np.abs(omega - OMEGA_RES)))
    # Below threshold the occupation SATURATES (steady pair production
    # balanced by loss, the below-threshold parametric oscillator), above it
    # grows without bound; in both cases the late-time increments scale by
    # eta lambda^2 per period (the approach to the steady state is governed
    # by the same Floquet multiplier).
    for eta, grows in ((eta_th * 1.05, True), (eta_th * 0.95, False)):
        run = floquet_run(V0, drive, 80, fmap=fm, eta=eta, claim_ground_entry=False)
        assert run.ledger_result.passed
        inc = np.diff(run.occupations[:, j])
        r_inc = inc[-1] / inc[-2]
        assert abs(r_inc - eta * lam**2) < 5e-3, (
            f"eta={eta:.5f} (threshold {eta_th:.5f}): late increment ratio {r_inc:.5f} "
            f"vs eta lambda^2 = {eta * lam**2:.5f}"
        )
        assert (r_inc > 1.0) == grows
        if not grows:  # bounded: the increments shrink monotonically at late times
            assert inc[-1] < inc[-20] and run.occupations[-1, j] < 2.0 * run.occupations[-20, j]
        V_bare = lossy_floquet(V0, fm.S, eta, 0.0, 80, K0=K0)
        assert np.max(np.abs(V_bare - run.V)) < 1e-10
    # Q mapping round trip, principal resonance eta = exp(-pi/Q)
    eta = loss_per_period(50.0, OMEGA_RES, 2 * OMEGA_RES)
    assert abs(eta - np.exp(-np.pi / 50.0)) < 1e-15
    assert abs(quality_factor_from_eta(eta, OMEGA_RES, 2 * OMEGA_RES) - 50.0) < 1e-10


def test_lossy_protocol_path_matches_fast_path(chain):
    K0, V0 = chain
    drive = modulated_K(K0, 0.1, 2 * OMEGA_RES, "sin")
    run = floquet_run(V0, drive, 6, eta=0.9, nbar=0.2, n_steps=16, conv_tol=1e-10)
    res = run_floquet_protocol(V0, drive, 6, n_substeps_per_period=run.fmap.n_steps,
                               eta=0.9, nbar=0.2)
    assert np.max(np.abs(res.V - run.V)) < 1e-10
    assert abs(res.dissipated_total - run.dissipated_total) < 1e-9
    assert res.all_passed() and run.ledger_result.passed
    # nbar = 0.2 is hotter than the weakly driven field (net energy IN, booked
    # as negative dissipation); into a vacuum bath the amplified field loses energy
    cold = floquet_run(V0, drive, 6, eta=0.9, nbar=0.0, fmap=run.fmap)
    assert cold.ledger_result.passed and cold.dissipated_total > 0.0, (
        f"vacuum-bath dissipation {cold.dissipated_total:.4e}"
    )


def test_pulse_train_converges_and_ramps_reduce_production(chain):
    K0, V0 = chain
    drive = modulated_K(K0, 0.1, 2 * OMEGA_RES, "sin")
    pt = pulse_train(V0, drive, 2, gap=3.0, n_periods_flat=2, n_periods_ramp=1,
                     n_substeps_per_period=8, conv_tol=1e-9, max_halvings=7)
    assert pt.converged and pt.movement < 1e-9
    res = pt.result
    assert res.all_passed() and abs(res.audits["ledger"].details["defect"]) < 1e-9
    gap_work = sum(e["work"] for e in res.step_log if e["label"].startswith("gap"))
    assert gap_work == 0.0
    off_work = sum(abs(e["work"]) for e in res.step_log if e["label"].startswith("off"))
    assert off_work < 1e-8  # telescoped remainder of the last piece, vanishing with dt
    assert pt.observables["n_total"] > 0.0 and res.work_total > 0.0
    # same number of driven periods (8) without ramps produces more photons
    run = floquet_run(V0, drive, 8, n_steps=8, conv_tol=1e-9)
    omega, _ = normal_modes(K0)
    assert np.sum(run.occupations[-1]) > pt.observables["n_total"], (
        f"unramped {np.sum(run.occupations[-1]):.4e} vs ramped train {pt.observables['n_total']:.4e}"
    )
    assert pt.t_final == pytest.approx(8 * drive.period + 2 * 3.0)


def test_bang_bang_bound_is_attained_and_never_exceeded():
    for d in (0.1, 0.3, 0.6):
        lam_bb, T_rel = bang_bang_bound(d)
        w1, w2 = np.sqrt(1 + d), np.sqrt(1 - d)
        t1, t2 = 0.5 * np.pi / w1, 0.5 * np.pi / w2
        T = t1 + t2
        drv = modulated_K(np.array([[1.0]]), d, 2 * np.pi / T, ((1.0, t1 / T), (-1.0, t2 / T)))
        fm = floquet_map(drv)
        assert fm.exact
        assert abs(np.log(fm.multiplier) - np.arctanh(d)) < 1e-12, (
            f"d={d}: attained ln lambda {np.log(fm.multiplier):.12f} vs artanh {np.arctanh(d):.12f}"
        )
        assert abs(lam_bb - np.exp(np.arctanh(d))) < 1e-12
        for profile in ("cos", "square", "triangle"):
            r = max_gain_over_frequency(d, profile, n_steps=16, conv_tol=1e-9, n_scan=41)
            assert r["ln_multiplier"] <= np.arctanh(d) + 1e-9, (
                f"{profile} d={d}: ln lambda {r['ln_multiplier']:.8f} exceeds artanh {np.arctanh(d):.8f}"
            )
            if profile == "cos":
                # growth *rate* of the cosine drive is (pi/4)(1 + O(d)) of the
                # bang-bang rate artanh(d)/T* (measured 0.785 -> 0.79 on this grid)
                bb_rate = np.arctanh(d) / (T_rel * np.pi)
                assert r["rate"] > 0.75 * bb_rate, (
                    f"cos d={d}: rate {r['rate']:.6f} vs bang-bang rate {bb_rate:.6f}"
                )


def test_amplification_scan_is_self_consistent():
    depths = np.array([0.05, 0.2, 0.5, 0.9])
    scan = amplification_bound_scan(depths, np.linspace(1.5, 2.5, 21), [1.0, 0.9, 0.5],
                                    n_steps=16, conv_tol=1e-9, omega_tol=1e-7)
    assert np.all(scan.eta_threshold == loss_threshold(np.exp(scan.ln_multiplier)))
    ratio = scan.ln_multiplier / scan.small_depth_ln_multiplier
    assert np.all(ratio >= 1.0 - 1e-6) and np.all(ratio <= 1.02), (
        f"cosine ln lambda / (pi d/4) = {ratio}"
    )
    assert np.all(scan.ln_multiplier <= scan.bang_bang_ln_multiplier)
    assert scan.conjecture["max_ratio_to_bang_bang_per_period"] < 1.0
    assert scan.gain.shape == (4, 21, 3)
    assert np.all(scan.gain[:, :, 2] <= scan.gain[:, :, 0])
    # the grid maximum never beats the refined maximum
    assert np.all(np.log(np.max(scan.multiplier_grid, axis=1)) <= scan.ln_multiplier + 1e-9)
