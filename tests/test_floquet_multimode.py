"""The non-commuting multimode bound (papers/time-modulated-vacua, Layer 5, 2026-09-12 pass).

Pins, in order:
- Theorem M (vacuum/floquet/bounds_multimode.py): the proved chain
  (d/2)||K0^{-1/4} P K0^{-1/4}|| <= (d/2)||P K0^{-1/2}|| <= omega_max d_eff/2 on random systems,
  and the single-mode constant lambda*(d) < d/2 (the rate optimum reaches 0.6758 of the proved
  bound at d = 0.9, never more);
- the mutation: with the first-order constant 1/pi in place of 1/2 the "bound" is violated by
  the single-mode rate optimum embedded in a two-mode K0 (ratio 1.0616 at d = 0.9), so a wrong
  constant is caught; a wrong norm (omega_min in place of the K0^{-1/4} weighting) is caught too;
- Theorem F: the two-mode sum-frequency growth rate (d_eff/pi) sqrt(omega1 omega2) under the
  square wave, exact to 2e-4 at d_eff = 0.02, and its ratio sqrt(omega1) (d/pi)/lambda*(d) to the
  conjectured bound;
- the archived battery (60 random + 3 attacked patterns of summary.json, regenerated bit-exactly
  by notebook_multimode.py) satisfies the proved bound: max proved ratio 0.26059, max refined
  ratio 0.40089, max conjectured ratio 0.16872, and the proved bound lies below the conjectured
  one on all 63 rows (0.492 to 0.870 of it);
- the verdict file: the attack never exceeded the conjectured bound (max ratio <= 1 + 1e-6, and it
  does reach 1 from below), the proved ratio stays below 0.7, the first-order Schur search stays
  at or below 2/pi (grid value);
- multimode_check(kind='pattern') reports proved_ratio / refined_ratio consistently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vacuum.core.models import harmonic_chain_K
from vacuum.floquet import (
    frequency_weighted_norm,
    multimode_attack,
    multimode_bound_proved,
    multimode_check,
    multimode_ratios,
    normal_mode_system,
    operator_norm_depth,
    pattern_monodromy,
    random_controls,
    rate_optimum,
    rate_optimum_control,
    rwa_growth_rate,
    sum_frequency_first_order,
)

DATA = Path(__file__).resolve().parents[1] / "papers" / "time-modulated-vacua" / "data" / "s3_multimode_verdict.json"


def _ln_rho(S):
    return float(max(0.0, np.log(np.max(np.abs(np.linalg.eigvals(S))))))


def test_theorem_M_chain_on_random_systems_and_the_single_mode_constant():
    rng = np.random.default_rng(12)
    for N in (2, 3, 6):
        for _ in range(5):
            Q = rng.normal(size=(N, N))
            K0 = Q @ Q.T + N * np.eye(N)
            P = rng.normal(size=(N, N))
            P = 0.5 * (P + P.T)
            d = 0.6 / operator_norm_depth(K0, P, 1.0)
            mb = multimode_bound_proved(K0, P, d)
            assert mb["rate_bound"] <= mb["energy_form"] * (1 + 1e-12) <= mb["omega_max_form"] * (1 + 1e-12)
            assert mb["rate_bound"] == pytest.approx(0.5 * d * frequency_weighted_norm(K0, P))
            assert mb["d_eff"] == pytest.approx(0.6)
            # basis covariance: an orthogonal change of basis leaves every member of the chain unchanged
            U, _ = np.linalg.qr(rng.normal(size=(N, N)))
            mb2 = multimode_bound_proved(U @ K0 @ U.T, U @ P @ U.T, d)
            for k in ("rate_bound", "energy_form", "omega_max_form"):
                assert mb2[k] == pytest.approx(mb[k], rel=1e-10)
    # N = 1: the proved bound is omega0 d/2 and Theorem B's optimum sits below it: 0.6366 (d -> 0) .. 0.6890 (d -> 1)
    for d, expect in ((0.1, 0.637000), (0.5, 0.646732), (0.9, 0.675809)):
        mb = multimode_bound_proved([[4.0]], [[4.0]], d)
        assert mb["rate_bound"] == pytest.approx(2.0 * d / 2)
        assert 2.0 * rate_optimum(d).lambda_star / mb["rate_bound"] == pytest.approx(expect, abs=2e-6)


def test_wrong_constant_or_wrong_norm_in_the_proved_bound_is_caught():
    """The rate optimum of the omega_max mode of a two-mode K0 (commuting P) is an exact solution
    whose growth rate is lambda*(d): it sits at 0.6758 of Theorem M at d = 0.9 and VIOLATES the
    same formula with the first-order constant 1/pi (1.0616) or with omega_min in place of the
    K0^{-1/4} weighting (3.4/1)."""
    d = 0.9
    K0 = np.diag([0.3 ** 2, 1.0])
    P = np.diag([0.0, 1.0])
    v, tau = rate_optimum_control(d, 1.0)
    S = pattern_monodromy(K0, P, d, v, tau)
    rate = _ln_rho(S) / float(np.sum(tau))
    assert rate == pytest.approx(rate_optimum(d).lambda_star, rel=1e-9)
    mb = multimode_bound_proved(K0, P, d)
    assert rate / mb["rate_bound"] == pytest.approx(0.675809, abs=2e-6)
    assert rate < mb["rate_bound"]
    wrong_constant = (1.0 / np.pi) * d * mb["norm_C"]
    assert rate / wrong_constant == pytest.approx(1.061558, abs=2e-6) and rate > wrong_constant
    wrong_norm = 0.5 * d * 0.3 * mb["norm_A"]  # omega_min ||A|| instead of ||K0^{-1/4} P K0^{-1/4}||
    assert rate > wrong_norm and rate / wrong_norm == pytest.approx(0.675809 / 0.3, rel=1e-5)


def test_theorem_F_two_mode_sum_frequency_first_order():
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    for w1 in (0.3, 0.9):
        om = np.array([w1, 1.0])
        T = 2 * np.pi / (w1 + 1.0)
        s, taus = np.array([1.0, -1.0]), np.array([T / 2, T / 2])
        r = multimode_ratios(om, A, 0.02, s, taus)
        fo = sum_frequency_first_order(w1, 1.0, 0.02)
        assert fo["rate"] == pytest.approx(0.02 / np.pi * np.sqrt(w1))
        assert r["rate"] / fo["rate"] == pytest.approx(1.0, abs=2e-4)
        assert r["ratio_conjectured"] == pytest.approx(np.sqrt(w1) * (0.02 / np.pi) / rate_optimum(0.02).lambda_star, abs=2e-4)
        K0, P, _ = normal_mode_system(om, A)
        assert rwa_growth_rate(K0, P, 0.02, s, taus) == pytest.approx(fo["rate"], rel=1e-3)
        assert r["ratio_refined"] > r["ratio_conjectured"] and r["ratio_proved"] < 0.7
    # the first-order rate never exceeds the conjectured bound: sqrt(w1) (d/pi)/lambda*(d) < 1
    assert sum_frequency_first_order(0.99, 1.0, 0.05)["ratio"] < 1.0


def test_proved_bound_holds_on_the_random_pattern_battery_of_test_floquet_bound():
    K0 = harmonic_chain_K(6, 0.5, bc="dirichlet")
    rng = np.random.default_rng(5)
    worst_p, worst_r, worst_c = 0.0, 0.0, 0.0
    for _ in range(12):
        P = rng.normal(size=(6, 6))
        P = 0.5 * (P + P.T)
        d = 0.5 / operator_norm_depth(K0, P, 1.0) * rng.uniform(0.3, 1.0)
        s = random_controls(rng, 1, 24, "sign_fourier")[0]
        T = rng.uniform(1.0, 4.0)
        c = multimode_check(K0, s, T, d, kind="pattern", pattern=P)
        assert c["proved_ratio"] == pytest.approx(c["ln_full"] / (T * c["proved_bound"]))
        assert c["proved_bound"] == pytest.approx(0.5 * d * frequency_weighted_norm(K0, P))
        worst_p, worst_r, worst_c = max(worst_p, c["proved_ratio"]), max(worst_r, c["refined_ratio"]), max(worst_c, c["rate_ratio"])
    assert 0.0 < worst_p < 1.0 and worst_r <= 1.0 and worst_c <= 1.0


def test_archived_battery_satisfies_the_proved_bound():
    with open(DATA) as fh:
        arc = json.load(fh)["archive"]
    assert arc["archive_match"]["n_archived"] == 63 and arc["archive_match"]["max_abs_dln_full"] == 0.0
    assert arc["n_rows"] == 63 and arc["proved_below_conjectured_count"] == 63
    assert arc["max_rate_ratio"] == pytest.approx(0.16872334442430495, abs=1e-9)
    assert arc["max_proved_ratio"] == pytest.approx(0.2605884231387432, abs=1e-9)
    assert arc["max_refined_ratio"] == pytest.approx(0.40088516969801463, abs=1e-9)
    assert arc["proved_over_conjectured_min"] == pytest.approx(0.49244246, abs=1e-6)
    assert arc["proved_over_conjectured_max"] == pytest.approx(0.86974427, abs=1e-6)
    for row in arc["rows"]:
        assert row["proved_ratio"] == pytest.approx(row["ln_full"] / (row["period"] * row["proved_bound"]), abs=1e-12)
        assert row["proved_ratio"] < 1.0 and row["proved_bound"] < row["conjectured_bound"]


def test_verdict_attack_never_exceeds_the_conjecture_and_reaches_it_from_below():
    with open(DATA) as fh:
        v = json.load(fh)
    assert v["quick"] is False and v["__build__"]["git_head"] and v["__build__"]["wall_clock_s"] > 0
    verdict = v["verdict"]
    assert verdict["conjecture_refuted"] is False
    assert 0.999 < verdict["max_ratio_conjectured"] <= 1.0 + 1e-6
    assert verdict["max_ratio_refined"] <= 1.0 + 1e-6
    assert 0.6 < verdict["max_ratio_proved"] < 0.7
    # first-order agreement: 1.2e-3 over d_eff <= 0.05 (worst at omega1 = 0.1, where d||C||/omega_min is
    # largest), 6e-5 at d_eff = 0.02 with omega1 >= 0.3
    assert v["sum_frequency"]["max_abs_dev_first_order_small_d"] < 2e-3
    rows = [r for r in v["sum_frequency"]["rows"] if r["d_eff"] == 0.02 and r["omega1"] >= 0.3]
    assert rows and max(abs(r["exact_over_first_order"] - 1.0) for r in rows) < 1e-4
    assert v["schur"]["max_over_two_over_pi"] <= 1.0 + 1e-3
    assert v["schur"]["max_exact_refined_ratio_d001"] <= 1.0 + 1e-6


def test_multimode_attack_smoke_finds_the_single_mode_optimum_from_below():
    r = multimode_attack(2, 4, 0.3, rng=np.random.default_rng(1), n_iter=30, batch=48, polish_iter=500)
    assert 0.95 < r["ratio_conjectured"] <= 1.0 + 1e-6
    assert r["ratio_proved"] < 0.7 and r["ratio_refined"] >= r["ratio_conjectured"] - 1e-12


def _number_metric_sqrt(K0):
    """W^{1/2} for the number form N = (p.K0^{-1/2}p + x.K0^{1/2}x)/2 on (x, p)."""
    w2, U = np.linalg.eigh(0.5 * (np.asarray(K0, float) + np.asarray(K0, float).T))
    n = w2.size
    Om, Omi = (U * np.sqrt(w2)) @ U.T, (U / np.sqrt(w2)) @ U.T
    W = 0.5 * np.block([[Om, np.zeros((n, n))], [np.zeros((n, n)), Omi]])
    e, V = np.linalg.eigh(W)
    return (V * np.sqrt(e)) @ V.T


def test_theorem_M_operator_norm_form_is_tight_and_the_wrong_constant_is_caught():
    """(M2) in its strong form: the operator norm of S_F in the number metric — not only the
    spectral radius — obeys ln ||S_F||_N <= (d/2)||C|| int|s|, on 40 random (K0, P, s, taus) with
    N <= 4 and d_eff up to 0.95.  The batch maximum 0.879585 is the calibration: a bound inflated
    by more than 14 % would no longer be reached, and the same statement with the first-order
    constant 1/pi in place of 1/2 is VIOLATED by the top row of the batch (mutated ratio 1.38165)."""
    rng = np.random.default_rng(17)
    ratios = []
    for _ in range(40):
        N = int(rng.integers(1, 5))
        Q = rng.normal(size=(N, N))
        K0 = Q @ Q.T + 0.3 * N * np.eye(N)
        P = rng.normal(size=(N, N))
        P = 0.5 * (P + P.T)
        d = rng.uniform(0.05, 0.95) / operator_norm_depth(K0, P, 1.0)
        M = int(rng.integers(2, 9))
        s, taus = rng.uniform(-1, 1, M), rng.uniform(0.05, 3.0, M)
        S = pattern_monodromy(K0, P, d, s, taus)
        bound = 0.5 * d * frequency_weighted_norm(K0, P) * float(np.sum(np.abs(s) * taus))
        Wh = _number_metric_sqrt(K0)
        ratios.append(float(np.log(np.linalg.norm(Wh @ S @ np.linalg.inv(Wh), 2))) / bound)
    ratios = np.array(ratios)
    assert np.all(ratios <= 1.0 + 1e-12)                      # Theorem M, operator-norm form
    assert max(ratios) == pytest.approx(0.879585, abs=1e-5)   # near-tight: pins the constant
    mutated = ratios / (2.0 / np.pi)                          # constant 1/pi instead of 1/2
    assert int(np.sum(mutated > 1.0)) == 1 and max(mutated) == pytest.approx(1.3816483, abs=2e-6)


def test_corollary_M1_the_measured_range_of_lambda_star_over_d_half():
    """lambda*(d)/(d/2) on a 200-point grid of (1e-6, 1 - 1e-6): monotone increasing from 2/pi
    (d -> 0) to 0.68909937 (d -> 1).  Theorem M at N = 1 proves only lambda*(d) <= d/2; the two
    endpoints quoted in the draft and the MANIFEST are measured here, not proved."""
    g = np.linspace(1e-6, 1.0 - 1e-6, 200)
    v = np.array([rate_optimum(x).lambda_star / (0.5 * x) for x in g])
    assert np.all(np.diff(v) > 0)
    assert v.min() == pytest.approx(2.0 / np.pi, abs=1e-8)
    assert v.max() == pytest.approx(0.68909937, abs=2e-8)
    assert np.all(v < 1.0)


def test_verdict_per_section_maxima_are_the_ones_quoted_in_the_MANIFEST():
    """The section-by-section maxima of the attack, quoted in the MANIFEST's numbers table.  The
    CEM attack (36 runs) stays strictly BELOW 1 (0.9999999999999953) and it is the 36 detuned-pair
    polishes that touch 1 from above by 1.7e-14 of roundoff — the first pass's draft attributed the
    detuned maximum to the CEM attack, which this test now makes impossible to repeat."""
    with open(DATA) as fh:
        v = json.load(fh)
    att, det, sf, sch = v["attack"], v["detuned"], v["sum_frequency"], v["schur"]
    assert (att["n"], det["n"], len(sf["rows"]), len(sch["rows"])) == (36, 36, 56, 9)
    assert 1.0 - 1e-12 < att["max_ratio_conjectured"] < 1.0          # strictly below the conjecture
    assert 1.0 < det["max_ratio_conjectured"] < 1.0 + 1e-12          # above only by ln-rho roundoff
    assert 1.0 < det["max_ratio_refined"] < 1.0 + 1e-12
    assert att["max_ratio_proved"] == pytest.approx(0.66572804, abs=1e-8)
    assert min(r["ratio_conjectured"] for r in att["rows"]) == pytest.approx(0.7256884, abs=1e-7)
    assert sf["max_ratio_conjectured"] == pytest.approx(0.99494292, abs=1e-8)
    # the sum-frequency rows sit at 2/pi of Theorem M (the first-order value), to 1.3e-5
    assert max(r["ratio_proved"] for r in sf["rows"]) == pytest.approx(0.63660648, abs=1e-8)
    assert max(r["ratio_proved"] for r in sf["rows"]) == pytest.approx(2.0 / np.pi, abs=2e-5)
    assert sch["max_over_two_over_pi"] == pytest.approx(0.98641854, abs=1e-8)
    assert v["verdict"]["n_polished_optima"] == 128
