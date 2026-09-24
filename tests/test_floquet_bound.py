"""Layer 5: the amplification bound is a theorem (vacuum/floquet/bounds.py).

Pinned:
- the Prüfer system (3)-(4) is the exact reduction of x'' + omega0^2 (1 + d s) x = 0
  (checked against the exact piecewise propagator to 1e-10);
- the per-cycle bound ln rho(S_F) <= Z artanh(d) (Theorem A / Corollary A)
  holds on the whole adversarial battery — five random families at random
  periods (winding numbers 1..3), projected-gradient and cross-entropy
  optimisers at five periods, four depths — with max ratio <= 1 + 1e-9,
  and the quarter-period bang-bang attains it to 1e-10 (the canary);
- the winding number Z is the tongue index on the Mathieu chart (Z = m on
  the m-th tongue of the cosine drive);
- the rate optimum of Theorem B: the Dinkelbach root equals a free
  two-dwell optimisation of the Meissner trace to 1e-9, its drive attains
  the rate to 1e-12, and the quarter-period bang-bang is strictly below it
  (measured excess 0.98 % at d = 0.3, 3.1 % at 0.5, 13.5 % at 0.8);
- multimode: for the commuting 'coupling' kind the 2N x 2N monodromy is
  block-diagonal (defect ~1e-14) and every mode obeys its own bound; for a
  non-commuting 'pattern' the *conjectured* rate bound with the
  operator-norm depth holds on a random battery (measured max ratio
  recorded in the assertion message; this one is tested, not proved).
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from vacuum.core import harmonic_chain_K
from vacuum.floquet.bounds import (
    CONTROL_FAMILIES,
    adversarial_search,
    bang_bang_control,
    bound_ratio,
    multimode_check,
    operator_norm_depth,
    per_cycle_bound,
    piecewise_monodromy,
    prufer_rhs,
    random_controls,
    rate_optimum,
    rate_optimum_control,
)

DEPTHS = (0.1, 0.3, 0.5, 0.8)


def _two_piece(d, t_high, t_low):
    """Exact 2 x 2 monodromy of +1 for t_high then -1 for t_low (omega0 = 1)."""
    M = piecewise_monodromy(np.array([[1.0, -1.0]]), np.array([[t_high, t_low]]), d)[0]
    tr = np.trace(M)
    return 0.0 if abs(tr) <= 2.0 else np.log(0.5 * (abs(tr) + np.sqrt(tr * tr - 4.0)))


def test_prufer_system_is_the_exact_reduction():
    """(3)-(4): integrate d ln r/dt, dtheta/dt with RK4 on a piecewise-constant
    control and compare with the exact propagator's r and theta."""
    d, w0 = 0.6, 1.3
    rng = np.random.default_rng(3)
    s = random_controls(rng, 1, 12, "piecewise")[0]
    tau = np.full(12, 0.35)
    x, p = 0.8, -0.3
    r0 = np.hypot(x, p / w0)
    th = np.arctan2(-p / w0, x)
    lnr = np.log(r0)
    n_sub = 4000
    for j in range(12):
        h = tau[j] / n_sub
        for _ in range(n_sub):
            def f(y):
                return np.array(prufer_rhs(y[1], s[j], d, w0))
            y = np.array([lnr, th])
            k1 = f(y); k2 = f(y + 0.5 * h * k1); k3 = f(y + 0.5 * h * k2); k4 = f(y + h * k3)
            lnr, th = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    M = piecewise_monodromy(s[None, :], tau[None, :], d, w0)[0]
    x1, p1 = M @ np.array([x, p])
    r1 = np.hypot(x1, p1 / w0)
    th1 = np.arctan2(-p1 / w0, x1)
    assert abs(lnr - np.log(r1)) < 1e-10, f"ln r: Prüfer {lnr:.12f} vs exact {np.log(r1):.12f}"
    assert abs((th - th1 + np.pi) % (2 * np.pi) - np.pi) < 1e-9
    assert th > np.arctan2(-p / w0, x)  # the phase only ever increases


def test_bang_bang_attains_the_bound_and_rate_optimum_attains_its_rate():
    for d in DEPTHS:
        v, t = bang_bang_control(d)
        b = bound_ratio(v[None, :], t[None, :], d)
        assert abs(b["ratio"][0] - 1.0) < 1e-10, f"d={d}: canary ratio {b['ratio'][0]:.14f}"
        assert b["Z"][0] == 1.0
        assert abs(b["ln_multiplier"][0] - per_cycle_bound(d)) < 1e-12
        r = rate_optimum(d)
        vr, tr = rate_optimum_control(d)
        br = bound_ratio(vr[None, :], tr[None, :], d)
        rate = br["ln_multiplier"][0] / np.sum(tr)
        assert abs(rate - r.lambda_star) < 1e-12, f"d={d}: rate {rate:.14f} vs lambda* {r.lambda_star:.14f}"
        assert br["ratio"][0] < 1.0 - 1e-6  # the rate optimum gains less per cycle
        assert abs(br["ln_multiplier"][0] - r.ln_growth_per_cycle) < 1e-12


def test_rate_optimum_matches_free_two_dwell_optimisation():
    """Theorem B against a Nelder-Mead search over the two dwell times."""
    measured = {}
    for d in (0.3, 0.5, 0.8):
        r = rate_optimum(d)
        w2, w1 = np.sqrt(1 + d), np.sqrt(1 - d)
        q2, q1 = 0.5 * np.pi / w2, 0.5 * np.pi / w1
        best = None
        for f2 in (0.9, 1.0, 1.2):
            for f1 in (0.6, 0.85, 1.0):
                res = minimize(lambda x: -_two_piece(d, x[0], x[1]) / (x[0] + x[1]),
                               x0=[q2 * f2, q1 * f1], method="Nelder-Mead",
                               options=dict(xatol=1e-12, fatol=1e-15, maxiter=5000))
                if best is None or res.fun < best.fun:
                    best = res
        free_rate = -best.fun
        assert abs(free_rate - r.lambda_star) < 1e-9, (
            f"d={d}: free two-dwell optimum {free_rate:.12f} vs Dinkelbach root {r.lambda_star:.12f}"
        )
        assert abs(best.x[0] - r.tau_high) < 1e-5 and abs(best.x[1] - r.tau_low) < 1e-5
        assert r.excess > 0.0 and r.tau_high > q2 and r.tau_low < q1
        measured[d] = r.excess
    assert 0.009 < measured[0.3] < 0.011 and 0.03 < measured[0.5] < 0.033 and 0.13 < measured[0.8] < 0.14, (
        f"quarter-period bang-bang rate deficit {measured} (expected 0.98 %, 3.1 %, 13.5 %)"
    )


def test_winding_number_is_the_tongue_index():
    """Z = m inside the m-th cosine tongue: the 'per cycle' normalisation of Corollary A.

    omega_mod is taken at the centre of the scipy tongue (``mathieu_tongue_edges``);
    the sampled cosine (512 pieces) is an admissible control in its own right.
    """
    from vacuum.floquet import mathieu_tongue_edges

    d = 0.5
    M = 512
    for m in (1, 2, 3):
        lo, hi = mathieu_tongue_edges(1.0, d, m)
        wmod = 0.5 * (lo + hi)
        T = 2 * np.pi / wmod
        phi = 2 * np.pi * (np.arange(M) + 0.5) / M
        s = np.cos(phi)[None, :]
        b = bound_ratio(s, np.full(M, T / M), d)
        assert b["ln_multiplier"][0] > 0.0, f"m={m}: tongue centre {wmod:.6f} came out stable"
        assert b["Z"][0] == m, f"m={m}: winding {b['Z'][0]} (theta advance {b['theta_advance'][0]:.6f})"
        assert b["ratio"][0] <= 1.0 + 1e-9
        # per period the m-th tongue may exceed artanh(d) (Z = m cycles), never m artanh(d)
        if m >= 2:
            assert b["ln_multiplier"][0] <= m * per_cycle_bound(d)


def test_adversarial_battery_never_beats_the_bound():
    res = adversarial_search(depths=DEPTHS, seed=7, n_random=150, M=48, n_iter=120,
                             n_restarts=3, periods_over_Tstar=(0.85, 1.0, 1.15, 2.0, 2.4))
    assert res["max_ratio"] <= 1.0 + 1e-9, f"bound violated: {res['worst_row']}"
    assert res["bang_bang_ratio_max_dev"] < 1e-10
    fams = {r["family"] for r in res["rows"]}
    assert set(CONTROL_FAMILIES) <= fams and "opt_gradient" in fams and "opt_cem" in fams
    Zs = {r["Z"] for r in res["rows"] if np.isfinite(r["Z"])}
    assert {1.0, 2.0} <= Zs, f"winding numbers explored: {Zs}"
    # the optimisers get close to the bound (a uniform grid cannot place the switch exactly)
    best_opt = max(r["ratio"] for r in res["rows"] if r["family"] == "opt_gradient")
    assert best_opt > 0.99, f"gradient ascent reached only {best_opt:.4f} of the bound"


def test_coupling_kind_is_block_diagonal_and_mode_bounded():
    K0 = harmonic_chain_K(8, 0.5, bc="dirichlet")
    rng = np.random.default_rng(11)
    worst_defect, worst_ratio = 0.0, 0.0
    for T in (np.pi / 1.5, np.pi / 0.8, np.pi / 0.55):
        for fam in ("clipped_fourier", "sign_fourier"):
            s = random_controls(rng, 1, 32, fam)[0]
            c = multimode_check(K0, s, T, 0.4, kind="coupling")
            assert c["exact"] and c["symplectic_defect"] < 1e-12
            worst_defect = max(worst_defect, c["block_defect"])
            worst_ratio = max(worst_ratio, c["max_ratio"])
    assert worst_defect < 1e-12, f"block-diagonalisation defect {worst_defect:.2e}"
    assert worst_ratio <= 1.0 + 1e-9, f"a normal mode beat its own bound: ratio {worst_ratio:.12f}"


def test_pattern_kind_rate_bound_holds_on_a_random_battery():
    """CONJECTURED multimode extension (not proved): ln rho/T <= omega_max lambda*(d_eff)."""
    K0 = harmonic_chain_K(6, 0.5, bc="dirichlet")
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(12):
        P = rng.normal(size=(6, 6))
        P = 0.5 * (P + P.T)
        d = 0.5 / operator_norm_depth(K0, P, 1.0) * rng.uniform(0.3, 1.0)  # d_eff in [0.15, 0.5]
        s = random_controls(rng, 1, 24, "sign_fourier")[0]
        T = rng.uniform(1.0, 4.0)
        c = multimode_check(K0, s, T, d, kind="pattern", pattern=P)
        assert c["exact"] and 0.0 < c["d_eff"] <= 0.5 + 1e-12
        worst = max(worst, c["rate_ratio"])
    assert worst <= 1.0 + 1e-9, f"pattern kind: rate ratio {worst:.6f} exceeds the conjectured bound"
    assert worst > 0.0


def test_mass_drive_bound_holds_per_mode_and_refuses_the_excluded_regime():
    """kind='mass': the bound is per-mode with d_k = depth m^2/omega_k^2 (Eq. (13)),
    and d_max >= 1 is out of contract, not a violation.

    The audit's counterexample (mass_sq = 0.6 at depth 1 on
    harmonic_chain_K(6, 0.05)) must raise rather than return a ratio > 1:
    d_max = 2.99 on the Dirichlet chain, 240 on the periodic one.
    """
    from vacuum.floquet.bounds import mass_mode_depths

    K0 = harmonic_chain_K(6, 0.05, bc="dirichlet")
    K0_per = harmonic_chain_K(6, 0.05, bc="periodic")
    rng = np.random.default_rng(23)
    s = random_controls(rng, 1, 24, "sign_fourier")[0]

    for K, expect in ((K0, 2.9916), (K0_per, 240.0)):
        info = mass_mode_depths(K, 0.6, 1.0)
        assert not info["valid"], f"d_max = {info['d_max']:.6g} wrongly reported in contract"
        assert abs(info["d_max"] - expect) < 1e-3 * expect
        with pytest.raises(ValueError, match="out of contract"):
            multimode_check(K, s, 2.0, 1.0, kind="mass", pattern=0.6)

    # the boundary itself: mass_sq_max is exactly omega_min^2/depth
    w2_min = float(np.min(np.linalg.eigvalsh(K0)))
    for depth in (0.3, 1.0):
        info = mass_mode_depths(K0, w2_min / depth, depth)
        assert abs(info["d_max"] - 1.0) < 1e-12  # the boundary itself, to roundoff
        assert abs(info["mass_sq_max"] - w2_min / depth) < 1e-12 * w2_min / depth
        just_inside = mass_mode_depths(K0, 0.999 * w2_min / depth, depth)
        assert just_inside["valid"] and just_inside["d_max"] < 1.0
        just_outside = mass_mode_depths(K0, 1.001 * w2_min / depth, depth)
        assert not just_outside["valid"] and just_outside["d_max"] > 1.0
        with pytest.raises(ValueError, match="out of contract"):
            multimode_check(K0, s, 2.0, depth, kind="mass",
                            pattern=1.001 * w2_min / depth)

    # in contract: every mode obeys its own artanh(d_k), and the block structure holds
    worst = 0.0
    for m_sq, depth in ((0.02, 0.5), (0.05, 0.3)):
        info = mass_mode_depths(K0, m_sq, depth)
        assert info["valid"]
        for T in (2.0, 3.4):
            c = multimode_check(K0, s, T, depth, kind="mass", pattern=m_sq)
            assert c["exact"] and c["block_defect"] < 1e-12
            assert np.allclose(c["d_modes"], depth * m_sq / info["omega"] ** 2)
            worst = max(worst, c["max_ratio"])
    assert worst <= 1.0 + 1e-9, f"a mass-drive mode beat its own artanh(d_k): {worst:.12f}"
