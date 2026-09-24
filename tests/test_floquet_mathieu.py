"""Anchor 1 (Layer 5): the Mathieu stability chart from the parametric oscillator.

The single-mode drive x'' + omega^2 (1 + d cos omega_mod t) x = 0 is the
Mathieu equation with a = 4 omega^2/omega_mod^2, q = -2 omega^2 d/omega_mod^2
(McLachlan 1947, Sec. 2.10; Abramowitz-Stegun 20.1.1).  The resonance
tongues emanate from omega_mod = 2 omega/m and their boundaries are the
characteristic values a_m(q), b_m(q) (``scipy.special.mathieu_a/b``).

Pinned:
- |tr S_F| = 2 on the Mathieu characteristic curves (tongue edges of order
  m = 1..4 at depths 0.1, 0.3, 0.6) to 1e-10 — the monodromy's trace is
  2 cos(pi nu) with nu integer/half-integer exactly on the edges;
- tongue edges located by root-finding on the numerical |tr S_F| - 2 agree
  with the scipy characteristic values to 1e-8 relative in omega_mod;
- the stability classification of a 16 x 300 (depth, omega_mod) grid
  agrees with the Mathieu chart on every point;
- tongue tips approach 2 omega/m as depth -> 0;
- the growth rate at the principal resonance is depth*omega/4 (1 + O(d^2))
  (Landau-Lifshitz Sec. 27).
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from vacuum.floquet import (
    floquet_map,
    mathieu_is_stable,
    mathieu_parameters,
    mathieu_tongue_edges,
    max_gain_over_frequency,
    modulated_K,
    resonance_frequency,
    stability_chart,
)

K1 = np.array([[1.0]])
EDGE_TRACE_TOL = 1e-10
EDGE_LOCATION_TOL = 1e-8
DEPTHS = (0.1, 0.3, 0.6)
ORDERS = (1, 2, 3, 4)


def _trace(depth, omega_mod, conv_tol=1e-12):
    fm = floquet_map(modulated_K(K1, depth, omega_mod, "cos"), n_steps=64,
                     method="cf4", conv_tol=conv_tol, max_halvings=10)
    assert fm.converged
    return float(np.trace(fm.S))


@pytest.mark.parametrize("m", ORDERS)
@pytest.mark.parametrize("depth", DEPTHS)
def test_trace_is_two_on_mathieu_edges(m, depth):
    lo, hi = mathieu_tongue_edges(1.0, depth, m)
    assert lo < resonance_frequency(1.0, m) * (1 + 0.02) and hi > lo
    for w in (lo, hi):
        defect = abs(abs(_trace(depth, w)) - 2.0)
        assert defect < EDGE_TRACE_TOL, (
            f"m={m} depth={depth} edge omega_mod={w:.12f}: |tr S_F| - 2 = {defect:.3e}"
        )
    # inside the tongue |tr| > 2, just outside |tr| < 2
    mid = 0.5 * (lo + hi)
    assert abs(_trace(depth, mid)) > 2.0
    gap = hi - lo
    assert abs(_trace(depth, lo - 0.5 * gap)) < 2.0
    assert abs(_trace(depth, hi + 0.5 * gap)) < 2.0


@pytest.mark.parametrize("m", (1, 2))
@pytest.mark.parametrize("depth", (0.1, 0.3))
def test_numerical_edges_match_characteristic_values(m, depth):
    lo, hi = mathieu_tongue_edges(1.0, depth, m)
    gap = hi - lo

    def g(w):
        return abs(_trace(depth, w)) - 2.0

    w_lo = brentq(g, lo - 0.25 * gap, lo + 0.25 * gap, xtol=1e-13, rtol=1e-13)
    w_hi = brentq(g, hi - 0.25 * gap, hi + 0.25 * gap, xtol=1e-13, rtol=1e-13)
    for w_num, w_ref, name in ((w_lo, lo, "lower"), (w_hi, hi, "upper")):
        rel = abs(w_num - w_ref) / w_ref
        assert rel < EDGE_LOCATION_TOL, (
            f"m={m} depth={depth} {name} edge: numerical {w_num:.12f} vs "
            f"Mathieu {w_ref:.12f} (rel {rel:.2e})"
        )


def test_chart_classification_matches_mathieu():
    depths = np.linspace(0.05, 0.8, 16)
    omegas = np.linspace(0.3, 4.0, 300)
    chart = stability_chart(K1, depths, omegas, n_steps=32, conv_tol=1e-10)
    assert chart.info["converged"]
    mism = 0
    for i, d in enumerate(depths):
        for j, w in enumerate(omegas):
            a, q = mathieu_parameters(1.0, d, w)
            if mathieu_is_stable(a, q) != (not chart.unstable[i, j]):
                mism += 1
    assert mism == 0, f"{mism} of {chart.unstable.size} grid points misclassified"
    # tongues of order 1, 2, 3 are resolved on the grid at the largest depth:
    # the grid point nearest the Mathieu tongue centre is unstable (the
    # tongue is displaced from the q = 0 tip 2/m by O(q^2) at finite depth)
    row = chart.unstable[-1]
    spacing = omegas[1] - omegas[0]
    for m in (1, 2, 3):
        lo, hi = mathieu_tongue_edges(1.0, depths[-1], m)
        assert hi - lo > spacing, f"m={m} tongue narrower than the grid at depth {depths[-1]}"
        j = int(np.argmin(np.abs(omegas - 0.5 * (lo + hi))))
        assert row[j], f"no instability at the m={m} tongue centre {0.5 * (lo + hi):.4f}"


def test_tongue_tips_approach_two_omega_over_m():
    for m in (1, 2, 3):
        lo, hi = mathieu_tongue_edges(1.0, 1e-3, m)
        tip = resonance_frequency(1.0, m)
        assert lo <= tip * (1 + 1e-3) and hi >= tip * (1 - 1e-3)
        assert (hi - lo) / tip < 2e-3 * (1 if m == 1 else 0.1), (
            f"m={m}: tongue width {hi - lo:.3e} at depth 1e-3"
        )


def test_principal_resonance_growth_rate():
    for d, tol in ((0.02, 2e-4), (0.05, 5e-4), (0.1, 2e-3)):
        r = max_gain_over_frequency(d, "cos", n_steps=16, conv_tol=1e-10)
        expected = np.pi * d / 4.0  # ln lambda per period T = pi: rate d/4
        ratio = r["ln_multiplier"] / expected
        assert abs(ratio - 1.0) < tol, (
            f"depth {d}: ln lambda {r['ln_multiplier']:.8f} vs pi d/4 {expected:.8f} "
            f"(ratio {ratio:.6f}) at omega_mod {r['omega_mod']:.6f}"
        )
        assert abs(r["omega_mod"] - 2.0) < 0.1
