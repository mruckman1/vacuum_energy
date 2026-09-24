"""Mathieu-equation reference values for the single-mode parametric oscillator.

The sinusoidally driven mode x'' + omega^2 (1 + depth cos(omega_mod t)) x = 0
becomes, with z = omega_mod t / 2, the canonical Mathieu equation

    d^2x/dz^2 + (a - 2 q cos 2z) x = 0,
    a = 4 omega^2 / omega_mod^2,   q = -2 omega^2 depth / omega_mod^2,

(McLachlan, "Theory and Application of Mathieu Functions", Oxford 1947,
Sec. 2.10; Abramowitz-Stegun Sec. 20.1, Eq. 20.1.1).  Its stability chart
is bounded by the characteristic values a_m(q) (even solutions) and b_m(q)
(odd solutions), ``scipy.special.mathieu_a`` / ``mathieu_b``: for fixed q the
solutions are bounded (stable) for a between consecutive characteristic
values of opposite parity and unstable inside the resonance tongues
[min(a_m, b_m), max(a_m, b_m)] for m >= 1, and below a_0(q)
(Abramowitz-Stegun Fig. 20.1; McLachlan Sec. 4.10).  The tongues emanate
from a = m^2 at q = 0, i.e. from omega_mod = 2 omega / m — the parametric
resonances.  For q < 0 the characteristic values obey a_m(-q) = a_m(q),
b_m(-q) = b_m(q) for even m and a_m(-q) = b_m(q) for odd m (A-S 20.2.30),
which is why the tongue is written with min/max.

These are *reference* values for the Layer-5 anchor: the monodromy built by
:func:`vacuum.floquet.floquet_map` must reproduce |tr S_F| = 2 on the
characteristic curves and the tongue boundaries in omega_mod to the stated
tolerance (tests/test_floquet_mathieu.py).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import mathieu_a, mathieu_b

__all__ = [
    "mathieu_parameters",
    "mathieu_characteristic",
    "mathieu_tongue",
    "mathieu_is_stable",
    "mathieu_tongue_edges",
    "resonance_frequency",
]


def mathieu_parameters(omega, depth, omega_mod):
    """(a, q) of the Mathieu form for x'' + omega^2 (1 + depth cos omega_mod t) x = 0."""
    a = 4.0 * float(omega) ** 2 / float(omega_mod) ** 2
    q = -0.5 * a * float(depth)
    return a, q


def mathieu_characteristic(m, q):
    """(a_m(q), b_m(q)) with the q -> -q symmetry made explicit (b_0 = a_0)."""
    m = int(m)
    q = float(q)
    qa = abs(q)
    a_pos = float(mathieu_a(m, qa))
    b_pos = float(mathieu_b(m, qa)) if m >= 1 else a_pos
    if q >= 0.0 or m % 2 == 0:
        return a_pos, b_pos
    return b_pos, a_pos  # odd m: a_m(-q) = b_m(q), b_m(-q) = a_m(q)


def mathieu_tongue(m, q):
    """(lower, upper) edge in `a` of the m-th unstable tongue at this q."""
    a_m, b_m = mathieu_characteristic(m, q)
    return (min(a_m, b_m), max(a_m, b_m))


def mathieu_is_stable(a, q, m_max=60):
    """True iff (a, q) lies in a stable region of the Mathieu chart.

    Unstable: a < a_0(q), or inside a tongue [min, max](a_m, b_m), m >= 1.
    Boundary points count as unstable (|tr S_F| = 2 exactly there).
    """
    a = float(a)
    a0, _ = mathieu_characteristic(0, q)
    if a <= a0:
        return False
    for m in range(1, int(m_max) + 1):
        lo, hi = mathieu_tongue(m, q)
        if lo <= a <= hi:
            return False
        if lo > a:  # tongues are ordered in a; nothing above can contain a
            return True
    raise RuntimeError("increase m_max: a beyond the tabulated tongues")


def resonance_frequency(omega, m):
    """Tongue tip omega_mod = 2 omega / m (parametric resonance of order m)."""
    return 2.0 * float(omega) / int(m)


def mathieu_tongue_edges(omega, depth, m, bracket=0.5):
    """(omega_lo, omega_hi): edges in omega_mod of the m-th tongue at `depth`.

    Solves a(omega_mod) = edge_m(q(omega_mod)) for both edges of
    :func:`mathieu_tongue` by Brent's method around the tip 2 omega / m
    (the bracket is +-`bracket` relative).  Since a decreases with
    omega_mod, the *upper* tongue edge in `a` is the *lower* edge in
    omega_mod.
    """
    m = int(m)
    tip = resonance_frequency(omega, m)
    lo_w, hi_w = tip * (1.0 - bracket / m), tip * (1.0 + bracket / m)

    def g_upper(w):  # a - max edge: zero at the low-omega_mod boundary
        a, q = mathieu_parameters(omega, depth, w)
        return a - mathieu_tongue(m, q)[1]

    def g_lower(w):  # a - min edge: zero at the high-omega_mod boundary
        a, q = mathieu_parameters(omega, depth, w)
        return a - mathieu_tongue(m, q)[0]

    # a(omega_mod) is decreasing, so each g has exactly one sign change in
    # the bracket; the tongue is not centred on the q = 0 tip for m >= 2
    # (a_m(q) - m^2 = O(q^2)), so locate the sign change by a scan first.
    ws = np.linspace(lo_w, hi_w, 401)

    def _root(g):
        vals = np.array([g(w) for w in ws])
        idx = np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
        if idx.size == 0:
            raise ValueError(
                f"no tongue edge of order m={m} found in the bracket "
                f"[{lo_w:.6g}, {hi_w:.6g}] at depth {depth}"
            )
        i = int(idx[-1]) if g is g_upper else int(idx[0])
        return brentq(g, ws[i], ws[i + 1], xtol=1e-14, rtol=1e-14, maxiter=500)

    w_lo = _root(g_upper)
    w_hi = _root(g_lower)
    return float(w_lo), float(w_hi)
