"""Closed forms for the half-infinite entanglement Hamiltonian of the gapped (staggered-mass) chain.

THE RESULT AND ITS SOURCE
-------------------------
For exactly the chain of vacuum.modular.lattice,

    H = -(1/2) sum_n (c_n^dag c_{n+1} + h.c.) + mu sum_n (-1)^n c_n^dag c_n,

V. Eisler, "On the Bisognano-Wichmann entanglement Hamiltonian of nonrelativistic
fermions", J. Stat. Mech. (2025) 013101, arXiv:2410.16433, Sec. V, Eqs. (64)-(72),
DERIVES the entanglement Hamiltonian of the half-infinite chain n >= 1 in closed form:

    Eq. (64)  the Hamiltonian above (his mu is our m; hopping 1/2 in both);
    Eq. (67)  the tridiagonal operator T that commutes with the half-chain
              correlation matrix (proof: his Appendix A.4),
                  t_m = -(1/2) m  (bond m, m+1),   d_m = (-1)^m mu (m - 1/2)  (site m);
    Eq. (70)  the modulus  kappa = 1/sqrt(1 + mu^2)  from the phase
              phi_q = kappa F(q, kappa) of the exact eigenvector ansatz (68)-(69);
    Eq. (71)  the quantization  lambda_l = pi/(2 kappa K(kappa)) (2l -/+ 1/2), l in Z,
              the upper sign for mu > 0 and the lower for mu < 0;
    Eq. (72)  H = 4 kappa K(kappa') T,
              eps_l = 2 pi K(kappa')/K(kappa) * (2l - 1/2)  [mu > 0]  or  (2l + 1/2)  [mu < 0].

K is the complete elliptic integral of the first kind in the MODULUS convention
(his F(q, kappa), K(delta')); mpmath.ellipk takes the parameter kappa^2. The derivation
is by the commuting-operator / exact-eigenvector method, NOT by a corner transfer
matrix: Eisler remarks (Sec. VI) that the staggered chain's shifted level structure
differs from the dimerized chain's CTM result and that it is unclear whether any
classical 2D integrable model's CTM reproduces it. The corner-transfer-matrix
literature (Peschel-Kaulke-Legeza, Ann. Phys. (Leipzig) 8, 153 (1999), Eqs. (9)-(15);
Eisler-Di Giulio-Tonni-Peschel, J. Stat. Mech. (2020) 103102, Eqs. (38)-(41)) supplies
the same STRUCTURE -- physical couplings multiplied by their distance from the boundary,
an equidistant ladder, eps = pi K(k')/K(k) -- for the transverse Ising and dimerized
hopping chains, which is what this module was first written against; the function names
below keep the "ctm_" prefix for API stability, but the closed form they implement for
THIS chain is Eisler 2025's.

DICTIONARY (Eisler 2025 -> this module), stated once
-----------------------------------------------------
Sites: his n = 1, 2, ... from the boundary; ours j = n - 1 = 0, 1, ... with site j at
x = j + 1/2 and bond (j, j+1) at x = j + 1 (dictionary D1 of casini_huerta.py).
Sign of the mass at the edge: his site 1 carries -mu, our site 0 carries +m, so our
LEFT edge is his chain with mu = -m; our right edge (site L-1, potential m(-1)^{L-1},
i.e. -m for even L), read after reflection, is his chain with mu = +m.

    our h_{j,j+1} = -2 pi s (j+1) (1/2) = 2 pi s * t_{j+1},
    our h_{j,j}   =  2 pi s (j+1/2) m (-1)^j = 2 pi s * d_{j+1}|_{mu = -m},

so  h = 2 pi s(m) * T^{Eisler}(mu = -/+ m)  exactly, and Eq. (72) gives

    2 pi s(m) = 4 kappa K(kappa')   =>   s(m) = (2/pi) kappa K(kappa'),
    kappa = 1/sqrt(1 + m^2) = sech(1/xi)   (xi = 1/asinh(m), so cosh(1/xi) = sqrt(1 + m^2)),
    |eps_l| = (2l + 1) * pi K(kappa')/K(kappa),   l = 0, 1, 2, ...

with the per-edge SIGNED pattern from the shifted quantization: the +m edge (mu = -m,
levels 2l + 1/2) carries +eps, -3 eps, +5 eps, -7 eps, ...; the -m edge (mu = +m,
levels 2l - 1/2) carries -eps, +3 eps, -5 eps, +7 eps, ... -- particle-hole
asymmetric per edge, symmetric (and doubly degenerate in |eps|) for a segment with
two far-apart edges. ctm_slope, ctm_level_spacing and eisler_2025_* below implement
these; the two families are algebraically identical and are cross-checked in
tests/test_modular_lab.py.

WHAT IS REPRODUCED HERE (measured, tests/test_modular_lab.py)
--------------------------------------------------------------
The substantive check is LATTICE vs Eq. (72) (test_anchor7_lattice_edge_matches_eisler_eq72);
the agreement of ctm_slope/ctm_level_spacing with a transcription of Eq. (72) is the same
formula on both sides and is kept only as a transcription check
(test_eq72_transcription_check_closed_form_vs_ctm_py). Lattice vs Eq. (72), on segments with
L >= 23 xi: edge slope from both diagonals 2.2e-16, 2.2e-16 (m = 1, L = 40) and
0.0e+00, 0.0e+00 (m = 1/2, L = 80) against 4 kappa K(kappa'), asserted at 1e-12; the
signed ladder with edge attribution to 6.7e-16, asserted at 1e-8 (mutations caught: the
dimerized modulus e^{-1/xi} for sech(1/xi), -27 % / +41 %; a dropped kappa in the prefactor,
+41 %; 2 pi -> pi in the level unit, -50 %; a swapped +-1/2 shift, O(1)). In more detail, at
m = 2, 1, 1/2, 0.3:
  * the edge slope from the on-site term and from the nearest-neighbour bond, both
    against 4 kappa K(kappa')/(2 pi) of Eq. (72): agreement 0 ... 2.2e-16;
  * the lowest level against pi K(kappa')/K(kappa): 0 ... 2.2e-16; the ladder ratios
    3, 5, 7 to 1e-8;
  * the SIGNED ladder with edge attribution (eigenvector weight on the left half):
    +eps, -3eps, +5eps, -7eps on the +m edge and -eps, +3eps, -5eps, +7eps on the -m
    edge, exactly Eq. (72)'s 2l + 1/2 / 2l - 1/2 rule (m = 1, L = 40; m = 1/2, L = 80);
  * the whole near-edge block against lattice_boost (both diagonals) to 3.8e-09 and
    6.0e-12, with the window bounded by the far edge, not by nb/xi.
Nothing here is fitted: kappa(m), the prefactor and the ladder are Eisler's; the lattice
numbers are computed from the closed-form correlation matrix at 60-120 digits.

Where the closed form applies: it is the HALF-INFINITE chain's entanglement Hamiltonian,
i.e. within a few xi of one edge of a segment whose other edge is >~ 20 xi away; past the
middle of a finite segment the profile is the triangle Delta(x) = min(x, L - x) of
Eisler et al. 2020, Eqs. (42)-(44) (triangle_h below); near criticality (xi >~ L) the
edge slope crosses over to the critical 1 - 1/L of Eisler-Peschel 2017, Eq. (27). In the
scaling limit s(m) = 1 - 1/(4 xi^2) + O(xi^-4) -> 1 (Bisognano-Wichmann with v = 1), and
the bare lattice boost 2 pi sum_j x_j h_j has the spectrum (2l + 1) pi^2/(2 kappa K(kappa)),
which collapses to a continuum as kappa -> 1.
"""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np

__all__ = [
    "ctm_modulus",
    "ctm_level_spacing",
    "ctm_slope",
    "ctm_slope_expansion",
    "eisler_2025_prefactor",
    "eisler_2025_levels",
    "lattice_boost",
    "triangle_h",
]


def ctm_modulus(m):
    """(kappa, kappa') = (1/sqrt(1 + m^2), m/sqrt(1 + m^2)) = (sech(1/xi), tanh(1/xi)); Eisler 2025 Eq. (70)."""
    m = float(m)
    if m <= 0.0:
        raise ValueError("the closed forms need a gap, m > 0")
    r = math.sqrt(1.0 + m * m)
    return 1.0 / r, m / r


def _mp_modulus(m, dps):
    """kappa, kappa' in working precision (a float kappa rounded to 1e-16 costs 5e-14 in K(kappa) near kappa -> 1)."""
    m = float(m)
    if m <= 0.0:
        raise ValueError("the closed forms need a gap, m > 0")
    with mp.workdps(int(dps)):
        kap = 1 / mp.sqrt(1 + mp.mpf(m) ** 2)
        return kap, mp.sqrt(1 - kap ** 2)


def ctm_level_spacing(m, dps=30):
    """eps = pi K(kappa')/K(kappa): |eps_l| = (2l + 1) eps for the half chain (Eisler 2025 Eq. (72))."""
    with mp.workdps(int(dps)):
        k, kp = _mp_modulus(m, dps)
        return float(mp.pi * mp.ellipk(kp ** 2) / mp.ellipk(k ** 2))


def ctm_slope(m, dps=30):
    """s(m) = (2/pi) kappa K(kappa'): the edge slope in units of 2 pi, i.e. Eisler 2025 Eq. (72)'s 4 kappa K(kappa') / (2 pi)."""
    with mp.workdps(int(dps)):
        k, kp = _mp_modulus(m, dps)
        return float(2 / mp.pi * k * mp.ellipk(kp ** 2))


def ctm_slope_expansion(xi):
    """Scaling-limit expansion s = 1 - 1/(4 xi^2) + O(xi^-4)."""
    xi = float(xi)
    return 1.0 - 0.25 / (xi * xi)


def eisler_2025_prefactor(mu, dps=30):
    """Eisler 2025 Eq. (72): the prefactor 4 kappa K(kappa') in H = 4 kappa K(kappa') T, kappa = 1/sqrt(1 + mu^2).

    Written directly from the paper (modulus convention, parameter kappa^2 for mpmath) so that
    ctm_slope can be tested against the published form: 2 pi ctm_slope(mu) == this.
    """
    mu = abs(float(mu))
    with mp.workdps(int(dps)):
        kap = 1 / mp.sqrt(1 + mp.mpf(mu) ** 2)
        kapp = mp.sqrt(1 - kap ** 2)
        return float(4 * kap * mp.ellipk(kapp ** 2))


def eisler_2025_levels(mu, ells=range(-3, 4), dps=30):
    """Eisler 2025 Eq. (72): eps_l = 2 pi K(kappa')/K(kappa) (2l - 1/2) for mu > 0, (2l + 1/2) for mu < 0.

    The sign of `mu` is the sign of the potential on HIS first site n = 1, i.e. minus the
    potential on our edge site (see the dictionary in the module docstring): pass mu = -m
    for our +m edge and mu = +m for our -m edge. Returns the signed levels for l in `ells`.
    """
    mu = float(mu)
    if mu == 0.0:
        raise ValueError("the gapped closed form needs mu != 0")
    with mp.workdps(int(dps)):
        kap = 1 / mp.sqrt(1 + mp.mpf(mu) ** 2)
        kapp = mp.sqrt(1 - kap ** 2)
        unit = 2 * mp.pi * mp.ellipk(kapp ** 2) / mp.ellipk(kap ** 2)
        shift = mp.mpf(-1) / 2 if mu > 0 else mp.mpf(1) / 2
        return np.array([float(unit * (2 * l + shift)) for l in ells])


def lattice_boost(L, m, hopping=0.5, slope=None):
    """2 pi s(m) T on the first L sites: Eisler 2025 Eqs. (67), (72) in our labels.

    Site j (0-based, j = 0 at the edge) sits at x = j + 1/2, bond (j, j+1) at
    x = j + 1 (dictionary D1 of vacuum.modular.casini_huerta):
        h_{j,j+1} = -2 pi s (j + 1) hopping,   h_jj = 2 pi s (j + 1/2) m (-1)^j.
    `slope` defaults to ctm_slope(m); pass 1.0 for the bare lattice boost.
    """
    L = int(L)
    s = ctm_slope(m) if slope is None else float(slope)
    x_site = np.arange(L) + 0.5
    x_bond = np.arange(1, L, dtype=float)
    h = np.zeros((L, L))
    h[np.arange(L), np.arange(L)] = 2.0 * np.pi * s * x_site * float(m) * (-1.0) ** np.arange(L)
    h[np.arange(L - 1), np.arange(1, L)] = -2.0 * np.pi * s * x_bond * float(hopping)
    h[np.arange(1, L), np.arange(L - 1)] = h[np.arange(L - 1), np.arange(1, L)]
    return h


def triangle_h(L, m, hopping=0.5, slope=None):
    """The three-diagonal 'triangle' approximation of Eisler et al. 2020, Eqs. (42)-(44):

    the half-chain line from both edges, 2 pi s min(x, L - x) times the local terms.
    """
    L = int(L)
    s = ctm_slope(m) if slope is None else float(slope)
    x_site = np.arange(L) + 0.5
    x_bond = np.arange(1, L, dtype=float)
    w_site = np.minimum(x_site, L - x_site)
    w_bond = np.minimum(x_bond, L - x_bond)
    h = np.zeros((L, L))
    h[np.arange(L), np.arange(L)] = 2.0 * np.pi * s * w_site * float(m) * (-1.0) ** np.arange(L)
    h[np.arange(L - 1), np.arange(1, L)] = -2.0 * np.pi * s * w_bond * float(hopping)
    h[np.arange(1, L), np.arange(L - 1)] = h[np.arange(L - 1), np.arange(1, L)]
    return h
