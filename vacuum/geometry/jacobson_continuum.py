"""Continuum companion to :mod:`vacuum.geometry.jacobson`: the exact 1+1
massless/chiral statements behind the lattice numbers of
``papers/toy-jacobson`` (anchor 1, the continuum limit of the residual).

Two certified computations, both independent of the lattice code:

1. **The continuum first law with the exact CHM modular Hamiltonian.**
   For the interval A = (-R, R) in the vacuum of a 1+1 massless chiral
   field, Bisognano-Wichmann / Hislop-Longo / Casini-Huerta-Myers give
   K_A = 2 pi int_A beta(x) T_00(x) dx + const with beta(x) = (R^2 - x^2)/(2R),
   whose modular flow is the conformal Killing flow dx/dt = 2 pi beta(x).
   In the modular coordinate z(x) = ln((R + x)/(R - x)) (dz/dx = 1/beta) the
   flow is z -> z + 2 pi t, so the ONE-PARTICLE modular generator is the
   self-adjoint differential operator

       delta_beta = -2 pi i (beta(x) d/dx + beta'(x)/2)      on L^2(-R, R),

   with generalized eigenfunctions psi_s(x) = beta(x)^{-1/2} e^{i s z(x)} /
   sqrt(2 pi) and spectrum 2 pi s, s in R (``modular_coordinate``).  Vacuum
   thermality at modular temperature 1/(2 pi) fixes the symplectic
   eigenvalue nu(s) = coth(pi s)/2, hence the entropy weight
   g(nu) = ln((nu + 1/2)/(nu - 1/2)) = 2 pi s.  ``continuum_first_law_defect``
   takes a normalized Gaussian squeeze wave packet in the modular frame,
   computes dS from the modular spectrum (entropy route) and d<K> from the
   POSITION-space integral against the weight (the CHM route, an mpmath
   quadrature over x in which beta appears explicitly), and returns the
   defect dS - d<K>.  It is 0 to quadrature precision for the CHM weight and
   O(eps) for a weight scaled by 1 + eps -- the mutation test.

2. **The O(a^2) weight-placement error of the lattice bond split.**
   ``bond_split_weight_coefficient`` is the closed-form leading coefficient
   of rho_site - rho_bond, the difference between the site-centred
   (trapezoid) and bond-centred (midpoint) discretisations of
   int beta T_00 on the massless chain; see ``papers/toy-jacobson/draft/
   proof_continuum_limit.md`` Sec. 4 for the derivation.

References: Bisognano-Wichmann, J. Math. Phys. 17 (1976) 303; Hislop-Longo,
Commun. Math. Phys. 84 (1982) 71; Casini-Huerta-Myers, JHEP 05 (2011) 036,
Sec. 2; Casini-Huerta, J. Phys. A 42 (2009) 504007 (the 2d scalar zero mode);
Blanco-Casini-Hung-Myers, JHEP 08 (2013) 060 (the first law).
"""

from __future__ import annotations

import mpmath as mp
import numpy as np

__all__ = [
    "chm_beta",
    "modular_coordinate",
    "modular_coordinate_inverse",
    "modular_symplectic_eigenvalue",
    "entropy_weight",
    "modular_wavepacket",
    "continuum_first_law_defect",
    "conformal_killing_defect",
    "bond_split_weight_coefficient",
    "bond_split_discretization_prediction",
]


# ---------------------------------------------------------------------------
# the CHM weight and its modular coordinate
# ---------------------------------------------------------------------------


def chm_beta(x, R):
    """The CHM/BW conformal-Killing weight beta(x) = (R^2 - x^2)/(2R) of the
    interval (-R, R) (Casini-Huerta-Myers 2011 Sec. 2; d = 2).  Works with
    floats and with mpmath types."""
    return (R * R - x * x) / (2 * R)


def modular_coordinate(x, R):
    """z(x) = ln((R + x)/(R - x)), the coordinate in which the modular flow of
    the interval (-R, R) is the translation z -> z + 2 pi t; dz/dx = 1/beta."""
    return mp.log((R + x) / (R - x))


def modular_coordinate_inverse(z, R):
    """x(z) = R tanh(z/2), the inverse of :func:`modular_coordinate`."""
    return R * mp.tanh(z / 2)


def modular_symplectic_eigenvalue(s):
    """nu(s) = coth(pi s)/2, the symplectic eigenvalue of the interval's
    reduced vacuum at modular frequency s > 0 (thermality of the vacuum at
    modular temperature 1/(2 pi): mean occupation n = 1/(e^{2 pi s} - 1),
    nu = n + 1/2)."""
    return mp.coth(mp.pi * s) / 2


def entropy_weight(nu):
    """g(nu) = ln((nu + 1/2)/(nu - 1/2)), the weight with which a bosonic
    symplectic eigenvalue enters dS = (1/2) sum_k g(nu_k) (dV)_kk in the
    Williamson frame (``vacuum.geometry.jacobson._g_of_nu``).  For the
    interval's vacuum, g(nu(s)) = 2 pi s exactly -- the content of the first
    law paired with Bisognano-Wichmann.  nu - 1/2 decays like e^{-2 pi s}, so
    call this at a working precision that resolves it (see
    ``continuum_first_law_defect``); nu <= 1/2 is not a physical state."""
    half = mp.mpf(1) / 2
    if nu <= half:
        raise ValueError(f"nu must exceed 1/2 (got {nu}); raise mp.mp.dps")
    return mp.log((nu + half) / (nu - half))


def modular_wavepacket(z, z0, sigma, s0):
    """The normalized Gaussian modular wave packet and its z-derivative,

        Phi(z) = (pi sigma^2)^{-1/4} exp(-(z - z0)^2/(2 sigma^2)) e^{i s0 z},

    with int |Phi|^2 dz = 1 and modular-frequency content
    |phi(s)|^2 = (sigma/sqrt(pi)) exp(-sigma^2 (s - s0)^2) (a Gaussian of
    width 1/sigma about s0).  Returns (Phi, dPhi/dz)."""
    amp = (mp.pi * sigma ** 2) ** mp.mpf(-0.25)
    g = amp * mp.e ** (-((z - z0) ** 2) / (2 * sigma ** 2))
    phase = mp.e ** (1j * s0 * z)
    Phi = g * phase
    dPhi = Phi * (-(z - z0) / sigma ** 2 + 1j * s0)
    return Phi, dPhi


# ---------------------------------------------------------------------------
# 1. the continuum first law, certified
# ---------------------------------------------------------------------------


def continuum_first_law_defect(R=1.0, s0=4.0, sigma=2.0, z0=0.0, weight=None,
                               weight_prime=None, dps=40, quad_maxdegree=10):
    """dS - d<K> for a Gaussian squeeze variation of the interval (-R, R) in
    the 1+1 massless/chiral vacuum, with K the exact continuum modular
    Hamiltonian 2 pi int weight(x) T_00(x) dx.

    The variation populates the modular mode content |phi(s)|^2 of the
    normalized wave packet ``modular_wavepacket`` (z0, sigma, s0); at first
    order both sides are linear in that content, so the check is the
    equality of two one-particle expectation values:

      * entropy route   dS  = int_0^inf g(nu(s)) |phi(s)|^2 ds, with
        nu(s) = coth(pi s)/2 from the vacuum's modular spectrum and g the
        bosonic entropy weight -- NOT simplified to 2 pi s analytically,
        so the identity g(nu(s)) = 2 pi s is part of what is certified;
      * CHM route       d<K> = <chi| -2 pi i (w d/dx + w'/2) |chi>, a
        quadrature over x in (-R, R) of the position-space modular generator
        built from the weight w, on the packet chi(x) = Phi(z(x))/sqrt(beta(x))
        (normalized: int |chi|^2 dx = int |Phi|^2 dz = 1).

    ``weight`` is a callable w(x) (default: the CHM parabola ``chm_beta``).
    A wrong weight -- e.g. ``lambda x: 1.001 * chm_beta(x, R)`` -- moves the
    CHM route only, and the defect becomes O(1e-3): the mutation test.

    Everything is mpmath at ``dps`` digits; no lattice code is used.
    Returns a dict with dS, dK, defect, rel_defect and diagnostics
    (norm_s, norm_x, im_dK, neg_s_tail).
    """
    with mp.workdps(int(dps)):
        Rm, s0m, sig, z0m = mp.mpf(R), mp.mpf(s0), mp.mpf(sigma), mp.mpf(z0)
        w = (lambda x: chm_beta(x, Rm)) if weight is None else weight

        # --- entropy route: the modular spectrum of the vacuum -------------
        def wsq(s):  # |phi(s)|^2
            return (sig / mp.sqrt(mp.pi)) * mp.e ** (-(sig ** 2) * (s - s0m) ** 2)

        s_lo = max(mp.mpf(0), s0m - 12 / sig)
        s_hi = s0m + 12 / sig
        # nu(s) - 1/2 = e^{-2 pi s}/(1 - e^{-2 pi s}) underflows to 0 at working
        # precision once 2 pi s > dps ln 10, and entropy_weight would divide by
        # zero; the s-quadrature therefore carries the extra digits it needs.
        guard = int(3 * float(s_hi)) + 10
        with mp.workdps(int(dps) + guard):
            dS = +mp.quad(lambda s: entropy_weight(modular_symplectic_eigenvalue(s)) * wsq(s),
                          [s_lo, s0m, s_hi], maxdegree=quad_maxdegree)
            norm_s = +mp.quad(wsq, [s_lo, s0m, s_hi], maxdegree=quad_maxdegree)
        # the s < 0 tail is unphysical (nu < 1/2) and is excluded from dS; it is
        # the only term by which the two routes differ by construction.
        neg_tail = mp.quad(lambda s: 2 * mp.pi * s * wsq(s), [-s_hi, mp.mpf(0)],
                           maxdegree=quad_maxdegree) if s_lo == 0 else mp.mpf(0)

        # --- CHM route: the position-space integral against the weight -----
        def chi_and_dchi(x):
            z = modular_coordinate(x, Rm)
            b = chm_beta(x, Rm)
            bp = -x / Rm                      # beta'(x)
            Phi, dPhi = modular_wavepacket(z, z0m, sig, s0m)
            rb = mp.sqrt(b)
            chi = Phi / rb
            dchi = dPhi / (b * rb) - Phi * bp / (2 * b * rb)
            return chi, dchi

        wp = (lambda x: mp.diff(w, x)) if weight_prime is None else weight_prime

        def integrand(x):
            chi, dchi = chi_and_dchi(x)
            return (-2j * mp.pi) * mp.conj(chi) * (w(x) * dchi + wp(x) * chi / 2)

        # the interval is truncated at |z - z0| = 16 sigma, where |Phi|^2 is
        # e^{-256}: beta never reaches 0, and the truncation is far below the
        # quadrature precision (reported as 'z_cut').
        z_cut = 16 * sig
        pts = [modular_coordinate_inverse(z0m + k * sig, Rm)
               for k in (-16, -8, -3, 0, 3, 8, 16)]
        dK_c = mp.quad(integrand, pts, maxdegree=quad_maxdegree)
        norm_x = mp.quad(lambda x: abs(chi_and_dchi(x)[0]) ** 2, pts,
                         maxdegree=quad_maxdegree)
        dK = mp.re(dK_c)
        defect = dS - dK
        return {
            "dS": dS, "dK": dK, "defect": defect,
            "rel_defect": abs(defect) / abs(dS),
            "norm_s": norm_s, "norm_x": norm_x,
            "im_dK": abs(mp.im(dK_c)), "neg_s_tail": abs(neg_tail),
            "z_cut": z_cut,
            "R": Rm, "s0": s0m, "sigma": sig, "z0": z0m, "dps": int(dps),
        }


def conformal_killing_defect(R=1.0, weight=None, xi=None, x0=None, half_width=None, dps=40):
    """The shape test of the weight: 2 pi int_A w(x) dT_00(x) dx for the
    first-order energy density of a diffeomorphism x -> x + eps xi(x) with xi
    supported STRICTLY inside A.

    Such a diffeomorphism is implemented by a unitary of the algebra of A, so
    dS = 0 exactly and the first law forces d<K> = 0.  In a c = 1 chiral CFT
    the vacuum energy density responds only through the anomaly,
    dT_00 = -(c/24 pi) xi'''(x), so the requirement is

        int_A w(x) xi'''(x) dx = 0 for every such xi   <=>   w''' = 0,

    i.e. the weight must be a quadratic polynomial -- the conformal Killing
    condition.  This test is blind to the NORMALIZATION of w (it is linear in
    w) and is therefore complementary to ``continuum_first_law_defect``,
    which fixes the normalization but is evaluated on one packet.

    ``xi`` defaults to a C^infty bump of half-width ``half_width`` (default
    R/4) centred at ``x0`` (default R/4 -- deliberately OFF centre, so that
    the test is not blinded to odd w''' by the parity of a centred bump).
    Returns dict: defect (the integral, times 2 pi (-1/24 pi)), scale (the
    same integral with |xi'''|, for a relative measure), rel_defect.
    """
    with mp.workdps(int(dps)):
        Rm = mp.mpf(R)
        w = (lambda x: chm_beta(x, Rm)) if weight is None else weight
        h = Rm / 4 if half_width is None else mp.mpf(half_width)
        c0 = Rm / 4 if x0 is None else mp.mpf(x0)
        if xi is None:
            def xi(x, h=h, c0=c0):  # C^infty bump, support (c0 - h, c0 + h)
                t = (x - c0) / h
                if abs(t) >= 1:
                    return mp.mpf(0)
                return mp.e ** (-1 / (1 - t ** 2))
        def d3(x):
            return mp.diff(xi, x, 3)
        c_anom = -2 * mp.pi / (24 * mp.pi)
        pts = [c0 - h + h / 1000, c0 - h / 2, c0, c0 + h / 2, c0 + h - h / 1000]
        defect = c_anom * mp.quad(lambda x: w(x) * d3(x), pts)
        scale = abs(c_anom) * mp.quad(lambda x: abs(w(x) * d3(x)), pts)
        return {"defect": defect, "scale": scale,
                "rel_defect": abs(defect) / scale if scale != 0 else mp.mpf(0)}


# ---------------------------------------------------------------------------
# 2. the O(a^2) weight-placement error of the lattice bond split
# ---------------------------------------------------------------------------


def bond_split_weight_coefficient(kappa):
    """A(kappa), the continuum-limit coefficient of the leading discretisation
    error of the SITE-centred CHM ansatz relative to the BOND-centred one on
    the massless chain:

        rho_site - rho_bond = A(kappa)/L^2 (1 + O(1/L)),   kappa = k R = pi L/lambda,

    for a squeeze of the normal mode of wavenumber k = 2 pi/lambda and a
    centred ball of L sites (R = L/2, lattice spacing a = 1).  Derivation in
    ``papers/toy-jacobson/draft/proof_continuum_limit.md`` Sec. 4:

      * the two ansaetze differ only in where beta is sampled on a bond.  The
        site form gives a bond the trapezoid weight (beta_i + beta_j)/2, the
        bond form the midpoint weight beta(x_m); for the quadratic CHM beta
        the difference is EXACTLY a^2 beta''/8 = -1/(8R) on interior bonds,
        and beta_edge/2 ~ 1/2 on each of the two bonds straddling dB (whose
        midpoint sits on the boundary, where beta = 0);
      * a squeeze of the mode u(x) = sin(k x) changes the bond energy density
        by (eps omega/2) cos^2(k x_m) and the site kinetic density by
        -(eps omega/2) sin^2(k x_i), so with C(kappa) and D(kappa) below

            A(kappa) = -4 C(kappa)/D(kappa),
            C(kappa) = -(1 + sin(2 kappa)/(2 kappa))/8 + cos^2(kappa)/2,
            D(kappa) = -cos(2 kappa)/(2 kappa^2) + sin(2 kappa)/(4 kappa^3).

    A(kappa) is the coefficient of the DIFFERENCE rho_site - rho_bond, NOT of
    the site-centred floor: the bond-centred floor is not derived here (it is
    predicted to vanish at this order), so A accounts for the site coefficient
    only up to that underived remainder -- 3.8 % of it at L = 8 and 13.6 % at
    L = 32 at the dominant wavelength (N = 2400), and more at short
    wavelength.  A(0+) = -3/2 analytically; at the wavelength anchor 1's fit
    is dominated by, kappa = 0.199, A = -1.4249 (closed form) / -1.4469
    (exact identity) against the measured difference -1.4469 and the measured
    site coefficient -1.3936.
    NOTE: D(kappa) is a cancellation of terms of size 1/kappa^3, so evaluating
    the closed form below kappa ~ 1e-4 in float64 is noise-limited (the error
    is 2.6e-9 at kappa = 1e-4, 1.4e-6 at 1e-5, 5.9e-3 at 1e-7); use
    kappa = 1e-4 to read off the limit numerically.
    The O(1/L) remainder is NOT bounded here (see the draft, ASSUMPTION A4).
    It is measured: 3.6e-02 in the worst archived cell (L = 8, kappa = 0.785)
    and 8.4e-04 in the best (L = 32, kappa = 0.209) -- the archive's
    max/min_rel_err_continuum.  Do NOT quote max_rel_err_discrete = 8.3e-04
    ('0.1 %') for this step: that is a different approximation (the exact
    lattice sums kept, only the (1 - rho_bond) factor dropped).
    """
    k = np.asarray(kappa, dtype=float)
    C = -(1.0 + np.sin(2 * k) / (2 * k)) / 8.0 + np.cos(k) ** 2 / 2.0
    D = -np.cos(2 * k) / (2 * k ** 2) + np.sin(2 * k) / (4 * k ** 3)
    return -4.0 * C / D


def bond_split_discretization_prediction(L, wavelength, discrete=True, rho_bond=None):
    """The predicted rho_site - rho_bond for a centred ball of L sites and a
    squeeze of the lattice normal mode of wavelength ``wavelength`` (in
    lattice units), from :func:`bond_split_weight_coefficient`.

    ``discrete=True`` also evaluates the same expression with the exact
    lattice sums (beta at the L site centres and the L - 1 interior bond
    midpoints, the two straddling bonds, and the lattice dispersion
    omega = 2 sin(k/2)) rather than their L -> infinity limits: this keeps
    the O(1/L) terms that the closed form drops, and is the sharper
    prediction.  The denominator uses the bond-centred pairing d<K>_bond in
    place of dS, an error of relative size rho_bond ~ 5e-2/L^2.

    Given the measured bond-centred residual ``rho_bond``, the relation is an
    EXACT lattice identity (draft Sec. 4, Prop. 4.3):

        rho_site - rho_bond = -(1 - rho_bond) * (dK_site - dK_bond)/dK_bond,

    since dS = dK_bond/(1 - rho_bond) by the definition of rho_bond and both
    dK's are the closed-form finite sums below.  ``A_exact`` returns that,
    and it reproduces the lattice residual to 1e-9 (machine).

    Returns dict: kappa, A_continuum, rho_continuum, A_discrete,
    rho_discrete (the last two None if ``discrete`` is False), A_exact
    (None unless ``rho_bond`` is given).
    """
    L = int(L)
    lam = float(wavelength)
    R = L / 2.0
    k = 2.0 * np.pi / lam
    kappa = k * R
    A_c = float(bond_split_weight_coefficient(kappa))
    out = {"L": L, "wavelength": lam, "kappa": float(kappa),
           "A_continuum": A_c, "rho_continuum": A_c / L ** 2,
           "A_discrete": None, "rho_discrete": None, "A_exact": None}
    if not discrete:
        return out
    x_site = np.arange(L) + 0.5 - R                 # site centres
    x_mid = np.arange(1, L) - R                     # interior bond midpoints
    beta = lambda x: (R * R - x * x) / (2.0 * R)    # noqa: E731
    b_site, b_mid = beta(x_site), beta(x_mid)
    b_edge = b_site[0]                              # = b_site[-1]
    # numerator: sum_bonds (w_site - w_bond) * cos^2(k x_m)
    dK_delta = (np.sum((-1.0 / (8.0 * R)) * np.cos(k * x_mid) ** 2)
                + 2.0 * (b_edge / 2.0) * np.cos(k * R) ** 2)
    # denominator: the bond-centred pairing, same common factor (eps omega/2)
    dK_bond = (-np.sum(b_site * np.sin(k * x_site) ** 2)
               + np.sum(b_mid * np.cos(k * x_mid) ** 2))
    rho_d = -dK_delta / dK_bond
    out["A_discrete"] = float(rho_d * L ** 2)
    out["rho_discrete"] = float(rho_d)
    if rho_bond is not None:
        out["A_exact"] = float(rho_d * L ** 2 * (1.0 - float(rho_bond)))
    return out
