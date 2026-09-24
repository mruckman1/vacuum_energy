"""Circuit -> lattice coupling map for the Waterloo flux-qubit / waveguide proposal.

This module DERIVES the oscillator-detector coupling ``lam_osc`` that the
circuit-QED noise-threshold candidate (papers/circuit-qed-noise-thresholds)
feeds to ``attach_detectors`` from the published circuit parameters of

    [TB26] A. Teixido-Bonfill, X. Dai, A. Lupascu, E. Martin-Martinez,
           Phys. Rev. A 113, 043732 (2026); arXiv:2505.01516v1 (all equation
           numbers below are the arXiv v1 numbering, read from the PDF text on
           2026-09-03),
    [J23]  N. Janzen, X. Dai, S. Ren, J. Shi, A. Lupascu, Phys. Rev. Research
           5, 033155 (2023); arXiv:2208.05571v2 (the measured device),

replacing the identification ``lam_osc = |lambda_TB| sqrt(2 Omega)`` that the
candidate's MANIFEST recorded as "a convention, not a derivation".  Every step
is written out, every unit conversion is named, and where the paper leaves a
choice open the alternatives are enumerated with their numerical consequence
(:func:`lambda_osc_from_circuit` returns all of them).  Conventions on the
lattice side are docs/API.md (hbar = 1, V_vac = I/2, nats) and the loader
:mod:`vacuum.experiments_io` (energies / omega_ref, lengths in light-crossing
units v / omega_ref).


1. The paper's model in physical units
======================================

**Field.**  TB26 Eq. (3): the transmission line is
H_tl = (1/2) int dx [ (d_x Phi)^2 / l_0 + q^2 / c_0 ], with the flux field
Phi(x) and charge density q(x) = c_0 d_t Phi (Eq. 4) canonically conjugate,
[Phi(x), q(y)] = i hbar delta(x - y).  Its mode expansion, Eq. (5):

    Phi(t, x) = sqrt(hbar Z_0) int dk (4 pi |k|)^(-1/2)
                              ( e^{i(omega_k t - k x)} a_k^dag + H.c. ),
    [a_k, a_k'^dag] = delta(k - k'),  omega_k = v |k|,
    v = 1 / sqrt(c_0 l_0),  Z_0 = sqrt(l_0 / c_0);  Eq. (7): v ~ 1.2e8 m/s,
    Z_0 ~ 50 Ohm.

Substituting the expansion into [Phi, q] gives N_k^2 = hbar Z_0 / (4 pi |k|):
Eq. (5) is nothing but the canonical normalization of a 1+1 massless field,
with sqrt(hbar Z_0) carrying every dimension.  The cutoff field Phi_C
(Eqs. 8-9) multiplies the mode weight by C(omega) = e^{-|omega| / 2 Omega_cut},
Omega_cut / 2 pi = 50 GHz.

**Interaction.**  From the lumped line + TC+FQ circuit (Fig. 4, Eq. 16),
Delta x -> 0 (Eqs. 17-19): H_int = -(1/l_0) Phi_- d_x Phi(x_d) with the flux
jump across the coupler Phi_- = phi_0 gamma_5, phi_0 = hbar / 2e (Sec. II.3);
with the cutoff field, Eq. (22): H_int = -(phi_0 / l_0) gamma_5 d_x Phi_C(x_d).
Because the line current is I(x) = -d_x Phi / l_0, this is H_int =
phi_0 gamma_5 I(x_d): the coupler phase couples to the line CURRENT (an
inductive coupling; J23 Eq. (B9) is the same term).  The renormalization
term Phi_-^2 / (2 l_0 Delta x) is dropped by the paper (Sec. II.3, footnote 3)
and therefore by this map.  Two-level truncation, adiabatic free evolution and
transverse coupling (Sec. III.1, Eqs. 27-28: gamma_5 -> gamma_x sigma_x,
gamma_x(t) = gamma chi(t)) give the variable-gap spatial-derivative (VGSD)
detector, Eq. (31),

    H_int(t) = -(phi_0 / l_0) gamma chi(t) mu(t) d_x Phi_C(t, x_d),
    mu(t) = e^{i phi(t)} sigma^+ + H.c.,
    Omega(t) = Omega^0 + Delta-Omega chi(t)          (Eqs. 29, 32),
    Omega^0 / 2 pi ~ 7.3 GHz,  Delta-Omega / 2 pi ~ -23 gamma GHz   (Eq. 30).

gamma is the matrix element Re <1|gamma_5|0> of the coupler phase between the
qubit levels (Eq. 28), dimensionless, gamma in [-0.02, 0.22] (Sec. III.1.3).


2. The paper's dimensionless coupling lambda_TB
==============================================

Sec. IV writes the same physics with a dimensionless field, Eq. (36):

    phi_C(t, x) = int dk C(omega_k) (4 pi |k|)^(-1/2)
                          ( e^{i(omega_k t - k x)} a_k^dag + H.c. ) = Phi_C / sqrt(hbar Z_0),

and Eq. (37): H_I(t) = hbar c sum_nu lambda_nu chi_nu(t) mu_nu(t) d_x phi_C(t, x_nu).
Matching Eq. (31) to Eq. (37) with l_0 = Z_0 / v (Sec. VI.A) is Eq. (66):

    lambda = -(phi_0 / (hbar v l_0)) gamma sqrt(hbar Z_0)
           = -gamma sqrt(hbar / (4 e^2 Z_0)) = -gamma sqrt(R_K / (8 pi Z_0))
           ~ -4.53 gamma,          R_K = h / e^2 = 25812.807 Ohm (SI-2019 exact),

and lambda^2 = pi alpha with alpha = R_K gamma^2 / (8 pi^2 Z_0) ~ 6.54 gamma^2,
Eq. (33) -- the spin-boson coupling of the device paper.  Cross-check that
fixes the NORMALIZATION of lambda physically, independently of any field
convention: J23 Eq. (10), Gamma_1 = (phi_0^2 / hbar Z_0) |gamma_5,10|^2 Delta,
equals lambda^2 Delta, and J23 Eq. (6), Gamma_1 = pi alpha Delta, is the same
number -- both are the Golden-rule rate of Eq. (37) on the massless line,
Gamma = lambda^2 Omega (step 5 below).  So lambda_TB is *the* dimensionless
coupling in units where the field-induced relaxation rate at the gap is
lambda^2 Omega.  Physical scale: (phi_0 / l_0) gamma sqrt(hbar Z_0) (Omega^0 / v)
= hbar Omega^0 |lambda_TB|, i.e. |lambda_TB| Omega^0 / 2 pi = 0.66 GHz at
gamma = 0.02 -- the coupling energy is 9 % of the gap
(:func:`coupling_energy_SI`).


3. Lattice units: which conversion factors survive
==================================================

The loader fixes hbar = 1, energies in units of omega_ref = 2 pi x 7.3 GHz
= Omega^0 (the free gap), and lengths in v / omega_ref (one site = one unit
of light-crossing time).  With xi = x omega_ref / v,

    H_I / (hbar omega_ref) = lambda chi mu (v / omega_ref) d_x phi = lambda chi mu d_xi phi,

for ANY omega_ref: the derivative coupling is scale-free.  The field is
invariant too: with k = kappa omega_ref / v and a_k = a_kappa sqrt(v / omega_ref)
(to keep the delta normalization),

    dk (4 pi |k|)^(-1/2) a_k = dkappa (4 pi |kappa|)^(-1/2) a_kappa,

the canonical massless field of 1+1 dimensions is dimensionless and
scale-invariant.  Hence

  (i)   no Z_0 or v factor survives beyond Eq. (66): the paper already absorbed
        Z_0 through sqrt(hbar Z_0) in Eq. (5) and v through l_0 = Z_0 / v;
  (ii)  no lattice-spacing factor enters the field AMPLITUDE;
  (iii) the lattice spacing enters only through the derivative,
        d_x = (omega_ref / v) d_xi, which the choice of units turns into
        d_xi with coefficient 1; what does depend on omega_ref is the gap,
        Omega_lat = Omega^0 / omega_ref (= 1 here), and it is through Omega_lat
        that the 'xx' mapping of step 6 remembers the spacing.

**The lattice field is the same field.**  ``harmonic_chain_K``:
H = (1/2) sum_j [pi_j^2 + m^2 phi_j^2 + (phi_{j+1} - phi_j)^2] is the unit-spacing
discretization of (1/2) int dxi [Pi^2 + (d_xi phi)^2 + m^2 phi^2] with
phi_j = phi(xi_j); on the infinite lattice

    phi_j = int_{-pi}^{pi} dkappa (4 pi omega_kappa)^(-1/2)
                   ( b_kappa e^{i kappa j - i omega_kappa t} + H.c. ),
    [b_kappa, b_kappa'^dag] = delta(kappa - kappa'),
    omega_kappa^2 = m^2 + 4 sin^2(kappa / 2)  ->  |kappa|  (m -> 0, kappa -> 0),

which is Eq. (36) with C = 1.  The paper's exponential cutoff at
Omega_cut = 6.85 omega_ref is replaced by the lattice's own band edge
omega_max = 2 (the candidate's model limitation 1); the mass m is the
candidate's synthetic IR regulator (limitation 2).  The normalization is
checked numerically in tests/test_circuit_mapping.py: the ground-state
<phi_j^2> of a periodic chain equals int domega S_phi^lat(omega) / 2 pi with
the spectral density of step 5.


4. Two-level -> oscillator: MAP-1
=================================

tests/test_gate_crossvalidation.py (Gate A, step 1): the oscillator monopole
x_d(t) = (a e^{-i Omega t} + a^dag e^{+i Omega t}) / sqrt(2 Omega) is
mu(t) / sqrt(2 Omega) under a <-> sigma^-, for whatever field operator it
multiplies, so

    lambda_osc = lambda_UDW sqrt(2 Omega_lat)                      (MAP-1).

The sign of lambda is immaterial for the oscillator model (x_d -> -x_d).

**Derivative transcription ('xp').**  ``attach_detectors(coupling='xp')``
couples x_d to pi_j = d_t phi_j (the d_t phi model of Teixido-Bonfill &
Martin-Martinez, PRD 110, 105016 (2024), Eq. (2)).  For the massless
continuum field d_t d_t' W = d_x d_x' W = int dk |k| (4 pi)^(-1) e^{...}, so
every second-order matrix element (L, M and the +/- split) of the d_t and
d_x models coincide; the paper's Eq. (37) transcribes EXACTLY to

    lambda_osc^xp = |lambda_TB| sqrt(2 Omega_lat)      (no further factor),

with the sole lattice caveat that pi_j carries the mode weight omega_kappa^2
while the finite difference phi_{j+1} - phi_j carries omega_kappa^2 - m^2:
a (1 - m^2 / Omega^2)^(1/2) = 0.980 amplitude difference at the gap.


5. Spectral densities (Golden rule)
===================================

For H_int = lambda_osc x_d O(t) the field-induced Golden-rule rate of the
detector at gap Omega is Gamma = lambda_osc^2 |<1|x_d|0>|^2 S_O(Omega)
= lambda_osc^2 S_O(Omega) / (2 Omega), S_O(omega) = int dt e^{i omega t}
<O(t) O(0)>.  With the expansions above (two roots +/- kappa, two branches):

    continuum massless:  S_phi(omega) = 1 / omega,   S_{d phi}(omega) = omega,
    infinite lattice:    S_phi^lat(omega) = 1 / |sin kappa(omega)|,
                         4 sin^2(kappa / 2) = omega^2 - m^2   (d omega / d kappa = sin kappa / omega),
                         S_pi = omega^2 S_phi^lat,   S_{Delta phi} = (omega^2 - m^2) S_phi^lat.

Two-level, derivative coupling, continuum: Gamma = lambda_TB^2 Omega = pi alpha
Omega -- J23 Eqs. (6), (10), the statement of step 2.  At Omega = 1, m = 0.2:
sin kappa = 0.854, S_phi^lat = 1.171 (the lattice's density of states at half
the band exceeds the continuum's by 17 %).


6. The amplitude ('xx') model of the noise map
==============================================

The noise map uses ``coupling='xx'``, H_int = lambda_osc x_d phi_j.  No
transcription exists: phi_j and d_xi phi are different operators
(S ~ 1/omega vs ~ omega).  The map is therefore an EQUIVALENCE at the operating
point under a stated criterion: the 'xx' detector at the free gap Omega^0 has
the same field-induced Golden-rule rate as a reference.  Three references,
three formulas (``criterion`` argument), all of the form
lambda_osc = |lambda_TB| M(Omega_lat, m):

  'xp' (DEFAULT)  -- the reference is the transcribed derivative model ON THE
        SAME LATTICE: lambda_osc^2 S_phi^lat / 2 Omega = (lambda_TB sqrt(2 Omega))^2
        S_pi / 2 Omega  =>  M = Omega_lat sqrt(2 Omega_lat).  Mass- and
        lattice-independent, and identical to matching the continuum-massless
        rates lambda_osc^2 / 2 Omega^2 = lambda_TB^2 Omega.  This is the
        default because it isolates the operator change (amplitude vs
        derivative) from the lattice's density-of-states excess, which the
        'xp' transcription shares and which is already declared as model
        limitation 1.  AT Omega_lat = 1 THIS IS sqrt(2) |lambda_TB| = the number
        the candidate had assumed: the identification lambda_UDW == lambda_TB
        omitted the factor Omega_lat = Omega^0 / omega_ref, which is unity
        because the loader's energy unit is the free gap.
  'finite-difference' -- the paper's d_x taken literally on the lattice as
        phi_{j+1} - phi_j:  M = sqrt(2 Omega (Omega^2 - m^2))  = 0.980 x 'xp'
        at Omega = 1, m = 0.2.
  'physical-rate' -- the lattice 'xx' detector at the DEVICE's physical rate
        lambda_TB^2 Omega:  M = Omega sqrt(2 |sin kappa(Omega)|) = 0.924 x 'xp'.

Where the paper leaves the matching frequency open: at the fully switched-on
gap Omega_lat = 1 - 0.46/7.3 = 0.937 instead of the free gap, 'xp' gives
0.907x and 'physical-rate' 0.845x the default.  Tracking the gap in time
(lambda_osc(t) ~ Omega(t)^{3/2}) is not done: the paper's lambda is constant
(chi(t) carries the switching), and the map applies lambda_osc chi(t) with
Omega_d(t) from Eq. (29).

What the matching does NOT do: make the amplitude model reproduce the
derivative model's harvesting.  Across the switching band (~ 2 pi / T_lat =
0.62 around Omega = 1) the two spectral shapes differ; at the static-gap
operating point the perturbative companions at the same lambda_UDW = 0.1 give
E_N = 7.25e-4 ('xx') vs 2.07e-4 ('xp'), P_A = 2.37e-5 vs 7.0e-6, and a
communication fraction |M_comm|/|M| of 0.29 vs 0.98.  That factor ~3.5 is a
MODEL difference (the candidate's limitations 3-4), recorded in the
device-point archive, not a coupling uncertainty.


7. Uncertainty
==============

No input of the map carries an ``error`` field in the parameter files
(gamma, Z_0, v, Omega^0 are all quoted with "~" and no error bar; only the
DiCarlo cross-check file has errors).  What is documented is the paper's own
rounding: scenario 2 of Table I is defined by lambda = -0.1 (Table I; the
captions of Figs. 8 and 18 state that the scenario is simulated at
lambda = -0.1), while its gamma = 0.02 gives -0.0906 through Eq. (66) and its
alpha = 0.003 gives -0.0971 through lambda^2 = pi alpha.  The map takes the
simulated value -0.1 and the SAMPLE standard deviation of the three readings,
0.0048 (4.8 %), as the 1-sigma spread; ``lambda_osc_from_circuit`` propagates
it linearly (sigma_osc = M sigma_TB) and would add, in quadrature, any
``error`` a file carries on gamma (d lambda / d gamma = -4.53) or Z_0
(d lambda / d Z_0 = -lambda / 2 Z_0).  Result at the candidate's operating
point: lambda_osc = 0.1414 +/- 0.0068 (default criterion), with the
alternatives above spanning 0.118-0.148.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from vacuum.experiments_io import EV_SI, H_PLANCK_SI, HBAR_SI

__all__ = [
    "R_K_SI",
    "PHI_0_REDUCED_SI",
    "CRITERIA",
    "DEFAULT_CRITERION",
    "lambda_tb_from_gamma",
    "gamma_from_lambda_tb",
    "alpha_from_gamma",
    "alpha_from_lambda_tb",
    "lambda_tb_from_alpha",
    "coupling_energy_SI",
    "lattice_wavenumber",
    "amplitude_spectral_density",
    "derivative_spectral_density",
    "golden_rule_rate",
    "mapping_factor",
    "lambda_udw_from_osc",
    "lambda_osc_from_udw",
    "lambda_xp_osc_from_tb",
    "lambda_osc_from_tb",
    "lambda_tb_from_osc",
    "ScenarioReadings",
    "scenario_lambda_readings",
    "MappingResult",
    "lambda_osc_from_circuit",
    "circuit_from_lambda_osc",
]

# --- constants (SI-2019 exact) ------------------------------------------------

#: von Klitzing constant R_K = h / e^2 (Ohm), exact in the SI since 2019.
R_K_SI = H_PLANCK_SI / EV_SI**2
#: reduced magnetic flux quantum phi_0 = hbar / 2e (Wb), TB26 Sec. II.3.
PHI_0_REDUCED_SI = HBAR_SI / (2.0 * EV_SI)

#: matching criteria of step 6, in the order the docstring lists them
CRITERIA: Tuple[str, ...] = ("xp", "finite-difference", "physical-rate")
DEFAULT_CRITERION = "xp"


# --- step 2: the paper's dimensionless coupling --------------------------------


def lambda_tb_from_gamma(gamma: float, Z0_ohm: float = 50.0) -> float:
    """TB26 Eq. (66): lambda = -gamma sqrt(R_K / (8 pi Z_0)) (signed, ~ -4.53 gamma at 50 Ohm)."""
    if not (Z0_ohm > 0.0):
        raise ValueError(f"Z0 must be positive, got {Z0_ohm}")
    return -float(gamma) * math.sqrt(R_K_SI / (8.0 * math.pi * float(Z0_ohm)))


def gamma_from_lambda_tb(lambda_tb: float, Z0_ohm: float = 50.0) -> float:
    """Inverse of :func:`lambda_tb_from_gamma`."""
    if not (Z0_ohm > 0.0):
        raise ValueError(f"Z0 must be positive, got {Z0_ohm}")
    return -float(lambda_tb) / math.sqrt(R_K_SI / (8.0 * math.pi * float(Z0_ohm)))


def alpha_from_gamma(gamma: float, Z0_ohm: float = 50.0) -> float:
    """TB26 Eq. (33): alpha = R_K gamma^2 / (8 pi^2 Z_0) (~ 6.54 gamma^2 at 50 Ohm)."""
    if not (Z0_ohm > 0.0):
        raise ValueError(f"Z0 must be positive, got {Z0_ohm}")
    return R_K_SI * float(gamma) ** 2 / (8.0 * math.pi**2 * float(Z0_ohm))


def alpha_from_lambda_tb(lambda_tb: float) -> float:
    """TB26 Sec. VI.A below Eq. (66): lambda^2 = pi alpha."""
    return float(lambda_tb) ** 2 / math.pi


def lambda_tb_from_alpha(alpha: float) -> float:
    """|lambda| = sqrt(pi alpha) (the magnitude; the sign is the sign of gamma)."""
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    return math.sqrt(math.pi * float(alpha))


def coupling_energy_SI(gamma: float, Z0_ohm: float, v_m_s: float, omega_rad_s: float) -> float:
    """The interaction energy scale of TB26 Eq. (31) at frequency omega, in joules.

    (phi_0 / l_0) gamma sqrt(hbar Z_0) (omega / v) with l_0 = Z_0 / v, i.e. the
    prefactor of Eq. (31) times the mode amplitude sqrt(hbar Z_0) times the
    wavenumber omega / v of the derivative.  Step 2 of the module docstring
    states, and the tests check, that this equals hbar omega |lambda_TB| for
    every omega, v and Z_0 -- the dimensional-analysis identity behind Eq. (66).
    """
    if not (Z0_ohm > 0.0 and v_m_s > 0.0 and omega_rad_s > 0.0):
        raise ValueError("Z0, v and omega must be positive")
    l0 = float(Z0_ohm) / float(v_m_s)
    return (
        (PHI_0_REDUCED_SI / l0)
        * abs(float(gamma))
        * math.sqrt(HBAR_SI * float(Z0_ohm))
        * (float(omega_rad_s) / float(v_m_s))
    )


# --- step 5: spectral densities -------------------------------------------------


def lattice_wavenumber(omega: float, field_mass: float) -> float:
    """kappa in (0, pi] with omega^2 = m^2 + 4 sin^2(kappa / 2) (infinite chain)."""
    m = float(field_mass)
    w = float(omega)
    s2 = (w * w - m * m) / 4.0
    if not (0.0 <= s2 <= 1.0):
        raise ValueError(
            f"omega = {w} is outside the lattice band [m, sqrt(m^2 + 4)] = "
            f"[{m}, {math.sqrt(m * m + 4.0)}]"
        )
    return 2.0 * math.asin(math.sqrt(s2))


def amplitude_spectral_density(omega: float, field_mass: Optional[float] = None) -> float:
    """S_phi(omega) = int dt e^{i omega t} <phi(t) phi(0)> of the canonical 1+1 field.

    ``field_mass=None``: continuum massless, 1 / omega.  Otherwise the
    infinite unit-spacing lattice of ``harmonic_chain_K``, 1 / |sin kappa(omega)|
    (two roots +/- kappa, |d omega / d kappa| = |sin kappa| / omega).
    """
    w = float(omega)
    if not (w > 0.0):
        raise ValueError(f"omega must be positive, got {w}")
    if field_mass is None:
        return 1.0 / w
    kappa = lattice_wavenumber(w, field_mass)
    s = math.sin(kappa)
    if s <= 0.0:  # the band edges: |sin kappa| -> 0, S diverges (van Hove)
        raise ValueError(f"omega = {w} is at a band edge of the lattice (sin kappa = 0)")
    return 1.0 / s


def derivative_spectral_density(
    omega: float, field_mass: Optional[float] = None, operator: str = "xp"
) -> float:
    """Spectral density of the derivative operator the detector couples to.

    operator 'xp': pi_j = d_t phi_j, weight omega^2 (continuum: omega);
    'finite-difference': phi_{j+1} - phi_j, weight omega^2 - m^2
    (continuum: omega, the same as 'xp' -- d_t d_t' W = d_x d_x' W for the
    massless field).
    """
    w = float(omega)
    if operator not in ("xp", "finite-difference"):
        raise ValueError(f"operator must be 'xp' or 'finite-difference', got {operator!r}")
    if field_mass is None:
        return w  # omega^2 * (1 / omega)
    S = amplitude_spectral_density(w, field_mass)
    if operator == "xp":
        return w * w * S
    return (w * w - float(field_mass) ** 2) * S


def golden_rule_rate(lam: float, gap: float, S_at_gap: float, two_level: bool = False) -> float:
    """Gamma = lam^2 |<1|x_d|0>|^2 S(Omega): oscillator |<1|x_d|0>|^2 = 1 / 2 Omega; qubit 1."""
    if not (gap > 0.0):
        raise ValueError(f"gap must be positive, got {gap}")
    me2 = 1.0 if two_level else 1.0 / (2.0 * float(gap))
    return float(lam) ** 2 * me2 * float(S_at_gap)


# --- steps 4 and 6: the maps -----------------------------------------------------


def lambda_udw_from_osc(lambda_osc: float, gap_lat: float) -> float:
    """MAP-1: lambda_UDW = lambda_osc / sqrt(2 Omega)."""
    if not (gap_lat > 0.0):
        raise ValueError(f"gap must be positive, got {gap_lat}")
    return float(lambda_osc) / math.sqrt(2.0 * float(gap_lat))


def lambda_osc_from_udw(lambda_udw: float, gap_lat: float) -> float:
    """MAP-1 inverse: lambda_osc = lambda_UDW sqrt(2 Omega)."""
    if not (gap_lat > 0.0):
        raise ValueError(f"gap must be positive, got {gap_lat}")
    return float(lambda_udw) * math.sqrt(2.0 * float(gap_lat))


def lambda_xp_osc_from_tb(lambda_tb: float, gap_lat: float) -> float:
    """Step 4: the exact transcription of TB26 Eq. (37) to ``coupling='xp'``:
    lambda_osc^xp = |lambda_TB| sqrt(2 Omega_lat) (MAP-1 on the same operator)."""
    return lambda_osc_from_udw(abs(float(lambda_tb)), gap_lat)


def mapping_factor(
    gap_lat: float, criterion: str = DEFAULT_CRITERION, field_mass: Optional[float] = None
) -> float:
    """M(Omega_lat, m) with lambda_osc('xx') = |lambda_TB| M, step 6.

    'xp':                M = Omega sqrt(2 Omega)            (mass-independent)
    'finite-difference': M = sqrt(2 Omega (Omega^2 - m^2))
    'physical-rate':     M = Omega sqrt(2 |sin kappa(Omega)|)

    The lattice criteria need ``field_mass``; 'xp' ignores it.  Each formula is
    exactly lambda_osc^2 S_phi(Omega) / 2 Omega = (reference rate), see
    :func:`golden_rule_rate`; the tests assert the identity for each.
    """
    W = float(gap_lat)
    if not (W > 0.0):
        raise ValueError(f"gap must be positive, got {W}")
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
    if criterion == "xp":
        return W * math.sqrt(2.0 * W)
    if field_mass is None:
        raise ValueError(f"criterion {criterion!r} needs the lattice field_mass")
    m = float(field_mass)
    if criterion == "finite-difference":
        if W * W <= m * m:
            raise ValueError(f"gap {W} is below the lattice mass gap {m}")
        return math.sqrt(2.0 * W * (W * W - m * m))
    # 'physical-rate'
    kappa = lattice_wavenumber(W, m)
    return W * math.sqrt(2.0 * abs(math.sin(kappa)))


def lambda_osc_from_tb(
    lambda_tb: float,
    gap_lat: float,
    criterion: str = DEFAULT_CRITERION,
    field_mass: Optional[float] = None,
) -> float:
    """lambda_osc for ``coupling='xx'`` from the paper's lambda (step 6)."""
    return abs(float(lambda_tb)) * mapping_factor(gap_lat, criterion, field_mass)


def lambda_tb_from_osc(
    lambda_osc: float,
    gap_lat: float,
    criterion: str = DEFAULT_CRITERION,
    field_mass: Optional[float] = None,
) -> float:
    """Inverse of :func:`lambda_osc_from_tb` (returns the magnitude |lambda_TB|)."""
    return abs(float(lambda_osc)) / mapping_factor(gap_lat, criterion, field_mass)


# --- step 7: the parameter file --------------------------------------------------


def _qfield(quantities: Mapping[str, Any], name: str, field_name: str):
    """A ``Quantity`` attribute or a plain-dict key (fixtures), else None."""
    q = quantities[name]
    if isinstance(q, Mapping):
        return q.get(field_name)
    return getattr(q, field_name, None)


def _qval(quantities: Mapping[str, Any], name: str, index: Optional[int] = None):
    v = _qfield(quantities, name, "value")
    if v is None:
        raise ValueError(f"quantity {name!r} has no value")
    if index is not None:
        return float(v[index])
    return float(v)


def _qerr(quantities: Mapping[str, Any], name: str, index: Optional[int] = None) -> float:
    e = _qfield(quantities, name, "error")
    if e is None:
        return 0.0
    if index is not None:
        return float(e[index])
    return float(e)


@dataclass(frozen=True)
class ScenarioReadings:
    """The published readings of one Table I scenario's coupling (step 7)."""

    scenario: int  # 1-based
    table1: float  # Table I column lambda (signed) -- the simulated value
    eq66: float  # lambda_per_gamma * gamma (signed)
    alpha: float  # -sqrt(pi alpha) with the sign of table1
    gamma: float
    alpha_value: float
    Z0_ohm: float
    sigma_gamma: float  # file 'error' on gamma (0 if absent)
    sigma_Z0: float  # file 'error' on Z0 (0 if absent)

    @property
    def readings(self) -> Tuple[float, float, float]:
        return (self.table1, self.eq66, self.alpha)

    @property
    def mean(self) -> float:
        return float(np.mean(self.readings))

    @property
    def sample_std(self) -> float:
        return float(np.std(self.readings, ddof=1))

    @property
    def sigma_propagated(self) -> float:
        """Errors a file carries on gamma / Z_0, propagated through Eq. (66)."""
        coef = math.sqrt(R_K_SI / (8.0 * math.pi * self.Z0_ohm))
        d_gamma = coef * self.sigma_gamma
        d_Z0 = abs(self.eq66) / (2.0 * self.Z0_ohm) * self.sigma_Z0
        return math.hypot(d_gamma, d_Z0)

    @property
    def sigma(self) -> float:
        """Reading spread and propagated file errors, in quadrature."""
        return math.hypot(self.sample_std, self.sigma_propagated)


def scenario_lambda_readings(experiment: Any, scenario_index: int = 1) -> ScenarioReadings:
    """Read scenario ``scenario_index`` (0-based) of TB26 Table I three ways.

    ``experiment`` is the loaded ``teixido-bonfill_2026_table1`` file
    (:func:`vacuum.experiments_io.load_experiment`) or any mapping with the
    same quantity names.  Uses the transcribed ``characteristic_impedance``
    for the Eq. (66) coefficient rather than the printed 4.53, so the reading
    reacts to a corrected Z_0 (the printed coefficient is checked in tests).
    """
    q = experiment.quantities if hasattr(experiment, "quantities") else experiment
    i = int(scenario_index)
    lam1 = _qval(q, "table1_scenario_lambda", i)
    gam = _qval(q, "table1_scenario_gamma", i)
    alp = _qval(q, "table1_scenario_alpha", i)
    Z0 = _qval(q, "characteristic_impedance")
    lam66 = lambda_tb_from_gamma(gam, Z0)
    lam_a = math.copysign(lambda_tb_from_alpha(alp), lam1 if lam1 != 0.0 else -1.0)
    return ScenarioReadings(
        scenario=i + 1,
        table1=lam1,
        eq66=lam66,
        alpha=lam_a,
        gamma=gam,
        alpha_value=alp,
        Z0_ohm=Z0,
        sigma_gamma=_qerr(q, "table1_scenario_gamma", i),
        sigma_Z0=_qerr(q, "characteristic_impedance"),
    )


@dataclass(frozen=True)
class MappingResult:
    """The derived oscillator coupling with its uncertainty and its alternatives."""

    lambda_osc: float
    sigma: float
    lambda_tb: float  # signed, the value used
    sigma_tb: float
    gap_lat: float  # Omega^0 / omega_ref
    criterion: str
    field_mass: Optional[float]
    factor: float  # M(Omega_lat, m) of the criterion used
    readings: ScenarioReadings
    #: every enumerated alternative, name -> lambda_osc
    alternatives: Dict[str, float] = field(default_factory=dict)
    #: name -> (file quantity, note) of every input
    inputs: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    formula: str = ""

    @property
    def lambda_udw(self) -> float:
        """MAP-1 image of the derived coupling (for the perturbative columns)."""
        return lambda_udw_from_osc(self.lambda_osc, self.gap_lat)

    @property
    def band(self) -> Tuple[float, float]:
        """[min, max] over the central value +/- sigma and all alternatives."""
        vals = [self.lambda_osc - self.sigma, self.lambda_osc + self.sigma]
        vals.extend(self.alternatives.values())
        return (float(min(vals)), float(max(vals)))

    def as_dict(self) -> Dict[str, Any]:
        r = self.readings
        return {
            "lambda_osc": self.lambda_osc,
            "sigma": self.sigma,
            "lambda_tb": self.lambda_tb,
            "sigma_tb": self.sigma_tb,
            "lambda_udw_map1": self.lambda_udw,
            "gap_lat": self.gap_lat,
            "criterion": self.criterion,
            "field_mass": self.field_mass,
            "factor": self.factor,
            "readings": {
                "scenario": r.scenario,
                "table1": r.table1,
                "eq66_at_gamma": r.eq66,
                "sqrt_pi_alpha": r.alpha,
                "gamma": r.gamma,
                "alpha": r.alpha_value,
                "Z0_ohm": r.Z0_ohm,
                "mean": r.mean,
                "sample_std": r.sample_std,
                "sigma_propagated_from_file_errors": r.sigma_propagated,
            },
            "alternatives": dict(self.alternatives),
            "band": list(self.band),
            "inputs": {k: {"quantity": f, "note": n} for k, (f, n) in self.inputs.items()},
            "formula": self.formula,
        }


def lambda_osc_from_circuit(
    experiment: Any,
    gap_lat: Optional[float] = None,
    scenario_index: int = 1,
    criterion: str = DEFAULT_CRITERION,
    field_mass: Optional[float] = None,
    lambda_tb: Optional[float] = None,
    sigma_tb: Optional[float] = None,
    shifted_gap_lat: Optional[float] = None,
) -> MappingResult:
    """The derived ``lam_osc`` (``coupling='xx'``) with propagated uncertainty.

    Parameters
    ----------
    experiment : Experiment or mapping
        The loaded ``circuit_qed_harvesting/teixido-bonfill_2026_table1`` file.
    gap_lat : float, optional
        Omega^0 / omega_ref.  Default: 1.0, the loader convention of the
        candidate (omega_ref = 2 pi x ``qubit_gap_frequency``).
    scenario_index : int
        0-based Table I scenario (default 1 = scenario 2, the operating point).
    criterion : {'xp', 'finite-difference', 'physical-rate'}
        Step 6 of the module docstring.
    field_mass : float, optional
        The lattice IR mass; required by the lattice criteria and used to
        evaluate the lattice alternatives (they are omitted when None).
    lambda_tb, sigma_tb : float, optional
        Override the value / 1-sigma of the paper's coupling (default: the
        scenario's simulated Table I value and the reading spread of step 7,
        or the file's own derived ``scenario2_coupling_lambda_tb`` when present
        and consistent).
    shifted_gap_lat : float, optional
        The fully switched-on gap (Omega^0 + Delta-Omega) / omega_ref, for the
        "match at the shifted gap" alternatives (omitted when None).
    """
    q = experiment.quantities if hasattr(experiment, "quantities") else experiment
    readings = scenario_lambda_readings(experiment, scenario_index)
    if gap_lat is None:
        gap_lat = 1.0
    W = float(gap_lat)
    if lambda_tb is None:
        lambda_tb = readings.table1
    if sigma_tb is None:
        sigma_tb = readings.sigma
    if abs(lambda_tb) == 0.0:
        raise ValueError("the scenario's coupling is zero (the weak-coupling limit): nothing to map")

    M = mapping_factor(W, criterion, field_mass)
    lam_osc = abs(float(lambda_tb)) * M
    sig = float(sigma_tb) * M

    alts: Dict[str, float] = {}
    for c in CRITERIA:
        if c != "xp" and field_mass is None:
            continue
        alts[f"{c}@free_gap"] = lambda_osc_from_tb(lambda_tb, W, c, field_mass)
        if shifted_gap_lat is not None:
            alts[f"{c}@shifted_gap"] = lambda_osc_from_tb(lambda_tb, float(shifted_gap_lat), c, field_mass)
    alts["reading_table1@default"] = lambda_osc_from_tb(readings.table1, W, criterion, field_mass)
    alts["reading_eq66@default"] = lambda_osc_from_tb(readings.eq66, W, criterion, field_mass)
    alts["reading_sqrt_pi_alpha@default"] = lambda_osc_from_tb(readings.alpha, W, criterion, field_mass)
    alts["identification_map1_only"] = lambda_osc_from_udw(abs(float(lambda_tb)), W)
    if field_mass is not None:
        alts["lower_envelope(eq66_reading,physical-rate)"] = lambda_osc_from_tb(
            readings.eq66, W, "physical-rate", field_mass
        )

    inputs = {
        "lambda_tb": ("table1_scenario_lambda[%d]" % scenario_index,
                      "Table I; Figs. 8/18 captions: the value the paper simulates"),
        "gamma": ("table1_scenario_gamma[%d]" % scenario_index, "Eq. (66) reading"),
        "alpha": ("table1_scenario_alpha[%d]" % scenario_index, "lambda^2 = pi alpha reading"),
        "Z0": ("characteristic_impedance", "Eq. (7); Eq. (66) coefficient"),
        "gap": ("qubit_gap_frequency", "Eq. (30); Omega_lat = Omega^0 / omega_ref"),
        "R_K": ("(SI-2019 exact)", "h / e^2"),
    }
    if field_mass is not None:
        inputs["field_mass"] = ("(synthetic, candidate)", "lattice IR regulator; enters the lattice criteria only")
    formula = (
        "lambda_osc = |lambda_TB| * M(Omega_lat, m); M['xp'] = Omega_lat sqrt(2 Omega_lat), "
        "M['finite-difference'] = sqrt(2 Omega_lat (Omega_lat^2 - m^2)), "
        "M['physical-rate'] = Omega_lat sqrt(2 |sin kappa(Omega_lat)|); "
        "lambda_TB = -gamma sqrt(R_K / 8 pi Z_0) (TB26 Eq. 66), lambda_TB^2 = pi alpha (Eq. 33); "
        "sigma_osc = M sigma_TB, sigma_TB = sample std of the three published readings "
        "(+) file errors on gamma, Z_0 in quadrature"
    )
    return MappingResult(
        lambda_osc=lam_osc, sigma=sig, lambda_tb=float(lambda_tb), sigma_tb=float(sigma_tb),
        gap_lat=W, criterion=criterion, field_mass=field_mass, factor=M, readings=readings,
        alternatives=alts, inputs=inputs, formula=formula,
    )


def circuit_from_lambda_osc(
    lambda_osc: float,
    gap_lat: float,
    Z0_ohm: float = 50.0,
    criterion: str = DEFAULT_CRITERION,
    field_mass: Optional[float] = None,
) -> Dict[str, float]:
    """The inverse map: an oscillator coupling -> |lambda_TB|, gamma, alpha.

    Returns the magnitudes (the sign of gamma is the sign of lambda_TB, which
    the oscillator model cannot see).
    """
    lam_tb = lambda_tb_from_osc(lambda_osc, gap_lat, criterion, field_mass)
    return {
        "lambda_tb_abs": lam_tb,
        "gamma_abs": abs(gamma_from_lambda_tb(lam_tb, Z0_ohm)),
        "alpha": alpha_from_lambda_tb(lam_tb),
        "lambda_udw": lambda_udw_from_osc(lambda_osc, gap_lat),
    }
