"""Anchor 3 (Layer 5): dynamical-Casimir photon spectra vs Wilson et al. 2011.

C. M. Wilson et al., Nature 479, 376 (2011), Eqs. (1)-(2), with the theory
of Johansson, Johansson, Wilson, Nori, PRA 82, 052509 (2010), Eqs. (21),
(22), (51), (52): a harmonically modulated effective length
L_eff(t) = L_eff^0 + delta L_eff cos(omega_d t) radiates the parabolic
photon-flux density

    n_out^DCE(omega) = (delta L_eff / v)^2 omega (omega_d - omega),   omega < omega_d,

with total flux Gamma = (omega_d/12 pi)(v_e/v)^2, v_e = delta L_eff omega_d,
and the thermal input adds |S|^2 n_in(omega_d - omega) (stimulated pairs).

Parameter regime, loaded from ``experiments/dce_microwave/wilson_2011_table1``
through ``vacuum.experiments_io`` (the only unit-conversion path): the
11.30 GHz drive of Fig. 2c / Fig. 3, the line speed c_0 = 0.4 c and the
effective boundary velocity v_e = 0.05 c give the perturbation parameter
epsilon = (delta L_eff/c_0)(omega_d/2) = 0.0625 (Johansson Eq. (22)).  With
the drive placed at omega_d = 0.5 lattice units one site is 0.845 mm,
delta L_eff = 0.25 sites (0.21 mm) and the static effective length is the
DECLARED simulation choice L_eff^0 = 0.5 sites = 0.42 mm — Johansson's own
Fig. 4 value is 0.44 mm — which keeps k_d L_eff^0 = 0.25 inside the theory's
validity window (their k L_eff << 1; the file's literal 10 % inductance
modulation would put k_d L_eff^0 at 1.25, where the lattice flux is 0.62 of
the ideal formula — recorded in papers/time-modulated-vacua).  The 50 mK
sample temperature maps to T = 0.0461 lattice units, reproducing the
published n_th = 0.008 at 5 GHz (loader cross-check: 0.0083).

What is compared, and the stated tolerances (all measured values are in the
assertion messages):

- parabola SHAPE: least-squares amplitude of omega (omega_d - omega) against
  the 10-bin lattice flux density, relative RMS residual < 3 %, R^2 > 0.99;
- PREFACTOR: fitted amplitude / (delta L_eff/v)^2 within 3 % (the residual is
  the lattice: dispersion 2 sin(k/2), the site inertia at the boundary, and
  k_d L_eff^0 = 0.25 corrections), and the total flux within 3 % of Gamma;
- DEPTH SCALING: total flux exponent in delta L_eff = 2.00 +- 0.03 between
  depth scales 0.5 and 1 (perturbative regime, epsilon <= 0.0625);
- THERMAL (50 mK): produced photons / stimulated prediction within 5 % over
  the 4-6 GHz analysis band, i.e. the stimulated factor
  1 + n_in(omega_d - omega) and Wilson's statement that thermal effects are
  negligible (n = 0.008 against the vacuum coefficient 1) both hold.

Lattice estimator (vacuum.floquet.dce): occupations of the static normal
modes of the closed chain, read as a RATE between two times both shorter
than the round trip 2N to the far Dirichlet wall (the chain is then the
semi-infinite line), one flux-density estimate per mode from its own
spacing, averaged per bin.
"""

import numpy as np
import pytest

from vacuum.core import ground_state_cov, thermal_state_cov
from vacuum.experiments_io import bose_occupation
from vacuum.floquet import (
    dce_chain_K,
    dce_spectrum,
    floquet_map,
    modulated_K,
    parabola_fit,
    power_iterate,
    wilson_flux_density,
    wilson_regime,
    wilson_total_flux,
)

N_SITES = 160
L0 = 0.5
OMEGA_D_LATTICE = 0.5
MAP_TOL = 1e-8
SHAPE_RMS_TOL = 0.03
R2_MIN = 0.99
PREFACTOR_TOL = 0.03
FLUX_TOL = 0.03
EXPONENT_TOL = 0.03
THERMAL_TOL = 0.05


@pytest.fixture(scope="module")
def regime():
    return wilson_regime(OMEGA_D_LATTICE)


@pytest.fixture(scope="module")
def chain(regime):
    K0 = dce_chain_K(N_SITES, L0)
    V0 = ground_state_cov(K0)
    dL = regime.dL_over_v_lattice
    drive = modulated_K(K0, dL / L0, regime.omega_d_lattice, "cos", kind="boundary", L0=L0)
    fm = floquet_map(drive, n_steps=32, conv_tol=MAP_TOL, max_halvings=8)
    T = drive.period
    n_max = int(1.9 * N_SITES / T)  # both readings inside one round trip
    n1 = n_max // 3
    V1 = power_iterate(V0, fm.S, n1)
    V2 = power_iterate(V1, fm.S, n_max - n1)
    return K0, V0, drive, fm, V1, V2, (n_max - n1) * T, n1, n_max


def test_regime_is_the_published_one(regime):
    assert abs(regime.epsilon - 0.0625) < 1e-12, f"epsilon {regime.epsilon}"
    assert abs(regime.dL_over_v_lattice - 0.25) < 1e-12
    assert abs(regime.lattice_spacing_m - 0.000845) < 1e-5
    assert abs(regime.n_th_at_reference - regime.n_th_reference_published) < 5e-4, (
        f"loader n_th {regime.n_th_at_reference:.4f} vs published 0.008"
    )
    assert OMEGA_D_LATTICE * L0 <= 0.25  # k_d L_eff^0 inside the theory's window
    band = regime.analysis_band_lattice
    assert band[0] < OMEGA_D_LATTICE / 2 < band[1]  # omega_d/2 sits in the analysis band


def test_monodromy_converged(chain):
    K0, V0, drive, fm, V1, V2, t, n1, n_max = chain
    assert fm.converged and fm.movement < MAP_TOL and fm.symplectic_defect < 1e-9
    assert n_max * drive.period < 2 * N_SITES


def test_parabolic_spectrum_shape_and_prefactor(chain, regime):
    K0, V0, drive, fm, V1, V2, t, n1, n_max = chain
    dL = regime.dL_over_v_lattice
    sp = dce_spectrum(V2, K0, t, regime.omega_d_lattice, dL, n_bins=10, V_ref=V1)
    fit = parabola_fit(sp)
    assert fit["relative_rms_residual"] < SHAPE_RMS_TOL and fit["R2"] > R2_MIN, (
        f"parabola shape: relative RMS residual {fit['relative_rms_residual']:.4f}, "
        f"R^2 {fit['R2']:.4f}; n_out/prediction per bin {np.round(sp.n_out / sp.prediction, 3)}"
    )
    assert abs(fit["A_ratio"] - 1.0) < PREFACTOR_TOL, (
        f"prefactor: fitted A / (dL/v)^2 = {fit['A_ratio']:.4f} (Wilson Eq. 2 / Johansson Eq. 52)"
    )
    ratio = sp.total_flux / sp.total_flux_prediction
    assert abs(ratio - 1.0) < FLUX_TOL, (
        f"total flux {sp.total_flux:.4e} vs Gamma_DCE {sp.total_flux_prediction:.4e} "
        f"(ratio {ratio:.4f})"
    )
    assert abs(sp.total_flux_prediction - wilson_total_flux(regime.omega_d_lattice, dL)) < 1e-15
    # symmetric about omega_d/2 (pairs omega, omega_d - omega)
    assert abs(sp.n_out[4] - sp.n_out[5]) < 0.05 * sp.n_out[4]


def test_flux_scales_as_depth_squared(chain, regime):
    K0, V0, drive, fm, V1, V2, t, n1, n_max = chain
    dL = regime.dL_over_v_lattice
    sp1 = dce_spectrum(V2, K0, t, regime.omega_d_lattice, dL, n_bins=10, V_ref=V1)
    half = modulated_K(K0, 0.5 * dL / L0, regime.omega_d_lattice, "cos", kind="boundary", L0=L0)
    fm_h = floquet_map(half, n_steps=32, conv_tol=MAP_TOL, max_halvings=8)
    W1 = power_iterate(V0, fm_h.S, n1)
    W2 = power_iterate(W1, fm_h.S, n_max - n1)
    sp_h = dce_spectrum(W2, K0, t, regime.omega_d_lattice, 0.5 * dL, n_bins=10, V_ref=W1)
    exponent = np.log(sp1.total_flux / sp_h.total_flux) / np.log(2.0)
    assert abs(exponent - 2.0) < EXPONENT_TOL, (
        f"flux scaling exponent in delta L_eff: {exponent:.4f} "
        f"(fluxes {sp_h.total_flux:.4e} at half depth, {sp1.total_flux:.4e} at full)"
    )
    assert abs(sp_h.total_flux / sp_h.total_flux_prediction - 1.0) < FLUX_TOL


def test_thermal_input_at_50mK_is_stimulated_and_negligible(chain, regime):
    K0, V0, drive, fm, V1, V2, t, n1, n_max = chain
    T_lat = regime.temperature_lattice
    Vth = thermal_state_cov(K0, T_lat)
    Vt1 = power_iterate(Vth, fm.S, n1)
    Vt2 = power_iterate(Vt1, fm.S, n_max - n1)
    dL = regime.dL_over_v_lattice
    wd = regime.omega_d_lattice
    sp = dce_spectrum(Vt2, K0, t, wd, dL, n_bins=10, V_ref=Vt1)
    n_in = np.vectorize(lambda w: bose_occupation(w, T_lat) if w > 0 else 0.0)
    stimulated = wilson_flux_density(sp.omega_centers, wd, dL, n_in=n_in) - n_in(sp.omega_centers)
    lo, hi = regime.analysis_band_lattice
    band = (sp.omega_centers > lo) & (sp.omega_centers < hi)
    worst = float(np.max(np.abs(sp.n_out[band] / stimulated[band] - 1.0)))
    assert worst < THERMAL_TOL, (
        f"50 mK: produced / stimulated prediction in the analysis band deviates by {worst:.3f}; "
        f"stimulated factors {np.round(stimulated[band] / sp.prediction[band], 4)}"
    )
    # thermal effects negligible in the band: stimulated factor within 3 % of 1
    assert np.all(stimulated[band] / sp.prediction[band] < 1.03)
    assert n_in(0.5 * wd) < 0.02
