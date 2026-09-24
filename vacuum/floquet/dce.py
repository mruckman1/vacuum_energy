"""Dynamical Casimir effect of a modulated boundary: the Wilson 2011 anchor.

Model.  A transmission line terminated by a flux-tunable SQUID obeys the
boundary condition of Johansson, Johansson, Wilson, Nori, PRA 82, 052509
(2010), Eq. (10); dropping the SQUID capacitance it is the mixed condition
Phi(0) + L_eff dPhi/dx|_0 = 0 with the effective length
L_eff = (Phi_0/2 pi)^2 / (E_J L_0) (their Eq. (17)) — a perfect mirror a
distance L_eff behind the SQUID.  Modulating E_J harmonically modulates
L_eff(t) = L_eff^0 + delta L_eff cos(omega_d t), and first-order scattering
theory gives (their Eqs. (21), (51), (52); Wilson et al., Nature 479, 376
(2011), Eqs. (1)-(2))

    a_out(omega) = R(omega) a_in(omega) + S(omega) a_in^dag(omega_d - omega),
    S(omega)     = -i (delta L_eff / v) sqrt(omega (omega_d - omega)),
    n_out(omega) = n_in(omega) + |S|^2 n_in(omega_d - omega) + |S|^2,

so the vacuum (dynamical Casimir) contribution is the parabola

    n_out^DCE(omega) = (delta L_eff / v)^2 omega (omega_d - omega),   omega < omega_d,

the photon flux per unit time and unit *frequency* bandwidth: the total
flux is Gamma_DCE = int n_out domega / 2 pi = (omega_d / 12 pi)(v_e/v)^2 with
v_e = delta L_eff omega_d (Wilson 2011, text after Eq. (2)).  Validity:
epsilon = (delta L_eff / v)(omega_d / 2) << 1 (Johansson Eq. (22)) and
k L_eff^0 << 1 (Johansson, after Eq. (17)).

Lattice transcription (:func:`vacuum.floquet.dce_chain_K`,
``modulated_K(kind='boundary')``): unit spacing a and unit speed v; a spring
1/L_eff(t) to ground on site 0 realises Phi(0) = L_eff (Phi_1 - Phi_0), the
effective mirror at x = -L_eff, and the chain ends in a Dirichlet wall at
site N.  Run for a time t < 2N (no round trip to the far wall) the closed
chain *is* the semi-infinite line: the photons produced at the boundary
have not yet returned, and their number in a frequency bin of the static
normal modes of K0 counts the emitted flux,

    sum_{j in bin} <n_j>(t)  =  t * int_bin n_out(omega) domega / 2 pi,

which :func:`dce_spectrum` inverts to a binned flux density that is
compared with the parabola (shape, prefactor) and with the depth scaling
(delta L_eff)^2 in tests/test_floquet_dce.py.  The lattice dispersion
omega = 2 sin(k/2) is within 1 % of linear for omega < 0.5, and the exact
lattice reflection phase tan(kL) = sin k / (1/L_eff - 1 + cos k) reduces to
L = L_eff for k -> 0, so the comparison is made at omega_d <= 0.5 in lattice
units and the residual is reported, not hidden.

:func:`wilson_regime` loads ``experiments/dce_microwave/wilson_2011_table1``
through ``vacuum.experiments_io`` (the ONLY place published units become
lattice units) and returns the dimensionless drive: epsilon from the
effective boundary velocity v_e = 0.05 c, the line speed c_0 = 0.4 c and the
11.30 GHz drive of Fig. 2c / Fig. 3 give delta L_eff / c_0 and hence, at a
chosen lattice omega_d, delta L_eff in sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from vacuum.experiments_io import load_experiment, to_natural, bose_occupation

from .drives import dce_chain_K, modulated_K
from .spectra import mode_occupations, normal_modes

__all__ = [
    "wilson_flux_density",
    "wilson_total_flux",
    "wilson_regime",
    "WilsonRegime",
    "DCESpectrum",
    "dce_spectrum",
    "parabola_fit",
    "C_LIGHT_SI",
]

#: speed of light (SI 2019 exact definition) — only used to convert the
#: published *fractions of c* in the Wilson file to a velocity ratio.
C_LIGHT_SI = 299792458.0


def wilson_flux_density(omega, omega_d, dL_over_v, n_in=None):
    """n_out(omega) of Wilson 2011 Eq. (2) / Johansson 2010 Eq. (52).

    ``dL_over_v`` = delta L_eff / v (a time; dimensionless in lattice units
    where v = 1).  With ``n_in(omega)`` (callable, thermal input) the
    stimulated term |S|^2 n_in(omega_d - omega) and the reflected input
    n_in(omega) are included; default vacuum input.
    """
    omega = np.asarray(omega, dtype=float)
    S2 = dL_over_v**2 * np.where(omega < omega_d, omega * (omega_d - omega), 0.0)
    if n_in is None:
        return S2
    nin = np.vectorize(n_in)
    return nin(omega) + S2 * nin(np.abs(omega_d - omega)) + S2


def wilson_total_flux(omega_d, dL_over_v):
    """Gamma_DCE = (omega_d/12 pi)(v_e/v)^2, v_e = delta L_eff omega_d (Wilson 2011)."""
    return omega_d / (12.0 * np.pi) * (dL_over_v * omega_d) ** 2


@dataclass(frozen=True)
class WilsonRegime:
    """Dimensionless Wilson 2011 drive parameters (see :func:`wilson_regime`)."""

    omega_d_lattice: float  # drive frequency in lattice units
    epsilon: float  # (delta L_eff / v)(omega_d / 2), Johansson Eq. (22)
    dL_over_v_lattice: float  # delta L_eff in sites (v = 1)
    omega_ref: float  # rad/s per lattice unit
    lattice_spacing_m: float
    line_speed_m_s: float
    boundary_velocity_m_s: float
    temperature_lattice: float
    analysis_band_lattice: tuple
    n_th_at_reference: float
    n_th_reference_published: float
    source: str


def wilson_regime(omega_d_lattice=0.5, drive_key="pump_frequency_fig2c"):
    """Load the Wilson 2011 file and map it to lattice units.

    The lattice energy unit is fixed by putting the published drive at
    ``omega_d_lattice`` (default 0.5, well inside the linear-dispersion
    window of the unit-spacing chain): omega_ref = omega_d / omega_d_lattice.
    The lattice spacing follows from the line speed, a = c_0 / omega_ref,
    and delta L_eff from the published effective boundary velocity,
    v_e = delta L_eff omega_d.
    """
    exp = load_experiment("dce_microwave/wilson_2011_table1")
    q = exp.quantities
    omega_d, _ = to_natural(q[drive_key].value, q[drive_key].units)  # rad/s
    c0 = float(q["line_speed_fraction_of_c"].value) * C_LIGHT_SI
    ve = float(q["effective_boundary_velocity_fraction"].value) * C_LIGHT_SI
    omega_ref = omega_d / float(omega_d_lattice)
    a = c0 / omega_ref  # one lattice site in metres (v = 1 in lattice units)
    dL = ve / omega_d  # delta L_eff in metres
    dL_lat = dL / a
    eps = 0.5 * dL_lat * float(omega_d_lattice)
    T_nat, _ = to_natural(q["sample_temperature"].value, q["sample_temperature"].units)
    T_lat = T_nat / omega_ref
    band = to_natural(q["analysis_band"].value, q["analysis_band"].units)[0]
    band_lat = tuple(b / omega_ref for b in band)
    w_ref_nat, _ = to_natural(q["thermal_occupation_reference_frequency"].value,
                              q["thermal_occupation_reference_frequency"].units)
    n_th = bose_occupation(w_ref_nat, T_nat)
    return WilsonRegime(
        omega_d_lattice=float(omega_d_lattice), epsilon=float(eps),
        dL_over_v_lattice=float(dL_lat), omega_ref=float(omega_ref),
        lattice_spacing_m=float(a), line_speed_m_s=float(c0),
        boundary_velocity_m_s=float(ve), temperature_lattice=float(T_lat),
        analysis_band_lattice=band_lat, n_th_at_reference=float(n_th),
        n_th_reference_published=float(q["thermal_occupation_at_reference"].value),
        source=exp.source,
    )


@dataclass
class DCESpectrum:
    """Binned photon-flux density from a boundary-driven lattice run."""

    omega_edges: np.ndarray
    omega_centers: np.ndarray
    n_out: np.ndarray  # flux density per unit time per unit frequency (Hz-like)
    n_out_err: np.ndarray  # spread across the modes of a bin (std of the per-mode estimate)
    occupations: np.ndarray  # per normal mode (ascending omega)
    omega_k: np.ndarray
    t: float
    omega_d: float
    n_periods: int
    prediction: np.ndarray  # Wilson parabola on the centers
    dL_over_v: float
    total_flux: float
    total_flux_prediction: float


def dce_spectrum(V, K0, t, omega_d, dL_over_v, *, n_bins=10, omega_max=None,
                 n_in=None, V_ref=None, n_periods=None):
    """Bin the produced photons into a flux density and compare with Wilson.

    Parameters
    ----------
    V : covariance after time t under the boundary drive.
    K0 : static coupling matrix (:func:`dce_chain_K`).
    t : float — elapsed drive time (must be < 2N for the open-line reading).
    omega_d, dL_over_v : drive frequency and delta L_eff (lattice units).
    n_bins : bins over [0, omega_max] (default omega_max = omega_d).
    n_in : callable, optional — thermal input occupation for the prediction.
    V_ref : covariance whose occupations are subtracted (e.g. the thermal
        initial state), so ``occupations`` are the *produced* photons.
    """
    omega, U = normal_modes(K0)
    occ = mode_occupations(V, omega=omega, U=U)
    if V_ref is not None:
        occ = occ - mode_occupations(V_ref, omega=omega, U=U)
    omega_max = float(omega_d) if omega_max is None else float(omega_max)
    edges = np.linspace(0.0, omega_max, int(n_bins) + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    # One flux-density estimate per normal mode, n_j / (t * Delta omega_j / 2 pi)
    # with Delta omega_j the local mode spacing (each mode owns one spacing of
    # bandwidth), averaged within the bin.  Summing the bin's photons over the
    # bin width instead would alternate with the integer number of modes a
    # bin happens to contain (a pure binning artefact at 2-3 modes per bin).
    spacing = np.gradient(omega)
    per_mode = occ / (float(t) * spacing / (2.0 * np.pi))
    n_out = np.zeros(n_bins)
    err = np.zeros(n_bins)
    for b in range(n_bins):
        sel = (omega >= edges[b]) & (omega < edges[b + 1])
        if np.any(sel):
            n_out[b] = float(np.mean(per_mode[sel]))
            err[b] = np.std(per_mode[sel]) / np.sqrt(np.sum(sel)) if np.sum(sel) > 1 else np.nan
        else:
            n_out[b] = np.nan
            err[b] = np.nan
    pred = wilson_flux_density(centers, omega_d, dL_over_v, n_in=n_in)
    total = float(np.sum(occ[omega < omega_d]) / float(t))
    return DCESpectrum(
        omega_edges=edges, omega_centers=centers, n_out=n_out, n_out_err=err,
        occupations=occ, omega_k=omega, t=float(t), omega_d=float(omega_d),
        n_periods=int(n_periods) if n_periods is not None else -1,
        prediction=pred, dL_over_v=float(dL_over_v), total_flux=total,
        total_flux_prediction=float(wilson_total_flux(omega_d, dL_over_v)),
    )


def parabola_fit(spectrum: DCESpectrum, mask=None):
    """Least-squares amplitude A of n_out = A omega (omega_d - omega).

    Returns dict(A, A_predicted = dL_over_v^2, relative_rms_residual, R2).
    """
    x = spectrum.omega_centers
    y = spectrum.n_out
    m = np.isfinite(y) if mask is None else (np.isfinite(y) & np.asarray(mask))
    g = x[m] * (spectrum.omega_d - x[m])
    A = float(np.sum(g * y[m]) / np.sum(g * g))
    resid = y[m] - A * g
    rel = float(np.sqrt(np.mean(resid**2)) / np.sqrt(np.mean((A * g) ** 2)))
    ss_tot = float(np.sum((y[m] - np.mean(y[m])) ** 2))
    R2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"A": A, "A_predicted": spectrum.dL_over_v**2, "A_ratio": A / spectrum.dL_over_v**2,
            "relative_rms_residual": rel, "R2": R2, "n_points": int(np.sum(m))}
