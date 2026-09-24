"""Blind re-derivation of the circuit-QED -> lattice-oscillator coupling.

Computes the oscillator-detector coupling lam_osc of ``vacuum.detectors``
(``attach_detectors``: H_int = lam_osc x_d p_i for coupling='xp',
lam_osc x_d x_i for 'xx', unit lattice, hbar = 1) from the TRANSCRIBED
entries of ``experiments/circuit_qed_harvesting/teixido-bonfill_2026_table1``
only.  The chain is written out, equation by equation, in
``derivation.md`` next to this file; the functions below are the
arithmetic of that chain and nothing else.

Every quantity read from the parameter file is checked to carry
``provenance == 'transcribed'``; a 'derived' entry raises.  This module
was written WITHOUT reading ``vacuum/detectors/circuit_mapping.py`` or
the noise-threshold paper (see derivation.md, section 0).

Run:  .venv/bin/python docs/coupling_rederivation/rederive.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Make the repo importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vacuum.experiments_io import H_PLANCK_SI, HBAR_SI, load_experiment, to_natural  # noqa: E402

__all__ = [
    "E_CHARGE_SI",
    "R_K_SI",
    "PHI0_SI",
    "EXPERIMENT",
    "transcribed",
    "lambda_tb_per_gamma",
    "alpha_per_gamma2",
    "lambda_tb_from_gamma",
    "lattice_gap",
    "lambda_osc_xp",
    "lambda_osc_xx",
    "rederive",
]

# SI-2019 exact defined constants (BIPM SI Brochure 9th ed.).  e is the
# elementary charge in coulomb; experiments_io stores the same digits as
# EV_SI (1 eV in joule) -- we name the charge explicitly here.
E_CHARGE_SI = 1.602176634e-19  # C  (exact)
R_K_SI = H_PLANCK_SI / E_CHARGE_SI**2  # von Klitzing constant h/e^2, ohm (exact)
PHI0_SI = HBAR_SI / (2.0 * E_CHARGE_SI)  # reduced flux quantum hbar/2e, Wb

EXPERIMENT = "circuit_qed_harvesting/teixido-bonfill_2026_table1"


# ----------------------------------------------------------------- inputs --


def transcribed(exp, name: str):
    """Return the Quantity ``name`` of ``exp`` iff its provenance is 'transcribed'.

    The blind re-derivation must not consume any 'derived' entry (those
    encode the earlier derivation under test), so this is a hard error,
    not a warning.
    """
    q = exp.quantities[name]
    prov = q.extra.get("provenance")
    if prov != "transcribed":
        raise ValueError(
            f"quantity {name!r} has provenance {prov!r}; the re-derivation "
            f"consumes only 'transcribed' values"
        )
    return q


# ------------------------------------------- Teixido-Bonfill et al. (2026) --
# arXiv:2505.01516v1 equation numbers (printed numbering, verified against
# the four anchors the parameter file cites: Eqs. 7, 30, 33, 66).


def lambda_tb_per_gamma(Z0_ohm: float) -> float:
    """Eq. (66): lambda_nu / gamma_nu = -(phi_0/(hbar v l_0)) sqrt(hbar Z_0) = -sqrt(R_K/(8 pi Z_0)).

    Derivation.md section 1 re-derives this from Eqs. (5), (22), (31),
    (36)-(37); this is the closed form, evaluated with SI-exact constants.
    """
    return -math.sqrt(R_K_SI / (8.0 * math.pi * Z0_ohm))


def lambda_tb_per_gamma_explicit(Z0_ohm: float, v_m_s: float) -> float:
    """Eq. (66) in its un-simplified form -(phi_0/(hbar v l_0)) sqrt(hbar Z_0), l_0 = Z_0/v.

    Kept separately so the test can check the algebraic simplification
    (the velocity v must cancel exactly).
    """
    l0 = Z0_ohm / v_m_s  # inductance per unit length, Z_0 = v l_0 (below Eq. 5)
    return -(PHI0_SI / (HBAR_SI * v_m_s * l0)) * math.sqrt(HBAR_SI * Z0_ohm)


def alpha_per_gamma2(Z0_ohm: float) -> float:
    """Eq. (33) = Eq. (A6): alpha / gamma^2 = R_K / (8 pi^2 Z_0)."""
    return R_K_SI / (8.0 * math.pi**2 * Z0_ohm)


def lambda_tb_from_gamma(gamma: float, Z0_ohm: float) -> float:
    """Eq. (66) applied: the paper's dimensionless VGSD coupling lambda_nu."""
    return lambda_tb_per_gamma(Z0_ohm) * gamma


# ---------------------------------------------------- lattice conversion --


def lattice_gap(Omega_rad_s: float, omega_ref_rad_s: float) -> float:
    """Detector gap in lattice units: Omega_lat = Omega / omega_ref.

    On the unit lattice the energy unit is hbar v / a, i.e. the lattice
    spacing is a = v / omega_ref (derivation.md, section 2).  The
    waveguide speed v cancels: only the ratio of angular frequencies
    survives.
    """
    return Omega_rad_s / omega_ref_rad_s


def lambda_osc_xp(lambda_tb: float, Omega_lat: float) -> float:
    """'xp' (derivative / momentum) oscillator coupling, lattice units.

    lam_osc,xp = |lambda_TB| * sqrt(2 Omega_lat)

    - |lambda_TB| is the two-level coupling in lattice units EXACTLY (the
      paper's lambda is dimensionless and the lattice spacing cancels
      between the 1/a of the spatial derivative and the energy unit
      hbar v / a; derivation.md sections 2-3).
    - sqrt(2 Omega_lat) is the two-level -> oscillator matrix-element
      factor: <1|x_d|0> = 1/sqrt(2 Omega_d) for H_d = (p_d^2 + Omega_d^2
      x_d^2)/2 versus <1|sigma_x|0> = 1 (section 4).
    - The overall sign is unobservable for two identical detectors
      (x_d -> -x_d on both); we return the magnitude.
    """
    if Omega_lat <= 0.0:
        raise ValueError(f"Omega_lat must be > 0, got {Omega_lat}")
    return abs(lambda_tb) * math.sqrt(2.0 * Omega_lat)


def lambda_osc_xx(lambda_tb: float, Omega_lat: float) -> float:
    """'xx' (amplitude) oscillator coupling matched to the same spectral weight
    at the detector gap: lam_osc,xx = Omega_lat * lam_osc,xp (section 5).

    NOT an exact equivalence -- amplitude and derivative couplings see
    different spectral densities (1/omega vs omega); the match holds at
    omega = Omega_lat only.
    """
    return Omega_lat * lambda_osc_xp(lambda_tb, Omega_lat)


# ------------------------------------------------------------ the numbers --


def rederive(exp=None, scenario: int = 2, energy_unit="gap", gap="free", Z0_ohm=None):
    """Compute lam_osc for one Table-I scenario from transcribed inputs.

    Parameters
    ----------
    exp : Experiment, optional
        Loaded parameter file (default: EXPERIMENT).
    scenario : int
        Table I scenario 1..6 (scenario 1 is the lambda -> 0 limit).
    energy_unit : {'gap', 'cutoff'} or float
        Reference angular frequency omega_ref fixing the lattice energy
        unit hbar omega_ref (spacing a = v / omega_ref): 'gap' uses the
        free-qubit gap Omega^0 (detector gap = 1 on the lattice, a = 2.6 mm);
        'cutoff' uses Omega_cut (a = v/Omega_cut = 0.38 mm, the paper's
        own suggested discretization, Sec. II.3 footnote); a float is
        taken as omega_ref in rad/s.
    gap : {'free', 'coupled'}
        Which physical gap the static lattice oscillator carries:
        Omega^0 (Eq. 30, 7.3 GHz) or Omega^0 + Delta-Omega of the
        scenario (Table I, e.g. 6.8 GHz for scenario 2).
    Z0_ohm : float, optional
        Override of the impedance (default: the transcribed 50 ohm); used
        only for the from-gamma route and the uncertainty budget.

    Returns
    -------
    dict with the inputs actually used and the outputs:
        lambda_tb_table  : Table I lambda of the scenario (the simulated value)
        lambda_tb_gamma  : Eq. (66) applied to the Table I gamma (rounding spread)
        Omega_lat        : detector gap in lattice units
        a_m              : lattice spacing in metres (informational; cancels)
        lambda_osc_xp    : from lambda_tb_table
        lambda_osc_xp_lo : from lambda_tb_gamma  (the lower end of the spread)
        lambda_osc_xx, lambda_osc_xx_lo : likewise for 'xx'
        invariant        : lambda_osc_xp^2 / (2 Omega_lat) = lambda_tb^2
    """
    if exp is None:
        exp = load_experiment(EXPERIMENT)
    if not 1 <= scenario <= 6:
        raise ValueError("scenario must be 1..6")
    i = scenario - 1

    Z0 = float(transcribed(exp, "characteristic_impedance").value) if Z0_ohm is None else Z0_ohm
    v = float(transcribed(exp, "waveguide_speed").value)
    lam_table = float(transcribed(exp, "table1_scenario_lambda").value[i])
    gamma = float(transcribed(exp, "table1_scenario_gamma").value[i])
    dOm_GHz = float(transcribed(exp, "table1_scenario_gap_variation").value[i])
    lam_coef = float(transcribed(exp, "lambda_per_gamma").value)  # -4.53 printed

    q0 = transcribed(exp, "qubit_gap_frequency")
    Omega0, _ = to_natural(q0.value, q0.units)  # rad/s
    qc = transcribed(exp, "uv_cutoff_frequency")
    Omega_cut, _ = to_natural(qc.value, qc.units)  # rad/s
    dOm, _ = to_natural(dOm_GHz, "GHz")

    Omega = Omega0 if gap == "free" else Omega0 + dOm
    if energy_unit == "gap":
        omega_ref = Omega0
    elif energy_unit == "cutoff":
        omega_ref = Omega_cut
    else:
        omega_ref = float(energy_unit)

    lam_gamma = lambda_tb_from_gamma(gamma, Z0)
    Om_lat = lattice_gap(Omega, omega_ref)
    out = {
        "scenario": scenario,
        "Z0_ohm": Z0,
        "v_m_s": v,
        "gamma": gamma,
        "lambda_per_gamma_SI": lambda_tb_per_gamma(Z0),
        "lambda_per_gamma_printed": lam_coef,
        "lambda_tb_table": lam_table,
        "lambda_tb_gamma": lam_gamma,
        "Omega_rad_s": Omega,
        "omega_ref_rad_s": omega_ref,
        "Omega_lat": Om_lat,
        "a_m": v / omega_ref,
        "lambda_osc_xp": lambda_osc_xp(lam_table, Om_lat),
        "lambda_osc_xp_lo": lambda_osc_xp(lam_gamma, Om_lat),
        "lambda_osc_xx": lambda_osc_xx(lam_table, Om_lat),
        "lambda_osc_xx_lo": lambda_osc_xx(lam_gamma, Om_lat),
        "invariant": lambda_osc_xp(lam_table, Om_lat) ** 2 / (2.0 * Om_lat),
    }
    return out


def _main():
    exp = load_experiment(EXPERIMENT)
    print(f"source: {exp.source[:90]}...")
    print(f"R_K = h/e^2 = {R_K_SI:.6f} ohm (SI-exact)")
    Z0 = float(transcribed(exp, "characteristic_impedance").value)
    print(
        f"Eq. (66) coefficient  -sqrt(R_K/(8 pi Z0)) at Z0 = {Z0:g} ohm: "
        f"{lambda_tb_per_gamma(Z0):+.5f}   (printed: {transcribed(exp, 'lambda_per_gamma').value})"
    )
    print(
        f"Eq. (33) coefficient   R_K/(8 pi^2 Z0)                : "
        f"{alpha_per_gamma2(Z0):.5f}   (printed: {transcribed(exp, 'spin_boson_alpha_coefficient').value})"
    )
    print()
    hdr = f"{'sc':>2} {'lam_TB':>7} {'gamma':>6} | {'unit':>6} {'gap':>7} {'Om_lat':>8} {'a[mm]':>6} | {'lam_xp':>8} {'(lo)':>8} {'lam_xx':>8} {'(lo)':>8}"
    print(hdr)
    print("-" * len(hdr))
    for sc in range(1, 7):
        for unit in ("gap", "cutoff"):
            for gp in ("free", "coupled"):
                r = rederive(exp, sc, unit, gp)
                print(
                    f"{sc:>2} {r['lambda_tb_table']:>7.3f} {r['gamma']:>6.2f} | "
                    f"{unit:>6} {gp:>7} {r['Omega_lat']:>8.4f} {r['a_m']*1e3:>6.2f} | "
                    f"{r['lambda_osc_xp']:>8.5f} {r['lambda_osc_xp_lo']:>8.5f} "
                    f"{r['lambda_osc_xx']:>8.5f} {r['lambda_osc_xx_lo']:>8.5f}"
                )
    print()
    r = rederive(exp, 2, "gap", "free")
    print("HEADLINE (scenario 2, lambda_TB = -0.10, free gap 7.3 GHz):")
    print(f"  gap-unit lattice (Omega_lat = 1, a = {r['a_m']*1e3:.2f} mm):     lam_osc,xp = {r['lambda_osc_xp']:.4f}  lam_osc,xx = {r['lambda_osc_xx']:.4f}")
    r = rederive(exp, 2, "cutoff", "free")
    print(f"  cutoff-unit lattice (Omega_lat = {r['Omega_lat']:.3f}, a = {r['a_m']*1e3:.2f} mm): lam_osc,xp = {r['lambda_osc_xp']:.4f}  lam_osc,xx = {r['lambda_osc_xx']:.5f}")
    print(f"  unit-free invariant lam_osc,xp^2/(2 Omega_lat) = lambda_TB^2 = {r['invariant']:.4f}")


if __name__ == "__main__":
    _main()
