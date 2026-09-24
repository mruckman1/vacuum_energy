"""vacuum.modular — numerical modular (entanglement) Hamiltonians (Layer 4).

Fermionic side (Peschel-Eisler, exact): interval / two-interval / multi-interval
entanglement Hamiltonians at float64 or arbitrary precision, the
Bisognano-Wichmann profile, the Casini-Huerta two-interval reference (local and
nonlocal terms) with the lattice-to-continuum dictionary, bilocal weights,
modular flow and the multi-interval scan. Bosonic side (Gaussian, via
vacuum.core): region_modular, modular_energy_profile and the Layer-1 first-law
anchor first_law_check. Conventions per docs/API.md: hbar = 1, vacuum nu = 1/2,
block quadrature ordering, nats; hopping t = 1/2 and k_F = pi/2 on the fermion side.
"""

from .bosonic import modular_energy_profile, region_modular
from .casini_huerta import (
    casini_huerta_exact,
    ch_beta,
    ch_bilinear,
    ch_conjugate_points,
    ch_mixing_angle,
    ch_mobius_conjugate,
    ch_mutual_information,
    ch_nonlocal_weight,
    ch_single_interval_flow,
    ch_trajectory,
    ch_z,
    ch_zprime,
    from_chiral_gauge,
    interval_of_sites,
    intervals_from_parts,
    site_positions,
    to_chiral_gauge,
    which_interval,
)
from .fermionic import (
    ModularPrecisionWarning,
    bilocal_weight,
    bw_profile,
    chiral_bilinear,
    eisler_peschel_eq26,
    eisler_peschel_eq27,
    eisler_peschel_eq54,
    entanglement_spectrum,
    entropy_from_h,
    flow_wavefunction,
    interval_modular,
    locality_score,
    modular_flow,
    modular_flow_unitary,
    multi_interval_modular,
    nonlocal_decay,
    part_centroids,
    part_weights,
    required_dps,
    two_interval_modular,
    wavepacket,
)
from .ctm import ctm_level_spacing, ctm_modulus, ctm_slope, ctm_slope_expansion, lattice_boost, triangle_h
from .first_law import first_law_check
from .lattice import (
    correlation_length,
    infinite_chain_C,
    massive_chain_C,
    ring_C,
    staggered_mass_single_particle,
    to_float,
    to_mp,
)
from .scan import modular_hamiltonian_auto, multi_interval_scan, sites_of_config

from .ctm import (
    eisler_2025_prefactor,
    eisler_2025_levels,
)

__all__ = [
    # fermionic
    "ModularPrecisionWarning",
    "entanglement_spectrum",
    "required_dps",
    "interval_modular",
    "two_interval_modular",
    "multi_interval_modular",
    "entropy_from_h",
    "bw_profile",
    "bilocal_weight",
    "chiral_bilinear",
    "nonlocal_decay",
    "locality_score",
    "modular_flow",
    "modular_flow_unitary",
    "flow_wavefunction",
    "wavepacket",
    "part_weights",
    "part_centroids",
    "eisler_peschel_eq26",
    "eisler_peschel_eq27",
    "eisler_peschel_eq54",
    # lattice closed forms
    "infinite_chain_C",
    "ring_C",
    "massive_chain_C",
    "correlation_length",
    "staggered_mass_single_particle",
    "to_mp",
    "to_float",
    # Casini-Huerta reference
    "casini_huerta_exact",
    "ch_z",
    "ch_zprime",
    "ch_beta",
    "ch_conjugate_points",
    "ch_mobius_conjugate",
    "ch_nonlocal_weight",
    "ch_bilinear",
    "ch_trajectory",
    "ch_single_interval_flow",
    "ch_mixing_angle",
    "ch_mutual_information",
    "to_chiral_gauge",
    "from_chiral_gauge",
    "site_positions",
    "interval_of_sites",
    "intervals_from_parts",
    "which_interval",
    # bosonic
    "region_modular",
    "modular_energy_profile",
    "first_law_check",
    # corner transfer matrix closed forms
    "ctm_modulus",
    "ctm_level_spacing",
    "ctm_slope",
    "ctm_slope_expansion",
    "lattice_boost",
    "triangle_h",
    # scan
    "multi_interval_scan",
    "modular_hamiltonian_auto",
    "sites_of_config",
    "eisler_2025_prefactor",
    "eisler_2025_levels",
]
