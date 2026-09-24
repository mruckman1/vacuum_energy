"""vacuum.geometry -- geometry from entanglement, computationally (Layer 4).

Three instruments plus the honest experiment (PLAN.md Layer 4; README.md in
this directory for the planned API and anchors):

* :mod:`vacuum.geometry.mera`      -- scale-invariant ternary MERA for the
  critical Ising chain: energy, scaling dimensions, central charge, minimal
  cuts and geodesics through the network (Evenbly-Vidal 2009; Pfeifer-
  Evenbly-Vidal 2009; Swingle 2012).
* :mod:`vacuum.geometry.migeometry` -- mutual-information distances, flat and
  hyperbolic embeddings with their stress, Gromov delta, and
  ``failure_map`` over massive / disordered / detuned chains.
* :mod:`vacuum.geometry.happy`     -- the HaPPY pentagon code and hexagon
  state in the stabilizer formalism: greedy wedges, min cuts, exact
  Ryu-Takayanagi checks, bulk reconstruction (Pastawski-Yoshida-Harlow-
  Preskill 2015).
* :mod:`vacuum.geometry.ising_exact` -- exact Majorana reference for the
  TFIM/Kitaev chain (block entropies, mutual information, finite disordered
  chains).
* :mod:`vacuum.geometry.jacobson`  -- entanglement-equilibrium residuals and
  ``constrain_dynamics`` for the toy-Jacobson leap (L2).
* :mod:`vacuum.geometry.jacobson2d` -- the 2+1 extension of the toy: an N x N
  square-lattice scalar, disks of radius R, the CHM weight on the lattice's own
  bond-split energy density, and the conformal improvement term that d = 3
  needs and d = 2 does not.
* :mod:`vacuum.geometry.jacobson_continuum` -- the continuum statements the
  lattice residual approximates: the exact first law paired with the CHM
  modular Hamiltonian of an interval (mpmath, no lattice code) and the
  O(a^2) weight-placement error of the bond split.

Conventions: hbar = 1, nats, block quadrature ordering (docs/API.md).
"""

from .mera import (
    DATA_DIR,
    E0_ISING_EXACT,
    TernaryMERA,
    ascend,
    ascend_avg,
    block_entropies_level0,
    central_charge_estimate,
    descend,
    descend_avg,
    energy_density,
    environment,
    fixed_point_density,
    ising_two_site_term,
    load_mera,
    local_init_mera,
    mera_geodesic,
    mera_ising,
    mera_network,
    minimal_cut,
    minimal_cut_entropy,
    one_site_scaling_superoperator,
    optimize_mera,
    random_mera,
    save_mera,
    scaling_dimensions,
    scaling_dimensions_by_parity,
)
from .migeometry import (
    MODEL_FAMILIES,
    boson_correlation_length,
    distance_law,
    failure_map,
    geometry_report,
    hyperbolic_embedding,
    hyperbolicity,
    mi_distance,
    mi_matrix,
    reconstruct_metric,
)
from .happy import (
    HaPPYCode,
    MinCut,
    RTResult,
    StabilizerTableau,
    boundary_entropy,
    bulk_reconstruction,
    five_qubit_code_generators,
    greedy_wedge,
    happy_code,
    layer_counts,
    min_cut,
    perfect_tensor_generators,
    reconstructible,
    rt_entropy,
)
from .ising_exact import (
    E0_TFIM,
    kitaev_chain_gamma,
    majorana_block_entropy,
    majorana_mutual_information,
    tfim_block_G,
    tfim_block_entropy,
    tfim_correlation_length,
    tfim_majorana_g,
    tfim_mutual_information,
)
from .jacobson_continuum import (
    bond_split_discretization_prediction,
    bond_split_weight_coefficient,
    chm_beta,
    conformal_killing_defect,
    continuum_first_law_defect,
    entropy_weight,
    modular_coordinate,
    modular_coordinate_inverse,
    modular_symplectic_eigenvalue,
    modular_wavepacket,
)
from .jacobson2d import (
    XI_IMPROVE_3D,
    chm_beta_2d,
    disk_ball,
    equilibrium_residual_2d,
    first_law_audit_2d,
    free_weight_residual_2d,
    lattice_coords,
    local_modular_form_2d,
    sector_split_2d,
    sine_basis,
    square_lattice_K,
    squeeze_modes_2d,
    vacuum_block_2d,
)
from .jacobson import (
    bulk_dispersion,
    bw_weights,
    centred_ball,
    constrain_dynamics,
    coupling_chain_K,
    dispersion_departure,
    dispersion_power_K,
    eh_profiles,
    entanglement_equilibrium_residual,
    equilibrium_residual,
    first_law_audit,
    fit_dynamics,
    k4_sensitivity,
    local_energy_forms,
    local_modular_form,
    neighbour_laplacian,
    random_symmetric_chain_K,
    residual_scan,
    squeeze_modes,
    squeeze_variation,
    squeezed_ball_covariance,
)

__all__ = [
    # mera
    "E0_ISING_EXACT", "TernaryMERA", "block_entropies_level0",
    "central_charge_estimate", "energy_density", "fixed_point_density",
    "ising_two_site_term", "load_mera", "mera_geodesic", "mera_ising",
    "minimal_cut", "minimal_cut_entropy", "optimize_mera", "random_mera",
    "save_mera", "scaling_dimensions",
    "DATA_DIR", "local_init_mera", "ascend", "descend", "ascend_avg", "descend_avg",
    "environment", "scaling_dimensions_by_parity", "one_site_scaling_superoperator",
    "mera_network",
    # mi geometry
    "MODEL_FAMILIES", "boson_correlation_length", "distance_law",
    "failure_map", "geometry_report", "hyperbolic_embedding", "hyperbolicity",
    "mi_distance", "mi_matrix", "reconstruct_metric",
    # happy
    "HaPPYCode", "boundary_entropy", "bulk_reconstruction", "greedy_wedge",
    "happy_code", "layer_counts", "min_cut", "reconstructible", "rt_entropy",
    "StabilizerTableau", "MinCut", "RTResult", "five_qubit_code_generators",
    "perfect_tensor_generators",
    # exact ising
    "E0_TFIM", "kitaev_chain_gamma", "majorana_block_entropy",
    "majorana_mutual_information", "tfim_block_entropy", "tfim_block_G",
    "tfim_correlation_length", "tfim_majorana_g", "tfim_mutual_information",
    # jacobson
    "bw_weights", "constrain_dynamics", "entanglement_equilibrium_residual",
    "local_energy_forms",
    "neighbour_laplacian", "coupling_chain_K", "random_symmetric_chain_K",
    "dispersion_power_K", "bulk_dispersion", "dispersion_departure", "centred_ball",
    "local_modular_form", "squeeze_modes", "squeeze_variation",
    "squeezed_ball_covariance", "equilibrium_residual", "first_law_audit",
    "eh_profiles", "fit_dynamics", "residual_scan", "k4_sensitivity",
    # jacobson: the 2+1 extension (papers/toy-jacobson, section J)
    "XI_IMPROVE_3D", "sine_basis", "square_lattice_K", "lattice_coords",
    "disk_ball", "chm_beta_2d", "vacuum_block_2d", "squeeze_modes_2d",
    "local_modular_form_2d", "equilibrium_residual_2d", "first_law_audit_2d",
    "free_weight_residual_2d", "sector_split_2d",
    # jacobson: the continuum companion (papers/toy-jacobson, anchor 1 proof)
    "chm_beta", "modular_coordinate", "modular_coordinate_inverse",
    "modular_symplectic_eigenvalue", "entropy_weight", "modular_wavepacket",
    "continuum_first_law_defect", "conformal_killing_defect",
    "bond_split_weight_coefficient", "bond_split_discretization_prediction",
]
