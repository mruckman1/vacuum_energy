"""vacuum.composite — composites of validated layers (Layer 7 leaps).

L4, harvest-then-teleport (:mod:`vacuum.composite.harvest_teleport`): the
Layer 2 nonperturbative harvesting engine ('xx' and 'xp' couplings)
feeding the Layer 3 measurement-plus-feedforward machinery, with the full
composite ledger — switching work in, ebits harvested, energy made
remotely available at Bob — and every standing audit run on every row;
:mod:`vacuum.composite.fock_conditioning` adds the non-Gaussian
(photon-number-resolved, on/off) measurements on Alice through the Fock
representation of the harvested two-mode state.
"""

from .fock_conditioning import (
    FockConditioningResult,
    complex_covariance,
    ergotropy_finite,
    fock_conditioning,
    fock_density_matrix,
    fock_moments,
    fock_tensor,
    hermite_matrix,
    optimal_fock_squeezing,
    predicted_cutoff,
    tmsv_fock_gain_closed_form,
)
from .harvest_teleport import (
    CompositeRow,
    EbitsResult,
    HarvestConfig,
    HarvestOutcome,
    Measurement,
    SurplusResult,
    composite_ledger,
    conditional_covariance,
    daemonic_surplus,
    ebits,
    field_coupling,
    gaussian_measurement,
    harvest,
    hotta_variant,
    joint_ergotropy,
    landauer_bound,
    local_ergotropy,
    local_extraction_coupled,
    mode_blocks,
    no_signaling_audit,
    optimal_measurement,
    pair_hamiltonian,
    two_mode_squeezed_thermal,
)

__all__ = [
    "HarvestConfig",
    "HarvestOutcome",
    "EbitsResult",
    "SurplusResult",
    "Measurement",
    "CompositeRow",
    "field_coupling",
    "harvest",
    "ebits",
    "mode_blocks",
    "gaussian_measurement",
    "conditional_covariance",
    "local_ergotropy",
    "local_extraction_coupled",
    "landauer_bound",
    "optimal_measurement",
    "daemonic_surplus",
    "no_signaling_audit",
    "pair_hamiltonian",
    "hotta_variant",
    "joint_ergotropy",
    "two_mode_squeezed_thermal",
    "composite_ledger",
    "FockConditioningResult",
    "complex_covariance",
    "ergotropy_finite",
    "fock_conditioning",
    "fock_density_matrix",
    "fock_moments",
    "fock_tensor",
    "hermite_matrix",
    "optimal_fock_squeezing",
    "predicted_cutoff",
    "tmsv_fock_gain_closed_form",
]
