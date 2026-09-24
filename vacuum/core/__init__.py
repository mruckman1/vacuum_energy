"""vacuum.core — Gaussian covariance-matrix engine of the Vacuum Program.

Conventions (docs/API.md, non-negotiable): hbar = 1, vacuum covariance I/2
(symplectic eigenvalue nu = 1/2 per mode), block quadrature ordering
R = (x_1..x_N, p_1..p_N) with Omega = [[0, I], [-I, 0]], entropies and
negativities in nats.
"""

from .gaussian import (
    Omega,
    entanglement_hamiltonian,
    entropy,
    entropy_from_nu,
    evolve,
    ground_state_cov,
    log_negativity,
    mean_energy,
    modular_energy,
    mutual_information,
    partial_transpose,
    reduce,
    symplectic_eigenvalues,
    symplectic_from_quadratic,
    symplectic_inverse,
    thermal_state_cov,
    williamson,
)
from .dynamics import chain_propagator
from .channels import (
    amplifier,
    amplifier_XY,
    apply_channel,
    loss,
    loss_XY,
    thermal_noise,
    thermal_noise_XY,
)
from .models import harmonic_chain_K
from .precision import (
    entropy_mp,
    entropy_precise,
    ground_state_cov_mp,
    longdouble_refine,
    near_vacuum_floor,
    symplectic_eigenvalues_mp,
    symplectic_eigenvalues_precise,
    symplectic_margins_mp,
)
from .radial import partial_wave_entropies, sphere_entropy, srednicki_K

__all__ = [
    # gaussian
    "Omega",
    "ground_state_cov",
    "thermal_state_cov",
    "symplectic_eigenvalues",
    "williamson",
    "entropy",
    "entropy_from_nu",
    "reduce",
    "partial_transpose",
    "log_negativity",
    "mutual_information",
    "entanglement_hamiltonian",
    "modular_energy",
    "symplectic_from_quadratic",
    "symplectic_inverse",
    "evolve",
    "mean_energy",
    # channels
    "apply_channel",
    "loss",
    "thermal_noise",
    "amplifier",
    "loss_XY",
    "thermal_noise_XY",
    "amplifier_XY",
    # dynamics
    "chain_propagator",
    # models
    "harmonic_chain_K",
    # precision
    "near_vacuum_floor",
    "symplectic_eigenvalues_mp",
    "entropy_mp",
    "symplectic_eigenvalues_precise",
    "entropy_precise",
    "longdouble_refine",
    "symplectic_margins_mp",
    "ground_state_cov_mp",
    # radial (Srednicki geometry)
    "srednicki_K",
    "partial_wave_entropies",
    "sphere_entropy",
]
