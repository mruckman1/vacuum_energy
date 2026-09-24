"""vacuum.interacting — interacting vacua: the lattice lambda-phi^4 chain
via DMRG / TEBD (PLAN.md Layer 7, leap L5).

Modules
-------
:mod:`vacuum.interacting.phi4`
    The truncated-boson TeNPy model (projected local operators, basis
    frequency choice, n_max as an explicit convergence axis), DMRG ground
    states, and the pure-numpy perturbation theory (E_1, E_2, tadpole mass
    renormalization, Hartree chain) that anchors it at small lambda.
:mod:`vacuum.interacting.observables`
    Two-point functions and the covariance, block entropies, dense reduced
    density matrices, exact partial-transpose negativity and mutual
    information of small (possibly disjoint) blocks, the Gaussian-covariance
    proxy for large blocks, and the PSD local energy density.
:mod:`vacuum.interacting.qei_mps`
    The first interacting QEI number: the smeared local energy along a
    worldline, minimized over a stated family of locally squeezed MPS by
    TEBD real-time evolution (a variational upper bound on the infimum),
    against the free exact infimum of ``vacuum.inequalities.qei``.
:mod:`vacuum.interacting.qei_exact`
    The *exact* lattice QEI infimum: the time-smeared energy is built as one
    Hermitian MPO by operator-space (Heisenberg) TEBD of h_x against the
    phi^4 Hamiltonian and quadrature in t, and DMRG on that MPO returns its
    lowest eigenvalue -- the infimum over ALL states, with the extremal
    state exhibited. Anchored at lambda = 0 against the Gaussian Williamson
    infimum of ``vacuum.inequalities.qei``; the MPO truncation replaces the
    variational family gap as the uncertainty.
:mod:`vacuum.interacting.audits`
    Passivity of the interacting ground state under random local unitaries;
    the chi-doubling gate.

TeNPy is imported lazily inside the guarded modules; the numpy-only pieces
(local operators, perturbation theory, Hartree chain, the free family
optimization) work without it. Conventions: hbar = 1, unit lattice spacing,
block quadrature ordering, nats (docs/API.md).
"""

from .phi4 import (
    CRITICAL_RATIO_LAMBDA_OVER_MU2,
    GroundState,
    first_order_covariance,
    free_coupling_matrix,
    hartree_coupling_matrix,
    hartree_mass_squared,
    local_boson_ops,
    mps_energy,
    oscillator_frequency,
    perturbative_energy,
    phi4_ground_mps,
    phi4_model,
    phi4_model_class,
)
from .observables import (
    block_entropies,
    covariance_from_mps,
    energy_density_profile,
    entropy_of_rho,
    gaussian_proxy_mutual_information,
    gaussian_proxy_negativity,
    local_energy_density,
    log_negativity_mps,
    mutual_information_mps,
    reduced_density_matrix,
    two_point_functions,
)
from .audits import (
    apply_local_unitary,
    passivity_audit_mps,
    random_local_unitary,
    relative_change,
)
from .qei_exact import (
    QEIExactResult,
    occupation_weights,
    SmearedOperator,
    free_exact_infimum,
    heisenberg_smeared_operator,
    local_energy_infimum_dense,
    local_energy_operator_dense,
    qei_exact_mps,
)
from .qei_mps import (
    QEIVariationalResult,
    SqueezeFamily,
    apply_squeeze_family,
    family_symplectic,
    free_family_energy,
    optimize_free_family,
    qei_variational_mps,
    smeared_energy_mps,
    squeeze_family,
)

from .qei_ed import (
    dense_hamiltonian,
    dense_local_energy,
    parity_sectors,
    time_kernel,
    QEIDenseResult,
    qei_exact_dense,
    smeared_operator_dense,
    ChebyshevPropagator,
    SmearedOperatorMatvec,
    QEILanczosResult,
    qei_exact_lanczos,
)
from .qei_exact import (
    first_order_infimum_slope,
)
from .qei_reference import (
    standing_wave_modes,
    free_dispersion,
    harmonic_inversion,
    peak_width,
    DispersionResult,
    interacting_dispersion_tebd,
    excitation_gap_dmrg,
    fit_dispersion_couplings,
    cosine_coefficients,
    keff_from_couplings,
    keff_density_terms,
    free_infimum_keff,
    energy_pieces_dense,
    heisenberg_smeared_operator_dense,
    decompose_smeared_energy,
    schroedinger_pieces,
)

__all__ = [
    "CRITICAL_RATIO_LAMBDA_OVER_MU2",
    "GroundState",
    "free_coupling_matrix",
    "local_boson_ops",
    "oscillator_frequency",
    "phi4_model_class",
    "phi4_model",
    "phi4_ground_mps",
    "mps_energy",
    "perturbative_energy",
    "first_order_covariance",
    "hartree_coupling_matrix",
    "hartree_mass_squared",
    "two_point_functions",
    "covariance_from_mps",
    "block_entropies",
    "reduced_density_matrix",
    "entropy_of_rho",
    "log_negativity_mps",
    "mutual_information_mps",
    "gaussian_proxy_negativity",
    "gaussian_proxy_mutual_information",
    "energy_density_profile",
    "local_energy_density",
    "random_local_unitary",
    "apply_local_unitary",
    "passivity_audit_mps",
    "relative_change",
    "SqueezeFamily",
    "squeeze_family",
    "family_symplectic",
    "free_family_energy",
    "optimize_free_family",
    "apply_squeeze_family",
    "smeared_energy_mps",
    "qei_variational_mps",
    "QEIVariationalResult",
    # qei_exact: the exact infimum (operator TEBD + DMRG on O_f)
    "local_energy_operator_dense",
    "local_energy_infimum_dense",
    "SmearedOperator",
    "heisenberg_smeared_operator",
    "QEIExactResult",
    "qei_exact_mps",
    "free_exact_infimum",
    "occupation_weights",
    "dense_hamiltonian",
    "dense_local_energy",
    "parity_sectors",
    "time_kernel",
    "QEIDenseResult",
    "qei_exact_dense",
    "smeared_operator_dense",
    "ChebyshevPropagator",
    "SmearedOperatorMatvec",
    "QEILanczosResult",
    "qei_exact_lanczos",
    "first_order_infimum_slope",
    "standing_wave_modes",
    "free_dispersion",
    "harmonic_inversion",
    "peak_width",
    "DispersionResult",
    "interacting_dispersion_tebd",
    "excitation_gap_dmrg",
    "fit_dispersion_couplings",
    "cosine_coefficients",
    "keff_from_couplings",
    "keff_density_terms",
    "free_infimum_keff",
    "energy_pieces_dense",
    "heisenberg_smeared_operator_dense",
    "decompose_smeared_energy",
    "schroedinger_pieces",
]
