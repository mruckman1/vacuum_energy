"""vacuum.inequalities — QEI stress-tester and work statistics (Layer 3).

**M3.4, the QEI core** (:mod:`vacuum.inequalities.qei`).  The pulled-back
sampling operator: local PSD energy density, Heisenberg pullback along a
worldline, quadrature accumulation into O_f, and the collapse of the
smeared-energy minimization to ONE Williamson problem with the extremal
state exhibited. 2d anchors target Flanagan's optimal massless bound
(PRD 56, 4922 (1997), Eq. (8)); the Fewster-Eveson constants
(PRD 58, 084010 (1998), Eqs. (37)/(40)) are carried for comparison and for
the 4d scan machinery (Srednicki radial lattice + ball smearing), which is
exposed here for the papers candidate.

**S8, the certified constant** (:mod:`vacuum.inequalities.qei_certified`).
The computer-assisted two-sided enclosure of the 4d Gaussian-sampled sharp
constant: an exhibited finitely-many-mode squeezed state evaluated in interval
arithmetic (the rigorous lower bound on C_4d/C_FE) against the Fewster-Eveson
theorem (the rigorous upper bound, = 1).  Derivation and error budget:
``papers/qei-2d-sharp-constant/draft/proof_4d_gaussian_constant.md``.

**M3.5, work statistics** (:mod:`vacuum.inequalities.work_stats`).
Two-point-measurement work distributions for Gaussian quenches and drives:
the closed-form characteristic function for quadratic Hamiltonians (primary
route), the dense Fock-space TPM with its occupation-cutoff tail test
(validation route, N <= 3), the Jarzynski and Crooks defects as standing
audits, and :func:`counting_statistics`, whose output is the payload
:meth:`vacuum.audits.EnergyLedger.refine_work` consumes as the
beyond-the-mean ledger refinement (contract:
:data:`vacuum.audits.ledger.WORK_STATS_REQUIRED_KEYS`).

Conventions: hbar = 1, V_vac = I/2, nu >= 1/2, block ordering
R = (x_1..x_N, p_1..p_N), nats, beta = 1/T (docs/API.md).
"""

from .qei import (
    FEWSTER_EVESON_CONSTANT_2D,
    FORD_ROMAN_4D_LORENTZIAN,
    SHARP_CONSTANT_2D,
    FHandle,
    ModeSpaceResult,
    OptimizeSamplingResult,
    QEIResult,
    QEIUnphysicalError,
    QEIViolationError,
    SamplingFamily,
    SamplingOp,
    assert_physical,
    chain_energy_density_form,
    chain_worldline_minimize,
    cos2_f,
    cos2_family,
    embed_extremal,
    fprime_sq_integral,
    gaussian_f,
    gaussian_family,
    min_symplectic_margin,
    mode_space_extremal_state,
    mode_space_minimize,
    naive_volume,
    optimize_sampling,
    qei_audit,
    qei_bound_2d,
    qei_minimize,
    qei_minimize_mp,
    qei_ratio,
    radial_ball_qei,
    radial_ball_weights,
    radial_energy_density_form,
    radial_sampling_operator,
    radial_sector_minimize,
    sampling_operator,
    shell_volume,
    smeared_energy,
    squeezed_pocket,
)
from .work_stats import (
    COUNTING_STATISTICS_KEYS,
    DenseTPM,
    counting_statistics,
    crooks_max_defect,
    deffner_lutz_cf,
    deffner_lutz_cf_sudden,
    deffner_lutz_final_n_moments,
    deffner_lutz_q_star,
    delta_free_energy,
    dense_hamiltonian,
    dense_jarzynski_average,
    dense_tpm,
    dense_work_cf,
    dense_work_cumulants,
    dense_work_distribution,
    dense_work_moments,
    dense_work_quantile,
    gaussian_crooks_defect,
    gaussian_free_energy,
    gaussian_jarzynski_average,
    gaussian_work_cf,
    gaussian_work_cf_at,
    gaussian_work_cumulants,
    gaussian_work_moments,
    jarzynski_defect,
    log_partition_gaussian,
    mode_frequencies,
    protocol_symplectic,
    quadratic_form_matrix,
    reference_frequency,
    reverse_segments,
)

from .qei import (
    FEWSTER_EVESON_CONSTANT_4D,
    lorentzian_f,
    sqrt_lorentzian_f,
    bump_f,
    gaussian_poly_f,
    fhat_of,
    fsecond_sq_integral,
    momentum_grid,
    MomentumResult,
    worldline_momentum_minimize,
    ball_momentum_minimize,
    rho2_coefficient,
    momentum_qei_ratio,
    optimize_sampling_momentum,
)

from .ffr_moments import (
    FFR2012_Y_INF,
    FFR2012_Y_INF_ERR,
    FFR2012_TABLE_I,
    FFR2012_TABLE_II,
    chain_integrals_exact,
    run_structure_cumulants,
    moments_from_cumulants,
    ffr_moment_sequence,
    jacobi_recurrence,
    stieltjes_edge,
    stieltjes_sequence,
    ffr_fit_y_inf,
    carleman_diagnostics,
    ffr_identity_A4,
    kernel_chain_integrals,
    point_mass_perturbation,
    FFR2012_TABLE_III,
    accelerated_sequence,
)

from .qei_certified import (
    CERTIFIED_PINNED_RATIO,
    FEWSTER_EVESON_RATIO_UPPER_BOUND,
    CertifiedEnclosure,
    KernelElements,
    certified_enclosure,
    certified_lower_bound,
    phi_gram,
    phi_kernel_elements,
    ratio_from_energy,
    trial_state_covariance,
)

__all__ = [
    "SHARP_CONSTANT_2D",
    "FEWSTER_EVESON_CONSTANT_2D",
    "FORD_ROMAN_4D_LORENTZIAN",
    "FHandle",
    "SamplingFamily",
    "SamplingOp",
    "QEIResult",
    "OptimizeSamplingResult",
    "QEIUnphysicalError",
    "QEIViolationError",
    "gaussian_f",
    "cos2_f",
    "gaussian_family",
    "cos2_family",
    "fprime_sq_integral",
    "qei_bound_2d",
    "chain_energy_density_form",
    "sampling_operator",
    "qei_minimize",
    "smeared_energy",
    "qei_ratio",
    "qei_audit",
    "embed_extremal",
    "squeezed_pocket",
    "min_symplectic_margin",
    "assert_physical",
    "optimize_sampling",
    "radial_ball_weights",
    "radial_energy_density_form",
    "radial_sampling_operator",
    "radial_ball_qei",
    # the cancellation-free sector minimizer of the 4d scan (leap L1)
    "ModeSpaceResult",
    "mode_space_minimize",
    "radial_sector_minimize",
    "chain_worldline_minimize",
    "mode_space_extremal_state",
    "qei_minimize_mp",
    "shell_volume",
    "naive_volume",
    # --- M3.5 work statistics ---------------------------------------------
    # plumbing and exact Gaussian free energies
    "quadratic_form_matrix",
    "mode_frequencies",
    "reference_frequency",
    "log_partition_gaussian",
    "gaussian_free_energy",
    "delta_free_energy",
    "protocol_symplectic",
    "reverse_segments",
    # primary route: the closed-form characteristic function
    "gaussian_work_cf",
    "gaussian_work_cf_at",
    "gaussian_work_moments",
    "gaussian_work_cumulants",
    "gaussian_jarzynski_average",
    "gaussian_crooks_defect",
    # validation route: dense Fock-space TPM (N <= 3, cutoff tail test)
    "DenseTPM",
    "dense_hamiltonian",
    "dense_tpm",
    "dense_work_cf",
    "dense_work_moments",
    "dense_work_cumulants",
    "dense_work_distribution",
    "dense_work_quantile",
    "dense_jarzynski_average",
    "crooks_max_defect",
    # standing audits and the EnergyLedger.refine_work payload
    "jarzynski_defect",
    "counting_statistics",
    "COUNTING_STATISTICS_KEYS",
    # Deffner-Lutz, PRE 77, 021128 (2008) single-mode anchor
    "deffner_lutz_cf",
    "deffner_lutz_cf_sudden",
    "deffner_lutz_q_star",
    "deffner_lutz_final_n_moments",
    "FEWSTER_EVESON_CONSTANT_4D",
    "lorentzian_f",
    "sqrt_lorentzian_f",
    "bump_f",
    "gaussian_poly_f",
    "fhat_of",
    "fsecond_sq_integral",
    "momentum_grid",
    "MomentumResult",
    "worldline_momentum_minimize",
    "ball_momentum_minimize",
    "rho2_coefficient",
    "momentum_qei_ratio",
    "optimize_sampling_momentum",
    "FFR2012_Y_INF",
    "FFR2012_Y_INF_ERR",
    "FFR2012_TABLE_I",
    "FFR2012_TABLE_II",
    "chain_integrals_exact",
    "run_structure_cumulants",
    "moments_from_cumulants",
    "ffr_moment_sequence",
    "accelerated_sequence",
    "FFR2012_TABLE_III",
    "jacobi_recurrence",
    "stieltjes_edge",
    "stieltjes_sequence",
    "ffr_fit_y_inf",
    "carleman_diagnostics",
    "ffr_identity_A4",
    "kernel_chain_integrals",
    "point_mass_perturbation",
    # S8: the computer-assisted certificate for C_4d(gaussian)/C_FE
    "CERTIFIED_PINNED_RATIO",
    "FEWSTER_EVESON_RATIO_UPPER_BOUND",
    "CertifiedEnclosure",
    "KernelElements",
    "certified_enclosure",
    "certified_lower_bound",
    "phi_gram",
    "phi_kernel_elements",
    "ratio_from_energy",
    "trial_state_covariance",
]
