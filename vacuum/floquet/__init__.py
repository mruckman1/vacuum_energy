"""vacuum.floquet — time-modulated vacua (Layer 5).

Periodically driven quadratic lattice fields on the validated
``vacuum.core`` stack: drives (:mod:`.drives`), one-period maps, stability
charts and momentum gaps (:mod:`.monodromy`), the Mathieu reference
(:mod:`.mathieu`), photon-pair / squeezing spectra and (k, -k) negativity
(:mod:`.spectra`), audited runs with loss, pulse trains and amplification
bounds (:mod:`.runs`), the Wilson 2011 dynamical-Casimir anchor
(:mod:`.dce`) and the pump-enhanced-harvesting composite with Layer 2
(:mod:`.composite`).

Conventions per docs/API.md: hbar = 1, V_vac = I/2, nu >= 1/2, block
quadrature ordering, negativities in nats.  Amplification runs are the one
place energy legitimately enters: every ledger charges it to the drive, and
passivity of the *undriven* vacuum is a standing audit on every run.
"""

from .drives import KINDS, PROFILES, Drive, dce_chain_K, modulated_K, profile_function
from .monodromy import (
    CF4_ALPHA,
    CF4_NODES,
    Band,
    FloquetMap,
    StabilityChart,
    batched_monodromy,
    cf4_pieces,
    floquet_exponent,
    floquet_map,
    mode_monodromy,
    mode_propagator_coefficients,
    momentum_gaps,
    multiplier,
    propagator,
    stability_chart,
    symplectic_defect,
)
from .mathieu import (
    mathieu_characteristic,
    mathieu_is_stable,
    mathieu_parameters,
    mathieu_tongue,
    mathieu_tongue_edges,
    resonance_frequency,
)
from .spectra import (
    FLOOR_TOL,
    EntanglementGrowth,
    TravelingWaveBasis,
    entanglement_growth,
    mode_occupations,
    normal_mode_symplectic,
    normal_modes,
    pair_negativity,
    pair_spectrum,
    power_iterate,
    squeezing_from_occupation,
    squeezing_spectrum,
    to_traveling_waves,
    traveling_wave_basis,
)
from .runs import (
    AmplificationScan,
    FloquetRun,
    PulseTrainResult,
    amplification_bound_scan,
    bang_bang_bound,
    envelope_cos2,
    field_loss,
    floquet_run,
    floquet_steps,
    loss_per_period,
    loss_threshold,
    lossy_floquet,
    matched_bath_cov,
    mathieu_small_depth_rate,
    max_gain_over_frequency,
    pulse_train,
    quality_factor_from_eta,
    run_floquet_protocol,
)
from .dce import (
    C_LIGHT_SI,
    DCESpectrum,
    WilsonRegime,
    dce_spectrum,
    parabola_fit,
    wilson_flux_density,
    wilson_regime,
    wilson_total_flux,
)
from .composite import PumpedHarvest, cos2_window, detector_occupation, pumped_harvest

from .bounds import (
    per_cycle_bound,
    prufer_rhs,
    RateOptimum,
    rate_optimum,
    rate_optimum_control,
    bang_bang_control,
    piecewise_monodromy,
    ln_multiplier,
    prufer_winding,
    bound_ratio,
    CONTROL_FAMILIES,
    random_controls,
    optimize_control,
    adversarial_search,
    operator_norm_depth,
    mass_mode_depths,
    multimode_check,
)
from .bounds_multimode import (
    frequency_weighted_norm,
    multimode_bound_proved,
    normal_mode_system,
    pattern_monodromy,
    rwa_growth_rate,
    sum_frequency_first_order,
    multimode_ratios,
    multimode_attack,
)
from .composite import (
    HarvestSplit,
    floquet_window_rows,
    pair_integrals_on_grid,
    pumped_harvest_split,
)

__all__ = [
    # drives
    "Drive", "modulated_K", "dce_chain_K", "profile_function", "PROFILES", "KINDS",
    # monodromy
    "propagator", "mode_propagator_coefficients", "FloquetMap", "floquet_map",
    "multiplier", "floquet_exponent", "symplectic_defect", "cf4_pieces",
    "mode_monodromy", "batched_monodromy", "StabilityChart", "stability_chart",
    "Band", "momentum_gaps", "CF4_ALPHA", "CF4_NODES",
    # mathieu
    "mathieu_parameters", "mathieu_characteristic", "mathieu_tongue",
    "mathieu_is_stable", "mathieu_tongue_edges", "resonance_frequency",
    # spectra
    "normal_modes", "normal_mode_symplectic", "mode_occupations", "power_iterate",
    "pair_spectrum", "TravelingWaveBasis", "traveling_wave_basis",
    "to_traveling_waves", "squeezing_spectrum", "squeezing_from_occupation",
    "pair_negativity", "EntanglementGrowth", "entanglement_growth", "FLOOR_TOL",
    # runs
    "FloquetRun", "floquet_run", "lossy_floquet", "matched_bath_cov", "field_loss", "floquet_steps",
    "run_floquet_protocol", "PulseTrainResult", "pulse_train", "envelope_cos2",
    "loss_per_period", "quality_factor_from_eta", "loss_threshold",
    "bang_bang_bound", "mathieu_small_depth_rate", "AmplificationScan",
    "amplification_bound_scan", "max_gain_over_frequency",
    # dce
    "wilson_flux_density", "wilson_total_flux", "wilson_regime", "WilsonRegime",
    "DCESpectrum", "dce_spectrum", "parabola_fit", "C_LIGHT_SI",
    # composite
    "PumpedHarvest", "pumped_harvest", "cos2_window", "detector_occupation",
    "per_cycle_bound",
    "prufer_rhs",
    "RateOptimum",
    "rate_optimum",
    "rate_optimum_control",
    "bang_bang_control",
    "piecewise_monodromy",
    "ln_multiplier",
    "prufer_winding",
    "bound_ratio",
    "CONTROL_FAMILIES",
    "random_controls",
    "optimize_control",
    "adversarial_search",
    "operator_norm_depth",
    "mass_mode_depths",
    "multimode_check",
    # bounds_multimode
    "frequency_weighted_norm", "multimode_bound_proved", "normal_mode_system", "pattern_monodromy",
    "rwa_growth_rate", "sum_frequency_first_order", "multimode_ratios", "multimode_attack",
    "HarvestSplit",
    "floquet_window_rows",
    "pair_integrals_on_grid",
    "pumped_harvest_split",
]
