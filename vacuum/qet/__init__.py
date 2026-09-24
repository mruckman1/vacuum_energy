"""vacuum.qet — quantum energy teleportation (Layer 3).

M3.1: Hotta's minimal two-qubit QET model, exact dim-4 linear algebra
(:mod:`vacuum.qet.hotta`), transcribed from arXiv:1101.3954 Sec. 3.
M3.2: QET on transverse-field Ising chains — exact diagonalization in
:mod:`vacuum.qet.chains` (sparse eigsh, L <= 14), and the MPS/DMRG mirror
in :mod:`vacuum.qet.dmrg` behind the ``.[tensor]`` extra.  The DMRG names
are re-exported here *lazily* (PEP 562 ``__getattr__``): ``vacuum.qet.dmrg``
and ``from vacuum.qet import tfim_ground_mps`` both work, and neither pulls
TeNPy in — the extra is imported only inside :func:`vacuum.qet.dmrg._tenpy`,
when a DMRG function is actually called, so a default install imports this
package and runs the whole ED rung unchanged.
Layer 7 L3: the resource theory of vacuum manipulation in
:mod:`vacuum.qet.resource` (free operations, the QET potential and negativity
monotones, the smooth-entropy one-shot bound; core-only, cvxpy only inside the
SDP entropies) is re-exported eagerly.
M3.3: strong local passivity in :mod:`vacuum.qet.slp` — the closed-form
Alhambra-Styliaris-Rodriguez-Briones et al. certificate (the sharp form of
the Frey-Funo-Hotta single-qubit-region criterion) plus a Kraus search, both
core-only; the Choi semidefinite program behind the ``.[sdp]`` extra (cvxpy
loads lazily inside the SDP functions, so this package import never requires
it); and the LOCC layer that reruns the M3.1 protocol on the same states.
Ikeda's IBM-hardware run of the minimal model (PRApplied 20, 024051 (2023))
is reproduced exactly in :mod:`vacuum.qet.ikeda` (dense gate-list
simulation, parameters from ``experiments/ibm_qet/``) and, behind the
``.[ibm]`` extra, on qiskit/Aer in :mod:`vacuum.qet.ikeda_qiskit` — the
Layer 7 "hardware as noisy field" on-ramp — whose names are re-exported
lazily here exactly like the DMRG ones (qiskit is imported only inside that
module's functions).
"""

import importlib

from .chains import (
    apply_site_gate,
    bob_rotation_gate,
    chain_distance_sweep,
    energy_profile,
    evolve_dip_profiles,
    local_hamiltonian,
    measure_site_x,
    optimize_rotation,
    premeasurement_reduced_states,
    reduced_density,
    rotation_optimum_closed,
    run_chain_qet,
    tfim_ground_state,
    tfim_hamiltonian,
    tfim_local_energy_ops,
)
from .hotta import (
    bob_reduced_state,
    bob_rotation,
    concurrence,
    e_a_closed,
    e_b_closed,
    e_b_optimal,
    ground_state,
    hamiltonian,
    hamiltonian_terms,
    hotta_sweep,
    measurement_projector,
    optimize_e_b,
    run_protocol,
    theta_optimal,
    trace_distance,
)
from .ikeda import (
    BITSTRINGS,
    FORMS,
    IKEDA_FIG2_FILE,
    IKEDA_TABLE2_FILE,
    OBSERVABLES,
    PAPER_PAIRS,
    bob_angle,
    counts_to_probabilities,
    e0_closed,
    e1_closed,
    expectation_from_counts,
    gate_matrix,
    h1_closed,
    ikeda_circuit_spec,
    ikeda_ibm,
    ikeda_no_signaling,
    ikeda_params,
    ikeda_sweep,
    prep_angle,
    simulate_spec,
    spec_prefix,
    terminal_measure_index,
    v_closed,
)
from .resource import (
    apply_instrument,
    bob_correlation_matrix,
    bob_energy_out,
    bond_operators,
    chain_potential_sparse,
    chain_region_ledger,
    chain_resource_ledger,
    classical_min_max_entropy,
    conditional_min_entropy_cq,
    daemonic_gain,
    energy_signalling_defect,
    free_class_extremality,
    free_projectors,
    gaussian_monotonicity_audit,
    gaussian_qet_potential,
    guessing_probability_cq,
    hotta_resource_ledger,
    ky_fan_floor,
    local_extractable,
    max_entropy_direct,
    max_entropy_sdp,
    min_entropy,
    min_entropy_sdp,
    monotonicity_audit,
    negativity,
    nonfree_energy_nonsignalling_instrument,
    one_shot_bound,
    one_shot_bound_entropic,
    one_shot_bound_marginal,
    optimal_local_extraction,
    qet_potential,
    qet_potential_operator,
    qet_surplus,
    random_free_instrument,
    random_instrument,
    random_local_unitary,
    random_state,
    search_free_instrument,
    single_site_extractable,
    smooth_min_entropy,
    split_hamiltonian,
    split_hamiltonian_symmetric,
    star_hamiltonian,
    surplus_counterexample,
    two_way_monotonicity_audit,
    two_way_potential,
    unitary_energy_spread,
)
from .slp import (
    apply_local_channel,
    choi_cost_matrix,
    choi_to_kraus,
    eigenmixture,
    embed_region_operator,
    extractable_energy_kraus,
    extractable_energy_sdp,
    ffh_pair_eigenenergies,
    ffh_pair_eigenmixture,
    ffh_pair_gibbs_populations,
    ffh_pair_hamiltonian,
    ffh_pair_local_energy,
    gibbs_state,
    locc_extractable_energy,
    reduced_commutator,
    slp_certificate,
    slp_critical_temperature,
    slp_map,
    state_energy,
    xxx_pair_eigenbasis,
    xxx_pair_eigenmixture,
    xxx_pair_hamiltonian,
)

__all__ = [
    "hamiltonian",
    "hamiltonian_terms",
    "ground_state",
    "measurement_projector",
    "bob_rotation",
    "e_a_closed",
    "e_b_closed",
    "theta_optimal",
    "e_b_optimal",
    "run_protocol",
    "optimize_e_b",
    "bob_reduced_state",
    "trace_distance",
    "concurrence",
    "hotta_sweep",
    # M3.2 — QET on chains (ED; the DMRG mirror lives in vacuum.qet.dmrg)
    "tfim_hamiltonian",
    "tfim_local_energy_ops",
    "tfim_ground_state",
    "energy_profile",
    "apply_site_gate",
    "measure_site_x",
    "bob_rotation_gate",
    "local_hamiltonian",
    "optimize_rotation",
    "rotation_optimum_closed",
    "run_chain_qet",
    "chain_distance_sweep",
    "evolve_dip_profiles",
    "reduced_density",
    "premeasurement_reduced_states",
    # M3.3 — strong local passivity (the SDP rung needs the .[sdp] extra)
    "state_energy",
    "gibbs_state",
    "eigenmixture",
    "choi_cost_matrix",
    "choi_to_kraus",
    "embed_region_operator",
    "apply_local_channel",
    "reduced_commutator",
    "slp_certificate",
    "extractable_energy_kraus",
    "extractable_energy_sdp",
    "locc_extractable_energy",
    "slp_map",
    "slp_critical_temperature",
    "ffh_pair_hamiltonian",
    "ffh_pair_eigenenergies",
    "ffh_pair_eigenmixture",
    "ffh_pair_gibbs_populations",
    "ffh_pair_local_energy",
    "xxx_pair_hamiltonian",
    "xxx_pair_eigenbasis",
    "xxx_pair_eigenmixture",
    # Ikeda (2023) IBM protocol — dense path (core-only)
    "IKEDA_FIG2_FILE",
    "IKEDA_TABLE2_FILE",
    "OBSERVABLES",
    "FORMS",
    "BITSTRINGS",
    "PAPER_PAIRS",
    "ikeda_params",
    "prep_angle",
    "bob_angle",
    "e0_closed",
    "h1_closed",
    "v_closed",
    "e1_closed",
    "ikeda_circuit_spec",
    "terminal_measure_index",
    "spec_prefix",
    "gate_matrix",
    "simulate_spec",
    "counts_to_probabilities",
    "expectation_from_counts",
    "ikeda_ibm",
    "ikeda_no_signaling",
    "ikeda_sweep",
    # Layer 7 L3 — resource theory of vacuum manipulation (vacuum.qet.resource;
    # core-only, the SDP entropies load cvxpy through slp at call time)
    "split_hamiltonian",
    "bond_operators",
    "free_projectors",
    "energy_signalling_defect",
    "random_free_instrument",
    "random_instrument",
    "random_local_unitary",
    "apply_instrument",
    "bob_energy_out",
    "qet_potential",
    "local_extractable",
    "qet_surplus",
    "surplus_counterexample",
    "negativity",
    "random_state",
    "monotonicity_audit",
    "ky_fan_floor",
    "min_entropy",
    "smooth_min_entropy",
    "conditional_min_entropy_cq",
    "min_entropy_sdp",
    "max_entropy_sdp",
    "max_entropy_direct",
    "classical_min_max_entropy",
    "one_shot_bound",
    "one_shot_bound_marginal",
    "hotta_resource_ledger",
    "chain_resource_ledger",
    "gaussian_qet_potential",
    "gaussian_monotonicity_audit",
    # L3 close-out: the symmetric two-way theory, the free-class characterization,
    # the exact single-site bound with its entropic form, and the L = 12 route
    "free_class_extremality",
    "star_hamiltonian",
    "nonfree_energy_nonsignalling_instrument",
    "search_free_instrument",
    "split_hamiltonian_symmetric",
    "qet_potential_operator",
    "optimal_local_extraction",
    "two_way_potential",
    "two_way_monotonicity_audit",
    "bob_correlation_matrix",
    "single_site_extractable",
    "unitary_energy_spread",
    "guessing_probability_cq",
    "one_shot_bound_entropic",
    "daemonic_gain",
    "chain_potential_sparse",
    "chain_region_ledger",
    # Ikeda (2023) on qiskit/Aer (`.[ibm]` extra), resolved lazily below
    "build_circuit",
    "transpile_native",
    "exact_statevector",
    "exact_density_matrix",
    "readout_assignment_matrix",
    "apply_readout",
    "noise_model_from_params",
    "noisy_probabilities",
    "calibration_matrix",
    "mitigate_probabilities",
    "run_on_backend",
    "ikeda_ibm_aer",
    "IBM_BASIS_GATES",
    "COUPLING_MAP",
    "READOUT_MODELS",
    "to_dense_vector",
    "to_dense_matrix",
    "probabilities_to_dict",
    # Layer 7 L6 — QET on the qubit-encoded field (vacuum.hardware), lazily
    "encoded_field_qet_dense",
    "encoded_field_qet_optimize_theta",
    "encoded_field_qet_audits",
    "encoded_field_qet_exact_aer",
    "encoded_field_qet_sampled",
    # M3.2 — the DMRG mirror (`.[tensor]` extra), resolved lazily below
    "tfim_ground_mps",
    "mps_energy_profile",
    "mps_measure_site_x",
    "mps_apply_rotation",
    "mps_local_energy",
    "mps_optimize_rotation",
    "mps_rotation_optimum_closed",
    "run_chain_qet_mps",
]

#: Names served from :mod:`vacuum.qet.dmrg` on first attribute access.  Kept
#: out of the eager import list purely for import cost — ``dmrg`` itself has
#: no extra-gated top-level import (the TeNPy import is inside ``_tenpy``),
#: so touching one of these without the extra installed raises the module's
#: own guided ImportError at *call* time, not here.
_DMRG_NAMES = frozenset({
    "tfim_ground_mps",
    "mps_energy_profile",
    "mps_measure_site_x",
    "mps_apply_rotation",
    "mps_local_energy",
    "mps_optimize_rotation",
    "mps_rotation_optimum_closed",
    "run_chain_qet_mps",
})

#: Names served from :mod:`vacuum.qet.ikeda_qiskit` on first attribute
#: access — the ``.[ibm]`` extra, handled exactly like the DMRG rung: the
#: module itself imports qiskit only inside ``_qiskit()`` at call time, so a
#: default install resolves these names and gets a guided ImportError only
#: when one is actually called.
_IBM_NAMES = frozenset({
    "build_circuit",
    "transpile_native",
    "exact_statevector",
    "exact_density_matrix",
    "readout_assignment_matrix",
    "apply_readout",
    "noise_model_from_params",
    "noisy_probabilities",
    "calibration_matrix",
    "mitigate_probabilities",
    "run_on_backend",
    "ikeda_ibm_aer",
    "IBM_BASIS_GATES",
    "COUPLING_MAP",
    "READOUT_MODELS",
    "to_dense_vector",
    "to_dense_matrix",
    "probabilities_to_dict",
})


#: Layer 7 L6 (vacuum.hardware.protocols): the QET protocol on the
#: qubit-encoded harmonic chain, served under an ``encoded_field_`` prefix
#: so nothing here shadows the minimal-model names; the dense functions are
#: core-only, the Aer ones import qiskit only at call time.
_HARDWARE_NAMES = {
    "encoded_field_qet_dense": "qet_dense",
    "encoded_field_qet_optimize_theta": "qet_optimize_theta",
    "encoded_field_qet_audits": "qet_audits",
    "encoded_field_qet_exact_aer": "qet_exact_aer",
    "encoded_field_qet_sampled": "qet_sampled",
}


def __getattr__(name):
    """PEP 562: resolve the DMRG, qiskit and hardware rungs' public names on demand.

    ``importlib.import_module`` rather than ``from . import ...``: the latter
    probes the package attribute of the submodule's name through this very
    hook, which recurses whenever a served name coincides with a module name.
    """
    if name in _DMRG_NAMES:
        return getattr(importlib.import_module(".dmrg", __name__), name)
    if name in _IBM_NAMES:
        return getattr(importlib.import_module(".ikeda_qiskit", __name__), name)
    if name in _HARDWARE_NAMES:
        return getattr(importlib.import_module("vacuum.hardware.protocols"), _HARDWARE_NAMES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _DMRG_NAMES | _IBM_NAMES | set(_HARDWARE_NAMES))
