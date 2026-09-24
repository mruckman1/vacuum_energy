"""vacuum.hardware — public quantum devices as variational vacuum simulators
(PLAN.md Layer 7, L6 "Hardware as noisy field").

- :mod:`vacuum.hardware.vacuum_sim` — the Fock-truncated qubit encoding of
  the Layer-1 harmonic chain (Galerkin projection, d levels per site), its
  exact diagonalization, the convergence of the encoded vacuum to
  ``vacuum.core.ground_state_cov``, and the hardware-efficient VQE on Aer's
  statevector simulator with a numpy arbiter.
- :mod:`vacuum.hardware.protocols` — the two minimal protocols on the
  encoded vacuum, each as dense linear algebra AND as Aer circuits:
  (a) quantum energy teleportation (sign-of-x measurement, feed-forward
  displacement; for N = 2, d = 2 exactly Hotta's minimal model) with the
  standing audits, (b) minimal entanglement harvesting by two ancilla UDW
  qubits under a smooth switching, Trotterized, with two-qubit tomography
  and the Gaussian oscillator-detector reference of ``vacuum.detectors``.
- :mod:`vacuum.hardware.noise` — the sourced ibm_cairo calibration point
  (Ikeda 2023, Table III) reused from ``vacuum.qet.ikeda_qiskit``, the
  labelled gate-error / readout / T1 sweeps around it, the survival
  analysis (3-sigma signal loss), and the one-line real-device on-ramp.

qiskit / qiskit-aer are imported lazily inside the functions that need
them (the ``.[ibm]`` extra); every dense path is core-only.  Names are
resolved lazily here (PEP 562) so ``import vacuum.hardware`` stays cheap.
"""

import importlib

_MODULES = {
    "vacuum_sim": (
        "TruncatedChain", "truncated_chain", "site_operators", "embed_site",
        "hamiltonian_dense", "touched_hamiltonian_dense", "exact_ground_state",
        "covariance_from_state", "gaussian_reference", "encoding_convergence",
        "PauliTerm", "pauli_decompose", "hamiltonian_pauli_terms", "touched_pauli_terms",
        "pauli_terms_dense", "pauli_label", "commuting_groups", "hea_statevector",
        "hea_circuit", "VQEResult", "vqe_ground_state", "state_fidelity",
        "PAULIS", "site_pauli_terms", "pauli_term_dense", "hea_parameter_count",
        "hea_overlap_and_grad", "hea_energy_and_grad", "hea_state_jacobian", "hea_embed_layers",
        "FidelityResult", "fidelity_ceiling", "fidelity_depth_scan", "ansatz_energy",
    ),
    "protocols": (
        "qet_dense", "qet_optimize_theta", "qet_audits", "qet_circuits", "qet_exact_aer",
        "qet_sampled", "qet_from_counts", "HarvestSpec", "harvest_spec", "harvest_dense",
        "harvest_gaussian_reference", "harvest_circuit", "harvest_exact_aer",
        "harvest_sampled", "harvest_from_counts", "negativity_two_qubit",
        "linear_coupling_map", "run_on_backend",
        "IBM_BASIS_GATES", "alice_projectors", "alice_basis_change", "bob_displacement",
        "reduced_site_state", "trace_distance", "MeasurementGroup", "measurement_groups",
        "group_statistics", "expectation_from_groups", "readout_calibration_circuits",
        "calibration_from_counts", "apply_tensored_mitigation", "harvest_terms",
        "trotter_schedule", "partial_transpose_B", "harvest_coupling_map",
        "tomography_settings", "tomography_circuits", "rho_from_pauli_expectations",
    ),
    "noise": (
        "calibration_point", "scale_calibration", "noise_model_from_calibration",
        "sweep_grid", "survival_threshold", "survival_sweep", "shots_for_n_sigma",
        "device_run", "SWEEP_AXES", "SWEEP_FILE",
    ),
    "real_device": (
        "RECORD_FORMAT",
        "READOUT_REGISTER",
        "TWO_QUBIT_GATES",
        "FAMILIES",
        "FREE_TIER_SPEC",
        "WORKING_POINT",
        "BudgetError",
        "FreeTierJob",
        "Submission",
        "working_point",
        "build_free_tier_job",
        "batch_circuits",
        "assert_spec",
        "two_qubit_gate_name",
        "select_physical_qubits",
        "transpile_for_backend",
        "calibration_snapshot",
        "backend_info",
        "backend_kind",
        "qpu_time_estimate",
        "submission_info",
        "submit",
        "save_submission",
        "load_submission",
        "counts_from_result",
        "collect",
        "build_record",
        "record_sha256",
        "validate_record",
        "save_record",
        "load_record",
        "default_record_path",
        "exact_counts",
        "predict_on_backend",
        "analyze",
        "verdict",
        "run_batch",
        "fake_backend",
        "runtime_service",
        "least_busy_backend",
        "nz_shots",
        "environment_info",
            "analysis_path",
        "save_analysis",
        "load_analysis",
    ),
}
_NAME_TO_MODULE = {name: mod for mod, names in _MODULES.items() for name in names}

__all__ = sorted(_NAME_TO_MODULE) + sorted(_MODULES)


def __getattr__(name):
    if name in _MODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _NAME_TO_MODULE:
        return getattr(importlib.import_module(f".{_NAME_TO_MODULE[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
