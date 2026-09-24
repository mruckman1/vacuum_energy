# Draft outline — Which minimal vacuum protocol survives a public quantum device

**Status: SKELETON.** No number below has been moved out of `../data/`.
Per `papers/README.md`, nothing enters a draft until the full known-results
suite is green on the producing build (recorded in `../MANIFEST.md`).
Placeholders are written `[[data:<file>:<json path>]]` and are filled by
reading that file, never by retyping a number from a log.

---

## 0. The gating statement this draft has to make first

The map's *shape* rests on one sourced device calibration — the ibm_cairo
column of Table III of Ikeda (2023), read through
`vacuum.qet.ikeda_qiskit.noise_model_from_params`; its *axes* do not. The
multipliers on gate error, readout error and T1 are SYNTHETIC grids declared
in `experiments/ibm_qet/ikeda_2023_fig2_variant-noise-sweep.json`, the
tiling of a two-qubit calibration over a 3–5-qubit register is a modelling
choice, the shot budgets are choices. One run on hardware exists (§8) and it
is the only device claim in this draft; the map itself stays a model claim. The
claim is therefore about *this noise model of that device*, not about the
device; what makes it worth writing is that both the sourced point and the
IBM free-tier budget (10 min per 28-day window; QPU-time estimate formula)
are transcribed, so the "what would a free run test" statement is concrete.

## 1. Setup — the encoded vacuum

- Layer-1 chain H = ½Σ[p² + m²x² + (x_{i+1}−x_i)²], N = 3, m = 1,
  Dirichlet; Galerkin Fock truncation to d = 2^q levels per site in the
  local-oscillator basis ω_i = √K_ii (Klco–Savage 2019 oscillator basis;
  Macridin et al. 2018 Fock cutoff). State the ordering convention (site 0
  least significant = qiskit little-endian; no permutation anywhere).
- Convergence to `vacuum.core.ground_state_cov(K)`: table from
  `[[data:s1_encoding_convergence.json]]` — energy error is a Rayleigh–Ritz
  upper bound decreasing monotonically; covariance error at d = 2, 4, 8
  for N = 2, 3, 4 and m = 1, 0.3, 0.
- The N = 2, d = 2 encoding *is* Hotta's minimal QET model
  (h = ω/2, k = 1/(4ω)): closed-form residuals
  `[[data:s3a_qet_dense.json:[N=2,d=2].hotta_E_A_residual / hotta_E_B_residual]]`.
  Consequence to state plainly: the lattice field sits at k/h = 1/(2ω²) ≤ 1/4,
  the weak-coupling corner of the minimal model, so E_B/E_A is
  `[[data:s3a_qet_dense.json:[N=3,d=2,A=0,B=1].E_B_over_E_A]]` (≈1 %) against
  Ikeda's 16 % at h = k = 1 — the encoded vacuum is a *harder* QET medium
  than the paper's tuned pair.

## 2. Variational preparation

- Hardware-efficient R_Y / linear-CNOT ansatz (Kandala et al. 2017), Aer
  statevector evaluator, parameter-shift gradients, L-BFGS-B, seeded restarts;
  numpy arbiter agreement `[[data:s2_vqe.json:*.aer_vs_numpy]]`.
- Table from `[[data:s2_vqe.json]]`: energy error vs ED of the same
  truncated H, infidelity, covariance error vs ED, covariance error vs the
  Gaussian vacuum (= the encoding's own error), for (N, d) up to 8 qubits.
  Say which instances needed 3 entangling layers and which plateaued.

## 3. Protocol (a): QET on the encoded field

- Alice: sign of x_A (W_A^† basis change, top qubit = sign); Bob: conditional
  displacement exp(−iμθp_B) (Hotta 2008 PRD; Nambu–Hotta 2010 harmonic chain);
  deferred (Ikeda Fig. 1(C)) and feed-forward circuits; the three families
  plus the inverted-bit control.
- Dense ledger with audits (closure, E_A ≥ E_B, no-signaling, cut channel):
  `[[data:s3a_qet_dense.json]]`; Aer noiseless vs dense:
  `[[data:s3b_qet_working_point.json:ideal_aer_E_B_vs_dense]]`.
- The three signals and why they differ under noise (Bob's controlled
  rotations = 4 extra CNOTs; readout scaling of 1- vs 2-qubit terms;
  mitigation): sourced-point values
  `[[data:s3b_qet_working_point.json:sourced_point.{none,tensored}]]`.
- Survival thresholds per signal and axis:
  `[[data:s3c_qet_survival.json:thresholds]]`; the "all" axis is the
  headline for E_B_ctrl / E_B_raw (how much *uniformly* better the device
  must be), the gate/readout axes for E_B_info.

## 4. Protocol (b): minimal harvesting

- Two ancilla UDW qubits, cos² switching, Strang product formula over
  commuting groups; the circuit's dense mirror (1e-15) and its Trotter error
  against RK4 `[[data:s4a_harvest_trotter.json]]`; causal-contact flag (the
  detectors are NOT spacelike; say what that means with the M2.5 split as
  the tool that would quantify it).
- Gaussian limit `[[data:s4b_harvest_gaussian_limit.json]]`: at weak
  coupling the qubit-detector E_N matches the oscillator-detector stack
  (MAP-1) to ~1 % at d = 4/8 and ~5 % at d = 2; at λ = 1 the two-level
  saturation separates them — state this as the limit the encoding allows.
- Sourced point: dead (λ_min > 0) at 4 and 6 steps, with and without
  mitigation `[[data:s4c_harvest_working_point.json:sourced_point]]`;
  thresholds `[[data:s4d_harvest_survival.json:thresholds]]`.

## 5. The survival map

- Figure `survival_map.svg` (z vs ν, four axes, both protocols, sourced
  point marked); grid rows in `s3c`/`s4d`.
- Sampled cross-checks (Aer with shots vs the shot-free σ):
  `[[data:s5_sampled_crosschecks.json]]` pulls.

## 6. What a free-tier run would need and what number it would test

- From `[[data:s6_free_tier.json]]`: qubits, circuits, native depth and
  CNOT count, shots per circuit for 3σ at the sourced noise, the QPU time
  from IBM's transcribed formula, whether it fits 10 min, and the predicted
  value ± error of E_B_info. State the one-line on-ramp
  (`vacuum.hardware.noise.device_run` → `qet_from_counts`).
- The honest negative: E_B_ctrl and E_B_raw are not testable on this
  calibration; harvesting is not testable at 52 CNOTs.

## 7. Limitations (all inherited, none absorbed)

- Table III calibrates one pair on one day; tiling; no crosstalk, no
  leakage, no coherent errors; relaxation kept at the sourced T1/T2 when the
  gate axis is scaled (floor); tensored (not full) readout mitigation.
- d = 2 truncation errors are 5 % in the covariance (d = 4 rows say what
  changes); N = 3 is not a field; the harvesting detectors are in causal
  contact.

## 8. Real-device run — the excursion, done: prediction held on `ibm_fez`

**Status: RUN (one job, one device, one day).** Record
`[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.json]]` (calibration snapshot at submission, sha256), analysis
`[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json]]` (the audited numbers at n_sigma = 5, re-derived from the
record by `scripts/run_ibm_free_tier.py --analyze --save-analysis`).
This is the program's one hardware excursion (PLAN.md L6): a claim about
a real device — narrow by construction — with the calibration snapshot
as its provenance. The dry-run records under `data/real_device/dry_run/`
are simulations and are not cited.

- **What ran.** Exactly the batch of §6 — 11 circuits on 3 qubits
  (9 measurement + 2 calibration), native depth
  `[[data:s6_free_tier.json:qet_information_signal.depth_native]]`,
  `[[data:s6_free_tier.json:qet_information_signal.cx_count_max]]` CNOTs —
  transpiled to `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.json:backend.two_qubit_gate]]` on physical qubits
  `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.json:backend.physical_qubits]]` of `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.json:backend.name]]`
  (`[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.json:backend.num_qubits]]` qubits; calibration
  `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.json:backend.calibration.last_update_date]]`, T1/T2, readout, sx
  and CZ errors of the path in `backend.calibration`), ISA depth 40 / 38
  (protocol / wrong), 8 two-qubit gates, no routing;
  `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.json:job.shots]]` shots per circuit through
  `qiskit_ibm_runtime.SamplerV2` with no server-side mitigation, twirling
  or dynamical decoupling (`job.sampler_options`); job
  `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.json:job.job_id]]`.

- **Predicted before the run (§6, `s6`, produced 2026-09-03 on the sourced
  ibm_cairo calibration):** E_B_info =
  `[[data:s6_free_tier.json:qet_information_signal.predicted_value_at_sourced_noise]]`
  ± `[[data:s6_free_tier.json:qet_information_signal.predicted_std_error]]`
  (10⁵ shots), ideal
  `[[data:s6_free_tier.json:qet_information_signal.ideal_value]]`;
  teleportation proper lost:
  E_B_ctrl `[[data:s6_free_tier.json:qet_teleportation_signal.predicted_E_B_ctrl_at_sourced_noise]]`,
  E_B_raw `[[data:s6_free_tier.json:qet_teleportation_signal.predicted_E_B_raw_at_sourced_noise]]`.
  Say plainly that the prediction was made on a 2023 Falcon's calibration
  (CNOT 7.3e-3, readout ~8e-3) and tested on a 2026 Heron (CZ ~1.9e-3,
  readout ~3.5e-3): the information signal is fixed by the circuit's
  structure, not the chip.

- **Measured.** E_B_info = `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:E_B_info]]` ±
  `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:E_B_info_std_error]]`, z = `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:z.E_B_info]]`;
  pull vs the preregistered value `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:prediction.pull_vs_sourced_noise]]`,
  vs the ideal `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:prediction.pull_vs_ideal]]`.
  E_A = `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:E_A]]` ± `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:E_A_std_error]]`
  (dense `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:dense.E_A]]`); E_B_ctrl = `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:E_B_ctrl]]` ±
  `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:E_B_ctrl_std_error]]`; E_B_raw = `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:E_B_raw]]` ±
  `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:E_B_raw_std_error]]`; E_B_cut = `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:E_B_cut]]`.
  Verdict: `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:verdict.statement]]`.

- **Raw vs mitigated — evidence against a mitigation artifact.** The
  tensored inverse of the two calibration circuits
  (`[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:readout_calibration]]`: assignment errors 0.3–0.4 % on
  prepared 0, 0.6–0.9 % on prepared 1) moves E_B_info from
  `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:ledger.raw.E_B_info]]` ± `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:ledger.raw.E_B_info_std_error]]`
  (unmitigated) to `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:ledger.mitigated.E_B_info]]` — a shift of
  ~0.2σ; the unmitigated value is itself z ≈ 6.2. Give the raw and
  mitigated rows side by side (`ledger.raw` / `ledger.mitigated`) for
  E_A, E_B_ctrl, E_B_raw too: the sign and size of every conclusion is the
  same without mitigation.

- **The honest reading of E_B_ctrl < 0.** Teleportation proper is lost on
  this device (−1.8σ, and within 1σ of the map's prediction at the sourced
  noise): Bob's controlled rotations heat his site by more than the 1.2 %
  of E_A the encoded field lets him extract (§3). What survives at 6σ is
  the equal-depth information statement — the same gates on the right bit
  leave Bob lower in energy than on the wrong bit. That, and only that, is
  claimed; harvesting (§4) was not run.

- **Audited by construction (all four passed at 5σ; numbers in
  `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:audits]]`).** (i) energy-ledger closure on the measured
  energies: defect `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:closure.defect]]`, σ `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:closure.sigma]]`
  (ΔE `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:audits.energy_ledger.details.delta_E]]` vs E_A − E_B),
  with the two locality residuals — the spectator site's energy
  (`audits.energy_ledger.details.spectator_residual`, +1.5σ: idle
  decoherence during Bob's gates, which the `from_backend` rehearsal does
  not model) and Bob's no-signaling energy
  (`audits.energy_ledger.details.bob_residual`); (ii) E_A ≥ E_B; (iii)
  no-signaling from the ground/control marginal counts, three basis pairs,
  worst χ²/χ²_crit `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:no_signaling.worst_ratio]]` (one pair at
  p = 0.078, recorded); (iv) signal vs noiseless ideal, one-sided, pull
  `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.analysis.json:prediction.pull_vs_ideal_audited]]`. A failed audit would
  have been an event, not a number (exit status 3); none failed.

- **What the on-ramp taught (for the next run).** Least-busy device →
  `ibm_fez`; layout score `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.json:layout_score.score]]` over
  `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.json:layout_score.n_candidates]]` candidate paths. Submitted
  `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.json:job.submitted_utc]]`, execution span 32 s (queue ≈ 6 s),
  collected `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.json:job.collected_utc]]` — under two minutes of wall
  time. QPU time: IBM's usage meter `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.json:job.execution.usage]]` s
  against the transcribed formula's `[[data:real_device/ibm_fez_dadkfvl1ierc738l0ch0.json:job.qpu_time_estimate.seconds]]` s
  (conservative by ~25 %); the 600 s / 28-day Open Plan funds twenty such
  runs. Shots: 9 600 (a 5σ target on the pre-flight, which said z = 5.45
  at fez's live calibration; the device gave 6.23). The pre-flight was
  printed and not persisted — fixed: the sidecar and record now carry it,
  and `--save-analysis` writes the analysis as data.

- **What the user must do to repeat it** (one command): a free IBM
  Quantum account (Open Plan) and token; `.venv/bin/pip install
  qiskit-ibm-runtime`; `export IBM_QUANTUM_TOKEN=<token>`;
  `.venv/bin/python scripts/run_ibm_free_tier.py --n-sigma 5 --shots 9600`
  (`--backend NAME`, `--no-wait` / `--collect <job id>`, `--dry-run`,
  `--estimate-only`). Cost by IBM's formula: 2 s + 0.00035 s × 11 × shots —
  `[[data:s6_free_tier.json:qet_information_signal.qpu_time_s_at_3sigma_shots]]` s
  at the spec's 3 439 shots (sized on cairo; calibration-dependent, hence
  the pre-flight), 39 s at 9 600.
