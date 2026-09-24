# MANIFEST — hardware-as-noisy-field

Result candidate of Layer 7, **L6 "Hardware as noisy field"** (PLAN.md): the two
minimal vacuum protocols ported to a public gate-model device *in simulation*,
under the one sourced device calibration in the repository, with the survival
map of their signals.

---

## The claim

**Prior art (literature check, 2026-09-23; `docs/LITERATURE_CHECK_2026-09-23.md`).** Energy teleportation has been run on IBM
hardware before: Ikeda, Phys. Rev. Applied 20, 024051 (2023), on six devices with readout
mitigation, observed the interaction term ⟨V⟩ negative on every device and notes that Bob's full
gain is smaller once the positive local term is included; Xie, Sajjan & Kais, arXiv:2409.03973
(2024), and Hassan, Quaderi & Mahdy, arXiv:2408.07997 (2024; ibm_brisbane, ibm_kyiv,
ibm_sherbrooke), report further IBM runs. This candidate's real-device row — the information
signal positive at 6.2σ, teleportation proper lost on ibm_fez — sharpens Ikeda's caveat on one
device on one day. It is a measurement in their lineage, not a priority claim.

The Layer-1 harmonic-chain vacuum can be encoded on qubits by a Galerkin Fock
truncation whose error in the ground-state covariance falls exponentially in
the levels per site (N = 3, m = 1: **4.97e-2 → 3.72e-4 → 5.83e-9** at
d = 2, 4, 8 against `vacuum.core.ground_state_cov`), prepared variationally to
the exact truncated ground state (energy error **≤ 1e-14**, covariance error
**≤ 1.4e-8** for every instance up to 4 qubits at 3 layers, and up to 8 qubits once
the ansatz is deep enough to hold the state — 7 layers at (3,4), 31 at (4,4), `s2b`), and used as the medium for
(a) quantum energy teleportation and (b) two-detector entanglement harvesting.
On the ibm_cairo Table III calibration of Ikeda (2023), applied homogeneously
to a 3- or 5-qubit register with tensored readout mitigation:

* **QET, teleportation proper is lost.** Ikeda's raw signal E_B_raw and the
  Bob-idle-control signal E_B_ctrl are **negative** at the sourced point
  (−0.0264 ± 0.0018 and −0.0112 ± 0.0024 at 10⁵ shots; ideal +0.0104): Bob's
  two controlled rotations (4 CNOTs) heat his site by more than the 1.2 % of
  E_A the encoded field lets him extract. They come back only for a device
  **uniformly** better by ν ≈ **0.15** (E_B_raw) / **0.18** (E_B_ctrl) — CNOT
  error ~7e-4, readout ~1.3e-3, T1 ~0.5–1 ms — and not on any single axis.
* **QET, the information signal survives.** E_B_info = ⟨H_B⟩_wrong-bit −
  ⟨H_B⟩_right-bit (equal-depth control; ideal 0.0415) is **0.0401 ± 0.0025**
  at the sourced point (z = 16) and survives to **22×** the sourced gate
  errors (CNOT 9.7 %), **35×** the readout error (0.30), any T1 down to
  T1/64, and **21×** on all axes together.
* **Harvesting is dead at the sourced point** at every Trotter depth tried
  (3–8 steps, 28–68 CNOTs): λ_min(ρ^{T_B}) > 0. The 6-step circuit (ideal
  E_N = 0.068 nats, exact-evolution 0.082) revives only at **ν ≈ 0.46** on all
  axes (CNOT ~2.0e-3, readout ~3.9e-3, T1 ~0.2–0.3 ms) or **0.51** on the
  gate axis alone with mitigation; the 4-step circuit needs ν ≈ 0.30.

So the one protocol an experimentalist could read off a free-tier device on
this calibration is the QET *information* test, and `data/s6_free_tier.json`
says exactly what that run is: 3 qubits, 11 circuits (9 measurement + 2
calibration), native depth 36, ≤ 8 CNOTs, **3 439 shots per circuit** for 3σ
(15 s of QPU time by IBM's transcribed estimate; 387 s at the 10⁵ shots used
here, inside the 600 s Open-Plan window), testing the number
**E_B_info = 0.040 ± 0.002**.

**Status of the map: a claim about this noise model of that device, not about
the device.** The sweep axes are declared synthetic (see Provenance). **Status of
the free-tier statement: run, on a real device, and held** — `ibm_fez`,
2026-09-04, E_B_info = +0.04853 ± 0.00779 (z = 6.23) against the preregistered
+0.0401 (pull +1.08), with E_B_ctrl < 0 as predicted; the record, its calibration
snapshot and the audited analysis are the section "Real-device run" below.

---

## The producing build

| item | value |
|---|---|
| generator | `papers/hardware-as-noisy-field/notebook.py`, entry point `main()` |
| generator sha256 | `87e4eba041bcb05f2a1505f1e01b1cef77060090b5bbb3c4d7327e4d6898386e` |
| git HEAD at production (`git rev-parse HEAD`) | `45f5ee60be4ffb6606cb501441b24c8918c19ad1` (working tree dirty: several Layer 4–7 packages were being built concurrently, uncommitted) |
| produced (UTC) | 2026-09-03T03:05:20Z |
| package | `vacuum` 0.1.0 (editable); Python 3.13.3, numpy 2.5.2, scipy 1.18.1, qiskit 2.5.2, qiskit-aer 0.17.2 |
| command | `.venv/bin/python papers/hardware-as-noisy-field/notebook.py` |
| wall clock | **132.8 s** (budget 15 min) |
| pytest, this candidate | **31 passed in 19.5 s** (`tests/test_hardware.py`) |
| pytest, full suite | **880 passed, 4 failed, 5 warnings in 418.6 s** (`.venv/bin/python -m pytest -q`, run on the same code state, concurrently with the notebook). All four failures are in `tests/test_harvest_teleport.py` — the Layer-7 L4 candidate being built by another agent at the same time; nothing in `vacuum.hardware`, its tests or this notebook imports that module. **Per `papers/README.md` the build is therefore NOT admissible as it stands**: the candidate becomes admissible only from a build in which that suite is green — the integration record below is that build. `tests/test_hardware.py` itself: 31 passed. |

The sha256 and git HEAD are embedded in `data/summary.json` (`__build__`).

### S2b — the producing build of `data/s2b_vqe_deep.json`

| item | value |
|---|---|
| generator | `papers/hardware-as-noisy-field/notebook_vqe_deep.py`, entry point `main()` |
| generator sha256 | `e201f96a1de7be35f84bbc9e5bd99dc6e1a93401007e79e9b9ec7679d73e755a` |
| git HEAD at production | `29c8912663b270f75cf0436eab0b00efa0028478` |
| produced (UTC) | 2026-09-12T21:24:07+00:00 (the archive is assembled from `data/jobs/`; the jobs themselves ran 2026-09-11/12) |
| package | `vacuum` 0.1.0 (editable); Python 3.13.3, numpy 2.5.2, scipy 1.18.1, qiskit 2.5.2, qiskit-aer 0.17.2 |
| command | `.venv/bin/python papers/hardware-as-noisy-field/notebook_vqe_deep.py --workers 2` (2 worker subprocesses, 1 BLAS/OpenMP thread each) |
| wall clock | **15379.4 s** summed over the 75 jobs (`jobs_wall_clock_s`), the fix-round-1 continuations and the two below probes being 27 of them; the two fix-round-1 invocations took 318.0 s and 2 147.0 s of wall on 2 workers, and the assembling invocation itself is 0.3 s once `data/jobs/` is populated |
| pytest, this candidate | **99 passed, 1 skipped in 33.9 s** (`tests/test_hardware.py tests/test_exports.py tests/test_hardware_real_device.py tests/test_integration.py`); the skip is `test_layerwise_scan_reaches_roundoff_by_depth_eight`, behind `VACUUM_SLOW_TESTS=1` |

**What S2b settles.** The `s2` (3,4) and (4,4) rows were not an optimizer failure.
Maximising the state fidelity |<psi_ED|psi(theta)>|^2 directly at 3 layers --
Levenberg-Marquardt on the state residual with the exact Jacobian, from 8 seeded starts
plus 2 iterated-local-search rounds, and independently with L-BFGS-B on the infidelity --
plateaus at **6.4305e-5** (3,4) and **2.9305e-4** (4,4), which is where the `s2`
energy-VQE already was (6.560e-5 / 3.231e-4). Re-running the energy-VQE at 3 layers
*from that fidelity optimum* returns 2.332e-4 / 6.561e-5 and 1.396e-3 / 3.134e-4: the
same wall. Depth moves it: the envelope of the best infidelity per depth falls
6.43e-5 -> 3.32e-6 -> 4.10e-8 -> 1.69e-11 -> **2.22e-16** at L = 3..7 for (3,4), and
2.93e-4 -> ... -> 3.87e-8 at L = 11, 2.41e-8 at L = 20, 4.10e-12 at L = 30, **-4.44e-16** at L = 31 for
(4,4). The energy-VQE warm-started at those depths (gtol 1e-12, 1000 iterations, both
evaluators) converges on the first L-BFGS-B step.

*Every depth is continued before its number is read.* Each depth's best point is
re-optimised from itself at a larger Jacobian budget (`CEILING_CONTINUE`, tag
`..._cont`), and the archive records whether the best start converged (`xtol`/`ftol`)
or exhausted its budget. One layer below the published depth a harder **below probe**
runs (incumbent + 6 seeded starts + 4 iterated-local-search rounds, 2500 Jacobian
evaluations each) and is archived with what it reached. This is not decoration: fix
round 1 of this section published **8 layers** at (3,4) because the depth-7 cold starts
had exited on their 400-evaluation budget while still descending. See the retraction
below.

*Method note (ASSUMPTION, not proved).* The ceiling at a depth is a maximisation from
finitely many starts; a plateau shared by every converged start is evidence, not proof,
that the depth does not suffice. Only the depths that reach roundoff are *proved*
sufficient -- the state is exhibited, and every number for it is re-derived from the
archived parameters by `tests/test_hardware.py::TestVQEDeepRows`. The published depth is
therefore a **sufficient** depth (an upper bound), never a proved minimum, and the
archive's `sufficient_depth.*.is_proved_minimal` is `false` by construction.

**RETRACTION (S2b, fix round 1, 2026-09-12).** The row *"minimal depth reaching
infidelity <= 1e-12 at (3,4): 8 layers, 54 parameters"* — MANIFEST numbers table,
`s2b_vqe_deep.json` `minimal_depth.N3_d4.minimal_layers = 8`, and
`tests/test_hardware.py` `S2B_MINIMAL = {"N3_d4": (8, 10)}` with its assertion that no
depth below 8 reaches the acceptance — **is withdrawn and replaced by 7 layers, 48
parameters**. Cause: at depth 7 every cold Levenberg-Marquardt start that came near the
minimum exited with *"The maximum number of function evaluations is exceeded"* at the
scan's 400-evaluation budget, so the archived 3.56e-12 was an unfinished descent, not
the ansatz's ceiling; continuing that same point reaches roundoff in ~230 further
evaluations (`tests/test_hardware.py::TestVQEDeepRows::test_continuation_closes_the_seventh_layer`,
live, ~8 s). Found by two independent verifiers. The wording *"minimal"* is withdrawn
for both cases, not only the one that moved: the same argument never established
minimality at (4,4) either. What replaces it is the continuation stage at every depth,
the below probe, and the `is_proved_minimal: false` field.

**CORRECTION (S2b, fix round 2, 2026-09-12).** Two statements in the Limitations list
below were stale — left over from before commit `1c5881b` of fix round 1, which ran the
(4,4) continuations and below probe in the same round but did not rewrite this prose.
No published number is withdrawn: both corrected statements contradicted the MANIFEST's
own numbers table, the archive and the pinned tests, which were already right. (i) The
open-question bullet (a) said *"L = 30 reached 1.78e-11 ... no below probe has been run
at (4,4)"* and told the reader to add `BELOW_PROBE[(4, 4)]` and a `"stage2"` entry that
already exist; the measured value is **4.100276e-12** (`sufficient_depth.N4_d4
.below_probe`, matching the numbers-table row), and bullet (a) now carries the real
remaining knobs — raise those two existing entries and delete the two cached job files.
(ii) The non-monotone list read *"depths 12, 16, 20, 24, 28"*; the archive's
`non_monotone_depths` and `tests/test_hardware.py` say **[16, 24, 28]** — the round-1
continuations improved L = 12 to 2.473845e-08 and L = 20 to 2.409933e-08. Found by an
independent verifier. No number, data file, archive field or notebook knob changed in
this round. What was added is the guard that would have caught it:
`tests/test_hardware.py::TestVQEDeepRows::test_manifest_limitations_agree_with_the_archive`
parses this file's three factual (4,4) lists — the non-monotone depths, the
budget-exhausted depths and the never-scanned depths — plus the below-probe value, and
asserts each against the archive's own `non_monotone_depths`,
`budget_exhausted_depths`, `depths_not_scanned_below` and `below_probe.infidelity`, so
prose that drifts from the archive fails instead of being read as fact (each of the four
lists was mutated to its stale round-1 form and the test failed on all four; 0.6 s).

---

**Integration: full suite 1201 passed on the quiescent tree at HEAD `c18a206`, 2026-09-04** (`.venv/bin/python -m pytest -q`, twice in a row on the final tree: 1201 passed, 10 warnings in 731.91s (0:12:11) and 1201 passed, 10 warnings in 740.37s (0:12:20); with tenpy, cvxpy, jax, jaxlib, qiskit, qiskit_aer and qiskit_ibm_runtime blocked at the import system, 1026 passed, 76 skipped in 428.84s (0:07:08); all 15 notebook `--quick` paths exit 0, the slowest in 307 s). The tree is commit `c18a206` plus this pass's uncommitted edits — README, STATUS, PRIORITY, MANIFEST and draft prose reconciling the rollups to the MANIFESTs, plus one print in `interacting-vacua-first-numbers/notebook_qei_exact.py` made None-safe so its `--quick` path exits 0 (no numerical code, no data). `data/` was produced on the build recorded above; this run is the admissibility record under `papers/README.md` for the candidate's code as it stands on that tree, and a full re-run of `notebook.py` there regenerates `data/` with a new `generator_sha256`; the producing-build fields above are left as recorded.

## Modules used

| module | role |
|---|---|
| `vacuum.hardware.vacuum_sim` | Galerkin Fock truncation (site 0 least significant = qiskit little-endian, no permutations anywhere), Pauli decomposition, exact diagonalization, `encoding_convergence`, hardware-efficient VQE on **Aer statevector** (batched `parameter_binds`, parameter-shift gradients, L-BFGS-B, seeded restarts) with the numpy arbiter; for S2b also the exact adjoint gradients (`hea_overlap_and_grad`, `hea_energy_and_grad`, `hea_state_jacobian`), exact layer embedding (`hea_embed_layers`), the state-fidelity maximiser `fidelity_ceiling` (Levenberg-Marquardt on the state residual, or L-BFGS-B) and `fidelity_depth_scan` (layerwise growing), `ansatz_energy` (the cross-evaluator check), and `vqe_ground_state`'s `warm_start` / `warm_kwargs` / `gradient='adjoint'` keywords (defaults unchanged, so every pinned `s2` row is reproduced by the same call) |
| `vacuum.hardware.protocols` | QET (sign-of-x_A measurement, conditional displacement, four circuit families incl. the inverted-bit control; `qet_dense` arbiter; `qet_audits`), harvesting (two ancilla UDW qubits, cos² switching from `vacuum.detectors.switching`, Strang product formula over commuting groups, dense Trotter mirror + RK4-exact reference, 9-setting tomography, bootstrap σ), tensored readout mitigation, `harvest_gaussian_reference` |
| `vacuum.detectors.nonperturbative.run_harvesting` | the Gaussian oscillator-detector reference (MAP-1: λ_osc = λ√(2Ω)) |
| `vacuum.qet.hotta` | closed forms anchoring the N = 2, d = 2 encoding (h = ω/2, k = 1/(4ω)) |
| `vacuum.qet.ikeda_qiskit` | `noise_model_from_params` (the sourced calibration, every number cited), `_depolarizing_to_match`, `run_on_backend` — reused, not duplicated |
| `vacuum.hardware.noise` | calibration point read from the Ikeda description, labelled sweeps, N-qubit tiling, survival bisection, `device_run` |
| `vacuum.hardware.real_device` | the free-tier batch of `s6` built from the working point (11 circuits, spec asserted), noise-selected 3-qubit path and ISA transpilation with the two-qubit budget asserted, `SamplerV2` submission, the provenance record (`RECORD_FORMAT`), the audited analysis (ledger closure on measured energies, E_A ≥ E_B, no-signaling from marginal counts, all through `qet_audits`), the verdict; rehearsed on `qiskit_ibm_runtime.fake_provider`, then run once on `ibm_fez` (section "Real-device run") |
| `vacuum.experiments_io` | loads `experiments/ibm_qet/ikeda_2023_fig2.json` and the sweep variant file (provenance enforced at load) |
| `vacuum.audits` | `AuditResult`/`AuditViolation` behind `qet_audits` |

---

## Provenance

### Transcribed (not synthetic)

| file | what is taken |
|---|---|
| `experiments/ibm_qet/ikeda_2023_fig2.json` | Table III, ibm_cairo column: T1/T2, Pauli-X errors, readout assignment errors, CNOT error and gate time, single-qubit gate-time range (Sec. II.D) — through `vacuum.qet.ikeda_qiskit.noise_model_from_params` |
| `experiments/ibm_qet/ikeda_2023_fig2_variant-noise-sweep.json` (transcribed entries) | IBM Open Plan budget **600 s per 28-day window**; IBM QPU-time estimate **2 s + 0.00035 s × (circuits × shots)** and the default `rep_delay` 250 µs (IBM Quantum documentation, retrieved 2026-09-02, URLs in the file) |

### SYNTHETIC (declared; the claim inherits this)

| input | where declared | why synthetic |
|---|---|---|
| gate-error / readout / T1 multiplier grids | the `_variant-noise-sweep.json` file | only ν = 1 is the device |
| tiling of the two Table III rows over 3–5 qubits; the one CNOT error on every edge | `vacuum.hardware.noise.noise_model_from_calibration` docstring and every description dict | Table III calibrates one pair |
| shots per circuit (10⁵ QET, 10⁴ per tomography setting) | the variant file / notebook constants | budget choices |
| chain N = 3, m = 1, Dirichlet, d = 2; sites; gap 1.5; λ = 1; width 1.5; 6 steps | notebook docstring | model choices; the d = 4, distance-2 and n_steps rows quantify them |
| Bob's angle θ | optimised numerically per instance (`qet_optimize_theta`), never a closed form | — |

Not modelled: crosstalk, leakage, coherent errors, drift; relaxation stays at
the sourced T1/T2 when the gate axis is scaled (the depolarizing part clips to
0 below the relaxation floor); readout mitigation is tensored, not full.

---

## Validation numbers (all in `data/`)

| check | value |
|---|---|
| Pauli decomposition vs dense H_d | < 1e-12 (tests) |
| encoding covariance error, N = 2, d = 2/4/8/16 | 2.43e-2 / 4.02e-5 / 3.02e-11 / 1.7e-16 (`s1`) |
| Rayleigh–Ritz: E_0(d) − E_0 exact ≥ 0, monotone | yes on every row of `s1` |
| N = 2, d = 2 ↔ Hotta closed forms (E_A, E_B*) | residuals < 1e-12 (`s3a`, tests) |
| VQE energy / covariance / infidelity, (N,d) = (2,2),(3,2),(4,2),(2,4) | ≤ 1e-14 / ≤ 1.4e-8 / ≤ 2.4e-15 (`s2`) |
| VQE (3,4) and (4,4), **3 layers**, 3 restarts × 400 it. | **not converged**: energy error 2.334e-4 / 1.426e-3, infidelity 6.560e-5 / 3.231e-4 (`s2`) — superseded by the `s2b` rows below, kept as recorded |
| why: expressibility ceiling of the **3-layer** ansatz | direct fidelity maximisation at 3 layers (8 seeded LM starts + 2 iterated-local-search rounds, and independently L-BFGS-B) plateaus at infidelity **6.4305e-5** (3,4) and **2.9305e-4** (4,4) (`s2b`) — an upper bound on what the ansatz can reach there; the `s2` energy-VQE reached 6.560e-5 / 3.231e-4, i.e. the **ansatz**, not the optimizer |
| **sufficient** depth reaching infidelity ≤ 1e-12 (an upper bound, **not** a proved minimum) | **(3,4): 7 layers** (48 parameters, 63 real dof — three layers *below* the counting depth 10); **(4,4): 31 layers** (256 parameters, 255 real dof — *equal to* the counting depth) (`s2b`). Round 1 published 8 at (3,4); see the retraction above |
| the probe one layer below, (3,4) | at **6 layers** (42 parameters) the incumbent optimum + 6 seeded LM starts + 4 iterated-local-search rounds, 2500 Jacobian evaluations each, converge to **1.694e-11** (best start ends on `xtol`, gradient norm 5.2e-16, and two independent starts land there) — evidence at a stated budget that 6 does not reach 1e-12, **not** a proof (`s2b`) |
| the probe one layer below, (4,4) | at **30 layers** (248 parameters) the incumbent + 1 iterated-local-search round, 400 Jacobian evaluations each, reach **4.100e-12** and were **still descending** when the budget ran out (gradient norm 6.7e-8). The depth-30 number has fallen 3.64e-10 → 1.78e-11 → 5.61e-12 → 4.10e-12 as the budget grew, so **30 is not excluded** and is likely sufficient; the (4,4) row rests on the state exhibited at 31 (`s2b`) |
| VQE (3,4) at 7 layers / (4,4) at 31 layers, warm-started from the fidelity optimum, **both evaluators** | **converged**: \|energy error\| ≤ 3.2e-15, \|infidelity\| ≤ 4.5e-16, covariance error ≤ 1.1e-15, Aer-vs-numpy ≤ 4.9e-15, cross-evaluator energy ≤ 4.9e-15 (`s2b`) — the standard of the four rows above. (3,4): dE −1.78e-15 (aer) / −4.44e-16 (numpy), infidelity 2.22e-16, covariance 3.33e-16, 98 evaluations / 1.1 s (aer) |
| downstream (3,4) on the VQE state (7 layers) vs the ED state | QET ledgers (A/B = 0/1, 0/2) max \|Δ\| **1.15e-9** — the θ-optimizer's own tolerance, Δθ = 2.4e-9 and 5.1e-9, with the two states agreeing to 2.2e-16; both QET audits pass on the VQE state. Harvesting E_N **0.0** and ρ_AB **1.1e-16** (`s2b`). Nothing downstream runs at (4,4) |
| Aer statevector vs numpy ansatz energy | ≤ 4e-15 |
| QET: noiseless Aer vs dense ledger | 4.4e-16 on E_B (`s3b`); 1e-10 asserted in tests for d = 2 and d = 4 |
| QET audits (closure, E_A ≥ E_B, no-signaling) | pass on every dense row and the ideal Aer run; ledger defect 1.1e-15, no-signaling 2.3e-16 |
| harvesting: noiseless Aer vs dense Trotter mirror (ρ_AB) | 1.3e-15 (`s4c`); d = 4 also < 1e-10 (tests) |
| harvesting: Trotter error of the 6-step circuit vs RK4-exact | −0.0132 nats on 0.0816 (`s4a`; the circuit *is* the 6-step protocol, its mirror matches it, and the error is stated) |
| Gaussian limit, N = 2, λ ≤ 0.2 | qubit vs oscillator detectors within **0.05–0.6 %** at d = 4/8, 4.3–4.5 % at d = 2; λ = 1: +20 % (d ≥ 4) — the two-level saturation, outside the leading-order map (`s4b`) |
| sampled Aer (with the noise model, real shots) vs shot-free σ | pulls 0.40 (E_B_info), −0.33 (E_B_ctrl), 0.62 / 1.14 (λ_min) (`s5`) |

Causality: the harvesting detectors are in causal contact (window 3 > lattice
separation 1); the flag is carried in every row. No spacelike claim is made.

---

## Data

| file | contents |
|---|---|
| `data/s1_encoding_convergence.json` | per (N, m, d): E_0(d), E_0 exact, dE, covariance errors |
| `data/s2_vqe.json` | per (N, d): energies, errors, restarts, evaluation counts |
| `data/s2b_vqe_deep.json` | S2b: the expressibility ceiling by depth (raw, continued and enveloped), the layerwise-growing scans, the sufficient depth per case with the probe one layer below it, the warm-started energy-VQE rows on both evaluators, and the (3,4) downstream comparison rows |
| `data/jobs/s2b_*.spec.json` / `s2b_*.json` | one spec and one result per S2b job (75 of them, 15379.4 s of single-thread wall); `notebook_vqe_deep.py` skips a job whose result exists, so a crashed run resumes. Tags: `ceiling_*` the depth scan, `*_cont` its continuation, `probe_*` the below probe, `layerwise_*`, `vqe_*`, `downstream_*` |
| `data/s3a_qet_dense.json` | dense ledgers incl. the Hotta anchor and the distance rows |
| `data/s3b_qet_working_point.json` | the N = 3, d = 2 working point: dense, ideal Aer, sourced-point raw/mitigated, noise description |
| `data/s3c_qet_survival.json` | thresholds (3 signals × 4 axes) with full bisection histories; declared-grid rows |
| `data/s4a_harvest_trotter.json` | RK4-exact reference and the Trotter table |
| `data/s4b_harvest_gaussian_limit.json` | Gaussian vs qubit-detector E_N over (N, d, λ) |
| `data/s4c_harvest_working_point.json` | ideal and sourced-point results per n_steps, raw/mitigated |
| `data/s4d_harvest_survival.json` | thresholds (n_steps 4/6 × mitigation × 4 axes) and grid rows |
| `data/s5_sampled_crosschecks.json` | sampled vs shot-free at three points |
| `data/s6_free_tier.json` | the free-tier statement |
| `data/summary.json` | everything above condensed, with `__build__` |
| `data/summary_quick.json` | the `--quick` configuration (same schema) |
| `draft/survival_map.svg` | z vs ν on the declared grids, both protocols, sourced point marked |

## Real-device run — **the program's one hardware excursion: done, prediction held**

**The claim (about a real device).** On IBM's `ibm_fez` (Heron r2, 156 qubits, CZ), on
2026-09-04, on physical qubits [136, 143, 142], the encoded-vacuum QET *information*
signal is positive at five standard errors:

> **E_B_info = ⟨H_B⟩_wrong-bit − ⟨H_B⟩_right-bit = +0.04853 ± 0.00779 (z = 6.23)**, 9 600
> shots per circuit, tensored readout mitigation; unmitigated **+0.04715 ± 0.00765
> (z = 6.16)**. The classical bit is worth a positive amount of Bob's local energy on a
> public device today.

Its provenance is the record `data/real_device/ibm_fez_dadkfvl1ierc738l0ch0.json`
(sha256 `a3499fba…35f1`; job `dadkfvl1ierc738l0ch0`; the calibration snapshot taken at
submission, `properties().last_update_date` 2026-09-04T14:37:12−07:00, sha256 of the full
properties dict `f3ed9e05…3bb6`) with its `.submitted.json` sidecar and
`….analysis.json` (the audited analysis at 5σ, `--save-analysis`). Admissibility: this is
PLAN.md's one hardware excursion (Layer 7, L6), a claim about a real device and not about
a noise model — narrow by construction (one chip, one day, one path of three qubits, one
number, at the stated shots) — admissible under `papers/README.md` on the same footing as
every other candidate: the integration record of the tree it was produced on is the
coordinator's; the number is re-derived from the persisted counts by
`scripts/run_ibm_free_tier.py --analyze … --n-sigma 5` bit for bit, and
`tests/test_hardware_real_device.py::TestRealDeviceRecord` pins the record's sha256, its
provenance, every headline number, all four audits and the verdict (the first real-device
fixture in the repository).

**Predicted before the run, and where.** `data/s6_free_tier.json` (produced 2026-09-03,
before any token existed) preregistered the number, the batch and the value:
E_B_info = **0.0401 ± 0.0025** at 10⁵ shots on the *sourced ibm_cairo* calibration (Ikeda
2023, Table III), ideal **0.0415**; and, from the survival map (`s3b`/`s3c`), that
teleportation proper is *lost* at that noise: E_B_ctrl = **−0.0112**, E_B_raw = **−0.0264**.
The prediction was made on ibm_cairo's calibration (a 2023 Falcon; CNOT error 7.3e-3,
readout ~8e-3) and held on ibm_fez (a 2026 Heron; CZ error 1.9–2.0e-3, readout 3.3–4.3e-3,
T1 114–131 µs, sx 1.5–3.7e-4): the information signal is set by the circuit's structure,
not by the chip.

**Measured (tensored mitigation; raw alongside), and the pulls.**

| quantity | ibm_fez, 9 600 shots | pre-registered (cairo noise) | pull | dense ideal | pull |
|---|---|---|---|---|---|
| **E_B_info** | **+0.04853 ± 0.00779** | +0.0401 | **+1.08** | +0.0415 | +0.90 |
| E_B_info, unmitigated | +0.04715 ± 0.00765 | — | — | +0.0415 | +0.74 |
| E_A | +0.84999 ± 0.00193 | +0.8468 | +1.7 | +0.8538 | −2.0 |
| E_B_ctrl (teleportation proper) | **−0.01381 ± 0.00755** | −0.0112 | −0.34 | +0.0104 | −3.2 |
| E_B_raw (Ikeda's definition) | −0.02897 ± 0.00569 | −0.0264 | −0.46 | +0.0104 | −6.9 |
| E_B_cut (Hotta's cut channel) | −0.0381 | −0.0312 | — | −0.0103 | — |

**The honest reading of E_B_ctrl < 0.** Teleportation proper — Bob extracting energy from
his site relative to the Bob-idle control — is *lost* on this device, at −1.8σ and inside
1σ of what the survival map said it would be: Bob's two controlled rotations (four
two-qubit gates plus the readout) heat his site by more than the 1.2 % of E_A the encoded
field lets him extract. What survives, at 6σ, is the equal-depth *information* statement:
with the same gates, acting on the right bit leaves Bob 0.0485 lower in energy than acting
on the wrong bit. That is the number the survival map said a free-tier run could read, and
it is the only number claimed. Nothing about harvesting (dead at 52 CNOTs, §4) was run.

**Not a mitigation artifact.** The readout assignment on these qubits is 0.3–0.4 % (prep
0) and 0.6–0.9 % (prep 1); the tensored inverse moves E_B_info by 0.0014 — 0.18σ — and the
unmitigated value is itself z = 6.16. E_A, E_B_ctrl and E_B_raw move by 0.4 %, 0.0002 and
0.007 respectively (the raw ledger is in the analysis file).

**Audits on the real counts (all four passed, at the run's n_sigma = 5).**

| audit | statistic | tolerance | verdict |
|---|---|---|---|
| energy ledger closure, (E_final − E_ground) − (E_A − E_B) | defect **−0.0031**, σ 0.0087 (z −0.36); ΔE = 0.8607, E_A − E_B = 0.8638 | 5σ = 0.0435 | PASS |
| — spectator residual ⟨h_C⟩_final − ⟨h_C⟩_ground | +0.0064 ± 0.0042 (z 1.5) | (diagnostic) | — |
| — Bob residual ⟨H_B⟩_meas − ⟨H_B⟩_ground (no-signaling in energy) | −0.0050 ± 0.0071 (z −0.7) | (diagnostic) | — |
| no-signaling from marginal counts, ground vs control, 3 pairs | χ² = 6.81 / 1.45 / 2.97 on 3 dof (p = 0.078 / 0.69 / 0.40); worst χ²/χ²_crit **0.21** at 5σ (**0.48** at 3σ) | ≤ 1 | PASS |
| signal vs noiseless ideal (one-sided) | pull **+0.90** (excess +0.0070) | 5σ = 0.0390 | PASS |
| `hardware_qet` (closure, E_A ≥ E_B, no-signaling) | E_A − E_B = 0.864 ≫ 0 | 5σ | PASS |

The device did what the model could not: the spectator site's energy rose by 1.5σ during
Bob's gates (idle decoherence the `from_backend` rehearsal does not model), and the
ground-vs-control Z-marginals differ at p = 0.078 in one pair — both well inside tolerance
and both recorded.

**The run itself (what the on-ramp taught).** Least-busy operational device → `ibm_fez`;
layout score 0.0157 over 244 candidate paths; ISA depth 40/38 (protocol/wrong), 22 (the
rest), 8 CZ, no routing; `qiskit_ibm_runtime.SamplerV2`, tags `vacuum-program`,
`L6-free-tier`, twirling and dynamical decoupling left off, seed none (a device). Submitted
22:36:45 UTC, execution span 22:36:51–22:37:23 (32 s; queue ≈ 6 s), collected 22:38:29 —
1 min 44 s wall. **QPU time: IBM's usage meter 30 s** against the transcribed formula's
39 s for 11 × 9 600 executions (the formula is conservative by ~25 % here); the Open Plan
budget of 600 s per 28 days would fund this run twenty times. Shots: the spec's 3 439 (sized
on cairo, z = 3) were raised to 9 600 for a 5σ target on the pre-flight; the pre-flight
shot-free prediction at fez's live calibration said z = 5.45 at 9 600 shots and the device
gave 6.23 (E_B_info +0.77σ above the pre-flight value — the device's signal was slightly
larger than its own noise model's). Two lessons folded back into the code: the pre-flight
prediction was printed but not persisted (the sidecar now carries it, and the record
inherits it); and the analysis is now written next to the record (`--save-analysis`) so the
draft can cite it as data. The seed-0 bug of the rehearsal (below) does not touch a device
run (no simulator seed).

**Rehearsal (superseded by the run; kept as the format exemplar).** `--dry-run` on
`FakeSherbrooke` (Eagle r3, ECR; calibration snapshot 2025-02-26): path [122, 123, 124],
ISA depth 40, 8 ECR; shot-free at that calibration E_B_info = 0.0391 ± 0.0138 at 3 439
shots (z = 2.83 — 3 863 shots needed for 3σ there); the seed-1 sampled record
`data/real_device/dry_run/fake_sherbrooke_2e71ada7-….json` gave +0.0602 ± 0.0134
(z = 4.48), all four audits passed; two earlier seed-0 records in that directory are
non-reproducible (qiskit-ibm-runtime 0.49's local sampler treats `seed_simulator = 0` as
unset; `submit()` now refuses seed 0 on simulated backends). Noiseless, the analysis
reproduces the dense arbiter to 1e-15. Corruption tests: Bob's bit flipped in the control
family fails the closure, the marginal test and `hardware_qet`; Alice's bit flipped in the
ground family fails E_A ≥ E_B; Bob's bit flipped in the *wrong* family — invisible to those
three — fails `signal_vs_ideal` (pull +130.8) and returns `claim_holds = False`. The
one-sided ideal bound is a statistical test and tightens with shots.

**Code.** `vacuum/hardware/real_device.py`: `build_free_tier_job()` (the `s6` batch, spec
asserted), `select_physical_qubits()` / `transpile_for_backend()` (lowest-noise path,
two-qubit budget re-asserted, routing forbidden), `submit()` / `collect()` (record format
`vacuum.hardware.real_device/1` with the calibration snapshot at submission and a sha256),
`analyze()` (four standing audits, tensored mitigation, raw alongside; `strict=True` raises
`AuditViolation`), `predict_on_backend()` (the shot-free survival analysis at a backend's
own calibration, the pre-flight), `verdict()`. `scripts/run_ibm_free_tier.py` is the one
command. `tests/test_hardware_real_device.py`: **18 passed** (with `tests/test_hardware.py`,
49).

## What is NOT claimed

- About the device, only what the record supports: E_B_info > 0 at 5σ on `ibm_fez` qubits
  [136, 143, 142] on 2026-09-04 at 9 600 shots, with E_B_ctrl and E_B_raw negative. Not
  teleportation proper on a device (it is lost there, as predicted), not harvesting (not run),
  not any other chip, day or path; the dry-run records under `data/real_device/dry_run/` are
  simulations on a fake backend's calibration snapshot. Nothing beyond N = 3 sites and d ≤ 4.
- No spacelike harvesting. No statement about E_B_ctrl / E_B_raw being
  measurable on the current calibration.
- The (3,4) and (4,4) rows are now ground states, at **7** and **31** layers
  (`s2b`), and the old 3-layer rows are the ansatz's expressibility ceiling,
  not a stuck optimizer. What is *not* claimed about them: a depth's number is
  the best infidelity found over finitely many seeded starts at the archived
  knobs — an **upper bound** on the attainable infidelity, never a proven
  supremum. So the published depths are **sufficient**, not minimal, and the
  archive's `is_proved_minimal` is `false`. Round 1 of this section did claim
  "minimal" and was wrong by a layer at (3,4); the retraction is recorded
  above. What is now done before a depth's number is read: (i) every depth is
  **continued** from its own optimum at a larger Jacobian budget, and the
  archive records whether the best start converged or exhausted its budget
  (`budget_exhausted`; a test fails if any depth below the published one is
  left exhausted); (ii) one layer below, the **below probe** runs a harder
  search and is archived with what it reached — at (3,4), L = 6 converges to
  **1.694e-11** (gradient norm 5.2e-16, two independent starts) and does not
  reach 1e-12. Neither is a proof that 6 layers cannot hold the state; both
  are numbers the next reader can rerun.
- Open, with the exact knobs. **(a) Whether (4,4) closes below 31 layers.**
  The continuation stage and the below probe have **both already been run
  here** (fix round 1, commit `1c5881b`): `CEILING_CONTINUE[(4, 4)]` is
  `{"depths": "all", "maxiter": {default 1000; 16, 20, 24, 28: 500; 30: 1200},
  "stage2": {30: 1000}}` and `BELOW_PROBE[(4, 4)]` is `{"restarts": 0,
  "perturb_rounds": 1, "maxiter": 400}`. The probe at L = 30 reached
  **4.100276e-12** — a factor 4 from the 1e-12 acceptance — with its best
  start ending on *"The maximum number of function evaluations is exceeded"*
  and gradient norm 6.65e-8, i.e. **still descending**; the depth-30 number
  fell 3.6359e-10 (scan) → 1.7801e-11 (continuation) → 5.6093e-12 (stage 2)
  → 4.1003e-12 (probe) as the budget grew. So **30 is not excluded and is
  likely sufficient**; sufficient depth 31 is what is *exhibited*, not
  evidence that 30 fails. **This is the weakest part of the row:** the (4,4)
  depths 10, 11, 20 and 30 all ended on their Jacobian budget
  (`budget_exhausted_depths` in the archive, pinned by a test), and depths
  13–15, 17–19, 21–23, 25–27 and 29 were never scanned at all (above 12 the
  sampled depths are 16, 20, 24, 28, 30, 31, 32, 36). To push it further, in
  `papers/hardware-as-noisy-field/notebook_vqe_deep.py`: **raise**
  `CEILING_CONTINUE[(4, 4)]["stage2"][30]` (1000 now) and
  `BELOW_PROBE[(4, 4)]["maxiter"]` (400) / `["perturb_rounds"]` (1) and delete
  `data/jobs/s2b_ceiling_N4_d4_L30_cont2.json` and
  `data/jobs/s2b_probe_N4_d4_L30.json` to force a re-run; raise
  `CEILING_CONTINUE[(4, 4)]["maxiter"]` past 1000 and delete the matching
  `*_cont.json` for the still-exhausted 10, 11 and 20; and add the unscanned
  depths to the second `(4, 4)` block of `CEILING_CASES`. At ~1.2 s per
  Jacobian evaluation at L = 30, each 1000 evaluations costs ~20 min on one
  thread. If a depth below 31 then reaches 1e-12, `S2B_SUFFICIENT`,
  `S2B_CEILING`, `S2B_BELOW_PROBE` and `S2B_BUDGET_EXHAUSTED` in
  `tests/test_hardware.py` and the rows above must move in the same pass. **(b) Whether (3,4) closes below 7.** The
  below probe says 6 does not at 2500 evaluations x 11 starts; a harder search
  there (raise `BELOW_PROBE[(3, 4)]`, delete
  `data/jobs/s2b_probe_N3_d4_L6.json`) is the way to test it, and the L = 6
  number has already fallen 9.30e-10 -> 4.05e-10 -> 1.69e-11 as the budget
  grew, so it is not settled. **(c)** Nothing here proves a *lower* bound on
  the depth; the honest object is the upper-bound-per-depth curve.
- The parameter-counting depth (smallest L with n(L+1) ≥ 2ⁿ − 1, the real dof
  of a real n-qubit state) is a **necessary** condition for the R_Y/CX image to
  cover that sphere, not a theorem about one target state: (3,4) closes three
  layers below it (48 parameters for 63 real dof). The cold-start search is
  also **not monotone** in depth at (4,4) (depths 16, 24, 28 are worse than a
  shallower depth already achieved — `non_monotone_depths` in the archive,
  pinned by a test; the list was [12, 16, 20, 24, 28] before the fix-round-1
  continuations improved L = 12 to 2.473845e-08 and L = 20 to 2.409933e-08);
  only the envelope — min over
  depths ≤ L, attainable there because `hea_embed_layers` embeds a shallower
  optimum exactly — bounds the ansatz.
- The deep states are not used by any protocol row today: `s3`/`s4` run at
  (3,2), and the (3,4) `s3a`/`s4b` rows are ED-state rows (the S2b downstream
  block shows the 7-layer VQE state reproduces them to the θ-optimizer's own
  tolerance). A future noisy run at (3,4) could use the 7-layer circuit, but
  at 7 layers and 35 CX gates it is far deeper than the 3-layer circuit the
  noise study assumed — a cost nobody has budgeted.

## Reproducing

```
.venv/bin/python papers/hardware-as-noisy-field/notebook.py            # ~2.5 min
.venv/bin/python papers/hardware-as-noisy-field/notebook.py --quick    # ~20 s
.venv/bin/python papers/hardware-as-noisy-field/notebook_vqe_deep.py --workers 2  # S2b; resumable from data/jobs/ (~0.3 s when cached, 12 751 s of job wall cold, 2 workers x 1 thread)
.venv/bin/python papers/hardware-as-noisy-field/notebook_vqe_deep.py --quick --workers 2   # S2b smoke, ~25 s, temp dir (all five phases)
VACUUM_SLOW_TESTS=1 .venv/bin/python -m pytest tests/test_hardware.py -q -k layerwise_scan_reaches   # the 67 s live layerwise scan
.venv/bin/python -m pytest tests/test_hardware.py -q                   # 31 tests, ~20 s
.venv/bin/python -m pytest tests/test_hardware_real_device.py -q       # 18 tests, ~12 s (needs qiskit-ibm-runtime)
.venv/bin/python scripts/run_ibm_free_tier.py --analyze papers/hardware-as-noisy-field/data/real_device/ibm_fez_dadkfvl1ierc738l0ch0.json --n-sigma 5   # re-derive the device verdict
.venv/bin/python scripts/run_ibm_free_tier.py --dry-run                # rehearsal on FakeSherbrooke, ~15 s
.venv/bin/python scripts/run_ibm_free_tier.py --n-sigma 5 --shots 9600 # another device run (IBM_QUANTUM_TOKEN; ~30 s of QPU)
.venv/bin/python -m pytest -q                                          # admissibility gate
```
