# MANIFEST — mi-coherence-harvesting

Result candidate of **Layer 2's pivot** (PLAN.md: "if harvested negativity under
honest noise models sits orders of magnitude below detectability for all
realistic parameters, the layer pivots to mutual-information and coherence
harvesting — measurable classical-quantum correlations rather than distillable
entanglement"). The sibling candidate `circuit-qed-noise-thresholds` fired that
trigger for `E_N` and carried one MI column; this candidate walks the pivot and
answers its question with numbers, for five correlation measures, against a
resolvability floor derived for *each* of them.

---

## The claim

For the same model, the same sourced operating point (Table I scenario 2 of
Teixidó-Bonfill et al., PRA 113, 043732 (2026)) and the same audited code path
as the sibling — 1+1 lattice field, two harmonic (oscillator-UDW) detectors,
`'xx'` coupling, cos² switching, published rates as Gaussian channels — with
the four surfaces rebuilt at **12 × 12** (576 surface points + 12 static-gap
split companions + 3 finite-size controls = **591 rows**), each point carrying
`E_N`, `I(A:B)`, the Gaussian quantum discords `D^{←}`, `D^{→}`, the classical
correlations `J`, the relative entropy of coherence `C_r` in the detectors'
local energy bases and the correlated coherence `C_cc`:

1. **The pivot's answer at the device point is: none of them, at the floor the
   same tomography actually sets.** In the device box (30 / 50 mK × measured
   Γ₁ / plot-digitised Γ_φ × sourced separations × the derived coupling, 16
   anchor points) `max E_N = 0` exactly, and the surviving correlations are
   `MI = 2.399e-4`, `D^{←} = 2.248e-4`, `C_cc = 2.398e-4`, `C_r = 3.187e-4`
   nats — **0.22–0.32×** the sibling's E_N-derived floor of 1e-3 nats, and
   **0.13–0.38×** this candidate's own per-point Monte-Carlo floors
   (`MI` floor 1.90e-3, `D^{←}` 6.43e-4, `C_cc` 8.44e-4, `C_r` 8.45e-4 nats).
   Every measure is GO at the *band's low edge* (7.82e-5) and NO-GO at the
   derived floor, the band's high edge and the model floor. The pivot's
   optimistic reading — "MI survives where E_N dies" — is true of the *value*
   and false of the *measurement*: MI's own floor is **1.9× higher** than the
   E_N floor that was transferred to it, because a non-negative correlation
   measure picks up a positive bias from reconstruction noise on an
   uncorrelated state, while a concurrence does not.
2. **The one exception is local, not correlated.** Two device-box points
   (`dephasing-T30-X2e+08`, `dephasing-T50-X2e+08`) have `C_r` above its model
   floor (2.10e-5 and 2.24e-5 vs 1.30e-5 and 1.36e-5 nats). At those points the
   device's own dephasing has heated the detectors to ⟨n_d⟩ = 0.269, and the
   *correlated* coherence `C_cc` (1.23e-6, 3.13e-6) is far below its floor.
   The only thing resolvable at the device point is single-detector coherence
   created by the noise.
3. **Where the measurable region opens: at 2.1–2.5× the derived coupling.** On
   the (T, λ_osc) surface, at the model floor, the first resolvable coupling at
   30 mK is λ_osc = 0.35 for MI, 0.30 for `D^{←}` and `C_cc`, 0.30 for `C_r`,
   against the derived λ_osc = 0.14142 ± 0.00677; at 80 mK it falls to
   0.20 / 0.17 / 0.20 / 0.17. `E_N` opens at 0.17 but only at T ≤ 27.5 mK and is
   exactly zero above. So the measurable-correlation region opens on the
   coupling axis, not the temperature axis, and it opens for discord and
   correlated coherence **before** it opens for mutual information (D and C_cc
   are smaller quantities but have proportionally lower floors).
4. **What survives the sudden-death wall is thermal.** Along temperature at the
   lowest-noise column, `E_N` falls 9.82e-4 → 0 (dead from 30 mK) while MI,
   `D^{←}` and `C_cc` rise monotonically (11 of 12 steps increasing) by
   **5.4–6.6×** to their maxima at 50 mK, and `C_r` by 2.4×. This is the
   ordering of E. G. Brown, PRA 88, 062336 (2013) ("thermal amplification of
   field-correlation harvesting"), reproduced in the circuit-QED twin. It also
   means a positive measurement of MI or discord in the device box would not be
   evidence of *harvesting*: it is correlation the temperature creates.
5. **The communication split of the MI: at the device separation the
   leading-order MI is signalling, not harvesting.** PKMM's Eq. (76) MI depends
   on the cross term `L_AB` and not on the pair term `M`, so the
   Tjoa–Martín-Martínez split passes through it. On the static-gap companions
   `|C_comm|/|C|` falls from 7.8e2 (1 site) through 1.08 (9 sites) to 4.4e-1
   (11), 1.1e-2 (12), 6.9e-5 (14) and 3.8e-12 (23 sites) — the same light-cone
   decay the sibling's `|M_comm|/|M|` shows (8.5e-1 → 1.8e-12). Inside the cone
   the anti-commutator half alone **exceeds the population bound**
   (`|C_vac|/√(P_A P_B)` up to 675), so the "harvested-only" MI is not the MI of
   a state and is reported as NaN with `vac_part_physical = False`; the split is
   physical from 9 sites out, where 99.4 % of the leading-order MI is
   signalling (MI 2.23e-6, MI_vac 1.35e-8). Outside the cone `MI_comm → 0` and
   the whole MI is harvested. The **nonperturbative** control is the
   spacelike-window subset of the separation surface (**60 rows** at ≥ 11 sites for a
   10.09-site window): max MI 4.86e-5, `D^{←}` 4.77e-5, `C_cc` 4.86e-5 nats,
   **none of them resolvable at any floor** — the MI sits **63×** below the
   median of its own model floor there. Genuinely spacelike correlation
   harvesting in this model is nearly two orders below detection.

**Status.** A claim about the **MODEL**, not the device. It inherits every input
of `circuit-qed-noise-thresholds` — including its derived coupling
(λ_osc = 0.14142 ± 0.00677, provenance `derived`, read from the parameter file
at run time), its synthetic IR mass and boundary buffer, its synthetic surface
axes and the plot-digitised Γ_φ. Nothing here re-derives them.

---

## The producing build

| item | value |
|---|---|
| generator | `papers/mi-coherence-harvesting/notebook.py`, entry point `main()` |
| generator sha256 | `edc36bf55fa47817d80be5fe9b04c5304abacc86fdf619c6ba9df0bd53260006` |
| new module | `vacuum/detectors/correlations.py`, sha256 `a036fd5c880e2dc136774c9d9e1c32be3edb7c578bd42cc04177e38e1a94d9fa` |
| sibling (imported, pinned snapshot) | `papers/circuit-qed-noise-thresholds/notebook.py`, sha256 `c203fce56422ea04fa7a480b8a3466103650821c218738660a637d45872b8219`; the repository file was byte-identical at the end of the run (`sibling_unchanged_during_run = true`) |
| `git rev-parse HEAD` at production | `ecd16cf5c22ae8b4fb5e110ab72ea4a07142c0a8` (**worktree dirty**: other agents were editing Layers 3–7 modules and the sibling notebook concurrently; this candidate imports only the pinned snapshot and the committed `vacuum.*` API) |
| produced (UTC) | 2026-09-04T15:47:21Z |
| package | `vacuum` 0.1.0 (editable) |
| environment | Python 3.13.3, numpy 2.5.2, scipy 1.18.1, macOS-14.6.1-arm64, 16 cores |
| command | `.venv/bin/python papers/mi-coherence-harvesting/notebook.py --workers 13` |
| wall clock | **1611.6 s (26.9 min)** for 591 points; 18916 CPU-s summed; slowest point 218.9 s. (The wall clock is inflated: the candidate's own pytest file ran concurrently on the same 16 cores for part of it. The compute *budget* for the map is ~15 min on a quiet machine; `--surfaces` runs a subset.) |
| pytest, this candidate | **17 passed in 55.5 s** (`tests/test_correlation_harvesting.py`), slowest test 29.8 s |
| pytest, full suite | see "Admissibility" below |

**Concurrency discipline.** The tree was being edited by other agents throughout.
The notebook therefore **pins a byte-for-byte snapshot** of the sibling notebook
at run start, imports that snapshot in the parent and in every worker, records
its sha256 in every archive, and re-reads the repository file at the end to
report whether it drifted. An earlier run of this candidate was discarded when
that check fired.

---

**Integration: full suite 1201 passed on the quiescent tree at HEAD `c18a206`, 2026-09-04** (`.venv/bin/python -m pytest -q`, twice in a row on the final tree: 1201 passed, 10 warnings in 731.91s (0:12:11) and 1201 passed, 10 warnings in 740.37s (0:12:20); with tenpy, cvxpy, jax, jaxlib, qiskit, qiskit_aer and qiskit_ibm_runtime blocked at the import system, 1026 passed, 76 skipped in 428.84s (0:07:08); all 15 notebook `--quick` paths exit 0, the slowest in 307 s). The tree is commit `c18a206` plus this pass's uncommitted edits — README, STATUS, PRIORITY, MANIFEST and draft prose reconciling the rollups to the MANIFESTs, plus one print in `interacting-vacua-first-numbers/notebook_qei_exact.py` made None-safe so its `--quick` path exits 0 (no numerical code, no data). `data/` was produced on the build recorded above; this run is the admissibility record under `papers/README.md` for the candidate's code as it stands on that tree, and a full re-run of `notebook.py` there regenerates `data/` with a new `generator_sha256`; the producing-build fields above are left as recorded.

---

## Modules used

| module | what it does here |
|---|---|
| `vacuum.protocol` (`run_protocol`, `GeneratorStep`, `ChannelStep`) | **the** code path: every one of the 591 points is a protocol run through the M-I.4 runner with its ledger and audits (reached through the sibling's `_run_at_level`). Nothing bypasses it. |
| `vacuum.detectors.correlations` (**new, this candidate**) | the measures: `gaussian_correlations` (MI, D^{←}, D^{→}, J), `gaussian_coherence` (C_r, C_cc with a two-criterion cutoff gate), `qubit_correlations` (the 4×4 side), `mutual_information_split` (the MI's communication split), `monte_carlo_floors` / `correlation_floors` (the resolvability model) |
| `vacuum.detectors.nonperturbative` | `initial_state` (passivity-audited at T = 0), `detector_block`, `detector_occupations`, `harvested_log_negativity` (mp-margin rule) |
| `vacuum.detectors.imperfections` | published rates → Strang-interleaved Gaussian channels on the detector modes |
| `vacuum.detectors.communication` | the M2.5 `M` split on the static-gap companions, and the commutator kernel this candidate reuses for the cross term |
| `vacuum.detectors.udw` / `.kernels` / `.switching` | the perturbative pair state, the lattice Wightman kernel, cos² switching |
| `vacuum.core` | `entropy`, `mutual_information`, `entanglement_hamiltonian`, `reduce`, `symplectic_eigenvalues` — the Gaussian algebra behind every measure |
| `vacuum.experiments_io` (`load_experiment`) | the **only** place published SI values become lattice units; the tomography model's shots and readout fidelity are read here at run time |
| `papers/circuit-qed-noise-thresholds/notebook.py` | imported, not copied (pinned snapshot): parameter loading, operating point, geometry, switching, the one-level runner, the dt gate |

---

## The measures, and what each one is

All in **nats**, on the two-detector block at the end of the protocol.

- **I(A:B)** — `vacuum.core.mutual_information`, wrapped.
- **Gaussian quantum discord D^{←}(A|B), D^{→}(B|A)** — the closed form of
  G. Adesso and A. Datta, PRL 105, 030501 (2010), arXiv:1003.4979, Eqs. (3)–(4),
  obtained independently by P. Giorda and M. G. F. Paris, PRL 105, 020503
  (2010), arXiv:1003.3207. **Gaussian-measurement caveat, carried on every
  number:** the conditional entropy is minimised over Gaussian measurements
  only; S. Pirandola et al., PRL 113, 140405 (2014), arXiv:1309.2215, prove
  those optimal for a large family of two-mode Gaussian states (all two-mode
  squeezed thermal states), so outside that family the reported discord is an
  **upper bound** on the unrestricted discord.
- **Classical correlation J = I − D** in the same restriction, so the
  decomposition identity holds by construction and is checked against the
  independent `vacuum.core` route.
- **C_r** — relative entropy of coherence (T. Baumgratz, M. Cramer,
  M. B. Plenio, PRL 113, 140401 (2014)) in the product of the detectors' local
  **energy (Fock) eigenbases** at the exit gap; S(ρ) exact from the symplectic
  spectrum, S(Δρ) the Shannon entropy of the joint photon-number distribution.
- **C_cc** — correlated coherence C_r(AB) − C_r(A) − C_r(B) (K. C. Tan,
  H. Kwon, C.-Y. Park, H. Jeong, PRA 94, 022329 (2016), arXiv:1603.01958): the
  part of the coherence that is not local. A product of squeezed states has
  C_r > 0 and C_cc = 0, which is why **C_cc, not C_r, is the coherence-harvesting
  statement** — see claim 2 above, where the distinction decides the verdict.

The perturbative companions carry the same five quantities computed from the
4 × 4 density matrix (discord minimised over projective measurements; PKMM
Eq. (76) MI alongside).

---

## The floors — three kinds, and why one nats value cannot serve five measures

The sibling's floor (`negativity_resolution_nats` = 1e-3, band
[7.82e-5, 0.04]) is a **concurrence** floor from the Waterloo group's own
MLE-tomography simulation (S. Ren, MSc thesis, Waterloo 2022, Sec. 4.3 /
Fig. 4.5), converted through E_N ≲ ln(1 + C). MI, discord and coherence
respond to reconstruction noise completely differently — they are non-negative
and pick up a *positive bias* on an uncorrelated state — so this candidate
derives floors for them from the **same** tomography, and reports all three:

1. **flat levels** — the sibling's derived value and band, applied to every
   measure (kept for comparability; a proxy, and labelled as one);
2. **the Monte-Carlo model** (`monte_carlo_floors`, assumptions (A1)–(A5)
   stated in the module): Ren's shots (1e7 per prerotation) and readout
   fidelity (80 %) read from `experiments/circuit_qed_harvesting/ren_2022_sec4-3.json`
   at run time → σ = 5.27e-4 (single-qubit Paulis), 8.78e-4 (two-qubit);
   independent Gaussian errors; the Smolin–Gambetta–Smith (PRL 108, 070502
   (2012)) nearest-physical-state projection in place of the iterative MLE;
   **floor = the 0.95 quantile of the measure over 500 reconstructions of the
   uncorrelated reference ρ_A ⊗ ρ_B**; the two-qubit state is the {0,1}² Fock
   block of the Gaussian state;
3. **an analytic companion** (`correlation_floors`): the measures of the state
   that carries one |ge⟩–|eg⟩ coherence at δ = E_floor/2 on top of the product
   of the marginals — the transparent δ²/P (quadratic-response) statement.

**Calibration against Ren 2022's own reported numbers** (`summary.json →
calibration`): on his separable input |eg⟩⟨eg| the model's spurious
concurrence maxes at 2.05e-3 against his plot-read Fig. 4.5(b) axis extent of
1.2e-3 (ratio 1.70 — the model is, if anything, slightly *pessimistic*), and
its E_N 0.95-quantile is 1.29e-3 against the derived 1e-3 nats floor (1.3×).
On his printed optimal-point X state (populations 0.63/0.37/0/6e-7 with
|ρ₂₃| = 5e-3; the printed matrix is not PSD — min eigenvalue −6.76e-5 — so the
nearest physical state is used, concurrence 0.0100 vs his stated 0.0087), the
model returns **46.4 %** zero reconstructions against his "nearly half", with
the non-zero range [2.4e-5, 1.12e-2] against his reported [1e-3, 1e-2].

---

## Data

| file | contents |
|---|---|
| `data/rows.npz` | 591 rows × 195 columns: four 12 × 12 surfaces (`sweep` ∈ {dephasing, relaxation, separation, coupling}, 144 each), 12 static-gap `split` companions with both communication splits, 3 finite-size `control` rows. Column names in `__columns__`, build in `__build__`. |
| `data/surfaces.npz` | the same surfaces as dense arrays: per surface, `T_mK`, the x axis, and `{E_N, MI, D_B, D_A, C_cc, C_r, J_B, J_A, n_d_A, work}` plus every per-point floor. |
| `data/summary.json` | the deliverables: `go_no_go[surface].contours[measure][level]` (row- and column-wise crossings, brackets, sudden-death flags — including the contour of *measure − per-point floor*), `device_anchors` per measure, `pivot.{device_box, per_surface, opening}`, `communication_split_rows`, `spacelike_control`, `thermal_amplification`, `calibration`, `finite_size_control`, `sibling_crosscheck`, the provenance map. |

Every row carries: parameters in published **and** lattice units; the five
measures plus J and the local coherences; ⟨n_d⟩, ledger `work`/`dissipated`;
resolvability flags at three flat levels, at the Monte-Carlo floor and at the
analytic floor, for every measure; the convergence provenance (`dt_converged`,
`halvings`, `movement`, plus the *new* measures' movement across the accepted
halving and the coherence's cutoff, criterion, gf defect and tail weight); the
audit flags; the two-qubit proxy's weight and its own measures; and, on the
companions, both communication splits.

---

## Audit and convergence status (all 591 rows)

| check | result |
|---|---|
| **ledger closure** (mandatory, every point) | all passed; worst defect **8.99e-11** (tolerance 1e-9) |
| **passivity** on the claimed T = 0 ground state | all passed, twice per point |
| **precision** on the reported detector block | all passed; `flagged` fired on 0 rows |
| **mp-margin rule** on `E_N` | live on every row, fired on 0 |
| **causality** (runner's audit) | deliberately OFF (1D-chain ordering); checked physically through `|M_comm|/|M|`, `|C_comm|/|C|` and the spacelike-window subset |
| **dt-halving gate** (E_N, ⟨n_d⟩, I(A:B), `conv_tol` = 1e-9) | converged on all 591; worst accepted movement **9.97e-10** |
| **the new measures under the same halving** (recorded, not gating) | worst discord movement **5.43e-10**; worst coherence movement **1.50e-08** |
| **Fock-cutoff gate on the coherence** | converged on all 591 (max cutoff 32); accepted by the *generating-function* criterion on **591/591** rows, worst gf defect **4.96e-12**; worst truncated-entropy defect 1.10e-10 |
| **I = J + D identity** | worst defect **5.87e-15** over all rows |
| **reproduce-the-sibling control** | 591/591 shared grid points matched against `papers/circuit-qed-noise-thresholds/data/rows.npz` (produced at HEAD `6634641`, generator sha `1394970e…`): max \|ΔE_N\| = **0.0**, max \|ΔMI\| = **0.0** — exact |
| **finite size** (buffers 6/12/24, N = 31/43/67) | relative spreads: E_N 0.31 %, MI 1.07 %, D^{←} 1.09 %, C_cc 1.14 %, C_r 0.38 % — the dominant numerical systematic, three orders above the dt gate |
| **perturbative companions** | 11 of 12 in the trustworthy regime (`perturbative_ok`); the 46-site companion's quadrature refused (analytically-zero integral below `atol`) and is flagged, not reported |
| **discord branch** (Adesso–Datta Eq. (4)) | homodyne 573, squeezed 7, uncorrelated 11 — both analytic branches are exercised on the map |

---

## Validation of the new module (`tests/test_correlation_harvesting.py`, 17 tests)

| anchor | result |
|---|---|
| Gaussian discord closed form vs brute-force minimisation over pure Gaussian seeds (r, φ), both directions, 14 random two-mode states | worst defect **6.1e-12** (gate 1e-8); both branches exercised |
| I = J + D on 60 random states | worst **2.7e-15** (gate 1e-12) |
| product states (thermal and squeezed) | every measure 0 to ≤ 1e-10; C_r > 0 with C_cc = 0 on a *squeezed* product, as it must be |
| single-mode thermal p_n from the Fock construction / from the generating function | 5.0e-11 / 4.2e-15 |
| two-mode squeezed vacuum: C_r vs the closed form H(p_nn) | **< 1e-9**; S(ρ) = 0 to 1e-12; off-diagonal photon-number mass < 1e-12 |
| generating function refereeing the Fock diagonal, cutoff 12 → 24 | error falls by ≥ 20× per doubling; covariance round trip 2.6e-6 at cutoff 32 |
| coherence basis follows the detector gaps | ground state C_r ≈ 0 in its own basis, large in the wrong one |
| two-qubit discord vs S. Luo's Bell-diagonal closed form (Werner state) | **2.2e-16**; Bell state D = MI/2 = ln 2 to 1e-12 |
| PKMM Eq. (76) vs the 4 × 4 MI | agrees to < 5 % and the defect falls ~16× per factor-4 reduction of λ² |
| tomography model | precision matches Ren's shots/fidelity exactly; the projection is PSD, unit-trace and idempotent; calibration as above |
| MI communication split | `|C_comm|/|C| < 1e-6` and `MI_comm ≤ 1e-9 × MI` at spacelike separation, > 0.05 in causal contact; `|C| ≤ √(P_A P_B)` always |
| the map | 6-row smoke through the audited path: full schema, all audits, all gates, the snapshot pinning and the derived coupling |

## Model limitations (this candidate's own, on top of the inherited ones)

1. **The discord is a Gaussian discord.** The conditional entropy is minimised
   over Gaussian measurements (Adesso–Datta / Giorda–Paris). Pirandola et al.,
   PRL 113, 140405 (2014) prove those optimal for a large family of two-mode
   Gaussian states (all two-mode squeezed thermal states); a harvested state
   need not be in that family, so `D_B`, `D_A` are **upper bounds** on the
   unrestricted discord. The two-qubit discords are minimised over projective
   measurements — general POVMs can lower them slightly, same direction.
2. **Coherence names a basis.** `C_r` is the relative entropy of coherence in
   the product of the detectors' local energy (Fock) eigenbases *at the exit
   gap*. A different basis is a different number; the basis choice is the
   physically meaningful one for an energy-resolving readout, and the
   correlated coherence `C_cc` (Tan et al. 2016) is the part of it that is not
   local — it is what "coherence harvesting" must mean, and it vanishes on
   product states while `C_r` does not.
3. **The tomography model is a model of Ren 2022's protocol, not that
   protocol.** Assumptions (A1)–(A5) are stated in
   `vacuum/detectors/correlations.py`: independent Gaussian Pauli errors with
   the binomial/visibility standard error, the Smolin–Gambetta–Smith 2-norm
   projection in place of the thesis's iterative MLE, and (for the Gaussian
   detectors) the {0,1}² Fock block as the two-qubit proxy. Its calibration
   against the two numbers the thesis reports is in `summary.json →
   calibration`.
4. **The floors are per-state, so a contour of "measure − floor" is not a
   contour of the measure.** Both are archived; the flat E_N-derived levels are
   kept alongside for comparability with the sibling.
5. **The MI communication split is perturbative and is this candidate's
   construction**, not a published estimator: TMM21's logic (the split of the
   Wightman kernel into its commutator and anti-commutator halves) applied to
   the cross term L_AB that PKMM's Eq. (76) MI depends on. The nonperturbative
   rows have no split of their own; their causal control is the
   spacelike-window subset.

## What is NOT claimed

- Nothing about the actual device's harvestable correlations. This is a claim
  about the **model**, and it inherits every input of the noise-threshold
  candidate (the coupling derivation, the synthetic IR mass and buffer, the
  synthetic surface axes, the plot-digitised Γ_φ).
- No claim that the Gaussian discord *is* the discord (measurement class).
- No claim that the Monte-Carlo floor is the Waterloo protocol's own
  sensitivity: it is a stated model of it, calibrated against the thesis's two
  reported numbers.
- Nothing beyond `'xx'` coupling, harmonic detectors, a 1+1 lattice with the
  inherited UV/IR cutoffs.

## Reproducing

```
.venv/bin/python papers/mi-coherence-harvesting/notebook.py --workers 13   # the archive
.venv/bin/python papers/mi-coherence-harvesting/notebook.py --quick        # smoke, writes to a temp dir
.venv/bin/python papers/mi-coherence-harvesting/notebook.py --from-rows papers/mi-coherence-harvesting/data/rows.npz
.venv/bin/python -m pytest tests/test_correlation_harvesting.py -q         # the module's anchors + a 6-row map smoke
.venv/bin/python -m pytest -q                                              # admissibility gate
```

`--surfaces dephasing,coupling` runs a subset; `--workers 1` runs sequentially
through the identical per-point code; `--from-rows` rebuilds `surfaces.npz` and
`summary.json` from an existing `rows.npz` without re-running any protocol.

---

## Admissibility (papers/README.md)

- **Claim status: MODEL claim only.** No device claim is made; the candidate
  inherits the sibling's input provenance in full (the derived coupling and its
  1σ, the synthetic IR mass, buffer and axes, the plot-digitised Γ_φ).
- **No optimiser numbers** (none used), so no plain-numpy round trip is owed.
- **No audit anomaly** was encountered, so the anomaly protocol did not fire.
- **The full known-results suite on the producing build: `9 failed, 1136 passed,
  7 warnings in 2925.55 s` (`.venv/bin/python -m pytest -q`, run immediately
  after production on the same worktree). The `papers/README.md` gate was
  therefore NOT met on that build; it is met by the integration record above —
  those nine failures were all one tree-wide export-registration item and were
  closed by the integration commits, this candidate's own line
  (`vacuum.detectors.correlations` in `vacuum/detectors/__init__.py`) included.**
  Every Layer-1/2/3 anchor and every standing audit passes even there;
  the nine failures are a single tree-wide integration item created by four
  agents adding new submodules concurrently:

  | failure | owner |
  |---|---|
  | `test_exports.py::test_submodule_all_is_subset_of_package_all[detectors]` | **shared**: `vacuum.detectors.circuit_mapping` (another agent) **and** `vacuum.detectors.correlations` (this candidate) are not re-exported from `vacuum/detectors/__init__.py` |
  | the same test for `composite`, `floquet`, `geometry`, `hardware`, `inequalities`, `opt` | other agents' new submodules (`fock_conditioning`, `bounds`, `composite`, `real_device`, …) |
  | `test_exports.py::test_import_with_every_extra_blocked` | `vacuum.opt.multistart` (another agent) requires JAX at import |
  | `test_circuit_mapping.py::test_geometry_holds_the_physical_buffer_fixed` | another agent's new `circuit_mapping` module |

  **This candidate's part of it is one line** — adding `correlations` (and its
  `__all__`) to `vacuum/detectors/__init__.py` — and it is deliberately NOT
  applied here: that file is owned by the concurrent `circuit_mapping` work and
  the same test cannot pass until both modules are registered, so the fix
  belongs to one integration edit rather than two colliding ones. Nothing
  numerical depends on it; `vacuum.detectors.correlations` imports and runs
  with every optional extra blocked (checked), and this candidate imports it
  by its module path.
- This candidate's own tests: **17 passed in 55.5 s**. The
  reproduce-the-sibling control (591/591 rows, max \|ΔE_N\| = \|ΔMI\| = 0.0) is
  independent evidence that the imported code path is the one the sibling's
  archive was produced with.

