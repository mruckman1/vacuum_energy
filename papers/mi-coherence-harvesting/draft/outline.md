# Draft outline — the Layer 2 pivot: MI, discord and coherence harvesting

Working title: *What a circuit-QED harvesting experiment could still measure
when the entanglement is gone: mutual information, Gaussian discord and
coherence at the proposed operating point, against a tomography floor derived
for each*

Every number in the text is a `[[data:key]]` reference into
`data/summary.json` (or a named array in `data/surfaces.npz`), never a
literal. Nothing below restates a number; the MANIFEST does, once, with its
build.

## 0. Abstract (write last; the synthetic and derived inputs are NAMED here)

- The sibling candidate `circuit-qed-noise-thresholds` fired PLAN.md's Layer 2
  pivot trigger for `E_N` and carried one MI column. This candidate walks the
  pivot: the same four surfaces at 12 × 12 through the same audited runner,
  with **five** correlation measures per point and a floor derived for each
  from the same tomography simulation (`[[data:resolution.model_floors]]`).
- The three findings: which measure (if any) is resolvable in the device box;
  where the measurable region opens in coupling and temperature; and that the
  measure that survives the E_N sudden-death wall is *thermal* — it grows with
  temperature, which is a classical-correlation statement, not an entanglement
  one (`[[data:thermal_amplification]]`, Brown 2013's ordering reproduced).
- Named in the abstract: the coupling is now a **derived** input with a stated
  1σ (`[[data:operating_point.coupling_mode]]`,
  `[[data:operating_point.coupling_provenance]]`) — inherited, not re-derived
  here; the IR mass, the buffer and the four surface axes are synthetic; the
  device's Γ_φ is plot-digitised; the discord is minimised over **Gaussian**
  measurements only (`[[data:gaussian_measurement_caveat]]`).

## 1. Setup — what is new, and what is inherited unchanged

- Inherited verbatim by import (`notebook.load_sibling`, sha256 recorded at
  both ends of the run): parameter loading through
  `vacuum.experiments_io.load_experiment` at run time, the operating point,
  geometry, cos² switching, the one-level runner through
  `vacuum.protocol.run_protocol`, and the dt-halving gate on E_N, ⟨n_d⟩, I(A:B).
- New per point (`vacuum.detectors.correlations`): I(A:B), the Gaussian
  discords D^{←}, D^{→} and classical correlations J (Adesso–Datta Eqs. (3)–(4)
  / Giorda–Paris), the relative entropy of coherence C_r in the detectors'
  local Fock bases and the correlated coherence C_cc (Tan et al. 2016), the
  {0,1}² Fock proxy, and the per-point tomography floors.
- Why a coherence measure needs a basis, and which one is used (the detector
  energy eigenbasis at the *exit* gap); why C_cc and not C_r is the
  correlation statement (a product of squeezed states has C_r > 0, C_cc = 0).

## 2. Methods notes that belong in the paper

- The discord closed form and its **Gaussian-measurement caveat**: optimal for
  the Pirandola et al. (2014) family, an upper bound otherwise. Validation:
  brute force over pure Gaussian seeds, both branches exercised
  (`tests/test_correlation_harvesting.py`).
- The coherence's two convergence knobs: the Fock cutoff (doubling gate, with
  the measured ~tanh(r)^cutoff rate) and dt (movement across the accepted
  halving, recorded not gating: `[[data:worst_coherence_movement_dt]]`).
- **The floors, three kinds, and why one nats value cannot serve all five
  measures.** (i) the sibling's flat E_N-derived levels, kept for
  comparability; (ii) the Monte-Carlo model (A1)–(A5) — Ren 2022's shots and
  readout fidelity read from `experiments/`, Gaussian Pauli noise,
  Smolin–Gambetta–Smith projection, floor = the 0.95 quantile of the measure
  on reconstructions of the *uncorrelated* reference; (iii) the analytic
  companion (one coherence at the precision δ = E_floor/2 on the product of
  the marginals). Calibration of (ii) against Ren's own reported numbers:
  `[[data:calibration]]`.
- The bias direction: MI, discord and coherence are non-negative functions
  with a *positive* noise bias, so their floors are set by reconstruction
  noise on an uncorrelated state, not by a resolution in nats — this is the
  methodological point of the paper and the reason the pivot's naive reading
  is too optimistic.

## 3. Result 1 — the device box, per measure (Fig. 1)

- `[[data:pivot.device_box]]`: max of each measure over the anchor points and
  its ratio to each floor; `best_points` gives the point and its own floor.
- Fig. 1: bar chart of the five measures at the device box against their three
  floors (log scale), with E_N = 0 shown as the sudden-death wall.
- **The finding to state plainly:** the verdict flips with the *kind* of floor,
  not with the measure — every measure is GO at the band's low edge, NO-GO at
  the derived level, NO-GO at the model floor. Naming the floor is therefore
  part of naming the result.
- **The one exception, and why it is not a correlation.** The only device-box
  reading above its own model floor is `C_r` at the two highest-dephasing
  anchors, where the device's noise has heated the detectors to ⟨n_d⟩ ≈ 0.27;
  `C_cc` there is an order of magnitude *below* its floor. Local coherence
  created by the readout's own dephasing is not harvested coherence.

## 4. Result 2 — the four surfaces, five contours each (Figs. 2–5)

- One figure per surface: the measure's heat map with the MC-floor contour
  and the flat derived contour overlaid (`[[data:go_no_go]]`).
- The MI-only and discord-only fractions per surface and level
  (`[[data:pivot.per_surface]]`): where a correlation is resolvable and E_N is
  not, and — the new statement — where *nothing* is.

## 5. Result 3 — where the measurable region opens

- `[[data:pivot.opening]]`: per temperature, the first resolvable coupling
  (and the last tolerable Γ_φ, Γ₁, separation) for each measure and each floor.
- The coupling axis carries the derived λ's 1σ; state whether the opening is
  inside or outside it.

## 6. Result 4 — thermal amplification, and what it means for the pivot

- `[[data:thermal_amplification]]`: along temperature at the lowest-noise
  column, E_N dies while MI, D and C_cc rise — Brown (2013) PRA 88, 062336 in
  the circuit-QED twin (`increasing_fraction` = 0.909 on both noise surfaces,
  `ratio_max_over_T0` 5.4–6.6 for MI/D/C_cc and 2.4 for C_r). The correlation
  that survives the wall is the one temperature *creates*, so a positive
  measurement there is not evidence of vacuum harvesting: it is thermal
  correlation. This is the honest reading of the pivot and belongs in the
  abstract.

## 7. Result 5 — the communication split of the MI (Fig. 6)

- Perturbative rows: the cross term L_AB splits like M does (TMM21 logic
  applied to PKMM Eq. (76)); `[[data:communication_split_rows]]` gives
  |C_comm|/|C| and the MI with and without the signalling half, next to the
  sibling's |M_comm|/|M|.
- The two facts the numbers carry: the halves are real and can interfere
  destructively (|C| < |C_vac| at causal contact), and the anti-commutator
  part alone can exceed the population bound, in which case the
  "harvested-only" MI is not the MI of a state (`vac_part_physical` False
  inside the cone, True from 9 sites out).
- The quantitative statement for the paper: at the sourced 19.2 mm the
  leading-order MI is essentially all signalling; outside the light cone it is
  essentially all harvested and **63× below its own model floor**.
- The nonperturbative control: `[[data:spacelike_control]]` — rows outside the
  switching light cone, where every correlation is harvested by construction.

## 8. What is NOT claimed

- Nothing about the device's yield (the coupling derivation is the sibling's,
  and its model limitations travel).
- No claim that the Gaussian discord is *the* discord (measurement class).
- No claim that the Monte-Carlo floor is the Waterloo protocol's own
  sensitivity: it is a model of it, with (A1)–(A5) stated, calibrated against
  the two numbers the thesis reports.

### Figure list (each must name the `data/` array and the generating call)

| # | Figure | Array | Generator |
|---|--------|-------|-----------|
| 1 | device box, five measures vs three floors | `summary.json: pivot.device_box` | `notebook.pivot_verdict` |
| 2 | MI on (T, Γ_φ) with MC-floor contour | `surfaces.npz: dephasing/{MI,MI_floor_mc}` | `notebook.measure_contours` |
| 3 | D_B on (T, λ) with the E_N contour overlaid | `surfaces.npz: coupling/{D_B,E_N}` | same |
| 4 | C_cc on (T, separation), light cone drawn | `surfaces.npz: separation/C_cc` | same |
| 5 | opening curves: first resolvable λ per T, per measure | `summary.json: pivot.opening.coupling` | `pivot_verdict` |
| 6 | \|C_comm\|/\|C\| and \|M_comm\|/\|M\| vs separation | `summary.json: communication_split_rows` | `split_rows` |

### Admissibility checklist (papers/README.md)

- [ ] Full known-results suite GREEN on the exact producing build — recorded in
      MANIFEST.md.
- [ ] Every number in the text traced to a `data/` key (no text yet).
- [x] Synthetic and derived inputs named for the abstract (Sec. 0).
- [x] No optimiser numbers (none used), so no round-trip re-run owed.
- [x] No audit anomaly encountered.
- [ ] Device claim: NOT made; this is a claim about the model, and it inherits
      the sibling's input provenance.
