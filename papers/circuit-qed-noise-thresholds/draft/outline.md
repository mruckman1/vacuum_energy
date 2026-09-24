# Draft outline — circuit-QED noise-threshold surfaces (M2.7, candidate #1, v2)

Working title: *A go/no-go map for circuit-QED entanglement harvesting at the
proposed operating point: the temperature wall, the dephasing floor, and what
mutual information can still see*

Every number in the text is a `[[data:key]]` reference into
`data/summary.json` (or a named array in `data/surfaces.npz`), never a literal.
Nothing below restates a number; the MANIFEST does, once, with its build.

## 0. Abstract (write last; the synthetic and unsourced inputs are NAMED here)

- One model, one sourced operating point (Table I scenario 2 of the 2026
  proposal: `[[data:operating_point]]`), four 16×16 surfaces, one derived
  resolvability floor with its recorded band (`[[data:resolution]]`).
- The three findings: the 27.5–30 mK temperature wall on every surface with
  the device on its wrong side; dephasing binding two orders below the
  device's range; nothing GO at the band's high edge. Then the pocket and the
  pivot verdict.
- Say in the abstract: the coupling is DERIVED from the circuit
  (`[[data:device_point_derived_coupling.coupling_map]]`, Teixidó-Bonfill Eqs. 5,
  31, 36, 37, 66 + Janzen Eqs. 6, 10 + MAP-1; `vacuum/detectors/circuit_mapping.py`)
  with a 1σ that is the paper's own rounding spread of the scenario coupling
  (`[[data:device_point_derived_coupling.coupling_map.readings]]`); the surfaces
  archive was produced at the same number under the earlier convention
  identification (`[[data:operating_point.coupling_status]]`) and the device
  point is re-run at the derived value, ±1σ and the enumerated alternatives
  (`[[data:device_point_derived_coupling.verdict]]`). What remains synthetic:
  the IR mass, the boundary buffer, the sweep grids and the surface axes; the
  device's Γ_φ is plot-digitised; and the model is the amplitude ('xx')
  stand-in for the proposal's derivative coupling
  (`[[data:device_point_derived_coupling.model_mismatch_companion]]`).

## 1. Setup — the digital twin and what goes through what

- 1+1 lattice field (unit spacing, Dirichlet, synthetic IR mass), two harmonic
  oscillator detectors, `'xx'` coupling, cos² switching = the proposal's
  cosine ramps with S_f = 0 (transcribed shape, not an approximation).
- Units: ω_ref = 2π × 7.3 GHz (Eq. 30), v = 1.2e8 m/s (Sec. II.1); one site =
  `[[data:operating_point.site_mm]]`; the window in lattice time
  `[[data:operating_point.window_lat]]`.
- The operating point, item by item, with the file and location of each
  number (`[[data:provenance]]`); why scenario 2 (the only Table I α inside the
  measured range) and why 19.2 mm (the shortest published t_d).
- Every point through `vacuum.protocol.run_protocol`: ledger, passivity,
  precision, mp-margin, dt-halving gate on E_N, ⟨n_d⟩ *and* MI.
- The resolvability floor: derived from the group's own MLE tomography
  simulation (Ren 2022) with the concurrence→E_N conversion stated; the band's
  low edge (the group's own optimal-point state) and high edge (measured
  transmon std-dev) — three contours per surface, not one.

## 2. Methods notes that belong in the paper

- Sudden death and why crossings are brackets first, interpolations second
  (`sudden_death` flag in every crossing record).
- The boundary finder (`notebook.axis_crossings` / `surface_contour`) and its
  tautology-proof unit test (exact recovery of a planar and a log-linear
  boundary, exact bracket of a step).
- Uncertainty budget, in order: (i) the amplitude-vs-derivative MODEL
  difference (`[[data:device_point_derived_coupling.model_mismatch_companion]]`:
  the perturbative companions' E_N ratio at the same λ_UDW — a model limitation,
  not an error bar, quantified once); (ii) the coupling's 1σ (4.8 %, the
  scenario coupling's reading spread) and the mapping alternatives' band
  (`[[data:device_point_derived_coupling.coupling_map.band]]`), both re-run at
  the device point; (iii) finite-size systematic `[[data:finite_size_control]]`;
  (iv) the plot-digitised Γ_φ (factor ~2); (v) the dt gate, three orders below
  (iii).
- The coupling derivation, in one paragraph with the equation chain
  (Eq. 31 → Eq. 37 with Eq. 5/36 and ℓ₀ = Z₀/v → Eq. 66; Janzen Eqs. 6/10 fix
  the normalisation as Γ₁ = λ²Ω; lattice units make every conversion factor
  unity at ω_ref = Ω⁰; MAP-1; the Golden-rule equivalence for the 'xx'
  operator), and the alternatives enumerated with their factors
  (`[[data:device_point_derived_coupling.coupling_map.alternatives]]`).
- The perturbative companions: static-gap rows carrying M_vac/M_comm
  (`[[data:communication_split_rows]]`), agreement with the nonperturbative
  E_N from 5 sites outward, refusal at 46 sites flagged.

## 3. Result 1 — the temperature wall (Figs. 1–2)

- Fig. 1: the (T, Γ_φ) surface as a heat map of E_N with the three contours
  (`surfaces.npz: dephasing/E_N`, `[[data:go_no_go.dephasing.E_N_contour]]`),
  the device anchors marked (`[[data:go_no_go.dephasing.device_anchors]]`),
  and the Orgiazzi measured floor as a reference line.
- Fig. 2: the same for (T, Γ₁) with the measured Γ₁ range shaded.
- The sentence: on every surface E_N dies between the two grid temperatures
  bracketing the fridge base; at 30 and 50 mK E_N is exactly zero at every
  device anchor and every floor in the band. Write it as a conditional on the
  coupling.
- The 20 mK design temperature: the one GO pocket near the device, at the
  band-low floor, and what it requires of Γ_φ.

## 4. Result 2 — the dephasing floor and the relaxation ceiling

- Γ_φ crossings per temperature at the band-low floor
  (`[[data:go_no_go.dephasing.E_N_contour.band_low.rows]]`) against the
  plot-digitised device range; the derived-floor surface has no GO cell —
  say why (the noiseless corner's margin vs the Γ_φ = 1e4 Hz cell).
- Γ₁ crossings (`[[data:go_no_go.relaxation.E_N_contour]]`): derived floor
  binds at ~10 MHz; band-low floor binds above the measured maximum.
- The knife edge: at T = 0 the derived coupling is GO at 1e-3 by under one
  percent, the finite-size systematic is a third of that, and −1σ of the
  coupling is NO-GO (`[[data:device_point_derived_coupling.verdict.vacuum_0mK_E_N_GO_at_derived_floor_by_coupling]]`);
  a contour drawn at exactly the derived floor at this coupling is not robust,
  and the coupling surface plus the device-point re-run are where the
  robustness is read. None of this touches the device box, where E_N is zero.

## 5. Result 3 — separation, the light cone, and the communication split (Fig. 3)

- The (T, separation) surface with the light cone (10 sites) drawn
  (`surfaces.npz: separation/E_N`, `[[data:go_no_go.separation.E_N_contour]]`).
- Derived floor: the sourced 19.2 mm is the last GO site at T ≤ 10 mK;
  band-low: 2–4 sites outside the cone with |M_comm|/|M| ≤ 5e-4 — the only
  spacelike harvesting in the model, and where it lives.
- 60 mm and 120 mm dead at every temperature; contrast with the proposal's
  derivative-coupling spacelike result (out of scope here, `'xp'`).
- The lattice-parity wiggle at 30 mK (alive at 1, 3, 4 sites, dead at 2):
  report, do not interpret beyond "lattice".

## 6. Result 4 — the coupling surface, read at the derived coupling (Fig. 4, Table 1)

- (T, λ_osc) surface (`[[data:go_no_go.coupling.E_N_contour]]`): the GO band
  in λ at each floor, its collapse above 25 mK, the strong-coupling death.
- Where the derived coupling and its band sit on it: the derived value is the
  archive's anchored value (the missing factor Ω_lat is unity at ω_ref = Ω⁰);
  the ±1σ and the alternatives span the two grid cells below it.
- Table 1: the device point re-run at six couplings × five temperatures × nine
  anchors (`[[data:device_point_derived_coupling.per_coupling_headline]]`): the
  device box (30/50 mK) has E_N = 0 at every coupling and every floor; the
  design temperature's band-low GO survives at every coupling; the T = 0
  derived-floor GO does not survive −1σ. The archive cross-check
  (`[[data:device_point_derived_coupling.archive_cross_check]]`) shows the
  same code path reproducing the surfaces' anchor cells.
- Say what the derivation changed and what it did not: the number stayed, its
  status moved from convention to derived, and the blocker moved from "the
  coupling" to "the amplitude operator" — which Sec. 6b then closes.

## 6b. Result 4b — the same map with the proposal's operator: `'xp'` vs `'xx'` side by side (Fig. 4b, Table 2)

- The derivative coupling is the exact MAP-1 transcription of Eq. (37)
  (λ_osc^xp = |λ_TB|√(2Ω_lat)); the main (T, Γ_φ) surface is re-run with it at
  the derived coupling and ±1σ (`surfaces_xp.npz: dephasing@c0/*`,
  `[[data:xp_derivative_coupling.surface_headline]]`), plus the device-point
  rows, the static-gap split companions with the derivative kernel and the
  finite-size controls, every point through `run_protocol`.
- Fig. 4b: E_N on (T, Γ_φ) for `'xp'` next to Fig. 1's `'xx'`, same three
  contours; the wall positions per Γ_φ column for both operators
  (`[[data:xp_derivative_coupling.surface_headline.c0.temperature_wall]]` vs
  `[[data:xp_derivative_coupling.xx_archive_surface.temperature_wall]]`).
- Table 2: the device box, the design temperature and the vacuum rows under
  both operators at the three couplings
  (`[[data:xp_derivative_coupling.device_point_side_by_side]]`), with the count
  of anchors whose verdict differs between the operators
  (`[[data:xp_derivative_coupling.verdict.n_device_point_verdicts_differing_from_xx]]`).
- The split companions side by side
  (`[[data:xp_derivative_coupling.communication_split_side_by_side]]`): the
  derivative model's |M_comm|/|M| at each separation against the amplitude
  model's — what "genuine harvesting" means for each operator at the
  operating point.
- Gap tracking (`[[data:gap_tracking.comparison_vs_untracked]]`): λ(t) ∝ Ω(t)^{1/2}
  (`'xp'`) and Ω(t)^{3/2} (`'xx'`) on the device-point rows; the maximum
  relative change of E_N and whether any verdict moves.
- Write the verdict for the coupling the proposal actually uses, and the
  admissibility sentence exactly as the MANIFEST states it.

## 6c. Result 4c — the UV cutoff, measured (Fig. 4c, Table 3)

- The lattice's Brillouin edge IS the model's cutoff, so it is moved by the
  lattice spacing at fixed physical geometry, with the field renormalisation
  x = √a·φ stated (`notebook.lattice_at_spacing`); a = 1 reproduces the
  fourth-pass archive bit-for-bit, and ⟨φ(0)φ(r)⟩ → K₀(mr)/2π pins the √a.
- Table 3 (`[[data:uv_cutoff_bracket.per_cutoff]]`): five cutoffs, 14.7 → 73 GHz,
  *through* the device's own 50 GHz — dephasing anchors exactly zero at every
  one, no verdict drift anywhere
  (`[[data:uv_cutoff_bracket.verdict]]`), alive anchors moving ≤ 25 %.
- Fig. 4c: the (T, Γ_φ) surface at 24.4 and 50.0 GHz beside the 14.7 GHz one
  (`surfaces: [[data:uv_cutoff_bracket.surface]]`) — the wall sits at
  [30, 35] mK and the dephasing tolerance in [1e4, 1e5] Hz at all three.
- The audit event and its handling
  (`[[data:uv_cutoff_bracket.verdict]]`, `cutoff_bracket.json → audit_event`):
  one cell of the first pass exceeded the *absolute* ledger tolerance by 1 %;
  the tolerance was not relaxed, the synthetic column was dropped. Say this in
  the methods section — it is the only audit that has ever fired in this
  candidate.

## 7. Result 5 — the best pocket per unit switching work

- `[[data:best_pockets_per_switching_work]]` at each floor; the ledger work
  column; efficiency (weak coupling, 4 sites) vs yield (λ ≈ 0.3). Upgrade path
  (M3.5): per unit work at a stated quantile.

## 8. Result 6 — the pivot question, answered with numbers (Fig. 5)

- `[[data:pivot.device_box]]`: E_N = 0, MI = 0.24× floor in the device box.
- `[[data:pivot.per_surface]]`: the MI-only fractions; the MI contours
  (`[[data:go_no_go.*.MI_contour]]`) — MI grows with T (thermal, classical) and
  with λ, E_N does not.
- Fig. 5: MI heat map on the coupling surface with the E_N derived contour
  overlaid — the region where MI is resolvable and E_N is zero.
- The verdict, with both caveats (E_N-derived floor used as an MI proxy;
  conditional on the amplitude model — the coupling itself is now derived and
  the device-point re-run holds at every value in its band).

## 9. Controls, and what would falsify this

- Two drivers agree to roundoff (`[[data:cross_check_run_harvesting]]`).
- Boundary buffers 6/12/24 (`[[data:finite_size_control]]`).
- Controls added in the fourth pass: the `'xp'` two-driver check
  (`[[data:xp_derivative_coupling.cross_check_run_harvesting]]`), the `'xp'`
  finite-size control (`[[data:xp_derivative_coupling.finite_size_control]]`),
  and the IR-mass control (`[[data:ir_mass_control]]`).
- Falsifiers, in the order they would bite: a measured (text-stated) Γ_φ below
  ~1e5 Hz; a qubit (rather than oscillator) treatment changing the ordering at
  this λ; an exponential-cutoff (rather than hard-edge) field changing the
  dephasing tolerance. *Already tested and did not falsify:* the `'xp'`
  operator, the coupling's ±1σ and four alternatives, the IR mass over a factor
  8, gap tracking, the boundary buffer, and **the UV cutoff over 14.7–73 GHz**.

## 10. What is NOT claimed

- Nothing about the device's yield beyond the amplitude model: the coupling is
  derived, but the model's operator is not the proposal's (derivative
  coupling; the perturbative companions differ by the recorded factor);
  nothing that turns the surface axes into measurements; nothing about the
  sign of λ (invisible to the oscillator model).

---

### Figure list (each must name the `data/` array and the generating call)

| # | Figure | Array | Generator |
|---|--------|-------|-----------|
| 1 | E_N on (T, Γ_φ) with three contours and device anchors | `surfaces.npz: dephasing/*`; `summary.json: go_no_go.dephasing` | `notebook.surface_specs('dephasing')`, `surface_contour` |
| 2 | E_N on (T, Γ₁) | `surfaces.npz: relaxation/*` | same, `'relaxation'` |
| 3 | E_N on (T, separation) with the light cone; \|M_comm\|/\|M\| inset | `surfaces.npz: separation/*`; `summary.json: communication_split_rows` | `'separation'`, `split_specs` |
| 4 | E_N on (T, λ_osc) | `surfaces.npz: coupling/*` | `'coupling'` |
| 5 | MI on (T, λ_osc) with the E_N derived contour overlaid | `surfaces.npz: coupling/MI` | `pivot_verdict` |
| 6 | E_N/W_drive ranking | `summary.json: best_pockets_per_switching_work` | `best_pockets` |
| T1 | Device point at the derived coupling: verdicts per coupling × temperature × anchor, three floors | `device_point.json: per_coupling`; `rows_device_point.npz`; `summary.json: device_point_derived_coupling` | `notebook.py --device-point` (`device_point_specs`, `device_point_summary`) |
| 4b | E_N on (T, Γ_φ) with the derivative coupling, three contours, device anchors; the `'xx'` wall overlaid | `surfaces_xp.npz: dephasing@c0/*` (and `@c-1s`, `@c+1s`); `summary_xp.json: surface` | `notebook.py --xp-map` (`xp_map_specs`, `xp_map_summary`) |
| T2 | `'xx'` vs `'xp'` device point: box / design / vacuum verdicts at three couplings; split companions side by side; gap tracking | `summary_xp.json: device_point_side_by_side, communication_split_side_by_side`; `gap_tracking.json` | `--xp-map`, `--gap-tracking` |

### Admissibility checklist (papers/README.md)

- [ ] Full known-results suite GREEN on the exact producing build — run and
      recorded in MANIFEST.md (828 passed / 3 failed, all in an untracked
      Layer-4 MERA module this candidate does not import); the gate is NOT
      met until those are resolved.
- [ ] Every number in the text traced to a `data/` key (no text yet).
- [x] Synthetic and unsourced inputs named for the abstract (Sec. 0).
- [x] No optimiser numbers (none used), so no round-trip re-run owed.
- [x] No audit anomaly encountered.
- [x] Coupling derived (`vacuum/detectors/circuit_mapping.py`, parameter-file
      entry `oscillator_coupling_lambda_osc_scenario2`, provenance `derived`);
      the device point re-run at it, ±1σ and the alternatives (`--device-point`).
- [x] Derivative coupling (`'xp'`) map run at the derived coupling and ±1σ;
      gap-tracking and IR-mass controls run (`--xp-map`, `--gap-tracking`,
      `--ir-control`).
- [x] Device statement: a **negative** one, stated verbatim in MANIFEST
      ("Admissibility after the `'xp'` re-run") — the device's own dephasing
      forbids resolvable harvesting, with a 100–200× margin that no remaining
      input can close.
- [x] UV-cutoff bracket run (14.7–73 GHz, through the device's 50 GHz): no
      anchor verdict is cutoff-sensitive (`--cutoff-bracket`).
- [ ] Positive device claim: NOT supported — and now blocked by nothing except
      the absence of a GO region (max E_N 2.33e-4 at the device's own cutoff,
      4.3× under the derived floor). No remaining input can create one.
