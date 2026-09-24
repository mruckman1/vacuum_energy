# Draft outline — Optimized harvesting waveforms under fixed energy, noise and signaling; the vacuum-to-Bell-pair exchange rate

**Status: SKELETON.** No number below has been moved out of `../data/` yet.
Per `papers/README.md`, nothing enters a draft until the full known-results
suite is green on the exact producing build and every optimized number has
survived its plain-numpy round trip. Placeholders are written
`[[data:<json path>]]` and refer to keys in `../data/summary.json`; fill them
by reading that file, never by retyping a number from a log.

---

## 0. What is and is not being claimed

- The differentiable stack (`vacuum.opt.gjax_detectors`) is a mirror, not a
  new physics engine: forward values agree with `vacuum.detectors` to
  `[[data:anchors.forward_max_dev_V]]` and gradients with finite differences
  of the numpy stepper to `[[data:anchors.grad_rel_err_lam]]` /
  `[[data:anchors.grad_rel_err_gap]]`. Every reported number is a numpy
  round-trip number (gated at 1e-9, ledger closed, mp-margin rule,
  precision audit); the JAX values appear only as `*_jax` diagnostics.
- The operating point inherits the noise-threshold candidate's provenance
  verbatim: window, separation, coupling, IR mass and buffer are SYNTHETIC;
  the 30 mK, 8.7 MHz and 1e7 Hz anchors are the device paper's (the last
  plot-digitized). `[[data:operating_point.provenance]]`.
- Three constraints define a fair comparison against the published cos^2
  protocol: equal ledger drive work (`[[data:waveforms[*].W_budget]]`), the
  same noise floor, and a communication fraction |M_comm|/|M| not above the
  baseline's `[[data:headline.signaling_bound]]` (TMM21 split on the same
  kernel, shape and smearing). The bound is enforced inside the loop by
  the second-order JAX proxy and CHECKED on the numpy split.
- Two operating points, and the difference matters. The FIRST is the
  noise-threshold candidate's **superseded v1 base point** (5.7 GHz, 6 mm,
  0.3 ns, lambda = 0.3) — where Results 1-3 live, and which is NOT the
  shipped dense map. The SECOND is the **shipped Table-I-scenario-2 dense-map
  base point** (7.3 GHz shifted -0.46 GHz along the envelope per Eq. (30),
  0.22 ns cosine ramps, 19.2 mm, the derived oscillator coupling), read from
  `experiments/` at run time so a re-derived coupling propagates:
  `[[data:scenario2_twin.operating_point]]`, `[[data:scenario2_twin.operating_point.lam_source]]`.
  Result 4 lives there and is the one that speaks to the noise map's verdict.

## 1. Anchors before any improvement claim (§ "the literature is the test suite")

- PKMM re-discovery from a generic start: interior gap optimum
  `[[data:pkmm.alpha_star]]` (bracketed `[[data:pkmm.alpha_star_bracketed]]`)
  at d/T = 6, Delta/T = 3; the digitized Fig. 2 b.1 death-line point
  d*/T = 9.803 recovered as `[[data:pkmm.death_line_beta_star]]`
  (rel. dev. `[[data:pkmm.death_line_rel_dev]]`). A gradient method started on
  the flat large-gap plateau does not move (tests/test_opt_objectives.py).
- Hotta angle: worst |theta* − closed form| over the (h, k) rows of
  `[[data:qet]]`; QEI ascent: tau0* `[[data:qei.tau0_star]]` vs numpy
  `[[data:qei.tau0_numpy]]`, ratio `[[data:qei.ratio_star]]`.
- Distillation: coded 16×16 rounds vs Eq. (7) of BBPSSW and DEJMPS,
  `[[data:anchors.distill_coded_vs_recurrence]]`.
- Inverse design: critical-chain recovery residual
  `[[data:anchors.inverse_design_recovery_residual]]`.

## 2. Result 1 — the constrained waveform gain (the headline, and the pivot)

- Table: per noise floor (`vacuum`, `30mK`, `30mK+G1+Gphi`): baseline E_N and
  work; constrained E_N, work, comm fraction, gate level, audits, admissible;
  gain factor. `[[data:headline.constrained_all_floors]]`.
- The unconstrained run beside it (`[[data:headline.unconstrained_all_floors]]`):
  what the optimizer does when the signaling bound is off — the waveform
  raises |M_comm|/|M| from the baseline's value and buys negativity with
  field-mediated signaling. Inadmissible under the bound; reported because
  it is the honest explanation of where "order-of-magnitude" gains come
  from at causal separation.
- Pivot verdict, verbatim from the archive: `[[data:headline.pivot_verdict]]`.
- **Is it the global optimum?** The archived rows were the best of TWO
  starts. The global search (`vacuum.opt.multistart`, 32 starts in the
  k <= 5 harmonic family: published, the archived optimum embedded,
  cross-entropy seeded, random at three scales; basins by waveform L2
  distance; the dt gate cap raised 16 -> 18) reports per floor the
  best / median / spread of the admissible gains, the number of distinct
  basins, and whether the archived number stands or is beaten:
  `[[data:headline.global_search]]`, per-floor detail in
  `[[data:global_search]]` (every trial, every basin, every round trip with
  its gate level and movement). Table: archived vs global per floor.
  Statement to make plainly: a multistart is evidence about the landscape,
  not a proof of global optimality.
- Figure: baseline vs optimized lam(t) at each floor (from
  `waveforms.npz`), with the ledger work and comm fraction annotated.
- The dephasing floor: the published protocol is dead at the device's
  minimum Gamma_phi (E_N = 0); state whether an admissible waveform revives
  it (`[[data:headline.constrained_all_floors.30mK+G1+Gphi]]`).

## 2b. Result 4 — the scenario-2 twin: can a waveform move a NO-GO verdict?

The noise-threshold candidate's shipped claim is that at Table I scenario 2
the **device point is NO-GO at every resolution floor**. That claim was made
for ONE waveform — the published cos^2. This section asks the optimizer's
question of it: under the SAME three constraints, at the device's own
conditions (30 mK, measured Gamma_1 = 8.7 MHz, plot-digitized minimum
Gamma_phi = 1e7 Hz), does an admissible waveform take the device point
across the derived resolution floor?

- The base point is re-derived from `experiments/` at run time and
  cross-checked against the shipped archive before anything is optimized:
  the T = 0 shifted-gap row against the map's finite-size control and the
  static-gap row against its split-X7 row,
  `[[data:scenario2_twin.crosscheck]]` (relative deviations, and whether the
  coupling matches the map's).
- Before / after at equal work, with the communication fraction:
  `[[data:headline.scenario2_twin]]` — E_N before `[[data:scenario2_twin.E_N_before]]`,
  after `[[data:scenario2_twin.E_N_after]]`, floor
  `[[data:scenario2_twin.resolution_derived]]`, band
  `[[data:scenario2_twin.resolution_band]]`, verdict
  `[[data:scenario2_twin.verdict]]`.
- The signaling caveat, stated in the text: the perturbative split has
  static gaps only, so the in-loop proxy and the numpy check are taken on
  the static-gap T = 0 sibling of the same waveform — for the baseline and
  the candidate alike (`[[data:scenario2_twin.signaling_note]]`); the
  shifted-gap device rows carry the nonperturbative decomposition of § 2c
  instead.
- Whatever the sign of the answer, the honest framing is the same: this
  moves (or fails to move) the verdict of a MODEL whose coupling, IR mass
  and buffer are declared inputs. It is not a statement about the device.

## 2c. The signaling proxy is second order — how far off is it?

The open item: the in-loop proxy and the round-trip split are both O(lambda^2),
and the operating point is not perturbative. The nonperturbative counterpart
(`vacuum.opt.multistart.nonperturbative_split`) needs no perturbative state:
the Gaussian evolution is affine, so the detectors' output cross-covariance
splits EXACTLY into the field-state part and the detector-zero-point part
(plus, on noisy floors, a channel-injected part), and the pair correlation
<a_A a_B> inherits the split.

- It reduces to the TMM21 split: |f_np - |M_comm|/|M|| = O(lambda^2),
  measured 2.6e-4 / 2.9e-5 / 2.3e-6 / 6.4e-8 at lam = 0.3 / 0.1 / 0.03 / 0.01
  on the Gate-A lattice, additivity to 1e-14, and exactly zero signal part in
  a spacelike layout (tests/test_opt_multistart.py).
- The one-number consistency check on the final waveforms:
  `[[data:headline.proxy_consistency.max_delta_final]]`, per floor in
  `[[data:proxy_consistency.summary]]`. Report the number as what it is —
  at lam = 0.3 the O(lambda^4) terms are order one, so the proxy is a
  CONSTRAINT FUNCTIONAL, not an estimate of the true signaling share.
- What survives the correction is the ORDERING, not the value:
  `[[data:proxy_consistency.summary.np_bound_holds_all]]` records whether the
  optimized waveform's nonperturbative signal share also stays at or below
  the baseline's — the property the constraint was imposed for.
  `[[data:proxy_consistency.small_lambda]]` shows the same waveform's f_np
  walking down toward the proxy as lam_max is scaled; the spacelike control
  is `[[data:proxy_consistency.spacelike]]`.

## 3. Result 2 — the vacuum-to-Bell-pair exchange rate

- `[[data:exchange_rates]]`: distilled Bell pairs (F >= 0.99) per unit
  switching work for the baseline and the admissible optimized protocol,
  DEJMPS and BBPSSW; number of rounds; the leading-order qubit projection
  (A − 1/2 = N^(2)). State plainly that the rate is tiny (2^n input pairs
  per distilled pair at fidelity 1/2 + epsilon) and that DEJMPS beats
  BBPSSW by orders of magnitude on harvested states.

## 4. Result 3 — inverse design showcase

- Recovery: `[[data:anchors.inverse_design_recovery_residual]]` at
  N = `[[data:anchors.inverse_design_recovery_N]]`.
- Designed vacua: `[[data:inverse_design.designs]]` — a prescribed block
  negativity ×1.5/2/3 of the free chain's, achieved to the stated residual,
  K positive definite by construction, passivity/precision audits of the
  designed ground state, the Lieb-Robinson velocity bound of the design,
  and the negativity profile elsewhere on the chain (what the design costs).

## 5. Limitations (carried from the noise-threshold candidate, plus)

- Oscillator detectors, 'xx' coupling, unit-spacing UV cutoff, synthetic IR
  mass; the signaling proxy and the round-trip split are second-order in
  the coupling (ratios are coupling-independent at leading order).
- The waveform family is the cos^2 envelope times a low-order Fourier
  series on the SAME window: the optimizer cannot move the support, so the
  causal status of the layout is fixed by the configuration.
- Gate levels: optimized waveforms need finer dt than the baseline
  (`[[data:waveforms[*].rows.constrained.halvings]]`); the `converged` flag is
  carried per row and an unconverged row is inadmissible. The global search
  raises the cap to 18 and records the accepted level and the movement at
  acceptance per round trip (`[[data:global_search.*.gate]]`), so "converged"
  is a number in the archive, not a flag.
- A multistart is not a proof of global optimality: 32 starts, one waveform
  family (cos^2 envelope x k <= 5 harmonics on a fixed window), one basin
  metric and one tolerance. It bounds the landscape from below and counts
  basins; it cannot certify that nothing better exists.
- The scenario-2 twin inherits every synthetic input of the noise-threshold
  candidate (the IR mass and boundary buffer are declared choices, the
  coupling's status is whatever the parameter file says at run time) and its
  signaling column is the static-gap sibling's, not the shifted-gap run's.
- Improvements below the 1e-4 equal-work tolerance are not called
  improvements: the archived optimum "stands" unless a round trip beats it
  by more than that relative margin.

## 6. Reproduction

`.venv/bin/python papers/differentiable-vacuum/notebook.py --workers 4`
(full; `--quick` for the smoke size; `--lambda-osc X` overrides the
scenario-2 coupling), producing build `[[data:build.build]]`, generator
sha256 `[[data:build.generator_sha256]]`, wall clock
`[[data:wall_clock_s]]` of which `[[data:search_wall_clock_s]]` is the
global search and the twin.
