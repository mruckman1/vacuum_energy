# Draft outline — Amplification bounds, loss thresholds and pump-enhanced harvesting in time-modulated lattice vacua

**Status: SKELETON.** No number below has been moved out of `../data/` yet.
Per `papers/README.md`, nothing enters a draft until the full known-results
suite is green on the exact producing build. Placeholders are written
`[[data:<json path>]]` and refer to keys in `../data/summary.json`; fill them by
reading that file, never by retyping a number from a log.

---

## 0. What is claimed, and what is not

Three result candidates share one engine (`vacuum.floquet`, Layer 5) and one
accounting discipline (every driven run charges its photons to the drive
through `vacuum.audits.EnergyLedger`; passivity of the *undriven* vacuum is a
standing audit).

1. **A per-cycle amplification bound, proved for all measurable drives — a
   rederivation of a folklore bound, with its priors in Sec. 2** (`vacuum/floquet/bounds.py`,
   Theorem A). For the uniformly modulated quadratic field — every normal mode
   obeying `x'' + ω_k²(1 + d s(t)) x = 0` with `|s(t)| ≤ 1` *measurable* and
   `0 ≤ d < 1` — the unstable Floquet solution of any `T`-periodic `s` satisfies
   `ln max|eig S_F| ≤ Z · artanh(d)`, `Z` = its number of zeros per period
   (`Z = 1` on the principal tongue, `Z = m` on the `m`-th), with equality iff
   `s = sgn(sin 2θ)` in the Prüfer phase, i.e. iff the drive is the quarter-period
   two-level (bang-bang) one. Nothing is assumed about periodicity in the
   underlying inequality, so non-periodic modulations obey it per zero crossing
   as well. The growth *rate* optimum (Theorem B) is a *different* bang-bang —
   up-switch `arctan λ*` before the turning point — whose Lyapunov exponent
   `ω₀ λ*(d)` exceeds the quarter-period drive's `artanh(d)/T*` by
   `[[data:bound_theorem.rate_optimum]]` (relative excess `O(d²)`: ≈ 1 % at
   `d = 0.3`, 3 % at 0.5, 13.5 % at 0.8). The cosine drive reaches
   `(π/4) d (1 + c₂ d² + c₄ d⁴)` per cycle with `c₂ = [[data:bounds.conjecture.c2]]`,
   `c₄ = [[data:bounds.conjecture.c4]]` (fit residual
   `[[data:bounds.conjecture.fit_max_residual]]`). The adversarial battery
   (five random families, gradient and cross-entropy optimisers, 
   `[[data:bound_theorem.adversarial.n_random]]` controls per family and depth on
   `[[data:bound_theorem.adversarial.M]]` exact pieces) reaches at most
   `[[data:bound_theorem.verdict.max_ratio_any_control]]` of the bound; the
   bang-bang canary attains it to `[[data:bound_theorem.verdict.bang_bang_attainment_dev]]`.
   **Still conjectured:** the extension to *non-commuting* (pattern) modulation,
   `ln max|eig S_F| / T ≤ ω_max λ*(d_eff)` with the operator-norm depth
   `d_eff = d ‖K₀^{-1/2} P K₀^{-1/2}‖` — tested to
   `[[data:bound_theorem.verdict.pattern_max_rate_ratio]]` of the bound
   (`[[data:bound_theorem.verdict.pattern_max_rate_ratio_attacked]]` under CEM
   attack), not proved.
2. **The loss threshold** `η λ_max(d)² = 1`, mapped to a quality factor at the
   principal resonance, `Q_th(d) = π / (2 ln λ_max(d))`, i.e. `d_th ≈ 2/Q` for
   the cosine drive. Platform anchors use only transcribed numbers
   (`[[data:loss_threshold.platforms]]`). (A per-period statement at the
   principal resonance; Theorem B's rate correction does not touch it.)
3. **The pump-enhanced-harvesting map** — the composite with Layer 2 — with the
   verdict `[[data:composite.verdict]]`: maximum enhancement
   `[[data:composite.verdict.max_enhancement]]` at
   `[[data:composite.verdict.at]]`, the efficiency per unit drive work relative
   to the unpumped baseline
   `[[data:composite.verdict.efficiency_ratio_best_pumped_to_baseline]]`, and,
   new, its **vacuum/communication split** and **finite-size trend**: on every
   composite row the M2.5 estimator (Tjoa–Martín-Martínez, PRD 104, 125005
   (2021)) applied to the Floquet-prepared kernel gives the communication
   fraction `|M_comm|/|M|` — `[[data:composite_split.verdict.baseline_comm_fraction]]`
   for the unpumped windows, at most
   `[[data:composite_split.verdict.pumped_comm_fraction_max]]` for any pumped
   row (`[[data:composite_split.verdict.pumped_comm_fraction_max_spacelike]]`
   on the spacelike ones) — and the `N = 16 → 32 → 64` doubling at fixed
   detector geometry shows the `N = 16` map is **not converged**: the connected
   enhancements move by factors 2–6 between 16 and 32 sites and by
   `[[data:composite_finite_size.verdict.enhancement_rel_change_last_doubling]]`
   between 32 and 64 (`[[data:composite_finite_size.table]]`), so the headline
   to quote is the `N = 64` one (≈ 10–33× in connected windows, the 156×
   baseline being dead for `N ≥ 32`) with the last-doubling change as its
   finite-size error.

Not claimed: anything about a specific device's gain (the lattice has a UV
cutoff at 2 in lattice units); anything beyond `'xx'` oscillator detectors;
the multimode bound for non-commuting patterns (tested, above); an
*exact* split — the estimator is second order in `λ` and, at `λ_max = 0.4`,
its own negativity overestimates the nonperturbative `E_N` by
`[[data:composite_split.verdict.perturbative_over_exact_E_N]]`, so the split's
*fractions* are reported, never its negativity. The earlier "4.75× in the
spacelike-window control" is withdrawn as a label: that row (`ω_mod = 1.6`,
`n_win = 2`) has `t_win = 7.85 >` separation 7 and is a connected window
(`[[data:composite_split.verdict.mislabel_note]]`); every truly spacelike
baseline is dead (`E_N = 0`) and what the pump does there is *revival*, with
the split quantifying how little of it is signalling.

## 1. Setup

- The field: `harmonic_chain_K` (periodic for the pair/momentum-gap work,
  Dirichlet for the composite), ħ = 1, block ordering, `V_vac = I/2`.
- Drives (`modulated_K`): `K(t) = K0 (1 + d f(ω_mod t))` — coupling modulation,
  the photonic-time-crystal model (Lustig et al. 2023) — with profiles cos,
  sin, triangle, square and explicit piecewise-constant pieces; mass-only,
  patterned and boundary (`dce_chain_K`, Johansson et al. 2010 Eq. (10), (17))
  variants.
- Monodromy (`floquet_map`): exact static exponentials for piecewise-constant
  profiles; CF4 commutator-free Magnus (Blanes–Moan 2006) otherwise, with a
  dt-halving gate on the map itself (`[[data:bounds.convergence]]` shows the
  tolerance sweeps). Every substep is an exact symplectic exponential.
- Observables: normal-mode occupations, traveling-wave `(k, −k)` basis,
  squeezing spectrum `r_k = arcsinh √n_k`, pair negativity (mp-margin rule).
- The four validation anchors and their measured values (from the test files,
  permanent regression tests): Mathieu edges (|tr S_F| − 2 ≈ 1e-14 on the
  characteristic curves; numerical edges vs scipy characteristic values to
  1e-8; 0/4800 chart misclassifications); `E_N(k,−k) = 2 arcsinh √n_k` to
  1e-8 (measured ≤ 1.5e-10); Wilson 2011 parabola (prefactor, shape, depth
  scaling, 50 mK stimulated factor, tolerances 3 %/3 %/0.03/5 %); ledger
  closure to 1e-9 with `W_drive = Σ ω_k n_k`.

## 2. Result 1 — the amplification bound (theorem; a rederivation with priors)

**Setting.** `x'' + ω₀²(1 + d s(t)) x = 0`, `|s| ≤ 1` measurable, `0 ≤ d < 1`;
Prüfer variables `x = r cos θ`, `p = −ω₀ r sin θ` (Prüfer, Math. Ann. 95, 499
(1926); Hartman, ODE, Ch. XI). Exactly:

```
d ln r / dt = (ω₀ d / 2) s(t) sin 2θ,        dθ / dt = ω₀ (1 + d s(t) cos²θ) ≥ ω₀(1 − d) > 0.
```

**Theorem A (per cycle).** With θ as the clock, `d ln r/dθ = g(θ, s) =
(d s sin 2θ/2)/(1 + d s cos²θ)` is monotone in `s` at every θ (a Möbius
function of `s` whose derivative has the sign of `sin 2θ`), so the pointwise
maximiser is `s* = sgn(sin 2θ)` — unique a.e., no singular arcs — and
`∫₀^π max_s g dθ = ½ ln(1+d) − ½ ln(1−d) = artanh(d)`. Hence for any solution
and any `[t₁, t₂]` over which θ advances by `nπ` (n zeros of x):
`ln r(t₂)/r(t₁) ≤ n artanh(d)`, equality iff `s = sgn(sin 2θ)` a.e.
*Corollary (Floquet).* For `T`-periodic `s` with `max|eig S_F| > 1` the leading
eigen-solution has integer winding `Z ≥ 1` and `ln max|eig S_F| ≤ Z artanh(d)`;
`Z ≤ T ω₀ √(1+d)/π`, so `Z = 1` on the principal tongue — the normalisation
"per cycle" = per zero of the unstable solution = per drive period at the
principal resonance, `Z = m` on the `m`-th tongue (test: `Z` equals the
Mathieu tongue index). The two-level result of `bang_bang_bound` (Meissner
1918) is the equality case.

**Why no Pontryagin machinery.** In the original clock the problem is
control-affine and the maximum principle gives bang-bang arcs *plus* a
possible singular set (Boscain–Sigalotti–Sugny, PRX Quantum 2, 030203 (2021),
Secs. 6.3, 9.2; for the spectral-radius objective on bilinear systems, Hochma &
Margaliot, arXiv:1407.3437). The change of clock `t → θ` (the same move that
yields Lyapunov-type criteria for Hill's equation by optimal control, Zu & Li,
Appl. Math. J. Chin. Univ. 33, 253 (2018)) turns it into a *static* pointwise
maximisation whose switching function `(d/2) sin 2θ` vanishes only at isolated
phases.

**Prior work (priority check; docs/PRIORITY.md).** The equality case — the
quarter-period two-level drive gains `ω_max/ω_min = √((1+d)/(1−d))` per half
period, i.e. `artanh(d)` in `ln r` per zero — is Meissner (1918) and textbook
(Arnold, *Mathematical Methods of Classical Mechanics*, §25: the swing with
piecewise-constant frequency). That the extremal drive of a bounded-frequency
oscillator is bang-bang is established in the swing / parametric-oscillator
optimal-control literature: Lavrovskii–Formal'skii, J. Appl. Math. Mech. 57, 311
(1993); Piccoli–Kulkarni, IEEE Control Syst. Mag. 25(4), 48 (2005); Andresen,
Hoffmann, Nulton, Tsirlin, Salamon, Eur. J. Phys. 32, 827 (2011),
doi:10.1088/0143-0807/32/3/018 (minimum time to a target energy with
`ω_min ≤ ω ≤ ω_max`, bang-bang); Salamon–Hoffmann–Rezek–Kosloff, PCCP 11, 1027
(2009); Stefanatos–Ruths–Li, PRA 82, 063422 (2010); Hochma–Margaliot,
arXiv:1407.3437 (spectral radius of bilinear systems); Zu–Li (2018) (Lyapunov,
Borg and Krein criteria for Hill's equation by optimal control). Theorem B is the
growth-rate (Lyapunov-exponent) form of that minimum-time problem. Not found
stated in this search: inequality (8) for arbitrary *measurable* `s` in its
per-zero form with the a.e. equality case, and the closed-form root equation for
`λ*(d)` with the quantified sub-optimality of the quarter-period schedule.
**Classification: POSSIBLY KNOWN** — a rederivation with citations, not a
discovery of the program; the manuscript must say so in its first paragraph.

**Theorem B (rate).** By Dinkelbach's parametric equivalence (Management Sci.
13, 492 (1967)), the maximal Lyapunov exponent is `ω₀ λ*(d)` with `λ*` the
unique root of `artanh(d/(1+λ²)) = λ τ(λ)`,
`τ(λ) = [π/2 + arctan(λ/√(1+d))]/√(1+d) + [π/2 − arctan(λ/√(1−d))]/√(1−d)`;
the optimal drive switches down at `x = 0` and up at Prüfer phase
`π − arctan λ*` (before the turning point), dwelling
`[[data:bound_theorem.rate_optimum]]` (`tau_high_over_quarter`,
`tau_low_over_quarter`). It gains `artanh(d/(1+λ*²)) < artanh(d)` per cycle
and cycles faster; the quarter-period bang-bang is never rate-optimal.
Cross-check: a free two-dwell optimisation of the Meissner trace reproduces
`λ*` to 1e-9 (`tests/test_floquet_bound.py`).

- Figure: `ln λ_max(d)` for cos / triangle / square (50 %) against `artanh(d)`
  (now the theorem's line) and `(π/4) d`; second cosine tongue
  (`[[data:bounds.conjecture.second_tongue_max_ratio_to_first]]`, consistent
  with `Z = 2` there, i.e. a bound of `2 artanh(d)` per period).
- Figure: rate optimum `λ*(d)` vs `artanh(d)/T*(d)` and the cosine rate.
- Table: the adversarial battery
  (`[[data:bound_theorem.adversarial]]`, rows in `../data/bound_adversarial.npz`):
  max ratio per family / optimiser / depth, all ≤ 1; the canary at 1.
- Table: convergence provenance per point of the profile scan (`n_steps`,
  `halvings`, `movement`, golden-section iterations) and the tolerance sweeps.
- The multimode statement: for coupling modulation the chart equals the
  single-mode law at `ω_mod/ω_k` (`[[data:bounds.lattice_check.worst_abs_diff_multiplier]]`)
  because the `2N×2N` monodromy is block-diagonal in the normal modes
  (`[[data:bound_theorem.verdict.coupling_worst_block_defect]]`), every mode
  obeying its own `Z_k artanh(d)`. Mass modulation shares the block structure
  but not the depth: mode `k` sees `d_k = d m²/ω_k²`, so the bound reads
  `Z_k artanh(d_k)` and holds only while `d_max = d m²/ω_min² < 1` (beyond it
  the softest mode inverts within the cycle — excluded regime, refused by
  `multimode_check`). Momentum-gap (tongue) widths ∝ d/2
  (`[[data:bounds.tongue_widths]]`). For non-commuting patterns the
  operator-norm-depth rate bound is a conjecture with the battery evidence
  above.

## 3. Result 2 — the loss threshold and the two regimes

- `η_th(d)`, `Q_th(d)`, and `d_th(Q)` on a log grid (`../data/loss_threshold.npz`).
- The mapping stated once: `η = exp(−2π ω / (ω_mod Q))`, `exp(−π/Q)` at the
  principal resonance; the matched multimode bath (`matched_bath_cov`) is the
  channel — it is CP and leaves the undriven vacuum exactly invariant (the
  bare site-basis bath heats it, documented).
- Microwave: Wilson's parasitic resonances `Q ~ 30–50` versus their 10 %
  modulation — `[[data:loss_threshold.platforms.wilson2011_parasitic_Q_low]]`.
- Optical: Wang et al. 2024 — non-resonant PTCs need `Δn/n ~ 1`; low-loss
  materials saturate below 1 %; the required `Q` for a bare 1 % modulation
  from this candidate's curve versus their resonant loss threshold
  `γ/ω_r < 0.05` — `[[data:loss_threshold.platforms.wang2024_resonant_loss_threshold]]`,
  `[[data:loss_threshold.platforms.wang2024_silicon_NIR]]`.
- Below threshold the field does *not* return to vacuum: it saturates at a
  steady pair-production/loss balance (the open-line DCE regime of Wilson);
  above it the Floquet multiplier wins (the resonator regime).

## 4. Result 3 — pump-enhanced harvesting (the composite)

- Protocol figure: pump `n_prep` periods, cos² window `n_win` periods, pump
  on/off during the window; the ledger's pump and switching work per point.
- Map: `E_N / E_N(baseline)` over `(ω_mod, d, n_prep, during)`; the local
  noise `n_d`; `E_N` per unit work versus the baseline.
- **The split** (`pumped_harvest_split`, `../data/composite_split.npz`). The
  Floquet-prepared Gaussian state hands the estimator the two-time kernel
  `W_ij(t,t') = [S(t) V_f S(t')ᵀ]_ij + (i/2)[S(t) Ω S(t')ᵀ]_ij` — symmetrized
  (state, Hadamard) part plus the state-independent Pauli–Jordan commutator,
  `S(t)` the *driven* window propagator when the pump stays on — evaluated on
  the window grid (cumulative Simpson on the time-ordered triangle, halving
  gate; at depth 0 it reproduces `communication_split` on `LatticeWightman` to
  1e-7). Columns per row: `|M|`, `|M_vac|`, `|M_comm|`, `|M_comm|/|M|`,
  TMM21's `N⁻/N`, the commutator's drive shift `|M_comm − M_comm^static|`.
  Reading: the unpumped connected windows are largely communication-assisted
  (`[[data:composite_split.verdict.baseline_comm_fraction]]`); the pumped
  rows are dominated by pre-existing pairs
  (`|M_vac|/|M| ≥ [[data:composite_split.verdict.pumped_vac_fraction_min_where_E_N_positive]]`
  wherever `E_N > 0`), with the pump's own modification of the retarded
  propagator recorded separately. Headline rows:
  `[[data:composite_split.verdict.headline_rows]]`.
- The causal control, restated honestly: rows with `t_win < separation`
  (flag `spacelike`) all have dead baselines; the pump revives them
  (`[[data:composite.verdict.max_E_N_spacelike]]`) with
  `|M_comm|/|M| ≤ [[data:composite_split.verdict.pumped_comm_fraction_max_spacelike]]`
  (lattice leakage level). The `ω_mod = 1.6`, `n_win = 2` rows are *connected*
  (`t_win = 7.85`).
- **Finite size** (`section_finite_size`, `../data/composite_finite_size.npz`):
  `N = 16, 32, 64`, detectors centred at separation 7, Dirichlet walls
  receding; per `(config, ω_mod)` the baseline and pumped `E_N`, the
  enhancement and the communication fraction vs `N`:
  `[[data:composite_finite_size.table]]`. The last-doubling relative change
  of the enhancement, `[[data:composite_finite_size.verdict.enhancement_rel_change_last_doubling]]`,
  is the finite-size error to quote with the headline; the driver is the
  discrete mode nearest the detector gap (`nearest_mode_gap_offset`,
  `field_n_at_gap` in the table), not the walls.
- The `N = 16` composite map (`../data/composite.npz`, 66 rows) stays as the
  parameter survey; its absolute enhancements are superseded by the `N = 64`
  values of the finite-size table wherever both exist.
- Reading: if the enhancement survives the spacelike control, the pumped field
  hands pre-existing `(k, −k)` correlations to the detectors (entanglement
  *transfer* from a squeezed field, not vacuum harvesting) — which is what the
  split now says quantitatively; the price is the pump work and the detector
  heating.

## 5. Uncertainty budget

- Gate tolerances are *not* the error bar: monodromy gate 1e-10, composite
  gate 1e-9 on `E_N`, split gate 1e-7 on `(M, M_comm, P_A, P_B)`, DCE gate 1e-8.
- Theorem A/B carry no numerical uncertainty; the battery is a check of the
  code (`max_ratio` vs the proved 1), not evidence for the theorem.
- The DCE comparison's residual is the lattice (dispersion, boundary inertia,
  `k_d L_eff⁰ = 0.25`); the `k L_eff⁰` sweep (`../data/summary.json['dce']`)
  quantifies the theory's own validity window.
- Composite: the finite-size trend of Sec. 4 (last-doubling change of the
  enhancement, `[[data:composite_finite_size.verdict.enhancement_rel_change_last_doubling_max]]`
  at worst), one detector geometry; the split is second order in `λ` and its
  negativity is off by `[[data:composite_split.verdict.perturbative_over_exact_E_N]]`
  from the exact one — quote fractions only.

## 6. Reproducing

```
.venv/bin/python papers/time-modulated-vacua/notebook.py            # full (all six sections)
.venv/bin/python papers/time-modulated-vacua/notebook.py --quick    # smoke
.venv/bin/python papers/time-modulated-vacua/notebook.py --sections bound_theorem,split,finite_size
.venv/bin/python -m pytest tests/test_floquet_*.py -q
.venv/bin/python -m pytest -q                                       # admissibility gate
```
