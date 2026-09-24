# MANIFEST — `interacting-vacua-first-numbers`

## The claim

For the lattice λφ⁴ chain `H = ½Σ[π² + m²φ² + (Δφ)²] + (λ/4!)Σφ⁴` in its symmetric
phase (bare `m = 0.5`, `λ ∈ [0, 4]`, i.e. `λ/μ_H² ≤ 4.40` against the critical
`≈ 65`), DMRG ground states in a truncated local oscillator basis give the first
numbers on how interactions reshape the entanglement structure of the vacuum and on
the quantum-energy-inequality (QEI) margin. **Anchored:** at λ = 0 the pipeline
reproduces the exact Gaussian core to ≤ 1e-6 in energy, covariance, block entropy,
disjoint-block negativity and mutual information (`data/s1_free_anchor.csv`); at
small λ the ground energy matches second-order perturbation theory (residual ∝ λ³)
and the covariance the O(λ) tadpole chain; every number is re-run at 2χ and moves by
< 1%; the passivity audit passes at every λ. **Found (entanglement):** (i) the
mutual-information decay length of the interacting vacuum falls from 0.876 (λ = 0)
to 0.512 (λ = 4) and equals, to within 1.5%, the free chain's decay length at the
self-consistent Hartree mass evaluated with the same fit (`s2c_proxy_mixedness.csv`)
— mass renormalization accounts for the MI structure; (ii) the exact 2+2-block
negativity is suppressed by interactions (4.090e-02 → 1.935e-02 at separation 1)
while its sudden-death separation does not move (E_N at separation 2 stays ≤ 3e-8,
the n_max = 6 truncation floor); (iii) the Gaussian-covariance proxy for 8+8-block
negativity is shown to be a *mixed-state* proxy (Gaussian entropy S_G(V) up to
0.176 nats; its sudden death collapses to d = 2 while the free chain at the same
Hartree mass has none inside d ≤ 14 at λ ≤ 0.5 and d ≥ 11 beyond) — reported, not
used.

**Found (the QEI margin) — one statement.** The smeared local energy
`∫f²⟨:h_x(t):⟩dt` (Gaussian `f`, `τ₀ = 0.75`, `L = 16`) has an exact lattice
infimum: the lowest eigenvalue of the smeared-energy operator `O_f = ∫dt f² h_x(t)`
built as one MPO by operator-space Heisenberg TEBD (`notebook_qei_exact.py`,
`vacuum.interacting.qei_exact`), cross-checked by an independent dense
exact-diagonalization implementation (`qei_ed`; the two agree to ≤ 1.9e-4 where the
MPO route's operator truncation is converged (generous operator knobs, measured at L = 6, n_max = 2–3, λ = 0 and 4: −7.6e-5 to −1.9e-4; at L = 6 n_max 4, L = 7 n_max 3 and L = 8 n_max 2 only archived-knob rows exist, 0.05–3.5 % off, so no ED agreement is claimed beyond L = 6, n_max ≤ 3; and at the *sweep's own* knobs the two routes agree to 0.3–1.4 % at λ = 0 but disagree by 3.4 % → 11.9 % → 39.4 % at λ = 4 as n_max goes 5 → 6 → 8, so the cross-validation holds at λ = 0 and fails at λ = 4)). It is compared
with the free chain's exact Williamson infimum at the same smearing and lattice at
three masses: the bare mass (`E_free`), the self-consistent Hartree mass (`E_H`),
and the *measured* interacting dispersion (`E_free(K_eff)`,
`notebook_qei_reference.py`, `vacuum.interacting.qei_reference`). What is
established, and at what resolution:

1. **The free bound survives.** `exact/E_free ≤ 1` at every λ > 0, falling to 0.29
   at λ = 4, at every cutoff run (`s5b`, `s5e`, `s5f`, `s5h`).
2. **Interactions renormalize the reference by a mass shift, and the sharpest
   quadratic reference is deeper than Hartree.** The interacting dispersion at
   λ = 4, measured by TEBD correlators and harmonic inversion (`s7a`), is a sharp
   quasiparticle at all 16 momenta (spectral weight ≥ 0.9986, γ/gap ≤ 1e-4) and is
   nearest-neighbour to 7e-5 (`μ_eff² = 0.879`, `J₁ = 0.999`, `|J_{r≥2}| < 1e-5`);
   the true gap² sits 3.3 % *below* the Hartree one, so the free infimum at K_eff is
   4.5 % deeper than `E_H` (0.9 % at λ = 1; `E_free(K_eff)/E_H = 1.0452`, a pure
   Gaussian, cutoff-free factor). The hypothesis "the residual is an artefact of the
   Hartree reference" is **rejected**: against the sharpest quadratic reference the
   residual is larger, not smaller.
3. **The residual beyond any quadratic reference is a first-order quartic tax —
   both routes agree, and this is the finding.** The exact O(λ) slope of the infimum
   in the free extremal state (Hellmann–Feynman + Wick,
   `qei_exact.first_order_infimum_slope`, validated against ED finite differences to
   4e-4) gives `d(1 − E_min/E_H)/dλ = +0.081` at λ = 0, L-independent (0.089 / 0.082
   / 0.081 / 0.081 at L = 6 / 7 / 8 / 16): the extremal energy rises as +0.0119 λ, of
   which +0.0045 λ is the normal-ordered ⟨φ⁴⟩ itself — **positive**, because the
   extremal state anti-squeezes φ at the sampling point — and +0.0075 λ the mean-field
   dynamics driven by the state's *own* ⟨φ²⟩, against +0.0101 λ for a Hartree reference
   driven by the vacuum's. The independent decomposition of the archived extremal
   state (`s7c`) measures the same sign directly: the quartic piece is +8.7–8.8 % of
   |E_min| at λ = 1 and +9.7–10.7 % at λ = 4, with ⟨φ_x²⟩ = 0.343 vs 0.332 in the
   vacuum — the structure of the free Williamson state, which the interaction does
   not undo. The reviewer's alternative ("Hartree absorbs O(λ), the residual is
   O(λ²) backreaction") is not what the exact first-order calculation gives.
4. **Magnitude: resolved at λ ≤ 1, BRACKETED at λ ≥ 2.** With the calibration band
   driven to 0.19 % at λ = 0 (n_max 8, χ_op 32, weight 0.5; `s5h`), the
   control-normalized residual against `E_H` is 1.4 / 1.5 / 2.6 % at λ = 0.25 / 0.5 /
   1 — 7–13 bands from zero, and required by item 3 — lying below the linear
   prediction 2.0 / 4.0 / 8.1 % by an amount consistent with a second-order
   coefficient ≈ −0.10 from both small-λ points, or with a λ-dependent cutoff floor;
   λ = 0.1 at this band would decide and was not run. The 7–13 bands measure distance from zero at the sweep's settings, not convergence in n_max: the ladder is non-monotone at every λ, including λ ≤ 1, and the ED anchor at L = 6 moves the small-λ residual by 20–30 % between n_max 3 and 4. At λ ≥ 2 **no single value is
   quoted; a bracket is** (below): the infimum is *not monotone* in n_max on either route (MPO: exact/E_H
   at λ = 4 reads 0.873, 0.905, 0.954, 0.999, 0.986, 0.976, 0.993 for n_max 6…12 at
   the archived χ_op; ED at L = 7 rises from n_max 2 to 3), the archived knobs
   (χ_op 16, weight 0.25) *inflate* the apparent residual (the MPO infimum sits
   2.7–3.5 % above the ED one at λ = 4, L = 6–7; raising χ_op / the weight / the SVD
   resolution moves exact/E_H at λ = 4 by +0.03 to +0.17), and across the settings
   run exact/E_H at λ = 4 spans 0.87–1.04, i.e. 0.84–1.00 against K_eff. Every
   λ = 4 number in `s5*` and `s7*` is therefore a number *at its knobs*: B's
   `exact/E_free(K_eff) = 0.835` (archived knobs) maps the archived 0.873 by the
   cutoff-free 1.0452 and becomes 0.913 / 0.956 / 0.950 at n_max 8 / 9 / 12 with the
   same knobs, 0.901 at (n_max 6, χ_op 64, weight 0.5) and 0.917 with the exact SVD;
   the archived-knob decomposition's quartic fraction (+10 %) is likewise a value at
   those knobs — its H-picture audit fails by 10.8 %, the same size — and only its
   sign is carried forward.

   **The λ ≥ 2 verdict: BRACKETED, not resolved** (2026-09-13, `s8_convergence_plane2.json`;
   unit 2 of the triage). Every number below is the **frozen unit-2 snapshot**
   (`unit2_snapshot`): the **15** `plane2` rows that had landed *with their own λ = 0 controls*
   when this paragraph was written, named by tag in the record so the quote stays
   reproducible while the pool keeps landing rows (the live `verdict` beside it moves).
   **Fix round 1 (2026-09-13) added the 15th tag**, `plane2_nmax10_chiop32_w0.5_lam2`: it was
   on disk when the freeze commit was made and both its own λ = 0 control and its λ = 4
   sibling were already in the list, so its omission was arbitrary — and it was the omission
   that left the n_max axis *unmeasured at λ = 2* and made "n_max is the smallest axis at both
   λ" sayable. With it the λ = 2 n_max 8 → 10 move is **5.471 pp = 28.5 bands**, the *second*
   axis, not the smallest. Adding it changes no bracket, no systematic and no c₂ range below.
   `plane2` is the (n_max, χ_op, **weight**) plane at L = 16, τ₀ 0.75, sketched SVD, each
   column carrying its own λ = 0 control and an exact-ED anchor at the same
   (χ_op, weight, λ, SVD mode) at L ≤ 6; 36 jobs are specified. The
   rule, applied by the harvest and pinned by tests: a magnitude is QUOTED only if the
   systematic — the largest spread of the control-normalized residual over the axes actually
   walked (weight, χ_op, n_max) together with the route's own same-spec reproducibility — is
   within 3 × the sweep's λ = 0 band (3 × 0.19229 %) AND at least two settings clear the
   ED-anchor gate |rel dev vs ED| ≤ 2 %. Neither λ passes:

   | λ | settings | R_ctrl vs `E_H` | R_ctrl vs `E_free(K_eff)` | systematic | anchor-qualified |
   |---|---|---|---|---|---|
   | 2 | 5 | **[−1.797 %, +4.415 %]** | **[+0.301 %, +6.385 %]** | 6.212 pp = **32.3 bands** (weight 0.5 → 1 at χ_op 32); **n_max 8 → 10 at (χ_op 32, w 0.5) 5.471 pp = 28.5 bands**; χ_op 32 → 64 1.198 pp; the route's own same-spec floor **1.941 pp = 10.1 bands** | 1 of 5 |
   | 4 | 5 | **[−34.349 %, +15.131 %]** | **[−28.541 %, +18.800 %]** | 45.879 pp = **238.6 bands** (weight 0.5 → 1 at χ_op 32); χ_op 32 → 64 32.618 pp; n_max 8 → 10 **0.728 pp = 3.8 bands**; same-spec floor 0.126 pp | 1 of 5 |

   The bracket is wider than the quantity, and at λ = 4 it spans both signs. Six things are
   measured, not argued.

   (i) **Every λ ≥ 2 row fails the harvest's own 5 % reference-consistency gate** (10 of 10
   on the frozen snapshot) — 20.30 / 198.91 / 10.96 / 15.91 / 26.31 % of |E_min| at λ = 2 and
   5.27 / 257.29 / 7.24 / 113.84 / 27.70 % at λ = 4 — and that gate was already measured anti-correlated with the true error, so it
   neither certifies nor disqualifies any row: the ED anchor is the only accuracy evidence
   there is.

   (ii) **The anchor gate clears one setting per λ, and they are different settings.** At
   λ = 2 it is (χ_op 64, weight 0.5), max |dev| **1.551 %** over 3 anchored points, R_ctrl
   **+3.217 %** vs `E_H` / **+5.212 %** vs K_eff; at λ = 4 it is (χ_op 64, weight **1**),
   max |dev| **1.672 %** over 4 anchored points at L 5 n_max 6 / 8 and L 6 n_max 5, R_ctrl
   **−1.731 %** vs `E_H` / **+2.666 %** vs K_eff. One setting per λ gives no spread, so the
   bracket and not the point estimate is the claim.

   (iii) **The verdict does not hinge on the gate's value** (`anchor_gate_sensitivity`, gates
   1 / 2 / 2.5 / 3 / 5 %). The (χ_op 64, weight 1) column at λ = 2 — which landed during this
   pass — misses the 2 % gate by 0.14 pp (max |dev| **2.138 %**, from L 5 n_max 6), so the
   gate value is the one thing that could have been tuned to an answer. It was not, and it
   would not have helped: admitting it at a 2.5 % gate qualifies **four** λ = 2 settings
   whose spread is **28.5 bands**, and at λ = 4 no gate from 1 % to 5 % qualifies more than
   one setting. **At no gate in that range does either λ resolve.**

   (iv) **The SIGN survives more of the plane than the magnitude.** Against K_eff the
   residual is positive at **5 of 5** settings *of the frozen snapshot* at λ = 2 and **4 of 5**
   at λ = 4; the single exception is (χ_op 32, weight 1), the worst-diagnosed
   row **of that snapshot** (reference inconsistency 257.3 % of |E_min|, the largest of its
   15 rows, and a 26.4 % ED anchor) — **not** the worst-diagnosed row on the tree, which is
   the λ = 2 exception named next, at 370.2 %. *On the live tree*,
   which at this commit carries 18 of the 36 `plane2` rows, the same counts read **5 of 6** and
   **4 of 6** (`verdict[*]["n_positive_keff"]`): the extra λ = 2 negative is
   (n_max 10, χ_op 32, weight 1) at **−19.637 %** vs K_eff, which fails every gate on the tree
   (control 15.5 % off 1, reference inconsistency 370.2 %, ED anchor 18.2 %). The live counts
   move as the pool lands rows; the frozen ones do not. Against `E_H` the sign is *not*
   robust — 2 of 5 positive at λ = 2, 3 of 5 at λ = 4 (frozen) — because the weight-1 columns' own
   λ = 0 controls sit 1.7 % and 8.2 % above 1. So "a residual beyond mass renormalization
   persists at λ ≥ 2" is supported against K_eff at every adequately anchored setting; its
   size is not.

   (v) **The weighting finding now rests on three (L, n_max) points, not one.** At λ = 4 the
   (χ_op 64, weight 1) column is the only one inside 2 % of exact ED at L 5 n_max 6
   (−0.517 %), L 5 n_max 8 (+0.463 % / −1.188 %) and L 6 n_max 5 (−1.672 %), while
   (χ_op 64, weight 0.5) is 34.5 % / 15.7 % / 3.4 % out at those same three points,
   (χ_op 32, weight 0.5) 11.9 % / 39.4 % / 3.4 % and (χ_op 32, weight 1) 26.4 % / 12.3 % /
   4.5 % (worst archived thread count at each point, the value the gate uses). That
   discharges the caution against a one-point requeue — and it still does not resolve the
   magnitude, because being closest to ED at L ≤ 6 is not a bound at L = 16.

   (vi) **Confrontation with perturbation theory.** The exact first-order slope
   dR/dλ = **+0.0809088** predicts R = **+16.182 %** at λ = 2 and **+32.364 %** at λ = 4;
   *both brackets lie entirely below it*, the anchor-qualified rows included. As an effective
   second-order coefficient c₂ = (R − 0.0809 λ)/λ², the landed settings give
   **[−0.0449, −0.0294]** at λ = 2 and **[−0.0417, −0.0108]** at λ = 4, against
   **−0.0679 / −0.1061 / −0.1028 / −0.0553** read off λ = 0.1 / 0.25 / 0.5 / 1. The "≈ −0.10"
   of the small-λ reading is the λ = 0.25 and 0.5 pair alone — the four λ ≤ 1 values span a
   factor 1.9 — and **on the frozen snapshot no λ ≤ 1 value overlaps either λ ≥ 2 bracket**,
   so a single quadratic with a λ-independent c₂ does not describe λ ≤ 1 and λ ≥ 2 together at
   any setting *of that snapshot*. **On the live tree this is already false** — see the
   falsification note in the unit-2 section below, where a row landed that widens the live
   λ = 2 c₂ bracket to [−0.0958, −0.0294] and so swallows the λ = 0.1 and λ = 1 values. The λ = 0.1 point (0.741 % measured against the 0.809 % prediction, a 0.35-band
   deficit) is consistent with pure first order and therefore constrains the slope, not c₂.

   **What would close it, and what it would cost** (`s8_convergence_plane2.json` →
   `unit2_snapshot.closure_cost`, hours at the plane2 wall calibration **1.5038×** measured
   over the snapshot's own **15** two-thread plane2 jobs — the live median drifts as the pool
   lands rows, so it is frozen with the snapshot; the 15th job moved the median from 1.6767×,
   which is how tightly a wall calibration over 15 jobs is determined). Completing the
   n_max 10 / 12 rungs of the λ = 4-qualified column (χ_op 64, weight 1) is **6 jobs,
   13.968 h nominal / 21.005 h calibrated**; the same for the λ = 2-qualified column
   (χ_op 64, weight 0.5); both, **12 jobs, 27.936 h / 42.010 h** of 2-thread job wall. That would *test* resolution, not deliver it: the
   criterion also requires a **second** (χ_op, weight) pair to clear the ED gate at that λ,
   which no landed setting provides, and the frozen snapshot measures the n_max 8 → 10 axis
   at **both** λ — **28.5 bands** at (χ_op 32, weight 0.5, λ = 2) and **3.8 bands** at
   (χ_op 32, weight 0.5, λ = 4) — both above the 3-band criterion (on the live tree the
   same axis at (χ_op 32, weight 1) reads 20.358 pp = 105.9 bands at λ = 2 and
   3.603 pp = 18.7 bands at λ = 4). ASSUMPTION, stated and not proved: that the (χ_op, weight) operator-truncation
   error measured against exact ED at L ≤ 6, n_max ≤ 8 transfers to L = 16, n_max 8–12. No
   exact answer exists at L = 16 at any λ > 0, so this is the only accuracy evidence there
   is, and it is why the verdict is a bracket.

5. **Scaling.** Over the τ₀ range the archived knobs resolve (0.375–0.75; the λ = 0
   control fails at τ₀ ≥ 1.0, reading 1.085 and 1.410), the residual is a roughly
   τ₀-independent fraction of the free bound: exponent −1.2 ± 0.8 against K_eff
   (−1.1 ± 1.1 against `E_H`) where the reviewer's quartic-squeezing scenario
   predicted +2. Within the resolved range the interaction shifts the QEI *constant*,
   not its exponent; the ± 0.8 is the fit's own error and does not include the
   archived-knob systematic of item 4, which is comparable to the residuals fitted.

**Status: first numbers at one mass, one lattice spacing.** The QEI margin is an
exact lattice infimum, not a variational bound; its uncertainty is the operator-MPO
truncation, calibrated at λ = 0 (0.19 % at the sweep's settings) and *measured* to be
≥ 10 % at λ = 4 (the band does not transfer): the λ = 4 residual is **bracketed**,
[−34.3 %, +15.1 %] against `E_H` across the settings run, not resolved (item 4). Admissible as a claim about the lattice
model on the integration record; not offered as a continuum statement. The out-of-cone
content of the archived extremal MPS (O(1) excitations at the cone edge and chain
ends, where `O_f` is the identity and the DMRG sweeps are blind) is a caveat on any
"footprint" reading of that state, not on the eigenvalue. No anomaly-protocol event.

## What this candidate no longer claims

Stated once, here; every other file defers to this list.

1. **"Mass renormalization accounts for 86–95 % of the reduction, with a residual
   growing with λ."** An unconverged number presented as converged (an early draft
   of §6.5.8). Replaced by item 4 above.
2. **"The whole reduction is mass renormalization" (the variational study's flat
   0.72).** An artefact of the variational family's λ-dependent shortfall (71 / 75 /
   78 % of the infimum at λ = 0 / 1 / 4, `s5b`); and item 3 shows a first-order
   residual is required.
3. **The φ⁴ shell mechanism** for the n_max ladder's irregularity. Occupation parity
   is broken by the φ_iφ_{i+1} bond and the λ = 0 control wobbles with no φ⁴ term.
4. **"The λ = 0 control is non-monotone, therefore the wobble is the MPO truncation."**
   The dense-ED route, with no MPO anywhere, shows the λ = 0 Fock ladder of the
   smeared infimum sign-alternating (+1.9e-3, −2.0e-3, −6.6e-3, +5.9e-4, −1.7e-3,
   +9.7e-5 at n_max 4…9, L = 4; and +7.9 % → −1.7 % from n_max 3 to 4 at L = 6 with
   the paper's own smearing).
5. **The Rayleigh–Ritz monotonicity premise for the smeared infimum.** Only the t = 0
   density h_x is a projection P h P; the smeared operator of the truncated chain is
   `∫f² e^{iH_trunc t} h e^{−iH_trunc t}` with `e^{−iH_trunc t} ≠ P e^{−iHt} P`, a
   different dynamical system whose infimum is bounded by the exact one on neither
   side. A plateau or a reversal in an n_max ladder of the infimum cannot be read
   either way without the operator-truncation axis held converged.
6. **"The ≈ 1.3 % λ = 0 band is the MPO truncation."** At n_max 6 it is χ_op-independent
   (χ_op 16 → 64 leaves it at 1.3–1.5 %): it is the truncated chain's own infimum, i.e.
   the Fock dynamics of item 5. Trotter error, "negligible" at λ = 0, is 1 % at λ = 4.
7. **"The residual is ≤ ~10 % at λ = 4 and consistent with zero."** Neither half is
   supported: at λ ≤ 1 it is resolved from zero and first order; at λ = 4 its size is
   not resolved at all (item 4 above), against Hartree or against K_eff.
8. **"At λ = 4 the operator truncation at the sweep's knobs is a 3–12 % effect wherever
   an exact answer exists."** (Fix round 1, written 12:25 PDT 2026-09-12; **retracted in
   fix round 2**.) False on this tree the moment the L = 5, n_max 8, λ = 4 ED rung landed
   at 13:40 PDT: the sweep-knob pair at that point reads **+39.4 %**, and the plane-knob
   pair at the same point **+42.3 %** — three to four times the quoted ceiling, at the
   sweep's *own* Fock cutoff n_max = 8. The measured statement is the opposite of a bound:
   at fixed χ_op 32 the λ = 4 MPO–ED deviation **grows** with n_max, +3.4e-2 (n_max 5)
   → +1.19e-1 (6) → +3.94e-1 (8), i.e. the route understates the magnitude of
   E_min by a factor 1.6508 at the cutoff every quoted S8 λ = 4 ratio and the first plane
   point are taken at (1.7317 at the plane's knobs). Replaced
   by the bullet "What the sweep's knobs cost at λ = 4" below and by the numbers-table row;
   pinned by `tests/test_interacting_qei_s8.py::test_lambda4_mpo_error_grows_with_nmax_at_the_sweep_knobs`.
   The correction makes item 4 harder, not easier: no λ ≥ 2 magnitude is quoted anywhere.
9. **"The exact SVD is 328× further from 1, and weight 0.5 → 1 at L = 16 is 1190× worse"**
   (S8 triage, fix round 0, written 2026-09-12). **Retracted in fix round 1 of the triage**,
   the same day. Both factors share one denominator — |0.999931 − 1| = **6.9e-05**, the
   distance of the thread-matched sketched / weight-0.5 leg from the exactly-known
   exact/E_free = 1 — and that denominator is **29× below the 0.199 % same-spec BLAS-thread
   scatter this candidate's own archive records at that very spec**
   (`s8_weight_triage.json → repeat_runs_same_spec`, L 16 n_max 8 χ_op 32 w 0.5 λ = 0:
   −2.2652090e-02 at 2 threads, −2.2697211e-02 at 4). The leg is not resolved from 1 at all,
   so the ratios are not reproducible: re-running that leg at 4 threads (1.001923) turns
   328× into **11.8×** and 1190× into **42.5×**. Replaced by the band-aware factors
   **≥ 11.8×** (exact SVD) and **≥ 42.5×** (weight 0.5 → 1), each the worse leg's smallest
   distance from 1 over the better leg's largest across all archived thread counts, archived
   as `banded_penalties_l16_lam0` and pinned by
   `tests/test_interacting_qei_s8.py::test_exact_svd_is_a_cost_and_not_an_accuracy_gain` and
   `::test_weight_1_is_cheap_and_the_L_ranking_of_the_weight_axis_reverses`. The *directions*
   are unaffected — weight 1's 8.178 % clears **its own** measured 0.029 % band by 285×, and
   the exact SVD's 2.261 % is **11.35× the *sketched* leg's** 0.199 % band (the exact-SVD
   route's own thread scatter at that spec is **unmeasured**; see item 10) — so the plane's
   retirement and the weight axis both stand. Also retracted
   with it: **"plane2 is 43.6 h against the plane's ≈ 850 h"**, which calibrated one side
   only; like for like the plane is 856 h and plane2 **73.0 h**, a **11.7×** saving, and 87 %
   of plane2's estimate is an n_max 10/12 extrapolation
   (`wall_model_calibration`, pinned by `::test_the_plane_is_retired_and_plane2_is_what_is_queued`).
10. **Three *status labels* written in fix round 1 of the S8 triage, retracted in fix
    round 2 the same day** (the numbers they decorate are unchanged; what is withdrawn is
    the claim that they are resolved measurements):
    (a) **"the exact-SVD leg is 11.4× *its own* band: resolved"** (the L 16, n_max 8, χ_op 32,
    w 0.5, λ = 0 row). That leg has **one** archived thread count; the archive itself records
    `same_spec_band_worse: null`, `worse_leg_resolved_above_its_band: null`,
    `n_threads_archived_worse: 1`. The measured statement is **11.35× the *sketched* leg's**
    0.199 % band, under the stated ASSUMPTION that the exact-SVD route's scatter is no larger.
    (b) **"the 4-thread pair of the weight comparison is thread-matched *and* band-clearing on
    both legs' denominators"** (a comment in `tests/test_interacting_qei_s8.py`). The data say
    the opposite for that row: `better_leg_resolved_above_its_band: false`, because
    `err_better_max` 1.9229e-03 < `same_spec_band_better` 1.9919e-03 (0.9653× of the band).
    The 4-thread ratio equals the band-aware bound only because that leg **is** the larger of
    the weight-0.5 leg's two archived distances from 1; **42.5276× is a lower bound**, not a
    resolved ratio.
    (c) **"the MPO route is deterministic at a fixed seed and BLAS threading moves it by
    < 1e-8"** (the tolerance rationale in the same test file, and the claim in §S8 triage that
    it "holds at λ = 0"). It is false at λ = 0 too: **0.199 %** at L 16, n_max 8, χ_op 32,
    w 0.5, λ = 0 (−2.2652090e-02 at 2 threads vs −2.2697211e-02 at 4), 2e7× the claimed 1e-8.
    The pins are tight because they read archived job files, not because the route reproduces.
    All three are pinned by `::test_exact_svd_is_a_cost_and_not_an_accuracy_gain` and
    `::test_weight_1_is_cheap_and_the_L_ranking_of_the_weight_axis_reverses`, which now assert
    the `null` / `false` flags directly, so the withdrawn wording cannot be re-derived.
11. **Three statements written in unit 2 of the S8 triage, retracted in its fix round 1 the
    same day** (again: no measured value changes; what is withdrawn is what was claimed about
    the values):
    (a) **"n_max is the smallest axis"** as a statement about *both* λ. It is true at λ = 4
    (0.728 pp = 3.8 bands) and **false at λ = 2**, where the landed n_max 8 → 10 pair at
    (χ_op 32, weight 0.5) moves the residual **5.471 pp = 28.5 bands** — second only to the
    weight axis's 32.3 bands. The statement was only sayable because the frozen snapshot
    omitted `plane2_nmax10_chiop32_w0.5_lam2`, a row that was on disk at the freeze commit and
    whose own λ = 0 control was already frozen. **The tag is now in the snapshot**, the λ = 2
    ranking is restated as weight 6.212 pp > n_max 5.471 pp > χ_op 1.198 pp, and the test that
    was supposed to pin the ranking (`::test_the_weight_axis_carries_the_item4_systematic_not_chi_op_and_not_n_max`)
    no longer passes vacuously when the n_max spread is `None` — it now requires the spread to
    exist at λ = 2. Nothing else in the snapshot moved except the wall calibration
    (1.6767× → **1.5038×**, so 46.841 h → **42.010 h** calibrated), the λ = 2 K_eff sign count
    (4 of 4 → **5 of 5**), the refcons tally (9 of 9 → **10 of 10**) and the 2.5 %-gate
    qualified count (3 → **4**, spread 28.1 → **28.5 bands**).
    (b) **"against K_eff the residual is positive at every landed setting at λ = 2"**
    (`docs/DISCOVERIES.md` and `draft/outline.md`, written without a scope). On the live tree
    it is **5 of 6**: (n_max 10, χ_op 32, weight 1) reads **−19.637 %** vs K_eff. The scoped
    statement — 5 of 5 on the frozen 15-row snapshot, 5 of 6 live — is what those files now
    carry, matching `README.md`, `docs/STATUS.md` and `docs/PRIORITY.md`.
    (c) **"the live λ = 2 c₂ bracket swallows the λ = 0.25 and 0.5 values"**. Arithmetically
    wrong: −0.1061 and −0.1028 lie **below** the live bracket's lower edge −0.09584. The values
    it does contain are **λ = 0.1 (−0.0679) and λ = 1 (−0.0553)**. The conclusion the sentence
    was making — that the no-overlap statement is false on the live tree — is unchanged.
    Also corrected in fix round 1, not a claim but a pointer: the item-4 numbers-table rows
    cited `verdict[...]` as their source while quoting `unit2_snapshot.verdict[...]`. They now
    cite the frozen path they actually quote, and each carries the live reading beside it.

12. **Two unscoped superlatives left standing by fix round 1, retracted in fix round 2**
    (2026-09-13; again no measured value changes — both are scope errors of the same class as
    item 11, caught by the verifier on this candidate's own data):
    (a) **"the row with *the tree's* worst reference inconsistency (257.3 % of \|E_min\|)"**,
    said of the λ = 4 K_eff sign exception (n_max 8, χ_op 32, w 1) in the item-4 numbers table.
    257.3 % is the worst of the **frozen snapshot's 15 rows**; the **live tree's** worst is
    **370.2 %**, at `plane2_nmax10_chiop32_w1_lam2` — the very row the same table cell names as
    the extra λ = 2 K_eff sign exception. The superlative is now written scoped, with the live
    worst beside it, and **both halves are pinned**: the frozen maximum equals the exception
    row's 2.5728792918907546, and a strictly worse live row that is *not* in the frozen tag set
    must exist (`::test_the_lambda_ge_2_sign_survives_more_of_the_plane_than_its_magnitude`).
    The old assertion `refcons_over_E_min > 2.5` could not catch this, because the 370.2 % row
    clears it too.
    (b) **"the only n_max spread the frozen snapshot measures — 8 → 10 at (χ_op 32, weight 0.5,
    λ = 4)"**, in the "What would close it" paragraph of item 4. False since fix round 1's own
    edit: adding `plane2_nmax10_chiop32_w0.5_lam2` gave the frozen snapshot that axis at **λ = 2
    as well**, at **5.471 pp = 28.5 bands**. The paragraph now states both — **28.5 bands** at
    λ = 2 and **3.784 bands** at λ = 4, both above the 3-band criterion, with the live tree's
    same axis at (χ_op 32, weight 1) reading 20.358 pp = 105.9 bands (λ = 2) and
    3.603 pp = 18.7 bands (λ = 4) — and the two band counts are pinned in
    `::test_the_weight_axis_carries_the_item4_systematic_not_chi_op_and_not_n_max`. The error was
    conservative (it understated how far closure is) but it was a false statement about the
    frozen data inside the paragraph that states the closure decision.
    (c) Not a claim of this candidate but a consequence of its commit `102a89c`, fixed here:
    `docs/STATUS.md` still said *"the a = ½ pairs have still not landed — the one remaining skip
    in `tests/test_interacting_qei_s8.py`"*. Both halves were false as of 2026-09-13: the χ_op 32
    pair is on disk and pinned, and that file plus `tests/test_exports.py` run **61 passed,
    0 skipped**. `docs/STATUS.md` now carries the negative closure (control 1.1865, refcons
    20.01 % / 175.32 %, `usable false`, χ_op 48 still queued), and the *dated* repo-wide skip
    inventories in `docs/STATUS.md` and `README.md` — which describe the 2026-09-12 integration
    run at HEAD `d4f2ed0` — now say in one clause that their third skip has since fired into a
    pin, leaving that run's own counts (1311 passed, 3 skipped) untouched as the historical
    record they are.

## Modules and inputs used

- `vacuum.interacting.phi4` — `phi4_ground_mps`, `phi4_model`, `local_boson_ops`,
  `oscillator_frequency`, `perturbative_energy`, `first_order_covariance`,
  `hartree_coupling_matrix`, `hartree_mass_squared`, `CRITICAL_RATIO_LAMBDA_OVER_MU2`
- `vacuum.interacting.observables` — `covariance_from_mps`, `block_entropies`,
  `log_negativity_mps`, `mutual_information_mps`, `gaussian_proxy_negativity`,
  `gaussian_proxy_mutual_information`, `energy_density_profile`
- `vacuum.interacting.qei_mps` — `qei_variational_mps`, `smeared_energy_mps`,
  `apply_squeeze_family`, `squeeze_family`, `optimize_free_family`
- `vacuum.interacting.qei_exact` — `qei_exact_mps`, `heisenberg_smeared_operator`,
  `local_energy_operator_dense`, `local_energy_infimum_dense`, `free_exact_infimum`,
  `occupation_weights` (the **exact** infimum: operator-space TEBD of `h_x` into one
  Hermitian MPO, then DMRG on it)
- `vacuum.interacting.qei_exact` (diagnostics A additions) — `first_order_infimum_slope`
  (the exact O(λ) slope of the infimum), the `svd_dense_limit` convergence knob
- `vacuum.interacting.qei_ed` — `qei_exact_dense`, `dense_hamiltonian`,
  `parity_sectors`, `time_kernel` (the **second implementation**: dense Heisenberg
  evolution by full eigendecomposition, parity-blocked, no Trotter step, no operator
  truncation; shares only the Fock cutoff with the MPO route)
- `vacuum.interacting.qei_ed` (S8 additions) — `qei_exact_lanczos`, `ChebyshevPropagator`,
  `SmearedOperatorMatvec`, `QEILanczosResult` (the second implementation **beyond the
  dense limit**: O_f applied on the MPO route's trapezoid grid by Chebyshev time
  evolution of each parity block, lowest eigenvalue by Lanczos; L = 6 at n_max = 6 and
  L = 5 at n_max = 8 without a dense matrix)
- `vacuum.interacting.qei_reference` — `interacting_dispersion_tebd`,
  `harmonic_inversion`, `excitation_gap_dmrg`, `fit_dispersion_couplings`,
  `keff_from_couplings`, `free_infimum_keff`, `decompose_smeared_energy` (the
  measured dispersion, the K_eff reference and the extremal-state decomposition)
- `vacuum.interacting.audits` — `passivity_audit_mps`, `relative_change`
- `vacuum.inequalities.qei` — `gaussian_f`, `sampling_operator`, `qei_minimize`,
  `qei_bound_2d` (the free exact infimum and the Flanagan continuum bound)
- `vacuum.core` — `ground_state_cov`, `entropy`, `log_negativity`,
  `mutual_information`, `mean_energy`, `reduce`
- TeNPy 1.1.1 (`.[tensor]` extra): `TwoSiteDMRGEngine`, `TEBDEngine`,
  `MPS.get_rho_segment`, `MPS.correlation_function`
- **No `experiments/` files.** External inputs are published analytic constants and
  conventions, cited at their point of use:
  - Milsted, Haegeman, Osborne, PRD **88**, 085030 (2013), arXiv:1302.5582 — the
    lattice Hamiltonian and λ/4! normalization; critical `λ/μ_R² ≈ 66`
  - Kadoh et al., JHEP **05** (2019) 184, arXiv:1811.12376 — `[λ/μ_c²]_cont = 10.913(56)`
    (λ/4 normalization; ×6 in ours)
  - Flanagan, PRD **56**, 4922 (1997), Eq. (8) — the sharp 2d constant `1/(6π)`
  - Audenaert, Eisert, Plenio, Werner, PRA **66**, 042327 (2002) — sudden death of
    small-block negativity in the harmonic chain
  - Fewster & Hollands, Rev. Math. Phys. **17**, 577 (2005); Bostelmann, Cadamuro,
    Fewster, PRD **88**, 025019 (2013) — the only interacting QEI results, cited as
    the boundary of what is proven (neither applies to φ⁴)
  No number in `data/` is transcribed from a plot or a table.

## Files

```
notebook.py                    re-runnable top to bottom, no hidden state
                               (NOTEBOOK_SMOKE=1 runs a 90 s miniature to a scratch dir)
data/build_info.json           the build the numbers came from, all parameters
data/s1_free_anchor.csv        lambda = 0 vs the Gaussian core at L = 32, n_max sweep
data/s2_sweep.csv              per-lambda table: energies, PT, Hartree mass, small-block
                               negativities, fitted decay lengths, chi-doubling worst
data/s2_mi_profiles.csv        I(d), single sites, both chi, plus the Gaussian proxy
data/s2_proxy_negativity.csv   8+8-block proxy E_N(d) and I(d), both chi
data/s2_fits.json              the decay-length fits, residuals, windows
data/s2_chi_doubling.json      every relative and absolute change, per lambda, with the
                               floors (1e-8 general, 1e-5 for the proxy negativity)
data/s2_covariances.npz        the interacting covariances behind the proxies
data/s2c_proxy_mixedness.csv   global symplectic spectrum and Gaussian entropy of the
                               interacting covariance; free-at-Hartree-mass references
                               (xi_MI with the same fit, 8+8 death separation)
data/s2b_nmax_checks.csv       n_max = 5..8 at lambda = 4
data/s3_qei.csv                the QEI table (E_var, free exact, family, Hartree,
                               bound, ratios, 2chi, dt/2, truncation)
data/s3_qei_curves.json        E(s) along both shape lines, theta*, the h(t) traces
data/s3b_qei_nmax.csv          n_max = 6, 8, 10 re-evaluation of the lambda = 1 state
data/s4_audits.csv             passivity + energy-density closure per lambda
data/summary.json              every headline number, wall times, anomaly register
draft/outline.md               the manuscript outline with all tables

notebook_qei_exact.py          the EXACT lattice infimum (the L5 closure), re-runnable
                               (--quick runs a ~2 min miniature to a scratch dir)
data/s5_build_info.json        build + every parameter of the exact-infimum run
data/s5a_free_anchor_exact.csv lambda = 0: the exact-MPO infimum vs the Gaussian
                               (Williamson) infimum, with the chi_op and weight_decay
                               ladders that decompose the residual
data/s5b_exact_vs_lambda.csv   the exact infimum vs lambda beside the variational number
                               at the SAME cutoff (the measured family gap), the free
                               bound at the Hartree mass, the extremal state's
                               physicality and its Schroedinger-picture recheck
data/s5c_breadth.csv           a second smearing width (tau0 = 0.5) and a second mass
                               (m = 0.8)
data/s5e_convergence.csv       the n_max x chi_op convergence ladder behind the
                               'is the headline converged?' answer (it is not)
data/s5e_convergence.json      per-lambda drift, bounds, and the (inapplicable) Aitken
data/s5_summary.json           headline numbers, anomaly register, headline_is_converged

notebook_qei_exact.py --diagnostics   diagnostics A (the reviewer's plan): parallel
                               subprocesses of the same file (--job), one evaluation
                               each, kept in data/jobs/<section>_<tag>.json (83 rows);
                               --no-new-jobs summarizes the finished rows only
data/s5f_nmax_ladder_lam4.csv/.json   n_max 6..12 ON THE INFIMUM at lambda = 4, archived knobs
data/s5g_calibration.csv/.json        the lambda = 0 band decomposed one knob at a time
                               (Fock, chi_op x weight, exact vs sketched SVD, Trotter,
                               accumulator; the same knobs at lambda = 4)
data/s5h_lambda_sweep.csv/.json       7-point lambda sweep at n_max 8 / chi_op 32 / weight 0.5,
                               the residual, its band, the exponent fit, the O(lambda) anchor
data/s5i_ed_control.csv/.json         dense-ED infimum at L = 6-8 vs the MPO route at the same
                               (L, n_max, lambda); ED n_max ladders; the ED small-lambda anchor
data/s5f_diagnostics_build_info.json, s5f_diagnostics_summary.json
draft/section_diagnostics_A.md the diagnostics record (§6.5.9 of the outline)

notebook_qei_reference.py      diagnostics B: the measured dispersion, K_eff, the
                               decomposition, the tau0 sweep, controls (--quick ~3 min;
                               --only <subset of abcdes>)
data/s7a_dispersion*.csv/.json omega(k) at lambda = 0 / 1 / 4: poles, weights, widths,
                               DMRG gap cross-check, deviation from free and Hartree
data/s7b_reference.csv         E_free(K_eff) beside E_H, the fitted couplings, ratios
data/s7b_exact_regenerated.csv the archived E_min regenerated to every printed digit
data/s7c_decomposition.csv     the four pieces of h_x on the extremal state, both pictures
data/s7d_tau0_sweep.csv        tau0 = 0.375 .. 1.5 at lambda = 4 with lambda = 0 controls
data/s7e_controls.csv          L = 16 -> 24; the a = 1/2 point (control FAILED, 1.173)
data/s7e_dispersion_fine.csv/.json    the a = 1/2 dispersion (complete; ratio not quoted)
data/s7_extremal_mps_lam*.npz, s7_vacuum_mps_lam*.npz   the archived extremal states and vacua
data/s7_build_info_*.json, s7_summary.json
draft/section_reference.md     the diagnostics-B record (§6.6 of the outline)

notebook_qei_exact.py --diagnostics --only s8 [--s8-parts lam01,tau0,plane,mpo_anchor,ed] [--plan]
                               S8 (Layer 5, items 4-5): one priority-ordered idempotent job pool,
                               data/jobs/s8_<tag>.json (69 jobs; --plan prints the table and the
                               wall estimate without running; --no-new-jobs summarizes what is on disk)
data/s8_weight_triage.json     S8 TRIAGE 2026-09-12: every MPO row at L <= 6 with an exact-ED
                               partner at the same (L, n_max, lambda, trapezoid grid), ordered
                               in weight_decay at fixed chi_op / chi_dmrg / SVD mode; plus the
                               L = 16 weight cost pairs and the sketched-vs-exact-SVD pairs
data/s8_plan.json              every S8 job with its knobs, priority and estimated wall (the wall model
                               and its calibration are in the file)
data/s8_convergence_plane.csv/.json   the (n_max, chi_op) plane at lambda 0 / 4 / 2 (3 at two settings):
                               n_max {8, 10, 12} x chi_op {32, 64}, weight 0.5, EXACT SVD; the lambda = 0
                               control at identical knobs and the control-normalized residual per setting
data/s8_tau0_rows.csv/.json    tau0 = 1.0, 1.5 at lambda = 4 with lambda = 0 controls at chi_op 32 and 48
                               (n_max 8, weight 0.5), and the a = 1/2 point (L 32, m 0.25, lambda 1,
                               tau0 1.5) with its control at the same two chi_op
data/s8_lam0.1.json            lambda = 0.1 at the sweep's settings beside the O(lambda) prediction
data/s8_ed_anchor.json         the matrix-free ED rungs (L = 6 n_max 3-6, L = 5 n_max 6-8, lambda 0 and 4),
                               their validation against the archived dense rows, and every MPO-vs-ED pair
                               at the same (L, n_max, lambda, tau0, grid): sweep knobs, plane knobs, generous
data/s8_diagnostics_build_info.json, s8_diagnostics_summary.json
```

Regenerate everything with:

```
.venv/bin/python papers/interacting-vacua-first-numbers/notebook.py
.venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py
.venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py --diagnostics --workers 8
.venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_reference.py
# S8 (runners share one idempotent pool; each writes only its parts' summaries; rerun to resume)
# TRIAGE 2026-09-12: part "plane" is RETIRED; "plane2" + "wt" replace it (see "S8 triage").
.venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py --diagnostics --only s8 --s8-parts lam01,tau0,plane2,wt,mpo_anchor,keff --workers 5
.venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py --diagnostics --only s8 --s8-parts ed --workers 1
.venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py --diagnostics --only s8 --s8-parts wt --workers 4          # the weight ladder alone (1 thread per worker)
.venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py --diagnostics --only s8 --s8-parts wt --no-new-jobs        # rebuild s8_weight_triage.json from disk
.venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py --diagnostics --only s8 --s8-parts plane2 --no-new-jobs    # rebuild s8_convergence_plane2.json (the item 4 verdict) from disk, ~2 s
.venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py --diagnostics --only s8 --s8-parts plane --workers 5       # the RETIRED plane, if it is ever wanted back
# fix round 2: the two same-spec 4-thread repeats that give each SVD route its own band (355 s + 159 s)
for t in exactsvd sketched; do OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py \
  --job  papers/interacting-vacua-first-numbers/data/jobs/s8_svdband_${t}_chiop16_wd0.5_thr4.spec.json \
  --job-out papers/interacting-vacua-first-numbers/data/jobs/s8_svdband_${t}_chiop16_wd0.5_thr4.json; done
.venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py --diagnostics --only s8 --s8-parts keff --workers 1          # the lambda = 2 K_eff row (4 min)
.venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py --diagnostics --only s8 --no-new-jobs   # summarize what is on disk
# (each runner sets OMP/OPENBLAS/MKL/VECLIB threads = 2 for its workers; a --s8-parts runner writes
#  s8_plan_<parts>.json / s8_diagnostics_build_info_<parts>.json and leaves the full records alone;
#  the --no-new-jobs harvest regenerates s8_plan.json, s8_convergence_plane.*, s8_tau0_rows.*,
#  s8_lam0.1.json and s8_ed_anchor.json from whatever is in data/jobs/, and launches nothing)
```

Smoke paths (each exits 0 to a scratch directory):

```
NOTEBOOK_SMOKE=1 .venv/bin/python papers/interacting-vacua-first-numbers/notebook.py          # ~90 s
.venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py --quick          # ~2 min
.venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_reference.py --quick      # ~3 min
```

## Build this candidate was produced from

| record | notebook | date (UTC) | `git rev-parse HEAD` | wall |
|---|---|---|---|---|
| `data/s1*`–`s4*`, `summary.json` | `notebook.py` | 2026-09-03T00:00:55Z | `bd846820d712260eb9a4b6fcd0fbcf2f26b81f8d` | 33.3 min (contended; ~20 min idle) |
| `data/s5_*`, `s5a`–`s5e` | `notebook_qei_exact.py` | 2026-09-04T18:56:33Z | `0fc72167f467ccb5bb38200d182ecb60043813c2` | 16.5 min (988 s) |
| `data/s5f`–`s5i`, `data/jobs/` | `notebook_qei_exact.py --diagnostics` | 2026-09-05T01:37:16Z (summary pass; jobs 2026-09-04 22:52–01:35 UTC) | `7e7a1b921d75ac2c5e2c1e88ad4fe9a6e175dfa9` (+ the then-uncommitted diagnostics code, committed as `d83cdc3`) | 83 evaluations, 28.1 h of summed job wall over ≈ 2.7 h on a machine shared with diagnostics B and other runs (load 12–16× the 16 cores; idle-equivalent ≈ 6–8 h); s5f 93 min, s5g 798 min, s5h 365 min, s5i 433 min summed |
| `data/s7b`–`s7e` | `notebook_qei_reference.py --only bcde` | 2026-09-04T23:05:21Z | `6d8a2dd46d5a93033be9c7d72f87ceb94b39a816` | per-row times in the CSVs (a = ½ control 2296 s) |
| `data/s7a*`, `s7_summary.json` | `notebook_qei_reference.py --only a`, `--only s` | 2026-09-04T23:22:24Z, 2026-09-05T01:06:17Z | `b4eb0b38df29824d510802ea8b9723100623a990` | `s7e_dispersion_fine`: 4091 s |

Details of the first row:

- Date: **2026-09-03T00:00:55Z**
- Build: `git rev-parse HEAD` = `bd846820d712260eb9a4b6fcd0fbcf2f26b81f8d` (uncommitted working tree: the
  `vacuum/interacting/` package, `tests/test_interacting.py` and this directory are
  new on top of it; nothing else in the tree was modified)
- Package: `vacuum` 0.1.0, editable install; Python 3.13.3, numpy 2.5.2,
  TeNPy 1.1.1, macOS-14.6.1-arm64-arm-64bit-Mach-O
- Notebook wall time: **33.3 min (1995 s)** against a 25 min budget — measured while up to
  eight other Python jobs (two full pytest suites and another candidate's notebook)
  ran on the same laptop; sections: S1 118 s, S2 811 s, S3 1063 s, S4 2 s. From the uncontended
  calibration timings (DMRG L = 32 ≈ 10 s, one TEBD trajectory ≈ 5 s) the
  uncontended estimate is ~20 min; not re-measured.
- `data/s2c_proxy_mixedness.csv` and the proxy-floor regating of
  `data/s2_chi_doubling.json` were produced by the S2c code of the shipped
  `notebook.py` applied to the shipped `s2_covariances.npz` / `s2_proxy_negativity.csv`
  (pure numpy on stored arrays, deterministic) after the run above; a re-run of the
  notebook regenerates them byte-for-byte except timestamps.

## Admissibility (per `papers/README.md`)

**Full known-results suite on the producing build:**

```
$ .venv/bin/python -m pytest -q            # whole tests/ tree as found on the shared working tree
2 failed, 882 passed in 421.89s (0:07:01)
  FAILED tests/test_harvest_teleport.py::test_surplus_bounded_by_landauer_and_full_joint_ergotropy
  FAILED tests/test_harvest_teleport.py::test_classically_correlated_state_has_surplus_without_negativity
  (both in another candidate's in-progress, uncommitted vacuum/composite work — not
   imported, used or touched by this candidate)

$ .venv/bin/python -m pytest -q tests/test_<the 23 pre-existing files> tests/test_interacting.py
611 passed in 239.86s (0:03:59)     # every known-results anchor + this candidate's 13 tests, green
```

`tests/test_interacting.py` (13 tests, 207 s under 8 concurrent jobs; 65–87 s on a lightly loaded laptop) is part of that
run; the pre-existing 561 tests are untouched.

**Test files added by the L5 closure and the two diagnostics passes** (each run on
its own on the tree it was produced on):

```
tests/test_interacting_qei_exact.py     10 passed in 24.4 s   (the exact infimum; + the sketch-vs-exact
                                                               SVD control and the O(lambda) anchor)
tests/test_interacting_qei_ed.py         8 passed in 674 s under 12x oversubscription (~2 min idle)
                                                              (the second implementation; the
                                                               not-Rayleigh-Ritz ladder; slope vs ED)
tests/test_interacting_qei_reference.py  7 tests, diagnostics B (commit 62bf22f)
tests/test_interacting_qei_s8.py         6 passed, 4 skipped (the S8 harvest records; the 4 skips are
                                                              chi_op 48, a = 1/2 and the two plane tests,
                                                              which skip until their rows land)
```

**Fix round 1, 2026-09-12** (S8 harvest, after the verifier findings): the three files above
rerun together give **`36 passed, 1 skipped in 373.33 s`** at 14:02 PDT — `test_interacting_qei_ed.py`
is now 16 tests (`test_s8_L6_lambda4_ed_ladder_is_still_rising_at_nmax_5`,
`test_s8_L5_ladders_converge_at_lambda_0_and_do_not_at_lambda_4` and
`test_s8_lambda4_L5_nmax6_rung_is_pinned_with_its_coarse_quadrature` added; the S8 rungs L 6
n_max 5 λ 0/4, L 5 n_max 6 λ 0, L 5 n_max 7 λ 0/4 and L 5 n_max 8 λ 0 pinned), and
`test_interacting_qei_s8.py` is 10 passed / 1 skipped (only the a = ½ pairs still skip);
`notebook_qei_exact.py --quick` exit 0 in 191.0 s. `tests/test_exports.py` read 31 passed at
12:26 PDT and **1 failed, 30 passed** at 13:32 — the failure is
`vacuum.geometry.jacobson2d` (12 public names not reachable from `vacuum.geometry`), a module
of the toy-jacobson L7 track committed in `3c5dee2`, untouched by this candidate and left to
its owner.

**Fix round 2, 2026-09-12** (S8 harvest, after the second verifier pass): FINAL STATE — the
same three files rerun together give **`39 passed, 1 skipped in 267.27 s`** at 15:39 PDT
(`test_interacting_qei_s8.py` 13 passed / 1 skipped, with
`test_chi_op_ladder_at_nmax8_and_what_it_says_about_the_refcons_gate` added for the knob
ladder), `tests/test_exports.py` **31 passed**, and `notebook_qei_exact.py --quick` exit 0 in
163.1 s. The intermediate state of this round (before the knob ladder landed) gave
**`38 passed, 1 skipped in 370.79 s`** at 14:45 PDT
(`test_interacting_qei_s8.py` 12 passed / 1 skipped — only the a = ½ pairs still skip — with
`test_lambda4_mpo_error_grows_with_nmax_at_the_sweep_knobs` and
`test_control_chi_op_sequences_are_pinned` added, and the vacuous `generous` level assertion
replaced by pins on `worst_rel_dev_by_knobs`); `notebook_qei_exact.py --quick` exit 0 in
177.5 s; `tests/test_exports.py` **31 passed** (the `vacuum.geometry.jacobson2d` breakage of
round 1 was fixed by its owner). MUTATION CHECK of the new pins, in a scratch copy of the
tree (the round-1 suite passed all three of these unchanged): (a) setting both L = 5,
n_max 8, λ = 4 deviations to 0.010 and recomputing the derived fields now fails 2 tests
(`test_mpo_vs_ed_pairs_are_pinned_…`, `test_lambda4_mpo_error_grows_with_nmax_…`);
(b) shrinking only `worst_rel_dev_by_knobs.sweep_knobs` to 0.12 fails 1; (c) fabricating an
improved τ₀ = 1.0 χ_op 48 control (1.0384 → 1.0100 with its refcons 7.65 % → 3 %) fails 2.

**Integration: full suite 1201 passed on the quiescent tree at HEAD `c18a206`, 2026-09-04** (`.venv/bin/python -m pytest -q`, twice in a row on the final tree: 1201 passed, 10 warnings in 731.91s (0:12:11) and 1201 passed, 10 warnings in 740.37s (0:12:20); with tenpy, cvxpy, jax, jaxlib, qiskit, qiskit_aer and qiskit_ibm_runtime blocked at the import system, 1026 passed, 76 skipped in 428.84s (0:07:08); all 15 notebook `--quick` paths exit 0, the slowest in 307 s). The tree is commit `c18a206` plus this pass's uncommitted edits — README, STATUS, PRIORITY, MANIFEST and draft prose reconciling the rollups to the MANIFESTs, plus one print in `interacting-vacua-first-numbers/notebook_qei_exact.py` made None-safe so its `--quick` path exits 0 (no numerical code, no data). `data/` was produced on the build recorded above; this run is the admissibility record under `papers/README.md` for the candidate's code as it stands on that tree, and a full re-run of `notebook.py` there regenerates `data/` with a new `generator_sha256`; the producing-build fields above are left as recorded.

## S8 (Layer 5, items 4–5): design, assumptions, and the launch record

Written by the launch unit on 2026-09-11; the harvest unit that follows fills the numbers.

- **What was launched.** One idempotent job pool (`data/jobs/s8_<tag>.json`, 69 jobs,
  `data/s8_plan.json`) in two runners: pool A = the MPO parts (`lam01`, `tau0`, `plane`,
  `mpo_anchor`; 5 workers × 2 BLAS threads = 10 cores) and pool B = the ED part (1 worker
  × 2 threads). A finished job is never recomputed; rerunning either command resumes.
- **Ordering.** Jobs carry a `priority`; the pool runs highest first and longest-first
  within a priority, so the pairs a verdict needs land first: λ = 0.1 (est. 0.2 h), the
  plane at (n_max 8, χ_op 32) with its λ = 0 control (1.2 h each), the τ₀ rows at χ_op 32,
  the plane at (10, 32) (3.7 h each), the L = 5–6 MPO anchor rows where ED is exact, the τ₀
  rows at χ_op 48, (8, 64) (8.1 h each), the a = ½ point, (12, 32) (9.5 h), λ = 2 at the
  cheap settings, (10, 64) (26 h each), λ = 3, and last (12, 64) (**71 h each**, three
  jobs) and λ = 2 at the two χ_op 64 corners. The plane's χ_op 64 exact-SVD column is
  340 of the pool's 418 estimated hours; the harvest unit may kill the pool once the
  cheaper settings have decided the question, and the plan file says what each remaining
  job would cost.
- **Wall model** (2 threads, idle; in `s8_plan.json`): base = 900 s ((n_max+1)/9)⁴
  (χ_op/32)^1.5 (n_steps/8) (L/16), plus for the exact SVD 0.75 × n_updates × 12.3 s ×
  (χ_op (n_max+1)² / 2592)³. Calibration on 2026-09-11: the s5h (8, 32, sketched) point
  ≈ 900 s idle (3000 s under the 12–16× load of the archived run); a complex SVD of a
  2592² block 12.3 s, of 3872² 43.3 s at 2 threads (Accelerate); 11 Trotter layers per
  order-4 step; the light-cone window gives 363 bond updates per L = 16, τ₀ = 0.75 trajectory,
  about three quarters of them at full χ_op. The exponents (4 in n_max+1, 1.5 in χ_op) are
  fitted to the archived s5f/s5g walls (the notebook's earlier (n_max+1)⁶ (χ_op/16)² model
  overestimates both). It is an estimate; the measured `time_s` of every job is in its row.
- **Knobs of the plane.** weight 0.5, exact SVD (`svd_dense_limit = inf`), χ_acc = 2 χ_op,
  χ_mpo = 4 χ_op, χ_dmrg 16, χ_gs 16, dt 0.375 order 4, pad 2, L 16, τ₀ 0.75, m 0.5 — the
  sweep's settings with the SVD made exact and the two operator knobs walked. ASSUMPTION:
  "λ = 3 at the two best knob settings" was specified before any plane row existed; it is
  run at (10, 32) and (8, 64), the two mid-cost settings that span both axes, not at the
  (12, 64) corner (71 h per job).
- **τ₀ rows.** n_max 8, weight 0.5, sketched SVD (the sweep's), χ_op 32 and 48 (χ_acc 2 χ_op,
  χ_mpo 4 χ_op), dt 0.375 for τ₀ ≥ 0.75 exactly as `s7d` (`_dt_for`). CORRECTION (fix round 1,
  2026-09-12): an earlier version of this line said the rows "extend the archived sweep with
  only χ_op changed (16 → 32 / 48)". That is false and is withdrawn: the archived τ₀ ≥ 1 rows
  are `s7d`, run by `notebook_qei_reference.py` at **n_max 6, χ_op 16, χ_acc 32, χ_mpo 64,
  χ_dmrg 12, weight 0.25**, while the S8 rows are at **n_max 8, χ_op 32/48, χ_acc 2 χ_op,
  χ_mpo 4 χ_op, χ_dmrg 16, weight 0.5** — six knobs apart, and `s5g` shows n_max 6 → 8 alone
  moves a λ = 0 reading by ≈ 0.4 % at τ₀ = 0.75. The axis is separated by measurement instead:
  the `tau0_iso` rows run the **S8 recipe at χ_op 16** (n_max 8, weight 0.5, χ_acc 32, χ_mpo 64,
  χ_dmrg 16), so (s7d → tau0_iso) is the (n_max, weight, χ_dmrg) step and (tau0_iso → χ_op 32)
  is the operator-bond step. See the S8 harvest section for the measured split.
- **The a = ½ point.** `phi4.py` has no lattice-spacing knob (unit spacing throughout);
  the point is realized as `s7e` did, by rescaling the bare physics: L = 32, m = 0.25,
  λ = 1, τ₀ = 1.5 lattice units (m a, λ a², τ₀ / a), with the λ = 0 control at the same
  (L, m, τ₀), at χ_op 32 and 48, n_max 8, weight 0.5.
- **The ED anchor.** `qei_exact_lanczos`: the MPO route's trapezoid grid (dt 0.375, 8
  steps) applied by Chebyshev evolution (Bessel-truncated at 1e-16; a step is exact to
  1e-15 against `expm`), ARPACK Lanczos with ncv 80 and residual tol 1e-10 (measured at
  L = 6, n_max = 3: 842 → 442 O_f applications on the odd block with the eigenvalue 6e-12
  from the dense row), warm-started from the block's H ground state plus a fixed-seed
  random admixture (a pure reflection eigenvector at odd L misses the other reflection
  sector: the odd block at L = 5 came out 4.6e-4 high before the admixture; caught by
  `tests/test_interacting_qei_ed.py`). Each rung is re-solved on the dt 0.125 grid in the
  winning block (the quadrature check). Rungs: L = 6 n_max 3–6 and L = 5 n_max 6–8 at
  λ = 0 and 4; the dense route at L = 6 n_max 5 as the cross-validation at 23 328 states
  per block. The comparison rows on the MPO side (`mpo_anchor`): the sweep's knobs at
  L = 5 (n_max 6, 8) and L = 6 (n_max 5, 6); the plane's knobs (χ_op 32 and 64, exact SVD)
  at L = 5 n_max 8; the generous knobs (χ_op 64, weight 1, exact SVD, no window, χ_dmrg 32)
  at L = 6 n_max 6 and L = 5 n_max 6–8; the archived `s5i` generous rows at L = 6 n_max 5
  are reused.

## S8 harvest (Layer 5, items 4–5): what the pool decided, and what it did not

Harvest unit, 2026-09-12 (+ fix round 1 the same day). Both pools died in the 2026-09-11
21:22 reboot (8 results finished before it) and were restarted at 10:28:53 PDT 2026-09-12
with the same commands. CORRECTION (fix round 1): an earlier version of this paragraph said
"the harvest waited 16 h (deadline 02:35 PDT 2026-09-13) for the verdict set … and then
summarized what was on disk". **That wait did not happen and the sentence is withdrawn.**
The harvest was cut short ≈ 36 min after the restart — every S8 summary record it wrote is
stamped `date_utc` 2026-09-12T18:04:23Z = 11:04:23 PDT — and summarized what was on disk at
that moment with `--diagnostics --only s8 --no-new-jobs`. The verdict set it was waiting for
— λ = 0.1, the plane at (8, 32) / (10, 32) / (12, 32) / (8, 64) for λ 0 / 2 / 4, the τ₀ rows at
χ_op 32 and 48, the a = ½ pairs, the plane-knob anchor rows, and the ED rungs L 6 n_max 5
(matrix-free and dense) and L 5 n_max 6 — was therefore only partly on disk: **the plane part
(20 jobs, 370.5 h of 2-thread job wall in `s8_plan.json`) had produced no row, which is what
the plan predicts** (the four running (8, 32) and (10, 32) jobs are estimated at 1.18 h and 3.66 h each and were still
running, 1.5 h in, when the fix round checked at 11:58 PDT). An empty plane after 36 min is
not evidence of a sick pool. A partial-parts runner now writes
`s8_plan_<parts>.json` etc. and never clobbers the full records (the launch's pool-B runner
had rewritten `s8_plan.json` as its 16-job 'ed' plan); the full plan (70 jobs: the 69 plus a
`keff` row) was regenerated with `--only s8 --plan`.

- **λ = 0.1 — decided: the first-order slope survives, the floor reading is excluded.**
  At the sweep's settings exact/E_H = 0.99450 (E_min −2.157277e-02, reference inconsistency
  2.3e-5, norm ratio 0.912, 803 s); control-normalized residual **0.741 %** against the exact
  first-order prediction 0.1 × 0.0809 = 0.809 %: **0.35 bands below it** (band 0.19 %). The
  alternative reading of the small-λ deficit — a λ-independent cutoff floor at the ≈ 1.4 %
  the λ = 0.25 / 0.5 points read — would have put the point 3.3 bands *above* the
  prediction; it is excluded. The second-order reading (c₂ ≈ −0.10, i.e. 0.71 % here) is
  not separable from pure first order at this λ (0.5 bands apart). What survives: the
  residual at small λ is the exact first-order quartic tax, with the λ = 0.25 → 1 deficit a
  genuine higher-order downturn, not a cutoff artefact. `data/s8_lam0.1.json`;
  `tests/test_interacting_qei_s8.py::test_lambda_0p1_is_the_first_order_slope_not_a_floor`.
- **τ₀ ≥ 1 at χ_op 32 (item 5) — the controls improve, the rows stay unusable.** The
  λ = 0 control reads **1.0361** at τ₀ = 1.0 and **1.2531** at τ₀ = 1.5, against 1.085 and
  1.410 for the archived `s7d` rows — which are six knobs away (n_max 6, weight 0.25,
  χ_op 16, χ_acc 32, χ_mpo 64, χ_dmrg 12), so the improvement is *not* attributable to χ_op;
  the isolation rows below split it. The first passes the 5 % control gate of `s7s`; but the λ = 4 row it would
  normalize (exact/E_H 1.581, i.e. a *negative* residual of −53 %) carries a reference
  inconsistency |E_ref − E_ref_static| of **38 % of its own E_min** (1.3–5 % at the
  resolved λ ≤ 1 sweep points; 2.7 % in the control itself), and the τ₀ = 1.5 pair 113 %
  and 117 %: those E_min are not resolved at all (the evolving operator loses 17–20 % of
  its norm, `norm_ratio` 0.80–0.83). The harvest therefore gates a τ₀ row on BOTH the
  control (|c − 1| < 0.05) and the reference inconsistency (≤ 5 % of |E_min|, row and
  control), fits a power law to positive residuals only, and quotes no ratio from τ₀ ≥ 1 at
  χ_op 32; the exponent statement stays restricted to τ₀ = 0.375–0.75. **The first χ_op 48
  rows landed at 13:46–13:58 PDT** and are unusable too: τ₀ = 1.5 λ = 4 (E_min −1.9248737e-03,
  exact/E_H 10.842, reference inconsistency **654 % of |E_min|**, norm 0.800), its λ = 0 control
  (E_min −5.3538589e-03, **1.21742**, inconsistency 64 % of |E_min|, norm 0.846) and τ₀ = 1.0
  λ = 4 (E_min −4.0700781e-03, exact/E_H 1.626, inconsistency 47.6 %, norm 0.834). **Its
  λ = 0 control landed at 14:12 PDT** (fix round 2): E_min −1.4356626e-02, **1.038428**,
  inconsistency **7.65 % of |E_min|**, norm 0.884, 2222 s. At fixed n_max 8 / weight 0.5 /
  χ_dmrg 16 the two τ₀ ≥ 1 controls now have complete χ_op sequences, and they say opposite
  things about χ_op and the same thing about the row's usability:
  • τ₀ = 1.5: **1.33774 (16) → 1.25306 (32) → 1.21742 (48)**, steps −0.0847 then −0.0356 —
    monotone *toward* 1, still 21.7 % off, steps shrinking by ≈ 0.42; but its own reference
    inconsistency is **2.3 % → 117 % → 64 %** of |E_min|, so the gate is failed at every
    χ_op above 16. EXTRAPOLATION (not a measurement): continuing the control at that ratio
    lands near 1.19, i.e. χ_op alone does not bring this row through the 5 % control gate.
  • τ₀ = 1.0: **1.03379 (16) → 1.03614 (32) → 1.03843 (48)**, steps **+0.00235 then
    +0.00229** — monotone *away* from 1, by an almost constant amount per χ_op doubling-ish
    step, while its inconsistency runs **9.8 % → 2.7 % → 7.65 %** and so crosses back over
    the 5 % gate the χ_op 32 row passed. Raising χ_op at τ₀ = 1.0 buys nothing: the control
    gets worse and the reference gets worse again.
  Together: **the τ₀ ≥ 1 rows are limited by something the operator bond dimension does not
  fix**, and there is now a measured χ_op sequence at each τ₀ saying so, not an extrapolation.
  `data/s8_tau0_rows.json` (`control_chi_op_sequence`);
  `tests/test_interacting_qei_s8.py::test_tau0_rows_at_chiop48_are_pinned`,
  `::test_control_chi_op_sequences_are_pinned`.
- **The χ_op axis, isolated (fix round 1) — χ_op is not what fixed the τ₀ = 1.0 control.**
  Two extra λ = 0 rows run the S8 recipe *at χ_op 16* (n_max 8, weight 0.5, χ_acc 32,
  χ_mpo 64, χ_dmrg 16, sketched SVD, the same dt/steps), so the six-knob gap between `s7d`
  and the S8 rows splits into two measured steps. τ₀ = 1.0: **1.085 → 1.03379 → 1.03614**,
  i.e. the (n_max, weight, χ_dmrg) step carries −0.0512 — all of the improvement — and
  χ_op 16 → 32 moves the control **+0.0023, away from 1**. τ₀ = 1.5: **1.410 → 1.33774 →
  1.25306**, −0.0723 then −0.0847: there the two steps are comparable and χ_op carries the
  larger half. What χ_op does buy at τ₀ = 1.0 is the reference consistency: 9.8 % of |E_min|
  at χ_op 16 against 2.7 % at χ_op 32 (the gate is 5 %). The isolation rows are themselves
  unusable as rows (norm_ratio 0.879 / 0.846, E_min −1.429253e-02 / −5.882997e-03, 577 s /
  827 s at 1 thread). `data/s8_tau0_rows.json` (`chiop16_isolation`);
  `tests/test_interacting_qei_s8.py::test_chiop16_isolation_splits_the_knobs`.
- **What the sweep's knobs cost at λ = 4 (fix round 1, CORRECTED in fix round 2, item 4 —
  a bound on the route, not the magnitude).** The MPO-vs-ED pairs on the *same*
  (L, n_max, trapezoid grid) exist at the sweep's own knobs (χ_op 32, weight 0.5, sketched
  SVD). At λ = 0 the route reproduces exact ED to **+3.36e-3** (L 6, n_max 5), **+6.36e-3**
  (L 5, n_max 6) and **+1.38e-2** (L 5, n_max 8). At λ = 4, on the same points:
  **+3.37e-2** (L 6, n_max 5), **−1.19e-1** (L 5, n_max 6), **+3.94e-1** (L 5, n_max 8).
  The plane's knobs (χ_op 32, weight 0.5, **exact** SVD) at L 5, n_max 8 read −3.89e-3
  (λ = 0) and **+4.23e-1** (λ = 4). The generous knobs (χ_op 64, weight 1, exact SVD) at
  L 6, n_max 5 read −1.65e-3 and −1.52e-2. So the measured fact is **not** a bounded
  few-per-cent effect: at fixed χ_op 32 the λ = 4 operator-truncation error **grows with
  n_max** — 3.4 % → 11.9 % → **39.4 %** at n_max 5 / 6 / 8 — and its largest value sits at
  the sweep's own Fock cutoff n_max = 8, the cutoff every quoted S8 λ = 4 ratio and the
  first plane point are taken at; an exact SVD there makes it worse (42.3 %), not better.
  In absolute terms at L 5, n_max 8, λ = 4 the MPO reads E_min −1.3060750e-02 (sweep) /
  −1.2451097e-02 (plane) against the exact −2.1561127e-02, i.e. exact/E_H 1.978 / 1.885
  against 3.265: the route **understates the magnitude of the λ = 4 infimum by a factor
  1.6508 (sweep) / 1.7317 (plane)** at n_max 8.
  At λ = 0 the same knobs at the same point are off by 1.4 % — a factor 29 apart. The
  earlier "3–12 % wherever an exact answer exists" is **retracted** (no-longer-claims item 8).
  CAVEATS: (a) the exact side of the n_max 8 pair is itself un-converged in n_max (the L = 5
  λ = 4 exact ladder is still falling 38.6 % per rung there), so this is a disagreement
  between two routes neither of which has converged, not an error bar on a known answer;
  (b) the L = 5, n_max 6, λ = 4 ED rung's own trapezoid grid is coarse (the dt 0.125
  re-solve moves it +1.32e-3 relative, 20–1000× every other rung), and the n_max 8 rung's
  moves +4.96e-4. This does **not** give the λ ≥ 2 magnitude; it says the sweep's and the
  plane's knobs cannot be trusted at λ = 4 at the cutoff they are run at.
  `data/s8_ed_anchor.json` (`pairs_mpo_vs_ed`, `worst_rel_dev_by_knobs`,
  `cross_validation_by_knobs.*.worst_by_n_max`);
  `tests/test_interacting_qei_s8.py::test_mpo_vs_ed_pairs_are_pinned_and_the_cross_validation_levels_hold`,
  `::test_sweep_knob_pairs_are_an_order_of_magnitude_worse_at_lambda_4`,
  `::test_lambda4_mpo_error_grows_with_nmax_at_the_sweep_knobs`,
  `tests/test_interacting_qei_ed.py::test_s8_lambda4_L5_nmax6_rung_is_pinned_with_its_coarse_quadrature`.
- **The knob ladder at the paper's own cutoff (fix round 2, run 14:48–15:29 PDT 2026-09-12) —
  what actually costs the λ = 4 accuracy, and why the harvest's gate cannot see it.** The
  39.4 % MPO-vs-ED gap above is at χ_op 32. Seven MPO rows now sit at **L = 5, n_max 8,
  τ₀ = 0.75, λ = 4** — the paper's own Fock cutoff, where the exact answer
  **E_min = −2.1561127e-02** exists (`jobs/s8_edlz_L5_nmax8_lam4.json`) — walking one knob
  at a time off the sweep's settings (weight 0.5, sketched SVD, χ_dmrg 16):

  | χ_op | weight | SVD | χ_dmrg | E_min | dev vs exact | refcons/E_min | norm |
  |---|---|---|---|---|---|---|---|
  | 32 | 0.5 | sketched | 16 | −1.3060750e-02 | **+39.42 %** | 6.73 % | 0.872 |
  | 32 | 0.5 | **exact** | 16 | −1.2451097e-02 | **+42.25 %** | 33.90 % | 0.885 |
  | 48 | 0.5 | sketched | 16 | −1.8352004e-02 | **+14.88 %** | 2.66 % | 0.873 |
  | 64 | 0.5 | sketched | 16 | −1.8613818e-02 | **+13.670 %** | 1.21 % | 0.873 |
  | 64 | 0.5 | sketched | **32** | −1.8613841e-02 | **+13.669 %** | 1.21 % | 0.873 |
  | 96 | 0.5 | sketched | 16 | −2.2531586e-02 | **−4.50 %** | 6.56 % | 0.873 |
  | 64 | **1.0** | sketched | 16 | −2.1817208e-02 | **−1.19 %** | 36.06 % | 0.926 |

  Four readings, each a measurement at this one point, not a general claim:
  1. **χ_dmrg is not the limiter.** 16 → 32 at χ_op 64 moves the deviation by 1.1e-5 relative
     (13.6695 % → 13.6694 %). The DMRG bond is already converged.
  2. **The weighting is.** weight_decay 0.5 → 1 at χ_op 64, everything else held, takes the
     deviation from **+13.67 % to −1.19 %** — a factor 11. The generous knobs' weight 1 is
     doing the work the generous-knob agreement was credited to.
  3. **χ_op alone helps but is not monotone and changes sign:** +39.4 → +14.9 → +13.7 →
     **−4.5 %** at 32 / 48 / 64 / 96. |error| falls throughout, but the route crosses the
     exact answer between χ_op 64 and 96, so no χ_op in this ladder gives a one-sided
     approximation, and χ_op 64 → 96 is not a convergence one could extrapolate.
     (Being *below* the same-grid exact infimum is not an anomaly-protocol event: the MPO
     route minimizes a *truncated* operator, which is not bounded below by the exact one —
     see no-longer-claims item 5. It is an operator-truncation artefact, and it is one more
     reason the route's λ = 4 number at these knobs is not an approximation from a known side.)
  4. **The harvest's own 5 % reference-consistency gate does not track λ = 4 accuracy — here
     it is anti-correlated with it.** Across these seven rows the best refcons (**1.21 %**,
     χ_op 64) belongs to a row that is **13.7 % wrong**, and the best accuracy (**1.19 %**,
     weight 1) belongs to a row whose refcons is **36.06 %**, seven times the gate. The gate
     is **necessary, not sufficient, and not even monotone in the error**: no λ ≥ 2 magnitude
     may be quoted on the strength of passing it, including from a future plane point.
     This is measured, not argued, and it is an independent reason item 4 stays open.

  At λ = 0 the same χ_op 64 / weight 0.5 row is **+1.21 %** off exact (refcons 1.29 %) —
  a factor 11 below its own λ = 4 error at identical knobs.
  ASSUMPTION recorded, not asserted: that the weight-1 improvement at L = 5, n_max 8 carries
  to L = 16; nothing here measures that, and the weight-1 row's 36 % reference inconsistency
  means it could not be quoted as a row even if it did.
  Jobs `jobs/s8_anchor_chiop{48,64,96}_L5_nmax8_lam{0,4}.json`,
  `jobs/s8_anchor_chiop64_{chidmrg32,w1}_L5_nmax8_lam4.json` (2 threads, idempotent);
  `data/s8_ed_anchor.json` (`chiop48_n8`, `chiop64_n8`, `chiop64_dmrg32_n8`, `chiop96_n8`,
  `chiop64_w1_n8`);
  `tests/test_interacting_qei_s8.py::test_chi_op_ladder_at_nmax8_and_what_it_says_about_the_refcons_gate`.
  WALL-TIME CAVEAT: the χ_op 48 job records `time_s` 1386 s but was SIGSTOPped by the OS for
  most of that window; the undisturbed jobs took 136–326 s. None of these walls is used in
  the wall model.
- **The ED anchor beyond n_max 4.** The matrix-free rungs reproduce the archived dense rows
  on the same grid to 1.6e-13 … 5.6e-12 (L 6, n_max 3–4, λ 0 and 4; the fine grid lands
  2.3e-7 … 4.1e-7 from the analytic kernel). L 6 n_max 5 λ = 0: E_min −2.0019521e-02
  (exact/E_H 1.00624, odd sector, 1199 s; the L = 6 λ = 0 ladder 0.9214 → 1.0173 → 1.0062
  is non-monotone as the dense one was). Further rungs and the MPO-vs-ED pairs per knob
  set: numbers table and `data/s8_ed_anchor.json`.
- **The exact ladders at L = 5 (landed 12:50–13:00 PDT, fix round 1) — item 4 is not cutoff-safe
  even on the exact route.** At λ = 0 the exact ratio converges on 1: **1.000739 / 0.999114
  / 1.000049** at n_max 6 / 7 / 8 (4.9e-5 from 1 at the top rung). At λ = 4, same L, same
  grid, the exact infimum **falls 64.8 % between n_max 6 and 7 and a further 38.6 % to
  n_max 8** (E_min −9.439314e-03 → −1.5554289e-02 → −2.1561127e-02; exact/E_H 1.4292 →
  2.3551 → **3.2646**), with Lanczos residuals 3.6e-11 … 8.5e-11 — every rung is an exact
  eigenvalue of its truncated space, it is the *ladder* that is far from converged (the fall
  is slowing, but at n_max 8 the exact λ = 4 infimum is still 3.3× the Hartree free bound and
  still moving). (Rayleigh–Ritz is satisfied: a larger Fock space can only lower the
  infimum.) The L = 6 λ = 4 ladder, two rungs lower, is still rising too (0.4931 / 0.8536 /
  0.8789 at n_max 3 / 4 / 5). So no route — exact or MPO — has reached a cutoff at which the
  λ = 4 magnitude is stable, and no λ ≥ 2 number is quoted from any of them.
  `data/s8_ed_anchor.json` (`ed_ladders`);
  `tests/test_interacting_qei_ed.py::test_s8_L5_ladders_converge_at_lambda_0_and_do_not_at_lambda_4`.
- **The plane (item 4) — the first point landed, and it does not close it.** At 13:05–13:16
  PDT 2026-09-12 the first setting of the plane finished: n_max 8, χ_op 32, weight 0.5,
  **exact SVD**, 2.7 h per job at 2 threads. λ = 0 control **1.02261** (E_min −2.3165923e-02),
  λ = 4 **0.95979** (E_min −7.2190762e-03) → control-normalized residual **+6.14 %** against
  E_H and **+10.20 %** against K_eff. It closes nothing, for three measured reasons: (i) the
  point's own λ = 0 control is 2.3 % off 1, *worse* than the sweep's sketched-SVD control
  (1.00192) at the same (n_max, χ_op) — the exact SVD did not buy a better control here;
  (ii) the λ = 4 row's reference inconsistency is **56.5 % of its own |E_min|**, eleven times
  the 5 % gate the τ₀ rows are held to (the gate is now computed and recorded for every plane
  row); (iii) one setting gives neither the n_max spread nor the χ_op spread the verdict
  criterion needs, so `resolved` is false; and (iv, fix round 2) **at exactly these knobs and
  this n_max the route is 42.3 % from the exact answer at λ = 4** where an exact answer
  exists (L = 5, n_max 8: MPO −1.2451097e-02 vs exact −2.1561127e-02), against 0.39 % at
  λ = 0 — so the λ = 4 number this point produces is not an approximation to the λ = 4
  infimum at any accuracy the item needs. For scale, the sweep's own reference inconsistency
  at λ = 2 / 3 / 4 is 13.2 % / 30.8 % / 2.5 % of |E_min| (`jobs/s5h_sweep_lam*.json`), so a
  large inconsistency at λ ≥ 2 is not new with this row. 18 of the plane's 20 jobs remain; the two that finished record 9694 s and 10074 s against a 4253 s estimate, i.e. **2.37×**, which projects the remaining 18 at 873 h of 2-thread job wall.
  WALL-MODEL CAVEAT (fix round 2, 15:32 PDT): **`time_s` in this environment is an upper bound on cost, not a measurement of it.** All six S8 pool workers were found SIGSTOPped (`ps` STAT `T`) at 15:32 PDT — which is why no pool row landed between 14:12 and then — and were resumed with `kill -CONT` (nothing was killed; both pool parents 19494/19495 and all six workers are alive and running). A job records wall-clock time across any such suspension: the fix round's own χ_op 48 probe recorded `time_s` 1386 s for what took the undisturbed χ_op 64 probe 231 s. The 2.37× factor is therefore **not** established as a compute-cost measurement, and the 873 h projection built on it is an upper bound of unknown tightness. Nothing downstream quotes it as physics.
  `data/s8_convergence_plane.json` (`verdict`, `wall_calibration`, `plane_jobs_unfinished`);
  `tests/test_interacting_qei_s8.py::test_first_plane_point_is_not_usable_and_does_not_close_item_4`.

## S8 triage (2026-09-12): what the queued plane was worth, and what replaced it

Unit 1 of the triage workflow. The question was whether the 18 remaining jobs of the
`plane` part — 313 h of nominal job wall, all at `weight_decay = 0.5` and all with the
exact SVD — earn their cost. They do not, for two measured reasons, and the plane has
been retired and replaced.

**(0) The wall model understates the plane by 2.3×, so it was never 313 h.** The pool's
headline uses one global calibration (median measured/estimated over *all* finished jobs,
0.82–0.85), which is set by the cheap ED rows (ratios 0.13–1.12). The two *finished*
exact-SVD plane jobs ran **2.28× and 2.37×** their estimate (`plane_nmax8_chiop32_lam4`
2.69 h vs 1.18 h est; `..._lam0` 2.80 h vs 1.18 h), the four finished exact-SVD
`anchor_generous` rows 1.13–2.65×, and each of the two `plane_nmax10_chiop32` jobs killed
below had already consumed **3.77 h** of 2-thread work against its 3.7 h estimate and was
still running (so its own ratio is > 1.02 and was still rising). So the honest projection
for the 18 remaining plane jobs was **≈ 856 h of 2-thread job wall** (368 h nominal × 2.326,
`wall_model_calibration.plane_retired_remaining_calibrated_h`), not 313 h — about seven days
of wall clock on five workers.

**FIX ROUND 1 — the same calibration on both sides.** The first write-up applied the 2.326×
to the retired plane and quoted its *replacement* at nominal, and attributed the 2.3× to the
exact SVD specifically. Neither is right. The wall model under-estimates the whole L = 16
plane class: the seven landed *sketched* `plane2` jobs run **0.90–1.88×, median 1.677×**
(`wall_model_calibration.fix_round_1.plane2`, n = 7 — a **frozen** tag set, because the live
`per_part.plane2` median moves every time the pool lands another row; it was already 1.590
at n = 8 by the time this was committed), so the exact-SVD-specific excess is only
**1.388×** on top of that. Calibrated like for like, `plane2` is **73.0 h** (43.6 h nominal
× 1.677), not 43.6 h, and the saving is **11.7×**, not the ≈ 19× that 856 h / 43.6 h implies.
Further, **37.8 h of plane2's 43.6 h nominal (87 %) is the n_max 10 and 12 rows**, whose
estimates come from the `(n_max+1)^4` term of the wall model with **no sketched job above
n_max 8 ever timed on this tree** — so both 43.6 h and 73.0 h are extrapolations quoted as
costs, and only the n_max 8 rows (5.7 h nominal) rest on measured times.

**(1) The plane's distinguishing knob buys nothing.** `svd_dense_limit = "inf"` (the exact
SVD in the operator truncation) is what separates the plane from the cheap sweep rows and
is where its cost lives. At **identical** (L, n_max, χ_op, weight, λ) it costs **1.75–21.98×**
over the ten archived same-spec pairs (`svd_mode_pairs`; the cheapest is the L 16, n_max 6,
χ_op 16, w 0.5 pair at 1.751× on 2 threads / 2.225× on 4 — the fix-round-2 row below) and
**7.44–21.98×** at the four points tabulated here. It does **not** improve the answer at
either of the two places where truth is known:

| point (all weight 0.5, χ_op 32) | sketched SVD | exact SVD | cost | truth |
|---|---|---|---|---|
| L 16, n_max 8, λ = 0 — the only exactly-known number at the paper's own L (both rows 2 BLAS threads; the sketched leg also runs 1.001923 at 4 threads) | exact/free **0.999931** (1353 s, the `plane2` row) | exact/free **1.022613** (10074 s) | **7.4×** | 1 |
| L 5, n_max 8, λ = 4 — the only exactly-known λ = 4 number at the paper's cutoff | **+39.425 %** (171 s) | **+42.252 %** (3127 s) | **18.3×** | E_min −2.1561126717e-02 |
| L 5, n_max 8, λ = 0 | +1.380 % (148 s) | **−0.389 %** (3249 s) | 22.0× | −1.8462671543e-02 |
| L 16, n_max 6, λ = 0 | 1.011667 (593 s) | 1.022527 (4686 s) | 7.9× | 1 |

The exact SVD wins one of the four (L 5, λ = 0) and loses the two that matter: the λ = 4
anchor and the λ = 0 control at L = 16. Retiring it is the whole cost saving.

**FIX ROUND 1 — how far the L = 16 λ = 0 penalty may be quoted.** The first write-up said
"**328×** further from 1". That is **retracted** (no-longer-claims item 9). Its denominator,
|0.999931 − 1| = 6.9e-05, is **29× smaller than the 0.199 % same-spec thread scatter this
same archive records at this same spec** — the sketched leg is *not resolved from 1 at all*,
and re-running it at 4 BLAS threads (1.001923) moves the headline to 11.8×. The quotable
number is the band-aware one, archived as
`banded_penalties_l16_lam0[axis=svd, n_max 8].factor_conservative`:

| L 16, n_max 8, χ_op 32, w 0.5, λ = 0 | distance from 1 | vs the **sketched** leg's same-spec band (0.199 %) |
|---|---|---|
| sketched, 2 threads | 6.9e-05 | **inside** its own band: not resolved from 1 |
| sketched, 4 threads | 1.923e-03 | at its own band (0.9654×): not resolved from 1 |
| exact SVD, 2 threads | **2.261e-02** | **11.35× the sketched leg's band**; its **own** band is **unmeasured** (one archived thread count) |
| **penalty that may be quoted** | **≥ 11.8×** further from 1, for **7.4×** the wall | |

**ASSUMPTION (fix round 2, unproved).** The exact-SVD route's same-spec thread scatter is
assumed no larger than the sketched route's at the same knobs. The exact-SVD leg of this
spec has **one** archived thread count, so
`banded_penalties_l16_lam0[axis=svd, n_max 8]` carries `same_spec_band_worse: null` and
`worse_leg_resolved_above_its_band: null` — the archive declines to call that leg resolved,
and fix round 1's wording ("11.4× **its own** band: resolved") is **withdrawn**
(no-longer-claims item 10). The 11.35× is against the *sketched* leg's band.

**What fix round 2 measured about that assumption.** Both routes were run at **2 and at 4
BLAS threads** at the cheapest exact-SVD spec at this same L — L 16, n_max 6, χ_op 16,
χ_acc 32, χ_mpo 64, w 0.5, λ = 0 — as jobs `svdband_{sketched,exactsvd}_chiop16_wd0.5_thr4`.
Both come back **bit-identical** across the thread counts (`rel_spread_E_min` exactly
**0.0** on both legs; sketched E_min −2.2800211601990195e-02 at 438.4 s / 159.1 s, exact SVD
−2.2865136075353676e-02 at 767.8 s / 353.9 s): **the exact SVD is not the noisier route
where both bands are measured**, which is the direction the assumption needs. Two things it
does *not* establish, and the pins say so: the n_max 8 / χ_op 32 exact-SVD leg still has one
archived thread count (`::test_exact_svd_is_a_cost_and_not_an_accuracy_gain` asserts
`same_spec_band_worse is None`, `n_threads_archived_worse == 1`, on purpose), and since the
thread count alone changes *nothing at all* here, the 0.199 % spread at n_max 8 / χ_op 32
is **not shown to be a thread effect** — it is an unexplained same-spec difference between
two jobs that differ in thread count *and* in when they were run. It remains a conservative
noise floor; "thread band" remains an ASSUMPTION until one spec is run twice at the same
thread count. Pinned by
`::test_the_exact_svd_route_is_not_the_noisier_route_where_both_bands_are_measured`.

**The measurement that would close (a) outright**, not done here because it is 1.5–3 h of
4-thread wall against the user's running pools: re-run the *headline* exact-SVD spec itself
at 4 threads. The spec is queued on disk, unrun, as
`data/jobs/s8_svdband_planeknobs_nmax8_chiop32_lam0_thr4.spec.json` (the plane job's own
knobs, `threads: 4`); one command runs it, and the `wt` harvest then folds it into
`banded_penalties_l16_lam0[axis=svd, n_max 8]` and makes `same_spec_band_worse` a measured
number — at which point the three `null` / `== 1` pins in
`::test_exact_svd_is_a_cost_and_not_an_accuracy_gain` fail on purpose and must be replaced
by the measured band, with a note here:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
.venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py \
  --job     papers/interacting-vacua-first-numbers/data/jobs/s8_svdband_planeknobs_nmax8_chiop32_lam0_thr4.spec.json \
  --job-out papers/interacting-vacua-first-numbers/data/jobs/s8_svdband_planeknobs_nmax8_chiop32_lam0_thr4.json
```

The *direction* — the exact SVD is the worse knob at the one exactly-known number at the
paper's own L — is unaffected: 2.261 % against a leg that never leaves its 0.199 % band.
The retirement decision rests on that direction and on the 7.4× cost, not on the magnitude.

**(2) `weight_decay` cannot be a fixed 0.5, and it cannot be a fixed 1 either: the
ordering reverses with L.** Yesterday's single point (L 5, n_max 8, χ_op 64, λ = 4) had
weight 0.5 → 1 worth a factor 11 (+13.670 % → **−1.188 %**). At **L = 16**, where the
λ = 0 control is exactly known, the same change at n_max 8, χ_op 32 goes the other way and
by more: exact/free **0.999931 → 1.082086** on the thread-matched `plane2` pair and
1.001923 → 1.081776 on the archived 4-thread pair, i.e. **at least 42.5× worse**
(`banded_penalties_l16_lam0[axis=weight_decay, n_max 8].factor_conservative`; the weight-1
leg's 8.18 % is 285× *its* own 0.029 % band and is resolved, while the weight-0.5 leg's
distance from 1 is inside its 0.199 % band and is not — so the thread-matched **1190×** of
the first write-up is **retracted**, no-longer-claims item 9). It is
the same sign at every L = 16 pair measured (n_max 6, χ_op 32: 1.011667 → 1.035981;
n_max 6, χ_op 64: 1.015018 → 1.026544; n_max 6, χ_op 32 exact SVD: 1.022527 → 1.036178).
So the MANIFEST's own standing ASSUMPTION — "that the L = 5 knob ranking carries to
L = 16" — is **measured false on the weight axis at λ = 0**, and a plane run at any single
weight produces a number no anchor can certify.

**(3) The weight axis is non-monotone too, and the route has a 1.7–2.4 % reproducibility
floor at λ = 4.** The four-rung weight ladder at the one point where an exact λ = 4 answer
exists at the paper's own Fock cutoff (L = 5, n_max 8, χ_op 64, χ_dmrg 16, sketched SVD,
1 BLAS thread; exact E_min −2.1561126717e-02) reads

| weight_decay | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| deviation from exact | **+38.613 %** | **+15.718 %** | **−11.834 %** | **+0.463 %** |

— a sign change between 0.5 and 0.75 and *back* between 0.75 and 1.0. So weight behaves
exactly as χ_op does (+39.4 % → −4.5 % with a sign change between 64 and 96): **no weight
is a one-sided approximation**, and "the weighting is the limiter" is a statement about the
*endpoint* w = 1 being closest at this point, not about a convergence direction.

Worse, the same spec run at a different BLAS thread count gives a different answer. Two
runs whose specs differ **only** in `threads` (the seed of the randomized sketch is fixed
at 0 in both):

| L 5, n_max 8, χ_op 64, λ = 4, sketched | 2 threads | 1 thread | spread |
|---|---|---|---|
| weight 0.5 | −1.8613818e-02 (+13.670 %) | −1.8172225e-02 (+15.718 %) | **2.430 %** |
| weight 1.0 | −2.1817208e-02 (−1.188 %) | −2.1461251e-02 (+0.463 %) | **1.659 %** |

The truncation decisions amplify the non-associativity of the threaded reductions: the
operator MPO reaches χ_mpo 240 on one and 244 on the other. **This is a floor under every
λ = 4 magnitude at these knobs and it is larger than the weight-1 advantage it is used to
establish** (0.46–1.19 %). It also falsifies the tolerance rationale carried in
`tests/test_interacting_qei_s8.py` ("BLAS threading moves it by < 1e-8"). **Fix round 2:
that sentence is false at λ = 0 as well, not only at λ = 4** — the same spec at L 16,
n_max 8, χ_op 32, w 0.5, λ = 0 moves **0.199 %** between 2 and 4 BLAS threads, 2e7× the
claimed 1e-8 — and it is now removed from that file's docstrings (module docstring and
`test_exact_svd_is_a_cost_and_not_an_accuracy_gain`), replaced by the measured statement:
the pins are tight (1e-6) because they are on archived job-file contents, which never
change, while the route's own same-spec reproducibility is **0.06–5.12 %** (and exactly
**0.0** at L 16, n_max 6, χ_op 16 — see the fix-round-2 measurement below). Nothing in the candidate quotes a λ = 4 magnitude, so no number is
retracted by this — but any future one must carry it.

**(4) The answer to A: "the weighting is the limiter at λ ≥ 2" is FALSE as stated — it is
a λ = 4 statement, not a λ ≥ 2 statement.** The λ = 2 and λ = 0 rungs at the same point and
knobs (L = 5, n_max 8, χ_op 64, χ_dmrg 16, sketched, 1 thread), against the new exact
matrix-free Lanczos values **−9.388966078354377e-03** at λ = 2 (fine-grid quadrature check
8.6e-7, 2791 s) and −1.8462671543e-02 at λ = 0:

| deviation from exact | w 0.25 | w 0.5 | w 0.75 | w 1.0 |
|---|---|---|---|---|
| λ = 0 | — | **+1.268 %** | **−1.056 %** | **−0.272 %** |
| λ = 2 | **+1.435 %** | **+0.753 %** | **−63.331 %** | **−0.845 %** |
| λ = 4 | +38.613 % | **+15.718 %** | −11.834 % | **+0.463 %** |

At λ = 2 **three of the four rungs are inside 1.5 %** (0.25, 0.5 and 1.0) — the weighting
costs essentially nothing there — while at λ = 4 weight 0.5 is 15.7 % out and weight 1 is 0.46 % out. And the
w = 0.75 rung at λ = 2 is a **−63 %** interior catastrophe: the axis is not merely
non-monotone, it has a blow-up between two accurate neighbours. That rung is the one row
the 5 % reference-consistency gate does catch (refcons/|E_min| **115.7 %**) — but the same
gate rejects the *accurate* w = 1 row (**44.8 %** against a 0.85 % error), so it still
certifies nothing. The thread-scatter is λ-dependent in the same way: **0.061 %** at λ = 0
against **2.430 %** at λ = 4, at identical knobs.

So the honest answer to A, at the one point where the three-λ ladder is complete, is:
**a λ = 4 statement about the endpoint w = 1 on a non-monotone axis, not a general
statement about λ ≥ 2.** The other two anchor points (L 6 n_max 5, L 5 n_max 6) and the
χ_op-32 column are still running — their λ = 2 exact references have landed at
**−1.0118762207130372e-02** (L 6 n_max 5, exact/E_H 0.931663, fine-grid 5.6e-6) and
**−9.385658023800647e-03** (L 5 n_max 6, 0.922426, 3.6e-6). plane2 running weight as an
axis is the right design for exactly this reason.

**(5) And the route's own thread scatter reaches the numbers this candidate quotes.** Seven
same-spec pairs are now archived (`repeat_runs_same_spec`). At **L = 16, n_max 8, χ_op 32,
weight 0.5, sketched** — the sweep's own settings — the relative spread between two BLAS
thread counts is **0.199 % at λ = 0** (−2.2652090e-02 on 2 threads, −2.2697211e-02 on 4),
**2.031 % at λ = 2** (−1.1347773e-02 vs −1.1578241e-02) and 0.142 % at λ = 4. So the
"0.19 % λ = 0 control at the sweep's settings" that the candidate quotes throughout is
**the size of the route's own thread scatter**, and the λ = 2 sweep row carries a 2 %
scatter that no gate reports. Nothing is retracted here — the λ ≤ 1 magnitudes are quoted
with a ± 0.2 % band that this is inside, and no λ ≥ 2 magnitude is quoted — but the band on
any λ ≥ 2 number must start at 2 %.

**What weight 1 costs (item B).** Measured at L = 16 at identical knobs, not estimated:

| L 16, λ = 0 | weight 0.5 | weight 1.0 | cost ratio | χ_op reached |
|---|---|---|---|---|
| n_max 8, χ_op 32, sketched (2 threads, both `plane2`) — exact/free **0.999931 → 1.082086** | 1353 s | 1549 s | **1.14×** | 32 → 32 |
| n_max 8, χ_op 32, sketched (4 threads, the archived sweep pair) — 1.001923 → 1.081776 | 2973 s | 3367 s | **1.13×** | 32 → 32 |
| n_max 6, χ_op 32, sketched | 593 s | 746 s | **1.26×** | 32 → 32 |
| n_max 6, χ_op 64, sketched | 1648 s | 2785 s | **1.69×** | **49 → 64** |
| n_max 6, χ_op 32, **exact SVD** | 4686 s | 4678 s | **1.00×** | 32 → 32 |

Weight 1 is cheap when χ_op saturates (1.00–1.26×) and costs 1.69× only where weight 0.5
does *not* saturate the requested bond (49 of 64): the down-weighting lets the truncation
stop early, which is exactly where it also throws information away. Discarded operator
weight rises from 1.5e-3 to 7.9e-2 (n_max 8, χ_op 32) — but that is measured in two
*different* norms and is not a like-for-like comparison; only the deviations above are.
**Weight 1 is not the expensive option the caution anticipated.** Running both weights
costs less than the exact SVD did.

**The decision.** The `plane` part is **retired** (still buildable with an explicit
`--s8-parts plane`; nothing quotes it), together with the two `anchor_planeknobs64` jobs
that anchored its χ_op-64 exact-SVD knobs. It is replaced by:

* **`plane2`** — the same (n_max 8/10/12) × (χ_op 32/64) grid at λ = 0/2/4, **sketched
  SVD**, with `weight_decay` an explicit **axis** (0.5 *and* 1.0 both run and both
  anchored): 36 jobs, **43.6 h nominal / 73.0 h calibrated** (× 1.677, the median of its
  own seven landed jobs at fix round 1, `wall_model_calibration.fix_round_1`), against 368 h nominal / **856 h calibrated** (× 2.326) for the 18
  retired jobs — a like-for-like saving of **11.7×**. 37.8 h of the 43.6 h is n_max 10 and
  12, which no sketched job has ever reached on this tree, so that part is a wall-model
  extrapolation and not a measured cost.
* **`wt`** — the exact-ED-anchored weight ladder the plane2 columns rest on: L 5 n_max 8,
  L 6 n_max 5, L 5 n_max 6 × χ_op 32/64 × weight 0.25/0.5/0.75/1.0 × λ 0/2/4, 72 jobs,
  **5.8 h**. Archived to `data/s8_weight_triage.json`.
* **`ed`** — three new matrix-free Lanczos references at **λ = 2** (L 5 n_max 6, L 6 n_max 5,
  L 5 n_max 8), because item 4 is about λ ≥ 2 and the anchors only existed at λ = 0 and 4.

Plan total: **161 jobs, 99.2 h** of 2-thread job wall (was 70 jobs, 418 h). Pool A was
stopped and relaunched on the new parts; **17.4 CPU-hours (8.7 h of 2-thread job wall)**
of in-flight work was discarded — four retired-plane jobs (8.5 h) and one τ₀ `a = ½` job
(0.2 h) that the new pool re-runs. Pool B (the ED part) was never touched.

**ASSUMPTION (recorded, not proved).** plane2's anchors are at L ≤ 6, because that is where
ED is affordable; the knob error they measure is the knob error *at L ≤ 6*. Finding (2)
above shows that this does **not** transfer to L = 16 on the weight axis at λ = 0. plane2
therefore cannot, by itself, certify a λ ≥ 2 magnitude at L = 16; it certifies the knob
dependence and gives the λ = 0 control at L = 16 as the second, independent gate. **Item 4
stays open**, and the triage did not close it — it stopped 856 h (calibrated) from being spent on a
design that could not have closed it either.

### Fix round 2 (2026-09-12): three status labels withdrawn, one assumption put under test

A verifier pass on fix round 1 returned three findings; all three are upheld and all three
are fixed here at the data/label layer, not in prose. They are recorded as no-longer-claims
item 10 (a)–(c). **No quoted number changed** — 11.760048636573513, 42.52761134108153,
856.13 h, 73.03 h, 11.72× all stand — what changed is what the repo *claims to know* about
them:

1. **(a)** The exact-SVD leg at L 16, n_max 8, χ_op 32, λ = 0 was labelled "11.4× **its own**
   band: resolved". Its own band is **unmeasured** (`n_threads_archived_worse: 1`). The
   MANIFEST now says **11.35× the *sketched* leg's** 0.199 % band and carries the ASSUMPTION
   explicitly; the test asserts `same_spec_band_worse is None` so the label cannot return.
2. **(b)** A test comment claimed the 4-thread weight pair was "band-clearing on both legs'
   denominators". The archive says `better_leg_resolved_above_its_band: false` for that very
   row (1.9229e-03 against a 1.9919e-03 band, 0.9653×). 42.5276× is a **lower bound**; the
   test now asserts the `false` flag and `err_better_max < same_spec_band_better`.
3. **(c)** The tolerance rationale "BLAS threading moves it by < 1e-8" is removed from both
   docstrings that carried it and from the §S8 sentence that said it "holds at λ = 0": the
   measured same-spec scatter is **0.199 %** at that λ = 0 spec and up to **5.120 %** elsewhere
   (L 5, n_max 6, χ_op 32, w 0.5, λ = 4, 1 vs 2 threads: −1.0049170e-02 vs −1.0563711e-02),
   the worst in the archive now that the `wt` grid is complete.

**The measurement this round adds.** The (a) assumption — that the exact-SVD route's thread
scatter is no larger than the sketched route's — is now under test at the cheapest exact-SVD
spec at the paper's own L: **L 16, n_max 6, χ_op 16, χ_acc 32, χ_mpo 64, w 0.5, λ = 0**,
whose 2-thread legs are archived (sketched 438.4 s, `ratio_exact_over_free_exact` 1.0064697;
exact SVD 767.8 s, 1.0093356). Both are re-run at **4 BLAS threads** as
`jobs/s8_svdband_{exactsvd,sketched}_chiop16_wd0.5_thr4.json`, which gives **both routes a
same-spec band at one identical spec** and so answers the assumption directly rather than
by analogy. Launched 2026-09-12 23:10 local, ~15–25 min of 4-thread wall, sequential, off
the pools. When they land, `--s8-parts wt --no-new-jobs` folds them into
`repeat_runs_same_spec` and `banded_penalties_l16_lam0[axis=svd, n_max 6, χ_op 16]`
automatically (the harvest globs `data/jobs/s8_*.json`). **They landed** (355 s and 159 s
of 4-thread wall, off the pools): both bands are exactly 0.0 — see the SVD table above and
the two new numbers-table rows. The `wt` grid also completed in this window, which moved the
worst same-spec spread in the archive from 2.430 % to **5.120 %** (L 5, n_max 6, χ_op 32,
w 0.5, λ = 4, 1 vs 2 threads); every place that quoted the archive-wide worst is updated.

**Archive hazard found and repaired.** At 23:07 local, pool C (started ~21:22, i.e. *before*
fix round 1's notebook edits at 22:42; 105.3 min of wall, then it exited) finished its `wt` part and rewrote `data/s8_weight_triage.json`
from its **old in-memory code**, silently dropping `repeat_runs_same_spec`,
`worst_rel_spread_same_spec`, `banded_penalties_l16_lam0`, `wall_model_calibration` and the
`tag_*` fields of the pair tables (three tests went red on a file nobody had edited). It was
rebuilt with the current notebook (`--s8-parts wt --no-new-jobs`, ~1 s) and all 51 tests pass
again. **Any pool started before a notebook edit will re-clobber the record it owns when it
finishes** — after editing the notebook, restart the pool that writes the affected part, or
re-run the harvest afterwards.

### S8 triage unit 2 (2026-09-13): the item 4 verdict, and what it is not

**What was added.** A `plane2` harvest block in `notebook_qei_exact.py` writes
`data/s8_convergence_plane2.json` (+ `.csv`): per (n_max, χ_op, weight, λ) the two ratios,
the setting's own λ = 0 control, the control-normalized residual against `E_H` and against
`E_free(K_eff)`, the 5 % reference-consistency gate, the ED-anchor gate (max
|rel dev vs exact ED| over every anchored point at the same (χ_op, weight, λ, SVD mode) at
L ≤ 6), the route's own same-spec reproducibility at that spec, the per-λ verdict with its
axis spreads and gate sensitivity, the confrontation with the exact O(λ) slope, and the
costed-out closure path. **No new compute was run for it** — it reads the pools' own output
— and no job, pool or plan was touched.

**The quoted numbers are a frozen snapshot, by design.** The pool lands `plane2` rows while
the prose is being written, and it did so three times during this pass — the live bracket,
the live wall calibration and the live qualified counts all moved. So the record carries
`unit2_snapshot`: a **named tag list** of the **15** rows that had landed *with their own
λ = 0 controls* when item 4's paragraph was written, the same verdict rule applied to exactly
those rows, and a wall calibration taken over exactly those jobs. (Fix round 1 added the 15th,
`plane2_nmax10_chiop32_w0.5_lam2`; see item 4 and no-longer-claims 11.) Every number in item 4,
in the numbers table and in the tests reads the snapshot; the live `verdict` and
`closure_cost` sit beside it and are free to move. Adding a tag to that list is a deliberate
act to be made together with the prose it changes.

**The verdict is BRACKETED at both λ, stated in item 4 above with its numbers.** The
three things most likely to be misread:

1. **The bracket is not an error bar.** It is min/max of the control-normalized residual
   over the knob settings that have landed, and at λ = 4 it spans both signs. It will move
   as the plane fills. What cannot move is that the systematic already exceeds the 3-band
   quotability criterion by 11× (λ = 2) and 80× (λ = 4): the weight pair that carries it is
   landed and archived, and a max over pairs cannot shrink when pairs are added.
2. **The anchor gate's value is not what decides it.** Recomputed at gates 1 / 2 / 2.5 / 3
   and 5 %, no λ resolves at any of them, and the one setting that sits just outside the
   2 % gate ((χ_op 64, weight 1) at λ = 2, max |dev| 2.138 %) widens the qualified bracket
   to 28.5 bands rather than closing it.
3. **The sign is a weaker claim than the size and it is stated separately.** Against K_eff
   the residual is positive at 5 of 5 settings of the **frozen snapshot** at λ = 2 and 4 of 5
   at λ = 4 (**on the live tree, 5 of 6 and 4 of 6**); against `E_H` it is 2 of 5 and 3 of 5
   frozen, because the weight-1 columns' λ = 0 controls sit 1.72 % and 8.21 % above 1. Nothing
   here upgrades the sign statement into a magnitude, and no statement of it may be written
   without its scope: "every landed setting" is false on the live tree at λ = 2.

**a = ½ (item 5's remaining sub-item): the χ_op 32 pair LANDED during fix round 1, and it
closes the sub-item NEGATIVELY at those knobs.** `s8_ahalf_chiop32_lam0` finished at 00:30
local (1957 s) beside `s8_ahalf_chiop32_lam1` (14837 s), so the pair now exists at L 32,
m 0.25, τ₀ 1.5, n_max 8, χ_op 32, weight 0.5, dt 0.375, order 4, n_steps 16 — and it is **not
usable**, for the same two reasons τ₀ ≥ 1 is not: its λ = 0 control is **1.18650**, 18.65 %
off 1, and the reference inconsistency is **175.3 %** of |E_min| at λ = 1 and **20.01 %** at
λ = 0, both far above the harvest's own 5 % gate. The raw readings are exact/E_H **2.03834**
at λ = 1 against the control, i.e. **1.71794** normalized (archived χ_op 16 reference: control
1.173, ratio 1.548). **No a = ½ ratio is quoted**, and `scaling_refit.a_half.rows[0].usable`
is `false` in the record. The last skip in `tests/test_interacting_qei_s8.py` is therefore
**gone** — not removed, but *fired into a pin*: the test now asserts the three landed values
to 1e-9 and asserts `usable is False`, `control_ok is False` and both gate failures, so a
future run that made the point usable would fail it rather than pass quietly. The two χ_op 48
rows are still queued; when they land, re-run `--s8-parts tau0 --no-new-jobs` and add them to
`_AHALF_PINS`. **Reconciled in fix round 2** (2026-09-13): `docs/STATUS.md` carried
*"the a = ½ pairs have still not landed — the one remaining skip"*, both halves false, and it
now carries the negative closure and the 61 passed / 0 skipped count; `docs/STATUS.md` and
`README.md` also carried the 2026-09-12 integration run's skip inventory naming
`tests/test_interacting_qei_s8.py:316`, and each now says in one clause that that skip has
since fired into a pin, leaving the historical run's own counts unchanged. `docs/PRIORITY.md`
names no a = ½ sentence. `draft/section_reference.md` §6.6.6 also said *"no a = ½ row at
χ_op 32 or 48 has landed"* and now reports the landed χ_op 32 pair and its failure.
See no-longer-claims item 12(c).

**Tests.** `tests/test_interacting_qei_s8.py` goes from 22 to **30** tests; with
`tests/test_exports.py` the pair reads **61 passed, 0 skipped** after fix round 1 (it read
60 passed, 1 skipped until the a = ½ control landed and turned that skip into a pin). Per-setting
numbers are pinned to 1e-9 relative because they are archived job values; the aggregates are
recomputed from the record's own rows inside the test and the stored value asserted equal, so
they do not go stale as the pool lands rows. Mutation-checked eight ways on a scratch copy
restored immediately (flipping `resolved`; widening the λ = 4 bracket; flipping a
gate-sensitivity resolution; erasing the one K_eff-negative setting; moving a small-λ c₂;
perturbing one E_min by 0.1 %; changing a closure hour; inflating an anchor maximum) — each
fails the intended test. **Fix round 1** added no test function (still 30) and no data other
than the 15th frozen tag; it added assertions and removed one vacuous guard. Mutation-checked
seven more ways on a scratch copy restored immediately (nulling the λ = 2 n_max spread;
shrinking it to 0.4 pp; setting the frozen λ = 2 K_eff sign count to 4 of 5; setting the LIVE
λ = 2 sign count to all-positive; moving the falsifying row's residual to −40 %; restoring the
23.421 h closure hour; restoring the 2.5 %-gate count to 3) — each fails the intended test and
the restored file is green at 60 passed, 1 skipped.

**Two things landed during this pass that falsified statements written minutes earlier.**
Both are recorded rather than quietly repaired, because the pattern — a pool row arriving
between the measurement and the prose — is the same hazard the fix-round-2 archive clobber
found.

1. **The no-overlap statement is frozen-snapshot-only, and the live tree already breaks it.**
   Item 4 (vi) says no λ ≤ 1 value of c₂ lies inside either λ ≥ 2 bracket. That is true of
   the frozen snapshot and *false* on the live tree as of the next harvest:
   **(n_max 10, χ_op 32, weight 1) at λ = 2** landed reading R_ctrl **−22.154 %**, which
   drags the live λ = 2 c₂ bracket to **[−0.0958, −0.0294]** and swallows the **λ = 0.1 and
   λ = 1** values (**−0.0679**, **−0.0553**). *(Fix round 1 correction: the first writing of
   this note named the λ = 0.25 and 0.5 values instead. Those two, −0.1061 and −0.1028, lie
   BELOW the bracket's lower edge −0.09584 and are still outside it; the conclusion — that the
   no-overlap statement is false on the live tree — is unchanged, only the identification of
   which points overlap.)* That row fails **every** gate on the tree — its own λ = 0
   control is **1.15541**, 15.5 % off 1; its reference inconsistency is **370.2 %** of
   |E_min|; its ED anchor is **18.2 %** — and it still widens the bracket, which is precisely
   why item 4 is bracketed and not resolved. The live λ = 2 systematic went from 32.3 to
   **109.7 bands** with it. The test asserts the no-overlap statement on the frozen verdict
   only and says so in its docstring.
2. **A third coarse-quadrature ED rung landed:** **(L 6, n_max 6, λ = 4)**, E_min
   **−8.353380718243208e-03**, `rel_quadrature_fine_minus_grid` **8.3906e-04** (Lanczos
   residual 5.7e-11), which made `tests/test_interacting_qei_ed.py`'s "every other rung is
   within 1e-4" assertion false on a file nobody had edited. The rung is **added** to
   `_S8_PIN_COARSE_QUADRATURE` with its measured deviation — the assertion is not weakened —
   and the pattern is now explicit: every coarse rung is a λ = 4 rung at the largest n_max
   reached for its L (L 5 n_max 6 1.3171e-03, L 6 n_max 6 8.3906e-04, L 5 n_max 8
   4.9612e-04). Any MPO-vs-ED comparison at those three points inherits the deviation.

**Reproducing / resuming (idempotent; starts nothing):**

```
.venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py --diagnostics --only s8 --s8-parts plane2 --no-new-jobs
.venv/bin/python -m pytest tests/test_interacting_qei_s8.py tests/test_exports.py -q -p no:cacheprovider
```

The archive-hazard warning above applies here too: pool A was started before this block was
added, so re-run the `plane2` harvest after it exits.

## Audit status

- **Anchor 1 (λ = 0).** `data/s1_free_anchor.csv`: at n_max = 8, |ΔE| = 1.5e-09,
  max|ΔV| = 9.8e-09, max|ΔS| = 2.2e-09, |ΔE_N| (1+1 adjacent) = 1.5e-07,
  |ΔE_N| (2+2 adjacent) = 5.8e-07; at n_max = 6, |ΔE_N| (2+2, separation 1)
  = 2.2e-07. Rayleigh–Ritz monotonicity in n_max holds (tested).
- **Anchor 2 (perturbation theory).** `tests/test_interacting.py`: ED-verified `E₁, E₂`;
  DMRG residual ratio 7.5 (target 8 for ∝ λ³); covariance residual ratio 3.5
  (target 4). In the sweep, `E₀ − E_PT2` = +8.25e-04 at λ = 0.25 (`s2_sweep.csv`).
- **Anchor 3 (χ-doubling).** Worst relative change over all reported sweep numbers
  above the resolution floors: 1.1e-03 (`s2_chi_doubling.json`; the 1e-5 floor
  for the proxy negativity exists because within 1e-7 of sudden death the float64
  partial-transpose route turns 1e-9 covariance differences into O(10%) relative
  changes — worst absolute change there 9.3e-9); QEI numbers: 3.0e-06
  (`s3_qei.csv`). Truncation errors are in every row.
- **Anchor 4 (passivity).** Every ground state of the sweep passes under 18 random
  one-/two-site unitaries (strengths 0.05–1); worst energy change +9.049e-03
  (`s4_audits.csv`). The audit's ability to fire is tested on an active state.
- **QEI pipeline.** λ = 0 TEBD vs exact Gaussian smeared energy: < 1e-5 (test);
  the `trunc_err` column of `s3_qei.csv` is TeNPy's accumulated discarded weight over
  all ~1500 bond updates of a trajectory (≤ 8.8e-4, a bookkeeping upper bound); the
  observable-level gate is the χ = 32 re-evaluation, `rel_change_2chi` ≤ 3.0e-6;
  dt-halving of the reported states: 1.1e-05; n_max 6→8→10 re-evaluation at
  λ = 1: 6: -1.13449e-02 / 8: -1.13775e-02 / 10: -1.13838e-02 (`s3b_qei_nmax.csv`).
- **Second implementation (anomaly protocol, `s5i_ed_control.json`).** Dense ED vs
  the MPO route at the same (L, n_max, λ), L = 6–8: with generous operator knobs
  (χ_op 64, plain norm, exact SVD, no window) the MPO infimum reproduces the ED one to
  ≤ 1.9e-4 relative at every such point — L = 6, n_max = 2–3 only; the generous rows at
  L = 6 n_max 4–5, L = 7 and L = 8 were not run; with the archived knobs it sits 1.5–6.3e-3 below
  the ED infimum at λ = 0 and 2.7–3.5e-2 *above* it at λ = 4 (n_max ≥ 3). Trotter and
  quadrature ≤ 6e-5 everywhere.
- **The O(λ) anchor.** `first_order_infimum_slope` vs ED central differences at L = 4,
  n_max = 9: 4e-4 relative; quadrature-converged to 1e-6 at the paper point.
- **Calibration band.** λ = 0 control 1.0019 at the sweep's settings (`s5h`); the same
  settings at λ = 4 are not converged (`s5g`, `s5f`).
- **Dispersion resolution (`s7a`).** At λ = 0 the pipeline returns the free ω(k) on all
  16 modes to 1.7e-6 and the DMRG gap to 4.5e-7; the K_eff pipeline reproduces
  `E_free_exact` and `E_H` to 1e-15.
- **Controls that failed and are not used.** τ₀ = 1.0 and 1.5 at the archived `s7d` knobs
  (n_max 6, χ_op 16, weight 0.25: λ = 0 control 1.085, 1.410), at the S8 recipe run at
  χ_op 16 (n_max 8, weight 0.5: 1.03379, 1.33774 — the first fails the 5 % reference-
  consistency gate at 9.8 %) and at χ_op 32 (1.0361, 1.2531), and the a = ½ point at the
  `s7e` knobs (control 1.173): no ratio is quoted from any of them; the a = ½ *dispersion* completed and is consistent (32 modes, weight ≥ 0.9993,
  γ/gap ≤ 7.3e-5, max |ω − ω_H| = 6.5e-3), so it is kept as a dispersion record only.
- **Anomaly protocol.** `summary.json`, `s5_summary.json`: `anomaly_events = []` — no
  state fell below the free exact infimum by more than the stated error at any λ, on
  either implementation. Nothing is open under the protocol; what is open is the
  λ ≥ 2 magnitude (Status).

## Headline numbers

| quantity | value | file |
|---|---|---|
| λ = 0 anchor, worst deviation at n_max = 8 | 5.8e-07 | `s1_free_anchor.csv` |
| ξ_MI at λ = 0 / 1 / 4 | 0.876 / 0.689 / 0.512 | `s2_fits.json` |
| ξ_MI ÷ ξ_MI^free(μ_H), worst over λ | 1.5% | `s2c_proxy_mixedness.csv` |
| Gaussian entropy of the interacting covariance at λ = 4 | 0.176 nats | `s2c_proxy_mixedness.csv` |
| E_N(2+2, sep 1) at λ = 0 / 1 / 4 | 4.090e-02 / 3.104e-02 / 1.935e-02 | `s2_sweep.csv` |
| proxy death separation (8+8) at λ = 0 / 4 vs free at μ_H | none in d ≤ 14 / 2 vs none in d ≤ 14 / 11 | `s2_fits.json`, `s2c_proxy_mixedness.csv` |
| free exact infimum `E_free_exact` (τ₀ = 0.75, m = 0.5) | -2.2654e-02 | `s3_qei.csv` |
| Flanagan continuum bound for this `f` | -6.2688e-02 | `s3_qei.csv` |
| family quality at λ = 0, `E_free_family/E_free_exact` | 0.723 | `s3_qei.csv` |
| E_var at λ = 0 / 1 / 4 | -1.6366e-02 / -1.1377e-02 / -5.1819e-03 | `s3_qei.csv` |
| E_var / E_free_exact at λ = 0 / 1 / 4 | 0.722 / 0.502 / 0.229 | `s3_qei.csv` |
| E_var / E_hartree_exact at λ = 0 / 1 / 4 | 0.722 / 0.723 / 0.689 | `s3_qei.csv` |
| worst χ-doubling (sweep / QEI) | 1.1e-03 / 3.0e-06 | `s2_chi_doubling.json`, `s3_qei.csv` |
| λ = 0 anchor, exact-MPO vs Gaussian infimum (archived knobs n_max 6, χ_op 16, w 0.25) | −1.27e-02 rel — the Fock dynamics, χ_op-independent | `s5a_free_anchor_exact.csv`, `s5g_calibration.csv` |
| λ = 0 band at the sweep's settings (n_max 8, χ_op 32, w 0.5) | **0.19 %** (0.16 % at χ_op 64) | `s5h_lambda_sweep.csv`, `s5g` |
| exact/E_H at λ = 0.25 / 0.5 / 1 / 2 / 3 / 4 (sweep settings) | 0.9883 / 0.9871 / 0.9763 / 0.9752 / 0.8988 / 0.8834 | `s5h_lambda_sweep.csv` |
| residual 1 − exact/E_H, control-normalized, λ ≤ 1 | 1.4 / 1.5 / 2.6 % (± 0.19 %) — **resolved** | `s5h_lambda_sweep.json` |
| exact O(λ) slope of the residual, dR/dλ at 0 | **+0.081** (L = 16; 0.089 / 0.082 / 0.081 at L = 6 / 7 / 8) | `s5h_lambda_sweep.json`, `s5i_ed_control.json` |
| ... its pieces (per unit λ) | E_min: +0.0119 = quartic ⟨:φ⁴:⟩ +0.0045 + dynamical +0.0075; E_H: +0.0101 | `s5h_lambda_sweep.json` |
| λ = 4 n_max ladder ON THE INFIMUM, archived knobs, n_max 6…12 | 0.873, 0.905, 0.954, 0.999, 0.986, 0.976, 0.993 — **not monotone** | `s5f_nmax_ladder_lam4.csv` |
| exact/E_H at λ = 4 across the settings run | 0.87–1.04 (**unresolved**) | `s5f`, `s5g`, `s5h` |
| MPO vs ED at the same (L, n_max, λ): generous knobs (L = 6, n_max 2–3 only) / archived knobs (L = 6–8, n_max 2–4) | ≤ 1.9e-4 / 1.5–6.3e-3 (λ = 0), −4.5e-3 to +3.5e-2 (λ = 4) | `s5i_ed_control.json` |
| ED λ = 0 Fock ladder of the smeared infimum (L = 4; L = 6 at τ₀ = 0.75) | sign-alternating: +1.9e-3 … +9.7e-5 (n_max 4…9); +7.9 % → −1.7 % (n_max 3 → 4) | `s5i_ed_control.json`, `tests/test_interacting_qei_ed.py` |
| interacting dispersion at λ = 4: quasiparticle weight / γ/gap / band range | ≥ 0.9986 / ≤ 1e-4 / nearest-neighbour to 7e-5 (`μ_eff² = 0.879`, `J₁ = 0.999`) | `s7a_dispersion.csv` |
| gap² vs Hartree at λ = 4 | 0.879 vs 0.909 (−3.3 %) | `s7a`, `s7b_reference.csv` |
| E_free(K_eff)/E_H at λ = 1 / 4 (cutoff-free) | 1.0090 / 1.0452 | `s7b_reference.csv` |
| exact/E_free(K_eff) at λ = 4: archived knobs; n_max 8 / 9 / 12 same knobs; (n_max 6, χ_op 64, w 0.5) | 0.835; 0.913 / 0.956 / 0.950; 0.901 — all values *at their knobs* | `s7b`, `s5f`, `s5g` |
| quartic piece of the extremal state, λ = 1 / 4 (archived knobs; H-audit −3.6 % / −10.8 %) | **+8.7–8.8 % / +9.7–10.7 %** of \|E_min\| (positive) | `s7c_decomposition.csv` |
| τ₀ exponent of the residual (τ₀ = 0.375–0.75, archived knobs) vs prediction | −1.2 ± 0.8 (K_eff), −1.1 ± 1.1 (E_H) vs +2 | `s7d_tau0_sweep.csv`, `s7_summary.json` |
| L = 16 → 24 at λ = 4 | E_min −1.1 %, exact/E_H 0.873 → 0.882 | `s7e_controls.csv` |
| free bound survives: exact/free at λ = 4 | 0.29–0.33 at every cutoff tested | `s5b`, `s5e`, `s5f`, `s5h` |
| variational family gap (λ = 0 / 1 / 4) | 0.712 / 0.750 / 0.777 of the infimum | `s5b_exact_vs_lambda.csv` |
| notebook wall time | 33.3 min (1995 s) | `summary.json` |
| **S8** λ = 0.1 at the sweep's settings: exact/E_H; control-normalized residual vs the O(λ) prediction | 0.99450; **0.741 % vs 0.809 %** (−0.35 bands; the ≈ 1.4 % floor reading excluded at 3.3 bands) | `s8_lam0.1.json` |
| **S8** τ₀ = 1.0 / 1.5 λ = 0 controls at the S8 recipe (n_max 8, w 0.5, χ_op 32, χ_dmrg 16) | 1.0361 / 1.2531 — **no ratio quoted**: the λ = 4 rows' reference inconsistency is 38 % / 113 % of their E_min (exact/E_H there 1.581 / 26.8, values at their knobs). The archived 1.085 / 1.410 are `s7d` at n_max 6, w 0.25, χ_op 16, χ_dmrg 12 — six knobs apart, so the improvement is **not** attributable to χ_op; see the isolation row | `s8_tau0_rows.json` |
| **S8** τ₀ ≥ 1 rows at χ_op 48 (n_max 8, w 0.5, χ_dmrg 16), fix round 1–2 | τ₀ 1.0: λ = 0 E_min −1.4356626e-02 / 1.038428 (refcons 7.65 % of E_min), λ = 4 E_min −4.0700781e-03 / 1.626150 (47.6 %). τ₀ 1.5: λ = 0 E_min −5.3538589e-03 / 1.217421 (64 %), λ = 4 E_min −1.9248737e-03 / 10.84228 (654 %, norm 0.800). **No ratio quoted from any of them** — every one fails a gate | `s8_tau0_rows.json` |
| **S8** E_free(K_eff)/E_H at λ = 2 (s7a pipeline; μ_eff² 0.5995 vs Hartree 0.6125, J₁ 0.9997, weight ≥ 0.9992, γ/gap ≤ 2e-4) | **1.0210** (λ = 1 / 4: 1.0090 / 1.0452) | `jobs/s8_keff_lam2.json`, `s8_convergence_plane.json` |
| **S8** ED matrix-free vs archived dense rows, same grid (L 6, n_max 3–4, λ 0 / 4) | 1.6e-13 … 5.6e-12 relative | `s8_ed_anchor.json` |
| **S8** ED L = 6 λ = 0 ladder, exact/E_H at n_max 3 / 4 / 5 | 0.9214 / 1.0173 / 1.0062 (non-monotone, as the dense ladder) | `s8_ed_anchor.json` |
| **S8** τ₀ = 1.5 λ = 0 control vs χ_op at fixed n_max 8, w 0.5, χ_dmrg 16 | **1.33774 (χ_op 16) → 1.25306 (32) → 1.21742 (48)**; steps −0.0847, −0.0356 — monotone toward 1, shrinking ≈ 0.42 per step, still 21.7 % off 1; its own reference inconsistency runs 2.3 % → 117 % → 64 % of E_min, failing the 5 % gate above χ_op 16 | `s8_tau0_rows.json` (`control_chi_op_sequence`) |
| **S8** τ₀ = 1.0 λ = 0 control vs χ_op at the same fixed knobs (fix round 2) | **1.03379 (χ_op 16) → 1.03614 (32) → 1.03843 (48)**; steps **+0.00235, +0.00229** — monotone *away* from 1; reference inconsistency 9.8 % → 2.7 % → 7.65 % of E_min, back across the 5 % gate at χ_op 48 | `s8_tau0_rows.json` (`control_chi_op_sequence`) |
| **S8** the χ_op axis isolated: τ₀ = 1.0 / 1.5 λ = 0 control at the S8 recipe run at χ_op 16 (n_max 8, w 0.5, χ_dmrg 16) | **1.03379 / 1.33774** — so the s7d → S8 change splits into (n_max, w, χ_dmrg) **−0.0512 / −0.0723** and χ_op 16 → 32 **+0.0023 / −0.0847**: at τ₀ = 1.0 χ_op moves the control *away* from 1 | `s8_tau0_rows.json` (`chiop16_isolation`) |
| **S8** ED λ = 4 ladder at L = 6, exact/E_H at n_max 3 / 4 / 5; L = 5 at n_max 6 | 0.4931 / 0.8536 / **0.8789** (still rising: +0.0253 from n_max 4 → 5, against −0.0111 in the λ = 0 ladder); 1.4292 at L = 5 | `s8_ed_anchor.json` |
| **S8** the first plane point (n_max 8, χ_op 32, w 0.5, **exact SVD**): λ = 0 control; λ = 4 ratio; control-normalized residual vs E_H / K_eff | 1.02261; 0.95979; **+6.14 % / +10.20 %** — **not usable**: reference inconsistency 56.5 % of |E_min| (gate 5 %), control 2.3 % off 1 (worse than the sweep's 1.00192), one setting so `resolved` = false | `s8_convergence_plane.json` |
| **S8** exact ED ladders at L = 5: exact/E_H at λ = 0, n_max 6 / 7 / 8; at λ = 4, n_max 6 / 7 / 8 | 1.000739 / 0.999114 / **1.000049** (converged on 1 to 4.9e-5) vs **1.4292 / 2.3551 / 3.2646** at n_max 6 / 7 / 8 (E_min −9.4393e-03 → −1.5554e-02 → −2.1561e-02: −64.8 % then −38.6 %, residuals ≤ 8.5e-11) | `s8_ed_anchor.json` |
| **S8** MPO vs exact ED on the same (L, n_max, grid) at the **sweep's knobs** (χ_op 32, w 0.5, sketched), λ = 0 / λ = 4, at n_max 5 / 6 / 8 | λ = 0: +3.36e-3 (L 6 n5), +6.36e-3 (L 5 n6), +1.38e-2 (L 5 n8). λ = 4: +3.37e-2 (L 6 n5), −1.19e-1 (L 5 n6), **+3.94e-1 (L 5 n8)** — the λ = 4 error **grows with n_max** and is largest at the sweep's own cutoff | `s8_ed_anchor.json` |
| **S8** the knob ladder at L 5, n_max 8, λ = 4 vs the exact E_min −2.1561127e-02 (τ₀ 0.75, sketched SVD unless noted) | χ_op 32 / 48 / 64 / 96 at w 0.5, χ_dmrg 16: **+39.42 % / +14.88 % / +13.670 % / −4.50 %** (non-monotone, sign change between 64 and 96). χ_dmrg 32 at χ_op 64: +13.669 % (the DMRG bond is converged). **weight 1 at χ_op 64: −1.19 %** — the weighting, not χ_op or χ_dmrg, is what costs the λ = 4 accuracy | `s8_ed_anchor.json` |
| **S8** the 5 % reference-consistency gate vs the true λ = 4 error, same seven rows | refcons/E_min 6.73 / 33.90 / 2.66 / **1.21** / 1.21 / 6.56 / **36.06 %** against errors +39.42 / +42.25 / +14.88 / **+13.670** / +13.669 / −4.50 / **−1.19 %**: the best refcons is 13.7 % wrong and the best row (1.19 %) is seven gates out. **The gate is necessary, not sufficient, and not monotone in the error** | `s8_ed_anchor.json` |
| **S8** the same χ_op 64 / w 0.5 knobs at λ = 0, L 5, n_max 8 | **+1.21 %** off exact (E_min −1.8239597e-02, refcons 1.29 %) — a factor 11 below the λ = 4 error at identical knobs | `s8_ed_anchor.json` |
| **S8** MPO vs exact ED at the **plane's knobs** (χ_op 32, w 0.5, **exact** SVD) at L 5, n_max 8 | λ = 0: −3.89e-3; λ = 4: **+4.23e-1** (MPO E_min −1.2451097e-02 vs exact −2.1561127e-02, exact/E_H 1.885 vs 3.265 — the route understates the magnitude of E_min by 1.7317×; 1.6508× at the sweep's knobs). Generous knobs (χ_op 64, w 1, exact SVD) at L 6 n5: −1.65e-3 / −1.52e-2 | `s8_ed_anchor.json` |
| **S8 triage** the exact SVD at identical knobs, thread-matched (2 threads): L 16 n_max 8 χ_op 32 w 0.5 λ = 0 (exact/E_free must be 1) | **0.999931 (sketched, 1353 s) → 1.022613 (exact SVD, 10074 s)** for **7.4×** the wall. Band-aware penalty **≥ 11.8×** further from 1 (exact-SVD distance 2.261e-02 over the sketched leg's largest archived distance 1.923e-03, at 4 threads); the sketched leg is **inside** its own 0.199 % same-spec thread band and is not resolved from 1; the exact-SVD leg is **11.35× the sketched leg's band** and its **own** band is unmeasured (one archived thread count — ASSUMPTION, item 10). The **328×** of fix round 0 is retracted (item 9) | `s8_weight_triage.json` → `banded_penalties_l16_lam0` |
| **S8 triage** the exact SVD at L 5 n_max 8 λ = 4 vs exact E_min −2.1561126717e-02 | **+39.425 % (171 s) → +42.252 % (3127 s)**, i.e. worse, for **18.3×** the wall (at λ = 0 it wins: +1.380 % → −0.389 %, 22.0×) | `s8_weight_triage.json` |
| **S8 triage** weight 0.5 → 1 at L = 16, λ = 0, identical knobs | n_max 8 χ_op 32: exact/free **0.999931 → 1.082086** thread-matched, **1.001923 → 1.081776** at 4 threads; band-aware penalty **≥ 42.5×** worse (8.178e-02 over 1.923e-03) — weight 1's 8.18 % is 285× its own 0.029 % band, weight 0.5's distance from 1 is inside its 0.199 % band; cost **1.14×**. The **1190×** of fix round 0 is retracted (item 9). n_max 6 χ_op 32: 1.011667 → 1.035981, cost 1.26×; n_max 6 χ_op 32 exact SVD: 1.022527 → 1.036178, cost **1.00×**. The L = 5 ranking (weight 1 better by 11× at λ = 4) **does not transfer** | `s8_weight_triage.json` → `banded_penalties_l16_lam0` |
| **S8 triage** the weight ladder at L 5, n_max 8, χ_op 64, λ = 4 (1 thread, sketched) vs exact −2.1561126717e-02 | w 0.25 / 0.5 / 0.75 / 1.0: **+38.613 % / +15.718 % / −11.834 % / +0.463 %** — non-monotone, sign change between 0.5 and 0.75 and back; w = 1 is closest but is **not** a one-sided approximation | `s8_weight_triage.json` |
| **S8 triage, fix round 2** both SVD routes at the same spec, 2 vs 4 BLAS threads (L 16, n_max 6, χ_op 16, χ_acc 32, χ_mpo 64, w 0.5, τ₀ 0.75, λ = 0, χ_dmrg 16) | **bit-identical on both legs**: sketched E_min **−2.2800211601990195e-02** (438.4 s at 2 thr, 159.1 s at 4), exact SVD **−2.2865136075353676e-02** (767.8 s, 353.9 s); both `rel_spread_E_min` **exactly 0.0**. The exact SVD is **not** the noisier route where both bands are measured — the ASSUMPTION behind the 11.35× at n_max 8 — and the 0.199 % spread there is **not shown to be a thread effect**. Cost 1.751× (2 thr) / 2.225× (4 thr) | `jobs/s8_svdband_{exactsvd,sketched}_chiop16_wd0.5_thr4.json`, `s8_weight_triage.json` (`repeat_runs_same_spec`, `banded_penalties_l16_lam0`) |
| **S8 triage, fix round 2** the worst same-spec spread in the archive, `wt` grid complete (16 repeat groups) | **5.120 %** at L 5, n_max 6, χ_op 32, w 0.5, λ = 4, sketched, 1 vs 2 threads (−1.0049170e-02 vs −1.0563711e-02), up from 2.430 %. The route's same-spec reproducibility spans **0.0 % to 5.12 %** and is spec-dependent | `s8_weight_triage.json` (`worst_rel_spread_same_spec`) |
| **S8 triage** the route's reproducibility at λ = 4: same spec, 2 vs 1 BLAS thread (seed fixed at 0) | L 5 n_max 8 χ_op 64 sketched — w 0.5: −1.8613818e-02 vs −1.8172225e-02, **2.430 %** spread; w 1: −2.1817208e-02 vs −2.1461251e-02, **1.659 %**. A floor larger than the w = 1 advantage it would establish | `s8_weight_triage.json` (`repeat_runs_same_spec`) |
| **S8 triage** exact ED at **λ = 2** (matrix-free Lanczos, tol 1e-10, ncv 80, τ₀ 0.75) | L 5 n_max 6: **−9.385658023800647e-03** (exact/E_H 0.922426, fine-grid 3.6e-6); L 6 n_max 5: **−1.0118762207130372e-02** (0.931663, 5.6e-6) | `jobs/s8_edlz_L5_nmax6_lam2.json`, `jobs/s8_edlz_L6_nmax5_lam2.json` |
| **S8 triage** the weight ladder at λ = 2 and λ = 0 (L 5, n_max 8, χ_op 64, 1 thread, sketched) vs exact −9.388966078354377e-03 / −1.8462671543e-02 | λ = 2: w 0.25 **+1.435 %**, w 0.5 **+0.753 %**, w 0.75 **−63.331 %**, w 1 **−0.845 %**. λ = 0: w 0.5 +1.268 %, w 0.75 −1.056 %, w 1 −0.272 %. **At λ = 2 three of the four rungs are inside 1.5 %** — "the weighting is the limiter at λ ≥ 2" is a λ = 4 statement only | `s8_weight_triage.json` |
| **S8 triage** the same-spec thread spread is λ-dependent | L 5 n_max 8 χ_op 64 w 0.5 sketched: **0.061 %** at λ = 0 against **2.430 %** at λ = 4 | `s8_weight_triage.json` (`repeat_runs_same_spec`) |
| **S8 triage** exact ED at λ = 2, L 5 n_max 8 (the paper's Fock cutoff) | **−9.388966078354377e-03**, exact/E_H 0.922751, fine-grid quadrature 8.6e-7, 2791 s | `jobs/s8_edlz_L5_nmax8_lam2.json` |
| **S8 triage** the route's thread scatter at the sweep's own L = 16 settings (n_max 8, χ_op 32, w 0.5, sketched; 2 vs 4 threads, seed 0 both) | λ = 0 **0.199 %** (−2.2652090e-02 / −2.2697211e-02), λ = 2 **2.031 %** (−1.1347773e-02 / −1.1578241e-02), λ = 4 0.142 %. The quoted "0.19 % λ = 0 control" is the size of this scatter | `s8_weight_triage.json` (`repeat_runs_same_spec`) |
| **S8 triage** the wall model's calibration, per part, applied to both sides | retired `plane` (exact SVD) **2.282 / 2.371×**, median **2.326×** (n = 2) → 368 h nominal = **856.1 h**; `plane2` (sketched, same grid) **0.903–1.880×**, median **1.677×** (n = 7, the frozen `fix_round_1` tag set; the live `per_part` median drifts as the pool lands rows) → 43.556 h nominal = **73.03 h**; exact-SVD-specific excess **1.388×**; like-for-like saving **11.72×** (nominal saving 8.45×). **37.81 h of plane2's 43.56 h (87 %) is n_max 10/12 extrapolation** — no sketched job above n_max 8 is timed on this tree | `s8_weight_triage.json` → `wall_model_calibration` |
| **S8 triage** the requeue | `plane` retired (18 jobs, 368 h nominal / **856 h calibrated**) → **`plane2` 36 jobs, 43.6 h nominal / 73.0 h calibrated** (same grid, sketched SVD, weight an anchored axis 0.5/1.0) + **`wt` 72 jobs 5.8 h** (the ED-anchored weight ladder) + 3 λ = 2 Lanczos references. Plan **161 jobs / 99.2 h nominal**, was 70 / 418 h nominal | `s8_plan.json`, `s8_weight_triage.json` |
| **S8 item 4** the λ = 2 bracket of the control-normalized residual (`plane2`, L 16, sketched SVD; **frozen 15-row snapshot**, 5 settings) | **[−1.797 %, +4.415 %]** vs `E_H`, **[+0.301 %, +6.385 %]** vs `E_free(K_eff)`. exact/E_H 0.955782 / 1.101529 / 0.967108 / 1.027187 / 1.023109 and exact/K_eff 0.936088 / 1.078831 / 0.947180 / 1.006021 / 1.002028 at (n_max 8, χ_op 32, w 0.5) / (8, 32, w 1) / (8, 64, w 0.5) / (8, 64, w 1) / (**10, 32, w 0.5**), each against its own λ = 0 control 0.9999311 / 1.0820864 / 0.9992576 / 1.0171625 / 1.0124217; R_ctrl +4.415 / −1.797 / +3.217 / **−0.985** / −1.056 %. **BRACKETED, not resolved**: systematic 6.212 pp = **32.3 bands** of the sweep's 0.19229 %. *(The live `verdict["lam=2"]` beside the snapshot reads [−22.154 %, +4.415 %] over 6 settings, 109.7 bands, and moves as the pool lands rows.)* | `s8_convergence_plane2.json` → `unit2_snapshot.verdict["lam=2"]` |
| **S8 item 4** the λ = 4 bracket, same frozen snapshot (5 settings, n_max 8 and 10) | **[−34.349 %, +15.131 %]** vs `E_H`, **[−28.541 %, +18.800 %]** vs `E_free(K_eff)` — **it spans both signs**. exact/E_H 0.884635 / 1.453771 / 0.848060 / 1.034773 / 0.903053 at (8, 32, 0.5) / (8, 32, 1) / (8, 64, 0.5) / (8, 64, 1) / (10, 32, 0.5). Systematic 45.879 pp = **238.6 bands**. *(Live: 6 settings, same systematic.)* | `s8_convergence_plane2.json` → `unit2_snapshot.verdict["lam=4"]` |
| **S8 item 4** which axis carries the systematic at λ ≥ 2 (frozen 15-row snapshot) | λ = 2: weight 0.5 → 1 at χ_op 32 **6.212 pp**, **n_max 8 → 10 at (χ_op 32, w 0.5) 5.471 pp (= 28.5 bands)**, χ_op 32 → 64 at w 0.5 **1.198 pp**, the route's own same-spec floor at the identical spec **1.941 pp** (= 10.1 bands, larger than the χ_op axis). λ = 4: weight **45.879 pp**, χ_op **32.618 pp**, n_max 8 → 10 **0.728 pp** (= 3.8 bands), same-spec floor 0.126 pp. **The weight axis dominates at both λ. n_max is the smallest axis at λ = 4 only: at λ = 2 it is the second largest, 28.5 bands, and every axis is above the 3-band criterion at both λ** | `s8_convergence_plane2.json` → `unit2_snapshot.verdict[*]["spreads"]` |
| **S8 item 4** the ED-anchor gate at λ ≥ 2 (max \|rel dev vs exact ED\| over every anchored point at the same χ_op, weight, λ and SVD mode, L ≤ 6) | λ = 2: (32, 0.5) **2.330 %**, (32, 1) **18.214 %**, (64, 0.5) **1.551 %** ✓, (64, 1) **2.138 %** (misses by 0.14 pp). λ = 4: (32, 0.5) **39.425 %**, (32, 1) **26.424 %**, (64, 0.5) **34.489 %**, (64, 1) **1.672 %** ✓. **One setting per λ clears 2 %, and they are different settings** | `s8_convergence_plane2.json` → `anchor_table`, `unit2_snapshot.verdict[*]["anchor_qualified"]` |
| **S8 item 4** the anchor-qualified residual, quoted as a point only under its own setting | λ = 2 at (χ_op 64, weight 0.5): R_ctrl **+3.217 %** vs `E_H`, **+5.212 %** vs K_eff, anchor ≤ 1.551 % over 3 points. λ = 4 at (χ_op 64, weight **1**): R_ctrl **−1.731 %** vs `E_H`, **+2.666 %** vs K_eff, anchor ≤ 1.672 % over 4 points. **Not a λ-series** — two different operator representations — and one setting per λ gives no systematic | `s8_convergence_plane2.json` → `unit2_snapshot.verdict[*]["anchor_qualified"]` |
| **S8 item 4** the λ ≥ 2 bracket against the exact first-order slope dR/dλ = +0.0809088 | O(λ) predicts **+16.182 %** (λ = 2) and **+32.364 %** (λ = 4); **every landed setting lies below it at both λ**. Implied c₂ = (R − 0.0809 λ)/λ²: **[−0.0449, −0.0294]** at λ = 2, **[−0.0417, −0.0108]** at λ = 4, against **−0.0679 / −0.1061 / −0.1028 / −0.0553** at λ = 0.1 / 0.25 / 0.5 / 1 — **on the frozen snapshot no λ ≤ 1 value overlaps either λ ≥ 2 bracket**, so one λ-independent c₂ fits neither range together. *(On the live tree the λ = 2 c₂ bracket is already [−0.0958, −0.0294], which contains the λ = 0.1 and λ = 1 values.)* | `s8_convergence_plane2.json` → `unit2_snapshot.verdict[*]["implied_second_order_coefficient"]`, `implied_second_order_coefficient_small_lambda` |
| **S8 item 4** every λ ≥ 2 `plane2` row of the frozen snapshot against the harvest's own 5 % reference-consistency gate | **10 of 10 fail**: 20.30 / 198.91 / 10.96 / 15.91 / 26.31 % at λ = 2 and 5.27 / 257.29 / 7.24 / 113.84 / 27.70 % at λ = 4. The gate was already measured anti-correlated with the true λ = 4 error, so it neither certifies nor disqualifies a row | `s8_convergence_plane2.json` → `unit2_snapshot.verdict[*]["rows_failing_refcons_gate"]` |
| **S8 item 4** what closure would cost, at the plane2 wall calibration measured over the frozen snapshot's own jobs (**1.5038×**, n = 15 two-thread jobs) | the n_max 10 / 12 rungs of the λ = 4-qualified column (χ_op 64, w 1): **6 jobs, 13.968 h nominal / 21.005 h calibrated**; the same for the λ = 2-qualified column (χ_op 64, w 0.5): **6 jobs, 13.968 h / 21.005 h**; both **12 jobs, 27.936 h / 42.010 h**. It would *test* resolution, not deliver it: a **second** anchor-qualified (χ_op, weight) pair is also required and none is on the queue | `s8_convergence_plane2.json` → `unit2_snapshot.closure_cost` |
| **S8 item 4** the verdict against the anchor gate's own value (gates 1 / 2 / 2.5 / 3 / 5 %; frozen 15-row snapshot) | λ = 2: **0 / 1 / 4 / 4 / 4** settings qualify; at 2.5 % and above the qualified bracket is **[−1.056 %, +4.415 %]**, spread **28.5 bands**. λ = 4: **0 / 1 / 1 / 1 / 1** — no gate in that range qualifies a second setting. **At no gate from 1 % to 5 % does either λ resolve**, so the verdict is not an artefact of the 2 % choice | `s8_convergence_plane2.json` → `unit2_snapshot.verdict[*]["anchor_gate_sensitivity"]` |
| **S8 item 4** the SIGN of the λ ≥ 2 residual, which survives more of the plane than its size | against **K_eff**, **on the frozen 15-row snapshot**: positive at **5 of 5** settings at λ = 2 and **4 of 5** at λ = 4, the exception (n_max 8, χ_op 32, w 1) being **the frozen snapshot's** worst-diagnosed row (reference inconsistency **257.3 %** of \|E_min\|, the largest of its 15 rows, and a 26.4 % ED anchor) — **not** the tree's worst, which is **370.2 %** at the λ = 2 exception named next. **On the live tree at this commit (18 of 36 rows): 5 of 6 and 4 of 6** — the extra λ = 2 negative is (n_max 10, χ_op 32, w 1) at **−19.637 %**, control 15.5 % off 1, refcons 370.2 %, ED anchor 18.2 %. Against `E_H`: **2 of 5** and **3 of 5** frozen — not robust, because the weight-1 columns' λ = 0 controls sit 1.72 % and 8.21 % above 1 | `s8_convergence_plane2.json` → `unit2_snapshot.verdict[*]["n_positive_keff"]` (frozen), `verdict[*]["n_positive_keff"]`, `negative_keff_settings` (live) |
| **S8 item 5 / a = ½** the χ_op 32 pair, landed 2026-09-13 — **not usable**, so no ratio is quoted | λ = 0 control **exact/E_H = 1.1865002227506591** (18.65 % off 1, E_min −1.7665588525033193e−02, 1957 s); λ = 1 **exact/E_H = 2.03833904983217** (E_min −7.7595670293511e−03, 14837 s), **1.7179424080567773** normalized by the control. Reference inconsistency **175.32 %** of \|E_min\| at λ = 1 and **20.01 %** at λ = 0, both above the 5 % gate; `control_ok false`, `usable false`. Archived χ_op 16 reference: control 1.173, ratio 1.548 | `s8_tau0_rows.json` → `scaling_refit["a_half"]["rows"]`, `rows[axis=a_half]` |

## Status

- **Entanglement-structure numbers: ready to draft** as lattice results at one mass.
- **QEI method: the infimum, not a bound — ready to draft at λ = 0**, with a second
  implementation (dense ED) agreeing to ≤ 1.9e-4 where the first is converged (L = 6,
  n_max ≤ 3, generous knobs). Beyond that the agreement degrades and is λ-dependent:
  archived-knob comparisons are 0.05–3.5 % off; at the sweep's own knobs the λ = 0 pairs
  are 0.3–1.4 % off at n_max 5–8, but the λ = 4 pairs are 3.4 % → 11.9 % → **39.4 %** off
  at n_max 5 / 6 / 8 (42.3 % at the plane's knobs, n_max 8). **The method is cross-validated
  at λ = 0 and is not cross-validated at λ = 4 at the cutoff the paper uses.**
- **QEI mechanism: ready to draft as the finding.** The residual beyond mass
  renormalization is a first-order quartic tax (exact slope +0.081 per unit λ, sign
  measured independently on the extremal state); renormalization of the reference is
  a mass shift (sharp quasiparticle, nearest-neighbour band, gap² 3.3 % below Hartree),
  and against the sharpest quadratic reference the residual is larger.
- **QEI magnitude at λ ≤ 1: ready to draft** as 1.4–2.6 % ± 0.2 % (control-normalized,
  against `E_H`; 0.9 % more against K_eff at λ = 1).
- **QEI magnitude at λ ≥ 2: BRACKETED, not resolved — a bracket is quotable, a value is
  not.** On the frozen unit-2 snapshot — the **15** `plane2` rows that had landed with their own λ = 0
  controls when this was written, named by tag in `s8_convergence_plane2.json` so the quote
  stays reproducible (L = 16, τ₀ 0.75, sketched SVD, each column anchored to exact ED at
  L ≤ 6) the control-normalized residual
  spans **[−1.797 %, +4.415 %]** against `E_H` at λ = 2 (**[+0.301 %, +6.385 %]** against
  K_eff) and **[−34.349 %, +15.131 %]** at λ = 4 (**[−28.541 %, +18.800 %]**), the λ = 4
  bracket spanning both signs. The systematic is **32.3 bands** of the sweep's 0.19229 %
  at λ = 2 and **238.6 bands** at λ = 4 — 11× and 80× the 3-band quotability criterion —
  and it is carried by the **weight** axis at both λ (6.212 and 45.879 pp), not by χ_op
  (1.198 and 32.618 pp). The n_max axis is the smallest **at λ = 4 only** (0.728 pp = 3.8
  bands, the only spread anywhere near the criterion and still above it); **at λ = 2 it is the
  second largest axis, 5.471 pp = 28.5 bands** at (χ_op 32, weight 0.5). (On the live tree it
  is larger still: the (χ_op 32, weight 1) column's 8 → 10 move at λ = 2 is 20.358 pp.) The archived "≥ 0.905 at λ = 4" stays withdrawn as a bound.
  **What unit 2 of the triage added:** (a) the weighting finding no longer rests on one
  point — at λ = 4 the (χ_op 64, weight 1) column is the only one inside 2 % of exact ED at
  all three anchored points (L 5 n_max 6 −0.517 %, L 5 n_max 8 +0.463 % / −1.188 %, L 6
  n_max 5 −1.672 %), against 34.5 / 15.7 / 3.4 % for (χ_op 64, weight 0.5); (b) the
  ED-anchor gate clears exactly **one** setting per λ and they are **different** settings
  (χ_op 64 weight 0.5 at λ = 2 giving **+3.217 %**, χ_op 64 weight **1** at λ = 4 giving
  **−1.731 %**), so there is no spread to measure and no λ-series to read; (c) the verdict
  is **not an artefact of the gate's value** — at gates 1 / 2 / 2.5 / 3 / 5 % the λ = 2
  qualified count is 0 / 1 / 4 / 4 / 4 with a 28.451-band spread wherever it exceeds one, and
  λ = 4 never exceeds one; (d) the **sign** survives more of the plane than the size —
  against K_eff the residual is positive at 5 of 5 settings of the frozen snapshot at λ = 2
  and 4 of 5 at λ = 4 (the exception being **the frozen snapshot's** worst-diagnosed row,
  reference inconsistency 257.3 %; the *tree's* worst is 370.2 %, at the λ = 2 exception)
  — **on the live tree, 5 of 6 and 4 of 6** — while against `E_H` it
  is 2 of 5 and 3 of 5 frozen. **Every λ ≥ 2 row fails the harvest's own
  5 % reference-consistency gate** (5.27 % to 257.29 %), a gate already measured
  anti-correlated with the true error. Against the exact first-order slope +0.0809088 both
  brackets lie **entirely below** the O(λ) prediction (+16.182 % and +32.364 %), and the
  implied c₂ — [−0.0449, −0.0294] at λ = 2, [−0.0417, −0.0108] at λ = 4 — **does not overlap
  any λ ≤ 1 value on the frozen snapshot** (−0.0679 / −0.1061 / −0.1028 / −0.0553 at
  λ = 0.1 / 0.25 / 0.5 / 1), so one λ-independent quadratic fits neither range together
  *there*; on the live tree the λ = 2 c₂ bracket is already [−0.0958, −0.0294] and contains
  the λ = 0.1 and λ = 1 values. **Cost to test closure:** the
  n_max 10 / 12 rungs of both anchor-qualified columns are **12 jobs, 27.936 h nominal /
  42.010 h calibrated** at the 1.5038× plane2 calibration measured over the snapshot's own
  15 jobs; that would test
  resolution, not deliver it, because the criterion also needs a second anchor-qualified
  (χ_op, weight) pair that no landed setting provides. ASSUMPTION, stated and not proved:
  the (χ_op, weight) truncation error measured against exact ED at L ≤ 6, n_max ≤ 8
  transfers to L = 16, n_max 8–12; no exact answer exists at L = 16 at any λ > 0, which is
  why the verdict is a bracket.
- **Scaling: ready to draft as a null result within the resolved τ₀ range**
  (0.375–0.75): no shift of the exponent. **τ₀ ≥ 1: the χ_op step has now been taken and it
  does not rescue these rows.** Every τ₀ ≥ 1 row run at χ_op 16, 32 and 48 (n_max 8,
  weight 0.5, χ_dmrg 16) fails the harvest's own gates: at τ₀ = 1.5 the control improves
  monotonically (1.33774 → 1.25306 → 1.21742) but is still 21.7 % off 1 and its reference
  inconsistency runs 2.3 % → 117 % → 64 % of |E_min|; at τ₀ = 1.0 the control moves *away*
  from 1 (1.03379 → 1.03614 → 1.03843) and its inconsistency crosses back over the 5 % gate
  (9.8 % → 2.7 % → 7.65 %). So item 5 is closed **negatively** for τ₀ ≥ 1: no ratio is
  quotable there and χ_op is not the missing knob. **The a = ½ χ_op 32 pair landed on
  2026-09-13 and is not usable either**: control 1.18650 (18.65 % off 1), reference
  inconsistency 175.3 % of |E_min| at λ = 1 and 20.01 % at λ = 0, exact/E_H 2.03834 →
  1.71794 normalized. No a = ½ ratio is quoted; the χ_op 48 pair is still queued.
- **Next decisive cheap point — run (S8):** λ = 0.1 at the sweep's settings reads
  0.741 % against the 0.809 % first-order prediction (0.35 bands below; `s8_lam0.1.json`):
  the floor reading of the small-λ deficit is excluded at 3.3 bands and the first-order
  slope survives; the λ = 0.25 → 1 deficit is a higher-order downturn. What is decisive
  next is stated under "QEI magnitude at λ ≥ 2" above.
