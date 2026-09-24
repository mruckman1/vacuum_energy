# Draft outline — A resource theory of vacuum manipulation, bootstrapped by small-system numerics

**Status: SKELETON.** Definitions and numerical theorems are final in
`vacuum/qet/resource.py` (module docstring) and `tests/test_resource_theory.py`;
every number below is a placeholder `[[data:<file>:<key>]]` pointing into
`../data/`, filled only from a build whose full suite is green (`MANIFEST.md`).

---

## 0. What is new and what is not

- Not new: strong local passivity (Frey–Funo–Hotta 2014; Alhambra et al. 2019),
  Hotta's QET energetics (arXiv:1101.3954), the smooth-entropy calculus
  (Tomamichel, arXiv:1504.00233), the QRT axioms (Chitambar–Gour, arXiv:1806.06107).
- New (to our knowledge): a QRT whose free operations make Hotta's protocol
  *free* and whose monotone is the energy Bob cashes; its closed form on
  commuting bonds; the ledger form of monotonicity (5b); the identification
  of the "QET surplus" as a non-monotone; the one-shot bound and its
  saturation by the minimal model; and the numerical maps below.

## 1. Definitions (state verbatim from the module docstring)

1. Bipartition and Hotta's booking of the interaction on Bob's side, Eq. (1);
   the bond operators X_j (operator-Schmidt), `bond_operators`.
2. Free operations O1–O4. The energetic no-signalling condition (2):
   `Abar^dag (x) id (H_B^loc) = H_B^loc`. **Theorem 2 (new): the free class is
   exactly the commuting-Kraus class iff the joint bond eigenvalue vectors are
   in convex position** — no unitality assumed, superseding the Bloch-ball +
   Arias–Gheondea–Gudder argument, which reached only a single-qubit A. Where
   the hypothesis fails, an explicit *non-unital* free instrument outside the
   commutant (Eq. (8)) and a certified SDP (Eq. (20)) that measures the slack.
   Closure under composition; golden rule (Chitambar–Gour Def. 1).
2'. The **symmetric (two-way) booking**, Eq. (9): the interaction split evenly
   (equivalently booked to a third "bond" register). Same free class on both
   sides — the booking changes only Bob's status, not Eq. (2). Free operations
   for the two-way theory: energy-non-signalling instruments on *either* side,
   two-way classical communication, and a local CPTP map on the *receiving*
   side per round; W_<-> of Eq. (13), finite rounds (round number is itself an
   LOCC resource — Chitambar–Gour Sec. II).
3. Free states: W_-> = 0. The vacuum is a resource.
4. Monotone 1: the QET potential W_->, Eq. (3); Proposition A (closed form (4)).
5. Monotone 2: negativity (Vidal–Werner), LOCC ⊃ O.
6. The surplus Q = W_-> − W_loc and why it is not a monotone.

## 2. Numerical theorems (each is a test)

- **T1 (Hotta is free and spends E_B).** W_->(|g>) = E_B^max (Eq. 11) to
  `[[data:s2_hotta_grid.npz:max_abs_W_minus_EB]]` over a 33×33 (h,k) grid;
  branches after the free measurement carry exactly W_-> as local ergotropy;
  after Bob's reset they are free and the ledger drop equals E_B
  (`test_the_protocol_is_one_free_sequence_and_e_b_is_spent`).
- **T2 (ledger monotonicity).** Over `[[data:s1_monotonicity.json:total_samples]]`
  random free operations on random states (pure / mixed / ground-mixtures /
  thermal; 2–6 qubits; 2–4 Gaussian modes) the worst violation of (5a)/(5b)
  and of negativity monotonicity is `[[data:s1_monotonicity.json:worst_violation_all]]`
  (roundoff), while non-free instruments on A raise W_-> by up to
  `[[data:s1_monotonicity.json:qubit_cases[*].teeth_nonfree_instrument]]` and
  energy injection on B by up to `[[...teeth_bob_injection]]`, never by more
  than the injected energy (`teeth_bob_injection_bound_defect`).
- **T3 (the surplus is not a monotone).** Closed-form LOCC counterexample
  Q: 0 → sqrt(h²+4k²) − h under a local unitary on A
  (`[[data:s4_surplus.json:counterexamples]]`); under Bob's maps Q never
  decreases (`min_change_of_Q_under_bob_maps`), so it is not what Bob spends.
- **T4 (one-shot bound, saturation).** Chain (i)–(v). On the minimal model
  E_B/bound = 1 to `[[data:s2_hotta_grid.npz:max_abs_ratio_minus_1]]`
  (relative; absolute roundoff) — Bob's conditional states are pure, H_min = 0.
- **T5 (one-shot bound, chains).** No violation on L = 4..10, g ∈ {0.5,1,2},
  all ε on the grid (`[[data:s3_chain_rows.npz]]`, `any_bound_violation`).
  Tightness: region-unitary bound `[[s3:max_E_B_over_bound_exact]]` (loose);
  marginal-fixed SDP relaxation `[[s3:max_E_B_over_bound_marginal]]`
  (saturated at |A−B| = 2, 0.98 at |A−B| = 1); whole-complement ceiling
  `[[s3:max_E_B_over_W_full]]`.
- **T5' (the rotation is optimal only at g ≥ J).** At g = 1 and g = 2 the
  single-site rotation saturates the single-site CPTP potential W_site
  (`ratio_site`, rows with E_B > 1e-6); at g = 0.5 the CPTP optimum is up to
  12× larger at |A−B| = 1 (SDP-certified in `MANIFEST.md`): in the ordered
  phase Bob's conditional state is mixed and a non-unitary map (partial
  reset) beats every rotation. `test_rotation_is_not_optimal_in_ordered_phase`.
- **T7 (two-way communication gives Bob nothing — for commuting-Kraus A).**
  W_-> is *linear*, W_->(rho) = Tr[W_op rho] (Theorem 4), and W_op − H_B^loc is
  supported on Alice's side, so Theorem 1's inequalities are **equalities** for
  commuting-Kraus instruments on A and arbitrary instruments on B — by
  Theorem 2 the whole free class exactly under convex position, which covers
  every model here. Every finite-round two-way protocol therefore pays Bob exactly
  W_->(rho): `[[data:s6_two_way.json:max_bob_only_minus_one_way]]` over the
  minimal model and L = 5..8 chains. The random-protocol audit measures the
  *conserved* ledger, not merely monotonicity: drift
  `[[s6:worst_ledger_drift]]` with teeth `[[s6:min_teeth]]`
  (`test_two_way_ledger_is_conserved_with_teeth`).
- **T7' (what two-way does buy).** With **both** devices cashing out the value
  exceeds W_-> and grows with the round count
  (`[[s6:max_two_way_over_one_way]]`), but the surplus is paid by the
  measurement apparatus: the net of the injections is round-independent
  (`[[s6:net_is_round_independent]]`). The symmetric theory's honest statement
  is that two-way LOCC redistributes energy between two devices; it does not
  make the vacuum a richer resource.
- **T7'' (the scope is sharp, and the breach is a result).** Off convex
  position T7 is false, and it fails from *one-way* communication alone: the
  non-unital witness raises Prop. A's closed form 0.036634 → 0.225331 on the
  star model, a factor **6.1509**, at energy-signalling defect 9.2e−16. So
  `qet_potential` computes the *commuting-Kraus* potential, which off convex
  position is neither the true potential nor a monotone — the paper should
  present this as the sharp boundary of Theorem 4, not as a caveat
  (`test_theorem_4_is_scoped_to_commuting_kraus_and_the_witness_breaks_it`).
- **T8 (the free class, sharply).** Every model this candidate uses is in
  convex position, so Prop. A and Theorem 1 stand
  (`[[data:s7_free_class.json:rows]]`); on `star_hamiltonian`, which is not,
  the wider free class yields `[[s7:max_gain_factor]]`× the closed form (4) and
  the closed form is not even monotone there. The SDP certificate agrees with
  the analytic verdict on every row (`[[s7:sdp_agrees_everywhere]]`).
- **T9 (the tight bound is exact, not entropic).** For a single-qubit Bob the
  one-way extractable energy under unitaries is the orthogonal-Procrustes
  closed form (15) — exact to **4.4e-16 in energy units** over 96 rows
  (L = 6..12) against the achieved rotation and against Hotta Eq. (11),
  replacing the marginal-fixed SDP's `[[data:s8_entropic_summary.json:max_ratio_marginal]]`
  with an exact solver-free answer. (Quote the absolute figure, not
  `[[s8:max_ratio_closed_form]]` = 1.0000219: that ratio's deviation is
  roundoff on rows where E_B ~ 1e-11; on rows with E_B > 1e-6 it is 1 to
  1.1e-10.)
- **T9' (the entropic form, and its honest looseness).** Prop. C:
  E_B ≤ 2c(exp(−H_min(X|R)) − 1/2) — Bob extracts only what his region *knows*
  about Alice's outcome, and the bound vanishes exactly when the bit is
  unguessable. Valid on every row, tightness only `[[s8:max_ratio_guess]]`;
  the b-active and row-sum refinements are loose for the same reason (an O(1)
  trace distance in a nearly energy-neutral direction). Also: **E_B is exactly
  the daemonic ergotropy** of Francica et al. (arXiv:1608.00124, Eqs. (3), (5)),
  the unconditional ergotropy vanishing by strong local passivity.
- **T10 (smoothing and size).** Prop. E: over Tomamichel's own sub-normalized
  ball the smooth min-entropy is unchanged until the normalized value saturates
  at ln d, where it adds exactly −ln(1−eps²)
  (`[[s8:subnorm_gain_at_saturation]]`); the sub-normalized chain (iii') never
  tightens a bound (`[[s8:subnormalized_ever_helps]]`). Chains reach **L = 12**
  through the sparse Prop. A and the region-only ledger
  (`[[s8:L_max]]`), retiring the L ≤ 10 wall.
- **T6 (Gaussian mirror).** Closed form = conditional-state integral to
  `[[data:s5_gaussian.npz]]` identity defect; Theorem 1 under Gaussian free
  operations (`gaussian_cases`).

## 3. Charts

- `fig_hotta_W_arrow.svg`: W_-> = E_B = bound over (h,k) (log scale).
- `fig_hotta_W_over_EA.svg`: exchange rate W_->/E_A (max
  `[[data:s2:max_W_over_E_A]]` at `[[s2:argmax_W_over_E_A_hk]]`).
- `fig_chain_L10_g1.svg`: E_B, W_site, region bound, marginal-fixed bound,
  negativity vs distance; hlines W_-> (Bob = complement) and E_A.
- `fig_gaussian_W_vs_mass.svg`.

## 4. Discussion / limitations (write honestly)

- The symmetric two-way theory now exists (Eq. (9), Eq. (13)) and its verdict
  is negative *for Bob*: two-way communication cannot beat one-way, because the
  ledger is an equality (Theorem 4). What two-way buys is a second device being
  filled at the measurement apparatus's expense. The finite-round value is the
  honest object — unbounded-round LOCC is not a closed set.
- Alice's free class is now characterized exactly (Theorem 2), with no
  unitality assumption, by convex position of the bond spectrum; where that
  fails the closed form (4) is a strict under-estimate and not a monotone. All
  models here satisfy it, but the *general* theory must carry the hypothesis.
- Prop. A (closed form) needs commuting bonds and Bob = complement; the
  restricted-Bob potential is exact for rank-one free blocks, otherwise a
  projective lower bound.
- For a single-site Bob the tight statement turned out to be **exact and
  analytic** (Prop. B), not entropic. The entropic form (Prop. C) is proved and
  is the first one that vanishes with the correlation rather than with the
  region's energy scale, but it is ~2 orders of magnitude loose, and we
  identify why (the cancellation inside ||T||_* that no norm-based relaxation
  retains). An entropic quantity that keeps that cancellation is still open.
- Smoothing: Prop. E bounds what the sub-normalized ball could ever buy
  (−ln(1−eps²), and only at saturation), so the earlier "normalized ball"
  caveat is retired rather than merely restated; the ε-dependence still never
  helps on these pure-global-state protocols.
- L = 12 is reached without a dense 2^L, but only for the single-site-Bob
  observables and the whole-complement ceiling; the Bob = complement CPTP
  optimum beyond L ≈ 10 is still out of reach, and nothing here is a
  thermodynamic limit.
