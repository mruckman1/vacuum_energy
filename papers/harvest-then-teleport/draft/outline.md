# Draft outline — Harvest, then extract with a message: the composite exchange rate between vacuum correlation and communication-assisted energy access

**Status: SKELETON.** No number below has been moved out of `../data/` yet.
Per `papers/README.md`, nothing enters a draft until the full known-results
suite is green on the exact producing build. Placeholders are written
`[[data:<json path>]]` and refer to keys in `../data/summary.json`; fill them
by reading that file, never by retyping a number from a log.

---

## 0. What the surplus is, and what it is not (first paragraph of the paper)

The quantity this paper maps is **not teleported energy**. It is the
**classical-communication-assisted extraction from an active state**: a
detector pair that has just harvested entanglement from the vacuum is
*locally active* — the switching that harvested the correlations also
excited each detector, and Bob can extract that excitation with no message
from Alice — so the only thing the harvested correlations can honestly be
said to buy is the *surplus* `E_daemonic − E_local` that Bob's extraction
gains from Alice's measurement outcome (the daemonic gain of Francica et
al. 2017), under Bob's own oscillator Hamiltonian. Hotta's quantum energy
teleportation is a different object: extraction from a *locally passive,
globally correlated* reference (a ground state), where Bob's
no-communication extraction is exactly zero. That object appears here only
as the **true-QET reference column** — the teleportation analogue, run on
the joint ground state of a weak Hotta-type `g x_A x_B` coupling of the
same pair, which uses that coupling's ground-state correlations and not the
harvest (`[[data:hotta_reference_base]]`). The title and abstract say
"communication-assisted extraction", never "teleportation", for the mapped
quantity. State in the same paragraph:

- the surplus is paid for by *correlations* (mutual information), not by
  negativity: separable, classically correlated detector pairs have a
  strictly positive surplus — `[[data:n_dead_negativity_positive_surplus]]`
  rows of the `'xx'` map (`[[data:n_dead_negativity_positive_surplus_all_rows]]`
  of all rows) have `E_N = 0` and `surplus > 0`; no bound of the form
  (energy scale) × f(E_N) exists for mixed states; the bound that holds is
  the Landauer/Szilard one, `surplus ≤ k T_B I(A:B)`
  (`[[data:audits.surplus_le_landauer_all]]`);
- the surplus is maximised over two measurement families on Alice and
  reported for both: the pure *Gaussian* (general-dyne) family in closed
  form (Kua–Serafini–Genoni 2025), and the *non-Gaussian* photon-number
  resolving and on/off detections computed from the Fock representation of
  the harvested state with Bob's general (passive-state) ergotropy. Neither
  family is known to be optimal (Bernards et al. 2019 exhibit POVMs beating
  every projective measurement), so the reported best is a **lower bound**
  on the unrestricted daemonic gain; on every row of this map the Gaussian
  family wins (`[[data:fock.rows_where_non_gaussian_wins]]` rows where it
  does not), and photon counting reaches at most
  `[[data:fock.max_pnr_over_gaussian]]` of it (on/off at most
  `[[data:fock.max_onoff_over_gaussian]]`). A squeezed-number-basis PNR
  column rides along as a *diagnostic only*, never as evidence that
  non-Gaussian beats Gaussian: counting photons in an s-squeezed basis
  tends to a homodyne as s → ∞, which is itself a Gaussian measurement.

## 1. Setup

- The digital twin: 1+1 lattice field (`harmonic_chain_K`, Dirichlet),
  two pointlike oscillator detectors (`attach_detectors`), coupling
  `'xx'` (amplitude) or `'xp'` (derivative, `x_d ∂_t φ`, the model of
  Teixidó-Bonfill & Martín-Martínez 2024), cos² switching with Bob's
  profile delayed by `delay` (0 = simultaneous, `delay = separation` =
  full light contact), base point `[[data:build.config.BASE]]`.
- The IR side is a convergence study, not a fixed regulator: the working
  rows use the synthetic mass 0.2 of papers/circuit-qed-noise-thresholds,
  and the IR sequences take `m → 0` at N = 41 (massless is admissible on
  the Dirichlet chain — no zero mode; the periodic massless chain is
  rejected by `vacuum.core`) and then `N = 41 → 121` massless, for both
  couplings (`[[data:ir]]`).
- Everything runs through `vacuum.protocol.run_protocol`; list the standing
  audits and where each fires (ledger `[[data:audits.max_ledger_defect]]`,
  no-signaling `[[data:audits.max_no_signaling_defect]]`, causality
  `[[data:audits.max_causality_outside]]`, passivity on the vacuum and on the
  Hotta reference, precision, the M2.2 gate at 1e-9 with the accepted
  levels `[[data:audits.halvings_range]]`, the Fock cutoff ladder
  `[[data:fock.cutoff_range]]` with worst movement `[[data:fock.worst_movement]]`
  and the Fock-vs-Gaussian `E_local` consistency `[[data:fock.all_local_consistent]]`).
- Model limitations: UV (unit lattice), oscillator not qubit detectors,
  two couplings and a delay axis but one switching shape, two measurement
  families only (the best of them is a lower bound on the daemonic gain),
  the Lüders re-preparation model for Alice's measurement cost, and the
  1+1 Dirichlet box as the only IR completion (its N-dependence is
  reported, not removed).

## 2. Method: the three ledgers

- Stage 1, joules → ebits: `W_in` (ledger work, on the general quadratic
  form for `'xp'`) and `E_N` (nats, bits), hashing lower bound,
  symmetric-state EoF; ebits per unit work.
- Stage 2, ebits → joules at Bob: `E_local`, `E_daemonic`, `surplus`, the
  Gaussian measurement that attains it (heterodyne / homodyne /
  general-dyne), Alice's measurement cost `E_A^meas` (exactly Ω_A at the
  heterodyne); the non-Gaussian column — PNR, on/off, PNR in the best
  squeezed number basis — from the Fock representation (`A = X(I − σ_Q⁻¹)`
  recursion, Quesada et al. 2019) and the passive-state ergotropy
  (Allahverdyan et al. 2004); closed-form anchors on the two-mode squeezed
  vacuum: PNR gain `Ω sinh² r` (= Gaussian), on/off gain `Ω tanh² r`.
- The exchange rate `surplus / W_in`, the best-of-families rate
  `[[data:best_exchange_rate_best_family_all_rows]]`, the all-in rate
  `surplus / (W_in + E_A^meas)`, and joules per ebit.
- The Hotta column: `g`, `W_couple`, `E_A`, `E_B = E_B^corr + E_B^inj`,
  `E_B/E_A`, and the harvested state's own extraction under `H_g`.

## 3. Results (from `../data/summary.json`)

- The `'xx'` map: exchange rate along coupling, duration, separation, gap,
  temperature (`[[data:profiles]]`), with the ebits-per-joule and
  joules-per-ebit stages separated; best exchange rate and where:
  `[[data:best_exchange_rate]]`; best among harvesting points:
  `[[data:best_exchange_rate_harvesting]]`; best ebits per unit work:
  `[[data:best_ebits_per_work]]`; saturation along each axis:
  `[[data:saturation]]`; closest approach to Landauer
  `[[data:best_landauer_ratio]]`.
- Gaussian vs non-Gaussian measurements: the largest PNR/Gaussian ratio on
  the map and the state it occurs on
  (`[[data:fock.argmax_pnr_sq_over_gaussian]]`), the smallest
  (`[[data:fock.min_pnr_over_gaussian]]`), on/off never above PNR
  (`[[data:fock.onoff_le_pnr_all]]`), and the physical reading: on
  near-vacuum weakly correlated states the quadrature correlations are
  first order in the coupling and the number correlations second order,
  so homodyning wins. State the honest uncertainty: two families searched,
  a grid over squeezed number bases, no claim of optimality.
- The `'xp'` coupling: along separation and coupling
  (`[[data:xp.profiles]]`), its base point `[[data:xp.xp_base]]` against
  the `'xx'` base `[[data:base]]` — fewer joules in, fewer joules out, more
  ebits per joule; the delay axis for both couplings
  (`[[data:xp.profiles.delay]]`): the derivative coupling harvests at full
  light contact where `'xx'` has `E_N = 0`
  (`[[data:xp.light_contact_rows]]`), and on *out-of-cone* windows
  neither coupling carries negativity
  (`[[data:xp.spacelike_rows_with_negativity]]`) while both carry a
  positive surplus (`[[data:xp.spacelike_exchange_rate_range]]`) — that
  exchange rate is nonzero for the same reason it is nonzero at
  `E_N = 0` anywhere: it is paid by mutual information.
- Say plainly that these rows are **outside the nominal light cone, not
  strictly spacelike**: on a lattice the cone is fuzzy, and the measured
  commutator leakage at the detector pair reaches
  `[[data:scan.max_commutator_leakage]]` on the closest rows (2 sites
  beyond the cone), falling to `[[data:scan.max_commutator_leakage_far_rows]]`
  on the far rows (recorded per row as `commutator_leakage` in
  `../data/xp_spacelike_scan.json`). Because the result is a null and
  leakage can only *add* detector correlation, the null is **conservative**
  — it could only have been faked into a false positive, and was not.
- The IR limit: the mass sequence at N = 41 (`[[data:ir.mass]]`; massless
  over regulated `[[data:ir.massless_over_regulated_xx]]`,
  `[[data:ir.massless_over_regulated_xp]]`; movement over the last step of
  the sequence `[[data:ir.mass_limit_movement_xx]]`), then the massless
  size sequence (`[[data:ir.size]]`) with the log-N fits
  (`[[data:ir.size_fit_xx]]`, `[[data:ir.size_fit_xp]]`): for `'xx'` both
  `W_in` and the surplus grow like `ln N` and the exchange rate tends to
  the ratio of the two slopes; for `'xp'` (no coupling to the k = 0 mode)
  the numbers are N-independent to the stated relative change. Say
  plainly that four chain sizes are a trend, not a limit theorem.
- What the true-QET reference delivers per injected quantum at the same
  coupling scale: `[[data:hotta_reference_base.ref_ratio]]`.

## 4. What is not claimed

- Nothing about a device; nothing beyond harmonic detectors on a 1+1
  Dirichlet lattice with cos² switching; two measurement families (their
  best is a lower bound on the daemonic gain); the IR statement is a
  convergence trend over `m → 0` and four chain sizes.
- No claim that the surplus is "teleported" energy: it is
  communication-assisted extraction from an active state, said so in the
  title and the first paragraph; the true-QET column is the teleportation
  analogue on a different (ground) state.
- No claim that the derivative coupling harvests outside the light cone on
  this lattice: it does not on any swept row, nor at any point of the
  archived scan over gap, duration, separation, coupling strength and mass
  (`../data/xp_spacelike_scan.json`, generated by `../scan_spacelike.py`);
  its advantage is at light contact. Those rows are outside the *nominal*
  cone with a measured leakage up to
  `[[data:scan.max_commutator_leakage]]`, not strictly spacelike; the null
  is conservative under that leakage and is never propped up by it.
