# Discoveries

What this program found that was not, to our knowledge, known before. Infrastructure,
reproductions and methods-only results are excluded; `docs/PRIORITY.md` classifies every
candidate with the citation that supersedes it where one exists. Every number below is
pinned by a regression test and lives in a `papers/` candidate with its data. Status
labels follow `papers/README.md`: a *model* claim is admissible on the green suite; a
*device* claim needs the calibration snapshot as provenance; anything against a published
number is a *finding*, never a correction, until an expert from that lineage has read it —
and nothing is new until the compared paper's own authors' later work has been searched.

Ranked by how much weight each result carries.

**Correction, 2026-09-23.** Until today the first item here was "the 4d quantum energy
inequality permits 2.53× more negative energy than the literature estimate", described as the
only result in the program that moved a wall outward. It is not a discovery. Schiappacasse, Fewster & Ford, PRD 97, 025013 (2018)
— two of the three authors of the 2012 estimate — diagonalized the same operator for the same
sampling function and found 0.0593338; this program's 0.059762 reproduces theirs to 0.72 %. The
program's literature search never looked at the 2012 authors' own follow-ups. Item 1 below is
rewritten around what the 2018 paper does not contain. Nothing that remains carries the weight
the withdrawn item was thought to; the order is kept, but it now ranks modest results.

---

## 1. The 4d quantum-energy-inequality constant for Gaussian sampling, with a proof of its lower end

*`papers/qei-2d-sharp-constant/`. New for its family on a literature check re-done on 2026-09-23 to the compared authors' own follow-ups, with the one source known to hold 4d sharp computations unread; see the last paragraph. The method is prior art.*

The Gaussian-sampled 4d constant is **C₄d/C_FE = 0.4574904 ± 2e−6** (= 1/(34.97 π²)) from a
lattice-free continuum solve, with the lattice route agreeing at 0.005 % and the ρ² correction
derived by Hellmann–Feynman rather than fitted. **Gaussians are not optimal in 4d**: across six
sampling families the ratio spans 0.283 (Lorentzian weight) to 0.611 ((1−u²)²), a factor 2.16,
so the true sharp constant is only bounded below, at ≥ 0.611 C_FE.

**Half of it is now a theorem (2026-09-12), and only half.** The numerical continuum limit
above has a certified **lower** endpoint: `C_4d(Gaussian)/C_FE ≥ 0.457490709365`, proved,
not fitted. The route is an explicitly exhibited 64-mode Gaussian state of the radial
sectors whose covariance is *verified* to be a state by an interval Cholesky of
[[N, M], [M, I+N]], evaluated in `mpmath.iv` with a proved Γ-series tail bound — and with
**no quadrature in ω anywhere**, so there is no discretisation remainder to estimate and
the entire 2.46e−08 gap to the archived value is Galerkin deficiency on a monotone ladder.
The analytic formulation behind it is derived from the mode expansion and matched line by
line to the code that produced the archived number, with no discrepancy. **What is not
proved is the other endpoint.** The only rigorous upper bound remains Fewster–Eveson's
`R ≤ 1`, cited not computed, so the certified enclosure is **[0.4574907, 1]**, of width
0.542 — one-sided in substance. Quoting it as a narrow certified enclosure would
misrepresent it. The obstruction is structural, not effort: ‖K^{−1/2}K_−K^{−1/2}‖ = 1 and
Tr(K_−K^{−1}K_−) = ∞, so every perturbative lower bound on the underlying energy diverges,
and the exact dual certificate needs certified operator square roots on the half-line.

**The Lorentzian point of that family is a reproduction, and one small thing about it may be
new.** For Fewster–Ford–Roman's Lorentzian weight the same instrument gives x₀ = 0.059762 (three
routes agree: lattice-free continuum 0.059762 ± 5e−7, Srednicki lattice 0.059684 with its own box
law, exhibited state 0.059762), 2.53× their 2012 moment estimate y∞ = 0.02361. That reproduces
Schiappacasse, Fewster & Ford, PRD 97, 025013 (2018), who diagonalized the same operator in a finite cavity and
found 0.0593338 (ours is 0.72 % deeper, in infinite space). Their paper sets the two values side
by side as "of the order of" each other and does not explain the gap. This program does: it
reproduces the 2012 number by the 2012 method on its own kernel (y∞ = 0.0236173 against
0.0236175; all 44 printed Table I entries; Table II to 4.9e−12), and shows that method cannot see
the bottom of the spectrum — a point mass at −0.0598 carrying 10 % of the weight moves the
65-moment edge only to 0.0252, because the moments grow as (3n)! and their tail dominates the
reconstruction. That explanation is a methods point.

**How new the Gaussian constant is, stated carefully after the Lorentzian miss.** Re-checked
on 2026-09-23 against the compared authors' own follow-ups: the whole Fewster–Ford–Roman /
Schiappacasse–Fewster–Ford series, the 252 papers in INSPIRE citing it or the two foundational
QEI references (19 searched in full text), and full-text searches. No Gaussian-sampled 4d sharp
value appears, and a 2024 paper co-authored by Kontou states that none of the four-dimensional
QEI bounds has been proven sharp. The one source known to hold 4d sharp computations could not be read: S. P. Dawson's 2006 York PhD
thesis, which computed 4d sharp bounds numerically and is not online; the 2018 paper describes
its sampling function as a squared Lorentzian. The same check found that the diagonalization
route is not this program's: Dawson's thesis and the 2018 paper both compute the sharp bound as
the lowest eigenvalue of the diagonalized smeared operator. What is this program's is the value
for the Gaussian weight, the lattice-free continuum solve behind it, and the certified lower
endpoint.

---

## 2. First exact quantum-energy-inequality infimum of an interacting lattice field

*`papers/interacting-vacua-first-numbers/`. Model claim; resolved at λ ≤ 1, unresolved above.*

No infimum over all states of an interacting lattice theory was found in the literature,
re-checked on 2026-09-23 against its compared authors' own later papers and full-text searches
for lattice and tensor-network computations. The closest work, Mandrysch, PRD 109, 085022
(2024), computes optimal bounds numerically in continuum integrable models restricted to one-
and two-particle states.
Building the time-and-space-smeared energy of the λφ⁴ chain as a matrix-product operator
by operator-space TEBD and taking its lowest eigenvalue by DMRG gives one, validated
against a dense second implementation to 1.9e−4 at L = 6. What it says:

- **The free bound survives** at every coupling (exact/free ≤ 1, falling to 0.29 at λ = 4).
- **Interactions renormalize the reference by a mass shift, and nothing else.** The
  measured interacting dispersion at λ = 4 is a sharp quasiparticle at every momentum
  (≥ 99.86 % of the spectral weight), nearest-neighbour to 7e−5, with the true gap 3.3 %
  *below* the Hartree value. "The reference is wrong, not just the mass" was tested and
  rejected.
- **Beyond any quadratic reference, the residual is a first-order quartic tax.** An exact
  Hellmann–Feynman/Wick theory gives dR/dλ = +0.081, independent of system size, and the
  quartic term measured on the extremal state is positive (+10 % of |E_min|): the extremal
  state anti-squeezes the field, so the interaction costs rather than helps. At λ ≤ 1 the
  residual is resolved — 1.4 / 1.5 / 2.6 % against a 0.19 % band. That band is the route's
  own **BLAS-thread scatter**, measured at that spec, not a convergence estimate; elsewhere
  in the archive the same-spec spread reaches 5.12 %, so it must be measured wherever a
  number is quoted. At λ ≥ 2 its magnitude is
  **bracketed, not resolved** — [−1.797 %, +4.415 %] against `E_H` at λ = 2 and
  [−34.349 %, +15.131 %] at λ = 4 ([+0.301 %, +6.385 %] and [−28.541 %, +18.800 %] against
  K_eff), a systematic of 32 and 239 λ = 0 bands carried by the *occupation weighting* of
  the operator MPO. Those brackets are the **frozen 15-row snapshot** — 15 of the 36
  anchored plane rows, named by tag as `unit2_snapshot` in the candidate's record so the
  quote stays reproducible while the pool lands more. What survives there is the **sign**,
  not the size: against K_eff the residual is positive at **5 of 5** settings of that
  snapshot at λ = 2 and at **4 of 5** at λ = 4 — **on the live tree as committed on
  2026-09-13, where two further n_max 10 rows had landed, 5 of 6 and 4 of 6** (the live
  verdict sits beside the snapshot in the same record and moves as the pool lands rows);
  the extra λ = 2 negative fails every gate
  the harvest applies (its own λ = 0 control 15.5 % off 1, reference inconsistency 370 %,
  ED anchor 18.2 %). "Positive at every landed setting" was written here without that scope
  and is withdrawn as such (candidate no-longer-claims 11b). **The complete record
  (2026-09-13).** All 36 rows landed, and they widen the bracket rather than narrowing it:
  6.7-fold to [−34.382 %, +6.979 %] at λ = 2 and 2.9-fold to [−127.079 %, +15.861 %] at λ = 4
  against `E_H`, with 12 of 12 settings failing the consistency gate at λ = 2. The sign weakens
  but survives as a majority at λ = 2: against K_eff the residual is positive at 9 of 12
  settings there and 7 of 12 at λ = 4. The
  infimum is also not monotone in the Fock cutoff on any route, because the
  truncated chain is a different dynamical system, not the same operator on nested
  subspaces. **And on 2026-09-12 that got worse, not better.** An exact-diagonalisation
  anchor at the *sweep's own* Fock cutoff puts the tensor-network route **39 % from the
  exact answer at λ = 4** (42 % at the convergence plane's knobs), against 0.3–1.4 % at
  λ = 0 — and the error **grows** with the cutoff instead of being bounded. An earlier
  statement that the truncation was "a 3–12 % effect wherever an exact answer exists" is
  retracted. The method is cross-validated at λ = 0 and **is not cross-validated at λ = 4
  at the cutoff this candidate uses**; no λ ≥ 2 magnitude is quoted anywhere as a value —
  only as the bracket above.
- **Scaling.** The interaction shifts the constant, not the exponent: the residual's power
  in 1/τ₀ is −1.2 ± 0.8 against a predicted +2.

An earlier statement from this candidate — that mass renormalization accounts for 86–95 %
of the reduction — was retracted as unconverged. The version above is smaller than hoped
and, unlike its predecessors, correct.

---

## 3. A superconducting-circuit harvesting experiment cannot resolve entanglement at its own dephasing rate

*`papers/circuit-qed-noise-thresholds/`, `papers/mi-coherence-harvesting/`. A prediction for one experiment, not new physics; the program's only device-level statement, and it is negative.*

For the Waterloo flux-qubit proposal (Teixidó-Bonfill et al. 2026), with the coupling
derived from the circuit Hamiltonian (and re-derived blind, in agreement) and the
proposal's own derivative coupling: at the device's measured dephasing the harvested
negativity is **exactly zero** at every temperature including T = 0, at every resolution
floor, with a ≥ 100× margin in Γ_φ against a ~2× digitization uncertainty. Temperature is
not the binding constraint. The statement survives the coupling's ±1σ, the infrared mass
over a factor 8, gap tracking, and a lattice-cutoff bracket run through the device's own
50 GHz. The plan's fallback — harvesting mutual information, discord or coherence
instead — also fails: every measure sits at 0.13–0.38× its own tomography floor at the
device point, and the measurable region opens only at 2.1–2.5× the derived coupling.

Two findings on the way there: dephasing kills harvesting before it touches its origin
(on every surviving point the vacuum-correlation fraction stays in [0.9994, 0.9996]), and
the derivative coupling's advantage is at light contact, not on spacelike windows, where
neither coupling harvests anything.

---

## 4. Energy teleportation's information signal survives on a real quantum processor; teleportation proper does not

*`papers/hardware-as-noisy-field/`. Device claim, narrow by construction: one device, one day.*

On ibm_fez (2026-09-04, physical qubits 136/143/142, 9600 shots per circuit), the
classical bit in Hotta's protocol is worth **E_B_info = +0.0485 ± 0.0078** of Bob's
local energy (z = 6.23). The prediction, made on a different chip's calibration before
the run, was +0.0401 and held at pull +1.08; the unmitigated value (+0.047) shows the
signal is not a mitigation artifact; all four standing audits pass on the real counts.
Teleportation proper is lost (E_B_ctrl = −0.014 ± 0.008): Bob's extra gates heat more
than the 1.2 % he can extract, exactly as the survival map said they would.

It was not the first hardware run. Ikeda (Phys. Rev. Applied 20, 024051 (2023)) ran energy
teleportation on six IBM devices, and Xie, Sajjan & Kais and Hassan, Quaderi & Mahdy did so in
2024. Ikeda observed the interaction term negative everywhere and noted that Bob's full gain
shrinks once the positive local term is counted; this run separates the two and finds
teleportation proper lost on one device on one day. It is a measurement in their lineage.

---

## 5. Small theorems and results

Each modest and pinned by a test with teeth. A re-check on 2026-09-23 (`docs/LITERATURE_CHECK_2026-09-23.md`)
found several of them already known; each bullet now says which.

**Energy teleportation and passivity** (`papers/resource-theory-vacuum-manipulation/`):
- The one-way extractable energy W→ is a *linear* functional of the state, so **two-way
  classical communication gives Bob nothing** for commuting-Kraus instruments — and the
  boundary is exhibited: off convex position a single free instrument raises the closed
  form 6.15× *one-way*.
- The natural "QET surplus" (LOCC-extractable minus locally extractable) is **not a
  monotone**; W→ and negativity are.
- For a single-qubit receiver the tight extraction bound is exact: **E_B = Tr T + ‖T‖\_\***,
  matching Hotta's protocol to 4.4e−16 over 96 configurations. *Known:* this is the closed
  formula for a qubit's local ergotropy in Salvia, De Palma & Giovannetti, PRA 107, 012405
  (2023), applied branch by branch.
- Energy-non-signalling instruments are exactly the commuting-Kraus ones **iff** the bond
  spectrum is in convex position — no unitality assumption needed.
- At weak transverse field (g = 0.5) the textbook single-site rotation is up to **12.4×
  below** the optimal local operation on Bob's site. *Known phenomenon, new number:* that a local
  channel can beat every local rotation is the gap between local CPTP extraction (Alhambra et
  al., PRL 2019) and local ergotropy (Salvia et al.). The two-way, surplus and convex-position
  results above survived the re-check.
- *Manuscript:* a PRA-format draft is in `papers/resource-theory-vacuum-manipulation/draft/paper/`.
  It is not submitted: three journal-only papers are still to be read, and an expert read is
  still to be arranged. Writing it found three more credits (literature check, part 3).

**Harvest-then-teleport** (`papers/harvest-then-teleport/`):
- **Correlation, not negativity, pays.** Eleven of 27 configurations have zero
  negativity and positive communication-assisted energy surplus; the best exchange rate
  (surplus/work = 0.125) occurs where negativity is already dead. The Landauer form
  surplus ≤ kT_B·I(A:B) holds everywhere (max ratio 0.74); the conjectured
  joint-ergotropy bound is **false** by explicit counterexample. *Known in general:* that
  correlations rather than entanglement pay for measurement-assisted work, and the
  mutual-information bound, are Perarnau-Llobet et al. (PRX 2015) and Manzano, Plastina &
  Zambrini (PRL 2018); what is new is the harvested-state setting. The refuted conjecture was
  this program's own.
- Non-Gaussian conditioning (photon counting, on/off detection) **never** beats the best
  Gaussian measurement on harvested states (0 of 61; at most 0.159 of the Gaussian
  optimum), because quadrature correlations are first order in the coupling and number
  correlations second. Not found in the re-check: the closest work (Kua, Serafini & Genoni,
  2025) treats Gaussian measurements only.

**Time-modulated vacua** (`papers/time-modulated-vacua/`):
- For any measurable modulation bounded by depth d, the per-cycle growth obeys
  ln ρ(S_F) ≤ Z·artanh(d), Z the winding number, with equality iff the drive is
  quarter-period bang-bang — proved in Prüfer variables with no singular arcs. Bang-bang
  optimality itself is 1990s control theory; the per-zero form for measurable drives and
  the closed-form rate optimum were not found in the literature. **The quarter-period
  bang-bang drive is not rate-optimal** (deficit 1.0 / 3.1 / 13.5 % at d = 0.3 / 0.5 / 0.8).
- For **non-commuting** modulation patterns K(t) = K₀ + d s(t) P — where the single-mode
  argument has no purchase — a bound is now proved too, though a weaker one than was
  wanted: `ln ρ(S_F)/T ≤ (d/2)‖K₀^{−1/4} P K₀^{−1/4}‖ ≤ ω_max d_eff/2` for every measurable
  |s| ≤ 1 (Theorem M, 2026-09-12), by a Grönwall estimate on the number norm in the
  canonical complex coordinates; its operator-norm form is near-tight (0.880 of the bound
  is attained) and the same statement with the constant 1/π is violated. A first-order
  result (Theorem F) adds that the frequency governing a coupling is the **geometric mean**
  √(ω₁ω₂) of the two modes, not ω_max, so a low-frequency pair cannot beat the conjecture
  at first order *(Theorem F is classical combination-resonance theory: Svidzinsky et al.,
  arXiv:1407.3727, Eq. (15); Theorem M stays POSSIBLY KNOWN — searched, not found, standard
  technique)*. **The conjecture itself — that the single-mode constant λ*(d_eff) governs
  the multimode case — is still neither proved nor refuted**: 128 polished adversarial
  optima **reach** it — largest archived ratio 1.000000000000017 (refined
  1.000000000000036), i.e. attained and exceeded by +1.7e−14, within the 10⁻¹³ roundoff of
  ln ρ, always at an effectively single-mode configuration (ω → (1, 1)) where Theorem B
  already attains it — and never break it (`conjecture_refuted = false`). It is listed
  here because the *proved* bound is a result; the conjecture is not, and is not counted
  as one.
- Pump-enhanced harvesting is **not null**: modulating the field before and during
  detector coupling raises harvested negativity 10–33× (finite-size converged) and
  revives dead spacelike baselines — but loses to plain vacuum harvesting per unit of
  drive work wherever the vacuum harvests appreciably. Loss threshold for amplification:
  Q_th = π/(2 ln λ_max), placing the demonstrated microwave regime and the optical race
  on opposite sides. No prior study of harvesting from a pumped field was found; the threshold
  itself is the gain-equals-loss condition.

**Geometry from entanglement** (`papers/geometry-from-entanglement-failure-map/`,
`papers/toy-jacobson/`):
- A failure map of metric reconstruction from mutual information on
  non-holographic vacua: off criticality the geometry degrades *with structure* (a
  log-to-linear crossover at correlation length ≈ 13–15 sites, curvature radius rising
  monotonically), while under disorder it degrades *into noise*. *Half known:* Leighton-Trudel
  (arXiv:2507.09749, 2025) proves that a mutual-information distance stops being a metric once
  correlations cluster exponentially; the disorder half was not found.
- With the modular weight free rather than assumed, entanglement equilibrium on the
  massless wave chain **finds** the Casini–Huerta–Myers parabola from nothing (overlap
  ≥ 0.9992 across six bases). Lifshitz dynamics are distinguished by the *complexity* of
  the admissible weight — a 3-mode parabola versus ≥ 11–14 modes of a bumpy profile —
  not by its existence. A parsimony statement; the earlier "recovers the velocity" claim
  was the weight's own normalization and is withdrawn. The parabola itself on the lattice is
  Di Giulio & Tonni (J. Stat. Mech. 2020); the route through the first law is this toy's.
- **In 2+1 the same toy measures a coefficient of the continuum theory instead of checking
  one** (2026-09-12). On an N × N lattice with disks the canonical ansatz fails and does not
  improve with radius (residual 0.36–0.45, flat in R), because in d ≥ 3 the CFT stress
  tensor carries the conformal improvement −ξ∇²(φ²) that vanishes identically in d = 2 —
  which is why the 1+1 toy never saw it. Fitting that one term cell by cell returns
  **ξ = 0.1278 ± 0.0040** against the free scalar's (d−2)/(4(d−1)) = **1/8**, and the
  *shape* is identified as well as the number (3.1 % scatter across 15 cells for the ∇²β
  form against 7.7–32.9 % for three wrong shapes, one of which would need a negative ξ).
  With ξ = 1/8 the residual collapses to 0.005–0.075. The coefficient itself is textbook;
  what is new is that entanglement equilibrium is sharp enough to measure it. Two precedents,
  found 2026-09-23: Javerzat & Tonni (JHEP 2022) confirmed the sphere's lattice entanglement
  Hamiltonian against the conformal prediction in two and three spatial dimensions, and Herzog
  & Nishioka (JHEP 2016) derived the boundary term this toy drops. Two
  qualifications belong in the same breath: the improvement's entangling-circle contact
  term is **dropped, not derived** (keeping it makes the fit ten times worse), and ξ is
  fitted on 15 cells, not derived.
- **The 1+1 lattice residual's rate is derived, and a published exponent of this program's
  own was a grid artefact** (2026-09-12). The site- and bond-centred discretisations of
  ∫βT₀₀ differ, on a squeeze of a lattice normal mode, by an exact closed-form finite sum
  whose continuum limit is A(κ)/L² with A(0⁺) = −3/2; over nine cells the identity
  reproduces the measured residual to 6.3e−12. So the leading part of the site-centred
  floor is the trapezoid rule's weight-placement error rather than a property of the first
  law, and the **exponent is exactly −2** — the fitted −2.27 was the (L, λ, N) grid
  drifting the mode index between cells. What this does **not** establish, and is labelled
  ASSUMPTION in the draft, is that the lattice residual converges to the continuum one at
  all; the derivation bounds the difference between two discretisations, not the distance
  to the continuum.

---

## Not included here

Reproductions at certified precision (the multi-interval Casini–Huerta screening, Eisler's
gapped-chain slope, the amplitude-coupling Gate A, the Layer-1 anchors), methods results
(the tomography-floor observation, the differentiable-protocol gains, the inverse-design
recovery), and the toolkit itself. They are classified, with citations, in
`docs/PRIORITY.md`.

## What the discoveries still need

Revised 2026-09-12, after a pass that worked six open items and closed two of them, and
again 2026-09-23, when item 1's headline turned out to be a reproduction.

**Item 1** no longer needs a reader from the Fewster lineage for the Lorentzian comparison:
Fewster co-published that number in 2018. It needs two things instead. First, the one source the 2026-09-23 literature
check could not read: Dawson's 2006 York thesis, which is not online and would have to be
obtained from the university library. Second, the **upper** half of the certified enclosure — a certified upper
bound on the same quantity, which requires certified enclosures of the operator square roots of
K ± K_− on the half-line (or a dual certificate that avoids them). Until then the Gaussian value
is proved from below and cited from above.

**Item 2's λ ≥ 2 magnitude** is where the description has changed most, and in the
unfavourable direction. It does not merely need "a deeper Fock cutoff with an
exact-diagonalization anchor beyond n_max = 4": at the cutoff the candidate already uses,
the anchor now exists and shows the tensor-network route 39 % from the exact answer at
λ = 4 with the error *growing* in the cutoff. The knobs are measured: the variational bond
is converged, the operator bond dimension is non-monotone in the error and changes sign,
the **weighting** is the dominant cost, and the reference-consistency gate the sweep uses
is uninformative about the λ = 4 error. As of 2026-09-13 the plane has been walked far enough to give a **bracket** and not a
value: a frozen 15 of 36 anchored rows, a systematic of 32 and 239 λ = 0 bands at λ = 2 and 4, and
an ED-anchor gate that clears exactly one setting per λ — two *different* settings, so
there is no spread to measure. So what it still needs is the convergence plane walked
with an **exact anchor at every quoted setting** — affordable only at small L — or an
operator representation that does not spend its bond dimension on high-occupation matrix
elements. The 20-job weight-0.5 exact-SVD `plane` was **retired** on 2026-09-12 with 18 of
its jobs unspent (368 h nominal / 856 h calibrated) and replaced by the 36-job anchored
`plane2`. All 36 of its rows landed on 2026-09-13, and the complete plane **widened** the
bracket instead of narrowing it — 6.7-fold at λ = 2 and 2.9-fold at λ = 4 — so the operator
representation, not more of this plane, is what item 2 needs.

**Item 3** needs a circuit-QED experimentalist to read
`vacuum/detectors/circuit_mapping.py`. Unchanged, and not replaceable by computation.

**Item 4** is a measurement on one device on one day; what it needs is another device, or
another day, not an argument.

**Among the small results of item 5:** the multimode amplification conjecture is still open
— what is proved is a weaker bound with the constant 1/2, and the conjecture itself has
been attacked to 1.000000000000017 — reached, and exceeded only by +1.7e−14 of ln-ρ
roundoff, never broken — which is evidence and not a proof. The 2+1
entanglement-equilibrium measurement of ξ needs its
entangling-surface contact term compared with Herzog & Nishioka's derivation (JHEP 2016) rather than dropped, and a 3+1 run to see
whether the toy tracks (d−2)/(4(d−1)) across dimensions. The 1+1 rate derivation needs the
step it does not take: that the lattice residual converges to the continuum one at all. And
the 1+1 reflection-even sector is only half explained — its large residual is the massless
scalar's IR zero mode, but what is left after projecting that mode out is unexplained and
is worse, per unit variation, than the odd sector by 1.9× to 659×; an earlier claim that it was *better* is retracted in that candidate's MANIFEST.

Everything else is finished and labeled.
