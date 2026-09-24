# Implementation Plan — Layers 2 and 3
## vacuum.detectors, vacuum.qet, vacuum.inequalities, and the shared infrastructure they need

Scope: everything required to take the repo from its current state (Layers 0–1 verified, 62 tests) to a complete Layer 2 (harvesting engine with the circuit-QED noise-threshold deliverable) and Layer 3 (QET, strong local passivity, the QEI stress-tester with work statistics). Layers 4–6 are out of scope except for a closing section on what to keep clean so they drop in later. Everything below codes against docs/API.md: ħ = 1, V_vac = I/2, ν ≥ 1/2, block ordering R = (x₁…x_N, p₁…p_N), entropies and negativities in nats, every protocol through `vacuum.audits` before a number is reported.

---

## 0. Sequencing overview and gates

Build order, with the reason for each position:

**M-I Shared infrastructure** → **M2.1 Kernels** (Wightman functions, switching, smearing) → **M2.2 Nonperturbative Gaussian stack** (fast win: rides the validated core, exact to machine precision) → **M2.3 Perturbative UDW** (continuum and lattice backends) → **M2.4 Cross-validation gate** (perturbative ↔ nonperturbative agreement; nothing downstream is trusted until this passes) → **M2.5 No-go and communication split** → **M2.6 Imperfections and parameter files** → **M2.7 Noise-threshold sweep** (first papers/ candidate).

In parallel after M-I, since they are independent of Layer 2: **M3.1 Hotta minimal model** → **M3.2 QET chains (ED first, DMRG behind the extra)** → **M3.3 Strong local passivity**, and **M3.4 QEI core** (the pulled-back sampling operator and the 2d sharp-constant anchor) → **M3.5 4d scan + work statistics** (second papers/ candidate seed).

Two hard gates. Gate A (after M2.4): the perturbative and nonperturbative engines agree on identical setups to O(λ²) accuracy, or nothing built on either is admissible. Gate B (standing): every milestone lands with its regression tests in the anchor suite and all audits wired; a milestone without tests is not done.

---

## M-I. Shared infrastructure (small, do first)

**I.1 — Promote `chain_propagator` to core.** It currently lives in `vacuum.audits.causality`, but detectors, inequalities, and later floquet all need it, and physics modules importing from audits inverts the dependency direction. Move the spectral propagator into `vacuum/core/dynamics.py` (exported from `vacuum.core`); `vacuum.audits.causality` re-imports it. Zero behavior change; the causality tests already cover it.

**I.2 — Thermal states in core.** Add `thermal_state_cov(K, T)` beside `ground_state_cov`: with K = U diag(ω²) Uᵀ,

    V_xx = (1/2) U diag(coth(ω/2T)/ω) Uᵀ,   V_pp = (1/2) U diag(ω·coth(ω/2T)) Uᵀ,

T = 0 reproducing `ground_state_cov` exactly (test), and `mean_energy` against the analytic Bose sum (test). Needed by every noise map in Layer 2 and by work statistics in Layer 3.

**I.3 — The experiments loader.** `vacuum/experiments_io.py` (or `vacuum.detectors.load_experiment` delegating to it): reads `experiments/**.json`, enforces the provenance schema from `experiments/README.md` — mandatory `source`, `location`, `retrieved`, `transcribed_by`, per-quantity `units`, `variant_of`/`changes` for variants — and refuses to load on any missing field (load-time `ValueError`, not a warning). Unit conversion to ħ = 1 lattice conventions happens here and only here, with the conversion factors themselves unit-tested on a fabricated fixture file. Ship one real file per platform as the fixture set: `circuit_qed_harvesting/teixido-bonfill_2026_table1.json`, `ibm_qet/ikeda_2023_fig2.json`, `dce_microwave/wilson_2011_table1.json` (the last is consumed by Layer 5 but its loader is identical).

**I.4 — The protocol runner.** A thin harness `vacuum/protocol.py`: takes an initial covariance, a sequence of (generator, duration) or (channel, args) steps, an `EnergyLedger`, and runs the standing audits at entry and exit (passivity on any state claimed to be a ground state, precision on every reported reduced state, causality once per (K, t_max) pair, ledger closure mandatory). Every Layer 2/3 pipeline goes through this one code path, so the audit discipline is structural rather than per-author.

Definition of done for M-I: loader round-trips the three fixture files; thermal covariance tests pass; propagator import moved with no test regressions; the runner executes a trivial quench with a closing ledger.

---

## M2. Layer 2 — vacuum.detectors

### M2.1 Kernels: Wightman functions, switching, smearing

Two Wightman backends, because the anchors live in different spacetimes.

**Lattice backend (native).** With K = U diag(ω²) Uᵀ cached per K:

    W_ij(t, t′) = Σ_k U_ik U_jk (1/2ω_k) e^{−iω_k (t−t′)}

implemented as `wightman_lattice(K) -> callable(Δt) -> complex (N,N)` returning the matrix U diag(e^{−iωΔt}/2ω) Uᵀ, with the real part (Hadamard) and imaginary part (∝ Pauli–Jordan commutator, state-independent) separately accessible — the imaginary part is exactly the `(S(t)Ω)` object the causality audit already checks, which is itself a cross-test. Smeared two-point kernels are Fᵀ W(Δt) F′ for spatial profiles F, F′.

**Continuum 3+1 backend (for the PKMM anchor).** Massless field, inertial pointlike or Gaussian-smeared detectors. Implement via the mode-sum with the smearing supplying UV convergence: for Gaussian smearing of width σ the radial k-integral carries e^{−k²σ²}, making all P- and M-integrands absolutely convergent 1D integrals — no iε gymnastics. Pointlike evaluation is the σ→0 limit and is only exposed for the closed-form cross-checks; production code always smears. Cutoff-independence (vary a hard k-cutoff around the Gaussian tail, observables move < 1e−8) is a permanent test, not a one-off check.

**Switching and smearing library.** `switching(kind, T, t0)` with `gaussian` and `cos2` (compact support) profiles — sharp switching is not offered; the delta limit exists only inside the Simidzija closed form (M2.5). `smearing(kind, center, width, N)` returns lattice profiles with Σ F = 1. Both return plain arrays plus an analytic-derivative handle (needed by the QEI module and later by JAX mirroring: keep these pure functions of their arguments, no state).

Numerical hazards logged here: quadrature of oscillatory e^{iΩt} factors — use Gauss–Legendre panels sized to min(1/Ω, switching width)/8 with a doubling convergence test; the time-ordered θ(t−t′) kernel integrates over a triangle, so use a triangular tensor-product rule rather than masking a square grid (masking costs an order of accuracy at the diagonal).

Tests: lattice W reproduces equal-time ⟨x_i x_j⟩ = (K^{−1/2}/2)_ij and ⟨p_i p_j⟩ from the second derivative; Im W matches the causality module's commutator to 1e−12; continuum backend reproduces the known closed-form single-detector response for Gaussian switching in 3+1 (transition probability of a static UDW detector, textbook result) to 1e−8.

### M2.2 Nonperturbative Gaussian stack

The exact engine, first, because it is pure `vacuum.core` composition.

**`attach_detectors(K, sites_or_profiles, gaps) -> K_tot`.** Detectors are extra oscillator modes appended after the N field modes. The coupled quadratic form at coupling λ and smearing F:

    K_tot = [[ K,        λ F ],
             [ λ Fᵀ,     Ω_d² ]]

(one column per detector; multiple detectors append columns). x–x coupling only in v1 — it stays inside H = ½(p·p + xᵀK_tot x) so the whole existing core applies unchanged. A derivative/p-coupling variant (needed to mirror the derivative-coupling result of the 2026 paper) is a v2 flag that generalizes H_mat off block-diagonal; design the function signature now (`coupling='xx' | 'xp'`), implement `'xx'` first.

**`protocol_evolve(V, K_of_t, ts) -> V_out`.** Time-dependent λ(t) and Ω_d(t) make H_mat(t) piecewise: step with the midpoint rule, S_k = `symplectic_from_quadratic(H_mat(t_k + dt/2), dt)`, V ← S_k V S_kᵀ. Each step is exactly symplectic (that's the point of building steps from `symplectic_from_quadratic`), so the only error is Magnus truncation: the convergence gate is dt-halving until every reported observable moves < 1e−9, and the harness records the dt at which each result converged. Channels (loss, thermal, dephasing on the detector modes) interleave between steps via `vacuum.core.channels` — Trotterized open dynamics, with the ledger's `dissipate` entries fed from the channel's energy bookkeeping (compute ⟨H⟩ before and after each channel application; the difference is the dissipation entry, which makes ledger closure automatic rather than estimated).

**`detector_block(V_out, N_field) -> V_AB`** is `reduce` on the last two modes; **`harvested_log_negativity(V_AB)`** is `core.log_negativity` with one addition: when the partial-transpose minimum symplectic eigenvalue sits within 1e−10 of 1/2, recompute the margin via `symplectic_margins_mp` and report E_N from the margin. Near-death negativity is precisely where the physics lives and precisely where float64 lies; this rule is mandatory, and the precision audit's `flagged` field is carried into every stored result row.

**Initial states.** λ(0) = 0, so the initial covariance is `ground_state_cov(blockdiag(K, Ω_d²))` — vacuum ⊗ detector ground states — or `thermal_state_cov` of the field block for finite-temperature runs. The passivity audit runs on the T = 0 initial state every time (it is claimed to be a ground state; make the claim checkable).

Tests: with λ = 0 nothing evolves (V constant to 1e−14); adiabatically slow λ ramp-on/off returns the detectors to their ground states (E_N < 1e−10, ⟨n_d⟩ < 1e−8); sudden strong coupling produces nonzero E_N that the dt-halving gate confirms converged; total ledger closes to 1e−9 with the switching drive charged as `work` (compute the drive work as the discrete sum of ⟨∂H/∂t⟩dt via the λ(t), Ω_d(t) derivatives — test it against ΔE on a closed run).

### M2.3 Perturbative UDW engine

Second-order two-qubit state in the basis {|gg⟩, |ge⟩, |eg⟩, |ee⟩}:

    ρ_AB = [[1 − P_A − P_B, 0, 0, M*],
            [0, P_B, C*, 0],
            [0, C, P_A, 0],
            [M, 0, 0, 0]] + O(λ⁴)

with P_ν the single-detector transition probability (double integral of χ_ν χ_ν e^{−iΩ(t−t′)} against the pulled-back Wightman kernel at detector ν), C the cross term, and M the time-ordered pair term integrated over the triangle t′ < t. Transcribe the kernel phases and signs directly from Pozas-Kerstjens–Martín-Martínez, PRD 92, 064042 (2015), Eqs. (2.5)–(2.11), into one file, `vacuum/detectors/udw.py`, with the paper equation number cited next to each expression in comments — the anchor plots are the arbiter of convention slips, so make slips findable. Pair negativity:

    N = max(0, √(|M|² + (P_A − P_B)²/4) − (P_A + P_B)/2),   E_N = ln(1 + 2N)  (nats).

Both backends plug in: `udw_pair_state(kernel, det_A, det_B, lam, gap)` takes a kernel object from M2.1, so the same code runs continuum-3+1 (anchor) and lattice-1+1 (the circuit-QED digital twin's native spacetime — a microwave waveguide is a 1+1 field, which is why the lattice backend is the physically right one for the deliverable).

Anchors: reproduce the PKMM harvested-negativity maps versus separation, gap, and smearing — spot values to plot-digitization tolerance (2%), qualitative structure exactly (death lines, gap dependence); single-detector P_ν against the closed form to 1e−8.

### M2.4 Cross-validation gate (Gate A)

Same physical setup both ways: lattice field, oscillator detectors in the nonperturbative stack versus the perturbative lattice-backend UDW with harmonic-detector matrix elements (for weak coupling the two-level and oscillator detectors agree at leading order in λ² for gap-resonant response; document the mapping in the test file). Requirement: E_N agreement with relative deviation bounded by c·λ² over a λ-sweep (fit the deviation's λ-scaling and assert the exponent ≥ 2 with R² > 0.999). This is the load-bearing test of the entire layer: it validates the perturbative transcription against machinery that cannot have convention errors, and the exact engine against an independent derivation. It becomes a permanent, parameterized regression test, and no M2.5+ code merges before it passes.

### M2.5 The no-go and the communication split

**Simidzija–Martín-Martínez null.** Implement the delta-coupled closed form from Simidzija & Martín-Martínez, **PRD 96, 065008 (2017), arXiv:1707.00016**, Theorems 1–3, analytically (it is exact, not quadrature) and assert N = 0 to 1e−14 for vacuum and for the paper's stated state class, across random parameter draws. This is the suite's designated exact null — the test that proves the engine can output zero.

> **Citation correction (integration, 2026-09-01; ratified by the spec owner 2026-09-01).** This line originally cited "PRD 98, 085007 (2018), arXiv:1809.05547". That paper (*Harvesting correlations from thermal and squeezed coherent states*) was fetched and checked: it contains no delta coupling and no such closed form — it is second-order perturbation theory throughout. The delta-coupled nonperturbative closed form and the N = 0 theorem are in PRD 96, 065008 (2017), which is what `vacuum/detectors/nogo.py` transcribes; the general no-go is PRD 97, 125002 (2018), arXiv:1803.11214. A second correction rides along: the state class the 2017 theorems actually prove the null for is arbitrary **coherent** states (displaced vacua), a *subclass* of Gaussian states — not the thermal/squeezed class of PRD 98, 085007. The tests sweep coherent states built from real displaced-vacuum classical solutions of the chain, and **no claim is made** anywhere in the code about delta-coupled harvesting from thermal or squeezed states.

**`communication_split(kernel, det_A, det_B, …) -> (M_vac, M_comm)`.** Rerun the M-integral twice: once with the full W in the time-ordered kernel, once with only its imaginary (commutator/retarded) part; M_comm is the second, M_vac = M − M_comm. Per Tjoa–Martín-Martínez this separates field-mediated signaling from genuine vacuum correlation. Report both alongside every harvested E_N from here on — the results table schema gains permanent columns (E_N, M_vac, M_comm, |M_comm|/|M|) — and reproduce the 2026 derivative-coupling claim qualitatively once the `'xp'` coupling lands: for causally connected detectors, the correlation-part dominance survives.

Test: for strictly spacelike-separated switchings (compact `cos2` supports, separation outside the light cone), |M_comm| < 1e−12 — and this doubles as a physics-level causality audit on the whole detector stack.

### M2.6 Imperfections and parameter files

`with_imperfections(V, eta, nbar, dephasing)` composes `core.channels` on the detector modes; field-side temperature enters through `thermal_state_cov`. Dephasing on a Gaussian mode = `thermal_noise` on p only or phase-covariant loss — pick the standard circuit-QED Lindblad mapping, document it, and test CP via the Holevo–Werner condition already enforced in channels. Transcribe the Waterloo parameter sets (coupler strengths, gap ranges, switching bandwidths, waveguide temperatures) into `experiments/circuit_qed_harvesting/` under the provenance rules; the loader converts to lattice units in one tested place.

### M2.7 The noise-threshold sweep (papers candidate #1)

Open `papers/circuit-qed-noise-thresholds/`. The sweep: harvested E_N (with mp-margin handling) over (temperature, coupling, switching profile, separation, gap schedule), on the Teixidó-Bonfill parameter files plus declared variants, every point through the protocol runner, every row carrying its audit flags and its M_vac/M_comm split. Deliverable: the go/no-go boundary — the temperature and loss thresholds below which E_N is resolvable at stated measurement precision — plus the best parameter pocket per unit switching energy (the ledger's work entries make "per unit switching energy" a computed column, not an estimate). Admissibility per `papers/README.md`: full suite green on the exact producing build, MANIFEST recorded.

---

## M3. Layer 3 — vacuum.qet and vacuum.inequalities

### M3.1 Hotta minimal model (exact, dim-4)

`vacuum/qet/hotta.py`, dense 4×4 linear algebra, no Gaussian machinery. Implement the minimal model exactly as specified in Hotta's review (arXiv:1101.3954) and Ikeda, PRApplied 20, 024051 (2023): the two-qubit Hamiltonian with its constant offsets fixing E_ground = 0, Alice's projective σ_x measurement (POVM projectors P_A(μ) = (1 + μσ_x^A)/2), the classical bit μ, Bob's conditional rotation U_B(μ, θ), and the ledger {E_A, E_B(θ)}. Transcribe the closed-form E_A and the θ-optimal E_B from the paper with equation numbers in comments (same discipline as M2.3), then:

- **Anchor:** the coded protocol's numerically-swept optimum matches the transcribed closed forms to 1e−12 across an (h, k) grid.
- **Standing audit:** `hotta_sweep` asserts E_A ≥ E_B everywhere (tolerance −1e−12); a violation is an anomaly-protocol event by definition.
- **Causality anchor:** Bob's reduced state before receiving μ is independent of whether Alice measured (trace distance < 1e−14) — no-signaling, coded as a test.
- **Entanglement-as-fuel anchor:** ground-state entanglement (concurrence) before versus after Alice's measurement; the protocol consumes it, and cutting the classical channel zeroes ⟨E_B⟩.

### M3.2 QET on chains

`ising_ground_state(L, g, method)`: exact diagonalization first (L ≤ 14, scipy sparse eigsh on the 2^L space), transverse-field Ising H = −J Σσ_x σ_x − g Σσ_z at criticality g = J. Local energy density h_i with the bond terms split symmetrically; report ⟨h_i⟩ − ⟨h_i⟩_GS always (difference form — the dip is tiny against the extensive ground energy, and the difference form is what survives roundoff). Protocol: projective measurement at site A (Kraus update + renormalize), classical outcome, conditional single-site rotation at B with angle optimized numerically per outcome. Deliverables from ED alone: the energy-dip profile in space and time, its decay with |A − B| at criticality versus off-critical, and the exchange-rate curve E_B^max/E_A versus distance — the first systematic numbers for the Layer 3 discovery target.

DMRG behind `.[tensor]` (TeNPy): same protocol as MPS operations — project, canonicalize, track truncation error, and gate every claim on bond-dimension convergence (χ-doubling moves the dip < 1%). ED validates DMRG at L = 14 exactly before any L = 100 number is admissible.

### M3.3 Strong local passivity

Two rungs. Analytic rung: the Frey–Funo–Hotta single-qubit-region criterion, implemented directly for 2–6 qubit systems, charted over (state family, coupling, region) — `slp_map`. SDP rung (behind a new `.[sdp]` extra, cvxpy): the general certificate, min over CP maps on region R via the Choi program (C ⪰ 0, partial trace = I), which both verifies the analytic rung where they overlap (test) and extends to multi-site regions. The LOCC variant — where classical communication reopens extraction — is the Hotta protocol rerun on the same states, so the boundary chart's two layers (local-forbidden, LOCC-allowed) come from code already written. Keep the SDP dimensions honest: 2–4 site regions of ≤ 8-qubit systems; that is where the phase maps live.

### M3.4 QEI stress-tester core — the pulled-back sampling operator

The module's central design decision, stated once: **the entire smeared-energy minimization collapses to a Williamson problem.**

Construction. Local bare energy density h_i for the chain, chosen PSD by construction (½p_i² plus half of each adjacent bond's PSD quadratic form plus ½m²x_i²). Free Heisenberg evolution pulls it back to t = 0: h_i(t) = S(t)ᵀ h_i S(t) with S(t) = `chain_propagator(K, t)`. For sampling function f, build the **sampling operator**

    O_f = Σ_a w_a f(t_a)² · S(t_a)ᵀ h_i S(t_a)    (quadrature weights w_a),

a (2N × 2N) PSD quadratic form. The smeared normal-ordered energy in state V is ½Tr(O_f (V − V_vac)), linear in V, so its infimum over all physical states (V + iΩ/2 ⪰ 0) is attained at the "vacuum of O_f":

    E_min(f) = ½ Σ_k σ_k(O_f) − ½ Tr(O_f V_vac),

with σ_k the symplectic eigenvalues of O_f — one `williamson` call. The extremal state is exhibited, not just bounded: it is the Gaussian state whose covariance is the O_f-vacuum, pushed forward along the worldline, which is exactly the "optimizing states exhibited" promise of the README. `qei_ratio` = E_min(f) / bound(f); `optimize_sampling` ascends the ratio over a smooth f-family (Gaussian, compact bumps, few-parameter splines) using the analytic-derivative handles from M2.1's switching library.

Anchors, in order: (i) every engineered squeezed-pocket state satisfies the Ford–Roman bound — a violation is an audit event, and the test suite includes a deliberately *unphysical* V (ν < 1/2) asserting the machinery would catch it; (ii) the 2d massless sharp constant: fine chain, small m, interior site, sampling widths between lattice spacing and light-crossing time, ratio → the known optimal 2d constant under Richardson extrapolation in (a/τ₀, m·τ₀) — target within 1% extrapolated; (iii) the Jarzynski/Crooks audits from M3.5 hold on every driven protocol the module touches.

Numerical hazards: O_f assembly is O(N²·n_t) memory-light if accumulated, never stored per-t; near-null symplectic directions of O_f (long-wavelength modes barely sampled) make σ_k small and the Williamson conditioning poor — regularize by restricting to the sampled light-cone's mode support and verify support-independence; and the (i)-test's clamps must be disabled inside the stress-tester (`entropy_from_nu`'s floor-clamp is correct for entropies and wrong for detecting unphysical inputs — use `symplectic_margins_mp` raw).

**The 4d scan (leap L1 seed).** Same construction on the Srednicki radial lattice: T_00 near the origin assembled from the low-l partial waves (l-sum convergence test as in `sphere_entropy`), spatially smeared over a small ball (mandatory in 4d for continuum sanity; verify smearing-radius independence in the extrapolation), scan f-families, Richardson-extrapolate, and record the conjectured constant with its extremal state. This is a milestone with an explicit risk note: if the continuum extrapolation is unstable, the pivot in PLAN.md fires (redirect to interacting-field data) and the instability study itself is written up in the candidate directory.

### M3.5 Work statistics

Two-point-measurement work distributions for Gaussian quenches and drives. Primary route: the closed-form characteristic function for quadratic Hamiltonians and Gaussian initial states (transcribe from the quadratic-work-statistics literature, equation-cited in comments). Validation route: dense Fock-space TPM for N ≤ 3 modes, occupation cutoff with tail test — the two routes agree to 1e−8 before the Gaussian route is used at any N. Audits shipped: Jarzynski |⟨e^{−βW}⟩ − e^{−βΔF}| < 1e−10 (dense) / 1e−6 (Gaussian, N large), Crooks symmetry on forward/reverse pairs. Wire `counting_statistics` output into the `EnergyLedger` as the beyond-mean refinement the ledger docstring promises; from here on, Layer 2 protocol rows can carry work *distributions*, which is what makes "negativity per unit switching energy" upgradeable to "per unit work at stated quantile" in the noise-threshold candidate.

**Papers candidate #2:** `papers/qei-2d-sharp-constant/` — the 2d recovery as the methods paper establishing the pulled-back-operator technique, with the 4d conjecture as its successor candidate once M3.4's scan stabilizes.

---

## Test matrix (new permanent anchors, with tolerances)

| Test | Requirement |
|---|---|
| Lattice W equal-time moments | match K^{∓1/2}/2 to 1e−12 |
| Im W vs causality commutator | 1e−12 |
| Continuum single-detector response | closed form to 1e−8; cutoff-independence 1e−8 |
| λ = 0 protocol inertness | 1e−14 |
| Adiabatic return | E_N < 1e−10, ⟨n_d⟩ < 1e−8 |
| Ledger closure, all protocols | 1e−9 (default) |
| Gate A cross-validation | deviation exponent ≥ 2 in λ, R² > 0.999 |
| PKMM maps | spot values 2%, structure exact |
| Simidzija null | N < 1e−14 |
| Spacelike M_comm | < 1e−12 |
| Hotta closed forms | 1e−12; E_A ≥ E_B everywhere |
| No-signaling (Hotta, chains) | trace distance < 1e−14 |
| ED↔DMRG at L = 14 | dip agreement < 1e−8; χ-doubling < 1% |
| FFH ↔ SDP overlap | verdict agreement, margin 1e−9 |
| QEI: FR bound on all states | violation = audit event; unphysical-V canary caught |
| 2d sharp constant | ratio → known constant, 1% extrapolated |
| Work statistics dense↔Gaussian | 1e−8; Jarzynski 1e−10 dense |

## Dependencies and extras policy

Core stays numpy/scipy/mpmath. New extras: `.[tensor]` (TeNPy — M3.2 DMRG), `.[sdp]` (cvxpy — M3.3 general rung), `.[ibm]` (qiskit — the Ikeda hardware excursion, simulation-first). Nothing in Layers 2–3 imports an extra outside its guarded module, and every extra-gated feature has a core-only fallback path (ED for DMRG, FFH for SDP, noiseless simulation for hardware) so the default install runs the full anchor suite.

## Pre-positioning for Layers 4–6 (write nothing extra, forbid three things)

Keep every new module pure-functional over arrays (no hidden state, no in-place mutation of inputs) — that is the entire cost of making the Layer 6 JAX mirror a transcription rather than a rewrite. Keep kernels, switching, and smearing as data-plus-pure-functions so `vacuum.opt` can differentiate through them and `vacuum.floquet` can reuse the stepper unchanged (a Floquet drive is `protocol_evolve` with periodic K(t) — it is already built after M2.2). And keep every result row carrying (audit flags, dt/χ/cutoff at convergence, parameter-file provenance) — Layer 4's failure maps and Layer 6's optimizer baselines are only as good as the metadata the Layer 2/3 sweeps record now.

## Definition of done, whole plan

`pytest` green including every row of the matrix above; both papers/ candidate directories populated with re-runnable notebooks and MANIFESTs; the noise-threshold map produced end-to-end through the protocol runner with no bypass path; and the README's layer-status table updated with measured values for Layers 2 and 3 in the same style as Layer 1 — each anchor's number visible in its assertion message.
