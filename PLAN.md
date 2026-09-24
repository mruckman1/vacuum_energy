# The Vacuum Program
## A code-first research plan for investigating the entanglement structure of the quantum vacuum

**Orientation.** This is a discovery program, not a curriculum. It is organized as six layers plus a leap tier, each with validation anchors (known results that must be reproduced before anything new is trusted), discovery targets (questions with no published answer), and pivot triggers (conditions under which the direction changes). Nothing is timeboxed. The program follows one governing principle established over the whole investigation that led here: the vacuum is closed as an energy source and open as a structured, correlated, manipulable medium. The code treats the closures as invariants and the openings as search space.

**Why code can carry this.** The free-field vacuum is a Gaussian state. On a lattice it is fully specified by a covariance matrix, and every quantity of interest — entanglement entropy, logarithmic negativity, mutual information, energy density, modular Hamiltonians, evolution under quadratic dynamics — is exact linear algebra on that matrix. This makes the subject unusually computable: most of the harvesting literature is itself numerical, and several of the field's open problems (sharp quantum-inequality constants, two-interval modular Hamiltonians, interacting-field energy inequalities, Floquet amplification bounds) are accessible to careful numerics before they are accessible to proof.

---

## Layer 0 — Infrastructure and the standing audits

**Repository structure.**

```
vacuum/
  core/        # covariance matrices, symplectic algebra, Williamson, Gaussian channels
  fermions/    # correlation-matrix methods, entanglement Hamiltonians
  detectors/   # UDW perturbative + Gaussian nonperturbative harvesting engine
  qet/         # quantum energy teleportation models, exact + tensor-network
  inequalities/# quantum energy inequality stress-testing and sharp-constant search
  modular/     # numerical modular Hamiltonians
  geometry/    # MERA, holographic codes, metric reconstruction
  floquet/     # time-modulated vacua, amplification, photonic-time-crystal models
  opt/         # JAX differentiable stack, RL protocol search, inverse design
  audits/      # conservation, passivity, causality, precision checks (run on everything)
  experiments/ # parameter files mirroring published setups (circuit QED, IBM hardware)
  papers/      # one directory per result candidate: notebook, data, draft
```

**The literature is the test suite.** Every reproduced result from the anchor papers becomes a permanent regression test. New physics claims are only admissible from a build in which the entire known-results suite passes. This converts the field's history into continuous integration and is the single most important discipline in the program.

**The standing audits.** Every simulated protocol, in every layer, automatically passes through four checks. The energy ledger: total energy accounting must close; any protocol that appears to net energy without an external drive or classical-information input is flagged as a bug, because passivity of the ground state (Pusz–Woronowicz) forbids it. The passivity audit: local unitaries on the vacuum must never lower energy expectation. The causality audit: field commutators evaluated at spacelike lattice separations must vanish to precision tolerance, and any harvested correlation must be re-run against a communication-only surrogate to separate genuine vacuum correlation from signaling through the field. The precision audit: symplectic eigenvalues near 1/2 are the danger zone; the stack escalates float64 → longdouble → mpmath automatically when Williamson spectra approach the vacuum floor. The audits encode the hard walls established by the physics — thermodynamics, passivity, causality — as executable constraints. If an audit ever fails on validated code, that is not an error report; it is the most interesting event the program can produce, and gets the full anomaly protocol (below).

**Compute posture.** Layers 1–3 run on a laptop. Layers 4–6 want a single GPU for tensor networks and optimization sweeps. Nothing requires a cluster until the interacting-field leap.

---

## Layer 1 — The vacuum as a data structure

**Build.** A 1D chain of N coupled harmonic oscillators discretizing the Klein–Gordon field, Hamiltonian H = ½ Σ [πᵢ² + m²φᵢ² + (φᵢ₊₁ − φᵢ)²], with the ground-state covariance blocks obtained from the square root of the coupling matrix. On top of it: Williamson decomposition, symplectic evolution, Gaussian channels (loss, thermal noise, amplification), partial transpose via momentum sign flip, and the observable set — von Neumann entropy from symplectic eigenvalues, logarithmic negativity, mutual information, local energy density. A parallel fermionic module implements tight-binding chains through correlation-matrix methods, which are needed later for exact modular Hamiltonians. A radial-lattice 3D scalar module reproduces the geometry Srednicki used.

**Validation anchors.** Four, in order of difficulty. The area law for a sphere in the 3D massless field, coefficient ≈ 0.30 in lattice units (Srednicki 1993). The logarithmic entropy of a block in the critical chain, S = (c/3) ln ℓ + const with c = 1, including the known finite-size corrections (Calabrese–Cardy). The decay and sudden death of logarithmic negativity between two disjoint blocks as separation grows — the quantitative statement of how scarce harvestable vacuum entanglement is. The first-law check: for small perturbations of a region, δS = δ⟨K⟩ against the known modular Hamiltonian of an interval.

**Discovery targets.** None yet — this layer is deliberately pure reproduction. Its product is trust.

**Pivot trigger.** None. This layer is unconditional.

---

## Layer 2 — The harvesting engine

**Build, perturbative half.** Unruh–DeWitt detectors to second order: Wightman functions on the lattice and in the continuum, double quadrature against smooth switching functions (sharp switching produces spurious divergences and is banned by default), smeared detector profiles, and the full negativity formula from the local-excitation and nonlocal terms. Reproduce the harvested-negativity maps of Pozas-Kerstjens–Martín-Martínez (2015) across separation, energy gap, and smearing. Verify the Simidzija–Martín-Martínez no-go for delta-coupled detectors as an exact null. Implement the estimator that splits detector correlations into a genuinely vacuum-mediated part and a communication part sourced by the field's retarded propagation — this split is what makes any later claim honest.

**Build, nonperturbative half.** Replace qubit detectors with oscillator detectors so the detectors-plus-field composite is one Gaussian system evolving symplectically — exact to machine precision, no perturbation theory. Load the circuit-QED parameter sets from the Waterloo program (flux qubits with tunable ultrastrong couplers; the 2026 variable-gap detector model), including the regime where the energy gap varies during coupling. Add realistic imperfections as Gaussian channels: thermal occupation of the waveguide, photon loss, dephasing, finite switching bandwidth.

**Deliverable: a digital twin of the harvesting experiment.** The concrete product of this layer is a preregistered set of predictions for the superconducting-circuit demonstration that has not yet been performed anywhere: harvested negativity as a function of temperature, coupling strength, switching profile, and detector separation, with explicit thresholds below which the experiment cannot succeed. A noise-threshold map for an experiment the community is visibly building toward is a publishable contribution from pure code.

**Discovery targets.** The go/no-go noise boundary for circuit-QED harvesting. The parameter pockets maximizing harvested negativity per unit switching energy. Whether the derivative-coupling result — that even causally connected detectors draw their entanglement primarily from field correlations — survives realistic noise, which would relax the brutal requirement of strict spacelike separation.

**Pivot trigger.** If harvested negativity under honest noise models sits orders of magnitude below detectability for all realistic parameters, the layer pivots to mutual-information and coherence harvesting (measurable classical-quantum correlations rather than distillable entanglement) and to the QET emphasis of Layer 3, and the negative result — with clean audits — is written up as the boundary map it is.

---

## Layer 3 — Energy: teleportation, passivity boundaries, and the inequality stress-tester

**Build, QET.** The minimal Hotta model exactly: two coupled qubits, Alice's measurement injecting E_A, the classical bit, Bob's conditional rotation extracting E_B, sweeping the full parameter space and confirming E_A ≥ E_B everywhere as a standing audit rather than a result. Then scale: critical Ising chains via DMRG (TeNPy), measurement-plus-feedback as quantum channels, watching local energy density at Bob's site dip below the ground-state value and mapping how far, how fast, and how it decays. Reproduce the Ikeda 2023 IBM-hardware protocol in simulation, then — as the program's one hardware excursion — on the free tier of actual devices, ending a code-only path on a physical quantum machine.

**Build, passivity boundaries.** Numerically chart strong local passivity (Frey–Funo–Hotta; Alhambra et al. 2019): for which states, couplings, and regions is local energy extraction impossible even with the most general local operations, and exactly where does classical communication reopen it. Nobody has produced systematic numerical maps of this boundary. This is also the seed of the leap-tier resource theory.

**Build, the inequality stress-tester.** Engineer Gaussian states with negative-energy pockets — squeezed, multimode, spatially structured — and drive optimization against the Ford–Roman bound: minimize the ratio of achieved negative averaged energy to the bound, over states and over smooth sampling functions. For Gaussian states this is a tractable quadratic-form problem, and the sharp-constant question reduces to eigenvalue problems for nonlocal operators, exactly the unclaimed mathematics flagged in the program's genesis. The sharp constant is known in two dimensions and open in four; numerically converged conjectures for the 4D constants, with the optimizing states exhibited, would be a direct contribution to the Fewster program. Attach the work-statistics module here: two-point-measurement energy distributions for quenched and driven protocols, Jarzynski and Crooks as further standing audits, full counting statistics so every protocol in the repository carries its complete energy ledger, not just its mean.

**Discovery targets.** Sharp-constant conjectures in cases with no published tight bound. The strong-local-passivity phase maps. The exchange rate between injected measurement energy and remotely available energy in QET as a function of entanglement structure — the beginning of a quantitative answer to what "manipulating energy already there" can achieve.

**Pivot trigger.** If the sharp-constant search stalls in optimization plateaus, redirect that effort to the interacting-field leap (Layer 7, L5), where even crude numbers are new.

---

## Layer 4 — Geometry from entanglement, computationally

**Build.** Three instruments. First, a modular-Hamiltonian laboratory on free fermion chains via Peschel–Eisler methods: exact entanglement Hamiltonians for a single interval, validated against the Bisognano–Wichmann local form at criticality and against the Casini–Huerta exact two-interval results — then pushed into configurations where no closed form exists, tracking how nonlocal couplings in the modular Hamiltonian grow and decay. This is research-grade territory reachable with validated code. Second, a MERA implementation for the critical Ising model: entropies from minimal cuts through the network, and an emergent-geometry probe that defines distances from mutual information (d ∼ −ln I between regions) and tests how closely the reconstructed metric approaches the hyperbolic geometry the Swingle correspondence predicts. Third, the HaPPY holographic code in stabilizer formalism, where bulk reconstruction and the Ryu–Takayanagi formula can be verified by direct computation.

**The honest experiment of this layer.** Run the metric-reconstruction pipeline on systems that are *not* holographic by construction — massive chains, disordered couplings, non-critical points — and quantify precisely where and how the geometry-from-correlations correspondence degrades. The it-from-entanglement program is anchored in AdS-like settings; a systematic numerical failure map on ordinary lattice vacua does not exist and would be valuable to both believers and skeptics. The adversarial audit of this whole investigation demoted geometry-steering to a read-only speculation; this layer is the read head, built.

**Validation anchors.** Single-interval modular flow; Casini–Huerta two-interval data; RT on the HaPPY code; the entanglement first law δS = δ⟨K⟩ under perturbations.

**Discovery targets.** The nonlocality structure of multi-interval modular Hamiltonians. The quantitative failure boundary of metric reconstruction off-criticality. A toy Jacobson test: impose entanglement equilibrium (δS = δ⟨K⟩ for all small balls) as a constraint on lattice dynamics and measure how far it pushes the dynamics toward wave-equation behavior.

**Pivot trigger.** If reconstruction degrades uninformatively (noise rather than structure), the failure map itself is the paper, and the layer's tooling feeds Layer 6 optimization instead.

---

## Layer 5 — Time-modulated vacua

**Build.** Floquet dynamics on the lattice field: periodically modulated mass or couplings, Mathieu-type stability analysis, momentum-gap identification, photon-pair production spectra, entanglement growth of amplified mode pairs, and squeezing spectra — the dynamical Casimir effect and the photonic-time-crystal mechanism in one engine, with loss and finite pulse trains as Gaussian channels. No rigorous amplification bounds exist in this literature yet; the code's job is to produce numerically converged conjectures for them, mapping maximum amplification against modulation depth, frequency, and loss.

**The composite question.** Couple this layer to Layer 2: modulate the field before and during detector coupling, and ask whether Floquet-prepared field states raise harvestable entanglement downstream — pump-enhanced harvesting. Harvesting from squeezed and coherent states has been studied; harvesting from driven, momentum-gapped states has not been systematically mapped. This is a genuine composite with no published answer and both halves already validated.

**Discovery targets.** Conjectured amplification bounds for time-modulated vacua. The pump-enhanced-harvesting map. Loss thresholds separating the microwave regime (demonstrated) from the optical regime (the live experimental race).

**Pivot trigger.** If pump enhancement is null, the null is informative (it bounds what the time-crystal race can deliver for correlation applications) and the layer contracts to the amplification-bound work.

---

## Layer 6 — The differentiable vacuum

**Build.** Rewrite the Gaussian core in JAX so every pipeline in Layers 2, 3, and 5 is end-to-end differentiable: gradients flow through symplectic evolution, through switching waveforms, detector trajectories, gap schedules, modulation profiles, and multi-detector layouts. Then optimize under honest constraints — fixed energy budget, fixed signaling bound, fixed noise floor — for objectives that matter: harvested negativity, QET delivered energy, inequality-ratio minimization, amplification per loss. Where landscapes are rugged, wrap the simulators for reinforcement-learning search. Add a distillation back end: feed optimized harvested two-detector states through standard distillation protocols and compute the actual yield of usable Bell pairs per protocol run — the vacuum-to-Bell-pair compiler, with an exchange rate printed at the end.

**Why this layer can produce discoveries.** The harvesting and QET literatures overwhelmingly evaluate hand-chosen configurations. Systematic gradient-based protocol optimization over the full waveform space is nearly untouched, and order-of-magnitude improvements over published protocols are plausible precisely because nobody has looked with these tools. Optimized, noise-robust waveforms for the circuit-QED experiment, handed to the groups building it, would make this program a participant in the first demonstration rather than a spectator.

**Inverse design.** The final instrument: prescribe a target correlation or negativity profile and solve for the coupling matrix that produces it as a ground state — programmable vacua as an inverse spectral problem, the conceptual sibling of the inverse Casimir problem, and the cleanest expression of the program's thesis that the vacuum's structure is a design variable.

**Pivot trigger.** If optimized gains over published protocols are marginal (< 2×), the result still standardizes the field's baselines; effort shifts to inverse design, which has value independent of the margin.

---

## Layer 7 — The leaps

Explicitly speculative, pivot-rich, undertaken only from a fully validated stack.

**L1 — Sharp constants as conjecture-generation.** Push the Layer 3 stress-tester until the optimizing states stabilize, publish the conjectured 4D constants with the extremal states exhibited, and put them in front of the people who prove such things.

**L2 — Toy Jacobson.** The Layer 4 entanglement-equilibrium constraint, taken seriously: how much of wave dynamics is recoverable from δS = δ⟨K⟩ alone on a lattice with no gravity anywhere in the definition. Any nontrivial answer, positive or negative, touches the deepest claim in the field at the only scale where it is currently testable.

**L3 — A resource theory of vacuum manipulation.** Free operations: local passivity-preserving maps plus classical communication. Compute monotones numerically on small systems, derive one-shot QET bounds via smooth entropies, and chart what the theory says is the maximum any protocol of the Layer 2/3 type can ever achieve. This theory was identified early in the program as absent from the literature; small-system numerics is how it gets bootstrapped.

**L4 — Harvest-then-teleport.** Close the loop: harvest entanglement from the vacuum, then spend it as the channel for energy teleportation, and compute the full composite ledger — joules of measurement energy in, ebits harvested, joules made remotely available out. The composite exchange rate between vacuum correlation and energy access has never been mapped and is the quantitative form of the question that started this entire investigation.

**L5 — Interacting vacua.** λφ⁴ chains via DMRG and uniform matrix product states: how interactions modify negativity decay lengths, harvesting yields, and — above all — quantum energy inequality margins, where beyond the 2D integrable exceptions essentially no data exists. First numbers here are new regardless of what they are. This is the one leap that eventually wants a cluster.

**L6 — Hardware as noisy field.** Port the minimal protocols to public quantum devices as variational vacuum simulators — not for advantage, but because a protocol that survives real noise is a protocol an experimentalist will read.

---

## The anomaly protocol and the discovery register

**When something looks impossible.** Suspect the numerics first: precision escalation, lattice-artifact checks (dispersion, boundaries, finite-size scaling toward the continuum), switching-function smoothness, and the communication-vs-vacuum split. Then suspect the model assumptions. Only when a clean anomaly survives the full audit stack on two independent implementations does it enter the register as a candidate. The program's priors are explicit: passivity, the quantum inequalities, and causality are expected to hold, and the fastest route to embarrassment is announcing their violation from a rounding error at ν = 1/2.

**What counts as a discovery here.** Numerically converged sharp-constant conjectures with extremal states. The noise-threshold map that tells the circuit-QED groups whether their harvesting experiment can work, and the optimized waveforms that widen its margin. Order-of-magnitude protocol improvements from the differentiable stack. First interacting-field inequality data. The quantitative failure boundary of geometry-from-entanglement off holography. The composite harvest-to-energy-access exchange rate. Conjectured amplification bounds for time-modulated vacua. Every one is reachable from a laptop-to-GPU stack, none requires new physics, and several sit in territory no one has entered because the tooling this plan builds did not exist in one place.

**The direction, restated once.** Extraction is closed; the closure is encoded as executable audits and never contested by the program. Everything on the structure side — correlation, redistribution, delivery, amplification, design — is open, computable, and mostly unexplored. The program walks the open side with the closed side as its instrument panel.

---

## Anchor bibliography by layer

Layer 1: Srednicki (1993); Calabrese–Cardy (2004); Vidal–Latorre–Rico–Kitaev (2003); Peschel–Eisler review (2009). Layer 2: Reznik (2003); Pozas-Kerstjens–Martín-Martínez (2015); Simidzija–Martín-Martínez (2018); Tjoa–Martín-Martínez on communication contributions; Teixidó-Bonfill et al. (2025/26) and the Waterloo flux-qubit theses; Settembrini et al. (2022) for the measured vacuum correlations. Layer 3: Hotta (2008, 2011 review); Ikeda (2023); Frey–Funo–Hotta (2014); Alhambra et al. (2019); Ford–Roman (1995, 1997); Fewster–Eveson (1998); Fewster's lecture notes on quantum energy inequalities; Bostelmann–Cadamuro for the interacting exception. Layer 4: Bisognano–Wichmann (1976); Casini–Huerta (2009); Swingle (2012); Pastawski–Yoshida–Harlow–Preskill (2015); Faulkner–Guica–Hartman–Myers–Van Raamsdonk (2014); Jacobson (1995, 2015). Layer 5: Wilson et al. (2011); Dodonov review (2025); the photonic-time-crystal literature from 2022 onward. Layer 6: Bloch–Messiah/symplectic optimization literature; standard distillation protocols (BBPSSW, DEJMPS). Constraint spine throughout: Pusz–Woronowicz (1978); Faulkner–Leigh–Parrikar–Wang (2016); Balakrishnan–Faulkner–Khandker–Wang (2019); Chandrasekaran–Longo–Penington–Witten (2022).
