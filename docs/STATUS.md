# Status ledger — what is verified, what is a candidate, what is open

One paragraph per layer, kept in step with `README.md`'s layer table and the thirteen
MANIFESTs under `papers/`. "Verified" means a permanent regression test reproduces a
published number within a tolerance calibrated on the measured value; "candidate" means a
claim with no published answer, produced by a re-runnable notebook, admissible only as a
claim about the stated *model* (with the one device-level exception recorded in Layer 2)
and only under `papers/README.md`; "open" collects the unresolved items the agents
recorded in MANIFESTs and test docstrings. What is *new* versus a reproduction or a possibly-known result is
classified separately, candidate by candidate with citations, in `docs/PRIORITY.md` — read
that before reading any "verified" here as "discovered".
**Integration pass of 2026-09-12** at HEAD `d4f2ed0` plus this pass's own edits (prose only:
README, STATUS, PRIORITY, DISCOVERIES — no code, no data, no MANIFEST):
**1311 passed, 3 skipped, 10 warnings in 2158.92 s (0:35:58)**, zero failures
(`.venv/bin/python -m pytest -q -p no:cacheprovider`, 4 threads). The three skips are named
rather than counted: `tests/test_qei_certified.py:177` (the n_basis = 40 rung, behind
`QEI_CERT_SLOW=1`), `tests/test_hardware.py:844` (the layerwise depth scan, behind
`VACUUM_SLOW_TESTS=1`), and `tests/test_interacting_qei_s8.py:316` — *"no a = 1/2 pair has
landed"*, a Layer-5 pool row that had not arrived *at that run*. **That third skip is gone
as of 2026-09-13**: the a = ½ χ_op 32 pair landed and the skip fired into a pin (it was not
deleted), so the S8 file + `tests/test_exports.py` now give 61 passed, 0 skipped and the
repo-wide skip count at that integration pass no longer describes the tree. **The tree was not quiescent, and that is
part of the record:** the Layer-5 S8 job pools (PIDs 19494, 19495) were running throughout
with six workers, and their result files were still landing while this was written (six
untracked at commit time — two generous-knob MPO anchors at L = 5 and L = 6, an L = 6 exact
ladder rung, and three job specs). Those files belong to that track, are in no archive, no
numbers table and no pin, and were deliberately left for it to harvest. So this run is an admissibility record for everything *except* rows
that land after it; the 2026-09-04 record below is the last quiescent one.
The suite grew 1211 → 1311 (+100) over the 2026-09-12 work: the certified-enclosure tests,
the multimode-bound tests, the 2+1 and continuum-proof tests, the deep-VQE rows and the S8
pins, including several written specifically so that a wrong number or drifting prose fails.
Wall clock roughly trebled (12 min → 36 min) because ten of the machine's sixteen cores were
held by those pools.

Previous integration pass of 2026-09-04 on the quiescent tree at HEAD `c18a206` plus that pass's
own edits (prose only: README, STATUS, PRIORITY, one MANIFEST note and one draft line
reconciled to the MANIFESTs, and one `--quick` print in `notebook_qei_exact.py` made None-safe — no
numerical code, no data):
1211 tests green twice in a row (1211 passed, 10 warnings in 731.91s (0:12:11); 1211 passed, 10 warnings in 740.37s (0:12:20)); with every optional extra
blocked at the import system 1029 passed, 76 skipped, 0 failed; every notebook's `--quick`
path runs (15 notebooks across 13 candidates, the slowest in 307 s).

**Layer 0 — standing audits.** Verified: the energy ledger closes on every protocol in
every layer (worst defects 8.4e−12 in Layer 2, 9.95e−11 across the 1043-point `'xx'`
noise map, 8.99e−11 across the 591-point MI/coherence map, 5.90e−10 in the L4 composite,
4.0e−12 in the Floquet composite) — and, since 2026-09-04, closes against a **scaled**
criterion rather than an absolute one: the identity `close()` checks telescopes exactly,
so its only error is float64 roundoff, whose size is set by the energies being
differenced and by the number of accumulations booked. The criterion is
tol = max(1e−12, 4ε·√n_modes·n_ops·scale) with `scale` the booked |E| magnitudes and
`n_ops` the booked operations (a drive's midpoint substeps included), calibrated at 31×
the worst constant measured over a protocol scan (8–160 modes, 34–49152 operations,
⟨H⟩ = 5.7–530). **The trade is not uniformly stricter and is stated as such:** ~1000×
stricter at the floor (a small protocol is held to 1e−12 instead of 1e−9) and stricter
for every protocol whose roundoff floor is below 1e−9 — the criterion was *corrected*, and at the two motivating cells that correction happens to loosen it: **102× looser at the two
circuit-QED cells that motivated the change** (tol 1.02e−7 there against the old 1e−9) —
which is the point, since those cells' own float64 floor is ~1e−9, but it is a widening
where it matters most and it is what buys scale covariance: a protocol can no longer fire
or not fire on how energetic it is or how many substeps its convergence gate reached.
**What it catches and what it cannot:** it bounds the defect *relative to the protocol's
own energy scale*, so it catches any violation large relative to that scale — a mis-booked
drive, an unbooked channel, a phantom work term — at any system size, and arbitrarily
small ones in low-energy protocols through the floor; it **cannot** catch a violation that
is small relative to that scale but large relative to its own entry (on the ⟨H⟩ = 227,
49152-operation cell a 10 % mis-booking of a 1e−6 work entry does not fire, while a 100 %
one does). No criterion can do better there — that protocol's float64 floor is itself
~1e−9 — but fine-grained book errors are only resolvable in protocols small or cold
enough for the absolute floor to bind. One further trust assumption: the operation count
is *declared by the caller*, validated as a positive integer under a hard cap, recorded on
every ledger row and reported (with `n_entries`, `n_ops_excess`, `max_n_ops`) in the
closure result, so an inflated declaration is auditable after the fact but is not verified
against the integrator that ran; passivity of the vacuum under random local
symplectics (and, in L5, of the interacting DMRG ground state under random local
unitaries, the audit firing on a squeezed MPS); causality leakage < 1.7e−10 outside the
cone; the precision ladder float64 → longdouble → mpmath, needed in earnest by the
modular laboratory (float64 loses h beyond ~18 sites), by the multi-interval scan at
dps 106 and by the Lifshitz toy-Jacobson audit (1.3e−7 in float64, 8.6e−9 on the ladder).
Nothing is a candidate here.

*The anomaly register has its first entry — real, numerics, now resolved.*
On the first pass of the UV-cutoff bracket of `circuit-qed-noise-thresholds` the
**ledger-closure audit fired** on one cell — lattice spacing a = 0.292 (N = 103),
T = 80 mK, Γ_φ = 1e9 Hz — with a defect of −1.0107e−9 against the *absolute* 1e−9
tolerance: a 1 % overshoot of the tolerance and 4.45e−12 of that state's own energy
(⟨H⟩ = 227.0, 16384 channel plus 8192 work entries), i.e. float64 accumulation over the
entry count and not a physics defect. **The tolerance was not relaxed.** The single
*synthetic* Γ_φ column above the device's range (1e9 Hz) was dropped from the bracket
surfaces instead; both device anchors (1e7 and 2e8 Hz) are retained, E_N is exactly 0 on
the dropped column at every cutoff already measured, and every reported row passed the
unrelaxed audit (worst defect 4.73e−10). The event and the action are in that
candidate's MANIFEST and in `data/cutoff_bracket.json → audit_event`.

**The diagnosis was then confirmed, and the criterion was the marginal element.** A
41-run scaling study reproduced the defect (1.0107e−9 at halving level 9), measured its
growth with the entry count (pooled log-log slope **+0.81** over N = 48…24576) and its
proportionality to ⟨H⟩ (relative defect 4.45e−12 at ⟨H⟩ = 227 against 4.49e−12 at
⟨H⟩ = 55.3) — ≈ 0.6 ulp per midpoint substep, arithmetic and nothing else. The exposure
was structural, not data-specific: Γ_φ = 2e8 Hz, a *retained* device anchor, also crosses
the absolute criterion (1.0527e−9) at halving level 9, and the reported rows pass only
because the dt gate stops at level 8. **The fix is that the criterion now scales** (the
Layer 0 paragraph above), which is stricter, not looser, in relative terms; the tolerance
was never raised. Re-running both marginal cells under it: at level 9 the dropped 1e9 Hz
column closes to −9.985e−10 against a tolerance of 1.03e−7 (103× margin) and the 2e8 Hz
anchor to −1.0516e−9 against 1.02e−7 (97× margin, reproducing the recorded number to
0.1 %); at level 8, −1.26e−10 and −1.21e−10. **Neither fires**, so the event is closed by
correcting the criterion, with the original diagnosis intact. The dropped column remains
absent from the *archived* bracket surfaces (restoring it needs a re-run; it carries no
verdict information, E_N = 0.0), but it is no longer excluded because of an audit. Every
recorded worst-defect number in every MANIFEST is unchanged — they were measurements of
the defect, not of the tolerance — and every one of them is inside the old absolute 1e−9
as recorded and inside the scaled criterion as well: the tightest of them, the L4
composite's 5.90e−10 (1.7× inside the old tolerance, the thinnest margin in the
repository), sits on a protocol whose roundoff floor alone is 1.4e−9 at the test config's
16386 booked operations, so the scaled criterion gives it two more orders of room where
the absolute one had almost none.

Open: the audits have been pointed at a *rehearsed* device analysis but not at a real
one. The L6 on-ramp runs measured counts → energies → ledger closure at 3σ of the
propagated shot error, E_A ≥ E_B at 3σ and marginal no-signaling, and the whole chain is
exercised on a dated fake-backend calibration snapshot with falsification tests that
fire; what has never happened is the same code reading a real device's counts.

**Layer 1 — the vacuum as a data structure.** Verified: Srednicki's area-law coefficient
a = 0.294915; Calabrese–Cardy c = 1.0000004 (boson) and 1.0000188 (fermion, Jin–Korepin
correction reproduced); negativity sudden death at d = 14 sites with 40-digit margins; the
entanglement first law to 6.6e−10. No candidates, by design. Open: nothing; this layer is
pure reproduction and its product is trust.

**Layer 2 — the harvesting engine, and the program's first device-level statement.**
Verified: kernels and the exact Gaussian engine (moments and Im W ≤ 5.6e−17, continuum
response 7.4e−16), the PKMM negativity maps (closed forms ≤ 6.1e−16, Fig. 2 death lines
to 0.255 % out of sample), Gate A for the amplitude coupling (exponents 2.013 / 2.009 /
2.009) and for the derivative coupling (2.005 / 2.019 / 2.022, canaries rejected), the
Simidzija null (exactly 0.0 on three backends), the Tjoa–Martín-Martínez split, and the
M2.5 reproduction that causally connected derivative-coupled detectors take
|M_vac|/|M| = 0.9994 from field correlations (0.931 for xx).

*What closed, and with what number.* **The coupling identification — the single blocker
on the noise map's device claim — is closed by derivation.** `vacuum/detectors/circuit_mapping.py`
derives λ_osc from the proposal's own circuit Hamiltonian in seven cited steps:
λ_osc = |λ_TB| Ω_lat √(2Ω_lat) = **0.1414** (rounding spread ±0.0068 — the paper's own
three roundings of one quantity, no propagated measurement error; mapping band
[0.1185, 0.1482], which is what the device point was re-run across), provenance
`derived`. An independent BLIND re-derivation (`docs/coupling_rederivation/`, written without
reading the existing one, 14 tests) reaches the same equations, the same conversion factors,
the same √(2Ω) matrix-element map and the same number, 0.141 (+0.01/−0.02), adding a third
normalization source (Forn-Díaz 2017), an all-orders ∂ₓ↔∂ₜ argument and a dynamical test of
the two-level-to-oscillator map through the repo's own evolution; it flags the cutoff-unit
convention (λ_osc = 0.054 there) as something the write-up must state, and ~5 % of
unpropagated Z₀ uncertainty, both inside the existing band. The number is
unchanged (the omitted factor Ω_lat = Ω⁰/ω_ref is unity when the lattice energy unit is
the free gap) but its status is not, and the device-point rows were re-run at it, at ±1σ
and at four alternatives spanning 0.118–0.148. **The operator was then closed too:** the
map was re-run under the proposal's own *derivative* coupling (a 922-row `'xp'` pass whose
main (T × Γ_φ) surface is 16 × 16 at three couplings, a 90-row gap-tracking variant, a
288-row IR-mass control and a 369-row UV-cutoff bracket),
and it supports a **negative device statement** — the first statement about a device this
program has been able to make:

> the device's own pure dephasing, by itself, forbids resolvable entanglement harvesting.
> At both plot-digitised dephasing anchors (Γ_φ = 1e7 and 2e8 Hz) E_N is exactly zero
> down to T = 0, at the derived coupling and both ±1σ, at all three resolution floors, and
> at every IR mass over a factor 8. **The statement is independent of the resolution
> floor**: E_N is not small there but exactly zero — zero is below any floor — and the
> zero is real sudden death, not a partial-transpose clamp (ν̃ = 0.5134 at the Γ_φ = 1e7
> anchor, 1.3e-2 above 1/2), already exact from Γ_φ = 1e5 Hz upward. The floor-dependent
> restatement — the model tolerates only Γ_φ ≲ 5e4–1e5 Hz at T ≤ 12.5 mK, **100–200×
> below** the device's digitised range against a digitisation uncertainty of ~2× — is the
> weaker form.

Temperature is *not* the binding constraint under the derivative coupling: at 30 mK with
the measured minimum relaxation (Γ₁ = 8.7 MHz) E_N = 1.24e−4 (1.27e−4 with no relaxation
at all), the wall moves up to 30–35 mK and stops being
a sudden death. The statement is robust across a UV-cutoff bracket **14.7 → 73 GHz** at
fixed physical geometry — *through* the device's own 50 GHz line cutoff — with every
dephasing anchor still exactly zero, no anchor verdict drifting and the alive anchors
moving at most +25 %; across IR mass 0.05–0.4 (0.37–2.92 GHz) with zero verdict drift;
across gap tracking (`'xp'` |rel ΔE_N| ≤ 2.3 %, no verdict changed, while `'xx'` moves
33.6 % and loses its only derived-floor GO); and at a 0.0025 % finite-size spread. **The
Layer 2 pivot is walked**, and it fails at the device too: `mi-coherence-harvesting`
measures MI, both Gaussian discords, the classical correlation, the relative entropy of
coherence and the correlated coherence on 591 audited rows, each against a floor derived
for *that* quantity from the same Ren (2022) tomography, and finds max E_N = 0 with
survivors MI 2.399e−4, D^← 2.248e−4, C_cc 2.398e−4, C_r 3.187e−4 nats at 0.13–0.38× their
own floors — MI's own floor being **1.9× higher** than the negativity floor that had been
transferred to it, because a non-negative measure picks up a positive bias from
reconstruction noise where a concurrence does not. The one resolvable thing in the device
box is *local* coherence the noise creates (C_r above its floor at ⟨n_d⟩ = 0.269, with
C_cc far below); the measurable region opens on the coupling axis at 2.1–2.5× the derived
coupling, not on temperature; and the 60-row spacelike control sits 63× below its own
floor.

Candidates: `circuit-qed-noise-thresholds` (the `'xx'` surfaces remain a model claim; the
`'xp'` map carries the device statement above), `derivative-coupling-noise-survival` (on
all 27 surviving points of a 48-point thermal/loss/dephasing map |M_vac|/|M| ∈ [0.9994,
0.9996]) and `mi-coherence-harvesting`.

Open: a **positive** device claim is not supported and cannot become one by measuring an
input — nothing on any `'xp'` surface reaches the derived floor (max E_N 1.90e−4, and
2.33e−4 at the device's own 50 GHz, 4.3× under 1e−3), and 0 % of the surface is GO at that
floor anywhere in the cutoff bracket. What bounds any positive claim is not an input but
the model's own idealisations: oscillator detectors rather than qubits (MAP-1, leading
order in λ), a hard Brillouin edge rather than the device's exponential
C(ω) = e^{−|ω|/2Ω_cut}, and 1+1 rather than 3+1 — each worth tens of percent against the
negative statement's two orders of magnitude, so they do not threaten it. Γ_φ is still
plot-digitised (factor ~2) and there is no published error bar on γ, Z₀, v or Ω⁰ (the 1σ
used is the paper's own rounding spread, 4.8 %). The derivative-coupling survival map is
still one configuration, one λ, one switching shape, with its split taken from the
perturbative estimator on the thermal kernel (the nonperturbative engine has no
M_vac/M_comm of its own), its 1e−4 nats level is a *stated* model choice and not the
sibling's derived floor (against which its whole grid is sub-floor: max E_N 6.34e−4), and
the 1+1 claim does not extend to 3+1 at full light contact. In the pivot candidate the
discord is a *Gaussian* discord (an upper bound on the unrestricted one outside
Pirandola's family), the coherence names a basis, the tomography model is a stated model
of Ren 2022's protocol calibrated against its two reported numbers, and the MI
communication split is this candidate's construction rather than a published estimator.

**Layer 3 — energy.** Verified: Hotta's minimal model to 2.7e−14; QET on the critical TFIM
chain, ED ↔ DMRG to 4.6e−10 with the χ-ladder gate; the Ikeda 2023 IBM protocol in dense
simulation (paper values to 1e−4) and on Aer with the Table III noise model (qualitative
reproduction, the device's run-time gate noise exceeding its calibration recorded); strong
local passivity, FFH ↔ SDP over the 48-point Gibbs grid to 2.1e−12 with Bob's σ_y rotation
optimal over all CPTP maps; the 2d sharp QEI constant at 1.0e−7 with the extremal state and
the ν < 1/2 canary; the two-point-measurement work statistics with Jarzynski and Crooks as
audits. Candidate: the 4d sharp-constant conjecture, now **C_4d/C_FE = 0.45749 ± 0.00010**
(0.02 %), leap L1 — see there. Open: the 4d number is still a *numerical* continuum limit,
not a proof; the sharp constant is f-dependent, so the sup over all admissible sampling
functions is open; the QEI battery has never seen a violation, so the anomaly protocol has
still never been exercised on a *quantum-inequality* event (the register's one entry is a
ledger event); and Layer 3's Ikeda reproduction itself has not been re-executed on hardware by
this program — the program's one hardware excursion is L6's encoded-field QET information
signal on `ibm_fez` (2026-09-04: E_B_info = +0.04853 ± 0.00779, z = 6.23, the preregistered
prediction held), closed there.

**Layer 4 — geometry from entanglement.** Verified: Bisognano–Wichmann and Eisler–Peschel
profiles (0.14/L), Casini–Huerta two intervals with the lattice dictionary (local and
nonlocal weights converging as 1/L², modular-flow teleportation to 7.7 %), the fermionic
first law 2.6e−10, flow unitarity 3e−15; the χ = 8 MERA (e₀ 8.9e−6, Δ_σ = 0.1238,
Δ_ε = 1.0014, c = 0.5037); HaPPY RT exact on every connected region of the hexagon code;
Swingle's d = 2.06 ln r. Candidates: `multi-interval-modular-hamiltonians`,
`geometry-from-entanglement-failure-map` (degradation with structure on the clean axes,
into noise under disorder) and the toy-Jacobson leap (L2, below).

*What closed, and with what number.* Three of the multi-interval candidate's open items:
the n ≥ 3 deviation from the CH formula falls as A/L² at fixed gap/L (A = 1.906…1.953 for
three equal intervals with gap = L at L = 12, 16, 24, 32, up to 96 sites and dps 106) and
extrapolates to **+8e−05** — exact agreement in the continuum limit; the decay length of
the inter-interval couplings in a gapped chain is asymptotically the correlation length
itself, **ξ_h/ξ = 0.9985 ± 0.0257** from local decay rates at gaps of 8–16 ξ over
ξ = 5, 10, 20, 40 (the earlier 1.3–1.4 was a window fit contaminated by the massive
propagator's algebraic prefactor); and the gapped edge slope has the corner-transfer-matrix
closed form **s(m) = (2/π) k I(k′)** with k = 1/√(1 + m²) = sech(1/ξ) and level spacing
ε = π I(k′)/I(k) (`vacuum/modular/ctm.py`), reproducing the fitted 0.997/0.990/0.945/0.835
at ξ = 10/5/2.1/1.1 with residuals −4.2e−04, −2.0e−08 and ≤ 1e−12, and the lattice edge
slope to ≤ 1e−08 for m ≥ 0.2.

Closed by the literature: k(m) = sech(1/ξ) and hence s(m) are Eisler's published closed form — J. Stat. Mech. (2025) 013101, arXiv:2410.16433, Eqs. (64)–(72), for exactly this chain — matched to 0.0 at seven masses and to 2.2e−16 on the lattice edge slope by a test that writes his equations independently of `ctm.py`; the signed ladder reproduces his 2ℓ ± ½ edge rule. It is a reproduction at certified precision, and (per his Sec. VI) not a corner-transfer-matrix result.
2020 could not build a continuum picture) are untouched; the failure map's disorder axis is
seed-averaged over six seeds at N = 240 and the small-mass harmonic chain is an exception
dominated by its IR zero mode.

**Layer 5 — time-modulated vacua.** Verified: Mathieu tongue edges ≤ 1e−8, pair negativity
E_N = 2 arcsinh √⟨n_k⟩ to 1.5e−10, the Wilson 2011 DCE spectrum (prefactor 1.014, shape
1.9 %, depth exponent 1.99, thermal factor within 5 %), the ledger closing on the drive to
1e−14, the completely positive matched loss bath.

*What closed, and with what number.* **The amplification bound is a theorem**
(`vacuum/floquet/bounds.py`, `tests/test_floquet_bound.py`). In Prüfer variables the Hill
oscillator's gain per unit phase is pointwise monotone in the drive, so the maximiser is
s = sgn(sin 2θ) and the Pontryagin singular-arc question that made this a conjecture is
moot. *Theorem A:* for K(t) = K₀(1 + d s(t)) with |s| ≤ 1 measurable and 0 ≤ d < 1, every
solution obeys `ln r(t₂)/r(t₁) ≤ n artanh(d)` over n zeros of x — periodic or not — with
equality iff s = sgn(sin 2θ) a.e.; the Floquet corollary is
`ln max|eig S_F| ≤ Z artanh(d)` with Z the Prüfer winding, attained only by the
quarter-period bang-bang. *Theorem B:* the maximal Lyapunov exponent is ω_k λ*(d) with λ*
the unique root of artanh(d/(1+λ²)) = λ τ(λ), so **bang-bang is not rate-optimal** (λ*
exceeds artanh(d)/T* by 0.10 % at d = 0.1, 3.14 % at 0.5 and 44.1 % at 0.95; a free
two-dwell optimisation of the Meissner trace reproduces λ* to 1e−9). *Priors (2026-09-04;
`docs/PRIORITY.md` (a), carried in `bounds.py`):* the equality case is classical — Meissner 1918,
Arnold §25 — and bang-bang optimality of bounded-frequency pumping is established
(Lavrovskii–Formal'skii 1993; Andresen et al. 2011), Theorem B being the growth-rate form of that
literature's minimum-time problem; what was not found is the per-zero inequality for arbitrary
measurable s with its a.e. equality case and the closed-form λ*(d) — **POSSIBLY KNOWN, a
rederivation with citations** until a control theorist has read the module. The 2000-control ×
five-family × six-depth adversarial battery on exact piecewise monodromies, plus
projected-gradient and CEM, is now a check of the code, not evidence for the claim:
largest ratio to the bound 0.99975, bang-bang attaining it to 5.3e−14, cosine at 0.7854.
**The composite headline is corrected by finite size:** the N = 16 survey's 5–116× (and
its 156×) are finite-size numbers; at N = 64 the connected-window enhancements are 9.6,
22.0, 27.0 and 32.6×, i.e. **≈ 10–33×**, stable to ≤ 1.33 % from N = 32 while the
baselines move ≤ 2e−5. Every genuinely spacelike baseline is dead, so the spacelike result
is *revival* (E_N 4.60e−3 at N = 64) and the earlier "4.75× in the spacelike control" is
withdrawn — that window was connected. **The vacuum-vs-communication split of the pumped
harvest exists now:** the M2.5 estimator on the Floquet-prepared two-time kernel gives
|M_comm|/|M| up to 0.757 on the *unpumped* connected windows and ≤ 0.022 on every pumped
d = 0.1 row (|M_vac|/|M| ≥ 0.917 wherever E_N > 0) — entanglement *transfer* from a
Floquet-squeezed field, not vacuum harvesting.

*What closed on 2026-09-12, and what did not: the non-commuting multimode bound is
**partially closed** — a weaker bound is proved, the conjecture itself is still open.*
For pattern modulation K(t) = K₀ + d s(t) P, **Theorem M is proved**
(`vacuum/floquet/bounds_multimode.py`, `draft/multimode_bound.md`,
`tests/test_floquet_multimode.py`): in the canonical complex coordinates
a = (K₀^{1/4}x + i K₀^{−1/4}p)/√2 the coupling splits into a number-preserving
beam-splitter part and a squeezing part, and a Grönwall bound on the number norm gives
`ln max|eig S_F|/T ≤ (d/2)‖K₀^{−1/4} P K₀^{−1/4}‖ ≤ (d/2)‖P K₀^{−1/2}‖ ≤ ω_max d_eff/2`
for **every** measurable |s| ≤ 1. At one mode that reads ω₀ d/2 against Theorem B's
ω₀ λ*(d), so it is at most π/2 looser than the conjecture and sharper wherever
‖C‖/(ω_max‖A‖) < λ*(d_eff)/(d_eff/2) — sharper on all 63 archived battery rows
(0.492–0.870 of the conjectured bound), which reach at most **0.261** of the proved one.
Its operator-norm form is near-tight (batch maximum 0.879585 of the bound over 40 random
systems) and the same statement with the first-order constant 1/π is violated (1.3816483),
so the constant 1/2 is not slack. **Theorem F** (first order, the rotating-wave step
labelled ASSUMPTION in module and draft) gives the two-mode sum-frequency rate
(d_eff/π)√(ω₁ω₂): the frequency that governs a coupling is the geometric mean, not ω_max,
so a low-frequency pair cannot beat the conjecture at first order. **The conjecture
`ln max|eig S_F|/T ≤ ω_max λ*(d_eff)` itself remains neither proved nor refuted** — 128
archived polished optima (CEM + Nelder–Mead over ω, P, s and free durations at N = 2, 3;
detuned pairs; a first-order Schur-multiplier search) **reach** it: the largest archived
ratio is 1.000000000000017 (refined 1.000000000000036), so the bound is attained and
exceeded by +1.7e−14, inside the 10⁻¹³ roundoff of ln ρ, always at an effectively
single-mode configuration (a detuned pair driven to ω → (1, 1)) where Theorem B already
attains it, and never broken (`conjecture_refuted = false`); 1566 further exploratory
rows (scratch-only, explicitly labelled unarchived and unpinned) reach 1 + 5.4e−14. The refined conjecture
(‖C‖/‖A‖) λ*(d_eff) survived the same attack. Corollary M1's endpoints,
λ*(d)/(d/2) ∈ [0.63662 = 2/π, 0.68910], are **measured** on a 200-point grid, not proved;
only λ*(d) ≤ d/2 is proved. The archived data file regenerates **bit-exactly** from the
committed notebook at a later HEAD (0 differing floats outside build info and timers).

Candidate (`time-modulated-vacua`): the loss threshold Q_th = π/(2 ln λ_max) with its
platform mapping, and pump-enhanced harvesting (≈ 10–33× in connected windows, revival of
dead spacelike baselines, still work-inefficient wherever the vacuum harvests appreciably —
2.25 vs ≤ 0.53 nats per unit work at ω_mod = 1.6). Open: the multimode bound for
**non-commuting** modulation patterns, `ln max|eig S_F|/T ≤ ω_max λ*(d_eff)` with the
operator-norm depth, is **still conjectured** — what is proved is Theorem M with the
constant 1/2 and Theorem F at first order, and the archived battery reaches at most 0.169
of the conjectured bound and 0.261 of the proved one; also unproved there: that the
first-order Schur-multiplier norm of ŝ(ω_i + ω_j) never exceeds 2/π for more than one
resonant harmonic (tested to 0.986 × 2/π archived, 0.9999 exploratory, at N ≤ 4 and
harmonics ≤ 8), and non-attainment of Theorem M for N ≥ 2 (observed, archived maximum
0.6657). No device gain
is claimed (UV cutoff two lattice units, no line response A(ω)); the composite is still one
detector geometry (separation 7); and the M2.5 estimator is second order in λ, its
negativity off by 2.3–24× from the exact E_N at λ_max = 0.4, so only its *fractions* are
claimed.

**Layer 6 — the differentiable vacuum.** Verified: the JAX mirror agrees with numpy to
≤ 3e−15 on the Layer-1 anchors (9e−15 on the negativity-death curve) with gradients to
2.2e−9 against Richardson finite differences and finite at the vacuum floor; the detector
mirror to 2.4e−14 with protocol gradients to 3.6e−9; the PKMM optimum and death line
re-discovered from generic starts; Hotta's angle to 1.5e−13; BBPSSW/DEJMPS to 7e−16;
inverse design recovers the critical chain at 8.8e−13.

*What closed, and with what number.* All three Layer-6 open items. (i) *Best of two starts.*
A global multistart search in the k ≤ 5 family, under the same energy budget, noise floor
and signaling bound and through the same plain-numpy round trip, **beats every archived
optimum**: 30 mK 3.730 → **4.222×**, vacuum 2.892 → **3.183×**, the revived dephasing floor
6.52e−3 → **7.66e−3 nats**. Those are the numbers a manuscript should carry; the pivot
verdict (≥ 2×) is unchanged and strengthened. (ii) *The twin sat at a superseded base
point.* A second twin at the **shipped Table-I-scenario-2** point, every input re-read from
`experiments/` at run time and cross-checked against the map's own archive to 1.1e−13, puts
the optimizer against that map's NO-GO verdict — **and the verdict stands**: every
admissible round-tripped candidate stays dead, ν̃_PT moving only 0.513385 → 0.513368, a
clamp with a 1.34e−2 margin rather than a near miss. The NO-GO is not an artifact of the
waveform choice. (iii) *The signaling proxy is second order.* An exact **nonperturbative**
field-state / detector-zero-point split of the pair correlation is now available and is
shown to *be* the TMM21 split where TMM21 is defined (2.6e−4 / 2.9e−5 / 2.3e−6 / 6.4e−8 at
λ_osc = 0.3 / 0.1 / 0.03 / 0.01, clean λ² scaling, additivity to 1e−14, < 1e−8 spacelike).

Candidate (`differentiable-vacuum`): the gains above under equal drive work, the same noise
floor and a communication fraction ≤ 0.2159; the DEJMPS exchange rate 1.95e−5 → 3.44e−4
pairs per unit work; the < 2× pivot does not apply. Open: the global search **bounds the
landscape from below, it does not certify optimality** — 9–10 of 32 configured starts ran
before the per-floor wall-clock cap; optimized waveforms still need dt gate levels 15–16 of
a cap of 16; the waveform family cannot move the support, so the signaling bound is what
stops the optimizer exploiting the retarded signal; the proxy is now *known* to be off by
up to 0.139 at λ_osc = 0.3, where the O(λ⁴) terms are order one — it is a constraint
functional, not an estimate of the signaling share, and what survives the correction is the
ordering it was imposed for; the work check's 1e−4 tolerance is the ledger's own gate
uncertainty; and both twins inherit the noise-threshold candidate's declared synthetic
inputs.

**Layer 7 — the leaps.**

*L1 (sharp constants).* The 4d constant is now **C_4d/C_FE = 0.45749 ± 0.00010** (0.02 %),
C_4d = 2.897094e−3 = 1/(34.97 π²). What closed it: the `R/τ` expansion is **derived**
rather than fitted — the ball form factor χ_B(q)/V_B = 1 − q²R²/10 + O(R⁴) is even and
entire in R, so only even powers appear, with d₂ = −0.24932 in closed form (Hellmann–
Feynman) against the archived *fitted* −0.24748 — and the same variational problem is then
solved with **no lattice at all**, in continuum momentum space, giving 0.4574904 (grid
uncertainty 2e−6) and agreeing with the re-extrapolated lattice grid to **2.44e−05** (0.005 %;
the earlier "3e−07" was inflated by two extrapolations that imported the continuum d₂ and is
withdrawn); the 2d
control on that route returns Flanagan's sharp constant to 1e−04 for seven families. The
old 0.4591 ± 0.0027 is superseded and its band contains the new value. Also closed:
**Gaussians are not optimal in 4d** — six families span C_sharp(f)/C_FE from 0.2833
(Ford–Roman) to 0.6110 (the compact bump (1−u²)²), a factor 2.16 — so "the 4d sharp
constant" is f-dependent, this is the Gaussian one, and 0.6110 × C_FE is a lower bound on
the f-independent sup. **FFR's own number is reproduced first**: their 65 exact moments, all 44 Table I entries, Table II to 4.9e−12, and their fit give y∞ = 0.0236173 (theirs 0.0236175) by their method on our kernel — no convention is in question; a point mass at −0.0598 of weight 10 % moves the 65-moment edge only to 0.0252, so the method is blind to the lower support. Three routes agree in FFR units — lattice-free continuum solve 0.059762 ± 5e−7, Srednicki lattice 0.059684 with the box law measured for this weight (g^−2.5), exhibited state 0.059762 — so finite-box IR bias is excluded. Framing: our number is the infimum of the *spectrum* of T_f; theirs estimates the lower edge of the *vacuum distribution's* support; the two coincide in 2d (FFR 2010) and are conjectured equal in 4d, and **we do not settle which** — only the reproduction, the exhibited state as a spectral statement, and the blindness mechanism. **Corrected 2026-09-23: a reproduction, not a disagreement.** The acceptance test run on 2026-09-04 gives, in FFR's units (x = (4πτ²)²ρ; their FE anchor 27/128 reproduced exactly), x₀ = 0.05976 ± 1e−5 against their y∞ = 0.02361 ± 1e−5, a factor 2.53, with every rigorous statement of theirs respected (their Eq. (64) y∞ ≤ x₀; Fewster–Eveson; Ford–Roman). This paragraph recorded that as a claim against a published number, "the single item most in need of an expert's eyes". It had been published: Schiappacasse, Fewster & Ford, PRD 97, 025013 (2018), arXiv:1711.09477 — two of the three 2012 authors — diagonalized the same operator for the same weight in a finite cavity and found C_shift = −0.0593338, setting it beside the 2012 value themselves; ours reproduces it to +0.72 % (`tests/test_qei.py::test_ffr2012_lorentzian_acceptance` (vi) now pins both comparisons). What remains this candidate's own is the 2012 number reproduced by the 2012 method and the demonstration that the method is blind to the lower edge — the explanation of a factor the 2018 paper reports without explaining. The candidate's MANIFEST lists the withdrawn framing as no-longer-claims item 1.

*What closed on 2026-09-12, and on which side: the proof behind the numerical continuum
limit is **partially closed** — one endpoint of an enclosure is now a theorem, the other is
still a citation.* The analytic formulation is derived and matched to the code line by
line: the f²-smeared worldline energy of the 4d massless field reduces **exactly** (only
l = 0 and l = 1 survive, the T₀₀ angular factor being (1 + k̂·k̂′)/2) to four identical
copies of one radial quadratic form, with sector weights c₀ = 1/(4π²) and c₁ = 1/(12π²)
summing to 1/(2π²), so inf E_f = E_Q/(2π²) and R = C_4d/C_FE = −32 E_Q/(3√π) independent
of τ₀ — and `vacuum/inequalities/qei.py` is confirmed to build exactly that operator (no
discrepancy found; the one difference, the sign of the pair kernel, is immaterial under
a → ia and is recorded rather than hidden). On that formulation, **`C_4d(Gaussian)/C_FE ≥
0.457490709365` is a computer-assisted theorem** (`vacuum/inequalities/qei_certified.py`,
`papers/qei-2d-sharp-constant/draft/proof_4d_gaussian_constant.md`,
`data/s8_certified_enclosure.json`, `tests/test_qei_certified.py`): an explicitly exhibited
64-mode Gaussian state whose covariance is *verified* to be a state by an interval Cholesky
of [[N, M], [M, I+N]], evaluated in `mpmath.iv` with a **proved** Γ-series tail bound
(Wendel's inequality), with **no quadrature in ω anywhere** — so there is no discretisation
remainder to estimate, and the whole gap to the archived value is Galerkin deficiency. That
lower endpoint sits **2.46e−08** below the momentum route's 0.45749073394875556, i.e. 1/81
of that route's own stated 2e−06 grid uncertainty, on a monotone ladder
(n_basis 16 → 64: 1.52e−04, 1.77e−05, 3.21e−06, 7.59e−07, 2.15e−07, 6.92e−08, 2.46e−08)
whose interval arithmetic contributes nothing (relative enclosure width 10^−385.6 at
n_basis = 64). **The other endpoint did not close.** The only rigorous upper bound is still
Fewster–Eveson's R ≤ 1, *cited not computed*, so the certified enclosure is
**[0.4574907, 1]**, of width 0.5425 — anyone quoting "a certified enclosure of width 1e−5"
would be misquoting it. The obstruction is structural and is diagnosed rather than
papered over: ‖K^{−1/2} K_− K^{−1/2}‖ = 1 exactly and Tr(K_− K^{−1} K_−) = ∞, so every
perturbative lower bound on E_Q diverges, and the exact dual certificate needs certified
operator square roots of K ± K_− on the half-line. A non-rigorous Powers–Størmer
evaluation suggests R ≤ 0.576 is where a certified upper endpoint could land; it is
archived under a key naming it an **ESTIMATE — NOT CERTIFIED**, a test asserts that label,
and it must not be quoted. Open: the certified **upper** endpoint (and with it the proof
that the archived 0.4574904 ± 2e−06 is the answer rather than a numerical claim on its
upper side); the same machinery applied to the other five sampling families, where the
f-dependence 0.2833–0.6110 is still purely numerical (the closed-form Γ series is
Gaussian-specific); no certification of the 2d control was attempted; and the sup
over all admissible f — the (1−u²)^p trend points at the W^{2,2} regularity boundary and no
finite family search settles it.

*L2 (`toy-jacobson`): candidate, **de-circularized** on 2026-09-04 after an outside reviewer objected that K_loc used the CHM weight as an input. With the weight free (nonnegative Chebyshev families or per-site, LP over odd modes singly and in pairs), the minimizer on the massless wave chain *is* the CHM parabola — coefficients (1.0010, −1.0007, −0.0003) vs (1, −1, 0), overlap 1.0000 — with r_free = 4.1e−5 / 1.2e−5 / 2.5e−6 at L = 8/16/32, while Lifshitz (z = 2) is distinguished from the wave chain by the **complexity** of the admissible weight, not by its existence: the wave chain's scale-covariant weight is the 3-mode CHM parabola (found; overlap ≥ 0.9992 across six bases; 1.5e−4 shared across L = 8/16/32), while Lifshitz admits a scale-covariant nonnegative weight only at ≥ 11–12 modes of a bumpy profile (first below 1e−2 at 11 Chebyshev modes with z = 3, below 1e−3 at 12 hat modes; 0.09 at 8 modes; below 1e−2 by 11–13 modes and below 1e−3 by 12–14, z-dependent; 5e−4 only by 20 Chebyshev modes). With enough modes every tested dynamics fits — the toy tests parsimony of the modular weight; the 3-mode 0.27–0.68 and 8-mode 0.09 floors were family properties. Withdrawn: the "recovers v² = 0.980" claim — velocity is a unit under a free weight and that number was CHM's normalization; the mass is seen only through scale covariance (≈ 0.01). Caveats recorded with rank counts: the signed minimum for Lifshitz is 1e−3 with a negative-dipped weight 10–84× CHM, and an unconstrained per-site weight fits both dynamics to ~1e−6 because the variation space has rank 5–6 (no teeth).*

*What closed on 2026-09-12, and what did not — two units, both **partially closed**.*

**(i) The proof behind anchor 1's continuum limit.** `draft/proof_continuum_limit.md`,
`vacuum/geometry/jacobson_continuum.py` (mpmath, **no lattice code**),
`data/H_continuum_check.json`, `tests/test_toy_jacobson_continuum.py` (20 passed).
*Proved:* the first law from S = −Tr ρ ln ρ with its regularity stated; that the
**continuum** residual is identically zero on every first-order variation — no limit
taken — so the lattice r(K) is pure discretisation error and the free-weight minimiser is
β exactly *provided the variation family is complete*, a hypothesis known to fail here
(rank 5–6 per ball), which is why anchor 9's claim is parsimony and not existence; and
Lemma 2.2, that the reflection-odd variations are exactly blind to the 1+1 zero mode
(measured top-mode share 1.5e−17). The CHM/BW form is **cited, not reproved**.
*New, and the rate:* an **exact** lattice identity — the site-centred (trapezoid) and
bond-centred (midpoint) discretisations of ∫βT₀₀ differ on a squeeze of a lattice normal
mode by a closed-form finite sum, ρ_site − ρ_bond = −(1 − ρ_bond)·ΔdK/dK_bond (Prop. 4.3,
no approximation), whose continuum limit is A(κ)/L² with κ = πL/λ, A(0⁺) = −3/2. At
N = 2400 over nine (L, λ) cells the exact identity reproduces the measured residual to
**6.3e−12** and the closed form to **3.6e−02**. Consequences: the dominant part of anchor
5's site floor 1.394/L² is *derived* — it is the trapezoid rule's weight-placement error,
not a property of the first law — and the **exponent is exactly −2** at fixed λ/L, so the
fitted **−2.27 is a fit artefact** of the (L, λ, N) grid. The continuum first law with the
exact CHM modular Hamiltonian is certified at |δS − δ⟨K⟩|/|δS| = **4.3e−32** (dps 40), with
mutations that fire (a weight scaled by 1.001 → 1.0e−03; β(1 + 0.05x²) → 1.37e−02).
*What did **not** close, and is labelled ASSUMPTION in the draft:* **A5 — that the absolute
lattice residual converges to the continuum one, r_∞ = 0, is not proved**; §4 proves only
the size of the difference between two discretisations of the same continuum object. **A4**
— the O(1/L) remainder of the closed form is *measured* (3.6e−02 worst cell, 8.4e−04 best),
not bounded. The site floor is therefore **not** derived "to 0.1 %": that figure
(8.3e−04) belongs to the *difference*, and the site coefficient is that difference plus the
**underived** bond-centred floor 0.053/L² — unaccounted at 3.8 % (L = 8) to 13.6 %
(L = 32) at the dominant wavelength and 116 % in the worst cell. Both conflations were
caught by verifiers and are recorded as corrections in the MANIFEST, the draft's status
table now quotes the archived remainders, and a test reads that table and fails if it
drifts from the data file.

**(ii) The even sector and the 2+1 extension.** *Item (a), 1+1 reflection-even — partially
closed, and narrower than first written.* The even-sector first-law defect is the massless
scalar's **IR zero mode**: it approaches the ball's top even Williamson-mode entropy
contribution as the variation's wavelength grows (ratio 1.144, 1.139, 1.158 at L = 8, 16, 32
**at λ = 16L**; it is *not* that number over the family — −0.752, −1.288, −3.738 at
λ = 4L — what is uniform is the convergence, |ratio − 1| falling monotonically
1.75 → 0.99 → 0.14 at L = 8); the obstruction is IR, not UV (at fixed L = 8 the mode's ν₁
grows 0.934 → 1.199 and r_even 2.04 → 3.60 as the chain goes N = 200 → 3200, while r_odd
falls); it is not a numerical artefact (four independent refutations, the projected δS
agreeing with a finite-difference route to 1.5e−06 and the mpmath rung to 2.1e−11); and the
(L, m) scan cannot decide it, reported as the negative result it is. Projecting the softest
even modes out drives the **absolute** defect 1.34e−02 → 5.3e−10. **RETRACTED (fix round 2):**
"once the soft even modes are removed the local CHM form reproduces the even sector better
than the odd one in absolute terms (5.3e−10 vs 4.8e−06)". |δS − δ⟨K_loc⟩| is not scale
invariant — the same projection shrinks ‖δV_B‖_F too — and per unit variation the
conclusion **reverses**: matched cell by cell the projected even sector is **1.9×–659×
worse** than the odd sector and never better. So the even sector's *large* residual
(r ≈ 2–3) is the zero mode and is removed by projecting it out, but the *remaining*
residual (r ≈ 0.016–0.17) is **not** explained, and the even sector is **not** shown to be
a place where the local CHM form works in 1+1. *Item (b), 2+1 — closed, with a number.*
The toy now runs on an N × N Dirichlet square lattice with disks R = 3…10 at N = 128
(`vacuum/geometry/jacobson2d.py`). The canonical ansatz **fails** there and does not
improve with R (r = 0.451, 0.380, 0.370, 0.357, 0.387, 0.382), with the deficit entirely in
the xx quadrature. The missing operator is the free scalar's **conformal improvement**
Θ₀₀ = T₀₀ − ξ∇²(φ²), identically zero in d = 2 — which is why the 1+1 toy never saw it —
and 1/8 in d = 3; fitting it cell by cell **measures** ξ = **0.1278 ± 0.0040** (odd, 15
cells) and 0.1322 ± 0.0043 (even) against the conformal 1/8, and the *shape* is identified
too (relative scatter 3.1 % for ∇²β against 7.7–32.9 % for three wrong shapes, one of which
needs a negative ξ). With ξ = 1/8 the residual collapses to 0.005–0.075, converged in the
box. This is the first place in the candidate where entanglement equilibrium *measures* a
coefficient of the continuum theory rather than checking one — and it is the decisive
control for item (a), since in 2+1 there is no IR zero mode and the even/odd residual ratio
is 1.8–24.2 (median 3.6) against 1+1's 3476, 8481, 6297. Also **retracted (fix round 1):**
"the exact modular Hamiltonian's pp diagonal equals 2πβ to 3 %" (no data file, no test, and
the slope drifts 1.106 → 0.743 over R = 3…10); what survives is the **row sum**, flat in R
and 2–14 % high, now archived and pinned. Open after both units: r_∞ = 0 and the
O(1/L) remainder (A4/A5 above); the underived bond floor, which is what limits the site-floor
derivation; the unexplained leftover even-sector residual after zero-mode projection; the
improvement's entangling-circle contact term, **dropped not derived** (keeping it gives
0.73 against 0.075); ξ **fitted on 15 cells, not derived**, and no 3+1 run to test whether
the toy tracks (d−2)/(4(d−1)); one ball position in 2+1, a jagged lattice boundary, and
λ ≤ N + 1. Still no gravity, area term or Einstein equation anywhere, in either dimension.

*L3 (`resource-theory-vacuum-manipulation`).* Candidate — the QET potential and negativity
are monotones (12 500 random free operations, worst violation 2.07e−14, teeth up to 7.7),
the plan's QET surplus is not, the one-shot bound is saturated by Hotta's model and holds on
chains, and the single-site rotation can sit 12.4× below the CPTP optimum at g = 0.5. All
three open items are now theorems. **Two-way communication gives Bob nothing:** W_→ is a
*linear* functional, W_→(ρ) = Tr[W_op ρ] with W_op's non-Hotta part supported on Alice's
side alone (Theorem 4), so Theorem 1's inequalities are equalities **for commuting-Kraus
instruments on A** — hence, by Theorem 2, for the whole free class exactly where the bond
spectrum is in convex position (every shipped model) — and every finite-round two-way
protocol there pays Bob exactly W_→. Off convex position part (b) is **false and fails
one-way**: one free instrument (defect 9.2e−16) raises the closed form 0.0366 → 0.2253, a
factor 6.15 — worst deviation 1.3e−14 over the
minimal model and L = 5..8, worst ledger drift 9.0e−15 over 1180 random two-way protocols,
with teeth (a non-free instrument on A breaks it by ≥ 1.083); what two-way buys is a second
device, and the net of the injections is round-independent. **The free class is
characterized without unitality** (Theorem 2: an energy-non-signalling instrument on A is
commuting-Kraus iff every joint bond eigenvalue vector is an extreme point of their convex
hull), with an explicit non-unital counterexample outside the commutant and a certified SDP
over *all* free instruments agreeing with the analytic verdict on every row — 0.000 on
every model used here, 2.000 / 6.000 on two- and three-Alice-qubit star models, where the
wider class extracts 5.15× more and the closed form is not even monotone. **The tight bound
for a single-site Bob is exact, not entropic:** E_B = Tr T + ‖T‖_* with the determinant fix
(orthogonal Procrustes) reproduces the chain protocol *and* Hotta's Eq. (11) to 4.4e−16 in
energy units over 96 rows, so the marginal-fixed SDP's 0.973 was its own relaxation gap.
Chains now reach L = 12. Open: the two-way result is for *this* free class and finite rounds
(unbounded-round LOCC is not a closed set); Theorem 2 is for commuting bonds, and where
convex position fails the *optimal* wider-class instrument is not identified (one witness is
exhibited, not the optimum); the exact bound is for a single-qubit Bob, and a multi-qubit
Bob still has only the marginal-fixed SDP; the proved entropic bound
E_B ≤ 2c(e^{−H_min(X|R)} − ½) is ~2 orders loose (tightness 0.0080) and an entropic quantity
retaining the ‖T‖_* cancellation is open; the Bob-equals-complement CPTP optimum past
L ≈ 10 is out of reach, and nothing here is a thermodynamic limit.

*L4 (`harvest-then-teleport`).* Candidate — the composite exchange rate surplus/W_in peaks
near 6 % on harvesting points and at 0.125 where negativity is already dead, eleven of 27
`'xx'` rows have E_N = 0 with positive surplus (correlation, not negativity, pays), the
Landauer bound holds at ≤ 0.74 and the conjectured joint-ergotropy bound is false. Three
open items closed, on a map extended to 61 audited rows. **Non-Gaussian conditioning never
beats Gaussian anywhere on the map:** photon-number-resolved conditioning on Alice reaches
at most 0.159 of the Gaussian optimum and on/off at most 0.103, the largest ratios on the
hottest, most strongly coupled row; on a two-mode squeezed vacuum the two families tie
exactly (PNR gain Ω sinh² r) while on/off gives the strictly smaller Ω tanh² r, both closed
forms anchored to 1e−10. "Gaussian is a lower bound" is therefore now a measured statement
about which families were tried. **The derivative coupling harvests at light contact, not on
spacelike windows:** at delay = separation = 12 `'xp'` carries E_N = 0.0012 where `'xx'` has
exactly none, while on every row outside the nominal light cone — not *strictly*
spacelike: measured lattice commutator leakage reaches 2.76e−02 two sites beyond the cone
and 1.85e−05 at six, so the null is conservative — and on a dedicated 32-point scan neither
coupling carries any negativity (smallest PT margin 8.0e−07 *above* the separability floor)
— yet both carry a positive surplus, for the same reason it is positive at E_N = 0 anywhere.
**The IR limit is taken:** massless is admissible on the Dirichlet chain, the m → 0 sequence
is regular, and over four chain sizes both the `'xx'` switching work and its surplus are
linear in ln N (slopes 0.0427 and 0.00539), so the base-point exchange rate rises 32.2 %
from N = 41 to 121 toward the log-slope ratio 0.126, while the `'xp'` rate is IR-safe
(−0.3 %) because it does not couple to the k = 0 mode. Open: the surplus is
communication-assisted extraction from an active state, not teleportation (the draft says
so); whether an *unrestricted* POVM beats the Gaussian family is still open
(Bernards et al. 2019); the squeezed-basis PNR column is a diagnostic grid, not an optimum,
and interpolates to a Gaussian homodyne; four chain sizes are a trend, not a limit theorem,
and every other row is quoted at the regulated mass; the Hotta reference speaks for the
coupled pair's ground state, not the harvest.

*L5 (`interacting-vacua-first-numbers`): candidate, **diagnosed on 2026-09-04** — the QEI margin is an **exact lattice infimum** (MPO route, cross-checked by a dense-ED second implementation to ≤ 1.9e−4 where the operator truncation is converged (measured at L = 6, n_max ≤ 3; no ED agreement is claimed beyond that); `vacuum/interacting/qei_ed.py`). The free bound survives (exact/E_free = 0.29 at λ = 4). Interactions renormalize the reference by a **mass shift**: the λ = 4 dispersion is a sharp quasiparticle at every k (≥ 99.86 % spectral weight), nearest-neighbour to 7e−5, gap² 3.3 % *below* Hartree, so the sharpest quadratic reference K_eff is 4.5 % deeper and "the reference is wrong" is rejected — the residual against it is larger. Beyond any quadratic reference the residual is a **first-order quartic tax**: exact Hellmann–Feynman/Wick slope dR/dλ = +0.081 (L-independent), the sign measured directly on the extremal state (+10 % of |E_min|). **Resolved at λ ≤ 1** (1.4 / 1.5 / 2.6 % against a 0.19 % band — and that band is now known to be **BLAS-thread scatter**, not a convergence estimate: the same spec run at 2 and at 4 threads moves E_min by 0.199 % at L 16, n_max 8, χ_op 32, λ = 0, and the archive's worst same-spec spread over 16 repeat groups is **5.12 %** at L 5, n_max 6, χ_op 32, λ = 4, with 1.4–2.4 % common at λ = 4. An earlier "threading moves E_min by < 1e-8" is retracted. The band is therefore per-spec and must be measured wherever a number is quoted); **BRACKETED, not resolved, at λ ≥ 2** — the infimum is not monotone in n_max on any route (the truncated chain is a different dynamical system), the archived knobs inflate it, and the anchored (n_max, χ_op, weight) plane gives only a bracket: the control-normalized residual spans [−1.797 %, +4.415 %] against E_H at λ = 2 and [−34.349 %, +15.131 %] at λ = 4 (a 32- and 239-band systematic carried by the occupation weighting); against K_eff its SIGN is positive at 5 of 5 and 4 of 5 settings of the frozen 15-row snapshot (5 of 6 and 4 of 6 live), which is the weaker claim the data do support. Every λ = 4 number is therefore a value *at its knobs*: B's archived exact/E_free(K_eff) = 0.835 maps the archived 0.873 by the cutoff-free 1.0452 and re-reads as 0.913 / 0.956 / 0.950 at n_max 8 / 9 / 12 with the same knobs (0.901 at n_max 6, χ_op 64, weight 0.5; 0.917 with the exact SVD), and the archived-knob decomposition's +10 % quartic fraction is likewise a value at those knobs — only its sign is carried forward. Scaling: the constant shifts, the exponent does not (τ₀ exponent −1.2 ± 0.8 vs the predicted +2, within τ₀ = 0.375–0.75). Retracted, listed once in the candidate's §6.7: the 86–95 %; "the whole reduction is mass renormalization"; the φ⁴ shell mechanism; "non-monotone control ⇒ MPO truncation"; Rayleigh–Ritz monotonicity for the smeared infimum; "the 1.3 % band is MPO truncation"; "≥ 0.905, ≤ 10 %, consistent with zero"; and, from the S8 triage itself, the "328×" exact-SVD penalty (its denominator was 29× below the route's own thread scatter), "BLAS threading moves E_min by < 1e-8", three status labels and two unscoped superlatives — **twelve items in all**. Open: the λ ≥ 2 magnitude — bracketed, not resolved (see the S8 harvest below), τ₀ ≥ 1.0, one mass and one lattice spacing usable, and the archived extremal MPS's O(1) content outside the light cone (flat directions of the compressed operator).*

*What the S8 job pool decided on 2026-09-12, and what it did not — **partially closed**, and
the part that matters is **not** closed.* **Item 5 (τ₀ ≥ 1) is closed NEGATIVELY by
measurement.** Every τ₀ ≥ 1 row run at χ_op 16, 32 and 48 (n_max 8, weight 0.5,
χ_dmrg 16) fails the harvest's own gates: at τ₀ = 1.5 the λ = 0 control improves
monotonically (1.33774 → 1.25306 → 1.21742) but is still 21.7 % off 1 with reference
inconsistency running 2.3 % → 117 % → 64 % of |E_min|; at τ₀ = 1.0 it moves *away* from 1
(1.03379 → 1.03614 → 1.03843) and its inconsistency crosses back over the 5 % gate
(9.8 % → 2.7 % → 7.65 %). **χ_op is not the missing knob at either τ₀**, no ratio is
quotable there, and the scaling exponent stays restricted to τ₀ = 0.375–0.75. (The a = ½ χ_op 32 pair
**landed 2026-09-13 and closes that sub-item NEGATIVELY**: λ = 0 control
`exact/E_H = 1.1865` (18.65 % off 1), reference inconsistency 20.01 % at λ = 0 and 175.32 %
at λ = 1, `control_ok false`, `usable false`, so **no a = ½ ratio is quoted**; the χ_op 48
pair is still queued. That was the S8 file's last skip and it is gone —
`tests/test_interacting_qei_s8.py` + `tests/test_exports.py` now run **61 passed,
0 skipped**.) **Item 4 (the λ ≥ 2 magnitude) is BRACKETED, not resolved — it is not closed.**

**FINAL, 2026-09-13: the pool finished and the answer got WORSE.** All 159 S8 jobs landed; the complete 36-row plane replaces the frozen 15-row snapshot the brackets above were quoted from, and it widens them 6.7-fold at λ = 2 and 2.9-fold at λ = 4: the control-normalized residual spans **[−34.382 %, +6.979 %]** against `E_H` at λ = 2 (**[−31.613 %, +8.896 %]** against `K_eff`) and **[−127.079 %, +15.861 %]** at λ = 4 (**[−117.262 %, +19.498 %]**), over 12 settings at each λ, with **12 of 12** rows failing the 5 % reference-consistency gate at λ = 2 and 11 of 12 at λ = 4. The knob spreads are 30.8 % (weight), 41.4 % (χ_op) and 20.4 % (n_max) at λ = 2, and **142.0 %**, 32.6 % and 46.7 % at λ = 4, against a 0.192 % band. Five of the six knob spreads grew as the plane completed; the χ_op spread at λ = 4 is unchanged at 32.6 %. At λ = 2 the three axes are comparable (20–41 %); at λ = 4 the weighting dominates (142 % against 33 % and 47 %). Spreads of 20–142 % against a 0.192 % band, growing as higher-knob rows land, are the signature of a route that does not converge here rather than one short of resolution. The sign is weaker than the snapshot suggested, not gone: against `K_eff` the residual is positive at 9 of 12 settings at λ = 2 and 7 of 12 at λ = 4 (5 of 5 and 4 of 5 on the snapshot), and against `E_H` at 4 of 12 and 6 of 12. *(Corrected 2026-09-23: this paragraph first said the bracket widened "about an order of magnitude", that adding rows moved it outward "on every axis", and that the sign "no longer survives"; all three overstated the data in `s8_convergence_plane2.json`.)* Item 4 is therefore **not closed, and more of this plane will not close it**; closing it needs an operator representation that does not spend bond dimension on high-occupation matrix elements, with an exact anchor at every quoted setting.
Its replacement plane (`plane2`: n_max, χ_op and **weight** at the sketched SVD, every
column anchored to exact ED at L ≤ 6) has landed 15 rows with their own λ = 0 controls, out of 36 specified — frozen by tag as
`unit2_snapshot` so the quote stays reproducible — and gives a **bracket**:
the control-normalized residual spans **[−1.797 %, +4.415 %]** against `E_H` at λ = 2
(**[+0.301 %, +6.385 %]** against K_eff) and **[−34.349 %, +15.131 %]** at λ = 4
(**[−28.541 %, +18.800 %]**), the λ = 4 bracket spanning both signs. The systematic is
**32.3 bands** of the sweep's 0.19229 % at λ = 2 and **238.6 bands** at λ = 4 — 11× and
80× the 3-band quotability criterion — and it is carried by the **weight** axis (6.212 and
45.879 pp), not χ_op (1.198 and 32.618 pp). n_max is the smallest axis **at λ = 4 only**
(0.728 pp = 3.8 bands); **at λ = 2 it is the second largest, 5.471 pp = 28.5 bands**. The ED-anchor
gate clears exactly **one** setting per λ and they are **different** settings (χ_op 64
weight 0.5 at λ = 2, **+3.217 %**; χ_op 64 weight **1** at λ = 4, **−1.731 %**), so there is
no spread to measure and no λ-series to read; the verdict is unchanged at every anchor gate
from 1 % to 5 %. The **sign** survives where the size does not: against K_eff the residual
is positive at 5 of 5 settings of the frozen snapshot at λ = 2 and 4 of 5 at λ = 4 (**on the
live tree, 5 of 6 and 4 of 6**; the extra λ = 2 negative fails every gate). Every λ ≥ 2 row fails the
block's own 5 % reference-consistency gate (5.27 % to 257.29 % of |E_min|), a gate already
measured anti-correlated with the true error. Both brackets lie **entirely below** the exact
O(λ) prediction (+16.182 % and +32.364 %), and the implied second-order coefficient
([−0.0449, −0.0294] at λ = 2, [−0.0417, −0.0108] at λ = 4) **overlaps no λ ≤ 1 value on that
snapshot** (−0.0679 / −0.1061 / −0.1028 / −0.0553 at λ = 0.1 / 0.25 / 0.5 / 1), so one
λ-independent quadratic fits neither range together *there* — on the live tree the λ = 2 c₂
bracket is already [−0.0958, −0.0294] and contains the λ = 0.1 and λ = 1 values. Testing
closure would cost **12 jobs, 27.936 h nominal / 42.010 h calibrated** and would still not deliver it, because a second
anchor-qualified (χ_op, weight) pair is also required and none is on the queue
(`papers/interacting-vacua-first-numbers/data/s8_convergence_plane2.json`).
**RETRACTED this pass** (the candidate's no-longer-claims list, item 8): *"at λ = 4 the
operator truncation at the sweep's knobs is a 3–12 % effect wherever an exact answer
exists."* It was false on this tree the moment the L = 5, n_max 8, λ = 4 ED rung landed:
at the sweep's **own** Fock cutoff the MPO–ED deviation is **+39.4 %** (and **+42.3 %** at
the plane's knobs, exact SVD), three to four times the quoted ceiling. The measured
statement is the opposite of a bound — at fixed χ_op 32 the λ = 4 error **grows** with the
cutoff, 3.4e−02 (L 6, n_max 5) → 1.19e−01 (L 5, n_max 6) → 3.94e−01 (L 5, n_max 8), while
the λ = 0 error at the same points stays 0.34–1.38 % (a factor 29 apart at n_max 8) — so
the route **understates the magnitude of E_min by a factor 1.65** at the cutoff every
quoted S8 λ = 4 ratio and the first plane point are taken at. A seven-row knob ladder at
that one point (L = 5, n_max 8, τ₀ 0.75, λ = 4, against the exact −2.1561127e−02) says
what closing item 4 would now require: χ_dmrg 16 → 32 changes the answer by ~1e−06 (the
variational bond is **converged**); χ_op 32/48/64/96 gives +39.4 / +14.9 / +13.7 / −4.5 %,
|error| falling **non-monotonically and crossing the exact answer**, so no χ_op there is a
one-sided approximation; and weight_decay 0.5 → 1 at χ_op 64 is worth a **factor 11**
(+13.7 % → −1.19 %) — the **weighting**, not the operator bond dimension, is what costs
λ = 4 accuracy there. Most consequential for the plan: the harvest's own 5 %
reference-consistency gate **does not track the λ = 4 error** — it is not monotone in it,
the best-gated row (refcons 1.21 %) is 13.7 % wrong and the most accurate row (−1.19 %)
fails the gate at 36.1 % — so passing the gate certifies nothing about a λ ≥ 2 magnitude,
an independent measured reason item 4 stays open even when plane rows begin to pass — and
the plane2 harvest confirms it, with all ten λ ≥ 2 rows of the frozen 15-row snapshot —
and all twelve on the live tree — failing that gate. **No
λ ≥ 2 magnitude is quoted anywhere in the candidate; only a bracket is**, and the method is now stated as
cross-validated at λ = 0 and **not** cross-validated at λ = 4 at the cutoff the paper uses.
Two caveats travel with this: the exact side of the n_max 8 pairs is *itself* unconverged
in n_max (the L = 5 λ = 4 exact ladder is still falling 38.6 % per rung), so these are
comparisons between two unconverged routes and not error bars on a known answer; and that
the L = 5 knob ranking carries to the paper's L = 16 is an **ASSUMPTION**, recorded as such.
Two rows sit *below* the same-grid exact ED infimum (χ_op 96 by 4.5 %, weight-1 by 1.2 %);
that is an operator-truncation artefact — the MPO minimises a truncated operator, which is
not bounded below by the exact one — and is explicitly **not** an anomaly-protocol event.

**The pool is still running, and its wall-clock figures are upper bounds.** Both S8 pools
(PIDs 19494 and 19495, started 10:28:53 PDT 2026-09-12) were alive at this integration
pass with six workers: the two n_max 10 plane jobs, the L = 6 exact ladder, and three
generous-knob MPO anchors. **18 of the 20 plane jobs remain.** Nothing new was launched by
the integration pass and nothing was killed. To harvest whatever is on disk without
starting a job:
`OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 .venv/bin/python papers/interacting-vacua-first-numbers/notebook_qei_exact.py --diagnostics --only s8 --no-new-jobs`,
then re-run `tests/test_interacting_qei_s8.py` and add the new rows to the pin tables.
Operational note for whoever runs the pools next: all six workers were found **SIGSTOPped**
(`ps STAT T`) for ~80 minutes on 2026-09-12 and were resumed with `kill -CONT`, so a job's
recorded `time_s` spans that suspension — the "2.37× the wall estimate" and the 873 h plane
projection are **upper bounds of unknown tightness**, not cost measurements. No physics
number depends on them.

*L6 (`hardware-as-noisy-field`).* Candidate — on the sourced ibm_cairo calibration QET
teleportation proper is lost (Bob's CNOTs heat more than the 1.2 % he extracts), the
equal-depth information signal survives to 22× the gate error, harvesting is dead and
revives at ≈ 0.46× the noise, and the free-tier run is specified (3 qubits, 11 circuits,
depth 36, 3439 shots). **The on-ramp is now built, audited and rehearsed end to end**
(`vacuum/hardware/real_device.py`, `scripts/run_ibm_free_tier.py`,
`tests/test_hardware_real_device.py`): the batch is built from the archived working point
with its spec re-asserted, one lowest-noise 3-qubit path is selected from the live
`backend.target`, ISA transpilation re-asserts the two-qubit budget and forbids routing,
submission goes through `SamplerV2`, and the analysis runs the *measured* energies through
the standing audits (ledger closure at 3σ of the propagated shot error, E_A ≥ E_B at 3σ,
marginal no-signaling by two-sample χ²), all recorded in a hashed provenance record. The dry
run on `fake_sherbrooke` (Eagle r3, calibration snapshot 2025-02-26, physical qubits
[122, 123, 124], ISA depth 40, 8 two-qubit gates, 3439 shots, 15.2 s of QPU time by IBM's
transcribed estimate) gives **E_B_info = +0.0474 ± 0.0133 (z = 3.55)** sampled and
+0.0391 ± 0.0138 (z = 2.83) shot-free against the dense ideal 0.0415, with every audit
passing and teleportation proper still negative (E_B_raw −0.0262 ± 0.0098). The audits have
teeth: a record with Bob's bit flipped in the control family fails closure, the marginal
no-signaling test and its own sha256; Alice's flipped fails E_A ≥ E_B.

**Closed: the L6 excursion ran on a real device (2026-09-04).** With the user's token, `scripts/run_ibm_free_tier.py --no-wait --n-sigma 5 --shots 9600` submitted the 11-circuit batch to `ibm_fez` (156 qubits, cz; calibration 2026-09-04 14:37) on physical qubits [136, 143, 142], job `dadkfvl1ierc738l0ch0`, ISA depth 40 with 8 two-qubit gates, ~2 min in queue, 30 s of QPU on the Open Plan. Predicted before submission (shot-free Aer at the live calibration): E_B_info = +0.0405, z = 5.45; the preregistered prediction at the *sourced* ibm_cairo noise was +0.0401. Measured, tensored mitigation: **E_B_info = +0.04853 ± 0.00779, z = 6.23** (pull +1.08 against the ibm_cairo prediction, +0.90 against the noiseless ideal +0.0415), E_A = +0.84999 ± 0.00193 (dense +0.85385), E_B_ctrl = −0.01381 ± 0.00755 — teleportation proper is lost on this device, as the survival map said — and unmitigated E_B_info = +0.04715 ± 0.00765, so the signal is not a mitigation artifact. All four audits passed on the real counts (ledger defect −0.0031 at σ 0.0087; no-signaling χ²/crit 0.21 at the run's 5σ (0.48 at 3σ); E_A ≥ E_B; signal-vs-ideal one-sided pull +0.90). The record, its calibration snapshot, sha256 and persisted analysis live under `papers/hardware-as-noisy-field/data/real_device/` and are pinned by `TestRealDeviceRecord` — the repository's first real-device fixture. The claim is narrow by construction: one device, one day, the calibration snapshot as provenance. Rerun with the same command; the token is read from the gitignored `.env`. **Closed with a number on 2026-09-12: the two VQE rows are ground states — and one
published depth is retracted.** The (3, 4) and (4, 4) encodings that Layer 6 had recorded
as *convergence failures of the 3-layer ansatz* are now exhibited at roundoff (`s2b`,
`notebook_vqe_deep.py`, `data/s2b_vqe_deep.json`, `tests/test_hardware.py::TestVQEDeepRows`):
the failures were the ansatz's **expressibility ceiling**, not a stuck optimizer — 3-layer
fidelity ceilings 6.430476e−05 and 2.930502e−04, and the energy-VQE restarted *from* those
fidelity optima reproduces the old s2 rows to the third digit. **(3, 4) closes at 7 layers**
(48 parameters, infidelity 2.220446e−16, energy error ≤ 1.8e−15 on both evaluators,
covariance error ≤ 3.4e−16 — the s2 converged standard), **three layers below the
parameter-counting depth 10**; **(4, 4) at 31 layers** (256 parameters, infidelity
−4.441e−16). **RETRACTED (fix round 1):** the first version of this row published (3, 4) at
**8** layers and called both depths *minimal*. Continuing the archive's own budget-truncated
depth-7 optimum (its best start had ended on "the maximum number of function evaluations is
exceeded") reaches roundoff in seconds, so 8 was wrong by a layer; and no depth here is
proved minimal — `is_proved_minimal` is `false` by construction, the published depths are
**sufficient** depths, upper bounds over finitely many seeded starts. The discipline that
caught it is now standing: every depth is continued from its own optimum at a larger
Jacobian budget, the archive records whether each best start converged or exhausted its
budget, and a test fails if any depth below the published one is left exhausted. **(4, 4)
is the weak row and is labelled so:** depths 10, 11, 20 and 30 all ended on their Jacobian
budget, depths 13–15, 17–19, 21–23, 25–27 and 29 were never scanned, and the below probe at
L = 30 reached **4.100276e−12** — a factor 4 from the 1e−12 acceptance — with gradient norm
6.65e−08, i.e. **still descending**, so 30 is *not excluded and is probably sufficient*.
At (3, 4) the probe one layer below converged at 1.694e−11 (gradient norm 5.2e−16, two
independent starts) and does not reach 1e−12 — evidence, explicitly not a proof that 6
layers cannot hold the state. The round-2 fix was prose plus one guard: the Limitations
section had kept three factual (4, 4) lists that its own archive had superseded, and a new
test now parses those lists out of the MANIFEST and asserts them against the archive's
fields, so prose that drifts from the data fails instead of being read as fact. Downstream,
the 7-layer VQE state reproduces the (3, 4) QET and harvesting ledgers computed on the ED
state (max |diff| 1.148e−09, which is the θ-optimizer's own tolerance, the two states
agreeing to 2.2e−16). Also open in L6: N = 3 sites and d ≤ 4 only, whether (4, 4) closes
below 31 layers (the exact knobs and the two cached job files to delete are named in that
candidate's MANIFEST) and whether (3, 4) closes below 7; nothing here proves a *lower*
bound on depth — the honest object is the upper-bound-per-depth curve; the deep (3, 4)
state at 35 CX gates is far deeper than the 3-layer circuit the noise study assumed, a cost
nobody has budgeted; no spacelike harvesting, and the noise is a
homogeneous scaling of one calibration point; the rehearsal does not model crosstalk, the
spectator's idle decay, drift between calibration and execution, or the live queue.

**Cross-cutting, open.** Several of the thirteen candidates were produced on
concurrently-edited trees whose full suite carried failures in *other* candidates'
then-unfinished modules (their MANIFESTs record it); the integration run on the quiescent
tree is their admissibility record, and each should be re-run on a committed green build
before a number moves from `data/` into a manuscript. The two directions PLAN.md named that
the code had not walked are now walked: the Layer 2 MI/coherence pivot is a candidate of its
own (and fails at the device), and the Layer 3 hardware excursion has run — L6 on `ibm_fez`,
2026-09-04, the preregistered prediction held (E_B_info = +0.04853 ± 0.00779, z = 6.23). What
is left, after the 2026-09-12 pass worked all six of them, is this — stated with the honest
label first:

1. **The λ ≥ 2 magnitude of the L5 residual: BRACKETED, not resolved — not closed, and more of
   the same plane will not close it.** All 36 `plane2` rows landed on 2026-09-13. On the frozen
   15-row snapshot the residual bracketed at [−1.797 %, +4.415 %] (λ = 2) and
   [−34.349 %, +15.131 %] (λ = 4) against `E_H`; the complete record **widens** those 6.7-fold
   and 2.9-fold, to **[−34.382 %, +6.979 %]** and **[−127.079 %, +15.861 %]**, with 12 of 12
   settings failing the block's own 5 % gate at λ = 2 and 11 of 12 at λ = 4. The pass measured
   three obstacles: the MPO–ED error at λ = 4 *grows* with the Fock cutoff to +39.4 % at the
   cutoff the paper uses (the "3–12 %" bound is retracted), χ_op is non-monotone with a sign
   change while the **weighting** is the real limiter at λ = 4, and the 5 % reference-consistency
   gate certifies nothing about the λ = 4 error. What would close it is not more of this plane but
   an operator representation that does not spend bond dimension on high-occupation matrix
   elements, with an exact ED anchor at every quoted setting. The harvest is final (commit
   `f65292c`) and the pools have exited.
2. **The 2.53× against Fewster–Ford–Roman 2012: closed on 2026-09-23 as a reproduction.** Schiappacasse, Fewster & Ford, PRD 97, 025013 (2018)
   published the same value (C_shift = −0.0593338; ours 0.059762, +0.72 %), so no expert's
   reading of it is owed. The literature check that should have come first was done the same
   day: no Gaussian-sampled 4d sharp value in the whole series or the 252 papers citing it or
   the two foundational QEI references (19 searched in full text), with the one source known to hold 4d sharp computations unread —
   Dawson's 2006 York thesis, not online (record: `docs/LITERATURE_CHECK_2026-09-23.md`). It also found the candidate's diagonalization method to
   be prior art (Dawson 2006; Schiappacasse–Fewster–Ford 2018). The interacting infimum's novelty
   survived the same re-check (`docs/PRIORITY.md` (e)). A second round the same day covered the
   remaining candidates (record, part 2): Proposition B, Theorem F, the off-criticality failure
   map and harvest-then-teleport's general messages are known; the resource-theory core, the
   circuit-QED no-go, the harvested-state Gaussian result, pump-enhanced harvesting and the
   failure map's disorder half survived; the hardware row gained its IBM predecessors.
3. **The proofs behind the numerical continuum limits: half closed, and the halves are
   named.** *4d QEI constant:* the **lower** endpoint is now a computer-assisted theorem,
   `C_4d(Gaussian)/C_FE ≥ 0.457490709365`; the **upper** endpoint is still only
   Fewster–Eveson's R ≤ 1, so the certified enclosure is [0.4574907, 1] and the archived
   value remains, on its upper side, a numerical claim. *toy-Jacobson:* the **rate** is now
   an exact lattice identity with a derived closed form (and the fitted −2.27 exponent is
   shown to be a fit artefact of the grid), but **r_∞ = 0 is not proved** (ASSUMPTION A5),
   the closed form's O(1/L) remainder is measured and not bounded (A4), and the
   bond-centred floor that limits the site-floor derivation is underived.
4. **The non-commuting multimode amplification bound: still conjectured.** What Layer 5
   added is a *proved weaker* bound (Theorem M, constant 1/2) and a first-order rate
   (Theorem F); the conjecture itself was attacked to 1.000000000000017 — reached, and
   exceeded only by +1.7e−14 of ln-ρ roundoff, never broken — and so is neither proved
   nor refuted.
5. **New, from the same pass:** the 1+1 reflection-even sector of the toy-Jacobson leap is
   *partially* explained — its large residual is the massless scalar's IR zero mode and
   vanishes when that mode is projected out, but the **remaining** even-sector residual is
   unexplained and is 1.9×–659× the odd sector's per unit variation (the earlier
   "better than the odd sector" comparison is retracted); and in the 2+1 toy the conformal
   improvement's entangling-circle contact term is **dropped, not derived**, with ξ fitted
   on 15 cells rather than derived.
6. **Closed this pass, with numbers:** the (3, 4) and (4, 4) qubit-encoded ground states
   (7 and 31 layers, the published 8 retracted), and τ₀ ≥ 1 in Layer 5 — closed
   *negatively*, χ_op being measured not to be the missing knob.
