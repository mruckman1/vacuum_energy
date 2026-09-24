# Priority ledger — what is new, what is reproduced, what is conditional

One row per result candidate under `papers/`, against the literature as actually fetched
on 2026-09-04, with the 2026-09-12 additions (a′) and (d′) marked as what they are: one of
them rests on a literature check that has **not** been run. The brief is an outside senior reviewer's cold read of `README.md`; the
verdicts below are the program's, checked citation by citation. Classifications:

- **NEW** — no prior statement of the result found; the candidate's own status still governs.
- **REPRODUCTION-AT-CERTIFIED-PRECISION** — a published formula or number, confirmed on the
  lattice with stated convergence and the precision ladder; useful, not a discovery.
- **METHODS** — a technique or an observation about technique that transfers.
- **CONDITIONAL** — a statement whose force rests on one mapping or one input that an
  outside expert has not checked.
- **UNDER RE-EXAMINATION** — a sibling pass is re-deriving the candidate; no verdict here
  (no row carries it now: the re-examination of `toy-jacobson` closed on 2026-09-04).
- **POSSIBLY KNOWN** — the content is folklore or sits in a literature this search could not
  fully reach; held as a rederivation with citations until an expert says otherwise.

Nothing in this file changes a number. It changes what the numbers are called.

---

## The literature check, (a)–(e) — plus (a′) and (d′), added 2026-09-12

**(a) The artanh(d) bound and the rate optimum (Layer 5, `vacuum/floquet/bounds.py`).**
*Equality case — classical.* The two-level drive switching at the quarter periods gains
ω_max/ω_min = √((1+d)/(1−d)) per half period, i.e. artanh(d) in ln r per zero of x:
Meissner, Schweiz. Bauzeitung **72**, 95 (1918); textbook in Arnold, *Mathematical Methods
of Classical Mechanics*, §25 (the swing with piecewise-constant frequency).
*Bang-bang optimality of parametric pumping with a bounded frequency — established.*
Lavrovskii–Formal'skii, "Optimal control of the pumping and damping of a swing", J. Appl.
Math. Mech. **57**, 311 (1993); Piccoli–Kulkarni, IEEE Control Syst. Mag. **25**(4), 48
(2005); Andresen, Hoffmann, Nulton, Tsirlin, Salamon, "Optimal control of the parametric
oscillator", Eur. J. Phys. **32**, 827 (2011), doi:10.1088/0143-0807/32/3/018 — minimum
time to a target energy with ω_min ≤ ω ≤ ω_max, bang-bang, switching instants from
transcendental conditions; Salamon–Hoffmann–Rezek–Kosloff, PCCP **11**, 1027 (2009);
Stefanatos–Ruths–Li, PRA **82**, 063422 (2010); Stefanatos, IEEE TAC **62**, 4290 (2017);
for the spectral-radius objective on bilinear systems, Hochma–Margaliot, arXiv:1407.3437
(necessary conditions, bang-bang and singular arcs); and the optimal-control route to
Hill's-equation stability criteria with a bounded coefficient (Lyapunov, Neigauz–Lidskii,
Borg, Krein) is Zu–Li, Appl. Math. J. Chin. Univ. **33**, 253 (2018),
doi:10.1007/s11766-018-3602-0.
*Verdict.* Theorem A's content — the extremal drive is bang-bang and its gain is
artanh(d) per zero — is folklore of the Meissner–Hill and swing-control literature. What
this search did **not** find stated is the inequality for arbitrary *measurable* s in its
per-zero (Prüfer-winding) form with the a.e. equality case, i.e. the change of clock that
makes the Pontryagin singular-arc question moot. Theorem B is the growth-rate
(Lyapunov-exponent) form of the minimum-time problem the papers above solve; the
closed-form root equation for λ*(d) and the quantified sub-optimality of the quarter-period
schedule were not found in that form. **Classification: POSSIBLY KNOWN — a rederivation
with citations.** The 1960s–70s Soviet parametric-control literature (Chernousko,
Akulenko, Sokolov, *Control of Oscillations*) was not reachable in this pass and is the
most likely place for the per-zero statement.

**(a′) The proved multimode bound for non-commuting patterns (Layer 5,
`vacuum/floquet/bounds_multimode.py`, Theorem M; 2026-09-12).** For
K(t) = K₀ + d s(t) P the pass proved
`ln max|eig S_F|/T ≤ (d/2)‖K₀^{−1/4} P K₀^{−1/4}‖ ≤ ω_max d_eff/2` for every measurable
|s| ≤ 1, by a Grönwall estimate on the number norm in the canonical complex coordinates
(the coupling splits into a number-preserving beam-splitter part and a squeezing part
bounded by ‖C‖). *Prior art actually checked:* none in the 2026-09-12 pass. On 2026-09-23 INSPIRE and web searches for growth-rate bounds on multimode parametric drives found no such statement; the nearest paper read, Arenz et al., "Amplification of quadratic Hamiltonians", Quantum 4, 271 (2020), bounds nothing of this form. The ingredients are standard —
Bogoliubov/complex-coordinate splitting and Grönwall bounds on ⟨N⟩ are routine in the
parametric-amplifier and Gaussian-dynamics literature, and Krein–Gelfand–Lidskii strong
stability theory (cited in the draft) covers stability criteria for linear Hamiltonian
systems with periodic coefficients — but no search for this statement as a *growth-rate
bound for non-commuting multimode parametric drives* was carried out in this pass.
**Classification: POSSIBLY KNOWN — the search was done on 2026-09-23 and found nothing, and the technique is standard, so the classification stands**, on
the same footing as (a) and for the same reason: the technique is folklore even where the
statement may not be. What is *not* in doubt is the honest status of the thing it was
meant to settle: the conjecture `ln max|eig S_F|/T ≤ ω_max λ*(d_eff)` is **neither proved
nor refuted**, and Theorem M is at most π/2 looser than it at one mode. Corollary M1's
endpoints (λ*(d)/(d/2) ∈ [2/π, 0.68910]) and the non-attainment of Theorem M for N ≥ 2 are
**measured, not proved**, and Theorem F's rotating-wave step is labelled ASSUMPTION, and Theorem F itself is classical: the sum-frequency (combination) resonance gain δ/(4√(ω₁ω₂)) is Eq. (15) of Svidzinsky, Zhang, Wang, Wang & Scully, arXiv:1407.3727, in a theory going back to Yakubovich–Starzhinskii.

**(b) Casini–Huerta's n-interval formula and its lattice confirmations (Layer 4).**
Casini–Huerta, "Reduced density matrix and internal dynamics for multicomponent
regions", CQG **26**, 185005 (2009), arXiv:0903.5284: the modular Hamiltonian of the
massless Dirac field on *any* number of disjoint intervals, local term plus a bilocal term
coupling each point to one conjugate point in every other interval; the "screening" of an
outer pair by a middle interval is contained in the weight 2π/((x − x_c) z′(x_c)), since
z(x) carries every endpoint. Lattice confirmations: Eisler–Tonni–Peschel, "Local and
non-local properties of the entanglement Hamiltonian for two disjoint intervals", J. Stat.
Mech. (2022) 083101, arXiv:2204.03966 (two intervals, general filling, arbitrary
intervals, the long-range hopping recovered from the lattice); Arias–Blanco–Casini–Huerta,
"Local temperatures and local terms in modular Hamiltonians", PRD **95**, 065005 (2017),
arXiv:1611.08517 (the *local* term of massive scalar and Dirac fields checked numerically
on one and two intervals; the nonlocal terms not analysed); Eisler–Peschel, J. Phys. A
**50**, 284003 (2017) (single interval). No lattice check at n = 3, 4 was found.
*Verdict.* **REPRODUCTION-AT-CERTIFIED-PRECISION** of CH 2009; the n = 3, 4 lattice
numerics, the 1/L² law with its extrapolation, and the asymptotic ξ_h = ξ are
measurements not previously published, of formulas and of a decay one expects.

**(c) Peschel's corner-transfer-matrix results for the gapped chain, staggered-mass case.**
Peschel–Kaulke–Legeza, Ann. Phys. (Leipzig) **8**, 153 (1999), arXiv:cond-mat/9810174
(transverse Ising and XXZ, homogeneous: ε_j = (2j+1)ε or 2jε, ε = πK(k′)/K(k), k = λ or
1/λ); Eisler–Di Giulio–Tonni–Peschel, J. Stat. Mech. (2020) 103102, arXiv:2007.01804,
Eqs. (22), (40)–(41) (the *dimerized hopping* chain: prefactor 2I(k′), k = (1−δ)/(1+δ),
levels 2lε); and — decisive — **Eisler, "On the Bisognano–Wichmann entanglement Hamiltonian
of nonrelativistic fermions", J. Stat. Mech. (2025) 013101, arXiv:2410.16433, Sec. V,
Eqs. (64)–(72)**: for the hopping chain with hopping 1/2 and staggered potential μ(−1)ⁿ —
exactly `vacuum.modular.lattice`'s chain — the correlation matrix commutes with the
tridiagonal boost T (t_m = −m/2, d_m = (−1)^m μ (m − ½)) and the half-infinite entanglement
Hamiltonian is H = 4κK(κ′)T with κ = 1/√(1 + μ²), spectrum 2πK(κ′)/K(κ)·(2ℓ ∓ ½). In the
candidate's units that is 2π s(m) = 4kI(k′), k(m) = 1/√(1 + m²) = sech(1/ξ), ε = πI(k′)/I(k)
with odd multiples — the formula the candidate conjectured from the CTM structure and
confirmed to 1e-14 without knowing the derivation existed.
*Verdict.* **REPRODUCTION-AT-CERTIFIED-PRECISION** of Eisler 2025. The "open problem"
of deriving k(m) recorded in `vacuum/modular/ctm.py` is closed by that paper; the module
docstring now cites it, states the Eisler-2025 → module dictionary once, and labels the closed
forms as his (the `ctm_` prefixes are kept for API stability).

**(d) A prior numerical value of a 4d QEI constant for Gaussian sampling.**
Fewster–Eveson, PRD **58**, 084010 (1998), arXiv:gr-qc/9805024: the 4d bound for smooth
sampling functions (the C_FE the candidate normalises to), explicitly *not* sharp (3/2 off
Flanagan's bound in 2d). Fewster–Ford–Roman, "Probability distributions for quantum stress
tensors in four dimensions", PRD **85**, 125038 (2012), arXiv:1204.3570: for **Lorentzian**
sampling f(t) = τ/[π(t² + τ²)] of the massless scalar, the lower bound of the probability
distribution of the smeared energy density is estimated from 65 moments as
y∞ = 0.02361 ± 1e−5 in x = (4πτ²)²A, with the explicit sentence "these are also estimates
of the optimal quantum inequality bounds for each field" (−0.0472 for the electromagnetic
field). No Gaussian-sampled 4d value was found (Fewster–Ford–Roman, arXiv:1409.7516,
repeats the Lorentzian; Fewster, Lectures, arXiv:1208.5399, states that "optimal bounds are
lacking in general" — a 4d sharp bound has no known closed form; checked 2026-09-23: Schiappacasse, Fewster & Ford, PRD 97, 025013 (2018) diagonalizes the Lorentzian and
compactly supported weights, no Gaussian; Fewster & Ford, PRD 101, 025006 (2020), gives
closed-form Gaussian bounds only in 2d; PRD 103, 125014 (2021) uses compactly supported
sampling in a cavity).
*Verdict.* The candidate's C_4d/C_FE = 0.45749 ± 0.00010 for the Gaussian family is
**NEW** for its family — **re-checked 2026-09-23 to the compared authors' own follow-ups**: the
whole Fewster–Ford–Roman / Schiappacasse–Fewster–Ford series (2012, 2014, 2015, 2018, 2020, 2021,
2022; the 2010 and 2024 papers are two-dimensional), the 252 papers in INSPIRE citing it or
Fewster–Eveson 1998 or Fewster's 2012 lectures (titles screened, 19 searched in full text), and
INSPIRE full-text searches. No Gaussian-sampled 4d sharp value appears anywhere, and Kontou et al.
(2024), arXiv:2405.05963, state that none of the four-dimensional QEI bounds has been proven
sharp. The one source known to hold 4d sharp computations could not be read: S. P. Dawson's 2006 University of York PhD thesis, "Bounds on negative energy
densities in quantum field theories on flat and curved space-times", which computed 4d sharp
bounds numerically; it is not in the university's online archive, and Schiappacasse–Fewster–Ford
describe its sampling function as a squared Lorentzian, not a Gaussian. NEW therefore holds to the
standard now required, with that thesis unread — obtained by a different route (variational, extremal state exhibited, lattice plus lattice-free
continuum solve) from the prior 4d values.
**The Lorentzian weight: REPRODUCTION, corrected 2026-09-23 from "DISAGREES".** The acceptance
test run on 2026-09-04 gives, in FFR's units (x = (4πτ²)²ρ; their FE anchor 27/128 reproduced
exactly), x₀ = 0.05976 against their y∞ = 0.02361 ± 1e−5, a factor 2.53, with every rigorous
statement of theirs respected (their Eq. (64) y∞ ≤ x₀; Fewster–Eveson; Ford–Roman). This file
called that "a claim against a published number by the field's leading authors … the single
item most in need of an expert's eyes". It had been published: **Schiappacasse, Fewster & Ford, PRD 97, 025013 (2018),
arXiv:1711.09477** — two of the three 2012 authors — diagonalized the same operator for the same
weight in a finite cavity and found C_shift = −0.0593338, setting it beside the 2012 value
themselves; ours reproduces it to +0.72 % (`tests/test_qei.py::test_ffr2012_lorentzian_acceptance`
(vi)). What this candidate adds to their paper is **METHODS** at most: FFR's own number
reproduced by their method on our kernel (their 65 exact moments, all 44 Table I entries, Table II
to 4.9e−12, y∞ = 0.0236173 against 0.0236175), and the demonstration that a finite-moment
reconstruction is blind to the lower support (a point mass at −0.0598 of weight 10 % moves the
65-moment edge only to 0.0252) — the explanation of a factor their 2018 paper reports without
explaining. Three routes agree in FFR units (continuum 0.059762 ± 5e−7, lattice 0.059684 with its
own box law, exhibited state 0.059762), so finite-box IR bias is excluded. **How it was missed:**
the search behind the NEW classification looked for other groups' work and for Gaussian values,
never for the 2012 authors' own follow-ups.

**(d′) A certified (interval-arithmetic) bound on a 4d QEI constant (Layer 1, 2026-09-12).**
The 2026-09-12 pass turned the *lower* half of the Gaussian 4d constant into a
computer-assisted theorem: `C_4d(Gaussian)/C_FE ≥ 0.457490709365` from an explicitly
exhibited 64-mode Gaussian state, its state property verified by an interval Cholesky,
evaluated in `mpmath.iv` with a proved Γ-series tail bound and **no quadrature in ω**.
*Prior art.* Computer-assisted proofs with interval arithmetic are an established genre
(Tucker's Lorenz attractor, Ann. Math. 2002; Hales' Kepler conjecture / Flyspeck; Gutierrez
et al. and the Rigorous Numerics school), and *variational upper* bounds on a ground-state
energy from an exhibited trial state are the oldest tool in the subject — what is new here
is only the application: the first **rigorous** two-sided statement about a sharp QEI
constant in 4d, one endpoint proved and one cited. No prior certified QEI constant was
found; Fewster's lectures (arXiv:1208.5399) state that the 4d sharp bound has no closed
form. **Classification: NEW for the endpoint, METHODS for the route** — and the honest
qualifier belongs in the same sentence: **the enclosure is one-sided in substance**
([0.4574907, 1], width 0.5425), the upper endpoint being Fewster–Eveson's `R ≤ 1`, *cited
not computed*. Quoting this as "a certified enclosure of width 1e−5" would misrepresent it.
The obstruction to the other half is structural and is stated in the draft:
‖K^{−1/2}K_−K^{−1/2}‖ = 1 and Tr(K_−K^{−1}K_−) = ∞, so every perturbative lower bound on
E_Q diverges and the exact dual certificate needs certified operator square roots on the
half-line. A non-rigorous Powers–Størmer evaluation puts a certified upper endpoint near
0.576; it is archived under a key that names it an ESTIMATE — NOT CERTIFIED and must not be
quoted.

**(e) Prior numerical exact QEI infima for interacting lattice theories.**
None found. The interacting QEI literature is continuum: Fewster–Hollands, Rev. Math.
Phys. **17**, 577 (2005) (2d CFT); Bostelmann–Cadamuro–Fewster, PRD **88**, 025019 (2013),
arXiv:1304.7682 (massive Ising); Bostelmann–Cadamuro–Mandrysch, Ann. Henri Poincaré
(2023) (several species, bound states); Mandrysch, "Numerical results on quantum energy
inequalities in integrable models at the two-particle level", PRD **109**, 085022 (2024),
arXiv:2312.14960 (numerical optimal bounds at the one- and two-particle level, sinh-Gordon,
S-matrix bootstrap — not an infimum over all states); Fröb–Cadamuro, arXiv:2212.07377
(Sine-Gordon, an absolute bound, not sharp).
*Verdict.* An infimum over *all* states of an interacting lattice theory, with the
extremal state exhibited, by operator-space TEBD of the smeared-energy MPO plus DMRG on
its lowest eigenvalue, is **NEW** as an object and as a method. **Re-checked 2026-09-23 to the
compared authors' own follow-ups:** the 2023–2026 papers of Mandrysch, Cadamuro, Bostelmann,
Fröb, Fewster and Hollands contain no interacting infimum over all states (their later work is on
modular theory, relative entropy and thermal L^p inequalities); INSPIRE full-text searches for
energy inequalities with DMRG, tensor networks or matrix-product states, and for negative energy
densities with DMRG, return nothing of the kind; the nearest new item, Ikeda, arXiv:2604.09973
(2026), studies energy teleportation in the sine-Gordon/Thirring model on a lattice, not a QEI
infimum. Unlike the diagonalization route of (d), the MPO route is needed because the smeared
energy of λφ⁴ is not quadratic. The verdict stands. Its numbers were diagnosed on
2026-09-04 (`interacting-vacua-first-numbers`, row below): the residual beyond any quadratic
reference is a first-order quartic tax, resolved at λ ≤ 1 and BRACKETED at λ ≥ 2.
**2026-09-12 — the method's own cross-validation is now the limiting factor at λ = 4, and
the candidate says so.** An exact-diagonalisation anchor at the sweep's *own* Fock cutoff
(L = 5, n_max 8) puts the MPO route **+39.4 %** from the exact answer at λ = 4 (+42.3 % at
the convergence plane's knobs), against 0.3–1.4 % at λ = 0, and the error **grows** with
the cutoff rather than being bounded — an earlier "3–12 % wherever an exact answer exists"
is retracted in the candidate's no-longer-claims list. So the object is still NEW; the
λ ≥ 2 *numbers* are not merely unresolved but are produced by a route not yet
cross-validated at that coupling and cutoff, and none is quoted as a value. **2026-09-13
adds the bracket.** The anchored (n_max, χ_op, **weight**) plane, 15 of 36 rows landed
with their own λ = 0 controls, frozen by tag as `unit2_snapshot` —
gives the control-normalized residual as **[−1.797 %, +4.415 %]** against `E_H` at λ = 2
and **[−34.349 %, +15.131 %]** at λ = 4 (**[+0.301 %, +6.385 %]** and
**[−28.541 %, +18.800 %]** against K_eff), a systematic of **32.3** and **238.6** λ = 0
bands carried by the occupation weighting. The ED anchor clears one setting per λ and they
are different settings, so item 4 is **BRACKETED, not resolved**. What does survive: the
sign, positive against K_eff at 5 of 5 settings of that snapshot at λ = 2 and 4 of 5 at
λ = 4 (**on the live tree, 5 of 6 and 4 of 6** — the extra λ = 2 negative fails every gate
the harvest applies).

---

## The thirteen candidates

| candidate | claim as currently stated (README, one line) | classification | nearest prior work | what the candidate adds beyond it | what an expert would need to accept it |
|---|---|---|---|---|---|
| `circuit-qed-noise-thresholds` | the device point of the Waterloo proposal (Teixidó-Bonfill et al., PRA 113, 043732 (2026), Table I scenario 2) is NO-GO at every floor; with the coupling derived from the circuit Hamiltonian and the map re-run under the proposal's own derivative operator, the device's pure dephasing alone forbids resolvable harvesting by 100–200× in rate | **CONDITIONAL** (on the mapping in `vacuum/detectors/circuit_mapping.py`); the `'xx'` surfaces are a model claim | the proposal itself and its noise discussion; Pozas-Kerstjens–Martín-Martínez, PRD 92, 064042 (2015); Simidzija–Martín-Martínez, PRD 96, 065008 (2017); Tjoa–Martín-Martínez, PRD 104, 125005 (2021) (the vacuum/communication split); Janzen et al. 2023 (the measured α) | a nonperturbative Gaussian-channel noise map at the device point with the coupling *derived*, a UV/IR bracket through the device's 50 GHz, and a negative device statement — physically plausible (dephasing kills a two-orders-of-magnitude-smaller signal) and of interest to exactly one group | a circuit-QED person reading `circuit_mapping.py`: the λ_osc derivation, the transcription of the measured Γ_φ into the dephasing channel, and the `'xp'` operator; the proposal's authors agreeing that scenario 2 is the device point. An internal blind re-derivation (`docs/coupling_rederivation/`, written without sight of the module) agrees on every equation and lands at the same λ_osc = 0.141 — it narrows the mapping risk but is not the outside reader |
| `derivative-coupling-noise-survival` | the derivative coupling's entanglement stays ≥ 99.9 % field-correlation (\|M_vac\|/\|M\|) on every point where it survives noise | **METHODS** (the Tjoa–MM split carried through Gaussian noise channels); model claim | Tjoa–Martín-Martínez, PRD 104, 125005 (2021); the derivative coupling of the proposal; the sibling's device inputs | the split evaluated nonperturbatively under temperature, loss and dephasing; the observation that noise kills negativity before it moves the correlation share; the 1e−4 nats floor is a stated model choice, not derived | a harvesting person checking that the second-order estimator's *fractions* remain meaningful at λ = 0.1 under noise; nothing device-level is claimed |
| `mi-coherence-harvesting` | MI, discord and coherence harvesting also fail at the device point, each against a floor derived for it; MI's own tomography floor is 1.9× the negativity floor that had been transferred to it | **METHODS** (the floor observation transfers to any harvesting proposal that reports mutual information) | Gaussian discord: Adesso–Datta, PRL 105, 030501 (2010); Giorda–Paris, PRL 105, 020503 (2010); the MI/discord harvesting literature; the sibling map | per-measure Monte-Carlo tomography floors at the device's shot budget; the 1.9× observation (plug-in MI estimators are biased upward at low counts, so MI's floor is *higher* than negativity's, not lower) | a quantum-estimation person checking the Monte-Carlo floor construction (estimator bias, the tomography model) before the 1.9× is quoted as a general rule |
| `qei-2d-sharp-constant` | the sharp 2d constant 1/(6π) recovered at 1e−7 with the extremal state; the 4d Gaussian constant C_4d/C_FE = 0.45749 ± 0.00010 from a derived ρ-form and a lattice-free continuum solve; Gaussians not optimal in 4d (factor 2.16 across six families); **and (2026-09-12) a certified lower endpoint, C_4d(Gaussian)/C_FE ≥ 0.457490709365, proved by interval arithmetic on an exhibited 64-mode state** | 2d: **REPRODUCTION-AT-CERTIFIED-PRECISION**; 4d Gaussian: **NEW** (for its family; re-checked 2026-09-23 to the compared authors' own follow-ups; the one source known to hold 4d sharp computations unread — Dawson's 2006 York thesis, not online); the certified lower endpoint **NEW** as an interval-certified bound for the Gaussian weight, on the same footing, **METHODS** for the route; **4d Lorentzian: REPRODUCTION** of Schiappacasse–Fewster–Ford 2018 (corrected 2026-09-23) (see (d′) above) — with the qualifier that the enclosure is one-sided in substance, [0.4574907, 1], its upper endpoint being Fewster–Eveson's cited R ≤ 1 | Flanagan, PRD 56, 4922 (1997) (2d sharp); Fewster–Eveson, PRD 58, 084010 (1998) (4d, not sharp); Fewster–Ford–Roman, PRD 85, 125038 (2012) (Lorentzian 4d sharp estimate 0.02361 ± 1e−5 from 65 moments); Schiappacasse–Fewster–Ford, PRD 97, 025013 (2018) (Lorentzian 4d lowest eigenvalue −0.0593338 by cavity diagonalization); S. P. Dawson, PhD thesis, York (2006) (4d sharp bounds by Colpa diagonalization, torus, squared Lorentzian; not online); Fewster, arXiv:1208.5399 | a lattice pull-back formulation and a lattice-free continuum solve of the diagonalization route to the sharp constant — the **route itself is prior art** (Dawson's 2006 York thesis applied Colpa's diagonalization of the quadratic boson Hamiltonian to QI bounds in 4d; Schiappacasse–Fewster–Ford 2018 identify the lowest eigenvalue of the diagonalized operator as the QEI bound; found 2026-09-23); a value with a measurement-type uncertainty for the Gaussian family; the f-dependence of "the 4d sharp constant" across six families and the lower bound 0.6110 × C_FE on the f-independent sup (two earlier points of that dependence exist: Schiappacasse–Fewster–Ford 2018 find the Fewster–Eveson-type bound 4.6× the sharp one for a compactly supported weight, and report Dawson's ≈ 3× for a squared Lorentzian) | a QEI person (Fewster lineage) reading the pulled-back operator against the continuum variational problem — the massless IR handling and the W^{2,2} regularity edge in particular — and, for the Gaussian row, the literature check that should have preceded it. The Lorentzian-weight comparison is **closed as a REPRODUCTION**: our x₀ = 0.059762 against Schiappacasse–Fewster–Ford 2018's 0.0593338 (+0.72 %); it was recorded here from 2026-09-04 to 2026-09-23 as a disagreement with a published number flagged for expert review |
| `multi-interval-modular-hamiltonians` | Casini–Huerta reproduced on the lattice; n = 3, 4 intervals agree exactly in the continuum limit (+8e−05); the nonlocality decays with the correlation length itself (ξ_h/ξ = 0.9985 ± 0.0257); the gapped edge slope has the closed form s(m) = (2/π) k I(k′), k = sech(1/ξ) | **REPRODUCTION-AT-CERTIFIED-PRECISION** | Casini–Huerta, CQG 26, 185005 (2009) (the n-interval formula, screening included); Eisler–Tonni–Peschel, J. Stat. Mech. (2022) 083101 (the lattice dictionary, n = 2); Arias–Blanco–Casini–Huerta, PRD 95, 065005 (2017) (massive local term, n = 1, 2); Eisler–Di Giulio–Tonni–Peschel, J. Stat. Mech. (2020) 103102 (the gapped triangle, dimerized chain); **Eisler, J. Stat. Mech. (2025) 013101, Sec. V, Eqs. (64)–(72)** (H = 4κK(κ′)T, κ = 1/√(1+μ²) for exactly this chain — the candidate's s(m), k(m)) | lattice numerics at n = 3, 4 with their onset in the gap, the 1/L² law and its +8e−05 extrapolation; the asymptotic ξ_h = ξ from local decay rates (an expected value, not previously measured); a zero-parameter 1e−14 confirmation of Eisler 2025 on two independent observables; the finite-segment crossover from the half-infinite line to the triangle | someone from the Casini–Huerta / Peschel–Eisler lineage reading the dictionary D1–D6 and the n ≥ 3 tables; the manuscript must present itself as a reproduction and cite Eisler 2025 in its first paragraph; `vacuum/modular/ctm.py`'s docstring now cites Eisler 2025 with the dictionary stated once |
| `geometry-from-entanglement-failure-map` | metric reconstruction from MI distances fails with structure off criticality (ξ_× ≈ 0.3 L_probe) and into noise under disorder — a failure boundary in the (ξ/L, W) plane | **METHODS** (a quantified negative map); model claim — re-checked 2026-09-23: the off-criticality half is known (Leighton-Trudel, arXiv:2507.09749 (2025): a mutual-information distance is a metric at criticality and not in gapped phases); the disorder half was not found | Swingle, PRD 86, 065007 (2012) (MERA ↔ AdS); Van Raamsdonk, GRG 42, 2323 (2010); Cao–Carroll–Michalakis, PRD 95, 024031 (2017) (space from MI); Evenbly–Vidal, J. Stat. Phys. 145, 891 (2011) (MERA geometry) | the two failure axes distinguished (structure vs noise) with a curvature-radius trend, on a free-fermion chain, with the 1e−12 precision artefact separated from physics | a tensor-network / holography person judging the "degrades at Swingle's factorization scale" reading and the choice of Gromov-δ and stress metrics; a MERA person on the χ = 8 ternary construction |
| `time-modulated-vacua` | the artanh(d) amplification bound proved for all measurable drives (Theorem A) and the rate optimum (Theorem B: the quarter-period bang-bang attains the per-cycle bound but is not rate-optimal); **(2026-09-12) Theorem M, a proved bound for non-commuting patterns with the constant 1/2, and Theorem F at first order — the conjecture `≤ ω_max λ*(d_eff)` itself neither proved nor refuted**; loss thresholds Q_th = π/(2 ln λ_max) mapped to platforms; pump-enhanced harvesting ≈ 10–33× at N = 64, dominated by pre-existing pairs | Theorems A/B: **POSSIBLY KNOWN** (rederivation with citations); Theorem M: **POSSIBLY KNOWN** — searched 2026-09-23, not found as stated, technique standard (see (a′) above); Theorem F: **REPRODUCTION** of classical combination-resonance theory (Svidzinsky et al., arXiv:1407.3727, Eq. (15); Yakubovich–Starzhinskii); pump-enhanced harvesting not found (model claim); the unrefuted conjecture is a **conjecture**, not a result; loss threshold: model claim; composite: model claim | see (a) above — Meissner 1918; Arnold §25; Lavrovskii–Formal'skii 1993; Piccoli–Kulkarni 2005; Andresen et al. 2011; Stefanatos–Ruths–Li 2010; Hochma–Margaliot 2014; Zu–Li 2018. Thresholds: Wilson et al., Nature 479, 376 (2011); Wang et al. 2024 (transcribed). Composite: Tjoa–MM 2021 split | the per-zero measurable-s form of the bound with its a.e. equality case; the closed-form λ*(d) root equation and the quantified rate sub-optimality of the quarter-period drive (0.10 % at d = 0.1 … 44.1 % at 0.95); an adversarial battery on exact piecewise monodromies; the multimode block-diagonal statement; the vacuum/communication split of a Floquet-prepared harvest with its finite-size control | a control theorist who knows the parametric-control literature reading `bounds.py` against Lavrovskii–Formal'skii and Andresen et al. (does the per-zero statement already exist?); a DCE experimentalist for the platform table; a harvesting person for the split's use at λ_max = 0.4 |
| `differentiable-vacuum` | admissible optimized waveforms gain 4.22× at 30 mK under a global search and revive the dead dephasing floor under equal work and a signaling bound; a second twin cannot overturn the NO-GO | **METHODS** (a differentiable Gaussian stack with Daleckii–Krein JVPs and admissibility constraints in the loop); model claim | switching-function optimisation for harvesting (Pozas-Kerstjens–MM 2015 and successors); gradient-based optimal control (GRAPE, Krotov); differentiable Gaussian simulators | equal-work, noise-floor and communication-fraction constraints imposed *inside* the optimisation, with a plain-numpy audited round trip of every optimum | a harvesting person judging whether the communication-fraction bound is the right admissibility criterion; an optimisation person on the multistart/global-search claims |
| `toy-jacobson` | with the local weight free (nonnegative smooth families or per-site, LP over odd-mode squeezes), δS = δ⟨K_loc⟩ singles out the CHM parabola for the massless wave chain (found, not assumed: overlap 1.0000, r_free = 4.1e−5 / 1.2e−5 / 2.5e−6 at L = 8/16/32) and Lifshitz is distinguished by the complexity of the admissible local weight, not its existence (CHM is the 3-mode parabola for the wave chain; Lifshitz needs ≥ 11–12 modes of a bumpy profile — 0.09 at 8 modes; below 1e−2 by 11–13 modes and below 1e−3 by 12–14, z-dependent; 5e−4 only by 20 Chebyshev modes — so the toy tests parsimony); it does not see the velocity — the earlier "recovers v² = 0.980" is withdrawn (a unit under a free weight); **(2026-09-12) the lattice→continuum rate is now derived** (an exact lattice identity for trapezoid-minus-midpoint weight placement, closed form A(κ)/L², A(0⁺) = −3/2, reproducing the measured residual to 6.3e−12 — so the fitted −2.27 exponent is a grid artefact and the true exponent is −2), **the toy runs in 2+1**, and there **the conformal improvement's coefficient is measured**, ξ = 0.1278 ± 0.0040 against (d−2)/(4(d−1)) = 1/8 | **METHODS / narrow result** (re-examined and de-circularized 2026-09-04 after the reviewer's objection that the CHM weight was an input; a model claim about 1+1 lattice chains). The 2+1 ξ measurement is **NEW for the toy and a REPRODUCTION of a known coefficient**: ξ = (d−2)/(4(d−1)) is textbook (Callan–Coleman–Jackiw improvement), so what is new is that entanglement equilibrium *measures* it rather than assuming it (re-checked 2026-09-23: the lattice agreement of a sphere's entanglement Hamiltonian with the conformal prediction in two and three spatial dimensions is Javerzat–Tonni, JHEP 02 (2022) 086; the boundary term the toy drops is derived in Herzog–Nishioka, JHEP 12 (2016) 138; the 1+1 CHM weight on the lattice is Di Giulio–Tonni, J. Stat. Mech. (2020) 033102) — the first place in this candidate where a continuum coefficient is measured rather than checked; the shape is identified too (3.1 % cell-to-cell scatter for ∇²β against 7.7–32.9 % for three wrong shapes). The rate derivation is **METHODS**, and it is explicitly *not* a proof that the lattice residual converges to the continuum one (ASSUMPTION A5) | Jacobson, PRL 116, 201101 (2016); Casini–Huerta–Myers, JHEP 05 (2011) 036; Blanco–Casini–Hung–Myers, JHEP 08 (2013) 060 (the first law); Callan–Coleman–Jackiw, Ann. Phys. 59, 42 (1970) (the improved stress tensor); Casini–Huerta, J. Phys. A 42 (2009) 504007 (the 1+1 massless zero mode the even sector runs into) | the free-weight test itself — an LP over the lattice's local energy pieces that finds CHM rather than assumes it, with the variation family's rank (5–6) reported next to the parameter count, which is why a per-site weight has no teeth; the Lifshitz parsimony statement | a modular-Hamiltonian person on whether "nonnegative smooth local weight" is the right class (a signed weight is not excluded: the signed Lifshitz minimum is 1e−3 with a 10–84× negative-dipped weight) and on the even sector, still inconsistent after zero-mode projection |
| `resource-theory-vacuum-manipulation` | the QET potential W_→ is a ledger monotone under energy-non-signalling instruments + classical communication + Bob's CPTP maps; the surplus is not; Theorem 2 characterises the free class; the tight single-site bound is E_B = Tr T + ‖T‖_* | **NEW** for the resource-theory core (the ledger monotone, the non-monotone surplus, two-way paying Bob nothing, the convex-position characterization, the saturated one-shot bound) — re-checked 2026-09-23; while drafting the paper the same day, three more credits were found: Prop. A's effective Hamiltonian is Xie–Sajjan–Kais's "local effective Hamiltonian" (Entropy 27, 1147 (2025)); Itoh et al. (PRE 113, 024108 (2026)) also connect feedback extraction to daemonic ergotropy; Fan et al.'s "QET passive" states (PRA 110, 052424) may be the free states, unread in full; **REPRODUCTION** for Proposition B's single-qubit closed form (Salvia–De Palma–Giovannetti, PRA 107, 012405 (2023), Sec. III.A); the 12.4× gap of a local channel over a local rotation is a known phenomenon (Alhambra et al., PRL 123, 190601 (2019); Salvia et al.) with new numbers; model claim | Hotta, Phys. Lett. A 372, 5671 (2008) and arXiv:1101.3954 (QET); Frey–Funo–Hotta, PRE 90, 012127 (2014) and Alhambra et al., PRL 123, 190601 (2019) (strong local passivity); Francica et al., npj QI 3, 12 (2017) (daemonic ergotropy); Åberg, Nat. Commun. 4, 1925 (2013) and Horodecki–Oppenheim, Nat. Commun. 4, 2059 (2013) (one-shot work) | the monotone statement with the surplus counter-example; the free-instrument breaking of the closed form by 6.1509×; the exact single-site bound against an entropic one | a quantum-thermodynamics resource-theory person reading Theorems 1–2 of `vacuum/qet/resource.py`, in particular whether "energy-non-signalling instruments" is the natural free class against the SLP literature's operation classes |
| `harvest-then-teleport` | the composite exchange rate peaks near 6 % and is paid by correlation, not negativity; non-Gaussian conditioning never beats Gaussian; `'xp'` harvests at light contact but never spacelike; the massless limit is bounded by a log-slope ratio 0.126 | **METHODS** (a lower bound on daemonic gain from a harvested state; not teleported energy, by construction); model claim. Re-checked 2026-09-23: the harvested-state setting and the Gaussian-beats-photon-counting result were not found; "correlation, not negativity, pays" and surplus ≤ kT I(A:B) are known in general (Perarnau-Llobet et al., PRX 5, 041011 (2015); Manzano–Plastina–Zambrini, PRL 121, 120602 (2018)) | Hotta (QET); Francica et al. 2017; Kua–Serafini–Genoni 2025 (Gaussian daemonic closed forms); Bernards–Kleinmann–Gühne–Paternostro 2019 (POVMs beating projective measurements) | the composite pipeline with the passivity audit deciding what may be called teleportation; Gaussian vs photon-number-resolving conditioning on the *harvested* state | a Gaussian-QI / quantum-thermo person; the reported best is a lower bound, and an expert will ask for the SDP over all POVMs on a small Fock truncation |
| `interacting-vacua-first-numbers` | in λφ⁴ the free QEI bound survives at every λ (exact/E_free = 0.29 at λ = 4); at the exact lattice infimum interactions renormalize the reference by a mass shift (the λ = 4 dispersion is a sharp quasiparticle, gap² 3.3 % below Hartree, so "the reference is wrong" is rejected) and the residual beyond any quadratic reference is a first-order quartic tax (dR/dλ = +0.081 by exact Hellmann–Feynman/Wick theory, sign measured on the extremal state) — resolved at λ ≤ 1 (1.4 / 1.5 / 2.6 %), BRACKETED at λ ≥ 2 ([−1.797 %, +4.415 %] at λ = 2 and [−34.349 %, +15.131 %] at λ = 4 against E_H; no value quoted) | **NEW** as object and method (the exact QEI infimum of an interacting lattice theory); numbers: **diagnosed 2026-09-04** — the mechanism and the λ ≤ 1 magnitude stand, the λ ≥ 2 magnitude is bracketed and open; **twelve** earlier statements retracted, listed once in the candidate's §6.7, the **eighth on 2026-09-12** (items 9–12 followed on 2026-09-12/13, all from the S8 triage auditing itself) (the "3–12 % operator-truncation effect at λ = 4": the measured deviation at the sweep's own Fock cutoff is +39.4 %, and it *grows* with the cutoff). At λ = 4 the method is **not cross-validated at the cutoff the candidate uses**, which is a statement about the method, not only about the numbers | see (e) above — Bostelmann–Cadamuro–Fewster 2013; Bostelmann–Cadamuro–Mandrysch 2023; Mandrysch, PRD 109, 085022 (2024); Fröb–Cadamuro 2022; Fewster–Hollands 2005 — all continuum, none an infimum over all states of a lattice theory | the smeared-energy operator built as one Hermitian MPO by operator-space TEBD, its lowest eigenvalue by DMRG, the extremal state exhibited, calibrated on the exact Gaussian infimum at λ = 0 (1.27 % at the archived knobs, 0.19 % at the sweep's); the exact first-order slope of the infimum in the free extremal state; a dense-ED second implementation agreeing to ≤ 1.9e−4 where converged | a DMRG/TEBD person reading `vacuum/interacting/qei_exact.py` (Heisenberg-picture truncation error vs bond dimension; n_max convergence — the infimum is not monotone in n_max on any route, the truncated chain being a different dynamical system); a QEI person on the lattice normal ordering of the smeared energy |
| `hardware-as-noisy-field` | on the sourced ibm_cairo calibration QET teleportation proper is lost, its information signal survives, harvesting is dead; the audited free-tier on-ramp, rehearsed on a dated fake backend, then **ran on `ibm_fez` (2026-09-04): E_B_info = +0.04853 ± 0.00779, z = 6.23, against the preregistered +0.0401 (pull +1.08), E_B_ctrl < 0 as predicted, all four audits passing on the real counts** | **CONDITIONAL** for the simulated-device rows (a homogeneous calibration transcription); the real-device row is a **measurement, narrow by construction** — one device, one day, the calibration snapshot as provenance, the record pinned as a test fixture — in the lineage of Ikeda 2023's hardware QET, not a priority claim over it; further IBM runs are Xie–Sajjan–Kais, arXiv:2409.03973 (2024), and Hassan–Quaderi–Mahdy, arXiv:2408.07997 (2024). Ikeda observed the interaction term ⟨V⟩ negative on six devices and notes Bob's full gain is smaller once the positive local term counts; this row's "teleportation proper lost on ibm_fez" sharpens that caveat on one device (re-checked 2026-09-23) | Ikeda, PRApplied 20, 024051 (2023) (the IBM QET run); Hotta's minimal model; Jordan–Lee–Preskill, Science 336, 1130 (2012) and Klco–Savage, PRA 99, 052335 (2019) (scalar fields on qubits) | the encoded-field QET and harvesting circuits with ledger audits under a sourced noise model; the E_B_info signal that survives when teleportation proper does not — and a preregistered prediction, made on ibm_cairo's calibration, that held on a 2026 Heron device. **(2026-09-12)** the two encodings that had been recorded as convergence failures are now exhibited ground states of the R_Y/CX hardware-efficient ansatz — **sufficient** depth 7 at (3, 4) (three layers below the parameter-counting depth 10) and 31 at (4, 4) — with the old 3-layer rows re-read as the ansatz's expressibility ceiling. That is **METHODS**, not a priority claim: expressibility-ceiling curves for hardware-efficient ansätze are a standard object, what is added is the continuation discipline (every depth continued from its own optimum, `budget_exhausted` recorded per depth and pinned) that caught this candidate's own "minimal depth 8" as wrong by a layer | a Heron operator reading the record (`data/real_device/ibm_fez_dadkfvl1ierc738l0ch0.json`): crosstalk, the spectator's idle decay and drift between calibration and execution are not modelled; a qiskit noise-model person checking that a homogeneous calibration transcription is fair to the device |

---

## Where the value actually is

From the reviewer's brief, kept verbatim as the program's value statement:

> "Where the value actually is. The infrastructure. A certified Gaussian and lattice-QFT
> toolkit with the literature as regression tests, executable audits, and a precision
> ladder resolving margins to 1e-30 is rare. The 'literature as CI' idea is good and should
> spread. Reproduction at certified precision has value on its own terms in a field with no
> replication culture."

The program's own reading of that: `README.md`'s "The literature is the test suite" is
the product, and the thirteen candidates are its demonstrations. Four of them carry
something the literature does not have (the interacting infimum, the Gaussian 4d
constant, the MI floor, the conditional device NO-GO); two are reproductions at certified
precision that were mislabelled and are now labelled; one is a rederivation of folklore
with citations; one was re-examined and de-circularized (toy Jacobson: CHM found with a free
weight, Lifshitz is distinguished by the complexity of the admissible local weight, not its existence (CHM is the 3-mode parabola for the wave chain; Lifshitz needs ≥ 11–12 modes of a bumpy profile — 0.09 at 8 modes; below 1e−2 by 11–13 modes and below 1e−3 by 12–14, z-dependent; 5e−4 only by 20 Chebyshev modes — so the toy tests parsimony), the velocity claim withdrawn); the rest are model claims whose worth is
that every number in them can be regenerated, audited and refuted.

*Addendum, 2026-09-12.* A pass that worked six open items moved two rows' *character*
without moving their classification. The 4d constant is now **proved from below and cited
from above** — a numerical claim with one certified endpoint is a different object from a
numerical claim with none, even when the enclosure is one-sided. And the interacting
infimum's **method**, not only its numbers, is now the limiting factor at λ = 4: the route
is 39 % from an exact answer at the Fock cutoff the candidate uses, and the error grows
with the cutoff. Two of that day's four retractions were of the program's own statements
made *earlier the same day* — which is the file's own standard working as intended, and is
recorded here rather than in a changelog because it is the kind of thing a reviewer should
be able to find.

*Addendum, 2026-09-23.* The program's headline result was not new. The 4d Lorentzian-weight
bound, 2.53× Fewster–Ford–Roman's 2012 estimate and classified here as a disagreement with a
published number, had been published in 2018 by two of those three authors (Schiappacasse, Fewster & Ford, PRD 97, 025013 (2018),
C_shift = −0.0593338; ours reproduces it to 0.72 %). The search that classified it looked for
other groups' work and for Gaussian values; it never looked at the 2012 authors' own
follow-ups, which is exactly where a revision of a 2012 estimate appears. Every NEW in this
file was classified by searches of the same kind. The two that matter most — the interacting
infimum and the Gaussian 4d constant — should be re-checked to that standard (the compared
authors' own later papers, read in full) before either is cited as new. **Done the same day.** **And the rest was re-checked the same day** (`docs/LITERATURE_CHECK_2026-09-23.md`,
part 2): four more items turned out to be known — Proposition B's closed form (Salvia et al.
2023), Theorem F (classical combination resonance), the off-criticality failure map
(Leighton-Trudel 2025), and harvest-then-teleport's general messages (Perarnau-Llobet et al.
2015; Manzano et al. 2018) — and the toy-Jacobson 2+1 result gained two precedents
(Javerzat–Tonni 2022; Herzog–Nishioka 2016). What survived: the resource-theory core, the
circuit-QED no-go, the harvested-state setting and its Gaussian-measurement result,
pump-enhanced harvesting, the disorder half of the failure map, and the tomography floor.
**Result of the first re-check:** The
interacting infimum stands as NEW. The Gaussian constant stands as NEW with the one source known to hold 4d sharp computations
unread (Dawson's 2006 York thesis, not online); the record is `docs/LITERATURE_CHECK_2026-09-23.md`. The re-check found something else instead: the qei
candidate's *method* — the infimum over all states as the lowest eigenvalue of one diagonalized
quadratic operator — is prior art (Dawson 2006; Schiappacasse–Fewster–Ford 2018), and that row is
corrected.

## What would move a row

- **`time-modulated-vacua` → NEW or → REPRODUCTION:** a control theorist locates (or fails
  to locate) the per-zero measurable-s inequality in Chernousko–Akulenko–Sokolov or the
  Lavrovskii–Formal'skii line. Either way the manuscript's first paragraph cites the swing
  literature.
- **`qei-2d-sharp-constant` → a proved constant:** a certified **upper** bound on the same
  quantity. The lower endpoint is now a theorem; the upper is Fewster–Eveson's cited `R ≤ 1`.
  What it needs is named in `draft/proof_4d_gaussian_constant.md` §7: certified enclosures of
  the operator square roots of K ± K_− on the half-line (a certified rational/Chebyshev
  approximation of √ on a spectral interval plus rigorous control of σ(K) at both ends), or
  a dual certificate that avoids them. The non-rigorous target is ≈ 0.576.
- **`toy-jacobson` → a proof rather than a rate:** ASSUMPTION A5, that the lattice residual
  converges to the continuum one (r_∞ = 0), and a derivation of the bond-centred floor
  0.053/L² — the Euler–Maclaurin boundary term is the named candidate. Independently, a 3+1
  run would test whether the toy tracks ξ = (d−2)/(4(d−1)) = 1/6 as it tracked 1/8 in 2+1,
  and a derivation (rather than the present dropping) of the improvement's entangling-surface
  contact term would remove the load-bearing assumption of the 2+1 result.
- **`time-modulated-vacua` → Theorem M classified:** the literature check was run on 2026-09-23
  and did not find it; what would move it now is a control-theory or quantum-optics reader who
  knows the statement. Until then it sits at POSSIBLY KNOWN, like Theorems A/B.
- For the 4d value: the Lorentzian comparison is **closed** (a reproduction of Schiappacasse–Fewster–Ford 2018), and the authors'-own-follow-ups check of the Gaussian row was done on 2026-09-23 with nothing found. What would still move it: a copy of Dawson's 2006 York thesis (not online; the university library holds it), read for any Gaussian-sampled 4d computation, and the certified **upper** half of the enclosure.
- **`multi-interval-modular-hamiltonians`:** nothing moves it; the manuscript is a
  reproduction paper with three new lattice measurements, and should be written as one.
- **`interacting-vacua-first-numbers` → NEW with numbers at λ ≥ 2:** the diagnostics closed on
  2026-09-04 (the mechanism and the λ ≤ 1 magnitude); what remains is the λ ≥ 2 magnitude,
  now **bracketed and not resolved** ([−1.797 %, +4.415 %] at λ = 2 and
  [−34.349 %, +15.131 %] at λ = 4 against `E_H`, a 32- and 239-band systematic). **(Frozen 15-row snapshot; the pool finished on 2026-09-13 and the complete 36-row record WIDENS both brackets to [−34.382 %, +6.979 %] at λ = 2 and [−127.079 %, +15.861 %] at λ = 4 against `E_H` — completing the plane made the answer worse, not better, and 12 of 12 settings fail the consistency gate at λ = 2.)**
  **2026-09-12 changed what that costs.** The route to it is no longer "n_max ≥ 8 and
  χ_op ≥ 32": χ_dmrg is measured converged, χ_op is non-monotone in the error with a sign
  change, the **weighting** is the dominant cost (a factor 11 at one measured point), and the
  5 % reference-consistency gate the sweep uses is uninformative about the λ = 4 error. The
  plane must be walked with an **exact ED anchor at every quoted setting** (affordable at
  L ≤ 6), not on the gate; 18 of the plane's 20 jobs are unfinished and their pool was
  still running at the 2026-09-12 integration pass.
- **`circuit-qed-noise-thresholds` → a device statement:** a circuit-QED reader signs off
  on `circuit_mapping.py`.
- **`toy-jacobson`:** re-examined on 2026-09-04; what would move it now is a signed weight (the
  Lifshitz is distinguished by the complexity of the admissible local weight, not its existence (CHM is the 3-mode parabola for the wave chain; Lifshitz needs ≥ 11–12 modes of a bumpy profile — 0.09 at 8 modes; below 1e−2 by 11–13 modes and below 1e−3 by 12–14, z-dependent; 5e−4 only by 20 Chebyshev modes — so the toy tests parsimony)), the even sector after zero-mode
  projection, and a ball family beyond 1+1 — the last of which was **delivered on 2026-09-12**
  (2+1 disks, with the conformal improvement measured), while the even sector after zero-mode
  projection is only *partly* answered: its large residual is the 1+1 IR zero mode, but the
  remainder after projection is unexplained and is 1.9×–659× the odd sector's per unit
  variation.
