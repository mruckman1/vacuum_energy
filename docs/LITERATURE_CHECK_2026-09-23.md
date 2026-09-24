# Literature check, 2026-09-23

The record of the novelty re-check done after the program's headline result — the 4d
Lorentzian QEI infimum, 2.53× Fewster–Ford–Roman 2012 — turned out to have been published
by two of the same authors in 2018 (Schiappacasse, Fewster & Ford, PRD 97, 025013). The
search that had classified it NEW looked for other groups' work and for Gaussian values,
never for the compared authors' own follow-ups. This file applies that standard to the two
remaining claims that mattered most, so the verdicts in `docs/PRIORITY.md` (d) and (e) can be
audited and rerun.

**The standard.** For a claim that something is new relative to named papers: list the
compared authors' own later papers and everything citing the compared papers, screen the
titles, and search the full text of every plausible one for the compared quantity.
"Right" and "new" are separate checks.

## Claim A — the Gaussian-sampled 4d QEI constant, C₄d/C_FE = 0.4574904

**Seeds** (INSPIRE record id; citing records at the time of the check):

| paper | arXiv | recid | citing |
|---|---|---|---|
| Fewster–Ford–Roman 2010 (2d) | 1004.0179 | 850489 | 49 |
| Fewster–Ford–Roman 2012 | 1204.3570 | 1111011 | 45 |
| Fewster–Ford–Roman 2014 | 1409.7516 | 1319173 | 2 |
| Schiappacasse–Fewster–Ford 2018 | 1711.09477 | 1638976 | 11 |
| Fewster–Ford 2020 | 1909.07295 | 1754299 | 15 |
| Wu–Ford–Schiappacasse 2022 | 2211.12001 | 2513730 | 2 |
| Fewster–Eveson 1998 | gr-qc/9805024 | 470209 | 121 |
| Fewster, lectures 2012 | 1208.5399 | 1181999 | 95 |

Union of citing records: **252**. All titles screened (50 follow the probability-distribution
series; 202 cite only the two foundational QEI references).

**Searched in full text** for Gaussian sampling, "optimal", "sharp", "lowest eigenvalue",
"C_shift" and 4d values — none contains a Gaussian-sampled 4d sharp value:

- The series: Schiappacasse–Fewster–Ford 2018 (Lorentzian and compactly supported weights,
  no Gaussian), Fewster–Ford 2015 (arXiv:1508.02359), Fewster–Ford 2020 (Gaussian closed forms
  are 2d CFT), Wu–Ford–Schiappacasse 2021 (arXiv:2104.04446) and 2022 (compactly supported
  sampling in a cavity). The 2010 and 2024 (arXiv:2412.02917) papers are two-dimensional.
- Fewster–Teo 2000 (gr-qc/9908073: QIs as eigenvalue problems, applied to mirrors and
  δ-pulses, not to sampling-function constants); Eveson–Fewster 2007 (math-ph/0702074);
  Fewster–Kontou 2018 (1809.05047); Fewster–Thompson 2023 (2301.01698); Fewster's 2012 lectures
  (1208.5399: "optimal bounds are lacking in general"); Kontou–Sanders 2020 review (2003.01815); Ford 2022 review
  (2212.01520); Kontou et al. 2024 (2405.05963: "This has not been proven for any of the
  four-dimensional QEIs bounds discussed"); Fliss et al., Modave lectures 2026 (2605.18964);
  Palessandro 2026 (2608.11817); Abreu–Visser 2010 (1001.1180); Huang–Ford 2020 (2005.08355);
  Taylor et al. 2023 (2312.17155); the premetric-electrodynamics QEI paper 2017 (1709.01760).
- Abstract only: Shu et al., Chin. Phys. Lett. 23, 25 (2006) — a Fewster–Eveson-type
  spacetime-averaged bound, not a sharp constant.

**INSPIRE full-text searches** (`fulltext:"optimal quantum inequality" and fulltext:Gaussian and
fulltext:"four dimensions"`; `fulltext:"quantum energy inequality" and fulltext:sharp and
fulltext:Gaussian and date>2012`) return only papers listed above.

**Not read.** S. P. Dawson, "Bounds on negative energy densities in quantum field theories on
flat and curved space-times", PhD thesis, University of York (2006) — the one source **known to
hold 4d sharp computations**. It is not in the White Rose eTheses archive (294 theses listed for
2006, none by Dawson), nor on arXiv or INSPIRE. The 2018 paper describes it as Colpa
diagonalization on a torus with a *squared-Lorentzian* sampling function, finding the
Fewster–Eveson-type bound about 3× the sharp one. Also unread: Fewster, "Quantum Energy
Inequalities", in *Wormholes, Warp Drives and Energy Conditions* (Springer, 2017) — a review, not
on arXiv.

**Verdict.** NEW for its family, to this standard, with Dawson's thesis unread.

**Found on the way — the method is prior art.** Computing the sharp bound as the lowest
eigenvalue of the diagonalized smeared quadratic operator was done by Dawson (2006) and by
Schiappacasse–Fewster–Ford (2018). The candidate's lattice pull-back is a formulation of that
route, not a new one; `papers/qei-2d-sharp-constant/MANIFEST.md` no-longer-claims item 2. Two
points of the sampling-function dependence also predate the candidate's six-family scan: the
2018 paper's Fewster–Eveson-type bound at 4.6× the sharp value for a compactly supported weight,
and Dawson's ≈ 3× for a squared Lorentzian.

## Claim B — an exact QEI infimum over all states of an interacting lattice theory

**Compared authors** (from `docs/PRIORITY.md` (e)): Fewster–Hollands 2005 (2d CFT);
Bostelmann–Cadamuro–Fewster 2013 (massive Ising, 1304.7682); Bostelmann–Cadamuro–Mandrysch 2023
(2302.00063); Mandrysch 2024 (PRD 109, 085022, 2312.14960 — numerical optimal bounds in
continuum integrable models at the one- and two-particle level); Fröb–Cadamuro 2022
(2212.07377, sine-Gordon, an absolute bound).

**Their papers since 2023**, by INSPIRE author query — Mandrysch (6), Cadamuro (11), Bostelmann
(5), Fröb (25), Fewster (16), Hollands (19): no interacting infimum over all states. The later
work is on modular theory, relative entropy (including Fröb, "Relative entropy for λφ⁴ in the
Rindler wedge", arXiv:2607.07810 — not a QEI) and thermal Lᵖ inequalities.

**Full-text searches** for energy inequalities with DMRG, tensor networks or matrix-product
states, for quantum inequalities on a lattice with φ⁴, and for negative energy densities with
DMRG: nothing of the kind. Two web searches likewise. Nearest new item: Ikeda, "Quantum energy
teleportation across lattice and continuum", arXiv:2604.09973 (2026) — energy teleportation in
the massive Thirring / sine-Gordon model on a lattice, not a QEI infimum. It bears on this
program's energy-teleportation candidates, which were not re-checked here.

**Verdict.** NEW as an object and as a method, unchanged. Unlike claim A's route, the MPO route
is needed because the smeared energy of λφ⁴ is not quadratic. The candidate's own numbers remain
resolved only at λ ≤ 1 (see `docs/STATUS.md`).

## Part 2 — the remaining candidates (same day)

The same standard applied to everything else the documents called new. "Citing" counts are
INSPIRE records at the time of the check; "full text" means the arXiv PDF was searched for the
specific claim, not skimmed.

### Energy-teleportation resource theory (`resource-theory-vacuum-manipulation`)

Seeds: Hotta's QET review (arXiv:1101.3954) and PRD 78, 045006 (2008) (arXiv:0803.2272);
Frey–Funo–Hotta, PRE 90, 012127 (2014); Alhambra et al., PRL 123, 190601 (2019);
Rodríguez-Briones et al. (arXiv:2203.16269). Union of citing records: **193**, 77 on topic.
Full text: Salvia–De Palma–Giovannetti (2206.14842), Wang–Yao (2405.13886), Alhambra et al.
(1902.02357), Frey–Funo–Hotta (1404.5081), Hotta 2010 (1002.0200), Itoh–Masaki–Matsueda
(2408.11522), Bhattacharyya et al. (2307.16746), Rodríguez-Briones et al. (2203.16269),
Castellano et al. (2401.10996). Abstract only: Haque (2601.18718), and three journal-only 2024
papers by Fan, Wu, Wang, Liu & Liu (Quantum Inf. Process. 23; PRA 110, 052424; PRA 109, 062208).

- **Proposition B, Tr T + ‖T‖_* with the determinant fix — REPRODUCTION.** Salvia, De Palma &
  Giovannetti, PRA 107, 012405 (2023), Sec. III.A: the local ergotropy of a qubit is
  max over SO(3) of Tr[OM − M] = Tr|M| − Tr M, with the smallest singular value flipped when
  det M < 0. Same closed form; the candidate applies it per measurement branch.
- **The 12.4× local-channel-over-rotation gap — known phenomenon, new number.** Local CPTP
  extraction as a Choi-matrix semidefinite program is Alhambra et al. (2019), credited as such in
  Salvia et al.'s introduction, which also notes it bounds local ergotropy from above.
- **The core — NEW, holds.** The ledger monotone under energy-non-signalling free operations, the
  non-monotone surplus, two-way communication paying Bob nothing, the convex-position
  characterization and the saturated one-shot bound appear in none of the papers read. The three
  Fan et al. papers analyse QET through a correlation matrix and remain unread in full.

### Harvest-then-teleport (`harvest-then-teleport`)

Seeds: Francica et al., npj Quantum Inf. 3, 12 (2017); Perarnau-Llobet et al., PRX 5, 041011
(2015); Kua–Serafini–Genoni, Quantum Sci. Technol. 11, 015014 (2026). Citing: **366**, 108 on topic.
Harvesting combined with teleportation, ergotropy or work extraction: **55** records, none on the
combined setting. Full text: 2506.22288, 1407.7765, 2102.13606, 1805.08184, 1907.01970,
2606.31150, 2505.24596, 2202.05050.

- **"Correlation, not negativity, pays" and surplus ≤ kT I(A:B) — known in general.**
  Perarnau-Llobet et al. (2015): separable states store nearly as much work as entangled ones,
  work proportional to mutual information. Manzano, Plastina & Zambrini, PRL 121, 120602 (2018):
  measurement-assisted work bounded by the shared classical correlations. The harvested-state
  instance is new as a setting.
- **The joint-ergotropy conjecture refuted** — the program's own conjecture; measurement-assisted
  extraction exceeding unitary ergotropy is the premise of daemonic ergotropy.
- **Gaussian measurements never lose to photon counting on harvested states — not found.**
  Kua–Serafini–Genoni treat general-dyne (Gaussian) measurements only; Bernards et al. (2019)
  compare generalised with projective measurements, not Gaussian with photon counting.

### Floquet theorems (`time-modulated-vacua`)

INSPIRE: harvesting with a driven, modulated or dynamical-Casimir field — one tangential record
(arXiv:2502.02643). Two web searches. Full text: Svidzinsky, Zhang, Wang, Wang & Scully,
arXiv:1407.3727. The general-literature database (OpenAlex) rate-limited every query and could
not be used.

- **Theorem F — REPRODUCTION.** Eq. (15) of arXiv:1407.3727 gives the sum-combination-resonance
  gain δ/(4√(ω₁ω₂)); the resonance itself "has been discovered in electronic circuits in 1950's".
- **Theorem M — POSSIBLY KNOWN, unchanged.** Not found as stated; the nearest paper read, Arenz et
  al., "Amplification of quadratic Hamiltonians", Quantum 4, 271 (2020), states no such bound.
  The control-theory literature is where it would be, and it is outside INSPIRE.
- **Pump-enhanced harvesting — not found.** The loss threshold is the gain-equals-loss condition.

### Toy Jacobson 2+1 (`toy-jacobson`)

Seeds: Jacobson, PRL 116, 201101 (2016); Javerzat–Tonni, JHEP 02 (2022) 086; Di Giulio–Tonni,
J. Stat. Mech. (2020) 033102. Citing: **312**, 39 on topic. Full text: Javerzat–Tonni
(2111.05154), Huerta–van der Velde (2301.00294), Chen (2202.00680). Abstract: Herzog–Nishioka
(1610.02261).

- Javerzat–Tonni find the lattice entanglement Hamiltonian of a sphere for the free massless
  scalar in agreement with the conformal prediction in two and three spatial dimensions at low
  angular momentum — a prior lattice confirmation by a different route.
- Herzog–Nishioka, JHEP 12 (2016) 138, derive the modular Hamiltonian's boundary term for a
  non-minimally coupled scalar (planar interface; Weyl-related geometries at conformal coupling,
  the ball included). This is the term the toy drops; it should be compared with theirs.
- **What holds:** measuring ξ through the entanglement first law. No paper does that.

### Geometry failure map (`geometry-from-entanglement-failure-map`)

Seeds: Cao–Carroll–Michalakis (arXiv:1606.08444); Qi (arXiv:1309.6282). Citing: **397**, 93 on
topic, plus 37 since 2021-08 under a wider keyword set. Full text: Leighton-Trudel
(2507.09749).

- **Off criticality — known.** Leighton-Trudel (2025) proves a mutual-information distance is a
  metric at criticality and superadditive in gapped phases, checked exactly in the
  transverse-field Ising chain; his earlier numerical study is not indexed anywhere reached.
- **Disorder — not found.** Leighton-Trudel has no disorder content.

### Circuit-QED no-go (`circuit-qed-noise-thresholds`)

Seeds: the proposal, Teixidó-Bonfill, Dai, Lupascu & Martín-Martínez, arXiv:2505.01516 (its
subject is detector-gap variation), and Janzen–Dai–Ren (2208.05571). Citing: **31**. Authors'
later papers: none on noise. Title searches for harvesting with decoherence, dissipation, noise,
dephasing, open systems or loss: only photosynthetic light-harvesting papers.
**Verdict: not found; the no-go stands, conditional on the mapping as before.**

### Hardware run (`hardware-as-noisy-field`)

Seed: Ikeda, Phys. Rev. Applied 20, 024051 (2023). Citing: **49**. Full text: Ikeda 2023, Hassan
et al. (2408.07997), Xie et al. (2409.03973). Ikeda ran six IBM devices, observed ⟨V⟩ negative on
all, and notes Bob's full gain is smaller with the positive local term; Hassan et al. ran
ibm_brisbane, ibm_kyiv and ibm_sherbrooke. **Verdict: a measurement in that lineage, as the
candidate already said; the documents now cite it.**

### Tomography floor and derivative coupling

Title searches for harvesting with tomography, estimation or finite statistics, and for noise
follow-ups to Tjoa–Martín-Martínez 2021: nothing relevant. Both stay METHODS.

### Not re-checked

`differentiable-vacuum` (a software methods claim, never listed as new physics) and the two
reproductions already labelled as such (`multi-interval-modular-hamiltonians`, the gapped-chain
modular results).

## Part 3 — found while writing the resource-theory manuscript (same day)

Writing `papers/resource-theory-vacuum-manipulation/draft/paper/main.tex` meant checking papers
from the part 2 lists against specific statements, not only the core claims. That turned up
three more overlaps, none with the core.

- **Proposition A's effective Hamiltonian — PRIOR CONCEPT.** Xie, Sajjan & Kais, Entropy 27,
  1147 (2025), arXiv:2502.05288, Sec. III, introduce Bob's "local effective Hamiltonian"
  H_B^(±) = −hZ_B ± κX_B. They note that Bob's optimal operation rotates his state to its ground
  state. That is Proposition A's H^(a), for a product state. What Proposition A adds is
  optimality over all commuting-Kraus instruments and all Bob channels, for any state, with Bob
  holding the complement; that is not in their paper. The manuscript credits them.
- **The daemonic-ergotropy reading — ALSO MADE by Itoh, Masaki & Matsueda**, PRE 113, 024108
  (2026), arXiv:2408.11522. They note that the maximal energy extracted by feedback unitaries under
  a fixed projective measurement is Francica et al.'s daemonic ergotropy, and they derive
  information-thermodynamic bounds on it. The candidate never claimed the reading as new; the
  manuscript cites them for both.
- **Free states — a POSSIBLE ANALOGUE in a journal-only paper.** Fan et al., PRA 110, 052424
  (2024), define "QET passive" states. These may be the free states (W→ = 0) under a different
  class of operations. The paper is still unread in full; the manuscript calls them an
  analogue, and its pre-submission list requires reading the paper.

The three Fan/Wu abstracts were read from INSPIRE (`doi:` queries). Fan et al., PRA 110: a
correlation matrix, steering, "strong QET", QET passive states, noisy channels, N qubits. Wu et
al., PRA 109: strongly locally passive pure states of the minimal model by correlation-matrix
analysis, and efficiency > 1 for non-ground states. Fan et al., QIP 23: quantum resources and
efficiency, necessary and sufficient conditions for the minimal model, Gibbs states and spin
chains. None mentions a resource theory, a monotone, a free class, convex position, linearity or
two-way communication. **Verdict unchanged: the core survives; three more credits.**

**Retry for the full texts, later the same day.** Unpaywall and Semantic Scholar list all three
papers as closed with no open copy, and an arXiv author search finds no preprint. Semantic
Scholar lists 7 citing papers, 4 of them on arXiv; those mention the three only in passing.
Abd-Rabbou et al. (arXiv:2601.18327) cite Fan et al., PRA 110, as framing QET "within the
resource theories of quantum steering". That is one more reason to read it before the paper is
presented as more than *a* resource theory of QET. Still unread in full. The manuscript now
disclaims the two-qubit cases of its Propositions 1 and 3.

## Rerunning it

INSPIRE REST API, `https://inspirehep.net/api/literature?q=<query>&size=250`: record ids from
`arxiv:<id>`; citing papers from `refersto:recid:<id>`; author output from
`a "Surname, Given" and date>=2023`; the full-text queries as quoted above. The screening
lists and the full texts used lived in the session scratch directory and are not archived; the
queries regenerate the lists, and the counts above will grow as the literature does.
