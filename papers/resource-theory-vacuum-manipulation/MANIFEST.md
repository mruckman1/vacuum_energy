# MANIFEST — resource-theory-vacuum-manipulation

Result candidate for Layer 7, **L3** (PLAN.md): a resource theory of vacuum
manipulation, bootstrapped by small-system numerics.

---

## The claim

**Prior art (literature check, 2026-09-23; `docs/LITERATURE_CHECK_2026-09-23.md`).** Two results below are not this
candidate's. (i) Proposition B's single-qubit closed form — the orthogonal-Procrustes maximum
Tr T + ‖T‖_* with the determinant fix — is the closed formula for the local ergotropy of a
two-level system in Salvia, De Palma & Giovannetti, PRA 107, 012405 (2023), arXiv:2206.14842,
Sec. III.A (max over SO(3) of Tr[OM − M] = Tr|M| − Tr M, the smallest singular value flipped when
det M < 0); here it is applied branch by branch to the conditional states of energy teleportation.
(ii) That a local channel on Bob's site can extract more than any local rotation (the 12.4× in the
ordered phase) is the known gap between local CPTP extraction, a Choi-matrix semidefinite program
(Frey–Funo–Hotta, PRE 90, 012127 (2014); Alhambra et al., PRL 123, 190601 (2019)), and local
ergotropy (Salvia et al.); the transverse-field-Ising number is a model measurement. The
resource-theory core — W→ as a ledger monotone under energy-non-signalling free operations, the
surplus as a non-monotone, two-way communication paying Bob nothing for commuting-Kraus
instruments, the convex-position characterization and the saturated one-shot bound — was not found
among the 193 papers citing Hotta's QET papers, Frey–Funo–Hotta, Alhambra et al. or
Rodríguez-Briones et al. Three 2024 papers by Fan, Wu, Wang, Liu & Liu (Quantum Inf. Process. 23;
PRA 110, 052424; PRA 109, 062208), which analyse energy teleportation through a correlation
matrix, are journal-only and were read at abstract level only.

Take free operations to be (O1) instruments on Alice's region whose average
channel leaves Bob's local energy operator `H_B^loc = H - H_A` invariant
(energy-non-signalling; Hotta's sigma_x projectors are the finest such
instrument), (O2) classical communication A -> B, (O3) any CPTP map on Bob's
region, (O4) mixing/relabelling. Then the **QET potential** — the one-way-LOCC
extractable energy of Bob's region, `W_->` — is a *ledger monotone*: `W_->` plus the energy already
delivered to Bob's device never increases on average under free operations
(module docstring Theorem 1, Eqs. (5a)-(5b)). `W_->` alone does not increase
under free operations that take no energy from Bob's device; a Bob channel that
injects energy, which (O3) allows, can raise it, by at most the energy injected.
[Corrected 2026-09-23, while writing the manuscript: this sentence used to say
`W_->` never increases under free operations, which is false for injecting
channels. The audit's own "teeth" are such channels.] On
commuting bonds it has the closed form (4) (Alice's projective bond measurement
+ Bob's reset), which on Hotta's minimal model equals `E_B^max` of
arXiv:1101.3954 Eq. (11) to `3.7e-15` over a
33x33 (h, k) grid: **Hotta's protocol is one
free-operation sequence and `E_B` is the resource it spends.** Negativity is a
second (LOCC) monotone; the plan's "QET surplus" `W_-> - W_loc` is **not** a
monotone (closed-form LOCC counter-example, rise `0.4142` at (h,k)=(1,0.5);
it never decreases under Bob's maps, min change `-8.7e-14` over 120 samples).
A smooth-entropy one-shot bound on Bob's unitary yield (chain (i)-(v), Ky Fan
floor at `H_min^eps` of his conditional state, duality to `H_max^eps(R|E)`)
is **saturated by the minimal model** (E_B/bound = 1 to `1.5e-11` relative,
roundoff absolute) and never violated on critical/off-critical TFIM chains
L = 4..10 at any smoothing eps (126 rows; violations: False).

**Monotonicity, numerically:** 12500 random free operations
(free instruments with 1-3 Kraus operators, Bob channels, Bob unitaries; Gaussian
free symplectics, Bob symplectics and loss) on random pure / mixed / ground-mixture /
thermal states of 2-6 qubits and 2-4 Gaussian modes: worst violation
**2.07e-14** (roundoff). Teeth: non-free Alice instruments raise
`W_->` by up to 7.72 (min over cases 0.96),
Bob-side energy injection by up to 7.50 (min 2.19),
never by more than the injected energy (ledger defect <= 9.2e-15).

**Tightness on chains (L = 8, g = 1, A = 0; the marginal-fixed SDP is computed for L <= 8):**

| |A-B| | E_B (1-site rotation) | W_site (1-site CPTP) | region-unitary bound (eps=0) | marginal-fixed SDP bound | E_B/bound_R | E_B/bound_marg |
|---|---|---|---|---|---|---|
| 1 | 3.315e-02 | 3.315e-02 | 9.179e-01 | 3.599e-02 | 0.036 | 0.921 |
| 2 | 1.205e-03 | 1.205e-03 | 1.404e-01 | 3.737e-03 | 0.009 | 0.323 |
| 5 | 2.470e-05 | 2.470e-05 | 9.104e-02 | 7.587e-03 | 2.713e-04 | 3.255e-03 |
| 7 | 6.681e-06 | 6.681e-06 | 3.166e-02 | 7.363e-03 | 2.110e-04 | 9.075e-04 |

Whole-complement ceiling `W_->(|g>)` = 1.6769e-01 at L = 8 (1.7593e-01 at L = 10; E_A = 8.5121e-01); the
protocol reaches at most 0.686 of that ceiling (over all L, g; at g = 2, d = 1).

**The rotation is optimal only at and above criticality.** On rows with
E_B > 1e-6 the single-site rotation saturates the single-site CPTP potential
W_site to 1.4e-07 at g = 1 and 1.6e-07 at g = 2, but in the
ordered phase g = 0.5 the CPTP optimum exceeds it by up to
12.4x (Kraus route, restarts = 2; certified by the Choi SDP at
L = 6, d = 1: W_site = 9.6967e-03 vs E_B = 1.6543e-03, ratio 5.86). Bob's optimal
free operation there is non-unitary — the conditional state of his site is
mixed, and a partial reset beats every rotation. Rows with E_B <= 1e-6
(g = 2, large distance) deviate by <= 1.1e-03, the local search's resolution.
The marginal-fixed SDP relaxation is saturated at L = 4, |A-B| = 2 (max ratio
1.000000) and reaches 0.921 at L = 8, |A-B| = 1; the purely entropic region bound is loose
(max ratio 0.041). Smoothing never tightened a bound
on these pure-global-state protocols (eps_best > 0 in 0 of 126 rows).

### The L3 close-out (this pass)

**A symmetric two-way theory, and its negative verdict for Bob.** Booking the
interaction symmetrically (Eq. (9); equivalently to a third "bond" register)
leaves Eq. (2) — hence Alice's free class — unchanged, and lets both parties
act: free operations are energy-non-signalling instruments on *either* side,
two-way classical communication, and a CPTP map on the *receiving* side per
round (`W_<->`, Eq. (13), finite rounds). The answer is sharp, and it is a
theorem rather than a scan: **`W_->` is a LINEAR functional**,
`W_->(rho) = Tr[W_op rho]` with `W_op = H_B^loc - sum_a lambda_min(H^(a)) Pi_a (x) I`,
whose non-Hotta part is supported on Alice's side alone (Theorem 4). Therefore
Theorem 1's inequalities are **equalities** for commuting-Kraus instruments on
A and arbitrary instruments on B, and every finite-round two-way protocol pays
Bob exactly `W_->(rho)`: over the minimal model and L = 5..8 chains the largest
deviation is `1.3e-14`.
**Two-way communication gives Bob nothing — within that scope.** By Theorem 2
the scope is the whole free class exactly where the bond spectrum is in convex
position, which is every model used here. **Off convex position the statement
is false, and it fails one-way**: on `star_hamiltonian(1.0, 0.5)` with
A = (0, 1) the non-unital witness — genuinely free, energy-signalling defect
`9.2e-16` — raises Prop. A's closed form from `0.036634` to `0.225331`, a
factor **`6.1509`**, with no communication from Bob at all. So the object
`qet_potential` computes is the *commuting-Kraus* potential `W_->^CK`; off
convex position it is neither the true potential nor a monotone, part (a) of
Theorem 4 (Bob's instruments pay for every rise) surviving while part (b)
fails. A regression test pins the factor, and
`two_way_monotonicity_audit` — whose sampling is commuting-Kraus, i.e. the
theorem's scope — now additionally evaluates the witness and reports the
breach as `worst_violation_wider_class` rather than being structurally blind
to it (the archived breach on the ground state is
`data/s7_free_class.json:monotonicity_violation` = `0.188697`; the audited
models in `data/s6_two_way.json` are all in convex position, so their
`worst_violation_wider_class` is 0 and their
`free_class_is_commuting_kraus` is true). The audit measures the *conserved*
ledger over 1180 random two-way protocols (Alice's free
instruments, Bob's measurements and channels, communication in both
directions): worst drift `9.0e-15`, with teeth
(a non-free instrument on A breaks it by at least `1.083`).
What two-way *does* buy is a second device: with both cashing out the value
reaches `20.5x` `W_->` at three rounds, but the
surplus is paid by the measurement apparatus — the net of the injections is
**round-independent** (`net_is_round_independent: true`), e.g. -0.8219 at every
round count for (h, k) = (1, 0.5).

**The free class, characterized without unitality.** **Theorem 2: an
energy-non-signalling instrument on A is commuting-Kraus if and only if every
joint bond eigenvalue vector x(a) is an extreme point of their convex hull** —
proved from the block identities (7), with no unitality assumption, superseding
the Bloch-ball + Arias-Gheondea-Gudder argument that reached only a
single-qubit A. Where the hypothesis fails there is an explicit **non-unital**
free instrument outside the commutant (Eq. (8); defect `9.2e-16`,
commutator `0.177`, unitality defect `0.75`), and a certified SDP over
*all* free instruments (Eq. (20)) measures the total weight outside the
commutant: `0.000` on every model this candidate uses and
`2.000` / `6.000` on the two- and three-Alice-qubit star models — the SDP
verdict agrees with the analytic one on every row. **Consequence:** the closed
form (4) is the *commuting-Kraus* value. On the two-qubit star model the wider
free class extracts `5.15x` more and the closed form is not even monotone
there (rise `0.189`); on the three-qubit star a witness still exists but that
particular one buys Bob nothing (`6.1e-14`). Every model in this
candidate — single-qubit A, and every region of a TFIM chain, whose commuting
sigma_x bonds have hypercube-vertex spectra — is in convex position, so Prop. A
and Theorem 1 stand on all of them, but the general theory now carries the
hypothesis explicitly.

**The tight bound for a single-site Bob is exact, not entropic.** For a
single-qubit Bob the one-way extractable energy under unitaries has a closed
form: with `T_ij = Tr[(sigma_j^b (x) A_i) tau]` the energy-correlation matrix,
`E_B = Tr T + ||T||_*` with the determinant fix (Eq. (15), orthogonal
Procrustes / Kabsch). It reproduces the chain protocol's conditional rotation
*and* Hotta's Eq. (11) to **`4.4e-16` in energy units** over 96 rows
(L = 6..12, g in {0.5, 1, 2}) — so the marginal-fixed SDP's 0.973 is its own
relaxation gap, and the "tight relaxation, not yet entropic" item is closed by
an exact solver-free formula instead. (The *ratio* deviates by up to 2.2e-5 on
rows where E_B ~ 1e-11; on rows with E_B > 1e-6 it is 1 to `1.1e-10`.)
An entropic bound is nonetheless derived and proved (Prop. C):
`E_B <= 2 c (exp(-H_min(X|R)) - 1/2)` — Bob extracts only what his region
*knows* about Alice's outcome, and it vanishes exactly when her bit is
unguessable, which is the passivity/no-signalling statement. It holds on every
row and is **loose**: tightness `0.0080` against the closed form's 1.
The b-active and row-sum refinements are loose for the same reason, identified
here: Alice's outcome moves Bob's conditional state by an O(1) trace distance
in a nearly energy-neutral direction, and the tight content is a cancellation
inside `||T||_*` that no norm-based entropic relaxation retains. Separately,
**QET's E_B is exactly a daemonic ergotropy** (Francica et al.,
arXiv:1608.00124, Eqs. (3) and (5)) with the free class constraining the
daemon, the unconditional ergotropy vanishing by strong local passivity.

**Smoothing, and L = 12.** Prop. E settles the sub-normalized question by
bounding what the enlargement could ever buy: over Tomamichel's own ball
(Def. 6.8, `S_.`) the smooth min-entropy is **unchanged** until the normalized
value saturates at `ln d` (measured gain `2.2e-15` on generic states),
where it adds exactly `-ln(1 - eps^2)` (measured against predicted to
`1e-15`). The sub-normalized chain (iii') is implemented, with the trace-t Ky
Fan floor and the doubled Lemma-3.17 penalty, and **never tightens a bound**.
Chains now reach **L = 12** — the sparse Prop. A (Eq. (19)) and a region-only
ledger, neither of which builds a dense 2^L, agreeing with the dense route to
`1e-9` where both run.

**Status: admissible as a claim about the models** (finite qubit systems with
commuting bonds; Gaussian harmonic chains), on a build whose full suite is green
(below). Nothing here is a statement about a continuum field theory.

## Manuscript (2026-09-23)

`draft/paper/main.tex` is a PRA-format draft, 11 pages; build it with `latexmk -pdf main.tex`.
Its figures are drawn by pgfplots from tables that `draft/paper/export_figdata.py` writes from
`data/`. Paper numbering: Prop. 1 = Prop. A; Theorem 3 = Theorem 4; Prop. 3 = Prop. B,
presented as Salvia–De Palma–Giovannetti's Eq. (19) applied branch by branch, not as this
candidate's result; Prop. 4 = chain (ii) plus Prop. C. The paper's Appendix B has the full map.

What writing it changed:
- The ledger statement is now exact (the corrected sentence above).
- Off convex position, what fails is the bound by `W_->^CK`. Whether two-way communication beats
  the true one-way `W_->` there is open, and the paper says so.
- Two statements were verified while writing and are not archived in `data/` (the numbers are
  in the paper). The ordered-phase gap is certified by the Choi SDP at L = 6, d = 1: `9.69672e-3`
  against `1.65426e-3`, ratio `5.8617`. The surplus rises under a FREE, energy-extracting Bob
  channel: ρ3 = ½(|+⟩⟨+|⊗|0⟩⟨0| + |−⟩⟨−|⊗|1⟩⟨1|) followed by a reset of B to |1⟩ takes Q from 0
  to `0.414214` at (h, k) = (1, 0.5), while `W_->` falls by exactly the delivered `1.0`.
- New closed forms, all elementary. `W_->(|g⟩) = √(h²+4k²) − (h²+2k²)/√(h²+k²)` is identical to
  Hotta's Eq. (11), because (h²+4k²)(h²+k²) = (h²+2k²)² + h²k²; checked to 3.6e-15 on 81 points.
  With r = k/h, E_B^max/E_A < r²/(2(1+2r²)) < 1/4. The SDP optimum of Eq. (20) equals the total
  rank of the non-extreme blocks: 2 and 6 on the star models, as archived.
- Scope notes. Unital energy-non-signalling instruments are commuting-Kraus for ANY spectrum
  (Arias–Gheondea–Gudder), so Theorem 2's content is about non-unital instruments. The one-shot
  bound's saturation on the minimal model reflects pure conditional states, and the paper
  presents it that way, not as a tight entropic principle.
- Prior art found while writing: `docs/LITERATURE_CHECK_2026-09-23.md`, part 3. Xie–Sajjan–Kais's
  "local effective Hamiltonian" is Prop. A's H^(a). Itoh et al. also connect feedback extraction
  to daemonic ergotropy. Fan et al.'s "QET passive" states may be the free states.

Before submission:
1. Read Fan et al., PRA 110, 052424 and QIP 23, 367, and Wu et al., PRA 109, 062208, in full.
   They are journal-only, so this needs a library copy or a request to the authors. An
   open-access retry on 2026-09-23 found them still closed: Unpaywall and Semantic Scholar
   list no copy, arXiv has no preprint, and their 7 citing papers mention them only in passing.
   An outside reader's advice, taken the same day: until they are read, no public description
   of the paper should lead with Proposition 1 (the closed form). Lead with the core instead:
   convex position, linearity, and two-way communication paying Bob nothing. The paper itself
   now disclaims the two-qubit cases of Propositions 1 and 3, which the correlation-matrix
   analyses in those papers may contain.
2. Have a QET researcher read the draft. Not needed before a preprint listing.
3. Done 2026-09-23: author, affiliation, e-mail, ORCID and repository URL, and the AI-use
   statement in the acknowledgments ("AI assistants, including Anthropic's Claude models, were
   used extensively ..."). The author may edit the statement, and should name any other AI
   tools that were used.
4. Obtain an arXiv endorsement, and upload a freshly built `main.bbl` (arXiv does not run BibTeX).

---

## The producing build

| item | value |
|---|---|
| generator | `papers/resource-theory-vacuum-manipulation/notebook.py`, entry point `main()` |
| generator sha256 | `ddd1890971609aab155b52c7d91a8386aaad75e06d989d0d4e76786bd387e68f` |
| produced (UTC) | 2026-09-05T18:01:13Z |
| git HEAD (`git rev-parse HEAD`) | `5e98182b80822ea1858622ec704be462f148c4d1` — code-identical to the settled tree `461b9db` (the diff `461b9db..5e98182` is `README.md` and `docs/STATUS.md` only); HEAD advanced to `5133379` (again docs-only) while the run was in flight. This candidate's own files: `vacuum/qet/resource.py`, `tests/test_resource_theory.py`, `papers/resource-theory-vacuum-manipulation/**`, plus the seventeen re-export lines in `vacuum/qet/__init__.py` that `tests/test_exports.py` requires of any new public name |
| environment | Python 3.13.3, numpy 2.5.2, cvxpy 1.9.2, macOS-14.6.1-arm64-arm-64bit-Mach-O |
| command | `set -o pipefail; .venv/bin/python -u papers/resource-theory-vacuum-manipulation/notebook.py 2>&1 \| tee clean_run.log` — exit 0 |
| wall clock | 475 s, quiescent machine (no other notebook running) |
| reproduction | every archived field of the 2026-09-04 `data/` — 6 JSON and 4 `.npz` files, every array and every summary scalar — is reproduced to 1e-12 relative; **nothing moved** (the field-by-field diff is the check, not the regenerated `summary.json`'s say-so) |
| pytest, this candidate | `tests/test_resource_theory.py`: **44 passed** in 21.46 s |
| pytest, full suite | **1211 passed, 0 failed, 10 warnings in 917 s (0:15:17)** (`.venv/bin/python -m pytest -q -p no:cacheprovider`) on the producing tree, run immediately after the notebook. The count is one above the settled-tree 1210 because the working tree carried one other agent's in-flight, uncommitted edit to `tests/test_qei.py` and `papers/qei-2d-sharp-constant/notebook.py`, which added a test and passed; nothing outside this candidate was touched by this pass |

The sha256 of the generator is embedded in every `.npz` (`__build__`) and in
`summary.json`.

**Admissibility under `papers/README.md`: MET for this candidate.** The rule
is that the full known-results suite passes on the exact code state that
produced the numbers. On 2026-09-05 the notebook was run end to end
(pipefail, exit 0) on the quiescent tree at `5e98182` (code-identical to the
settled `461b9db`), the full suite then passed on that same tree with zero
failures, and the regenerated `data/` reproduces the archived `data/` field by
field. This supersedes the two earlier records: the 2026-09-04 producing run,
whose tree was red in seven *other* packages under concurrent edits (recorded
here at the time, and the reason the gate was then declared unmet), and the
integration coordinator's 1201-passed run at `c18a206` on 2026-09-04, which
was an admissibility record for the code but not a reproduction of this
candidate's `data/`. Every number in this MANIFEST and in `draft/outline.md`
now traces to `data/` produced on a green build.

## Modules used

| module | role |
|---|---|
| `vacuum.qet.resource` (new) | definitions, `qet_potential` (closed form), `negativity`, `monotonicity_audit`, smooth entropies (`smooth_min_entropy` dense, `min_entropy_sdp`/`max_entropy_sdp` cvxpy), `one_shot_bound`, `one_shot_bound_marginal`, the ledgers, the Gaussian mirror. This pass adds the close-out: `free_class_extremality`, `nonfree_energy_nonsignalling_instrument`, `search_free_instrument` (SDP certificate), `star_hamiltonian`; `split_hamiltonian_symmetric`, `qet_potential_operator`, `two_way_potential`, `two_way_monotonicity_audit`, `optimal_local_extraction`; `single_site_extractable` (Prop. B), `bob_correlation_matrix`, `unitary_energy_spread`, `guessing_probability_cq`, `one_shot_bound_entropic`, `daemonic_gain`; `chain_potential_sparse` and `chain_region_ledger` (L = 12); `smooth_min_entropy(subnormalized=)`, `ky_fan_floor(trace=)`, `one_shot_bound(subnormalized=)` |
| `vacuum.qet.hotta` | the minimal model (closed forms, coded protocol) — read-only |
| `vacuum.qet.chains` | TFIM ED protocol, local Hamiltonians, reduced densities — read-only |
| `vacuum.qet.slp` | `extractable_energy_kraus`/`_sdp`, certificate, Choi plumbing — read-only |
| `vacuum.core` | `harmonic_chain_K`, `ground_state_cov`, `thermal_state_cov`, `log_negativity`, `loss`, `mean_energy` |
| `vacuum.audits` | `AuditResult`/`AuditViolation` for the monotonicity audits |
| `experiments/` | none |

## Literature anchors (transcribed, cited at the point of use)

Hotta arXiv:1101.3954 Sec. 3 (Eqs. 8, 11, 12, the line after 13);
Chitambar-Gour arXiv:1806.06107 Def. 1, Eqs. (57)-(61); Tomamichel
arXiv:1504.00233v5 Defs. 3.9, 3.15, 6.2, 6.4, 6.8, 6.9, Lemmas 3.17, 6.3, 6.7,
Eqs. (6.25), (6.37), Prop. 6.14; Vidal-Werner PRA 65, 032314 (2002);
Frey-Funo-Hotta PRE 90, 012127 (2014) and Alhambra et al. PRL 123, 190601
(2019) through `vacuum.qet.slp`; Arias-Gheondea-Gudder JMP 43, 5872 (2002)
for the fixed-point/commutant statement (unital channels — now needed only as
the historical route, Theorem 2 superseding it). This pass adds:
Tomamichel Defs. 3.12 (generalized fidelity), 3.15 (purified distance), 6.8
(the sub-normalized eps-ball) and Eqs. (6.27)-(6.29) (guessing probability,
Koenig-Renner-Schaffner) for Props. C and E; Chitambar-Gour Sec. II on LOCC
round number as a resource and the non-closedness of LOCC, for the
finite-round definition of `W_<->`; and G. Francica, J. Goold, M. Paternostro,
F. Plastina, "Daemonic ergotropy: enhanced work extraction from quantum
correlations", npj Quantum Information 3, 12 (2017) [arXiv:1608.00124],
Eqs. (1)-(3) and (5), for the identification of E_B as a daemonic gain.

## Data

```
data/summary.json          every headline number, build info
data/s1_monotonicity.json  per-case worst violations, teeth, sample counts
data/s2_hotta_grid.npz     33x33 (h,k): W_arrow, E_B, E_A, negativity, bounds, ratios
data/s3_chain_rows.npz     one row per (L, g, B): E_B, W_site, W_full, all bounds, ratios
data/s4_surplus.json       counter-examples; Q under free / Bob operations
data/s5_gaussian.npz       harmonic chains N = 2..4 vs mass
draft/fig_chain_L12_bounds.svg  exact vs entropic vs region bound at L = 12
draft/fig_*.svg            the charts (plain SVG, no plotting dependency)
draft/outline.md           the manuscript outline
```

## What is NOT claimed

- The two-way result is a statement about *this* free class and finite rounds.
  Unbounded-round LOCC is not a closed set (Chitambar-Gour Sec. II), and the
  symmetric theory's positive content — two devices filled at the measurement
  apparatus's expense — is not a claim that the vacuum yields more energy.
- Theorem 2 characterizes the free class for COMMUTING bonds (Prop. A's
  setting). Non-commuting bond operators admit no finest free measurement and
  are untouched here; so is the question of which *wider*-class instrument is
  optimal where convex position fails (we exhibit one, not the optimum).
- Prop. B is for a single-qubit Bob (any region size around him). A
  multi-qubit Bob has no such closed form here, and the marginal-fixed SDP
  remains the tight relaxation there.
- The entropic bound (Prop. C) is proved but ~2 orders of magnitude loose; an
  entropic quantity retaining the ||T||_* cancellation is open. The daemonic
  identification is a reading of Francica et al.'s definitions, not a new bound.
- L = 12 covers the single-site-Bob observables and the whole-complement
  ceiling only; the Bob = complement CPTP optimum past L ~ 10 is still out of
  reach, and nothing here is a thermodynamic limit.
- Nothing about continuum fields, DMRG-size chains, or hardware.

## Reproducing

```
.venv/bin/python papers/resource-theory-vacuum-manipulation/notebook.py          # full, < 15 min
.venv/bin/python papers/resource-theory-vacuum-manipulation/notebook.py --quick  # ~40 s
.venv/bin/python -m pytest tests/test_resource_theory.py -q                      # ~15 s
.venv/bin/python -m pytest -q                                                    # admissibility gate
```
