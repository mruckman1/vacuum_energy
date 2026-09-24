# MANIFEST — `toy-jacobson`

Result candidate of Layer 4 (PLAN.md, discovery target "A toy Jacobson test") and Layer 7 leap
L2 ("Toy Jacobson: how much of wave dynamics is recoverable from δS = δ⟨K⟩ alone on a lattice with
no gravity anywhere in the definition").

---

## The motivating objection (2026-09-04)

An outside reviewer wrote, verbatim:

> "Overclaimed: toy Jacobson. K_loc is the CHM conformal-Killing integral, which is derived
> from conformal symmetry. Finding that the first law then holds for conformal dynamics and
> fails for Lifshitz is a consistency check on CHM, not Lorentz invariance emerging from
> entanglement. The third-neighbour degeneracy test is clever and the scaling is clean. The
> headline is circular. The version that would mean something optimizes over the weight
> function too and asks whether any local K works for non-relativistic dynamics."

The reviewer is right about the headline. Anchors 1–8 below pair δS with **one** local form,
K_loc = 2π Σ βᵢ hᵢ with β the CHM parabola — a weight *derived from* conformal symmetry — so
"r → 0 for the wave chain, r = O(1) for Lifshitz" tests whether the lattice reproduces CHM, not
whether entanglement equilibrium selects Lorentz-invariant dynamics. Anchor 9 (Sec. G of the
notebook, `free_weight_residual` / `universal_weight_residual`) frees the weight and is the
version that means something. What was circular, and what the new test establishes, is stated
in full under "What was circular; what the free-weight test establishes" below.

## The headline (rewritten twice after the adversarial audits of 2026-09-04)

**With a free nonnegative local weight, entanglement equilibrium distinguishes the wave chain
from z = 2 Lifshitz dynamics by the complexity of the admissible weight, not by its existence:
the wave chain's scale-covariant weight is the 3-mode CHM parabola (found; overlap ≥ 0.9992
across six bases; r = 1.5e-04 shared across L = 8, 16, 32, only with amplitude ∝ R), while
Lifshitz admits a scale-covariant nonnegative weight only at ≥ 11–12 modes of a bumpy profile
(below 1e-02 by 11–13 modes and below 1e-03 by 12–14, z-dependent; 5e-04 only by 20 Chebyshev modes; 0.09 at 8 modes — and beyond ~13 modes the family exceeds the variation family's 8–9
conditions at the 10⁻³ level).** What that does and does not say: with enough modes *every* tested
dynamics fits, so the toy tests the parsimony of the modular weight — how many modes of a
nonnegative scale-covariant profile δS = δ⟨K_loc⟩ requires — and finds a gap of a factor ~4 in
mode count between the wave chain and z = 2; it does **not** establish that Lorentz-invariant
dynamics are singled out by the existence of a local modular Hamiltonian. The 3-mode per-ball
numbers (0.270, 0.507, 0.679) and the 8-mode shared floor (0.091) for Lifshitz are properties of
those families, reported as such; per ball, four modes already fit a Lifshitz ball (7.0e-04,
6.7e-03, 8.2e-02 at L = 8, 16, 32). The free weight absorbs the velocity exactly (the vacuum of
v²K is the vacuum of K — v is a unit, and the earlier "v² = 0.980 recovered" was the CHM
normalisation's unit) and a mass per ball; only the shared profile sees the mass, and the
constrained fit with it lands at m/v = 0.002, c₂/c₁ = −0.008. A per-site weight has as many
parameters as the variation family has conditions (rank 5–6 at 10⁻⁶) and fits everything; the
rank is reported next to every parameter count.

---

## The claim (anchors 1–8, the CHM weight assumed; anchor 9, the weight free)

**Prior art (literature check, 2026-09-23; `docs/LITERATURE_CHECK_2026-09-23.md`).** (i) The 1+1 free-weight result's *weight*
is known on the lattice: Di Giulio & Tonni, J. Stat. Mech. (2020) 033102, arXiv:1911.07188, extract
the CHM/Bisognano–Wichmann weight from the lattice entanglement Hamiltonian of an interval in the
massless harmonic chain (cited here as a cross-check); what is this candidate's is finding it
through the first law with the weight left free. (ii) In 2+1, Javerzat & Tonni, JHEP 02 (2022)
086, arXiv:2111.05154, find the lattice entanglement Hamiltonian of a sphere for the free massless
scalar in agreement with the conformal-field-theory prediction in two and three spatial dimensions
at low angular momentum — a prior lattice confirmation, by a different route, of the form whose
improvement coefficient anchor 10 measures. (iii) The entangling-surface boundary term that anchor
10 *drops* has been derived: Herzog & Nishioka, JHEP 12 (2016) 138, arXiv:1610.02261, for a planar
interface and, at conformal coupling, for Weyl-related geometries including the ball; Chen,
arXiv:2202.00680 (2022), treats the same boundary terms in the generalized first law. The dropped
term should be compared with theirs before "keeping it makes the fit ten times worse" is read as
physics. What remains this candidate's is measuring ξ through entanglement equilibrium — a new
route to a known coefficient.

The entanglement first law δS = δ⟨K⟩ is an identity; Jacobson's entanglement-equilibrium argument
(PRL **116**, 201101 (2016), Eqs. (14)–(19)) carries content only because the ball's modular
Hamiltonian is the *local* Casini–Huerta–Myers form 2π∫(R² − r²)/(2R) T₀₀ (JHEP 05 (2011) 036,
Sec. 2), a property of Lorentz-invariant dynamics. On a 1+1 harmonic chain we measure the
residual r(K) = max |δS − δ⟨K_loc⟩|/|δS| over centred balls and long-wavelength reflection-odd
squeeze variations, with K_loc the CHM weight times the lattice's own PSD bond-split energy
density and δS from the exact Gaussian modular Hamiltonian. **(1)** For the massless
nearest-neighbour chain r falls from 9.09e-02 (L = 4) to 3.4e-04 (L = 48) as
L^{-2.27} with continuum extrapolations r_∞ = -1.9e-04 (1/L²) and
+4.6e-04 (fitted power): consistent with 0 at the 5×10⁻⁴ level; the velocity that
minimises r converges as 1 − v* ≈ 0.85/L² (0.9466, 0.9868, 0.9968, 0.9993, 1.0003 for L = 4, 8, 16, 32, 64).
**(2)** The exact first law holds by independent routes at ≤ 8.6e-09 for all nine dynamics
audited (float64 alone: 1.3e-07, the Lifshitz chain, which needs the mpmath rung of the
precision ladder). **(3)** Non-relativistic dynamics stay away from zero and grow with the ball:
Lifshitz ω ~ k² gives r = 0.954, 0.974, 0.991 at L = 8, 16, 32
(44×, 185×, 911× the wave chain), ω ~ √k gives 4.3–9.9, 1/n²
couplings 0.54–0.16; a wrong velocity gives r = |1 − v| (v = ½: 0.5000); a mass gives
r ≈ (mL)^~1 saturating near 0.33. Over 44 dynamics r = 0.79/L² + 0.89·|ω(π/L)/(π/L) − 1|
(rms relative scatter 0.24): the residual tracks the departure of the bulk dispersion from
ω = k at the ball's own scale, on top of an ansatz-discretisation floor 1.2/L² (the improved
Laplacian, dispersion linear to O(k⁶), has r = 0.0192, 0.0047, 0.00093). **(4)** Minimising
r over K = m² + c₁L₁ + c₂L₂ from the generic start (c₁, c₂, m) = (2.0, -0.3, 0.2) lands at
(c₁, c₂, m) = (1.634, -0.163, 0.0e+00): v² = c₁ + 4c₂ = 0.980 — the unit built into the CHM
normalisation, *not* a recovered velocity (see "What was circular") — and m = 0 (the
residual doubles within Δc₁ = ±0.013, Δm = 0.010) with r = 0.0064, 3.4× *below* the
nearest-neighbour chain's 0.0217; the coupling range is **not** recovered — along c₂ at v = 1 every
value in [−0.4, 0.22] is within a factor 2 of the minimum (valley width 0.61), and a second
low-mass start lands at c₂ = +0.23. Two of four starts (m₀ ≥ 0.4) are captured by the gapped
shoulder at the m = 1 bound (r = 0.054): along m the residual has a secondary dip at m ≈ 3–4
(r = 0.12, ball entropy < 1 % of the massless value), which Jacobson's premise that the UV
entanglement is that of the gapless vacuum excludes; here it is the mass bound. Two structural
limits are measured, not assumed: reflection-even variations have r = 3.07, 2.83, 2.15 (L = 8, 16, 32)
for the wave chain itself — the 1+1 massless scalar's zero mode, whose entropy no local energy
density can see — and local changes of the couplings are cutoff-dominated variations that the
toy does not use. **(5)** The ~1.2–1.4/L² floor (1.2 from the improved chain, 1.39 from the
nearest-neighbour one) is a property of the *discretisation chosen*, not a bound: putting β at the bond midpoints (the midpoint rule for the gradient energy, which also
makes the bonds straddling ∂B drop out — their midpoint sits exactly on the boundary) lowers the
massless chain's residual from 1.39/L² to **0.053/L²**, a factor 26 at L = 8 and 24 at L = 16
(a Richardson mix 0.1·site + 0.9·bond gives 0.09/L²). **(6)** The k⁴ coefficient is now
resolved, but by breaking a degeneracy rather than by lowering the floor: within the
(c₁, c₂) family at v = 1 the k⁴ dispersion coefficient A₄ = Σ cₙ n⁴ = 1 + 12c₂ is *affine* in
c₂, so the dispersion and any range-dependent discretisation error are exactly degenerate.
Adding the third-neighbour coupling (c₁ + 4c₂ + 9c₃ = 1, A₄ = 1 + 12c₂ + 72c₃) makes
∂ρ/∂c₃ = 6 ∂ρ/∂c₂ a falsifiable requirement for a pure-dispersion response. Over a 31-point
(c₂, c₃) grid the site-centred ansatz **passes** it — 6.043 ± 0.086, 5.989 ± 0.265,
6.052 ± 0.093 (λ = 4L, 8L, 16L; mean ± sd over L = 8, 16, 32) — so the residual's coupling
response is a function of the k⁴ dispersion alone, with sensitivity
C ≡ ∂ρ/∂(ω(π/L)/(π/L) − 1) = **−0.974 ± 0.034 (λ = 4L), +0.151 ± 0.014 (8L),
+0.461 ± 0.021 (16L)**; for it, fitting in A₄ alone is as good as the free two-parameter fit
(rms ratio 1.00–1.03 over all nine (L, λ) cells). The bond-centred ansatz *fails* it
(2.221 ± 0.161, 1.951 ± 0.097, 1.906 ± 0.082; rms ratio 4.0–32.9): its lower floor is bought
with a range-dependent artefact, and at A₄ = 0 with a second coupling switched on its floor
(+6.6e-03 at L = 8, λ = 4L) is no longer far below the site form's (−1.26e-02). **(7)** A ball
**attached** to the Dirichlet wall has one entangling point, and with the exact boundary-CFT
weight β(y) = (B² − y²)/(2B) (y from the wall) the residual is 2.25e-02, 7.47e-03, 3.29e-03 at
L = 8, 16, 32 — at the longest wavelength 3.0e-03, 5.1e-04, 2.5e-04, i.e. the centred ball's
odd-sector 8.4e-04, 2.6e-04, 3.3e-04 to within a factor 3.6, 1.9 and 0.8:
**r stays consistent at a second position**, but only with
the boundary-adapted weight — the bulk parabola used there gives 0.880, 3.28, 9.93 (39×, 439×,
3015× worse). A ball *detached* from the wall stays obstructed at O(1) even with the
image-corrected local weight (1.02–10.6 at 3 and 9 sites, 0.098–0.59 at 33), as it must: an interval at a
distance from a boundary carries a bilocal term to its mirror image
(Eisler–Tonni–Peschel 2022, Sec. 6). **(8)** Projecting the ball's top Williamson mode out of
the *variation* — both quadratures, in the ball's Williamson frame, so δS and δ⟨K_loc⟩ remain the
same functional of the same δV — removes the zero-mode obstruction from the reflection-even
sector: 2.93, 2.23, 2.07 → **0.0444, 0.0510, 0.167** (L = 8, 16, 32). The even sector does
**not** become consistent with the odd one *in the ratio* — it stays 53×, 194×, 508× larger and
does not decay with L — and Layer 7 unit 2 says why (section "Item (a)" below, superseding the
"near-cancellation of δS" explanation given here before, which is measurably false): the defect
**is** the ball's top even Williamson mode (defect/g(ν₁)w₁ = 1.144, 1.139, 1.158 **at λ = 16L**
only — the same ratio is −0.752, −1.288, −3.738 at λ = 4L, so the identification is a *convergence
in the wavelength*, |ratio − 1| falling monotonically 1.75 → 0.99 → 0.14 at L = 8 and 4.74 → 0.49 →
0.16 at L = 32, not one number over the family), it is an IR
effect (ν₁ 0.934 → 1.199 and r_even 2.04 → 3.60 as N goes 200 → 3200 at fixed L = 8, while r_odd
falls), and projecting the softest even modes drives the **absolute** defect 1.34e-2 → 5.3e-10 —
which is *not* an accuracy statement, because the same projection shrinks the variation itself
(‖δV_B‖_F by up to 10⁶): **per unit variation** the projected even sector stays **1.9×–659×
worse** than the odd sector, matched cell by cell, and is never better (retraction, fix round 2). **(10)** In **2+1** (N × N square lattice, disks R = 3…10, N = 128)
the canonical CHM ansatz fails flat — r = 0.36–0.45 with δ⟨K⟩/δS = 0.549–0.676 — because in d ≥ 3
the CFT stress tensor carries the conformal improvement −ξ∇²(φ²) that vanishes identically in
d = 2; fitting it recovers **ξ = 0.1278 ± 0.0040** against (d−2)/(4(d−1)) = **1/8** (3.1 % scatter,
against 7.7–32.9 % for three wrong operator shapes), after which r = 0.005–0.075, the free radial
weight returns the CHM parabola at overlap 0.992–0.999 and normalisation 0.78–0.99 (1.60–1.67
without the improvement), z = 2 Lifshitz stays at 0.96–1.26, and — the decisive control for item
(a) — the reflection-even sector agrees with the odd one to a factor 1.8–24 (median 3.6) against
3476/8481/6297 in 1+1. **(9)** With the weight free (Sec. G): on the extended
variation family (all reflection-odd modes of wavelength ≥ 4L, singly and in pairs — 1275, 325,
78 variations at L = 8, 16, 32, N = 1600 — whose numerical rank is 4, 6, 7–8 at 10⁻⁴, 10⁻⁶,
10⁻⁸) the CHM weight gives r = 9.1e-04, 2.8e-04, 3.3e-04 (bond-centred) for the wave chain and
the free three-mode nonnegative weight r_free = 4.1e-05, 1.2e-05, 2.5e-06 with the CHM parabola
as minimiser (two modes, constant + parabola, already give 4.7e-05, 1.2e-05, 6.8e-06 at
(1.001, −1.0005)·πR/2); for the Lifshitz chain r = 1.0–1.2 and, with three modes, r_free = 0.270, 0.507, 0.679 — but
with four modes 7.0e-04, 6.7e-03, 8.2e-02: a per-ball nonnegative weight *exists* at fixed L
and degrades 100× from L = 8 to 32 (the signed, unphysical three-mode minimum is 1.1e-03, 1.0e-03, 3.2e-04, reached with a weight 10, 31, 84×
the CHM scale that dips negative near the edge; the per-site weight with L/2 + 1 parameters
reaches 4.5e-05, 2.5e-06, 1.6e-06 for Lifshitz *and* 2.3e-06, 6.1e-07, 1.1e-07 for the wave
chain — no teeth, as its parameter count is at the family's rank); one profile shared across
L = 8, 16, 32 gives 1.5e-04 (three modes, z = 1, overlap 1.000 at every L) and 2.6e-05 (eight
modes) for the wave chain against 0.92 (three modes, z = 1), 0.74 (z = 2) and 0.091 (eight
modes, z = 2) for Lifshitz — but the Lifshitz floor keeps falling with the mode count (the
ladder below): 0.023 at 11–12 Chebyshev modes (z = 2), 5.5e-03 at 11 (z = 3), 6.9e-04 at 12 hat
modes (z = 2), below 1e-02 by 11–13 modes and below 1e-03 by 12–14, z-dependent; 5e-04 only by 20 Chebyshev modes, so a scale-covariant nonnegative weight
*exists* for Lifshitz; the statement that survives is the mode-count gap (3 vs ≥ 11–12). Across the family (L = 8, 16, 32): z = ½ 5.3e-02, 4.8e-02, 3.9e-02, z = 3/2 0.14, 0.16, 0.12,
1/n² couplings 5.9e-02, 4.5e-02, 2.2e-02, 1/n³ 1.7e-02, 1.2e-02, 6.5e-03 stay bounded away from
zero (shared profile 0.38, 0.078, 0.39, 0.10); c₂ = +0.1 1.9e-03, 4.6e-04, 9.8e-05, the improved
Laplacian 7.9e-05, 5.6e-05, 2.3e-05, random mirror-symmetric bonds (W = 0.3) 5.7e-05, 2.5e-05,
3.4e-04 sit with the wave chain; v = ½ gives *identical* numbers to v = 1 with the weight scaled by exactly 2 (the
velocity is a unit); m = 0.2 gives 8.2e-06, 1.9e-06, 5.3e-06 per ball with the weight rescaled
by 1.17, 1.33, 1.47 and the quartic mode growing to 0.27 — the shared profile sees it at
2.7e-02 (best of up to eight modes; 2.2e-02 at m = 0.05, 9.7e-04 at m = 1; and m = 0.01,
mL = 0.08–0.16, at 2.1e-03 against 6.3e-05 massless with three modes). The constrained fit
over (c₁, c₂, m) from the same generic starts as anchor 4: with the per-ball free weight nothing
is pinned — landings (1.24, −0.11, 0.43) and (4.0, −0.015, 0.007) at r_free = 2.3e-06 and
5.0e-07, both *below* the wave chain's own 3.7e-05; with the shared profile the low-mass start
lands at m/v = 2.3e-03, c₂/c₁ = −0.0076 (r = 2.4e-05, the velocity flat by construction) and the
massive start is captured by the gapped shoulder at the m = 1 and c₁ = 0.05 bounds, m/v = 5.5
(r = 4.3e-04, 18× above). The
two-mode squeezes obey the exact first law by finite differences at ≤ 2.5e-09. **Status: a
lattice statement with no experimental input (every number is computed by `notebook.py` from
the package); the full known-results suite passed on the producing tree (see "The producing
build"), so the claim is admissible per papers/README.md subject to the caveat stated there.**

---

## The producing build

| item | value |
|---|---|
| generator | `papers/toy-jacobson/notebook.py`, entry point `main()` |
| generator sha256 | `818814015c752ea78854c2e3ea59b1dd3c7b6292741ae509d0271c7b858912c1` |
| produced (UTC) | 2026-09-05T04:13:14Z (whole notebook re-run with Section G including the mode ladder G6; the A–F numbers and the G1–G5 numbers are unchanged to the digits quoted) |
| git HEAD | `3af08227fdbad2c5da9592e005b94d6feb0fff96` (working tree dirty: this candidate, `vacuum/geometry/jacobson.py`, `tests/test_toy_jacobson.py` and concurrent work by other agents are uncommitted; nothing is committed by this candidate) |
| package | vacuum 0.1.0 (editable) |
| environment | Python 3.13.3, numpy 2.5.2, scipy 1.18.1, mpmath 1.4.1, macOS-14.6.1-arm64-arm-64bit-Mach-O |
| command | `.venv/bin/python papers/toy-jacobson/notebook.py` |
| wall clock | 327 s on an idle machine (budget 40 min; an earlier run under concurrent load took 1418 s); Section G alone 115 s; `--quick` 39 s |
| pytest, this candidate | **32 passed in 22.6 s** (`tests/test_toy_jacobson.py` 29, `tests/test_geometry_jacobson.py` 3), on the producing tree |
| pytest, full suite | **not run by this pass** (instruction: only the two owned test files were run; no claim is made here about the state of the full suite). The line below is the earlier record. Admissibility per papers/README.md for the anchor-9 numbers awaits a full-suite run on a quiescent build; nothing here has moved into a manuscript. |
| pytest, full suite (earlier record) | **974 passed, 5 warnings in 509.79s (0:08:29)** (`.venv/bin/python -m pytest -q`, 2026-09-03, on the uncommitted working tree in the same window as the A–D data production). Caveat: the tree was uncommitted and being edited concurrently by other agents, so the run is to be repeated on a quiescent, committed build before any number moves from `data/` into a manuscript. |

**Integration note, 2026-09-04.** The full suite was re-run on this tree (`.venv/bin/python -m pytest -q`): **9 failed, 1136 passed in 2925.6 s (48:45)**, on a tree being edited concurrently by other agents. Eight of the nine belong to packages this candidate does not own — `tests/test_exports.py` for `vacuum.composite`, `detectors`, `floquet`, `hardware`, `inequalities`, `opt` (public names not re-exported) and its blocked-extras import check, plus `tests/test_circuit_mapping.py`. The ninth, `test_submodule_all_is_subset_of_package_all[geometry]`, **was this candidate's own**: `k4_sensitivity` was in `vacuum.geometry.jacobson.__all__` before it was re-exported from `vacuum/geometry/__init__.py`, and the suite reached that test in the window between the two edits. It is fixed — a re-run of `tests/test_exports.py` after the fix leaves 7 failures, none of them `[geometry]` or `[modular]`. Evidence for this candidate on the same tree: **58 passed in 147.7 s** (`tests/test_modular_lab.py` 32, `tests/test_toy_jacobson.py` 23, `tests/test_geometry_jacobson.py` 3) plus `tests/test_first_law.py`, and both notebooks' `--quick` paths exit 0. **Admissibility per papers/README.md required the full known-results suite to be re-run green on a quiescent build** before any number here moves from `data/` into a manuscript; that run is the Integration line below.

The sha256 is embedded in every archive under `data/` (`__build__` key) and in
`data/build_info.json`.

---

**Integration: full suite 1201 passed on the quiescent tree at HEAD `c18a206`, 2026-09-04** (`.venv/bin/python -m pytest -q`, twice in a row on the final tree: 1201 passed, 10 warnings in 731.91s (0:12:11) and 1201 passed, 10 warnings in 740.37s (0:12:20); with tenpy, cvxpy, jax, jaxlib, qiskit, qiskit_aer and qiskit_ibm_runtime blocked at the import system, 1026 passed, 76 skipped in 428.84s (0:07:08); all 15 notebook `--quick` paths exit 0, the slowest in 307 s). The tree is commit `c18a206` plus this pass's uncommitted edits — README, STATUS, PRIORITY, MANIFEST and draft prose reconciling the rollups to the MANIFESTs, plus one print in `interacting-vacua-first-numbers/notebook_qei_exact.py` made None-safe so its `--quick` path exits 0 (no numerical code, no data). `data/` was produced on the build recorded above; this run is the admissibility record under `papers/README.md` for the candidate's code as it stands on that tree, and a full re-run of `notebook.py` there regenerates `data/` with a new `generator_sha256`; the producing-build fields above are left as recorded.

## Modules used

| module | what it does here |
|---|---|
| `vacuum.geometry.jacobson` (`coupling_chain_K`, `random_symmetric_chain_K`, `dispersion_power_K`, `bulk_dispersion`, `dispersion_departure`, `local_modular_form`, `bw_weights`, `squeeze_modes`, `squeezed_ball_covariance`, `equilibrium_residual`, `first_law_audit`, `eh_profiles`, `fit_dynamics`, `residual_scan`, `k4_sensitivity`; and, for anchor 9, `local_pieces`, `pair_variation`, `smooth_variations`, `weight_family`, `free_weight_residual`, `universal_weight_residual`, `fit_dynamics(residual='free' \| 'universal')` — importable from the module; not added to `__all__` because `vacuum/geometry/__init__.py` is outside this candidate's ownership and `tests/test_exports.py` requires the two lists to agree) | the toy: models, the CHM/BW local form from the PSD bond-split energy density, the long-wavelength odd-parity squeeze variations, the residual with per-mode floor diagnostics and mpmath escalation, the independent-route first-law audit, the Di Giulio–Tonni profile cross-check, the constrained fit; and, added for this pass: the discretisation options of the CHM ansatz (β at site centres / bond midpoints / Richardson mixes, `shift`, and the boundary-adapted `wall` weight), balls at arbitrary positions, the symmetric zero-mode projection (`zero_mode='project'`), and the (c₂, c₃) degeneracy test `k4_sensitivity`; and for anchor 9 the local energy pieces of the bond split, the first-order two-mode squeezes, the long-wavelength variation family with its numerical rank, the weight families (even Chebyshev at sites / bond midpoints, per site + boundary bond), the minimax LP (scipy HiGHS) with nonnegativity, the shared-profile residual, and the free-weight constrained fit |
| `vacuum.geometry.jacobson_continuum` (`chm_beta`, `modular_coordinate`, `modular_coordinate_inverse`, `modular_symplectic_eigenvalue`, `entropy_weight`, `modular_wavepacket`, `continuum_first_law_defect`, `conformal_killing_defect`, `bond_split_weight_coefficient`, `bond_split_discretization_prediction`; new 2026-09-12, exported from `vacuum/geometry/__init__.py`, `tests/test_exports.py` green) | the continuum companion to the lattice residual (`draft/proof_continuum_limit.md`): the one-particle modular generator −2πi(β∂ₓ + β′/2) of an interval with its exact spectrum 2πs and ν(s) = coth(πs)/2, the certified continuum first law dS − d⟨K⟩ for an explicit Gaussian squeeze (mpmath, no lattice code), the conformal-Killing shape test ∫w ξ‴ = 0 ⟺ w‴ = 0, and the closed-form O(a²) weight-placement coefficient A(κ) of the site- vs bond-centred CHM ansatz |
| `vacuum.geometry.jacobson2d` (`sine_basis`, `square_lattice_K`, `lattice_coords`, `disk_ball`, `chm_beta_2d`, `vacuum_block_2d`, `squeeze_modes_2d`, `local_modular_form_2d`, `equilibrium_residual_2d`, `first_law_audit_2d`, `free_weight_residual_2d`, `XI_IMPROVE_3D`; new 2026-09-12, exported from `vacuum/geometry/__init__.py`, `tests/test_exports.py` green) | the 2+1 extension (section J): the N × N Dirichlet square lattice (z = 1 wave, z = 2 biharmonic Lifshitz) as a sparse K with its analytic product-mode spectrum, disks of radius R with the CHM weight β(r) = (R² − r²)/(2R), the disk's vacuum covariance assembled from the analytic modes (no N² × N² eigh), reflection-odd/-even long-wavelength squeezes, the local form on the lattice's own PSD bond split with β at the site centres or the bond midpoints, the conformal improvement term in its weight-side and field-side discretisations, the residual with the same floor diagnostics, the finite-difference first-law audit, and the free radial-Chebyshev weight by minimax LP |
| `vacuum.core` (`entanglement_hamiltonian`, `williamson`, `symplectic_inverse`, `modular_energy`, `entropy`, `ground_state_cov`) | the exact Gaussian modular Hamiltonian (ground truth for K_ball) and entropies |
| `vacuum.core.precision` (`entropy_precise`, `symplectic_margins_mp`) | the float64 → mpmath ladder used by the audit and by the floor escalation |
| `vacuum.modular` (`locality_score`) | read-only cross-check of the exact modular Hamiltonian's bandedness |
| **No `experiments/` files** | published inputs are analytic and cited at their point of use: Jacobson 2016 Eqs. (14)–(19); Casini–Huerta–Myers 2011 Sec. 2; Bisognano–Wichmann 1976; Di Giulio–Tonni 2020 Eq. (20); Casini–Huerta 2009 (the 2d scalar's zero mode); Blanco–Casini–Hung–Myers 2013 (first law) |

---

## Data

| file | contents |
|---|---|
| `data/build_info.json` | the build |
| `data/A_massless_chain.npz` | anchor 1: r per ball size and per wavelength ratio (odd), the even-sector residuals, power-law fits and extrapolations, every (ball, variation) record with floor diagnostics, the exact-EH profiles (pp row sums, r²-resummed xx weights, zero-mode row sums, locality scores) at L = 16, 32, the constrained velocity v*(L) |
| `data/B_first_law_audits.npz` | anchor 2: dS by finite differences vs the modular route, float64 and ladder, nine dynamics × L = 4, 8, 16 |
| `data/C_residual_map.npz`, `data/C_residual_map.json` | anchor 3 / the map: 44 dynamics × L = 8, 16, 32 — r (max and rms), δ_v1, δ_shape, locality, pp row sum at the centre, the fit r = A/L² + C δ and the per-family tracking coefficients; every (dynamics, ball, variation) record |
| `data/D_constrained_dynamics.npz` | anchor 4: the four landings, the 1D scans along c₁, c₂, m around the best landing, the c₂ valley at v = 1, the gapped branch r(m) with the L = 8 ball entropy |
| `data/E_ansatz_and_k4.npz` | anchors 5–6: the residual of four discretisations of the CHM ansatz on the massless chain (site-centred β, bond-centred β, two Richardson mixes) at L = 8, 16, 32; and, over the 31-point (c₂, c₃) grid at v = 1, the fits ρ = ρ₀ + λ₂c₂ + λ₃c₃ and ρ = ρ₀ + λ_A A₄ per (ansatz, L, wavelength) with l₃/l₂, both rms, C and the floor at A₄ = 0 |
| `data/F_second_position_even.npz` | anchors 7–8: r for the wave and Lifshitz chains under reflection-odd/even/parity-blind variations with the zero mode excluded by parity vs projected out of the variation; and, for balls attached to the wall, 3/9/33 sites from it, and centred, the residual with the boundary-adapted weight against the bulk parabola |
| `data/G_free_weight.npz` | anchor 9: per dynamics × L the archived r, the CHM residual on the extended variation family (site / bond), r_free for every (family, discretisation) with parameter counts and the family's rank, the signed minima, the recovered Chebyshev coefficients / overlap / normalisation, the shared-profile residuals for every (family, discretisation, z), the recovered site weights for the wave, Lifshitz and m = 0.2 chains against CHM, the free and universal fits' landings and invariants (m/v, c₂/c₁), the v / m / c₂ scans under both residuals, and the two-mode first-law audit |
| `data/H_continuum_check.json` | the Layer-7 continuum companion (`notebook_continuum.py`, its own `__build__`): H1 the certified continuum first law + its mutations, H2 the conformal-Killing shape test + its mutations, H3 the nine (L, λ) bond-split cells at N = 2400 with `A_site_measured`, `A_bond_measured`, `A_measured`, `A_continuum`, `A_discrete`, `A_exact`, `bond_share_of_site`, `rel_err_site_only`, `rel_err_discrete`, `rel_err_continuum`, and the top-level `max_rel_err_{exact,discrete,continuum,site_only}` + `min_rel_err_{continuum,site_only}` |
| `data/I_even_sector.npz`, `data/I_even_sector.json` | Layer 7 unit 2 item (a) (`notebook_even.py`, its own `__build__`): I1 the even-sector defect against the ball's top Williamson-mode contribution **at all three wavelengths** (`defect_over_zero_mode_by_ratio`, `defect_over_zero_mode_dev_from_one_by_ratio`, `top_mode_share_even_min_lambda16L`, `top_mode_share_even_min_all_ratios`) plus the (N, L) scan of ν₁, r_even and r_odd; I2 the projection ladder n = 0…6 with `r`, `abs_defect_max`, `abs_dS_min`, the odd-sector row and (fix round 2) the scale-invariant `defect_per_dV` = \|δS − δ⟨K⟩\|/‖δV_B‖_F, `dV_B_fro`, `even_over_odd_per_dV_by_ratio` and `summary_per_unit_variation`; I3 the precision controls (`dS_finite_difference`, `dS_mpmath_margins`, `cancellation`, `term_by_term_rel_gap`, `digits_lost`); I4 the (L, m) scan; I5 the 2+1 cross-reference |
| `data/J_2p1_residual.json`, `J_2p1_improvement.json`, `J_2p1_even_odd.json`, `J_2p1_free_weight.json`, `J_2p1_controls.json`, `J_2p1_sector_split.json`, `J_2p1_summary.json`, `J_2p1_summary.npz`, `J_2p1_build_info.json` | Layer 7 unit 2 item (b) (`notebook_2p1.py`, its own `__build__`): J1 r(R) for wave/Lifshitz × odd/even × ξ = 0, 1/8 × β at sites/bond midpoints, every row with its floor diagnostics; J2 the per-cell improvement fit `xi_fit` with `dS`, `dK_canonical`, `deficit`, `phi2_response` and the three wrong-shape coefficients `alt_coefficients` behind `shape_identifiability`; J3 the even/odd comparison with ν_top; J4 the free radial-Chebyshev weight (coefficients, overlap, normalisation, rank profile); J5 the first-law audits, the N-convergence and the two improvement discretisations; J6 the quadrature (xx/pp) split of the canonical pairing per (R, wavelength) — `dS_xx`, `dS_pp`, `dK_xx`, `dK_pp`, `pp_ratio`, `xx_ratio`, `dS_pp_over_dS` — for both parities, and the disk's pp weight against 2πβ under two named statistics (`diag` and `rowsum`, each with `slope_lsq`, `sum_ratio`, `median_ratio`, `max_abs_dev` and the interior r < R − 1 versions) plus `offdiag_over_diag`; `J_2p1_summary.npz` carries the headline arrays (r per R for every dynamics/parity/ξ, the per-cell ξ_fit and δ⟨K⟩/δS, the even/odd ratios, ν_top, and the N-convergence ladder) |
| `data/summary.json` | the headline numbers quoted above (`anchor7_free_weight` for anchor 9) |

---

## Anchors — measured vs target (all also permanent tests in `tests/test_toy_jacobson.py`)

| anchor | target / tolerance | measured (notebook, full) |
|---|---|---|
| 1 r(L) decreases toward the CHM form, wave chain | strictly decreasing; exponent in (−3, −1.5) | 9.1e-02 → 3.9e-02 → 2.2e-02 → 9.6e-03 → 5.3e-03 → 2.3e-03 → 1.2e-03 → 3.4e-04 for L = 4, 6, 8, 12, 16, 24, 32, 48; exponent -2.27 (ratios 4/8/16: -2.03, -2.42, -2.27) |
| 1 continuum extrapolation (now with a derivation: `draft/proof_continuum_limit.md`, 2026-09-12) | \|r_∞\| < 5e-4 (the scatter of the two fits). The **1/L² form is the one the derivation supports** (the leading ansatz error is exactly O(1/L²) at fixed λ/L, Prop. 4.3 there); the free-power fit is a robustness check on the fitting procedure, not an independent continuum estimate | -1.9e-04 (1/L²; its fitted amplitude 1.406 sits within 1–3 % of the derived coefficient of the ansatz-to-ansatz **difference** ρ_site − ρ_bond, 1.425 (closed form) / 1.447 (exact identity) at the dominant wavelength, and within 1 % of the *measured site* coefficient 1.3936 at L = 8 — the derivation accounts for the difference, and the gap to the site coefficient is the **underived** bond floor, 3.8 % of it at L = 8 rising to 13.6 % at L = 32), +4.6e-04 (fitted power). r_∞ = 0 itself is **not proved**: ASSUMPTION A5 of the proof |
| 1 odd variations blind to the zero mode | top Williamson mode share < 1e-12 | 1.5e-17 |
| 1 even variations obstructed | r > 0.3 | 3.07, 2.83, 2.15 |
| 2 exact first law, every dynamics, independent routes | 1e-8 | 8.6e-09 (ladder); float64 1.3e-07 (Lifshitz) |
| 3 Lifshitz bounded away from 0 and non-decaying | r > 0.9, ratio to wave > 20 | 0.954, 0.974, 0.991; ratios 44, 185, 911 |
| 3 ω ~ √k, 1/n² couplings | > 1, > 0.1 | 4.3–9.9; 0.16–0.54 |
| r tracks departure: velocity | \|r − \|1 − v\|\| < 0.03 | v = 0.5/0.8/1.25 at L = 32: 0.5000/0.2001/0.2514 |
| r tracks departure: map fit | — | r = 0.785/L² + 0.891 δ_v1, scatter 0.24, 71 points |
| 4 constrained dynamics from a generic start, CHM weight assumed (v² is the normalisation's unit, withdrawn as a velocity measurement — anchor 9) | \|v² − 1\| < 0.05, m < 0.02, \|c₂\| < 0.35, r ≤ 1.05 r_wave | v² = 0.980, m = 0.0e+00, c₂ = -0.163, r = 0.0064 vs 0.0217 |
| 5 bond-centred β lowers the floor | A = rL² in (1.2, 1.6) site, < 0.10 bond, ratio > 15 | 1.394, 1.364 (site); 0.053, 0.058 (bond); 26×, 24× |
| 6 pure-dispersion test l₃/l₂ = 6, site ansatz | \|l₃/l₂ − 6\| < 0.8, rms(A₄ only)/rms(3 par) < 1.2 | 6.043 ± 0.086 (λ = 4L); ratio 1.02–1.04 |
| 6 k⁴ sensitivity at λ = 4L | C ∈ (−1.15, −0.75) | −0.974 ± 0.034 (L = 8, 16, 32) |
| 6 bond ansatz fails the same test | l₃/l₂ < 3.5, rms ratio > 2 | 2.221 ± 0.161; 2.3–6.1 |
| 7 ball attached to the wall, boundary-CFT weight | r < 0.03 (L = 8), < 0.012 (16); parabola > 30× worse | 2.25e-02, 7.47e-03; 39×, 439× |
| 7 detached ball stays obstructed | r > 0.3 | 1.30, 1.86 (9 sites from the wall) |
| 8 zero mode projected out of the variation (even sector) | 2.93, 2.23 → < 0.10; still > 20× the odd sector | 0.0444, 0.0510; 53×, 194× |
| 8′ **item (a)**: the even-sector defect IS the ball's top (even) Williamson mode, *in the long-wavelength limit* | defect / (g(ν₁)·w₁) within 10–20 % of 1 at λ = 16L, every L; \|ratio − 1\| falling monotonically in λ at every L | **1.144, 1.139, 1.158** at λ = 16L (L = 8, 16, 32, N = 1600); **−0.752, −1.288, −3.738** at λ = 4L and **1.987, 1.596, 1.495** at λ = 8L, i.e. \|ratio − 1\| = 1.75/0.99/0.14, 2.29/0.60/0.14, 4.74/0.49/0.16 |
| 8′ the even obstruction is IR, not UV | ν₁ and r_even strictly increasing in the chain length N at fixed L; r_odd not | ν₁ **0.934 → 1.199** and r_even **2.04 → 3.60** for N = 200, 400, 800, 1600, 3200 at L = 8; r_odd 1.9e-3 → 8.3e-4 |
| 8′ absolute defect vs the ratio | \|δS − δ⟨K⟩\| falls > 10⁴ from 0 to 5 projected modes while ‖δV_B‖_F falls > 10³ with it; the ratio stays in [1e-2, 2e-1] | **1.34e-2 → 6.7e-5 → 2.8e-7 → 5.3e-10** (L = 8) with ‖δV_B‖_F **9.2e-3 → 4.5e-8**; r = 0.044, 0.027, 0.048 |
| 8′ **the sector comparison, per unit variation** (replaces the retracted absolute comparison) | \|δS − δ⟨K⟩\|/‖δV_B‖_F, matched cell by cell: the projected even sector never below the odd sector | **1.9×–659×** worse (projected, n = 5); **96.6×–369.4×** worse unprojected; odd itself 6.6e-4, 1.0e-4, 6.1e-5 at L = 8, 16, 32 |
| 8′ the top-mode share of the even variations is *not* ~1 over the family | minimum over **all** even rows, not only λ = 16L | **0.2119** (L = 32, λ = 4L) against **1.0096** (minimum over λ = 16L rows) |
| 8′ not a precision artefact (cause (ii) refuted) | mpmath gap < 1e-8, finite-difference gap < 1e-4, Σ\|c_k\|/\|δS\| < 1.01, term-by-term = subtraction to 1e-8 | **2.1e-11, 1.5e-6, 1.000014**, 2.0e-11 |
| 8′ the (L, m) scan is confounded (honest negative) | r_odd rises above 0.29 at m = 1 while r_even stays > 1.6 at every mass | r_odd 4.6e-4 → 0.33; r_even 1.64–53.3; ν₁ 1.14 → 0.5005 at mL = 96 |
| 10 **item (b)**: the 2+1 canonical CHM ansatz fails | r > 0.30 at every R and no decay with R | **0.357–0.451** (R = 3…10, N = 128); δ⟨K⟩/δS = **0.549–0.676** |
| 10 the 2+1 deficit is the conformal improvement, and ξ is recovered | fitted ξ within 0.008 of (d−2)/(4(d−1)) = 1/8, both parities | **0.1278 ± 0.0040** (odd, R ≥ 4, 15 cells, range [0.1204, 0.1336]); **0.1322 ± 0.0043** (even) |
| 10 the improvement's *shape* is identified, not just its coefficient | the ∇²β shape's relative scatter over the 15 cells beats every wrong shape by ≥ 2× | **3.1 %** against 32.9 % (β-weighted), 22.9 % (rim only), 7.7 % (field-side Laplacian, and its coefficient −0.140 has the wrong sign) |
| 10 with ξ = 1/8 the 2+1 residual collapses | < 0.03 for R ≥ 4, ≥ 5× below the canonical form at every R | **0.0131, 0.0102, 0.0125, 0.0239, 0.0052** (R = 4, 5, 6, 8, 10); 0.0749 at R = 3 |
| 10 Lifshitz z = 2 stays obstructed in 2+1 | canonical > 0.90 at every R; with the CFT ansatz it grows with R; ratio to the wave lattice > 30 at R = 8, 10 | **0.955–0.986**; 0.117 (R = 4) → **1.264** (R = 10); **35×, 244×** |
| 10 the free radial weight recovers the CHM parabola in 2+1 | overlap > 0.99 at two even Chebyshev modes; normalisation within 30 % of 1 *with* the improvement | overlap **0.992, 0.997, 0.999**; normalisation **0.78, 0.80, 0.99** (R = 4, 6, 8) against **1.67, 1.60, 1.66** canonical; r_free = 4.2e-4, 5.0e-4, 1.9e-4 |
| 10 the quadrature split of the 2+1 pairing (archived in fix round 1) | pp within 10 % of 1 and xx below 0.70 for R ≥ 4, with the pp sector's share of δS recorded | pp **0.907–1.019**, xx **0.636–0.680** (R ≥ 4); **0.700–0.704** / **0.561–0.569** at R = 3; δS_pp/δS ∈ **[−0.133, −0.006]** |
| 10 the pp weight of the exact K_B: diagonal vs row sum (**retraction pin**) | the diagonal slope drifts > 0.25 from 1 across R and falls monotonically; the row-sum slope stays within 14 % (9 % for R ≥ 4) and > 2 % | diag **1.106 → 0.743** (R = 3 → 10), monotone; rowsum **1.021–1.138**; pp off-diagonal mass **0.149 → 0.739** |
| 10 **the decisive even/odd test** (item (a) settled in 2+1) | even/odd < 30 with median < 10, against > 2000 in 1+1 | **1.8, 2.1, 3.0, 4.3, 6.8, 24.2** (R = 4, 3, 8, 5, 6, 10; median 3.6) against 1+1's **3476, 8481, 6297** (L = 8, 16, 32) |
| 9 free weight recovers CHM on the wave chain | r_free ≤ r_chm for every family containing CHM; r_free < 1e-4 and < 0.05 r_chm; Chebyshev coefficients / (πR/2) within 0.01 of (1, −1, 0); overlap > 0.9999; normalisation within 1 % | 3.9e-05, 9.7e-06 (L = 8, 16; ratios 0.041, 0.020); (1.0010, −1.0007, −0.0003), (1.0005, −1.0005, −0.0000); 1.000000; 1.0009, 1.0005 |
| 9 the variation family is low-rank; a per-site weight has no teeth | rank(1e-6) ≤ 8; per-site r_free < 1e-4 for wave AND Lifshitz | rank 6, 6 / 5, 5; 2.1e-06, 4.5e-07 / 4.5e-05, 1.9e-06 |
| 9 Lifshitz floors are family properties | cheb4 per ball (nonnegative, below the rank) r_free < 2e-3 at L = 8 and < 2e-2 at L = 16, growing > 3× (so "no local weight" cannot come back); three modes > 0.2, > 0.4 (recorded); eight-mode shared floor > 0.05 (recorded); wave shared < 1e-4, parabola z = 1 < 3e-4 with overlap > 0.999, z = 2 > 0.3 | 6.9e-04, 5.6e-03 (8.1×); 0.270, 0.482; 0.091; 2.6e-05, 1.5e-04 / 1.000, 0.55 |
| 9 the parsimony gap, basis-independent (Chebyshev AND hat; z = 1, 2, 3; L = 8, 16, 32; N = 1600) | wave: 3 Chebyshev modes < 2e-4, 4 hats < 2e-4, and > 0.1 at z ≠ 1; Lifshitz: > 1e-2 for every n ≤ 10, > 1e-3 for every n ≤ 11, in both bases and at every z; AND min(n = 12, 14) < 3e-3 at every (basis, z) — a scale-covariant weight exists; conditions at 1e-3 in [6, 14] | 1.5e-04, 1.5e-04; 0.21–0.86; Lifshitz ≥ 2.3e-02 (n ≤ 10), ≥ 5.5e-03 (n ≤ 11); 5.3e-04–2.0e-03 by 12–14; 8 / 9 conditions |
| 9 two-mode squeezes obey the exact first law | rel_err < 1e-7; first-order δV < 1e-6 | 8.6e-10; 5.9e-07 |
| 9 velocity absorbed, per-ball mass absorbed, shared profile pins m | r_free(v²K) = r_free(K) to 1e-6; per-ball r_free(m = 0.05) < 3× massless; shared r(m = 0.05) > 20× massless; per-ball r_free of a massive wrong-velocity landing < wave chain; universal fit from (1.5, 0.1, 0.05): m/v < 0.05, \|c₂/c₁\| < 0.1, r < 3e-4 | exact; 1.8e-05 vs 3.7e-05; 2.3e-02 vs 6.3e-05; 2.3e-06 < 3.7e-05; m/v = 2.8e-03, c₂/c₁ = −0.013, r = 2.9e-05 |

---

## The proof behind anchor 1's continuum limit (Layer 7, 2026-09-12)

`draft/proof_continuum_limit.md` states and proves the continuum statement that the lattice
residual approximates, and derives the rate. `vacuum/geometry/jacobson_continuum.py` (new;
mpmath, **no lattice code**) carries the two computational steps; `data/H_continuum_check.json`
carries their numbers with build info; `tests/test_toy_jacobson_continuum.py` pins them
(**18 passed in 7.9 s**, including three mutation tests, a nine-cell pin on the archived
bond-split table, and an honesty pin that the site floor is *not* derived to the accuracy of
the difference). Regenerate with
`.venv/bin/python papers/toy-jacobson/notebook_continuum.py` (21 s; `--quick` 6 s, writes a
temp dir). `notebook.py` and `data/A–G` are untouched, so the recorded generator sha256 still
matches the archive.

**What is proved.** (i) The first law δS = δ⟨K_A⟩ from S = −Tr ρ ln ρ, with the regularity it
needs stated, used in the relative-entropy form of Blanco–Casini–Hung–Myers 2013. (ii) The
CHM/BW form of an interval's modular Hamiltonian is *cited*, not reproved
(Bisognano–Wichmann 1975/76; Hislop–Longo 1982; Casini–Huerta–Myers 2011), together with the
1+1 subtlety stated precisely: the theorem is about the **current/derivative** algebra, not
about the algebra generated by the massless scalar φ itself, whose interval reduction carries
the zero-mode/IR term of Casini–Huerta 2009 — and Lemma 2.2 proves that the reflection-odd
variations of anchor 1 are *exactly* blind to that mode (measured top-mode share 1.5e-17),
which is why the odd sector is the one where the theorem applies and the even sector is O(1).
(iii) Hence the **continuum residual is identically zero on every first-order variation** —
no limit is taken — so the lattice number r(K) is a pure discretisation error, and the
free-weight minimiser is β exactly, *provided* the variation family is complete; that
hypothesis is known to fail here (rank 5–6 per ball), which is precisely why anchor 9's claim
is parsimony and not existence.

**What is new (the rate).** An exact lattice identity: the site-centred (trapezoid) and
bond-centred (midpoint) discretisations of ∫βT₀₀ differ, on a squeeze of a lattice normal
mode, by a closed-form finite sum, giving

  ρ_site − ρ_bond = −(1 − ρ_bond)·ΔdK/dK_bond   (Prop. 4.3, no approximation),

whose continuum limit is **A(κ)/L² with κ = πL/λ**, A(κ) = −4C(κ)/D(κ), **A(0⁺) = −3/2**, and
A(π/4), A(π/8), A(π/16) = −0.352, −1.208, −1.427. Certificate at N = 2400, L = 8, 16, 32 ×
three wavelengths: the exact identity reproduces the measured residual to **6.3e-12**, the
discrete form without the (1 − ρ_bond) factor to **8.3e-04**, the closed form to **3.6e-02**.
Consequences. (a) The dominant part of anchor 5's site floor 1.394/L² is *derived* — it is
the trapezoid rule's weight-placement error, not a property of the first law. **Correction
(round-1 fix, 2026-09-12):** an earlier wording of this section said the site floor itself is
derived "to 0.1 %". It is not. The 0.1 % (8.3e-04) figure is the accuracy of the prediction
of the **difference** ρ_site − ρ_bond; the site coefficient is that difference *plus* the
bond floor, which is **not** derived (see below). The unaccounted fraction is |A_bond/A_site|
= **3.8 % (L = 8), 4.2 % (L = 16), 13.6 % (L = 32)** at the dominant wavelength λ = 16 L, and
116 % in the worst cell (L = 32, λ = 4 L, N = 2400, where the N-limited bond floor exceeds the
site coefficient). Archived per cell as `rel_err_site_only` in `data/H_continuum_check.json`
and pinned by
`tests/test_toy_jacobson_continuum.py::test_archived_site_floor_is_not_derived_to_one_part_in_a_thousand`.
(b) The exponent is **exactly −2 at fixed λ/L** — visible in the data as A = −1.4222 (L = 16)
vs −1.4180 (L = 32) at κ = 0.209, and −0.3262 vs −0.3238 at κ = 0.795 — so the fitted
**−2.27 is a fit artefact** of the (L, λ, N) grid: the mode index drifts κ between cells, and
the bond-centred piece is N-limited at large L (+0.053, +0.058, +0.170 at L = 8, 16, 32 on
N = 2400, against 0.052 at N = 6400). Every one of these nine cells (L, λ, κ, A_site, A_bond,
A_measured, A_continuum, A_exact) is pinned by
`tests/test_toy_jacobson_continuum.py::test_archived_bond_split_rows_are_pinned` at 1e-4
relative on the coefficients and 1e-6 on the mode.

**What is NOT proved, and is labelled so in the draft.** ASSUMPTION A4: the O(1/L) remainder
of the continuum limit A(κ) is measured, not bounded — `max_rel_err_continuum` = **3.6e-02**
in the worst cell (L = 8, κ = 0.785) and `min_rel_err_continuum` = **8.4e-04** in the best
(L = 32, κ = 0.209). **Correction (round-2 fix, 2026-09-12):** the draft's status table — the
one place readers are told to quote from — still carried the round-1 conflation in its Cor. 4.4
row, calling this remainder "measured at 0.1 %"; 0.1 % (8.3e-04) is `max_rel_err_discrete`, a
different approximation (only the (1 − ρ_bond) factor dropped, the lattice sums kept exact),
43× smaller. The row now quotes 3.6 % / 0.084 %, the adjacent §4.4 row says Prop. 4.3 predicts
the difference *exactly* (6.3e-12) rather than "to 0.1 %", and two new tests
(`test_the_closed_form_remainder_is_not_the_discrete_one`, which pins both ends of the A4
window and asserts `max_rel_err_continuum` > 30× `max_rel_err_discrete`, and
`test_the_draft_status_table_quotes_the_measured_remainders`, which reads the two draft rows
and requires their percentages to be the archived ones) make the wording unrepeatable. ASSUMPTION A5: that the
*absolute* lattice residual converges to the continuum one — i.e. r_∞ = 0 — is **not proved**;
§4 proves only the size of the difference between two discretisations of the same continuum
object. The bond-centred floor 0.053/L² is **observed only**: the derivation predicts zero at
that order and is silent about the remainder (Euler–Maclaurin boundary term, lattice
dispersion, finite N are listed as candidates; the measured weak wavelength dependence argues
for a boundary term, which is a plausibility argument and is left open).

**Certified numerics, with their mutations.** The continuum first law with the exact CHM
modular Hamiltonian, for three Gaussian squeeze packets (mpmath, dps = 40): δS = 2πs₀ to 20
digits and |δS − δ⟨K⟩|/|δS| = **4.3e-32** (6.0e-27 at dps = 25, the test setting), the residue
being exactly the packet's unphysical s < 0 tail. Mutations: a weight scaled by 1.001 gives
**1.0e-03** (exactly |1 − 1.001|), by 0.995 gives 5.0e-03, and β(1 + 0.05x²) — still
nonnegative, still vanishing at ±R — gives **1.37e-02**. A second, complementary check (the
conformal-Killing shape test, blind to normalisation by construction): ∫_A w ξ‴ = 0 for every
ξ supported inside A ⟺ w‴ = 0; measured 3.2e-47 for the CHM parabola against 2.59e-04
(β + 0.1x³), 2.61e-04 (β + 0.1x⁴), 6.0e-04 (β cosh x).

**Admissibility.** The round-2 fix pass ran `tests/test_toy_jacobson_continuum.py`
(**20 passed, 6.9 s**) and `tests/test_toy_jacobson_continuum.py` +
`tests/test_toy_jacobson.py` + `tests/test_geometry_jacobson.py` + `tests/test_exports.py`
(**83 passed, 66.0 s**) on its own tree, and regenerated `data/H_continuum_check.json` in full
mode (17.6 s, N = 2400; every previously archived number unchanged, two new keys added). It did
**not** re-run the full suite, so no number here moves into a manuscript before the next
quiescent-build run. (Round 1's figures were 18 passed / 63 passed.)

## Item (a), Layer 7 unit 2 (2026-09-12): the even sector is the zero-mode sector, and the ratio is the wrong instrument

Anchor 8 left the reflection-even sector "still inconsistent" and the Limitations bullet said the
leftover was "a near-cancellation of δS". Both are now measured, and the stated *mechanism* was
wrong even though the conclusion (no quantitative content in the ratio) was right. Archive:
`data/I_even_sector.{json,npz}`, generator `notebook_even.py`, N = 1600 unless stated.

**(a1) The defect IS the zero mode — as a limit in the variation's wavelength.** For the
reflection-even squeeze the first-law defect δS − δ⟨K_loc⟩ approaches the entropy contribution
g(ν₁)·w₁ of the ball's **top (reflection-even) Williamson mode** — the lattice's φ̄ — as λ grows,
reaching 14 % at the longest wavelength in the family: the ratio is **1.1443, 1.1388, 1.1575** at
L = 8, 16, 32 **at λ = 16L**, one number within 1.5 % of itself across a factor 4 in ball size.
*It is not that number over the whole family* (correction, fix round 2): the same ratio is
**−0.7516, −1.2879, −3.7377** at λ = 4L and **1.9869, 1.5961, 1.4949** at λ = 8L. What is uniform
is the **convergence**: |ratio − 1| falls monotonically with λ at every L —
**1.75 → 0.99 → 0.14** (L = 8), **2.29 → 0.60 → 0.14** (L = 16), **4.74 → 0.49 → 0.16** (L = 32) —
which is what an identification that holds in the long-wavelength (continuum-CFT) limit and is
contaminated at short wavelength by the lattice's other soft modes looks like. Consistently, the
even variations' share of the top mode is ~1 only at λ = 16L (minimum **1.0096**); over the whole
even family the minimum is **0.2119** (L = 32, λ = 4L), so the short-wavelength even variations
excite the zero mode only partly. The odd sector is exactly blind to that mode (top-mode share
≤ 1e-12, the parity argument of the proof's Lemma 2.2). The λ = 16L row is quoted as the headline
because it is the only rung of the family in which the variation is long compared with every
lattice scale in the ball; the other two rungs are archived and quoted here, not dropped.

**(a2) The obstruction is IR, not UV.** At **fixed** ball size L = 8, as the chain grows
N = 200 → 400 → 800 → 1600 → 3200, the ball's top symplectic eigenvalue grows
ν₁ = **0.9341, 1.0067, 1.0747, 1.1387, 1.1993** (increments per doubling 0.073, 0.068, 0.064,
0.061 — logarithmic, not saturating: this is Casini–Huerta's (1/2) ln ln(1/mL) zero mode,
J. Phys. A **42** (2009) 504007) and the even residual grows with it, **2.039, 2.082, 2.248,
2.927, 3.601**, while the odd residual *falls*, 1.9e-3 → 8.3e-4. A UV/discretisation defect cannot
do this; an IR zero mode must.

**(a3) The ratio is the wrong instrument, and the MANIFEST's stated mechanism was wrong.**
Projecting the ball's softest even Williamson modes out of the variation (n = 0, 1, 3, 5 modes of
the 2L-dimensional Williamson frame; modes 2, 4, 6 are odd and do nothing) drives the **absolute**
defect \|δS − δ⟨K_loc⟩\| from **1.34e-2 → 6.72e-5 → 2.84e-7 → 5.29e-10** at L = 8
(2.92e-2 → 1.51e-8 at L = 16; 6.26e-2 → 2.57e-7 at L = 32), i.e. by seven orders of magnitude.
**That collapse is not an accuracy statement** (correction, fix round 2; see the retraction below):
the projection shrinks the *variation itself* by as much, ‖δV_B‖_F = 9.2e-3 → 4.5e-8 at L = 8 and
1.5e-1 → 4.6e-5 at L = 32, so |δS − δ⟨K_loc⟩| alone cannot be compared with the odd sector's. The
scale-invariant statistic is the defect **per unit variation**, |δS − δ⟨K_loc⟩|/‖δV_B‖_F; matched
cell by cell (same L, same λ) the projected even sector is **1.9×–659× worse** than the odd sector
(unprojected: 96.6×–369.4×) and is **never better** — the same direction the relative residual
already gave. The **relative** residual likewise saturates at
0.016–0.17 — because those same modes carry essentially all of the even sector's δS, so the
projection removes numerator and denominator together. The previously stated reason ("δS is a
near-cancellation") is **measurably false**: Σ_k \|c_k\| / \|δS\| = **1.000014** at worst, i.e. the
per-mode contributions to the projected δS all carry the same sign. **Correction to anchor 8 and to
the Limitations bullet:** the saturation is a projected-denominator effect, not a cancellation
inside δS.

**(a4) Cause (ii) — a numerical artefact — is refuted four ways.** (1) The projected δS from an
independent route, Richardson-extrapolated central differences of the exact entropy along the
projected direction: relative gap ≤ **1.45e-06**. (2) The mpmath rung (symplectic margins at
dps 40): **2.10e-11**. (3) The cancellation ratio above, 1.000014. (4) Assembling δS − δ⟨K_loc⟩
**term by term** in the ball's Williamson frame (per-mode differences g(ν_k)W_kk − A_kk W_kk plus
the off-diagonal remainder, never forming the difference of two O(1) numbers) reproduces the naive
subtraction to **2.0e-11** relative, and the subtraction costs fewer than **3** of float64's 16
decimal digits. There is nothing wrong with the numerics; there is no fix of kind (ii) to make.

**(a5) The (L, m) scan cannot decide it — reported as the negative result it is.** Giving the chain
a mass does gap the zero mode (ν₁ = 1.14 → 0.5005 at mL = 96) but it also destroys conformality, so
the CHM form is wrong in *both* sectors: r_odd rises 4.6e-4 → 0.33 as mL goes 0 → 32 while r_even
stays O(2) at every mass (1.64–53.3 over the whole grid). The even/odd ratio falls from ~3500 to
~5 purely because the denominator rises. **No conclusion is drawn from the mass scan.**

**(a6) The decisive test is 2+1, and it agrees with (i).** In 2+1 the massless scalar has no IR
zero mode. Section J below measures the even and odd sectors there with the same machinery: the
even/odd residual ratio is **1.8–24.2 (median 3.6)** over R = 3…10, and the *same* improvement
coefficient is fitted in both sectors (0.1278 odd, 0.1322 even), against 1+1's
**3476, 8481, 6297**. The disk's top symplectic eigenvalue stays bounded (≤ 0.747) instead of
running with the box.

**Verdict replacing "still inconsistent".** Cause **(i)**, with (iii) as its restatement: the 1+1
even-sector obstruction is the massless scalar's IR zero mode, an O(1) non-derivative piece of the
ball's modular Hamiltonian with no local-energy counterpart, and the reflection-even variations are
exactly the ones that excite it — at the long-wavelength end of the family, where (a1)'s ratio
converges to 1. Removing the soft even modes removes that contribution from δS: the absolute defect
falls seven orders of magnitude. **What it does not do is make the local form accurate there**
(corrected, fix round 2): the projected variations are 10⁵–10⁶ times smaller, and per unit variation
the projected even sector stays 1.9×–659× worse than the odd sector. So the honest verdict is
narrower than the one first written: the even sector's *large* residual (r ≈ 2–3) is the zero mode
and is removed by projecting it out, but the *remaining* even-sector residual (r ≈ 0.016–0.17, and
1.9×–659× the odd sector's per unit variation) is **not** explained here, and the even sector is
**not** shown to be a place where the local CHM form works in 1+1. What can be said is that the
1+1 even sector's dominant obstruction is an IR zero mode absent in 2+1 (a6), and that the relative
residual there is dominated by it. *(The lattice's top Williamson mode is only approximately φ̄ — that is
why the ratio in (a1) is 1.14 and not 1.00 even at λ = 16L, and why the absolute defect keeps
falling past the first projected mode.)*

**RETRACTION (Layer 7 unit 2, fix round 2, 2026-09-12).** Round 1 of this section wrote
*"Once the soft even modes are removed, the local CHM form reproduces the remaining even-sector
variations better than it reproduces the odd ones in absolute terms (5.3e-10 vs 4.8e-6 at L = 8)"*,
and anchors row 8′, the claim paragraph (8) and the Limitations bullet all carried the same
"ending ≥ 10× below the odd sector's own 4.8e-6". **That comparison is retracted.**
|δS − δ⟨K_loc⟩| is not scale invariant, and the projection that produces 5.3e-10 also shrinks the
variation: ‖δV_B‖_F = 9.16e-3, 5.00e-3, 2.66e-3 (odd, λ = 4L/8L/16L, L = 8) against
4.46e-8, 6.48e-9, 1.94e-9 (even, 5 modes projected) — a factor 2·10⁵ to 1.4·10⁶. Normalised, the
conclusion **reverses**: |δS − δ⟨K_loc⟩|/‖δV_B‖_F is **5.27e-4, 6.30e-4, 6.56e-4** in the odd
sector against **1.19e-2, 6.10e-2, 1.50e-1** in the projected even sector at L = 8, i.e. the
projected even sector is **22.5×, 96.7×, 228.6×** *worse*, and over all nine (L, λ) cells
**1.9×–659×** worse, never better. This is the same direction as the relative residual the MANIFEST
quotes elsewhere (0.048 even vs 8.4e-4 odd at L = 8); the two statements were never reconciled
because the absolute one was not a comparison at all. What survives is the *collapse itself* —
projecting the soft even modes removes the zero mode's contribution to δS and drives the absolute
defect down seven orders of magnitude — and nothing about the local form's accuracy in the even
sector. Archived as `defect_per_dV`, `dV_B_fro` and `even_over_odd_per_dV_by_ratio` in
`data/I_even_sector.json` (plus `defect_per_dV_ladder`, `even_over_odd_per_dV_matched` in the
`.npz`), pinned by
`tests/test_toy_jacobson.py::test_per_unit_variation_the_projected_even_sector_stays_worse_than_the_odd`.
The two assertions that pinned the retracted comparison (`got[2] < 3 * odd_abs`,
`got[3] < 0.1 * odd_abs` in
`test_projecting_the_soft_even_modes_kills_the_absolute_defect_not_the_ratio`) are removed and
replaced there by a pin on the ‖δV_B‖_F collapse, which is what that test's absolute ladder
actually measures.

**Re-pin (fix round 2).** The archive field `top_mode_share_even_min` was computed over the
λ = 16L rows only (`notebook_even.py` filtered `ratio == 16`) but named and pinned as if it were the
minimum over the even sector; the test asserted `> 0.5` against it. The field is **removed** and
replaced by two explicitly named ones, `top_mode_share_even_min_lambda16L` = **1.0096** and
`top_mode_share_even_min_all_ratios` = **0.2119**, both pinned. No other archived number moved:
`data/I_even_sector.{json,npz}` was regenerated after the notebook change and every previously
archived value is **bit-identical**.

## Item (b), Layer 7 unit 2 (2026-09-12): the 2+1 toy, and the conformal improvement term

`vacuum/geometry/jacobson2d.py`, `notebook_2p1.py`, archive `data/J_2p1_*.json` (N = 128 box,
disks R = 3…10, reflection-odd "dipole" and reflection-even long-wavelength squeezes, λ/2R = 4, 8,
16, the lattice's own PSD bond-split energy density, β(r) = (R² − r²)/(2R) at the site centres and
at the bond midpoints). The vacuum covariance of a disk is assembled from the analytic product
modes, so no N² × N² eigendecomposition is needed; the first law is verified by finite differences
of the exactly squeezed disk state at **≤ 2.4e-08** for both dynamics and both parities.

**(b1) The canonical ansatz fails in 2+1, and the failure does not go away with R.**
r = **0.451, 0.380, 0.370, 0.357, 0.387, 0.382** at R = 3, 4, 5, 6, 8, 10 — flat, not decaying —
with δ⟨K_loc⟩/δS = **0.549–0.676**. Splitting the pairing by quadrature — archived in
`data/J_2p1_sector_split.json` (section J6; reflection-odd, N = 128, β at the site centres, ξ = 0)
and pinned, after fix round 1 found it quoted here as a live diagnostic — puts the deficit in the
**xx** sector: over R = 4…10, δ⟨K⟩_pp/δS_pp = **0.907–1.019** and δ⟨K⟩_xx/δS_xx = **0.636–0.680**.
At R = 3 *both* are lower (**0.700–0.704** and **0.561–0.569**), so this is a statement about
R ≥ 4, not about the whole R = 3…10 list above. It must also be read with its weight: for a
squeeze δV^pp = −ω u uᵀ is the small piece, and the pp sector carries only **0.6–13 %** of δS
(δS_pp/δS ∈ [−0.133, −0.006] for R ≥ 4). So: the p² part of the ansatz pairs at the ten-per-cent
level and the x-part is a third short — a **non-derivative operator is missing**, which (b2)
identifies and measures.

**RETRACTION (Layer 7 unit 2, fix round 1, 2026-09-12).** Round 1 of this section also wrote
*"the exact modular Hamiltonian's pp diagonal does equal 2πβ_i to 3 % — so the weight is right"*,
with no data file, no test and no named statistic. That sentence is **retracted**. The disk's
exact G_B^pp is *not* diagonal — its off-diagonal mass grows from **0.149** of the diagonal's at
R = 3 to **0.739** at R = 10 — and the least-squares slope of diag(G_B^pp) on 2πβ drifts
monotonically **1.106, 1.049, 1.002, 0.944, 0.864, 0.743** at R = 3, 4, 5, 6, 8, 10, so "to 3 %"
holds only near R = 5 and only for one unstated statistic. What survives is the statement about
the k → 0 pp weight, the **row sum** Σ_j G^pp_ij (the combination `jacobson.eh_profiles` already
uses in 1+1): its slope on 2πβ is **1.138, 1.059, 1.046, 1.021, 1.081, 1.063** over the same
radii — flat in R, and **2–14 %** high (≤ 9 % for R ≥ 4), not 3 %. All of it is archived by J6 and
pinned by
`tests/test_toy_jacobson_2d.py::test_the_pp_weight_of_the_exact_modular_hamiltonian_drifts_with_R`.

**(b2) The missing operator is the free scalar's conformal improvement, and the toy measures its
coefficient.** For a free scalar the CFT stress tensor is Θ₀₀ = T₀₀ − ξ∇²(φ²) with
ξ = (d−2)/(4(d−1)) — **identically zero in d = 2**, which is why the 1+1 toy never saw it, and
**1/8 in d = 3**. Fitting cell by cell
δS = δ⟨K_can⟩ + 2πξ (2/R) Σ_B δ⟨φ²⟩ (the improvement with the integration by parts done on the
**weight**, ∇²β = −2/R inside the disk) gives **ξ = 0.1278 ± 0.0040** (reflection-odd, R ≥ 4, 15
cells, range [0.1204, 0.1336]) and **0.1322 ± 0.0043** (reflection-even) — the conformal value 1/8
to 2–6 %. This is the first place in this candidate where entanglement equilibrium *measures* a
coefficient of the continuum theory rather than checking one.

*Identifiability.* A one-parameter fit is exact in every single cell whatever operator is used, so
what identifies the improvement is the **scatter of the fitted coefficient across the 15 cells**
(R = 4…10 × three wavelengths). The ∇²β shape gives relative scatter **3.1 %**; three wrong shapes
give **32.9 %** (β-weighted φ², coefficient 0.260), **22.9 %** (rim sites only, 0.551) and
**7.7 %** (the field-side Laplacian, coefficient **−0.140**, i.e. a *negative* ξ, excluded for a
free scalar on top of its 2.5× larger scatter). The shape, not only the number, is picked out.

**(b3) With ξ = 1/8 the 2+1 residual collapses.** r = **0.0749, 0.0131, 0.0102, 0.0125, 0.0239,
0.0052** at R = 3, 4, 5, 6, 8, 10 — a factor 5–74 below the canonical form at every radius, and
converged in the box (at R = 5: 1.35e-2, 1.13e-2, 1.04e-2, 1.02e-2 for N = 48, 64, 96, 128). The
residual is **not** monotone in R: with ξ fixed at 1/8 the leftover is the per-cell spread of
ξ_fit (±3 %) times the deficit (≈ 0.4), i.e. ≈ 0.012, which is exactly what is measured.

**(b4) The improvement's discretisation is not free.** The same operator written as
−ξ Σ_B β_i (∇²φ²)_i with the five-point stencil (the Laplacian on the **field**, keeping the
entangling-circle contact term that the discrete integration by parts produces) gives
r = **0.726** instead of **0.0749** — ten times worse than doing nothing. The weight-side form is
the one that reproduces the first law; the contact term ∮φ² is a lattice-ambiguous
entangling-surface term and is **dropped, not derived**. This is an ASSUMPTION of section J.

**(b5) Lifshitz z = 2 stays obstructed in 2+1.** With the canonical ansatz r = **0.955–0.986**
(R = 3…10); with the CFT ansatz (ξ = 1/8, which has no business there) it *grows* with the disk,
0.117 (R = 4) → **1.264** (R = 10), so the contrast with the wave lattice at the two largest disks
is **35×** and **244×**. The contrast is real but weaker and less uniform than 1+1's 44–911×.

**(b6) The free radial weight recovers the CHM parabola in 2+1.** Over even Chebyshev profiles in
r/R (two modes, i.e. constant + parabola, which is *below* the variation family's numerical rank 3),
minimax with w ≥ 0 on the disk gives overlap with the CHM profile **0.992, 0.997, 0.999** at
R = 4, 6, 8 — but the recovered normalisation is **1.67, 1.60, 1.66** with the canonical energy
density and **0.78, 0.80, 0.99** with the improvement, at r_free = **4.2e-4, 5.0e-4, 1.9e-4**
(against r_CHM = 1.3e-2, 1.2e-2, 2.4e-2). So the *shape* of the modular weight is recovered in 2+1
exactly as it was in 1+1, and the *normalisation* is what the improvement term fixes.

**Limits of section J.** The disk is a lattice disk (sites with r < R), so its boundary is jagged
and the residual wiggles with R at the few-per-cent level; the wavelength is capped by the box
(λ ≤ N + 1), so R = 10 reaches only λ/2R ≈ 6.5; the improvement's surface term is dropped (b4);
ξ is fitted on 15 cells, not derived; and the Lifshitz comparison uses the CFT ansatz on a
non-conformal theory, which is a diagnostic, not a fair test. Everything here is reflection-odd or
-even single-mode squeezes; the 2+1 free-weight test uses one weight family and one parity. The
quadrature split (b1) is archived at one β discretisation (site centres) and ξ = 0; its
reflection-**even** cell is archived too but is not quoted, because with the canonical ansatz the
even δ⟨K⟩ has the *opposite sign* to δS (δ⟨K⟩/δS = −0.723…−0.014, r = 1.33–1.72), which makes the
even sector ratios (pp 0.806–0.985, xx 0.00004–0.071) uninterpretable as "how well does this
sector pair".

**Admissibility (Layer 7 unit 2, items (a) and (b)).** This pass ran
`tests/test_toy_jacobson.py` + `tests/test_toy_jacobson_2d.py` +
`tests/test_toy_jacobson_continuum.py` + `tests/test_geometry_jacobson.py` +
`tests/test_exports.py`: **106 passed in 54.9 s** on its own tree (36 + 16 + 20 + 3 + 31) after fix
round 2 (**105** after round 1, **103** before it). The seven item-(a) tests and the sixteen
item-(b) tests were **mutation-checked**: perturbing `defect_over_zero_mode` to 1.30, `nu_top` at
N = 3200 to 0.95, the n = 5 absolute defect to 1e-3, `max_cancellation` to 1.9, the fitted ξ to
0.20, the R = 4 even/odd ratio to 500, the odd pp-ratio range floor to 0.91 and the R = 10 pp
diagonal slope to 0.99 each fails the test that pins it, and — fix round 2 —
`defect_over_zero_mode_by_ratio[8][4]` to +1.10, `defect_over_zero_mode_dev_from_one_by_ratio[16][8]`
to 0.05, `top_mode_share_even_min_all_ratios` to 0.9, `even_over_odd_matched_min` to 0.05 and
`L32_n5.even_over_odd_per_dV_by_ratio[16]` to 0.5 each fails one of the two rewritten item-(a)
tests. The archives were restored from a scratch copy afterwards (`git status` clean). Regenerating the whole J archive with `--force` after adding
section J6 reproduced every previously pinned number **bit for bit** (only `__build__` and
`wall_clock_s` changed). No test was deleted and no tolerance was loosened. **Two assertions were removed in fix round 2**
(`got[2] < 3 * odd_abs` and `got[3] < 0.1 * odd_abs` in
`test_projecting_the_soft_even_modes_kills_the_absolute_defect_not_the_ratio`) and one re-pinned
(`top_mode_share_even_min > 0.5`); all three pinned the retracted comparisons above, and each is
replaced by a **stronger** pin — the scale-invariant `even_over_odd_per_dV` cell by cell, the
‖δV_B‖_F collapse, and the two explicitly named top-mode-share minima. The reason is recorded in
the retraction and re-pin paragraphs of the item (a) section. `notebook_even.py`
and `notebook_2p1.py` both exit 0 in `--quick` mode writing outside `data/`. The **full suite was
not re-run by this pass**, so per `papers/README.md` no number in sections I or J is admissible
into a manuscript before the next quiescent-build run.

## What was circular; what the free-weight test establishes

**Circular (anchors 1–8 as previously headlined).** The local form K_loc = 2π Σ βᵢ hᵢ took β from
CHM/Bisognano–Wichmann, i.e. from the conformal symmetry of the very dynamics the test was said
to "select". With β fixed, r(K) measures how far the ball's exact modular Hamiltonian is from the
CHM form *for the given K*: it must vanish for the massless wave chain (CHM is a theorem there,
up to lattice corrections) and it must fail for anything whose modular Hamiltonian is not the
CHM one — which is every non-conformal K. That r tracks |ω(π/L)/(π/L) − 1|, that a wrong velocity
gives r = |1 − v|, that v* → 1 and that the fit lands at v² = 0.980, are all consequences of the
normalisation 2π(R² − x²)/(2R) *assuming* v = 1: the vacuum of v²K is the same state as the
vacuum of K, so no state-based residual can measure v, and "v² recovered" was the unit built into
the weight. The mass and dispersion-shape parts of anchors 3–4 stand as measurements of the
lattice's distance from CHM, not as selection of Lorentz invariance. Anchors 5–8 (discretisation
floor, the k⁴ degeneracy test, the boundary-CFT weight at the wall, the projected zero mode) are
unaffected: they are statements *about* the CHM form on the lattice and never claimed more.

**Established by anchor 9 (the weight free).** (i) *Existence and identity*: among all
nonnegative local weights in a three-mode smooth family, sampled at sites or at bond midpoints,
the one that makes δS = δ⟨K_loc⟩ hold for every long-wavelength single- and two-mode squeeze of
the wave chain is the CHM parabola with its CHM normalisation (to 0.1 %), and it does so
20–130× better than the CHM discretisation itself. (ii) *Parsimony, not existence, for z = 2*: a
three-mode nonnegative weight leaves 27–68 % of δS unaccounted for, but four modes fit a single
ball (7.0e-04 at L = 8, degrading to 8.2e-02 at L = 32; an independent four-monomial basis gives
6.9e-04, 5.6e-03 at L = 8, 16), and one profile R^z f(x/R) shared across the three balls fits
too once ≥ 11–12 modes are allowed (the ladder: 0.091 at 8 modes, 0.023 at 11–12 with z = 2,
5.5e-03 at 11 with z = 3, 6.9e-04 at 12 hat modes: below 1e-02 by 11–13 modes and below 1e-03 by 12–14, z-dependent; 5e-04 only by 20 Chebyshev modes; the monomial basis reproduces the
Chebyshev rows). So a local, nonnegative, scale-covariant weight *exists*
for Lifshitz; what distinguishes it from the wave chain is that the wave chain's is the 3-mode
parabola and Lifshitz's needs ≥ 11–12 modes of a bumpy profile, at which point the family is as
large as the variation family's conditions at that level (per-ball rank 2–3 at 10⁻³, 8–9 in
total). The sign-indefinite three-mode minimum (1e-3, with a profile 10–84× CHM's scale) is
recorded, not used. (iii) *What the free weight
absorbs*: the velocity exactly (a unit) and a mass per ball; only scale covariance across
ball sizes — which is what CHM's R(1 − x²/R²)/2 *is* — sees the mass, and then at mL ~ 0.1.
(iv) *A limitation of the test itself*: Jacobson's regime ball ≪ wavelength is low-rank (5–6
independent conditions at 10⁻⁶ per ball), so any weight with that many parameters fits any
dynamics; the free-weight residual has content only at fixed parameter count below the rank,
and the rank is reported next to every number. The claim is therefore not "Lorentz invariance
emerges from entanglement" but: *on this lattice, δS = δ⟨K_loc⟩ with a local, nonnegative,
smooth K_loc singles out the CHM weight for the wave chain at three modes and admits a weight
for z = 2 dynamics only at ≥ 11–12 modes — a parsimony gap, not an existence gap; it does not
see the velocity, and it sees the mass only through scale covariance.*

## The mode ladder (anchor 9): the parsimony statement in numbers

Shared profile R^z f(x/R) across L = 8, 16, 32, bond-centred, w ≥ 0, N = 1600
(`data/G_free_weight.npz`, `ladder_*`; the auditor's independent hat and monomial bases agree).
Conditions the variation family imposes, summed over the three balls: 6 (Lifshitz) / 9 (wave) at 10⁻²,
8 / 9 at 10⁻³, 15 / 18 at 10⁻⁶.

| modes n | 3 | 4 | 6 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wave, Chebyshev, z = 1 | 1.5e-4 | 1.2e-4 | 1.2e-4 | 2.6e-5 | 2.6e-5 | 2.6e-5 | 2.6e-5 | 2.5e-5 | 2.0e-5 | 2.0e-5 | 1.9e-5 | 1.9e-5 |
| wave, Chebyshev, z = 2 | 0.55 | 0.52 | 0.34 | 0.26 | 0.24 | 0.22 | 0.22 | 0.21 | 0.21 | 0.21 | 0.21 | 0.13 |
| wave, Chebyshev, z = 3 | 0.86 | 0.85 | 0.77 | 0.74 | 0.73 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.64 |
| wave, hat, z = 1 | 8.3e-3 | 1.5e-4 | 3.4e-5 | 2.7e-5 | 1.7e-3 | 2.1e-5 | 2.0e-5 | 2.0e-5 | 8.3e-5 | 1.9e-5 | 1.9e-5 | 1.9e-5 |
| Lifshitz, Chebyshev, z = 1 | 0.92 | 0.59 | 0.49 | 0.26 | 0.11 | 0.091 | 0.079 | 0.079 | 5.3e-4 | 5.3e-4 | 5.3e-4 | 5.3e-4 |
| Lifshitz, Chebyshev, z = 2 | 0.74 | 0.37 | 0.14 | 0.091 | 0.033 | 0.025 | 0.023 | 0.023 | 5.8e-4 | 5.4e-4 | 5.3e-4 | 5.3e-4 |
| Lifshitz, Chebyshev, z = 3 | 0.88 | 0.56 | 0.15 | 0.062 | 0.062 | 0.040 | 5.5e-3 | 4.7e-3 | 2.5e-3 | 7.7e-4 | 6.3e-4 | 5.3e-4 |
| Lifshitz, hat, z = 1 | 1.0 | 0.67 | 0.68 | 0.12 | 0.99 | 0.12 | 0.60 | 2.0e-3 | 0.61 | 5.3e-4 | 2.6e-3 | 5.3e-4 |
| Lifshitz, hat, z = 2 | 1.0 | 0.12 | 0.13 | 0.11 | 0.99 | 0.11 | 0.027 | 6.9e-4 | 0.027 | 5.3e-4 | 2.6e-3 | 5.3e-4 |
| Lifshitz, hat, z = 3 | 1.0 | 0.55 | 0.55 | 0.13 | 1.0 | 0.12 | 0.034 | 6.2e-4 | 0.19 | 5.9e-4 | 2.9e-3 | 5.3e-4 |

(Even monomials ξ^{2m} reproduce the Chebyshev rows to the digits shown up to n = 12, as they
must — same span. The hat rows alternate because the node set changes with n.) First below
1e-2 / 1e-3: Lifshitz Chebyshev z = 1: 13 / 13, z = 2: 13 / 13, z = 3: 11 / 14; hat z = 1: 12 / 14,
z = 2: 12 / 12, z = 3: 12 / 12. Wave chain: 3 / 3 (Chebyshev), 4 / 4 (hat), and never at z ≠ 1.
The 5.3e-4 floor reached by every z from 13 modes is the regime where the family has as many
parameters as the variation family has conditions at that level; it is not a Lifshitz property.

## What is new and what is a consistency check

Consistency checks: the exact first law (Blanco–Casini–Hung–Myers 2013; Layer-1 anchor); the
CHM/BW profile of the exact modular Hamiltonian in the continuum-limit sense of Di Giulio–Tonni
2020 (row sums and r²-resummed weights within 1–13 % of 2πβ at L = 16–32); the zero-mode term of
the 2d massless scalar (Casini–Huerta 2009) appearing as an O(1) even-sector residual.

New, to our knowledge: the CHM/BW weight of an interval attached to a Dirichlet wall,
β(y) = (B² − y²)/(2B), as the *only* local form that passes the test there (the bulk parabola
fails by 39–3015×) — the residual detects the boundary condition; the degeneracy argument that
makes the k⁴ coefficient measurable (∂ρ/∂c₃ = 6 ∂ρ/∂c₂) and the wavelength-resolved sensitivity
C; the demonstration that the ansatz's discretisation floor is a choice (1.39/L² → 0.053/L²);
the residual r(K) as a quantitative, ball-size-resolved measure of how far
a lattice dynamics is from admitting the local CHM modular Hamiltonian; its L^-2.27 approach to 0
for the wave chain and its O(1), growing values for z ≠ 1; the map r ≈ 0.79/L² + 0.89 δ_v1 across
44 dynamics with the 1.2/L² ansatz floor; the velocity renormalisation 1 − v* ≈ 0.85/L²; the
constrained-dynamics landscape (velocity and mass pinned, coupling range a flat valley, the gapped
dip at m ≈ 3–4 and the need for Jacobson's UV-entanglement premise to exclude it); the observation
that local coupling changes are cutoff-dominated variations unsuitable for the test.

---

## Limitations and open problems

- **No longer 1+1 only** (Layer 7 unit 2, item (b)): the toy now runs in 2+1 on an N × N
  Dirichlet square lattice with disks of radius R = 3…10 at N = 128 (`vacuum/geometry/jacobson2d.py`,
  section J above). What that extension cost is itself a limitation: with the *canonical* lattice
  energy density the 2+1 residual is 0.36–0.45 and does not decay with R, because in d ≥ 3 the CFT
  stress tensor carries the conformal improvement −ξ∇²(φ²) that vanishes identically in d = 2.
  Restoring it with the measured ξ = 0.1278 ± 0.0040 (against 1/8) brings the residual to
  0.005–0.075, but the improvement's entangling-surface contact term ∮φ² is **dropped, not
  derived** (keeping it gives 0.73, ten times worse), and ξ is **fitted on 15 cells, not derived**.
  Still no gravity, area term or Einstein equation anywhere, in either dimension; the operator
  statement K_B = K_loc is not tested, only its pairing with smooth variations. Two ball positions
  are measured in 1+1 (centred and attached to the wall); a ball *detached* from the wall is
  obstructed at O(1) and the toy has nothing to say there without a bilocal term. In 2+1 only the
  centred disk is measured, its lattice boundary is jagged (so r wiggles with R at the percent
  level), and the box caps the variation wavelength at λ ≤ N + 1, i.e. λ/2R ≈ 6.5 at R = 10.
- The floor of the *site-centred* ansatz is 1.39/L² and the bond-centred one reaches
  0.053/L², but the two cannot be combined: the bond-centred form fails the pure-dispersion
  test (l₃/l₂ = 1.9–2.2 instead of 6), so its low floor is a cancellation for the
  nearest-neighbour chain, not a better lattice CHM form. The k⁴ coefficient quoted here is
  therefore measured with the site-centred ansatz, whose floor is unchanged; what resolved it
  was the third-neighbour coupling, not a smaller floor.
- The k⁴ sensitivity C is *not a single number*: it changes sign and magnitude with the
  variation's wavelength (−0.974, +0.151, +0.461 at λ = 4L, 8L, 16L) while being independent
  of L. The map's single coefficient 0.89 is the max-over-wavelengths value, dominated by
  λ = 4L, and it sharpens to |C| = 0.974 ± 0.034 *for the k⁴ family only*; it does not become
  one law across wavelengths, and the fit of Sec. 4 (r = 0.785/L² + 0.891 δ, scatter 0.24) is
  left as the coarse summary it is.
- The bond-centred residual's own L = 32 value (0.174/L²) is limited by the chain length
  N = 2400, not by the ansatz: at N = 6400 the same quantity is 0.052/L² at L = 8, 12, 16 and
  still falling at L = 24.
- **The even sector is the zero-mode sector** (Layer 7 unit 2, item (a); section above; the
  earlier bullet's *mechanism* was wrong and is corrected here). The reflection-even first-law
  defect equals the ball's top even Williamson-mode entropy contribution to 14 % **at the longest
  wavelength in the family, λ = 16L** (1.144, 1.139, 1.158 at L = 8, 16, 32); it is **not** that
  number over the family (−0.752, −1.288, −3.738 at λ = 4L), and what is uniform is the convergence
  in λ, |ratio − 1| falling monotonically 1.75 → 0.99 → 0.14 (L = 8), 2.29 → 0.60 → 0.14 (16),
  4.74 → 0.49 → 0.16 (32). It is an IR effect: at fixed L = 8 the mode's ν₁ grows 0.934 → 1.199
  and r_even grows 2.04 → 3.60 as the chain goes N = 200 → 3200, while r_odd falls. Projecting the
  softest even Williamson modes out of the variation drives the **absolute** defect 1.34e-2 →
  5.3e-10 (L = 8) while the **relative** residual saturates at 0.016–0.17 — because those modes
  carry essentially all of the even sector's δS, so the projection removes numerator and
  denominator together. **The absolute defect is not comparable across sectors** (retraction, fix
  round 2, in the item (a) section): the projection shrinks ‖δV_B‖_F by 2·10⁵–1.4·10⁶ too, and per
  unit variation the projected even sector is **1.9×–659× worse** than the odd sector, never
  better. **Correction:** the earlier statement
  that the leftover is "a near-cancellation of δS" is measurably false (Σ|c_k|/|δS| = 1.000014); the
  saturation is a projected-denominator effect. It is not a numerical artefact either: the projected
  δS agrees with an independent finite-difference route to 1.5e-6 and with the mpmath rung to
  2.1e-11, and assembling δS − δ⟨K_loc⟩ term by term (no big subtraction) reproduces the subtraction
  to 2.0e-11. **What remains a limitation:** in 1+1 the even sector is **not** shown to be a place
  where the local CHM form works — after the zero mode is projected out, the residual is still
  0.016–0.17 relative and 1.9×–659× the odd sector's per unit variation, and that leftover is not
  explained; no reweighting is proposed that would remove it. The (L, m) scan cannot isolate the
  zero mode either, because a mass destroys conformality in both sectors (r_odd 4.6e-4 → 0.33 while
  r_even stays O(2)). The claim that the *dominant* excess is the zero mode rests on (a1) in the
  long-wavelength limit only, plus (a2), (a3) and the 2+1 control (even/odd 1.8–24 there against
  3476/8481/6297 in 1+1) — not on a derivation.
- The minimiser of Sec. 5 still lands on c₂ = −0.16 or +0.23 depending on the start: the
  constrained fit was run with the site-centred ansatz at L ≤ 16, where the k⁴ signal is below
  the floor. The k⁴ measurement above does not change the landing; it explains it.
- The gapped branch is excluded by a bound (m ≤ 1), i.e. by a premise, not by the residual alone.
- Random couplings are mirror-symmetric (to keep parity exact); their residual is seed-dependent
  and shows no systematic L-dependence at L ≤ 32.
- The Lifshitz audit reaches 8.6e-9 only with the mpmath rung and at N = 800; at N = 1600 its ball
  covariance (⟨x²⟩ ~ N²) already loses the 1e-11 margins in float64 storage.
- Anchor 9's class of "local K" is a weighted sum of the lattice's local energy pieces (the PSD
  bond split's site and bond terms), with a smooth nonnegative weight of three modes as the
  headline and up to eight in the shared-profile test. It is not the most general banded
  quadratic form near the ball (that question is the `locality_score` of `eh_profiles`, 0.99
  wave vs 0.53 Lifshitz, and a general banded G with enough entries would fit the low-rank
  variation family trivially). The site-symmetrised split hᵢ = pᵢ²/2 + xᵢ(Kx)ᵢ/2 has identically
  zero response to every single-mode squeeze (a lattice virial identity) and is unusable here;
  a curvature-type split (Lx)ᵢ²/8 for the Lifshitz chain was not tried.
- The Lifshitz verdict is about parsimony, not existence: per ball, four nonnegative modes fit
  (7.0e-04 → 8.2e-02 from L = 8 to 32); shared across balls, ≥ 11–14 modes fit (below 1e-02 by 11–13 modes and below 1e-03 by 12–14, z-dependent; 5e-04 only by 20 Chebyshev modes); the three-mode numbers 0.27–0.68 and the eight-mode floor 0.091 are properties of those
  families. With enough modes every tested dynamics fits — the toy measures how many. Nonnegativity (needed for e^{−K_loc} to be a state when the pieces
  are PSD; the Lifshitz bond split is not PSD piecewise) is kept; the signed minima (1e-3 per
  ball, 6.6e-4 shared, with weights 10–84× the CHM scale) are recorded, not claimed either way.
- Jacobson's regime is low-rank: the variation family has 5–6 independent conditions per ball at
  10⁻⁶ (7–8 at 10⁻⁸), so every "free weight" statement is at a stated parameter count. Wider
  families (λ ≥ 2L) raise the rank to 6–8 and leave the wave/Lifshitz contrast at ≥ 400×; they
  were explored, not archived.
- The free weight sees neither the velocity (a unit) nor, per ball, a mass; the shared-profile
  residual sees the mass only in the crossover mL ~ 0.1–10 and is small again (1e-3 at m = 1)
  deep in the gapped regime, where a scale-covariant boundary profile exists — the same gapped
  shoulder as anchor 4, and the same premise (gapless UV entanglement) is needed to exclude it.
- The coupling range under the shared profile is resolved towards the nearest-neighbour
  discretisation (c₂ = 0 at v = 1: 6.3e-05, against 2.5e-04 for the improved Laplacian and
  4.9e-03 for c₂ = +0.1) — a lattice statement about which energy density the bond-centred
  weight is exact for, not a measurement of the continuum k⁴ term (anchor 6 remains the
  measurement of that).

## Reproducing

```
.venv/bin/python papers/toy-jacobson/notebook.py            # ~6 min idle (24 min under load), writes data/A-G
.venv/bin/python papers/toy-jacobson/notebook.py --quick    # ~40 s
.venv/bin/python papers/toy-jacobson/notebook_continuum.py  # ~18 s, writes data/H_continuum_check.json
.venv/bin/python papers/toy-jacobson/notebook_even.py       # ~47 s, writes data/I_even_sector.{json,npz}  (item (a))
.venv/bin/python papers/toy-jacobson/notebook_2p1.py        # ~65 s, writes data/J_2p1_*.json               (item (b), sections J1-J6)
.venv/bin/python papers/toy-jacobson/notebook_even.py --quick --out DIR   # ~5 s smoke
.venv/bin/python papers/toy-jacobson/notebook_2p1.py --quick --out DIR    # ~1 s smoke
.venv/bin/python -m pytest tests/test_toy_jacobson.py -q       # 36 anchors (29 + the 7 of item (a)), ~24 s
.venv/bin/python -m pytest tests/test_toy_jacobson_2d.py -q    # 16 anchors of item (b), ~0.7 s
.venv/bin/python -m pytest tests/test_toy_jacobson_continuum.py -q   # 20, ~7 s
.venv/bin/python -m pytest -q                               # admissibility gate
```

`notebook_2p1.py` writes one file per section and **skips any section whose file already exists**
(`--force` regenerates), so a crashed or interrupted run resumes instead of restarting.
Every section of both notebooks runs with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=VECLIB_MAXIMUM_THREADS=2`.
