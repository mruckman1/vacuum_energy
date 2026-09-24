# MANIFEST — `multi-interval-modular-hamiltonians`

Result candidate of Layer 4 (PLAN.md, "Geometry from entanglement, computationally",
instrument 1: the modular-Hamiltonian laboratory on free-fermion chains).

---

## The claim

On the half-filled hopping chain, the exact entanglement Hamiltonian
h = ln((1 − C_A)/C_A) of unions of intervals — computed from closed-form correlation
matrices in arbitrary precision, because float64 is honest only to ~18 sites — reproduces
the Casini–Huerta continuum modular Hamiltonian (CQG **26**, 185005 (2009), Eqs. (33),
(44)–(48)) after the lattice dictionary D1–D6 stated once in
`vacuum/modular/casini_huerta.py`: the local Bisognano–Wichmann weight β(x) = 2π/z′(x)
**and** the nonlocal weight 2π/((x − x_c) z′(x_c)) between conjugate points, with
deviations falling as 1/L² (row sums 9.1e-2 → 9.9e-3 → 2.2e-3 and smeared bilinears
2.3e-2 → 7.1e-3 → 1.7e-3 at scales 1, 2, 4), and its modular flow carries a right-moving
packet along the CH trajectories and "teleports" the weight sin²θ(τ) of CH Eq. (59)
into the other interval (max relative deviation 7.7 % for 0.05 ≤ |τ| ≤ 0.3). Those are
**consistency checks** — of CH 2009, of Eisler–Peschel 2017 (J. Phys. A **50**, 284003,
Eqs. (26), (27), (54)) and of Eisler–Tonni–Peschel 2022 (J. Stat. Mech. 083101), whose
Eqs. (31)–(32) coincide with the dictionary derived here. Beyond the published closed
forms, this candidate measures: (i) for **three and four intervals** the n-interval CH
formula holds on the lattice once the gaps reach ≈ L (all pairs within 4.1 % for gaps
≥ 12 sites at L = 10–12; 28 % at gap 6; no relation at gap 3), an intermediate interval
**screens** the coupling of the outer pair by a factor that the summed lattice weights
track to within 6 % of the CH prediction even at gap 3 (0.56/0.75/0.88/0.95/0.98 vs
0.59/0.76/0.89/0.95/0.98 for gaps 3…48), and the adjacent-pair coupling decays with
separation with exponent −0.63 in the summed weight (CH: −0.62; the Frobenius norm
decays faster, −0.92, as the coupling spreads over more sites); (ii) in **gapped
(staggered-mass) chains** the BW profile deforms from the parabola into a triangle whose
edge slope is 2π·s(m) with s = 0.997, 0.990, 0.945, 0.835 at ξ = 10, 5, 2.1, 1.1 and
whose linear stretch grows from 2 to 16 bonds, while the inter-interval couplings decay
exponentially with a length ξ_h = 1.00, 1.34, 1.43 ξ at ξ = 10, 5, 2.1 — **superseded by (iv)
below: those are window-fit crossover values at gaps of a few ξ, and the asymptotic length is
ξ_h/ξ = 0.9985 ± 0.0257 = 1**. **Three items that the first pass left open are now closed.**
(iii) *n ≥ 3 at larger L.* At fixed gap/L the deviation of the summed lattice bilocal weight
from the n-interval CH weight falls as **A/L²** — 1.32e-2, 7.54e-3, 3.38e-3, 1.91e-3 for
three equal intervals with gap = L at L = 12, 16, 24, 32 (A = 1.906, 1.930, 1.947, 1.953) —
and the L → ∞ extrapolation of the CH agreement is **exact to +8e-05** (three intervals,
gap = L; +4.4e-04 at gap = L/2 with A = 3.76, +5.4e-04 for four intervals with A = 2.27; the
noisier row-by-row deviation extrapolates to −4.4e-03, −5.5e-02, −1.3e-02). (iv) *ξ_h/ξ.*
Extending the gapped scan to ξ = 20 and 40 and to gaps of 16 ξ, the **local** decay rate of
the alternating row weight (a single exponential fitted over a window still carries the
algebraic prefactor of the massive propagator — that is what produced the 1.3–1.4) gives
**ξ_h/ξ = 0.9985 ± 0.0257** over the ten local rates at gaps ≥ 8 ξ spanning ξ = 5, 10, 20, 40:
it tends to a constant and that constant is **1** — the modular nonlocality decays with
exactly the correlation length. (v) *s(m) in closed form — a published result, reproduced.* The half-infinite
entanglement Hamiltonian is known exactly for precisely this chain (hopping 1/2, staggered
potential μ(−1)ⁿ): Eisler, J. Stat. Mech. (2025) 013101, arXiv:2410.16433, Sec. V,
Eqs. (64)–(72) — H = 4κK(κ′)T with κ = 1/√(1 + μ²) and T the lattice boost
(t_m = −m/2, d_m = (−1)^m μ (m − ½)), derived there by the commuting-operator method; in this
candidate's units that is
**s(m) = (2/π) k I(k′), ε = π I(k′)/I(k), k = 1/√(1 + m²) = sech(1/ξ), k′ = tanh(1/ξ)**,
with single-particle levels (2l + 1) ε; the lattice reproduces the edge slope (taken from the
staggered on-site term) to **≤ 5.2e-05** over m = 0.02…2 and to **≤ 1e-08** for m ≥ 0.2, and
ε₀ to 1e-15 for m ≥ 0.2. Against the fitted 0.997/0.990/0.945/0.835 at ξ = 10/5/2.1/1.1 the
residual is **−4.2e-04, −2.0e-08 and ≤ 1e-12** (nearest-neighbour bond, L = 40–60); in the
scaling limit s = 1 − 1/(4ξ²) + O(ξ⁻⁴). **Status: a claim about the lattice model with no experimental inputs (every number is computed by `notebook.py` from the package), admissibility rests on the integration record under "The producing build" — the full regression suite was not green on the concurrently-edited tree it was produced on.** **Priority (docs/PRIORITY.md): REPRODUCTION-AT-CERTIFIED-PRECISION** — of Casini–Huerta 2009 (whose n-interval formula, screening included, is what (i) and (iii) check), of Eisler–Tonni–Peschel 2022 (the dictionary) and of Eisler 2025 (item (v)); what is added is lattice numerics — n = 3, 4 with the 1/L² law, and the asymptotic ξ_h = ξ of (iv) — not new formulas.

---

## The producing build

| item | value |
|---|---|
| generator | `papers/multi-interval-modular-hamiltonians/notebook.py`, entry point `main()` |
| generator sha256 | `b5b9467c3903928cd7aeb52108fab97638aa756d3f683548aa8d58fa0d5ce984` |
| produced (UTC) | 2026-09-04T15:03:45Z (whole notebook re-run with the three new sections S4–S6; the S1–S3 numbers are unchanged) |
| git HEAD | `d248e3c5ab69093a2f7a3aada8574222e64ca01e` (working tree dirty: this candidate's own changes plus concurrent work by other agents; nothing is committed by this candidate) |
| package | `vacuum` 0.1.0 (editable) |
| environment | Python 3.13.3, numpy 2.5.2, scipy 1.18.1, mpmath 1.4.1, macOS-14.6.1-arm64 |
| command | `.venv/bin/python papers/multi-interval-modular-hamiltonians/notebook.py` |
| wall clock | 607.6 s (budget 20 min) |
| pytest, this candidate | **32 passed in 63.7 s** (`tests/test_modular_lab.py`) |
| pytest, full suite | see the 2026-09-04 integration note below; the 2026-09-02 record is kept for the historical build: **799 passed, 10 failed, 6 errors in 916 s** (`.venv/bin/python -m pytest -q`, 2026-09-02, on a working tree being edited concurrently by the Layer-2/3/5/6 agents: the 6 errors are `tests/test_noise_threshold_smoke.py`, whose `data/` archives a concurrent agent deleted; the 10 failures are in files this candidate does not own and touch no `vacuum.modular` code). Every test that imports `vacuum.modular` passes on the same tree: **62 passed in 54 s** (`tests/test_first_law.py`, `tests/test_modular_lab.py`, `tests/test_gjax.py`). **Admissibility per papers/README.md is therefore NOT yet established**: the full known-results suite must be re-run green on a quiescent build before any number here moves from `data/` into a manuscript. |

**Integration note, 2026-09-04.** The full suite was re-run on this tree (`.venv/bin/python -m pytest -q`): **9 failed, 1136 passed in 2925.6 s (48:45)**, on a tree being edited concurrently by other agents. Eight of the nine belong to packages this candidate does not own — `tests/test_exports.py` for `vacuum.composite`, `detectors`, `floquet`, `hardware`, `inequalities`, `opt` (public names not re-exported) and its blocked-extras import check, plus `tests/test_circuit_mapping.py`. The ninth, `test_submodule_all_is_subset_of_package_all[geometry]`, **was this candidate's own**: `k4_sensitivity` was in `vacuum.geometry.jacobson.__all__` before it was re-exported from `vacuum/geometry/__init__.py`, and the suite reached that test in the window between the two edits. It is fixed — a re-run of `tests/test_exports.py` after the fix leaves 7 failures, none of them `[geometry]` or `[modular]`. Evidence for this candidate on the same tree: **58 passed in 147.7 s** (`tests/test_modular_lab.py` 32, `tests/test_toy_jacobson.py` 23, `tests/test_geometry_jacobson.py` 3) plus `tests/test_first_law.py`, and both notebooks' `--quick` paths exit 0. **Admissibility per papers/README.md required the full known-results suite to be re-run green on a quiescent build** before any number here moves from `data/` into a manuscript; that run is the Integration line below.

The sha256 is embedded in every archive under `data/` (`__build__` key) and in
`data/build_info.json`. `notebook.py` raises on any `ModularPrecisionWarning`, so every
modular Hamiltonian in `data/` was computed with the entanglement spectrum resolved with
≥ 15 digits to spare (the precision audit of docs/API.md, on the fermionic side).

---

**Integration: full suite 1201 passed on the quiescent tree at HEAD `c18a206`, 2026-09-04** (`.venv/bin/python -m pytest -q`, twice in a row on the final tree: 1201 passed, 10 warnings in 731.91s (0:12:11) and 1201 passed, 10 warnings in 740.37s (0:12:20); with tenpy, cvxpy, jax, jaxlib, qiskit, qiskit_aer and qiskit_ibm_runtime blocked at the import system, 1026 passed, 76 skipped in 428.84s (0:07:08); all 15 notebook `--quick` paths exit 0, the slowest in 307 s). The tree is commit `c18a206` plus this pass's uncommitted edits — README, STATUS, PRIORITY, MANIFEST and draft prose reconciling the rollups to the MANIFESTs, plus one print in `interacting-vacua-first-numbers/notebook_qei_exact.py` made None-safe so its `--quick` path exits 0 (no numerical code, no data). `data/` was produced on the build recorded above; this run is the admissibility record under `papers/README.md` for the candidate's code as it stands on that tree, and a full re-run of `notebook.py` there regenerates `data/` with a new `generator_sha256`; the producing-build fields above are left as recorded.

## Modules used

| module | what it does here |
|---|---|
| `vacuum.modular.ctm` (`ctm_slope`, `ctm_level_spacing`, `ctm_modulus`, `ctm_slope_expansion`, `lattice_boost`, `triangle_h`) | the corner-transfer-matrix closed forms for the gapped chain: s(m) = (2/π) k I(k′), ε = π I(k′)/I(k), the half-infinite boost operator and the triangle form of a finite segment, with the references and the status of the modulus map stated in the module docstring |
| `vacuum.modular.lattice` (`infinite_chain_C`, `massive_chain_C`, `ring_C`) | closed-form correlation matrices at arbitrary precision (EP 2017 Eq. (3); staggered mass derived in the module docstring and checked against brute force to 1e-15) |
| `vacuum.modular.fermionic` (`interval_modular`, `bw_profile`, `bilocal_weight`, `chiral_bilinear`, `nonlocal_decay`, `locality_score`, `modular_flow*`, `wavepacket`, `eisler_peschel_eq26/27/54`) | the laboratory: mp diagonalization, BW profiles ('nn' and resummed, ETP-2022 Eq. (31)), alternating row sums (ETP-2022 Eq. (32)), smeared bilinears, modular flow |
| `vacuum.modular.casini_huerta` (`ch_beta`, `ch_conjugate_points`, `ch_nonlocal_weight`, `ch_bilinear`, `ch_trajectory`, `ch_mixing_angle`, `ch_mutual_information`, `casini_huerta_exact`) | the transcribed CH 2009 continuum reference and the dictionary D1–D6 |
| `vacuum.modular.scan` (`multi_interval_scan`, `modular_hamiltonian_auto`) | the n-interval / massive sweeps with automatic precision escalation |
| `vacuum.fermions` (`block_entropy`, `entanglement_hamiltonian`) | the validated Layer-1 formulas (float64 path; entropies) |
| **No `experiments/` files** | only published analytic results enter, each cited at its point of use: CH 2009 Eqs. (33), (44)–(48), (53), (59), (63); EP 2017 Eqs. (3), (22), (26), (27), (54); ETP 2022 Eqs. (8)–(9), (31)–(33); Bisognano–Wichmann 1976 |

---

## Data

| file | contents |
|---|---|
| `data/build_info.json` | the build |
| `data/s1_single_interval_bw.npz` | anchor 1: L = 20, 40, 60, 80 — nn vs EP Eq. (27), resummed vs parabola, ε_max vs Eq. (54), edge bond, h₃ peak vs Eq. (26); the L = 40 profiles |
| `data/s1_two_interval_ch.npz` | anchor 2: scales 1, 2, 4 — resummed local vs 2π/z′, nn centre correction, row sums vs CH (both directions), sin² bilinear, 4×4 Gaussian grid, mutual information vs the CH cross ratio; scale-2 profiles and weights |
| `data/s1_modular_flow.npz` | teleportation P₂(τ) vs sin²θ(τ), centroids vs CH trajectories; single-interval centroid vs Eq. (53) |
| `data/s2_three_four_intervals.npz` | every pair of every 3/4-interval configuration (gap, Frobenius, max, summed lattice and CH weights, interior deviations, locality), the screening table, the decay table and exponents, one four-interval h |
| `data/s3_offcritical_single.npz` | m = 0.02…1: β_nn and resummed profiles, edge slope, linear extent, plateau, diagonal/2nd/3rd-neighbour maxima |
| `data/s3_offcritical_two.npz` | m × gap table of inter-interval couplings, the ξ_h fits, the critical exponent, the gapped three-interval pairs |
| `data/s4_multi_large_L.npz` | n = 3 and 4 at L = 10…32 with gap/L fixed: per-configuration row-by-row and summed-weight deviations from the n-interval CH formula, and the 1/L² fits with their L → ∞ extrapolations |
| `data/s5_xi_h_asymptotics.npz` | gapped two-interval couplings for ξ = 5, 10, 20, 40 at gaps out to 16 ξ (six (m, L) families, 36 configurations); the table of local decay rates ξ_h^loc(d)/ξ for the alternating row weight and for the Frobenius norm |
| `data/s6_ctm_slope.npz` | m = 0.02…2: the CTM closed forms s(m), ε against the lattice edge slope (bond and on-site), the level-ladder ratios, the near-edge block against `lattice_boost`, the scaling-limit expansion, and the critical 1 − 1/L for comparison |
| `data/summary.json` | the headline numbers quoted above |

---

## Anchors — measured vs target (all also permanent tests in `tests/test_modular_lab.py`)

| anchor | target / tolerance | measured |
|---|---|---|
| 1 BW: nn hopping vs EP 2017 Eq. (27) | < 0.2/L | 6.9e-3, 2.6e-3, 1.5e-3, 9.8e-4 (L = 20, 40, 60, 80) |
| 1 BW: resummed profile vs 2πx(L−x)/L | < 1.5/L² | 2.0e-3, 5.6e-4, 2.6e-4, 1.5e-4 |
| 1 BW: β(1)/2π − 1 → 0 as −1/L | < 1.1/L + 1e-3 | −0.048, −0.024, −0.016, −0.012 |
| 1 BW: ε_max vs EP Eq. (54) | < 0.05 | +0.040, +0.021, +0.016, +0.013 |
| 1 BW: h₃ peak vs EP Eq. (26) (L = 60) | < 6 %, converging | −6.8 % (40), −4.0 % (60), −2.7 % (80) |
| 1 BW: ring chord form (N = 200, L = 40) | < 1.5/L² | 4.6e-4 |
| 2 CH: resummed local vs 2π/z′ (scales 1, 2, 4) | 4e-2, 8e-3, 2e-3 | 2.1e-2, 3.5e-3, 8.1e-4 |
| 2 CH: nonlocal row sums (interior) vs 2π/((x−x_c)z′(x_c)) | 0.15, 2e-2, 5e-3 | 9.1e-2, 9.9e-3, 2.2e-3 |
| 2 CH: smeared sin² bilinear | 5e-2, 1.5e-2, 4e-3 | 2.3e-2, 7.1e-3, 1.7e-3 |
| 2 CH: 4×4 Gaussian grid, Frobenius (scales 2, 4) | 2e-2, 5e-3 | 9.8e-3, 2.0e-3 |
| 2 CH: I(A₁:A₂) vs (1/3) log cross-ratio (scales 2, 4) | 1e-3, 5e-4 | −3.2e-4, −8.0e-5 |
| 2 CH: teleportation P₂(τ) vs sin²θ(τ), Eq. (59) | 8 % | 4.7, 3.6, 0.4, 3.0 % (τ = 0.1, 0.2, 0.3, −0.1) |
| 2 CH: packet centroid vs trajectories Eqs. (53)/(55) | 0.5 site | ≤ 0.36 (two intervals), ≤ 0.46 (one) |
| 3 first law δS = Tr(h δC_A), independent routes | 1e-8 | 2.6e-10, 3.8e-10, 2.2e-11 (one interval); ≤ 1.8e-10 (two) |
| 4 modular flow unitarity, spectrum of C_A | 1e-12 | ≤ 3e-15 (invariance ≤ 1e-14; eigh vs expm ≤ 2.3e-13) |
| 5 n = 3, gap = L: summed weight vs CH ∝ 1/L² | A = r L² constant to 6 %; extrapolation < 1e-3 | A = 1.906, 1.930, 1.947 (L = 12, 16, 24); r_∞ = +8e-05 |
| 6 ξ_h/ξ from the local decay rate at gaps ≥ 6 ξ | \|ξ_h/ξ − 1\| < 0.06 | 1.031, 0.992, 0.980 (m = 0.1, L = 12, gaps 32…128) |
| 7 CTM: edge slope and ε vs (2/π) k I(k′), π I(k′)/I(k) | 1e-12 | ≤ 1.2e-14 (m = 1, L = 40; m = 0.5, L = 80); level ratios 3, 5, 7 to 1e-8 |
| 7 CTM: near-edge block vs `lattice_boost` (both diagonals) | 1e-08 | 3.8e-09, 6.0e-12 |
| 7 the middle of a finite segment is the triangle, not the boost | bond at 3L/4 = boost/3 to 2 % | 0.3333, 0.3326 (triangle form to 1e-5, 2.3e-3) |
| 7 fitted s(ξ = 10, 5, 2.1, 1.1) vs the closed form | 5e-3 (ξ = 10), 5e-4 (ξ ≤ 5) | −4.2e-04, −2.0e-08, +5e-13, 0 |

The persistent lattice correction of the nearest-neighbour hopping (EP 2017: 7.6 % at the
centre of one interval) is 15.7 % at the centre of each of two equal intervals (stable
across scales 2 and 4 to 1e-3) — a lattice number, not a CH deviation; the resummed
profile removes it.

---

## What is new and what is a consistency check

Consistency checks: anchors 1–4 above (CH 2009; EP 2017; ETP 2022, whose row-sum and
perpendicular-sum prescriptions the dictionary reproduces independently; the modular-flow
teleportation of CH Sec. 3.3, which we believe has not previously been checked on a
lattice but is a direct consequence of the CH kernel); the triangular off-critical
profile (Eisler–Di Giulio–Tonni–Peschel 2020 for dimerized chains; here the
staggered-mass chain).

Lattice numerics not previously published, to our knowledge (from `data/s2_*` –
`data/s5_*`) — **measurements of known formulas, not new formulas** (docs/PRIORITY.md):
the lattice verification of CH's n-interval nonlocal term for n = 3, 4 (CH 2009 proves it
for any n; ETP 2022 checked n = 2 on the lattice; Arias–Blanco–Casini–Huerta, PRD **95**,
065005 (2017) checked the *local* term of massive fields on one and two intervals) with its
onset (gaps ≳ L) and its failure at gaps of a few sites; the screening of an outer pair by
an intermediate interval — a property the CH weight 2π/((x − x_c) z′(x_c)) already
contains, since z(x) carries every endpoint — and the finding that the *summed* lattice
weights follow CH even where the row-by-row structure does not; the separation exponents;
the linear extent of the off-critical BW profile (its slope is Eisler 2025, below); the
decay length ξ_h of inter-interval modular couplings relative to ξ, measured
asymptotically (ξ_h = ξ — the value one expects, since the nonlocal part of h is built from
correlators decaying as e^{−d/ξ}, but not previously measured) rather than as a crossover
value; and the 1/L² law with which the n ≥ 3 CH formula is approached, with its
extrapolation.

The gapped slope s(m) = (2/π) k I(k′) with k = 1/√(1 + m²) = sech(1/ξ) is **not new**: the
first pass of this candidate conjectured it from the CTM *structure* (Peschel–Kaulke–Legeza
1999, Eqs. (9)–(15); Eisler–Di Giulio–Tonni–Peschel 2020, Eqs. (38)–(41), whose dimerized
hopping chain has k = (1 − δ)/(1 + δ); the motivation sketched in `vacuum/modular/ctm.py` —
sublattice particle–hole map, dispersion matching to a TI field h = e^{−2/ξ}, the Landen
relation) and confirmed it to 1e-14 without knowing that the derivation exists — Eisler, J. Stat. Mech. (2025) 013101, arXiv:2410.16433, Sec. V,
Eqs. (64)–(72): for exactly this chain the correlation matrix commutes with the tridiagonal
boost T (t_m = −m/2, d_m = (−1)^m μ (m − ½)), giving H = 4κK(κ′)T, κ = 1/√(1 + μ²), spectrum
2πK(κ′)/K(κ)·(2ℓ ∓ ½), i.e. 2π s(m) = 4 k I(k′), k(m) = sech(1/ξ) and odd multiples of
ε = π I(k′)/I(k). The "open problem" of deriving k(m) and s(m) recorded in
`vacuum/modular/ctm.py` and in the first version of this MANIFEST is closed by that paper;
the module docstring now cites Eisler 2025, states the Eisler → module dictionary once and
labels the closed forms as his (the `ctm_` prefixes are kept for API stability). What this candidate contributes is the certified-precision,
zero-parameter confirmation on two independent observables — inverting the measured level
spacing alone gives a modulus agreeing with sech(1/ξ) to 2e-16…2.3e-14 at m = 0.3, 1, 0.5, 2,
and feeding that k (which never sees the slope) into s = (2/π) k I(k′) predicts the
real-space edge slope to 0…1.3e-14 — plus the finite-segment crossover from the
half-infinite line to the triangle and to the critical 1 − 1/L (Limitations).

---

## Limitations and open problems

- Sizes: the n ≥ 3 discovery table of `data/s2_*` uses L = 10–12 per interval; the 1/L²
  trend and the extrapolation are measured separately in `data/s4_*` at L up to 32
  (96 sites, dps 106). The *summed* weight converges cleanly as A/L²; the row-by-row
  interior deviation is noisier (single rows near the trimmed 20 % edge window dominate)
  and its negative extrapolation, −4e-03 to −6e-02, is an artefact of that noise, not a
  measured disagreement.
- ξ_h/ξ = 1 is read off the *local* decay rate at gaps of 8–16 ξ, at four correlation
  lengths and three interval sizes; the scatter 0.026 is the whole uncertainty quoted. The
  dependence on L is real at intermediate gaps (larger intervals reach the asymptote
  later), and the Frobenius norm of the whole off-block — which mixes many distances — is
  still at effective lengths up to 1.8 ξ at gaps of a few ξ.
- The CTM closed form is a half-infinite-chain statement, verified within a few ξ of ONE
  edge of a long segment; its window is bounded by the *other* edge, not by the distance
  from its own. At m = 1, L = 40 the two CTM diagonals hold to 3.8e-09 over the first 8
  sites (far edge 21 ξ away) but only to 3e-05 over the first 12 (far edge 14 ξ). The
  `block_rel_dev` column of `data/s6_*` is quoted over min(L/2, 6 ξ) sites and therefore
  *shows* that crossover (0.0 at m = 2, 0.42 at m = 0.02) rather than a failure of the
  closed form. In the middle of a finite segment the profile is the triangle
  2π s(m) min(x, L − x) (Eisler et al. 2020, Eqs. (42)–(44)); near criticality (ξ ≳ L) the
  edge slope crosses over to the critical 1 − 1/L (at ξ = 50, L = 40 the measured
  nearest-neighbour slope is 0.9873, between the critical 0.975 and the CTM 0.9999).
- The modulus map k(m) = sech(1/ξ) is derived in Eisler (2025), Sec. V; here it is verified
  at certified precision (see above).
- Half filling only; one mass model (staggered potential); the bosonic side is the
  Layer-1 first-law machinery plus `region_modular`/`modular_energy_profile` — no bosonic
  multi-interval reference exists in closed form.
- The long-range couplings of a gapped chain *near* criticality, where Eisler et al. 2020
  (Sec. 6) could not build a consistent continuum picture, are untouched.

## Reproducing

```
.venv/bin/python papers/multi-interval-modular-hamiltonians/notebook.py           # ~10 min, writes data/
.venv/bin/python papers/multi-interval-modular-hamiltonians/notebook.py --quick   # ~2 min
.venv/bin/python -m pytest tests/test_modular_lab.py -q                           # 32 anchors, ~64 s
.venv/bin/python -m pytest -q                                                     # admissibility gate
```
