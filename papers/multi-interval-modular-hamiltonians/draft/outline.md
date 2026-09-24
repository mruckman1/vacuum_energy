# Nonlocal structure of multi-interval modular Hamiltonians on free-fermion chains — outline

Numbers are from `data/summary.json` (generator sha256 `b5b9467c…`); nothing here is
independent of that file.

**Priority (docs/PRIORITY.md): REPRODUCTION-AT-CERTIFIED-PRECISION** — of Casini–Huerta
2009 (the n-interval formula, screening included), Eisler–Tonni–Peschel 2022 (the lattice
dictionary), Eisler–Di Giulio–Tonni–Peschel 2020 (the gapped triangle) and Eisler 2025 (the
staggered-chain closed form of Sec. 5c) — plus lattice numerics at n = 3, 4 and the
asymptotic ξ_h = ξ measurement. The title and abstract must say so.

## 1. Setting and the numerical fact

- Half-filled hopping chain, t = 1/2, k_F = π/2, v_F = 1; h = ln((1 − C_A)/C_A).
- ε_max ≈ 1.7627 L (EP 2017 Eq. (54); reproduced to 0.04): the real-space h is dominated
  by modes float64 cannot resolve beyond ~18 sites (measured: 42 % error at the centre of a
  40-site interval, 8e-4 at its edge). Closed-form C (infinite chain, ring, staggered
  mass) + mpmath diagonalization with automatic escalation; a `ModularPrecisionWarning`
  is the fermionic precision audit.

## 2. The lattice-to-continuum dictionary (D1–D6)

- Site j ↔ cell (j, j+1); chiral gauge H_R(i,j) = e^{ik_F(j−i)} h_ij; H_R = conj(H_CH);
  local weight from the resummed odd-range hoppings; nonlocal weight from alternating
  row sums; smooth test functions; modular-time direction; ring ↔ circle.
- Coincides with ETP 2022 Eqs. (31)–(32), derived here from the continuum kernel.

## 3. Anchors (consistency checks) — Figures 1–3

- Fig. 1: single interval, β_nn(x) vs EP Eq. (27) (0.14/L) and resummed β vs the
  conformal parabola (0.9/L²); inset: β(1)/2π − 1 = −1/L.
- Fig. 2: two intervals, scales 1/2/4: local resummed weight vs 2π/z′; alternating row
  sums vs 2π/((x−x_c)z′(x_c)); the 4×4 smeared-bilinear grid; all ∝ 1/L².
- Fig. 3: modular flow — P₂(τ) vs sin²θ(τ) (CH Eq. (59)), centroids vs the CH
  trajectories; the left-mover mirror.
- Table: first law (1e-10), unitarity (1e-15), mutual information vs cross ratio (1e-4).

## 4. Three and four intervals (new lattice numerics; the formula is CH 2009's) — Figures 4–6

- Fig. 4: the four-interval h (gap 12) with the CH conjugate hyperbolae overlaid; the
  interior row-sum deviations per pair (≤ 4.1 % for gaps ≥ 12; 28 % at 6; breakdown at 3).
- Fig. 5: screening — outer-pair summed weight with/without the middle interval, lattice
  vs CH (0.56…0.98 vs 0.59…0.98 for gaps 3…48): the CH formula predicts screening and the
  lattice obeys it even where the row-by-row structure has not converged.
- Fig. 6: decay with separation — summed weight exponent −0.63 (CH −0.62, crossover
  toward 1/d), Frobenius −0.92; locality score 0.97–0.999; off-block fraction 8e-5–2.5e-2.
- Unequal intervals: (8, 16, 12) and (16, 8, 12) with gaps 5–11: deviations up to 46 % —
  the gap, not the interval ratio, controls the onset.

## 4b. The 1/L² approach to the n-interval formula and its extrapolation (new lattice numerics) — Fig. 6b

`data/s4_multi_large_L.npz`. At fixed gap/L the *summed* bilocal weight per pair
approaches the n-interval CH weight as A/L²:

| three equal, gap = L | L = 12 | 16 | 24 | 32 |
|---|---|---|---|---|
| deviation | 1.32e-2 | 7.54e-3 | 3.38e-3 | 1.91e-3 |
| × L² | 1.906 | 1.930 | 1.947 | 1.953 |

Least squares in 1/L² extrapolates to **+8e-05** (A = 1.898, max fit residual 4.4e-05):
the CH n-interval formula is exact in the continuum limit at the 1e-4 level. Same fit,
gap = L/2: A = 3.76, extrapolation +4.4e-04; four equal intervals, gap = L: A = 2.27,
+5.4e-04. The row-by-row interior deviation (4.1 % at L = 10–12 in Sec. 4) falls to 3.5e-3
at L = 24 and 2.3e-3 at L = 32 but is noisy — its own 1/L² extrapolation, −4e-03…−6e-02,
is a fit artefact of single rows near the trimmed edge window, and is reported as such.

## 5. Off-critical chains (new lattice numerics; the slope's closed form is Eisler 2025, Sec. 5c) — Figures 7–8

- Fig. 7: β_nn(x) for m = 0.02…1 (ξ = 50…1.1) at L = 40: parabola → rounded triangle;
  edge slope 2π·s(m) with s = 0.987, 0.996, 0.997, 0.990, 0.945, 0.835; linear extent 2, 2,
  3, 6, 12, 16 bonds; plateau up to 1.48× the critical apex (74 % of the ideal triangle).
- Fig. 8: two intervals (L = 12), Frobenius norm of the off-block vs gap for each m:
  exponential with ξ_h/ξ = 1.00, 1.34, 1.43 (ξ = 10, 5, 2.1) — **superseded by Sec. 5b: these
  are single-exponential window fits over gaps of a few ξ, i.e. crossover values; the
  asymptotic length is ξ_h/ξ = 0.9985 ± 0.0257 = 1**. The critical curve is a power
  law (exponent −0.89); the gapped three-interval outer pair is suppressed as e^{−Δ/ξ_h}.

## 5b. ξ_h/ξ is asymptotically 1 (new measurement of the expected value) — Fig. 8b

`data/s5_xi_h_asymptotics.npz`: ξ = 5, 10, 20, 40 (m = 0.2, 0.1, 0.05, 0.025), interval
sizes L = 12, 24, 48, gaps out to 16 ξ (36 configurations, dps up to 108). A single
exponential fitted over a *window* of gaps returns an effective length contaminated by the
algebraic prefactor of the massive propagator — that is the origin of the 1.3–1.4 of
Sec. 5. The asymptotic statement needs the **local** rate
ξ_h^loc(d) = Δd / ln[W(d)/W(d + Δd)]. On the alternating row weight (the dictionary-D2
observable) the ten local rates at gaps ≥ 8 ξ give

  **ξ_h/ξ = 0.9985 ± 0.0257,**

i.e. ξ_h = ξ: the modular nonlocality between two intervals decays with exactly the
correlation length, with no anomalous factor. Larger intervals reach the asymptote later
(at L = 24 the local rate is still 1.05 at d = 11 ξ); the Frobenius norm of the whole
off-block, which mixes many distances, is still at effective lengths up to 1.8 ξ at gaps
of a few ξ and is reported alongside.

## 5c. The gapped slope in closed form: Eisler's boost operator, reproduced at certified precision — Fig. 8c

`data/s6_ctm_slope.npz`, `vacuum/modular/ctm.py`. For a gapped integrable chain the
half-infinite entanglement Hamiltonian is the CTM operator: the physical Hamiltonian with
each local term multiplied by its distance from the boundary, with an exactly equidistant
single-particle spectrum (Peschel–Kaulke–Legeza, Ann. Phys. (Leipzig) **8**, 153 (1999),
Eqs. (9)–(15); Eisler–Di Giulio–Tonni–Peschel, J. Stat. Mech. (2020) 103102,
Eqs. (38)–(41)). For the staggered-mass chain the exact half-infinite result is Eisler,
J. Stat. Mech. (2025) 013101, arXiv:2410.16433, Sec. V, Eqs. (64)–(72) — H = 4κK(κ′)T,
κ = 1/√(1 + μ²), T the boost with t_m = −m/2, d_m = (−1)^m μ (m − ½), derived by the
commuting-operator method — which in this candidate's units fixes the slope of Sec. 5:

  **s(m) = (2/π) k I(k′),  ε = π I(k′)/I(k),  k = 1/√(1 + m²) = sech(1/ξ), k′ = tanh(1/ξ),**

with h_{j,j+1} = −2π s (j + 1) t and h_jj = 2π s (j + ½) m (−1)^j from the edge, and levels
ε_l = (2l + 1) ε. Measured: the on-site edge slope matches to ≤ 5.2e-05 over m = 0.02…2 and
to ≤ 1e-08 for m ≥ 0.2; ε₀ to 1e-15 for m ≥ 0.2; the level ratios are 3, 5, 7 to 1e-8. The
fitted slopes of Sec. 5 are the closed form: residual **−4.2e-04** at ξ = 10 (L = 40),
−2.0e-08 at ξ = 5 and ≤ 1e-12 at ξ = 2.1 and 1.1. In the scaling limit s = 1 − 1/(4ξ²) + O(ξ⁻⁴) → 1,
recovering Bisognano–Wichmann.

*Where the form applies.* It is a **half-infinite** statement. Its window is bounded by the
*far* edge of the segment, not by the distance from its own: at m = 1, L = 40 the two CTM
diagonals hold to 3.8e-09 over the first 8 sites (far edge 21 ξ away) and only to 3e-05 over
the first 12 (far edge 14 ξ). Past the middle the profile is the triangle
2π s(m) min(x, L − x) (Eisler et al. 2020, Eqs. (42)–(44)): the bond at 3L/4 is 1/3 of the
CTM line and equals the triangle to 1e-5 (m = 1) and 2.3e-3 (m = 0.5). The CTM operator is
tridiagonal; the true h carries longer-range hoppings (4e-08 of the scale at m = 1, 1.1e-3
at m = 0.5 over the first 12 sites). Near criticality (ξ ≳ L) the edge slope leaves the CTM
value for the critical 1 − 1/L: at ξ = 50, L = 40 the measured nearest-neighbour slope is
0.9873, between the critical 0.975 and the CTM 0.9999.

*Status of the derivation.* Everything in the closed form is derived in Eisler (2025) for
exactly this chain; the first pass of this candidate had conjectured the identification
k(m) = sech(1/ξ) from the CTM structure of the transverse Ising and dimerized hopping chains
(Peschel–Kaulke–Legeza 1999; Eisler et al. 2020; the motivation sketched in
`vacuum/modular/ctm.py` — sublattice particle–hole map, dispersion matching to a TI field
h = e^{−2/ξ}, the Landen relation) and verified it to 1e-14 without knowing the derivation
existed. What is contributed here is the certified-precision, zero-parameter
confirmation on two independent observables: inverting the measured *level spacing* alone
gives a modulus agreeing with sech(1/ξ) to 2e-16, 2e-16, 3.6e-15, 2.3e-14 (m = 0.3, 1, 0.5, 2),
and feeding that k, which never sees the slope, into s = (2/π) k I(k′) predicts the
*real-space edge slope* to 0.0, 8.9e-16, 1.8e-15, 1.3e-14. The manuscript must present
Sec. 5c as a reproduction and cite Eisler 2025 in its first paragraph (docs/PRIORITY.md).

## 6. What this does not settle

- The three items Secs. 4b, 5b, 5c close are no longer open. What remains (the modulus map
  k(m) is derived in Eisler 2025 and confirmed here): other fillings and mass models; the bosonic
  two-interval problem (no closed-form reference); the long-range couplings of a gapped
  chain *near* criticality, where Eisler et al. 2020 (Sec. 6) could not construct a
  consistent continuum picture; and everything at n ≥ 3 with unequal intervals beyond the
  two configurations of Sec. 4.

## 7. Relation to the program

- The instrument for Layer 4's read head: nonlocality is now a measured function of
  geometry and correlation length. Feeds the toy-Jacobson leap (entanglement equilibrium
  on multi-interval balls) and the off-holography failure map (PLAN.md, Layer 4).
