# Toy Jacobson: the local modular Hamiltonian that entanglement equilibrium admits on a lattice — the 3-mode CHM parabola for the wave chain, ≥ 11–12 modes for z = 2 — outline

Result candidate `toy-jacobson` (PLAN.md Layer 4 discovery target, Layer 7 leap L2). Every
number below is produced by `notebook.py` (full mode, N up to 2400) and stored under `data/`;
the MANIFEST records the build. Status: **candidate; the full known-results suite was not re-run
for the free-weight pass (only the two owned test files: 31 passed), so the anchor-9 numbers await
the admissibility run** (see MANIFEST).

## The motivating objection, and what changed

An outside reviewer: *"K_loc is the CHM conformal-Killing integral, which is derived from
conformal symmetry. Finding that the first law then holds for conformal dynamics and fails for
Lifshitz is a consistency check on CHM, not Lorentz invariance emerging from entanglement. The
third-neighbour degeneracy test is clever and the scaling is clean. The headline is circular. The
version that would mean something optimizes over the weight function too and asks whether any
local K works for non-relativistic dynamics."* (full text in the MANIFEST). Correct. Secs. 2–5c
assume the CHM weight and are, as headlined before, a consistency check on CHM; the earlier
headline — "entanglement equilibrium recovers the low-energy dispersion of the wave equation" —
was circular in exactly this sense, and its "v² = 0.980 recovered" was the unit built into the
weight's normalisation (the vacuum of v²K is the vacuum of K; no state-based residual can measure
v). Sec. 5d is the version that means something: the weight is a free variable and the question is
whether *any* local, nonnegative, smooth K_loc works. The MANIFEST's "What was circular; what the
free-weight test establishes" is the itemised statement.

## 0. The claim in one paragraph

The entanglement first law δS = δ⟨K⟩ is an identity for any state and any dynamics; Jacobson's
entanglement-equilibrium argument (PRL 116, 201101 (2016), Eqs. (14)–(19)) has content only
because the ball's modular Hamiltonian K is *local* — the Casini–Huerta–Myers conformal-Killing
energy 2π∫(R² − r²)/(2R) T₀₀ (JHEP 05 (2011) 036, Sec. 2; Bisognano–Wichmann 1976) — which is a
property of Lorentz-(conformally-)invariant dynamics. We pose the lattice version with no
gravity anywhere: for a 1+1 harmonic chain with coupling matrix K, the residual
r(K) = max |δS − δ⟨K_loc⟩|/|δS| over centred balls and long-wavelength reflection-odd
squeeze variations, with K_loc = 2π Σ βᵢ hᵢ built from the lattice's own PSD energy density.
We find: (i) r → 0 for the massless nearest-neighbour wave chain as L^{-2.0…-2.4} with a continuum
extrapolation consistent with 0 at the 5×10⁻⁴ level; (ii) r stays O(1) and grows with the ball
for every non-relativistic dispersion (ω ~ k², ω ~ √k, 1/n² couplings); (iii) across 44 lattice
dynamics r ≈ 0.79/L² + 0.89·|ω(π/L)/(π/L) − 1| (rms scatter 24 %), equal to |1 − v| for a wrong
velocity; (iv) minimising r over K = m² + c₁L₁ + c₂L₂ from generic starts pins the velocity
(v² = c₁ + 4c₂ → 0.98, i.e. v* = 1 − 0.85/L²) and the mass (m → 0, resolution 0.01) but leaves the
coupling range c₂ a flat valley over [−0.4, 0.22]: entanglement equilibrium recovers the
*low-energy dispersion* of the wave equation, not its lattice discretisation; (v) the 1/L²
floor in (iii) is a property of the discretisation chosen, not a bound — β at the bond
midpoints gives 0.053/L² instead of 1.39/L² — but that does not by itself resolve the k⁴
term, which is exactly degenerate with a range-dependent ansatz error inside the (c₁, c₂)
family; adding c₃ makes ∂ρ/∂c₃ = 6 ∂ρ/∂c₂ a falsifiable test, which the site-centred ansatz
passes (6.04 ± 0.09) and the bond-centred one fails, and the k⁴ sensitivity is then
C = −0.974 ± 0.034 at λ = 4L (wavelength-dependent, L-independent); (vi) a ball *attached* to
the Dirichlet wall gives the same residual as a centred one **provided** the local form is the
boundary CFT's (B² − y²)/(2B) — with the bulk parabola it is 39–3015× worse — while a ball
*detached* from the wall stays obstructed at O(1) (its modular Hamiltonian carries a bilocal
image term); and (vii) projecting the ball's zero mode out of the variation removes the O(1)
reflection-even obstruction (2.9 → 0.044) without making the even sector consistent with the
odd one. **(viii) The weight set free (Sec. 5d):** with K_loc = Σ_k w_k O_k over the lattice's
local energy pieces and w ≥ 0 from a three-mode smooth family, minimised over long-wavelength
single- *and two-mode* squeezes (rank 4, 6, 7–8 at 10⁻⁴, 10⁻⁶, 10⁻⁸ — the regime ball ≪ λ is
low-rank, and a per-site weight fits any dynamics), the wave chain's minimiser is the CHM
parabola (coefficients (1.0010, −1.0007, −0.0003)·πR/2, overlap 1.0000, normalisation 1.001)
with r_free = 4.1e-05, 1.2e-05, 2.5e-06 at L = 8, 16, 32, 20–130× below the CHM discretisation;
the Lifshitz chain has r_free = 0.270, 0.507, 0.679 with three modes but 7.0e-04, 6.7e-03,
8.2e-02 with four — a per-ball weight exists at fixed L — and a profile R^z f(x/R) shared across
the three balls fits it too once ≥ 11–12 modes are allowed (0.091 at 8 modes, 0.023 at 11–12
with z = 2, 5.5e-03 at 11 with z = 3, 6.9e-04 at 12 hat modes: below 1e-02 by 11–13 modes and below 1e-03 by 12–14, z-dependent; 5e-04 only by 20 Chebyshev modes),
against 3 Chebyshev modes for the wave chain: what discriminates is the *complexity* of the
admissible weight, not its existence; the free weight absorbs the velocity exactly and a mass per ball, the shared profile sees
the mass (2.7e-02 at m = 0.2 with eight modes; 2.1e-03 at m = 0.01 with three), and the
constrained fit with the shared profile lands at m/v = 0.002, c₂/c₁ = −0.008. Items (i)–(vii) are consistency checks on CHM and
lattice statements about its discretisation; (viii) is the claim.

## 1. Setup (what is defined, what is assumed)

- Dynamics H = ½(p·p + xᵀKx), Dirichlet chain, N even; balls = centred intervals of even L.
- Local energy density hᵢ: PSD bond split (as `vacuum.inequalities.qei`), Σᵢhᵢ = H exactly;
  the site-symmetrised split differs by a discrete divergence (`split='site'`).
- Ansatz K_loc = 2π Σ_{i∈B} βᵢ hᵢ, βᵢ = (i − a + ½)(b − i + ½)/(b − a + 1): the CHM weight at
  site centres. The edge hᵢ contains half a bond straddling ∂B (an O(1/L) lattice ambiguity).
  Sec. 5b measures three other discretisations of the same continuum object (β at the bond
  midpoints, and two Richardson mixes) and a boundary-adapted weight for a ball attached to a
  Dirichlet wall (Sec. 5c).
- Variations: squeeze one normal mode u_a of K (x → e^ε x, p → e^{−ε} p along u_a), wavelength
  λ = 4L, 8L, 16L. δV_xx = uuᵀ/ω, δV_pp = −ω uuᵀ is smooth on the lattice scale; δ⟨T₀₀⟩ ≈
  const over the ball (Jacobson's ℓ ≪ L_excitation, Eq. (18)). Local coupling changes are *not*
  used: on a lattice they are local rescalings whose xx and pp energy responses are cutoff-
  dominated and cancel to a delicate remainder (Sec. 6).
- Parity: modes with a node at the ball centre are exactly blind to every reflection-even
  Williamson mode of the ball (measured overlap ≤ 1.5×10⁻¹⁷), in particular to the massless
  scalar's zero mode. Reflection-even variations have r = 3.07, 2.83, 2.15 at L = 8, 16, 32 for the
  wave chain itself: the exact modular Hamiltonian carries a term ~ g(ν₁) φ̄² (row sums
  Σⱼ G^{xx}_{ij} = 0.076 (L = 16), 0.044 (L = 32) per row, decaying only like 1/ln N) with no
  local-energy counterpart — the ½ ln ln(1/mL) entropy term of the 2d massless scalar
  (Casini–Huerta 2009), a property of the continuum theory, not of the lattice. The toy is posed
  in the odd (derivative-algebra) sector. Sec. 5c replaces the parity argument by an explicit
  projection of the ball's top Williamson mode out of the variation, which is what an off-centre
  ball (no reflection parity) requires.
- δS is (½)Tr[G_B δV_B] with the exact Gaussian modular Hamiltonian (Williamson); the identity
  is checked by finite differences of the entropy of the exactly squeezed ball state.

## 2. Anchor 1 — the massless wave chain approaches the CHM local form (Fig. 1)

`data/A_massless_chain.npz`, N = 2400.

| L | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|---|---|
| r (max over λ/L = 4, 8, 16) | 9.1e-2 | 3.9e-2 | 2.2e-2 | 9.6e-3 | 5.3e-3 | 2.3e-3 | 1.2e-3 | 3.4e-4 |

- Lattice-correction exponent (L ≥ 8): −2.27 (max); −2.03, −2.42, −2.27 for λ/L = 4, 8, 16.
- Continuum extrapolation: r_∞ = −1.9×10⁻⁴ (fit r = r_∞ + A/L², A = 1.41) and +4.6×10⁻⁴ (fit with
  the measured exponent): consistent with 0 at the 5×10⁻⁴ level set by the scatter of the two.
- Cross-check on the exact modular Hamiltonian (Di Giulio–Tonni 2020 combination of all
  diagonals): at L = 16, pp row sums / 2πβ = 1.08 (edge) … 1.11 (centre), r²-resummed xx weight /
  2πβ = 1.01 at the centre; `locality_score` (bandwidth 1) = 0.994 (L = 16), 0.988 (L = 32).
- Constrained velocity: minimising r over K = v²L at fixed L gives v* = 0.9466, 0.9868, 0.9968,
  0.9993, 1.0003 for L = 4, 8, 16, 32, 64: 1 − v* ≈ 0.85/L² — the lattice renormalises the speed
  of light seen by entanglement equilibrium by O(a²/L²), and r(v*) = 3.3e-2 … 1.6e-4.

## 3. Anchor 2 — the exact first law holds for every dynamics (Table 1)

`data/B_first_law_audits.npz`, N = 800, L = 4, 8, 16, nine dynamics (wave, m = 0.2, c₂ = +0.1,
improved c₂ = −1/12, c₂ = −1, Lifshitz, z = ½, random W = 0.5, 1/n² couplings). Independent routes
(entropy finite differences of the exactly squeezed ball vs (½)Tr[G δV]) agree to
≤ 8.6×10⁻⁹ with the precision ladder (float64 → mpmath when a symplectic eigenvalue is within
10⁻¹⁰ of ½); ≤ 3.5×10⁻⁹ for all but the Lifshitz chain, whose ball covariance (⟨x²⟩ ~ N²) puts the
float64 margins below the floor and gives 1.3×10⁻⁷ without the ladder. The residual therefore
measures locality, not numerics.

## 4. Anchor 3 and the map — r(K) versus departure from Lorentz invariance (Fig. 2)

`data/C_residual_map.npz/.json`, N = 1600, L = 8, 16, 32, 44 dynamics; δ_v1(L) ≡
|ω(π/L)/(π/L) − 1| from the bulk symbol (the ansatz assumes v = 1).

- Teeth: Lifshitz ω = (2 sin k/2)² gives r = 0.954, 0.975, 0.991 (44×, 185×, 910× the wave chain);
  z = ½: 4.3, 5.9, 9.9; z = ¾: 1.3–2.3; z = 1.25: 0.55–0.70; z = 1.5: 0.79–0.91 — all growing with L.
- Long-range cₙ ∝ n^{−α} (v = 1): α = 2 (ω ~ √k): 0.54, 0.37, 0.16; α = 3: 0.24, 0.15, 0.065;
  α = 4: 0.045, 0.022, 0.009; α = 6: 0.023, 0.006, 0.001 (indistinguishable from nearest-neighbour).
- Velocity: r = 0.4975/0.4994/0.5000 (v = ½), 0.196/0.199/0.200 (v = 0.8), 0.277/0.257/0.251
  (v = 1.25), 1.04/1.01/1.00 (v = 2): r = |1 − v| to 4 % at L ≥ 16.
- Mass: m = 0.05: 0.024, 0.071, 0.153; m = 0.2: 0.147, 0.251, 0.327; saturates near 0.33 for
  mL ≫ 1 with m ≤ 1; dips to 0.04–0.06 at m = 4 (Sec. 5) and grows again, 0.53 at m = 8.
- Random mirror-symmetric bonds cₓ ∈ [1 − W, 1 + W]: W = 0.1: 0.002–0.041; W = 0.3: 0.006–0.074;
  W = 0.6: 0.012–0.11 — seed-dependent, no systematic decay with L at these sizes.
- NNN at v = 1 (c₁ = 1 − 4c₂): r(L = 16) = 0.0586 (c₂ = −3), 0.0198 (−1), 0.0065 (−0.3),
  0.0041 (−0.15), 0.0047 (−1/12), 0.0053 (0), 0.0058 (+0.1), 0.0065 (+0.2).
- The map: weighted fit over the 71 (dynamics, L) points with δ_v1 ≤ 0.6 gives
  r = 0.785/L² + 0.891·δ_v1, rms relative scatter 0.24. Per-family tracking coefficients
  (r − A/L²)/δ_v1: velocity 0.99 (0.79–1.20), NNN 1.02 (0.96–1.11), long-range 0.92 (0.59–1.04),
  mass 1.24 (0.58–2.7), z ≠ 1: 2.3 (1.5–7.3). The additive floor is the ansatz's own
  discretisation: the improved chain (δ_v1 = 8×10⁻⁶) has r = 0.0192, 0.0047, 0.0009 = 1.2/L².
- So: r tracks the departure of the bulk dispersion from ω = k at the ball's own scale, to a
  factor ~1 for analytic (velocity, mass, coupling-range) departures and ~2 for changes of the
  dynamical exponent, on top of a 1.2/L² floor that no dispersion can remove — but that a
  different *discretisation of the same continuum ansatz* can (Sec. 5b).

## 5. Anchor 4 — constrained dynamics: where the minimiser lands (Fig. 3)

`data/D_constrained_dynamics.npz`, N = 512, balls L = 8, 16, λ/L = 4, 8, 16; family
K = m² + c₁L₁ + c₂L₂, Nelder–Mead (max objective, 2 restarts, ≤ 300 evaluations each) with the
box c₁ ∈ [0.05, 4], c₂ ∈ [−0.5, 0.5], m ∈ [0, 1]. Wave chain reference r = 0.0217.

| start (c₁, c₂, m) | landing (c₁, c₂, m) | v² | r |
|---|---|---|---|
| (2.0, −0.3, 0.2) | (1.634, −0.163, 0.000) | 0.980 | 0.0064 |
| (1.5, 0.1, 0.05) | (0.058, 0.231, 6e-4) | 0.981 | 0.0140 |
| (0.5, 0.25, 0.4) | (0.898, 0.247, 1.0 = bound) | 1.88 | 0.054 |
| (0.3, 0.4, 0.8) | (0.779, 0.274, 1.0 = bound) | 1.87 | 0.054 |

- Recovered sharply *under the CHM weight* — a unit, not a measurement, and withdrawn as a velocity
  claim (preamble; Sec. 5d): v² = 0.980 (v = 0.990, the L = 8–16 value of v* of Sec. 2;
  the residual doubles within ±0.013 in c₁), and masslessness (m = 0; the residual doubles by
  m = 0.010).
- Not recovered: the coupling range. Along c₂ at v = 1 the residual is 0.0166 (c₂ = −0.16) to
  0.0336 (c₂ = −0.4) and 0.022 (c₂ = 0) — every c₂ in [−0.4, 0.22] is within a factor 2 of the
  minimum ("resolution" width 0.61 = the whole physical range). The two low-m landings sit at
  opposite ends of the valley (c₂ = −0.16 and +0.23) with residuals 3.4× and 1.5× *below* the
  nearest-neighbour chain's. The best landing has k⁴ coefficient +0.08 in ω² (the improved
  Laplacian would be 0, nearest-neighbour −1/12): the constraint cannot see k⁴ structure below
  the 1.2/L² floor.
- Starts with m₀ ≥ 0.4 are captured by a shoulder at the m = 1 bound (r = 0.054 > r_wave): the
  gapped branch. Along m at v = 1 (balls 4, 8, 16) r = 0.09 (m = 0), 0.33 (0.5), 0.32 (1), 0.23
  (2), 0.12 (3), 0.14 (4), 0.28 (5), 0.67 (8), 1.65 (16), 3.4 (32) while the L = 8 ball entropy
  falls from 1.33 to 9×10⁻³ (m = 3) and 2×10⁻⁶ (m = 32): a secondary dip at m ≈ 3–4 where the
  ball is nearly a product state, bounded away from 0 and above the massless value with the
  L = 4 ball included, but comparable (0.04–0.06) at L = 8, 16 alone. Jacobson's premise that the
  UV entanglement is that of the gapless vacuum is what excludes it; here it is the mass bound.

## 5b. The ansatz floor is a choice, and the k⁴ coefficient (new) — Fig. 4

`data/E_ansatz_and_k4.npz`, N = 2400, L = 8, 16, 32, λ/L = 4, 8, 16.

- **Floor.** The CHM weight can be attached to the lattice energy density in more than one
  way; all agree in the continuum and differ at O(1/L²). Putting β at the *bond midpoints*
  (the midpoint rule for the gradient energy — which also makes the bonds straddling ∂B drop
  out, since their midpoint sits exactly on the boundary) instead of the site centres lowers
  the massless chain's residual from **1.39/L²** to **0.053/L²** (A = r·L²: 1.394, 1.364,
  1.248 site vs 0.053, 0.058, 0.174 bond at L = 8, 16, 32 — the L = 32 bond value is limited
  by N, not by the ansatz: at N = 6400 it is 0.052/L² at L = 8, 12, 16). Richardson mixes:
  0.5·site + 0.5·bond gives 0.67/L², 0.1·site + 0.9·bond gives 0.09/L².
- **Why that does not settle k⁴.** Within the family K = c₁L₁ + c₂L₂ at v = 1 the k⁴
  dispersion coefficient A₄ = Σcₙn⁴ = 1 + 12c₂ is an *affine* function of c₂, so the
  dispersion and any range-dependent discretisation error of the ansatz are exactly
  degenerate — and both scale as 1/L². No amount of floor lowering separates them.
- **What does.** Add the third-neighbour coupling: with c₁ + 4c₂ + 9c₃ = 1,
  A₄ = 1 + 12c₂ + 72c₃ and ω(k)/k − 1 = −A₄k²/24, so a response that is a function of the
  dispersion alone must obey **∂ρ/∂c₃ = 6 ∂ρ/∂c₂** exactly. Over a 31-point (c₂, c₃) grid
  (c₂ ∈ [−1, 0.5], c₃ ∈ [−0.2, 0.2]) the signed residual ρ = (δS − δ⟨K_loc⟩)/δS is fitted as
  ρ = ρ₀ + λ₂c₂ + λ₃c₃ at each (L, λ/L):

  | λ/L | site: l₃/l₂ | site: C | bond: l₃/l₂ | bond: C |
  |---|---|---|---|---|
  | 4  | **6.043 ± 0.086** | **−0.974 ± 0.034** | 2.221 ± 0.161 | +0.641 ± 0.061 |
  | 8  | **5.989 ± 0.265** | +0.151 ± 0.014 | 1.951 ± 0.097 | +0.774 ± 0.045 |
  | 16 | **6.052 ± 0.093** | +0.461 ± 0.021 | 1.906 ± 0.082 | +0.815 ± 0.042 |

  (mean ± sd over L = 8, 16, 32; C ≡ ∂ρ/∂(ω(π/L)/(π/L) − 1) at the ball scale). The
  site-centred ansatz **passes** the test — for it, fitting in A₄ alone is as good as the free
  two-parameter fit (rms ratio 1.00–1.03 over all nine (L, λ/L) cells) — so its coupling
  response *is* the k⁴ dispersion, and C is the k⁴ coefficient. The bond-centred ansatz
  **fails** it (rms ratio 4.0–32.9): its low floor is bought with a range-dependent artefact,
  and it is not a better lattice CHM form. Floors at A₄ = 0 (dispersion-free), from the same
  fits: site −1.26e-2, −3.02e-3, −6.99e-4 and bond +6.6e-3, +1.9e-3, +5.4e-4 at
  L = 8, 16, 32 (λ = 4L) — the bond form's advantage on the nearest-neighbour chain does not
  survive once a second coupling is switched on.
- **Does the map's law sharpen?** Partly. At λ = 4L, which is what the max-over-wavelengths
  residual r follows, |C| = 0.974 ± 0.034 — consistent with 1 and sharper than the map's
  single 0.891, and now with an uncertainty and a validity test. But C changes sign and
  magnitude with the wavelength (−0.974, +0.151, +0.461) at fixed L, so
  r = A/L² + C·δ_v1 is a law *per wavelength*, not one law; the coarse fit of Sec. 4 stands
  as a summary, not as a sharpened statement.

## 5c. A second ball position, and the zero mode projected rather than avoided (new) — Fig. 5

`data/F_second_position_even.npz`, N = 1600, bond-centred β, parity-blind variations with the
zero mode projected out.

- **Projection instead of parity.** Project the ball's top Williamson mode (φ̄) out of the
  *variation* — both quadratures, in the ball's Williamson frame — so δS and δ⟨K_loc⟩ stay the
  same linear functional of the same δV. Odd variations are unchanged (they never saw it:
  8.42e-04, 2.63e-04, 3.29e-04 either way). Even variations fall from **2.93, 2.23, 2.07** to
  **0.0444, 0.0510, 0.167** at L = 8, 16, 32 — the O(1) obstruction *is* the zero mode. But
  the even sector does **not** become consistent with the odd one: it stays 53×, 194×, 508×
  larger and does not decay with L. After the projection δS is a near-cancellation (the even
  variation's whole energy response lives in the projected direction), so the *relative*
  residual saturates at the percent level however many even Williamson modes are removed.
  Canary: on the Lifshitz chain the projection changes nothing in the odd sector (1.19, 1.09,
  1.05) and makes the even sector meaningless (δS → 0).
- **A ball attached to the Dirichlet wall.** There the wall is not an entangling surface: the
  region has one entangling point and the exact conformal-Killing weight is
  β(y) = (B² − y²)/(2B) with y measured from the wall (the node of every mode) and
  B = x_b − x_wall. With it, r = **2.25e-02, 7.47e-03, 3.29e-03** at L = 8, 16, 32, and at the
  longest wavelength 3.0e-03, 5.1e-04, 2.5e-04 — the centred ball's odd-sector values
  (8.4e-04, 2.6e-04, 3.3e-04) to within factors 3.6, 1.9 and 0.8; the max over wavelengths is
  set by the λ = 4L row, where the ball sits inside the mode's first quarter-wave. **r stays consistent at a
  second position.** With the *bulk parabola* at the same position it is 0.880, 3.28, 9.93
  (39×, 439×, 3015× worse): the residual detects the boundary condition, and the local form
  the toy needs is the boundary CFT's, not the bulk one.
- **A ball detached from the wall** stays obstructed at O(1) at 3, 9 and 33 sites (1.02–10.6,
  and 0.098–0.590 at 33 sites) even with the image-corrected *local* weight 2π/z′ of the
  interval and its mirror. That is expected and is the point: an interval at a distance from a
  boundary has a **bilocal** term coupling each point to its image partner
  (Eisler–Tonni–Peschel 2022, J. Stat. Mech. 083101, Sec. 6), which no local energy density
  can represent — the same phenomenon the two-interval candidate measures in the bulk.

## 5d. The weight set free: is there any local modular Hamiltonian? (new) — Fig. 6

`data/G_free_weight.npz`, N = 1600, L = 8, 16, 32; fits and scans at N = 512, L = 8, 16.

- **Definition.** K_loc = Σ_k w_k O_k over the local energy pieces of the PSD bond split (site
  terms pᵢ²/2 + Mᵢxᵢ²/2 and bonds (cᵢⱼ/2)(xᵢ − xⱼ)² touching the ball; Σ O_k = H). Weight
  families: even Chebyshev profiles f(ξ), ξ = (x − x_c)/R, with n = 1–4 modes, sampled at the
  sites (bond weight = mean of the end sites, 0 outside) or at the bond midpoints (the two
  discretisations of Sec. 5b; CHM = (πR/2)(T₀ − T₂) is the n = 2 member); and one weight per
  site (mirror pairs tied) plus, in the bond discretisation, one weight for the straddling
  bonds. w ≥ 0 (else e^{−K_loc} is not a state). Variations: every reflection-odd mode of
  wavelength ≥ 4L, squeezed singly and in pairs (the first-order two-mode squeeze
  δV_xx = A(ω_a + ω_b)/(2ω_aω_b), δV_pp = −A(ω_a + ω_b)/2, A = (u_a u_bᵀ + u_b u_aᵀ)/2, exactly
  first-law-obeying by finite differences at ≤ 2.5e-09, and blind to the zero mode like the odd
  singles): 1275, 325, 78 variations. r_free = min over the family of max_v |δS_v − Σ w_k
  δ⟨O_k⟩_v|/|δS_v| by linear programming (HiGHS); r_free ≤ r_chm by construction. The numerical
  rank of the scaled design Q/|δS| — the number of independent conditions the family imposes on
  *any* local weight — is 4, 6, 7–8 (wave) and 3, 5, 7 (Lifshitz) at 10⁻⁴, 10⁻⁶, 10⁻⁸.
- **The wave chain: CHM found.** r_chm on this family 9.1e-04, 2.8e-04, 3.3e-04 (bond-centred;
  2.3e-02, 5.6e-03, 1.2e-03 site-centred). Three modes: r_free = 4.1e-05, 1.2e-05, 2.5e-06 with
  θ/(πR/2) = (1.0010, −1.0007, −0.0003), (1.0003, −1.0002, −0.0001), (1.0003, −1.0004, 0.0001)
  against CHM's (1, −1, 0): overlap 1.0000, normalisation 1.001, 1.000, 1.000 — the quartic
  mode is not used. Two modes already: 4.7e-05, 1.2e-05, 6.8e-06 at (1.001, −1.0005). Four
  modes: 5.0e-06, 6.0e-06, 2.5e-06, coefficients (1.002, −1.000, 0.002, −0.005). One profile
  R^z f(ξ) shared across the three balls: z = 1, three modes, 1.5e-04 with overlap 1.000 at every
  L; eight modes 2.6e-05; z = 2: 0.55. The site-centred discretisation is 1–2 orders worse at every
  n (3.0e-03, 7.2e-04, 1.5e-04) — the straddling bonds' weight, which the bond form sets to the
  boundary value, is what the site form cannot free.
- **Lifshitz: a weight exists — per ball at four modes, shared across balls at ≥ 11–12 — and
  the statement is parsimony.** r_chm 1.0–1.2. Three nonnegative modes: 0.270, 0.507, 0.679 — a
  property of the three-mode family. Four modes (still below the rank-5 family): **7.0e-04,
  6.7e-03, 8.2e-02** — a nonnegative local weight *does* fit any single ball, and the fit
  degrades 100× from L = 8 to 32 (an independent four-monomial basis: 6.9e-04, 5.6e-03 at
  L = 8, 16). Shared across L = 8, 16, 32 (the mode ladder, Fig. 6b; Chebyshev / hat / monomial
  bases, z = 1, 2, 3): 0.091 at 8 modes (z = 2) is an eight-mode family property; the floor
  falls to 0.033 (9), 0.025 (10), 0.023 (11–12) with z = 2, 5.5e-03 at 11 with z = 3, 6.9e-04 at
  12 hat modes — below 1e-02 by 11–13 modes and below 1e-03 by 12–14, z-dependent; 5e-04 only by 20 Chebyshev modes; beyond ~13 modes the family has more parameters than the variation
  family has conditions at the 10⁻³ level (per-ball rank 2–3 at 10⁻³, 8–9 in
  total). The wave chain: 1.5e-04 at 3 Chebyshev modes (1.5e-04 at 4 hats), 2.6e-05 at 8,
  1.9e-05 at 20, and ≥ 0.13 at every mode count for z ≠ 1. First below 1e-2 / 1e-3: Lifshitz
  11–13 / 12–14 modes depending on basis and z; wave 3 / 3. What survives is a mode-count gap
  of ~4×, not an existence gap — with enough modes every tested dynamics fits. Per site + boundary (5, 9, 17
  parameters, rank 5 at 10⁻⁶): 4.5e-05, 2.5e-06, 1.6e-06 — and 2.3e-06, 6.1e-07, 1.1e-07 for the
  wave chain: the "sharpest" family fits everything and says nothing; the recovered Lifshitz
  weight there is 20–550× CHM at scattered sites. Signed weights (not modular Hamiltonians):
  1.1e-03, 1.0e-03, 3.2e-04 with scale 10, 31, 84× CHM and a negative dip near the edge — the
  mechanism is that the Lifshitz energy density's response to the ball's entangled (linear-
  profile) modes vanishes in the bulk and has the wrong sign at the edge for a nonnegative
  bond-centred weight. Shared profile across L = 8, 16, 32 (15 conditions): best of
  {3, 4, 6, 8 modes} × {site, bond} × {z = 1, 2} is **0.091** (eight modes, bond, z = 2; signed
  6.6e-04); three modes 0.92 (z = 1), 0.74 (z = 2).
- **The rest of the family (per ball, three modes; N = 1600, L = 8, 16, 32; r_chm → r_free,
  then the shared-profile best).** z = ½: 0.83, 1.6, 2.7 → 5.3e-02, 4.8e-02, 3.9e-02 (the weight
  shrinks with L: z < 1); shared 0.38. z = 3/2: 0.71, 0.77, 0.83 → 0.14, 0.16, 0.12; shared 0.078.
  1/n²: 0.54, 0.30, 0.082 → 5.9e-02, 4.5e-02, 2.2e-02; shared 0.39. 1/n³: 0.17, 0.074, 0.031 →
  1.7e-02, 1.2e-02, 6.5e-03; shared 0.10. c₂ = +0.1: 1.2e-02, 2.8e-03, 4.6e-04 → 1.9e-03, 4.6e-04,
  9.8e-05 (θ = (0.99, −0.99, 0.004)); shared 4.6e-04. Improved Laplacian: 1.2e-02, 3.0e-03,
  9.6e-04 → 7.9e-05, 5.6e-05, 2.3e-05; shared 6.4e-05. Random W = 0.3: 3.0e-02, 2.9e-02, 1.6e-02
  → 5.7e-05, 2.5e-05, 3.4e-04; shared 1.0e-03. **Velocity**: v = ½ gives numbers identical to
  v = 1 with θ scaled by 2.000 — a unit. **Mass**: m = 0.2 gives 0.15, 0.25, 0.32 (CHM) →
  8.2e-06, 1.9e-06, 5.3e-06 (free) with scale 1.17, 1.33, 1.47 and the quartic mode 0.07, 0.16,
  0.27; m = 0.05: 2.4e-02, 7.1e-02, 0.15 → 6.5e-07, 1.9e-05, 2.1e-05; m = 1: 0.30, 0.32, 0.32 →
  2.9e-05, 2.2e-05, 5.8e-06 — absorbed per ball at every m (all ≤ 3e-05); the shared profile
  (best of 3–8 modes): 2.2e-02 (m = 0.05), 2.7e-02 (0.2), 9.7e-04 (1.0) against 2.6e-05
  massless, and with three modes at N = 512: 2.1e-03 (0.01), 6.6e-03 (0.02), 2.3e-02 (0.05),
  4.5e-02 (0.1), 6.4e-02 (0.2), 4.3e-02 (0.5), 1.7e-02 (1.0) against 6.3e-05 — the mass is seen
  only through scale covariance, and only in the crossover mL ~ 0.1–10.
- **Constrained dynamics with the weight free (N = 512, L = 8, 16, same starts as Sec. 5).**
  Per-ball free residual: (2.0, −0.3, 0.2) → (1.24, −0.11, 0.43), r_free = 2.3e-06; (1.5, 0.1,
  0.05) → (4.0 = bound, −0.015, 0.007), 5.0e-07 — both below the wave chain's own 3.7e-05:
  nothing is pinned. Shared profile (three modes, z ∈ {1, 2}): (1.5, 0.1, 0.05) → m/v = 2.3e-03,
  c₂/c₁ = −0.0076, r = 2.4e-05; (2.0, −0.3, 0.2) → the m = 1 and c₁ = 0.05 bounds, m/v = 5.5, at
  r = 4.3e-04: the gapped shoulder.
  Scans (v = 1): m → universal 6.3e-05 (0), 2.1e-03 (0.01), 6.6e-03 (0.02), 2.3e-02 (0.05),
  4.5e-02 (0.1), 6.4e-02 (0.2), 4.3e-02 (0.5), 1.7e-02 (1.0), per-ball free flat at ≤ 3.7e-05;
  c₂ (v = 1) → universal 6.9e-03 (−0.4), 1.5e-03 (−0.2), 2.5e-04 (−1/12), 6.3e-05 (0), 4.9e-03
  (+0.1), 8.1e-03 (+0.2), per-ball free 3.3e-03, 1.5e-04, 7.7e-05, 3.7e-05, 1.8e-03, 3.0e-03;
  v → identical at 0.5, 0.8, 1, 1.25, 2. So the coupling range is resolved *towards nearest
  neighbour* — a lattice statement (the bond-centred weight is exact for NN bonds), not a k⁴
  measurement (Sec. 5b remains that) — and the velocity never.

## 6. What the toy does and does not say

Does say: on a 1+1 lattice with no gravity, no area term and no Einstein equation, the
requirement δS = δ⟨K_loc⟩ for long-wavelength excitations with K_loc a local, nonnegative, smooth
weighting of the lattice's own energy density is satisfied by the wave chain with *one* weight —
the CHM parabola, found by the minimisation and not assumed (with cosine, Gaussian, hat, |ξ|^p
and monomial bases as well, overlap ≥ 0.9992) — and, for z = 2 dynamics, by weights that exist
too but are not parsimonious: four modes per ball (7.0e-04 at L = 8, 8.2e-02 at L = 32), ≥ 11–12
modes of a bumpy profile shared across ball sizes (0.09 at eight modes; below 1e-02 by 11–13 modes and below 1e-03 by 12–14, z-dependent; 5e-04 only by 20 Chebyshev modes).
The toy therefore tests the *parsimony* of the modular weight — a ~4× gap in mode count between
the wave chain and z = 2 — and does not establish that Lorentz-invariant dynamics are singled
out by the existence of a local modular Hamiltonian. It does not see the velocity
(a unit), it sees the mass only through scale covariance of the profile across ball sizes, and
it resolves the coupling range towards the nearest-neighbour discretisation. With the CHM weight
*assumed*, the residual is a quantitative, L-resolved measure of the lattice's distance from the
CHM form, and its response to a k⁴ deformation is a function of the bulk dispersion alone
(Sec. 5b) — that part is a consistency check on CHM and is now labelled as such. It also detects the
*boundary condition*: at a ball attached to a Dirichlet wall the boundary CFT's weight works
and the bulk parabola fails by 39–3015× (Sec. 5c). The 1+1 massless scalar's zero mode is an
obstruction in the reflection-even sector for *every* dynamics.

Does not say: anything about the area term, η, or gravity (nothing sources δA); d > 2; balls at
a *distance* from a boundary (their modular Hamiltonian is bilocal); the operator statement K_B = K_loc
(only its pairing with smooth variations — lattice modular Hamiltonians have long-range
xx couplings whose r²-resummation, not their nearest-neighbour entry, is the CHM weight);
variations that are local changes of the couplings (cutoff-dominated, Sec. 1). Nor anything about the most general banded
modular Hamiltonian (only weighted energy densities), nor about weights allowed to go negative
(recorded at 1e-3 for Lifshitz, not claimed), nor at parameter counts at or above the rank of
the long-wavelength family (5–6 per ball). Open: the
site-centred ansatz's 1.39/L² floor is unchanged (the bond-centred form that reaches
0.053/L² fails the pure-dispersion test and is not a better CHM form); the k⁴ sensitivity is
wavelength-dependent and no single C summarises it; balls *detached* from a boundary need a
bilocal term the toy does not have; the even sector remains without quantitative content even
after the zero mode is projected out; d = 2 (radial lattice) is untouched.

## Figures (from data/)

1. r(L) for the wave chain, odd and even sectors, with the L^{−2.27} fit and v*(L).
2. r versus δ_v1(L) for all 44 dynamics at L = 8, 16, 32 with the line 0.79/L² + 0.89 δ.
4. The (c₂, c₃) plane: ρ contours with the A₄ = const lines, for the site- and bond-centred
   ansatz side by side (the first has its contours parallel to A₄, the second does not);
   inset, the floor r·L² of the four discretisations.
5. r versus ball position (attached, 3, 9, 33 sites, centred) with the boundary-adapted and
   bulk weights; and the even-sector residual before and after the projection, against the
   odd sector, versus L.
3. The landscape: r along c₁, c₂ (v = 1 valley) and m around the best landing; r(m) on the
   gapped branch with the ball entropy.
6. The weight set free: (a) the recovered site weights of the wave chain at L = 8, 16, 32 on top
   of the CHM parabola, and the Lifshitz chain's best nonnegative and signed weights; (b) the
   mode ladder: the shared-profile residual against the number of modes (Chebyshev and hat,
   z = 1, 2, 3) for the wave and Lifshitz chains with the variation family's conditions at 10⁻³
   marked — the parsimony gap; (c) r versus r_free across the dynamics family; (d) the m and c₂ scans under the
   per-ball and the shared-profile residual.
