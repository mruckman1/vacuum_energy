# Interacting vacua, first numbers: entanglement decay and the exact lattice quantum-energy-inequality margin in the λφ⁴ chain

*Draft outline — papers candidate #3, PLAN.md Layer 7 leap L5 ("Interacting vacua").*
*Status: first numbers; methods anchored; every number is a lattice number at one mass and one lattice spacing, with the systematics named in §7. The QEI section reports the exact lattice infimum (§6.5), its diagnostics (§6.5.9, `section_diagnostics_A.md`), the measured-dispersion reference and the extremal-state decomposition (§6.6, `section_reference.md`), and the single list of what this candidate no longer claims (§6.7). The L5 claim is stated once, in §1, and `../MANIFEST.md` carries the same statement.*

All numbers below are produced by `../notebook.py` and live in `../data/`; every
quoted value names the file it comes from. Build, wall time and the admissibility
run are in `../MANIFEST.md`.

---

## 1. The claim in one paragraph

For the lattice λφ⁴ chain in its symmetric phase, DMRG ground states in a truncated
local oscillator basis (n_max Fock states at the occupation-minimizing frequency)
reproduce the free Gaussian vacuum of `vacuum.core` at λ = 0 to ≤ 1e-6 in energy,
covariance, block entropy and disjoint-block negativity (`data/s1_free_anchor.csv`,
`tests/test_interacting.py`), match second-order perturbation theory at small λ, and
pass the passivity audit. On that footing we report, as functions of λ at fixed bare
mass m = 0.5: (i) the mutual-information decay length of the interacting vacuum,
which tracks the Hartree-renormalized mass; (ii) the exact negativity of 1+1 and 2+2
site blocks and its sudden death, which interactions *suppress* without moving the
death separation; (iii) a Gaussian-covariance proxy for the negativity decay of 8+8
site blocks, shown to be a mixed-state proxy and not used; and (iv) — the number with
no precedent — the smeared local energy density along a worldline, as the **exact
lattice infimum**: the lowest eigenvalue of the smeared-energy operator built as an
MPO by operator-space Heisenberg TEBD and minimized by DMRG (§6.5), cross-checked by
a dense exact-diagonalization second implementation (§6.5.9; agreement ≤ 1.9e-4
wherever the MPO route's operator truncation is converged — measured at L = 6, n_max ≤ 3;
at L = 6 n_max 4, L = 7 and L = 8 only archived-knob rows exist, 0.05–3.5 % off; at the sweep's own knobs and n_max 5–8 the routes agree to 0.3–1.4 % at λ = 0 and disagree by up to 39 % at λ = 4), compared with the free
theory's exact infimum at the same smearing and lattice at the bare mass, at the
Hartree mass, and at the *measured* interacting dispersion (§6.6). **Established:**
the free QEI bound survives the interaction (exact/E_free ≤ 1 at every λ, 0.29 at
λ = 4); the interaction renormalizes the reference by a **mass shift** — the λ = 4
dispersion is a sharp quasiparticle at all 16 momenta and nearest-neighbour to 7e-5,
with the true gap² 3.3 % *below* Hartree, so the sharpest quadratic reference is 4.5 %
deeper than `E_H` and "the residual is the wrong reference" is rejected; and beyond any
quadratic reference there is a residual that is a **first-order quartic tax** — the
exact O(λ) slope of the infimum in the free extremal state is d(1 − E_min/E_H)/dλ =
+0.081, L-independent, 38 % of the extremal energy's slope being the normal-ordered
⟨φ⁴⟩ itself, *positive* because the extremal state anti-squeezes φ at the sampling
point, which the independent decomposition of the extremal state measures directly
(+9–11 % of |E_min| at λ = 4). **Resolved at λ ≤ 1, BRACKETED at λ ≥ 2:** with the
λ = 0 calibration band driven to 0.19 %, the residual is 1.4 / 1.5 / 2.6 % at λ = 0.25
/ 0.5 / 1 (7–13 bands from zero — distance from zero at the sweep's settings, not
convergence in n_max, which is non-monotone at every λ); at λ ≥ 2 the infimum is not monotone in n_max on
either route, the archived operator knobs inflate the apparent residual, and the
(n_max, χ_op, weight) plane gives a **bracket, not a value**: the control-normalized
residual spans −1.8 % to +4.4 % against `E_H` at λ = 2 and −34.3 % to +15.1 % at λ = 4
(+0.3 % to +6.4 % and −28.5 % to +18.8 % against K_eff), a systematic of 32 and 239
λ = 0 bands carried by the *occupation weighting*, so no λ ≥ 2 value is quoted and
every λ = 4 number in this candidate is a number at its knobs. Those brackets are the
**frozen 15-row snapshot** (`unit2_snapshot`: 15 of 36 anchored rows, named by tag so
the quote stays reproducible while the pool lands more). The **sign** survives where the
size does not: against K_eff the residual is positive at **5 of 5** settings of that
snapshot at λ = 2 and at **4 of 5** at λ = 4; **on the live tree as committed on
2026-09-13, 5 of 6 and 4 of 6**, the extra λ = 2 negative failing every gate the harvest
applies (the live verdict sits beside the snapshot and moves as the pool lands rows). Within the τ₀ range the method resolves (0.375–0.75) the interaction shifts the
QEI constant, not its exponent (−1.2 ± 0.8 against a predicted +2). What this
candidate previously claimed and no longer does is listed once, in §6.7. No
anomaly-protocol event was raised.

## 2. Setting, conventions, and where the phase transition is

- `H = ½ Σ_i [π_i² + m² φ_i² + (φ_{i+1} − φ_i)²] + (λ/4!) Σ_i φ_i⁴`, unit lattice
  spacing, ħ = 1, Dirichlet walls `φ_{−1} = φ_L = 0` (so λ = 0 is exactly
  `vacuum.core.harmonic_chain_K(L, m, 'dirichlet')`). The lattice form and the λ/4!
  normalization are those of Milsted, Haegeman & Osborne, PRD **88**, 085030 (2013),
  arXiv:1302.5582, Eq. (3).
- Local basis: the Fock states `|0⟩…|n_max⟩` of an oscillator of frequency
  `ω* = sqrt(⟨π²⟩/⟨φ²⟩)` of the free chain at the middle site — the frequency that
  minimizes the mean local occupation, `n̄_min = ν_loc − ½` (ν_loc the local
  symplectic eigenvalue; `vacuum/interacting/phi4.py` docstring). Every local
  operator is the Rayleigh–Ritz projection `P O P` of the exact operator, so the DMRG
  energy is an upper bound on the true ground energy and is non-increasing in n_max
  (tested). n_max is a reported convergence axis, never a hidden parameter.
- **Phase transition.** With bare m² > 0 the chain is symmetric for every λ ≥ 0: the
  tadpole `δm² = (λ/2)⟨φ²⟩` only raises the mass, and the ℤ₂ transition needs a
  negative bare m². In renormalized units it sits at `λ/μ_R² ≈ 65` in this λ/4!
  normalization (MHO 2013 quote ≈ 66; Kadoh et al., JHEP **05** (2019) 184,
  arXiv:1811.12376, give 10.913(56) in the λ/4 normalization, ×6 = 65.5). The sweep
  reaches `λ/μ_H² ≤ 6` with `μ_H²` the self-consistent Hartree mass
  (`data/s2_sweep.csv`, columns `mu_H2`, `lam_over_mu_H2_over_critical`), i.e. at
  most a tenth of the critical ratio — deep in the symmetric phase by construction.
- Chain: `L = 32`, `m = 0.5` (free correlation length `ξ = 1/arccosh(1 + m²/2) = 2.0`
  sites; local `n̄ = 0.054`), `n_max = 8` (`d = 9`) for everything except the dense
  2+2-site reduced states (`n_max = 6`, `d⁴ = 2401`), `χ = 32` with every number
  re-run at `χ = 64`. QEI: `L = 16`, `n_max = 8`, `χ = 16` (re-run at 32),
  Gaussian sampling `f(t) = exp(−t²/2τ₀²)`, `τ₀ = 0.75`, `ρ = f²` convention of
  `vacuum.inequalities.qei`.

## 3. Anchors (all in `tests/test_interacting.py`; paper-resolution repeat in `data/s1_free_anchor.csv`)

1. **λ = 0 vs the Gaussian core.** Exponential convergence in n_max of the DMRG
   energy, the full covariance `⟨φφ⟩, ⟨ππ⟩`, all block entropies, 1+1 and 2+2 block
   negativities (adjacent and separated) and mutual informations; ≤ 1e-6 reached
   (§4, table).
2. **Perturbation theory.** `E(λ) = E₀ + λE₁ + λ²E₂ + O(λ³)` with `E₁ = ⅛ Σ X_ii²`
   and the two-/four-particle `E₂` derived in `phi4.py`; verified against dense ED on
   two sites (residual/λ³ constant to 1%) and against DMRG at L = 10 (residual ratio
   λ = 0.2 : 0.1 of 7.5, target 8). The covariance matches the O(λ) tadpole chain
   `K + λ diag(⟨φ²⟩/2)` with an O(λ²) residual (ratio 3.5, target 4) that captures
   98% of the shift at λ = 0.1.
3. **χ-doubling** moves nothing above 1e-7 on the test chain; truncation errors are
   in every info record. In the sweep the gate is applied above a resolution floor
   (1e-8 in general; 1e-5 for the Gaussian-proxy negativity, whose float64
   partial-transpose route turns the 1e-9 covariance difference between χ = 32 and 64
   into O(10%) relative changes on numbers within 1e-7 of sudden death — absolute
   changes are recorded for all of them, worst 9.3e-9).
4. **Passivity.** 24 random one- and two-site unitaries (strengths 0.05–1) never lower
   ⟨H⟩ on the λ = 1 ground state; the audit fires (−0.16) on the vacuum after a local
   squeeze when the sampler contains the inverse squeeze.
5. **QEI pipeline.** At λ = 0 the TEBD-evolved smeared energy of a locally squeezed
   MPS equals the exact Gaussian smeared energy `½Tr(O_f(V − V_vac))` to < 1e-5
   (n_max = 8; the residual is the local-basis truncation of the squeezer — 7.6e-5 at
   n_max = 6 — while Trotter/χ/dt/Simpson errors are < 1e-7).

## 4. The λ = 0 anchor at paper resolution (`data/s1_free_anchor.csv`)

| n_max | χ | trunc err | |ΔE| | max|ΔV| | max|ΔS| | |ΔE_N| 1+1 adj | |ΔE_N| 2+2 adj | |ΔE_N| 2+2 sep 1 | |ΔI| 1+1 d=3 | |ΔI| 1+1 d=6 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 32 | 8.6e-16 | 1.1e-04 | 1.0e-04 | 8.2e-05 | 2.8e-04 | 6.3e-04 | 3.0e-05 | 1.4e-05 | 8.7e-07 |
| 6 | 32 | 2.7e-15 | 4.1e-07 | 8.4e-07 | 4.8e-07 | 5.9e-06 | 1.9e-05 | 2.2e-07 | 8.2e-08 | 5.1e-09 |
| 8 | 32 | 4.0e-15 | 1.5e-09 | 9.8e-09 | 2.2e-09 | 1.5e-07 | 5.8e-07 | — | 3.6e-10 | 2.7e-11 |

## 5. Entanglement structure vs λ (`data/s2_sweep.csv`, `s2_mi_profiles.csv`, `s2_proxy_negativity.csv`, `s2_fits.json`)

### 5.1 Energies, Hartree mass, distance from criticality

| λ | E₀ (χ=32) | E₀ − E_PT2 | E₀ − E_PT1 | μ_H² | λ/μ_H² | λ/μ_H² ÷ 65.5 | S_half | trunc err | worst χ-doubling (gated) |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 22.50639797 | +1.47e-09 | +1.47e-09 | 0.2500 | 0.000 | 0.000 | 0.13548 | 4.0e-15 | 5.4e-05 |
| 0.25 | 22.68547123 | +8.25e-04 | -7.15e-03 | 0.3026 | 0.826 | 0.013 | 0.12289 | 8.5e-16 | 3.4e-05 |
| 0.5 | 22.85272176 | +5.78e-03 | -2.61e-02 | 0.3522 | 1.420 | 0.022 | 0.11333 | 2.7e-16 | 2.1e-05 |
| 1.0 | 23.16114864 | +3.75e-02 | -9.02e-02 | 0.4446 | 2.249 | 0.034 | 0.09933 | 4.6e-17 | 7.2e-07 |
| 2.0 | 23.70678022 | — | — | 0.6125 | 3.266 | 0.050 | 0.08148 | 7.0e-18 | 5.4e-07 |
| 4.0 | 24.62752990 | — | — | 0.9087 | 4.402 | 0.067 | 0.06186 | 7.2e-19 | 1.1e-03 |

### 5.2 Mutual-information decay length

`I(d)` between single sites at separation `d = 0…8`, fitted as `ln I = a − d/ξ_MI` on
`d ≥ 2` (`data/s2_fits.json`, residuals listed there). Reference line: half the
lattice correlation length of a free field at the Hartree mass,
`ξ_H/2 = 1/[2 arccosh(1 + μ_H²/2)]` (MI ∝ correlator² asymptotically).

| λ | ξ_MI (χ=32) | ξ_MI (χ=64) | fit max resid (ln) | pts | ξ_H/2 | ξ_MI ÷ (ξ_H/2) | ξ_free/2 |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.8761 | 0.8761 | 6.77e-02 | 7 | 1.0102 | 0.867 | 1.0102 |
| 0.25 | 0.8089 | 0.8089 | 6.55e-02 | 7 | 0.9201 | 0.879 | 1.0102 |
| 0.5 | 0.7594 | 0.7594 | 6.37e-02 | 7 | 0.8546 | 0.889 | 1.0102 |
| 1.0 | 0.6893 | 0.6893 | 6.06e-02 | 7 | 0.7633 | 0.903 | 1.0102 |
| 2.0 | 0.6033 | 0.6033 | 5.61e-02 | 7 | 0.6545 | 0.922 | 1.0102 |
| 4.0 | 0.5118 | 0.5118 | 4.99e-02 | 7 | 0.5432 | 0.942 | 1.0102 |

### 5.3 Exact small-block negativity and its sudden death

Exact partial transpose of the dense reduced state of the two blocks (all
intermediate sites traced; `observables.log_negativity_mps`, cost `O(d^{3(|A|+|B|)})`).
For 1+1 blocks the negativity is nonzero only for adjacent sites; for 2+2 blocks only
at separations 0 and 1 — the free-chain sudden death (Audenaert–Eisert–Plenio–Werner
2002) is reproduced at λ = 0 and **the death separation does not move with λ**; the
live values decrease monotonically with λ.

| λ | E_N 1+1 adj (n_max 8) | E_N 2+2 adj (n_max 6) | E_N 2+2 sep 1 | E_N 2+2 sep 2 | I 2+2 sep 1 |
|---|---|---|---|---|---|
| 0.0 | 2.4645e-01 | 3.1598e-01 | 4.0898e-02 | 1.1e-11 | 5.2559e-02 |
| 0.25 | 2.3776e-01 | 3.0026e-01 | 3.7642e-02 | 3.2e-11 | 4.5929e-02 |
| 0.5 | 2.3046e-01 | 2.8755e-01 | 3.5044e-02 | 1.7e-10 | 4.1082e-02 |
| 1.0 | 2.1850e-01 | 2.6759e-01 | 3.1041e-02 | 1.2e-09 | 3.4280e-02 |
| 2.0 | 2.0068e-01 | 2.3953e-01 | 2.5628e-02 | 5.9e-09 | 2.6178e-02 |
| 4.0 | 1.7687e-01 | 2.0458e-01 | 1.9350e-02 | 3.0e-08 | 1.8096e-02 |

### 5.4 Gaussian-proxy negativity of 8+8 blocks — and why it is only a proxy

The negativity of the Gaussian state with the interacting vacuum's covariance
(`covariance_from_mps` → `vacuum.core.log_negativity`). Exact at λ = 0 (checked
against the core to 1e-7 at all separations); at λ > 0 it is the negativity of a
**mixed** Gaussian state: the covariance of a non-Gaussian pure state has global
symplectic eigenvalues above ½ and a Gaussian entropy `S_G(V) > 0`
(`data/s2c_proxy_mixedness.csv`, table below), and a mixed Gaussian state sheds
long-range negativity fast. The proxy's sudden death collapses from beyond d = 14 at
λ = 0 to d = 2 at λ ≥ 2 while the *free* chain at the same Hartree mass, with the same
blocks, has no death in the window at any λ — so the collapse is the proxy's
mixedness, not the vacuum's entanglement, and the exact 2+2 numbers of §5.3 (death
separation unmoved, live values halved at λ = 4) are the ones to trust. A decay length
is fittable only at λ ≤ 0.5; it is reported for completeness and not used.

| λ | ξ_N proxy (χ=32) | ξ_N proxy (χ=64) | fit max resid (ln) | pts | death d | ξ_H/2 |
|---|---|---|---|---|---|---|
| 0.0 | 0.8517 | 0.8648 | 1.11e+00 | 14 | none in d ≤ 14 | 1.0102 |
| 0.25 | 0.6356 | 0.6356 | 7.16e-01 | 5 | 6 | 0.9201 |
| 0.5 | 0.4592 | 0.4592 | 5.81e-01 | 4 | 5 | 0.8546 |
| 1.0 | — | — | — | 2 | 3 | 0.7633 |
| 2.0 | — | — | — | 1 | 2 | 0.6545 |
| 4.0 | — | — | — | 1 | 2 | 0.5432 |

Proxy mixedness and the like-for-like free references at the Hartree mass
(`data/s2c_proxy_mixedness.csv`; `ξ_MI^free(μ_H)` is the free chain's single-site MI
decay length with the *same* fit window, the cleaner "mass renormalization only"
reference than `ξ_H/2`):

| λ | ν_min | ν_max | S_G(V) [nats] | ξ_MI (interacting) | ξ_MI^free(μ_H), same fit | ratio | proxy death (interacting V) | free death at μ_H (8+8) |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.500000 | 0.500000 | 0.0000 | 0.876 | 0.876 | 1.000 | none in d ≤ 14 | none in d ≤ 14 |
| 0.25 | 0.500004 | 0.500058 | 0.0063 | 0.809 | 0.808 | 1.001 | 6 | none in d ≤ 14 |
| 0.5 | 0.500015 | 0.500165 | 0.0178 | 0.759 | 0.758 | 1.002 | 5 | none in d ≤ 14 |
| 1.0 | 0.500050 | 0.500397 | 0.0440 | 0.689 | 0.686 | 1.005 | 3 | 14 |
| 2.0 | 0.500142 | 0.500796 | 0.0944 | 0.603 | 0.598 | 1.009 | 2 | 12 |
| 4.0 | 0.500347 | 0.501358 | 0.1761 | 0.512 | 0.505 | 1.015 | 2 | 11 |

### 5.5 n_max checks at λ = 4 (`data/s2b_nmax_checks.csv`)

| n_max | χ | trunc err | E₀ | E_N 1+1 adj | E_N 2+2 sep 1 | I 1+1 d=3 | ⟨n⟩ centre |
|---|---|---|---|---|---|---|---|
| 5 | 32 | 9.6e-20 | 24.63351981 | 1.765461e-01 | 1.929008e-02 | 1.264373e-03 | 0.03426 |
| 6 | 32 | 3.0e-19 | 24.62854404 | 1.767377e-01 | 1.935023e-02 | 1.273985e-03 | 0.03441 |
| 7 | 32 | 5.4e-19 | 24.62804189 | 1.768716e-01 | 1.937578e-02 | 1.278054e-03 | 0.03444 |
| 8 | 32 | 7.2e-19 | 24.62752990 | 1.768697e-01 | — | 1.277693e-03 | 0.03447 |

## 6. The QEI number (`data/s3_qei.csv`, `s3_qei_curves.json`, `s3b_qei_nmax.csv`)

### 6.1 Definition

`E_f(ψ) = ∫dt f(t)² [⟨ψ(t)|h_i|ψ(t)⟩ − ⟨Ω|h_i|Ω⟩]`, `|ψ(t)⟩ = e^{−iHt}|ψ⟩`, `h_i` the
PSD local energy density (interaction term included; same bond split as
`vacuum.inequalities.qei`), `|Ω⟩` the interacting ground state (its constant energy
density subtracted — normal ordering with respect to the interacting vacuum). At λ = 0
this is exactly the smeared normal-ordered energy of the Layer-3 stress-tester, whose
infimum over *all* states is one Williamson problem (`qei_minimize`): `E_min^free`.

### 6.2 The variational family and the search

`|ψ(θ)⟩ = Π_bonds S2_b(s_b) Π_j S1_j(r_j) |Ω⟩` on a 5-site window centred on the
worldline (9 real parameters: single-site squeezers `exp(r(b†² − b²)/2)` and two-mode
squeezers `exp(s(b_j†b_{j+1}† − b_j b_{j+1}))`), time-reversal symmetric so only
`t ≥ 0` is evolved (4th-order TEBD, `dt = 0.2`, Simpson on the step grid). The shape
`θ*` is optimized in the *free* theory (cheap Gaussian evaluations) at the Hartree
mass (and, for λ ≥ 1, also at the bare mass); the interacting energy is then evaluated
along the amplitude line `sθ*` on a five-point grid with one parabolic refinement, and
the **minimum of the evaluated energies** is reported — a state that was constructed
and measured, hence a genuine upper bound on the infimum. At λ = 0 the family reaches
0.723 of the exact free infimum (`E_free_family / E_free_exact`): the
Williamson extremal state has a longer squeezing kernel than a 5-site window holds.

### 6.3 Results

| λ | E_var (upper bound) | shape / s* | E_free_exact | E_free_family | E_hartree_exact | E_var/E_free_exact | E_var/E_hartree_exact | E_var/E_free_family | E_var/bound_2d | χ-doubling | dt/2 | TEBD trunc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | -1.6366e-02 | hartree / 0.995 | -2.2654e-02 | -1.6382e-02 | -2.2654e-02 | 0.7224 | 0.7224 | 0.9990 | 0.2611 | 3.0e-06 | 3.3e-06 | 8.8e-04 |
| 0.25 | -1.4787e-02 | hartree / 0.985 | -2.2654e-02 | -1.6382e-02 | -2.0405e-02 | 0.6527 | 0.7247 | 0.9027 | 0.2359 | 2.0e-06 | — | 5.7e-04 |
| 0.5 | -1.3472e-02 | hartree / 0.978 | -2.2654e-02 | -1.6382e-02 | -1.8578e-02 | 0.5947 | 0.7252 | 0.8224 | 0.2149 | 1.6e-06 | — | 4.7e-04 |
| 1.0 | -1.1377e-02 | hartree / 0.967 | -2.2654e-02 | -1.6382e-02 | -1.5731e-02 | 0.5022 | 0.7233 | 0.6945 | 0.1815 | 1.3e-06 | 3.6e-06 | 3.6e-04 |
| 2.0 | -8.4776e-03 | hartree / 0.951 | -2.2654e-02 | -1.6382e-02 | -1.1873e-02 | 0.3742 | 0.7140 | 0.5175 | 0.1352 | 9.7e-07 | — | 2.0e-04 |
| 4.0 | -5.1819e-03 | hartree / 0.929 | -2.2654e-02 | -1.6382e-02 | -7.5215e-03 | 0.2287 | 0.6889 | 0.3163 | 0.0827 | 6.0e-07 | 1.1e-05 | 6.5e-05 |

n_max re-evaluation of the reported λ = 1 state (`data/s3b_qei_nmax.csv`):

| n_max | E_f | TEBD trunc | χ |
|---|---|---|---|
| 6 | -1.134493e-02 | 4.3e-05 | 16 |
| 8 | -1.137747e-02 | 2.5e-05 | 16 |
| 10 | -1.138377e-02 | 1.0e-05 | 16 |

Reading: `E_var/E_free_exact < 1` at every λ — no state in the family beats the free
lattice infimum, so the free QEI bound is not violated by anything we can construct.
`E_var/E_hartree_exact` is flat at the λ = 0 family quality: once the mass
renormalization is accounted for, the interacting vacuum admits **no additional**
negative smeared energy within this family. Comparison with the Flanagan continuum
bound `−(1/6π)∫f′²` (`bound_2d_flanagan`) is dominated by the lattice/mass ratio
already present at λ = 0 (`ratio_var_over_bound2d`).

### 6.4 Convergence of the QEI number

χ-doubling (`rel_change_2chi`, ≤ 3e-6), dt-halving (`rel_change_half_dt`, ≤ 1.1e-5)
and n_max re-evaluation of the reported state at λ = 1 (`data/s3b_qei_nmax.csv`,
8 → 10 moves it by 5.5e-4 relative) are all in the tables above; all sit below the 1%
gate. The "TEBD trunc" column is TeNPy's accumulated discarded Schmidt weight summed
over every bond update of the trajectory (~1500 updates at χ = 16; up to 8.8e-4 for
the most strongly squeezed states) — a bookkeeping upper bound on the state error,
not the observable error, which the χ = 32 re-evaluation measures directly.

## 6.5 The exact lattice infimum (`../notebook_qei_exact.py`, `data/s5*`)

### 6.5.1 Why the variational number was not enough

§6.3 reports `E_var`, the smeared energy minimized over a 9-parameter family of
locally squeezed MPS: a state that was built and measured, hence a rigorous
**upper bound** on the infimum. Its weakness is stated in §7: the gap to the
true infimum was unknown, and at λ = 0 the same family reaches only 0.72 of the
free exact infimum. Both of the paper's QEI ratios were therefore ratios of an
upper bound to an exact number, and the sentence "the free bound survives and
the whole reduction is mass renormalization" rested on the assumption that the
family's 28 % shortfall is λ-independent.

### 6.5.2 The infimum is an eigenvalue

The time-smeared energy is a Hermitian operator on the chain,

  `O_f = ∫ dt f(t)² h_x(t)`,  `h_x(t) = U(t)† h_x U(t)`,  `U(t) = e^{−iHt}`,

so its infimum over **all** states is simply its lowest eigenvalue, and the
minimizing state is the corresponding eigenvector. Nothing is variational about
it. In the free theory `O_f` is a quadratic form and the eigenvalue problem is
the one Williamson problem of `vacuum.inequalities.qei`; at λ > 0 `h_x(t)` is
not local and no closed form exists, but for the short smearing times used
(τ₀ = 0.75, t ≤ 4τ₀ = 3 lattice units) it is representable as an MPO:

1. **Operator-space (Heisenberg) TEBD.** Vectorize `h_x` on the doubled local
   space `C^d ⊗ C^d`; the Heisenberg step `h → u† h u` for a two-site Trotter
   gate `u` is `u* ⊗ u` acting on that doubled space, so the ordinary two-site
   TEBD machinery — the same `model.H_bond` and the same Suzuki–Trotter
   sequence as the Schrödinger route of §6.3, applied in reverse order —
   evolves the *operator*. Gates outside the light cone `|j − x| ≤ ⌈t⌉ + 2` are
   skipped: there `u† 1 u = 1` exactly.
2. **Quadrature.** `h(−t) = h(t)ᵀ` for the real symmetric `H`, so
   `O_f = Q + Qᵀ` with `Q = ∫₀^T f² h(t) dt`, accumulated by the trapezoid rule
   on the TEBD grid (spectrally accurate for a Gaussian weight; the aliasing of
   the ~2ω_max content of `h(t)` against 2π/dt is suppressed by `e^{−ν²τ₀²/4}`).
   `Q + Qᵀ = 2 Re Q` is real symmetric and is converted to a TeNPy MPO.
3. **DMRG on `O_f`** as if it were the Hamiltonian, warm-started from `|Ω⟩`.
   Its ground energy is `λ_min(O_f)`; the reported infimum is
   `E_min = λ_min(O_f) − ⟨Ω|O_f|Ω⟩` (normal ordering with respect to the
   interacting vacuum, exactly the subtraction of §6.1), and the DMRG ground
   state **is** the extremal state.

`vacuum/interacting/qei_exact.py`; the whole construction is checked against a
dense exact-diagonalization build of the same integral at L = 5 (below).

### 6.5.3 What limits it — two errors of opposite sign

The new number is exact in principle and approximate in practice, in three
stated ways. Two matter, and they pull in opposite directions, which is what
makes the λ = 0 anchor a genuine calibration rather than a consistency check:

| approximation | mechanism | sign of its effect on `E_min` |
|---|---|---|
| Fock cutoff `n_max` | Rayleigh–Ritz restriction of the local basis | **raises** it (a smaller space cannot reach as low) |
| operator MPO truncation `chi_op` | discarded operator weight | **lowers** it (it can manufacture sub-infimum states) |
| Trotter `dt`, trapezoid in `t` | order-4 splitting, Gaussian weight | negligible (measured by dt-halving) |

The MPO truncation is the **new uncertainty of this candidate** and it replaces
the old unknown variational gap. *(Superseded in part by §6.5.9 A.2 and §6.7 items
5–6: the sign column below is right for the operator truncation alone, but the Fock
cutoff of the* smeared *infimum does not have a fixed sign — the truncated chain is a
different dynamical system, not a Rayleigh–Ritz subspace — and at n_max 6 the ≈ 1.3 %
λ = 0 residual is that Fock dynamics, χ_op-independent, not the MPO truncation; the
Trotter error is 0.1–0.2 % at λ = 0 but 1 % at λ = 4.)* It is not a bookkeeping number: at
`chi_op = 8` with the plain Frobenius norm the λ = 0 infimum lands 17 % *below*
the exact free answer — a state that does not exist, produced by discarding
operator weight. That is precisely why the λ = 0 anchor is run at the paper
point and its residual quoted as the systematic band on every λ > 0 number.

**The occupation-weighted truncation norm.** The Frobenius norm treats every
matrix element of `h(t)` alike, but the matrix elements of `π²`, `φ²` and `φ⁴`
between high-occupation states are the largest and the least relevant: the
ground state of `O_f` is a mildly squeezed, low-occupation state (the vacuum
carries `n̄ ≈ 0.05` per site). Weighting the two-site block by `G = w ⊗ w`,
`w_n = decay^{n/2}`, before the SVD makes the truncation minimize the error the
low-energy sector actually sees. The weight is a positive on-site *diagonal
product* operator, so applying and undoing it is exact and local and only the
choice of discarded direction changes. At fixed `chi_op` it cuts the λ = 0
error by about 4×, and it is what makes the study affordable
(`occupation_weights`; the `weight_decay = 1` row of the S5a table is the
unweighted control).

### 6.5.4 The λ = 0 anchor at paper resolution (`data/s5a_free_anchor_exact.csv`)

Free chain, L = 16, m = 0.5, τ₀ = 0.75, n_max = 6. The exact answer is one
Williamson problem: `E_min = -2.2653650134e-02` (`qei_minimize`). The deviation
of the operator-TEBD + DMRG route from it is the **total systematic**, and the
reference row's value is carried as a band on every λ > 0 number below.

| chi_op | weight_decay | E_min | dev | rel dev | trunc_op (weighted) | ‖h(T)‖/‖h(0)‖ | chi_mpo | MPO variance | op s | DMRG s |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.25 | -2.294223e-02 | -2.89e-04 | -1.27e-02 | 1.9e-04 | 0.91295 | 64 | 1.4e-06 | 25 | 132 |
| 16 | 1.0 | -2.337997e-02 | -7.26e-04 | -3.21e-02 | 8.6e-02 | 0.95839 | 64 | 3.3e-06 | 52 | 166 |

Reading: the weighted norm (`weight_decay = 0.25`) is worth a factor ≈ 2.5 in the
residual at fixed chi_op against the unweighted control (`weight_decay = 1`), at the
same cost. *(The rest of the original reading — "the residual is negative, hence the
operator truncation, not the Fock cutoff" — is withdrawn, §6.7 item 6: χ_op 16 → 64
leaves this residual at −1.3 to −1.5 %, and the dense-ED route reproduces a negative
Fock-dynamics deviation of the same size with no MPO; see §6.5.9 A.2.)* Trotter and quadrature are pinned separately, against a dense
exact-diagonalization build of the same integral, by
`tests/test_interacting_qei_exact.py` (MPO ≡ ED operator to < 1e-6 relative,
λ_min to < 1e-7).

### 6.5.5 The exact infimum vs λ, and the measured family gap (`data/s5b_exact_vs_lambda.csv`)

Same point as §6.3 (m = 0.5, τ₀ = 0.75, L = 16), n_max = 6. `E_var` is the §6.2
variational family re-run at the **same** cutoff, so the gap is like-for-like.

| λ | E_exact (infimum) | E_var (family) | family gap | E_var/E_exact | E_free_exact | E_hartree_exact | exact/free | exact/Hartree | var/Hartree | trunc_op (w) | Schröd. recheck | ν−½ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -2.294223e-02 | -1.632701e-02 | +6.615e-03 | 0.7117 | -2.265365e-02 | -2.265365e-02 | 1.0127 | 1.0127 | 0.7207 | 1.9e-04 | 1.5e-02 | +3.4e-06 |
| 1 | -1.510075e-02 | -1.132825e-02 | +3.773e-03 | 0.7502 | -2.265365e-02 | -1.573067e-02 | 0.6666 | 0.9600 | 0.7201 | 1.0e-04 | 1.4e-02 | +7.4e-05 |
| 4 | -6.565878e-03 | -5.103453e-03 | +1.462e-03 | 0.7773 | -2.265365e-02 | -7.521511e-03 | 0.2898 | 0.8729 | 0.6785 | 1.2e-04 | 3.1e-02 | +3.0e-04 |

### 6.5.6 Breadth: a second width and a second mass (`data/s5c_breadth.csv`)

| axis | m | τ₀ | λ | E_exact | E_var | E_var/E_exact | E_free_exact | E_hartree_exact | exact/free | exact/Hartree | trunc_op (w) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tau0 | 0.5 | 0.5 | 0 | -3.396650e-02 | -2.806971e-02 | 0.8264 | -3.392870e-02 | -3.392870e-02 | 1.0011 | 1.0011 | 1.8e-05 |
| tau0 | 0.5 | 0.5 | 4 | -1.801108e-02 | -1.553352e-02 | 0.8624 | -3.392870e-02 | -1.887426e-02 | 0.5309 | 0.9543 | 2.1e-05 |
| mass | 0.8 | 0.75 | 4 | -4.466149e-03 | -2.886987e-03 | 0.6464 | -1.135856e-02 | -4.642449e-03 | 0.3932 | 0.9620 | 6.1e-05 |


### 6.5.8 Is the headline converged?  No — the drift, measured (`data/s5e_convergence.csv`)

§6.5.5 reports `exact/Hartree` at one cutoff (n_max = 6, χ_op = 16). That is not
enough to quote a ratio, because both refinements available to us move the λ > 0
numbers, and move them the same way — upward, toward the free-at-Hartree bound:

| λ | n_max = 4 | 5 | 6 | 7 | (n_max = 6, χ_op 16 → 24) |
|---|---|---|---|---|---|
| 4 | 0.8483 | 0.8727 | 0.8729 | **0.9053** | 0.8729 → 0.8782 (+0.0053) |
| 1 | — | 0.9492 | 0.9600 | — | 0.9600 → 0.9618 (+0.0018) |
| 0 (control, must be 1.0000) | — | 0.9993 | 1.0127 | 1.0088 | 1.0127 → 1.0163 (+0.0036) |

Two things follow, and one deduction that an earlier version of this section drew from
this table is withdrawn.

1. **The ladder cannot be extrapolated, and a plateau is not convergence.** The 5 → 6
   step moves the ratio by 2e-4 and the 6 → 7 step by 3.2e-2. §6.5.9 extends the λ = 4
   ladder to n_max = 12 at these knobs: 0.873, 0.905, 0.954, **0.999**, 0.986, 0.976,
   0.993 — it climbs and recedes, while the operator's unweighted norm loss grows from
   15 % to 28 % and the MPO's ⟨Ω|O_f|Ω⟩ becomes inconsistent with the exact reference by
   the size of E_min. A ladder in n_max at fixed χ_op walks a diagonal of the
   (n_max, χ_op) plane; neither axis can be read from it.
2. **Part of the drift is common-mode, not physics.** The λ = 0 control must equal
   1.0000; it reads 0.9993, 1.0127, 1.0088 at n_max = 5, 6, 7 and 1.0163 at χ_op = 24.
   Dividing each λ by the control at the same cutoff cancels the common-mode bias to
   first order: ≈ 0.897 at λ = 4 (n_max 7), ≈ 0.948 at λ = 1 (n_max 6). Those are still
   numbers at their knobs (§6.5.9 A.2, A.4), not bounds.

**Withdrawn.** The earlier reading — "the λ = 0 control is non-monotone; the exact
untruncated infimum is monotone by Rayleigh–Ritz; therefore the wobble is the MPO
truncation" — is wrong at its middle step and is retracted (§6.7, items 4–5). Only the
t = 0 density is a projection P h P; the smeared infimum of the truncated chain is
not a Rayleigh–Ritz quantity, and the dense-ED route with no MPO shows the same
sign-alternating λ = 0 wobble (§6.5.9 A.1). What the table above does establish is
narrower and unchanged: at these knobs the λ = 4 ratio is unconverged and the honest
statement is neither a value nor a bound.

### 6.5.7 Verdict (merged with §6.5.9 and §6.6)

Three statements survive, one is new, and one is withdrawn.

1. **The free QEI bound survives the interaction.** `exact/free` ≤ 1 at every λ > 0 and
   falls to 0.29–0.33 at λ = 4 at every cutoff run (n_max 4–12, χ_op 16–64): no state of
   the interacting theory reaches the free lattice infimum.
2. **The family gap is real and λ-dependent.** `E_var/E_exact` = 0.712 → 0.750 → 0.777
   over λ = 0 … 4 (§6.5.5): the archived variational reading of a flat 0.72 was the
   family's own shortfall, and is retracted (§6.7).
3. **Mass renormalization is most of the reduction, and the reference is a mass shift.**
   The measured dispersion (§6.6) is nearest-neighbour to 7e-5 with a sharp
   quasiparticle at every k; the gap² is 3.3 % below Hartree and the sharpest quadratic
   reference is 4.5 % deeper than `E_H`.
4. **New — the residual beyond any quadratic reference is a first-order quartic tax.**
   Exact first-order theory (§6.5.9 A.3) gives dR/dλ = +0.081 at λ = 0 with a positive
   normal-ordered ⟨φ⁴⟩ piece; the extremal-state decomposition (§6.6.4) measures the
   quartic term positive. At λ ≤ 1 the residual is resolved from zero (1.4–2.6 %, band
   0.19 %); at λ ≥ 2 its size is **bracketed, not resolved** — [+0.301 %, +6.385 %]
   against K_eff at λ = 2 and [−28.541 %, +18.800 %] at λ = 4 across the settings run,
   a spread of 32 and 239 λ = 0 bands (`s8_convergence_plane2.json`).
5. **Withdrawn.** "exact/Hartree ≥ 0.905 at λ = 4, residual ≤ ~10 % and consistent
   with zero" — neither a value nor a bound: the ladder is not monotone and the
   archived knobs inflate the residual (§6.5.9 A.1, A.4).

The breadth points (τ₀ = 0.5; m = 0.8; §6.5.6) and the τ₀ sweep (§6.6.5) are at the
archived knobs and are read as in §6.6.5: no μ_H τ₀ trend and no exponent shift is
claimed within the resolved range.

### 6.5.9 Diagnostics A — the reviewer's plan, executed (`section_diagnostics_A.md`, `data/s5f*`–`s5i*`)

The full record is `draft/section_diagnostics_A.md` (tables A.1–A.4 and the verdict
A.5). In one paragraph: (A.1) the n_max ladder *on the infimum* at λ = 4 with the
archived knobs is 0.873, 0.905, 0.954, 0.999, 0.986, 0.976, 0.993 for n_max 6…12 —
not monotone, with the operator's unweighted norm loss growing to 28 % and the
reference inconsistency reaching the size of E_min; the Rayleigh–Ritz premise is false
for the smeared infimum on any route (the dense-ED λ = 0 ladder sign-alternates at L = 4
and crosses the Gaussian value between n_max 3 and 4 at L = 6 with the paper's own
smearing). (A.2) The λ = 0 band decomposed one knob at a time: at n_max 6 the ≈ 1.3 %
is χ_op-independent (the truncated chain's own infimum), plain-Frobenius truncation is
unusable (−3 to −8 %), the randomized-SVD sketch is a knob at the 0.3–1 % level at λ = 0
and 6 % at λ = 4, Trotter is 0.1–0.2 % at λ = 0 and 1 % at λ = 4; the band reaches
**0.16–0.19 %** at n_max 8, χ_op 32–64, weight 0.5 (≈ 14× the archived cost per point)
while the same knobs move λ = 4 by +0.03 to +0.17. (A.3) The 7-point sweep at those
settings: exact/E_H = 1.0019, 0.9883, 0.9871, 0.9763, 0.9752, 0.8988, 0.8834 at
λ = 0, 0.25, 0.5, 1, 2, 3, 4; the control-normalized residual is a staircase (1.4, 1.5,
2.6, 2.7, 10.3, 11.8 %), so the fitted small-λ exponent 0.46 (0.30–0.62 within the
band) has no footing; the exact O(λ) slope of the residual is +0.081 (L-independent),
and the measured λ ≤ 0.5 points lie below it by an amount consistent with a
second-order coefficient ≈ −0.10 or a cutoff floor. (A.4) The dense-ED second
implementation (`vacuum.interacting.qei_ed`) agrees with the MPO route to ≤ 1.9e-4 at
generous operator knobs at every (L, n_max, λ) run with them — L = 6, n_max 2–3, λ = 0 and
4; the generous rows at L = 6 n_max 4–5, L = 7 and L = 8 were not run — and locates the archived knobs'
error: −1.5 to −6.3e-3 at λ = 0, **+2.7 to +3.5e-2 at λ = 4** (the archived truncation
raises the infimum, inflating the apparent residual); the ED small-λ anchor at L = 6
reproduces 0.7 of the exact slope at n_max 4, as the MPO route does at L = 16.
Verdict A.5: the existence and first-order character of the residual is physics; its
measured λ-dependence beyond λ ≈ 1 is truncation.

## 6.6 The sharpest free reference, and what the extremal state is made of (`section_reference.md`, `data/s7*`)

The full record is `draft/section_reference.md` (§6.6.1–6.6.7). In one paragraph:
the interacting dispersion at λ = 4, from TEBD correlators projected on the Dirichlet
standing waves and harmonically inverted (λ = 0 calibration: 1.7e-6 in ω on all 16
modes; DMRG gap cross-check 2.7e-4), is a sharp quasiparticle at every k (weight
≥ 0.9986, γ/gap ≤ 1e-4, resolution-limited widths) and is nearest-neighbour to 7e-5
(`μ_eff² = 0.879`, `J₁ = 0.999`, `|J_{r≥2}| < 1e-5`): "whole-dispersion
renormalization" is a mass shift, and the true gap² is 3.3 % *below* Hartree. The free
infimum at K_eff is 4.5 % deeper than `E_H` (0.9 % at λ = 1; cutoff-free), so the
reviewer's "the reference is wrong" is rejected — the residual against the sharpest
quadratic reference is larger, not smaller. The decomposition of the archived extremal
state into the four pieces of h_x, in two pictures, finds the quartic term **positive**
(+8.7–8.8 % of |E_min| at λ = 1, +9.7–10.7 % at λ = 4): the extremal state
anti-squeezes φ and squeezes π at the sampling point, as the free Williamson state
does, and the quadratic pieces alone reach 0.94–1.01 of `E_free(K_eff)`. The τ₀ sweep
at λ = 4 gives a residual exponent of −1.2 ± 0.8 (K_eff; −1.1 ± 1.1 against `E_H`) over
τ₀ = 0.375–0.75 against a predicted +2; L = 16 → 24 moves E_min by −1.1 %.

**Reconciliation with §6.5.9.** Every §6.6 number is at the archived knobs
(n_max 6, χ_op 16, weight 0.25), which §6.5.9 shows inflate the λ = 4 residual and
are not converged there. Because `E_free(K_eff)/E_H = 1.0452` is cutoff-free, every
λ = 4 row maps by the same factor: `exact/E_free(K_eff) = 0.835` at the archived knobs
becomes 0.913 / 0.956 / 0.950 at n_max 8 / 9 / 12 with the same knobs, 0.901 at
(n_max 6, χ_op 64, weight 0.5) and 0.917 with the exact SVD — so against K_eff the
λ = 4 residual is 0–17 % and unresolved, exactly as against `E_H`; the archived "≥ 0.866"
is not a bound. The decomposition's +10 % quartic fraction is a value at those knobs
(its Heisenberg-picture audit fails by 10.8 %, the same size); its *sign* is what is
carried forward, and it is the sign exact first-order theory gives. The τ₀ exponent's
± 0.8 is the fit's own error and does not include the archived-knob systematic, which
is comparable to the residuals fitted; the statement carried forward is "no evidence
of an exponent shift within τ₀ = 0.375–0.75", and τ₀ ≥ 1.0 (controls 1.085, 1.410) and
the a = ½ point (control 1.173) are not usable at χ_op = 16 — the a = ½ *dispersion*
completed (`s7e_dispersion_fine`: weight ≥ 0.9993, γ/gap ≤ 7.3e-5, max |ω − ω_H|
= 6.5e-3) and is kept as a dispersion record only. Two further caveats from §6.6: the
archived extremal MPS carries O(1) content outside the sampled light cone (flat
directions of `O_f` that the DMRG sweeps leave arbitrary; the eigenvalue and the
pieces at x are blind to it, any "footprint" reading is not), and L is converged at
the 1 % level, not better.

## 6.7 What this candidate no longer claims

Stated once; §1, §6.5.7 and `../MANIFEST.md` defer to this list.

1. **"Mass renormalization accounts for 86–95 % of the reduction, with a residual
   growing with λ"** — an unconverged number presented as converged.
2. **"The whole reduction is mass renormalization"** (the variational study's flat
   0.72) — the family's λ-dependent shortfall; and a first-order residual is required.
3. **The φ⁴ shell mechanism** for the ladder's irregularity — occupation parity is
   broken by the φ_iφ_{i+1} bond; the λ = 0 control wobbles with no φ⁴ term.
4. **"Non-monotone λ = 0 control ⇒ MPO truncation"** — the dense-ED route shows the
   same sign-alternating wobble with no MPO (§6.5.9 A.1, `tests/test_interacting_qei_ed.py`).
5. **The Rayleigh–Ritz monotonicity premise for the smeared infimum** — only h_x at
   t = 0 is a projection; `e^{−iH_trunc t} ≠ P e^{−iHt} P`.
6. **"The λ = 0 band is the MPO truncation"** — at n_max 6 it is χ_op-independent
   Fock dynamics; and Trotter is not negligible at λ = 4 (1 %).
7. **"exact/Hartree ≥ 0.905 at λ = 4; residual ≤ ~10 % and consistent with zero"** —
   neither a value nor a bound; at λ ≤ 1 the residual is resolved and first order, at
   λ ≥ 2 its size is **bracketed and not resolved** against Hartree and against K_eff
   (λ = 4: [−34.3 %, +15.1 %] and [−28.5 %, +18.8 %]).

## 7. Audits, anomaly protocol, and what this is not (`data/s4_audits.csv`, `summary.json`)

- Passivity of every ground state in the sweep under random local unitaries: passed
  at all λ (worst change listed). Energy-density closure `Σ_i⟨h_i⟩ = ⟨H⟩` to 1e-13.
- **Anomaly protocol.** No state fell below the free exact infimum by more than its
  stated error at any λ, on either implementation (`anomaly_events: []` in
  `summary.json` and `s5_summary.json`); the second implementation required by the
  protocol exists (§6.5.9 A.4) and agrees to ≤ 1.9e-4 where the first is converged
  (L = 6, n_max ≤ 3 at generous knobs; elsewhere the comparisons are at the archived knobs, 0.05–3.5 % off, or at the sweep's and plane's knobs, where they agree to 0.3–1.4 % at λ = 0 and disagree by up to 39–42 % at λ = 4, n_max 8 — §6.5.9 A.4 addendum).
  Had an event occurred, the order of suspicion is fixed in `qei_mps.py` / `qei_exact.py`:
  operator truncation (χ_op, weight, SVD sketch), Trotter step, n_max, wall
  reflections, the reference subtraction, then the physics.
- **What this is not.** One mass, one lattice spacing (the a = ½ point failed its
  λ = 0 control at χ_op = 16; its dispersion is a record only), one window size, and a
  τ₀ range of a factor two. The QEI number is an exact lattice infimum whose
  uncertainty is the operator-MPO truncation: 0.19 % at λ = 0 at the sweep's settings,
  ≥ 10 % at λ = 4 at every setting run — the λ = 0 band does not transfer, and the
  λ ≥ 2 residual is quoted only as a **bracket** ([−34.3 %, +15.1 %] against `E_H` at
  λ = 4), never as a value or a bound. The archived extremal MPS has arbitrary content
  outside the light cone. The large-block negativity decay length is a Gaussian
  proxy. The MI decay length is a 7-point fit with lattice corrections. None of the
  λ > 0 numbers has a continuum limit taken.

## 8. Next

In order of what each buys per hour: (1) λ = 0.1 at the sweep's settings — **done
(S8, `data/s8_lam0.1.json`)**: 0.741 % against the exact first-order prediction 0.81 %,
0.35 bands below it, so the floor reading of the small-λ deficit is excluded at 3.3
bands and the first-order slope survives; (2) the (n_max, χ_op, weight, SVD) plane at
λ = 4 with the reference inconsistency and the unweighted norm loss driven to the λ = 0
level (n_max ≥ 10, χ_op ≥ 64, weight 0.5, exact SVD — 5–10 h per λ point on this
machine), or an operator representation that does not spend bond dimension on
high-occupation matrix elements; the dense-ED ladder at L = 6, n_max = 5 (≈ 30 min per λ
point idle) as its control; (3) the same knobs for the decomposition, so the quartic
fraction becomes a number; (4) τ₀ ≥ 1 and the a = ½ point at χ_op ≥ 32–48, which the
scaling exponent and the continuum extrapolation both need — the χ_op 32 rung of **both**
has since been run and **fails**: τ₀ = 1.5 and 1.0 stay off their controls, and the a = ½
χ_op 32 pair (landed 2026-09-13) moves *away* from 1 (control 1.1865 against 1.173 at
χ_op 16) with 20.01 % / 175.32 % reference inconsistency, so χ_op is not the missing knob
and only the a = ½ χ_op 48 pair is still queued; (5) the negativity-moments
route for 4+4 blocks; (6) the Ising side of the ℤ₂ transition, where the
Bostelmann–Cadamuro–Fewster bound exists as a check.
