# The non-commuting multimode bound: what is proved, what is first order, what survived the attack

Status of the candidate's "Multimode" paragraph after this pass (2026-09-12). The conjectured
bound `ln max|eig S_F| / T ≤ ω_max λ*(d_eff)` for pattern modulation `K(t) = K0 + d s(t) P`
was **neither proved nor refuted**: a targeted attack (sum-frequency, difference-frequency,
near-degenerate and multi-harmonic structures at N = 2, 3, free bang-bang timing, CEM +
Nelder-Mead over every parameter; 128 archived polished optima, plus 1566 further runs that
live only in the first pass's scratch — re-tallied in the 2026-09-12 audit, not archived here and
not pinned)
never broke it — the largest archived ratio is 1.000000000000017, **reached** at an effectively
single-mode configuration (a degenerate pair, or the ω_max mode alone) where Theorem B already
attains it, and above 1 only by +1.7e-14, inside the 1e-13 roundoff of ln ρ. What **is** proved is a Grönwall-type
theorem (Theorem M below) with the constant 1/2 in place of λ*(d)/d ≈ 1/π, in a *sharper*
depth–frequency combination than ω_max d_eff; and a first-order theorem (Theorem F) that
settles the sum-frequency question analytically in the small-d limit. All numbers quoted here
are in `data/s3_multimode_verdict.json` and pinned in `tests/test_floquet_multimode.py`.

Notation. `H(t) = p·p/2 + x·K(t)x/2`, `K(t) = K0 + d s(t) P`, `K0 > 0`, `P = Pᵀ`, `s` measurable
with `|s| ≤ 1`. `A := K0^{-1/2} P K0^{-1/2}`, `d_eff := d‖A‖` (operator norm; so
`(1 − d_eff) K0 ≤ K(t) ≤ (1 + d_eff) K0`, and we assume `d_eff < 1`). `C := K0^{-1/4} P K0^{-1/4}
= K0^{1/4} A K0^{1/4}`. `ω_max = ‖K0‖^{1/2}`. `ρ(S_F) = max|eig S_F|` for a T-periodic s.
`λ*(d)` is the single-mode rate optimum of Theorem B (`rate_optimum`), `λ*(d) = d/π (1 + O(d²))`.

## 1. Theorem M (proved): the number-norm Grönwall bound

**Theorem M.** For every solution `(x(t), p(t))` and every `t₁ < t₂`, with the number form
`𝒩 := (p·K0^{-1/2} p + x·K0^{1/2} x)/2`,

    ln 𝒩(t₂)/𝒩(t₁) ≤ d ‖C‖ ∫_{t₁}^{t₂} |s(t)| dt.                                  (M1)

Consequently, for T-periodic s,

    ln ρ(S_F) ≤ (d/2) ‖C‖ ∫₀ᵀ |s| dt ≤ (d/2) ‖C‖ T,                                  (M2)
    ln ρ(S_F)/T ≤ (d/2) ‖K0^{-1/4} P K0^{-1/4}‖ ≤ (d/2) ‖P K0^{-1/2}‖ ≤ ω_max d_eff / 2.   (M3)

*Proof.* Let `Ω := K0^{1/2}` and `a := (K0^{1/4} x + i K0^{-1/4} p)/√2 ∈ ℂᴺ`, so `𝒩 = |a|²`.
Hamilton's equations `x' = p`, `p' = −K(t) x` give

    a' = (K0^{1/4} p − i K0^{-1/4} K0 x − i d s K0^{-1/4} P x)/√2 = −iΩ a − (i d s/2) C (a + ā),

using `x = K0^{-1/4}(a + ā)/√2` in the last term. Then

    d|a|²/dt = 2 Re(ā·a') = 2 Re(−i ā·Ωa) − d s Re(i ā·Ca) − d s Re(i ā·Cā) = d s Im(ā·Cā),

because `ā·Ωa` and `ā·Ca` are real (Ω, C real symmetric). Since `|ā·Cā| = |⟨a, Cā⟩| ≤ ‖C‖|a|²`,
`d ln|a|²/dt ≤ d |s| ‖C‖`, which integrates to (M1). For (M2): `√𝒩 = |a|` is the Euclidean norm
of `a ∈ ℂᴺ ≅ ℝ²ᴺ`, so it is a genuine norm on phase space; (M1) applied to every solution bounds
the induced operator norm of the real map S_F by `exp((d/2)‖C‖∫|s|)` (the exponent is halved
because 𝒩 = |a|²), and an induced operator norm is submultiplicative, so it dominates the spectral
radius `ρ(S_F)` (Gelfand; this covers the complex eigenvalues of the real matrix S_F too). For the chain (M3): with `ρ(XY) = ρ(YX)`,

    ‖K0^{1/4} A K0^{1/4}‖² = ρ(K0^{1/4} A K0^{1/2} A K0^{1/4}) = ρ((A K0^{1/2})²) ≤ ‖A K0^{1/2}‖²
                          = ‖P K0^{-1/2}‖² ≤ ω_max² ‖A‖².                                  ∎

Remarks. (i) The middle member of (M3) is the energy-form Grönwall bound of route (i) of the
task: with `E = (p·p + x·K0x)/2`, `dE/dt = −d s p·Px` and `|p·Px| ≤ ‖P K0^{-1/2}‖ E`. The number
form is sharper because the coupling splits into a beam-splitter part `−(i d s/2) C a`
(𝒩-preserving) and a squeezing part `−(i d s/2) C ā`; only the latter can grow 𝒩. (ii) Nothing is
assumed about periodicity in (M1). (iii) Equality in (M1) would need `Im(ā·Cā) = ‖C‖|a|²` on a
set of positive measure, i.e. the phase-space direction frozen at a maximal-squeezing
configuration. For N = 1 this is impossible (Theorem B gives the exact supremum, and
`λ*(d)/(d/2) ≤ 0.6891`; the Prüfer phase never stalls, dt/dθ ≤ 1/(ω₀(1 − d)) in Eq. (12) of
`bounds.py`); for N ≥ 2 non-attainment is **not proved** here — it is only observed (largest
*archived* ratio to the proved bound: **0.6657**, Section 3 row C; 0.6758 in the unarchived
exploratory set, which is exactly λ*(0.9)/(0.9/2) at a single-mode rate optimum at d = 0.9).

**Corollary M1 (the constants).** For N = 1, `‖C‖ = ω₀` and (M3) reads `ln ρ/T ≤ ω₀ d/2`, while
Theorem B gives the exact supremum `ω₀ λ*(d)`; hence `λ*(d) ≤ d/2` for all `d ∈ (0, 1)` —
a *proved* consequence of Theorem M at N = 1. Numerically (200-point grid on (10⁻⁶, 1 − 10⁻⁶),
`rate_optimum`, pinned in `tests/test_floquet_multimode.py`) `λ*(d)/(d/2)` increases monotonically from `2/π = 0.63662` (d → 0) to `0.68910`
(d → 1), so the inequality is strict on the grid; the endpoints are measured, not proved.
Therefore, where `‖C‖ = ω_max‖A‖` (top eigenvector of A on the ω_max mode), Theorem M is looser
than the conjecture by the factor `(d_eff/2)/λ*(d_eff) ∈ [1.4512, π/2]`; where
`‖C‖/(ω_max‖A‖) < λ*(d_eff)/(d_eff/2)` it is **sharper** than the conjecture. On the archived
8-site battery the proved bound lies below the conjectured one on **every** row (Section 4).

**Corollary M2 (Z-form).** Because (M1) holds on every window, `ln ρ ≤ (d/2)‖C‖ · (duration of
the |s| = 1 part of the period)`: drives that idle (s = 0) for part of the period gain nothing.

## 2. Theorem F (first order): the two-mode sum-frequency rate and the rôle of ω_max

Write `a = e^{−iΩt} b`. Then `b' = −(i d s/2)[e^{iΩt} C e^{−iΩt} b + e^{iΩt} C e^{iΩt} b̄]`; the
second matrix has entries `C_ij e^{i(ω_i+ω_j)t}`. For T-periodic s, averaging over the fast phases
(first order in d, the rotating-wave step — an ASSUMPTION of the standard kind, valid for
`d‖C‖ ≪ ω_min`, verified numerically below) leaves a norm-preserving part and the squeezing
generator

    G_ij = (d/2) C_ij ŝ(ω_i + ω_j),   ŝ(W) := (1/T)∫₀ᵀ s(t) e^{iWt} dt for W ∈ (2π/T)ℤ, else 0,

the convention "else 0" being the secular part (non-resonant terms average away over many
periods). The retained part of the first matrix is `M_ij = C_ij ŝ(ω_i − ω_j)`, which is
**Hermitian** (`s` real ⇒ `ŝ(−W) = conj ŝ(W)`), so it cannot change `|b|`; when no difference
frequency is resonant it vanishes and `b'' = G Ḡ b`, whose growth rate is `σ_max(G)`
(`rwa_growth_rate`). In general `d|b|²/dt = 2 Im(b̄·G b̄) ≤ 2‖G‖|b|²`, so `σ_max(G)` is an upper
bound on the first-order rate whether or not M vanishes. Since
`|ŝ(W)| ≤ 2/π` for every `|s| ≤ 1` and `W ≠ 0` (the square wave attains it):

**Theorem F.** (a) Two modes coupled only off-diagonally, `C = C₁₂(e₁e₂ᵀ + e₂e₁ᵀ)`, driven by the
square wave at the sum frequency `ω₁ + ω₂ = 2π/T`, grow at first order at the rate

    rate = (d/π)|C₁₂| = (d_eff/π) √(ω₁ω₂),                                              (F1)

which is `√(ω₁ω₂)/ω_max × (d/π)/λ*(d_eff) ≤ √(ω₁ω₂)/ω_max ≤ 1` of the conjectured bound (the
middle step uses `λ*(d) ≥ d/π`, i.e. the *measured* lower end 2/π of Corollary M1, not a proof): a pair
with `ω₁, ω₂ < ω_max` cannot beat it at first order, and the frequency governing the
sum-frequency resonance is the **geometric mean**, i.e. exactly the `K0^{-1/4}` weighting of
`‖C‖`. (b) For a single resonant frequency (every resonant pair at the same `W = 2π/T`)
the first-order rate is `≤ (d/π)‖C_res‖ ≤ (d/π)‖C‖ ≤ (d/π) ω_max ‖A‖`, where `C_res` (the entries
with `ω_i + ω_j = W`) is the average of unitary **congruences** `e^{−iθW} U_θ C U_θᵀ`,
`U_θ = diag(e^{iθω_i})`, over one period of θ — each of which has the same operator norm as C
(`U` unitary ⇒ `(UCUᵀ)^H(UCUᵀ) = conj(U) C^H C conj(U)^{-1}`), so `‖C_res‖ ≤ ‖C‖` by convexity.
(This is the squeezing-sector analogue of a pinching; the relevant invariance is congruence, not
conjugation.) (c) With several harmonics the first-order rate is **at most** `(d/2)‖C ∘ Ŝ‖`
(Schur product, `Ŝ_ij = ŝ(ω_i+ω_j)`; equality when no difference frequency is resonant), which
exceeds `(d/π)‖C‖` iff the Schur-multiplier norm of Ŝ exceeds 2/π — an open question we only
tested (Section 3: archived maximum 0.9864 × 2/π over nine frequency sets, exploratory maximum
0.99991 × 2/π; the 48-piece square wave itself sits at 1.0000 × 2/π on that grid).

Numerical check of (F1) (`data/s3_multimode_verdict.json['sum_frequency']`, exact monodromies):
`|exact rate / first-order rate − 1| ≤ 1.2·10⁻³` over `d_eff ≤ 0.05`, `ω₁ ∈ [0.1, 0.99]` (worst at
ω₁ = 0.1, d_eff = 0.05, where d‖C‖/ω_min is largest; `≤ 6·10⁻⁵` at d_eff = 0.02, ω₁ ≥ 0.3), the
exact ratio to the conjectured bound being `√ω₁ · (d/π)/λ*(d)` to the same accuracy — the analytic
answer to the task's question: **a mode pair with ω₁, ω₂ ≪ ω_max cannot beat the bound**; its
rate is smaller by the factor √(ω₁ω₂)/ω_max.

## 3. The attack (numerics; `notebook_multimode.py` sections C–E, `multimode_attack`)

Parametrisation, without loss of generality (the ratio is invariant under orthogonal changes of
basis, time scaling and the rescaling `P → cP, d → d/c`): K0 = diag(ω²) in its normal modes with
`ω_max = 1`, `A = K0^{-1/2} P K0^{-1/2}` symmetric with `‖A‖ = 1` so that `d = d_eff`, the control
a piecewise-constant `s_j ∈ [−1, 1]` on M pieces with **free durations** τ_j (bang-bang timing and
the period are optimised, not imposed). Every monodromy is the exact ordered product of static
symplectic exponentials, so each configuration is itself an admissible drive and no
discretisation caveat applies. Objective: `R_conj = ln ρ(S_F) / (T ω_max λ*(d_eff))`; the refined
ratio `R_ref = ln ρ/(T (‖C‖/‖A‖) λ*(d_eff))` and the proved ratio `R_M = ln ρ/(T (d/2)‖C‖)` are
recorded for every configuration.

Structures searched (the ones single-mode theory does not see, as the task lists them):
- **sum-frequency resonance** ω₁ + ω₂ = 2π/T with P purely off-diagonal (Section 2; exact
  monodromy at all d, then the dwell times, ω₁ and A freed under Nelder–Mead);
- **detuned near-degenerate pairs** ω = (1 − δ, 1), δ ∈ [0.002, 0.4], `A = R(φ) diag(1, q) R(φ)ᵀ`
  (large off-diagonal elements in the normal-mode basis at fixed d_eff), started from 1–4 cycles of
  the rate-optimum bang-bang with a period pre-scan and polished over all parameters;
- **triples** ω = (1 − 2δ, 1 − δ, 1) with random A (difference- and sum-frequency combinations
  available to the multi-piece drive);
- **unstructured CEM** over (ω, A, s, τ) at N = 2, 3, M = 4, 8 (16 in the exploratory runs),
  d_eff ∈ {0.05, 0.3, 0.8} ({0.05, 0.2, 0.5, 0.8} exploratory), several seeds, each followed by
  a Nelder–Mead polish (≤ 6000 evaluations);
- the **first-order Schur-multiplier search** (Section 2(c)): ω_i = m_i/2 with integers m_i of one
  parity (so every sum frequency is a harmonic of the 2π-periodic s), `σ_max(C ∘ Ŝ)/‖C‖`
  maximised over (s, C) for N = 2, 3, 4, and the best of each set re-checked with the exact
  monodromy at d_eff = 0.01.

Exploratory campaign (the first pass's scratch scripts, **not archived in this repository and
not pinned by any test**; the 2026-09-12 audit re-tallied the scratch JSONL row by row):
56 sum-frequency polishes, 96 CEM runs (N = 2: 48, N = 3: 48; M = 4, 8, 16; four depths),
1296 detuned-pair polishes (N = 2, δ ≤ 0.4, 1–4 cycles), 54 triples (N = 3), 64 Schur runs —
1566 rows in all, of which the largest conjectured ratio is **1 + 5.4·10⁻¹⁴** (a polished
sum-frequency row; +3.8·10⁻¹⁴ over the detuned set, +3.8·10⁻¹⁵ over the CEM set), i.e.
**no configuration exceeded the conjectured ratio 1 beyond the 10⁻¹³ roundoff of ln ρ**, and every optimiser
that reached 1 did so at an effectively single-mode configuration (a degenerate pair ω₁ = ω₂ = 1,
where P restricted to the pair is a rotated commuting pattern and Theorem A applies, or the
ω_max mode alone with the rate-optimum drive). The refined ratio behaved identically (max 1.0000
at the same configurations). The archived, pinned subset (`data/s3_multimode_verdict.json`,
`tests/test_floquet_multimode.py`; build 2026-09-12T18:00:14Z, HEAD 1ab5089, 299.8 s):

| section | configurations | max ratio to conjectured | max to refined | max to proved |
|---|---|---|---|---|
| C attack (CEM + polish; N = 2, 3; M = 4, 8; d_eff 0.05, 0.3, 0.8; 3 seeds) | 36 | **0.9999999999999953** (N = 2, M = 4, d_eff = 0.8: top eigenvector of A on the ω_max mode alone, T = 3.6283 = the rate-optimum cycle) — smallest of the 36 optima 0.7257 | 0.9999999999999956 | **0.6657** |
| D detuned pairs (δ ∈ {0.002, 0.03, 0.2}, φ ∈ {0.5, 1.2}, 1–2 cycles, d_eff 0.05, 0.3, 0.7) | 36 | **1.000000000000017** (all 36 within [0.99999994, 1.00000000000002]; the optimiser drives ω → (1, 1) and T → the rate-optimum period) | 1.000000000000036 | 0.6579 |
| B sum frequency (square wave, exact) | 56 | 0.99494 (ω₁ = 0.99, d = 0.02) | 0.99996 | 0.63661 (= 2/π at first order) |
| E Schur (first order; N = 2, 3, 4; nine sets) | 9 | 0.9864 × 2/π; exact refined ratio at d_eff = 0.01: 0.98639 | — | — |

Every optimum at 1 is an effectively single-mode configuration; the conjectured and the refined
bound are therefore *attained* (by Theorem B) but never exceeded in this pass.

## 4. The archived battery against the proved bound (`section_archive`)

The 60 random patterns of `summary.json['bound_theorem']['multimode_pattern']` (8-site Dirichlet
chain, m = 0.5, ω_max = 2.0321, d_eff ∈ [0.1, 0.7]) are regenerated from the same RNG stream
(seed 7, the 24 coupling rows drawn first) and reproduce the archived `ln ρ` and `d_eff` **exactly**
(max |Δ| = 0.0 on all 60 rows); the three CEM-attacked rows use their archived `ln ρ` with the
regenerated pattern (archived d_eff matched to 0.0). Result (63 rows, 62 unstable):

| ratio of ln ρ/T to | conjectured ω_max λ*(d_eff) | refined (‖C‖/‖A‖) λ*(d_eff) | **proved (d/2)‖C‖** |
|---|---|---|---|
| largest over the battery | 0.16872 | 0.40089 | **0.26059** |

The proved bound is **below the conjectured one on every row**: `(d/2)‖C‖ / (ω_max λ*(d_eff))`
ranges from 0.492 to 0.870, because on a chain with a generic pattern the top eigenvector of A is
spread over modes of different frequencies and `‖C‖ ≪ ω_max ‖A‖`. So for the battery the theorem
of this pass is not only a proof but a *sharper* statement than the conjecture it replaces.

## 5. What remains unproved, and answers to the task's three questions

Unproved (explicitly):
1. `ln ρ(S_F)/T ≤ ω_max λ*(d_eff)` beyond first order — the original conjecture. Evidence: no
   counterexample in ≈ 1600 polished optima over the structures above; equality reached only at
   effectively single-mode configurations.
2. The refined conjecture `ln ρ(S_F)/T ≤ (‖C‖/‖A‖) λ*(d_eff)` (sharper; same evidence; reduces to
   Theorems A/B for N = 1 and for commuting patterns because λ*(x)/x is increasing).
3. First order beyond a single resonant harmonic: whether the Schur-multiplier norm of
   `Ŝ_ij = ŝ(ω_i + ω_j)` can exceed 2/π for some `|s| ≤ 1` and frequency set (tested to ≤ 2/π,
   Section 3; the congruence-average argument of Theorem F(b) proves it only for one resonant
   harmonic). If it could, the conjecture would fail at small d; the search says it does not, at
   N ≤ 4 and harmonics ≤ 8 (archived sets `m` up to (2, 4, 6, 8): ω_i = m_i/2, sum frequencies
   ≤ 8 × 2π/T).
4. Non-attainment of Theorem M for N ≥ 2 (Remark (iii)); and any multimode analogue of the Prüfer
   clock — the 1-D argument uses that the phase-space direction rotates at a rate ≥ ω₀(1 − d) and
   that the gain per π of phase is bounded; for N ≥ 2 the aggregate phase of ā·Cā can stall (two
   modes with opposite-sign depths and different frequencies), so a per-mode or value-function
   (HJB) argument would be needed. Krein–Gelfand–Lidskii strong-stability theory gives stability
   criteria (∫‖H‖ small), not growth-rate bounds, and was not pursued.

Answers. (a) *Should ω_max be replaced?* At first order the frequency that governs a resonance is
the geometric mean of the modes it couples (Theorem F); the proved bound carries no separate
frequency at all — `‖K0^{-1/4} P K0^{-1/4}‖` is the depth–frequency combination, of which
`ω_max‖K0^{-1/2} P K0^{-1/2}‖` is only an upper bound (attained iff the top eigenvector of A lies
on an ω_max mode). (b) *Is the operator-norm depth the right depth?* It is the right *bracket*
(`(1 ± d_eff) K0`) and it is what λ* must be evaluated at (a pair of degenerate modes at ω_max
saturates ω_max λ*(d_eff)), but as a *rate* it overcounts low-frequency couplings; the K0^{-1/4}
weighting fixes that in Theorem M and in the refined conjecture. (c) *Can a low-frequency pair
beat the bound?* No at first order (factor √(ω₁ω₂)/ω_max ≤ 1, Theorem F, exact to 1.2·10⁻³ at
d_eff ≤ 0.05); numerically no at any d tried (Section 3).

## 6. Audit trail (second pass, 2026-09-12, a different agent)

The first pass left this work uncommitted and unreviewed; a second pass re-derived it and fixed
what it got wrong. What was checked and what changed:

- **Theorem M re-derived line by line** and confirmed: the change of variables, `a' = −iΩa −
  (i d s/2)C(a + ā)`, the reality of `ā·Ωa` and `ā·Ca`, `|ā·Cā| ≤ ‖C‖|a|²`, and the chain (M3)
  through `ρ(XY) = ρ(YX)`. The (M1) → (M2) step is now stated properly (the exponent halves
  because `𝒩 = |a|²`; `ρ ≤ ‖·‖` holds because an inner-product-induced operator norm is
  submultiplicative). Numerically (`tests/test_floquet_multimode.py`), the *operator-norm* form
  holds on 40 random systems (N ≤ 4, d_eff ≤ 0.95) with batch maximum **0.879585** — the bound is
  nearly attained in the 𝒩-norm, so it is not slack, and the same statement with the first-order
  constant 1/π is violated (mutated ratio **1.38165**).
- **Corollary M1's constants re-measured** on a 200-point grid: `λ*(d)/(d/2)` rises monotonically
  from `2/π = 0.63661977` to `0.68909937`; the first pass quoted 0.6890, which the endpoint
  exceeds. Only `λ*(d) ≤ d/2` is proved (Theorem M at N = 1); the endpoints are measured.
- **Theorem F**: the rotating-wave step is an ASSUMPTION (now labelled in the module as well as
  here); `|ŝ(W)| ≤ 2/π` re-derived (`|ŝ| = max_φ (1/T)∫ s cos(Wt+φ) ≤ (1/T)∫|cos| = 2/π`, attained
  by the square wave); the pinching claim of (b) is really an average of unitary **congruences**
  `U C Uᵀ`, not conjugates, and is restated; (c) is an upper bound on the first-order rate, an
  equality only when no difference frequency is resonant (the retained difference-frequency part
  is Hermitian and drops out of `d|b|²/dt`).
- **Numbers re-checked against `data/s3_multimode_verdict.json` value by value.** Four were wrong
  in the first pass's draft and are corrected above: the CEM attack's maximum conjectured ratio is
  **0.9999999999999953**, not 1.000000000000017 (that is the *detuned* section's); the
  sum-frequency rows reach **0.63661** of the proved bound, not 0.676; the largest archived ratio
  to Theorem M is **0.6657**, not 0.676 (0.676 is λ*(0.9)/(0.9/2), reached only in the unarchived
  exploratory set); the detuned interval's upper end is 1.00000000000002, not 1. The exploratory
  campaign was re-tallied from the first pass's scratch (1566 rows, largest excess **+5.4·10⁻¹⁴**,
  not 4·10⁻¹⁴) and is now labelled unarchived and unpinned. The Schur search covers harmonics
  ≤ 8, not ≤ 11.
- **Reproduction.** `notebook_multimode.py` was re-run end to end on this tree (348.3 s, HEAD
  `e834b7b`, same generator sha256 `defa7d3c`): every float in `data/s3_multimode_verdict.json`
  reproduces **bit-exactly** (0 differing fields outside `__build__` and the per-section timers),
  so the archived file is the deterministic output of the committed notebook.
- **Unchanged verdict.** The conjecture `ln ρ(S_F)/T ≤ ω_max λ*(d_eff)` is still neither proved
  nor refuted; Theorem M and Theorem F are what this candidate claims.
