# The 4d Gaussian-sampled QEI constant: exact formulation and a certified bound

*Candidate `papers/qei-2d-sharp-constant`, section S8. Companion code:
`vacuum/inequalities/qei_certified.py`, `papers/qei-2d-sharp-constant/notebook_certified.py`,
`tests/test_qei_certified.py`; data `data/s8_certified_enclosure.json`.*

**Status in one line.** The analytic formulation behind the archived number
`C_4d/C_FE = 0.4574904` is derived here in full and matched to the code
formula-by-formula; the **lower** half of a two-sided enclosure is closed as a
computer-assisted theorem to `2.5e-08`; the **upper** half is *not* closed — the
only rigorous upper bound we have is the Fewster–Eveson theorem `R ≤ 1`.
The enclosure delivered is therefore

>   **0.457490709 ≤ C_4d(Gaussian)/C_FE ≤ 1**

with the archived value `0.45749073394875556` sitting `2.46e-08` above the
certified lower endpoint — a factor `81` inside the momentum route's own stated
grid uncertainty of `2e-06`. Section 7 says exactly what is missing and why.

---

## 0. Notation and the object

Massless real scalar `φ` in 4d Minkowski, `ħ = c = 1`, normal ordering with
respect to the Minkowski vacuum. `f` real, even, smooth, compactly supported or
of Gaussian decay. The **smeared energy along the inertial worldline `x = 0`**
is, in the `f²` convention used by Fewster & Eveson (PRD **58**, 084010 (1998),
their Eq. (40)) and by `vacuum.inequalities.qei`,

```
    E_f  :=  ∫ dt  f(t)²  :T_00:(t, 0).                                    (0.1)
```

The sharp constant for this `f` and the ratio to the Fewster–Eveson constant are

```
    C_sharp(f) := - inf_ψ ⟨ψ| E_f |ψ⟩ / ∫ f''(t)² dt ,
    C_FE := 1/(16 π²),      R(f) := C_sharp(f)/C_FE ,                      (0.2)
```

the infimum running over normalised vectors in the form domain of `E_f`.
Fewster–Eveson Eq. (40) is the theorem `R(f) ≤ 1` for every admissible `f`.
The object certified here is `R` for the Gaussian `f(t) = exp(-t²/(2τ₀²))`,
for which

```
    F̂(ν) := ∫ f(t)² e^{iνt} dt = τ₀ √π e^{-ν²τ₀²/4},
    ∫ f''² dt = 3√π/(4 τ₀³).                                               (0.3)
```

---

## 1. Reduction of `E_f` to a radial quadratic form

### 1.1 Mode expansion

With `[a_k, a_{k'}^†] = (2π)³ δ³(k − k')` and

```
    φ(t, x) = ∫ d³k/(2π)³ (2ω)^{-1/2} [ a_k e^{ik·x − iωt} + h.c. ],   ω = |k|,
```

one has at the spatial origin

```
    φ̇(t,0)   = −i ∫ d³k/(2π)³ √(ω/2)     [ a_k e^{−iωt} − a_k^† e^{iωt} ],
    ∂_j φ(t,0) = i ∫ d³k/(2π)³ k_j/√(2ω) [ a_k e^{−iωt} − a_k^† e^{iωt} ].  (1.1)
```

Wick-ordering the squares and smearing with `f²` (so that every `e^{iνt}`
becomes `F̂(ν)`, and `F̂` is real and even because `f` is real and even) gives,
after relabelling `k ↔ k'` in the two `a^†a` terms,

```
    E_f = ∫∫ d³k d³k'/(2π)^6  {  A(k,k') a_k^† a_{k'}
                               + ½ B(k,k') ( a_k^† a_{k'}^† + a_k a_{k'} ) },
    A(k,k') = ½ √(ωω') (1 + k̂·k̂') F̂(ω − ω'),
    B(k,k') = −½ √(ωω') (1 + k̂·k̂') F̂(ω + ω').                            (1.2)
```

The factor `(1 + k̂·k̂')/2` is `T_00 = ½(φ̇² + (∇φ)²)`: the angular factor of
`:φ̇²:` is `1`, that of `:(∇φ)²:` is `k̂·k̂'`. (This is exactly the split
documented in `worldline_momentum_minimize`'s `operator` argument.)

### 1.2 Partial waves

`1 = 4π Y_{00}(k̂) Y_{00}^*(k̂')` and `k̂·k̂' = (4π/3) Σ_m Y_{1m}(k̂) Y_{1m}^*(k̂')`,
so only `l = 0` and `l = 1` occur. Define radial modes

```
    a_{lm}(ω) := (ω/(2π)^{3/2}) ∫ dΩ  Y_{lm}^*(k̂) a(ω, k̂),                (1.3)
```

which satisfy `[a_{lm}(ω), a_{l'm'}(ω')^†] = δ_{ll'} δ_{mm'} δ(ω − ω')` (check:
substitute `[a_k,a^†_{k'}] = (2π)³ δ(ω−ω') δ²(k̂−k̂')/ω²`). Carrying `d³k/(2π)³ =
ω² dω dΩ/(2π)³` through (1.2) with (1.3) gives an **exact decoupling into four
identical copies of one radial problem**:

```
    E_f = Σ_{l=0,1} Σ_{m}  c_l  H[a_{lm}],     c_0 = 1/(4π²),  c_1 = 1/(12π²),
    H[a] = ∫∫ dω dω' { K(ω,ω') a^†(ω) a(ω')
                       − ½ K_-(ω,ω') ( a^†(ω)a^†(ω') + a(ω)a(ω') ) },      (1.4)
    K(ω,ω')   := (ω ω')^{3/2} F̂(ω − ω'),
    K_-(ω,ω') := (ω ω')^{3/2} F̂(ω + ω').
```

Since `c_0 + 3 c_1 = 1/(4π²) + 1/(4π²) = 1/(2π²)` and the four sectors are
independent copies of the *same* form,

```
    inf_ψ ⟨E_f⟩  =  (1/(2π²)) · E_Q ,
    E_Q := inf { ⟨H⟩ over states of the single radial mode algebra }.       (1.5)
```

The sign of the pair kernel is immaterial: `a → i a` is a Bogoliubov
transformation sending `K_- → −K_-` and fixing `K`, so `E_Q` is the same for
`±K_-`. From here on we use `H = a^†Ka + ½(a^†K_-a^† + h.c.)`, the code's sign.

### 1.3 The ratio, and its independence of `τ₀`

With (0.3) at `τ₀ = 1`, `F̂(ν) = √π e^{-ν²/4}` and

```
    R = (E_Q/(2π²)) / ( −(1/(16π²)) · 3√π/4 ) = −32 E_Q / (3√π).           (1.6)
```

`R` does not depend on `τ₀`: under the dilation `(Uψ)(x) = τ₀^{-1/2}ψ(x/τ₀)` the
kernel `K` at scale `τ₀` is unitarily equivalent to `τ₀^{-3} K` at scale 1, so
`E_Q(τ₀) = τ₀^{-3} E_Q(1)`, while `∫f''²` carries the same `τ₀^{-3}`.
**Implemented as** `qei_certified.ratio_from_energy`; pinned by
`tests/test_qei_certified.py::test_ratio_from_energy_matches_the_archived_conversion`
against the archived `(e_Q, ratio)` pair of `data/s6b_momentum_4d_worldline.json`.

---

## 2. Identification with the code, line by line

Read against `vacuum/inequalities/qei.py` at the commit recorded in the MANIFEST.

| derivation | code | verdict |
|---|---|---|
| `F̂(ν) = ∫f² e^{iνt}dt` | `fhat_of(f)`; for `gaussian_f(τ₀,t₀)` the analytic `f2hat` returns `τ₀√π e^{−ν²τ₀²/4} e^{iνt₀}` | identical; **the code smears with `f²`, not `f`** |
| `∫f''²` | `fsecond_sq_integral(f)` → `gaussian_f.fpp2 = 3√π/(4τ₀³)` | identical |
| `K, K_-` on a quadrature grid | `_worldline_kernels(om, wom, fhat, 3)`: `s = √w · ω^{3/2}`, `A_ij = s_i s_j F̂(ω_i−ω_j)`, `B_ij = s_i s_j F̂(ω_i+ω_j)` | `A_ij = √(w_i w_j) K(ω_i,ω_j)` — the symmetric Nyström discretisation of `K`, likewise `B` of `K_-` |
| grid | `momentum_grid(ω_max, n, power=2)` — Gauss–Legendre in `u = ω^{1/2}`, Jacobian folded into the weights | quadrature choice only; irrelevant to the present certificate, which uses no grid |
| `H = a^†Ka + ½(a^†K_-a^† + h.c.)` | `_bogoliubov_ground_energy(A,B)` returns the ground energy of `a^†Aa + ½(aᵀ conj(B) a + h.c.)`; `B` real here | identical |
| `c_0 = 1/(4π²)`, `c_1 = 1/(12π²)`, `Σ = 1/(2π²)` | `e0 = eQ/(4π²)`; `sectors = ((0,1,e0),(1,3,e0/3))`; `e_min = e0 + 3(e0/3) = eQ/(2π²)` | identical; the archived row `n=400` has `sector_l1_per_copy = sector_l0/3` to all printed digits |
| `bound = −C_FE ∫f''²`, `R = e_min/bound` | `bound = -FEWSTER_EVESON_CONSTANT_4D * fsecond_sq_integral(f)`, `ratio = e_min/bound` | identical |
| sign of the pair kernel | derivation gives `−K_-`, code uses `+K_-` | **immaterial** (§1.2); recorded, not a defect |

Two code-side details that are approximations *in the code* and are absent from
the certificate, recorded here rather than silently:

* **`mode_floor`.** `_bogoliubov_ground_energy` discards `K`-eigenvectors below
  `1e-14·λ_max` and replaces their contribution by the second-order estimate
  `e_dropped = −½ Σ |B̃|²/(α+α')`. On the archived grids `e_dropped` is
  `−8.1e-12` (`n=400`) to `−2.4e-11` (`n=500`), i.e. `≈1e-10` of `e_Q`. It is a
  controlled approximation, not an error, but it is *not* a two-sided bound.
* **Nyström discretisation.** The archived ladder converges in `(n, ω_max)` to
  `0.45749073` but carries no rigorous quadrature remainder. The present
  certificate uses no quadrature in `ω` at all.

**Independent check of the formulation.** `qei_certified` never calls `qei`'s
kernels; it builds `⟨φ_m, Kφ_n⟩` from a closed-form Γ-series (§4). The two
routes are compared directly in `notebook_certified.s8a_element_crosscheck` and
in `tests/test_qei_certified.py::test_elements_match_the_qei_module_kernel`:
a 4000-node 2D Gauss–Legendre quadrature of the kernel built by
`qei.fhat_of(qei.gaussian_f(1.0))` reproduces the series elements to
**8.7e-13** (relative, worst of 36 entries, both kernels) — the quadrature, not
the series, is the loose side. **No discrepancy between derivation and code was
found.**

---

## 3. Well-posedness

Write `g(ω) = ω^{3/2} e^{−ω²/4}` and use `F̂(ω∓ω') = √π e^{−ω²/4}e^{−ω'²/4}e^{±ωω'/2}`
at `τ₀ = 1`. Expanding `e^{±ωω'/2}` in powers,

```
    K   = √π Σ_{n≥0} (2^n n!)^{-1} |φ_n⟩⟨φ_n| ,
    K_- = √π Σ_{n≥0} (−1)^n (2^n n!)^{-1} |φ_n⟩⟨φ_n| ,
    φ_n(ω) := ω^{n+3/2} e^{−ω²/4} ∈ L²(0,∞).                               (3.1)
```

**(a) `K ⪰ 0`, unbounded, with `{φ_n}` in its form domain.** Positivity is
(3.1). `K = √π M_{ω^{3/2}} 𝒞 M_{ω^{3/2}}` with `𝒞` the compression to `(0,∞)` of
convolution by `e^{−(ω−ω')²/4}`; `𝒞 ⪰ 0` and bounded, so `K` grows like `2πω³`
and is unbounded. Each `⟨φ_m, Kφ_n⟩` is a convergent Gaussian integral (the
quadratic form `½(ω²+ω'²) − ½ωω'` is positive definite), so `span{φ_n}` lies in
the form domain.

**(b) `K_-` is trace class, with an explicit bound.** Shifting the Fourier
contour by `−iκ` (legitimate: the integrand is entire with Gaussian decay),

```
    e^{−(ω+ω')²/4} = π^{-1/2} ∫_ℝ e^{−(ξ+iκ)²} e^{i(ξ+iκ)(ω+ω')} dξ,
    ⇒ K_- = ∫_ℝ e^{−(ξ+iκ)²} |h_ξ⟩⟨h̄_ξ| dξ,  h_ξ(ω) = ω^{3/2} e^{iξω−κω}.
```

Each summand is rank one with trace norm `‖h_ξ‖² = 3/(8κ⁴)`, and
`|e^{−(ξ+iκ)²}| = e^{κ²−ξ²}`, so

```
    ‖K_-‖₁ ≤ √π e^{κ²} · 3/(8κ⁴),   minimised at κ = √2:
    ‖K_-‖₁ ≤ 3 √π e² / 32 = 1.22766…                                       (3.2)
```

(The numerically computed value on the `n = 400` Nyström matrix is `1.2114`,
consistent with (3.2).) In particular `K_-` is Hilbert–Schmidt.

**(c) `K ± K_- ⪰ 0`.** From (3.1), `K + K_- = 2√π Σ_{n even} (2^n n!)^{-1}
|φ_n⟩⟨φ_n| ⪰ 0` and `K − K_- = 2√π Σ_{n odd} … ⪰ 0`. Hence the quadratic form
is non-negative in each quadrature separately and `E_Q ≤ 0`.

**(d) The form is bounded below.** `E_Q ≥ −3√π/32 = −0.16617` — this is
Fewster–Eveson Eq. (40) read through (1.5)–(1.6). (An independent, weaker route:
for every finite-rank compression the Powers–Størmer inequality
`‖X^{1/2}−Y^{1/2}‖²_HS ≤ ‖X−Y‖₁` plus `|Tr M| ≤ ‖M‖₁` gives ground energy
`≥ −½‖K_-‖₁ ≥ −0.614` by (3.2). We record it as a check on the machinery, not
as the operative bound.)

**(e) Diagonalisability (cited, not used).** With `K > 0`, `K_-` Hilbert–Schmidt
and `K ± K_- ⪰ 0`, `H` is the standard bosonic quadratic Hamiltonian of
Nam, Napiórkowski & Solovej, *J. Funct. Anal.* **270** (2016) 4340–4368
(Theorems 1–2); their hypotheses (a)–(c) are (a)–(c) above. Their conclusion —
that `inf spec H` is attained on a quasi-free (Gaussian) state and equals
`½ Tr[ ((K−K_-)^{1/2}(K+K_-)(K−K_-)^{1/2})^{1/2} − K ]` — is what the code's
Riccati/Takagi route computes. **Nothing in §4–§6 depends on it**: the
certificate only uses that *some* explicitly exhibited Gaussian state has the
energy we compute.

---

## 4. Exact matrix elements in the `φ` basis, with a proved tail

Take the (non-orthogonal, linearly independent, total) family
`φ_n(ω) = ω^{n+3/2} e^{−ω²/4}`, `n = 0,1,2,…`. With `I_j := ∫_0^∞ ω^{j+3}e^{−ω²/2}dω
= 2^{(j+2)/2} Γ((j+4)/2)` (substitute `u = ω²/2`):

```
    G_{mn} := ⟨φ_m, φ_n⟩ = 2^{(m+n)/2 + 1} Γ((m+n+4)/2),                    (4.1)

    𝔄_{mn} := ⟨φ_m, K φ_n⟩   = √π · 2^{(m+n)/2+2} Σ_{k≥0}      t_k^{mn},
    𝔅_{mn} := ⟨φ_m, K_- φ_n⟩ = √π · 2^{(m+n)/2+2} Σ_{k≥0} (−1)^k t_k^{mn},  (4.2)
    t_k^{mn} := Γ((m+k+4)/2) Γ((n+k+4)/2) / k! ,
```

obtained by expanding `e^{±ωω'/2}` and integrating term by term (all terms
positive for `𝔄`; absolute convergence, established next, licenses the
interchange for `𝔅`).

**Tail lemma.** By Wendel's inequality (`Γ(x+s) ≤ x^s Γ(x)`, `0 ≤ s ≤ 1`,
`x > 0`; J. G. Wendel, *Amer. Math. Monthly* **55** (1948) 563),

```
    t_{k+1}/t_k = Γ((m+k+5)/2)Γ((n+k+5)/2) / (Γ((m+k+4)/2)Γ((n+k+4)/2)(k+1))
                ≤ √((m+k+4)(n+k+4)) / (2(k+1))
                ≤ (N+k+3) / (2(k+1))   for m, n ≤ N−1,
```

and `(N+k+3)/(2(k+1))` is decreasing in `k`. Hence for any `K` with
`q := (N+K+3)/(2(K+1)) < 1` (i.e. `K > N+1`),

```
    Σ_{k>K} t_k  ≤  t_K · q/(1−q).                                          (4.3)
```

This is the **only** truncation in the whole certificate, and it is two-sided:
the partial sum under-estimates `𝔄_{mn}` by at most (4.3) and brackets
`𝔅_{mn}` by at most (4.3) in absolute value. `qei_certified._series_terms_needed`
chooses `K` so that (4.3) is below `10^{−(dps−8)}` of the leading term;
`phi_kernel_elements` returns `iv.mpf` intervals whose half-width is
`(4.3) + 4(K+3)·2^{−prec}·Σ_k t_k + 64 ulp`, the middle term bounding the
accumulated mpmath rounding of a `K`-term sum (mpmath's `mpf` arithmetic is
correctly rounded at the working precision) and the last a conservative pad for
the final few operations. Measured relative half-widths (`max_element_rel_width`,
`element_rel_width_log10` in the data file): `10^{−210.5}` at
`n_basis = 16, dps = 213` and `10^{−287.3}` at `n_basis = 24, dps = 290`, i.e.
`≈ 10^{−(dps−2)}`; for `n_basis ≥ 32` the value underflows double precision and
the `float` fields read `0.0` / `−inf`.

---

## 5. The certificate

### 5.1 Two lemmas

**Lemma 1 (energy of an arbitrary state).** Let `{e_i}` be an orthonormal set in
`L²(0,∞)` and `a_i` the corresponding modes. For any state `ρ` in the form
domain with `N_{ij} = Tr(ρ a_i^† a_j)`, `M_{ij} = Tr(ρ a_i a_j)`,

```
    ⟨H⟩_ρ = Σ_{ij} 𝔎_{ij} N_{ij} + Re Σ_{ij} 𝔎^-_{ij} \bar M_{ij},
    𝔎_{ij} = ⟨e_i, K e_j⟩,  𝔎^-_{ij} = ⟨e_i, K_- e_j⟩.                      (5.1)
```

This is the definition of `H` plus normal ordering; nothing else.

**Lemma 2 (which `(N, M)` are states).** For finitely many modes and real
symmetric `N`, `M`, there exists a Gaussian state with those two-point functions
**iff**

```
    Γ(N,M) := [[ N , M ], [ M , I + N ]]  ⪰  0.                             (5.2)
```

*Necessity*: `⟨c^†c⟩ ≥ 0` for `c = Σ u_i a_i + Σ v_i a_i^†` is exactly
`u·Nu + 2u·Mv + v·(I+N)v ≥ 0`. *Sufficiency*: conjugating (5.2) by the rotation
`e_± = (1,±1)/√2` gives `[[σ_x, −I/2],[−I/2, σ_p]] ⪰ 0` with
`σ_x = N+M+I/2`, `σ_p = N−M+I/2`, which is `σ_x ⪰ 0` and
`σ_x^{1/2} σ_p σ_x^{1/2} ⪰ I/4`, i.e. every symplectic eigenvalue of
`diag(σ_x, σ_p)` is `≥ 1/2` — Williamson's theorem then produces the Gaussian
state (Simon, Sudarshan & Mukunda, PRA **36** (1987) 3868, §II; Holevo,
*Probabilistic and Statistical Aspects of Quantum Theory*, Ch. V). Note
`Γ(N,M) ⪰ 0 ⟹ Γ(N+δI, M) ⪰ δ I`, which is how the certificate gets a strictly
positive margin.

### 5.2 The exhibited state and the theorem

Fix `N ∈ ℕ`, let `V_N = span{φ_0, …, φ_{N−1}}`, let `G = L Lᵀ` be the Cholesky
factorisation of (4.1) and `ê := L^{−1} φ` the resulting orthonormal basis of
`V_N`. Put `Â = L^{−1} 𝔄 L^{−ᵀ}`, `B̂ = L^{−1} 𝔅 L^{−ᵀ}` (the compressions of
`K`, `K_-` to `V_N` in that basis). Choose real symmetric `(N₀, M₀)` obeying
(5.2), and let `ρ⋆` be the Gaussian state of Lemma 2 on the `V_N` modes tensored
with the **vacuum** on `V_N^⊥`; use the *same* `ρ⋆` in the `l = 0` sector and in
each of the three `l = 1` sectors. Cross terms between `V_N` and `V_N^⊥` drop
out of (5.1) because the vacuum factor has vanishing two-point functions, so

> **Theorem (certified lower bound).** For every `(N₀, M₀)` satisfying (5.2),
>
> ```
>     E_Q ≤ Tr(Â N₀) + Tr(B̂ M₀)  =:  e_ub ,
>     R = C_4d(Gaussian)/C_FE ≥ −32 e_ub / (3√π).                           (5.3)
> ```

There is **no discretisation error**: `V_N` is a genuine subspace of
`L²(0,∞)`, `ρ⋆` is a genuine state of the field, and (5.3) is the exact energy
of an exact state. Every inequality used is an inequality, not an estimate.

### 5.3 What the computer does, and the error budget

`qei_certified.certified_lower_bound(n_basis=N, dps=D)`:

| step | arithmetic | rigorous? | measured size (N = 64, D = 674) |
|---|---|---|---|
| `G` from (4.1) | mpmath `mpf`, padded to an interval | yes (closed form + 64-ulp pad) | rel. half-width `< 1e-660` |
| `𝔄, 𝔅` from (4.2) | `mpf` partial sums + tail bound (4.3) + rounding bound | yes | rel. half-width `< 1e-600` |
| `G = L Lᵀ` | **interval** Cholesky, outward-rounded | yes (also *proves* `G ≻ 0`) | `cond(G) ≈ 1e60` |
| `L^{−1}` | **interval** forward substitution | yes | — |
| `Â, B̂` | **interval** congruence | yes | — |
| `(N₀, M₀)` | `mpf` eigen-solve of the `N`-mode Bogoliubov problem on the interval midpoints, then `N₀ ← N₀ + δI`, `δ = 10^{−D/3}` | **not needed to be** — it is only a *choice of trial state* | `δ = 1e-224` |
| `Γ(N₀,M₀) ⪰ 0` | **interval** Cholesky of the `2N × 2N` block (5.2) | yes — raises rather than asserts | all `2N` pivots `> 0` |
| `e_ub = Tr(ÂN₀)+Tr(B̂M₀)` | **interval** | yes | relative width `10^{−385.6}` (`energy_width_log10`; the two `energy_interval` endpoints agree to all 17 printed digits) |
| `R ≥ −32 e_ub/(3√π)` | **interval** | yes | — |

The regularisation `δ` needs no correction term: the energy that is evaluated is
the energy of the regularised state. `kernel_scale` and `b_scale` multiply the
*interval* kernels after the trial state has been built from the unmutated ones,
so a mutated run is an honest bound for the mutated problem.

---

## 6. The numbers

`data/s8_certified_enclosure.json`, produced by
`papers/qei-2d-sharp-constant/notebook_certified.py` (2 threads; the seven rungs
cost `602 s` in total, the whole notebook `≈ 13 min` cold, `81 s` when it resumes
an existing ladder).
Every row is a **rigorous lower bound** on `R`; `gap` is to the archived
momentum-space value `0.45749073394875556` (`s6b_momentum_4d_worldline.json`,
`n = 800`, `ω_max = 24`), which the MANIFEST quotes as `0.4574904 ± 2e-06`.

| `n_basis` | `dps` | series terms `K` | certified `R ≥` | gap to archived | `cond(G)` | wall |
|---|---|---|---|---|---|---|
| 16 | 213 | 899 | `0.457338470621` | `1.52e-04` | `1e6` | `2 s` |
| 24 | 290 | 1468 | `0.457473083063` | `1.77e-05` | `1e13` | `9 s` |
| 32 | 367 | 1761 | `0.457487527623` | `3.21e-06` | `1e21` | `20 s` |
| 40 | 444 | 2026 | `0.457489974677` | `7.59e-07` | `1e30` | `46 s` |
| 48 | 520 | 2286 | `0.457490519055` | `2.15e-07` | `1e39` | `96 s` |
| 56 | 597 | 2567 | `0.457490664733` | `6.92e-08` | `1e49` | `186 s` |
| **64** | **674** | **2847** | **`0.457490709365`** | **`2.46e-08`** | `1e60` | `259 s` |

The gap falls by a factor `≈ 0.27` per `+8` basis functions (observed, not
proved — §7.4). The interval half-widths never enter: at `n_basis = 64` the
energy interval is narrower than `1e-100` of the value, so the whole gap is
Galerkin deficiency, i.e. the bound is *true but not yet saturated*: the energy
interval's own relative width is `10^{−157.7}` at `n_basis = 16` and
`10^{−385.6}` at `n_basis = 64` (`energy_width_log10`).

**Formulation check** (`s8a`): the Γ-series elements against a 4000-node 2D
Gauss–Legendre quadrature of `qei.fhat_of(qei.gaussian_f(1.0))` — worst relative
disagreement `8.36e-13` (`K`) and `8.72e-13` (`K_-`) over 36 entries.

**Mutations** (`s8c`, at `n_basis = 40`), the falsification handles:

| mutation | certified `R ≥` | excludes `0.45749073`? |
|---|---|---|
| none | `0.457489974677` | no (correct: the bound must sit below) |
| both kernels `× (1+1e-4)` | `0.457535723674` | **yes** |
| `K_- × (1+1e-4)` | `0.457628840433` | **yes** |
| `K_- × (1−1e-4)` | `0.457351108920` | no, and it moves *down* (signed) |

The `K_- × (1+1e-4)` row is the regression test
`tests/test_qei_certified.py::test_mutated_kernel_enclosure_excludes_the_archived_value`
(run there at `n_basis = 20`, where the same mutation gives `0.457575965256`,
`8.5e-05` above the archived value). A second mutation test feeds `_chol_iv` a
`(N, M)` pair that violates (5.2) and requires it to raise.

**Reproducing.**

```
.venv/bin/python papers/qei-2d-sharp-constant/notebook_certified.py                 # ~9 min, n_basis <= 48
.venv/bin/python papers/qei-2d-sharp-constant/notebook_certified.py --ladder 16,24,32,40,48,56,64
.venv/bin/python papers/qei-2d-sharp-constant/notebook_certified.py --quick         # ~10 s, writes outside data/
.venv/bin/python -m pytest tests/test_qei_certified.py -q -p no:cacheprovider       # 16 s
QEI_CERT_SLOW=1 .venv/bin/python -m pytest tests/test_qei_certified.py -q -p no:cacheprovider
```

---

## 7. What is **not** proved

This section is the deliverable's other half. Read it before quoting anything.

### 7.1 The upper endpoint of the enclosure is not ours

`R ≤ 1` is Fewster & Eveson's theorem, not a computation of this pass. The
enclosure width `1 − 0.4574906 = 0.542` is therefore entirely the *missing*
rigorous **lower** bound on `E_Q`. Nothing here shows that `R < 1`, let alone
that `R ≈ 0.4575`.

### 7.2 Why the missing half is hard here: the problem is critical

The standard route to a lower bound on a bosonic quadratic form is
`H ≥ −½ Tr(K_- C^{−1} K_-)` for any `C ≻ 0` with `2K − C − K_-C^{−1}K_- ⪰ 0`.
Every natural choice diverges:

* Writing `K = √π W^†W`, `K_- = √π W^†ΣW` with `(Wψ)_n = ⟨φ_n,ψ⟩/√(2^n n!)` and
  `Σ = diag((−1)^n)`, the operator `V := π^{1/4} W K^{−1/2}` is an isometry onto
  `\overline{ran\,W}`, and `K^{−1/2}K_-K^{−1/2} = V^†ΣV`.
* **ASSUMPTION (numerically supported, not proved here):** `ran W` is dense, so
  `V` is unitary. Then `κ := ‖K^{−1/2}K_-K^{−1/2}‖ = ‖Σ‖ = 1` exactly and
  `Tr[(K^{−1/2}K_-K^{−1/2})²] = Tr(Σ²) = ∞`. The Nyström evidence: at
  `n = 400, ω_max = 14` the largest `|eigenvalue|` of `A^{−1/2}BA^{−1/2}` on the
  30 kept modes is `0.999999999821`, and `Tr(K_-K^{−1}K_-)` restricted to those
  modes is already `5.57` and still growing with `n`.
* Consequently `C = θK` gives only `E_Q ≥ −∞`, and any `C` whose inverse tames
  the tail violates `2K − C − K_-C^{−1}K_- ⪰ 0`. There is no small parameter.
* The exact certificate `E_Q ≥ −½ Tr Δ` subject to
  `[[2K − Δ, −K_-],[−K_-, Δ]] ⪰ 0` is saturated by
  `Δ⋆ = K − Re[(K−K_-)^{1/2}(K+K_-)^{1/2}]`, i.e. it **needs operator square
  roots of `K ± K_-`** — unbounded operators on the half-line. Enclosing those
  rigorously is the open work.

### 7.3 A non-rigorous estimate of what that half would give

The chain `E_Q ≥ −½‖𝒜_e^{1/2} − 𝒜_o^{1/2}‖²_HS`, with
`𝒜_e = (K+K_-)/2`, `𝒜_o = (K−K_-)/2` (from `|Tr M| ≤ ‖M‖₁` applied to
`M = 𝒜_e^{1/2}𝒜_o^{1/2}`), is a genuine inequality, but its right-hand side
involves exactly those square roots. **Evaluated non-rigorously** on the
`n = 400`, `ω_max = 14` Nyström matrices it gives `E_Q ≥ −0.09572`, i.e.

```
    R ≤ 0.5761      [ESTIMATE — NOT CERTIFIED: float64, no quadrature
                     remainder, no operator-tail control]
```

We record it only to show that a certified upper bound near `0.46` is plausible
and that the obstruction is technical, not a sign that `0.4575` is wrong. It
must not be quoted as a bound. (Its coarsening via Powers–Størmer,
`E_Q ≥ −½‖K_-‖₁ ≥ −0.614` with the *proved* (3.2), is rigorous but gives
`R ≤ 3.69`, weaker than Fewster–Eveson.)

### 7.4 Other unproved or unverified items

1. **The archived value is not proved to be the limit.** We prove `R ≥ lo`. That
   `R = 0.45749073…` remains a numerical claim of the momentum-space route
   (`s6b`), now *bounded below* to `7e-08` rather than merely reproduced.
2. **Galerkin convergence rate.** The ladder's geometric decrease
   (ratio `≈ 0.27` per `+8` modes) is observed, not proved. It is used only to
   choose `n_basis`, never inside a bound.
3. **The code's own algorithm is not certified.** `_bogoliubov_ground_energy`'s
   `mode_floor` deflation and the `momentum_grid` quadrature carry no two-sided
   remainder (§2). The certificate replaces them; it does not validate them.
4. **Only the Gaussian `f`, only `T_00`, only `d = 4`, only the worldline.** The
   `f`-dependence reported in the MANIFEST (`0.2833`–`0.6110` across six
   families) is untouched; so is `ball_momentum_minimize` and the `ρ`-expansion.
5. **Lemma 2's sufficiency is finite-dimensional**, which is all we use (the
   state is vacuum outside `V_N`). No claim is made about infinite-mode
   quasi-free states.
6. **Nam–Napiórkowski–Solovej is cited, not re-proved** (§3(e)); and it is not
   load-bearing for (5.3).
7. **mpmath is trusted** for correctly-rounded `mpf` arithmetic and for the
   outward rounding of `mpmath.iv`. The interval Cholesky, triangular inverse
   and congruence are ours (`qei_certified._chol_iv`, `_tri_inv_iv`,
   `_congruence_iv`); `_chol_iv` *raises* rather than asserting, so a failure to
   certify is never silently reported as a certificate.
