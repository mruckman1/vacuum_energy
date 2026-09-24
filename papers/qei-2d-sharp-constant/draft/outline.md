# The pulled-back sampling operator: exact quantum-energy-inequality minimization on a lattice

*Draft outline — papers candidate #2, spec `docs/PLAN_LAYERS_2_3.md` §M3.4/M3.5.*
*Status: 2d methods result ready to write; the 4d successor is a **conjecture with a derived `R/τ` expansion**: `C_4d(Gaussian)/C_FE = 0.45749` from a lattice-free continuum solution (grid uncertainty `2e−6`), reproduced by the lattice route at `0.457466 ± 0.000053` (agreeing to `2.4e-05`, `0.46σ`) — see §6 for the lattice scan, §6a for the derivation and §6b for the sampling-function dependence, which is a factor 2.16 and makes "the" constant `f`-dependent.*

All numbers below are produced by `../notebook.py` and live in `../data/`.
Every quoted value names the file it comes from. Wall time for the whole
notebook: **342.7 s = 5.7 min** (budget 30 min; the 2d sections take 35 s).

---

## 1. The claim in one paragraph

For a free scalar field discretized as a harmonic chain, the entire quantum
energy inequality (QEI) minimization — *infimum of the smeared normal-ordered
energy over **all** physical states* — collapses to a single Williamson
problem. Build the local bare energy density `h_i` as a positive-semidefinite
quadratic form, pull it back along the worldline with the exact free
propagator `S(t)`, and accumulate the quadrature

    O_f = sum_a w_a f(t_a)^2 S(t_a)^T h_i S(t_a).

The smeared energy `E_f(V) = (1/2) Tr(O_f (V - V_vac))` is *linear* in the
covariance, so its infimum over the physical cone `V + iΩ/2 ⪰ 0` is attained
at the `O_f`-vacuum and equals `(1/2) Σ_k σ_k(O_f) − (1/2) Tr(O_f V_vac)`
with `σ_k` the symplectic eigenvalues of `O_f`. This is not a bound: it is
the exact minimum, and the same decomposition **exhibits the optimizing
state**. Run on the 2d massless chain, it reproduces Flanagan's *optimal*
(sharp) constant `1/(6π)` to `1.0e-7` after the three known lattice/box
artifacts are modelled (`8.2e-6` with only two of them) — against a spec
tolerance of 1%.

The methods point is that a *sharp* QEI constant is recovered, not merely a
valid bound. The route itself — the infimum as the lowest eigenvalue of one diagonalized
quadratic operator — is prior art: Dawson's 2006 York thesis and Schiappacasse, Fewster &
Ford (2018) use it (§8); what is ours is the lattice pull-back and the continuum solve. The weaker Fewster–Eveson 2d constant `1/(4π)` sits at exactly
`3/2` of it, and every ratio reported against it is `2/3` of the sharp one,
as it must be.

**Correction (2026-09-23).** This paragraph used to call the Lorentzian-weight result of
§6b the only finding in the program that moved a wall outward — "the optimal
Lorentzian-sampled 4d bound permits 2.53× more negative energy than the literature
estimate". It is a reproduction: Schiappacasse, Fewster & Ford, PRD **97**, 025013 (2018),
obtained `C_shift = −0.0593338` for the same operator and weight, and ours agrees to
+0.72 % (§6b.1, §8). What §6b adds to their paper is the 2012 number reproduced by the 2012
method, and why that method misses the spectral edge.

## 2. Setting and conventions

- Chain `H = ½ Σ_i [p_i² + m² x_i² + (x_{i+1} − x_i)²]`, Dirichlet walls,
  lattice spacing `a = 1`; every "scale" (`τ0`, `T`) is in units of `a`, so
  `a/scale = 1/scale`. `ħ = 1`, `V_vac = I/2`, block ordering
  `R = (x_1…x_N, p_1…p_N)` (`docs/API.md`).
- Smearing convention: `ρ = f²`. Flanagan, PRD **56**, 4922 (1997), Eq. (8),
  `∫ρ⟨T_00⟩ ≥ −(1/24π)∫ρ'²/ρ`, becomes
  `∫ f² ⟨:T_00:⟩ ≥ −(1/6π) ∫ f'²` — **sharp**, attained by multimode squeezed
  states. Fewster–Eveson, PRD **58**, 084010 (1998), Eq. (37) gives `1/(4π)`
  in the same convention; Eq. (40) gives the 4d worldline bound
  `−(1/16π²) ∫ f''²` used in §6 (there, `C_FE ≡ 1/(16π²)`).
- `ratio r ≡ E_min(f) / bound_sharp(f)`; `r → 1` is recovery of the sharp
  constant, `r > 1` would be a lattice state beating the continuum theorem
  (never observed).
- Box ratio `g ≡ N/τ0` (Gaussian sampling); wall ratio
  `w ≡ (N−1)/2 / t_max` (family-independent form used in §4).

## 3. The 2d recovery (data: `s2a_a_axis.csv`, `s2b_box_axis.csv`, `s2_two_parameter_fit.json`)

### 3.1 The two-artifact model

The `a/τ0` sweep at fixed `g = 100` (7 points, `τ0 = 6…24`, `N = 601…2401`)
and the box sweep at fixed `τ0 = 8` (6 points, `g = 40…220`) are described,
jointly and with no free continuum offset, by exactly two lattice artifacts:

    1 − r  =  A / g²  +  c / τ0²,     A = 4.8866,  c = 0.173767

with max |residual| `1.59e-5`, i.e. **0.45%** of `1 − r`, over all 13 points.
`c/τ0²` is the leading `O(a²)` dispersion error (`ω_k = 2|sin(k/2)|` vs `|k|`);
`A/g²` is the Dirichlet finite-box bias. Both are properties of the
*discretization*, not of the physics, so the continuum, infinite-box value of
the model is `r = 1` — the sharp constant — with nothing fitted to make it so.

The test of the claim is whether the data *want* a constant offset:

| model | free offset `b0` | significance | recovered `r` |
|---|---|---|---|
| `b0 + A/g² + c/τ0²` | `+8.20e-6 ± 3.40e-6` | `+2.41 σ` | `0.99999180` |
| `b0 + A/g² + c/τ0² + d/τ0⁴` | `−1.01e-7 ± 1.44e-6` | `−0.07 σ` | `1.00000010` |

The `2.4σ` residue of the two-artifact fit is absorbed entirely by the next
order in the *same* artifact (`d = −0.0755`), leaving an offset consistent
with zero at `0.07σ`. **Recovery of the sharp constant: `|r − 1| = 1.0e-7`
(three-artifact model), `8.2e-6` (two-artifact model).** Spec tolerance: 1%.

For comparison, the single-axis Richardson estimator used by
`tests/test_qei.py` (fit `r = r_inf + c1 h + c2 h²` in `h = 1/τ0²` at fixed
box) gives `r_inf = 0.99951380` at `g = 100` with residual `1.41e-6`. That is
not a worse method — it is the *same* number: `1 − 0.9995138 = 4.86e-4` is
`A/g² = 4.8866/10⁴` to three digits, i.e. the residual is the finite box, and
the notebook's improvement over the test is precisely that the second axis is
swept so the box bias is measured instead of left in.

### 3.2 The mass axis (`s2c_mass_axis.csv`)

`μ = m τ0 ∈ {0.08, 0.04, 0.02, 0.01, 0.005}` at `τ0 = 8`: `r(μ)` rises
monotonically to the massless value with effective exponent `p = 1.66`;
geometric Richardson in `μ` gives `0.99680923` against the directly computed
`m = 0` value `0.99679143`, deviation `1.78e-5`. Together with §3.1 this is
the joint `(a/τ0, m τ0) → (0,0)` limit the spec asks for.

### 3.3 Shape independence: four sampling families (`s2d_families.csv`, `s2d_family_fits.json`)

Flanagan's bound is optimal for *every* admissible smearing, so every family
must climb to the same constant. Four families, spanning compact/non-compact
support and Gaussian/algebraic/`C¹` regularity, each fitted with its own
`1 − r = b0 + A/w² + c/s²`:

| family | `f(t)` | `∫f'²` | `r_inf` | `b0` | `c_scale` |
|---|---|---|---|---|---|
| gaussian | `exp(−t²/2τ²)` | `√π/2τ` | `0.99996369` | `+3.6e-5 ± 1.4e-5` | `0.1708` |
| cos²     | `cos²(πt/2T)`, compact | `π²/4T` | `1.00007583` | `−7.6e-5 ± 3.6e-5` | `1.6550` |
| bump4    | `(1 − (t/T)²)²`, compact `C¹` | `256/105T` | `0.99990568` | `+9.4e-5 ± 7.8e-5` | `1.9535` |
| sech     | `sech(t/τ)` | `2/3τ` | `0.99963916` | `+3.6e-4 ± 1.3e-4` | `0.1394` |

Worst deviation from the sharp constant across families: **`3.6e-4`**. The
`c_scale` column is a by-product worth reporting: the lattice-dispersion
coefficient is a factor `~11` larger for the compact families than for the
smooth-tailed ones, because a compactly supported `f` has a broader Fourier
support and therefore samples the zone-boundary dispersion error harder. The
`bump4` and `sech` families are new here (defined in `notebook.py` §S1 with
analytic `f`, `f'`, `f''` and closed-form `∫f'²`, all finite-difference
checked to `1e-10` and `4e-12` respectively).

### 3.4 Optimizing over the sampling family (`s2e_optimize.json`)

A two-parameter family `f = e^{−s²/2}(1 + c s²)`, `s = t/τ` (which *changes
sign* for `c < 0`; legitimate because `ρ = f² ≥ 0` and `ρ'²/ρ = 4f'²` stays
finite through simple zeros) driven by `optimize_sampling`'s analytic
envelope (Hellmann–Feynman) gradient finds an interior optimum on a fixed
`N = 241` lattice at `θ* = (τ = 5.8027, c = +0.1599)`, `r = 0.99239172`, 11
iterations. A plain re-run of `θ*` through the ordinary code path reproduces
the ratio to `3.1e-12` (`papers/README.md` round-trip corollary).

## 4. The extremal states, exhibited (`s3_extremal_*.npz`, `s3_extremal_summary.csv`)

Three optimizers are archived in full: covariance `V_star`, its symplectic
spectrum, the symplectic spectrum `σ(O_f)`, the operator `O_f` and the
vacuum reference on the support, plus space and time profiles of the
normal-ordered energy density.

| setup | `N` | `E_min` | `r` | purity dev | gap/`|E|` | `min_t ⟨:h(t):⟩` |
|---|---|---|---|---|---|---|
| gaussian `τ0=8` | 801 | `−5.858118e-3` | `0.996791` | `6.7e-10` | `1.07e-4` | `−8.2484e-4` |
| cos² `T=24`     | 577 | `−5.431965e-3` | `0.995932` | `5.6e-10` | `1.22e-4` | `−4.5335e-4` |
| bump4 `T=24`    | 577 | `−5.363354e-3` | `0.995172` | `8.3e-10` | `1.32e-4` | `−3.6766e-4` |

Points to make in the paper:

- **They are pure states.** `max_k |ν_k(V_star) − 1/2| ≈ 7e-10`: multimode
  squeezed vacua, exactly the optimal-state structure Flanagan identifies
  analytically. The residual `1e-10` is the `eps_reg = 1e-8` regularizer that
  keeps the singular `O_f` invertible for the one `williamson` call.
- **The optimality gap is measured, not assumed.** The exhibited state
  achieves `E_min` to `1.1–1.3e-4` relative; that gap is the price of
  squeezing the near-null directions of `O_f` finitely rather than infinitely.
- **`O_f` is genuinely PSD-singular.** `σ(O_f)` spans `2.2e-21 … 7.80`. The
  null directions are a lattice artifact with a clean physical reading:
  zone-boundary (`k ≈ π`) wavepackets have zero group velocity, never reach
  the worldline, and are exactly invisible to the sampled density. This is
  why the spectrum must be taken inversion-free and why the extremal state
  needs its own regularized Williamson call — a design point worth a
  subsection.
- **Two independent closure checks.** (i) `smeared_energy` of the
  full-lattice embedding reproduces `achieved` to `0.0e0` (exactly, as it
  must: it reads only the support block). (ii) Integrating the time-resolved
  density `⟨:h_site(t):⟩` of the embedded state against `f²` over the *whole*
  lattice reproduces it to `1.2e-6` (gaussian), `2.8e-5` (cos²), `1.4e-5`
  (bump4) — that residue *is* the mode-support truncation error of the whole
  construction, measured rather than assumed. The local batched propagator
  used for the time profile agrees with `vacuum.core.chain_propagator` to
  `1.5e-16`.
- **The physical picture.** The time profile of the optimizer dips negative
  over the sampling window (minimum `−8.25e-4` for `τ0 = 8`) and is flanked
  in space by compensating positive energy — the quantum-interest structure,
  visible directly in the archived arrays.

## 5. Audits (`s5_audit_battery.csv`)

Nine engineered states (vacuum, the extremal state itself, a `50/50` mixture,
single-site `r = 2` squeezing, a 13-site squeezed band, four random pockets)
× both 2d constants at paper resolution (`N = 801`, `τ0 = 8`): all pass,
worst margin `1.95e-5`. Physicality is gated on **raw** symplectic margins
(`symplectic_margins_mp`, no entropy clamps), and every state's smeared
energy is separately checked to sit at or above the exhibited global minimum.
The `ν < 1/2` canary and the unphysical-input path are covered in
`tests/test_qei.py`, which passes on the producing build.

## 6. The 4d scan — a conjectured sharp constant with a stated uncertainty

**Status: conjecture, `C_4d = (0.4591 ± 0.0027) × C_FE`, total uncertainty
0.58%. Superseded by §6a**, which derives the `R/τ` form that dominated this
budget and re-extrapolates to `0.457466 ± 0.000053`, in agreement with an
independent lattice-free continuum solution `0.4574904`; the value below is
inside that band and is kept as the lattice scan's own record. This is the Layer 7 L1 deliverable (`PLAN.md`): the optimizing
states have stabilized, the constant is extrapolated with its error budget,
and the extremal state is exhibited. The first version of this section
(git history; the pivot of `M3.4` had fired) carried ~10% and three named
systematics. Each is closed below with a measured number; §6.6 lists what
the conjecture still rests on.

### 6.1 What is computed

Same machinery on the Srednicki radial lattice (PRL **71**, 666 (1993),
Eq. 10). `(l,m)` sectors decouple, so the global infimum of the
ball-and-time-smeared energy is the degeneracy-weighted sum
`Σ_l (2l+1) E_min^(l)` of independent per-sector minima. The FE 4d worldline
bound holds on every inertial worldline, so integrating it over the ball is
a rigorous inequality for our quantity and

    C_eff  =  −E_min / ( V_ball · ∫ f''² dt ),     ratio = C_eff / C_FE ≤ 1

is a well-defined diagnostic; in the continuum `C_eff` depends on `R/τ`
alone, and the worldline constant is its `R/τ → 0` limit from below (the
ball infimum needs every worldline at its own optimum at once). Gaussian
sampling, `∫ f''² = 3√π/4τ³`, `C_FE = 1/(16π²)` in the `f²` convention
(FE Eq. (40)).

### 6.2 The cancellation-free sector minimizer (`s4a_noise_floor_revisited.json`, `s4b_mp_crosscheck.json`)

`vacuum.inequalities.qei.radial_sector_minimize` solves the *same*
variational problem as `qei_minimize` in the sector's normal-mode basis,
where `V_vac = I/2` and Heisenberg evolution is a phase `e^{−iω_j t}`. The
sampled form becomes `c†Ac + ½(cᵀB̄c + h.c.) + ½Tr A` with
`A_jk = A^h_jk F̂(ω_j−ω_k)`, `B_jk = B^h_jk F̂(ω_j+ω_k)` and
`F̂ = ∫ f² e^{iνt} dt` — the time integral done analytically, never by
accumulating a `2N×2N` matrix — and the minimum is

    E_min = ½ Tr(B̄ Z),    B + AZ + ZAᵀ + Z B̄ Z = 0,    equivalently  E_min = −Σ_k σ_k ‖v_k‖²,

a bilinear of the small anomalous coupling `B` and the small squeezing `Z`
(the extremal state is `exp(½ c†Zc†)|0⟩`): no `O(l)` cancellation, and each
Bogoliubov quasimode contributes a non-positive `−σ_k‖v_k‖²`. The Riccati
equation is solved from the invariant subspace of the Bogoliubov matrix and
polished by damped Newton; modes whose sampling weight is below `1e-14` of
the peak (unresolvable in float64) stay in the vacuum, their second-order
contribution reported separately. Regression against the Williamson route
where that route is reliable: `7e-13` (2d chain), `4e-12`, `2e-11`,
`3e-10` at `l = 0, 1, 2` (`tests/test_qei.py`).

**What the archived floor actually was.** Re-running the archived
instability configuration `(N, n_ball, τ) = (200, 3, 9)` with both routes
(`s4a_noise_floor_revisited.json`) separates three things the first version
lumped together as "float64 cancellation":

| per-sector value | `l = 3` | `l = 5` | `l = 8` | `l = 12` |
|---|---|---|---|---|
| Williamson, cone support (as archived) | `−4.79e-8` | `−4.37e-8` | `−2.67e-8` | `−1.28e-8` |
| Williamson, full support | `−1.890e-8` | `−8.10e-13` | `+3.98e-13` | `+4.26e-13` |
| mode space, same 4.5σ window | `−1.890e-8` | `−9.46e-13` | `−3.49e-15` | `−1.46e-17` |
| mode space, exact Fourier transform | `−1.890e-8` | `−7.99e-13` | `−5.69e-19` | `−1.55e-25` |

- The archived `2.5e-8` floor (`1.8e-10` of `Tr(O V_vac)`) is dominated by
  the **cone-support truncation** (`weight_floor` trim + mixed reduced
  vacuum), not by arithmetic: on the full support the float64 Williamson
  route's floor at this size is `1.7e-13`, sign-indefinite. The diagnosis
  in the first draft was therefore partly wrong, and is corrected here.
- The **4.5σ time window** is a fourth, previously unnoticed artifact: the
  edge of the window gives `F̂` an algebraic tail, so the windowed sectors
  fall like a power of `l` (`−1.5e-17` at `l = 12`) where the exact Gaussian
  transform falls super-exponentially (`−1.5e-25`). At `l ≤ 2` the window
  costs `1e-7`; the scan uses the exact transform.
- The mode-space route's own floor is the resolution of the sampling
  weights: "frozen" modes (sampled too briefly to rotate, worth `−α_j/2`
  each) below `~100 ε` of the peak weight are unresolvable. Measured by
  varying the mode floor `1e-13 … 1e-15`: worst sector `4.6e-13` (at
  `l = 4`, `0.3%` of that sector, `6e-9` of the total), decaying with
  `‖B‖²` at higher `l`. It does not grow with `l`, never flips sign, and the
  relative stopping rule `|term| < 1e-12 |total|` is met at `l = 6–9` on
  every grid point; the `l`-sum is run to convergence.

**Precision cross-check through `vacuum.core.precision`**
(`qei_minimize_mp`: mpmath `iΩO` eigenproblem + mp-native vacuum covariance,
exact for the float64 operator as assembled). On the untruncated
`(64, 3, 9)` operator (128-dim, 47 s per sector) it agrees with the
mode-space route to `6.4e-14`, `2.6e-14`, `8.7e-15` at `l = 2, 3, 4` (`3e-8`,
`1e-6`, `6e-5` relative) — and with the full-support float64 route to the
same level, because at this size neither has lost digits yet; the check's
own limit, the float64 rounding of `O_f` acting through
`½Tr(δO (V*−V_vac))`, is `9e-14`. The cross-check therefore certifies the
*formulation* (an independent exact-arithmetic evaluation of the same
operator lands on the same number) rather than the tail; the tail is
certified by the floor measurement above and by the test-size demonstration
in `tests/test_qei.py` (`N = 48`, `l = 7`: Williamson `−7.1e-15`, mode space
`−4.08e-14`, mpmath `−4.14e-14` — 1.8 digits, and the Williamson terms change
sign at `l = 9` while the mode-space terms fall by `×50` per `l`).

### 6.3 The ball measure, derived (`s4c_volume_measure.csv`)

Srednicki's Eq. 10 writes `H = Σ_lm H_lm`, `H_lm = (1/2a) Σ_j [π_j² + …]`:
the `j`-sum is the rectangle rule for `∫dr` of the radial density of
`φ_lm(r) = r × (angular part)` — the `r²` of `d³x` is already absorbed —
so site `j` stands for the shell `(j−½)a < r < (j+½)a` (its bond terms sit
at `(j±½)a` and are split half/half) and, after the `(l,m)` sum, carries
the energy in that shell, of volume `V_j = 4πa³(j² + 1/12)`. The ball
operator on sites `1…n` is the energy in `a/2 < r < (n+½)a`:

    V_shell = Σ_j V_j = (4π/3)[(n+½)³ − (½)³] a³,   R_outer = (n+½) a,

exact to the `O(a²)` midpoint accuracy of the discretization. The naive
`(4π/3)(na)³` is the same data with an exactly known factor,
`V_naive/V_shell = 1 − 3/(2n) + O(1/n²)` (`−0.48` at `n = 2`, `−0.17` at
`n = 8`) — an `O(a/R)` error whose `3/(4n²)` remainder a linear Richardson
term does not absorb. Consequently every ratio below uses `V_shell` and
`ρ ≡ R/τ = (n+½)/τ`, and the "alternative converges to it" statement is
quantitative in §6.5: the naive-measure extrapolation misses the shell one
by `−0.050` (linear fit), `+0.009` (quadratic), `−0.0014` (cubic).

### 6.4 The finite box, measured (`s4d_box_sweep.json`, `s4d_wall_check.json`)

With the exact transform the outer Dirichlet wall's causal reach is
irrelevant (`f²(6τ) = e^{−36}`), but the box discretizes the sector
spectrum below `1/τ` — the 4d counterpart of the 2d `A/g²` bias of §3.
Sweeping `g = N/τ ∈ {6, 8, 10, 14, 20, 30}` at five `(n_ball, ρ)` points:

    C/C_FE (g) = C_∞ + A/g² + B/g⁴,   A ≈ −0.002,  B = −10.04 ± 0.30,   fit residual ≤ 2e-6,

i.e. a pure `1/g⁴` law with a universal coefficient (a pure `1/g²` fit
fails at `4–8e-4`). It is `1.7%` at `g = 6` — the reason the first
version's fixed-`R/τ` sequences could not agree — and `1.5e-4` at the grid's
`g = 16`, where it is corrected for with residual `5e-6`; the independent
wall check (`g = 16` vs `24`) gives `2.4e-4` against the law's `2.3e-4`.

### 6.5 The double limit, both orders (`s4d_4d_scan.csv`, `s4e_extrapolation.json`)

Product grid `n_ball ∈ {2,3,4,5,6,8}` × `ρ ∈ {½, ⅖, ⅓, ¼, ⅙, ⅛}`
(`τ = (n+½)/ρ = 5 … 68`, `N = 40 … 1096`, 36 points, every `l`-sum converged
to `1e-12`, 42 s total). Rows are fixed-`ρ` sequences, columns fixed-`n`
sequences, so the two orders of the limit are extrapolated independently.

**Order 1** (`a/τ → 0` at fixed `R/τ`, then `R/τ → 0`). At each `ρ`,
`C(x) = C_ρ + c₁x + c₂x²` in `x = a/τ` (the `O(1/n²)` measure remainder is
`O(x²)` at fixed `ρ`) fits six points with max residual `3e-5`:

| `ρ = R/τ` | ½ | ⅖ | ⅓ | ¼ | ⅙ | ⅛ |
|---|---|---|---|---|---|---|
| `C_ρ / C_FE` | `0.40247` | `0.42099` | `0.43171` | `0.44286` | `0.45115` | `0.45412` |
| stderr | `8e-5` | `8e-5` | `9e-5` | `9e-5` | `9e-5` | `9e-5` |

rising monotonically toward `ρ → 0`, as the worldline bound requires. Then
`C_ρ → C_4d`: `+d₂ρ²` gives `0.45697` (residual `7e-4`, a poor model),
`+d₂ρ² + d₄ρ⁴` gives **`0.45795`** (residual `2e-5`, the best), `+d₂ρ² + d₃ρ³`
`0.45819`, `+d₁ρ + d₂ρ²` `0.46019`, `ρ²` on `ρ ≤ ⅓` only `0.45769`.

**Order 2** (`R/τ → 0` at fixed `a/R`, then `a/R → 0`). At each `n`,
`C(ρ) = C_n + b₁ρ + b₂ρ²` over the six `ρ` (residual `2e-4`): `C_n/C_FE =
0.45342, 0.45612, 0.45741, 0.45811, 0.45853, 0.45899` for `n = 2 … 8`; then
`C_n = C_4d + e₂/(n+½)²` gives `0.45938` and `+ e₁/(n+½)` **`0.46019`**
(residual `3e-5`, the best).

**Verdict.** Central value the mean of the two best-residual models;
uncertainty the linear sum of the half-spread over all nine models and
both orders (`0.35%`), the `a/τ`-fit model dependence (quadratic vs cubic
`C_ρ`, `5e-4`), the largest fit standard error (`5e-4`) and the box-correction
residual (`5e-6`):

    C_4d / C_FE  =  0.4591 ± 0.0027  (0.58%),      order dependence 0.49%,
    C_4d  =  (2.907 ± 0.017) × 10⁻³  =  1 / (34.9 π²)      [f² convention, Gaussian sampling].

The two orders of the limit agree within their combined uncertainty; the
target was `≤ 2%`. The 2d analogue of this ratio is exactly `2/3`; in 4d
the FE constant is not sharp and the data say the sharp one is `0.459` of
it. For orientation only: the first version's `0.42 ± 0.04` band contained
this value at its upper edge, its `R/τ = ⅓` shell fit (`0.4338`) is within
`0.5%` of today's `C_⅓ = 0.4317`, and its `R/τ = ⅙` fit (`0.4563`) was `1.1%`
high — the box bias at `g ≈ 6–12` and the window tail, in the direction the
`1/g⁴` law predicts.

### 6.6 The extremal state, exhibited (`s4f_extremal_4d_l{0..7}.npz`, `s4f_extremal_4d_profile.npz`, `s4f_extremal_4d_summary.json`)

At `(ρ, n_ball) = (¼, 6)` (`τ = 26`, `N = 430`, `C/C_FE = 0.4409` raw), all
eight contributing sectors are archived: the squeezing matrix `Z`, the
sampling weights, the symplectic spectrum, the per-quasimode terms
`−σ_k‖v_k‖²`, the Takagi squeezing spectrum `tanh r_k`, the site-basis
covariance on the ball + light-cone core for `l ≤ 2`, and the `t = 0`
normal-ordered energy-density profile in `r` (per site and per shell
volume). Every sector state is pure to `≤ 4e-13` and achieves its sector
minimum exactly (gap `0`) except `l = 0`, whose null symplectic directions
are lifted by a `1e-10` weight shift at a gap of `7e-6` (the mode-space
counterpart of the 2d `eps_reg` exhibit, whose gap was `1e-4`). The state
is a few-mode multimode squeezed vacuum: `l = 0` has 3 quasimodes with
`r > 0.1` (max `0.74`), `l = 1` has 7 (max `1.13`), `l = 2` five (`0.86`),
`l = 4` one (`0.19`), `l ≥ 5` none — the squeezing content is what decays
super-exponentially in `l`, which is the physical content of the converged
`l`-sum. Sector energies `−1.22e-4, −3.99e-5, −2.36e-7, −8.5e-10, −2.2e-12,
−4.2e-15, −1.0e-17, −7.1e-19` for `l = 0 … 7`.

### 6.7 What the conjecture rests on, and what it does not

- **Rests on:** the exactness of the per-sector minimum (formulation
  regression-tested to `1e-12`, mpmath cross-check to `1e-13`), the derived
  measure and the `1/g⁴` box law (both measured, not assumed), and
  extrapolation models whose residuals (`2e-5`, `3e-5`) are far below the
  quoted uncertainty. The largest uncertainty component *was* the form of
  the `R/τ → 0` expansion (`ρ²+ρ⁴` vs `ρ²+ρ³` vs `ρ+ρ²`: `0.7%` spread).
  **§6a derives it** — even powers only, `d₂ = -0.24932` in closed form —
  which excludes two of the three models and leaves the `a/τ` extrapolation
  dominant.
- **Does not rest on:** the archived Williamson route (used only for
  comparison), any `l`-truncation, or the time window.
- **Settled in §6b, and the answer is no:** sampling-function independence
  fails in 4d. Six families give `C_sharp(f)/C_FE` from `0.283` to `0.611`,
  a factor `2.16`; the Gaussian value `0.45749` is neither the largest nor
  the smallest. The 2d family-independence of §3.3 is a property of
  Flanagan's bound, not a general one, so the 4d constant must always name
  its `f`. (`cos2_f` and `bump4_f` now carry analytic `f2hat`, so the exact-
  transform lattice route is available for them too.)
- **Not a theorem in any case:** lattice, finite `τ`, finite `R`, Gaussian
  states (the minimizer of a quadratic form is Gaussian, so the state
  optimization is global; the lattice-to-continuum step is numerical).

## 6a. The same problem in the continuum: the `R/τ` expansion, derived (§S6, `s6a…json`–`s6d…json`)

The `R/τ` form was §6.7's largest uncertainty (`0.70%` model spread, "not
derived"). It is derived here, and the derivation is checked against an
independent, lattice-free solution of the *same* variational problem.

### 6a.1 The continuum solver (`vacuum.inequalities.qei.worldline_momentum_minimize`, `ball_momentum_minimize`)

For a free field the smeared normal-ordered energy is a quadratic form in the
continuum mode operators, so its infimum over all states is a Bogoliubov
ground energy with no lattice at all. With `F̂(ν) = ∫f² e^{iνt}dt` and the
ball form factor `χ_B(q) = ∫_B e^{iq·x}d³x`, expanding `χ_B(|k − k′|)` in
spherical Bessel overlaps `J_L(ω,ω′) = ∫_0^R r² j_L(ωr) j_L(ω′r) dr` and using
`(1 + k̂·k̂′)P_L = P_L + [(L+1)P_{L+1} + L P_{L−1}]/(2L+1)` decouples the
`(l,m)` sectors into `2l+1` identical copies of

    A_l = (ωω′)^{3/2} F̂(ω−ω′)[(2l+1)J_l + l J_{l−1} + (l+1)J_{l+1}]/(π(2l+1)),
    B_l = (ωω′)^{3/2} F̂(ω+ω′)[(2l+1)J_l − l J_{l−1} − (l+1)J_{l+1}]/(π(2l+1)),

each solved by the same cancellation-free Riccati route as §6.2. As `R → 0`
only `J_0 → R³/3` survives, and the worldline infimum per unit volume is
`E_Q/(2π²)` with `E_Q` the ground energy of `(ωω′)^{3/2}F̂` — shared **equally**
by `l = 0` and the three `l = 1` copies (`l0_share` = `0.500`,
`s6b…json`): the `(1 + k̂·k̂′)` structure of `T_00` has no `l ≥ 2` content, which
is *why* the lattice `l`-sum of §6.5 converges super-exponentially.

**2d control** (`s6a…json`). The same construction in 2d must return
Flanagan's sharp constant for *every* admissible `f`. Eight families
(Gaussian, `sech`, Lorentzian, the Ford–Roman weight, `cos²`, `bump4`, and the
modulated Gaussian at `c = ±0.3`): worst `|r − 1| = 1.1e-04`
for the seven whose `f` has no zero. The eighth is a finding: at `c = −0.3`,
where `f` changes sign, the ratio converges to `0.99605` — the
`∫f'²` form is *not* the sharp bound when the weight `ρ = f²` has zeros
(Flanagan's optimal-state construction assumes `ρ > 0`), so §3.4's optimizer
was ascending toward a bound that is not attainable there.

### 6a.2 The 4d worldline constant (`s6b…json`)

    C_4d / C_FE = 0.4574904     (Gaussian, f² convention),
    C_4d = 2.897094e-03 = 1/(34.97 π²),

stable to `2e-09` between the last two momentum grids
(`n = 600, Ω = 20` vs `n = 800, Ω = 24`) and scale-invariant to `1e−9`. It sits
**inside** the archived lattice band `0.4591 ± 0.0027`, `0.35%` below its
central value — the archived value was pulled up by the `ρ + ρ²` model that
the derivation below excludes.

### 6a.3 The expansion, derived

`χ_B(q)/V_B = 1 − q²R²/10 + O(R⁴)` is an **even, entire** function of `R`.
Hence: the `l = 0, 1` kernels shift at `O(R²)` — first-order shift by
Hellmann–Feynman on the worldline extremal state, `⟨a†a⟩ = Z̄M`, `⟨aa⟩ = M`,
`M = (1 − ZZ̄)^{−1}Z` — and the `l = 2` sector, whose kernel is `O(R²)` as a
*whole operator*, contributes its own ground energy at `O(R²)` exactly
(`inf(λQ) = λ inf Q` for `λ > 0`); sector `l` opens at `O(R^{2l−2})`. Therefore

    C_eff(ρ)/C_FE = C_4d/C_FE + d₂ρ² + d₄ρ⁴ + …   —  **even powers, no odd term, no logs**,

    d₂ = d₂(HF; l = 0,1) + d₂(l = 2) = -0.28464 + 0.03532 = **-0.24932**,

the `l = 2` piece being `+5 E_{Q6}/(150π²)` in units of `C_FE ∫f''²` with `E_{Q6}`
the ground energy of `(ωω′)^{5/2}F̂` (`rho2_coefficient`). Checks:

- Hellmann–Feynman vs central finite differences on the same perturbed
  kernels: `1e-07` (`l = 0`), `5e-09` (`l = 1`).
- The derived `d₂` against the small-`ρ` slope of the ball solver at 15 radii:
  `-0.249320` fitted vs `-0.249321` derived (`1e-06`).
- **Parity**, on `ρ ≤ 0.2` (`s6c…json`, `parity_fits_rho_le_0.2`): the even
  model `C + d₂ρ² + d₄ρ⁴ + d₆ρ⁶` fits to `1.9e-09` and returns the
  worldline constant to `2e-09`; a free `ρ³` comes out at
  `+2.1e-03` (`0.8%` of `d₂`), a free `ρ` at `-1.6e-05`, a free
  `ρ² log ρ` at `+1.5e-04` — all consistent with zero.
- Full-range even series `ρ ≤ ½`: `d₂ = -0.24931`, `d₄ = +0.12634`,
  `d₆ = -0.07537`, `d₈ = +0.04850`, residual `9e-08`.

**The fitted form of §6.5 vs the derived one.** The data-chosen `ρ² + ρ⁴`
model had `d₂ = -0.24748`, `0.7%` off the derived `-0.24932` (a two-term
truncation of a series with a large `d₄`); its competitors `ρ² + ρ³` and
`ρ + ρ²` are now **excluded**. The `ρ + ρ²` model returns `0.46019` with a linear
coefficient `-0.0246` where the derivation and the continuum ball data both
say `0` (`-1.6e-05`); it is numerically the archived order-2 value
`0.46019`, i.e. the half of the archived `0.4591` average that pulled it
high, so removing it is most of the `0.35%` shift.

### 6a.4 What the lattice error actually is, and the new constant (`s6c…json`)

Subtracting the exactly-computed `C_eff(ρ)` from the archived 36-point grid
gives the lattice error at fixed `ρ` as a function of `a/R = 1/(n+½)`:

- log–log slopes over `n = 2 … 8`: `1.82 … 1.86` — the leading exponent is
  **`(a/R)²`**, not `(a/R)`; a free linear coefficient comes out at
  `+4.5e-05 … +8.2e-05`, i.e. zero.
- Global fit over all 36 points: `-0.0605(a/R)² +0.0357(a/R)³ -0.0021(a/τ)²`,
  residual `3e-04`. The `(a/τ)²` piece is the §3.1 dispersion error `c/τ₀²`;
  the `(a/R)²` piece is the midpoint accuracy of the shell measure (§6.3).

Refitting each fixed-`ρ` row in `x = a/τ` **without** the linear term
(`C_ρ + c₂x² + c₃x³`, which the exponent above licenses) reproduces the
continuum `C_eff(ρ)` at every `ρ`: deviations `0.5`: `+5.6e-06`  `0.4`: `+5.4e-06`  `0.3333`: `+4.9e-06`  `0.25`: `+4.1e-06`  `0.1667`: `+3.3e-06`  `0.125`: `+2.9e-06`. That is the
lattice machinery validated against an independent exact solution to `5e−6`.
Extrapolating those `C_ρ` with the **derived** even form:

| route | `ρ²+ρ⁴` | `ρ²+ρ⁴+ρ⁶` | `d₂` fixed at the derived value |
|---|---|---|---|
| raw grid refit, `x²+x³` | `0.457449` | `0.457490` | `0.457544` |
| raw grid refit, `x²+x³+x⁴` | `0.457442` | `0.457483` | `0.457536` |
| archived `C_ρ` (quadratic `x`) | `0.457951` | `0.457992` | `0.458027` |
| archived `C_ρ` (cubic `x`) | `0.457426` | `0.457466` | `0.457517` |

**The lattice-only number, with the derived form.** Of the six
extrapolations in the table, **only four are independent of the continuum
solve**: the two `d₂`-fixed columns pin `d₂` to the derived (continuum) value
and so cannot be used either in the central value or in the agreement figure.
The four independent ones are `0.457449`, `0.457490`, `0.457442`,
`0.457483` (the two `a/τ` fits × the two free-`d₂` `ρ`-forms), giving

    C_4d / C_FE = 0.457466 ± 0.000053   (0.011%),   ± 0.000131 conservative,

budget along the two model axes (added linearly, no double counting): the
`a/τ` fit model `7.0e-06` (`x²+x³` vs `x²+x³+x⁴`), the `ρ`-form model
`4.1e-05` (`ρ⁴` vs `ρ⁴+ρ⁶`, `d₂` free — the dominant term), and the
box-correction residual `5e-06`. Two systematics are recorded in
`lattice_only_excluded_systematics` and deliberately **not** summed, each with
its reason in the JSON:

- `d2_pinned_shift` = `7.8e-05` — the two `d₂`-pinned variants
  (`0.457544`, `0.457536`) import the continuum value of `d₂`. Adding it
  gives the **conservative** `± 0.000131` quoted above, which is what a reader
  who is happy to use the derived `d₂` numerically should take.
- `archived_a_tau_models_spread` = `3.0e-04` — the half-spread over the
  re-extrapolated *archived* `C_ρ` tables, whose quadratic-in-`x` rows carry the
  linear `a/τ` term the measured `(a/R)²` exponent rules out. Including it
  would double-count a model already excluded by measurement; it is recorded
  so a reader who distrusts the exponent can restore it.

**Honest agreement with the continuum route: `2.44e-05` (`0.005%`), i.e.
`0.46` of the lattice route's own uncertainty** — not the `3e−07` an earlier
version of this section quoted, which was an artifact of averaging the
`d₂`-pinned values (they sit above the four independent ones and happened to
drag the mean onto the continuum number). Both routes remain inside the
archived `0.4591 ± 0.0027`.

**What the two routes share, and what they do not.** The agreement is worth
quoting only to the extent the routes are independent, so, explicitly:

| input | lattice route | continuum route |
|---|---|---|
| the energies themselves | Srednicki radial lattice, cancellation-free Riccati minimizer, exact Fourier sampling, converged `l`-sum | Gauss-Legendre momentum grid, Bogoliubov ground energy of the same form |
| the ball measure | derived shell volume (§6.3) | exact `V_B = 4πR³/3` |
| the finite box | measured `−10.04/g⁴` law (§6.4) | absent by construction |
| `d₂`, `d₄` | **fitted from lattice data** (free) | computed |
| the *form* of the `ρ` series | **shared** — even powers, from §6a.3's derivation | derived and verified there |
| the *absence of a linear* `a/τ` *term* | **shared and not independent** — it is measured in §6a.4 by comparing the lattice grid *to the continuum* `C_eff(ρ)` | — |

So the agreement tests the lattice numbers and the two extrapolation steps
against an independent exact solution, *given* two functional forms, one of
which (the missing linear term) was itself read off the comparison. It is a
strong consistency check, not a blind prediction; the blind version — the
archived quadratic-in-`x` rows, extrapolated with the derived even form and no
knowledge of the exponent — lands at `0.45795`, `1.0e−3` from the continuum.

**The `0.58%` fit uncertainty is retired: the constant is `0.45749` (continuum,
`± 2e−6`), cross-checked by the lattice route at `0.457466 ± 0.000053`
(`± 0.000131` conservative), the two agreeing to `2.4e-05`.**

## 6b. Beyond Gaussian sampling: the sup over `f` (`s6d…json`)

The sharp constant of the *theorem* is the **sup over `f`** of
`C_sharp(f) = −inf_ψ ∫f²⟨:T_00:⟩ / ∫f''²`. §3.3's 2d families all give the
same number because Flanagan's bound is `f`-independent; in 4d nothing forces
that, and Fewster–Eveson say so explicitly ("we do not expect our bound to be
optimal", PRD **58**, 084010 (1998), Conclusion; "the bound is known not to be
optimal", Fewster, arXiv:1208.5399, §1.3). Each family below is one continuum
solve; the 2d control column is the same handle through the 2d kernel, which
must return `1`.

| sampling function | `C_sharp(f)/C_FE` | 2d control |
|---|---|---|
| `(1−u²)²` (compact, `C¹`) | `0.61096` ± `6e-04` | `0.999998` |
| `cos²(πt/2T)` (compact, `C¹`) | `0.58185` ± `1e-05` | `0.999999` |
| **Gaussian `e^{−t²/2τ²}`** | `0.45749` | `1.000000` |
| `sech(t/τ)` (exponential) | `0.37488` | `1.000000` |
| Lorentzian `f = 1/(1+s²)` (algebraic) | `0.33133` | `1.000000` |
| Ford–Roman weight, `f² = τ²/(t²+τ²)` (algebraic, cusped `F̂`) | `0.28332` ± `9e-06` | `0.999986` |

- **Gaussians are not optimal in 4d**: `gaussian_optimal = false`. The compact
  families beat them by up to `34%`, the algebraic ones lose up to `38%`, and
  every value is `< 1` (the Fewster–Eveson bound holds throughout).
- **The smoothness axis.** `(1−u²)^p`, `p = 2 … 8`: `p=2`: `0.6103`  `p=2.5`: `0.5754`  `p=3`: `0.5510`  `p=4`: `0.5235`  `p=6`: `0.4989`  `p=8`: `0.4876`. The ratio rises
  monotonically as `p` falls toward the admissibility edge (`f'' ∈ L²` needs
  `p > 3/2`; Fewster's `W^{2,2}` condition, arXiv:1208.5399 §1.3), so the sup
  over *this* family sits at its boundary and the optimizer
  (`optimize_sampling_momentum`) walks to `p* = 2.06` — an edge optimum, not an
  interior one. Compact, minimally-smooth weights are the direction of the sup.
- **Interior optima that are genuine.** `e^{−s²/2}(1 + c s²)`: `c* = 0.2870`,
  ratio `0.48843` vs the Gaussian's `0.45749` (`+6.8%`); adding `d s⁴`,
  `(c*, d*) = (0.313, 0.065)`, ratio `0.50624`. Both 2d controls return
  `1.000000` / `1.000000`.
- **The 2012 moment estimate** (Fewster–Ford–Roman 2012, Lorentzian
  weight): their `y_∞ = 0.02361` is **reproduced by their method on our
  kernel** (§6b.1; see its footnote on what the extraction consumes), and the gap to our `x₀ = 0.0598` lives entirely in the
  extension step `y_∞ → x₀`, which an exhibited state settles; that `x₀` reproduces
  Schiappacasse–Fewster–Ford's 2018 diagonalization, `C_shift = −0.0593338`, to +0.72 %.

### 6b.1 The acceptance test: Fewster–Ford–Roman's Lorentzian value, their method on our kernel (`s6f_ffr_method.json`, `s6e_ffr2012_comparison.json`)

Two published values of a *sharp* 4d constant exist for the Lorentzian weight.
Schiappacasse, Fewster & Ford, PRD **97**, 025013 (2018) [arXiv:1711.09477], diagonalize
the operator in a finite cavity and find `C_shift = −0.0593338`, which our `x₀` reproduces
to +0.72 % (this candidate missed that paper until 2026-09-23). The earlier one is the
moment estimate of Fewster, Ford & Roman, PRD **85**, 125038 (2012) [arXiv:1204.3570],
for the Lorentzian weight `f(t) = τ/(π(t²+τ²))` (their `f` is our `f²`, the Ford–Roman sampling
function, `sqrt_lorentzian_f`). Before anyone weighs our number against theirs,
the conventions question is closed by construction: **our pipeline outputs
their number on demand, by their method.** [Footnote: the Stieltjes extraction consumes the exact (A6) moments; the identification with OUR operator is the rank-one B, K_n = vᵀA′^{n−1}v matching (A6) to n = 16 (5.6e−7), a₂ = 9/2 and a₃ = 1890 from the kernel, and the φ² control.]

**Their method, transcribed** (`vacuum.inequalities.ffr_moments`). Sec. III /
Appendix A: the vacuum moments `μ_n = ⟨T^n⟩` of `T = ∫f :φ̇²: dt` are sums over
connected necklace graphs (Wick, Eq. (14)); in Fourier space each line carries
`∫_0^∞ dω ω^p` with `p = 3`, each vertex `f̂(ω_j ± ω_k)`, and for the Lorentzian
`f̂ = e^{−|ω|τ}` factorizes at every double-source/double-target vertex, so a
graph is a product over its *runs* of the chain integrals (A3)
`K_n = ∫ dk₁…dk_n k₁^p (k₂…k_n)^p e^{−k₁} e^{−Σ|k_{i+1}−k_i|} e^{−k_n}`, evaluated
exactly by the recurrence (A6); the sum over run structures is generated by
the derivation (A8), `C₂ = 32K₁²`, `C_n = 8ⁿ𝒦_n`, and `M = e^W` (Eqs. 21–23);
dimensionless `a_n = (4πτ²)^{2n}μ_n` (Eq. 20). Sec. IV: for a distribution on
`[−x₀, ∞)`, `M(N,y)_{mn} = a_{m+n+1} + y a_{m+n} ⪰ 0` for `y ≥ x₀` (Eqs. 58–62);
`y_N` = largest root of `det M(N,y) = 0` (Eq. 63), increasing, `y_∞ = lim y_N ≤ x₀`
(Eq. 64); 65 moments, Table II to `N = 32` at 40 digits, the fit
`y_N = a + bN^{−1/2} + cN^{−1} + dN^{−3/2}` on `21 ≤ N ≤ 33` (Eq. 68) giving
`y_∞(φ̇²) = 0.02361 ± 1e−5`, offered as the optimal bound (Eq. 69). Two of our
own devices: (A8) is a derivation, so `𝒦_n = (d/ds)^{n−2}[½K₁(s)²]` along the flow
`dK_k/ds = K_{k+1} + K₁Σ_{i+j=k}K_iK_j` — all 65 connected moments as exact
rationals in 25 s, no enumeration of the `~10⁶` run structures — and `y_N` as
minus the smallest Gauss node from the exact Chebyshev recurrence, the
well-conditioned form of their determinant.

**Their numbers, reproduced.**

| step | theirs | ours (their method) |
|---|---|---|
| Table I, `φ̇²`, 22 printed entries (5 s.f.) | `a₂ = 9/2, a₃ = 1890, …` | exact `9/2, 1890, 10206243/4`; worst rel. dev. `3.8e-05` (= rounding) |
| Table I, `φ²` control, 22 entries | `2, 48, 1740, …` | worst `4.2e-05` |
| (A11)–(A14) | explicit `C₄…C₇` | reproduced exactly |
| Table II, `y_N(φ̇²)`, `N = 2…32` (11 digits) | `0.01071401240 … 0.02129828002` | worst abs. dev. `4.9e-12` |
| Table II, `y_N(φ²)` control | `→ 1/6` | worst `4.9e-12` |
| Eq. (68) fit, `y_∞(φ̇²)` | `0.0236174942666` | **`0.0236173057`** |
| Eq. (65) fit, `y_∞(φ²)` | `0.166666666057` | `0.166666666057` |

**On our kernel.** Our operator for this weight has a rank-one anomalous
block, `B = vvᵀ/(2π)`, `v = ω^{3/2}e^{−ω}`, so its necklace traces are exactly
their chain integrals `K_n = vᵀA'^{n−1}v`; Nyström values on three grids
converge monotonically to the exact (A6) values (`K₁` to `1e−13`, Richardson
worst `5.6e-07` for `n ≤ 16`), and the identity (A4) that turns the
chain into the recurrence holds in direct quadrature to `0e+00`; `a₂ = 9/2`
is exact by hand from our kernel. **`y_∞ = 0.0236173` is FFR's `0.02361 ± 1e−5`,
reproduced within their error, and no convention is in question.** [Footnote: the Stieltjes extraction consumes the exact (A6) moments; the identification with OUR operator is the rank-one B, K_n = vᵀA′^{n−1}v matching (A6) to n = 16 (5.6e−7), a₂ = 9/2 and a₃ = 1890 from the kernel, and the φ² control.]

**What is, and is not, being compared — the sentence that matters.** Our
number is the infimum of the **spectrum** of the sampled operator `T_f` over
all states (the exhibited state is a rigorous lower bound on its magnitude).
Fewster–Ford–Roman's `y_∞` is an estimate of the lower edge of the support of
the **vacuum probability distribution** of `T_f`. In two dimensions the two
coincide — Fewster, Ford & Roman, PRD **81**, 121901(R) (2010): "The result is a
shifted Gamma distribution with the shift given by the previously known
optimal quantum inequality bound" — and in four dimensions the identification
is a conjecture (their 2012 Sec. II, via a Reeh–Schlieder-type overlap
argument), which Reeh–Schlieder does not settle because the spectral
projections of `T_f` need not be local operators. Our point-mass experiment
(§6b.1, item 2) shows that the 65-moment method cannot distinguish
"indeterminate moment problem, true vacuum-support edge deeper" from
"vacuum-support edge genuinely above the spectral bottom". **We do not settle
which of those it is.** We establish only that (i) their number is reproduced
by their method on our kernel; (ii) an exhibited state beats it as a
*spectral* statement, `x₀^{spec} ≥ 0.05976 > y_∞ + 1e−5`; and (iii) the mechanism by
which their method is blind to it is demonstrated. If the second reading is
the true one, their `y_∞` may be the correct vacuum-support edge and ours the
correct optimal quantum-inequality bound, with no contradiction between the
numbers — only between the identification of the two quantities.

**Three routes for the same weight, in FFR units** (`s6g_lorentzian_routes.json`):

| route | `x₀` | what can bias it |
|---|---|---|
| lattice-free momentum-space worldline solve (grid ladder to `n = 2400`, `1/n²` Richardson) | **`0.059762`** ± `5e-07` | no box, no lattice — infrared bias cannot enter |
| exhibited box-wavepacket state (rigorous lower bound) | `0.059762` | none (a physical state, exact expectation) |
| Srednicki lattice route: `ρ = ¼, ⅙, ⅛`, `n_ball = 3, 4, 6`, box law **measured for this weight** (`g^{−2.50}`, `B = -2.371`), `(a/R)²` then the derived even `ρ`-form | `0.059684` (`-0.131%`) | the finite box: with the Gaussian `−10.04/g⁴` law instead, the per-`ρ` misses are `-2.3e-03`; `-2.4e-03`; `-2.4e-03` |

Per-`ρ` lattice-minus-continuum with the measured law: `ρ = 0.2500`: `-3.0e-04`; `ρ = 0.1667`: `-3.4e-04`; `ρ = 0.1250`: `-3.6e-04`. **Infrared-bias
verdict: closed.** The quoted `x₀` comes from the route that has no box; the
lattice route confirms it once its own systematic is measured for this weight,
and the exhibited state confirms it without any discretization.

**So the disagreement lives entirely in the extension step `y_∞ → x₀`.** Three
things about that step, shown rather than asserted:

1. *Growth and Carleman.* `a_n ≈ (3n−3)!·3.36ⁿ` (their Appendix B class,
   "a power times `(3n−4)!`"). Both Carleman series converge — Stieltjes partial
   sum `2.667` with terms `~n^{−1.72}`, Hamburger `0.505` — so the
   sufficient criteria for determinacy are **inconclusive** (their Sec. I:
   "not covered by well-known sufficient criteria"), and the growth class is
   Stieltjes' own indeterminate example `(3n+2)!` (their Sec. VIII).
2. *The test is blind to a lower edge at `−0.0598`.* Add a point mass at
   `x = −0.059761` to the true distribution and rerun their test: weight `0.001` → `y₃₂ = 0.02134`; weight `0.01` → `y₃₂ = 0.02168`; weight `0.1` → `y₃₂ = 0.02515`
   (unperturbed `0.02130`). A 10 % mass at the true edge moves `y₃₂` by
   `0.004`; a 0.1 % mass by `4e−5`. The 65-moment edge is set by the
   `(3n)!` upper tail, not by the lower support.
3. *An exhibited state sits there.* `box_wavepacket_state` (§6b.2 below, and
   `s6e…json`): a normalizable state with `⟨x⟩ = 0.059761`, so rigorously
   `x₀ ≥ 0.0598`, exceeding the *reproduced* `y_∞ + 1e−5` by `0.0361`
   (a factor `2.53`). Their Eq. (64) `y_∞ ≤ x₀` holds; their conjecture
   `y_∞ = x₀` does not. On their own Sec. IV reasoning — for an indeterminate
   problem "`y_∞(A) = x_F(A) < x₀(A)`", the Friedrichs edge — items 1–3 together
   are positive evidence that the Lorentzian `φ̇²` moment problem is
   indeterminate, the direction they themselves expected ("on balance our
   expectation is that the problem is indeed indeterminate", Sec. VIII).

**The same `f`.** This is the Lorentzian point of §6b's family curve:
`C_sharp/C_FE = 0.283` for the Ford–Roman weight (solver `x₀ = 0.05976`), against
the Gaussian `0.457` and `(1−u²)²` `0.611` — the comparison with FFR is made at
their `f`, and every rigorous statement in their paper is respected (`y_∞ ≤ x₀`,
Fewster–Eveson `27/128`, Ford–Roman `3/2`; `φ̇²`, `(∇φ)²`, `T₀₀` share one infimum,
so their Eq. (57) holds exactly).

**How to take this.** A reproduction, not a finding (corrected 2026-09-23). Their 2012
number is right *as the output of their method* — we get it too — and the spectral edge
is where Schiappacasse, Fewster & Ford (2018) put it by diagonalization, `0.0593338`, which
our `x₀` reproduces to +0.72 %. Their 2018 paper reports the gap to the 2012 value ("of the
order of") without explaining it; items 1–2 above are the explanation, and that is what
this section adds.

### 6b.2 The exhibited state and the lattice cross-check (`s6e_ffr2012_comparison.json`)

`box_wavepacket_state` restricts the `φ̇²` form to orthonormal box wavepackets
with exact matrix elements (diagonal boxes split along the cusp,
node-independent to `8e-13`), takes the Bogoliubov vacuum of that finite form
tensored with the vacuum on the complement — a physical state; the complement
contributes exactly zero — and evaluates its expectation exactly (`‖Z‖ = 0.28`,
no clipping): `⟨x⟩ = 0.059761` at `1000` boxes (ladder `0.059701, 0.059747,
0.059758, 0.059761`), climbing toward the solver's `x₀ = 0.05976`. The same weight
on the Srednicki lattice at `ρ = ¼` agrees with the continuum ball value to
`-2.0e-04` once the finite-box law is measured *for this weight* (`g^{−2.50}`, not
the Gaussian's `g^{−4}`; with the Gaussian law wrongly applied the gap is `-2.3e-03`).

**The conjecture, stated precisely.** With `C_FE = 1/(16π²)`:

    C_4d(Gaussian)      = 0.45749 × C_FE = 2.8971e-03,   grid uncertainty 2e−6,
                          lattice route 0.457466 ± 0.000053 (± 0.000131 conservative),
                          which shares the derived rho-FORM but no fitted value,
                          and agrees with the continuum to 2.4e-05 = 0.46 of its own error;
    C_4d(sup over f)    ≥ 0.6110 × C_FE   (attained at f = (1−u²)² among those tested),
    x₀(Ford–Roman weight) ≥ 0.0598 rigorously (exhibited state), ≈ 0.0598 (solver)  [FFR-2012 units]
                          vs the 2012 moment estimate 0.02361 ± 1e−5 and the 2018 cavity diagonalization
                          0.0593338, which this reproduces to +0.72 % (§6b.1: a reproduction),
                          and the trend along `(1−u²)^p` says the sup is at the
                          `W^{2,2}` edge, so this is a lower bound on the sup, not an estimate of it.

The **honest uncertainty budget** for the Gaussian constant:

| component | size | status |
|---|---|---|
| continuum momentum grid (`n`, `Ω_max`) | `2e−6` | measured, converged |
| lattice `a/τ` fit model (`x²+x³` vs `x²+x³+x⁴`) | `7.0e-06` | measured |
| lattice `ρ`-form residual (`ρ⁴` vs `ρ⁴+ρ⁶`, `d₂` free) | `4.1e-05` | **dominant lattice term**; the leading coefficient is derived, not fitted |
| lattice `d₂`-pinned variants (not independent of the continuum) | `7.8e-05` | excluded from the headline, added in the conservative figure |
| finite box (`−10.04/g⁴`, §6.4) | `5e-06` | measured |
| sampling family | — | **not an uncertainty: a different quantity.** `C_4d(f)` varies by a factor `2.16` over the families tested, so "the 4d sharp constant" must name its `f` |

What is still not proved: that the momentum-space solver's `ρ → 0` series has
no exponentially small non-analytic piece (the argument gives even powers of
`R` from an entire form factor, and the fits see none to `1e−9`); the sup over
*all* admissible `f`, which the `p → 3/2` trend says is at the regularity
boundary and which no finite family search can settle; and the whole
continuum-limit step remains numerical.

## 7. Figures to draw from `data/`

1. `1 − r` versus `1/τ0²` at fixed `g`, and versus `1/g²` at fixed `τ0`, with
   the two-artifact plane through both (from `s2a_a_axis.csv`,
   `s2b_box_axis.csv`).
2. Four-family collapse: `1 − r + A/w²` versus `1/s²` per family, all
   through the origin (`s2d_families.csv`).
3. The extremal state: space profile `⟨:h_j:⟩` and time profile
   `⟨:h_site(t):⟩` with `f²(t)` overlaid — the negative dip and its
   positive flanks (`s3_extremal_gaussian_tau8.npz`).
4. `σ(O_f)` on a log scale, showing the 20-decade span and the null block
   (same file).
5. 4d, the double limit: `C/C_FE` versus `a/τ` at each fixed `R/τ` (six
   rows, quadratic fits) and the extrapolated `C_ρ` versus `ρ²` with the two
   best `ρ → 0` models; inset the order-2 columns `C_n` versus `1/(n+½)`
   (`s4d_4d_scan.csv`, `s4e_extrapolation.json`).
6. 4d, the floor: `|E_min^(l)|` versus `l` for the archived cone-support
   Williamson route, the full-support Williamson route, and the mode-space
   route with windowed and exact sampling (`s4a_noise_floor_revisited.json`) —
   four curves, one plot, the whole §6.2 story.
7. 4d, the box: `C/C_FE` versus `1/g⁴` at the five sweep points collapsing
   onto one line of slope `−10` (`s4d_box_sweep.json`).
8. 4d, the state: the `r`-profile of the extremal state's `t = 0` energy
   density per shell volume, and the Takagi squeezing spectra `r_k` per
   sector (`s4f_extremal_4d_profile.npz`, `s4f_extremal_4d_l*.npz`).

## 8. Related work to position against

- Flanagan, PRD **56**, 4922 (1997) — the sharp 2d constant and its
  optimal-state characterization (the target and the arbiter).
- Fewster & Eveson, PRD **58**, 084010 (1998) — the general method and the
  2d/4d constants used for comparison.
- Ford & Roman, PRD **55**, 2082 (1997) — the classic 4d Lorentzian-sampling
  constant `3/(32π²)` (a different sampling function; carried in the module
  as a reference value, not used as our denominator).
- Srednicki, PRL **71**, 666 (1993) — the radial lattice the 4d scan runs on.
- Fewster, *Lectures on quantum energy inequalities*, arXiv:1208.5399, §1.3 —
  "the bound is known not to be optimal", and the `W^{2,2}` regularity class
  that §6b's `p → 3/2` trend runs into; §5.2, "optimal bounds are lacking in
  general".
- Fewster, Ford & Roman, PRD **85**, 125038 (2012), Eqs. (64), (69) — moment
  estimates of the *optimal* 4d bound for the Lorentzian weight, the
  literature cross-check of §6b.
- Schiappacasse, Fewster & Ford, PRD **97**, 025013 (2018), arXiv:1711.09477 —
  Bogoliubov diagonalization of the Lorentzian-averaged `:φ̇²:` in a finite
  cavity; its lowest eigenvalue `C_shift = −0.0593338` is the prior value of §6b's
  `x₀`, which this candidate reproduces (+0.72 %). Missed here until 2026-09-23.
- S. P. Dawson, "Bounds on negative energy densities in quantum field theories on flat and
  curved space-times", PhD thesis, University of York (2006) — Colpa diagonalization of the
  smeared operator for 4d QI bounds (torus, squared-Lorentzian sampling), per the 2018 paper;
  the first use of this candidate's route that we know of. Not online; not read.
- Fewster & Ford, PRD **101**, 025006 (2020), arXiv:1909.07295, and PRD **103**,
  125014 (2021) of the same series — space-and-time averaging; no Gaussian-sampled 4d
  lowest eigenvalue found (the 2020 closed-form Gaussian bounds are two-dimensional).
- Ford, Helfer & Roman, PRD **66**, 124012 (2002) — no purely spatially
  averaged QEI exists in 4d, which is why the quantity of §6 must be smeared
  in time as well as space.

## 9. Open threads

- The lattice-dispersion coefficient `c_scale` varies by `~11×` across
  sampling families. Is it predictable from `∫ |f̂(ω)|² ω⁴ dω` (the leading
  dispersion correction weighted by the sampling spectrum)? Cheap to test and
  would turn a fitted nuisance into a derived quantity.
- The `2.4σ` residue of the two-artifact model is absorbed by `d/τ0⁴`. Is
  `d = −0.0755` the predicted next dispersion order? Same calculation.
- Massive 2d: the mass-axis exponent came out at `p = 1.66`, not `2`. Worth
  a short subsection — is it a genuine `m^{5/3}`-ish approach or a crossover
  between two corrections?
- 4d sampling-function independence (§6.7): add `f2hat` to `cos2_f`/`bump4`
  and rerun the grid; agreement with `0.459 ± 0.003` would turn the
  Gaussian-sampled conjecture into a conjecture for the constant.
- The 4d box coefficient `B ≈ −10.0` is the same at all five sweep points
  to `3%`; is it a property of the `l = 0, 1` infrared spectrum alone
  (`Σ_l` of a Casimir-like `(τ/N)⁴` shift)? Derivable, probably.
- ~~The `R/τ` expansion of `C_eff`~~ — **done** (§6a): even powers,
  `d₂ = -0.24932` from Hellmann–Feynman on `l = 0,1` plus the `l = 2`
  sector's own ground energy. What remains open is the *sup over `f`*
  (§6b): the trend along `(1−u²)^p` points at the `W^{2,2}` regularity
  boundary, where `∫f''²` stays finite but the optimizer wants ever less
  smooth `f` — is the sup attained, and by what?
- The `f`-dependence itself is now a quantity worth a figure: `C_sharp(f)`
  correlates with how much weight `|f̂|²ω⁴` puts at high `ω` (the compact
  families win, the algebraic ones lose). A one-parameter predictor would
  turn §6b's table into a curve.
