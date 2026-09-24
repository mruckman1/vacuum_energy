# MANIFEST — `qei-2d-sharp-constant`

## The claim

**Prior art for the method (found 2026-09-23).** Computing the sharp bound as the lowest
eigenvalue of the diagonalized smeared quadratic operator is not new. S. P. Dawson's 2006 University of York PhD thesis
applied Colpa's diagonalization of the quadratic boson Hamiltonian to quantum-inequality bounds in
4d (a torus, squared-Lorentzian sampling), as described in Schiappacasse, Fewster & Ford,
PRD 97, 025013 (2018), which itself identifies the lowest eigenvalue as the QEI bound. What this
candidate adds is the lattice pull-back formulation below, the lattice-free continuum solve, the
derived ρ-expansion and the interval-certified lower endpoint (no-longer-claims item 2).

The pulled-back sampling operator turns the quantum-energy-inequality
minimization — the infimum of the smeared normal-ordered energy over **all**
physical states — into a single Williamson problem, and thereby recovers the
**sharp** (optimal) 2d massless constant `1/(6π)` of Flanagan, PRD 56, 4922
(1997), Eq. (8) from a lattice computation, while *exhibiting* the optimizing
state rather than only bounding the energy. Building the PSD local energy
density `h_i`, pulling it back with the free propagator `S(t)` and
accumulating `O_f = Σ_a w_a f(t_a)² S(t_a)ᵀ h_i S(t_a)` makes the smeared
energy linear in the covariance, so its infimum is `½Σ_k σ_k(O_f) − ½Tr(O_f
V_vac)` in the symplectic spectrum of `O_f`. Measured recovery: the whole 2d
data set (13 points across the `a/τ0` and box axes, plus four independent
sampling-function families) is described by the known lattice artifacts
`1 − r = A/g² + c/τ0² (+ d/τ0⁴)` with **no continuum offset** —
`b0 = −1.01e-7 ± 1.44e-6` (`−0.07σ`), i.e. `r = 1.00000010`, `|r − 1| = 1.0e-7`
against a spec tolerance of 1%.

**4d successor (Layer 7, leap L1):** the same variational problem, solved
sector by sector on the Srednicki radial lattice by a *cancellation-free*
minimizer in the sector's normal-mode basis (`E_min = ½Tr(B̄Z)` with `Z` the
Riccati squeezing matrix, equivalently `−Σ_k σ_k‖v_k‖²`; exact Fourier
sampling; no support truncation), run to `l`-sum convergence on a 36-point
`(n_ball, R/τ)` product grid with the ball measure derived from Srednicki's
Eq. 10 and the finite-box bias measured (`−10.0/g⁴`) and corrected, gives,
with both orders of the double limit `a/τ → 0`, `R/τ → 0` extrapolated
independently and agreeing to 0.49%,

    C_4d / C_FE = 0.4591 ± 0.0027  (0.58%),   C_4d = (2.907 ± 0.017)e-3 = 1/(34.9 π²)

(Gaussian sampling, `f²` convention, `C_FE = 1/(16π²)` of Fewster–Eveson
Eq. (40)) — a **conjecture with a stated uncertainty**, extremal states
exhibited (`draft/outline.md` §6). **That number is superseded by the L1
hardening below and must not be quoted: the value to quote is
`C_4d/C_FE = 0.457466 ± 0.000053`** (a derived `ρ`-expansion form and a
lattice-free continuum solve; the old band contains it, and the old 0.58 % was a
fit uncertainty over an undetermined expansion form, not a measurement). The first version's three systematics
(a per-sector noise floor, the volume convention, unseparated limits) are
each closed with a measured number; the floor's diagnosis is corrected
there (it was mostly support truncation, plus a time-window tail).

**L1 hardening (`draft/outline.md` §6a, §6b; `data/s6*.json`).** The `0.58%`
was a *fit* uncertainty dominated by the undetermined form of the `R/τ`
expansion. That form is now **derived**: because the ball form factor
`χ_B(q)/V_B = 1 − q²R²/10 + O(R⁴)` is even and entire in `R`, only even
powers appear (no `ρ³`, no logs), and

    C_eff(ρ)/C_FE = C_4d/C_FE + d₂ρ² + d₄ρ⁴ + …,
    d₂ = -0.28464 (Hellmann–Feynman, l = 0,1) + 0.03532 (l = 2) = -0.24932

against the archived *fitted* `-0.24748`. The same variational problem is then
solved with **no lattice at all**, in continuum momentum space
(`worldline_momentum_minimize` / `ball_momentum_minimize`, new in
`vacuum.inequalities.qei`), giving

    C_4d / C_FE = 0.4574904  (Gaussian, momentum-grid uncertainty 2e−6),
    C_4d = 2.897094e-03 = 1/(34.97 π²),

with the 2d control returning Flanagan's sharp constant to
`1e-04` for seven families. Re-extrapolating the **lattice** grid with
the derived form — after the lattice-vs-continuum comparison shows its error
at fixed `ρ` is `(a/R)²` with **no linear term** (log–log slopes
`1.82–1.86`, free linear coefficient `≈ 6e−5`) — gives the independent

    C_4d / C_FE = 0.457466 ± 0.000053  (0.011%; ± 0.000131 conservative),

from the **four** extrapolations that use no continuum-computed value (the two
`d₂`-pinned variants are excluded — they import the derived `d₂`, so they may
not enter a number whose point is independence, and their `7.8e-05` shift is
the difference between the two uncertainties quoted). Honest agreement with
the continuum value: **`2.4e-05` = `0.46` of the lattice route's own error**
(an earlier version of this MANIFEST quoted `3e-07`, which was an artifact of
averaging the `d₂`-pinned values in). `draft/outline.md` §6a.4 tabulates what
the two routes share — the even `ρ` form, and the absence of a linear `a/τ`
term, which is itself *measured against the continuum* — and what they do not:
every lattice energy, the shell measure, the box law, and the fitted `d₂`.
Both lie inside the archived band. **Beyond Gaussian sampling:** six families give `C_sharp(f)/C_FE` from
`0.2833` (the Ford–Roman weight) to `0.6110` (the compact bump `(1−u²)²`) — a factor `2.16`. **Gaussians are
not optimal in 4d**; the sup over `f` is `≥ 0.6110 × C_FE`, and the
`(1−u²)^p` trend puts it at the `W^{2,2}` regularity edge. So the object
"the 4d sharp constant" is `f`-dependent and every quotation must name its
sampling function — which is consistent with Fewster–Eveson's own
"we do not expect our bound to be optimal".

**The acceptance test against the literature (§6b.1–6b.2, `s6f_ffr_method.json`, `s6e_ffr2012_comparison.json`).**
The 2012 moment estimate — Fewster, Ford & Roman, PRD **85**,
125038 (2012), `y_∞(φ̇²) = 0.02361 ± 1e−5` for the Lorentzian weight from 65
moments — is **reproduced by their method on our kernel** [Footnote: the Stieltjes extraction consumes the exact (A6) moments; the identification with OUR operator is the rank-one B, K_n = vᵀA′^{n−1}v matching (A6) to n = 16 (5.6e−7), a₂ = 9/2 and a₃ = 1890 from the kernel, and the φ² control.]
(`vacuum.inequalities.ffr_moments`, their Appendix A moments as exact
rationals, their Sec. IV Stieltjes test, their Eq. (68) fit): Table I to the
printed rounding (44 entries), Table II to `5e-12` (62 entries), `y_∞ = 0.0236173`
against their `0.0236175`, the `φ²` control on `1/6` to 12 digits; our kernel's
chain integrals match their exact `K_n` to `6e-07` and `a₂ = 9/2` by hand. No
convention is in question. [Footnote: the Stieltjes extraction consumes the exact (A6) moments; the identification with OUR operator is the rank-one B, K_n = vᵀA′^{n−1}v matching (A6) to n = 16 (5.6e−7), a₂ = 9/2 and a₃ = 1890 from the kernel, and the φ² control.] The disagreement with our `x₀ = 0.0598` therefore
lives entirely in the step `y_∞ → x₀`: the sequence grows as `(3n−3)!·3.36ⁿ`
(Carleman inconclusive; Stieltjes' indeterminate class), the 65-moment edge
is blind to a point mass at `−0.0598` (weight `0.1` moves `y₃₂` by `0.004`), and
**an exhibited physical state** (`box_wavepacket_state`) has `⟨x⟩ = 0.05976`, so
`x₀ ≥ 0.05976` rigorously, `2.53×` the reproduced `y_∞` — their Eq. (64)
`y_∞ ≤ x₀` holds, their conjecture `y_∞ = x₀` does not, and on their own Sec. IV
reasoning the moment problem is indeterminate (`y_∞ = x_F < x₀`), as they
expected. Same `f`: this is the Lorentzian point `0.283 C_FE` of the family
curve (Gaussian `0.457`, `(1−u²)²` `0.611`). **The sentence that matters:** our number is the infimum of the *spectrum* of
`T_f`; their `y_∞` estimates the lower edge of the *vacuum probability
distribution's* support. The two coincide in 2d (FFR, PRD **81**, 121901(R)
(2010): "the shift given by the previously known optimal quantum inequality
bound"); in 4d it is conjectured, and Reeh–Schlieder does not settle it (the
spectral projections of `T_f` need not be local). Our point-mass test shows
the moment method cannot distinguish "indeterminate, true support edge
deeper" from "vacuum-support edge above the spectral bottom". **We do not
settle which** — only (i) reproduction, (ii) an exhibited state beating
`y_∞` as a spectral statement, (iii) the blindness mechanism. Three routes
for this weight agree (`s6g…json`): continuum `x₀ = 0.05976` (no box, no
lattice), exhibited state `0.05976`, lattice route `0.05968` with the box law
measured for this weight (`g^{−2.50}`; the Gaussian `g^{−4}` law would miss by
~1 %) — the infrared-bias objection is closed. **Prior work, found 2026-09-23: this is a reproduction, not a finding.** Schiappacasse,
Fewster & Ford, PRD **97**, 025013 (2018), arXiv:1711.09477 — two of the three 2012
authors — diagonalized the same operator (the normal-ordered `φ̇²` of the 4d massless
scalar, Lorentzian weight) by a Bogoliubov transformation in a finite spherical cavity and
obtained its lowest eigenvalue `C_shift = −0.0593338` in these units, setting it beside the
2012 value themselves ("of the order of the predicted value from the analysis using high
moments, x0 = −0.0236"). Our continuum `x₀ = 0.059762` reproduces theirs to **+0.72 %**
(infinite space against their cavity; `tests/test_qei.py::test_ffr2012_lorentzian_acceptance`
(vi)). From 2026-09-04 to 2026-09-23 this paragraph presented the `2.53×` as a finding
against a published number and as the only result in the program that moved a wall
outward; that framing is withdrawn (no-longer-claims item 1). Their paper also takes the
lowest eigenvalue to be both the QEI bound and the lower edge of the vacuum distribution —
in their cavity the vacuum sits on the ground state with probability `0.970` — which bears
on "the sentence that matters" above; in the continuum the question stands as stated
there. **What the 2018 paper does not contain, and this candidate adds:** the 2012 number
reproduced by the 2012 method on our kernel, and the demonstration that the 65-moment
reconstruction is blind to the bottom of the spectrum — an explanation of the factor
`≈ 2.5` that the 2018 paper reports without explaining.

**S8, the certified enclosure (Layer 1, this pass; `draft/proof_4d_gaussian_constant.md`,
`data/s8_certified_enclosure.json`).** The Gaussian continuum number is no
longer only numerical. The analytic formulation behind it is derived in full —
the mode expansion of the 4d massless scalar, the `f²`-smeared `T_00`, the exact
decoupling into the `l = 0` copy and three `l = 1` copies of **one** radial
quadratic form with kernels `K = (ww')^{3/2} Fhat(w-w')`,
`K_- = (ww')^{3/2} Fhat(w+w')` and sector weights `c_0 = 1/(4π²)`,
`c_1 = 1/(12π²)`, `c_0 + 3c_1 = 1/(2π²)`, giving `R = -32 E_Q/(3√π)` — and
matched to `vacuum/inequalities/qei.py` line by line (`fhat_of`,
`fsecond_sq_integral`, `_worldline_kernels`, `_bogoliubov_ground_energy`,
`worldline_momentum_minimize`'s sector assembly). **No discrepancy was found**;
the only difference is the sign of the pair kernel, which is immaterial
(`a → ia`). Well-posedness is established: `K ⪰ 0` unbounded, `K_-` trace class
with the proved bound `‖K_-‖₁ ≤ 3√π e²/32 = 1.2277` (numerically `1.2114`),
`K ± K_- ⪰ 0`. A **new module `vacuum/inequalities/qei_certified.py`** then turns
one half of an enclosure into a computer-assisted theorem: exact Γ-series matrix
elements in the basis `φ_n(w) = w^{n+3/2}e^{-w²/4}` with a proved geometric tail
(Wendel's inequality), interval Cholesky / triangular inverse / congruence, an
**exhibited** finitely-many-mode squeezed state whose covariance is *verified* to
be a state by an interval Cholesky of `[[N,M],[M,I+N]]`, and its energy evaluated
in `mpmath.iv`:

    0.457490709365 <= C_4d(Gaussian)/C_FE <= 1

the lower endpoint a theorem (`n_basis = 64`, `dps = 674`, 2847 series terms),
the upper endpoint **Fewster & Eveson, PRD 58, 084010 (1998), Eq. (40) — cited,
not computed here**. The archived momentum value `0.45749073394875556` sits
`2.46e-08` above the certified lower endpoint, i.e. `1/81` of that route's own
stated grid uncertainty. **The upper half is NOT closed.** The problem is
critical (`‖K^{-1/2}K_-K^{-1/2}‖ = 1`, `Tr(K_-K^{-1}K_-) = ∞`), so every
standard perturbative lower bound on `E_Q` diverges and the exact dual
certificate needs operator square roots of `K ± K_-`; a non-rigorous evaluation
of the Powers-Størmer chain on the `n = 400` Nyström matrices suggests a
certified `R ≤ 0.576` is within reach, but that number is an **ESTIMATE, not a
bound** (`draft/proof_4d_gaussian_constant.md` §7).

## What this candidate no longer claims

1. **"The 4d Lorentzian-sampled bound is 2.53× the literature estimate — a finding against
   a published number, and the only result in the program that moved a wall outward"**
   (2026-09-04 to 2026-09-23, here, in `draft/outline.md` §1 and §6b.1, and in the
   repository's `README.md`, `docs/PRIORITY.md`, `docs/STATUS.md` and
   `docs/DISCOVERIES.md`). The numbers stand; the novelty does not. Schiappacasse, Fewster &
   Ford, PRD 97, 025013 (2018), published the same lowest eigenvalue for the same operator
   and weight, `C_shift = −0.0593338`, which ours reproduces to +0.72 %. The literature search
   behind the NEW classification looked for other groups' work and for Gaussian values; it
   never checked the 2012 authors' own follow-ups, which is where a revision of their
   estimate would appear.
2. **"The pulled-back sampling operator is a new route to sharp QEI constants"** (the "what is
   new" column of this candidate's row in `docs/PRIORITY.md`, until 2026-09-23). The route — the
   infimum over all states as the lowest eigenvalue of one diagonalized quadratic operator — was
   used by Dawson (PhD thesis, York, 2006) and by Schiappacasse, Fewster & Ford (2018). The
   lattice pull-back is this candidate's formulation of it, not a new route.

## Modules and inputs used

- `vacuum.inequalities.qei` — **new in the L1 hardening pass:**
  `worldline_momentum_minimize` / `ball_momentum_minimize` /
  `momentum_qei_ratio` / `momentum_grid` / `MomentumResult` (the same
  variational problem in continuum momentum space, no lattice),
  `rho2_coefficient` (the derived `ρ²` coefficient),
  `optimize_sampling_momentum` (family optimization on that route),
  `fhat_of` / `fsecond_sq_integral` / `FEWSTER_EVESON_CONSTANT_4D`, the
  handles `lorentzian_f`, `sqrt_lorentzian_f`, `bump_f`, `gaussian_poly_f`,
  analytic `f2hat` / `fpp2` on `gaussian_f` and `cos2_f`, the `operator`
  split (`'T00'`/`'phidot2'`/`'gradphi2'`) of `worldline_momentum_minimize`,
  and `box_wavepacket_state` (the exhibited variational state of §6b.2);
- `vacuum.inequalities.qei_certified` — **new in S8:** `phi_gram`,
  `phi_kernel_elements` (exact Γ-series elements + proved tail),
  `trial_state_covariance`, `certified_lower_bound` / `certified_enclosure`
  (`CertifiedEnclosure`, `KernelElements`), `ratio_from_energy`,
  `CERTIFIED_PINNED_RATIO`, `FEWSTER_EVESON_RATIO_UPPER_BOUND`; it imports
  **nothing** from `vacuum.inequalities.qei` (the two routes are compared, not
  shared)
- `vacuum.inequalities.ffr_moments` — **new:** Fewster–Ford–Roman 2012's
  moment method transcribed (`chain_integrals_exact`, `run_structure_cumulants`,
  `ffr_moment_sequence`, `jacobi_recurrence`, `stieltjes_edge`, `ffr_fit_y_inf`,
  `carleman_diagnostics`, `ffr_identity_A4`, `kernel_chain_integrals`,
  `point_mass_perturbation`; their Tables I and II as constants);
  previously — `sampling_operator`, `qei_minimize`,
  `qei_ratio`, `qei_bound_2d`, `qei_audit`, `embed_extremal`,
  `smeared_energy`, `squeezed_pocket`, `optimize_sampling`,
  `gaussian_f`, `cos2_f`, `FHandle`, `SamplingFamily`,
  `radial_ball_weights`, `radial_sampling_operator`, `radial_ball_qei`
  (`method='mode_space'`), **new in this candidate:**
  `radial_sector_minimize` / `mode_space_minimize` / `chain_worldline_minimize`
  (the cancellation-free minimizer, `ModeSpaceResult`),
  `mode_space_extremal_state`, `qei_minimize_mp` (precision cross-check
  through `vacuum.core.precision`), `shell_volume`, `naive_volume`,
  `FHandle.f2hat` (analytic Fourier transform of `f²`),
  `SHARP_CONSTANT_2D`, `FEWSTER_EVESON_CONSTANT_2D`,
  `chain_energy_density_form`, `fprime_sq_integral`
- `vacuum.core` — `harmonic_chain_K`, `ground_state_cov`,
  `symplectic_eigenvalues`, `chain_propagator`, `reduce`
- `vacuum.core.radial` — `srednicki_K` (the partial-wave Hamiltonian)
- `vacuum.core.precision` — `_abs_eig_pairs_mp`, `ground_state_cov_mp`
  (inside `qei_minimize_mp`)
- **No `experiments/` files.** The only external inputs are published
  *analytic* constants, each cited at its point of use:
  - Flanagan, PRD **56**, 4922 (1997), Eq. (8) → `1/(6π)` (sharp 2d)
  - Fewster & Eveson, PRD **58**, 084010 (1998), Eq. (37) → `1/(4π)` (2d),
    Eq. (40) → `1/(16π²)` (4d worldline, `f²` convention)
  - Ford & Roman, PRD **55**, 2082 (1997) → `3/(32π²)` (4d Lorentzian
    sampling; carried as a reference value, not used as a denominator)
  - Srednicki, PRL **71**, 666 (1993), Eq. 10 → the radial lattice and, by
    the derivation in `shell_volume`, the ball measure
  No number in `data/` is transcribed from a plot or a table; every one is
  computed by `notebook.py` from these closed forms plus the package.

## Files

```
notebook.py                          re-runnable top to bottom, no hidden state
data/build_info.json                 the build the numbers came from (git HEAD, dirty-file list)
data/s1_family_selfcheck.json        FD checks of the new families' derivatives
data/s2a_a_axis.csv                  a/tau0 sweep, Gaussian, g = 100
data/s2b_box_axis.csv                box sweep at tau0 = 8
data/s2_two_parameter_fit.json       the artifact model and its free-offset tests
data/s2c_mass_axis*.{csv,json}       m*tau0 -> 0
data/s2d_families.csv                four families x (scale sweep + wall scan)
data/s2d_family_fits.json            per-family r_inf
data/s2e_optimize.json               optimize_sampling + plain-numpy replay
data/s3_extremal_*.npz               2d V_star, nu(V_star), sigma(O_f), O_f, profiles
data/s3_extremal_summary.csv         the 2d diagnostics table
data/s4a_noise_floor_revisited.json  archived (200,3,9) config: cone/full Williamson vs
                                     mode space (windowed / exact), mode-floor spread per l
data/s4b_mp_crosscheck.json          mpmath Williamson vs float64 vs mode space, l = 2,3,4
data/s4c_volume_measure.csv          V_shell, V_naive, the 1/n expansion
data/s4d_box_sweep.json              C(g) at five points: A/g^2 + B/g^4 fits, B = -10.04
data/s4d_4d_scan.csv                 the 36-point (n_ball, R/tau) grid, raw and box-corrected
data/s4d_lterms_rho*_nb*.npz         per-l terms and per-sector diagnostics behind each point
data/s4d_wall_check.json             g = 16 vs 24 at one point vs the g^-4 law
data/s4e_extrapolation.json          order-1 and order-2 fits, all models, the verdict
data/s4f_extremal_4d_l{0..7}.npz     4d extremal state per sector: Z, weights, sigma,
                                     mode terms, squeezing spectrum, V_star core (l <= 2)
data/s4f_extremal_4d_profile.npz     t = 0 energy-density profile in r (per site / volume)
data/s4f_extremal_4d_summary.json    the 4d exhibit's diagnostics table
data/s5_audit_battery.csv            9 states x 2 constants, all gated
data/s6a_momentum_2d_control.json    Flanagan's sharp constant from the continuum kernel, 8 families
data/s6b_momentum_4d_worldline.json  C_4d/C_FE for Gaussian sampling, momentum-grid ladder
data/s6c_rho_expansion.json          the derived R/tau expansion, its parity checks, the
                                     lattice-vs-continuum error exponent, re-extrapolations
data/s6d_families_4d.json            C_sharp(f)/C_FE per sampling family + three optimizations
data/s6g_lorentzian_routes.json      the Ford-Roman weight through three routes side by side
                                     (continuum worldline, exhibited state, lattice with its own box law)
data/s6f_ffr_method.json             FFR-2012 by THEIR method on our kernel: 65 exact moments, Table I,
                                     (A4), our-kernel chain integrals, Table II, fit (68) -> y_inf, Carleman,
                                     point-mass blindness, the exhibited state next to the reproduced y_inf
data/s6e_ffr2012_comparison.json     the FFR-2012 acceptance test: phidot^2/(grad phi)^2/T00 infima
                                     for the Ford-Roman weight, the weight-specific box law,
                                     lattice-vs-continuum at rho = 1/4, verdict
data/s8_certified_enclosure.json     S8: the certified ladder (n_basis 16..64), the
                                     element cross-check against the qei kernel, the
                                     mutations, the enclosure, the clearly-labelled
                                     NON-CERTIFIED Powers-Stormer diagnostic, build info
notebook_certified.py                S8 alone, re-runnable; --quick writes outside data/
data/summary.json                    every headline number in one place
draft/proof_4d_gaussian_constant.md  S8: theorem statement, the derivation, the code
                                     correspondence, the error budget, and the explicit
                                     list of what is NOT proved
draft/outline.md                     the manuscript outline, incl. the 4d conjecture (section 6),
                                     the derived R/tau expansion (6a) and the f-dependence (6b)
```

Regenerate everything with:

```
.venv/bin/python papers/qei-2d-sharp-constant/notebook.py
.venv/bin/python papers/qei-2d-sharp-constant/notebook.py --section S6   # S6 alone, against the archived S4 tables
.venv/bin/python papers/qei-2d-sharp-constant/notebook_certified.py --ladder 16,24,32,40,48,56,64   # S8, ~9 min
.venv/bin/python papers/qei-2d-sharp-constant/notebook_certified.py --quick                          # S8 smoke, ~10 s, outside data/
.venv/bin/python -m pytest tests/test_qei_certified.py -q -p no:cacheprovider                        # 16 s
QEI_CERT_SLOW=1 .venv/bin/python -m pytest tests/test_qei_certified.py -q -p no:cacheprovider        # + the n_basis = 40 rung
```

## Build this candidate was produced from

- Date: **2026-09-02** (local; `data/build_info.json` carries the UTC stamp)
- Repo: git, HEAD **`3dc22ffe8b6a603086e7e61ed9d93e1190200834`**
  (`Checkpoint: Layer 7 L5 …`). The producing tree is that commit plus
  **uncommitted edits to exactly** `vacuum/inequalities/qei.py`,
  `tests/test_qei.py`, `papers/qei-2d-sharp-constant/notebook.py` and
  `papers/qei-2d-sharp-constant/draft/outline.md` (listed in
  `build_info.json: git_owned_files_status`); nothing is committed by this
  candidate, per instructions. Reproduce by checking out that HEAD and
  applying those four files.
- Package: `vacuum` 0.1.0, editable install; Python 3.13.3, numpy 2.5.2,
  scipy 1.18.1, mpmath 1.4.1, macOS-14.6.1-arm64
- **L1 hardening pass (this pass): `data/s6*.json` and the `S6_*` keys of
  `summary.json` were produced on 2026-09-04 by
  `notebook.py --section S6` on a tree that began the pass at HEAD `7ec96cb`
  and ended it at HEAD `04d6338` (four commits landed from other layers
  while it ran — Layer 2, Layer 5, L6 — none of which touches
  `vacuum/inequalities/`, `vacuum/core/` or this candidate), plus
  uncommitted edits to
  exactly `vacuum/inequalities/qei.py`, `tests/test_qei.py`,
  `papers/qei-2d-sharp-constant/notebook.py`, `.../MANIFEST.md` and
  `.../draft/outline.md`** (nothing committed, per instructions). That
  section reads the archived `data/s4d_4d_scan.csv` and
  `data/s4e_extrapolation.json` and writes only `s6*.json` plus the `S6_*`
  keys of `summary.json`; **no S1–S5 file was regenerated or altered**, so
  every 2d and lattice-4d number above is bit-identical to the build below.
  S6 wall time **844 s = 14.1 min** (S6a–S6c ~4 min, S6d ~10 min);
  `--quick` runs the whole notebook including a reduced S6 in **205 s** and
  reproduces every S6 conclusion (`C_4d/C_FE = 0.4574904`, lattice-only
  `0.457466 ± 0.000053`, `d₂ = −0.24932`, Gaussian not optimal). `tests/
  test_qei.py`: **32 passed in 61 s** on this tree (25 before, +7 for S6).
- **Audit follow-ups (same pass, HEAD `62bf22f`):** (a) the lattice-vs-continuum
  agreement recomputed without the `d₂`-pinned fits (`--section S6c`,
  honest figure `2.4e−5`, §6a.4); (b) the acceptance test against
  Fewster–Ford–Roman 2012 (`--section S6e`, ≈ 22 min: the `φ̇²`/`(∇φ)²`/`T_00`
  infima for the Ford–Roman weight, a weight-specific finite-box sweep on
  the lattice, and the lattice-vs-continuum check at `ρ = ¼`); (c) one
  robustness addition to `vacuum.inequalities.qei._riccati_solve`: a
  QR-based (`gelsy`) fallback reached only where LAPACK's SVD-based least
  squares raised `LinAlgError` on a large cusped-kernel sector — it changes
  no previously computed number. `tests/test_qei.py`: **35 passed**
  (+`test_d2_pinned_extrapolation_is_not_continuum_independent`,
  +`test_ffr2012_lorentzian_acceptance`,
  +`test_box_wavepacket_state_beats_ffr` — the exhibited state's
  `⟨x⟩` must exceed FFR's `y_∞ + 1e−5` by a stated margin of `0.036`, be
  normalizable, node-independent to `1e−7`, and sit below the solver's
  infimum; the Gaussian control likewise). The S6e section re-run
  (`--section S6e`, ≈ 25 min) now also archives the exhibited-state ladder
  (150 … 1000 boxes). (d) `--section S6f` (≈ 3 min; moments 36 s,
  Stieltjes 4 s) runs Fewster–Ford–Roman's method on our kernel
  through the new module `vacuum/inequalities/ffr_moments.py`, with five
  regression tests (`test_ffr_table_I_moments_reproduced`,
  `test_ffr_kernel_is_their_kernel`, `test_ffr_table_II_and_y_inf_reproduced`,
  `test_ffr_carleman_and_blindness`, `test_exhibited_state_above_reproduced_y_inf`).
  `tests/test_qei.py`: **40 passed in 81 s**. (e) `--section S6g` (≈ 55 min,
  dominated by the Lorentzian-weight lattice points at `N` up to 850 and the
  `n = 2400` momentum solve) puts the Ford–Roman weight through the three
  routes side by side — continuum worldline `0.059762`, lattice route
  `0.059684` with the box law measured for this weight (`g^{−2.50}`), exhibited
  state `0.059762` — with `test_lorentzian_three_routes_and_ir_bias` as the
  regression (one lattice point corrected with that law must land within its
  `(a/R)²` term of the continuum ball value, and the Gaussian `g^{−4}` law must
  visibly fail). Function rename `ffr_moments` → `ffr_moment_sequence` so the
  module is no longer shadowed by the package re-export; `tests/test_exports.py`
  and `tests/test_qei.py`: **72 passed** (42 in `test_qei.py`).
- Notebook wall time (S0–S5, unchanged): **342.7 s = 5.7 min** (budget 30 min;
  with S6 the full run is ~20 min): S2–S3 (2d)
  22 s, S4 320 s — of which S4b (three 128-dim mpmath eigenproblems) 141 s,
  S4d (box sweep + 36-point grid + wall check) ~100 s.
- The 2d numbers are unchanged from the previous build to every printed
  digit (`s2_two_parameter_fit.json`, `s2d_family_fits.json`,
  `s3_extremal_summary.csv`, `s5_audit_battery.csv` regenerate identically;
  only their timestamps move). The only edit to `notebook.py` after the
  data run is the provenance path fix in `s0_provenance` (it re-wrote
  `build_info.json` alone).

## Admissibility (per `papers/README.md`)

**Full known-results suite on the exact producing tree:**

```
$ .venv/bin/python -m pytest -q
932 passed, 5 warnings in 592.72s (0:09:52)      # 2026-09-02, HEAD 3dc22ff + the four edited files
```

`tests/test_qei.py` (25 tests, 15 s) now also pins the mode-space
minimizer: agreement with the Williamson route to `7e-13` (chain), `4e-12`
/ `2e-11` / `3e-10` (`l = 0, 1, 2`), the high-`l` demonstration with the
mpmath cross-check (`N = 48`, `l = 7`: Williamson 83% off, mode space 1.4%
off the mp value; Williamson terms change sign at `l = 9`, mode-space terms
fall by `×50` per `l`), purity and gap of the exhibited state, the
converged `l`-sum entry point, and the measure identities. The 4d scan
itself stays out of the suite (spec M3.4).

**Integration: full suite 1201 passed on the quiescent tree at HEAD `c18a206`, 2026-09-04** (`.venv/bin/python -m pytest -q`, twice in a row on the final tree: 1201 passed, 10 warnings in 731.91s (0:12:11) and 1201 passed, 10 warnings in 740.37s (0:12:20); with tenpy, cvxpy, jax, jaxlib, qiskit, qiskit_aer and qiskit_ibm_runtime blocked at the import system, 1026 passed, 76 skipped in 428.84s (0:07:08); all 15 notebook `--quick` paths exit 0, the slowest in 307 s). The tree is commit `c18a206` plus this pass's uncommitted edits — README, STATUS, PRIORITY, MANIFEST and draft prose reconciling the rollups to the MANIFESTs, plus one print in `interacting-vacua-first-numbers/notebook_qei_exact.py` made None-safe so its `--quick` path exits 0 (no numerical code, no data). `data/` was produced on the build recorded above; this run is the admissibility record under `papers/README.md` for the candidate's code as it stands on that tree, and a full re-run of `notebook.py` there regenerates `data/` with a new `generator_sha256`; the producing-build fields above are left as recorded.

## Audit status

- **Precision.** Physicality is decided on **raw** symplectic margins
  (`vacuum.core.precision.symplectic_margins_mp`, no entropy clamps) inside
  `qei_audit`; the `ν < 1/2` canary and the escalation path are covered by
  `tests/test_qei.py` on this build. The 4d sector minimizer is
  cross-checked against an mpmath Williamson evaluation of the same
  operator (`qei_minimize_mp`) to `1e-13` at `l = 2, 3, 4` (S4b), the
  check's own limit being the float64 assembly of `O_f` (`9e-14`).
- **QEI battery (S5).** 9 engineered states × 2 constants at paper
  resolution (`N = 801`, `τ0 = 8`): all pass, worst margin `1.95e-5`. Every
  state's smeared energy is separately asserted to sit at or above the
  exhibited global minimum. No violation by a physical state was observed
  at any point in this candidate, so **no anomaly-protocol event is open**.
  In 4d, `ratio ≤ 1` (the integrated FE bound) holds on all 36 grid points
  and all 30 box-sweep points (max `0.4544` raw).
- **Extremal-state closure.** 2d: `smeared_energy` of the full-lattice
  embedding reproduces `achieved` exactly; the independent untruncated
  time-integral to `1.2e-6 … 2.8e-5`. 4d: every sector state is pure to
  `≤ 4e-13`, achieves its sector minimum with gap `0` (`l ≥ 1`) or `7e-6`
  (`l = 0`, weight-shift regularized), and the exhibited state's exact
  variational energy agrees with the float64 trace of the untruncated
  `O_f` against it to `1e-8` (test).
- **Propagator cross-check.** The batched normal-mode propagator used for
  the 2d time profiles agrees with `vacuum.core.chain_propagator` to
  `1.5e-16`.
- **Optimizer round trip** (`papers/README.md` corollary). The
  `optimize_sampling` optimum is re-run through the ordinary code path:
  ratio reproduced to `3.1e-12`.
- **4d systematics, each measured.** Sector floor: `4.6e-13` worst sector
  (mode-floor variation), `l`-sums converged to `1e-12` relative on every
  point. Measure: derived, alternative differs by the exact factor
  `1 − 3/(2n) + O(1/n²)`. Box: `−10.04 ± 0.30 / g⁴`, fit residual `≤ 2e-6`,
  wall check `2.4e-4` observed vs `2.3e-4` predicted. Limits: order 1 vs
  order 2 `0.49%`; model spread `0.70%`; `a/τ`-fit model dependence
  `5e-4`; fit stderr `5e-4`.

## S8 — the certified enclosure (Layer 1, this pass)

- **Build.** `data/s8_certified_enclosure.json` produced 2026-09-12 (UTC stamps
  in the file) by `notebook_certified.py --ladder 16,24,32,40,48,56,64` at HEAD
  `7efd1b5c` plus this pass's own uncommitted files, 2 threads. The seven rungs
  cost **602 s** (`2, 9, 20, 50, 103, 182, 234 s`); the file's `build.wall_seconds`
  reads `81 s` because the final pass resumed an existing ladder and recomputed
  only S8a/S8c/S8e. Python 3.13.3, mpmath 1.4.1, numpy 2.5.2. **Reproduced:** an
  independent cold re-run of the whole ladder returned every rung identical to all
  12 printed digits and the same enclosure endpoint `0.457490709365`.
- **What is proved.** `C_4d(Gaussian)/C_FE ≥ 0.457490709365` — a theorem, not a
  fit: an explicitly exhibited 64-mode Gaussian state of the radial sectors,
  whose covariance is verified to satisfy the Robertson–Schrödinger condition by
  an interval Cholesky, evaluated in `mpmath.iv` with a proved Γ-series tail
  bound. Every step is an inequality; there is **no quadrature in ω at all**, so
  there is no discretisation remainder to estimate. Ladder (gap to the archived
  `0.45749073394875556`): `n_basis = 16 → 1.52e-04`, `24 → 1.77e-05`,
  `32 → 3.21e-06`, `40 → 7.59e-07`, `48 → 2.15e-07`, `56 → 6.92e-08`,
  `64 → 2.46e-08`; monotone increasing, as Galerkin requires. The interval
  arithmetic contributes nothing to that gap: the energy enclosure's own relative
  width is `10^-157.7` at `n_basis = 16` and `10^-385.6` at `n_basis = 64`
  (`energy_width_log10`), so the whole gap is Galerkin deficiency.
- **What is not proved.** The upper endpoint is Fewster–Eveson's `R ≤ 1`, cited.
  The enclosure width `0.542` is entirely the missing rigorous lower bound on
  `E_Q`. `draft/proof_4d_gaussian_constant.md` §7 lists seven further items,
  including the one **ASSUMPTION** used only in the diagnosis of why that half is
  hard (density of `ran W`, numerically supported at `κ = 0.999999999821`).
- **Formulation check (not a self-report).** `qei_certified` shares no code with
  `qei`. Its Γ-series matrix elements reproduce a 4000-node 2D Gauss–Legendre
  quadrature of the kernel built by `qei.fhat_of(qei.gaussian_f(1.0))` to
  `8.4e-13` (`K`) and `8.7e-13` (`K_-`), worst of 36 entries.
- **Mutations (falsification handles).** At `n_basis = 40`: both kernels
  `×(1+1e-4)` → `0.457535724` (**excludes** the archived value); `K_-` alone
  `×(1+1e-4)` → `0.457628840` (**excludes**); `K_-` `×(1−1e-4)` → `0.457351109`
  (moves down, correctly does not exclude). A second mutation feeds the verifier
  a `(N, M)` pair violating the state condition and requires it to raise.
- **The non-certified diagnostic, labelled as such.** `s8e` records
  `R ≤ 0.576051` from the Powers–Størmer chain evaluated in float64 on the
  `n = 400` Nyström matrices. **That is an ESTIMATE, not a bound** (no quadrature
  remainder, no operator-tail control) and must not be quoted; it is kept only to
  show that the missing half of the enclosure has room. Its rigorous coarsening,
  `R ≤ 16‖K_-‖₁/(3√π) ≤ 3.6945` with the proved `‖K_-‖₁ ≤ 3√π e²/32`, is weaker
  than Fewster–Eveson's `1` and is therefore not used.
- **Tests.** `tests/test_qei_certified.py`: **12 passed, 1 skipped in 14 s**
  (`test_ratio_from_energy_matches_the_archived_conversion`,
  `test_phi_gram_closed_form`, `test_elements_match_the_qei_module_kernel`,
  `test_enclosure_contains_the_archived_value`,
  `test_ladder_increases_and_never_crosses_the_archived_value`,
  `test_mutated_kernel_enclosure_excludes_the_archived_value`,
  `test_state_positivity_is_actually_verified`,
  `test_certified_enclosure_alias`, plus four that pin the archive itself —
  `test_s8_archive_pins_the_enclosure`,
  `test_s8_archive_ladder_is_monotone_and_converging`,
  `test_s8_archive_formulation_and_mutations`,
  `test_s8_archive_powers_stormer_is_labelled_not_certified`; the skipped one is
  the `n_basis = 40` rung behind `QEI_CERT_SLOW=1`). `tests/test_exports.py`: 31
  passed with the new module registered in `vacuum/inequalities/__init__.py`.
  `tests/test_qei.py`: **42 passed in 352 s**, unchanged by this pass.

## Headline numbers

| quantity | value | file |
|---|---|---|
| sharp 2d recovery, 3-artifact model | `r = 1.00000010`, `|r−1| = 1.0e-7` | `s2_two_parameter_fit.json` |
| sharp 2d recovery, 2-artifact model | `r = 0.99999180`, `|r−1| = 8.2e-6` | `s2_two_parameter_fit.json` |
| lattice-dispersion coefficient (Gaussian) | `c = 0.173767` | `s2_two_parameter_fit.json` |
| Dirichlet box coefficient (2d) | `A = 4.8866` | `s2_two_parameter_fit.json` |
| worst family deviation (4 families) | `3.6e-4` | `s2d_family_fits.json` |
| mass-axis Richardson vs direct `m=0` | dev `1.8e-5` (`p = 1.66`) | `s2c_mass_axis_fit.json` |
| 2d extremal-state purity `max|ν−½|` | `6.7e-10` | `s3_extremal_summary.csv` |
| 2d extremal-state optimality gap | `1.07e-4` relative | `s3_extremal_summary.csv` |
| **4d, Gaussian sampling (continuum, no lattice)** | **`C_4d/C_FE = 0.4574904`, grid unc. `2e−6`** | `s6b_momentum_4d_worldline.json` |
| **4d, Gaussian: CERTIFIED enclosure (S8)** | **`0.457490709365 ≤ C_4d/C_FE ≤ 1`** (lower endpoint a computer-assisted theorem; upper endpoint = Fewster–Eveson Eq. (40), cited) | `s8_certified_enclosure.json` |
| S8 gap, certified lower endpoint to the archived momentum value | `2.46e-08` = `1/81` of that route's stated `2e-06` | `s8_certified_enclosure.json` |
| S8 Galerkin ladder, `n_basis = 16 … 64` | gap `1.52e-04 → 2.46e-08`, monotone | `s8_certified_enclosure.json` |
| S8 formulation check: Γ-series elements vs 2D quadrature of the `qei` kernel | `8.4e-13` / `8.7e-13` (`K` / `K_-`) | `s8_certified_enclosure.json` |
| S8 mutation `K_- × (1+1e-4)` (`n_basis = 40`) | `0.457628840` — **excludes** the archived value | `s8_certified_enclosure.json` |
| S8 proved trace-norm bound `‖K_-‖₁` | `≤ 3√π e²/32 = 1.22782` (measured `1.21135`) | `s8_certified_enclosure.json` |
| S8 Powers–Størmer upper estimate — **NOT CERTIFIED** | `R ≤ 0.576051` (float64, `n = 400` Nyström, no remainder control) | `s8_certified_enclosure.json` |
| S8 rigorous-but-weak upper bound from `‖K_-‖₁` alone | `R ≤ 3.6945` (weaker than Fewster–Eveson's `1`) | `s8_certified_enclosure.json` |
| 4d, Gaussian, lattice route with the derived `ρ`-form (4 continuum-free fits) | `0.457466 ± 0.000053` (`± 0.000131` conservative) | `s6c_rho_expansion.json` |
| lattice-vs-continuum agreement (honest, `d₂`-pinned fits excluded) | `2.4e-05` = `0.46σ` | `s6c_rho_expansion.json` |
| derived `ρ²` coefficient | `d₂ = -0.24932` = `-0.28464` + `0.03532` | `s6c_rho_expansion.json` |
| lattice error exponent at fixed `ρ` | `(a/R)²` (slopes `1.82–1.86`) | `s6c_rho_expansion.json` |
| 4d sup over `f` (6 families) | `≥ 0.6110` (compact bump `(1−u²)²`); Gaussian not optimal | `s6d_families_4d.json` |
| 4d `f`-dependence, min → max | `0.2833` → `0.6110` (factor `2.16`) | `s6d_families_4d.json` |
| Ford–Roman weight, three routes: continuum / lattice / exhibited state | `0.05976` / `0.05968` / `0.05976` (IR bias closed) | `s6g_lorentzian_routes.json` |
| FFR-2012 `y_∞` by **their method on our kernel** | `0.0236173` vs `0.0236175` (Table I/II reproduced) | `s6f_ffr_method.json` |
| Ford–Roman weight, **exhibited state** (rigorous) | `x₀ ≥ 0.05976` = `y_∞ + 1e−5` + `0.0361` | `s6e_ffr2012_comparison.json` |
| Ford–Roman weight vs FFR-2012 `y_∞ = 0.02361 ± 1e−5` | `x₀ = 0.05976` (ratio `2.53`); their Eq. (64), FE, FR all respected | `s6e_ffr2012_comparison.json` |
| Ford–Roman weight vs **Schiappacasse–Fewster–Ford 2018** `C_shift = −0.0593338` (finite cavity) | `x₀ = 0.059762`, **+0.72 %** — a **reproduction** of their value (corrected 2026-09-23 from "a finding") | `s6g_lorentzian_routes.json`; `test_ffr2012_lorentzian_acceptance` (vi) |
| same weight, lattice vs continuum at `ρ = ¼` (box law re-measured, `p = 2.50`) | `-2.0e-04` | `s6e_ffr2012_comparison.json` |
| 2d control on the continuum route, 7 families | worst `|r−1| = 1e-04` | `s6a_momentum_2d_control.json` |
| **4d conjecture (archived lattice scan)** | **`C_4d/C_FE = 0.4591 ± 0.0027` (0.58%)** | `s4e_extrapolation.json` |
| 4d, order 1 / order 2 | `0.45795` / `0.46019` | `s4e_extrapolation.json` |
| 4d `C_ρ/C_FE`, `ρ = ½ … ⅛` | `0.4025, 0.4210, 0.4317, 0.4429, 0.4512, 0.4541` | `s4e_extrapolation.json` |
| 4d box law | `B = −10.04 ± 0.30` in `B/g⁴` | `s4d_box_sweep.json` |
| archived per-sector floor → new | `2.5e-8` (cone) / `1.7e-13` (full) → `4.6e-13` worst, super-exponential tail | `s4a_noise_floor_revisited.json` |
| mp cross-check, `l = 2,3,4` | `6.4e-14, 2.6e-14, 8.7e-15` abs (limit `9e-14`) | `s4b_mp_crosscheck.json` |
| 4d extremal state, `(ρ, n) = (¼, 6)` | purity `≤ 4e-13`, gaps `0` / `7e-6` (`l=0`), `r_max = 1.13` | `s4f_extremal_4d_summary.json` |
| notebook wall time, S0–S5 / S6 | `342.7 s` / `844 s` | `summary.json` |

## Status

- **2d methods result: ready to draft.** Anchored, shape-independent,
  artifact-modelled, extremal states archived. Unchanged by this pass.
- **4d successor: the two items that were open are closed.** (i) The `R/τ`
  expansion is **derived** (even powers, `d₂ = -0.24932` in closed form),
  which excludes two of the three competing models and retires the `0.7%`
  spread; the constant for Gaussian sampling is `0.4574904 × C_FE` from a
  lattice-free continuum solution, cross-checked by the lattice route at
  `0.457466 ± 0.000053` (the four fits that import no continuum value),
  the two agreeing to `2.4e-05` = `0.46` of the lattice error; the lattice
  budget is now dominated by the `ρ`-form choice with `d₂` free, as it should be. (ii) A second, third … sixth sampling
  family was run, and the answer changes the claim: `C_sharp(f)` is **not**
  family-independent in 4d (factor `2.16` across families), so the candidate
  is a conjecture for `C_4d(Gaussian)` plus a *lower bound* `0.6110 × C_FE` on
  the sampling-independent sharp constant. Still open: the sup over all
  admissible `f` — the `(1−u²)^p` trend points at the `W^{2,2}` regularity
  boundary and no finite family search settles it — and — **partially closed this pass (S8)** — the
  status of the continuum limit as a proof. It is no longer purely numerical:
  the analytic formulation is derived and matched to the code, and the **lower**
  half of an enclosure is a computer-assisted theorem,
  `C_4d(Gaussian)/C_FE ≥ 0.457490709365` (`2.46e-08` below the archived value).
  The **upper** half is still open — the only rigorous upper bound is
  Fewster–Eveson's `R ≤ 1` — so the enclosure is `[0.4574907, 1]` and the
  archived `0.4574904 ± 2e-06` is still, on its upper side, a numerical claim.
