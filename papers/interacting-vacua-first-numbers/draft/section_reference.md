# 6.6 The sharpest free reference, and what the extremal state is made of (`../notebook_qei_reference.py`, `data/s7*`)

> **Supersession banner (diagnostics A, `section_diagnostics_A.md`; the retraction list is §6.7 of `outline.md`).**
> Every number in this section is at the archived knobs (n_max 6, χ_op 16, weight 0.25).
> Diagnostics A found that at λ = 4 those knobs are not converged and *inflate* the
> apparent residual (the MPO infimum sits 2.7–3.5 % above the dense-ED one at L = 6–7),
> and that the infimum is not monotone in n_max (0.873 … 0.999 … 0.993 for n_max 6 … 12).
> Consequently: (i) the premise quoted in §6.6.1 — "exact/E_H ≥ 0.905, residual ≤ ~10 %,
> not resolved from zero" — is retracted (§6.7 item 7); (ii) the "≥ 0.866" of §6.6.3 is
> **not a bound** and the "~14–17 %" is a value at the archived knobs — against K_eff the
> λ = 4 residual is **bracketed, not resolved** — [−28.541 %, +18.800 %] against K_eff
> across the `plane2` settings of the frozen 15-row `unit2_snapshot`, a 239-band spread
> carried by the occupation weighting
> (`s8_convergence_plane2.json`; 0.835 maps to 0.913 / 0.956 / 0.950 at
> n_max 8 / 9 / 12 with the same knobs, 0.901–0.917 with better ones); (iii) the
> decomposition's +10 % quartic fraction (§6.6.4) is likewise a value at those knobs —
> its sign is what is carried forward, and it agrees with the exact first-order theory
> (dR/dλ = +0.081, quartic piece positive); (iv) the τ₀ exponent's ± 0.8 (§6.6.5) does
> not include the archived-knob systematic. What this section establishes independently
> of the knobs stands: the sharp quasiparticle, the nearest-neighbour band, the gap²
> 3.3 % below Hartree, the cutoff-free factor E_free(K_eff)/E_H = 1.0452, the sign of
> the quartic term, and the L = 16 → 24 convergence.

### 6.6.1 The objection

§6.5 measures `exact/E_H`, the exact lattice infimum against the free chain's
Williamson infimum at the self-consistent Hartree mass. That reference matches
the interacting theory at one point of the band — the gap — and nowhere else.
Interactions renormalize the whole single-particle dispersion ω(k), the free
infimum depends on all of ω(k), and a free chain matched at k = 0 has a
different mode structure across the band from one matched everywhere. So the
≤ ~10 % residual of §6.5.8 is defined against a reference that need not be the
sharpest quadratic one. This section measures ω(k), builds the free chain
whose ω(k) *is* the interacting one, measures the infimum against it, and
decomposes the extremal state into the four pieces of the energy density so
that the sign of the quartic term is read, not deduced. Every number below is
at the archived cutoff of §6.5.5 (L = 16, m = 0.5, n_max = 6, χ_op = 16,
weight_decay = 0.25) unless stated; the regenerated infimum reproduces the
archived `E_min` to every printed digit at λ = 0, 1, 4
(`s7b_exact_regenerated.csv`), and the extremal MPS and vacua are now
archived (`s7_extremal_mps_lam*.npz`, `s7_vacuum_mps_lam*.npz`).

### 6.6.2 The interacting dispersion (`data/s7a_dispersion.csv`, `.json`)

**Method.** `C(j, t) = ⟨Ω|φ_j(t) φ_x(0)|Ω⟩` (x = 8 the sampled site) is
obtained by TEBD of `φ_x|Ω⟩` under the φ⁴ Hamiltonian (order-4
Suzuki–Trotter, dt = 0.1, χ = 32, T = 10; TEBD discarded weight ≤ 3.6e-6) and
projected on the standing waves of the Dirichlet chain,
`v_n(j) ∝ sin(k_n (j+1))`, `k_n = nπ/17` — the exact normal modes of every free
Dirichlet chain, so at λ = 0 each projected signal is one harmonic
`e^{−iω_n t}`. Each of the 16 projected signals is harmonically inverted by the
matrix-pencil method (`qei_reference.harmonic_inversion`): complex poles
`z_p = e^{−i(ω_p − iγ_p)dt}` with amplitudes `A_p`. The quasiparticle at `k_n`
is the dominant pole; its **spectral-weight fraction** `|A_dom| / Σ_p |A_p|`,
its decay rate `γ`, and the FWHM of a windowed Fourier transform relative to
the window's own response are the sharpness measures (on a finite chain the
spectrum is discrete: "sharp" means one pole carries the weight, it does not
decay over the trajectory, and the line is resolution-limited). The lowest
frequency is cross-checked against an independent excited-state DMRG
(`excitation_gap_dmrg`; the model is shifted so both energies are negative).

**Resolution, measured.** At λ = 0 the pipeline returns the exact free
dispersion on all 16 modes to `max |ω − ω_free| = 1.7e-6`, unit weight
(1 − 2e-16), `|γ|/ω ≤ 1e-6`, and the DMRG gap to 4.5e-7: the combined
Trotter + truncation + Fock-cutoff error of the measurement is ~2e-6 in ω.

| λ | gap (DMRG) | ω(k₁) (pencil) | dev | weight min over k | max |γ|/gap | FWHM/window max | max |ω − ω_H| | rms |ω − ω_H| |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.5329692 | 0.5329687 | −4.5e-7 | 1.0000 | 1.0e-6 | 1.0013 | 1.7e-6 | 9.7e-7 |
| 1 | 0.6880309 | 0.6881926 | +1.6e-4 | 0.9995 | 3.4e-4 | 1.0013 | 3.4e-3 | 1.3e-3 |
| 4 | 0.9553616 | 0.9556331 | +2.7e-4 | 0.9986 | 1.0e-4 | 1.0013 | 1.5e-2 | 9.1e-3 |

**Verdict on the quasiparticle.** At λ = 4 a sharp one-particle peak exists at
every one of the 16 momenta, including the five modes above the two-particle
threshold 2 × 0.956 = 1.91: the dominant pole carries ≥ 99.86 % of the
projected spectral weight at every k, its decay rate is ≤ 1e-4 of the gap
(γT ≤ 1e-3 over the trajectory), and every line is resolution-limited
(FWHM within 0.13 % of the window's own). "The mass" is well defined; the
framing of §6.5 does not need restating on that account.

**The band.** At λ = 4 the interacting dispersion (n = 1…16) is
`[0.9556, 1.0070, 1.0855, 1.1834, 1.2935, 1.4095, 1.5265, 1.6409, 1.7497,
1.8503, 1.9411, 2.0202, 2.0865, 2.1391, 2.1772, 2.2003]` against the Hartree
curve `[0.9707, 1.0209, 1.0978, 1.1941, 1.3028, 1.4177, 1.5340, 1.6479, 1.7563,
1.8569, 1.9475, 2.0268, 2.0933, 2.1462, 2.1845, 2.2078]`: **below** Hartree at
every k, by −0.0151 at the gap, −0.0064 mid-band, −0.0075 at the zone edge.
Its cosine series is nearest-neighbour: the unconstrained coefficients are
`μ² = 0.87939`, `J₁ = 0.99899`, `|J_{r≥2}| ≤ 5.5e-5` (the noise floor of the
embedded modes), and the non-negative fit over r ≤ 4 gives
`μ_eff² = 0.87907`, `J₁ = 0.99900`, `J₂ = 0`, `J₃ = 1.9e-6`, `J₄ = 8.6e-6` with
a maximum misfit of 7.5e-5 in ω. So at this coupling "the whole dispersion is
renormalized" reduces to two numbers: a **mass shift** and a 0.1 %
hopping renormalization. The interacting gap² is 3.3 % *below* the Hartree
one (0.879 vs `μ_H² = 0.909`) — the second-order self-energy is negative, the
Hartree mass is slightly too heavy — and that, not the band shape, is what
separates the two references. (λ = 1: `μ_eff² = 0.4395` vs `μ_H² = 0.4446`,
`J₁ = 0.99991`, misfit 1.0e-4.)

### 6.6.3 K_eff and the free infimum at K_eff (`data/s7b_reference.csv`)

The inverse (sine) Fourier transform of ω(k)² on the chain's own lattice is
`K_eff = Σ_n ω_n² v_n v_nᵀ`; written as the cosine series
`ω²(k) = μ² + Σ_r J_r (2 − 2 cos rk)` it is `K_eff = μ² I + Σ_r J_r M_r` with
`M_r` the range-r Dirichlet Laplacian (`M₁` is the chain's own), positive
definite, admitting a PSD *local* energy density whenever `μ², J_r ≥ 0`: the
half/half bond split of `vacuum.inequalities.qei` generalized to range-r
bonds, odd-image wall bonds wholly on their site (`keff_density_terms`;
`Σ_x h_x = diag(K_eff, I)` exactly and every `h_x` PSD, tested). The couplings
are the non-negative fit above, so the density is PSD by construction.
`E_free(K_eff)` is then the same Williamson problem as `E_H` with the same f,
lattice, site, cone and quadrature — only the pullback S(t) and the density
split are K_eff's. The pipeline reproduces `E_free_exact` at `(m², 1)` and
`E_H` at `(μ_H², 1)` to 1e-15 (tested, and the `E_hartree_by_keff_pipeline`
column), so whatever separates `E_free(K_eff)` from `E_H` comes from ω(k).

| λ | τ₀ | E_min | E_H | E_free(K_eff) | K_eff/H | exact/E_H | **exact/E_free(K_eff)** | λ = 0 control | exact/E_H norm. | **exact/K_eff norm.** |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.75 | −1.51008e-02 | −1.57307e-02 | −1.58725e-02 | 1.0090 | 0.9600 | **0.9514** | 1.0127 | 0.948 | **0.939** |
| 4 | 0.75 | −6.56588e-03 | −7.52151e-03 | −7.86135e-03 | 1.0452 | 0.8729 | **0.8352** | 1.0127 | 0.862 | **0.825** |

The sharpest quadratic reference is a *lighter* free chain than Hartree, its
infimum is 4.5 % deeper at λ = 4 (0.9 % at λ = 1), and the ratio falls, not
rises: 0.8729 → 0.8352 at the reported cutoff, 0.862 → 0.825 after the λ = 0
normalization of §6.5.8. `E_free(K_eff)/E_H = 1.0452` is a pure Gaussian
number with no MPO cutoff in it, so every entry of the §6.5.8 ladder maps by
the same factor: the archived "best-cutoff lower bound `≥ 0.905`" (n_max = 7) maps to
`≥ 0.866`, its normalized value ≈ 0.897 to ≈ 0.858. **The residual does
not dissolve into dispersion renormalization; against the sharpest reference
it is ~14–17 % at the cutoffs run, and it is defined against a reference that
is now non-Gaussian by construction.** *[Superseded — see the banner: "≥ 0.866" is
not a bound (the ladder is not monotone and the archived knobs inflate the residual),
and "14–17 %" is a value at those knobs; the cutoff-free statement that survives is
the factor 1.0452 and "larger against K_eff than against Hartree".]*

### 6.6.4 The decomposition on the extremal state (`data/s7c_decomposition.csv`)

`h_x = ½π² + (∇φ)²-bonds + ½m²φ² + (λ/4!)φ⁴` is split into its four pieces
(`energy_pieces_dense`; their sum is `local_energy_operator_dense` exactly).
Two independent routes: **(H)** each piece is evolved by the same
operator-space TEBD as `O_f` into its own MPO `O_a` and
`E_a = ⟨ψ*|O_a|ψ*⟩ − ⟨Ω|O_a|Ω⟩`; **(S)** the four pieces are measured along one
Schrödinger-picture TEBD trajectory of ψ* and of Ω (χ = 24, same dt), where
`Σ_a E_a` equals the Schrödinger re-measurement of §6.5.5 identically. All
pieces are normal-ordered against the interacting vacuum.

| λ | E_min | picture | kinetic ½π² | gradient | mass ½m²φ² | **quartic λφ⁴/4!** | Σ pieces | Σ − E_min (rel) | quartic / |E_min| |
|---|---|---|---|---|---|---|---|---|---|
| 0 | −2.29422e-02 | H | −1.7401e-02 | −1.0326e-02 | +4.772e-03 | 0 | −2.29550e-02 | −5.6e-4 | 0 |
| 0 | | S | −1.7366e-02 | −1.0330e-02 | +4.791e-03 | 0 | −2.29048e-02 | +1.6e-3 | 0 |
| 1 | −1.51008e-02 | H | −1.0729e-02 | −7.938e-03 | +1.704e-03 | **+1.318e-03** | −1.56446e-02 | −3.6e-2 | +8.7 % |
| 1 | | S | −1.0505e-02 | −7.733e-03 | +1.711e-03 | **+1.332e-03** | −1.51951e-02 | −6.2e-3 | +8.8 % |
| 4 | −6.56588e-03 | H | −3.963e-03 | −4.208e-03 | +2.60e-04 | **+6.35e-04** | −7.27499e-03 | −1.08e-1 | +9.7 % |
| 4 | | S | −3.708e-03 | −3.995e-03 | +2.81e-04 | **+7.04e-04** | −6.71923e-03 | −2.3e-2 | +10.7 % |

**The audit.** In the Heisenberg picture the four pieces sum to the infimum to
−0.06 % at λ = 0, −3.6 % at λ = 1 and −10.8 % at λ = 4: the pieces are
truncated *separately* at χ_op = 16, each truncation manufactures its own
sub-infimum content (the same sign as the λ = 0 anchor's, §6.5.3), and unlike
the total — which is stationary — every piece is first-order sensitive to it.
The audit therefore passes at λ = 0 and fails at λ = 4 at the same 10 % level
as the unresolved cutoff drift of §6.5.8; the Schrödinger pieces, which share
one trajectory and sum exactly to the recheck, deviate from `E_min` by 2.3 %.
Piece by piece the two pictures agree to ≤ 4 % of |E_min| at λ = 4 (kinetic
3.9 %, gradient 3.2 %, mass 0.3 %, quartic 1.0 %), which is the band on the
decomposition. Within it, the reading is unambiguous:

- **The quartic term is positive** at both λ, +8.7–8.8 % of |E_min| at λ = 1
  and +9.7–10.7 % at λ = 4: it *opposes* the negative energy. The reviewer's
  scenario — the extremal state squeezes φ below vacuum at the sampling
  point, making ⟨:φ⁴:⟩ negative and the quartic term "help" — is contradicted
  by measurement: at x the extremal state is **anti-squeezed in φ and
  squeezed in π**, ⟨φ_x²⟩ = 0.3428 vs 0.3319 in Ω (+3.3 %), ⟨π_x²⟩ = 0.7998 vs
  0.8215 (−2.6 %), ⟨φ_x⁴⟩ = 0.3387 vs 0.3181. This is the structure of the free
  Williamson extremal state too (at λ = 0, δ⟨φ_x²⟩ = +0.059, δ⟨π_x²⟩ = −0.056,
  the MPS reproducing the Gaussian δ⟨φ_x²⟩ to 3e-4): the negative smeared
  energy is carried by the kinetic and gradient pieces, the mass and quartic
  pieces pay for it.
- **The quadratic sector alone reaches the K_eff reference.** At λ = 4 the
  three quadratic pieces sum to −7.91e-03 (H) / −7.42e-03 (S), i.e.
  1.006 / 0.944 of `E_free(K_eff) = −7.861e-03`; the quartic piece then
  removes 8–9 % of it. Within the two-picture band, the residual against the
  sharpest quadratic reference *is* the quartic term: the suppression beyond
  quadratic renormalization lives in the non-Gaussian sector, with the sign
  that suppresses.

*On the archived extremal state.* Its content outside the sampled light
cone is arbitrary: there `O_f` is the identity up to the 1e-6 compression
residue, those directions are flat, and the DMRG sweeps leave O(1)
excitations at the cone edge and the chain ends (δ⟨π²⟩ up to 7 at sites 3–4,
`s7_extremal_mps_lam*.npz`). The eigenvalue and the pieces at x are blind to
it (the operator weight there is ≤ 8e-7 of the centre's, and the Gaussian
tail of f keeps it out of the Schrödinger recheck); a "footprint" of the state
is meaningful only on the inner cone |j − x| ≤ 2, and the boundary question
is settled by the eigenvalue (§6.6.6).

### 6.6.5 The τ₀ sweep (`data/s7d_tau0_sweep.csv`, `s7_summary.json`)

λ = 4 with a λ = 0 control at every τ₀, same cutoff; dt scaled with τ₀ below
0.75 (8 steps), held at 0.375 above it.

| τ₀ | steps | E_min(λ=4) | λ = 0 control | exact/E_H | norm. | residual vs H | E_free(K_eff) | residual vs K_eff (norm.) | quartic / |E_min| | H-audit |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.375 | 8 | −2.3511e-02 | 0.9887 | 0.9216 | 0.932 | 0.068 | −2.5916e-02 | 0.082 | +13.5 % | −1.9e-3 |
| 0.5 | 8 | −1.8011e-02 | 1.0011 | 0.9543 | 0.953 | 0.047 | −1.9324e-02 | 0.069 | +10.4 % | −1.2e-2 |
| 0.75 | 8 | −6.5659e-03 | 1.0127 | 0.8729 | 0.862 | 0.138 | −7.8614e-03 | 0.175 | +9.7 % | −1.1e-1 |
| 1.0 | 11 | −3.3385e-03 | **1.085** | 1.334 | — | — | — | — | — | — |
| 1.5 | 16 | −3.5381e-03 | **1.410** | 19.9 | — | — | — | — | — | — |

The two longest smearings are **not usable at χ_op = 16**: the λ = 0 control,
which must read 1, reads 1.085 and 1.410 (the evolving operator loses 18 %
and 23 % of its norm to truncation, `norm_ratio`), so the MPO route has
failed there and those rows carry no ratio; the fits use τ₀ ≤ 0.75 only.
On the three usable points the normalized residual is 0.068, 0.047, 0.138
(vs H) and 0.082, 0.069, 0.175 (vs K_eff): it *falls* from 0.375 to 0.5 and
*rises* to 0.75 — not a power law; forcing `log r` vs `log(1/τ₀)` through the
three points gives an exponent of **−1.13 ± 1.10** (H) and **−1.17 ± 0.81**
(K_eff), opposite in sign to the predicted **+2** and consistent with zero.
The quartic fraction, |q| = 0.135, 0.104, 0.097, does grow toward small τ₀ but
as `(1/τ₀)^{0.46 ± 0.19}`, a quarter of the predicted power, and with the
*positive* sign throughout. **Interactions do not shift the scaling exponent
of the QEI at this coupling: the residual is a roughly τ₀-independent
fraction of the free bound over the factor-2 range that the method resolves,
and the quartic mechanism the prediction rests on has the wrong sign.** The
range is limited by the truncation, not the physics; extending it to τ₀ ≥ 1
needs χ_op ≥ 32–48 (the next rung of §6.5.8's "cost of closing it").

*S8 addendum (`data/s8_tau0_rows.json`).* Rerun at the sweep's own settings —
n_max 8, weight 0.5, χ_op 32, χ_acc 64, χ_mpo 128, χ_dmrg 16, six knobs away from
the χ_op = 16 rows of the table above (n_max 6, weight 0.25, χ_dmrg 12) — the
λ = 0 control reads 1.036 (τ₀ = 1.0) and 1.253 (τ₀ = 1.5) instead of 1.085 and
1.410. The improvement is therefore *not* a χ_op effect by itself; which knob
carries it is measured separately and reported in `../MANIFEST.md`
(`chiop16_isolation`). The λ = 4 rows are still not usable: their reference inconsistency |E_ref − E_ref_static|
is 38 % and 113 % of their own E_min (1.3–5 % at the resolved λ ≤ 1 points), and
the operator loses 17–20 % of its norm. The exact/E_H they read (1.58 at τ₀ = 1.0,
26.8 at τ₀ = 1.5, where E_H itself is −1.8e-4) are values at their knobs and are not
quoted; the exponent statement stays restricted to τ₀ = 0.375–0.75. The χ_op 48
rows are reported in `../MANIFEST.md` (S8 harvest) as they land.

### 6.6.6 Controls (`data/s7e_controls.csv`)

**L sweep (boundary).** L = 16 → 24 at λ = 4, τ₀ = 0.75: `E_min` −6.5659e-03 →
−6.6357e-03 (−1.1 %), `E_H` unchanged to 4e-6, the λ = 0 control 1.0127 →
1.0081, `exact/E_H` 0.8729 → 0.8822, normalized 0.862 → 0.875. The
eigenvalue is L-converged at the 1 % level and the residual survives with
margin; the extremal MPS's tail, being flat-direction content (§6.6.4), is not
a support measure (edge footprint 11.5 vs centre 0.033 at L = 16; 0.05 at
L = 24).

**Second lattice spacing.** a = ½ at fixed bare physical parameters
(L = 32, m = 0.25, λ = 1, τ₀ = 1.5 lattice units, 16 steps): the λ = 0
control reads **1.173**, the same failure as τ₀ ≥ 1 above — at χ_op = 16 the
operator TEBD cannot carry the operator over 6 lattice units of time — so no
ratio is quoted from this point (`exact/E_H = 1.55` and the quartic fraction
+8.7 % are recorded, not used). The reviewer's three-spacing control needs
the χ_op step first; its dispersion measurement at this spacing was started
and is reported in `s7e_dispersion_fine.*` if present.

*S8 addendum (fix round 2, 2026-09-12).* The χ_op step has since been taken on
the τ₀ ≥ 1 rows — which are the same failure mode — and it does **not** fix
them. At fixed n_max 8, weight 0.5, χ_dmrg 16 the λ = 0 control runs
1.33774 → 1.25306 → 1.21742 at χ_op 16 → 32 → 48 for τ₀ = 1.5 (monotone toward
1, still 21.7 % off) and 1.03379 → 1.03614 → 1.03843 for τ₀ = 1.0 (monotone
*away* from 1), while the operator's reference inconsistency — the other gate —
is 2.3 % → 117 % → 64 % and 9.8 % → 2.7 % → 7.65 % of |E_min| respectively.
The a = ½ **χ_op 32** pair landed 2026-09-13 and runs that test at this
spacing: it **fails**. The λ = 0 control moves from 1.173 (χ_op 16) to
**1.1865** (χ_op 32) — *away* from 1, the τ₀ = 1.0 pattern — with reference
inconsistency **20.01 %** of |E_min| at λ = 0 and **175.32 %** at λ = 1, both
far above the 5 % gate, so `control_ok` and `usable` are both false and **no
a = ½ ratio is quoted** (the λ = 1 reading would have been 1.71794 normalized
by its control). χ_op alone does not deliver the a = ½ control, exactly as the
τ₀ ≥ 1 evidence predicted. The χ_op 48 pair is still queued
(`data/s8_tau0_rows.json`, `scaling_refit["a_half"]["rows"]`,
`control_chi_op_sequence`; `tests/test_interacting_qei_s8.py::test_a_half_rows_are_pinned_with_their_controls`,
`::test_control_chi_op_sequences_are_pinned`).

### 6.6.7 Verdict

The reviewer's premise — that the reference is wrong, not just the mass — is
tested and does not hold at λ ≤ 4: the interacting dispersion is
nearest-neighbour to 7e-5 with `J₁ = 0.999`, so "renormalizing the whole
dispersion" is a mass shift, and the shift goes the *other* way from what
would rescue the archived claim: the true gap² is 3.3 % below the Hartree
one, the sharpest free reference is 4.5 % deeper than `E_H`, and
`exact/E_free(K_eff) = 0.835` (0.825 normalized) beside `exact/E_H = 0.873`
(0.862) at the reported cutoff, `≥ 0.866` (≈ 0.858 normalized) at the best
cutoff of §6.5.8 *[not a bound — banner]*. **The residual survives against the sharpest quadratic
reference, larger than before by a cutoff-free factor 1.045, and it is
non-Gaussian by construction** *[its existence and first-order character, yes —
§6.5.9 A.3; its size at λ = 4 is BRACKETED and not resolved, [−28.5 %, +18.8 %]
against K_eff across the settings run]*. The decomposition says what it is: the
quadratic pieces of the extremal state reach 0.94–1.01 of the K_eff free
infimum and the quartic piece, **positive** (+9.7–10.7 % of |E_min|, because
the extremal state anti-squeezes φ at the sampling point exactly as the free
Williamson state does), takes it back. The quasiparticle is sharp at every k
(weight ≥ 0.9986, γ/gap ≤ 1e-4), so nothing in §6.5's framing changes on
that account. The τ₀ dependence is flat within the factor-2 range the
method resolves (exponent −1.1 ± 1 against a predicted +2; the quartic
fraction grows only as τ₀^{−0.46}), so the interaction shifts the QEI
*constant*, not its scaling exponent, on the evidence available. What remains
open is unchanged in kind and now better located: the ~10 % cutoff drift of
§6.5.8 is the same size as the quartic term itself, and the H-picture
decomposition audit fails by exactly that amount at λ = 4, so the next rung
(n_max = 8, χ_op ≥ 32) is what turns "the residual is the quartic term" from a
reading within a 6 % band into a number.
