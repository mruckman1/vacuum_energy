## 6.5.9 Diagnostics A — the reviewer's plan, executed (`../notebook_qei_exact.py --diagnostics`, `data/s5f*`–`s5i*`)

*Status: a diagnostics record, not a result section. It settles what the §6.5.8 drift is made of and retracts one deduction of §6.5.8; it does not produce a converged λ = 4 number. Every row is one evaluation of the same pipeline as §6.5 (L = 16, m = 0.5, τ₀ = 0.75, worldline x = 8) unless marked ED. All wall times were measured on a machine shared with another run (load 12–16× the core count for the first hour); idle-machine times are ≈ 3–5× shorter.*

An outside reviewer read §6.5.8 and proposed, in this order: (1) the n_max ladder *on the infimum* at λ = 4; (2) decomposing the λ = 0 calibration band into its sources and driving it to ≤ 0.3 %; (3) a denser λ sweep with the small-λ exponent read against the repository's perturbation theory; and (4), per the anomaly protocol, a second implementation — a dense exact-diagonalization (ED) infimum at small L checked against the MPO–DMRG route. The premise behind (1) was that the Fock cutoff is itself a suppression mechanism (the extremal state is more excited than the vacuum, so the truncation raises the infimum in the direction of the residual) and that Rayleigh–Ritz makes the ladder monotone and hence interpretable. The four items were run as parallel subprocesses of the notebook (`--diagnostics`; every evaluation is kept in `data/jobs/`), with one new module, `vacuum/interacting/qei_ed.py` (the ED route), one new pure-Gaussian function, `qei_exact.first_order_infimum_slope` (the O(λ) anchor), and one new convergence knob, `svd_dense_limit` (exact vs sketched SVD in the operator TEBD).

### A.1 The n_max ladder on the infimum at λ = 4 (`data/s5f_nmax_ladder_lam4.csv`)

Archived knobs held fixed (χ_op = 16, weight 0.25, χ_mpo ≤ 64, dt = 0.375 order 4, χ_DMRG = 12); only n_max moves.

| n_max | E_min | exact/Hartree | step | ‖h(T)‖/‖h(0)‖ | trunc_op (w) | ⟨Ω\|O_f\|Ω⟩ − ‖f‖²⟨Ω\|h\|Ω⟩ | wall (shared machine) |
|---|---|---|---|---|---|---|---|
| 6 | −6.566e-03 | 0.8729 | — | 0.846 | 1.2e-04 | 1.1e-03 | 213 s |
| 7 | −6.809e-03 | 0.9053 | +0.032 | 0.836 | 6.1e-05 | 1.9e-03 | 327 s |
| 8 | −7.177e-03 | 0.9542 | +0.049 | 0.820 | 3.6e-05 | 6.2e-03 | 538 s |
| 9 | −7.514e-03 | **0.9990** | +0.045 | 0.804 | 2.7e-05 | 7.3e-03 | 690 s |
| 10 | −7.414e-03 | 0.9857 | −0.013 | 0.792 | 2.1e-05 | 5.7e-03 | 898 s |
| 11 | −7.341e-03 | 0.9760 | −0.010 | 0.761 | 1.7e-05 | 3.9e-03 | 1228 s |
| 12 | −7.469e-03 | 0.9931 | +0.017 | 0.725 | 1.9e-05 | 4.2e-03 | 1689 s |

(E_Hartree = −7.5215e-03 throughout.) Three readings.

1. **The ladder is not monotone on this route.** E_min falls through n_max = 9 and then rises at 10 and 11; the ratio climbs 0.873 → 0.999 and recedes to 0.976 before returning to 0.993. Nothing here can be extrapolated, and the archived "≥ 0.905, still rising" is superseded by "0.87–1.00 depending on the rung".
2. **The operator truncation grows with n_max at fixed χ_op.** The weighted discarded weight `trunc_op` is small and shrinking, but the *unweighted* norm retained by h(T) falls from 85 % to 72 %: each added Fock level adds high-occupation content that χ_op = 16 at weight 0.25 throws away. The reference inconsistency — the MPO's ⟨Ω|O_f|Ω⟩ against the exact ‖f‖²⟨Ω|h|Ω⟩, which the subtraction cancels only to the extent the discarded operator acts alike on Ω and on the extremal state — reaches 6–7e-03, i.e. the size of E_min itself (−7e-03). A ladder in n_max at fixed χ_op is therefore a ladder along a *diagonal* of the (n_max, χ_op) plane, and the two axes cannot be read separately from it. This is the reviewer's mechanism with the sign left open: the Fock cutoff does not simply raise the infimum here, because raising n_max also changes what the operator truncation discards.
3. **The premise "Rayleigh–Ritz makes the ladder monotone" is false for the smeared infimum, independently of any MPO.** Only the t = 0 operator h_x is a projection P h P (its λ_min *is* monotone in n_max — the dense ladder of §6.5.8 and `tests/test_interacting_qei_exact.py` stand). The smeared operator of the truncated chain is ∫f² e^{iH_trunc t} h e^{−iH_trunc t} dt with H_trunc = P H P, and e^{−iH_trunc t} is not the projection of e^{−iHt}: the truncated chain is a different dynamical system whose infimum is bounded by the exact one on neither side. The ED route (A.4) exhibits this at λ = 0 with no MPO anywhere: at L = 4, m = 1, τ₀ = 0.5 the relative deviation of the truncated chain's infimum from the exact Gaussian value is +1.9e-3, −2.0e-3, −6.6e-3, +5.9e-4, −1.7e-3, +9.7e-5 for n_max = 4…9 — converging, sign-alternating, with the extremal parity sector flipping along the way (`tests/test_interacting_qei_ed.py::test_fock_ladder_of_the_smeared_infimum_is_not_rayleigh_ritz`). **This retracts the deduction of §6.5.8 item 1**: the non-monotone λ = 0 control (0.9993 → 1.0127 → 1.0088) was attributed to the MPO truncation because the untruncated ratio "must" be monotone; it need not be, and the same wobble appears with no MPO at all. The operational lesson of §6.5.8 — a plateau is not convergence — stands; the attribution does not.

### A.2 The calibration band, decomposed (`data/s5g_calibration.csv`)

The reference is the exact Gaussian infimum −2.2653650e-02; "rel dev" is (E_min − E_Gauss)/|E_Gauss| (negative: the route sits *below* the true infimum). One knob moves per row; the base is the archived row (n_max 6, χ_op 16, weight 0.25, χ_mpo ≤ 64, dt 0.375, χ_DMRG 12: −1.27e-02).

| axis | row (one knob changed from the base) | rel dev vs Gaussian, λ = 0 | shift vs base | trunc_op (w) | ‖h(T)‖/‖h(0)‖ | wall (shared) |
|---|---|---|---|---|---|---|
| base | `lam0_base` | -1.27e-02 | +0.00e+00 | 1.9e-04 | 0.913 | 405 s |
| Fock cutoff (χ_op 16, w 0.25) | `lam0_fock_nmax7` | -8.81e-03 | +3.88e-03 | 1.3e-04 | 0.906 | 452 s |
| Fock cutoff (χ_op 16, w 0.25) | `lam0_fock_nmax8` | -8.73e-03 | +3.96e-03 | 8.7e-05 | 0.901 | 428 s |
| Fock cutoff (χ_op 16, w 0.25) | `lam0_fock_nmax9` | -1.27e-02 | +8.13e-05 | 6.4e-05 | 0.897 | 516 s |
| Fock cutoff (χ_op 16, w 0.25) | `lam0_fock_nmax10` | -1.50e-02 | -2.26e-03 | 5.3e-05 | 0.894 | 904 s |
| Fock cutoff (χ_op 16, w 0.25) | `lam0_fock_nmax11` | -1.54e-02 | -2.61e-03 | 4.7e-05 | 0.891 | 1950 s |
| Fock cutoff (χ_op 16, w 0.25) | `lam0_fock_nmax12` | -1.28e-02 | -8.74e-05 | 3.9e-05 | 0.891 | 2499 s |
| operator truncation χ_op × weight, n_max 6 | `lam0_chiop16_wd0.5` | -6.47e-03 | +6.19e-03 | 3.6e-03 | 0.929 | 438 s |
| operator truncation χ_op × weight, n_max 6 | `lam0_chiop16_wd1` | -3.21e-02 | -1.91e-02 | 8.6e-02 | 0.958 | 424 s |
| operator truncation χ_op × weight, n_max 6 | `lam0_chiop32_wd0.25` | -1.55e-02 | -2.73e-03 | 1.8e-04 | 0.913 | 534 s |
| operator truncation χ_op × weight, n_max 6 | `lam0_chiop32_wd0.5` | -1.17e-02 | +1.06e-03 | 3.5e-03 | 0.928 | 593 s |
| operator truncation χ_op × weight, n_max 6 | `lam0_chiop32_wd1` | -3.60e-02 | -2.29e-02 | 7.0e-02 | 0.966 | 746 s |
| operator truncation χ_op × weight, n_max 6 | `lam0_chiop64_wd0.25` | -1.29e-02 | -1.55e-04 | 1.8e-04 | 0.913 | 1693 s |
| operator truncation χ_op × weight, n_max 6 | `lam0_chiop64_wd0.5` | -1.50e-02 | -2.25e-03 | 3.5e-03 | 0.929 | 1648 s |
| operator truncation χ_op × weight, n_max 6 | `lam0_chiop64_wd1` | -2.65e-02 | -1.36e-02 | 5.2e-02 | 0.975 | 2785 s |
| exact dense SVD instead of the sketch | `lam0_exactsvd_chiop16_wd0.25` | -7.66e-03 | +5.02e-03 | 1.8e-04 | 0.912 | 732 s |
| exact dense SVD instead of the sketch | `lam0_exactsvd_chiop16_wd0.5` | -9.34e-03 | +3.36e-03 | 3.4e-03 | 0.928 | 768 s |
| exact dense SVD instead of the sketch | `lam0_exactsvd_chiop32_wd0.5` | -2.25e-02 | -9.67e-03 | 2.6e-03 | 0.945 | 4686 s |
| exact dense SVD instead of the sketch | `lam0_exactsvd_chiop32_wd1` | -3.62e-02 | -2.31e-02 | 6.9e-02 | 0.966 | 4678 s |
| final MPO cap / accumulator | `lam0_chimpo_uncapped` | not run (budget) | | | | |
| final MPO cap / accumulator | `lam0_chiacc64` | -1.48e-02 | -2.01e-03 | 1.9e-04 | 0.913 | 393 s |
| Trotter | `lam0_dt0.1875` | -1.10e-02 | +1.73e-03 | 1.3e-04 | 0.918 | 398 s |
| Trotter | `lam0_dt0.125` | -1.18e-02 | +8.95e-04 | 1.0e-04 | 0.921 | 335 s |
| Trotter | `lam0_order2` | -1.15e-02 | +1.22e-03 | 2.0e-04 | 0.914 | 324 s |
| DMRG | `lam0_chidmrg24` | not run (budget) | | | | |
| DMRG | `lam0_sweeps16` | not run (budget) | | | | |
| DMRG | `lam0_chigs32` | not run (budget) | | | | |
| light-cone window | `lam0_pad4` | not run (budget) | | | | |
| combined, n_max 8 | `lam0_best_nmax8_chiop32_wd1` | -8.18e-02 | -6.82e-02 | 7.9e-02 | 0.962 | 3367 s |
| combined, n_max 8 | `lam0_best_nmax8_chiop64_wd0.5` | -1.57e-03 | +1.10e-02 | 1.5e-03 | 0.914 | 4594 s |
| sweep settings (n_max 8, χ_op 32, w 0.5, χ_DMRG 16) | `sweep_lam0` | -1.92e-03 | +1.07e-02 | 1.5e-03 | 0.913 | 2973 s |

The same knobs at λ = 4 (n_max 6 unless stated; no exact reference — the shift *is* the band there):

| row | exact/Hartree | shift vs base (rel. E_min) | trunc_op (w) | ‖h(T)‖/‖h(0)‖ | ⟨Ω\|O_f\|Ω⟩ inconsistency | wall |
|---|---|---|---|---|---|---|
| `lam4_base` | 0.8729 | +0.00e+00 | 1.2e-04 | 0.846 | 1.1e-03 | 412 s |
| `lam4_chiop32_wd0.5` | 0.8990 | -2.98e-02 | 5.5e-03 | 0.886 | 6.6e-05 | 659 s |
| `lam4_chiop64_wd0.5` | 0.9415 | -7.86e-02 | 5.2e-03 | 0.885 | 5.2e-04 | 1827 s |
| `lam4_chiop32_wd1` | 1.0421 | -1.94e-01 | 1.6e-01 | 0.924 | 1.4e-03 | 751 s |
| `lam4_chiop64_wd1` | 1.0052 | -1.51e-01 | 1.4e-01 | 0.933 | 4.2e-03 | 2845 s |
| `lam4_exactsvd_chiop32_wd0.5` | 0.9587 | -9.83e-02 | 4.4e-03 | 0.897 | 4.6e-04 | 4607 s |
| `lam4_dt0.1875` | 0.8822 | -1.06e-02 | 9.3e-05 | 0.858 | 1.0e-03 | 475 s |
| `lam4_chidmrg24` | not run (budget) | | | | | |
| `lam4_pad4` | not run (budget) | | | | | |
| sweep settings (n_max 8, χ_op 32, w 0.5) | 0.8834 | -1.19e-02 | 2.0e-03 | 0.859 | 1.7e-04 | 3219 s |

Reading, axis by axis (rows marked "not run" did not fit the budget on the shared machine; the pattern is set by the rows that did):

- **Fock cutoff at fixed χ_op.** n_max 6 → 12 at (χ_op 16, w 0.25) leaves the λ = 0 error at −0.9 to −1.5 % with no convergence — the same non-monotone wobble as the archived 5/6/7 control, now over seven rungs. Raising n_max alone does not buy anything, because (A.1, reading 2) each level adds operator content that the fixed χ_op discards.
- **Operator truncation, χ_op × weight, at n_max 6.** At weight 0.25 the error is χ_op-*independent* (−1.27 / −1.55 / −1.29 % at 16 / 32 / 64); at weight 0.5 it does not improve either (−1.17 / −1.50 % at 32 / 64); at weight 1 (plain Frobenius) it is −2.7 to −3.6 % with 5–7 % of the operator's weight discarded. So at n_max 6 the ≈ 1.3 % band is **not** the operator truncation at all: it is the truncated chain's own infimum sitting below the exact one — the Fock-dynamics effect of A.1 reading 3, whose sign is not fixed and which the ED route shows at the same size (−0.66 % at L = 4, n_max 6). The archived attribution of the band to the MPO truncation (§6.5.3–6.5.4) was therefore right about the sign convention and wrong about the source at n_max 6.
- **Sketched vs exact SVD.** The operator TEBD truncates blocks larger than 600 by a randomized sketch (χ + 16 columns, one power iteration). Forcing the exact dense SVD moves the λ = 0 number by +5.0e-03 (χ_op 16, w 0.25), −2.9e-03 (16, 0.5), −1.1e-02 (32, 0.5) and −2e-04 (32, 1) relative, and at λ = 4 it moves exact/Hartree from 0.899 to **0.959** (χ_op 32, w 0.5). The sketch is therefore a knob at the level of the band at λ = 0 and at the level of the *residual* at λ = 4 — the direction in which the truncation is resolved changes the λ = 4 answer by 6 %. It is exposed as `svd_dense_limit` and must be treated as a convergence axis with χ_op and the weight.
- **Trotter.** dt 0.375 → 0.1875 → 0.125 moves λ = 0 by +1.7e-03 and +0.9e-03 relative (order 4; the two shifts are not ordered because each dt is a different truncation trajectory, so ≈ 1e-03 is the Trotter *plus* sketch floor at this dt). At λ = 4 the same halving moves E_min by −1.1e-02 relative (0.8729 → 0.8822): the Trotter error is ten times larger there, as the φ⁴ bond terms make the splitting worse, and it is *not* negligible against the residual being measured — §6.5.3's "negligible" holds at λ = 0 only.
- **Order 2 / accumulator.** Order-2 Trotter at the same dt: +1.2e-03; doubling the accumulator bond dimension: −2.0e-03. Both at the level of the sketch noise above.
- **DMRG, window, final cap.** Not re-run (budget); the archived development runs (cap uncapped: 2.5e-04; χ_DMRG truncation 1e-07 in every row here) put them below the band.
- **Driving it down.** At n_max 8 with weight 0.5 the picture changes: (χ_op 32) −1.9e-03 and (χ_op 64) −1.6e-03 — converged in χ_op at ≈ **0.16–0.19 %**, the reviewer's ≤ 0.3 % reached; plain Frobenius at the same n_max (χ_op 32) is −8.2e-02 and unusable. Cost: 3000–4600 s per evaluation on the shared machine (≈ 12–18 min idle, ≈ 14× the archived per-point cost), and — the decisive caveat — **the same settings at λ = 4 are not converged**: exact/Hartree there reads 0.883 (n_max 8, χ_op 32, w 0.5), 0.899 / 0.942 (n_max 6, χ_op 32 / 64, w 0.5), 0.954 (n_max 8, χ_op 16, w 0.25), 1.042 / 1.005 (n_max 6, χ_op 32 / 64, w 1). The band that was driven to 0.2 % at λ = 0 is ≥ 10 % at λ = 4.

### A.3 The seven-point λ sweep and the O(λ) anchor (`data/s5h_lambda_sweep.csv`, `.json`)

Settings chosen from A.2 to put the λ = 0 control inside 0.3 %: n_max = 8, χ_op = 32 (χ_acc 64, χ_mpo ≤ 128), weight 0.5, sketched SVD, dt = 0.375 order 4, χ_DMRG = 16. Each point cost 2970–3260 s on the shared machine (≈ 12–15 min idle; ≈ 14× the archived per-point cost). R ≡ 1 − exact/Hartree; "R_ctrl" divides the ratio by the λ = 0 control first. The last column is the exact first-order prediction (below).

| λ | E_min | E_Hartree | exact/Hartree | exact/free | R | R_ctrl | 0.0809 λ (exact O(λ)) | trunc_op (w) | ‖h(T)‖/‖h(0)‖ |
|---|---|---|---|---|---|---|---|---|---|
| 0 | −2.26972e-02 | −2.26537e-02 | **1.0019** | 1.0019 | −0.0019 | 0 | 0 | 1.5e-03 | 0.913 |
| 0.25 | −2.01667e-02 | −2.04054e-02 | 0.9883 | 0.890 | 0.0117 | 0.0136 | 0.0202 | 1.2e-03 | 0.910 |
| 0.5 | −1.83393e-02 | −1.85782e-02 | 0.9871 | 0.810 | 0.0129 | 0.0148 | 0.0405 | 9.2e-04 | 0.908 |
| 1 | −1.53577e-02 | −1.57307e-02 | 0.9763 | 0.678 | 0.0237 | 0.0256 | 0.0809 | 8.9e-04 | 0.898 |
| 2 | −1.15782e-02 | −1.18728e-02 | 0.9752 | 0.511 | 0.0248 | 0.0267 | 0.162 | 1.2e-03 | 0.880 |
| 3 | −8.38659e-03 | −9.33078e-03 | 0.8988 | 0.370 | 0.1012 | 0.1029 | 0.243 | 1.5e-03 | 0.864 |
| 4 | −6.64434e-03 | −7.52151e-03 | 0.8834 | 0.293 | 0.1166 | 0.1183 | 0.324 | 2.0e-03 | 0.859 |

**Band.** The λ = 0 control reads 1.0019: a band of **0.19 %**, the reviewer's ≤ 0.3 % reached (A.2 says what it cost). The free bound survives at every λ (exact/free ≤ 1, → 0.29 at λ = 4), now with a 0.2 % instrument.

**Exponent.** A power law R_ctrl = a λ^p fitted on λ ≤ 1 gives p = 0.46 (0.30–0.62 across the extreme placements of the end points inside the band). But the fit has no footing: the residual is a *staircase* — R_ctrl(0.25) ≈ R_ctrl(0.5) ≈ 1.4 %, R_ctrl(1) ≈ R_ctrl(2) ≈ 2.6 %, R_ctrl(3) ≈ R_ctrl(4) ≈ 10–12 % — with each tread flat to within a band and each riser several bands high. That is the signature of a quantity set by the cutoff, not by a smooth function of λ: A.1 and A.2 show the λ = 4 value moving by ±0.05 with n_max at fixed χ_op and by +0.04 to +0.17 with χ_op at fixed n_max, while the same knobs move λ = 0 by < 0.4 %. The λ = 0 band does **not** transfer to λ ≥ 2 (§6.5.8 already said so; A.2 quantifies it), so R(3) and R(4) here are settings-dependent numbers, not measurements of the residual.

**The O(λ) anchor.** The repository's perturbation theory (`phi4.py`) is for the ground energy and the equal-time covariance; there was no first-order theory of the *infimum*. `qei_exact.first_order_infimum_slope` supplies it exactly: by Hellmann–Feynman, dλ_min(O_f)/dλ at 0 is the expectation of the Duhamel derivative v_x(t) + i∫₀ᵗ[V(s), h_x(t)]ds in the free extremal state ψ* (the Gaussian vacuum of O_f that `qei_minimize` exhibits), and in a Gaussian state Wick's theorem closes both terms: ⟨v_x(t)⟩ = ⅛⟨φ_x(t)²⟩², and i⟨[φ_i(s)⁴/24, h_x(t)]⟩ = 6⟨φ_i(s)²⟩·i⟨[φ_i(s)²/24, h_x(t)]⟩ — the mean-field identity [φ⁴, q] → 6⟨φ²⟩[φ², q] holds exactly for commutators with a quadratic form. The reference ⟨Ω_λ|O_f|Ω_λ⟩ = ‖f‖²⟨h_x⟩_λ is differentiated through the O(λ) tadpole covariance (exact to first order, `phi4.py`) plus the direct ⅛X². The formula is validated against ED finite differences at L = 4, n_max = 9 to 4e-4 (`test_first_order_slope_matches_finite_differences`). At the paper point:

    dE_min/dλ = +1.1930e-02 = quartic +3.622e-02 + dynamical +7.481e-03 − reference +3.177e-02
    dE_Hartree/dλ = +1.0097e-02,   dR/dλ = (dE_H − dE_min)/E_free = **+0.0809**

and 0.089 / 0.082 / 0.081 at L = 6 / 7 / 8: L-independent to 1 % beyond L = 7. **The residual beyond mass renormalization is first order in λ, with a positive slope of 8 % per unit λ.** Mechanism: every O(λ) piece is mean-field, but with the *state's own* ⟨φ²⟩ — the extremal state's along its trajectory — whereas the Hartree mass uses the vacuum's; the difference is the O(λ) "quartic tax" the reviewer named as one of the two hypotheses. The other hypothesis ("Hartree absorbs the O(λ) mean-field piece, so a pure backreaction residual emerges at O(λ²)") is not what the exact first-order calculation gives.

**Anchoring the small-λ end.** The measured R_ctrl lies *below* the linear prediction: 1.36 vs 2.02 % at λ = 0.25 (a deficit of 0.66 %, 3.5 bands) and 1.48 vs 4.05 % at λ = 0.5 (2.6 %). Read as a second-order coefficient, R = 0.0809 λ + c λ², both points give the same c ≈ −0.10 (−0.106 and −0.103) — which would be a consistent anchor — but the flat tread R(0.25) ≈ R(0.5) equally admits a λ-dependent cutoff floor of ≈ 1.4 %; the two readings separate at λ = 0.1 (prediction 0.81 − 0.10 = 0.7 %, 3.5 bands), which was not affordable here. The ED route at L = 6 (A.4) tests the same anchor with no MPO in the way.

### A.4 Second implementation: dense ED at L = 6–8 vs the MPO–DMRG route (`data/s5i_ed_control.csv`, `.json`)

`vacuum.interacting.qei_ed` builds O_f by dense Heisenberg evolution — the full eigendecomposition H = V diag(w) Vᵀ of the truncated chain, in which ⟨a|O_f|b⟩ = h_ab F(w_a − w_b) with F the Fourier transform of f² — so there is no Trotter step, no time grid (an analytic kernel; the MPO route's trapezoid grid is available as a second kernel so that its quadrature can be matched exactly) and no operator truncation. It shares exactly one approximation with the MPO route, the Fock cutoff, through the same `local_boson_ops`, the same basis frequency and the same PSD h_x, so at equal (L, n_max) the two routes differ by the MPO route's Trotter + truncation error and nothing else. Global φ → −φ parity splits the problem into two blocks (both sector minima are reported; the extremal state need not share the vacuum's parity), which is what makes n_max = 5 at L = 6 (23 328 states per sector) affordable. Cost: `eigh` of each sector — single-threaded under Accelerate, ≈ 13 min per sector at 23 k; L = 8 at n_max = 3 (32 768 per sector) was started and abandoned as over budget. The construction is tested against the TeNPy model's bond terms, against the full-matrix (parity-free) operator, and — at L = 5, n_max = 2 — against the MPO route to 4e-9 (`tests/test_interacting_qei_ed.py`).

**ED n_max ladders** (E_min; ratio = exact/Hartree at that L; λ = 0 column is the relative deviation from the exact Gaussian infimum at the same L; sector = parity of the extremal state):

| L | λ | n_max: E_min (exact/Hartree; λ=0: rel dev vs Gaussian) [sector] | E_min non-increasing? | wall |
|---|---|---|---|---|
| 6 | 0 | 1: -1.88221e-03 (+9.1e-01) [0]; 2: -1.52682e-02 (+2.3e-01) [0]; 3: -1.83318e-02 (+7.9e-02) [1]; 4: -2.02395e-02 (-1.7e-02) [0] | True | 0, 0, 32, 1177 s |
| 6 | 0.1 | 3: -1.75305e-02 (0.9169) [1]; 4: -1.93284e-02 (1.0109) [0] | True | 31, 1164 s |
| 6 | 0.25 | 3: -1.63974e-02 (0.9073) [1]; 4: -1.80961e-02 (1.0013) [0] | True | 31, 1174 s |
| 6 | 0.5 | 3: -1.46776e-02 (0.8859) [1]; 4: -1.63321e-02 (0.9858) [0] | True | 33, 1179 s |
| 6 | 1 | 4: -1.35857e-02 (0.9581) [0] | None | 1170 s |
| 6 | 4 | 1: -5.34867e-04 (0.0764) [0]; 2: -3.30852e-03 (0.4729) [0]; 3: -3.44966e-03 (0.4931) [1]; 4: -5.97167e-03 (0.8535) [0] | True | 0, 0, 31, 1311 s |
| 7 | 0 | 2: -1.68720e-02 (+2.1e-01) [0]; 3: -1.98877e-02 (+6.7e-02) [0] | True | 6, 1188 s |
| 7 | 4 | 2: -3.91489e-03 (0.5299) [0]; 3: -3.86018e-03 (0.5225) [0] | False | 6, 1480 s |
| 8 | 0 | 2: -1.72397e-02 (+2.1e-01) [0] | None | 505 s |
| 8 | 4 | 2: -3.94073e-03 (0.5293) [0] | None | 318 s |

**MPO route vs ED at the same (L, n_max, λ)** (same Fock cutoff, same basis, same h_x; 'same quadrature' compares against the ED kernel on the MPO route's own trapezoid grid, so it isolates Trotter + operator truncation + DMRG):

| L | n_max | λ | MPO knobs | E_min MPO | E_min ED | rel dev | rel dev, same quadrature | quadrature-only (ED trapezoid vs analytic) | trunc_op (w) | wall MPO / ED |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 2 | 0 | archived | -1.529184e-02 | -1.526820e-02 | -1.5e-03 | -1.5e-03 | -4.2e-07 | 1.2e-03 | 8 / 0 s |
| 6 | 2 | 0 | generous | -1.526937e-02 | -1.526820e-02 | -7.6e-05 | -7.6e-05 | -4.2e-07 | 6.2e-05 | 96 / 0 s |
| 6 | 3 | 0 | archived | -1.831516e-02 | -1.833178e-02 | +9.1e-04 | +9.1e-04 | -4.2e-06 | 1.3e-03 | 40 / 32 s |
| 6 | 3 | 0 | generous | -1.833530e-02 | -1.833178e-02 | -1.9e-04 | -1.9e-04 | -4.2e-06 | 5.1e-03 | 607 / 32 s |
| 6 | 4 | 0 | archived | -2.036669e-02 | -2.023947e-02 | -6.3e-03 | -6.3e-03 | -9.5e-06 | 5.7e-04 | 99 / 1177 s |
| 6 | 2 | 4 | archived | -3.320607e-03 | -3.308521e-03 | -3.7e-03 | -3.7e-03 | -2.4e-06 | 2.1e-03 | 8 / 0 s |
| 6 | 2 | 4 | generous | -3.308961e-03 | -3.308521e-03 | -1.3e-04 | -1.3e-04 | -2.4e-06 | 2.8e-04 | 103 / 0 s |
| 6 | 3 | 4 | archived | -3.341910e-03 | -3.449657e-03 | +3.1e-02 | +3.1e-02 | -5.9e-06 | 2.1e-03 | 39 / 31 s |
| 6 | 3 | 4 | generous | -3.450047e-03 | -3.449657e-03 | -1.1e-04 | -1.1e-04 | -5.9e-06 | 1.8e-02 | 545 / 31 s |
| 6 | 4 | 4 | archived | -5.760927e-03 | -5.971675e-03 | +3.5e-02 | +3.5e-02 | -5.9e-05 | 1.1e-03 | 111 / 1311 s |
| 7 | 3 | 0 | archived | -1.987798e-02 | -1.988771e-02 | +4.9e-04 | +4.9e-04 | -4.4e-06 | 1.4e-03 | 67 / 1188 s |
| 7 | 3 | 4 | archived | -3.755397e-03 | -3.860183e-03 | +2.7e-02 | +2.7e-02 | -5.6e-06 | 2.2e-03 | 73 / 1480 s |
| 8 | 2 | 0 | archived | -1.727539e-02 | -1.723970e-02 | -2.1e-03 | -2.1e-03 | -5.6e-07 | 1.4e-03 | 21 / 505 s |
| 8 | 2 | 4 | archived | -3.958388e-03 | -3.940727e-03 | -4.5e-03 | -4.5e-03 | -2.6e-06 | 2.3e-03 | 23 / 318 s |


Reading.

1. **Where the MPO route is converged it agrees with the second implementation; where it is not, the ED locates the error and its sign.** With generous operator knobs (χ_op 64, plain norm, exact SVD, no window, χ_DMRG 32) the MPO infimum reproduces the ED one to ≤ 1.9e-04 relative at every (L, n_max, λ) run with them — L = 6, n_max = 2 and 3, λ = 0 and 4 (four rows); the generous rows at L = 6 n_max 4–5, L = 7 and L = 8 were queued and not reached before the budget stop, so at L = 6 n_max 4, L = 7 n_max 3 and L = 8 n_max 2 only archived-knob comparisons exist (0.05–3.5 % off) and no ED agreement is claimed there. The two implementations share only the definition of h_x and the Fock basis. With the *archived* knobs (χ_op 16, w 0.25) the MPO route sits 1.5–6.3e-03 **below** the ED infimum at λ = 0 (the manufacturing direction) and 2.7–3.5e-02 **above** it at λ = 4, n_max ≥ 3 (L = 6 and 7): at λ = 4 the archived operator truncation *raises* the infimum, i.e. inflates the apparent residual — the same direction and size as the λ = 4 drift of §6.5.8 and of A.2 (χ_op 16 → 64 at w 0.5: 0.873 → 0.942). The archived λ = 4 residual is, at least in part, this truncation.

*S8 addendum (fix round 1, 2026-09-12; **corrected in fix round 2, same day**; `data/s8_ed_anchor.json`).* The comparison now exists at the **sweep's own knobs** (χ_op 32, weight 0.5, sketched SVD — the settings every quoted ratio of §6.5.8 and of the S8 harvest is taken at) and three rungs higher, against the matrix-free Lanczos ED route on the same trapezoid grid. At λ = 0 the MPO route reproduces the exact infimum to +3.36e-03 (L = 6, n_max 5), +6.36e-03 (L = 5, n_max 6) and +1.38e-02 (L = 5, n_max 8). At λ = 4, on the same three points, the deviations are +3.37e-02, −1.19e-01 and **+3.94e-01** — they do not settle, they **grow with n_max at fixed χ_op**, and the largest sits at n_max = 8, the sweep's own Fock cutoff. The plane's knobs (χ_op 32, weight 0.5, *exact* SVD) at that point read −3.89e-03 (λ = 0) and **+4.23e-01** (λ = 4): resolving the truncation exactly makes the λ = 4 disagreement larger, not smaller. In absolute terms the MPO reads E_min −1.306075e-02 (sweep) and −1.245110e-02 (plane) against the exact −2.156113e-02 — exact/E_H 1.978 and 1.885 against 3.265 — so at n_max 8, λ = 4 the MPO route understates the magnitude of the infimum by a factor 1.6508 (sweep knobs) and 1.7317 (plane knobs). The generous knobs (χ_op 64, weight 1, exact SVD, no window) at L = 6, n_max 5 read −1.65e-03 (λ = 0) and −1.52e-02 (λ = 4), so at *that* rung raising χ_op with an exact SVD cuts the λ = 4 error by about half, not by a decade; no generous-knob pair exists above n_max 5, so nothing is claimed about whether χ_op 64 survives at n_max 8. Two caveats on the exact side: the L = 5, n_max 6, λ = 4 ED rung's dt 0.375 quadrature is loose (the dt 0.125 re-solve moves it +1.32e-03, against ≤ 5.8e-05 elsewhere) and the n_max 8 rung's moves +4.96e-04; and the L = 5 λ = 4 exact ladder is itself still falling 38.6 % per rung at n_max 8, so these pairs compare two un-converged routes rather than measuring an error against a known answer. Trotter and quadrature are ≤ 6e-05 everywhere on the MPO side (the "quadrature-only" column): the entire MPO–ED difference is operator truncation. **What is established** is therefore the negative statement, and the earlier wording of this addendum ("a 3–12 % effect wherever an exact answer exists") is **retracted** (MANIFEST no-longer-claims item 8): at λ = 4 the operator truncation at the sweep's and the plane's knobs is a tens-of-per-cent effect at the cutoff the paper uses, an order of magnitude or more above its λ = 0 level at the same points (a factor 29 apart at L = 5, n_max 8), and it is not bounded by anything measured here.


*S8 addendum, part 2 (fix round 2, 2026-09-12): which knob costs the λ = 4 accuracy, and what the reference-consistency gate is worth.* At the one point where an exact λ = 4 answer exists at the paper's own Fock cutoff (L = 5, n_max 8, τ₀ = 0.75; exact E_min = −2.156113e-02), seven MPO rows walk one knob at a time off the sweep's settings (weight 0.5, sketched SVD, χ_DMRG 16). Deviations from the exact answer: χ_op 32 **+39.42 %**, 48 **+14.88 %**, 64 **+13.670 %**, 96 **−4.50 %**; χ_DMRG 32 at χ_op 64 **+13.669 %**; and weight_decay 1 at χ_op 64 **−1.19 %**. Three things follow. (i) The DMRG bond is already converged — doubling it changes the answer by 1.1e-5 relative. (ii) The **weighting**, not the operator bond dimension, is what costs the accuracy: turning it off at χ_op 64 is worth a factor 11. (iii) The χ_op axis helps but is non-monotone and changes sign between χ_op 64 and 96, so no setting in this ladder approximates the exact infimum from a known side (the MPO minimizes a *truncated* operator, which is not bounded below by the exact one — reading 5 of §6.5.9 — so a row below the exact answer is a truncation artefact, not an anomaly event). Most consequential for the QEI magnitude at λ ≥ 2: the reference-consistency diagnostic |E_ref − E_ref,static| / |E_min|, which the S8 harvest uses as a 5 % usability gate, is **anti-correlated** with the true error across these rows — 6.73 / 33.90 / 2.66 / 1.21 / 1.21 / 6.56 / 36.06 % against errors +39.42 / +42.25 / +14.88 / +13.670 / +13.669 / −4.50 / −1.19 %. The row with the best reference consistency (1.21 %, comfortably inside the gate) is 13.7 % wrong, and the most accurate row (1.19 %) misses the gate by a factor seven. **Passing the gate certifies nothing about a λ ≥ 2 magnitude**, and no such magnitude is quoted from any row in this paper on the strength of it. At λ = 0 the same χ_op 64 knobs are +1.21 % off exact, a factor 11 better than at λ = 4, which is the same λ-dependence the first addendum reports. Caveat: this is one (L, n_max, τ₀) point; that the knob ranking carries to L = 16 is an assumption, not a measurement.
2. **The ED n_max ladders are not monotone either, and the sign of the Fock error flips at the paper's own smearing.** At L = 6, λ = 0 the truncated chain's infimum lies *above* the Gaussian one by 91 %, 23 % and 7.9 % at n_max = 1, 2, 3 and **below** it by 1.7 % at n_max = 4 — the crossing of A.1 reading 3, now at τ₀ = 0.75, m = 0.5. At λ = 4, L = 7 the ladder *rises* from n_max 2 to 3 (E_min −3.915e-03 → −3.860e-03). At L = 6, λ = 4 the ratio climbs 0.08 → 0.47 → 0.49 → 0.85 through n_max 4 with no sign of settling; the n_max = 5 rungs (23 328 states per sector) were started and killed unfinished after 2.7 h of single-threaded `eigh` on the shared machine, so the reviewer's "generous n_max" stops at 4 here — at which the Fock error is still 1.7 % at λ = 0 and evidently much larger at λ = 4. **On neither route is the ladder a Rayleigh–Ritz sequence, and on neither route has n_max ≤ 12 (MPO, L = 16) or ≤ 4 (ED, L = 6) converged the λ = 4 infimum.**
3. **The small-λ anchor, with no MPO in the way.** At L = 6 the exact first-order slope is dR/dλ = +0.0891. The ED residual, control-normalized at each n_max, is 0.49 / 1.53 / 3.85 % (n_max 3) and 0.63 / 1.57 / 3.10 % (n_max 4) at λ = 0.1 / 0.25 / 0.5, against 0.89 / 2.23 / 4.46 % predicted; at λ = 1 and 4 (n_max 4): 5.8 % and 16 % against 8.9 % and 36 %. So the second implementation confirms a first-order residual of the predicted sign and order — at λ = 0.25 the measured/predicted ratio is 0.70 on the ED route (L = 6, n_max 4) and 0.67 on the MPO route (L = 16, n_max 8) — but at n_max 3–4 the Fock error (which changes sign between those rungs) is itself a few per cent of E_min and moves the small-λ residual by 20–30 %, so the O(λ²) coefficient cannot be read from it; n_max ≥ 5 at L = 6 (≈ 30 min per λ point idle) is the rung that would.

### A.5 Verdict

**Does the residual move with truncation (not physics) or hold (physics)?** Both, in different λ ranges, and the diagnostics say where the line is.

1. **At λ ≥ 2 it moves — by more than its own size.** exact/Hartree at λ = 4 spans 0.87–1.04 across the settings tested while the λ = 0 control spans 0.998–1.036; the n_max ladder at fixed χ_op climbs to 0.999 and recedes; χ_op at fixed n_max adds +0.03 to +0.17; resolving the truncation by an exact instead of a sketched SVD adds +0.06; halving dt adds +0.01; the operator's unweighted norm loss is 11–28 %; and the MPO's own reference ⟨Ω|O_f|Ω⟩ is inconsistent with the exact one by the size of E_min. None of the λ = 4 numbers of this paper is converged, and the archived "lower bound ≥ 0.905, residual ≤ ~10 %" must be replaced by: **at λ = 4 the residual beyond mass renormalization is somewhere between 0 and ≈ 12 % and this instrument cannot currently say where.** The λ = 0 calibration, even at 0.2 %, does not transfer (§6.5.8 said so; A.2 shows it is a factor of ≥ 50).
2. **At λ ≤ 1 a residual is resolved from zero — and it is required.** With the band at 0.19 %, R_ctrl = 1.4 % (λ = 0.25), 1.5 % (0.5) and 2.6 % (1) are 7–13 bands from zero; and the exact first-order calculation gives dR/dλ = +0.081, L-independent — the Hartree mass, built from the vacuum's ⟨φ²⟩, cannot absorb the O(λ) shift of an extremal state whose own ⟨φ²⟩ differs. **The existence of a residual beyond mass renormalization is physics, first order in λ, with sign and slope fixed exactly.** Its measured values lie below the linear prediction (by 0.7 % at λ = 0.25, 2.6 % at 0.5), consistent with an O(λ²) coefficient ≈ −0.10 from both points, but the staircase shape of R(λ) leaves a λ-dependent cutoff floor equally admissible; λ = 0.1 at the present band is the discriminating point. The 7–13 bands measure distance from zero at the sweep's settings, not convergence in n_max: the ladder is non-monotone at every λ, including λ ≤ 1, and the ED anchor at L = 6 moves the small-λ residual by 20–30 % between n_max 3 and 4.
3. **The archived λ = 0 deduction is retracted.** The non-monotone control was not evidence of MPO trouble: the smeared infimum of the truncated chain is not a Rayleigh–Ritz quantity, and the ED route shows the same sign-alternating wobble with no MPO at all (A.1 reading 3; `tests/test_interacting_qei_ed.py`). Likewise, the ≈ 1.3 % band at n_max 6 was the Fock dynamics, not the operator truncation (A.2) — and the ED at L = 6 shows the same crossing at the paper's smearing (+7.9 % at n_max 3, −1.7 % at n_max 4).
4. **The second implementation agrees with the first where the first is converged, and locates its error where it is not.** With generous operator knobs the MPO route reproduces the ED infimum to ≤ 1.9e-04 at every (L, n_max, λ) tested with them (L = 6, n_max ≤ 3); with the archived knobs it sits 1.5–6.3e-03 below it at λ = 0 and 2.7–3.5e-02 *above* it at λ = 4 — the same order as the λ = 4 drift of §6.5.8, and in the direction of inflating the apparent residual (A.4). No anomaly-protocol event: nothing beats an exact infimum by more than its stated error.

What closing item 1 would take is now specific: the (n_max, χ_op, weight) plane must be walked with the λ = 4 reference inconsistency and the unweighted norm loss driven to the λ = 0 level (they are 5e-03 and 15–28 % now; 3e-04 and 9 % at λ = 0 in the sweep), i.e. n_max ≥ 10 with χ_op ≥ 64 at weight 0.5, or an operator representation that does not spend its bond dimension on high-occupation matrix elements at all; at the measured scaling that is 5–10 h per λ point on this machine. The small-λ end, by contrast, is one λ = 0.1 point at the present settings away from a decisive comparison with the exact slope.
