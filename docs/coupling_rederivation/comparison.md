# Comparison: blind re-derivation vs `vacuum/detectors/circuit_mapping.py`

Written 2026-09-04 **after** `derivation.md` (number committed there, section 7) and only then
reading `vacuum/detectors/circuit_mapping.py` (module docstring steps 1–7 and code), the
"The coupling — derived" section of `papers/circuit-qed-noise-thresholds/MANIFEST.md`, and the two
`derived` coupling entries of the parameter file. Nothing else of the existing work was read
(not the notebook, data, STATUS, README rows or git log).

## Verdict

**Agree — on every equation, every conversion factor, the treatment of every ambiguity the
existing derivation lists, and the number.** Both routes give, for TB26 Table I scenario 2 in the
loader's gap-unit convention ($\omega_{\rm ref} = \Omega^0$, $\Omega_{\rm lat} = 1$):

$$\lambda_{\rm osc}^{\rm xp} = \lambda_{\rm osc}^{\rm xx} = \sqrt2\,|\lambda_{\rm TB}| = 0.14142 .$$

No error was found in the existing derivation. The differences are in what each route *adds*
and in how the uncertainty is packaged; none changes $\lambda_{\rm osc}$ or the device verdict.

## Step-by-step

| step | existing (`circuit_mapping.py`) | this re-derivation | same? |
|---|---|---|---|
| line Hamiltonian, mode expansion | TB26 Eqs. (3)–(5), (7); $N_k^2 = \hbar Z_0/4\pi\|k\|$ from $[\Phi,q]$ | Eqs. (3)–(5), (7); same normalization read off the $\hbar=v=1$ form of $H_{\rm TL}$ | yes |
| circuit → line coupling | Eqs. (16)–(19), (22): $-(\varphi_0/\ell_0)\hat\gamma_5\partial_x\Phi_C$; current coupling; renormalization term dropped | Eqs. (16)–(22); same; J23 App. B sign noted as unobservable | yes |
| two-level / transversal / VGSD | Eqs. (27)–(32): $\gamma_x\in[-0.02,0.22]$, $\Omega(t)=\Omega^0+\Delta\Omega\chi$, Eq. (31) | same | yes |
| paper's $\lambda$ | Eqs. (36), (37), (66): $\lambda=-\gamma\sqrt{R_K/8\pi Z_0} = -4.53\gamma$; $\lambda^2=\pi\alpha$ (33) | same; $4.5322$ from SI-exact $R_K$ | yes |
| physical anchor of the normalization | J23 Eqs. (6), (10): $\Gamma_1 = \lambda^2\Delta = \pi\alpha\Delta$ | same, **plus** FD17 Supp. S7 ($\Gamma_1/\Delta = R_Q\|\varphi_\beta\|^2/2\pi Z_0$, $R_Q=R_K/4$) and TB26 App. A (A6) — three separately derived statements of the same $R_K/8\pi Z_0$ | yes (one more source) |
| $Z_0$, $v$ | absorbed in Eq. (66); nothing survives | same; $v$ shown to cancel explicitly (test) | yes |
| lattice spacing vs field amplitude | canonical 1+1 field is scale-invariant; spacing enters only via the derivative → $\Omega_{\rm lat}$ | same; **plus** the explicit identity $H_{\rm TL} = (\hbar v/a)\,H_{\rm lat}$ with $x_i = \hat\phi(x_i)$, $p_i = (a/v)\partial_t\hat\phi$, and a numerical pin $\tfrac12(K^{1/2})_{0r} = 2/\pi(1-4r^2)\to-1/2\pi r^2$ (no stray factor of $a$, 2 or $\pi$) | yes |
| energy unit | $\omega_{\rm ref}=\Omega^0$ (loader convention), $\Omega_{\rm lat}=1$ | left open (A3): gap units **or** cutoff units $a = v/\Omega_{\rm cut}=0.38$ mm, TB26's own suggested $\Delta x$; unit-free statement $\lambda_{\rm osc}^2/2\Omega_{\rm lat} = \lambda_{\rm TB}^2$ | same physics; different emphasis (see "what this route adds", 3) |
| two-level → oscillator (MAP-1) | $\lambda_{\rm osc} = \lambda_{\rm UDW}\sqrt{2\Omega_{\rm lat}}$ from $x_d(t) = \mu(t)/\sqrt{2\Omega}$ under $a\leftrightarrow\sigma^-$ (Gate A) | same factor; **plus** a direct dynamical test: exact qubit⊗Fock vs the repo's `attach_detectors` Gaussian evolution agree at $O(\lambda^2)$ for `'xx'` and `'xp'`, residual $\propto\lambda^2$, and the naive $\lambda_{\rm osc}=\lambda_{\rm TB}$ fails by exactly $1/2\Omega$ | yes |
| `'xp'` transcription | exact: $\partial_t\partial_{t'}W = \partial_x\partial_{x'}W$ for the massless field; lattice caveat $(1-m^2/\Omega^2)^{1/2}=0.980$ | exact, by the chiral decomposition (all orders for the Gaussian model, not only the second-order matrix elements); lattice identity $DV_{xx}D^{\rm T}=V_{pp}$ at $m=0$ pinned numerically; same mass caveat | yes (stronger argument) |
| `'xx'` map | equal Golden-rule rate at the free gap to the transcribed `'xp'` model: $M = \Omega_{\rm lat}\sqrt{2\Omega_{\rm lat}}$; declared a matching, not a transcription; alternatives `finite-difference` ×0.980, `physical-rate` ×0.924, shifted gap ×0.907/×0.845 | $\lambda_{\rm xx} = \Omega_{\rm lat}\lambda_{\rm xp}$ from equal on-resonance spectral weight — the same formula; declared not an equivalence; only the shifted-gap alternative considered (I did not consider the lattice-DOS variants because in cutoff units $\Omega_{\rm lat}=0.146$ sits where the lattice is linear) | yes |
| sign of $\lambda$ | unobservable | unobservable | yes |
| scenario-2 value | $-0.10$ (Table I, Figs. 8/18 captions) with readings $-0.0906$ (Eq. 66) and $-0.0971$ ($\sqrt{\pi\alpha}$) | $-0.10$ (Table I) with the Eq.-(66) reading $-0.0906$; I read $\alpha=0.003$ as a rounding of $0.1^2/\pi = 0.0032$, i.e. not an independent third reading | value same; spread packaged differently |
| uncertainty | $\sigma$ = sample std of the three readings, $0.0048$ (4.8 %) → $\lambda_{\rm osc} = 0.1414\pm0.0068$; mapping band $[0.1185, 0.1482]$; MANIFEST itself says the band, not $\pm1\sigma$, is the interval that matters | one-sided rounding spread $-9.4\%$ ($0.128$–$0.141$), gap choice $-3.5\%$, and a **device-level** $Z_0$ systematic $\mp5\%$ (G08 measured $59.7\ \Omega$ on a nominal-50 design; FD17 "close to nominal") that the existing map propagates as exactly 0 because no file carries an `error` on $Z_0$ | same conclusion (no published error bars); see discrepancy 1 |

## Discrepancies (none changes $\lambda_{\rm osc}$)

1. **$Z_0$ systematic.** The existing $\sigma$ is the paper's rounding spread only;
   `sigma_propagated_from_file_errors = 0`. For a *device* statement I would carry
   $Z_0 = 50\ ^{+10}_{-5}\ \Omega$ ($\lambda\propto Z_0^{-1/2}$: $-8.5\%/+5\%$). Numerically the existing
   mapping band $[0.1185, 0.1482]$ already contains my full range $[0.128, 0.149]$, so the device
   point was in effect run across it; the label "rounding spread" on the $\pm1\sigma$ is just narrower
   than a device error bar, as the MANIFEST already warns. Recommendation: add an `error` of
   $\sim5\ \Omega$ to `characteristic_impedance` (the code already propagates it) so the $\pm1\sigma$ becomes
   a device-level number.
2. **Third "reading" $\sqrt{\pi\alpha}$.** Treating Table I's $\alpha = 0.003$ as an independent reading
   of $\lambda$ is a choice; it is consistent with $\lambda=-0.1$ to the printed precision, so I would not
   count it. Effect: the sample std would be the two-reading spread instead (same order). Immaterial.
3. **Energy unit and the cutoff.** Both routes agree the coupling is scale-free and that in gap units
   $\Omega_{\rm lat}=1$. The existing map notes that the lattice band edge $\omega_{\max}=2$ (14.6 GHz)
   replaces the device's 50 GHz exponential cutoff ("model limitation 1"). Two consequences worth
   stating explicitly (they bear on fidelity, not on $\lambda_{\rm osc}$): (i) in gap units the detector sits
   at half the lattice band, where the dispersion is already nonlinear — group velocity
   $\cos(\kappa/2) = 0.866$ at $\Omega_{\rm lat}=1$, i.e. signals at the qubit frequency travel 13 % slower
   than $v$, and the lattice DOS is 15 % above the continuum (the existing `physical-rate` ×0.924 is this
   same effect); (ii) in cutoff units ($a = 0.38$ mm, $\Omega_{\rm lat}=0.146$) both effects are $<0.4\%$ and
   the 50 GHz cutoff can be imposed as a smearing $F_i = aF_{\rm eff}$ (TB26 Eq. 13, $\sum F_i = 1$). The
   derivative coupling is UV-weighted ($S\propto\omega$), and TB26 App. C shows a 20 % change in $N$
   between $\Omega_{\rm cut}/2\pi = 50$ GHz and $\infty$ for cosine ramps and the loss of spacelike harvesting
   for the trapezoid — so the *cutoff representation* is a larger lever on the verdict than any
   coupling ambiguity. This is already declared as a limitation; I flag it because the coupling
   derivation is now settled and this is where the remaining model risk sits.
4. **Equation numbering.** The existing module read the PDF; I assigned numbers by counting in the
   LaTeX source and checked them against the parameter file's anchors. Every equation number cited by
   both agrees (3, 4, 5, 7, 16–19, 22, 27–33, 36, 37, 66; A6 = 33).

## Effect on the device verdict

None qualitatively: the number is identical, and the uncertainty range I derive lies inside the
band the device point was already run across. The one addition that could matter for a device
statement is a $Z_0$ error bar (∓5 %), which stays inside that band.

## What this independent route adds

1. A third published normalization (FD17 Supp. S7, 2017) giving the same $R_K/8\pi Z_0$, so Eq. (66)
   is anchored by TB26 App. A, J23 App. B and FD17 independently.
2. An all-orders argument (chiral decomposition of the massless 1+1 vacuum) that the `'xp'`
   transcription of $\partial_x$ is exact for the Gaussian oscillator model, plus the exact lattice
   identity $DV_{xx}D^{\rm T}=V_{pp}$ at $m=0$ (`test_momentum_equals_discrete_derivative_on_massless_chain`).
3. A numerical pin of the lattice field's normalization against the continuum
   $\langle\partial_x\phi\,\partial_x\phi\rangle = -1/2\pi x^2$, excluding any stray factor
   (`test_lattice_field_is_the_continuum_dimensionless_field`).
4. A direct dynamical verification of MAP-1 through the repo's own `attach_detectors` for both
   operators, including the falsification of the naive identification
   (`test_two_level_to_oscillator_factor_is_sqrt_2Omega`).
5. The explicit statement that $v$ cancels (`test_eq66_velocity_cancels`, `test_waveguide_speed_cancels`)
   and that $\lambda_{\rm osc}$ depends on the lattice spacing only through $\Omega_{\rm lat}$, with the
   cutoff-unit value $0.054$ (`'xp'`) / $0.0079$ (`'xx'`) tabulated alongside the gap-unit $0.141$.
6. An independent, script-level reproduction from transcribed inputs only
   (`rederive.py`, refusing `derived` entries), pinned in `tests/test_coupling_rederivation.py`
   (14 tests).
