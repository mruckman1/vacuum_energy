# Blind re-derivation of the circuit-QED coupling for the lattice oscillator detector

**Status: written BEFORE reading `vacuum/detectors/circuit_mapping.py`,
`tests/test_circuit_mapping.py`, `papers/circuit-qed-noise-thresholds/`, `docs/STATUS.md`,
the README Layer-2 rows, or any git log.** The number is committed in section 7 of this
file; the comparison with the existing derivation is in `comparison.md`, written afterwards.

Date: 2026-09-04. Sources actually read (all fetched from arXiv / the publishers):

| tag | source |
|---|---|
| **TB26** | A. Teixidó-Bonfill, X. Dai, A. Lupascu, E. Martín-Martínez, arXiv:2505.01516v1 (PRA 113, 043732 (2026)). LaTeX source `main.tex`; printed equation numbers below were assigned by counting numbered equations in the source and validated against the four anchors the parameter file cites (Eqs. 7, 30, 33, 66) and against Eqs. 53 and 70–72. |
| **J23** | N. Janzen, X. Dai, S. Ren, J. Shi, A. Lupascu, Phys. Rev. Research 5, 033155 (2023), arXiv:2208.05571 (LaTeX source). |
| **FD17** | P. Forn-Díaz et al., Nat. Phys. 13, 39 (2017), arXiv:1602.00416 (LaTeX source, incl. Supplementary S7). |
| **G08** | M. Göppl et al., J. Appl. Phys. 104, 113904 (2008), arXiv:0807.4094 (PDF). |
| **BMMM13** | A. Brown, E. Martín-Martínez, N. Menicucci, R. Mann, PRD 87, 084062 (2013), arXiv:1212.1973 (LaTeX source) — the oscillator-detector reference cited by `attach_detectors`. |
| **TBMM24** | A. Teixidó-Bonfill, E. Martín-Martínez, PRD 110, 105016 (2024), arXiv:2406.14637 — the derivative-coupling reference cited for `coupling='xp'`. |
| repo | `docs/API.md`, `vacuum/core/models.py`, the signature/docstring of `attach_detectors` (`vacuum/detectors/nonperturbative.py`), the `udw.py` module docstring, `vacuum/experiments_io.py`, and the **transcribed** entries of `experiments/circuit_qed_harvesting/teixido-bonfill_2026_table1.json` (the four `derived` entries were not read). |

Notation: TB26's flux field is $\hat\Phi$ (dimension Wb), its dimensionless field is $\hat\phi$;
the repo's lattice field is $x_i \equiv \varphi_i$ with conjugate $p_i$, $\hbar = 1$, spacing $a = 1$.

---

## 0. What is being derived

`attach_detectors(K, sites, gaps, lambdas, coupling)` builds, on the unit lattice with
$H_{\rm field} = \tfrac12\sum_i[p_i^2 + m^2 x_i^2 + (x_{i+1}-x_i)^2]$ (`harmonic_chain_K`, API.md),

$$
H_{\rm int}^{\rm xx} = \lambda_{\rm osc}\, x_d \sum_i F_i x_i, \qquad
H_{\rm int}^{\rm xp} = \lambda_{\rm osc}\, x_d \sum_i F_i p_i, \qquad
H_d = \tfrac12\left(p_d^2 + \Omega_d^2 x_d^2\right),
$$

with $\sum_i F_i = 1$ (`profile_matrix`: pointlike = one-hot; `Smearing` normalizes to 1).
The question is: what is $\lambda_{\rm osc}$ (both forms) for the flux-qubit + tunable-coupler +
transmission-line device of J23 as modelled by TB26, and with what uncertainty.

---

## 1. Circuit Hamiltonian → UDW form in the paper's own units

**1.1 Transmission line.** TB26 Eq. (3): $H_{\rm TL} = \tfrac12\int dx\,[(\partial_x\Phi)^2/\ell_0 + q^2/c_0]$,
with $q = c_0\,\partial_t\Phi$ (Eq. 4), $Z_0 = \sqrt{\ell_0/c_0}$, $v = 1/\sqrt{c_0\ell_0}$.
Mode expansion, TB26 Eq. (5):

$$
\hat\Phi(t,x) = \sqrt{\hbar Z_0}\int\frac{dk}{\sqrt{4\pi|k|}}\left(e^{i(\omega_k t - kx)}\hat a_k^\dagger + {\rm H.c.}\right),
\quad [\hat a_k,\hat a_{k'}^\dagger] = \delta(k-k'),\ \omega_k = v|k|,
$$

which gives Eq. (6), $H_{\rm TL} = \tfrac12\int dk\,\hbar\omega_k(a^\dagger a + a a^\dagger)$. Parameters, Eq. (7):
$v \approx 1.2\times10^8$ m/s, $Z_0 \approx 50\ \Omega$ (both "$\approx$"). UV cutoff: Eq. (8)–(9),
$\hat\Phi_{\rm cut}$ carries the weight ${\rm cut}(\omega) = e^{-|\omega|/2\Omega_{\rm cut}}$, $\Omega_{\rm cut}/2\pi = 50$ GHz;
equivalently a smearing $F_{\rm eff}$ with $\int F_{\rm eff}\,dx = {\rm cut}(0) = 1$ (Eqs. 10–13).

**1.2 Coupling of the circuit to the line.** Lumped model (Fig. 5, unnumbered Hamiltonian $\hat H_{\Delta x}$),
change of variables $\Phi_\pm = \Phi_b \pm \Phi_a$, continuum limit, dropping the divergent renormalization term
$\Phi_-^2/(2\ell_0\Delta x)$ and the $O(\Delta x)$ term (Eq. 20–21): 

$$
\hat H_{\rm int} = -\frac{1}{\ell_0}\hat\Phi_-\,\partial_x\hat\Phi(x_{\rm TC+FQ}),\qquad \hat\Phi_- = \varphi_0\hat\gamma_5,\ \varphi_0 = \frac{\hbar}{2e},
$$

i.e. **TB26 Eq. (22)**: $\hat H_{\rm int} = -\dfrac{\varphi_0}{\ell_0}\,\hat\gamma_5\,\partial_x\hat\Phi_{\rm cut}(x_{\rm TC+FQ})$.
Physically this is (current in the line) × (flux drop across the coupler junction): an *inductive*,
i.e. *spatial-derivative*, coupling. J23 App. B writes the same term with the opposite overall sign,
$H_{\rm int} = \tfrac{1}{l_0}\gamma_5\phi_0\partial_x\Phi(0)$ (sign convention of $\gamma_5$; unobservable).

**1.3 Two-level + transversal + adiabatic reduction.** Eq. (23) $H_{\rm qb} = \hbar\Omega(f_\beta)|1\rangle\langle 1|$;
Eq. (27) $\hat\gamma_5^{\rm qb} = \gamma_x\sigma_x + \gamma_y\sigma_y + \gamma_z\sigma_z + \gamma_{id}\mathbb 1$ with
$\gamma_y = \gamma_z = \gamma_{id} = 0$ at the symmetry point, $\gamma_x = {\rm Re}\langle 1|\hat\gamma_5|0\rangle \in [-0.02, 0.22]$
for $f_\beta\in[0.3,0.5]$ (Sec. III.A.3, footnote 6 parameters = J23 best fit); $\gamma_x(t) = \gamma\chi(t)$;
gap Eq. (29)–(30): $\Omega(t) = \Omega^0 + \Delta\Omega\,\chi(t)$, $\Omega^0/2\pi \approx 7.3$ GHz,
$\Delta\Omega/2\pi \approx -23\gamma$ GHz. Result, **TB26 Eq. (31)**:

$$
\hat H_{\rm int}(t) = -\frac{\varphi_0}{\ell_0}\,\gamma\,\chi(t)\,\hat\mu(t)\,\partial_x\hat\Phi_{\rm cut}(t,x_D),
\qquad \hat\mu(t) = e^{i\varphi(t)}\hat\sigma^+ + {\rm H.c.}\ \ (32).
$$

**1.4 The generic VGSD detector and the paper's $\lambda$.** TB26 defines a dimensionless field, Eq. (36),

$$
\hat\phi_{\rm cut}(t,x) = \int dk\,\frac{{\rm cut}(\omega_k)}{\sqrt{4\pi|k|}}\left(e^{i(\omega_k t-kx)}\hat a_k^\dagger + {\rm H.c.}\right),
\qquad\text{so that}\qquad \hat\Phi_{\rm cut} = \sqrt{\hbar Z_0}\,\hat\phi_{\rm cut},
$$

and the interaction, **Eq. (37)**: $\hat H_I(t) = \hbar c\sum_\nu\lambda_\nu\chi_\nu(t)\hat\mu_\nu(t)\,\partial_x\hat\phi_{\rm cut}(t,x_\nu)$,
with $c \to v$ in the circuit (Sec. VI.A). Matching (31) to (37):

$$
\hbar v\,\lambda = -\frac{\varphi_0}{\ell_0}\gamma\sqrt{\hbar Z_0}
\ \Rightarrow\
\lambda = -\frac{\varphi_0\sqrt{\hbar Z_0}}{\hbar v\ell_0}\gamma
= -\frac{1}{2e}\sqrt{\frac{\hbar}{Z_0}}\,\gamma
= -\sqrt{\frac{\hbar}{4e^2 Z_0}}\,\gamma
= -\sqrt{\frac{R_K}{8\pi Z_0}}\,\gamma,
$$

using $v\ell_0 = Z_0$ and $R_K = h/e^2$ — **TB26 Eq. (66)**, $\lambda_\nu \approx -4.53\,\gamma_\nu$ at $Z_0 = 50\ \Omega$.
With SI-exact constants: $R_K = 25812.8075\ \Omega$, $\sqrt{R_K/(8\pi\cdot 50)} = 4.5322$. ✔

**1.5 Three independent checks of the normalization (same physics, separately derived):**

* TB26 App. A, Eq. (A6): spectral density $J(\omega) = \frac{R_K}{8\pi Z_0}\gamma^2\omega\,e^{-\omega/\Omega_{\rm cut}} = \pi\alpha\omega$,
  $\alpha = \frac{R_K}{8\pi^2 Z_0}\gamma^2 \approx 6.54\gamma^2$ (Eq. 33). Hence $\lambda^2 = \pi\alpha$ exactly
  ($4.5322^2 = \pi\times 6.5385$ ✔), as TB26 states after (66).
* FD17 Supp. S7: $\Gamma_1/\Delta = J(\Delta)/\Delta = \frac{\hbar}{4e^2 Z_0}|\varphi_\beta|^2 = \frac{1}{2\pi}\frac{R_Q}{Z_0}|\varphi_\beta|^2$
  with $R_Q = h/(2e)^2 = R_K/4$ — identical to $R_K|\varphi_\beta|^2/(8\pi Z_0) = \lambda^2$ with $\varphi_\beta \leftrightarrow \gamma$.
  Note FD17 stresses the factor 2 from the two semi-infinite lines ($k<0$ and $k>0$); it is contained in $\int dk$ over all $k$.
* J23 Sec. II.C / App. B: $\Gamma_1 = \frac{\phi_0^2}{\hbar Z_0}|\gamma_{5,10}|^2\Delta$, $\Gamma_1 = \pi\alpha\Delta$ ⇒ same $\alpha$.
  ($\phi_0^2/\hbar Z_0 = R_K/8\pi Z_0 = 20.541$ at 50 Ω, checked numerically.)

**Table I consistency (TB26 Sec. VI.A):** with $-4.532\gamma$: $\gamma = 0.02, 0.07, 0.14, 0.22, -0.02 \to
\lambda = -0.0906, -0.317, -0.634, -0.997, +0.0906$, printed as $-0.1, -0.3, -0.65, -1, 0.1$;
$\Delta\Omega/2\pi = 5.2\lambda$ GHz $= -0.52, -1.6, -3.4, -5.2, +0.52 \to$ printed $-0.5, -1.6, -3.4, -5.2, 0.5$;
$\alpha = \lambda^2/\pi = 0.0032, 0.029, 0.13, 0.32, 0.0032 \to$ printed $0.003, 0.03, 0.1, 0.3, 0.003$.
So **scenario 2 is defined by $\lambda = -0.10$ as simulated** (its $\Delta\Omega = -0.5$ GHz follows from $5.2\lambda$,
not from $-23\gamma = -0.46$); $\gamma = 0.02$ is a rounded label ($-4.53\times0.02 = -0.0906$).

---

## 2. Field normalization: TB26's $\hat\phi$ *is* the lattice field, energy unit $\hbar v/a$

Write $H_{\rm TL}$ in terms of $\hat\phi = \hat\Phi/\sqrt{\hbar Z_0}$ using $q = c_0\partial_t\Phi$:

$$
H_{\rm TL} = \frac{\hbar Z_0}{2}\int dx\left[c_0(\partial_t\phi)^2 + \frac{(\partial_x\phi)^2}{\ell_0}\right]
= \frac{\hbar v}{2}\int dx\left[\left(\frac{\partial_t\phi}{v}\right)^2 + (\partial_x\phi)^2\right],
$$

since $Z_0c_0 = 1/v$ and $Z_0/\ell_0 = v$. With $\tau = vt$ (time in length units): $H_{\rm TL}/(\hbar v) = \tfrac12\int dx[(\partial_\tau\phi)^2 + (\partial_x\phi)^2]$.
So $\hat\phi$ is the canonically normalized massless scalar in $\hbar = v = 1$ units, $[\phi(x),\partial_\tau\phi(x')] = i\delta(x-x')$;
its mode normalization $1/\sqrt{4\pi|k|}$ (Eq. 36) is exactly the standard one, and $\hat\phi$ is **dimensionless and scale-free**
(no length unit enters it). Check: $\langle\partial_x\phi(x)\partial_x\phi(0)\rangle = \int dk\,|k|e^{ikx}/4\pi = -1/(2\pi x^2)$.

Discretize with spacing $a$: $\int dx \to a\sum_i$, $\partial_x\phi \to (\phi_{i+1}-\phi_i)/a$, canonical momentum
$\Pi_i = a\,\partial_\tau\phi_i$, $[\phi_i,\Pi_j] = i\delta_{ij}$:

$$
H_{\rm TL} = \frac{\hbar v}{a}\cdot\frac12\sum_i\left[\Pi_i^2 + (\phi_{i+1}-\phi_i)^2\right] \equiv \frac{\hbar v}{a}\,H_{\rm lat},
$$

which is `harmonic_chain_K` at $m=0$ **with no field rescaling**: $x_i = \hat\phi(x_i)$, $p_i = \Pi_i = \partial\phi_i/\partial t_{\rm lat}$,
$t_{\rm lat} = vt/a$. The lattice energy unit is $\hbar\omega_{\rm ref}$ with $\omega_{\rm ref} \equiv v/a$; equivalently
$a = v/\omega_{\rm ref}$ — precisely `experiments_io`'s light-crossing convention ($L_{\rm lat} = L\,\omega_{\rm ref}/v$).
Lattice check of the *absence* of a stray factor: for the massless chain $\langle\Delta\phi_0\Delta\phi_r\rangle = \tfrac12(K^{1/2})_{0r}
= 2/[\pi(1-4r^2)] \to -1/(2\pi r^2)$, the continuum value with $a=1$ (pinned in the test).

Consequence for a smearing: $\hat\Phi_{\rm cut}(x) = \int dx' F_{\rm eff}(x'-x)\hat\Phi(x')$ (Eq. 10) becomes $\sum_i (aF_{\rm eff}(x_i-x))\,\phi_i$,
so the lattice profile is $F_i = aF_{\rm eff}$ with $\sum_i F_i = 1$ — the repo's normalization. No extra factor.

---

## 3. Spatial-derivative coupling ↔ the repo's `'xp'` (momentum) coupling: exact, and $a$-independent

TB26 couples to $\partial_x\hat\phi$; `'xp'` couples to $p_i = \partial_{t_{\rm lat}}\phi_i$ (TBMM24 Eq. (2): $\lambda\chi\mu\int F\,\partial_t\hat\phi$).
For the massless 1+1 vacuum the two are *interchangeable for every detector observable*:

* Continuum: $\phi = \phi_L(x+\tau) + \phi_R(x-\tau)$, $\partial_x\phi = \partial_\tau\phi_L - \partial_\tau\phi_R$; the vacuum is a product over
  chiralities with vanishing $L$–$R$ cross-correlators, so $\langle\partial_x\phi(\mathsf x)\partial_{x'}\phi(\mathsf x')\rangle
  = \langle\partial_\tau\phi(\mathsf x)\partial_{\tau'}\phi(\mathsf x')\rangle$ for all $\mathsf x,\mathsf x'$ (both $= \int dk\,|k|e^{ik\Delta x - i|k|\Delta\tau}/4\pi$).
  Since the field is Gaussian and enters $L_{\mu\nu}, M$ (TB26 Eqs. 49–50, 55) only through this two-point function, all
  second-order — and, for the Gaussian oscillator model, all — detector quantities coincide.
* Lattice: $\langle\Delta\phi_k\Delta\phi_{-k}\rangle = |e^{ik}-1|^2/(2\omega_k) = \omega_k/2 = \langle p_kp_{-k}\rangle$ mode by mode
  ($\omega_k^2 = 4\sin^2(k/2)$), so $D V_{xx} D^{\rm T} = V_{pp}$ exactly for the massless translation-invariant chain
  (pinned in the test; boundaries and the half-site shift of the bond midpoint are the only differences on a finite Dirichlet chain).

Now convert Eq. (37) to lattice units. With $\partial_x\hat\phi \to (\phi_{i+1}-\phi_i)/a$ and energies in units of $\hbar v/a$:

$$
\frac{\hat H_I}{\hbar v/a} = \lambda\,\chi(t)\,\hat\mu(t)\,(\phi_{i+1}-\phi_i)\ \ \equiv\ \ \lambda\,\chi\,\hat\mu\,p_i\quad(\text{in observables}).
$$

**The paper's dimensionless $\lambda$ is the lattice two-level `'xp'` coupling exactly, for any lattice spacing**:
the $1/a$ of the derivative cancels against the energy unit $\hbar v/a$, because a derivative coupling in 1+1 is
marginal (dimensionless). Equivalently via the time route: $\hbar v\lambda\partial_x\phi \to \hbar\lambda\partial_t\phi = \hbar\omega_{\rm ref}\lambda\,p_i$ ✔.
Neither $v$ nor $Z_0$ nor $a$ enters beyond the $-4.53$ already inside $\lambda$. (Sec. VI.A of TB26 uses $v$ only through $t_d = d/v$.)

---

## 4. Two-level → oscillator: the matrix-element factor $\sqrt{2\Omega_{\rm lat}}$

The lattice detector is $H_d = \tfrac12(p_d^2 + \Omega_d^2x_d^2)$ (unit mass, $\hbar=1$): $x_d = (b+b^\dagger)/\sqrt{2\Omega_d}$,
$\langle 1|x_d|0\rangle = 1/\sqrt{2\Omega_d}$, $\langle 0|x_d^2|0\rangle = 1/2\Omega_d$ (= `ground_state_cov`'s $\tfrac12K^{-1/2}$).
The qubit monopole has $\langle 1|\sigma_x|0\rangle = 1$. To second order both detectors' $L_{\mu\nu}$ and $M$ are
(detector matrix element)$^2$ × (the same field integrals), and the oscillator's radiative decay rate equals the qubit's
$\Gamma_1$ under the same condition, so the faithful map is

$$
\lambda_{\rm osc}\langle 1|x_d|0\rangle = \lambda_{\rm TB}\langle 1|\sigma_x|0\rangle
\ \Rightarrow\
\boxed{\ \lambda_{\rm osc}^{\rm xp} = |\lambda_{\rm TB}|\sqrt{2\Omega_{\rm lat}},\qquad \Omega_{\rm lat} = \Omega/\omega_{\rm ref} = \Omega a/v\ }.
$$

Convention hazard: BMMM13 (the reference cited by `attach_detectors`) writes the oscillator monopole as
$\hat\mu = \hat a_d + \hat a_d^\dagger$ **and** the field as $\sum_n(a_nv_n + a_n^\dagger v_n)$ with *no* $1/\sqrt{2\omega}$ factors,
so their $\lambda$ is neither the repo's $\lambda_{\rm osc}$ nor TB26's $\lambda$. In the repo's $x_d$ convention the
$\sqrt{2\Omega_{\rm lat}}$ is mandatory; identifying $\lambda_{\rm osc} = \lambda_{\rm TB}$ would be off by $\sqrt{2\Omega_{\rm lat}}$
($=\sqrt2$ in gap units, $0.54$ in cutoff units). The test pins this with a direct numerical qubit-vs-oscillator comparison
(exact $2\times$Fock qubit dynamics vs the repo's Gaussian evolution through `attach_detectors`) for both `'xx'` and `'xp'`.

The sign of $\lambda_{\rm osc}$ is unobservable for identical detectors ($x_d\to-x_d$ on both).

Which $\Omega$: the lattice oscillator has a static gap (`attach_detectors`), whereas TB26's gap is $\Omega^0 + \Delta\Omega\chi(t)$
(Eq. 29). The two natural static choices are $\Omega^0$ ($2\pi\times7.3$ GHz) and the fully-coupled gap $\Omega^0+\Delta\Omega$
($2\pi\times6.8$ GHz for scenario 2). The gap variation itself is a modelling difference, not a coupling-constant issue.

---

## 5. `'xx'` (amplitude) coupling: a spectral match, not an equivalence

The field operators seen by the two couplings have spectral weights $\langle p_kp_{-k}\rangle = \omega_k/2$ and
$\langle x_kx_{-k}\rangle = 1/2\omega_k$. A detector of gap $\Omega_{\rm lat}$ interrogates $\omega = \Omega_{\rm lat}$
(for switchings long compared with $1/\Omega$), so equal on-resonance weight requires $\lambda_{\rm xx}^2/2\Omega = \lambda_{\rm xp}^2\Omega/2$:

$$
\boxed{\ \lambda_{\rm osc}^{\rm xx} = \Omega_{\rm lat}\,\lambda_{\rm osc}^{\rm xp} = \sqrt2\,|\lambda_{\rm TB}|\,\Omega_{\rm lat}^{3/2}\ }.
$$

In gap units ($\Omega_{\rm lat} = 1$) the two forms coincide numerically. This is **not** a faithful map of the device:
the spectral densities differ off resonance ($\omega$ vs $1/\omega$), which matters for TB26's short switchings
($T\Omega^0 \approx 3.2$ for scenario 1's Gaussian $T = 0.07$ ns) and for the two-detector $M$ term; and the massless
amplitude coupling is IR-log-divergent in 1+1. `'xp'` is the physical one.

---

## 6. Ambiguities and their numerical spread

| # | ambiguity | readings | effect on $\lambda_{\rm osc}^{\rm xp}$ |
|---|---|---|---|
| A1 | scenario-2 $\lambda_{\rm TB}$: Table I value vs Eq. (66)×$\gamma$ | $-0.10$ vs $-0.0906$ | $-9.4\%$ (one-sided; Table I is what the paper simulated) |
| A2 | static oscillator gap | $\Omega^0 = 2\pi\cdot7.3$ GHz vs $\Omega^0+\Delta\Omega = 2\pi\cdot6.8$ GHz | $\sqrt{6.8/7.3}$: $-3.5\%$ if $\omega_{\rm ref}$ is held at $\Omega^0$; $0$ if the lattice gap is set to 1 either way |
| A3 | energy unit / lattice spacing (a repo choice, not a paper ambiguity) | gap units $\omega_{\rm ref}=\Omega^0$ ($a = 2.6$ mm) vs cutoff units $\omega_{\rm ref}=\Omega_{\rm cut}$ ($a = v/\Omega_{\rm cut} = 0.38$ mm, TB26 footnote 2) | factor $\sqrt{\Omega^0/\Omega_{\rm cut}} = \sqrt{0.146} = 0.382$. The unit-free statement is $\lambda_{\rm osc}^2/2\Omega_{\rm lat} = \lambda_{\rm TB}^2$. A lattice must have $\Omega_{\rm cut,lat} < 2$ (Nyquist) to represent the cutoff at all, which forces $\omega_{\rm ref} \gtrsim \Omega_{\rm cut}/2$; gap units cannot resolve the 50 GHz cutoff. |
| A4 | oscillator normalization ($x_d$ vs $a+a^\dagger$, §4) | repo's $x_d$ is unambiguous | none in the repo convention; $\sqrt{2\Omega_{\rm lat}}$ if BMMM13's monopole were assumed |
| A5 | $Z_0$ (device level only; TB26's scenarios fix $\lambda$ directly) | $50\ \Omega$ "≈" (TB26, FD17 "close to nominal"), G08 measured $59.7\ \Omega$ for their geometry | $\lambda\propto Z_0^{-1/2}$: $\pm5\%$ for $\pm5\ \Omega$; $-8.5\%$ at $59.7\ \Omega$ |
| A6 | $v$: $1.2\times10^8$ m/s (TB26) vs G08 $c/\sqrt{5.05} = 1.33\times10^8$ | — | **cancels exactly** (enters only $t_d = d/v$ and $a$ in metres) |
| A7 | $\partial_x$ vs $\partial_t$ coupling | — | none (§3, exact); on a finite Dirichlet chain only boundary/half-site effects |
| A8 | smearing normalization | $\sum_iF_i = 1$ required | none with the repo's `Smearing`; a profile normalized to $\sum F_i = 1/a$ would be off by $1/a$ |
| A9 | `'xx'` map | on-resonance spectral match | not an equivalence (§5) |
| A10 | device reach | J23 *measured* $\alpha\in[6.2\times10^{-5}, 2.19\times10^{-2}]$ ($\Gamma_1$ up to 1.85 GHz at $f_\beta = 0.44$), simulation predicts more; discrepancy unexplained in J23 | scenario 2 ($\alpha = 0.003$) is inside the measured range; scenarios 4–5 ($\alpha = 0.1, 0.3$) rely on extrapolation |

---

## 7. The number (committed before Step 4)

For **TB26 Table I scenario 2** ($\lambda_{\rm TB} = -0.10$, $\gamma = 0.02$, $\alpha = 0.003$, $\Delta\Omega/2\pi = -0.5$ GHz),
free gap $\Omega^0/2\pi = 7.3$ GHz, transcribed $Z_0 = 50\ \Omega$:

* **Unit-free:** $\lambda_{\rm osc}^{\rm xp}{}^2/(2\Omega_{\rm lat}) = \lambda_{\rm TB}^2 = 0.0100$ (range $0.0082$–$0.0100$ from A1).
* **Gap-unit lattice** ($\omega_{\rm ref} = \Omega^0$, $\Omega_{\rm lat} = 1$, $a = 2.62$ mm):
  $\lambda_{\rm osc}^{\rm xp} = \sqrt2\times0.10 = \mathbf{0.141}$; $\lambda_{\rm osc}^{\rm xx} = \mathbf{0.141}$.
  Spread: A1 → 0.128; A2 → 0.136; A5 (device) → 0.134–0.149. Quoted: $\lambda_{\rm osc}^{\rm xp} = 0.14\ ^{+0.01}_{-0.02}$.
* **Cutoff-unit lattice** ($\omega_{\rm ref} = \Omega_{\rm cut}$, $\Omega_{\rm lat} = 0.146$, $a = 0.382$ mm):
  $\lambda_{\rm osc}^{\rm xp} = 0.10\sqrt{0.292} = \mathbf{0.0540}$; $\lambda_{\rm osc}^{\rm xx} = 0.146\times0.0540 = \mathbf{0.0079}$.
  Spread: A1 → 0.0490; A2 → 0.0521; A5 → 0.051–0.057. Quoted: $\lambda_{\rm osc}^{\rm xp} = 0.054\ ^{+0.003}_{-0.006}$, $\lambda_{\rm osc}^{\rm xx} = 0.0079\ ^{+0.0005}_{-0.0009}$.
* **General:** $\lambda_{\rm osc}^{\rm xp} = 4.532\,|\gamma|\sqrt{2\Omega/\omega_{\rm ref}}\,\sqrt{50\ \Omega/Z_0}$, all scenarios in `rederive.py`'s table
  (e.g. scenario 5, $\lambda_{\rm TB} = -1$: $1.41$ gap units, $0.54$ cutoff units).

Equation chain in one line: (3),(5) → (22) → (27),(31) → (36),(37) → (66) [$\lambda_{\rm TB} = -\sqrt{R_K/8\pi Z_0}\,\gamma$]
→ §2 [$\hat\phi$ = lattice $x_i$, energy unit $\hbar v/a$] → §3 [$\lambda_{\rm qubit,lat} = \lambda_{\rm TB}$ exactly]
→ §4 [$\times\sqrt{2\Omega_{\rm lat}}$] → §5 [$\times\Omega_{\rm lat}$ for `'xx'`].

The uncertainty that matters for a *device* verdict is dominated by A1/A5 (~10%) on top of the convention choice A3,
which is not an uncertainty but must be stated with any quoted $\lambda_{\rm osc}$.
