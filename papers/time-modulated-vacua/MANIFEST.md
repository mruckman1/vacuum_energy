# MANIFEST — time-modulated-vacua

Result candidate of Layer 5 (PLAN.md, `vacuum/floquet/README.md`).

---

## The claims

**Prior art (literature check, 2026-09-23; `docs/LITERATURE_CHECK_2026-09-23.md`).** Theorem F is classical: the first-order
growth rate of a two-mode sum-frequency (combination) parametric resonance, with its 1/√(ω₁ω₂) —
geometric-mean — dependence, is the combination resonance of linear systems with periodic
coefficients (Yakubovich & Starzhinskii), stated for two coupled oscillators as the gain
δ/(4√(ω₁ω₂)) in Svidzinsky, Zhang, Wang, Wang & Scully, arXiv:1407.3727, Eq. (15); the square-wave
factor 2/π is a Fourier coefficient. Theorem M was searched for (INSPIRE, web) and not found as
stated; its technique is standard, so it stays POSSIBLY KNOWN. Pump-enhanced harvesting was not
found: no paper studies entanglement harvesting from a parametrically driven or dynamical-Casimir
field. The loss threshold Q_th = π/(2 ln λ_max) is the gain-equals-loss condition; its content is
the platform placement.

**(a) Amplification bound — a theorem of the program, with its priors** (`vacuum/floquet/bounds.py`, Theorems A and B; tests `tests/test_floquet_bound.py`; priority: docs/PRIORITY.md — a rederivation of a folklore bound, see *Prior work* below). For a quadratic lattice field under uniform coupling modulation K(t) = K0 (1 + d s(t)) — the photonic-time-crystal model, every normal mode a Hill oscillator x'' + ω_k²(1 + d s(t)) x = 0 — with |s(t)| ≤ 1 **measurable** and 0 ≤ d < 1:

*Theorem A (per cycle).* In the Prüfer variables x = r cos θ, p = −ω_k r sin θ the equation is exactly d ln r/dt = (ω_k d/2) s sin 2θ, dθ/dt = ω_k (1 + d s cos²θ) ≥ ω_k(1 − d) > 0; with θ as the clock, d ln r/dθ = (d s sin 2θ/2)/(1 + d s cos²θ) is monotone in s at every phase, so the pointwise maximiser is s = sgn(sin 2θ) (unique a.e., no singular arcs) and its integral over any π-window of θ is ½ ln(1+d) − ½ ln(1−d) = artanh(d). Hence for every solution, `ln r(t₂)/r(t₁) ≤ n artanh(d)` whenever θ advances by nπ (n = zeros of x in [t₁, t₂)), with equality iff s = sgn(sin 2θ) a.e. — periodic or not. *Corollary (Floquet):* for T-periodic s, `ln max|eig S_F| ≤ Z artanh(d)`, Z the number of zeros per period of the unstable Floquet solution (its Prüfer winding; Z = 1 on the principal tongue, Z = m on the m-th — the normalisation "per cycle" means), attained only by the quarter-period two-level (bang-bang) drive. This closes MANIFEST claim (a) of the producing run, which had proved it for two-level drives only and conjectured Pontryagin bang-bang optimality for the rest; the change of clock makes the Pontryagin singular-arc question moot (module docstring, with sources: Prüfer, Math. Ann. 95, 499 (1926); Boscain–Sigalotti–Sugny, PRX Quantum 2, 030203 (2021) Secs. 6.3/9.2; Hochma–Margaliot, arXiv:1407.3437; Zu–Li, Appl. Math. J. Chin. Univ. 33, 253 (2018); Meissner, Schweiz. Bauzeit. 72, 95 (1918)).

*Theorem B (rate).* The maximal Lyapunov exponent over all admissible s is ω_k λ*(d), λ* the unique root of artanh(d/(1+λ²)) = λ τ(λ), τ(λ) = [π/2 + arctan(λ/√(1+d))]/√(1+d) + [π/2 − arctan(λ/√(1−d))]/√(1−d) (Dinkelbach, Management Sci. 13, 492 (1967)); the optimal drive switches down at x = 0 and up at Prüfer phase π − arctan λ*, *before* the turning point. The quarter-period bang-bang is therefore **not rate-optimal**: λ* exceeds artanh(d)/T* by 0.025 % (d = 0.05), 0.10 % (0.1), 0.98 % (0.3), 3.14 % (0.5), 13.5 % (0.8), 26.1 % (0.9), 44.1 % (0.95) — dwell ratios to the quarter periods 1.084 (high) / 0.857 (low) at d = 0.5 — while gaining less per cycle (artanh(d/(1+λ*²)) < artanh(d)). A free two-dwell optimisation of the Meissner trace reproduces λ* to 1e-9 (test). The loss-threshold claim (b) is a per-period statement at the principal resonance and is unaffected.

*Prior work (priority check, 2026-09-04; docs/PRIORITY.md).* The equality case — the quarter-period two-level drive gains ω_max/ω_min = √((1+d)/(1−d)) per half period, i.e. artanh(d) in ln r per zero — is Meissner (1918) and textbook (Arnold, *Mathematical Methods of Classical Mechanics*, §25: the swing with piecewise-constant frequency). That the extremal drive of a bounded-frequency oscillator is bang-bang is established in the swing / parametric-oscillator optimal-control literature: Lavrovskii–Formal'skii, J. Appl. Math. Mech. **57**, 311 (1993); Piccoli–Kulkarni, IEEE Control Syst. Mag. **25**(4), 48 (2005); Andresen, Hoffmann, Nulton, Tsirlin, Salamon, Eur. J. Phys. **32**, 827 (2011), doi:10.1088/0143-0807/32/3/018 (minimum time to a target energy with ω_min ≤ ω ≤ ω_max, bang-bang); Salamon–Hoffmann–Rezek–Kosloff, PCCP **11**, 1027 (2009); Stefanatos–Ruths–Li, PRA **82**, 063422 (2010); for the spectral-radius objective on bilinear systems, Hochma–Margaliot, arXiv:1407.3437; and the optimal-control route to Hill's-equation criteria with a bounded coefficient (Lyapunov, Borg, Krein) is Zu–Li (2018). Theorem B is the growth-rate (Lyapunov-exponent) form of that minimum-time problem, whose switching instants those papers likewise fix by transcendental conditions. What this search did **not** find stated: inequality (8) of `bounds.py` for arbitrary *measurable* s in its per-zero form with the a.e. equality case, and the closed-form root equation for λ*(d) with the quantified sub-optimality of the quarter-period schedule. **Classification: POSSIBLY KNOWN** — read Theorems A and B as a rederivation with citations, not as a discovery of the program, until a control theorist who knows the 1960s–70s parametric-control literature has read `bounds.py` against Lavrovskii–Formal'skii and Andresen et al.

*Numerics as a check of the code, not evidence for the theorem* (`data/bound_adversarial.npz`, `summary.json['bound_theorem']`): 2000 random controls per family × five families (multi-frequency Fourier, clipped Fourier, sign-Fourier bang-bang, random piecewise, saturating smooth) × six depths (0.05–0.95) on 128 exact pieces at random periods (winding numbers 1–8 explored), plus projected-gradient (analytic trace gradient, 8 restarts) and cross-entropy optimisers at 8 periods per depth, all evaluated with **exact** piecewise monodromies (a discretised control is itself admissible, so no discretisation caveat): largest ratio ln max|eig S_F| / (Z artanh d) = **0.99975** (the rate-optimum drive at d = 0.05; per family: piecewise 0.99943, gradient 0.99957, sign-Fourier 0.99729, clipped 0.98410, smooth 0.97400, Fourier 0.77914, CEM 0.82980); the quarter-period bang-bang attains 1 to **5.3e-14**. The cosine drive keeps its earlier converged law (π/4) d (1 + c₂ d² + c₄ d⁴), c₂ = 0.01974, c₄ = −0.00526, i.e. 0.7854 of the bound.

*Multimode.* For `kind='coupling'` the 2N × 2N monodromy is block-diagonal in the normal modes — 24 random controls on an 8-site chain: block defect ≤ **2.7e-14**, every mode ≤ its own Z_k artanh(d) (worst 0.9906) — so the per-mode bound is exact on the lattice. `kind='mass'` also commutes with K0, but its relative depth is **per mode**, d_k = d·m_sq/ω_k² (`mass_mode_depths`), and the bound applies only while d_max = d·m_sq/ω_min² < 1; beyond it the softest mode inverts within the cycle and no claim here covers the drive (`modulated_K` validates only mass_sq > 0, so the regime is reachable: m_sq = 0.6 at depth 1 on `harmonic_chain_K(6, 0.05)` gives d_max = 2.99 Dirichlet / 240 periodic). `multimode_check(kind='mass')` raises there and otherwise checks each mode against its own artanh(d_k); pinned in `tests/test_floquet_bound.py`. For **non-commuting** patterns K(t) = K0 + d s(t) P (operator-norm depth d_eff = d ‖K0^{-1/2} P K0^{-1/2}‖) the 2026-09-12 pass (`vacuum/floquet/bounds_multimode.py`, `draft/multimode_bound.md`, `data/s3_multimode_verdict.json`, `tests/test_floquet_multimode.py`) **proved Theorem M**, the number-norm Grönwall bound `ln max|eig S_F|/T ≤ (d/2) ‖K0^{-1/4} P K0^{-1/4}‖ ≤ (d/2) ‖P K0^{-1/2}‖ ≤ ω_max d_eff/2` for every measurable |s| ≤ 1 (in the canonical complex coordinates the coupling splits into a number-preserving beam-splitter part and a squeezing part bounded by ‖C‖, C = K0^{-1/4} P K0^{-1/4}); for one mode it reads ω₀ d/2 against Theorem B's ω₀ λ*(d), so λ*(d)/(d/2) ∈ [0.63662, 0.68910] (measured, 200-point grid) and the theorem is at most π/2 looser than the conjecture where ‖C‖ = ω_max‖A‖ and sharper wherever ‖C‖/(ω_max‖A‖) < λ*(d_eff)/(d_eff/2) — on the archived battery below it is sharper on all 63 rows (0.492–0.870 of the conjectured bound). **Theorem F** (first order, rotating wave) gives the two-mode sum-frequency growth rate (d_eff/π)√(ω₁ω₂) under the square wave, i.e. √(ω₁ω₂)/ω_max × (d/π)/λ*(d_eff) ≤ 1 of the conjectured bound (the middle step uses the measured λ*(d) ≥ d/π; exact to 1.155·10⁻³ at d_eff ≤ 0.05): a low-frequency pair cannot beat the bound, and the frequency that governs a coupling is the geometric mean, not ω_max. The original rate bound `ln max|eig S_F|/T ≤ ω_max λ*(d_eff)` **remains conjectured** (neither proved nor refuted): the attack — sum-, difference- and near-degenerate structures, multi-harmonic bang-bang drives with free timing, CEM + Nelder–Mead over (ω, P, s, durations) at N = 2, 3, a first-order Schur-multiplier search — reached **1.000000000000017** of it (128 archived optima; 1566 further unarchived scratch rows re-tallied in the audit, max 1 + 5.4·10⁻¹⁴), always from below at an effectively single-mode configuration where Theorem B attains it, never above; the refined conjecture `ln max|eig S_F|/T ≤ (‖C‖/‖A‖) λ*(d_eff)` (sharper, reduces to Theorems A/B for one mode and commuting patterns) survived the same attack (max 1.000000000000036). The archived 60 random + 3 attacked patterns are regenerated bit-exactly and reach 0.169 of the conjectured, 0.401 of the refined and **0.261** of the proved bound.

**(b) Loss threshold.** With the K0-matched loss channel (transmissivity η per period) the photon-number multiplier is η λ_max², so η_th(d) = 1/λ_max(d)²; at the principal resonance η = exp(−π/Q), hence `Q_th(d) = π/(2 ln λ_max(d))`, d_th ≈ 2/Q (cosine) and d_th = tanh(π/2Q) (bang-bang). Below threshold the driven field does not return to vacuum: it saturates at a steady pair-production/loss balance (the open-line DCE regime); above it the Floquet multiplier wins. Platform mapping with transcribed numbers only:

| platform anchor (source) | Q | η per period | d_th (cos) | d_th (bang-bang) | 10 % modulation above threshold? | 1 % above threshold? |
|---|---|---|---|---|---|---|
| wilson2011_parasitic_Q_low — Wilson et al. 2011, p. 7: 'Q ~ 30-50' | 30 | 0.9006 | 0.06666 | 0.05231 | True | False |
| wilson2011_parasitic_Q_high — Wilson et al. 2011, p. 7: 'Q ~ 30-50' | 50 | 0.9391 | 0.04 | 0.03141 | True | False |
| wang2024_resonant_loss_threshold — Wang et al. 2024 Sec. I.3: amplification remains significant if gamma/omega_r < 0.05 | 20 | 0.8546 | 0.09998 | 0.07838 | True | False |
| wang2024_silicon_NIR — Wang et al. 2024 Sec. I.3: silicon gamma/omega_r ~ 1e-5 | 100000 | 1.0000 | 2e-05 | 1.571e-05 | True | True |

Reading: the demonstrated microwave regime (Wilson 2011: 10 % SQUID-inductance modulation, parasitic line resonances Q ~ 30–50) sits above the parametric threshold of those resonances — consistent with the ~Q² enhancement of the photon rate the paper reports — while a bare optical modulation at the ≤ 1 % depth that low-loss materials sustain (Wang et al. 2024) needs Q ≳ 200 optical-period-equivalent quality factors; Wang et al.'s resonant route (γ/ω_r < 0.05, i.e. Q > 20) works only because the resonance multiplies the effective modulation depth (their 350×), not because the bare threshold is lower.

**(c) Pump-enhanced harvesting (the composite).** Two harmonic detectors (x–x coupling, gap 1.2, λ_max = 0.4, cos² window) on a Dirichlet chain (m = 0.5) at separation 7; field pumped by a 'sin' coupling modulation for n_prep periods and optionally during the window. The 66-row survey (N = 16, sites [4, 11], `data/composite.npz`) stands as recorded — maximum enhancement 156 at {"omega_mod": 2.8, "depth": 0.1, "n_prep": 2, "n_win": 3, "pump_during_coupling": true}, best pumped E_N per unit total drive work 0.664 vs unpumped {"1.6": 2.2506, "2.0": 0.4084, "2.4": 0.1913, "2.8": 0.0092, "3.2": 0.0} (ratio 1.63); per unit of drive work the pump remains a worse deal than the vacuum wherever the vacuum harvests appreciably — **but two of its readings are corrected by this pass:**

*Causal labels.* The "spacelike-window control" rows at ω_mod = 1.6 (n_win = 2) have t_win = 2T = 7.85 > separation 7 (`spacelike` flag False): the "4.75× in the spacelike control" of the previous MANIFEST was a **connected** window and is withdrawn as a label. Every row with t_win < 7 has a dead baseline (E_N = 0), so the spacelike result is *revival* (E_N up to 5.2e-3 at ω_mod = 2.0, d = 0.1, N = 16), not an enhancement; conversely the 156× headline row (ω_mod = 2.8, t_win = 6.73) *is* spacelike.

*Vacuum/communication split* (`pumped_harvest_split`, `data/composite_split.npz`, all 66 rows re-run on one build). The M2.5 estimator (Tjoa–Martín-Martínez PRD 104, 125005 (2021), Eqs. (22)–(33)) applied to the kernel the Floquet-prepared Gaussian state presents to the detectors, W_ij(t, t') = [S(t) V_f S(t')ᵀ]_ij + (i/2)[S(t) Ω S(t')ᵀ]_ij — symmetrized (state) part plus the state-independent Pauli–Jordan commutator, S(t) the *driven* window propagator when the pump stays on — evaluated on the window grid (cumulative Simpson on the time-ordered triangle, halving gate 1e-7 on M, M_comm, P_A, P_B; converged on all rows, worst movement 9.5e-8, 128–512 grid points per period; at depth 0 it reproduces `communication_split` on `LatticeWightman` to 5e-9 in M and 2e-12 in M_comm). Result: the **unpumped connected windows are largely communication-assisted** — |M_comm|/|M| = 0.523 (ω_mod = 1.6), 0.757 (2.0), 0.155 (2.4), 0.020 (2.8), 0.003 (3.2) — while **every pumped d = 0.1 row is dominated by pre-existing pairs**: 0.022, 0.0097, 0.0043, 0.0009, 0.0003 (connected, ω_mod = 1.6–3.2), 0.0139 (the connected t_win = 7.85 row) and 4.1e-4, 5e-5, 1e-5, 0 on the genuinely spacelike rows (lattice-leakage level). Over all 66 rows the pumped communication fraction never exceeds **0.120** (a d = 0.02 row whose pump *lowers* E_N), |M_vac|/|M| ≥ **0.917** wherever E_N > 0, and TMM21's N⁻/N ≤ 0.046. The pump's own modification of the retarded propagator (|M_comm − M_comm^static|) reaches 0.9× the static commutator term but stays < 1 % of |M|. Caveat: the estimator is second order in λ; at λ_max = 0.4 its negativity overestimates the exact E_N by 2.3–24×, so only its *fractions* are claimed.

*Finite size* (`section_finite_size`, `data/composite_finite_size.npz`): N = 16, 32, 64 at fixed geometry (detectors centred at separation 7: sites [4, 11], [12, 19], [28, 35]; d = 0.1, n_prep = 2, pump on during the window; 36 audited rows, all converged at 64–128 substeps per period, worst ledger defect 4.0e-12). **The N = 16 survey is not converged in absolute enhancement:** connected ω_mod = 2.0: 78.2 → 32.6 → 32.6; ω_mod = 1.6: 21.2 → 9.77 → 9.64; ω_mod = 2.4: 30.8 → 22.0 → 22.0; the connected t_win = 7.85 row at ω_mod = 1.6: 4.75 → 27.0 → 27.0; ω_mod = 2.8: the baseline dies for N ≥ 32 (156 → revival, pumped E_N 5.39e-3 stable to 7e-6); spacelike ω_mod = 2.0 revival: 5.22e-3 → 4.60e-3 → 4.60e-3. The baselines move by ≤ 0.8 % between 16 and 32 and ≤ 2e-5 between 32 and 64; the pumped values move by factors 0.3–5.7 between 16 and 32 (the discrete mode nearest the gap: offset −0.034 at N = 16, −0.009 at 32, +0.005 at 64) and by ≤ **1.33 %** between 32 and 64 (ω_mod = 1.6; ≤ 2.3e-4 elsewhere). The communication fractions are N-stable to ≤ 0.9 %. **Headline to quote: ≈ 10–33× in connected windows at N = 64 (9.6, 32.6, 22.0, 27.0), with the 32 → 64 change as its finite-size error;** the 156× and the 5–116× range of the N = 16 survey are finite-size numbers.

**Status: claims about the model** (lattice, 'xx' oscillator detectors, one detector geometry); nothing is claimed about a specific device. The original sweep's full suite was **not** green on its concurrently-edited producing tree (9 failed / 6 errors, all in other agents' in-progress files — see "The producing build"); admissibility for those sections rests on the integration record below. The three new sections were produced on a later, again concurrently-edited working tree (below) with the Layer-5 test files green; the full-suite record they were owed is the integration line below, which covers them.


---

## The producing build

| item | value |
|---|---|
| generator | `papers/time-modulated-vacua/notebook.py`, entry point `main()` |
| generator sha256 (sections bounds, dce, composite) | `15190065d3ef3336fd51c3677f28a71dafd224480db00798892743cd297af31d` |
| git HEAD (working tree, uncommitted Layer 5 files) | `bd846820d712260eb9a4b6fcd0fbcf2f26b81f8d` |
| produced (UTC) | 2026-09-02T23:58:15Z |
| package / environment | vacuum 0.1.0; Python 3.13.3, numpy 2.5.2, macOS-14.6.1-arm64-arm-64bit-Mach-O |
| command | `.venv/bin/python papers/time-modulated-vacua/notebook.py` |
| wall clock | 626.3 s (bounds 443 s, dce 161 s, composite 22 s; budget 20 min) — measured while three other agents' processes shared the CPU; the quick smoke run is 26 s |
| pytest, full suite | **9 failed, 780 passed, 5 warnings, 6 errors in 746.35s (0:12:26)** (`.venv/bin/python -m pytest -q`, run concurrently with the sweep on the same code state) |
| pytest, Layer 5 | **52 passed in 63.92s (0:01:03)** (`tests/test_floquet_*.py`) |

The same sha256 is embedded in every archive under `data/` (`__build__` key). `--quick` archives (`*_quick.*`) are the smoke configuration.

**Integration: full suite 1201 passed on the quiescent tree at HEAD `c18a206`, 2026-09-04** (`.venv/bin/python -m pytest -q`, twice in a row on the final tree: 1201 passed, 10 warnings in 731.91s (0:12:11) and 1201 passed, 10 warnings in 740.37s (0:12:20); with tenpy, cvxpy, jax, jaxlib, qiskit, qiskit_aer and qiskit_ibm_runtime blocked at the import system, 1026 passed, 76 skipped in 428.84s (0:07:08); all 15 notebook `--quick` paths exit 0, the slowest in 307 s). The tree is commit `c18a206` plus this pass's uncommitted edits — README, STATUS, PRIORITY, MANIFEST and draft prose reconciling the rollups to the MANIFESTs, plus one print in `interacting-vacua-first-numbers/notebook_qei_exact.py` made None-safe so its `--quick` path exits 0 (no numerical code, no data). `data/` was produced on the build recorded above; this run is the admissibility record under `papers/README.md` for the candidate's code as it stands on that tree, and a full re-run of `notebook.py` there regenerates `data/` with a new `generator_sha256`; the producing-build fields above are left as recorded.

**Partial run (sections bound_theorem, split, finite_size)** — recorded per section in `summary.json['__build_by_section__']` and in `summary.json['__build__']['last_partial_run']`; the earlier sections and their build fields are untouched:

| item | value |
|---|---|
| generator sha256 | `c491ab94ff232606467ae838de1c6515569edbfa4145161add9397bf43975e94` |
| git HEAD (working tree, uncommitted files of several agents) | `eb2bb8fee7761a2ff65a021d5b6ac946f515de03` |
| produced (UTC) | 2026-09-04T00:26:25Z |
| command | `.venv/bin/python papers/time-modulated-vacua/notebook.py --sections bound_theorem,split,finite_size` |
| wall clock | 196.2 s (bound_theorem 132.9 s, split 25.1 s, finite_size 38.2 s) |
| pytest, Layer 5 | **62 passed in 60.90 s** (`tests/test_floquet_*.py`, incl. the new `test_floquet_bound.py`, 14 tests, and the 3 split tests in `test_floquet_composite.py`); the pre-existing 48 unchanged |
| pytest, full suite | not run in this pass (only the owned files and `tests/test_floquet_*.py`, per the pass's rules) |

## Multimode bound — numbers of the 2026-09-12 pass (`data/s3_multimode_verdict.json`; pinned in `tests/test_floquet_multimode.py`)

| quantity | value | knobs / where |
|---|---|---|
| Theorem M (proved) | ln ρ/T ≤ (d/2)‖K0^{-1/4} P K0^{-1/4}‖ ≤ (d/2)‖P K0^{-1/2}‖ ≤ ω_max d_eff/2 | every measurable \|s\| ≤ 1, d_eff < 1; `multimode_bound_proved` |
| single-mode constant: λ*(d)/(d/2) | 0.637000 (d = 0.1), 0.646732 (0.5), 0.675809 (0.9); monotone from 2/π = 0.63662 (d → 0) to 0.68910 (d → 1) on a 200-point grid | `rate_optimum` vs Theorem M at N = 1 |
| mutation: constant 1/π instead of 1/2 | violated, ratio 1.061558 (rate optimum of the ω_max mode in K0 = diag(0.09, 1), d = 0.9) | `test_wrong_constant_or_wrong_norm_in_the_proved_bound_is_caught` |
| Theorem M in its **operator-norm** form (ln ‖S_F‖ in the number metric, not only ln ρ) | holds on 40 random systems; batch maximum **0.879585** of the bound (so the bound is near-tight, not slack); the 1/π mutation is violated by the top row at **1.3816483** | N ≤ 4, d_eff ≤ 0.95, 2–8 pieces, seed 17; `test_theorem_M_operator_norm_form_is_tight_and_the_wrong_constant_is_caught` |
| Corollary M1 range, measured | λ*(d)/(d/2) monotone increasing, min **0.63661977** (= 2/π, d → 0), max **0.68909937** (d → 1) | 200-point grid on (1e-6, 1 − 1e-6); `test_corollary_M1_the_measured_range_of_lambda_star_over_d_half` |
| archived battery (60 random + 3 attacked patterns, 8-site chain, ω_max = 2.0321) vs proved bound | max ratio **0.2605884** (refined 0.4008852; conjectured 0.1687233); proved bound below the conjectured one on 63/63 rows, 0.4924–0.8697 of it | regeneration bit-exact: max \|Δ ln ρ\| = 0.0, \|Δ d_eff\| = 0.0 |
| Theorem F check: exact/first-order sum-frequency rate | \|ratio − 1\| ≤ 1.155·10⁻³ over d_eff ≤ 0.05, ω₁ ∈ [0.1, 0.99]; ≤ 6·10⁻⁵ at d_eff = 0.02, ω₁ ≥ 0.3 | square wave at ω₁ + 1, A = [[0,1],[1,0]] |
| attack (36 CEM + Nelder–Mead runs, N = 2, 3, M = 4, 8, d_eff 0.05/0.3/0.8, 3 seeds): max ratio to the conjectured bound | **0.9999999999999953** (refined 0.9999999999999956, proved **0.6657**); smallest of the 36 optima 0.7257 | best row: N = 2, M = 4, d_eff = 0.8, ω = (0.656, 1), T = 3.6283 = the rate-optimum cycle (top eigenvector of A on the ω_max mode alone) |
| **largest ratio to the conjectured bound anywhere in the archive** | **1.000000000000017** — a *detuned-pair* row, not a CEM row (refined 1.000000000000036); **reached** at ω → (1, 1), where Theorem B already attains it, and exceeded by +1.7e-14 — inside the 1e-13 roundoff of ln ρ — never broken | 128 archived polished optima (36 attack + 36 detuned + 56 sum-frequency) |
| detuned pairs (ω = (1 − δ, 1), δ ∈ {0.002, 0.03, 0.2}) | 36 polished optima in [0.99999994, 1.000000000000017] | from 1–2 cycles of the rate-optimum bang-bang |
| first-order Schur search, max σ_max(C∘Ŝ)/‖C‖ over (s, C) | 0.9864 × 2/π (nine frequency sets, N = 2–4, harmonics ≤ 8; exact refined ratio at d_eff = 0.01: 0.98639); exploratory 0.99991 × 2/π | 48-piece 2π-periodic s |
| verdict | `conjecture_refuted = False`; conjectured and refined bounds attained (Theorem B) and exceeded only by +1.7e-14 / +3.6e-14, inside the 1e-13 ln ρ roundoff, never broken; the archived ratio to Theorem M never above **0.6657** (0.6758 in the unarchived exploratory set = λ*(0.9)/(0.9/2)) | 128 archived polished optima; 1566 further rows in the first pass's scratch, re-tallied in the 2026-09-12 audit (max 1 + 5.4·10⁻¹⁴) but **not archived here and not pinned** |

Producing build of this step: `papers/time-modulated-vacua/notebook_multimode.py` (sha256 `defa7d3c979605625830255d8460ea5ed7e212dc52bde6770af5bc3330094277`), git HEAD `1ab508986194a076e046f955536e1f63e868c1cb` (working tree with the new files uncommitted; other agents' commits concurrent), produced 2026-09-12T18:00:14Z, wall clock 299.8 s (one thread), Python 3.13.3 / numpy 2.5.2; `--quick` 1.7 s (temporary directory, nothing under `data/`).

| audit item (second pass, 2026-09-12, reviewing agent) | value |
|---|---|
| reproduction of `data/s3_multimode_verdict.json` | re-ran `notebook_multimode.py` end to end at git HEAD `e834b7bc8801676daab535fa550fc56c8ee88331`, same generator sha256 `defa7d3c…`, wall clock **348.3 s** (2 threads, machine shared with the Layer-5 pools): **every float reproduces bit-exactly** — 0 differing fields outside `__build__` and the per-section timers |
| mathematics | Theorem M re-derived line by line (change of variables, reality of ā·Ωa and ā·Ca, ‖ā·Cā‖ ≤ ‖C‖\|a\|², the ρ(XY) = ρ(YX) chain, and the (M1) → (M2) operator-norm step, now stated with the halving and with submultiplicativity); Theorem F's rotating-wave step labelled ASSUMPTION in the module as well as the draft; \|ŝ(W)\| ≤ 2/π re-derived; the "pinching" of Theorem F(b) corrected to an average of unitary **congruences** U C Uᵀ; F(c) corrected from an equality to an upper bound on the first-order rate |
| number defects found in the first pass's draft and fixed | (i) the CEM attack's maximum conjectured ratio is 0.9999999999999953, not 1.000000000000017 (that is the detuned section's); (ii) the sum-frequency rows reach 0.63661 of the proved bound, not 0.676; (iii) the largest archived ratio to Theorem M is 0.6657, not 0.676; (iv) the detuned interval's upper end is 1.00000000000002, not 1; (v) the exploratory excess is +5.4·10⁻¹⁴, not 4·10⁻¹⁴, over 1566 (not "≈1500") unarchived scratch rows; (vi) the Schur search covers harmonics ≤ 8, not ≤ 11; (vii) λ*(d)/(d/2) reaches 0.68910, not 0.6890 |
| code changes of the audit | `multimode_check(kind='pattern')` no longer divides by zero at d_eff = 0 (`proved_ratio` → 0.0); two tests added (the operator-norm form of Theorem M with its 1/π mutation, and the measured Corollary M1 range) |
| pytest (this tree, 2026-09-12) | `tests/test_floquet_multimode.py` **10 passed in 1.1 s** (the tenth pins the per-section maxima of the attack, so the misattribution fixed above cannot recur); the Layer-5 set `bound, multimode, core, dce, runs, mathieu` **55 passed in 102.2 s**; with `tests/test_floquet_bound.py` + `tests/test_exports.py` **49 passed in 13.2 s** (an earlier run in this pass had `test_exports.py::…[geometry]` failing on another agent's then-unregistered `vacuum/geometry/jacobson_continuum.py`, unrelated to this candidate and since committed as `3c5dee2`); `notebook_multimode.py --quick` exits 0 in 1.1 s |


## Modules used

| module | role |
|---|---|
| `vacuum.floquet.drives` (`modulated_K`, `dce_chain_K`) | the drives, incl. the Johansson boundary condition |
| `vacuum.floquet.monodromy` (`floquet_map`, `stability_chart`, CF4 pieces) | gated one-period maps, stacked 2×2 mode maps |
| `vacuum.floquet.mathieu` | scipy characteristic values for the tongue widths |
| `vacuum.floquet.runs` (`max_gain_over_frequency`, `bang_bang_bound`, `loss_threshold`, `matched_bath_cov`) | the profile scans and the loss mapping |
| `vacuum.floquet.bounds` (`rate_optimum`, `bound_ratio`, `prufer_winding`, `adversarial_search`, `multimode_check`) | Theorems A/B, the exact batched piecewise monodromies, the adversarial battery, the multimode checks |
| `vacuum.floquet.bounds_multimode` (`multimode_bound_proved`, `frequency_weighted_norm`, `pattern_monodromy`, `rwa_growth_rate`, `sum_frequency_first_order`, `multimode_ratios`, `multimode_attack`) | Theorem M (proved multimode bound), Theorem F (first order), exact free-duration pattern monodromies, the attack; `multimode_check(kind='pattern')` reports `proved_ratio` / `refined_ratio` next to `rate_ratio` |
| `vacuum.floquet.dce` (`wilson_regime`, `dce_spectrum`, `parabola_fit`) | the Wilson 2011 regime at paper scale (N = 300) |
| `vacuum.floquet.composite` (`pumped_harvest`) → `vacuum.protocol.run_protocol` | the composite: one audited protocol per point |
| `vacuum.floquet.composite` (`pumped_harvest_split`, `floquet_window_rows`, `pair_integrals_on_grid`) → `vacuum.detectors.communication` | the M2.5 split on the Floquet-prepared two-time kernel |
| `vacuum.detectors` (`attach_detectors(coupling='xx')`, `harvested_log_negativity`) | Layer 2 stack (mp-margin rule, `flagged` carried) |
| `vacuum.audits` (`EnergyLedger`, passivity, precision) | the standing audits on every point |
| `vacuum.experiments_io` + `experiments/dce_microwave/wilson_2011_table1.json` | the only unit-conversion path |

## Parameter provenance

Transcribed: Wilson 2011 drive 11.30 GHz, c₀ = 0.4 c, v_e = 0.05 c, 50 mK, n_th = 0.008 at 5 GHz, 4–6 GHz band, 10 % inductance modulation, parasitic Q ~ 30–50 (paper text); Wang et al. 2024 statements (Δn/n ~ 1 non-resonant, < 1 % low-loss, γ/ω_r < 0.05 threshold, silicon γ/ω_r ~ 1e-5). **SYNTHETIC (declared simulation choices):** the lattice placement ω_d = 0.5 (one site = 0.845 mm), the static effective length L_eff⁰ = 0.5 sites = 0.42 mm (Johansson 2010 Fig. 4 uses 0.44 mm), the composite geometry (N = 16/32/64, m = 0.5, sites, gap, λ_max, window lengths, pump grids), the bound/threshold grids, the adversarial families and seeds (2026 / 7 / 99), the 8-site chain and random patterns of the multimode checks.


## Data

| file | contents |
|---|---|
| `data/bounds.npz` | per (profile, m, depth): ω*, ln λ, rate, period, golden-section iterations, `n_steps`, `halvings`, `movement` |
| `data/bound_adversarial.npz` | the battery: per (depth, family/optimiser, period): ratio ln max\|eig S_F\|/(Z artanh d), Z, ln multiplier, sample counts |
| `data/loss_threshold.npz` | depth, ln λ_cos, η_th, Q_th, Q grid, d_th(Q) for cos and bang-bang |
| `data/dce.npz` | binned spectra (centres, n_out, prediction) per row of the k L_eff⁰ / depth / temperature sweep |
| `data/composite.npz` | every composite point (N = 16): E_N, flagged, n_d, works, enhancement, causal flag, convergence and audit columns |
| `data/composite_split.npz` | the same 66 rows re-run with the split: \|M\|, \|M_vac\|, \|M_comm\|, fractions, N⁻/N, P_A, P_B, \|C\|, static-commutator term and drive shift, perturbative E_N, grid provenance |
| `data/composite_finite_size.npz` | N = 16/32/64 rows: E_N, enhancement, n_d, field occupation at the gap, nearest-mode offset, split columns, audits |
| `data/s3_multimode_verdict.json` | the multimode verdict step (`notebook_multimode.py`): the archived 63 pattern rows re-scored against the proved and refined bounds (bit-exact regeneration), the sum-frequency first-order table, the 36 CEM attacks, the 36 detuned-pair polishes, the 9 Schur-multiplier sets, the verdict, build info |
| `data/summary.json` | everything above plus the theorem statements, the rate-optimum table, the multimode rows, the split verdict with headline rows, the finite-size table, tolerance sweeps, tongue widths, lattice check, DCE rows, verdict |

## Audit status (composite rows) and convergence

- ledger closure (1e-9): all passed; worst defect **9.67e-13** (survey), **3.98e-12** (finite-size rows); passivity on every entry vacuum: all passed; precision audit on the detector block: all passed; mp-margin flag fired on 0 rows; all audits passed: **True**.
- dt-halving gate (1e-9 on E_N and both n_d): converged on all 66 + 36 rows; worst accepted movement 9.74e-10 (survey), 8.8e-10 (finite size); substeps per period 64–256 (CF4 pieces, two static exponentials each).
- split gate (1e-7 relative on M, M_comm, P_A, P_B): converged on all rows; worst movement 9.5e-8; 128–512 grid points per period; additivity M = M_vac + M_comm to ≤ 1e-16.
- bound convergence (tolerance sweeps, cosine): d=0.1: ln λ spread over gate ∈ {1e-6..1e-12} × ω_tol ∈ {1e-5..1e-9} = 2.90e-09; d=0.5: 1.32e-08; d=0.9: 1.71e-09. The battery and the multimode rows use exact piecewise monodromies (symplectic defect ≤ 3e-14) and need no gate.
- multimode lattice check (N = 24 chain vs single-mode law at ω_mod/ω_k): worst |Δλ| = 7.33e-12; block-diagonalisation defect on random controls ≤ 2.7e-14.

## DCE regime rows (N = 300)

| L_eff⁰ (sites) | k_d L_eff⁰ | depth scale | T (lattice) | prefactor A/(δL/v)² | shape RMS | R² | flux/Γ_DCE |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.250 | 1.0 | 0.0000 | 1.0076 | 0.0087 | 0.9995 | 1.0100 |
| 0.5 | 0.250 | 0.5 | 0.0000 | 1.0138 | 0.0085 | 0.9996 | 1.0165 |
| 0.5 | 0.250 | 1.5 | 0.0000 | 1.0076 | 0.0087 | 0.9995 | 1.0100 |
| 0.3 | 0.150 | 1.0 | 0.0000 | 1.0068 | 0.0095 | 0.9994 | 1.0094 |
| 1.0 | 0.500 | 1.0 | 0.0000 | 0.9750 | 0.0095 | 0.9994 | 0.9749 |
| 2.5 | 1.250 | 1.0 | 0.0000 | 0.6257 | 0.0409 | 0.9908 | 0.6160 |
| 0.5 | 0.250 | 1.0 | 0.0461 | 1.0214 | 0.1083 | 0.9084 | 1.0522 |

Depth-scaling exponents: 1.991 (×0.5→×1), 0 (×1→×1.5). The k_d L_eff⁰ = 1.25 row (the file's literal 10 % modulation of a 2.5-site effective length) shows the ideal formula overestimating the lattice flux — the theory's own k L_eff ≪ 1 condition, not a lattice failure.


## What is NOT claimed

- The mass-drive bound outside d_max = depth·mass_sq/ω_min² < 1 (the softest mode inverts; `mass_mode_depths` is the gate, `multimode_check` refuses).
- For **non-commuting** (pattern) modulation, the rate bound with the single-mode constant, `ln max|eig S_F|/T ≤ ω_max λ*(d_eff)`, and its refinement `(‖C‖/‖A‖) λ*(d_eff)`: conjectured, attacked (128 archived optima, max 1.000000000000017, always at effectively single-mode configurations; plus 1566 unarchived scratch rows) and not refuted, but not proved beyond first order; what is proved is Theorem M with the constant 1/2 (`(d/2)‖K0^{-1/4} P K0^{-1/4}‖`, `draft/multimode_bound.md`), and Theorem F at first order. Also unproved: that the first-order Schur-multiplier norm of ŝ(ω_i + ω_j) never exceeds 2/π for more than one resonant harmonic (tested ≤ 0.986 × 2/π archived, 0.9999 exploratory), and non-attainment of Theorem M for N ≥ 2. (The single-mode / commuting-kind bound is proved — Theorem A.)
- Any device gain (UV cutoff 2 lattice units; no A(ω) line response).
- An *exact* vacuum-vs-communication split: the M2.5 estimator is second order in λ and its negativity is off by 2.3–24× from the nonperturbative E_N at λ_max = 0.4; only the fractions |M_comm|/|M|, |M_vac|/|M| are claimed.
- The absolute enhancements of the N = 16 survey (`data/composite.npz`): superseded by the N = 64 values of the finite-size table wherever both exist; one detector geometry (separation 7) only.
- A "spacelike enhancement": every spacelike baseline is dead; the spacelike result is revival.


## Reproducing

```
.venv/bin/python papers/time-modulated-vacua/notebook.py           # full (all six sections)
.venv/bin/python papers/time-modulated-vacua/notebook.py --quick   # smoke
.venv/bin/python papers/time-modulated-vacua/notebook.py --sections bound_theorem,split,finite_size   # this pass (merges into summary.json)
.venv/bin/python papers/time-modulated-vacua/notebook_multimode.py            # multimode verdict step (300 s; --quick 1 s)
.venv/bin/python -m pytest tests/test_floquet_*.py -q
.venv/bin/python -m pytest -q                                      # admissibility gate
```
