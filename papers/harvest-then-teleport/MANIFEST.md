# MANIFEST — harvest-then-teleport

Result candidate of Layer 7, leap **L4** (PLAN.md): "harvest entanglement from the
vacuum, then spend it as the channel for energy teleportation, and compute the full
composite ledger — joules of measurement energy in, ebits harvested, joules made
remotely available out."

---

## The claim

**Prior art (literature check, 2026-09-23; `docs/LITERATURE_CHECK_2026-09-23.md`).** The setting — measurement-assisted energy
extraction from a state two detectors harvested from a field — was not found in the literature.
Two of its messages are known in general: that correlations, classical ones included, rather than
entanglement pay for measurement-assisted work, and that such work is bounded by temperature times
mutual information, are in Perarnau-Llobet et al., PRX 5, 041011 (2015), and Manzano, Plastina &
Zambrini, PRL 121, 120602 (2018) — claim 2 below verifies them on harvested states. Claim 4
refutes the task's own conjecture; that measurement-assisted extraction can exceed unitary
ergotropy is the premise of daemonic ergotropy (Francica, Goold, Plastina & Paternostro, npj
Quantum Inf. 3, 12 (2017)). Claim 6, that Gaussian measurements never lose to photon counting on
harvested states, was not found: the closest work, Kua, Serafini & Genoni, Quantum Sci. Technol.
11, 015014 (2026), arXiv:2506.22288, treats Gaussian (general-dyne) measurements only.

**Definition first, because the literal question has no literal answer.** Hotta's
QET extracts energy from a *locally passive, globally correlated* ground state. A
detector pair that has just harvested entanglement is *locally active* — the
switching that harvested it also excited it — so "QET on the harvested state"
would report Bob's own local ergotropy as teleported energy. **The surplus mapped
here is therefore not teleported energy: it is the classical-communication-assisted
extraction from an active state** — Bob's daemonic ergotropy under his own
oscillator Hamiltonian minus his no-communication ergotropy (Francica et al. 2017,
npj Quantum Inf. 3, 12), maximised over two measurement families on Alice: the
pure Gaussian (general-dyne) family in the closed forms of Kua–Serafini–Genoni 2025
(anchored to 1e-10 in `tests/test_harvest_teleport.py`) and, new in this revision,
the non-Gaussian photon-number-resolving and on/off detections computed from the
Fock representation of the harvested state with Bob's general (passive-state)
ergotropy. Neither family is known to be optimal (Bernards–Kleinmann–Gühne–
Paternostro 2019 exhibit POVMs that beat every projective measurement), so the
reported best is a **lower bound on the unrestricted daemonic gain**. The
true-QET reference — the teleportation analogue — is the second column: a weak
Hotta-type `g x_A x_B` coupling of the same pair, run on that coupling's *joint
ground state* (audited locally passive); it uses the ground-state correlations,
not the harvest.

With that definition, on a 1+1 Dirichlet lattice field (N = 41, working IR
regulator m = 0.2 — see claim 8 for the m → 0, N → ∞ study), two pointlike
oscillator detectors, cos² switching, swept about a base point that harvests
(λ = 0.3, Ω = 1, separation 2, half-width 3, T = 0), the map of 61 audited
rows (27 of them the simultaneous-switching `'xx'` map of the original claim)
says:

1. **The `'xx'` exchange rate `surplus / W_in` peaks at about 6 % and saturates with
   coupling strength**: 0.0599 at λ = 0.8 and 0.0596 at λ = 1.0 (where E_N has
   already died), from 0.0045 at λ = 0.1. Along the gap axis it peaks at Ω = 0.5
   (0.0534) and falls to 2.9e-06 at Ω = 3; along separation it falls by three
   orders of magnitude from 2 to 16 sites; along duration it peaks at the base
   half-width 3.
2. **The largest `'xx'` rate, 0.125, is at field temperature T = 0.8, where the
   harvested negativity is exactly zero** (`temperature-T0.8`). The rate is monotone
   increasing in T (0.0285, 0.0291, 0.0347, 0.0537, 0.0871, 0.125). 11 of the 27 `'xx'` rows (21 of all
   61) have E_N = 0 and surplus > 0: the surplus is paid by mutual
   information (thermal and classical correlations included), not by negativity.
   No bound of the form (energy scale) × f(E_N) exists (a separable classically
   correlated state has positive surplus, `tests/test_harvest_teleport.py`); the
   bound that holds on every row is the Landauer/Szilard one,
   `surplus ≤ k T_B I(A:B)` (True), reached to 74 % at `coupling-lam1`.
3. **The two stages point in opposite directions.** Ebits per unit switching work
   peak at weak coupling and higher gap (1.92 bits per unit work at Ω = 1.5,
   1.16 at λ = 0.1); joules per ebit peak at strong coupling (2.21 at
   λ = 0.8, where the ebits per joule are 0.0271). The product of the two stages
   is the exchange rate, capped near 6 % among harvesting `'xx'` points
   (0.0599 at `coupling-lam0.8`).
4. **The daemonic surplus is not bounded by the joint ergotropy of the pair** —
   `surplus_le_full_ergotropy_all = False`; the test file exhibits the clean case
   (two thermal modes at different temperatures on a beam splitter: zero joint
   ergotropy, positive surplus — a measurement is not a unitary). The task's
   conjecture "surplus never exceeds the ergotropy of the joint state" remains
   **false**.
5. **The true-QET reference delivers 0.00312 of the injected quantum** at
   g = 0.1 Ω_A Ω_B (E_A = Ω_A at the heterodyne, E_B = 0.00312, of which 0.000625
   is carried by the ground-state correlations and 0.0025 is Alice's pointer
   noise re-routed through the coupling); the harvested state under the same
   coupling gives a surplus of 0.00429 at the base point.
6. **Non-Gaussian measurements do not beat the Gaussian family anywhere on the
   map.** On every one of the 61 rows the best family is Gaussian
   (rows where it is not: none); photon-number-resolved conditioning in
   Alice's own number basis reaches at most 0.159 of the Gaussian optimum and
   on/off detection at most 0.103, the largest ratio being on `coupling-lam1`
   (E_N = 0, ν_B − ½ = 0.403: Gaussian 1.74e-01, PNR 2.76e-02,
   on/off 1.79e-02); at the base point the ratio is 0.0185 (PNR 2.35e-05,
   on/off 2.34e-05 against Gaussian 0.00127), and it falls to 4.5e-07 on the
   coldest rows. A third, *hybrid* column rides along as a diagnostic and is not part
   of the claim: PNR in a squeezed number basis (best converged basis of a
   s ∈ (1, 2, 4) × four-angle grid), at most 0.195 of the Gaussian optimum.
   It is a hybrid because counting photons in an s-squeezed basis tends to a
   *homodyne* as s → ∞ — a measurement already inside the Gaussian family — and its
   truncation gets harder exactly as it approaches that limit, so bases whose
   required cutoff exceeds the ceiling are skipped and non-converged ones rejected
   (the reported value falls back to plain PNR when no squeezed basis converges).

   The reading: on near-vacuum, weakly correlated states the quadrature
   correlations are first order in the detector coupling and the number–number
   correlations second order, so homodyning Alice tells Bob more than counting her
   photons; the ratio grows with the occupation, and is largest on the hottest,
   most strongly coupled row. On the two-mode squeezed vacuum, by contrast, the two
   families TIE exactly — PNR gain Ω sinh² r, the Gaussian value, because on a pure
   state every rank-one measurement purifies Bob — while on/off gives the strictly
   smaller Ω tanh² r (both closed forms anchored to 1e-10). So the answer to "is the
   Gaussian lower bound tight?" on this map is: the Gaussian family is the better of
   the two computed bounds on **every** harvested row, exceeding photon counting by a
   factor of 6.3 on the hottest row and 2.2e+06 on the
   coldest, and Fock-resolved conditioning beats it by nothing. Whether an
   unrestricted POVM does better is open (Bernards et al. 2019).
7. **The derivative (`'xp'`) coupling harvests at light contact, not on spacelike
   windows.** Its base point (0.00613 of switching work, E_N = 0.00579,
   1.36 bits per unit work) harvests more ebits per joule than `'xx'`
   (0.779) but converts them worse: surplus 2.89e-05, exchange rate 0.00471
   against 0.0285 (the detectors stay far colder, and the surplus is priced by
   ν_B − ½). Along the delay axis at separation 12 (E_N by delay — `'xx'`:
   0: 0 | 6: 0 | 9: 0 | 12: 0 | 15: 0.000136 | 18: 0.00153; `'xp'`: 0: 0 | 6: 0 | 9: 0 | 12: 0.0012 | 15: 0.00183 | 18: 0.00198) the derivative coupling carries negativity at full
   light contact (delay = 12) where `'xx'` has exactly none (delay-delay12-xp: E_N = 0.0012, W_in = 0.00629, surplus = 7.36e-06, rate = 0.00117 | `'xx'` at the
   same point: delay-delay12: E_N = 0, W_in = 0.0444, surplus = 1.99e-04, rate = 0.00449), the configuration Teixidó-Bonfill & Martín-Martínez 2024
   identify as genuine harvesting in causal contact. **On every out-of-cone row of
   the map neither coupling carries negativity** (such rows: 3 for each coupling; with E_N > 0:
   none for either coupling), while both carry a positive surplus (all positive: yes for both couplings;
   exchange-rate ranges 'xx' 2.3e-05 … 1.1e-03, 'xp' 4.1e-08 … 9.6e-06): that exchange rate is nonzero for the same
   reason it is nonzero at E_N = 0 anywhere — it is paid by mutual information, not
   entanglement. The archived scan (`data/xp_spacelike_scan.json`, 32 `'xp'`
   points on the N = 61 chain over mass 0.0/0.2, λ 0.6, Ω 0.5–3.0, half-width 2.0/4.0, at the map's own 1e-9 gate) finds
   0 points with E_N > 0 (smallest partial-transpose margin 8.0e-07
   ABOVE the floor, i.e. separable by that margin on every point).

   **These rows are "outside the nominal light cone", not "strictly spacelike", and
   the MANIFEST does not claim otherwise.** The lattice cone is fuzzy: the free-field
   commutator |[R_A(t), R_B(0)]| at the detector pair leaks past it with a
   Lieb-Robinson tail, recorded per row as `commutator_leakage` (max over the window
   and over the φ-φ, φ-π and π-π blocks; the π-π block binds for the derivative
   coupling). Measured, it reaches **2.76e-02** on the closest rows
   (d − 2 T_sw = 2 sites) and falls to 1.85e-05 on the far rows
   (d − 2 T_sw = 6). Because the result is a NULL — E_N exactly 0 everywhere — and
   residual leakage can only ADD detector correlation, never remove it, the leakage
   could only have manufactured a false POSITIVE and did not: **the null is
   conservative**, not an artifact of residual signalling. The same caveat applies to
   the map's own out-of-cone rows (separation ≥ 8 at T_sw = 3), whose closest row
   also sits 2 sites beyond the cone. The `'xp'` rows' best exchange rate is
   0.0102 (`xp_coupling-lam1-xp`), best among harvesting rows 0.00503
   (`ir_mass-mass0-xp`), best ebits per unit work 1.58 (`ir_size-N_field121-xp`).
8. **The IR limit.** Massless is admissible on the Dirichlet chain (positive definite
   K, no zero mode; the periodic massless chain is rejected up front, the
   `vacuum.core` convention). At N = 41 the m → 0 sequence is regular — `'xx'`:
   m=0: rate 0.06, W 0.112, E_N 0.026 | m=0.01: rate 0.0597, W 0.111, E_N 0.026 | m=0.02: rate 0.0587, W 0.109, E_N 0.026 | m=0.05: rate 0.053, W 0.0961, E_N 0.0259 | m=0.1: rate 0.0423, W 0.0735, E_N 0.0256 | m=0.2: rate 0.0285, W 0.0447, E_N 0.0241; `'xp'`: m=0: rate 0.00503, W 0.00572, E_N 0.00621 | m=0.01: rate 0.00503, W 0.00573, E_N 0.00621 | m=0.02: rate 0.00502, W 0.00574, E_N 0.0062 | m=0.05: rate 0.00501, W 0.00579, E_N 0.00616 | m=0.1: rate 0.00495, W 0.00594, E_N 0.00604 | m=0.2: rate 0.00471, W 0.00613, E_N 0.00579 — with the last step of the sequence
   (m = 0.01 → 0) moving the `'xx'` rate by 3.6e-04 and W_in by 8.0e-04 (`'xp'` rate
   by 5.2e-07). Removing the regulator changes the `'xx'` base point by a factor
   2.5 in W_in, 5.27 in surplus and 2.11 in exchange rate (E_N by 1.08);
   the `'xp'` point moves by 0.934 in W_in and 1.07 in rate. Then the massless
   size sequence — `'xx'`: N=41: rate 0.06, W 0.112, surplus 0.00671, E_N 0.026 | N=61: rate 0.0675, W 0.129, surplus 0.00868, E_N 0.0258 | N=81: rate 0.0726, W 0.141, surplus 0.0102, E_N 0.0257 | N=121: rate 0.0794, W 0.158, surplus 0.0125, E_N 0.0256; `'xp'`: N=41: rate 0.00503, W 0.00572, surplus 2.88e-05, E_N 0.00621 | N=61: rate 0.00502, W 0.0057, surplus 2.86e-05, E_N 0.00622 | N=81: rate 0.00502, W 0.00569, surplus 2.86e-05, E_N 0.00623 | N=121: rate 0.00501, W 0.00569, surplus 2.85e-05, E_N 0.00623. For `'xx'` both the
   switching work and the surplus grow linearly in ln N (slopes 0.0427 and
   0.00539 per unit ln N, largest residuals 2.7e-05 and 8.4e-05 over four
   sizes), so the exchange rate rises by +32.2 % from N = 41 to 121 and
   tends to the ratio of the two slopes, 0.126, while E_N changes by
   -1.5 %; for `'xp'`, which does not couple to the k = 0 mode, W_in changes
   by -0.6 %, the surplus by -0.9 %, E_N by +0.3 % and the
   rate by -0.3 % over the same range. **The IR-limit statement of the
   map is therefore: the `'xx'` exchange rate at the base point is an increasing
   function of the box size whose massless limit is bounded by the log-slope ratio
   0.126, and the `'xp'` rate is IR-safe.** Four chain sizes are a trend, not
   a limit theorem, and the map's other rows are quoted at the regulated mass.

**Status: admissible as a claim about the model** conditional on the integration
record below. **Not a claim about any device.**

---

## The producing build

| item | value |
|---|---|
| generator | `papers/harvest-then-teleport/notebook.py`, entry point `main()` |
| generator sha256 | `8ddcb8469f1caedd11be1027af24d8ba6619d397ae216b753d9870de3143c20a` |
| produced (UTC) | `2026-09-04T16:16:27+00:00 (system clock of the producing machine)` |
| git HEAD | `006a1f3af6bfce108815f21134a14a5c4ab199e7` (this candidate's revisions to `vacuum/composite/`, its test and this directory are uncommitted additions on top) |
| package | `vacuum` 0.1.0 (editable) |
| environment | `Python 3.13.3, numpy 2.5.2, scipy 1.18.1, macOS-14.6.1-arm64-arm-64bit-Mach-O` |
| command | `.venv/bin/python papers/harvest-then-teleport/notebook.py` |
| wall clock | 5753 s for 61 rows (slowest row `ir_size-N_field121`, 543 s). **Contended**: the producing machine was running several other agents' multi-worker jobs throughout; the identical map on a quiet tree took 1144 s, and the per-row CPU cost is the honest figure. |
| scan generator | `papers/harvest-then-teleport/scan_spacelike.py`, sha256 `b68e118b53224bdc48e5de80ac0e7b30779fa2dcd0b04559fd4bd3ac58cab336`, produced `2026-09-04T18:38:21+00:00`, 796 s |
| pytest, this candidate | **39 passed in 31 s** (`tests/test_harvest_teleport.py`, uncontended; the file's budget is 90 s) |
| pytest, full suite | see the integration record appended below by the integration pass |

The generator's sha256 and the git HEAD are embedded in every archive under `data/`
(`__build__` key / `build`). No number came from an optimiser other than the
deterministic bounded search over Alice's Gaussian measurement family and the
deterministic (s, θ) grid over squeezed number bases for the PNR column.

**Integration: full suite 1201 passed on the quiescent tree at HEAD `c18a206`, 2026-09-04** (`.venv/bin/python -m pytest -q`, twice in a row on the final tree: 1201 passed, 10 warnings in 731.91s (0:12:11) and 1201 passed, 10 warnings in 740.37s (0:12:20); with tenpy, cvxpy, jax, jaxlib, qiskit, qiskit_aer and qiskit_ibm_runtime blocked at the import system, 1026 passed, 76 skipped in 428.84s (0:07:08); all 15 notebook `--quick` paths exit 0, the slowest in 307 s). The tree is commit `c18a206` plus this pass's uncommitted edits — README, STATUS, PRIORITY, MANIFEST and draft prose reconciling the rollups to the MANIFESTs, plus one print in `interacting-vacua-first-numbers/notebook_qei_exact.py` made None-safe so its `--quick` path exits 0 (no numerical code, no data). `data/` was produced on the build recorded above; this run is the admissibility record under `papers/README.md` for the candidate's code as it stands on that tree, and a full re-run of `notebook.py` there regenerates `data/` with a new `generator_sha256`; the producing-build fields above are left as recorded.

## Modules used

| module | what it does here |
|---|---|
| `vacuum.composite.harvest_teleport` | the composite: `harvest` (through `run_protocol`, `'xx'` or `'xp'`, Bob's switching delayed by `delay`), `ebits`, `daemonic_surplus`, `optimal_measurement`, `hotta_variant`, `joint_ergotropy`, `landauer_bound`, `no_signaling_audit`, `composite_ledger` |
| `vacuum.composite.fock_conditioning` (new) | the Fock representation of the two-mode Gaussian state (`fock_tensor`: the `A = X(I − σ_Q⁻¹)` Hermite recursion), `ergotropy_finite` (passive-state construction), `fock_conditioning` (PNR and on/off on Alice, cutoff ladder started at `predicted_cutoff`), `optimal_fock_squeezing` (converged-only squeezed-basis diagnostic) |
| `vacuum.protocol` (`run_protocol`, `GeneratorStep`) | **the** code path of every harvest: switching work charged as the telescoped ⟨∂H/∂t⟩dt sum on the (2n, 2n) form for `'xp'`, ledger closure mandatory |
| `vacuum.detectors.nonperturbative` (`attach_detectors` with `coupling='xx'`/`'xp'`, `initial_state`, `detector_block`, `detector_occupations`, `harvested_log_negativity`, `run_harvesting` for the scan) | the coupled quadratic form, the λ(0)=0 vacuum with its mandatory passivity audit, the mp-margin-safe E_N |
| `vacuum.detectors.switching` | the cos² profiles (Alice's and Bob's delayed one) |
| `vacuum.core` (`harmonic_chain_K` incl. m = 0 Dirichlet, `ground_state_cov`, `symplectic_eigenvalues`, `mutual_information`, `entropy_from_nu`) | the field, the Hotta reference ground state, the entropic columns |
| `vacuum.audits` (`EnergyLedger`, `passivity_audit`, `causality_audit`, precision via the runner) | the standing audits |

`vacuum.qet.hotta` / `vacuum.qet.slp` are the conceptual anchors and are re-derived for
the Gaussian pair rather than called.

---

## Definitions and formulas (all in the module docstrings, each an anchor test)

- Conditional covariance after a pure Gaussian measurement on Alice: the Schur
  complement `V_B − Cᵀ(V_A + σ_M)⁻¹C` (Giedke–Cirac 2002; Kua–Serafini–Genoni Eq. 16),
  in the pointer frame; homodyne as its exact limit.
- `E_local = ½Tr(H_B V_B) − Ω_B ν_B`, `E_daemonic = ½Tr(H_B V_B) − Ω_B ν_(B|A)`,
  `surplus = Ω_B(ν_B − ν_(B|A))` (Kua–Serafini–Genoni Eqs. 33, 42); anchors: two-mode
  squeezed vacuum `surplus = Ω sinh²r`, squeezed-thermal Eqs. (50)/(51), to 1e-10.
- Non-Gaussian column: Fock elements of the zero-mean Gaussian state from
  `⟨m|ρ|n⟩ = d_(m,n)`, `d_(k+e_i) = (k_i+1)^(-1/2) Σ_j A_ij √k_j d_(k−e_j)`,
  `d_0 = 1/√det σ_Q`, `σ_Q = W V W† + I/2`, `A = X(I − σ_Q⁻¹)` (Quesada et al.
  2019, Eqs. 6–11, A1–A4, zero-mean case: the hafnian of the row/column-repeated
  matrix = the multidimensional Hermite polynomial); anchors: vacuum, thermal,
  TMSV amplitudes and the first/second moments against V to 1e-12. Bob's conditional
  states `ρ_B^(k) = ⟨k|ρ|k⟩_A` (PNR) and `ρ_B − ρ_B^(0)` (click); general ergotropy
  `E(ρ) = Tr(Hρ) − Σ λ_j↓ ε_j↑` (Allahverdyan–Balian–Nieuwenhuizen 2004), extended to
  unnormalised PSD ρ by homogeneity; `E_local^full = E(ρ_B)` equals the Gaussian
  `E_local` to 1e-10 on every state tried; PNR ≥ on/off (refinement + convexity);
  closed forms on the TMSV: PNR gain `Ω sinh²r`, on/off gain `Ω tanh²r`. Cutoff
  ladder 16, 24, … until every gain moves < 1e-10 (retained elements are exact, so
  the truncation error is bounded by the discarded probability times the largest
  level).
- Resource bound: `surplus ≤ k T_B I(A:B)`, `k T_B = Ω_B / ln((ν_B+½)/(ν_B−½))`. Holds on
  all 61 rows.
- Ebits: E_N in bits, the Devetak–Winter hashing lower bound, the symmetric-state EoF
  of Giedke et al. 2003 (NaN on the delayed rows, which are not mirror-symmetric).
- Hotta column: `E_A = Tr(H_A σ_M)` (= Ω_A heterodyne), `E_B = E_B^corr + E_B^inj`,
  on the passivity-audited ground state of `H_A + H_B + g x_A x_B`; `E_B ≤ E_A` checked
  on every row (True).
- `'xp'` coupling: `H_int = λ(t) x_d Fᵀ p_field` in the x–p blocks of the full quadratic
  form (`attach_detectors(..., coupling='xp')`, the d_t φ model of Teixidó-Bonfill &
  Martín-Martínez 2024, Eq. 2); the stepper, the telescoped drive work and the ledger
  take the general form, and the exit snapshot is booked under `diag(K_tot, I)` again
  because λ(t_end) = 0 (a test re-derives W_in from it).
- Windows: Alice's cos² support [0, 2T_sw], Bob's delayed by `delay`; spacelike iff
  the window length < separation; light contact iff `delay = separation`.
- IR: `mass = 0` allowed only with Dirichlet boundaries (`HarvestConfig` raises on the
  periodic massless chain: zero mode); the size sequence uses the log-linear fit
  `y = a + b ln N` on four chain sizes.

---

## Audit status

| audit | result over all 61 rows |
|---|---|
| **ledger closure** (mandatory, every level of every row; the (2n, 2n) form on `'xp'` rows) | all passed; worst defect **`5.90e-10`** (criterion as run: absolute 1e-9 — a 1.7x margin, the thinnest in the repository. The criterion has since been corrected to scale with the booked energies and entry count (`vacuum.audits.EnergyLedger.closure_tolerance`); on this composite's own protocol the roundoff floor alone is 1.4e-9 at 16386 booked operations, so the recorded defects sit two orders inside it — the thin margin was an artefact of the absolute criterion, not of the physics. No number here changes.) |
| **passivity** on the claimed T = 0 vacuum | all passed — `initial_state` and the runner's `claim_ground_entry` (thermal rows make no ground-state claim) |
| **passivity** on the Hotta reference ground state | all passed |
| **precision** on the reported detector block | all passed; `flagged` fired on no rows; mp-margin rule fired on no rows |
| **causality** of the field over the window | all passed; worst out-of-cone commutator **`6.33e-13`** |
| **no-signaling** (Bob's unconditional state rebuilt from conditional covariance + averaged displacement, five measurements) | all passed; worst deviation **`1.11e-16`** (tolerance 1e-14) |
| **Hotta inequality** E_B ≤ E_A on the reference | True |
| **Landauer bound** on the surplus | True |
| **dt-halving gate** (spec M2.2, 1e-9 on E_N, PT margin, both occupations, ν_B, ν_(B|A)) | converged on all rows (True); accepted level 5–11 (substeps per row 2048 – 131072); worst accepted movement `9.69e-10`; drive-work movement (recorded, not gating) worst `3.05e-09` |
| **Fock cutoff ladder** (every gain moves < 1e-10 under one cutoff step; the ladder starts at the occupation-predicted cutoff and the movement check still decides) | converged on all rows (True); accepted cutoffs [24, 56]; worst movement `8.6e-11`; worst trace deficit `6.6e-14` |
| **Fock–Gaussian consistency** (`E_local` from the Fock spectrum vs Eq. (3), 1e-8 relative) | True on all rows; on/off ≤ PNR on all rows: True |
| **out-of-cone scan** (32 points, gate 1e-9) | converged True; ledgers closed True; flagged rows none; measured cone leakage ≤ **2.76e-02** (close rows) and ≤ 1.85e-05 (far rows), recorded per row — the null is conservative under it |

All audits passed on every row: True. The joint-ergotropy comparison is a *reported
column*, not an audit (claim 4); the Fock columns are reported columns with their own
convergence record.

---

## Data

| file | contents |
|---|---|
| `data/rows.npz` | 61 rows × 103 columns (`__columns__`), build info (`__build__`); round-trips through `notebook.load_rows` |
| `data/summary.json` | the headline numbers, the five `'xx'` axis profiles, the saturation diagnostics, the Hotta reference, the `fock`, `xp` and `ir` sections, the audit summary, the build |
| `data/rows_quick.npz` | the four-point smoke configuration (N = 25, gate 1e-7, incl. an `'xp'` light-contact row), same schema |
| `data/xp_spacelike_scan.json` | the 32-point out-of-cone `'xp'` scan (`scan_spacelike.py`), per-point E_N, PT margin, **measured `commutator_leakage`** and `cone_margin_sites`, work, gate level, ledger verdict |

Sweep composition: base 1, coupling 6, delay 10, duration 5, gap 5, ir_mass 10, ir_size 6, separation 5, temperature 5, xp_coupling 2, xp_separation 6. Rows shared by two axes (the base point; delay 0 =
separation 12; the massless N = 41 rows) are run once and reported under both.

---

## Headline table (from `data/summary.json`)

| quantity | value | where |
|---|---|---|
| best `surplus / W_in`, `'xx'` map | **0.125** | `temperature-T0.8` (E_N = 0) |
| best `surplus / W_in`, harvesting `'xx'` rows | **0.0599** | `coupling-lam0.8` |
| best `surplus / W_in`, all rows | 0.125 | `temperature-T0.8` |
| best best-of-families rate, all rows | 0.125 | `temperature-T0.8` (family: Gaussian on every row) |
| best ebits per unit work, `'xx'` map | **1.92 bits** | `gap-gap1.5` (W_in = 0.00331) |
| best ebits per unit work, all rows | 1.92 bits | `gap-gap1.5` (W_in = 0.00331, E_N = 0.00441) |
| closest approach to Landauer | 74 % | `coupling-lam1` |
| base point (`'xx'`) | rate 0.0285, 0.779 bits/work, 0.0366 work/bit, E_N = 0.0241 nats | λ = 0.3, Ω = 1, d = 2, T_sw = 3 |
| base point (`'xp'`) | rate 0.00471, 1.36 bits/work, E_N = 0.00579 nats, W_in = 0.00613 | same parameters, derivative coupling |
| largest PNR / Gaussian ratio (squeezed basis) | 0.195 | `coupling-lam1` |
| Hotta reference E_B / E_A | 0.00312 (E_B^corr = 0.000625) | g = 0.1 |
| `'xx'` rows with E_N = 0 and surplus > 0 | 11 of 27 | separation ≥ 4, T ≥ 0.4, short/long windows, λ = 1 |
| `'xx'` massless log-slope ratio (asymptotic base-point rate) | 0.126 | N = 41 … 121 |

Measurement chosen by the Gaussian search: homodyne on 60 rows, general-dyne
on 1, heterodyne on 0; best family: gaussian.

---

## What is NOT claimed

- Nothing about a device, a rate in SI units, or a specific coupling technology.
- No "teleportation" of the surplus: it is communication-assisted extraction from
  an active state, and the draft says so in its title and first paragraph; the
  true-QET column is the teleportation analogue on a different (ground) state.
- Nothing beyond harmonic detectors, a 1+1 Dirichlet lattice, cos² switching, and
  two measurement families on Alice: the reported best of them is a lower bound on
  the unrestricted daemonic gain (an optimal POVM is not known). The squeezed-basis
  PNR diagnostic is a grid (s ∈ (1, 2, 4), four angles) restricted to bases that
  converge, not an optimum, and it interpolates to a Gaussian homodyne as s → ∞, so
  it is reported as a diagnostic and never as "non-Gaussian beating Gaussian".
- No claim that the derivative coupling harvests outside the light cone on this
  lattice: it does not on any map row nor on the 32-point scan; its advantage is
  at light contact. The scan is a lattice statement at one chain size, and its rows
  are "outside the nominal cone" with a measured leakage up to 2.76e-02, not
  strictly spacelike — the null is conservative under that leakage, never propped up
  by it.
- The IR statement is a convergence trend over m → 0 at N = 41 and four massless
  chain sizes at the base point only; the other rows are quoted at m = 0.2.
- The Hotta reference numbers are for the coupled pair's ground state; they say
  nothing about the harvest.

## Reproducing

```
.venv/bin/python papers/harvest-then-teleport/notebook.py            # ~19 min on a quiet machine, writes data/
.venv/bin/python papers/harvest-then-teleport/notebook.py --quick    # 4 points, ~7 s
.venv/bin/python papers/harvest-then-teleport/scan_spacelike.py      # 796 s, data/xp_spacelike_scan.json
.venv/bin/python -m pytest tests/test_harvest_teleport.py -q         # anchors, < 90 s
.venv/bin/python -m pytest -q                                        # admissibility gate
```
