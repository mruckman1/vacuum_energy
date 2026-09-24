# MANIFEST — differentiable-vacuum

Result candidate of Layer 6 (PLAN.md, "The differentiable vacuum"): optimized
harvesting waveforms for the circuit-QED operating point under fixed energy
budget, noise floor and signaling bound; the QET and QEI optima recovered by
the same optimizer; the vacuum-to-Bell-pair exchange rate; an inverse-design
showcase.

---

## The claim (one paragraph)

At the noise-threshold candidate's operating point (gap 5.7 GHz, 6 mm
separation, 0.3 ns cos² window, λ = 0.3, IR mass 0.2, buffer 6; ω_ref = 2π ×
7.3 GHz, v = 1.2e8 m/s), a switching waveform from the family
λ(t) = λ_max χ(t)[a₀ + Σ_k a_k cos(kπu/T) + b_k sin(kπu/T)] (cos² envelope,
k ≤ 3, the SAME compact window) optimized by L-BFGS-B on the JAX gradient
with **equal ledger drive work**, the **same noise floor** and a
**communication fraction |M_comm|/|M| not above the published protocol's
0.215923** (Tjoa–Martín-Martínez split on the same
kernel, shape and smearing) reaches, after the plain-numpy round trip
(mandatory passivity audit, dt-halving gate at 1e-9, closing ledger,
mp-margin rule, precision audit, numpy communication split), the harvested
log-negativity gains in the table below: the largest FINITE admissible gain is 3.730x at 30mK; at 30mK+G1+Gphi the published protocol is DEAD (E_N = 0) and the admissible waveform revives it to E_N = 6.523e-03 nats (the archive records this as an undefined 'inf' factor).
**Pivot verdict, verbatim from the archive:** gain infx at floor 30mK+G1+Gphi under fixed energy budget, noise floor and signaling bound (>= 2x).
Read against PLAN.md Layer 6: the pivot trigger (< 2x) does NOT apply: every finite admissible gain exceeds 2x.

The SAME optimization without the signaling bound (2.777x at 30mK (comm 0.617), 2.245x at vacuum (comm 0.573)) raises
|M_comm|/|M| far above the baseline's and buys negativity with field-mediated
signaling; those waveforms are inadmissible under the bound and are archived
only as the explanation of where unconstrained "gains" come from at causal
separation.

**Superseded by Result 1b:** the gains in the paragraph above are the best of TWO starts.
The global search of Result 1b beats every one of them under the same three constraints
and the same round trip — 30mK 3.730 -> **4.222**; vacuum 2.892 -> **3.183**; 30mK+G1+Gphi 6.523e-03 -> **7.661e-03** nats (revival) — so those are the numbers a
manuscript should carry; the pivot verdict (≥ 2×) is unchanged and strengthened.

**The three Layer-6 open items this build closes.** (i) *Best of two starts.* A
global search per floor in the k ≤ 5 family (32 starts configured; 9–10 actually run
before this run's 720 s per-floor wall-clock cap, on a shared machine), with basins,
the raised dt gate and a round trip on every candidate above the running best. It
answers "are the archived optima the best this family offers?" with **no** on all three
floors, and it bounds the landscape from below rather than certifying optimality
(Result 1b). (ii) *The twin sat at the superseded v1
base point.* A SECOND twin is added at the shipped Table-I-scenario-2 dense-map base
point, every input re-read from `experiments/` at run time, and optimized at the device
conditions where the map's verdict is NO-GO (Result 4). (iii) *The signaling proxy is
second order.* An exact, nonperturbative field-state / detector-zero-point decomposition
of the pair correlation is compared with the proxy on every final waveform (Result 5).

**Status.** Admissible as a claim about the MODEL (the digital twin at the
noise-threshold candidate's superseded v1 base point — 5.7 GHz, 6 mm, 0.3 ns,
λ = 0.3 — NOT the shipped Table-I-scenario-2 dense map, whose max E_N is 2.95e-3
and which contains no 0.010093 row; the synthetic-input caveats apply verbatim:
window, separation, coupling, IR mass and buffer are declared simulation
choices; 30 mK, 8.7 MHz and 1e7 Hz are the device paper's fridge base,
minimum Γ₁ and plot-digitized minimum Γ_φ). NOT a claim about the device.

---

## The producing build

| item | value |
|---|---|
| generator | `papers/differentiable-vacuum/notebook.py`, entry point `main()` |
| generator sha256 | `4531c73bf535c1812b396ff41c28ee89ac698566ef8577b00a00f4fce18adf4b` |
| build (`git rev-parse HEAD`) | `d248e3c5ab69093a2f7a3aada8574222e64ca01e` (tree dirty: True — this pass's Layer-6 files were uncommitted at production, as recorded below) |
| produced (UTC) | 2026-09-04T15:07:04Z |
| environment | Python 3.13.3, numpy 2.5.2, scipy 1.18.1, jax 0.11.1 (CPU, float64), macOS-14.6.1-arm64-arm-64bit-Mach-O |
| command | `.venv/bin/python papers/differentiable-vacuum/notebook.py --workers 4` |
| wall clock | 4904 s (of which 1608 s is the global search + the twin, on 4 worker processes; the machine was shared with another candidate's 13-worker run) |
| pytest, full suite (known-results anchors + standing audits + Layer 6), same code state | **NOT RUN on this build.** The last full-suite record is the integration pass's (1005 passed twice on the quiescent tree at `fe29006`); the Layer-6 numbers below were produced by a pass whose remit was its own modules, and the tree carried several other candidates' concurrent edits (see `git status` in the note under this table). **The `papers/README.md` admissibility gate is therefore NOT met for the new Results 1b/4/5 on the producing build** — it is met by the integration record below, a green full suite on the quiescent tree that contains them. |
| pytest, this layer (`tests/test_opt_*.py`) | **49 passed in 140.32s (0:02:20) (`.venv/bin/python -m pytest tests/test_opt_*.py -q`, 2026-09-04, this pass's 7 new tests in tests/test_opt_multistart.py included)** |

**Tree state at production (2026-09-04).** The Layer-6 files of this pass (`vacuum/opt/multistart.py`, the `vacuum/opt/optimize.py` additions, `tests/test_opt_multistart.py`, this candidate's `notebook.py`/`data/`/`MANIFEST.md`/`draft/`) were uncommitted, and four other candidates were being edited concurrently by other agents (`papers/harvest-then-teleport`, `vacuum/composite`, `vacuum/geometry`, `docs/STATUS.md`). Nothing outside this candidate's own modules was read as an input except the READ-ONLY `papers/circuit-qed-noise-thresholds/data/summary.json` cross-check rows and the `experiments/` parameter files. The scenario-2 coupling was read at run time and its VALUE (0.14142135623730953) is identical to the one the parameter file carries now; only its recorded 1σ spread was refined afterwards by the concurrent coupling-derivation pass.

Round-trip rule (papers/README.md): every θ★ above was re-run through the numpy
stack; JAX values appear in the archive only as `*_jax` / `comm_proxy`
diagnostics. Equal work is finished on the gated ledger (λ_max rescaled by
√(W_budget/W_gated), ≤ 2 re-projections, recorded per row as
`gated_reprojections` / `work_over_budget`); the equal-work check tolerance is
the work column's gate uncertainty (1e-4 relative, `WORK_CHECK_RTOL`). No audit
anomaly was encountered.

---

**Integration: full suite 1201 passed on the quiescent tree at HEAD `c18a206`, 2026-09-04** (`.venv/bin/python -m pytest -q`, twice in a row on the final tree: 1201 passed, 10 warnings in 731.91s (0:12:11) and 1201 passed, 10 warnings in 740.37s (0:12:20); with tenpy, cvxpy, jax, jaxlib, qiskit, qiskit_aer and qiskit_ibm_runtime blocked at the import system, 1026 passed, 76 skipped in 428.84s (0:07:08); all 15 notebook `--quick` paths exit 0, the slowest in 307 s). The tree is commit `c18a206` plus this pass's uncommitted edits — README, STATUS, PRIORITY, MANIFEST and draft prose reconciling the rollups to the MANIFESTs, plus one print in `interacting-vacua-first-numbers/notebook_qei_exact.py` made None-safe so its `--quick` path exits 0 (no numerical code, no data). `data/` was produced on the build recorded above; this run is the admissibility record under `papers/README.md` for the candidate's code as it stands on that tree, and a full re-run of `notebook.py` there regenerates `data/` with a new `generator_sha256`; the producing-build fields above are left as recorded.

## Anchors (recomputed by the generator; the permanent versions are the tests)

| anchor | measured | tolerance | test |
|---|---|---|---|
| forward agreement, Gate-A setup A: max dV / dW / dE_N | 1.1e-14 / 2.1e-17 / 3.2e-15 | 1e-10 | test_opt_gjax_detectors |
| gradient vs Richardson FD (λ_max, gap) through the full protocol | 1.3e-10 / 1.8e-10 rel | 1e-6 | test_opt_gjax_detectors |
| PKMM re-discovery: interior gap optimum ΩT (d/T = 6, Δ/T = 3, σ/T = 0.02), Nelder-Mead from ΩT = 9 | 4.773975 (bracketed 4.773975) | 1e-4 | test_opt_objectives |
| PKMM re-discovery: Fig. 2 b.1 death-line point, ΩT = 9.1, from d/T = 5 | d★/T = 9.80944 vs digitized 9.803 (rel 6.6e-04) | 2 % | test_opt_objectives |
| Hotta angle, worst |θ★ − closed form| over 4 (h, k) | 1.5e-13 | 1e-8 | test_opt_objectives |
| QEI ascent (N = 161): τ₀★ / ratio vs numpy `optimize_sampling` | 5.4889 / 0.988599 vs 5.4824 / 0.988599 | 5e-2 / 1e-6 | test_opt_objectives |
| distillation: coded 16×16 rounds vs BBPSSW Eq. (7) and DEJMPS Eq. (7) | 3.3e-16 | 1e-12 | test_opt_distill |
| inverse design: critical Dirichlet chain (N = 64) recovered from its ⟨x x⟩ | residual 8.8e-13, max dK 9.9e-11 | 1e-8 | test_opt_inverse_design |

---

## Result 1 — waveform gains at the operating point (round-trip numbers)

Baseline = the published cos² protocol at θ₀ through the identical round trip
(its 30 mK E_N reproduces the noise-threshold archive row `A-base-T30mK`,
0.010093115, and its vacuum row 0.014396543). Budget = baseline ledger work.

| floor | variant | E_N base | E_N opt | gain | W base / W opt | comm base -> opt | gate level / converged | audits | budget / signaling check | admissible | start |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30mK | constrained | 1.0093e-02 | 3.7648e-02 | 3.730 | 4.4387e-02 / 4.4387e-02 | 0.2159 -> 0.2140 | 15 / True | all pass | True / True | YES | warm |
| 30mK | unconstrained | 1.0093e-02 | 2.8033e-02 | 2.777 | 4.4387e-02 / 4.4387e-02 | 0.2159 -> 0.6168 | 15 / True | all pass | True / False | no | published |
| vacuum | constrained | 1.4397e-02 | 4.1640e-02 | 2.892 | 3.5440e-02 / 3.5440e-02 | 0.2159 -> 0.2156 | 16 / True | all pass | True / True | YES | warm |
| vacuum | unconstrained | 1.4397e-02 | 3.2319e-02 | 2.245 | 3.5440e-02 / 3.5440e-02 | 0.2159 -> 0.5727 | 15 / True | all pass | True / False | no | published |
| 30mK+G1+Gphi | constrained | 0.0000e+00 | 6.5235e-03 | revived (base 0) | 4.4157e-02 / 4.4157e-02 | 0.2159 -> 0.2157 | 16 / True | all pass | True / True | YES | warm |
| 30mK+G1+Gphi | unconstrained | 0.0000e+00 | 9.6060e-03 | revived (base 0) | 4.4157e-02 / 4.4158e-02 | 0.2159 -> 0.5890 | 16 / True | all pass | True / False | no | published |

`gain_per_work`, every per-row flag (`flagged`, `regime`, `min_nu_pt`,
`dt_converged`), the per-start trials and the optimizer messages are in
`data/summary.json`; the θ★ vectors and detector blocks V_AB in
`data/waveforms.npz`.

---

## Result 1b — is the archived optimum the global optimum? (the global search)

The archived rows of Result 1 were the **best of two starts**.
`vacuum.opt.multistart` runs **32 starts per floor** in the **k ≤ 5 harmonic
family**: the published protocol, the archived optimum embedded in the larger family,
2 cross-entropy runs seeded on the same penalized twin loss (best sample and fitted
mean of each), and random perturbations of the shape at scales
0.1, 0.2, 0.4. Budget-projected waveforms are clustered into
**basins** by relative L² distance (tolerance 0.05, sign canonicalized — the 'xx'
model is invariant under λ → −λ), the top 3 basin representatives get a longer
polish, and **every basin representative whose twin negativity exceeds the running best
round-tripped admissible E_N is round-tripped** (the running best is seeded with the
archived value). The dt gate keeps the archive's own 1e−9 — never weakened — with the
**cap raised 16 → 18**; the accepted level and the movement at acceptance are recorded
per round trip. Gains are factors over the same floor's published baseline.

| floor | archived E_N | archived gain | best E_N | best gain | verdict | starts run | basins (ok / all) | twin gains best / median / spread | round-trip gains best / median / spread | round trips / admissible | gate level / movement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30mK | 3.7648e-02 | 3.730 | 4.2611e-02 | 4.222 | **beaten** | 10 / 32 (capped) | 11 / 12 | 4.222 / 2.513 / 2.649 | 4.222 / 4.222 / 0.000 | 1 / 1 | 16 / 3.8e-10 |
| vacuum | 4.1640e-02 | 2.892 | 4.5824e-02 | 3.183 | **beaten** | 10 / 32 (capped) | 12 / 12 | 3.183 / 2.055 / 2.330 | 3.183 / 3.183 / 0.000 | 1 / 1 | 16 / 4.1e-10 |
| 30mK+G1+Gphi | 6.5235e-03 | revived | 7.6607e-03 | revived | **beaten** | 9 / 32 (capped) | 9 / 10 | 7.659e-03 / 0.000e+00 / 7.659e-03 nats | 7.661e-03 / 7.661e-03 / 0.000e+00 nats | 1 / 1 | 16 / 3.2e-10 |

Per-floor verdicts, verbatim from the archive (`headline.global_search.<floor>.verdict`):

- **30mK** — BEATEN: an admissible k <= 5 waveform reaches E_N 4.26112e-02 (archived 3.76484e-02, +13.18 %, above the 0.0001 equal-work tolerance) in basin 0 (archived+polish)
- **vacuum** — BEATEN: an admissible k <= 5 waveform reaches E_N 4.58238e-02 (archived 4.16395e-02, +10.05 %, above the 0.0001 equal-work tolerance) in basin 0 (random0:s0.1+polish)
- **30mK+G1+Gphi** — BEATEN: an admissible k <= 5 waveform reaches E_N 7.66069e-03 (archived 6.52345e-03, +17.43 %, above the 0.0001 equal-work tolerance) in basin 0 (archived+polish)

An improvement is only *called* one when it exceeds the **1e−4 equal-work tolerance**
(`WORK_CHECK_RTOL`, the work column's own gate uncertainty): two rows whose ledger work
agrees with the budget to that margin are not distinguishable below it. Every trial,
every basin with its member starts, every round trip and the full search configuration
are in `data/summary.json → global_search`; where a floor is beaten the new θ★ and
V_AB are archived in `data/waveforms.npz` under `<floor>__global__*`.

The `30mK+G1+Gphi` row has no gain *factor* — its published baseline is exactly 0 — so
its two statistics columns are E_N in **nats**, not multiples (`gains_are_factors` is
false for that floor in the archive).

**A multistart is evidence about the landscape, not a proof of global optimality**: one
waveform family (cos² envelope × k ≤ 5 harmonics on a fixed window), one basin metric,
and a wall-clock cap per floor (720 s) that this run HIT on every floor —
10, 10, 9 of 32 starts were run before the cap
(`time_limited` is recorded per floor; the machine was shared with another candidate's
13-worker run). The reported basin counts and spreads are therefore lower bounds on what
32 starts would have found.

---

## Result 4 — the scenario-2 twin: the optimizer against the noise map's verdict

The noise-threshold candidate's shipped claim is that at **Table I scenario 2** the
device point is **NO-GO at every resolution floor**. That claim was established for ONE
waveform — the published cos². This twin asks the optimizer's question of it, at the
**shipped dense-map base point** (NOT the superseded v1 point of Results 1–3):

| input | value | source, read at run time |
|---|---|---|
| gap | 7.3 GHz, shifted -0.46 GHz along the envelope (γ = 0.02) | `circuit_qed_harvesting/teixido-bonfill_2026_table1`: `qubit_gap_frequency`, `gap_variation_coefficient` × `table1_scenario_gamma[1]` (Eq. 30) |
| window | 0.22 ns cos² (10.0908 lattice) | `circuit_qed_harvesting/teixido-bonfill_2026_table1`: `switching_duration_cosine_ramps_by_scenario[1]` |
| separation | 19.2 mm = 7 sites (t_d = 0.16 ns) | `circuit_qed_harvesting/teixido-bonfill_2026_table1_variant-switching`: `detector_separation_marginal` |
| coupling λ_osc | 0.141421 ± 0.006769 | circuit_qed_harvesting/teixido-bonfill_2026_table1:oscillator_coupling_lambda_osc_scenario2 (derived; derived_from scenario2_coupling_lambda_tb, qubit_gap_frequency) |
| device conditions | 30.0 mK, Γ₁ = 8.7 MHz, Γ_φ = 1e+07 Hz | `circuit_qed_harvesting/janzen_2023_fig6` (Γ_φ plot-digitized) |
| resolution floor | 0.001 nats (band 7.82e-05–0.04) | `circuit_qed_harvesting/teixido-bonfill_2026_table1`: `negativity_resolution_nats` |
| IR mass / buffer | 0.2 / 6 sites | SYNTHETIC (the map's declared choices) |

Every input is **re-read from `experiments/` at run time**, so a re-derived coupling
propagates without editing this candidate; `--lambda-osc X` overrides it and is recorded
as an override in `scenario2_twin.operating_point.lam_source`. Before anything is
optimized the twin is cross-checked against the shipped archive (READ ONLY):

| cross-check | this twin | shipped map | rel. dev. |
|---|---|---|---|
| T = 0, shifted gap (its finite-size control row, buffer 6) | 1.009064e-03 | 1.009064e-03 | 1.10e-13 |
| T = 0, static gap (its `split-X7` row) | 7.030009e-04 | 7.030009e-04 | 1.90e-12 |
| static-gap communication fraction | 0.292560 | 0.292560 | 5.6e-17 |
| coupling equals the map's | 0.141421 | 0.141421 | yes |

**Before / after at the device conditions, at equal work:**

| row | E_N | work | comm. fraction (static-gap sibling) | ν̃_PT min | gate level | admissible |
|---|---|---|---|---|---|---|
| published cos² (the map's waveform) | 0.0000e+00 | 2.2306e-03 | 0.2926 | 0.513385 | 12 | baseline |
| best optimized | n/a | n/a | n/a | n/a | n/a | n/a |

**Verdict, verbatim from the archive:** NO-GO: every admissible round-tripped candidate stays dead (top round trip E_N 0.000e+00, nu~_min 0.513368, work 2.2306e-03, static-gap communication fraction 0.2892 <= bound 0.2926); the device point stays at E_N = 0.000e+00 (nu~_min 0.513385)

Search detail: 12 trials from 9 starts (of
16 configured, wall-clock capped) in
9 bound-respecting basins; 1 round trip(s),
1 admissible, gate cap 18, accepted level
12, movement at acceptance 5.4e-10 (< 1e−9).
Across the whole search the smallest partial-transpose symplectic eigenvalue moved only
from 0.513385 (published cos²) to 0.513368 (best round trip) — it never
approaches 1/2 from below, so E_N = 0 here is a **clamp with a wide margin**
(1.34e-02 and 1.34e-02 above the floor), not a near miss.
**The map's NO-GO verdict at this point is not an artifact of its waveform choice.**

The signaling column carries a stated caveat: the perturbative M2.5 split exists for
**static gaps only**, so the in-loop proxy and the numpy check are taken on the
static-gap, T = 0 sibling of the same waveform — for the baseline and every candidate
alike (`scenario2_twin.signaling_note`). The shifted-gap device rows carry the
nonperturbative decomposition of Result 5 instead.

Whatever its sign, this verdict is about the **model** — one whose coupling, IR mass and
boundary buffer are the noise-threshold candidate's declared inputs. It is not a
statement about the device.

---

## Result 5 — the signaling proxy is second order: how far off is it?

The open item: the in-loop proxy and the round-trip split are both O(λ²) ratios, while
the operating point (λ_osc = 0.3) is not perturbative. The Gaussian evolution supplies a
**nonperturbative counterpart that needs no perturbative state at all**
(`vacuum.opt.multistart.nonperturbative_split`): the 'xx' protocol is an affine map
V_out = S V Sᵀ + Y and the initial state is a product of the field state and the detector
ground states, so the detectors' output cross-covariance splits **exactly** into a
field-state part (the field's own correlations — Hadamard content) and a
detector-zero-point part (a detector's ground-state fluctuation carried to the other by
the retarded propagator — commutator content), plus, on noisy floors, a channel-injected
part. On the pair correlation m = ⟨a_A a_B⟩ this defines f_np = |M_signal| / |M_full|.

That it **is** the TMM21 split in the limit where TMM21 is defined (permanent anchors,
`tests/test_opt_multistart.py`): |f_np − |M_comm|/|M|| = **2.6e−4 / 2.9e−5 / 2.3e−6 /
6.4e−8** at λ_osc = 0.3 / 0.1 / 0.03 / 0.01 on the Gate-A lattice (clean λ² scaling),
additivity M_full = M_field + M_signal to 1e−14, |M_field| → |M_vac| and |M_signal| →
|M_comm| to 1 % at λ = 0.03, and **f_np < 1e−8 in a spacelike layout**.

**The one-number consistency check on the final waveforms: max |f_np − proxy| = 0.1390.**

| floor | final waveform | baseline proxy | baseline f_np (T=0) | final proxy | final f_np (T=0) | \|Δ\| | f_np at the floor, base → final | proxy / nonpert. bound holds |
|---|---|---|---|---|---|---|---|---|
| 30mK | global | 0.2159 | 0.7154 | 0.2149 | 0.1217 | 0.0932 | 0.6236 → 0.1127 | yes / yes |
| vacuum | global | 0.2159 | 0.7154 | 0.2157 | 0.1106 | 0.1051 | 0.7154 → 0.1106 | yes / yes |
| 30mK+G1+Gphi | global | 0.2159 | 0.7154 | 0.2158 | 0.0768 | 0.1390 | 0.6275 → 0.0717 | yes / yes |

Read that number for what it is. At λ_osc = 0.3 the O(λ⁴) terms are **order one**
(|M_full| is ≈ 1.9× the second-order |M|), so the proxy is **not** an estimate of the
true signaling share — it is a *constraint functional*: monotone in the right direction
and evaluated identically on the baseline and on every candidate. What survives the
correction is the **ordering**, which is the property the constraint was imposed for:
`np_bound_holds_all` = **True**, `proxy_ordering_matches_all` =
**True**.

The same waveform walking down toward the proxy as its amplitude is scaled (T = 0 floor):

| floor | λ scale | λ_max | f_np | proxy | \|Δ\| |
|---|---|---|---|---|---|
| vacuum | 0.3 | -0.06788 | 0.2037 | 0.2157 | 0.0121 |
| vacuum | 0.1 | -0.02263 | 0.2144 | 0.2157 | 0.0014 |
| vacuum | 0.03 | -0.00679 | 0.2156 | 0.2157 | 0.0001 |

Spacelike control — the same optimized waveform with the detectors moved *just* beyond
the full window (separation 15 sites vs window 13.76, a margin of only
1.24 lattice units, so this is a MARGINAL spacelike layout, not the clean one of the
unit test). The signal share falls by ~3 orders of magnitude but not to zero — the
residue is edge contact plus lattice leakage, and E_N is dead there in any case:

| floor | separation (sites) | window | f_np | E_N |
|---|---|---|---|---|
| 30mK | 15 | 13.760 | 4.87e-04 | 0.000e+00 |
| vacuum | 15 | 13.760 | 6.20e-04 | 0.000e+00 |
| 30mK+G1+Gphi | 15 | 13.760 | 2.43e-04 | 0.000e+00 |

---

## Result 2 — the vacuum-to-Bell-pair exchange rate

Leading-order qubit projection of the detector block (Gate-A map; A − 1/2 =
N^(2)), Bell twirl, recurrence distillation to F ≥ 0.99; rate = expected
distilled pairs per harvesting run / ledger work.

| floor | variant | protocol | Φ⁺ fidelity in | rounds | F out | pairs per run | work | rate (pairs / unit work) |
|---|---|---|---|---|---|---|---|---|
| 30mK | baseline | DEJMPS | 0.50500 | 12 | 0.9973 | 8.669e-07 | 4.4387e-02 | 1.953e-05 |
| 30mK | baseline | BBPSSW | 0.50500 | 32 | 0.9902 | 6.192e-16 | 4.4387e-02 | 1.395e-14 |
| 30mK | constrained | DEJMPS | 0.51753 | 10 | 0.9986 | 1.528e-05 | 4.4387e-02 | 3.441e-04 |
| 30mK | constrained | BBPSSW | 0.51753 | 26 | 0.9929 | 2.053e-12 | 4.4387e-02 | 4.625e-11 |
| 30mK | unconstrained | DEJMPS | 0.51341 | 10 | 0.9987 | 1.336e-05 | 4.4387e-02 | 3.011e-04 |
| 30mK | unconstrained | BBPSSW | 0.51341 | 27 | 0.9915 | 4.475e-13 | 4.4387e-02 | 1.008e-11 |
| vacuum | baseline | DEJMPS | 0.50710 | 10 | 0.9939 | 8.022e-06 | 3.5440e-02 | 2.264e-04 |
| vacuum | baseline | BBPSSW | 0.50710 | 31 | 0.9932 | 3.746e-15 | 3.5440e-02 | 1.057e-13 |
| vacuum | constrained | DEJMPS | 0.51924 | 9 | 0.9964 | 4.286e-05 | 3.5440e-02 | 1.209e-03 |
| vacuum | constrained | BBPSSW | 0.51924 | 25 | 0.9913 | 5.512e-12 | 3.5440e-02 | 1.555e-10 |
| vacuum | unconstrained | DEJMPS | 0.51552 | 9 | 0.9964 | 3.948e-05 | 3.5440e-02 | 1.114e-03 |
| vacuum | unconstrained | BBPSSW | 0.51552 | 26 | 0.9908 | 1.418e-12 | 3.5440e-02 | 4.001e-11 |
| 30mK+G1+Gphi | baseline | DEJMPS | 0.48982 | 0 | 0.4898 | 0.000e+00 | 4.4157e-02 | 0.000e+00 |
| 30mK+G1+Gphi | baseline | BBPSSW | 0.48982 | 0 | 0.4898 | 0.000e+00 | 4.4157e-02 | 0.000e+00 |
| 30mK+G1+Gphi | constrained | DEJMPS | 0.50200 | 16 | 0.9945 | 4.273e-09 | 4.4157e-02 | 9.676e-08 |
| 30mK+G1+Gphi | constrained | BBPSSW | 0.50200 | 37 | 0.9901 | 1.033e-18 | 4.4157e-02 | 2.339e-17 |
| 30mK+G1+Gphi | unconstrained | DEJMPS | 0.50391 | 14 | 0.9957 | 6.304e-08 | 4.4158e-02 | 1.428e-06 |
| 30mK+G1+Gphi | unconstrained | BBPSSW | 0.50391 | 34 | 0.9924 | 7.011e-17 | 4.4158e-02 | 1.588e-15 |

---

## Result 3 — inverse design

Free chain N = 16, m = 0.5 (Dirichlet), blocks A = [3, 4, 5], B = [7, 8, 9]; banded
ansatz K = L Lᵀ + εI (band 1), regularized toward the free chain (reg 1e-3).

| target | E_N free | target | achieved | residual | min eig K | max |ΔK| | v_LR bound | audits |
|---|---|---|---|---|---|---|---|---|
| x1.5 | 0.05235 | 0.07853 | 0.07852 | 9.9e-06 | 0.188 | 0.070 | 0.932 | {'passivity': True, 'precision': True, 'causality': None} |
| x2.0 | 0.05235 | 0.10470 | 0.10469 | 9.3e-06 | 0.118 | 0.105 | 1.051 | {'passivity': True, 'precision': True, 'causality': None} |
| x3.0 | 0.05235 | 0.15705 | 0.15705 | 5.1e-06 | 0.042 | 0.136 | 1.066 | {'passivity': True, 'precision': True, 'causality': None} |

(Causality audit: None = not applicable, the N = 16 chain is too small for a
non-vacuous light-cone check; the velocity bound is stated instead.)

---

## Modules used

`vacuum.opt.gjax` (differentiable core), `vacuum.opt.gjax_detectors` (twins of
the 'xx' detector stack + the fixed-grid second-order communication split),
`vacuum.opt.objectives` / `vacuum.opt.optimize` (objectives, constraints,
drivers, equal-work projection, the public `make_eval` / `make_value`),
`vacuum.opt.multistart` (global search: starts, basins, the raised gate, the
equal-work round trip, the nonperturbative communication decomposition),
`vacuum.opt.rl` (the cross-entropy seeding), `vacuum.opt.distill` (BBPSSW/DEJMPS),
`vacuum.opt.inverse_design`; round trips through `vacuum.detectors`
(`initial_state`, `protocol_evolve`, `harvested_log_negativity`,
`pair_state_with_split`), `vacuum.audits`, `vacuum.qet.hotta`,
`vacuum.inequalities.qei`; units through `vacuum.experiments_io` from
`experiments/circuit_qed_harvesting/teixido-bonfill_2026_table1.json`,
`teixido-bonfill_2026_table1_variant-switching.json` and `janzen_2023_fig6.json`
(the scenario-2 twin reads its whole base point from these at run time).

## Limitations and unresolved problems

- Oscillator detectors, 'xx' coupling, unit-spacing UV cutoff, synthetic IR
  mass (inherited); the signaling proxy and the round-trip split are second
  order in the coupling (a ratio, coupling-independent at leading order).
- The waveform family cannot move the support: the causal status of the
  layout is fixed by configuration; the signaling bound is what keeps the
  optimizer from exploiting the retarded field signal.
- The penalized landscape is multimodal: from the published start the vacuum
  floor stalls near 1.3× (per-start trials in the archive); the Result-1
  numbers are the best of two starts (published, warm). Result 1b replaces
  the "not round-tripped" caveat with a measured statement — 32 starts,
  basins counted, every candidate above the running best round-tripped —
  but a multistart bounds the landscape from below; it does not certify
  global optimality, and it explores ONE waveform family on a fixed window.
- Optimized waveforms need finer dt than the baseline (gate levels 15–16 in
  Result 1); an unconverged row is inadmissible by construction. Result 1b
  raises the cap to 18 and records the accepted level and the movement at
  acceptance per round trip, so "converged" is a number in the archive.
- The scenario-2 twin inherits every synthetic input of the noise-threshold
  candidate (IR mass and boundary buffer are declared choices; the coupling's
  status is whatever the parameter file says at run time), and its signaling
  column is the static-gap sibling's, not the shifted-gap run's — the
  perturbative split has static gaps only.
- The nonperturbative decomposition of Result 5 is exact for the Gaussian
  'xx' evolution and reduces to TMM21 as λ → 0, but it is a statement about
  the pair correlation ⟨a_A a_B⟩, not a full nonperturbative harvesting /
  signalling separation of E_N; and at the operating point it disagrees with
  the second-order proxy by 0.14, which is why the proxy is used as a
  constraint functional and not reported as a signaling share.
- The energy check's 1e-4 tolerance is the work column's own gate
  uncertainty; the gate controls E_N and ⟨n_d⟩, not the drive work.
