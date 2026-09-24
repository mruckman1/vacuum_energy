# MANIFEST — `geometry-from-entanglement-failure-map`

## The claim

**Prior art (literature check, 2026-09-23; `docs/LITERATURE_CHECK_2026-09-23.md`).** The off-criticality half of this map is
known: Leighton-Trudel, arXiv:2507.09749 (2025), shows that a mutual-information distance in
one-dimensional chains is a metric at criticality (power-law mutual information, exponent ≤ 2) and
fails the triangle inequality in gapped phases (exponential clustering), proved under stated
scaling conditions and checked exactly in the transverse-field Ising chain. The quantified
crossover below (correlation length ≈ 13–15 sites, the curvature radius) measures that phenomenon
on the harmonic chain. The disorder half was not found among the 397 papers citing
Cao–Carroll–Michalakis or Qi.

Distances defined from mutual information, d = −ln(I/I_max), reconstruct on the
critical Ising (Kitaev) chain the hyperbolic geometry Swingle's MERA/AdS map
predicts — d = 2.06 ln r + const (4Δ_ψ = 2), a stress-minimizing H² embedding
(stress 0.060, curvature radius 1.37) beating the flat one (0.108), and d_MI linear
in the MERA graph geodesic (R² 0.95) — and this reconstruction has a **quantifiable
failure boundary** on vacua that are not holographic by construction. Coming down
from criticality the log distance law gives way to a linear one at a correlation
length ξ_× ≈ 0.3 L_probe (detuned Ising, both phases: ξ_× = 13.0 and 14.6 for a
probe of L = 46 sites) or ≈ 0.9 L_probe (massive boson, ξ_× = 40), while the fitted
curvature radius rises monotonically as ξ falls (log–log correlation −0.60 / −0.87;
1.4 at criticality, ≈ 4 at ξ ≈ 6, > 10³ at ξ ≈ 3) with zero translation-invariance
scatter: the geometry **degrades with structure**, at Swingle's factorization
scale. Disorder in the critical Ising couplings degrades it **into noise**:
translation invariance is lost pair by pair (scatter > 0.1 at W ≈ 0.25), Gromov
δ/diam triples (> 0.2 at W ≈ 0.30), the log fit fails (R² < 0.9 at W ≈ 0.30) and no
flat limit appears (curvature radius 1.1–1.5 up to W = 0.7). Only when correlations
fall below the 1e-12 floor (ξ ≲ 2–3) do the clean axes turn noise-like, which is
reported as the precision artifact it is. The pivot-trigger question of PLAN.md
Layer 4 is therefore answered: **both, on different axes, and distinguishably** —
the failure boundary is a line in the (ξ/L, W) plane whose two edges have
different character. The small-mass harmonic chain is the recorded exception: its
zero-mode-dominated single-site MI is nearly immune to spring disorder
(scatter 0.1 only at W ≈ 0.85).

## Modules and inputs used

- `vacuum.geometry.migeometry` — `mi_matrix`, `mi_distance`, `reconstruct_metric`
  (flat / hyperbolic), `hyperbolicity`, `distance_law`, `geometry_report`,
  `failure_map` (families `massive_boson`, `kitaev_tfim`, `disordered_kitaev`,
  `disordered_boson`)
- `vacuum.geometry.ising_exact` — infinite-chain Majorana correlators
  (closed form at g = 1, quadrature otherwise), finite disordered Kitaev chains
- `vacuum.geometry.mera` — the cached χ = 8 Z₂-symmetric ternary MERA
  (`vacuum/geometry/data/mera_ising_chi8.npz`) for graph geodesics and bond-count
  minimal cuts
- `vacuum.core` — `harmonic_chain_K`, `ground_state_cov`, `mutual_information`
- **No `experiments/` files.** External inputs are analytic only, cited at the point
  of use: Pfeuty, Ann. Phys. 57, 79 (1970) (e₀ = −4/π, ξ = 1/|ln g|); Swingle,
  PRD 86, 065007 (2012), Secs. IV–V; Evenbly–Vidal, PRB 79, 144108 (2009);
  Pfeifer–Evenbly–Vidal, PRA 79, 040301(R) (2009), Eqs. 9 and 16.

## Files

```
notebook.py                     re-runnable top to bottom, no hidden state (32 s)
data/build_info.json            the build the numbers came from
data/s1_critical_reference.{npz,json}  critical MI matrix, MERA geodesics, cut counts,
                                the d_MI vs geodesic fit
data/s2_mass_axis.csv           15 masses x every diagnostic (+ xi, xi/L)
data/s3_detuning_axis.csv       25 fields g in [0.5, 2] x every diagnostic
data/s4_disorder_kitaev.csv     9 strengths x 6 seeds (mean, std)
data/s4_disorder_boson.csv      9 strengths x 6 seeds (mean, std)
data/summary.json               crossovers, thresholds, the pivot answer
draft/outline.md                the manuscript outline
```

Regenerate everything with:

```
.venv/bin/python papers/geometry-from-entanglement-failure-map/notebook.py
```

## Build and audit status

- Produced from build `git rev-parse HEAD` = **6634641699ca** (recorded in
  `data/build_info.json`; the working tree carries uncommitted edits by several
  concurrent agents, so the hash identifies the base, not the exact tree).
- Validation anchors of this layer (`tests/test_geometry_mera.py`,
  `test_geometry_mi.py`, `test_geometry_happy.py`, `test_geometry_jacobson.py`):
  MERA energy 8.9e-6 relative; Δ_σ = 0.1238, Δ_ε = 1.0014, c = 0.5037; block
  entropies to 0.05%; HaPPY RT/greedy/reconstruction; exact Ising reference to
  1e-14 against ED; first-law residual O(ε²).
- Standing audits: this candidate evaluates ground-state correlations only — no
  protocol, no energy ledger, no channel; passivity/causality are not exercised.
- Full-suite record (admissibility rule, papers/README.md):
  `.venv/bin/python -m pytest tests -q` on 2026-09-02 (working tree at base
  6634641699ca + this candidate + other agents' uncommitted checkpoints):
  **831 passed, 0 failed, 5 warnings in 407.63 s**; the four geometry files
  take ~15 s together. HEAD has since moved to 7393cde40dcb (checkpoints of
  other agents' finished work); the notebook was not re-run on that commit.

**Integration: full suite 1201 passed on the quiescent tree at HEAD `c18a206`, 2026-09-04** (`.venv/bin/python -m pytest -q`, twice in a row on the final tree: 1201 passed, 10 warnings in 731.91s (0:12:11) and 1201 passed, 10 warnings in 740.37s (0:12:20); with tenpy, cvxpy, jax, jaxlib, qiskit, qiskit_aer and qiskit_ibm_runtime blocked at the import system, 1026 passed, 76 skipped in 428.84s (0:07:08); all 15 notebook `--quick` paths exit 0, the slowest in 307 s). The tree is commit `c18a206` plus this pass's uncommitted edits — README, STATUS, PRIORITY, MANIFEST and draft prose reconciling the rollups to the MANIFESTs, plus one print in `interacting-vacua-first-numbers/notebook_qei_exact.py` made None-safe so its `--quick` path exits 0 (no numerical code, no data). `data/` was produced on the build recorded above; this run is the admissibility record under `papers/README.md` for the candidate's code as it stands on that tree, and a full re-run of `notebook.py` there regenerates `data/` with a new `generator_sha256`; the producing-build fields above are left as recorded.
