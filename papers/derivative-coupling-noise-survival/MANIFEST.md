# MANIFEST — derivative-coupling-noise-survival

Layer 2 discovery target (PLAN.md, "Discovery targets": *whether the
derivative-coupling result — that even causally connected detectors draw their
entanglement primarily from field correlations — survives realistic noise*),
opened as a candidate directory as soon as it produced curves (papers/README.md).

---

## The claim (about the model — see "Provenance of the inputs")

For one causally connected pair of derivative-coupled (x_d–π, i.e. ∂ₜφ) harmonic
detectors on the 1+1 massless lattice field — compact cos² switchings of half-width
T = 4 meeting on the light cone (t_B − t_A = d = 12 sites), gap Ω = 0.5, oscillator
coupling λ = 0.1 (λ_UDW = 0.1) — the derivative-coupling result holds and **survives
every realistic imperfection on the swept grid that leaves any entanglement at all**:

* at T = 0 the pair harvests E_N = 6.3447e-04 nats while the pair term is
  |M_vac|/|M| = 0.9994, |M_comm|/|M| = 0.0337 — field correlations, not
  communication; the amplitude coupling at the same light contact has
  |M_comm|/|M| = 0.365 (|M_vac|/|M| = 0.931) and harvests nothing
  (N⁽²⁾ = -6.51e-03 < 0; the control row's exact E_N = 0.00e+00);
* across the noise grid — field temperature T ∈ {0, 0.02, …, 0.10} (T/Ω up to
  0.2), detector loss κ ∈ {0, 0.02} into a bath at the same T (total
  transmissivity e^{−0.4}), pure dephasing γ_φ ∈ {0, 3e-6, 1e-5, 3e-5} (integrated
  D up to 6e-4 against a T = 0 margin |M| − P = 3.4e-4) — 27 of 48 points
  keep E_N > 0 and 22 keep E_N > 1e-4 nats (**a stated model floor, not a derived
  one** — see the note below);
* on every surviving point the field-correlation share of the pair term stays
  |M_vac|/|M| ∈ [0.9994, 0.9996] and |M_comm|/|M| ∈ [0.0286, 0.0337]:
  **noise kills the negativity long before it changes where the correlation
  comes from.** The split is a function of the field temperature alone (the
  commutator part is state-independent; the anti-commutator part grows with T, and
  in fact the vacuum share *rises* slightly with T: T=0.0: 0.9994, T=0.02: 0.9994, T=0.04: 0.9995, T=0.06: 0.9995, T=0.08: 0.9995, T=0.1: 0.9996); the detector-side
  channels rescale the pair moment without re-weighting it, which the exact
  nonperturbative |m|/|M_T| column lets one check point by point
  (surviving points: |m|/|M_T| ∈ [0.830, 1.019]);
* the death lines: the clean line survives the whole temperature grid (E_N = 1.23e-4
  at T = 0.10, the top of the grid; the second-order estimate |M_T| − P_T places its
  death between T = 0.10 and 0.12), while with loss on (κ = 0.02, bath at T) it dies
  between T = 0.06 and 0.08; pure dephasing kills it at T = 0 between γ_φ = 1e-05 and 3e-05
  (γ_φ = 1e-05 (last surviving) / 3e-05 (first dead) with loss on). Loss at nbar = 0 alone never kills it — it
  attenuates m and n together — which is why the map's loss axis matters only
  through the bath occupation it carries.

**On the 1e-4 nats floor.** It is a round number chosen for this map, in lattice
units, with no device inputs; it is *not* the floor the sibling candidate later
derived. `papers/circuit-qed-noise-thresholds` derives a resolvability floor from
the Waterloo group's own tomography simulation (S. Ren, MSc thesis, Waterloo 2022,
Sec. 4.3) — **1e-3 nats**, plausible band [7.82e-5, 0.04] — and
`papers/mi-coherence-harvesting` derives per-measure Monte-Carlo floors from the
same source. Read against those numbers this map's whole grid is sub-floor: the
largest E_N anywhere on it is **6.34e-4 nats**, so **0 of 48** points are
resolvable at the derived 1e-3 nats and 23 of 48 at the band's low edge 7.82e-5.
The "22 resolvable" count above is therefore a statement at this map's own stated
level, and the *claim* of this candidate — where the entanglement comes from,
|M_vac|/|M| ≥ 0.999 on every point that carries any — does not depend on any
floor at all.

So the answer to the target question is **yes, for this model**: within the whole
region where a derivative-coupled causally connected pair still shows any
negativity under thermal field, loss and dephasing, that negativity is drawn from
field correlations at the ≥ 99.9 % level, and the relaxation of the strict
spacelike-separation requirement that TB-MM24 point to is not a vacuum-only
artifact.

**Status: a claim about the MODEL, not a device prediction.** See "Provenance of
the inputs". The producing agent did not run the full suite (see "The producing
build"); admissibility per `papers/README.md` rests on the integration record below.

---

## Where the result comes from (the literature it reproduces)

* A. Teixidó-Bonfill, E. Martín-Martínez, *Derivative coupling enables genuine
  entanglement harvesting in causal communication*, Phys. Rev. D 110, 105016
  (2024), arXiv:2406.14637 — the ∂ₜφ model (Eq. (2)), the split M = M⁺ + M⁻
  (Eq. (16)), the 1+1 light-cone cancellation of M⁻ (Eq. (27)).
* A. Teixidó-Bonfill, X. Dai, A. Lupascu, E. Martín-Martínez, arXiv:2505.01516
  (Phys. Rev. A 2026) — restates the claim for the circuit-QED (∂ₓΦ, Eq. (31))
  model; split Eq. (63).
* E. Tjoa, E. Martín-Martínez, PRD 104, 125005 (2021) — the M_vac/M_comm estimator
  (`vacuum.detectors.communication`).

The qualitative reproduction itself is a permanent test,
`tests/test_xp_coupling.py::test_derivative_coupling_entanglement_is_field_correlations_not_communication`
(measured fractions in its assertion message), and the exact engine behind every
E_N here passed its own Gate A for the derivative coupling (`tests/test_gate_xp.py`:
deviation exponent ≥ 2.005, R² ≥ 0.99998 across three setups, with canaries).

## Provenance of the inputs — READ THIS

The configuration is a **model configuration in lattice units** (N = 60 massless
Dirichlet chain; detectors at sites 24 and 36; cos² half-width 4; Ω = 0.5; λ = 0.1;
the T/κ/γ_φ grid above). It was chosen so that (i) the resonant field modes sit on
the linear part of the lattice dispersion (group velocity 0.97 — the lattice light
cone is the continuum one), (ii) the noiseless point harvests with a healthy margin
(|M| − P = 3.4e-4 at |M| = 1.29e-3), and (iii) the whole map fits a laptop budget.
**No `experiments/` parameter file is consumed** and no number here is a prediction
for a specific device. Turning this map into a device prediction means re-running
`notebook.py` on `experiments/circuit_qed_harvesting/` inputs through the
provenance-enforcing loader, as `papers/circuit-qed-noise-thresholds` does — and
inheriting that candidate's caveats about synthetic inputs. The ∂ₓΦ coupling of the
Waterloo device model (arXiv:2505.01516 Eq. (31)) is on the lattice the amplitude
coupling with a finite-difference profile (`spatial_derivative_profile`); the light-
contact test shows it behaves like the ∂ₜφ model used here (|M_vac|/|M| = 0.999),
but the map itself is the ∂ₜφ ('xp') model of TB-MM24 Eq. (2).

## What runs, and through what

| item | value |
|---|---|
| generator | `papers/derivative-coupling-noise-survival/notebook.py`, entry point `main()`; `--quick` runs the 4-point smoke path guarded by `tests/test_xp_coupling.py` |
| modules | `vacuum.detectors.nonperturbative` (`attach_detectors(coupling='xp')`, `initial_state`, `harvested_log_negativity` with the mp-margin rule), `vacuum.protocol.run_protocol` (every point; Strang channel/drive/channel segments, exponential-midpoint substeps, passivity audit on the claimed T = 0 ground state, precision audit on the reported detector block, mandatory ledger closure), `vacuum.detectors.imperfections` (`imperfections_from_rates` → `imperfection_step`; field temperature via `initial_state(T)` = `thermal_state_cov`), `vacuum.detectors.communication.communication_split(coupling='xp')` on the KMS thermal π–π kernel, `vacuum.core.mean_energy` general-H_mat path for the ledger |
| dt gate | spec M2.2 dt-halving on {E_N, PT margin, n_A, n_B} to 1e-9 (the margin is continuous, so a dead point cannot converge trivially at E_N = 0); accepted dt recorded per row (`dt_converged`, `levels_run`) |
| audits | every row: ledger closure ≤ 1e-9 (max defect 1.2e-10), passivity at T = 0, precision flag, `all_audits_passed` = True |
| data | `data/rows.npz` (all `ROW_COLUMNS` per point + JSON `meta` with the configuration and git head), `data/summary.json` (the headline numbers above), `data/rows_quick.npz` (smoke) |
| compute | 823 s wall for 49 rows on the producing machine |

## The producing build

| item | value |
|---|---|
| git head | bd846820d712 (working tree; the orchestrator commits at wave boundaries) |
| package | vacuum 0.1.0, editable install; Python 3.13.3, numpy 2.5.2 |
| tests run on this state | `tests/test_xp_coupling.py`, `tests/test_gate_xp.py`, `tests/test_nonperturbative.py`, `tests/test_udw.py`, `tests/test_gate_crossvalidation.py`, `tests/test_nogo_communication.py`, `tests/test_protocol.py`, `tests/test_imperfections.py`, `tests/test_core.py`: 288 passed in 166.15 s (2026-09-02; concurrent CPU load from other agents) |
| full known-results suite | NOT run by this agent (the coordinator's wave-boundary run is the admissibility record); per papers/README.md nothing moves from `data/` to a manuscript before that run is recorded here |

**Integration: full suite 1201 passed on the quiescent tree at HEAD `c18a206`, 2026-09-04** (`.venv/bin/python -m pytest -q`, twice in a row on the final tree: 1201 passed, 10 warnings in 731.91s (0:12:11) and 1201 passed, 10 warnings in 740.37s (0:12:20); with tenpy, cvxpy, jax, jaxlib, qiskit, qiskit_aer and qiskit_ibm_runtime blocked at the import system, 1026 passed, 76 skipped in 428.84s (0:07:08); all 15 notebook `--quick` paths exit 0, the slowest in 307 s). The tree is commit `c18a206` plus this pass's uncommitted edits — README, STATUS, PRIORITY, MANIFEST and draft prose reconciling the rollups to the MANIFESTs, plus one print in `interacting-vacua-first-numbers/notebook_qei_exact.py` made None-safe so its `--quick` path exits 0 (no numerical code, no data). `data/` was produced on the build recorded above; this run is the admissibility record under `papers/README.md` for the candidate's code as it stands on that tree, and a full re-run of `notebook.py` there regenerates `data/` with a new `generator_sha256`; the producing-build fields above are left as recorded.

## Honest limits

* One configuration, one λ, one switching shape: the map is a first real curve
  set, not a parameter study. The dephasing death line in particular depends on
  the T = 0 margin |M| − P, i.e. on λ, Ω and T_switch.
* The split fraction is the second-order (perturbative) estimator evaluated with
  the thermal field kernel; the nonperturbative engine has no M_vac/M_comm of its
  own. The |m|/|M_T| column is the bridge: it is 1.02 at the clean point and
  moves with noise exactly as a channel-rescaled pair moment should.
* 1+1 lattice only. In 3+1 TB-MM24 do not claim correlation dominance at full
  light contact (M⁻ has three peaks with zeros between them), and this repo's
  continuum backend agrees (|M_comm|/|M| ≈ 1 at full contact for both couplings);
  the claim reproduced here is the 1+1 one.
