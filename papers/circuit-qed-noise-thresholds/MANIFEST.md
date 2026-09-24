# MANIFEST — circuit-qed-noise-thresholds

Result candidate #1 of Layer 2 (docs/PLAN_LAYERS_2_3.md, milestone **M2.7**),
second version: the 1D scans of the first version are replaced by dense 2D
go/no-go **surfaces** at a **sourced** operating point, with the resolvability
floor read from a parameter file instead of assumed. **Fifth pass (2026-09-04, the UV cutoff):** the last unmeasured
input is measured — the lattice spacing is varied at fixed physical geometry so
the model's Brillouin cutoff sweeps **14.7 → 73 GHz**, bracketing the device's
own 50 GHz line cutoff, and the `'xp'` device anchors and surface are re-run at
every cutoff (Deliverable 8). **Fourth pass (2026-09-04, the proposal's
operator):** the map is re-run with the **derivative coupling** the proposal
actually uses (`'xp'`) — the model-level blocker the derivation identified: the
main (T × Γ_φ) surface at the derived coupling and ±1σ, the device-point rows,
the derivative-kernel split companions and the finite-size controls
(Deliverable 5), plus a gap-tracking variant (Deliverable 6) and an **IR-mass
control** (Deliverable 7) that measures the last synthetic input whose effect
was unquantified. **Third pass
(2026-09-04, coupling derivation):** the oscillator coupling — the one input
that was a convention — is derived from the circuit in
`vacuum/detectors/circuit_mapping.py`, stored as a `derived` quantity in the
base parameter file, and the device-point rows are re-run at it, at ±1σ and
at every enumerated alternative (`--device-point`, Deliverable 4).

---

## The claim

For entanglement harvesting by two detectors in a superconducting microwave
waveguide — modelled as a 1+1 lattice field with harmonic (oscillator-UDW)
detectors, `'xx'` coupling — at the operating point of the Waterloo proposal
that lies closest to the measured device (**Table I scenario 2** of
Teixidó-Bonfill et al., PRA 113, 043732 (2026): α = 0.003, the only scenario
inside the α ∈ [6.2e-5, 2.19e-2] measured by Janzen et al. 2023; gap 7.3 GHz
shifted by −0.46 GHz along the switching envelope per Eq. (30) at γ = 0.02;
the scenario's published cosine-ramps window T = 0.22 ns (Fig. 18 caption),
which is exactly this library's cos² profile; the proposal's shortest published
light-crossing delay t_d = 0.16 ns, i.e. d = 19.2 mm; oscillator coupling
λ_osc = 0.1414, mapping band [0.1185, 0.1482] — **derived** from the scenario's
circuit parameters —
see "The coupling" — which is numerically the value the surfaces were
produced at). **Items 1–5 below are the four `'xx'` (amplitude) surfaces; the
proposal's own derivative coupling is Deliverable 5, and where the two
disagree — the temperature wall, the dephasing floor and the pivot — the
`'xp'` numbers are the ones that speak about the device:**

1. **The go/no-go boundary is a temperature wall at 27.5–30 mK on every
   surface, and the device sits on its wrong side.** Harvested log-negativity
   `E_N` dies by *sudden death* between 27.5 and 30 mK at every dephasing rate
   ≤ 5e4 Hz, every relaxation rate ≤ 200 MHz, every separation and every
   coupling on the grids. At the device's fridge base (30 mK) and assumed
   transmission-line temperature (50 mK), `E_N = 0` **exactly** at every
   measured relaxation rate, every plot-digitised dephasing rate, every sourced
   separation and the anchored coupling — NO-GO at the derived resolution
   (1e-3 nats) *and* at both edges of its plausible band [7.82e-5, 0.04].
2. **Dephasing binds two orders of magnitude below the device.** At the
   band's low edge the dephasing crossing is Γ_φ ∈ [2e5, 5e5] Hz (T ≤ 20 mK),
   20–50× below the device's plot-digitised lower end 1e7 Hz; at the derived
   floor no point of the (T, Γ_φ) surface is GO at all — even Γ_φ = 1e4 Hz
   erases the noiseless corner's 0.9 % margin. Relaxation binds only above the
   measured range: Γ₁ ∈ [2.0, 3.0] GHz at T ≤ 15 mK (band-low floor), against
   a measured maximum of 1.85 GHz.
3. **Nothing on any surface is GO at the band's high edge (0.04 nats).** The
   largest `E_N` on all four surfaces is 2.95e-3 nats (13× below 0.04).
4. **The best pocket per unit switching work** is at *weak* coupling and
   short separation: `E_N/W_drive = 4.08` nats per unit drive energy at 4 sites
   (10.5 mm), T = 0 (derived floor); 4.62 at λ = 0.05 (band-low floor).
   Efficiency and yield point in opposite directions.
5. **Pivot verdict (PLAN.md Layer 2): the trigger fires for `E_N`; mutual
   information survives the device box but is itself sub-floor.** Inside the
   device box (30/50 mK × measured/digitised rates × sourced separations),
   max `E_N` = 0 while max I(A:B) = 2.4e-4 nats — 0.24× the derived floor,
   3× the band's low edge. MI becomes resolvable at the derived floor only at
   ≥ 2× the anchored coupling (λ_osc ≥ 0.29 at 30 mK), where `E_N` is exactly
   zero: 36 % of the coupling surface is "MI-only" at 1e-3 nats.

**Status.** As a claim about the **model**: every standing audit, the dt gate,
the two-driver cross-check and the candidate's own 22 tests pass on the
producing build, but the `papers/README.md` gate — *the full known-results
suite green on the exact producing build* — is **not met**: 3 tests of an
untracked, in-progress Layer-4 MERA module fail on this worktree (see "The
producing build"). The rule admits no exception, so the model claim is
**admissible only once those failures are resolved or that module is removed
from the tree** — which the integration run recorded below did (the MERA module
landed and its tests pass); nothing in this candidate depends on it.

**Admissibility after the coupling derivation (2026-09-04).** At production
the claim was **not admissible as a claim about the device** because of one
input, the coupling: λ_osc had been anchored to the scenario's |λ| = 0.1 by
identifying the proposal's dimensionless λ with this stack's MAP-1 λ_UDW — a
convention choice, not a derivation. That input is now **derived**
(section "The coupling" below; `vacuum/detectors/circuit_mapping.py`, the
parameter file's `oscillator_coupling_lambda_osc_scenario2`, provenance
`derived`): λ_osc = |λ_TB| Ω_lat √(2Ω_lat) = **0.1414** (rounding spread
±0.0068; mapping band [0.1185, 0.1482] — see "Uncertainty" for what each
is and is not), the same
number, because the factor the identification omitted — Ω_lat = Ω⁰/ω_ref —
is unity when the lattice energy unit is the free gap. The device-point rows
were re-run at that value, at ±1σ and at four alternatives spanning
0.118–0.148 (Deliverable 4): the device box is **NO-GO at every floor and
every coupling** (max E_N = 2.0e-7 at the lower envelope, 0 elsewhere), and
the 35 rows shared with the surfaces archive reproduce it to the last bit.
The claim is admissible **for the model at the derived coupling**, and the
fourth pass re-ran it under the proposal's own operator: see **"Admissibility
after the `'xp'` re-run"** below for the device statement the evidence now
supports. The table below is the status *before* that re-run, kept because it
records what each input contributed:

| remaining non-device input | status | can it still block a device claim? |
|---|---|---|
| the `'xx'` (amplitude) operator in place of the proposal's ∂ₓΦ coupling | model limitation 3–4; quantified once: at the same λ_UDW the perturbative companions give E_N 7.25e-4 (`'xx'`) vs 2.07e-4 (`'xp'`), a factor **3.5**, with communication fractions 0.29 vs 0.98 (`data/device_point.json → model_mismatch_companion`) | **it was the blocker — now closed.** The device's harvesting is the derivative model's; the fourth pass re-ran the map with `'xp'` (Deliverable 5), and the two operators disagree on exactly the verdicts the device claim turns on. |
| IR mass m = 0.2 (1.46 GHz) | synthetic regulator; it enters the derived coupling only through the lattice alternatives (`physical-rate`, `finite-difference`: −8 % / −2 %), which were run | it could then move the wall by an unmeasured amount — **now measured** (Deliverable 7): under `'xp'` no device-anchor verdict moves over m = 0.05–0.4, and the dephasing anchors are zero at every mass. |
| boundary buffer 6 sites | synthetic, **measured**: 0.31 % on E_N at the operating point (finite-size control) | no |
| the three file sweep grids and the four surface axes | synthetic *coordinates*; every device anchor lies on them by construction and the verdicts are grid reads | no — they cannot change a verdict at an anchor; they only fix where the contours are resolved |
| plot-digitised Γ_φ (factor ~2) | inherited hazard | the dephasing anchors are NO-GO by more than a factor 2 in Γ_φ (dephasing crossing 2e5–5e5 Hz vs digitised ≥ 1e7 Hz); cannot flip the device box, can move the 20 mK pocket's requirement |
| UV cutoff 14.6 GHz (lattice) vs the device's 50 GHz | model limitation 1 | untested; the proposal reports 20 % sensitivity to Ω_cut → ∞ for its spacelike case |
| no published error bar on γ, Z₀, v, Ω⁰ | the 1σ used is the paper's own rounding spread (4.8 %) | a group-supplied error bar widening the band past the 30 mK sudden-death cell would reopen the question; the sensitivity is on the coupling surface |

Of that list, (1) the `'xp'` re-run and (2) the IR-mass control were done in
the fourth pass (Deliverables 5–7); (3) the lattice UV cutoff remains, and is
now the single unmeasured input.

---

## Admissibility after the `'xp'` re-run (2026-09-04)

**The device statement the evidence supports.** For two detectors coupled to a
1+1 superconducting waveguide by **the proposal's own derivative coupling**, at
Table I scenario 2's published operating point (7.3 GHz gap shifted −0.46 GHz
along the published 0.22 ns cosine-ramp window, 19.2 mm separation), with the
coupling **derived** from the circuit:

> **The device's own pure dephasing, by itself, forbids resolvable
> entanglement harvesting.** At every device dephasing anchor
> (Γ_φ ∈ {1e7, 2e8} Hz, plot-digitised) the harvested log-negativity is
> **exactly zero** — at every temperature down to T = 0, at the derived
> coupling and at both ±1σ, at all three resolution floors, and at every IR
> mass over a factor 8 (Deliverable 7).
>
> **The statement does not depend on the resolution floor at all.** E_N is not
> *small* at the device's dephasing, it is **exactly zero**, and zero is below
> any floor — so the derived floor, its band [7.82e-5, 0.04] and the whole
> Ren-2022 tomography derivation behind them drop out of this statement
> entirely. The zero is genuine sudden death and not a partial-transpose
> clamp: at the Γ_φ = 1e7 anchor the smallest partial-transpose symplectic
> eigenvalue is ν̃ = **0.5134**, i.e. 1.3e-2 *above* the separability value 1/2
> (the mp-margin rule fired on no row), and E_N is already exactly zero from
> **Γ_φ = 1e5 Hz upward** (8.17e-5 at 5e4 Hz, 0 at 1e5). The floor-dependent
> way to say the same thing — the model tolerates only Γ_φ ≲ 5e4–1e5 Hz at
> T ≤ 12.5 mK, **100–200× below** the device's digitised range against a
> digitisation uncertainty of ~2× — is the weaker statement, and is quoted
> below only because it is the one that transfers to other floors.
>
> **Temperature is not the binding constraint under the derivative coupling.**
> At the fridge base (30 mK) with the measured minimum relaxation
> (Γ₁ = 8.7 MHz, no dephasing) the model still yields E_N = 1.24e-4 — and
> 1.27e-4 on the noiseless sourced-separation row (Γ₁ = 0) — resolvable at the
> band's low edge, where the amplitude model gave exactly zero. The temperature
> wall moved *up* (to 30–35 mK) and stopped being a sudden death. What kills
> the device point is dephasing, not the fridge.

This is a **negative** device statement, and it is the direction the evidence
is strong in: at the device's own dephasing the quantity is identically zero,
not merely under a threshold, while every remaining uncertainty is tens of
percent of a quantity that is already exactly zero there. It is *not* a claim
that the experiment yields a measurable amount of anything.

**What still cannot be claimed, and the single next blocker.** A *positive*
device claim — predicting a measurable yield — is not supported: nothing on any
`'xp'` surface reaches the derived floor (max E_N = 1.90e-4, 5.3× under 1e-3)
at any coupling in the band. The remaining inputs, with their measured status:

| input | status after the fourth pass | can it block the statement above? |
|---|---|---|
| the coupling **operator** | **closed**: the map is now run with the proposal's `'xp'` operator; `'xx'` is kept as the archived comparison | no |
| the coupling **value** | derived; run at ±1σ and four alternatives | no — the box verdict is identical across the whole band |
| IR mass m | **measured** (Deliverable 7): under `'xp'` every device-anchor verdict is *invariant* over m = 0.05–0.4 (0.37–2.92 GHz) at all three floors; the dephasing anchors are exactly zero at every mass | no |
| boundary buffer | measured: 0.0025 % relative spread on E_N under `'xp'` (buffers 6/12/24) — 100× tighter than the amplitude model's 0.31 % | no |
| sweep grids and surface axes | coordinates; every device anchor is a grid point, so verdicts are reads | no |
| plot-digitised Γ_φ (factor ~2) | inherited hazard | no — it cannot close a 100–200× gap |
| lattice UV cutoff vs the device's 50 GHz | **measured** (Deliverable 8): the cutoff was swept 14.7 → 73 GHz at fixed physical geometry, *through* the device's 50 GHz. Every dephasing anchor stays exactly zero, no anchor verdict drifts anywhere in the bracket, the temperature wall does not move, and the alive (contrast) anchors change by at most +25 % — the same scale the proposal's own App. C reports for Ω_cut → ∞ | **no** |
| oscillator detectors, not qubits (MAP-1) | leading order in λ; the perturbative companions sit in the trustworthy regime at this coupling | not for a two-orders-of-magnitude margin; it does bound any *positive* claim |

That was written before the fifth pass, which ran exactly the thing it asked
for. **No input of this model can now block the negative statement**: every one
of them has been swept and none moves a device-anchor verdict. What is left is
not an input but the model's own idealisations — oscillator detectors rather
than qubits, a hard Brillouin edge rather than the device's exponential C(ω),
and 1+1 — each worth tens of percent against a margin of two orders of
magnitude.

**A positive claim is now blocked by nothing except the absence of a GO
region.** At the device's own cutoff (50 GHz) the largest `E_N` anywhere on the
surface is 2.33e-4 — 4.3× under the derived floor — and 0 % of the surface is
GO at that floor at any cutoff in the bracket. There is no remaining input
whose measurement could produce a GO region: the surface simply does not reach
the floor, and the device's operating point is not even where the model's own
band-low pocket lives (T ≤ 30 mK *and* Γ_φ ≤ 1e5 Hz).

---

## The producing build

| item | value |
|---|---|
| generator | `papers/circuit-qed-noise-thresholds/notebook.py`, entry point `main()` |
| generator sha256 | `1394970e63aeb8e142de8f2693fb219d27a50c48c00f7b3860cd19d54e22f9d7` |
| `git rev-parse HEAD` at production | `6634641699cade393a5ab78682dc939873a6cfdd` (**worktree dirty**: `vacuum/detectors/*` was being edited concurrently by another agent, adding `coupling='xp'`; this candidate uses only the `'xx'` public API. The task started at HEAD `77a288918a0c81150a933f4e02d4985696f46e91`.) |
| produced (UTC) | 2026-09-03T00:09:46Z |
| package | `vacuum` 0.1.0 (editable) |
| environment | Python 3.13.3, numpy 2.5.2, scipy 1.18.1, macOS-14.6.1-arm64, 16 cores |
| command | `.venv/bin/python papers/circuit-qed-noise-thresholds/notebook.py --workers 14` |
| wall clock | **776.7 s** (12.9 min) for 1043 points; 9694 CPU-s summed; slowest point 97.3 s |
| pytest, full suite | **828 passed, 3 failed, 5 warnings in 428.4 s** (`.venv/bin/python -m pytest -q`, run immediately after production on the same worktree). The 3 failures are all in `tests/test_geometry_mera.py` (Layer 4 MERA: `test_optimizer_lowers_energy_smoke`, `test_geodesic_and_minimal_cut_scaling`, `test_minimal_cut_entropy_reproduces_c_over_3_log`) — an **untracked, in-progress** module (`vacuum/geometry/mera.py` and its test are uncommitted work of another agent) that this candidate does not import. **The papers/README.md gate is therefore NOT met on this build.** |
| pytest, this candidate | **22 passed in 16.9 s** (`tests/test_noise_threshold_smoke.py`) |

**The third- and fourth-pass builds (2026-09-04).** All four added archives
(`device_point`, `summary_xp`/`rows_xp`/`surfaces_xp`, `gap_tracking`,
`ir_mass_control`) were regenerated back-to-back on the *final* code state, so
one generator sha256 — `0717fa69ab9e57249cb0121a9b5e33e9d54de6d0308810e772fca472c3290691`,
git HEAD `d248e3c5` with the worktree dirty (this pass's own uncommitted work:
`vacuum/detectors/circuit_mapping.py`, `tests/test_circuit_mapping.py`, the two
derived parameter-file entries and `notebook.py`) — describes every one of
them, and each carries its own `build` block recording it. Commands and wall
clocks: `--device-point` 270 rows / 268 s, `--xp-map` 922 rows / 866 s,
`--gap-tracking` 90 rows / 74 s, `--ir-control` 288 rows / 259 s (14 workers,
same environment as above). Tests on that state: **116 passed** across
`tests/test_circuit_mapping.py` (49), `tests/test_noise_threshold_smoke.py`
(22) and `tests/test_experiments_io.py` (45); the full-suite gate is the
coordinator's to re-run on a committed tree, and the second-pass archives
(`rows.npz`, `surfaces.npz`) are untouched by this pass.

**The fifth-pass build.** `data/cutoff_bracket.json` and `rows_cutoff.npz` were
produced at generator sha256 `6fb4a4b193429302…`, git HEAD `267aede8` (dirty:
this pass's own work), 2026-09-04T16:18Z. That sha differs from the fourth
pass's `0717fa69…` because the fifth pass **added** code — the
`--cutoff-bracket` mode and the `lattice_spacing` field, which defaults to 1.0.
The fourth-pass archives were not regenerated because the addition is provably
inert on their path: at `a = 1` the new construction is the identity
(`lattice_at_spacing` returns `harmonic_chain_K` itself, rescale 1.0), and a
device-point row re-run at `a = 1` on the fifth-pass code reproduces the
fourth-pass archive's `E_N` and `MI` with a difference of **exactly 0.0** —
asserted in `tests/test_circuit_mapping.py`, not merely observed.

The same sha256 is embedded in every archive under `data/` (`__build__` key),
together with the git HEAD and the dirty flag.

Admissibility per `papers/README.md`: the full regression suite was run on this
code state immediately after production (row above) and is **not green** —
three Layer-4 MERA tests from uncommitted work outside this candidate fail;
every Layer-1/2/3 anchor and every standing audit passes. Re-run the gate once
that module lands or is removed; the archives' `generator_sha256`/`git_head`
say which code state a re-run must match. No number here came from an
optimiser, so no plain-numpy round-trip is owed. No audit anomaly was
encountered, so the anomaly protocol did not fire. HEAD moved twice during the
task (`77a2889` → `6634641` at production → `567ffcf` by the time the suite
finished) through checkpoint commits by other agents; the recorded producing
HEAD is `6634641` with a dirty worktree.

---

**Integration: full suite 1201 passed on the quiescent tree at HEAD `c18a206`, 2026-09-04** (`.venv/bin/python -m pytest -q`, twice in a row on the final tree: 1201 passed, 10 warnings in 731.91s (0:12:11) and 1201 passed, 10 warnings in 740.37s (0:12:20); with tenpy, cvxpy, jax, jaxlib, qiskit, qiskit_aer and qiskit_ibm_runtime blocked at the import system, 1026 passed, 76 skipped in 428.84s (0:07:08); all 15 notebook `--quick` paths exit 0, the slowest in 307 s). The tree is commit `c18a206` plus this pass's uncommitted edits — README, STATUS, PRIORITY, MANIFEST and draft prose reconciling the rollups to the MANIFESTs, plus one print in `interacting-vacua-first-numbers/notebook_qei_exact.py` made None-safe so its `--quick` path exits 0 (no numerical code, no data). `data/` was produced on the build recorded above; this run is the admissibility record under `papers/README.md` for the candidate's code as it stands on that tree, and a full re-run of `notebook.py` there regenerates `data/` with a new `generator_sha256`; the producing-build fields above are left as recorded.

## The coupling — derived (2026-09-04)

The derivation is the module docstring of `vacuum/detectors/circuit_mapping.py`
(steps 1–7, every equation number cited from arXiv:2505.01516v1 and
arXiv:2208.05571v2, both read from the PDF text); its tests are
`tests/test_circuit_mapping.py` (19 tests). In brief:

1. **The circuit (physical units).** Eq. (3) line Hamiltonian; Eq. (5) mode
   expansion Φ = √(ħZ₀)∫dk(4π|k|)^{−1/2}(…) — the canonical normalisation, with
   √(ħZ₀) carrying every dimension; Eqs. (16)–(19), (22): H_int =
   −(φ₀/ℓ₀)γ̂₅∂ₓΦ_C(x_d) — the coupler phase couples to the line *current*;
   the VGSD detector, Eq. (31): H_int(t) = −(φ₀/ℓ₀)γχ(t)μ̂(t)∂ₓΦ_C(t, x_d).
2. **The paper's λ.** Eq. (36) φ_C = Φ_C/√(ħZ₀); Eq. (37) H_I = ħc Σ λ χ μ̂ ∂ₓφ_C;
   matching with ℓ₀ = Z₀/v gives Eq. (66), λ = −γ√(R_K/8πZ₀) = −4.53γ, and
   λ² = πα with Eq. (33). The normalisation is fixed *physically* by the
   device paper: Janzen Eq. (10) Γ₁ = (φ₀²/ħZ₀)|γ₅,₁₀|²Δ = λ²Δ = παΔ (Eq. 6) is
   the Golden-rule rate of Eq. (37) on the massless line. Physical scale:
   |λ_TB|Ω⁰/2π = 0.66 GHz at γ = 0.02 (9 % of the gap).
3. **Lattice units.** H_I/(ħω_ref) = λχμ̂∂_ξφ for *any* ω_ref (the derivative
   coupling is scale-free), and the canonical 1+1 massless field is
   scale-invariant (dk(4π|k|)^{−1/2}a_k is invariant under k → κω_ref/v): no
   Z₀ or v factor survives beyond Eq. (66), no lattice-spacing factor enters
   the field amplitude, and the spacing enters only through the derivative,
   i.e. through Ω_lat = Ω⁰/ω_ref = 1. The `harmonic_chain_K` field is the same
   field (tested: its ground-state ⟨φ²⟩ equals ∫dω S_φ^lat/2π).
4. **MAP-1** (Gate A) gives the oscillator: the derivative (`'xp'`)
   transcription is *exact*, λ_osc^xp = |λ_TB|√(2Ω_lat) (∂_t and ∂ₓ Wightman
   functions coincide for the massless field).
5. **The `'xx'` model** has no transcription (different operator, S ∝ 1/ω vs
   ω); it is matched by equal field-induced Golden-rule rate at the free gap to
   the transcribed `'xp'` model on the same lattice: λ_osc = |λ_TB|Ω_lat√(2Ω_lat)
   — identical to the continuum matching λ_osc²/2Ω² = λ_TB²Ω = παΩ, the rate
   that *defines* the device's α. At Ω_lat = 1 this is √2·|λ_TB|: the anchored
   number, now derived.
6. **Alternatives enumerated and run** (`data/device_point.json →
   coupling_map.alternatives`): finite-difference ∂ₓ on the lattice ×0.980;
   physical-rate matching on the lattice ×0.924 (the lattice's density of
   states at half the band); matching at the switched-on gap 0.937Ω⁰ ×0.907 /
   ×0.845; the Eq. (66) reading of λ ×0.906; lower envelope 0.1185. Gap
   tracking λ_osc(t) ∝ Ω(t)^{3/2} (≤ 9 %) is bounded, not run.
7. **Uncertainty.** No input carries a published error bar (γ, Z₀, v, Ω⁰ are
   all "≈"). Scenario 2 is printed three ways — λ = −0.1 (Table I, the value
   the paper simulates per the Fig. 8/18 captions), Eq. (66)×γ = −0.0906,
   √(πα) = 0.0971 — and their sample standard deviation, 0.0048 (4.8 %), is the
   1σ; the code propagates any `error` field a file gains on γ or Z₀.

   **What that ±0.0068 on λ_osc is, and what it is not.** It *is* the spread of
   three published roundings of a single quantity, propagated linearly
   (σ_osc = M σ_TB). It is **not** a device error bar and not a propagated
   measurement uncertainty: `sigma_propagated_from_file_errors` is exactly
   **0.0**, because no input to the map carries a published error. Nor is it a
   symmetric confidence interval — the three readings are asymmetric about the
   simulated value (−0.1, −0.0906, −0.0971), so −1σ (λ_osc = 0.1347) does
   **not** reach the Eq. (66) reading (0.1282), and ±1σ is therefore *not* the
   right interval for "could the mapping have given something else".
   **Where the range is what matters, quote the mapping band [0.1185, 0.1482]**
   — the envelope of the enumerated criteria and readings (step 6) — which is
   what the device point was actually re-run across (six couplings, Deliverable
   4), rather than ±1σ. The ±1σ is reported because it is the paper's own
   rounding spread and travels with the parameter file's `error` field.

Parameter-file record: `teixido-bonfill_2026_table1.json →
scenario2_coupling_lambda_tb` (−0.1 ± 0.0048) and
`oscillator_coupling_lambda_osc_scenario2` (0.14142 ± 0.00677), both
`derived`, `derived_from` naming the transcribed columns; the notebook refuses
to load if the file's value and the code's disagree. The surfaces archive
(`data/rows.npz`) keeps the tag `unsourced` on its rows as the historical
record of the build that made it (`load_inputs(coupling='anchored')`, the
default, reproduces that configuration); the device-point archive carries
`derived`.

## Modules used (read-only; `'xx'` public API only)

| module | what it does here |
|---|---|
| `vacuum.detectors.circuit_mapping` (new, 2026-09-04) | the circuit → lattice coupling map: λ_TB from γ/Z₀ (Eq. 66), the spectral densities, MAP-1, the `'xx'` matching criteria and their inverses, the reading spread and error propagation |
| `vacuum.protocol` (`run_protocol`, `GeneratorStep`, `ChannelStep`) | **the** code path: every one of the 1043 points is a protocol run through the M-I.4 runner with its ledger and audits. Nothing bypasses it. |
| `vacuum.detectors.nonperturbative` (`attach_detectors(coupling='xx')`, `initial_state`, `detector_block`, `detector_occupations`, `harvested_log_negativity`, `run_harvesting`) | the coupled quadratic form, the λ(0)=0 initial state with its mandatory passivity audit, the mp-margin-safe `E_N`, and (once) the independent cross-check driver |
| `vacuum.detectors.imperfections` (`imperfections_from_rates`, `imperfection_step`, `bath_occupation`) | published rates → Strang-interleaved Gaussian channels on the detector modes |
| `vacuum.detectors.switching` / `.kernels` / `.udw` / `.communication` | cos² switching; the lattice Wightman kernel; the M2.3 pair state and the M2.5 M_vac/M_comm split on the static-gap companions |
| `vacuum.core` (`harmonic_chain_K`, `mutual_information`) | the field; I(A:B) = S(A)+S(B)−S(AB) on the detector block |
| `vacuum.audits` (`EnergyLedger`, passivity, precision, ledger closure) | the standing audits, run by the runner on every point |
| `vacuum.experiments_io` (`load_experiment`) | the **only** place published SI values become lattice units |

---

## Parameter-file provenance — what is sourced, what is not

Eight files, all loaded through `vacuum.experiments_io.load_experiment`
(refuses incomplete provenance); every quantity's per-file `provenance` tag is
read by the notebook and re-checked by `tests/test_noise_threshold_smoke.py`.

**Verified by loading (test `test_file_level_synthetic_content_is_exactly_the_three_grids`):
across all eight files the only quantities tagged `synthetic` are the three
sweep grids of `janzen_2023_fig6_variant-noise-sweep.json` —
`waveguide_temperature_grid`, `relaxation_rate_grid`,
`pure_dephasing_rate_grid`.** `teixido-bonfill_2026_table1.json` is 37
transcribed + 4 derived / 0 synthetic (the two coupling entries added
2026-09-04); the switching variant is fully sourced
(file-level flag only for the *selection* of operating points).

### Sourced (transcribed / derived / selection)

| input | file | value and location |
|---|---|---|
| ω_ref, waveguide speed | `teixido-bonfill_2026_table1` | 2π × 7.3 GHz (Eq. 30); v = 1.2e8 m/s (Sec. II.1) |
| scenario λ, γ, α | same | Table I scenario 2: −0.1, 0.02, 0.003 |
| gap-variation law | same | ΔΩ/2π = −23 γ GHz (Eq. 30) → −0.46 GHz |
| switching window and shape | same | 0.22 ns cosine ramps, S_f = 0 (Fig. 18 caption; Eq. 71) |
| light-crossing delay → separation | `..._variant-switching` | t_d = 0.16 ns (Fig. 23(c)); d = v t_d = 19.2 mm (derived) |
| other sourced separations | `janzen_..._variant-noise-sweep`, proposal | 60 mm (t_d = 0.5 ns, Fig. 15), 120 mm (Sec. VI.A) |
| resolution floor | proposal (derived from `ren_2022_sec4-3`) | 1.0e-3 nats; band [7.82e-5, 0.04]; transmon cross-check 0.01 (`dicarlo_2009_fig3`) |
| temperature anchors | `janzen_2023_fig6`, `ren_2022`, `orgiazzi_2016`, `teixido-bonfill_2025_thesis-outlook` | 30 mK fridge base, 50 mK TL; 20 mK design; 40 mK; expected 20–30 mK |
| Γ₁ anchors | `janzen_2023_fig6` | measured 8.7 MHz, 1.85 GHz |
| Γ_φ anchors | `janzen_2023_fig6` | **plot-digitised** 1e7, 2e8 Hz (order of magnitude); measured Ramsey floor of another Lupascu device (`orgiazzi_2016`) ≈ 1.2e5 Hz-equivalent for comparison |
| oscillator coupling λ_osc (device-point archive) | `teixido-bonfill_2026_table1:oscillator_coupling_lambda_osc_scenario2` | **derived** 0.1414 (rounding spread ±0.0068, mapping band [0.1185, 0.1482]) from Table I scenario 2 (λ, γ, α), Eq. (7) Z₀ and Eq. (30) Ω⁰ through Eqs. (5), (31), (36), (37), (66) + Janzen Eqs. (6), (10) + MAP-1 — section "The coupling" |

### Synthetic or unsourced (the claim inherits this)

| input | status | why |
|---|---|---|
| `coupling_lambda` = 0.1414 on the **surfaces archive** (`data/rows.npz`) | tagged `unsourced` at production — **since derived, same number** | at production λ_osc = \|λ_TB\|·√(2Ω) identified the proposal's λ with MAP-1's λ_UDW, a convention; the derivation (section "The coupling") shows the identification omitted exactly one factor, Ω_lat = 1. The tag on the archive's rows is kept as the record of that build; the device-point archive (`data/rows_device_point.npz`) carries `derived` and reproduces the shared rows bit-for-bit. |
| the three file grids | synthetic (declared in the file) | sweep extensions beyond the published anchors |
| the four surface axes | synthetic (this candidate) | refinements of the file grids so the death is resolved to a cell: T (16 values, anchors 20/30/40/50 mK on the axis), Γ_φ 1-2-5 per decade over [1e4, 1e9] Hz, Γ₁ 1-2-5 per decade over [1, 5000] MHz plus 8.7 and 1850, separation 1–12, 14, 17, 23, 46 sites (19.2/60/120 mm at 7/23/46), λ_osc 0.05–0.6 |
| `field_mass` = 0.2, `boundary_buffer` = 6 | synthetic lattice regulators | limitation (2) below; the buffer's effect is *measured* (controls) |

Two provenance hazards inherited from the sources and **not** absorbed in code:
the device's Γ_φ is plot-digitised (factor ~2), and every published *rate*
carries the factor-2π convention ambiguity the parameter files record.

---

## Model limitations

1. **UV.** The archives use unit lattice spacing ⇒ ω_max = 2 ω_ref = 14.7 GHz,
   below the device's 50 GHz cutoff; gaps and switching bandwidths sit under
   it. **Measured, not assumed, since Deliverable 8:** the cutoff was swept
   14.7 → 73 GHz through the device's value at fixed physical geometry and no
   anchor verdict moved (E_N changes by ≤ 25 % where it is alive). What remains
   is the cutoff's *shape* — a hard Brillouin edge here against the device's
   exponential C(ω) — not its scale.
2. **IR.** A synthetic mass m = 0.2 (≙ 1.46 GHz) regulates the massless 1+1
   chain's IR divergence; the boundary control measures what is left.
3. **Oscillator detectors, not qubits.** MAP-1 is used for the perturbative
   columns and, inside the derivation, for the coupling; the sign of λ is
   invisible to the oscillator model.
4. **Two operators, both run.** The four surfaces of Deliverables 1–3 use the
   amplitude coupling `'xx'`; the proposal's own **derivative** coupling
   `'xp'` — the exact MAP-1 transcription of Eq. (37) — is run in
   Deliverable 5 on the main (T × Γ_φ) surface and on the device point, and is
   the operator the device statement rests on. The two are genuinely different
   models: at the same λ_UDW the perturbative companions differ by 3.5× in E_N
   at the operating point (communication fraction 0.29 vs 0.98), and the
   nonperturbative surfaces differ by 5.2× in max E_N. Where they disagree,
   the `'xp'` number is the one that speaks about the proposal.

---

## Data

| file | contents |
|---|---|
| `data/rows.npz` | 1043 rows × 79 columns: four 16×16 surfaces (`sweep` ∈ {dephasing, relaxation, separation, coupling}, 256 each), 16 static-gap `split` companions carrying the M2.5 columns, 3 finite-size `control` rows. Column names in `__columns__`, build in `__build__`. |
| `data/surfaces.npz` | the same surfaces as dense arrays: `<surface>/T_mK`, `<surface>/<x-axis>`, `<surface>/{E_N, MI, work, E_N_per_work, MI_per_work, n_d_A, mp_margin_flagged, all_audits_passed, dt_converged}`; `resolution_levels_nats`. |
| `data/summary.json` | the deliverables: `go_no_go[surface].{E_N_contour, MI_contour}[level]` (row-wise and column-wise crossings with brackets, interpolated positions and the sudden-death flag), `device_anchors` verdicts, `best_pockets_per_switching_work[level]`, `pivot`, controls, cross-check, split rows, the full provenance map. |
| `data/rows_quick.npz`, `data/summary_quick.json` | the `--quick` configuration (4×4 + 3 rows), same schema, same code. |
| `data/rows_device_point.npz` | **270 rows** (2026-09-04, `--device-point`): 6 couplings × 5 temperatures (0, 20, 30, 40, 50 mK) × 9 anchors (Γ_φ digitised min/max, Γ₁ measured min/max, d = 19.2/60/120 mm, and two *combined* Γ₁+Γ_φ anchors the surfaces never ran), same 79-column schema, same `run_point`. |
| `data/rows_xp.npz`, `data/surfaces_xp.npz` | **922 rows** (Deliverable 5, `--xp-map`): the main (T × Γ_φ) 16 × 16 surface at three couplings (768 rows), the device point at the same three (135), the derivative-kernel split companions (16) and the finite-size controls (3), all `coupling='xp'`. `surfaces_xp.npz` holds the dense arrays keyed `dephasing@c0/…`, `@c-1s`, `@c+1s`, each with its `lambda_osc`. |
| `data/summary_xp.json` | Deliverable 5: contours and **temperature-wall positions** per Γ_φ column at all three floors, the `'xx'` archive's walls beside them, per-coupling device-point verdicts, the row-by-row `'xx'` vs `'xp'` side-by-side, the split companions side by side, the two-driver check and the finite-size control. |
| `data/rows_gap_tracking.npz`, `data/gap_tracking.json` | Deliverable 6 (`--gap-tracking`): 90 device-point rows with λ(t) tracking the gap, both operators. |
| `data/rows_cutoff.npz`, `data/cutoff_bracket.json` | Deliverable 8 (`--cutoff-bracket`): 369 rows over five lattice spacings (UV cutoff 14.7–73 GHz) — the `'xp'` device anchors at every cutoff plus the reduced surface at 24.4 and 50.0 GHz — with the per-cutoff verdicts, wall positions, dephasing margins and the recorded `audit_event`. |
| `data/rows_ir_control.npz`, `data/ir_mass_control.json` | Deliverable 7 (`--ir-control`): 288 device-point rows at four IR masses × both operators — the control that measures the last synthetic input. |
| `data/device_point.json` | Deliverable 4: the coupling map with every alternative, per-coupling verdicts at the three floors (device box, design 20 mK, 40 mK, vacuum), the archive cross-check (35 shared rows), the `'xx'`/`'xp'` model-mismatch companion, the remaining synthetic inputs. Every archive of the third and fourth passes carries its own `build` block (generator sha256, git HEAD, UTC) and all four were regenerated on the final code state, so one sha describes them all; **`summary.json` gained the keys `device_point_derived_coupling`, `xp_derivative_coupling`, `gap_tracking` and `ir_mass_control`** (headline subsets, each with its build block) — every other key of `summary.json` is untouched, and `rows.npz`/`surfaces.npz` are the untouched second-pass archives. |

Every row carries: parameters in published **and** lattice units; `E_N`, `MI`,
⟨n_d⟩, ledger `work` and `dissipated`, `E_N_per_work`, `MI_per_work`; the
resolvability flags at the derived floor and at both band edges for `E_N` and
for `MI`; the convergence provenance (`dt_converged`, `halvings`, `movement`,
`work_movement`, `n_substeps`); the audit flags (`mp_margin_flagged`,
`precision_flagged`, `passivity_entry_passed`, `passivity_initial_passed`,
`ledger_passed`, `ledger_defect`, `causality_audited`, `all_audits_passed`);
and the M2.5 columns on the companions.

---

## Audit status

| audit | result over all 1043 rows |
|---|---|
| **ledger closure** (mandatory, every point) | all passed; worst defect **9.95e-11** (criterion as run: absolute 1e-9; the criterion has since been corrected to scale with the booked energies and entry count — Deliverable 8, "Audit event" — and every recorded defect here is well inside both) |
| **passivity** on the claimed T = 0 ground state | all passed — twice per point (`initial_state`, and the runner's `claim_ground_entry`) |
| **precision** on the reported detector block | all passed; `flagged` fired on **0** rows |
| **mp-margin rule** on `E_N` (M2.2) | live on every row, fired on **0** — sudden death parks the partial-transpose spectrum nowhere near the floor. `test_mp_margin_rule_is_live_on_this_code_path` drives the same `run_point` to a λ = 0 floor state and asserts the rule fires, so the zero is a result, not a dead column. |
| **causality** (runner's Lieb–Robinson audit) | deliberately OFF (1D-chain ordering assumption); checked physically through `M_comm` on the split rows: \|M_comm\|/\|M\| falls from 0.85 (1 site) to 4.7e-4 (12 sites) to 1.8e-12 (23 sites) across the 10-site light cone. |
| **dt-halving gate** (M2.2, `conv_tol = 1e-9` on E_N, ⟨n_d⟩ **and MI** — one observable more than `run_harvesting`, never fewer) | converged on all 1043; accepted level ≤ 10; worst accepted movement **9.99e-10**; worst work movement 4.8e-8 (recorded, not gating). Tolerances were not relaxed anywhere. |
| **two independent drivers** | static-gap base point through `run_protocol` vs `run_harvesting`: \|ΔE_N\| = **6.7e-16** |
| **finite size** | operating point at buffers 6/12/24 (N = 31/43/67): E_N = 1.00906e-3 / 1.00628e-3 / 1.00598e-3, spread **3.1e-6 (0.31 %)** — the dominant numerical systematic, three orders above the gate |
| **perturbative companions** | 15 of 16 split rows in the trustworthy regime (`perturbative_ok`); the 46-site (120 mm) companion's quadrature refused (pair term below `atol`) and is flagged `perturbative_quadrature_failed`, not reported. Nonperturbative vs perturbative E_N agree to 1–7 % over 5–12 sites (0.4 % at 9 sites); the 1–4-site rows are causal-contact points where O(λ²) is not enough (1.49e-3 vs 0.99e-3 at 1 site), and the 14-site row differs by 10 % at the 2e-5 level. |
| **`'xp'` map rows** (922, Deliverable 5) | ledger closed on all, worst defect **9.77e-11**; passivity twice per T = 0 row; precision flagged 0; mp-margin fired 0; dt gate converged on all at unchanged tolerances; **two independent drivers on the `'xp'` static-gap point: \|ΔE_N\| = 1.11e-15** (`run_protocol` vs `run_harvesting`, which share no evolution code); finite size at buffers 6/12/24 gives E_N = 2.16705/2.16708/2.16710e-4, relative spread **2.5e-5** — 100× tighter than the amplitude model's 0.31 %. |
| **UV-cutoff bracket rows** (369, Deliverable 8) | all audits passed and all converged at unrelaxed tolerances; worst ledger defect **4.73e-10**; mp-margin fired 0. One cell of the *first* pass fired the ledger audit and was handled by dropping a synthetic column, not by relaxing the tolerance; the event is now **resolved** — diagnosed as float64 roundoff and closed by correcting the criterion to scale with ⟨H⟩ and the entry count, under which neither that cell nor the retained 2e8 Hz anchor fires (97–422× margin) — see Deliverable 8, "Audit event". |
| **IR-mass control rows** (288, Deliverable 7) | all audits passed; worst ledger defect **1.69e-10**; all converged. |
| **gap-tracking rows** (90, Deliverable 6) | all audits passed; worst ledger defect 4.76e-11; all converged. |
| **device-point rows** (270, 2026-09-04) | ledger closed on all, worst defect **4.83e-11**; passivity passed twice per T = 0 row; precision flagged 0; mp-margin fired 0; dt gate converged on all (tolerances unchanged); **archive cross-check: 35 rows shared with `rows.npz` reproduce E_N, MI and work with max \|Δ\| = 0.0** — the same code path on a later HEAD. |

---

## Deliverable 1 — the go/no-go boundary as a curve

Crossings are given as the bracketing grid cell and the (log-)linear
interpolation inside it; cells with a `sudden_death` flag (one side exactly 0)
are brackets only. Full curves in `summary.json → go_no_go`.

**Temperature wall (all surfaces, low-noise edge).** Band-low floor
(7.82e-5): T ∈ [27.5, 30] mK — sudden death — at Γ_φ ≤ 5e4 Hz, Γ₁ ≤ 200 MHz,
separations ≤ 7 sites, λ_osc ∈ [0.125, 0.25]; the wall bends down to
[22.5, 25] mK at Γ_φ = 2e5 Hz, [17.5, 20] mK at Γ₁ = 1.85 GHz, [12.5, 15] mK
at λ_osc = 0.4. Derived floor (1e-3): the wall is at [10, 12.5] mK at the
anchored coupling and sourced separation, and the region is *empty* above
25 mK for every coupling.

**Dephasing.** Band-low: Γ_φ ∈ [2e5, 5e5] Hz at T ≤ 20 mK (interp 4.3e5 → 3.8e5),
[1e5, 2e5] at 25 mK, [5e4, 1e5] at 27.5 mK. Derived: no GO cell on the
surface (max 9.82e-4 at Γ_φ = 1e4 Hz, T = 0). Band-high: none.

**Relaxation.** Derived: Γ₁ ∈ [8.7, 10] MHz at T ≤ 5 mK, [5, 8.7] MHz at 10 mK,
nothing beyond 12.5 mK. Band-low: [2.0, 3.0] GHz at T ≤ 15 mK, [1.85, 2.0] at
17.5 mK, [1.5, 1.85] at 20 mK, [0.2, 0.5] GHz at 27.5 mK.

**Separation** (light cone at 10 sites for the 10.1-site window). Derived at
T ≤ 10 mK: [7, 8] sites (18.3–20.9 mm) — the sourced 19.2 mm is the *last* GO
site. Band-low at T ≤ 10 mK: [12, 14] sites (31–37 mm), i.e. 2–4 sites
*outside* the light cone with \|M_comm\|/\|M\| ≤ 4.7e-4 — the model's only
genuinely spacelike harvesting, and it lives at the band's low edge only.
60 mm and 120 mm: E_N = 0 at every temperature.

**Coupling** (the axis of the input that was unsourced at production and is
now derived; the derived value and its band [0.118, 0.148] lie on the cells
0.1–0.1414–0.17). Derived: λ_osc ∈ [0.1407, 0.382] at
T ≤ 10 mK, [0.161, 0.358] at 20 mK, [0.182, 0.320] at 22.5 mK, empty from
25 mK; the anchored 0.1414 is GO by 0.9 % at T = 0. Band-low: up to
λ_osc ∈ [0.40, 0.45], dying at [27.5, 30] mK for 0.125 ≤ λ_osc ≤ 0.25.
Strong coupling kills harvesting outright (E_N = 0 for λ_osc ≥ 0.45 at T = 0):
local excitation overtakes the pair term.

**Where the device sits** (grid reads, `summary.json → go_no_go[*].device_anchors`):

| anchor | E_N | MI | derived | band-low | band-high |
|---|---|---|---|---|---|
| 30 mK × Γ_φ ∈ {1e7, 2e8} Hz | 0 | 1.2e-5 / 1.2e-6 | NO-GO | NO-GO | NO-GO |
| 50 mK × Γ_φ ∈ {1e7, 2e8} Hz | 0 | 4.1e-5 / 3.1e-6 | NO-GO | NO-GO | NO-GO |
| 30 mK × Γ₁ ∈ {8.7 MHz, 1.85 GHz} | 0 | 1.9e-4 / 5.5e-5 | NO-GO | NO-GO | NO-GO |
| 50 mK × Γ₁ ∈ {8.7 MHz, 1.85 GHz} | 0 | 2.4e-4 / 3.7e-5 | NO-GO | NO-GO | NO-GO |
| 30 / 50 mK × d ∈ {19.2, 60, 120} mm | 0 | ≤ 2.4e-4 | NO-GO | NO-GO | NO-GO |
| 20 mK (design) × Γ₁ = 8.7 MHz, d = 19.2 mm, anchored λ | 7.8e-4 | 1.1e-4 | NO-GO | **GO** | NO-GO |
| 20 mK × Γ₁ = 1.85 GHz | 6.3e-5 | 2.7e-5 | NO-GO | NO-GO | NO-GO |
| 20 mK × Γ_φ ∈ {1e7, 2e8} Hz | 0 | ≤ 4.4e-6 | NO-GO | NO-GO | NO-GO |

The only GO anywhere near the device is the design temperature (20 mK) at the
lowest measured Γ₁, at the band's *low* edge, and it requires Γ_φ < ~4e5 Hz —
25× below the device's plot-digitised range.

---

## Deliverable 2 — best pocket per unit switching work

`E_N_per_work = E_N / W_drive`, W_drive the ledger's summed `work` entries.

| floor | pocket | E_N | W_drive | E_N/W |
|---|---|---|---|---|
| derived (1e-3) | `separation-T0-X4`: 4 sites (10.5 mm), T = 0, anchored λ | 2.27e-3 | 5.58e-4 | **4.08** |
| band-low (7.82e-5) | `coupling-T0-X0.05`: λ_osc = 0.05, 19.2 mm, T = 0 | 1.32e-4 | 2.86e-5 | **4.62** |

The largest `E_N` on the surfaces (2.95e-3, λ_osc = 0.3) buys its yield with
~18× the drive work of the efficient pocket. Best *MI* per unit work is at
1 site / 80 mK (MI 7.6e-3, ratio 0.73) with `E_N = 0` — thermal, classical
correlations, not entanglement.

---

## Deliverable 3 — the pivot verdict (PLAN.md Layer 2)

PLAN.md: "if harvested negativity under honest noise models sits orders of
magnitude below detectability for all realistic parameters, the layer pivots
to mutual-information and coherence harvesting." Numbers
(`summary.json → pivot`):

- Device box (30/50 mK × measured/digitised rates × sourced separations ×
  anchored coupling, 16 anchor points): **max E_N = 0** (the device sits ON the
  sudden-death wall: min ν̃_PT ≈ 0.500008 at the 30 mK anchors, 8e-6 above the
  floor and ≈5× the 3e-6 finite-size systematic — the zero is the partial-transpose
  clamp, not a margin), **max MI = 2.40e-4 nats** = 0.24× the derived floor,
  3.1× the band's low edge.
- Everywhere on all four surfaces, max E_N = 2.95e-3 = 2.9× the derived
  floor, 0.07× the band's high edge.
- MI resolvable where E_N is not ("MI-only"), at the derived floor: 0 % of the
  dephasing surface, 0 % relaxation, 6.6 % separation (1–4 sites, T ≥ 35 mK),
  **35.5 % of the coupling surface** (λ_osc ≥ 0.29–0.42 at every T: MI up to
  1.66e-2 with E_N = 0). At the band-low floor: 20 % / 25 % / 18 % / 52 %.

**Verdict for the amplitude (`'xx'`) model, as originally recorded:** for
`E_N` the trigger fires at the sourced operating point — the device's own
temperature and dephasing leave *no* entanglement, at any resolution in the
band, and the only GO pocket needs 20 mK plus a dephasing rate the device paper
does not show. Mutual information is the correlation that survives the device
box, but at the anchored coupling it is *also* below the derived floor (0.24×)
and resolvable only at the band's low edge; it becomes resolvable at 1e-3 nats
at ≥ 2× the anchored coupling, where the entanglement is gone. On that model
the pivot to MI/coherence harvesting was therefore numerically justified as the
*only* measurable correlation in the device box, and not a free win — with two
caveats: the floor was derived for E_N (a concurrence floor), not for MI, so
"MI resolvable" uses the same nats value as a proxy; and the whole verdict is
for `'xx'`.

**Superseded for the device by Deliverable 5.** Under the proposal's own
derivative coupling the device-box MI is 1.69e-6 nats — 142× smaller than the
amplitude model's and NO-GO at *every* floor including the band's low edge —
and the MI-only fraction of the `'xp'` surface is 0 % at every floor. The
correlation that "survived the device box" above is an artefact of the
amplitude stand-in, so **the MI pivot is not supported at the device point by
the operator the proposal actually uses.** The `'xx'` conclusion stands only as
a statement about the `'xx'` model (where it also holds at every coupling in
the band: max device-box MI 0.16–0.27× the derived floor across the six
couplings of Deliverable 4).

---

## Deliverable 4 — the device point at the derived coupling (2026-09-04)

`notebook.py --device-point --workers 14`: 270 rows in 118.6 s (1530 CPU-s),
every point through `run_protocol` with the same gate, audits and schema as
the surfaces. Couplings (`data/device_point.json → couplings`): the derived
value, its ±1σ, and the alternatives of "The coupling" step 6. Floors as
before: derived 1e-3, band [7.82e-5, 0.04].

| coupling | λ_osc | device box 30/50 mK (18 rows): max E_N / max MI | verdict derived / band-low / band-high | 20 mK: max E_N, band-low GO anchors | T = 0: max E_N, derived-floor |
|---|---|---|---|---|---|
| derived | 0.1414 | **0** / 2.40e-4 | NO-GO / NO-GO / NO-GO | 7.87e-4 (Γ₁ = 8.7 MHz; d = 19.2 mm) | 1.009e-3 GO (2 anchors) |
| −1σ | 0.1347 | 0 / 2.16e-4 | NO-GO ×3 | 7.19e-4 (same two) | 9.20e-4 NO-GO |
| +1σ | 0.1482 | 0 / 2.65e-4 | NO-GO ×3 | 8.58e-4 (same two) | 1.101e-3 GO |
| physical-rate (lattice) | 0.1307 | 0 / 2.03e-4 | NO-GO ×3 | 6.80e-4 | 8.70e-4 NO-GO |
| Eq. (66) reading | 0.1282 | 0 / 1.95e-4 | NO-GO ×3 | 6.55e-4 | 8.38e-4 NO-GO |
| lower envelope | 0.1185 | 2.0e-7 / 1.65e-4 | NO-GO ×3 | 5.64e-4 | 7.21e-4 NO-GO |

Reading: the device box is NO-GO at every floor and every coupling in the
band; at the lower envelope a 2.0e-7 trace of E_N survives at 30 mK,
19.2 mm (the sudden-death wall sits *on* the 30 mK cell: min ν̃_PT =
0.5 − 1.0e-7, against a finite-size systematic of 3e-6 on E_N — the trace is
inside the numerical noise of the wall's position), 390× under the band-low
floor. MI in the device box is
0.16–0.27× the derived floor and resolvable only at the band's low edge —
the pivot verdict of Deliverable 3 stands at every coupling. The 20 mK design
pocket (band-low floor, Γ₁ = 8.7 MHz or noiseless 19.2 mm, no dephasing)
survives at every coupling; the derived-floor GO at T = 0 (0.9 % margin at
the derived value) survives +1σ and not −1σ or any lower alternative. The two
*combined* anchors (measured Γ₁ with the digitised Γ_φ together) give E_N = 0
at every temperature including 0 and every coupling: at the device's
dephasing the amplitude model harvests nothing even in the vacuum.

**What the derivation changed:** the number, nothing; its status, from
convention to derived; the blocker, from "the coupling" to "the operator"
(model limitation 4). The `'xx'`/`'xp'` companion ratio of 3.5 in E_N, if it
carried over to the nonperturbative rows, would put the derivative model's
device-box negativity at ≈ 0.3× the amplitude model's — i.e. further from
GO, not closer — but that is an inference, not a run; the `'xp'` map is owed.

---

## Deliverable 5 — the same map with the proposal's operator (`'xp'`, 2026-09-04)

`notebook.py --xp-map --workers 14`: 922 rows in 323 s (3379 CPU-s), every
point through `run_protocol` with the same gate, audits and schema as the
`'xx'` surfaces. `'xp'` is the **exact** MAP-1 transcription of the proposal's
Eq. (37) (λ_osc^xp = |λ_TB| √(2Ω_lat)); the split companions use the derivative
Wightman kernel of `vacuum.detectors.udw_derivative`.

**The (T × Γ_φ) surface, 16 × 16, three couplings.**

| floor | `'xp'` GO cells (c0 / −1σ / +1σ) | `'xx'` archive | max E_N `'xp'` (c0) | max E_N `'xx'` |
|---|---|---|---|---|
| derived 1e-3 | **0 / 0 / 0** of 256 | 0 of 256 | 1.90e-4 | 9.82e-4 |
| band-low 7.82e-5 | 25 / 21 / 29 (9.8 / 8.2 / 11.3 %) | 47 (18.4 %) | — | — |
| band-high 0.04 | 0 / 0 / 0 | 0 | — | — |

**The temperature wall moved up, and stopped being a sudden death.** Per Γ_φ
column at the band-low floor (`summary_xp.json → surface.per_coupling.c0.temperature_wall`):

| Γ_φ (Hz) | `'xp'` wall | `'xx'` wall |
|---|---|---|
| 1e4 | **[30, 35] mK**, smooth, interp 31.5 | [27.5, 30] mK, *sudden death*, interp 29.1 |
| 2e4 | [27.5, 30] mK, smooth, interp 29.5 | [27.5, 30] mK, sudden death, interp 29.0 |
| 5e4 | **[12.5, 15] mK**, interp 14.8 | [27.5, 30] mK, interp 28.3 |
| 1e5 | no GO cell at all | [25, 27.5] mK |
| 2e5 | no GO cell | [22.5, 25] mK |
| ≥ 5e5 | no GO cell | no GO cell |

**…and the dephasing floor moved down by 4–5×.** The band-low dephasing
crossing is Γ_φ ∈ [5e4, 1e5] Hz at T ≤ 12.5 mK, [2e4, 5e4] at 15–27.5 mK and
[1e4, 2e4] at 30 mK, against `'xx'`'s [2e5, 5e5] Hz at T ≤ 22.5 mK. So the
derivative coupling survives the *fridge* better and the device's *dephasing*
worse — and dephasing is what the device has: at all eight surface anchors
(20/30/40/50 mK × the digitised Γ_φ) E_N = 0 and every verdict is NO-GO at
every floor, under both operators.

**The device point, `'xx'` vs `'xp'` side by side** (`summary_xp.json →
device_point_side_by_side`; 5 of 45 anchor rows change verdict at c0):

| row | E_N `'xx'` | E_N `'xp'` | `'xx'` verdict | `'xp'` verdict |
|---|---|---|---|---|
| 30 mK, Γ₁ = 8.7 MHz | 0 | **1.24e-4** | NO-GO ×3 | **GO at band-low** |
| 30 mK, 19.2 mm (noiseless) | 0 | **1.27e-4** | NO-GO ×3 | **GO at band-low** |
| 0 mK, Γ₁ = 8.7 MHz | 1.00e-3 | 2.14e-4 | GO at derived | NO-GO at derived, GO at band-low |
| 0 mK, 19.2 mm | 1.01e-3 | 2.17e-4 | GO at derived | NO-GO at derived, GO at band-low |
| 0 mK, Γ₁ = 1.85 GHz | 1.31e-4 | 0 | GO at band-low | NO-GO ×3 |
| any row with device dephasing, any T | 0 | **0** | NO-GO ×3 | NO-GO ×3 |
| every row at 40 mK and 50 mK | 0 | **0** | NO-GO ×3 | NO-GO ×3 |

Device box (30/50 mK, 18 rows): max E_N = 1.268e-4 (c0), 1.159e-4 (−1σ),
1.379e-4 (+1σ) — GO at the band's low edge, NO-GO at the derived floor and the
band's high edge, at every coupling; attained **only** at 30 mK on the two
dephasing-free anchors. The two combined anchors (measured Γ₁ *with* the
digitised Γ_φ) give E_N = 0 at every temperature including 0.

**The pivot verdict weakens under the proposal's operator.** Device-box max
I(A:B) is 1.69e-6 nats under `'xp'` against 2.40e-4 under `'xx'` — 142× smaller,
and **NO-GO at every floor including the band's low edge**, where the amplitude
model's MI was GO. On the `'xp'` surface the MI-only fraction is 0 at every
floor. Mutual information is therefore *not* a fallback under the derivative
coupling: the correlation that survived the device box in Deliverable 3 was an
artefact of the amplitude stand-in.

**Split companions, derivative kernel** (`communication_split_side_by_side`):
E_N(`'xx'`)/E_N(`'xp'`) runs 13.2 at 1 site, 2.8–3.5 over 4–7 sites, 40.7 at
11 sites, with `'xp'` dead from 12 sites where `'xx'` survives to 14. At the
sourced 19.2 mm (7 sites, the light-cone-boundary case t_d = T) the
communication fraction |M_comm|/|M| is 0.984 for `'xp'` against 0.293 for
`'xx'`; across separations the `'xp'` fraction is strongly non-monotonic
(0.44, 0.06, 0.82, 0.91, 0.20, 0.60, 0.98, 0.91, …), i.e. the split oscillates
near the cone and no genuine-harvesting claim is made from it here.

---

## Deliverable 6 — the gap-tracking variant (2026-09-04)

`--gap-tracking`: 90 device-point rows with λ(t) = λ_osc χ(t) (Ω(t)/Ω⁰)^e, the
mapping factor evaluated on the *instantaneous* gap of Eq. (29) rather than the
free gap — e = 1/2 for `'xp'` (MAP-1) and e = 3/2 for `'xx'` (the matching
factor). The bound stated in the third pass was on λ (≤ 3.2 % and ≤ 9 % at full
switch-on); the measured effect on E_N is larger, because near the death edge
E_N is not simply λ²:

| operator | exponent | max \|rel ΔE_N\| | verdicts changed |
|---|---|---|---|
| `'xp'` (the proposal's) | 0.5 | **2.3 %** | **none** |
| `'xx'` | 1.5 | **33.6 %** | 2 of 45: the T = 0 derived-floor GO at Γ₁ = 8.7 MHz and at 19.2 mm is lost (1.001e-3 → 8.59e-4; 1.009e-3 → 8.66e-4) |

So gap tracking does not touch the operator the device statement rests on, and
it removes the amplitude model's only derived-floor GO — the knife-edge margin
of Deliverable 4 does not survive its own convention choice.

---

## Deliverable 7 — the IR-mass control (2026-09-04)

`--ir-control`: 288 device-point rows at m = 0.05, 0.1, 0.2, 0.4 lattice units
(0.37–2.92 GHz, a factor 8 around the declared regulator m = 0.2) × both
operators × 4 temperatures × 9 anchors. This measures the last synthetic input
whose effect on a verdict had never been quantified (the buffer's is measured
by the finite-size control; the grids and axes are coordinates).

| operator | device-box max E_N by mass (0.05 / 0.1 / 0.2 / 0.4) | verdicts independent of m? |
|---|---|---|
| `'xp'` | 1.29e-4 / 1.29e-4 / 1.27e-4 / 1.03e-4 | **yes, at all three floors** — zero drift over the whole range |
| `'xx'` | 0 / 0 / 0 / 1.48e-4 | **no**: 5 verdicts drift at m = 0.4 (T = 0 loses its derived-floor GO, 1.00e-3 → 3.34e-4; 30 mK *gains* a band-low GO) |

**At every mass and for both operators, every device dephasing anchor is
exactly zero.** The IR regulator therefore cannot block the device statement,
and the amplitude model is the mass-sensitive one — a third reason (after the
5.2× yield gap and the 100× looser finite-size behaviour) that the `'xp'`
numbers are the ones that speak about the proposal.

---

## Deliverable 8 — the UV-cutoff bracket (2026-09-04)

`notebook.py --cutoff-bracket --workers 14`: 369 rows in 1149 s (15 584 CPU-s,
slowest point 492 s), all `'xp'` at the derived coupling.

**Method — why this is a control and not a different model.** The lattice's
Brillouin edge *is* the model's UV cutoff, ω_max = √(m² + 4/a²), so the cutoff
is moved by the lattice spacing `a`. Spacing is varied at **fixed physical
geometry** (the sourced 19.2 mm separation, the 0.22 ns window, the buffer) and
**fixed physical rates**, with the field renormalisation made explicit
(`notebook.lattice_at_spacing`): discretising at spacing `a`, the variable
canonically conjugate to the site momentum is x_j = √a·φ_j, so

    K_a = harmonic_chain_K(N, m·a) / a²,     λ → λ / √a,

the second factor because a pointlike detector couples to the *physical* field
φ(ξ_d) = x_j/√a (amplitude) or ∂_t φ(ξ_d) = p_j/√a (derivative). At `a = 1`
this is the identity and reproduces the fourth-pass archive **bit-for-bit**
(`tests/test_circuit_mapping.py`), and the √a is pinned independently by a
continuum check: ⟨φ(0)φ(r)⟩ at fixed physical r converges to K₀(mr)/2π as
a → 0. The lattice cutoff is a **hard band edge** while the device's is the
exponential C(ω) = e^{−|ω|/2Ω_cut} of Eq. (9), so the bracket is on the cutoff
**scale**, not its shape.

| spacing a | UV cutoff | N | dephasing anchors: max E_N | contrast anchors: max E_N | worst ledger |
|---|---|---|---|---|---|
| 1.0 (the archives) | 14.67 GHz | 31 | **0** | 2.167e-4 | 3.4e-11 |
| 0.6 | 24.38 GHz | 50 | **0** | 2.471e-4 | 1.2e-10 |
| 0.4 | 36.53 GHz | 74 | **0** | 2.697e-4 | 9.4e-11 |
| **0.292** | **50.01 GHz** (the device's own) | 103 | **0** | 2.600e-4 | 3.6e-10 |
| 0.2 | 73.01 GHz | 149 | **0** | 2.417e-4 | 4.7e-10 |

**Result.** The bracket contains the device's 50 GHz, and across all of it:

- **every dephasing anchor is exactly zero at every cutoff** — the negative
  device statement is cutoff-independent;
- **no anchor verdict drifts anywhere in the bracket**, at any of the three
  floors, for the dephasing anchors *or* the contrast anchors;
- the **temperature wall does not move**: at Γ_φ = 1e4 Hz the band-low wall
  is [30, 35] mK at 14.7 GHz, at 24.4 GHz and at 50.0 GHz alike;
- the **dephasing tolerance does not move**: the band-low Γ_φ crossing is
  bracketed by [1e4, 1e5] Hz at every cutoff, so the margin to the device's
  digitised 1e7 Hz is **≥ 100×** at 14.7, 24.4 and 50 GHz — the same number
  the fourth pass reported, now measured rather than inferred;
- where E_N is *alive*, the cutoff changes its size by at most **+25 %**
  (2.167e-4 → 2.697e-4, peaking near 36.5 GHz and turning over by 73 GHz), and
  at the device's own 50 GHz it is 20 % above the archives' value — the same
  scale the proposal's App. C reports for Ω_cut → ∞, which was the inference
  the fourth pass could only quote.

Nothing on any surface reaches the derived floor at any cutoff (max E_N
2.33e-4 at 50 GHz; 0 % of the surface GO at 1e-3 everywhere in the bracket).

**Audit event (recorded per the anomaly protocol).** On the first pass of this
bracket the energy-ledger closure audit **fired** on one cell — a = 0.292
(N = 103), T = 80 mK, Γ_φ = 1e9 Hz — with a defect of −1.0107e-9 against the
**absolute** 1e-9 closure criterion: a 1 % overshoot of the criterion and
4.45e-12 of that state's own energy (⟨H⟩ = 227.0, 16384 channel entries plus
8192 work entries), i.e. float64 accumulation over the entry count rather than
a physics defect. **The criterion was not relaxed**; the single synthetic Γ_φ
extension column (1e9 Hz — 0.7 decades above the top device anchor, not a
distant outlier) was dropped from the bracket surfaces instead. Both device
anchors (1e7 and 2e8 Hz) are retained, E_N is exactly 0 on the dropped column
at every cutoff measured — and that column is *retained with E_N = 0 on all 64
of its rows* in the committed `rows.npz` and `rows_xp.npz`, so nothing was
removed from the evidence — and **every row reported here passed the unrelaxed
audit** (worst defect 4.73e-10).

**The diagnosis is confirmed, and the exposure is not confined to the dropped
column.** An independent 41-run scaling study reproduces the defect exactly
(1.0107e-9 at halving level 9), shows it scaling with entry count (pooled
log-log slope +0.81 over N = 48…24576) and shows it ⟨H⟩-proportional (relative
defect 4.45e-12 at ⟨H⟩ = 227 against 4.49e-12 at ⟨H⟩ = 55.3) — float64
accumulation, as diagnosed. But **Γ_φ = 2e8 Hz, a *retained device anchor*,
also crosses the absolute criterion (1.0527e-9) at halving level 9**; the rows
reported here pass only because the dt gate converges at level 8. Dropping the
synthetic column therefore removed the **trigger, not the exposure**. No
verdict moves either way — E_N is exactly 0 on those cells at every cutoff, so
the device statement is untouched — but the honest summary is: *the event was
real, it was numerics, and the marginal element is the criterion itself*, an
absolute 1e-9 that scales with neither ⟨H⟩ nor entry count.

**Resolved (2026-09-04) by correcting the criterion, not by relaxing it.**
Because the closure identity telescopes exactly, roundoff is its *only* error,
and its size is set by the energies being differenced and by how many
accumulations were booked; `vacuum.audits.EnergyLedger` therefore now applies
tol = max(1e-12, 4ε·√n_modes·n_ops·scale), which is **stricter** than the
absolute 1e-9 wherever the roundoff floor is below it (~1000× at the floor) and
is scale-covariant — and, stated plainly, **102× looser at these two cells**
(1.02e-7 against the old 1e-9), which is the trade: their own float64 floor is
~1e-9. The criterion bounds error relative to a protocol's *own* energy scale,
so on a cell like this a violation small against ⟨H⟩ = 227 but large against
its own entry (a 10 % mis-booking of a 1e-6 work term) would not fire, while a
100 % one would; fine-grained book errors are resolvable only where the
absolute floor binds.
Both marginal cells were re-run under it (a = 0.292, T = 80 mK, `'xp'`, derived
coupling): at halving level 9 the dropped 1e9 Hz column closes to −9.985e-10
against a tolerance of 1.03e-7 (**103× margin**) and the retained 2e8 Hz anchor
to −1.0516e-9 against 1.02e-7 (**97× margin**, reproducing the recorded
1.0527e-9 to 0.1 %); at level 8 they close to −1.26e-10 and −1.21e-10
(≈ 410× margin). **Neither fires.** The dropped column stays out of the
*archived* surfaces — restoring it needs a re-run of the bracket, and it still
carries no verdict information (E_N = 0.0 at this cutoff at both levels) — but
it is no longer excluded because of an audit.

The event and the action are carried in `data/cutoff_bracket.json →
audit_event`; that archive holds the **superseded wording** from its producing
run (it says "8192 booked channel entries", "⟨H⟩ = 230" and "5 decades above"),
which the generator and this MANIFEST now correct — the archive's *numeric*
rows are unaffected and were not regenerated for a prose fix.

---

## What is NOT claimed

- **No positive statement about the device's yield.** The device statement of
  the `'xp'` re-run is a *negative* one (dephasing forbids resolvable
  harvesting); nothing on any `'xp'` surface reaches the derived floor, so no
  measurable amount is predicted for anything.
- Nothing about the cutoff's *shape*: Deliverable 8 brackets the cutoff
  **scale** (14.7–73 GHz, through the device's 50 GHz) with a hard Brillouin
  edge; the device's exponential C(ω) = e^{−|ω|/2Ω_cut} is not reproduced.
- Nothing about the sign of λ, which the oscillator model cannot see.
- Nothing that treats the surface axes as experimental parameters; the device
  anchors on them are.
- Nothing beyond `'xx'` coupling, harmonic detectors, a 1+1 lattice with a
  14.6 GHz UV cutoff and a synthetic 1.46 GHz IR cutoff; in particular nothing
  about the derivative-coupling spacelike result of the proposal.
- No Γ₁ or Γ_φ *value* from the plot-digitised inputs beyond their stated
  order of magnitude.

## Reproducing

```
.venv/bin/python papers/circuit-qed-noise-thresholds/notebook.py --workers 14   # ~13 min on 16 cores, writes data/
.venv/bin/python papers/circuit-qed-noise-thresholds/notebook.py --quick        # ~5 s, 19 rows
.venv/bin/python papers/circuit-qed-noise-thresholds/notebook.py --device-point --workers 14  # ~2 min, 270 rows at the derived coupling
.venv/bin/python papers/circuit-qed-noise-thresholds/notebook.py --xp-map --workers 14        # ~5.5 min, 922 rows with the proposal's derivative coupling
.venv/bin/python papers/circuit-qed-noise-thresholds/notebook.py --gap-tracking --workers 14  # ~30 s, lambda(t) tracking the gap
.venv/bin/python papers/circuit-qed-noise-thresholds/notebook.py --ir-control --workers 14    # ~2 min, the IR-mass control
.venv/bin/python papers/circuit-qed-noise-thresholds/notebook.py --cutoff-bracket --workers 14 # ~19 min, the UV-cutoff bracket (14.7-73 GHz)
.venv/bin/python -m pytest tests/test_noise_threshold_smoke.py -q               # schema + boundary-finder guard
.venv/bin/python -m pytest tests/test_circuit_mapping.py -q                     # the coupling derivation (19 tests)
.venv/bin/python -m pytest -q                                                   # admissibility gate
```

`--surfaces dephasing,coupling` runs a subset; `--workers 1` runs sequentially
through the identical per-point code. The archives record the generator's
sha256 and git HEAD, so a re-run that disagrees is a build difference, not a
mystery.
