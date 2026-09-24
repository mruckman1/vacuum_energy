# vacuum.detectors — the harvesting engine (Layer 2)

Status: **built and anchored** (`docs/PLAN_LAYERS_2_3.md`, M2.1–M2.7). This README is
the original build brief; the shipped surface is `vacuum/detectors/__init__.py`'s
`__all__`, and the binding signatures are the ones in the plan document, which
supersedes the sketch below where the two differ. Everything runs on the
`vacuum.core` covariance stack (R block-ordered, V_vac = I/2, ν ≥ 1/2, all
negativities in nats).

Anchors in force: PKMM death lines to 0.26 % (spec 2 %) and the paper's own closed
forms Eqs. (29)–(32) to 6e−16; the Simidzija null at exactly 0 over 200 random draws
per backend; spacelike |M_comm| = 1.5e−19 (spec 1e−12); Gate A (perturbative ↔
nonperturbative) deviation exponent ≥ 2.009 at R² ≥ 0.99999. Tests:
`tests/test_kernels.py`, `test_nonperturbative.py`, `test_udw.py`,
`test_gate_crossvalidation.py`, `test_nogo_communication.py`, `test_imperfections.py`.
Derivative coupling (`coupling='xp'`, the ∂ₜφ model of Teixidó-Bonfill & Martín-Martínez,
PRD 110, 105016 (2024); `vacuum/detectors/udw_derivative.py`) is in force too: Gate A for xp
(`tests/test_gate_xp.py`, deviation exponent ≥ 2.005 at R² ≥ 0.99998 over three setups, with
canaries), the derivative-coupled single-detector response against an exact mode sum to 1e−12,
and the M2.5 light-contact reproduction (`tests/test_xp_coupling.py`: |M_vac|/|M| = 0.999 under
derivative coupling vs |M_comm|/|M| = 0.365 under amplitude coupling for a causally connected
pair). Its noise-survival map lives in `papers/derivative-coupling-noise-survival/`.

**Build, perturbative half.** Unruh–DeWitt detectors to second order in the coupling:
Wightman functions on the lattice and in the continuum, double quadrature against smooth
switching functions (sharp switching produces spurious divergences and is banned by
default), smeared detector profiles, and the full two-detector negativity formula from
the local-excitation and nonlocal terms. Includes the estimator that splits detector
correlations into a genuinely vacuum-mediated part and a communication part sourced by
the field's retarded propagation — the split that makes later claims honest.

**Build, nonperturbative half.** Oscillator detectors instead of qubits, so
detectors + field form one Gaussian system evolving symplectically — exact to machine
precision, no perturbation theory. Circuit-QED parameter sets from the Waterloo program
(flux qubits with tunable ultrastrong couplers; the 2026 variable-gap detector model),
including gap variation during coupling. Realistic imperfections enter as Gaussian
channels via `vacuum.core.channels`: thermal waveguide occupation, photon loss,
dephasing, finite switching bandwidth.

**Validation anchors.** The harvested-negativity maps of Pozas-Kerstjens &
Martín-Martínez, PRD 92, 064042 (2015), across separation, energy gap, and smearing;
the Simidzija–Martín-Martínez no-go for delta-coupled detectors, PRD 98, 085007 (2018),
verified as an exact null; communication-vs-vacuum split per Tjoa–Martín-Martínez.

**Discovery targets.** The go/no-go noise boundary for circuit-QED harvesting (the
deliverable: a preregistered noise-threshold map for the not-yet-performed
superconducting demonstration). Parameter pockets maximizing harvested negativity per
unit switching energy. Whether the derivative-coupling result — entanglement drawn
primarily from field correlations even for causally connected detectors — survives
realistic noise.

**Pivot trigger.** If harvested negativity under honest noise models sits orders of
magnitude below detectability for all realistic parameters, pivot to mutual-information
and coherence harvesting and to the QET emphasis of Layer 3; the negative result, with
clean audits, is written up as the boundary map it is.

## Planned API (house style, subject to docs/API.md conventions)

```
# --- perturbative (UDW, qubit detectors, second order) ---
wightman_lattice(K, sites, ts) -> W               # ⟨0|φ_i(t) φ_j(t′)|0⟩ from mode expansion of K
switching(kind, T, t0, **shape) -> chi            # smooth χ(t); kind='gaussian'|'cos2'; no sharp cutoffs
smearing(kind, center, width, N) -> F             # spatial profile F(x) on the lattice, Σ F = 1
udw_pair_state(K, det_A, det_B, lam, gap) -> rho4 # joint 2-qubit density matrix to O(λ²);
                                                  #   det_* = (F, chi, site) descriptors
harvested_negativity(rho4) -> float               # log-negativity of the qubit pair, nats
communication_split(K, det_A, det_B, ...) -> (M_vac, M_comm)
                                                  # Tjoa–Martín-Martínez estimator: vacuum-mediated vs
                                                  #   retarded-signaling contributions to the M-term

# --- nonperturbative (oscillator detectors, one Gaussian system) ---
attach_detectors(K, sites, gaps) -> K_tot         # composite coupling matrix, detectors as extra modes
protocol_evolve(V, K_tot, g_of_t, gap_of_t, ts) -> V_out
                                                  # time-ordered product of exp(dt·Ω H(t)) steps
detector_block(V_out, n_field) -> V_AB            # 4×4 reduced covariance of the two detector modes
harvested_log_negativity(V_AB) -> float           # E_N = max(0, −ln 2ν̃_min), nats (core.log_negativity)
with_imperfections(V, eta, nbar, dephasing) -> V  # circuit-QED noise via vacuum.core.channels
load_experiment(name) -> params                   # reads experiments/ parameter files (see its README)
```

Every protocol run passes through `vacuum.audits` (ledger, passivity, causality,
precision) before any number is reported.
