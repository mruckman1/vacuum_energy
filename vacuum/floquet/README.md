# vacuum.floquet — time-modulated vacua (Layer 5)

Status: **implemented** (build 77a2889+, tests `tests/test_floquet_*.py`, result
candidate `papers/time-modulated-vacua/`). Pure `vacuum.core` machinery:
periodically driven quadratic Hamiltonians are still symplectic on the covariance
matrix (ħ = 1, block ordering, ν ≥ 1/2, negativities in nats).

**Build.** Floquet dynamics on the lattice field: periodically modulated mass or
couplings, Mathieu-type stability analysis, momentum-gap identification, photon-pair
production spectra, entanglement growth of amplified mode pairs, and squeezing spectra —
the dynamical Casimir effect and the photonic-time-crystal mechanism in one engine, with
loss and finite pulse trains as Gaussian channels (`vacuum.core.channels` plus the
K0-matched multimode bath of `runs.matched_bath_cov`, which leaves the undriven vacuum
exactly invariant).

**Validation anchors (permanent regression tests, measured values in the assertion messages).**

| anchor | test | measured |
|---|---|---|
| Mathieu chart from the single-mode parametric oscillator: \|tr S_F\| = 2 on the scipy characteristic curves (m = 1..4, depths 0.1/0.3/0.6); numerical tongue edges vs `mathieu_a/b`; 16 × 300 grid classification | `test_floquet_mathieu.py` | ≤ 1e-14; ≤ 1e-8 rel (tol); 0/4800 mismatches |
| two-mode-squeezed (k, −k) pairs: E_N = 2 arcsinh √⟨n_k⟩ for every pair; growth rate = ln max\|eig S_F\| | `test_floquet_pairs.py` | ≤ 1.5e-10 (tol 1e-8); 5e-7 |
| Wilson et al. 2011 DCE spectrum in the file's regime (ε = 0.0625, ω_d = 11.30 GHz, c₀ = 0.4c): parabola prefactor, shape, (δL_eff)² scaling, 50 mK stimulated factor | `test_floquet_dce.py` | 1.4 % / 1.9 % RMS / exponent 1.99 / ≤ 2 % (tols 3 %/3 %/0.03/5 %) |
| ledger closes against the drive, W = Σ ω_k n_k; passivity of the undriven vacuum; fast path = `run_protocol` | `test_floquet_pairs.py`, `test_floquet_runs.py` | defect 0–1e-13 (tol 1e-9); 1e-12 |

## API (implemented; house style)

```
# --- drives and one-period maps (drives.py, monodromy.py, mathieu.py) ---
modulated_K(K0, depth, omega_mod, profile='cos', kind='coupling'|'mass'|'pattern'|'boundary', ...) -> Drive
    # Drive is the callable t -> K(t) (drops into vacuum.protocol.GeneratorStep); profiles cos/sin/
    # triangle/square/callable/((value, fraction), ...); 'boundary' = harmonic L_eff(t) of the
    # SQUID mirror (Johansson 2010 Eq. 10/17), K0 from dce_chain_K(N, L_eff)
floquet_map(K_of_t, period=None, n_steps=64, method='cf4'|'midpoint', conv_tol=1e-10) -> FloquetMap
    # monodromy: exact static exponentials for piecewise-constant profiles; CF4 (4th-order
    # commutator-free Magnus) or midpoint otherwise, dt-halving gate on the map, provenance recorded
stability_chart(K0, depth_grid, omega_grid, ...) -> StabilityChart   # max |eig S_F|; mode-resolved
                                                                      # (stacked 2x2, gated) for diagonal drives
momentum_gaps(chart) -> bands[i][j]                                   # unstable k-bands (Band records)
mathieu_parameters / mathieu_characteristic / mathieu_tongue_edges    # scipy reference values

# --- production, squeezing, entanglement (spectra.py) ---
pair_spectrum(V0, S_F, n_periods, K0) -> n_k          # <n_k> in the normal modes of K0
traveling_wave_basis(K0) -> basis                     # (k, -k) symplectic recombination of degenerate pairs
squeezing_spectrum(V, K0 | basis=) -> r_k             # arcsinh sqrt<n_k>
pair_negativity(V, k, K0 | basis=) -> E_N             # (k, -k) block, mp-margin rule
entanglement_growth(V0, S_F, n_periods, K0) -> EntanglementGrowth

# --- realism and bounds (runs.py) ---
floquet_run(V0, drive, n_periods, eta=1, nbar=0, bath='matched') -> FloquetRun   # ledger + passivity, fast path
lossy_floquet(V0, S_F, eta, nbar, n_periods, K0=...) -> V
run_floquet_protocol(...) / floquet_steps(...)        # the same drive as vacuum.protocol steps (CF4 pieces)
pulse_train(V0, drive, n_pulses, gap, ...) -> PulseTrainResult   # ramped envelopes through run_protocol, gated
amplification_bound_scan(depth_grid, omega_grid, eta_grid) -> AmplificationScan   # converged max-gain surface
max_gain_over_frequency, bang_bang_bound, loss_threshold, loss_per_period, quality_factor_from_eta

# --- Wilson 2011 anchor (dce.py) ---
wilson_regime(omega_d_lattice) ; dce_spectrum(V, K0, t, omega_d, dL, V_ref=) ; parabola_fit

# --- composite with Layer 2 (composite.py) ---
pumped_harvest(K0, drive, sites, gaps, lam_max, n_prep, n_win, ...) -> PumpedHarvest
    # Floquet-prepare, then harvest; ONE run_protocol call per point (passivity claim, precision
    # audit on the detector block, ledger closure with pump + switching work), gated at 1e-9
```

**Discovery outputs** (`papers/time-modulated-vacua/`): the conjectured per-cycle
amplification bound ln max|eig S_F| ≤ artanh(d) (bang-bang optimum, attained; cosine
reaches (π/4)d(1 + c₂d² + …)), the loss threshold η_th = 1/λ_max² with the Q mapping
η = exp(−π/Q) at the principal resonance, and the pump-enhanced-harvesting map with its
causal (spacelike-window) control.

Amplification runs are the one place energy legitimately enters: the ledger charges it
all to the drive, and passivity of the *undriven* vacuum remains a standing audit.
