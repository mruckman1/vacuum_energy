# vacuum.qet — quantum energy teleportation and passivity boundaries (Layer 3)

Status: **built and anchored** (`docs/PLAN_LAYERS_2_3.md`, M3.1–M3.3). This README is
the original build brief; the shipped surface is `vacuum/qet/__init__.py`'s `__all__`,
and the binding signatures are the plan document's — where the sketch below disagrees
(e.g. it names `qet_chain_protocol(psi, site_A, site_B, meas, feedback)`; the shipped
entry point is `run_chain_qet(L, g, A, B, ...)` per M3.2), the plan wins. Codes against
`vacuum.core` conventions (ħ = 1, ν ≥ 1/2, block ordering, nats) and, for spin chains,
small exact Hilbert spaces plus DMRG via the optional `.[tensor]` extra (TeNPy). The
SDP rung of M3.3 sits behind `.[sdp]` (cvxpy); both extras import lazily inside their
guarded modules, so a core-only install imports and runs everything but those rungs.

Anchors in force: Hotta's closed forms to 2.7e−14 with E_A ≥ E_B everywhere (worst
margin +4.69e−4) and no-signaling at exactly 0; ED ↔ DMRG dip agreement 4.6e−10
(5.1e−15 at a common exact rotation angle), χ-doubling gated with measured teeth;
FFH ↔ SDP verdicts agreeing with worst energy gap 2.1e−12 (spec 1e−9). Tests:
`tests/test_hotta.py`, `test_qet_chains.py`, `test_slp.py`.

Ikeda (2023) IBM protocol (`vacuum/qet/ikeda.py`, core-only; `ikeda_qiskit.py` behind
`.[ibm]`): `ikeda_ibm(params)` runs the paper's three readout circuits (Fig. 1, Eqs.
(7)–(14)) exactly from the gate list of `ikeda_circuit_spec`, parameters from
`experiments/ibm_qet/`; closed-form residuals ≤ 1.6e−15, every printed Table I/II
analytical value matched to one unit in the last printed digit, feed-forward ≡ deferred
to 6e−17, no-signaling 3e−16. `ikeda_ibm_aer` reruns it on Aer with a noise model built
only from Table III (ibm_cairo) numbers; `run_on_backend(circuit, backend, shots)` is
the one-line hardware hook (never exercised on a device here). Tests: `tests/test_ikeda.py`.

**Build, QET.** The minimal Hotta model exactly: two coupled qubits, Alice's measurement
injecting E_A, a classical bit, Bob's conditional rotation extracting E_B, sweeping the
full parameter space and confirming E_A ≥ E_B everywhere as a *standing audit* rather
than a result. Then scale: critical Ising chains via DMRG, measurement-plus-feedback as
quantum channels, watching local energy density at Bob's site dip below the ground-state
value and mapping how far, how fast, and how it decays. Reproduce the Ikeda 2023
IBM-hardware protocol in simulation, then — the program's one hardware excursion — on the
free tier of actual devices.

**Build, passivity boundaries.** Numerically chart strong local passivity
(Frey–Funo–Hotta 2014; Alhambra et al. 2019): for which states, couplings, and regions is
local energy extraction impossible even under the most general local operations, and
exactly where does classical communication reopen it. No systematic numerical maps of
this boundary exist. This is the seed of the leap-tier resource theory (PLAN.md, L3).

**Validation anchors.** Hotta (2008; 2011 review) minimal-model energetics reproduced
exactly, E_B formula included; Ikeda (2023) IBM protocol values in noiseless simulation;
E_A ≥ E_B holding over every swept point (ledger + passivity audits); Bob's extraction
vanishing when the classical channel is cut (causality audit).

**Discovery targets.** The strong-local-passivity phase maps. The exchange rate between
injected measurement energy and remotely available energy as a function of entanglement
structure — the beginning of a quantitative answer to what "manipulating energy already
there" can achieve. Decay of extractable energy with Alice–Bob distance on critical
chains.

**Pivot trigger.** None specific to this half of Layer 3; if the sibling sharp-constant
search (`vacuum.inequalities`) stalls, its effort redirects to the interacting-field leap
while QET work continues.

## Planned API (house style)

```
# --- minimal Hotta model (exact, dim-4 Hilbert space; not Gaussian) ---
hotta_minimal(h, k) -> model                     # two-qubit Hamiltonian, ground state, couplings
hotta_protocol(model, meas_basis, theta) -> ledger
                                                 # ledger = {E_A, E_B, S_AB, bit}; E_B > 0 = success
hotta_sweep(h_grid, k_grid, theta_grid) -> map   # standing audit: assert (E_A ≥ E_B).all()

# --- chains ---
ising_ground_state(L, g, method='exact'|'dmrg') -> psi     # critical point g = 1
local_energy_density(psi, site) -> float         # ⟨T_00⟩-analogue; report relative to ground value
qet_chain_protocol(psi, site_A, site_B, meas, feedback) -> (psi_out, ledger)
energy_dip_profile(psi_out, sites) -> profile    # how far below ground, how it decays in space/time
ikeda_ibm(params) -> ledger                      # reproduces Ikeda (2023); params from experiments/

# --- strong local passivity ---
slp_certificate(rho, H, region) -> bool          # is any local CP map on `region` energy-extracting?
slp_map(state_family, coupling_grid, region) -> bool_map
                                                 # the boundary chart; LOCC variant flags where
                                                 #   classical communication reopens extraction
```

Every protocol carries its full energy ledger via `vacuum.audits.energy_ledger`; any
apparent net gain without an external drive or classical-information input is a bug by
definition (Pusz–Woronowicz), and audit failures on validated code invoke the anomaly
protocol in PLAN.md.
