# vacuum.opt — the differentiable vacuum (Layer 6, planned)

Status: build brief; not yet implemented. Requires the `.[diff]` extra (JAX); RL search
and big sweeps want a single GPU. The JAX mirror must reproduce `vacuum.core` to
tolerance before any optimized number is trusted (same conventions: ħ = 1, V_vac = I/2,
ν ≥ 1/2, block ordering, nats).

**Build.** Rewrite the Gaussian core in JAX so every pipeline in Layers 2, 3, and 5 is
end-to-end differentiable: gradients flow through symplectic evolution, switching
waveforms, detector trajectories, gap schedules, modulation profiles, and multi-detector
layouts. Optimize under honest constraints — fixed energy budget, fixed signaling bound,
fixed noise floor — for objectives that matter: harvested negativity, QET delivered
energy, inequality-ratio minimization, amplification per loss. Where landscapes are
rugged, wrap the simulators for reinforcement-learning search. Add a distillation back
end: feed optimized harvested two-detector states through standard protocols (BBPSSW,
DEJMPS) and compute the actual yield of usable Bell pairs per protocol run — the
vacuum-to-Bell-pair compiler, with an exchange rate printed at the end.

**Inverse design.** Prescribe a target correlation or negativity profile and solve for
the coupling matrix that produces it as a ground state — programmable vacua as an
inverse spectral problem, the cleanest expression of the thesis that the vacuum's
structure is a design variable.

**Validation anchors.** JAX core vs numpy core: entropies, negativities, evolutions
agree to float64 tolerance on the Layer 1 anchor suite. Gradient checks against finite
differences through a full harvesting protocol. Optimizer re-discovers published
hand-tuned optima (e.g. the Pozas-Kerstjens–Martín-Martínez best-case parameters) before
being allowed to claim improvements beyond them. Distillation yields on known two-mode
squeezed inputs match the BBPSSW/DEJMPS literature formulas.

**Why this layer can produce discoveries.** The harvesting and QET literatures
overwhelmingly evaluate hand-chosen configurations; systematic gradient-based protocol
search over the full waveform space is nearly untouched, so order-of-magnitude
improvements are plausible simply because nobody has looked with these tools.

**Pivot trigger.** If optimized gains over published protocols are marginal (< 2×), the
result still standardizes the field's baselines; effort shifts to inverse design, which
has value independent of the margin.

## Planned API (house style)

```
# --- differentiable core (vacuum.opt.gjax mirrors vacuum.core.gaussian) ---
gjax.ground_state_cov(K) -> V                    # jit/grad-safe twins of the numpy signatures
gjax.evolve, gjax.log_negativity, gjax.entropy, ...
                                                 # identical semantics, verified against numpy core

# --- protocol optimization ---
make_objective(kind, **cfg) -> f(theta)          # kind ∈ {'negativity','qet_energy','qei_ratio',
                                                 #   'gain_per_loss'}; theta parametrizes waveforms,
                                                 #   schedules, layouts
constraints(energy_budget, noise_floor, signaling_bound) -> C
optimize(f, C, theta0, method='adam'|'lbfgs') -> (theta_star, history)
rl_search(env_name, cfg) -> policy               # for rugged landscapes; env wraps the simulators

# --- distillation back end ---
distill_yield(V_AB, protocol='DEJMPS', n_rounds) -> (bell_pairs_per_run, fidelity)
vacuum_to_bell_rate(protocol_cfg) -> exchange_rate    # ebits out per joule of switching energy in

# --- inverse design ---
inverse_design(target, ansatz='banded', reg) -> K     # coupling matrix whose ground state matches a
                                                      #   target correlation/negativity profile;
                                                      #   K ≻ 0 enforced (stable lattice)
design_residual(K, target) -> float
```

Optimized protocols are candidates, not results: each θ★ is re-run through the plain
numpy stack and the full audit suite, and only numbers surviving that round trip are
admissible under the rule in `papers/README.md`.
