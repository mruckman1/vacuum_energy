# API contract — Layer 0/1 core

This file is the coordination contract for the Vacuum Program codebase. Every module
codes against these conventions. Change them only with a repo-wide migration.

## Global conventions (non-negotiable)

- **ħ = 1.** Single-mode vacuum covariance is **V = I/2**; symplectic eigenvalues of
  physical states satisfy **ν ≥ 1/2**, with ν = 1/2 the vacuum floor (this matches the
  plan's "danger zone near 1/2" language).
- **Quadrature ordering: block form.** For N modes the operator vector is
  R = (x₁,…,x_N, p₁,…,p_N). The symplectic form is Ω = [[0, I_N], [−I_N, 0]],
  so [R_i, R_j] = i Ω_ij.
- **Covariance definition:** V_ij = ½⟨{R_i − ⟨R_i⟩, R_j − ⟨R_j⟩}⟩.
- **All entropies and negativities in nats** (natural log). Log-negativity:
  E_N = Σ_k max(0, −ln(2 ν̃_k)) over symplectic eigenvalues ν̃ of the partial transpose.
- A symplectic matrix S satisfies S Ω Sᵀ = Ω; states evolve V → S V Sᵀ.
- Mode index sets are always Python sequences of 0-based mode indices (not quadrature
  indices); functions internally map mode k → quadrature rows (k, N+k).

## vacuum.core.gaussian

```
Omega(N) -> (2N, 2N) ndarray                  # symplectic form, block convention
ground_state_cov(K) -> V                      # ground state of H = ½(pᵀp + xᵀKx):
                                              #   V = diag-block( ½ K^{-1/2}, ½ K^{1/2} )
symplectic_eigenvalues(V) -> (N,) ndarray     # spectrum of |iΩV|, descending; physical ⇒ ≥ ½
williamson(V) -> (S, nu)                      # V = S · diag(nu, nu) · Sᵀ, S symplectic
entropy(V) -> float                           # Σ_k [(ν+½)ln(ν+½) − (ν−½)ln(ν−½)], safe at ν=½
reduce(V, modes) -> V_A                       # reduced covariance of the listed modes
partial_transpose(V, modes_B) -> V_pt         # flip p-sign on modes_B
log_negativity(V, modes_A, modes_B) -> float  # E_N of the A∪B reduced state, PT on B
mutual_information(V, modes_A, modes_B) -> float
entanglement_hamiltonian(V_A) -> G            # ρ_A ∝ exp(−½ RᵀGR); via Williamson:
                                              #   G = (S⁻¹)ᵀ diag(g(ν), g(ν)) S⁻¹,
                                              #   g(ν) = ln((ν+½)/(ν−½)); clamp/escalate near ν=½
modular_energy(G, V_A) -> float               # ½ Tr(G V_A) — used for δ⟨K⟩ = ½ Tr(G δV_A)
symplectic_from_quadratic(H_mat, t) -> S      # exp(t Ω H_mat) for H = ½ RᵀH_mat R
evolve(V, S) -> S V Sᵀ
mean_energy(V, K) -> float                    # ⟨H⟩ = ½ Tr(V_pp) + ½ Tr(K V_xx); a (2N, 2N) K is
                                              #   read as a general quadratic form H_mat (x–p terms
                                              #   allowed): ⟨H⟩ = ½ Tr(H_mat V), identical on diag(K, I)
```

## vacuum.core.channels

Gaussian channels act as V → X V Xᵀ + Y (complete positivity: Y + i(Ω − XΩXᵀ)/2 ⪰ 0):

```
loss(V, modes, eta, nbar=0.0)       # beamsplitter to a thermal environment
thermal_noise(V, modes, nbar_added) # classical Gaussian noise addition
amplifier(V, modes, gain, nbar=0.0) # phase-insensitive amplification
apply_channel(V, X, Y)
```

## vacuum.core.models

```
harmonic_chain_K(N, m, bc='periodic'|'dirichlet') -> K
    # H = ½ Σ [πᵢ² + m²φᵢ² + (φᵢ₊₁ − φᵢ)²]  (plan's normalization)
```

## vacuum.core.radial  (Srednicki geometry)

```
srednicki_K(N, l) -> K   # partial-wave radial Hamiltonian, Srednicki PRL 71, 666 (1993)
sphere_entropy(N, n, l_max=None) -> float  # S(n) = Σ_l (2l+1) S_l(n), adaptive l cutoff
```

## vacuum.core.precision

Escalation ladder float64 → longdouble → mpmath, triggered when min(ν) − ½ < tolerance
or when Williamson conditioning degrades. Expose a context/helper the observable
functions call; must be testable directly.

## vacuum.fermions.correlation

```
tight_binding_C(N, filling=0.5, bc=...) -> C   # C_ij = ⟨c†_i c_j⟩ in the ground state
block_entropy(C, sites) -> float               # from eigenvalues of the site-block of C
entanglement_hamiltonian(C_A) -> h             # h = ln((1 − C_A)/C_A)  (Peschel)
```

## vacuum.audits

```
energy_ledger(...)        # ⟨H⟩ accounting before/after operations; must close
                          #   (closure criterion scales with the booked ⟨H⟩ and
                          #    entry count: max(1e−12, 4ε√n_modes·n_ops·scale))
passivity_audit(...)      # local symplectics on the vacuum never lower ⟨H⟩
causality_audit(...)      # SΩSᵀ = Ω preservation + Lieb–Robinson commutator decay
precision_audit(...)      # flags min(ν) − ½ below tolerance; triggers escalation
```

An audit failure on validated code is an *event*, not an error — see the anomaly
protocol in PLAN.md. Audits raise/flag loudly; they never silently pass.

## Testing

- `tests/` at repo root, pytest, run with `.venv/bin/python -m pytest`.
- Validation anchors are permanent regression tests; each records the measured number
  in its assertion message so the value is visible in failures.
- Keep any single test under ~60 s on a laptop; cache expensive lattice sweeps in
  module-scoped fixtures.
