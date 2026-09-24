# vacuum.geometry — geometry from entanglement, computationally (Layer 4)

Status: **implemented** (build 77a2889 lineage; see `tests/test_geometry_*.py`).
Uses `vacuum.core` / `vacuum.fermions` for correlations and mutual information
(nats throughout); the MERA needs only numpy (+ `opt_einsum` if present for
shape-aware contraction paths; a static path table is the fallback).

**Build.** Two of Layer 4's three instruments live here (the third, the modular lab, is
`vacuum.modular`). First, a MERA implementation for the critical Ising model: entropies
from minimal cuts through the network, and an emergent-geometry probe that defines
distances from mutual information (d ∼ −ln I between regions) and tests how closely the
reconstructed metric approaches the hyperbolic geometry the Swingle correspondence
predicts. Second, the HaPPY holographic code in stabilizer formalism, where bulk
reconstruction and the Ryu–Takayanagi formula can be verified by direct computation.

**The honest experiment.** `failure_map` runs the metric-reconstruction pipeline on
systems that are *not* holographic by construction — massive chains, disordered
couplings, detuned Ising chains — and tabulates where and how the correspondence
degrades. The result candidate lives in
`papers/geometry-from-entanglement-failure-map/`.

## Modules

| module | contents |
|---|---|
| `mera.py` | scale-invariant ternary MERA (Evenbly–Vidal 2009; Pfeifer–Evenbly–Vidal 2009), Z₂-symmetric tensors, cached optimized tensors in `data/`, energy, parity-resolved scaling dimensions, central charge (PEV Eq. 16), minimal cuts, graph geodesics |
| `migeometry.py` | `mi_matrix`, `mi_distance`, flat and hyperbolic stress-minimizing embeddings, Gromov δ, log/linear distance law, `failure_map` over four model families |
| `happy.py` | stabilizer engine (GF(2) tableau with signs), the {5,4} pentagon code and {6,4} hexagon state (HaPPY 2015), greedy wedge, max-flow min cut, exact entropies, exact and greedy bulk reconstruction |
| `ising_exact.py` | exact Majorana reference for the TFIM/Kitaev chain (infinite chain closed forms and quadrature; finite disordered chains) |
| `jacobson.py` | `entanglement_equilibrium_residual` (exact first law / Bisognano–Wichmann local form) and `constrain_dynamics` for the toy-Jacobson leap; `equilibrium_residual` with the CHM ansatz's discretisation options (β at site centres or bond midpoints, Richardson mixes, boundary-adapted weight near a Dirichlet wall), balls at arbitrary positions, and the ball zero mode either excluded by parity or projected out of the variation; `k4_sensitivity` for the (c₂, c₃) degeneracy test |

## Measured anchors (this build, χ = 8 cache `data/mera_ising_chi8.npz`)

- MERA ground energy density: rel. error 8.9e-6 vs e₀ = −4/π (test gate 5e-5;
  Evenbly–Vidal report 5 digits at χ = 4 after long optimization).
- Scaling dimensions (two-site scaling superoperator, parity-resolved):
  Δ_σ = 0.1238 (1/8), Δ_ε = 1.0014 (1), descendants 1.1245, 1.1303 (9/8) and
  1.95–2.00 (2); gates 4% / 3% (PEV χ = 22: 0.124997, 1.0001).
- Central charge, PEV Eq. 16: c = 0.5037 (gate 3%; PEV χ = 22: 0.5007). Physical
  block entropies S(1) = 0.4742, S(2) = 0.5937 vs exact 0.4739, 0.5934.
- Geometry: graph geodesic ≈ 6 log₃ r bonds (2 layers of 3 bonds; one more
  layer costs exactly 6); the bond-count minimal cut of a 3^k-site block crosses
  4k − 1 bonds; the minimal-cut entropy increment per tripling is
  (c_MERA/3) ln 3 = 0.1845 vs exact (1/6) ln 3 = 0.1831.
- MI geometry on the critical Kitaev chain: d = 2.06 ln r + const (4Δ_ψ = 2),
  hyperbolic stress 0.060 vs flat 0.108, fitted curvature radius 1.4; at
  ξ ≈ 3 the radius exceeds 10³ and the linear law wins.
- HaPPY: single pentagon = [[5,1,3]] facts incl. X̄ ∼ −ZXZII; layer counts
  (5,0),(10,5),(25,15), boundaries 20/55/145; hexagon state obeys RT on every
  connected region (Theorem 2); pentagon code with product bulk: RT exact on
  89.5% (1 layer) of connected regions, exactly when the greedy geodesics from
  A and Aᶜ match, Theorem 3 bound never violated; greedy wedge ⇔ exact
  reconstructibility of the central logical algebra on all connected regions
  tested (1 and 2 layers); centre reconstructible from ≥ 10 of 20 qubits
  (best case), guaranteed from 13 (Eq. 5.1 asymptotics 0.524 / 0.724).
- Toy Jacobson: exact first-law residual O(ε²) (ratio ≈ 4 per halving), ≪ |δS|.

## API (house style)

```
# --- MERA (critical Ising) ---
mera_ising(chi=8, cache='auto') -> TernaryMERA     # cached optimized tensors
energy_density(mera) -> float                        # per site; compare to -4/pi
scaling_dimensions(mera, n, which='two-site')        # Delta = -log_3 |lambda|
scaling_dimensions_by_parity(mera, n)                # {'even', 'odd'}: sigma odd, epsilon even
central_charge_estimate(mera) -> (c, S2, S1)         # PEV Eq. 16
minimal_cut(mera, region, weights='count'|'lnchi') -> (bound_nats, cut_edges)  # max-flow
minimal_cut_entropy(mera, region) -> (S_est, n_bonds, bound)
mera_geodesic(mera, i, j) -> int                     # graph distance in bonds

# --- mutual-information geometry (any Gaussian/fermionic state) ---
mi_matrix(V_or_C_or_Gamma, regions, kind) -> I       # pairwise MI, nats
mi_distance(I, floor=1e-12) -> (D, info)             # d_ij = −ln(I_ij / I_max)
reconstruct_metric(D, dim, method='mds'|'hyperbolic') -> (X, info)  # stress, distortion, radius
hyperbolicity(D) -> (delta, delta/diam)              # Gromov four-point delta
geometry_report(I, positions) -> dict                # the failure-map row
failure_map(model_family, param_grid) -> rows        # the honest experiment, tabulated

# --- HaPPY code (stabilizer formalism) ---
happy_code(n_layers, tiling='pentagon'|'hexagon') -> HaPPYCode
rt_entropy(code, region) -> RTResult(S, cut, greedy_A, greedy_Ac, residual_empty, rt_holds)
greedy_wedge(code, region) -> MinCut(size, legs, wedge)
bulk_reconstruction(code, (tensor, 'X'|'Y'|'Z'), region) -> (pauli_string, sign) | None
reconstructible(code, tensor, region, pauli)        # exact GF(2) test

# --- toy Jacobson (leap L2) ---
entanglement_equilibrium_residual(V, K, balls, mode='exact'|'bw') -> r
constrain_dynamics(K_family, balls, perturb) -> (K_star, table)
equilibrium_residual(K, sizes=|balls=, ratios=, parity='odd'|'even'|'any',
                     beta_at='site'|'bond'|'mix', shift=, wall=, zero_mode='parity'|'project')
bw_weights(ball, shift=0.0, x=None, wall=None)      # wall: attached -> (B^2-y^2)/(2B)
k4_sensitivity(N, sizes, ratios, c2_list, c3_list, **kw) -> l3/l2 vs 6, C
```

Distances derived from mutual information inherit the nats convention; report embedding
quality (stress, δ-hyperbolicity) alongside every reconstructed metric — a picture
without its distortion number is inadmissible.
