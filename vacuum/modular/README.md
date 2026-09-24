# vacuum.modular — numerical modular Hamiltonians (Layer 4)

Status: **implemented** (Layer-4 instrument 1, the modular-Hamiltonian laboratory on
free-fermion chains) plus the Layer-1 bosonic first-law anchor. Bosonic side codes
against `vacuum.core` (Williamson-based `entanglement_hamiltonian`, ν ≥ 1/2, nats);
fermionic side against `vacuum.fermions` (Peschel correlation-matrix methods) with the
closed-form, arbitrary-precision correlation matrices of `vacuum.modular.lattice`.

**Build.** A modular-Hamiltonian laboratory on free-fermion chains via Peschel–Eisler
methods: exact entanglement Hamiltonians for a single interval, validated against the
Bisognano–Wichmann local form at criticality and against the Casini–Huerta exact
two-interval results — then pushed into configurations where no closed form exists,
tracking how nonlocal couplings in the modular Hamiltonian grow and decay. The bosonic
first-law machinery (δS = δ⟨K⟩) lives here too, built on
`core.entanglement_hamiltonian` and `core.modular_energy`.

**The numerical fact that shapes the module.** h = ln((1−C_A)/C_A) is dominated in real
space by the entanglement modes with occupations exponentially close to 0/1
(ε_max ≈ 1.7627 L at criticality, Eisler–Peschel 2017 Eq. (54)); float64 is honest only
up to ~18 sites. Every larger block is diagonalized in mpmath from a closed-form C
(`lattice.infinite_chain_C`, `ring_C`, `massive_chain_C`), and insufficient precision is
flagged with a `ModularPrecisionWarning` — never silently clamped.

**Validation anchors (tests/test_modular_lab.py, all permanent).**
1. Bisognano–Wichmann at criticality: the nearest-neighbour profile matches the exact
   lattice form of Eisler–Peschel 2017 Eq. (27) to 0.2/L (measured 6.9e-3, 2.6e-3, 1.5e-3
   at L = 20, 40, 60); the resummed profile (their Eq. (22), Eisler–Tonni–Peschel 2022
   Eq. (31)) is the conformal parabola 2πx(L−x)/L to 1.5/L²; β(x) → 2π·dist near the edge
   with the 1/L parabola correction; ε_max matches Eq. (54) to 0.05; the ring reproduces
   the chord (circle) form.
2. Casini–Huerta two intervals (CQG 26, 185005 (2009), Eqs. (33), (44)–(48)): after the
   lattice dictionary stated in `casini_huerta.py` (D1–D6), the local weight 2π/z′(x) and
   the nonlocal weight 2π/((x−x_c) z′(x_c)) between conjugate points are reproduced with
   deviations ∝ 1/L² (row sums 9e-2 → 1e-2 → 2e-3, smeared bilinears 2e-2 → 7e-3 → 2e-3
   at scales 1, 2, 4); the modular flow "teleports" a right-moving packet with the weight
   sin²θ(τ) of CH Eq. (59) to better than 8 %.
3. The fermionic entanglement first law δS = Tr(h δC_A) to 1e-8 by independent routes
   (measured ~1e-10).
4. Modular flow preserves the spectrum of C_A to 1e-12 (measured ~1e-14).
5. n ≥ 3 intervals at growing L with gap/L fixed: the summed bilocal weight approaches the
   n-interval CH weight as A/L² (A = 1.91 for three equal intervals with gap = L, measured
   at L = 12, 16, 24, 32) and extrapolates to +8e-05 — exact agreement in the continuum.
6. Gapped chains: the decay length of the inter-interval modular couplings is the
   correlation length itself, ξ_h/ξ = 0.9985 ± 0.0257 from the local decay rates at gaps of
   8–16 ξ (ξ = 5, 10, 20, 40).
7. `ctm.py` — the corner-transfer-matrix closed form for the gapped chain: within a few ξ of
   one edge of a long segment the entanglement Hamiltonian IS the half-infinite CTM operator
   2π s(m) Σ_x x h_x with s(m) = (2/π) k I(k′) and levels (2l+1)·π I(k′)/I(k),
   k = 1/√(1+m²) = sech(1/ξ) (matched to 1e-14; the module docstring states which parts are
   published and which are verified rather than derived). Past the middle of a finite
   segment the profile is the triangle 2π s min(x, L−x); for ξ ≳ L it crosses over to the
   critical 1 − 1/L.

**Discovery targets** (papers/multi-interval-modular-hamiltonians/): three and four
intervals, unequal intervals, and gapped (staggered-mass) chains — the range and decay of
the nonlocal couplings versus inter-interval distance and correlation length, and the
off-critical deformation of the BW profile. Feeds the toy-Jacobson leap (PLAN.md L2)
jointly with `vacuum.geometry`.

## API (house style)

```
# --- fermionic (exact, via vacuum.fermions + closed-form C) ---
interval_modular(C, sites, dps=None) -> h            # h = ln((1 − C_A)/C_A); dps=None float64 (flags clips), int = mpmath
two_interval_modular(C, A1, A2, dps=None) -> h       # h on A1 + A2
multi_interval_modular(C, parts, dps=None) -> h
bw_profile(h, hopping=0.5, k_F=None, mode='nn'|'resummed'|'row', offset=0) -> (x, beta)
bilocal_weight(h, sites, rows, cols) -> S            # alternating row sums (ETP-2022 Eq. 32) → CH nonlocal weight
chiral_bilinear(h, sites, g, f, rows, cols) -> complex
casini_huerta_exact(A1, A2, N=None) -> H_R           # CH kernel sampled on the lattice (right-mover gauge)
ch_beta / ch_nonlocal_weight / ch_conjugate_points / ch_mobius_conjugate / ch_bilinear / ch_mixing_angle / ...
nonlocal_decay(h, sites, parts) -> dict             # inter-block couplings vs distance
modular_flow(h, C_A, s) -> C_s                       # U C_A U†, U = e^{−ish}; flow_wavefunction, wavepacket
locality_score(M, bandwidth=1, block=1) -> float
infinite_chain_C / ring_C / massive_chain_C(sites, ..., dps=None)   # closed-form C at any precision

# --- bosonic (Gaussian, via vacuum.core) ---
region_modular(V, K, modes) -> G                     # escalates the ν-spectrum to mpmath near ν = 1/2 (reported in info)
first_law_check(K, modes, perturb, eps) -> dict      # δS vs ½ Tr(G δV_A), both in nats (unchanged API)
modular_energy_profile(G, V_A) -> per_mode

# --- shared ---
multi_interval_scan(config_grid) -> [report, ...]   # the push into no-closed-form territory
```

Convention note: modular/entanglement Hamiltonians are dimensionless generators
(ρ_A ∝ e^{−K}); entropies in nats; fermion chain hopping t = 1/2, k_F = π/2, v_F = 1;
site j sits at x = j + 1/2 and bonds at integers (dictionary D1). Symplectic eigenvalues
at the vacuum floor ν = 1/2 make the bosonic G singular — `region_modular` reports the
floor modes and their mpmath margins rather than clamping silently.
