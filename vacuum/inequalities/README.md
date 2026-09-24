# vacuum.inequalities — the QEI stress-tester and sharp-constant search (Layer 3)

Status: **built and anchored** (`docs/PLAN_LAYERS_2_3.md`, M3.4–M3.5). This README is
the original build brief; the shipped surface is `vacuum/inequalities/__init__.py`'s
`__all__`, and the binding signatures are the plan document's — the sketch below
predates the central design decision that the whole minimization collapses to one
Williamson problem on the pulled-back sampling operator `O_f`, so
`optimize_state`/`sharp_constant_scan` are answered by `qei_minimize` +
`embed_extremal` (the optimum is *solved*, not searched). Runs on the `vacuum.core`
Gaussian stack (ħ = 1, ν ≥ 1/2, block ordering, nats).

Anchors in force: the 2d massless sharp constant recovered at ratio 0.999235
extrapolated (0.077 % from Flanagan's 1/(6π), spec 1 %), with the extremal states
exhibited and archived; every engineered squeezed pocket satisfies the bound and the
ν < 1/2 canary is caught on raw margins; dense ↔ Gaussian work statistics agree to
7.7e−13 (spec 1e−8) with Jarzynski defects 3.4e−12 dense (spec 1e−10) and 3.2e−15
Gaussian. Tests: `tests/test_qei.py`, `tests/test_work_stats.py`.

**Build.** Engineer Gaussian states with negative-energy pockets — squeezed, multimode,
spatially structured — and drive optimization against the Ford–Roman bound: minimize the
ratio of achieved negative averaged energy to the bound, over states *and* over smooth
sampling functions. For Gaussian states this is a tractable quadratic-form problem, and
the sharp-constant question reduces to eigenvalue problems for nonlocal operators. The
sharp constant is known in two dimensions (Fewster–Eveson / Flanagan lineage) and open in
four; numerically converged conjectures for the 4D constants, with the optimizing states
exhibited, would be a direct contribution to the Fewster program. Attached here: the
work-statistics module — two-point-measurement energy distributions for quenched and
driven protocols, Jarzynski and Crooks as further standing audits, full counting
statistics so every protocol in the repository carries its complete energy ledger, not
just its mean.

**Validation anchors.** The 2D sharp constant recovered by the optimizer (known value,
Fewster–Eveson 1998); Ford–Roman (1995, 1997) bound satisfied by every engineered state
(a QEI *violation* is an audit event, not a result); Jarzynski equality
⟨e^{−βW}⟩ = e^{−βΔF} and Crooks symmetry on quenched Gaussian protocols; lattice →
continuum extrapolation of the achieved ratio (dispersion and finite-size checks).

**Discovery targets.** Sharp-constant conjectures in cases with no published tight bound
— above all 4D — with the extremal states and sampling functions exhibited (this is
leap L1 in PLAN.md). Structure of the optimizing states: mode content, squeezing
profile, spatial support.

**Pivot trigger.** If the sharp-constant search stalls in optimization plateaus,
redirect the effort to the interacting-field leap (PLAN.md Layer 7, L5), where even
crude numbers are new.

## Planned API (house style)

```
# --- averaged energy and bounds ---
energy_density_op(K, site) -> (2N,2N) H_loc      # quadratic form for the local energy density
smeared_energy(V, K, f) -> float                 # ∫ dt f(t)² ⟨:T_00:⟩ along a static worldline,
                                                 #   normal-ordered against V_vac = I/2
ford_roman_bound(f, dim) -> float                # −C_dim ∫ (f′)²-type bound for sampling function f
qei_ratio(V, K, f, dim) -> float                 # achieved / bound ∈ [0, 1]; sharp constant = sup

# --- state and sampling-function optimization ---
squeezed_pocket(K, modes, r, phi) -> V           # engineered negative-energy Gaussian states
optimize_state(K, f, ansatz) -> (ratio, V_star)  # quadratic-form / eigenvalue problem in V
optimize_sampling(V, K, family) -> (ratio, f_star)
sharp_constant_scan(dim, N_grid) -> report       # joint optimization + continuum extrapolation;
                                                 #   emits extremal (V_star, f_star) for the record

# --- work statistics (two-point measurement) ---
work_distribution(V0, protocol, K) -> (w, P)     # TPM distribution for a Gaussian quench/drive
jarzynski_audit(w, P, beta, dF) -> float         # |⟨e^{−βw}⟩ − e^{−βΔF}|; standing audit
crooks_audit(P_fwd, P_rev, beta, dF) -> float
counting_statistics(protocol) -> cumulants       # full ledger beyond the mean
```

All optimizations run under the standing audits; states are validated physical
(min ν ≥ 1/2 − tol, with `vacuum.core.precision` escalation near the floor) before any
ratio is reported.
