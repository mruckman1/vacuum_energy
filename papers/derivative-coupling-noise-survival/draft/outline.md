# Outline — does derivative-coupling harvesting survive noise? (draft, model-level)

Status: outline only. Numbers below are copied from `data/summary.json` of the
producing build recorded in MANIFEST.md; nothing here is a device prediction.

1. **Question.** TB-MM24 (PRD 110, 105016) / arXiv:2505.01516: with derivative
   coupling, causally connected detectors harvest from field correlations, not
   communication — which would relax the strict spacelike-separation requirement of
   harvesting experiments. Does it survive waveguide temperature, detector loss and
   dephasing?
2. **Engine.** Oscillator detectors with x_d–π coupling appended to the lattice field
   as one Gaussian system (`attach_detectors(coupling='xp')`), evolved through the
   audited protocol runner; Gate A for the derivative coupling (exponent ≥ 2.005,
   R² ≥ 0.99998) ties it to the second-order derivative-UDW state.
3. **The split at light contact (vacuum).** 1+1 massless lattice, d = 12, cos² T = 4,
   Ω = 0.5: xp |M_vac|/|M| = 0.9994, |M_comm|/|M| = 0.0337, E_N = 6.3447e-04;
   amplitude coupling |M_comm|/|M| = 0.365 and no negativity. The ∂ₓΦ
   circuit-QED model behaves like the ∂ₜφ one (|M_vac|/|M| = 0.999).
4. **The noise map.** 48 points (T × κ × γ_φ). Survival region: 27
   points with E_N > 0, 22 above 1e-4 nats — this map's own *stated* level, not a
   derived floor: against the sibling candidate's floor derived from Ren 2022
   (1e-3 nats, band [7.82e-5, 0.04]) the whole grid is sub-floor (max E_N
   6.34e-4; 0 of 48 at 1e-3, 23 of 48 at 7.82e-5), which the claim of §5 does not
   depend on. Death lines: the clean line survives to T = 0.10 (top of the grid;
   second-order death between 0.10 and 0.12), dies between T = 0.06 and 0.08 with loss on;
   γ_φ between 1e-05 and 3e-05 (T = 0).
5. **The answer.** On every surviving point |M_vac|/|M| ∈ [0.9994, 0.9996]:
   the correlation origin is untouched by the noise that kills the negativity.
   Field temperature only *raises* the correlation share (T=0.0: 0.9994, T=0.02: 0.9994, T=0.04: 0.9995, T=0.06: 0.9995, T=0.08: 0.9995, T=0.1: 0.9996).
6. **Caveats / next.** Single configuration; perturbative split with a
   nonperturbative bridge column (|m|/|M_T| ∈ [0.830, 1.019] on
   surviving points); 1+1 only; device inputs via `experiments/` not yet consumed.
