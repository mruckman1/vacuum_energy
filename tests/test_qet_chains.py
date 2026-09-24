"""Tests for vacuum.qet.chains and vacuum.qet.dmrg — M3.2, QET on chains.

Spec anchors (docs/PLAN_LAYERS_2_3.md, M3.2 and the test matrix):

1. ED protocol on the critical TFIM (L = 14, sparse eigsh), symmetric
   bond-split local energy density, ALWAYS difference form vs ground value;
   projective sigma^x measurement at A (Kraus + renormalize), conditional
   rotation at B optimized numerically per outcome.
2. Deliverable numbers recorded in assertion messages: the energy-dip
   profile in space and time, its decay with |A - B| critical vs
   off-critical, and the exchange-rate curve E_B_max/E_A vs distance.
3. No-signaling on chains: Bob's reduced state pre-communication is
   measurement-independent (trace distance < 1e-14, ED).
4. DMRG (TeNPy, `.[tensor]` extra, pytest.importorskip): ED<->DMRG at
   L = 14 with dip agreement < 1e-8; chi-doubling moves the dip < 1%; one
   L = 64 run recording the dip decay with convergence noted.

   Two refinements beyond the letter of the spec. (a) Run both engines with
   the *exact* rotation angle (theta_method='closed') and the residual
   ED<->DMRG dip difference drops from ~5e-10 to ~1e-14: the 5e-10 was two
   independent bounded searches disagreeing at their xatol, not tensor-
   network error, and the distinction matters for what the gate certifies.
   (b) The L = 64 run is a chi = 4, 8, 16, 32, 64 ladder rather than a
   single doubling, so the "< 1%" gate is shown to have teeth — chi = 4 and
   chi = 8 FAIL it (the dip moves by 118% and 3.2% of its depth) and only
   chi >= 16 passes. A convergence gate that no reachable setting fails
   certifies nothing.

Geometry throughout the ED anchors: L = 14 open chain, Alice at A = 3,
Bob swept over B = 4..10 (distances 1..7); critical g = J = 1 versus
off-critical g = 1.5.
"""

import logging

import numpy as np
import pytest

from vacuum.qet import (
    chain_distance_sweep,
    energy_profile,
    evolve_dip_profiles,
    local_hamiltonian,
    measure_site_x,
    premeasurement_reduced_states,
    rotation_optimum_closed,
    run_chain_qet,
    tfim_ground_state,
    tfim_hamiltonian,
    tfim_local_energy_ops,
    trace_distance,
)

L14 = 14
A_SITE = 3
B_MAIN = 6           # the d = 3 ledger used for profiles and DMRG validation
B_LIST = range(4, 11)  # distances 1..7
TS = [0.0, 0.75, 1.5, 2.25]

# measured ED anchors (this build; regression guards, tolerances platform-safe)
E_GS_CRIT = -17.47100405473177
E_A_CRIT = 0.6844283819343615
E_B_D1_CRIT = 1.750640e-2
E_B_D3_CRIT = 1.265713e-4


def _fmt(arr, precision=6):
    return np.array2string(np.asarray(arr), precision=precision, max_line_width=200)


# ---------------------------------------------------------------------------
# module-scoped fixtures (cache the expensive builds)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def crit14():
    """Critical chain: Hamiltonian, bond-split densities, eigsh ground state."""
    H = tfim_hamiltonian(L14, 1.0)
    h_ops = tfim_local_energy_ops(L14, 1.0)
    E0, gs = tfim_ground_state(L14, 1.0)
    return {"L": L14, "g": 1.0, "H": H, "h_ops": h_ops, "E0": E0, "gs": gs}


@pytest.fixture(scope="module")
def ledger3(crit14):
    """The d = 3 protocol ledger (A = 3, B = 6) on the critical chain."""
    return run_chain_qet(
        L14, 1.0, A_SITE, B_MAIN,
        ground=(crit14["E0"], crit14["gs"]), h_ops=crit14["h_ops"],
    )


@pytest.fixture(scope="module")
def ledger3_closed(crit14):
    """Same ledger with the exact closed-form rotation angle instead of the
    bounded scalar search — removes the optimizer's xatol from the ED side
    of the ED<->DMRG comparison."""
    return run_chain_qet(
        L14, 1.0, A_SITE, B_MAIN,
        ground=(crit14["E0"], crit14["gs"]), h_ops=crit14["h_ops"],
        theta_method="closed",
    )


@pytest.fixture(scope="module")
def sweep_crit():
    return chain_distance_sweep(L14, 1.0, A_SITE, B_LIST)


@pytest.fixture(scope="module")
def sweep_off():
    return chain_distance_sweep(L14, 1.5, A_SITE, B_LIST)


@pytest.fixture(scope="module")
def time_profiles(ledger3):
    return evolve_dip_profiles(ledger3, TS)


# ---------------------------------------------------------------------------
# construction: bond split, ground state
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_local_energies_sum_to_hamiltonian(self, crit14):
        """Symmetric bond split: sum_i h_i = H exactly, each h_i Hermitian."""
        total = crit14["h_ops"][0]
        for h in crit14["h_ops"][1:]:
            total = total + h
        diff = (total - crit14["H"]).tocsr()
        worst = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
        assert worst < 1e-13, f"sum h_i vs H: max |diff| = {worst:.3e}"
        for h in crit14["h_ops"]:
            asym = (h - h.T).tocsr()
            worst_h = float(np.max(np.abs(asym.data))) if asym.nnz else 0.0
            assert worst_h < 1e-15, f"h_i not symmetric: {worst_h:.3e}"
        assert len(crit14["h_ops"]) == L14

    def test_ground_state_anchor(self, crit14):
        """eigsh ground state: Rayleigh quotient, profile sum, energy value."""
        E0, gs, H = crit14["E0"], crit14["gs"], crit14["H"]
        assert abs(np.linalg.norm(gs) - 1.0) < 1e-13
        rayleigh = float(np.real(np.vdot(gs, H @ gs)))
        assert abs(rayleigh - E0) < 1e-10, (
            f"<gs|H|gs> = {rayleigh:.15f} vs eigsh E0 = {E0:.15f}"
        )
        prof_sum = float(np.sum(energy_profile(gs, crit14["h_ops"])))
        assert abs(prof_sum - E0) < 1e-10, (
            f"sum <h_i> = {prof_sum:.15f} vs E0 = {E0:.15f}"
        )
        assert abs(E0 - E_GS_CRIT) < 1e-8, (
            f"critical L=14 ground energy E0 = {E0:.12f} "
            f"(anchor {E_GS_CRIT:.12f}); E0/L = {E0 / L14:.6f}"
        )


# ---------------------------------------------------------------------------
# Alice's measurement: Kraus + renormalize, local energy injection
# ---------------------------------------------------------------------------


class TestMeasurement:
    def test_kraus_probabilities_and_normalization(self, crit14):
        branches = measure_site_x(crit14["gs"], L14, A_SITE)
        p_tot = sum(b["p"] for b in branches.values())
        assert abs(p_tot - 1.0) < 1e-13, f"sum p = {p_tot!r}"
        for mu, b in branches.items():
            # <sigma_x> = 0 in the definite-parity ground state
            assert abs(b["p"] - 0.5) < 1e-10, f"p({mu:+d}) = {b['p']!r}"
            norm = np.linalg.norm(b["state"])
            assert abs(norm - 1.0) < 1e-13, f"branch mu={mu:+d} norm {norm!r}"

    def test_measurement_injects_only_local_field_energy(self, crit14, ledger3):
        """dip_meas is zero away from A; E_A = g <sigma^z_A>_GS exactly
        (the sigma^x projectors commute with every bond term)."""
        dip = ledger3["dip_meas"].copy()
        e_a = ledger3["E_A"]
        assert abs(dip[A_SITE] - e_a) < 1e-12
        dip[A_SITE] = 0.0
        worst = float(np.max(np.abs(dip)))
        assert worst < 1e-12, f"dip_meas leaks off A: max |dip| = {worst:.3e}"
        # commuting-measurement identity, computed independently
        from vacuum.qet.chains import _kron_chain, _SZ  # test-only reach-in
        sz_A = float(np.real(np.vdot(
            crit14["gs"], _kron_chain(L14, {A_SITE: _SZ}) @ crit14["gs"])))
        assert abs(e_a - 1.0 * sz_A) < 1e-12, (
            f"E_A = {e_a:.15f} vs g<sz_A>_GS = {sz_A:.15f}"
        )
        assert abs(e_a - E_A_CRIT) < 1e-8, (
            f"measured E_A = {e_a:.12f} (anchor {E_A_CRIT:.12f})"
        )


# ---------------------------------------------------------------------------
# the protocol ledger at d = 3
# ---------------------------------------------------------------------------


class TestProtocolLedger:
    def test_extraction_positive_and_bounded(self, ledger3):
        """0 < E_B < E_A: teleportation is real but never returns more than
        the measurement injected."""
        e_a, e_b = ledger3["E_A"], ledger3["E_B"]
        assert e_b > 0.0, f"no extraction: E_B = {e_b:.3e}"
        assert e_b < e_a, f"E_B = {e_b:.6e} >= E_A = {e_a:.6e}"
        assert abs(e_b - E_B_D3_CRIT) / E_B_D3_CRIT < 1e-5, (
            f"E_B(d=3, critical) = {e_b:.9e} (anchor {E_B_D3_CRIT:.6e}), "
            f"ratio E_B/E_A = {ledger3['ratio']:.6e}"
        )

    def test_ledger_consistency_and_dip_locality(self, ledger3):
        """Difference-form bookkeeping closes: sum(dip_meas - dip_final)
        = E_B; the rotation only moves <h_i> on B-1..B+1, whose dip sums
        to -E_B — and the full dip profile is recorded here."""
        delta = ledger3["dip_final"] - ledger3["dip_meas"]
        e_b = ledger3["E_B"]
        assert abs(-float(np.sum(delta)) - e_b) < 1e-12, (
            f"sum(dip_meas - dip_final) = {-np.sum(delta):.3e} vs E_B = {e_b:.3e}"
        )
        hood = [B_MAIN - 1, B_MAIN, B_MAIN + 1]
        hood_sum = float(np.sum(delta[hood]))
        assert abs(hood_sum + e_b) < 1e-12, (
            f"dip over B-neighborhood = {hood_sum:.3e} vs -E_B = {-e_b:.3e}"
        )
        outside = delta.copy()
        outside[hood] = 0.0
        worst = float(np.max(np.abs(outside)))
        assert worst < 1e-12, (
            f"rotation leaked off B-1..B+1: {worst:.3e}; "
            f"dip_final = {_fmt(ledger3['dip_final'])}"
        )

    def test_per_outcome_rotation_and_closed_form(self, ledger3, crit14):
        """Each outcome's numerically-swept optimum matches the closed
        trigonometric form (dE to 1e-12, theta to 1e-6), and extracts >= 0."""
        h_loc = local_hamiltonian(crit14["h_ops"], B_MAIN)
        for mu, o in ledger3["outcomes"].items():
            assert o["dE"] >= 0.0, f"outcome {mu:+d} extracted {o['dE']:.3e} < 0"
            th_c, de_c = rotation_optimum_closed(
                o["state_meas"], L14, B_MAIN, h_loc, mu)
            assert abs(o["dE"] - de_c) < 1e-12, (
                f"mu={mu:+d}: numeric dE = {o['dE']:.12e} vs closed {de_c:.12e}"
            )
            d_th = abs((o["theta"] - th_c + np.pi / 2) % np.pi - np.pi / 2)
            assert d_th < 1e-6, (
                f"mu={mu:+d}: theta numeric {o['theta']:.10f} vs closed {th_c:.10f}"
            )

    def test_theta_method_equivalence(self, ledger3, ledger3_closed):
        """theta_method='closed' reproduces the numerically-optimized ledger:
        E_B to 1e-12 (a stationary point, so xatol enters quadratically) and
        the dip profile to the optimizer's linear-in-xatol resolution."""
        d_eb = abs(ledger3_closed["E_B"] - ledger3["E_B"])
        d_dip = float(np.max(np.abs(
            ledger3_closed["dip_final"] - ledger3["dip_final"])))
        assert d_eb < 1e-12, (
            f"E_B closed {ledger3_closed['E_B']:.15e} vs numeric "
            f"{ledger3['E_B']:.15e} (|diff| = {d_eb:.3e})"
        )
        assert d_dip < 1e-8, f"dip_final differs by {d_dip:.3e}"


# ---------------------------------------------------------------------------
# no-signaling on chains (spec: < 1e-14, ED)
# ---------------------------------------------------------------------------


class TestNoSignaling:
    @pytest.mark.parametrize("g", [1.0, 1.5])
    def test_bob_cannot_see_alices_measurement(self, crit14, g):
        """Bob's reduced state (site B and the 3-site block around it)
        before the classical bit arrives is independent of whether Alice
        measured: trace distance < 1e-14."""
        if g == 1.0:
            gs = crit14["gs"]
        else:
            _, gs = tfim_ground_state(L14, g)
        worst = 0.0
        for sites in ([B_MAIN], [B_MAIN - 1, B_MAIN, B_MAIN + 1]):
            rho_no, rho_yes = premeasurement_reduced_states(gs, L14, A_SITE, sites)
            assert abs(np.real(np.trace(rho_no)) - 1.0) < 1e-13
            assert abs(np.real(np.trace(rho_yes)) - 1.0) < 1e-13
            worst = max(worst, trace_distance(rho_no, rho_yes))
        assert worst < 1e-14, (
            f"no-signaling violated at g={g}: trace distance {worst:.3e}"
        )


# ---------------------------------------------------------------------------
# deliverables: dip decay with |A - B|, exchange-rate curve
# ---------------------------------------------------------------------------


class TestDistanceSweep:
    def test_dip_decay_critical(self, sweep_crit):
        """E_B(d) > 0 and strictly decreasing over d = 1..7 at criticality;
        the decay curve is the recorded deliverable."""
        e_b = [r["E_B"] for r in sweep_crit["rows"]]
        curve = [(r["d"], f"{r['E_B']:.6e}") for r in sweep_crit["rows"]]
        assert all(x > 0.0 for x in e_b), f"non-positive E_B in {curve}"
        assert all(a > b for a, b in zip(e_b, e_b[1:])), (
            f"critical E_B(d) not strictly decreasing: {curve}"
        )
        assert abs(e_b[0] - E_B_D1_CRIT) / E_B_D1_CRIT < 1e-5, (
            f"E_B(d=1) = {e_b[0]:.9e} (anchor {E_B_D1_CRIT:.6e}); "
            f"critical decay curve (d, E_B) = {curve}"
        )

    def test_dip_decay_off_critical(self, sweep_off):
        e_b = [r["E_B"] for r in sweep_off["rows"]]
        curve = [(r["d"], f"{r['E_B']:.6e}") for r in sweep_off["rows"]]
        assert all(x > 0.0 for x in e_b), f"non-positive E_B in {curve}"
        assert all(a > b for a, b in zip(e_b, e_b[1:])), (
            f"off-critical (g=1.5) E_B(d) not strictly decreasing: {curve}"
        )

    def test_critical_decay_outlives_off_critical(self, sweep_crit, sweep_off):
        """Normalized to d = 1, the gapped chain's E_B(d) falls below the
        critical chain's at every d >= 2 — ground-state correlations (the
        QET fuel) decay exponentially off criticality, algebraically at it."""
        crit = [r["E_B"] for r in sweep_crit["rows"]]
        off = [r["E_B"] for r in sweep_off["rows"]]
        norm_c = [x / crit[0] for x in crit]
        norm_o = [x / off[0] for x in off]
        rows = [
            (d + 2, f"crit {norm_c[d + 1]:.3e}", f"off {norm_o[d + 1]:.3e}")
            for d in range(len(crit) - 1)
        ]
        for d in range(1, len(crit)):
            assert norm_o[d] < norm_c[d], (
                f"off-critical decay not faster at d={d + 1}: {rows}"
            )
        # and the gap grows with distance (factor > 10 by d = 5)
        assert norm_o[4] < 0.1 * norm_c[4], (
            f"normalized E_B(d=5): critical {norm_c[4]:.3e} vs "
            f"off-critical {norm_o[4]:.3e}; full comparison {rows}"
        )

    def test_exchange_rate_curve(self, sweep_crit, sweep_off):
        """The exchange rate E_B_max/E_A vs distance — the M3.2 discovery-
        target deliverable — strictly decreasing on both chains."""
        for name, sweep in (("critical", sweep_crit), ("off-critical", sweep_off)):
            ratio = [r["ratio"] for r in sweep["rows"]]
            curve = [(r["d"], f"{r['ratio']:.6e}") for r in sweep["rows"]]
            assert all(a > b for a, b in zip(ratio, ratio[1:])), (
                f"{name} exchange rate not strictly decreasing: {curve}"
            )
            assert ratio[0] < 1.0, f"{name} exchange rate exceeds 1: {curve}"
        crit_curve = [(r["d"], f"{r['ratio']:.6e}") for r in sweep_crit["rows"]]
        assert sweep_crit["rows"][0]["ratio"] > 1e-2, (
            f"critical exchange-rate curve (d, E_B_max/E_A) = {crit_curve}, "
            f"E_A = {sweep_crit['E_A']:.9f}"
        )


# ---------------------------------------------------------------------------
# deliverable: the dip profile in space and time
# ---------------------------------------------------------------------------


class TestTimeProfile:
    def test_energy_conservation_under_free_evolution(self, ledger3, time_profiles):
        """sum_i dip_i(t) = E_A - E_B at every recorded time (free evolution
        conserves <H>)."""
        expected = ledger3["E_A"] - ledger3["E_B"]
        sums = time_profiles.sum(axis=1)
        worst = float(np.max(np.abs(sums - expected)))
        assert worst < 1e-9, (
            f"free evolution broke energy conservation by {worst:.3e}; "
            f"sums(t) = {_fmt(sums, 12)} vs E_A - E_B = {expected:.12f}"
        )

    def test_dip_localized_then_spreads(self, ledger3, time_profiles):
        """t = 0: dip supported on A and B-1..B+1 only; later times: the
        disturbance propagates into the rest of the chain (recorded)."""
        p0 = time_profiles[0].copy()
        support = [A_SITE, B_MAIN - 1, B_MAIN, B_MAIN + 1]
        p0[support] = 0.0
        worst0 = float(np.max(np.abs(p0)))
        assert worst0 < 1e-12, f"t=0 dip leaks off A,B: {worst0:.3e}"
        outside_sites = [0, 1] + list(range(B_MAIN + 3, L14))
        spread = float(np.max(np.abs(time_profiles[2][outside_sites])))
        assert spread > 1e-3, (
            f"dip failed to propagate by t={TS[2]}: max outside = {spread:.3e}"
        )
        # far end of the chain still quiet at t = 0.75 (Lieb-Robinson tail)
        far = float(abs(time_profiles[1][L14 - 1]))
        assert far < 1e-6, (
            f"site {L14 - 1} already disturbed at t={TS[1]}: {far:.3e}"
        )
        # record the space-time deliverable
        msg = "; ".join(f"t={t}: {_fmt(time_profiles[j], 4)}"
                        for j, t in enumerate(TS))
        assert time_profiles.shape == (len(TS), L14), msg


# ---------------------------------------------------------------------------
# pure-functional discipline (Layer-6 JAX-mirror pre-positioning)
# ---------------------------------------------------------------------------


class TestPurity:
    def test_no_input_mutation(self, crit14):
        gs_before = crit14["gs"].copy()
        measure_site_x(crit14["gs"], L14, A_SITE)
        run_chain_qet(L14, 1.0, A_SITE, B_MAIN,
                      ground=(crit14["E0"], crit14["gs"]),
                      h_ops=crit14["h_ops"])
        assert np.array_equal(crit14["gs"], gs_before), "ground state mutated"


# ---------------------------------------------------------------------------
# DMRG rung (TeNPy behind the .[tensor] extra)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qdmrg():
    pytest.importorskip("tenpy")
    logging.getLogger("tenpy").setLevel(logging.WARNING)
    from vacuum.qet import dmrg
    return dmrg


@pytest.fixture(scope="module")
def mps14(qdmrg, crit14):
    """L = 14 DMRG at chi_max = 150 (exact manifold), driven to machine
    convergence, plus the MPS protocol ledger for the ED validation."""
    ground = qdmrg.tfim_ground_mps(L14, 1.0, chi_max=150, max_sweeps=30,
                                   max_E_err=1e-14)
    ledger = qdmrg.run_chain_qet_mps(L14, 1.0, A_SITE, B_MAIN, ground=ground)
    return {"ground": ground, "ledger": ledger}


CHI_LADDER = (4, 8, 16, 32, 64)


@pytest.fixture(scope="module")
def mps64(qdmrg):
    """The one L = 64 run, as a chi-doubling ladder.

    chi = 4 cold-started, every later rung warm-started from the previous
    (so each rung is the same state refined, and the differences are
    bond-dimension effects rather than DMRG landing in different minima).
    Each rung gets the d = 3 protocol; the converged top rung also gets
    d = 1 and d = 5 for the dip-decay deliverable. The ladder exists so
    the "chi-doubling moves the dip < 1%" gate can be shown to have teeth:
    the low rungs must FAIL it and the high ones PASS.
    """
    L, g, A = 64, 1.0, 28
    grounds, leds, psi = {}, {}, None
    for chi in CHI_LADDER:
        grounds[chi] = qdmrg.tfim_ground_mps(
            L, g, chi_max=chi, max_sweeps=12, max_E_err=1e-12,
            mixer=(chi <= 8), lanczos_tight=False, psi0=psi)
        psi = grounds[chi][1]
        leds[chi] = qdmrg.run_chain_qet_mps(L, g, A, A + 3,
                                            ground=grounds[chi],
                                            theta_method="closed")
    top = CHI_LADDER[-1]
    by_d = {d: qdmrg.run_chain_qet_mps(L, g, A, A + d, ground=grounds[top],
                                       theta_method="closed")
            for d in (1, 3, 5)}
    # dip movement per doubling, as a fraction of the converged dip depth
    depth = -float(np.min(leds[top]["dip_final"]))
    shifts = {
        (lo, hi): float(np.max(np.abs(
            leds[hi]["dip_final"] - leds[lo]["dip_final"]))) / depth
        for lo, hi in zip(CHI_LADDER, CHI_LADDER[1:])
    }
    rel_eb = {
        (lo, hi): abs(leds[hi]["E_B"] - leds[lo]["E_B"]) / leds[hi]["E_B"]
        for lo, hi in zip(CHI_LADDER, CHI_LADDER[1:])
    }
    return {"L": L, "A": A, "grounds": grounds, "leds": leds, "by_d": by_d,
            "shifts": shifts, "rel_eb": rel_eb, "depth": depth, "top": top}


class TestDMRGValidatesAgainstED:
    def test_ground_state_agreement(self, crit14, mps14):
        E0_mps, info = mps14["ground"][0], mps14["ground"][2]
        d_e0 = abs(E0_mps - crit14["E0"])
        assert d_e0 < 1e-10, (
            f"L=14 ground energy: DMRG {E0_mps:.15f} vs ED {crit14['E0']:.15f} "
            f"(|diff| = {d_e0:.3e}; convergence {info})"
        )
        prof_mps = mps14["ledger"]["profile_gs"]
        prof_ed = energy_profile(crit14["gs"], crit14["h_ops"])
        worst = float(np.max(np.abs(prof_mps - prof_ed)))
        assert worst < 1e-10, f"ground <h_i> profiles differ by {worst:.3e}"

    def test_dip_agreement_below_1e8(self, ledger3, mps14):
        """THE M3.2 validation gate: ED and DMRG dip profiles at L = 14
        agree to < 1e-8 (spec test matrix), E_A and E_B likewise."""
        led_mps = mps14["ledger"]
        d_meas = float(np.max(np.abs(led_mps["dip_meas"] - ledger3["dip_meas"])))
        d_final = float(np.max(np.abs(led_mps["dip_final"] - ledger3["dip_final"])))
        assert d_meas < 1e-10, f"post-measurement dip differs by {d_meas:.3e}"
        assert d_final < 1e-8, (
            f"ED<->DMRG dip agreement {d_final:.3e} (spec < 1e-8); "
            f"DMRG dip_final = {_fmt(led_mps['dip_final'])}"
        )
        d_ea = abs(led_mps["E_A"] - ledger3["E_A"])
        d_eb = abs(led_mps["E_B"] - ledger3["E_B"])
        assert d_ea < 1e-10, f"E_A: DMRG vs ED |diff| = {d_ea:.3e}"
        assert d_eb < 1e-10, (
            f"E_B: DMRG {led_mps['E_B']:.12e} vs ED {ledger3['E_B']:.12e} "
            f"(|diff| = {d_eb:.3e})"
        )

    def test_mps_rotation_closed_form_matches_numeric(self, qdmrg, mps14):
        """The MPS branch optimizer is cross-validated by the closed
        trigonometric form, exactly as the ED one is."""
        psi_gs = mps14["ground"][1]
        branches = qdmrg.mps_measure_site_x(psi_gs, A_SITE)
        for mu, br in branches.items():
            th_n, de_n = qdmrg.mps_optimize_rotation(
                br["psi"], B_MAIN, 1.0, mu=mu)
            th_c, de_c = qdmrg.mps_rotation_optimum_closed(
                br["psi"], B_MAIN, 1.0, mu=mu)
            assert abs(de_n - de_c) < 1e-12, (
                f"MPS mu={mu:+d}: numeric dE = {de_n:.12e} vs closed {de_c:.12e}"
            )
            d_th = abs((th_n - th_c + np.pi / 2) % np.pi - np.pi / 2)
            assert d_th < 1e-6, (
                f"MPS mu={mu:+d}: theta numeric {th_n:.10f} vs closed {th_c:.10f}"
            )

    def test_dip_agreement_is_tensor_network_limited(self, qdmrg, mps14,
                                                     ledger3_closed):
        """With the exact rotation angle on both sides, the residual ED<->DMRG
        difference is the tensor network itself, not two optimizers' xatol —
        and it sits four orders below the spec's 1e-8 gate."""
        led_mps = qdmrg.run_chain_qet_mps(L14, 1.0, A_SITE, B_MAIN,
                                          ground=mps14["ground"],
                                          theta_method="closed")
        d_final = float(np.max(np.abs(
            led_mps["dip_final"] - ledger3_closed["dip_final"])))
        d_eb = abs(led_mps["E_B"] - ledger3_closed["E_B"])
        assert d_final < 1e-12, (
            f"ED<->DMRG dip agreement at a common exact angle: {d_final:.3e} "
            f"(spec gate 1e-8); E_B |diff| = {d_eb:.3e}"
        )
        assert d_eb < 1e-12, f"E_B |diff| = {d_eb:.3e}"

    def test_chi_doubling_at_L14(self, qdmrg, ledger3):
        """chi = 8 -> 16 at L = 14 (real truncation at chi = 8): the dip
        moves < 1% (spec test matrix)."""
        leds = {}
        for chi in (8, 16):
            gnd = qdmrg.tfim_ground_mps(L14, 1.0, chi_max=chi, max_sweeps=20)
            leds[chi] = qdmrg.run_chain_qet_mps(L14, 1.0, A_SITE, B_MAIN,
                                                ground=gnd)
        depth = -float(np.min(leds[16]["dip_final"]))
        dip_shift = float(np.max(np.abs(
            leds[16]["dip_final"] - leds[8]["dip_final"]))) / depth
        rel_eb = abs(leds[16]["E_B"] - leds[8]["E_B"]) / leds[16]["E_B"]
        assert dip_shift < 1e-2, (
            f"L=14 chi 8->16 moved the dip by {dip_shift:.3e} of its depth"
        )
        assert rel_eb < 1e-2, (
            f"L=14 chi 8->16 moved E_B by {rel_eb:.3e} relative "
            f"(E_B(chi=16) = {leds[16]['E_B']:.9e}, ED {ledger3['E_B']:.9e})"
        )


class TestDMRGLargeChain:
    def test_chi_doubling_gate_has_teeth(self, mps64):
        """The chi-doubling gate must REJECT under-converged bond dimensions,
        not just pass everywhere. On the L = 64 ladder chi = 4 -> 8 moves the
        d = 3 dip by more than its own depth; the gate is only satisfied once
        chi >= 16. Recorded so the admissible-chi claim is a measurement."""
        shifts, rel = mps64["shifts"], mps64["rel_eb"]
        table = "; ".join(
            f"chi {lo}->{hi}: dip {shifts[(lo, hi)]:.3e}, "
            f"E_B {rel[(lo, hi)]:.3e}"
            for lo, hi in zip(CHI_LADDER, CHI_LADDER[1:])
        )
        assert shifts[(4, 8)] > 1e-2, (
            f"chi=4 -> 8 should visibly move the dip (the gate would be "
            f"vacuous otherwise): {table}"
        )
        assert shifts[(8, 16)] > 1e-2, (
            f"chi=8 -> 16 should still fail the 1% gate at L=64: {table}"
        )

    def test_chi_doubling_moves_dip_below_one_percent(self, mps64):
        """L = 64: once converged (chi 16 -> 32 and 32 -> 64), E_B and the
        whole dip profile move < 1% (spec test matrix)."""
        shifts, rel = mps64["shifts"], mps64["rel_eb"]
        for lo, hi in ((16, 32), (32, 64)):
            assert shifts[(lo, hi)] < 1e-2, (
                f"L=64 chi {lo}->{hi} moved the dip by {shifts[(lo, hi)]:.3e} "
                f"of its depth {mps64['depth']:.3e}"
            )
            assert rel[(lo, hi)] < 1e-2, (
                f"L=64 chi {lo}->{hi} moved E_B(d=3) by {rel[(lo, hi)]:.3e} "
                f"relative (E_B = {mps64['leds'][hi]['E_B']:.9e})"
            )

    def test_dip_decay_recorded_with_convergence(self, mps64):
        """The L = 64 deliverable: dip decay over d = 1, 3, 5 recorded at the
        converged bond dimension, DMRG convergence noted (sweeps, per-sweep
        dE, truncation error)."""
        by_d = mps64["by_d"]
        top = mps64["top"]
        e_b = {d: by_d[d]["E_B"] for d in (1, 3, 5)}
        assert e_b[1] > e_b[3] > e_b[5] > 0.0, f"L=64 decay broken: {e_b}"
        for d in (1, 3, 5):
            assert 0.0 < e_b[d] < by_d[d]["E_A"], (
                f"L=64 d={d}: E_B = {e_b[d]:.6e} outside (0, E_A)"
            )
        infos = {chi: mps64["grounds"][chi][2] for chi in (16, 32, top)}
        for chi in (16, 32, top):
            assert infos[chi]["converged"], (
                f"L=64 DMRG not converged at chi={chi}: {infos[chi]}"
            )
        assert infos[top]["max_trunc_err"] < 1e-8, (
            f"L=64 decay (d, E_B) = {[(d, f'{e_b[d]:.6e}') for d in (1, 3, 5)]}, "
            f"ratios = {[(d, format(by_d[d]['ratio'], '.6e')) for d in (1, 3, 5)]}, "
            f"E_A = {by_d[3]['E_A']:.9f}, E_gs = {by_d[3]['E_gs']:.9f}; "
            f"convergence {infos}"
        )

    def test_measurement_locality_survives_truncation(self, mps64):
        """dip_meas at L = 64 is still supported on A alone, at truncation-
        noise level away from it."""
        led = mps64["by_d"][3]
        A = mps64["A"]
        dip = led["dip_meas"].copy()
        e_a = led["E_A"]
        assert abs(dip[A] - e_a) < 1e-8
        dip[A] = 0.0
        worst = float(np.max(np.abs(dip)))
        assert worst < 1e-8, (
            f"L=64 dip_meas leaks off A at {worst:.3e}; E_A = {e_a:.9f}"
        )
