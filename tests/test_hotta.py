"""Tests for vacuum.qet.hotta — M3.1, the minimal QET model (exact, dim 4).

The four spec anchors (docs/PLAN_LAYERS_2_3.md, M3.1):

1. Closed-form anchor: the coded protocol's E_A, E_B(theta), and the
   numerically-swept optimal E_B match the transcriptions of
   arXiv:1101.3954 Eqs. (8), (14), (9)-(11) to 1e-12 across an (h, k) grid.
2. Standing audit: hotta_sweep asserts E_A >= E_B everywhere
   (tolerance -1e-12).
3. Causality anchor: Bob's reduced state before receiving the classical bit
   is independent of whether Alice measured (trace distance < 1e-14) —
   no-signaling as a test.
4. Entanglement-as-fuel anchor: ground-state concurrence k/sqrt(h^2+k^2) is
   fully consumed by Alice's measurement (post-measurement branches are
   exactly product states), and cutting the classical channel zeroes the
   extractable <E_B> (max over theta drops to 0; positive extraction
   requires the bit).

Transcription sanity (the offsets, ground state, projectors, unitarity) is
checked first so a convention slip is findable before the anchors fire.
"""

import itertools

import numpy as np
import pytest

from vacuum.qet import (
    bob_reduced_state,
    bob_rotation,
    concurrence,
    e_a_closed,
    e_b_closed,
    e_b_optimal,
    ground_state,
    hamiltonian,
    hamiltonian_terms,
    hotta_sweep,
    measurement_projector,
    optimize_e_b,
    run_protocol,
    theta_optimal,
    trace_distance,
)

# the (h, k) grid for every anchor: strong/weak field vs strong/weak coupling
H_VALS = [0.05, 0.3, 1.0, 2.0, 5.0]
K_VALS = [0.1, 0.5, 1.0, 2.5, 4.0]
GRID = list(itertools.product(H_VALS, K_VALS))
THETAS = np.linspace(0.0, np.pi / 2.0, 13)


# ---------------------------------------------------------------------------
# transcription sanity
# ---------------------------------------------------------------------------


class TestTranscription:
    def test_ground_energy_is_zero_by_the_offsets(self):
        """Eq. (5)-(7) constant offsets fix min eig(H) = 0 exactly."""
        worst = 0.0
        for h, k in GRID:
            e0 = float(np.linalg.eigvalsh(hamiltonian(h, k))[0])
            worst = max(worst, abs(e0))
        assert worst < 1e-12, f"|E_ground| up to {worst:.3e} (offsets broken)"

    def test_closed_form_ground_state_matches_eigh(self):
        worst_overlap = 0.0
        worst_energy = 0.0
        for h, k in GRID:
            H = hamiltonian(h, k)
            g = ground_state(h, k)
            assert abs(np.linalg.norm(g) - 1.0) < 1e-14
            evals, evecs = np.linalg.eigh(H)
            worst_overlap = max(worst_overlap, 1.0 - abs(np.vdot(evecs[:, 0], g)))
            worst_energy = max(worst_energy, abs(np.real(np.vdot(g, H @ g))))
        assert worst_overlap < 1e-12, (
            f"closed-form |g> vs eigh: 1-|overlap| up to {worst_overlap:.3e}"
        )
        assert worst_energy < 1e-12, f"<g|H|g> up to {worst_energy:.3e}"

    def test_hamiltonian_terms_sum_and_are_hermitian(self):
        for h, k in [(1.0, 0.5), (0.3, 2.0)]:
            H_A, H_B, V = hamiltonian_terms(h, k)
            for M in (H_A, H_B, V):
                assert np.max(np.abs(M - M.conj().T)) < 1e-15
            assert np.max(np.abs(H_A + H_B + V - hamiltonian(h, k))) < 1e-15

    def test_projectors_complete_idempotent_hermitian(self):
        Pp, Pm = measurement_projector(+1), measurement_projector(-1)
        assert np.max(np.abs(Pp + Pm - np.eye(4))) < 1e-15
        for P in (Pp, Pm):
            assert np.max(np.abs(P @ P - P)) < 1e-15
            assert np.max(np.abs(P - P.conj().T)) < 1e-15
        assert np.max(np.abs(Pp @ Pm)) < 1e-15

    def test_bob_rotation_is_unitary_and_conditional(self):
        for mu in (+1, -1):
            for theta in THETAS:
                U = bob_rotation(mu, theta)
                assert np.max(np.abs(U @ U.conj().T - np.eye(4))) < 1e-14
        # theta = 0 is the identity; the two outcomes get different rotations
        assert np.max(np.abs(bob_rotation(+1, 0.0) - np.eye(4))) < 1e-15
        assert np.max(np.abs(bob_rotation(+1, 0.3) - bob_rotation(-1, 0.3))) > 0.1

    def test_bad_inputs_raise(self):
        with pytest.raises(ValueError):
            hamiltonian(0.0, 1.0)
        with pytest.raises(ValueError):
            ground_state(1.0, -0.5)
        with pytest.raises(ValueError):
            measurement_projector(0)
        with pytest.raises(ValueError):
            bob_rotation(2, 0.1)


# ---------------------------------------------------------------------------
# anchor 1: closed-form match to 1e-12 over the (h, k) grid
# ---------------------------------------------------------------------------


class TestClosedFormAnchor:
    def test_e_a_matches_eq8(self):
        """Protocol E_A vs E_A = h^2/sqrt(h^2+k^2) [Eq. (8)] to 1e-12."""
        worst = 0.0
        for h, k in GRID:
            got = run_protocol(h, k, 0.0)["E_A"]
            worst = max(worst, abs(got - e_a_closed(h, k)))
        assert worst < 1e-12, f"E_A vs Eq. (8): worst |diff| = {worst:.3e}"

    def test_e_b_of_theta_matches_eq14(self):
        """Protocol E_B(theta) vs Eq. (14) over a theta x (h, k) grid."""
        worst = 0.0
        for h, k in GRID:
            for theta in THETAS:
                got = run_protocol(h, k, theta)["E_B"]
                worst = max(worst, abs(got - e_b_closed(h, k, theta)))
        assert worst < 1e-12, f"E_B(theta) vs Eq. (14): worst |diff| = {worst:.3e}"

    def test_swept_optimum_matches_eq11_and_eqs9_10(self):
        """Numerically-swept optimal E_B vs Eq. (11) to 1e-12; the argmax
        angle vs theta* from Eqs. (9)-(10) (value-flat optimum: 1e-5)."""
        worst_val = 0.0
        worst_ang = 0.0
        for h, k in GRID:
            theta_num, e_b_num = optimize_e_b(h, k)
            worst_val = max(worst_val, abs(e_b_num - e_b_optimal(h, k)))
            worst_ang = max(worst_ang, abs(theta_num - theta_optimal(h, k)))
        assert worst_val < 1e-12, (
            f"swept optimal E_B vs Eq. (11): worst |diff| = {worst_val:.3e}"
        )
        assert worst_ang < 1e-5, (
            f"swept argmax vs Eqs. (9)-(10) theta*: worst |diff| = {worst_ang:.3e}"
        )

    def test_closed_forms_internally_consistent(self):
        """Eq. (14) evaluated at Eqs. (9)-(10) theta* reproduces Eq. (11)."""
        worst = 0.0
        for h, k in GRID:
            worst = max(
                worst,
                abs(e_b_closed(h, k, theta_optimal(h, k)) - e_b_optimal(h, k)),
            )
        assert worst < 1e-13, f"Eq.(14)@theta* vs Eq.(11): worst |diff| = {worst:.3e}"

    def test_energy_bookkeeping_closes(self):
        """E_meas = E_ground + E_A and E_final = E_meas - E_B, per run."""
        for h, k in [(1.0, 0.5), (0.3, 2.0), (5.0, 4.0)]:
            led = run_protocol(h, k, theta_optimal(h, k))
            assert abs(led["E_ground"]) < 1e-13
            assert abs(led["E_meas"] - led["E_ground"] - led["E_A"]) < 1e-13
            assert abs(led["E_final"] - led["E_meas"] + led["E_B"]) < 1e-13
            p_tot = sum(o["p"] for o in led["outcomes"].values())
            assert abs(p_tot - 1.0) < 1e-14
            # sigma_x outcomes are equiprobable on |g> (<g|sigma_x^A|g> = 0)
            assert abs(led["outcomes"][+1]["p"] - 0.5) < 1e-14


# ---------------------------------------------------------------------------
# anchor 2: standing audit E_A >= E_B everywhere
# ---------------------------------------------------------------------------


class TestHottaSweep:
    def test_sweep_passes_with_margin(self):
        res = hotta_sweep(H_VALS, K_VALS, tol=1e-12, strict=True)
        assert res.passed, f"hotta_sweep FAILED: {res}"
        assert res.worst_violation == 0.0
        min_margin = res.details["min_margin"]
        assert min_margin >= -1e-12, (
            f"E_A - E_B dipped to {min_margin:.3e} at {res.details['argmin']}"
        )
        assert len(res.details["rows"]) == len(GRID)
        # teleportation is real but never free: 0 < E_B < E_A on the grid
        for row in res.details["rows"]:
            assert row["E_B"] > 0.0, f"no extraction at (h,k)=({row['h']},{row['k']})"
            assert row["E_B"] < row["E_A"]

    def test_sweep_reports_measured_minimum_margin(self):
        res = hotta_sweep([1.0], [0.5], tol=1e-12)
        row = res.details["rows"][0]
        expected = e_a_closed(1.0, 0.5) - e_b_optimal(1.0, 0.5)
        assert abs(row["margin"] - expected) < 1e-12, (
            f"margin {row['margin']:.15f} vs closed-form {expected:.15f}"
        )


# ---------------------------------------------------------------------------
# anchor 3: no-signaling
# ---------------------------------------------------------------------------


class TestNoSignaling:
    def test_bob_cannot_see_alices_measurement(self):
        """Trace distance between Bob's reduced states with/without Alice's
        measurement, before the classical bit arrives: < 1e-14."""
        worst = 0.0
        for h, k in GRID:
            rho_no = bob_reduced_state(h, k, alice_measured=False)
            rho_yes = bob_reduced_state(h, k, alice_measured=True)
            for rho in (rho_no, rho_yes):
                assert abs(np.real(np.trace(rho)) - 1.0) < 1e-14
            worst = max(worst, trace_distance(rho_no, rho_yes))
        assert worst < 1e-14, f"no-signaling trace distance up to {worst:.3e}"

    def test_trace_distance_is_a_distance(self):
        rho = bob_reduced_state(1.0, 0.5, alice_measured=False)
        assert trace_distance(rho, rho) == 0.0
        sigma = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        d = trace_distance(rho, sigma)
        assert 0.0 < d <= 1.0 + 1e-15


# ---------------------------------------------------------------------------
# anchor 4: entanglement as fuel
# ---------------------------------------------------------------------------


class TestEntanglementFuel:
    def test_ground_state_concurrence(self):
        """C(|g>) = k/sqrt(h^2+k^2): for |g> = a|++> + b|--> the Wootters
        concurrence is 2|ab| = sqrt(1 - h^2/s^2) = k/s (a, b from the
        closed-form ground state)."""
        worst = 0.0
        for h, k in GRID:
            got = concurrence(ground_state(h, k))
            expected = k / np.hypot(h, k)
            worst = max(worst, abs(got - expected))
        assert worst < 1e-12, f"C(|g>) vs k/s: worst |diff| = {worst:.3e}"

    def test_measurement_consumes_all_entanglement(self):
        """Each post-measurement branch P_A(mu)|g>/sqrt(p) is exactly a
        product state (the coefficient matrix [[a, mu b], [mu a, b]] has
        determinant ab(1 - mu^2) = 0), so the fuel is fully spent."""
        worst_pure = 0.0
        worst_mixed = 0.0
        for h, k in GRID:
            led = run_protocol(h, k, 0.0)
            for mu in (+1, -1):
                worst_pure = max(worst_pure, concurrence(led["outcomes"][mu]["state"]))
            worst_mixed = max(worst_mixed, concurrence(led["rho_meas"]))
        assert worst_pure < 1e-13, (
            f"post-measurement branch concurrence up to {worst_pure:.3e}"
        )
        assert worst_mixed < 1e-13, (
            f"post-measurement ensemble concurrence up to {worst_mixed:.3e}"
        )
        # and the fuel was there to spend: C(|g>) > 0 on the whole grid
        assert min(concurrence(ground_state(h, k)) for h, k in GRID) > 0.01

    def test_cutting_the_channel_zeroes_extractable_energy(self):
        """With the classical channel cut (Bob gets an independent coin) the
        Eq. (14) cross term averages away: E_B_cut(theta) <= 0 for every
        theta, the best achievable is exactly 0 (at theta = 0, do nothing),
        while the open channel extracts E_B > 0 at theta*."""
        for h, k in [(1.0, 0.5), (0.3, 2.0), (2.0, 2.0)]:
            theta_star = theta_optimal(h, k)
            open_led = run_protocol(h, k, theta_star, channel_open=True)
            cut_led = run_protocol(h, k, theta_star, channel_open=False)
            assert open_led["E_B"] > 1e-4, (
                f"open channel failed to extract at (h,k)=({h},{k})"
            )
            assert cut_led["E_B"] < -1e-6, (
                f"cut channel at theta* should cost energy, got {cut_led['E_B']:.3e}"
            )
            # sweep theta: nothing positive survives the cut
            e_cut = [
                run_protocol(h, k, th, channel_open=False)["E_B"] for th in THETAS
            ]
            assert max(e_cut) <= 1e-14, (
                f"cut-channel <E_B> went positive: max = {max(e_cut):.3e}"
            )
            assert abs(run_protocol(h, k, 0.0, channel_open=False)["E_B"]) < 1e-14
            # numerical sweep agrees: the cut-channel optimum is zero
            _, e_b_cut_best = optimize_e_b(h, k, channel_open=False)
            assert abs(e_b_cut_best) < 1e-12, (
                f"cut-channel swept optimum {e_b_cut_best:.3e} != 0"
            )
