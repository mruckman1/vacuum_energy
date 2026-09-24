"""Layer 7, L3 — the resource theory of vacuum manipulation (vacuum.qet.resource).

Numerical theorems, each with its tolerance stated in the assertion:

  * the free class is energy-non-signalling (defect at roundoff) and the
    non-free class is not;
  * W_->(|g>) equals Hotta's E_B^max (arXiv:1101.3954 Eq. (11)) to 1e-12 on
    the (h, k) grid, and the single-site chain potential equals the M3.2
    rotation protocol's E_B;
  * Theorem 1 (ledger monotonicity of W_->) and the negativity monotone hold
    over thousands of random free operations on random states with the worst
    violation at roundoff, while non-free operations raise W_-> by O(1) — the
    test has teeth;
  * the QET surplus is not an LOCC monotone (closed-form counterexample);
  * the smooth entropies: dense water-filling vs SDP, primal/dual brackets,
    classical closed forms, pure-state duality;
  * the one-shot bound is saturated by the minimal model (ratio 1 to 1e-12)
    and holds — for every smoothing epsilon — on the chain protocol;
  * the Gaussian mirror: closed form vs conditional-state integral, and
    Theorem 1 under Gaussian free operations.

cvxpy-gated tests use ``pytest.importorskip``; everything else is core-only.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from vacuum.core import harmonic_chain_K
from vacuum.qet import chains, hotta, slp
from vacuum.qet import resource as R

H_VALS = [0.05, 0.3, 1.0, 2.0, 5.0]
K_VALS = [0.1, 0.5, 1.0, 2.5, 4.0]
GRID = list(itertools.product(H_VALS, K_VALS))


def _ground_rho(h, k):
    g = hotta.ground_state(h, k)
    return hotta.hamiltonian(h, k), np.outer(g, g.conj())


def _tfim_dense(L, g):
    return np.asarray(chains.tfim_hamiltonian(L, g).toarray(), dtype=complex)


# --------------------------------------------------------------------------
# definitions: bipartition, bond operators, the free class
# --------------------------------------------------------------------------


def test_split_reconstructs_and_bond_is_sigma_x():
    H, _ = _ground_rho(1.0, 0.5)
    H_A, H_loc = R.split_hamiltonian(H, (0,))
    assert np.max(np.abs(H_A + H_loc - H)) < 1e-14
    # H_A is Alice's own term (h sigma_z (x) I) up to a constant
    core = H_A - np.trace(H_A) / 4.0 * np.eye(4)
    assert np.max(np.abs(core - np.kron(np.diag([1.0, -1.0]), np.eye(2)))) < 1e-14
    H_B, Xs, Ys = R.bond_operators(H, (0,))
    assert len(Xs) == 1
    dev = min(np.max(np.abs(Xs[0] - s * R._SX / np.sqrt(2.0))) for s in (1.0, -1.0))
    assert dev < 1e-13, f"bond operator is not sigma_x/sqrt2: dev {dev:.3e}"
    rec = np.kron(np.eye(2), H_B) + sum(np.kron(X, Y) for X, Y in zip(Xs, Ys))
    assert np.max(np.abs(rec - H_loc)) < 1e-13


def test_free_projectors_are_hottas_measurement():
    H, _ = _ground_rho(0.7, 1.3)
    P, xs = R.free_projectors(H, (0,))
    assert len(P) == 2 and xs.shape == (2, 1)
    want = [(np.eye(2) + mu * R._SX) / 2.0 for mu in (+1, -1)]
    for Pi in P:
        assert min(np.max(np.abs(Pi - w)) for w in want) < 1e-12


def test_energy_non_signalling_separates_free_from_non_free():
    rng = np.random.default_rng(0)
    for n, region in [(2, (0,)), (4, (0,)), (4, (1, 2)), (5, (2,))]:
        H = _ground_rho(1.0, 0.5)[0] if n == 2 else _tfim_dense(n, 1.0)
        P, _ = R.free_projectors(H, region)
        worst_free, best_nonfree = 0.0, np.inf
        for _ in range(20):
            K = R.random_free_instrument(P, int(rng.integers(1, 4)), rng)
            worst_free = max(worst_free, R.energy_signalling_defect(H, K, region))
            K2 = R.random_instrument(2 ** len(region), int(rng.integers(1, 3)), rng)
            best_nonfree = min(best_nonfree, R.energy_signalling_defect(H, K2, region))
        assert worst_free < 1e-12, f"n={n} {region}: free defect {worst_free:.3e}"
        assert best_nonfree > 1e-2, f"n={n} {region}: a random instrument looked free"
    # Hotta's projectors are free; a rotation about y on A is not
    H, _ = _ground_rho(1.0, 0.5)
    Ps = [(np.eye(2) + mu * R._SX) / 2.0 for mu in (+1, -1)]
    assert R.energy_signalling_defect(H, Ps, (0,)) < 1e-15
    Ry = np.cos(0.3) * np.eye(2) - 1j * np.sin(0.3) * R._SY
    assert R.energy_signalling_defect(H, [Ry], (0,)) > 0.1


# --------------------------------------------------------------------------
# monotone 1 on the validated protocols
# --------------------------------------------------------------------------


def test_qet_potential_is_hottas_e_b_max_on_the_grid():
    """W_->(|g>) = E_B^max, arXiv:1101.3954 Eq. (11), to 1e-12 on the (h, k) grid."""
    worst = 0.0
    for h, k in GRID:
        H, rho = _ground_rho(h, k)
        pot = R.qet_potential(H, rho, (0,))
        worst = max(worst, abs(pot["W"] - hotta.e_b_optimal(h, k)))
        # Bob's branches are pure product states; his reset = Hotta's rotation
        for b in pot["branches"]:
            assert abs(np.linalg.eigvalsh(b["sigma_B"])[-1] - 1.0) < 1e-12
    assert worst < 1e-12, f"W_-> vs Hotta Eq. (11): worst |diff| = {worst:.3e}"


def test_qet_potential_agrees_with_slp_locc_layer():
    """Same number as vacuum.qet.slp's LOCC layer (core-only Kraus route)."""
    H, rho = _ground_rho(1.0, 1.5)
    proj = [hotta.measurement_projector(mu) for mu in (+1, -1)]
    locc = slp.locc_extractable_energy(H, rho, 1, proj, method="kraus",
                                       kraus_kwargs={"restarts": 3})["locc"]
    W = R.qet_potential(H, rho, (0,))["W"]
    assert abs(W - locc) < 1e-8, f"W_-> {W:.12f} vs slp LOCC {locc:.12f}"


def test_single_site_potential_equals_chain_protocol():
    """Restricted Bob (one site) via the Kraus search = the M3.2 rotation's E_B."""
    L, g, A = 5, 1.0, 0
    H = _tfim_dense(L, g)
    _, psi = chains.tfim_ground_state(L, g)
    rho = np.outer(psi, psi.conj()).astype(complex)
    for B in (1, 2):
        W = R.qet_potential(H, rho, (A,), region_B=(B,), method="kraus",
                            kraus_kwargs={"restarts": 3})["W"]
        e_b = chains.run_chain_qet(L, g, A, B)["E_B"]
        assert abs(W - e_b) < 1e-7, f"B={B}: W_site {W:.10f} vs E_B {e_b:.10f}"


def test_the_protocol_is_one_free_sequence_and_e_b_is_spent():
    """Ground state: W_-> = E_B > 0; after Alice's free measurement + the bit the
    branches carry exactly W_-> as local ergotropy; after Bob's cash-out the
    branches are free (W_-> = 0) and the decrease equals E_B."""
    for h, k in [(1.0, 0.5), (0.3, 2.0), (2.0, 2.0)]:
        H, rho = _ground_rho(h, k)
        pot = R.qet_potential(H, rho, (0,))
        assert pot["W"] > 1e-4
        P, _ = R.free_projectors(H, (0,))
        branches = R.apply_instrument(rho, P, (0,))
        after_bit = sum(p * R.qet_potential(H, r, (0,))["W"] for p, r in branches)
        assert abs(after_bit - pot["W"]) < 1e-12, "the optimal free measurement loses nothing"
        spent = 0.0
        for (p, r), b in zip(branches, pot["branches"]):
            # Bob's optimal map: reset to the ground state of his effective Hamiltonian
            w, U = np.linalg.eigh(b["H_eff"])
            reset = [np.outer(U[:, 0], np.eye(2)[i]) for i in range(2)]
            wB = R.bob_energy_out(H, r, reset, (0,))
            out = slp.apply_local_channel(r, reset, (1,))
            assert R.qet_potential(H, out, (0,))["W"] < 1e-12, "post-cash-out branch not free"
            assert R.negativity(out, (0,)) < 1e-14
            spent += p * wB
        assert abs(spent - pot["W"]) < 1e-12, f"spent {spent:.12f} != W_-> {pot['W']:.12f}"
        assert abs(spent - hotta.e_b_optimal(h, k)) < 1e-12


# --------------------------------------------------------------------------
# Theorem 1: the numerical monotonicity proof (with teeth)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("H_fn,region,n_samples", [
    (lambda: _ground_rho(1.0, 0.5)[0], (0,), 2000),
    (lambda: _ground_rho(0.3, 2.0)[0], (1,), 600),
    (lambda: _tfim_dense(4, 1.0), (0,), 600),
    (lambda: _tfim_dense(4, 1.0), (1, 2), 300),
    (lambda: _tfim_dense(5, 0.7), (2,), 200),
])
def test_monotonicity_theorem(H_fn, region, n_samples):
    res = R.monotonicity_audit(H_fn(), region, n_samples=n_samples, seed=7, tol=1e-10)
    w = res.details["worst"]
    assert res.passed, res
    assert res.worst_violation < 1e-12, (
        f"worst monotonicity violation {res.worst_violation:.3e} over {n_samples} samples: {w}"
    )
    assert res.details["max_signalling_defect"] < 1e-12
    teeth = res.details["teeth"]
    assert teeth["nonfree_instrument"] > 0.05, f"non-free Alice ops never raised W: {teeth}"
    assert teeth["bob_injection"] > 0.05, f"energy injection never raised W: {teeth}"
    # the rise under injection never exceeds the injected energy (Eq. (5b) ledger)
    assert teeth["bob_injection_bound_defect"] < 1e-12, teeth


def test_negativity_is_half_concurrence_on_pure_two_qubit_states():
    rng = np.random.default_rng(3)
    for _ in range(50):
        rho = R.random_state(2, rng, "pure")
        v = np.linalg.eigh(rho)[1][:, -1]
        assert abs(R.negativity(rho, (0,)) - hotta.concurrence(v) / 2.0) < 1e-12
    for h, k in GRID[::4]:
        _, rho = _ground_rho(h, k)
        assert abs(R.negativity(rho, (0,)) - 0.5 * k / np.hypot(h, k)) < 1e-12


def test_surplus_is_not_an_locc_monotone():
    for h, k in [(1.0, 0.5), (0.4, 1.7)]:
        ce = R.surplus_counterexample(h, k)
        assert abs(ce["Q_before"]) < 1e-12
        assert ce["rise"] > 0.1
        assert abs(ce["rise"] - ce["rise_closed_form"]) < 1e-12, ce
        assert abs(ce["W_arrow_after"] - ce["W_arrow_before"]) < 1e-12, "W_-> must not move"
        assert abs(ce["W_loc_after"] - 2.0 * h) < 1e-12


# --------------------------------------------------------------------------
# smooth entropies
# --------------------------------------------------------------------------


def test_ky_fan_floor():
    h = np.diag([0.0, 1.0, 2.0, 3.0])
    assert R.ky_fan_floor(h, 0.0) == 0.0
    assert abs(R.ky_fan_floor(h, np.log(2.0)) - 0.5) < 1e-13
    assert abs(R.ky_fan_floor(h, np.log(3.0)) - 1.0) < 1e-13
    assert abs(R.ky_fan_floor(h, np.log(4.0)) - 1.5) < 1e-13
    assert abs(R.ky_fan_floor(h, 10.0) - 1.5) < 1e-13
    # fractional filling: e^s = 2.5 -> weights 0.4, 0.4, 0.2
    assert abs(R.ky_fan_floor(h, np.log(2.5)) - (0.4 * 0 + 0.4 * 1 + 0.2 * 2)) < 1e-13
    # it is the true LP minimum: brute-force over random feasible tau
    rng = np.random.default_rng(1)
    s = np.log(2.5)
    best = np.inf
    for _ in range(300):
        tau = R.random_state(2, rng, "mixed")
        w = np.linalg.eigvalsh(tau)
        if w[-1] <= np.exp(-s):
            best = min(best, float(np.real(np.trace(tau @ h))))
    assert best >= R.ky_fan_floor(h, s) - 1e-12


def test_smooth_min_entropy_dense_properties():
    rng = np.random.default_rng(5)
    for n in (1, 2, 3):
        rho = R.random_state(n, rng, "mixed")
        d = 2**n
        vals = [R.smooth_min_entropy(rho, e) for e in (0.0, 0.05, 0.1, 0.3, 0.6)]
        assert abs(vals[0] - R.min_entropy(rho)) < 1e-14
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), vals
        assert vals[-1] <= np.log(d) + 1e-12
        assert R.smooth_min_entropy(rho, 0.999) > np.log(d) - 1e-6
        # the optimizer is inside the ball and attains the value
        val, sig = R.smooth_min_entropy(rho, 0.2, return_state=True)
        w_r, U = np.linalg.eigh(rho)
        w_s = np.linalg.eigvalsh(sig)
        assert abs(-np.log(w_s[-1]) - val) < 1e-10
        sq = U @ np.diag(np.sqrt(np.clip(w_r, 0, None))) @ U.conj().T
        F = float(np.sum(np.sqrt(np.clip(np.linalg.eigvalsh(sq @ sig @ sq), 0, None)))) ** 2
        assert np.sqrt(max(0.0, 1.0 - F)) <= 0.2 + 1e-9


def test_classical_closed_forms_and_cq_formula():
    p = np.array([[0.4, 0.1], [0.2, 0.3]])
    hmin, hmax = R.classical_min_max_entropy(p)
    assert abs(hmin - (-np.log(0.4 + 0.3))) < 1e-14
    assert abs(hmax - np.log((np.sqrt(0.4) + np.sqrt(0.2)) ** 2 + (np.sqrt(0.1) + np.sqrt(0.3)) ** 2)) < 1e-14
    # Eq. (6.25) with diagonal conditional states reproduces the guessing form
    pb = p.sum(axis=0)
    states = [np.diag(p[:, b] / pb[b]) for b in range(2)]
    assert abs(R.conditional_min_entropy_cq(pb, states) - hmin) < 1e-14


def test_sdp_entropies_match_dense_and_closed_forms():
    pytest.importorskip("cvxpy")
    rng = np.random.default_rng(11)
    rho = R.random_state(2, rng, "mixed", rank=3)
    # unconditional smooth min-entropy: SDP vs water-filling
    for eps in (0.0, 0.1, 0.3):
        sdp = R.min_entropy_sdp(rho, (4, 1), eps=eps)["H_min"]
        dense = R.smooth_min_entropy(rho, eps)
        assert abs(sdp - dense) < 1e-6, f"eps={eps}: SDP {sdp:.9f} vs dense {dense:.9f}"
    # conditional, eps = 0: certified bracket
    res = R.min_entropy_sdp(rho, (2, 2))
    assert res["gap"] < 1e-7, res
    # classical-classical state: closed forms for both entropies
    p = np.array([[0.35, 0.05, 0.1], [0.2, 0.25, 0.05]])
    rho_cc = np.diag(p.reshape(-1)).astype(complex)
    hmin, hmax = R.classical_min_max_entropy(p)
    assert abs(R.min_entropy_sdp(rho_cc, (2, 3))["H_min"] - hmin) < 1e-7
    assert abs(R.max_entropy_sdp(rho_cc, (2, 3))["H_max"] - hmax) < 1e-6
    assert abs(R.max_entropy_direct(rho_cc, (2, 3)) - hmax) < 1e-5
    # pure-state duality: H_max(A|B)_psi = -H_min(A) = ln lambda_max(rho_A)
    psi = R.random_state(2, rng, "pure")
    rho_A = np.einsum("abcb->ac", psi.reshape(2, 2, 2, 2))
    assert abs(R.max_entropy_sdp(psi, (2, 2))["H_max"] - np.log(np.linalg.eigvalsh(rho_A)[-1])) < 1e-6
    # and the smooth duality Prop. 6.14 on a pure 3-qubit state, R = 2 qubits
    psi3 = R.random_state(3, rng, "pure")
    t = psi3.reshape(4, 2, 4, 2)
    rho_R = np.einsum("abcb->ac", t)
    for eps in (0.05, 0.2):
        assert abs(R.max_entropy_sdp(psi3, (4, 2), eps=eps)["H_max"]
                   + R.smooth_min_entropy(rho_R, eps)) < 1e-5


def test_sdp_entropies_need_the_extra(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cvxpy":
            raise ImportError("no cvxpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"vacuum\[sdp\]"):
        R.min_entropy_sdp(np.eye(4) / 4.0, (2, 2))


# --------------------------------------------------------------------------
# the one-shot bound
# --------------------------------------------------------------------------


def test_one_shot_bound_is_saturated_by_the_minimal_model():
    worst_abs, worst_rel = 0.0, 0.0
    for h, k in GRID:
        led = R.hotta_resource_ledger(h, k)
        assert np.all(led["H_min"] < 1e-12), "Bob's conditional states are pure"
        assert np.all(led["E_B"] <= led["bound_eps"] + 1e-12), "bound violated at some eps"
        assert led["E_B"] <= led["ceiling"] + 1e-12
        worst_abs = max(worst_abs, abs(led["E_B"] - led["bound_exact"]))
        worst_rel = max(worst_rel, abs(led["ratio_exact"] - 1.0))
        assert led["slp"] and led["W_loc_bound"] < 1e-12
        assert abs(led["E_B_protocol"] - led["E_B"]) < 1e-9
    # saturation: |E_B - bound| at roundoff in energy units (E_B is ~1e-4 at the
    # weak-field corner, so the ratio carries ~1e-11 of that same roundoff)
    assert worst_abs < 1e-12, f"|E_B - bound| up to {worst_abs:.3e}"
    assert worst_rel < 1e-9, f"E_B / bound deviates from 1 by {worst_rel:.3e}"


def test_one_shot_bound_holds_on_the_chain_and_is_not_saturated():
    led = R.chain_resource_ledger(6, 1.0, 0, [1, 2, 3, 4], method=None)
    ratios = []
    for row in led["rows"]:
        assert row["E_B"] <= row["bound_exact"] + 1e-12, row
        assert np.all(row["E_B"] <= row["bound_eps"] + 1e-12), row
        assert row["E_B"] <= row["bound_cq"] + 1e-12, row
        assert row["E_B"] <= row["ceiling_R"] + 1e-12
        assert row["E_B"] <= led["W_arrow_full"] + 1e-12
        assert row["bound_best"] <= row["bound_exact"] + 1e-12
        assert 0.0 < row["ratio_exact"] < 1.0, row
        ratios.append(row["ratio_exact"])
    assert led["W_arrow_full"] > max(r["E_B"] for r in led["rows"])
    assert led["W_arrow_full"] <= led["E_A"] + 1e-12, "the theory's ceiling respects E_A >= E_B"


def test_rotation_is_not_optimal_in_ordered_phase():
    """At g = J/2 Bob's single-site CPTP optimum beats the conditional rotation
    by a large factor (the conditional state of his site is mixed); at the
    critical point the rotation saturates it (test_single_site_potential_equals_chain_protocol)."""
    L, A, B = 5, 0, 1
    H = _tfim_dense(L, 0.5)
    _, psi = chains.tfim_ground_state(L, 0.5)
    rho = np.outer(psi, psi.conj()).astype(complex)
    e_b = chains.run_chain_qet(L, 0.5, A, B, theta_method="closed")["E_B"]
    W = R.qet_potential(H, rho, (A,), region_B=(B,), method="kraus",
                        kraus_kwargs={"restarts": 3})["W"]
    assert W > 2.0 * e_b, f"g=0.5: CPTP site potential {W:.3e} vs rotation {e_b:.3e}"
    assert W <= R.qet_potential(H, rho, (A,))["W"] + 1e-12, "site potential above the complement ceiling"


# --------------------------------------------------------------------------
# Gaussian mirror
# --------------------------------------------------------------------------


def test_gaussian_potential_closed_form_and_monotonicity():
    for N, m, modes_A in [(2, 0.7, (0,)), (3, 0.5, (0,)), (4, 0.5, (0, 1)), (4, 0.3, (1,))]:
        K = harmonic_chain_K(N, m, "dirichlet")
        res = R.gaussian_qet_potential(K, modes_A)
        assert abs(res["W"] - res["W_identity"]) < 1e-12, res
        assert res["W"] > 1e-6, "the Gaussian vacuum carries no QET potential?"
        assert res["log_negativity"] > 0.0
    K = harmonic_chain_K(3, 0.5, "dirichlet")
    aud = R.gaussian_monotonicity_audit(K, (0,), n_samples=240, seed=2, tol=1e-9)
    assert aud.passed and aud.worst_violation < 1e-10, aud.details
    assert aud.details["teeth"]["nonfree_alice"] > 1e-3
    K4 = harmonic_chain_K(4, 0.5, "dirichlet")
    aud = R.gaussian_monotonicity_audit(K4, (0, 1), n_samples=160, seed=3, tol=1e-9)
    assert aud.passed and aud.worst_violation < 1e-10, aud.details


# --------------------------------------------------------------------------
# L3 close-out A: the free class beyond unitality (Theorem 2)
# --------------------------------------------------------------------------


def test_free_class_is_commuting_kraus_iff_bonds_are_in_convex_position():
    """Theorem 2. Where the joint bond eigenvalue vectors are extreme points the
    free class IS the commuting-Kraus class (no unitality assumed); where one is
    a convex combination of the others an explicit NON-unital energy-non-signalling
    instrument exists that does not commute with the bonds."""
    # every model the candidate uses is in convex position
    for H, region in [(_ground_rho(1.0, 0.5)[0], (0,)), (_ground_rho(0.3, 2.0)[0], (1,)),
                      (_tfim_dense(5, 1.0), (0,)), (_tfim_dense(5, 1.0), (1, 2)),
                      (_tfim_dense(6, 1.0), (2, 3))]:
        info = R.free_class_extremality(H, region)
        assert info["in_convex_position"], info
        assert R.nonfree_energy_nonsignalling_instrument(H, region) is None
    # the star model is not: eigenvalue 0 of sx_0 + sx_1 is the midpoint of +-2
    H = R.star_hamiltonian(1.0, 0.5)
    info = R.free_class_extremality(H, (0, 1))
    assert not info["in_convex_position"], info
    assert sum(1 for e in info["extreme"] if not e) == 1
    w = R.nonfree_energy_nonsignalling_instrument(H, (0, 1))
    assert w is not None
    assert w["defect"] < 1e-12, f"the witness must be exactly free: {w['defect']:.3e}"
    assert w["commutator"] > 1e-2, "the witness must NOT be commuting-Kraus"
    assert w["unitality_defect"] > 1e-2, "the witness must be non-unital"
    assert abs(sum(w["weights"]) - 1.0) < 1e-12


def test_the_wider_free_class_beats_the_closed_form_where_theorem_2_fails():
    """Consequence: Prop. A's closed form is the COMMUTING-KRAUS value. Where the
    bond spectrum is not in convex position the true free class extracts strictly
    more, and the closed form is not even monotone under it."""
    H = R.star_hamiltonian(1.0, 0.5)
    w, U = np.linalg.eigh(H)
    rho = np.outer(U[:, 0], U[:, 0].conj())
    W_closed = R.qet_potential(H, rho, (0, 1))["W"]
    K = R.nonfree_energy_nonsignalling_instrument(H, (0, 1))["kraus"]
    yield_free = sum(p * slp.extractable_energy_kraus(H, r, (2,), restarts=4)["extractable"]
                     for p, r in R.apply_instrument(rho, K, (0, 1)))
    assert yield_free > 2.0 * W_closed, (
        f"witness yield {yield_free:.6f} vs closed form {W_closed:.6f}")
    after = sum(p * R.qet_potential(H, r, (0, 1))["W"]
                for p, r in R.apply_instrument(rho, K, (0, 1)))
    assert after - W_closed > 0.1, "the closed form must visibly fail monotonicity here"


def test_sdp_certificate_confirms_theorem_2():
    """Eq. (20): maximizing the weight outside the commutant over ALL free
    instruments is an SDP. Its optimum is 0 exactly where Theorem 2 says the free
    class is commuting-Kraus, and positive exactly where it is not."""
    pytest.importorskip("cvxpy")
    for H, region in [(_ground_rho(1.0, 0.5)[0], (0,)), (_tfim_dense(5, 1.0), (1, 2))]:
        res = R.search_free_instrument(H, region)
        assert res["in_convex_position"]
        assert res["is_commuting_kraus_only"], res
        assert abs(res["w_off"]) < 1e-8, f"a free non-commuting instrument exists? {res}"
    res = R.search_free_instrument(R.star_hamiltonian(1.0, 0.5), (0, 1))
    assert not res["in_convex_position"]
    assert not res["is_commuting_kraus_only"]
    assert res["w_off"] > 1.0, res
    assert res["commutator"] > 1e-2, res


# --------------------------------------------------------------------------
# L3 close-out B: the two-way theory (Theorem 4)
# --------------------------------------------------------------------------


def test_qet_potential_is_linear_and_its_non_hotta_part_lives_on_alice():
    """Theorem 4: W_->(rho) = Tr[W_op rho], and W_op - H_B^loc is supported on A."""
    rng = np.random.default_rng(0)
    for H, region, n in [(_ground_rho(1.0, 0.5)[0], (0,), 2), (_tfim_dense(5, 1.0), (0,), 5),
                         (_tfim_dense(6, 1.0), (2, 3), 6)]:
        W_op = R.qet_potential_operator(H, region)
        worst = 0.0
        for i in range(24):
            rho = R.random_state(n, rng, ("pure", "mixed", "ground_mix", "thermal")[i % 4], H=H)
            worst = max(worst, abs(R.qet_potential(H, rho, region)["W"]
                                   - R.state_energy(rho, W_op)))
        assert worst < 1e-12, f"W_-> is not linear: worst |diff| {worst:.3e}"
        _, H_loc = R.split_hamiltonian(H, region)
        G = W_op - H_loc
        dB = 2 ** (n - len(region))
        G_A = np.einsum("abcb->ac", R._as_region_blocks(G, region, n)) / dB
        dev = np.max(np.abs(G - R._embed_pair(np.kron(G_A, np.eye(dB)), region, n)))
        assert dev < 1e-10, f"W_op - H_B^loc is not supported on A: {dev:.3e}"


def test_symmetric_booking_has_the_same_free_class():
    """The symmetric split (9) sums to H and leaves Eq. (2) — hence the free
    instruments — unchanged; only Bob's status changes."""
    for H, region in [(_ground_rho(1.0, 0.5)[0], (0,)), (_tfim_dense(5, 1.0), (2,))]:
        E_A, E_B, V = R.split_hamiltonian_symmetric(H, region)
        assert np.max(np.abs(E_A + E_B - H)) < 1e-12
        n = int(round(np.log2(H.shape[0])))
        dA, dB = 2 ** len(region), 2 ** (n - len(region))
        # V is traceless on both factors
        assert np.max(np.abs(np.einsum("abcb->ac", R._as_region_blocks(V, region, n)))) < 1e-10
        assert np.max(np.abs(np.einsum("abad->bd", R._as_region_blocks(V, region, n)))) < 1e-10
        P, _ = R.free_projectors(H, region)
        rng = np.random.default_rng(2)
        K = R.random_free_instrument(P, 2, rng)
        assert R.energy_signalling_defect(H, K, region) < 1e-12


def test_two_way_communication_gives_bob_nothing():
    """Eq. (12): with only Bob's device being filled, every finite-round two-way
    protocol delivers exactly W_->(rho), for any number of rounds."""
    for h, k in [(1.0, 0.5), (0.3, 2.0)]:
        H, rho = _ground_rho(h, k)
        W = R.qet_potential(H, rho, (0,))["W"]
        for rounds in (1, 2, 3):
            res = R.two_way_potential(H, rho, (0,), rounds=rounds, cash_both=False)
            assert abs(res["W"] - W) < 1e-10, (
                f"{rounds} rounds: two-way {res['W']:.12f} != one-way {W:.12f}")
    L, g = 5, 1.0
    H = _tfim_dense(L, g)
    _, psi = chains.tfim_ground_state(L, g)
    rho = np.outer(psi, psi.conj()).astype(complex)
    W = R.qet_potential(H, rho, (0,))["W"]
    assert abs(R.two_way_potential(H, rho, (0,), rounds=2, cash_both=False)["W"] - W) < 1e-10


def test_two_way_beats_one_way_only_by_spending_the_measurement_devices():
    """With BOTH devices cashing out the two-way value exceeds W_-> and grows with
    the round count — but the net of the free instruments' injections does not."""
    H, rho = _ground_rho(1.0, 0.5)
    vals = [R.two_way_potential(H, rho, (0,), rounds=r) for r in (1, 2, 3)]
    W = vals[0]["W_one_way"]
    assert abs(vals[0]["W"] - W) < 1e-10, "one round with the receiver cashing is W_->"
    assert vals[1]["W"] > 5.0 * vals[0]["W"], vals[1]["W"]
    assert vals[2]["W"] > vals[1]["W"] + 0.5, "the ratchet must keep paying"
    nets = [v["net"] for v in vals]
    assert max(nets) - min(nets) < 1e-9, f"the net must be round-independent: {nets}"
    assert all(nt < 0.0 for nt in nets), "the measurement devices pay for the surplus"


@pytest.mark.parametrize("H_fn,region,n_samples", [
    (lambda: _ground_rho(1.0, 0.5)[0], (0,), 40),
    (lambda: _tfim_dense(4, 1.0), (0,), 24),
    (lambda: _tfim_dense(5, 1.0), (2,), 30),
])
def test_two_way_ledger_is_conserved_with_teeth(H_fn, region, n_samples):
    """Theorem 4 numerically: over random finite-round two-way protocols (Alice's
    free instruments, Bob's measurements and channels, both directions of
    communication) Bob's take plus the remaining W_-> is CONSERVED, not merely
    bounded; a non-free instrument on A breaks it upward."""
    res = R.two_way_monotonicity_audit(H_fn(), region, n_samples=n_samples, seed=11,
                                       rounds=3, tol=1e-10)
    assert res.passed, res
    assert res.details["worst_violation_of_(14)"] < 1e-12, res.details
    assert res.details["worst_ledger_drift"] < 1e-12, (
        f"the ledger is not conserved: drift {res.details['worst_ledger_drift']:.3e}")
    assert res.details["teeth"] > 0.05, f"non-free protocols never broke it: {res.details}"


# --------------------------------------------------------------------------
# L3 close-out C: the exact single-site bound and its entropic form
# --------------------------------------------------------------------------


def _chain_region(L, g, A, B, J=1.0):
    """(h_R, branches, E_B, dims) with Bob's site the first factor of R."""
    ground = chains.tfim_ground_state(L, g, J)
    h_ops = chains.tfim_local_energy_ops(L, g, J)
    led = chains.run_chain_qet(L, g, A, B, J=J, ground=ground, h_ops=h_ops,
                               theta_method="closed")
    Rr = tuple(q for q in (B - 1, B, B + 1) if 0 <= q < L)
    order = [B] + [q for q in Rr if q != B]
    k = len(order)
    h_loc = np.asarray(chains.local_hamiltonian(h_ops, B).toarray(), dtype=complex)
    h_R = R._partial_trace_B(h_loc, tuple(sorted(order)), L) / 2 ** (L - k)
    perm = [sorted(order).index(q) for q in order]
    h_R = R._herm(np.transpose(h_R.reshape((2,) * (2 * k)),
                               perm + [k + i for i in perm]).reshape(2**k, 2**k))
    br = [(o["p"], chains.reduced_density(o["state_meas"].astype(complex), L, order))
          for o in led["outcomes"].values()]
    return h_R, br, led["E_B"], (2, 2 ** (k - 1))


def test_single_site_closed_form_is_exact():
    """Prop. B (15): the Procrustes formula reproduces the chain protocol's E_B and
    Hotta's Eq. (11) to roundoff — an exact answer where the SDP was a 0.92 bound."""
    worst = 0.0
    for L, g in [(6, 1.0), (8, 1.0), (8, 0.5)]:
        for B in range(1, 4):
            h_R, br, E_B, dims = _chain_region(L, g, 0, B)
            val = sum(p * R.single_site_extractable(h_R, t, dims)["value"] for p, t in br)
            worst = max(worst, abs(val - E_B) / max(E_B, 1e-12))
    assert worst < 1e-9, f"closed form vs achieved rotation: worst relative {worst:.3e}"
    for h, k in GRID[::3]:
        H, rho = _ground_rho(h, k)
        pot = R.qet_potential(H, rho, (0,))
        val = sum(b["p"] * R.single_site_extractable(b["H_eff"], b["sigma_B"], (2, 1))["value"]
                  for b in pot["branches"])
        assert abs(val - hotta.e_b_optimal(h, k)) < 1e-12, (h, k, val)


def test_closed_form_never_exceeds_the_relaxations_it_replaces():
    """The exact value sits below every relaxation: the marginal-fixed SDP, the
    region-unitary bound, and Bob's single-site CPTP optimum sits above it."""
    pytest.importorskip("cvxpy")
    for L, g, B in [(6, 1.0, 1), (6, 1.0, 2), (8, 1.0, 1)]:
        h_R, br, E_B, dims = _chain_region(L, g, 0, B)
        exact = sum(p * R.single_site_extractable(h_R, t, dims)["value"] for p, t in br)
        marg = R.one_shot_bound_marginal(h_R, br, dims, (B,))["bound"]
        region = R.one_shot_bound(h_R, br, eps_grid=(0.0,))["bound_exact"]
        w_site = sum(p * slp.extractable_energy_kraus(h_R, t, (0,), restarts=3)["extractable"]
                     for p, t in br)
        assert exact <= marg + 1e-9 <= region + 1e-9, (exact, marg, region)
        assert exact <= w_site + 1e-7, "a CPTP map cannot do worse than the best unitary"


def test_entropic_bound_is_valid_and_its_looseness_is_recorded():
    """Prop. C (17): E_B <= 2 c (exp(-H_min(X|R)) - 1/2) holds on every row, is zero
    exactly when Alice's bit is unguessable from Bob's region, and is LOOSE — the
    honest measurement that keeps the entropic form out of the tight column."""
    worst_ratio = 0.0
    for L, g in [(6, 1.0), (8, 1.0)]:
        for B in (1, 2, 3):
            h_R, br, E_B, dims = _chain_region(L, g, 0, B)
            ent = R.one_shot_bound_entropic(h_R, br, dims)
            assert E_B <= ent["bound_guess"] + 1e-12, (L, g, B, ent)
            assert E_B <= ent["bound_rows"] + 1e-12
            assert E_B <= ent["bound_guess_cptp"] + 1e-12
            assert abs(ent["exact"] - E_B) < 1e-10
            assert 0.5 <= ent["p_guess"] <= 1.0
            assert ent["H_min_XR"] <= np.log(2.0) + 1e-12
            worst_ratio = max(worst_ratio, E_B / ent["bound_guess"])
    assert worst_ratio < 0.05, "if this ever tightens, the paper's claim must change"
    # the bound vanishes with the correlation: a product state gives p_guess = 1/2
    h_R, br, _, dims = _chain_region(6, 1.0, 0, 1)
    rho_R = sum(p * t for p, t in br)
    flat = R.one_shot_bound_entropic(h_R, [(0.5, rho_R), (0.5, rho_R)], dims)
    assert abs(flat["p_guess"] - 0.5) < 1e-12
    assert flat["bound_guess"] < 1e-12, "no information must mean no bound"


def test_e_b_is_a_daemonic_gain():
    """Prop. (18): QET's teleported energy IS the daemonic ergotropy of Francica
    et al. (arXiv:1608.00124, Eqs. (3), (5)), the unconditional ergotropy being
    zero because the vacuum is strongly locally passive."""
    for h, k in [(1.0, 0.5), (0.3, 2.0), (2.0, 2.0)]:
        H, rho = _ground_rho(h, k)
        pot = R.qet_potential(H, rho, (0,))
        br = [(b["p"], b["sigma_B"]) for b in pot["branches"]]
        hs = [b["H_eff"] for b in pot["branches"]]
        H_B, Xs, Ys = R.bond_operators(H, (0,))
        rho_A = R._partial_trace_B(rho, (0,), 2)
        rho_B = R._partial_trace_A(rho, (0,), 2)
        h_mean = H_B + sum(np.real(np.trace(X @ rho_A)) * Y for X, Y in zip(Xs, Ys))
        d = R.daemonic_gain(hs, br, (2, 1), h_ref=h_mean, rho_ref=rho_B)
        assert abs(d["W_conditional"] - hotta.e_b_optimal(h, k)) < 1e-12
        assert d["W_unconditional"] < 1e-12, "the vacuum must be passive with no message"
        assert abs(d["daemonic_gain"] - hotta.e_b_optimal(h, k)) < 1e-12


# --------------------------------------------------------------------------
# L3 close-out D: sub-normalized smoothing, and L = 12
# --------------------------------------------------------------------------


def test_subnormalized_smoothing_only_bites_at_saturation():
    """Prop. E: over Tomamichel's own ball (Def. 6.8, sub-normalized) the smooth
    min-entropy is unchanged until the normalized value saturates at ln d, where
    it adds exactly -ln(1 - eps^2)."""
    rng = np.random.default_rng(4)
    for n in (1, 2, 3):
        for _ in range(2):
            rho = R.random_state(n, rng, "mixed")
            for eps in (0.05, 0.4):
                a = R.smooth_min_entropy(rho, eps)
                b = R.smooth_min_entropy(rho, eps, subnormalized=True)
                assert b >= a - 1e-12
                assert abs(b - a) < 1e-9, f"n={n} eps={eps}: {a} vs {b}"
    for n in (1, 2):
        rho = np.eye(2**n, dtype=complex) / 2**n
        for eps in (0.2, 0.4):
            gain = (R.smooth_min_entropy(rho, eps, subnormalized=True)
                    - R.smooth_min_entropy(rho, eps))
            assert abs(gain - (-np.log(1.0 - eps * eps))) < 1e-9, (n, eps, gain)
    # the sub-normalized Ky Fan floor at trace 1 is the old one
    h = np.diag([0.0, 1.0, 2.0, 3.0])
    for s in (0.0, np.log(2.5), 10.0):
        assert abs(R.ky_fan_floor(h, s, trace=1.0) - R.ky_fan_floor(h, s)) < 1e-14
    assert R.ky_fan_floor(h, np.log(2.0), trace=0.5) < R.ky_fan_floor(h, np.log(2.0))


def test_subnormalized_one_shot_bound_is_valid_and_never_helps():
    """Chain (iii'): the enlarged ball buys at most -ln(1-eps^2) of entropy and pays
    twice the penalty at a trace-t floor, so it never tightens a bound here."""
    grid = (0.0, 0.01, 0.03, 0.1, 0.2, 0.3)
    for h, k in [(1.0, 0.5), (0.3, 2.0)]:
        H, rho = _ground_rho(h, k)
        pot = R.qet_potential(H, rho, (0,))
        br = [(b["p"], b["sigma_B"]) for b in pot["branches"]]
        hs = [b["H_eff"] for b in pot["branches"]]
        sub = R.one_shot_bound(hs, br, eps_grid=grid, subnormalized=True)
        assert np.all(hotta.e_b_optimal(h, k) <= sub["bound_eps"] + 1e-12), sub["bound_eps"]
        assert sub["eps_best"] == 0.0, "smoothing must still not help"
    for L, g, B in [(6, 1.0, 1)]:
        h_R, br, E_B, _ = _chain_region(L, g, 0, B)
        sub = R.one_shot_bound(h_R, br, eps_grid=grid, subnormalized=True)
        norm = R.one_shot_bound(h_R, br, eps_grid=grid)
        assert np.all(E_B <= sub["bound_eps"] + 1e-12)
        assert sub["eps_best"] == 0.0 and norm["eps_best"] == 0.0
        assert abs(sub["bound_exact"] - norm["bound_exact"]) < 1e-12


def test_sparse_chain_potential_matches_dense_and_reaches_L12():
    """Prop. A without a dense 2^L: exact against qet_potential for L <= 9, and
    L = 12 in seconds (the candidate's old wall was L <= 10)."""
    for L, g in [(5, 1.0), (7, 1.0), (9, 0.5)]:
        H = _tfim_dense(L, g)
        _, psi = chains.tfim_ground_state(L, g)
        rho = np.outer(psi, psi.conj()).astype(complex)
        dense = R.qet_potential(H, rho, (0,))["W"]
        sparse = R.chain_potential_sparse(L, g, 0)["W"]
        assert abs(dense - sparse) < 1e-9, f"L={L} g={g}: {dense} vs {sparse}"
    big = R.chain_potential_sparse(12, 1.0, 0)
    assert big["W"] > 0.17 and np.isfinite(big["W"])
    assert all(abs(b["p"] - 0.5) < 1e-9 for b in big["branches"])


def test_region_ledger_matches_the_dense_ledger_and_runs_at_L12():
    """Bob's single site can be judged inside R = {B-1, B, B+1} alone; the region
    ledger reproduces the dense one at L = 6 and then goes to L = 12."""
    dense = R.chain_resource_ledger(6, 1.0, 0, [1, 2], method="kraus",
                                    kraus_kwargs={"restarts": 2})
    region = R.chain_region_ledger(6, 1.0, 0, [1, 2], marginal=False)
    assert abs(region["W_arrow_full"] - dense["W_arrow_full"]) < 1e-9
    for rr, dr in zip(region["rows"], dense["rows"]):
        assert abs(rr["E_B"] - dr["E_B"]) < 1e-12
        assert abs(rr["W_site"] - dr["W_site"]) < 1e-7, (rr["W_site"], dr["W_site"])
        assert abs(rr["bound_exact"] - dr["bound_exact"]) < 1e-9
        assert rr["unconditionally_passive"]
    led = R.chain_region_ledger(12, 1.0, 0, [1], marginal=False)
    assert led["L"] == 12 and led["W_arrow_full"] > 0.0
    for row in led["rows"]:
        assert row["E_B"] <= row["exact_unitary"] + 1e-12
        assert row["E_B"] <= row["bound_exact"] + 1e-12
        assert row["E_B"] <= led["W_arrow_full"] + 1e-12
        assert row["E_B"] <= led["E_A"] + 1e-12


def test_theorem_4_is_scoped_to_commuting_kraus_and_the_witness_breaks_it():
    """SCOPE OF EQ. (12), pinned. Theorem 4(b) is proved for commuting-Kraus
    instruments, which by Theorem 2 is the whole free class exactly where the bond
    spectrum is in convex position. Off convex position Eq. (12) is FALSE and fails
    from ONE-WAY communication alone: a single genuinely free (non-commuting-Kraus)
    instrument on A raises Prop. A's closed form by 6.1509x. This test exists so the
    scope cannot silently widen again."""
    H = R.star_hamiltonian(1.0, 0.5)
    assert not R.free_class_extremality(H, (0, 1))["in_convex_position"]
    w, U = np.linalg.eigh(H)
    rho = np.outer(U[:, 0], U[:, 0].conj())
    K = R.nonfree_energy_nonsignalling_instrument(H, (0, 1))["kraus"]
    # the instrument is genuinely free: it cannot signal energy to Bob at all
    assert R.energy_signalling_defect(H, K, (0, 1)) < 1e-12
    W_before = R.qet_potential(H, rho, (0, 1))["W"]
    W_after = sum(p * R.qet_potential(H, r, (0, 1))["W"]
                  for p, r in R.apply_instrument(rho, K, (0, 1)))
    assert abs(W_before - 0.03663409637065862) < 1e-12, W_before
    assert abs(W_after - 0.2253310108905166) < 1e-9, W_after
    assert abs(W_after / W_before - 6.150854892410876) < 1e-6, W_after / W_before
    # ONE free instrument on A, no communication from B: this is not a two-way effect
    assert len(K) >= 2
    # and W_-> is still linear, so part (a) of Theorem 4 survives: only (b) fails
    W_op = R.qet_potential_operator(H, (0, 1))
    assert abs(R.state_energy(rho, W_op) - W_before) < 1e-12
    # the audit must SURFACE this, not hide it behind its commuting-Kraus sampling
    aud = R.two_way_monotonicity_audit(H, (0, 1), n_samples=8, seed=5, rounds=2,
                                       tol=1e-10, strict=False)
    assert aud.details["free_class_is_commuting_kraus"] is False
    assert aud.details["worst_violation_wider_class"] > 0.1, aud.details
    assert aud.details["worst_ledger_drift"] < 1e-12, "part (a) must still hold"
    # where convex position DOES hold there is nothing outside the commutant to find
    aud2 = R.two_way_monotonicity_audit(_ground_rho(1.0, 0.5)[0], (0,), n_samples=8,
                                        seed=5, rounds=2, tol=1e-10)
    assert aud2.details["free_class_is_commuting_kraus"] is True
    assert aud2.details["worst_violation_wider_class"] == 0.0


def test_bob_only_two_way_really_lets_bob_measure_and_books_his_injection():
    """`cash_both=False` must be genuinely two-way — either party may measure and
    broadcast — and must book Bob's own measurement injection against his tally,
    since Theorem 4 bounds the NET take. Both sides are explored and the answer is
    still exactly W_->."""
    H, rho = _ground_rho(1.0, 0.5)
    W = R.qet_potential(H, rho, (0,))["W"]
    for rounds in (1, 2, 3):
        res = R.two_way_potential(H, rho, (0,), rounds=rounds, cash_both=False)
        assert abs(res["W"] - W) < 1e-10, (rounds, res["W"], W)
        assert res["b_first_skipped"] is False, "Bob's branch must be evaluated here"
    # without booking Bob's self-injection the Bob-measures branch would ratchet:
    # his free instrument injects and he cashes it straight back out
    injected = R.two_way_potential(H, rho, (0,), rounds=2, cash_both=False)["injected"]
    assert injected > 0.1, "Bob's own instrument does inject energy"
    # cash_both=True is unchanged by the fix
    both = R.two_way_potential(H, rho, (0,), rounds=2)
    assert both["W"] > 5.0 * W


def test_two_way_search_actually_iterates_over_both_first_movers():
    """PINS THE TWO-SIDED SEARCH. `two_way_potential` is only two-way because its
    loop tries BOTH parties as the first mover; if that iteration is ever reduced
    to ("A",) the protocol is one-way again while still returning a plausible
    number (the Alice branch attains W_-> on its own, so a value check cannot see
    the regression). These are parameter points where measuring on BOB's side first
    strictly wins, so the optimum can only come from the B branch."""
    for L, g, A, ratio in [(4, 1.0, (1,), 2.6729), (5, 0.5, (1,), 2.8800)]:
        H = _tfim_dense(L, g)
        _, psi = chains.tfim_ground_state(L, g)
        rho = np.outer(psi, psi.conj()).astype(complex)
        res = R.two_way_potential(H, rho, A, rounds=2)
        assert set(res["by_side"]) == {"A", "B"}, (
            f"both first movers must be evaluated, got {sorted(res['by_side'])}")
        assert res["measured_first"] == "B", res["by_side"]
        assert res["W"] == pytest.approx(res["by_side"]["B"], abs=1e-12)
        assert res["by_side"]["B"] > res["by_side"]["A"], res["by_side"]
        assert res["by_side"]["B"] / res["by_side"]["A"] == pytest.approx(ratio, abs=1e-3)
        # and the B branch really is the better protocol, not a bookkeeping artifact
        assert res["W"] > 2.5 * res["by_side"]["A"]
    # the Bob-only booking explores both sides too (minimal model: Bob is one qubit)
    H, rho = _ground_rho(1.0, 0.5)
    res = R.two_way_potential(H, rho, (0,), rounds=2, cash_both=False)
    assert set(res["by_side"]) == {"A", "B"}, res
    assert res["b_first_skipped"] is False
    assert res["by_side"]["B"] == pytest.approx(R.qet_potential(H, rho, (0,))["W"], abs=1e-10)
