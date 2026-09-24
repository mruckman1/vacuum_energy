"""Anchors for the harvest-then-teleport composite (Layer 7, L4).

Spec anchors (task L4):

(a) two-mode squeezed vacuum of known r: the Gaussian daemonic ergotropy
    equals the closed form of the module docstring, surplus = E_daemonic
    = Omega_B sinh^2 r with E_local = 0, to 1e-10; the squeezed-THERMAL
    heterodyne/homodyne closed forms (Kua-Serafini-Genoni Eqs. (50),
    (51)) to 1e-10;
(b) product states give zero surplus;
(c) the surplus never exceeds the resource bound surplus <= k T_B I(A:B)
    (Landauer / Szilard form, Eq. (8) of the module), verified on random,
    squeezed-thermal and classically correlated states — with the
    documented facts that no bound of the form (scale) x f(E_N) exists for
    mixed states (a separable, classically correlated state has a positive
    surplus), that the joint *Gaussian* ergotropy is not a bound, and that
    the task's conjecture "surplus <= ergotropy of the joint state" is
    FALSE even for the full unitary ergotropy: two thermal modes at
    different temperatures mixed on a beam splitter are unitarily passive
    (zero joint ergotropy) yet carry a positive daemonic surplus — the
    Szilard engine (test below exhibits it);
(d) no-signaling: Bob's unconditional state is independent of Alice's
    measurement choice to 1e-14;
(e) the ledger closes to 1e-9 including the switching work;
(f) the exchange rate is deterministic under a re-run and under a
    plain-numpy re-derivation from the stored covariance, to 1e-12.

Plus the Hotta reference: the no-communication extraction from the
locally passive (audited) ground state of the coupled pair is zero, and
E_B <= E_A.

Open items of L4 closed here (docs/STATUS.md):

(g) NON-GAUSSIAN measurements on Alice (vacuum/composite/fock_conditioning):
    the Fock representation of the two-mode Gaussian state is pinned on
    the vacuum, thermal and two-mode-squeezed closed forms and on its
    first and second moments (1e-12); on the two-mode squeezed vacuum the
    photon-number-resolved daemonic gain equals Omega sinh^2 r (= the
    Gaussian value: every rank-one measurement purifies Bob) and the
    on/off gain equals Omega tanh^2 r, both to 1e-10; on a pure NON-phase-
    invariant state PNR and the Gaussian optimum tie at Omega (nu_B - 1/2);
    on mixed states Bob's unconditional ergotropy from the Fock spectrum
    equals the Gaussian closed form to 1e-10, PNR >= on/off (refinement +
    convexity), product states give zero, and the cutoff ladder converges;
(h) the 'xp' (derivative) coupling runs through the same composite with
    the ledger closing on the general quadratic form, and harvests at
    full LIGHT CONTACT (delay = separation) where the same 'xp' pair
    switched simultaneously on a spacelike window has E_N = 0;
(i) the IR side: mass = 0 is admissible on the Dirichlet chain (positive
    definite K, no zero mode) and the massless row closes its ledger; the
    periodic massless chain is rejected up front (zero mode, the
    vacuum.core convention).

Measured numbers appear in assertion messages (repo testing convention).
Whole file budget: well under 90 s.
"""

import math

import numpy as np
import pytest
from scipy.linalg import expm

from vacuum.audits import AuditViolation
from vacuum.core import Omega, ground_state_cov, log_negativity, mean_energy
from vacuum.composite import (
    HarvestConfig,
    composite_ledger,
    conditional_covariance,
    daemonic_surplus,
    ebits,
    ergotropy_finite,
    field_coupling,
    fock_conditioning,
    fock_moments,
    fock_tensor,
    gaussian_measurement,
    harvest,
    hotta_variant,
    joint_ergotropy,
    landauer_bound,
    local_ergotropy,
    local_extraction_coupled,
    mode_blocks,
    no_signaling_audit,
    optimal_fock_squeezing,
    optimal_measurement,
    pair_hamiltonian,
    tmsv_fock_gain_closed_form,
    two_mode_squeezed_thermal,
)
from vacuum.composite.fock_conditioning import complex_covariance, predicted_cutoff

LN2 = math.log(2.0)


def _random_two_mode_state(rng, scale=0.3):
    """S diag(nu, nu) S^T with a random symplectic S and nu > 1/2."""
    A = rng.normal(size=(4, 4))
    S = expm(Omega(2) @ (scale * (A + A.T)))
    nu = 0.5 + rng.exponential(0.5, size=2)
    return S @ np.diag(np.r_[nu, nu]) @ S.T


def _classical_correlated(N):
    """Separable, classically correlated: I/2 + N J (x) I_2, J = ones(2, 2)."""
    V = 0.5 * np.eye(4)
    J = np.ones((2, 2))
    V[:2, :2] += N * J
    V[2:, 2:] += N * J
    return V


# --------------------------------------------------------------------------
# (a) closed forms on two-mode squeezed vacuum / thermal states
# --------------------------------------------------------------------------


@pytest.mark.parametrize("r,Om", [(0.3, 1.0), (0.9, 1.7), (1.4, 0.6)])
def test_tmsv_daemonic_surplus_closed_form(r, Om):
    V = two_mode_squeezed_thermal(r, 0.0, (Om, Om))
    target = Om * math.sinh(r) ** 2
    for meas in ("optimal", "heterodyne", "homodyne", ("homodyne", 1.1), ("general", 5.0, 0.4)):
        res = daemonic_surplus(V, Om, measurement=meas, gap_A=Om)
        assert abs(res.E_local) < 1e-12, f"{meas}: E_local {res.E_local:.3e} on a locally thermal Bob"
        assert abs(res.surplus - target) < 1e-10, (
            f"{meas}: surplus {res.surplus:.15e} vs Omega sinh^2 r = {target:.15e}"
        )
        assert abs(res.E_daemonic - target) < 1e-10
    # the only E_N-monotone statement made: pure states, surplus = Omega sinh^2(E_N/2)
    E_N = log_negativity(V, [0], [1])
    assert abs(E_N - 2.0 * r) < 1e-12
    assert abs(daemonic_surplus(V, Om, "optimal", gap_A=Om).surplus
               - Om * math.sinh(0.5 * E_N) ** 2) < 1e-10
    # pure state: every measurement ties, the reported one is the heterodyne
    assert daemonic_surplus(V, Om, "optimal", gap_A=Om).measurement.kind == "heterodyne"


@pytest.mark.parametrize("r,N,Om", [(0.7, 0.4, 1.0), (0.3, 1.2, 2.5), (1.1, 0.05, 0.8)])
def test_tmst_matches_kua_serafini_genoni(r, N, Om):
    """Eqs. (50) (heterodyne) and (51) (homodyne) of arXiv:2506.22288."""
    V = two_mode_squeezed_thermal(r, N, (Om, Om))
    het = daemonic_surplus(V, Om, "heterodyne", gap_A=Om).surplus
    hom = daemonic_surplus(V, Om, "homodyne", gap_A=Om).surplus
    kua_het = Om * (2 * N + 1) ** 2 * math.sinh(2 * r) ** 2 / (2 + 2 * (2 * N + 1) * math.cosh(2 * r))
    kua_hom = Om * (2 * N + 1) * math.sinh(r) ** 2
    assert abs(het - kua_het) < 1e-10, f"heterodyne {het:.15e} vs Eq. (50) {kua_het:.15e}"
    assert abs(hom - kua_hom) < 1e-10, f"homodyne {hom:.15e} vs Eq. (51) {kua_hom:.15e}"
    assert het > hom, "heterodyne should beat homodyne on a squeezed thermal state"
    opt = daemonic_surplus(V, Om, "optimal", gap_A=Om)
    assert opt.surplus >= het - 1e-12 and opt.surplus >= hom - 1e-12
    assert abs(opt.surplus - het) < 1e-10, "heterodyne is optimal in the phase-invariant case"


def test_homodyne_is_the_general_dyne_limit():
    rng = np.random.default_rng(3)
    V = _random_two_mode_state(rng)
    for th in (0.0, 0.3, 1.2, 2.9):
        Vh = conditional_covariance(V, ("homodyne", th), gap_A=1.4)
        d6 = np.max(np.abs(conditional_covariance(V, ("general", 1e6, th), gap_A=1.4) - Vh))
        d8 = np.max(np.abs(conditional_covariance(V, ("general", 1e8, th), gap_A=1.4) - Vh))
        assert d8 < 1e-8, f"theta={th}: s=1e8 general-dyne is {d8:.3e} from the homodyne form"
        assert 30.0 < d6 / max(d8, 1e-300) < 300.0, (
            f"theta={th}: deviation should fall like 1/s: {d6:.3e} -> {d8:.3e}"
        )


def test_local_ergotropy_and_coupled_extraction_agree_without_coupling():
    Om = (1.3, 0.8)
    V = two_mode_squeezed_thermal(0.6, 0.2, Om)
    S = np.eye(4)
    S[np.ix_([1, 3], [1, 3])] = np.diag([math.exp(0.4), math.exp(-0.4)])  # squeeze Bob
    V = S @ V @ S.T
    _, V_B, _ = mode_blocks(V)
    erg, E, E_pass, nu = local_ergotropy(V_B, Om[1])
    assert erg > 1e-3
    _, H = pair_hamiltonian(Om, 0.0)
    ext, _, E0 = local_extraction_coupled(V, H)
    assert abs(ext - erg) < 1e-8, f"numerical extraction {ext:.12e} vs closed form {erg:.12e}"


# --------------------------------------------------------------------------
# (b) product states
# --------------------------------------------------------------------------


def test_product_states_give_zero_surplus():
    rng = np.random.default_rng(7)
    for _ in range(5):
        # thermal-squeezed product: local symplectics on a thermal product
        Sa = expm(Omega(1) @ (0.4 * (lambda A: A + A.T)(rng.normal(size=(2, 2)))))
        Sb = expm(Omega(1) @ (0.4 * (lambda A: A + A.T)(rng.normal(size=(2, 2)))))
        V = np.zeros((4, 4))
        nuA, nuB = 0.5 + rng.exponential(0.5, size=2)
        V[np.ix_([0, 2], [0, 2])] = Sa @ (nuA * np.eye(2)) @ Sa.T
        V[np.ix_([1, 3], [1, 3])] = Sb @ (nuB * np.eye(2)) @ Sb.T
        res = daemonic_surplus(V, 1.2, "optimal", gap_A=0.9)
        assert res.surplus == 0.0, f"product state surplus {res.surplus:.3e}"
        assert ebits(V).E_N == 0.0
        assert res.landauer_bound < 1e-12


# --------------------------------------------------------------------------
# (c) resource bounds
# --------------------------------------------------------------------------


def test_surplus_bounded_by_landauer_on_random_states():
    rng = np.random.default_rng(11)
    states = [_random_two_mode_state(rng) for _ in range(30)]
    states += [two_mode_squeezed_thermal(0.8, 0.3, (1.0, 1.5)), _classical_correlated(0.4)]
    worst_landauer = 0.0
    n_within_joint = 0
    for V in states:
        Om = tuple(np.exp(0.4 * rng.normal(size=2)))
        res = daemonic_surplus(V, Om[1], "optimal", gap_A=Om[0])
        bound, I_AB, nu_B, kT = landauer_bound(V, Om[1])
        assert res.surplus <= bound + 1e-12, (
            f"surplus {res.surplus:.6e} > k T_B I(A:B) = {bound:.6e} (I={I_AB:.4f}, kT={kT:.4f})"
        )
        worst_landauer = max(worst_landauer, res.surplus / bound if bound > 0 else 0.0)
        je = joint_ergotropy(V, Om)
        assert je["tail"] < 1e-12
        assert je["full"] >= je["gaussian"] - 1e-12 and je["full"] >= -1e-12
        n_within_joint += res.E_daemonic <= je["full"] + 1e-9
    assert worst_landauer <= 1.0
    # typical (hot, squeezed) states sit below the joint ergotropy; that is a
    # statistic, not a bound -- see the beam-splitter counterexample below
    assert n_within_joint == len(states), f"{n_within_joint}/{len(states)} within the joint ergotropy"


def _beam_splitter(phi):
    c, s = math.cos(phi), math.sin(phi)
    S = np.zeros((4, 4))
    S[:2, :2] = [[c, -s], [s, c]]
    S[2:, 2:] = [[c, -s], [s, c]]
    return S


def test_daemonic_surplus_exceeds_joint_ergotropy_on_a_passive_thermal_pair():
    """Two thermal modes of equal frequency at different temperatures, mixed
    on a beam splitter: the spectrum {(1-q1)q1^n (1-q2)q2^m} is already
    energy-sorted for q1^2 <= q2 (here q1 = 3e-3, q2 = 9e-4), so the pair is
    unitarily PASSIVE -- zero joint ergotropy, Gaussian and full -- and the
    beam splitter (energy conserving) cannot change that.  Bob's marginal is
    thermal (E_local = 0), yet Alice's heterodyne outcome purifies Bob and
    the daemon extracts a positive surplus: a Szilard engine, not a unitary.
    Measured on this rig: surplus 9.10e-7 against a joint ergotropy < 1e-12
    and a Landauer bound of order 1e-5."""
    nb1, nb2, phi = 3e-3, 9e-4, 1.0
    V0 = np.diag([0.5 + nb1, 0.5 + nb2, 0.5 + nb1, 0.5 + nb2])
    S = _beam_splitter(phi)
    V = S @ V0 @ S.T
    je = joint_ergotropy(V, (1.0, 1.0))
    assert abs(je["gaussian"]) < 1e-12 and abs(je["full"]) < 1e-12, (je["gaussian"], je["full"])
    assert log_negativity(V, [0], [1]) == 0.0
    res = daemonic_surplus(V, 1.0, "optimal", gap_A=1.0)
    assert abs(res.E_local) < 1e-15, "Bob's marginal is thermal"
    assert res.surplus > 5e-7, f"surplus {res.surplus:.3e} on the unitarily passive pair"
    assert res.E_daemonic > je["full"] + 1e-7
    assert res.surplus <= res.landauer_bound, (res.surplus, res.landauer_bound)


def test_classically_correlated_state_has_surplus_without_negativity():
    """E_N = 0, surplus > 0: no (scale) x f(E_N) bound with f(0) = 0 exists,
    and the joint Gaussian ergotropy (zero here) is not a bound either."""
    N = 0.5
    V = _classical_correlated(N)
    assert log_negativity(V, [0], [1]) == 0.0
    res = daemonic_surplus(V, 1.0, "optimal", gap_A=1.0)
    assert res.surplus > 0.1, f"surplus {res.surplus:.4e} on the classically correlated state"
    # closed forms: nu_B = 1/2 + N; heterodyne nu_{B|A} = 1/2 + N - N^2/(1+N)
    # (surplus N^2/(1+N)), homodyne nu_{B|A} = sqrt(1/4 + N) -- heterodyne wins here
    hom = daemonic_surplus(V, 1.0, "homodyne", gap_A=1.0).surplus
    assert abs(hom - (0.5 + N - math.sqrt(0.25 + N))) < 1e-12
    assert abs(res.surplus - N * N / (1.0 + N)) < 1e-12, (res.surplus, N * N / (1 + N))
    assert res.surplus > hom and res.measurement.kind == "heterodyne"
    je = joint_ergotropy(V, (1.0, 1.0))
    assert abs(je["gaussian"]) < 1e-12, f"joint Gaussian ergotropy {je['gaussian']:.3e} (expected 0)"
    assert je["full"] >= res.E_daemonic - 1e-9
    assert res.surplus <= res.landauer_bound


def test_optimal_measurement_dominates_named_points():
    rng = np.random.default_rng(5)
    for _ in range(6):
        V = _random_two_mode_state(rng)
        m, gain = optimal_measurement(V, omega_B=1.3, gap_A=0.7)
        het = daemonic_surplus(V, 1.3, "heterodyne", gap_A=0.7).surplus
        hom = max(daemonic_surplus(V, 1.3, ("homodyne", th), gap_A=0.7).surplus
                  for th in np.linspace(0, math.pi, 13))
        assert gain >= het - 1e-12 and gain >= hom - 1e-12, (gain, het, hom, m.label())


# --------------------------------------------------------------------------
# (d) no-signaling
# --------------------------------------------------------------------------


def test_no_signaling_bob_unconditional_state_is_measurement_independent():
    rng = np.random.default_rng(2)
    for _ in range(5):
        V = _random_two_mode_state(rng)
        res = no_signaling_audit(V, gap_A=1.1, tol=1e-14)
        assert res.passed, repr(res)
        assert res.details["max_deviation"] < 1e-14 * max(1.0, np.max(np.abs(V)))
    # the audit has teeth: a state whose "conditional" pieces were tampered fails
    V = two_mode_squeezed_thermal(0.5, 0.1)
    V_bad = V.copy()
    V_bad[1, 1] += 1e-6  # V_B block altered after the fact -> reconstruction cannot match
    V_A, V_B, C = mode_blocks(V)
    with pytest.raises(AuditViolation):
        # rebuild from the ORIGINAL blocks but compare against the tampered V_B
        from vacuum.composite.harvest_teleport import _from_blocks
        no_signaling_audit(_from_blocks(V_A, V_B + np.diag([1e-6, 0.0]), C * (1 + 1e-6)),
                           measurements=[gaussian_measurement("heterodyne")], tol=1e-30)


# --------------------------------------------------------------------------
# the Hotta reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("gaps,g", [((1.0, 1.0), 0.1), ((1.0, 1.3), 0.25), ((0.7, 2.0), 0.5)])
def test_hotta_reference_is_locally_passive_and_teleports(gaps, g):
    V = two_mode_squeezed_thermal(0.4, 0.2, gaps)
    hv = hotta_variant(V, gaps, g, "heterodyne")
    ref = hv["reference"]
    assert ref["passivity"].passed, repr(ref["passivity"])
    assert abs(ref["E_local"]) <= 1e-12, f"no-communication extraction {ref['E_local']:.3e} on the ground state"
    assert ref["E_A"] == pytest.approx(gaps[0], rel=1e-14), "heterodyne injects one quantum Omega_A"
    assert ref["E_B"] > 0.0, f"teleported E_B {ref['E_B']:.3e} not positive"
    assert ref["E_B_corr"] > 0.0
    assert ref["E_B"] <= ref["E_A"], f"Hotta inequality violated: E_B {ref['E_B']} > E_A {ref['E_A']}"
    assert ref["hotta_inequality"]
    assert abs(ref["nu_B_cond"] - 0.5) < 1e-12, "pure reference: any pure measurement purifies Bob"
    # the harvested (active) state under H_g: local extraction positive, books consistent
    har = hv["harvested"]
    assert har["E_local"] > 0.0
    assert har["E_B"] == pytest.approx(har["E_B_corr"] + har["E_B_inj"])
    V_A, V_B, C = mode_blocks(V)
    assert har["W_couple"] == pytest.approx(g * C[0, 0])
    with pytest.raises(ValueError, match="finite-strength"):
        hotta_variant(V, gaps, g, "homodyne")
    with pytest.raises(ValueError, match="unbounded"):
        pair_hamiltonian(gaps, gaps[0] * gaps[1])


def test_ebits_columns_on_pure_state_are_consistent():
    r = 0.65
    eb = ebits(two_mode_squeezed_thermal(r, 0.0))
    assert abs(eb.E_N_bits - 2 * r / LN2) < 1e-12
    # pure state: hashing bound = entropy of entanglement = EoF (symmetric closed form)
    assert abs(eb.hashing_bits - eb.eof_sym_bits) < 1e-12, (eb.hashing_bits, eb.eof_sym_bits)
    assert eb.symmetric and eb.distillable
    mixed = ebits(two_mode_squeezed_thermal(r, 0.5))
    assert mixed.hashing_bits <= mixed.eof_sym_bits <= mixed.E_N_bits + 1e-12 or mixed.E_N_bits == 0.0


# --------------------------------------------------------------------------
# (e), (f): the harvest through vacuum.protocol, the ledger, determinism
# --------------------------------------------------------------------------

TEST_CFG = HarvestConfig(
    N_field=25, mass=0.2, separation=2, gap=2.0, lam=0.6, switching="cos2", T_sw=2.0,
    n_segments0=8, substeps=4, conv_tol=1e-9, max_halvings=12,
)


@pytest.fixture(scope="module")
def composite_row():
    return composite_ledger(TEST_CFG)


def test_config_geometry():
    assert TEST_CFG.sites == (11, 13) and TEST_CFG.mirror_symmetric
    assert TEST_CFG.window == (0.0, 4.0) and not TEST_CFG.spacelike
    assert HarvestConfig(N_field=41, separation=12, T_sw=3.0).spacelike


def test_ledger_closes_with_switching_work(composite_row):
    out = composite_row.harvest
    defect = out.ledger_result.details["defect"]
    assert out.ledger_result.passed and abs(defect) <= 1e-9, f"ledger defect {defect:.3e}"
    table = out.ledger.table
    kinds = [r["kind"] for r in table]
    assert kinds.count("snapshot") == 2 and kinds.count("dissipation") == 0
    assert kinds.count("work") == out.n_segments
    assert out.W_in > 0.0, f"switching work {out.W_in:.3e} not positive"
    # W_in is the switching work: it equals Delta<H> between the ledger's own snapshots
    snaps = [r["value"] for r in table if r["kind"] == "snapshot"]
    assert abs((snaps[1] - snaps[0]) - out.W_in) <= 1e-9
    # ... and an independent recomputation of the exit energy under K_tot(lambda=0)
    E_exit = mean_energy(out.V, out.K_tot0)
    E_entry = mean_energy(ground_state_cov(out.K_tot0), out.K_tot0)
    assert abs((E_exit - E_entry) - out.W_in) <= 1e-9, (E_exit - E_entry, out.W_in)


def test_composite_row_passes_every_standing_audit(composite_row):
    r = composite_row.row
    assert r["converged"] and r["movement"] < 1e-9
    assert r["ledger_passed"] and r["passivity_initial_passed"] and r["passivity_entry_passed"]
    assert r["passivity_ref_passed"] and r["precision_passed"] and r["causality_passed"]
    assert r["no_signaling_passed"] and r["no_signaling_defect"] < 1e-14
    assert r["hotta_inequality"] and r["ref_E_local"] <= 1e-12
    assert r["all_audits_passed"], {k: v for k, v in r.items() if "passed" in k}
    assert r["E_N"] > 1e-3, f"test pocket should harvest: E_N = {r['E_N']:.3e}"
    assert r["surplus"] >= 0.0 and r["surplus_le_landauer"]
    assert isinstance(r["surplus_le_full_ergotropy"], bool)  # a reported comparison, not an audit
    assert r["surplus"] >= r["surplus_het"] - 1e-12 and r["surplus"] >= r["surplus_hom"] - 1e-12
    assert np.isfinite(r["exchange_rate"]) and r["exchange_rate"] > 0.0
    assert r["E_local"] > 0.0, "a freshly harvested detector pair is locally ACTIVE"
    assert r["hg_E_local"] > 0.0


def test_exchange_rate_deterministic_and_numpy_round_trip(composite_row):
    r1 = composite_row.row
    r2 = composite_ledger(TEST_CFG, fock=False).row  # the Gaussian columns are the subject
    for key in ("exchange_rate", "surplus", "W_in", "E_N", "ebits_per_work", "surplus_het"):
        assert abs(r1[key] - r2[key]) <= 1e-12 * max(1.0, abs(r1[key])), (key, r1[key], r2[key])
    # plain-numpy re-derivation of the surplus from the stored covariance
    V = composite_row.harvest.V_AB
    Om = TEST_CFG.gap
    iA, iB = [0, 2], [1, 3]
    V_A, V_B, C = V[np.ix_(iA, iA)], V[np.ix_(iB, iB)], V[np.ix_(iA, iB)]
    m = composite_row.surplus.measurement
    if m.kind == "homodyne":
        u = m.u
        V_c = V_B - np.outer(C.T @ u, C.T @ u) / (u @ V_A @ u)
    else:
        V_c = V_B - C.T @ np.linalg.inv(V_A + m.sigma) @ C
    surplus_np = Om * (math.sqrt(np.linalg.det(V_B)) - math.sqrt(np.linalg.det(V_c)))
    assert abs(surplus_np - r1["surplus"]) <= 1e-12, (surplus_np, r1["surplus"])
    assert abs(surplus_np / r1["W_in"] - r1["exchange_rate"]) <= 1e-12


def test_harvest_rejects_mismatched_field():
    with pytest.raises(ValueError, match="N_field"):
        harvest(field_coupling(HarvestConfig(N_field=9)), TEST_CFG)


# --------------------------------------------------------------------------
# (g) non-Gaussian measurements: the Fock representation and its anchors
# --------------------------------------------------------------------------


def test_fock_representation_closed_forms():
    # vacuum -> |0><0|
    D = fock_tensor(0.5 * np.eye(2), 6)
    assert abs(D[0, 0] - 1.0) < 1e-15 and np.max(np.abs(D)) - 1.0 < 1e-15
    # thermal -> (1-q) q^n on the diagonal, nothing off it
    nb = 0.7
    q = nb / (nb + 1.0)
    D = fock_tensor((nb + 0.5) * np.eye(2), 30)
    rho = D.reshape(30, 30)
    assert np.max(np.abs(np.real(np.diag(rho)) - (1 - q) * q ** np.arange(30))) < 1e-14
    assert np.max(np.abs(rho - np.diag(np.diag(rho)))) == 0.0
    # two-mode squeezed vacuum -> sqrt(1-q) q^(n/2) on |n, n>, q = tanh^2 r
    r = 0.5
    q = math.tanh(r) ** 2
    D = fock_tensor(two_mode_squeezed_thermal(r, 0.0), 24)
    amp = math.sqrt(1.0 - q) * q ** (np.arange(24) / 2.0)
    target = np.zeros((24, 24, 24, 24))
    for m in range(24):
        for n in range(24):
            target[m, m, n, n] = amp[m] * amp[n]
    assert np.max(np.abs(D - target)) < 1e-14, f"TMSV Fock elements off by {np.max(np.abs(D - target)):.3e}"


def test_fock_representation_moments_match_covariance():
    """<a^dag a>, <a a>, <a_A a_B>, <a_A a_B^dag> of the truncated matrix vs V."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4))
    S = expm(Omega(2) @ (0.25 * (A + A.T)))
    nu = 0.5 + rng.exponential(0.3, size=2)
    V = S @ np.diag(np.r_[nu, nu]) @ S.T
    sig = complex_covariance(V)                    # (1/2)<{xi, xi^dag}>, xi = (a_A, a_B, a_A^dag, a_B^dag)
    D = fock_tensor(V, 40)
    mom = fock_moments(D)
    assert abs(1.0 - mom["trace"].real) < 1e-12 and abs(mom["trace"].imag) < 1e-14
    for s in range(2):
        assert abs(mom["n"][s] - (sig[s, s].real - 0.5)) < 1e-12, (mom["n"][s], sig[s, s].real - 0.5)
        assert abs(mom["aa"][s] - sig[s, 2 + s]) < 1e-12
    assert abs(mom["a1a2"] - sig[0, 3]) < 1e-12 and abs(mom["a1a2d"] - sig[0, 1]) < 1e-12
    rho = D.reshape(1600, 1600)
    assert np.max(np.abs(rho - rho.conj().T)) < 1e-14


def test_ergotropy_finite_closed_forms():
    Om = 1.3
    eps = Om * (np.arange(3) + 0.5)
    erg, E, E_pass, mn = ergotropy_finite(np.diag([0.2, 0.5, 0.3]), eps)
    assert abs(E - Om * (0.2 * 0.5 + 0.5 * 1.5 + 0.3 * 2.5)) < 1e-14
    assert abs(E_pass - Om * (0.5 * 0.5 + 0.3 * 1.5 + 0.2 * 2.5)) < 1e-14
    assert abs(erg - Om * 0.4) < 1e-14 and mn >= 0.0
    psi = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    erg, E, E_pass, _ = ergotropy_finite(np.outer(psi, psi), eps)
    assert abs(E - Om) < 1e-14 and abs(E_pass - 0.5 * Om) < 1e-14 and abs(erg - 0.5 * Om) < 1e-14
    # homogeneity: the outcome-weighted (unnormalised) form
    erg2, _, _, _ = ergotropy_finite(0.25 * np.outer(psi, psi), eps)
    assert abs(erg2 - 0.25 * erg) < 1e-15
    with pytest.raises(ValueError, match="Hermitian"):
        ergotropy_finite(np.array([[0.5, 0.1], [0.3, 0.5]]), eps[:2])


@pytest.mark.parametrize("r,Om", [(0.3, 1.0), (0.6, 1.7), (0.8, 0.6)])
def test_tmsv_fock_gains_closed_form(r, Om):
    """PNR: Omega sinh^2 r (Bob left in |k>); on/off: Omega tanh^2 r (click branch
    is the shifted geometric mixture, ergotropy exactly Omega)."""
    V = two_mode_squeezed_thermal(r, 0.0, (Om, Om))
    fc = fock_conditioning(V, (Om, Om))
    pnr, onoff = tmsv_fock_gain_closed_form(r, Om)
    assert fc.converged and fc.trace_deficit < 1e-12, (fc.cutoffs_tried, fc.movement, fc.trace_deficit)
    assert abs(fc.E_local_full) < 1e-12, "Bob is thermal in his own H_B"
    assert abs(fc.gain_pnr - pnr) < 1e-10, f"PNR gain {fc.gain_pnr:.15e} vs Omega sinh^2 r = {pnr:.15e}"
    assert abs(fc.gain_onoff - onoff) < 1e-10, f"on/off gain {fc.gain_onoff:.15e} vs Omega tanh^2 r = {onoff:.15e}"
    assert fc.gain_onoff < fc.gain_pnr
    # pure state: the Gaussian optimum ties with PNR (any rank-one measurement purifies Bob)
    gs = daemonic_surplus(V, Om, "optimal", gap_A=Om)
    assert abs(gs.surplus - fc.gain_pnr) < 1e-10, (gs.surplus, fc.gain_pnr)
    assert abs(fc.p_click - math.tanh(r) ** 2) < 1e-12
    assert fc.no_signaling_defect < 1e-14 and fc.min_eig > -1e-13


def test_pure_non_phase_invariant_state_pnr_ties_gaussian():
    """Local squeezers on a TMSV: every rank-one measurement gives E_B - Omega_B/2."""
    gaps = (1.0, 1.3)
    V = two_mode_squeezed_thermal(0.4, 0.0, gaps)
    Sl = np.eye(4)
    Sl[np.ix_([0, 2], [0, 2])] = np.array([[math.cos(0.4), -math.sin(0.4)],
                                           [math.sin(0.4), math.cos(0.4)]]) @ np.diag([math.exp(0.2), math.exp(-0.2)])
    Sl[np.ix_([1, 3], [1, 3])] = np.diag([math.exp(-0.2), math.exp(0.2)])
    V = Sl @ V @ Sl.T
    fc = fock_conditioning(V, gaps)
    gs = daemonic_surplus(V, gaps[1], "optimal", gap_A=gaps[0])
    closed = gaps[1] * (gs.nu_B - 0.5)
    assert fc.converged, (fc.cutoffs_tried, fc.movement)
    assert abs(fc.gain_pnr - closed) < 1e-10, (fc.gain_pnr, closed)
    assert abs(gs.surplus - closed) < 1e-10, (gs.surplus, closed)
    assert abs(fc.E_local_full - gs.E_local) < 1e-10, (fc.E_local_full, gs.E_local)
    assert gs.E_local > 1e-3, "the local squeezing makes Bob active"
    assert fc.gain_onoff < fc.gain_pnr - 1e-3


def test_fock_conditioning_on_mixed_states_is_consistent():
    """E_local from the Fock spectrum = Gaussian Eq. (3); PNR >= on/off >= 0;
    no-signaling of the PNR POVM; the squeezed-basis search dominates."""
    rng = np.random.default_rng(4)
    states = [two_mode_squeezed_thermal(0.3, 0.1, (1.0, 1.0)), _classical_correlated(0.3)]
    for _ in range(3):
        A = rng.normal(size=(4, 4))
        S = expm(Omega(2) @ (0.15 * (A + A.T)))
        nu = 0.5 + rng.uniform(0.02, 0.3, size=2)
        states.append(S @ np.diag(np.r_[nu, nu]) @ S.T)
    ratios = []
    for V in states:
        gaps = (1.0, 1.0)
        fc = fock_conditioning(V, gaps)
        gs = daemonic_surplus(V, gaps[1], "optimal", gap_A=gaps[0])
        assert fc.converged, (fc.cutoffs_tried, fc.movement)
        assert abs(fc.E_local_full - gs.E_local) < 1e-10, (fc.E_local_full, gs.E_local)
        # PNR refines on/off and the ergotropy is convex, so PNR >= on/off >= 0
        # (both up to the roundoff of differences of O(1) energies)
        assert fc.gain_pnr >= fc.gain_onoff - 1e-12, (fc.gain_pnr, fc.gain_onoff)
        assert fc.gain_onoff >= -1e-12, f"on/off gain {fc.gain_onoff:.3e} is negative beyond roundoff"
        assert fc.no_signaling_defect < 1e-14 and fc.min_eig > -1e-13
        # the truncated PNR POVM sums to Tr rho_trunc = 1 - deficit (an identity), deficit < 1e-10
        assert abs(np.sum(fc.p_k) - (1.0 - fc.trace_deficit)) < 1e-14 and fc.trace_deficit < 1e-10
        ratios.append(fc.gain_pnr / gs.surplus)
    # the squeezed-basis search dominates plain PNR (checked once: it is a grid
    # over states, and the per-state cost is the file's budget)
    V = states[0]
    fc = fock_conditioning(V, (1.0, 1.0))
    sq = optimal_fock_squeezing(V, (1.0, 1.0), s_grid=(1.0, 2.0), n_theta=2)
    assert sq["gain"] >= fc.gain_pnr - 1e-12, (sq["gain"], fc.gain_pnr)
    s1 = [g for g in sq["grid"] if g[0] == 1.0]
    assert len(s1) == 1 and abs(s1[0][2] - fc.gain_pnr) < 1e-12, "the s = 1 entry is plain PNR"
    assert sq["result"].converged, "the reported squeezed basis must be a converged one"
    assert not (set(sq["skipped"]) & {(g[0], g[1]) for g in sq["grid"]})
    # on these mixed states the Gaussian optimum beats photon counting (measured 0.04 - 0.45)
    assert max(ratios) < 1.0, f"PNR/Gaussian ratios {np.round(ratios, 4)}"


def test_fock_gains_vanish_on_product_states():
    V = np.zeros((4, 4))
    V[np.ix_([0, 2], [0, 2])] = 0.9 * np.eye(2)                                  # thermal Alice
    V[np.ix_([1, 3], [1, 3])] = 0.5 * np.diag([math.exp(0.6), math.exp(-0.6)])   # squeezed Bob
    fc = fock_conditioning(V, (1.0, 1.2))
    assert abs(fc.gain_pnr) < 1e-13 and abs(fc.gain_onoff) < 1e-13, (fc.gain_pnr, fc.gain_onoff)
    assert fc.E_local_full > 1e-3


def test_composite_row_carries_fock_columns(composite_row):
    r = composite_row.row
    fc = composite_row.fock
    assert fc is not None and r["fock_converged"] and r["fock_local_consistent"]
    assert abs(r["E_local_fock"] - r["E_local"]) <= 1e-8 * max(1.0, abs(r["E_local"]))
    assert 0.0 <= r["gain_fock_onoff"] <= r["gain_fock_pnr"] + 1e-12 <= r["gain_fock_pnr_sq"] + 2e-12
    assert r["best_family"] == "gaussian" and r["surplus_best"] == r["surplus"]
    assert r["exchange_rate_best"] == r["exchange_rate"]
    # measured on this pocket: PNR reaches ~2% of the Gaussian optimum, the squeezed basis ~2%
    assert r["fock_sq_over_gaussian"] < 0.1, f"PNR(sq)/Gaussian = {r['fock_sq_over_gaussian']:.4f}"
    assert r["fock_trace_deficit"] < 1e-12 and r["fock_movement"] < 1e-10
    assert composite_row.fock_sq["result"].converged, "the reported squeezed basis is converged"


# --------------------------------------------------------------------------
# (h) the 'xp' coupling through the composite; light contact vs spacelike
# --------------------------------------------------------------------------


from dataclasses import replace as _replace  # noqa: E402

XP_CFG = _replace(TEST_CFG, coupling="xp")
LC_CFG = HarvestConfig(N_field=27, mass=0.2, separation=6, gap=2.0, lam=0.6, switching="cos2", T_sw=1.5,
                       delay=6.0, coupling="xp", n_segments0=8, substeps=4, conv_tol=1e-9, max_halvings=12)
SL_CFG = _replace(LC_CFG, delay=0.0)


@pytest.fixture(scope="module")
def xp_row():
    return composite_ledger(XP_CFG)


@pytest.fixture(scope="module")
def light_contact_rows():
    # the subject is E_N and the window, not the Fock columns: skip them
    return composite_ledger(LC_CFG, fock=False).row, composite_ledger(SL_CFG, fock=False).row


def test_config_coupling_and_delay():
    assert XP_CFG.coupling == "xp" and XP_CFG.window == (0.0, 4.0) and not XP_CFG.light_contact
    assert LC_CFG.window == (0.0, 9.0) and LC_CFG.light_contact and not LC_CFG.spacelike
    assert SL_CFG.window == (0.0, 3.0) and SL_CFG.spacelike and not SL_CFG.light_contact
    lc41 = HarvestConfig(N_field=41, separation=12, T_sw=3.0, delay=12.0)
    assert lc41.light_contact and lc41.window == (0.0, 18.0) and lc41.t_center_B == 15.0
    with pytest.raises(ValueError, match="coupling"):
        HarvestConfig(coupling="pp")
    with pytest.raises(ValueError, match="delay"):
        HarvestConfig(delay=-1.0)


def test_xp_row_closes_ledger_on_the_general_form_and_harvests(xp_row, composite_row):
    out = xp_row.harvest
    r = xp_row.row
    assert r["coupling"] == "xp"
    defect = out.ledger_result.details["defect"]
    assert out.ledger_result.passed and abs(defect) <= 1e-9, f"ledger defect {defect:.3e}"
    # lambda(t_end) = 0: the exit form is diag(K_tot0, I) again, so the work is Delta<H> under K_tot0
    E_exit = mean_energy(out.V, out.K_tot0)
    E_entry = mean_energy(ground_state_cov(out.K_tot0), out.K_tot0)
    assert abs((E_exit - E_entry) - out.W_in) <= 1e-9, (E_exit - E_entry, out.W_in)
    assert r["all_audits_passed"] and r["converged"] and r["movement"] < 1e-9
    assert r["E_N"] > 1e-3, f"'xp' test pocket should harvest: E_N = {r['E_N']:.3e}"
    assert r["E_local"] > 0.0 and r["surplus"] > 0.0 and r["surplus_le_landauer"]
    # measured: the derivative coupling injects an order of magnitude less switching work here
    assert r["W_in"] < composite_row.row["W_in"], (r["W_in"], composite_row.row["W_in"])
    assert r["fock_converged"] and r["fock_local_consistent"] and r["best_family"] == "gaussian"


def test_xp_harvests_at_light_contact_not_on_the_spacelike_window(light_contact_rows):
    lc, sl = light_contact_rows
    assert lc["light_contact"] and not lc["spacelike"] and lc["all_audits_passed"]
    assert sl["spacelike"] and not sl["light_contact"] and sl["all_audits_passed"]
    assert lc["E_N"] > 1e-4, f"light-contact 'xp' pair: E_N = {lc['E_N']:.3e} (measured 5.5e-4)"
    assert sl["E_N"] == 0.0, f"spacelike 'xp' pair: E_N = {sl['E_N']:.3e} (expected exactly 0)"
    # the surplus is paid by correlation, not negativity: positive on the spacelike row too
    assert sl["surplus"] > 0.0 and sl["I_AB"] > 0.0 and sl["exchange_rate"] > 0.0
    assert lc["exchange_rate"] > sl["exchange_rate"]


# --------------------------------------------------------------------------
# (i) the IR side: mass = 0 on the Dirichlet chain, the zero-mode convention
# --------------------------------------------------------------------------


def test_massless_dirichlet_row_is_admissible_and_closes(composite_row):
    cfg0 = _replace(TEST_CFG, mass=0.0)
    K = field_coupling(cfg0)
    assert np.min(np.linalg.eigvalsh(K)) > 0.0, "massless Dirichlet chain must be positive definite (no zero mode)"
    r = composite_ledger(cfg0, fock=False).row
    assert r["mass"] == 0.0 and r["all_audits_passed"] and abs(r["ledger_defect"]) <= 1e-9
    assert r["E_N"] > 1e-3 and r["surplus"] > 0.0 and np.isfinite(r["exchange_rate"])
    # measured: removing the regulator adds IR modes to the drive work (5.0e-2 vs 4.4e-2 here)
    assert r["W_in"] > composite_row.row["W_in"], (r["W_in"], composite_row.row["W_in"])
    with pytest.raises(ValueError, match="zero mode"):
        HarvestConfig(mass=0.0, bc="periodic")
    with pytest.raises(ValueError, match="mass"):
        HarvestConfig(mass=-0.1)


def test_predicted_cutoff_starts_the_ladder_without_deciding_it():
    """The prediction only picks the first rung; the movement check decides.

    A hot state needs a high cutoff (the geometric tail q^n, q = nbar/(nbar+1),
    decays slowly), a near-vacuum one needs the floor; forcing a deliberately
    bad start must not change the converged answer.
    """
    from vacuum.composite.fock_conditioning import _to_oscillator_units
    hot = two_mode_squeezed_thermal(0.55, 0.45, (1.0, 1.0))
    cold = two_mode_squeezed_thermal(0.1, 0.0, (1.0, 1.0))
    ones = np.array([1.0, 1.0])
    n_hot = predicted_cutoff(_to_oscillator_units(hot, ones))
    n_cold = predicted_cutoff(_to_oscillator_units(cold, ones))
    assert n_hot > n_cold and n_cold == 16, (n_hot, n_cold)
    assert predicted_cutoff(0.5 * np.eye(4)) == 16, "the vacuum needs no levels"
    # the ladder converges from the prediction and from a deliberately low start
    auto = fock_conditioning(hot, (1.0, 1.0))
    forced = fock_conditioning(hot, (1.0, 1.0), cutoff0=24)
    assert auto.converged and forced.converged
    assert auto.cutoffs_tried[0] == n_hot and forced.cutoffs_tried[0] == 24
    assert abs(auto.gain_pnr - forced.gain_pnr) < 1e-10, (auto.gain_pnr, forced.gain_pnr)


def test_squeezed_number_bases_are_reported_only_when_converged():
    """s -> infinity is a homodyne (a Gaussian measurement), and its truncation
    diverges as it approaches that limit: bases past the ceiling are skipped,
    non-converged ones rejected, and the reported gain is always converged."""
    hot = two_mode_squeezed_thermal(0.55, 0.45, (1.0, 1.0))
    sq = optimal_fock_squeezing(hot, (1.0, 1.0), s_grid=(1.0, 2.0, 4.0, 8.0), n_theta=2)
    assert sq["result"].converged and sq["s"] == 1.0, (sq["s"], len(sq["grid"]))
    assert len(sq["skipped"]) + len(sq["rejected"]) > 0, "hot state: squeezed bases cannot converge here"
    plain = fock_conditioning(hot, (1.0, 1.0)).gain_pnr
    assert abs(sq["gain"] - plain) < 1e-12, "falls back to plain PNR when no squeezed basis converges"
    # a cold state does accept squeezed bases, and they can only help
    cold = two_mode_squeezed_thermal(0.2, 0.02, (1.0, 1.0))
    sqc = optimal_fock_squeezing(cold, (1.0, 1.0), s_grid=(1.0, 2.0, 4.0), n_theta=2)
    assert len(sqc["grid"]) > 1 and sqc["result"].converged
    assert sqc["gain"] >= fock_conditioning(cold, (1.0, 1.0)).gain_pnr - 1e-12
