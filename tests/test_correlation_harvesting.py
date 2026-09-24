"""Tests for the Layer 2 pivot: MI / discord / coherence harvesting.

Module under test: ``vacuum.detectors.correlations`` and the candidate
``papers/mi-coherence-harvesting/notebook.py`` (its ``--quick`` configuration
through the sibling's audited code path).

Anchors and identities pinned here (every measured number appears in the
assertion message, repo convention):

- Gaussian discord closed form (Adesso-Datta PRL 105, 030501 (2010),
  Eqs. (3)-(4)) against a brute-force minimisation over pure Gaussian seeds
  (squeezing r, angle phi) on random two-mode states, BOTH directions,
  agreement 1e-8, both branches (finite squeezing / homodyne) exercised;
- the decomposition identity I = J + D to 1e-12, with I from the core's
  symplectic-eigenvalue route and J, D from the invariants;
- product states give zero for every correlation measure (and the
  correlated coherence vanishes on a product of *squeezed* states while the
  local coherence does not);
- the Fock construction against closed forms: single-mode thermal p_n,
  two-mode squeezed vacuum p(n, n) and its C_r = H(p_nn) (pure state), the
  generating-function diagonal on random states, covariance round trip,
  and the cutoff-doubling convergence;
- the two-qubit discord against S. Luo's Bell-diagonal closed form
  (PRA 77, 042303 (2008)) on a Werner state, and ln 2 on a Bell state;
- PKMM Eq. (76) against the 4x4 MI at leading order (the defect scales as
  lambda^4);
- the tomography model against Ren 2022's own reported numbers (spurious
  concurrence on the separable input, the zero fraction on the
  optimal-point state) and the projection's idempotence/physicality;
- the MI communication split: C_comm vanishes at spacelike separation on
  the lattice, is non-zero in causal contact;
- a 6-row (2x2 + 2) smoke of the map with its schema and audit flags.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from vacuum.core import entropy, ground_state_cov, harmonic_chain_K, reduce as reduce_cov
from vacuum.detectors import UDWDetector, switching, wightman_lattice
from vacuum.detectors import correlations as C

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "mi-coherence-harvesting" / "notebook.py"


# --------------------------------------------------------------------------
# Gaussian discord: closed form vs brute force; decomposition identity
# --------------------------------------------------------------------------


def test_gaussian_discord_matches_brute_force_both_directions_both_branches():
    rng = np.random.default_rng(7)
    worst = 0.0
    branches = set()
    for i in range(14):
        V = C.random_two_mode_state(rng)
        gc = C.gaussian_correlations(V)
        branches.update({gc.branch_A, gc.branch_B})
        for measured, closed in (("B", gc.D_B), ("A", gc.D_A)):
            bf = C.brute_force_gaussian_discord(V, (0,), (1,), measured=measured, n_r=25, n_phi=24)
            diff = abs(closed - bf["D"])
            worst = max(worst, diff)
            assert diff < 1e-8, (
                f"state {i} measured={measured}: closed form {closed:.12e} vs brute force "
                f"{bf['D']:.12e} (diff {diff:.3e}, branch {gc.branch_B if measured == 'B' else gc.branch_A})"
            )
    assert {"squeezed", "homodyne"} <= branches, f"branches exercised: {branches}"
    assert worst < 1e-8, f"worst closed-form vs brute-force defect {worst:.3e}"


def test_mutual_information_equals_classical_plus_discord():
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(60):
        V = C.random_two_mode_state(rng)
        gc = C.gaussian_correlations(V)
        for name, val in (("B", gc.J_B + gc.D_B), ("A", gc.J_A + gc.D_A), ("inv", gc.I_invariants)):
            d = abs(gc.I - val)
            worst = max(worst, d)
            assert d < 1e-12, f"I = {gc.I:.15e} vs J+D ({name}) = {val:.15e}, defect {d:.3e}"
    assert worst < 1e-12, f"worst I - (J + D) defect {worst:.3e}"


def test_discord_is_nonnegative_and_bounded_by_mutual_information():
    rng = np.random.default_rng(3)
    for _ in range(40):
        V = C.random_two_mode_state(rng)
        gc = C.gaussian_correlations(V)
        assert 0.0 <= gc.D_B <= gc.I + 1e-12, f"D_B={gc.D_B:.3e} I={gc.I:.3e}"
        assert 0.0 <= gc.D_A <= gc.I + 1e-12, f"D_A={gc.D_A:.3e} I={gc.I:.3e}"


def test_product_states_give_zero_for_every_measure():
    # two uncoupled detector oscillators of different gaps, one thermal
    gaps = (0.7, 1.9)
    V = np.diag([(0.5 + 0.2) / gaps[0], 0.5 / gaps[1], (0.5 + 0.2) * gaps[0], 0.5 * gaps[1]])
    gc = C.gaussian_correlations(V)
    coh = C.gaussian_coherence(V, gaps)
    for name, val in (("I", gc.I), ("D_B", gc.D_B), ("D_A", gc.D_A), ("J_B", gc.J_B),
                      ("J_A", gc.J_A), ("C_r", coh.C_r), ("C_cc", coh.C_cc)):
        assert abs(val) < 1e-10, f"product thermal state: {name} = {val:.3e} (should be 0)"
    assert gc.branch_B == "uncorrelated" and gc.branch_A == "uncorrelated"
    # a product of SQUEEZED states: local coherence yes, correlated coherence no
    # (r = 0.15: the projected exponent's ground state converges as ~tanh(r)^cutoff,
    # so a larger squeezing would need a cutoff beyond the 32 the module caps at)
    r = 0.15
    Vs = np.diag([0.5 * math.exp(-2 * r), 0.5 * math.exp(2 * r), 0.5 * math.exp(2 * r), 0.5 * math.exp(-2 * r)])
    gc = C.gaussian_correlations(Vs)
    coh = C.gaussian_coherence(Vs, None)
    assert abs(gc.I) < 1e-12 and abs(gc.D_B) < 1e-12, f"squeezed product: I={gc.I:.3e} D={gc.D_B:.3e}"
    assert coh.C_r > 0.02, f"local squeezing coherence C_r = {coh.C_r:.3e} should be > 0"
    assert abs(coh.C_cc) < 1e-9, f"correlated coherence of a product state {coh.C_cc:.3e}"
    # qubit side
    prod = np.kron(np.diag([0.9, 0.1]), np.diag([0.8, 0.2]))
    q = C.qubit_correlations(prod)
    for k in ("MI", "D_B", "D_A", "C_r", "C_cc", "E_N", "concurrence"):
        assert abs(q[k]) < 1e-12, f"product qubit pair: {k} = {q[k]:.3e}"


# --------------------------------------------------------------------------
# Fock representation and coherence
# --------------------------------------------------------------------------


def test_fock_thermal_distribution_and_generating_function():
    nbar = 0.3
    nu = nbar + 0.5
    V1 = np.diag([nu, nu])
    q = nbar / (1.0 + nbar)
    p_exact = (1 - q) * q ** np.arange(16)
    p = C.photon_number_distribution(V1, None, 16)
    err = float(np.max(np.abs(p - p_exact)))
    assert err < 1e-9, f"thermal p_n from the Fock construction: max error {err:.3e}"
    p_gf = C.photon_number_distribution_gf(V1, None, 10)
    err_gf = float(np.max(np.abs(p_gf - p_exact[:10])))
    assert err_gf < 1e-12, f"thermal p_n from the generating function: max error {err_gf:.3e}"


def test_two_mode_squeezed_vacuum_coherence_closed_form_and_convergence():
    r = 0.4
    c2, s2 = math.cosh(2 * r) / 2, math.sinh(2 * r) / 2
    V = np.array([[c2, s2, 0, 0], [s2, c2, 0, 0], [0, 0, c2, -s2], [0, 0, -s2, c2]])
    res = C.gaussian_coherence(V, None, cutoff0=4, tol=1e-10)
    t = math.tanh(r)
    p_nn = (1 - t * t) * t ** (2 * np.arange(200))  # analytic, far past any cutoff
    H = float(-np.sum(p_nn * np.log(p_nn)))
    assert res.converged, (
        f"not converged: movement {res.movement:.3e}, gf defect {res.gf_defect:.3e}, "
        f"tail {res.tail_weight:.3e}")
    assert res.criterion in ("movement", "generating_function"), res.criterion
    assert res.gf_defect < 1e-11, f"generating-function defect at acceptance {res.gf_defect:.3e}"
    assert abs(res.C_r - H) < 1e-9, f"TMSV C_r = {res.C_r:.12e} vs H(p_nn) = {H:.12e}"
    assert abs(res.S_exact) < 1e-12 and abs(res.S_trunc) < 1e-9, (
        f"pure state entropies: exact {res.S_exact:.3e}, truncated {res.S_trunc:.3e}")
    off = float(res.p_joint.sum() - np.trace(res.p_joint))
    assert off < 1e-12, f"TMSV off-diagonal photon-number mass {off:.3e}"
    # convergence history: both error estimates shrink across doublings
    moves = [h["movement"] for h in res.history if h["movement"] == h["movement"]]
    gfs = [h["gf_defect"] for h in res.history]
    assert moves[-1] < moves[0] and gfs[-1] < gfs[0], f"movements {moves}, gf defects {gfs}"
    # the accepted level's diagonal IS the closed-form one, to the gf tolerance
    p_exact = np.diag((1 - t * t) * t ** (2 * np.arange(res.cutoff)))
    assert float(np.max(np.abs(res.p_joint - p_exact))) < 1e-11, "TMSV diagonal vs closed form"
    # C_cc = C_r - C_A - C_B with the marginals thermal (diagonal): C_cc = C_r
    assert abs(res.C_cc - res.C_r) < 1e-9, f"TMSV C_cc {res.C_cc:.3e} vs C_r {res.C_r:.3e}"


def test_generating_function_referees_fock_diagonal_on_random_states():
    """The closed-form generating function is the referee; the Fock diagonal converges to it.

    The truncated-exponent construction converges as ~tanh(r)^cutoff (module
    docstring), so the check is convergence under a cutoff doubling, not a
    fixed tolerance at one cutoff: the hottest of the four states here
    (<n_d> = 0.78) is 2.8e-6 off at cutoff 24 and 4.9e-8 at 32.
    """
    rng = np.random.default_rng(5)
    for i in range(3):
        V = C.random_two_mode_state(rng, strength=0.3)
        pg = C.photon_number_distribution_gf(V, None, 6, radius=0.5)
        errs = []
        for cutoff in (12, 24):
            rho = C.fock_density_matrix(V, None, cutoff)
            p = np.real(np.diag(rho)).reshape(cutoff, cutoff)
            errs.append(float(np.max(np.abs(pg - p[:6, :6]))))
            assert abs(p.sum() - 1.0) < 1e-10, f"state {i} cutoff {cutoff}: trace {p.sum():.12f}"
        assert errs[1] < 1e-5, f"state {i}: gf vs Fock diagonal at cutoff 24, max error {errs[1]:.3e}"
        assert errs[1] < 0.05 * errs[0], (
            f"state {i}: cutoff 12 -> 24 error {errs[0]:.3e} -> {errs[1]:.3e} (ratio "
            f"{errs[1] / errs[0]:.3e}, expect < 5e-2)")
        # Covariance round trip and truncated entropy, at cutoff 32: the moments
        # are read back with TRUNCATED quadratures, whose x^2 + p^2 under-weights
        # the top retained level, so these defects track the tail population and
        # need a higher cutoff than the diagonal does (1.1e-4 at 24, 2.6e-6 at 32
        # on the <n_d> = 0.78 state).
        rho = C.fock_density_matrix(V, None, 32)
        R = C.fock_quadratures(32, 2)
        Vrt = np.array([[0.5 * np.trace(rho @ (R[a] @ R[b] + R[b] @ R[a])).real for b in range(4)]
                        for a in range(4)])
        err_cov = float(np.max(np.abs(Vrt - V)))
        assert err_cov < 1e-5, f"state {i}: covariance round trip error {err_cov:.3e}"
        dS = abs(C._trunc_entropy(rho) - entropy(V))
        assert dS < 1e-5, f"state {i}: truncated entropy defect {dS:.3e}"


def test_coherence_basis_follows_the_detector_gaps():
    """The ground state of H_d = (p^2 + Omega^2 x^2)/2 is diagonal in ITS Fock basis."""
    gaps = (1.3, 1.5)  # mild enough that the deliberately wrong basis (a squeezed state) still converges
    V = ground_state_cov(np.diag([gaps[0] ** 2, gaps[1] ** 2]))
    res = C.gaussian_coherence(V, gaps)
    assert abs(res.C_r) < 1e-10, f"detector ground state C_r in its own basis = {res.C_r:.3e}"
    res_wrong = C.gaussian_coherence(V, None, tol=1e-8)  # the unit-frequency basis: squeezed -> coherent
    assert res_wrong.C_r > 0.01, f"C_r in the wrong basis = {res_wrong.C_r:.3e} (should be large)"


# --------------------------------------------------------------------------
# Qubit pair
# --------------------------------------------------------------------------


def test_qubit_discord_luo_werner_anchor_and_bell_state():
    z = 0.6
    psi = np.array([0, 1, -1, 0]) / math.sqrt(2)
    rho = z * np.outer(psi, psi) + (1 - z) * np.eye(4) / 4
    # S. Luo, PRA 77, 042303 (2008): for the Werner state the optimal projective
    # measurement is any direction; D = ln 2 - S(rho) + h((1+z)/2), h binary entropy (nats).
    h = lambda x: -x * math.log(x) - (1 - x) * math.log(1 - x)
    S = -((1 + 3 * z) / 4 * math.log((1 + 3 * z) / 4) + 3 * (1 - z) / 4 * math.log((1 - z) / 4))
    D_luo = math.log(2) - S + h((1 + z) / 2)
    d = C.qubit_discord(rho, "B")
    assert abs(d["D"] - D_luo) < 1e-12, f"Werner z={z}: D = {d['D']:.15e} vs Luo {D_luo:.15e}"
    dA = C.qubit_discord(rho, "A")
    assert abs(dA["D"] - D_luo) < 1e-12, f"Werner (A measured): {dA['D']:.15e}"
    bell = np.outer(psi, psi)
    q = C.qubit_correlations(bell)
    assert abs(q["D_B"] - math.log(2)) < 1e-12, f"Bell discord {q['D_B']:.15e} vs ln 2"
    assert abs(q["MI"] - 2 * math.log(2)) < 1e-12, f"Bell MI {q['MI']:.15e} vs 2 ln 2"
    assert abs(q["E_N"] - math.log(2)) < 1e-12 and abs(q["concurrence"] - 1.0) < 1e-12, (
        f"Bell E_N {q['E_N']:.3e}, concurrence {q['concurrence']:.3e}")


def test_pkmm_mutual_information_matches_4x4_at_leading_order():
    """PKMM Eq. (76) is the O(lambda^2) MI: the defect to the 4x4 MI falls as lambda^4."""
    def state(scale):
        P, c, M = 4e-3 * scale, 1.6e-3 * scale, 1.2e-3 * scale
        rho = np.zeros((4, 4), dtype=complex)
        rho[0, 0] = 1 - 2 * P
        rho[1, 1] = rho[2, 2] = P
        rho[1, 2] = rho[2, 1] = c
        rho[0, 3] = rho[3, 0] = M
        return P, c, rho
    defects = []
    for scale in (1.0, 0.5, 0.25):
        P, c, rho = state(scale)
        mi_pkmm = C.pkmm_mutual_information(P, P, c)
        mi_4 = C.qubit_mutual_information(rho)
        defects.append(abs(mi_pkmm - mi_4))
        assert abs(mi_pkmm - mi_4) < 0.05 * mi_4, f"scale {scale}: Eq. (76) {mi_pkmm:.6e} vs 4x4 {mi_4:.6e}"
    ratio = defects[0] / defects[2]
    assert ratio > 8.0, f"defect ratio over a factor-4 reduction of lambda^2: {ratio:.2f} (expect ~16)"


def test_qubit_pair_from_udw_reorders_pkmm_basis():
    r = np.zeros((4, 4), dtype=complex)
    r[0, 0], r[1, 1], r[2, 2] = 0.9, 0.06, 0.04  # gg, eg, ge in PKMM order
    r[1, 2], r[2, 1] = 0.01j, -0.01j            # C at (eg, ge)
    r[0, 3], r[3, 0] = 0.02, 0.02               # M at (gg, ee)
    s = C.qubit_pair_from_udw(r)
    assert s[1, 1] == 0.04 and s[2, 2] == 0.06, "A x B order puts |ge> at index 1 and |eg> at 2"
    assert s[0, 3] == 0.02 and abs(s[1, 2] + 0.01j) < 1e-15, "M stays at (gg, ee); C moves with its levels"


# --------------------------------------------------------------------------
# Resolvability model
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def precision():
    return C.tomography_precision_from_experiment()


def test_tomography_precision_from_ren_2022(precision):
    assert precision.n_shots == 1e7 and abs(precision.readout_fidelity - 0.8) < 1e-12
    expect_1 = 1.0 / (0.6 * math.sqrt(1e7))
    assert abs(precision.sigma_single - expect_1) < 1e-15, f"sigma_1 = {precision.sigma_single:.6e}"
    assert abs(precision.sigma_double - expect_1 / 0.6) < 1e-15, f"sigma_2 = {precision.sigma_double:.6e}"


def test_projection_is_physical_and_idempotent():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(6, 4, 4)) + 1j * rng.normal(size=(6, 4, 4))
    H = 0.5 * (A + np.swapaxes(A, 1, 2).conj())
    H = H / np.trace(H, axis1=1, axis2=2)[:, None, None].real
    P = C.project_to_physical(H)
    w = np.linalg.eigvalsh(P)
    assert np.all(w >= -1e-12), f"min eigenvalue after projection {w.min():.3e}"
    tr = np.trace(P, axis1=1, axis2=2).real
    assert np.allclose(tr, 1.0, atol=1e-12), f"traces {tr}"
    P2 = C.project_to_physical(P)
    assert np.max(np.abs(P2 - P)) < 1e-12, "projection is not idempotent"


def test_model_calibration_against_ren_2022(precision):
    """Spurious concurrence on |eg><eg| ~ the plot read; the optimal-point zero fraction ~ 'nearly half'."""
    eg = np.zeros((4, 4), dtype=complex)
    eg[2, 2] = 1.0
    mc = C.monte_carlo_floors(eg, precision, n_samples=500, seed=1)
    spurious = mc["measures"]["concurrence"]["reference_max"]
    ratio = spurious / 1.2e-3  # Ren Fig. 4.5(b) axis extent (plot read, upper bound)
    assert 0.5 < ratio < 4.0, f"spurious concurrence max {spurious:.3e} vs plot read 1.2e-3 (ratio {ratio:.2f})"
    e_floor = mc["measures"]["E_N"]["floor"]
    assert 0.3 < e_floor / 1e-3 < 3.0, f"model E_N floor on |eg><eg| {e_floor:.3e} vs derived 1e-3 nats"
    # the printed optimal-point matrix (populations 0.63/0.37/0/6e-7, |rho_23| = 5e-3), projected
    rho = np.diag([0.63, 0.37, 0.0, 6e-7]).astype(complex)
    rho[1, 2], rho[2, 1] = 0.005j, -0.005j
    rho = C.project_to_physical(rho)
    conc = C.concurrence(rho)
    # the projection cannot keep a 6e-7 population next to a 7e-5 negative
    # eigenvalue, so sqrt(rho_11 rho_44) -> 0 and C -> 2 |rho_23| = 0.0100
    assert 0.008 < conc < 0.0105, f"concurrence of the projected printed state {conc:.4f} vs stated 0.0087"
    mcx = C.monte_carlo_floors(rho, precision, n_samples=500, seed=2)
    zf = mcx["measures"]["concurrence"]["target_zero_fraction"]
    assert 0.25 < zf < 0.75, f"reconstructed-concurrence zero fraction {zf:.3f} vs 'nearly half'"


def test_analytic_floor_companion_limits():
    delta = C.coherence_precision_from_negativity_floor(1e-3)
    assert delta == 5e-4
    # zero coherence: the floor state IS the product of the marginals
    zero = C.floor_state(3e-3, 7e-3, 0.0)
    assert np.max(np.abs(zero - np.kron(np.diag([1 - 3e-3, 3e-3]), np.diag([1 - 7e-3, 7e-3])))) < 1e-15
    q0 = C.qubit_correlations(zero, refine=False)
    assert all(abs(q0[k]) < 1e-12 for k in ("MI", "D_B", "C_cc")), q0
    # P << delta: capped by positivity, MI -> 2 P ln 2 (+ O(P^2))
    P = 1e-4
    small = C.correlation_floors(P, P, delta, refine=False)
    assert abs(small["MI"] / (2 * P * math.log(2)) - 1.0) < 2e-3, f"capped MI floor {small['MI']:.6e}"
    # P >> delta: quadratic response, MI ~ delta^2 / P
    P = 1e-2
    large = C.correlation_floors(P, P, delta, refine=False)
    assert abs(large["MI"] / (delta**2 / P) - 1.0) < 0.05, (
        f"quadratic-response MI floor {large['MI']:.3e} vs delta^2/P = {delta**2 / P:.3e}")
    assert 0.0 < large["D_B"] <= large["MI"] and large["C_cc"] > 0.0


# --------------------------------------------------------------------------
# MI communication split
# --------------------------------------------------------------------------


def test_mutual_information_split_vanishes_at_spacelike_separation():
    N, m = 34, 0.2
    K = harmonic_chain_K(N, m, bc="dirichlet")
    kernel = wightman_lattice(K)
    T_half = 3.0  # cos2 window [0, 6]: light cone 6 sites
    chi = switching("cos2", T_half, t0=T_half)
    def det(site):
        F = np.zeros(N)
        F[site] = 1.0
        return UDWDetector(chi, F)
    quad = {"rtol": 1e-8, "atol": 1e-16, "max_doublings": 12}
    far = C.mutual_information_split(kernel, det(8), det(22), 0.1, 1.0, **quad)   # 14 sites > 6
    near = C.mutual_information_split(kernel, det(8), det(10), 0.1, 1.0, **quad)  # 2 sites < 6
    assert far["comm_fraction_C"] < 1e-6, f"spacelike |C_comm|/|C| = {far['comm_fraction_C']:.3e}"
    assert far["vac_part_physical"], f"spacelike |C_vac|/sqrt(P_A P_B) = {far['C_vac_over_population']:.3e}"
    assert abs(far["MI_comm"]) <= 1e-9 * max(far["MI_pkmm"], 1e-30), (
        f"spacelike MI_comm = {far['MI_comm']:.3e} of MI {far['MI_pkmm']:.3e}")
    assert near["comm_fraction_C"] > 0.05, f"causal-contact |C_comm|/|C| = {near['comm_fraction_C']:.3e}"
    assert near["MI_pkmm"] > 0.0 and near["P_A"] > 0.0
    # the full C always respects the population bound (W is a positive kernel)
    for s in (far, near):
        assert abs(s["C"]) <= math.sqrt(s["P_A"] * s["P_B"]) * (1 + 1e-9), (
            f"|C| = {abs(s['C']):.3e} > sqrt(P_A P_B) = {math.sqrt(s['P_A'] * s['P_B']):.3e}")
    if not near["vac_part_physical"]:
        assert math.isnan(near["MI_vac"]) and near["C_vac_over_population"] > 1.0


# --------------------------------------------------------------------------
# The map: 6-row smoke through the sibling's audited code path
# --------------------------------------------------------------------------


def _load_notebook():
    if not NOTEBOOK_PATH.exists():  # pragma: no cover
        pytest.skip(f"notebook not found at {NOTEBOOK_PATH}")
    spec = importlib.util.spec_from_file_location("pivot_mi_coherence_notebook", NOTEBOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def quick_map():
    nb_pivot = _load_notebook()
    snapshot = nb_pivot.snapshot_sibling()
    nb = nb_pivot.load_sibling(snapshot)
    inputs = nb_pivot.load_sibling_inputs(nb)
    prec = C.tomography_precision_from_experiment(nb_pivot.REN)
    specs = nb_pivot.quick_specs(nb, inputs)
    assert len(specs) == 6, f"quick configuration must be 2x2 + 2 rows, got {len(specs)}"
    rows = nb_pivot.run_specs(specs, inputs, nb, prec, verbose=False, mc_samples=100,
                              sibling_path=snapshot)
    summary = nb_pivot.summarize(nb, rows, inputs, prec, wall_clock_s=0.0, mc_samples=100,
                                 sibling_path=snapshot)
    return nb_pivot, nb, inputs, rows, summary, snapshot


def test_map_smoke_schema_audits_and_measures(quick_map):
    nb_pivot, nb, inputs, rows, summary, snapshot = quick_map
    surface = [r for r in rows if r["sweep"] == nb.MAIN_SURFACE]
    assert len(surface) == 4, "the 2 x 2 main surface"
    for r in rows:
        missing = [c for c in nb_pivot.ROW_COLUMNS if c not in r]
        assert not missing, f"{r['label']}: missing columns {missing}"
        assert r["all_audits_passed"] and r["ledger_passed"] and r["converged"], r["label"]
        assert r["coherence_converged"], f"{r['label']}: coherence cutoff not converged"
        assert abs(r["MI"] - r["MI_invariants"]) < 1e-12, (
            f"{r['label']}: MI {r['MI']:.3e} vs invariants {r['MI_invariants']:.3e}")
        assert 0.0 <= r["D_B"] <= r["MI"] + 1e-12 and 0.0 <= r["D_A"] <= r["MI"] + 1e-12, r["label"]
        assert r["C_cc"] >= -1e-9 and r["C_r"] >= r["C_cc"] - 1e-12, r["label"]
        assert 0.99 < r["proxy_weight"] <= 1.0 + 1e-12, f"{r['label']}: proxy weight {r['proxy_weight']}"
        for m in nb_pivot.MEASURES:
            assert r[f"{m}_resolvable_derived"] == (r[m] > inputs.resolution.derived), r["label"]
        assert r["MI_floor_mc"] > 0.0 and r["MI_resolvable_mc"] == (r["MI"] > r["MI_floor_mc"]), r["label"]
        assert r["spacelike_window"] == (r["separation_sites"] > r["window_lat"])
    split = [r for r in rows if r["sweep"] == "split"]
    assert len(split) == 1 and split[0]["has_perturbative_row"], "one static-gap split row"
    s = split[0]
    assert s["comm_fraction_C"] >= 0.0 and s["MI_pkmm"] > 0.0, f"split row: {s['comm_fraction_C']}, {s['MI_pkmm']}"
    assert abs(s["MI_pkmm"] - s["pert_MI"]) < 0.05 * s["pert_MI"] + 1e-12, (
        f"Eq. (76) {s['MI_pkmm']:.3e} vs 4x4 MI {s['pert_MI']:.3e} on the split row")
    assert summary["all_audits_passed"] and summary["all_coherence_converged"]
    assert set(summary["go_no_go"][nb.MAIN_SURFACE]["contours"]) == set(nb_pivot.MEASURES)
    assert "device_box" in summary["pivot"] and "calibration" in summary
    # the run is pinned to a byte-for-byte snapshot of the sibling notebook, and
    # that snapshot's sha256 is what the archive records (a concurrently edited
    # tree cannot split an archive across two code states)
    assert Path(snapshot).read_bytes() == nb_pivot.SIBLING_PATH.read_bytes()
    assert summary["build"]["sibling_sha256"] == nb_pivot.sibling_sha256(snapshot)
    assert summary["build"]["sibling_snapshot"] == str(snapshot)
    # the derived coupling is the one the map runs at
    assert inputs.point.coupling_mode == "derived" and inputs.point.coupling_sigma > 0.0
