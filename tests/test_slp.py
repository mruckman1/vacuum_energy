"""M3.3 — strong local passivity: analytic rung, SDP rung, LOCC layer.

Anchors (docs/PLAN_LAYERS_2_3.md, "Test matrix"):
    | FFH <-> SDP overlap | verdict agreement, margin 1e-9 |

plus the milestone's own requirements: the LOCC layer reuses
:mod:`vacuum.qet.hotta` to show classical communication reopening extraction
on states the local certificate forbids, and a ground state is SLP everywhere.

Every number that the milestone claims from a paper is checked against an
independent computation in this file:

  * FFH Eq. (16) spectrum  <- direct eigh of FFH Eq. (15),
  * FFH Eq. (23) Gibbs differences <- Boltzmann weights on Eq. (16),
  * FFH Eqs. (20)-(21) local energy Omega_o <- the Kraus search (core-only)
    *and* the Choi SDP (cvxpy), neither of which knows the closed form,
  * ASRSM Eq. (3) cost operator <- brute-force Tr[H (E (x) I) rho] for random
    CPTP maps,
  * ASRSM Thm. 1 certificate <- the SDP verdict,
  * ASRSM Sec. A.1 robustness bound <- the achieved extraction.

cvxpy-gated tests use ``pytest.importorskip`` so a default install stays green.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vacuum.qet import chains, hotta, slp

# --------------------------------------------------------------------------
# shared fixtures and helpers
# --------------------------------------------------------------------------

# The FFH Eq. (15) Gibbs family: coupling grid spans the SLP transition, and
# every temperature is deliberately away from the critical one so a verdict
# comparison is not a coin flip on solver noise (checked in
# test_ffh_grid_is_verdict_unambiguous).
FFH_KAPPAS = (0.3, 0.7, 1.0, 1.8, 2.5, 4.0)
FFH_TEMPS = (0.15, 0.25, 0.4, 0.7, 1.2, 2.0, 3.5, 6.0)

# The milestone's stated margin for the FFH <-> SDP overlap row.
OVERLAP_MARGIN = 1e-9


@pytest.fixture(scope="module")
def ffh_grid():
    """(kappa, T, H, rho, Omega_o) for the whole FFH Gibbs grid, built once."""
    rows = []
    for kappa in FFH_KAPPAS:
        H = slp.ffh_pair_hamiltonian(kappa)
        for T in FFH_TEMPS:
            p = slp.ffh_pair_gibbs_populations(kappa, T)
            rows.append((kappa, T, H, slp.ffh_pair_eigenmixture(kappa, p),
                         slp.ffh_pair_local_energy(kappa, p)))
    return rows


@pytest.fixture(scope="module")
def ffh_sdp(ffh_grid):
    """The SDP rung run once over the whole FFH grid (single-qubit region)."""
    pytest.importorskip("cvxpy")
    return [slp.extractable_energy_sdp(H, rho, 0) for _, _, H, rho, _ in ffh_grid]


def _rand_herm(d, rng):
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    return (A + A.conj().T) / 2.0


def _rand_rho(d, rng):
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    R = A @ A.conj().T
    return R / float(np.real(np.trace(R)))


def _rand_isometry(rows, cols, rng):
    A = rng.standard_normal((rows, cols)) + 1j * rng.standard_normal((rows, cols))
    U, _, Vh = np.linalg.svd(A, full_matrices=False)
    return U @ Vh


def _rand_cptp_kraus(dA, r, rng):
    """r Kraus operators on a d_A-dimensional region, sum K^dag K = I exactly."""
    T = _rand_isometry(r * dA, dA, rng)
    return [T.reshape(r, dA, dA)[mu] for mu in range(r)]


def _tfim_dense(L, g):
    return np.asarray(chains.tfim_hamiltonian(L, g).toarray(), dtype=complex)


# --------------------------------------------------------------------------
# region plumbing
# --------------------------------------------------------------------------


def test_embed_region_operator_matches_kron_for_leading_region():
    """A region of the leading sites embeds as the plain Kronecker product."""
    rng = np.random.default_rng(0)
    K = _rand_herm(4, rng)
    got = slp.embed_region_operator(K, (0, 1), 4)
    want = np.kron(K, np.eye(4))
    dev = float(np.max(np.abs(got - want)))
    assert dev < 1e-14, f"leading-region embedding deviates by {dev:.3e}"


def test_embed_region_operator_scattered_sites():
    """A scattered region embeds site-wise: single-site ops factor exactly."""
    rng = np.random.default_rng(1)
    a = _rand_herm(2, rng)
    b = _rand_herm(2, rng)
    # region (0, 2) of 3 qubits, K = a (x) b acting on sites 0 and 2
    got = slp.embed_region_operator(np.kron(a, b), (0, 2), 3)
    want = np.kron(np.kron(a, np.eye(2)), b)
    dev = float(np.max(np.abs(got - want)))
    assert dev < 1e-14, f"scattered-region embedding deviates by {dev:.3e}"


def test_apply_local_channel_is_trace_preserving_and_pure():
    rng = np.random.default_rng(2)
    rho = _rand_rho(8, rng)
    rho_copy = rho.copy()
    kraus = _rand_cptp_kraus(2, 4, rng)
    out = slp.apply_local_channel(rho, kraus, (1,))
    assert abs(float(np.real(np.trace(out))) - 1.0) < 1e-13
    assert float(np.max(np.abs(rho - rho_copy))) == 0.0, "input state was mutated"


@pytest.mark.parametrize("n,region", [(2, (0,)), (3, (1,)), (3, (0, 2)), (4, (1, 2))])
def test_choi_cost_matrix_reproduces_post_map_energy(n, region):
    """ASRSM Eq. (3): Tr[H (E (x) I)(rho)] = Tr[J C] for every CPTP E on A."""
    rng = np.random.default_rng(100 + n)
    d = 2**n
    dA = 2 ** len(region)
    H = _rand_herm(d, rng)
    rho = _rand_rho(d, rng)
    C = slp.choi_cost_matrix(H, rho, region)
    worst = 0.0
    for _ in range(4):
        kraus = _rand_cptp_kraus(dA, dA * dA, rng)
        direct = slp.state_energy(slp.apply_local_channel(rho, kraus, region), H)
        # J = sum_mu k_mu k_mu^dag with k_mu[(i, o)] = (K_mu)_{o i}
        J = sum(np.outer(K.T.reshape(dA * dA), K.T.reshape(dA * dA).conj())
                for K in kraus)
        via_choi = float(np.real(np.trace(J @ C)))
        worst = max(worst, abs(direct - via_choi))
    assert worst < 1e-12, f"Eq. (3) identity fails by {worst:.3e} on n={n} {region}"


def test_choi_to_kraus_round_trip():
    """choi_to_kraus inverts the Choi construction on an exact CPTP map."""
    rng = np.random.default_rng(3)
    dA = 4
    kraus = _rand_cptp_kraus(dA, 5, rng)
    J = sum(np.outer(K.T.reshape(dA * dA), K.T.reshape(dA * dA).conj())
            for K in kraus)
    back = slp.choi_to_kraus(J)
    completeness = sum(K.conj().T @ K for K in back)
    dev_c = float(np.max(np.abs(completeness - np.eye(dA))))
    rho = _rand_rho(2**3, rng)
    a = slp.apply_local_channel(rho, kraus, (1, 2))
    b = slp.apply_local_channel(rho, back, (1, 2))
    dev_a = float(np.max(np.abs(a - b)))
    assert dev_c < 1e-12, f"recovered Kraus set not CPTP: {dev_c:.3e}"
    assert dev_a < 1e-12, f"recovered channel differs by {dev_a:.3e}"


# --------------------------------------------------------------------------
# the analytic certificate: structure identities
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n,region", [(2, (0,)), (3, (1,)), (4, (0, 2))])
def test_certificate_lambda_identities(n, region):
    """Lambda^T = Tr_B[H rho], Tr Lambda = Tr[rho H], herm dev = |Tr_B[H,rho]|.

    These are the two structural facts the certificate's docstring leans on:
    the Hermiticity half of ASRSM Thm. 1 is the vanishing of the reduced
    commutator, and Tr[Lambda] is real (which is what makes the robustness
    bound unconditional).
    """
    rng = np.random.default_rng(200 + n)
    d = 2**n
    H = _rand_herm(d, rng)
    rho = _rand_rho(d, rng)
    cert = slp.slp_certificate(H, rho, region)
    blocks = slp._as_region_blocks(H @ rho, region, n)
    tr_b_hrho = np.einsum("abcb->ac", blocks)
    dev_lam = float(np.max(np.abs(cert["Lambda"].T - tr_b_hrho)))
    dev_tr = abs(complex(np.trace(cert["Lambda"])) - slp.state_energy(rho, H))
    comm = slp.reduced_commutator(H, rho, region)
    dev_comm = abs(cert["herm_dev"] - float(np.max(np.abs(comm))))
    assert dev_lam < 1e-13, f"Lambda^T != Tr_B[H rho] by {dev_lam:.3e}"
    assert dev_tr < 1e-13, f"Tr Lambda != Tr[rho H] by {dev_tr:.3e}"
    assert dev_comm < 1e-13, f"herm_dev != max|Tr_B[H,rho]| by {dev_comm:.3e}"


def test_reduced_commutator_vanishes_for_commuting_states():
    """[H, rho] = 0 (Gibbs, eigenmixtures) kills the Hermiticity half exactly."""
    H = slp.ffh_pair_hamiltonian(1.3)
    for rho in (slp.gibbs_state(H, 0.8),
                slp.ffh_pair_eigenmixture(1.3, [0.5, 0.2, 0.2, 0.1])):
        drift = float(np.max(np.abs(slp.reduced_commutator(H, rho, 0))))
        assert drift < 1e-14, f"reduced commutator {drift:.3e} on a commuting state"


def test_certificate_does_not_mutate_inputs():
    rng = np.random.default_rng(4)
    H = _rand_herm(8, rng)
    rho = _rand_rho(8, rng)
    H0, rho0 = H.copy(), rho.copy()
    slp.slp_certificate(H, rho, (1,))
    slp.choi_cost_matrix(H, rho, (0, 2))
    slp.extractable_energy_kraus(H, rho, (1,), restarts=1)
    assert float(np.max(np.abs(H - H0))) == 0.0
    assert float(np.max(np.abs(rho - rho0))) == 0.0


# --------------------------------------------------------------------------
# FFH model transcriptions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kappa", [0.2, 1.0, 3.0])
def test_ffh_eq16_spectrum(kappa):
    """FFH Eq. (16): the Eq. (15) pair has energies (-m, -kappa, kappa, m)."""
    w = np.linalg.eigvalsh(slp.ffh_pair_hamiltonian(kappa))
    dev = float(np.max(np.abs(w - slp.ffh_pair_eigenenergies(kappa))))
    assert dev < 1e-13, f"Eq. (16) spectrum off by {dev:.3e} at kappa={kappa}"


@pytest.mark.parametrize("kappa,T", [(0.4, 0.3), (1.0, 1.0), (2.5, 4.0)])
def test_ffh_eq23_gibbs_differences(kappa, T):
    """FFH Eq. (23): delta_0 = (2/Z) sinh(m/T), delta_1 = (2/Z) sinh(kappa/T)."""
    p = slp.ffh_pair_gibbs_populations(kappa, T)
    m = float(np.hypot(kappa, 2.0))
    Z = 2.0 * (np.cosh(kappa / T) + np.cosh(m / T))
    d0_dev = abs((p[0] - p[3]) - 2.0 * np.sinh(m / T) / Z)
    d1_dev = abs((p[1] - p[2]) - 2.0 * np.sinh(kappa / T) / Z)
    assert d0_dev < 1e-13, f"Eq. (23) delta_0 off by {d0_dev:.3e}"
    assert d1_dev < 1e-13, f"Eq. (23) delta_1 off by {d1_dev:.3e}"


def test_ffh_maximally_mixed_local_energy_is_one():
    """Sanity on Eq. (21): eta = xi = 0 gives Omega_o = 1, the reset value.

    Resetting one spin of I/4 to |sz = -1> against the sz (x) I term lowers
    the energy by exactly 1, and no operation does better.
    """
    for kappa in (0.5, 2.0):
        omega = slp.ffh_pair_local_energy(kappa, [0.25] * 4)
        assert abs(omega - 1.0) < 1e-14, f"Omega_o(I/4) = {omega!r}, expected 1"


def test_ffh_local_energy_matches_kraus_search(ffh_grid):
    """FFH Eqs. (20)-(21) against a search that knows no closed form (core-only).

    The Kraus search is a restart-certified *lower* bound on Omega_o, so
    agreement in both directions pins the transcription.
    """
    worst = 0.0
    worst_at = None
    for kappa, T, H, rho, omega in ffh_grid[::5]:
        found = slp.extractable_energy_kraus(H, rho, 0, restarts=4)["extractable"]
        if abs(found - omega) > worst:
            worst, worst_at = abs(found - omega), (kappa, T)
    assert worst < 1e-9, (
        f"FFH Eq. (21) vs Kraus search differ by {worst:.3e} at "
        f"(kappa, T) = {worst_at}"
    )


def test_ffh_grid_is_verdict_unambiguous(ffh_grid):
    """The comparison grid never sits on the transition: |Omega_o| is 0 or >1e-5.

    Without this, "verdict agreement to margin 1e-9" would be a statement
    about solver noise rather than about physics.
    """
    smallest_nonzero = min((om for _, _, _, _, om in ffh_grid if om > 1e-12),
                           default=np.inf)
    largest_zero = max((om for _, _, _, _, om in ffh_grid if om <= 1e-12),
                       default=0.0)
    n_slp = sum(1 for _, _, _, _, om in ffh_grid if om <= 1e-12)
    assert largest_zero < 1e-12, f"'SLP' points reach Omega_o = {largest_zero:.3e}"
    assert smallest_nonzero > 1e-5, (
        f"grid has a marginal point, Omega_o = {smallest_nonzero:.3e}"
    )
    assert 0 < n_slp < len(ffh_grid), (
        f"grid must straddle the transition; {n_slp}/{len(ffh_grid)} are SLP"
    )


def test_certificate_verdict_matches_ffh_closed_form(ffh_grid):
    """ASRSM Thm. 1 verdict == (FFH Omega_o == 0) across the grid, both regions.

    Eq. (15) is swap-symmetric, so the two single-qubit regions must agree.
    """
    for kappa, T, H, rho, omega in ffh_grid:
        want = omega <= 1e-12
        for region in (0, 1):
            cert = slp.slp_certificate(H, rho, region)
            assert cert["slp"] == want, (
                f"kappa={kappa} T={T} region={region}: certificate says "
                f"slp={cert['slp']} (lambda_min={cert['lambda_min']:.3e}), "
                f"FFH Omega_o={omega:.3e}"
            )
            assert omega <= cert["extractable_bound"] + 1e-9, (
                f"ASRSM robustness bound {cert['extractable_bound']:.6e} below "
                f"the true Omega_o={omega:.6e} at kappa={kappa}, T={T}"
            )


def test_xxx_gibbs_states_are_slp_at_every_temperature():
    """FFH Eq. (33): the Heisenberg XXX pair is SLP at all T (their Eq. (34))."""
    H = slp.xxx_pair_hamiltonian()
    energies, vectors = slp.xxx_pair_eigenbasis()
    dev = float(np.max(np.abs(H @ vectors - vectors * energies)))
    assert dev < 1e-13, f"stated XXX eigenbasis is not one, residual {dev:.3e}"
    worst = 0.0
    for T in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 50.0, 1e3):
        cert = slp.slp_certificate(H, slp.gibbs_state(H, T), 0)
        worst = max(worst, cert["extractable_bound"])
        assert cert["slp"], f"XXX Gibbs at T={T} judged non-SLP: {cert}"
    assert worst < 1e-12, f"XXX extractable bound reaches {worst:.3e}"


def test_xxx_eigenmixtures_of_the_degenerate_level():
    """Mixing only the triplet level is a valid state and stays SLP-checkable."""
    H = slp.xxx_pair_hamiltonian()
    rho = slp.xxx_pair_eigenmixture([0.0, 1 / 3, 1 / 3, 1 / 3])
    assert abs(float(np.real(np.trace(rho))) - 1.0) < 1e-14
    assert float(np.min(np.linalg.eigvalsh(rho))) > -1e-14
    # the triplet-only mixture is far from the ground state: extraction is open
    cert = slp.slp_certificate(H, rho, 0)
    assert not cert["slp"], "the pure-triplet mixture should not be SLP"
    assert cert["extractable_bound"] > 0.0


def test_slp_critical_temperature_brackets_the_ffh_transition():
    """FFH: Gibbs states are SL passive only at or below a critical temperature."""
    for kappa in (0.5, 1.0, 2.0, 4.0):
        H = slp.ffh_pair_hamiltonian(kappa)
        res = slp.slp_critical_temperature(H, 0)
        t_c = res["T_c"]
        assert 0.0 < t_c < np.inf, f"no transition found at kappa={kappa}: {res}"
        below = slp.ffh_pair_local_energy(
            kappa, slp.ffh_pair_gibbs_populations(kappa, t_c * 0.98))
        above = slp.ffh_pair_local_energy(
            kappa, slp.ffh_pair_gibbs_populations(kappa, t_c * 1.02))
        assert below <= 1e-12, (
            f"kappa={kappa}: FFH Omega_o = {below:.3e} just below T_c={t_c:.6f}"
        )
        assert above > 1e-9, (
            f"kappa={kappa}: FFH Omega_o = {above:.3e} just above T_c={t_c:.6f}"
        )


def test_slp_critical_temperature_infinite_for_xxx():
    res = slp.slp_critical_temperature(slp.xxx_pair_hamiltonian(), 0)
    assert res["T_c"] == np.inf, f"XXX should be SLP at every T, got {res}"


# --------------------------------------------------------------------------
# a ground state is SLP everywhere (sanity)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("L", [3, 4, 5])
def test_tfim_ground_state_is_slp_on_every_region(L):
    """The global ground state minimizes the energy, so nothing local lowers it.

    Checked on every single-site and every nearest-neighbour two-site region
    of the critical transverse-field Ising chain (M3.2's ground states).
    """
    H = _tfim_dense(L, 1.0)
    _, psi = chains.tfim_ground_state(L, 1.0)
    rho = np.outer(psi, psi.conj()).astype(complex)
    regions = [(q,) for q in range(L)] + [(q, q + 1) for q in range(L - 1)]
    worst = 0.0
    for region in regions:
        cert = slp.slp_certificate(H, rho, region)
        worst = max(worst, cert["extractable_bound"], cert["herm_dev"])
        assert cert["slp"], f"L={L} region={region} ground state judged non-SLP: {cert}"
    assert worst < 1e-11, f"L={L}: ground-state SLP margin degrades to {worst:.3e}"


def test_hotta_and_ffh_ground_states_are_slp():
    """The same sanity on the two dim-4 models this milestone is anchored to."""
    H = hotta.hamiltonian(1.0, 1.5)
    g = hotta.ground_state(1.0, 1.5)
    rho = np.outer(g, g.conj())
    for region in (0, 1):
        cert = slp.slp_certificate(H, rho, region)
        assert cert["slp"] and cert["extractable_bound"] < 1e-12, cert
    for kappa in (0.5, 2.0):
        H = slp.ffh_pair_hamiltonian(kappa)
        rho = slp.ffh_pair_eigenmixture(kappa, [1.0, 0.0, 0.0, 0.0])
        cert = slp.slp_certificate(H, rho, 0)
        assert cert["slp"] and cert["extractable_bound"] < 1e-12, cert


# --------------------------------------------------------------------------
# slp_map: the chart
# --------------------------------------------------------------------------


def test_slp_map_shapes_rows_and_transition():
    couplings = [0.5, 1.0, 2.0, 4.0]
    temps = [0.2, 0.5, 1.0, 3.0]
    chart = slp.slp_map(slp.ffh_pair_hamiltonian, slp.gibbs_state,
                        couplings, temps)
    assert chart["verdicts"].shape == (4, 4, 2)
    assert len(chart["rows"]) == 32
    # swap symmetry of FFH Eq. (15): the two regions give the same chart
    assert np.array_equal(chart["verdicts"][:, :, 0], chart["verdicts"][:, :, 1])
    # SLP at low T, not at high T, and monotone in T at fixed coupling
    for i, kappa in enumerate(couplings):
        row = chart["verdicts"][i, :, 0]
        assert row[0], f"kappa={kappa} not SLP at the lowest temperature"
        assert not row[-1], f"kappa={kappa} still SLP at the highest temperature"
        flips = int(np.sum(row[:-1].astype(int) != row[1:].astype(int)))
        assert flips == 1, f"kappa={kappa}: verdict flips {flips} times in T"
    # stronger coupling survives to higher temperature (FFH's central claim)
    n_slp = chart["verdicts"][:, :, 0].sum(axis=1)
    assert np.all(np.diff(n_slp) >= 0), f"SLP window not growing with kappa: {n_slp}"
    for row in chart["rows"]:
        assert row["extractable_bound"] >= 0.0
        assert set(row) >= {"coupling", "state_param", "region", "slp",
                            "lambda_min", "herm_dev", "extractable_bound"}


def test_slp_map_charts_a_six_qubit_chain():
    """The analytic rung's stated domain: 2-6 qubits, over (coupling, state, region).

    Coupling = the transverse field g of the critical Ising chain, state family
    = Gibbs temperature, regions = the bulk site and a bulk pair. Charting all
    three axes at n = 6 is the milestone's `slp_map` deliverable.
    """
    chart = slp.slp_map(
        lambda g: _tfim_dense(6, g),
        slp.gibbs_state,
        [0.5, 1.0, 2.0],
        [0.05, 0.3, 1.0, 4.0],
        regions=[(2,), (2, 3)],
    )
    assert chart["verdicts"].shape == (3, 4, 2)
    for k in range(2):
        for i, g in enumerate(chart["couplings"]):
            row = chart["verdicts"][i, :, k]
            assert row[0], f"g={g} region {chart['regions'][k]} not SLP at T=0.05"
            assert not row[-1], f"g={g} region {chart['regions'][k]} still SLP at T=4"
    # a bigger region can only be easier to extract from, so it loses SLP first
    assert chart["verdicts"][:, :, 1].sum() <= chart["verdicts"][:, :, 0].sum()
    # the commuting Gibbs family kills the Hermiticity half identically
    assert float(np.max(chart["herm_dev"])) < 1e-13


def test_slp_map_rejects_out_of_scope_sizes():
    def big(_):
        return np.eye(2**7)

    def one(_):
        return np.eye(2)

    with pytest.raises(ValueError, match="2-6 qubit"):
        slp.slp_map(big, lambda H, p: H / H.shape[0], [1.0], [1.0])
    with pytest.raises(ValueError, match="2-6 qubit"):
        slp.slp_map(one, lambda H, p: H / 2.0, [1.0], [1.0])


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


def test_input_validation():
    H = slp.ffh_pair_hamiltonian(1.0)
    rho = slp.gibbs_state(H, 1.0)
    with pytest.raises(ValueError, match="not Hermitian"):
        slp.slp_certificate(H + 1j * np.kron(slp._SX, slp._I2), rho, 0)
    with pytest.raises(ValueError, match="not a power of 2"):
        slp.slp_certificate(np.eye(3), np.eye(3) / 3.0, 0)
    with pytest.raises(ValueError, match="outside system"):
        slp.slp_certificate(H, rho, 5)
    with pytest.raises(ValueError, match="repeated sites"):
        slp.slp_certificate(H, rho, (0, 0))
    with pytest.raises(ValueError, match="at least one site"):
        slp.slp_certificate(H, rho, ())
    with pytest.raises(ValueError, match="qubits but rho is on"):
        slp.choi_cost_matrix(H, np.eye(8) / 8.0, 0)
    with pytest.raises(ValueError, match="probability vector"):
        slp.ffh_pair_eigenmixture(1.0, [0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="need T > 0"):
        slp.gibbs_state(H, 0.0)
    with pytest.raises(ValueError, match="need kappa > 0"):
        slp.ffh_pair_hamiltonian(0.0)
    with pytest.raises(ValueError, match="identity"):
        slp.locc_extractable_energy(H, rho, 1, [np.eye(4)] * 2, method="certificate")


def test_gibbs_state_limits():
    H = slp.ffh_pair_hamiltonian(1.0)
    mixed = slp.gibbs_state(H, np.inf)
    assert float(np.max(np.abs(mixed - np.eye(4) / 4.0))) < 1e-14
    cold = slp.gibbs_state(H, 1e-4)
    w, U = np.linalg.eigh(H)
    proj = np.outer(U[:, 0], U[:, 0].conj())
    assert float(np.max(np.abs(cold - proj))) < 1e-10
    # no overflow at tiny T despite exp(-E/T)
    assert np.all(np.isfinite(cold))


# --------------------------------------------------------------------------
# SDP rung (cvxpy) — the milestone's headline row
# --------------------------------------------------------------------------


def test_ffh_sdp_overlap_verdict_and_margin(ffh_grid, ffh_sdp):
    """TEST MATRIX ROW: FFH <-> SDP overlap, verdict agreement, margin 1e-9.

    On the overlap (single-qubit regions, where the FFH closed form applies)
    the analytic certificate, the FFH Eq. (21) closed form and the Choi SDP
    must return the same verdict, and the SDP's certified bracket must
    contain the closed-form value to within 1e-9.
    """
    worst_val = 0.0
    worst_gap = 0.0
    worst_at = None
    for (kappa, T, H, rho, omega), sdp in zip(ffh_grid, ffh_sdp, strict=True):
        cert = slp.slp_certificate(H, rho, 0)
        want = omega <= 1e-12
        assert sdp["slp"] == want, (
            f"kappa={kappa} T={T}: SDP says slp={sdp['slp']} "
            f"(extractable={sdp['extractable']:.3e}, "
            f"upper={sdp['extractable_upper']:.3e}), FFH Omega_o={omega:.3e}"
        )
        assert sdp["slp"] == cert["slp"], (
            f"kappa={kappa} T={T}: SDP {sdp['slp']} vs certificate {cert['slp']}"
        )
        err = max(abs(sdp["extractable"] - omega),
                  abs(sdp["extractable_upper"] - omega))
        if err > worst_val:
            worst_val, worst_at = err, (kappa, T)
        worst_gap = max(worst_gap, sdp["gap"])
    assert worst_val < OVERLAP_MARGIN, (
        f"FFH Eq. (21) vs SDP bracket differ by {worst_val:.3e} at "
        f"(kappa, T) = {worst_at}; margin is {OVERLAP_MARGIN:.0e}"
    )
    assert worst_gap < OVERLAP_MARGIN, (
        f"widest certified SDP bracket {worst_gap:.3e} exceeds the margin"
    )


def test_sdp_bracket_is_valid_and_achieved(ffh_grid, ffh_sdp):
    """The reported extraction is achieved by the returned Choi matrix.

    'extractable' comes from a Choi matrix repaired onto the exact feasible
    set, so re-applying it to rho must reproduce e_min; 'extractable_upper'
    comes from a repaired dual, so it must not be smaller.
    """
    worst_apply = 0.0
    worst_tp = 0.0
    for (_, _, H, rho, _), sdp in zip(ffh_grid, ffh_sdp, strict=True):
        assert sdp["extractable"] >= -1e-15, sdp
        assert sdp["gap"] >= 0.0
        assert sdp["extractable_upper"] >= sdp["extractable"] - 1e-15
        # the returned Choi must be exactly feasible: PSD and Tr_out = I
        w_min = float(np.linalg.eigvalsh(sdp["choi"])[0])
        assert w_min > -1e-12, f"returned Choi has eigenvalue {w_min:.3e}"
        kraus = slp.choi_to_kraus(sdp["choi"])
        completeness = sum(K.conj().T @ K for K in kraus)
        worst_tp = max(worst_tp, float(np.max(np.abs(completeness - np.eye(2)))))
        out = slp.apply_local_channel(rho, kraus, sdp["region"])
        worst_apply = max(worst_apply, abs(slp.state_energy(out, H) - sdp["e_min"]))
    assert worst_tp < 1e-10, f"returned Choi is not CPTP by {worst_tp:.3e}"
    assert worst_apply < 1e-10, (
        f"returned Choi does not achieve e_min: off by {worst_apply:.3e}"
    )


def test_sdp_matches_kraus_search_on_random_states():
    """Two independent optimizers, one convex and one local, on the same data."""
    pytest.importorskip("cvxpy")
    rng = np.random.default_rng(7)
    worst = 0.0
    for n, region in [(2, (0,)), (3, (1,)), (3, (2,))]:
        H = _rand_herm(2**n, rng)
        rho = _rand_rho(2**n, rng)
        sdp = slp.extractable_energy_sdp(H, rho, region)
        kraus = slp.extractable_energy_kraus(H, rho, region, restarts=4)
        # the SDP is convex, so it is the truth; the search can only fall short
        assert kraus["extractable"] <= sdp["extractable_upper"] + 1e-9, (
            f"n={n} {region}: local search {kraus['extractable']:.9f} exceeds the "
            f"certified optimum {sdp['extractable_upper']:.9f}"
        )
        worst = max(worst, abs(kraus["extractable"] - sdp["extractable"]))
    assert worst < 1e-7, f"SDP and Kraus search differ by {worst:.3e}"


def test_sdp_respects_the_analytic_robustness_bound():
    """ASRSM Sec. A.1: extraction never exceeds d_A * eps from the certificate."""
    pytest.importorskip("cvxpy")
    rng = np.random.default_rng(8)
    for n, region in [(2, (0,)), (3, (0, 1)), (4, (1, 2))]:
        H = _rand_herm(2**n, rng)
        rho = _rand_rho(2**n, rng)
        cert = slp.slp_certificate(H, rho, region)
        sdp = slp.extractable_energy_sdp(H, rho, region)
        assert sdp["extractable_upper"] <= cert["extractable_bound"] + 1e-6, (
            f"n={n} {region}: extraction {sdp['extractable_upper']:.6f} above the "
            f"robustness bound {cert['extractable_bound']:.6f}"
        )
        assert sdp["extractable"] > 0.0, "random states should not be SLP"


def test_sdp_multisite_region_of_a_chain():
    """The SDP rung extends past the analytic overlap: a 2-site region, L = 4."""
    pytest.importorskip("cvxpy")
    H = _tfim_dense(4, 1.0)
    rho = slp.gibbs_state(H, 2.0)
    single = slp.extractable_energy_sdp(H, rho, (1,))
    pair = slp.extractable_energy_sdp(H, rho, (1, 2))
    assert pair["d_A"] == 4 and single["d_A"] == 2
    # a larger region can do everything a smaller one can
    assert pair["extractable"] >= single["extractable"] - 1e-8, (
        f"two-site extraction {pair['extractable']:.9f} below one-site "
        f"{single['extractable']:.9f}"
    )
    assert pair["gap"] < 1e-6, f"multi-site bracket too wide: {pair['gap']:.3e}"


def test_certificate_and_sdp_agree_on_multisite_regions():
    """Beyond the FFH overlap: ASRSM Thm. 1 vs the Choi program, 2- and 3-site.

    The analytic certificate is exact for regions of any size, so the two
    rungs must agree everywhere, not just where the FFH closed form exists.
    The Gibbs family is swept across its SLP transition so the comparison has
    both verdicts in it.
    """
    pytest.importorskip("cvxpy")
    H = _tfim_dense(6, 1.0)
    verdicts = []
    worst_extracted = 0.0
    # (2, 3) across the transition, plus one 3-site region (d_A = 8, the SDP
    # rung's expensive end) at a temperature well clear of it.
    cases = [((2, 3), T) for T in (0.05, 0.2, 0.5, 1.0, 3.0)] + [((1, 2, 3), 1.0)]
    for region, T in cases:
        rho = slp.gibbs_state(H, T)
        cert = slp.slp_certificate(H, rho, region)
        sdp = slp.extractable_energy_sdp(H, rho, region)
        assert sdp["gap"] < 1e-8, f"T={T} {region}: bracket {sdp['gap']:.3e}"
        assert cert["slp"] == sdp["slp"], (
            f"T={T} region={region}: certificate {cert['slp']} "
            f"(lambda_min={cert['lambda_min']:.3e}) vs SDP {sdp['slp']} "
            f"(extractable={sdp['extractable']:.3e})"
        )
        assert sdp["extractable_upper"] <= cert["extractable_bound"] + 1e-8, (
            f"T={T} {region}: extraction {sdp['extractable_upper']:.6f} above "
            f"the robustness bound {cert['extractable_bound']:.6f}"
        )
        worst_extracted = max(worst_extracted, sdp["extractable"])
        verdicts.append(cert["slp"])
    assert any(verdicts) and not all(verdicts), (
        f"the multi-site sweep must straddle the transition, got {verdicts}"
    )
    assert worst_extracted > 1e-3, "the sweep never actually extracted anything"


def test_sdp_eight_qubit_system_two_site_region():
    """The SDP rung's stated ceiling: a 2-site region of an 8-qubit system."""
    pytest.importorskip("cvxpy")
    H = _tfim_dense(8, 1.0)
    rho = slp.gibbs_state(H, 1.5)
    res = slp.extractable_energy_sdp(H, rho, (3, 4))
    cert = slp.slp_certificate(H, rho, (3, 4))
    assert res["slp"] == cert["slp"]
    assert res["gap"] < 1e-8, f"bracket {res['gap']:.3e} at n=8"
    assert res["extractable"] > 1e-6


def test_sdp_raises_when_no_solver_succeeds():
    """A dead solver must not be reported as 'the identity map was optimal'."""
    pytest.importorskip("cvxpy")
    H = slp.ffh_pair_hamiltonian(1.0)
    rho = slp.gibbs_state(H, 5.0)
    with pytest.raises(RuntimeError, match="produced no solution"):
        slp.extractable_energy_sdp(H, rho, 0, solver="NOT_A_SOLVER")
    # HIGHS is installed but cannot take an SDP cone: same loud failure
    with pytest.raises(RuntimeError, match="produced no solution"):
        slp.extractable_energy_sdp(H, rho, 0, solver="HIGHS")


def test_sdp_dimension_caps():
    pytest.importorskip("cvxpy")
    big = np.eye(2**9)
    with pytest.raises(ValueError, match="8-qubit"):
        slp.extractable_energy_sdp(big, big / big.shape[0], 0)
    H = _tfim_dense(6, 1.0)
    with pytest.raises(ValueError, match="4-site"):
        slp.extractable_energy_sdp(H, slp.gibbs_state(H, 1.0), (0, 1, 2, 3, 4))


def test_sdp_ground_state_is_slp():
    """The SDP rung agrees with the certificate on the ground-state sanity."""
    pytest.importorskip("cvxpy")
    H = _tfim_dense(4, 1.0)
    _, psi = chains.tfim_ground_state(4, 1.0)
    rho = np.outer(psi, psi.conj()).astype(complex)
    for region in ((0,), (1,), (1, 2)):
        res = slp.extractable_energy_sdp(H, rho, region)
        assert res["slp"], f"ground state judged non-SLP on {region}: {res}"
        assert res["extractable_upper"] < 1e-9, res


# --------------------------------------------------------------------------
# the LOCC layer — classical communication reopens extraction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("h,k", [(1.0, 1.5), (0.5, 1.0), (2.0, 0.3)])
def test_locc_reopens_extraction_hotta(h, k):
    """The milestone's LOCC requirement, on vacuum.qet.hotta's minimal model.

    Bob's qubit is SLP both before Alice measures (it is the global ground
    state) and after she measures but before her bit arrives (the ensemble
    average). Conditioning the very same local operation on the classical
    outcome extracts Hotta's E_B — and the SDP's outcome-optimal *channel*
    turns out to be exactly his conditional rotation, so the LOCC total
    reproduces the closed form of arXiv:1101.3954 Eq. (11).
    """
    pytest.importorskip("cvxpy")
    H = hotta.hamiltonian(h, k)
    g = hotta.ground_state(h, k)
    rho_g = np.outer(g, g.conj())
    projectors = [hotta.measurement_projector(mu) for mu in (+1, -1)]

    res = slp.locc_extractable_energy(H, rho_g, 1, projectors)

    # layer 1: nothing local on Bob's side, with or without the measurement
    assert slp.slp_certificate(H, rho_g, 1)["slp"], "the ground state is not SLP on B"
    cert_meas = slp.slp_certificate(H, res["rho_meas"], 1)
    assert cert_meas["slp"], (
        f"post-measurement ensemble is not SLP on B: {cert_meas['lambda_min']:.3e}"
    )
    assert res["local"] < 1e-9, (
        f"local extraction on B without the bit: {res['local']:.3e}"
    )
    # layer 2: with the classical bit, extraction reopens at Hotta's value
    e_b = hotta.e_b_optimal(h, k)
    assert res["locc"] > 1e-6, f"LOCC extracted nothing: {res['locc']:.3e}"
    dev = abs(res["locc"] - e_b)
    assert dev < 1e-9, (
        f"LOCC extraction {res['locc']:.12f} vs Hotta Eq. (11) {e_b:.12f}, "
        f"off by {dev:.3e}"
    )
    assert res["gap"] > 1e-6, f"communication opened no gap: {res['gap']:.3e}"
    # Alice's side is not passive after her own measurement — she injected E_A
    assert not slp.slp_certificate(H, res["rho_meas"], 0)["slp"]
    # and the QET ledger still closes: E_A >= E_B (hotta_sweep's standing audit)
    assert res["e_meas"] - res["e_initial"] >= res["locc"] - 1e-12


def test_locc_core_only_route_agrees():
    """The LOCC layer also runs without cvxpy (Kraus fallback), same answer."""
    h, k = 1.0, 1.5
    H = hotta.hamiltonian(h, k)
    rho_g = np.outer(hotta.ground_state(h, k), hotta.ground_state(h, k).conj())
    projectors = [hotta.measurement_projector(mu) for mu in (+1, -1)]
    res = slp.locc_extractable_energy(H, rho_g, 1, projectors, method="kraus",
                                      kraus_kwargs={"restarts": 4})
    e_b = hotta.e_b_optimal(h, k)
    assert res["local"] < 1e-9, f"Kraus route extracts locally: {res['local']:.3e}"
    assert abs(res["locc"] - e_b) < 1e-8, (
        f"Kraus LOCC {res['locc']:.12f} vs Hotta Eq. (11) {e_b:.12f}"
    )


def test_locc_branches_are_a_valid_ensemble():
    h, k = 1.0, 1.5
    H = hotta.hamiltonian(h, k)
    rho_g = np.outer(hotta.ground_state(h, k), hotta.ground_state(h, k).conj())
    projectors = [hotta.measurement_projector(mu) for mu in (+1, -1)]
    res = slp.locc_extractable_energy(H, rho_g, 1, projectors, method="certificate")
    ps = [b["p"] for b in res["branches"]]
    assert abs(sum(ps) - 1.0) < 1e-13, f"branch probabilities sum to {sum(ps)}"
    recon = sum(b["p"] * b["rho"] for b in res["branches"])
    assert float(np.max(np.abs(recon - res["rho_meas"]))) < 1e-13
    # the measurement injects energy (Hotta Eq. (8)), so rho_meas is above |g>
    assert res["e_meas"] - res["e_initial"] > 0.0
    with pytest.raises(ValueError, match="method must be"):
        slp.locc_extractable_energy(H, rho_g, 1, projectors, method="nope")


def test_locc_layer_on_the_critical_ising_chain():
    """The LOCC layer beyond dim 4, against M3.2's chain protocol.

    Alice measures sigma_x at site A (step I of :func:`chains.run_chain_qet`).
    Bob's site is strongly local passive before *and* after that measurement,
    so the certificate forbids every channel on B alone; conditioned on her
    outcome, the optimal channel extracts exactly what M3.2's conditional
    single-site rotation extracts — Hotta's rotation is not merely a good
    local operation on Bob's site, it saturates the CPTP optimum. The yield
    falls off with |A - B|, which is M3.2's exchange-rate curve.
    """
    pytest.importorskip("cvxpy")
    L, g, A = 6, 1.0, 0
    H = _tfim_dense(L, g)
    _, psi = chains.tfim_ground_state(L, g)
    rho = np.outer(psi, psi.conj()).astype(complex)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    projectors = [slp.embed_region_operator((np.eye(2) + mu * sx) / 2.0, (A,), L)
                  for mu in (+1, -1)]
    yields = []
    for B in (1, 2, 3):
        res = slp.locc_extractable_energy(H, rho, B, projectors)
        assert slp.slp_certificate(H, rho, B)["slp"], f"ground state not SLP at B={B}"
        cert = slp.slp_certificate(H, res["rho_meas"], B)
        assert cert["slp"], (
            f"B={B}: post-measurement state not SLP, "
            f"lambda_min={cert['lambda_min']:.3e}"
        )
        assert res["local"] < 1e-9, f"B={B}: local extraction {res['local']:.3e}"
        chain = chains.run_chain_qet(L, g, A, B)
        dev = abs(res["locc"] - chain["E_B"])
        assert dev < 1e-8, (
            f"B={B}: CPTP-optimal LOCC {res['locc']:.12f} vs the M3.2 rotation "
            f"{chain['E_B']:.12f}, off by {dev:.3e}"
        )
        e_a = res["e_meas"] - res["e_initial"]
        assert e_a >= res["locc"] - 1e-12, (
            f"B={B}: extracted {res['locc']:.9f} > injected {e_a:.9f}"
        )
        yields.append(res["locc"])
    assert yields[0] > yields[1] > yields[2] > 0.0, (
        f"LOCC yield must decay with distance, got {yields}"
    )


def test_slp_map_locc_layer():
    """The boundary chart's two layers in one call (local-forbidden, LOCC-open)."""
    pytest.importorskip("cvxpy")

    def ham(h):
        return hotta.hamiltonian(h, 1.5)

    def ground(H, _):
        w, U = np.linalg.eigh(H)
        return np.outer(U[:, 0], U[:, 0].conj())

    def projectors(_H, _rho, _region):
        return [hotta.measurement_projector(mu) for mu in (+1, -1)]

    chart = slp.slp_map(ham, ground, [0.5, 1.0, 2.0], [None], regions=[(1,)],
                        locc_projectors_fn=projectors)
    assert chart["verdicts"].all(), "ground states must be SLP on Bob's qubit"
    for row, h in zip(chart["rows"], [0.5, 1.0, 2.0], strict=True):
        assert row["locc_reopens"], row
        assert abs(row["locc"] - hotta.e_b_optimal(h, 1.5)) < 1e-9, row
        assert row["locc_local"] < 1e-9, row
    assert np.all(chart["locc_gap"] > 1e-6)


def test_extractable_energy_sdp_needs_the_extra(monkeypatch):
    """Without cvxpy the SDP rung raises a pointed ImportError, nothing else."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cvxpy":
            raise ImportError("no cvxpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    H = slp.ffh_pair_hamiltonian(1.0)
    with pytest.raises(ImportError, match=r"vacuum\[sdp\]"):
        slp.extractable_energy_sdp(H, slp.gibbs_state(H, 1.0), 0)


def test_no_solver_warnings_leak(ffh_grid):
    """The SDP rung must not spray 'solution may be inaccurate' at its caller."""
    pytest.importorskip("cvxpy")
    _, _, H, rho, _ = ffh_grid[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        slp.extractable_energy_sdp(H, rho, 0)
    leaked = [str(w.message) for w in caught if "inaccurate" in str(w.message)]
    assert not leaked, f"solver warnings leaked: {leaked}"
