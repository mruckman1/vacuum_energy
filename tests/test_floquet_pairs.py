"""Anchors 2 and 4 (Layer 5): two-mode-squeezed (k, -k) pairs and the ledger.

Anchor 2 — a uniformly modulated periodic chain produces pure two-mode
squeezed vacua on every traveling-wave pair (k, -k): E_N(k,-k) = 2 r with
r = arcsinh(sqrt <n_k>) read from the squeezing spectrum (Adesso-Serafini-
Illuminati PRA 70, 022318 (2004), Eq. (20); Vidal-Werner 2002).  Pinned to
1e-8 for the resonant pair and every other pair; the per-period growth of r
equals ln max|eig S_F| (the Floquet exponent) asymptotically; the standing
wave (cos, sin) partition of the same pair carries *no* negativity.

Anchor 4 — the energy ledger closes to 1e-9 on every run and the drive is
closes with the drive work booked as W_drive = sum_k omega_k <n_k> (an
algebraic identity with the final covariance under K_0, so this tests ledger
closure and bookkeeping, not an independent photon count); passivity of
the undriven vacuum holds, and the audit fires on the amplified state when
handed the inverse drive (it has teeth).  The fast monodromy path and
``vacuum.protocol.run_protocol`` agree on the state and the work.
"""

import numpy as np
import pytest

from vacuum.audits import passivity_audit
from vacuum.core import Omega, ground_state_cov, harmonic_chain_K, log_negativity, symplectic_inverse
from vacuum.floquet import (
    entanglement_growth,
    floquet_map,
    floquet_run,
    modulated_K,
    normal_mode_symplectic,
    normal_modes,
    pair_negativity,
    pair_spectrum,
    power_iterate,
    run_floquet_protocol,
    squeezing_spectrum,
    traveling_wave_basis,
)

N, M = 8, 0.5
OMEGA_RES = 1.5  # the k = pi/2 pair: omega^2 = m^2 + 4 sin^2(pi/4) = 2.25
DEPTH = 0.1
N_PERIODS = 20
TMSV_TOL = 1e-8
LEDGER_TOL = 1e-9


@pytest.fixture(scope="module")
def chain():
    K0 = harmonic_chain_K(N, M)
    V0 = ground_state_cov(K0)
    basis = traveling_wave_basis(K0, mass=M)
    drive = modulated_K(K0, DEPTH, 2 * OMEGA_RES, "cos")
    fm = floquet_map(drive, n_steps=32, conv_tol=1e-11, max_halvings=10)
    return K0, V0, basis, drive, fm


def test_traveling_wave_basis_structure(chain):
    K0, V0, basis, drive, fm = chain
    T = basis.T
    assert np.max(np.abs(T @ Omega(N) @ T.T - Omega(N))) < 1e-14
    assert basis.pairs == ((1, 2), (3, 4), (5, 6)) and basis.singles == (0, 7)
    assert abs(basis.omega[3] - OMEGA_RES) < 1e-12
    assert abs(basis.k[3] - np.pi / 2) < 1e-12
    # the vacuum has zero occupation in every traveling mode
    assert np.max(np.abs(squeezing_spectrum(V0, basis=basis))) < 1e-7


def test_pair_negativity_equals_twice_squeezing(chain):
    K0, V0, basis, drive, fm = chain
    g = entanglement_growth(V0, fm.S, N_PERIODS, K0, basis=basis)
    V = power_iterate(V0, fm.S, N_PERIODS)
    r_k = squeezing_spectrum(V, basis=basis)
    for c, (i, j) in enumerate(g.pairs):
        E = pair_negativity(V, c, basis=basis)
        two_r = 2.0 * float(r_k[i])
        assert abs(E - two_r) < TMSV_TOL, (
            f"pair {(i, j)} omega={basis.omega[i]:.4f}: E_N={E:.12f} vs 2r={two_r:.12f} "
            f"(diff {E - two_r:.2e})"
        )
        assert abs(g.E_N[-1, c] - E) < 1e-12
        assert abs(r_k[i] - r_k[j]) < 1e-9  # k and -k partners share r
    # the resonant pair dominates
    res = 1
    assert g.E_N[-1, res] > 3.0 and g.E_N[-1, res] > 5 * max(g.E_N[-1, 0], g.E_N[-1, 2]), (
        f"E_N per pair after {N_PERIODS} periods: {g.E_N[-1]}"
    )
    assert not g.flagged[-1, res]


def test_growth_rate_is_the_floquet_exponent(chain):
    K0, V0, basis, drive, fm = chain
    g = entanglement_growth(V0, fm.S, N_PERIODS, K0, basis=basis, pairs=[1])
    inc_r = np.diff(g.r[:, 0])[-3:]
    inc_E = np.diff(g.E_N[:, 0])[-3:] / 2.0
    lnlam = np.log(fm.multiplier)
    assert np.max(np.abs(inc_r - lnlam)) < 1e-5, (
        f"r increments {inc_r} vs ln lambda {lnlam:.8f}"
    )
    assert np.max(np.abs(inc_E - inc_r)) < 1e-9
    # E_N grows linearly in the number of periods once the transient is gone
    assert g.E_N[-1, 0] > g.E_N[N_PERIODS // 2, 0] > 0.0


def test_standing_wave_partition_carries_no_negativity(chain):
    K0, V0, basis, drive, fm = chain
    V = power_iterate(V0, fm.S, N_PERIODS)
    omega, U = normal_modes(K0)
    Vn = normal_mode_symplectic(U) @ V @ normal_mode_symplectic(U).T
    E_standing = log_negativity(Vn, [3], [4])
    E_traveling = pair_negativity(V, 1, basis=basis)
    assert E_standing < 1e-9 and E_traveling > 3.0, (
        f"standing-wave E_N {E_standing:.2e}, traveling-wave E_N {E_traveling:.4f}"
    )


def test_pair_spectrum_peaks_on_the_resonant_pair(chain):
    K0, V0, basis, drive, fm = chain
    n_k = pair_spectrum(V0, fm.S, N_PERIODS, K0)
    omega, _ = normal_modes(K0)
    top = np.argsort(n_k)[-2:]
    assert set(top) == {3, 4} and abs(omega[3] - OMEGA_RES) < 1e-12
    assert n_k[3] > 5.0 and np.all(n_k >= -1e-12), f"n_k = {n_k}"


# --------------------------------------------------------------------------
# Anchor 4: ledger and passivity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ("sin", "cos", "square"))
def test_ledger_closes_with_drive_work_identity(profile):
    K0 = harmonic_chain_K(N, M)
    V0 = ground_state_cov(K0)
    drive = modulated_K(K0, DEPTH, 2 * OMEGA_RES, profile)
    run = floquet_run(V0, drive, 10, n_steps=16, conv_tol=1e-10)
    defect = abs(run.ledger_result.details["defect"])
    assert run.ledger_result.passed and defect < LEDGER_TOL, f"ledger defect {defect:.2e}"
    omega, _ = normal_modes(K0)
    photons = float(np.sum(omega * run.occupations[-1]))
    assert abs(run.work_total - photons) < LEDGER_TOL, (
        f"{profile}: drive work {run.work_total:.12f} vs sum_k omega_k n_k {photons:.12f}"
    )
    assert run.work_total > 0.0 and run.dissipated_total == 0.0
    assert run.passivity is not None and run.passivity.passed


def test_fast_path_matches_run_protocol():
    K0 = harmonic_chain_K(N, M)
    V0 = ground_state_cov(K0)
    for profile in ("cos", "square"):
        drive = modulated_K(K0, DEPTH, 2 * OMEGA_RES, profile)
        run = floquet_run(V0, drive, 6, n_steps=16, conv_tol=1e-10)
        res = run_floquet_protocol(V0, drive, 6, n_substeps_per_period=run.fmap.n_steps)
        dV = np.max(np.abs(res.V - run.V))
        dW = abs(res.work_total - run.work_total)
        assert dV < 1e-10 and dW < 1e-9, (
            f"{profile}: |dV| {dV:.2e}, |dW| {dW:.2e} between floquet_run and run_protocol"
        )
        assert res.all_passed() and abs(res.audits["ledger"].details["defect"]) < LEDGER_TOL
        assert res.audits["passivity_entry"].passed


def test_passivity_holds_undriven_and_fires_on_the_amplified_state(chain):
    K0, V0, basis, drive, fm = chain
    assert passivity_audit(K0, V=V0, n_samples=100).passed
    V = power_iterate(V0, fm.S, N_PERIODS)
    S_undo = symplectic_inverse(np.linalg.matrix_power(fm.S, N_PERIODS))

    def undo_sampler(rng, n):
        return S_undo

    res = passivity_audit(K0, V=V, sampler=undo_sampler, n_samples=1, strict=False)
    assert not res.passed and res.worst_violation > 1.0, (
        f"the amplified state should be active: worst violation {res.worst_violation:.4f}"
    )
