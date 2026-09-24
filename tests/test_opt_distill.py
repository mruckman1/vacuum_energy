"""Layer-6 anchor: the distillation back end (vacuum.opt.distill).

1. The CODED protocols (dense 16x16 two-pair algebra: local rotations,
   bilateral CNOT, target measurement, post-selection, partial trace)
   reproduce the literature recurrences to 1e-12:
   - BBPSSW: Bennett et al., PRL 76, 722 (1996), Eq. (7) on the Werner
     state of Eq. (4) (mostly-phi+ form, their step A1);
   - DEJMPS: Deutsch et al., PRL 77, 2818 (1996), Eq. (7) on Bell-diagonal
     inputs (A, B, C, D) in the basis order (phi+, psi-, psi+, phi-).
2. Yields: prod p_r / 2^n and the fidelity trajectory; the target-fidelity
   stopping rule; non-distillable inputs (A <= 1/2) yield nothing.
3. Gaussian inputs: the leading-order qubit projection of a two-mode block
   gives A - 1/2 = PKMM's negativity estimator |m| - (n_A + n_B)/2, the
   PKMM pair-state route agrees, and the vacuum projects to A = D = 1/2.
4. The exchange rate on an audited harvesting run: distilled Bell pairs
   per unit ledger work, with the run's audits carried.

Pure numpy; no JAX needed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vacuum.core import harmonic_chain_K
from vacuum.detectors import switching
from vacuum.opt import distill as D

TOL = 1e-12


def test_bell_basis_and_twirl():
    Bm = D.bell_states()
    assert np.allclose(Bm.conj().T @ Bm, np.eye(4), atol=1e-15)
    rho = D.bell_diagonal_state([0.6, 0.1, 0.2, 0.1])
    assert np.allclose(D.bell_diagonal_coefficients(rho), [0.6, 0.1, 0.2, 0.1], atol=1e-15)
    assert abs(np.trace(rho) - 1.0) < 1e-15
    # a non-Bell-diagonal state: twirl keeps the diagonal
    rs = np.random.RandomState(0)
    A = rs.randn(4, 4) + 1j * rs.randn(4, 4)
    rho = A @ A.conj().T
    rho /= np.trace(rho).real
    tw = D.twirl_to_bell_diagonal(rho)
    assert np.allclose(D.bell_diagonal_coefficients(tw), D.bell_diagonal_coefficients(rho), atol=1e-14)
    assert np.allclose(tw, tw.conj().T)


@pytest.mark.parametrize("F", [0.51, 0.55, 0.7, 0.85, 0.95, 0.99])
def test_bbpssw_coded_round_matches_eq7(F):
    rho = D.werner_state(F, target="phi+")
    _, F2, p = D.bbpssw_coded_round(rho)
    F_ref, p_ref = D.bbpssw_recurrence(F)
    assert abs(F2 - F_ref) < TOL, f"BBPSSW F={F}: coded F'={F2!r} vs Eq. (7) {F_ref!r}"
    assert abs(p - p_ref) < TOL, f"BBPSSW F={F}: coded p={p!r} vs Eq. (7) denominator {p_ref!r}"
    assert F2 > F  # purification above 1/2


def test_bbpssw_fixed_points():
    """F = 1/2 is a fixed point of Eq. (7) (no purification), F = 1 as well."""
    assert abs(D.bbpssw_recurrence(0.5)[0] - 0.5) < TOL
    assert abs(D.bbpssw_recurrence(1.0)[0] - 1.0) < TOL


def test_dejmps_coded_round_matches_eq7():
    rs = np.random.RandomState(1)
    worst = 0.0
    for _ in range(8):
        c = rs.dirichlet([4, 1, 1, 1])
        _, coded, p = D.dejmps_coded_round(D.bell_diagonal_state(c))
        A, B, C, Dd, N = D.dejmps_recurrence(*c)
        dev = max(np.max(np.abs(np.array(coded) - np.array([A, B, C, Dd]))), abs(p - N))
        worst = max(worst, dev)
        assert dev < TOL, f"DEJMPS input {c}: coded {coded} vs Eq. (7) {(A, B, C, Dd)}, p {p} vs N {N}"
        assert abs(sum(coded) - 1.0) < 1e-14
    print(f"\nDEJMPS coded vs Eq. (7): worst deviation {worst:.2e}")


def test_dejmps_rotations_are_essential():
    """Without the R_x(+-pi/2) rotations the bilateral-CNOT round is a different
    map (it is BBPSSW's, which never mixes B and D), so the anchor above pins
    the transcription of the rotations, not only of the CNOTs.  (Flipping
    BOTH rotation senses is a symmetry of Bell-diagonal states and gives the
    same map — checked as well.)"""
    c = np.array([0.7, 0.05, 0.1, 0.15])
    rho = D.bell_diagonal_state(c)
    src, p = D._bilateral_cnot_and_select(D._two_pair_state(rho))
    no_rot = D.bell_diagonal_coefficients(src)
    A, B, C, Dd, N = D.dejmps_recurrence(*c)
    assert np.max(np.abs(no_rot - np.array([A, B, C, Dd]))) > 1e-3
    Ra = np.cos(np.pi / 4) * np.eye(2) + 1j * np.sin(np.pi / 4) * np.array([[0, 1], [1, 0]])
    Rb = Ra.conj().T
    R = D._op4(Ra, 0) @ D._op4(Ra, 2) @ D._op4(Rb, 1) @ D._op4(Rb, 3)
    src2, _ = D._bilateral_cnot_and_select(R @ D._two_pair_state(rho) @ R.conj().T)
    assert np.max(np.abs(D.bell_diagonal_coefficients(src2) - np.array([A, B, C, Dd]))) < TOL


def test_yield_accounting():
    c = [0.55, 0.02, 0.03, 0.40]  # a harvested-like state: mostly phi+ and phi-
    res = D.distill_yield(c, "DEJMPS", n_rounds=3)
    assert res.n_rounds == 3
    assert abs(res.success_probability - np.prod(res.pass_probabilities)) < 1e-15
    assert abs(res.bell_pairs_per_run - res.success_probability / 8.0) < 1e-15
    assert res.fidelities[0] == 0.55 and res.fidelity == res.coefficients[-1][0]
    assert all(b > a for a, b in zip(res.fidelities[:-1], res.fidelities[1:]))  # (not true for every input: A'=(A^2+B^2)/N)
    coded = D.distill_yield(c, "DEJMPS", n_rounds=3, method="coded")
    assert abs(coded.fidelity - res.fidelity) < TOL and abs(coded.bell_pairs_per_run - res.bell_pairs_per_run) < TOL
    bb = D.distill_yield(c, "BBPSSW", n_rounds=3)
    bb_coded = D.distill_yield(c, "BBPSSW", n_rounds=3, method="coded")
    assert abs(bb.fidelity - bb_coded.fidelity) < TOL
    # DEJMPS is the more efficient recurrence on this input
    assert res.fidelity > bb.fidelity
    # target-fidelity stopping
    t = D.distill_yield(c, "DEJMPS", target_fidelity=0.999)
    assert t.fidelity >= 0.999 and t.fidelities[-2] < 0.999
    # tuple unpacking contract
    y, f = D.distill_yield(c, "DEJMPS", n_rounds=2)
    assert 0 < y < 0.25 and 0.55 < f < 1.0
    # non-distillable input: fidelity <= 1/2 makes no progress
    dead = D.distill_yield([0.49, 0.21, 0.2, 0.1], "DEJMPS", target_fidelity=0.9, max_rounds=10)
    assert dead.n_rounds == 0 and dead.bell_pairs_per_run == 0.0 and dead.fidelity <= 0.5 + 1e-12
    # fixed rounds on a non-distillable input: the recurrence runs, nothing is a Bell pair
    dead2 = D.distill_yield([0.45, 0.2, 0.2, 0.15], "DEJMPS", n_rounds=2)
    assert dead2.bell_pairs_per_run == 0.0 and dead2.fidelity <= 0.5


def test_qubit_projection_of_gaussian_block():
    """A - 1/2 of the projected block equals |m| - (n_A + n_B)/2 (PKMM Eq. (68))."""
    gaps = (1.3, 2.0)
    n_A, n_B, m, c = 0.012, 0.009, 0.02 * np.exp(0.4j), 0.003 - 0.002j
    # build a covariance with exactly these moments (inverse of oscillator_moments)
    W = np.zeros((4, 4))
    W[0, 0] = n_A + 0.5
    W[2, 2] = n_A + 0.5
    W[1, 1] = n_B + 0.5
    W[3, 3] = n_B + 0.5
    W[0, 1] = W[1, 0] = m.real + c.real
    W[2, 3] = W[3, 2] = -m.real + c.real
    W[0, 3] = W[3, 0] = m.imag + c.imag
    W[2, 1] = W[1, 2] = m.imag - c.imag
    Dinv = np.diag(np.concatenate([1 / np.sqrt(gaps), np.sqrt(gaps)]))
    V_AB = Dinv @ W @ Dinv
    mom = D.oscillator_moments(V_AB, gaps)
    assert abs(mom["n_A"] - n_A) < 1e-14 and abs(mom["n_B"] - n_B) < 1e-14
    assert abs(mom["m"] - m) < 1e-14 and abs(mom["c"] - c) < 1e-14
    coeffs = D.bell_diagonal_from_covariance(V_AB, gaps)
    assert abs(np.sum(coeffs) - 1.0) < 1e-14 and np.all(coeffs >= 0)
    assert abs((coeffs[0] - 0.5) - (abs(m) - 0.5 * (n_A + n_B))) < 1e-14
    # the vacuum projects to (1/2, 0, 0, 1/2): |gg> = (phi+ + phi-)/sqrt2
    assert np.allclose(D.bell_diagonal_from_covariance(0.5 * np.eye(4), (1.0, 1.0)), [0.5, 0, 0, 0.5], atol=1e-15)


def test_pair_state_route_agrees_with_moment_route():
    from vacuum.detectors import UDWDetector, udw_pair_state, wightman_lattice

    N = 20
    K = harmonic_chain_K(N, 0.6, bc="dirichlet")
    kernel = wightman_lattice(K)
    F = np.eye(N)
    lam, gap = 0.1, 2.0
    st = udw_pair_state(kernel, UDWDetector(switching("cos2", 2.0, 0.0), F[8]),
                        UDWDetector(switching("cos2", 2.0, 0.0), F[9]), lam / math.sqrt(2 * gap), gap)
    coeffs = D.bell_diagonal_from_pair_state(st)
    assert abs((coeffs[0] - 0.5) - st.negativity_estimator) < 1e-14
    assert st.negativity_estimator > 0.0  # harvesting -> distillable
    res = D.distill_yield(coeffs, "DEJMPS", target_fidelity=0.99)
    assert res.fidelity >= 0.99 and res.bell_pairs_per_run > 0.0
    print(f"\nPKMM pair state -> DEJMPS: {res.n_rounds} rounds, yield {res.bell_pairs_per_run:.3e}/run, F={res.fidelity:.4f}")


def test_vacuum_to_bell_rate_on_audited_run():
    N = 20
    K = harmonic_chain_K(N, 0.6, bc="dirichlet")
    chi = switching("cos2", 2.0, 0.0)
    cfg = dict(K=K, sites_or_profiles=[8, 9], gaps=(2.0, 2.0), lambda_of_t=lambda t: 0.24 * chi(t),
               ts=np.array([-2.0, 2.0]), conv_tol=1e-7, max_halvings=12)
    xr = D.vacuum_to_bell_rate(cfg, protocol="DEJMPS", target_fidelity=0.99)
    assert xr.harvest.ledger_result.passed and xr.harvest.passivity.passed
    assert xr.E_N > 0 and xr.fidelity >= 0.99 and xr.n_rounds >= 1
    assert xr.rate == xr.bell_pairs_per_run / xr.work
    # the same run handed back as a HarvestResult gives the same rate
    xr2 = D.vacuum_to_bell_rate(xr.harvest, protocol="DEJMPS", target_fidelity=0.99)
    assert xr2.rate == xr.rate
    print(f"\nexchange rate: {xr.rate:.4e} Bell pairs per unit work "
          f"({xr.bell_pairs_per_run:.3e}/run, {xr.n_rounds} DEJMPS rounds, F={xr.fidelity:.4f}, W={xr.work:.4e})")
