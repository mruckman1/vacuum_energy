"""Layer-6 anchor: inverse design of vacua (vacuum.opt.inverse_design).

1. RECOVERY: with the critical Dirichlet chain's own ground-state x-x
   correlations as the target, the banded ansatz (K = L L^T + eps I, L
   lower-bidiagonal) recovers ``harmonic_chain_K`` from the identity start
   — residual < 1e-8 (measured ~1e-12), K to 1e-6 absolute — up to the
   ansatz's symmetry (column signs of L), which is checked to be exactly a
   symmetry.  The covariance target on a smaller chain runs the standing
   audits of the designed vacuum: passivity, precision, causality (with
   the Lieb-Robinson velocity bound derived from the design itself, = 1
   for a unit-coupling chain).
2. DESIGN: a prescribed negativity between two chosen blocks, twice the
   free chain's, is reached to 1e-4 nats with K positive definite by
   construction and the designed vacuum passing passivity and precision.
3. Parametrizations round-trip (banded, full, eigen) and ``design_residual``
   is consistent with the optimizer's own residual.

Requires JAX (importorskip).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from vacuum.core import ground_state_cov, harmonic_chain_K, log_negativity  # noqa: E402
from vacuum.opt.inverse_design import (  # noqa: E402
    coupling_from_theta,
    design_residual,
    inverse_design,
    lieb_robinson_velocity_bound,
    theta_from_coupling,
)


def test_recovers_critical_chain_from_its_correlations():
    N = 64
    K = harmonic_chain_K(N, 1e-6, bc="dirichlet")
    V = ground_state_cov(K)
    res = inverse_design({"kind": "corr_xx", "V_xx": V[:N, :N]}, ansatz="banded", band=1, N=N, audit=False)
    dK = float(np.max(np.abs(res.K - K)))
    assert res.residual < 1e-8, f"recovery residual {res.residual:.3e}"
    assert dK < 1e-6, f"max |K - harmonic_chain_K| = {dK:.3e}"
    assert res.min_eigenvalue > 0.0
    assert abs(design_residual(res.K, {"kind": "corr_xx", "V_xx": V[:N, :N]}) - res.residual) < 1e-15
    # the ansatz symmetry: L -> L D with D = diag(+-1) leaves K unchanged
    th = theta_from_coupling(K, "banded", 1)
    rows, cols = np.tril_indices(N)
    from vacuum.opt.inverse_design import banded_indices

    r, c = banded_indices(N, 1)
    signs = np.where(np.random.RandomState(0).rand(N) < 0.5, -1.0, 1.0)
    th_flipped = th * signs[c]
    K1 = np.asarray(coupling_from_theta(th, N, "banded", 1, xp=np))
    K2 = np.asarray(coupling_from_theta(th_flipped, N, "banded", 1, xp=np))
    assert np.max(np.abs(K1 - K2)) < 1e-13
    print(f"\ncritical-chain recovery: residual {res.residual:.2e}, max|dK| {dK:.2e}, {len(res.history)} evaluations")


def test_covariance_target_with_audits():
    N = 32
    K = harmonic_chain_K(N, 1e-6, bc="dirichlet")
    V = ground_state_cov(K)
    res = inverse_design({"kind": "covariance", "V": V}, ansatz="banded", band=1, N=N)
    assert res.residual < 1e-8
    assert res.all_passed(), {k: getattr(v, "passed", v) for k, v in res.audits.items()}
    assert res.audits["passivity"].passed and res.audits["precision"].passed
    assert not isinstance(res.audits["causality"], dict) and res.audits["causality"].passed
    assert abs(res.v_max_bound - 1.0) < 1e-6, f"velocity bound {res.v_max_bound}"


def test_negativity_design_doubles_block_negativity():
    N = 16
    K0 = harmonic_chain_K(N, 0.5, bc="dirichlet")
    A, B = [3, 4, 5], [7, 8, 9]
    e0 = log_negativity(ground_state_cov(K0), A, B)
    assert e0 > 0.01
    target = {"kind": "negativity", "pairs": [(A, B, 2.0 * e0)]}
    res = inverse_design(target, ansatz="banded", band=1, K_ref=K0, reg=1e-3, maxiter=500)
    achieved = res.achieved["E_N"][0]
    assert abs(achieved - 2.0 * e0) < 1e-4, f"achieved {achieved:.6f} vs target {2 * e0:.6f}"
    assert res.min_eigenvalue > 0.0
    assert res.audits["passivity"].passed and res.audits["precision"].passed
    assert np.allclose(res.K, res.K.T)
    # the design stayed banded and close to the reference (the regularizer)
    assert np.max(np.abs(np.triu(res.K, 2))) < 1e-12
    assert np.max(np.abs(res.K - K0)) < 0.5
    print(f"\nnegativity design: E_N {e0:.5f} -> {achieved:.5f} (target {2 * e0:.5f}), "
          f"residual {res.residual:.1e}, min eig {res.min_eigenvalue:.3f}, v_max bound {res.v_max_bound:.3f}")


def test_parametrizations_round_trip():
    N = 10
    K = harmonic_chain_K(N, 0.7, bc="dirichlet")
    for ansatz, band in (("banded", 1), ("full", None), ("eigen", None)):
        th = theta_from_coupling(K, ansatz, 1 if band is None else band)
        Kj = np.asarray(coupling_from_theta(th, N, ansatz, 1 if band is None else band))
        Kn = np.asarray(coupling_from_theta(th, N, ansatz, 1 if band is None else band, xp=np))
        assert np.max(np.abs(Kj - K)) < 1e-12, ansatz
        assert np.max(np.abs(Kn - Kj)) < 1e-12, ansatz
    v, band = lieb_robinson_velocity_bound(K)
    # massive chain: max_k |sin k| / sqrt(m^2 + 4 sin^2(k/2)) < 1 (it is 1 only as m -> 0)
    assert band == 1 and 0.5 < v < 1.0
    v0, _ = lieb_robinson_velocity_bound(harmonic_chain_K(N, 1e-6, bc="dirichlet"))
    assert abs(v0 - 1.0) < 1e-6
    with pytest.raises(ValueError):
        inverse_design({"kind": "bogus"}, N=4)
