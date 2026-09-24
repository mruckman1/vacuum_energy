"""Layer 5 engine: drives, exact propagators, monodromy integrators, charts.

Pinned here (measured values in the assertion messages, repo convention):

- the indefinite-safe spectral propagator equals ``chain_propagator`` on
  positive-definite K and ``expm`` on indefinite K;
- piecewise-constant profiles give the *exact* Meissner monodromy (analytic
  trace formula) with no discretisation error;
- the midpoint rule converges at order 2 and the CF4 commutator-free Magnus
  scheme at order 4 (Blanes-Moan 2006), measured from a dt sweep;
- the dt-halving gate records a converged level and every substep is
  symplectic to roundoff;
- the stacked 2 x 2 mode path equals the full monodromy for a diagonal
  drive, and ``momentum_gaps`` finds the resonant mode at omega_mod = 2 omega_k.

Conventions per docs/API.md: hbar = 1, block ordering, V_vac = I/2.
"""

import numpy as np
import pytest
from scipy.linalg import expm

from vacuum.core import Omega, chain_propagator, harmonic_chain_K
from vacuum.floquet import (
    Drive,
    batched_monodromy,
    dce_chain_K,
    floquet_map,
    mode_monodromy,
    modulated_K,
    momentum_gaps,
    multiplier,
    propagator,
    stability_chart,
    symplectic_defect,
)


@pytest.fixture(scope="module")
def K12():
    return harmonic_chain_K(12, m=0.3)


# --------------------------------------------------------------------------
# propagator
# --------------------------------------------------------------------------


def test_propagator_matches_chain_propagator(K12):
    d = np.max(np.abs(propagator(K12, 0.37) - chain_propagator(K12, 0.37)))
    assert d < 1e-13, f"propagator vs chain_propagator: {d:.3e}"


def test_propagator_indefinite_matches_expm(K12):
    Kn = K12 - 3.0 * np.eye(12)  # some negative eigenvalues: inverted modes
    H = np.zeros((24, 24))
    H[:12, :12] = Kn
    H[12:, 12:] = np.eye(12)
    S = propagator(Kn, 0.3)
    d = np.max(np.abs(S - expm(0.3 * Omega(12) @ H)))
    assert d < 1e-12, f"indefinite propagator vs expm: {d:.3e}"
    assert symplectic_defect(S) < 1e-12


# --------------------------------------------------------------------------
# drives
# --------------------------------------------------------------------------


def test_modulated_K_kinds_and_profiles(K12):
    d = modulated_K(K12, 0.2, 1.7, "cos")
    assert isinstance(d, Drive) and d.diagonal and not d.piecewise_constant
    assert np.allclose(d(0.0), 1.2 * K12) and np.allclose(d(d.period / 2), 0.8 * K12)
    dm = modulated_K(K12, 0.5, 1.7, "sin", kind="mass", mass_sq=0.09)
    assert np.allclose(dm(0.0), K12)
    assert np.allclose(dm(dm.period / 4), K12 + 0.045 * np.eye(12))
    P = np.zeros(12)
    P[3] = 1.0
    dp = modulated_K(K12, 0.3, 1.7, "square", kind="pattern", pattern=P)
    assert not dp.diagonal and dp.piecewise_constant
    assert dp(0.1)[3, 3] == pytest.approx(K12[3, 3] + 0.3)
    assert dp(0.1 + dp.period / 2)[3, 3] == pytest.approx(K12[3, 3] - 0.3)
    Kb = dce_chain_K(12, 0.5)
    db = modulated_K(Kb, 0.4, 0.5, "cos", kind="boundary", L0=0.5)
    assert db(0.0)[0, 0] == pytest.approx(1.0 + 1.0 / (0.5 * 1.4))
    assert np.allclose(db(0.0)[1:, 1:], Kb[1:, 1:])
    with pytest.raises(ValueError):
        modulated_K(Kb, 1.0, 0.5, kind="boundary", L0=0.5)
    with pytest.raises(ValueError):
        modulated_K(K12, 0.1, 1.0, kind="mass")
    custom = modulated_K(K12, 0.1, 1.0, ((1.0, 0.25), (-1.0, 0.75)))
    assert custom.piecewise_constant and custom.profile == "pieces"
    assert custom.profile_value(0.1 * custom.period) == 1.0
    assert custom.profile_value(0.5 * custom.period) == -1.0


# --------------------------------------------------------------------------
# monodromy: exactness, orders, gate
# --------------------------------------------------------------------------


def test_square_profile_is_exact_meissner():
    d = 0.3
    drv = modulated_K(np.array([[1.0]]), d, 2.0, "square")
    fm = floquet_map(drv)
    assert fm.exact and fm.converged
    T = np.pi
    w1, w2 = np.sqrt(1 + d), np.sqrt(1 - d)
    tr = 2 * np.cos(w1 * T / 2) * np.cos(w2 * T / 2) - (w1 / w2 + w2 / w1) * np.sin(
        w1 * T / 2
    ) * np.sin(w2 * T / 2)
    lam = 0.5 * (abs(tr) + np.sqrt(tr * tr - 4)) if abs(tr) > 2 else 1.0
    assert abs(fm.multiplier - lam) < 1e-13, (
        f"Meissner multiplier {fm.multiplier:.15f} vs analytic {lam:.15f}"
    )
    assert fm.symplectic_defect < 1e-14


def test_integrator_orders():
    drv = modulated_K(np.array([[1.0]]), 0.3, 2.0, "cos")
    ref = floquet_map(drv, n_steps=64, method="cf4", conv_tol=1e-12, max_halvings=10)
    assert ref.converged and ref.movement < 1e-12
    for method, order, tol in (("midpoint", 2.0, 0.05), ("cf4", 4.0, 0.05)):
        errs = [
            np.max(np.abs(floquet_map(drv, n_steps=n, method=method, conv_tol=None).S - ref.S))
            for n in (16, 32, 64)
        ]
        orders = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
        assert all(abs(o - order) < tol for o in orders), (
            f"{method}: measured orders {orders} (expected {order}); errors {errs}"
        )


def test_gate_records_convergence_and_symplecticity():
    drv = modulated_K(harmonic_chain_K(6, 0.4), 0.2, 2.2, "cos")
    fm = floquet_map(drv, n_steps=8, conv_tol=1e-10, max_halvings=10)
    assert fm.converged and fm.movement < 1e-10
    assert fm.n_steps == 8 * 2**fm.halvings
    assert len(fm.history) == fm.halvings + 1
    assert fm.symplectic_defect < 1e-12, f"symplectic defect {fm.symplectic_defect:.2e}"
    with pytest.raises(RuntimeError):
        floquet_map(drv, n_steps=8, conv_tol=1e-30, max_halvings=2)


def test_mode_path_equals_full_monodromy():
    K = harmonic_chain_K(6, 0.4)
    drv = modulated_K(K, 0.25, 1.9, "cos")
    fm = floquet_map(drv, n_steps=16, conv_tol=1e-11, max_halvings=10)
    w2, U = np.linalg.eigh(K)
    M, info = mode_monodromy(drv, w2, n_steps=16, conv_tol=1e-11)
    # full map in the normal-mode basis must be block-diagonal with these 2x2 blocks
    T = np.zeros((12, 12))
    T[:6, :6] = U.T
    T[6:, 6:] = U.T
    Sm = T @ fm.S @ T.T
    worst = 0.0
    for k in range(6):
        blk = np.array([[Sm[k, k], Sm[k, 6 + k]], [Sm[6 + k, k], Sm[6 + k, 6 + k]]])
        worst = max(worst, np.max(np.abs(blk - M[k])))
    assert worst < 1e-9, f"mode path vs full monodromy: {worst:.2e}"
    # exact square path via the batched map
    drs = modulated_K(K, 0.25, 1.9, "square")
    Ms, info_s = mode_monodromy(drs, w2)
    assert info_s["exact"]
    fs = floquet_map(drs)
    Sm = T @ fs.S @ T.T
    for k in range(6):
        blk = np.array([[Sm[k, k], Sm[k, 6 + k]], [Sm[6 + k, k], Sm[6 + k, 6 + k]]])
        assert np.max(np.abs(blk - Ms[k])) < 1e-12


def test_stability_chart_and_momentum_gaps():
    N, m = 16, 0.5
    K = harmonic_chain_K(N, m)
    omega_k = np.sqrt(np.linalg.eigvalsh(K))
    target = 1.5  # the k = pi/2 pair of this chain
    j = int(np.argmin(np.abs(omega_k - target)))
    chart = stability_chart(K, [0.1, 0.3], [2 * target, 2 * target * 1.3], n_steps=16,
                            conv_tol=1e-9, mass=m)
    assert chart.info["converged"] and chart.mode_multiplier.shape == (2, 2, N)
    bands = momentum_gaps(chart)
    b = bands[1][0]  # depth 0.3, omega_mod = 2 omega_k
    assert len(b) >= 1 and any(band.mode_lo <= j <= band.mode_hi for band in b), (
        f"resonant mode {j} (omega {omega_k[j]:.3f}) not in bands {b}"
    )
    assert abs(b[0].k_lo - np.pi / 2) < 1e-6
    # per-mode multiplier agrees with a direct single-mode floquet_map
    fm = floquet_map(modulated_K(np.array([[target**2]]), 0.3, 2 * target, "cos"),
                     n_steps=16, conv_tol=1e-11)
    assert abs(chart.mode_multiplier[1, 0, j] - fm.multiplier) < 1e-8, (
        f"chart {chart.mode_multiplier[1, 0, j]:.10f} vs single-mode {fm.multiplier:.10f}"
    )
    # the detuned column is stable for the same mode at small depth
    assert chart.mode_multiplier[0, 1, j] == pytest.approx(1.0)
