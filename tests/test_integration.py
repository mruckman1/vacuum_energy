"""Integration-layer tests: package exports and the mp-native covariance path.

These tests pin the cross-module contracts added during integration:

1. vacuum.core re-exports the radial (Srednicki) module and the public
   longdouble rung, so downstream layers never import private helpers.
2. The mp-native covariance path (ground_state_cov_mp / symplectic_margins_mp,
   vacuum.core.precision) resolves symplectic margins nu - 1/2 far below the
   float64 representation floor at 1/2 (~1.1e-16) — the capability the
   negativity-sudden-death anchor (Calabrese-Cardy-Tonni region) needs, so
   other modules do not have to rebuild covariances in mpmath by hand.

Conventions per docs/API.md: hbar = 1, vacuum nu = 1/2, block ordering
R = (x_1..x_N, p_1..p_N), entropies in nats.
"""

import warnings

import mpmath as mp
import numpy as np

from vacuum.core import (
    ground_state_cov,
    ground_state_cov_mp,
    harmonic_chain_K,
    longdouble_refine,
    partial_wave_entropies,
    sphere_entropy,
    srednicki_K,
    symplectic_eigenvalues,
    symplectic_eigenvalues_mp,
    symplectic_margins_mp,
)


def test_radial_package_exports():
    """srednicki_K/sphere_entropy are reachable from vacuum.core (docs/API.md)."""
    K = srednicki_K(20, 0)
    assert K.shape == (20, 20)
    assert np.allclose(K, K.T)
    with warnings.catch_warnings():
        # the documented l-sum truncation cap warning is expected here
        warnings.simplefilter("ignore", RuntimeWarning)
        S5 = sphere_entropy(20, 5)
    assert np.isfinite(S5) and S5 > 0.0
    (S5_pw,) = partial_wave_entropies(srednicki_K(20, 0), [5])
    assert np.isfinite(S5_pw) and S5_pw >= 0.0


def test_longdouble_refine_public_name():
    """Public alias exists and matches the float64 route away from the floor."""
    from vacuum.core import precision as core_precision

    assert core_precision.longdouble_refine is core_precision._longdouble_refine
    K = harmonic_chain_K(6, m=1.5)
    V = ground_state_cov(K) + np.diag([0.3] * 6 + [0.0] * 6)  # off the floor
    nu_ld = np.asarray(longdouble_refine(V), dtype=float)
    nu_64 = symplectic_eigenvalues(V)
    assert np.max(np.abs(np.sort(nu_ld) - np.sort(nu_64))) < 1e-10


def test_ground_state_cov_mp_matches_float64():
    """mp-native ground-state covariance agrees entrywise with the float64 route."""
    K = harmonic_chain_K(6, m=1.0)
    V64 = ground_state_cov(K)
    Vmp = ground_state_cov_mp(K, dps=40)
    n2 = 12
    Vmp_f = np.array([[float(Vmp[i, j]) for j in range(n2)] for i in range(n2)])
    err = np.max(np.abs(Vmp_f - V64))
    assert err < 1e-13, f"mp vs float64 ground-state covariance differ by {err:.3e}"


def test_mp_native_margins_resolve_below_float64_floor():
    """The chain vacuum sits exactly on nu = 1/2; mp-native margins show it.

    The float64 route leaves ~1e-16 noise around the floor; the mp-native
    route (mp.matrix in, difference out) must land below 1e-30.
    """
    K = harmonic_chain_K(5, m=2.0)
    Vmp = ground_state_cov_mp(K, dps=40)
    margins = symplectic_margins_mp(Vmp, dps=40)
    assert margins.shape == (5,)
    worst = float(np.max(np.abs(margins)))
    assert worst < 1e-30, f"mp-native vacuum margin {worst:.3e} not below 1e-30"
    # and the same mp.matrix feeds symplectic_eigenvalues_mp directly
    nus = symplectic_eigenvalues_mp(Vmp, dps=40)
    assert np.allclose(nus, 0.5, atol=1e-15)


def test_weak_tmsv_margin_is_r_squared():
    """Two-mode squeezed vacuum, r = 1e-10: reduced-mode margin = r^2 + O(r^4).

    nu_A = cosh(2r)/2, so nu_A - 1/2 = (cosh 2r - 1)/2 ~ r^2 = 1e-20 — four
    orders below the float64 spacing at 1/2, hence invisible to the float64
    route but exact for the mp-native one (Serafini, Quantum Continuous
    Variables, ch. 7 conventions with hbar = 1, V_vac = I/2).
    """
    r_val = 1e-10
    with mp.workdps(60):
        r = mp.mpf("1e-10")
        c = mp.cosh(2 * r)
        V_A = mp.matrix(2, 2)
        V_A[0, 0] = c / 2  # x-x
        V_A[1, 1] = c / 2  # p-p
    (margin,) = symplectic_margins_mp(V_A, dps=60)
    rel = abs(margin / r_val**2 - 1.0)
    assert rel < 1e-6, f"margin {margin:.6e} vs r^2 {r_val**2:.6e}, rel err {rel:.2e}"
    # The float64 route on the rounded covariance cannot see this margin:
    # whatever it reports near 1/2 is pure roundoff (|noise| up to ~1e-15),
    # 1e5+ times larger than the true 1e-20 signal.
    V64 = np.array([[float(V_A[0, 0]), 0.0], [0.0, float(V_A[1, 1])]])
    nu64 = symplectic_eigenvalues(V64)[0]
    assert abs(nu64 - 0.5) < 1e-14  # indistinguishable from exact vacuum
