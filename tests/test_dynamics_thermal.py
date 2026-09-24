"""M-I shared-infrastructure tests (spec: docs/PLAN_LAYERS_2_3.md, section M-I).

I.1 — chain_propagator promoted to vacuum.core.dynamics, re-imported by
      vacuum.audits.causality: both import paths must resolve to the same
      function object, with verbatim behavior.
I.2 — thermal_state_cov(K, T): T = 0 reproduces ground_state_cov to 1e-12;
      mean_energy against the analytic Bose sum sum_k (omega_k/2) coth(omega_k/2T)
      to 1e-10.
"""

import numpy as np
import pytest

import vacuum.audits
import vacuum.audits.causality
import vacuum.core
import vacuum.core.dynamics
from vacuum.core import (
    Omega,
    chain_propagator,
    ground_state_cov,
    harmonic_chain_K,
    mean_energy,
    symplectic_eigenvalues,
    thermal_state_cov,
)


@pytest.fixture(scope="module")
def chain32():
    """N=32, m=1 periodic harmonic chain coupling matrix."""
    return harmonic_chain_K(32, m=1.0, bc="periodic")


@pytest.fixture(scope="module")
def chain12_open():
    """Small open chain with a different mass, for cross-checks."""
    return harmonic_chain_K(12, m=0.35, bc="dirichlet")


# ---------------------------------------------------------------------------
# I.1 — propagator promotion
# ---------------------------------------------------------------------------


def test_chain_propagator_same_object_both_paths():
    """vacuum.audits.chain_propagator is vacuum.core.chain_propagator (shim)."""
    assert vacuum.audits.chain_propagator is vacuum.core.chain_propagator, (
        "vacuum.audits.chain_propagator and vacuum.core.chain_propagator "
        "must be the same function object"
    )
    assert (
        vacuum.audits.causality.chain_propagator
        is vacuum.core.dynamics.chain_propagator
    ), "causality shim must re-export the vacuum.core.dynamics function object"


def test_chain_propagator_is_symplectic(chain32):
    """S Omega S^T = Omega to machine precision (spectral construction)."""
    N = chain32.shape[0]
    Om = Omega(N)
    S = chain_propagator(chain32, t=3.7)
    defect = float(np.max(np.abs(S @ Om @ S.T - Om)))
    assert defect < 1e-12, f"symplectic defect {defect:.3e}"


def test_chain_propagator_spectral_matches_expm(chain12_open):
    """'spectral' and 'expm' constructions agree (verbatim-behavior check)."""
    t = 1.9
    S_spec = chain_propagator(chain12_open, t, method="spectral")
    S_expm = chain_propagator(chain12_open, t, method="expm")
    dev = float(np.max(np.abs(S_spec - S_expm)))
    assert dev < 1e-10, f"spectral vs expm propagator deviation {dev:.3e}"


def test_chain_propagator_identity_at_t0(chain32):
    S0 = chain_propagator(chain32, t=0.0)
    dev = float(np.max(np.abs(S0 - np.eye(2 * chain32.shape[0]))))
    assert dev < 1e-14, f"S(0) deviates from identity by {dev:.3e}"


def test_chain_propagator_rejects_bad_inputs(chain32):
    with pytest.raises(ValueError):
        chain_propagator(chain32, 1.0, method="nope")
    with pytest.raises(ValueError):
        chain_propagator(-np.eye(4), 1.0)  # not positive definite


def test_causality_audit_still_green(chain32):
    """The audit that consumed the moved function still passes (no regression)."""
    res = vacuum.audits.causality_audit(chain32, t=4.0, buffer=8.0)
    assert res.passed, f"causality audit failed after I.1 move: {res.details}"


# ---------------------------------------------------------------------------
# I.2 — thermal states
# ---------------------------------------------------------------------------


def _bose_sum(K, T):
    """Analytic mean energy sum_k (omega_k/2) coth(omega_k/2T), hbar = k_B = 1."""
    w2 = np.linalg.eigh(0.5 * (K + K.T))[0]
    omega = np.sqrt(w2)
    if T == 0.0:
        coth = np.ones_like(omega)
    else:
        coth = 1.0 / np.tanh(omega / (2.0 * T))
    return float(np.sum(0.5 * omega * coth))


def test_thermal_T0_equals_ground_state(chain32, chain12_open):
    """T = 0 reproduces ground_state_cov to 1e-12 (spec I.2, coth limit)."""
    for K in (chain32, chain12_open):
        V0 = ground_state_cov(K)
        Vt = thermal_state_cov(K, 0.0)
        dev = float(np.max(np.abs(Vt - V0)))
        assert dev < 1e-12, f"T=0 thermal vs ground state max deviation {dev:.3e}"


def test_thermal_mean_energy_matches_bose_sum(chain32, chain12_open):
    """mean_energy(thermal_state_cov(K,T), K) = sum_k (omega_k/2)coth(omega_k/2T)."""
    for K in (chain32, chain12_open):
        for T in (0.0, 0.05, 0.3, 1.0, 4.0):
            E = mean_energy(thermal_state_cov(K, T), K)
            E_ref = _bose_sum(K, T)
            dev = abs(E - E_ref)
            assert dev < 1e-10, (
                f"T={T}: mean_energy {E:.12f} vs Bose sum {E_ref:.12f}, "
                f"|diff| = {dev:.3e}"
            )


def test_thermal_state_is_physical_and_stationary(chain32):
    """nu >= 1/2 at every T, and the thermal state is stationary under S(t)."""
    for T in (0.0, 0.2, 2.0):
        V = thermal_state_cov(chain32, T)
        nu_min = float(np.min(symplectic_eigenvalues(V)))
        assert nu_min >= 0.5 - 1e-12, f"T={T}: min nu {nu_min:.12f} below vacuum floor"
    # [H, rho_thermal] = 0: V is invariant under the free propagator.
    V = thermal_state_cov(chain32, 0.7)
    S = chain_propagator(chain32, t=2.3)
    dev = float(np.max(np.abs(S @ V @ S.T - V)))
    assert dev < 1e-10, f"thermal state not stationary: moved by {dev:.3e}"


def test_thermal_energy_monotone_in_T(chain12_open):
    """Mean energy grows with temperature (basic thermodynamic sanity)."""
    Ts = [0.0, 0.1, 0.5, 1.0, 3.0]
    Es = [mean_energy(thermal_state_cov(chain12_open, T), chain12_open) for T in Ts]
    assert all(b > a for a, b in zip(Es, Es[1:])), f"energies not increasing: {Es}"


def test_thermal_rejects_bad_inputs():
    with pytest.raises(ValueError):
        thermal_state_cov(harmonic_chain_K(4, m=1.0), -0.1)  # negative T
    with pytest.raises(ValueError):
        thermal_state_cov(-np.eye(3), 1.0)  # K not positive definite
