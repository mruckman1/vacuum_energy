"""Layer-1 validation anchor: Calabrese-Cardy central charge of the chain.

The ground state of the near-critical harmonic chain (free massless boson,
c = 1) obeys, for a block of l contiguous sites of a periodic chain of N,

    S(l) = (c/3) * ln[(N/pi) sin(pi l / N)] + c1'

(Calabrese-Cardy, JSTAT P06002 (2004), Eq. 3.8 chord length; the lattice
verification goes back to Vidal-Latorre-Rico-Kitaev, PRL 90, 227902 (2003)).
Fitting S(l) against the chord log gives slope c/3, so the anchor is
c = 3 * slope with |c - 1| <= 0.05 (measured here: |c - 1| ~ 4e-7).

Mass regulator (documented tuning)
----------------------------------
The exactly massless periodic chain has a zero mode (omega_0 = m -> 0), so
K = m^2 I + Laplacian is singular at m = 0 and ground_state_cov requires
m > 0. The regulator must satisfy m*N << 1 (finite-size criticality: the
correlation length 1/m must dwarf the chain) while keeping the covariance
finite (the zero mode puts entries ~ 1/(2*N*m) in the x-block). Measured
central charge vs m at N = 200, window l = 10..100:

    m = 1e-3 -> c = 0.9112      m = 1e-5 -> c = 0.99914
    m = 1e-4 -> c = 0.9906      m = 1e-6 -> c = 1.0000004  (chosen)
                                m = 1e-7 -> c = 1.00009

m = 1e-6 gives max |V| ~ 2.5e3 (comfortably finite in float64) and a fit
clean to ~3e-5 residuals. The fit window starts at l = 10 because the
smallest blocks carry the largest zero-mode / lattice contamination, and
ends at l = N/2 where the chord length peaks (S(l) = S(N-l) by purity).

Conventions: hbar = 1, block quadrature ordering, entropies in nats
(docs/API.md).
"""

import numpy as np
import pytest

from vacuum.core import (
    entropy,
    ground_state_cov,
    harmonic_chain_K,
    reduce,
    symplectic_eigenvalues,
)

N_SITES = 200
MASS = 1e-6  # zero-mode regulator, m*N = 2e-4 << 1 (see module docstring)
FIT_BLOCKS = np.arange(10, 101, 5)  # l-window away from zero-mode-dominated small l


@pytest.fixture(scope="module")
def critical_chain():
    """Near-critical periodic chain: coupling matrix and ground-state covariance."""
    K = harmonic_chain_K(N_SITES, m=MASS, bc="periodic")
    V = ground_state_cov(K)
    return K, V


@pytest.fixture(scope="module")
def block_entropies(critical_chain):
    """S(l) in nats for the fit window (cached: 20 reduced-state entropies)."""
    _, V = critical_chain
    return np.array([entropy(reduce(V, range(l))) for l in FIT_BLOCKS])


def test_regulated_covariance_finite_and_pure(critical_chain):
    """The mass regulator keeps V finite and the global state pure (S_tot ~ 0)."""
    _, V = critical_chain
    assert np.all(np.isfinite(V)), "regulated covariance has non-finite entries"
    vmax = float(np.max(np.abs(V)))
    assert vmax < 1e6, f"covariance blow-up: max |V_ij| = {vmax:.3e} (m too small)"

    nu_min = float(np.min(symplectic_eigenvalues(V)))
    s_tot = entropy(V)
    assert nu_min >= 0.5 - 1e-8, f"unphysical nu_min = {nu_min!r} < 1/2"
    assert s_tot < 1e-5, (
        f"ground state should be pure: S_total = {s_tot:.3e} nats "
        f"(nu_min = {nu_min:.12f})"
    )


def test_central_charge_calabrese_cardy(block_entropies):
    """THE ANCHOR: c = 3 * slope of S(l) vs ln[(N/pi) sin(pi l/N)], c -> 1.

    S(l) = (c/3) ln[(N/pi) sin(pi l/N)] + c1', so the slope against the
    chord log is c/3 (equivalently: slope 1 against x = (1/3) ln[chord]).
    """
    chord_log = np.log((N_SITES / np.pi) * np.sin(np.pi * FIT_BLOCKS / N_SITES))
    slope, intercept = np.polyfit(chord_log, block_entropies, 1)
    c = 3.0 * slope

    # Fit-quality guard: a contaminated window (zero mode, lattice effects)
    # shows up as structured residuals long before it moves c by 0.05.
    resid = block_entropies - (slope * chord_log + intercept)
    max_resid = float(np.max(np.abs(resid)))
    assert max_resid < 1e-3, (
        f"S(l) fit is not clean: max |residual| = {max_resid:.2e} nats "
        f"(measured c = {c:.6f})"
    )

    # Never widen this tolerance (task/PLAN.md): |c - 1| <= 0.05, aim 0.02.
    assert abs(c - 1.0) <= 0.05, (
        f"central charge anchor failed: measured c = {c:.6f} "
        f"(|c - 1| = {abs(c - 1.0):.2e}, slope = {slope:.6f}, "
        f"intercept c1' = {intercept:.6f}, N = {N_SITES}, m = {MASS}, "
        f"window l = {FIT_BLOCKS[0]}..{FIT_BLOCKS[-1]})"
    )
    print(
        f"\nCalabrese-Cardy anchor: c = {c:.6f} (target 1, |c-1| = {abs(c-1):.2e}), "
        f"c1' = {intercept:.4f}, max fit residual = {max_resid:.2e} nats"
    )


def test_entropy_symmetry_under_complement(critical_chain):
    """Purity check on the fit ingredients: S(l) = S(N - l) for the global pure state."""
    _, V = critical_chain
    for l in (25, 60, 90):
        s_l = entropy(reduce(V, range(l)))
        s_comp = entropy(reduce(V, range(l, N_SITES)))
        assert abs(s_l - s_comp) < 1e-6, (
            f"S({l}) = {s_l:.9f} vs S({N_SITES - l}) = {s_comp:.9f} nats: "
            f"complement symmetry broken (global state not pure?)"
        )
