"""Validation anchor: Srednicki area law for the 3D massless scalar.

M. Srednicki, "Entropy and area", PRL 71, 666 (1993): tracing the vacuum of
the massless scalar field outside (equivalently: inside) a sphere of radius
R = (n + 1/2)a on an N-site radial lattice gives S = c_A (R/a)^2 with
c_A ~= 0.30 in lattice units; the refined coefficient is ~0.295
(Lohmayer-Neuberger-Schwimmer-Theisen, Phys. Lett. B 685, 222 (2010)).

N = 60, fit window n = 10..30 (away from both the origin and the outer
Dirichlet wall). The l-sweep runs once in a module-scoped fixture; all block
entropies for one l come from a single ground-state covariance. Entropies in
nats (hbar = 1, nu_vac = 1/2, docs/API.md).
"""

import warnings

import numpy as np
import pytest

from vacuum.core import gaussian as g
from vacuum.core.radial import partial_wave_entropies, sphere_entropy, srednicki_K

N_LATTICE = 60
FIT_NS = np.arange(10, 31)  # inner block sizes in the fit window


@pytest.fixture(scope="module")
def sphere_profile():
    """S(n) for n = 10..30 at N = 60: one adaptive l-sweep for all n."""
    with warnings.catch_warnings():
        # The 1e-8 tail tolerance is unreachable below the l cap (the tail
        # decays like a power of l); the truncation is ~0.1% and diagnosed.
        warnings.simplefilter("ignore", RuntimeWarning)
        S, diag = sphere_entropy(N_LATTICE, FIT_NS, return_diagnostics=True)
    return S, diag


def _linear_fit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1.0 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)
    return slope, intercept, r2


def test_area_law_slope(sphere_profile):
    """THE ANCHOR: S(n) vs R^2 = (n + 1/2)^2 is linear with slope ~0.295."""
    S, _ = sphere_profile
    R2 = (FIT_NS + 0.5) ** 2
    slope, intercept, r2 = _linear_fit(R2, S)
    assert 0.28 <= slope <= 0.32, (
        f"area-law coefficient out of range: measured slope = {slope:.6f} "
        f"(Srednicki 1993: 0.30; refined ~0.295), intercept = {intercept:.4f}"
    )
    assert r2 > 0.9999, (
        f"area-law fit not linear enough: R^2 = {r2:.8f} "
        f"(slope = {slope:.6f}, intercept = {intercept:.4f})"
    )


def test_entropies_positive_and_growing(sphere_profile):
    """S(n) is positive and monotone in n across the fit window."""
    S, _ = sphere_profile
    assert np.all(S > 0.0)
    assert np.all(np.diff(S) > 0.0), "S(n) must grow with the sphere radius"


def test_l_sweep_diagnostics(sphere_profile):
    """The adaptive l-sum reports its truncation honestly."""
    _, diag = sphere_profile
    assert diag["l_stop"] >= 300, "l-sum stopped implausibly early"
    assert diag["max_tail_term"] < 1e-3, (
        f"tail term at l_stop={diag['l_stop']} is {diag['max_tail_term']:.3e}; "
        "truncation would distort the area coefficient"
    )
    if not diag["converged"]:
        assert diag["l_stop"] == diag["cap"], "unconverged sum must have hit the cap"
    # terms decay by orders of magnitude across the sweep
    terms = diag["terms"]
    assert terms[-1] < 1e-4 * terms.max()


def test_srednicki_K_matches_independent_expansion():
    """Rebuild K_l from the sum-of-squares form of H_l and compare exactly.

    H_l's potential = (1/2) sum_j [l(l+1)/j^2] phi_j^2
                    + (1/2) sum_{j=1}^{N} (j+1/2)^2 (phi_{j+1}/(j+1) - phi_j/j)^2
    with phi_{N+1} = 0 (Srednicki PRL 71, 666 (1993), Eq. 10). Assembling
    K = sum of outer products of the difference vectors must reproduce
    srednicki_K entry for entry.
    """
    N, l = 17, 3
    K_ref = np.diag([l * (l + 1) / j**2 for j in range(1, N + 1)]).astype(float)
    for j in range(1, N + 1):  # coupling term j, with phi_{N+1} = 0
        v = np.zeros(N)
        v[j - 1] = -1.0 / j
        if j < N:
            v[j] = 1.0 / (j + 1)
        K_ref += (j + 0.5) ** 2 * np.outer(v, v)
    K = srednicki_K(N, l)
    np.testing.assert_allclose(K, K_ref, rtol=0, atol=1e-13)

    # spot-check the closed-form entries
    assert K[0, 0] == pytest.approx(9 / 4 + l * (l + 1))
    j = 5
    assert K[j - 1, j - 1] == pytest.approx(
        ((j - 0.5) ** 2 + (j + 0.5) ** 2) / j**2 + l * (l + 1) / j**2
    )
    assert K[j - 1, j] == pytest.approx(-((j + 0.5) ** 2) / (j * (j + 1)))
    # outer boundary: Srednicki's phi_{N+1} = 0 keeps the (N+1/2)^2/N^2 piece
    # (the free-boundary variant without it is singular at l = 0).
    assert K[N - 1, N - 1] == pytest.approx(
        ((N - 0.5) ** 2 + (N + 0.5) ** 2) / N**2 + l * (l + 1) / N**2
    )
    w = np.linalg.eigvalsh(srednicki_K(N, 0))
    assert w.min() > 0.0, "K_0 must be positive definite (Dirichlet outer wall)"


def test_partial_wave_entropies_match_gaussian_core():
    """Fast X-P block path == generic gaussian.reduce/entropy path."""
    for l in (0, 1, 7):
        K = srednicki_K(40, l)
        V = g.ground_state_cov(K)
        for n in (4, 13, 27):
            fast = partial_wave_entropies(K, [n])[0]
            slow = g.entropy(g.reduce(V, range(n)))
            assert fast == pytest.approx(slow, abs=1e-10), (l, n)
            # purity: complement block carries the same entropy
            comp = g.entropy(g.reduce(V, range(n, 40)))
            assert fast == pytest.approx(comp, abs=1e-9), (l, n)


def test_sphere_entropy_scalar_and_cap_warning():
    """Scalar n returns a float; hitting the l cap warns and flags it."""
    with pytest.warns(RuntimeWarning, match="hit the cap"):
        S, diag = sphere_entropy(30, 8, l_max=50, return_diagnostics=True)
    assert isinstance(S, float) and S > 0.0
    assert diag["converged"] is False and diag["l_stop"] == 50
