"""Layer-1 validation anchor: sudden death of two-block logarithmic negativity.

Two disjoint blocks of the harmonic-chain ground state lose ALL logarithmic
negativity at a finite separation d_death — E_N(d) is exactly zero beyond it,
not merely small. This is the quantitative statement of how scarce
harvestable vacuum entanglement is (PLAN.md Layer 1; Audenaert-Eisert-Plenio-
Werner, PRA 66, 042327 (2002); Marcovitch-Retzker-Plenio-Reznik, PRA 80,
012325 (2009); Calabrese-Cardy-Tonni, PRL 109, 130502 (2012) for the CFT
scaling of two-interval negativity).

Chain-parameter choice (documented tuning, all measured with 40-digit
arithmetic on N = 200, blocks of L = 10)
----------------------------------------
The death separation tracks the correlation length xi = 1/m of the chain
family K = m^2 I + Laplacian:

    m <= 1e-4 (near-critical): no death for d <= 30 (min PT nu still
               ~2e-5 below 1/2 at d = 30)
    m = 0.8 -> d_death = 22      m = 2.0 -> d_death = 15
    m = 1.0 -> d_death = 20      m = 3.0 -> d_death = 14  (chosen)
    m = 1.2 -> d_death = 18      m = 4.0 -> d_death = 13

so placing the death point inside the required d <= 15 window for 10-site
blocks needs a gapped member of the family; m = 3 puts it at d_death = 14
with a genuinely positive PT margin (min nu - 1/2 ~ +1.8e-30) for every
dead separation.

Why the curve is computed in mpmath (precision-audit escalation)
----------------------------------------------------------------
The PT symplectic spectrum approaches the vacuum floor nu = 1/2
exponentially in d, and beyond d ~ 5 the float64 pipeline sits in the
PLAN.md "danger zone": SVD roundoff leaves |nu - 1/2| noise of ~5e-15, so
float64 log_negativity returns junk ~1e-14 instead of the true values
(1e-20..0). Escalation cannot reuse vacuum.core.precision here, because
symplectic_eigenvalues_mp takes an already-rounded float64 covariance; the
periodic chain, however, is circulant, so the exact covariance is a Fourier
sum evaluable directly in mpmath:

    <x_i x_j> = (1/2N) sum_k cos(2 pi k (i-j)/N) / omega_k
    <p_i p_j> = (1/2N) sum_k cos(2 pi k (i-j)/N) * omega_k
    omega_k^2 = m^2 + 4 sin^2(pi k / N)

(Srednicki-normalization dispersion; V is x-p block diagonal, so after the
momentum-sign-flip partial transpose on B the symplectic spectrum is
nu_i = sqrt(eig(X^{1/2} P~ X^{1/2}))). The float64 vacuum.core route is
cross-validated against this exact route at the separations where float64
is trustworthy.

Conventions: hbar = 1, block quadrature ordering, negativities in nats
(docs/API.md).
"""

import numpy as np
import mpmath as mp
import pytest

from vacuum.core import (
    ground_state_cov,
    harmonic_chain_K,
    log_negativity,
    reduce,
)

N_SITES = 200
MASS = 3.0  # gapped member of the chain family: d_death = 14 (module docstring)
L_BLOCK = 10
SEPARATIONS = np.arange(0, 31)  # d = 0..30, d = 0 means adjacent blocks
DPS = 40  # resolves PT margins ~1e-30 with ~8 digits of headroom
MONO_SLACK = 1e-12


def _blocks(d):
    """Site lists of the two L_BLOCK-site blocks separated by d empty sites."""
    A = list(range(0, L_BLOCK))
    B = list(range(L_BLOCK + d, 2 * L_BLOCK + d))
    return A, B


def _mp_chain_correlators(N, m, r_values, dps):
    """Exact PBC ground-state correlators <x_0 x_r>, <p_0 p_r> as mpf dicts."""
    with mp.workdps(dps):
        mm = mp.mpf(m)
        w = [mp.sqrt(mm**2 + 4 * mp.sin(mp.pi * k / N) ** 2) for k in range(N)]
        step = 2 * mp.pi / N
        xc, pc = {}, {}
        for r in sorted(r_values):
            sx = mp.mpf(0)
            sp = mp.mpf(0)
            for k in range(N):
                c = mp.cos(step * ((k * r) % N))
                sx += c / w[k]
                sp += c * w[k]
            xc[r] = sx / (2 * N)
            pc[r] = sp / (2 * N)
    return xc, pc


def _mp_block_negativity(N, m, sites_A, sites_B, dps=DPS):
    """(E_N in nats, min PT nu - 1/2) for two blocks, exact-to-dps arithmetic.

    Builds the reduced X and P Toeplitz blocks from the circulant Fourier
    sums, applies the partial transpose as the momentum sign flip on B
    (docs/API.md convention), and takes nu = sqrt(eig(X^{1/2} P~ X^{1/2})),
    valid because the ground-state V is x-p block diagonal.
    E_N is exactly 0.0 iff every PT symplectic eigenvalue is >= 1/2.
    """
    sites = list(sites_A) + list(sites_B)
    n_A = len(sites_A)
    n = len(sites)
    dists = {min(abs(i - j), N - abs(i - j)) for i in sites for j in sites}
    xc, pc = _mp_chain_correlators(N, m, dists, dps)
    with mp.workdps(dps):
        X = mp.matrix(n, n)
        Pt = mp.matrix(n, n)
        for a in range(n):
            for b in range(n):
                r = min(abs(sites[a] - sites[b]), N - abs(sites[a] - sites[b]))
                X[a, b] = xc[r]
                flip = -1 if (a < n_A) != (b < n_A) else 1  # PT: p_B -> -p_B
                Pt[a, b] = flip * pc[r]
        ex, Ux = mp.eigsy(X)
        Xhalf = Ux * mp.diag([mp.sqrt(e) for e in ex]) * Ux.T
        lam = mp.eigsy(Xhalf * Pt * Xhalf, eigvals_only=True)
        half = mp.mpf(1) / 2
        e_n = mp.mpf(0)
        min_margin = None
        for l in lam:
            nu = mp.sqrt(l)
            margin = nu - half
            if min_margin is None or margin < min_margin:
                min_margin = margin
            if nu < half:
                e_n -= mp.log(2 * nu)
        return float(e_n), float(min_margin)


@pytest.fixture(scope="module")
def chain():
    """float64 library route: coupling matrix and ground-state covariance."""
    K = harmonic_chain_K(N_SITES, m=MASS, bc="periodic")
    V = ground_state_cov(K)
    return K, V


@pytest.fixture(scope="module")
def negativity_curve():
    """E_N(d) and min PT margin for d = 0..30, exact mpmath route (cached)."""
    e_n = np.empty(SEPARATIONS.size)
    margin = np.empty(SEPARATIONS.size)
    for i, d in enumerate(SEPARATIONS):
        A, B = _blocks(int(d))
        e_n[i], margin[i] = _mp_block_negativity(N_SITES, MASS, A, B)
    return e_n, margin


def test_exact_route_matches_library_covariance(chain):
    """The Fourier-sum covariance agrees with ground_state_cov entry by entry."""
    _, V = chain
    A, B = _blocks(5)
    sites = A + B
    n = len(sites)
    V_red = reduce(V, sites)
    xc, pc = _mp_chain_correlators(N_SITES, MASS, range(N_SITES // 2 + 1), 30)
    for a in range(n):
        for b in range(n):
            r = min(abs(sites[a] - sites[b]), N_SITES - abs(sites[a] - sites[b]))
            dx = abs(V_red[a, b] - float(xc[r]))
            dp = abs(V_red[n + a, n + b] - float(pc[r]))
            assert dx < 1e-12 and dp < 1e-12, (
                f"covariance mismatch at sites ({sites[a]},{sites[b]}): "
                f"dX = {dx:.2e}, dP = {dp:.2e}"
            )
            assert abs(V_red[a, n + b]) < 1e-14, "ground state should have <xp> = 0"


def test_library_negativity_agrees_where_float64_is_trustworthy(chain, negativity_curve):
    """vacuum.core.log_negativity matches the exact route above the noise floor.

    Measured float64 relative error: ~1e-13 at d = 0,1 (E_N ~ 1e-2..1e-3) and
    ~3e-7 at d = 2,3 (E_N ~ 1e-7..1e-9, PT margin approaching the SVD floor).
    """
    _, V = chain
    e_n_mp, _ = negativity_curve
    for d, rtol in ((0, 1e-9), (1, 1e-9), (2, 1e-4), (3, 1e-4)):
        A, B = _blocks(d)
        e64 = log_negativity(V, A, B)
        rel = abs(e64 - e_n_mp[d]) / e_n_mp[d]
        assert rel < rtol, (
            f"float64 log_negativity off at d={d}: {e64:.9e} vs exact "
            f"{e_n_mp[d]:.9e} (rel {rel:.2e} > {rtol})"
        )


def test_adjacent_blocks_entangled(chain, negativity_curve):
    """(a) E_N at d = 0 is strictly positive and of sane magnitude (> 0.01 nats)."""
    _, V = chain
    e_n_mp, _ = negativity_curve
    A, B = _blocks(0)
    e0_lib = log_negativity(V, A, B)
    assert e_n_mp[0] > 0.01 and e0_lib > 0.01, (
        f"adjacent 10-site blocks should be entangled: E_N(0) = {e_n_mp[0]:.6f} "
        f"nats (library float64: {e0_lib:.6f})"
    )


def test_negativity_nonincreasing(negativity_curve):
    """(b) E_N(d) is non-increasing in d (up to 1e-12 numerical slack)."""
    e_n, _ = negativity_curve
    diffs = np.diff(e_n)
    worst = float(np.max(diffs))
    assert worst <= MONO_SLACK, (
        f"E_N(d) not monotone: max increase {worst:.3e} at "
        f"d = {int(SEPARATIONS[1 + int(np.argmax(diffs))])}; curve = "
        + np.array2string(e_n, formatter={'float_kind': lambda v: f'{v:.3e}'})
    )


def test_sudden_death(negativity_curve):
    """(c)+(d) finite death separation d_death <= 15; E_N exactly 0 beyond it."""
    e_n, margin = negativity_curve
    curve_str = ", ".join(
        f"d={int(d)}: {e:.3e}" for d, e in zip(SEPARATIONS, e_n)
    )
    print("\nE_N(d) curve (nats, exact-to-40-digit route):\n  " + curve_str)

    dead = np.flatnonzero(e_n == 0.0)
    assert dead.size > 0, (
        "no sudden death observed for d = 0..30; E_N(d) curve: " + curve_str
    )
    d_death = int(SEPARATIONS[dead[0]])
    print(
        f"sudden death at d_death = {d_death} "
        f"(min PT margin beyond death: {np.min(margin[dead[0]:]):+.2e})"
    )

    assert d_death <= 15, (
        f"death separation d_death = {d_death} > 15; E_N(d) curve: " + curve_str
    )
    tail = e_n[dead[0]:]
    assert np.all(tail == 0.0), (
        f"E_N revives after d_death = {d_death}: tail = {tail!r}; "
        f"curve: " + curve_str
    )
    # Genuine death, not underflow: every dead separation has its full PT
    # spectrum strictly above the vacuum floor nu = 1/2.
    min_dead_margin = float(np.min(margin[dead[0]:]))
    assert min_dead_margin > 0.0, (
        f"dead separations touch the vacuum floor: min PT margin "
        f"{min_dead_margin:.3e} (d_death = {d_death})"
    )
    # And every live separation is genuinely live (PT eigenvalue below 1/2).
    assert np.all(margin[: dead[0]] < 0.0), (
        "a 'live' separation has no PT eigenvalue below 1/2; curve: " + curve_str
    )
