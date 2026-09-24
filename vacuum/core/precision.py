"""Precision escalation ladder: float64 -> longdouble -> mpmath.

Symplectic eigenvalues near the vacuum floor nu = 1/2 are the danger zone
(PLAN.md, Layer 0 precision audit): entropy and modular-Hamiltonian formulas
contain ln(nu - 1/2), so relative error in nu - 1/2 — not in nu — is what
matters, and float64 eigen-solvers lose it exactly where the physics lives.

The ladder:
1. float64 (vacuum.core.gaussian, SVD of V^{1/2} Omega V^{1/2});
2. longdouble refinement of the matrix products, where the platform's
   longdouble is actually wider than float64 (it is not on darwin/arm64,
   in which case this rung is skipped);
3. mpmath at `dps` decimal digits on the i Omega V eigenproblem.

`near_vacuum_floor` is the guard the observable functions call; when it trips,
`symplectic_eigenvalues_precise` / `entropy_precise` escalate automatically.
"""

from __future__ import annotations

import numpy as np
import mpmath as mp

from .gaussian import Omega, entropy_from_nu, symplectic_eigenvalues

__all__ = [
    "near_vacuum_floor",
    "symplectic_eigenvalues_mp",
    "symplectic_margins_mp",
    "ground_state_cov_mp",
    "entropy_mp",
    "symplectic_eigenvalues_precise",
    "entropy_precise",
    "longdouble_refine",
]

VACUUM_NU = 0.5


def near_vacuum_floor(nu, tol=1e-10):
    """True if any symplectic eigenvalue is within tol of the vacuum floor 1/2.

    This is the precision-audit guard: min(nu) - 1/2 < tol means float64
    results for entropy-like quantities are no longer trustworthy and the
    caller should escalate.
    """
    nu = np.asarray(nu, dtype=float)
    return bool(np.min(nu) - VACUUM_NU < tol)


def _abs_eig_pairs_mp(V, dps):
    """Symplectic eigenvalues of V as a sorted (descending) list of mpf.

    Eigenvalues of i Omega V come in pairs +-nu; take moduli, sort descending,
    keep one of each pair.

    Accepts either a float ndarray or an ``mp.matrix``. An mp.matrix input is
    used verbatim (no float64 rounding), so covariances built natively in
    mpmath — e.g. by :func:`ground_state_cov_mp` — retain symplectic margins
    nu - 1/2 far below the ~1e-16 float64 representation floor.
    """
    with mp.workdps(dps):
        if isinstance(V, mp.matrix):
            n2 = V.rows
            N = n2 // 2
            M = mp.matrix(n2, n2)
            # (Omega V)[i, :] = V[N+i, :], (Omega V)[N+i, :] = -V[i, :]
            for i in range(N):
                for j in range(n2):
                    M[i, j] = mp.mpc(0, 1) * V[N + i, j]
                    M[N + i, j] = mp.mpc(0, -1) * V[i, j]
        else:
            Vf = np.asarray(V, dtype=float)
            n2 = Vf.shape[0]
            N = n2 // 2
            OmV = Omega(N) @ Vf  # row permutation + sign flips: exact in float64
            M = mp.matrix(n2, n2)
            for i in range(n2):
                for j in range(n2):
                    M[i, j] = mp.mpc(0, 1) * mp.mpf(float(OmV[i, j]))
        E = mp.eig(M, left=False, right=False)
        mags = sorted((abs(e) for e in E), reverse=True)
        return [mags[2 * k] for k in range(N)]


def symplectic_eigenvalues_mp(V, dps=50):
    """Symplectic eigenvalues via mpmath at `dps` digits, descending.

    Solves the i Omega V eigenproblem in arbitrary precision so nu - 1/2 is
    resolved far below float64 roundoff; returns float64 values (the precision
    gain lives in the *computation*; use entropy_mp for fully-mp observables,
    or symplectic_margins_mp when the margins nu - 1/2 themselves are below
    the float64 spacing at 1/2, ~1.1e-16).

    Accepts an ndarray (rounded to float64) or an ``mp.matrix`` covariance
    (used verbatim — no rounding; pair with ground_state_cov_mp).
    """
    return np.array([float(nu) for nu in _abs_eig_pairs_mp(V, dps)])


def symplectic_margins_mp(V, dps=50):
    """Margins nu_k - 1/2 above the vacuum floor, descending in nu, via mpmath.

    The margin of a near-floor mode (e.g. the partial-transpose spectrum of a
    weakly entangled Gaussian state, Serafini 'Quantum Continuous Variables'
    ch. 7) can be ~1e-30 — representable in float64 even though 1/2 + 1e-30
    is not. Computing nu - 1/2 inside mpmath and returning the *difference*
    as float64 preserves it. Pass an ``mp.matrix`` covariance (e.g. from
    ground_state_cov_mp) to avoid any float64 rounding of V itself.
    """
    nus = _abs_eig_pairs_mp(V, dps)
    with mp.workdps(dps):
        half = mp.mpf(1) / 2
        return np.array([float(nu - half) for nu in nus])


def ground_state_cov_mp(K, dps=50):
    """Ground-state covariance of H = (1/2)(p.p + x.K.x) as an mp.matrix.

    mp-native counterpart of vacuum.core.gaussian.ground_state_cov:
    V = blockdiag(K^{-1/2}/2, K^{1/2}/2) (Srednicki, PRL 71, 666 (1993)),
    computed via an mpmath symmetric eigendecomposition of K so that
    symplectic margins nu - 1/2 survive below the float64 floor (the float64
    route loses them when V is rounded). Accepts a float ndarray or an
    ``mp.matrix`` K; returns an ``mp.matrix`` in block quadrature ordering.
    O(N^3) mp arithmetic — intended for modest N (lattice blocks, not full
    chains at N ~ hundreds unless you can wait).
    """
    with mp.workdps(dps):
        if isinstance(K, mp.matrix):
            N = K.rows
            M = (K + K.T) / 2
        else:
            Kf = np.asarray(K, dtype=float)
            N = Kf.shape[0]
            M = mp.matrix(N, N)
            for i in range(N):
                for j in range(N):
                    M[i, j] = (mp.mpf(float(Kf[i, j])) + mp.mpf(float(Kf[j, i]))) / 2
        E, Q = mp.eigsy(M)
        if min(E) <= 0:
            raise ValueError("K must be positive definite; min eigenvalue "
                             f"{float(min(E))!r} <= 0")
        half = mp.mpf(1) / 2
        roots = [mp.sqrt(E[k]) for k in range(N)]
        V = mp.matrix(2 * N, 2 * N)
        for a in range(N):
            for b in range(a, N):
                xx = mp.mpf(0)
                pp = mp.mpf(0)
                for k in range(N):
                    qq = Q[a, k] * Q[b, k]
                    xx += qq / roots[k]
                    pp += qq * roots[k]
                V[a, b] = V[b, a] = half * xx
                V[N + a, N + b] = V[N + b, N + a] = half * pp
        return V


def entropy_mp(V, dps=50):
    """Von Neumann entropy (nats) computed end-to-end in mpmath.

    S = sum_k [(nu+1/2)ln(nu+1/2) - (nu-1/2)ln(nu-1/2)] evaluated with mpf
    arithmetic; modes with nu <= 1/2 (vacuum floor, incl. roundoff below it)
    contribute exactly 0.
    """
    nus = _abs_eig_pairs_mp(V, dps)
    with mp.workdps(dps):
        half = mp.mpf(1) / 2
        total = mp.mpf(0)
        for nu in nus:
            y = nu - half
            if y <= 0:
                continue  # vacuum mode: (nu+1/2)ln(nu+1/2) - y ln y -> 0
            x = nu + half
            total += x * mp.log(x) - y * mp.log(y)
        return float(total)


def _longdouble_refine(V):
    """Longdouble rung: recompute the SVD input products in extended precision.

    LAPACK offers no longdouble eigensolver, so the refinement recomputes the
    matrix products of the float64 route (V^{1/2} Omega V^{1/2}) in
    longdouble before rounding back for the SVD, trimming accumulation error.
    Only meaningful where longdouble is wider than float64.
    """
    V = 0.5 * (np.asarray(V, dtype=np.longdouble) + np.asarray(V, dtype=np.longdouble).T)
    N = V.shape[0] // 2
    w, U = np.linalg.eigh(V.astype(float))
    w_ld = np.clip(w.astype(np.longdouble), 0.0, None)
    U_ld = U.astype(np.longdouble)
    root = (U_ld * np.sqrt(w_ld)) @ U_ld.T
    A = root @ Omega(N).astype(np.longdouble) @ root
    A64 = (0.5 * (A - A.T)).astype(float)
    s = np.linalg.svd(A64, compute_uv=False)
    return np.sort(s)[::-1][::2].copy()


# Public name for the longdouble rung, so downstream modules (vacuum.audits)
# do not need to reach for the private helper. Kept as an alias so both names
# refer to the same implementation.
longdouble_refine = _longdouble_refine


def symplectic_eigenvalues_precise(V, tol=1e-10, dps=50):
    """Escalation-ladder symplectic eigenvalues, descending.

    float64 first; if `near_vacuum_floor(nu, tol)` trips, escalate through
    longdouble (where available) to mpmath at `dps` digits. A spectrum that is
    genuinely at the floor (an actual vacuum) still trips the guard — the
    ladder then simply returns the most precise values available.
    """
    nu = symplectic_eigenvalues(V)
    if not near_vacuum_floor(nu, tol):
        return nu
    if np.finfo(np.longdouble).eps < np.finfo(np.float64).eps:
        nu = _longdouble_refine(V)
        if not near_vacuum_floor(nu, tol):
            return nu
    return symplectic_eigenvalues_mp(V, dps=dps)


def entropy_precise(V, tol=1e-10, dps=50):
    """Entropy (nats) with automatic precision escalation near the floor."""
    nu = symplectic_eigenvalues(V)
    if not near_vacuum_floor(nu, tol):
        return entropy_from_nu(nu)
    return entropy_mp(V, dps=dps)
