"""Two-point-measurement work statistics for Gaussian quenches and drives (M3.5).

Framework (equation-cited; arXiv:cond-mat/0703189 fetched 2026-09-01)
---------------------------------------------------------------------
P. Talkner, E. Lutz, P. Hanggi, Phys. Rev. E 75, 050102(R) (2007),
"Fluctuation theorems: Work is not an observable" (TLH below):

- Eq. (2): the work characteristic function is the Fourier transform of the
  work density, G(u) = Int dw e^{iuw} p(w).
- Eq. (3): G(u) = Tr[e^{iu H_H(t_f)} e^{-iu H(0)} rho(0)], with
  H_H(t_f) = U^dag H(t_f) U the final Hamiltonian in the Heisenberg picture
  and rho(0) the initial (Gibbs) state.
- Eq. (5): Jarzynski, <e^{-beta w}> = Z(t_f)/Z(0) = e^{-beta DeltaF}.
- Eq. (11): the two-point-measurement (TPM) form,
  G(u) = sum_{n,m} e^{iu(e_m(t_f)-e_n(0))} |<phi_m(t_f)|U|phi_n(0)>|^2
         e^{-beta e_n(0)}/Z(0).

Two routes, cross-gated (spec M3.5: dense <-> Gaussian agreement 1e-8
before the Gaussian route is trusted at any N).

Primary (Gaussian) route: closed-form characteristic function
-------------------------------------------------------------
All Hamiltonians here are the repo's standard quadratic family
H(K) = (1/2)(p.p + x.K.x) = (1/2) R^T M(K) R with M(K) = blockdiag(K, I)
in block ordering R = (x_1..x_N, p_1..p_N), hbar = 1 (docs/API.md). The
protocol unitary U (piecewise-constant drive) has an exact symplectic
representative S: U^dag R U = S R with S = S_n...S_1 and
S_k = chain_propagator(K_k, dt_k) = e^{dt_k Omega M(K_k)}, so
H_H(t_f) = U^dag H(K1) U = (1/2) R^T (S^T M(K1) S) R.

TLH Eq. (3) is then a trace of a product of exponentials of quadratic boson
forms. For such products (cf. R. Balian and E. Brezin, Nuovo Cimento B 64,
37 (1969), the algebra of exponentials of quadratic boson forms; the
specific normalization used here is DERIVED in this docstring and
cross-validated against the dense TPM route to 1e-8 in
tests/test_work_stats.py -- no equation number is cited from Balian-Brezin):

    Tr[ prod_j e^{-i z_j Hhat_j} ] = +- det( prod_j e^{z_j Omega M_j} - I )^{-1/2},

where e^{-it Hhat} has Heisenberg action R_H(t) = e^{t Omega M} R (Serafini
2017, Sec. 3.1, as used throughout vacuum.core), analytically continued in
t. Single-mode check fixing the normalization: for Hhat = w(a^dag a + 1/2),
M = w I_2, z = -i beta: e^{-i beta Omega M} = cosh(beta w) I - i sinh(beta w) J
(J = [[0,1],[-1,0]]), det(e^{-i beta Omega M} - I) = 2 - 2 cosh(beta w)
= -4 sinh^2(beta w / 2), whose inverse square root is +-1/(2 sinh(beta w/2))
= +-Tr e^{-beta Hhat}. The sign/branch ambiguity of the square root is fixed
by CONTINUITY along a path from u = 0, where the ratio below is exactly 1.

With M0 = M(K0), M1 = M(K1), M1_H = S^T M1 S, and using
Omega S^T = S^{-1} Omega (symplectic S), the TLH Eq. (3) trace gives

    G(u) = [ det(T(0) - I) / det(T(u) - I) ]^{1/2},
    T(u) = S^{-1} e^{-u Omega M1} S  e^{(u - i beta) Omega M0},

with T(0) = e^{-i beta Omega M0} reproducing 1/Z0. The branch of the square
root is tracked by phase-unwrapping log det(T(u) - I) along a fine path
anchored at u = 0 (path step chosen against the phase-rate bound
|d arg det / du| <~ sum_k (w0_k + w1_k); tested by step-halving).

TPM moments on the Gaussian route are exact Wick traces: for zero-mean
Gaussian rho with covariance V and G_mat = V + (i/2) Omega (so that
<R_i R_j> = G_mat_ij), quadratic forms Q_A = (1/2) R^T A R with symmetric
A, B obey

    <Q_A> = (1/2) Tr(A V),
    <Q_A Q_B> - <Q_A><Q_B> = (1/2) Tr(A G_mat B G_mat^T)

(Wick's theorem on the three pairings of <R R R R>; the two crossed pairings
are equal for symmetric A, B). Since the Gibbs initial state commutes with
H(0), the TPM second moment is <W^2> = <Q1^2> + <Q0^2> - 2 Re<Q0 Q1> with
Q0 = Hhat_0, Q1 = Hhat_1^H (TLH Eq. (11) expanded; the cross term is real
because rho commutes with H(0)), hence

    var(W) = c(Q1,Q1) + c(Q0,Q0) - 2 Re c(Q0,Q1).

kappa_1 and kappa_2 therefore have closed forms; kappa_3 and kappa_4 come
from Richardson-extrapolated central differences of ln G(u) at u = 0
(:func:`gaussian_work_cumulants`). The differencing is validated by turning
it on kappa_1, kappa_2 where the exact answer is known (agreement <= 1.2e-7
relative) and against the dense route for kappa_3, kappa_4 (<= 5.2e-6).

Single-mode analytic anchor (arXiv:0711.3914 fetched 2026-09-01)
----------------------------------------------------------------
S. Deffner and E. Lutz, Phys. Rev. E 77, 021128 (2008), "Nonequilibrium
work distribution of a quantum harmonic oscillator" (DL below), read from
the arXiv PDF (v1):

- Eq. (17), verbatim structure with eps0 = hbar w0, eps1 = hbar w1,
  Delta eps = eps1 - eps0 and hbar = 1 here:

    G(mu) = sqrt(2) (1 - e^{-beta eps0}) e^{i mu Delta eps/2} /
            sqrt[ Q*(1 - e^{2 i mu eps1})(1 - e^{-2(i mu + beta) eps0})
                  + (1 + e^{2 i mu eps1})(1 + e^{-2(i mu + beta) eps0})
                  - 4 e^{i mu eps1} e^{-(i mu + beta) eps0} ].

- Eq. (15) defines the adiabaticity parameter

    Q* = [ w0^2 (w1^2 X(tau)^2 + Xdot(tau)^2)
           + (w1^2 Y(tau)^2 + Ydot(tau)^2) ] / (2 w0 w1),

  where X, Y solve the classical equation Ddot X + w^2(t) X = 0 (their
  Eq. (9)) with X(0) = 0, Xdot(0) = 1, Y(0) = 1, Ydot(0) = 0 (their
  Eq. (12)). For a piecewise-constant drive those two solutions are exactly
  the columns of the classical (= symplectic) propagator S, so
  :func:`deffner_lutz_q_star` reads Q* straight off S with no quadrature;
  the sudden quench S = I gives Q* = (w0^2 + w1^2)/(2 w0 w1).
- Eq. (18): G(i beta) = sinh(beta eps0/2)/sinh(beta eps1/2) = e^{-beta DeltaF},
  the Jarzynski equality, independent of Q*.
- Eq. (30): <m>_n = (n + 1/2) Q* - 1/2, and Eq. (31):
  sigma^2_{m,n} = (1/2)(Q*^2 - 1)(n^2 + n + 1) -- exact anchors for the
  dense route's transition matrix P, used in the tests.

Validation (dense) route: Fock-space TPM, N <= 3 modes
------------------------------------------------------
Truncated per-mode occupation cutoff n_max in a Fock basis of REFERENCE
frequency omega_ref: x = (a + a^dag)/sqrt(2 omega_ref),
p = i sqrt(omega_ref/2) (a^dag - a), so p^2 = -(omega_ref/2)(a^dag - a)^2
and x^2 = (a + a^dag)^2/(2 omega_ref) on the truncated space. The reference
frequency matters a great deal numerically: the truncation error is
controlled by how squeezed the true eigenstates are relative to the basis,
so omega_ref defaults to sqrt(w_min w_max) over every K in the protocol
(minimizing max_k |ln(w_k/omega_ref)|). Measured on the 3-mode anchor of
tests/test_work_stats.py this buys ~4 orders of magnitude at fixed n_max
against the omega_ref = 1 basis.

H(K) is assembled term-by-term from Kronecker products (real symmetric, and
PSD by construction: p^2 = (omega_ref/2) d^T d with d = a^dag - a real
antisymmetric, and sum_ij K_ij x_i x_j = sum_k lambda_k (sum_i U_ik x_i)^2
for K = U diag(lambda) U^T PSD). TPM per TLH Eq. (11) with the exact
eigenbases of the truncated H(K0), H(K1) and the exact (unitary) truncated
protocol propagator. Truncation is DIAGNOSED, never assumed: `tail_weight`
(thermal weight on the top occupation rung, worst of the initial and the
time-evolved state) and `ground_energy_error` (|E_0 - sum_k w_k/2|) ship
with every result, and the tests double the cutoff and require reported
quantities to move < tol.

Numerical hazard, stated once: `tail_weight` is a probability weight, so it
is a NECESSARY but not sufficient cutoff diagnostic. Moment-type observables
converge later than it does, because <W^2> sums P[m,n] against E1_m^2 and so
reaches further up the truncated spectrum -- measured on the shipped N = 1
anchor at beta = 0.4: tail_weight = 4.2e-13 at n_max = 140 while the
variance was still 1.2e-7 off in relative terms, and needed n_max ~ 200. The
cutoff-doubling test, not the flag, is the gate.

Crooks
------
Time reversal for these real Hamiltonians is complex conjugation, so the
reverse protocol's propagator is U_R = U_1 U_2 ... U_n = U_F^T (segments in
reverse order) and its TPM matrix is P_R = P_F^T exactly. Hence, pair by
pair on the truncated spaces,

    q_F[m,n] = q_R[n,m] exp(beta (W_mn - DeltaF)),
    q_F[m,n] = p0_n P_F[m,n],   q_R[n,m] = p1_m P_R[n,m],

which is Crooks' relation p_F(W) = p_R(-W) e^{beta(W - DeltaF)} (G. E.
Crooks, Phys. Rev. E 60, 2721 (1999)) resolved at the level of individual
transitions (:func:`crooks_max_defect`). The Fourier transform of the same
identity is the characteristic-function form

    G_F(u) = G_R(-u + i beta) e^{-beta DeltaF},

which :func:`gaussian_crooks_defect` checks on the Gaussian route with no
truncation anywhere.

Free energies
-------------
Exact Gaussian free energy, product over modes: Z(K) = prod_k
[2 sinh(beta w_k/2)]^{-1}, i.e. ln Z = -sum_k [beta w_k/2 +
ln(1 - e^{-beta w_k})], the standard oscillator partition function
(elementary; overflow-safe log1p form).

Everything is pure-functional over arrays: no hidden state, no input
mutation (Layer 6 JAX-mirror pre-positioning). Conventions: hbar = 1,
V_vac = I/2, nu >= 1/2, block ordering, nats (docs/API.md).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from vacuum.core import Omega, chain_propagator, symplectic_inverse, thermal_state_cov

__all__ = [
    "DenseTPM",
    "quadratic_form_matrix",
    "mode_frequencies",
    "log_partition_gaussian",
    "gaussian_free_energy",
    "delta_free_energy",
    "protocol_symplectic",
    "reverse_segments",
    "gaussian_work_cf",
    "gaussian_work_cf_at",
    "gaussian_jarzynski_average",
    "gaussian_work_moments",
    "gaussian_work_cumulants",
    "gaussian_crooks_defect",
    "deffner_lutz_cf",
    "deffner_lutz_q_star",
    "deffner_lutz_cf_sudden",
    "deffner_lutz_final_n_moments",
    "reference_frequency",
    "dense_hamiltonian",
    "dense_tpm",
    "dense_work_cf",
    "dense_work_moments",
    "dense_work_cumulants",
    "dense_jarzynski_average",
    "dense_work_distribution",
    "dense_work_quantile",
    "crooks_max_defect",
    "jarzynski_defect",
    "counting_statistics",
    "COUNTING_STATISTICS_KEYS",
]


# ---------------------------------------------------------------------------
# quadratic-form plumbing and exact free energies
# ---------------------------------------------------------------------------


def quadratic_form_matrix(K):
    """M(K) = blockdiag(K, I): H(K) = (1/2) R^T M(K) R (block ordering)."""
    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    M = np.zeros((2 * N, 2 * N))
    M[:N, :N] = 0.5 * (K + K.T)
    M[N:, N:] = np.eye(N)
    return M


def _spectral(K):
    """(omega, U) with K = U diag(omega^2) U^T, omega > 0 enforced."""
    K = np.asarray(K, dtype=float)
    w2, U = np.linalg.eigh(0.5 * (K + K.T))
    if np.min(w2) <= 0.0:
        raise ValueError(f"K must be positive definite: min eigenvalue {np.min(w2):.3e}")
    return np.sqrt(w2), U


def mode_frequencies(K):
    """Normal-mode frequencies omega_k = sqrt(eig(K)), ascending."""
    omega, _ = _spectral(K)
    return omega


def _check_beta(beta):
    beta = float(beta)
    if not (beta > 0.0) or not np.isfinite(beta):
        raise ValueError(f"beta must be positive and finite, got {beta}")
    return beta


def _as_segments(segments):
    """Materialize an iterable of (K, dt) into a list of (ndarray, float)."""
    return [(np.asarray(K, dtype=float), float(dt)) for K, dt in segments]


def log_partition_gaussian(K, beta):
    """ln Z for H(K) at inverse temperature beta (exact product over modes).

    Z = prod_k [2 sinh(beta w_k / 2)]^{-1}  =>
    ln Z = -sum_k [beta w_k / 2 + log1p(-e^{-beta w_k})]   (overflow-safe).
    """
    beta = _check_beta(beta)
    omega = mode_frequencies(K)
    return float(-np.sum(0.5 * beta * omega + np.log1p(-np.exp(-beta * omega))))


def gaussian_free_energy(K, beta):
    """F = -ln Z / beta for H(K) at inverse temperature beta (exact)."""
    beta = _check_beta(beta)
    return -log_partition_gaussian(K, beta) / beta


def delta_free_energy(K0, K1, beta):
    """DeltaF = F(K1) - F(K0), from the exact mode-product free energy."""
    return gaussian_free_energy(K1, beta) - gaussian_free_energy(K0, beta)


def protocol_symplectic(segments, n_modes):
    """Exact symplectic representative of a piecewise-constant drive.

    segments: iterable of (K, dt) in time order; the protocol unitary is
    U = U_n ... U_1 (rightmost acts first) and the Heisenberg action is
    U^dag R U = S R with S = S_n ... S_1, S_k = chain_propagator(K_k, dt_k).
    Covariances evolve V -> S V S^T. An empty segment list is the sudden
    quench, S = I.
    """
    S = np.eye(2 * int(n_modes))
    for K, dt in segments:
        S = chain_propagator(K, float(dt)) @ S
    return S


def reverse_segments(segments):
    """The time-reversed drive: the same (K, dt) pairs in reverse order.

    With real Hamiltonians, time reversal is complex conjugation and the
    reverse protocol's propagator is U_R = U_1 ... U_n = U_F^T, hence
    P_R = P_F^T (module docstring, Crooks section).
    """
    return list(reversed(_as_segments(segments)))


# ---------------------------------------------------------------------------
# Gaussian route: determinant characteristic function (TLH Eq. (3))
# ---------------------------------------------------------------------------


def _flow(omega, U, z):
    """e^{z Omega M(K)} for complex z, from the cached spectral data of K.

    Analytic continuation of the chain propagator's normal-mode form
    (vacuum.core.dynamics): [[cos(Wz), W^{-1} sin(Wz)], [-W sin(Wz), cos(Wz)]],
    W = sqrt(K); complex trigonometry for complex z.
    """
    c = np.cos(omega * z)
    s = np.sin(omega * z)
    C = (U * c) @ U.T
    Sw = (U * (s / omega)) @ U.T
    Ws = (U * (omega * s)) @ U.T
    return np.block([[C, Sw], [-Ws, C]])


def _refine_path(points, step):
    """Piecewise-linear path through `points` with spacing <= step.

    Returns (path, idx) with path[idx[j]] == points[j] exactly.
    """
    points = np.asarray(points)
    path = [points[0]]
    idx = [0]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.ceil(abs(b - a) / step)))
        seg = a + (b - a) * (np.arange(1, n + 1) / n)
        seg[-1] = b  # exact endpoint, no roundoff drift
        path.extend(seg.tolist())
        idx.append(len(path) - 1)
    return np.asarray(path), np.asarray(idx, dtype=int)


def _default_step(w0, w1):
    """Path step from the phase-rate bound |d arg det/du| <~ sum(w0) + sum(w1).

    0.5/rate keeps per-step phase increments ~0.5 rad, a 6x margin under the
    pi limit of np.unwrap; tested by step-halving (numerical-hazard gate).
    """
    rate = float(np.sum(w0) + np.sum(w1))
    return 0.5 / max(rate, 1.0)


def _cf_on_path(K0, K1, S, beta, path, idx):
    """G at path[idx], branch-tracked along `path` (path must contain 0).

    log det(T(u) - I) is evaluated with np.linalg.slogdet and its phase
    unwrapped along the path; the branch is anchored at the path point u = 0,
    where G = 1 exactly (T(0) = e^{-i beta Omega M0} reproduces 1/Z0).
    """
    beta = _check_beta(beta)
    w0, U0 = _spectral(K0)
    w1, U1 = _spectral(K1)
    S = np.asarray(S, dtype=float)
    Sinv = symplectic_inverse(S)
    eye = np.eye(S.shape[0])

    where_zero = np.flatnonzero(np.asarray(path) == 0.0)
    if where_zero.size == 0:
        raise ValueError("branch-tracking path must contain u = 0")
    i0 = int(where_zero[0])

    # Anti-aliasing gate (predictive, before any work): det(T(u) - I) is a
    # finite Laurent polynomial in the phases e^{i u w0_k}, e^{i u w1_k}, so
    # |d arg det/du| <= sum_k (w0_k + w1_k) =: rate. Requiring the path step
    # to keep the per-step phase increment under 2 rad puts it strictly below
    # the pi aliasing limit of np.unwrap, whatever the path.
    rate = float(np.sum(w0) + np.sum(w1))
    gaps = np.abs(np.diff(np.asarray(path)))
    if gaps.size and float(np.max(gaps)) * rate > 2.0:
        raise ValueError(
            "branch-tracking path under-resolved: max step "
            f"{float(np.max(gaps)):.4g} exceeds the anti-aliasing bound "
            f"2/rate = {2.0 / rate:.4g} (rate = sum of both spectra = {rate:.4g})"
        )

    logabs = np.empty(path.size)
    phase = np.empty(path.size)
    for j, u in enumerate(path):
        T = Sinv @ _flow(w1, U1, -u) @ S @ _flow(w0, U0, u - 1j * beta)
        sign, la = np.linalg.slogdet(T - eye)
        if sign == 0:
            raise FloatingPointError(
                f"det(T(u) - I) vanished at u = {u!r}: the work characteristic "
                "function has a pole on the branch-tracking path"
            )
        logabs[j] = la
        phase[j] = np.angle(sign)
    phase = np.unwrap(phase)
    steps = np.abs(np.diff(phase))
    if steps.size and np.max(steps) > 2.5:
        raise ValueError(
            "characteristic-function branch tracking under-resolved (observed): "
            f"max phase step {np.max(steps):.2f} rad; decrease path_step"
        )
    logdet = logabs + 1j * phase
    G = np.exp(-0.5 * (logdet - logdet[i0]))
    return G[idx]


def gaussian_work_cf(K0, K1, S, beta, us, path_step=None):
    """TPM work characteristic function G(u) on a grid of real u.

    TLH Eq. (3) evaluated in the metaplectic determinant form (module
    docstring): G(u) = [det(T(0)-I)/det(T(u)-I)]^{1/2},
    T(u) = S^{-1} e^{-u Omega M1} S e^{(u - i beta) Omega M0}, branch
    continuous from G(0) = 1. Returns a complex array aligned with `us`.
    """
    us = np.atleast_1d(np.asarray(us, dtype=float))
    if path_step is None:
        path_step = _default_step(mode_frequencies(K0), mode_frequencies(K1))
    pts = np.union1d(us, [0.0])  # sorted, unique, 0 included
    path, idx = _refine_path(pts, path_step)
    G_pts = _cf_on_path(K0, K1, np.asarray(S, dtype=float), beta, path, idx)
    # map each requested u back to its slot in the sorted unique grid
    where = np.searchsorted(pts, us)
    return G_pts[where]


def gaussian_work_cf_at(K0, K1, S, beta, u, path_step=None):
    """G at a single (possibly complex) u, branch-tracked from 0 along a line.

    u = i*beta gives <e^{-beta W}> (Jarzynski, TLH Eq. (5)); general complex
    u supports the Crooks characteristic-function identity
    G_F(u) = G_R(-u + i beta) e^{-beta DeltaF} (module docstring).
    """
    u = complex(u)
    if path_step is None:
        path_step = _default_step(mode_frequencies(K0), mode_frequencies(K1))
    if u == 0.0:
        return 1.0 + 0.0j
    pts = np.array([0.0 + 0.0j, u])
    path, idx = _refine_path(pts, path_step)
    G_pts = _cf_on_path(K0, K1, np.asarray(S, dtype=float), beta, path, idx)
    return complex(G_pts[1])


def gaussian_jarzynski_average(K0, K1, S, beta, path_step=None):
    """<e^{-beta W}> = G(i beta) via the determinant route (real output).

    The imaginary residual is a numerical diagnostic and must be tiny; it is
    asserted small here (1e-8 relative) rather than silently discarded.
    """
    G = gaussian_work_cf_at(K0, K1, S, beta, 1j * _check_beta(beta), path_step)
    if abs(G.imag) > 1e-8 * max(1.0, abs(G.real)):
        raise FloatingPointError(
            f"Jarzynski average has non-negligible imaginary part {G.imag:.3e}"
        )
    return float(G.real)


def gaussian_work_moments(K0, K1, S, beta):
    """Exact TPM mean and variance of W on the Gaussian route (Wick traces).

    mean = (1/2) Tr[(M1_H - M0) V_beta];
    var  = c(M1H,M1H) + c(M0,M0) - 2 Re c(M0,M1H),
    c(A,B) = (1/2) Tr(A G_mat B G_mat^T), G_mat = V_beta + (i/2) Omega
    (derivation in the module docstring; TLH Eq. (11) moments with a Gibbs
    initial state). Returns (mean, variance) as floats.
    """
    beta = _check_beta(beta)
    K0 = np.asarray(K0, dtype=float)
    N = K0.shape[0]
    S = np.asarray(S, dtype=float)
    V = thermal_state_cov(K0, 1.0 / beta)
    M0 = quadratic_form_matrix(K0)
    M1H = S.T @ quadratic_form_matrix(K1) @ S
    mean = 0.5 * float(np.trace((M1H - M0) @ V))

    G_mat = V + 0.5j * Omega(N)
    Gt = G_mat.T

    def c(A, B):
        return 0.5 * np.trace(A @ G_mat @ B @ Gt)

    var_c = c(M1H, M1H) + c(M0, M0) - 2.0 * np.real(c(M0, M1H))
    var = float(np.real(var_c))
    if abs(np.imag(var_c)) > 1e-9 * max(1.0, abs(var)):
        raise FloatingPointError(
            f"work variance has non-negligible imaginary part {np.imag(var_c):.3e}"
        )
    return mean, var


# central finite-difference stencils, O(h^4) accurate, for derivatives 1..4
# of a smooth function sampled on the uniform grid u_j = j*h, j = -3..3.
_FD_STENCILS = {
    1: (np.array([0.0, 1.0 / 12, -2.0 / 3, 0.0, 2.0 / 3, -1.0 / 12, 0.0]), 1),
    2: (np.array([0.0, -1.0 / 12, 4.0 / 3, -5.0 / 2, 4.0 / 3, -1.0 / 12, 0.0]), 2),
    3: (np.array([1.0 / 8, -1.0, 13.0 / 8, 0.0, -13.0 / 8, 1.0, -1.0 / 8]), 3),
    4: (np.array([-1.0 / 6, 2.0, -13.0 / 2, 28.0 / 3, -13.0 / 2, 2.0, -1.0 / 6]), 4),
}


def _cumulants_at_step(K0, K1, S, beta, order, h, path_step):
    """Raw O(h^4) finite-difference cumulants on the stencil of width 3h."""
    us = np.arange(-3, 4) * h
    G = gaussian_work_cf(K0, K1, S, beta, us, path_step=path_step)
    # ln G is smooth and G(0) = 1, so the principal branch is safe on this
    # small symmetric stencil; guard rather than assume.
    max_arg = float(np.max(np.abs(np.angle(G))))
    if max_arg > 2.0:
        raise ValueError(
            "cumulant stencil too wide for the principal log branch "
            f"(max |arg G| = {max_arg:.2f}); decrease h"
        )
    L = np.log(G)
    out = np.empty(order)
    for n in range(1, order + 1):
        coeff, power = _FD_STENCILS[n]
        deriv = complex(np.dot(coeff, L) / h**power)
        kappa = deriv * (-1j) ** n
        if abs(kappa.imag) > 1e-6 * max(1.0, abs(kappa.real)):
            raise FloatingPointError(
                f"cumulant {n} has non-negligible imaginary part {kappa.imag:.3e}"
            )
        out[n - 1] = kappa.real
    return out


def gaussian_work_cumulants(
    K0,
    K1,
    S,
    beta,
    order=4,
    h=None,
    path_step=None,
    richardson=True,
    exact_low_order=True,
):
    """Cumulants kappa_1..kappa_order of W (kappa_1 = mean, kappa_2 = variance).

    G(u) = exp(sum_n kappa_n (iu)^n/n!), so kappa_n = (-i)^n d^n ln G/du^n at
    0; the derivatives are O(h^4) central differences on the exact
    characteristic function (module docstring), Richardson-extrapolated in
    the step, kappa = (16 kappa(h/2) - kappa(h))/15, which cancels the h^4
    term and leaves O(h^6). With `exact_low_order` (the default) kappa_1 and
    kappa_2 are taken from the closed-form Wick traces of
    :func:`gaussian_work_moments` instead -- they are exact there, so there
    is no reason to difference for them; set it False to exercise the finite
    differences against those exact values (which is what the tests do).

    `h` defaults to 0.05/max(omega) over the two spectra: the stencil's
    truncation error falls as h^4 while the eps/h^n roundoff amplification of
    slogdet grows, and 0.05/w_max sits near the joint optimum across the
    shipped anchors. Measured relative accuracy of the returned cumulants on
    the three anchor protocols of tests/test_work_stats.py (N = 1, 2, 3):
    kappa_1, kappa_2 <= 1.2e-7 by finite difference (exact when
    `exact_low_order`), kappa_3, kappa_4 <= 5.2e-6.

    Returns a length-`order` ndarray. kappa_3/kappa_2^{3/2} is the skewness
    and kappa_4/kappa_2^2 the excess kurtosis of the work distribution.
    """
    order = int(order)
    if order < 1 or order > 4:
        raise ValueError(f"order must be in 1..4, got {order}")
    beta = _check_beta(beta)
    w0 = mode_frequencies(K0)
    w1 = mode_frequencies(K1)
    if h is None:
        h = 0.05 / max(float(np.max(w0)), float(np.max(w1)))
    h = float(h)
    if not (h > 0.0):
        raise ValueError(f"h must be positive, got {h}")
    if path_step is None:
        path_step = min(_default_step(w0, w1), h)

    if exact_low_order and order <= 2:
        mean, var = gaussian_work_moments(K0, K1, S, beta)
        return np.array([mean, var])[:order]

    out = _cumulants_at_step(K0, K1, S, beta, order, h, path_step)
    if richardson:
        fine = _cumulants_at_step(
            K0, K1, S, beta, order, 0.5 * h, min(path_step, 0.5 * h)
        )
        out = (16.0 * fine - out) / 15.0
    if exact_low_order:
        mean, var = gaussian_work_moments(K0, K1, S, beta)
        out = np.concatenate([[mean, var], out[2:]])
    return out


def gaussian_crooks_defect(K0, K1, segments=(), beta=1.0, us=(0.0,), path_step=None):
    """Max |G_F(u) - G_R(-u + i beta) e^{-beta DeltaF}| over the grid `us`.

    The characteristic-function form of Crooks' relation (module docstring),
    checked on the Gaussian route with no truncation anywhere: the forward
    protocol is (K0, segments, K1) and the reverse is (K1, reversed
    segments, K0). Returns a float; a passing audit has it at roundoff.
    """
    beta = _check_beta(beta)
    segs = _as_segments(segments)
    K0 = np.asarray(K0, dtype=float)
    K1 = np.asarray(K1, dtype=float)
    N = K0.shape[0]
    us = np.atleast_1d(np.asarray(us, dtype=float))

    S_f = protocol_symplectic(segs, N)
    S_r = protocol_symplectic(reverse_segments(segs), N)
    dF = delta_free_energy(K0, K1, beta)

    G_f = gaussian_work_cf(K0, K1, S_f, beta, us, path_step=path_step)
    G_r = np.array(
        [
            gaussian_work_cf_at(K1, K0, S_r, beta, -u + 1j * beta, path_step=path_step)
            for u in us
        ]
    )
    return float(np.max(np.abs(G_f - G_r * np.exp(-beta * dF))))


# ---------------------------------------------------------------------------
# single-mode analytic anchor (Deffner-Lutz)
# ---------------------------------------------------------------------------


def deffner_lutz_cf(us, w0, w1, beta, q_star):
    """Deffner-Lutz PRE 77, 021128 (2008), Eq. (17), hbar = 1 (eps = omega).

    Characteristic function of TPM work for a single parametric oscillator
    w0 -> w1 with Gibbs initial state; q_star is Husimi's adiabaticity
    parameter (their Eq. (15), see :func:`deffner_lutz_q_star`). Principal
    square root: valid on the moderate |u| ranges used by the anchor tests
    (the multimode determinant route is the production code path; this
    closed form is the independent published anchor). `us` may be complex --
    mu = i beta is DL Eq. (18), the Jarzynski equality.
    """
    beta = _check_beta(beta)
    us = np.atleast_1d(np.asarray(us, dtype=complex))
    e0, e1 = float(w0), float(w1)
    z = 1j * us
    num = np.sqrt(2.0) * (1.0 - np.exp(-beta * e0)) * np.exp(z * (e1 - e0) / 2.0)
    den = (
        q_star * (1.0 - np.exp(2.0 * z * e1)) * (1.0 - np.exp(-2.0 * (z + beta) * e0))
        + (1.0 + np.exp(2.0 * z * e1)) * (1.0 + np.exp(-2.0 * (z + beta) * e0))
        - 4.0 * np.exp(z * e1) * np.exp(-(z + beta) * e0)
    )
    return num / np.sqrt(den)


def deffner_lutz_q_star(S, w0, w1):
    """Adiabaticity parameter Q*, DL Eq. (15), read off the classical propagator.

    Their Eqs. (9)/(12): X, Y solve Ddot q + w^2(t) q = 0 with
    (X, Xdot)(0) = (0, 1) and (Y, Ydot)(0) = (1, 0). For H = (p^2 + w^2 x^2)/2
    (unit mass) the classical flow is exactly the symplectic propagator S in
    block ordering, so (Y, Ydot)(tau) is its first column and (X, Xdot)(tau)
    its second:

        Q* = [w0^2 (w1^2 X^2 + Xdot^2) + (w1^2 Y^2 + Ydot^2)] / (2 w0 w1).

    Q* = 1 is the adiabatic limit (DL Eq. (19) ff.); the sudden quench S = I
    gives Q* = (w0^2 + w1^2)/(2 w0 w1). Single mode only (S is 2x2).
    """
    S = np.asarray(S, dtype=float)
    if S.shape != (2, 2):
        raise ValueError(f"deffner_lutz_q_star is single-mode: S must be 2x2, got {S.shape}")
    Y, X = float(S[0, 0]), float(S[0, 1])
    Yd, Xd = float(S[1, 0]), float(S[1, 1])
    w0, w1 = float(w0), float(w1)
    return (w0**2 * (w1**2 * X**2 + Xd**2) + (w1**2 * Y**2 + Yd**2)) / (2.0 * w0 * w1)


def deffner_lutz_cf_sudden(us, w0, w1, beta):
    """DL Eq. (17) in the sudden-quench limit of their Eq. (15).

    Q* -> (w0^2 + w1^2)/(2 w0 w1) as tau -> 0 (X(tau) -> 0, Xdot(tau) -> 1,
    Y(tau) -> 1, Ydot(tau) -> 0 in Eq. (15)), i.e.
    :func:`deffner_lutz_q_star` at S = I.
    """
    return deffner_lutz_cf(us, w0, w1, beta, deffner_lutz_q_star(np.eye(2), w0, w1))


def deffner_lutz_final_n_moments(n, q_star):
    """(<m>_n, sigma^2_{m,n}) from DL Eqs. (30) and (31).

        <m>_n = (n + 1/2) Q* - 1/2,
        sigma^2_{m,n} = (1/2)(Q*^2 - 1)(n^2 + n + 1),

    the mean and variance of the final quantum number given the initial
    level n. Exact anchors for the dense route's transition matrix P.
    """
    n = float(n)
    q = float(q_star)
    return (n + 0.5) * q - 0.5, 0.5 * (q**2 - 1.0) * (n * n + n + 1.0)


# ---------------------------------------------------------------------------
# dense route: Fock-space TPM (TLH Eq. (11)), N <= 3 modes
# ---------------------------------------------------------------------------


class DenseTPM(NamedTuple):
    """Dense TPM data: spectra, transition probabilities, diagnostics.

    P[m, n] = |<phi_m(K1)|U|phi_n(K0)>|^2 (doubly stochastic on the truncated
    space), p0 the Gibbs populations of the truncated H(K0); log_Z0/log_Z1
    are TRUNCATED log partition functions (their defect against
    :func:`log_partition_gaussian` is part of the tail test).
    tail_weight_initial / tail_weight_final are the thermal weights sitting
    on the top occupation rung of the reference Fock basis in the initial and
    the time-evolved state, and tail_weight is the worse of the two;
    ground_energy_error is the worst |E_0 - sum_k omega_k/2| of the two
    truncated Hamiltonians. The tail weights are probability weights and are
    a necessary, not sufficient, cutoff check (module docstring): moments
    converge later than they do, so gate on cutoff doubling.
    """

    n_modes: int
    n_max: int
    omega_ref: float
    beta: float
    E0: np.ndarray
    E1: np.ndarray
    P: np.ndarray
    p0: np.ndarray
    log_Z0: float
    log_Z1: float
    tail_weight_initial: float
    tail_weight_final: float
    tail_weight: float
    ground_energy_error: float


def reference_frequency(K0, K1, segments=()):
    """omega_ref = sqrt(w_min w_max) over every K in the protocol.

    The Fock basis of frequency omega_ref sees each normal mode squeezed by
    ln(w_k/omega_ref); the geometric mean of the extreme frequencies is the
    minimizer of max_k |ln(w_k/omega_ref)|, hence the shortest occupation
    tails at fixed cutoff (module docstring).
    """
    ws = [mode_frequencies(K0), mode_frequencies(K1)]
    ws += [mode_frequencies(K) for K, _ in _as_segments(segments)]
    allw = np.concatenate(ws)
    return float(np.sqrt(np.min(allw) * np.max(allw)))


def _single_mode_ops(n_max, omega_ref):
    """Truncated x and p^2 for one mode in the omega_ref Fock basis.

    x = (a + a^dag)/sqrt(2 omega_ref), p = i sqrt(omega_ref/2)(a^dag - a), so
    p^2 = -(omega_ref/2)(a^dag - a)^2 = (omega_ref/2) d^T d with
    d = a^dag - a real antisymmetric -- both real symmetric and PSD on the
    (n_max+1)-dimensional space.
    """
    n = np.arange(1, n_max + 1, dtype=float)
    a = np.diag(np.sqrt(n), 1)
    x = (a + a.T) / np.sqrt(2.0 * omega_ref)
    d = a.T - a
    p2 = -0.5 * omega_ref * (d @ d)
    return x, p2


def _kron_chain(ops, d, n_modes):
    """Kronecker chain over modes; ops maps mode index -> (d, d) block,
    identity elsewhere. Mode 0 is the leftmost tensor factor."""
    out = np.eye(1)
    for k in range(n_modes):
        out = np.kron(out, ops.get(k, np.eye(d)))
    return out


def dense_hamiltonian(K, n_max, omega_ref=1.0):
    """Truncated Fock-space H(K) = (1/2)(p.p + x.K.x), real symmetric.

    Assembled term-by-term from Kronecker products (never via full-space
    operator products): cross terms x_i x_j (i != j) act on disjoint tensor
    factors, so the truncated product is the Kronecker product exactly.
    `omega_ref` selects the Fock basis (see :func:`reference_frequency`).
    """
    K = np.asarray(K, dtype=float)
    K = 0.5 * (K + K.T)
    N = K.shape[0]
    d = int(n_max) + 1
    x, p2 = _single_mode_ops(int(n_max), float(omega_ref))
    x2 = x @ x
    dim = d**N
    H = np.zeros((dim, dim))
    for i in range(N):
        H += 0.5 * _kron_chain({i: p2}, d, N)
        H += 0.5 * K[i, i] * _kron_chain({i: x2}, d, N)
        for j in range(i + 1, N):
            if K[i, j] != 0.0:
                H += K[i, j] * _kron_chain({i: x, j: x}, d, N)
    return H


def _log_partition_truncated(E, beta):
    """log sum_n e^{-beta E_n}, overflow-safe (shift by the ground energy)."""
    E = np.asarray(E, dtype=float)
    return float(-beta * E[0] + np.log(np.sum(np.exp(-beta * (E - E[0])))))


def _top_rung_mask(d, n_max, N):
    """Boolean mask of basis states with at least one mode at occupation n_max."""
    digits = np.arange(d**N)
    top = np.zeros(d**N, dtype=bool)
    for _ in range(N):
        top |= digits % d == n_max
        digits = digits // d
    return top


def dense_tpm(
    K0, K1, segments=(), beta=1.0, n_max=8, omega_ref=None, max_dim=20000
):
    """Dense TPM work statistics for a quench/drive, N <= 3 modes.

    TLH Eq. (11) evaluated exactly on the truncated Fock space: eigenbases of
    the truncated H(K0) and H(K1), protocol unitary U = U_n ... U_1 with
    U_k = exp(-i H(K_k) dt_k) built from per-segment eigendecompositions
    (exactly unitary on the truncated space). segments=() is the sudden
    quench (U = I). Returns a :class:`DenseTPM`; truncation diagnostics
    (tail_weight_initial/final, ground_energy_error) ship with the result and
    the cutoff-doubling test is the actual gate.

    Parameter-range note: exp(-beta W) sums downstream stay in float64 range
    for beta * E_max <~ 500; the shipped tests sit far inside that.
    """
    beta = _check_beta(beta)
    K0 = np.asarray(K0, dtype=float)
    K1 = np.asarray(K1, dtype=float)
    segs = _as_segments(segments)
    N = K0.shape[0]
    if N > 3:
        raise ValueError(f"dense route is for N <= 3 modes, got N = {N}")
    if K1.shape != K0.shape:
        raise ValueError(f"K0 and K1 must have the same shape, got {K0.shape}, {K1.shape}")
    n_max = int(n_max)
    if n_max < 2:
        raise ValueError(f"n_max must be >= 2, got {n_max}")
    d = n_max + 1
    dim = d**N
    if dim > int(max_dim):
        raise ValueError(
            f"dense Fock dimension {dim} = ({n_max}+1)^{N} exceeds max_dim={max_dim}; "
            "raise max_dim deliberately if the cost is intended"
        )
    if omega_ref is None:
        omega_ref = reference_frequency(K0, K1, segs)
    omega_ref = float(omega_ref)

    H0 = dense_hamiltonian(K0, n_max, omega_ref)
    H1 = dense_hamiltonian(K1, n_max, omega_ref)
    E0, W0 = np.linalg.eigh(H0)
    E1, W1 = np.linalg.eigh(H1)

    if not segs:
        UW0 = W0  # U = I
        Mtrans = W1.T @ W0
        P = Mtrans**2
    else:
        U = np.eye(dim, dtype=complex)
        for K_seg, dt in segs:
            Es, Ws = np.linalg.eigh(dense_hamiltonian(K_seg, n_max, omega_ref))
            U = (Ws * np.exp(-1j * Es * dt)) @ Ws.T @ U
        UW0 = U @ W0
        Mtrans = W1.T @ UW0
        P = np.abs(Mtrans) ** 2

    shifted = np.exp(-beta * (E0 - E0[0]))
    p0 = shifted / np.sum(shifted)

    # thermal weight on the top occupation rung of the reference Fock basis,
    # for the initial state rho0 = sum_n p0_n |phi_n(K0)><phi_n(K0)| and for
    # the time-evolved state U rho0 U^dag (the latter is what the second
    # measurement sees, and it is the tail the m-sum of TLH Eq. (11) probes).
    top = _top_rung_mask(d, n_max, N)
    tail_initial = float(np.sum(((W0**2) @ p0)[top]))
    tail_final = float(np.sum(((np.abs(UW0) ** 2) @ p0)[top]))

    zp0 = 0.5 * np.sum(mode_frequencies(K0))
    zp1 = 0.5 * np.sum(mode_frequencies(K1))
    ground_energy_error = float(max(abs(E0[0] - zp0), abs(E1[0] - zp1)))

    return DenseTPM(
        n_modes=N,
        n_max=n_max,
        omega_ref=omega_ref,
        beta=beta,
        E0=E0,
        E1=E1,
        P=P,
        p0=p0,
        log_Z0=_log_partition_truncated(E0, beta),
        log_Z1=_log_partition_truncated(E1, beta),
        tail_weight_initial=tail_initial,
        tail_weight_final=tail_final,
        tail_weight=max(tail_initial, tail_final),
        ground_energy_error=ground_energy_error,
    )


def dense_work_cf(tpm, us):
    """G(u) = sum_{n,m} p0_n P[m,n] e^{iu(E1_m - E0_n)} (TLH Eq. (11))."""
    us = np.atleast_1d(np.asarray(us, dtype=float))
    G = np.empty(us.size, dtype=complex)
    for j, u in enumerate(us):
        G[j] = np.exp(1j * u * tpm.E1) @ (tpm.P @ (tpm.p0 * np.exp(-1j * u * tpm.E0)))
    return G


def dense_work_moments(tpm):
    """(mean, variance) of the dense TPM work distribution."""
    Pp = tpm.P @ tpm.p0
    mean = float(tpm.E1 @ Pp - tpm.p0 @ tpm.E0)
    second = float(
        (tpm.E1**2) @ Pp
        + tpm.p0 @ (tpm.E0**2)
        - 2.0 * (tpm.E1 @ (tpm.P @ (tpm.p0 * tpm.E0)))
    )
    return mean, second - mean**2


def dense_work_cumulants(tpm, order=4):
    """Cumulants kappa_1..kappa_order of the dense TPM work distribution.

    Computed from the exact discrete distribution via central moments:
    kappa_1 = mean, kappa_2 = mu_2, kappa_3 = mu_3, kappa_4 = mu_4 - 3 mu_2^2.
    """
    order = int(order)
    if order < 1 or order > 4:
        raise ValueError(f"order must be in 1..4, got {order}")
    W, q = dense_work_distribution(tpm)
    mean = float(q @ W)
    dW = W - mean
    mu = [float(q @ dW**k) for k in range(2, 5)]  # mu_2, mu_3, mu_4
    kappas = [mean, mu[0], mu[1], mu[2] - 3.0 * mu[0] ** 2]
    return np.array(kappas[:order])


def dense_jarzynski_average(tpm):
    """<e^{-beta W}> summed over the explicit work values W_mn = E1_m - E0_n.

    Deliberately routed through the sampled W matrix (not the algebraic
    collapse to Z1/Z0) so the test against e^{-beta DeltaF} from the exact
    mode-product free energy has independent content.
    """
    W = tpm.E1[:, None] - tpm.E0[None, :]
    q = tpm.P * tpm.p0[None, :]
    return float(np.sum(q * np.exp(-tpm.beta * W)))


def dense_work_distribution(tpm, floor=0.0):
    """Flattened (W_values, probabilities), ascending in W; drops q <= floor."""
    W = (tpm.E1[:, None] - tpm.E0[None, :]).ravel()
    q = (tpm.P * tpm.p0[None, :]).ravel()
    keep = q > floor
    W, q = W[keep], q[keep]
    order = np.argsort(W)
    return W[order], q[order]


def dense_work_quantile(tpm, probs):
    """Quantile(s) of the exact discrete work distribution.

    W_q = inf{ w : CDF(w) >= q }, the standard lower quantile of a discrete
    distribution. `probs` may be a scalar or a sequence in [0, 1]; returns a
    float or an ndarray to match. This is what upgrades a protocol row's
    "per unit switching energy" to "per unit work at a stated quantile".
    """
    probs_arr = np.atleast_1d(np.asarray(probs, dtype=float))
    if np.any(probs_arr < 0.0) or np.any(probs_arr > 1.0):
        raise ValueError("quantile probabilities must lie in [0, 1]")
    W, q = dense_work_distribution(tpm)
    cdf = np.cumsum(q)
    cdf = cdf / cdf[-1]  # normalize away the truncated-space rounding
    idx = np.searchsorted(cdf, probs_arr, side="left")
    idx = np.clip(idx, 0, W.size - 1)
    out = W[idx]
    return float(out[0]) if np.ndim(probs) == 0 else out


def crooks_max_defect(fwd, rev, q_floor=1e-12):
    """Max relative Crooks defect over significant forward/reverse pairs.

    Crooks' relation p_F(W) = p_R(-W) e^{beta(W - DeltaF)} (Crooks, PRE 60,
    2721 (1999)) resolved transition by transition. With

        q_F[m,n] = fwd.p0[n] * fwd.P[m,n]      (initial level n of H(K0),
                                                final level m of H(K1)),
        q_R[n,m] = rev.p0[m] * rev.P[n,m]      (reverse run: initial level m
                                                of H(K1), final level n of
                                                H(K0)),
        W[m,n]   = fwd.E1[m] - fwd.E0[n],

    the identity is q_F[m,n] = q_R[n,m] exp(beta (W[m,n] - DeltaF)) with
    DeltaF = (log_Z0 - log_Z1)/beta taken from the SAME truncated spectra, so
    the check is exact on the truncated space rather than truncation-limited.
    `rev` must be dense_tpm(K1, K0, reverse_segments(segments), ...) at the
    same beta, n_max and omega_ref.

    Returns the max relative defect over pairs whose forward weight exceeds
    q_floor * max(q_F).
    """
    if fwd.beta != rev.beta:
        raise ValueError(f"beta mismatch: forward {fwd.beta}, reverse {rev.beta}")
    if fwd.P.shape != rev.P.shape:
        raise ValueError(
            f"truncation mismatch: forward P {fwd.P.shape}, reverse P {rev.P.shape}"
        )
    beta = fwd.beta
    W = fwd.E1[:, None] - fwd.E0[None, :]  # (M, N)
    q_F = fwd.P * fwd.p0[None, :]  # (M, N)
    # q_R[n, m] laid out on the same (m, n) grid:
    q_R_on_mn = rev.P.T * rev.p0[:, None]
    dF_trunc = (fwd.log_Z0 - fwd.log_Z1) / beta
    with np.errstate(over="ignore"):
        predicted = q_R_on_mn * np.exp(beta * (W - dF_trunc))
    mask = q_F > float(q_floor) * float(np.max(q_F))
    defect = np.abs(q_F[mask] - predicted[mask]) / q_F[mask]
    return float(np.max(defect))


def jarzynski_defect(average, K0, K1, beta):
    """|<e^{-beta W}> - e^{-beta DeltaF}|, DeltaF from the exact mode product.

    Spec M3.5 audit: < 1e-10 (dense) / 1e-6 (Gaussian). TLH Eq. (5).
    """
    beta = _check_beta(beta)
    target = np.exp(log_partition_gaussian(K1, beta) - log_partition_gaussian(K0, beta))
    return float(abs(float(average) - target))


# ---------------------------------------------------------------------------
# ledger-facing summary (the beyond-mean refinement)
# ---------------------------------------------------------------------------

#: keys every :func:`counting_statistics` dict carries (the contract the
#: EnergyLedger's `refine_work` validates against).
COUNTING_STATISTICS_KEYS = (
    "route",
    "n_modes",
    "beta",
    "mean",
    "variance",
    "std",
    "delta_F",
    "jarzynski_average",
    "jarzynski_defect",
    "second_law_margin",
)


def counting_statistics(
    K0,
    K1,
    segments=(),
    beta=1.0,
    route="gaussian",
    n_max=10,
    omega_ref=None,
    path_step=None,
    cumulants=False,
    quantiles=(),
):
    """Work counting statistics of a quench/drive, as a plain dict.

    The object :meth:`vacuum.audits.EnergyLedger.refine_work` accepts: it
    carries the TPM mean (which must agree with the ledger's mean-work
    entry), the variance/std, the exact DeltaF, the Jarzynski average and its
    defect (TLH Eq. (5)), and the second-law margin <W> - DeltaF (>= 0 by
    Jensen on Eq. (5)). Route 'gaussian' (any N, exact closed form) or
    'dense' (N <= 3, validation route; adds the truncation diagnostics and
    supports `quantiles`). `cumulants=True` adds kappa_3, kappa_4, skewness
    and excess kurtosis.
    """
    beta = _check_beta(beta)
    K0 = np.asarray(K0, dtype=float)
    K1 = np.asarray(K1, dtype=float)
    segs = _as_segments(segments)
    N = K0.shape[0]
    dF = delta_free_energy(K0, K1, beta)
    quantile_probs = np.atleast_1d(np.asarray(quantiles, dtype=float))
    if route not in ("gaussian", "dense"):
        raise ValueError(f"route must be 'gaussian' or 'dense', got {route!r}")
    if route == "gaussian" and quantile_probs.size:
        raise ValueError(
            "work quantiles need the exact discrete distribution: "
            "call counting_statistics(..., route='dense', quantiles=...)"
        )

    extras = {}
    if route == "gaussian":
        S = protocol_symplectic(segs, N)
        mean, var = gaussian_work_moments(K0, K1, S, beta)
        jz = gaussian_jarzynski_average(K0, K1, S, beta, path_step=path_step)
        if cumulants:
            k = gaussian_work_cumulants(K0, K1, S, beta, order=4, path_step=path_step)
            extras.update(_cumulant_extras(k))
    else:
        tpm = dense_tpm(
            K0, K1, segs, beta=beta, n_max=n_max, omega_ref=omega_ref
        )
        mean, var = dense_work_moments(tpm)
        jz = dense_jarzynski_average(tpm)
        extras.update(
            {
                "n_max": float(tpm.n_max),
                "omega_ref": float(tpm.omega_ref),
                "tail_weight": float(tpm.tail_weight),
                "tail_weight_initial": float(tpm.tail_weight_initial),
                "tail_weight_final": float(tpm.tail_weight_final),
                "ground_energy_error": float(tpm.ground_energy_error),
            }
        )
        if cumulants:
            extras.update(_cumulant_extras(dense_work_cumulants(tpm, order=4)))
        for p in quantile_probs:
            extras[f"work_q{float(p):g}"] = float(dense_work_quantile(tpm, float(p)))

    stats = {
        "route": route,
        "n_modes": float(N),
        "beta": beta,
        "mean": float(mean),
        "variance": float(var),
        "std": float(np.sqrt(max(var, 0.0))),
        "delta_F": float(dF),
        "jarzynski_average": float(jz),
        "jarzynski_defect": jarzynski_defect(jz, K0, K1, beta),
        "second_law_margin": float(mean - dF),
    }
    stats.update(extras)
    return stats


def _cumulant_extras(kappas):
    """Third/fourth cumulants plus the dimensionless shape numbers."""
    k1, k2, k3, k4 = (float(v) for v in kappas)
    out = {"kappa_3": k3, "kappa_4": k4}
    if k2 > 0.0:
        out["skewness"] = k3 / k2**1.5
        out["excess_kurtosis"] = k4 / k2**2
    return out
