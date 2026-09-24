"""The non-commuting multimode bound: what is proved, what is first order, what is attacked.

Setting (as in :mod:`vacuum.floquet.bounds`, "Multimode").  H(t) = p.p/2 + x.K(t) x/2 with

    K(t) = K0 + d s(t) P,   K0 > 0,  P = P^T,  |s(t)| <= 1 measurable,               (1)

A := K0^{-1/2} P K0^{-1/2},  d_eff := d ||A||  (so (1 - d_eff) K0 <= K(t) <= (1 + d_eff) K0),
C := K0^{-1/4} P K0^{-1/4} = K0^{1/4} A K0^{1/4},  omega_max := ||K0||^{1/2}.

Theorem M (proved here; the number-norm Gronwall bound).  For every solution and every
t1 < t2, with the "number" form N = (p.K0^{-1/2} p + x.K0^{1/2} x)/2 = sum_k |a_k|^2,

    ln N(t2)/N(t1) <= d ||C|| int_{t1}^{t2} |s(t)| dt,                                   (2)

hence for T-periodic s, with rho(S_F) = max|eig S_F|,

    ln rho(S_F) <= (d/2) ||C|| int_0^T |s| dt <= (d/2) ||C|| T,                           (3)
    ln rho(S_F)/T <= (d/2) ||K0^{-1/4} P K0^{-1/4}|| <= (d/2) ||P K0^{-1/2}|| <= omega_max d_eff/2.   (4)

Proof.  In the canonical complex coordinates a = (K0^{1/4} x + i K0^{-1/4} p)/sqrt2 Hamilton's
equations read a' = -i Omega a - (i d s/2) C (a + conj a), Omega = K0^{1/2}.  Then
d|a|^2/dt = 2 Re(conj(a).a') = d s Im(conj(a).C conj(a)) because conj(a).Omega a and
conj(a).C a are real (Omega, C real symmetric), and |conj(a).C conj(a)| <= ||C|| |a|^2.
Integrating gives (2); the operator norm of the real map S_F in the N-norm is bounded by
exp((d/2)||C|| int|s|) and dominates rho(S_F), giving (3).  The chain (4):
||K0^{1/4} A K0^{1/4}||^2 = rho(K0^{1/4} A K0^{1/2} A K0^{1/4}) = rho((A K0^{1/2})^2)
<= ||A K0^{1/2}||^2 = ||P K0^{-1/2}||^2 <= omega_max^2 ||A||^2  (rho(XY) = rho(YX)).  The middle
member is the energy-form Gronwall bound (E = (p.p + x.K0 x)/2 gives d ln E/dt <= d|s| ||P K0^{-1/2}||).
The N-norm is the sharper Lyapunov form because the "beam-splitter" part -(i d s/2) C a of the
coupling is N-preserving and only the squeezing part C conj(a) can grow N.  QED.

Constants.  For N = 1, ||C|| = omega0 and (4) reads ln rho/T <= omega0 d/2, while Theorem B gives
the exact supremum omega0 lambda*(d); so lambda*(d) <= d/2 for every d (numerically
0.6366 <= lambda*(d)/(d/2) <= 0.6890 on (0, 1)).  Theorem M is therefore at most pi/2 (small d)
to 1.451 (d -> 1) times the conjectured bound omega_max lambda*(d_eff) where ||C|| = omega_max ||A||,
and *sharper* than the conjecture wherever ||C||/(omega_max ||A||) < lambda*(d_eff)/(d_eff/2).

First order (Theorem F, rotating-wave).  The averaging step below is an ASSUMPTION of the
standard kind (first order in d, valid for d||C|| << omega_min; checked numerically in
``papers/time-modulated-vacua/notebook_multimode.py`` section B to 1.2e-3 at d_eff <= 0.05), not a
theorem of this module.  With a = e^{-i Omega t} b the growth of |b|^2 is driven
by the matrix C(t)_ij = C_ij e^{i(omega_i + omega_j) t}; averaging over one period of a T-periodic s
leaves a Hermitian (norm-preserving) difference-frequency part and the squeezing generator
G_ij = (d/2) C_ij shat(omega_i + omega_j),
shat(W) = (1/T) int_0^T s e^{iWt} dt (nonzero only for W in (2 pi/T) Z), and b'' = G conj(G) b:
the first-order growth rate is sigma_max(G) (:func:`rwa_growth_rate`).  Two modes coupled only
off-diagonally (C = C_12 (e1 e2^T + e2 e1^T)) at the sum frequency omega_1 + omega_2 = 2 pi/T with
the square wave (|shat| = 2/pi) give

    rate = (d/pi) |C_12| = (d_eff/pi) sqrt(omega_1 omega_2),                                (5)

i.e. sqrt(omega_1 omega_2)/omega_max x (d/pi)/lambda*(d_eff) <= 1 of the conjectured bound: a
low-frequency pair cannot beat it at first order, and the frequency that governs the sum-frequency
resonance is the geometric mean, which is exactly the K0^{-1/4} weighting of ||C||.  For a single
resonant frequency the first-order rate is <= (d/pi) ||C_res|| <= (d/pi) ||C||: C_res (the entries
with omega_i + omega_j = 2 pi/T) is the average over theta of the unitary congruences
e^{-i theta W} U_theta C U_theta^T, U_theta = diag(e^{i theta omega_i}), each of which has the same
operator norm as C.  With several harmonics the first-order rate is at most (d/2) ||C o Shat||
(Schur product; equality when no difference frequency is resonant) and whether that can exceed
(d/pi) ||C|| is an open Schur-multiplier question -- tested numerically, not proved (see the draft).

Refined conjecture (tested, not proved).  ln rho(S_F)/T <= (||C||/||A||) lambda*(d_eff), which
equals the original conjecture when the top eigenvector of A sits on the omega_max mode and is
sharper otherwise; it reduces to Theorem A/B for one mode and for commuting patterns
(lambda*(x)/x is increasing).  :func:`multimode_ratios` reports the ratio of ln rho/T to the
conjectured, the refined and the proved bounds; :func:`multimode_attack` maximises the conjectured
ratio over (omega, A, s, durations) by CEM + Nelder-Mead in the normal-mode parametrisation.

Everything is pure-functional over arrays; all monodromies are exact ordered products of static
symplectic exponentials (a piecewise-constant s is itself admissible).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import minimize

from .bounds import rate_optimum

__all__ = [
    "frequency_weighted_norm",
    "multimode_bound_proved",
    "normal_mode_system",
    "pattern_monodromy",
    "rwa_growth_rate",
    "sum_frequency_first_order",
    "multimode_ratios",
    "multimode_attack",
]


def _sym(P):
    P = np.asarray(P, dtype=float)
    return 0.5 * (P + P.T)


def _K0_powers(K0):
    w2, U = np.linalg.eigh(_sym(K0))
    if np.min(w2) <= 0.0:
        raise ValueError("K0 must be positive definite")
    return w2, U


def frequency_weighted_norm(K0, P):
    """||K0^{-1/4} P K0^{-1/4}||_2, the norm of C in Theorem M (Eq. (4))."""
    w2, U = _K0_powers(K0)
    R = (U / w2 ** 0.25) @ U.T
    return float(np.linalg.norm(R @ _sym(P) @ R, 2))


def multimode_bound_proved(K0, P, depth):
    """Theorem M: the proved rate bound of the non-commuting pattern drive K0 + depth s(t) P.

    Returns a dict: ``rate_bound`` = (d/2)||K0^{-1/4} P K0^{-1/4}|| (ln rho / T is at most this
    for every measurable |s| <= 1), the two weaker members of the chain (4) ``energy_form`` =
    (d/2)||P K0^{-1/2}|| and ``omega_max_form`` = omega_max d_eff/2, together with ``norm_C``,
    ``d_eff``, ``omega_max`` and ``conjectured`` = omega_max lambda*(d_eff) for comparison
    (inf when d_eff = 0).  Requires d_eff < 1 (K(t) > 0).
    """
    K0 = _sym(K0)
    P = _sym(P)
    d = float(depth)
    if d < 0.0:
        raise ValueError("depth must be >= 0")
    w2, U = _K0_powers(K0)
    Rh = (U / np.sqrt(w2)) @ U.T
    A = Rh @ P @ Rh
    norm_A = float(np.linalg.norm(A, 2))
    d_eff = d * norm_A
    if d_eff >= 1.0:
        raise ValueError(f"d_eff = {d_eff:.6g} >= 1: K(t) is not positive for all |s| <= 1")
    norm_C = frequency_weighted_norm(K0, P)
    energy = float(np.linalg.norm(P @ Rh, 2))
    w_max = float(np.sqrt(np.max(w2)))
    conj = w_max * rate_optimum(d_eff).lambda_star if d_eff > 0.0 else np.inf
    refined = (norm_C / norm_A) * rate_optimum(d_eff).lambda_star if d_eff > 0.0 else np.inf
    return {"rate_bound": 0.5 * d * norm_C, "energy_form": 0.5 * d * energy,
            "omega_max_form": 0.5 * w_max * d_eff, "norm_C": norm_C, "norm_A": norm_A,
            "d_eff": d_eff, "omega_max": w_max, "conjectured": float(conj),
            "refined_conjecture": float(refined)}


# --------------------------------------------------------------------------
# Exact monodromies of pattern drives with free piece durations
# --------------------------------------------------------------------------


def normal_mode_system(omega, A):
    """(K0, P, C) in the normal-mode basis: K0 = diag(omega^2), P = diag(omega) A diag(omega)
    (so K0^{-1/2} P K0^{-1/2} = A) and C = K0^{-1/4} P K0^{-1/4} = diag(sqrt omega) A diag(sqrt omega)."""
    omega = np.asarray(omega, dtype=float)
    if np.any(omega <= 0.0):
        raise ValueError("omega must be positive")
    A = _sym(A)
    return np.diag(omega ** 2), A * np.outer(omega, omega), A * np.outer(np.sqrt(omega), np.sqrt(omega))


def _propagators(Ks, taus):
    w2, U = np.linalg.eigh(Ks)
    if np.min(w2) <= 0.0:
        raise ValueError("K(t) lost positivity on a piece (d_eff >= 1?)")
    w = np.sqrt(w2)
    wt = w * taus[:, None]
    c, s = np.cos(wt), np.sin(wt)
    Ut = np.swapaxes(U, 1, 2)
    M, N = Ks.shape[0], Ks.shape[1]
    S = np.empty((M, 2 * N, 2 * N))
    S[:, :N, :N] = (U * c[:, None, :]) @ Ut
    S[:, :N, N:] = (U * (s / w)[:, None, :]) @ Ut
    S[:, N:, :N] = -(U * (w * s)[:, None, :]) @ Ut
    S[:, N:, N:] = S[:, :N, :N]
    return S


def pattern_monodromy(K0, P, depth, s, taus):
    """Exact monodromy (2N x 2N) of K(t) = K0 + depth s_j P on pieces of durations taus_j
    (ordered product of static symplectic exponentials; no discretisation error)."""
    K0 = _sym(K0)
    P = _sym(P)
    s = np.asarray(s, dtype=float).ravel()
    taus = np.asarray(taus, dtype=float).ravel()
    if s.shape != taus.shape:
        raise ValueError("s and taus must have the same length")
    if np.any(np.abs(s) > 1.0 + 1e-15) or np.any(taus <= 0.0):
        raise ValueError("need |s| <= 1 and positive durations")
    Ks = K0[None] + float(depth) * s[:, None, None] * P[None]
    Ps = _propagators(Ks, taus)
    S = np.eye(2 * K0.shape[0])
    for j in range(s.size):
        S = Ps[j] @ S
    return S


def _ln_rho(S):
    return float(max(0.0, np.log(np.max(np.abs(np.linalg.eigvals(S))))))


def rwa_growth_rate(K0, P, depth, s, taus, *, tol=1e-9):
    """First-order (rotating-wave) growth rate sigma_max(G), G_ij = (d/2) C_ij shat(omega_i + omega_j),
    for the piecewise-constant T-periodic control (s_j on durations taus_j); pairs whose sum
    frequency is not a multiple of 2 pi/T (to ``tol``) are non-resonant at first order and dropped.
    Basis-covariant: computed in the normal modes of K0."""
    K0 = _sym(K0)
    w2, U = _K0_powers(K0)
    omega = np.sqrt(w2)
    Pn = U.T @ _sym(P) @ U
    C = Pn / np.outer(np.sqrt(omega), np.sqrt(omega))
    s = np.asarray(s, dtype=float).ravel()
    taus = np.asarray(taus, dtype=float).ravel()
    T = float(np.sum(taus))
    edges = np.concatenate([[0.0], np.cumsum(taus)])
    N = omega.size
    G = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            W = omega[i] + omega[j]
            m = W * T / (2.0 * np.pi)
            if abs(m - round(m)) > tol:
                continue
            shat = np.sum(s * (np.exp(1j * W * edges[1:]) - np.exp(1j * W * edges[:-1]))) / (1j * W * T)
            G[i, j] = 0.5 * float(depth) * C[i, j] * shat
    return float(np.linalg.norm(G, 2))


def sum_frequency_first_order(omega1, omega2, depth_eff):
    """Eq. (5): first-order growth rate (d_eff/pi) sqrt(omega1 omega2) of the pure two-mode coupling
    at the sum frequency under the square wave, and its ratio to the conjectured bound
    omega_max lambda*(d_eff)."""
    w1, w2, d = float(omega1), float(omega2), float(depth_eff)
    rate = d / np.pi * np.sqrt(w1 * w2)
    conj = max(w1, w2) * rate_optimum(d).lambda_star
    return {"rate": rate, "conjectured": conj, "ratio": rate / conj}


def multimode_ratios(omega, A, depth_eff, s, taus):
    """ln rho and its ratios to the conjectured (omega_max lambda*(d_eff)), refined
    ((||C||/||A||) lambda*(d_eff)) and proved ((d/2)||C||, Theorem M) rate bounds, in the
    normal-mode parametrisation (A is rescaled to ||A|| = 1 so that depth_eff is d_eff)."""
    omega = np.asarray(omega, dtype=float)
    A = _sym(A)
    nA = np.linalg.norm(A, 2)
    if nA == 0.0:
        raise ValueError("A must be nonzero")
    A = A / nA
    d = float(depth_eff)
    K0, P, C = normal_mode_system(omega, A)
    taus = np.asarray(taus, dtype=float).ravel()
    T = float(np.sum(taus))
    lr = _ln_rho(pattern_monodromy(K0, P, d, s, taus))
    nC = float(np.linalg.norm(C, 2))
    w_max = float(np.max(omega))
    lam = rate_optimum(d).lambda_star
    conj, refined, proved = w_max * lam, nC * lam, 0.5 * d * nC
    return {"ln_rho": lr, "rate": lr / T, "period": T, "d_eff": d, "norm_C": nC, "omega_max": w_max,
            "conjectured_bound": conj, "refined_bound": refined, "proved_bound": proved,
            "ratio_conjectured": lr / (T * conj), "ratio_refined": lr / (T * refined),
            "ratio_proved": lr / (T * proved)}


# --------------------------------------------------------------------------
# The attack: maximise ln rho / (T omega_max lambda*(d_eff)) over (omega, A, s, durations)
# --------------------------------------------------------------------------


def _unpack(theta, N, M):
    theta = np.asarray(theta, dtype=float)
    k = 0
    om = np.ones(N)
    if N > 1:
        om[:-1] = 1.0 / (1.0 + np.exp(-theta[k:k + N - 1]))
        k += N - 1
    nA = N * (N + 1) // 2
    A = np.zeros((N, N))
    iu = np.triu_indices(N)
    A[iu] = theta[k:k + nA]
    A = A + np.triu(A, 1).T
    k += nA
    s = np.tanh(theta[k:k + M])
    k += M
    taus = np.exp(np.clip(theta[k:k + M], -8.0, 4.0))
    return om, A, s, taus


def _pack(om, A, s, taus):
    N = len(om)
    th = []
    if N > 1:
        p = np.clip(np.asarray(om[:-1], dtype=float), 1e-6, 1 - 1e-6)
        th += list(np.log(p / (1 - p)))
    th += list(np.asarray(A, dtype=float)[np.triu_indices(N)])
    th += list(np.arctanh(np.clip(np.asarray(s, dtype=float), -1 + 1e-9, 1 - 1e-9)))
    th += list(np.log(np.asarray(taus, dtype=float)))
    return np.array(th)


def multimode_attack(N, M, depth_eff, *, rng=None, n_iter=120, batch=96, elite_frac=0.12,
                     polish_iter=4000, start=None, key="ratio_conjectured"):
    """CEM over (omega with omega_max = 1, A normalised, s in [-1, 1]^M, durations) maximising
    ``key`` of :func:`multimode_ratios` at fixed d_eff, then a Nelder-Mead polish.

    ``start``: optional (omega, A, s, taus) tuple to seed the population mean (e.g. a resonant
    square wave).  Returns the best configuration with all ratios.  A configuration with
    ``ratio_conjectured`` > 1 would refute the conjectured bound; the Theorem M ratio is always < 1.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    dim = (N - 1) + N * (N + 1) // 2 + 2 * M
    d = float(depth_eff)

    def score(theta):
        om, A, s, taus = _unpack(theta, N, M)
        if np.linalg.norm(A, 2) == 0.0:
            return 0.0
        try:
            return multimode_ratios(om, A, d, s, taus)[key]
        except (ValueError, np.linalg.LinAlgError):
            return 0.0

    mu = np.zeros(dim) if start is None else _pack(*start)
    sig = np.full(dim, 1.2)
    n_elite = max(3, int(elite_frac * batch))
    best_t, best_v = mu.copy(), score(mu)
    for _ in range(int(n_iter)):
        pop = mu[None, :] + sig[None, :] * rng.normal(size=(batch, dim))
        pop[0] = best_t
        vals = np.array([score(p) for p in pop])
        idx = np.argsort(vals)[::-1][:n_elite]
        if vals[idx[0]] > best_v:
            best_v, best_t = float(vals[idx[0]]), pop[idx[0]].copy()
        mu = 0.7 * pop[idx].mean(axis=0) + 0.3 * mu
        sig = np.maximum(0.7 * pop[idx].std(axis=0) + 0.3 * sig, 1e-6)
        if np.max(sig) < 1e-5:
            break
    res = minimize(lambda t: -score(t), best_t, method="Nelder-Mead",
                   options={"maxiter": int(polish_iter), "xatol": 1e-9, "fatol": 1e-12, "adaptive": True})
    theta = res.x if -res.fun >= best_v else best_t
    om, A, s, taus = _unpack(theta, N, M)
    A = A / np.linalg.norm(A, 2)
    out = multimode_ratios(om, A, d, s, taus)
    out.update(omega=om, A=A, s=s, taus=taus, cem_value=float(best_v), N=int(N), M=int(M),
               polish_nfev=int(res.nfev))
    return out
