"""The amplification bound of a bounded modulation: theorem, rate optimum, adversary.

Problem.  A single mode of the uniformly modulated lattice field
(``modulated_K(..., kind='coupling')`` block-diagonalises into these)

    x'' + omega0^2 (1 + d s(t)) x = 0,      |s(t)| <= 1 measurable,  0 <= d < 1,   (1)

or, in the symplectic form of :mod:`vacuum.floquet.monodromy`, R' = Omega H(t) R
with H(t) = diag(omega0^2 (1 + d s(t)), 1) on R = (x, p).  For a T-periodic s
the monodromy S_F = S(T) is real symplectic with multipliers lambda, 1/lambda
and rho(S_F) = max|eig S_F|.  The Layer-5 result candidate conjectured
(papers/time-modulated-vacua, MANIFEST claim (a)) that

    ln rho(S_F) <= artanh(d)   per cycle,                                   (2)

attained by the two-level (bang-bang) drive with quarter-period dwells, and
proved it only for two-level drives (Meissner's trace identity).  This module
proves (2) for every admissible s (Theorem A), fixes the meaning of "per
cycle", gives the growth-*rate* optimum in closed form (Theorem B), places
both against the prior literature on bang-bang parametric pumping (next
paragraph: the bang-bang mechanism and the equality case are classical; the
per-zero inequality for arbitrary measurable s and the closed-form rate
optimum are what was not found), and carries the adversarial numerics that
try to break the theorem on discretised controls (which are themselves admissible
controls, so the exact piecewise monodromy tests the theorem with no
discretisation caveat).

Prior work (each item fetched and checked before citing)
--------------------------------------------------------
The equality case is classical: the two-level drive with quarter-period
dwells, growing by omega2/omega1 per cycle, is Meissner's square-wave Hill
equation [E. Meissner, Schweiz. Bauzeit. 72, 95 (1918)] and the textbook
swing of V. I. Arnold, "Mathematical Methods of Classical Mechanics", 2nd ed.,
GTM 60 (Springer, 1989), Sec. 25 "Parametric resonance"; ``bang_bang_bound``
reproduces it.  Bang-bang optimality of parametric pumping with a *bounded*
frequency is also prior art: E. K. Lavrovskii and A. M. Formal'skii, J. Appl.
Math. Mech. 57, 311 (1993), doi:10.1016/0021-8928(93)90059-U (optimal pumping
and damping of a swing); B. Piccoli and J. Kulkarni, IEEE Control Syst. Mag.
25(4), 48 (2005), doi:10.1109/MCS.2005.1499390 (Pontryagin minimum principle:
standing at the lowest point and squatting at the highest is time-optimal for
amplitude growth); B. Andresen, K. H. Hoffmann, J. Nulton, A. Tsirlin and
P. Salamon, Eur. J. Phys. 32, 827 (2011), doi:10.1088/0143-0807/32/3/018
(minimum time to a target energy with omega_min <= omega <= omega_max);
D. Stefanatos, J. Ruths and J.-S. Li, Phys. Rev. A 82, 063422 (2010)
(minimum-time bounded-frequency control, solution of bang-bang form);
G. Hochma and M. Margaliot, arXiv:1407.3437 (maximum principle for the
spectral radius of bilinear systems, bang-bang and singular controls); J. Zu
and Y. Li, Appl. Math. J. Chin. Univ. 33, 253 (2018) (Hill-equation stability
criteria by optimal control).  Against that literature:

* rederivations — the bang-bang structure of the optimum, its switching at
  the turning points (x = 0 and p = 0), the per-cycle value artanh(d) =
  ln(omega2/omega1) of the quarter-period drive, and the *fact* that the
  growth-rate optimum is bang-bang, which is the infinite-horizon
  (Dinkelbach) form of the minimum-time problems above;
* NOT found in the literature (and therefore stated as this module's own):
  (i) the per-zero inequality ln r(t2)/r(t1) <= n artanh(d) for *arbitrary
  measurable* |s| <= 1 (non-periodic included), with its equality
  characterisation, obtained by using the Prüfer phase as the clock so that
  the maximisation is pointwise and singular arcs are absent — Theorem A
  and its Floquet corollary with the winding-number normalisation; (ii) the
  closed-form rate optimum lambda*(d) of Eq. (10)-(12) and its comparison
  with the quarter-period drive — Theorem B.

Theorem A (per-zero form; equality case classical)
--------------------------------------------------
Prüfer variables in the (x, p/omega0) plane [H. Prüfer, Math. Ann. 95, 499
(1926); the polar form in which zeros of x are exactly the phases
theta = pi/2 mod pi and the phase is monotone, cf. arXiv:2508.12369 Sec. 2 and
the Sturm comparison theory of Hartman, "Ordinary Differential Equations",
Ch. XI]:

    x = r cos theta,   p = x' = -omega0 r sin theta.

Differentiating (1) gives the exact first-order system (derivation in the
docstring of :func:`prufer_rhs`)

    d ln r / dt = (omega0 d / 2) s(t) sin 2 theta,                         (3)
    d theta / dt = omega0 (1 + d s(t) cos^2 theta).                         (4)

Because 1 + d s cos^2 theta >= 1 - d > 0, theta is strictly increasing and
bi-Lipschitz in t, so it can serve as the clock: for any admissible s,

    d ln r / d theta = g(theta, s) := (d s sin 2theta / 2) / (1 + d s cos^2 theta).   (5)

At fixed theta, g is a Möbius function of s with

    dg/ds = (d sin 2theta / 2) / (1 + d s cos^2 theta)^2,

whose sign is sign(sin 2theta) *independently of s*: g is monotone in s at
every phase, so the pointwise maximiser is unique wherever sin 2theta != 0,

    s*(theta) = sgn(sin 2theta),                                            (6)

and, substituting u = 1 +/- d cos^2 theta,

    int_0^pi max_s g(theta, s) dtheta
      = (1/2) ln(1 + d) - (1/2) ln(1 - d) = artanh(d).                      (7)

Hence for every solution of (1) and every t1 < t2 with
theta(t2) - theta(t1) = n pi (n = number of zeros of x in [t1, t2)):

    ln r(t2)/r(t1) <= n artanh(d),    equality iff s = sgn(sin 2theta) a.e.  (8)

Corollary A (Floquet form; the Z-normalisation is this module's).  Let s be T-periodic with rho(S_F) > 1.  The
eigenvector of the leading (real) multiplier generates a solution whose
Prüfer angle advances by exactly Z pi over one period (Z >= 1 an integer:
the direction returns to itself and theta is increasing) and whose r is
multiplied by rho(S_F); so

    ln rho(S_F) <= Z artanh(d),     Z = zeros per period of the unstable
                                       Floquet solution (its winding number), (9)

with equality iff the drive is the quarter-period bang-bang.  Z <= T omega2/pi
(omega2 = omega0 sqrt(1 + d): a pi-window of theta lasts at least
int_0^pi dtheta / (omega0 (1 + d cos^2 theta)) = pi/omega2), so on the
principal tongue (T ~ pi/omega0) Z = 1 and (9) is literally the conjectured
(2).  This is the normalisation the outline uses: "per cycle" = per zero of
the unstable Floquet solution = per drive period on the principal tongue; on
the m-th tongue Z = m.  Nothing is assumed about periodicity in Theorem A,
so non-periodic modulations obey (8) as well.

Why no Pontryagin machinery is needed (and where the prior work stops).  The
papers above pose finite-horizon minimum-time or maximal-amplitude problems
for two-level or bounded controls and solve them with the maximum principle.
Written as a control problem in the
original time, maximising ln r over |s| <= 1 has the pseudo-Hamiltonian
H = p_r (omega0 d/2) s sin 2theta + p_theta omega0 (1 + d s cos^2 theta), affine
in s, so the maximum principle makes the optimum bang-bang wherever the
switching function is nonzero and leaves the singular arcs (switching
function vanishing on an interval) to a separate higher-order analysis
[Boscain, Sigalotti, Sugny, PRX Quantum 2, 030203 (2021), Sec. 6.3 (regular
vs singular controls) and Sec. 9.2 (switching function); for the
spectral-radius objective on bilinear systems see Hochma & Margaliot,
arXiv:1407.3437, where singular controls do occur in general].  The time
change theta -> clock reduces the problem to a *static* pointwise
maximisation (5)-(6), whose "switching function" (d/2) sin 2theta vanishes
only at isolated phases: there are no singular arcs at all, and the same
change of clock is what the optimal-control derivations of the classical
Hill-equation stability criteria exploit [Zu & Li, Appl. Math. J. Chin.
Univ. 33, 253 (2018)].

Theorem B (rate optimum; the growth-rate form of the minimum-time problem)
--------------------------------------------------------------------------
Theorem B is the growth-rate (infinite-horizon) version of the minimum-time
problems of Andresen et al., Stefanatos et al. and Piccoli–Kulkarni: bang-bang
structure is theirs, the closed form below was not found.  The per-cycle
optimum is *not* the rate optimum.  Writing dt/dtheta =
h(theta, s)/omega0, h = 1/(1 + d s cos^2 theta), the largest Lyapunov exponent
over all admissible s is, by Dinkelbach's parametric equivalence for ratio
objectives [W. Dinkelbach, Management Sci. 13, 492 (1967)],

    Lambda_max(d) = omega0 lambda*(d),   Phi(lambda*) = 0,
    Phi(lambda) := int_0^pi max_s [g(theta, s) - lambda h(theta, s)] dtheta,   (10)

Phi being strictly decreasing with Phi(0) = artanh(d) > 0.  The maximiser in
(10) is again pointwise and bang-bang: d/ds of (d s sin 2theta/2 - lambda)/
(1 + d s cos^2 theta) has the sign of cos^2 theta (tan theta + lambda), so

    s*(theta; lambda) = +1 on (0, pi/2) and (pi - arctan lambda, pi),
                        -1 on (pi/2, pi - arctan lambda):                     (11)

switch *down* at x = 0 exactly, switch *up* the Prüfer angle arctan(lambda)
*before* the turning point.  With c^2 = cos^2(pi - arctan lambda) = 1/(1 + lambda^2)
the integrals close:

    G(lambda) = int g(theta, s*) dtheta = artanh(d / (1 + lambda^2)),
    tau(lambda) = int h(theta, s*) dtheta
       = [pi/2 + arctan(lambda/sqrt(1+d))]/sqrt(1+d)
       + [pi/2 - arctan(lambda/sqrt(1-d))]/sqrt(1-d),                        (12)
    Phi(lambda) = G(lambda) - lambda tau(lambda).

lambda = 0 in (12) is the quarter-period bang-bang (G = artanh d,
tau = omega0 T*), and Phi(artanh(d)/(omega0 T*)) > 0 strictly, so the
quarter-period drive is *never* rate-optimal: the optimum dwells longer than
a quarter period at the high frequency (arctan(lambda*/sqrt(1+d)) more) and
less at the low one, gains artanh(d/(1 + lambda*^2)) < artanh(d) per cycle,
and cycles faster.  Relative excess of lambda* over artanh(d)/(omega0 T*):
O(d^2) — 0.025 % at d = 0.05, 1.0 % at 0.3, 3.1 % at 0.5, 13.5 % at 0.8
(:func:`rate_optimum`; cross-checked in tests/test_floquet_bound.py against a
free two-dwell optimisation of the Meissner trace, E. Meissner, Schweiz.
Bauzeit. 72, 95 (1918)).  The loss threshold of the candidate,
Q_th = pi/(2 ln lambda_max), is a per-period statement at the principal
resonance and is untouched.

Multimode.  For ``kind='coupling'`` the drive commutes with K0, the 2N x 2N
monodromy is block-diagonal in the normal modes, each mode sees the *same*
relative depth d, and (9) holds mode by mode with the mode's own Z_k —
:func:`multimode_check` verifies the block structure to roundoff.

``kind='mass'`` also commutes with K0 (block structure unchanged), but the
relative depth is *not* d: K(t) = K0 + m_sq d s(t) I gives mode k the
frequency-squared omega_k^2 + m_sq d s(t), i.e. its own relative depth

    d_k = d m_sq / omega_k^2,      d_max = d m_sq / omega_min^2,            (13)

so (9) applies mode by mode as ln rho_k <= Z_k artanh(d_k) and **only where
d_max < 1**.  Beyond that the softest mode's omega_k^2(t) changes sign within
the cycle (an inverted oscillator; artanh is not even defined) and NO claim of
this module covers the drive: ``modulated_K`` accepts any ``mass_sq > 0``, so
d m_sq >= omega_min^2 is reachable and is *out of contract*, not a violation —
e.g. m_sq = 0.6 at depth 1 on ``harmonic_chain_K(6, 0.05)`` gives d_max = 2.99
(Dirichlet) or 240 (periodic, whose softest mode is the m^2 = 0.0025 zero-mode
regulator), and a ratio against the inapplicable artanh(d) is meaningless.  :func:`mass_mode_depths` returns
(13) with its validity flag and is the gate to call before quoting the bound
for a mass drive; :func:`multimode_check` with ``kind='mass'`` refuses the
excluded regime outright.  For a non-commuting pattern K(t) = K0 + d s(t) P
the natural depth is the operator-norm one, d_eff = d ||K0^{-1/2} P K0^{-1/2}||
(so (1 - d_eff) K0 <= K(t) <= (1 + d_eff) K0), and the *conjectured* extension
is the rate bound ln rho(S_F)/T <= omega_max lambda*(d_eff); it is tested,
not proved (the 1-D Prüfer clock has no multi-mode analogue).  What IS proved for the
non-commuting case is Theorem M of :mod:`vacuum.floquet.bounds_multimode`:
ln rho(S_F)/T <= (d/2) ||K0^{-1/4} P K0^{-1/4}|| <= omega_max d_eff / 2 for every measurable
|s| <= 1 (number-norm Gronwall), and ``multimode_check(kind='pattern')`` reports the ratio
against it (``proved_ratio``) next to the conjectured one.

Everything is pure-functional over arrays.  All monodromies below are exact
ordered products of static symplectic exponentials
(:func:`vacuum.floquet.mode_propagator_coefficients`), so every number is the
theorem's own object evaluated to roundoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from .monodromy import mode_propagator_coefficients

__all__ = [
    "per_cycle_bound",
    "prufer_rhs",
    "RateOptimum",
    "rate_optimum",
    "rate_optimum_control",
    "bang_bang_control",
    "piecewise_monodromy",
    "ln_multiplier",
    "prufer_winding",
    "bound_ratio",
    "CONTROL_FAMILIES",
    "random_controls",
    "optimize_control",
    "adversarial_search",
    "operator_norm_depth",
    "mass_mode_depths",
    "multimode_check",
]


# --------------------------------------------------------------------------
# The theorem's objects
# --------------------------------------------------------------------------


def per_cycle_bound(depth):
    """artanh(d): the per-cycle bound (9) (per zero of the unstable solution)."""
    d = np.asarray(depth, dtype=float)
    if np.any(d < 0.0) or np.any(d >= 1.0):
        raise ValueError("depth must satisfy 0 <= d < 1")
    return np.arctanh(d)


def prufer_rhs(theta, s, depth, omega0=1.0):
    """(d ln r/dt, d theta/dt) of Eqs. (3)-(4) at Prüfer angle theta, control s.

    Derivation: with x = r cos theta, p = -omega0 r sin theta, the equations
    x' = p and p' = -omega0^2 (1 + d s) x read
        (r'/r) cos theta - theta' sin theta = -omega0 sin theta,
        (r'/r) sin theta + theta' cos theta =  omega0 (1 + d s) cos theta;
    multiplying by (cos, sin) and adding gives r'/r = (omega0 d s/2) sin 2theta,
    multiplying by (-sin, cos) and adding gives theta' = omega0 (1 + d s cos^2 theta).
    """
    theta = np.asarray(theta, dtype=float)
    s = np.asarray(s, dtype=float)
    d = float(depth)
    w = float(omega0)
    return (0.5 * w * d * s * np.sin(2.0 * theta), w * (1.0 + d * s * np.cos(theta) ** 2))


@dataclass(frozen=True)
class RateOptimum:
    """The rate optimum of Theorem B (omega0 = 1 units unless stated).

    ``lambda_star``: maximal Lyapunov exponent / omega0; ``theta_up``: Prüfer
    angle of the up-switch (pi - arctan lambda*); ``tau_high``/``tau_low``:
    dwell times at omega2 = omega0 sqrt(1+d) / omega1 = omega0 sqrt(1-d);
    ``period``: the optimal cycle; ``ln_growth_per_cycle`` =
    artanh(d/(1 + lambda*^2)) < artanh(d); ``quarter_period_rate`` =
    artanh(d)/T* (the quarter-period bang-bang) and ``excess`` =
    lambda*/quarter_period_rate - 1.
    """

    depth: float
    omega0: float
    lambda_star: float
    theta_up: float
    tau_high: float
    tau_low: float
    period: float
    ln_growth_per_cycle: float
    quarter_period_rate: float
    excess: float


def _G(lam, d):
    return np.arctanh(d / (1.0 + lam * lam))


def _tau(lam, d):
    w2 = np.sqrt(1.0 + d)
    w1 = np.sqrt(1.0 - d)
    return (0.5 * np.pi + np.arctan(lam / w2)) / w2 + (0.5 * np.pi - np.arctan(lam / w1)) / w1


def _Phi(lam, d):
    return _G(lam, d) - lam * _tau(lam, d)


def rate_optimum(depth, omega0=1.0):
    """Theorem B: the maximal growth rate over all bounded modulations of depth d.

    Solves Phi(lambda) = 0 of Eq. (10) with the closed forms (12) by Brent's
    method to 1e-15 (Phi is strictly decreasing, Phi(0) = artanh d > 0).
    """
    d = float(depth)
    if not 0.0 < d < 1.0:
        raise ValueError("rate_optimum needs 0 < d < 1")
    w0 = float(omega0)
    hi = np.arctanh(d) / _tau(0.0, d)  # quarter-period rate: Phi > 0 there
    while _Phi(hi, d) > 0.0:
        hi *= 2.0
    lam = brentq(lambda x: _Phi(x, d), 0.0, hi, xtol=1e-15, maxiter=500)
    w2 = np.sqrt(1.0 + d)
    w1 = np.sqrt(1.0 - d)
    tau_high = (0.5 * np.pi + np.arctan(lam / w2)) / (w0 * w2)
    tau_low = (0.5 * np.pi - np.arctan(lam / w1)) / (w0 * w1)
    q_rate = w0 * np.arctanh(d) / _tau(0.0, d)
    return RateOptimum(
        depth=d, omega0=w0, lambda_star=float(w0 * lam), theta_up=float(np.pi - np.arctan(lam)),
        tau_high=float(tau_high), tau_low=float(tau_low), period=float(tau_high + tau_low),
        ln_growth_per_cycle=float(_G(lam, d)), quarter_period_rate=float(q_rate),
        excess=float(w0 * lam / q_rate - 1.0),
    )


def bang_bang_control(depth, omega0=1.0):
    """The per-cycle optimum (6) as ``(values, durations)``: +1 for a quarter
    period at omega2 (from p = 0 to x = 0), then -1 for a quarter period at
    omega1 — the two-piece Meissner drive of ``bang_bang_bound``."""
    d = float(depth)
    w0 = float(omega0)
    w2 = w0 * np.sqrt(1.0 + d)
    w1 = w0 * np.sqrt(1.0 - d)
    return np.array([1.0, -1.0]), np.array([0.5 * np.pi / w2, 0.5 * np.pi / w1])


def rate_optimum_control(depth, omega0=1.0):
    """The rate optimum (11) as ``(values, durations)`` (+1 dwell first)."""
    r = rate_optimum(depth, omega0)
    return np.array([1.0, -1.0]), np.array([r.tau_high, r.tau_low])


# --------------------------------------------------------------------------
# Exact batched monodromies of piecewise-constant controls
# --------------------------------------------------------------------------


def _step(M, w2, dt):
    """M <- P(w2, dt) M for stacked 2 x 2 maps (P as in mode_propagator_coefficients)."""
    c, sw, ws = mode_propagator_coefficients(w2, dt)
    out = np.empty_like(M)
    out[..., 0, 0] = c * M[..., 0, 0] + sw * M[..., 1, 0]
    out[..., 0, 1] = c * M[..., 0, 1] + sw * M[..., 1, 1]
    out[..., 1, 0] = -ws * M[..., 0, 0] + c * M[..., 1, 0]
    out[..., 1, 1] = -ws * M[..., 0, 1] + c * M[..., 1, 1]
    return out


def _as_batch(s, tau):
    s = np.atleast_2d(np.asarray(s, dtype=float))
    tau = np.asarray(tau, dtype=float)
    tau = np.broadcast_to(tau, s.shape) if tau.ndim <= 1 else tau
    if tau.shape != s.shape:
        raise ValueError(f"durations {tau.shape} do not match controls {s.shape}")
    if np.any(np.abs(s) > 1.0 + 1e-15):
        raise ValueError("controls must satisfy |s| <= 1")
    if np.any(tau <= 0.0):
        raise ValueError("durations must be positive")
    return s, tau


def piecewise_monodromy(s, tau, depth, omega0=1.0):
    """Exact monodromies (B, 2, 2) of piecewise-constant controls.

    ``s`` (B, M) piece values in [-1, 1], ``tau`` (M,) or (B, M) durations;
    omega^2 = omega0^2 (1 + depth s) on each piece.  Ordered product of exact
    static exponentials (no discretisation error): the object the theorem
    speaks about, since a piecewise-constant s is itself admissible.
    """
    s, tau = _as_batch(s, tau)
    w2 = float(omega0) ** 2 * (1.0 + float(depth) * s)
    B, M = s.shape
    Mon = np.zeros((B, 2, 2))
    Mon[:, 0, 0] = 1.0
    Mon[:, 1, 1] = 1.0
    for j in range(M):
        Mon = _step(Mon, w2[:, j], tau[:, j])
    return Mon


def ln_multiplier(Mon):
    """ln max|eig| of stacked 2 x 2 symplectic maps (0 when |tr| <= 2)."""
    tr = Mon[..., 0, 0] + Mon[..., 1, 1]
    at = np.abs(tr)
    out = np.zeros_like(at)
    un = at > 2.0
    out[un] = np.log(0.5 * (at[un] + np.sqrt(at[un] ** 2 - 4.0)))
    return out


def _leading_eigvec(Mon):
    """Real eigenvector of the leading multiplier (B, 2); NaN where stable."""
    tr = Mon[:, 0, 0] + Mon[:, 1, 1]
    at = np.abs(tr)
    lam = np.where(at > 2.0, 0.5 * (at + np.sqrt(np.maximum(at**2 - 4.0, 0.0))), np.nan)
    lam = lam * np.sign(tr)
    v1 = np.stack([Mon[:, 0, 1], lam - Mon[:, 0, 0]], axis=1)
    v2 = np.stack([lam - Mon[:, 1, 1], Mon[:, 1, 0]], axis=1)
    use1 = np.linalg.norm(v1, axis=1) >= np.linalg.norm(v2, axis=1)
    v = np.where(use1[:, None], v1, v2)
    n = np.linalg.norm(v, axis=1)
    return v / np.where(n > 0, n, 1.0)[:, None]


def prufer_winding(s, tau, depth, omega0=1.0, *, max_substep_angle=0.25 * np.pi):
    """Winding number Z of the unstable Floquet solution (zeros per period).

    Propagates the leading eigenvector of the exact monodromy through the
    pieces in substeps small enough that the Prüfer angle theta = atan2(-p/omega0, x)
    advances by less than pi per substep (the angle is monotone, so wrapped
    increments are unambiguous) and returns round(Delta theta / pi).  NaN for
    stable rows (no real eigenvector).
    """
    s, tau = _as_batch(s, tau)
    w0 = float(omega0)
    w2 = w0**2 * (1.0 + float(depth) * s)
    Mon = piecewise_monodromy(s, tau, depth, w0)
    v = _leading_eigvec(Mon)
    unstable = np.isfinite(v[:, 0])
    theta_acc = np.zeros(s.shape[0])
    x = v[:, 0].copy()
    p = v[:, 1].copy()
    prev = np.arctan2(-p / w0, x)
    wmax = np.sqrt(np.max(w2))
    for j in range(s.shape[1]):
        n_sub = int(np.ceil(wmax * np.max(tau[:, j]) / max_substep_angle)) + 1
        dt = tau[:, j] / n_sub
        c, sw, ws = mode_propagator_coefficients(w2[:, j], dt)
        for _ in range(n_sub):
            x, p = c * x + sw * p, -ws * x + c * p
            ang = np.arctan2(-p / w0, x)
            inc = np.mod(ang - prev, 2.0 * np.pi)  # in [0, 2 pi); true increment in (0, pi)
            theta_acc += inc
            prev = ang
    Z = np.round(theta_acc / np.pi)
    return np.where(unstable, Z, np.nan), np.where(unstable, theta_acc, np.nan)


def bound_ratio(s, tau, depth, omega0=1.0):
    """ln rho(S_F) / (Z artanh d) per row of a batch (the theorem's ratio, Eq. (9)).

    Returns a dict with ``ln_multiplier``, ``Z`` (NaN when stable), ``ratio``
    (0 when stable) and ``theta_advance``.  The bound says ratio <= 1.
    """
    s, tau = _as_batch(s, tau)
    Mon = piecewise_monodromy(s, tau, depth, omega0)
    ln = ln_multiplier(Mon)
    Z, adv = prufer_winding(s, tau, depth, omega0)
    bound = np.where(np.isfinite(Z), Z, 1.0) * per_cycle_bound(depth)
    ratio = np.where(ln > 0.0, ln / bound, 0.0)
    return {"ln_multiplier": ln, "Z": Z, "ratio": ratio, "theta_advance": adv,
            "monodromy": Mon}


# --------------------------------------------------------------------------
# Adversarial controls
# --------------------------------------------------------------------------

CONTROL_FAMILIES = ("fourier", "clipped_fourier", "sign_fourier", "piecewise", "smooth")


def _fourier_series(rng, n, M, kmax=6):
    phi = 2.0 * np.pi * (np.arange(M) + 0.5) / M
    K = rng.integers(1, kmax + 1, size=n)
    out = np.zeros((n, M))
    for i in range(n):
        k = np.arange(1, K[i] + 1)
        a = rng.normal(size=K[i]) / k
        b = rng.normal(size=K[i]) / k
        out[i] = a @ np.cos(np.outer(k, phi)) + b @ np.sin(np.outer(k, phi))
    return out


def random_controls(rng, n, M, family, *, kmax=6):
    """A batch (n, M) of admissible piecewise-constant controls, |s| <= 1.

    Families: ``'fourier'`` (random multi-frequency series with up to ``kmax``
    harmonics, scaled to max|s| = 1), ``'clipped_fourier'`` (the same, scaled
    up by 1-4 and clipped: flat-topped), ``'sign_fourier'`` (its sign:
    bang-bang with random switch phases), ``'piecewise'`` (2-8 random
    segments of uniform values), ``'smooth'`` (tanh of an amplified random
    series: smooth and saturating).
    """
    if family not in CONTROL_FAMILIES:
        raise ValueError(f"family must be one of {CONTROL_FAMILIES}, got {family!r}")
    if family == "piecewise":
        out = np.zeros((n, M))
        for i in range(n):
            nseg = int(rng.integers(2, 9))
            cuts = np.sort(rng.choice(np.arange(1, M), size=nseg - 1, replace=False))
            vals = rng.uniform(-1.0, 1.0, size=nseg)
            if rng.random() < 0.5:
                vals = np.sign(vals)
            out[i] = np.repeat(vals, np.diff(np.concatenate(([0], cuts, [M]))))
        return out
    f = _fourier_series(rng, n, M, kmax)
    amp = np.max(np.abs(f), axis=1, keepdims=True)
    f = f / np.where(amp > 0, amp, 1.0)
    if family == "fourier":
        return f
    if family == "clipped_fourier":
        return np.clip(f * rng.uniform(1.0, 4.0, size=(n, 1)), -1.0, 1.0)
    if family == "sign_fourier":
        return np.sign(f + 1e-300)
    return np.tanh(f * rng.uniform(1.0, 6.0, size=(n, 1)))


def _trace_and_gradient(s, tau, depth, omega0):
    """tr S_F and d tr / d s_j for a batch: prefix/suffix products, exact dP/d(omega^2)."""
    s, tau = _as_batch(s, tau)
    d = float(depth)
    w0 = float(omega0)
    w2 = w0**2 * (1.0 + d * s)
    B, M = s.shape
    P = np.empty((B, M, 2, 2))
    dP = np.empty((B, M, 2, 2))
    w = np.sqrt(w2)  # d < 1 => w2 > 0
    wt = w * tau
    c, sn = np.cos(wt), np.sin(wt)
    P[..., 0, 0] = c
    P[..., 0, 1] = sn / w
    P[..., 1, 0] = -w * sn
    P[..., 1, 1] = c
    # d/d(w2) with dw/dw2 = 1/(2w)
    dP[..., 0, 0] = -tau * sn / (2.0 * w)
    dP[..., 0, 1] = (tau * w * c - sn) / (2.0 * w**3)
    dP[..., 1, 0] = -(sn + wt * c) / (2.0 * w)
    dP[..., 1, 1] = dP[..., 0, 0]
    I2 = np.broadcast_to(np.eye(2), (B, 2, 2))
    pre = [I2]  # pre[j] = P_{j-1} ... P_0
    for j in range(M):
        pre.append(P[:, j] @ pre[-1])
    suf = [I2] * (M + 1)  # suf[j] = P_{M-1} ... P_{j+1}
    for j in range(M - 2, -1, -1):
        suf[j] = suf[j + 1] @ P[:, j + 1]
    Mon = pre[-1]
    tr = Mon[:, 0, 0] + Mon[:, 1, 1]
    grad = np.empty((B, M))
    for j in range(M):
        G = suf[j] @ dP[:, j] @ pre[j]
        grad[:, j] = (G[:, 0, 0] + G[:, 1, 1]) * (w0**2 * d)
    return tr, grad, Mon


def optimize_control(depth, period, M, *, method="gradient", rng=None, omega0=1.0,
                     n_restarts=8, n_iter=300, batch=128, elite_frac=0.1, s0=None,
                     objective=None):
    """Maximise |tr S_F| (hence ln rho) over s in [-1, 1]^M on a uniform grid.

    ``method='gradient'``: projected gradient ascent on the exact trace with
    the analytic gradient (:func:`_trace_and_gradient`), ``n_restarts``
    random starts, normalised steps with a decaying step size.
    ``method='cem'``: cross-entropy method — Gaussian population (``batch``)
    clipped to the box, elites re-fit each iteration.  ``objective``
    (callable s-batch -> values to maximise) overrides the single-mode trace
    for the multimode search (then only 'cem' is available).  Returns a dict
    with the best control, its exact ``ln_multiplier``, ``Z`` and ``ratio``
    (single mode) or the objective value (custom).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    tau = np.full(M, float(period) / M)
    d, w0 = float(depth), float(omega0)

    def score(sb):
        if objective is not None:
            return np.asarray(objective(sb), dtype=float)
        tr, _, _ = _trace_and_gradient(sb, tau, d, w0)
        return np.abs(tr)

    if method == "gradient":
        if objective is not None:
            raise ValueError("gradient ascent is single-mode only; use method='cem'")
        best_s, best_v = None, -np.inf
        starts = [s0] if s0 is not None else []
        starts += [random_controls(rng, 1, M, "fourier")[0] for _ in range(n_restarts)]
        for st in starts:
            s = np.clip(np.asarray(st, dtype=float), -1.0, 1.0)[None, :]
            step = 0.5
            v_prev = -np.inf
            for it in range(n_iter):
                tr, g, _ = _trace_and_gradient(s, tau, d, w0)
                v = abs(tr[0])
                if v < v_prev - 1e-12:
                    step *= 0.5
                v_prev = max(v, v_prev)
                g = g * np.sign(tr)[:, None]
                nrm = np.linalg.norm(g)
                if nrm == 0.0:
                    break
                s = np.clip(s + step * g / nrm, -1.0, 1.0)
                if step < 1e-6:
                    break
            tr, _, _ = _trace_and_gradient(s, tau, d, w0)
            if abs(tr[0]) > best_v:
                best_v, best_s = abs(tr[0]), s[0].copy()
    elif method == "cem":
        mu = np.zeros(M) if s0 is None else np.clip(np.asarray(s0, dtype=float), -1, 1)
        sig = np.full(M, 0.7)
        n_elite = max(2, int(elite_frac * batch))
        best_s, best_v = None, -np.inf
        for it in range(n_iter):
            pop = np.clip(mu[None, :] + sig[None, :] * rng.normal(size=(batch, M)), -1.0, 1.0)
            if best_s is not None:
                pop[0] = best_s
            v = score(pop)
            idx = np.argsort(v)[::-1][:n_elite]
            if v[idx[0]] > best_v:
                best_v, best_s = float(v[idx[0]]), pop[idx[0]].copy()
            mu = 0.7 * pop[idx].mean(axis=0) + 0.3 * mu
            sig = np.maximum(0.7 * pop[idx].std(axis=0) + 0.3 * sig, 1e-4)
            if np.max(sig) < 1e-3:
                break
    else:
        raise ValueError("method must be 'gradient' or 'cem'")
    out = {"control": best_s, "durations": tau, "period": float(period), "depth": d,
           "method": method, "M": int(M)}
    if objective is None:
        br = bound_ratio(best_s[None, :], tau, d, w0)
        out.update(ln_multiplier=float(br["ln_multiplier"][0]), Z=float(br["Z"][0]),
                   ratio=float(br["ratio"][0]))
    else:
        out["objective"] = float(best_v)
    return out


def adversarial_search(depths=(0.1, 0.3, 0.5, 0.8), *, seed=0, n_random=400, M=64,
                       omega0=1.0, optimizers=("gradient", "cem"), n_iter=200,
                       n_restarts=6, periods_over_Tstar=(0.85, 1.0, 1.15, 2.0, 2.4)):
    """The battery that tries to break Eq. (9) on admissible piecewise controls.

    For each depth: ``n_random`` controls per family in ``CONTROL_FAMILIES``
    at random periods T in [0.5, 3] pi/omega0 (winding numbers Z = 1..3), the
    two optimisers at the listed periods (in units of the quarter-period
    bang-bang cycle T*), the exact quarter-period bang-bang itself (the
    canary that must give ratio 1) and the rate optimum (11).  Returns a dict
    of rows and the global ``max_ratio`` / ``bang_bang_ratio``.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    canary = []
    for d in depths:
        d = float(d)
        vals, durs = bang_bang_control(d, omega0)
        T_star = float(np.sum(durs))
        br = bound_ratio(vals[None, :], durs[None, :], d, omega0)
        canary.append({"depth": d, "ratio": float(br["ratio"][0]), "Z": float(br["Z"][0]),
                       "ln_multiplier": float(br["ln_multiplier"][0]), "period": T_star})
        vr, dr = rate_optimum_control(d, omega0)
        brr = bound_ratio(vr[None, :], dr[None, :], d, omega0)
        rows.append({"depth": d, "family": "rate_optimum", "period": float(np.sum(dr)),
                     "ratio": float(brr["ratio"][0]), "Z": float(brr["Z"][0]),
                     "ln_multiplier": float(brr["ln_multiplier"][0]), "n": 1})
        for fam in CONTROL_FAMILIES:
            s = random_controls(rng, n_random, M, fam)
            T = rng.uniform(0.5, 3.0, size=n_random) * np.pi / omega0
            tau = np.repeat(T[:, None] / M, M, axis=1)
            b = bound_ratio(s, tau, d, omega0)
            k = int(np.argmax(b["ratio"]))
            rows.append({"depth": d, "family": fam, "period": float(T[k]),
                         "ratio": float(b["ratio"][k]), "Z": float(b["Z"][k]),
                         "ln_multiplier": float(b["ln_multiplier"][k]), "n": int(n_random),
                         "n_unstable": int(np.sum(b["ln_multiplier"] > 0))})
        for f in periods_over_Tstar:
            for meth in optimizers:
                r = optimize_control(d, f * T_star, M, method=meth, rng=rng, omega0=omega0,
                                     n_iter=n_iter, n_restarts=n_restarts)
                rows.append({"depth": d, "family": f"opt_{meth}", "period": float(f * T_star),
                             "ratio": r["ratio"], "Z": r["Z"], "ln_multiplier": r["ln_multiplier"],
                             "n": 1})
    finite = [r for r in rows if np.isfinite(r["ratio"])]
    worst = max(finite, key=lambda r: r["ratio"])
    return {"rows": rows, "canary": canary, "max_ratio": float(worst["ratio"]),
            "worst_row": worst, "bang_bang_ratio_max_dev": float(max(abs(c["ratio"] - 1.0)
                                                                     for c in canary)),
            "M": int(M), "n_random": int(n_random), "seed": int(seed)}


# --------------------------------------------------------------------------
# Multimode
# --------------------------------------------------------------------------


def operator_norm_depth(K0, P, depth):
    """d_eff = depth * ||K0^{-1/2} P K0^{-1/2}||_2 (so (1 -/+ d_eff) K0 bracket K(t))."""
    K0 = np.asarray(K0, dtype=float)
    P = np.asarray(P, dtype=float)
    w2, U = np.linalg.eigh(0.5 * (K0 + K0.T))
    if np.min(w2) <= 0.0:
        raise ValueError("K0 must be positive definite")
    R = (U / np.sqrt(w2)) @ U.T
    return float(depth) * float(np.linalg.norm(R @ (0.5 * (P + P.T)) @ R, 2))


def mass_mode_depths(K0, mass_sq, depth):
    """Per-mode relative depths of a mass drive, Eq. (13), with the validity gate.

    K(t) = K0 + mass_sq * depth * s(t) I sends mode k to omega_k^2 + mass_sq
    depth s(t), i.e. relative depth d_k = depth mass_sq / omega_k^2.  Theorem A
    applies mode by mode as ln rho_k <= Z_k artanh(d_k), and only while
    d_max = depth mass_sq / omega_min^2 < 1: at d_max >= 1 the softest mode's
    omega_k^2(t) changes sign inside the cycle and no bound of this module
    covers the drive (``modulated_K`` itself validates only mass_sq > 0).

    Returns
    -------
    dict with ``d_modes`` (ascending omega_k), ``d_max``, ``omega``,
    ``valid`` (bool: d_max < 1) and ``mass_sq_max`` = omega_min^2 / depth, the
    largest mass_sq for which the bound is in contract at this depth.
    """
    K0 = np.asarray(K0, dtype=float)
    w2, _ = np.linalg.eigh(0.5 * (K0 + K0.T))
    if np.min(w2) <= 0.0:
        raise ValueError("K0 must be positive definite")
    d = float(depth)
    if d < 0.0:
        raise ValueError("depth must be >= 0")
    m_sq = float(mass_sq)
    if m_sq <= 0.0:
        raise ValueError("mass_sq must be > 0 (the m^2 being modulated)")
    d_modes = d * m_sq / w2
    return {"d_modes": d_modes, "d_max": float(np.max(d_modes)),
            "omega": np.sqrt(w2), "valid": bool(np.max(d_modes) < 1.0),
            "mass_sq_max": float("inf") if d == 0.0 else float(np.min(w2) / d)}


def multimode_check(K0, s, period, depth, *, kind="coupling", pattern=None):
    """Test Eq. (9) (and its conjectured extension) on a lattice with a common control.

    ``s`` (M,) piecewise-constant profile values in [-1, 1] on equal pieces
    of ``period``.  ``kind='mass'``: K(t) = K0 + mass_sq depth s I (``pattern``
    carries mass_sq) — each mode gets its own depth d_k of
    :func:`mass_mode_depths` and the per-mode ratios are taken against
    artanh(d_k); the excluded regime d_max >= 1 raises (the bound is not
    defined there, so returning a ratio would be a false claim).
    ``kind='coupling'``: K(t) = K0 (1 + depth s) — returns
    the full-monodromy ln rho, the per-mode ln rho_k and Z_k, the per-mode
    ratios ln rho_k/(Z_k artanh d), and the block-diagonalisation defect
    |ln rho_full - max_k ln rho_k|.  ``kind='pattern'``: K(t) = K0 + depth s P
    — returns ln rho of the full map, d_eff, and the rate ratio
    ln rho / (T omega_max lambda*(d_eff)) of the conjectured multimode bound.
    """
    from .drives import modulated_K
    from .monodromy import floquet_map, multiplier

    K0 = np.asarray(K0, dtype=float)
    s = np.asarray(s, dtype=float).ravel()
    M = s.size
    T = float(period)
    pieces = tuple((float(v), 1.0 / M) for v in s)
    w2, _ = np.linalg.eigh(0.5 * (K0 + K0.T))
    omega = np.sqrt(w2)
    if kind == "coupling":
        drv = modulated_K(K0, depth, 2.0 * np.pi / T, pieces)
        fm = floquet_map(drv)
        ln_full = float(np.log(max(fm.multiplier, 1.0)))
        # per mode: omega_k^2 (1 + d s) == omega0 = omega_k in the theorem
        tau = np.full(M, T / M)
        lnk = np.empty(omega.size)
        Zk = np.empty(omega.size)
        for k, wk in enumerate(omega):
            b = bound_ratio(s[None, :], tau, depth, wk)
            lnk[k] = b["ln_multiplier"][0]
            Zk[k] = b["Z"][0]
        with np.errstate(invalid="ignore"):
            ratio = np.where(lnk > 0, lnk / (np.where(np.isfinite(Zk), Zk, 1.0) * np.arctanh(depth)), 0.0)
        return {"kind": kind, "ln_full": ln_full, "ln_modes": lnk, "Z_modes": Zk,
                "ratio_modes": ratio, "max_ratio": float(np.max(ratio)),
                "block_defect": float(abs(ln_full - np.max(lnk))), "exact": fm.exact,
                "symplectic_defect": fm.symplectic_defect}
    if kind == "pattern":
        if pattern is None:
            raise ValueError("kind='pattern' needs a pattern")
        P = np.asarray(pattern, dtype=float)
        if P.shape == (K0.shape[0],):
            P = np.diag(P)
        drv = modulated_K(K0, depth, 2.0 * np.pi / T, pieces, kind="pattern", pattern=P)
        fm = floquet_map(drv)
        ln_full = float(np.log(max(fm.multiplier, 1.0)))
        d_eff = operator_norm_depth(K0, P, depth)
        w_max = float(np.max(omega))
        rate_bound = T * w_max * rate_optimum(d_eff).lambda_star if d_eff > 0 else np.inf
        cyc_bound = np.arctanh(d_eff) * T * w_max * np.sqrt(1.0 + d_eff) / np.pi
        # Theorem M (bounds_multimode): the proved rate bound (d/2)||K0^{-1/4} P K0^{-1/4}|| and the
        # refined conjecture (||C||/||A||) lambda*(d_eff); both ratios are reported next to the
        # conjectured omega_max lambda*(d_eff) one.
        from .bounds_multimode import multimode_bound_proved
        mb = multimode_bound_proved(K0, P, depth)
        return {"kind": kind, "ln_full": ln_full, "d_eff": d_eff, "omega_max": w_max,
                "rate_ratio": float(ln_full / rate_bound), "cycle_count_ratio": float(ln_full / cyc_bound),
                "proved_bound": mb["rate_bound"],
                "proved_ratio": float(ln_full / (T * mb["rate_bound"])) if mb["rate_bound"] > 0 else 0.0,
                "refined_bound": mb["refined_conjecture"],
                "refined_ratio": float(ln_full / (T * mb["refined_conjecture"])) if d_eff > 0 else 0.0,
                "norm_C": mb["norm_C"], "exact": fm.exact, "symplectic_defect": fm.symplectic_defect}
    if kind == "mass":
        if pattern is None:
            raise ValueError("kind='mass' needs mass_sq (passed as `pattern`)")
        m_sq = float(np.asarray(pattern).ravel()[0])
        info = mass_mode_depths(K0, m_sq, depth)
        if not info["valid"]:
            raise ValueError(
                f"mass drive out of contract: d_max = {info['d_max']:.6g} >= 1 "
                f"(depth {depth} x mass_sq {m_sq} vs omega_min^2 "
                f"{float(np.min(omega) ** 2):.6g}); the softest mode inverts within "
                f"the cycle and Theorem A does not apply.  Largest in-contract "
                f"mass_sq at this depth: {info['mass_sq_max']:.6g}"
            )
        drv = modulated_K(K0, depth, 2.0 * np.pi / T, pieces, kind="mass", mass_sq=m_sq)
        fm = floquet_map(drv)
        ln_full = float(np.log(max(fm.multiplier, 1.0)))
        tau = np.full(M, T / M)
        lnk = np.empty(omega.size)
        Zk = np.empty(omega.size)
        for k, wk in enumerate(omega):
            # mode k is x'' + omega_k^2 (1 + d_k s) x = 0 with d_k of Eq. (13)
            b = bound_ratio(s[None, :], tau, float(info["d_modes"][k]), wk)
            lnk[k] = b["ln_multiplier"][0]
            Zk[k] = b["Z"][0]
        bound = np.where(np.isfinite(Zk), Zk, 1.0) * np.arctanh(info["d_modes"])
        ratio = np.where(lnk > 0, lnk / bound, 0.0)
        return {"kind": kind, "ln_full": ln_full, "ln_modes": lnk, "Z_modes": Zk,
                "d_modes": info["d_modes"], "d_max": info["d_max"],
                "ratio_modes": ratio, "max_ratio": float(np.max(ratio)),
                "block_defect": float(abs(ln_full - np.max(lnk))), "exact": fm.exact,
                "symplectic_defect": fm.symplectic_defect}
    raise ValueError("kind must be 'coupling', 'mass' or 'pattern'")
