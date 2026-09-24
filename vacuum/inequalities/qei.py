"""QEI stress-tester core (M3.4): the pulled-back sampling operator.

Central design decision (docs/PLAN_LAYERS_2_3.md, M3.4): **the entire
smeared-energy minimization collapses to a Williamson problem.**

Construction
------------
The local bare energy density of the chain H = (1/2) sum_i [p_i^2 + m^2 x_i^2
+ (x_{i+1} - x_i)^2] is chosen PSD by construction:

    h_i = (1/2) p_i^2 + (1/2) m^2 x_i^2 + (1/4)(x_{i+1}-x_i)^2 + (1/4)(x_i-x_{i-1})^2

(half of each adjacent bond's PSD quadratic form; boundary/wall bonds are
assigned wholly to their single site so that sum_i h_i = H exactly). As a
quadratic form, h_i is a (2N, 2N) PSD matrix with the convention
energy = (1/2) R^T h_i R, R = (x_1..x_N, p_1..p_N) (docs/API.md).

Free Heisenberg evolution pulls the density back to t = 0,
h_i(t) = S(t)^T h_i S(t) with S(t) = vacuum.core.chain_propagator(K, t).
For a sampling function f the *sampling operator* is the quadrature sum

    O_f = sum_a w_a f(t_a)^2 S(t_a)^T h_i S(t_a),

accumulated in place (never stored per-t). The smeared normal-ordered energy
of a Gaussian state V is E_f(V) = (1/2) Tr(O_f (V - V_vac)), linear in V, so
its infimum over ALL physical states (V + i Omega/2 >= 0; the minimizer of a
quadratic form's expectation is a Gaussian state, so Gaussian minimization is
global) is attained at the "vacuum of O_f":

    E_min(f) = (1/2) sum_k sigma_k(O_f) - (1/2) Tr(O_f V_vac),

with sigma_k the symplectic eigenvalues of O_f, solved by one Williamson
problem whose decomposition O_f = S_w diag(sigma, sigma) S_w^T also exhibits
the extremal state V_star = (1/2) (S_w S_w^T)^{-1} — the pure Gaussian
O_f-ground state (a multimode squeezed vacuum, exactly the optimal-state
structure Flanagan identifies in 2d, PRD 56, 4922 (1997), Sec. II). See
`qei_minimize` for how the spectrum and the exhibited state are split
between an inversion-free route and the one `williamson` call: on the
lattice O_f is intrinsically PSD-singular (zone-boundary packets off-site
have zero group velocity and are exactly invisible to the worldline), so
V^{-1/2}-based Williamson on the raw O_f is a division cliff.

Mode-support regularization (numerical hazard, spec M3.4): modes outside the
sampled light cone contribute near-null symplectic directions to O_f. O_f
is therefore assembled restricted to the light-cone site support of the
worldline (|j - site| <= t_max + pad) and then trimmed to the sites carrying
at least `weight_floor` of the peak diagonal weight. The restriction is
*exact* for the truncated operator: any physical covariance on the support
extends to a physical lattice state by direct sum, and any lattice state
restricts to a physical support state, so inf_V (1/2)Tr(O_pad V) over the
full lattice equals the restricted minimum, where O_pad is O_f with its
(f(dist)^2- and Lieb-Robinson-suppressed) outside rows dropped. The
truncation error is measured, not assumed: E_min under pad and weight_floor
variation must agree (support-independence check, tested).

Unphysical-state detection uses `vacuum.core.precision.symplectic_margins_mp`
RAW — no entropy clamps: `entropy_from_nu`'s floor clamp is correct for
entropies and wrong for detecting nu < 1/2 inputs (spec M3.4 hazard note).

The 2d bound (which constant is the target?)
--------------------------------------------
Verified against the papers (fetched 2026-08-31 from arXiv ar5iv renderings):

- E. E. Flanagan, Phys. Rev. D 56, 4922 (1997) [gr-qc/9706006], Eq. (8):
  for the massless scalar in 2d Minkowski, along an inertial worldline with
  smearing rho(t) >= 0 (his Eq. (1) normalizes int rho = 1, but both sides
  are 1-homogeneous in rho, so the bound is normalization-free),

      int dt rho(t) <T_00> >= -(1/(24 pi)) int dt rho'(t)^2 / rho(t),

  and this is OPTIMAL (sharp): it is the maximum possible lower bound,
  attained by multimode squeezed states. With rho = f^2 (this module's
  convention), rho'^2/rho = 4 f'^2, giving

      int dt f(t)^2 <T_00>  >=  -(1/(6 pi)) int dt f'(t)^2.          [SHARP]

- C. J. Fewster and S. P. Eveson, Phys. Rev. D 58, 084010 (1998)
  [gr-qc/9805024], Eq. (37): same quantity bounded by
  -(1/(16 pi)) int f'^2/f (their f is the smearing weight itself); with
  f -> f^2 this is -(1/(4 pi)) int f'^2 — weaker than Flanagan's optimal
  bound by exactly the factor 3/2 they state. Their Eq. (40) is the 4d bound
  -(1/(16 pi^2)) int ((sqrt f)'')^2, i.e. -(1/(16 pi^2)) int f''^2 in the
  f^2 convention (kept here for the Layer-3.5 4d scan; the classic
  Ford-Roman PRD 55, 2082 (1997) Lorentzian-sampling constant is
  -3/(32 pi^2 tau_0^4)).

The sharp-constant anchor (spec anchor ii) therefore targets Flanagan's
optimal 2d constant: qei_ratio -> 1 with `constant='sharp'` as
(a/tau_0, m tau_0) -> 0.

4d machinery (papers candidate seed, kept out of the heavy tests): the same
construction on the Srednicki radial lattice — T_00 near the origin assembled
from low-l partial waves of H_l (vacuum.core.radial, Srednicki PRL 71, 666
(1993), Eq. 10), spatially smeared over a small ball (mandatory in 4d),
sector-by-sector Williamson minimization with the (2l+1) degeneracy sum.

All functions are pure over arrays: no hidden state, no input mutation
(Layer-6 JAX-mirror pre-positioning).

Note on switching: the spec points at vacuum/detectors/switching.py for the
analytic-derivative switching library. That module exists, but exposes only
chi and its first derivative; the sampling-function optimization here also
needs f'' and the family-parameter derivatives df/dtheta, so the smooth
f-families are implemented locally with the same pure data-plus-functions
style.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple, Optional

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve_sylvester

from vacuum.core import (
    Omega,
    ground_state_cov,
    harmonic_chain_K,
    reduce as _reduce_cov,
    symplectic_eigenvalues,
    symplectic_inverse,
    williamson,
)
from vacuum.core.precision import symplectic_margins_mp
from vacuum.core.radial import srednicki_K

__all__ = [
    # bounds / constants
    "SHARP_CONSTANT_2D",
    "FEWSTER_EVESON_CONSTANT_2D",
    "FORD_ROMAN_4D_LORENTZIAN",
    "qei_bound_2d",
    "fprime_sq_integral",
    # sampling functions / families
    "FHandle",
    "SamplingFamily",
    "gaussian_f",
    "cos2_f",
    "gaussian_family",
    "cos2_family",
    # operators and minimization
    "chain_energy_density_form",
    "sampling_operator",
    "SamplingOp",
    "qei_minimize",
    "QEIResult",
    "smeared_energy",
    "qei_ratio",
    "qei_audit",
    "embed_extremal",
    # states
    "squeezed_pocket",
    # physicality (raw margins, no clamps)
    "QEIUnphysicalError",
    "QEIViolationError",
    "min_symplectic_margin",
    "assert_physical",
    # sampling-function optimization
    "optimize_sampling",
    "OptimizeSamplingResult",
    # 4d / Srednicki radial machinery (papers-agent entry points)
    "radial_ball_weights",
    "radial_energy_density_form",
    "radial_sampling_operator",
    "radial_ball_qei",
    # cancellation-free sector minimizer (mode space) + precision cross-check
    "ModeSpaceResult",
    "mode_space_minimize",
    "radial_sector_minimize",
    "chain_worldline_minimize",
    "mode_space_extremal_state",
    "qei_minimize_mp",
    "shell_volume",
    "naive_volume",
    # continuum momentum-space solution (no lattice): worldline, ball, rho expansion
    "FEWSTER_EVESON_CONSTANT_4D",
    "lorentzian_f",
    "sqrt_lorentzian_f",
    "bump_f",
    "gaussian_poly_f",
    "fhat_of",
    "fsecond_sq_integral",
    "momentum_grid",
    "MomentumResult",
    "worldline_momentum_minimize",
    "ball_momentum_minimize",
    "rho2_coefficient",
    "momentum_qei_ratio",
    "optimize_sampling_momentum",
]

# --------------------------------------------------------------------------
# Bound constants (all in the int f(t)^2 <T_00> dt smearing convention)
# --------------------------------------------------------------------------

#: Flanagan, PRD 56, 4922 (1997), Eq. (8) with rho = f^2: the OPTIMAL (sharp)
#: 2d massless bound  E >= -SHARP_CONSTANT_2D * int f'(t)^2 dt.
SHARP_CONSTANT_2D = 1.0 / (6.0 * math.pi)

#: Fewster-Eveson, PRD 58, 084010 (1998), Eq. (37) with their f -> f^2:
#: E >= -FEWSTER_EVESON_CONSTANT_2D * int f'(t)^2 dt (3/2 weaker than sharp).
FEWSTER_EVESON_CONSTANT_2D = 1.0 / (4.0 * math.pi)

#: Fewster-Eveson, PRD 58, 084010 (1998), Eq. (40), massless 4d worldline bound
#: in the f^2 convention: int f^2 <:T_00:> dt >= -FEWSTER_EVESON_CONSTANT_4D
#: * int f''(t)^2 dt. Not sharp ("we do not expect our bound to be optimal",
#: their Sec. IV); the sharp constant for Gaussian f is 0.4575 of it
#: (:func:`worldline_momentum_minimize`).
FEWSTER_EVESON_CONSTANT_4D = 1.0 / (16.0 * math.pi**2)

#: Ford-Roman, PRD 55, 2082 (1997): 4d Lorentzian-sampling constant in
#: rho >= -FORD_ROMAN_4D_LORENTZIAN / tau_0^4. Reference value for the 4d
#: scan (not used by the 2d anchors).
FORD_ROMAN_4D_LORENTZIAN = 3.0 / (32.0 * math.pi**2)


class QEIUnphysicalError(ValueError):
    """An input covariance violates nu >= 1/2 (V + i Omega/2 not PSD)."""


class QEIViolationError(ValueError):
    """A *physical* state violated a QEI bound — an audit event, never expected."""


# --------------------------------------------------------------------------
# Sampling functions (smooth f-families with analytic derivatives)
# --------------------------------------------------------------------------


class FHandle(NamedTuple):
    """A sampling function as pure data + functions.

    f, df, d2f are vectorized callables (t ndarray -> ndarray) for f, f', f''.
    [t_lo, t_hi] is the (effective) support used for quadrature; fp2 is the
    analytic value of int f'(t)^2 dt when known (None otherwise).
    """

    f: Callable[[np.ndarray], np.ndarray]
    df: Callable[[np.ndarray], np.ndarray]
    d2f: Callable[[np.ndarray], np.ndarray]
    t_lo: float
    t_hi: float
    fp2: Optional[float]
    label: str
    #: optional analytic Fourier transform of the smearing weight,
    #: f2hat(nu) = int f(t)^2 exp(i nu t) dt (vectorized over nu). Used by the
    #: mode-space minimizer to sample the exact, un-windowed smearing; None
    #: sends it to the same composite Gauss-Legendre rule as `sampling_operator`.
    f2hat: Optional[Callable[[np.ndarray], np.ndarray]] = None
    #: optional analytic value of int f''(t)^2 dt (the Fewster-Eveson 4d bound
    #: functional); None sends :func:`fsecond_sq_integral` to quadrature.
    fpp2: Optional[float] = None


def gaussian_f(tau0, t0=0.0, n_sigmas=4.5):
    """Gaussian sampling function f(t) = exp(-(t-t0)^2 / (2 tau0^2)).

    Analytic: f' = -(t-t0)/tau0^2 f, f'' = ((t-t0)^2/tau0^4 - 1/tau0^2) f,
    int f'^2 dt = sqrt(pi)/(2 tau0), int f''^2 dt = 3 sqrt(pi)/(4 tau0^3).
    Effective support t0 +- n_sigmas*tau0 (tail weight of f^2 beyond 4.5
    sigma is ~1.6e-9 relative).
    """
    tau0 = float(tau0)
    if tau0 <= 0:
        raise ValueError(f"tau0 must be > 0, got {tau0}")
    t0 = float(t0)

    def f(t):
        s = (np.asarray(t, dtype=float) - t0) / tau0
        return np.exp(-0.5 * s * s)

    def df(t):
        u = np.asarray(t, dtype=float) - t0
        return -(u / tau0**2) * f(t)

    def d2f(t):
        u = np.asarray(t, dtype=float) - t0
        return (u * u / tau0**4 - 1.0 / tau0**2) * f(t)

    def f2hat(nu):
        # int exp(-(t-t0)^2/tau0^2) e^{i nu t} dt = tau0 sqrt(pi) e^{-nu^2 tau0^2/4} e^{i nu t0}
        nu = np.asarray(nu, dtype=float)
        return (tau0 * math.sqrt(math.pi) * np.exp(-0.25 * nu * nu * tau0 * tau0)
                * np.exp(1j * nu * t0))

    return FHandle(
        f=f,
        df=df,
        d2f=d2f,
        t_lo=t0 - n_sigmas * tau0,
        t_hi=t0 + n_sigmas * tau0,
        fp2=math.sqrt(math.pi) / (2.0 * tau0),
        label=f"gaussian(tau0={tau0:g}, t0={t0:g})",
        f2hat=f2hat,
        fpp2=3.0 * math.sqrt(math.pi) / (4.0 * tau0**3),
    )


def cos2_f(T, t0=0.0):
    """Compact-support bump f(t) = cos^2(pi (t-t0) / (2T)) on |t-t0| < T.

    Analytic: with u = pi (t-t0) / T,
    f' = -(pi/(2T)) sin(u), f'' = -(pi^2/(2T^2)) cos(u),
    int f'^2 dt = pi^2 / (4 T), int f''^2 dt = pi^4 / (4 T^3). Exactly zero
    outside [t0-T, t0+T]; f'' jumps at the edges (f is in W^{2,2}, the class
    Fewster's lecture notes (arXiv:1208.5399, Sec. 1.3) give for the 4d
    bound). Fourier transform of the weight, f^2 = cos^4 = (3 + 4 cos 2u
    + cos 4u)/8 on the support:
    f2hat(nu) = (T/8)[6 sinc(nu T) + 4 sinc((nu +- pi/T) T) + sinc((nu +- 2pi/T) T)]
    (sinc x = sin x / x, both signs summed) times e^{i nu t0}.
    """
    T = float(T)
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    t0 = float(t0)

    def _mask(t):
        u = np.asarray(t, dtype=float) - t0
        return u, (np.abs(u) < T)

    def f(t):
        u, inside = _mask(t)
        out = np.zeros_like(u)
        out[inside] = np.cos(0.5 * math.pi * u[inside] / T) ** 2
        return out

    def df(t):
        u, inside = _mask(t)
        out = np.zeros_like(u)
        out[inside] = -(math.pi / (2.0 * T)) * np.sin(math.pi * u[inside] / T)
        return out

    def d2f(t):
        u, inside = _mask(t)
        out = np.zeros_like(u)
        out[inside] = -(math.pi**2 / (2.0 * T**2)) * np.cos(math.pi * u[inside] / T)
        return out

    def f2hat(nu):
        nu = np.asarray(nu, dtype=float)
        a = math.pi / T

        def sinc(x):
            return np.sinc(x * T / math.pi)  # sin(x T)/(x T)

        val = (T / 8.0) * (6.0 * sinc(nu) + 4.0 * sinc(nu + a) + 4.0 * sinc(nu - a)
                           + sinc(nu + 2.0 * a) + sinc(nu - 2.0 * a))
        return val * np.exp(1j * nu * t0)

    return FHandle(
        f=f,
        df=df,
        d2f=d2f,
        t_lo=t0 - T,
        t_hi=t0 + T,
        fp2=math.pi**2 / (4.0 * T),
        label=f"cos2(T={T:g}, t0={t0:g})",
        f2hat=f2hat,
        fpp2=math.pi**4 / (4.0 * T**3),
    )


def lorentzian_f(tau, n_taus=200.0):
    """Algebraic tail  f(t) = 1 / (1 + (t/tau)^2)  (the smearing weight f^2 is
    the squared Lorentzian).

    Analytic (s = t/tau): f' = -2 s (1+s^2)^{-2} / tau,
    f'' = (6 s^2 - 2)(1+s^2)^{-3} / tau^2, int f'^2 dt = pi/(4 tau),
    int f''^2 dt = 3 pi/(4 tau^3), and
    f2hat(nu) = (pi tau / 2)(1 + |nu| tau) e^{-|nu| tau}  (C^1 at nu = 0).
    The support [-n_taus tau, n_taus tau] is only used by quadrature
    fallbacks (the tail of f''^2 beyond it is ~ n_taus^{-5}/5).
    """
    tau = float(tau)
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    def f(t):
        s = np.asarray(t, dtype=float) / tau
        return 1.0 / (1.0 + s * s)

    def df(t):
        s = np.asarray(t, dtype=float) / tau
        return -2.0 * s / (1.0 + s * s) ** 2 / tau

    def d2f(t):
        s = np.asarray(t, dtype=float) / tau
        return (6.0 * s * s - 2.0) / (1.0 + s * s) ** 3 / tau**2

    def f2hat(nu):
        x = np.abs(np.asarray(nu, dtype=float)) * tau
        return 0.5 * math.pi * tau * (1.0 + x) * np.exp(-x)

    return FHandle(f=f, df=df, d2f=d2f, t_lo=-n_taus * tau, t_hi=n_taus * tau,
                   fp2=math.pi / (4.0 * tau), label=f"lorentzian(tau={tau:g})",
                   f2hat=f2hat, fpp2=3.0 * math.pi / (4.0 * tau**3))


def sqrt_lorentzian_f(tau, n_taus=200.0):
    """Ford-Roman sampling:  f(t) = (1 + (t/tau)^2)^{-1/2}, i.e. the smearing
    weight f^2 is the Lorentzian tau^2/(t^2 + tau^2) of Ford & Roman, PRD 55,
    2082 (1997) (their normalization tau/(pi (t^2 + tau^2)) is a constant
    factor that cancels in every ratio).

    Analytic (s = t/tau): f' = -s (1+s^2)^{-3/2} / tau,
    f'' = (2 s^2 - 1)(1+s^2)^{-5/2} / tau^2, int f'^2 dt = pi/(8 tau),
    int f''^2 dt = 27 pi/(128 tau^3) — so the Fewster-Eveson bound for this
    weight is -27/(2048 pi^2 tau^4) in the Ford-Roman normalization, 9/64 of
    the Ford-Roman bound 3/(32 pi^2 tau^4) (FE 1998, remark after Eq. (40));
    f2hat(nu) = pi tau e^{-|nu| tau} (a cusp at nu = 0, so momentum-space
    quadratures converge algebraically for this handle).
    """
    tau = float(tau)
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    def f(t):
        s = np.asarray(t, dtype=float) / tau
        return (1.0 + s * s) ** -0.5

    def df(t):
        s = np.asarray(t, dtype=float) / tau
        return -s * (1.0 + s * s) ** -1.5 / tau

    def d2f(t):
        s = np.asarray(t, dtype=float) / tau
        return (2.0 * s * s - 1.0) * (1.0 + s * s) ** -2.5 / tau**2

    def f2hat(nu):
        x = np.abs(np.asarray(nu, dtype=float)) * tau
        return math.pi * tau * np.exp(-x)

    return FHandle(f=f, df=df, d2f=d2f, t_lo=-n_taus * tau, t_hi=n_taus * tau,
                   fp2=math.pi / (8.0 * tau), label=f"sqrt_lorentzian(tau={tau:g})",
                   f2hat=f2hat, fpp2=27.0 * math.pi / (128.0 * tau**3))


def bump_f(T, p=2.0):
    """Compact power bump  f(t) = (1 - u^2)^p on |u| < 1, u = t/T, p >= 2.

    p = 2 is the C^1 quartic bump of the papers notebook (f'' jumps at the
    edges); larger p is smoother. Analytic, with B the Beta function and
    I(a, b) = int_{-1}^{1} u^{2a} (1 - u^2)^b du = B(a + 1/2, b + 1):
    f' = -2 p u (1-u^2)^{p-1} / T,
    f'' = [-2p (1-u^2)^{p-1} + 4 p (p-1) u^2 (1-u^2)^{p-2}] / T^2,
    int f'^2 = 4 p^2 I(1, 2p-2) / T,
    int f''^2 = [4 p^2 I(0, 2p-2) - 16 p^2 (p-1) I(1, 2p-3)
                 + 16 p^2 (p-1)^2 I(2, 2p-4)] / T^3,
    f2hat(nu) = T sqrt(pi) Gamma(2p+1) (2/(nu T))^{2p+1/2} J_{2p+1/2}(nu T)
    (the Fourier transform of (1-u^2)^{2p}; algebraic tail ~ nu^{-(2p+1)}).
    """
    from scipy.special import beta as _beta, gamma as _gamma, jv as _jv

    T = float(T)
    p = float(p)
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if p < 2.0:
        raise ValueError(f"need p >= 2 for f'' in L^2 (got p={p})")

    def _u(t):
        u = np.asarray(t, dtype=float) / T
        return u, np.abs(u) < 1.0

    def f(t):
        u, inside = _u(t)
        out = np.zeros_like(u)
        out[inside] = (1.0 - u[inside] ** 2) ** p
        return out

    def df(t):
        u, inside = _u(t)
        out = np.zeros_like(u)
        out[inside] = -2.0 * p * u[inside] * (1.0 - u[inside] ** 2) ** (p - 1.0) / T
        return out

    def d2f(t):
        u, inside = _u(t)
        out = np.zeros_like(u)
        ui = u[inside]
        out[inside] = (-2.0 * p * (1.0 - ui**2) ** (p - 1.0)
                       + 4.0 * p * (p - 1.0) * ui**2 * (1.0 - ui**2) ** (p - 2.0)) / T**2
        return out

    def I(a, b):
        return float(_beta(a + 0.5, b + 1.0))

    fp2 = 4.0 * p * p * I(1, 2 * p - 2) / T
    fpp2 = (4.0 * p * p * I(0, 2 * p - 2) - 16.0 * p * p * (p - 1.0) * I(1, 2 * p - 3)
            + 16.0 * p * p * (p - 1.0) ** 2 * I(2, 2 * p - 4)) / T**3
    m = 2.0 * p
    mu = m + 0.5
    c0 = T * math.sqrt(math.pi) * float(_gamma(m + 1.0))
    lim0 = c0 / float(_gamma(mu + 1.0))

    def f2hat(nu):
        x = np.abs(np.asarray(nu, dtype=float)) * T
        small = x < 1e-6
        xs = np.where(small, 1.0, x)
        val = c0 * (2.0 / xs) ** mu * _jv(mu, xs)
        return np.where(small, lim0, val)

    return FHandle(f=f, df=df, d2f=d2f, t_lo=-T, t_hi=T, fp2=fp2,
                   label=f"bump(T={T:g}, p={p:g})", f2hat=f2hat, fpp2=fpp2)


class SamplingFamily(NamedTuple):
    """Parametrized smooth f-family with analytic parameter derivatives.

    make(theta) -> FHandle; df_dtheta(theta, ts) -> (n_params, nt) array of
    partial f / partial theta_j evaluated on the time grid ts.
    """

    make: Callable[[np.ndarray], FHandle]
    df_dtheta: Callable[[np.ndarray, np.ndarray], np.ndarray]
    n_params: int
    label: str


def gaussian_family(t0=0.0, n_sigmas=4.5):
    """One-parameter Gaussian family, theta = [tau0]."""

    def make(theta):
        return gaussian_f(float(np.asarray(theta).ravel()[0]), t0=t0, n_sigmas=n_sigmas)

    def df_dtheta(theta, ts):
        tau0 = float(np.asarray(theta).ravel()[0])
        u = np.asarray(ts, dtype=float) - t0
        fv = np.exp(-0.5 * (u / tau0) ** 2)
        # d/d tau0 exp(-u^2/(2 tau0^2)) = (u^2 / tau0^3) f
        return (u * u / tau0**3 * fv)[None, :]

    return SamplingFamily(make=make, df_dtheta=df_dtheta, n_params=1, label="gaussian")


def cos2_family(t0=0.0):
    """One-parameter compact-bump family, theta = [T]."""

    def make(theta):
        return cos2_f(float(np.asarray(theta).ravel()[0]), t0=t0)

    def df_dtheta(theta, ts):
        T = float(np.asarray(theta).ravel()[0])
        u = np.asarray(ts, dtype=float) - t0
        inside = np.abs(u) < T
        out = np.zeros_like(u)
        # d/dT cos^2(pi u / 2T) = sin(pi u / T) * (pi u) / (2 T^2)
        out[inside] = np.sin(math.pi * u[inside] / T) * (math.pi * u[inside]) / (2.0 * T**2)
        return out[None, :]

    return SamplingFamily(make=make, df_dtheta=df_dtheta, n_params=1, label="cos2")


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------


def fprime_sq_integral(f: FHandle, *, use_analytic=True, nodes_per_panel=24, n_panels=None):
    """int f'(t)^2 dt over [t_lo, t_hi] (analytic value used when available)."""
    if use_analytic and f.fp2 is not None:
        return float(f.fp2)
    span = f.t_hi - f.t_lo
    if n_panels is None:
        n_panels = max(8, int(math.ceil(span)))
    ts, ws = _gauss_legendre_grid(f.t_lo, f.t_hi, n_panels, nodes_per_panel)
    d = f.df(ts)
    return float(np.sum(ws * d * d))


def qei_bound_2d(f: FHandle, constant="sharp", **quad_kw):
    """Lower bound (a NEGATIVE number) on int f(t)^2 <:T_00:> dt in 2d.

    constant='sharp': Flanagan PRD 56, 4922 (1997), Eq. (8) (rho = f^2), the
    optimal massless 2d bound -(1/(6 pi)) int f'^2 — the anchor target.
    constant='fewster_eveson': FE PRD 58, 084010 (1998), Eq. (37), the
    3/2-weaker bound -(1/(4 pi)) int f'^2.
    """
    if constant == "sharp":
        C = SHARP_CONSTANT_2D
    elif constant == "fewster_eveson":
        C = FEWSTER_EVESON_CONSTANT_2D
    else:
        raise ValueError(f"constant must be 'sharp' or 'fewster_eveson', got {constant!r}")
    return -C * fprime_sq_integral(f, **quad_kw)


# --------------------------------------------------------------------------
# Local energy density (chain) as sparse PSD rank-1 terms
# --------------------------------------------------------------------------


def _chain_density_terms(N, site, m, bc):
    """PSD rank-1 decomposition of the chain's local energy density h_site.

    Returns a list of (coeff, x_idx, x_val, p_idx, p_val): the quadratic form
    is h = sum coeff * u u^T with u the sparse 2N-vector (x part, p part),
    and energy = (1/2) R^T h R. Interior bonds are split half/half between
    their two sites; Dirichlet wall bonds belong wholly to their single site,
    so that sum_site h_site = diag(K, I) exactly (regression-tested).
    """
    N = int(N)
    site = int(site)
    if not 0 <= site < N:
        raise ValueError(f"site must be in [0, {N}), got {site}")
    if bc not in ("periodic", "dirichlet"):
        raise ValueError(f"bc must be 'periodic' or 'dirichlet', got {bc!r}")
    m = float(m)
    terms = []
    # (1/2) p_site^2
    terms.append((1.0, np.array([], dtype=int), np.array([]), np.array([site]), np.array([1.0])))
    # (1/2) m^2 x_site^2
    if m != 0.0:
        terms.append((m * m, np.array([site]), np.array([1.0]), np.array([], dtype=int), np.array([])))
    if N == 1:
        if bc == "dirichlet":  # two wall bonds x_0^2, both wholly on site 0
            terms.append((2.0, np.array([0]), np.array([1.0]), np.array([], dtype=int), np.array([])))
        return terms
    empty_i = np.array([], dtype=int)
    empty_v = np.array([])
    # left bond
    if site > 0:
        terms.append((0.5, np.array([site - 1, site]), np.array([-1.0, 1.0]), empty_i, empty_v))
    elif bc == "periodic":
        terms.append((0.5, np.array([N - 1, 0]), np.array([-1.0, 1.0]), empty_i, empty_v))
    else:  # dirichlet wall bond (x_0 - 0)^2, wholly on site 0
        terms.append((1.0, np.array([0]), np.array([1.0]), empty_i, empty_v))
    # right bond
    if site < N - 1:
        terms.append((0.5, np.array([site, site + 1]), np.array([-1.0, 1.0]), empty_i, empty_v))
    elif bc == "periodic":
        terms.append((0.5, np.array([N - 1, 0]), np.array([-1.0, 1.0]), empty_i, empty_v))
    else:  # dirichlet wall bond (0 - x_{N-1})^2, wholly on site N-1
        terms.append((1.0, np.array([N - 1]), np.array([1.0]), empty_i, empty_v))
    return terms


def _terms_to_dense(N, terms):
    """Dense (2N, 2N) PSD matrix from sparse rank-1 terms."""
    h = np.zeros((2 * N, 2 * N))
    for coeff, xi, xv, pi_, pv in terms:
        u = np.zeros(2 * N)
        u[xi] = xv
        u[N + pi_] = pv
        h += coeff * np.outer(u, u)
    return 0.5 * (h + h.T)


def chain_energy_density_form(N, site, m=0.0, bc="dirichlet"):
    """PSD local energy density h_site of the harmonic chain, (2N, 2N).

    Energy = (1/2) R^T h R; sum over all sites reproduces diag(K, I) with
    K = harmonic_chain_K(N, m, bc) exactly (bond-splitting convention in
    :func:`_chain_density_terms`).
    """
    return _terms_to_dense(N, _chain_density_terms(N, site, m, bc))


# --------------------------------------------------------------------------
# Quadrature grid
# --------------------------------------------------------------------------


def _gauss_legendre_grid(t_lo, t_hi, n_panels, nodes_per_panel):
    """Composite Gauss-Legendre nodes/weights over [t_lo, t_hi]."""
    x, w = leggauss(int(nodes_per_panel))
    edges = np.linspace(float(t_lo), float(t_hi), int(n_panels) + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    ts = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    ws = (half[:, None] * w[None, :]).ravel()
    return ts, ws


# --------------------------------------------------------------------------
# Sampling-operator assembly (accumulated, restricted to the mode support)
# --------------------------------------------------------------------------


class SamplingOp(NamedTuple):
    """Assembled sampling operator, restricted to its light-cone mode support.

    O is the (2 n_s, 2 n_s) PSD quadratic form on the support modes
    (block ordering within the support); support lists the site indices;
    V_vac_sub is the lattice vacuum reduced to the support (the normal-
    ordering reference); dO holds optional parameter derivatives
    partial O / partial theta_j in the same basis.
    """

    O: np.ndarray
    support: np.ndarray
    V_vac_sub: np.ndarray
    N: int
    site: int
    t_max: float
    n_nodes: int
    dO: tuple
    meta: dict


def _assemble_pullback(K, terms, fvals_of_t, t_lo, t_hi, support, panel_width,
                       nodes_per_panel, dweights_of_t=None, chunk=48,
                       weight_floor=0.0):
    """Accumulate O = sum_a w_a f(t_a)^2 S(t_a)^T h S(t_a) on the support.

    K: (N, N) SPD coupling matrix; terms: sparse rank-1 decomposition of h;
    fvals_of_t: callable f(t); support: sorted site indices; the propagator
    S(t) is evaluated spectrally from one eigh(K) (exact normal-mode form,
    vacuum.core.dynamics.chain_propagator's construction) and only its
    support rows are ever formed. dweights_of_t (optional): callable
    ts -> (n_params, nt) of d(f^2)/d theta_j for gradient assembly.

    weight_floor > 0 applies the mode-support trim AFTER accumulation: sites
    whose total diagonal weight O_xx[j,j] + O_pp[j,j] falls below
    weight_floor * max are dropped from the support before the Williamson
    stage. Such sites sit outside the effectively sampled light cone (their
    weight ~ f(dist)^2 or the Lieb-Robinson skin); keeping them makes O
    numerically singular and the Williamson conditioning hopeless, dropping
    them changes E_min at the weight scale — which is what the
    support-independence check measures (spec M3.4 hazard note).

    Returns (O, dO_tuple, V_vac_sub, support_kept, n_nodes, panel_width_used).
    """
    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    w2, U = np.linalg.eigh(0.5 * (K + K.T))
    if np.min(w2) <= 0:
        raise ValueError(f"K must be positive definite: min eig {np.min(w2):.3e}")
    w = np.sqrt(w2)

    if panel_width is None:
        # >= ~4 GL panels per period of the fastest normal mode: the GL-10
        # panel error parameter (width * omega / (4 n))^{2n} is then ~1e-24.
        panel_width = min(1.6, 2.4 / float(np.max(w)))
    span = float(t_hi) - float(t_lo)
    n_panels = max(1, int(math.ceil(span / panel_width)))
    ts, wq = _gauss_legendre_grid(t_lo, t_hi, n_panels, nodes_per_panel)
    fvals = np.asarray(fvals_of_t(ts), dtype=float)
    node_w = wq * fvals * fvals  # >= 0
    gmats = None
    if dweights_of_t is not None:
        gvals = np.asarray(dweights_of_t(ts), dtype=float)  # (n_params, nt)
        gmats = gvals * wq[None, :]

    S_idx = np.asarray(support, dtype=int)
    n_s = S_idx.size
    U_S = U[S_idx, :]

    coeffs = np.array([c for c, *_ in terms])
    ahat = np.stack([(xv[:, None] * U[xi, :]).sum(axis=0) if xi.size else np.zeros(N)
                     for _, xi, xv, _, _ in terms])          # (n_terms, N)
    bhat = np.stack([(pv[:, None] * U[pi_, :]).sum(axis=0) if pi_.size else np.zeros(N)
                     for _, _, _, pi_, pv in terms])          # (n_terms, N)
    n_terms = len(terms)

    O = np.zeros((2 * n_s, 2 * n_s))
    dO = [np.zeros((2 * n_s, 2 * n_s)) for _ in range(gmats.shape[0])] if gmats is not None else []

    nt = ts.size
    for start in range(0, nt, chunk):
        idx = np.arange(start, min(start + chunk, nt))
        nc = idx.size
        c = np.cos(np.outer(w, ts[idx]))          # (N, nc)
        s = np.sin(np.outer(w, ts[idx]))
        sw = s / w[:, None]
        ws_ = s * w[:, None]
        # S(t)^T u for u = (a; b): top = C a - W sin b ; bottom = W^{-1} sin a + C b
        pre_top = np.empty((N, n_terms * nc))
        pre_bot = np.empty((N, n_terms * nc))
        for r in range(n_terms):
            cols = slice(r * nc, (r + 1) * nc)
            pre_top[:, cols] = c * ahat[r][:, None] - ws_ * bhat[r][:, None]
            pre_bot[:, cols] = sw * ahat[r][:, None] + c * bhat[r][:, None]
        Y = np.concatenate([U_S @ pre_top, U_S @ pre_bot], axis=0)  # (2 n_s, n_terms*nc)
        scale = np.concatenate([coeffs[r] * node_w[idx] for r in range(n_terms)])
        O += (Y * scale) @ Y.T
        for j in range(len(dO)):
            gscale = np.concatenate([coeffs[r] * gmats[j, idx] for r in range(n_terms)])
            dO[j] += (Y * gscale) @ Y.T

    O = 0.5 * (O + O.T)
    dO = tuple(0.5 * (D + D.T) for D in dO)

    if weight_floor > 0.0:
        site_w = np.diag(O)[:n_s] + np.diag(O)[n_s:]
        keep = np.flatnonzero(site_w >= weight_floor * float(np.max(site_w)))
        if keep.size < n_s:
            q = np.concatenate([keep, keep + n_s])
            O = O[np.ix_(q, q)]
            dO = tuple(D[np.ix_(q, q)] for D in dO)
            S_idx = S_idx[keep]
            U_S = U_S[keep, :]
            n_s = keep.size

    inv_w = 1.0 / w
    X = 0.5 * (U_S * inv_w) @ U_S.T   # <x x> block of the vacuum on the support
    P = 0.5 * (U_S * w) @ U_S.T       # <p p> block
    V_vac_sub = np.zeros((2 * n_s, 2 * n_s))
    V_vac_sub[:n_s, :n_s] = 0.5 * (X + X.T)
    V_vac_sub[n_s:, n_s:] = 0.5 * (P + P.T)
    return O, dO, V_vac_sub, S_idx, nt, panel_width


def _cone_support(N, site, radius, bc):
    """Site indices within the sampled light cone (lattice speed <= 1)."""
    if bc == "periodic":
        d = np.abs(np.arange(N) - site)
        d = np.minimum(d, N - d)
        return np.flatnonzero(d <= radius)
    lo = max(0, site - radius)
    hi = min(N - 1, site + radius)
    return np.arange(lo, hi + 1)


def sampling_operator(N, site, f: FHandle, *, m=0.0, bc="dirichlet", pad=8,
                      support=None, panel_width=None, nodes_per_panel=10,
                      t_lo=None, t_hi=None, df_dtheta=None, weight_floor=1e-9):
    """Assemble the pulled-back sampling operator O_f for a chain worldline.

    O_f = sum_a w_a f(t_a)^2 S(t_a)^T h_site S(t_a), accumulated over a
    composite Gauss-Legendre grid on [t_lo, t_hi] (default: the handle's
    support), restricted to the light-cone mode support
    |j - site| <= ceil(max|t|) + pad and then trimmed to the sites carrying
    at least `weight_floor` of the peak diagonal weight (spec M3.4
    regularization; the trim keeps the Williamson problem positive definite
    — outer-cone sites carry weights ~ f(dist)^2 down to below machine
    epsilon). Both pad and weight_floor are support-independence axes: the
    tests vary them and require E_min stability. Pass `support` (site
    indices) to override the cone; pass `df_dtheta` (ts -> (n_params, nt)
    array of partial f/partial theta_j) to also accumulate the parameter
    derivatives dO_j = sum_a w_a d(f^2)/d theta_j (t_a) S^T h S used by
    :func:`optimize_sampling`'s envelope gradient.
    """
    N = int(N)
    site = int(site)
    t_lo = float(f.t_lo if t_lo is None else t_lo)
    t_hi = float(f.t_hi if t_hi is None else t_hi)
    t_max = max(abs(t_lo), abs(t_hi))
    K = harmonic_chain_K(N, m, bc)
    if support is None:
        support = _cone_support(N, site, int(math.ceil(t_max)) + int(pad), bc)
    support = np.unique(np.asarray(support, dtype=int))
    terms = _chain_density_terms(N, site, m, bc)

    dweights = None
    if df_dtheta is not None:
        ffun = f.f

        def dweights(ts):
            return 2.0 * np.asarray(ffun(ts))[None, :] * np.asarray(df_dtheta(ts))

    O, dO, V_vac_sub, support, n_nodes, pw = _assemble_pullback(
        K, terms, f.f, t_lo, t_hi, support, panel_width, nodes_per_panel,
        dweights_of_t=dweights, weight_floor=weight_floor,
    )
    meta = {"m": float(m), "bc": bc, "pad": int(pad), "panel_width": float(pw),
            "nodes_per_panel": int(nodes_per_panel), "t_lo": t_lo, "t_hi": t_hi,
            "weight_floor": float(weight_floor), "label": f.label}
    return SamplingOp(O=O, support=support, V_vac_sub=V_vac_sub, N=N, site=site,
                      t_max=t_max, n_nodes=int(n_nodes), dO=dO, meta=meta)


# --------------------------------------------------------------------------
# The Williamson minimization
# --------------------------------------------------------------------------


def _symplectic_spectrum_psd(O):
    """Symplectic eigenvalues of a symmetric PSD matrix, descending.

    Inversion-free: SVD of A = O^{1/2} Omega O^{1/2} (each sigma twice), with
    tiny negative eigenvalues of O clipped to zero. Exactly the algorithm of
    `vacuum.core.symplectic_eigenvalues`, which now also accepts the singular
    PSD sampling operators generic here (zone-boundary null modes; see the
    qei_minimize docstring) — the core-API gap this copy was written around
    was closed. Kept local only to avoid a cross-layer import in the hot
    optimization loop; equality with core is regression-tested.
    """
    O = np.asarray(O, dtype=float)
    O = 0.5 * (O + O.T)
    n = O.shape[0] // 2
    w, U = np.linalg.eigh(O)
    if np.min(w) < -1e-10 * max(1.0, float(np.max(np.abs(w)))):
        raise ValueError(f"matrix is not PSD: min eigenvalue {np.min(w):.3e}")
    root = (U * np.sqrt(np.clip(w, 0.0, None))) @ U.T
    A = root @ Omega(n) @ root
    s = np.linalg.svd(0.5 * (A - A.T), compute_uv=False)
    return np.sort(s)[::-1][::2].copy()


class QEIResult(NamedTuple):
    """Result of the smeared-energy minimization (one Williamson problem).

    e_min = (1/2) sum sigma - (1/2) Tr(O V_vac_sub); V_star is the extremal
    covariance on the support block (pure Gaussian, the vacuum of the
    eps-regularized O — see :func:`qei_minimize`); achieved is
    (1/2) Tr(O (V_star - V_vac_sub)), the energy the exhibited state
    actually reaches; gap = achieved - e_min >= 0 is its measured
    optimality gap (near-null symplectic directions of O_f are squeezed
    only finitely by the exhibit); margin_star is the float64 min
    symplectic margin of V_star; cond is the ordinary eigenvalue condition
    number of the regularized operator handed to `williamson`.
    """

    e_min: float
    sigma: np.ndarray
    V_star: np.ndarray
    achieved: float
    gap: float
    trace_vac: float
    support: np.ndarray
    margin_star: float
    cond: float
    eps_reg: float


def qei_minimize(op: SamplingOp, *, eps_reg=1e-8):
    """Minimize the smeared normal-ordered energy over ALL physical states.

    inf_V (1/2) Tr(O_f (V - V_vac)) = (1/2) sum_k sigma_k(O_f)
                                      - (1/2) Tr(O_f V_vac),

    with sigma_k the symplectic eigenvalues of O_f, and the extremal state
    exhibited from ONE `williamson` call (spec M3.4).

    Numerical-hazard note, made concrete (spec M3.4): on the lattice O_f is
    intrinsically PSD-*singular* — beyond the barely-sampled long-wavelength
    modes, zone-boundary (k ~ pi) packets sitting off-site have vanishing
    group velocity, never reach the worldline, and are exactly invisible to
    the sampled density (a pure lattice artifact; in the 2d continuum every
    massless mode moves at c). Two consequences:

    - The symplectic spectrum is therefore computed WITHOUT inverting O_f,
      via the SVD of O^{1/2} Omega O^{1/2} (`core.symplectic_eigenvalues`,
      PSD-safe): every sigma_k, including the near-null ones (which
      contribute ~0 to the sum — the infimum squeezes them away), enters
      e_min stably. `core.williamson`'s V^{-1/2}-based construction would
      hit a division cliff here.
    - The extremal state is exhibited from the ONE `williamson` call on the
      regularized operator O + eps_reg*lambda_max*I (positive definite by
      construction): V_star = (1/2)(S S^T)^{-1}, a physical pure Gaussian
      (multimode squeezed — Flanagan's optimal-state structure) that
      squeezes the near-null directions finitely instead of infinitely. Its
      optimality gap `achieved - e_min` is computed exactly and reported,
      never assumed zero; the anchors assert it is small.
    """
    O = op.O
    ev = np.linalg.eigvalsh(O)
    lam_max = float(ev[-1])
    if lam_max <= 0.0:
        raise ValueError("sampling operator is identically zero on its support")
    # e_min from the inversion-free symplectic spectrum of the raw O_f
    sigma = _symplectic_spectrum_psd(O)
    trace_vac = 0.5 * float(np.einsum("ij,ji->", O, op.V_vac_sub))
    e_min = 0.5 * float(np.sum(sigma)) - trace_vac

    # extremal state from the ONE williamson call on the regularized form
    O_reg = O + (eps_reg * lam_max) * np.eye(O.shape[0])
    S, _sigma_reg = williamson(O_reg)
    Sinv = symplectic_inverse(S)
    V_star = 0.5 * (Sinv.T @ Sinv)
    V_star = 0.5 * (V_star + V_star.T)
    achieved = 0.5 * float(np.einsum("ij,ji->", O, V_star - op.V_vac_sub))
    margin_star = float(np.min(symplectic_eigenvalues(V_star)) - 0.5)
    cond = float(lam_max / (ev[0] + eps_reg * lam_max))
    return QEIResult(e_min=e_min, sigma=sigma, V_star=V_star, achieved=achieved,
                     gap=achieved - e_min, trace_vac=trace_vac,
                     support=op.support, margin_star=margin_star, cond=cond,
                     eps_reg=float(eps_reg))


def smeared_energy(V, op: SamplingOp):
    """(1/2) Tr(O_f (V - V_vac)) for a state V (full-lattice or support-sized).

    A full-lattice covariance (2N x 2N) is reduced to the support block first;
    the smeared energy reads only that block, so the restriction is exact for
    the truncated operator. No physicality gate here — use :func:`qei_audit`
    for the gated path.
    """
    V = np.asarray(V, dtype=float)
    n_s = op.support.size
    if V.shape[0] == 2 * op.N and op.N != n_s:
        V_sub = _reduce_cov(V, op.support)
    elif V.shape[0] == 2 * n_s:
        V_sub = V
    else:
        raise ValueError(
            f"V has dimension {V.shape[0]}; expected {2 * op.N} (full lattice) "
            f"or {2 * n_s} (support block)"
        )
    return 0.5 * float(np.einsum("ij,ji->", op.O, V_sub - op.V_vac_sub))


def qei_ratio(op: SamplingOp, f: FHandle, *, constant="sharp", eps_reg=1e-8):
    """E_min(f) / bound(f) — the sharp-constant diagnostic (-> 1 in 2d).

    Both numerator and denominator are negative; the ratio is positive and
    approaches 1 from below as (a/tau0, m tau0) -> 0 when `constant='sharp'`
    (Flanagan's optimal bound). Values above 1 would mean the lattice state
    beats the continuum bound — never observed and not expected at sane
    resolutions; values well below 1 at coarse resolution are lattice
    artifacts, which the Richardson anchor extrapolates away.
    """
    res = qei_minimize(op, eps_reg=eps_reg)
    bound = qei_bound_2d(f, constant=constant)
    return res.e_min / bound


def embed_extremal(res: QEIResult, op: SamplingOp):
    """Embed the extremal support-block state into the full lattice.

    Direct sum: V_star on the support modes, the lattice vacuum reduced to
    the complement on the rest, zero cross-covariance — physical because a
    direct sum of physical covariances is physical. Its smeared energy under
    the truncated operator equals `res.achieved` exactly (which sits within
    `res.gap` of e_min); under an *untruncated* operator it would differ
    further by the (measured) support-truncation error. Works for chain and
    Srednicki-radial sampling operators alike.
    """
    if op.meta.get("bc") == "srednicki":
        K = srednicki_K(op.N, op.meta["l"])
    else:
        K = harmonic_chain_K(op.N, op.meta["m"], op.meta["bc"])
    V = ground_state_cov(K)
    out = V.copy()
    N = op.N
    S_idx = op.support
    comp = np.setdiff1d(np.arange(N), S_idx)
    q_sup = np.concatenate([S_idx, S_idx + N])
    q_comp = np.concatenate([comp, comp + N])
    out[np.ix_(q_sup, q_sup)] = res.V_star
    out[np.ix_(q_sup, q_comp)] = 0.0
    out[np.ix_(q_comp, q_sup)] = 0.0
    return out


# --------------------------------------------------------------------------
# Physicality: raw symplectic margins (NO entropy clamps) + audit gate
# --------------------------------------------------------------------------


def min_symplectic_margin(V, *, tol=1e-10, dps=40, mp_mode_limit=24):
    """min_k nu_k - 1/2, raw (no clamps), with tol-aware mpmath escalation.

    The question the stress-tester asks is "is the margin below -tol?", so
    the escalation is decided relative to `tol`: a float64 margin at or
    above -tol/2 is a pass and returns as is (mpmath cannot flip a genuine
    pass beyond roundoff at that distance); anything below that is a
    candidate violation and is confirmed with `symplectic_margins_mp` RAW —
    spec M3.4: `entropy_from_nu`'s floor clamp is correct for entropies and
    wrong for detecting nu < 1/2 inputs — whenever the dimension permits
    (mp eigensolves are O(n^3) in arbitrary precision). Above
    `mp_mode_limit` modes, a violation deeper than 2*tol is accepted from
    float64 (decisive), while a genuinely borderline value raises rather
    than guessing.
    """
    V = np.asarray(V, dtype=float)
    nu64 = symplectic_eigenvalues(V)
    m64 = float(nu64[-1] - 0.5)
    tol = float(tol)
    if m64 >= -0.5 * tol:
        return m64
    n_modes = V.shape[0] // 2
    if n_modes <= mp_mode_limit:
        return float(np.min(symplectic_margins_mp(V, dps=dps)))
    if m64 < -2.0 * tol:
        return m64  # deep violation: float64 is decisive
    raise ValueError(
        f"borderline margin {m64:.3e} (tol {tol:.1e}) at {n_modes} modes (> "
        f"mp_mode_limit={mp_mode_limit}); reduce to the support block or "
        "raise mp_mode_limit to escalate"
    )


def assert_physical(V, *, tol=1e-10, dps=40, mp_mode_limit=24):
    """Raise :class:`QEIUnphysicalError` unless min(nu) >= 1/2 - tol (raw)."""
    margin = min_symplectic_margin(V, tol=tol, dps=dps, mp_mode_limit=mp_mode_limit)
    if margin < -tol:
        raise QEIUnphysicalError(
            f"state is unphysical: min symplectic margin nu - 1/2 = "
            f"{margin:.6e} < -{tol:.1e} (V + i Omega/2 not PSD)"
        )
    return margin


def qei_audit(V, op: SamplingOp, f: FHandle, *, constant="sharp", tol=1e-10,
              dps=40, mp_mode_limit=24, strict=True):
    """Gated smeared-energy vs bound check for one engineered state.

    Physicality first (raw margins on the support block — the block the
    smeared energy reads; an unphysical input raises QEIUnphysicalError,
    which is how the unphysical-V canary is caught). Then the QEI margin
    E_f(V) - bound(f) >= 0; a violation by a *physical* state is an audit
    event (QEIViolationError under strict=True) — the anomaly-protocol
    path, never an expected outcome.
    """
    V = np.asarray(V, dtype=float)
    n_s = op.support.size
    V_sub = _reduce_cov(V, op.support) if (V.shape[0] == 2 * op.N and op.N != n_s) else V
    nu_margin = assert_physical(V_sub, tol=tol, dps=dps, mp_mode_limit=mp_mode_limit)
    e_f = smeared_energy(V_sub, op)
    bound = qei_bound_2d(f, constant=constant)
    margin = e_f - bound
    passed = margin >= -1e-12 * max(1.0, abs(bound))
    if strict and not passed:
        raise QEIViolationError(
            f"QEI violated by a physical state: E_f = {e_f:.6e} < bound = "
            f"{bound:.6e} (margin {margin:.3e}) — audit event"
        )
    return {"e_f": e_f, "bound": bound, "margin": margin,
            "nu_margin": nu_margin, "passed": bool(passed)}


# --------------------------------------------------------------------------
# Engineered states
# --------------------------------------------------------------------------


def squeezed_pocket(K, modes, r, phi=0.0):
    """Locally squeezed vacuum of H = (1/2)(p.p + x.K.x) — pocket states.

    Applies single-mode squeezers S_k = R(phi_k) diag(e^{-r_k}, e^{r_k})
    R(phi_k)^T (symplectic) to the listed modes of the ground state.
    r and phi broadcast over modes. Pure function; always physical (a
    symplectic image of the vacuum), with locally structured energy —
    the stress-tester's engineered-state family.
    """
    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    modes = [int(k) for k in modes]
    r_arr = np.broadcast_to(np.asarray(r, dtype=float), (len(modes),))
    phi_arr = np.broadcast_to(np.asarray(phi, dtype=float), (len(modes),))
    V0 = ground_state_cov(K)
    S = np.eye(2 * N)
    for k, rk, pk in zip(modes, r_arr, phi_arr):
        c, s = math.cos(pk), math.sin(pk)
        R = np.array([[c, -s], [s, c]])
        D = np.diag([math.exp(-rk), math.exp(rk)])
        blk = R @ D @ R.T
        idx = np.array([k, N + k])
        S[np.ix_(idx, idx)] = blk
    return S @ V0 @ S.T


# --------------------------------------------------------------------------
# Sampling-function optimization (analytic envelope gradients)
# --------------------------------------------------------------------------


class OptimizeSamplingResult(NamedTuple):
    ratio: float
    theta: np.ndarray
    f: FHandle
    e_min: float
    bound: float
    n_iter: int
    converged: bool
    message: str


def optimize_sampling(N, site, family: SamplingFamily, theta0, *, bounds,
                      m=0.0, bc="dirichlet", constant="sharp", pad=8,
                      panel_width=None, nodes_per_panel=10, maxiter=60,
                      eps_reg=1e-10):
    """Ascend qei_ratio over a smooth f-family with analytic gradients.

    The time grid and mode support are FIXED from the widest handle over the
    parameter box (so the envelope theorem applies to the assembled
    operator): dE_min/d theta_j = (1/2) Tr(dO_j (V_star - V_vac_sub)) —
    Hellmann-Feynman at the exhibited extremal state, no finite differences.
    The bound gradient uses the analytic f'' via integration by parts:
    d/d theta_j int f'^2 = -2 int f'' (partial f/partial theta_j) dt
    (boundary terms vanish: f' -> 0 at the support edges for both families).

    Returns the best ratio, parameters, and handle. Maximization via
    L-BFGS-B on -ratio within `bounds` (sequence of (lo, hi) per parameter).
    """
    from scipy.optimize import minimize as _minimize

    theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
    bounds = [(float(lo), float(hi)) for lo, hi in bounds]
    corners = [family.make(np.array([lo for lo, _ in bounds])),
               family.make(np.array([hi for _, hi in bounds]))]
    t_lo = min(h.t_lo for h in corners)
    t_hi = max(h.t_hi for h in corners)
    t_max = max(abs(t_lo), abs(t_hi))
    support = _cone_support(N, site, int(math.ceil(t_max)) + int(pad), bc)

    if constant == "sharp":
        C = SHARP_CONSTANT_2D
    elif constant == "fewster_eveson":
        C = FEWSTER_EVESON_CONSTANT_2D
    else:
        raise ValueError(f"constant must be 'sharp' or 'fewster_eveson', got {constant!r}")

    # fixed quadrature grid for the bound-side integrals
    bts, bws = _gauss_legendre_grid(t_lo, t_hi, max(8, int(math.ceil(t_hi - t_lo))), 20)

    def objective(theta):
        fh = family.make(theta)
        op = sampling_operator(
            N, site, fh, m=m, bc=bc, support=support, panel_width=panel_width,
            nodes_per_panel=nodes_per_panel, t_lo=t_lo, t_hi=t_hi,
            df_dtheta=lambda ts: family.df_dtheta(theta, ts),
        )
        res = qei_minimize(op, eps_reg=eps_reg)
        dfp = fh.df(bts)
        fp2 = float(np.sum(bws * dfp * dfp))
        B = -C * fp2
        d2 = fh.d2f(bts)
        dtheta = family.df_dtheta(theta, bts)  # (n_params, nt)
        dB = -C * (-2.0) * (dtheta * (bws * d2)[None, :]).sum(axis=1)
        dE = np.array([
            0.5 * float(np.einsum("ij,ji->", D, res.V_star - op.V_vac_sub))
            for D in op.dO
        ])
        r = res.e_min / B
        dr = (dE * B - res.e_min * dB) / (B * B)
        return -r, -dr

    out = _minimize(objective, theta0, jac=True, method="L-BFGS-B",
                    bounds=bounds, options={"maxiter": int(maxiter)})
    theta_star = np.atleast_1d(out.x)
    fh = family.make(theta_star)
    op = sampling_operator(N, site, fh, m=m, bc=bc, support=support,
                           panel_width=panel_width, nodes_per_panel=nodes_per_panel,
                           t_lo=t_lo, t_hi=t_hi)
    res = qei_minimize(op, eps_reg=eps_reg)
    dfp = fh.df(bts)
    B = -C * float(np.sum(bws * dfp * dfp))
    return OptimizeSamplingResult(
        ratio=res.e_min / B, theta=theta_star, f=fh, e_min=res.e_min, bound=B,
        n_iter=int(out.nit), converged=bool(out.success), message=str(out.message),
    )


# --------------------------------------------------------------------------
# 4d machinery: Srednicki radial-lattice T_00 assembly + ball smearing
# (papers-candidate entry points; the heavy 4d scan itself stays OUT of the
#  test suite per spec M3.4)
# --------------------------------------------------------------------------


def radial_ball_weights(N, n_ball, profile="uniform"):
    """Radial smearing weights w_j >= 0 for a small ball around the origin.

    'uniform': w_j = 1 for lattice sites j <= n_ball (0-based idx < n_ball),
    'cos2': smooth rolloff w_j = cos^2(pi j_c / (2 n_ball)) over the ball
    (j_c the site center distance in lattice units). Raw weights — callers
    wanting a per-volume density divide by their ball measure.
    """
    N = int(N)
    n_ball = int(n_ball)
    if not 1 <= n_ball <= N - 1:
        raise ValueError(f"need 1 <= n_ball <= N-1={N - 1}, got {n_ball}")
    w = np.zeros(N)
    if profile == "uniform":
        w[:n_ball] = 1.0
    elif profile == "cos2":
        j = np.arange(n_ball, dtype=float)
        w[:n_ball] = np.cos(0.5 * math.pi * j / n_ball) ** 2
    else:
        raise ValueError(f"profile must be 'uniform' or 'cos2', got {profile!r}")
    return w


def _radial_density_terms(N, l, weights):
    """PSD rank-1 terms of the w-weighted energy density of Srednicki's H_l.

    H_l = (1/2) sum_j pi_j^2 + (1/2) sum_j [l(l+1)/j^2] phi_j^2
        + (1/2) sum_{j=1}^{N} (j+1/2)^2 (phi_{j+1}/(j+1) - phi_j/j)^2,
    phi_{N+1} = 0 (Srednicki PRL 71, 666 (1993), Eq. 10; vacuum.core.radial).
    Site terms carry w_j; interior bonds carry (w_j + w_{j+1})/2 (symmetric
    split); the outer phi_{N+1}=0 bond touches only site N and carries w_N,
    so w == 1 reproduces diag(K_l, I) exactly (regression-tested).
    """
    N = int(N)
    l = int(l)
    w = np.asarray(weights, dtype=float)
    if w.shape != (N,):
        raise ValueError(f"weights must have shape ({N},), got {w.shape}")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative for a PSD density")
    ll1 = float(l * (l + 1))
    empty_i = np.array([], dtype=int)
    empty_v = np.array([])
    terms = []
    for j0 in range(N):  # 0-based site index; physical site j = j0 + 1
        if w[j0] == 0.0:
            continue
        terms.append((w[j0], empty_i, empty_v, np.array([j0]), np.array([1.0])))
        if ll1 > 0.0:
            terms.append((w[j0] * ll1 / (j0 + 1.0) ** 2,
                          np.array([j0]), np.array([1.0]), empty_i, empty_v))
    for j0 in range(N - 1):  # interior bond j = j0+1 couples sites j0, j0+1
        cw = 0.5 * (w[j0] + w[j0 + 1])
        if cw == 0.0:
            continue
        jj = j0 + 1.0
        vec = np.array([-(jj + 0.5) / jj, (jj + 0.5) / (jj + 1.0)])
        terms.append((cw, np.array([j0, j0 + 1]), vec, empty_i, empty_v))
    if w[N - 1] > 0.0:  # outer phi_{N+1} = 0 bond, wholly on site N
        jj = float(N)
        terms.append((w[N - 1], np.array([N - 1]),
                      np.array([-(jj + 0.5) / jj]), empty_i, empty_v))
    return terms


def radial_energy_density_form(N, l, weights):
    """Dense (2N, 2N) PSD w-weighted energy density of the l-th partial wave.

    Energy = (1/2) R^T h R; with w == 1 this is exactly diag(srednicki_K(N, l),
    I) — the whole-lattice identity the smoke tests pin down.
    """
    return _terms_to_dense(int(N), _radial_density_terms(N, l, weights))


def radial_sampling_operator(N, l, weights, f: FHandle, *, pad=8, support=None,
                             panel_width=None, nodes_per_panel=10,
                             t_lo=None, t_hi=None, weight_floor=1e-9):
    """Pulled-back sampling operator for the w-weighted radial T_00, sector l.

    Same construction as :func:`sampling_operator` with K = srednicki_K(N, l)
    and h = the ball-smeared density; default support = the weighted sites
    extended by the light cone ceil(t_max) + pad (radial lattice couplings
    are O(1), so lattice speed <= ~1 near the origin; verify with the
    support-independence check as always).
    """
    N = int(N)
    w = np.asarray(weights, dtype=float)
    t_lo = float(f.t_lo if t_lo is None else t_lo)
    t_hi = float(f.t_hi if t_hi is None else t_hi)
    t_max = max(abs(t_lo), abs(t_hi))
    K = srednicki_K(N, l)
    if support is None:
        occupied = np.flatnonzero(w > 0)
        radius = int(math.ceil(t_max)) + int(pad)
        hi = min(N - 1, int(occupied.max()) + radius)
        lo = max(0, int(occupied.min()) - radius)
        support = np.arange(lo, hi + 1)
    support = np.unique(np.asarray(support, dtype=int))
    terms = _radial_density_terms(N, l, w)
    O, dO, V_vac_sub, support, n_nodes, pw = _assemble_pullback(
        K, terms, f.f, t_lo, t_hi, support, panel_width, nodes_per_panel,
        weight_floor=weight_floor)
    meta = {"l": int(l), "pad": int(pad), "panel_width": float(pw),
            "nodes_per_panel": int(nodes_per_panel), "t_lo": t_lo, "t_hi": t_hi,
            "weight_floor": float(weight_floor), "label": f.label, "m": 0.0,
            "bc": "srednicki"}
    return SamplingOp(O=O, support=support, V_vac_sub=V_vac_sub, N=N, site=-1,
                      t_max=t_max, n_nodes=int(n_nodes), dO=dO, meta=meta)


def radial_ball_qei(N, n_ball, f: FHandle, *, l_max=40, tol=1e-8,
                    profile="uniform", pad=8, panel_width=None,
                    nodes_per_panel=10, eps_reg=1e-8, weight_floor=1e-9,
                    method="williamson", rel_tol=1e-12, fhat="auto",
                    mode_floor=1e-14):
    """E_min of the ball-smeared T_00 near the origin: sum over partial waves.

    The state space factorizes over (l, m) sectors, so the global minimum is
    the degeneracy-weighted sum of per-sector Williamson minima:

        E_min = sum_l (2l+1) E_min^{(l)},

    with the l-sum extended until |(2l+1) E_min^{(l)}| < tol (absolute) or
    < rel_tol * |total| twice in a row (the sphere_entropy adaptive pattern)
    or l_max is hit. Returns a dict with the total, the per-l terms,
    per-sector results, and a convergence flag. This is the papers-agent
    entry point for the 4d scan — the scan itself (f-families x ball radii x
    Richardson) stays out of the test suite per spec M3.4.

    method='williamson' (default, legacy) minimizes each sector with
    :func:`qei_minimize` on the support-restricted, time-windowed operator;
    its per-sector value is a float64 cancellation (1/2)sum sigma -
    (1/2)Tr(O V_vac) whose noise floor ~ eps * lambda_max * n grows with l
    while the physical terms fall super-exponentially, so the l-sum cannot
    be run to convergence (papers/qei-2d-sharp-constant, draft/outline.md
    section 6). method='mode_space' uses :func:`radial_sector_minimize` —
    the cancellation-free minimizer in the sector's normal-mode basis (exact
    Fourier sampling when `fhat='auto'` finds f.f2hat, no support
    truncation); per-sector results are then :class:`ModeSpaceResult`.
    """
    if method not in ("williamson", "mode_space"):
        raise ValueError(f"method must be 'williamson' or 'mode_space', got {method!r}")
    w = radial_ball_weights(N, n_ball, profile=profile)
    terms = []
    results = []
    total = 0.0
    below = 0
    converged = False
    l_stop = 0
    for l in range(int(l_max) + 1):
        if method == "williamson":
            op = radial_sampling_operator(N, l, w, f, pad=pad, panel_width=panel_width,
                                          nodes_per_panel=nodes_per_panel,
                                          weight_floor=weight_floor)
            res = qei_minimize(op, eps_reg=eps_reg)
        else:
            res = radial_sector_minimize(N, l, w, f, panel_width=panel_width,
                                         nodes_per_panel=nodes_per_panel,
                                         fhat=fhat, mode_floor=mode_floor)
        term = (2 * l + 1) * res.e_min
        total += term
        terms.append(term)
        results.append(res)
        l_stop = l
        # absolute floor (legacy) OR relative floor: the mode-space route has no
        # noise floor to speak of, so its natural stopping rule is relative
        small = abs(term) < tol or abs(term) < rel_tol * abs(total)
        below = below + 1 if small else 0
        if below >= 2:
            converged = True
            break
    return {"e_min_total": total, "terms": np.asarray(terms), "l_stop": l_stop,
            "converged": converged, "results": results, "n_ball": int(n_ball),
            "profile": profile, "method": method}


# --------------------------------------------------------------------------
# Cancellation-free sector minimizer in mode space (the fix named in
# papers/qei-2d-sharp-constant/draft/outline.md section 6.5)
# --------------------------------------------------------------------------
#
# Why the Williamson route loses digits. Per sector, E_min = (1/2) sum sigma
# - (1/2) Tr(O V_vac) is the difference of two O(l) numbers; the symplectic
# spectrum comes out of an SVD with absolute error ~ eps * lambda_max(O) per
# value, so the difference carries an absolute noise floor ~ eps *
# lambda_max * n that GROWS with l (centrifugal barrier) while the physical
# value falls super-exponentially. The 4d l-sum therefore had to be
# truncated at a "knee" and the truncation was an uncontrolled systematic.
#
# The fix: write the quadratic form in the normal-mode (Bogoliubov) basis of
# the sector Hamiltonian, where V_vac = I/2 exactly and Heisenberg evolution
# is c_j(t) = exp(-i omega_j t) c_j. With c = (x' + i p')/sqrt(2) the sampled
# form is
#
#     (1/2) R^T O R = c^dag A c + (1/2)(c^T conj(B) c + c^dag B conj(c)) + (1/2) Tr A,
#
#     A_jk = A^h_jk Fhat(omega_j - omega_k),   B_jk = B^h_jk Fhat(omega_j + omega_k),
#
# where A^h, B^h are the density's number-conserving and anomalous parts and
# Fhat(nu) = int f(t)^2 e^{i nu t} dt is the Fourier transform of the smearing
# weight: the time integral is done analytically (or by the same Gauss-
# Legendre rule as `sampling_operator`), never by accumulating a 2N x 2N
# matrix. (1/2) Tr A is the vacuum expectation, i.e. (1/2) Tr(O V_vac), and
# the ground energy of the normal-ordered form is
#
#     E_min = (1/2) Tr( conj(B) Z ),     B + A Z + Z A^T + Z conj(B) Z = 0,
#
# with Z the (complex symmetric) squeezing matrix of the extremal state
# |psi> ~ exp((1/2) c^dag Z c^dag) |0>. E_min is a bilinear of the small
# anomalous coupling B and the small squeezing Z — no cancellation: an
# absolute error dB in B changes E_min only at second order, and modes the
# smearing cannot see (B_jk super-exponentially small) contribute exactly
# what they should, ~0, instead of eps * lambda_max. Equivalently, in terms
# of the Bogoliubov quasiparticles b_k = u_k c + v_k c^dag that diagonalize
# the form (frequencies sigma_k = the symplectic eigenvalues),
#
#     E_min = - sum_k sigma_k ||v_k||^2,
#
# a sum of individually non-positive per-mode quantities: the per-mode
# matching of (1/2)sum sigma against (1/2)Tr(O V_vac) that section 6.5 asks
# for. The Riccati equation is solved in the eigenbasis of A (sampling
# weights alpha_j >= 0) from the invariant subspace of the Bogoliubov matrix
# [[A, B], [-conj(B), -A]] and polished by damped Newton (Sylvester steps);
# modes whose sampling weight is below `mode_floor * max(alpha)` — i.e.
# unresolvable in float64 — are left in the vacuum and their second-order
# contribution is reported separately as `e_dropped`.
#
# What remains: the sampling weights alpha_j themselves are eigenvalues of A
# with absolute error ~ eps * max(alpha), so "frozen" modes (sampled only
# briefly, sigma ~ 0, each worth -alpha_j/2) below that level are genuinely
# unresolvable; the per-sector absolute floor of this route is therefore
# ~ eps * max(alpha) per such mode (measured ~1e-13 at the 4d scan sizes)
# instead of eps * lambda_max * n — and, unlike the old floor, it never
# produces a term larger than the sector's own sampling weight.


class ModeSpaceResult(NamedTuple):
    """Result of the cancellation-free (mode-space) sector minimization.

    e_min = e_kept + e_dropped is the minimum of the smeared normal-ordered
    energy over all physical states; e_kept is the exact minimum over the
    modes kept above `mode_floor` (a rigorous upper bound on the infimum),
    e_dropped the second-order estimate of the rest; e_vac = (1/2) Tr A =
    (1/2) Tr(O V_vac) is the vacuum reference the Williamson route
    subtracts; sigma is the symplectic spectrum of the kept block
    (eigenvalues of A + Z conj(B)), descending; alpha the sampling weights
    (eigenvalues of A, all modes, descending); mode_terms the per-quasimode
    contributions -sigma_k ||v_k||^2 (diagnostic; their sum equals e_kept up
    to the conditioning of the quasimode eigenvectors); Z, alpha_kept,
    B_kept, U_A, U, omega describe the extremal state in the kept
    A-eigenbasis (see :func:`mode_space_extremal_state`); residual is the
    Frobenius norm of the Riccati residual at return, z_norm = ||Z||_2 of the
    stored (state) Z; meta['state_reg'] > 0 flags that Z was re-solved with
    shifted weights to make the state normalizable (meta['z_norm_raw'] is
    the unshifted solution's norm; e_min never uses the shift).
    """

    e_min: float
    e_kept: float
    e_dropped: float
    e_vac: float
    sigma: np.ndarray
    alpha: np.ndarray
    omega: np.ndarray
    n_kept: int
    mode_terms: np.ndarray
    Z: np.ndarray
    alpha_kept: np.ndarray
    B_kept: np.ndarray
    U_A: np.ndarray
    U: np.ndarray
    residual: float
    n_newton: int
    z_norm: float
    meta: dict


def _mode_space_AB(K, terms, f, t_lo, t_hi, panel_width, nodes_per_panel, fhat):
    """A, B, omega, U of the sampled form in the normal-mode basis of K."""
    K = np.asarray(K, dtype=float)
    K = 0.5 * (K + K.T)
    N = K.shape[0]
    w2, U = np.linalg.eigh(K)
    if np.min(w2) <= 0:
        raise ValueError(f"K must be positive definite: min eig {np.min(w2):.3e}")
    w = np.sqrt(w2)
    sw = np.sqrt(w)
    A_h = np.zeros((N, N), dtype=complex)
    B_h = np.zeros((N, N), dtype=complex)
    for coeff, xi, xv, pi_, pv in terms:
        ux = np.zeros(N)
        ux[xi] = xv
        up = np.zeros(N)
        up[pi_] = pv
        # x = U w^{-1/2} x', p = U w^{1/2} p'  ->  u.R = alpha.x' + beta.p'
        gamma = (U.T @ ux) / sw - 1j * (U.T @ up) * sw
        gc = gamma.conj()
        A_h += (0.5 * coeff) * np.outer(gc, gamma)
        B_h += (0.5 * coeff) * np.outer(gc, gc)
    if fhat == "auto":
        fhat = "analytic" if getattr(f, "f2hat", None) is not None else "quadrature"
    if fhat == "analytic":
        if getattr(f, "f2hat", None) is None:
            raise ValueError("fhat='analytic' needs an FHandle with f2hat")
        Fm = np.asarray(f.f2hat(w[:, None] - w[None, :]), dtype=complex)
        Fp = np.asarray(f.f2hat(w[:, None] + w[None, :]), dtype=complex)
    elif fhat == "quadrature":
        if panel_width is None:
            panel_width = min(1.6, 2.4 / float(np.max(w)))
        span = float(t_hi) - float(t_lo)
        n_panels = max(1, int(math.ceil(span / panel_width)))
        ts, wq = _gauss_legendre_grid(t_lo, t_hi, n_panels, nodes_per_panel)
        fv = np.asarray(f.f(ts), dtype=float)
        node_w = wq * fv * fv
        P = np.exp(1j * np.outer(w, ts))          # e^{i omega_j t_a}
        PW = P * node_w[None, :]
        Fm = PW @ P.conj().T                      # Fhat(omega_j - omega_k)
        Fp = PW @ P.T                             # Fhat(omega_j + omega_k)
    else:
        raise ValueError(f"fhat must be 'auto', 'analytic' or 'quadrature', got {fhat!r}")
    A = A_h * Fm
    B = B_h * Fp
    A = 0.5 * (A + A.conj().T)
    B = 0.5 * (B + B.T)
    return A, B, w, U


def _riccati_residual(alpha, B, Bc, Z):
    return B + alpha[:, None] * Z + Z * alpha[None, :] + Z @ Bc @ Z


def _riccati_solve(alpha, B, newton_max=20):
    """Squeezing matrix Z of the ground state of c^dag diag(alpha) c +
    (1/2)(c^T conj(B) c + h.c.): invariant-subspace start + damped Newton."""
    n = alpha.size
    Bc = B.conj()
    al = np.diag(alpha).astype(complex)
    if n == 0:
        return np.zeros((0, 0), dtype=complex), 0.0, 0
    M = np.block([[al, B], [-Bc, -al]])
    ev, X = np.linalg.eig(M)
    pos = np.argsort(-ev.real)[:n]
    X1, X2 = X[:n, pos], X[n:, pos]
    try:
        Zb, *_ = np.linalg.lstsq(X1.T, X2.T, rcond=None)   # conj(Z) = X2 X1^{-1}
    except np.linalg.LinAlgError:
        # LAPACK's SVD-based gelsd can fail to converge on the near-degenerate
        # eigenvector matrices of large sectors with slowly-decaying (cusped)
        # smearing transforms; the QR-based gelsy driver does not use an SVD.
        # This branch is reached only where the call above raised, so it never
        # changes a result that was previously computed.
        from scipy.linalg import lstsq as _sp_lstsq
        Zb = _sp_lstsq(X1.T, X2.T, lapack_driver="gelsy")[0]
    Z = Zb.T.conj()
    Z = 0.5 * (Z + Z.T)
    R = _riccati_residual(alpha, B, Bc, Z)
    rn = float(np.linalg.norm(R))
    scale = max(float(np.linalg.norm(B)), 1e-300)
    n_iter = 0
    for _ in range(int(newton_max)):
        if rn <= 1e-16 * scale:
            break
        At = al + Z @ Bc
        try:
            D = solve_sylvester(At, At.T, -R)
        except Exception:  # singular closed loop: accept the current iterate
            break
        D = 0.5 * (D + D.T)
        t = 1.0
        accepted = False
        for _ls in range(20):
            Zt = Z + t * D
            Rt = _riccati_residual(alpha, B, Bc, Zt)
            rt = float(np.linalg.norm(Rt))
            if rt < rn:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            break
        improved = rt < 0.9 * rn
        Z, R, rn = Zt, Rt, rt
        n_iter += 1
        if not improved:
            break
    return Z, rn, n_iter


def _mode_terms(alpha, B, Z):
    """Per-quasimode terms -sigma_k ||v_k||^2 and sigma (descending)."""
    n = alpha.size
    if n == 0:
        return np.zeros(0), np.zeros(0)
    Bc = B.conj()
    At = np.diag(alpha).astype(complex) + Z @ Bc
    sig, W = np.linalg.eig(At.T)                 # columns: left eigenvectors of At
    Um = W.T                                      # rows u_k
    ZZ = Z @ Z.conj()
    nrm = np.einsum("kj,jl,kl->k", Um, np.eye(n) - ZZ, Um.conj()).real
    vv = np.einsum("kj,jl,kl->k", Um, ZZ, Um.conj()).real
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = -sig.real * vv / nrm
    order = np.argsort(-sig.real)
    return sig.real[order], terms[order]


def mode_space_minimize(K, terms, f: FHandle, *, t_lo=None, t_hi=None,
                        panel_width=None, nodes_per_panel=10, fhat="auto",
                        mode_floor=1e-14, newton_max=20, state_reg=1e-10,
                        label=None):
    """Cancellation-free minimum of the smeared normal-ordered energy.

    K is the (N, N) SPD coupling matrix of the free field H = (1/2)(p.p +
    x.K.x); `terms` the sparse PSD rank-1 decomposition of the local energy
    density (see :func:`_chain_density_terms` / :func:`_radial_density_terms`);
    f the sampling function. Solves the SAME variational problem as
    :func:`qei_minimize` — inf over all physical states of
    (1/2) Tr(O_f (V - V_vac)) — on the full lattice (no support truncation)
    but in the normal-mode basis, where the answer is a bilinear of the
    small anomalous coupling and the small squeezing (module note above).

    fhat: 'analytic' samples the exact Fourier transform f.f2hat of f^2 (the
    un-windowed continuum smearing; Gaussian handles carry one);
    'quadrature' uses the composite Gauss-Legendre rule of
    :func:`sampling_operator` on [t_lo, t_hi] (identical time integral, so
    the two routes agree to roundoff on the untruncated problem); 'auto'
    picks analytic when available.

    mode_floor: modes with sampling weight alpha_j < mode_floor * max(alpha)
    (unresolvable in float64 at the default 1e-14 ~ 100 eps) are left in the
    vacuum; their second-order contribution is returned as e_dropped and
    included in e_min. state_reg: weight shift (relative to max(alpha)) used
    only to obtain a normalizable extremal state when the sector has
    numerically-null symplectic eigenvalues (see the note in the body; the
    energy never uses it). Returns a :class:`ModeSpaceResult`.
    """
    t_lo = float(f.t_lo if t_lo is None else t_lo)
    t_hi = float(f.t_hi if t_hi is None else t_hi)
    A, B, w, U = _mode_space_AB(K, terms, f, t_lo, t_hi, panel_width,
                                nodes_per_panel, fhat)
    N = A.shape[0]
    e_vac = 0.5 * float(np.trace(A).real)
    alpha, UA = np.linalg.eigh(A)
    alpha = np.clip(alpha[::-1], 0.0, None)
    UA = UA[:, ::-1]
    a_max = float(alpha[0]) if alpha.size else 0.0
    keep = alpha > mode_floor * a_max if a_max > 0 else np.zeros(N, dtype=bool)
    # c = U_A c~  =>  conj(B~) = U_A^T conj(B) U_A
    Bt_full = (UA.T @ B.conj() @ UA).conj()
    Bt_full = 0.5 * (Bt_full + Bt_full.T)
    ak = alpha[keep]
    Bt = Bt_full[np.ix_(keep, keep)]
    UAk = UA[:, keep]
    # second-order estimate of everything touching a dropped mode
    den = alpha[:, None] + alpha[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        w2 = np.where(den > 0.0, np.abs(Bt_full) ** 2 / den, 0.0)
    kk = np.outer(keep, keep)
    e_dropped = -0.5 * float(np.sum(w2[~kk]))
    Z, rn, n_iter = _riccati_solve(ak, Bt, newton_max=newton_max)
    e_kept = 0.5 * float(np.trace(Bt.conj() @ Z).real) if ak.size else 0.0
    # The energy is insensitive to which of the two Riccati branches a
    # numerically-null (sigma ~ 0) quasimode lands on — both give
    # (sigma - alpha)/2 up to O(sigma) — but the STATE is not: a wrong
    # branch is the non-normalizable (|z| > 1) root. For the exhibited
    # state and the per-mode diagnostics re-solve with the sampling weights
    # shifted by state_reg * max(alpha) (the mode-space analogue of
    # qei_minimize's eps_reg), which lifts sigma off zero; the exact energy
    # of that state on the UNshifted problem is what
    # mode_space_extremal_state reports as `achieved`.
    z_norm = float(np.linalg.norm(Z, 2)) if ak.size else 0.0
    eps_used = 0.0
    Z_state, sig_a = Z, ak
    if ak.size and z_norm >= 1.0 - 1e-9:
        eps = float(state_reg)
        for _ in range(4):
            Zr, _rnr, _nr = _riccati_solve(ak + eps * a_max, Bt, newton_max=newton_max)
            if np.linalg.norm(Zr, 2) < 1.0 - 1e-12:
                Z_state, sig_a, eps_used = Zr, ak + eps * a_max, eps
                break
            eps *= 100.0
    sig, mterms = _mode_terms(sig_a, Bt, Z_state)
    meta = {"fhat": ("analytic" if (fhat == "auto" and getattr(f, "f2hat", None) is not None)
                     else ("quadrature" if fhat == "auto" else fhat)),
            "mode_floor": float(mode_floor), "t_lo": t_lo, "t_hi": t_hi,
            "panel_width": panel_width, "nodes_per_panel": int(nodes_per_panel),
            "label": f.label if label is None else label, "N": int(N),
            "state_reg": eps_used, "z_norm_raw": z_norm}
    return ModeSpaceResult(e_min=e_kept + e_dropped, e_kept=e_kept,
                           e_dropped=e_dropped, e_vac=e_vac, sigma=sig,
                           alpha=alpha, omega=w, n_kept=int(ak.size),
                           mode_terms=mterms, Z=Z_state, alpha_kept=ak, B_kept=Bt,
                           U_A=UAk, U=U, residual=float(rn), n_newton=int(n_iter),
                           z_norm=float(np.linalg.norm(Z_state, 2)) if ak.size else 0.0,
                           meta=meta)


def radial_sector_minimize(N, l, weights, f: FHandle, **kw):
    """Cancellation-free E_min^(l) of the ball-smeared radial T_00, sector l.

    Same problem as ``qei_minimize(radial_sampling_operator(N, l, w, f,
    support=all, weight_floor=0))``, solved by :func:`mode_space_minimize`
    in the normal-mode basis of srednicki_K(N, l). Keyword arguments are
    those of :func:`mode_space_minimize`.
    """
    N = int(N)
    w = np.asarray(weights, dtype=float)
    res = mode_space_minimize(srednicki_K(N, l), _radial_density_terms(N, l, w), f, **kw)
    res.meta.update({"l": int(l), "bc": "srednicki", "m": 0.0, "n_ball": int(np.sum(w > 0))})
    return res


def chain_worldline_minimize(N, site, f: FHandle, *, m=0.0, bc="dirichlet", **kw):
    """Cancellation-free E_min(f) for a chain worldline (2d): the mode-space
    counterpart of ``qei_minimize(sampling_operator(N, site, f, ...))`` on the
    untruncated lattice."""
    N = int(N)
    res = mode_space_minimize(harmonic_chain_K(N, m, bc),
                              _chain_density_terms(N, site, m, bc), f, **kw)
    res.meta.update({"site": int(site), "bc": bc, "m": float(m)})
    return res


def _takagi(Z):
    """Takagi factorization Z = W diag(s) W^T of a complex symmetric matrix.

    Via the real symmetric embedding G = [[Re Z, Im Z], [Im Z, -Re Z]], whose
    eigenpairs (s, [a; b]) with s >= 0 give Takagi vectors w = a + i b
    (Z conj(w) = s w); the -s partners are [-b; a], so the +s vectors are
    automatically orthonormal in C^n. Exact for degenerate s.
    """
    n = Z.shape[0]
    G = np.block([[Z.real, Z.imag], [Z.imag, -Z.real]])
    ev, Q = np.linalg.eigh(0.5 * (G + G.T))
    idx = np.argsort(-ev)[:n]
    s = np.clip(ev[idx], 0.0, None)
    W = Q[:n, idx] + 1j * Q[n:, idx]
    return s, W


def _variational_energy(alpha, B, Z):
    """Exact smeared energy of the (normalizable) state exp((1/2) c^dag Z c^dag)|0>:
    (1/2) Tr(conj(B) Z) + (1/2) Tr(R(Z) conj(M)), M = (I - Z conj(Z))^{-1} Z."""
    n = alpha.size
    Bc = B.conj()
    R = _riccati_residual(alpha, B, Bc, Z)
    M = np.linalg.solve(np.eye(n) - Z @ Z.conj(), Z)
    return 0.5 * float(np.trace(Bc @ Z).real) + 0.5 * float(np.trace(R @ M.conj()).real)


def mode_space_extremal_state(res: ModeSpaceResult, *, clip=1.0 - 1e-8):
    """Exhibit the extremal state of a :class:`ModeSpaceResult` on the lattice.

    Returns ``(V, info)``: V the (2N, 2N) pure Gaussian covariance in the
    site basis (block ordering) of the vacuum of the Bogoliubov quasiparticles
    b = U_b c~ + V_b conj(c~), U_b = (I - Z conj Z)^{-1/2}, V_b = -U_b Z, on
    the kept modes (dropped modes stay in the vacuum), and info a dict with
    the exact smeared energy `achieved` of that state, `gap = achieved -
    e_min` (>= 0 up to roundoff), the squeezing spectrum `squeeze_s`
    (Takagi singular values of Z, tanh of the per-mode squeezing
    parameters), and `clipped`. Modes squeezed to the normalizability
    boundary (Takagi value >= clip: sigma ~ 0 directions, the same
    "infinitely squeezed null directions" the Williamson route regularizes
    with eps_reg) are clipped to `clip`; the price is in `gap`.
    """
    Z = res.Z
    n = Z.shape[0]
    N = res.U.shape[0]
    s, W = _takagi(Z)
    clipped = bool(np.any(s >= clip))
    if clipped:
        Zc = (W * np.minimum(s, clip)) @ W.T
        Zc = 0.5 * (Zc + Zc.T)
    else:
        Zc = Z
    achieved = _variational_energy(res.alpha_kept, res.B_kept, Zc) + res.e_dropped
    ZZ = Zc @ Zc.conj()
    ev, Q = np.linalg.eigh(0.5 * (ZZ + ZZ.conj().T))
    ev = np.clip(ev, 0.0, 1.0 - 1e-15)
    Ub = (Q * (1.0 / np.sqrt(1.0 - ev))) @ Q.conj().T
    Vb = -Ub @ Zc
    I = np.eye(n)
    Wc = np.block([[I, 1j * I], [I, -1j * I]]) / math.sqrt(2.0)      # (c; c^dag) = Wc (q; p)
    Wi = np.block([[I, I], [-1j * I, 1j * I]]) / math.sqrt(2.0)      # inverse
    Bog = np.block([[Ub, Vb], [Vb.conj(), Ub.conj()]])
    Sb = (Wi @ Bog @ Wc).real                                        # real symplectic
    Vt = 0.5 * np.linalg.inv(Sb.T @ Sb)                              # b-vacuum in kept quadratures
    IN = np.eye(N)
    WNi = np.block([[IN, IN], [-1j * IN, 1j * IN]]) / math.sqrt(2.0)
    Mk = np.block([[res.U_A, np.zeros((N, n))], [np.zeros((N, n)), res.U_A.conj()]])
    SA = (WNi @ Mk @ Wc).real                                        # kept modes -> normal modes
    Vp = SA @ Vt @ SA.T + 0.5 * (np.eye(2 * N) - SA @ SA.T)          # rest: vacuum
    U, w = res.U, res.omega
    SH = np.block([[U / np.sqrt(w)[None, :], np.zeros((N, N))],
                   [np.zeros((N, N)), U * np.sqrt(w)[None, :]]])      # normal -> site
    V = SH @ Vp @ SH.T
    V = 0.5 * (V + V.T)
    info = {"achieved": float(achieved), "gap": float(achieved - res.e_min),
            "squeeze_s": s, "clipped": clipped, "z_norm": res.z_norm,
            "n_kept": int(n)}
    return V, info


def qei_minimize_mp(op: SamplingOp, K=None, *, dps=30):
    """Precision-escalated cross-check of :func:`qei_minimize` through
    vacuum.core.precision: the symplectic spectrum of the float64 operator
    O_f by the mpmath i*Omega*V eigenproblem, and Tr(O_f V_vac) against the
    mp-native ground-state covariance, both summed in `dps` digits, so the
    cancellation (1/2) sum sigma - (1/2) Tr(O V_vac) is exact for the
    operator as assembled. What it cannot remove is the float64 rounding of
    O_f itself; its effect on the minimum is first-order,
    (1/2) Tr(dO (V_star - V_vac)), i.e. ~ eps * ||O|| * ||V_star - V_vac||,
    which is far below the Williamson route's floor but not below the
    mode-space route's — use it where the sector energy exceeds that.
    Cost: O((2 n_s)^3) mp arithmetic; keep n_s <= ~100.
    """
    from vacuum.core.precision import _abs_eig_pairs_mp, ground_state_cov_mp
    import mpmath as mp

    if K is None:
        if op.meta.get("bc") == "srednicki":
            K = srednicki_K(op.N, op.meta["l"])
        else:
            K = harmonic_chain_K(op.N, op.meta["m"], op.meta["bc"])
    O = np.asarray(op.O, dtype=float)
    n2 = O.shape[0]
    nus = _abs_eig_pairs_mp(O, dps)
    with mp.workdps(dps):
        half_sum = mp.fsum(nus) / 2
        V_full = ground_state_cov_mp(K, dps)
        N = op.N
        q = np.concatenate([op.support, op.support + N])
        tr = mp.mpf(0)
        for a, i in enumerate(q):
            for b, j in enumerate(q):
                o = O[a, b]
                if o != 0.0:
                    tr += mp.mpf(float(o)) * V_full[int(j), int(i)]
        tr_half = tr / 2
        e_min = half_sum - tr_half
        return {"e_min": float(e_min), "half_sum_sigma": float(half_sum),
                "trace_vac": float(tr_half), "dps": int(dps), "dim": int(n2)}


def shell_volume(n_ball, a=1.0):
    """Continuum volume carried by radial sites 1..n_ball of the Srednicki lattice.

    Srednicki's Eq. 10 writes H = sum_lm H_lm with H_lm = (1/2a) sum_j [...],
    the sum over j being the rectangle rule for int dr of the radial density
    of phi_lm(r) = r x (angular component) — the r^2 of d^3x is absorbed in
    that normalization, so site j stands for the shell (j-1/2)a < r < (j+1/2)a
    (its bond terms sit at (j +- 1/2)a and are split half/half). Summing over
    (l, m) turns the site-j term into the energy in that shell, of volume
    V_j = (4 pi/3)[(j+1/2)^3 - (j-1/2)^3] a^3 = 4 pi a^3 (j^2 + 1/12), so the
    ball operator of sites 1..n_ball is the energy in a/2 < r < (n_ball+1/2) a:

        V_shell = sum_j V_j = (4 pi/3) [ (n_ball + 1/2)^3 - (1/2)^3 ] a^3,

    exact to the O(a^2) midpoint-rule accuracy of the discretization. The
    alternative :func:`naive_volume` (4 pi/3)(n_ball a)^3 differs by the
    exactly known factor V_naive/V_shell = 1 - 3/(2 n_ball) + O(1/n_ball^2),
    an O(a/R) error that a linear Richardson term in a/tau at fixed R/tau
    only partly absorbs (its 3/(4 n^2) tail is 19% at n_ball = 2).
    """
    n = float(n_ball)
    return (4.0 * math.pi / 3.0) * ((n + 0.5) ** 3 - 0.125) * float(a) ** 3


def naive_volume(n_ball, a=1.0):
    """(4 pi/3)(n_ball a)^3 — the naive ball measure (see :func:`shell_volume`)."""
    return (4.0 * math.pi / 3.0) * float(n_ball) ** 3 * float(a) ** 3


# --------------------------------------------------------------------------
# The same variational problem in the continuum, in momentum space (no lattice)
# --------------------------------------------------------------------------
#
# For the free massless field the smeared normal-ordered energy is a quadratic
# form in the mode operators, so its infimum over ALL states is a Bogoliubov
# ground energy that can be written down directly in the continuum. With
# phi = int d^3k [u_k a_k + h.c.], u_k = e^{i(k.x - w t)} / sqrt(2 w (2 pi)^3),
# and :T_00: = (1/2)(pi^2 + (grad phi)^2),
#
#   int_B d^3x int dt f(t)^2 :T_00:
#     = sum_{k,k'} sqrt(w w') (1 + k^.k'^) / (2 (2 pi)^3)
#       x [ Fhat(w - w') chi_B(k' - k) a_k^dag a_k'
#           - (1/2) Fhat(w + w') chi_B(k + k') (a_k a_k' + h.c.) ],
#
# Fhat(nu) = int f^2 e^{i nu t} dt the transform of the smearing weight and
# chi_B(q) = int_B e^{i q.x} d^3x the ball form factor (-> V_B on a worldline).
# Expanding a_k = sum_lm a_lm(w) Y_lm(k^)/w and the form factor in Legendre
# polynomials of k^.k'^ (spherical-Bessel addition theorem: chi_B(|k - k'|)
# = 4 pi sum_L (2L+1) J_L(w, w') P_L(k^.k'^), J_L = int_0^R r^2 j_L(w r)
# j_L(w' r) dr, and (1 + x) P_L = P_L + [(L+1) P_{L+1} + L P_{L-1}]/(2L+1)),
# the (l, m) sectors decouple into 2l+1 identical copies of the kernels
#
#   A_l(w, w') = (w w')^{3/2} Fhat(w - w') [(2l+1) J_l + l J_{l-1} + (l+1) J_{l+1}] / (pi (2l+1)),
#   B_l(w, w') = (w w')^{3/2} Fhat(w + w') [(2l+1) J_l - l J_{l-1} - (l+1) J_{l+1}] / (pi (2l+1)),
#
# and the infimum is E(R) = sum_l (2l+1) E_l, E_l the ground energy of
# a^dag A_l a + (1/2)(a^T B_l a + h.c.) (the sign of B is a phase). The two
# J-combinations differ only in the signs of the NEIGHBOURS l -+ 1: the
# anomalous term's form factor is chi_B(k + k'), which is chi_B(k - k') with
# x = k^.k'^ -> -x, and P_L(-x) = (-1)^L P_L(x), so the l -+ 1 terms of
# (1 + x) P_l -- which have the opposite parity to l -- flip sign while the
# J_l term does not. (Both parts of :T_00: = (1/2)(pi^2 + (grad phi)^2) carry
# the SAME angular factor (1 + k^.k'^) sqrt(w w'), the anomalous one with a
# minus sign, which is why A and B share `base` in the code.) As R -> 0
# only J_0 -> R^3/3 survives: per copy A_0 = V_B (w w')^{3/2} Fhat(w - w')
# /(4 pi^2) and A_1 = A_0/3, so the worldline infimum E_wl = V_B E_Q/(2 pi^2),
# E_Q the ground energy of the kernel (w w')^{3/2} Fhat, is shared EQUALLY by
# l = 0 and the three l = 1 copies — the (1 + k^.k'^) structure of T_00 has no
# l >= 2 content, which is why the lattice l-sum converges super-exponentially.
# In 2d the same steps give the chiral kernel sqrt(w w') Fhat with
# E_2d = E_Q2/pi, so Flanagan's sharp bound is the control: E_Q2 -> -(1/6)
# int f'^2 for EVERY admissible f. The w-continuum is discretized on a
# Gauss-Legendre grid (w = u^2 absorbs the w^{3/2} endpoint behaviour) and
# each kernel becomes a finite Bogoliubov problem solved by the same
# cancellation-free Riccati route as the lattice sectors.
#
# The R/tau expansion follows from chi_B(q)/V_B = 1 - q^2 R^2/10 + O(R^4), an
# even entire function of R: the l = 0, 1 kernels shift at O(R^2) (their
# first-order energy shift is Hellmann-Feynman on the worldline extremal
# state), and the l = 2 sector, whose kernel is O(R^2) as a WHOLE, contributes
# its own ground energy at O(R^2) exactly (inf(lambda Q) = lambda inf Q);
# sector l opens at O(R^{2l-2}). Hence, with rho = R/tau,
#
#   C_eff(rho)/C_FE = C_4d/C_FE + d2 rho^2 + d4 rho^4 + ...   (even powers, no logs),
#   d2 = d2(HF; l = 0, 1) + d2(l = 2)                          (`rho2_coefficient`).


class MomentumResult(NamedTuple):
    """Continuum momentum-space infimum of the smeared normal-ordered energy.

    e_min is the infimum (worldline: per unit volume in 4d, i.e. E_wl/V_B, or
    the line energy in 2d); bound the reference bound -C int (f^{(d/2)})^2
    (Flanagan sharp in 2d, Fewster-Eveson Eq. (40) in 4d); ratio = e_min /
    bound (-> 1 in 2d; the sharp-over-FE constant in 4d). sectors lists
    (l, copies, e_l) per copy; n_kept / e_dropped / residual are the Riccati
    diagnostics of :func:`mode_space_minimize` (worst sector).
    """

    e_min: float
    bound: float
    ratio: float
    d: int
    n: int
    omega_max: float
    n_kept: int
    e_dropped: float
    residual: float
    sectors: tuple
    meta: dict


def momentum_grid(omega_max, n, power=2):
    """Gauss-Legendre nodes/weights on 0 < w < omega_max after w = u^power.

    power = 2 (default) makes the (w w')^{3/2} and sqrt(w w') kernels analytic
    in the integration variable; the returned weights include the Jacobian.
    """
    n = int(n)
    if n < 2 or float(omega_max) <= 0:
        raise ValueError(f"need n >= 2 and omega_max > 0, got n={n}, omega_max={omega_max}")
    x, w = leggauss(n)
    umax = float(omega_max) ** (1.0 / power)
    u = 0.5 * umax * (x + 1.0)
    wu = 0.5 * umax * w
    return u**power, power * u ** (power - 1) * wu


def _rms_width(f: FHandle, nodes=2000):
    """sqrt(int t^2 f^2 / int f^2) over the handle's support (grid defaults)."""
    ts, ws = _gauss_legendre_grid(f.t_lo, f.t_hi, max(8, int(math.ceil(nodes / 24))), 24)
    w2 = ws * np.asarray(f.f(ts), dtype=float) ** 2
    m0 = float(np.sum(w2))
    m1 = float(np.sum(w2 * ts)) / m0
    return math.sqrt(max(float(np.sum(w2 * (ts - m1) ** 2)) / m0, 1e-300))


def fhat_of(f: FHandle, *, n_panels=None, nodes_per_panel=24):
    """The smearing-weight transform Fhat(nu) = int f(t)^2 e^{i nu t} dt.

    Returns a vectorized callable: the handle's analytic ``f2hat`` when it
    carries one, otherwise a composite Gauss-Legendre rule over [t_lo, t_hi]
    (the same rule family as :func:`sampling_operator`; the panel count
    defaults to ~ the number of oscillations resolvable for |nu| up to the
    grid's omega_max, which callers should cover by passing `n_panels`).
    """
    if getattr(f, "f2hat", None) is not None:
        return f.f2hat
    span = float(f.t_hi - f.t_lo)
    if n_panels is None:
        n_panels = max(8, int(math.ceil(span)))
    ts, ws = _gauss_legendre_grid(f.t_lo, f.t_hi, int(n_panels), int(nodes_per_panel))
    c = ws * np.asarray(f.f(ts), dtype=float) ** 2

    def f2hat(nu):
        nu = np.asarray(nu, dtype=float)
        flat = nu.ravel()
        # (n_nu, n_t) phase matrix; chunked to bound memory
        out = np.empty(flat.size, dtype=complex)
        step = max(1, int(2e6 // max(ts.size, 1)))
        for a in range(0, flat.size, step):
            out[a:a + step] = np.exp(1j * np.outer(flat[a:a + step], ts)) @ c
        return out.reshape(nu.shape)

    return f2hat


def fsecond_sq_integral(f: FHandle, *, use_analytic=True, nodes_per_panel=24, n_panels=None):
    """int f''(t)^2 dt (analytic ``fpp2`` when available, else quadrature)."""
    if use_analytic and getattr(f, "fpp2", None) is not None:
        return float(f.fpp2)
    span = f.t_hi - f.t_lo
    if n_panels is None:
        n_panels = max(8, int(math.ceil(span)))
    ts, ws = _gauss_legendre_grid(f.t_lo, f.t_hi, n_panels, nodes_per_panel)
    d = np.asarray(f.d2f(ts), dtype=float)
    return float(np.sum(ws * d * d))


def _bogoliubov_ground_energy(A, B, *, mode_floor=1e-14, newton_max=20):
    """Ground energy of a^dag A a + (1/2)(a^T conj(B) a + h.c.), A Hermitian
    PSD, B complex symmetric (kernels on a quadrature grid), by the
    cancellation-free Riccati route of :func:`mode_space_minimize`.

    Returns (e, info): e = e_kept + e_dropped; info carries Z (squeezing in
    the kept A-eigenbasis), alpha_kept, U_A (kept eigenvectors), the dropped
    second-order estimate, the Riccati residual and Newton count.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    A = 0.5 * (A + A.conj().T)
    B = 0.5 * (B + B.T)
    alpha, UA = np.linalg.eigh(A)
    alpha = np.clip(alpha[::-1], 0.0, None)
    UA = UA[:, ::-1]
    a_max = float(alpha[0]) if alpha.size else 0.0
    keep = alpha > mode_floor * a_max if a_max > 0 else np.zeros(alpha.size, dtype=bool)
    Bt_full = (UA.T @ B.conj() @ UA).conj()
    Bt_full = 0.5 * (Bt_full + Bt_full.T)
    ak = alpha[keep]
    Bt = Bt_full[np.ix_(keep, keep)]
    den = alpha[:, None] + alpha[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        w2 = np.where(den > 0.0, np.abs(Bt_full) ** 2 / den, 0.0)
    kk = np.outer(keep, keep)
    e_dropped = -0.5 * float(np.sum(w2[~kk]))
    Z, rn, n_iter = _riccati_solve(ak, Bt, newton_max=newton_max)
    e_kept = 0.5 * float(np.trace(Bt.conj() @ Z).real) if ak.size else 0.0
    info = {"Z": Z, "alpha_kept": ak, "U_A": UA[:, keep], "B_kept": Bt,
            "n_kept": int(ak.size), "e_dropped": e_dropped, "residual": float(rn),
            "n_newton": int(n_iter)}
    return e_kept + e_dropped, info


def _worldline_kernels(om, wom, fhat, power_of_omega):
    """A, B for the kernel (w w')^{power/2} Fhat(w -+ w') on the grid."""
    s = np.sqrt(wom) * om ** (0.5 * power_of_omega)
    base = np.outer(s, s)
    W, Wp = om[:, None], om[None, :]
    A = base * np.asarray(fhat(W - Wp))
    B = base * np.asarray(fhat(W + Wp))
    return A, B


def _grid_defaults(f, n, omega_max, omega_scaled):
    if omega_max is None:
        omega_max = float(omega_scaled) / _rms_width(f)
    if n is None:
        n = 400
    return int(n), float(omega_max)


def worldline_momentum_minimize(f: FHandle, d=4, *, n=None, omega_max=None,
                                omega_scaled=12.0, power=2, mode_floor=1e-14,
                                fhat_panels=None, operator="T00"):
    """Continuum infimum of int f^2 <:T_00:> dt along an inertial worldline.

    `operator` (4d only): 'T00' (default) is (1/2)(phidot^2 + (grad phi)^2);
    'phidot2' and 'gradphi2' are the two Wick squares on their own. Their
    angular factors are 1 and k^.k'^ respectively (T_00's is (1 + k^.k'^)/2),
    so phidot^2 lives in l = 0 alone with kernel 2 A_0 and (grad phi)^2 in the
    three l = 1 copies with kernel 2 A_1 -- and each has the SAME infimum
    E_Q/(2 pi^2) as T_00 itself: the T_00 extremal state (l = 0 (+) l = 1
    squeezing) minimizes both pieces at once. This is the exact form of the
    relation x0(rho_S) = x0(phidot^2) that Fewster, Ford & Roman, PRD 85,
    125038 (2012), Eq. (57), reach by treating the components as independent
    distributions; `bound` for these operators is the T_00 Fewster-Eveson
    bound, which their Sec. IV applies to phidot^2 and rho_S alike.

    d = 2: the two chiral kernels sqrt(w w') Fhat each carry 1/(2 pi), so
    e_min = E_Q2/pi and ratio = e_min / (-(1/(6 pi)) int f'^2) -> 1 for every
    admissible f (Flanagan, PRD 56, 4922 (1997), Eq. (8)) — the control.
    d = 4: e_min = E_wl/V_B = E_Q4/(2 pi^2) (per unit volume) from the l = 0
    copy and the three l = 1 copies of the kernel (w w')^{3/2} Fhat, and
    ratio = e_min / (-(1/(16 pi^2)) int f''^2) is the sharp constant for this
    f in units of the Fewster-Eveson constant (their Eq. (40)).

    Grid: n Gauss-Legendre nodes in u = w^{1/power} up to omega_max (default
    omega_scaled / rms width of f^2 — 17/tau for a Gaussian, where the
    Gaussian result is converged to 1e-7; weights with algebraic Fourier
    tails need a larger omega_max and converge like 1/omega_max, see the
    papers notebook). `fhat_panels` sets the quadrature of Fhat for handles
    without an analytic ``f2hat`` (must resolve e^{i omega_max t} over the
    support).
    """
    if d not in (2, 4):
        raise ValueError(f"d must be 2 or 4, got {d}")
    n, omega_max = _grid_defaults(f, n, omega_max, omega_scaled)
    om, wom = momentum_grid(omega_max, n, power=power)
    fhat = fhat_of(f, n_panels=fhat_panels)
    pw = 1 if d == 2 else 3
    A, B = _worldline_kernels(om, wom, fhat, pw)
    eQ, info = _bogoliubov_ground_energy(A, B, mode_floor=mode_floor)
    if operator not in ("T00", "phidot2", "gradphi2"):
        raise ValueError(f"operator must be 'T00', 'phidot2' or 'gradphi2', got {operator!r}")
    if d == 2:
        if operator != "T00":
            raise ValueError("operator splits are implemented for d = 4 only")
        e_min = eQ / math.pi
        bound = -SHARP_CONSTANT_2D * fprime_sq_integral(f)
        sectors = (("right", 1, eQ / (2.0 * math.pi)), ("left", 1, eQ / (2.0 * math.pi)))
    else:
        e0 = eQ / (4.0 * math.pi**2)
        bound = -FEWSTER_EVESON_CONSTANT_4D * fsecond_sq_integral(f)
        if operator == "T00":
            e_min = e0 + 3.0 * (e0 / 3.0)
            sectors = ((0, 1, e0), (1, 3, e0 / 3.0))
        elif operator == "phidot2":          # angular factor 1: l = 0 only, kernel 2 A_0
            e_min = 2.0 * e0
            sectors = ((0, 1, 2.0 * e0),)
        else:                                # angular factor k^.k'^: l = 1 only, kernel 2 A_1
            e_min = 3.0 * (2.0 * e0 / 3.0)
            sectors = ((1, 3, 2.0 * e0 / 3.0),)
    meta = {"label": f.label, "power": int(power), "mode_floor": float(mode_floor),
            "fhat": "analytic" if getattr(f, "f2hat", None) is not None else "quadrature",
            "e_Q": float(eQ), "operator": operator}
    return MomentumResult(e_min=float(e_min), bound=float(bound), ratio=float(e_min / bound),
                          d=int(d), n=int(n), omega_max=float(omega_max),
                          n_kept=info["n_kept"], e_dropped=float(info["e_dropped"]),
                          residual=info["residual"], sectors=sectors, meta=meta)


def momentum_qei_ratio(f: FHandle, d=4, **kw):
    """ratio of :func:`worldline_momentum_minimize` (2d: -> 1; 4d: C_sharp(f)/C_FE)."""
    return worldline_momentum_minimize(f, d=d, **kw).ratio


def _bessel_overlaps(om, R, l_max, n_r=64):
    """J_L(w_i, w_j) = int_0^R r^2 j_L(w_i r) j_L(w_j r) dr for L = 0 .. l_max+1."""
    from scipy.special import spherical_jn

    x, w = leggauss(int(n_r))
    r = 0.5 * R * (x + 1.0)
    wr = 0.5 * R * w
    out = []
    for L in range(int(l_max) + 2):
        P = spherical_jn(L, np.outer(om, r)) * (r * np.sqrt(wr))[None, :]
        out.append(P @ P.T)
    return out


def ball_momentum_minimize(f: FHandle, R, *, n=None, omega_max=None, omega_scaled=12.0,
                           power=2, l_max=16, rel_tol=1e-12, n_r=64, mode_floor=1e-14,
                           fhat_panels=None):
    """Continuum infimum of the ball-and-time smeared energy at radius R.

    E(R) = sum_l (2l+1) E_l with the sector kernels of the module note (the
    spherical-Bessel overlaps J_L on a Gauss-Legendre rule in r), the l-sum
    run until |(2l+1) E_l| < rel_tol |E| twice. Returns a dict with
    e_min_total, the per-l terms, V_ball, ratio_FE = -E/(V_B C_FE int f''^2)
    (the lattice scan's C_eff/C_FE at rho = R/tau), and the grid used. The
    R -> 0 limit reproduces :func:`worldline_momentum_minimize` (d = 4).
    """
    R = float(R)
    if R <= 0:
        raise ValueError(f"R must be > 0, got {R}")
    n, omega_max = _grid_defaults(f, n, omega_max, omega_scaled)
    om, wom = momentum_grid(omega_max, n, power=power)
    fhat = fhat_of(f, n_panels=fhat_panels)
    J = _bessel_overlaps(om, R, l_max, n_r=n_r)
    s = np.sqrt(wom) * om**1.5
    base = np.outer(s, s)
    W, Wp = om[:, None], om[None, :]
    Fm = np.asarray(fhat(W - Wp))
    Fp = np.asarray(fhat(W + Wp))
    total = 0.0
    terms, infos = [], []
    below = 0
    converged = False
    for l in range(int(l_max) + 1):
        Jl = (2 * l + 1) * J[l]
        Jm = l * J[l - 1] if l >= 1 else 0.0
        Jp = (l + 1) * J[l + 1]
        A = base * Fm * (Jl + Jm + Jp) / (math.pi * (2 * l + 1))
        B = base * Fp * (Jl - Jm - Jp) / (math.pi * (2 * l + 1))
        e, info = _bogoliubov_ground_energy(A, B, mode_floor=mode_floor)
        term = (2 * l + 1) * e
        total += term
        terms.append(term)
        infos.append({k: info[k] for k in ("n_kept", "e_dropped", "residual", "n_newton")})
        if abs(term) < rel_tol * abs(total):
            below += 1
            if below >= 2:
                converged = True
                break
        else:
            below = 0
    V_B = 4.0 * math.pi * R**3 / 3.0
    fpp2 = fsecond_sq_integral(f)
    return {"e_min_total": float(total), "terms": np.asarray(terms), "l_stop": len(terms) - 1,
            "converged": converged, "R": R, "V_ball": V_B, "fpp2": fpp2,
            "e_per_volume": float(total / V_B),
            "ratio_FE": float(-total / (V_B * FEWSTER_EVESON_CONSTANT_4D * fpp2)),
            "n": int(n), "omega_max": float(omega_max), "sector_info": infos,
            "label": f.label}


def rho2_coefficient(f: FHandle, *, n=None, omega_max=None, omega_scaled=12.0, power=2,
                     mode_floor=1e-14, fhat_panels=None, fd_check=True, fd_step=1e-4):
    """The derived O(R^2) coefficient of the ball-smeared infimum.

    With chi_B(q)/V_B = 1 - q^2 R^2/10 + O(R^4) the per-copy kernels shift by
    (Legendre coefficients of (1 + x)|k -+ k'|^2, x = k^.k'^):
        l = 0: dA = -(R^2/10)(w^2 + w'^2 - (2/3) w w') A_0,  dB = -(R^2/10)(w^2 + w'^2 + (2/3) w w') B_0,
        l = 1: dA = -(R^2/10)(w - w')^2 A_1,                 dB = -(R^2/10)(w + w')^2 B_1,
    whose first-order energy shift is <psi*| dO |psi*> on the worldline
    extremal state (Hellmann-Feynman: <a^dag a> = conj(Z) M, <a a> = M,
    M = (1 - Z conj Z)^{-1} Z in the kept A-eigenbasis), and the l = 2 sector
    opens with the O(R^2) kernel (2 R^2/15) w w' (w w')^{3/2} Fhat V_B/(20 pi^2)
    per copy (five copies), whose exact ground energy is R^2 V_B E_Q6/(150 pi^2)
    with E_Q6 the ground energy of the kernel (w w')^{5/2} Fhat. Returns a dict
    with the two pieces and their sum as
        k2 = d[C_eff/C_FE]/d(R^2)  (per unit R^2; multiply by tau^2 for d2 in rho = R/tau),
    plus the worldline ratio and, when `fd_check`, the finite-difference
    check of the Hellmann-Feynman shifts on the same perturbed kernels.
    """
    n, omega_max = _grid_defaults(f, n, omega_max, omega_scaled)
    om, wom = momentum_grid(omega_max, n, power=power)
    fhat = fhat_of(f, n_panels=fhat_panels)
    s = np.sqrt(wom) * om**1.5
    base = np.outer(s, s)
    W, Wp = om[:, None], om[None, :]
    Fm = np.asarray(fhat(W - Wp))
    Fp = np.asarray(fhat(W + Wp))
    pieces = {}
    e_wl = 0.0
    de_hf = 0.0
    for l, c in ((0, 1.0), (1, 1.0 / 3.0)):
        A = c * base * Fm / (4.0 * math.pi**2)
        B = c * base * Fp / (4.0 * math.pi**2)
        e, info = _bogoliubov_ground_energy(A, B, mode_floor=mode_floor)
        Z, UA = info["Z"], info["U_A"]
        nk = Z.shape[0]
        M = np.linalg.solve(np.eye(nk) - Z @ Z.conj(), Z)      # <a a>
        Nn = Z.conj() @ M                                       # <a^dag a>
        if l == 0:
            gA = -(1.0 / 10.0) * (W**2 + Wp**2 - (2.0 / 3.0) * W * Wp)
            gB = -(1.0 / 10.0) * (W**2 + Wp**2 + (2.0 / 3.0) * W * Wp)
        else:
            gA = -(1.0 / 10.0) * (W - Wp) ** 2
            gB = -(1.0 / 10.0) * (W + Wp) ** 2
        dA, dB = gA * A, gB * B
        dAt = UA.conj().T @ dA @ UA
        dBt = (UA.T @ dB.conj() @ UA).conj()
        de = float(np.sum(dAt * Nn).real) + float(np.sum(dBt.conj() * M).real)
        rec = {"e": float(e), "de_dR2": de, "copies": 2 * l + 1, "n_kept": info["n_kept"]}
        if fd_check:
            h = float(fd_step)
            ep, _ = _bogoliubov_ground_energy(A + h * dA, B + h * dB, mode_floor=mode_floor)
            em, _ = _bogoliubov_ground_energy(A - h * dA, B - h * dB, mode_floor=mode_floor)
            rec["de_dR2_fd"] = (ep - em) / (2.0 * h)
        pieces[l] = rec
        e_wl += (2 * l + 1) * e
        de_hf += (2 * l + 1) * de
    # l = 2 at leading order: kernel (w w')^{5/2} Fhat
    s6 = np.sqrt(wom) * om**2.5
    base6 = np.outer(s6, s6)
    eQ6, info6 = _bogoliubov_ground_energy(base6 * Fm, base6 * Fp, mode_floor=mode_floor)
    de_l2 = 5.0 * eQ6 / (150.0 * math.pi**2)
    pieces[2] = {"e_Q6": float(eQ6), "de_dR2": float(de_l2), "copies": 5, "n_kept": info6["n_kept"]}
    fpp2 = fsecond_sq_integral(f)
    norm = FEWSTER_EVESON_CONSTANT_4D * fpp2
    return {"ratio": float(-e_wl / norm), "e_wl_per_volume": float(e_wl),
            "k2_hf": float(-de_hf / norm), "k2_l2": float(-de_l2 / norm),
            "k2": float(-(de_hf + de_l2) / norm), "pieces": pieces,
            "n": int(n), "omega_max": float(omega_max), "fpp2": fpp2, "label": f.label}


def optimize_sampling_momentum(family: SamplingFamily, theta0, *, bounds, d=4, n=None,
                               omega_max=None, omega_scaled=12.0, method="L-BFGS-B",
                               maxiter=60, fhat_panels=None, **kw):
    """Ascend the continuum ratio of :func:`worldline_momentum_minimize` over a
    sampling family (the momentum-space counterpart of :func:`optimize_sampling`).

    In 2d the landscape is flat at 1 (every admissible f saturates
    Flanagan's bound) — the control; in 4d the ratio is the sharp constant
    of the family member in units of the Fewster-Eveson constant, so the
    maximum over the family is the family's best lower estimate of the
    sup over f. The ratio is scale invariant, so families should expose
    shape parameters only (a scale parameter would make the objective flat
    along it). Gradients are finite-difference (the objective is one
    momentum-space solve); `bounds` is a sequence of (lo, hi). The grid
    (n, omega_max) is fixed across the family from the first handle unless
    given explicitly.
    """
    from scipy.optimize import minimize as _minimize

    theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
    bounds = [(float(lo), float(hi)) for lo, hi in bounds]
    f0 = family.make(theta0)
    n, omega_max = _grid_defaults(f0, n, omega_max, omega_scaled)
    cache = {}

    def objective(theta):
        key = tuple(np.round(np.asarray(theta, dtype=float), 12))
        if key not in cache:
            fh = family.make(np.asarray(theta, dtype=float))
            cache[key] = worldline_momentum_minimize(fh, d=d, n=n, omega_max=omega_max,
                                                     fhat_panels=fhat_panels, **kw).ratio
        return -cache[key]

    if method == "L-BFGS-B":
        out = _minimize(objective, theta0, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": int(maxiter), "eps": 1e-4})
    else:
        out = _minimize(objective, theta0, method=method, bounds=bounds,
                        options={"maxiter": int(maxiter) * 20})
    theta_star = np.atleast_1d(np.asarray(out.x, dtype=float))
    fh = family.make(theta_star)
    res = worldline_momentum_minimize(fh, d=d, n=n, omega_max=omega_max, fhat_panels=fhat_panels, **kw)
    return OptimizeSamplingResult(
        ratio=float(res.ratio), theta=theta_star, f=fh, e_min=float(res.e_min),
        bound=float(res.bound), n_iter=int(getattr(out, "nit", len(cache))),
        converged=bool(out.success), message=str(out.message))


def gaussian_poly_f(tau, coeffs, t0=0.0, n_sigmas=8.0):
    """Gaussian times an even polynomial:  f(t) = e^{-s^2/2} P(s^2),  s = (t-t0)/tau,
    P(u) = sum_k coeffs[k] u^k — everything analytic.

    With E = e^{-s^2/2}, G(u) = 2 P'(u) and S(u) = sum_k a_k 2k(2k-1) u^{k-1}:
    f' = E s (G - P) / tau,  f'' = E [S - 2uG + (u-1)P] / tau^2, so both
    int f'^2 and int f''^2 are Gaussian moments of polynomials in u,
    int e^{-s^2} s^{2n} ds = sqrt(pi) (2n-1)!!/2^n (exact, no quadrature).
    The smearing weight is f^2 = e^{-s^2} Q(u), Q = P^2, whose transform uses
    int e^{-s^2} s^{2j} e^{i k s} ds = sqrt(pi) e^{-k^2/4} (-1/2)^j He_{2j}(k/sqrt 2)
    (He the probabilists' Hermite polynomials), i.e.

        f2hat(nu) = tau sqrt(pi) e^{-(nu tau)^2/4} sum_j Q_j (-1/2)^j He_{2j}(nu tau/sqrt 2).

    `coeffs = [1]` is :func:`gaussian_f`; `[1, c]` is the modulated Gaussian of
    the papers notebook (which changes sign for c < 0 — legitimate, the weight
    f^2 >= 0, though the 2d sharp bound is then not attained: see the notebook's
    S6a). The default support is +- 8 sigma because the polynomial delays the
    decay of f^2; it is used only by quadrature fallbacks.
    """
    from numpy.polynomial import polynomial as _P
    from numpy.polynomial.hermite_e import hermeval as _hermeval

    tau = float(tau)
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    t0 = float(t0)
    a = np.atleast_1d(np.asarray(coeffs, dtype=float))
    if a.ndim != 1 or a.size < 1 or not np.any(a):
        raise ValueError(f"coeffs must be a nonzero 1d array of polynomial coefficients, got {coeffs!r}")

    k = np.arange(a.size, dtype=float)
    G = (2.0 * k * a)[1:] if a.size > 1 else np.zeros(1)          # 2 P'(u)
    S = (2.0 * k * (2.0 * k - 1.0) * a)[1:] if a.size > 1 else np.zeros(1)
    if G.size == 0:
        G = np.zeros(1)
    if S.size == 0:
        S = np.zeros(1)

    def _m(n):  # int e^{-s^2} s^{2n} ds = sqrt(pi) (2n-1)!!/2^n
        out = math.sqrt(math.pi)
        for i in range(int(n)):
            out *= (2.0 * i + 1.0) / 2.0
        return out

    def _gauss_int(poly):  # int e^{-s^2} poly(u) ds, u = s^2
        return float(sum(cj * _m(j) for j, cj in enumerate(poly)))

    Q = _P.polymul(a, a)
    R = _P.polymul(_P.polysub(G, a), _P.polysub(G, a))             # (G - P)^2
    W = _P.polyadd(_P.polysub(S, 2.0 * _P.polymulx(G)),            # S - 2uG + (u-1)P
                   _P.polysub(_P.polymulx(a), a))
    fp2 = _gauss_int(_P.polymulx(R)) / tau                          # int f'^2 dt
    fpp2 = _gauss_int(_P.polymul(W, W)) / tau**3                    # int f''^2 dt

    def _s(t):
        return (np.asarray(t, dtype=float) - t0) / tau

    def f(t):
        s = _s(t)
        return np.exp(-0.5 * s * s) * _P.polyval(s * s, a)

    def df(t):
        s = _s(t)
        u = s * s
        return np.exp(-0.5 * u) * s * (_P.polyval(u, G) - _P.polyval(u, a)) / tau

    def d2f(t):
        s = _s(t)
        u = s * s
        return np.exp(-0.5 * u) * _P.polyval(u, W) / tau**2

    herm = np.zeros(2 * Q.size)  # coefficients of sum_j Q_j (-1/2)^j He_{2j}
    for j, qj in enumerate(Q):
        herm[2 * j] += qj * (-0.5) ** j

    def f2hat(nu):
        nu = np.asarray(nu, dtype=float)
        kk = nu * tau
        return (tau * math.sqrt(math.pi) * np.exp(-0.25 * kk * kk)
                * _hermeval(kk / math.sqrt(2.0), herm) * np.exp(1j * nu * t0))

    lab = ",".join(f"{c:g}" for c in a)
    return FHandle(f=f, df=df, d2f=d2f, t_lo=t0 - n_sigmas * tau, t_hi=t0 + n_sigmas * tau,
                   fp2=fp2, label=f"gaussian_poly(tau={tau:g}, P=[{lab}])",
                   f2hat=f2hat, fpp2=fpp2)


def box_wavepacket_state(f: FHandle, *, n_boxes=300, omega_max=None, omega_scaled=12.0,
                         power=2, nodes=12, operator="phidot2", mode_floor=1e-14,
                         state_reg=1e-10, clip=1.0 - 1e-8):
    """An EXPLICIT physical state whose smeared-energy expectation is a
    rigorous upper bound on the worldline infimum (a rigorous lower bound on
    |x0|), with no appeal to solver convergence.

    Orthonormal box wavepackets b_j = int_{box j} a(w) dw / sqrt(Delta_j)
    (disjoint boxes tiling 0 < w < omega_max, uniform in u = w^{1/power}) are
    normalizable modes of the continuum field. The smeared operator restricted
    to their span has EXACT matrix elements
        A_jk = int_{box j} int_{box k} K_A(w, w') dw dw' / sqrt(Delta_j Delta_k),   B_jk likewise,
    computed by Gauss-Legendre on each box pair -- exponentially convergent
    because the integrand is analytic on every piece once the diagonal boxes
    are split along w = w' (the only non-analytic locus of a cusped transform
    such as the Ford-Roman weight's e^{-|nu| tau}). The Bogoliubov vacuum of
    that finite quadratic form, tensored with the ordinary vacuum on the
    orthogonal complement, is a physical (normalizable) Gaussian state, and
    because the operator is normal-ordered the complement contributes exactly
    zero, so its expectation value is the exact variational energy of the kept
    block (:func:`_variational_energy`, which includes the Riccati residual;
    Takagi values are clipped at `clip` so the state is normalizable). Hence

        inf_psi <O>  <=  <O>_{this state},

    an inequality that holds for the continuum theory by construction; the
    only numerical ingredient is the quadrature of analytic double integrals,
    whose error is measured by doubling `nodes` (the papers notebook does).
    Modes whose sampling weight is below `mode_floor` of the largest are left
    in the exact vacuum (rigorous: their contribution is then exactly zero,
    and no second-order estimate is added). If the Riccati start lands on the
    non-normalizable branch (|Z| >= 1: numerically null Bogoliubov
    frequencies), Z is re-solved with the weights shifted by `state_reg`
    times the largest -- the resulting state is still an explicit physical
    state and its energy is still evaluated EXACTLY on the unshifted form
    (the same device as :func:`mode_space_minimize`).

    `operator`: 'phidot2' (kernel 2 A_0, l = 0; the object whose moments
    Fewster-Ford-Roman 2012 compute), 'gradphi2' (l = 1 copies) or 'T00'. The
    T_00 state is the l = 0 state (+) the three l = 1 copies squeezed with the
    same Z (the kernels are proportional, so the minimizer is the same), and
    its expectation equals the phidot2 one exactly -- returned for all three.
    Returns a dict: e_state (per unit volume), ratio_FE = e_state / (-C_FE
    int f''^2), the boxes, A, B, Z, n_kept, z_norm, riccati_residual and
    `e_riccati` (the (1/2)Tr(conj(B) Z) value the residual-free formula would
    give, for comparison). In the Ford-Roman weight's case the FFR-2012
    dimensionless value is x0_state = -ratio_FE * 27/128 >= ... i.e.
    x0 >= ratio_FE * 27/128.
    """
    if operator not in ("T00", "phidot2", "gradphi2"):
        raise ValueError(f"operator must be 'T00', 'phidot2' or 'gradphi2', got {operator!r}")
    n_boxes = int(n_boxes)
    nodes = int(nodes)
    if omega_max is None:
        omega_max = float(omega_scaled) / _rms_width(f)
    fhat = fhat_of(f)
    edges_u = np.linspace(0.0, float(omega_max) ** (1.0 / power), n_boxes + 1)
    edges = edges_u**power
    lo, hi = edges[:-1], edges[1:]
    delta = hi - lo
    x, w = leggauss(nodes)
    # all nodes of all boxes: (n_boxes, nodes)
    om = 0.5 * (hi - lo)[:, None] * x[None, :] + 0.5 * (hi + lo)[:, None]
    wq = 0.5 * (hi - lo)[:, None] * w[None, :]
    omf, wf = om.ravel(), wq.ravel()
    s = (omf ** 1.5)
    KA = np.outer(s, s) * np.asarray(fhat(omf[:, None] - omf[None, :]))
    KB = np.outer(s, s) * np.asarray(fhat(omf[:, None] + omf[None, :]))
    W = np.outer(wf, wf)
    A = (KA * W).reshape(n_boxes, nodes, n_boxes, nodes).sum(axis=(1, 3))
    B = (KB * W).reshape(n_boxes, nodes, n_boxes, nodes).sum(axis=(1, 3))
    # diagonal boxes: split along w = w' (the cusp of a non-analytic transform)
    for j in range(n_boxes):
        a, b = lo[j], hi[j]
        oo = 0.5 * (b - a) * x + 0.5 * (b + a)
        ww = 0.5 * (b - a) * w
        accA = 0.0
        for oi, wi in zip(oo, ww):
            for (p, q) in ((a, oi), (oi, b)):
                o2 = 0.5 * (q - p) * x + 0.5 * (q + p)
                w2 = 0.5 * (q - p) * w
                accA += wi * float(np.sum(w2 * (oi * o2) ** 1.5 * np.asarray(fhat(oi - o2)).real))
        A[j, j] = accA
    A = A / np.sqrt(np.outer(delta, delta))
    B = B / np.sqrt(np.outer(delta, delta))
    # per-copy normalization: T00's l = 0 kernel is A_0 = (w w')^{3/2} Fhat/(4 pi^2) per unit
    # volume; phidot^2 is 2 A_0 (l = 0 alone), (grad phi)^2 is 2 A_0/3 per l = 1 copy
    A0, B0 = A / (4.0 * math.pi**2), B / (4.0 * math.pi**2)
    Aop, Bop = 2.0 * A0, 2.0 * B0
    Aop = 0.5 * (Aop + Aop.conj().T)
    Bop = 0.5 * (Bop + Bop.T)
    alpha, UA = np.linalg.eigh(Aop)
    alpha = np.clip(alpha[::-1], 0.0, None)
    UA = UA[:, ::-1]
    keep = alpha > mode_floor * alpha[0] if mode_floor > 0 else np.ones(alpha.size, dtype=bool)
    ak = alpha[keep]
    UAk = UA[:, keep]
    Bt = (UAk.T @ Bop.conj() @ UAk).conj()
    Bt = 0.5 * (Bt + Bt.T)
    Z, rn, n_iter = _riccati_solve(ak, Bt)
    z_norm = float(np.linalg.norm(Z, 2))
    eps_used = 0.0
    if ak.size and z_norm >= 1.0 - 1e-9:
        eps = float(state_reg)
        for _ in range(6):
            Zr, rr_, nr_ = _riccati_solve(ak + eps * float(ak[0]), Bt)
            if np.linalg.norm(Zr, 2) < 1.0 - 1e-12:
                Z, rn, n_iter, eps_used = Zr, rr_, nr_, eps
                break
            eps *= 100.0
    sv, Wt = _takagi(Z)
    clipped = bool(np.any(sv >= clip))
    if clipped:
        Z = (Wt * np.minimum(sv, clip)) @ Wt.T
        Z = 0.5 * (Z + Z.T)
    e_riccati = 0.5 * float(np.trace(Bt.conj() @ Z).real)
    e_state = _variational_energy(ak, Bt, Z)          # EXACT expectation in the exhibited state
    fpp2 = fsecond_sq_integral(f)
    bound = -FEWSTER_EVESON_CONSTANT_4D * fpp2
    if operator == "gradphi2":
        pass  # three copies of (1/3) of the phidot2 form: the same total, see docstring
    return {"e_state": float(e_state), "ratio_FE": float(e_state / bound), "bound": float(bound),
            "e_riccati": float(e_riccati), "riccati_residual": float(rn), "n_newton": int(n_iter),
            "n_boxes": n_boxes, "nodes": nodes, "omega_max": float(omega_max), "power": int(power),
            "edges": edges, "A": Aop, "B": Bop, "Z": Z, "alpha_kept": ak, "n_kept": int(ak.size),
            "z_norm": z_norm, "z_norm_used": float(np.linalg.norm(Z, 2)), "clipped": clipped,
            "state_reg_used": float(eps_used), "operator": operator, "label": f.label,
            "statement": "inf <O> <= e_state for the continuum theory (explicit normalizable "
                         "Gaussian state; complement in vacuum contributes 0)"}
