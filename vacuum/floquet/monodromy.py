"""One-period maps (monodromy), stability charts and momentum gaps (Layer 5).

For the periodically driven quadratic Hamiltonian H(t) = (1/2)(p.p + x.K(t).x)
the Heisenberg quadratures evolve linearly, R(t) = S(t) R(0), with
S' = Omega H_mat(t) S (block ordering, hbar = 1; A. Serafini, "Quantum
Continuous Variables", CRC 2017, Sec. 3.1).  The *monodromy* (Floquet map)
S_F = S(T) over one period T = 2 pi / omega_mod is symplectic, and by Floquet's
theorem the long-time behaviour is governed by its eigenvalues: a mode is
parametrically unstable iff some |lambda| > 1 (the multipliers come in
reciprocal pairs lambda, 1/lambda for symplectic S_F) — this is the Mathieu
stability problem for a sinusoidal drive (N. W. McLachlan, "Theory and
Application of Mathieu Functions", Oxford 1947; see
:mod:`vacuum.floquet.mathieu`) and the momentum-gap problem of photonic time
crystals (Lustig et al., Opt. Express 31, 9165 (2023), Sec. 2).

Integrators
-----------
* piecewise-constant profiles: S_F is the ordered product of *exact* static
  exponentials, one per piece — no discretisation error at all;
* ``'midpoint'``: the exponential-midpoint rule S_k = exp(dt Omega H(t_k+dt/2)),
  second-order Magnus (Blanes, Casas, Oteo, Ros, Phys. Rep. 470, 151 (2009),
  Sec. 5.4) — the house integrator of ``vacuum.protocol.midpoint_evolve``;
* ``'cf4'``: the fourth-order commutator-free Magnus scheme of Blanes and
  Moan, Appl. Numer. Math. 56, 1519 (2006) (also Phys. Rep. 470, Sec. 5.5):
  two static exponentials per substep at the Gauss-Legendre nodes,
  S_k = exp(dt Omega (a1 H1 + a2 H2)) exp(dt Omega (a2 H1 + a1 H2)),
  a_{1,2} = (3 -/+ 2 sqrt 3)/12, H_i = H(t_k + c_i dt), c_{1,2} = 1/2 -/+ sqrt3/6.
  Each factor is a *static quadratic form for a duration dt/2* (a1 + a2 =
  1/2 keeps the p-p block at the identity), so the same scheme can be
  spelled as a list of static ``GeneratorStep`` s for ``vacuum.protocol``.

Every substep of every scheme is an exact symplectic exponential of a
symmetric quadratic form (built by :func:`propagator`, the spectral form of
``vacuum.core.dynamics.chain_propagator`` extended to indefinite K), so the
only error is Magnus truncation, controlled by the dt-halving convergence
gate (the same discipline as ``vacuum.detectors.nonperturbative``): halve dt
until the monodromy moves by less than ``conv_tol`` and record the accepted
dt.

Pure-functional over arrays; nothing is mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vacuum.core import Omega

from .drives import Drive

__all__ = [
    "propagator",
    "mode_propagator_coefficients",
    "FloquetMap",
    "floquet_map",
    "multiplier",
    "floquet_exponent",
    "symplectic_defect",
    "cf4_pieces",
    "mode_monodromy",
    "batched_monodromy",
    "StabilityChart",
    "stability_chart",
    "Band",
    "momentum_gaps",
    "CF4_ALPHA",
    "CF4_NODES",
]

_SQRT3 = np.sqrt(3.0)
#: Blanes-Moan CF4 weights (a1, a2) and Gauss-Legendre nodes (c1, c2).
CF4_ALPHA = ((3.0 - 2.0 * _SQRT3) / 12.0, (3.0 + 2.0 * _SQRT3) / 12.0)
CF4_NODES = (0.5 - _SQRT3 / 6.0, 0.5 + _SQRT3 / 6.0)

METHODS = ("midpoint", "cf4")


# --------------------------------------------------------------------------
# Exact propagators of static quadratic forms
# --------------------------------------------------------------------------


def mode_propagator_coefficients(w2, dt):
    """(c, s/w, w s) for x'' + w2 x = 0 over dt, any sign of w2 (vectorised).

    Returns the entries of the 2 x 2 symplectic map [[c, s/w], [-w s, c]] on
    (x, p): trigonometric for w2 > 0, hyperbolic for w2 < 0 (inverted
    oscillator, sin(wt)/w -> sinh(|w|t)/|w|, w sin(wt) -> -|w| sinh(|w|t)),
    and the free-particle limit (1, dt, 0) at w2 = 0.
    """
    w2 = np.asarray(w2, dtype=float)
    dt = np.asarray(dt, dtype=float)
    w2, dt = np.broadcast_arrays(w2, dt)
    c = np.empty_like(w2)
    sw = np.empty_like(w2)
    ws = np.empty_like(w2)
    pos = w2 > 0.0
    neg = w2 < 0.0
    zer = ~(pos | neg)
    w = np.sqrt(w2[pos])
    c[pos] = np.cos(w * dt[pos])
    s = np.sin(w * dt[pos])
    sw[pos] = s / w
    ws[pos] = w * s
    wa = np.sqrt(-w2[neg])
    c[neg] = np.cosh(wa * dt[neg])
    sh = np.sinh(wa * dt[neg])
    sw[neg] = sh / wa
    ws[neg] = -wa * sh
    c[zer] = 1.0
    sw[zer] = dt[zer]
    ws[zer] = 0.0
    return c, sw, ws


def propagator(K, dt):
    """Exact symplectic propagator exp(dt Omega diag(K, I)) for symmetric K.

    Spectral form: K = U diag(w2) U^T, S = [[C, Sw], [-Ws, C]] with
    C = U cos(w dt) U^T etc. (``vacuum.core.dynamics.chain_propagator``),
    extended to indefinite K through :func:`mode_propagator_coefficients`
    so a drive may swing K through zero (depth >= 1) without leaving the
    exact-exponential path.  Symplectic to eigh roundoff.
    """
    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    w2, U = np.linalg.eigh(0.5 * (K + K.T))
    c, sw, ws = mode_propagator_coefficients(w2, float(dt))
    C = (U * c) @ U.T
    Sw = (U * sw) @ U.T
    Ws = (U * ws) @ U.T
    return np.block([[C, Sw], [-Ws, C]])


def symplectic_defect(S):
    """max |S Omega S^T - Omega|."""
    S = np.asarray(S, dtype=float)
    Om = Omega(S.shape[0] // 2)
    return float(np.max(np.abs(S @ Om @ S.T - Om)))


def multiplier(S):
    """Largest Floquet multiplier max_i |eig(S)| (>= 1 for symplectic S)."""
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(S, dtype=float)))))


def floquet_exponent(S, period):
    """Growth rate per unit time, ln(max|eig S|)/period (0 when stable)."""
    return float(np.log(max(multiplier(S), 1.0)) / float(period))


def _as_K(K_of_t, t, N):
    K = np.asarray(K_of_t(t), dtype=float)
    if K.shape != (N, N):
        raise ValueError(f"K_of_t must return ({N}, {N}), got {K.shape}")
    return K


def cf4_pieces(K_of_t, t0, dt):
    """The two static (K, duration) pieces of one CF4 substep, in order.

    First (applied first) K_a = 2 a2 K(t1) + 2 a1 K(t2) for dt/2, then
    K_b = 2 a1 K(t1) + 2 a2 K(t2) for dt/2, t_i = t0 + c_i dt.  Because
    2 a1 + 2 a2 = 1 each piece is a genuine H = (1/2)(p.p + x.K.x) — the
    pieces can be handed to ``vacuum.protocol`` as static generator steps.
    """
    a1, a2 = CF4_ALPHA
    c1, c2 = CF4_NODES
    K1 = np.asarray(K_of_t(t0 + c1 * dt), dtype=float)
    K2 = np.asarray(K_of_t(t0 + c2 * dt), dtype=float)
    Ka = 2.0 * a2 * K1 + 2.0 * a1 * K2
    Kb = 2.0 * a1 * K1 + 2.0 * a2 * K2
    return ((Ka, 0.5 * dt), (Kb, 0.5 * dt))


def _period_map_fixed(K_of_t, period, n_steps, method, t0, N):
    """Monodromy at a fixed substep count (no gate)."""
    dt = float(period) / int(n_steps)
    S = np.eye(2 * N)
    for k in range(int(n_steps)):
        ta = t0 + k * dt
        if method == "midpoint":
            S = propagator(_as_K(K_of_t, ta + 0.5 * dt, N), dt) @ S
        else:
            for Kp, tau in cf4_pieces(K_of_t, ta, dt):
                S = propagator(Kp, tau) @ S
    return S, dt


def _exact_piecewise_map(drive: Drive, t0):
    """Exact monodromy of a piecewise-constant drive from static exponentials."""
    T = drive.period
    S = np.eye(2 * drive.N)
    t = t0
    for value, frac in drive.pieces:
        tau = frac * T
        K = drive(t + 0.5 * tau)  # constant on the piece; midpoint is safe
        S = propagator(K, tau) @ S
        t += tau
    return S


@dataclass
class FloquetMap:
    """A one-period symplectic map with its convergence provenance.

    ``movement`` is max|S_l - S_{l-1}| / max(1, max|S_l|) at the accepted
    level; ``history`` has one row per halving level.  ``exact`` marks a
    piecewise-constant drive assembled from static exponentials.
    """

    S: np.ndarray
    period: float
    t0: float
    n_steps: int
    dt: float
    method: str
    exact: bool
    converged: Optional[bool]
    halvings: int
    movement: Optional[float]
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def multiplier(self) -> float:
        return multiplier(self.S)

    @property
    def exponent(self) -> float:
        return floquet_exponent(self.S, self.period)

    @property
    def symplectic_defect(self) -> float:
        return symplectic_defect(self.S)


def floquet_map(
    K_of_t,
    period=None,
    n_steps=64,
    *,
    method="cf4",
    conv_tol=1e-10,
    max_halvings=12,
    t0=0.0,
    N=None,
    require_convergence=True,
):
    """Monodromy S_F over one period (README API: ``floquet_map``).

    Parameters
    ----------
    K_of_t : Drive or callable t -> (N, N) K
        A :class:`Drive` supplies its own period and, when piecewise
        constant, is integrated *exactly* (``n_steps``, ``method`` and the
        gate are then moot and ``exact=True`` is recorded).
    period : float, optional
        Required for a bare callable; defaults to ``K_of_t.period``.
    n_steps : int
        Substeps per period at the coarsest gate level (doubled per level).
    method : {'cf4', 'midpoint'}
        Magnus integrator (module docstring).
    conv_tol : float or None
        dt-halving gate: accept the first level whose monodromy moved by
        less than this (relative to max(1, max|S|)) under a halving; None
        disables the gate (single pass, ``converged=None``).
    max_halvings : int
        Levels tried before giving up.
    t0 : float
        Phase origin: the map is S(t0 + T <- t0).
    N : int, optional
        Number of modes for a bare callable that cannot be probed (inferred
        from ``K_of_t(t0)`` otherwise).
    require_convergence : bool
        Raise RuntimeError if the gate does not close (default); else return
        the finest level with ``converged=False``.

    Returns
    -------
    FloquetMap
    """
    if isinstance(K_of_t, Drive):
        if period is None:
            period = K_of_t.period
        N = K_of_t.N
        if K_of_t.piecewise_constant:
            S = _exact_piecewise_map(K_of_t, float(t0))
            return FloquetMap(
                S=S,
                period=float(period),
                t0=float(t0),
                n_steps=len(K_of_t.pieces),
                dt=float("nan"),
                method="exact",
                exact=True,
                converged=True,
                halvings=0,
                movement=0.0,
                history=[],
            )
    if period is None:
        raise ValueError("period is required for a bare callable K_of_t")
    if N is None:
        N = int(np.asarray(K_of_t(float(t0))).shape[0])
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    period = float(period)
    t0 = float(t0)

    if conv_tol is None:
        S, dt = _period_map_fixed(K_of_t, period, n_steps, method, t0, N)
        return FloquetMap(S, period, t0, n_steps, dt, method, False, None, 0, None, [])

    history: List[Dict[str, Any]] = []
    S_prev = None
    S = None
    dt = None
    movement = None
    converged = False
    level = 0
    for level in range(int(max_halvings) + 1):
        n = n_steps * 2**level
        S, dt = _period_map_fixed(K_of_t, period, n, method, t0, N)
        if S_prev is not None:
            movement = float(np.max(np.abs(S - S_prev)) / max(1.0, np.max(np.abs(S))))
        history.append(
            {"level": level, "n_steps": n, "dt": dt, "movement": movement,
             "multiplier": multiplier(S)}
        )
        if movement is not None and movement < conv_tol:
            converged = True
            break
        S_prev = S
    if not converged and require_convergence:
        raise RuntimeError(
            f"floquet_map: monodromy did not converge to {conv_tol:g} within "
            f"{max_halvings} dt-halvings (last movement "
            f"{'n/a' if movement is None else f'{movement:.3e}'})"
        )
    return FloquetMap(
        S=S,
        period=period,
        t0=t0,
        n_steps=n_steps * 2**level,
        dt=dt,
        method=method,
        exact=False,
        converged=converged,
        halvings=level,
        movement=movement,
        history=history,
    )


# --------------------------------------------------------------------------
# Mode-resolved (2 x 2) monodromies: the fast path for diagonal drives
# --------------------------------------------------------------------------


def _batched_step(M, w2, dt):
    """M <- P(w2, dt) @ M for stacked 2 x 2 maps (broadcast over leading dims)."""
    c, sw, ws = mode_propagator_coefficients(w2, dt)
    m00, m01, m10, m11 = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    out = np.empty_like(M)
    out[..., 0, 0] = c * m00 + sw * m10
    out[..., 0, 1] = c * m01 + sw * m11
    out[..., 1, 0] = -ws * m00 + c * m10
    out[..., 1, 1] = -ws * m01 + c * m11
    return out


def batched_monodromy(w2_of_phase, period, n_steps, method="cf4", pieces=None):
    """Stacked 2 x 2 monodromies of x'' + w2(theta) x = 0, theta = 2 pi t / T.

    Parameters
    ----------
    w2_of_phase : callable theta -> array (...,)
        Frequency-squared as a function of the drive phase, for a stack of
        independent modes / grid points (any leading shape).
    period : float or array broadcastable to the stack
        Drive period(s).
    n_steps : int
        Substeps per period.
    method : {'cf4', 'midpoint'}
    pieces : ((value, fraction), ...) or None
        If given, the profile is piecewise constant and the map is exact:
        ``w2_of_phase`` is evaluated at the centre phase of each piece.

    Returns
    -------
    M : array (..., 2, 2)
    """
    period = np.asarray(period, dtype=float)
    if pieces is not None:
        theta = 0.0
        M = None
        for value, frac in pieces:
            w2 = np.asarray(w2_of_phase(theta + np.pi * frac), dtype=float)
            dt = frac * period
            w2, dt = np.broadcast_arrays(w2, dt)
            if M is None:
                M = np.zeros(w2.shape + (2, 2))
                M[..., 0, 0] = 1.0
                M[..., 1, 1] = 1.0
            M = _batched_step(M, w2, dt)
            theta += 2.0 * np.pi * frac
        return M
    n = int(n_steps)
    dtheta = 2.0 * np.pi / n
    a1, a2 = CF4_ALPHA
    c1, c2 = CF4_NODES
    M = None
    for k in range(n):
        th = k * dtheta
        if method == "midpoint":
            w2 = np.asarray(w2_of_phase(th + 0.5 * dtheta), dtype=float)
            dt = period / n
            w2, dt = np.broadcast_arrays(w2, dt)
            if M is None:
                M = np.zeros(w2.shape + (2, 2))
                M[..., 0, 0] = 1.0
                M[..., 1, 1] = 1.0
            M = _batched_step(M, w2, dt)
        elif method == "cf4":
            w1 = np.asarray(w2_of_phase(th + c1 * dtheta), dtype=float)
            w2b = np.asarray(w2_of_phase(th + c2 * dtheta), dtype=float)
            dt = 0.5 * period / n
            wa = 2.0 * a2 * w1 + 2.0 * a1 * w2b
            wb = 2.0 * a1 * w1 + 2.0 * a2 * w2b
            wa, dt = np.broadcast_arrays(wa, dt)
            wb = np.broadcast_to(wb, wa.shape)
            if M is None:
                M = np.zeros(wa.shape + (2, 2))
                M[..., 0, 0] = 1.0
                M[..., 1, 1] = 1.0
            M = _batched_step(M, wa, dt)
            M = _batched_step(M, wb, dt)
        else:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    return M


def mode_monodromy(
    drive: Drive,
    omega_k_sq,
    n_steps=64,
    *,
    method="cf4",
    conv_tol=1e-10,
    max_halvings=12,
    require_convergence=True,
):
    """Per-normal-mode 2 x 2 monodromies of a diagonal drive, gated.

    Returns ``(M, info)`` with M of shape (len(omega_k_sq), 2, 2) and info a
    dict (n_steps, halvings, movement, converged, exact).  For a
    piecewise-constant profile the maps are exact.
    """
    if not drive.diagonal:
        raise ValueError("mode_monodromy needs a diagonal drive ('coupling'/'mass')")
    w2 = np.asarray(omega_k_sq, dtype=float)

    def w2_of_phase(theta):
        return drive.mode_freq_sq(w2, theta / drive.omega_mod)

    if drive.piecewise_constant:
        M = batched_monodromy(w2_of_phase, drive.period, 1, pieces=drive.pieces)
        return M, {"n_steps": len(drive.pieces), "halvings": 0, "movement": 0.0,
                   "converged": True, "exact": True}
    if conv_tol is None:
        M = batched_monodromy(w2_of_phase, drive.period, n_steps, method)
        return M, {"n_steps": int(n_steps), "halvings": 0, "movement": None,
                   "converged": None, "exact": False}
    M_prev = None
    movement = None
    converged = False
    n = int(n_steps)
    for level in range(int(max_halvings) + 1):
        n = int(n_steps) * 2**level
        M = batched_monodromy(w2_of_phase, drive.period, n, method)
        if M_prev is not None:
            movement = float(np.max(np.abs(M - M_prev)) / max(1.0, np.max(np.abs(M))))
            if movement < conv_tol:
                converged = True
                break
        M_prev = M
    if not converged and require_convergence:
        raise RuntimeError(
            f"mode_monodromy: not converged to {conv_tol:g} in {max_halvings} halvings "
            f"(last movement {movement})"
        )
    return M, {"n_steps": n, "halvings": level, "movement": movement,
               "converged": converged, "exact": False}


def _multiplier_2x2(M):
    """max |eig| of stacked 2 x 2 symplectic maps from the trace (det = 1)."""
    tr = M[..., 0, 0] + M[..., 1, 1]
    at = np.abs(tr)
    out = np.ones_like(at)
    unstable = at > 2.0
    out[unstable] = 0.5 * (at[unstable] + np.sqrt(at[unstable] ** 2 - 4.0))
    return out, tr


# --------------------------------------------------------------------------
# Stability chart and momentum gaps
# --------------------------------------------------------------------------


@dataclass
class StabilityChart:
    """max |eig S_F| over a (depth, omega_mod) grid (README: ``stability_chart``).

    ``multiplier[i, j]`` is the largest Floquet multiplier at
    (depth_grid[i], omega_grid[j]); for diagonal drives ``mode_multiplier``
    resolves it per normal mode (ascending ``omega_k``) and ``trace`` holds
    the per-mode monodromy traces (|tr| > 2 <=> unstable), which is what
    :func:`momentum_gaps` reads.  ``k`` carries the chain momenta when the
    chart was built with a chain mass (dispersion inversion), else None.
    """

    depth_grid: np.ndarray
    omega_grid: np.ndarray
    multiplier: np.ndarray
    omega_k: np.ndarray
    kind: str
    profile: str
    mode_multiplier: Optional[np.ndarray] = None
    trace: Optional[np.ndarray] = None
    k: Optional[np.ndarray] = None
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def unstable(self) -> np.ndarray:
        """Boolean (depth, omega) mask of parametric instability."""
        return self.multiplier > 1.0 + self.info.get("tol", 1e-9)


def _chain_momenta(omega_k, mass):
    """Invert omega^2 = m^2 + 4 sin^2(k/2) (unit-spacing chain) for k in [0, pi]."""
    arg = np.sqrt(np.clip((omega_k**2 - float(mass) ** 2) / 4.0, 0.0, 1.0))
    return 2.0 * np.arcsin(arg)


def stability_chart(
    K0,
    depth_grid,
    omega_grid,
    *,
    profile="cos",
    kind="coupling",
    mass_sq=None,
    pattern=None,
    L0=None,
    site=0,
    n_steps=32,
    method="cf4",
    conv_tol=1e-10,
    max_halvings=12,
    mass=None,
    tol=1e-9,
):
    """Largest Floquet multiplier per (depth, omega_mod) (README API).

    For diagonal kinds ('coupling', 'mass') the whole grid is integrated
    at once as stacked 2 x 2 mode maps (mode-resolved output); other kinds
    fall back to one full monodromy per grid point.  ``mass`` (the chain
    mass, if K0 is a unit-spacing harmonic chain) labels modes by momentum.
    """
    from .drives import modulated_K

    K0 = np.asarray(K0, dtype=float)
    depth_grid = np.asarray(depth_grid, dtype=float).ravel()
    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    w2, _ = np.linalg.eigh(0.5 * (K0 + K0.T))
    omega_k = np.sqrt(np.clip(w2, 0.0, None))
    k = None if mass is None else _chain_momenta(omega_k, mass)
    nd, nw, N = depth_grid.size, omega_grid.size, w2.size
    probe = modulated_K(K0, float(depth_grid[0]), float(omega_grid[0]), profile,
                        kind=kind, mass_sq=mass_sq, pattern=pattern, L0=L0, site=site)
    info: Dict[str, Any] = {"tol": float(tol), "method": method}

    if probe.diagonal:
        D = depth_grid[:, None, None]
        W = omega_grid[None, :, None]
        w2s = w2[None, None, :]
        f = probe.f

        if kind == "coupling":
            def w2_of_phase(theta):
                return w2s * (1.0 + D * f(theta))
        else:
            def w2_of_phase(theta):
                return w2s + float(mass_sq) * D * f(theta)

        period = 2.0 * np.pi / W
        if probe.piecewise_constant:
            M = batched_monodromy(w2_of_phase, period, 1, pieces=probe.pieces)
            info.update(n_steps=len(probe.pieces), halvings=0, movement=0.0,
                        converged=True, exact=True)
        else:
            M_prev = None
            movement = None
            converged = conv_tol is None
            n = int(n_steps)
            for level in range(int(max_halvings) + 1):
                n = int(n_steps) * 2**level
                M = batched_monodromy(w2_of_phase, period, n, method)
                if conv_tol is None:
                    break
                if M_prev is not None:
                    movement = float(
                        np.max(np.abs(M - M_prev)) / max(1.0, np.max(np.abs(M)))
                    )
                    if movement < conv_tol:
                        converged = True
                        break
                M_prev = M
            if not converged:
                raise RuntimeError(
                    f"stability_chart: not converged to {conv_tol:g} in "
                    f"{max_halvings} halvings (last movement {movement})"
                )
            info.update(n_steps=n, halvings=level, movement=movement,
                        converged=converged, exact=False)
        mode_mult, tr = _multiplier_2x2(M)
        mult = np.max(mode_mult, axis=2)
        return StabilityChart(
            depth_grid=depth_grid, omega_grid=omega_grid, multiplier=mult,
            omega_k=omega_k, kind=kind, profile=probe.profile,
            mode_multiplier=mode_mult, trace=tr, k=k, info=info,
        )

    mult = np.empty((nd, nw))
    worst = {"halvings": 0, "movement": 0.0, "n_steps": 0}
    for i, d in enumerate(depth_grid):
        for j, w in enumerate(omega_grid):
            drv = modulated_K(K0, float(d), float(w), profile, kind=kind,
                              mass_sq=mass_sq, pattern=pattern, L0=L0, site=site)
            fm = floquet_map(drv, n_steps=n_steps, method=method,
                             conv_tol=conv_tol, max_halvings=max_halvings)
            mult[i, j] = fm.multiplier
            worst["halvings"] = max(worst["halvings"], fm.halvings)
            worst["n_steps"] = max(worst["n_steps"], fm.n_steps)
            if fm.movement is not None:
                worst["movement"] = max(worst["movement"], fm.movement)
    info.update(worst, converged=True, exact=probe.piecewise_constant)
    return StabilityChart(
        depth_grid=depth_grid, omega_grid=omega_grid, multiplier=mult,
        omega_k=omega_k, kind=kind, profile=probe.profile, k=k, info=info,
    )


@dataclass(frozen=True)
class Band:
    """A contiguous band of parametrically unstable normal modes."""

    mode_lo: int
    mode_hi: int  # inclusive
    omega_lo: float
    omega_hi: float
    multiplier_max: float
    k_lo: Optional[float] = None
    k_hi: Optional[float] = None


def momentum_gaps(chart: StabilityChart, tol=None):
    """Unstable k-bands per grid point (README API: ``momentum_gaps``).

    Returns ``bands[i][j]`` = tuple of :class:`Band` for
    (depth_grid[i], omega_grid[j]): contiguous runs of normal modes whose
    Floquet multiplier exceeds 1 + tol — the momentum gaps of the photonic
    time crystal (Lustig et al. 2023, Fig. 1(d)) resolved on the lattice.
    Needs a mode-resolved chart (diagonal drive).
    """
    if chart.mode_multiplier is None:
        raise ValueError("momentum_gaps needs a mode-resolved chart (diagonal drive)")
    tol = chart.info.get("tol", 1e-9) if tol is None else float(tol)
    nd, nw, N = chart.mode_multiplier.shape
    out = []
    for i in range(nd):
        row = []
        for j in range(nw):
            m = chart.mode_multiplier[i, j]
            unstable = m > 1.0 + tol
            bands = []
            start = None
            for idx in range(N + 1):
                on = idx < N and unstable[idx]
                if on and start is None:
                    start = idx
                elif not on and start is not None:
                    lo, hi = start, idx - 1
                    bands.append(
                        Band(
                            mode_lo=lo,
                            mode_hi=hi,
                            omega_lo=float(chart.omega_k[lo]),
                            omega_hi=float(chart.omega_k[hi]),
                            multiplier_max=float(np.max(m[lo:hi + 1])),
                            k_lo=None if chart.k is None else float(chart.k[lo]),
                            k_hi=None if chart.k is None else float(chart.k[hi]),
                        )
                    )
                    start = None
            row.append(tuple(bands))
        out.append(tuple(row))
    return tuple(out)
