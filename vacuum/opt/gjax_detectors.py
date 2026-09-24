"""Differentiable twins of the nonperturbative detector stack (Layer 6).

JAX mirrors of the public ``coupling='xx'`` API of
``vacuum.detectors.nonperturbative``, ``vacuum.detectors.smearing`` and
``vacuum.detectors.imperfections`` built on :mod:`vacuum.opt.gjax`.  Every
function is a pure, jit/grad-safe function of arrays with the conventions
of docs/API.md (hbar = 1, V_vac = I/2, nu >= 1/2, block quadrature ordering
R = (x_1..x_N, p_1..p_N), nats).  Static structure (numbers of modes,
detector mode sets, grid sizes, channel axes) is trace-time data; the
physics (couplings, gaps, smearing centres/widths, waveform samples, channel
rates, temperature) may be traced.

What is mirrored, and how exactly
---------------------------------
- :func:`gaussian_profile` / :func:`cos2_profile` / :func:`profile_matrix`
  mirror ``vacuum.detectors.smearing.Smearing`` (normalized, sum F = 1) and
  ``nonperturbative.profile_matrix`` — with the centre and width traceable,
  which is what "separation-smearing" gradients need.
- :func:`attach_detectors` mirrors the 'xx' branch of
  ``nonperturbative.attach_detectors`` (the (N_tot, N_tot) coupling matrix
  K_tot = [[K, lam F], [lam F^T, diag(Omega_d^2)]]).  The derivative
  coupling 'xp' is NOT mirrored here: it is being finished in
  ``vacuum.detectors`` by another build and its public contract is not
  final; this module uses the 'xx' API only.
- :func:`initial_state` mirrors ``nonperturbative.initial_state`` (T = 0:
  ground state of blockdiag(K, Omega_d^2); T > 0: thermal field block
  tensor detector ground states) — minus the passivity audit, which is a
  numpy-side standing audit and is run on every round trip
  (``vacuum.opt.objectives``), never inside a traced function.
- :func:`protocol_evolve` mirrors ``nonperturbative.protocol_evolve_fixed``
  step for step: exponential-midpoint (second-order Magnus) symplectic
  stepping S_k = exp(dt Omega H(t_k + dt/2)) over a FIXED grid of
  substeps (:func:`evolution_grid` reproduces the numpy substep layout of
  a grid ``ts`` with ``splits`` substeps per interval), the telescoped
  drive work sum_k (1/2) Tr((H_k - H_{k-1}) V) (Alicki's first-law split,
  J. Phys. A 12, L103 (1979)), Strang-interleaved channels with the
  before/after mean-energy dissipation entries, and the closing boundary
  term against H(t_end).  The dt-halving convergence GATE of the numpy
  stack is deliberately not mirrored: a gate is a data-dependent loop, and
  the traced twin runs at one declared resolution.  The gate is exactly
  what the round trip supplies — every optimized theta* is re-run through
  the gated numpy ``protocol_evolve`` before it is reported.
- :func:`imperfection_XY` / :func:`with_imperfections` /
  :func:`imperfection_channel_XY` mirror
  ``vacuum.detectors.imperfections`` (frequency-aware loss then isotropic
  or single-axis Gaussian dephasing, Holevo-Werner CP by construction; the
  rate form eta = e^{-kappa dt}, D = gamma_phi (1 - eta)/kappa is the exact
  semigroup propagator, expm1-stable at kappa -> 0).
- :func:`detector_block`, :func:`detector_occupations`,
  :func:`harvested_log_negativity` mirror their numpy namesakes, except for
  the mp-margin rule: ``nonperturbative.harvested_log_negativity``
  escalates to mpmath when the partial-transpose minimum symplectic
  eigenvalue sits within 1e-10 of 1/2.  A traced function cannot; the
  twin returns the float64 value (flagged on the numpy round trip), and
  near-death negativities are therefore only ever REPORTED from the numpy
  stack.

The dead-region surrogate
-------------------------
``gjax.log_negativity`` is E_N = sum_k [nu~_k < 1/2](-ln 2 nu~_k) with the
subgradient convention 0 on the dead side (gjax design note iii): once the
detectors' partial transpose leaves the negative margin, E_N and its
gradient are identically zero and a gradient optimizer is stranded.
:func:`negativity_surrogate` returns

    E~_N = -ln(2 nu~_min),   nu~_min = min symplectic eigenvalue of the PT,

which for the TWO-detector block equals E_N exactly whenever E_N > 0 (a
two-mode Gaussian state has at most one partial-transpose symplectic
eigenvalue below 1/2 — Serafini, "Quantum Continuous Variables", CRC 2017,
Sec. 7.1; Adesso-Serafini-Illuminati, PRA 70, 022318 (2004)) and continues
smoothly to NEGATIVE values in the dead region, where its gradient points
toward the sudden-death boundary.  For blocks of more than two modes
E~_N is a lower bound on E_N (only the smallest eigenvalue is counted); the
objective layer uses the surrogate for climbing and reports E_N.

References: Brown-Martin-Martinez-Menicucci-Mann, PRD 87, 084062 (2013)
(oscillator detectors); Blanes-Casas-Oteo-Ros, Phys. Rep. 470, 151 (2009),
Sec. 5.4 (midpoint Magnus); Strang, SIAM J. Numer. Anal. 5, 506 (1968);
Holevo-Werner, PRA 63, 032312 (2001), Eq. (3.4); Vidal-Werner, PRA 65,
032314 (2002) (log-negativity).
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Sequence, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - exercised only without jax
    raise ImportError(
        "vacuum.opt.gjax_detectors needs JAX (install the '.[diff]' extra)"
    ) from exc

from vacuum.opt import gjax as G

jax.config.update("jax_enable_x64", True)

__all__ = [
    "VACUUM_NU",
    "DEPHASING_AXES",
    "gaussian_profile",
    "cos2_profile",
    "point_profile",
    "profile_matrix",
    "attach_detectors",
    "hamiltonian_matrix",
    "initial_state",
    "direct_sum_cov",
    "evolution_grid",
    "EvolveResult",
    "protocol_evolve",
    "detector_block",
    "detector_occupations",
    "harvested_log_negativity",
    "negativity_surrogate",
    "pt_min_symplectic_eigenvalue",
    "mode_scaling",
    "scale_channel",
    "loss_channel_XY",
    "dephasing_XY",
    "imperfection_XY",
    "with_imperfections",
    "imperfection_channel_XY",
    "rate_channel",
    "run_harvesting",
    "HarvestOut",
]

VACUUM_NU = 0.5
DEPHASING_AXES = ("isotropic", "p", "x")
_F64 = jnp.float64


def _asarray(x):
    return jnp.asarray(x, dtype=_F64)


# --------------------------------------------------------------------------
# Smearing profiles (twins of vacuum.detectors.smearing)
# --------------------------------------------------------------------------


def _lattice_delta(N, center, bc):
    idx = jnp.arange(int(N), dtype=_F64)
    delta = idx - _asarray(center)
    if bc == "periodic":
        delta = jnp.mod(delta + 0.5 * N, N) - 0.5 * N  # minimal image
    elif bc != "open":
        raise ValueError(f"bc must be 'open' or 'periodic', got {bc!r}")
    return delta


def gaussian_profile(N, center, width, bc="open"):
    """Normalized Gaussian lattice profile g_i = exp(-(i-c)^2/(2w^2)), sum F = 1.

    Twin of ``Smearing('gaussian', center, width, N, bc)``; ``center`` and
    ``width`` may be traced (the analytic dF/dcenter, dF/dwidth of the
    numpy class are what autodiff reproduces).  Periodic minimal-image
    displacement is not differentiable exactly at the wrap |delta| = N/2,
    as documented for the numpy class.
    """
    delta = _lattice_delta(N, center, bc)
    g = jnp.exp(-0.5 * (delta / _asarray(width)) ** 2)
    return g / jnp.sum(g)


def cos2_profile(N, center, width, bc="open"):
    """Normalized compact cos^2 lattice profile (C^1 in the parameters)."""
    delta = _lattice_delta(N, center, bc)
    w = _asarray(width)
    inside = jnp.abs(delta) <= w
    g = jnp.where(inside, jnp.cos(0.5 * jnp.pi * delta / w) ** 2, 0.0)
    return g / jnp.sum(g)


def point_profile(N, site):
    """One-hot pointlike profile at a (static) lattice site."""
    site = int(site)
    if not 0 <= site < int(N):
        raise ValueError(f"site {site} outside the lattice 0..{int(N) - 1}")
    return jnp.zeros(int(N), dtype=_F64).at[site].set(1.0)


def profile_matrix(N, sites_or_profiles):
    """Column matrix F (N, n_det): ints are pointlike sites, arrays are profiles."""
    N = int(N)
    cols = []
    for k, sp in enumerate(sites_or_profiles):
        if isinstance(sp, (int, np.integer)):
            cols.append(point_profile(N, sp))
        else:
            col = _asarray(sp)
            if col.shape != (N,):
                raise ValueError(f"detector {k}: profile shape {col.shape} != ({N},)")
            cols.append(col)
    if not cols:
        raise ValueError("need at least one detector site/profile")
    return jnp.stack(cols, axis=1)


# --------------------------------------------------------------------------
# Coupled quadratic form (the 'xx' branch of nonperturbative.attach_detectors)
# --------------------------------------------------------------------------


def attach_detectors(K, F, gaps, lambdas=0.0):
    """K_tot = [[K, lam F], [lam F^T, diag(Omega_d^2)]] (x-x coupling).

    ``F`` is the (N, n_det) profile matrix (:func:`profile_matrix`),
    ``gaps`` the (n_det,) detector frequencies, ``lambdas`` a scalar or
    (n_det,) coupling.  No positivity check on the gaps under tracing (the
    numpy twin raises on gaps <= 0; the round trip reproduces that).
    """
    K = _asarray(K)
    F = _asarray(F)
    N, n_det = F.shape
    gaps = jnp.broadcast_to(_asarray(gaps), (n_det,))
    lam = jnp.broadcast_to(_asarray(lambdas), (n_det,))
    C = F * lam[None, :]
    top = jnp.concatenate([0.5 * (K + K.T), C], axis=1)
    bot = jnp.concatenate([C.T, jnp.diag(gaps ** 2)], axis=1)
    return jnp.concatenate([top, bot], axis=0)


def hamiltonian_matrix(K_tot):
    """H_mat = diag(K_tot, I) for H = (1/2)(p.p + x.K_tot.x)."""
    K_tot = _asarray(K_tot)
    n = K_tot.shape[0]
    Z = jnp.zeros((n, n), dtype=_F64)
    return jnp.block([[0.5 * (K_tot + K_tot.T), Z], [Z, jnp.eye(n, dtype=_F64)]])


def _generator_form(H, n):
    H = _asarray(H)
    if H.shape == (n, n):
        return hamiltonian_matrix(H)
    if H.shape == (2 * n, 2 * n):
        return 0.5 * (H + H.T)
    raise ValueError(
        f"generator must return ({n}, {n}) K_tot or ({2 * n}, {2 * n}) H_mat, got {H.shape}"
    )


# --------------------------------------------------------------------------
# Initial states
# --------------------------------------------------------------------------


def direct_sum_cov(V1, V2):
    """Direct sum of two block-ordered covariances (modes of V1 first)."""
    V1 = _asarray(V1)
    V2 = _asarray(V2)
    n1 = V1.shape[0] // 2
    n2 = V2.shape[0] // 2
    n = n1 + n2
    idx1 = np.concatenate([np.arange(n1), n + np.arange(n1)])
    idx2 = np.concatenate([n1 + np.arange(n2), n + n1 + np.arange(n2)])
    V = jnp.zeros((2 * n, 2 * n), dtype=_F64)
    V = V.at[jnp.ix_(idx1, idx1)].set(V1)
    V = V.at[jnp.ix_(idx2, idx2)].set(V2)
    return V


def initial_state(K, F, gaps, T=0.0):
    """lam(0) = 0 initial covariance: vacuum (T = 0) or thermal field, detector grounds.

    Twin of ``nonperturbative.initial_state`` without the passivity audit
    (numpy-side, run on the round trip).  ``T`` may be traced; the T = 0
    branch is selected at trace time from the concrete flag ``T == 0``
    when ``T`` is concrete, otherwise the thermal formula (whose T -> 0
    limit is the vacuum, gjax ``thermal_state_cov``) is used throughout.
    """
    K = _asarray(K)
    F = _asarray(F)
    n_det = F.shape[1]
    gaps = jnp.broadcast_to(_asarray(gaps), (n_det,))
    K_tot0 = attach_detectors(K, F, gaps, 0.0)
    concrete_zero = (not isinstance(T, jax.core.Tracer)) and float(T) == 0.0
    if concrete_zero:
        return G.ground_state_cov(K_tot0)
    V_field = G.thermal_state_cov(K, T)
    V_det = G.ground_state_cov(jnp.diag(gaps ** 2))
    return direct_sum_cov(V_field, V_det)


# --------------------------------------------------------------------------
# Fixed-grid midpoint/Strang evolution (twin of protocol_evolve_fixed)
# --------------------------------------------------------------------------


def evolution_grid(ts, splits=1):
    """(t_mid, dt) arrays of the substeps numpy's ``protocol_evolve_fixed`` takes.

    Interval i of the strictly increasing grid ``ts`` is divided into
    ``splits`` equal substeps of length dt_i = (ts[i+1] - ts[i])/splits;
    substep k starts at t_a = ts[i] + k dt_i and its generator is
    evaluated at t_a + dt_i/2.  Returned as concrete numpy arrays (the
    grid is static data).
    """
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 1 or ts.size < 2 or np.any(np.diff(ts) <= 0.0):
        raise ValueError("ts must be a strictly increasing grid of >= 2 times")
    splits = int(splits)
    if splits < 1:
        raise ValueError(f"splits must be >= 1, got {splits}")
    t_mid, dts = [], []
    for i in range(ts.size - 1):
        dt = (ts[i + 1] - ts[i]) / splits
        for k in range(splits):
            t_a = ts[i] + k * dt
            t_mid.append(t_a + 0.5 * dt)
            dts.append(dt)
    return np.asarray(t_mid), np.asarray(dts)


class EvolveResult(NamedTuple):
    """Output of :func:`protocol_evolve` (all entries are jnp arrays).

    ``work`` is the telescoped drive work; ``dissipated_by_channel[j]`` the
    summed before/after mean-energy difference (positive = energy out)
    around every application of channel j; ``H_start``/``H_end`` the
    boundary quadratic forms (the ledger snapshots are booked under them).
    """

    V: jnp.ndarray
    work: jnp.ndarray
    dissipated: jnp.ndarray
    dissipated_by_channel: jnp.ndarray
    H_start: jnp.ndarray
    H_end: jnp.ndarray


def protocol_evolve(V, K_of_t, t_mid, dts, t_start, t_end, channels=()):
    """Midpoint/Strang evolution of V over the substeps (t_mid, dts).

    Parameters
    ----------
    V : (2n, 2n) array
        Input covariance.
    K_of_t : callable t -> (n, n) K_tot or (2n, 2n) H_mat
        The traced generator (closing over the protocol parameters).
    t_mid, dts : (n_steps,) arrays
        Substep midpoints and lengths (:func:`evolution_grid`).
    t_start, t_end : float
        Grid boundaries: H(t_start) opens the work telescoping, H(t_end)
        closes it, exactly as in ``protocol_evolve_fixed``.
    channels : sequence of callables dt -> (X, Y)
        Gaussian channels applied in Strang order — (X, Y) for a half
        substep before and after every symplectic step; each callable owns
        its dt-scaling (:func:`rate_channel`).  The dissipation account is
        fed from mean energies under the substep's H_mid, as in numpy.

    Returns
    -------
    EvolveResult
    """
    V = _asarray(V)
    n = V.shape[0] // 2
    Om = G.Omega(n)
    channels = tuple(channels)
    n_ch = len(channels)
    H_start = _generator_form(K_of_t(_asarray(t_start)), n)
    H_end = _generator_form(K_of_t(_asarray(t_end)), n)

    def half_channels(V, H_mid, dt, diss):
        for j, ch in enumerate(channels):
            X, Y = ch(0.5 * dt)
            E_b = 0.5 * jnp.sum(H_mid * V)
            V = X @ V @ X.T + Y
            diss = diss.at[j].add(E_b - 0.5 * jnp.sum(H_mid * V))
        return V, diss

    def body(carry, step):
        V, H_prev, work, diss = carry
        t_m, dt = step
        H_mid = _generator_form(K_of_t(t_m), n)
        work = work + 0.5 * jnp.sum((H_mid - H_prev) * V)
        V, diss = half_channels(V, H_mid, dt, diss)
        S = jax.scipy.linalg.expm(dt * (Om @ H_mid))
        V = S @ V @ S.T
        V, diss = half_channels(V, H_mid, dt, diss)
        return (V, H_mid, work, diss), None

    carry0 = (V, H_start, jnp.zeros((), dtype=_F64), jnp.zeros(n_ch, dtype=_F64))
    steps = (_asarray(t_mid), _asarray(dts))
    (V, H_prev, work, diss), _ = jax.lax.scan(body, carry0, steps)
    work = work + 0.5 * jnp.sum((H_end - H_prev) * V)
    return EvolveResult(
        V=V, work=work, dissipated=jnp.sum(diss), dissipated_by_channel=diss,
        H_start=H_start, H_end=H_end,
    )


# --------------------------------------------------------------------------
# Detector observables
# --------------------------------------------------------------------------


def detector_block(V_out, N_field):
    """Reduced covariance of the detector modes appended after N_field field modes."""
    V_out = _asarray(V_out)
    n_tot = V_out.shape[0] // 2
    N_field = int(N_field)
    if not 0 <= N_field < n_tot:
        raise ValueError(f"N_field={N_field} leaves no detector modes of {n_tot} total")
    return G.reduce(V_out, range(N_field, n_tot))


def detector_occupations(V, N_field, gaps):
    """<n_d> = (Omega <x^2> + <p^2>/Omega - 1)/2 per detector (mean-zero states)."""
    V = _asarray(V)
    n_tot = V.shape[0] // 2
    N_field = int(N_field)
    n_det = n_tot - N_field
    gaps = jnp.broadcast_to(_asarray(gaps), (n_det,))
    i = jnp.arange(N_field, n_tot)
    xx = V[i, i]
    pp = V[n_tot + i, n_tot + i]
    return 0.5 * (gaps * xx + pp / gaps - 1.0)


def _default_split(n_AB, modes_A, modes_B):
    if modes_A is None and modes_B is None:
        return (0,), tuple(range(1, n_AB))
    if modes_A is None or modes_B is None:
        raise ValueError("pass both modes_A and modes_B, or neither")
    return tuple(int(m) for m in modes_A), tuple(int(m) for m in modes_B)


def harvested_log_negativity(V_AB, modes_A=None, modes_B=None):
    """E_N of the detector block (float64; mp-margin rule lives in numpy)."""
    V_AB = _asarray(V_AB)
    n_AB = V_AB.shape[0] // 2
    A, B = _default_split(n_AB, modes_A, modes_B)
    return G.log_negativity(V_AB, A, B)


def pt_min_symplectic_eigenvalue(V_AB, modes_A=None, modes_B=None):
    """Smallest partial-transpose symplectic eigenvalue of the A|B split."""
    V_AB = _asarray(V_AB)
    n_AB = V_AB.shape[0] // 2
    A, B = _default_split(n_AB, modes_A, modes_B)
    return G.pt_min_symplectic_eigenvalue(V_AB, A, B)


def negativity_surrogate(V_AB, modes_A=None, modes_B=None):
    """-ln(2 nu~_min): equals E_N when alive (two modes), negative when dead.

    See the module docstring ("The dead-region surrogate").
    """
    return -jnp.log(2.0 * pt_min_symplectic_eigenvalue(V_AB, modes_A, modes_B))


# --------------------------------------------------------------------------
# Imperfections (twins of vacuum.detectors.imperfections)
# --------------------------------------------------------------------------


def _mode_list(N, modes):
    m = tuple(int(k) for k in modes)
    if not m:
        raise ValueError("need at least one mode index")
    if len(set(m)) != len(m):
        raise ValueError(f"repeated mode index in {m}")
    for k in m:
        if not 0 <= k < N:
            raise IndexError(f"mode index {k} out of range for N={N}")
    return m


def mode_scaling(N, modes, gaps):
    """Diagonal of Sigma = diag(sqrt(Omega_d) on x rows, 1/sqrt(Omega_d) on p rows)."""
    N = int(N)
    m = _mode_list(N, modes)
    s = jnp.ones(2 * N, dtype=_F64)
    if gaps is None:
        return s
    g = jnp.broadcast_to(_asarray(gaps), (len(m),))
    idx = np.asarray(m)
    s = s.at[idx].set(jnp.sqrt(g))
    s = s.at[N + idx].set(1.0 / jnp.sqrt(g))
    return s


def scale_channel(X, Y, s):
    """X' = Sigma^{-1} X Sigma, Y' = Sigma^{-1} Y Sigma^{-1}, Sigma = diag(s)."""
    X = _asarray(X)
    Y = _asarray(Y)
    s = _asarray(s)
    inv = 1.0 / s
    return (inv[:, None] * X) * s[None, :], (inv[:, None] * Y) * inv[None, :]


def loss_channel_XY(N, modes, eta, nbar=0.0, gaps=None):
    """Frequency-aware detector loss (X, Y): ``gjax.loss_XY`` conjugated by the rescaling."""
    N = int(N)
    m = _mode_list(N, modes)
    X, Y = G.loss_XY(N, m, eta, nbar)
    if gaps is None:
        return X, Y
    return scale_channel(X, Y, mode_scaling(N, m, gaps))


def dephasing_XY(N, modes, D, axis="isotropic", gaps=None):
    """Gaussian dephasing (X = I, Y~ = D I_2 or 2 D e_a e_a^T), frequency-aware."""
    N = int(N)
    m = _mode_list(N, modes)
    if axis not in DEPHASING_AXES:
        raise ValueError(f"axis must be one of {DEPHASING_AXES}, got {axis!r}")
    D = _asarray(D)
    if axis == "isotropic":
        X, Y = G.thermal_noise_XY(N, m, D)
    else:
        X = jnp.eye(2 * N, dtype=_F64)
        off = 0 if axis == "x" else N
        idx = off + np.asarray(m)
        Y = jnp.zeros((2 * N, 2 * N), dtype=_F64).at[idx, idx].set(2.0 * D)
    if gaps is None:
        return X, Y
    return scale_channel(X, Y, mode_scaling(N, m, gaps))


def imperfection_XY(N, modes, eta=1.0, nbar=0.0, dephasing=0.0, axis="isotropic", gaps=None):
    """Composed map: loss first, then dephasing — (Xd Xl, Xd Yl Xd^T + Yd)."""
    N = int(N)
    m = _mode_list(N, modes)
    Xl, Yl = loss_channel_XY(N, m, eta, nbar, gaps=gaps)
    Xd, Yd = dephasing_XY(N, m, dephasing, axis=axis, gaps=gaps)
    return Xd @ Xl, Xd @ Yl @ Xd.T + Yd


def with_imperfections(V, modes, eta=1.0, nbar=0.0, dephasing=0.0, axis="isotropic", gaps=None):
    """dephasing(loss(V)) on the detector modes (twin of the numpy function)."""
    V = _asarray(V)
    N = V.shape[0] // 2
    X, Y = imperfection_XY(N, modes, eta=eta, nbar=nbar, dephasing=dephasing, axis=axis, gaps=gaps)
    return X @ V @ X.T + Y


def imperfection_channel_XY(N, modes, dt, kappa=0.0, nbar=0.0, gamma_phi=0.0,
                            axis="isotropic", gaps=None):
    """(X, Y) of the exact combined loss+dephasing semigroup over a duration dt.

    eta = exp(-kappa dt), D = gamma_phi (1 - eta)/kappa (-> gamma_phi dt as
    kappa -> 0, via expm1): the closure of
    ``vacuum.detectors.imperfections.imperfection_channel``.  ``kappa``,
    ``nbar``, ``gamma_phi`` and ``dt`` may be traced.
    """
    kappa = _asarray(kappa)
    dt = _asarray(dt)
    eta = jnp.exp(-kappa * dt)
    pos = kappa > 0.0
    kappa_safe = jnp.where(pos, kappa, 1.0)
    window = jnp.where(pos, -jnp.expm1(-kappa_safe * dt) / kappa_safe, dt)
    return imperfection_XY(
        N, modes, eta=eta, nbar=nbar, dephasing=_asarray(gamma_phi) * window,
        axis=axis, gaps=gaps,
    )


def rate_channel(N, modes, kappa=0.0, nbar=0.0, gamma_phi=0.0, axis="isotropic", gaps=None):
    """Channel callable ``dt -> (X, Y)`` for :func:`protocol_evolve`."""
    N = int(N)
    m = _mode_list(N, modes)

    def channel(dt):
        return imperfection_channel_XY(
            N, m, dt, kappa=kappa, nbar=nbar, gamma_phi=gamma_phi, axis=axis, gaps=gaps
        )

    return channel


# --------------------------------------------------------------------------
# End-to-end (the "full harvesting protocol" the gradient anchors go through)
# --------------------------------------------------------------------------


class HarvestOut(NamedTuple):
    E_N: jnp.ndarray
    surrogate: jnp.ndarray
    work: jnp.ndarray
    dissipated: jnp.ndarray
    n_d: jnp.ndarray
    V_AB: jnp.ndarray
    V: jnp.ndarray


def run_harvesting(K, F, gaps_of_t, lam_of_t, ts, splits=1, T=0.0, channels=()):
    """initial_state -> protocol_evolve -> detector block -> E_N, at fixed resolution.

    ``gaps_of_t`` and ``lam_of_t`` are traced callables t -> (n_det,) (or
    scalar, broadcast).  The initial state is built at lam = 0 with the
    gaps at ``ts[0]``, as in the numpy stack.  Returns a
    :class:`HarvestOut` (E_N, its dead-region surrogate, the drive work,
    the dissipation, occupations at the exit gaps, V_AB, V).
    """
    K = _asarray(K)
    F = _asarray(F)
    N = K.shape[0]
    n_det = F.shape[1]
    ts = np.asarray(ts, dtype=float)
    t_mid, dts = evolution_grid(ts, splits)
    gaps0 = jnp.broadcast_to(_asarray(gaps_of_t(_asarray(ts[0]))), (n_det,))
    V0 = initial_state(K, F, gaps0, T)

    def K_of_t(t):
        return attach_detectors(K, F, gaps_of_t(t), lam_of_t(t))

    ev = protocol_evolve(V0, K_of_t, t_mid, dts, ts[0], ts[-1], channels=channels)
    V_AB = detector_block(ev.V, N)
    gaps1 = jnp.broadcast_to(_asarray(gaps_of_t(_asarray(ts[-1]))), (n_det,))
    return HarvestOut(
        E_N=harvested_log_negativity(V_AB),
        surrogate=negativity_surrogate(V_AB),
        work=ev.work,
        dissipated=ev.dissipated,
        n_d=detector_occupations(ev.V, N, gaps1),
        V_AB=V_AB,
        V=ev.V,
    )


# --------------------------------------------------------------------------
# Second-order pair term and its communication split (twin of M2.3 / M2.5)
# --------------------------------------------------------------------------
#
# The signaling bound of the optimization layer needs |M_comm|/|M| INSIDE the
# traced objective.  The pair term M of the second-order two-qubit UDW state
# [PKMM, PRD 92, 064042 (2015), Eq. (18)] is linear in the Wightman kernel
# and bilinear in the two switching functions, so the split of Tjoa &
# Martin-Martinez, PRD 104, 125005 (2021) [TMM21], Eqs. (31)-(33) — M = M^+
# + M^- with M^+ from Re W (Hadamard, genuine harvesting) and M^- from
# i Im W (commutator, field-mediated signaling) — is transcribed here on a
# FIXED triangular quadrature grid, step for step the algorithm of
# ``vacuum.detectors.udw._pair_quad`` (outer Gauss-Legendre panels, inner
# composite rule ending on the diagonal), with the lattice kernel's
# exponential-sum form W_AB(dt) = sum_k c_k e^{-i w_k dt},
# c_k = (U^T F_A)_k (U^T F_B)_k / (2 w_k) (``LatticeWightman.smeared``), and
# the commutator half on the doubled frequency set {+w_k, -w_k} with
# amplitudes {c_k/2, -conj(c_k)/2} (``communication.WightmanPart``).  The
# numpy stack certifies its M with a panel-halving gate; the twin runs at
# one declared panel width, and the numpy split on the round trip is what
# is reported.  Second order in the coupling: the fraction is a ratio of
# O(lambda^2) matrix elements, hence coupling-independent at leading order.


def pair_term_grid(a, b, panel, n_gl=16, n_gl_inner=4):
    """Fixed triangular grid of ``udw._pair_quad``: (t, w, tp, wp) numpy arrays.

    Outer nodes t with weights w on [a, b] (GL panels of width <= panel,
    ``n_gl`` points each); inner nodes tp with weights wp on the segments
    [a, t_0], [t_0, t_1], ... (``n_gl_inner`` points each), so that the
    cumulative inner integral ends exactly on the diagonal t' = t_i.
    """
    from numpy.polynomial.legendre import leggauss

    a, b, panel = float(a), float(b), float(panel)
    x, wg = leggauss(int(n_gl))
    n_panels = max(1, int(np.ceil((b - a) / panel - 1e-12)))
    edges = np.linspace(a, b, n_panels + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    half = 0.5 * (edges[1:] - edges[:-1])
    t = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    w = (half[:, None] * wg[None, :]).ravel()
    edges2 = np.concatenate(([a], t))
    xg, wgi = leggauss(int(n_gl_inner))
    mid2 = 0.5 * (edges2[1:] + edges2[:-1])
    half2 = 0.5 * (edges2[1:] - edges2[:-1])
    tp = (mid2[:, None] + half2[:, None] * xg[None, :]).ravel()
    wp = (half2[:, None] * wgi[None, :]).ravel()
    return t, w, tp, wp


def initial_panel(gaps, widths):
    """min(1/max|Omega|, min switching width)/8 — the M2.1 hazard-note rule."""
    om = max(abs(float(g)) for g in gaps)
    width = min(float(w) for w in widths)
    return (min(1.0 / om, width) / 8.0) if om > 0.0 else width / 8.0


def smeared_amplitudes(omega, U, F_A, F_B):
    """c_k = (U^T F_A)_k (U^T F_B)_k / (2 omega_k) — ``LatticeWightman.smeared`` (traced F)."""
    omega = _asarray(omega)
    U = _asarray(U)
    a = U.T @ _asarray(F_A)
    b = U.T @ _asarray(F_B)
    return a * b / (2.0 * omega)


def pair_term_fixed(freqs, amps, chiA_t, chiB_t, chiA_tp, chiB_tp, gap_A, gap_B, grid,
                    lam_A=1.0, lam_B=1.0):
    """M = -lam_A lam_B sum_k c_k [T1_k + T2_k] on a fixed grid (PKMM Eq. (18)).

    ``freqs`` real (n_k,), ``amps`` complex (n_k,) (may be traced);
    ``chiA_t``.. the switching values on the outer/inner nodes of ``grid``
    (:func:`pair_term_grid`), traced.  Transcription of
    ``vacuum.detectors.udw._pair_quad``.
    """
    t, w, tp, wp = grid
    t, w, tp, wp = (_asarray(x) for x in (t, w, tp, wp))
    m = t.shape[0]
    n_in = tp.shape[0] // m
    freqs = _asarray(freqs)
    amps = jnp.asarray(amps, dtype=jnp.complex128)
    gap_A = _asarray(gap_A)
    gap_B = _asarray(gap_B)
    u1 = wp * chiB_tp * jnp.exp(1j * gap_B * tp)
    u2 = wp * chiA_tp * jnp.exp(1j * gap_A * tp)
    v1 = w * chiA_t * jnp.exp(1j * gap_A * t)
    v2 = w * chiB_t * jnp.exp(1j * gap_B * t)
    Zin = jnp.exp(1j * freqs[:, None] * tp[None, :])
    Zout = jnp.exp(-1j * freqs[:, None] * t[None, :])
    I1 = jnp.cumsum((Zin * u1[None, :]).reshape(-1, m, n_in).sum(axis=2), axis=1)
    I2 = jnp.cumsum((Zin * u2[None, :]).reshape(-1, m, n_in).sum(axis=2), axis=1)
    T1 = jnp.sum(Zout * v1[None, :] * I1, axis=1)
    T2 = jnp.sum(Zout * v2[None, :] * I2, axis=1)
    return -_asarray(lam_A) * _asarray(lam_B) * jnp.sum(amps * (T1 + T2))


def communication_split_fixed(freqs, amps, chiA_t, chiB_t, chiA_tp, chiB_tp, gap_A, gap_B, grid,
                              lam_A=1.0, lam_B=1.0):
    """(M, M_vac, M_comm, comm_fraction) on a fixed grid [TMM21 Eqs. (31)-(33)].

    M_comm uses the commutator kernel i Im W = sum_k [(c_k/2) e^{-i w_k dt}
    - (conj c_k / 2) e^{+i w_k dt}]; M_vac = M - M_comm; comm_fraction =
    |M_comm|/|M| (0 when M = 0, as in ``CommunicationSplit.comm_fraction``).
    """
    freqs = _asarray(freqs)
    amps = jnp.asarray(amps, dtype=jnp.complex128)
    M = pair_term_fixed(freqs, amps, chiA_t, chiB_t, chiA_tp, chiB_tp, gap_A, gap_B, grid, lam_A, lam_B)
    f2 = jnp.concatenate([freqs, -freqs])
    a2 = jnp.concatenate([0.5 * amps, -0.5 * jnp.conj(amps)])
    M_comm = pair_term_fixed(f2, a2, chiA_t, chiB_t, chiA_tp, chiB_tp, gap_A, gap_B, grid, lam_A, lam_B)
    denom = jnp.abs(M)
    frac = jnp.where(denom > 0.0, jnp.abs(M_comm) / jnp.where(denom > 0.0, denom, 1.0), 0.0)
    return M, M - M_comm, M_comm, frac


__all__ += [
    "pair_term_grid",
    "initial_panel",
    "smeared_amplitudes",
    "pair_term_fixed",
    "communication_split_fixed",
]
