"""papers/qei-2d-sharp-constant — re-runnable notebook.

Candidate claim (methods result)
-------------------------------
The *pulled-back sampling operator* collapses the quantum-energy-inequality
minimization "inf over all physical states of the smeared normal-ordered
energy" into a single Williamson problem, and therefore recovers the SHARP
(optimal) 2d massless constant of Flanagan, PRD 56, 4922 (1997), Eq. (8),
from a lattice computation — while *exhibiting* the extremal state rather
than only bounding it. Section 4 then runs the 4d scan on the Srednicki
radial lattice with the cancellation-free (mode-space) sector minimizer,
closes the three systematics that made the first version exploratory, and
extrapolates the conjectured sharp 4d constant with a stated uncertainty
(Layer 7, leap L1 in PLAN.md).

Run
---
    .venv/bin/python papers/qei-2d-sharp-constant/notebook.py
    .venv/bin/python papers/qei-2d-sharp-constant/notebook.py --quick  # smoke, ~1 min

``--quick`` runs S0-S3 and S5 at reduced resolution and the S4a-S4c
systematics at small size, skips the 4d scan / extrapolation / extremal
state (S4d-S4f), and writes to a fresh temporary directory (or ``--out DIR``)
so a smoke run never overwrites the archive under ``data/``.

Top-to-bottom, no hidden state; every array behind every number is written
to ``data/`` by the section that computed it, and the section is named in
each file's ``_source`` field. Total runtime budget: <= 30 min on a laptop
(measured wall time is printed at the end and recorded in MANIFEST.md).

Conventions (docs/API.md): hbar = 1, V_vac = I/2, nu >= 1/2, block ordering
R = (x_1..x_N, p_1..p_N), nats. Lattice spacing a = 1 throughout, so every
"scale" below (tau0, T) is in units of a and every ratio a/scale is 1/scale.

Structure
---------
S0  build provenance
S1  extra sampling-function families (bump4, sech, modulated Gaussian),
    with finite-difference self-checks of every analytic derivative
S2  the 2d recovery at higher resolution than tests/test_qei.py:
    S2a  a/tau0 axis, Gaussian, box ratio g = 100 (test used g = 80, 5 pts)
    S2b  box axis, r(g) at fixed tau0 -> the two-parameter model
             1 - r  =  A / g^2  +  c / tau0^2
    S2c  mass axis, m*tau0 -> 0
    S2d  four f-families, each with its own (scale, wall) fit -> r_inf
    S2e  optimize_sampling on a 2-parameter family (analytic envelope grad)
S3  extremal states exhibited: covariances, symplectic spectra, space and
    time profiles of the negative-energy pocket
S4  4d Srednicki-lattice scan with the cancellation-free minimizer:
    S4a  the archived noise-floor configuration re-run: floor gone
    S4b  mpmath Williamson cross-check (vacuum.core.precision)
    S4c  the ball measure derived from Srednicki's Eq. 10
    S4d  the (n_ball, R/tau) product grid, exact Fourier sampling
    S4e  both orders of the double limit a/tau -> 0, R/tau -> 0
    S4f  the 4d extremal state exhibited
S5  audit battery at the paper's resolution
S6  the same variational problem in the continuum, in momentum space (no
    lattice): the 2d Flanagan control for six families, the 4d worldline
    constant, the DERIVED R/tau expansion (even powers; d2 from Hellmann-
    Feynman on l = 0, 1 plus the l = 2 sector), the lattice error exponent
    against the continuum, the re-extrapolation of the archived lattice
    tables, and the 4d sampling-family scan and optimizations (compact,
    Lorentzian-like, modulated Gaussian) with the 2d control of each

``--section S6`` runs S6 alone against the archived S4 tables in ``data/``
(it writes only ``s6*.json`` and merges its headline into ``summary.json``).

Nothing here modifies any vacuum.* module; the notebook is a pure consumer
of vacuum.core / vacuum.inequalities (the mode-space minimizer it uses in
S4 lives in vacuum.inequalities.qei and is regression-tested in
tests/test_qei.py).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from vacuum.core import (
    chain_propagator,
    ground_state_cov,
    harmonic_chain_K,
    reduce as reduce_cov,
    symplectic_eigenvalues,
)
from vacuum.core.radial import srednicki_K
from vacuum.inequalities.qei import (
    FEWSTER_EVESON_CONSTANT_4D,
    _radial_density_terms,
    ball_momentum_minimize,
    bump_f,
    gaussian_poly_f,
    lorentzian_f,
    mode_space_extremal_state,
    naive_volume,
    optimize_sampling_momentum,
    qei_minimize_mp,
    radial_ball_qei,
    radial_sector_minimize,
    fsecond_sq_integral,
    box_wavepacket_state,
    rho2_coefficient,
    shell_volume,
    sqrt_lorentzian_f as q_sqrt_lorentzian_f,
    worldline_momentum_minimize,
)
from vacuum.inequalities import (
    FEWSTER_EVESON_CONSTANT_2D,
    SHARP_CONSTANT_2D,
    FHandle,
    SamplingFamily,
    chain_energy_density_form,
    cos2_f,
    embed_extremal,
    fprime_sq_integral,
    gaussian_f,
    optimize_sampling,
    qei_audit,
    qei_bound_2d,
    qei_minimize,
    radial_ball_weights,
    radial_sampling_operator,
    sampling_operator,
    smeared_energy,
    squeezed_pocket,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

#: Fewster-Eveson 4d constant in the f^2 smearing convention:
#: PRD 58, 084010 (1998), Eq. (40), int f^2 <T_00> >= -(1/(16 pi^2)) int f''^2.
FE_CONSTANT_4D = 1.0 / (16.0 * math.pi**2)
assert FE_CONSTANT_4D == FEWSTER_EVESON_CONSTANT_4D


# =========================================================================
# S0. provenance
# =========================================================================


def s0_provenance():
    """Record the exact build this run's numbers came from."""
    import scipy

    import vacuum

    info = {
        "_source": "notebook.py::s0_provenance",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "vacuum_package": str(Path(vacuum.__file__).resolve().parent),
        "vacuum_version": getattr(vacuum, "__version__", "0.1.0 (pyproject)"),
        "constants": {
            "SHARP_CONSTANT_2D (Flanagan 1997 Eq. 8, rho=f^2)": SHARP_CONSTANT_2D,
            "FEWSTER_EVESON_CONSTANT_2D (FE 1998 Eq. 37)": FEWSTER_EVESON_CONSTANT_2D,
            "FE_CONSTANT_4D (FE 1998 Eq. 40, f^2 convention)": FE_CONSTANT_4D,
        },
    }
    try:
        root = HERE.parents[1]  # repo root (papers/<slug>/ -> repo)
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, check=True, timeout=5, text=True)
        info["git"] = out.stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--",
                                "vacuum/inequalities/qei.py", "tests/test_qei.py",
                                "papers/qei-2d-sharp-constant/notebook.py",
                                "papers/qei-2d-sharp-constant/MANIFEST.md",
                                "papers/qei-2d-sharp-constant/draft"],
                               capture_output=True, timeout=5, text=True).stdout
        info["git_owned_files_dirty"] = bool(dirty.strip())
        info["git_owned_files_status"] = dirty.strip().splitlines()
    except Exception:
        info["git"] = "unavailable; build identified by the pytest run in MANIFEST.md"
    _write_json("build_info.json", info)
    return info


# =========================================================================
# S1. extra sampling-function families
# =========================================================================
#
# All are pure data-plus-functions in the FHandle contract of
# vacuum.inequalities.qei: (f, f', f'', support, analytic int f'^2, label).
# The 2d bound only needs f' in L^2, so C^1 bumps are admissible.


def bump4_f(T, t0=0.0):
    """Compact C^1 quartic bump  f(t) = (1 - u^2)^2,  u = (t - t0)/T.

    f'  = 4 (u^3 - u) / T
    f'' = (12 u^2 - 4) / T^2
    int f'^2 dt = (16/T) int_{-1}^{1} (u^3 - u)^2 du
                = (16/T) (2/3 - 4/5 + 2/7) = 256 / (105 T).
    """
    T = float(T)
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    t0 = float(t0)

    def _u(t):
        u = (np.asarray(t, dtype=float) - t0) / T
        return u, np.abs(u) < 1.0

    def f(t):
        u, inside = _u(t)
        out = np.zeros_like(u)
        out[inside] = (1.0 - u[inside] ** 2) ** 2
        return out

    def df(t):
        u, inside = _u(t)
        out = np.zeros_like(u)
        out[inside] = 4.0 * (u[inside] ** 3 - u[inside]) / T
        return out

    def d2f(t):
        u, inside = _u(t)
        out = np.zeros_like(u)
        out[inside] = (12.0 * u[inside] ** 2 - 4.0) / T**2
        return out

    _b = bump_f(T, 2.0)  # analytic transform of f^2 (spherical Bessel) and int f''^2 = 128/(5 T^3)

    def f2hat(nu):
        nu = np.asarray(nu, dtype=float)
        return _b.f2hat(nu) * np.exp(1j * nu * t0)

    return FHandle(f=f, df=df, d2f=d2f, t_lo=t0 - T, t_hi=t0 + T,
                   fp2=256.0 / (105.0 * T), label=f"bump4(T={T:g})",
                   f2hat=f2hat, fpp2=_b.fpp2)


def sech_f(tau, t0=0.0, n_taus=14.0):
    """Non-compact, non-Gaussian tail  f(t) = sech(u),  u = (t - t0)/tau.

    f'  = -sech(u) tanh(u) / tau
    f'' =  sech(u) (tanh^2 u - sech^2 u) / tau^2
    int f'^2 dt = (1/tau) int sech^2 tanh^2 du = 2 / (3 tau),
    int f''^2 dt = (1/tau^3) int sech^2 (1 - 2 sech^2)^2 du = 14 / (15 tau^3),
    f2hat(nu) = int sech^2(t/tau) e^{i nu t} dt = pi tau (nu tau) / sinh(pi nu tau / 2).
    Truncating at |u| = n_taus loses 1 - tanh^3(n_taus) ~ 6 e^{-2 n_taus}
    of int f'^2 — 4e-12 relative at the default n_taus = 14.
    """
    tau = float(tau)
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    t0 = float(t0)

    def f(t):
        u = (np.asarray(t, dtype=float) - t0) / tau
        return 1.0 / np.cosh(u)

    def df(t):
        u = (np.asarray(t, dtype=float) - t0) / tau
        return -np.tanh(u) / np.cosh(u) / tau

    def d2f(t):
        u = (np.asarray(t, dtype=float) - t0) / tau
        return (np.tanh(u) ** 2 - 1.0 / np.cosh(u) ** 2) / np.cosh(u) / tau**2

    def f2hat(nu):
        nu = np.asarray(nu, dtype=float)
        x = nu * tau
        small = np.abs(x) < 1e-8
        xs = np.where(small, 1.0, x)
        val = np.where(small, 2.0, math.pi * xs / np.sinh(0.5 * math.pi * xs))
        return tau * val * np.exp(1j * nu * t0)

    return FHandle(f=f, df=df, d2f=d2f, t_lo=t0 - n_taus * tau,
                   t_hi=t0 + n_taus * tau, fp2=2.0 / (3.0 * tau),
                   label=f"sech(tau={tau:g})", f2hat=f2hat,
                   fpp2=14.0 / (15.0 * tau**3))


def modulated_gaussian_f(tau, c, t0=0.0, n_sigmas=6.0):
    """Two-parameter family  f = E (1 + c s^2),  s = (t-t0)/tau, E = e^{-s^2/2}.

    f'  = (E s / tau) (2c - 1 - c s^2)
    f'' = (E / tau^2) [ (2c - 1) - (5c - 1) s^2 + c s^4 ]
    int f'^2 has no compact closed form used here — fp2 = None sends
    `fprime_sq_integral` to its quadrature branch (the bound is then
    computed by the same Gauss-Legendre rule as everything else).
    f may change sign; the smearing weight rho = f^2 >= 0 regardless, and
    Flanagan's rho'^2/rho = 4 f'^2 identity is finite through simple zeros
    (S6a shows the continuum 2d ratio then falls below 1 — the sharp bound
    for a weight with zeros is not the ∫f'^2 form; the 4d optimum sits at
    c > 0 where f has no zero).
    Analytic, by Gaussian moments m_{2k} = sqrt(pi) (2k-1)!!/2^k:
    f2hat(nu) = tau sqrt(pi) e^{-k^2/4} [1 + 2c (1/2 - k^2/4)
                + c^2 (3/4 - 3 k^2/4 + k^4/16)],  k = nu tau,
    int f''^2 = (1/tau^3) [a^2 m0 + 2ab m2 + (b^2 + 2ac) m4 + 2bc m6 + c^2 m8],
    a = 2c - 1, b = 1 - 5c.
    """
    tau = float(tau)
    c = float(c)
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    t0 = float(t0)

    def _s(t):
        return (np.asarray(t, dtype=float) - t0) / tau

    def f(t):
        s = _s(t)
        return np.exp(-0.5 * s * s) * (1.0 + c * s * s)

    def df(t):
        s = _s(t)
        return np.exp(-0.5 * s * s) * s * (2.0 * c - 1.0 - c * s * s) / tau

    def d2f(t):
        s = _s(t)
        s2 = s * s
        return (np.exp(-0.5 * s2) *
                ((2.0 * c - 1.0) - (5.0 * c - 1.0) * s2 + c * s2 * s2) / tau**2)

    def f2hat(nu):
        nu = np.asarray(nu, dtype=float)
        k = nu * tau
        g = tau * math.sqrt(math.pi) * np.exp(-0.25 * k * k)
        return (g * (1.0 + 2.0 * c * (0.5 - 0.25 * k * k)
                     + c * c * (0.75 - 0.75 * k * k + k**4 / 16.0)) * np.exp(1j * nu * t0))

    sp = math.sqrt(math.pi)
    m0, m2, m4, m6, m8 = sp, sp / 2, 3 * sp / 4, 15 * sp / 8, 105 * sp / 16
    a, b = 2.0 * c - 1.0, 1.0 - 5.0 * c
    fpp2 = (a * a * m0 + 2 * a * b * m2 + (b * b + 2 * a * c) * m4 + 2 * b * c * m6 + c * c * m8) / tau**3

    return FHandle(f=f, df=df, d2f=d2f, t_lo=t0 - n_sigmas * tau,
                   t_hi=t0 + n_sigmas * tau, fp2=None,
                   label=f"modgauss(tau={tau:g}, c={c:g})", f2hat=f2hat, fpp2=fpp2)


def modulated_gaussian_family(t0=0.0, n_sigmas=6.0):
    """SamplingFamily wrapper, theta = [tau, c], with analytic d f/d theta.

    d f/d tau = (E s^2 / tau) (1 + c s^2 - 2c),   d f/d c = E s^2.
    """

    def make(theta):
        th = np.asarray(theta, dtype=float).ravel()
        return modulated_gaussian_f(th[0], th[1], t0=t0, n_sigmas=n_sigmas)

    def df_dtheta(theta, ts):
        th = np.asarray(theta, dtype=float).ravel()
        tau, c = float(th[0]), float(th[1])
        s = (np.asarray(ts, dtype=float) - t0) / tau
        E = np.exp(-0.5 * s * s)
        s2 = s * s
        d_tau = E * s2 * (1.0 + c * s2 - 2.0 * c) / tau
        d_c = E * s2
        return np.vstack([d_tau, d_c])

    return SamplingFamily(make=make, df_dtheta=df_dtheta, n_params=2,
                          label="modulated_gaussian")


def s1_family_selfcheck():
    """Finite-difference check of every analytic derivative added in S1."""
    h = 1e-6
    rows = []
    cases = [
        (bump4_f(6.0), np.linspace(-5.7, 5.7, 41)),          # interior of the bump
        (sech_f(3.0), np.linspace(-24.0, 24.0, 61)),
        (modulated_gaussian_f(4.0, 0.6), np.linspace(-18.0, 18.0, 61)),
    ]
    for fh, ts in cases:
        fd1 = (fh.f(ts + h) - fh.f(ts - h)) / (2 * h)
        fd2 = (fh.df(ts + h) - fh.df(ts - h)) / (2 * h)
        d1 = float(np.max(np.abs(fh.df(ts) - fd1)))
        d2 = float(np.max(np.abs(fh.d2f(ts) - fd2)))
        q = fprime_sq_integral(fh, use_analytic=False, nodes_per_panel=40,
                               n_panels=400)
        rel = abs(q - fh.fp2) / abs(fh.fp2) if fh.fp2 is not None else float("nan")
        rows.append({"label": fh.label, "dev_fprime": d1, "dev_fsecond": d2,
                     "fp2_analytic": fh.fp2, "fp2_quad": q, "fp2_rel_dev": rel})
        assert d1 < 1e-7, f"{fh.label}: f' vs FD {d1:.2e}"
        assert d2 < 1e-6, f"{fh.label}: f'' vs FD {d2:.2e}"
        if fh.fp2 is not None:
            assert rel < 1e-10, f"{fh.label}: int f'^2 analytic vs quad {rel:.2e}"

    # the 2-parameter family's parameter derivatives
    fam = modulated_gaussian_family()
    theta = np.array([4.0, 0.6])
    ts = np.linspace(-18.0, 18.0, 61)
    ana = fam.df_dtheta(theta, ts)
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        fd = (fam.make(theta + e).f(ts) - fam.make(theta - e).f(ts)) / (2 * h)
        d = float(np.max(np.abs(ana[j] - fd)))
        rows.append({"label": f"modgauss d/dtheta[{j}]", "dev_fprime": d,
                     "dev_fsecond": float("nan"), "fp2_analytic": None,
                     "fp2_quad": float("nan"), "fp2_rel_dev": float("nan")})
        assert d < 1e-7, f"modgauss dtheta[{j}] vs FD {d:.2e}"
    _write_json("s1_family_selfcheck.json",
                {"_source": "notebook.py::s1_family_selfcheck", "rows": rows})
    return rows


# =========================================================================
# S2. the 2d recovery at higher resolution
# =========================================================================


def _ratio_at(f: FHandle, N, *, m=0.0, bc="dirichlet", constant="sharp", **kw):
    """One (f, lattice) point: the sharp-constant ratio E_min(f) / bound(f)."""
    N = int(N)
    site = N // 2
    op = sampling_operator(N, site, f, m=m, bc=bc, **kw)
    res = qei_minimize(op)
    return res.e_min / qei_bound_2d(f, constant=constant), res, op


def _odd(n):
    n = int(n)
    return n + 1 if n % 2 == 0 else n


def _gaussian_point(tau0, g=100.0, mu=0.0, **kw):
    """Gaussian sampling at box ratio g = N / tau0 and mass m = mu / tau0."""
    N = _odd(round(g * tau0))
    f = gaussian_f(tau0)
    r, res, op = _ratio_at(f, N, m=mu / tau0, **kw)
    return {"tau0": float(tau0), "g": float(g), "mu": float(mu), "N": N,
            "n_support": int(op.support.size), "e_min": float(res.e_min),
            "bound": float(qei_bound_2d(f)), "ratio": float(r),
            "gap_rel": float(res.gap / abs(res.e_min)),
            "margin_star": float(res.margin_star)}


def s2a_a_axis(taus=(6.0, 8.0, 10.0, 13.0, 16.0, 20.0, 24.0), g=100.0):
    """a/tau0 axis at fixed box ratio: the headline Richardson table."""
    rows = [_gaussian_point(t, g=g) for t in taus]
    _write_csv("s2a_a_axis.csv", rows, "notebook.py::s2a_a_axis")
    return rows


def s2b_box_axis(tau0=8.0, gs=(40.0, 56.0, 80.0, 120.0, 160.0, 220.0)):
    """Box axis at fixed tau0: isolates the Dirichlet finite-box bias."""
    rows = [_gaussian_point(tau0, g=g) for g in gs]
    _write_csv("s2b_box_axis.csv", rows, "notebook.py::s2b_box_axis")
    return rows


def s2_two_parameter_fit(rows_a, rows_b):
    """Joint fit of the model  1 - r = A/g^2 + c/tau0^2  to S2a + S2b.

    Both corrections are pure lattice/box artifacts of the *discretization*,
    not of the physics: A/g^2 is the Dirichlet finite-box bias (walls at
    +-N/2 = +-(g/2) tau0), c/tau0^2 the leading lattice-dispersion error
    (omega_k = 2|sin(k/2)| vs |k|, an O(a^2) effect). The continuum,
    infinite-box value of the model is r = 1 EXACTLY, i.e. the sharp
    Flanagan constant; the fit's job is to show the data are described by
    exactly those two artifacts with no residual offset, so the reported
    recovery is `r_extrap = 1 - A/g^2 - c/tau0^2` evaluated at
    (g, tau0) -> (inf, inf), and the *test* of the claim is the fit residual
    plus the independent cross-check below.
    """
    rows = list(rows_a) + list(rows_b)
    y = np.array([1.0 - r["ratio"] for r in rows])
    X = np.array([[1.0 / r["g"] ** 2, 1.0 / r["tau0"] ** 2] for r in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    A, c = float(coef[0]), float(coef[1])
    pred = X @ coef
    resid = y - pred
    # relative residual of the two-artifact model, per point
    rel = float(np.max(np.abs(resid) / np.maximum(y, 1e-16)))
    # free-offset variant: does the data want a constant offset (= a wrong
    # continuum constant)? If b0 is consistent with 0, the sharp constant is
    # recovered with no fudge.
    X0 = np.hstack([np.ones((len(rows), 1)), X])
    coef0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    b0 = float(coef0[0])
    dof = len(rows) - 3
    s2 = float(np.sum((y - X0 @ coef0) ** 2)) / max(dof, 1)
    cov = s2 * np.linalg.inv(X0.T @ X0)
    b0_err = float(np.sqrt(cov[0, 0]))
    # robustness: does the residual offset survive the next artifact order?
    # 1 - r = b0 + A/g^2 + c/tau0^2 + d/tau0^4  (an O(a^4) dispersion term)
    X4 = np.hstack([X0, (X[:, 1] ** 2)[:, None]])
    coef4, *_ = np.linalg.lstsq(X4, y, rcond=None)
    dof4 = len(rows) - 4
    s24 = float(np.sum((y - X4 @ coef4) ** 2)) / max(dof4, 1)
    cov4 = s24 * np.linalg.inv(X4.T @ X4)
    b04, b04_err = float(coef4[0]), float(np.sqrt(cov4[0, 0]))
    out = {"_source": "notebook.py::s2_two_parameter_fit",
           "model": "1 - ratio = A/g^2 + c/tau0^2",
           "A_box": A, "c_lattice": c, "n_points": len(rows),
           "max_rel_residual": rel,
           "max_abs_residual": float(np.max(np.abs(resid))),
           "free_offset_b0": b0, "free_offset_b0_stderr": b0_err,
           "free_offset_b0_over_stderr": b0 / b0_err if b0_err > 0 else float("nan"),
           "r_extrap_continuum": 1.0 - b0,
           "recovery_error_vs_sharp": abs(b0),
           "quartic_model_b0": b04, "quartic_model_b0_stderr": b04_err,
           "quartic_model_b0_over_stderr":
               b04 / b04_err if b04_err > 0 else float("nan"),
           "quartic_model_d": float(coef4[3]),
           "quartic_model_r_extrap": 1.0 - b04,
           "quartic_model_recovery_error": abs(b04)}
    _write_json("s2_two_parameter_fit.json", out)
    return out


def s2_single_axis_richardson(rows_a):
    """The test's own estimator, at higher resolution: r_inf from a 1-axis
    Richardson fit r = r_inf + c1 h + c2 h^2 in h = 1/tau0^2, at fixed g."""
    taus = np.array([r["tau0"] for r in rows_a])
    ratios = np.array([r["ratio"] for r in rows_a])
    h = 1.0 / taus**2
    M = np.vstack([np.ones_like(h), h, h * h]).T
    coef, *_ = np.linalg.lstsq(M, ratios, rcond=None)
    r_inf = float(coef[0])
    resid = float(np.max(np.abs(M @ coef - ratios)))
    return {"r_inf": r_inf, "residual": resid, "g": rows_a[0]["g"],
            "taus": taus.tolist(), "ratios": ratios.tolist()}


def s2c_mass_axis(tau0=8.0, mus=(0.08, 0.04, 0.02, 0.01, 0.005), g=100.0):
    """m*tau0 axis: r(mu) -> r(0), the second leg of the joint (a, m) limit."""
    rows = [_gaussian_point(tau0, g=g, mu=mu) for mu in mus]
    r0 = _gaussian_point(tau0, g=g, mu=0.0)
    rv = np.array([r["ratio"] for r in rows])
    # geometric Richardson on the last three (halving mu)
    d1, d2 = rv[-2] - rv[-3], rv[-1] - rv[-2]
    p = math.log2(d1 / d2) if d2 != 0 else float("nan")
    r0_extrap = float(rv[-1] + d2 / (2.0**p - 1.0)) if np.isfinite(p) else float("nan")
    out = {"rows": rows, "r_direct_m0": r0["ratio"], "exponent_p": float(p),
           "r_extrap_mu0": r0_extrap,
           "dev_extrap_vs_direct": abs(r0_extrap - r0["ratio"])}
    _write_csv("s2c_mass_axis.csv", rows + [dict(r0, mu=0.0)],
               "notebook.py::s2c_mass_axis")
    _write_json("s2c_mass_axis_fit.json",
                dict(out, _source="notebook.py::s2c_mass_axis", rows=None))
    return out


# --- S2d: four families, each with its own (scale, wall) fit ---------------

FAMILIES = {
    "gaussian": (gaussian_f, (4.0, 6.0, 8.0, 10.0, 13.0, 16.0), 4.5),
    "cos2": (cos2_f, (8.0, 12.0, 16.0, 24.0, 32.0, 40.0), 1.0),
    "bump4": (bump4_f, (8.0, 12.0, 16.0, 24.0, 32.0, 40.0), 1.0),
    "sech": (sech_f, (2.0, 3.0, 4.0, 5.0, 6.0, 8.0), 14.0),
}
#: value = (constructor, scale values s, t_max / s for that family)


def _family_point(name, s, wall_ratio):
    """One family point at wall distance D = wall_ratio * t_max."""
    ctor, _, tm_over_s = FAMILIES[name]
    fh = ctor(s)
    t_max = tm_over_s * s
    N = _odd(2 * int(math.ceil(wall_ratio * t_max)) + 1)
    r, res, op = _ratio_at(fh, N)
    return {"family": name, "scale": float(s), "wall_ratio": float(wall_ratio),
            "N": N, "t_max": float(t_max), "n_support": int(op.support.size),
            "e_min": float(res.e_min), "bound": float(qei_bound_2d(fh)),
            "ratio": float(r), "gap_rel": float(res.gap / abs(res.e_min))}


def s2d_families(wall_ratio=12.0, box_scan_walls=(6.0, 9.0, 12.0, 18.0)):
    """Every family climbs to the SAME sharp constant.

    For each family, fit  1 - r = A/w^2 + c/s^2  (w = wall distance / t_max,
    s = the family's scale parameter) over a scale sweep at fixed w plus a
    wall sweep at one mid scale. The continuum/infinite-box value of that
    model is again r = 1; the deliverable per family is the free-offset
    b0 (should be 0 to within its own standard error).
    """
    rows = []
    for name, (_, scales, _tm) in FAMILIES.items():
        for s in scales:
            rows.append(_family_point(name, s, wall_ratio))
        mid = scales[len(scales) // 2 - 1]
        for w in box_scan_walls:
            if w == wall_ratio:
                continue
            rows.append(_family_point(name, mid, w))
    _write_csv("s2d_families.csv", rows, "notebook.py::s2d_families")

    fits = {}
    for name in FAMILIES:
        sub = [r for r in rows if r["family"] == name]
        y = np.array([1.0 - r["ratio"] for r in sub])
        X = np.array([[1.0, 1.0 / r["wall_ratio"] ** 2, 1.0 / r["scale"] ** 2]
                      for r in sub])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        dof = len(sub) - 3
        s2 = float(np.sum((y - X @ coef) ** 2)) / max(dof, 1)
        cov = s2 * np.linalg.inv(X.T @ X)
        fits[name] = {"b0": float(coef[0]), "b0_stderr": float(np.sqrt(cov[0, 0])),
                      "A_wall": float(coef[1]), "c_scale": float(coef[2]),
                      "n_points": len(sub),
                      "r_inf": 1.0 - float(coef[0]),
                      "max_abs_residual": float(np.max(np.abs(y - X @ coef)))}
    _write_json("s2d_family_fits.json",
                dict(fits, _source="notebook.py::s2d_families"))
    return rows, fits


def s2e_optimize(N=241, site=120):
    """optimize_sampling on the 2-parameter modulated-Gaussian family.

    The lattice ratio has an interior maximum in (tau, c): wider f suppresses
    lattice-dispersion error but buys finite-box bias, and c != 0 trades
    curvature between the operator and the bound. The optimizer uses the
    analytic envelope (Hellmann-Feynman) gradient of vacuum.inequalities.
    """
    fam = modulated_gaussian_family()
    theta0 = np.array([4.0, 0.0])
    out = optimize_sampling(N, site, fam, theta0, bounds=[(3.0, 9.0), (-0.4, 0.9)],
                            m=0.0, bc="dirichlet", pad=8)
    rec = {"_source": "notebook.py::s2e_optimize", "N": N, "site": site,
           "theta0": theta0.tolist(), "theta_star": out.theta.tolist(),
           "ratio_star": float(out.ratio), "e_min": float(out.e_min),
           "bound": float(out.bound), "n_iter": int(out.n_iter),
           "converged": bool(out.converged), "message": out.message,
           "label": out.f.label}
    # plain re-run of the optimized handle (papers/README.md corollary:
    # optimizer output must survive a plain-numpy round trip)
    fh = fam.make(out.theta)
    r_replay, res, op = _ratio_at(fh, N)
    rec["ratio_replay_independent_grid"] = float(r_replay)
    rec["replay_dev"] = float(abs(r_replay - out.ratio))
    _write_json("s2e_optimize.json", rec)
    return rec


# =========================================================================
# S3. the extremal states, exhibited
# =========================================================================


def _gl_grid(t_lo, t_hi, n_panels, nodes_per_panel):
    """Composite Gauss-Legendre nodes/weights (same rule the module uses)."""
    from numpy.polynomial.legendre import leggauss

    x, w = leggauss(int(nodes_per_panel))
    edges = np.linspace(float(t_lo), float(t_hi), int(n_panels) + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    return ((mid[:, None] + half[:, None] * x[None, :]).ravel(),
            (half[:, None] * w[None, :]).ravel())


def _low_rank(h, tol=1e-12):
    """Exact low-rank factorization h = sum_k lam_k v_k v_k^T of the local
    density, exploiting that h touches only ~4 quadrature rows."""
    rows = np.flatnonzero(np.max(np.abs(h), axis=1) > tol)
    sub = h[np.ix_(rows, rows)]
    lam, W = np.linalg.eigh(sub)
    keep = np.flatnonzero(np.abs(lam) > tol * max(1.0, float(np.max(np.abs(lam)))))
    V = np.zeros((h.shape[0], keep.size))
    V[rows, :] = W[:, keep]
    return lam[keep], V


def _pullback_profile(K, h, dV, ts):
    """<:h(t):> = (1/2) Tr( S(t)^T h S(t) dV ) on a whole time grid at once.

    Uses ONE eigh(K) and the exact normal-mode form of the propagator
    (identical to vacuum.core.chain_propagator's 'spectral' branch, which is
    verified against it below) instead of one propagator build per time —
    4000 separate eigh(K) calls is the difference between seconds and hours.
    Pure function; nothing is mutated.
    """
    K = np.asarray(K, dtype=float)
    w2, U = np.linalg.eigh(0.5 * (K + K.T))
    w = np.sqrt(w2)
    N = K.shape[0]
    lam, Vecs = _low_rank(h)
    ts = np.asarray(ts, dtype=float)
    c = np.cos(np.outer(w, ts))          # (N, nt)
    s = np.sin(np.outer(w, ts))
    out = np.zeros(ts.size)
    for k in range(lam.size):
        ahat = U.T @ Vecs[:N, k]
        bhat = U.T @ Vecs[N:, k]
        # S(t)^T v = ( U [cos ahat - w sin bhat] ; U [sin/w ahat + cos bhat] )
        top = U @ (c * ahat[:, None] - (s * w[:, None]) * bhat[:, None])
        bot = U @ ((s / w[:, None]) * ahat[:, None] + c * bhat[:, None])
        P = np.concatenate([top, bot], axis=0)          # (2N, nt)
        out += lam[k] * np.einsum("in,in->n", P, dV @ P)
    return 0.5 * out


def _energy_profiles(res, op, K, m, bc, ts):
    """Space and time profiles of the extremal state's energy density.

    Space: <:h_j:> at t = 0 for every site j in the support.
    Time:  <:h_site(t):> = (1/2) Tr( S(t)^T h_site S(t) (V - V_vac) ).
    Both are normal-ordered (vacuum subtracted) and pure functions of the
    inputs; the time profile pulls back with exactly the S(t) the sampling
    operator uses (cross-checked against vacuum.core.chain_propagator).
    """
    V_full = embed_extremal(res, op)
    V_vac = ground_state_cov(K)
    dV = V_full - V_vac
    N = op.N
    space = np.array([
        0.5 * float(np.einsum("ij,ji->", chain_energy_density_form(N, int(j), m=m, bc=bc), dV))
        for j in op.support
    ])
    h_site = chain_energy_density_form(N, op.site, m=m, bc=bc)
    time_prof = _pullback_profile(K, h_site, dV, ts)
    return space, time_prof, V_full, h_site, dV


def s3_extremal_states():
    """Exhibit and archive the optimizing states for three setups."""
    setups = [
        ("gaussian_tau8", gaussian_f(8.0), _odd(round(100 * 8.0)), 0.0),
        ("cos2_T24", cos2_f(24.0), _odd(2 * int(math.ceil(12.0 * 24.0)) + 1), 0.0),
        ("bump4_T24", bump4_f(24.0), _odd(2 * int(math.ceil(12.0 * 24.0)) + 1), 0.0),
    ]
    summary = []
    for name, fh, N, m in setups:
        site = N // 2
        K = harmonic_chain_K(N, m, "dirichlet")
        op = sampling_operator(N, site, fh, m=m, bc="dirichlet")
        res = qei_minimize(op)
        nu_star = symplectic_eigenvalues(res.V_star)
        ts = np.linspace(fh.t_lo, fh.t_hi, 161)
        space, time_prof, V_full, h_site, dV = _energy_profiles(
            res, op, K, m, "dirichlet", ts)
        # Two independent closure checks on the exhibited state.
        # (a) the truncated operator: smeared_energy on the FULL-lattice
        #     embedding must reproduce `achieved` to roundoff (it reads only
        #     the support block, by construction).
        e_trunc = smeared_energy(V_full, op)
        # (b) the UNtruncated quantity: integrate the time-resolved density
        #     <:h_site(t):> of the embedded state against f^2 over the whole
        #     lattice. The gap between (a) and (b) is the mode-support
        #     truncation error of the whole construction, measured not assumed.
        tq, wq = _gl_grid(fh.t_lo, fh.t_hi, 120, 12)
        tp_fine = _pullback_profile(K, h_site, dV, tq)
        integ = float(np.sum(wq * fh.f(tq) ** 2 * tp_fine))
        # and the local propagator agrees with vacuum.core.chain_propagator
        t_probe = 0.37 * fh.t_hi
        S = chain_propagator(K, float(t_probe))
        ref = 0.5 * float(np.einsum("ij,ji->", S.T @ h_site @ S, dV))
        prop_dev = abs(ref - float(_pullback_profile(K, h_site, dV, [t_probe])[0]))
        rec = {"name": name, "label": fh.label, "N": N, "site": site,
               "n_support": int(op.support.size),
               "e_min": float(res.e_min), "achieved": float(res.achieved),
               "gap": float(res.gap), "gap_rel": float(res.gap / abs(res.e_min)),
               "bound_sharp": float(qei_bound_2d(fh)),
               "ratio": float(res.e_min / qei_bound_2d(fh)),
               "purity_dev_max": float(np.max(np.abs(nu_star - 0.5))),
               "margin_star": float(res.margin_star),
               "cond": float(res.cond),
               "sigma_min": float(res.sigma.min()), "sigma_max": float(res.sigma.max()),
               "t0_energy_density_at_site": float(space[np.searchsorted(op.support, site)]),
               "min_time_profile": float(np.min(time_prof)),
               "truncated_operator_dev": abs(e_trunc - res.achieved),
               "untruncated_time_integral": integ,
               "support_truncation_rel_dev":
                   abs(integ - res.achieved) / abs(res.achieved),
               "propagator_crosscheck_dev": prop_dev}
        summary.append(rec)
        np.savez_compressed(
            DATA / f"s3_extremal_{name}.npz",
            _source=np.array("notebook.py::s3_extremal_states"),
            V_star=res.V_star, sigma_Of=res.sigma, nu_V_star=nu_star,
            support=op.support, O_f=op.O, V_vac_sub=op.V_vac_sub,
            space_sites=op.support, space_energy_density=space,
            time_grid=ts, time_energy_density=time_prof,
            meta=np.array(json.dumps(rec)),
        )
    _write_csv("s3_extremal_summary.csv", summary,
               "notebook.py::s3_extremal_states")
    return summary


# =========================================================================
# S4. the 4d Srednicki-lattice scan with the cancellation-free minimizer
# =========================================================================
#
# What is computed, stated precisely, because the normalization is the whole
# difficulty:
#
#   E_min(R, tau) = inf over physical states of
#                     int_{ball} d^3x  int dt f(t)^2 <:T_00(t, x):>,
#
# sector by sector on the Srednicki radial lattice (PRL 71, 666 (1993),
# Eq. 10): the (l, m) sectors decouple, so the global infimum is the
# degeneracy-weighted sum  sum_l (2l+1) E_min^{(l)}  of independent per-sector
# minima, exactly as `radial_ball_qei` does it.
#
# The Fewster-Eveson 4d worldline bound (PRD 58, 084010 (1998), Eq. (40), in
# the f^2 convention) applies to EACH inertial worldline separately,
#
#   int dt f^2 <:T_00(t, x):>  >=  -(1/(16 pi^2)) int f''^2 dt,
#
# so integrating it over the ball gives a rigorous inequality for our
# quantity and makes
#
#   C_eff = -E_min / (V_ball * int f''^2 dt),     ratio = C_eff / C_FE  <=  1
#
# a well-defined diagnostic. In the continuum C_eff is a function of R/tau
# alone (dimensional analysis), and the worldline constant is its R/tau -> 0
# limit: C_eff(R/tau) <= C_4d with equality at 0, because the ball infimum
# needs every worldline at its own optimum simultaneously.
#
# The first version of this scan (archived in git history) had three
# systematics that together left ~10% and forced an "exploratory" label:
#  (1) a float64 NOISE FLOOR in the per-sector minimum (1/2)sum sigma -
#      (1/2)Tr(O V_vac), growing with l while the physics fell off, so the
#      l-sum had to be cut at a "knee";
#  (2) the ball VOLUME as an unstated lattice-measure convention (5-7%);
#  (3) the a/tau -> 0 and R/tau -> 0 limits not separated.
# Each is closed below: (1) by `radial_sector_minimize` (the mode-space
# Riccati form of the same variational problem, documented in
# vacuum/inequalities/qei.py — S4a shows the floor gone, S4b certifies it
# against an mpmath Williamson computation); (2) by deriving the measure from
# Srednicki's Eq. 10 (S4c: sites carry continuum shell volumes) and by
# taking the exact Fourier transform of f^2 instead of a 4.5-sigma window
# (whose edge gives the high-l tail an algebraic instead of super-exponential
# fall-off — a fourth, previously unnoticed artifact, measured in S4a);
# (3) by a product grid in (n_ball, R/tau) whose rows are fixed-R/tau
# sequences (a/tau -> 0 first, then R/tau -> 0: "order 1") and whose columns
# are fixed-n_ball sequences (R/tau -> 0 first, then a/R -> 0: "order 2"),
# so the two orders of the double limit are extrapolated independently and
# compared (S4e).


def _fpp2_gaussian(tau):
    """int f''(t)^2 dt for f = exp(-t^2/2 tau^2): (3 sqrt(pi)) / (4 tau^3)."""
    return 3.0 * math.sqrt(math.pi) / (4.0 * tau**3)


BOX_RATIO = 16.0  #: g = N/tau of the scan grid (box bias -10/g^4, S4d sweep)


def _lattice_N(n_ball, tau, g=BOX_RATIO, pad=8):
    """Radial sites: ball + g tau + pad. The wall's causal reach is irrelevant
    (f^2 at t = 6 tau is e^{-36}); what the box size controls is the INFRARED
    discretization of the sector spectrum, a bias measured in
    `s4d_box_sweep` to be -10/g^4 in C/C_FE (1.7% at g = 6, 0.03% at 16)."""
    return int(n_ball + math.ceil(g * tau) + pad)


def _ball_point(n_ball, tau, N=None, l_max=40, rel_tol=1e-12):
    """One (n_ball, tau) point with the cancellation-free l-sum; both volume
    measures; per-sector diagnostics."""
    N = _lattice_N(n_ball, tau) if N is None else int(N)
    f = gaussian_f(tau)
    t0 = time.time()
    out = radial_ball_qei(N, n_ball, f, l_max=l_max, tol=0.0, method="mode_space", rel_tol=rel_tol)
    secs = time.time() - t0
    total = out["e_min_total"]
    fpp2 = _fpp2_gaussian(tau)
    Vs, Vn = shell_volume(n_ball), naive_volume(n_ball)
    res = out["results"]
    rec = {"n_ball": int(n_ball), "tau": float(tau), "N": N,
           "rho": (n_ball + 0.5) / tau, "x": 1.0 / tau,
           "e_min_total": float(total), "l_stop": int(out["l_stop"]),
           "converged": bool(out["converged"]),
           "tail_frac": float(abs(out["terms"][-1]) / abs(total)),
           "C_shell": -total / (Vs * fpp2), "C_naive": -total / (Vn * fpp2),
           "ratio_shell_FE": -total / (Vs * fpp2) / FE_CONSTANT_4D,
           "ratio_naive_FE": -total / (Vn * fpp2) / FE_CONSTANT_4D,
           "V_shell": Vs, "V_naive": Vn, "fpp2": fpp2,
           "sum_e_dropped": float(sum((2 * l + 1) * r.e_dropped for l, r in enumerate(res))),
           "max_residual": float(max(r.residual for r in res)),
           "max_z_norm": float(max(r.z_norm for r in res)),
           "n_sectors_regularized": int(sum(1 for r in res if r.meta["state_reg"] > 0)),
           "seconds": secs}
    diag = {"terms": out["terms"],
            "e_min_l": np.array([r.e_min for r in res]),
            "e_vac_l": np.array([r.e_vac for r in res]),
            "e_dropped_l": np.array([r.e_dropped for r in res]),
            "n_kept_l": np.array([r.n_kept for r in res]),
            "z_norm_l": np.array([r.z_norm for r in res]),
            "residual_l": np.array([r.residual for r in res])}
    return rec, diag, out


def s4a_noise_floor_revisited(N=200, n_ball=3, tau=9.0, l_max=12):
    """The archived instability configuration, re-run with both minimizers.

    Per sector: the Williamson route on its cone support (as archived), the
    mode-space route on the same 4.5-sigma window (same time integral, so
    the two agree to roundoff at low l), the mode-space route with the exact
    Fourier transform (the continuum smearing), and the spread of the latter
    under the mode-floor knob (1e-13 .. 1e-15 of the peak sampling weight),
    which measures the only floor the new route has: sampling weights below
    ~eps x peak are unresolvable.
    """
    w = radial_ball_weights(N, n_ball)
    f = gaussian_f(tau)
    rows = []
    for l in range(int(l_max) + 1):
        op = radial_sampling_operator(N, l, w, f)
        r_old = qei_minimize(op)
        r_full = qei_minimize(radial_sampling_operator(N, l, w, f, support=np.arange(N),
                                                       weight_floor=0.0))
        m_q = radial_sector_minimize(N, l, w, f, fhat="quadrature")
        m_a = radial_sector_minimize(N, l, w, f, fhat="analytic")
        fl = [radial_sector_minimize(N, l, w, f, fhat="analytic", mode_floor=mf).e_min
              for mf in (1e-13, 1e-15)]
        rows.append({"l": l, "e_old_cone": float(r_old.e_min),
                     "e_old_full": float(r_full.e_min),
                     "trace_vac_old": float(r_old.trace_vac),
                     "n_support_old": int(op.support.size),
                     "e_new_window": float(m_q.e_min), "e_new": float(m_a.e_min),
                     "e_vac_new": float(m_a.e_vac), "e_dropped": float(m_a.e_dropped),
                     "n_kept": int(m_a.n_kept), "residual": float(m_a.residual),
                     "z_norm": float(m_a.z_norm),
                     "floor_spread": float(max(abs(x - m_a.e_min) for x in fl)),
                     "term_old": (2 * l + 1) * float(r_old.e_min),
                     "term_new": (2 * l + 1) * float(m_a.e_min)})
    tail = [r for r in rows if r["l"] >= 6]
    old_floor = float(np.median([abs(r["e_old_cone"]) for r in tail]))
    old_full_floor = float(np.median([abs(r["e_old_full"] - r["e_new_window"]) for r in tail]))
    new_floor = float(np.max([r["floor_spread"] for r in rows]))   # conservative: worst sector
    rec = {"_source": "notebook.py::s4a_noise_floor_revisited",
           "N": N, "n_ball": n_ball, "tau": tau, "l_max": l_max,
           "old_route_floor_abs": old_floor,
           "old_route_floor_rel_to_trace": old_floor / float(np.median([r["trace_vac_old"] for r in tail])),
           "old_route_full_support_floor_abs": old_full_floor,
           "new_route_floor_abs_worst_sector": new_floor,
           "digits_recovered_per_sector": math.log10(old_floor / max(new_floor, 1e-300)),
           "digits_recovered_vs_full_support_float64": math.log10(old_full_floor / max(new_floor, 1e-300)),
           "window_vs_exact_rel_dev_l": [abs(r["e_new_window"] - r["e_new"]) / abs(r["e_new"]) for r in rows],
           "total_old_to_lmax": float(sum(r["term_old"] for r in rows)),
           "total_new_to_lmax": float(sum(r["term_new"] for r in rows)),
           "rows": rows}
    _write_json("s4a_noise_floor_revisited.json", rec)
    return rec


def s4b_mp_crosscheck(N=64, n_ball=3, tau=9.0, ls=(2, 3, 4), dps=30):
    """Precision-escalated cross-check through vacuum.core.precision.

    Untruncated operator (support = all N sites) so that all three routes
    solve the identical problem: the float64 Williamson route, the
    mode-space route on the same window, and `qei_minimize_mp` (mpmath
    i*Omega*O eigenproblem + mp-native vacuum covariance, exact for the
    operator as assembled). The cross-check's own limit is the float64
    rounding of O_f, whose first-order effect on the minimum is
    (1/2)Tr(dO (V* - V_vac)); an order-of-magnitude bound is recorded so
    the "digits recovered" never claims more than the check can certify.
    """
    w = radial_ball_weights(N, n_ball)
    f = gaussian_f(tau)
    rows = []
    for l in ls:
        K = srednicki_K(N, l)
        op = radial_sampling_operator(N, l, w, f, support=np.arange(N), weight_floor=0.0)
        r_old = qei_minimize(op)
        m = radial_sector_minimize(N, l, w, f, fhat="quadrature")
        t0 = time.time()
        mpr = qei_minimize_mp(op, K, dps=dps)
        secs = time.time() - t0
        V, info = mode_space_extremal_state(m)
        dV = V - ground_state_cov(K)
        sens = 0.5 * 2.2e-16 * float(np.linalg.norm(op.O)) * float(np.linalg.norm(dV))
        err_old = abs(r_old.e_min - mpr["e_min"])
        err_new = abs(m.e_min - mpr["e_min"])
        rows.append({"l": l, "dim": mpr["dim"], "e_old": float(r_old.e_min),
                     "e_new": float(m.e_min), "e_mp": mpr["e_min"],
                     "abs_err_old": err_old, "abs_err_new": err_new,
                     "rel_err_old": err_old / abs(mpr["e_min"]),
                     "rel_err_new": err_new / abs(mpr["e_min"]),
                     "mp_sensitivity_bound": sens,
                     "digits_recovered_certified":
                         math.log10(err_old / max(err_new, sens, 1e-300)),
                     "seconds_mp": secs})
    rec = {"_source": "notebook.py::s4b_mp_crosscheck", "N": N, "n_ball": n_ball,
           "tau": tau, "dps": dps, "rows": rows}
    _write_json("s4b_mp_crosscheck.json", rec)
    return rec


def s4c_volume_measure(n_max=8):
    """The lattice measure of the ball, derived (docstring of shell_volume).

    Srednicki's Eq. 10 discretizes int dr of the radial density of
    phi_lm = r x (angular part): the r^2 of d^3x is already absorbed, the
    site-j term is the rectangle rule on (j-1/2)a < r < (j+1/2)a, and after
    the (l, m) sum it is the energy in that shell, of volume
    4 pi a^3 (j^2 + 1/12). The ball operator of sites 1..n is therefore the
    energy in a/2 < r < (n+1/2)a, whose continuum volume is V_shell; the
    naive (4 pi/3)(n a)^3 differs by the exactly known 1 - 3/(2n) + O(1/n^2).
    """
    rows = []
    for n in range(1, n_max + 1):
        Vs, Vn = shell_volume(n), naive_volume(n)
        rows.append({"n_ball": n, "V_shell": Vs, "V_naive": Vn,
                     "V_sites_sum": sum(4 * math.pi * (j * j + 1 / 12) for j in range(1, n + 1)),
                     "naive_over_shell_minus_1": Vn / Vs - 1.0,
                     "leading_minus_3_over_2n": -1.5 / n,
                     "remainder": Vn / Vs - 1.0 + 1.5 / n,
                     "R_outer_over_a": n + 0.5})
    _write_csv("s4c_volume_measure.csv", rows, "notebook.py::s4c_volume_measure")
    return rows


def s4d_box_sweep(points=((4, 0.5), (4, 0.25), (2, 0.25), (6, 0.25), (4, 0.125)),
                  gs=(6, 8, 10, 14, 20, 30)):
    """The finite-box (infrared) bias of the radial lattice, measured.

    At fixed (n_ball, tau) the ratio C/C_FE depends on the number of radial
    sites through the discretization of the sector spectrum below 1/tau
    (the 4d counterpart of the 2d Dirichlet bias A/g^2 of section 3). Sweep
    g = N/tau at five (n_ball, rho) points and fit C = C_inf + A/g^2 + B/g^4;
    the data want A ~ 0 and a universal B ~ -10, so the grid is run at
    g = BOX_RATIO and corrected by -B/g^4 (0.03% there), with the spread of
    B across the points as the correction's uncertainty.
    """
    rows, Bs = [], []
    for n, rho in points:
        tau = (n + 0.5) / rho
        cs = []
        for g in gs:
            rec, _, _ = _ball_point(n, tau, N=int(n + math.ceil(g * tau) + 8))
            cs.append((rec["N"] / tau, rec["ratio_shell_FE"], rec["N"]))
        gg = np.array([c[0] for c in cs])
        cc = np.array([c[1] for c in cs])
        X4 = np.vstack([np.ones_like(gg), 1 / gg**2, 1 / gg**4]).T
        c4, r4, s4 = _lsq(X4, cc)
        Xq = np.vstack([np.ones_like(gg), 1 / gg**4]).T
        cq, rq, sq = _lsq(Xq, cc)
        X2 = np.vstack([np.ones_like(gg), 1 / gg**2]).T
        c2, r2, s2 = _lsq(X2, cc)
        Bs.append(float(cq[1]))
        rows.append({"n_ball": n, "rho": rho, "tau": tau, "gs": gg.tolist(), "Ns": [c[2] for c in cs],
                     "C_over_FE": cc.tolist(),
                     "fit_g2_g4": {"C_inf": float(c4[0]), "A": float(c4[1]), "B": float(c4[2]),
                                   "max_resid": float(np.max(np.abs(r4)))},
                     "fit_g4": {"C_inf": float(cq[0]), "B": float(cq[1]),
                                "max_resid": float(np.max(np.abs(rq)))},
                     "fit_g2": {"C_inf": float(c2[0]), "A": float(c2[1]),
                                "max_resid": float(np.max(np.abs(r2)))}})
    rec = {"_source": "notebook.py::s4d_box_sweep", "rows": rows,
           "B_mean": float(np.mean(Bs)), "B_spread": float(np.max(Bs) - np.min(Bs)),
           "grid_box_ratio": BOX_RATIO,
           "grid_correction_typical": -float(np.mean(Bs)) / BOX_RATIO**4,
           "grid_correction_uncertainty": float(np.max(Bs) - np.min(Bs)) / BOX_RATIO**4}
    _write_json("s4d_box_sweep.json", rec)
    return rec


def s4d_scan(box, rhos=(0.5, 0.4, 1.0 / 3.0, 0.25, 1.0 / 6.0, 0.125),
             n_balls=(2, 3, 4, 5, 6, 8)):
    """The product grid: rho = R/tau = (n_ball + 1/2)/tau (R the outer radius
    of the shell region, S4c) times n_ball; tau = (n_ball + 1/2)/rho is a
    real parameter, the lattice spacing is 1. Rows (fixed rho) are the
    a/tau -> 0 sequences, columns (fixed n_ball) the R/tau -> 0 sequences.
    Every point is run at g = N/tau = BOX_RATIO and corrected by the measured
    box bias -B/g^4 (`box` from s4d_box_sweep); the raw value is kept.
    Plus a wall-independence check on one point (g = 16 vs 24)."""
    B = box["B_mean"]
    rows = []
    for rho in rhos:
        for n in n_balls:
            tau = (n + 0.5) / rho
            rec, diag, out = _ball_point(n, tau)
            rec["rho_nominal"] = float(rho)
            g = rec["N"] / tau
            rec["box_ratio"] = g
            rec["ratio_shell_FE_raw"] = rec["ratio_shell_FE"]
            rec["ratio_naive_FE_raw"] = rec["ratio_naive_FE"]
            rec["box_correction"] = -B / g**4
            rec["ratio_shell_FE"] = rec["ratio_shell_FE_raw"] - B / g**4
            rec["ratio_naive_FE"] = rec["ratio_naive_FE_raw"] * rec["ratio_shell_FE"] / rec["ratio_shell_FE_raw"]
            rows.append(rec)
            np.savez_compressed(
                DATA / f"s4d_lterms_rho{rho:.4f}_nb{n}.npz",
                _source=np.array("notebook.py::s4d_scan"),
                meta=np.array(json.dumps(rec)), **diag)
    _write_csv("s4d_4d_scan.csv", rows, "notebook.py::s4d_scan")
    n, tau = 4, (4 + 0.5) / 0.25
    a, _, _ = _ball_point(n, tau)
    b, _, _ = _ball_point(n, tau, N=int(n + math.ceil(1.5 * BOX_RATIO * tau) + 8))
    wall = {"_source": "notebook.py::s4d_scan", "n_ball": n, "tau": tau,
            "N": a["N"], "N_big": b["N"], "C_shell": a["C_shell"], "C_shell_big": b["C_shell"],
            "rel_dev_raw": abs(a["C_shell"] - b["C_shell"]) / abs(a["C_shell"]),
            "rel_dev_predicted_by_g4_law": abs(B) * (1 / (a["N"] / tau)**4 - 1 / (b["N"] / tau)**4) / a["ratio_shell_FE"]}
    _write_json("s4d_wall_check.json", wall)
    return rows, wall


def _lsq(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(y) - X.shape[1], 1)
    cov = float(resid @ resid) / dof * np.linalg.inv(X.T @ X)
    return coef, resid, np.sqrt(np.diag(cov))


def s4e_extrapolation(rows, box_unc=0.0):
    """Both orders of the double limit, with model dependence.

    Order 1 (rows of the grid): at fixed rho fit C(x) = C_rho + c1 x + c2 x^2
    in x = a/tau (the O(a) and O(a^2) lattice corrections; note that at fixed
    rho the O(1/n^2) measure remainder is O(x^2)), then C_rho -> C_4d as
    rho -> 0 with C_rho = C_4d + d2 rho^2 (model A), + d3 rho^3 (B),
    + d1 rho (C). Order 2 (columns): at fixed n_ball fit C(rho) = C_n + b1 rho
    + b2 rho^2, then C_n = C_4d + e2/(n+1/2)^2 (A'), + e1/(n+1/2) (B').
    The naive measure is extrapolated the same way (order 1) to show it
    converges to the shell result as the fit order grows.
    """
    rhos = sorted(set(r["rho_nominal"] for r in rows), reverse=True)
    ns = sorted(set(r["n_ball"] for r in rows))
    out = {"_source": "notebook.py::s4e_extrapolation"}

    # ---- order 1 ----
    o1 = []
    for rho in rhos:
        sub = sorted([r for r in rows if r["rho_nominal"] == rho], key=lambda r: r["x"])
        x = np.array([r["x"] for r in sub])
        y = np.array([r["ratio_shell_FE"] for r in sub])
        yn = np.array([r["ratio_naive_FE"] for r in sub])
        X2 = np.vstack([np.ones_like(x), x, x * x]).T
        X1 = np.vstack([np.ones_like(x), x]).T
        X3 = np.vstack([np.ones_like(x), x, x * x, x**3]).T
        c2, r2, s2 = _lsq(X2, y)
        c1, r1, s1 = _lsq(X1, y)
        c3, r3, s3 = _lsq(X3, y)
        cn1, rn1, sn1 = _lsq(X1, yn)
        cn2, rn2, sn2 = _lsq(X2, yn)
        cn3, rn3, sn3 = _lsq(X3, yn)
        o1.append({"rho": rho, "n_points": len(sub),
                   "C_rho": float(c2[0]), "C_rho_stderr": float(s2[0]),
                   "c1": float(c2[1]), "c2": float(c2[2]),
                   "max_resid": float(np.max(np.abs(r2))),
                   "C_rho_linear": float(c1[0]), "max_resid_linear": float(np.max(np.abs(r1))),
                   "C_rho_cubic": float(c3[0]), "max_resid_cubic": float(np.max(np.abs(r3))),
                   "C_rho_naive_linear": float(cn1[0]), "C_rho_naive_quadratic": float(cn2[0]),
                   "C_rho_naive_cubic": float(cn3[0]),
                   "naive_minus_shell_linear": float(cn1[0] - c2[0]),
                   "naive_minus_shell_quadratic": float(cn2[0] - c2[0]),
                   "naive_minus_shell_cubic": float(cn3[0] - c3[0])})
    rr = np.array([o["rho"] for o in o1])
    cc = np.array([o["C_rho"] for o in o1])
    models = {"A_rho2": np.vstack([np.ones_like(rr), rr**2]).T,
              "B_rho2_rho3": np.vstack([np.ones_like(rr), rr**2, rr**3]).T,
              "C_rho_rho2": np.vstack([np.ones_like(rr), rr, rr**2]).T,
              "D_rho2_rho4": np.vstack([np.ones_like(rr), rr**2, rr**4]).T}
    fits1 = {}
    for name, X in models.items():
        c, r, s = _lsq(X, cc)
        fits1[name] = {"C_4d_over_FE": float(c[0]), "stderr": float(s[0]),
                       "coef": c.tolist(), "max_resid": float(np.max(np.abs(r)))}
    # same models on the subset rho <= 1/3 (small-rho robustness)
    small = rr <= 1.0 / 3.0 + 1e-12
    Xs = np.vstack([np.ones(small.sum()), rr[small]**2]).T
    c, r, s = _lsq(Xs, cc[small])
    fits1["A_rho2_small_rho"] = {"C_4d_over_FE": float(c[0]), "stderr": float(s[0]),
                                 "coef": c.tolist(), "max_resid": float(np.max(np.abs(r))),
                                 "rhos": rr[small].tolist()}
    out["order1_rows"] = o1
    out["order1_fits"] = fits1

    # ---- order 2 ----
    o2 = []
    for n in ns:
        sub = sorted([r for r in rows if r["n_ball"] == n], key=lambda r: r["rho"])
        p = np.array([r["rho"] for r in sub])
        y = np.array([r["ratio_shell_FE"] for r in sub])
        X2 = np.vstack([np.ones_like(p), p, p * p]).T
        X2e = np.vstack([np.ones_like(p), p * p]).T
        X3 = np.vstack([np.ones_like(p), p, p * p, p**3]).T
        c2, r2, s2 = _lsq(X2, y)
        c2e, r2e, s2e = _lsq(X2e, y)
        c3, r3, s3 = _lsq(X3, y)
        o2.append({"n_ball": n, "n_points": len(sub), "C_n": float(c2[0]),
                   "C_n_stderr": float(s2[0]), "b1": float(c2[1]), "b2": float(c2[2]),
                   "max_resid": float(np.max(np.abs(r2))),
                   "C_n_even": float(c2e[0]), "max_resid_even": float(np.max(np.abs(r2e))),
                   "C_n_cubic": float(c3[0]), "max_resid_cubic": float(np.max(np.abs(r3)))})
    nn = np.array([o["n_ball"] + 0.5 for o in o2])
    cn = np.array([o["C_n"] for o in o2])
    fits2 = {}
    for name, X in {"Ap_inv_n2": np.vstack([np.ones_like(nn), 1 / nn**2]).T,
                    "Bp_inv_n_inv_n2": np.vstack([np.ones_like(nn), 1 / nn, 1 / nn**2]).T}.items():
        c, r, s = _lsq(X, cn)
        fits2[name] = {"C_4d_over_FE": float(c[0]), "stderr": float(s[0]),
                       "coef": c.tolist(), "max_resid": float(np.max(np.abs(r)))}
    out["order2_rows"] = o2
    out["order2_fits"] = fits2

    # ---- verdict ----
    # central value: the best-residual rho -> 0 model and the best-residual
    # 1/n -> 0 model, averaged; uncertainty: the full spread of all models
    # and both orders (half), the a/tau-fit model dependence (quadratic vs
    # cubic C_rho), the largest fit standard error, and the residual box
    # correction, added linearly (conservative).
    best1 = min(fits1.items(), key=lambda kv: kv[1]["max_resid"])
    best2 = min(fits2.items(), key=lambda kv: kv[1]["max_resid"])
    c1, c2v = best1[1]["C_4d_over_FE"], best2[1]["C_4d_over_FE"]
    central = 0.5 * (c1 + c2v)
    all_vals = [v["C_4d_over_FE"] for v in fits1.values()] + [v["C_4d_over_FE"] for v in fits2.values()]
    spread = max(all_vals) - min(all_vals)
    x_model = max(abs(o["C_rho_cubic"] - o["C_rho"]) for o in o1)
    fit_err = max(max(v["stderr"] for v in fits1.values()), max(v["stderr"] for v in fits2.values()))
    order_dev = abs(c1 - c2v)
    box_res = float(box_unc)
    unc = spread / 2 + x_model + fit_err + box_res
    out["verdict"] = {"best_model_order1": best1[0], "best_model_order2": best2[0],
                      "C_4d_over_FE_order1": c1, "C_4d_over_FE_order2": c2v,
                      "order_dependence_abs": order_dev,
                      "order_dependence_rel": order_dev / central,
                      "model_spread_abs": spread, "model_spread_rel": spread / central,
                      "a_over_tau_fit_model_dependence": x_model,
                      "max_fit_stderr": fit_err,
                      "box_correction_residual": box_res,
                      "C_4d_over_FE": central, "uncertainty_abs": unc,
                      "uncertainty_rel": unc / central,
                      "C_4d": central * FE_CONSTANT_4D,
                      "C_4d_uncertainty": unc * FE_CONSTANT_4D,
                      "one_over_C_4d_pi2": 1.0 / (central * FE_CONSTANT_4D * math.pi**2),
                      "converged_2pct": bool(unc / central <= 0.02)}
    _write_json("s4e_extrapolation.json", out)
    return out


def s4f_extremal_4d(rho=0.25, n_ball=6):
    """The 4d extremal state, exhibited, at a representative interior point.

    Per sector: the exact minimum, the squeezing matrix Z, the sampling
    weights, the symplectic spectrum, the per-quasimode terms
    -sigma_k ||v_k||^2, the Takagi squeezing spectrum, the site-basis
    covariance (l <= 2), its purity, the exact energy of the exhibited state
    and its gap, and the t = 0 normal-ordered energy-density profile in r
    (per site, and per unit volume with the S4c shell measure).
    """
    tau = (n_ball + 0.5) / rho
    N = _lattice_N(n_ball, tau)
    f = gaussian_f(tau)
    out = radial_ball_qei(N, n_ball, f, l_max=40, tol=0.0, method="mode_space", rel_tol=1e-12)
    total = out["e_min_total"]
    profile_total = np.zeros(N)
    summary = []
    for l, res in enumerate(out["results"]):
        V, info = mode_space_extremal_state(res)
        nu = symplectic_eigenvalues(V)
        K = srednicki_K(N, l)
        dV = V - ground_state_cov(K)
        prof = np.zeros(N)
        for j in range(N):
            e_j = np.zeros(N)
            e_j[j] = 1.0
            for coeff, xi, xv, pi_, pv in _radial_density_terms(N, l, e_j):
                u = np.zeros(2 * N)
                u[xi] = xv
                u[N + pi_] = pv
                prof[j] += 0.5 * coeff * float(u @ dV @ u)
        profile_total += (2 * l + 1) * prof
        s_sq = info["squeeze_s"]
        rec = {"l": l, "e_min": float(res.e_min), "term": (2 * l + 1) * float(res.e_min),
               "e_vac": float(res.e_vac), "n_kept": int(res.n_kept),
               "sigma_max": float(res.sigma.max()), "sigma_min": float(res.sigma.min()),
               "z_norm": float(res.z_norm), "state_reg": float(res.meta["state_reg"]),
               "residual": float(res.residual),
               "purity_dev_max": float(np.max(np.abs(nu - 0.5))),
               "achieved": info["achieved"], "gap_rel": info["gap"] / abs(res.e_min),
               "squeeze_s_max": float(s_sq.max()),
               "squeeze_r_max": float(np.arctanh(min(s_sq.max(), 1 - 1e-16))),
               "n_modes_r_gt_0.1": int(np.sum(s_sq > math.tanh(0.1))),
               "profile_ball_sum": float(prof[:n_ball].sum()),
               "profile_min": float(prof.min())}
        summary.append(rec)
        payload = {"_source": np.array("notebook.py::s4f_extremal_4d"),
                   "Z": res.Z, "alpha_kept": res.alpha_kept, "alpha_all": res.alpha,
                   "omega": res.omega, "sigma": res.sigma, "mode_terms": res.mode_terms,
                   "squeeze_s": s_sq, "nu_V_star": nu, "profile_site": prof,
                   "meta": np.array(json.dumps(rec))}
        if l <= 2:
            n_core = min(N, n_ball + int(math.ceil(4.0 * tau)))
            payload["V_star_core"] = reduce_cov(V, range(n_core))
            payload["V_star_core_sites"] = np.arange(1, n_core + 1)
            payload["U_A"] = res.U_A
        np.savez_compressed(DATA / f"s4f_extremal_4d_l{l}.npz", **payload)
    j = np.arange(1, N + 1)
    shell_vol = 4 * math.pi * (j * j + 1.0 / 12.0)
    rec = {"_source": "notebook.py::s4f_extremal_4d", "rho": rho, "n_ball": n_ball,
           "tau": tau, "N": N, "e_min_total": float(total), "l_stop": int(out["l_stop"]),
           "ratio_shell_FE": -total / (shell_volume(n_ball) * _fpp2_gaussian(tau)) / FE_CONSTANT_4D,
           "profile_closure_rel_dev":
               abs(profile_total[:n_ball].sum() + 0.5 * (profile_total[n_ball] if n_ball < N else 0)
                   - 0.0) and float("nan"),
           "sectors": summary}
    # closure of the t=0 profile is NOT expected (the ball smears over t as
    # well); what is checked is the achieved-vs-e_min gap per sector above.
    rec.pop("profile_closure_rel_dev")
    np.savez_compressed(DATA / "s4f_extremal_4d_profile.npz",
                        _source=np.array("notebook.py::s4f_extremal_4d"),
                        site=j, profile_site=profile_total,
                        profile_per_volume=profile_total / shell_vol,
                        shell_volume=shell_vol, meta=np.array(json.dumps({k: v for k, v in rec.items() if k != "sectors"})))
    _write_json("s4f_extremal_4d_summary.json", rec)
    return rec


# =========================================================================
# S5. audit battery at the paper's resolution
# =========================================================================


def s5_audit_battery(tau0=8.0, g=100.0):
    """Every engineered state satisfies both 2d bounds at paper resolution.

    Includes the extremal state itself (the tightest possible probe: it sits
    AT E_min) and mixtures/squeezed pockets around it. `qei_audit` gates on
    raw symplectic margins (no entropy clamps) and raises on any violation
    by a physical state — so a green battery is a positive result, not a
    silent pass.
    """
    N = _odd(round(g * tau0))
    site = N // 2
    fh = gaussian_f(tau0)
    K = harmonic_chain_K(N, 0.0, "dirichlet")
    op = sampling_operator(N, site, fh, m=0.0, bc="dirichlet")
    res = qei_minimize(op)
    V_vac = ground_state_cov(K)
    V_star = embed_extremal(res, op)

    states = {"vacuum": V_vac, "extremal": V_star,
              "mix_0.5": 0.5 * V_vac + 0.5 * V_star,
              "squeeze_site_r2": squeezed_pocket(K, [site], 2.0, 0.35),
              "squeeze_band": squeezed_pocket(K, range(site - 6, site + 7), 0.5, 0.9)}
    rng = np.random.default_rng(11)
    for i in range(4):
        modes = np.unique(rng.integers(site - 12, site + 13, size=4))
        states[f"random_{i}"] = squeezed_pocket(
            K, modes, rng.uniform(0.1, 1.5), rng.uniform(0.0, math.pi))

    rows = []
    for name, V in states.items():
        for constant in ("sharp", "fewster_eveson"):
            a = qei_audit(V, op, fh, constant=constant, tol=1e-8)
            rows.append({"state": name, "constant": constant,
                         "e_f": a["e_f"], "bound": a["bound"],
                         "margin": a["margin"], "nu_margin": a["nu_margin"],
                         "passed": bool(a["passed"])})
        e_f = smeared_energy(V, op)
        assert e_f >= res.e_min - 1e-9, (
            f"{name}: E_f {e_f:.6e} below the exhibited global minimum "
            f"{res.e_min:.6e} — impossible, audit event")
    worst = min(r["margin"] for r in rows)
    _write_csv("s5_audit_battery.csv", rows, "notebook.py::s5_audit_battery")
    return {"rows": rows, "worst_margin": float(worst),
            "n_states": len(states), "N": N, "tau0": tau0,
            "all_passed": all(r["passed"] for r in rows)}


# =========================================================================
# S6. the same variational problem in the continuum, in momentum space
# =========================================================================
#
# vacuum.inequalities.qei.worldline_momentum_minimize / ball_momentum_minimize
# solve the SAME infimum as S4 without a lattice: the free field's smeared
# energy is a quadratic form in the continuum mode operators, its partial-
# wave kernels are (w w')^{3/2} Fhat(w -+ w') times spherical-Bessel overlaps
# of the ball (module note in qei.py), and the w-continuum is a Gauss-
# Legendre grid. In 2d the chiral kernel sqrt(w w') Fhat must give Flanagan's
# sharp constant for EVERY f (control); in 4d the worldline kernel gives the
# sharp constant for the given f directly, and the ball kernels give C_eff(rho).
# The R/tau expansion is then derived: chi_B(q)/V_B = 1 - q^2 R^2/10 + O(R^4)
# is even and entire in R; the l = 0, 1 sectors shift at O(R^2) (Hellmann-
# Feynman on the worldline extremal state) and the l = 2 sector opens at
# O(R^2) as a whole operator (its infimum scales exactly), so
#     C_eff(rho)/C_FE = C_4d/C_FE + d2 rho^2 + d4 rho^4 + ...   (even, no logs).


def _lsq_fit(cols, y):
    X = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, y - X @ coef


def s6a_momentum_2d_control(grids=((400, 16.0), (800, 30.0), (1200, 100.0))):
    """Flanagan's sharp constant from the continuum kernel for six families.

    Each family at increasing (n, omega_max); the compact families' Fourier
    tails are algebraic and converge like a power of omega_max, the cusped
    Ford-Roman weight like 1/n^2. The modulated Gaussian is run at c = +0.3
    (no zero of f) and c = -0.3 (f changes sign at |s| = 1.83): the latter's
    ratio stays below 1, i.e. the int f'^2 form is not the sharp bound for a
    weight with zeros (Flanagan's optimal-state construction needs rho > 0).
    """
    fams = {"gaussian": gaussian_f(1.0), "cos2": cos2_f(1.0), "bump4": bump4_f(1.0),
            "sech": sech_f(1.0), "lorentzian": lorentzian_f(1.0),
            "sqrt_lorentzian": q_sqrt_lorentzian_f(1.0),
            "modgauss_c+0.3": modulated_gaussian_f(1.0, 0.3),
            "modgauss_c-0.3": modulated_gaussian_f(1.0, -0.3)}
    rows = {}
    for name, fh in fams.items():
        rows[name] = []
        for n, om in grids:
            om_eff = om if name in ("cos2", "bump4") else min(om, 40.0)
            n_eff = n if name in ("cos2", "bump4", "sqrt_lorentzian") else min(n, 800)
            t0 = time.time()
            r = worldline_momentum_minimize(fh, d=2, n=n_eff, omega_max=om_eff)
            rows[name].append({"n": r.n, "omega_max": r.omega_max, "ratio": r.ratio,
                               "n_kept": r.n_kept, "e_dropped": r.e_dropped,
                               "residual": r.residual, "seconds": time.time() - t0})
    rec = {"_source": "notebook.py::s6a_momentum_2d_control", "rows": rows,
           "worst_dev_no_zero_families": max(abs(v[-1]["ratio"] - 1.0) for k, v in rows.items()
                                             if k != "modgauss_c-0.3"),
           "modgauss_c-0.3_ratio": rows["modgauss_c-0.3"][-1]["ratio"]}
    _write_json("s6a_momentum_2d_control.json", rec)
    return rec


def s6b_momentum_4d_worldline(grids=((300, 12.0), (400, 14.0), (500, 16.0), (600, 20.0), (800, 24.0))):
    """C_4d/C_FE for Gaussian sampling from the continuum worldline kernel."""
    fh = gaussian_f(1.0)
    rows = []
    for n, om in grids:
        t0 = time.time()
        r = worldline_momentum_minimize(fh, d=4, n=n, omega_max=om)
        rows.append({"n": n, "omega_max": om, "ratio": r.ratio, "e_Q": r.meta["e_Q"],
                     "sector_l0": r.sectors[0][2], "sector_l1_per_copy": r.sectors[1][2],
                     "n_kept": r.n_kept, "e_dropped": r.e_dropped, "residual": r.residual,
                     "seconds": time.time() - t0})
    final = rows[-1]["ratio"]
    rec = {"_source": "notebook.py::s6b_momentum_4d_worldline", "rows": rows,
           "C_4d_over_FE": final, "C_4d": final * FE_CONSTANT_4D,
           "one_over_C_4d_pi2": 1.0 / (final * FE_CONSTANT_4D * math.pi**2),
           "grid_uncertainty": abs(rows[-1]["ratio"] - rows[-2]["ratio"]),
           "l0_share": rows[-1]["sector_l0"] / (rows[-1]["sector_l0"] + 3 * rows[-1]["sector_l1_per_copy"])}
    _write_json("s6b_momentum_4d_worldline.json", rec)
    return rec


def s6c_rho_expansion(scan_rows, ext, n=400, om=14.0,
                      rhos=(0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.125, 0.15, 1 / 6, 0.2, 0.25, 1 / 3, 0.4, 0.5)):
    """The R/tau expansion, derived and checked; the lattice against the continuum.

    (i) d2 = d2(HF; l = 0, 1) + d2(l = 2) from `rho2_coefficient` (with the
    finite-difference check of the Hellmann-Feynman shifts); (ii) the
    continuum ball at fixed rho: parity of the expansion (even fit residual,
    free rho^3 / rho / rho^2 log rho terms); (iii) the archived 36-point
    lattice grid minus the continuum C_eff(rho): the lattice error at fixed
    rho as a function of a/R = 1/(n_ball + 1/2) — its leading exponent from
    the data; (iv) the archived C_rho tables re-extrapolated with the derived
    even form, and the raw grid re-fitted at fixed rho with the exponent (iii)
    found, then extrapolated with the derived form: the lattice-only
    constant with its honest budget, against the continuum value.
    """
    f = gaussian_f(1.0)
    wl = worldline_momentum_minimize(f, d=4, n=n, omega_max=om)
    C4d = wl.ratio
    k = rho2_coefficient(f, n=n, omega_max=om)
    d2 = k["k2"]

    cont = {}
    for rho in rhos:
        b = ball_momentum_minimize(f, rho, n=n, omega_max=om)
        cont[rho] = {"C": b["ratio_FE"], "l_stop": b["l_stop"], "slope": (b["ratio_FE"] - C4d) / rho**2,
                     "terms": b["terms"].tolist()}
    rr = np.array(list(cont.keys()))
    cc = np.array([cont[r]["C"] for r in rr])
    small = rr <= 0.2 + 1e-12
    rs, cs = rr[small], cc[small]
    parity = {}
    for name, cols in (("even_r2_r4", [rs**0, rs**2, rs**4]),
                       ("even_r2_r4_r6", [rs**0, rs**2, rs**4, rs**6]),
                       ("plus_r3", [rs**0, rs**2, rs**3, rs**4]),
                       ("plus_r1", [rs**0, rs, rs**2, rs**4]),
                       ("plus_r2logr", [rs**0, rs**2, rs**2 * np.log(rs), rs**4])):
        coef, res = _lsq_fit(cols, cs)
        parity[name] = {"C0": float(coef[0]), "C0_minus_C4d": float(coef[0] - C4d),
                        "coef": coef[1:].tolist(), "max_resid": float(np.max(np.abs(res)))}
    coef, res = _lsq_fit([rs**2, rs**4, rs**6], cs - C4d)
    fixed = {"d2": float(coef[0]), "d4": float(coef[1]), "d6": float(coef[2]),
             "max_resid": float(np.max(np.abs(res))), "d2_minus_derived": float(coef[0] - d2)}
    # full-range even series (the model for the lattice rho's)
    coef_full, res_full = _lsq_fit([rr**2, rr**4, rr**6, rr**8], cc - C4d)
    series_full = {"d2": float(coef_full[0]), "d4": float(coef_full[1]), "d6": float(coef_full[2]),
                   "d8": float(coef_full[3]), "max_resid": float(np.max(np.abs(res_full)))}

    # (iii) lattice minus continuum on the archived grid
    rho_lat = sorted(set(r["rho_nominal"] for r in scan_rows), reverse=True)
    lat = {}
    ax, ay, ar = [], [], []
    for rho in rho_lat:
        sub = sorted([r for r in scan_rows if abs(r["rho_nominal"] - rho) < 1e-9], key=lambda r: r["n_ball"])
        nb = np.array([r["n_ball"] for r in sub], dtype=float)
        aR = 1.0 / (nb + 0.5)
        Cc = cont[rho]["C"] if rho in cont else ball_momentum_minimize(f, rho, n=n, omega_max=om)["ratio_FE"]
        y = np.array([r["ratio_shell_FE"] for r in sub]) - Cc
        slope = float(np.polyfit(np.log(aR), np.log(-y), 1)[0])
        c12, r12 = _lsq_fit([aR, aR**2], y)
        c123, r123 = _lsq_fit([aR, aR**2, aR**3], y)
        c23, r23 = _lsq_fit([aR**2, aR**3], y)
        lat[rho] = {"C_cont": float(Cc), "n_ball": nb.tolist(), "dev": y.tolist(),
                    "loglog_slope": slope,
                    "fit_e1_e2": {"e1": float(c12[0]), "e2": float(c12[1]), "max_resid": float(np.max(np.abs(r12)))},
                    "fit_e1_e2_e3": {"e1": float(c123[0]), "e2": float(c123[1]), "e3": float(c123[2]),
                                     "max_resid": float(np.max(np.abs(r123)))},
                    "fit_e2_e3": {"e2": float(c23[0]), "e3": float(c23[1]), "max_resid": float(np.max(np.abs(r23)))}}
        ax.append(aR); ay.append(y); ar.append(np.full_like(aR, rho))
    ax, ay, ar = map(np.concatenate, (ax, ay, ar))
    cg, rg = _lsq_fit([ax**2, ax**3, (ar * ax)**2], ay)
    global_fit = {"e2_aR2": float(cg[0]), "e3_aR3": float(cg[1]), "c_atau2": float(cg[2]),
                  "max_resid": float(np.max(np.abs(rg)))}
    slopes = [lat[r]["loglog_slope"] for r in rho_lat]
    e1s = [lat[r]["fit_e1_e2_e3"]["e1"] for r in rho_lat]

    # (iv) re-extrapolations
    o1 = ext["order1_rows"]
    rr1 = np.array([o["rho"] for o in o1])
    reext = {}
    for key in ("C_rho", "C_rho_cubic"):
        c1 = np.array([o[key] for o in o1])
        out = {}
        for name, cols in (("even_r2_r4", [rr1**0, rr1**2, rr1**4]), ("even_r2_r4_r6", [rr1**0, rr1**2, rr1**4, rr1**6])):
            coef, res = _lsq_fit(cols, c1)
            out[name] = {"C_4d_over_FE": float(coef[0]), "coef": coef[1:].tolist(), "max_resid": float(np.max(np.abs(res)))}
        coef, res = _lsq_fit([rr1**0, rr1**4], c1 - d2 * rr1**2)
        out["d2_fixed_r4"] = {"C_4d_over_FE": float(coef[0]), "d4": float(coef[1]), "max_resid": float(np.max(np.abs(res)))}
        reext[key] = out
    # the raw grid re-fitted at fixed rho with no linear term (exponent from (iii))
    new_rows = []
    for rho in rho_lat:
        sub = sorted([r for r in scan_rows if abs(r["rho_nominal"] - rho) < 1e-9], key=lambda r: r["x"])
        x = np.array([r["x"] for r in sub])
        y = np.array([r["ratio_shell_FE"] for r in sub])
        c23, r23 = _lsq_fit([x**0, x**2, x**3], y)
        c234, r234 = _lsq_fit([x**0, x**2, x**3, x**4], y)
        new_rows.append({"rho": rho, "C_rho_x2x3": float(c23[0]), "max_resid_x2x3": float(np.max(np.abs(r23))),
                         "C_rho_x2x3x4": float(c234[0]), "max_resid_x2x3x4": float(np.max(np.abs(r234))),
                         "C_cont": lat[rho]["C_cont"],
                         "dev_x2x3": float(c23[0] - lat[rho]["C_cont"]),
                         "dev_x2x3x4": float(c234[0] - lat[rho]["C_cont"])})
    rr2 = np.array([r["rho"] for r in new_rows])
    # Two axes of model choice, both fitted FROM THE LATTICE DATA:
    #   a/tau fit   x^2+x^3   vs  x^2+x^3+x^4      (the a/tau -> 0 step)
    #   rho form    rho^2+rho^4 vs +rho^6           (the R/tau -> 0 step, d2 FREE)
    # Those four values are the "independent" set: no number computed by the
    # continuum solver enters them (only the *forms* -- even powers, and no
    # linear a/tau term -- do; see `shared_inputs` below).
    # A third rho-form variant PINS d2 to the derived (continuum) value. It is
    # a legitimate extrapolation but it is NOT independent of the continuum
    # route, so it is excluded from the central value and from the
    # lattice-vs-continuum agreement, and reported separately.
    lattice_only = {}
    for key in ("C_rho_x2x3", "C_rho_x2x3x4"):
        cvals = np.array([r[key] for r in new_rows])
        coef, res = _lsq_fit([rr2**0, rr2**2, rr2**4], cvals)
        coef6, res6 = _lsq_fit([rr2**0, rr2**2, rr2**4, rr2**6], cvals)
        coeff, resf = _lsq_fit([rr2**0, rr2**4], cvals - d2 * rr2**2)
        lattice_only[key] = {"even_r2_r4": float(coef[0]), "even_r2_r4_r6": float(coef6[0]),
                             "d2_fixed_NOT_INDEPENDENT": float(coeff[0]), "fitted_d2": float(coef[1]),
                             "max_resid": float(np.max(np.abs(res)))}
    keys = ("C_rho_x2x3", "C_rho_x2x3x4")
    indep = [lattice_only[kk][m] for kk in keys for m in ("even_r2_r4", "even_r2_r4_r6")]
    pinned = [lattice_only[kk]["d2_fixed_NOT_INDEPENDENT"] for kk in keys]
    vals_arch = [reext[kk][m]["C_4d_over_FE"] for kk in reext for m in reext[kk]]
    central = float(np.mean(indep))
    # budget along the two independent axes (not the raw spread: the spread is
    # dominated by the rho-form axis and would double-count it)
    a_tau_axis = float(max(abs(lattice_only["C_rho_x2x3"][m] - lattice_only["C_rho_x2x3x4"][m])
                           for m in ("even_r2_r4", "even_r2_r4_r6")))
    rho_axis = float(max(abs(lattice_only[kk]["even_r2_r4"] - lattice_only[kk]["even_r2_r4_r6"])
                         for kk in keys))
    budget = {"a_tau_fit_model": a_tau_axis, "rho_form_model": rho_axis,
              "box_correction_residual": 4.6e-6}
    unc = sum(budget.values())
    # Systematics deliberately NOT summed into `unc`, each with its reason.
    excluded = {
        "d2_pinned_shift": {
            "value": float(max(abs(v - central) for v in pinned)),
            "reason": "the d2-pinned variants import the continuum value of d2, so they "
                      "cannot enter a number whose purpose is to be independent of the "
                      "continuum solve; their own values are reported in `lattice_only`. "
                      "Adding this term gives `conservative_uncertainty`."},
        "archived_a_tau_models_spread": {
            "value": float(max(vals_arch) - min(vals_arch)) / 2,
            "reason": "half-spread of the re-extrapolated ARCHIVED C_rho tables, whose "
                      "quadratic-in-x rows carry a linear a/tau term that the measured "
                      "(a/R)^2 exponent rules out; including it would double-count a model "
                      "already excluded by measurement. Recorded so a reader who distrusts "
                      "the exponent can restore it."}}
    conservative = unc + excluded["d2_pinned_shift"]["value"]
    verdict = {"C_4d_over_FE_continuum": float(C4d), "continuum_grid_uncertainty": 2e-6,
               "d2_derived": float(d2), "d2_hf_l01": float(k["k2_hf"]), "d2_l2": float(k["k2_l2"]),
               "d2_numerical_small_rho": fixed["d2"], "d2_archived_fit_r2_r4": float(ext["order1_fits"]["D_rho2_rho4"]["coef"][1]),
               "lattice_only_C_4d_over_FE": central, "lattice_only_uncertainty": float(unc),
               "lattice_only_conservative_uncertainty": float(conservative),
               "lattice_only_values_independent": indep,
               "lattice_only_values_d2_pinned_NOT_INDEPENDENT": pinned,
               "lattice_only_budget": budget,
               "lattice_only_excluded_systematics": excluded,
               "shared_inputs": {
                   "shared_forms_not_values": [
                       "the even-power rho series (derived analytically in rho2_coefficient; "
                       "its parity is also verified on continuum ball data)",
                       "the absence of a linear a/tau term at fixed rho (MEASURED by "
                       "comparing the lattice grid to the continuum C_eff(rho), so this "
                       "input is not independent of the continuum route)"],
                   "shared_values": ["none in the four independent values (d2 is fitted, "
                                     "not pinned); the two d2-pinned variants share d2"],
                   "independent": [
                       "every lattice energy (Srednicki radial lattice, cancellation-free "
                       "Riccati minimizer, exact Fourier sampling, converged l-sum)",
                       "the shell measure (derived from Srednicki Eq. 10)",
                       "the -10.04/g^4 box law (measured on the lattice)"]},
               "lattice_only_minus_continuum": float(central - C4d),
               "lattice_only_minus_continuum_in_sigma": float(abs(central - C4d) / unc),
               "archived_C_4d_over_FE": float(ext["verdict"]["C_4d_over_FE"]),
               "archived_uncertainty": float(ext["verdict"]["uncertainty_abs"]),
               "continuum_inside_archived_band": bool(abs(C4d - ext["verdict"]["C_4d_over_FE"]) < ext["verdict"]["uncertainty_abs"]),
               "lattice_error_loglog_slope_range": [float(min(slopes)), float(max(slopes))],
               "lattice_error_linear_coefficient_range": [float(min(e1s)), float(max(e1s))],
               "lattice_error_global_fit": global_fit,
               "even_fit_residual_small_rho": parity["even_r2_r4_r6"]["max_resid"],
               "odd_term_coefficient": parity["plus_r3"]["coef"][1],
               "linear_term_coefficient": parity["plus_r1"]["coef"][0],
               "log_term_coefficient": parity["plus_r2logr"]["coef"][1]}
    rec = {"_source": "notebook.py::s6c_rho_expansion", "grid": {"n": n, "omega_max": om},
           "rho2_coefficient": {kk: v for kk, v in k.items() if kk != "pieces"},
           "rho2_pieces": {str(l): v for l, v in k["pieces"].items()},
           "continuum_ball": {f"{r:.6f}": v for r, v in cont.items()},
           "parity_fits_rho_le_0.2": parity, "fixed_C4d_fit": fixed, "even_series_full_range": series_full,
           "lattice_minus_continuum": {f"{r:.6f}": v for r, v in lat.items()},
           "reextrapolation_archived_tables": reext, "raw_grid_refit_no_linear": new_rows,
           "lattice_only": lattice_only, "verdict": verdict}
    _write_json("s6c_rho_expansion.json", rec)
    return rec


def _richardson_omega(ladder):
    """Fit C(omega) = C_inf - K/omega - K2/omega^2 to an (omega, C) ladder."""
    om = np.array([x[0] for x in ladder], dtype=float)
    c = np.array([x[1] for x in ladder])
    coef1, res1 = _lsq_fit([om**0, 1 / om], c)
    coef2, res2 = _lsq_fit([om**0, 1 / om, 1 / om**2], c) if len(ladder) >= 3 else (coef1, res1)
    return {"C_inf_1": float(coef1[0]), "C_inf_2": float(coef2[0]), "K": float(-coef2[1]),
            "max_resid_2": float(np.max(np.abs(res2))), "ladder": [(float(a), float(b)) for a, b in ladder],
            "C_inf": float(coef2[0]), "unc": float(abs(coef2[0] - coef1[0]))}


def s6d_families_4d(quick=False):
    """The 4d sharp constant per sampling family, and the sup over families.

    Scale-only families are one number each (the ratio is scale invariant,
    checked): Gaussian, sech, Lorentzian (f = 1/(1+s^2)), the Ford-Roman
    weight (f^2 Lorentzian; cusped transform, n-ladder + 1/n^2 Richardson),
    cos^2 and the C^1 bump (1-u^2)^2 (algebraic transforms; omega_max ladder
    at fixed density and a 1/omega_max fit — the infimum's UV tail is the
    omega^4 |fhat|^2 tail of a function whose f'' jumps). The smoothness axis
    (1-u^2)^p, p = 2 ... 8, and three optimizations by
    `optimize_sampling_momentum`: the modulated Gaussian e^{-s^2/2}(1 + c s^2)
    over c, the bump over p, and e^{-s^2/2}(1 + c s^2 + d s^4) over (c, d);
    each optimization's 2d control is run at its optimum.
    The Ford-Roman weight is compared with Fewster-Ford-Roman, PRD 85,
    125038 (2012), Eq. (69): their moment-based estimate of the optimal bound
    x0(rho_S) ~ 0.0236 in units (4 pi tau^2)^2, obtained from y_inf <= x0 (their
    Eq. (64)) — a lower estimate when the moment problem is indeterminate,
    which they note is the case; ours is the exact infimum.
    """
    out = {}

    def one(fh, n, om, d=4):
        r = worldline_momentum_minimize(fh, d=d, n=n, omega_max=om)
        return r.ratio, {"n": n, "omega_max": om, "n_kept": r.n_kept, "e_dropped": r.e_dropped}

    # Gaussian-tailed / exponential-tailed: converged on one grid, scale check
    for name, fh, fh2, n, om in (("gaussian", gaussian_f(1.0), gaussian_f(2.0), 600, 20.0),
                                 ("sech", sech_f(1.0), sech_f(2.0), 800, 40.0),
                                 ("lorentzian", lorentzian_f(1.0), lorentzian_f(2.0), 800, 40.0)):
        r1, info = one(fh, n, om)
        r2, _ = one(fh2, n, om / 2.0)
        r2d, _ = one(fh, n, om, d=2)
        out[name] = {"ratio": r1, "scale_check_dev": abs(r1 - r2), "ratio_2d_control": r2d, **info}
    # Ford-Roman weight: cusp -> n ladder, 1/n^2 Richardson
    ns = (400, 800, 1200) if not quick else (300, 600)
    lad = []
    for nn in ns:
        r, info = one(q_sqrt_lorentzian_f(1.0), nn, 40.0)
        lad.append((nn, r))
    nn_ = np.array([x[0] for x in lad], dtype=float)
    cc_ = np.array([x[1] for x in lad])
    coef, res = _lsq_fit([nn_**0, 1 / nn_**2], cc_)
    r2d, _ = one(q_sqrt_lorentzian_f(1.0), ns[-1], 40.0, d=2)
    x0 = float(coef[0]) * 27.0 / 128.0
    out["sqrt_lorentzian"] = {"ratio": float(coef[0]), "ladder": lad, "unc": float(abs(coef[0] - cc_[-1])),
                              "ratio_2d_control": r2d,
                              "x0_FFR_units": x0, "x0_FE_FFR_units": 27.0 / 128.0, "x0_FordRoman_FFR_units": 1.5,
                              "x0_FFR2012_estimate": 0.0236, "x0_over_FFR2012": x0 / 0.0236,
                              "bound_over_tau4": -x0 / (16.0 * math.pi**2)}
    # compact: omega ladder at fixed density n/omega = 8
    ladder_pts = ((100, 800), (150, 1200), (200, 1600), (300, 2400), (400, 3200)) if not quick else ((60, 480), (100, 800), (150, 1200))
    for name, fh in (("cos2", cos2_f(1.0)), ("bump_p2", bump_f(1.0, 2.0))):
        lad = []
        for om, nn in ladder_pts:
            t0 = time.time()
            r, info = one(fh, nn, float(om))
            lad.append((om, r))
        fit = _richardson_omega(lad)
        r2d, _ = one(fh, ladder_pts[-1][1], float(ladder_pts[-1][0]), d=2)
        out[name] = {"ratio": fit["C_inf"], "unc": fit["unc"] + fit["max_resid_2"], "omega_fit": fit,
                     "ratio_2d_control": r2d}
    # smoothness axis (1-u^2)^p: 2-point Richardson in 1/omega
    pts = ((150, 1200), (200, 1600)) if not quick else ((60, 480), (100, 800))
    smooth = []
    for p in (2.0, 2.5, 3.0, 4.0, 6.0, 8.0):
        fh = bump_f(1.0, p)
        ra, _ = one(fh, pts[0][1], float(pts[0][0]))
        rb, _ = one(fh, pts[1][1], float(pts[1][0]))
        rich = rb + (rb - ra) * (1 / pts[1][0]) / (1 / pts[0][0] - 1 / pts[1][0])
        smooth.append({"p": p, "ratio_a": ra, "ratio_b": rb, "ratio": rich, "tail_size": abs(rb - ra)})
    out["bump_p_scan"] = smooth
    # optimizations (shape parameters only), with the 2d control at the optimum
    opts = {}
    fam_c = SamplingFamily(make=lambda th: gaussian_poly_f(1.0, [1.0, float(np.ravel(th)[0])]),
                           df_dtheta=None, n_params=1, label="modgauss_c")
    o = optimize_sampling_momentum(fam_c, [0.1], bounds=[(0.0, 1.5)], d=4, n=400, omega_max=16.0)
    o2 = worldline_momentum_minimize(o.f, d=2, n=400, omega_max=16.0).ratio
    opts["modgauss_c"] = {"theta_star": o.theta.tolist(), "ratio": o.ratio, "ratio_2d_control": o2,
                          "converged": o.converged, "n_iter": o.n_iter, "gaussian_ratio": out["gaussian"]["ratio"]}

    # f = e^{-s^2/2}(1 + c s^2 + d s^4) with the analytic (Hermite) transform,
    # int f'^2 and int f''^2 in closed form: vacuum.inequalities.qei.gaussian_poly_f
    fam_cd = SamplingFamily(make=lambda th: gaussian_poly_f(1.0, [1.0, float(np.ravel(th)[0]), float(np.ravel(th)[1])]),
                            df_dtheta=None, n_params=2, label="modgauss_cd")
    o = optimize_sampling_momentum(fam_cd, [0.3, 0.0], bounds=[(0.0, 1.5), (-0.2, 0.5)], d=4, n=400,
                                   omega_max=16.0, method="Nelder-Mead", maxiter=20)
    o2 = worldline_momentum_minimize(o.f, d=2, n=400, omega_max=16.0).ratio
    opts["modgauss_cd"] = {"theta_star": o.theta.tolist(), "ratio": o.ratio, "ratio_2d_control": o2,
                           "converged": o.converged, "n_iter": o.n_iter}
    fam_p = SamplingFamily(make=lambda th: bump_f(1.0, float(np.ravel(th)[0])), df_dtheta=None,
                           n_params=1, label="bump_p")
    pn, pom = (pts[0][1], float(pts[0][0]))
    o = optimize_sampling_momentum(fam_p, [3.0], bounds=[(2.0, 8.0)], d=4, n=pn, omega_max=pom)
    o2 = worldline_momentum_minimize(o.f, d=2, n=pn, omega_max=pom).ratio
    opts["bump_p"] = {"theta_star": o.theta.tolist(), "ratio_at_grid": o.ratio, "ratio_2d_control": o2,
                      "converged": o.converged, "n_iter": o.n_iter, "grid": {"n": pn, "omega_max": pom},
                      "note": "optimum at the admissibility edge p = 2 (f'' in L^2 needs p > 3/2; the ratio grows as p decreases)"}
    out["optimizations"] = opts
    ranking = sorted([(k, v["ratio"]) for k, v in out.items() if isinstance(v, dict) and "ratio" in v],
                     key=lambda kv: -kv[1])
    sup_est = max(r for _, r in ranking)
    rec = {"_source": "notebook.py::s6d_families_4d", "quick": quick, "families": out, "ranking": ranking,
           "gaussian_optimal": bool(ranking[0][0] == "gaussian"),
           "sup_over_tested_families": sup_est, "sup_family": ranking[0][0],
           "all_below_FE": bool(all(r < 1.0 for _, r in ranking))}
    _write_json("s6d_families_4d.json", rec)
    return rec


def s6e_ffr2012_comparison(quick=False):
    """The acceptance test against the one published sharp-4d estimate.

    Fewster, Ford & Roman, PRD 85, 125038 (2012) [arXiv:1204.3570], Sec. IV,
    Eqs. (64), (68), (69): for the Lorentzian weight f(t) = tau/(pi (t^2 +
    tau^2)) (their f is our f^2, the Ford-Roman sampling function) they
    compute 65 moments of the smeared Wick square phidot^2 and obtain, by a
    Stieltjes-type moment test, y_inf(phidot^2) = 0.02361 +- 1e-5 in the
    units x = (4 pi tau^2)^2 rho, with y_inf <= x0 rigorous (Eq. (64)) and
    y_inf = x0 conjectured; their Eq. (57) then sets x0(rho_S) = x0(phidot^2)
    by treating the components of T_00 as independent distributions. In the
    same units the Fewster-Eveson bound is 27/128 and Ford-Roman's is 3/2.

    Here: (i) the exact continuum infimum of phidot^2, (grad phi)^2 and T_00
    for that weight (they coincide -- one squeezing state serves all three,
    the exact form of their Eq. (57)); (ii) the same weight through the 2d
    kernel (Flanagan control); (iii) the same weight on the Srednicki lattice
    at rho = 1/4, n_ball = 3, 4, 6, extrapolated in (a/R)^2, against the
    continuum ball value at rho = 1/4 -- an independent discretization of the
    same infimum; (iv) the verdict: our x0 vs their y_inf, and which rigorous
    statements are respected.
    """
    fh = q_sqrt_lorentzian_f(1.0)
    grids = ((600, 40.0), (800, 40.0)) if quick else ((600, 40.0), (800, 40.0), (1200, 40.0))
    cont = []
    for gi, (n, om) in enumerate(grids):
        rec = {"n": n, "omega_max": om}
        # the three operators share E_Q by construction (one Bogoliubov solve
        # each); all three are run on the last grid, T00 alone on the others
        ops = ("T00", "phidot2", "gradphi2") if gi == len(grids) - 1 else ("T00",)
        for op in ops:
            t0 = time.time()
            r = worldline_momentum_minimize(fh, d=4, n=n, omega_max=om, operator=op)
            rec[op] = {"ratio_FE": r.ratio, "x0_FFR_units": r.ratio * 27.0 / 128.0,
                       "e_min_per_volume": r.e_min, "sectors": [list(x) for x in r.sectors],
                       "seconds": time.time() - t0}
        rec["ratio_2d_control"] = worldline_momentum_minimize(fh, d=2, n=n, omega_max=om).ratio
        cont.append(rec)
    last = cont[-1]
    x0 = last["T00"]["x0_FFR_units"]
    # 1/n^2 Richardson of the cusp-limited grid ladder
    nn = np.array([c["n"] for c in cont], float)
    xx = np.array([c["T00"]["x0_FFR_units"] for c in cont])
    coef, res = _lsq_fit([nn**0, 1.0 / nn**2], xx)
    x0_extrap = float(coef[0])
    # (iii) lattice cross-check at rho = 1/4. The finite-box (infrared) bias
    # is WEIGHT-DEPENDENT: the Gaussian's -10.04/g^4 law (S4d) does not apply
    # to a weight whose transform decays only as e^{-|nu| tau} -- the box
    # discretizes the spectrum below 1/tau, exactly where this weight has its
    # power. So the box law is measured here for this weight, with its
    # exponent FITTED (models g^-1, g^-2, g^-4 and g^-2 + g^-4 compared), and
    # every lattice point is corrected with the measured law before the
    # (a/R)^2 extrapolation.
    rho = 0.25

    def _lat_point(nb, g):
        tau = (nb + 0.5) / rho
        N = int(nb + math.ceil(g * tau) + 8)
        t0 = time.time()
        out = radial_ball_qei(N, nb, q_sqrt_lorentzian_f(tau), l_max=80, tol=0.0,
                              method="mode_space", rel_tol=1e-12)
        fpp2 = fsecond_sq_integral(q_sqrt_lorentzian_f(tau))
        Craw = -out["e_min_total"] / (shell_volume(nb) * fpp2) / FE_CONSTANT_4D
        return {"n_ball": nb, "tau": tau, "N": N, "g": N / tau, "C_raw": float(Craw),
                "l_stop": int(out["l_stop"]), "converged": bool(out["converged"]),
                "seconds": time.time() - t0}

    box_rows, box_failed = [], []
    for g in ((8, 12, 16, 24) if quick else (8, 12, 16, 24, 32)):
        try:
            box_rows.append(_lat_point(4, g))
        except np.linalg.LinAlgError as err:   # a sector eigenproblem that no driver converges
            box_failed.append({"g": g, "error": str(err)})
    gg = np.array([r["g"] for r in box_rows]); cb = np.array([r["C_raw"] for r in box_rows])
    box_fits = {}
    for name, cols in (("g-1", [gg**0, gg**-1.0]), ("g-2", [gg**0, gg**-2.0]), ("g-4", [gg**0, gg**-4.0]),
                       ("g-1_g-2", [gg**0, gg**-1.0, gg**-2.0]), ("g-2_g-4", [gg**0, gg**-2.0, gg**-4.0])):
        c, r = _lsq_fit(cols, cb)
        box_fits[name] = {"C_inf": float(c[0]), "coef": c[1:].tolist(), "max_resid": float(np.max(np.abs(r)))}
    # free-exponent fit C = C_inf + B g^-p by a 1-d scan on p
    best = None
    for pexp in np.linspace(0.5, 4.5, 81):
        c, r = _lsq_fit([gg**0, gg**-pexp], cb)
        rr = float(np.max(np.abs(r)))
        if best is None or rr < best[0]:
            best = (rr, float(pexp), float(c[0]), float(c[1]))
    box_law = {"exponent": best[1], "B": best[3], "C_inf_at_nb4": best[2], "max_resid": best[0],
               "gaussian_law_for_comparison": "-10.04/g^4 (S4d)"}

    def _boxcorr(row):
        return row["C_raw"] - box_law["B"] * row["g"] ** (-box_law["exponent"])

    at16 = [r for r in box_rows if abs(r["g"] * (4 + 0.5) / rho - (4 + math.ceil(BOX_RATIO * (4 + 0.5) / rho) + 8)) < 1e-9]
    pts = [_lat_point(3, BOX_RATIO)] + (at16[:1] if at16 else [_lat_point(4, BOX_RATIO)]) + ([] if quick else [_lat_point(6, BOX_RATIO)])
    lat = [dict(r, C_boxcorr=_boxcorr(r), C_boxcorr_gaussian_law=r["C_raw"] + 10.04 / r["g"] ** 4) for r in pts]
    nbv = np.array([r["n_ball"] for r in lat], float)
    aR = 1.0 / (nbv + 0.5)
    Cl = np.array([r["C_boxcorr"] for r in lat])
    Clg = np.array([r["C_boxcorr_gaussian_law"] for r in lat])
    c2, r2 = _lsq_fit([aR**0, aR**2], Cl)
    c2g, _ = _lsq_fit([aR**0, aR**2], Clg)
    c23, r23 = _lsq_fit([aR**0, aR**2, aR**3], Cl) if len(lat) >= 3 else (c2, r2)
    t0 = time.time()
    ballc = ball_momentum_minimize(fh, rho, n=(500 if quick else 800), omega_max=40.0, l_max=20)
    lattice_vs_cont = {"rho": rho, "rows": lat, "box_sweep_nb4": box_rows, "box_sweep_failed": box_failed,
                       "box_fits": box_fits,
                       "box_law_measured": box_law, "C_cont_rho": ballc["ratio_FE"],
                       "lattice_extrap_aR2": float(c2[0]), "lattice_extrap_aR2_aR3": float(c23[0]),
                       "lattice_extrap_aR2_with_GAUSSIAN_box_law": float(c2g[0]),
                       "dev_aR2": float(c2[0] - ballc["ratio_FE"]),
                       "dev_aR2_aR3": float(c23[0] - ballc["ratio_FE"]),
                       "dev_with_gaussian_box_law": float(c2g[0] - ballc["ratio_FE"]),
                       "continuum_seconds": time.time() - t0}
    # (v) the EXHIBITED state: orthonormal box wavepackets, exact restricted
    # matrix elements, exact expectation value -- a rigorous lower bound on x0
    # with no convergence claim (only analytic-piece quadratures, checked by
    # doubling the nodes). Ladder in the number of boxes.
    exhibited = []
    for nb_, nodes_ in (((150, 12), (300, 12), (300, 24)) if quick else ((150, 12), (300, 12), (300, 24), (600, 12), (1000, 12))):
        t0 = time.time()
        st = box_wavepacket_state(fh, n_boxes=nb_, nodes=nodes_, omega_max=40.0)
        exhibited.append({"n_boxes": nb_, "nodes": nodes_, "x_state_FFR_units": st["ratio_FE"] * 27.0 / 128.0,
                          "ratio_FE": st["ratio_FE"], "z_norm_used": st["z_norm_used"], "clipped": st["clipped"],
                          "state_reg_used": st["state_reg_used"], "n_kept": st["n_kept"],
                          "riccati_residual": st["riccati_residual"], "seconds": time.time() - t0})
    best_state = max(exhibited, key=lambda r: r["x_state_FFR_units"])
    node_dep = abs(exhibited[1]["x_state_FFR_units"] - exhibited[2]["x_state_FFR_units"])
    y_inf, y_err = 0.02361, 1e-5
    x0_FE, x0_FR = 27.0 / 128.0, 1.5
    verdict = {"x0_ours_FFR_units": x0, "x0_ours_grid_extrapolated": x0_extrap,
               "x_exhibited_state_FFR_units": best_state["x_state_FFR_units"],
               "x_exhibited_state_boxes": best_state["n_boxes"],
               "x_exhibited_state_quadrature_dependence": node_dep,
               "rigorous_lower_bound_x0": best_state["x_state_FFR_units"],
               "rigorous_margin_over_y_inf_plus_err": best_state["x_state_FFR_units"] - (y_inf + y_err),
               "rigorous_ratio_over_y_inf": best_state["x_state_FFR_units"] / y_inf,
               "x0_ours_uncertainty": float(abs(x0_extrap - x0) + 1e-5),
               "phidot2_equals_T00": bool(abs(last["phidot2"]["ratio_FE"] - last["T00"]["ratio_FE"]) < 1e-12),
               "gradphi2_equals_T00": bool(abs(last["gradphi2"]["ratio_FE"] - last["T00"]["ratio_FE"]) < 1e-12),
               "FFR2012_y_inf": y_inf, "FFR2012_y_inf_err": y_err,
               "ratio_ours_over_FFR": x0 / y_inf,
               "within_their_error_plus_our_budget": bool(abs(x0 - y_inf) < y_err + 5e-5),
               "respects_FFR_eq64_y_inf_le_x0": bool(x0 >= y_inf - y_err),
               "respects_Fewster_Eveson": bool(x0 <= x0_FE), "respects_Ford_Roman": bool(x0 <= x0_FR),
               "x0_FE_FFR_units": x0_FE, "x0_FordRoman_FFR_units": x0_FR,
               "FE_over_ours": x0_FE / x0,
               "lattice_route_dev_at_rho_quarter": lattice_vs_cont["dev_aR2_aR3"],
               "reading": ("our exact infimum exceeds their moment estimate by a factor ~2.5 while "
                           "satisfying their rigorous y_inf <= x0 (Eq. 64), the FE and Ford-Roman bounds; "
                           "their abstract states the distributions are not uniquely determined by the "
                           "moments, and their Sec. IV says that in that case y_inf = x_F < x0 -- the "
                           "Friedrichs edge, not the spectral edge. Their Eq. (57) identity "
                           "x0(rho_S) = x0(phidot^2) holds exactly here.")}
    rec = {"_source": "notebook.py::s6e_ffr2012_comparison", "quick": quick, "continuum": cont,
           "exhibited_state_ladder": exhibited,
           "lattice_vs_continuum_rho_quarter": lattice_vs_cont, "verdict": verdict}
    _write_json("s6e_ffr2012_comparison.json", rec)
    return rec


def s6f_ffr_method(quick=False):
    """Fewster-Ford-Roman 2012 reproduced BY THEIR METHOD on our kernel
    (vacuum.inequalities.ffr_moments): exact moments to n = 65 by their
    Appendix A (chain recurrence (A6), run-structure derivation (A8) as a
    flow, M = e^W), Table I; the identity (A4) and our kernel's Nystrom chain
    integrals against the exact ones; their Sec. IV Stieltjes test (Table II)
    and Eq. (68) fit -> y_inf; the phi^2 control (1/6); Carleman/growth
    diagnostics; the point-mass blindness experiment; and the placement of
    the exhibited state and the solver value next to the reproduced y_inf.
    """
    import vacuum.inequalities.ffr_moments as ffr
    from fractions import Fraction as Fr
    import mpmath as mp
    t0 = time.time()
    a3, C3, K3 = ffr.ffr_moment_sequence(3, 65)
    a1, C1, K1 = ffr.ffr_moment_sequence(1, 65)
    t_mom = time.time() - t0
    tabI = {}
    for p_, a in ((3, a3), (1, a1)):
        devs = {int(n): float(mp.mpf(a[n].numerator) / a[n].denominator) / ref - 1 for n, ref in ffr.FFR2012_TABLE_I[p_].items()}
        tabI[p_] = {"worst_rel_dev": max(abs(v) for v in devs.values()), "n_entries": len(devs),
                    "exact_low": {n: str(a[n]) for n in range(2, 6)}}
    a4_checks = [C3[4] == 8**4 * (K3[3] * K3[1] + K3[2]**2 + K3[1]**4),
                 C3[5] == 8**5 * (K3[4] * K3[1] + 3 * K3[3] * K3[2] + 8 * K3[2] * K3[1]**3)]
    ident = [dict(zip(("direct", "closed", "rel"), ffr.ffr_identity_A4(q_, k_))) for q_, k_ in ((3, 0.7), (7, 2.5), (12, 9.0))]
    kc = ffr.kernel_chain_integrals(q_sqrt_lorentzian_f(1.0), 3, 16, grids=(((800, 60.0), (1600, 60.0), (3200, 60.0)) if quick else ((1500, 60.0), (3000, 60.0), (6000, 60.0))))
    t0 = time.time()
    y3 = ffr.stieltjes_sequence(a3, 33, 2, dps=60)
    y1 = ffr.stieltjes_sequence(a1, 33, 2, dps=60)
    t_st = time.time() - t0
    tabII = {p_: max(abs(y[N] - ref) for N, ref in ffr.FFR2012_TABLE_II[p_].items()) for p_, y in ((3, y3), (1, y1))}
    fit3 = ffr.ffr_fit_y_inf(y3, 21, 33)
    Ns = np.arange(21, 34, dtype=float); ys = np.array([y1[int(N)] for N in Ns])
    c1, *_ = np.linalg.lstsq(np.vstack([np.ones_like(Ns), 1 / Ns, 1 / Ns**2]).T, ys, rcond=None)

    acc = ffr.accelerated_sequence(y3)             # labelled as FFR Table III (N = 21..30)
    tabIII_dev = max(abs(acc[N] - ref) for N, ref in ffr.FFR2012_TABLE_III.items())
    car = ffr.carleman_diagnostics(a3, 3, shift=1)
    blind = []
    for w in (Fr(1, 1000), Fr(1, 100), Fr(1, 10)):
        ap = ffr.point_mass_perturbation(a3, Fr(59761, 10**6), w)
        blind.append({"weight": float(w), "x_star": 0.059761, "y_N": {N: ffr.stieltjes_edge(ap, N, 60) for N in (10, 20, 32)}})
    st = box_wavepacket_state(q_sqrt_lorentzian_f(1.0), n_boxes=(300 if quick else 1000), nodes=12, omega_max=40.0)
    x_state = st["ratio_FE"] * 27.0 / 128.0
    solver = worldline_momentum_minimize(q_sqrt_lorentzian_f(1.0), d=4, n=(600 if quick else 1200), omega_max=40.0).ratio * 27.0 / 128.0
    verdict = {"their_method_on_our_kernel_y_inf": fit3["y_inf"], "FFR_y_inf": 0.0236174942666,
               "FFR_y_inf_quoted": ffr.FFR2012_Y_INF, "FFR_err": ffr.FFR2012_Y_INF_ERR,
               "reproduced_within_their_error": bool(abs(fit3["y_inf"] - ffr.FFR2012_Y_INF) < ffr.FFR2012_Y_INF_ERR),
               "table_I_worst_rel_dev": tabI, "table_II_worst_abs_dev": tabII,
               "phi2_control_y_inf": float(c1[0]), "phi2_control_exact": 1 / 6,
               "A11_A12_exact": all(a4_checks), "A4_identity_worst_rel": max(d["rel"] for d in ident),
               "kernel_chain_worst_rel_finest": kc["worst_rel_dev_finest"],
               "kernel_chain_worst_rel_richardson": kc["worst_rel_dev_richardson"],
               "kernel_chain_monotone": kc["all_monotone"],
               "kernel_chain_code_path": kc["code_path"], "kernel_rank_one_residual": kc["rank_one_residual"],
               "table_III_worst_abs_dev": tabIII_dev,
               "moments_provenance": ("the 65 moments fed to the Stieltjes test are the exact (A6)-recurrence "
                                      "values; the identification with OUR operator is the rank-one B, "
                                      "K_n = v^T A'^{n-1} v matching (A6) to n = 16, a_2 = 9/2 and a_3 = 1890 "
                                      "from the kernel, and the phi^2 control"),
               "carleman": {k: v for k, v in car.items() if k not in ("log_a_over_gamma",)},
               "growth_class": "a_n ~ (3n-3)! x %.3f^n (FFR App. B: power x (3n-4)!)" % car["geometric_factor_per_n"],
               "blindness": blind,
               "x_exhibited_state": x_state, "x_solver": solver,
               "margin_state_over_y_inf": x_state - (fit3["y_inf"] + ffr.FFR2012_Y_INF_ERR),
               "ratio_state_over_y_inf": x_state / fit3["y_inf"],
               "same_f_family_point": {"sqrt_lorentzian_C_over_CFE": st["ratio_FE"], "gaussian": 0.45749, "bump_p2": 0.611},
               "reading": ("their number is reproduced by their method on our kernel (moments, Table II, fit); "
                           "the disagreement therefore lives entirely in the extension step y_inf -> x0: the "
                           "65-moment Stieltjes edge of a (3n)!-class sequence is blind to a support edge at "
                           "-0.0598 (point-mass experiment), Carleman is inconclusive, and an exhibited "
                           "physical state sits at -0.0598 -- on their own Sec. IV reasoning the Lorentzian "
                           "phidot^2 moment problem is indeterminate and y_inf is the Friedrichs edge x_F < x0."),
               "seconds": {"moments": t_mom, "stieltjes": t_st}}
    rec = {"_source": "notebook.py::s6f_ffr_method", "quick": quick, "y_N_phidot2": y3, "y_N_phi2": y1,
           "fit_68": fit3, "accelerated_table_III_labels": {N: acc[N] for N in sorted(acc)},
           "kernel_chain": {k: v for k, v in kc.items() if k != "per_grid"}, "A4_identity": ident,
           "moments_phidot2_log10": {n: float(mp.log10(mp.mpf(a3[n].numerator) / a3[n].denominator)) for n in range(2, 66)},
           "verdict": verdict}
    _write_json("s6f_ffr_method.json", rec)
    return rec


def s6g_lorentzian_routes(quick=False):
    """The Ford-Roman weight through every route we have, side by side, in FFR units.

    (i) the lattice-free momentum-space worldline solve (`worldline_momentum_minimize`,
    cusp-limited grid ladder + 1/n^2 Richardson) -- no box, no lattice, so no
    infrared bias can enter it; (ii) the exhibited box-wavepacket state ladder
    (a rigorous lower bound on x0); (iii) the Srednicki-lattice route: C_eff(rho)
    at rho = 1/4, 1/6, 1/8 with n_ball = 3, 4, 6, corrected with the finite-box
    law MEASURED FOR THIS WEIGHT at rho = 1/4 (g^-p with fitted p; the Gaussian's
    -10.04/g^4 does not apply), extrapolated in (a/R)^2 at fixed rho and then
    rho -> 0 with the derived even form (d2 from `rho2_coefficient` for this
    weight), against the continuum ball values at the same rho. The IR-bias
    question is answered by (i) vs (ii) vs (iii).
    """
    import vacuum.inequalities.ffr_moments as ffr
    fh = q_sqrt_lorentzian_f(1.0)
    U = 27.0 / 128.0                                  # C/C_FE -> FFR units for this weight
    # (i) continuum worldline ladder
    ladder = []
    for n in ((600, 800, 1200) if quick else (800, 1200, 1600, 2400)):
        t0 = time.time()
        r = worldline_momentum_minimize(fh, d=4, n=n, omega_max=40.0)
        ladder.append({"n": n, "ratio_FE": r.ratio, "x0": r.ratio * U, "n_kept": r.n_kept, "seconds": time.time() - t0})
    nn = np.array([l["n"] for l in ladder], float); xx = np.array([l["x0"] for l in ladder])
    c2, r2 = _lsq_fit([nn**0, 1.0 / nn**2], xx)
    x0_cont = float(c2[0]); x0_cont_unc = float(abs(c2[0] - xx[-1]))
    # (ii) exhibited state ladder
    states = []
    for nb in ((300, 600) if quick else (300, 1000, 2000)):
        t0 = time.time()
        st = box_wavepacket_state(fh, n_boxes=nb, nodes=12, omega_max=40.0)
        states.append({"n_boxes": nb, "x_state": st["ratio_FE"] * U, "z_norm": st["z_norm_used"], "seconds": time.time() - t0})
    x_state = max(s_["x_state"] for s_ in states)
    # (iii) lattice route
    rhos = (0.25, 1 / 6, 0.125)
    k2 = rho2_coefficient(fh, n=(400 if quick else 800), omega_max=40.0, fd_check=False)
    d2 = k2["k2"]                                     # per unit R^2 at tau = 1 = d2 in rho
    cont_ball = {}
    for rho in rhos:
        t0 = time.time()
        b = ball_momentum_minimize(fh, rho, n=(500 if quick else 800), omega_max=40.0, l_max=20)
        cont_ball[rho] = {"C": b["ratio_FE"], "l_stop": b["l_stop"], "seconds": time.time() - t0}

    def _lat(nb, rho, g):
        tau = (nb + 0.5) / rho
        N = int(nb + math.ceil(g * tau) + 8)
        t0 = time.time()
        out = radial_ball_qei(N, nb, q_sqrt_lorentzian_f(tau), l_max=80, tol=0.0, method="mode_space", rel_tol=1e-12)
        fpp2 = fsecond_sq_integral(q_sqrt_lorentzian_f(tau))
        return {"n_ball": nb, "rho": rho, "tau": tau, "N": N, "g": N / tau,
                "C_raw": float(-out["e_min_total"] / (shell_volume(nb) * fpp2) / FE_CONSTANT_4D),
                "l_stop": int(out["l_stop"]), "converged": bool(out["converged"]), "seconds": time.time() - t0}
    # box law for this weight at rho = 1/4, n_ball = 4 (g ladder), exponent fitted
    sweep = [_lat(4, 0.25, g) for g in ((8, 12, 16, 24) if quick else (8, 12, 16, 24, 32))]
    gg = np.array([r["g"] for r in sweep]); cb = np.array([r["C_raw"] for r in sweep])
    best = None
    for pexp in np.linspace(0.5, 4.5, 81):
        c, r = _lsq_fit([gg**0, gg**-pexp], cb)
        rr = float(np.max(np.abs(r)))
        if best is None or rr < best[0]:
            best = (rr, float(pexp), float(c[0]), float(c[1]))
    box_law = {"exponent": best[1], "B": best[3], "max_resid": best[0]}
    rows = []
    for rho in rhos:
        for nb in ((3, 4) if quick else (3, 4, 6)):
            if rho == 0.25 and nb == 4:
                r = next(x for x in sweep if abs(x["g"] - 16.0 * (1 + 0)) < 1.0 or abs(x["g"] - sweep[2]["g"]) < 1e-9)
                r = sweep[2]
            else:
                r = _lat(nb, rho, BOX_RATIO)
            r = dict(r, C_boxcorr=r["C_raw"] - box_law["B"] * r["g"] ** (-box_law["exponent"]),
                     C_gaussian_law=r["C_raw"] + 10.04 / r["g"] ** 4)
            rows.append(r)
    per_rho = {}
    for rho in rhos:
        sub = [r for r in rows if abs(r["rho"] - rho) < 1e-12]
        aR = np.array([1.0 / (r["n_ball"] + 0.5) for r in sub])
        Cl = np.array([r["C_boxcorr"] for r in sub]); Cg = np.array([r["C_gaussian_law"] for r in sub])
        c, res = _lsq_fit([aR**0, aR**2], Cl)
        cgl, _ = _lsq_fit([aR**0, aR**2], Cg)
        per_rho[rho] = {"C_rho_lattice": float(c[0]), "C_rho_lattice_gaussian_law": float(cgl[0]),
                        "C_cont": cont_ball[rho]["C"], "dev": float(c[0] - cont_ball[rho]["C"]),
                        "dev_gaussian_law": float(cgl[0] - cont_ball[rho]["C"]), "points": sub}
    rr_ = np.array(rhos); cl = np.array([per_rho[r]["C_rho_lattice"] for r in rhos])
    cc = np.array([cont_ball[r]["C"] for r in rhos])
    # rho -> 0 with the derived even form, d2 fixed (3 points: fit C0 and d4), and free d2 (fit C0, d2; 3 points)
    cf, _ = _lsq_fit([rr_**0, rr_**4], cl - d2 * rr_**2)
    cfree, _ = _lsq_fit([rr_**0, rr_**2], cl)
    ccont, _ = _lsq_fit([rr_**0, rr_**4], cc - d2 * rr_**2)
    x0_lat = float(cf[0]) * U; x0_lat_free = float(cfree[0]) * U; x0_cont_from_ball = float(ccont[0]) * U
    verdict = {"x0_continuum_worldline": x0_cont, "x0_continuum_unc": x0_cont_unc,
               "x0_continuum_ball_extrapolated": x0_cont_from_ball,
               "x0_lattice_route": x0_lat, "x0_lattice_route_free_d2": x0_lat_free,
               "x0_exhibited_state": x_state,
               "lattice_minus_continuum": x0_lat - x0_cont, "lattice_rel": (x0_lat - x0_cont) / x0_cont,
               "state_minus_continuum": x_state - x0_cont,
               "box_law_this_weight": box_law, "gaussian_box_law": "-10.04/g^4",
               "per_rho_dev": {f"{r:.4f}": per_rho[r]["dev"] for r in rhos},
               "per_rho_dev_gaussian_law": {f"{r:.4f}": per_rho[r]["dev_gaussian_law"] for r in rhos},
               "d2_this_weight": d2,
               "ir_bias_verdict": ("the continuum route has no box and cannot carry infrared bias; the lattice "
                                   "route agrees with it once the box law measured for THIS weight is used, and "
                                   "misses by ~1% with the Gaussian g^-4 law -- the bias is real on the lattice and "
                                   "absent from the number quoted"),
               "FFR_y_inf": 0.02361}
    rec = {"_source": "notebook.py::s6g_lorentzian_routes", "quick": quick, "continuum_ladder": ladder,
           "exhibited_ladder": states, "continuum_ball": {f"{r:.4f}": v for r, v in cont_ball.items()},
           "box_sweep": sweep, "lattice_rows": rows, "per_rho": {f"{r:.4f}": {k: v for k, v in d.items() if k != "points"} for r, d in per_rho.items()},
           "verdict": verdict}
    _write_json("s6g_lorentzian_routes.json", rec)
    return rec


def _load_archived_s4():
    """The archived S4 tables (scan grid and extrapolation) from data/."""
    import csv
    rows = []
    with open(DATA / "s4d_4d_scan.csv") as fh:
        lines = [l for l in fh if not l.startswith("#")]
    for r in csv.DictReader(lines):
        rows.append({"rho_nominal": float(r["rho_nominal"]), "n_ball": int(r["n_ball"]), "x": float(r["x"]),
                     "ratio_shell_FE": float(r["ratio_shell_FE"]), "tau": float(r["tau"])})
    ext = json.loads((DATA / "s4e_extrapolation.json").read_text())
    return rows, ext


def run_s6(quick=False, scan_rows=None, ext=None):
    print("\n[S6] the same problem in the continuum: momentum space, no lattice")
    c2 = s6a_momentum_2d_control(grids=((300, 12.0), (600, 30.0)) if quick else ((400, 16.0), (800, 30.0), (1200, 100.0)))
    for name, rows in c2["rows"].items():
        print(f"     [S6a] {name:<16s} 2d ratio " + "  ".join(f"{r['ratio']:.7f}(n={r['n']},Om={r['omega_max']:g})" for r in rows))
    print(f"     [S6a] worst |ratio-1| over families without a zero of f: {c2['worst_dev_no_zero_families']:.1e}; "
          f"f with a zero (c=-0.3): {c2['modgauss_c-0.3_ratio']:.5f}")
    c4 = s6b_momentum_4d_worldline(grids=((300, 12.0), (400, 14.0)) if quick else ((300, 12.0), (400, 14.0), (500, 16.0), (600, 20.0), (800, 24.0)))
    for r in c4["rows"]:
        print(f"     [S6b] n={r['n']:4d} Om={r['omega_max']:5.1f}: C_4d/C_FE = {r['ratio']:.9f}  kept {r['n_kept']}")
    print(f"     [S6b] ==> C_4d/C_FE = {c4['C_4d_over_FE']:.7f} (grid {c4['grid_uncertainty']:.0e}); l=0 share {c4['l0_share']:.3f}; "
          f"C_4d = {c4['C_4d']:.5e} = 1/({c4['one_over_C_4d_pi2']:.3f} pi^2)")
    if scan_rows is None:
        scan_rows, ext = _load_archived_s4()
    ce = s6c_rho_expansion(scan_rows, ext, n=(300 if quick else 400), om=(12.0 if quick else 14.0),
                           rhos=((0.02, 0.05, 0.1, 0.125, 1 / 6, 0.2, 0.25, 1 / 3, 0.4, 0.5) if quick else
                                 (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.125, 0.15, 1 / 6, 0.2, 0.25, 1 / 3, 0.4, 0.5)))
    v = ce["verdict"]
    print(f"     [S6c] d2 derived = {v['d2_derived']:+.6f} = HF(l=0,1) {v['d2_hf_l01']:+.6f} + l=2 {v['d2_l2']:+.6f}; "
          f"numerical small-rho {v['d2_numerical_small_rho']:+.6f}; archived rho^2+rho^4 fit {v['d2_archived_fit_r2_r4']:+.4f}")
    print(f"     [S6c] parity: even fit resid {v['even_fit_residual_small_rho']:.1e}; free rho^3 {v['odd_term_coefficient']:+.1e}, "
          f"rho {v['linear_term_coefficient']:+.1e}, rho^2 log rho {v['log_term_coefficient']:+.1e}")
    for key, row in ce["lattice_minus_continuum"].items():
        print(f"     [S6c] rho={float(key):.4f}: lattice - continuum, n=2..8: " +
              " ".join(f"{d:+.2e}" for d in row["dev"]) + f"  log-log slope {row['loglog_slope']:.2f}; "
              f"e1,e2,e3 = {row['fit_e1_e2_e3']['e1']:+.1e}, {row['fit_e1_e2_e3']['e2']:+.4f}, {row['fit_e1_e2_e3']['e3']:+.4f}")
    print(f"     [S6c] lattice error: leading (a/R)^2 (log-log slopes {v['lattice_error_loglog_slope_range'][0]:.2f}-{v['lattice_error_loglog_slope_range'][1]:.2f}, "
          f"linear coefficient {v['lattice_error_linear_coefficient_range'][0]:+.1e}..{v['lattice_error_linear_coefficient_range'][1]:+.1e}); "
          f"global e2 {v['lattice_error_global_fit']['e2_aR2']:+.4f} e3 {v['lattice_error_global_fit']['e3_aR3']:+.4f} c(a/tau)^2 {v['lattice_error_global_fit']['c_atau2']:+.4f}")
    for key, d in ce["reextrapolation_archived_tables"].items():
        print(f"     [S6c] archived {key:<12s} with the derived even form: " +
              "  ".join(f"{m} {x['C_4d_over_FE']:.6f}" for m, x in d.items()))
    for key, d in ce["lattice_only"].items():
        print(f"     [S6c] raw grid refit ({key}): even {d['even_r2_r4']:.6f} / {d['even_r2_r4_r6']:.6f}"
              f"  [d2-pinned, NOT independent: {d['d2_fixed_NOT_INDEPENDENT']:.6f}]")
    print(f"     [S6c] independent values: " + ", ".join(f"{x:.6f}" for x in v["lattice_only_values_independent"])
          + "; d2-pinned (excluded): " + ", ".join(f"{x:.6f}" for x in v["lattice_only_values_d2_pinned_NOT_INDEPENDENT"]))
    for nm, exc in v["lattice_only_excluded_systematics"].items():
        print(f"     [S6c] NOT summed: {nm} = {exc['value']:.1e} -- {exc['reason'][:96]}...")
    print(f"     [S6c] ==> lattice-only C_4d/C_FE = {v['lattice_only_C_4d_over_FE']:.6f} +- {v['lattice_only_uncertainty']:.6f} "
          f"({v['lattice_only_conservative_uncertainty']:.6f} conservative) "
          f"(budget {v['lattice_only_budget']}); continuum {v['C_4d_over_FE_continuum']:.7f}; "
          f"difference {v['lattice_only_minus_continuum']:+.1e} = {v['lattice_only_minus_continuum_in_sigma']:.2f} of its own error; "
          f"archived {v['archived_C_4d_over_FE']:.4f} +- {v['archived_uncertainty']:.4f} "
          f"contains the continuum: {v['continuum_inside_archived_band']}")
    fam = s6d_families_4d(quick=quick)
    for name, r in fam["ranking"]:
        d = fam["families"][name]
        extra = f"  (+-{d['unc']:.0e})" if "unc" in d else ""
        print(f"     [S6d] {name:<16s} C_sharp(f)/C_FE = {r:.5f}{extra}   2d control {d.get('ratio_2d_control', float('nan')):.6f}")
    sl = fam["families"]["sqrt_lorentzian"]
    print(f"     [S6d] Ford-Roman weight in FFR-2012 units: x0 = {sl['x0_FFR_units']:.4f} (FE 27/128 = {sl['x0_FE_FFR_units']:.4f}, "
          f"Ford-Roman 3/2, FFR-2012 moment estimate {sl['x0_FFR2012_estimate']}, ratio {sl['x0_over_FFR2012']:.2f})")
    print("     [S6d] (1-u^2)^p: " + "  ".join(f"p={s['p']:g}: {s['ratio']:.4f}" for s in fam["families"]["bump_p_scan"]))
    for k, o in fam["families"]["optimizations"].items():
        print(f"     [S6d] optimize {k:<12s}: theta* {np.round(o['theta_star'], 4).tolist()} ratio {o.get('ratio', o.get('ratio_at_grid')):.5f} "
              f"2d control {o['ratio_2d_control']:.6f}")
    print(f"     [S6d] ==> Gaussian optimal in 4d: {fam['gaussian_optimal']}; sup over tested families {fam['sup_over_tested_families']:.4f} "
          f"({fam['sup_family']}); all below the FE bound: {fam['all_below_FE']}")
    fe = s6e_ffr2012_comparison(quick=quick)
    fv = fe["verdict"]
    ff = s6f_ffr_method(quick=quick)
    ffv = ff["verdict"]
    print(f"     [S6f] their method on our kernel: y_inf = {ffv['their_method_on_our_kernel_y_inf']:.7f} vs FFR 0.0236175 "
          f"(reproduced: {ffv['reproduced_within_their_error']}); Table I/II worst {ffv['table_I_worst_rel_dev'][3]['worst_rel_dev']:.0e}/{ffv['table_II_worst_abs_dev'][3]:.0e}; "
          f"exhibited state {ffv['x_exhibited_state']:.5f} = y_inf x {ffv['ratio_state_over_y_inf']:.2f}")
    print(f"     [S6e] exhibited state: rigorous x0 >= {fv['rigorous_lower_bound_x0']:.5f} "
          f"(> y_inf + err by {fv['rigorous_margin_over_y_inf_plus_err']:.4f})")
    print(f"     [S6e] FFR-2012 acceptance test, Ford-Roman weight: x0(ours) = {fv['x0_ours_FFR_units']:.5f} vs "
          f"y_inf = {fv['FFR2012_y_inf']} +- {fv['FFR2012_y_inf_err']} (ratio {fv['ratio_ours_over_FFR']:.2f}; "
          f"phidot^2 == T00: {fv['phidot2_equals_T00']}; Eq.(64)/FE/FR respected: "
          f"{fv['respects_FFR_eq64_y_inf_le_x0']}/{fv['respects_Fewster_Eveson']}/{fv['respects_Ford_Roman']}; "
          f"lattice-vs-continuum at rho=1/4: {fv['lattice_route_dev_at_rho_quarter']:+.1e})")
    return {"S6a": {k: v for k, v in c2.items() if k != "rows"}, "S6b": {k: v for k, v in c4.items() if k != "rows"},
            "S6e_ffr2012_verdict": fv, "S6f_ffr_method_verdict": ffv,
            "S6c_verdict": v, "S6d": {"ranking": fam["ranking"], "gaussian_optimal": fam["gaussian_optimal"],
                                      "sup_over_tested_families": fam["sup_over_tested_families"],
                                      "optimizations": fam["families"]["optimizations"]}}


# =========================================================================
# io helpers
# =========================================================================


def _write_csv(name, rows, source):
    DATA.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    lines = [f"# _source: {source}", ",".join(keys)]
    for r in rows:
        lines.append(",".join(
            ("" if r.get(k) is None else
             (f"{r[k]!r}".strip("'") if isinstance(r[k], str) else
              (f"{r[k]:d}" if isinstance(r[k], (int, np.integer)) and
               not isinstance(r[k], bool) else
               (str(bool(r[k])) if isinstance(r[k], (bool, np.bool_)) else
                f"{float(r[k]):.12g}"))))
            for k in keys))
    (DATA / name).write_text("\n".join(lines) + "\n")


def _write_json(name, obj):
    DATA.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    (DATA / name).write_text(json.dumps(obj, indent=2, default=_default) + "\n")


# =========================================================================
# main
# =========================================================================


def main(argv=None):
    global DATA
    ap = argparse.ArgumentParser(description="qei-2d-sharp-constant")
    ap.add_argument("--quick", action="store_true",
                    help="smoke configuration: reduced 2d axes, the S4 systematics at small size, "
                         "no 4d scan / extrapolation / extremal state (~1 min)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: data/; a fresh temporary directory for --quick, "
                         "so a smoke run never overwrites the archive)")
    ap.add_argument("--section", default=None, choices=["S6", "S6c", "S6e", "S6f", "S6g"],
                    help="run one section alone against the archived tables in data/ (S6 writes s6*.json "
                         "and merges its headline into summary.json; S6c re-runs only the R/tau-expansion "
                         "section, rewriting s6c_rho_expansion.json and the S6c entry of summary.json)")
    args = ap.parse_args(argv)
    quick = bool(args.quick)
    if args.out:
        DATA = Path(args.out)
    elif quick:
        DATA = Path(tempfile.mkdtemp(prefix="qei-2d-sharp-constant-quick-"))
    t_start = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    stamps = {}

    def mark(tag):
        stamps[tag] = time.time() - t_start

    if args.section == "S6g":
        g6 = s6g_lorentzian_routes(quick=quick)
        v = g6["verdict"]
        print(f"[S6g] Ford-Roman weight, FFR units: continuum worldline x0 = {v['x0_continuum_worldline']:.6f} (+-{v['x0_continuum_unc']:.0e}); "
              f"continuum ball rho->0 {v['x0_continuum_ball_extrapolated']:.6f}; lattice route {v['x0_lattice_route']:.6f} "
              f"(free d2 {v['x0_lattice_route_free_d2']:.6f}); exhibited state {v['x0_exhibited_state']:.6f}")
        print(f"[S6g] lattice - continuum = {v['lattice_minus_continuum']:+.2e} ({v['lattice_rel']*100:+.3f}%); per-rho devs {v['per_rho_dev']} "
              f"(Gaussian box law instead: {v['per_rho_dev_gaussian_law']}); box law p = {v['box_law_this_weight']['exponent']:.2f}, B = {v['box_law_this_weight']['B']:+.3f}")
        print(f"[S6g] {v['ir_bias_verdict']}")
        summ_path = DATA / "summary.json"
        summary = json.loads(summ_path.read_text()) if summ_path.exists() else {"_source": "notebook.py::main"}
        summary.setdefault("S6_momentum_space", {})["S6g_lorentzian_routes_verdict"] = v
        _write_json("summary.json", summary)
        print(f"Artifacts written to {DATA}")
        return summary

    if args.section == "S6f":
        ff = s6f_ffr_method(quick=quick)
        v = ff["verdict"]
        print(f"[S6f] THEIR METHOD ON OUR KERNEL: y_inf = {v['their_method_on_our_kernel_y_inf']:.10f} vs FFR {v['FFR_y_inf']:.10f} "
              f"(quoted {v['FFR_y_inf_quoted']} +- {v['FFR_err']}): reproduced {v['reproduced_within_their_error']}")
        print(f"[S6f] Table I worst rel dev: p=3 {v['table_I_worst_rel_dev'][3]['worst_rel_dev']:.1e}, p=1 {v['table_I_worst_rel_dev'][1]['worst_rel_dev']:.1e}; "
              f"Table II worst abs dev: p=3 {v['table_II_worst_abs_dev'][3]:.1e}, p=1 {v['table_II_worst_abs_dev'][1]:.1e}; phi^2 control y_inf {v['phi2_control_y_inf']:.12f}")
        print(f"[S6f] (A11),(A12) exact: {v['A11_A12_exact']}; (A4) identity worst rel {v['A4_identity_worst_rel']:.1e}; our-kernel chain K_n: finest {v['kernel_chain_worst_rel_finest']:.1e}, "
              f"Richardson {v['kernel_chain_worst_rel_richardson']:.1e}, monotone {v['kernel_chain_monotone']}")
        print(f"[S6f] growth {v['growth_class']}; Carleman {v['carleman']['verdict']}; Stieltjes sum {v['carleman']['stieltjes_partial_sum']:.3f} (terms ~ n^-{v['carleman']['stieltjes_term_decay_exponent']:.2f})")
        for b in v["blindness"]:
            print(f"[S6f] point mass w={b['weight']} at -{b['x_star']}: y_N = " + ", ".join(f"N={N}: {y:.6f}" for N, y in b["y_N"].items()))
        print(f"[S6f] exhibited state x = {v['x_exhibited_state']:.6f} (solver {v['x_solver']:.6f}) vs reproduced y_inf {v['their_method_on_our_kernel_y_inf']:.6f}: "
              f"margin {v['margin_state_over_y_inf']:+.4f} (x{v['ratio_state_over_y_inf']:.2f}); same f: C/C_FE = {v['same_f_family_point']['sqrt_lorentzian_C_over_CFE']:.4f}")
        summ_path = DATA / "summary.json"
        summary = json.loads(summ_path.read_text()) if summ_path.exists() else {"_source": "notebook.py::main"}
        summary.setdefault("S6_momentum_space", {})["S6f_ffr_method_verdict"] = v
        _write_json("summary.json", summary)
        print(f"Artifacts written to {DATA}")
        return summary

    if args.section == "S6e":
        fe = s6e_ffr2012_comparison(quick=quick)
        v = fe["verdict"]
        print(f"[S6e] x0(ours) = {v['x0_ours_FFR_units']:.5f} (grid-extrapolated {v['x0_ours_grid_extrapolated']:.5f}) "
              f"vs FFR-2012 y_inf = {v['FFR2012_y_inf']} +- {v['FFR2012_y_inf_err']}: ratio {v['ratio_ours_over_FFR']:.3f}; "
              f"within their error + our budget: {v['within_their_error_plus_our_budget']}")
        print(f"[S6e] EXHIBITED STATE ({v['x_exhibited_state_boxes']} boxes): <x> = {v['x_exhibited_state_FFR_units']:.6f} "
              f"(node dependence {v['x_exhibited_state_quadrature_dependence']:.0e}) => rigorous x0 >= {v['rigorous_lower_bound_x0']:.6f}, "
              f"margin over y_inf + err {v['rigorous_margin_over_y_inf_plus_err']:+.4f} (x{v['rigorous_ratio_over_y_inf']:.2f})")
        print(f"[S6e] phidot^2 == T00: {v['phidot2_equals_T00']}, (grad phi)^2 == T00: {v['gradphi2_equals_T00']}; "
              f"respects Eq.(64): {v['respects_FFR_eq64_y_inf_le_x0']}, FE: {v['respects_Fewster_Eveson']} (FE/ours {v['FE_over_ours']:.2f}), FR: {v['respects_Ford_Roman']}")
        lv = fe["lattice_vs_continuum_rho_quarter"]
        bl = lv["box_law_measured"]
        print("[S6e] box sweep (n_ball=4): " + ", ".join(f"g={r['g']:.1f}: {r['C_raw']:.6f}" for r in lv["box_sweep_nb4"]))
        print(f"[S6e] box law for this weight: C = C_inf + B g^-p with p = {bl['exponent']:.2f}, B = {bl['B']:+.4f} "
              f"(resid {bl['max_resid']:.1e}); fixed-exponent fits: " +
              ", ".join(f"{k}: {v['C_inf']:.6f} (res {v['max_resid']:.0e})" for k, v in lv["box_fits"].items()))
        print(f"[S6e] lattice rho=1/4 (measured box law): " + ", ".join(f"n={r['n_ball']}: {r['C_boxcorr']:.6f}" for r in lv["rows"])
              + f" -> (a/R)^2 {lv['lattice_extrap_aR2']:.6f} / +(a/R)^3 {lv['lattice_extrap_aR2_aR3']:.6f}; continuum {lv['C_cont_rho']:.6f}; "
              f"dev {lv['dev_aR2']:+.1e} / {lv['dev_aR2_aR3']:+.1e}; with the GAUSSIAN box law instead: dev {lv['dev_with_gaussian_box_law']:+.1e}")
        summ_path = DATA / "summary.json"
        summary = json.loads(summ_path.read_text()) if summ_path.exists() else {"_source": "notebook.py::main"}
        summary.setdefault("S6_momentum_space", {})["S6e_ffr2012_verdict"] = v
        _write_json("summary.json", summary)
        print(f"Artifacts written to {DATA}")
        return summary

    if args.section == "S6c":
        scan_rows, ext_arch = _load_archived_s4()
        ce = s6c_rho_expansion(scan_rows, ext_arch, n=(300 if quick else 400), om=(12.0 if quick else 14.0))
        v = ce["verdict"]
        print("[S6c] independent values: " + ", ".join(f"{x:.6f}" for x in v["lattice_only_values_independent"]))
        print("[S6c] d2-pinned (excluded, not independent): "
              + ", ".join(f"{x:.6f}" for x in v["lattice_only_values_d2_pinned_NOT_INDEPENDENT"]))
        for nm, exc in v["lattice_only_excluded_systematics"].items():
            print(f"[S6c] NOT summed: {nm} = {exc['value']:.2e}")
        print(f"[S6c] ==> lattice-only {v['lattice_only_C_4d_over_FE']:.6f} +- {v['lattice_only_uncertainty']:.6f} "
              f"(conservative {v['lattice_only_conservative_uncertainty']:.6f}); budget {v['lattice_only_budget']}")
        print(f"[S6c] continuum {v['C_4d_over_FE_continuum']:.7f}; honest agreement "
              f"{v['lattice_only_minus_continuum']:+.2e} = {v['lattice_only_minus_continuum_in_sigma']:.2f} sigma")
        summ_path = DATA / "summary.json"
        summary = json.loads(summ_path.read_text()) if summ_path.exists() else {"_source": "notebook.py::main"}
        summary.setdefault("S6_momentum_space", {})["S6c_verdict"] = v
        _write_json("summary.json", summary)
        print(f"Artifacts written to {DATA}")
        return summary

    if args.section == "S6":
        s6 = run_s6(quick=quick)
        total = time.time() - t_start
        summ_path = DATA / "summary.json"
        summary = json.loads(summ_path.read_text()) if summ_path.exists() else {"_source": "notebook.py::main"}
        summary["S6_momentum_space"] = s6
        summary["S6_wall_seconds"] = total
        _write_json("summary.json", summary)
        print(f"\nS6 wall time: {total:.1f} s  ({total / 60:.2f} min)")
        print(f"Artifacts written to {DATA}")
        return summary

    print("=" * 74)
    print("papers/qei-2d-sharp-constant — pulled-back sampling operator")
    print("=" * 74)

    info = s0_provenance()
    print(f"[S0] {info['date_utc']}  python {info['python']}  numpy {info['numpy']}")
    print(f"     sharp 2d constant 1/(6 pi) = {SHARP_CONSTANT_2D:.12f}")
    mark("S0")

    print("\n[S1] sampling-family self-checks (analytic vs finite difference)")
    for r in s1_family_selfcheck():
        print(f"     {r['label']:<28s} f' dev {r['dev_fprime']:.2e}"
              + (f"   int f'^2 rel dev {r['fp2_rel_dev']:.2e}"
                 if r["fp2_analytic"] is not None else ""))
    mark("S1")

    print("\n[S2a] a/tau0 axis, Gaussian, box ratio g = 100")
    rows_a = s2a_a_axis(taus=(6.0, 8.0, 10.0, 13.0), g=40.0) if quick else s2a_a_axis()
    for r in rows_a:
        print(f"     tau0 ={r['tau0']:5.1f}  N ={r['N']:5d}  n_sup ={r['n_support']:4d}"
              f"  ratio = {r['ratio']:.8f}   1-r = {1 - r['ratio']:.3e}")
    rich = s2_single_axis_richardson(rows_a)
    print(f"     single-axis Richardson r_inf = {rich['r_inf']:.8f} "
          f"(residual {rich['residual']:.2e})")
    mark("S2a")

    print("\n[S2b] box axis at tau0 = 8")
    rows_b = s2b_box_axis(gs=(20.0, 28.0, 40.0, 56.0)) if quick else s2b_box_axis()
    for r in rows_b:
        print(f"     g ={r['g']:6.1f}  N ={r['N']:5d}  ratio = {r['ratio']:.8f}"
              f"   1-r = {1 - r['ratio']:.3e}")
    fit = s2_two_parameter_fit(rows_a, rows_b)
    print(f"     model 1-r = A/g^2 + c/tau0^2 :  A = {fit['A_box']:.4f}, "
          f"c = {fit['c_lattice']:.5f}")
    print(f"     max |residual| = {fit['max_abs_residual']:.2e} "
          f"({fit['max_rel_residual'] * 100:.2f}% of 1-r)")
    print(f"     free continuum offset b0 = {fit['free_offset_b0']:+.3e} "
          f"+- {fit['free_offset_b0_stderr']:.1e} "
          f"({fit['free_offset_b0_over_stderr']:+.2f} sigma)")
    print(f"     ==> recovered ratio = {fit['r_extrap_continuum']:.8f}, "
          f"|r - 1| = {fit['recovery_error_vs_sharp']:.2e}")
    print(f"     with an extra d/tau0^4 artifact term: b0 = "
          f"{fit['quartic_model_b0']:+.3e} +- {fit['quartic_model_b0_stderr']:.1e} "
          f"({fit['quartic_model_b0_over_stderr']:+.2f} sigma), recovered ratio "
          f"{fit['quartic_model_r_extrap']:.8f}")
    mark("S2b")

    print("\n[S2c] mass axis at tau0 = 8")
    mass = s2c_mass_axis(mus=(0.08, 0.04, 0.02), g=40.0) if quick else s2c_mass_axis()
    for r in mass["rows"]:
        print(f"     mu = m*tau0 = {r['mu']:.4f}   ratio = {r['ratio']:.8f}")
    print(f"     direct m = 0: {mass['r_direct_m0']:.8f};  mu->0 Richardson "
          f"{mass['r_extrap_mu0']:.8f} (p = {mass['exponent_p']:.2f}), "
          f"dev {mass['dev_extrap_vs_direct']:.2e}")
    mark("S2c")

    print("\n[S2d] four f-families (wall ratio 12 sweep + wall scan each)")
    _, fits = s2d_families(wall_ratio=6.0, box_scan_walls=(6.0, 9.0)) if quick else s2d_families()
    for name, fv in fits.items():
        print(f"     {name:<9s} r_inf = {fv['r_inf']:.8f}   b0 = {fv['b0']:+.2e}"
              f" +- {fv['b0_stderr']:.1e}   A_wall = {fv['A_wall']:.3f}"
              f"   c_scale = {fv['c_scale']:.4f}")
    fam_spread = max(abs(v["r_inf"] - 1.0) for v in fits.values())
    print(f"     worst family deviation from the sharp constant: {fam_spread:.2e}")
    mark("S2d")

    print("\n[S2e] optimize_sampling, 2-parameter modulated-Gaussian family")
    opt = s2e_optimize(N=121, site=60) if quick else s2e_optimize()
    print(f"     theta* = (tau {opt['theta_star'][0]:.4f}, c {opt['theta_star'][1]:+.4f})"
          f"   ratio* = {opt['ratio_star']:.8f}  ({opt['n_iter']} iters, "
          f"converged {opt['converged']})")
    print(f"     plain re-run of theta*: {opt['ratio_replay_independent_grid']:.8f} "
          f"(dev {opt['replay_dev']:.2e})")
    mark("S2e")

    print("\n[S3] extremal states exhibited")
    ext = s3_extremal_states()
    for r in ext:
        print(f"     {r['name']:<14s} ratio {r['ratio']:.6f}  purity dev "
              f"{r['purity_dev_max']:.2e}  gap/|E| {r['gap_rel']:.2e}  "
              f"min <:h(t):> {r['min_time_profile']:+.4e}")
        print(f"                    sigma(O_f) in [{r['sigma_min']:.3e}, "
              f"{r['sigma_max']:.3e}]   truncated-op dev "
              f"{r['truncated_operator_dev']:.1e}   support-truncation rel dev "
              f"{r['support_truncation_rel_dev']:.1e}")
    mark("S3")

    print("\n[S4] 4d Srednicki-lattice scan with the cancellation-free minimizer")
    nf = s4a_noise_floor_revisited(N=100, n_ball=2, tau=6.0, l_max=4) if quick else s4a_noise_floor_revisited()
    print(f"     [S4a] archived floor config (N={nf['N']}, n_ball={nf['n_ball']}, "
          f"tau={nf['tau']}): Williamson-route floor {nf['old_route_floor_abs']:.1e} abs "
          f"({nf['old_route_floor_rel_to_trace']:.1e} of Tr) on its cone support, "
          f"{nf['old_route_full_support_floor_abs']:.1e} on the full support; mode-space floor "
          f"(worst sector) {nf['new_route_floor_abs_worst_sector']:.1e}: "
          f"{nf['digits_recovered_per_sector']:.1f} digits/sector ({nf['digits_recovered_vs_full_support_float64']:.1f} "
          "vs float64 alone)")
    for r in nf["rows"]:
        print(f"           l={r['l']:2d}  old(cone) {r['e_old_cone']:+.3e}  old(full) {r['e_old_full']:+.3e}  "
              f"new(window) {r['e_new_window']:+.3e}  new(exact) {r['e_new']:+.3e}  "
              f"floor-spread {r['floor_spread']:.1e}  kept {r['n_kept']}")
    mpc = s4b_mp_crosscheck(N=32, n_ball=2, tau=6.0, ls=(2,), dps=20) if quick else s4b_mp_crosscheck()
    for r in mpc["rows"]:
        print(f"     [S4b] l={r['l']}: old {r['e_old']:+.9e} new {r['e_new']:+.9e} mp {r['e_mp']:+.9e}"
              f"  |old-mp| {r['abs_err_old']:.1e} |new-mp| {r['abs_err_new']:.1e} (check limit "
              f"{r['mp_sensitivity_bound']:.0e}) -> {r['digits_recovered_certified']:.1f} digits certified "
              f"({r['seconds_mp']:.0f}s, dim {r['dim']})")
    vol = s4c_volume_measure(n_max=4) if quick else s4c_volume_measure()
    print("     [S4c] V_naive/V_shell - 1 = -3/(2n) + O(1/n^2):  " +
          "  ".join(f"n={r['n_ball']}: {r['naive_over_shell_minus_1']:+.3f}" for r in vol))
    if quick:
        box = {"B_mean": float("nan"), "B_spread": float("nan"), "grid_correction_typical": float("nan"),
               "grid_correction_uncertainty": 0.0, "rows": [], "skipped": "--quick"}
        rows4, wall = [], {"skipped": "--quick"}
        ext = {"order1_fits": {}, "order2_fits": {}, "verdict": {"skipped": "--quick"}}
        ex4 = {"skipped": "--quick"}
        print("     [S4d-S4f] box sweep, 4d scan, extrapolation and extremal state are skipped in "
              "--quick (the full run is the archive)")
        mark("S4")
    else:
        box = s4d_box_sweep()
        for r in box["rows"]:
            print(f"     [S4d] box sweep n_ball={r['n_ball']} rho={r['rho']:.3f}: C(g) = C_inf + A/g^2 + B/g^4 -> "
                  f"A {r['fit_g2_g4']['A']:+.4f} B {r['fit_g2_g4']['B']:+.2f} (resid {r['fit_g2_g4']['max_resid']:.0e}); "
                  f"pure B/g^4: B {r['fit_g4']['B']:+.2f} (resid {r['fit_g4']['max_resid']:.0e}); "
                  f"pure A/g^2 resid {r['fit_g2']['max_resid']:.0e}")
        print(f"     [S4d] B = {box['B_mean']:+.2f} (spread {box['B_spread']:.2f}); grid at g = {BOX_RATIO:g}: "
              f"correction {box['grid_correction_typical']:+.1e} +- {box['grid_correction_uncertainty']:.0e}")
        rows4, wall = s4d_scan(box)
        print(f"     [S4d] {len(rows4)} grid points; wall check g=16 vs 24: rel dev {wall['rel_dev_raw']:.1e} "
              f"(g^-4 law predicts {wall['rel_dev_predicted_by_g4_law']:.1e})")
        for r in rows4:
            print(f"           rho={r['rho']:.4f} nb={r['n_ball']} tau={r['tau']:5.1f} N={r['N']:3d} "
                  f"l_stop={r['l_stop']:2d} conv={r['converged']} tail {r['tail_frac']:.0e}  "
                  f"C/C_FE shell {r['ratio_shell_FE']:.6f} naive {r['ratio_naive_FE']:.6f}  "
                  f"maxres {r['max_residual']:.0e} {r['seconds']:.1f}s")
        ext = s4e_extrapolation(rows4, box_unc=box["grid_correction_uncertainty"])
        for o in ext["order1_rows"]:
            print(f"     [S4e] order 1, rho={o['rho']:.4f}: C_rho/C_FE = {o['C_rho']:.6f} +- {o['C_rho_stderr']:.1e} "
                  f"(resid {o['max_resid']:.1e}; linear {o['C_rho_linear']:.6f}, cubic {o['C_rho_cubic']:.6f}; "
                  f"naive-shell lin/quad/cub {o['naive_minus_shell_linear']:+.4f}/"
                  f"{o['naive_minus_shell_quadratic']:+.4f}/{o['naive_minus_shell_cubic']:+.4f})")
        for k, v in ext["order1_fits"].items():
            print(f"           rho -> 0 model {k:<18s}: {v['C_4d_over_FE']:.6f} +- {v['stderr']:.1e} (resid {v['max_resid']:.1e})")
        for o in ext["order2_rows"]:
            print(f"     [S4e] order 2, n_ball={o['n_ball']}: C_n/C_FE = {o['C_n']:.6f} +- {o['C_n_stderr']:.1e} "
                  f"(resid {o['max_resid']:.1e}; even {o['C_n_even']:.6f}, cubic {o['C_n_cubic']:.6f})")
        for k, v in ext["order2_fits"].items():
            print(f"           1/n -> 0 model {k:<18s}: {v['C_4d_over_FE']:.6f} +- {v['stderr']:.1e} (resid {v['max_resid']:.1e})")
        vd = ext["verdict"]
        print(f"     ==> C_4d / C_FE = {vd['C_4d_over_FE']:.5f} +- {vd['uncertainty_abs']:.5f} "
              f"({vd['uncertainty_rel'] * 100:.2f}%): best models {vd['best_model_order1']}/{vd['best_model_order2']}, "
              f"order dependence {vd['order_dependence_rel'] * 100:.2f}%, "
              f"model spread {vd['model_spread_rel'] * 100:.2f}%, a/tau-fit model dep {vd['a_over_tau_fit_model_dependence']:.1e}, "
              f"max fit stderr {vd['max_fit_stderr']:.1e}, box residual {vd['box_correction_residual']:.0e}; "
              f"C_4d = {vd['C_4d']:.4e} = 1/({vd['one_over_C_4d_pi2']:.2f} pi^2); "
              f"<= 2%: {vd['converged_2pct']}")
        ex4 = s4f_extremal_4d()
        print(f"     [S4f] extremal state at rho={ex4['rho']}, n_ball={ex4['n_ball']} (tau={ex4['tau']}, N={ex4['N']}): "
              f"C/C_FE {ex4['ratio_shell_FE']:.6f}, {ex4['l_stop'] + 1} sectors")
        for r in ex4["sectors"]:
            print(f"           l={r['l']}: e_min {r['e_min']:+.4e} purity {r['purity_dev_max']:.0e} gap {r['gap_rel']:.0e} "
                  f"r_max {r['squeeze_r_max']:.3f} (#r>0.1: {r['n_modes_r_gt_0.1']}) kept {r['n_kept']}")
        mark("S4")

    print("\n[S5] audit battery at paper resolution")
    aud = s5_audit_battery(tau0=4.0, g=40.0) if quick else s5_audit_battery()
    print(f"     {aud['n_states']} states x 2 constants on N = {aud['N']}: "
          f"all passed = {aud['all_passed']}, worst margin "
          f"{aud['worst_margin']:.4e}")
    mark("S5")

    if quick:
        s6 = run_s6(quick=True, scan_rows=None, ext=None) if (DATA / "s4d_4d_scan.csv").exists() else None
        if s6 is None:
            arch_rows, arch_ext = None, None
            try:
                _saved = DATA
                globals()["DATA"] = HERE / "data"
                arch_rows, arch_ext = _load_archived_s4()
            finally:
                globals()["DATA"] = _saved
            s6 = run_s6(quick=True, scan_rows=arch_rows, ext=arch_ext)
    else:
        s6 = run_s6(quick=False, scan_rows=rows4, ext=ext)
    mark("S6")

    total = time.time() - t_start
    summary = {
        "_source": "notebook.py::main",
        "quick": quick,
        "wall_seconds_total": total,
        "wall_seconds_by_section": stamps,
        "S2_richardson_single_axis_r_inf": rich["r_inf"],
        "S2_two_parameter_fit": fit,
        "S2_mass_axis": {k: v for k, v in mass.items() if k != "rows"},
        "S2d_family_fits": fits,
        "S2d_worst_family_deviation": fam_spread,
        "S2e_optimize": opt,
        "S3_extremal": ext,
        "S4a_noise_floor": {k: v for k, v in nf.items() if k != "rows"},
        "S4b_mp_crosscheck": mpc,
        "S4d_box_sweep": {k: v for k, v in box.items() if k != "rows"},
        "S4d_wall_check": wall,
        "S4e_extrapolation": {"order1_fits": ext["order1_fits"],
                              "order2_fits": ext["order2_fits"],
                              "verdict": ext["verdict"]},
        "S4f_extremal_4d": {k: v for k, v in ex4.items() if k != "sectors"},
        "S5_audit": {k: v for k, v in aud.items() if k != "rows"},
        "S6_momentum_space": s6,
    }
    _write_json("summary.json", summary)
    print(f"\nTotal wall time: {total:.1f} s  ({total / 60:.2f} min)")
    print(f"Artifacts written to {DATA}")
    return summary


if __name__ == "__main__":
    main()
