"""Optimizers and constraints for the differentiable vacuum (Layer 6).

``optimize(f, C, theta0, method)`` minimizes a scalar loss over a flat
parameter vector theta with one of three drivers:

- ``'adam'``   — Adam (Kingma & Ba, ICLR 2015, arXiv:1412.6980, Algorithm 1)
  on the JAX gradient, with optional box projection after every step;
- ``'lbfgs'``  — SciPy L-BFGS-B (Byrd-Lu-Nocedal-Zhu, SIAM J. Sci. Comput.
  16, 1190 (1995)) fed the JAX value-and-gradient, box bounds honoured;
- ``'nelder-mead'`` — SciPy Nelder-Mead, gradient-free, for plain numpy
  objectives (e.g. the PKMM closed forms) and rugged landscapes.

``f`` is either an :class:`~vacuum.opt.objectives.Objective` (its loss is
-quantity plus the constraint penalties on its auxiliaries) or a plain
callable ``theta -> scalar``; a callable that JAX cannot trace falls back
to central finite differences for the gradient methods (declared in the
history as ``grad='fd'``).

Constraints (:func:`constraints`) enter as smooth penalties on the
differentiable side — the energy budget as mu * relu(W/W_budget - 1)^2 on
the drive work from the twin's own work ledger — and as CHECKS on the
numpy round trip: the noise floor (the protocol must have been evaluated
under at least the floor's channel rates and field temperature), the
signaling bound (the communication split's |M_comm|/|M| from the
perturbative engine on the same lattice, switching and smearing), and
ledger/passivity/precision audit passes.  :meth:`Constraints.check`
returns the verdicts; a candidate failing any of them is inadmissible
under papers/README.md and is reported as such, never silently dropped.
:func:`project_energy_budget` rescales a coupling amplitude so the work
meets the budget exactly (a scalar root on the differentiable twin), for
the equal-work comparisons the gain factors are stated at.

Every optimizer returns ``(theta_star, history)``; ``history`` is a list
of dicts (iteration, loss, quantity, penalty, |grad|, theta snapshot) so
a run's trajectory is stored beside its result.  Optimized parameters are
CANDIDATES: they become results only after the plain-numpy round trip
(``Objective.roundtrip``) and the full standing audits (the round-trip rule
of vacuum/opt/README.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError("vacuum.opt.optimize needs JAX (install the '.[diff]' extra)") from exc

jax.config.update("jax_enable_x64", True)

__all__ = [
    "Constraints",
    "constraints",
    "optimize",
    "adam",
    "lbfgs",
    "nelder_mead",
    "fd_gradient",
    "project_energy_budget",
    "make_eval",
    "make_value",
]


# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Constraints:
    """Energy budget / noise floor / signaling bound (see module docstring).

    Attributes
    ----------
    energy_budget : float or None
        Upper bound on the drive work W (ledger work column).  Penalized on
        the differentiable side; verified on the round trip.
    noise_floor : mapping or None
        ``{'kappa', 'nbar', 'gamma_phi', 'T'}`` (lattice units) the
        protocol must at least be evaluated under.  A round trip reports
        the channel parameters it ran with; the check compares.
    signaling_bound : float or None
        Upper bound on |M_comm|/|M| (communication fraction of the pair
        term, TMM21 Eq. (31)-(33) split) — round-trip check only.
    weight : float
        Penalty weight mu.
    bounds : sequence of (lo, hi) or None
        Box constraints on theta, applied by projection (adam) or handed to
        L-BFGS-B; None entries are unbounded.
    """

    energy_budget: Optional[float] = None
    noise_floor: Optional[Dict[str, float]] = None
    signaling_bound: Optional[float] = None
    weight: float = 1.0e3
    bounds: Optional[Tuple[Tuple[Optional[float], Optional[float]], ...]] = None
    signaling_weight: Optional[float] = None

    def penalty(self, theta, aux):
        """Smooth penalty (jnp scalar) from the objective's auxiliaries.

        energy budget: mu relu(W/W_budget - 1)^2 on ``aux['work']``;
        signaling bound: mu_s relu(f - f_bound)^2 on ``aux['comm_fraction']``
        (the differentiable second-order proxy of the objective; the numpy
        split on the round trip is what :meth:`check` reads).
        """
        pen = jnp.zeros(())
        if self.energy_budget is not None and "work" in aux:
            x = aux["work"] / float(self.energy_budget) - 1.0
            pen = pen + float(self.weight) * jnp.where(x > 0.0, x * x, 0.0)
        if self.signaling_bound is not None and "comm_fraction" in aux:
            mu = float(self.weight if self.signaling_weight is None else self.signaling_weight)
            y = aux["comm_fraction"] - float(self.signaling_bound)
            pen = pen + mu * jnp.where(y > 0.0, y * y, 0.0)
        return pen

    def project(self, theta):
        """Clip theta to the box (identity without bounds)."""
        if self.bounds is None:
            return np.asarray(theta, dtype=float)
        th = np.array(theta, dtype=float, copy=True)
        for i, (lo, hi) in enumerate(self.bounds):
            if lo is not None:
                th[i] = max(th[i], lo)
            if hi is not None:
                th[i] = min(th[i], hi)
        return th

    def check(self, roundtrip, *, work_rtol=1e-6):
        """Verdicts on a round-trip dict; ``admissible`` = every check passed.

        Keys read from the round trip when present: ``work``,
        ``comm_fraction``, ``noise`` (dict of the channel/field parameters
        the run used), ``audits_passed`` (bool).
        """
        out: Dict[str, Any] = {}
        if self.energy_budget is not None:
            W = float(roundtrip.get("work", float("nan")))
            out["energy_budget"] = bool(W <= float(self.energy_budget) * (1.0 + work_rtol))
            out["work"] = W
        if self.signaling_bound is not None:
            cf = roundtrip.get("comm_fraction", None)
            if cf is None or (isinstance(cf, float) and math.isnan(cf)):
                out["signaling_bound"] = None  # not computable for this row
            else:
                out["signaling_bound"] = bool(float(cf) <= float(self.signaling_bound))
            out["comm_fraction"] = cf
        if self.noise_floor is not None:
            noise = roundtrip.get("noise", {}) or {}
            ok = True
            for key, floor in self.noise_floor.items():
                val = noise.get(key, None)
                if val is None or float(val) < float(floor) * (1.0 - 1e-12):
                    ok = False
            out["noise_floor"] = ok
        if "audits_passed" in roundtrip:
            out["audits"] = bool(roundtrip["audits_passed"])
        out["admissible"] = all(v for k, v in out.items() if isinstance(v, bool))
        return out


def constraints(energy_budget=None, noise_floor=None, signaling_bound=None, weight=1.0e3, bounds=None,
                signaling_weight=None):
    """Build a :class:`Constraints` (README signature)."""
    b = None if bounds is None else tuple((None if lo is None else float(lo), None if hi is None else float(hi)) for lo, hi in bounds)
    return Constraints(
        energy_budget=None if energy_budget is None else float(energy_budget),
        noise_floor=None if noise_floor is None else dict(noise_floor),
        signaling_bound=None if signaling_bound is None else float(signaling_bound),
        weight=float(weight),
        bounds=b,
        signaling_weight=None if signaling_weight is None else float(signaling_weight),
    )


# --------------------------------------------------------------------------
# Loss assembly
# --------------------------------------------------------------------------


def _is_objective(f):
    return hasattr(f, "quantity_aux") and hasattr(f, "roundtrip")


def _make_loss(f, C):
    """(loss(theta) -> (loss, info dict), differentiable?) for an Objective or callable."""
    C = C if C is not None else Constraints()
    if _is_objective(f):

        def loss(theta):
            q, aux = f.quantity_aux(theta)
            pen = C.penalty(theta, aux)
            return -q + pen, {"quantity": q, "penalty": pen}

        return loss, bool(getattr(f, "differentiable", True))

    def loss_plain(theta):
        val = f(theta)
        pen = C.penalty(theta, {})
        return val + pen, {"quantity": -val, "penalty": pen}

    return loss_plain, None  # differentiability decided by trying jax


def fd_gradient(fn, theta, h=1e-5):
    """Central finite-difference gradient of a scalar numpy function."""
    theta = np.asarray(theta, dtype=float)
    g = np.empty_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta)
        e[i] = h
        g[i] = (float(fn(theta + e)) - float(fn(theta - e))) / (2.0 * h)
    return g


def _jax_eval(loss, jit):
    """numpy-facing ``theta -> (loss, grad, info)`` from a traced loss."""

    def core(th):
        val, info = loss(th)
        return val, info

    vg = jax.value_and_grad(core, has_aux=True)
    if jit:
        vg = jax.jit(vg)

    def eval_(th):
        (val, info), g = vg(jnp.asarray(th, dtype=jnp.float64))
        return float(val), np.asarray(g, dtype=float), {k: float(v) for k, v in info.items()}

    return eval_


def _resolve_eval(loss, differentiable, jit, theta0):
    """Pick the gradient mode: 'jax' for traceable losses, 'fd' otherwise."""
    if differentiable is True:
        return _jax_eval(loss, jit), "jax"
    if differentiable is False:
        def eval_fd(th):
            val, info = loss(np.asarray(th, dtype=float))
            g = fd_gradient(lambda x: float(loss(x)[0]), th)
            return float(val), g, {k: float(v) for k, v in info.items()}

        return eval_fd, "fd"
    # plain callable: try jax on theta0, else finite differences
    try:
        core = lambda th: loss(th)[0]
        vg = jax.value_and_grad(core)
        val, g = vg(jnp.asarray(theta0, dtype=jnp.float64))
        if not (np.all(np.isfinite(np.asarray(g))) and np.isfinite(float(val))):
            raise ValueError("non-finite jax gradient")
        if jit:
            vg = jax.jit(vg)

        def eval_(th):
            v, gg = vg(jnp.asarray(th, dtype=jnp.float64))
            info = {}
            try:
                info = {k: float(x) for k, x in loss(th)[1].items()}
            except Exception:
                pass
            return float(v), np.asarray(gg, dtype=float), info

        return eval_, "jax"
    except Exception:
        def eval_fd(th):
            val, info = loss(np.asarray(th, dtype=float))
            g = fd_gradient(lambda x: float(loss(x)[0]), th)
            return float(val), g, {k: float(v) for k, v in info.items()}

        return eval_fd, "fd"


def make_eval(f, C=None, theta0=None, *, jit=True):
    """Public ``theta -> (loss, grad, info)`` of the penalized loss, plus its gradient mode.

    The evaluator :func:`optimize` builds internally, exposed so a search
    layer (``vacuum.opt.multistart``) can drive several starts, seed a
    derivative-free search and rank candidates on the SAME penalized loss
    the gradient drivers minimize.  ``theta0`` is only needed for a plain
    callable (to decide 'jax' vs 'fd'); an Objective supplies its own.
    """
    C = C if C is not None else Constraints()
    loss, differentiable = _make_loss(f, C)
    if theta0 is None and hasattr(f, "theta0"):
        theta0 = f.theta0
    return _resolve_eval(loss, differentiable, jit, theta0)


def make_value(f, C=None, *, jit=True):
    """Value-only ``theta -> (loss, info)`` of the penalized loss (jitted for an Objective).

    Half the cost of :func:`make_eval` where no gradient is needed (the
    cross-entropy seeding of the multistart, basin ranking).
    """
    C = C if C is not None else Constraints()
    loss, differentiable = _make_loss(f, C)
    if differentiable is True and jit:
        core = jax.jit(loss)

        def value(th):
            val, info = core(jnp.asarray(th, dtype=jnp.float64))
            return float(val), {k: float(v) for k, v in info.items()}

        return value

    def value_plain(th):
        val, info = loss(np.asarray(th, dtype=float))
        return float(val), {k: float(v) for k, v in info.items()}

    return value_plain


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------


def adam(eval_fn, theta0, *, n_iter=200, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8,
         project=None, tol=0.0, callback=None, keep_theta=True):
    """Adam on ``eval_fn(theta) -> (loss, grad, info)``; returns (theta_best, history).

    The BEST iterate (lowest loss) is returned, not the last one; ``tol``
    stops when |grad| falls below it.
    """
    th = np.asarray(theta0, dtype=float).copy()
    m = np.zeros_like(th)
    v = np.zeros_like(th)
    hist: List[Dict[str, Any]] = []
    best = (math.inf, th.copy())
    for it in range(1, int(n_iter) + 1):
        val, g, info = eval_fn(th)
        gn = float(np.linalg.norm(g))
        row = {"iter": it - 1, "loss": val, "grad_norm": gn, **info}
        if keep_theta:
            row["theta"] = th.copy()
        hist.append(row)
        if val < best[0]:
            best = (val, th.copy())
        if callback is not None:
            callback(it - 1, th, val, info)
        if gn <= tol:
            break
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        mhat = m / (1.0 - beta1 ** it)
        vhat = v / (1.0 - beta2 ** it)
        th = th - lr * mhat / (np.sqrt(vhat) + eps)
        if project is not None:
            th = np.asarray(project(th), dtype=float)
    val, g, info = eval_fn(th)
    row = {"iter": len(hist), "loss": val, "grad_norm": float(np.linalg.norm(g)), **info}
    if keep_theta:
        row["theta"] = th.copy()
    hist.append(row)
    if val < best[0]:
        best = (val, th.copy())
    return best[1], hist


def lbfgs(eval_fn, theta0, *, bounds=None, maxiter=500, gtol=1e-10, ftol=1e-15, callback=None,
          keep_theta=True, restarts=0, restart_tol=1e-12):
    """SciPy L-BFGS-B on ``eval_fn``; returns (theta_star, history).

    ``restarts`` > 0 re-launches L-BFGS-B from the returned point (fresh
    Hessian memory) up to that many times while a restart still lowers the
    loss by more than ``restart_tol`` relative — the standard cure for the
    'ABNORMAL_TERMINATION_IN_LNSRCH' stop on stiff penalized landscapes.
    The history records every restart's message.
    """
    from scipy.optimize import minimize

    hist: List[Dict[str, Any]] = []
    counter = {"n": 0}

    def fun(th):
        val, g, info = eval_fn(th)
        row = {"iter": counter["n"], "loss": val, "grad_norm": float(np.linalg.norm(g)), **info}
        if keep_theta:
            row["theta"] = np.asarray(th, dtype=float).copy()
        hist.append(row)
        counter["n"] += 1
        if callback is not None:
            callback(counter["n"], th, val, info)
        return val, g

    x = np.asarray(theta0, dtype=float)
    res = None
    for k in range(int(restarts) + 1):
        prev = None if res is None else float(res.fun)
        res = minimize(fun, x, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": int(maxiter), "gtol": gtol, "ftol": ftol})
        x = np.asarray(res.x, dtype=float)
        hist.append({"iter": counter["n"], "loss": float(res.fun), "grad_norm": float(np.linalg.norm(res.jac)),
                     "message": str(res.message), "success": bool(res.success), "nit": int(res.nit), "restart": k,
                     **({"theta": x.copy()} if keep_theta else {})})
        if prev is not None and not (prev - float(res.fun) > restart_tol * max(1.0, abs(prev))):
            break
    return x, hist


def nelder_mead(loss_fn, theta0, *, bounds=None, maxiter=2000, xatol=1e-8, fatol=1e-12,
                keep_theta=True):
    """SciPy Nelder-Mead on a scalar ``loss_fn(theta)``; returns (theta_star, history)."""
    from scipy.optimize import minimize

    hist: List[Dict[str, Any]] = []

    def fun(th):
        val = float(loss_fn(np.asarray(th, dtype=float)))
        row = {"iter": len(hist), "loss": val}
        if keep_theta:
            row["theta"] = np.asarray(th, dtype=float).copy()
        hist.append(row)
        return val

    res = minimize(fun, np.asarray(theta0, dtype=float), method="Nelder-Mead", bounds=bounds,
                   options={"maxiter": int(maxiter), "xatol": xatol, "fatol": fatol})
    hist.append({"iter": len(hist), "loss": float(res.fun), "message": str(res.message),
                 "success": bool(res.success), "nit": int(res.nit),
                 **({"theta": np.asarray(res.x, dtype=float).copy()} if keep_theta else {})})
    return np.asarray(res.x, dtype=float), hist


def optimize(f, C, theta0, method="adam", *, jit=True, **opts):
    """Minimize ``f`` (Objective or callable) under constraints ``C``.

    Parameters
    ----------
    f : Objective or callable theta -> scalar
        An Objective is minimized as -quantity + C.penalty(aux); a callable
        is minimized as f(theta) + C.penalty(theta, {}).
    C : Constraints or None
    theta0 : array
        Generic start.
    method : {'adam', 'lbfgs', 'nelder-mead'}
    opts : forwarded to the driver (n_iter, lr, maxiter, bounds, ...).
        ``bounds`` defaults to ``C.bounds``.

    Returns
    -------
    (theta_star, history) — ``history[-1]`` carries ``grad_mode`` ('jax'|'fd'|'none').
    """
    C = C if C is not None else Constraints()
    theta0 = np.asarray(theta0, dtype=float)
    loss, differentiable = _make_loss(f, C)
    bounds = opts.pop("bounds", C.bounds)
    if method == "nelder-mead":
        theta, hist = nelder_mead(lambda th: float(loss(th)[0]), theta0, bounds=bounds, **opts)
        hist[-1]["grad_mode"] = "none"
        return theta, hist
    eval_fn, mode = _resolve_eval(loss, differentiable, jit, theta0)
    if method == "adam":
        project = C.project if bounds is not None else None
        if bounds is not None and C.bounds is None:
            project = Constraints(bounds=tuple(bounds)).project
        theta, hist = adam(eval_fn, theta0, project=project, **opts)
    elif method == "lbfgs":
        theta, hist = lbfgs(eval_fn, theta0, bounds=bounds, **opts)
    else:
        raise ValueError(f"method must be 'adam', 'lbfgs' or 'nelder-mead', got {method!r}")
    hist[-1]["grad_mode"] = mode
    return theta, hist


# --------------------------------------------------------------------------
# Equal-work projection
# --------------------------------------------------------------------------


def project_energy_budget(objective, theta, budget, index, *, lo=1e-6, hi=None, xtol=1e-12):
    """Rescale theta[index] (a coupling amplitude) so the twin's work equals ``budget``.

    Solves W(theta with theta[index] = s) = budget by bracketed root finding
    on the differentiable twin (Brent), which is exact at the twin's
    resolution; the caller then round-trips the result through the gated
    numpy stack and reports the ledger work actually achieved.  Returns the
    projected theta.
    """
    from scipy.optimize import brentq

    theta = np.asarray(theta, dtype=float).copy()
    fn = jax.jit(lambda th: objective.quantity_aux(th)[1]["work"])

    def W(s):
        th = theta.copy()
        th[index] = s
        return float(fn(jnp.asarray(th))) - float(budget)

    s0 = abs(theta[index]) if theta[index] != 0.0 else 1.0
    hi = 4.0 * s0 if hi is None else hi
    # widen the bracket until the sign changes
    a, b = lo, hi
    fa, fb = W(a), W(b)
    n = 0
    while fa * fb > 0.0 and n < 40:
        b *= 2.0
        fb = W(b)
        n += 1
    if fa * fb > 0.0:
        raise RuntimeError("could not bracket the energy-budget root")
    s = brentq(W, a, b, xtol=xtol, rtol=1e-14, maxiter=200)
    theta[index] = s if theta[index] >= 0.0 else -s
    return theta
