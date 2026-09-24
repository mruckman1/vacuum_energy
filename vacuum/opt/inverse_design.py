"""Inverse design of vacua (Layer 6): a coupling matrix from a target profile.

Prescribe a correlation or negativity profile and solve for the coupling
matrix K whose ground state realizes it — programmable vacua as an
inverse spectral problem (PLAN.md, Layer 6).  The ground state of
H = (1/2)(p.p + x.K.x) has covariance V = blockdiag(K^{-1/2}, K^{1/2})/2
(``gjax.ground_state_cov``), so every target is a smooth function of K and
the design is gradient descent on

    loss(theta) = residual(K(theta), target)^2 + reg * ||K(theta) - K_ref||_F^2 / ||K_ref||_F^2,

with K > 0 ENFORCED BY THE PARAMETRIZATION, never by a penalty:

- ``ansatz='banded'``: K = L L^T + eps I with L lower-triangular of
  bandwidth ``band`` (a band-``band`` Cholesky factor; every positive-
  definite matrix of that bandwidth has one, so the ansatz is complete
  for banded couplings — a chain is band 1); the symmetries of the ansatz
  are the column signs of L (K is invariant under L -> L D, D = diag(+-1));
- ``ansatz='full'``: L dense lower-triangular (all positive-definite K);
- ``ansatz='eigen'``: K = Q diag(exp(l)) Q^T with Q = expm(A - A^T) an
  orthogonal matrix from a free generator A (all positive-definite K; the
  eigenvalues are exp(l) > 0 by construction).

Targets (:func:`design_residual` accepts the same dicts):

- ``{'kind': 'corr_xx', 'V_xx': (N,N)}``     — <x_i x_j> of the ground state,
  residual = ||V_xx(K) - target||_F / ||target||_F;
- ``{'kind': 'covariance', 'V': (2N,2N)}``   — the full covariance;
- ``{'kind': 'corr_profile', 'C': (N,)}``   — translation-averaged profile
  C(r) = mean_i <x_i x_{i+r}> (a periodic chain's fingerprint);
- ``{'kind': 'negativity', 'pairs': [(modes_A, modes_B, E_N_target), ...]}``
  — prescribed log-negativities between chosen mode blocks, residual =
  sqrt(sum (E_N(K) - target)^2) (nats).  Because ``gjax.log_negativity``
  is flat in the sudden-death region, negativity designs start from a
  reference chain on which the chosen blocks are already entangled (the
  ``K_ref`` argument), and the regularizer keeps the design near it.

The result carries the design K, the achieved residual, the optimizer
history, and the standing audits of the DESIGNED vacuum run with the numpy
stack (``vacuum.audits``): the passivity audit on the ground state of K,
the precision audit on the reported blocks, and the causality audit with a
Lieb-Robinson velocity bound derived from the design itself (for a
symmetric banded K the group velocity is bounded by
v_max <= max_k |d omega/d k| computed on the circulant majorant with
couplings max_i |K_{i,i+r}|, r = 1..band; stated in the result).  A
design is "physical" by construction (K > 0, nu = 1/2 exactly); the audits
check the numpy stack agrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError("vacuum.opt.inverse_design needs JAX (install the '.[diff]' extra)") from exc

from vacuum.opt import gjax as G

jax.config.update("jax_enable_x64", True)

__all__ = [
    "ANSATZE",
    "TARGET_KINDS",
    "banded_indices",
    "coupling_from_theta",
    "theta_from_coupling",
    "design_residual",
    "DesignResult",
    "inverse_design",
    "lieb_robinson_velocity_bound",
]

ANSATZE = ("banded", "full", "eigen")
TARGET_KINDS = ("corr_xx", "covariance", "corr_profile", "negativity")


# --------------------------------------------------------------------------
# Parametrizations (K > 0 by construction)
# --------------------------------------------------------------------------


def banded_indices(N, band):
    """(rows, cols) of the lower-triangular entries with 0 <= row - col <= band."""
    rows, cols = [], []
    for i in range(int(N)):
        for j in range(max(0, i - int(band)), i + 1):
            rows.append(i)
            cols.append(j)
    return np.asarray(rows), np.asarray(cols)


def coupling_from_theta(theta, N, ansatz="banded", band=1, eps=1e-8, xp=jnp):
    """K(theta) for the chosen ansatz (module docstring); jnp or numpy."""
    N = int(N)
    theta = xp.asarray(theta)
    if ansatz in ("banded", "full"):
        b = N - 1 if ansatz == "full" else int(band)
        rows, cols = banded_indices(N, b)
        if theta.shape != (rows.size,):
            raise ValueError(f"theta must have {rows.size} entries for ansatz {ansatz!r} (N={N}, band={b})")
        L = xp.zeros((N, N))
        if xp is jnp:
            L = L.at[rows, cols].set(theta)
        else:
            L[rows, cols] = theta
        return L @ L.T + float(eps) * xp.eye(N)
    if ansatz == "eigen":
        n_a = N * (N - 1) // 2
        if theta.shape != (n_a + N,):
            raise ValueError(f"theta must have {n_a + N} entries for ansatz 'eigen' (N={N})")
        iu = np.triu_indices(N, 1)
        A = xp.zeros((N, N))
        if xp is jnp:
            A = A.at[iu].set(theta[:n_a])
            Qm = jax.scipy.linalg.expm(A - A.T)
        else:
            from scipy.linalg import expm

            A[iu] = theta[:n_a]
            Qm = expm(A - A.T)
        lam = xp.exp(theta[n_a:])
        return (Qm * lam) @ Qm.T
    raise ValueError(f"ansatz must be one of {ANSATZE}, got {ansatz!r}")


def theta_from_coupling(K, ansatz="banded", band=1, eps=1e-8):
    """Inverse map for a positive-definite K (Cholesky / eigendecomposition)."""
    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    if ansatz in ("banded", "full"):
        L = np.linalg.cholesky(K - float(eps) * np.eye(N))
        b = N - 1 if ansatz == "full" else int(band)
        rows, cols = banded_indices(N, b)
        return L[rows, cols].copy()
    if ansatz == "eigen":
        w, Q = np.linalg.eigh(K)
        if np.linalg.det(Q) < 0.0:  # expm of an antisymmetric generator lands in SO(N)
            Q = Q.copy()
            Q[:, 0] = -Q[:, 0]
        from scipy.linalg import logm

        A = np.real(logm(Q))
        A = 0.5 * (A - A.T)
        iu = np.triu_indices(N, 1)
        return np.concatenate([A[iu], np.log(w)])
    raise ValueError(f"ansatz must be one of {ANSATZE}, got {ansatz!r}")


# --------------------------------------------------------------------------
# Targets and residuals
# --------------------------------------------------------------------------


def _check_target(target):
    if not isinstance(target, dict) or "kind" not in target:
        raise ValueError("target must be a dict with a 'kind' key")
    if target["kind"] not in TARGET_KINDS:
        raise ValueError(f"target kind must be one of {TARGET_KINDS}, got {target['kind']!r}")
    return target


def _residual_terms(K, target, xp):
    """Residual (scalar) for the target, evaluated with jnp (traced) or numpy."""
    kind = target["kind"]
    Gm = G if xp is jnp else None
    if kind in ("corr_xx", "covariance", "corr_profile"):
        if xp is jnp:
            V = G.ground_state_cov(K)
        else:
            from vacuum.core import ground_state_cov

            V = ground_state_cov(K)
        N = K.shape[0]
        if kind == "corr_xx":
            tgt = xp.asarray(target["V_xx"])
            return xp.linalg.norm(V[:N, :N] - tgt) / xp.linalg.norm(tgt)
        if kind == "covariance":
            tgt = xp.asarray(target["V"])
            return xp.linalg.norm(V - tgt) / xp.linalg.norm(tgt)
        tgt = xp.asarray(target["C"])
        Vxx = V[:N, :N]
        prof = xp.stack([xp.mean(xp.diagonal(Vxx, r)) for r in range(tgt.shape[0])])
        return xp.linalg.norm(prof - tgt) / xp.linalg.norm(tgt)
    # negativity
    if xp is jnp:
        V = G.ground_state_cov(K)
        vals = [G.log_negativity(V, list(A), list(B)) - float(t) for A, B, t in target["pairs"]]
    else:
        from vacuum.core import ground_state_cov, log_negativity

        V = ground_state_cov(K)
        vals = [log_negativity(V, list(A), list(B)) - float(t) for A, B, t in target["pairs"]]
    return xp.sqrt(xp.sum(xp.asarray(vals) ** 2))


def design_residual(K, target):
    """Residual of K against the target (numpy; module docstring for the norms)."""
    target = _check_target(target)
    return float(_residual_terms(np.asarray(K, dtype=float), target, np))


def lieb_robinson_velocity_bound(K):
    """Group-velocity bound for a symmetric banded K: max_k |d omega/d k| on the
    circulant majorant with couplings c_r = max_i |K_{i,i+r}| and mass
    m^2 = min_i K_ii - 2 sum_r c_r (floored at 0)."""
    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    c = np.array([np.max(np.abs(np.diagonal(K, r))) for r in range(1, N)])
    band = int(np.max(np.flatnonzero(c > 1e-14)) + 1) if np.any(c > 1e-14) else 0
    ks = np.linspace(0.0, math.pi, 4001)
    w2 = np.full_like(ks, max(0.0, float(np.min(np.diag(K)) - 2.0 * np.sum(c[:band]))))
    dw2 = np.zeros_like(ks)
    for r in range(1, band + 1):
        w2 = w2 + 2.0 * c[r - 1] * (1.0 - np.cos(r * ks))
        dw2 = dw2 + 2.0 * c[r - 1] * r * np.sin(r * ks)
    w = np.sqrt(np.maximum(w2, 1e-300))
    return float(np.max(np.abs(dw2 / (2.0 * w)))), band


# --------------------------------------------------------------------------
# The design
# --------------------------------------------------------------------------


@dataclass
class DesignResult:
    K: np.ndarray
    residual: float
    theta: np.ndarray
    history: List[Dict[str, Any]]
    ansatz: str
    band: int
    reg: float
    audits: Dict[str, Any] = field(default_factory=dict)
    v_max_bound: Optional[float] = None
    min_eigenvalue: float = float("nan")
    achieved: Dict[str, Any] = field(default_factory=dict)

    def all_passed(self):
        return all(getattr(v, "passed", True) for v in self.audits.values() if v is not None)


def inverse_design(target, ansatz="banded", reg=0.0, *, N=None, band=1, K_ref=None, theta0=None,
                   method="lbfgs", maxiter=5000, eps=1e-8, seed=0, gtol=1e-14, ftol=0.0,
                   audit=True, report_modes=None, causality_t=None):
    """Solve for K whose ground state matches ``target`` (module docstring).

    Returns a :class:`DesignResult` with K (positive definite by
    construction), the achieved residual, the optimizer history and — when
    ``audit`` — the passivity/precision/causality audits of the designed
    vacuum from the numpy stack.  ``theta0`` defaults to the identity
    coupling (K = I + eps) or, if ``K_ref`` is given, to K_ref itself.
    """
    from vacuum.opt.optimize import optimize

    target = _check_target(target)
    if N is None:
        if K_ref is not None:
            N = int(np.asarray(K_ref).shape[0])
        elif target["kind"] == "corr_xx":
            N = int(np.asarray(target["V_xx"]).shape[0])
        elif target["kind"] == "covariance":
            N = int(np.asarray(target["V"]).shape[0]) // 2
        else:
            raise ValueError("pass N (or K_ref) for this target kind")
    N = int(N)
    if ansatz not in ANSATZE:
        raise ValueError(f"ansatz must be one of {ANSATZE}, got {ansatz!r}")
    Kref = None if K_ref is None else np.asarray(K_ref, dtype=float)
    if theta0 is None:
        theta0 = theta_from_coupling(np.eye(N) + 0.0 if Kref is None else Kref, ansatz, band, eps)
        if Kref is None:
            theta0 = theta_from_coupling((1.0 + float(eps)) * np.eye(N), ansatz, band, eps)
    theta0 = np.asarray(theta0, dtype=float)
    Kref_j = None if Kref is None else jnp.asarray(Kref)
    ref_norm2 = None if Kref is None else float(np.sum(Kref ** 2))

    def loss(theta):
        K = coupling_from_theta(theta, N, ansatz, band, eps, jnp)
        r = _residual_terms(K, target, jnp)
        val = r * r
        if reg > 0.0 and Kref_j is not None:
            val = val + float(reg) * jnp.sum((K - Kref_j) ** 2) / ref_norm2
        return val

    theta, hist = optimize(loss, None, theta0, method=method, maxiter=maxiter, gtol=gtol, ftol=ftol,
                           keep_theta=False)
    K = np.asarray(coupling_from_theta(theta, N, ansatz, band, eps, np), dtype=float)
    K = 0.5 * (K + K.T)
    res = design_residual(K, target)
    out = DesignResult(K=K, residual=res, theta=theta, history=hist, ansatz=ansatz, band=int(band), reg=float(reg),
                       min_eigenvalue=float(np.min(np.linalg.eigvalsh(K))))
    if target["kind"] == "negativity":
        from vacuum.core import ground_state_cov, log_negativity

        V = ground_state_cov(K)
        out.achieved = {"E_N": [float(log_negativity(V, list(A), list(B))) for A, B, _ in target["pairs"]],
                        "target": [float(t) for _, _, t in target["pairs"]]}
    if audit:
        out.audits, out.v_max_bound = audit_design(K, report_modes=report_modes, causality_t=causality_t)
    return out


def audit_design(K, *, report_modes=None, causality_t=None):
    """Standing audits of the designed vacuum (numpy): passivity, precision, causality."""
    from vacuum.audits import causality_audit, passivity_audit, precision_audit
    from vacuum.core import ground_state_cov, reduce as reduce_cov

    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    V = ground_state_cov(K)
    audits: Dict[str, Any] = {"passivity": passivity_audit(K, V=V, strict=False)}
    modes_list = [tuple(range(min(4, N)))] if report_modes is None else [tuple(m) for m in report_modes]
    audits["precision"] = precision_audit(reduce_cov(V, modes_list[0]), strict=False)
    v_bound, band = lieb_robinson_velocity_bound(K)
    caus = None
    if band >= 1:
        t = float(causality_t) if causality_t is not None else max(1.0, 0.25 * N / max(v_bound, 1e-12))
        try:
            caus = causality_audit(K, t, v_max=max(v_bound, 1e-12), bc="open", strict=False)
        except ValueError as exc:
            caus = {"not_applicable": str(exc)}
    audits["causality"] = caus
    return audits, v_bound
