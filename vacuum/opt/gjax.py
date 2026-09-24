"""Differentiable Gaussian core (Layer 6): JAX twins of ``vacuum.core``.

Every function here is a pure, jit- and grad-safe mirror of the numpy
function of the same name in ``vacuum.core`` (gaussian, channels, models,
dynamics), with IDENTICAL conventions and signatures (docs/API.md):

- hbar = 1, vacuum covariance I/2, symplectic eigenvalues nu >= 1/2;
- block quadrature ordering R = (x_1..x_N, p_1..p_N), Omega = [[0, I], [-I, 0]];
- entropies and negativities in nats;
- mode index sets are Python sequences of 0-based mode indices.  They are
  *static* (trace-time) data: under ``jax.jit`` pass them as tuples through
  ``static_argnums``/``static_argnames`` or close over them.

Everything is a function of arrays returning arrays (0-d arrays for scalars),
so optimizers, distillation back ends and inverse design can compose these
freely with ``jax.grad``, ``jax.jit``, ``jax.vmap``.  float64 is enabled at
import (``jax_enable_x64``); float32 is not acceptable for this physics
(the vacuum floor nu = 1/2 has to be resolved to ~1e-16).

Value checks (PSD/PD input, channel parameter ranges, T >= 0) are performed
only when the checked quantity is *concrete*; under tracing (jit/grad/vmap)
they are skipped — a traced value cannot be branched on.  Numerics are
otherwise identical to the numpy core, verified in tests/test_gjax.py.

Numerical design decisions (the ones that matter for gradients)
---------------------------------------------------------------
(i) Symplectic eigenvalues.  The numpy core takes the singular values of the
    antisymmetric matrix A = V^{1/2} Omega V^{1/2}; each nu appears TWICE, so
    the spectrum is *always* degenerate in pairs.  JAX's built-in SVD/eigh
    JVPs contain 1/(s_i^2 - s_j^2) (resp. 1/(w_i - w_j)) factors and are not
    trustworthy on such spectra, and the eigh JVP is outright NaN on the
    degenerate V spectrum of the vacuum (V = I/2).  Two custom JVPs fix this
    exactly:

    * matrix functions f(V) through eigh use the Daleckii-Krein formula
      df = U [ dd_ij * (U^T dV U)_ij ] U^T with the divided difference
      dd_ij = (f(w_i) - f(w_j))/(w_i - w_j) -> f'(w) on the diagonal and
      in degenerate blocks.  For the square root and inverse square root the
      divided differences have closed forms with NO cancellation,
      1/(sqrt(w_i) + sqrt(w_j)) and -1/(sqrt(w_i) sqrt(w_j)(sqrt(w_i)+sqrt(w_j))),
      so the gradient is exact for any degeneracy pattern (degenerate K of a
      periodic chain, fully degenerate V of the vacuum);
    * the paired singular values are differentiated with dsigma_i = u_i^T dA v_i
      and returned as the pair means nu_k = (sigma_{2k} + sigma_{2k+1})/2.
      Within a degenerate block of the singular spectrum the individual
      dsigma_i are basis-dependent but their sum over the block is the
      invariant trace tr(P_V A^T dA)/sigma, so the pair mean has a well-defined
      derivative, and any symmetric spectral function (entropy, negativity,
      mutual information) gets the correct gradient even when several nu's
      coincide (all nu = 1/2 in the vacuum).

    The forward pass is the numpy algorithm step for step, so values agree
    with ``vacuum.core`` to float64 roundoff.  Alternatives were measured:
    eigh of A^T A = -A^2 (real symmetric, eigenvalues nu^2) squares the
    condition number and loses ~sqrt(eps)-level accuracy at the floor when
    nu_max >> 1 (the critical chain has nu_max ~ 70); eigh of the Hermitian
    iA gives the same accuracy as the SVD but in complex arithmetic.  The
    SVD route was kept because it is the numpy route.

(ii) ``entropy_from_nu`` clamps nu at the floor 1/2 with an explicit mask
    (``jnp.where``, not ``jnp.maximum``): value and gradient are exactly 0
    for nu <= 1/2 and ln((nu+1/2)/(nu-1/2)) above it — finite everywhere,
    never NaN.  Convention at the floor: the one-sided derivative from the
    clamped side (0), even though the derivative from above diverges.

(iii) ``log_negativity`` uses E_N = sum_k [nu_k < 1/2] * (-ln 2 nu_k).  At the
    sudden-death boundary nu = 1/2 the subgradient set is [-2 dnu, 0]; the
    convention here is 0 (the "dead" side), matching (ii).  Consequence for
    optimizers: in the dead region E_N and its gradient are identically zero,
    so a surrogate (e.g. the minimum partial-transpose symplectic eigenvalue,
    ``pt_min_symplectic_eigenvalue``) is needed to climb back to life.

(iv) ``entanglement_hamiltonian`` avoids the Williamson matrix S (whose
    gradient is basis-ambiguous) and uses the closed form
    G = V^{-1/2} phi(A^T A) V^{-1/2},  A = V^{-1/2} Omega V^{-1/2},
    phi(mu) = nu g(nu), nu = mu^{-1/2}, g(nu) = ln((nu+1/2)/(nu-1/2)),
    with nu clamped at ``nu_floor`` exactly as in the numpy core.  In the
    inverse formulation the vacuum floor sits at the TOP of the spectrum
    (mu = 4), so the squaring is benign there.  phi is differentiated with a
    generic divided difference that treats the symplectic pairs of A^T A
    exactly (pair-averaged before differentiation).  Because eigh returns
    the near-degenerate vacuum-floor cluster in a scrambled basis, phi(A^T A)
    is additionally symmetrized over symplectic partners with the complex
    structure J = A (A^T A)^{-1/2}: F -> (F + J F J^T)/2.  This is an
    identity in exact arithmetic (same function, same gradient) and in
    float64 restores the exact x/p partner symmetry of the Williamson form
    that the numpy Schur-block route has by construction — without it the
    large, mutually cancelling floor-mode contributions to the first law
    leak at the 1e-5 level (measured), with it the JAX route reproduces the
    first-law anchor to ~1e-10 like the core.

(v) ``chain_propagator(method='spectral')`` evaluates the exact normal-mode
    form (identical to numpy) but takes its tangent from the differentiable
    matrix exponential exp(t Omega diag(K, I)) — the same mathematical
    function, so the derivative is exact and free of divided-difference
    thresholds.

Caveats (documented, not hidden): only first derivatives are guaranteed at
degenerate spectra (the custom rules themselves call eigh/svd, whose JVPs
carry the degeneracy problem — Hessians at exactly degenerate points may be
NaN).  At nu = 0 (singular quadratic forms) the symplectic spectrum has a
square-root branch point; the gradient of sqrt at 0 is set to 0 rather than
infinity.  Gradients of ``entanglement_hamiltonian`` in the floor subspace
are ill-conditioned (g ~ ln(1/(nu-1/2))) in every implementation.

References: as in vacuum.core.gaussian; Daleckii-Krein (1965) for the
derivative of a matrix function; Higham, "Functions of Matrices" (2008).
"""

from __future__ import annotations

from functools import partial

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import expm as _expm
except ImportError as exc:  # pragma: no cover - exercised only without jax
    raise ImportError(
        "vacuum.opt.gjax needs JAX (install the '.[diff]' extra)"
    ) from exc

from vacuum.core import models as _models

jax.config.update("jax_enable_x64", True)

__all__ = [
    "Omega",
    "ground_state_cov",
    "thermal_state_cov",
    "symplectic_eigenvalues",
    "entropy",
    "entropy_from_nu",
    "reduce",
    "partial_transpose",
    "log_negativity",
    "log_negativity_from_nu",
    "pt_min_symplectic_eigenvalue",
    "mutual_information",
    "entanglement_hamiltonian",
    "modular_energy",
    "symplectic_from_quadratic",
    "symplectic_inverse",
    "evolve",
    "mean_energy",
    "apply_channel",
    "loss",
    "thermal_noise",
    "amplifier",
    "loss_XY",
    "thermal_noise_XY",
    "amplifier_XY",
    "harmonic_chain_K",
    "chain_propagator",
    "psd_sqrt",
    "pd_invsqrt",
    "sym_funm",
]

VACUUM_NU = 0.5
_F64 = jnp.float64


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _asarray(x):
    return jnp.asarray(x, dtype=_F64)


def _sym(V):
    """Symmetrize (defensive against roundoff asymmetry) — as in the core."""
    V = _asarray(V)
    return 0.5 * (V + V.T)


def _is_concrete(x):
    return not isinstance(x, jax.core.Tracer)


def _maybe_check(ok, msg):
    """Raise ValueError(msg) if ``ok`` is concrete and False; no-op when traced."""
    if _is_concrete(ok) and not bool(jnp.all(ok)):
        raise ValueError(msg)


def _quad_indices(N, modes):
    """Map 0-based mode indices to quadrature rows (k, N+k) — static data."""
    m = np.asarray(list(modes), dtype=int)
    if m.size and (m.min() < 0 or m.max() >= N):
        raise IndexError(f"mode index out of range for N={N}: {m}")
    return np.concatenate([m, m + N])


def Omega(N):
    """Symplectic form for N modes in block ordering, [[0, I_N], [-I_N, 0]]."""
    I = jnp.eye(N, dtype=_F64)
    Z = jnp.zeros((N, N), dtype=_F64)
    return jnp.block([[Z, I], [-I, Z]])


# ----------------------------------------------------------------------------
# matrix functions of symmetric matrices with a Daleckii-Krein custom JVP
# ----------------------------------------------------------------------------

def _pair_average(w):
    """Replace adjacent (ascending) eigenvalue pairs by their means."""
    wm = 0.5 * (w[0::2] + w[1::2])
    return jnp.repeat(wm, 2)


@partial(jax.custom_jvp, nondiff_argnums=(2, 3, 4))
def _sym_funms(A, params, fs, dds, paired):
    """tuple(f(A) for f in fs) for symmetric A, one eigh shared by all f.

    ``fs[i](w, params)`` is a vectorized scalar function of the eigenvalues,
    ``dds[i](w, params)`` returns its (n, n) divided-difference matrix, and
    ``paired`` pair-averages the ascending eigenvalues first (for matrices
    with an exactly paired spectrum, e.g. A^T A of an antisymmetric A).
    ``params`` is a tuple of differentiable array parameters passed to f.
    """
    w, U = jnp.linalg.eigh(_sym(A))
    if paired:
        w = _pair_average(w)
    return tuple(_sym((U * f(w, params)) @ U.T) for f in fs)


@_sym_funms.defjvp
def _sym_funms_jvp(fs, dds, paired, primals, tangents):
    A, params = primals
    dA, dparams = tangents
    w, U = jnp.linalg.eigh(_sym(A))
    if paired:
        w = _pair_average(w)
    M = U.T @ _sym(dA) @ U  # tangent in the eigenbasis
    has_params = bool(jax.tree_util.tree_leaves(params))
    outs, douts = [], []
    for f, dd in zip(fs, dds):
        fw = f(w, params)
        outs.append(_sym((U * fw) @ U.T))
        dout = U @ (dd(w, params) * M) @ U.T
        if has_params:
            _, dfw = jax.jvp(lambda p, f=f: f(w, p), (params,), (dparams,))
            dout = dout + (U * dfw) @ U.T
        douts.append(_sym(dout))
    return tuple(outs), tuple(douts)


def sym_funm(A, f, params=(), dd=None, paired=False, rel_tol=1e-9):
    """f(A) for symmetric A with a degeneracy-safe (Daleckii-Krein) JVP.

    ``f(w, params)`` must be a vectorized jnp function of the eigenvalues and
    differentiable in ``params`` (a tuple of arrays).  ``dd`` may supply a
    closed-form divided-difference matrix; by default it is built from f with
    the generic rule: (f_i - f_j)/(w_i - w_j) where the pair is separated by
    more than ``rel_tol * max(|w_i|, |w_j|)``, and the average of f'(w_i),
    f'(w_j) (second-order accurate) otherwise.  ``paired=True`` averages
    adjacent eigenvalue pairs first (exactly paired spectra).
    """
    if dd is None:
        dd = _dd_generic(f, rel_tol)
    return _sym_funms(_asarray(A), tuple(params), (f,), (dd,), bool(paired))[0]


def _dd_generic(f, rel_tol=1e-9):
    """Generic divided-difference matrix of f (see :func:`sym_funm`)."""

    def dd(w, params):
        fw = f(w, params)
        wi, wj = w[:, None], w[None, :]
        diff = wi - wj
        near = jnp.abs(diff) <= rel_tol * jnp.maximum(jnp.abs(wi), jnp.abs(wj))
        far = (fw[:, None] - fw[None, :]) / jnp.where(near, 1.0, diff)
        fp = jax.vmap(jax.grad(lambda x: f(x, params)))(w)
        return jnp.where(near, 0.5 * (fp[:, None] + fp[None, :]), far)

    return dd


def _f_sqrt(w, params):
    return jnp.sqrt(jnp.maximum(w, 0.0))


def _dd_sqrt(w, params):
    r = jnp.sqrt(jnp.maximum(w, 0.0))
    den = r[:, None] + r[None, :]
    ok = den > 0.0
    return jnp.where(ok, 1.0 / jnp.where(ok, den, 1.0), 0.0)


def _f_invsqrt(w, params):
    return 1.0 / jnp.sqrt(w)


def _dd_invsqrt(w, params):
    r = jnp.sqrt(w)
    return -1.0 / (r[:, None] * r[None, :] * (r[:, None] + r[None, :]))


def _check_psd(A, what="matrix"):
    """Concrete-only PSD check with the numpy core's tolerance."""
    A = _sym(A)
    if _is_concrete(A):
        w = jnp.linalg.eigvalsh(A)
        if float(jnp.min(w)) < -1e-10 * max(1.0, float(jnp.max(jnp.abs(w)))):
            raise ValueError(f"{what} is not PSD: min eigenvalue {float(jnp.min(w)):.3e}")


def _check_pd(A, what="matrix"):
    A = _sym(A)
    if _is_concrete(A):
        wmin = float(jnp.min(jnp.linalg.eigvalsh(A)))
        if wmin <= 0.0:
            raise ValueError(f"{what} must be positive definite: min eigenvalue {wmin:.3e}")


def psd_sqrt(V):
    """V^{1/2} of a symmetric PSD matrix (exact degeneracy-safe gradient)."""
    return _sym_funms(_asarray(V), (), (_f_sqrt,), (_dd_sqrt,), False)[0]


def pd_invsqrt(V):
    """V^{-1/2} of a symmetric positive-definite matrix (degeneracy-safe)."""
    return _sym_funms(_asarray(V), (), (_f_invsqrt,), (_dd_invsqrt,), False)[0]


# ----------------------------------------------------------------------------
# states
# ----------------------------------------------------------------------------

def ground_state_cov(K):
    """Ground-state covariance of H = (1/2)(p.p + x.K.x): blockdiag(K^{-1/2}, K^{1/2})/2."""
    K = _sym(K)
    _check_pd(K, "K")
    inv_root, root = _sym_funms(K, (), (_f_invsqrt, _f_sqrt), (_dd_invsqrt, _dd_sqrt), False)
    N = K.shape[0]
    Z = jnp.zeros((N, N), dtype=_F64)
    return jnp.block([[0.5 * inv_root, Z], [Z, 0.5 * root]])


def _coth_over(w, params):
    """coth(sqrt(w)/2T) with the T = 0 limit (coth -> 1) taken by a mask."""
    (T,) = params
    omega = jnp.sqrt(w)
    pos = T > 0.0
    T_safe = jnp.where(pos, T, 1.0)
    return jnp.where(pos, 1.0 / jnp.tanh(omega / (2.0 * T_safe)), 1.0)


def _f_thermal_xx(w, params):
    return 0.5 * _coth_over(w, params) / jnp.sqrt(w)


def _f_thermal_pp(w, params):
    return 0.5 * _coth_over(w, params) * jnp.sqrt(w)


def thermal_state_cov(K, T):
    """Thermal (Gibbs) covariance of H = (1/2)(p.p + x.K.x) at temperature T.

    V_xx = (1/2) K^{-1/2} coth(K^{1/2}/2T), V_pp = (1/2) K^{1/2} coth(K^{1/2}/2T);
    T = 0 reproduces :func:`ground_state_cov` (coth -> 1, zero gradient in T).
    """
    K = _sym(K)
    T = _asarray(T)
    _maybe_check(T >= 0.0, f"temperature must be >= 0, got {T}")
    _check_pd(K, "K")
    Vxx, Vpp = _sym_funms(
        K, (T,), (_f_thermal_xx, _f_thermal_pp),
        (_dd_generic(_f_thermal_xx), _dd_generic(_f_thermal_pp)), False,
    )
    N = K.shape[0]
    Z = jnp.zeros((N, N), dtype=_F64)
    return jnp.block([[Vxx, Z], [Z, Vpp]])


# ----------------------------------------------------------------------------
# symplectic spectrum and entropies
# ----------------------------------------------------------------------------

@jax.custom_jvp
def _paired_singular_values(A):
    """Pair means of the (descending) singular values of A: [nu1, nu2, ...]."""
    s = jnp.linalg.svd(A, compute_uv=False)
    return 0.5 * (s[0::2] + s[1::2])


@_paired_singular_values.defjvp
def _paired_singular_values_jvp(primals, tangents):
    (A,), (dA,) = primals, tangents
    U, s, Vh = jnp.linalg.svd(A, full_matrices=False)
    ds = jnp.einsum("ij,ij->j", U, dA @ Vh.T)  # diag(U^T dA V)
    nu = 0.5 * (s[0::2] + s[1::2])
    dnu = 0.5 * (ds[0::2] + ds[1::2])
    return nu, dnu


def symplectic_eigenvalues(V):
    """Symplectic eigenvalues nu_k = |spec(i Omega V)| of a PSD form, descending.

    Singular values of the antisymmetric V^{1/2} Omega V^{1/2} (each nu
    twice), exactly the numpy route; PSD-singular V is accepted and returns
    zeros for unsupported directions.  Gradient: design note (i).
    """
    V = _sym(V)
    N = V.shape[0] // 2
    _check_psd(V, "V")
    root = psd_sqrt(V)
    A = root @ Omega(N) @ root
    return _paired_singular_values(0.5 * (A - A.T))


def entropy_from_nu(nu):
    """Von Neumann entropy (nats) from symplectic eigenvalues.

    S = sum_k [(nu+1/2) ln(nu+1/2) - (nu-1/2) ln(nu-1/2)], exactly 0 at the
    vacuum floor; nu <= 1/2 contributes 0 with zero gradient (design note ii).
    """
    nu = _asarray(nu)
    above = nu > VACUUM_NU
    nu_c = jnp.where(above, nu, VACUUM_NU)
    x = nu_c + 0.5
    y = jnp.where(above, nu_c - 0.5, 1.0)  # 1.0 keeps log finite off-branch
    return jnp.sum(x * jnp.log(x) - jnp.where(above, y * jnp.log(y), 0.0))


def entropy(V):
    """Von Neumann entropy (nats) of the Gaussian state with covariance V."""
    return entropy_from_nu(symplectic_eigenvalues(V))


def reduce(V, modes):
    """Reduced covariance matrix of the listed modes (order preserved)."""
    V = _asarray(V)
    N = V.shape[0] // 2
    idx = _quad_indices(N, modes)
    return V[jnp.ix_(idx, idx)]


def partial_transpose(V, modes_B):
    """Partial transpose on modes_B: flip the sign of their momenta (P V P)."""
    V = _asarray(V)
    N = V.shape[0] // 2
    p = np.ones(2 * N)
    for m in modes_B:
        p[N + int(m)] = -1.0
    p = jnp.asarray(p)
    return (V * p).T * p


def log_negativity_from_nu(nu_t):
    """E_N = sum_k max(0, -ln 2 nu~_k) with subgradient 0 at nu~ = 1/2 (note iii)."""
    nu_t = _asarray(nu_t)
    live = nu_t < VACUUM_NU
    nu_s = jnp.where(live, nu_t, 1.0)
    return jnp.sum(jnp.where(live, -jnp.log(2.0 * nu_s), 0.0))


def _pt_spectrum(V, modes_A, modes_B):
    modes_A, modes_B = list(modes_A), list(modes_B)
    if set(modes_A) & set(modes_B):
        raise ValueError("modes_A and modes_B must be disjoint")
    VAB = reduce(V, modes_A + modes_B)
    B_local = range(len(modes_A), len(modes_A) + len(modes_B))
    return symplectic_eigenvalues(partial_transpose(VAB, B_local))


def log_negativity(V, modes_A, modes_B):
    """Logarithmic negativity (nats) between mode sets A and B (PT on B)."""
    return log_negativity_from_nu(_pt_spectrum(V, modes_A, modes_B))


def pt_min_symplectic_eigenvalue(V, modes_A, modes_B):
    """Smallest partial-transpose symplectic eigenvalue of the A:B split.

    Differentiable surrogate for the sudden-death region: E_N > 0 iff this is
    < 1/2, and it keeps a nonzero gradient where ``log_negativity`` is flat.
    """
    return jnp.min(_pt_spectrum(V, modes_A, modes_B))


def mutual_information(V, modes_A, modes_B):
    """Mutual information I(A:B) = S(A) + S(B) - S(AB), in nats."""
    modes_A, modes_B = list(modes_A), list(modes_B)
    if set(modes_A) & set(modes_B):
        raise ValueError("modes_A and modes_B must be disjoint")
    SA = entropy(reduce(V, modes_A))
    SB = entropy(reduce(V, modes_B))
    SAB = entropy(reduce(V, modes_A + modes_B))
    return SA + SB - SAB


# ----------------------------------------------------------------------------
# entanglement (modular) Hamiltonian
# ----------------------------------------------------------------------------

def _phi_factory(nu_floor):
    """phi(mu) = nu g(nu), nu = max(mu^{-1/2}, nu_floor), g = ln((nu+1/2)/(nu-1/2))."""
    nu_floor = float(nu_floor)

    def phi(mu, params):
        mu_s = jnp.maximum(mu, jnp.finfo(_F64).tiny)
        nu_raw = 1.0 / jnp.sqrt(mu_s)
        above = nu_raw > nu_floor
        nu = jnp.where(above, nu_raw, nu_floor)  # zero gradient when clamped
        return nu * jnp.log((nu + 0.5) / (nu - 0.5))

    return phi


def entanglement_hamiltonian(V_A, nu_floor=0.5 * (1.0 + 1e-14)):
    """Quadratic entanglement Hamiltonian G of a Gaussian state, rho ~ exp(-R^T G R/2).

    G = V^{-1/2} phi(A^T A) V^{-1/2} with A = V^{-1/2} Omega V^{-1/2}, equal to
    the Williamson form S^{-T} diag(g(nu), g(nu)) S^{-1} of the numpy core
    (design note iv).  nu is clamped at ``nu_floor`` (static float) before the
    log exactly as in the core.
    """
    V = _sym(V_A)
    N = V.shape[0] // 2
    _check_pd(V, "V_A")
    Xi = pd_invsqrt(V)
    A = Xi @ Omega(N) @ Xi
    A = 0.5 * (A - A.T)
    M = A.T @ A  # symmetric PSD, eigenvalues 1/nu^2 in exact pairs
    phi = _phi_factory(nu_floor)
    F, M_invhalf = _sym_funms(
        M, (), (phi, _f_invsqrt), (_dd_generic(phi), _dd_invsqrt), True
    )
    # Symplectic-partner symmetrization.  J = A (A^T A)^{-1/2} is the
    # orthogonal complex structure of A (J^T = -J, J^2 = -I, [J, A] = 0)
    # mapping each Williamson x-direction onto its p-partner.  Exactly,
    # phi(A^T A) commutes with J and F' = F; in float64 the vacuum-floor
    # cluster (|nu - 1/2| ~ 1e-14, a dozen modes in a critical interval) is
    # returned by eigh in a scrambled basis with phi NOT constant across it
    # (clamped vs noise-level nu), and F alone then breaks the x/p pairing
    # that makes the first law's large floor contributions cancel.  F'
    # commutes with J, hence is of the Williamson form S^{-T} diag(g, g)
    # S^{-1} with equal g on partners, exactly like the numpy core's
    # Schur-block construction.  Same function of A => same gradient.
    J = A @ M_invhalf
    F = 0.5 * (F + J @ F @ J.T)
    return _sym(Xi @ F @ Xi)


def modular_energy(G, V_A):
    """<K> = (1/2) Tr(G V_A) for K = (1/2) R^T G R (mean-zero state)."""
    return 0.5 * jnp.trace(_asarray(G) @ _asarray(V_A))


# ----------------------------------------------------------------------------
# dynamics
# ----------------------------------------------------------------------------

def symplectic_from_quadratic(H_mat, t):
    """S = exp(t Omega H_mat) for H = (1/2) R^T H_mat R (jax.scipy.linalg.expm)."""
    H_mat = _asarray(H_mat)
    N = H_mat.shape[0] // 2
    return _expm(_asarray(t) * (Omega(N) @ H_mat))


def symplectic_inverse(S):
    """S^{-1} = -Omega S^T Omega for symplectic S."""
    S = _asarray(S)
    Om = Omega(S.shape[0] // 2)
    return -Om @ S.T @ Om


def evolve(V, S):
    """V -> S V S^T."""
    S = _asarray(S)
    return S @ _asarray(V) @ S.T


def mean_energy(V, K):
    """<H> of a mean-zero Gaussian state.

    K of shape (N, N): (1/2) Tr(V_pp) + (1/2) Tr(K V_xx) for H = (1/2)(p.p + x.K.x);
    K of shape (2N, 2N): (1/2) Tr(K V) for the general quadratic form.
    """
    V = _asarray(V)
    K = _asarray(K)
    N = V.shape[0] // 2
    if V.ndim != 2 or V.shape != (2 * N, 2 * N):
        raise ValueError(f"V must be a (2N, 2N) covariance, got {V.shape}")
    if K.shape == (N, N):
        return 0.5 * jnp.trace(V[N:, N:]) + 0.5 * jnp.trace(K @ V[:N, :N])
    if K.shape == (2 * N, 2 * N):
        return 0.5 * jnp.trace(K @ V)
    raise ValueError(
        f"K must be an (N, N) coupling matrix or a (2N, 2N) quadratic form "
        f"for a {2 * N}-dimensional covariance, got {K.shape}"
    )


def _chain_H(K):
    N = K.shape[0]
    Z = jnp.zeros((N, N), dtype=_F64)
    return jnp.block([[K, Z], [Z, jnp.eye(N, dtype=_F64)]])


def _chain_propagator_expm(K, t):
    return symplectic_from_quadratic(_chain_H(_sym(K)), t)


@jax.custom_jvp
def _chain_propagator_spectral(K, t):
    K = _sym(K)
    w2, U = jnp.linalg.eigh(K)
    w = jnp.sqrt(w2)
    c = jnp.cos(w * t)
    s = jnp.sin(w * t)
    C = (U * c) @ U.T
    Sw = (U * (s / w)) @ U.T
    Ws = (U * (w * s)) @ U.T
    return jnp.block([[C, Sw], [-Ws, C]])


@_chain_propagator_spectral.defjvp
def _chain_propagator_spectral_jvp(primals, tangents):
    K, t = primals
    out = _chain_propagator_spectral(K, t)
    _, dout = jax.jvp(_chain_propagator_expm, primals, tangents)  # exact tangent
    return out, dout


def chain_propagator(K, t, method="spectral"):
    """S(t) = exp(t Omega diag(K, I)) of the lattice field H = (1/2)(p.p + x.K.x).

    'spectral' (default) is the exact normal-mode form via eigh(K) with its
    tangent taken from the matrix exponential (design note v); 'expm' calls
    :func:`symplectic_from_quadratic` directly.
    """
    K = _sym(K)
    t = _asarray(t)
    if method == "spectral":
        _check_pd(K, "K")
        return _chain_propagator_spectral(K, t)
    if method == "expm":
        return _chain_propagator_expm(K, t)
    raise ValueError(f"method must be 'spectral' or 'expm', got {method!r}")


# ----------------------------------------------------------------------------
# channels
# ----------------------------------------------------------------------------

def apply_channel(V, X, Y):
    """V -> X V X^T + Y."""
    X = _asarray(X)
    return X @ _asarray(V) @ X.T + _asarray(Y)


def _diag_channel(N, modes, x_val, y_val):
    q = _quad_indices(N, modes)
    X = jnp.eye(2 * N, dtype=_F64).at[q, q].set(x_val)
    Y = jnp.zeros((2 * N, 2 * N), dtype=_F64).at[q, q].set(y_val)
    return X, Y


def loss_XY(N, modes, eta, nbar=0.0):
    """(X, Y) of the loss channel: X = sqrt(eta), Y = (1-eta)(nbar+1/2) on `modes`."""
    eta = _asarray(eta)
    nbar = _asarray(nbar)
    _maybe_check((eta >= 0.0) & (eta <= 1.0), f"eta must lie in [0, 1], got {eta}")
    _maybe_check(nbar >= 0.0, f"nbar must be >= 0, got {nbar}")
    return _diag_channel(N, modes, jnp.sqrt(eta), (1.0 - eta) * (nbar + 0.5))


def loss(V, modes, eta, nbar=0.0):
    """Beamsplitter loss of transmissivity eta into a thermal environment nbar."""
    N = _asarray(V).shape[0] // 2
    X, Y = loss_XY(N, modes, eta, nbar)
    return apply_channel(V, X, Y)


def thermal_noise_XY(N, modes, nbar_added):
    """(X, Y) of classical Gaussian noise addition: X = I, Y = nbar_added on `modes`."""
    nbar_added = _asarray(nbar_added)
    _maybe_check(nbar_added >= 0.0, f"nbar_added must be >= 0, got {nbar_added}")
    return _diag_channel(N, modes, 1.0, nbar_added)


def thermal_noise(V, modes, nbar_added):
    """Add nbar_added of classical Gaussian noise to each quadrature variance."""
    N = _asarray(V).shape[0] // 2
    X, Y = thermal_noise_XY(N, modes, nbar_added)
    return apply_channel(V, X, Y)


def amplifier_XY(N, modes, gain, nbar=0.0):
    """(X, Y) of the phase-insensitive amplifier: X = sqrt(gain), Y = (gain-1)(nbar+1/2)."""
    gain = _asarray(gain)
    nbar = _asarray(nbar)
    _maybe_check(gain >= 1.0, f"gain must be >= 1, got {gain}")
    _maybe_check(nbar >= 0.0, f"nbar must be >= 0, got {nbar}")
    return _diag_channel(N, modes, jnp.sqrt(gain), (gain - 1.0) * (nbar + 0.5))


def amplifier(V, modes, gain, nbar=0.0):
    """Phase-insensitive amplification of the listed modes by gain >= 1."""
    N = _asarray(V).shape[0] // 2
    X, Y = amplifier_XY(N, modes, gain, nbar)
    return apply_channel(V, X, Y)


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------

def harmonic_chain_K(N, m, bc="periodic"):
    """K = m^2 I + chain Laplacian (Srednicki normalization); m may be traced."""
    L = jnp.asarray(_models.harmonic_chain_K(N, 0.0, bc=bc), dtype=_F64)
    m = _asarray(m)
    return L + (m ** 2) * jnp.eye(N, dtype=_F64)
