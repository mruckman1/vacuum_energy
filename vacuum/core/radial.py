"""Srednicki radial-lattice geometry for the 3D massless scalar (Layer 1).

Partial-wave decomposition of the massless scalar field in 3D on a radial
lattice of N sites (lattice spacing a = 1), following M. Srednicki,
"Entropy and area", Phys. Rev. Lett. 71, 666 (1993). For each angular
momentum l the radial Hamiltonian is (Srednicki Eq. 10, hbar = 1)

    H_l = (1/2) sum_{j=1}^{N} pi_j^2
        + (1/2) sum_{j=1}^{N} [l(l+1)/j^2] phi_j^2
        + (1/2) sum_{j=1}^{N} (j+1/2)^2 (phi_{j+1}/(j+1) - phi_j/j)^2,

with the Dirichlet outer boundary condition phi_{N+1} = 0. Writing
H_l = (1/2)(pi.pi + phi.K_l.phi), expanding the difference term gives the
symmetric tridiagonal coupling matrix (0-based row j-1 <-> lattice site j)

    K_l[0, 0]         = 9/4 + l(l+1)                             (site j = 1)
    K_l[j-1, j-1]     = ((j-1/2)^2 + (j+1/2)^2)/j^2 + l(l+1)/j^2 (2 <= j <= N)
    K_l[j-1, j] = K_l[j, j-1] = -(j+1/2)^2 / (j (j+1))           (1 <= j <= N-1)

Derivation note (verified independently from H_l): the site-j diagonal picks
up (j+1/2)^2/j^2 from the difference term with index j and (j-1/2)^2/j^2 from
the term with index j-1; site 1 has no j = 0 term, hence 9/4 = (3/2)^2. A
sometimes-quoted variant truncates the coupling sum at j = N-1 (free outer
boundary), which would give K_l[N-1, N-1] = (N-1/2)^2/N^2 + l(l+1)/N^2; that
variant is *singular* at l = 0 (phi_j proportional to j is an exact zero mode
of its potential), leaves the l = 0 ground state non-normalizable, and is not
what Srednicki used. We therefore keep Srednicki's phi_{N+1} = 0 term, whose
site-N diagonal is ((N-1/2)^2 + (N+1/2)^2)/N^2 + l(l+1)/N^2.

The ground state of each H_l is Gaussian with covariance
V_l = diag-block(K_l^{-1/2}/2, K_l^{1/2}/2) (vacuum.core.gaussian
.ground_state_cov). Tracing out all sites but the inner n gives the entropy
S_l(n) of the ball of radius R = (n + 1/2) a; the full-sphere entropy sums the
partial waves with their degeneracy,

    S(n) = sum_{l>=0} (2l+1) S_l(n),

and obeys the area law S = c_A (R/a)^2 with c_A ~= 0.30 (Srednicki 1993;
refined value ~0.295, cf. Lohmayer-Neuberger-Schwimmer-Theisen, Phys. Lett. B
685, 222 (2010)).

Conventions: hbar = 1, block quadrature ordering R = (x, p), vacuum
symplectic eigenvalue 1/2, entropies in nats (docs/API.md).
"""

from __future__ import annotations

import warnings

import numpy as np

from .gaussian import entropy_from_nu

__all__ = ["srednicki_K", "sphere_entropy", "partial_wave_entropies"]

#: default cap on the adaptive angular-momentum sum in sphere_entropy
L_CAP_DEFAULT = 2000


def srednicki_K(N, l):
    """Coupling matrix K_l of the Srednicki radial Hamiltonian.

    H_l = (1/2)(pi.pi + phi.K_l.phi) for the l-th partial wave of the 3D
    massless scalar on an N-site radial lattice with phi_{N+1} = 0
    (Srednicki, PRL 71, 666 (1993), Eq. 10; see module docstring for the
    explicit entries and derivation note on the outer boundary).

    Parameters
    ----------
    N : int
        Number of radial sites (N >= 2).
    l : int
        Angular momentum (l >= 0).

    Returns
    -------
    (N, N) ndarray
        Symmetric positive-definite tridiagonal coupling matrix.
    """
    N = int(N)
    l = int(l)
    if N < 2:
        raise ValueError(f"need N >= 2 radial sites, got N={N}")
    if l < 0:
        raise ValueError(f"angular momentum must be >= 0, got l={l}")

    j = np.arange(1, N + 1, dtype=float)  # lattice site index, 1-based
    ll1 = float(l * (l + 1))

    diag = ((j - 0.5) ** 2 + (j + 0.5) ** 2) / j**2 + ll1 / j**2
    diag[0] = 2.25 + ll1  # site j=1: only the (3/2)^2 difference term
    off = -((j[:-1] + 0.5) ** 2) / (j[:-1] * (j[:-1] + 1.0))

    K = np.diag(diag)
    idx = np.arange(N - 1)
    K[idx, idx + 1] = off
    K[idx + 1, idx] = off
    return K


def partial_wave_entropies(K, ns):
    """Entanglement entropies S_l(n) of the inner-n-site blocks, one K_l.

    Ground state of H = (1/2)(pi.pi + phi.K.phi): V = diag-block(X, P) with
    X = K^{-1/2}/2, P = K^{1/2}/2 (gaussian.ground_state_cov). For an
    X-P block-diagonal covariance the reduced state of a site set A is
    diag-block(X_A, P_A) and its symplectic eigenvalues are

        nu_k = sqrt(eig(X_A P_A))  =  sqrt(eig(X_A^{1/2} P_A X_A^{1/2})),

    since (i Omega V_A)^2 = diag-block(P_A X_A, X_A P_A). This is Srednicki's
    (1993) reduction, and agrees with gaussian.entropy(gaussian.reduce(V, A))
    to roundoff (regression-tested); it does one N x N eigendecomposition per
    l and one n x n eigenproblem per block size, so a full l-sweep stays fast.

    Parameters
    ----------
    K : (N, N) ndarray
        Symmetric positive-definite coupling matrix.
    ns : sequence of int
        Block sizes; entry n means the inner sites 1..n (modes 0..n-1).

    Returns
    -------
    (len(ns),) ndarray
        Von Neumann entropies in nats, one per requested n.
    """
    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    ns = np.asarray(ns, dtype=int)
    if ns.size and (ns.min() < 1 or ns.max() > N - 1):
        raise ValueError(f"block sizes must satisfy 1 <= n <= N-1={N - 1}, got {ns}")

    w, U = np.linalg.eigh(0.5 * (K + K.T))
    if w.min() <= 0.0:
        raise ValueError(f"K is not positive definite: min eigenvalue {w.min():.3e}")
    sw = np.sqrt(w)
    X = 0.5 * (U / sw) @ U.T  # <phi phi> = K^{-1/2}/2
    P = 0.5 * (U * sw) @ U.T  # <pi pi>   = K^{1/2}/2

    out = np.empty(ns.size)
    for i, n in enumerate(ns):
        Xa = X[:n, :n]
        Pa = P[:n, :n]
        wa, Ua = np.linalg.eigh(0.5 * (Xa + Xa.T))
        Xa_half = (Ua * np.sqrt(np.clip(wa, 0.0, None))) @ Ua.T
        M = Xa_half @ Pa @ Xa_half
        nu2 = np.linalg.eigvalsh(0.5 * (M + M.T))
        nu = np.sqrt(np.clip(nu2, 0.25, None))  # vacuum floor nu = 1/2
        out[i] = entropy_from_nu(nu)
    return out


def sphere_entropy(N, n, l_max=None, tol=1e-8, return_diagnostics=False):
    """Entanglement entropy of a sphere in the 3D massless scalar vacuum.

    S(n) = sum_{l>=0} (2l+1) S_l(n) in nats, where S_l(n) is the entropy of
    the inner n sites of the ground state of the Srednicki radial Hamiltonian
    H_l on N sites (Srednicki, PRL 71, 666 (1993)). By purity of the global
    ground state this equals the entropy of the exterior region, and it obeys
    the area law S ~= 0.30 (n + 1/2)^2 (refined coefficient ~0.295).

    The l-sum is extended adaptively until the tail term (2l+1) S_l(n) drops
    below `tol` for two consecutive l (for every requested n), or the cap is
    hit. The l-sweep is the expensive part, so `n` may be a sequence: all
    block entropies for a given l are read off one ground-state covariance.

    Parameters
    ----------
    N : int
        Number of radial lattice sites.
    n : int or sequence of int
        Inner block size(s), 1 <= n <= N-1 (radius R = (n + 1/2) a).
    l_max : int, optional
        Cap on the angular-momentum sum (default 2000). If the tolerance is
        not reached below the cap, the returned diagnostics carry
        ``converged=False`` and a RuntimeWarning is emitted; the truncation
        error is then of order ``max_tail_term * l_stop`` (the tail decays
        like a power of l).
    tol : float
        Tail-term tolerance on (2l+1) S_l (nats), default 1e-8.
    return_diagnostics : bool
        If True, return ``(S, diagnostics)`` where diagnostics is a dict with
        keys ``l_stop`` (last l summed), ``converged``, ``max_tail_term``
        (largest (2l+1) S_l(n) at l_stop over the requested n), ``cap``,
        ``tol``, and ``terms`` (the per-l summed tail terms max'd over n).

    Returns
    -------
    float or ndarray
        S(n) in nats — a float for scalar `n`, an array matching `n`
        otherwise. With ``return_diagnostics=True``, a ``(value, dict)``
        tuple.
    """
    n_arr = np.atleast_1d(np.asarray(n, dtype=int))
    scalar = np.isscalar(n) or (isinstance(n, np.ndarray) and n.ndim == 0)
    cap = L_CAP_DEFAULT if l_max is None else int(l_max)
    if cap < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}")

    S = np.zeros(n_arr.size)
    terms = []
    below = 0  # consecutive l with every tail term below tol
    converged = False
    l_stop = 0
    max_tail = np.inf

    for l in range(cap + 1):
        Sl = partial_wave_entropies(srednicki_K(N, l), n_arr)
        term = (2 * l + 1) * Sl
        S += term
        max_tail = float(term.max())
        terms.append(max_tail)
        l_stop = l
        below = below + 1 if max_tail < tol else 0
        if below >= 2:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"sphere_entropy(N={N}): l-sum hit the cap l_max={cap} with "
            f"max tail term {max_tail:.3e} > tol={tol:.1e}; result is a "
            f"(slightly) truncated l-sum.",
            RuntimeWarning,
            stacklevel=2,
        )

    value = float(S[0]) if scalar else S
    if return_diagnostics:
        diagnostics = {
            "l_stop": l_stop,
            "converged": converged,
            "max_tail_term": max_tail,
            "cap": cap,
            "tol": tol,
            "terms": np.asarray(terms),
        }
        return value, diagnostics
    return value
