"""Toy Jacobson (PLAN.md Layer 4 discovery target / Layer 7 leap L2):
entanglement equilibrium as a constraint on lattice dynamics.

The argument being toyed with
-----------------------------
Jacobson, PRL 116, 201101 (2016), arXiv:1505.04753, Eqs. (14)-(19):
the vacuum restricted to a small ball is thermal with respect to its
modular Hamiltonian K, rho = e^{-K}/Z, so any first-order state variation
obeys the entanglement first law delta S = delta<K> (his Eq. (16); Blanco-
Casini-Hung-Myers JHEP 08 (2013) 060). That identity carries no dynamical
content by itself. The content enters through Eq. (17): for a CONFORMAL
field the ball's K is the LOCAL conformal-Killing energy

    K = 2 pi int_B (R^2 - r^2)/(2R) T_00 d^{d-1}x

(Casini-Huerta-Myers, JHEP 05 (2011) 036, arXiv:1102.0440, Sec. 2: the
causal diamond of the ball is conformally the Rindler wedge, whose modular
Hamiltonian is the boost by Bisognano-Wichmann, J. Math. Phys. 17, 303
(1976)), and for an excitation much larger than the ball delta<K> becomes
proportional to the ball's energy, delta<K> = (Omega_{d-2} R^d/(d^2-1))
delta<T_00> (Jacobson Eq. (19)). Locality of K is a property of Lorentz-
(indeed conformally-) invariant dynamics: the exact modular Hamiltonian of
a ball in a generic lattice theory is nonlocal, and the first law then
says nothing about energy. On a 1+1 lattice with no gravity anywhere the
question of leap L2 is therefore: **how strongly does demanding
delta S = delta<K_loc> for every small ball, with K_loc the local
CHM/BW form built from the lattice's own energy density, pin the lattice
dynamics to the wave equation?**

Definitions used here
---------------------
* Dynamics: H = (1/2)(p.p + x^T K x) on an N-site chain (Dirichlet walls,
  N even). Balls are intervals of even length L centred on the chain.
* Local energy density (PSD bond split, the convention of
  vacuum.inequalities.qei): h_i = p_i^2/2 + M_i x_i^2/2 + (1/4) sum_j c_ij
  (x_i - x_j)^2 with c_ij = -K_ij (bond stiffness) and M_i = sum_j K_ij
  (on-site term; wall bonds sit wholly on the wall site), so that
  sum_i h_i = H exactly. `split='site'` keeps the older symmetrized
  x_i (K x)_i form; the two differ by a discrete divergence.
* Local ansatz: K_loc = 2 pi sum_{i in B} beta_i h_i with the discrete CHM
  weight beta_i = (i - a + 1/2)(b - i + 1/2)/(b - a + 1) = (R^2 - x_i^2)/(2R)
  on B = [a, b] (bw_weights). h_i at the edges contains half a bond that
  straddles the boundary; that is a lattice-scale ambiguity of order 1/L.
* Variations: Jacobson's excitation is long-wavelength, "ell <<
  L_excitation" (his Eq. (18)). The lattice version used here squeezes one
  normal mode u_a of K of wavelength lambda ~ ratio * L: V -> S V S^T with
  x -> e^{eps} x, p -> e^{-eps} p along u_a. To first order
  delta V_xx = uu^T/omega_a, delta V_pp = -omega_a uu^T -- smooth on the
  lattice scale (the class of variation for which a lattice modular
  Hamiltonian has a continuum limit at all, Di Giulio-Tonni, J. Stat.
  Mech. (2020) 033102, arXiv:1911.07188, Eq. (20)), with
  delta<T_00>(x) = (eps/2) omega_a u_0^2 cos(2 k_a x) approximately constant
  over the ball. Local changes of the couplings are NOT used as
  variations: on a lattice they act as local rescalings whose xx and pp
  energy responses are cutoff-dominated and cancel to a delicate
  remainder -- the UV sensitivity Jacobson's Eq. (18) excludes.
* Parity. Modes with a node at the ball's centre (`parity='odd'`) are
  exactly blind to every reflection-even Williamson mode of the ball, in
  particular to the 1+1 massless scalar's zero mode phi_bar: the exact
  modular Hamiltonian carries a term ~ g(nu_1) phi_bar^2 (row sums
  sum_j G^{xx}_ij ~ 0.1 per row that vanish only like 1/ln(N)) with NO
  local-energy counterpart, the origin of the (1/2) ln ln(1/mL) term in the
  interval entropy of the 2d massless scalar (Casini-Huerta, J. Phys. A 42,
  504007 (2009)). Reflection-even variations (`parity='even'`) therefore
  have an O(1) residual for EVERY dynamics on the massless chain -- a
  property of the 2d scalar, not of the lattice -- and the odd sector is
  the derivative-algebra (genuine CFT) sector in which the toy is posed.
* Residual: r(K) = max over balls B and variations v of
  |delta S_v(B) - delta<K_loc>_v(B)| / |delta S_v(B)|, with delta S computed
  from the exact Gaussian modular Hamiltonian G_B (vacuum.core, Williamson;
  per-mode floor diagnostics, mpmath escalation of the margins nu - 1/2
  when the floor modes carry weight) and checked against finite entropy
  differences of the exactly squeezed ball state (first_law_audit).

* The free-weight residual (the reviewer's version of the test). r(K)
  ASSUMES the CHM weight, which is derived from conformal symmetry, so
  r -> 0 for the wave chain and r = O(1) for Lifshitz is a consistency
  check on CHM, not Lorentz invariance emerging from entanglement.
  `free_weight_residual` frees the weight: K_loc = sum_k w_k O_k over the
  lattice's local energy pieces (site terms and bonds of the PSD bond
  split), w >= 0 drawn from a low-parameter smooth family (even Chebyshev
  profiles, sampled at the sites or at the bond midpoints) or one weight
  per site, minimised by linear programming over a variation family that
  includes the two-mode squeezes of pairs of odd long-wavelength modes
  (`pair_variation`); the numerical rank of that family is reported next
  to the parameter count, because Jacobson's regime ball << wavelength is
  low-rank (a per-site weight has as many parameters as there are
  conditions and fits anything). `universal_weight_residual` shares one
  profile R^z f(x/R) across ball sizes, the only version in which a
  many-parameter profile is over-determined. r_free <= r by construction;
  the minimising weight is a result (for the wave chain: the CHM parabola,
  found), and the velocity is exactly absorbed (the vacuum of K and of
  v^2 K is the same state: v is a unit). `fit_dynamics(residual='free' |
  'universal')` re-runs the constrained fit with the weight free.

What this module does not claim: no gravity, no area term, no
Einstein equation; a 1+1 lattice with a single ball position; the exact
modular Hamiltonian's continuum limit is taken in the sense of smooth
variations only. The physics (residual map, constrained dynamics, the
free-weight verdict, limitations) is measured in papers/toy-jacobson/.

Legacy hooks (kept unchanged): entanglement_equilibrium_residual with
mode='exact' (the first-law anchor of tests/test_geometry_jacobson.py) and
mode='bw', and the discrete-family path of constrain_dynamics.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from vacuum.core import (
    entanglement_hamiltonian,
    entropy,
    ground_state_cov,
    modular_energy,
    reduce,
    symplectic_eigenvalues,
    symplectic_inverse,
    williamson,
)
from vacuum.core.precision import entropy_precise, symplectic_margins_mp

__all__ = [
    # legacy
    "entanglement_equilibrium_residual",
    "constrain_dynamics",
    "local_energy_forms",
    "bw_weights",
    # models
    "neighbour_laplacian",
    "coupling_chain_K",
    "random_symmetric_chain_K",
    "dispersion_power_K",
    "bulk_dispersion",
    "dispersion_departure",
    # local form and variations
    "centred_ball",
    "local_modular_form",
    "squeeze_modes",
    "squeeze_variation",
    "squeezed_ball_covariance",
    # residuals and audits
    "equilibrium_residual",
    "first_law_audit",
    "eh_profiles",
    "fit_dynamics",
    "residual_scan",
    "k4_sensitivity",
]

# Must match the clamp inside vacuum.core.entanglement_hamiltonian.
_NU_FLOOR = 0.5 * (1.0 + 1e-14)


def _g_of_nu(nu):
    nu = np.maximum(np.asarray(nu, dtype=float), _NU_FLOOR)
    return np.log((nu + 0.5) / (nu - 0.5))


# ---------------------------------------------------------------------------
# lattice models
# ---------------------------------------------------------------------------


def neighbour_laplacian(N, n):
    """n-th-neighbour Dirichlet Laplacian L_n: diagonal 2, -1 at |i - j| = n.

    Sites within n of a wall keep the diagonal 2 (their missing partner is
    the wall's clamped phi = 0), the same convention as
    vacuum.core.harmonic_chain_K(bc='dirichlet') for n = 1. Bulk symbol
    2 - 2 cos(nk) = 4 sin^2(nk/2).
    """
    N, n = int(N), int(n)
    if n < 1:
        raise ValueError(f"neighbour range n must be >= 1, got {n}")
    K = np.zeros((N, N))
    np.fill_diagonal(K, 2.0)
    if n < N:
        i = np.arange(N - n)
        K[i, i + n] = -1.0
        K[i + n, i] = -1.0
    return K


def coupling_chain_K(N, c, m=0.0):
    """K = m^2 I + sum_{n>=1} c_n L_n (c = (c_1, c_2, ...)).

    Bulk dispersion omega^2(k) = m^2 + sum_n 4 c_n sin^2(nk/2), small-k
    velocity v^2 = sum_n c_n n^2. c = (1,) is the wave chain of
    harmonic_chain_K; c = (4/3, -1/12) is the fourth-order-accurate
    ("improved") lattice Laplacian, omega^2 = k^2 + O(k^6).
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    K = float(m) ** 2 * np.eye(int(N))
    for n, cn in enumerate(c, start=1):
        if cn != 0.0:
            K = K + cn * neighbour_laplacian(N, n)
    return K


def random_symmetric_chain_K(N, W, seed=0, m=0.0):
    """Nearest-neighbour chain with random bond stiffness c_x ~ U[1-W, 1+W],
    mirrored about the chain centre (c_x = c_{N-2-x}) so that reflection
    parity -- which the odd-mode variation family relies on -- stays exact.
    Dirichlet walls carry unit stiffness. Returns K (symmetric, positive
    definite for W < 1)."""
    N = int(N)
    W = float(W)
    if not 0.0 <= W < 1.0:
        raise ValueError(f"disorder strength must satisfy 0 <= W < 1, got {W}")
    rng = np.random.default_rng(seed)
    nb = N - 1
    half = rng.uniform(1.0 - W, 1.0 + W, size=(nb + 1) // 2)
    c = np.concatenate([half, half[::-1][nb % 2:]])[:nb]
    K = float(m) ** 2 * np.eye(N)
    idx = np.arange(nb)
    K[idx, idx] += c
    K[idx + 1, idx + 1] += c
    K[idx, idx + 1] -= c
    K[idx + 1, idx] -= c
    K[0, 0] += 1.0
    K[N - 1, N - 1] += 1.0
    return K


def dispersion_power_K(N, s):
    """K = L^s with L the Dirichlet Laplacian: omega(k) = (2 sin(k/2))^s,
    dynamical exponent z = s (s = 2 is the Lifshitz chain omega ~ k^2/2,
    s = 1/2 the long-range omega ~ sqrt(k)). Same eigenvectors as L."""
    lam, U = np.linalg.eigh(neighbour_laplacian(N, 1))
    return (U * lam ** float(s)) @ U.T


def bulk_dispersion(c, m, k):
    """omega(k) = sqrt(m^2 + sum_n 4 c_n sin^2(nk/2)) for coupling_chain_K."""
    c = np.atleast_1d(np.asarray(c, dtype=float))
    k = np.asarray(k, dtype=float)
    w2 = float(m) ** 2 * np.ones_like(k)
    for n, cn in enumerate(c, start=1):
        w2 = w2 + 4.0 * cn * np.sin(0.5 * n * k) ** 2
    return np.sqrt(np.maximum(w2, 0.0))


def dispersion_departure(c, m, L, kind="v1", n_k=64):
    """Departure of the bulk dispersion from Lorentz invariance at the
    ball's scale k_L = pi / L (wavelength 2L).

    kind='v1'    : |omega(k_L)/k_L - 1| -- deviation from the massless
                   dispersion with the unit velocity the local ansatz assumes
                   (velocity mismatch, mass and curvature all count).
    kind='shape' : relative L2 residual of the best linear fit omega ~ v k
                   through the origin on (0, k_L] (velocity-agnostic).
    """
    kL = np.pi / float(L)
    if kind == "v1":
        return float(abs(bulk_dispersion(c, m, kL) / kL - 1.0))
    if kind == "shape":
        k = kL * (np.arange(1, n_k + 1) / n_k)
        w = bulk_dispersion(c, m, k)
        v = float(np.dot(w, k) / np.dot(k, k))
        return float(np.linalg.norm(w - v * k) / np.linalg.norm(w))
    raise ValueError("kind must be 'v1' or 'shape'")


# ---------------------------------------------------------------------------
# local energy density and the CHM/BW local form
# ---------------------------------------------------------------------------


def local_energy_forms(K, split="bond"):
    """Quadratic forms H_i (2N x 2N) with h_i = (1/2) R^T H_i R the local
    energy density of H = (1/2)(p.p + x^T K x); sum_i H_i = diag(K, I).

    split='bond' : PSD bond split (vacuum.inequalities.qei convention):
                   h_i = p_i^2/2 + M_i x_i^2/2 + (1/4) sum_j c_ij (x_i - x_j)^2,
                   c_ij = -K_ij, M_i = sum_j K_ij. PSD whenever all bond
                   stiffnesses and on-site terms are non-negative.
    split='site' : the symmetrized h_i = p_i^2/2 + (1/2) x_i (K x)_i.
    """
    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    forms = []
    if split == "bond":
        c = -K.copy()
        np.fill_diagonal(c, 0.0)
        M = K.sum(axis=1)
        for i in range(N):
            H = np.zeros((2 * N, 2 * N))
            H[N + i, N + i] = 1.0
            H[i, i] += M[i]
            js = np.nonzero(c[i])[0]
            for j in js:
                cij = 0.5 * c[i, j]
                H[i, i] += cij
                H[j, j] += cij
                H[i, j] -= cij
                H[j, i] -= cij
            forms.append(H)
    elif split == "site":
        for i in range(N):
            H = np.zeros((2 * N, 2 * N))
            H[N + i, N + i] = 1.0
            H[i, :N] += 0.5 * K[i, :]
            H[:N, i] += 0.5 * K[i, :]
            forms.append(H)
    else:
        raise ValueError("split must be 'bond' or 'site'")
    return forms


def bw_weights(ball, shift=0.0, x=None, wall=None):
    """Discrete Bisognano-Wichmann / CHM profile on the interval ball = [a, ..., b].

    beta(x) = (x - x_a)(x_b - x)/(x_b - x_a), the 1+1D weight (R^2 - x^2)/(2R),
    with the ball boundaries at x_a = a - 1/2 - shift/2 and x_b = b + 1/2 +
    shift/2 (site i at position i). shift = 0 (default) is beta_i =
    (i - a + 1/2)(b - i + 1/2)/(b - a + 1) at the site centres, R = (b - a + 1)/2;
    shift = +1 puts the boundary on the outer site of each straddling bond,
    shift = -1 on the edge site itself. `x` (default: the ball's site
    positions) samples the weight elsewhere, e.g. at bond midpoints; outside
    the ball the weight is clipped to 0.

    `wall` = position of a Dirichlet wall (the clamped site, i.e. the node of
    every normal mode: -1 for the left wall of vacuum.core.harmonic_chain_K on
    sites 0..N-1, N for the right one). Two cases:
      * ball ATTACHED to the wall (it starts at the first live site, A <= 1):
        the wall is not an entangling surface, the region is (0, B) of a
        boundary CFT and the conformal-Killing weight is exactly
        beta(y) = (B^2 - y^2)/(2B), y = x - wall, B = x_b - wall (the A -> 0
        limit of the image formula below; Bisognano-Wichmann on a half line
        with one entangling point).
      * ball DETACHED: the local weight 2 pi/z'(x) of the interval together
        with its mirror image (the Casini-Huerta z of
        vacuum.modular.casini_huerta for the pair (x_a, x_b),
        (2 wall - x_b, 2 wall - x_a)), which reduces to the parabola far from
        the wall. This is only the LOCAL part: an interval at a distance from
        a boundary also carries a bilocal term coupling each point to its
        image partner (Eisler-Tonni-Peschel 2022, J. Stat. Mech. 083101,
        Sec. 6), which no local weight can represent.
    """
    ball = np.asarray(sorted(ball), dtype=float)
    a, b = ball[0], ball[-1]
    xa, xb = a - 0.5 - 0.5 * float(shift), b + 0.5 + 0.5 * float(shift)
    xs = np.asarray(ball if x is None else x, dtype=float)
    if wall is None:
        return np.maximum((xs - xa) * (xb - xs) / (xb - xa), 0.0)
    y = xs - float(wall)
    A, B = xa - float(wall), xb - float(wall)
    if A <= 1.0:  # attached to the wall: one entangling point, exact BCFT weight
        return np.maximum((B * B - y * y) / (2.0 * B), 0.0)
    inside = (y > A) & (y < B)
    out = np.zeros_like(y)
    yi = y[inside]
    zp = 1.0 / (yi - A) - 1.0 / (yi - B) + 1.0 / (yi + B) - 1.0 / (yi + A)
    out[inside] = 1.0 / zp
    return out


def centred_ball(N, L):
    """The interval of even length L centred on the chain of even length N
    (centre (N-1)/2, the node of every reflection-odd normal mode)."""
    N, L = int(N), int(L)
    if N % 2 or L % 2 or L > N:
        raise ValueError(f"need even N and even L <= N, got N={N}, L={L}")
    a0 = N // 2 - L // 2
    return list(range(a0, a0 + L))


def _halo(K, ball):
    """ball plus every site coupled to it by K (sorted), and the positions of
    the ball sites inside that list."""
    K = np.asarray(K)
    ball = [int(i) for i in ball]
    touched = set(ball)
    for i in ball:
        touched.update(int(j) for j in np.nonzero(K[i])[0])
    sites = sorted(touched)
    pos = {s: n for n, s in enumerate(sites)}
    return sites, [pos[i] for i in ball]


def local_modular_form(K, ball, split="bond", sites=None, beta_at="site", shift=0.0,
                       wall=None, mix=0.5):
    """G_loc of the local ansatz K_loc = 2 pi sum_{i in ball} beta_i h_i as a
    (2n x 2n) quadratic form on `sites` (default: the ball plus the sites its
    edge bonds reach), block ordering. <K_loc> = (1/2) Tr(G_loc V_sites).

    Discretisations of int_B beta(x) T_00(x) dx (all differ at O(1/L^2), the
    ansatz floor measured in papers/toy-jacobson):
      beta_at='site'  every term of h_i (site terms and half of each bond)
                      carries beta at the site centre; a bond straddling the
                      boundary contributes half its energy with the edge
                      site's weight;
      beta_at='bond'  site terms carry beta at the site, each bond (counted
                      once) carries beta at its midpoint -- the midpoint rule
                      for the gradient energy; a nearest-neighbour bond
                      straddling the boundary has its midpoint on the boundary
                      and drops out (shift = 0);
      beta_at='mix'   mix * G_site + (1 - mix) * G_bond (a Richardson pair);
    `shift` moves the ball boundaries by shift/2 (bw_weights); `wall` uses
    the image-corrected local weight near a Dirichlet wall. split='site' is
    the older symmetrized form (beta at sites only).
    """
    K = np.asarray(K, dtype=float)
    ball = [int(i) for i in ball]
    if sites is None:
        sites, _ = _halo(K, ball)
    sites = [int(s) for s in sites]
    if beta_at == "mix":
        Gs = local_modular_form(K, ball, split, sites, "site", shift, wall)
        Gb = local_modular_form(K, ball, split, sites, "bond", shift, wall)
        return float(mix) * Gs + (1.0 - float(mix)) * Gb
    if beta_at not in ("site", "bond"):
        raise ValueError("beta_at must be 'site', 'bond' or 'mix'")
    n = len(sites)
    pos = {s: k for k, s in enumerate(sites)}
    beta = 2.0 * np.pi * bw_weights(ball, shift=shift, wall=wall)
    G = np.zeros((2 * n, 2 * n))
    if split == "bond" and beta_at == "bond":
        M = K.sum(axis=1)
        inball = set(ball)
        for bt, i in zip(beta, sorted(ball)):
            pi = pos[i]
            G[n + pi, n + pi] += bt
            G[pi, pi] += bt * M[i]
        for i in sorted(ball):
            pi = pos[i]
            for j in np.nonzero(K[i])[0]:
                j = int(j)
                if j == i or (j in inball and j < i):
                    continue
                wb = 2.0 * np.pi * float(bw_weights(ball, shift=shift, x=[0.5 * (i + j)], wall=wall)[0])
                if wb == 0.0:
                    continue  # a bond whose midpoint is outside the ball carries no weight
                cij = -wb * K[i, j]  # wb c_ij: the whole bond once, at its midpoint
                if j not in pos:
                    raise ValueError(
                        f"bond ({i}, {j}) carries weight {wb:.3e} but site {j} is not in `sites`")
                pj = pos[j]
                G[pi, pi] += cij
                G[pj, pj] += cij
                G[pi, pj] -= cij
                G[pj, pi] -= cij
    elif split == "bond":
        M = K.sum(axis=1)
        for bt, i in zip(beta, sorted(ball)):
            pi = pos[i]
            G[n + pi, n + pi] += bt
            G[pi, pi] += bt * M[i]
            for j in np.nonzero(K[i])[0]:
                if j == i:
                    continue
                cij = -0.5 * bt * K[i, j]  # (bt/2) c_ij with c_ij = -K_ij
                pj = pos[int(j)]
                G[pi, pi] += cij
                G[pj, pj] += cij
                G[pi, pj] -= cij
                G[pj, pi] -= cij
    elif split == "site":
        for bt, i in zip(beta, sorted(ball)):
            pi = pos[i]
            G[n + pi, n + pi] += bt
            for j in np.nonzero(K[i])[0]:
                pj = pos[int(j)]
                G[pi, pj] += 0.5 * bt * K[i, j]
                G[pj, pi] += 0.5 * bt * K[i, j]
    else:
        raise ValueError("split must be 'bond' or 'site'")
    return G


# ---------------------------------------------------------------------------
# long-wavelength squeeze variations
# ---------------------------------------------------------------------------


def _parity(u):
    """+1 for a reflection-even vector, -1 for odd (normalized overlap)."""
    return float(np.dot(u, u[::-1]) / np.dot(u, u))


def squeeze_modes(eig, L, ratios=(4, 8, 16), parity="odd"):
    """Normal modes of K to squeeze, one per requested wavelength ratio.

    eig = (lam, U) from np.linalg.eigh(K) (ascending). For each ratio the
    target wavelength is ratio * L; the mode index a is chosen from the
    Dirichlet count 2(N+1)/(a+1) and then moved to the nearest index whose
    eigenvector has the requested reflection parity ('odd': node at the
    chain centre, exactly blind to the ball's even Williamson modes;
    'even': antinode). Returns a list of dicts with keys index, ratio,
    wavelength_target, wavelength (from the node count), parity, u, omega.
    """
    lam, U = eig
    N = U.shape[0]
    if parity == "any":
        want = None
    elif parity in ("odd", "even"):
        want = -1.0 if parity == "odd" else +1.0
    else:
        raise ValueError("parity must be 'odd', 'even' or 'any'")
    out = []
    for ratio in ratios:
        target = float(ratio) * L
        a0 = int(round(2.0 * (N + 1) / target)) - 1
        a0 = min(max(a0, 0), N - 1)
        chosen = a0 if want is None else None
        for step in range(0, 8):
            if chosen is not None:
                break
            for a in (a0 + step, a0 - step):
                if 0 <= a < N and np.sign(_parity(U[:, a])) == want:
                    chosen = a
                    break
        if chosen is None:
            raise RuntimeError(f"no {parity} mode near index {a0}")
        u = U[:, chosen]
        nodes = int(np.sum(np.sign(u[1:]) * np.sign(u[:-1]) < 0))
        out.append({
            "index": int(chosen),
            "ratio": float(ratio),
            "wavelength_target": target,
            "wavelength": 2.0 * (N + 1) / (nodes + 1),
            "parity": _parity(u),
            "u": u,
            "omega": float(np.sqrt(max(lam[chosen], 0.0))),
        })
    return out


def squeeze_variation(u_sites, omega):
    """First-order covariance change of a single-mode squeeze
    x -> e^{eps} x, p -> e^{-eps} p along the normal mode u (eps = 1 units),
    restricted to `sites`: delta V = diag(u u^T / omega, -omega u u^T)."""
    u = np.asarray(u_sites, dtype=float)
    n = u.size
    dV = np.zeros((2 * n, 2 * n))
    dV[:n, :n] = np.outer(u, u) / omega
    dV[n:, n:] = -omega * np.outer(u, u)
    return dV


def squeezed_ball_covariance(V_sites, u_sites, omega, eps):
    """Exact covariance of the squeezed vacuum S_eps V S_eps^T on `sites`,
    for an exact normal mode u of K (V_xx u = u/(2 omega), V_pp u = omega u/2):
    the mode's x-variance scales by e^{2 eps}, its p-variance by e^{-2 eps},
    a rank-one update in each sector. Checked against scipy expm to 2e-15."""
    V = np.array(V_sites, dtype=float, copy=True)
    u = np.asarray(u_sites, dtype=float)
    n = u.size
    V[:n, :n] += (np.exp(2.0 * eps) - 1.0) / (2.0 * omega) * np.outer(u, u)
    V[n:, n:] += (np.exp(-2.0 * eps) - 1.0) * (0.5 * omega) * np.outer(u, u)
    return V


def _vacuum_block(eig, sites):
    """Ground-state covariance of H = (1/2)(p.p + x K x) on `sites` from the
    eigendecomposition of K (V = diag(K^{-1/2}, K^{1/2})/2)."""
    lam, U = eig
    Us = U[sites, :]
    s = np.sqrt(lam)
    n = len(sites)
    V = np.zeros((2 * n, 2 * n))
    V[:n, :n] = 0.5 * (Us / s) @ Us.T
    V[n:, n:] = 0.5 * (Us * s) @ Us.T
    return 0.5 * (V + V.T)


# ---------------------------------------------------------------------------
# the entanglement-equilibrium residual
# ---------------------------------------------------------------------------


def _mode_contributions(V_B, dV_B, floor_tol, escalate_above, dps):
    """Williamson-mode decomposition of delta S = (1/2) Tr(G_B dV_B).

    Returns (dS, info): contributions per mode (1/2) g(nu_k) (W_kk + W_{L+k,L+k}),
    W = S^{-1} dV S^{-T}; floor modes (nu - 1/2 < floor_tol) get their margins
    re-evaluated in mpmath when they carry more than `escalate_above` of |dS|
    (the modular temperatures g are refined; the float64 symplectic basis is
    kept, which is exact for the sum over a degenerate floor cluster)."""
    S_w, nu = williamson(V_B)
    Sinv = symplectic_inverse(S_w)
    W = Sinv @ dV_B @ Sinv.T
    L = nu.size
    d = np.diag(W)
    weights = 0.5 * (d[:L] + d[L:])
    g = _g_of_nu(nu)
    contrib = g * weights
    dS = float(np.sum(contrib))
    floor_mask = (nu - 0.5) < floor_tol
    floor_part = float(np.sum(contrib[floor_mask]))
    scale = abs(dS) if dS != 0.0 else 1.0
    escalated = False
    margins_mp = None
    if floor_mask.any() and abs(floor_part) > escalate_above * scale:
        escalated = True
        margins_mp = symplectic_margins_mp(V_B, dps=dps)  # descending in nu
        g_ref = g.copy()
        for k in np.nonzero(floor_mask)[0]:
            mk = max(float(margins_mp[k]), 0.5e-14)
            g_ref[k] = np.log((1.0 + mk) / mk)
        contrib = g_ref * weights
        floor_part = float(np.sum(contrib[floor_mask]))
        dS = float(np.sum(contrib))
    info = {
        "nu": nu,
        "contributions": contrib,
        "top_mode_share": float(contrib[0] / dS) if dS != 0.0 else 0.0,
        "floor_mask": floor_mask,
        "n_floor": int(floor_mask.sum()),
        "floor_share": float(floor_part / dS) if dS != 0.0 else 0.0,
        "escalated": escalated,
        "margins_mp": margins_mp,
    }
    return dS, info


def equilibrium_residual(K, sizes=(4, 8, 16), ratios=(4, 8, 16), parity="odd",
                         split="bond", eig=None, return_details=False,
                         floor_tol=1e-10, escalate_above=1e-6, dps=40,
                         balls=None, zero_mode="parity", beta_at="site", shift=0.0,
                         wall=None, mix=0.5):
    """The entanglement-equilibrium residual of a lattice dynamics K,

        r(K) = max_{B, v} |delta S_v(B) - delta<K_loc>_v(B)| / |delta S_v(B)|,

    over centred balls of the given even sizes and the long-wavelength
    squeeze variations of `squeeze_modes` (one per wavelength ratio, of the
    given reflection parity). delta S is (1/2) Tr[G_B delta V_B] with G_B the
    exact modular Hamiltonian of the ball in the vacuum of K
    (vacuum.core.entanglement_hamiltonian; the identity delta S = delta<K_B>
    is checked by finite differences in `first_law_audit`), delta<K_loc> =
    (1/2) Tr[G_loc delta V] with G_loc from `local_modular_form`.

    Extensions (papers/toy-jacobson, items "k^4 coefficient" and "second
    ball position"): `balls` = explicit list of site lists (overrides `sizes`;
    a ball off the chain centre has no reflection parity, use parity='any');
    zero_mode='project' projects the ball's top Williamson mode (the 1+1
    massless scalar's zero mode phi_bar, whose nu_1 grows like ln N) out of
    the VARIATION -- both quadratures, in the ball's Williamson frame -- so
    that delta S and delta<K_loc> are the same linear functional of the same
    delta V, and no parity argument is needed. This is the symmetric
    alternative to excluding the zero mode by reflection parity;
    `beta_at`, `shift`, `wall`, `mix` select the discretisation of the local
    form (local_modular_form).

    Returns r, or with return_details=True a dict: 'residual' (max),
    'rms', 'per_size' {L: max r}, 'rows' (one record per (ball, variation):
    L, ratio, index, wavelength, parity, dS, dK, r, rho = (dS - dK)/dS signed,
    top_mode_share, floor_share, n_floor, escalated, nu_min_margin, ball_start),
    'eig' (reusable).
    """
    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    if eig is None:
        eig = np.linalg.eigh(K)
    lam, U = eig
    if lam[0] <= 0.0:
        raise ValueError(f"K must be positive definite, min eigenvalue {lam[0]:.3e}")
    if zero_mode not in ("parity", "project"):
        raise ValueError("zero_mode must be 'parity' or 'project'")
    from vacuum.core import williamson as _williamson  # local: avoids shadowing the module import
    if balls is None:
        balls = [centred_ball(N, L) for L in sizes]
    else:
        balls = [[int(i) for i in sorted(b)] for b in balls]
        sizes = [len(b) for b in balls]
    rows = []
    for ball in balls:
        L = len(ball)
        sites, ball_pos = _halo(K, ball)
        V_B = _vacuum_block(eig, ball)
        G_loc = local_modular_form(K, ball, split=split, sites=sites, beta_at=beta_at,
                                   shift=shift, wall=wall, mix=mix)
        n_s = len(sites)
        bidx = np.asarray(ball_pos, dtype=int)
        if zero_mode == "project":
            S_w, _ = _williamson(V_B)
            Si_w = symplectic_inverse(S_w)
            keep = np.ones(2 * L)
            keep[0] = keep[L] = 0.0  # the ball's top Williamson mode (phi_bar), x and p
        for mode in squeeze_modes(eig, L, ratios=ratios, parity=parity):
            u, w = mode["u"], mode["omega"]
            dV_B = squeeze_variation(u[ball], w)
            dV_sites = squeeze_variation(u[sites], w)
            if zero_mode == "project":
                Wm = Si_w @ dV_B @ Si_w.T
                dV_B = S_w @ (Wm * keep[:, None] * keep[None, :]) @ S_w.T
                sel = np.concatenate([bidx, bidx + n_s])
                dV_sites[np.ix_(sel, sel)] = dV_B  # the same variation on both sides
            dS, info = _mode_contributions(V_B, dV_B, floor_tol, escalate_above, dps)
            dS_used = dS
            dK = 0.5 * float(np.trace(G_loc @ dV_sites))
            r = abs(dS_used - dK) / abs(dS_used) if dS_used != 0.0 else np.inf
            rows.append({
                "L": int(L), "ratio": mode["ratio"], "index": mode["index"],
                "wavelength": mode["wavelength"], "parity": mode["parity"],
                "dS": dS_used, "dS_full": dS, "dK": dK, "r": float(r),
                "rho": float((dS_used - dK) / dS_used) if dS_used != 0.0 else np.inf,
                "top_mode_share": info["top_mode_share"],
                "floor_share": info["floor_share"], "n_floor": info["n_floor"],
                "escalated": info["escalated"],
                "nu_min_margin": float(info["nu"].min() - 0.5),
                "ball_start": int(ball[0]), "zero_mode": zero_mode,
            })
    rs = np.array([row["r"] for row in rows])
    r_max = float(np.max(rs)) if rs.size else 0.0
    if not return_details:
        return r_max
    per_size = {int(L): float(max(row["r"] for row in rows if row["L"] == L)) for L in sizes}
    return {"residual": r_max, "rms": float(np.sqrt(np.mean(rs ** 2))) if rs.size else 0.0,
            "per_size": per_size, "rows": rows, "eig": eig}


def first_law_audit(K, size, ratio=4, eps=4e-3, parity="odd", split="bond",
                    eig=None, use_precise=True, dps=40):
    """Independent-route check of the exact first law for one squeeze
    variation: the entropy of the EXACTLY squeezed ball state
    (squeezed_ball_covariance at +-eps, +-eps/2, Richardson-combined: O(eps^4))
    against delta S = (1/2) Tr[G_B delta V_B] from the modular Hamiltonian.

    With use_precise=True the entropies go through
    vacuum.core.precision.entropy_precise (float64 -> mpmath when a symplectic
    eigenvalue is within 1e-10 of the vacuum floor -- needed e.g. for the
    Lifshitz chain, whose badly scaled ball covariance puts float64 margins
    below the floor).

    Returns dict(dS_fd, dS_modular, rel_err, dK_local, residual, eps,
    nu_min_margin, precise, mode index/wavelength).
    """
    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    if eig is None:
        eig = np.linalg.eigh(K)
    ball = centred_ball(N, size)
    sites, _ = _halo(K, ball)
    mode = squeeze_modes(eig, size, ratios=(ratio,), parity=parity)[0]
    u, w = mode["u"], mode["omega"]
    V_B = _vacuum_block(eig, ball)
    G_B = entanglement_hamiltonian(V_B)
    dV_B = squeeze_variation(u[ball], w)
    dS_mod = modular_energy(G_B, dV_B)
    G_loc = local_modular_form(K, ball, split=split, sites=sites)
    dK = 0.5 * float(np.trace(G_loc @ squeeze_variation(u[sites], w)))
    S_of = entropy_precise if use_precise else entropy
    uB = u[ball]

    def central(e):
        Sp = S_of(squeezed_ball_covariance(V_B, uB, w, +e))
        Sm = S_of(squeezed_ball_covariance(V_B, uB, w, -e))
        return (Sp - Sm) / (2.0 * e)

    D_c, D_f = central(eps), central(0.5 * eps)
    dS_fd = (4.0 * D_f - D_c) / 3.0
    margin = float(symplectic_eigenvalues(V_B).min() - 0.5)
    return {
        "dS_fd": float(dS_fd), "dS_modular": float(dS_mod),
        "rel_err": float(abs(dS_fd - dS_mod) / abs(dS_mod)) if dS_mod != 0.0 else np.inf,
        "dK_local": dK,
        "residual": float(abs(dS_mod - dK) / abs(dS_mod)) if dS_mod != 0.0 else np.inf,
        "eps": float(eps), "nu_min_margin": margin, "precise": bool(use_precise),
        "index": mode["index"], "wavelength": mode["wavelength"], "parity": mode["parity"],
    }


def eh_profiles(K, size, eig=None):
    """Cross-check of the ball's exact modular Hamiltonian against the CHM/BW
    profile in the continuum-limit sense of Di Giulio-Tonni (2020) Eq. (20)
    (all diagonals combined): returns dict with 'beta' (2 pi beta_i),
    'pp_rowsum_over_beta' (sum_j G^{pp}_ij / 2 pi beta_i, the p^2 weight),
    'xx_resummed_over_beta' (-(1/2) sum_j (j-i)^2 G^{xx}_ij / 2 pi beta_i, the
    (d phi)^2 weight), 'xx_rowsum' (the zero-mode phi_bar^2 coefficient per
    row), 'locality' {bandwidth: vacuum.modular.locality_score(G_B)} for
    bandwidths 1, 2, 4, and 'nu'."""
    from vacuum.modular import locality_score  # read-only cross-check (no import cycle)

    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    if eig is None:
        eig = np.linalg.eigh(K)
    ball = centred_ball(N, size)
    L = len(ball)
    V_B = _vacuum_block(eig, ball)
    G = entanglement_hamiltonian(V_B)
    beta = 2.0 * np.pi * bw_weights(ball)
    Gxx, Gpp = G[:L, :L], G[L:, L:]
    idx = np.arange(L)
    r2 = (idx[:, None] - idx[None, :]).astype(float) ** 2
    return {
        "beta": beta,
        "pp_rowsum_over_beta": Gpp.sum(axis=1) / beta,
        "xx_resummed_over_beta": (-0.5 * (r2 * Gxx).sum(axis=1)) / beta,
        "xx_rowsum": Gxx.sum(axis=1),
        "locality": {b: float(locality_score(G, bandwidth=b, block=2)) for b in (1, 2, 4)},
        "nu": symplectic_eigenvalues(V_B),
    }


# ---------------------------------------------------------------------------
# constrained dynamics
# ---------------------------------------------------------------------------


def fit_dynamics(K_of_theta, theta0, sizes=(8, 16), ratios=(4, 8, 16), parity="odd",
                 split="bond", objective="max", method="Nelder-Mead", maxfev=250,
                 restarts=1, bounds=None, xatol=1e-4, fatol=1e-7, penalty=10.0,
                 escalate_above=1e-3, residual="chm", free_kwargs=None):
    """Minimize the entanglement-equilibrium residual over a parametrized
    family of dynamics theta -> K(theta) from the generic start theta0.

    residual='chm' (default) minimises r(K) with the CHM weight assumed;
    residual='free' minimises the free-weight residual r_free(K) of
    `free_weight_residual` (the headline family, both discretisations,
    nonnegative weights, re-optimised at every theta -- the weight is a
    free variable of the fit, so a rescaling of K, i.e. the velocity, is
    exactly absorbed); residual='universal' minimises
    `universal_weight_residual` (one profile R^z f(x/R) shared by every
    ball size). `free_kwargs` are passed to those functions (e.g.
    families, headline, exponents, ratio_min). With residual != 'chm' the
    `ratios`, `parity`, `split`, `objective` and `escalate_above` arguments
    are unused and info['details'] is the free-weight record at theta_star.

    objective='max' is r(K) itself (max over balls and variations), 'rms'
    the root mean square over the same set (smoother; the max is reported
    in info['details'] either way). The optimizer is restarted `restarts`
    times from its own endpoint (Nelder-Mead stalls on the ridges of a
    max-function). `bounds` (list of (lo, hi) per parameter, or None) are
    passed to scipy; they are where the family's premises live -- e.g. a
    mass bound m <= 1 encodes Jacobson's assumption that the UV
    entanglement is that of the gapless vacuum, excluding the secondary
    dip of the residual at m ~ 4 lattice units where the ball is nearly a
    product state (papers/toy-jacobson). Parameter sets for which K is not
    positive definite (or K_of_theta raises ValueError) receive `penalty`.
    mpmath escalation of floor-mode margins is only triggered above
    `escalate_above` of |delta S| (irrelevant at the 1e-2 residuals being
    minimized, and 6x the cost when it fires).

    Returns (theta_star, info) with info = {'residual', 'nfev', 'success',
    'message', 'history': [(theta, f)], 'details': equilibrium_residual
    details at theta_star}.
    """
    history = []
    fk = dict(free_kwargs or {})
    if residual not in ("chm", "free", "universal"):
        raise ValueError("residual must be 'chm', 'free' or 'universal'")

    def evaluate(K, eig):
        """(value, details) of the chosen residual at K."""
        if residual == "chm":
            det = equilibrium_residual(K, sizes=sizes, ratios=ratios, parity=parity,
                                       split=split, eig=eig, return_details=True,
                                       escalate_above=escalate_above)
            return (det["residual"] if objective == "max" else det["rms"]), det
        if residual == "free":
            kw = {"families": (), "signed": False, **fk}
            det = free_weight_residual(K, sizes=sizes, eig=eig, **kw)
            return det["residual"], det
        ukw = {"families": ("cheb3",), "exponents": (1.0,), "signed": False, **fk}
        design_kw = {k: ukw.pop(k) for k in ("ratio_min", "pairs", "parity", "floor_tol", "dps",
                                             "escalate_above", "discretisations", "nonneg") if k in ukw}
        base = free_weight_residual(K, sizes=sizes, eig=eig, families=(), headline="cheb2",
                                    signed=False, **design_kw)
        det = universal_weight_residual(K, sizes=sizes, design=base["design"],
                                        **{k: v for k, v in design_kw.items() if k in ("discretisations", "nonneg")},
                                        **ukw)
        return det["residual"], det

    def f(theta):
        try:
            K = np.asarray(K_of_theta(theta), dtype=float)
            lam, U = np.linalg.eigh(K)
            if lam[0] <= 1e-12:
                raise ValueError("not positive definite")
            val, _ = evaluate(K, (lam, U))
        except (ValueError, np.linalg.LinAlgError):
            val = float(penalty)
        history.append((np.array(theta, dtype=float), float(val)))
        return float(val)

    if method == "Nelder-Mead":
        options = {"maxfev": int(maxfev), "xatol": xatol, "fatol": fatol, "adaptive": True}
    else:
        options = {"maxfev": int(maxfev), "xtol": xatol, "ftol": fatol}
    x = np.asarray(theta0, dtype=float)
    nfev = 0
    for _ in range(int(restarts) + 1):
        res = minimize(f, x, method=method, bounds=bounds, options=options)
        x = np.asarray(res.x, dtype=float)
        nfev += int(res.nfev)
    theta_star = x
    K_star = np.asarray(K_of_theta(theta_star), dtype=float)
    _, details = evaluate(K_star, np.linalg.eigh(K_star))
    info = {"residual": float(res.fun), "nfev": nfev, "success": bool(res.success),
            "message": str(res.message), "history": history, "details": details}
    return theta_star, info


def residual_scan(K_of_theta, theta_star, axis, deltas, **residual_kwargs):
    """One-dimensional scan of the residual around theta_star along parameter
    `axis`: returns (deltas, r) -- the landscape whose width is the family's
    resolution on that parameter."""
    out = []
    for d in deltas:
        th = np.array(theta_star, dtype=float)
        th[axis] += d
        try:
            out.append(equilibrium_residual(np.asarray(K_of_theta(th), dtype=float), **residual_kwargs))
        except (ValueError, np.linalg.LinAlgError):
            out.append(np.nan)
    return np.asarray(deltas, dtype=float), np.array(out)


# ---------------------------------------------------------------------------
# separating the ansatz floor from the k^4 dispersion term
# ---------------------------------------------------------------------------


def k4_sensitivity(N, sizes=(8, 16, 32), ratios=(4, 8, 16),
                   c2_list=(-0.5, 0.0, 0.5), c3_list=(-0.1, 0.0, 0.1),
                   **residual_kwargs):
    """Signed residual rho = (dS - dK)/dS across the coupling family at v = 1,
    K = c_1 L_1 + c_2 L_2 + c_3 L_3 with c_1 + 4 c_2 + 9 c_3 = 1, whose bulk
    dispersion is

        omega(k)/k - 1 = -A_4 k^2/24 + O(k^4),   A_4 = sum_n c_n n^4
                                                     = 1 + 12 c_2 + 72 c_3

    (nearest-neighbour chain A_4 = 1; the fourth-order-accurate Laplacian
    c_2 = -1/12 has A_4 = 0). With c_3 = 0 the family is one-dimensional and
    A_4 is an AFFINE function of c_2, so every response linear in the couplings
    -- the k^4 dispersion and any range-dependent discretisation error of the
    local ansatz alike -- is degenerate with it and no k^4 coefficient can be
    extracted. Sweeping c_3 as well breaks the degeneracy: a response that is a
    function of the dispersion alone must satisfy

        d rho / d c_3 = 6 d rho / d c_2

    exactly. This routine fits rho = rho_0 + l_2 c_2 + l_3 c_3 at each (L,
    ratio), reports l_3/l_2 against that 6, and quotes the sensitivity as

        C = coefficient of (omega(k_L)/k_L - 1) at the ball scale k_L = pi/L,

    both from the (c_2, c_3) fit and from a one-parameter fit in A_4 alone
    (whose rms residual, compared with the two-parameter one, is the test of
    whether the dispersion explains the response). Returns
    {'c2', 'c3', 'A4', 'rows': {(L, ratio): {...}}}.
    """
    c2v, c3v, rho = [], [], {}
    for c2 in c2_list:
        for c3 in c3_list:
            c1 = 1.0 - 4.0 * c2 - 9.0 * c3
            try:
                K = coupling_chain_K(N, [c1, c2, c3])
                if np.linalg.eigvalsh(K)[0] <= 1e-9:
                    raise ValueError("not positive definite")
            except (ValueError, np.linalg.LinAlgError):
                continue
            c2v.append(c2)
            c3v.append(c3)
            det = equilibrium_residual(K, sizes=sizes, ratios=ratios, return_details=True,
                                       **residual_kwargs)
            for row in det["rows"]:
                rho.setdefault((row["L"], row["ratio"]), []).append(row["rho"])
    c2v, c3v = np.array(c2v), np.array(c3v)
    A4 = 1.0 + 12.0 * c2v + 72.0 * c3v
    out = {"c2": c2v, "c3": c3v, "A4": A4, "rows": {}}
    for key, vals in rho.items():
        y = np.array(vals)
        kL = np.pi / key[0]
        X = np.vstack([np.ones_like(c2v), c2v, c3v]).T
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        XA = np.vstack([np.ones_like(A4), A4]).T
        cA = np.linalg.lstsq(XA, y, rcond=None)[0]
        out["rows"][key] = {
            "rho": y, "floor": float(c[0]), "l2": float(c[1]), "l3": float(c[2]),
            "l3_over_l2": float(c[2] / c[1]) if c[1] != 0.0 else np.inf,
            "rms_3par": float(np.sqrt(np.mean((X @ c - y) ** 2))),
            "rms_A4_only": float(np.sqrt(np.mean((XA @ cA - y) ** 2))),
            "C": float(-cA[1] * 24.0 / kL ** 2),
            "C_from_c2": float(-c[1] * 24.0 / (12.0 * kL ** 2)),
            "C_from_c3": float(-c[2] * 24.0 / (72.0 * kL ** 2)),
            "floor_at_A4_0": float(c[0] - c[1] / 12.0),
        }
    return out


# ---------------------------------------------------------------------------
# the free-weight residual: is there ANY local modular Hamiltonian?
# ---------------------------------------------------------------------------
#
# Everything above pairs delta S with ONE local form, K_loc = 2 pi sum beta_i h_i
# with beta the CHM/BW parabola -- a weight derived from conformal symmetry.
# That the first law then holds for the wave chain and fails for Lifshitz is a
# consistency check on CHM, not a derivation of Lorentz invariance from
# entanglement. The functions below free the weight: K_loc = sum_k w_k O_k over
# the lattice's local energy pieces O_k (the p_i^2/2 + M_i x_i^2/2 site terms and
# the (c_ij/2)(x_i - x_j)^2 bond terms of the PSD bond split; sum_k O_k = H),
# with w drawn from a family -- smooth even Chebyshev profiles sampled at the
# sites or at the bond midpoints (the two discretisations of the CHM ansatz), or
# one free weight per site (plus the boundary bonds' own weight) -- and minimise
# the max relative first-law residual over the family by linear programming.
# r_free(K) is the minimum; the minimising weight is then a RESULT: for the
# wave chain it should be the CHM parabola, found rather than assumed.
#
# Three facts shape the protocol. (i) Jacobson's regime, ball << wavelength, is
# LOW-RANK: the single-mode squeezes of wavelength >= 4L impose only ~3-4
# independent conditions on a local weight at 1e-4 relative precision, so the
# first-order two-mode squeezes (`pair_variation`: products u_a u_b of two odd
# long-wavelength modes, also smooth on the lattice scale, also exactly
# first-law-obeying) are included, and the numerical rank of the variation
# family is reported next to the parameter count -- a weight with as many
# parameters as the family has independent conditions fits ANY dynamics and
# says nothing; the per-site weight is in that regime at every L, which is why
# `universal_weight_residual` shares one profile across ball sizes. (ii)
# beta >= 0 is not optional: K_loc = sum w_k O_k with w of either sign is
# unbounded below, so e^{-K_loc} is not a state; `nonneg=True` imposes w >= 0
# on every piece and the signed minimum is kept as a diagnostic. (iii) The
# site-symmetrised split h_i = p_i^2/2 + x_i (K x)_i / 2 has IDENTICALLY zero
# response to every single-mode squeeze (delta<p_i^2> = -omega u_i^2 against
# delta<x_i (K x)_i> = +omega u_i^2, a lattice virial identity), so the bond
# split is the only usable local density here and "re-splitting" means the
# site-centred / bond-centred sampling of the weight plus the straddling
# bonds' own weight.


def local_pieces(K, ball, tol=1e-12):
    """The local energy pieces of H = (1/2)(p.p + x^T K x) attached to `ball`,
    in the PSD bond split: for each site i in the ball O_i = p_i^2/2 +
    M_i x_i^2/2 (M_i = sum_j K_ij), and for each bond (i, j) with K_ij != 0
    and at least one end in the ball O_ij = (c_ij/2)(x_i - x_j)^2, c_ij =
    -K_ij, counted once. The ball's sum_i h_i (local_energy_forms,
    split='bond') is the site pieces + the interior bonds + HALF of each
    straddling bond. Entries |K_ij| < tol * max|K| are treated as zero (a K
    from an eigendecomposition, e.g. dispersion_power_K, carries roundoff
    that would otherwise be free parameters).

    Returns dict: 'sites' (ball plus every site a piece touches, sorted),
    'kind' (0 site / 1 bond), 'i', 'j' (j = i for site pieces), 'coeff'
    (M_i or c_ij), 'x' (the site, or the bond midpoint), 'straddling'
    (bond with one end outside), 'ball'.
    """
    K = np.asarray(K, dtype=float)
    scale = float(np.abs(K).max())
    Kc = np.where(np.abs(K) < tol * scale, 0.0, K)
    ball = [int(i) for i in sorted(ball)]
    inb = set(ball)
    M = Kc.sum(axis=1)
    kind, ii, jj, coeff, xs, strad = [], [], [], [], [], []
    for i in ball:
        kind.append(0); ii.append(i); jj.append(i); coeff.append(M[i]); xs.append(float(i)); strad.append(False)
    for i in ball:
        for j in np.nonzero(Kc[i])[0]:
            j = int(j)
            if j == i or (j in inb and j < i):
                continue
            kind.append(1); ii.append(i); jj.append(j); coeff.append(-Kc[i, j])
            xs.append(0.5 * (i + j)); strad.append(j not in inb)
    sites = sorted(inb.union(jj))
    return {"sites": sites, "kind": np.array(kind), "i": np.array(ii), "j": np.array(jj),
            "coeff": np.array(coeff, dtype=float), "x": np.array(xs), "straddling": np.array(strad),
            "ball": ball}


def pair_variation(u_a, omega_a, u_b, omega_b):
    """First-order covariance change of the two-mode squeeze generated by
    (1/2) R^T M R with M = ((0, A), (A, 0)), A = (u_a u_b^T + u_b u_a^T)/2:
    delta V = X V + V X^T with X = Omega M, for the vacuum of two exact
    normal modes (V_xx u = u/(2 omega), V_pp u = omega u/2):

        delta V_xx =  A (omega_a + omega_b)/(2 omega_a omega_b),
        delta V_pp = -A (omega_a + omega_b)/2.

    For a = b this is squeeze_variation (A = u u^T). Restricted to whatever
    sites u_a, u_b are given on. The pairing (1/2) Tr[G_B delta V_B] obeys the
    exact first law like any first-order Gaussian variation (checked by finite
    differences of the exactly transformed state in tests/test_toy_jacobson.py).
    """
    ua = np.asarray(u_a, dtype=float)
    ub = np.asarray(u_b, dtype=float)
    n = ua.size
    A = 0.5 * (np.outer(ua, ub) + np.outer(ub, ua))
    dV = np.zeros((2 * n, 2 * n))
    dV[:n, :n] = A * (omega_a + omega_b) / (2.0 * omega_a * omega_b)
    dV[n:, n:] = -A * (omega_a + omega_b) / 2.0
    return dV


def smooth_variations(eig, L, ratio_min=4.0, pairs=True, parity="odd", max_modes=None):
    """The long-wavelength variation family of the free-weight residual: every
    normal mode of K with wavelength >= ratio_min * L (Dirichlet count
    2(N+1)/(a+1)) and the requested reflection parity, squeezed singly and --
    with pairs=True -- in every pair (a <= b), the first-order two-mode
    squeeze of `pair_variation`. Odd x odd pairs are, like the odd singles,
    exactly blind to the ball's reflection-even Williamson modes (the zero
    mode): W_kk = 2 (row_k . u_a)(row_k . u_b) vanishes for every even row.

    Returns dict: 'modes' (indices a, ascending), 'omega', 'wavelength',
    'pairs' (list of (a, b)).
    """
    lam, U = eig
    N = U.shape[0]
    a_max = int(np.floor(2.0 * (N + 1) / (float(ratio_min) * L))) - 1
    want = {"odd": -1.0, "even": +1.0, "any": None}[parity]
    modes = [a for a in range(0, min(a_max, N - 1) + 1)
             if want is None or np.sign(_parity(U[:, a])) == want]
    if max_modes is not None:
        modes = modes[: int(max_modes)]
    if not modes:
        raise ValueError(f"no {parity} mode with wavelength >= {ratio_min} * {L} on N = {N}")
    prs = [(a, b) for ia, a in enumerate(modes) for b in (modes[ia:] if pairs else [a])]
    return {"modes": modes, "omega": np.sqrt(np.maximum(lam[modes], 0.0)),
            "wavelength": np.array([2.0 * (N + 1) / (a + 1) for a in modes]),
            "pairs": prs}


def _free_design(ball, eig, var, pieces, floor_tol=1e-10, escalate_above=1e-6, dps=40):
    """delta S_v (exact modular Hamiltonian) and the piece responses
    Q[v, k] = delta<O_k>_v for every variation v of `var`, without forming
    delta V: with c_x = (w_a + w_b)/(2 w_a w_b) and c_p = (w_a + w_b)/2,
    delta<p_i^2> = -c_p u_a,i u_b,i, delta<x_i^2> = c_x u_a,i u_b,i,
    delta<(x_i - x_j)^2> = c_x (u_a,i - u_a,j)(u_b,i - u_b,j), and
    delta S = sum_k g(nu_k) (W_kk + W_{L+k,L+k})/2 with W = S^{-1} dV_B S^{-T}
    in the ball's Williamson frame. As in `_mode_contributions`, the modular
    temperatures of floor modes (nu - 1/2 < floor_tol) are re-evaluated in
    mpmath (once per ball) when they carry more than `escalate_above` of
    |delta S| for any variation."""
    lam, U = eig
    L = len(ball)
    V_B = _vacuum_block(eig, ball)
    S_w, nu = williamson(V_B)
    Si = symplectic_inverse(S_w)
    g = _g_of_nu(nu)
    floor = (nu - 0.5) < floor_tol
    modes = var["modes"]
    UB = U[np.asarray(ball)][:, modes]
    Ax = Si[:, :L] @ UB  # (2L x n_modes): x-quadrature overlaps in the Williamson frame
    Ap = Si[:, L:] @ UB
    om = {a: float(w) for a, w in zip(modes, var["omega"])}
    col = {a: k for k, a in enumerate(modes)}
    cxs = np.array([(om[a] + om[b]) / (2.0 * om[a] * om[b]) for a, b in var["pairs"]])
    cps = np.array([0.5 * (om[a] + om[b]) for a, b in var["pairs"]])
    ia = np.array([col[a] for a, b in var["pairs"]])
    ib = np.array([col[b] for a, b in var["pairs"]])
    Wd = cxs[:, None] * Ax[:, ia].T * Ax[:, ib].T - cps[:, None] * Ap[:, ia].T * Ap[:, ib].T  # (n_v x 2L)
    weights = 0.5 * (Wd[:, :L] + Wd[:, L:])  # per-mode weights (n_v x L)

    def total(gg):
        return weights @ gg

    dS = total(g)
    escalated = False
    if floor.any():
        share = np.abs(weights[:, floor] @ g[floor]) / np.maximum(np.abs(dS), 1e-300)
        if share.max() > escalate_above:
            escalated = True
            margins = symplectic_margins_mp(V_B, dps=dps)
            for k in np.nonzero(floor)[0]:
                mk = max(float(margins[k]), 0.5e-14)
                g[k] = np.log((1.0 + mk) / mk)
            dS = total(g)
    ma = np.array([a for a, b in var["pairs"]])  # mode indices (columns of U)
    mb = np.array([b for a, b in var["pairs"]])
    ui = U[pieces["i"], :]
    uj = U[pieces["j"], :]
    site = pieces["kind"] == 0
    prod = ui[:, ma] * ui[:, mb]  # (n_pieces x n_v)
    dxx = (ui[:, ma] - uj[:, ma]) * (ui[:, mb] - uj[:, mb])
    Q = np.where(site[:, None], 0.5 * (-cps[None, :] * prod + pieces["coeff"][:, None] * cxs[None, :] * prod),
                 0.5 * pieces["coeff"][:, None] * cxs[None, :] * dxx).T
    info = {"nu": nu, "n_floor": int(floor.sum()), "nu_min_margin": float(nu.min() - 0.5),
            "escalated": escalated}
    return dS, Q, info


def _cheb_even(xi, n):
    """T_0, T_2, ..., T_{2(n-1)} at xi (clipped to [-1, 1]), shape (len(xi), n)."""
    t = np.arccos(np.clip(np.atleast_1d(np.asarray(xi, dtype=float)), -1.0, 1.0))
    return np.array([np.cos(2 * m * t) for m in range(int(n))]).T


def _hat_even(xi, n):
    """n piecewise-linear hat functions of |xi| on the nodes k/(n - 1),
    clipped to |xi| <= 1; shape (len(xi), n)."""
    u = np.clip(np.abs(np.atleast_1d(np.asarray(xi, dtype=float))), 0.0, 1.0)
    nodes = np.arange(n) / (n - 1)
    h = 1.0 / (n - 1)
    return np.maximum(0.0, 1.0 - np.abs(u[:, None] - nodes[None, :]) / h)


def weight_family(family, pieces, discretisation="bond"):
    """Parametrisation w = P theta of the piece weights of K_loc = sum_k w_k O_k.

    family = 'cheb<n>' : w = f(x) with f = sum_{m<n} theta_m T_{2m}(xi),
                          xi = (x - x_c)/R, R = L/2 (even in xi: the ball is
                          reflection-symmetric and odd variations cannot see
                          an odd part). Site pieces carry f(x_i); bond pieces
                          carry (f(x_i) + f(x_j))/2 with f = 0 outside the ball
                          (discretisation='site', the site-centred ansatz) or
                          f at the bond midpoint clipped to the ball
                          (discretisation='bond', the midpoint rule). 'cheb2'
                          (constant + parabola) is the smallest family
                          containing CHM, w = 2 pi (R^2 - x^2)/(2R) =
                          (pi R/2)(T_0 - T_2).
    family = 'hat<n>'  : the same with f = sum_k theta_k h_k(|xi|), h_k the n
                          piecewise-linear hat functions on the nodes
                          |xi| = k/(n - 1), k = 0..n-1 (a basis of a different
                          kind, used to check that a mode count is not a
                          Chebyshev artefact; n >= 2).
    family = 'mono<n>' : f = sum_{m<n} theta_m xi^{2m}, the even monomials
                          (ill-conditioned beyond n ~ 10; a third check).
    family = 'site'    : one free weight per site (mirror pairs tied), bond
                          pieces as above; discretisation='bond' adds ONE
                          parameter: the weight of the straddling bonds (the
                          boundary value, which the CHM parabola sets to 0).
    family = 'chm'     : the fixed CHM weights (P is one column, theta = 1).

    Returns (P, names) with P (n_pieces x n_params).
    """
    ball = pieces["ball"]
    L = len(ball)
    xc = 0.5 * (ball[0] + ball[-1])
    R = 0.5 * L
    x = pieces["x"]
    kind = pieces["kind"]
    i_, j_ = pieces["i"], pieces["j"]
    inb = np.array([k in set(ball) for k in j_])
    if discretisation not in ("site", "bond"):
        raise ValueError("discretisation must be 'site' or 'bond'")
    if family == "chm":
        bw_site = lambda xx: np.maximum((R * R - (np.asarray(xx, dtype=float) - xc) ** 2) / (2.0 * R), 0.0)  # noqa: E731
        if discretisation == "site":
            w = np.where(kind == 0, 2.0 * np.pi * bw_site(x),
                         np.pi * (bw_site(i_) + np.where(inb, bw_site(j_), 0.0)))
        else:
            w = 2.0 * np.pi * bw_site(x)
        return w[:, None], ["chm"]
    smooth = None
    if family.startswith("cheb"):
        n = int(family[4:])
        smooth = (lambda xi: _cheb_even(xi, n)), [f"T{2 * m}" for m in range(n)]
    elif family.startswith("hat"):
        n = int(family[3:])
        if n < 2:
            raise ValueError("hat<n> needs n >= 2")
        smooth = (lambda xi: _hat_even(xi, n)), [f"h{k}" for k in range(n)]
    elif family.startswith("mono"):
        n = int(family[4:])
        smooth = (lambda xi: np.clip(np.abs(np.atleast_1d(np.asarray(xi, dtype=float))), 0.0, 1.0)[:, None]
                  ** (2 * np.arange(n))[None, :]), [f"xi^{2 * m}" for m in range(n)]
    if smooth is not None:
        g, names = smooth
        f = lambda xx: g((np.asarray(xx, dtype=float) - xc) / R)  # noqa: E731
        if discretisation == "site":
            P = np.where((kind == 0)[:, None], f(x),
                         0.5 * (f(i_) + np.where(inb[:, None], f(j_), 0.0)))
        else:
            P = f(x)
        return P, names
    if family == "site":
        half = L // 2
        col = lambda s: min(int(s) - ball[0], L - 1 - (int(s) - ball[0]))  # noqa: E731
        extra = 1 if discretisation == "bond" else 0
        P = np.zeros((kind.size, half + extra))
        for k in range(kind.size):
            if kind[k] == 0:
                P[k, col(i_[k])] = 1.0
            elif inb[k]:
                P[k, col(i_[k])] += 0.5
                P[k, col(j_[k])] += 0.5
            elif extra:
                P[k, half] = 1.0
            else:
                P[k, col(i_[k])] = 0.5
        names = [f"beta_{k}" for k in range(half)] + (["beta_boundary"] if extra else [])
        return P, names
    raise ValueError(f"unknown weight family {family!r}")


def _minimax_weights(D, dS, P=None, nonneg=True):
    """min_theta max_v |dS_v - (D theta)_v| / |dS_v|, subject to P theta >= 0
    when nonneg (P = the piece-weight map, D = Q P), as a linear programme.
    Returns (theta, r)."""
    from scipy.optimize import linprog

    n_v, p = D.shape
    s = 1.0 / np.abs(dS)
    Ds = D * s[:, None]
    rows = [np.hstack([Ds, -np.ones((n_v, 1))]), np.hstack([-Ds, -np.ones((n_v, 1))])]
    rhs = [dS * s, -dS * s]
    if nonneg:
        Pn = D if P is None else P
        keep = np.abs(Pn).sum(axis=1) > 0
        rows.append(np.hstack([-Pn[keep], np.zeros((int(keep.sum()), 1))]))
        rhs.append(np.zeros(int(keep.sum())))
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    c = np.zeros(p + 1)
    c[-1] = 1.0
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * p + [(0.0, None)], method="highs")
    if not res.success:
        return np.full(p, np.nan), np.inf
    theta = res.x[:p]
    return theta, float(np.max(np.abs(dS - D @ theta) * s))


def _rank_profile(A, tols):
    sv = np.linalg.svd(A, compute_uv=False)
    sv = sv / sv[0]
    return {float(t): int((sv > t).sum()) for t in tols}


def free_weight_residual(K, sizes=(8, 16), ratio_min=4.0, pairs=True, parity="odd",
                         families=("cheb1", "cheb2", "cheb3", "cheb4", "site"),
                         discretisations=("site", "bond"), nonneg=True, signed=True,
                         headline="cheb3", eig=None, rank_tols=(1e-4, 1e-6, 1e-8),
                         floor_tol=1e-10, escalate_above=1e-6, dps=40):
    """The free-weight entanglement-equilibrium residual

        r_free(K; L) = min_{disc} min_{w in family, w >= 0}
                       max_v |delta S_v - sum_k w_k delta<O_k>_v| / |delta S_v|

    over the long-wavelength variation family of `smooth_variations` (odd
    single-mode and, with pairs=True, two-mode squeezes of wavelength >=
    ratio_min * L) and the local energy pieces of `local_pieces`, with the
    weight w drawn from `weight_family`. The CHM residual r on the SAME
    variation family is computed alongside for both discretisations, so
    r_free <= r holds by construction for every family containing the CHM
    weight (all but 'cheb1', a constant). The headline r_free(K) uses the
    `headline` family (default 'cheb3': constant + parabola + quartic, three
    parameters -- below the rank of the variation family, which is what
    gives the number teeth) and the better discretisation; the others are
    reported. Every quantity below is per ball size.

    Returns dict with, per size L (key int L):
      'r_chm'      {disc: r}            the CHM weight on this variation family
      'r_free'     {(family, disc): r}  nonnegative minimum
      'r_signed'   {(family, disc): r}  unconstrained minimum (signed=True)
      'theta', 'weights' (piece weights w), 'beta_sites' (w on the site pieces
                   / 2 pi, in ball order), 'overlap' (cos(w, w_chm)), 'scale'
                   (<w, w_chm>/<w_chm, w_chm>; 1 = the CHM normalisation),
                   'n_params', 'rank_family' {tol: rank of Q P / |dS|}
      'rank'       {tol: numerical rank of the scaled piece design Q / |dS|,
                    the number of independent conditions the family imposes}
      'n_variations', 'n_modes', 'n_pieces', 'n_floor', 'wavelength_min'
      'best'       the (family, disc) with the smallest r_free among the
                   families of at most `headline`'s parameter count
    and 'per_size' {L: r_free headline}, 'residual' (max over L), 'headline',
    'design' {L: (dS, Q, pieces)} (reused by universal_weight_residual), 'eig'.
    """
    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    if eig is None:
        eig = np.linalg.eigh(K)
    lam, U = eig
    if lam[0] <= 0.0:
        raise ValueError(f"K must be positive definite, min eigenvalue {lam[0]:.3e}")
    fams = list(families)
    if headline not in fams:
        fams.append(headline)
    out = {"eig": eig, "per_size": {}, "headline": headline, "design": {}}
    for L in sizes:
        ball = centred_ball(N, L)
        pieces = local_pieces(K, ball)
        var = smooth_variations(eig, L, ratio_min=ratio_min, pairs=pairs, parity=parity)
        dS, Q, info = _free_design(ball, eig, var, pieces, floor_tol=floor_tol,
                                   escalate_above=escalate_above, dps=dps)
        s = 1.0 / np.abs(dS)
        rec = {"r_chm": {}, "r_free": {}, "r_signed": {}, "theta": {}, "weights": {}, "beta_sites": {},
               "overlap": {}, "scale": {}, "n_params": {}, "rank_family": {},
               "rank": _rank_profile(Q * s[:, None], rank_tols),
               "n_variations": len(var["pairs"]), "n_modes": len(var["modes"]),
               "n_pieces": int(pieces["kind"].size), "n_floor": info["n_floor"],
               "escalated": info["escalated"], "nu_min_margin": info["nu_min_margin"], "dS": dS,
               "wavelength_min": float(var["wavelength"].min())}
        site_mask = pieces["kind"] == 0
        w_chm = {}
        for disc in discretisations:
            w_chm[disc] = weight_family("chm", pieces, disc)[0][:, 0]
            rec["r_chm"][disc] = float(np.max(np.abs(dS - Q @ w_chm[disc]) * s))
        for fam in fams:
            for disc in discretisations:
                P, _ = weight_family(fam, pieces, disc)
                key = (fam, disc)
                D = Q @ P
                rec["rank_family"][key] = _rank_profile(D * s[:, None], rank_tols)
                rec["n_params"][key] = int(P.shape[1])
                theta, r = _minimax_weights(D, dS, P, nonneg=nonneg)
                if signed:
                    rec["r_signed"][key] = _minimax_weights(D, dS, P, nonneg=False)[1]
                w = P @ theta
                rec["theta"][key] = theta
                rec["weights"][key] = w
                rec["beta_sites"][key] = w[site_mask] / (2.0 * np.pi)
                rec["r_free"][key] = r
                wc = w_chm[disc]
                nw = float(np.linalg.norm(w))
                rec["overlap"][key] = float(np.dot(w, wc) / (nw * np.linalg.norm(wc))) if nw > 0 else 0.0
                rec["scale"][key] = float(np.dot(w, wc) / np.dot(wc, wc))
        p_head = max(rec["n_params"][(headline, d)] for d in discretisations)
        cands = [k for k in rec["r_free"] if rec["n_params"][k] <= p_head and k[0] != "cheb1"]
        rec["best"] = min(cands, key=lambda k: rec["r_free"][k])
        out["per_size"][int(L)] = float(min(rec["r_free"][(headline, d)] for d in discretisations))
        out["design"][int(L)] = (dS, Q, pieces)
        out[int(L)] = rec
    out["residual"] = float(max(out["per_size"].values()))
    return out


def universal_weight_residual(K, sizes=(8, 16, 32), exponents=(1.0, 2.0), families=("cheb3", "cheb4", "cheb6", "cheb8"),
                              discretisations=("site", "bond"), nonneg=True, signed=True,
                              design=None, rank_tols=(1e-4, 1e-6, 1e-8), **kwargs):
    """One profile for every ball: w_L = R^z f(xi), xi = (x - x_c)/R, with the
    SAME Chebyshev coefficients f at every ball size in `sizes` and the
    amplitude scaling as R^z (z = 1 is the CHM scaling beta = R(1 - xi^2)/2;
    z = 2 the natural one for omega ~ k^2). Sharing the shape across balls
    is what gives a many-parameter profile more conditions than parameters:
    the ranks of the per-ball variation families add, the parameters do not.

    Minimises max over (L, v) of the relative residual by one LP per
    (family, disc, z); `design` = the 'design' entry of free_weight_residual
    (recomputed from K and kwargs if None). Returns dict keyed
    (family, disc, z): {'r', 'per_size' {L}, 'theta', 'overlap' {L}, 'scale'
    {L}, 'n_params', 'rank_family', 'r_signed'} plus 'per_ball_rank' {L: {tol:
    rank}}, 'n_conditions' (sum over L of the per-ball ranks at 1e-6), 'best'
    (the key with the smallest r) and 'residual' (that minimum)."""
    K = np.asarray(K, dtype=float)
    if design is None:
        design = free_weight_residual(K, sizes=sizes, families=(), headline="cheb2",
                                      discretisations=discretisations, signed=False, **kwargs)["design"]
    sizes = [int(L) for L in sizes]
    dS_all = np.concatenate([design[L][0] for L in sizes])
    s_all = 1.0 / np.abs(dS_all)
    out = {"sizes": sizes,
           "per_ball_rank": {L: _rank_profile(design[L][1] / np.abs(design[L][0])[:, None], rank_tols) for L in sizes}}
    out["n_conditions"] = int(sum(out["per_ball_rank"][L][1e-6] for L in sizes))
    best = None
    for fam in families:
        for disc in discretisations:
            Ps = {L: weight_family(fam, design[L][2], disc)[0] for L in sizes}
            Pc = {L: weight_family("chm", design[L][2], disc)[0][:, 0] for L in sizes}
            for z in exponents:
                blocks = [design[L][1] @ Ps[L] * (0.5 * L) ** float(z) for L in sizes]
                D = np.vstack(blocks)
                Pn = np.vstack([Ps[L] * (0.5 * L) ** float(z) for L in sizes])
                theta, r = _minimax_weights(D, dS_all, Pn, nonneg=nonneg)
                key = (fam, disc, float(z))
                rec = {"r": r, "theta": theta, "n_params": int(D.shape[1]), "per_size": {}, "overlap": {}, "scale": {},
                       "rank_family": _rank_profile(D * s_all[:, None], rank_tols)}
                if signed:
                    rec["r_signed"] = _minimax_weights(D, dS_all, Pn, nonneg=False)[1]
                for L in sizes:
                    res_L = np.abs(design[L][0] - blocks[sizes.index(L)] @ theta) / np.abs(design[L][0])
                    rec["per_size"][L] = float(np.max(res_L))
                    w = Ps[L] @ theta * (0.5 * L) ** float(z)
                    wc = Pc[L]
                    nw = float(np.linalg.norm(w))
                    rec["overlap"][L] = float(np.dot(w, wc) / (nw * np.linalg.norm(wc))) if nw > 0 else 0.0
                    rec["scale"][L] = float(np.dot(w, wc) / np.dot(wc, wc))
                out[key] = rec
                if best is None or r < out[best]["r"]:
                    best = key
    out["best"] = best
    out["residual"] = float(out[best]["r"]) if best is not None else np.inf
    return out


# ---------------------------------------------------------------------------
# legacy hooks (unchanged behaviour)
# ---------------------------------------------------------------------------


def entanglement_equilibrium_residual(V, K, balls, mode="exact",
                                      return_details=False, split="bond"):
    """max over balls of |delta S - delta <K_ball>| between the state V and
    the vacuum of the dynamics K.

    mode='exact' : delta<K> = (1/2) Tr[G_B delta V_B] with the exact modular
                   Hamiltonian -- the entanglement first law itself
                   (O(eps^2) for any dynamics; the anchor of
                   tests/test_geometry_jacobson.py).
    mode='bw'    : delta<K> = 2 pi sum_i beta_i delta<h_i> with the local
                   energy forms of `local_energy_forms(K, split)`.

    Parameters
    ----------
    V : (2N, 2N) covariance of the (perturbed) state, block ordering.
    K : (N, N) coupling matrix of the candidate dynamics.
    balls : sequence of mode-index sequences (contiguous intervals for 'bw').

    Returns the residual r (nats), or with return_details=True a dict with
    per-ball dS, dK and the residual. For the long-wavelength residual r(K)
    of the toy proper use `equilibrium_residual`.
    """
    V = np.asarray(V, dtype=float)
    K = np.asarray(K, dtype=float)
    V0 = ground_state_cov(K)
    if mode == "bw":
        forms = local_energy_forms(K, split=split)
        dV = V - V0
    dS_list, dK_list = [], []
    for ball in balls:
        ball = [int(b) for b in ball]
        VB, V0B = reduce(V, ball), reduce(V0, ball)
        dS = entropy(VB) - entropy(V0B)
        if mode == "exact":
            G = entanglement_hamiltonian(V0B)
            dK = modular_energy(G, VB - V0B)
        elif mode == "bw":
            beta = bw_weights(ball)
            dK = 2.0 * np.pi * sum(
                bt * 0.5 * float(np.trace(forms[i] @ dV))
                for bt, i in zip(beta, sorted(ball)))
        else:
            raise ValueError("mode must be 'exact' or 'bw'")
        dS_list.append(dS)
        dK_list.append(dK)
    dS_arr, dK_arr = np.array(dS_list), np.array(dK_list)
    resid = np.abs(dS_arr - dK_arr)
    r = float(np.max(resid)) if resid.size else 0.0
    if not return_details:
        return r
    return {"residual": r, "dS": dS_arr, "dK": dK_arr, "per_ball": resid,
            "mode": mode}


def constrain_dynamics(K_family, balls=None, perturb=None, eps=1e-2, mode="bw",
                       **fit_kwargs):
    """Select or fit the dynamics that best satisfies entanglement
    equilibrium with the local CHM/BW form.

    Discrete family (dict name -> K, or sequence of K): the legacy path --
    each member is scored by `entanglement_equilibrium_residual(mode=...)`
    under the state variation V = ground state of perturb(K, eps); returns
    (K_star, table) with table rows {name, residual, normalized (residual /
    max |dS|)} sorted by normalized residual.

    Continuous family (callable theta -> K): delegates to `fit_dynamics`
    (pass theta0 and any of its keyword arguments) and returns
    (theta_star, info) -- the long-wavelength residual r(K(theta)) minimized
    from the generic start theta0; residual='free' or 'universal' minimises
    the free-weight residuals instead (the weight is then a free variable
    of the fit).
    """
    if callable(K_family):
        if "theta0" not in fit_kwargs:
            raise ValueError("a continuous family needs theta0=...")
        return fit_dynamics(K_family, **fit_kwargs)
    if balls is None or perturb is None:
        raise ValueError("the discrete path needs balls and perturb")
    if isinstance(K_family, dict):
        items = list(K_family.items())
    else:
        items = list(enumerate(K_family))
    table = []
    for name, K in items:
        K = np.asarray(K, dtype=float)
        V = ground_state_cov(perturb(K, eps))
        det = entanglement_equilibrium_residual(V, K, balls, mode=mode,
                                                return_details=True)
        scale = float(np.max(np.abs(det["dS"]))) if det["dS"].size else 1.0
        table.append({"name": name, "residual": det["residual"],
                      "normalized": det["residual"] / scale if scale > 0 else np.inf})
    table.sort(key=lambda row: row["normalized"])
    best = table[0]["name"]
    K_star = dict(items)[best] if isinstance(K_family, dict) else items[best][1]
    return K_star, table
