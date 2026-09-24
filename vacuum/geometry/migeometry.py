"""Mutual-information geometry on Gaussian and fermionic vacua.

The read head of Layer 4 (PLAN.md): distances defined from mutual
information, d_ij = -ln(I_ij / I_max), the embedding of the resulting metric
in flat and hyperbolic space, its Gromov delta-hyperbolicity, and the
FAILURE MAP -- the same pipeline run on vacua that are not holographic by
construction (massive chains, disordered couplings, detuned Ising chains),
tabulating how the geometry degrades.

Why -ln I is a distance (Swingle, PRD 86, 065007 (2012), arXiv:0905.1317,
Sec. V): in the MERA/AdS picture a two-point function of an operator of
dimension Delta decays as exp(-Delta * ell) with ell the length of the
minimal curve joining the insertion points, so for two small regions
I ~ <O O>^2 ~ exp(-2 Delta ell) and -ln I is proportional to the geodesic
length. For boundary points at cutoff z = a on the upper half plane,
ds^2 = (dz^2 + dx^2)/z^2 (Swingle Sec. IV), ell = 2 arcsinh(|x - y|/2a)
~ 2 ln |x - y|: a critical chain must give d ~ ln r (hyperbolic, curved),
a gapped chain d ~ r / xi beyond the correlation length (flat, Swingle
Sec. IV: the geometry "ends" at the factorization scale).

Diagnostics (every reconstructed metric must carry its distortion numbers,
vacuum/geometry/README.md):

* stress   : Kruskal stress-1, sqrt(sum (d_emb - d)^2 / sum d^2);
* distortion : worst-case multiplicative distortion
  max(d_emb/d) / min(d_emb/d) over pairs;
* hyperbolicity : Gromov's four-point delta (Gromov 1987; Bridson-Haefliger
  III.H.1.22), delta = max over quadruples of half the difference of the
  two largest pair-sums, and delta / diameter.

Conventions: nats throughout (docs/API.md); a covariance V in block
quadrature ordering for bosons (vacuum.core), a Majorana covariance Gamma
for the Kitaev/TFIM fermions (vacuum.geometry.ising_exact), a correlation
matrix C for hopping fermions (vacuum.fermions).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from vacuum.core import ground_state_cov, harmonic_chain_K, mutual_information
from vacuum.fermions import block_entropy as _fermion_block_entropy
from vacuum.geometry.ising_exact import (
    kitaev_chain_gamma,
    majorana_mutual_information,
    tfim_correlation_length,
    tfim_mutual_information,
)

__all__ = [
    "mi_matrix",
    "mi_distance",
    "reconstruct_metric",
    "hyperbolic_embedding",
    "hyperbolicity",
    "distance_law",
    "geometry_report",
    "failure_map",
    "boson_correlation_length",
    "MODEL_FAMILIES",
]


# ---------------------------------------------------------------------------
# mutual information matrices
# ---------------------------------------------------------------------------


def mi_matrix(state, regions, kind="gaussian", g=None):
    """Pairwise mutual information (nats) between the listed regions.

    kind='gaussian' : state = bosonic covariance V (vacuum.core conventions)
    kind='majorana' : state = Majorana covariance Gamma (ising_exact)
    kind='fermion'  : state = correlation matrix C (vacuum.fermions)
    kind='tfim_inf' : state ignored; infinite TFIM/Kitaev chain at field g

    Overlapping regions get NaN (mutual information is undefined there).
    """
    regions = [list(map(int, r)) for r in regions]
    n = len(regions)
    I = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            A, B = regions[i], regions[j]
            if set(A) & set(B):
                I[i, j] = I[j, i] = np.nan
                continue
            if kind == "gaussian":
                v = mutual_information(state, A, B)
            elif kind == "majorana":
                v = majorana_mutual_information(state, A, B)
            elif kind == "fermion":
                v = (_fermion_block_entropy(state, A) + _fermion_block_entropy(state, B)
                     - _fermion_block_entropy(state, A + B))
            elif kind == "tfim_inf":
                v = tfim_mutual_information(A, B, 1.0 if g is None else g)
            else:
                raise ValueError(f"unknown kind {kind!r}")
            I[i, j] = I[j, i] = max(float(v), 0.0)
    return I


def mi_distance(I, floor=1e-12):
    """d_ij = -ln(I_ij / I_max), I_max the largest off-diagonal entry;
    entries below ``floor`` are clipped to it (d <= ln(I_max/floor)), so a
    vanished correlation shows up as a saturated, not infinite, distance.
    Returns (D, info) with info['floor_fraction'] the fraction of pairs
    that hit the floor."""
    I = np.asarray(I, dtype=float)
    n = I.shape[0]
    off = ~np.eye(n, dtype=bool)
    vals = I[off]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("need at least one finite pair")
    I_max = float(np.max(vals))
    if I_max <= floor:
        raise ValueError("all mutual informations are below the floor")
    clipped = np.maximum(I, floor)
    hit = (I < floor) & off & np.isfinite(I)
    D = -np.log(clipped / I_max)
    D[~off] = 0.0
    D = np.where(np.isfinite(I) | ~off, D, np.nan)
    return D, {"I_max": I_max, "floor": floor,
               "floor_fraction": float(np.sum(hit) / max(1, np.sum(off & np.isfinite(I))))}


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------


def _pair_mask(D):
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    m = np.isfinite(D[iu])
    return iu[0][m], iu[1][m]


def _stress_and_distortion(D, De, ii, jj):
    d = D[ii, jj]
    de = De[ii, jj]
    stress = float(np.sqrt(np.sum((de - d) ** 2) / np.sum(d**2)))
    pos = d > 0
    ratio = de[pos] / d[pos]
    distortion = float(np.max(ratio) / np.min(ratio)) if ratio.size and np.min(ratio) > 0 else np.inf
    return stress, distortion


def _classical_mds(D, dim):
    """Torgerson's classical scaling (NaNs filled by shortest paths)."""
    from scipy.sparse.csgraph import shortest_path

    n = D.shape[0]
    Dfull = D.copy()
    if np.isnan(Dfull).any():
        big = np.nanmax(Dfull) * 10
        W = np.where(np.isnan(Dfull), 0.0, Dfull)
        Dfull = shortest_path(W, directed=False, unweighted=False)
        Dfull[~np.isfinite(Dfull)] = big
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (Dfull**2) @ J
    w, U = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:dim]
    w = np.clip(w[idx], 0.0, None)
    return U[:, idx] * np.sqrt(w)


def _euclid_dist(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def reconstruct_metric(D, dim=2, method="mds", n_restarts=2, seed=0):
    """Embed a distance matrix in R^dim (method='mds': classical scaling
    refined by stress minimization) or in the hyperbolic plane H^dim
    (method='hyperbolic', see :func:`hyperbolic_embedding`).

    Returns (X, info): coordinates and info = {'stress', 'distortion',
    'method', 'dim', ...}. A picture without its distortion number is
    inadmissible (README) -- always report info alongside X.
    """
    D = np.asarray(D, dtype=float)
    if method == "hyperbolic":
        return hyperbolic_embedding(D, dim=dim, n_restarts=n_restarts, seed=seed)
    if method != "mds":
        raise ValueError("method must be 'mds' or 'hyperbolic'")
    ii, jj = _pair_mask(D)
    d = D[ii, jj]
    n = D.shape[0]
    X0 = _classical_mds(D, dim)
    rng = np.random.default_rng(seed)

    def f(xflat):
        X = xflat.reshape(n, dim)
        diff = X[ii] - X[jj]
        de = np.sqrt(np.sum(diff**2, axis=1) + 1e-300)
        res = de - d
        val = np.sum(res**2)
        coef = 2.0 * res / de
        grad = np.zeros_like(X)
        np.add.at(grad, ii, coef[:, None] * diff)
        np.add.at(grad, jj, -coef[:, None] * diff)
        return val, grad.ravel()

    best = None
    starts = [X0] + [X0 + 0.1 * np.std(X0) * rng.standard_normal(X0.shape)
                     for _ in range(n_restarts - 1)]
    for Xs in starts:
        res = minimize(f, Xs.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    X = best.x.reshape(n, dim)
    De = _euclid_dist(X)
    stress, distortion = _stress_and_distortion(D, De, ii, jj)
    return X, {"stress": stress, "distortion": distortion, "method": "mds",
               "dim": dim, "stress_classical": _stress_and_distortion(D, _euclid_dist(X0), ii, jj)[0]}


def _hyp_dist(V, R):
    """Distances on the hyperboloid model with curvature radius R; V are the
    spatial coordinates, x0 = sqrt(1 + |v|^2)."""
    s = np.sqrt(1.0 + np.sum(V**2, axis=1))
    c = s[:, None] * s[None, :] - V @ V.T
    c = np.maximum(c, 1.0)
    return R * np.arccosh(c)


def hyperbolic_embedding(D, dim=2, n_restarts=2, seed=0, R0=None):
    """Stress-minimizing embedding in H^dim (hyperboloid model) with the
    curvature radius R fitted alongside the coordinates:
    d_H(i,j) = R arccosh( sqrt(1+|v_i|^2) sqrt(1+|v_j|^2) - v_i.v_j ).
    Initialized from the flat MDS solution rescaled to the hyperboloid.
    Returns (X, info) with info['curvature_radius'] = R; R -> large means
    the data prefer a flat metric."""
    D = np.asarray(D, dtype=float)
    ii, jj = _pair_mask(D)
    d = D[ii, jj]
    n = D.shape[0]
    X0 = _classical_mds(D, dim)
    scale = np.max(d) if np.max(d) > 0 else 1.0
    rng = np.random.default_rng(seed)
    if R0 is None:
        R0 = scale / 2.0

    def unpack(p):
        return p[:-1].reshape(n, dim), np.exp(p[-1])

    def f(p):
        V, R = unpack(p)
        s = np.sqrt(1.0 + np.sum(V**2, axis=1))
        vi, vj = V[ii], V[jj]
        c = s[ii] * s[jj] - np.sum(vi * vj, axis=1)
        c = np.maximum(c, 1.0 + 1e-12)
        ac = np.arccosh(c)
        de = R * ac
        res = de - d
        val = np.sum(res**2)
        dde_dc = R / np.sqrt(c * c - 1.0)
        coef = 2.0 * res * dde_dc
        # dc/dv_i = (v_i/s_i) s_j - v_j
        gi = coef[:, None] * ((vi / s[ii, None]) * s[jj, None] - vj)
        gj = coef[:, None] * ((vj / s[jj, None]) * s[ii, None] - vi)
        grad = np.zeros_like(V)
        np.add.at(grad, ii, gi)
        np.add.at(grad, jj, gj)
        gR = np.sum(2.0 * res * ac) * R  # d/d(log R)
        return val, np.concatenate([grad.ravel(), [gR]])

    best = None
    for k in range(n_restarts):
        V0 = X0 / R0 * (1.0 if k == 0 else 1.0 + 0.2 * rng.standard_normal())
        if k > 0:
            V0 = V0 + 0.1 * np.std(V0) * rng.standard_normal(V0.shape)
        p0 = np.concatenate([V0.ravel(), [np.log(R0)]])
        res = minimize(f, p0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 3000})
        if best is None or res.fun < best.fun:
            best = res
    V, R = unpack(best.x)
    De = _hyp_dist(V, R)
    stress, distortion = _stress_and_distortion(D, De, ii, jj)
    return V, {"stress": stress, "distortion": distortion, "method": "hyperbolic",
               "dim": dim, "curvature_radius": float(R)}


# ---------------------------------------------------------------------------
# Gromov delta and the distance law
# ---------------------------------------------------------------------------


def hyperbolicity(D, max_points=60, seed=0):
    """Gromov four-point delta of a finite metric: for each quadruple
    (x,y,z,w) order the sums d(x,y)+d(z,w), d(x,z)+d(y,w), d(x,w)+d(y,z) as
    S1 >= S2 >= S3; delta = max (S1 - S2)/2. Returns (delta, delta/diam).
    Subsamples to ``max_points`` points for large inputs (O(n^4))."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if np.isnan(D).any():
        raise ValueError("hyperbolicity needs a complete distance matrix")
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, max_points, replace=False))
        D = D[np.ix_(idx, idx)]
        n = max_points
    diam = float(np.max(D))
    if n < 4:
        return 0.0, 0.0
    best = 0.0
    idx = np.arange(n)
    for x in range(n - 3):
        y = idx[x + 1:]
        # z, w > y: build arrays over (y, z, w) with y < z < w
        Y, Zc, W = np.meshgrid(y, y, y, indexing="ij")
        m = (Y < Zc) & (Zc < W)
        Y, Zc, W = Y[m], Zc[m], W[m]
        s1 = D[x, Y] + D[Zc, W]
        s2 = D[x, Zc] + D[Y, W]
        s3 = D[x, W] + D[Y, Zc]
        S = np.sort(np.stack([s1, s2, s3], axis=1), axis=1)
        if S.size:
            best = max(best, float(np.max(S[:, 2] - S[:, 1]) / 2.0))
    return best, (best / diam if diam > 0 else 0.0)


def distance_law(D, positions):
    """Fit d(r) for pairs at separation r = |x_i - x_j| to a ln r + b and to
    a r + b. Returns dict with slopes, intercepts and R^2 of both fits and
    the translation-invariance 'scatter': the r.m.s. spread of d among pairs
    at equal separation relative to its mean (0 for a clean chain)."""
    D = np.asarray(D, dtype=float)
    pos = np.asarray(positions, dtype=float)
    ii, jj = _pair_mask(D)
    r = np.abs(pos[ii] - pos[jj])
    d = D[ii, jj]
    keep = (r > 0) & np.isfinite(d)
    r, d = r[keep], d[keep]

    def fit(xv):
        A = np.vstack([xv, np.ones_like(xv)]).T
        coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        pred = A @ coef
        ss_res = np.sum((d - pred) ** 2)
        ss_tot = np.sum((d - d.mean()) ** 2)
        return float(coef[0]), float(coef[1]), float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    a_log, b_log, r2_log = fit(np.log(r))
    a_lin, b_lin, r2_lin = fit(r)
    # scatter at fixed separation
    scat = []
    for rv in np.unique(r):
        dv = d[r == rv]
        if dv.size > 1 and np.mean(dv) > 0:
            scat.append(np.std(dv) / np.mean(dv))
    return {"slope_log": a_log, "intercept_log": b_log, "r2_log": r2_log,
            "slope_lin": a_lin, "intercept_lin": b_lin, "r2_lin": r2_lin,
            "scatter": float(np.mean(scat)) if scat else 0.0,
            "n_pairs": int(r.size)}


def geometry_report(I, positions, floor=1e-12, dim=2, seed=0):
    """The full read-out for one state: distances, both embeddings, delta,
    distance law. Returns a flat dict of numbers (the failure-map row)."""
    D, dinfo = mi_distance(I, floor=floor)
    Xf, finfo = reconstruct_metric(D, dim=dim, method="mds", seed=seed)
    Xh, hinfo = reconstruct_metric(D, dim=dim, method="hyperbolic", seed=seed)
    delta, delta_rel = hyperbolicity(D)
    law = distance_law(D, positions)
    return {
        "I_max": dinfo["I_max"], "floor_fraction": dinfo["floor_fraction"],
        "diameter": float(np.nanmax(D)),
        "stress_flat": finfo["stress"], "distortion_flat": finfo["distortion"],
        "stress_hyp": hinfo["stress"], "distortion_hyp": hinfo["distortion"],
        "curvature_radius": hinfo["curvature_radius"],
        "hyperbolic_advantage": finfo["stress"] - hinfo["stress"],
        "delta": delta, "delta_rel": delta_rel,
        **law,
    }


# ---------------------------------------------------------------------------
# model families for the honest experiment
# ---------------------------------------------------------------------------


def boson_correlation_length(m):
    """Lattice correlation length of the harmonic chain: the decay rate of
    <phi_i phi_j> is 2 arcsinh(m/2) (pole of 1/(m^2 + 2 - 2 cos k))."""
    m = float(m)
    return np.inf if m == 0.0 else 1.0 / (2.0 * np.arcsinh(0.5 * m))


def _regions_single(n_points, spacing, offset=0):
    return [[offset + k * spacing] for k in range(n_points)]


def _family_massive_boson(m, N, n_points, spacing, seed=None):
    K = harmonic_chain_K(N, m, bc="periodic")
    V = ground_state_cov(K)
    regs = _regions_single(n_points, spacing)
    return mi_matrix(V, regs, "gaussian"), boson_correlation_length(m)


def _family_disordered_boson(W, N, n_points, spacing, seed=0, m=0.02):
    rng = np.random.default_rng(seed)
    K = harmonic_chain_K(N, m, bc="periodic")
    # random spring constants k_i in [1 - W, 1 + W] on every bond
    k = 1.0 + W * rng.uniform(-1.0, 1.0, size=N)
    Kd = np.zeros((N, N))
    for i in range(N):
        j = (i + 1) % N
        Kd[i, i] += k[i]; Kd[j, j] += k[i]
        Kd[i, j] -= k[i]; Kd[j, i] -= k[i]
    Kd += m * m * np.eye(N)
    V = ground_state_cov(Kd)
    regs = _regions_single(n_points, spacing, offset=N // 2 - (n_points * spacing) // 2)
    return mi_matrix(V, regs, "gaussian"), np.nan


def _family_kitaev_tfim(g, N, n_points, spacing, seed=None):
    regs = _regions_single(n_points, spacing)
    return mi_matrix(None, regs, "tfim_inf", g=g), tfim_correlation_length(g)


def _family_disordered_kitaev(W, N, n_points, spacing, seed=0, g=1.0):
    rng = np.random.default_rng(seed)
    J = 1.0 + W * rng.uniform(-1.0, 1.0, size=N - 1)
    gg = g * (1.0 + W * rng.uniform(-1.0, 1.0, size=N))
    Gamma = kitaev_chain_gamma(N, gg, J, bc="open")
    regs = _regions_single(n_points, spacing, offset=N // 2 - (n_points * spacing) // 2)
    return mi_matrix(Gamma, regs, "majorana"), np.nan


MODEL_FAMILIES = {
    "massive_boson": (_family_massive_boson, "m"),
    "disordered_boson": (_family_disordered_boson, "W"),
    "kitaev_tfim": (_family_kitaev_tfim, "g"),
    "disordered_kitaev": (_family_disordered_kitaev, "W"),
}


def failure_map(model_family, param_grid, n_points=24, spacing=2, N=None,
                seeds=(0,), floor=1e-12, dim=2):
    """The honest experiment: run the metric-reconstruction pipeline across
    a parameter grid of a non-holographic model family and tabulate where
    and how geometry-from-correlations degrades.

    model_family in MODEL_FAMILIES:
      'massive_boson'     : harmonic chain, parameter = mass m (xi = 1/m)
      'disordered_boson'  : random springs 1 +- W at small mass
      'kitaev_tfim'       : infinite TFIM/Kitaev chain, parameter g (xi = 1/|ln g|)
      'disordered_kitaev' : random J_j, g_j = 1 +- W at criticality

    Regions are single sites at positions k * spacing, k < n_points. Each row
    of the returned list carries the parameter, the correlation length, and
    every diagnostic of :func:`geometry_report` (stress of flat vs
    hyperbolic embeddings, curvature radius, Gromov delta, log/linear
    distance-law fits, translation-invariance scatter, floor fraction),
    averaged over ``seeds`` for the disordered families (geometric mean for
    the curvature radius, arithmetic otherwise; per-seed standard deviation
    in '<key>_std').
    """
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"unknown family {model_family!r}; choose from {sorted(MODEL_FAMILIES)}")
    fn, pname = MODEL_FAMILIES[model_family]
    if N is None:
        N = max(4 * n_points * spacing, 200)
    positions = [k * spacing for k in range(n_points)]
    rows = []
    for val in param_grid:
        reports, xi = [], None
        for s in seeds:
            I, xi = fn(val, N, n_points, spacing, seed=s)
            reports.append(geometry_report(I, positions, floor=floor, dim=dim))
        keys = reports[0].keys()
        row = {"family": model_family, "param": pname, "value": float(val),
               "xi": float(xi) if xi is not None else np.nan, "N": int(N),
               "n_points": n_points, "spacing": spacing, "n_seeds": len(seeds)}
        for k in keys:
            arr = np.array([r[k] for r in reports], dtype=float)
            if k == "curvature_radius":      # scale quantity: geometric mean
                row[k] = float(np.exp(np.mean(np.log(np.maximum(arr, 1e-300)))))
            else:
                row[k] = float(np.mean(arr))
            if len(seeds) > 1:
                row[k + "_std"] = float(np.std(arr))
        rows.append(row)
    return rows
