"""Toy Jacobson in 2+1: entanglement equilibrium on a square-lattice scalar.

The 1+1 toy (`vacuum.geometry.jacobson`, papers/toy-jacobson) asks how much of
wave dynamics is forced by demanding delta S = delta<K_loc> for small balls
with K_loc the local Casini-Huerta-Myers form. This module is its 2+1
extension, kept deliberately parallel:

* Dynamics: H = (1/2)(p.p + x^T K x) on an N x N square lattice with Dirichlet
  walls, site index i = ix * N + iy. The nearest-neighbour Dirichlet Laplacian
  Lap (diagonal 4, -1 on the four neighbours; wall neighbours dropped) has the
  analytic spectrum lam(qx, qy) = lam1(qx) + lam1(qy), lam1(q) =
  4 sin^2(pi q / (2(N+1))), with product-of-sines eigenvectors.  z = 1 (wave)
  takes K = Lap + m^2 (omega = |k| + O(k^3)); z = 2 (Lifshitz) takes
  K = Lap^2 + m^2 (omega = k^2 + O(k^4)).
* Region: the disk of radius R centred on the box centre ((N-1)/2, (N-1)/2),
  which is a half-integer point, so reflection about it maps sites to sites
  exactly.  disk_ball keeps the sites with r < R.
* Local energy density: the same PSD bond split as in 1+1, h_i = p_i^2/2 +
  M_i x_i^2/2 + (1/4) sum_j c_ij (x_i - x_j)^2 with c_ij = -K_ij and
  M_i = sum_j K_ij, so sum_i h_i = H exactly.
* Local ansatz: K_loc = 2 pi sum_{i in B} beta_i h_i with the CHM weight
  beta(r) = (R^2 - r^2)/(2R) (Casini-Huerta-Myers JHEP 05 (2011) 036, Sec. 2).
  `beta_at='site'` puts beta at the site centres, `beta_at='bond'` at the bond
  midpoints (the 2d analogue of the 1+1 midpoint rule).
* The conformal improvement term.  For a free scalar in d = 3 the CFT stress
  tensor is the IMPROVED one, T_00^imp = T_00^can - xi grad^2(phi^2) with
  xi = (d-2)/(4(d-1)) = 1/8; in d = 2 the improvement coefficient vanishes, so
  the 1+1 toy never saw it.  `improve` adds -2 pi xi sum_i beta_i (Lap phi^2)_i
  to K_loc; it is still a weighted sum of local operators, and `improve=None`
  in the free-weight fit makes xi a fitted parameter.
* Variations: single-mode squeezes of long wavelength, exactly as in 1+1.
  A mode with q_x even is reflection-ODD about the disk centre in x (a
  dipole-like variation) and is blind to every x-even Williamson mode of the
  disk; q_x odd gives the reflection-EVEN sector.  In 2+1 the massless scalar
  has no IR zero mode (the 1+1 one is the origin of Casini-Huerta's
  (1/2) ln ln(1/mL) term), so the even sector is the decisive test of what the
  1+1 even-sector obstruction was.

Everything is assembled from the analytic normal modes: the disk's vacuum
covariance is a (2n x 2n) block built by summing over the N^2 product modes,
so no N^2 x N^2 eigendecomposition is ever needed (K itself is only used for
the local energy density).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from vacuum.core import williamson, symplectic_inverse
from vacuum.core.gaussian import entropy
from vacuum.core.precision import entropy_precise
from vacuum.geometry.jacobson import (
    _cheb_even,
    _g_of_nu,
    _minimax_weights,
    _mode_contributions,
    _rank_profile,
    squeeze_variation,
    squeezed_ball_covariance,
)

__all__ = [
    "sine_basis",
    "square_lattice_K",
    "lattice_coords",
    "disk_ball",
    "chm_beta_2d",
    "vacuum_block_2d",
    "squeeze_modes_2d",
    "local_modular_form_2d",
    "equilibrium_residual_2d",
    "sector_split_2d",
    "first_law_audit_2d",
    "free_weight_residual_2d",
    "XI_IMPROVE_3D",
]

#: conformal improvement coefficient (d-2)/(4(d-1)) of a free scalar in d = 3.
XI_IMPROVE_3D = 0.125


# ---------------------------------------------------------------------------
# lattice, spectrum, region
# ---------------------------------------------------------------------------


def sine_basis(N):
    """(S, lam1) for the 1d Dirichlet Laplacian on N sites.

    S[i, q] = sqrt(2/(N+1)) sin(pi (i+1)(q+1)/(N+1)) (orthonormal columns) and
    lam1[q] = 4 sin^2(pi (q+1) / (2(N+1))). Under i -> N-1-i the column q has
    parity (-1)^q, so q odd (0-based) is reflection-ODD about the centre.
    """
    N = int(N)
    i = np.arange(1, N + 1)[:, None]
    q = np.arange(1, N + 1)[None, :]
    S = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * i * q / (N + 1))
    lam1 = 4.0 * np.sin(np.pi * np.arange(1, N + 1) / (2.0 * (N + 1))) ** 2
    return S, lam1


def square_lattice_K(N, z=1, m=0.0, dense=False):
    """K of the N x N Dirichlet square lattice: K = Lap^z + m^2, z = 1 or 2.

    Lap = kron(L1, I) + kron(I, L1) with L1 the 1d Dirichlet Laplacian
    (diagonal 2, -1 on neighbours), i.e. the 5-point stencil with diagonal 4.
    z = 1 is the wave lattice (omega = |k| + O(k^3)), z = 2 the biharmonic
    Lifshitz lattice (omega = k^2 + O(k^4)).  Returned as a scipy CSR matrix
    (N^2 x N^2; dense=True forces an array) -- only its rows on the ball are
    ever used, and N up to ~128 is wanted for long-wavelength variations.
    """
    N, z = int(N), int(z)
    L1 = sp.diags([np.full(N - 1, -1.0), np.full(N, 2.0), np.full(N - 1, -1.0)],
                  offsets=[-1, 0, 1], format="csr")
    I = sp.identity(N, format="csr")
    Lap = sp.kron(L1, I, format="csr") + sp.kron(I, L1, format="csr")
    if z == 1:
        K = Lap
    elif z == 2:
        K = (Lap @ Lap).tocsr()
    else:
        raise ValueError("z must be 1 (wave) or 2 (biharmonic Lifshitz)")
    if m:
        K = K + float(m) ** 2 * sp.identity(N * N, format="csr")
    K = (0.5 * (K + K.T)).tocsr()
    K.eliminate_zeros()
    return K.toarray() if dense else K


def _csr(K):
    """K as CSR (a no-op for a CSR input, a conversion for a dense one)."""
    return K if sp.issparse(K) else sp.csr_matrix(np.asarray(K, dtype=float))


def _halo_sites(K, ball):
    """`ball` plus every site its bonds reach (sorted), and the ball sites'
    positions inside that list -- the sparse analogue of jacobson._halo."""
    Ks = _csr(K)
    touched = set(int(i) for i in ball)
    for i in ball:
        touched.update(int(j) for j in Ks.indices[Ks.indptr[i]:Ks.indptr[i + 1]])
    sites = sorted(touched)
    pos = {s: n for n, s in enumerate(sites)}
    return sites, [pos[int(i)] for i in ball]


def lattice_coords(N):
    """(N^2, 2) array of (x, y) positions, site index i = ix * N + iy."""
    N = int(N)
    ix, iy = np.divmod(np.arange(N * N), N)
    return np.stack([ix, iy], axis=1).astype(float)


def disk_ball(N, R):
    """Sites of the disk of radius R centred on ((N-1)/2, (N-1)/2).

    N must be even so the centre is a half-integer point and reflection about
    it is an exact lattice symmetry. Returns dict with 'sites' (sorted list),
    'r' (radii), 'coords' (positions), 'centre', 'R', 'n'.
    """
    N = int(N)
    if N % 2:
        raise ValueError(f"need even N so the disk centre is between sites, got N={N}")
    c = 0.5 * (N - 1)
    xy = lattice_coords(N)
    r = np.hypot(xy[:, 0] - c, xy[:, 1] - c)
    sel = np.nonzero(r < float(R))[0]
    return {"sites": [int(s) for s in sel], "r": r[sel], "coords": xy[sel],
            "centre": (c, c), "R": float(R), "n": int(sel.size), "r_all": r}


def chm_beta_2d(r, R):
    """The CHM weight beta(r) = (R^2 - r^2)/(2R), clipped to 0 outside."""
    r = np.asarray(r, dtype=float)
    return np.maximum((float(R) ** 2 - r ** 2) / (2.0 * float(R)), 0.0)


# ---------------------------------------------------------------------------
# vacuum covariance of a disk, and the squeeze variations
# ---------------------------------------------------------------------------


def _omega(lam1, N, z, m):
    """omega(qx, qy) on the full N^2 mode grid, flattened as qx * N + qy."""
    lam = lam1[:, None] + lam1[None, :]
    w2 = lam ** int(z) + float(m) ** 2
    return np.sqrt(w2).ravel()


def vacuum_block_2d(N, sites, z=1, m=0.0, basis=None):
    """Ground-state covariance of H = (1/2)(p.p + x K x) restricted to `sites`,
    assembled from the analytic product modes (no N^2 x N^2 eigh).

    V = diag(K^{-1/2}, K^{1/2})/2 with K = Lap^z + m^2.
    """
    S, lam1 = sine_basis(N) if basis is None else basis
    sites = np.asarray(sites, dtype=int)
    ix, iy = np.divmod(sites, N)
    A = (S[ix, :, None] * S[iy, None, :]).reshape(sites.size, -1)  # (n, N^2)
    w = _omega(lam1, N, z, m)
    n = sites.size
    V = np.zeros((2 * n, 2 * n))
    V[:n, :n] = 0.5 * (A / w) @ A.T
    V[n:, n:] = 0.5 * (A * w) @ A.T
    return 0.5 * (V + V.T)


def squeeze_modes_2d(N, R, ratios=(4, 8, 16), parity="odd", z=1, m=0.0, basis=None,
                     q_transverse=1):
    """Long-wavelength single-mode squeezes of the 2d lattice vacuum.

    The mode is the product mode (qx, qy) with qy = `q_transverse` (1 by
    default: the smoothest transverse profile) and qx chosen so that the
    wavelength 2(N+1)/qx is closest to ratio * 2R (the ball diameter times the
    ratio, the 1+1 convention with L = 2R), with qx EVEN for parity='odd'
    (reflection-odd about the disk centre in x, a dipole-like variation) and
    qx ODD for parity='even'.

    Returns a list of dicts with keys qx, qy, ratio, wavelength, parity, u
    (full N^2 vector), omega.
    """
    S, lam1 = sine_basis(N) if basis is None else basis
    if parity not in ("odd", "even"):
        raise ValueError("parity must be 'odd' or 'even'")
    want_even_q = parity == "odd"  # qx even <=> reflection-odd
    qy = int(q_transverse)
    out = []
    for ratio in ratios:
        target = float(ratio) * 2.0 * float(R)
        q0 = 2.0 * (N + 1) / target
        cands = [q for q in (int(np.floor(q0)), int(np.ceil(q0)), int(np.floor(q0)) - 1,
                             int(np.ceil(q0)) + 1) if 1 <= q <= N and (q % 2 == 0) == want_even_q]
        if not cands:
            raise RuntimeError(f"no {parity} mode near q = {q0:.2f}")
        qx = min(cands, key=lambda q: abs(q - q0))
        u = np.outer(S[:, qx - 1], S[:, qy - 1]).ravel()
        w2 = (lam1[qx - 1] + lam1[qy - 1]) ** int(z) + float(m) ** 2
        out.append({"qx": int(qx), "qy": int(qy), "ratio": float(ratio),
                    "wavelength": 2.0 * (N + 1) / qx,
                    "parity": -1.0 if qx % 2 == 0 else +1.0,
                    "u": u, "omega": float(np.sqrt(w2))})
    return out


# ---------------------------------------------------------------------------
# the local modular ansatz
# ---------------------------------------------------------------------------


def _neighbours(N, i):
    """The four nearest lattice neighbours of site i inside the box."""
    ix, iy = divmod(int(i), int(N))
    out = []
    if ix > 0:
        out.append(i - N)
    if ix < N - 1:
        out.append(i + N)
    if iy > 0:
        out.append(i - 1)
    if iy < N - 1:
        out.append(i + 1)
    return out


def local_modular_form_2d(K, disk, N, sites=None, beta_at="site", improve=0.0,
                          weight_fn=None, improve_form="weight"):
    """G_loc of K_loc = sum_{i in B} w(r_i) h_i as a (2n x 2n) form on `sites`
    (default: the disk plus every site its bonds reach), block ordering;
    <K_loc> = (1/2) Tr(G_loc V_sites).

    h_i is the PSD bond split of H = (1/2)(p.p + x^T K x) (see the module
    docstring). `weight_fn` maps a radius to a weight and defaults to the CHM
    profile 2 pi (R^2 - r^2)/(2R); `beta_at='site'` evaluates it at the site
    centres (each bond carrying half of each endpoint's weight),
    `beta_at='bond'` evaluates it at the site centres for the site terms and at
    the bond midpoints for the bonds (counted once).  `improve` is the
    coefficient xi of the conformal improvement -xi grad^2(phi^2) of the free
    scalar's stress tensor (XI_IMPROVE_3D = 1/8 in d = 3; the term is
    implemented as -sum_i w(r_i) xi (grad^2 phi^2)_i with the 5-point stencil).
    """
    Ks = _csr(K)
    ball = [int(i) for i in disk["sites"]]
    R = float(disk["R"])
    if weight_fn is None:
        def weight_fn(r):
            return 2.0 * np.pi * chm_beta_2d(r, R)
    if sites is None:
        sites, _ = _halo_sites(Ks, ball)
    sites = [int(s) for s in sites]
    pos = {s: k for k, s in enumerate(sites)}
    n = len(sites)
    cx, cy = disk["centre"]
    w_site = np.asarray(weight_fn(np.asarray(disk["r"], dtype=float)), dtype=float)
    M = np.asarray(Ks.sum(axis=1)).ravel()

    def row(i):
        sl = slice(Ks.indptr[i], Ks.indptr[i + 1])
        return Ks.indices[sl], Ks.data[sl]

    G = np.zeros((2 * n, 2 * n))
    inb = set(ball)
    for wt, i in zip(w_site, ball):
        pi = pos[i]
        G[n + pi, n + pi] += wt
        G[pi, pi] += wt * M[i]
    if beta_at == "site":
        for wt, i in zip(w_site, ball):
            pi = pos[i]
            for j, kij in zip(*row(i)):
                j = int(j)
                if j == i:
                    continue
                cij = -0.5 * wt * kij
                pj = pos[j]
                G[pi, pi] += cij
                G[pj, pj] += cij
                G[pi, pj] -= cij
                G[pj, pi] -= cij
    elif beta_at == "bond":
        for i in ball:
            pi = pos[i]
            ixi, iyi = divmod(i, N)
            for j, kij in zip(*row(i)):
                j = int(j)
                if j == i or (j in inb and j < i):
                    continue
                ixj, iyj = divmod(j, N)
                rm = np.hypot(0.5 * (ixi + ixj) - cx, 0.5 * (iyi + iyj) - cy)
                wb = float(np.asarray(weight_fn(np.array([rm])))[0])
                if wb == 0.0:
                    continue
                cij = -wb * kij
                pj = pos[j]
                G[pi, pi] += cij
                G[pj, pj] += cij
                G[pi, pj] -= cij
                G[pj, pi] -= cij
    else:
        raise ValueError("beta_at must be 'site' or 'bond'")
    if improve:
        xi = float(improve)
        if improve_form == "weight":
            # -2 pi xi int_B beta grad^2(phi^2) with the integration by parts done on
            # the WEIGHT: grad^2 beta = -2/R inside the disk for the CHM parabola, so
            # the term is + 2 pi xi (2/R) sum_{i in B} phi_i^2.  The surface term
            # -xi oint phi^2 dropped here is a contact term on the entangling circle;
            # keeping it (improve_form='laplacian') is measured separately and does
            # NOT reproduce the first law (papers/toy-jacobson, section J).
            c = 2.0 * np.pi * xi * (2.0 / R)
            for i in ball:
                G[pos[i], pos[i]] += 2.0 * c
        elif improve_form == "laplacian":
            # the same operator discretised as -xi sum_{i in B} w_i (grad^2 phi^2)_i
            # with the 5-point stencil, i.e. with the entangling-circle contact term
            # of the discrete integration by parts kept.
            a = {}
            for wt, i in zip(w_site, ball):
                a[i] = a.get(i, 0.0) - 4.0 * wt
                for j in _neighbours(N, i):
                    a[j] = a.get(j, 0.0) + wt
            for j, aj in a.items():
                if j not in pos:
                    raise ValueError(f"improvement term reaches site {j} outside `sites`")
                G[pos[j], pos[j]] += -2.0 * xi * aj
        else:
            raise ValueError("improve_form must be 'weight' or 'laplacian'")
    return 0.5 * (G + G.T)


# ---------------------------------------------------------------------------
# the residual
# ---------------------------------------------------------------------------


def equilibrium_residual_2d(N, radii=(3, 4, 5, 6), ratios=(4, 8, 16), parity="odd",
                            z=1, m=0.0, beta_at="site", improve=0.0, K=None,
                            floor_tol=1e-10, escalate_above=1e-2, dps=40,
                            return_details=False, weight_fn=None, n_project=0,
                            improve_form="weight"):
    """r(K) = max_{R, v} |dS_v - d<K_loc>_v| / |dS_v| for disks of the given
    radii in an N x N Dirichlet box and the long-wavelength squeezes of
    `squeeze_modes_2d`, the 2+1 analogue of
    `vacuum.geometry.jacobson.equilibrium_residual`.

    dS = (1/2) Tr[G_B dV_B] through the exact Gaussian modular Hamiltonian of
    the disk (Williamson frame, per-mode floor diagnostics, mpmath escalation
    of the margins when the floor modes carry weight), d<K_loc> =
    (1/2) Tr[G_loc dV].  `n_project` > 0 projects the disk's top n_project
    Williamson modes out of the VARIATION (both quadratures), the 2+1 analogue
    of the 1+1 zero-mode projection.
    """
    N = int(N)
    basis = sine_basis(N)
    if K is None:
        K = square_lattice_K(N, z=z, m=m)
    rows = []
    for R in radii:
        disk = disk_ball(N, R)
        ball = disk["sites"]
        sites, ball_pos = _halo_sites(K, ball)
        V_B = vacuum_block_2d(N, ball, z=z, m=m, basis=basis)
        G_loc = local_modular_form_2d(K, disk, N, sites=sites, beta_at=beta_at,
                                      improve=improve, weight_fn=weight_fn,
                                      improve_form=improve_form)
        nb = len(ball)
        n_s = len(sites)
        bidx = np.asarray(ball_pos, dtype=int)
        S_w, nu_all = williamson(V_B)
        Si_w = symplectic_inverse(S_w)
        keep = np.ones(2 * nb)
        for k in range(int(n_project)):
            keep[k] = keep[nb + k] = 0.0
        for mode in squeeze_modes_2d(N, R, ratios=ratios, parity=parity, z=z, m=m,
                                     basis=basis):
            u, w = mode["u"], mode["omega"]
            dV_B = squeeze_variation(u[ball], w)
            dV_sites = squeeze_variation(u[sites], w)
            if n_project:
                Wm = Si_w @ dV_B @ Si_w.T
                dV_B = S_w @ (Wm * keep[:, None] * keep[None, :]) @ S_w.T
                sel = np.concatenate([bidx, bidx + n_s])
                dV_sites[np.ix_(sel, sel)] = dV_B
            dS, info = _mode_contributions(V_B, dV_B, floor_tol, escalate_above, dps)
            dK = 0.5 * float(np.trace(G_loc @ dV_sites))
            rows.append({
                "R": float(R), "n_ball": nb, "ratio": mode["ratio"], "qx": mode["qx"],
                "wavelength": mode["wavelength"], "parity": mode["parity"],
                "dS": float(dS), "dK": float(dK),
                "r": float(abs(dS - dK) / abs(dS)) if dS != 0.0 else np.inf,
                "abs_defect": float(abs(dS - dK)),
                "rho": float((dS - dK) / dS) if dS != 0.0 else np.inf,
                "top_mode_share": info["top_mode_share"],
                "floor_share": info["floor_share"], "n_floor": info["n_floor"],
                "escalated": bool(info["escalated"]),
                "nu_top": float(nu_all[0]), "nu_min_margin": float(nu_all.min() - 0.5),
                "cancellation": float(np.sum(np.abs(info["contributions"])) / abs(dS))
                if dS != 0.0 else np.inf,
                "n_project": int(n_project), "improve": float(improve),
            })
    rs = np.array([row["r"] for row in rows])
    r_max = float(np.max(rs)) if rs.size else 0.0
    if not return_details:
        return r_max
    per_R = {float(R): float(max(row["r"] for row in rows if row["R"] == R)) for R in radii}
    return {"residual": r_max, "rms": float(np.sqrt(np.mean(rs ** 2))) if rs.size else 0.0,
            "per_R": per_R, "rows": rows}


def first_law_audit_2d(N, R, ratio=4, eps=4e-3, parity="odd", z=1, m=0.0,
                       use_precise=False, basis=None):
    """Independent-route check of the exact first law for one 2d squeeze: the
    entropy of the EXACTLY squeezed disk state (squeezed_ball_covariance at
    +-eps, +-eps/2, Richardson-combined, O(eps^4)) against
    dS = (1/2) Tr[G_B dV_B] from the Williamson decomposition.
    """
    basis = sine_basis(N) if basis is None else basis
    disk = disk_ball(N, R)
    ball = disk["sites"]
    V_B = vacuum_block_2d(N, ball, z=z, m=m, basis=basis)
    mode = squeeze_modes_2d(N, R, ratios=(ratio,), parity=parity, z=z, m=m, basis=basis)[0]
    u, w = mode["u"][ball], mode["omega"]
    dV_B = squeeze_variation(u, w)
    dS_mod, info = _mode_contributions(V_B, dV_B, 1e-10, 1e-2, 40)
    S_of = entropy_precise if use_precise else entropy

    def central(e):
        return (S_of(squeezed_ball_covariance(V_B, u, w, +e))
                - S_of(squeezed_ball_covariance(V_B, u, w, -e))) / (2.0 * e)

    D_c, D_f = central(eps), central(0.5 * eps)
    dS_fd = (4.0 * D_f - D_c) / 3.0
    return {"dS_fd": float(dS_fd), "dS_modular": float(dS_mod),
            "rel_err": float(abs(dS_fd - dS_mod) / abs(dS_mod)) if dS_mod else np.inf,
            "R": float(R), "ratio": float(ratio), "parity": parity, "z": int(z),
            "n_ball": len(ball), "eps": float(eps),
            "nu_min_margin": float(info["nu"].min() - 0.5)}


# ---------------------------------------------------------------------------
# the weight set free (the 2+1 version of the reviewer's test)
# ---------------------------------------------------------------------------


def _radial_cheb_fns(n_modes, R):
    """Even Chebyshev profiles in xi = r/R, clipped to the disk: T_0, T_2, ...

    The CHM weight 2 pi (R^2 - r^2)/(2R) = (pi R/2)(T_0(xi) - T_2(xi)) is
    exactly the first two, so 'cheb2' already contains it.
    """
    def make(k):
        def f(r):
            r = np.asarray(r, dtype=float)
            xi = np.clip(r / float(R), 0.0, 1.0)
            out = _cheb_even(xi, n_modes)[:, k]
            return np.where(r < float(R), out, 0.0)
        return f
    return [make(k) for k in range(int(n_modes))]


def free_weight_residual_2d(N, radii=(4, 5, 6), ratios=(4, 6, 8, 12, 16), parity="odd",
                            z=1, m=0.0, beta_at="site", n_modes=(2, 3, 4), K=None,
                            improve=0.0, nonneg=True, fit_improvement=False,
                            improve_form="weight"):
    """Free radial weight: K_loc = sum_k theta_k O_k with O_k the disk's local
    energy pieces weighted by the even Chebyshev profile T_{2k}(r/R).

    For each disk radius separately, min_theta max_v |dS_v - d<K_loc>_v|/|dS_v|
    by linear programming (`_minimax_weights`), optionally under w(r_i) >= 0 at
    every ball site.  Returns per-radius residuals, the recovered coefficients
    in units of pi R / 2 (the CHM parabola is (1, -1, 0, ...)), the overlap of
    the recovered profile with the CHM one, the CHM residual on the same
    variation family, and the design matrix's rank profile.

    With `fit_improvement` the improvement coefficient xi is one further free
    (sign-unconstrained) parameter, so the fit reports what xi the first law
    prefers; XI_IMPROVE_3D = 1/8 is the CFT value.
    """
    N = int(N)
    basis = sine_basis(N)
    if K is None:
        K = square_lattice_K(N, z=z, m=m)
    out = {}
    for R in radii:
        disk = disk_ball(N, R)
        ball = disk["sites"]
        sites, _ = _halo_sites(K, ball)
        V_B = vacuum_block_2d(N, ball, z=z, m=m, basis=basis)
        modes = squeeze_modes_2d(N, R, ratios=ratios, parity=parity, z=z, m=m, basis=basis)
        dS, dV = [], []
        for mode in modes:
            dV_B = squeeze_variation(mode["u"][ball], mode["omega"])
            s, _info = _mode_contributions(V_B, dV_B, 1e-10, 1e-2, 40)
            dS.append(s)
            dV.append(squeeze_variation(mode["u"][sites], mode["omega"]))
        dS = np.array(dS)
        G_chm = local_modular_form_2d(K, disk, N, sites=sites, beta_at=beta_at,
                                      improve=improve, improve_form=improve_form)
        r_chm = float(np.max([abs(dS[v] - 0.5 * np.trace(G_chm @ dV[v])) / abs(dS[v])
                              for v in range(len(dS))]))
        per_n = {}
        for nm in n_modes:
            fns = _radial_cheb_fns(nm, R)
            cols, P = [], []
            for f in fns:
                G = local_modular_form_2d(K, disk, N, sites=sites, beta_at=beta_at,
                                          improve=improve, weight_fn=f,
                                          improve_form=improve_form)
                cols.append([0.5 * float(np.trace(G @ d)) for d in dV])
                P.append(f(np.asarray(disk["r"])))
            if fit_improvement:
                G0 = local_modular_form_2d(K, disk, N, sites=sites, beta_at=beta_at,
                                           improve=0.0, improve_form=improve_form)
                G1 = local_modular_form_2d(K, disk, N, sites=sites, beta_at=beta_at,
                                           improve=1.0, improve_form=improve_form)
                Gi = G1 - G0
                cols.append([0.5 * float(np.trace(Gi @ d)) for d in dV])
                P.append(np.zeros(len(disk["r"])))
            D = np.array(cols).T
            Pm = np.array(P).T
            theta, r = _minimax_weights(D, dS, P=Pm, nonneg=nonneg)
            scale = np.pi * R / 2.0
            rec = theta[:nm] / scale
            chm_vec = np.zeros(nm)
            chm_vec[0] = 1.0
            if nm > 1:
                chm_vec[1] = -1.0
            w_rec = Pm[:, :nm] @ theta[:nm]
            w_chm = 2.0 * np.pi * chm_beta_2d(np.asarray(disk["r"]), R)
            den = np.linalg.norm(w_rec) * np.linalg.norm(w_chm)
            per_n[int(nm)] = {
                "r_free": float(r),
                "coefficients_over_piR2": [float(c) for c in rec],
                "overlap_with_chm": float(w_rec @ w_chm / den) if den > 0 else np.nan,
                "normalisation": float(np.linalg.norm(w_rec) / np.linalg.norm(w_chm))
                if np.linalg.norm(w_chm) > 0 else np.nan,
                "xi_fitted": float(theta[nm]) if fit_improvement else None,
                "n_parameters": int(theta.size),
            }
            per_n[int(nm)]["rank_profile"] = _rank_profile(D / np.abs(dS)[:, None],
                                                           (1e-4, 1e-6, 1e-8))
        out[float(R)] = {"r_chm": r_chm, "n_variations": int(dS.size),
                         "n_ball": len(ball), "per_n_modes": per_n}
    return out


# ---------------------------------------------------------------------------
# the quadrature split of the pairing, and the pp weight of the exact K_B
# ---------------------------------------------------------------------------


def _pp_weight_stats(v, beta, r, R, interior_margin):
    """Statistics of v_i / beta_i over a disk, with an interior restriction."""
    beta = np.asarray(beta, dtype=float)
    v = np.asarray(v, dtype=float)
    ratio = v / beta
    inner = np.asarray(r, dtype=float) < float(R) - float(interior_margin)
    if not inner.any():
        inner = np.ones(ratio.size, dtype=bool)
    return {
        "slope_lsq": float(v @ beta / (beta @ beta)),
        "sum_ratio": float(v.sum() / beta.sum()),
        "median_ratio": float(np.median(ratio)),
        "max_abs_dev": float(np.max(np.abs(ratio - 1.0))),
        "interior_median_ratio": float(np.median(ratio[inner])),
        "interior_max_abs_dev": float(np.max(np.abs(ratio[inner] - 1.0))),
        "n_interior": int(inner.sum()),
    }


def sector_split_2d(N, radii=(3, 4, 5, 6, 8, 10), ratios=(4, 8, 16), parity="odd",
                    z=1, m=0.0, beta_at="site", improve=0.0, K=None, basis=None,
                    improve_form="weight", interior_margin=1.0):
    """Quadrature (xx / pp) split of the 2+1 first-law pairing, plus the pp
    weight of the disk's exact modular Hamiltonian against 2 pi beta.

    Both G_B (exact) and G_loc (the local ansatz) are block diagonal for this
    vacuum, and so is the single-mode squeeze dV, so
    dS = dS_xx + dS_pp and d<K_loc> = dK_xx + dK_pp EXACTLY, with
    dS_xx = (1/2) Tr(G_B^xx dV^xx) and so on.  Per (R, ratio) the returned row
    carries those four numbers, the sector ratios dK_pp/dS_pp and dK_xx/dS_xx,
    and dS_pp/dS -- the weight the pp sector actually carries in dS, without
    which the sector ratios cannot be read (for a squeeze dV^pp = -omega u u^T
    is the small piece).

    Per disk it also returns the pp-weight statistic.  The local ansatz has
    G_loc^pp = diag(2 pi beta_i) exactly (the p_i^2/2 term of the bond split),
    so the question "is the weight right?" is whether the exact G_B^pp is that
    diagonal matrix.  It is NOT diagonal, and its off-diagonal mass grows with
    R ('offdiag_over_diag'), so two statistics are reported and neither is
    privileged: 'diag', diag(G_B^pp)_i / 2 pi beta_i, and 'rowsum',
    (sum_j G_B^pp_ij) / 2 pi beta_i -- the k -> 0 pp weight, the combination
    `vacuum.geometry.jacobson.eh_profiles` uses in 1+1.  Each gives the
    least-squares slope over the disk, the sum ratio, the median ratio, the
    worst site deviation, and the median/worst deviation over the interior
    r < R - interior_margin.
    """
    from vacuum.core.gaussian import entanglement_hamiltonian

    N = int(N)
    basis = sine_basis(N) if basis is None else basis
    if K is None:
        K = square_lattice_K(N, z=z, m=m)
    rows, per_R = [], {}
    for R in radii:
        disk = disk_ball(N, R)
        ball = disk["sites"]
        sites, _ = _halo_sites(K, ball)
        nb, ns = len(ball), len(sites)
        V_B = vacuum_block_2d(N, ball, z=z, m=m, basis=basis)
        G_B = entanglement_hamiltonian(V_B)
        G_loc = local_modular_form_2d(K, disk, N, sites=sites, beta_at=beta_at,
                                      improve=improve, improve_form=improve_form)
        Gxx, Gpp = G_B[:nb, :nb], G_B[nb:, nb:]
        Lxx, Lpp = G_loc[:ns, :ns], G_loc[ns:, ns:]
        for mode in squeeze_modes_2d(N, R, ratios=ratios, parity=parity, z=z, m=m,
                                     basis=basis):
            u, w = mode["u"], mode["omega"]
            dV_B = squeeze_variation(u[ball], w)
            dV_s = squeeze_variation(u[sites], w)
            dS_xx = 0.5 * float(np.trace(Gxx @ dV_B[:nb, :nb]))
            dS_pp = 0.5 * float(np.trace(Gpp @ dV_B[nb:, nb:]))
            dK_xx = 0.5 * float(np.trace(Lxx @ dV_s[:ns, :ns]))
            dK_pp = 0.5 * float(np.trace(Lpp @ dV_s[ns:, ns:]))
            dS, dK = dS_xx + dS_pp, dK_xx + dK_pp
            rows.append({
                "R": float(R), "ratio": float(mode["ratio"]),
                "wavelength": float(mode["wavelength"]), "parity": parity,
                "dS_xx": dS_xx, "dS_pp": dS_pp, "dK_xx": dK_xx, "dK_pp": dK_pp,
                "dS": dS, "dK": dK,
                "dK_over_dS": dK / dS, "dS_pp_over_dS": dS_pp / dS,
                "pp_ratio": dK_pp / dS_pp, "xx_ratio": dK_xx / dS_xx,
            })
        beta = 2.0 * np.pi * chm_beta_2d(np.asarray(disk["r"], dtype=float), R)
        d_pp = np.diag(Gpp)
        offd = float(np.abs(Gpp - np.diag(d_pp)).sum() / np.abs(d_pp).sum())
        per_R[str(int(R))] = {
            "n_ball": nb,
            "offdiag_over_diag": offd,
            "diag": _pp_weight_stats(d_pp, beta, disk["r"], R, interior_margin),
            "rowsum": _pp_weight_stats(Gpp.sum(axis=1), beta, disk["r"], R, interior_margin),
            "pp_ratio_min": float(min(x["pp_ratio"] for x in rows if x["R"] == float(R))),
            "pp_ratio_max": float(max(x["pp_ratio"] for x in rows if x["R"] == float(R))),
            "xx_ratio_min": float(min(x["xx_ratio"] for x in rows if x["R"] == float(R))),
            "xx_ratio_max": float(max(x["xx_ratio"] for x in rows if x["R"] == float(R))),
        }
    return {"N": N, "z": int(z), "parity": parity, "beta_at": beta_at,
            "improve": float(improve), "interior_margin": float(interior_margin),
            "radii": [float(R) for R in radii], "ratios": [float(t) for t in ratios],
            "rows": rows, "per_R": per_R,
            "pp_ratio_range": [float(min(x["pp_ratio"] for x in rows)),
                               float(max(x["pp_ratio"] for x in rows))],
            "xx_ratio_range": [float(min(x["xx_ratio"] for x in rows)),
                               float(max(x["xx_ratio"] for x in rows))]}
