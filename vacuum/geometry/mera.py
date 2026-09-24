"""Scale-invariant ternary MERA for the critical transverse-field Ising chain.

Layer 4's second instrument (PLAN.md): a multi-scale entanglement
renormalization ansatz whose optimized tensors give (i) the ground energy
density of the infinite critical chain, (ii) the conformal data through the
scaling superoperator, and (iii) the network geometry -- minimal cuts and
graph geodesics -- that the Swingle correspondence reads as discrete AdS.

Model and normalization
-----------------------
    H = - sum_i X_i X_{i+1} - sum_i Z_i         (g = J = 1, critical)

with the two-site term h = -XX - (ZI + IZ)/2 so that sum_i h_{i,i+1} = H on
the infinite chain. Exact ground energy density (Pfeuty, Ann. Phys. 57, 79
(1970)): e0 = -(1/pi) int_0^pi dk sqrt(2 - 2 cos k) = -4/pi.

Construction (Evenbly & Vidal, PRB 79, 144108 (2009), arXiv:0707.1454)
-----------------------------------------------------------------------
Ternary 1D MERA (their Fig. 3, Sec. II-III): each layer tau maps 3 sites of
lattice L_tau to one site of L_{tau+1} through disentanglers u (2 -> 2,
unitary) and isometries w (3 -> 1, w^dag w = 1, their Eq. 2). Two-site
operators stay two-site under coarse-graining; the ascending superoperator
A has three forms A_L, A_C, A_R according to the position of the operator
relative to the disentanglers (their Eq. 11, Fig. 9); the descending
superoperator is the dual, tr[o D(rho)] = tr[A(o) rho] (their Eqs. 12-13,
Fig. 10-11). Translation invariance uses the averages
A-bar = (A_L + A_C + A_R)/3 and D-bar (their Eqs. 34, 40), under which the
energy PER SITE is the same at every level, e = tr(rho_tau h_tau).

Optimization is the linearized single-tensor update of their Sec. IV
(Eqs. 63-67, steps L1-L4): with all other tensors frozen the energy is
E = tr(w Upsilon_w) + const; the environment Upsilon_w has singular value
decomposition V S W^dag and the update is w -> -W V^dag. The scale-invariant
top (their Sec. V.B, Eqs. 70-72; Pfeifer-Evenbly-Vidal PRA 79, 040301(R)
(2009), arXiv:0810.0580, steps A1-A2) replaces the tower of identical
layers by one pair (u, w), the fixed-point density matrix rho-hat of D-bar,
and the averaged Hamiltonian h-bar = sum_m A-bar^m(h_T) truncated after a
few terms (their footnote [16]: "k ~ 2, 3 terms").

Conformal data (Pfeifer-Evenbly-Vidal 2009)
-------------------------------------------
The one-site scaling superoperator S^(1) of their Fig. 2(ii) (an operator
on the central leg of w coarse-grains to a one-site operator) has
eigenvalues lambda_alpha with scaling dimensions Delta_alpha = -log_3
lambda_alpha (their Eq. 9). Exact Ising CFT values: Delta_I = 0,
Delta_sigma = 1/8, Delta_epsilon = 1. Their Table I (chi = 22):
Delta_sigma = 0.124997, Delta_epsilon = 1.0001. The central charge from
their Eq. 16, c = 3[S(rho-hat) - S(rho-hat^(1))] in bits, gave c = 0.5007
at chi = 22.

Geometry (Vidal PRL 99, 220405 (2007); Swingle PRD 86, 065007 (2012), Sec. IV)
------------------------------------------------------------------------------
The entropy of a block is bounded by any cut through the network,
S <= sum_{bonds in cut} ln chi_bond, and the minimal cut of a block of l
sites crosses O(log_3 l) bonds; Swingle reads the sum of bond entropies
along the minimal curve as the discrete Ryu-Takayanagi length, and the
graph distance between two sites as the AdS geodesic ~ 2 log_3 |i - j|.

All tensors are real (H is real symmetric; the optimum can be chosen real).
Every function here is pure over its array arguments; the only state is the
on-disk cache of optimized tensors under ``vacuum/geometry/data``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

__all__ = [
    "TernaryMERA",
    "ising_two_site_term",
    "E0_ISING_EXACT",
    "random_mera",
    "local_init_mera",
    "optimize_mera",
    "mera_ising",
    "save_mera",
    "load_mera",
    "energy_density",
    "fixed_point_density",
    "ascend",
    "descend",
    "ascend_avg",
    "descend_avg",
    "environment",
    "scaling_dimensions",
    "scaling_dimensions_by_parity",
    "one_site_scaling_superoperator",
    "central_charge_estimate",
    "block_entropies_level0",
    "mera_network",
    "minimal_cut",
    "minimal_cut_entropy",
    "mera_geodesic",
    "DATA_DIR",
]

DATA_DIR = Path(__file__).resolve().parent / "data"

#: exact critical ground energy per site of H = -sum XX - sum Z (Pfeuty 1970)
E0_ISING_EXACT = -4.0 / np.pi

_X = np.array([[0.0, 1.0], [1.0, 0.0]])
_Z = np.array([[1.0, 0.0], [0.0, -1.0]])
_I2 = np.eye(2)


def ising_two_site_term(g=1.0):
    """Two-site term h = -XX - g(ZI + IZ)/2 of H = -sum XX - g sum Z, (2,2,2,2).

    Index convention (used for every two-site operator in this module):
    o[s, t, s', t'] = <s t| o |s' t'>, rows first.
    """
    h = -np.kron(_X, _X) - 0.5 * g * (np.kron(_Z, _I2) + np.kron(_I2, _Z))
    return h.reshape(2, 2, 2, 2)


# ---------------------------------------------------------------------------
# The three ternary-MERA diagrams (Evenbly-Vidal Fig. 9/10), as einsum strings
# ---------------------------------------------------------------------------
#
# Tensor layouts (real):
#   w[a, b, c, x] : isometry, lower legs (a, b, c) left-to-right, upper leg x
#   u[s, t, c, d] : disentangler, lower (output) legs (s, t), upper (input)
#                   legs (c, d) = (right leg of w_k, left leg of w_{k+1})
#   o[S, T, s, t] : two-site operator <S T| o |s t>
#   rho[X, Y, x, y] : two-site density matrix at the coarse level
#
# Fine sites of unit cell k: (3k, 3k+1, 3k+2) = (u_{k-1}.t, w_k.b, u_k.s).
# The full energy diagram tr[A_p(o) rho'] for the three positions p is
# written once; ascending, descending and every environment are the same
# network with one tensor's labels moved to the output (pure bookkeeping,
# so the four objects are consistent by construction).
#
# operand order: (w_bra_left, w_bra_right, u_bra, o, u_ket, w_ket_left,
#                 w_ket_right, rho)
_DIAGRAMS = {
    # o on (u_k.s, u_k.t): sits on the disentangler
    "R": ("abCx", "Defy", "STCD", "STst", "stcd", "abcX", "defY", "XYxy"),
    # o on (w_k.b, u_k.s)
    "C": ("aBCx", "Defy", "StCD", "BSbs", "stcd", "abcX", "defY", "XYxy"),
    # o on (u_{k-1}.t, w_k.b): mirror of C
    "L": ("abCx", "DEfy", "sTCD", "TEte", "stcd", "abcX", "defY", "XYxy"),
}
_OPERANDS = ("wbl", "wbr", "ub", "o", "uk", "wkl", "wkr", "rho")
_PATH_CACHE = {}

# Pairwise contraction orders (opt_einsum 'dp' optimum at uniform chi; every
# diagram then costs O(chi^8) with chi^6 intermediates, Evenbly-Vidal Sec.
# II.D). Used when opt_einsum is unavailable; numpy's own 'greedy' search
# is catastrophically worse on these networks (~chi^12).
_STATIC_PATHS = {
    "L:None": [[4, 6], [3, 6], [2, 5], [1, 4], [2, 3], [1, 2], [0, 1]],
    "L:rho": [[0, 5], [3, 4], [2, 4], [1, 3], [0, 2], [0, 1]],
    "L:o": [[0, 4], [4, 5], [3, 4], [2, 3], [1, 2], [0, 1]],
    "L:uk": [[0, 4], [4, 5], [3, 4], [0, 3], [1, 2], [0, 1]],
    "L:wkl": [[4, 5], [3, 5], [2, 4], [1, 3], [1, 2], [0, 1]],
    "L:wkr": [[0, 5], [4, 5], [3, 4], [1, 3], [1, 2], [0, 1]],
    "C:None": [[1, 6], [5, 6], [4, 5], [3, 4], [2, 3], [1, 2], [0, 1]],
    "C:rho": [[1, 6], [3, 4], [2, 4], [1, 3], [0, 2], [0, 1]],
    "C:o": [[1, 5], [4, 5], [3, 4], [2, 3], [1, 2], [0, 1]],
    "C:uk": [[1, 5], [4, 5], [3, 4], [0, 3], [1, 2], [0, 1]],
    "C:wkl": [[1, 5], [4, 5], [3, 4], [1, 3], [1, 2], [0, 1]],
    "C:wkr": [[4, 5], [3, 5], [2, 4], [0, 3], [1, 2], [0, 1]],
    "R:None": [[1, 6], [2, 3], [1, 5], [3, 4], [2, 3], [1, 2], [0, 1]],
    "R:rho": [[0, 5], [0, 4], [1, 2], [0, 3], [1, 2], [0, 1]],
    "R:o": [[0, 4], [0, 3], [2, 4], [2, 3], [1, 2], [0, 1]],
    "R:uk": [[0, 4], [0, 3], [2, 4], [2, 3], [0, 2], [0, 1]],
    "R:wkl": [[1, 5], [2, 3], [1, 4], [2, 3], [1, 2], [0, 1]],
    "R:wkr": [[0, 5], [2, 3], [1, 4], [2, 3], [1, 2], [0, 1]],
}

try:  # optional accelerator: shape-aware dynamic-programming paths
    import opt_einsum as _oe
except Exception:  # pragma: no cover
    _oe = None


def _contract(pos, tensors, remove=None, out_labels=None):
    """Contract one diagram with the named operand removed (its labels become
    the output).  ``tensors`` is a dict with keys in _OPERANDS."""
    labels = _DIAGRAMS[pos]
    ops, subs = [], []
    for name, lab in zip(_OPERANDS, labels):
        if name == remove:
            continue
        ops.append(tensors[name])
        subs.append(lab)
    if remove is None:
        out = ""
    else:
        out = labels[_OPERANDS.index(remove)] if out_labels is None else out_labels
    expr = ",".join(subs) + "->" + out
    key = (expr, tuple(t.shape for t in ops))
    path = _PATH_CACHE.get(key)
    if path is None:
        if _oe is not None:
            path = _oe.contract_path(expr, *ops, optimize="dp")[0]
        else:
            path = [tuple(p) for p in _STATIC_PATHS[f"{pos}:{remove}"]]
        path = ["einsum_path"] + [tuple(p) for p in path]
        _PATH_CACHE[key] = path
    return np.einsum(expr, *ops, optimize=path)


def _layer_tensors(u, w, o=None, rho=None):
    return {"wbl": w, "wbr": w, "ub": u, "o": o, "uk": u, "wkl": w, "wkr": w,
            "rho": rho}


def ascend(o, u, w, pos):
    """A_pos(o): two-site operator lifted one layer (Evenbly-Vidal Eq. 11)."""
    return _contract(pos, _layer_tensors(u, w, o=o), remove="rho",
                     out_labels="xyXY")


def descend(rho, u, w, pos):
    """D_pos(rho): two-site density matrix lowered one layer (Eq. 12); the
    dual of ascend, tr[o D(rho)] = tr[A(o) rho] (Eq. 13)."""
    lab = _DIAGRAMS[pos][_OPERANDS.index("o")]
    # o[S T; s t] pairs with rho[s t; S T]: output labels = (cols, rows)
    out = lab[2:] + lab[:2]
    return _contract(pos, _layer_tensors(u, w, rho=rho), remove="o",
                     out_labels=out)


def ascend_avg(o, u, w):
    """A-bar(o) = (A_L + A_C + A_R)(o)/3 (Evenbly-Vidal Eq. 34)."""
    return (ascend(o, u, w, "L") + ascend(o, u, w, "C")
            + ascend(o, u, w, "R")) / 3.0


def descend_avg(rho, u, w):
    """D-bar(rho) = (D_L + D_C + D_R)(rho)/3 (Evenbly-Vidal Eq. 40)."""
    return (descend(rho, u, w, "L") + descend(rho, u, w, "C")
            + descend(rho, u, w, "R")) / 3.0


def environment(which, h, rho, u, w):
    """Linearized environment of the disentangler ('u') or isometry ('w') of
    one layer with lower Hamiltonian term h and upper density matrix rho,
    averaged over the three positions (Evenbly-Vidal Sec. IV, Figs. 21-22):
    the ket-side tensor is removed, all bra-side copies stay frozen, so that
    e = <Upsilon, tensor> for the per-site energy."""
    t = _layer_tensors(u, w, o=h, rho=rho)
    if which == "u":
        env = sum(_contract(p, t, remove="uk") for p in ("L", "C", "R"))
    elif which == "w":
        env = sum(_contract(p, t, remove="wkl") + _contract(p, t, remove="wkr")
                  for p in ("L", "C", "R"))
    else:
        raise ValueError("which must be 'u' or 'w'")
    return env / 3.0


def _polar_update(env, nrows):
    """w -> -W V^dag from the SVD of the environment (Evenbly-Vidal Eq. 67,
    steps L2-L3): minimizes <env, x> over isometries x^T x = 1."""
    shape = env.shape
    M = env.reshape(nrows, -1)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return (-(U @ Vt)).reshape(shape)


# ---------------------------------------------------------------------------
# MERA container, initialization, optimization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TernaryMERA:
    """Scale-invariant ternary MERA: transitional layers (us, ws) below one
    scale-invariant pair (u_s, w_s); h is the physical two-site term."""

    us: tuple
    ws: tuple
    u_s: np.ndarray
    w_s: np.ndarray
    h: np.ndarray
    meta: dict = field(default_factory=dict)
    #: Z2 parity labels (+-1) of the basis at every level, physical -> top;
    #: None for a MERA without symmetry bookkeeping
    parities: tuple = None

    @property
    def chi(self):
        return int(self.w_s.shape[-1])

    @property
    def n_transitional(self):
        return len(self.ws)

    @property
    def chis(self):
        """Bond dimensions from the physical level upward."""
        return [int(self.ws[0].shape[0])] + [int(w.shape[-1]) for w in self.ws] \
            + [self.chi]


def _random_isometry(shape, rng):
    """Random real isometry with the last index as the 'upper' leg."""
    n_in = int(np.prod(shape[:-1]))
    n_out = int(shape[-1])
    A = rng.standard_normal((n_in, n_out))
    Q, _ = np.linalg.qr(A)
    return Q.reshape(shape)


def random_mera(chi, chi0=None, n_transitional=2, d=2, h=None, seed=0,
                u_identity=True):
    """A fresh (unoptimized) MERA: identity disentanglers, random isometries.

    Bond dimensions: d -> chi0 (layer 0) -> chi (layer 1) -> chi (rest).
    chi0 defaults to min(chi, d**3), so the first isometry is lossless when
    chi >= d**3.
    """
    rng = np.random.default_rng(seed)
    if h is None:
        h = ising_two_site_term()
    chi0 = min(chi, d**3) if chi0 is None else chi0
    dims = [d, chi0] + [chi] * max(0, n_transitional - 1)
    us, ws = [], []
    for tau in range(n_transitional):
        din, dout = dims[tau], dims[tau + 1]
        u = np.eye(din * din).reshape(din, din, din, din) if u_identity \
            else _random_isometry((din, din, din * din), rng).reshape(din, din, din, din)
        us.append(u)
        ws.append(_random_isometry((din, din, din, dout), rng))
    u_s = np.eye(chi * chi).reshape(chi, chi, chi, chi) if u_identity \
        else _random_isometry((chi, chi, chi * chi), rng).reshape(chi, chi, chi, chi)
    w_s = _random_isometry((chi, chi, chi, chi), rng)
    return TernaryMERA(tuple(us), tuple(ws), u_s, w_s, np.asarray(h, float),
                       {"chi": chi, "chi0": chi0, "n_transitional": n_transitional,
                        "seed": seed})


def _parity_masks(p_in, p_out):
    """0/1 masks selecting the Z2-covariant entries of an isometry
    w[a,b,c,alpha] (p_a p_b p_c = p_alpha) and of a disentangler
    u[s,t,c,d] (p_s p_t = p_c p_d) -- the symmetric-tensor structure of
    Singh, Pfeifer, Vidal, PRA 82, 050301 (2010)."""
    p_in = np.asarray(p_in, dtype=float)
    p_out = np.asarray(p_out, dtype=float)
    P3 = p_in[:, None, None] * p_in[None, :, None] * p_in[None, None, :]
    mask_w = (P3[..., None] == p_out[None, None, None, :]).astype(float)
    P2 = p_in[:, None] * p_in[None, :]
    mask_u = (P2[:, :, None, None] == P2[None, None, :, :]).astype(float)
    return mask_w, mask_u


def _block_eigvecs_by_parity(H3, p_in, n_keep, match=None):
    """Lowest eigenvectors of the three-site block Hamiltonian, diagonalized
    sector by sector under Z2 = prod of site parities; returns (columns,
    parities). With ``match`` (a +-1 vector of length n_keep) the columns
    are chosen and ordered so that column j has parity match[j]."""
    p_in = np.asarray(p_in, dtype=float)
    P3 = (p_in[:, None, None] * p_in[None, :, None] * p_in[None, None, :]).reshape(-1)
    pools = {}
    for sct in (1.0, -1.0):
        idx = np.nonzero(P3 == sct)[0]
        e, v = np.linalg.eigh(H3[np.ix_(idx, idx)])
        full = np.zeros((H3.shape[0], len(e)))
        full[idx, :] = v
        pools[sct] = (e, full)
    if match is None:
        ens = np.concatenate([pools[1.0][0], pools[-1.0][0]])
        vecs = np.concatenate([pools[1.0][1], pools[-1.0][1]], axis=1)
        pars = np.concatenate([np.ones(len(pools[1.0][0])), -np.ones(len(pools[-1.0][0]))])
        order = np.argsort(ens)[:n_keep]
        return vecs[:, order], pars[order]
    match = np.asarray(match, dtype=float)
    cols = np.zeros((H3.shape[0], n_keep))
    used = {1.0: 0, -1.0: 0}
    for j, sct in enumerate(match):
        k = used[sct]
        if k >= pools[sct][1].shape[1]:
            raise ValueError("not enough eigenvectors of the requested parity")
        cols[:, j] = pools[sct][1][:, k]
        used[sct] += 1
    return cols, match.copy()


def local_init_mera(chi, chi0=None, n_transitional=2, d=2, h=None,
                    top_selects_parity=True):
    """Deterministic Z2-symmetric warm start: identity disentanglers and,
    layer by layer, isometries spanned by the lowest-energy eigenvectors of
    the three-site block Hamiltonian h_{12} + h_{23} built from the two-site
    term of that level, diagonalized within Z2 parity sectors so every
    coarse basis state carries a definite parity (a tree-tensor-network
    initialization; the next level's term is obtained with the averaged
    ascending superoperator). The scale-invariant layer's output parities
    must coincide with its input parities; with top_selects_parity=True the
    parity CONTENT (how many even / odd states) of the scale-invariant level
    is the one preferred by the top block Hamiltonian, and the last
    transitional layer is re-solved under that constraint."""
    if h is None:
        h = ising_two_site_term()
    h = np.asarray(h, dtype=float)
    chi0 = min(chi, d**3) if chi0 is None else chi0
    dims = [d, chi0] + [chi] * max(0, n_transitional - 1) + [chi]

    def build(force_last=None):
        parities = [np.array(([1.0, -1.0] * ((d + 1) // 2))[:d])]
        us, ws = [], []
        h_cur = h
        for tau in range(n_transitional + 1):
            din, dout = dims[tau], dims[tau + 1]
            hm = h_cur.reshape(din * din, din * din)
            H3 = np.kron(hm, np.eye(din)) + np.kron(np.eye(din), hm)
            H3 = 0.5 * (H3 + H3.T)
            if tau == n_transitional:
                match = parities[tau]
            elif tau == n_transitional - 1 and force_last is not None:
                match = force_last
            else:
                match = None
            cols, pars = _block_eigvecs_by_parity(H3, parities[tau], dout, match=match)
            w = cols.reshape(din, din, din, dout)
            u = np.eye(din * din).reshape(din, din, din, din)
            us.append(u)
            ws.append(w)
            parities.append(pars)
            h_cur = ascend_avg(h_cur, u, w)
        return us, ws, parities, H3

    us, ws, parities, H3_top = build()
    if top_selects_parity and n_transitional >= 1:
        # parity content preferred by the top block Hamiltonian (free choice)
        _, free_pars = _block_eigvecs_by_parity(H3_top, parities[-2], chi, match=None)
        n_even = int(np.sum(free_pars > 0))
        cur_even = int(np.sum(np.asarray(parities[-2]) > 0))
        if n_even != cur_even:
            force = np.array([1.0] * n_even + [-1.0] * (chi - n_even))
            us, ws, parities, _ = build(force_last=force)
    return TernaryMERA(tuple(us[:-1]), tuple(ws[:-1]), us[-1], ws[-1], h,
                       {"chi": chi, "chi0": chi0, "n_transitional": n_transitional,
                        "init": "local"}, tuple(parities[:-1]))


def _shift(h):
    """Constant making h negative semidefinite (Evenbly-Vidal Sec. IV.B)."""
    d2 = h.shape[0] * h.shape[1]
    lam = np.linalg.eigvalsh(h.reshape(d2, d2))[-1]
    return float(lam)


def fixed_point_density(u, w, rho0=None, n_iter=200, tol=1e-12):
    """rho-hat with D-bar(rho-hat) = rho-hat by power iteration (warm-started
    from rho0).  Returns (rho-hat, number of iterations, last change)."""
    chi = w.shape[-1]
    if rho0 is None:
        rho = np.eye(chi * chi).reshape(chi, chi, chi, chi) / (chi * chi)
    else:
        rho = rho0
    delta = np.inf
    for it in range(1, n_iter + 1):
        new = descend_avg(rho, u, w)
        new = 0.5 * (new + new.transpose(2, 3, 0, 1))
        new /= np.einsum("abab", new)
        delta = float(np.max(np.abs(new - rho)))
        rho = new
        if delta < tol:
            break
    return rho, it, delta


def _ascend_transitional(mera, h0):
    hs = [h0]
    for u, w in zip(mera.us, mera.ws):
        hs.append(ascend_avg(hs[-1], u, w))
    return hs


def _descend_transitional(mera, rho_top):
    rhos = [rho_top]
    for u, w in zip(reversed(mera.us), reversed(mera.ws)):
        rhos.append(descend_avg(rhos[-1], u, w))
    return list(reversed(rhos))  # rhos[tau] lives on lattice L_tau


def energy_density(mera, rho_hat=None, n_iter=300):
    """e = tr(rho_0 h) per site (exact for the given tensors; Evenbly-Vidal
    Eqs. 17 and 59 with the fixed point rho-hat at the top)."""
    if rho_hat is None:
        rho_hat, _, _ = fixed_point_density(mera.u_s, mera.w_s, n_iter=n_iter)
    rhos = _descend_transitional(mera, rho_hat)
    return float(np.einsum("STst,stST", mera.h, rhos[0]))


def _hbar(hT, u_s, w_s, rho_hat, n_hbar, traceless):
    """h-bar = sum_{m<n_hbar} A-bar^m(h_T) (Evenbly-Vidal Eq. 72); with
    traceless=True the identity component tr(rho-hat A-bar^m h_T) of every
    term beyond m = 0 is removed, so the damping (negative-shift) part of the
    environment is that of a single layer while the gradient direction keeps
    the full geometric series."""
    hbar = hT.copy()
    term = hT
    chi = hT.shape[0]
    I = np.eye(chi * chi).reshape(hT.shape)
    for _ in range(n_hbar - 1):
        term = ascend_avg(term, u_s, w_s)
        if traceless:
            e_m = float(np.einsum("STst,stST", term, rho_hat))
            hbar = hbar + term - e_m * I
        else:
            hbar = hbar + term
    return hbar


def optimize_mera(mera, n_sweeps=200, n_hbar=3, n_power=20, time_limit=None,
                  callback=None, rho_hat=None, traceless=True):
    """Energy minimization by linearized SVD updates (Evenbly-Vidal Sec. IV-V;
    Pfeifer-Evenbly-Vidal steps A1-A2). Pure: returns a new MERA plus a
    history dict; ``time_limit`` (seconds) stops the sweep loop early.

    Per sweep: (1) lift the shifted h through the transitional layers with
    A-bar; (2) form h-bar = sum_{m<n_hbar} A-bar_s^m(h_T) for the
    scale-invariant layer (Evenbly-Vidal Eq. 72, truncated); (3) refresh the
    fixed point rho-hat with n_power steps of D-bar_s; (4) descend to every
    level; (5) update u then w in each transitional layer, then the
    scale-invariant pair, from their environments.
    """
    lam = _shift(mera.h)
    h0 = mera.h - lam * np.eye(mera.h.shape[0] * mera.h.shape[1]).reshape(mera.h.shape)
    us, ws = list(mera.us), list(mera.ws)
    u_s, w_s = mera.u_s, mera.w_s
    T = len(ws)
    masks = None
    if mera.parities is not None:
        par = list(mera.parities)
        masks = [_parity_masks(par[t], par[t + 1]) for t in range(T)]
        masks.append(_parity_masks(par[T], par[T]))

    def _mask(env, tau, which):
        if masks is None:
            return env
        return env * masks[tau][0 if which == "w" else 1]
    if rho_hat is None:
        rho_hat, _, _ = fixed_point_density(u_s, w_s, n_iter=200)
    history = {"energy": [], "time": []}
    t0 = time.perf_counter()
    for sweep in range(n_sweeps):
        # (1)-(2)
        hs = [h0]
        for u, w in zip(us, ws):
            hs.append(ascend_avg(hs[-1], u, w))
        hT = hs[T]
        # (3)
        rho_hat, _, _ = fixed_point_density(u_s, w_s, rho0=rho_hat, n_iter=n_power)
        # (4)
        rhos = [rho_hat]
        for u, w in zip(reversed(us), reversed(ws)):
            rhos.append(descend_avg(rhos[-1], u, w))
        rhos = list(reversed(rhos))
        e = float(np.einsum("STst,stST", h0, rhos[0])) + lam
        history["energy"].append(e)
        history["time"].append(time.perf_counter() - t0)
        if callback is not None:
            callback(sweep, e)
        # (5) transitional layers, bottom-up, re-lifting h as we go
        h_low = h0
        for tau in range(T):
            rho_up = rhos[tau + 1]
            env_u = _mask(environment("u", h_low, rho_up, us[tau], ws[tau]), tau, "u")
            us[tau] = _polar_update(env_u, env_u.shape[0] * env_u.shape[1])
            env_w = _mask(environment("w", h_low, rho_up, us[tau], ws[tau]), tau, "w")
            ws[tau] = _polar_update(env_w, env_w.shape[0] * env_w.shape[1] * env_w.shape[2])
            h_low = ascend_avg(h_low, us[tau], ws[tau])
        # scale-invariant pair (h-bar with the fresh h_T)
        hbar = _hbar(h_low, u_s, w_s, rho_hat, n_hbar, traceless)
        env_u = _mask(environment("u", hbar, rho_hat, u_s, w_s), T, "u")
        u_s = _polar_update(env_u, env_u.shape[0] * env_u.shape[1])
        env_w = _mask(environment("w", hbar, rho_hat, u_s, w_s), T, "w")
        w_s = _polar_update(env_w, env_w.shape[0] * env_w.shape[1] * env_w.shape[2])
        if time_limit is not None and time.perf_counter() - t0 > time_limit:
            break
    meta = dict(mera.meta)
    meta.update({"n_sweeps_done": meta.get("n_sweeps_done", 0) + len(history["energy"]),
                 "last_energy": history["energy"][-1] if history["energy"] else None})
    out = TernaryMERA(tuple(us), tuple(ws), u_s, w_s, mera.h, meta, mera.parities)
    return out, history


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def _cache_path(chi, tag="ising"):
    return DATA_DIR / f"mera_{tag}_chi{int(chi)}.npz"


def save_mera(mera, path=None, extra_meta=None):
    path = Path(path) if path is not None else _cache_path(mera.chi)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"h": mera.h, "u_s": mera.u_s, "w_s": mera.w_s,
              "n_transitional": np.array(len(mera.ws))}
    for i, (u, w) in enumerate(zip(mera.us, mera.ws)):
        arrays[f"u{i}"] = u
        arrays[f"w{i}"] = w
    if mera.parities is not None:
        for i, p in enumerate(mera.parities):
            arrays[f"par{i}"] = np.asarray(p, dtype=float)
    np.savez(path, **arrays)
    meta = dict(mera.meta)
    if extra_meta:
        meta.update(extra_meta)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path


def load_mera(path):
    path = Path(path)
    with np.load(path) as z:
        T = int(z["n_transitional"])
        us = tuple(z[f"u{i}"] for i in range(T))
        ws = tuple(z[f"w{i}"] for i in range(T))
        h, u_s, w_s = z["h"], z["u_s"], z["w_s"]
        pars = tuple(z[f"par{i}"] for i in range(T + 1)) if "par0" in z.files else None
    meta = {}
    jp = path.with_suffix(".json")
    if jp.exists():
        with open(jp) as f:
            meta = json.load(f)
    return TernaryMERA(us, ws, u_s, w_s, h, meta, pars)


def mera_ising(chi=8, n_sweeps=2000, time_limit=150.0, cache="auto", seed=0,
               n_transitional=2, callback=None, init="local", n_hbar=3):
    """Optimized scale-invariant MERA for the critical Ising chain.

    cache='auto' loads vacuum/geometry/data/mera_ising_chi{chi}.npz when it
    exists (tests take this path); 'refresh' re-optimizes and overwrites;
    'none' optimizes without touching the disk. The default optimization
    budget is ``time_limit`` seconds (2.5 min) or ``n_sweeps`` sweeps,
    whichever comes first; the shipped caches were produced with
    time_limit=None and the full 2000 sweeps (about 220 s single-process at
    chi = 8, see the JSON sidecars), Z2-symmetric local warm start and the
    traceless three-term h-bar.
    """
    path = _cache_path(chi)
    if cache == "auto" and path.exists():
        return load_mera(path)
    if init == "local":
        m0 = local_init_mera(chi, n_transitional=n_transitional)
    else:
        m0 = random_mera(chi, n_transitional=n_transitional, seed=seed)
    m, hist = optimize_mera(m0, n_sweeps=n_sweeps, n_hbar=n_hbar,
                            time_limit=time_limit, callback=callback)
    m = TernaryMERA(m.us, m.ws, m.u_s, m.w_s, m.h,
                    {**m.meta, "n_sweeps": len(hist["energy"]),
                     "wall_time_s": hist["time"][-1] if hist["time"] else 0.0,
                     "energy_history_tail": hist["energy"][-5:]}, m.parities)
    if cache in ("auto", "refresh"):
        save_mera(m, path)
    return m


# ---------------------------------------------------------------------------
# conformal data
# ---------------------------------------------------------------------------


def one_site_scaling_superoperator(w):
    """Matrix of S^(1) (Pfeifer-Evenbly-Vidal Fig. 2(ii)) on vec(o), o a
    one-site operator on the central leg of w: S^(1)(o) = w^dag (1 x o x 1) w.
    Returned as a (chi_out^2, chi_in^2) matrix M[(x,X),(B,b)]."""
    M = np.einsum("aBcx,abcX->xXBb", w, w)
    chi_out, chi_in = w.shape[-1], w.shape[1]
    return M.reshape(chi_out * chi_out, chi_in * chi_in)


def scaling_dimensions(mera, n=8, which="one-site", tol=1e-10):
    """Scaling dimensions Delta = -log_3 |lambda| from the scaling
    superoperator of the scale-invariant layer (Pfeifer-Evenbly-Vidal Eq. 9,
    Delta_alpha = -log_3 lambda_alpha). which='one-site' diagonalizes S^(1)
    exactly (chi^2 x chi^2); which='two-site' uses ARPACK on the averaged
    two-site superoperator A-bar (their Fig. 1(v)). Returns the n smallest
    Delta (sorted) and the eigenvalues."""
    if which == "one-site":
        M = one_site_scaling_superoperator(mera.w_s)
        lam = np.linalg.eigvals(M)
    elif which == "two-site":
        chi = mera.chi
        dim = chi**4

        def mv(v):
            o = v.reshape(chi, chi, chi, chi)
            return ascend_avg(o, mera.u_s, mera.w_s).reshape(-1)

        op = LinearOperator((dim, dim), matvec=mv, dtype=float)
        k = min(n + 4, dim - 2)
        lam = eigs(op, k=k, which="LM", tol=tol, return_eigenvectors=False)
    else:
        raise ValueError("which must be 'one-site' or 'two-site'")
    mod = np.abs(lam)
    order = np.argsort(-mod)
    lam = lam[order][:n]
    with np.errstate(divide="ignore"):
        delta = -np.log(np.maximum(np.abs(lam), 1e-300)) / np.log(3.0)
    return delta, lam


def scaling_dimensions_by_parity(mera, n=6, which="two-site", tol=1e-10):
    """Scaling dimensions split by the Z2 parity of the scaling operator:
    the spin sigma is the lowest ODD operator, the energy epsilon the lowest
    nontrivial EVEN one (Pfeifer-Evenbly-Vidal 2009, Fig. 3; Ising CFT:
    Delta_sigma = 1/8, Delta_epsilon = 1, descendants at 9/8 (x2) and 2
    (x4)). which='one-site': S^(1) on the central leg of w, parity of
    o[B, b] = p_B p_b, dense; which='two-site': the averaged two-site
    superoperator A-bar of Fig. 1(v), parity of o[X, Y, x, y] =
    p_X p_Y p_x p_y, ARPACK restricted to the sector. Requires parity
    labels. Returns {'even': deltas, 'odd': deltas} (each sorted, n smallest)
    and 'leakage', the largest entry of w_s outside the covariant pattern."""
    if mera.parities is None:
        raise ValueError("MERA carries no parity labels")
    p = np.asarray(mera.parities[-1], dtype=float)
    chi = mera.chi
    out = {}
    if which == "one-site":
        M = one_site_scaling_superoperator(mera.w_s)
        P = (p[:, None] * p[None, :]).reshape(-1)
        for name, sct in (("even", 1.0), ("odd", -1.0)):
            idx = np.nonzero(P == sct)[0]
            lam = np.linalg.eigvals(M[np.ix_(idx, idx)])
            mod = np.sort(np.abs(lam))[::-1][:n]
            with np.errstate(divide="ignore"):
                out[name] = -np.log(np.maximum(mod, 1e-300)) / np.log(3.0)
    elif which == "two-site":
        P = (p[:, None, None, None] * p[None, :, None, None]
             * p[None, None, :, None] * p[None, None, None, :]).reshape(-1)
        for name, sct in (("even", 1.0), ("odd", -1.0)):
            idx = np.nonzero(P == sct)[0]
            dim = idx.size

            def mv(v, idx=idx):
                full = np.zeros(chi**4)
                full[idx] = v
                o = full.reshape(chi, chi, chi, chi)
                return ascend_avg(o, mera.u_s, mera.w_s).reshape(-1)[idx]

            op = LinearOperator((dim, dim), matvec=mv, dtype=float)
            k = min(n + 4, dim - 2)
            lam = eigs(op, k=k, which="LM", tol=tol, return_eigenvectors=False)
            mod = np.sort(np.abs(lam))[::-1][:n]
            with np.errstate(divide="ignore"):
                out[name] = -np.log(np.maximum(mod, 1e-300)) / np.log(3.0)
    else:
        raise ValueError("which must be 'one-site' or 'two-site'")
    mask_w, _ = _parity_masks(p, p)
    out["leakage"] = float(np.max(np.abs(mera.w_s * (1.0 - mask_w))))
    return out


def _entropy_psd(rho_mat):
    p = np.linalg.eigvalsh(0.5 * (rho_mat + rho_mat.T))
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


def central_charge_estimate(mera, rho_hat=None):
    """c = 3[S(rho-hat) - S(rho-hat^(1))]/ln 2 (Pfeifer-Evenbly-Vidal Eq. 16,
    converted from bits to nats): the entropy increment of the fixed-point
    state from one to two coarse sites. Returns (c, S2, S1)."""
    if rho_hat is None:
        rho_hat, _, _ = fixed_point_density(mera.u_s, mera.w_s, n_iter=300)
    chi = mera.chi
    S2 = _entropy_psd(rho_hat.reshape(chi * chi, chi * chi))
    rho1 = np.einsum("XYxY->Xx", rho_hat)
    rho1b = np.einsum("XYXy->Yy", rho_hat)
    S1 = 0.5 * (_entropy_psd(rho1) + _entropy_psd(rho1b))
    return 3.0 * (S2 - S1) / np.log(2.0), S2, S1


def block_entropies_level0(mera, rho_hat=None):
    """Entropies (nats) of one- and two-site blocks of the PHYSICAL chain,
    from the exact reduced density matrices of the MERA state, resolved by
    position in the 3-site unit cell. Returns dict with 'S1' (3 values, one
    per site of the cell), 'S2' (3 values: positions L, C, R)."""
    if rho_hat is None:
        rho_hat, _, _ = fixed_point_density(mera.u_s, mera.w_s, n_iter=300)
    rhos = _descend_transitional(mera, rho_hat)
    rho1 = rhos[1]
    u, w = mera.us[0], mera.ws[0]
    d = w.shape[0]
    S2, S1 = {}, {}
    for p in ("L", "C", "R"):
        r = descend(rho1, u, w, p)
        r = 0.5 * (r + r.transpose(2, 3, 0, 1))
        S2[p] = _entropy_psd(r.reshape(d * d, d * d))
        S1[p] = (_entropy_psd(np.einsum("XYxY->Xx", r)),
                 _entropy_psd(np.einsum("XYXy->Yy", r)))
    return {"S2": S2, "S1": S1}


# ---------------------------------------------------------------------------
# network geometry: minimal cuts and geodesics
# ---------------------------------------------------------------------------


def mera_network(mera, n_layers):
    """Finite ternary-MERA network on a periodic chain of N = 3**n_layers
    sites: nodes are physical sites, disentanglers, isometries and a top
    node; edges carry ln(chi_bond). Returns (node_index, edges) with edges as
    (i, j, weight, level) tuples; level = layer of the bond's lower end."""
    N = 3**n_layers
    chis = mera.chis
    nodes = {}

    def nid(key):
        if key not in nodes:
            nodes[key] = len(nodes)
        return nodes[key]

    edges = []
    for i in range(N):
        nid(("s", 0, i))
    for tau in range(n_layers):
        n_low = 3 ** (n_layers - tau)
        n_up = n_low // 3
        chi_low = chis[min(tau, len(chis) - 1)]
        chi_up = chis[min(tau + 1, len(chis) - 1)]
        wl, wu = np.log(chi_low), np.log(chi_up)
        for k in range(n_up):
            # disentangler u_k on fine sites (3k+2, 3k+3)
            uk = nid(("u", tau, k))
            edges.append((nid(("s", tau, (3 * k + 2) % n_low)), uk, wl, tau))
            edges.append((nid(("s", tau, (3 * k + 3) % n_low)), uk, wl, tau))
            # isometry w_k with legs (u_{k-1}.t, site 3k+1, u_k.s)
            wk = nid(("w", tau, k))
            edges.append((nid(("u", tau, (k - 1) % n_up)), wk, wl, tau))
            edges.append((nid(("s", tau, 3 * k + 1)), wk, wl, tau))
            edges.append((uk, wk, wl, tau))
            edges.append((wk, nid(("s", tau + 1, k)), wu, tau + 1))
    top = nid(("top",))
    for k in range(1):
        edges.append((nid(("s", n_layers, 0)), top, np.log(chis[-1]), n_layers))
    return nodes, edges


def minimal_cut(mera, region, n_layers=None, weights="count"):
    """Minimal cut separating the physical sites in ``region`` from the rest
    of the finite network (max-flow / min-cut). weights='count' minimizes
    the NUMBER of bonds crossed (the geometric minimal curve of Swingle's
    map: a block of 3^k sites is cut by 4k - 1 bonds, four per
    coarse-graining layer); weights='lnchi' minimizes sum ln chi_bond, the
    tightest form of Vidal's entropy bound S <= sum_{cut} ln chi. Returns
    (bound_nats, cut_edges) with bound = sum of ln chi over the bonds of
    the returned cut, cut_edges as (i, j, ln chi, level)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    region = sorted(set(int(r) for r in region))
    if n_layers is None:
        n_layers = max(2, int(np.ceil(np.log(max(region) + 2) / np.log(3))) + 1)
    N = 3**n_layers
    if max(region) >= N:
        raise ValueError("region does not fit in the network")
    nodes, edges = mera_network(mera, n_layers)
    n = len(nodes) + 2
    src, snk = n - 2, n - 1
    scale = 1000  # integer capacities for scipy's max-flow
    rows, cols, caps = [], [], []
    for (i, j, wgt, _) in edges:
        c = int(round(wgt * scale)) if weights == "lnchi" else scale
        rows += [i, j]
        cols += [j, i]
        caps += [c, c]
    big = 10**9
    for i in range(N):
        s = nodes[("s", 0, i)]
        if i in region:
            rows.append(src); cols.append(s); caps.append(big)
        else:
            rows.append(s); cols.append(snk); caps.append(big)
    G = csr_matrix((np.array(caps, dtype=np.int64), (rows, cols)), shape=(n, n))
    G.sum_duplicates()
    res = maximum_flow(G, src, snk)
    flow = res.flow if hasattr(res, "flow") else res.residual
    # residual graph reachability from the source gives the min cut
    resid = G - flow
    resid = resid.tocsr()
    reach = np.zeros(n, dtype=bool)
    stack = [src]
    reach[src] = True
    while stack:
        v = stack.pop()
        row = resid.getrow(v)
        for j, c in zip(row.indices, row.data):
            if c > 0 and not reach[j]:
                reach[j] = True
                stack.append(j)
    cut = [(i, j, wgt, lvl) for (i, j, wgt, lvl) in edges if reach[i] != reach[j]]
    bound = float(sum(e[2] for e in cut))
    return bound, cut


def minimal_cut_entropy(mera, region, n_layers=None, rho_hat=None):
    """Swingle's minimal-curve entropy estimate (PRD 86, 065007, Sec. IV):
    the number of bonds on the minimal cut times the entropy the MERA itself
    assigns to one cut bond. Each coarse-graining layer adds four bonds to
    the minimal cut of a block (on either side one disentangler bond and one
    isometry bond: n_cut = 4k - 1 for l = 3^k), and the entropy the MERA
    assigns to one layer is (c_MERA/3) ln 3 with c_MERA from
    :func:`central_charge_estimate` (Pfeifer-Evenbly-Vidal Eq. 16), so
    s_bond = (ln 3 / 4 ln 2)[S(rho-hat) - S(rho-hat^(1))]. Returns the
    estimate in nats (no ultraviolet constant: compare DIFFERENCES between
    block sizes to the exact block entropies, (c/3) ln(l2/l1)) together with
    the bond count and the naive ln-chi bound, as (S_est, n_bonds, bound)."""
    bound, cut = minimal_cut(mera, region, n_layers=n_layers, weights="count")
    n_bonds = len(cut)
    c, S2, S1 = central_charge_estimate(mera, rho_hat=rho_hat)
    s_bond = (np.log(3.0) / (4.0 * np.log(2.0))) * (S2 - S1)
    return float(n_bonds * s_bond), n_bonds, bound


def mera_geodesic(mera, i, j, n_layers=None):
    """Graph distance (number of bonds) between physical sites i and j
    through the finite network: the discrete AdS geodesic of Swingle's map.
    Each coarse-graining layer costs three bonds (site -> disentangler ->
    isometry -> coarse site) on the way up and three on the way down, so
    the distance grows as ~ 6 log_3 |i - j| bonds, i.e. 2 log_3 |i - j|
    layers."""
    from collections import deque

    i, j = int(i), int(j)
    if n_layers is None:
        n_layers = max(2, int(np.ceil(np.log(max(i, j) + 2) / np.log(3))) + 1)
    nodes, edges = mera_network(mera, n_layers)
    adj = {}
    for (a, b, _, _) in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    s, t = nodes[("s", 0, i)], nodes[("s", 0, j)]
    dist = {s: 0}
    q = deque([s])
    while q:
        v = q.popleft()
        if v == t:
            return dist[v]
        for nb in adj.get(v, []):
            if nb not in dist:
                dist[nb] = dist[v] + 1
                q.append(nb)
    raise RuntimeError("sites not connected")
