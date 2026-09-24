"""HaPPY holographic codes in the stabilizer formalism.

Pastawski, Yoshida, Harlow, Preskill, "Holographic quantum error-correcting
codes: toy models for the bulk/boundary correspondence", JHEP 06 (2015) 149,
arXiv:1503.06237 -- cited below as HaPPY.

Building blocks
---------------
* The [[5,1,3]] code (HaPPY App. A.1, Eq. A.1) with stabilizer generators
  S1 = XZZXI, S2 = IXZZX, S3 = XIXZZ, S4 = ZXIXZ and logical operators
  X-bar = XXXXX, Z-bar = ZZZZZ (Eq. A.3). Adding a sixth qubit entangled
  with the logical qubit gives the six-qubit perfect tensor, the [[6,0,4]]
  stabilizer state of Eq. A.5 with S5' = X^{x6}, S6' = Z^{x6}; any three of
  its qubits are maximally mixed (HaPPY Def. 2: a perfect tensor).
* The pentagon code (HaPPY Sec. 3, Fig. 4b, Theorem 1): a {5,4} tiling of
  the hyperbolic disc, one perfect tensor per pentagon, five planar legs
  contracted with neighbours or left open on the boundary, the sixth leg a
  bulk (logical) input. Layers are counted by graph distance from the
  central tensor as in HaPPY App. C.1: at layer n there are f_n tensors
  with one leg to the previous layer and g_n with two, (f,g)_{n+1} =
  [[2,1],[1,1]] (f,g)_n from (f_1, g_1) = (5, 0) (Eq. C.1-C.2), giving
  N_boundary = 4 f_n + 3 g_n boundary legs (Eq. C.4) and
  N_bulk = 1 + sum_k (f_k + g_k) (Eq. C.5).
* The hexagon state (HaPPY Fig. 4a): the same six-qubit perfect tensor on
  the {6,4} tiling with every leg contracted -- a holographic *state*, the
  setting of their Theorem 2.

Tensor contraction as stabilizer projection
-------------------------------------------
Contracting two legs is projecting the two qubits on the Bell state
|Phi+> (stabilized by XX and ZZ): each contraction is two Pauli measurements
postselected on +1, then the pair is removed. Since the network is an
isometry (HaPPY Theorem 1) the postselection is never null. The result is
the Choi state |Psi> = sum_a |a>_bulk (x) V|a>_boundary of the encoding
isometry V, a stabilizer state on N_boundary + N_bulk qubits. From it:

* boundary entropies of a code state with a product bulk input, in bits,
  S_A = rank_GF(2)(G|_A) - |A| (Fattal et al., quant-ph/0406168);
* reconstruction: O_A V = V L  iff  (O_A (x) L^T) stabilizes |Psi>, so a
  bulk Pauli L on bulk qubit b is reconstructible on A iff the stabilizer
  group has an element whose bulk part is L^T on b and whose boundary part
  is supported on A -- a GF(2) linear system.

Geometry
--------
* The greedy algorithm (HaPPY Sec. 4.2): start from A; absorb any tensor
  with at least half of its legs already in the region; the frontier when
  no move remains is the greedy geodesic gamma*_A, the absorbed tensors the
  causal wedge C[A] (Def. 5), every bulk operator in which is
  reconstructible on connected A (Theorem 5).
* Discrete Ryu-Takayanagi: S_A = |gamma_A| for connected A on a
  non-positively curved planar network of perfect tensors with all legs
  contracted (Theorem 2); for codes and disconnected regions
  S_A >= |gamma*_A intersect gamma*_{A^c}| (Theorem 3, Eq. 4.8) with
  equality S_A = |gamma*_A| when the two greedy geodesics coincide.
  The minimal cut itself is computed by max-flow/min-cut.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "HaPPYCode",
    "happy_code",
    "five_qubit_code_generators",
    "perfect_tensor_generators",
    "greedy_wedge",
    "min_cut",
    "boundary_entropy",
    "rt_entropy",
    "bulk_reconstruction",
    "reconstructible",
    "layer_counts",
    "StabilizerTableau",
    "MinCut",
    "RTResult",
]

# ---------------------------------------------------------------------------
# Pauli / stabilizer bookkeeping over GF(2)
# ---------------------------------------------------------------------------

# single-qubit code: 0 = I, 1 = X, 2 = Z, 3 = Y (x=1, z=1).
# _PHASE[p, q] = exponent g with P Q = i^g R (R Hermitian Pauli)
_PHASE = np.zeros((4, 4), dtype=np.int64)
_PHASE[1, 2] = 3  # X Z = -i Y
_PHASE[2, 1] = 1  # Z X =  i Y
_PHASE[1, 3] = 1  # X Y =  i Z
_PHASE[3, 1] = 3  # Y X = -i Z
_PHASE[3, 2] = 1  # Y Z =  i X
_PHASE[2, 3] = 3  # Z Y = -i X

_PAULI_XZ = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}
_XZ_PAULI = {v: k for k, v in _PAULI_XZ.items()}


def _pauli_mult(x1, z1, r1, x2, z2, r2):
    """Product of two Hermitian Pauli strings (+-1 signs r in {0, 1}); the
    strings must commute so that the product is Hermitian."""
    g = _PHASE[x1 + 2 * z1, x2 + 2 * z2].sum() % 4
    if g % 2:
        raise ValueError("product of anticommuting Paulis is not Hermitian")
    return x1 ^ x2, z1 ^ z2, (r1 + r2 + (g // 2)) % 2


def _rank_gf2(M):
    """Rank over GF(2) of a 0/1 matrix."""
    A = (np.asarray(M, dtype=np.uint8) & 1).copy()
    n_rows, n_cols = A.shape
    r = 0
    for c in range(n_cols):
        if r >= n_rows:
            break
        piv = np.nonzero(A[r:, c])[0]
        if piv.size == 0:
            continue
        p = r + piv[0]
        if p != r:
            A[[r, p]] = A[[p, r]]
        mask = A[:, c].astype(bool)
        mask[r] = False
        A[mask] ^= A[r]
        r += 1
    return r


def _solve_gf2(A, b):
    """Solve x A = b over GF(2) (x a row combination of the rows of A).
    Returns x (uint8 vector) or None if inconsistent."""
    A = (np.asarray(A, dtype=np.uint8) & 1)
    b = (np.asarray(b, dtype=np.uint8) & 1)
    n_rows, n_cols = A.shape
    # Augment each row with its identity tag to track the combination.
    M = np.concatenate([A, np.eye(n_rows, dtype=np.uint8)], axis=1)
    r = 0
    pivots = []
    for c in range(n_cols):
        if r >= n_rows:
            break
        piv = np.nonzero(M[r:, c])[0]
        if piv.size == 0:
            continue
        p = r + piv[0]
        if p != r:
            M[[r, p]] = M[[p, r]]
        mask = M[:, c].astype(bool)
        mask[r] = False
        M[mask] ^= M[r]
        pivots.append(c)
        r += 1
    # reduce b against the pivots
    x = np.zeros(n_rows, dtype=np.uint8)
    bb = b.copy()
    for i, c in enumerate(pivots):
        if bb[c]:
            bb ^= M[i, :n_cols]
            x ^= M[i, n_cols:]
    if bb.any():
        return None
    return x


@dataclass
class StabilizerTableau:
    """Generators of a stabilizer state: X part, Z part (n_gen x n uint8)
    and signs r (0 -> +1, 1 -> -1). Methods return NEW tableaux (pure)."""

    X: np.ndarray
    Z: np.ndarray
    r: np.ndarray

    @property
    def n(self):
        return int(self.X.shape[1])

    @property
    def n_gen(self):
        return int(self.X.shape[0])

    def copy(self):
        return StabilizerTableau(self.X.copy(), self.Z.copy(), self.r.copy())

    def anticommuting_rows(self, x, z):
        return np.nonzero(((self.X @ z) + (self.Z @ x)) % 2)[0]

    def membership_sign(self, x, z):
        """If +-P (P = (x, z)) is in the group return its sign r (0/1);
        else None."""
        A = np.concatenate([self.X, self.Z], axis=1)
        b = np.concatenate([x, z])
        comb = _solve_gf2(A, b)
        if comb is None:
            return None
        px = np.zeros(self.n, dtype=np.uint8)
        pz = np.zeros(self.n, dtype=np.uint8)
        pr = 0
        for i in np.nonzero(comb)[0]:
            px, pz, pr = _pauli_mult(px, pz, pr, self.X[i], self.Z[i], int(self.r[i]))
        return int(pr)

    def project(self, x, z, sign=0):
        """Measure the Pauli (x, z) and postselect the outcome (-1)^sign.
        Raises ValueError if the projection is null."""
        x = np.asarray(x, dtype=np.uint8)
        z = np.asarray(z, dtype=np.uint8)
        anti = self.anticommuting_rows(x, z)
        T = self.copy()
        if anti.size == 0:
            s = self.membership_sign(x, z)
            if s is None:
                raise RuntimeError("Pauli commutes with all generators but is "
                                   "not in the group: state is not pure")
            if s != sign:
                raise ValueError("null projection: the opposite eigenvalue is "
                                 "fixed by the stabilizer group")
            return T
        p = anti[0]
        for i in anti[1:]:
            T.X[i], T.Z[i], T.r[i] = _pauli_mult(T.X[i], T.Z[i], int(T.r[i]),
                                                 T.X[p], T.Z[p], int(T.r[p]))
        T.X[p], T.Z[p], T.r[p] = x.copy(), z.copy(), sign
        return T

    def contract(self, a, b):
        """Project qubits (a, b) on |Phi+> (XX = ZZ = +1) and remove them."""
        n = self.n
        xx = np.zeros(n, dtype=np.uint8); xx[a] = xx[b] = 1
        zz = np.zeros(n, dtype=np.uint8); zz[a] = zz[b] = 1
        zero = np.zeros(n, dtype=np.uint8)
        T = self.project(xx, zero).project(zero, zz)
        # rows carrying the two Bell generators
        rows_xx = [i for i in range(T.n_gen) if np.array_equal(T.X[i], xx) and not T.Z[i].any()]
        rows_zz = [i for i in range(T.n_gen) if np.array_equal(T.Z[i], zz) and not T.X[i].any()]
        if not rows_xx or not rows_zz:
            raise RuntimeError("Bell generators missing after contraction")
        ixx, izz = rows_xx[0], rows_zz[0]
        # clear support on qubit a using the Bell generators; support on b
        # then vanishes automatically (the pair is in |Phi+>)
        for i in range(T.n_gen):
            if i in (ixx, izz):
                continue
            if T.X[i, a]:
                T.X[i], T.Z[i], T.r[i] = _pauli_mult(T.X[i], T.Z[i], int(T.r[i]),
                                                     T.X[ixx], T.Z[ixx], int(T.r[ixx]))
            if T.Z[i, a]:
                T.X[i], T.Z[i], T.r[i] = _pauli_mult(T.X[i], T.Z[i], int(T.r[i]),
                                                     T.X[izz], T.Z[izz], int(T.r[izz]))
            if T.X[i, b] or T.Z[i, b]:
                raise RuntimeError("residual support on contracted qubit")
        keep_rows = [i for i in range(T.n_gen) if i not in (ixx, izz)]
        keep_cols = [q for q in range(n) if q not in (a, b)]
        return StabilizerTableau(T.X[np.ix_(keep_rows, keep_cols)],
                                 T.Z[np.ix_(keep_rows, keep_cols)],
                                 T.r[keep_rows])

    def project_single(self, qubit, pauli="Z", sign=0):
        x = np.zeros(self.n, dtype=np.uint8)
        z = np.zeros(self.n, dtype=np.uint8)
        x[qubit], z[qubit] = _PAULI_XZ[pauli]
        return self.project(x, z, sign)

    def entropy_bits(self, qubits):
        """S_A = rank(G|_A) - |A| in bits for a pure stabilizer state."""
        qubits = list(qubits)
        if not qubits:
            return 0
        sub = np.concatenate([self.X[:, qubits], self.Z[:, qubits]], axis=1)
        return int(_rank_gf2(sub) - len(qubits))

    def element_with(self, fixed_cols, target_x, target_z):
        """A group element whose (x, z) on ``fixed_cols`` equals the targets
        (all other columns free). Returns (x, z, r) or None."""
        A = np.concatenate([self.X[:, fixed_cols], self.Z[:, fixed_cols]], axis=1)
        b = np.concatenate([target_x, target_z]).astype(np.uint8)
        comb = _solve_gf2(A, b)
        if comb is None:
            return None
        px = np.zeros(self.n, dtype=np.uint8)
        pz = np.zeros(self.n, dtype=np.uint8)
        pr = 0
        for i in np.nonzero(comb)[0]:
            px, pz, pr = _pauli_mult(px, pz, pr, self.X[i], self.Z[i], int(self.r[i]))
        return px, pz, pr


def _strings_to_tableau(strings, signs=None):
    n = len(strings[0])
    X = np.zeros((len(strings), n), dtype=np.uint8)
    Z = np.zeros((len(strings), n), dtype=np.uint8)
    for i, s in enumerate(strings):
        for j, ch in enumerate(s):
            X[i, j], Z[i, j] = _PAULI_XZ[ch]
    r = np.zeros(len(strings), dtype=np.uint8) if signs is None \
        else np.asarray(signs, dtype=np.uint8)
    return StabilizerTableau(X, Z, r)


def five_qubit_code_generators():
    """S1..S4 of HaPPY Eq. (A.1) and the logical X-bar, Z-bar of Eq. (A.3),
    as Pauli strings."""
    return ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"], ("XXXXX", "ZZZZZ")


def perfect_tensor_generators():
    """The six generators of the [[6,0,4]] perfect-tensor state, HaPPY
    Eq. (A.5): S_j (x) I for j = 1..4, X^{x6}, Z^{x6}."""
    S, (Xb, Zb) = five_qubit_code_generators()
    return [s + "I" for s in S] + [Xb + "X", Zb + "Z"]


# ---------------------------------------------------------------------------
# hyperbolic tilings {p,4} built layer by layer (HaPPY App. C.1)
# ---------------------------------------------------------------------------

MinCut = namedtuple("MinCut", ["size", "legs", "wedge"])
RTResult = namedtuple("RTResult", ["S", "cut", "greedy_A", "greedy_Ac",
                                    "residual_empty", "rt_holds"])


def layer_counts(n_layers, p=5):
    """(f_n, g_n) for n = 1..n_layers: tensors with one / two legs to the
    previous layer (HaPPY Eq. C.1 for p = 5; the same recursion with
    (p-4, 1; 1, 1)... generalized as derived from the tiling)."""
    out = []
    f, g = p, 0
    for n in range(1, n_layers + 1):
        out.append((f, g))
        # every consecutive pair of layer-n tensors shares one 2-parent
        # child; each tensor with `out` free legs has out - 2 private
        # single-parent children (out = p - 1 for f-type, p - 2 for g-type)
        f, g = (p - 3) * f + (p - 4) * g, f + g
    return out


def _build_tiling(n_layers, p=5, bulk_leg=True):
    """Layered {p,4} tiling. Returns dict with tensors (list of leg lists),
    edges, boundary legs (cyclic order) and per-tensor layer."""
    n_legs = p + (1 if bulk_leg else 0)
    tensors = []      # tensors[t] = list of n_legs entries, filled below
    layer_of = []
    edges = []        # (t1, leg1, t2, leg2)

    def new_tensor(layer):
        tensors.append([None] * n_legs)
        layer_of.append(layer)
        return len(tensors) - 1

    def connect(t1, l1, t2, l2):
        eid = len(edges)
        edges.append((t1, l1, t2, l2))
        tensors[t1][l1] = ("e", eid)
        tensors[t2][l2] = ("e", eid)

    center = new_tensor(0)
    # current layer: list of (tensor, [free planar leg indices in cyclic order])
    current = [(center, list(range(p)))]
    for layer in range(1, n_layers + 1):
        M = len(current)
        if layer == 1:
            # all children of the center are single-parent
            nxt = []
            for leg in range(p):
                c = new_tensor(layer)
                connect(center, leg, c, 0)
                nxt.append((c, list(range(1, p))))
            current = nxt
            continue
        # shared children between cyclically consecutive tensors
        shared = {}
        for i in range(M):
            j = (i + 1) % M
            c = new_tensor(layer)
            shared[(i, j)] = c
        nxt = []
        for i in range(M):
            t, free = current[i]
            prev_i = (i - 1) % M
            nxt_i = (i + 1) % M
            c_prev = shared[(prev_i, i)]
            c_next = shared[(i, nxt_i)]
            # first free leg -> shared child with previous neighbour (its leg 1)
            connect(t, free[0], c_prev, 1)
            if i == 0:
                nxt.append((c_prev, list(range(2, p))))
            # middle legs -> private children
            for leg in free[1:-1]:
                c = new_tensor(layer)
                connect(t, leg, c, 0)
                nxt.append((c, list(range(1, p))))
            # last free leg -> shared child with next neighbour (its leg 0)
            connect(t, free[-1], c_next, 0)
            if i < M - 1:
                nxt.append((c_next, list(range(2, p))))
        current = nxt
    boundary = []
    for t, free in current:
        for leg in free:
            q = len(boundary)
            boundary.append((t, leg))
            tensors[t][leg] = ("b", q)
    for t in range(len(tensors)):
        if bulk_leg:
            tensors[t][p] = ("bulk", t)
        assert all(x is not None for x in tensors[t]), "unassigned leg"
    return {"tensors": tensors, "edges": edges, "boundary": boundary,
            "layer": layer_of, "p": p, "bulk_leg": bulk_leg}


@dataclass
class HaPPYCode:
    """A contracted holographic code/state.

    tiling   : the layered network (see _build_tiling)
    choi     : StabilizerTableau on qubits [boundary 0..N_b-1, bulk 0..N_bulk-1]
    n_boundary, n_bulk
    """

    tiling: dict
    choi: StabilizerTableau
    n_boundary: int
    n_bulk: int
    meta: dict = field(default_factory=dict)

    @property
    def n_tensors(self):
        return len(self.tiling["tensors"])

    def bulk_qubit(self, tensor):
        return self.n_boundary + int(tensor)

    def adjacency(self):
        """For each tensor: list of (leg, kind, target) with kind in
        {'e': neighbour tensor, 'b': boundary qubit, 'bulk': itself}."""
        edges = self.tiling["edges"]
        out = []
        for t, legs in enumerate(self.tiling["tensors"]):
            row = []
            for leg, (kind, ref) in enumerate(legs):
                if kind == "e":
                    t1, l1, t2, l2 = edges[ref]
                    other = t2 if t1 == t else t1
                    row.append((leg, "e", other, ref))
                elif kind == "b":
                    row.append((leg, "b", ref, None))
                else:
                    row.append((leg, "bulk", t, None))
            out.append(row)
        return out


def happy_code(n_layers, tiling="pentagon"):
    """Build the holographic pentagon code (tiling='pentagon', HaPPY Fig. 4b)
    or hexagon state (tiling='hexagon', Fig. 4a) with ``n_layers`` layers
    around the central tensor, contracted in the stabilizer formalism."""
    if tiling == "pentagon":
        p, bulk = 5, True
    elif tiling == "hexagon":
        p, bulk = 6, False
    else:
        raise ValueError("tiling must be 'pentagon' or 'hexagon'")
    til = _build_tiling(n_layers, p=p, bulk_leg=bulk)
    n_t = len(til["tensors"])
    n_legs = p + (1 if bulk else 0)
    gens = perfect_tensor_generators()
    # global qubit index: tensor t, leg l -> 6 t + l (before contraction)
    N = n_legs * n_t
    X = np.zeros((N, N), dtype=np.uint8)
    Z = np.zeros((N, N), dtype=np.uint8)
    for t in range(n_t):
        local = _strings_to_tableau(gens)
        sl = slice(n_legs * t, n_legs * (t + 1))
        X[sl, sl] = local.X
        Z[sl, sl] = local.Z
    T = StabilizerTableau(X, Z, np.zeros(N, dtype=np.uint8))
    # contract internal edges; track the surviving qubit labels
    labels = [(t, l) for t in range(n_t) for l in range(n_legs)]
    for (t1, l1, t2, l2) in til["edges"]:
        a = labels.index((t1, l1))
        b = labels.index((t2, l2))
        T = T.contract(a, b)
        labels = [lab for k, lab in enumerate(labels) if k not in (a, b)]
    # reorder: boundary qubits in cyclic order, then bulk qubits by tensor
    order = [labels.index(bl) for bl in til["boundary"]]
    n_boundary = len(order)
    if bulk:
        order += [labels.index((t, p)) for t in range(n_t)]
    T = StabilizerTableau(T.X[:, order], T.Z[:, order], T.r)
    return HaPPYCode(til, T, n_boundary, n_t if bulk else 0,
                     {"n_layers": n_layers, "tiling": tiling,
                      "layer_counts": layer_counts(n_layers, p)})


# ---------------------------------------------------------------------------
# greedy algorithm, min cut, entropies, reconstruction
# ---------------------------------------------------------------------------


def greedy_wedge(code, region):
    """HaPPY's greedy algorithm (Sec. 4.2) from boundary region A: absorb a
    tensor whenever at least half of its legs lie in the current region.
    Returns MinCut(size, legs, wedge): the greedy geodesic gamma*_A as the
    list of legs ('e', edge) / ('b', boundary qubit) crossing from the
    absorbed tensors, its size (bulk legs excluded), and the causal wedge
    C[A] = set of absorbed tensors (Def. 5)."""
    region = set(int(q) for q in region)
    adj = code.adjacency()
    n_legs = len(code.tiling["tensors"][0])
    half = (n_legs + 1) // 2
    absorbed = set()
    changed = True
    while changed:
        changed = False
        for t in range(code.n_tensors):
            if t in absorbed:
                continue
            k = 0
            for (leg, kind, tgt, ref) in adj[t]:
                if kind == "b" and tgt in region:
                    k += 1
                elif kind == "e" and tgt in absorbed:
                    k += 1
            if k >= half:
                absorbed.add(t)
                changed = True
    legs = []
    for t in absorbed:
        for (leg, kind, tgt, ref) in adj[t]:
            if kind == "e" and tgt not in absorbed:
                legs.append(("e", ref))
            elif kind == "b" and tgt not in region:
                legs.append(("b", tgt))
    # a boundary leg of A on an unabsorbed tensor is also part of the cut
    for q in region:
        t, leg = code.tiling["boundary"][q]
        if t not in absorbed:
            legs.append(("b", q))
    legs = sorted(set(legs))
    return MinCut(len(legs), legs, frozenset(absorbed))


def min_cut(code, region):
    """Minimal cut separating boundary region A from its complement, by
    max-flow/min-cut with unit capacity per planar leg (contracted legs and
    boundary legs alike; bulk legs carry no capacity). Returns MinCut."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    region = set(int(q) for q in region)
    n_t = code.n_tensors
    src, snk = n_t, n_t + 1
    rows, cols, caps = [], [], []
    for (t1, l1, t2, l2) in code.tiling["edges"]:
        rows += [t1, t2]; cols += [t2, t1]; caps += [1, 1]
    for q, (t, leg) in enumerate(code.tiling["boundary"]):
        if q in region:
            rows.append(src); cols.append(t); caps.append(1)
        else:
            rows.append(t); cols.append(snk); caps.append(1)
    n = n_t + 2
    G = csr_matrix((np.array(caps, dtype=np.int32), (rows, cols)), shape=(n, n))
    G.sum_duplicates()
    res = maximum_flow(G, src, snk)
    flow = res.flow if hasattr(res, "flow") else res.residual
    resid = (G - flow).tocsr()
    reach = np.zeros(n, dtype=bool)
    reach[src] = True
    stack = [src]
    while stack:
        v = stack.pop()
        row = resid.getrow(v)
        for j, c in zip(row.indices, row.data):
            if c > 0 and not reach[j]:
                reach[j] = True
                stack.append(j)
    P = frozenset(t for t in range(n_t) if reach[t])
    legs = []
    for eid, (t1, l1, t2, l2) in enumerate(code.tiling["edges"]):
        if reach[t1] != reach[t2]:
            legs.append(("e", eid))
    for q, (t, leg) in enumerate(code.tiling["boundary"]):
        if (q in region) != reach[t]:
            legs.append(("b", q))
    legs = sorted(set(legs))
    assert len(legs) == int(res.flow_value), (len(legs), res.flow_value)
    return MinCut(len(legs), legs, P)


def _state_tableau(code, bulk_state):
    """Choi tableau with the bulk legs projected on a product state
    ('zero': |0>, 'plus': |+>) -- a holographic state; or the Choi state
    itself (bulk_state=None)."""
    T = code.choi
    if bulk_state is None or code.n_bulk == 0:
        return T
    pauli = {"zero": "Z", "plus": "X"}[bulk_state]
    for b in range(code.n_bulk):
        T = T.project_single(code.n_boundary + b, pauli, 0)
    return T


def boundary_entropy(code, region, bulk_state="zero"):
    """Entropy (bits) of boundary region A in the code state with the bulk
    in the given product state (or of the Choi state if None)."""
    T = _state_tableau(code, bulk_state)
    return T.entropy_bits(sorted(set(int(q) for q in region)))


def rt_entropy(code, region, bulk_state="zero"):
    """Ryu-Takayanagi check for boundary region A: exact stabilizer entropy
    (bits) against the minimal cut (bond units, log 2 each), with the
    greedy geodesics from A and A^c (HaPPY Theorems 2-3).

    Returns RTResult(S, cut, greedy_A, greedy_Ac, residual_empty, rt_holds)
    with cut a MinCut; residual_empty is True when the two greedy geodesics
    coincide (no bipartite residual region), in which case HaPPY's argument
    gives S = |gamma*_A| exactly; rt_holds = (S == cut.size).
    """
    region = sorted(set(int(q) for q in region))
    S = boundary_entropy(code, region, bulk_state)
    cut = min_cut(code, region)
    gA = greedy_wedge(code, region)
    comp = [q for q in range(code.n_boundary) if q not in set(region)]
    gAc = greedy_wedge(code, comp)
    residual_empty = set(gA.legs) == set(gAc.legs)
    return RTResult(S, cut, gA, gAc, residual_empty, S == cut.size)


def reconstructible(code, tensor, region, pauli="X"):
    """Exact test: is the logical Pauli on bulk tensor ``tensor``
    reconstructible on boundary region A?  Returns (boundary Pauli string
    on all boundary qubits, sign) or None. The boundary operator O satisfies
    O V = V L with L the bulk Pauli (HaPPY Sec. 5.3-5.4 in stabilizer
    language)."""
    if code.n_bulk == 0:
        raise ValueError("a holographic state has no bulk legs")
    region = set(int(q) for q in region)
    n_b = code.n_boundary
    fixed = [q for q in range(n_b) if q not in region] + \
            [n_b + b for b in range(code.n_bulk)]
    tx = np.zeros(len(fixed), dtype=np.uint8)
    tz = np.zeros(len(fixed), dtype=np.uint8)
    pos = fixed.index(n_b + int(tensor))
    tx[pos], tz[pos] = _PAULI_XZ[pauli]
    sol = code.choi.element_with(fixed, tx, tz)
    if sol is None:
        return None
    px, pz, pr = sol
    sign = -1 if pr else 1
    if pauli == "Y":       # Y^T = -Y
        sign = -sign
    s = "".join(_XZ_PAULI[(int(px[q]), int(pz[q]))] for q in range(n_b))
    return s, sign


def bulk_reconstruction(code, bulk_op, region):
    """Greedy (causal-wedge) reconstruction, HaPPY Def. 5 / Theorem 5:
    ``bulk_op`` = (tensor index, 'X'|'Y'|'Z'). Returns the boundary Pauli
    (string over all boundary qubits, identity outside A) and its sign when
    the tensor lies in the greedy wedge of A; None otherwise."""
    tensor, pauli = bulk_op
    wedge = greedy_wedge(code, region).wedge
    if int(tensor) not in wedge:
        return None
    out = reconstructible(code, tensor, region, pauli)
    if out is None:  # cannot happen by Theorem 5; surface loudly if it does
        raise RuntimeError("tensor in greedy wedge but not reconstructible")
    return out
