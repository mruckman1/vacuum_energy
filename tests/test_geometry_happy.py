"""HaPPY holographic codes in the stabilizer formalism -- Layer 4 anchors.

Pastawski-Yoshida-Harlow-Preskill, JHEP 06 (2015) 149 (arXiv:1503.06237):

1. Building block: the [[5,1,3]] code (App. A.1). Any two boundary qubits
   of one pentagon are maximally mixed, the logical algebra is
   reconstructible from any three (weight-3 logical X-bar ~ -Z X Z I I,
   App. A.1) and from no pair.
2. Tiling combinatorics (App. C.1): (f_n, g_n) = (5,0), (10,5), (25,15);
   N_boundary = 4 f_n + 3 g_n = 20, 55, 145; N_bulk = 6, 21, 61 -- rate
   toward 1/sqrt(5) (Eq. C.6).
3. Ryu-Takayanagi (Sec. 4.1-4.2, Theorems 2-3): S_A <= |gamma_A| always;
   S_A >= |gamma*_A intersect gamma*_{A^c}| (Eq. 4.8); S_A = |gamma*_A|
   exactly whenever the greedy geodesics from A and A^c coincide (no
   bipartite residual region). For the hexagon STATE (all legs contracted,
   Theorem 2) RT holds for every connected region; for the pentagon code
   with a product bulk state the exceptions are exactly the regions with a
   residual region (Fig. 7, third example), and the measured fraction is
   recorded in the assertion message.
4. Reconstruction (Def. 5, Theorem 5): every bulk operator in the greedy
   causal wedge of a connected region is reconstructible on it; on the
   1- and 2-layer codes the exact (GF(2)) reconstructibility of the central
   logical algebra coincides with greedy-wedge membership on every
   connected region tested; the connected fraction guaranteeing central
   reconstruction is compared with Eq. (5.1), f_c = (5 + sqrt 5)/10.
"""

import itertools

import numpy as np
import pytest

from vacuum.geometry import (
    boundary_entropy,
    bulk_reconstruction,
    greedy_wedge,
    happy_code,
    layer_counts,
    min_cut,
    reconstructible,
    rt_entropy,
)


def _arc(n, start, length):
    return [(start + k) % n for k in range(length)]


@pytest.fixture(scope="module")
def codes():
    return {n: happy_code(n) for n in (0, 1, 2)}


@pytest.fixture(scope="module")
def hexagon():
    return {1: happy_code(1, tiling="hexagon"), 2: happy_code(2, tiling="hexagon")}


# ---------------------------------------------------------------------------
# 1. the [[5,1,3]] pentagon
# ---------------------------------------------------------------------------


def test_single_pentagon_is_perfect(codes):
    c = codes[0]
    assert c.n_boundary == 5 and c.n_bulk == 1
    for pair in itertools.combinations(range(5), 2):
        assert boundary_entropy(c, pair) == 2, f"pair {pair} not maximally mixed"
    for triple in itertools.combinations(range(5), 3):
        assert boundary_entropy(c, triple) == 2
    # Choi state: bulk qubit maximally entangled with the boundary
    assert c.choi.entropy_bits([5]) == 1


def test_single_pentagon_reconstruction(codes):
    c = codes[0]
    for triple in itertools.combinations(range(5), 3):
        for P in "XYZ":
            assert reconstructible(c, 0, triple, P) is not None, (triple, P)
    for pair in itertools.combinations(range(5), 2):
        assert reconstructible(c, 0, pair, "X") is None
        assert reconstructible(c, 0, pair, "Z") is None
    # HaPPY App. A.1: X-bar ~ -Z X Z I I on the codespace
    s, sign = reconstructible(c, 0, [0, 1, 2], "X")
    assert (s, sign) == ("ZXZII", -1), f"got {s}, {sign}"
    assert reconstructible(c, 0, range(5), "X") == ("XXXXX", 1)
    assert reconstructible(c, 0, range(5), "Z") == ("ZZZZZ", 1)
    # greedy: absorbed iff >= 3 of 6 legs in the region
    assert 0 in greedy_wedge(c, [0, 1, 2]).wedge
    assert 0 not in greedy_wedge(c, [0, 1]).wedge
    assert bulk_reconstruction(c, (0, "X"), [0, 1]) is None
    assert bulk_reconstruction(c, (0, "X"), [1, 2, 3]) is not None


# ---------------------------------------------------------------------------
# 2. tiling combinatorics (App. C.1)
# ---------------------------------------------------------------------------


def test_layer_counts_and_sizes(codes):
    assert layer_counts(3) == [(5, 0), (10, 5), (25, 15)]
    for n, (nb, nbulk) in {0: (5, 1), 1: (20, 6), 2: (55, 21)}.items():
        assert codes[n].n_boundary == nb, (n, codes[n].n_boundary)
        assert codes[n].n_bulk == nbulk
        assert codes[n].choi.n_gen == nb + nbulk
    f3, g3 = layer_counts(3)[-1]
    assert 4 * f3 + 3 * g3 == 145 and 1 + 5 + 15 + 40 == 61
    rates = [6 / 20, 21 / 55, 61 / 145]
    assert abs(rates[-1] - 1 / np.sqrt(5)) < abs(rates[0] - 1 / np.sqrt(5)), \
        f"rate sequence {rates} does not approach 1/sqrt5 = {1/np.sqrt(5):.3f}"
    # every internal edge joins consecutive layers; no same-layer edges
    for n in (1, 2):
        lay = codes[n].tiling["layer"]
        for (t1, l1, t2, l2) in codes[n].tiling["edges"]:
            assert abs(lay[t1] - lay[t2]) == 1


# ---------------------------------------------------------------------------
# 3. Ryu-Takayanagi
# ---------------------------------------------------------------------------


def _rt_survey(code, regions):
    holds, resid_empty, lower_ok, upper_ok = [], [], [], []
    for A in regions:
        r = rt_entropy(code, A)
        holds.append(r.rt_holds)
        resid_empty.append(r.residual_empty)
        common = set(r.greedy_A.legs) & set(r.greedy_Ac.legs)
        lower_ok.append(r.S >= len(common))
        upper_ok.append(r.S <= r.cut.size)
        if r.residual_empty:
            assert r.S == r.greedy_A.size == r.cut.size, (A, r)
    return np.array(holds), np.array(resid_empty), all(lower_ok), all(upper_ok)


def test_rt_pentagon_code_one_layer(codes):
    c = codes[1]
    n = c.n_boundary
    regions = [_arc(n, s, m) for s in range(n) for m in range(1, n)]
    holds, resid, lower_ok, upper_ok = _rt_survey(c, regions)
    assert upper_ok, "S_A > minimal cut somewhere"
    assert lower_ok, "Theorem 3 lower bound violated"
    # no residual region => RT exact (HaPPY Sec. 4.2); the converse need not hold
    assert holds[resid].all(), "RT failed on a region without residual region"
    frac = holds.mean()
    assert frac > 0.85, f"RT exact on {frac:.3f} of connected regions (1 layer); " \
        f"residual-free fraction {resid.mean():.3f}"


def test_rt_pentagon_code_two_layers_sampled(codes):
    c = codes[2]
    n = c.n_boundary
    regions = [_arc(n, s, m) for s in range(0, n, 5) for m in range(1, n, 3)]
    holds, resid, lower_ok, upper_ok = _rt_survey(c, regions)
    assert upper_ok and lower_ok
    assert holds[resid].all(), "RT failed on a region without residual region"
    assert holds.mean() > 0.7, f"RT exact on {holds.mean():.3f} of sampled regions " \
        f"(2 layers); residual-free fraction {resid.mean():.3f}"


def test_rt_hexagon_state_theorem2(hexagon):
    # Theorem 2: all legs contracted, non-positive curvature, connected A
    c = hexagon[1]
    assert c.n_boundary == 30 and c.n_bulk == 0
    n = c.n_boundary
    regions = [_arc(n, s, m) for s in range(n) for m in range(1, n)]
    holds, resid, lower_ok, upper_ok = _rt_survey(c, regions)
    assert holds.all(), f"Theorem 2 violated on {(~holds).sum()} of {len(regions)} regions"
    c2 = hexagon[2]
    assert c2.n_boundary == 114
    n = c2.n_boundary
    regions = [_arc(n, s, m) for s in range(0, n, 7) for m in range(1, n, 4)]
    holds, resid, lower_ok, upper_ok = _rt_survey(c2, regions)
    assert holds.all(), f"Theorem 2 violated (2 layers) on {(~holds).sum()} of {len(regions)}"


def test_rt_disconnected_region_lower_bound(codes):
    c = codes[1]
    n = c.n_boundary
    A = _arc(n, 0, 6) + _arc(n, 10, 6)
    r = rt_entropy(c, A)
    common = set(r.greedy_A.legs) & set(r.greedy_Ac.legs)
    assert len(common) <= r.S <= r.cut.size, r


# ---------------------------------------------------------------------------
# 4. reconstruction and the causal wedge
# ---------------------------------------------------------------------------


def test_theorem5_greedy_wedge_reconstructible(codes):
    c = codes[1]
    n = c.n_boundary
    checked = 0
    for s in range(0, n, 2):
        for m in range(1, n, 2):
            A = _arc(n, s, m)
            wedge = greedy_wedge(c, A).wedge
            for t in wedge:
                for P in "XZ":
                    out = bulk_reconstruction(c, (t, P), A)
                    assert out is not None
                    op, sign = out
                    assert all(op[q] == "I" for q in range(n) if q not in set(A)), \
                        "reconstruction leaks outside A"
                    checked += 1
    assert checked > 50


def _center_survey(code, regions):
    exact, greedy = [], []
    for A in regions:
        e = (reconstructible(code, 0, A, "X") is not None
             and reconstructible(code, 0, A, "Z") is not None)
        g = 0 in greedy_wedge(code, A).wedge
        exact.append(e)
        greedy.append(g)
    return np.array(exact), np.array(greedy)


def test_center_reconstruction_iff_greedy_one_layer(codes):
    c = codes[1]
    n = c.n_boundary
    regions = [_arc(n, s, m) for s in range(n) for m in range(1, n + 1)]
    exact, greedy = _center_survey(c, regions)
    assert np.array_equal(exact, greedy), \
        f"{np.sum(exact != greedy)} regions where exact != greedy (1 layer)"
    sizes = np.array([len(A) for A in regions])
    best = sizes[exact].min()
    guaranteed = min(m for m in range(1, n + 1) if exact[sizes == m].all())
    assert best == 10 and guaranteed == 13, (best, guaranteed)


def test_center_reconstruction_two_layers(codes):
    c = codes[2]
    n = c.n_boundary
    regions = [_arc(n, s, m) for s in range(0, n, 3) for m in range(20, n + 1, 2)]
    exact, greedy = _center_survey(c, regions)
    mism = int(np.sum(exact != greedy))
    assert mism == 0, f"{mism} of {len(regions)} regions where exact != greedy (2 layers)"
    sizes = np.array([len(A) for A in regions])
    fc = (5 + np.sqrt(5)) / 10   # HaPPY Eq. (5.1)
    big = sizes / n > fc
    assert exact[big].all(), "Eq. (5.1): every connected region with f_A > 0.724 reconstructs the centre"
    best = sizes[exact].min() / n
    assert best < fc, f"best-case fraction {best:.3f} should undercut f_c = {fc:.3f} (Eq. C.9 ~ 0.524)"


def test_min_cut_matches_flow_and_is_a_cut(codes):
    c = codes[2]
    n = c.n_boundary
    for s in (0, 7, 23):
        A = _arc(n, s, 17)
        mc = min_cut(c, A)
        assert mc.size == len(mc.legs)
        assert boundary_entropy(c, A) <= mc.size
