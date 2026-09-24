"""Layer-4 validation anchors for vacuum.modular — the modular-Hamiltonian laboratory.

Every tolerance below was calibrated on the measured value quoted in its
assertion message BEFORE being written down, and none may be weakened.

Anchor 1 — Bisognano-Wichmann at criticality (single interval, half-filled chain).
  The nearest-neighbour hopping of h is the exact lattice form of Eisler-Peschel,
  J. Phys. A 50, 284003 (2017), Eq. (27), h_{i,i+1} = pi x(1-x) 3F2(1/4,1/2,3/4;1,2;[4x(1-x)]^2),
  to 0.2/L (their scaling limit carries 1/L corrections; measured 0.14/L), including
  the 7.6 % enhancement at the centre; the resummed profile (their Eq. (22), Eisler-
  Tonni-Peschel 2022 Eq. (31)) is the conformal parabola 2 pi x(L-x)/L to 1.5/L^2;
  near the boundary beta(x) -> 2 pi dist with exactly the 1 - x/L parabola correction;
  eps_max matches their Eq. (54) to 0.05; the third-neighbour hopping converges to
  their Eq. (26); float64 fails beyond ~20 sites exactly as they state (Sec. 6) and the
  library flags it; a ring reproduces the chord (circle) form of the profile.
Anchor 2 — Casini-Huerta two intervals, CQG 26, 185005 (2009), Eqs. (33), (44)-(48), (53), (59),
  after the lattice dictionary D1-D6 of vacuum/modular/casini_huerta.py (which coincides
  with Eisler-Tonni-Peschel 2022 Eqs. (31)-(32)): local weight 2 pi/z', nonlocal weight
  2 pi/((x - x_c) z'(x_c)) between conjugate points, both with the 1/L^2 finite-size
  trend shown over scales 1, 2, 4; smeared bilinears; the mutual information cross
  ratio; and the modular flow, whose "teleportation" weight is sin^2 theta(tau) of Eq. (59).
Anchor 3 — the fermionic entanglement first law dS = Tr(h dC_A) to 1e-8 by independent routes.
Anchor 4 — the modular flow preserves the spectrum of C_A to 1e-12.
Anchor 5 — n >= 3 intervals at growing L (fixed gap/L): the deviation from the
  n-interval Casini-Huerta formula falls as A/L^2 and extrapolates to zero.
Anchor 6 — gapped chains at xi = 10, 20: the decay length of the inter-interval
  modular couplings tends to the correlation length itself, xi_h/xi -> 1.
Anchor 7 — the gapped Bisognano-Wichmann slope in closed form. For exactly this chain
  Eisler, J. Stat. Mech. (2025) 013101, arXiv:2410.16433, Sec. V, Eqs. (64)-(72), derives
  H = 4 kappa K(kappa') T with kappa = 1/sqrt(1 + mu^2) and the signed ladder
  eps_l = 2 pi K(kappa')/K(kappa) (2l -/+ 1/2); in our labels 2 pi s(m) = 4 kappa K(kappa'),
  kappa = sech(1/xi), |eps_l| = (2l + 1) pi K(kappa')/K(kappa) (dictionary in
  vacuum/modular/ctm.py). The lattice reproduces the prefactor, the ladder and its per-edge
  sign pattern to machine precision once the segment is long compared with xi; the
  half-infinite form applies near an edge, not in the middle (the triangle of Eisler et al.
  2020 there). The corner-transfer-matrix structure (Peschel-Kaulke-Legeza 1999; Eisler et al.
  2020) is the same for the transverse Ising and dimerized chains.

All entropies in nats; hopping 1/2, k_F = pi/2 (docs/API.md).
"""

from __future__ import annotations

import warnings

import mpmath as mp
import numpy as np
import pytest
from scipy.linalg import expm

from vacuum.core import (
    entanglement_hamiltonian as bosonic_eh,
    ground_state_cov,
    harmonic_chain_K,
    modular_energy,
    reduce,
)
from vacuum.fermions import block_entropy, tight_binding_C
from vacuum.modular import (
    ModularPrecisionWarning,
    bilocal_weight,
    bw_profile,
    casini_huerta_exact,
    ch_beta,
    ch_bilinear,
    ch_conjugate_points,
    ch_mixing_angle,
    ch_mobius_conjugate,
    ch_mutual_information,
    ch_nonlocal_weight,
    ch_single_interval_flow,
    ch_trajectory,
    ch_z,
    ch_zprime,
    chiral_bilinear,
    correlation_length,
    ctm_level_spacing,
    ctm_slope,
    ctm_slope_expansion,
    eisler_peschel_eq26,
    eisler_peschel_eq27,
    eisler_peschel_eq54,
    first_law_check,
    flow_wavefunction,
    from_chiral_gauge,
    infinite_chain_C,
    interval_modular,
    intervals_from_parts,
    lattice_boost,
    locality_score,
    massive_chain_C,
    modular_energy_profile,
    modular_flow,
    modular_flow_unitary,
    modular_hamiltonian_auto,
    multi_interval_scan,
    nonlocal_decay,
    part_centroids,
    part_weights,
    region_modular,
    ring_C,
    staggered_mass_single_particle,
    to_float,
    triangle_h,
    two_interval_modular,
    wavepacket,
)

# --------------------------------------------------------------------------- #
# fixtures (module-scoped: the mp diagonalizations are the expensive part)
# --------------------------------------------------------------------------- #
SINGLE_L = (20, 40, 60)
TWO_SCALES = (1, 2, 4)


def _crit(sites, dps):
    return infinite_chain_C(sites, dps=dps)


@pytest.fixture(scope="module")
def single_h():
    out = {}
    for L in SINGLE_L:
        h, info, dps = modular_hamiltonian_auto(_crit, list(range(L)), 0.0, "auto")
        assert info["sufficient"], f"L={L}: precision insufficient at dps={dps}: {info}"
        out[L] = (h, info)
    return out


def _two_config(s):
    L1 = L2 = 10 * s
    d = 5 * s
    return list(range(L1)), list(range(L1 + d, L1 + d + L2))


@pytest.fixture(scope="module")
def two_h():
    out = {}
    for s in TWO_SCALES:
        A1, A2 = _two_config(s)
        h, info, _ = modular_hamiltonian_auto(_crit, A1 + A2, 0.0, "auto")
        assert info["sufficient"]
        out[s] = (A1, A2, h)
    A1, A2 = list(range(20)), list(range(28, 38))  # unequal: 20 + 10 sites, gap 8
    h, info, _ = modular_hamiltonian_auto(_crit, A1 + A2, 0.0, "auto")
    assert info["sufficient"]
    out["unequal"] = (A1, A2, h)
    return out


@pytest.fixture(scope="module")
def ring_h():
    N, L = 200, 40
    h, info, _ = modular_hamiltonian_auto(lambda s, d: ring_C(N, s, dps=d), list(range(L)), 0.0, "auto")
    assert info["sufficient"]
    return N, L, h


GAPPED_CASES = ((1.0, 40), (0.5, 80))  # L - 16 >= 20 xi: the far edge is out of the near-edge window


@pytest.fixture(scope="module")
def gapped_h():
    """h of a long segment of the staggered-mass chain, shared by the closed-form anchors (~12 s)."""
    out = {}
    for m, L in GAPPED_CASES:
        h, info, dps = modular_hamiltonian_auto(
            lambda s_, d_, mm=m: massive_chain_C(s_, mm, dps=d_), list(range(L)), m, "auto")
        assert info["sufficient"], f"m={m}, L={L}: precision insufficient at dps={dps}"
        out[(m, L)] = h
    return out


def _resummed_dev(h, A, iv):
    x, b = bw_profile(h, mode="resummed", offset=A[0])
    return x, b / ch_beta(x, iv) - 1.0


# =========================================================================== #
# Anchor 1: Bisognano-Wichmann at criticality
# =========================================================================== #
def test_bw_nearest_neighbour_matches_eisler_peschel_eq27(single_h):
    devs = {}
    for L in SINGLE_L:
        h, _ = single_h[L]
        x, beta = bw_profile(h, mode="nn")
        ep27 = 2.0 * L * np.array([eisler_peschel_eq27(xx / L) for xx in x])  # beta = -2 h_{i,i+1}
        devs[L] = float(np.max(np.abs(beta / ep27 - 1.0)))
        assert devs[L] < 0.2 / L, (
            f"L={L}: nearest-neighbour profile deviates from Eisler-Peschel Eq. (27) by "
            f"{devs[L]:.3e} (tolerance 0.2/L = {0.2 / L:.3e}; measured 6.9e-3, 2.6e-3, 1.5e-3 at L = 20, 40, 60)"
        )
    assert devs[20] > devs[40] > devs[60], f"finite-size trend not decreasing: {devs}"
    # the centre enhancement 3F2(...; 1) = 1.076 quoted below their Eq. (27)
    h, _ = single_h[60]
    x, beta = bw_profile(h, mode="nn")
    par = 2 * np.pi * x * (60 - x) / 60
    centre = beta[29] / par[29]
    assert abs(centre - 1.0760) < 5e-3, f"centre enhancement {centre:.4f} vs Eisler-Peschel 1.076"


def test_bw_linear_near_boundary(single_h):
    """beta(x) -> 2 pi dist(x, edge): the parabola 2 pi x (1 - x/L) is exact to the 3F2 correction."""
    dev1 = {}
    for L in SINGLE_L:
        h, _ = single_h[L]
        x, beta = bw_profile(h, mode="nn")
        dev1[L] = abs(beta[0] / (2 * np.pi) - 1.0)
        assert dev1[L] < 1.1 / L + 1e-3, (
            f"L={L}: first bond beta/(2 pi) - 1 = {beta[0] / (2 * np.pi) - 1:+.4e}; the exact "
            f"parabola gives -1/L = {-1 / L:.4e} (measured -0.0481, -0.0245, -0.0164)"
        )
        for k in (0, 1):  # bonds x = 1, 2: the residual after the exact parabola IS the 3F2 factor of Eq. (27)
            xx = x[k]
            r = beta[k] / (2 * np.pi * xx * (1 - xx / L))
            f32 = eisler_peschel_eq27(xx / L) / (np.pi * (xx / L) * (1 - xx / L))
            assert abs(r - f32) < 1e-3, (
                f"L={L}, bond x={xx}: beta/(2 pi x (1 - x/L)) = {r:.5f} vs 3F2 correction {f32:.5f} "
                f"(diff {r - f32:+.1e}; measured <= 1.2e-4)"
            )
    ratio = dev1[20] / dev1[40]
    assert 1.8 < ratio < 2.2, f"edge deviation should halve from L=20 to L=40: ratio {ratio:.3f} (measured 1.967)"


def test_bw_resummed_profile_is_conformal_parabola(single_h):
    """Resumming the odd-range hoppings (Eisler-Peschel Eq. (22)) recovers 2 pi x (L-x)/L to 1.5/L^2."""
    for L in SINGLE_L:
        h, _ = single_h[L]
        x, beta = bw_profile(h, mode="resummed")
        dev = float(np.max(np.abs(beta / (2 * np.pi * x * (L - x) / L) - 1.0)))
        assert dev < 1.5 / L**2, (
            f"L={L}: resummed profile vs conformal parabola max rel dev {dev:.3e} "
            f"(tolerance 1.5/L^2 = {1.5 / L**2:.3e}; measured 2.0e-3, 5.6e-4 at L = 20, 40, ~0.9/L^2)"
        )


def test_eps_max_matches_eisler_peschel_eq54(single_h):
    for L in SINGLE_L:
        _, info = single_h[L]
        gap = info["eps_max"] - eisler_peschel_eq54(L)
        assert abs(gap) < 0.05, (
            f"L={L}: eps_max = {info['eps_max']:.4f} vs Eisler-Peschel Eq. (54) = {eisler_peschel_eq54(L):.4f} "
            f"(diff {gap:+.4f}; measured +0.040, +0.021, +0.016)"
        )


def test_third_neighbour_hopping_converges_to_eq26(single_h):
    """h_{i,i+3} peak vs Eisler-Peschel Eq. (26) with p = 1 (they need N ~ 100 for full convergence)."""
    peak = eisler_peschel_eq26(0.5, 1)
    dev = {L: abs(np.max(np.abs(np.diag(single_h[L][0], 3))) / L / peak - 1.0) for L in (40, 60)}
    assert dev[60] < 0.06, f"L=60: third-neighbour peak deviates {dev[60]:.3e} from Eq. (26) peak {peak:.5f} (measured 0.040)"
    assert dev[60] < dev[40], f"not converging: {dev}"


def test_particle_hole_structure(single_h):
    """At half filling h couples the two sublattices only: even-distance elements and the diagonal vanish."""
    for L in SINGLE_L:
        h, _ = single_h[L]
        even = max(np.max(np.abs(np.diag(h, 2 * p))) for p in range(0, L // 2))
        assert even < 1e-12, f"L={L}: even-distance elements {even:.2e}"


def test_float64_path_flags_and_fails_beyond_validity():
    """Eisler-Peschel 2017 Sec. 6: float64 is good to ~20 sites. Verify, and verify the loud flag."""
    N = 400
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        h64, info64 = interval_modular(ring_C(N, range(N)), range(40), return_info=True)
    assert any(issubclass(w.category, ModularPrecisionWarning) for w in rec), "float64 clipping at L=40 must warn"
    assert info64["n_clipped"] > 0 and not info64["sufficient"]
    hmp = interval_modular(ring_C(N, range(40), dps=65), range(40), dps=65)
    centre = abs(h64[19, 20] / hmp[19, 20] - 1.0)
    edge = abs(h64[0, 1] / hmp[0, 1] - 1.0)
    assert centre > 0.1, f"float64 h at L=40 should be wrong at O(1) in the centre: rel dev {centre:.3f} (measured 0.42)"
    assert edge < 1e-3, f"...but right at the edge (high modes live in the middle): rel dev {edge:.2e}"
    # L = 12 is inside the float64 window: both paths agree
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h12 = interval_modular(ring_C(N, range(N)), range(12))
    h12mp = interval_modular(ring_C(N, range(12), dps=40), range(12), dps=40)
    assert np.max(np.abs(h12 - h12mp)) < 1e-6, f"L=12 float64 vs mp: {np.max(np.abs(h12 - h12mp)):.2e}"


def test_ring_closed_form_matches_tight_binding():
    N = 200
    err = np.max(np.abs(ring_C(N, range(N)) - tight_binding_C(N, 0.5, "antiperiodic")))
    assert err < 1e-13, f"ring_C vs vacuum.fermions.tight_binding_C: {err:.2e}"
    # and the infinite chain is its N -> inf limit
    err2 = np.max(np.abs(ring_C(4000, range(20)) - infinite_chain_C(range(20))))
    assert err2 < 1e-6, f"ring N=4000 vs infinite chain: {err2:.2e}"


def test_massive_closed_form_matches_brute_force():
    N = 600
    for m, sites in ((0.5, list(range(6)) + list(range(9, 15))), (0.3, list(range(20)))):
        w, v = np.linalg.eigh(staggered_mass_single_particle(N, m))
        occ = v[:, : N // 2]
        Cb = occ @ occ.T
        shifted = [s + 300 for s in sites]
        Cbs = Cb[np.ix_(shifted, shifted)]
        e1 = np.max(np.abs(massive_chain_C(sites, m) - Cbs))
        e2 = np.max(np.abs(to_float(massive_chain_C(sites, m, dps=40)) - Cbs))
        assert max(e1, e2) < 1e-12, f"m={m}: closed form vs brute force: float64 {e1:.2e}, mp {e2:.2e}"


def test_ring_resummed_profile_is_chord_bw(ring_h):
    """On a ring the BW profile is the chord form 2N sin(pi x/N) sin(pi (L-x)/N)/sin(pi L/N) (dictionary D6)."""
    N, L, h = ring_h
    x, beta = bw_profile(h, mode="resummed")
    chord = 2 * N * np.sin(np.pi * x / N) * np.sin(np.pi * (L - x) / N) / np.sin(np.pi * L / N)
    line = 2 * np.pi * x * (L - x) / L
    dev = float(np.max(np.abs(beta / chord - 1.0)))
    assert dev < 1.5 / L**2, f"ring N={N}, L={L}: resummed vs chord BW {dev:.3e} (measured 4.6e-4)"
    assert np.max(np.abs(beta / line - 1.0)) > 2e-2, "the chord geometry should be resolved (line parabola off by 3.4e-2)"


# =========================================================================== #
# Anchor 2: Casini-Huerta two intervals
# =========================================================================== #
def test_conjugate_points_eq47_and_z_invariance(two_h):
    for s in TWO_SCALES:
        A1, A2, _ = two_h[s]
        iv = intervals_from_parts([A1, A2])
        x = np.arange(len(A1)) + 0.5
        roots = np.array([ch_conjugate_points(v, iv)[1] for v in x])
        mob = ch_mobius_conjugate(x, iv[0], iv[1])
        assert np.max(np.abs(roots - mob)) < 1e-9, f"scale {s}: roots vs CH Eq. (47): {np.max(np.abs(roots - mob)):.2e}"
        assert np.max(np.abs(ch_z(roots, iv) - ch_z(x, iv))) < 1e-10
        assert np.all(ch_zprime(x, iv) > 0) and np.all(ch_zprime(roots, iv) > 0)
    iv3 = [(0.0, 10.0), (14.0, 22.0), (30.0, 45.0)]
    cp = ch_conjugate_points(5.3, iv3)
    assert set(cp) == {1, 2} and all(abs(ch_z(v, iv3) - ch_z(5.3, iv3)) < 1e-10 for v in cp.values())


def test_local_part_resummed_matches_ch_beta(two_h):
    tol = {1: 4e-2, 2: 8e-3, 4: 2e-3}
    devs = {}
    for s in TWO_SCALES:
        A1, A2, h = two_h[s]
        iv = intervals_from_parts([A1, A2])
        n1 = len(A1)
        _, d1 = _resummed_dev(h[:n1, :n1], A1, iv)
        _, d2 = _resummed_dev(h[n1:, n1:], A2, iv)
        devs[s] = max(np.max(np.abs(d1)), np.max(np.abs(d2)))
        assert devs[s] < tol[s], (
            f"scale {s}: resummed local weight vs CH beta = 2 pi/z' max rel dev {devs[s]:.3e} "
            f"(tolerance {tol[s]}; measured 2.1e-2, 3.5e-3, 8.2e-4 at scales 1, 2, 4)"
        )
        if s == 4:
            assert abs(d1[n1 // 2 - 1]) < 1e-4, f"scale 4 centre bond: {d1[n1 // 2 - 1]:+.2e} (measured 3.5e-5)"
    assert devs[1] > devs[2] > devs[4], f"finite-size trend not decreasing: {devs}"


def test_nearest_neighbour_only_carries_lattice_correction(two_h):
    """The nn hopping alone matches CH at the edges but keeps a ~16 % lattice enhancement at the centre."""
    centre = {}
    for s in TWO_SCALES:
        A1, A2, h = two_h[s]
        iv = intervals_from_parts([A1, A2])
        n1 = len(A1)
        x, b = bw_profile(h[:n1, :n1], mode="nn", offset=A1[0])
        dev = b / ch_beta(x, iv) - 1.0
        edge = np.abs(dev[x <= 0.1 * n1])
        assert np.max(edge) < 1e-2, f"scale {s}: edge bonds nn vs CH {np.max(edge):.2e} (measured <= 4.7e-3)"
        centre[s] = float(dev[n1 // 2 - 1])
    assert 0.14 < centre[2] < 0.17 and abs(centre[4] - centre[2]) < 1e-2, (
        f"centre lattice correction should be a stable ~0.157: {centre} (measured 0.149, 0.157, 0.157)"
    )


def test_nonlocal_row_sums_match_ch_eq48(two_h):
    """Alternating row sums (Eisler-Tonni-Peschel Eq. (32)) -> 2 pi/((x - x_c) z'(x_c)) (CH Eq. (48))."""
    tol = {1: 0.15, 2: 2e-2, 4: 5e-3}
    devs = {}
    for s in TWO_SCALES:
        A1, A2, h = two_h[s]
        sites = A1 + A2
        iv = intervals_from_parts([A1, A2])
        n1, n2 = len(A1), len(A2)
        worst = 0.0
        for rows, cols, l, A, (a, b) in ((range(n1), range(n1, n1 + n2), 1, A1, iv[0]), (range(n1, n1 + n2), range(n1), 0, A2, iv[1])):
            S = bilocal_weight(h, sites, rows, cols)
            x = np.asarray(A) + 0.5
            W = np.array([ch_nonlocal_weight(v, iv, l) for v in x])
            inner = (x > a + 0.2 * (b - a)) & (x < b - 0.2 * (b - a))
            assert np.all(np.sign(S[inner]) == np.sign(W[inner])), "sign of the nonlocal coupling"
            worst = max(worst, float(np.max(np.abs(S / W - 1.0)[inner])))
        devs[s] = worst
        assert worst < tol[s], (
            f"scale {s}: interior row sums vs CH nonlocal weight max rel dev {worst:.3e} "
            f"(tolerance {tol[s]}; measured 9.1e-2, 9.9e-3, 2.2e-3 at scales 1, 2, 4)"
        )
    assert devs[1] > devs[2] > devs[4], f"finite-size trend not decreasing: {devs}"


def _sin2_envelopes(iv):
    (a1, b1), (a2, b2) = iv
    return (lambda x: np.sin(np.pi * (x - a1) / (b1 - a1)) ** 2, lambda y: np.sin(np.pi * (y - a2) / (b2 - a2)) ** 2)


def test_nonlocal_smeared_bilinear_sin2(two_h):
    tol = {1: 5e-2, 2: 1.5e-2, 4: 4e-3}
    devs = {}
    for s in TWO_SCALES:
        A1, A2, h = two_h[s]
        sites = A1 + A2
        iv = intervals_from_parts([A1, A2])
        n1 = len(A1)
        g, f = _sin2_envelopes(iv)
        M_lat = chiral_bilinear(h, sites, g, f, range(n1), range(n1, len(sites)))
        M_ch = ch_bilinear(iv, 0, 1, g, f)
        assert abs(M_lat.real) < 1e-12 * abs(M_lat), "the right-mover bilinear must be purely imaginary at half filling"
        devs[s] = abs(abs(M_lat) / abs(M_ch) - 1.0)
        assert devs[s] < tol[s], (
            f"scale {s}: smeared bilocal element lattice/CH - 1 = {abs(M_lat) / abs(M_ch) - 1:+.3e} "
            f"(tolerance {tol[s]}; measured -2.3e-2, +7.1e-3, +1.7e-3)"
        )
    assert devs[1] > devs[2] > devs[4], f"finite-size trend not decreasing: {devs}"


def test_nonlocal_smeared_grid(two_h):
    """A 4x4 grid of Gaussian envelopes (including off-hyperbola pairs) vs the CH integral."""
    tol = {2: 2e-2, 4: 5e-3}
    devs = {}
    for s in (2, 4):
        A1, A2, h = two_h[s]
        sites = A1 + A2
        iv = intervals_from_parts([A1, A2])
        n1, n2 = len(A1), len(A2)
        sig = n1 / 8
        cx = np.linspace(iv[0][0] + 0.2 * n1, iv[0][1] - 0.2 * n1, 4)
        cy = np.linspace(iv[1][0] + 0.2 * n2, iv[1][1] - 0.2 * n2, 4)
        Ml = np.zeros((4, 4), complex)
        Mc = np.zeros((4, 4), complex)
        for a, x0 in enumerate(cx):
            for b, y0 in enumerate(cy):
                g = lambda x, x0=x0: np.exp(-((x - x0) ** 2) / (2 * sig**2))
                f = lambda y, y0=y0: np.exp(-((y - y0) ** 2) / (2 * sig**2))
                Ml[a, b] = chiral_bilinear(h, sites, g, f, range(n1), range(n1, n1 + n2))
                Mc[a, b] = ch_bilinear(iv, 0, 1, g, f)
        devs[s] = float(np.linalg.norm(Ml - Mc) / np.linalg.norm(Mc))
        assert devs[s] < tol[s], f"scale {s}: grid Frobenius rel dev {devs[s]:.3e} (tolerance {tol[s]}; measured 9.8e-3, 2.0e-3)"
    assert devs[2] > devs[4]


def test_unequal_intervals(two_h):
    A1, A2, h = two_h["unequal"]
    sites = A1 + A2
    iv = intervals_from_parts([A1, A2])
    n1, n2 = len(A1), len(A2)
    _, d1 = _resummed_dev(h[:n1, :n1], A1, iv)
    _, d2 = _resummed_dev(h[n1:, n1:], A2, iv)
    assert np.max(np.abs(d1)) < 1e-2 and np.max(np.abs(d2)) < 1e-2, (
        f"unequal (20, 10, gap 8): resummed local dev {np.max(np.abs(d1)):.2e}, {np.max(np.abs(d2)):.2e} (measured 3.9e-3, 5.2e-3)"
    )
    for rows, cols, l, A, (a, b), tol in ((range(n1), range(n1, n1 + n2), 1, A1, iv[0], 0.1), (range(n1, n1 + n2), range(n1), 0, A2, iv[1], 5e-2)):
        S = bilocal_weight(h, sites, rows, cols)
        x = np.asarray(A) + 0.5
        W = np.array([ch_nonlocal_weight(v, iv, l) for v in x])
        inner = (x > a + 0.2 * (b - a)) & (x < b - 0.2 * (b - a))
        dev = float(np.max(np.abs(S / W - 1.0)[inner]))
        assert dev < tol, f"unequal: interior row sums dev {dev:.2e} (tolerance {tol}; measured 5.8e-2, 2.2e-2)"
    g, f = _sin2_envelopes(iv)
    r = abs(chiral_bilinear(h, sites, g, f, range(n1), range(n1, n1 + n2))) / abs(ch_bilinear(iv, 0, 1, g, f)) - 1.0
    assert abs(r) < 2e-2, f"unequal: smeared bilinear ratio - 1 = {r:+.2e} (measured 8.4e-3)"


def test_casini_huerta_exact_sampled_kernel(two_h):
    for s, tol in ((2, 1e-2), (4, 3e-3)):
        A1, A2, h = two_h[s]
        sites = A1 + A2
        iv = intervals_from_parts([A1, A2])
        n1, n2 = len(A1), len(A2)
        H = casini_huerta_exact(A1, A2)
        assert np.max(np.abs(H - H.conj().T)) < 1e-12, "H_R must be Hermitian"
        hR = from_chiral_gauge(H, sites)
        assert np.max(np.abs(hR.imag)) < 1e-12 and np.max(np.abs(hR.real - hR.real.T)) < 1e-12, "the dictionary must give a real symmetric h"
        hR = hR.real
        x, b = bw_profile(hR[:n1, :n1], mode="nn", offset=A1[0])
        assert np.max(np.abs(b / ch_beta(x, iv) - 1.0)) < 1e-10, "local part must be -pi/z' by construction"
        S = bilocal_weight(hR, sites, range(n1), range(n1, n1 + n2))
        W = np.array([ch_nonlocal_weight(v, iv, 1) for v in np.asarray(A1) + 0.5])
        assert np.max(np.abs(S / W - 1.0)) < 1e-10, "source-side row sums must reproduce CH Eq. (48) by construction"
        g, f = _sin2_envelopes(iv)
        M_s = chiral_bilinear(hR, sites, g, f, range(n1), range(n1, n1 + n2))
        M_l = chiral_bilinear(h, sites, g, f, range(n1), range(n1, n1 + n2))
        M_c = ch_bilinear(iv, 0, 1, g, f)
        assert abs(abs(M_s) / abs(M_c) - 1.0) < tol, f"scale {s}: sampled kernel vs exact integral {abs(M_s) / abs(M_c) - 1:+.2e} (measured 3.6e-3, 1.1e-3)"
        assert abs(M_s - M_l) / abs(M_c) < 1e-2, f"scale {s}: sampled kernel vs lattice h {abs(M_s - M_l) / abs(M_c):.2e}"


def test_mutual_information_vs_ch_cross_ratio(two_h):
    for s, tol in ((2, 1e-3), (4, 5e-4)):
        A1, A2, _ = two_h[s]
        sites = A1 + A2
        n1 = len(A1)
        C = infinite_chain_C(sites)
        MI = block_entropy(C, range(n1)) + block_entropy(C, range(n1, len(sites))) - block_entropy(C, range(len(sites)))
        iv = intervals_from_parts([A1, A2])
        diff = MI - ch_mutual_information(iv[0], iv[1])
        assert abs(diff) < tol, f"scale {s}: I(A1:A2) lattice {MI:.5f} vs CH cross ratio {ch_mutual_information(iv[0], iv[1]):.5f} (diff {diff:+.1e}; measured -3.2e-4, -0.8e-4)"


def test_modular_flow_teleportation_eq59(two_h):
    """A right-moving packet in A1 acquires the weight sin^2 theta(tau) in A2 (CH Eq. (59)) and rides the CH trajectories."""
    A1, A2, h = two_h[2]
    sites = A1 + A2
    n1 = len(A1)
    iv = intervals_from_parts([A1, A2])
    parts = [range(n1), range(n1, len(sites))]
    x0 = iv[0][0] + 0.5 * n1
    psiR = wavepacket(sites, x0, 2.0, chirality=+1)
    psiL = wavepacket(sites, x0, 2.0, chirality=-1)
    for tau in (0.1, 0.2, 0.3, -0.1):
        psi = flow_wavefunction(h, psiR, tau)
        P2 = part_weights(psi, parts)[1]
        pred = np.sin(ch_mixing_angle(x0, tau, iv[0], iv[1])) ** 2
        assert abs(P2 / pred - 1.0) < 0.08, f"tau={tau}: P2 = {P2:.4f} vs sin^2 theta = {pred:.4f} (rel dev {P2 / pred - 1:+.3f}; measured 0.047, 0.036, 0.004, 0.034)"
        c = part_centroids(psi, sites, parts)
        assert abs(c[0] - ch_trajectory(x0, tau, iv, 0)) < 0.5, f"tau={tau}: A1 centroid {c[0]:.2f} vs CH x1 {ch_trajectory(x0, tau, iv, 0):.2f} (measured <= 0.36)"
        assert abs(c[1] - ch_trajectory(x0, tau, iv, 1)) < 0.8, f"tau={tau}: A2 centroid {c[1]:.2f} vs CH x2 {ch_trajectory(x0, tau, iv, 1):.2f} (measured <= 0.70)"
        # left movers are the complex conjugate: they flow backwards (dictionary D5)
        P2L = part_weights(flow_wavefunction(h, psiL, -tau), parts)[1]
        assert abs(P2L - P2) < 1e-12


def test_single_interval_flow_eq53(single_h):
    L = 40
    h, _ = single_h[L]
    sites = list(range(L))
    x0 = 12.0
    psi0 = wavepacket(sites, x0, 2.0, chirality=+1)
    for tau in (0.05, 0.1, 0.2, -0.1):
        c = part_centroids(flow_wavefunction(h, psi0, tau), sites, [range(L)])[0]
        pred = ch_single_interval_flow(x0, tau, 0.0, float(L))
        assert abs(c - pred) < 0.5, f"tau={tau}: centroid {c:.3f} vs CH Eq. (53) {pred:.3f} (measured devs 0.09, 0.19, 0.36, 0.17)"


# =========================================================================== #
# Anchor 3: the fermionic first law
# =========================================================================== #
def _ring_C_with_bond(N, eps, bond):
    hh = staggered_mass_single_particle(N, 0.0)
    i, j = bond
    hh[i, j] -= eps
    hh[j, i] -= eps
    w, v = np.linalg.eigh(hh)
    assert w[N // 2] - w[N // 2 - 1] > 1e-6
    occ = v[:, : N // 2]
    return occ @ occ.T


def _first_law(N, sites, bond, h_list):
    def central(e):
        Cp = _ring_C_with_bond(N, e, bond)
        Cm = _ring_C_with_bond(N, -e, bond)
        dS = (block_entropy(Cp, sites) - block_entropy(Cm, sites)) / (2 * e)
        dC = (Cp[np.ix_(sites, sites)] - Cm[np.ix_(sites, sites)]) / (2 * e)
        return [dS] + [float(np.trace(h @ dC)) for h in h_list]

    a, b = central(1e-3), central(5e-4)
    return [(4 * y - x) / 3 for x, y in zip(a, b)]  # Richardson: O(e^4)


def test_first_law_single_interval_independent_routes():
    """dS = Tr(h dC_A): entropy differences vs the float64 Peschel h vs an mp-diagonalized h."""
    N, L = 200, 12
    sites = list(range(L))
    C0 = _ring_C_with_bond(N, 0.0, (5, 6))
    h64 = interval_modular(C0, sites)
    hmp = interval_modular(ring_C(N, sites, dps=40), range(L), dps=40)
    for bond in ((5, 6), (30, 31), (11, 12)):  # inside, far outside, straddling the edge
        dS, dK64, dKmp = _first_law(N, sites, bond, [h64, hmp])
        for name, dK in (("float64", dK64), ("mp", dKmp)):
            rel = abs(dS - dK) / abs(dS)
            assert rel < 1e-8, f"bond {bond} [{name}]: dS = {dS:.12f}, Tr(h dC) = {dK:.12f}, rel {rel:.2e} (measured 2.6e-10, 3.8e-10, 2.2e-11)"


def test_first_law_two_intervals():
    N = 200
    A1, A2 = list(range(6)), list(range(10, 16))
    sites = A1 + A2
    C0 = _ring_C_with_bond(N, 0.0, (2, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h = two_interval_modular(C0, A1, A2)
    for bond in ((2, 3), (7, 8), (12, 13), (50, 51)):
        dS, dK = _first_law(N, sites, bond, [h])
        rel = abs(dS - dK) / abs(dS)
        assert rel < 1e-8, f"two intervals, bond {bond}: dS = {dS:.10f}, Tr(h dC) = {dK:.10f}, rel {rel:.2e} (measured <= 1.8e-10)"


# =========================================================================== #
# Anchor 4: modular flow is unitary
# =========================================================================== #
def test_modular_flow_preserves_spectrum(single_h, two_h):
    cases = [("single L=40", single_h[40][0], list(range(40)))]
    A1, A2, h2 = two_h[2]
    cases.append(("two intervals scale 2", h2, A1 + A2))
    for name, h, sites in cases:
        C_A = infinite_chain_C(sites)
        spec0 = np.sort(np.linalg.eigvalsh(C_A))
        for s in (0.3, 1.0, 5.0):
            U = modular_flow_unitary(h, s)
            n = len(sites)
            assert np.linalg.norm(U @ U.conj().T - np.eye(n)) < 1e-12, f"{name}, s={s}: U not unitary"
            assert np.linalg.norm(U - expm(-1j * s * h)) < 1e-11, f"{name}, s={s}: eigh route vs expm (measured <= 2.3e-13)"
            Cs = modular_flow(h, C_A, s)
            assert np.linalg.norm(Cs - C_A) < 1e-12, f"{name}, s={s}: C_A must be invariant ([h, C_A] = 0): {np.linalg.norm(Cs - C_A):.1e}"
            spec = np.sort(np.linalg.eigvalsh(0.5 * (Cs + Cs.conj().T)))
            assert np.max(np.abs(spec - spec0)) < 1e-12, f"{name}, s={s}: spectrum moved by {np.max(np.abs(spec - spec0)):.1e} (measured ~1e-15)"
        # a genuinely different state is rotated, isospectrally
        Cm = massive_chain_C(sites, 0.3)
        Cs = modular_flow(h, Cm, 1.0)
        assert np.linalg.norm(Cs - Cm) > 1e-3
        assert np.max(np.abs(np.sort(np.linalg.eigvalsh(0.5 * (Cs + Cs.conj().T))) - np.sort(np.linalg.eigvalsh(Cm)))) < 1e-12


# =========================================================================== #
# Bosonic side and shared diagnostics
# =========================================================================== #
def test_region_modular_reports_floor_and_matches_core():
    N, m, modes = 60, 0.05, list(range(20, 30))
    K = harmonic_chain_K(N, m, "periodic")
    V = ground_state_cov(K)
    G, info = region_modular(V, None, modes, return_info=True)
    assert np.allclose(G, bosonic_eh(reduce(V, modes)))
    G2 = region_modular(None, K, modes)
    assert np.allclose(G, G2)
    assert info["escalated"] and info["n_floor"] >= 1 and info["margins_mp"] is not None, "floor modes must be reported, not clamped silently"
    V_A = reduce(V, modes)
    prof = modular_energy_profile(G, V_A)
    assert abs(prof.sum() - modular_energy(G, V_A)) < 1e-9 * abs(modular_energy(G, V_A))
    assert prof.shape == (len(modes),)
    r = first_law_check(K, modes, lambda K, e: K + e * np.eye(N), 1e-7)
    assert r["rel_err"] < 1e-6, f"bosonic first law rel_err {r['rel_err']:.2e}"


def test_locality_score(single_h):
    n = 8
    tri = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    assert locality_score(tri, 1) == 1.0
    far = tri.copy()
    far[0, 5] = far[5, 0] = 1.0
    assert abs(locality_score(far, 1) - 14 / 16) < 1e-12
    assert 0.999 < locality_score(single_h[40][0], 1) < 1.0
    G = bosonic_eh(reduce(ground_state_cov(harmonic_chain_K(30, 0.5, "periodic")), range(10, 16)))
    assert 0.0 < locality_score(G, 1, block=2) <= 1.0


def test_nonlocal_decay_structure(two_h):
    A1, A2, h = two_h[2]
    sites = A1 + A2
    n1 = len(A1)
    rep = nonlocal_decay(h, sites, [range(n1), range(n1, len(sites))])
    assert 0.0 < rep["offblock_fraction"] < 0.01
    p = rep["pairs"][0]
    assert p["gap"] == 11.0 and p["fro"] > 0 and p["profile"].shape == p["distance"].shape  # 10 empty sites -> site distance 11
    assert np.allclose(p["row_weight_pq"], bilocal_weight(h, sites, range(n1), range(n1, len(sites))))


def test_multi_interval_scan_smoke():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModularPrecisionWarning)
        reps = multi_interval_scan([
            {"label": "three", "intervals": [(0, 6), (9, 6), (18, 6)]},
            {"label": "massive", "intervals": [(0, 6), (9, 6)], "mass": 0.5},
        ])
    assert [r["label"] for r in reps] == ["three", "massive"]
    assert len(reps[0]["nonlocal"]["pairs"]) == 3 and "ch_weight_pq" in reps[0]["nonlocal"]["pairs"][0]
    assert reps[1]["xi"] == pytest.approx(1.0 / np.arcsinh(0.5)) and "ch_weight_pq" not in reps[1]["nonlocal"]["pairs"][0]
    assert all(len(r["bw"]) == len(r["intervals"]) for r in reps)


# =========================================================================== #
# Anchor 5: n >= 3 intervals at growing L -- the 1/L^2 trend and extrapolation
# =========================================================================== #
def test_three_intervals_converge_as_one_over_L2_and_extrapolate_to_ch():
    """At fixed gap/L the lattice weight sum approaches the n-interval CH weight as A/L^2.

    The row-by-row deviation is noisy (single rows near the trimmed edge dominate);
    the summed bilocal weight per interval -- the quantity the screening and decay
    results of papers/multi-interval-modular-hamiltonians use -- is clean.
    """
    Ls = (12, 16, 24)
    dev_sum, dev_row = [], []
    for L in Ls:
        d = L
        rep = multi_interval_scan([{"label": f"three_L{L}", "intervals": [(0, L), (L + d, L), (2 * L + 2 * d, L)]}])[0]
        pairs = rep["nonlocal"]["pairs"]
        dev_sum.append(max(abs(float(np.sum(np.abs(p["row_weight_pq"]))) / float(np.sum(np.abs(p["ch_weight_pq"]))) - 1.0)
                           for p in pairs))
        dev_row.append(max(max(p["dev_interior_pq"], p["dev_interior_qp"]) for p in pairs))
    dev_sum = np.array(dev_sum)
    dev_row = np.array(dev_row)
    Lv = np.array(Ls, dtype=float)
    A = dev_sum * Lv**2
    assert np.all(np.diff(dev_sum) < 0) and np.all(np.diff(dev_row) < 0), f"not converging: {dev_sum}, {dev_row}"
    assert np.max(np.abs(A / A.mean() - 1.0)) < 0.06, (
        f"summed-weight deviation is not A/L^2: A = {A} (measured 1.90, 1.92, 1.96 at L = 12, 16, 24)"
    )
    M = np.vstack([np.ones(Lv.size), Lv**-2.0]).T
    c = np.linalg.lstsq(M, dev_sum, rcond=None)[0]
    assert abs(c[0]) < 1e-3, (
        f"extrapolated L -> inf deviation {c[0]:+.2e} is not consistent with 0 "
        f"(fit {c[1]:.2f}/L^2; measured -1.2e-4)"
    )
    assert dev_row[-1] < 6e-3, f"L=24 row-by-row interior deviation {dev_row[-1]:.2e} (measured 3.5e-3)"


# =========================================================================== #
# Anchor 6: xi_h/xi -> 1 (the 1.3-1.4 of the first scan was a crossover)
# =========================================================================== #
def test_modular_coupling_decay_length_tends_to_the_correlation_length():
    """The LOCAL decay rate of the alternating row weight at gaps >~ 6 xi gives xi_h = xi.

    A single exponential fitted over a window of gaps returns an effective length
    that still carries the algebraic prefactor of the massive propagator, which is
    why a fit over gaps of a few xi gave 1.3-1.4.
    """
    m, L = 0.1, 12
    gaps = (32, 64, 96, 128)
    xi = correlation_length(m)
    reps = multi_interval_scan([{"label": f"d{d}", "intervals": [(0, L), (L + d, L)], "mass": m} for d in gaps])
    W = np.array([float(np.max(np.abs(r["nonlocal"]["pairs"][0]["row_weight_pq"]))) for r in reps])
    d = np.array(gaps, dtype=float)
    loc = (d[1:] - d[:-1]) / np.log(W[:-1] / W[1:]) / xi
    assert np.all(W[:-1] > W[1:]) and W[-1] > 1e-12, f"weights not decaying above the floor: {W}"
    assert np.max(np.abs(loc - 1.0)) < 0.06, (
        f"xi_h/xi at gap/xi = {np.round(0.5 * (d[1:] + d[:-1]) / xi, 1)} is {np.round(loc, 3)}; "
        f"the asymptotic value is 1 (measured 1.031, 0.992, 0.980)"
    )


# =========================================================================== #
# Anchor 7: the corner-transfer-matrix closed form for the gapped slope s(m)
# =========================================================================== #
def test_ctm_closed_form_reproduces_the_gapped_entanglement_hamiltonian(gapped_h):
    """s(m) = (2/pi) k I(k') and eps = pi I(k')/I(k), k = 1/sqrt(1 + m^2) (Eisler 2025 Eq. (72)).

    For a segment long compared with xi the entanglement Hamiltonian near an edge
    IS the half-infinite operator 2 pi s(m) sum_x x h_x: the nearest-neighbour
    hopping and the staggered on-site term are both linear in the distance from the
    edge with that one slope, and the single-particle levels are (2l + 1) eps.
    """
    nb = 8
    for m, L in GAPPED_CASES:  # L - 2 nb >= 20 xi: the far edge is out of range
        h = gapped_h[(m, L)]
        xi = correlation_length(m)
        s_ctm, eps_ctm = ctm_slope(m), ctm_level_spacing(m)
        s_bond = -2.0 * h[0, 1] / (2.0 * np.pi)
        s_site = h[0, 0] / m / (2.0 * np.pi * 0.5)
        for name, val in (("nn bond", s_bond), ("on-site", s_site)):
            assert abs(val / s_ctm - 1.0) < 1e-12, (
                f"m={m}: edge slope from the {name} is {val:.12f} vs CTM (2/pi) k I(k') = {s_ctm:.12f} "
                f"(rel {val / s_ctm - 1:+.1e}; measured <= 1.2e-14)")
        ev = np.sort(np.abs(np.linalg.eigvalsh(h)))
        assert abs(ev[0] / eps_ctm - 1.0) < 1e-12, (
            f"m={m}: lowest |eps| = {ev[0]:.12f} vs pi I(k')/I(k) = {eps_ctm:.12f}")
        for k, want in ((2, 3.0), (4, 5.0), (6, 7.0)):
            assert abs(ev[k] / ev[0] - want) < 1e-8, f"m={m}: level ratio {ev[k] / ev[0]:.9f} vs {want}"
        # within a few xi of ONE edge the three CTM diagonals ARE the lattice boost;
        # the deviation is set by the distance to the OTHER edge, not by nb/xi
        hb = lattice_boost(L, m)
        assert L - 2 * nb > 20 * xi, f"m={m}: window {nb} too wide for L={L}, xi={xi:.2f}"
        scale = np.max(np.abs(hb[:nb, :nb]))
        band = np.max([np.max(np.abs(np.diag(h, k)[:nb] - np.diag(hb, k)[:nb])) for k in (0, 1)]) / scale
        assert band < 1e-8, (
            f"m={m}: first {nb} sites, the two CTM diagonals vs the boost, rel dev {band:.1e} "
            f"(measured 3.8e-9 at m = 1, L = 40 and 6.0e-12 at m = 0.5, L = 80)")
        long_range = np.max(np.abs(np.diag(h, 3)[:nb])) / scale
        assert long_range < 1e-7, (
            f"m={m}: third-neighbour hopping in the CTM window is {long_range:.1e} of the boost scale "
            f"(the CTM operator is tridiagonal; measured 4.2e-8, 7.2e-11)")
        far = 3 * L // 4 - 1  # the bond at x = 3L/4: the boost says 3L/4, the triangle says L/4
        tri = triangle_h(L, m)
        assert abs(h[far, far + 1] / hb[far, far + 1]) < 0.5, (
            "the CTM line must NOT continue past the middle of a finite segment "
            f"(ratio {abs(h[far, far + 1] / hb[far, far + 1]):.3f}); the profile is the triangle")
        assert abs(h[far, far + 1] / hb[far, far + 1] - 1.0 / 3.0) < 0.02, (
            f"m={m}: the bond at 3L/4 is 1/3 of the boost value ({h[far, far + 1] / hb[far, far + 1]:.4f})")
        assert abs(h[far, far + 1] / tri[far, far + 1] - 1.0) < 0.01, (
            f"m={m}: bond at 3L/4 vs the triangle form of Eisler et al. 2020 Eqs. (42)-(44): "
            f"{h[far, far + 1] / tri[far, far + 1]:.5f}")


def test_ctm_slope_reproduces_the_fitted_gapped_slopes_and_its_scaling_limit():
    """The fitted s = 0.997/0.990/0.945/0.835 at xi = 10/5/2.1/1.1 IS (2/pi) k I(k')."""
    fitted = {0.1: 0.9970920281873782, 0.2: 0.9902151996908025,
              0.5: 0.9450063309274378, 1.0: 0.8346268416740733}  # papers/.../data/summary.json, L = 40
    for m, s_fit in fitted.items():
        s_ctm = ctm_slope(m)
        tol = 5e-4 if m >= 0.2 else 5e-3
        assert abs(s_fit / s_ctm - 1.0) < tol, (
            f"m={m} (xi={correlation_length(m):.2f}): fitted edge slope {s_fit:.7f} vs CTM {s_ctm:.7f} "
            f"(rel {s_fit / s_ctm - 1:+.1e}; the finite-L residual, measured -4.2e-4 at m = 0.1 and <= 4e-6 for m >= 0.2)")
    for m in (0.05, 0.1, 0.2):
        xi = correlation_length(m)
        assert abs(ctm_slope(m) - ctm_slope_expansion(xi)) < 1.0 / xi**4, (
            f"m={m}: s = {ctm_slope(m):.8f} vs 1 - 1/(4 xi^2) = {ctm_slope_expansion(xi):.8f}")
    assert ctm_slope(1e-4) < 1.0 and 1.0 - ctm_slope(1e-4) < 1e-7, "s -> 1 (Bisognano-Wichmann) as m -> 0"
    assert ctm_slope(0.02) > ctm_slope(0.5) > ctm_slope(2.0), "s(m) decreases with the gap"

def _eisler_eq72_from_the_paper(mu):
    """Eisler, J. Stat. Mech. (2025) 013101, Eqs. (70), (72), transcribed here independently of ctm.py:
    kappa = 1/sqrt(1 + mu^2); prefactor 4 kappa K(kappa'); level unit 2 pi K(kappa')/K(kappa)
    (K in the modulus convention; mpmath's ellipk takes the parameter kappa^2)."""
    with mp.workdps(30):
        kap = 1 / mp.sqrt(1 + mp.mpf(mu) ** 2)
        kapp = mp.sqrt(1 - kap ** 2)
        pref = 4 * kap * mp.ellipk(kapp ** 2)
        unit = 2 * mp.pi * mp.ellipk(kapp ** 2) / mp.ellipk(kap ** 2)
        return float(kap), float(pref), float(unit)


def test_eq72_transcription_check_closed_form_vs_ctm_py():
    """TRANSCRIPTION CHECK, not an anchor: the same formula on both sides.

    ctm_slope, ctm_level_spacing and eisler_2025_prefactor must be the (2/pi) kappa K(kappa'),
    pi K(kappa')/K(kappa) and 4 kappa K(kappa') of Eisler's Eq. (72) with kappa = 1/sqrt(1 + m^2)
    (= sech(1/xi)). Agreement here (measured 0.0 at all seven m once kappa is formed in working
    precision; a float kappa costs 4.6e-14 at m = 0.02) only shows that ctm.py transcribes the
    paper correctly. It says nothing about the lattice -- that is the anchor below.
    """
    from vacuum.modular.ctm import eisler_2025_prefactor

    for m in (0.02, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0):
        kap, pref, unit = _eisler_eq72_from_the_paper(m)
        assert abs(2 * np.pi * ctm_slope(m) / pref - 1.0) < 1e-14, (
            f"m={m}: 2 pi s = {2 * np.pi * ctm_slope(m):.16f} vs 4 kappa K(kappa') = {pref:.16f}")
        assert abs(ctm_level_spacing(m) / (unit / 2) - 1.0) < 1e-14, (
            f"m={m}: eps = {ctm_level_spacing(m):.16f} vs pi K(kappa')/K(kappa) = {unit / 2:.16f}")
        assert abs(kap - 1.0 / np.cosh(1.0 / correlation_length(m))) < 1e-15, f"m={m}: kappa = 1/sqrt(1+m^2) is sech(1/xi)"
        assert abs(eisler_2025_prefactor(m) / pref - 1.0) < 1e-15


def test_anchor7_lattice_edge_matches_eisler_eq72(gapped_h):
    """ANCHOR 7: the LATTICE entanglement Hamiltonian against Eisler's published Eq. (72).

    The gapped chain's exact h (closed-form C at 87-119 digits, segments with L >= 23 xi) is
    compared with H = 4 kappa K(kappa') T, kappa = 1/sqrt(1 + mu^2). BOTH halves -- the edge
    slope and the signed ladder -- take their reference from _eisler_eq72_from_the_paper alone,
    i.e. from Eq. (72) as transcribed in this file, and use nothing from ctm.py:
      * the edge slope 2 pi s from BOTH diagonals (on-site term h_00/(m/2), nearest-neighbour
        bond -2 h_01) against 4 kappa K(kappa'), tolerance 1e-12 -- measured 2.2e-16, 2.2e-16
        (m = 1, L = 40) and 0.0e+00, 0.0e+00 (m = 1/2, L = 80);
      * the SIGNED level ladder with edge attribution (eigenvector weight on the site-0 half)
        against eps_l = 2 pi K(kappa')/K(kappa) (2l -/+ 1/2): our +m edge is Eisler's mu = -m chain
        (2l + 1/2: +eps, -3eps, +5eps, -7eps), our -m edge his mu = +m chain (2l - 1/2: -eps, +3eps,
        -5eps, +7eps); tolerance 1e-8 -- measured 6.7e-16.
    Mutations these tolerances catch (relative size at m = 1 / 1/2): the dimerized-chain modulus
    convention k = e^(-1/xi) in place of sech(1/xi): slope -27 % / -18 %, spacing +41 % / +52 %;
    dropping kappa from the prefactor (4 K(kappa') alone): +41 % / +12 %; 2 pi -> pi in the level
    unit: -50 %; swapping the +-1/2 shift (i.e. the edge attribution): the +m-edge set
    {-7, -3, +1, +5} eps becomes {-5, -1, +3, +7} eps, O(1). The per-edge spectrum is
    particle-hole ASYMMETRIC (no -eps on the +m edge); the union over both edges is symmetric.
    """
    for m, L in GAPPED_CASES:
        h = gapped_h[(m, L)]
        kap, pref, unit = _eisler_eq72_from_the_paper(m)  # both halves below use only this transcription
        for name, val in (("on-site", h[0, 0] / m / 0.5), ("nn bond", -2.0 * h[0, 1] / 1.0)):  # both = 2 pi s
            assert abs(val / pref - 1.0) < 1e-12, (
                f"m={m}: edge slope from the {name}, 2 pi s = {val:.15f}, vs Eisler Eq. (72) "
                f"4 kappa K(kappa') = {pref:.15f} (rel {val / pref - 1:+.1e}; measured 2.2e-16, 2.2e-16, 0.0e+00, 0.0e+00)")
        w, v = np.linalg.eigh(h)
        idx = np.argsort(np.abs(w))[:8]
        left = np.array([float(np.sum(v[: L // 2, i] ** 2)) for i in idx])  # weight on the +m (site-0) half
        assert np.all((left < 1e-6) | (left > 1 - 1e-6)), f"m={m}: low modes must be edge-localized: {left}"
        plus_edge = np.sort(w[idx][left > 0.5])   # our +m edge = Eisler's mu = -m
        minus_edge = np.sort(w[idx][left < 0.5])  # our -m edge = Eisler's mu = +m
        # Eq. (72) directly: eps_l = unit * (2l + 1/2) for Eisler's mu < 0 (our +m edge), (2l - 1/2) for mu > 0
        want_plus = np.sort([unit * (2 * l + 0.5) for l in range(-2, 2)])   # -7, -3, +1, +5 (in eps = unit/2)
        want_minus = np.sort([unit * (2 * l - 0.5) for l in range(-1, 3)])  # -5, -1, +3, +7
        assert np.max(np.abs(plus_edge / want_plus - 1.0)) < 1e-8, (
            f"m={m}: +m edge signed ladder {plus_edge / (unit / 2)} vs Eq. (72) with mu = -m: {want_plus / (unit / 2)} (measured 6.7e-16)")
        assert np.max(np.abs(minus_edge / want_minus - 1.0)) < 1e-8, (
            f"m={m}: -m edge signed ladder {minus_edge / (unit / 2)} vs Eq. (72) with mu = +m: {want_minus / (unit / 2)} (measured 6.7e-16)")
        assert np.min(np.abs(plus_edge + unit / 2)) > unit / 4 and np.min(np.abs(minus_edge - unit / 2)) > unit / 4
        assert np.max(np.abs(np.sort(w[idx]) + np.sort(w[idx])[::-1])) < 1e-8
