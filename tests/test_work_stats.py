"""M3.5 work statistics: TPM anchors, the dense<->Gaussian gate, and audits.

Structure mirrors the spec's requirement list:

1.  plumbing and exact free energies (independent brute-force checks),
2.  the published single-mode anchor -- Deffner & Lutz, PRE 77, 021128
    (2008), Eqs. (15), (17), (18), (30), (31) -- against BOTH routes,
3.  THE GATE: dense Fock-space TPM vs the Gaussian determinant route to
    1e-8 at N = 1, 2, 3 (spec M3.5; the Gaussian route is not trusted at
    any N until this passes),
4.  occupation-cutoff tail test (cutoff doubling; and a deliberately too
    small cutoff must be *flagged*, not silently wrong),
5.  Jarzynski |<e^{-beta W}> - e^{-beta DeltaF}| < 1e-10 dense / 1e-6
    Gaussian, with DeltaF from the exact mode-product free energy,
6.  Crooks symmetry on forward/reverse pairs -- transition-resolved on the
    dense route, characteristic-function form on the Gaussian route,
7.  numerical-hazard gates (branch-tracking step halving, cumulant
    stencil), purity, and the EnergyLedger wiring.

Conventions per docs/API.md: hbar = 1, block ordering R = (x_1..x_N,
p_1..p_N), beta = 1/T.
"""

from __future__ import annotations

import numpy as np
import pytest

from vacuum.audits import AuditViolation, energy_ledger
from vacuum.audits.ledger import WORK_STATS_REQUIRED_KEYS
from vacuum.core import (
    Omega,
    ground_state_cov,
    harmonic_chain_K,
    mean_energy,
    thermal_state_cov,
)
from vacuum.inequalities.work_stats import (
    COUNTING_STATISTICS_KEYS,
    counting_statistics,
    crooks_max_defect,
    deffner_lutz_cf,
    deffner_lutz_cf_sudden,
    deffner_lutz_final_n_moments,
    deffner_lutz_q_star,
    delta_free_energy,
    dense_hamiltonian,
    dense_jarzynski_average,
    dense_tpm,
    dense_work_cf,
    dense_work_cumulants,
    dense_work_distribution,
    dense_work_moments,
    dense_work_quantile,
    gaussian_crooks_defect,
    gaussian_free_energy,
    gaussian_jarzynski_average,
    gaussian_work_cf,
    gaussian_work_cf_at,
    gaussian_work_cumulants,
    gaussian_work_moments,
    jarzynski_defect,
    log_partition_gaussian,
    mode_frequencies,
    protocol_symplectic,
    quadratic_form_matrix,
    reference_frequency,
    reverse_segments,
)

BETA = 2.0
U_GRID = np.linspace(-2.0, 2.0, 9)


def _chain(N, m):
    return harmonic_chain_K(N, m, bc="dirichlet")


# ---------------------------------------------------------------------------
# protocol definitions shared by the anchors (module-scoped: the dense builds
# are the expensive part of this file)
# ---------------------------------------------------------------------------


class Protocol:
    """A (K0, segments, K1, beta) quench/drive plus its symplectic."""

    def __init__(self, K0, K1, segments, beta=BETA):
        self.K0 = np.asarray(K0, dtype=float)
        self.K1 = np.asarray(K1, dtype=float)
        self.segments = [(np.asarray(K, float), float(dt)) for K, dt in segments]
        self.beta = float(beta)
        self.N = self.K0.shape[0]
        self.S = protocol_symplectic(self.segments, self.N)
        self.S_rev = protocol_symplectic(reverse_segments(self.segments), self.N)
        self.dF = delta_free_energy(self.K0, self.K1, self.beta)


# N = 1: single oscillator, w0 = 1.3 -> w1 = 2.1 through w = 3.0 for t = 0.6.
PROTO_1 = Protocol(
    np.array([[1.3**2]]), np.array([[2.1**2]]), [(np.array([[3.0**2]]), 0.6)]
)
# N = 1 sudden quench (the Deffner-Lutz Q* = (w0^2+w1^2)/(2 w0 w1) limit).
PROTO_1_SUDDEN = Protocol(np.array([[1.3**2]]), np.array([[2.1**2]]), [])
# N = 2 dirichlet chain, mass quench through a stiffer intermediate.
PROTO_2 = Protocol(_chain(2, 1.0), _chain(2, 1.15), [(_chain(2, 1.3), 0.5)])
# N = 3 dirichlet chain (the spec's N <= 3 dense limit).
PROTO_3 = Protocol(_chain(3, 2.0), _chain(3, 2.15), [(_chain(3, 2.4), 0.35)])

# dense cutoffs: chosen so the cutoff-doubling test still moves the answer
# below the 1e-8 gate (see TestTailTest for the convergence evidence).
NMAX = {1: 100, 2: 18, 3: 12}


@pytest.fixture(scope="module")
def tpm1():
    return dense_tpm(
        PROTO_1.K0, PROTO_1.K1, PROTO_1.segments, beta=PROTO_1.beta, n_max=NMAX[1]
    )


@pytest.fixture(scope="module")
def tpm1_sudden():
    return dense_tpm(
        PROTO_1_SUDDEN.K0, PROTO_1_SUDDEN.K1, (), beta=PROTO_1_SUDDEN.beta, n_max=NMAX[1]
    )


@pytest.fixture(scope="module")
def tpm2():
    return dense_tpm(
        PROTO_2.K0, PROTO_2.K1, PROTO_2.segments, beta=PROTO_2.beta, n_max=NMAX[2]
    )


@pytest.fixture(scope="module")
def tpm2_rev():
    p = PROTO_2
    return dense_tpm(
        p.K1,
        p.K0,
        reverse_segments(p.segments),
        beta=p.beta,
        n_max=NMAX[2],
        omega_ref=reference_frequency(p.K0, p.K1, p.segments),
    )


@pytest.fixture(scope="module")
def tpm3():
    return dense_tpm(
        PROTO_3.K0, PROTO_3.K1, PROTO_3.segments, beta=PROTO_3.beta, n_max=NMAX[3]
    )


# ---------------------------------------------------------------------------
# 1. plumbing and exact free energies
# ---------------------------------------------------------------------------


class TestPlumbing:
    def test_quadratic_form_matches_mean_energy(self):
        """(1/2) Tr(M(K) V) is exactly vacuum.core's <H> = mean_energy(V, K)."""
        K = _chain(5, 0.7)
        M = quadratic_form_matrix(K)
        for V in (ground_state_cov(K), thermal_state_cov(K, 0.9)):
            direct = mean_energy(V, K)
            via_M = 0.5 * np.trace(M @ V)
            assert abs(direct - via_M) < 1e-13, (
                f"<H> mismatch: mean_energy={direct:.15e}, (1/2)Tr(MV)={via_M:.15e}"
            )

    def test_log_partition_against_brute_force_sum(self):
        """ln Z from the mode product vs an explicit sum over Fock levels."""
        for K in (np.array([[1.7**2]]), np.diag([1.1**2, 2.3**2])):
            w = mode_frequencies(K)
            n = np.arange(0, 400)
            lnZ_brute = 0.0
            for wk in w:
                Zk = np.sum(np.exp(-BETA * wk * (n + 0.5)))
                lnZ_brute += np.log(Zk)
            lnZ = log_partition_gaussian(K, BETA)
            assert abs(lnZ - lnZ_brute) < 1e-13, (
                f"ln Z = {lnZ:.15e} vs brute force {lnZ_brute:.15e}"
            )

    def test_free_energy_and_delta(self):
        K0, K1 = _chain(4, 0.8), _chain(4, 1.3)
        F0 = gaussian_free_energy(K0, BETA)
        F1 = gaussian_free_energy(K1, BETA)
        assert abs(delta_free_energy(K0, K1, BETA) - (F1 - F0)) < 1e-14
        # stiffer chain => higher free energy
        assert F1 > F0

    def test_protocol_symplectic_is_symplectic(self):
        p = PROTO_2
        Om = Omega(p.N)
        defect = np.max(np.abs(p.S @ Om @ p.S.T - Om))
        assert defect < 1e-13, f"S Omega S^T - Omega = {defect:.3e}"
        assert np.allclose(protocol_symplectic([], 3), np.eye(6), atol=0.0)

    def test_reference_frequency_is_geometric_mean_of_extremes(self):
        p = PROTO_3
        allw = np.concatenate(
            [mode_frequencies(p.K0), mode_frequencies(p.K1)]
            + [mode_frequencies(K) for K, _ in p.segments]
        )
        nu = reference_frequency(p.K0, p.K1, p.segments)
        assert abs(nu - np.sqrt(allw.min() * allw.max())) < 1e-14

    def test_dense_hamiltonian_reproduces_oscillator_spectrum(self):
        """One mode, omega_ref = omega: H = w(a^dag a + aa^dag)/2, exactly.

        On the truncated space a a^dag = diag(1..n_max, 0), so H is diagonal
        at w(n + 1/2) for every rung BELOW the cutoff and sits at w*n_max/2
        on the top rung -- the whole truncation error of the matched basis,
        in one matrix element. This is why the cutoff must be diagnosed
        (tail weight) rather than assumed.
        """
        w, n_max = 1.9, 12
        H = dense_hamiltonian(np.array([[w**2]]), n_max=n_max, omega_ref=w)
        exact = w * (np.arange(n_max + 1) + 0.5)
        off_diag = np.max(np.abs(H - np.diag(np.diag(H))))
        assert off_diag < 1e-14, f"matched-basis H is not diagonal: {off_diag:.3e}"
        below = np.max(np.abs(np.diag(H)[:n_max] - exact[:n_max]))
        assert below < 1e-13, f"sub-cutoff spectrum off by {below:.3e}"
        assert abs(H[n_max, n_max] - w * n_max / 2.0) < 1e-13, (
            f"top rung = {H[n_max, n_max]:.12f}, expected the truncated "
            f"{w * n_max / 2.0:.12f} (exact value would be {exact[n_max]:.12f})"
        )

    def test_dense_hamiltonian_ground_energy_multimode(self):
        """Truncated ground energy -> sum_k omega_k/2 (the exact zero point)."""
        K = _chain(2, 1.1)
        nu = reference_frequency(K, K)
        E = np.linalg.eigvalsh(dense_hamiltonian(K, n_max=24, omega_ref=nu))
        exact = 0.5 * np.sum(mode_frequencies(K))
        assert abs(E[0] - exact) < 1e-10, f"E_0 = {E[0]:.12f} vs {exact:.12f}"


# ---------------------------------------------------------------------------
# 2. published single-mode anchor: Deffner & Lutz, PRE 77, 021128 (2008)
# ---------------------------------------------------------------------------


class TestDeffnerLutzAnchor:
    def test_q_star_sudden_limit(self):
        """DL Eq. (15) at S = I is (w0^2 + w1^2)/(2 w0 w1)."""
        w0, w1 = 1.3, 2.1
        q = deffner_lutz_q_star(np.eye(2), w0, w1)
        expected = (w0**2 + w1**2) / (2.0 * w0 * w1)
        assert abs(q - expected) < 1e-14, f"Q*_sudden = {q:.15f} vs {expected:.15f}"

    def test_q_star_adiabatic_limit(self):
        """DL Sec. III: a slow ramp drives Q* -> 1 (the adiabatic value)."""
        w0, w1 = 1.3, 2.1
        prev = None
        for tau in (2.0, 8.0, 32.0, 128.0):
            n_steps = 400
            ts = (np.arange(n_steps) + 0.5) / n_steps
            segs = [
                (np.array([[(w0 + (w1 - w0) * t) ** 2]]), tau / n_steps) for t in ts
            ]
            q = deffner_lutz_q_star(protocol_symplectic(segs, 1), w0, w1)
            assert q > 1.0 - 1e-12, f"Q* = {q:.12f} must be >= 1"
            if prev is not None:
                assert q - 1.0 < prev - 1.0, f"Q* - 1 not decreasing: {q - 1.0:.3e}"
            prev = q
        assert prev - 1.0 < 5e-3, f"slow ramp did not approach Q* = 1: {prev:.6f}"

    @pytest.mark.parametrize("proto_name", ["sudden", "driven"])
    def test_gaussian_cf_matches_dl_closed_form(self, proto_name):
        """Determinant route == DL Eq. (17) with Q* from their Eq. (15)."""
        p = PROTO_1_SUDDEN if proto_name == "sudden" else PROTO_1
        w0 = float(mode_frequencies(p.K0)[0])
        w1 = float(mode_frequencies(p.K1)[0])
        q = deffner_lutz_q_star(p.S, w0, w1)
        G_det = gaussian_work_cf(p.K0, p.K1, p.S, p.beta, U_GRID)
        G_dl = deffner_lutz_cf(U_GRID, w0, w1, p.beta, q)
        defect = float(np.max(np.abs(G_det - G_dl)))
        assert defect < 1e-12, (
            f"{proto_name}: |G_det - G_DL| = {defect:.3e} (Q* = {q:.6f})"
        )

    def test_dl_two_segment_drive(self):
        """The Q*-from-S reading also covers multi-segment piecewise drives."""
        w0, w1 = 1.3, 2.1
        segs = [(np.array([[3.0**2]]), 0.37), (np.array([[1.9**2]]), 0.51)]
        S = protocol_symplectic(segs, 1)
        q = deffner_lutz_q_star(S, w0, w1)
        G_det = gaussian_work_cf(
            np.array([[w0**2]]), np.array([[w1**2]]), S, BETA, U_GRID
        )
        G_dl = deffner_lutz_cf(U_GRID, w0, w1, BETA, q)
        defect = float(np.max(np.abs(G_det - G_dl)))
        assert defect < 1e-12, f"two-segment |G_det - G_DL| = {defect:.3e}"

    def test_dl_sudden_helper_agrees_with_q_star_form(self):
        a = deffner_lutz_cf_sudden(U_GRID, 1.3, 2.1, BETA)
        b = deffner_lutz_cf(
            U_GRID, 1.3, 2.1, BETA, deffner_lutz_q_star(np.eye(2), 1.3, 2.1)
        )
        assert np.max(np.abs(a - b)) == 0.0

    def test_dl_eq18_jarzynski_identity(self):
        """DL Eq. (18): G(i beta) = sinh(b e0/2)/sinh(b e1/2), Q*-independent."""
        w0, w1 = 1.3, 2.1
        target = np.sinh(BETA * w0 / 2.0) / np.sinh(BETA * w1 / 2.0)
        for q in (1.0, 1.4, 3.7):
            # DL Eq. (17) analytically continued to mu = i beta
            G = deffner_lutz_cf(np.array([1j * BETA]), w0, w1, BETA, q)[0]
            assert abs(G - target) < 1e-13, (
                f"Q* = {q}: G(i beta) = {G!r} vs sinh ratio {target:.15e}"
            )

    def test_dl_eq30_31_final_occupation_moments(self, tpm1, tpm1_sudden):
        """DL Eqs. (30)/(31) on the dense transition matrix P.

        <m>_n = (n + 1/2) Q* - 1/2 and sigma^2_{m,n} = (Q*^2 - 1)(n^2+n+1)/2
        are exact statements about P_{m,n} alone; they anchor the dense route
        independently of the Gaussian one.
        """
        for tpm, proto in ((tpm1_sudden, PROTO_1_SUDDEN), (tpm1, PROTO_1)):
            w0 = float(mode_frequencies(proto.K0)[0])
            w1 = float(mode_frequencies(proto.K1)[0])
            q = deffner_lutz_q_star(proto.S, w0, w1)
            m = np.arange(tpm.P.shape[0], dtype=float)
            for n in (0, 1, 2):
                col = tpm.P[:, n]
                mean = float(m @ col)
                var = float(((m - mean) ** 2) @ col)
                mean_x, var_x = deffner_lutz_final_n_moments(n, q)
                assert abs(mean - mean_x) < 1e-9, (
                    f"Q*={q:.5f} n={n}: <m> = {mean:.12f} vs DL Eq. (30) {mean_x:.12f}"
                )
                assert abs(var - var_x) < 1e-7, (
                    f"Q*={q:.5f} n={n}: sigma^2 = {var:.10f} vs DL Eq. (31) {var_x:.10f}"
                )


# ---------------------------------------------------------------------------
# 3. THE GATE: dense TPM <-> Gaussian determinant route, 1e-8 (spec M3.5)
# ---------------------------------------------------------------------------


GATE_TOL = 1e-8


def _gate(proto, tpm):
    """Return the four measured route-agreement defects for one protocol."""
    g_mean, g_var = gaussian_work_moments(proto.K0, proto.K1, proto.S, proto.beta)
    d_mean, d_var = dense_work_moments(tpm)
    g_cf = gaussian_work_cf(proto.K0, proto.K1, proto.S, proto.beta, U_GRID)
    d_cf = dense_work_cf(tpm, U_GRID)
    g_jz = gaussian_jarzynski_average(proto.K0, proto.K1, proto.S, proto.beta)
    d_jz = dense_jarzynski_average(tpm)
    return {
        "mean": abs(g_mean - d_mean),
        "variance": abs(g_var - d_var),
        "cf": float(np.max(np.abs(g_cf - d_cf))),
        "jarzynski": abs(g_jz - d_jz),
    }


class TestDenseGaussianGate:
    def test_gate_n1(self, tpm1):
        d = _gate(PROTO_1, tpm1)
        assert max(d.values()) < GATE_TOL, f"N=1 route disagreement: {d}"

    def test_gate_n2(self, tpm2):
        d = _gate(PROTO_2, tpm2)
        assert max(d.values()) < GATE_TOL, f"N=2 route disagreement: {d}"

    def test_gate_n3(self, tpm3):
        d = _gate(PROTO_3, tpm3)
        assert max(d.values()) < GATE_TOL, f"N=3 route disagreement: {d}"

    def test_gate_sudden_quench(self, tpm1_sudden):
        d = _gate(PROTO_1_SUDDEN, tpm1_sudden)
        assert max(d.values()) < GATE_TOL, f"sudden-quench disagreement: {d}"

    def test_trivial_quench_has_zero_work(self):
        """K1 = K0 with no drive: W == 0 identically on both routes."""
        K = _chain(2, 1.1)
        S = protocol_symplectic([], 2)
        mean, var = gaussian_work_moments(K, K, S, BETA)
        assert abs(mean) < 1e-13 and abs(var) < 1e-13, f"mean={mean:.3e} var={var:.3e}"
        G = gaussian_work_cf(K, K, S, BETA, U_GRID)
        assert np.max(np.abs(G - 1.0)) < 1e-11, f"G != 1: {np.max(np.abs(G - 1.0)):.3e}"
        assert abs(gaussian_jarzynski_average(K, K, S, BETA) - 1.0) < 1e-12
        assert abs(delta_free_energy(K, K, BETA)) < 1e-14

    @pytest.mark.parametrize("beta", [0.4, 1.0, 3.0, 6.0])
    def test_gate_across_temperatures(self, beta):
        """The branch tracking has to hold from hot to nearly-ground-state."""
        K0, K1 = np.array([[1.3**2]]), np.array([[2.1**2]])
        segs = [(np.array([[3.0**2]]), 0.6)]
        S = protocol_symplectic(segs, 1)
        # the hot end needs the deeper cutoff: the variance sums P against
        # E1_m^2, so it converges well after the plain tail weight does.
        tpm = dense_tpm(K0, K1, segs, beta=beta, n_max=int(80 + 70.0 / beta))
        g_mean, g_var = gaussian_work_moments(K0, K1, S, beta)
        d_mean, d_var = dense_work_moments(tpm)
        cf = float(
            np.max(
                np.abs(
                    gaussian_work_cf(K0, K1, S, beta, U_GRID) - dense_work_cf(tpm, U_GRID)
                )
            )
        )
        worst = max(abs(g_mean - d_mean), abs(g_var - d_var), cf)
        assert worst < GATE_TOL, (
            f"beta={beta}: mean {abs(g_mean - d_mean):.3e}, "
            f"var {abs(g_var - d_var):.3e}, cf {cf:.3e} "
            f"(tail weight {tpm.tail_weight:.3e})"
        )

    def test_gate_many_segment_drive(self):
        """A finely Trotterized ramp: many segments, both routes."""
        beta = 2.0
        w = 1.2 + 0.9 * np.linspace(0.0, 1.0, 12)
        segs = [(np.array([[wk**2]]), 0.08) for wk in w]
        K0, K1 = np.array([[1.2**2]]), np.array([[2.1**2]])
        S = protocol_symplectic(segs, 1)
        tpm = dense_tpm(K0, K1, segs, beta=beta, n_max=90)
        g_mean, g_var = gaussian_work_moments(K0, K1, S, beta)
        d_mean, d_var = dense_work_moments(tpm)
        cf = float(
            np.max(
                np.abs(
                    gaussian_work_cf(K0, K1, S, beta, U_GRID) - dense_work_cf(tpm, U_GRID)
                )
            )
        )
        worst = max(abs(g_mean - d_mean), abs(g_var - d_var), cf)
        assert worst < GATE_TOL, f"12-segment drive disagreement = {worst:.3e}"

    def test_determinant_cf_matches_wick_moments_at_large_n(self):
        """Extends the gate past N = 3 with two independent derivations.

        The determinant characteristic function (TLH Eq. (3) in metaplectic
        form) and the Wick traces of :func:`gaussian_work_moments` share no
        code beyond the propagator, so their agreement at N = 12 -- where the
        dense route cannot reach -- is real evidence, not a tautology.
        """
        N = 12
        K0, K1 = _chain(N, 0.7), _chain(N, 1.05)
        S = protocol_symplectic([(_chain(N, 1.5), 0.4)], N)
        mean, var = gaussian_work_moments(K0, K1, S, BETA)
        k = gaussian_work_cumulants(K0, K1, S, BETA, order=2, exact_low_order=False)
        assert abs(k[0] - mean) < 1e-6 * abs(mean), (
            f"N=12: CF kappa_1 = {k[0]:.12f} vs Wick mean {mean:.12f}"
        )
        assert abs(k[1] - var) < 1e-6 * abs(var), (
            f"N=12: CF kappa_2 = {k[1]:.12f} vs Wick variance {var:.12f}"
        )

    def test_cf_is_a_characteristic_function(self, tpm2):
        """|G(u)| <= 1, G(0) = 1, G(-u) = conj(G(u)) for a real work density."""
        p = PROTO_2
        us = np.linspace(-6.0, 6.0, 25)
        G = gaussian_work_cf(p.K0, p.K1, p.S, p.beta, us)
        assert abs(G[us.size // 2] - 1.0) < 1e-13
        assert np.max(np.abs(G)) < 1.0 + 1e-12, f"max |G| = {np.max(np.abs(G)):.12f}"
        G_neg = gaussian_work_cf(p.K0, p.K1, p.S, p.beta, -us)
        herm = float(np.max(np.abs(G_neg - np.conj(G))))
        assert herm < 1e-11, f"G(-u) != conj G(u): {herm:.3e}"


# ---------------------------------------------------------------------------
# 4. occupation-cutoff tail test
# ---------------------------------------------------------------------------


class TestTailTest:
    def test_cutoff_doubling_moves_nothing(self):
        """Doubling n_max moves every reported quantity below the gate."""
        p = PROTO_2
        lo = dense_tpm(p.K0, p.K1, p.segments, beta=p.beta, n_max=NMAX[2])
        hi = dense_tpm(p.K0, p.K1, p.segments, beta=p.beta, n_max=2 * NMAX[2])
        m_lo, v_lo = dense_work_moments(lo)
        m_hi, v_hi = dense_work_moments(hi)
        moves = {
            "mean": abs(m_hi - m_lo),
            "variance": abs(v_hi - v_lo),
            "jarzynski": abs(dense_jarzynski_average(hi) - dense_jarzynski_average(lo)),
            "cf": float(np.max(np.abs(dense_work_cf(hi, U_GRID) - dense_work_cf(lo, U_GRID)))),
        }
        assert max(moves.values()) < GATE_TOL, f"cutoff doubling moved results: {moves}"
        assert hi.tail_weight < lo.tail_weight, (
            f"tail weight did not shrink: {lo.tail_weight:.3e} -> {hi.tail_weight:.3e}"
        )

    def test_tail_weight_and_ground_energy_diagnostics_ship(self, tpm2, tpm3):
        for tpm in (tpm2, tpm3):
            assert tpm.tail_weight < 1e-10, (
                f"N={tpm.n_modes} n_max={tpm.n_max}: tail weight "
                f"{tpm.tail_weight:.3e} (initial {tpm.tail_weight_initial:.3e}, "
                f"final {tpm.tail_weight_final:.3e})"
            )
            assert tpm.ground_energy_error < 1e-10, (
                f"truncated E_0 error = {tpm.ground_energy_error:.3e}"
            )
            assert tpm.tail_weight == max(
                tpm.tail_weight_initial, tpm.tail_weight_final
            )

    def test_too_small_cutoff_is_flagged_not_silently_wrong(self):
        """The canary: a starved cutoff must show up in the diagnostics."""
        p = PROTO_1
        bad = dense_tpm(p.K0, p.K1, p.segments, beta=p.beta, n_max=3)
        g_mean, _ = gaussian_work_moments(p.K0, p.K1, p.S, p.beta)
        d_mean, _ = dense_work_moments(bad)
        err = abs(d_mean - g_mean)
        assert err > 1e-3, f"n_max=3 was accidentally accurate: err = {err:.3e}"
        assert bad.tail_weight > 1e-3, (
            f"starved cutoff not flagged: tail weight {bad.tail_weight:.3e} "
            f"while the mean is off by {err:.3e}"
        )

    def test_truncated_log_partition_converges_to_exact(self):
        p = PROTO_2
        exact = log_partition_gaussian(p.K0, p.beta)
        errs = []
        for n_max in (6, 12, 18):
            t = dense_tpm(p.K0, p.K1, p.segments, beta=p.beta, n_max=n_max)
            errs.append(abs(t.log_Z0 - exact))
        assert errs[-1] < 1e-11, f"truncated ln Z0 error {errs[-1]:.3e}"
        assert errs[0] > errs[1] > errs[2], f"ln Z0 not converging: {errs}"

    def test_dense_P_is_doubly_stochastic(self, tpm2):
        rows = np.max(np.abs(tpm2.P.sum(axis=1) - 1.0))
        cols = np.max(np.abs(tpm2.P.sum(axis=0) - 1.0))
        assert max(rows, cols) < 1e-12, f"P not doubly stochastic: {rows:.3e}/{cols:.3e}"

    def test_reference_frequency_beats_the_unit_basis(self):
        """The omega_ref choice is load-bearing, so pin the improvement."""
        p = PROTO_3
        g_mean, _ = gaussian_work_moments(p.K0, p.K1, p.S, p.beta)
        matched = dense_tpm(p.K0, p.K1, p.segments, beta=p.beta, n_max=8)
        unit = dense_tpm(
            p.K0, p.K1, p.segments, beta=p.beta, n_max=8, omega_ref=1.0
        )
        e_matched = abs(dense_work_moments(matched)[0] - g_mean)
        e_unit = abs(dense_work_moments(unit)[0] - g_mean)
        assert e_matched < 0.01 * e_unit, (
            f"omega_ref={matched.omega_ref:.4f} error {e_matched:.3e} vs "
            f"omega_ref=1 error {e_unit:.3e}"
        )


# ---------------------------------------------------------------------------
# 5. Jarzynski (spec: 1e-10 dense, 1e-6 Gaussian)
# ---------------------------------------------------------------------------


class TestJarzynski:
    @pytest.mark.parametrize("which", [1, 2, 3])
    def test_dense_jarzynski(self, which, tpm1, tpm2, tpm3):
        proto = {1: PROTO_1, 2: PROTO_2, 3: PROTO_3}[which]
        tpm = {1: tpm1, 2: tpm2, 3: tpm3}[which]
        defect = jarzynski_defect(
            dense_jarzynski_average(tpm), proto.K0, proto.K1, proto.beta
        )
        assert defect < 1e-10, f"N={which} dense Jarzynski defect = {defect:.3e}"

    @pytest.mark.parametrize("which", [1, 2, 3])
    def test_gaussian_jarzynski(self, which):
        proto = {1: PROTO_1, 2: PROTO_2, 3: PROTO_3}[which]
        avg = gaussian_jarzynski_average(proto.K0, proto.K1, proto.S, proto.beta)
        defect = jarzynski_defect(avg, proto.K0, proto.K1, proto.beta)
        assert defect < 1e-6, f"N={which} Gaussian Jarzynski defect = {defect:.3e}"

    def test_gaussian_jarzynski_large_n(self):
        """The Gaussian route is the one that has to work at large N."""
        K0, K1 = _chain(24, 0.6), _chain(24, 0.95)
        S = protocol_symplectic([(_chain(24, 1.4), 0.3)], 24)
        avg = gaussian_jarzynski_average(K0, K1, S, BETA)
        defect = jarzynski_defect(avg, K0, K1, BETA)
        assert defect < 1e-6, f"N=24 Jarzynski defect = {defect:.3e}"

    def test_second_law_margin_nonnegative(self):
        """<W> >= DeltaF (Jensen on the Jarzynski equality), every protocol."""
        rng = np.random.default_rng(20260901)
        for _ in range(8):
            N = int(rng.integers(1, 5))
            m0, m1, mm = rng.uniform(0.4, 1.6, size=3)
            t = float(rng.uniform(0.05, 1.2))
            beta = float(rng.uniform(0.5, 3.0))
            K0, K1 = _chain(N, m0), _chain(N, m1)
            S = protocol_symplectic([(_chain(N, mm), t)], N)
            mean, _ = gaussian_work_moments(K0, K1, S, beta)
            dF = delta_free_energy(K0, K1, beta)
            assert mean - dF > -1e-12, (
                f"second law violated: <W> - DeltaF = {mean - dF:.3e} "
                f"(N={N}, beta={beta:.3f})"
            )

    def test_jarzynski_defect_catches_a_wrong_average(self):
        avg = gaussian_jarzynski_average(
            PROTO_1.K0, PROTO_1.K1, PROTO_1.S, PROTO_1.beta
        )
        assert jarzynski_defect(1.01 * avg, PROTO_1.K0, PROTO_1.K1, PROTO_1.beta) > 1e-3


# ---------------------------------------------------------------------------
# 6. Crooks symmetry on forward/reverse pairs
# ---------------------------------------------------------------------------


class TestCrooks:
    def test_reverse_transition_matrix_is_the_transpose(self, tpm2, tpm2_rev):
        """P_R = P_F^T for real Hamiltonians (module docstring, Crooks part)."""
        defect = float(np.max(np.abs(tpm2_rev.P - tpm2.P.T)))
        assert defect < 1e-12, f"|P_R - P_F^T| = {defect:.3e}"
        assert np.max(np.abs(tpm2_rev.E0 - tpm2.E1)) < 1e-12
        assert np.max(np.abs(tpm2_rev.E1 - tpm2.E0)) < 1e-12

    def test_dense_crooks_transition_resolved(self, tpm2, tpm2_rev):
        defect = crooks_max_defect(tpm2, tpm2_rev)
        assert defect < 1e-10, f"max relative Crooks defect = {defect:.3e}"

    def test_dense_crooks_sudden_quench(self, tpm1_sudden):
        p = PROTO_1_SUDDEN
        rev = dense_tpm(
            p.K1,
            p.K0,
            (),
            beta=p.beta,
            n_max=NMAX[1],
            omega_ref=tpm1_sudden.omega_ref,
        )
        defect = crooks_max_defect(tpm1_sudden, rev)
        assert defect < 1e-10, f"sudden-quench Crooks defect = {defect:.3e}"

    @pytest.mark.parametrize("which", [1, 2, 3])
    def test_gaussian_crooks_cf_identity(self, which):
        """G_F(u) = G_R(-u + i beta) e^{-beta DeltaF}, no truncation anywhere."""
        p = {1: PROTO_1, 2: PROTO_2, 3: PROTO_3}[which]
        defect = gaussian_crooks_defect(
            p.K0, p.K1, p.segments, p.beta, us=np.linspace(-1.5, 1.5, 7)
        )
        assert defect < 1e-10, f"N={which} Gaussian Crooks defect = {defect:.3e}"

    def test_crooks_detects_a_mismatched_reverse_protocol(self):
        """Running the reverse leg with the WRONG drive must break Crooks."""
        p = PROTO_2
        fwd = dense_tpm(p.K0, p.K1, p.segments, beta=p.beta, n_max=10)
        nu = fwd.omega_ref
        wrong = dense_tpm(
            p.K1, p.K0, [(_chain(2, 2.1), 0.5)], beta=p.beta, n_max=10, omega_ref=nu
        )
        defect = crooks_max_defect(fwd, wrong)
        assert defect > 1e-3, f"mismatched reverse leg not detected: {defect:.3e}"

    def test_crooks_rejects_mismatched_inputs(self, tpm2):
        other = dense_tpm(
            PROTO_2.K1, PROTO_2.K0, reverse_segments(PROTO_2.segments),
            beta=PROTO_2.beta * 1.5, n_max=NMAX[2],
        )
        with pytest.raises(ValueError, match="beta mismatch"):
            crooks_max_defect(tpm2, other)
        small = dense_tpm(
            PROTO_2.K1, PROTO_2.K0, reverse_segments(PROTO_2.segments),
            beta=PROTO_2.beta, n_max=6,
        )
        with pytest.raises(ValueError, match="truncation mismatch"):
            crooks_max_defect(tpm2, small)


# ---------------------------------------------------------------------------
# 7a. cumulants (the beyond-mean content the ledger carries)
# ---------------------------------------------------------------------------


class TestCumulants:
    def test_finite_difference_kappa12_match_the_exact_wick_moments(self):
        """The FD machinery itself, checked against the exact Wick values.

        (Production calls take kappa_1, kappa_2 from the closed form; this
        turns that redundancy into a test of the differencing.)
        """
        for p in (PROTO_1, PROTO_2, PROTO_3):
            k = gaussian_work_cumulants(
                p.K0, p.K1, p.S, p.beta, exact_low_order=False
            )
            mean, var = gaussian_work_moments(p.K0, p.K1, p.S, p.beta)
            assert abs(k[0] - mean) < 1e-6 * abs(mean), (
                f"N={p.N}: kappa_1 = {k[0]:.12f} vs Wick mean {mean:.12f}"
            )
            assert abs(k[1] - var) < 1e-6 * abs(var), (
                f"N={p.N}: kappa_2 = {k[1]:.12f} vs Wick variance {var:.12f}"
            )

    def test_exact_low_order_is_exactly_the_wick_result(self):
        p = PROTO_3
        k = gaussian_work_cumulants(p.K0, p.K1, p.S, p.beta)
        mean, var = gaussian_work_moments(p.K0, p.K1, p.S, p.beta)
        assert k[0] == mean and k[1] == var

    @pytest.mark.parametrize("which", [1, 2, 3])
    def test_kappa34_match_the_dense_route(self, which, tpm1, tpm2, tpm3):
        p = {1: PROTO_1, 2: PROTO_2, 3: PROTO_3}[which]
        tpm = {1: tpm1, 2: tpm2, 3: tpm3}[which]
        kg = gaussian_work_cumulants(p.K0, p.K1, p.S, p.beta)
        kd = dense_work_cumulants(tpm)
        for n in (3, 4):
            rel = abs(kg[n - 1] - kd[n - 1]) / abs(kd[n - 1])
            assert rel < 2e-5, (
                f"N={which} kappa_{n}: Gaussian {kg[n - 1]:.9f} vs dense "
                f"{kd[n - 1]:.9f} (rel {rel:.3e})"
            )

    def test_cumulant_step_refinement_converges(self):
        """The raw stencils are O(h^4): halving h must shrink the error ~16x."""
        p = PROTO_1
        _, var = gaussian_work_moments(p.K0, p.K1, p.S, p.beta)
        w_max = float(np.max(mode_frequencies(p.K1)))
        errs = []
        for factor in (0.16, 0.08, 0.04):
            k = gaussian_work_cumulants(
                p.K0,
                p.K1,
                p.S,
                p.beta,
                h=factor / w_max,
                richardson=False,
                exact_low_order=False,
            )
            errs.append(abs(k[1] - var))
        ratios = [errs[0] / errs[1], errs[1] / errs[2]]
        assert min(ratios) > 12.0, f"kappa_2 not converging at O(h^4): {errs} {ratios}"

    def test_richardson_beats_the_raw_stencil(self):
        p = PROTO_1
        _, var = gaussian_work_moments(p.K0, p.K1, p.S, p.beta)
        common = dict(exact_low_order=False)
        raw = gaussian_work_cumulants(
            p.K0, p.K1, p.S, p.beta, richardson=False, **common
        )
        rich = gaussian_work_cumulants(p.K0, p.K1, p.S, p.beta, **common)
        e_raw, e_rich = abs(raw[1] - var), abs(rich[1] - var)
        assert e_rich < 0.1 * e_raw, (
            f"Richardson step-extrapolation did not help: {e_raw:.3e} -> {e_rich:.3e}"
        )

    def test_cumulant_order_validation(self, tpm1):
        with pytest.raises(ValueError, match="order"):
            gaussian_work_cumulants(PROTO_1.K0, PROTO_1.K1, PROTO_1.S, BETA, order=5)
        with pytest.raises(ValueError, match="order"):
            dense_work_cumulants(tpm1, order=0)
        with pytest.raises(ValueError, match="h must be positive"):
            gaussian_work_cumulants(PROTO_1.K0, PROTO_1.K1, PROTO_1.S, BETA, h=-1.0)
        assert gaussian_work_cumulants(
            PROTO_1.K0, PROTO_1.K1, PROTO_1.S, BETA, order=2
        ).shape == (2,)


# ---------------------------------------------------------------------------
# 7b. numerical-hazard gates and purity
# ---------------------------------------------------------------------------


class TestNumericalHazards:
    def test_branch_tracking_step_halving(self):
        """The CF must be independent of the branch-tracking path resolution."""
        p = PROTO_3
        us = np.linspace(-3.0, 3.0, 13)
        coarse = gaussian_work_cf(p.K0, p.K1, p.S, p.beta, us, path_step=0.05)
        fine = gaussian_work_cf(p.K0, p.K1, p.S, p.beta, us, path_step=0.025)
        finer = gaussian_work_cf(p.K0, p.K1, p.S, p.beta, us, path_step=0.0125)
        d1 = float(np.max(np.abs(coarse - fine)))
        d2 = float(np.max(np.abs(fine - finer)))
        assert max(d1, d2) < 1e-12, f"path-step dependence: {d1:.3e}, {d2:.3e}"

    def test_underresolved_path_raises(self):
        """A coarse path must be refused up front, never silently aliased.

        det(T(u) - I) is a Laurent polynomial in the phases e^{i u w_k}, so
        its argument winds at most at rate = sum of both spectra; a step
        above 2/rate could alias past np.unwrap's pi limit and is rejected.
        """
        p = PROTO_3
        rate = float(
            np.sum(mode_frequencies(p.K0)) + np.sum(mode_frequencies(p.K1))
        )
        ok_step = 1.5 / rate
        bad_step = 4.0 / rate
        gaussian_work_cf(p.K0, p.K1, p.S, p.beta, np.array([-2.0, 2.0]), path_step=ok_step)
        with pytest.raises(ValueError, match="under-resolved"):
            gaussian_work_cf(
                p.K0, p.K1, p.S, p.beta, np.array([-2.0, 2.0]), path_step=bad_step
            )

    def test_cf_at_matches_the_grid_route(self):
        p = PROTO_2
        for u in (-1.3, 0.0, 0.7, 2.2):
            a = gaussian_work_cf_at(p.K0, p.K1, p.S, p.beta, u)
            b = gaussian_work_cf(p.K0, p.K1, p.S, p.beta, [u])[0]
            assert abs(a - b) < 1e-12, f"u={u}: cf_at {a!r} vs cf {b!r}"

    def test_no_input_mutation(self, tpm2):
        """Pure-functional discipline (Layer 6 JAX mirror pre-positioning)."""
        p = PROTO_2
        K0 = p.K0.copy()
        K1 = p.K1.copy()
        segs = [(K.copy(), dt) for K, dt in p.segments]
        S = p.S.copy()
        gaussian_work_moments(K0, K1, S, p.beta)
        gaussian_work_cf(K0, K1, S, p.beta, U_GRID)
        counting_statistics(K0, K1, segs, p.beta)
        dense_work_moments(tpm2)
        dense_work_quantile(tpm2, 0.5)
        assert np.array_equal(K0, p.K0)
        assert np.array_equal(K1, p.K1)
        assert np.array_equal(S, p.S)
        assert all(np.array_equal(a[0], b[0]) for a, b in zip(segs, p.segments))

    def test_segments_may_be_a_one_shot_iterator(self):
        """Generators must not be silently consumed halfway through."""
        p = PROTO_2
        gen = ((K, dt) for K, dt in p.segments)
        stats = counting_statistics(p.K0, p.K1, gen, p.beta)
        ref = counting_statistics(p.K0, p.K1, p.segments, p.beta)
        assert abs(stats["mean"] - ref["mean"]) < 1e-14

    def test_input_validation(self):
        K = _chain(2, 1.0)
        with pytest.raises(ValueError, match="beta must be positive"):
            log_partition_gaussian(K, 0.0)
        with pytest.raises(ValueError, match="positive definite"):
            mode_frequencies(np.array([[1.0, 2.0], [2.0, 1.0]]))
        with pytest.raises(ValueError, match="N <= 3"):
            dense_tpm(_chain(4, 1.0), _chain(4, 1.1), beta=BETA, n_max=4)
        with pytest.raises(ValueError, match="n_max must be >= 2"):
            dense_tpm(K, K, beta=BETA, n_max=1)
        with pytest.raises(ValueError, match="max_dim"):
            dense_tpm(K, K, beta=BETA, n_max=200)
        with pytest.raises(ValueError, match="route must be"):
            counting_statistics(K, K, beta=BETA, route="montecarlo")
        with pytest.raises(ValueError, match="single-mode"):
            deffner_lutz_q_star(np.eye(4), 1.0, 1.0)


# ---------------------------------------------------------------------------
# 8. distribution utilities and counting_statistics
# ---------------------------------------------------------------------------


class TestDistributionAndSummary:
    def test_distribution_normalizes_and_reproduces_moments(self, tpm2):
        W, q = dense_work_distribution(tpm2)
        assert abs(q.sum() - 1.0) < 1e-12, f"sum q = {q.sum():.15f}"
        assert np.all(np.diff(W) >= 0.0)
        mean, var = dense_work_moments(tpm2)
        assert abs(float(q @ W) - mean) < 1e-12
        assert abs(float(q @ (W - mean) ** 2) - var) < 1e-10

    def test_quantiles_are_monotone_and_bracketed(self, tpm2):
        W, _ = dense_work_distribution(tpm2)
        probs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        qs = dense_work_quantile(tpm2, probs)
        assert np.all(np.diff(qs) >= 0.0), f"quantiles not monotone: {qs}"
        assert W.min() - 1e-12 <= qs[0] and qs[-1] <= W.max() + 1e-12
        assert isinstance(dense_work_quantile(tpm2, 0.5), float)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            dense_work_quantile(tpm2, 1.4)

    def test_counting_statistics_contract(self, tpm2):
        p = PROTO_2
        for route, kwargs in (("gaussian", {}), ("dense", {"n_max": NMAX[2]})):
            stats = counting_statistics(
                p.K0, p.K1, p.segments, p.beta, route=route, **kwargs
            )
            for key in COUNTING_STATISTICS_KEYS:
                assert key in stats, f"{route}: missing key {key!r}"
            for key in WORK_STATS_REQUIRED_KEYS:
                assert key in stats, f"{route}: ledger contract missing {key!r}"
            assert stats["jarzynski_defect"] < 1e-10, (
                f"{route}: Jarzynski defect {stats['jarzynski_defect']:.3e}"
            )
            assert stats["second_law_margin"] > 0.0
            assert abs(stats["std"] ** 2 - stats["variance"]) < 1e-12
            assert abs(stats["delta_F"] - p.dF) < 1e-14

    def test_counting_statistics_routes_agree(self):
        p = PROTO_2
        g = counting_statistics(p.K0, p.K1, p.segments, p.beta, route="gaussian")
        d = counting_statistics(
            p.K0, p.K1, p.segments, p.beta, route="dense", n_max=NMAX[2]
        )
        for key in ("mean", "variance", "jarzynski_average"):
            assert abs(g[key] - d[key]) < GATE_TOL, (
                f"{key}: gaussian {g[key]:.12f} vs dense {d[key]:.12f}"
            )

    def test_counting_statistics_extras(self):
        p = PROTO_2
        d = counting_statistics(
            p.K0,
            p.K1,
            p.segments,
            p.beta,
            route="dense",
            n_max=12,
            cumulants=True,
            quantiles=(0.5, 0.9),
        )
        assert {"kappa_3", "kappa_4", "skewness", "excess_kurtosis"} <= set(d)
        assert d["work_q0.5"] <= d["work_q0.9"]
        assert d["tail_weight"] == max(d["tail_weight_initial"], d["tail_weight_final"])
        g = counting_statistics(
            p.K0, p.K1, p.segments, p.beta, route="gaussian", cumulants=True
        )
        assert abs(g["kappa_3"] - d["kappa_3"]) < 1e-3 * abs(d["kappa_3"])
        with pytest.raises(ValueError, match="quantiles"):
            counting_statistics(
                p.K0, p.K1, p.segments, p.beta, route="gaussian", quantiles=(0.5,)
            )


# ---------------------------------------------------------------------------
# 9. EnergyLedger wiring: counting statistics as the beyond-mean refinement
# ---------------------------------------------------------------------------


class TestLedgerRefinement:
    def test_refine_work_records_and_closes(self):
        """A drive instrumented purely through counting statistics closes.

        The physical content: for a piecewise-constant drive on a Gibbs
        initial state, the TPM mean work <W> is exactly the mean-energy
        change of the state (the drive is the only energy source), so the
        ledger's snapshots and the work distribution have to agree.
        """
        p = PROTO_2
        V0 = thermal_state_cov(p.K0, 1.0 / p.beta)
        V1 = p.S @ V0 @ p.S.T

        led = energy_ledger(tol=1e-9)
        E0 = led.record("thermal in", V0, K=p.K0)
        stats = counting_statistics(p.K0, p.K1, p.segments, p.beta)
        led.refine_work("drive", stats)
        E1 = led.record("out", V1, K=p.K1)

        assert abs((E1 - E0) - stats["mean"]) < 1e-10, (
            f"Delta<H> = {E1 - E0:.12f} vs TPM <W> = {stats['mean']:.12f}"
        )
        res = led.close()
        assert res.passed, f"ledger did not close: {res}"
        assert abs(res.details["defect"]) < 1e-9

        kinds = [r["kind"] for r in res.details["table"]]
        assert kinds == ["snapshot", "work", "work_stats", "snapshot"]
        assert "drive" in res.details["work_statistics"]
        assert res.details["work_statistics"]["drive"]["std"] == stats["std"]

    def test_refine_work_cross_checks_an_existing_work_entry(self):
        p = PROTO_1
        stats = counting_statistics(p.K0, p.K1, p.segments, p.beta)
        led = energy_ledger(K=p.K0, tol=1e-9)
        led.work("drive", stats["mean"])
        led.refine_work("drive", stats)  # consistent: no extra work row
        kinds = [r["kind"] for r in led.table]
        assert kinds == ["work", "work_stats"]

    def test_refine_work_flags_a_mean_that_disagrees(self):
        p = PROTO_1
        stats = counting_statistics(p.K0, p.K1, p.segments, p.beta)
        led = energy_ledger(K=p.K0, tol=1e-9)
        led.work("drive", stats["mean"] + 1e-3)
        with pytest.raises(AuditViolation) as exc:
            led.refine_work("drive", stats)
        result = exc.value.result
        assert result.name == "work_statistics_consistency"
        assert abs(result.details["defect"] - 1e-3) < 1e-9

    def test_refine_work_requires_the_full_contract(self):
        led = energy_ledger(tol=1e-9)
        with pytest.raises(ValueError, match="missing required keys"):
            led.refine_work("drive", {"mean": 1.0})
        with pytest.raises(TypeError):
            led.refine_work("drive", 3.0)

    def test_work_statistics_view_is_a_copy(self):
        p = PROTO_1
        led = energy_ledger(K=p.K0)
        stats = counting_statistics(p.K0, p.K1, p.segments, p.beta)
        led.refine_work("drive", stats)
        view = led.work_statistics
        view["drive"]["mean"] = 12345.0
        assert led.work_statistics["drive"]["mean"] == stats["mean"]
        stats["mean"] = 999.0  # mutating the caller's dict must not leak in
        assert led.work_statistics["drive"]["mean"] != 999.0

    def test_ledger_repr_survives_a_stats_row(self):
        p = PROTO_1
        led = energy_ledger(K=p.K0)
        led.refine_work("drive", counting_statistics(p.K0, p.K1, p.segments, p.beta))
        text = repr(led)
        assert "work_stats" in text and "std=" in text
