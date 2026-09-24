"""Tests for vacuum.hardware — "hardware as noisy field" (PLAN.md Layer 7, L6).

The whole file needs the ``.[ibm]`` extra (qiskit + qiskit-aer) and is
skipped without it; the dense arbiters it exercises are core-only.

Anchors (every measured number is printed in the assertion message):

1. Encoding.  The Pauli decomposition of the Galerkin-truncated chain
   equals its dense matrix (1e-12); the on-site term is exactly
   omega (n + 1/2) at every d; the truncated vacuum converges to
   ``vacuum.core.ground_state_cov(K)`` monotonically in d (Rayleigh–Ritz:
   E_0(d) >= E_0 exact) with covariance error < 1e-9 at d = 8 and at
   roundoff by d = 16 for N = 2; for N = 2, d = 2 the encoded field IS
   Hotta's minimal QET model — E_A, E_B*, theta* against the closed forms
   of vacuum.qet.hotta (review arXiv:1101.3954 Eqs. (8), (11), (9)-(10))
   to 1e-12 / 1e-8.
2. VQE.  Aer's statevector energy equals the numpy arbiter (1e-12); the
   hardware-efficient ansatz reaches the ED ground energy of the same
   truncated H to 1e-10 (N = 3, d = 2) / 1e-8 (N = 2, d = 4), the
   truncated covariance to 1e-7 / 1e-5, and the Gaussian vacuum to the
   encoding's own error (< 6e-2 at d = 2, < 1e-4 at d = 4).
3. QET.  Dense ledger: closure (locality of the bookkeeping) 1e-12,
   E_A >= E_B, no-signaling 1e-12, cut channel E_B <= 0, wrong-bit
   consistency.  Noiseless Aer (native gates, density matrix) reproduces
   the dense ledger to 1e-10 and passes the audits; deferred and
   feed-forward sampled runs agree with the dense E_B within 4 sigma;
   tensored readout mitigation undoes pure readout flips to 1e-8.
4. Harvesting.  Noiseless Aer reproduces the dense Trotter mirror's
   rho_AB to 1e-10 (d = 2 and d = 4); the product formula converges to
   the RK4-exact evolution; the two-qubit negativity helper is exact on
   Bell/product states; at weak coupling the qubit-detector E_N agrees
   with the Gaussian oscillator-detector stack (MAP-1) to 1 % at d = 4
   and 10 % at d = 2, improving with d.
5. Noise.  The calibration point is read from the Ikeda file (every
   number cited); the 2-qubit model reproduces vacuum.qet.ikeda_qiskit's
   exactly; scaled points are labelled synthetic; the sweep file passes
   provenance; a 3-point smoke of the survival sweep (alive/alive/dead)
   and a short bisection; the device on-ramp runs on an Aer backend and a
   SamplerV2.
"""

import math

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from vacuum import experiments_io as xio  # noqa: E402
from vacuum.audits import AuditViolation  # noqa: E402
from vacuum.hardware import noise as nz  # noqa: E402
from vacuum.hardware import protocols as pr  # noqa: E402
from vacuum.hardware import vacuum_sim as vs  # noqa: E402
from vacuum.qet.hotta import e_a_closed, e_b_optimal, theta_optimal  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chain32():
    return vs.truncated_chain(3, 1.0, 2)


@pytest.fixture(scope="module")
def chain24():
    return vs.truncated_chain(2, 1.0, 4)


@pytest.fixture(scope="module")
def vqe32(chain32):
    return vs.vqe_ground_state(chain32, layers=2, restarts=3, seed=0, evaluator="aer")


@pytest.fixture(scope="module")
def vqe24(chain24):
    return vs.vqe_ground_state(chain24, layers=2, restarts=3, seed=0, evaluator="aer")


@pytest.fixture(scope="module")
def qet32(chain32, vqe32):
    theta, E_B = pr.qet_optimize_theta(chain32, vqe32.state, 0, 1)
    dense = pr.qet_dense(chain32, vqe32.state, 0, 1, theta)
    return {"theta": theta, "E_B": E_B, "dense": dense}


@pytest.fixture(scope="module")
def hspec():
    return pr.harvest_spec((0, 1), 1.5, 1.0, width=1.5, n_steps=6)


@pytest.fixture(scope="module")
def calib():
    return nz.calibration_point()


# ---------------------------------------------------------------------------
# 1. encoding
# ---------------------------------------------------------------------------


class TestEncoding:
    @pytest.mark.parametrize("N,d", [(2, 2), (3, 2), (2, 4), (3, 4), (4, 2)])
    def test_pauli_terms_equal_dense(self, N, d):
        ch = vs.truncated_chain(N, 1.0, d)
        H = vs.hamiltonian_dense(ch)
        Hp = vs.pauli_terms_dense(vs.hamiltonian_pauli_terms(ch), ch.n_qubits)
        err = np.max(np.abs(H - Hp))
        assert err < 1e-12, f"Pauli decomposition vs dense H ({N}, {d}): {err:.3e}"
        assert np.max(np.abs(H - H.conj().T)) < 1e-14
        for s in range(N):
            Ht = vs.touched_hamiltonian_dense(ch, s)
            Htp = vs.pauli_terms_dense(vs.touched_pauli_terms(ch, s), ch.n_qubits)
            assert np.max(np.abs(Ht - Htp)) < 1e-12

    @pytest.mark.parametrize("d", [2, 3, 4, 8])
    def test_galerkin_onsite_is_harmonic(self, d):
        """P (p^2/2 + omega^2 x^2/2) P = omega (n + 1/2) exactly."""
        om = 1.7
        ops = vs.site_operators(d, om)
        target = om * (ops["n"] + 0.5 * np.eye(d))
        err = np.max(np.abs(ops["h_site"] - target))
        assert err < 1e-13, f"on-site Galerkin term at d={d}: {err:.3e}"

    def test_encoding_converges_to_gaussian(self):
        rows = vs.encoding_convergence(2, 1.0, ds=(2, 4, 8, 16))
        dE = [r["dE"] for r in rows]
        cov = [r["cov_err"] for r in rows]
        assert all(x >= -1e-12 for x in dE), f"Rayleigh-Ritz violated: {dE}"
        assert all(dE[i + 1] <= dE[i] for i in range(3)), f"dE not monotone: {dE}"
        assert all(cov[i + 1] <= cov[i] for i in range(3)), f"cov_err not monotone: {cov}"
        assert cov[2] < 1e-9, f"N=2 cov error at d=8: {cov[2]:.3e}"
        assert cov[3] < 1e-12, f"N=2 cov error at d=16: {cov[3]:.3e}"
        rows3 = vs.encoding_convergence(3, 1.0, ds=(2, 4, 8))
        c3 = [r["cov_err"] for r in rows3]
        assert c3[0] < 6e-2 and c3[1] < 1e-3 and c3[2] < 1e-7, f"N=3 cov errors: {c3}"

    @pytest.mark.parametrize("m", [1.0, 0.3, 0.0])
    def test_n2_d2_is_hottas_minimal_model(self, m):
        ch = vs.truncated_chain(2, m, 2)
        E0, psi = vs.exact_ground_state(ch)
        om = ch.omega[0]
        h, k = om / 2.0, 1.0 / (4.0 * om)
        theta, E_B = pr.qet_optimize_theta(ch, psi, 0, 1)
        led = pr.qet_dense(ch, psi, 0, 1, theta)
        assert abs(led["E_A"] - e_a_closed(h, k)) < 1e-12, (m, led["E_A"], e_a_closed(h, k))
        assert abs(E_B - e_b_optimal(h, k)) < 1e-12, (m, E_B, e_b_optimal(h, k))
        th_ref = theta_optimal(h, k) / math.sqrt(om / 2.0)
        assert abs(theta - th_ref) < 1e-8, (m, theta, th_ref)


# ---------------------------------------------------------------------------
# 2. VQE
# ---------------------------------------------------------------------------


class TestVQE:
    def test_ansatz_numpy_equals_qiskit_statevector(self):
        Q = vs._qiskit()
        rng = np.random.default_rng(1)
        for n, L in [(2, 1), (3, 2), (4, 2), (6, 3)]:
            qc, th = vs.hea_circuit(n, L)
            p = rng.normal(size=len(th))
            sv = Q.Statevector(qc.assign_parameters({th[k]: p[k] for k in range(len(th))})).data
            err = np.max(np.abs(sv - vs.hea_statevector(p, n, L)))
            assert err < 1e-12, f"ansatz numpy vs Statevector (n={n}, L={L}): {err:.3e}"

    def test_vqe_reaches_ed_and_gaussian_limit(self, vqe32, vqe24):
        r = vqe32
        assert r.evaluator == "aer" and r.aer_vs_numpy < 1e-12, r.aer_vs_numpy
        assert abs(r.energy_error) < 1e-10, f"N=3 d=2 VQE energy error {r.energy_error:.3e}"
        assert r.cov_error < 1e-7, f"N=3 d=2 VQE covariance error {r.cov_error:.3e}"
        assert 1.0 - r.fidelity < 1e-10, f"N=3 d=2 infidelity {1 - r.fidelity:.3e}"
        assert r.cov_error_gaussian < 6e-2, f"d=2 vs Gaussian vacuum {r.cov_error_gaussian:.3e}"
        r = vqe24
        assert abs(r.energy_error) < 1e-8, f"N=2 d=4 VQE energy error {r.energy_error:.3e}"
        assert r.cov_error < 1e-5, f"N=2 d=4 VQE covariance error {r.cov_error:.3e}"
        assert r.cov_error_gaussian < 1e-4, f"d=4 vs Gaussian vacuum {r.cov_error_gaussian:.3e}"
        assert vqe32.cov_error_gaussian > vqe24.cov_error_gaussian


# ---------------------------------------------------------------------------
# 3. QET
# ---------------------------------------------------------------------------


class TestQET:
    def test_dense_ledger_and_audits(self, chain32, vqe32, qet32):
        led = qet32["dense"]
        assert led["ledger_defect"] < 1e-12, f"ledger defect {led['ledger_defect']:.3e}"
        assert led["no_signaling"] < 1e-12, f"no-signaling {led['no_signaling']:.3e}"
        assert led["E_A"] >= led["E_B"] > 0, (led["E_A"], led["E_B"])
        assert led["E_B_cut"] <= 1e-12, f"cut-channel E_B {led['E_B_cut']:.3e}"
        assert abs(led["E_B_info"] - (led["E_B"] - led["E_B_wrong"])) < 1e-12
        cut = pr.qet_dense(chain32, vqe32.state, 0, 1, qet32["theta"], channel_open=False)
        assert abs(cut["E_B"] - led["E_B_cut"]) < 1e-9, (cut["E_B"], led["E_B_cut"])
        pr.qet_audits(led, tol=1e-12, ledger_tol=1e-12, signaling_tol=1e-12)
        bad = dict(led, E_B=led["E_A"] + 1.0)
        with pytest.raises(AuditViolation):
            pr.qet_audits(bad)

    def test_noiseless_aer_matches_dense(self, chain32, vqe32, qet32, chain24, vqe24):
        dense = qet32["dense"]
        ex = pr.qet_exact_aer(chain32, vqe32.params, 2, 0, 1, qet32["theta"], ideal=dense, shots=100000)
        for key in ("E_A", "HB_ground", "HB_meas", "HB_final", "HB_wrong", "E_B", "E_B_raw",
                    "E_B_info", "E_B_cut"):
            assert abs(ex[key] - dense[key]) < 1e-10, (key, ex[key], dense[key])
        assert ex["no_signaling"] < 1e-10
        pr.qet_audits(ex, tol=1e-10, ledger_tol=1e-10, signaling_tol=1e-10)
        assert ex["cx_count"] >= 4 and ex["n_circuits"] >= 5
        # d = 4 (two qubits per site): the sign measurement and displacement
        # are genuine 2-qubit unitaries
        th4, _ = pr.qet_optimize_theta(chain24, vqe24.state, 0, 1)
        d4 = pr.qet_dense(chain24, vqe24.state, 0, 1, th4)
        ex4 = pr.qet_exact_aer(chain24, vqe24.params, 2, 0, 1, th4, ideal=d4)
        for key in ("E_A", "E_B", "E_B_info"):
            assert abs(ex4[key] - d4[key]) < 1e-10, (key, ex4[key], d4[key])
        assert ex4["no_signaling"] < 1e-10

    @pytest.mark.parametrize("form", ["deferred", "feedforward"])
    def test_sampled_forms_agree_with_dense(self, chain32, vqe32, qet32, form):
        from qiskit_aer import AerSimulator

        dense = qet32["dense"]
        sm = pr.qet_sampled(chain32, vqe32.params, 2, 0, 1, qet32["theta"], AerSimulator(),
                            shots=40000, seed=3, form=form, ideal=dense)
        z = (sm["E_B_ctrl"] - dense["E_B"]) / sm["E_B_ctrl_std_error"]
        assert abs(z) < 4.0, f"{form}: E_B_ctrl {sm['E_B_ctrl']:.5f} vs dense {dense['E_B']:.5f}, z={z:.2f}"
        assert abs(sm["E_A"] - dense["E_A"]) < 4.0 * sm["E_A_std_error"] + 1e-12

    def test_tensored_mitigation_undoes_readout_flips(self, chain32, vqe32, qet32, calib):
        """Readout flips only (no gate noise): mitigation recovers the
        noiseless ledger to 1e-8 and inflates the standard errors."""
        dense = qet32["dense"]
        flips = [0.03, 0.05, 0.02]
        raw = pr.qet_exact_aer(chain32, vqe32.params, 2, 0, 1, qet32["theta"], readout_flips=flips,
                               ideal=dense, shots=10000)
        mit = pr.qet_exact_aer(chain32, vqe32.params, 2, 0, 1, qet32["theta"], readout_flips=flips,
                               ideal=dense, shots=10000, mitigation="tensored")
        assert abs(raw["E_B_ctrl"] - dense["E_B"]) > 1e-3, "flips must actually bias the raw ledger"
        for key, dkey in (("E_A", "E_A"), ("E_B_ctrl", "E_B"), ("E_B_info", "E_B_info")):
            assert abs(mit[key] - dense[dkey]) < 1e-8, (key, mit[key], dense[dkey])
        assert mit["E_B_ctrl_std_error"] > raw["E_B_ctrl_std_error"]


# ---------------------------------------------------------------------------
# 4. harvesting
# ---------------------------------------------------------------------------


class TestHarvest:
    def test_negativity_helper(self):
        bell = np.zeros(4, dtype=complex)
        bell[0] = bell[3] = 1 / math.sqrt(2)
        rho = np.outer(bell, bell.conj())
        lmin, E_N, neg = pr.negativity_two_qubit(rho)
        assert abs(E_N - math.log(2.0)) < 1e-12 and abs(lmin + 0.5) < 1e-12
        prod = np.diag([1.0, 0, 0, 0]).astype(complex)
        assert pr.negativity_two_qubit(prod) == (pytest.approx(0.0, abs=1e-12), 0.0, 0.0)
        rng = np.random.default_rng(0)
        for _ in range(20):
            A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            rho = A @ A.conj().T
            rho /= np.trace(rho)
            w = np.linalg.eigvalsh(pr.partial_transpose_B(rho))
            assert np.sum(w < -1e-12) <= 1, "two-qubit PT has at most one negative eigenvalue"

    def test_noiseless_aer_matches_dense_trotter(self, chain32, vqe32, chain24, vqe24, hspec):
        tr = pr.harvest_dense(chain32, vqe32.state, hspec, method="trotter")
        ex = pr.harvest_exact_aer(chain32, vqe32.params, 2, hspec, shots=10000, n_boot=50)
        err = np.max(np.abs(ex["rho_AB"] - tr["rho_AB"]))
        assert err < 1e-10, f"Aer vs dense Trotter rho_AB (d=2): {err:.3e}"
        assert abs(ex["E_N"] - tr["E_N"]) < 1e-10
        assert ex["E_N"] > 0.05 and ex["z_score"] > 5.0, (ex["E_N"], ex["z_score"])
        assert ex["cx_count"] > 30
        tr4 = pr.harvest_dense(chain24, vqe24.state, hspec, method="trotter")
        ex4 = pr.harvest_exact_aer(chain24, vqe24.params, 2, hspec, shots=None)
        err4 = np.max(np.abs(ex4["rho_AB"] - tr4["rho_AB"]))
        assert err4 < 1e-10, f"Aer vs dense Trotter rho_AB (d=4): {err4:.3e}"

    def test_trotter_converges_to_exact(self, chain32, vqe32, hspec):
        ex = pr.harvest_dense(chain32, vqe32.state, hspec, method="exact", exact_min_steps=64,
                              exact_tol=1e-10)
        assert ex["converged"], ex
        errs = []
        for n in (6, 12, 24, 48):
            sp = pr.harvest_spec(hspec.sites, hspec.gap, hspec.lam, width=hspec.width, n_steps=n)
            errs.append(abs(pr.harvest_dense(chain32, vqe32.state, sp, method="trotter")["E_N"] - ex["E_N"]))
        assert all(errs[i + 1] < errs[i] for i in range(3)), f"Trotter errors {errs}"
        assert errs[-1] < 1e-3, f"48-step Trotter error {errs[-1]:.3e} (exact E_N {ex['E_N']:.5f})"

    def test_gaussian_limit(self):
        """Weak coupling, N = 2: qubit detectors on the d-truncated field vs
        oscillator detectors on the exact Gaussian field (MAP-1)."""
        sp = pr.harvest_spec((0, 1), 1.5, 0.1, width=1.5, n_steps=6)
        g = pr.harvest_gaussian_reference(vs.truncated_chain(2, 1.0, 2), sp)
        assert g["E_N"] > 5e-4, g["E_N"]
        rel = {}
        for d in (2, 4):
            ch = vs.truncated_chain(2, 1.0, d)
            E0, psi = vs.exact_ground_state(ch)
            ex = pr.harvest_dense(ch, psi, sp, method="exact", exact_min_steps=128, exact_tol=1e-9)
            rel[d] = abs(ex["E_N"] - g["E_N"]) / g["E_N"]
        assert rel[2] < 0.10, f"d=2 vs Gaussian: {rel[2]:.3%} (E_N_gauss={g['E_N']:.3e})"
        assert rel[4] < 0.01, f"d=4 vs Gaussian: {rel[4]:.3%} (E_N_gauss={g['E_N']:.3e})"
        assert rel[4] < rel[2]

    def test_sampled_tomography_within_sigma(self, chain32, vqe32, hspec):
        from qiskit_aer import AerSimulator

        tr = pr.harvest_dense(chain32, vqe32.state, hspec, method="trotter")
        sm = pr.harvest_sampled(chain32, vqe32.params, 2, hspec, AerSimulator(), shots=10000,
                                seed=5, n_boot=100)
        z = (sm["lambda_min"] - tr["lambda_min"]) / sm["lambda_min_std_error"]
        assert abs(z) < 4.0, f"sampled lambda_min {sm['lambda_min']:.4f} vs {tr['lambda_min']:.4f}, z={z:.2f}"
        assert sm["E_N"] > 0 and sm["z_score"] > 3.0


# ---------------------------------------------------------------------------
# 5. noise, survival, on-ramp
# ---------------------------------------------------------------------------


class TestNoise:
    def test_calibration_point_is_the_ikeda_table(self, calib):
        p = xio.load_experiment("ibm_qet/ikeda_2023_fig2").quantities
        assert calib["t1_ns"] == [p["qubit0_t1"].value * 1e3, p["qubit1_t1"].value * 1e3]
        assert calib["t2_ns"] == [p["qubit0_t2"].value * 1e3, p["qubit1_t2"].value * 1e3]
        assert calib["cnot_error"] == p["cnot_error"].value
        assert calib["readout_error"] == [p["qubit0_readout_assignment_error"].value,
                                          p["qubit1_readout_assignment_error"].value]
        assert calib["synthetic"] is False and calib["sweep"] is None
        assert "Table III" in calib["locations"]["cnot_error_location"]

    def test_two_qubit_model_reproduces_ikeda_qiskit(self, calib):
        from vacuum.qet.ikeda import ikeda_circuit_spec
        from vacuum.qet.ikeda_qiskit import noise_model_from_params, noisy_probabilities

        nm_a, d_a = nz.noise_model_from_calibration(calib, 2, [[0, 1], [1, 0]])
        nm_b, d_b = noise_model_from_params()
        for obs in ("V", "H1"):
            spec = ikeda_circuit_spec(None, obs, "deferred")
            pa = noisy_probabilities(spec, nm_a, d_a["readout_error"])["probabilities"]
            pb = noisy_probabilities(spec, nm_b, d_b["readout_error"])["probabilities"]
            err = max(abs(pa[k] - pb[k]) for k in pa)
            assert err < 1e-12, f"{obs}: N-qubit model vs ikeda_qiskit: {err:.3e}"

    def test_scaled_points_are_labelled(self, calib):
        s = nz.scale_calibration(calib, gate=3.0, readout=100.0, t1=4.0)
        assert s["synthetic"] is True and s["sweep"] == {"gate": 3.0, "readout": 100.0, "t1": 4.0}
        assert s["cnot_error"] == 3.0 * calib["cnot_error"]
        assert s["readout_error"] == [0.5, 0.5]
        assert all(t2 <= 2 * t1 + 1e-9 for t1, t2 in zip(s["t1_ns"], s["t2_ns"]))
        assert nz.scale_calibration(calib)["synthetic"] is False
        with pytest.raises(ValueError):
            nz.scale_calibration(calib, t1=0.0)

    def test_sweep_file_provenance(self):
        grid, exp = nz.sweep_grid("gate")
        assert exp.synthetic and exp.variant_of == "ikeda_2023_fig2" and exp.changes
        assert 1.0 in grid and len(grid) >= 5
        assert exp.quantities["open_plan_qpu_budget"].value == 600
        assert exp.quantities["open_plan_qpu_budget"].units == "s"
        assert set(exp.extras["synthetic_quantities"]) == {
            "gate_error_multiplier_grid", "readout_error_multiplier_grid", "t1_divisor_grid",
            "shots_per_circuit"}

    def test_survival_smoke(self, chain32, vqe32, qet32, hspec, calib):
        """Three points per protocol: alive / alive / dead along one axis,
        with z monotone; plus a short bisection on the QET information signal."""
        dense = qet32["dense"]
        cmap = pr.linear_coupling_map(chain32.n_qubits)

        def qet_info(c):
            nm, d = nz.noise_model_from_calibration(c, chain32.n_qubits, cmap)
            led = pr.qet_exact_aer(chain32, vqe32.params, 2, 0, 1, qet32["theta"], noise_model=nm,
                                   readout_flips=d["readout_error"], shots=100000, ideal=dense,
                                   mitigation="tensored")
            return {"signal": led["E_B_info"], "sigma": led["E_B_info_std_error"],
                    "E_B_ctrl": led["E_B_ctrl"], "E_A": led["E_A"]}

        rows = nz.survival_sweep(qet_info, calib, "gate", [1.0, 8.0, 64.0])
        zs = [r["z"] for r in rows]
        assert rows[0]["alive"] and rows[1]["alive"] and not rows[2]["alive"], zs
        assert zs[0] > zs[1] > zs[2], zs
        assert rows[0]["synthetic"] is False and rows[1]["synthetic"] is True
        assert rows[0]["E_B_ctrl"] < 0, "Bob-idle control is expected to lose at the sourced point"
        res = nz.survival_threshold(qet_info, calib, "gate", n_iter=3)
        assert res["alive_at_source"] and 8.0 < res["nu_star"] < 64.0, res["nu_star"]
        assert res["physical"]["cnot_error"] == pytest.approx(res["nu_star"] * calib["cnot_error"])

        hcmap = pr.harvest_coupling_map(chain32, hspec)

        def harv(c):
            nm, d = nz.noise_model_from_calibration(c, chain32.n_qubits + 2, hcmap)
            out = pr.harvest_exact_aer(chain32, vqe32.params, 2, hspec, noise_model=nm,
                                       readout_flips=d["readout_error"], shots=10000, n_boot=60)
            return {"signal": -out["lambda_min"], "sigma": out["lambda_min_std_error"], "E_N": out["E_N"]}

        rows = nz.survival_sweep(harv, calib, "all", [1.0 / 16.0, 1.0 / 4.0, 1.0])
        zs = [r["z"] for r in rows]
        assert rows[0]["alive"] and not rows[2]["alive"], zs
        assert zs[0] > zs[1] > zs[2], zs
        assert rows[2]["E_N"] == 0.0, "harvesting is expected dead at the sourced point"

    def test_device_on_ramp(self, chain32, vqe32, qet32, hspec):
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2

        nf = chain32.n_qubits
        body = pr.harvest_circuit(chain32, vqe32.params, 2, hspec)
        circs = pr.tomography_circuits(body, [nf, nf + 1])
        counts = nz.device_run(circs, AerSimulator(), 2000, seed=1)
        assert len(counts) == 9 and all(sum(c.values()) == 2000 for c in counts)
        out = pr.harvest_from_counts(counts, shots=2000, n_boot=30)
        assert out["E_N"] > 0
        # primitive-only route (SamplerV2), as a real IBM Runtime device needs
        counts_s = nz.device_run(circs[:2], AerSimulator(), 500, sampler=SamplerV2(seed=2))
        assert all(sum(c.values()) == 500 for c in counts_s)
        fam = pr.qet_circuits(chain32, vqe32.params, 2, 0, 1, qet32["theta"])
        qcounts = {}
        for name in ("ground", "control", "protocol", "wrong"):
            qcounts[name] = {}
            for opname, (groups, const) in fam[name]["operators"].items():
                mcs = pr.measurement_circuits(fam[name]["body"], groups, fam[name]["measured"])
                qcounts[name][opname] = nz.device_run(mcs, AerSimulator(), 2000, seed=4)
        led = pr.qet_from_counts(fam, qcounts, ideal=qet32["dense"], shots=2000)
        assert "E_B_info" in led and led["E_A_std_error"] > 0


# ---------------------------------------------------------------------------
# 2b. the (3,4) / (4,4) rows: expressibility ceiling, warm-started VQE
#     (papers/hardware-as-noisy-field/data/s2b_vqe_deep.json)
# ---------------------------------------------------------------------------

import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

S2B = Path(__file__).resolve().parents[1] / "papers" / "hardware-as-noisy-field" / "data" / "s2b_vqe_deep.json"


@pytest.fixture(scope="module")
def s2b():
    if not S2B.exists():
        pytest.skip(f"{S2B} not produced")
    with open(S2B) as fh:
        return json.load(fh)


class TestVQEDeep:
    def test_adjoint_gradients_are_exact(self):
        """The adjoint overlap gradient equals the amplitude shift rule
        d psi/d theta_k = psi(theta + pi e_k)/2 (the Jacobian's columns) to
        roundoff, and the adjoint energy gradient equals the parameter-shift
        gradient of the numpy evaluator."""
        rng = np.random.default_rng(3)
        for N, d, L in [(2, 4, 2), (3, 4, 3), (4, 4, 4)]:
            ch = vs.truncated_chain(N, 1.0, d)
            n = ch.n_qubits
            H = vs.hamiltonian_dense(ch)
            _, psi = vs.exact_ground_state(ch)
            p = rng.uniform(-3, 3, size=vs.hea_parameter_count(n, L))
            o, g = vs.hea_overlap_and_grad(p, n, L, psi)
            J = vs.hea_state_jacobian(p, n, L)
            assert abs(o - np.vdot(psi, vs.hea_statevector(p, n, L))) < 1e-14
            err = np.max(np.abs(psi.conj() @ J - g))
            assert err < 1e-13, f"adjoint vs shift-rule Jacobian ({N},{d},L={L}): {err:.2e}"
            E, gE = vs.hea_energy_and_grad(p, n, L, H)
            Eb = vs._NumpyEvaluator(H, n, L)(vs._parameter_shift_batch(p))
            assert abs(E - Eb[0]) < 1e-13
            err = np.max(np.abs(gE - 0.5 * (Eb[1::2] - Eb[2::2])))
            assert err < 1e-12, f"adjoint vs parameter-shift energy gradient ({N},{d},L={L}): {err:.2e}"

    def test_layerwise_embedding_preserves_state(self):
        rng = np.random.default_rng(4)
        for n, lf, lt in [(4, 1, 2), (6, 3, 5), (8, 2, 6)]:
            p = rng.uniform(-3, 3, size=vs.hea_parameter_count(n, lf))
            q = vs.hea_embed_layers(p, n, lf, lt)
            assert q.size == vs.hea_parameter_count(n, lt)
            err = np.max(np.abs(vs.hea_statevector(q, n, lt) - vs.hea_statevector(p, n, lf)))
            assert err == 0.0, f"embedding {lf}->{lt} layers on {n} qubits moved the state by {err:.2e}"
        with pytest.raises(ValueError):
            vs.hea_embed_layers(np.zeros(6), 3, 1, 0)

    def test_ceiling_at_the_converged_rows_is_roundoff(self):
        """The same method that finds the deep plateaus reports roundoff where
        the energy-VQE converged: (2,4) at 2 layers."""
        ch = vs.truncated_chain(2, 1.0, 4)
        fr = vs.fidelity_ceiling(ch, 2, restarts=4, seed=0, maxiter=200, stop_at=1e-14)
        assert fr.infidelity < 1e-13, f"(2,4) L=2 ceiling {fr.infidelity:.2e}"
        assert abs(fr.energy_error) < 1e-12 and fr.cov_error < 1e-6, (fr.energy_error, fr.cov_error)
        assert fr.restarts_run <= fr.restarts and fr.best_start in ("normal", "uniform")

    def test_ceiling_34_three_layers_is_the_pinned_plateau(self):
        """(3,4) at 3 layers: every converged start lands on the same fidelity
        plateau; it is the ansatz, not the optimizer.  The number is pinned
        at the value both optimizers (LM, L-BFGS-B) return from independent
        starts: infidelity 6.4305e-5 (the s2 energy-VQE row had 6.56e-5)."""
        ch = vs.truncated_chain(3, 1.0, 4)
        fr = vs.fidelity_ceiling(ch, 3, restarts=4, seed=0, maxiter=400)
        assert abs(fr.infidelity - 6.4305e-5) < 2e-8, f"(3,4) L=3 ceiling {fr.infidelity:.6e}"
        assert fr.grad_norm < 1e-8, f"not a stationary point: |grad| = {fr.grad_norm:.2e}"
        assert fr.energy_error > 1e-4, "the L=3 plateau costs ~2.4e-4 in energy"
        fl = vs.fidelity_ceiling(ch, 3, restarts=2, seed=0, maxiter=3000, method="lbfgs")
        assert abs(fl.infidelity - 6.4305e-5) < 2e-8, f"L-BFGS-B ceiling {fl.infidelity:.6e}"

    def test_vqe_warm_start_and_adjoint_keep_the_pinned_rows(self, chain24):
        """Backward compatibility: the default call is unchanged (same result
        as the module fixture path); the new keywords work; a given warm
        start is recorded and tried first; numpy/adjoint agrees with Aer."""
        r0 = vs.vqe_ground_state(chain24, layers=2, restarts=2, seed=0, evaluator="numpy", maxiter=200)
        assert r0.warm_start is None and r0.gradient == "shift" and r0.wall_s > 0
        r1 = vs.vqe_ground_state(chain24, layers=2, restarts=0, seed=0, evaluator="numpy", gradient="adjoint",
                                 warm_start=r0.params, maxiter=200, gtol=1e-12)
        assert r1.warm_start == "given" and r1.restarts == 1
        assert abs(r1.warm_energy_error - r0.energy_error) < 1e-12
        assert r1.energy_error <= r0.energy_error + 1e-14
        assert abs(vs.ansatz_energy(chain24, r1.params, 2, evaluator="aer") - r1.energy) < 1e-12
        with pytest.raises(ValueError):
            vs.vqe_ground_state(chain24, layers=2, evaluator="aer", gradient="adjoint")
        with pytest.raises(ValueError):
            vs.vqe_ground_state(chain24, layers=2, warm_start="nope")


# ---------------------------------------------------------------------------
# 7. S2b — the (3,4) and (4,4) rows: expressibility ceiling and the deep VQE
# ---------------------------------------------------------------------------

#: the standard the four converged s2 rows meet (MANIFEST numbers table):
#: energy error <= 1e-14, covariance error <= 1.4e-8, infidelity <= 2.4e-15
S2_ACCEPT = {"energy": 1e-14, "cov": 1.4e-8, "infidelity": 2.4e-15}
#: expressibility ceiling of the R_Y/CX hardware-efficient ansatz on the ED
#: ground state: best infidelity found per depth in data/s2b_vqe_deep.json,
#: AFTER that depth's continuation (its own optimum re-optimised at a larger
#: Jacobian budget) and, one layer below the sufficient depth, after the hard
#: below-probe.  Still an UPPER bound on the attainable infidelity, never a
#: proven supremum.  Anchor depths only; the archive carries every depth.
S2B_CEILING = {
    "N3_d4": {3: 6.430476e-5, 4: 3.323453e-6, 5: 4.096771e-8, 6: 1.694000e-11, 7: 0.0, 8: 0.0},
    "N4_d4": {3: 2.930502e-4, 11: 3.866703e-8, 20: 2.409933e-8, 30: 4.100276e-12, 31: 0.0},
}
#: (depth whose envelope reaches infidelity <= 1e-12, the parameter-counting
#: depth ceil((2^n - 1)/n) - 1 for n = N log2(d) qubits).
#:
#: RETRACTION (fix round 1).  This file pinned S2B_MINIMAL = {"N3_d4": (8, 10)}
#: and asserted that no depth below 8 reaches 1e-12.  Both were wrong: the
#: depth-7 cold starts had exited on their 400-Jacobian budget while still
#: descending, and continuing the archive's own depth-7 optimum reaches
#: roundoff (test_continuation_closes_the_seventh_layer, live).  The row is
#: now SEVEN layers, and it is published as a *sufficient* depth -- an upper
#: bound -- with the probe one layer below it archived as the evidence.
S2B_SUFFICIENT = {"N3_d4": (7, 10), "N4_d4": (31, 31)}
#: the hard probe one layer below the sufficient depth, and how it ended.
#: ``converged`` False means the probe was STILL DESCENDING at its budget: that
#: depth is not excluded at all, and the row above it rests only on the state
#: exhibited there.  At (4,4) that is the case -- L = 30 has fallen
#: 3.64e-10 -> 1.78e-11 -> 5.61e-12 -> 4.10e-12 as the budget grew.
S2B_BELOW_PROBE = {
    "N3_d4": {"layers": 6, "infidelity": 1.694000e-11, "converged": True,
              "restarts_run": 11, "maxiter": 2500, "grad_norm": 1e-12},
    "N4_d4": {"layers": 30, "infidelity": 4.100276e-12, "converged": False,
              "restarts_run": 2, "maxiter": 400, "grad_norm": 1e-6},
}
#: depths BELOW the published one whose best point ended on its Jacobian budget
#: rather than converging -- their numbers are unfinished descents.  Pinned so
#: that a regenerated archive cannot quietly add one (the round-1 defect).
S2B_BUDGET_EXHAUSTED = {"N3_d4": [], "N4_d4": [10, 11, 20, 30]}
#: the (3,4) ceiling job whose budget exhaustion caused the round-1 error
S2B_L7_JOB = S2B.parent / "jobs" / "s2b_ceiling_N3_d4_L7.json"


class TestVQEDeepRows:
    """`papers/hardware-as-noisy-field/data/s2b_vqe_deep.json` — the S2b
    archive that closes the two unconverged s2 rows.

    The state rows are not merely re-read: the archived parameter vectors are
    replayed through :func:`hea_statevector` against a live exact
    diagonalisation, so a corrupted or mislabelled row fails here.
    """

    def test_archive_build_and_acceptance(self, s2b):
        b = s2b["__build__"]
        for k in ("generator", "generator_sha256", "git_head", "produced_utc",
                  "python", "numpy", "scipy", "qiskit", "qiskit_aer", "vacuum"):
            assert b.get(k), f"build info missing {k}"
        assert b["generator"].endswith("notebook_vqe_deep.py")
        assert not b.get("quick"), "the archive must come from a full run"
        assert s2b["acceptance"]["infidelity"] == 1e-12
        kn = s2b["knobs"]
        assert kn["vqe"]["gtol"] == 1e-12 and kn["vqe"]["maxiter"] == 1000
        assert kn["threads_per_worker"] == 1
        assert s2b["jobs_wall_clock_s"] > 1e4, "the archive's jobs are the long ones"
        assert kn["continue_cases"]["N3_d4"]["depths"] == "all", (
            "every (3,4) depth must be continued past its scan budget")
        assert kn["below_probe"]["N3_d4"]["maxiter"] >= 2000

    def test_sufficient_depth_and_the_below_probe(self, s2b):
        """The depth whose envelope reaches infidelity <= 1e-12, and where it
        sits against n (L + 1) >= 2^n - 1 — necessary for the R_Y/CX image to
        cover the sphere of REAL n-qubit states, a heuristic (not a theorem)
        for one particular target.  (3,4) closes at 7 layers, three BELOW that
        depth (48 parameters for 63 real dof); (4,4) closes exactly at it.

        The row is an upper bound and the test says so: it does NOT assert
        that a smaller depth is impossible.  What it pins instead is (a) the
        envelope below the sufficient depth as measured, (b) that no depth
        below it was left budget-exhausted — the defect that made round 1
        publish 8 instead of 7 — and (c) the hard probe one layer below, with
        the budget it was given and what it reached.
        """
        for key, (L_suff, L_count) in S2B_SUFFICIENT.items():
            m = s2b["sufficient_depth"][key]
            assert m["reached"], f"{key}: no depth scanned reached 1e-12"
            assert m["sufficient_layers"] == L_suff, f"{key}: {m['sufficient_layers']} != {L_suff}"
            assert m["is_proved_minimal"] is False, "the row is an upper bound, never a proved minimum"
            assert m["counting_bound_layers"] == L_count, key
            assert m["best_infidelity_at_sufficient"] <= 1e-12
            tab = {t["layers"]: t for t in m["best_by_depth"]}
            env = [tab[L]["envelope_infidelity"] for L in sorted(tab)]
            assert all(a >= b for a, b in zip(env, env[1:])), f"{key}: envelope not monotone in depth"
            below = [L for L in sorted(tab) if L < L_suff]
            assert below, key
            assert min(tab[L]["envelope_infidelity"] for L in below) > 1e-12, (
                f"{key}: a depth below {L_suff} already reaches the acceptance -- the row is stale")
            stuck = [L for L in below if tab[L].get("budget_exhausted") is True]
            assert stuck == S2B_BUDGET_EXHAUSTED[key], (
                f"{key}: depths {stuck} below the published depth ended on their Jacobian budget -- "
                f"their numbers are unfinished descents, not the ansatz; pinned {S2B_BUDGET_EXHAUSTED[key]}")
            assert m["budget_exhausted_depths"] == [L for L in sorted(tab)
                                                    if tab[L].get("budget_exhausted") is True]
        n3 = s2b["sufficient_depth"]["N3_d4"]
        assert n3["n_params_at_sufficient"] == 48 and n3["real_state_dof"] == 63
        assert n3["equals_counting_bound"] is False and n3["layers_below_counting_bound"] == 3
        assert n3["depths_not_scanned_below"] == [], "every depth below 7 was scanned at (3,4)"
        n4 = s2b["sufficient_depth"]["N4_d4"]
        assert n4["n_params_at_sufficient"] == 256 and n4["real_state_dof"] == 255
        assert n4["equals_counting_bound"] is True
        # (4,4) sampled 13..30 only at 16, 20, 24, 28, 30: the archive must keep saying so
        assert n4["depths_not_scanned_below"] == [13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29]
        # the raw per-depth search is NOT monotone at (4,4): cold LM starts land
        # in local minima of an over-determined residual problem.  The envelope
        # (exact layer embedding) is what bounds the ansatz.
        assert n4["non_monotone_depths"] == [16, 24, 28]
        assert n3["non_monotone_depths"] == []
        for key, want in S2B_BELOW_PROBE.items():
            pr = s2b["sufficient_depth"][key]["below_probe"]
            assert pr is not None and pr["layers"] == want["layers"] == S2B_SUFFICIENT[key][0] - 1
            assert pr["reached_acceptance"] is False
            assert pr["infidelity"] > 1e-12, f"{key}: the probe at L={want['layers']} reaches the acceptance"
            assert abs(pr["infidelity"] - want["infidelity"]) <= 0.2 * want["infidelity"], (
                f"{key}: probe {pr['infidelity']:.6e} vs pinned {want['infidelity']:.6e}")
            assert pr["budget_exhausted"] is not want["converged"]
            assert pr["restarts_run"] == want["restarts_run"] and pr["maxiter"] == want["maxiter"]
            assert pr["grad_norm"] < want["grad_norm"], key
        # (3,4): the probe CONVERGED at 1.69e-11 (two independent starts, gradient
        # 5e-16), so "6 layers does not reach 1e-12" is evidence at a stated budget.
        # (4,4): the probe was still descending at L = 30, so 30 is not excluded at
        # all and the published 31 rests only on the state exhibited there.
        assert s2b["sufficient_depth"]["N3_d4"]["below_probe"]["budget_exhausted"] is False
        assert s2b["sufficient_depth"]["N4_d4"]["below_probe"]["budget_exhausted"] is True

    def test_manifest_limitations_agree_with_the_archive(self, s2b):
        """The MANIFEST's prose must state the archive's OWN lists.

        Fix round 2 exists because two Limitations statements were left stale
        by fix round 1's own (4,4) pass: they said "no below probe has been run
        at (4,4)" and quoted L = 30 at 1.78e-11 (the superseded continuation
        value, not the probe's 4.100276e-12), and listed depth 12 as
        non-monotone after the continuations had removed it -- while the
        numbers table 180 lines earlier, the archive and these tests already
        said otherwise.  Prose that contradicts the archive is a defect a
        reader acts on, so the three factual lists are pinned to the archive
        here and a regenerated archive drags the MANIFEST with it.
        """
        import re

        text = (S2B.parents[1] / "MANIFEST.md").read_text()
        n4 = s2b["sufficient_depth"]["N4_d4"]

        # The superseded round-1 sentences must not come back in the Limitations
        # section.  They are scoped to it and whitespace-normalised on purpose:
        # the CORRECTION record above QUOTES them, as this repository's
        # retraction convention requires, and must stay free to.
        head, sep, limitations = text.partition("## What is NOT claimed")
        assert sep, "MANIFEST no longer has a 'What is NOT claimed' section"
        flat = " ".join(limitations.split())
        for stale in ("no below probe has been run at (4,4)",
                      "add a `BELOW_PROBE[(4, 4)]` entry"):
            assert stale not in flat, (
                f"MANIFEST Limitations still says {stale!r}, but the (4,4) below "
                f"probe ran in fix round 1: L={n4['below_probe']['layers']} reached "
                f"{n4['below_probe']['infidelity']:.6e}")

        # the (4,4) below-probe value, as measured
        probe = n4["below_probe"]
        assert f"{probe['infidelity']:.6e}" in flat, (
            f"MANIFEST Limitations does not carry the measured (4,4) probe value "
            f"{probe['infidelity']:.6e}")

        def _ints(chunk):
            """expand '13-15, 17-19 and 29' (en dashes) to a sorted int list"""
            out = []
            for part in re.split(r",| and ", chunk):
                part = part.strip()
                if not part:
                    continue
                if "\u2013" in part or "-" in part:
                    a, b = re.split(r"\u2013|-", part)
                    out.extend(range(int(a), int(b) + 1))
                else:
                    out.append(int(part))
            return sorted(out)

        m = re.search(r"not monotone\*\* in depth at \(4,4\) \(depths ([0-9, ]+?) are worse", flat)
        assert m, "MANIFEST no longer states the (4,4) non-monotone depths"
        assert _ints(m.group(1)) == n4["non_monotone_depths"], (
            f"MANIFEST non-monotone list {m.group(1)!r} vs archive "
            f"{n4['non_monotone_depths']}")

        m = re.search(r"depths ([0-9,\s]+?(?: and \d+)?) all ended on their Jacobian budget", flat)
        assert m, "MANIFEST no longer states which (4,4) depths are budget-exhausted"
        assert _ints(m.group(1)) == n4["budget_exhausted_depths"], (
            f"MANIFEST budget-exhausted list {m.group(1)!r} vs archive "
            f"{n4['budget_exhausted_depths']}")

        m = re.search(r"depths\s+([0-9\u2013,\s]+?(?: and \d+)?) were never scanned", flat)
        assert m, "MANIFEST no longer states which (4,4) depths were never scanned"
        assert _ints(m.group(1)) == n4["depths_not_scanned_below"], (
            f"MANIFEST unscanned list {m.group(1)!r} vs archive "
            f"{n4['depths_not_scanned_below']}")

    def test_expressibility_ceiling_anchors(self, s2b):
        for key, pins in S2B_CEILING.items():
            tab = {t["layers"]: t for t in s2b["sufficient_depth"][key]["best_by_depth"]}
            for L, want in pins.items():
                got = tab[L]["best_infidelity"]
                if want == 0.0:
                    assert abs(got) < 1e-14, f"{key} L={L}: {got:.3e} is not at roundoff"
                else:
                    assert abs(got - want) <= 0.2 * want, f"{key} L={L}: {got:.6e} vs pinned {want:.6e}"

    def test_vqe_rows_meet_the_converged_standard(self, s2b):
        """The energy-VQE at the sufficient depth, warm-started from the
        fidelity optimum, on BOTH evaluators: the (3,4) and (4,4) rows now meet
        the standard of the four converged s2 rows.  Each row is REPLAYED from
        its archived parameters against a live exact diagonalisation."""
        seen = set()
        for r in s2b["vqe"]:
            key = f"N{r['N']}_d{r['d']}"
            if r["layers"] != s2b["sufficient_depth"][key]["sufficient_layers"]:
                continue
            seen.add((r["N"], r["d"], r["evaluator"]))
            ch = vs.truncated_chain(r["N"], 1.0, r["d"])
            E0, psi_ed = vs.exact_ground_state(ch)
            psi = vs.hea_statevector(np.asarray(r["params"], dtype=float), ch.n_qubits, r["layers"])
            E = float(np.real(np.vdot(psi, vs.hamiltonian_dense(ch) @ psi)))
            infid = 1.0 - vs.state_fidelity(psi_ed, psi)
            V, _ = vs.covariance_from_state(ch, psi)
            V_ed, _ = vs.covariance_from_state(ch, psi_ed)
            cov = float(np.max(np.abs(V - V_ed)))
            assert abs(E - E0) <= S2_ACCEPT["energy"], f"({r['N']},{r['d']}) L={r['layers']}: dE = {E - E0:.2e}"
            assert abs(infid) <= S2_ACCEPT["infidelity"], f"({r['N']},{r['d']}): infidelity {infid:.2e}"
            assert cov <= S2_ACCEPT["cov"], f"({r['N']},{r['d']}): covariance error {cov:.2e}"
            assert abs(E - r["energy"]) < 1e-12, "replayed energy disagrees with the archived row"
            assert abs(r["energy_error"]) <= S2_ACCEPT["energy"] and abs(r["infidelity"]) <= S2_ACCEPT["infidelity"]
            assert r["cov_error_vs_ED"] <= S2_ACCEPT["cov"]
            assert r["warm_start"] == "fidelity" and r["restarts"] == 1 and r["gtol"] == 1e-12
            assert r["cross_vs_energy"] < 1e-13, f"aer vs numpy cross-evaluation {r['cross_vs_energy']:.2e}"
            if r["evaluator"] == "aer":
                assert r["aer_vs_numpy"] is not None and r["aer_vs_numpy"] < 1e-13
        assert seen == {(3, 4, "aer"), (3, 4, "numpy"), (4, 4, "aer"), (4, 4, "numpy")}, seen
        assert {r["layers"] for r in s2b["vqe"] if r["N"] == 3 and r["layers"] != 3} == {7}, (
            "the (3,4) row is 7 layers, not the retracted 8")

    def test_three_layer_rows_are_the_ansatz_not_the_optimizer(self, s2b):
        """Warm-starting the L = 3 energy-VQE from the L = 3 FIDELITY optimum
        (the best state that ansatz holds) lands back on the s2 numbers: the
        two unconverged s2 rows were the ansatz's ceiling, not a stuck
        optimizer.  s2: (3,4) dE 2.334e-4, infidelity 6.560e-5; (4,4) dE
        1.426e-3, infidelity 3.231e-4."""
        rows = {r["N"]: r for r in s2b["vqe"] if r["layers"] == 3}
        assert set(rows) == {3, 4}
        r3 = rows[3]
        assert abs(r3["energy_error"] - 2.332e-4) < 5e-7, r3["energy_error"]
        assert abs(r3["infidelity"] - 6.561e-5) < 2e-7, r3["infidelity"]
        assert abs(r3["warm_infidelity"] - 6.4305e-5) < 2e-8, r3["warm_infidelity"]
        r4 = rows[4]
        assert abs(r4["energy_error"] - 1.396e-3) < 2e-5, r4["energy_error"]
        assert abs(r4["infidelity"] - 3.134e-4) < 5e-6, r4["infidelity"]
        assert r4["warm_infidelity"] < r4["infidelity"], "the fidelity optimum is the more faithful state"

    def test_downstream_34_rows_on_the_vqe_state(self, s2b):
        """The protocol rows the main notebook runs at (3,4) on ED states —
        the s3a QET ledgers (A/B = 0/1 and 0/2) and the s4b Gaussian-limit
        harvesting rows — repeated on the VQE state.  Nothing downstream runs
        at (4,4) and the archive says so."""
        q = s2b["downstream"]["N3_d4"]["downstream_qet"]
        assert [(c["site_A"], c["site_B"]) for c in q["cases"]] == [(0, 1), (0, 2)]
        assert q["layers"] == 7 and abs(q["infidelity"]) < 2.4e-15
        assert q["max_abs_diff"] < 1e-8, q["max_abs_diff"]
        assert all(c["audit_passed_VQE"] for c in q["cases"])
        # the residual difference is the theta optimizer's own tolerance, not the
        # state: the two states agree to 2.2e-16 and theta differs by ~2e-9.
        assert all(c["theta_abs_diff"] < 1e-7 for c in q["cases"])
        h = s2b["downstream"]["N3_d4"]["downstream_harvest"]
        assert [r["lam"] for r in h["rows"]] == [0.1, 0.4, 1.0]
        assert h["max_abs_diff_E_N"] < 1e-14 and h["max_rho_AB_abs_diff"] < 1e-14
        assert all(r["converged_ED"] and r["converged_VQE"] for r in h["rows"])
        assert "no downstream protocol row" in s2b["downstream"]["N4_d4"]["note"]

    def test_continuation_closes_the_seventh_layer(self):
        """LIVE, and the retraction of round 1 in one call.  The archived
        (3,4) depth-7 SCAN optimum stopped at infidelity 7.66e-11 with its
        best start reporting 'The maximum number of function evaluations is
        exceeded'.  Continuing that same point — no new start, no new
        information — reaches roundoff, so 8 layers was never needed.  This is
        why every depth now gets a continuation before its number is read."""
        with open(S2B_L7_JOB) as fh:
            row = json.load(fh)["row"]
        assert row["layers"] == 7 and row["n_params"] == 48
        assert row["infidelity"] > 1e-11, "the scan row is the budget-truncated one"
        best = min(row["history"], key=lambda h: h["infidelity"])
        assert "maximum number of function evaluations" in best["message"].lower()
        ch = vs.truncated_chain(3, 1.0, 4)
        fr = vs.fidelity_ceiling(ch, 7, restarts=0, maxiter=600, method="lm", stop_at=1e-13,
                                 init_params={"continue": np.asarray(row["params"], dtype=float)})
        assert abs(fr.infidelity) < 1e-13, f"(3,4) L=7 continuation {fr.infidelity:.3e}"
        assert abs(fr.energy_error) <= S2_ACCEPT["energy"] and fr.cov_error <= S2_ACCEPT["cov"]

    def test_ceiling_is_reproducible_at_depth(self):
        """Live: at (3,4) with 10 layers the LM fidelity maximisation reaches
        roundoff from a single cold start — the machinery behind the deep rows,
        rerun here."""
        ch = vs.truncated_chain(3, 1.0, 4)
        fr = vs.fidelity_ceiling(ch, 10, restarts=1, seed=10000, maxiter=400, stop_at=1e-13)
        assert abs(fr.infidelity) < 1e-13, f"(3,4) L=10 ceiling {fr.infidelity:.3e}"
        assert abs(fr.energy_error) < 1e-13 and fr.cov_error < 1e-13, (fr.energy_error, fr.cov_error)
        assert fr.restarts_run == 1 and fr.n_params == 66

    @pytest.mark.skipif(os.environ.get("VACUUM_SLOW_TESTS") != "1",
                        reason="~70 s; set VACUUM_SLOW_TESTS=1 to run")
    def test_layerwise_scan_reaches_roundoff_by_depth_eight(self):
        """Live (slow): the layerwise-growing scan, at its archived knobs, is
        already within 1e-11 at 7 layers and at roundoff by 8.  It is the
        CONTINUATION of the depth-7 optimum (the test above), not this scan,
        that closes 7 itself — the scan's budget is 400 Jacobian evaluations
        per depth."""
        ch = vs.truncated_chain(3, 1.0, 4)
        scan = vs.fidelity_depth_scan(ch, layers=range(1, 9), restarts=1, seed=7,
                                      maxiter=400, method="lm", stop_at=1e-13)
        got = {fr.layers: fr.infidelity for fr in scan}
        assert got[7] < 1e-11, got[7]
        assert abs(got[8]) < 1e-13, got[8]
