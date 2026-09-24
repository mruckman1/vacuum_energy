"""Tests for vacuum.qet.ikeda (dense, core-only) and vacuum.qet.ikeda_qiskit
(the ``.[ibm]`` extra, pytest.importorskip) — the Ikeda (2023) IBM-hardware
QET protocol, PLAN.md Layer 3 ("reproduce the Ikeda 2023 IBM-hardware
protocol in simulation") and Layer 7 L6 ("hardware as noisy field").

Paper: K. Ikeda, Phys. Rev. Applied 20, 024051 (2023) [arXiv:2301.02666v5].
Numbers from the paper enter ONLY through experiments/ibm_qet/*.json
(vacuum.experiments_io): the Fig. 2 file (device Table III column, Table I
(h,k) = (1,1) rows, Fig. 2(B) bar labels) and the Table II file.

Anchors:

1. Dense path vs the Hotta closed forms (vacuum.qet.hotta, review
   arXiv:1101.3954 Eqs. (8), (11), (14)) to 1e-12 on the M3.1 (h, k) grid
   and the paper's five (h, k) pairs, both circuit forms; the circuit's
   ground state is Eq. (4) and its final state is Eq. (6) exactly.
2. Paper theoretical values: every analytical entry of Tables I and II to
   the precision the paper prints — one unit in the fourth decimal (1e-4);
   the paper truncates rather than rounds (e.g. <V>(1, 0.5) exact -0.25989,
   printed -0.2598), so half a unit is NOT attainable and is not claimed.
   The paper's own qasm_simulator rows (10^5 shots) agree with the exact
   values within 3 sigma of their shot noise.
3. Feed-forward (Fig. 1(B)) and deferred (Fig. 1(C)) circuits give the same
   readout statistics (Appendix B) to 1e-14.
4. Standing audit: E_A >= E_B on every configuration — dense grid and paper
   pairs, and every qiskit family (ideal, noisy, sampled, mitigated).
5. No-signaling on the dense path: Bob's reduced state before the bit is
   independent of Alice's measurement, trace distance < 1e-14.
6. qiskit/Aer (importorskip): Statevector and noiseless density matrix vs
   the dense path to 1e-10; noiseless sampling at the paper's 10^5 shots vs
   its qasm_simulator row within 3 sigma; the noisy Aer run — noise model
   built ONLY from Table III (ibm_cairo) numbers, each cited — reproduces
   the paper's hardware numbers qualitatively: every energy degrades in the
   hardware's direction, <V> stays negative (the paper's headline
   observation), the 'fidelity' readout model lands <V> inside Table I's
   unmitigated error bar, both models bracket the hardware <V> between the
   ideal and the measured value, the noisy V histogram is closer (total
   variation) to the Fig. 2(B) raw bars than the ideal one is, and the
   paper's calibration-matrix mitigation moves <V> toward the ideal — while
   the mitigated simulation still sits BELOW the paper's mitigated <V>,
   i.e. the device's gate noise during the run exceeded its Table III
   calibration (recorded, not hidden).
"""

import itertools
import math
import subprocess
import sys

import numpy as np
import pytest

from vacuum import experiments_io as xio
from vacuum.qet import (
    BITSTRINGS,
    FORMS,
    IKEDA_FIG2_FILE,
    IKEDA_TABLE2_FILE,
    OBSERVABLES,
    PAPER_PAIRS,
    bob_angle,
    e1_closed,
    e_a_closed,
    e_b_closed,
    e_b_optimal,
    expectation_from_counts,
    ground_state,
    h1_closed,
    ikeda_circuit_spec,
    ikeda_ibm,
    ikeda_no_signaling,
    ikeda_params,
    ikeda_sweep,
    prep_angle,
    run_protocol,
    simulate_spec,
    theta_optimal,
    v_closed,
)

# the M3.1 (h, k) grid of tests/test_hotta.py, plus the paper's pairs
H_VALS = [0.05, 0.3, 1.0, 2.0, 5.0]
K_VALS = [0.1, 0.5, 1.0, 2.5, 4.0]
GRID = list(itertools.product(H_VALS, K_VALS))
ALL_PAIRS = GRID + list(PAPER_PAIRS)
PRINTED = 1e-4          # one unit in the last (fourth) printed decimal
PAPER_SHOTS = 100000    # Tables I-III: 10^5 shots


def tv_distance(p, q):
    """Total variation distance between two {"c1c0": p} distributions."""
    return 0.5 * sum(abs(p.get(b, 0.0) - q.get(b, 0.0)) for b in BITSTRINGS)


def hist_dict(values):
    """Fig. 2(B) bar list [p00, p01, p10, p11] -> {"c1c0": p}."""
    return dict(zip(BITSTRINGS, [float(v) for v in values]))


@pytest.fixture(scope="module")
def fig2():
    return ikeda_params(IKEDA_FIG2_FILE)


@pytest.fixture(scope="module")
def table2():
    return xio.load_experiment(IKEDA_TABLE2_FILE)


@pytest.fixture(scope="module")
def dense():
    """ikeda_ibm on the shipped file, both circuit forms."""
    return {form: ikeda_ibm(form=form) for form in FORMS}


# ---------------------------------------------------------------------------
# transcription sanity
# ---------------------------------------------------------------------------


class TestTranscription:
    def test_shipped_parameters(self, fig2):
        assert fig2["h"] == 1.0 and fig2["k"] == 1.0
        assert "024051" in fig2["source"] and "2301.02666" in fig2["source"]
        assert fig2["shots"] == PAPER_SHOTS
        assert fig2["device"]["backend_fig2B"] == "ibm_cairo"
        for name in ("cnot_error", "gate_time_cnot", "measurement_fidelity",
                     "qubit0_t1", "qubit1_t2", "qubit0_readout_assignment_error"):
            assert name in fig2, name

    def test_prep_angle_is_eq7_and_the_authors_arcsin_form(self):
        """Eq. (7): theta = -arccos(sqrt((1 - h/s)/2)); the author's code
        writes -arcsin(sqrt((1 + h/s)/2)) — identical since cos theta >= 0."""
        worst = 0.0
        for h, k in ALL_PAIRS:
            s = math.hypot(h, k)
            alt = -math.asin(math.sqrt((1.0 + h / s) / 2.0))
            worst = max(worst, abs(prep_angle(h, k) - alt))
            assert -math.pi / 2 < prep_angle(h, k) < 0.0
        assert worst < 1e-14, f"Eq. (7) vs arcsin form: worst |diff| = {worst:.3e}"

    def test_circuit_prepares_eq4_ground_state(self):
        """CNOT (R_Y(2 theta) (x) I)|00> [Eq. (7), (A4)] equals Eq. (4)
        amplitude by amplitude (hotta.ground_state) — sign included."""
        worst = 0.0
        for h, k in ALL_PAIRS:
            led = ikeda_ibm({"h": h, "k": k})
            worst = max(worst, led["ground_state_check"])
        assert worst < 1e-14, f"circuit |g> vs Eq. (4): worst |diff| = {worst:.3e}"

    def test_bob_angle_is_theta_star(self):
        """Eqs. (11)-(12) phi equals the review's theta* (Eqs. (9)-(10))."""
        worst = max(abs(bob_angle(h, k) - theta_optimal(h, k)) for h, k in ALL_PAIRS)
        assert worst < 1e-15, f"phi vs theta*: worst |diff| = {worst:.3e}"
        for h, k in ALL_PAIRS:
            assert 0.0 < bob_angle(h, k) < math.pi / 4

    def test_spec_structure_and_citations(self, fig2):
        for obs in OBSERVABLES:
            for form in FORMS:
                spec = ikeda_circuit_spec(fig2, obs, form)
                names = [g["name"] for g in spec["gates"]]
                assert names[:3] == ["ry", "cx", "h"]
                assert names[-1] == "measure"
                assert all(g["cite"] for g in spec["gates"]), "uncited gate"
                if obs == "E0":
                    assert "cry" not in names and "condition" not in str(spec)
                elif form == "deferred":
                    assert names.count("cry") == 2 and names.count("x") == 2
                    assert names.count("measure") == 1
                else:
                    assert names.count("measure") == 2
                    conds = [g.get("condition") for g in spec["gates"] if "condition" in g]
                    assert conds == [[0, 0], [0, 1]]
                assert ("h" in names[3:]) == (obs == "V")
        with pytest.raises(ValueError):
            ikeda_circuit_spec(fig2, "XX")
        with pytest.raises(ValueError):
            ikeda_circuit_spec(fig2, "V", "mid")
        with pytest.raises(ValueError):
            ikeda_ibm({"h": 0.0, "k": 1.0})

    def test_estimators_follow_eqs_a5_a6(self):
        """Eq. (A5): <Z_1> from bit c1; Eq. (A6): <X_0 X_1> from (-1)^{c0+c1};
        the E0 estimator reads X_0 from bit c0; constants from Eqs. (2)-(3)."""
        h, k = 1.0, 1.0
        s = math.sqrt(2.0)
        counts = {"00": 10, "01": 20, "10": 30, "11": 40}   # c1c0
        n = 100.0
        z1 = (10 + 20 - 30 - 40) / n
        x0 = (10 - 20 + 30 - 40) / n
        xx = (10 - 20 - 30 + 40) / n
        e = expectation_from_counts(counts, h, k, "H1")
        assert abs(e["value"] - (h * h / s + h * z1)) < 1e-15
        assert abs(expectation_from_counts(counts, h, k, "V")["value"]
                   - (2 * k * k / s + 2 * k * xx)) < 1e-15
        assert abs(expectation_from_counts(counts, h, k, "E0")["value"]
                   - (h * h / s + h * x0)) < 1e-15
        assert e["n"] == n and e["std_error"] == pytest.approx(
            h * math.sqrt(1 - z1 * z1) / math.sqrt(n))
        # register-spaced qiskit keys reduce to the last two characters
        assert expectation_from_counts({"1 0": 1, "0 1": 1}, h, k, "V")["pauli_mean"] == -1.0


# ---------------------------------------------------------------------------
# anchor 1: dense path vs Hotta closed forms, 1e-12
# ---------------------------------------------------------------------------


class TestDenseVsHotta:
    def test_ledger_matches_closed_forms(self):
        """E_A vs Eq. (9) [= review Eq. (8)], E_B vs review Eq. (11), E1 vs
        Eq. (14), <H_1>/<V> vs their derived forms; both circuit forms."""
        worst = {}
        for h, k in ALL_PAIRS:
            for form in FORMS:
                led = ikeda_ibm({"h": h, "k": k}, form=form)
                for key, val in led["residuals"].items():
                    worst[key] = max(worst.get(key, 0.0), val)
                assert abs(led["E_A"] - e_a_closed(h, k)) < 1e-12
                assert abs(led["E_B"] - e_b_optimal(h, k)) < 1e-12
                assert abs(led["E1"] + e_b_closed(h, k, led["phi"])) < 1e-12
        assert max(worst.values()) < 1e-12, f"closed-form residuals: {worst}"

    def test_derived_h1_v_forms_sum_to_eq14(self):
        worst = 0.0
        for h, k in ALL_PAIRS:
            for phi in np.linspace(0.0, math.pi / 4, 7):
                worst = max(worst, abs(h1_closed(h, k, phi) + v_closed(h, k, phi)
                                       - e1_closed(h, k, phi)))
                worst = max(worst, abs(e1_closed(h, k, phi) + e_b_closed(h, k, phi)))
        assert worst < 1e-13, f"<H1> + <V> vs Eq. (14): worst |diff| = {worst:.3e}"

    def test_estimator_energies_equal_difference_form_and_run_protocol(self):
        """The readout estimators reproduce Tr[rho H] bookkeeping
        (E_meas - E_ground, E_meas - E_final) and hotta.run_protocol at phi."""
        worst = 0.0
        for h, k in ALL_PAIRS:
            led = ikeda_ibm({"h": h, "k": k})
            rp = run_protocol(h, k, led["phi"])
            worst = max(worst,
                        abs(led["E_A_difference_form"] - led["E_A"]),
                        abs(led["E_B_difference_form"] - led["E_B"]),
                        abs(rp["E_A"] - led["E_A"]), abs(rp["E_B"] - led["E_B"]),
                        abs(led["H1_from_state"] - led["H1"]),
                        abs(led["V_from_state"] - led["V"]),
                        abs(led["E_ground"]))
        assert worst < 1e-12, f"estimator vs difference-form/run_protocol: {worst:.3e}"

    def test_circuit_state_is_eq6(self):
        """The circuit's pre-readout state, taken to the physical frame and
        X_0-dephased, is rho_QET of Eq. (6) built from hotta's P_0(mu) and
        U_1(mu) — exact."""
        worst = max(ikeda_ibm({"h": h, "k": k}, form=form)["circuit_state_check"]
                    for h, k in ALL_PAIRS for form in FORMS)
        assert worst < 1e-14, f"circuit state vs Eq. (6): worst |diff| = {worst:.3e}"

    def test_ideal_pauli_means(self, dense):
        """At (1, 1): <X_0> = 0 on |g>, <Z_1> = -(h/s) cos 2phi + (k/s) sin 2phi,
        <X_0 X_1> = -(h/s) sin 2phi - (k/s) cos 2phi (derivation in
        ikeda.h1_closed / v_closed)."""
        led = dense["deferred"]
        t = 2.0 * led["phi"]
        s = math.sqrt(2.0)
        assert abs(led["X0"]) < 1e-15
        assert abs(led["Z1"] - (-(1 / s) * math.cos(t) + (1 / s) * math.sin(t))) < 1e-14
        assert abs(led["X0X1"] - (-(1 / s) * math.sin(t) - (1 / s) * math.cos(t))) < 1e-14


# ---------------------------------------------------------------------------
# anchor 2: the paper's printed values
# ---------------------------------------------------------------------------


class TestPaperValues:
    def test_table_i_analytical_values_at_1_1(self, dense):
        """Table I (h,k) = (1,1): <E0> 0.7071, <H1> 0.2598, <V> -0.3746,
        <E1> -0.1147, to the printed precision (1e-4)."""
        led = dense["deferred"]
        res = led["paper_residuals"]
        assert set(res) == {"E0", "H1", "V", "E1"}
        worst = max(res.values())
        assert worst < PRINTED, f"Table I (1,1) residuals {res}"

    def test_table_ii_analytical_values_all_pairs(self, table2):
        """Table II: <E0>, <V>, <H1>, <E1> at the five (h,k) pairs."""
        pairs = [tuple(float(x) for x in p) for p in table2.quantities["pairs_hk"].value]
        assert pairs == [tuple(p) for p in PAPER_PAIRS]
        worst = 0.0
        rows = []
        for i, (h, k) in enumerate(pairs):
            led = ikeda_ibm({"h": h, "k": k})
            for key in ("E0", "V", "H1", "E1"):
                printed = float(table2.quantities[f"analytical_{key}"].value[i])
                d = abs(led[key] - printed)
                rows.append(((h, k), key, led[key], printed, d))
                worst = max(worst, d)
        assert worst < PRINTED, (
            "Table II analytical values: worst |exact - printed| = "
            f"{worst:.2e} (rows: {rows})"
        )
        # and the paper truncates: at least one residual exceeds half a unit
        assert worst > 0.5 * PRINTED, "expected truncation-sized residuals"

    def test_table_ii_qasm_rows_within_shot_noise(self, table2):
        """The paper's qasm_simulator values (10^5 shots) vs the exact
        values: within 3 sigma, sigma = max(printed error, shot-noise
        standard error of the estimator at 10^5 shots)."""
        pairs = [tuple(float(x) for x in p) for p in table2.quantities["pairs_hk"].value]
        worst_z = 0.0
        for i, (h, k) in enumerate(pairs):
            led = ikeda_ibm({"h": h, "k": k})
            sig_shot = {
                obs: expectation_from_counts(led["probabilities"][obs], h, k, obs,
                                             shots=PAPER_SHOTS)["std_error"]
                for obs in OBSERVABLES
            }
            sig_shot["E1"] = math.hypot(sig_shot["H1"], sig_shot["V"])
            for key in ("E0", "V", "H1", "E1"):
                q = table2.quantities[f"qasm_{key}"]
                val, err = float(q.value[i]), float(q.error[i])
                z = abs(led[key] - val) / max(err, sig_shot[key])
                worst_z = max(worst_z, z)
        assert worst_z < 3.0, f"Table II qasm_simulator vs exact: worst z = {worst_z:.2f}"

    def test_fig2b_simulator_bars_are_the_exact_distributions(self, dense, fig2):
        """Fig. 2(B) 'simulator' bar labels (3 decimals, 10^5 shots) vs the
        exact readout distributions: within label rounding + 3 sigma."""
        led = dense["deferred"]
        worst = 0.0
        for obs in ("V", "H1"):
            bars = hist_dict(fig2[f"fig2B_{obs}_simulator"])
            for b in BITSTRINGS:
                p = led["probabilities"][obs][b]
                tol = 0.0005 + 3.0 * math.sqrt(p * (1 - p) / PAPER_SHOTS)
                d = abs(bars[b] - p)
                worst = max(worst, d / tol)
                assert d < tol, f"Fig. 2(B) {obs} bar {b}: {bars[b]} vs exact {p:.4f}"
        assert worst < 1.0


# ---------------------------------------------------------------------------
# anchors 3-5: forms agree, standing audit, no-signaling
# ---------------------------------------------------------------------------


class TestProtocolInvariants:
    def test_feedforward_equals_deferred(self):
        """Appendix B: the two circuits yield exactly the same results."""
        worst_p = 0.0
        worst_e = 0.0
        for h, k in ALL_PAIRS:
            a = ikeda_ibm({"h": h, "k": k}, form="deferred")
            b = ikeda_ibm({"h": h, "k": k}, form="feedforward")
            for obs in OBSERVABLES:
                for bit in BITSTRINGS:
                    worst_p = max(worst_p, abs(a["probabilities"][obs][bit]
                                               - b["probabilities"][obs][bit]))
            for key in ("E_A", "E_B", "E0", "H1", "V", "E1"):
                worst_e = max(worst_e, abs(a[key] - b[key]))
        assert worst_p < 1e-14, f"feed-forward vs deferred probabilities: {worst_p:.3e}"
        assert worst_e < 1e-14, f"feed-forward vs deferred energies: {worst_e:.3e}"

    def test_feedforward_branches_and_deferred_purity(self, dense):
        assert dense["feedforward"]["specs"]["V"]["form"] == "feedforward"
        ff = simulate_spec(dense["feedforward"]["specs"]["V"])
        de = simulate_spec(dense["deferred"]["specs"]["V"])
        assert ff["n_branches_pre_readout"] == 2
        assert de["n_branches_pre_readout"] == 1
        rho = de["rho_pre_readout"]
        assert abs(np.real(np.trace(rho @ rho)) - 1.0) < 1e-14   # pure
        for sim in (ff, de):
            assert abs(sum(sim["probabilities"].values()) - 1.0) < 1e-14

    def test_standing_audit_e_a_ge_e_b(self):
        """E_A >= E_B on every configuration (tolerance -1e-12); also
        0 < E_B < E_A strictly — teleportation is real but never free."""
        for form in FORMS:
            res = ikeda_sweep(ALL_PAIRS, tol=1e-12, strict=True, form=form)
            assert res.passed and res.worst_violation == 0.0
            assert res.details["min_margin"] > 0.0, (
                f"E_A - E_B minimum {res.details['min_margin']:.3e} at {res.details['argmin']}"
            )
            for row in res.details["rows"]:
                assert 0.0 < row["E_B"] < row["E_A"], row

    def test_no_signaling_dense(self):
        """Bob's reduced state with vs without Alice's measurement, taken
        from the gate list: trace distance < 1e-14 on every pair."""
        worst = 0.0
        for h, k in ALL_PAIRS:
            out = ikeda_no_signaling({"h": h, "k": k})
            for rho in (out["rho_B_unmeasured"], out["rho_B_measured"]):
                assert abs(np.real(np.trace(rho)) - 1.0) < 1e-14
            worst = max(worst, out["trace_distance"])
        assert worst < 1e-14, f"no-signaling trace distance up to {worst:.3e}"

    def test_bob_marginal_is_the_ground_state_marginal(self):
        """Tr_A |g><g| = diag((1 - h/s)/2, (1 + h/s)/2)."""
        for h, k in PAPER_PAIRS:
            s = math.hypot(h, k)
            rho = ikeda_no_signaling({"h": h, "k": k})["rho_B_unmeasured"]
            expect = np.diag([(1 - h / s) / 2, (1 + h / s) / 2])
            assert np.max(np.abs(rho - expect)) < 1e-14


# ---------------------------------------------------------------------------
# anchor 6: qiskit / Aer (the .[ibm] extra)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qk():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from vacuum.qet import ikeda_qiskit
    return ikeda_qiskit


@pytest.fixture(scope="module")
def aer(qk):
    """The Aer ledgers at (1,1) for both readout models (seeded)."""
    return {ro: qk.ikeda_ibm_aer(readout=ro, seed=2023) for ro in ("table_iii", "fidelity")}


class TestQiskitExactness:
    def test_lazy_export_does_not_import_qiskit(self):
        code = ("import sys, vacuum.qet as q; assert 'qiskit' not in sys.modules; "
                "f = q.run_on_backend; g = q.ikeda_ibm_aer; "
                "assert f.__module__ == 'vacuum.qet.ikeda_qiskit'; print('ok')")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr

    def test_statevector_matches_dense(self, qk):
        """qiskit Statevector of the deferred circuit vs the dense pre-readout
        state (rank-1) and readout distribution, all observables/pairs, 1e-10."""
        worst_s = 0.0
        worst_p = 0.0
        for h, k in ALL_PAIRS:
            for obs in OBSERVABLES:
                spec = ikeda_circuit_spec({"h": h, "k": k}, obs, "deferred")
                sim = simulate_spec(spec)
                psi = qk.exact_statevector(spec)
                worst_s = max(worst_s, np.max(np.abs(np.outer(psi, psi.conj())
                                                     - sim["rho_pre_readout"])))
                pq = qk.probabilities_to_dict(np.abs(psi) ** 2)
                worst_p = max(worst_p, max(abs(pq[b] - sim["probabilities"][b])
                                           for b in BITSTRINGS))
        assert worst_s < 1e-10, f"Statevector vs dense state: {worst_s:.3e}"
        assert worst_p < 1e-10, f"Statevector vs dense probabilities: {worst_p:.3e}"
        with pytest.raises(ValueError):
            qk.exact_statevector(ikeda_circuit_spec(None, "V", "feedforward"))

    def test_noiseless_density_matrix_matches_dense(self, qk):
        worst = 0.0
        for h, k in PAPER_PAIRS:
            for obs in OBSERVABLES:
                spec = ikeda_circuit_spec({"h": h, "k": k}, obs, "deferred")
                rho, _ = qk.exact_density_matrix(spec)
                worst = max(worst, np.max(np.abs(rho - simulate_spec(spec)["rho_pre_readout"])))
        assert worst < 1e-10, f"Aer density matrix vs dense: {worst:.3e}"

    def test_ideal_family_equals_dense_ledger(self, aer, dense):
        led = dense["deferred"]
        for ro, out in aer.items():
            for key in ("E_A", "E_B", "E0", "H1", "V", "E1"):
                assert abs(out["ideal"][key] - led[key]) < 1e-10, (ro, key)
            assert abs(out["E_A"] - led["E_A"]) < 1e-10
            assert abs(out["E_B"] - led["E_B"]) < 1e-10

    def test_feedforward_circuit_samples_the_dense_distribution(self, qk):
        """The Fig. 1(B) circuit with if_test blocks on noiseless Aer:
        counts vs the dense probabilities within 5 sigma at 2e5 shots."""
        Q = qk._qiskit()
        spec = ikeda_circuit_spec(None, "V", "feedforward")
        ref = simulate_spec(spec)["probabilities"]
        n = 200000
        counts = qk.run_on_backend(qk.build_circuit(spec), Q.AerSimulator(), n, seed=11)
        assert sum(counts.values()) == n
        worst_z = 0.0
        for b in BITSTRINGS:
            z = abs(counts.get(b, 0) / n - ref[b]) / math.sqrt(ref[b] * (1 - ref[b]) / n)
            worst_z = max(worst_z, z)
        assert worst_z < 5.0, f"feed-forward sampling vs dense: worst z = {worst_z:.2f}"

    def test_run_on_backend_sampler_route(self, qk):
        """The primitive route (SamplerV2) returns the same statistics — the
        one-line change to a runtime device."""
        Q = qk._qiskit()
        from qiskit_aer.primitives import SamplerV2
        spec = ikeda_circuit_spec(None, "H1", "deferred")
        ref = simulate_spec(spec)["probabilities"]
        n = 100000
        qc = qk.build_circuit(spec)
        counts = qk.run_on_backend(qc, Q.AerSimulator(), n, seed=5, sampler=SamplerV2(seed=5))
        assert sum(counts.values()) == n
        worst_z = max(abs(counts.get(b, 0) / n - ref[b]) / math.sqrt(ref[b] * (1 - ref[b]) / n)
                      for b in BITSTRINGS)
        assert worst_z < 5.0, f"sampler route vs dense: worst z = {worst_z:.2f}"
        with pytest.raises(TypeError):
            qk.run_on_backend(qc, object(), 10)

    def test_noiseless_sampling_vs_paper_qasm_row(self, qk, fig2):
        """Noiseless Aer at the paper's 10^5 shots vs Table I's (1,1)
        qasm_simulator row: within 3 sigma, sigma^2 = printed^2 + shot^2."""
        Q = qk._qiskit()
        h, k = fig2["h"], fig2["k"]
        est = {}
        for i, obs in enumerate(OBSERVABLES):
            spec = ikeda_circuit_spec(fig2, obs, "deferred")
            counts = qk.run_on_backend(qk.build_circuit(spec), Q.AerSimulator(),
                                       fig2["shots"], seed=100 + i)
            est[obs] = expectation_from_counts(counts, h, k, obs)
        vals = {"E0": est["E0"]["value"], "H1": est["H1"]["value"], "V": est["V"]["value"]}
        vals["E1"] = vals["H1"] + vals["V"]
        sig = {obs: est[obs]["std_error"] for obs in OBSERVABLES}
        sig["E1"] = math.hypot(sig["H1"], sig["V"])
        worst_z = 0.0
        for key in ("E0", "H1", "V", "E1"):
            z = abs(vals[key] - fig2[f"qasm_{key}"]) / math.hypot(fig2[f"qasm_{key}_error"], sig[key])
            worst_z = max(worst_z, z)
        assert worst_z < 3.0, f"noiseless Aer vs Table I qasm_simulator (1,1): worst z = {worst_z:.2f}"


class TestQiskitNoisy:
    def test_noise_model_cites_every_table_iii_number(self, qk, fig2):
        for ro in ("table_iii", "fidelity", "none"):
            nm, desc = qk.noise_model_from_params(fig2, readout=ro)
            assert set(nm.basis_gates) == set(qk.IBM_BASIS_GATES)
            assert desc["t1_ns"] == [fig2["qubit0_t1"] * 1e3, fig2["qubit1_t1"] * 1e3]
            assert desc["t2_ns"] == [fig2["qubit0_t2"] * 1e3, fig2["qubit1_t2"] * 1e3]
            assert desc["cnot_error"] == fig2["cnot_error"]
            assert desc["cnot_gate_time_ns"] == fig2["gate_time_cnot"]
            assert desc["pauli_x_error"] == [fig2["qubit0_pauli_x_error"], fig2["qubit1_pauli_x_error"]]
            assert desc["single_qubit_gate_time_ns"] == 30.0     # midpoint of 20-40 ns
            for key in ("t1_t2_location", "pauli_x_error_location", "cnot_error_location",
                        "cnot_gate_time_location", "readout_note"):
                assert "Table III" in desc[key] or "Sec. II.D" in desc[key] or ro == "none"
            # the depolarizing parameter makes the total infidelity the printed error
            for lam, F in zip(desc["depolarizing_1q"], desc["relaxation_fidelity_1q"]):
                assert 0.0 <= lam < 1e-3 and F > 0.9998
            for lam in desc["depolarizing_2q"].values():
                assert 0.0 < lam < 0.01
        _, d3 = qk.noise_model_from_params(fig2, readout="table_iii")
        assert d3["readout_error"] == [fig2["qubit0_readout_assignment_error"],
                                       fig2["qubit1_readout_assignment_error"]]
        _, df = qk.noise_model_from_params(fig2, readout="fidelity")
        pr = 1.0 - math.sqrt(fig2["measurement_fidelity"])
        assert df["readout_error"] == [pr, pr]
        with pytest.raises(ValueError):
            qk.noise_model_from_params({"h": 1.0, "k": 1.0})
        with pytest.raises(ValueError):
            qk.noise_model_from_params(fig2, readout="guess")

    def test_native_circuits(self, aer):
        """Transpiled to {rz, sx, x, cx}: the deferred protocol costs 5 CNOTs
        (1 for |g>, 2 per controlled rotation), the E0 circuit 1."""
        for out in aer.values():
            assert out["cx_count"] == {"E0": 1, "H1": 5, "V": 5}, out["cx_count"]
            for obs, qc in out["circuits"].items():
                ops = set(qc.count_ops()) - {"save_density_matrix", "measure", "barrier"}
                assert ops <= set(qk_basis()), ops

    def test_sampled_matches_exact_noisy(self, aer):
        """Aer sampling with the noise model (10^5 shots) vs the shot-free
        noisy distributions: every bin within 5 sigma."""
        for ro, out in aer.items():
            n = out["shots"]
            worst_z = 0.0
            for obs in OBSERVABLES:
                for b in BITSTRINGS:
                    p = out["probabilities"]["noisy_exact"][obs][b]
                    q = out["probabilities"]["noisy_sampled"][obs][b]
                    worst_z = max(worst_z, abs(q - p) / math.sqrt(p * (1 - p) / n))
            assert worst_z < 5.0, f"{ro}: sampled vs exact noisy: worst z = {worst_z:.2f}"

    def test_degradation_direction_and_negative_v(self, aer, dense):
        """Every energy moves in the hardware's direction (toward no
        extraction) and <V> stays negative — the paper's headline claim
        ('we observed negative <V> for all parameter combinations in all
        quantum computers used', Sec. II.D)."""
        led = dense["deferred"]
        for ro, out in aer.items():
            for fam in ("noisy_exact", "noisy_sampled", "mitigated"):
                f = out[fam]
                assert f["V"] > led["V"], (ro, fam, f["V"])
                assert f["V"] < 0.0, (ro, fam, f["V"])
                assert f["H1"] > led["H1"], (ro, fam, f["H1"])
                assert f["E1"] > led["E1"], (ro, fam, f["E1"])

    def test_standing_audit_on_every_qiskit_family(self, aer):
        for ro, out in aer.items():
            for fam in ("ideal", "noisy_exact", "noisy_sampled", "mitigated"):
                f = out[fam]
                assert f["E_A"] - f["E_B"] >= -1e-12, (ro, fam, f["E_A"], f["E_B"])

    def test_v_brackets_hardware_and_fidelity_model_hits_error_bar(self, aer, dense, fig2):
        """Table I ibm_cairo unmitigated <V> = -0.1733 +- 0.0038: both
        readout models place the noisy <V> between the ideal value and the
        measured one (+3 sigma), and the 'fidelity' model lands inside the
        printed error bar (3 sigma)."""
        led = dense["deferred"]
        raw, err = fig2["ibm_cairo_unmitigated_V"], fig2["ibm_cairo_unmitigated_V_error"]
        zs = {}
        for ro, out in aer.items():
            v = out["noisy_exact"]["V"]
            assert led["V"] < v < raw + 3 * err, (ro, v)
            zs[ro] = abs(v - raw) / err
        assert zs["fidelity"] < 3.0, (
            f"'fidelity' model <V> = {aer['fidelity']['noisy_exact']['V']:.4f} vs "
            f"hardware {raw} +- {err}: z = {zs['fidelity']:.2f}"
        )
        # the ideal value is dozens of sigma away — the check has teeth
        assert abs(led["V"] - raw) / err > 20.0

    def test_v_histogram_closer_to_fig2b_raw_than_ideal(self, aer, dense, fig2):
        """Total-variation distance to the Fig. 2(B) raw V bars: the noisy
        simulation is closer than the ideal distribution, both models."""
        raw = hist_dict(fig2["fig2B_V_raw"])
        d_ideal = tv_distance(dense["deferred"]["probabilities"]["V"], raw)
        for ro, out in aer.items():
            d_noisy = tv_distance(out["probabilities"]["noisy_exact"]["V"], raw)
            assert d_noisy < d_ideal, (ro, d_noisy, d_ideal)
        assert d_ideal > 0.05

    def test_h1_hardware_between_ideal_and_model(self, aer, dense, fig2):
        """<H_1>: the hardware values — Table I unmitigated 0.2630 +- 0.0027
        (only 1.2 sigma above the ideal 0.2599) and the Fig. 2(B) raw bars
        (0.280) — lie between the ideal value and the noise model's (0.290 /
        0.299): the model over-predicts the Z_1 degradation. Recorded, not
        tuned away."""
        led = dense["deferred"]
        raw, err = fig2["ibm_cairo_unmitigated_H1"], fig2["ibm_cairo_unmitigated_H1_error"]
        h1_fig = expectation_from_counts(hist_dict(fig2["fig2B_H1_raw"]), 1.0, 1.0, "H1")["value"]
        assert led["H1"] < h1_fig
        for ro, out in aer.items():
            model = out["noisy_exact"]["H1"]
            assert led["H1"] - 3 * err < raw < model, (ro, led["H1"], raw, model)
            assert led["H1"] < h1_fig < model, (ro, h1_fig, model)
            assert (model - raw) / err > 3.0, (ro, model, raw)   # the over-prediction is real

    def test_mitigation_moves_v_toward_ideal_but_device_gate_noise_exceeds_calibration(
            self, aer, dense, fig2):
        """The paper's calibration-matrix mitigation on the simulated noisy
        distributions: <V> becomes more negative; and the mitigated
        simulation sits below the paper's mitigated ibm_cairo <V>
        (-0.2569 +- 0.0063) — the gate noise during the run exceeded the
        Table III calibration. Both readout models mitigate to the same
        gate-noise-only value (readout is exactly undone)."""
        led = dense["deferred"]
        mit, err = fig2["ibm_cairo_mitigated_V"], fig2["ibm_cairo_mitigated_V_error"]
        vals = []
        for ro, out in aer.items():
            v_noisy = out["noisy_exact"]["V"]
            v_mit = out["mitigated"]["V"]
            assert v_mit < v_noisy, (ro, v_mit, v_noisy)
            assert led["V"] < v_mit < mit - 3 * err, (ro, led["V"], v_mit, mit)
            vals.append(v_mit)
            C = out["calibration_matrix"]
            assert C.shape == (4, 4) and np.allclose(C.sum(axis=0), 1.0, atol=1e-12)
        assert abs(vals[0] - vals[1]) < 1e-6, vals

    def test_single_qubit_gate_time_range_is_immaterial(self, qk, fig2):
        """Sec. II.D gives only 20-40 ns: the two ends move every energy
        by < 1e-3 (T1, T2 ~ 100 us)."""
        a = qk.ikeda_ibm_aer(fig2, readout="fidelity", single_qubit_gate_time_ns=20.0,
                             mitigate=False, shots=1000)
        b = qk.ikeda_ibm_aer(fig2, readout="fidelity", single_qubit_gate_time_ns=40.0,
                             mitigate=False, shots=1000)
        worst = max(abs(a["noisy_exact"][key] - b["noisy_exact"][key])
                    for key in ("E0", "H1", "V", "E1"))
        assert worst < 1e-3, f"20 ns vs 40 ns single-qubit gate time: {worst:.2e}"

    def test_readout_matrix_and_mitigation_roundtrip(self, qk):
        A = qk.readout_assignment_matrix(0.01, 0.02)
        assert np.allclose(A.sum(axis=0), 1.0)
        p = {"00": 0.1, "01": 0.2, "10": 0.3, "11": 0.4}
        q = qk.apply_readout(p, A)
        back = qk.mitigate_probabilities(q, A)
        assert max(abs(back[b] - p[b]) for b in BITSTRINGS) < 1e-12


def qk_basis():
    from vacuum.qet.ikeda_qiskit import IBM_BASIS_GATES
    return IBM_BASIS_GATES
