"""Tests for the protocol runner (spec M-I.4, docs/PLAN_LAYERS_2_3.md).

Spec requirements pinned here:

- a trivial quench (piecewise-constant K change and back) runs with a
  closing ledger (1e-9);
- the standing audits all fire and pass on a clean run (passivity on the
  claimed ground state, causality once per (K, t_max) pair, precision on
  every reported reduced state, mandatory ledger closure);
- a cooked run (dropping a work entry) raises AuditViolation at close;
- the step API carries time-dependent generators (callable t -> H_mat,
  absolute protocol time) charging work entries, and channel steps charging
  dissipate entries from before/after mean-energy differences.

Conventions per docs/API.md: hbar = 1, V_vac = I/2, block ordering
R = (x_1..x_N, p_1..p_N). Measured numbers appear in assertion messages
(repo testing convention).
"""

import numpy as np
import pytest

from vacuum.audits import (
    AuditViolation,
    EnergyLedger,
    embed_local_symplectic,
)
from vacuum.core import ground_state_cov, harmonic_chain_K, mean_energy
from vacuum.protocol import (
    ChannelStep,
    GeneratorStep,
    coupling_from_hamiltonian,
    hamiltonian_matrix,
    midpoint_evolve,
    quadratic_mean_energy,
    run_protocol,
)

N = 16
M0, M1 = 1.0, 1.6
LEDGER_TOL = 1e-9  # spec M-I.4 / test matrix: "Ledger closure, all protocols 1e-9"
# The causality audit refuses to run vacuously; on the N = 16 ring (max
# distance 8) the default buffer of 15 leaves no outside pairs, so the tests
# shrink it. Tail size at the checked distances (d >= 7, t <= 1) is
# ~ t^(2d)/(2d)! ~ 1e-11, far below the audit's 1e-8 commutator tolerance.
CAUSALITY_KW = {"buffer": 6.0}


@pytest.fixture(scope="module")
def K0():
    return harmonic_chain_K(N, m=M0)


@pytest.fixture(scope="module")
def K1():
    return harmonic_chain_K(N, m=M1)


@pytest.fixture(scope="module")
def V0(K0):
    return ground_state_cov(K0)


# --------------------------------------------------------------------------
# The trivial quench: K change and back, closing ledger, all audits fire
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def quench_result(V0, K0, K1):
    """Piecewise-constant quench K0 -> K1 -> K0 through the runner."""
    return run_protocol(
        V0,
        K0,
        steps=[(K1, 0.6), (K0, 0.5)],
        claim_ground_entry=True,
        report_modes=[[0, 1], [8]],
        causality_kwargs=CAUSALITY_KW,
    )


def test_trivial_quench_ledger_closes(quench_result, V0, K0):
    res = quench_result
    defect = res.audits["ledger"].details["defect"]
    assert res.audits["ledger"].passed, f"ledger did not close: defect {defect:.3e}"
    assert abs(defect) <= LEDGER_TOL, f"closure defect {defect:.3e} > {LEDGER_TOL}"
    # The quench injects energy: net work equals the energy change (unitary
    # protocol, no dissipation), and it is macroscopic, not roundoff.
    dE = mean_energy(res.V, K0) - mean_energy(V0, K0)
    assert res.dissipated_total == 0.0
    assert abs(dE - res.work_total) <= LEDGER_TOL, (
        f"work_total {res.work_total:.12e} vs Delta E {dE:.12e}"
    )
    assert res.work_total > 1e-3, f"quench work {res.work_total:.3e} suspiciously small"
    kinds = [r["kind"] for r in res.ledger.table]
    assert kinds.count("work") == 2, f"expected 2 work entries, table kinds {kinds}"
    assert kinds.count("snapshot") == 2


def test_trivial_quench_all_audits_fire_and_pass(quench_result):
    res = quench_result
    a = res.audits
    # passivity fired on the claimed ground state and passed
    assert a["passivity_entry"] is not None
    assert a["passivity_entry"].passed, repr(a["passivity_entry"])
    assert a["passivity_entry"].name == "passivity"
    # causality fired once per (K, t_max) pair: K1 for 0.6, K0 for 0.5
    ts = sorted(r.details["t"] for r in a["causality"])
    assert len(a["causality"]) == 2, f"causality audits: {len(a['causality'])}"
    assert ts == pytest.approx([0.5, 0.6]), f"causality t_max values {ts}"
    assert all(r.passed for r in a["causality"]), [repr(r) for r in a["causality"]]
    # precision fired on every reported reduced state
    reported = [d["modes"] for d in a["precision"]]
    assert reported == [(0, 1), (8,)], f"reported blocks {reported}"
    assert all(d["result"].passed for d in a["precision"])
    # mandatory closure ran
    assert a["ledger"] is not None and a["ledger"].name == "energy_ledger"
    assert res.all_passed(), repr(res)


def test_quench_step_log_records_convergence_metadata(quench_result):
    log = quench_result.step_log
    assert [r["kind"] for r in log] == ["generator", "generator"]
    assert all(r["generator_changed"] for r in log)
    assert all(r["causality_audited"] for r in log)
    assert log[0]["t_start"] == 0.0 and log[1]["t_end"] == pytest.approx(1.1)
    assert quench_result.t_final == pytest.approx(1.1)


# --------------------------------------------------------------------------
# The cooked run: a dropped work entry must fail closure loudly
# --------------------------------------------------------------------------


class CookedLedger(EnergyLedger):
    """Mis-instrumented ledger: silently drops every work entry."""

    def work(self, label, amount):
        return float(amount)  # never recorded — the books cannot close


def test_cooked_run_raises_audit_violation_at_close(V0, K0, K1):
    with pytest.raises(AuditViolation) as excinfo:
        run_protocol(
            V0,
            K0,
            steps=[(K1, 0.6), (K0, 0.5)],
            ledger=CookedLedger(K=K0, tol=LEDGER_TOL),
            report_modes=[[0, 1]],
            causality_kwargs=CAUSALITY_KW,
        )
    result = excinfo.value.result
    assert result.name == "energy_ledger", f"raised by {result.name}"
    defect = result.details["defect"]
    assert abs(defect) > 1e-3, f"cooked defect {defect:.3e} should be macroscopic"
    assert result.details["work_in"] == 0.0  # the entries really were dropped


# --------------------------------------------------------------------------
# Inertness and exit-side ground-state claims
# --------------------------------------------------------------------------


def test_ground_state_inert_under_own_hamiltonian(V0, K0):
    """Evolving the ground state under its own K changes nothing.

    The ground-state covariance is stationary; no generator change means no
    work entry; both boundary ground claims hold.
    """
    res = run_protocol(
        V0,
        K0,
        steps=[(K0, 0.7)],
        claim_ground_entry=True,
        claim_ground_exit=True,
        report_modes=[[0, 1, 2]],
        causality_kwargs=CAUSALITY_KW,
    )
    drift = float(np.max(np.abs(res.V - V0)))
    assert drift < 1e-12, f"ground state drifted by {drift:.3e} under its own K"
    kinds = [r["kind"] for r in res.ledger.table]
    assert kinds.count("work") == 0, f"unexpected work entries: {kinds}"
    assert res.audits["passivity_exit"] is not None
    assert res.all_passed(), repr(res)
    defect = res.audits["ledger"].details["defect"]
    assert abs(defect) <= LEDGER_TOL, f"closure defect {defect:.3e}"


def test_claimed_ground_on_active_state_fires_passivity(V0, K0):
    """The ground-state claim is checkable: claiming a squeezed state fires.

    A deterministic sampler returns the exact inverse squeeze, which lowers
    <H> on the squeezed state — the passivity audit must report it before a
    single step runs.
    """
    r = 0.6
    S_sq = embed_local_symplectic(np.diag([np.exp(r), np.exp(-r)]), [0], N)
    V_active = S_sq @ V0 @ S_sq.T

    def unsqueeze_sampler(rng, n_modes):
        return embed_local_symplectic(np.diag([np.exp(-r), np.exp(r)]), [0], n_modes)

    with pytest.raises(AuditViolation) as excinfo:
        run_protocol(
            V_active,
            K0,
            steps=[(K0, 0.1)],
            claim_ground_entry=True,
            passivity_kwargs={"sampler": unsqueeze_sampler, "n_samples": 1},
            report_modes=[[0]],
            causality_kwargs=CAUSALITY_KW,
        )
    assert excinfo.value.result.name == "passivity"
    assert excinfo.value.result.worst_violation > 1e-3


# --------------------------------------------------------------------------
# Causality dedup: once per (K, t_max) pair
# --------------------------------------------------------------------------


def test_causality_once_per_K_tmax_pair(V0, K0, K1):
    """Repeated K pieces accumulate into a single (K, t_max) audit."""
    res = run_protocol(
        V0,
        K0,
        steps=[(K1, 0.3), (K0, 0.4), (K1, 0.3)],
        report_modes=[[0, 1]],
        causality_kwargs=CAUSALITY_KW,
    )
    ts = sorted(r.details["t"] for r in res.audits["causality"])
    assert len(res.audits["causality"]) == 2, (
        f"expected one audit per distinct K, got {len(res.audits['causality'])}"
    )
    assert ts == pytest.approx([0.4, 0.6]), f"t_max values {ts}"
    assert res.all_passed(), repr(res)


# --------------------------------------------------------------------------
# Time-dependent generators (the M2.2 / Floquet contract)
# --------------------------------------------------------------------------


def _ramp(K_a, K_b, t_on, t_off):
    """Smooth coupling ramp K(t): K_a at t_on -> K_b at t_off (cos^2 ease)."""
    def H_of_t(t):
        s = np.sin(0.5 * np.pi * np.clip((t - t_on) / (t_off - t_on), 0.0, 1.0)) ** 2
        return K_a + s * (K_b - K_a)

    return H_of_t


def test_time_dependent_generator_charges_work_and_closes(V0, K0, K1):
    """A driven ramp feeds a work entry and the ledger closes at any dt.

    The midpoint-telescoped work is exactly the energy change of the
    simulated piecewise-constant drive, so closure at 1e-9 holds for both
    substep counts while the work value itself converges under dt-halving.
    """
    works = {}
    for n_sub in (16, 32):
        res = run_protocol(
            V0,
            K0,
            steps=[GeneratorStep(_ramp(K0, K1, 0.0, 0.8), 0.8, n_substeps=n_sub)],
            report_modes=[[0, 1]],
            causality_kwargs=CAUSALITY_KW,
        )
        defect = res.audits["ledger"].details["defect"]
        assert abs(defect) <= LEDGER_TOL, (
            f"n_substeps={n_sub}: closure defect {defect:.3e} > {LEDGER_TOL}"
        )
        kinds = [r["kind"] for r in res.ledger.table]
        assert kinds.count("work") == 1, f"expected 1 work entry, got {kinds}"
        # exit Hamiltonian is K1: snapshot taken there, K carried in result
        assert np.allclose(res.K, K1, atol=1e-12)
        assert res.step_log[0]["time_dependent"]
        assert res.step_log[0]["dt"] == pytest.approx(0.8 / n_sub)
        works[n_sub] = res.work_total
    # dt-halving moves the work only by the Magnus truncation, not the books
    dw = abs(works[32] - works[16])
    assert dw < 5e-3, f"work not converging under dt-halving: |dW| = {dw:.3e}"
    assert works[32] > 0.0, f"ramp-on work should be positive, got {works[32]:.3e}"


def test_time_dependent_generator_gets_absolute_time(V0, K0, K1):
    """The callable receives absolute protocol time (Floquet contract)."""
    seen = []

    def H_of_t(t):
        seen.append(float(t))
        return _ramp(K0, K1, 0.5, 1.5)(t)

    res = run_protocol(
        V0,
        K0,
        steps=[(K0, 0.5), GeneratorStep(H_of_t, 1.0, n_substeps=8)],
        report_modes=[[0]],
        causality_kwargs=CAUSALITY_KW,
    )
    assert res.audits["ledger"].passed
    # endpoint calls at t0 = 0.5 and t_end = 1.5, midpoints strictly inside
    assert seen[0] == pytest.approx(0.5) and seen[-1] == pytest.approx(1.5)
    mids = seen[1:-1]
    assert len(mids) == 8
    expected = [0.5 + (k + 0.5) * (1.0 / 8) for k in range(8)]
    assert mids == pytest.approx(expected), f"midpoint times {mids}"


def test_time_dependent_step_requires_explicit_n_substeps(V0, K0, K1):
    with pytest.raises(ValueError, match="n_substeps"):
        run_protocol(V0, K0, steps=[GeneratorStep(_ramp(K0, K1, 0.0, 1.0), 1.0)])


def test_midpoint_evolve_is_pure(V0, K0, K1):
    """midpoint_evolve neither mutates its input nor hides state."""
    V_in = V0.copy()
    H_of_t = _ramp(K0, K1, 0.0, 0.5)
    V_out, w, H_start, H_end = midpoint_evolve(V_in, H_of_t, 0.0, 0.5, 8)
    assert np.array_equal(V_in, V0), "midpoint_evolve mutated its input"
    assert V_out is not V_in
    assert np.allclose(H_start, hamiltonian_matrix(K0))
    assert np.allclose(H_end, hamiltonian_matrix(K1))
    # telescoped work = exact energy change of the simulated drive
    dE = quadratic_mean_energy(V_out, H_end) - quadratic_mean_energy(V_in, H_start)
    assert abs(w - dE) < 1e-12, f"telescoping broken: W - dE = {w - dE:.3e}"


# --------------------------------------------------------------------------
# Channel steps: dissipate entries from before/after mean energies
# --------------------------------------------------------------------------


def test_channel_step_dissipation_closes(V0, K0, K1):
    """Loss inside a quench books its energy as dissipation; ledger closes."""
    from vacuum.core import loss

    res = run_protocol(
        V0,
        K0,
        steps=[
            (K1, 0.4),
            ChannelStep(loss, args=([3], 0.9), label="loss-mode3"),
            (loss, ([5], 0.8, 0.0)),  # raw-tuple channel spelling
            (K0, 0.4),
        ],
        report_modes=[[3], [5]],
        causality_kwargs=CAUSALITY_KW,
    )
    defect = res.audits["ledger"].details["defect"]
    assert abs(defect) <= LEDGER_TOL, f"closure defect {defect:.3e} > {LEDGER_TOL}"
    kinds = [r["kind"] for r in res.ledger.table]
    assert kinds.count("dissipation") == 2, f"table kinds {kinds}"
    assert kinds.count("work") == 2
    # the recorded dissipation is exactly the before/after energy difference,
    # which is what makes closure automatic (spec M2.2 wording)
    q_logged = [r["dissipated"] for r in res.step_log if r["kind"] == "channel"]
    assert res.dissipated_total == pytest.approx(sum(q_logged))
    assert res.all_passed(), repr(res)


# --------------------------------------------------------------------------
# Precision reporting and the carried flag
# --------------------------------------------------------------------------


def test_precision_flag_carried_for_floor_state(K0):
    """Reporting the full pure state trips the near-floor guard (flag rides).

    A unitary-only protocol keeps the state pure (nu = 1/2 exactly), so the
    default full-state report escalates: flagged=True, mpmath regime,
    physicality passed. Small N keeps the mp eigenproblem cheap.
    """
    n_small = 4
    K_s = harmonic_chain_K(n_small, m=1.0)
    K_s2 = harmonic_chain_K(n_small, m=1.4)
    V_s = ground_state_cov(K_s)
    res = run_protocol(
        V_s,
        K_s,
        steps=[(K_s2, 0.3), (K_s, 0.2)],
        report_modes=None,  # default: the full state is the report
        run_causality=False,  # 4-site ring has no outside-cone pairs
    )
    assert len(res.audits["precision"]) == 1
    det = res.audits["precision"][0]["result"].details
    assert det["flagged"] is True, f"floor state not flagged: {det}"
    assert det["regime"] == "mpmath"
    assert res.flagged is True
    assert res.all_passed(), repr(res)


def test_reported_blocks_off_floor_are_unflagged(quench_result):
    """Small interior blocks of the quenched chain sit off the floor."""
    for d in quench_result.audits["precision"]:
        det = d["result"].details
        assert det["min_nu"] > 0.5, f"block {d['modes']}: min nu {det['min_nu']}"
    assert quench_result.flagged is False


# --------------------------------------------------------------------------
# Step normalization and guardrails
# --------------------------------------------------------------------------


def test_raw_tuple_normalization_and_bad_step_rejected(V0, K0):
    with pytest.raises(TypeError, match="step 0"):
        run_protocol(V0, K0, steps=["not a step"])
    with pytest.raises(ValueError, match="duration must be >= 0"):
        run_protocol(V0, K0, steps=[(K0, -1.0)])


def test_zero_duration_step_is_pure_sudden_quench(V0, K0, K1):
    """(K1, 0.0) charges the sudden work but evolves nothing."""
    res = run_protocol(
        V0,
        K0,
        steps=[(K1, 0.0), (K0, 0.0)],
        report_modes=[[0]],
        run_causality=True,  # zero-duration pieces contribute no (K, t) pair
        causality_kwargs=CAUSALITY_KW,
    )
    assert len(res.audits["causality"]) == 0
    assert float(np.max(np.abs(res.V - V0))) == 0.0
    # forward and reverse sudden works cancel on the unchanged state
    assert res.work_total == pytest.approx(0.0, abs=1e-12)
    assert res.audits["ledger"].passed


def test_coupling_from_hamiltonian_roundtrip(K0):
    K_back = coupling_from_hamiltonian(hamiltonian_matrix(K0))
    assert np.array_equal(K_back, 0.5 * (K0 + K0.T))
    H_bad = hamiltonian_matrix(K0)
    H_bad[0, N] = 0.3  # x-p coupling: not diag(K, I) form
    assert coupling_from_hamiltonian(0.5 * (H_bad + H_bad.T)) is None
