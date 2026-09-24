"""Tests for the nonperturbative Gaussian detector stack (spec M2.2).

Spec anchors pinned here (docs/PLAN_LAYERS_2_3.md, M2.2 + test matrix):

- lambda = 0 protocol inertness: V constant to 1e-14, zero drive work;
- adiabatic ramp on/off returns the detectors to ground: E_N < 1e-10,
  <n_d> < 1e-8;
- sudden strong coupling produces nonzero E_N that the dt-halving gate
  confirms converged (converged dt recorded);
- ledger closes to 1e-9 on every protocol, with the switching drive
  charged as work — the telescoped discrete sum of <dH/dt> dt — and the
  independent analytic-derivative quadrature of <dH/dt> tested against
  Delta E on a closed run (second-order convergence demonstrated);
- channel interleaving books `dissipate` entries from before/after mean
  energies and the books still close to 1e-9;
- the MANDATORY mp-margin rule: PT min nu within 1e-10 of 1/2 escalates
  to symplectic_margins_mp, and the precision `flagged` field rides in
  every result row;
- initial states from ground_state_cov(blockdiag)/thermal_state_cov with
  the passivity audit on every claimed-T=0 initial state.

Conventions per docs/API.md: hbar = 1, V_vac = I/2, block ordering
R = (x_1..x_N, p_1..p_N), nats.  Measured numbers appear in assertion
messages (repo testing convention).
"""

import numpy as np
import pytest

from vacuum.core import (
    ground_state_cov,
    harmonic_chain_K,
    log_negativity,
    loss,
    mean_energy,
    reduce as reduce_cov,
    thermal_state_cov,
)
from vacuum.detectors import (
    Smearing,
    attach_detectors,
    detector_block,
    detector_coupling_derivative,
    detector_occupations,
    harvested_log_negativity,
    initial_state,
    profile_matrix,
    protocol_evolve,
    protocol_evolve_fixed,
    run_harvesting,
    switching,
)
from vacuum.protocol import midpoint_evolve

N_F = 8
M_FIELD = 1.0
SITES = (1, 5)
LEDGER_TOL = 1e-9  # spec test matrix: "Ledger closure, all protocols 1e-9"


@pytest.fixture(scope="module")
def K_field():
    return harmonic_chain_K(N_F, m=M_FIELD)


# --------------------------------------------------------------------------
# attach_detectors: structure, profiles, guardrails, 'xp' design stub
# --------------------------------------------------------------------------


def test_attach_detectors_structure(K_field):
    gaps = (1.3, 2.0)
    lam = (0.2, 0.7)
    K_tot = attach_detectors(K_field, SITES, gaps, lam)
    assert K_tot.shape == (N_F + 2, N_F + 2)
    assert np.array_equal(K_tot, K_tot.T)
    assert np.array_equal(K_tot[:N_F, :N_F], K_field)
    # one column per detector: lam_d at the detector's site, zero elsewhere
    for d, (site, g, l) in enumerate(zip(SITES, gaps, lam)):
        col = K_tot[:N_F, N_F + d]
        assert col[site] == l, f"detector {d} coupling {col[site]} != {l}"
        assert np.count_nonzero(col) == 1
        assert K_tot[N_F + d, N_F + d] == pytest.approx(g**2)
    # detector-detector block is diagonal (no direct coupling)
    assert K_tot[N_F, N_F + 1] == 0.0
    # scalar lambda broadcasts; lam = 0 is exactly block-diagonal
    K0 = attach_detectors(K_field, SITES, gaps, 0.0)
    assert np.array_equal(K0[:N_F, N_F:], np.zeros((N_F, 2)))


def test_attach_detectors_smearing_profile(K_field):
    F_sm = Smearing("gaussian", center=3.0, width=1.0, N=N_F)
    K_tot = attach_detectors(K_field, [F_sm, 6], (1.0, 1.0), 0.5)
    assert np.allclose(K_tot[:N_F, N_F], 0.5 * F_sm.F, atol=0.0)
    assert np.sum(F_sm.F) == pytest.approx(1.0)
    P = profile_matrix(N_F, [F_sm, 6])
    assert P.shape == (N_F, 2)
    assert np.array_equal(P[:, 0], F_sm.F)
    assert P[6, 1] == 1.0 and np.count_nonzero(P[:, 1]) == 1


def test_attach_detectors_guardrails(K_field):
    with pytest.raises(ValueError, match="site"):
        attach_detectors(K_field, (1, N_F), (1.0, 1.0))
    with pytest.raises(ValueError, match="gaps"):
        attach_detectors(K_field, SITES, (1.0,))  # one gap, two detectors
    with pytest.raises(ValueError, match="> 0"):
        attach_detectors(K_field, SITES, (1.0, 0.0))
    with pytest.raises(ValueError, match="profile shape"):
        attach_detectors(K_field, [np.ones(N_F + 1)], (1.0,))
    with pytest.raises(ValueError, match="coupling"):
        attach_detectors(K_field, SITES, (1.0, 1.0), coupling="pp")


def test_attach_detectors_xp_is_the_full_quadratic_form(K_field):
    """'xp' (derivative coupling) returns H_mat with the x_d-p_i blocks filled.

    The interaction lam x_d F^T p_field sits off the block diagonal of
    H_mat = [[diag(K, Omega^2), C], [C^T, I]]; at lam = 0 the form is
    exactly hamiltonian_matrix(K_tot) of the 'xx' path (the shared initial
    state).  The full derivative-coupling anchors live in
    tests/test_xp_coupling.py and tests/test_gate_xp.py.
    """
    from vacuum.protocol import hamiltonian_matrix

    gaps = (1.3, 2.0)
    lam = (0.2, 0.7)
    H = attach_detectors(K_field, SITES, gaps, lam, coupling="xp")
    n_tot = N_F + 2
    assert H.shape == (2 * n_tot, 2 * n_tot)
    assert np.array_equal(H, H.T)
    # x-x block: the uncoupled blockdiag(K, Omega^2); p-p block: identity
    assert np.array_equal(H[:n_tot, :n_tot], attach_detectors(K_field, SITES, gaps, 0.0))
    assert np.array_equal(H[n_tot:, n_tot:], np.eye(n_tot))
    # x_d - p_i coupling: one entry per detector at (p_site, x_d), nothing else
    C = H[n_tot:, :n_tot]
    for d_, (site, l) in enumerate(zip(SITES, lam)):
        assert C[site, N_F + d_] == l, f"detector {d_}: C[{site}, {N_F + d_}] != {l}"
    assert np.count_nonzero(C) == 2
    assert np.array_equal(H[:n_tot, n_tot:], C.T)
    # lam = 0 collapses to the 'xx' form's H_mat exactly
    assert np.array_equal(
        attach_detectors(K_field, SITES, gaps, 0.0, coupling="xp"),
        hamiltonian_matrix(attach_detectors(K_field, SITES, gaps, 0.0)),
    )


# --------------------------------------------------------------------------
# Initial states: ground_state_cov(blockdiag) / thermal, passivity audited
# --------------------------------------------------------------------------


def test_initial_state_ground_is_audited(K_field):
    init = initial_state(K_field, SITES, (1.0, 2.0))
    K_tot0 = attach_detectors(K_field, SITES, (1.0, 2.0), 0.0)
    assert np.array_equal(init.K_tot, K_tot0)
    assert np.array_equal(init.V, ground_state_cov(K_tot0))
    # spec M2.2: the passivity audit runs on the T = 0 state EVERY time
    assert init.passivity is not None
    assert init.passivity.name == "passivity"
    assert init.passivity.passed, repr(init.passivity)


def test_initial_state_thermal_field_ground_detectors(K_field):
    T = 0.8
    gaps = (1.0, 2.0)
    init = initial_state(K_field, SITES, gaps, T=T)
    assert init.passivity is None  # thermal state: no ground-state claim
    n_tot = N_F + 2
    # field block is exactly thermal_state_cov(K, T)
    Vf = reduce_cov(init.V, range(N_F))
    assert np.array_equal(Vf, thermal_state_cov(K_field, T))
    # detector modes are vacuum: diag(1/(2 Omega), Omega/2), zero occupation
    for d, g in enumerate(gaps):
        i = N_F + d
        assert init.V[i, i] == pytest.approx(0.5 / g, rel=1e-14)
        assert init.V[n_tot + i, n_tot + i] == pytest.approx(0.5 * g, rel=1e-14)
    occ = detector_occupations(init.V, N_F, gaps)
    assert np.max(np.abs(occ)) < 1e-14, f"detector occupations {occ}"
    # field-detector cross blocks vanish (product state)
    idx_f = np.concatenate([np.arange(N_F), n_tot + np.arange(N_F)])
    idx_d = np.concatenate([N_F + np.arange(2), n_tot + N_F + np.arange(2)])
    assert np.max(np.abs(init.V[np.ix_(idx_f, idx_d)])) == 0.0


def test_detector_occupation_matches_bose(K_field):
    """<n_d> of a thermal detector equals the Bose occupation exactly."""
    omega, gap, T = 1.3, 2.0, 1.1
    V = thermal_state_cov(np.diag([omega**2, gap**2]), T)
    occ = detector_occupations(V, 1, (gap,))
    nbar = 1.0 / np.expm1(gap / T)
    assert occ[0] == pytest.approx(nbar, rel=1e-12), (
        f"occupation {occ[0]:.15e} vs Bose {nbar:.15e}"
    )


# --------------------------------------------------------------------------
# lambda = 0 inertness (spec: 1e-14)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def inert_run(K_field):
    return run_harvesting(
        K_field,
        SITES,
        (1.0, 1.0),
        lambda t: 0.0,
        ts=[0.0, 1.0, 2.0, 3.0],
    )


def test_lambda_zero_inert(inert_run, K_field):
    res = inert_run
    V0 = ground_state_cov(attach_detectors(K_field, SITES, (1.0, 1.0), 0.0))
    drift = float(np.max(np.abs(res.V - V0)))
    assert drift < 1e-14, f"lambda = 0 drift {drift:.3e} >= 1e-14 (spec M2.2)"
    assert res.work == 0.0, f"drive work {res.work:.3e} != 0 with lambda = 0"
    assert res.dissipated == 0.0
    assert res.converged, "gate did not converge on the trivial protocol"
    assert res.dt_converged > 0.0
    defect = res.ledger_result.details["defect"]
    assert abs(defect) <= LEDGER_TOL, f"ledger defect {defect:.3e}"


def test_lambda_zero_negativity_flagged_at_floor(inert_run):
    """The uncoupled vacuum sits ON the floor: mp-margin rule must fire."""
    res = inert_run
    assert res.flagged is True, "PT spectrum at the floor was not flagged"
    assert res.neg.regime == "mpmath"
    assert res.E_N < 1e-12, f"E_N {res.E_N:.3e} on the uncoupled vacuum"
    assert abs(res.neg.margin_min) < 1e-13, (
        f"PT margin {res.neg.margin_min:.3e} on the uncoupled vacuum"
    )


# --------------------------------------------------------------------------
# Adiabatic ramp on/off returns the detectors to ground
# (spec: E_N < 1e-10, <n_d> < 1e-8)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def adiabatic_run(K_field):
    lam_max = 0.05
    chi = switching("gaussian", 2.5, t0=0.0)
    t_a, t_b = chi.support
    return run_harvesting(
        K_field,
        SITES,
        (2.0, 2.0),
        lambda t: lam_max * chi(t),
        ts=np.linspace(t_a, t_b, 41),
        dlambda_of_t=lambda t: lam_max * chi.derivative(t),
    )


def test_adiabatic_return(adiabatic_run):
    res = adiabatic_run
    assert res.converged, "adiabatic run failed the dt-halving gate"
    assert res.E_N < 1e-10, f"adiabatic E_N {res.E_N:.3e} >= 1e-10 (spec M2.2)"
    n_max = float(np.max(res.n_d))
    assert n_max < 1e-8, f"adiabatic <n_d> {n_max:.3e} >= 1e-8 (spec M2.2)"
    # near-floor spectrum: the mp-margin rule must have handled E_N
    assert res.flagged is True and res.neg.regime == "mpmath"
    defect = res.ledger_result.details["defect"]
    assert abs(defect) <= LEDGER_TOL, f"ledger defect {defect:.3e}"


def test_adiabatic_metadata_recorded(adiabatic_run):
    """The harness records the dt at which the result converged."""
    res = adiabatic_run
    ev = res.evolve
    assert ev.dt_converged == pytest.approx(
        np.max(np.diff(np.linspace(*switching("gaussian", 2.5).support, 41)))
        / 2**ev.halvings
    )
    assert ev.movement is not None and ev.movement < 1e-9
    assert len(ev.history) == ev.halvings + 1
    assert ev.history[-1]["dt_max"] == ev.dt_converged
    assert res.passivity is not None and res.passivity.passed


# --------------------------------------------------------------------------
# Sudden strong coupling: nonzero E_N, gate-confirmed converged
# --------------------------------------------------------------------------

SUDDEN_LAM = 0.8
SUDDEN_T_RAMP = 0.4
SUDDEN_T_END = 9.0


def _sudden_lambda(t):
    """Fast C^1 cos^2 ramp 0 -> SUDDEN_LAM over [0, T_ramp], then hold."""
    u = min(float(t), SUDDEN_T_RAMP)
    return SUDDEN_LAM * np.cos(
        0.5 * np.pi * (u - SUDDEN_T_RAMP) / SUDDEN_T_RAMP
    ) ** 2


@pytest.fixture(scope="module")
def sudden_run(K_field):
    ts = np.concatenate(
        [
            np.linspace(0.0, SUDDEN_T_RAMP, 9),
            np.linspace(SUDDEN_T_RAMP, SUDDEN_T_END, 44)[1:],
        ]
    )
    return run_harvesting(
        K_field, SITES, (1.0, 1.0), _sudden_lambda, ts, max_halvings=14
    )


def test_sudden_coupling_nonzero_converged_EN(sudden_run):
    res = sudden_run
    assert res.converged, "sudden run failed the dt-halving gate"
    assert res.E_N > 0.1, f"sudden E_N {res.E_N:.6e} not macroscopically nonzero"
    # measured on this rig: E_N ~ 1.5425e-1 at (lam=0.8, gaps=1, t=9)
    assert res.E_N == pytest.approx(0.15424794, abs=5e-6), (
        f"sudden E_N {res.E_N:.8e} moved off its pinned anchor"
    )
    # comfortably off the floor: float64 regime, unflagged, margin macroscopic
    assert res.flagged is False and res.neg.regime == "float64"
    assert res.neg.margin_min < -1e-3
    defect = res.ledger_result.details["defect"]
    assert abs(defect) <= LEDGER_TOL, f"ledger defect {defect:.3e}"


def test_sudden_convergence_metadata(sudden_run):
    ev = sudden_run.evolve
    assert ev.movement < 1e-9, f"gate accepted at movement {ev.movement:.3e}"
    assert ev.halvings >= 1
    # the recorded dt is the accepted run's coarsest substep
    base_dt = (SUDDEN_T_END - SUDDEN_T_RAMP) / 43
    assert ev.dt_converged == pytest.approx(base_dt / 2**ev.halvings)
    moves = [h["movement"] for h in ev.history[1:]]
    assert all(m is not None for m in moves)
    assert moves[-1] < 1e-9 and all(m >= 1e-9 for m in moves[:-1])


# --------------------------------------------------------------------------
# Ledger: telescoped drive work closes to 1e-9; the analytic <dH/dt>
# quadrature converges to Delta E at second order on a closed run
# --------------------------------------------------------------------------

DRIVE_GAPS = (2.0, 2.0)
DRIVE_LAM = 0.4
DRIVE_CHI = switching("cos2", 4.0, t0=4.0)  # compact bump on [0, 8]


def _drive_lambda(t):
    return DRIVE_LAM * DRIVE_CHI(t)


def _drive_dlambda(t):
    return DRIVE_LAM * DRIVE_CHI.derivative(t)


@pytest.fixture(scope="module")
def drive_fixed_runs(K_field):
    """Fixed-resolution closed runs at splits 32 and 64 over 32 intervals."""
    V0 = ground_state_cov(attach_detectors(K_field, SITES, DRIVE_GAPS, 0.0))

    def K_of_t(t):
        return attach_detectors(K_field, SITES, DRIVE_GAPS, _drive_lambda(t))

    def dK_of_t(t):
        return detector_coupling_derivative(N_F, SITES, _drive_dlambda(t))

    ts = np.linspace(0.0, 8.0, 33)
    runs = {
        s: protocol_evolve_fixed(V0, K_of_t, ts, splits=s, dK_of_t=dK_of_t)
        for s in (32, 64)
    }
    return V0, runs


def test_ledger_closes_with_telescoped_drive_work(drive_fixed_runs, K_field):
    """W = discrete sum of <dH/dt> dt (telescoped) closes against Delta E.

    The telescoped midpoint sum is exactly the energy change of the
    simulated piecewise-constant drive, so closure holds at 1e-9 (in fact
    roundoff) at ANY dt — closure tests instrumentation, the gate tests
    physics (vacuum.protocol discretization contract).
    """
    V0, runs = drive_fixed_runs
    E0 = mean_energy(V0, attach_detectors(K_field, SITES, DRIVE_GAPS, 0.0))
    for s, run in runs.items():
        dE = mean_energy(
            run.V, attach_detectors(K_field, SITES, DRIVE_GAPS, _drive_lambda(8.0))
        ) - E0
        defect = run.work - dE
        assert abs(defect) <= LEDGER_TOL, (
            f"splits={s}: telescoped work {run.work:.12e} vs Delta E "
            f"{dE:.12e}, defect {defect:.3e} > {LEDGER_TOL}"
        )
        assert run.work > 1e-6, f"drive work {run.work:.3e} suspiciously small"


def test_derivative_drive_work_converges_to_delta_E(drive_fixed_runs, K_field):
    """The analytic lam'(t) quadrature of <dH/dt> -> Delta E at O(dt^2)."""
    V0, runs = drive_fixed_runs
    E0 = mean_energy(V0, attach_detectors(K_field, SITES, DRIVE_GAPS, 0.0))
    errs = {}
    for s, run in runs.items():
        dE = mean_energy(
            run.V, attach_detectors(K_field, SITES, DRIVE_GAPS, _drive_lambda(8.0))
        ) - E0
        errs[s] = abs(run.work_deriv - dE)
    # second order: halving dt cuts the error by ~4
    ratio = errs[32] / errs[64]
    assert 3.3 < ratio < 4.7, (
        f"<dH/dt> quadrature not O(dt^2): errors {errs[32]:.3e} -> "
        f"{errs[64]:.3e}, ratio {ratio:.2f}"
    )
    assert errs[64] < 1e-9, (
        f"|W_deriv - Delta E| = {errs[64]:.3e} at n=2048 steps (>= 1e-9)"
    )


def test_drive_run_through_full_stack_ledger(K_field):
    """The same closed drive through run_harvesting: books close, work rides."""
    res = run_harvesting(
        K_field,
        SITES,
        DRIVE_GAPS,
        _drive_lambda,
        ts=np.linspace(0.0, 8.0, 33),
        dlambda_of_t=_drive_dlambda,
    )
    assert res.converged
    defect = res.ledger_result.details["defect"]
    assert abs(defect) <= LEDGER_TOL, f"ledger defect {defect:.3e} > 1e-9"
    kinds = [r["kind"] for r in res.ledger.table]
    assert kinds.count("work") == 1 and kinds.count("snapshot") == 2
    assert kinds.count("dissipation") == 0
    # closed run: the derivative-quadrature work agrees with the ledger's
    # telescoped work at the converged dt (both estimate the same integral)
    assert res.work_deriv is not None
    assert abs(res.work_deriv - res.work) < 1e-6, (
        f"W_deriv {res.work_deriv:.9e} vs W_ledger {res.work:.9e}"
    )


# --------------------------------------------------------------------------
# Channel interleaving: dissipate entries from before/after mean energies
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def channel_run(K_field):
    kappa = 0.3
    chi = switching("cos2", 3.0, t0=3.0)  # bump on [0, 6]

    def ch_loss(V, t, dt):
        # detector-mode loss; the callable owns its dt-scaling (Trotter)
        return loss(V, [N_F, N_F + 1], np.exp(-kappa * dt))

    return run_harvesting(
        K_field,
        SITES,
        (1.0, 1.0),
        lambda t: 0.5 * chi(t),
        ts=np.linspace(0.0, 6.0, 25),
        channels=(ch_loss,),
    )


def test_channel_interleaving_ledger_closes(channel_run):
    res = channel_run
    assert res.converged, "open (Strang) run failed the dt-halving gate"
    defect = res.ledger_result.details["defect"]
    assert abs(defect) <= LEDGER_TOL, (
        f"open-dynamics ledger defect {defect:.3e} > {LEDGER_TOL}"
    )
    kinds = [r["kind"] for r in res.ledger.table]
    assert kinds.count("dissipation") == 1, f"ledger kinds {kinds}"
    assert kinds.count("work") == 1 and kinds.count("snapshot") == 2
    # the dissipation entry is the summed before/after energy differences
    q = [r["value"] for r in res.ledger.table if r["kind"] == "dissipation"][0]
    assert q == pytest.approx(res.dissipated)
    assert res.dissipated != 0.0, "loss channel booked zero energy flow"
    assert res.E_N >= 0.0 and np.all(res.n_d > 0.0)


# --------------------------------------------------------------------------
# Finite-temperature field: the thermal_state_cov initial-state path, run
# end to end (spec M2.2 initial states; the temperature axis of M2.7)
# --------------------------------------------------------------------------

THERMAL_TS = (0.0, 0.15, 0.6)


@pytest.fixture(scope="module")
def thermal_runs(K_field):
    chi = switching("cos2", 3.0, t0=3.0)  # compact bump on [0, 6]
    return [
        run_harvesting(
            K_field,
            SITES,
            (1.0, 1.0),
            lambda t, _c=chi: 0.6 * _c(t),
            ts=np.linspace(0.0, 6.0, 25),
            T=T,
        )
        for T in THERMAL_TS
    ]


def test_thermal_field_degrades_harvesting(thermal_runs):
    """Field temperature kills harvested entanglement and heats the detectors.

    Measured on this rig (N=8, m=1, sites (1,5), gaps 1, lam 0.6, cos2 T=3):
    E_N = 5.695e-3 -> 5.410e-3 -> 0 (sudden death) and <n_d> = 3.598e-3 ->
    3.750e-3 -> 4.820e-2 across T = 0, 0.15, 0.6.  Only the ordering and
    the death are asserted; the numbers ride in the messages.
    """
    E = [r.E_N for r in thermal_runs]
    n = [float(r.n_d[0]) for r in thermal_runs]
    assert all(r.converged for r in thermal_runs), "a thermal run failed the gate"
    assert E[0] > 1e-3, f"T=0 harvest E_N {E[0]:.6e} too small to be a baseline"
    assert E[0] > E[1] > 0.0, f"E_N not degrading with T: {E}"
    assert E[2] == 0.0, f"E_N {E[2]:.3e} survived T={THERMAL_TS[2]} (expected death)"
    assert n[0] < n[1] < n[2], f"<n_d> not increasing with field T: {n}"
    # the separable hot run is genuinely off the floor, not a flagged zero
    assert thermal_runs[2].neg.min_nu_pt > 0.5 + 1e-3
    assert thermal_runs[2].flagged is False


def test_thermal_runs_close_their_books(thermal_runs):
    """Ledger closure holds on the thermal path too (spec: all protocols)."""
    for T, res in zip(THERMAL_TS, thermal_runs):
        defect = res.ledger_result.details["defect"]
        assert abs(defect) <= LEDGER_TOL, (
            f"T={T}: ledger defect {defect:.3e} > {LEDGER_TOL}"
        )
        # ground-state claim is audited at T = 0 only (a thermal state makes
        # no ground-state claim, so the Pusz-Woronowicz audit does not apply)
        if T == 0.0:
            assert res.passivity is not None and res.passivity.passed
        else:
            assert res.passivity is None


# --------------------------------------------------------------------------
# harvested_log_negativity: the mandatory mp-margin rule
# --------------------------------------------------------------------------


def _pair_ground(g):
    """Ground state of two unit-frequency modes with x-x coupling g.

    K2 = [[1, g], [g, 1]]: normal modes omega_pm = sqrt(1 -/+ g); the
    partial transpose has min symplectic eigenvalue
    nu~ = (1/2)((1-g)/(1+g))^(1/4), so E_N = (1/4) ln((1+g)/(1-g))
    (from V = blockdiag(K^(-1/2)/2, K^(1/2)/2) and the p-sign flip;
    verified against vacuum.core.log_negativity below).
    """
    return ground_state_cov(np.array([[1.0, g], [g, 1.0]]))


def test_negativity_float64_path_matches_core_and_closed_form():
    g = 1e-2
    V = _pair_ground(g)
    res = harvested_log_negativity(V)
    assert res.flagged is False and res.regime == "float64"
    E_core = log_negativity(V, [0], [1])
    E_closed = 0.25 * np.log((1.0 + g) / (1.0 - g))
    assert res.E_N == pytest.approx(E_core, rel=0.0, abs=0.0), (
        f"harvested {res.E_N!r} != core {E_core!r} (same float64 path)"
    )
    assert res.E_N == pytest.approx(E_closed, rel=1e-10), (
        f"E_N {res.E_N:.12e} vs closed form {E_closed:.12e}"
    )


def test_negativity_mp_margin_rule_fires_near_floor():
    """PT min nu within 1e-10 of 1/2 -> margins via mpmath (spec MANDATORY)."""
    g = 1e-12  # PT margin ~ -2.5e-13: far below float64 trust, flag zone
    V = _pair_ground(g)
    res = harvested_log_negativity(V)
    assert res.flagged is True, "near-floor PT spectrum was not flagged"
    assert res.regime == "mpmath"
    E_closed = 0.25 * np.log1p(2.0 * g / (1.0 - g))  # ~ g/2
    assert res.E_N == pytest.approx(E_closed, rel=1e-2), (
        f"mp-margin E_N {res.E_N:.6e} vs closed form {E_closed:.6e} "
        f"(margin {res.margin_min:.3e})"
    )
    assert res.margin_min == pytest.approx(-g / 4.0, rel=1e-2)


def test_negativity_product_vacuum_is_flagged_zero():
    V = 0.5 * np.eye(4)  # two-mode product vacuum, PT exactly at the floor
    res = harvested_log_negativity(V)
    assert res.flagged is True and res.regime == "mpmath"
    assert res.E_N == pytest.approx(0.0, abs=1e-14), f"E_N {res.E_N:.3e}"


def test_negativity_guardrails():
    V = 0.5 * np.eye(4)
    with pytest.raises(ValueError, match="disjoint"):
        harvested_log_negativity(V, [0], [0, 1])
    with pytest.raises(ValueError, match="cover"):
        harvested_log_negativity(V, [0], [])
    with pytest.raises(ValueError, match="both"):
        harvested_log_negativity(V, [0], None)


# --------------------------------------------------------------------------
# detector_block, stepper cross-checks, gate guardrails, purity
# --------------------------------------------------------------------------


def test_detector_block_is_reduce(sudden_run):
    V = sudden_run.V
    assert np.array_equal(
        detector_block(V, N_F), reduce_cov(V, [N_F, N_F + 1])
    )
    with pytest.raises(ValueError, match="detector"):
        detector_block(V, N_F + 2)


def test_fixed_stepper_matches_protocol_midpoint_evolve(K_field):
    """M2.2 rides the M-I.4 midpoint contract: identical substep layout,
    identical state and telescoped work (spec: use vacuum.protocol)."""
    V0 = ground_state_cov(attach_detectors(K_field, SITES, DRIVE_GAPS, 0.0))

    def K_of_t(t):
        return attach_detectors(K_field, SITES, DRIVE_GAPS, _drive_lambda(t))

    run = protocol_evolve_fixed(V0, K_of_t, [0.0, 8.0], splits=64)
    V_ref, w_ref, H_start, H_end = midpoint_evolve(V0, K_of_t, 0.0, 8.0, 64)
    dV = float(np.max(np.abs(run.V - V_ref)))
    assert dV < 1e-12, f"stepper mismatch vs midpoint_evolve: {dV:.3e}"
    assert run.work == pytest.approx(w_ref, abs=1e-12)
    assert np.allclose(run.H_start, H_start) and np.allclose(run.H_end, H_end)


def test_protocol_evolve_is_pure(K_field):
    V0 = ground_state_cov(attach_detectors(K_field, SITES, (1.0, 1.0), 0.0))
    V_in = V0.copy()
    protocol_evolve(V_in, lambda t: attach_detectors(K_field, SITES, (1.0, 1.0), 0.0),
                    [0.0, 1.0], max_halvings=2)
    assert np.array_equal(V_in, V0), "protocol_evolve mutated its input"


def test_gate_failure_raises_and_soft_mode_reports(K_field):
    K_of_t = lambda t: attach_detectors(K_field, SITES, (1.0, 1.0), 0.0)
    V0 = ground_state_cov(K_of_t(0.0))
    # max_halvings=0: only one level exists, no comparison -> cannot converge
    with pytest.raises(RuntimeError, match="did not converge"):
        protocol_evolve(V0, K_of_t, [0.0, 1.0], max_halvings=0)
    ev = protocol_evolve(
        V0, K_of_t, [0.0, 1.0], max_halvings=0, require_convergence=False
    )
    assert ev.converged is False and ev.movement is None
    with pytest.raises(ValueError, match="increasing"):
        protocol_evolve(V0, K_of_t, [1.0, 0.0])
    with pytest.raises(ValueError, match="conv_tol"):
        protocol_evolve(V0, K_of_t, [0.0, 1.0], conv_tol=0.0)
