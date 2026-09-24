"""Layer 5 x Layer 2 composite: pump-enhanced harvesting through run_protocol.

Pinned:
- every point (unpumped baseline and pumped) runs through
  ``vacuum.protocol.run_protocol`` with the passivity claim on the entry
  vacuum, the precision audit on the reported detector block and the
  mandatory ledger closure (1e-9), and the dt-halving gate converges on
  E_N and the detector occupations at 1e-9;
- a depth-0 drive reproduces the baseline to roundoff;
- the pumped point charges positive pump work and positive switching work,
  and its harvested E_N carries the mp-margin flag field;
- switching the pump off during coupling is a different protocol (the
  ledger says so) with its own converged E_N.
"""

import numpy as np
import pytest

from vacuum.core import harmonic_chain_K
from vacuum.floquet import modulated_K, pumped_harvest

N_F = 12
GAP = 1.2
SITES = (3, 8)
LAM = 0.4
OMEGA_MOD = 2 * GAP
CONV_TOL = 1e-9


@pytest.fixture(scope="module")
def field():
    return harmonic_chain_K(N_F, 0.5, bc="dirichlet")


@pytest.fixture(scope="module")
def baseline(field):
    return pumped_harvest(field, None, SITES, (GAP, GAP), LAM, 2, 3,
                          period=2 * np.pi / OMEGA_MOD, n_substeps_per_period=8,
                          conv_tol=CONV_TOL, max_halvings=6)


@pytest.fixture(scope="module")
def pumped(field):
    drive = modulated_K(field, 0.1, OMEGA_MOD, "sin")
    return pumped_harvest(field, drive, SITES, (GAP, GAP), LAM, 2, 3,
                          n_substeps_per_period=8, conv_tol=CONV_TOL, max_halvings=6)


def test_baseline_is_audited_and_converged(baseline):
    b = baseline
    assert b.converged and b.movement < CONV_TOL
    assert b.all_audits_passed and b.passivity_passed
    assert b.ledger_defect < 1e-9, f"ledger defect {b.ledger_defect:.2e}"
    assert b.E_N > 0.0 and b.work_pump == 0.0 and b.work_switching > 0.0
    assert b.result.audits["precision"][0]["modes"] == (N_F, N_F + 1)
    assert np.all(b.n_d >= 0.0)


def test_depth_zero_drive_equals_baseline(field, baseline):
    zero = modulated_K(field, 0.0, OMEGA_MOD, "sin")
    z = pumped_harvest(field, zero, SITES, (GAP, GAP), LAM, 2, 3,
                       n_substeps_per_period=baseline.n_substeps_per_period,
                       conv_tol=CONV_TOL, max_halvings=0, require_convergence=False)
    assert abs(z.E_N - baseline.E_N) < 1e-12, f"{z.E_N} vs {baseline.E_N}"
    assert abs(z.work_total - baseline.work_total) < 1e-12


def test_pumped_point_is_audited_and_charges_the_pump(pumped, baseline):
    p = pumped
    assert p.converged and p.movement < CONV_TOL
    assert p.all_audits_passed and p.passivity_passed
    assert p.ledger_defect < 1e-9, f"ledger defect {p.ledger_defect:.2e}"
    assert p.work_pump > 0.0 and p.work_switching > 0.0
    assert abs(p.work_total - (p.work_pump + p.work_switching)) < 1e-9
    assert p.E_N >= 0.0 and isinstance(p.flagged, bool)
    assert p.field_occupations.shape == (N_F,) and np.sum(p.field_occupations) > 0.0
    # the field pair nearest the detector gap has been pumped
    j = int(np.argmin(np.abs(p.field_omega - GAP)))
    assert p.field_occupations[j] > 0.01
    # report the enhancement factor in the message (the map itself lives in papers/)
    assert p.E_N > 0.0, f"pumped E_N {p.E_N:.4e} vs baseline {baseline.E_N:.4e}"


def test_pump_off_during_coupling_is_a_different_protocol(field, pumped):
    drive = modulated_K(field, 0.1, OMEGA_MOD, "sin")
    q = pumped_harvest(field, drive, SITES, (GAP, GAP), LAM, 2, 3, pump_during_coupling=False,
                       n_substeps_per_period=8, conv_tol=CONV_TOL, max_halvings=6)
    assert q.converged and q.all_audits_passed
    assert abs(q.work_pump - pumped.work_pump) < 1e-9  # same preparation
    assert q.work_switching != pytest.approx(pumped.work_switching, abs=1e-6)
    assert q.E_N >= 0.0


# --------------------------------------------------------------------------
# The M2.5 split on the Floquet-prepared kernel (composite.pumped_harvest_split)
# --------------------------------------------------------------------------
#
# Pinned:
# - at depth 0 the two-time kernel S(t)(V_vac + i Omega/2)S(t')^T is the
#   stationary vacuum Wightman function and the grid split reproduces
#   ``communication_split`` on ``LatticeWightman`` with the same cos^2 window
#   (M to 1e-7 relative, M_comm to 1e-10 absolute, P_A, C likewise);
# - M = M_vac + M_comm identically; P_A, P_B >= 0;
# - with the pump off during the window the commutator term is literally the
#   unpumped one (state independence, TMM21 Eq. (24)); with it on it moves,
#   and the pumped |M| is dominated by the Hadamard (state) part;
# - a spacelike window has a commutator fraction at the lattice-leakage level.


def _reference_split(field, sites, gap, lam, T, n_prep, n_win, rtol=1e-10):
    from vacuum.detectors.communication import pair_state_with_split
    from vacuum.detectors.kernels import LatticeWightman
    from vacuum.detectors.switching import switching
    from vacuum.detectors.udw import UDWDetector

    N = field.shape[0]
    F_A = np.zeros(N)
    F_A[sites[0]] = 1.0
    F_B = np.zeros(N)
    F_B[sites[1]] = 1.0
    chi = switching("cos2", 0.5 * n_win * T, n_prep * T + 0.5 * n_win * T)
    return pair_state_with_split(LatticeWightman(field), UDWDetector(chi, F_A),
                                 UDWDetector(chi, F_B), lam, gap, rtol=rtol)


def test_split_at_depth_zero_reproduces_the_m25_estimator(field):
    from vacuum.floquet.composite import pumped_harvest_split

    T = 2 * np.pi / OMEGA_MOD
    for n_win in (3, 2):
        sp = pumped_harvest_split(field, None, SITES, (GAP, GAP), LAM, 2, n_win, period=T)
        ref = _reference_split(field, SITES, GAP, LAM, T, 2, n_win)
        assert sp.converged and sp.movement < 1e-7
        assert abs(sp.M - ref.M) <= 1e-7 * abs(ref.M), f"M {sp.M} vs {ref.M}"
        assert abs(sp.M_comm - ref.comm.M_comm) <= 1e-10, f"M_comm {sp.M_comm} vs {ref.comm.M_comm}"
        assert abs(sp.P_A - ref.P_A) <= 1e-7 * ref.P_A and abs(sp.P_B - ref.P_B) <= 1e-7 * ref.P_B
        assert abs(sp.C - ref.C) <= 1e-7 * abs(ref.C), f"C {sp.C} vs {ref.C}"
        assert abs(sp.comm_fraction - ref.comm.comm_fraction) < 1e-6
        assert sp.meta["additivity_defect"] <= 1e-15 * max(1.0, abs(sp.M))
        assert sp.P_A >= 0.0 and sp.P_B >= 0.0
        assert sp.comm_drive_shift == 0.0  # no drive: the static commutator term is the term


def test_pumped_split_moves_the_state_part_and_keeps_the_commutator_honest(field, pumped):
    from vacuum.floquet.composite import pumped_harvest_split

    T = 2 * np.pi / OMEGA_MOD
    drive = modulated_K(field, 0.1, OMEGA_MOD, "sin")
    base = pumped_harvest_split(field, None, SITES, (GAP, GAP), LAM, 2, 3, period=T)
    on = pumped_harvest_split(field, drive, SITES, (GAP, GAP), LAM, 2, 3)
    off = pumped_harvest_split(field, drive, SITES, (GAP, GAP), LAM, 2, 3,
                               pump_during_coupling=False)
    for sp in (base, on, off):
        assert sp.converged and sp.meta["additivity_defect"] <= 1e-15 * max(1.0, abs(sp.M))
        assert sp.P_A >= 0.0 and sp.P_B >= 0.0 and 0.0 <= sp.comm_fraction
    # pump off during the window: the commutator term IS the baseline's (state independence,
    # TMM21 Eq. (24)).  Gated values agree to the gate tolerance; on one fixed grid they are
    # the same numbers.
    assert abs(off.M_comm - base.M_comm) <= 1e-6 * abs(base.M), (
        f"M_comm moved with the field state: {off.M_comm} vs {base.M_comm}"
    )
    assert off.comm_drive_shift == 0.0
    fixed = dict(n_grid_per_period=128, max_halvings=0, require_convergence=False)
    base_f = pumped_harvest_split(field, None, SITES, (GAP, GAP), LAM, 2, 3, period=T, **fixed)
    off_f = pumped_harvest_split(field, drive, SITES, (GAP, GAP), LAM, 2, 3,
                                 pump_during_coupling=False, **fixed)
    on_f = pumped_harvest_split(field, drive, SITES, (GAP, GAP), LAM, 2, 3, **fixed)
    assert abs(off_f.M_comm - base_f.M_comm) <= 1e-15 * max(1.0, abs(base_f.M)), (
        f"fixed grid: M_comm {off_f.M_comm} (pump off) vs {base_f.M_comm} (vacuum)"
    )
    assert abs(on_f.M_comm_static - base_f.M_comm) <= 1e-15 * max(1.0, abs(base_f.M))
    # pump on: the drive changes the retarded propagator, hence the commutator term
    assert on.comm_drive_shift > 0.0 and on_f.comm_drive_shift > 1e-3 * abs(on_f.M_comm)
    # the pumped pair term is dominated by pre-existing (Hadamard) correlations
    assert abs(on.M) > 5.0 * abs(base.M)
    assert on.comm_fraction < 0.1 * base.comm_fraction, (
        f"communication fraction {on.comm_fraction:.4f} (pumped) vs {base.comm_fraction:.4f} (vacuum)"
    )
    assert on.vac_fraction > 0.9
    # regime check: the second-order negativity is reported next to the exact one, not instead of it
    assert on.E_N_perturbative > 0.0 and pumped.E_N > 0.0
    assert on.field_n_total > 0.0


def test_spacelike_window_has_lattice_leakage_commutator_fraction(field):
    from vacuum.floquet.composite import pumped_harvest_split

    wmod = 2.0 * GAP  # T = 2.618, n_win = 1 -> t_win = 2.62 < separation 5
    drive = modulated_K(field, 0.1, wmod, "sin")
    sp = pumped_harvest_split(field, drive, SITES, (GAP, GAP), LAM, 2, 1)
    assert sp.spacelike and sp.converged
    assert abs(sp.M) > 0.0
    assert sp.comm_fraction < 1e-3, f"spacelike window: |M_comm|/|M| = {sp.comm_fraction:.3e}"
