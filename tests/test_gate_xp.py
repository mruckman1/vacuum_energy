"""GATE A for the derivative coupling: nonperturbative 'xp' vs derivative UDW.

The M2.4 cross-validation (tests/test_gate_crossvalidation.py) rerun for
``coupling='xp'``: the same lattice setups, computed two ways —

* nonperturbatively, ``attach_detectors(..., coupling='xp')`` — oscillator
  detectors whose position couples to the field momentum, H_int =
  lam(t) x_d F^T p_field, evolved exactly (Magnus, dt-halving gate) as a
  Gaussian state through the audited ``run_harvesting`` stack;
* perturbatively, the second-order two-qubit state of PKMM with the
  pulled-back kernel replaced by the pi-pi correlator d_t d_t' W
  (``udw_pair_state(..., coupling='xp')``; the d_t phi model of
  Teixido-Bonfill & Martin-Martinez, PRD 110, 105016 (2024), Eq. (2) —
  see ``vacuum.detectors.udw_derivative``).

The two-level <-> oscillator map is (MAP-1) of the amplitude gate
unchanged, lam^UDW = lam^osc / sqrt(2 Omega): step 1 of that derivation
concerns the *detector* monopole x_nu = mu_nu / sqrt(2 Omega_nu) and never
asks which field operator it multiplies.  Steps 2-6 go through verbatim
with W -> d_t d_t' W (every matrix element is linear in the kernel).

Requirement (spec M2.4, applied to 'xp'): E_N agreement with relative
deviation bounded by c*lambda^2 — exponent >= 2, R^2 > 0.999 — plus
canaries that make the gate fail: dropping the sqrt(2 Omega) map, and the
xp-specific slip of feeding the perturbative side the *amplitude* kernel
(a detector coupled to phi instead of d_t phi), which the gate must reject
loudly.

Measured on this rig (lambda = 0.1 .. 0.4): deviation exponent 2.005 (A),
2.019 (B), 2.022 (C) at R^2 = 0.999999 / 0.999987 / 0.999984, with
dev/lambda^2 flat to 0.7% / 2.8% / 3.1% at c = 1.60 / 0.17 / 37.1.  Setup
C's large prefactor is the near-cancellation of its derivative-coupled
negativity (N/|M| = 0.14, so a 5% error on |M| is a 35% error on E_N); the
matrix elements themselves deviate by <= 5.6 lambda^2 there.  Conventions:
docs/API.md.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from vacuum.core import harmonic_chain_K
from vacuum.detectors import (
    UDWDetector,
    run_harvesting,
    udw_pair_state,
    wightman_lattice,
)

# Shared helpers and setups of the amplitude gate (tests/ is on sys.path
# under pytest's default import mode; the helpers are pure functions).
from test_gate_crossvalidation import SETUPS, GateSetup, oscillator_moments, power_fit

#: The lambda sweep.  Starts higher than the amplitude gate's 0.07 because
#: setup C's derivative-coupled negativity is a near-cancellation
#: (N/|M| ~ 0.14: E_N = 1.1e-6 at lambda = 0.1) and the dt gate below has a
#: roundoff floor; the top end keeps every perturbative matrix element
#: < 1.3e-4 (asserted through meta['perturbative_ok']).
LAMBDAS: Tuple[float, ...] = (0.1, 0.14, 0.2, 0.28, 0.4)
#: dt-halving gate tolerance as a fraction of the perturbative E_N (a
#: relative gate keeps the Magnus bias a lambda-independent relative
#: quantity, see the amplitude gate), floored at CONV_ABS_FLOOR: the
#: stepper's observables carry ~1e-12 of accumulated roundoff over these
#: windows, and a tolerance below that never converges.  At the floor the
#: relative bias is 2e-12 / 1.1e-6 = 2e-6, three orders below the smallest
#: deviation fitted here.
CONV_REL_TOL = 2.0e-6
CONV_ABS_FLOOR = 2.0e-12
GATE_EXPONENT = 2.0
GATE_R2 = 0.999
#: Prefactor bounds.  The derivative kernel's omega^2 weight makes the
#: O(lambda^4) corrections larger than under amplitude coupling (measured
#: max dev/lambda^2 of the matrix elements: 5.6, on |m| vs |M| in setup C,
#: against 1.21 for the amplitude gate), and setup C's negativity is a
#: near-cancellation, N/|M| = 0.14, which amplifies the *relative* E_N
#: deviation by |M|/N ~ 7: measured c_E = 36 there (1.6 in setup A).  The
#: spread bound below (< 1.15 across the sweep) is what pins the law; these
#: bounds pin the prefactor at its measured scale.
MATRIX_ELEMENT_C = 8.0
E_N_C_BOUND = 50.0
LEADING_ORDER_DEV = 0.1
NAMES = tuple(SETUPS)


def _pert_state(setup: GateSetup, lam, kernel=None, lam_map=True, coupling="xp"):
    chi_A, chi_B = setup.chis
    det_A = UDWDetector(chi_A, setup.profiles[0])
    det_B = UDWDetector(chi_B, setup.profiles[1])
    lam_udw = (
        tuple(lam / math.sqrt(2.0 * g) for g in setup.gaps) if lam_map else (lam, lam)
    )
    if kernel is None:
        kernel = wightman_lattice(harmonic_chain_K(setup.N, setup.mass, bc=setup.bc))
    return udw_pair_state(
        kernel, det_A, det_B, lam_udw, setup.gaps, coupling=coupling,
        rtol=1e-11, max_doublings=16,
    )


def _run_sweep(setup: GateSetup):
    K = harmonic_chain_K(setup.N, setup.mass, bc=setup.bc)
    kernel = wightman_lattice(K)
    chi_A, chi_B = setup.chis
    profiles = [np.asarray(F, dtype=float) for F in setup.profiles]
    ts = np.array([
        min(chi_A.support[0], chi_B.support[0]),
        max(chi_A.support[1], chi_B.support[1]),
    ])
    pert, harv, mom = [], [], []
    for lam in LAMBDAS:
        st = _pert_state(setup, lam, kernel=kernel)

        def lambda_of_t(t, _l=lam):
            return np.array([_l * chi_A(t), _l * chi_B(t)])

        def dlambda_of_t(t, _l=lam):
            return np.array([_l * chi_A.derivative(t), _l * chi_B.derivative(t)])

        hr = run_harvesting(
            K, profiles, setup.gaps, lambda_of_t, ts,
            dlambda_of_t=dlambda_of_t,
            conv_tol=max(CONV_REL_TOL * st.log_negativity, CONV_ABS_FLOOR),
            max_halvings=18,
            coupling="xp",
        )
        pert.append(st)
        harv.append(hr)
        mom.append(oscillator_moments(hr.V_AB, setup.gaps))
    return {
        "setup": setup,
        "lams": np.array(LAMBDAS, dtype=float),
        "pert": tuple(pert),
        "harv": tuple(harv),
        "mom": tuple(mom),
    }


def _dev(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.abs(a - b) / np.abs(b)


@pytest.fixture(scope="module")
def sweeps() -> Dict[str, Dict[str, Any]]:
    return {name: _run_sweep(setup) for name, setup in SETUPS.items()}


@pytest.mark.parametrize("name", NAMES)
def test_gate_a_xp_deviation_exponent(sweeps, name):
    sw = sweeps[name]
    E_np = np.array([h.E_N for h in sw["harv"]])
    E_pt = np.array([s.log_negativity for s in sw["pert"]])
    dev = _dev(E_np, E_pt)
    assert np.all(dev > 0.0), f"[{name}] zero deviation — engines are not independent"
    exponent, r2, c = power_fit(sw["lams"], dev)
    assert exponent >= GATE_EXPONENT, (
        f"[{name}: {sw['setup'].note}] Gate A (xp) FAILED: deviation exponent "
        f"{exponent:.6f} < {GATE_EXPONENT} (R^2 = {r2:.8f}, c = {c:.4f}); deviations "
        f"{np.array2string(dev, precision=4)} at lambda {np.array2string(sw['lams'], precision=3)}"
    )
    assert r2 > GATE_R2, (
        f"[{name}] Gate A (xp) FAILED: R^2 {r2:.8f} <= {GATE_R2} (exponent {exponent:.6f}); "
        f"deviations {np.array2string(dev, precision=4)}"
    )


@pytest.mark.parametrize("name", NAMES)
def test_gate_a_xp_deviation_bounded_by_c_lambda_squared(sweeps, name):
    sw = sweeps[name]
    E_np = np.array([h.E_N for h in sw["harv"]])
    E_pt = np.array([s.log_negativity for s in sw["pert"]])
    ratio = _dev(E_np, E_pt) / sw["lams"] ** 2
    spread = float(np.max(ratio) / np.min(ratio))
    assert spread < 1.15, (
        f"[{name}] xp dev/lambda^2 spans {np.min(ratio):.4f}-{np.max(ratio):.4f} "
        f"(spread {spread:.4f}): the c*lambda^2 bound is drifting"
    )
    assert float(np.max(ratio)) < E_N_C_BOUND, (
        f"[{name}] c = {np.max(ratio):.4f} exceeds {E_N_C_BOUND} (measured 1.6 (A), 36 (C))"
    )


@pytest.mark.parametrize("name", NAMES)
def test_xp_matrix_elements_agree_at_leading_order(sweeps, name):
    sw = sweeps[name]
    lams = sw["lams"]
    cs_scale = np.array([math.sqrt(st.P_A * st.P_B) for st in sw["pert"]])
    checks = {
        "n_A vs P_A": ([m["n_A"] for m in sw["mom"]], [s.P_A for s in sw["pert"]], None),
        "n_B vs P_B": ([m["n_B"] for m in sw["mom"]], [s.P_B for s in sw["pert"]], None),
        "|m| vs |M|": ([abs(m["m"]) for m in sw["mom"]], [abs(s.M) for s in sw["pert"]], None),
        "|c| vs |C| (per sqrt(P_A P_B))": (
            [abs(m["c"]) for m in sw["mom"]], [abs(s.C) for s in sw["pert"]], cs_scale,
        ),
    }
    for label, (a, b, scale) in checks.items():
        diff = np.abs(np.asarray(a, float) - np.asarray(b, float))
        dev = diff / (np.abs(np.asarray(b, float)) if scale is None else scale)
        ratio = dev / lams**2
        assert float(np.max(ratio)) < MATRIX_ELEMENT_C, (
            f"[{name}] {label}: max(dev/lambda^2) = {np.max(ratio):.4f} exceeds "
            f"{MATRIX_ELEMENT_C}; deviations {np.array2string(dev, precision=4)}"
        )
        assert dev[0] < LEADING_ORDER_DEV, (
            f"[{name}] {label}: deviation {dev[0]:.4e} at lambda = {lams[0]} "
            f"(measured max 0.054, setup C |m| vs |M|)"
        )


@pytest.mark.parametrize("name", NAMES)
def test_every_xp_gate_point_is_audited(sweeps, name):
    sw = sweeps[name]
    for lam, hr, st in zip(sw["lams"], sw["harv"], sw["pert"]):
        assert hr.passivity is not None and hr.passivity.passed, f"[{name}] lambda={lam}"
        assert hr.ledger_result.passed, f"[{name}] lambda={lam}: {hr.ledger_result.details}"
        assert hr.converged and hr.dt_converged > 0.0, f"[{name}] lambda={lam}"
        assert hr.neg.margin_min < -1e-8, (
            f"[{name}] lambda={lam}: PT margin {hr.neg.margin_min:.3e} at the floor"
        )
        assert st.meta["perturbative_ok"] and st.meta["coupling"] == "xp", (
            f"[{name}] lambda={lam}: scale {st.meta['perturbative_scale']:.3e}"
        )


# --------------------------------------------------------------------------
# Canaries: the xp gate must be able to fail
# --------------------------------------------------------------------------


def test_canary_xp_missing_oscillator_to_qubit_normalization(sweeps):
    sw = sweeps["A"]
    setup = sw["setup"]
    kernel = wightman_lattice(harmonic_chain_K(setup.N, setup.mass, bc=setup.bc))
    bad = np.array([
        _pert_state(setup, lam, kernel=kernel, lam_map=False).log_negativity
        for lam in sw["lams"]
    ])
    E_np = np.array([h.E_N for h in sw["harv"]])
    dev = _dev(E_np, bad)
    exponent, r2, _ = power_fit(sw["lams"], dev)
    assert np.all(dev > 0.5), f"canary: deviations {dev} below 0.5"
    assert exponent < 0.1, (
        f"canary: dropping sqrt(2 Omega) still yields exponent {exponent:.4f} (R^2 {r2:.6f})"
    )


def test_canary_amplitude_kernel_fails_the_xp_gate(sweeps):
    """The xp-specific slip: a phi-coupled perturbative state vs the pi-coupled exact run.

    Feeding the gate the amplitude ('xx') second-order state — i.e. forgetting
    the d_t d_t' on the kernel — must not pass.  In setup A the two kernels
    happen to give E_N within 23-41% of each other (the resonant modes sit
    where omega^2 ~ 1-4, so |M| differs by only 1.3x — though |C| differs by
    3.5x), so what the gate would trip on is not the size of the deviation
    but its lambda-dependence: it is O(1) and does NOT fall as lambda^2
    (measured exponent 0.4 against the required >= 2).  Both facts are
    pinned.
    """
    sw = sweeps["A"]
    setup = sw["setup"]
    kernel = wightman_lattice(harmonic_chain_K(setup.N, setup.mass, bc=setup.bc))
    wrong = [_pert_state(setup, lam, kernel=kernel, coupling="xx") for lam in sw["lams"]]
    E_np = np.array([h.E_N for h in sw["harv"]])
    dev = _dev(E_np, [s.log_negativity for s in wrong])
    exponent, r2, _ = power_fit(sw["lams"], dev)
    assert np.all(dev > 0.1), (
        f"canary: the amplitude kernel deviates only {dev} from the xp exact run"
    )
    assert exponent < 1.0, (
        f"canary: amplitude-kernel deviation exponent {exponent:.4f} (R^2 {r2:.6f}) "
        f"with deviations {dev} — the xp gate cannot distinguish phi from d_t phi "
        "coupling (a lambda^2 law would give >= 2)"
    )
    c_ratio = abs(wrong[0].C) / abs(sw["pert"][0].C)
    assert not (0.5 < c_ratio < 2.0), (
        f"|C_xx|/|C_xp| = {c_ratio:.4f} too close to 1 (measured 3.5)"
    )
