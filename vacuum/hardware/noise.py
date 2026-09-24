"""Sourced device noise, labelled sweeps around it, and the survival analysis.

The ONE sourced calibration point is the ibm_cairo column of Table III of
K. Ikeda, Phys. Rev. Applied 20, 024051 (2023) [arXiv:2301.02666v5], as
shipped in ``experiments/ibm_qet/ikeda_2023_fig2.json`` and turned into an
Aer noise model by :func:`vacuum.qet.ikeda_qiskit.noise_model_from_params`
(T1/T2 thermal relaxation on every native gate, depolarizing errors sized so
the average gate infidelity equals the printed Pauli-X / CNOT error,
symmetric readout flips).  :func:`calibration_point` reads every number
back out of that function's cited description, so this module never
re-types a device number.

Two things are added and both are labelled:

* an **N-qubit** model with the same conventions
  (:func:`noise_model_from_calibration`): the two Table III rows are TILED
  over the register (qubit q carries the row of device qubit 13 or 14 for
  q even / odd) and the one printed CNOT error sits on every edge of the
  coupling map.  Table III calibrates a single pair; applying it to five or
  ten qubits is a modelling choice, stated here and in every description.
  At two qubits on the [0, 1] edge the model is the Ikeda one (tested).
* **sweeps** (:func:`scale_calibration`): a multiplier nu on the gate
  errors (Pauli-X and CNOT together; relaxation stays at the sourced
  T1/T2, so below the relaxation-limited infidelity the depolarizing part
  clips to zero and the effective error floors), on the readout flips
  (clipped at 1/2), or a divisor on T1 and T2 (both, so T2 <= 2 T1 is
  preserved), or on all three together (axis 'all': a uniformly
  better or worse device).  nu = 1 is the sourced point; every scaled
  calibration carries ``sweep`` and ``synthetic = True``.  The grids the paper
  notebook walks are declared in
  ``experiments/ibm_qet/ikeda_2023_fig2_variant-noise-sweep.json``.

Survival analysis
-----------------
A protocol's signal survives a noise level when it is positive at three
standard errors for a stated shot budget: for QET, E_B_ctrl > 3 sigma with
sigma from the exact noisy readout distributions at ``shots`` per circuit;
for harvesting, -lambda_min(rho^{T_B}) > 3 sigma with sigma the parametric
bootstrap of the nine tomography settings at ``shots`` each (equivalently
E_N > 0 at 3 sigma, since E_N = ln(1 - 2 lambda_min)).  Signals are
shot-free (Aer density matrix), so :func:`survival_threshold` can bisect
the multiplier nu in log space between an alive and a dead bracket and
return the threshold multiplier and its physical value (CNOT error,
readout error, T1).  The family of sampled runs in the tests confirms the
shot-free sigma against actual Aer sampling.

Real hardware: :func:`device_run` is the one-line on-ramp — it hands the
very circuits the survival map was computed with to
:func:`vacuum.qet.ikeda_qiskit.run_on_backend`; nothing in this repository
executes it against a device.
"""

from __future__ import annotations

import math

from vacuum.qet.ikeda import IKEDA_FIG2_FILE, ikeda_params
from vacuum.qet.ikeda_qiskit import (
    IBM_BASIS_GATES,
    _depolarizing_to_match,
    _qiskit,
    noise_model_from_params,
    run_on_backend,
)

__all__ = [
    "SWEEP_AXES",
    "SWEEP_FILE",
    "calibration_point",
    "scale_calibration",
    "noise_model_from_calibration",
    "sweep_grid",
    "survival_threshold",
    "survival_sweep",
    "shots_for_n_sigma",
    "device_run",
    "run_on_backend",
]

SWEEP_AXES = ("gate", "readout", "t1", "all")
#: declared (synthetic) sweep grids around the sourced point
SWEEP_FILE = "ibm_qet/ikeda_2023_fig2_variant-noise-sweep"


def calibration_point(params=None, readout="table_iii", single_qubit_gate_time_ns=None):
    """The sourced calibration as a plain dict, read from the description
    of :func:`vacuum.qet.ikeda_qiskit.noise_model_from_params`.

    Keys: ``t1_ns``, ``t2_ns``, ``pauli_x_error``, ``readout_error`` (each
    a 2-list, device qubits 13/14), ``cnot_error``, ``cnot_gate_time_ns``,
    ``single_qubit_gate_time_ns``, ``readout_model``, ``locations`` (the
    per-number citations), ``source``, ``sweep`` (None), ``synthetic``
    (False).
    """
    p = ikeda_params(params if params is not None else IKEDA_FIG2_FILE)
    _, desc = noise_model_from_params(p, readout=readout,
                                      single_qubit_gate_time_ns=single_qubit_gate_time_ns)
    return {
        "t1_ns": [float(x) for x in desc["t1_ns"]],
        "t2_ns": [float(x) for x in desc["t2_ns"]],
        "pauli_x_error": [float(x) for x in desc["pauli_x_error"]],
        "readout_error": [float(x) for x in desc["readout_error"]],
        "cnot_error": float(desc["cnot_error"]),
        "cnot_gate_time_ns": float(desc["cnot_gate_time_ns"]),
        "single_qubit_gate_time_ns": float(desc["single_qubit_gate_time_ns"]),
        "readout_model": readout,
        "locations": {k: v for k, v in desc.items() if k.endswith(("_location", "_note"))},
        "backend": desc["backend"],
        "source": p.get("source"),
        "sweep": None,
        "synthetic": False,
    }


def scale_calibration(calib, gate=1.0, readout=1.0, t1=1.0):
    """A labelled sweep point: gate errors x ``gate``, readout flips x
    ``readout`` (clipped at 0.5), T1 and T2 / ``t1``.  Returns a new dict
    with ``sweep`` = the multipliers and ``synthetic`` = True unless all
    three are 1."""
    gate, readout, t1 = float(gate), float(readout), float(t1)
    if gate < 0 or readout < 0 or t1 <= 0:
        raise ValueError("gate/readout multipliers must be >= 0 and the T1 divisor > 0")
    out = dict(calib)
    out["pauli_x_error"] = [gate * x for x in calib["pauli_x_error"]]
    out["cnot_error"] = gate * calib["cnot_error"]
    out["readout_error"] = [min(0.5, readout * x) for x in calib["readout_error"]]
    out["t1_ns"] = [x / t1 for x in calib["t1_ns"]]
    out["t2_ns"] = [x / t1 for x in calib["t2_ns"]]
    out["sweep"] = {"gate": gate, "readout": readout, "t1": t1}
    out["synthetic"] = not (gate == 1.0 and readout == 1.0 and t1 == 1.0)
    return out


def noise_model_from_calibration(calib, n_qubits, coupling_map):
    """Aer ``NoiseModel`` on ``n_qubits`` from a calibration dict (module
    docstring conventions; qubit q takes row q mod 2).  Returns
    (noise_model, description); ``description['readout_error']`` is the
    per-qubit flip list the exact paths apply analytically."""
    Q = _qiskit()
    n = int(n_qubits)
    t1 = [calib["t1_ns"][q % 2] for q in range(n)]
    t2 = [calib["t2_ns"][q % 2] for q in range(n)]
    eps1 = [calib["pauli_x_error"][q % 2] for q in range(n)]
    pro = [calib["readout_error"][q % 2] for q in range(n)]
    t_1q = float(calib["single_qubit_gate_time_ns"])
    t_cx = float(calib["cnot_gate_time_ns"])
    eps_cx = float(calib["cnot_error"])
    nm = Q.NoiseModel(basis_gates=list(IBM_BASIS_GATES))
    desc = {
        "n_qubits": n,
        "tiling": "qubit q carries Table III row (q mod 2): device qubit 13 for even q, 14 for odd q; "
                  "the single printed CNOT error on every coupling-map edge (modelling choice)",
        "t1_ns": t1, "t2_ns": t2, "pauli_x_error": eps1, "cnot_error": eps_cx,
        "readout_error": pro, "single_qubit_gate_time_ns": t_1q, "cnot_gate_time_ns": t_cx,
        "depolarizing_1q": [], "depolarizing_2q": {}, "relaxation_fidelity_1q": [],
        "relaxation_fidelity_2q": {}, "sweep": calib.get("sweep"), "synthetic": calib.get("synthetic", False),
        "source": calib.get("source"), "locations": calib.get("locations"),
        "coupling_map": [list(e) for e in coupling_map],
    }
    for q in range(n):
        relax = Q.thermal_relaxation_error(t1[q], t2[q], t_1q)
        lam, F_relax = _depolarizing_to_match(Q, relax, eps1[q], 2)
        err = Q.depolarizing_error(lam, 1).compose(relax) if lam > 0.0 else relax
        nm.add_quantum_error(err, ["x", "sx"], [q])
        desc["depolarizing_1q"].append(lam)
        desc["relaxation_fidelity_1q"].append(F_relax)
        if pro[q] > 0.0:
            nm.add_readout_error(Q.ReadoutError([[1.0 - pro[q], pro[q]], [pro[q], 1.0 - pro[q]]]), [q])
    for a, b in coupling_map:
        a, b = int(a), int(b)
        relax2 = Q.thermal_relaxation_error(t1[a], t2[a], t_cx).expand(
            Q.thermal_relaxation_error(t1[b], t2[b], t_cx))
        lam2, F2 = _depolarizing_to_match(Q, relax2, eps_cx, 4)
        err2 = Q.depolarizing_error(lam2, 2).compose(relax2) if lam2 > 0.0 else relax2
        nm.add_quantum_error(err2, ["cx"], [a, b])
        desc["depolarizing_2q"][f"{a}->{b}"] = lam2
        desc["relaxation_fidelity_2q"][f"{a}->{b}"] = F2
    return nm, desc


def sweep_grid(axis, root=None):
    """The declared multiplier grid of one axis from :data:`SWEEP_FILE`."""
    from vacuum import experiments_io as xio

    if axis not in SWEEP_AXES:
        raise ValueError(f"axis must be one of {SWEEP_AXES}")
    exp = xio.load_experiment(SWEEP_FILE, root=root)
    name = {"gate": "gate_error_multiplier_grid", "readout": "readout_error_multiplier_grid",
            "t1": "t1_divisor_grid", "all": "gate_error_multiplier_grid"}[axis]
    return [float(x) for x in exp.quantities[name].value], exp


def _kw(axis, nu):
    """Multiplier -> scale_calibration keywords; 'all' moves the three
    axes together (gate and readout x nu, T1/T2 / nu): the "uniformly
    better/worse device" direction."""
    if axis == "all":
        return {"gate": float(nu), "readout": float(nu), "t1": float(nu)}
    return {axis: float(nu)}


def survival_threshold(signal_fn, calib, axis, n_sigma=3.0, nu_min=1.0 / 64.0, nu_max=64.0,
                       n_iter=8):
    """Bisect the noise multiplier at which ``signal_fn`` drops to n_sigma.

    ``signal_fn(calib_scaled) -> dict`` with keys ``signal`` and ``sigma``
    (shot-free signal and its shot standard error).  Starting at nu = 1
    (the sourced point), the bracket is grown by factors of two until it
    encloses the crossing (bounded by ``nu_min``/``nu_max``), then bisected
    ``n_iter`` times in log nu.

    Returns dict: ``axis``, ``nu_star`` (None if the signal is alive at
    nu_max or dead at nu_min), ``alive_at_source``, ``bracket``,
    ``history`` (every (nu, signal, sigma, z) evaluated), ``physical``
    (the threshold in device units: CNOT error / readout flip of qubit 13
    / T1 of qubit 13 in us), ``source_value``.
    """
    if axis not in SWEEP_AXES:
        raise ValueError(f"axis must be one of {SWEEP_AXES}")
    history = []

    def z_at(nu):
        out = signal_fn(scale_calibration(calib, **_kw(axis, nu)))
        z = out["signal"] / out["sigma"] if out["sigma"] > 0 else math.inf
        history.append({"nu": float(nu), "signal": out["signal"], "sigma": out["sigma"], "z": z})
        return z

    z1 = z_at(1.0)
    alive = z1 > n_sigma
    result = {"axis": axis, "alive_at_source": alive, "n_sigma": float(n_sigma),
              "source_value": _physical(calib, axis, 1.0)}
    lo, hi = None, None
    if alive:
        nu = 1.0
        while nu < nu_max:
            nu_next = min(2.0 * nu, nu_max)
            if z_at(nu_next) > n_sigma:
                nu = nu_next
                if nu >= nu_max:
                    break
            else:
                lo, hi = nu, nu_next
                break
    else:
        nu = 1.0
        while nu > nu_min:
            nu_next = max(0.5 * nu, nu_min)
            if z_at(nu_next) > n_sigma:
                lo, hi = nu_next, nu
                break
            nu = nu_next
    if lo is None:
        result.update({"nu_star": None, "bracket": None, "history": history,
                       "physical": None,
                       "note": ("signal still alive at nu_max" if alive else "signal dead down to nu_min")})
        return result
    for _ in range(int(n_iter)):
        mid = math.sqrt(lo * hi)
        if z_at(mid) > n_sigma:
            lo = mid
        else:
            hi = mid
    nu_star = math.sqrt(lo * hi)
    result.update({"nu_star": nu_star, "bracket": (lo, hi), "history": history,
                   "physical": _physical(calib, axis, nu_star)})
    return result


def _physical(calib, axis, nu):
    if axis == "all":
        return {**_physical(calib, "gate", nu), **_physical(calib, "readout", nu),
                **_physical(calib, "t1", nu)}
    if axis == "gate":
        return {"cnot_error": nu * calib["cnot_error"],
                "pauli_x_error": [nu * x for x in calib["pauli_x_error"]]}
    if axis == "readout":
        return {"readout_error": [min(0.5, nu * x) for x in calib["readout_error"]]}
    return {"t1_us": [x / nu / 1e3 for x in calib["t1_ns"]], "t2_us": [x / nu / 1e3 for x in calib["t2_ns"]]}


def survival_sweep(signal_fn, calib, axis, multipliers):
    """Evaluate the signal on a declared grid of multipliers; rows carry nu,
    signal, sigma, z, alive (z > 3), the physical values and the sweep label."""
    rows = []
    for nu in multipliers:
        c = scale_calibration(calib, **_kw(axis, nu))
        out = signal_fn(c)
        z = out["signal"] / out["sigma"] if out["sigma"] > 0 else math.inf
        rows.append({"axis": axis, "nu": float(nu), "signal": out["signal"], "sigma": out["sigma"],
                     "z": z, "alive": z > 3.0, "physical": _physical(calib, axis, nu),
                     "synthetic": c["synthetic"], **{k: v for k, v in out.items() if k not in ("signal", "sigma")}})
    return rows


def shots_for_n_sigma(signal, sigma_single_shot_equivalent, shots_used, n_sigma=3.0):
    """Shots per circuit needed for ``signal`` to reach n_sigma, from the
    standard error ``sigma`` measured at ``shots_used``: sigma scales as
    1/sqrt(shots)."""
    if signal <= 0:
        return math.inf
    return int(math.ceil(shots_used * (n_sigma * sigma_single_shot_equivalent / signal) ** 2))


def device_run(circuits, backend, shots, sampler=None, seed=None, optimization_level=1):
    """The real-device on-ramp: counts for every circuit through
    :func:`vacuum.qet.ikeda_qiskit.run_on_backend`.  For an IBM Runtime
    device: ``device_run(circs, service.backend("ibm_..."), shots,
    sampler=SamplerV2(mode=backend))``.  Feed the result to
    :func:`vacuum.hardware.protocols.qet_from_counts` /
    :func:`vacuum.hardware.protocols.harvest_from_counts`.  Not executed
    anywhere in this repository against hardware."""
    return [run_on_backend(qc, backend, shots, seed=(None if seed is None else seed + i),
                           sampler=sampler, optimization_level=optimization_level)
            for i, qc in enumerate(circuits)]
