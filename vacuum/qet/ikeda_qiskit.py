"""Qiskit/Aer realisation of the Ikeda protocol — the ``.[ibm]`` extra.

Guarded module: qiskit and qiskit-aer are imported lazily inside
:func:`_qiskit`, so importing this module (and ``vacuum.qet``) never requires
them. The dense path in :mod:`vacuum.qet.ikeda` is the core-only fallback and
the arbiter of every number here — noiseless Aer must reproduce it to 1e-10
(tests/test_ikeda.py). Nothing in this module contacts hardware.

Paper: K. Ikeda, Phys. Rev. Applied 20, 024051 (2023) [arXiv:2301.02666v5].

What this module adds to the dense path
---------------------------------------
* :func:`build_circuit` — the gate list of
  :func:`vacuum.qet.ikeda.ikeda_circuit_spec` as a ``QuantumCircuit``
  (feed-forward conditions become ``if_test`` blocks; the deferred form is
  the circuit the paper ran, Fig. 1(C) right).
* :func:`exact_statevector` / :func:`exact_density_matrix` — noiseless
  references from ``qiskit.quantum_info.Statevector`` and Aer's
  density-matrix method, returned in the dense (hotta.py) basis ordering.
* :func:`noise_model_from_params` — an Aer ``NoiseModel`` built ONLY from
  the device numbers the paper prints for the Fig. 2 run (Table III,
  ibm_cairo column; Sec. II.D), every one cited in the returned description:
  T1/T2 thermal relaxation on every native gate, depolarizing errors sized
  so the average gate infidelity equals the printed Pauli-X / CNOT error,
  and symmetric readout assignment errors — either the per-qubit Table III
  values or one derived from the printed calibration-matrix measurement
  fidelity.
* :func:`noisy_probabilities` — shot-free noisy readout distributions
  (density matrix through the circuit transpiled to ibm_cairo's native
  gates, then the assignment matrix applied analytically), and the paper's
  calibration-matrix measurement-error mitigation of Sec. II.D
  (:func:`calibration_matrix`, :func:`mitigate_probabilities`).
* :func:`run_on_backend` — ``counts = run_on_backend(circuit, backend, shots)``
  for any backend with ``.run`` (Aer) or through a ``SamplerV2`` (IBM
  Runtime), so a real-device run is a one-line change of ``backend``.
* :func:`ikeda_ibm_aer` — the ledger: ideal, noisy-exact, noisy-sampled and
  mitigated E_A, E_B, <E_0>, <H_1>, <V>, <E_1>, with the paper's Fig. 2(B)
  / Table I numbers alongside when the parameter file carries them.

Qubit ordering: qiskit is little-endian (statevector index = 2 c1 + c0);
the dense basis of hotta.py / ikeda.py has qubit 0 as the FIRST factor
(index = 2 b0 + b1). ``_DENSE_PERM`` maps between them. Counts keys are
qiskit's "c1c0" strings — the convention of Fig. 2(B) and of
:func:`vacuum.qet.ikeda.expectation_from_counts`.

Noise-model conventions (documented so a reader can dispute them):

* Native basis {rz, sx, x, cx} (ibm_cairo, Falcon r5.11); ``rz`` is a
  virtual frame change and carries no error.
* Table III's "Pauli X error" is attached to both ``x`` and ``sx`` on the
  respective qubit; the "CNOT error" to ``cx`` in both directions.
* Depolarizing parameter: with relaxation channel R of average gate
  fidelity F_R and printed error eps, lambda = (F_R - (1 - eps))/(F_R - 1/d)
  (clipped at 0) so that F_avg(D_lambda o R) = 1 - eps — the same
  composition qiskit's own ``NoiseModel.from_backend`` uses.
* Single-qubit gate time: Sec. II.D prints only the range 20-40 ns; the
  default is its midpoint 30 ns (exposed as an argument; the tests show the
  two ends move every reported energy by < 1e-3).
* Readout ``'table_iii'``: per-qubit symmetric assignment error from the
  Table III rows (8.500e-3, 8.000e-3). Readout ``'fidelity'``: the printed
  "Measurement fidelity" 0.961935 is the calibration-matrix average
  correct-assignment probability; for independent symmetric per-qubit
  errors every diagonal entry is (1 - p)^2, so p = 1 - sqrt(0.961935).
  Both are offered because the calibration-matrix number was measured in
  the same job as the data (Sec. II.D) and is the larger of the two.
"""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import numpy as np

from vacuum.qet.ikeda import (
    BITSTRINGS,
    OBSERVABLES,
    counts_to_probabilities,
    expectation_from_counts,
    ikeda_circuit_spec,
    ikeda_params,
    terminal_measure_index,
)

__all__ = [
    "IBM_BASIS_GATES",
    "COUPLING_MAP",
    "READOUT_MODELS",
    "build_circuit",
    "transpile_native",
    "to_dense_vector",
    "to_dense_matrix",
    "probabilities_to_dict",
    "exact_statevector",
    "exact_density_matrix",
    "readout_assignment_matrix",
    "apply_readout",
    "noise_model_from_params",
    "noisy_probabilities",
    "calibration_matrix",
    "mitigate_probabilities",
    "run_on_backend",
    "ikeda_ibm_aer",
]

#: ibm_cairo native gate set (Falcon r5.11); ``rz`` is virtual (error-free).
IBM_BASIS_GATES = ("rz", "sx", "x", "cx")
#: The one edge used: device qubits [13, 14] -> circuit qubits [0, 1].
COUPLING_MAP = ((0, 1), (1, 0))
READOUT_MODELS = ("table_iii", "fidelity", "none")
#: dense index i = 2 b0 + b1  <-  qiskit index 2 b1 + b0
_DENSE_PERM = np.array([0, 2, 1, 3])


def _qiskit():
    """Lazy, guarded qiskit + qiskit-aer import (the ``.[ibm]`` extra)."""
    try:
        from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
        from qiskit.quantum_info import Statevector, average_gate_fidelity
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import (
            NoiseModel,
            ReadoutError,
            depolarizing_error,
            thermal_relaxation_error,
        )
        import qiskit_aer.library  # noqa: F401  (adds QuantumCircuit.save_density_matrix)
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise ImportError(
            "vacuum.qet.ikeda_qiskit needs qiskit and qiskit-aer — install the "
            "'.[ibm]' extra (pip install qiskit qiskit-aer); the dense path in "
            "vacuum.qet.ikeda is the core-only fallback"
        ) from exc
    return SimpleNamespace(
        ClassicalRegister=ClassicalRegister,
        QuantumCircuit=QuantumCircuit,
        QuantumRegister=QuantumRegister,
        transpile=transpile,
        Statevector=Statevector,
        average_gate_fidelity=average_gate_fidelity,
        AerSimulator=AerSimulator,
        NoiseModel=NoiseModel,
        ReadoutError=ReadoutError,
        depolarizing_error=depolarizing_error,
        thermal_relaxation_error=thermal_relaxation_error,
    )


# ---------------------------------------------------------------------------
# circuits
# ---------------------------------------------------------------------------


def _apply_gate(qc, qr, gate):
    name = gate["name"]
    qs = [qr[q] for q in gate["qubits"]]
    params = gate.get("params", [])
    if name == "ry":
        qc.ry(params[0], qs[0])
    elif name == "h":
        qc.h(qs[0])
    elif name == "x":
        qc.x(qs[0])
    elif name == "cx":
        qc.cx(qs[0], qs[1])
    elif name == "cry":
        qc.cry(params[0], qs[0], qs[1])
    else:
        raise ValueError(f"unknown gate {name!r}")


def build_circuit(spec, readout=True, name=None):
    """A ``QuantumCircuit`` from a gate list of :func:`ikeda_circuit_spec`.

    Registers are named ``q`` and ``c``. ``readout=False`` drops the
    terminal readout block only (a feed-forward mid-circuit measurement is
    kept, since the conditioned rotation needs it); conditions become
    ``if_test`` blocks on the named classical bit.
    """
    Q = _qiskit()
    qr = Q.QuantumRegister(int(spec["n_qubits"]), "q")
    cr = Q.ClassicalRegister(int(spec["n_clbits"]), "c")
    qc = Q.QuantumCircuit(qr, cr, name=name or f"ikeda_{spec['observable']}_{spec['form']}")
    i_term = terminal_measure_index(spec)
    for i, gate in enumerate(spec["gates"]):
        if gate["name"] == "measure":
            if i >= i_term and not readout:
                continue
            qc.measure([qr[q] for q in gate["qubits"]], [cr[c] for c in gate["clbits"]])
            continue
        cond = gate.get("condition")
        if cond is None:
            _apply_gate(qc, qr, gate)
        else:
            with qc.if_test((cr[int(cond[0])], int(cond[1]))):
                _apply_gate(qc, qr, gate)
    return qc


def transpile_native(circuit, optimization_level=1, seed_transpiler=0):
    """Transpile to ibm_cairo's native basis on the single [0, 1] edge.

    ``initial_layout=[0, 1]`` pins circuit qubit i to physical qubit i so the
    Table III per-qubit numbers land on the right qubit. ``optimization_level``
    1 was qiskit's default when the paper was run (2023).
    """
    Q = _qiskit()
    return Q.transpile(
        circuit,
        basis_gates=list(IBM_BASIS_GATES),
        coupling_map=[list(edge) for edge in COUPLING_MAP],
        initial_layout=[0, 1],
        optimization_level=int(optimization_level),
        seed_transpiler=seed_transpiler,
    )


# ---------------------------------------------------------------------------
# ordering helpers
# ---------------------------------------------------------------------------


def to_dense_vector(vec):
    """qiskit little-endian (4,) -> dense hotta.py ordering (qubit 0 first)."""
    return np.asarray(vec, dtype=complex)[_DENSE_PERM]


def to_dense_matrix(rho):
    """qiskit little-endian (4,4) -> dense hotta.py ordering."""
    rho = np.asarray(rho, dtype=complex)
    return rho[np.ix_(_DENSE_PERM, _DENSE_PERM)]


def probabilities_to_dict(v_dense):
    """Dense-ordered probability vector -> {"c1c0": p} (qiskit keys)."""
    v = np.asarray(v_dense, dtype=float)
    return {f"{b1}{b0}": float(v[2 * b0 + b1]) for b0 in (0, 1) for b1 in (0, 1)}


def _dict_to_dense(probs):
    p, _ = counts_to_probabilities(probs)
    v = np.zeros(4)
    for key, val in p.items():
        b1, b0 = int(key[0]), int(key[1])
        v[2 * b0 + b1] = val
    return v


# ---------------------------------------------------------------------------
# noiseless references
# ---------------------------------------------------------------------------


def exact_statevector(spec):
    """Noiseless pre-readout state from ``qiskit.quantum_info.Statevector``,
    dense ordering. Deferred form only (no mid-circuit measurement)."""
    if spec["form"] != "deferred":
        raise ValueError("exact_statevector needs the deferred form "
                         "(a Statevector cannot hold a mid-circuit measurement)")
    Q = _qiskit()
    qc = build_circuit(spec, readout=False)
    return to_dense_vector(Q.Statevector(qc).data)


def exact_density_matrix(spec, noise_model=None, optimization_level=1, seed_transpiler=0):
    """Pre-readout density matrix from Aer's density-matrix method.

    With a ``noise_model`` the circuit is first transpiled to the native
    basis (:func:`transpile_native`) so gate errors attach to the gates the
    device would run. Deferred form only: Aer samples mid-circuit
    measurements shot by shot, so the feed-forward ensemble is not a single
    saved density matrix.

    Returns (rho_dense, circuit_run).
    """
    if spec["form"] != "deferred":
        raise ValueError("exact_density_matrix needs the deferred form")
    Q = _qiskit()
    sim = Q.AerSimulator(method="density_matrix", noise_model=noise_model)
    qc = build_circuit(spec, readout=False)
    if noise_model is not None:
        qc = transpile_native(qc, optimization_level, seed_transpiler)
    else:
        qc = Q.transpile(qc, sim, optimization_level=0)
    qc.save_density_matrix(label="rho")
    res = sim.run(qc, shots=1).result()
    return to_dense_matrix(res.data(0)["rho"]), qc


# ---------------------------------------------------------------------------
# readout errors
# ---------------------------------------------------------------------------


def readout_assignment_matrix(p0, p1):
    """A[measured, true] for independent symmetric flips p0 (qubit 0), p1
    (qubit 1), dense ordering (qubit 0 the first Kronecker factor)."""
    def m(p):
        p = float(p)
        return np.array([[1.0 - p, p], [p, 1.0 - p]])
    return np.kron(m(p0), m(p1))


def apply_readout(probs, A):
    """Push a {"c1c0": p} distribution through an assignment matrix."""
    return probabilities_to_dict(np.asarray(A) @ _dict_to_dense(probs))


# ---------------------------------------------------------------------------
# the device noise model (Table III, ibm_cairo column)
# ---------------------------------------------------------------------------


def _depolarizing_to_match(Q, relax_error, eps, dim):
    """lambda such that F_avg(D_lambda o relax) = 1 - eps (0 if unreachable)."""
    F_relax = float(Q.average_gate_fidelity(relax_error.to_quantumchannel()))
    target = 1.0 - float(eps)
    if F_relax <= target:
        return 0.0, F_relax
    lam = (F_relax - target) / (F_relax - 1.0 / dim)
    return float(lam), F_relax


def noise_model_from_params(params=None, readout="table_iii", single_qubit_gate_time_ns=None):
    """Aer ``NoiseModel`` from the paper's printed device parameters.

    Parameters
    ----------
    params : see :func:`vacuum.qet.ikeda.ikeda_params`; needs the Table III
        quantities of the shipped Fig. 2 file (``qubit{0,1}_t1``,
        ``qubit{0,1}_t2`` in us, ``qubit{0,1}_pauli_x_error``,
        ``qubit{0,1}_readout_assignment_error``, ``cnot_error``,
        ``gate_time_cnot`` in ns, ``measurement_fidelity``,
        ``single_qubit_gate_time_range`` in ns).
    readout : 'table_iii' | 'fidelity' | 'none'
        See the module docstring.
    single_qubit_gate_time_ns : float, optional
        Default: midpoint of the printed 20-40 ns range (Sec. II.D).

    Returns
    -------
    (noise_model, description)
        ``description`` lists every number used, its units, its location in
        the paper, and the derived depolarizing parameters; key
        ``'readout_error'`` holds the per-qubit flip probabilities (p0, p1)
        that :func:`noisy_probabilities` applies analytically.
    """
    Q = _qiskit()
    p = ikeda_params(params)
    if readout not in READOUT_MODELS:
        raise ValueError(f"readout must be one of {READOUT_MODELS}, got {readout!r}")
    needed = ["qubit0_t1", "qubit0_t2", "qubit1_t1", "qubit1_t2",
              "qubit0_pauli_x_error", "qubit1_pauli_x_error", "cnot_error",
              "gate_time_cnot", "single_qubit_gate_time_range"]
    if readout == "table_iii":
        needed += ["qubit0_readout_assignment_error", "qubit1_readout_assignment_error"]
    if readout == "fidelity":
        needed += ["measurement_fidelity"]
    missing = [n for n in needed if n not in p]
    if missing:
        raise ValueError(f"parameter source lacks Table III quantities {missing}")
    units = p.get("_units", {})
    for name in ("qubit0_t1", "qubit0_t2", "qubit1_t1", "qubit1_t2"):
        if units.get(name, "us") != "us":
            raise ValueError(f"{name}: expected units 'us', got {units.get(name)!r}")
    for name in ("gate_time_cnot", "single_qubit_gate_time_range"):
        if units.get(name, "ns") != "ns":
            raise ValueError(f"{name}: expected units 'ns', got {units.get(name)!r}")

    t1 = [float(p["qubit0_t1"]) * 1e3, float(p["qubit1_t1"]) * 1e3]   # us -> ns
    t2 = [float(p["qubit0_t2"]) * 1e3, float(p["qubit1_t2"]) * 1e3]
    eps1 = [float(p["qubit0_pauli_x_error"]), float(p["qubit1_pauli_x_error"])]
    eps_cx = float(p["cnot_error"])
    t_cx = float(p["gate_time_cnot"])
    rng = [float(x) for x in p["single_qubit_gate_time_range"]]
    t_1q = float(single_qubit_gate_time_ns) if single_qubit_gate_time_ns is not None \
        else 0.5 * (rng[0] + rng[1])
    if readout == "table_iii":
        pro = [float(p["qubit0_readout_assignment_error"]),
               float(p["qubit1_readout_assignment_error"])]
        ro_note = ("per-qubit 'Readout assignment error', Table III ibm_cairo column "
                   "(first qubit 8.500e-3, second qubit 8.000e-3), symmetric flips")
    elif readout == "fidelity":
        pr = 1.0 - math.sqrt(float(p["measurement_fidelity"]))
        pro = [pr, pr]
        ro_note = ("derived: p = 1 - sqrt(F) with F = 'Measurement fidelity' 0.961935, "
                   "Table III ibm_cairo column (calibration-matrix average correct-"
                   "assignment probability, Sec. II.D), assuming independent symmetric "
                   "per-qubit flips so that every diagonal entry is (1-p)^2")
    else:
        pro = [0.0, 0.0]
        ro_note = "no readout error"

    nm = Q.NoiseModel(basis_gates=list(IBM_BASIS_GATES))
    desc = {
        "backend": "ibm_cairo (Table III column; Fig. 2 run), device qubits [13, 14] -> [0, 1]",
        "basis_gates": list(IBM_BASIS_GATES),
        "rz": "virtual frame change, no error",
        "t1_ns": t1, "t2_ns": t2,
        "t1_t2_location": "Table III, ibm_cairo column, t1(us)/t2(us) rows for the first and second qubit",
        "single_qubit_gate_time_ns": t_1q,
        "single_qubit_gate_time_location": "Sec. II.D: 'the duration of a single qubit operator is between 20-40 ns' (midpoint unless overridden)",
        "pauli_x_error": eps1,
        "pauli_x_error_location": "Table III, ibm_cairo column, 'Pauli X error' rows; attached to x and sx",
        "cnot_error": eps_cx,
        "cnot_error_location": "Table III, ibm_cairo column, 'CNOT error' (direct CNOT between [13,14])",
        "cnot_gate_time_ns": t_cx,
        "cnot_gate_time_location": "Table III, ibm_cairo column, 'Gate time (ns)'",
        "readout_model": readout,
        "readout_error": pro,
        "readout_note": ro_note,
        "depolarizing_1q": [],
        "depolarizing_2q": {},
        "relaxation_fidelity_1q": [],
        "relaxation_fidelity_2q": {},
    }
    for q in (0, 1):
        relax = Q.thermal_relaxation_error(t1[q], t2[q], t_1q)
        lam, F_relax = _depolarizing_to_match(Q, relax, eps1[q], 2)
        err = Q.depolarizing_error(lam, 1).compose(relax) if lam > 0.0 else relax
        nm.add_quantum_error(err, ["x", "sx"], [q])
        desc["depolarizing_1q"].append(lam)
        desc["relaxation_fidelity_1q"].append(F_relax)
        if pro[q] > 0.0:
            nm.add_readout_error(
                Q.ReadoutError([[1.0 - pro[q], pro[q]], [pro[q], 1.0 - pro[q]]]), [q]
            )
    for a, b in COUPLING_MAP:
        relax2 = Q.thermal_relaxation_error(t1[a], t2[a], t_cx).expand(
            Q.thermal_relaxation_error(t1[b], t2[b], t_cx)
        )
        lam2, F_relax2 = _depolarizing_to_match(Q, relax2, eps_cx, 4)
        err2 = Q.depolarizing_error(lam2, 2).compose(relax2) if lam2 > 0.0 else relax2
        nm.add_quantum_error(err2, ["cx"], [a, b])
        desc["depolarizing_2q"][f"{a}->{b}"] = lam2
        desc["relaxation_fidelity_2q"][f"{a}->{b}"] = F_relax2
    return nm, desc


# ---------------------------------------------------------------------------
# noisy, shot-free readout distributions and the paper's mitigation
# ---------------------------------------------------------------------------


def noisy_probabilities(spec, noise_model, readout_error, optimization_level=1, seed_transpiler=0):
    """Exact (shot-free) noisy readout distribution of one circuit.

    Gate noise through :func:`exact_density_matrix` on the native-gate
    circuit; readout flips applied analytically with
    :func:`readout_assignment_matrix` — identical in law to Aer's
    ``ReadoutError`` sampling, without the shot noise.

    Returns dict: ``probabilities`` ({"c1c0": p} as measured),
    ``probabilities_no_readout``, ``rho`` (dense), ``circuit`` (native),
    ``cx_count``, ``depth``, ``assignment_matrix``.
    """
    rho, tqc = exact_density_matrix(spec, noise_model, optimization_level, seed_transpiler)
    v_true = np.clip(np.real(np.diag(rho)), 0.0, None)
    v_true = v_true / v_true.sum()
    A = readout_assignment_matrix(*readout_error)
    v_meas = A @ v_true
    ops = tqc.count_ops()
    return {
        "probabilities": probabilities_to_dict(v_meas),
        "probabilities_no_readout": probabilities_to_dict(v_true),
        "rho": rho,
        "circuit": tqc,
        "cx_count": int(ops.get("cx", 0)),
        "depth": int(tqc.depth()),
        "assignment_matrix": A,
    }


def calibration_matrix(noise_model, readout_error, optimization_level=1, seed_transpiler=0):
    """The paper's measurement calibration matrix (Sec. II.D).

    "We prepared a list of 4 measurement calibration circuits for the full
    Hilbert space" — |b0 b1> prepared with X gates (which carry their own
    gate noise here, as on the device), read out through the same noisy
    simulator. Returns C[measured, prepared] in dense ordering
    (index 2 b0 + b1).
    """
    Q = _qiskit()
    sim = Q.AerSimulator(method="density_matrix", noise_model=noise_model)
    A = readout_assignment_matrix(*readout_error)
    C = np.zeros((4, 4))
    for b0, b1 in itertools.product((0, 1), (0, 1)):
        qr = Q.QuantumRegister(2, "q")
        cr = Q.ClassicalRegister(2, "c")
        qc = Q.QuantumCircuit(qr, cr, name=f"cal_{b1}{b0}")
        if b0:
            qc.x(qr[0])
        if b1:
            qc.x(qr[1])
        qc = transpile_native(qc, optimization_level, seed_transpiler)
        qc.save_density_matrix(label="rho")
        rho = to_dense_matrix(sim.run(qc, shots=1).result().data(0)["rho"])
        v = np.clip(np.real(np.diag(rho)), 0.0, None)
        C[:, 2 * b0 + b1] = A @ (v / v.sum())
    return C


def mitigate_probabilities(probs, C):
    """Invert the calibration matrix: p_true = C^{-1} p_measured, clipped at
    0 and renormalised (the paper: "we applied the calibration matrix to
    correct the measured results", Sec. II.D)."""
    v = np.linalg.solve(np.asarray(C, dtype=float), _dict_to_dense(probs))
    v = np.clip(v, 0.0, None)
    v = v / v.sum()
    return probabilities_to_dict(v)


# ---------------------------------------------------------------------------
# running on a backend (simulator here; a device is the same call)
# ---------------------------------------------------------------------------


def run_on_backend(circuit, backend, shots, seed=None, sampler=None, optimization_level=1):
    """Transpile for ``backend`` and run: returns ``{"c1c0": count}``.

    ``backend`` may be any object with ``.run`` (``AerSimulator``, a fake
    backend) or, for IBM Runtime devices that only expose primitives, pass
    ``sampler=SamplerV2(mode=backend)`` and the counts are read from the
    first classical register of ``circuit``. ``seed`` seeds the transpiler
    and, when the backend has a ``seed_simulator`` option, the simulator.
    A real-device run is therefore ``run_on_backend(qc, service.backend(
    "ibm_..."), shots, sampler=SamplerV2(mode=...))`` — this function never
    selects a device itself.
    """
    Q = _qiskit()
    if sampler is None and not hasattr(backend, "run"):
        raise TypeError(
            "backend has no .run(); pass sampler=SamplerV2(mode=backend) for "
            "primitive-only backends"
        )
    tqc = Q.transpile(circuit, backend=backend, optimization_level=int(optimization_level),
                      seed_transpiler=seed)
    if sampler is not None:
        pub = sampler.run([tqc], shots=int(shots)).result()[0]
        creg = circuit.cregs[0].name
        counts = getattr(pub.data, creg).get_counts()
    elif hasattr(backend, "run"):
        kwargs = {"shots": int(shots)}
        opts = getattr(backend, "options", None)
        if seed is not None and opts is not None and hasattr(opts, "seed_simulator"):
            kwargs["seed_simulator"] = int(seed)
        counts = backend.run(tqc, **kwargs).result().get_counts()
    else:  # pragma: no cover - guarded above
        raise TypeError("backend has no .run()")
    return {str(key).replace(" ", ""): int(val) for key, val in counts.items()}


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------


def _family(est):
    """E_A, E_B, E1 from a {observable: estimator dict} family."""
    E0 = est["E0"]["value"]
    H1 = est["H1"]["value"]
    V = est["V"]["value"]
    E1 = H1 + V                                   # Eq. (13) / Table I caption
    return {
        "E_A": E0, "E_B": -E1, "E0": E0, "H1": H1, "V": V, "E1": E1,
        "X0": est["E0"]["pauli_mean"], "Z1": est["H1"]["pauli_mean"],
        "X0X1": est["V"]["pauli_mean"],
        "std_error": {obs: est[obs]["std_error"] for obs in OBSERVABLES},
    }


def ikeda_ibm_aer(params=None, shots=None, seed=2023, readout="table_iii",
                 optimization_level=1, single_qubit_gate_time_ns=None, mitigate=True):
    """Run the paper's three circuits on Aer: ideal, noisy, sampled, mitigated.

    Parameters
    ----------
    params : see :func:`vacuum.qet.ikeda.ikeda_params` (default: the shipped
        Fig. 2 file — (h, k) = (1, 1), ibm_cairo Table III column).
    shots : int, optional (default: the file's ``shots``, 10^5).
    seed : int — transpiler and simulator seed for the sampled runs.
    readout, single_qubit_gate_time_ns : see :func:`noise_model_from_params`.
    optimization_level : transpiler level for the native-gate circuits.
    mitigate : also apply the paper's calibration-matrix mitigation.

    Returns
    -------
    dict
        ``E_A``, ``E_B`` (ideal, = the dense path), and four families
        ``ideal`` (Statevector), ``noisy_exact`` (density matrix + analytic
        readout), ``noisy_sampled`` (Aer with the noise model, ``shots``
        shots), ``mitigated`` (noisy_exact through the calibration matrix),
        each with E_A, E_B, E0, H1, V, E1, the Pauli means and the
        estimator standard errors; ``probabilities[family][observable]``;
        ``noise`` (the cited description); ``cx_count``/``depth`` of the
        native circuits; ``calibration_matrix``; ``paper`` (the file's
        printed analytical, qasm_simulator, ibm_cairo and Fig. 2(B) numbers
        when present); ``h``, ``k``, ``shots``, ``seed``, ``readout_model``.
    """
    Q = _qiskit()
    p = ikeda_params(params)
    h, k = p["h"], p["k"]
    shots = int(shots if shots is not None else p.get("shots", 100000))
    nm, desc = noise_model_from_params(p, readout=readout,
                                       single_qubit_gate_time_ns=single_qubit_gate_time_ns)
    pro = desc["readout_error"]
    C = calibration_matrix(nm, pro, optimization_level) if mitigate else None
    noisy_backend = Q.AerSimulator(noise_model=nm)

    est = {fam: {} for fam in ("ideal", "noisy_exact", "noisy_sampled", "mitigated")}
    probs = {fam: {} for fam in est}
    circuits = {}
    cx_count = {}
    depth = {}
    for i, obs in enumerate(OBSERVABLES):
        spec = ikeda_circuit_spec(p, obs, "deferred")
        psi = exact_statevector(spec)
        p_ideal = probabilities_to_dict(np.abs(psi) ** 2)
        probs["ideal"][obs] = p_ideal
        est["ideal"][obs] = expectation_from_counts(p_ideal, h, k, obs, shots=shots)

        noisy = noisy_probabilities(spec, nm, pro, optimization_level)
        probs["noisy_exact"][obs] = noisy["probabilities"]
        est["noisy_exact"][obs] = expectation_from_counts(noisy["probabilities"], h, k, obs,
                                                          shots=shots)
        circuits[obs] = noisy["circuit"]
        cx_count[obs] = noisy["cx_count"]
        depth[obs] = noisy["depth"]

        native = transpile_native(build_circuit(spec), optimization_level)
        counts = run_on_backend(native, noisy_backend, shots, seed=seed + i)
        probs["noisy_sampled"][obs], _ = counts_to_probabilities(counts)
        est["noisy_sampled"][obs] = expectation_from_counts(counts, h, k, obs)

        if mitigate:
            p_mit = mitigate_probabilities(noisy["probabilities"], C)
            probs["mitigated"][obs] = p_mit
            est["mitigated"][obs] = expectation_from_counts(p_mit, h, k, obs, shots=shots)
    families = {fam: _family(est[fam]) for fam in est if est[fam]}
    out = {
        "E_A": families["ideal"]["E_A"],
        "E_B": families["ideal"]["E_B"],
        **families,
        "probabilities": probs,
        "noise": desc,
        "circuits": circuits,
        "cx_count": cx_count,
        "depth": depth,
        "calibration_matrix": C,
        "h": h,
        "k": k,
        "shots": shots,
        "seed": seed,
        "readout_model": readout,
        "source": p.get("source"),
        "location": p.get("location"),
    }
    paper = {}
    for key in p:
        if key.startswith(("analytical_", "qasm_", "ibm_cairo_", "fig2B_")) \
                and not key.endswith("_error"):
            paper[key] = p[key]
            if key + "_error" in p:
                paper[key + "_error"] = p[key + "_error"]
    if paper:
        out["paper"] = paper
    return out
