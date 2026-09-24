"""The one hardware excursion: the L6 QET *information* signal on a real IBM
Quantum device (PLAN.md Layer 7, L6).  Run once, for real: ``ibm_fez``,
2026-09-04, job ``dadkfvl1ierc738l0ch0``, 9 600 shots per circuit — the record,
its calibration snapshot and the audited analysis live in
``papers/hardware-as-noisy-field/data/real_device/``; the standing audits have
been pointed at a real-device run and passed (E_B_info = +0.04853 +- 0.00779,
z = 6.23, against the preregistered +0.0401).

What this module does, and only this
------------------------------------
``papers/hardware-as-noisy-field/data/s6_free_tier.json`` says which of the
two minimal protocols a free-tier device can read on the sourced ibm_cairo
calibration and what that run is: **3 qubits, 11 circuits (9 measurement +
2 readout-calibration), native depth 36, <= 8 CNOTs, 3 439 shots per
circuit** for E_B_info > 0 at 3 sigma.  This module

* builds exactly that batch from the archived working point
  (:func:`build_free_tier_job`: N = 3, m = 1, d = 2, the 2-layer VQE state,
  Bob's optimised theta, the four circuit families of
  :func:`vacuum.hardware.protocols.qet_circuits` read out group by group,
  plus the two tensored-mitigation calibration circuits) and asserts the
  spec's depth / CNOT budget on the ibm_cairo native basis the survival map
  was computed with;
* transpiles it for a target backend's coupling map and basis gates on a
  chosen (or noise-selected) path of three physical qubits, with the
  two-qubit budget asserted again on the ISA circuits
  (:func:`transpile_for_backend`, :func:`select_physical_qubits`);
* submits it through ``qiskit_ibm_runtime.SamplerV2`` (:func:`submit`),
  taking the calibration snapshot from ``backend.properties()`` /
  ``backend.target`` at submission;
* collects the raw counts and persists them with full provenance
  (:func:`collect`, :func:`build_record`, :func:`save_record`,
  :func:`validate_record`);
* analyses the counts (:func:`analyze`) THROUGH the standing audits, so a
  real-device run is audited by construction:

  - **energy ledger** — first-law closure of the measured energies,
    (E_final - E_ground) = E_A - E_B, from the counts, with E_A the local
    operator <D_A>_0 on Alice's site and E_B Bob's local extraction
    <H_B>_meas - <H_B>_final, at a tolerance of n_sigma times the propagated
    shot standard error (multinomial per circuit, circuits independent).
    On hardware the measurable content of the closure is two locality
    residuals — the spectator site's energy is unchanged by the protocol,
    Bob's local energy is unchanged by Alice's measurement — because
    <h_A>_meas = <h_A>_final = omega_A holds by the measurement postulate
    for d = 2 (Alice's site is read in its sign basis) and is not a measured
    number; the ledger table says so;
  - **E_A >= E_B** at n_sigma;
  - **signal vs ideal** — the reported E_B_info may sit anywhere below its
    noiseless value (that is what the survival map measures) but not more
    than n_sigma standard errors above it: every channel in the model
    contracts the signal, so an excess is a mis-shaped result (counts on the
    wrong family or group, the wrong circuit, a mitigation blow-up), not a
    better device.  The ideal is recomputed from the record's own VQE
    parameters, so the record's stated prediction cannot buy it off;
  - **no-signaling from the marginal counts** — every ground-family and
    control-family circuit that reads the non-Alice qubits in the same
    basis must give the same marginal distribution on them (two-sample
    chi-square homogeneity, two-sided n_sigma level);

  the three checks are reported through
  :func:`vacuum.hardware.protocols.qet_audits` (the L6 standing audit) and
  :class:`vacuum.audits.AuditResult`;
* returns a verdict (:func:`verdict`): E_B_info > 0 at n_sigma AND every
  one of the four audits passed.  An audit failure on a real device is an *event* (PLAN.md
  anomaly protocol), not a number to report; ``strict=True`` raises
  :class:`vacuum.audits.AuditViolation`.

Nothing here selects or contacts a device on import.  The only network path
is :func:`runtime_service` -> :func:`submit` -> :func:`collect`, run by a
user with their own token (``scripts/run_ibm_free_tier.py``); the same code
runs unchanged in qiskit-ibm-runtime's local testing mode on a
``fake_provider`` backend (:func:`fake_backend`), whose calibration snapshot
is a real one (the date is in the record).  :func:`predict_on_backend` is
the shot-free survival analysis of :mod:`vacuum.hardware.noise` at that
backend's own calibration, on the very ISA circuits submitted (Aer density
matrix from ``AerSimulator.from_backend`` plus the backend's readout
assignment applied analytically), so a sampled verdict has a shot-free
prediction to agree with.

Record format (:data:`RECORD_FORMAT`): one JSON file per run,
``<backend>_<job id>.json`` — the spec; the working point (VQE parameters,
theta, the dense arbiter's ledger: everything needed to rebuild the
circuits and the arbiter without re-optimising); the backend (name, version,
kind, physical qubits, the calibration snapshot of the qubits and edges
used, the properties' last-update date, a sha256 of the full properties
dict); the job (id, timestamps, shots, sampler and its options, execution
metadata); one entry per circuit (label, native and ISA depth / two-qubit
counts, the raw counts); the software environment; and a sha256 of
everything else.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import chi2 as _chi2_dist
from scipy.stats import norm as _norm

from vacuum.audits import AuditResult, AuditViolation
from vacuum.hardware import protocols as pr
from vacuum.hardware import vacuum_sim as vs
from vacuum.hardware.protocols import (
    MeasurementGroup,
    calibration_from_counts,
    counts_to_vector,
    group_statistics,
    measurement_circuits,
    qet_audits,
    qet_circuits,
    qet_dense,
    qet_from_counts,
    readout_calibration_circuits,
)
from vacuum.hardware.vacuum_sim import PauliTerm, _merge, _multiply_disjoint, _site_terms, site_operators

__all__ = [
    "RECORD_FORMAT",
    "READOUT_REGISTER",
    "TWO_QUBIT_GATES",
    "FAMILIES",
    "FREE_TIER_SPEC",
    "WORKING_POINT",
    "BudgetError",
    "FreeTierJob",
    "Submission",
    "working_point",
    "build_free_tier_job",
    "batch_circuits",
    "assert_spec",
    "two_qubit_gate_name",
    "select_physical_qubits",
    "transpile_for_backend",
    "calibration_snapshot",
    "backend_info",
    "backend_kind",
    "qpu_time_estimate",
    "submission_info",
    "submit",
    "save_submission",
    "load_submission",
    "counts_from_result",
    "collect",
    "build_record",
    "record_sha256",
    "validate_record",
    "save_record",
    "load_record",
    "default_record_path",
    "analysis_path",
    "save_analysis",
    "load_analysis",
    "exact_counts",
    "predict_on_backend",
    "analyze",
    "verdict",
    "run_batch",
    "fake_backend",
    "runtime_service",
    "least_busy_backend",
    "nz_shots",
    "environment_info",
]

RECORD_FORMAT = "vacuum.hardware.real_device/1"
#: the classical register every batch circuit reads into (protocols' ``m0``)
READOUT_REGISTER = "m0"
#: native two-qubit gates of IBM devices, in order of preference
TWO_QUBIT_GATES = ("cx", "ecr", "cz")
FAMILIES = ("ground", "control", "protocol", "wrong")

#: The free-tier run as specified by the survival map — transcribed from
#: ``papers/hardware-as-noisy-field/data/s6_free_tier.json`` (key
#: ``qet_information_signal``); ``tests/test_hardware_real_device.py``
#: checks these against the archived file.
FREE_TIER_SPEC = {
    "source": "papers/hardware-as-noisy-field/data/s6_free_tier.json: qet_information_signal",
    "qubits": 3,
    "circuits": 11,
    "measurement_circuits": 9,
    "calibration_circuits": 2,
    "depth_native": 36,
    "cx_count_max": 8,
    "native_basis": ["rz", "sx", "x", "cx"],
    "shots_per_circuit_for_3sigma": 3439,
    "predicted_value_at_sourced_noise": 0.040103899558164624,
    "predicted_std_error_at_sourced_noise": 0.0024788229782663268,
    "predicted_std_error_shots": 100000,
    "ideal_value": 0.04152218677914288,
    "number_tested": ("E_B_info = <H_B>_wrong-bit - <H_B>_right-bit > 0 at 3 sigma: the classical "
                      "bit is worth a positive amount of Bob's local energy"),
    "open_plan_budget_s": 600.0,
}

#: The archived working point (notebook.py: N = 3, m = 1, Dirichlet, d = 2,
#: 2-layer hardware-efficient VQE with 3 seeded restarts, Alice on site 0,
#: Bob on site 1).
WORKING_POINT = {
    "N": 3, "m": 1.0, "d": 2, "bc": "dirichlet",
    "layers": 2, "restarts": 3, "seed": 0, "maxiter": 400,
    "site_A": 0, "site_B": 1,
}

_DENSE_KEYS = ("E_ground", "E_A", "E_B", "E_B_wrong", "E_B_info", "E_B_cut", "E_B_raw",
               "HB_ground", "HB_meas", "HB_final", "HB_wrong", "ledger_defect", "no_signaling")


class BudgetError(RuntimeError):
    """The batch exceeds the spec's circuit / depth / two-qubit budget."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _qk():
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    import qiskit_aer.library  # noqa: F401  (save_probabilities)
    return transpile, Statevector, AerSimulator, NoiseModel


def _utcnow():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def _jsonable(o):
    return json.loads(json.dumps(o, default=_json_default))


def _stats(tqc):
    ops = tqc.count_ops()
    two = int(sum(int(ops.get(g, 0)) for g in TWO_QUBIT_GATES))
    return {
        "depth": int(tqc.depth()),
        "cx_count": int(ops.get("cx", 0)),
        "two_qubit_count": two,
        "n_gates": int(sum(int(v) for k, v in ops.items() if k not in ("measure", "barrier"))),
        "ops": {str(k): int(v) for k, v in ops.items()},
    }


def _strip(qc):
    body = qc.copy()
    body.data = [inst for inst in qc.data if inst.operation.name not in ("measure", "barrier")]
    return body


def _audit_dict(res: AuditResult):
    return {"name": res.name, "passed": bool(res.passed), "worst_violation": float(res.worst_violation),
            "details": _jsonable(res.details)}


def _neg(terms):
    return [PauliTerm(t.paulis, -t.coeff) for t in terms]


def _terms_of(operator):
    """(groups, constant) of a protocols operator -> flat PauliTerm list."""
    groups, const = operator
    out = [PauliTerm((), complex(const))] if const else []
    for g in groups:
        out += list(g.terms)
    return out


# ---------------------------------------------------------------------------
# the batch
# ---------------------------------------------------------------------------


@dataclass
class FreeTierJob:
    """The built (and, after :func:`transpile_for_backend`, ISA) batch."""

    params: Dict[str, Any]
    chain: Any
    vqe: Any
    theta: float
    dense: Dict[str, Any]
    fam: Dict[str, Any]
    circuits: List[Any]
    labels: List[Dict[str, Any]]
    native: List[Any]
    native_stats: List[Dict[str, Any]]
    backend_name: Optional[str] = None
    backend_kind: Optional[str] = None
    physical_qubits: Optional[List[int]] = None
    two_qubit_gate: Optional[str] = None
    isa: Optional[List[Any]] = None
    isa_stats: Optional[List[Dict[str, Any]]] = None
    layout_score: Optional[Dict[str, Any]] = None

    @property
    def n_circuits(self):
        return len(self.circuits)

    @property
    def n_qubits(self):
        return int(self.chain.n_qubits)

    @property
    def native_depth(self):
        return max(s["depth"] for s in self.native_stats)

    @property
    def native_cx(self):
        return max(s["cx_count"] for s in self.native_stats)

    @property
    def isa_depth(self):
        return None if self.isa_stats is None else max(s["depth"] for s in self.isa_stats)

    @property
    def isa_two_qubit(self):
        return None if self.isa_stats is None else max(s["two_qubit_count"] for s in self.isa_stats)


def working_point(overrides=None):
    """The archived working point with optional VQE-level overrides; the
    batch IS the N = 3, d = 2 point of s6 and refuses anything else."""
    p = dict(WORKING_POINT)
    p.update(dict(overrides or {}))
    if (int(p["N"]), int(p["d"])) != (3, 2):
        raise ValueError("the free-tier batch is the N = 3, d = 2 working point of s6_free_tier.json "
                         f"(3 qubits); got N = {p['N']}, d = {p['d']}")
    return p


def batch_circuits(fam, chain):
    """The 11 logical circuits in batch order (families ground, control,
    protocol, wrong; operators in family order; groups in order; then the
    |0..0> and |1..1> calibration circuits) and one label per circuit."""
    circuits, labels = [], []
    for name in FAMILIES:
        f = fam[name]
        for opname, (groups, _const) in f["operators"].items():
            mcs = measurement_circuits(f["body"], groups, f["measured"], name=f"{name}_{opname}")
            for gi, (qc, g) in enumerate(zip(mcs, groups)):
                circuits.append(qc)
                labels.append({"index": len(labels), "kind": "measurement", "family": name,
                               "operator": opname, "group": gi, "name": qc.name,
                               "basis": {str(q): ch for q, ch in sorted(g.basis.items())},
                               "measured": [int(q) for q in f["measured"]]})
    n = chain.n_qubits
    for qc in readout_calibration_circuits(n):
        prepared = "1" * n if qc.name == "cal_ones" else "0" * n
        circuits.append(qc)
        labels.append({"index": len(labels), "kind": "calibration", "family": None, "operator": None,
                       "group": None, "name": qc.name, "prepared": prepared,
                       "measured": list(range(n))})
    for qc in circuits:
        if qc.cregs[-1].name != READOUT_REGISTER:
            raise RuntimeError(f"{qc.name}: readout register is {qc.cregs[-1].name!r}, expected {READOUT_REGISTER!r}")
    return circuits, labels


def assert_spec(job: FreeTierJob, spec=None):
    """Raise :class:`BudgetError` unless the batch is the spec's batch."""
    spec = FREE_TIER_SPEC if spec is None else spec
    n_meas = sum(1 for l in job.labels if l["kind"] == "measurement")
    n_cal = sum(1 for l in job.labels if l["kind"] == "calibration")
    problems = []
    if job.n_qubits != spec["qubits"]:
        problems.append(f"qubits {job.n_qubits} != {spec['qubits']}")
    if job.n_circuits != spec["circuits"] or n_meas != spec["measurement_circuits"] or n_cal != spec["calibration_circuits"]:
        problems.append(f"circuits {job.n_circuits} ({n_meas} + {n_cal}) != {spec['circuits']} "
                        f"({spec['measurement_circuits']} + {spec['calibration_circuits']})")
    if job.native_depth > spec["depth_native"]:
        problems.append(f"native depth {job.native_depth} > {spec['depth_native']}")
    if job.native_cx > spec["cx_count_max"]:
        problems.append(f"native CNOTs {job.native_cx} > {spec['cx_count_max']}")
    if job.isa_stats is not None:
        for lab, st, ns in zip(job.labels, job.isa_stats, job.native_stats):
            if st["two_qubit_count"] > spec["cx_count_max"]:
                problems.append(f"{lab['name']}: ISA two-qubit gates {st['two_qubit_count']} > {spec['cx_count_max']}")
            if st["two_qubit_count"] > ns["cx_count"]:
                problems.append(f"{lab['name']}: ISA two-qubit gates {st['two_qubit_count']} > native {ns['cx_count']} (routing inserted)")
    if problems:
        raise BudgetError("batch off spec: " + "; ".join(problems))
    return True


def build_free_tier_job(params=None, backend=None, physical_qubits=None, optimization_level=1,
                        seed_transpiler=0, vqe=None):
    """The exact 11-circuit batch of s6_free_tier.json.

    Builds the truncated chain, the VQE state (or takes ``vqe``, a
    :class:`vacuum.hardware.vacuum_sim.VQEResult` of the same point), Bob's
    optimised theta, the dense arbiter ledger (audited before anything is
    built on it), the four circuit families, the 9 measurement + 2
    calibration circuits, and their ibm_cairo-native transpilation
    (``vacuum.hardware.protocols.transpile_native`` on the linear coupling
    map — the basis the survival map used); asserts the spec.  With a
    ``backend`` the ISA transpilation is done too
    (:func:`transpile_for_backend`).
    """
    p = working_point(params)
    chain = vs.truncated_chain(p["N"], p["m"], p["d"], bc=p["bc"])
    if vqe is None:
        vqe = vs.vqe_ground_state(chain, layers=p["layers"], restarts=p["restarts"], seed=p["seed"],
                                  maxiter=p["maxiter"], evaluator="aer")
    A, B = int(p["site_A"]), int(p["site_B"])
    theta, _ = pr.qet_optimize_theta(chain, vqe.state, A, B)
    dense = qet_dense(chain, vqe.state, A, B, theta)
    qet_audits(dense, tol=1e-10, ledger_tol=1e-10, signaling_tol=1e-10)
    fam = qet_circuits(chain, vqe.params, vqe.layers, A, B, theta, form="deferred")
    circuits, labels = batch_circuits(fam, chain)
    cmap = fam["meta"]["coupling_map"]
    native = [pr.transpile_native(qc, cmap, optimization_level, seed_transpiler) for qc in circuits]
    job = FreeTierJob(params=p, chain=chain, vqe=vqe, theta=float(theta), dense=dense, fam=fam,
                      circuits=circuits, labels=labels, native=native,
                      native_stats=[_stats(t) for t in native])
    assert_spec(job)
    if backend is not None:
        transpile_for_backend(job, backend, physical_qubits, optimization_level, seed_transpiler)
    return job


# ---------------------------------------------------------------------------
# the target backend: layout, transpilation, calibration snapshot
# ---------------------------------------------------------------------------


def backend_kind(backend):
    mod = type(backend).__module__ or ""
    if "fake_provider" in mod:
        return "fake"
    if mod.startswith("qiskit_aer"):
        return "aer"
    if mod.startswith("qiskit_ibm_runtime"):
        return "ibm_runtime"
    return "other"


def two_qubit_gate_name(backend):
    names = [g for g in TWO_QUBIT_GATES if g in backend.target.operation_names]
    if not names:
        raise ValueError(f"backend {backend.name} has none of {TWO_QUBIT_GATES} in its target")
    return names[0]


def _target_prop(target, op, qargs, attr="error"):
    try:
        if op not in target.operation_names:
            return None
        props = target[op].get(tuple(int(q) for q in qargs))
        if props is None:
            return None
        v = getattr(props, attr, None)
        return None if v is None else float(v)
    except Exception:
        return None


def _edge_error(target, gate, a, b):
    for pair in ((a, b), (b, a)):
        e = _target_prop(target, gate, pair)
        if e is not None:
            return e
    return None


def _operational(target, q):
    try:
        qp = target.qubit_properties[q]
    except Exception:
        return True
    return bool(getattr(qp, "operational", True))


def _undirected_adjacency(backend):
    adj: Dict[int, set] = {}
    for a, b in backend.coupling_map.get_edges():
        adj.setdefault(int(a), set()).add(int(b))
        adj.setdefault(int(b), set()).add(int(a))
    return adj


def _layout_score(backend, path):
    target = backend.target
    gate = two_qubit_gate_name(backend)
    edge = {}
    for a, b in zip(path[:-1], path[1:]):
        edge[f"{a}-{b}"] = _edge_error(target, gate, a, b)
    ro = {str(q): _target_prop(target, "measure", (q,)) for q in path}
    sx = {str(q): _target_prop(target, "sx", (q,)) for q in path}
    parts = list(edge.values()) + list(ro.values()) + list(sx.values())
    score = sum(v for v in parts if v is not None)
    return {"score": float(score), "two_qubit_gate": gate, "edge_errors": edge, "readout_errors": ro,
            "sx_errors": sx, "complete": all(v is not None for v in parts),
            "rule": "sum of the two-qubit gate errors on the path edges and of the readout and sx "
                    "errors on its qubits (backend.target); lowest wins, ties by qubit index"}


def _is_path(backend, qubits):
    adj = _undirected_adjacency(backend)
    q = [int(x) for x in qubits]
    return len(set(q)) == len(q) and all(b in adj.get(a, ()) for a, b in zip(q[:-1], q[1:]))


def select_physical_qubits(backend, n=3):
    """The lowest-noise simple path of ``n`` physical qubits (Alice at one
    end, Bob in the middle, the spectator at the other end): every simple
    path of the undirected coupling map is scored by :func:`_layout_score`
    (operational qubits only).  Returns (path, score dict)."""
    n = int(n)
    adj = _undirected_adjacency(backend)
    target = backend.target
    ok = {q for q in adj if _operational(target, q)}
    paths = []

    def walk(path):
        if len(path) == n:
            if path[0] < path[-1]:          # a path and its reverse score alike
                paths.append(list(path))
            return
        for nxt in sorted(adj.get(path[-1], ())):
            if nxt in ok and nxt not in path:
                walk(path + [nxt])

    for start in sorted(ok):
        walk([start])
    if not paths:
        raise ValueError(f"no path of {n} operational qubits on {backend.name}")
    scored = [(_layout_score(backend, p), p) for p in paths]
    complete = [sp for sp in scored if sp[0]["complete"]] or scored
    best_score, best = min(complete, key=lambda sp: (sp[0]["score"], sp[1]))
    best_score = dict(best_score, n_candidates=len(paths))
    return best, best_score


def transpile_for_backend(job: FreeTierJob, backend, physical_qubits=None, optimization_level=1,
                          seed_transpiler=0):
    """ISA circuits for ``backend`` on a fixed path of physical qubits
    (all 11 circuits on the same qubits, so the calibration circuits
    calibrate the qubits the measurement circuits read); asserts the
    two-qubit budget and that no routing was inserted."""
    transpile, _, _, _ = _qk()
    if physical_qubits is None:
        physical_qubits, score = select_physical_qubits(backend, job.n_qubits)
    else:
        physical_qubits = [int(q) for q in physical_qubits]
        if len(physical_qubits) != job.n_qubits:
            raise ValueError(f"need {job.n_qubits} physical qubits, got {physical_qubits}")
        if not _is_path(backend, physical_qubits):
            raise ValueError(f"{physical_qubits} is not a path of the coupling map of {backend.name}")
        score = _layout_score(backend, physical_qubits)
    isa = [transpile(qc, backend=backend, initial_layout=list(physical_qubits),
                     optimization_level=int(optimization_level), seed_transpiler=seed_transpiler)
           for qc in job.circuits]
    for lab, tqc in zip(job.labels, isa):
        got = pr._readout_qubits(tqc)
        want = [physical_qubits[k] for k in lab["measured"]]
        if got != want:
            raise BudgetError(f"{lab['name']}: reads physical qubits {got}, expected {want} (routing permuted the register)")
    job.backend_name = str(backend.name)
    job.backend_kind = backend_kind(backend)
    job.physical_qubits = [int(q) for q in physical_qubits]
    job.two_qubit_gate = two_qubit_gate_name(backend)
    job.isa = isa
    job.isa_stats = [_stats(t) for t in isa]
    job.layout_score = score
    assert_spec(job)
    return job


def calibration_snapshot(backend, qubits):
    """Calibration of the qubits and edges used: ``backend.target`` numbers
    (T1/T2, readout / sx / x errors and durations, the two-qubit gate error
    and duration in both directions) plus, when the backend offers
    ``properties()``, its last-update date, per-qubit T1/T2/readout/
    prob_meas1_prep0/prob_meas0_prep1 and a sha256 of the full dict."""
    target = backend.target
    gate = two_qubit_gate_name(backend)
    qubits = [int(q) for q in qubits]
    snap: Dict[str, Any] = {
        "backend": str(backend.name),
        "backend_version": getattr(backend, "backend_version", None),
        "taken_utc": _utcnow(),
        "source": "backend.target",
        "two_qubit_gate": gate,
        "qubits": {},
        "edges": {},
    }
    for q in qubits:
        qp = None
        try:
            qp = target.qubit_properties[q]
        except Exception:
            pass
        t1 = getattr(qp, "t1", None)
        t2 = getattr(qp, "t2", None)
        fr = getattr(qp, "frequency", None)
        snap["qubits"][str(q)] = {
            "t1_us": None if t1 is None else float(t1) * 1e6,
            "t2_us": None if t2 is None else float(t2) * 1e6,
            "frequency_GHz": None if fr is None else float(fr) / 1e9,
            "readout_error": _target_prop(target, "measure", (q,)),
            "readout_length_ns": _scaled(_target_prop(target, "measure", (q,), "duration"), 1e9),
            "sx_error": _target_prop(target, "sx", (q,)),
            "x_error": _target_prop(target, "x", (q,)),
            "sx_length_ns": _scaled(_target_prop(target, "sx", (q,), "duration"), 1e9),
            "operational": _operational(target, q),
        }
    for a, b in zip(qubits[:-1], qubits[1:]):
        for pair in ((a, b), (b, a)):
            e = _target_prop(target, gate, pair)
            if e is not None:
                snap["edges"][f"{pair[0]}-{pair[1]}"] = {
                    "gate": gate, "error": e,
                    "length_ns": _scaled(_target_prop(target, gate, pair, "duration"), 1e9)}
    props = None
    try:
        props = backend.properties()
    except Exception as exc:  # pragma: no cover - backend-specific
        snap["properties_error"] = repr(exc)
    if props is not None:
        snap["source"] = "backend.properties() + backend.target"
        lu = getattr(props, "last_update_date", None)
        snap["last_update_date"] = lu.isoformat() if hasattr(lu, "isoformat") else (None if lu is None else str(lu))
        snap["properties_backend_version"] = getattr(props, "backend_version", None)
        for q in qubits:
            row = snap["qubits"][str(q)]
            for key, scale, out in (("T1", 1e6, "properties_t1_us"), ("T2", 1e6, "properties_t2_us"),
                                    ("readout_error", 1.0, "properties_readout_error"),
                                    ("prob_meas1_prep0", 1.0, "properties_prob_meas1_prep0"),
                                    ("prob_meas0_prep1", 1.0, "properties_prob_meas0_prep1")):
                try:
                    val = props.qubit_property(q, key)
                    val = val[0] if isinstance(val, tuple) else val
                    row[out] = float(val) * scale
                except Exception:
                    row[out] = None
        try:
            d = props.to_dict()
            snap["properties_sha256"] = hashlib.sha256(
                json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
            snap["properties_n_qubits"] = len(d.get("qubits", []))
        except Exception as exc:  # pragma: no cover
            snap["properties_sha256"] = None
            snap["properties_error"] = repr(exc)
    return snap


def _scaled(v, s):
    return None if v is None else float(v) * s


def backend_info(backend, physical_qubits):
    kind = backend_kind(backend)
    return {
        "name": str(backend.name),
        "kind": kind,
        "simulated": kind != "ibm_runtime",
        "class": f"{type(backend).__module__}.{type(backend).__qualname__}",
        "version": getattr(backend, "backend_version", None),
        "num_qubits": int(getattr(backend, "num_qubits", 0) or 0),
        "physical_qubits": [int(q) for q in physical_qubits],
        "two_qubit_gate": two_qubit_gate_name(backend),
        "calibration": calibration_snapshot(backend, physical_qubits),
    }


def qpu_time_estimate(n_circuits, shots):
    """IBM's transcribed QPU-time estimate (the sweep file: 2 s + 0.00035 s
    x circuits x shots) against the Open Plan budget."""
    from vacuum import experiments_io as xio
    from vacuum.hardware import noise as nz

    exp = xio.load_experiment(nz.SWEEP_FILE)
    t0 = float(exp.quantities["qpu_time_overhead"].value)
    per = float(exp.quantities["qpu_time_per_execution"].value)
    budget = float(exp.quantities["open_plan_qpu_budget"].value)
    t = t0 + per * int(n_circuits) * int(shots)
    return {"seconds": t, "formula": f"{t0} s + {per} s x (circuits x shots)", "circuits": int(n_circuits),
            "shots_per_circuit": int(shots), "executions": int(n_circuits) * int(shots),
            "open_plan_budget_s": budget, "fits_open_plan_budget": t <= budget,
            "source": exp.source}


# ---------------------------------------------------------------------------
# submission, collection, the record
# ---------------------------------------------------------------------------


@dataclass
class Submission:
    """What :func:`submit` returns: the JSON-serialisable ``info`` (saved as
    the ``.submitted.json`` sidecar) and the live runtime job."""

    info: Dict[str, Any]
    runtime_job: Any = None

    @property
    def job_id(self):
        return self.info["job_id"]


def _working_point_record(job: FreeTierJob):
    v = job.vqe
    return {
        **{k: job.params[k] for k in ("N", "m", "d", "bc", "layers", "restarts", "seed", "maxiter", "site_A", "site_B")},
        "theta": float(job.theta),
        "vqe_params": [float(x) for x in np.asarray(v.params).ravel()],
        "vqe_energy": float(v.energy), "vqe_energy_error": float(v.energy_error),
        "vqe_fidelity": float(v.fidelity), "vqe_evaluator": str(v.evaluator),
        "dense": {k: float(job.dense[k]) for k in _DENSE_KEYS},
    }


def submission_info(job: FreeTierJob, binfo, shots, job_id, sampler, sampler_options=None, seed=None,
                    submitted_utc=None, extra=None):
    if job.isa_stats is None:
        raise ValueError("transpile the job for the backend first")
    info = {
        "format": RECORD_FORMAT,
        "job_id": str(job_id),
        "submitted_utc": submitted_utc or _utcnow(),
        "shots": int(shots),
        "sampler": str(sampler),
        "sampler_options": _jsonable(sampler_options) if sampler_options is not None else None,
        "seed": None if seed is None else int(seed),
        "backend": binfo,
        "physical_qubits": list(job.physical_qubits),
        "layout_score": _jsonable(job.layout_score),
        "n_circuits": job.n_circuits,
        "labels": _jsonable(job.labels),
        "native_stats": _jsonable(job.native_stats),
        "isa_stats": _jsonable(job.isa_stats),
        "native_basis": list(pr.IBM_BASIS_GATES),
        "native_coupling_map": _jsonable(job.fam["meta"]["coupling_map"]),
        "working_point": _working_point_record(job),
        "spec": FREE_TIER_SPEC,
        "qpu_time_estimate": qpu_time_estimate(job.n_circuits, shots),
    }
    if extra:
        info.update(_jsonable(extra))
    return info


def _options_dict(sampler):
    try:
        return _jsonable(dataclasses.asdict(sampler.options))
    except Exception:
        try:
            return str(sampler.options)
        except Exception:  # pragma: no cover
            return None


def submit(job: FreeTierJob, backend, shots, sampler=None, seed=None, tags=None, physical_qubits=None):
    """Submit the ISA batch through ``qiskit_ibm_runtime.SamplerV2(mode=
    backend)`` (or the ``sampler`` given): all 11 circuits, the two
    calibration circuits included, ``shots`` each, no server-side
    mitigation, twirling or dynamical decoupling (the defaults) — the
    counts come back raw and the analysis mitigates them itself.  The
    calibration snapshot is taken at submission.  ``seed`` seeds the
    simulator in local testing mode only.  Returns a :class:`Submission`."""
    if job.isa is None or job.backend_name != str(backend.name):
        transpile_for_backend(job, backend, physical_qubits)
    if sampler is None:
        from qiskit_ibm_runtime import SamplerV2

        sampler = SamplerV2(mode=backend)
    if seed is not None and backend_kind(backend) != "ibm_runtime":
        if int(seed) == 0:
            # qiskit-ibm-runtime 0.49's local sampler treats seed_simulator = 0
            # as "unset" (falsy), so a run seeded with 0 is silently NOT
            # reproducible.  Refuse rather than persist a record whose 'seed'
            # field is a fiction.
            raise ValueError("seed 0 is ignored by qiskit-ibm-runtime's local sampler "
                             "(it tests the value for truth, not for None) — use a nonzero seed")
        try:
            sampler.options.simulator.seed_simulator = int(seed)
        except Exception:
            pass
    if tags:
        try:
            sampler.options.environment.job_tags = [str(t) for t in tags]
        except Exception:
            pass
    binfo = backend_info(backend, job.physical_qubits)
    submitted = _utcnow()
    rjob = sampler.run(list(job.isa), shots=int(shots))
    info = submission_info(job, binfo, shots, rjob.job_id(),
                           f"{type(sampler).__module__}.{type(sampler).__qualname__}",
                           _options_dict(sampler), seed, submitted)
    return Submission(info=info, runtime_job=rjob)


def save_submission(sub, path):
    info = sub.info if isinstance(sub, Submission) else sub
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(info, fh, indent=1, default=_json_default)
    return path


def load_submission(path):
    with open(path) as fh:
        info = json.load(fh)
    if info.get("format") != RECORD_FORMAT:
        raise ValueError(f"{path}: not a {RECORD_FORMAT} submission")
    return info


def counts_from_result(result, n_circuits):
    """Raw counts of the readout register of every pub, batch order."""
    out = []
    for i in range(int(n_circuits)):
        data = result[i].data
        reg = getattr(data, READOUT_REGISTER)
        out.append({str(k).replace(" ", ""): int(v) for k, v in reg.get_counts().items()})
    return out


def collect(runtime_job_or_id, submission, service=None, path=None):
    """Fetch the result of a submitted batch (a job object, or a job id
    looked up through ``service`` / the saved account) and persist the raw
    counts with the submission's provenance.  Returns the record."""
    info = submission.info if isinstance(submission, Submission) else submission
    if isinstance(runtime_job_or_id, str):
        if service is None:
            service = runtime_service()
        rjob = service.job(runtime_job_or_id)
    else:
        rjob = runtime_job_or_id
    result = rjob.result()
    counts = counts_from_result(result, info["n_circuits"])
    execution: Dict[str, Any] = {"result_metadata": _jsonable(getattr(result, "metadata", None)),
                                 "pub_metadata": [_jsonable(result[i].metadata) for i in range(len(counts))]}
    for attr in ("status", "usage", "creation_date", "backend"):
        try:
            v = getattr(rjob, attr)
            v = v() if callable(v) else v
            execution[attr] = _jsonable(getattr(v, "name", v))
        except Exception:
            pass
    job_id = str(rjob.job_id()) if hasattr(rjob, "job_id") else info["job_id"]
    if job_id != info["job_id"]:
        raise ValueError(f"job id {job_id} does not match the submission's {info['job_id']}")
    record = build_record(info, counts, execution)
    if path is not None:
        save_record(record, path)
    return record


def build_record(submission, counts, execution=None):
    """Assemble and hash the record (module docstring) from a submission's
    info and one counts dict per circuit."""
    info = submission.info if isinstance(submission, Submission) else submission
    labels = info["labels"]
    if len(counts) != len(labels):
        raise ValueError(f"{len(counts)} count dicts for {len(labels)} circuits")
    circuits = []
    for lab, ns, st, c in zip(labels, info["native_stats"], info["isa_stats"], counts):
        circuits.append({**lab, "native": ns, "isa": st, "counts": {str(k): v for k, v in c.items()}})
    record = {
        "format": RECORD_FORMAT,
        "spec": info["spec"],
        "working_point": info["working_point"],
        "backend": info["backend"],
        "layout_score": info.get("layout_score"),
        "native_basis": info.get("native_basis"),
        "native_coupling_map": info.get("native_coupling_map"),
        "job": {
            "job_id": info["job_id"],
            "submitted_utc": info["submitted_utc"],
            "collected_utc": _utcnow(),
            "shots": int(info["shots"]),
            "sampler": info["sampler"],
            "sampler_options": info.get("sampler_options"),
            "seed": info.get("seed"),
            "qpu_time_estimate": info.get("qpu_time_estimate"),
            "preflight": info.get("preflight"),
            "execution": execution or {},
        },
        "circuits": circuits,
        "environment": environment_info(),
    }
    record = _jsonable(record)
    record["sha256"] = record_sha256(record)
    return record


def record_sha256(record):
    body = {k: v for k, v in record.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     default=_json_default).encode()).hexdigest()


def validate_record(record, check_hash=True):
    """Structural validation of a record; raises ValueError."""
    if not isinstance(record, dict) or record.get("format") != RECORD_FORMAT:
        raise ValueError(f"not a {RECORD_FORMAT} record")
    if check_hash and record.get("sha256") != record_sha256(record):
        raise ValueError("record sha256 does not match its content (file edited or truncated)")
    spec = record.get("spec") or {}
    for key in ("working_point", "backend", "job", "circuits", "environment"):
        if key not in record:
            raise ValueError(f"record lacks {key!r}")
    wp = record["working_point"]
    for key in ("N", "m", "d", "bc", "layers", "site_A", "site_B", "theta", "vqe_params", "dense"):
        if key not in wp:
            raise ValueError(f"working_point lacks {key!r}")
    n = int(wp["N"]) * int(round(math.log2(int(wp["d"]))))
    b = record["backend"]
    for key in ("name", "kind", "simulated", "physical_qubits", "calibration"):
        if key not in b:
            raise ValueError(f"backend lacks {key!r}")
    if len(b["physical_qubits"]) != n:
        raise ValueError(f"backend.physical_qubits has {len(b['physical_qubits'])} entries, need {n}")
    j = record["job"]
    for key in ("job_id", "submitted_utc", "shots"):
        if key not in j:
            raise ValueError(f"job lacks {key!r}")
    circuits = record["circuits"]
    if len(circuits) != int(spec.get("circuits", FREE_TIER_SPEC["circuits"])):
        raise ValueError(f"{len(circuits)} circuits, spec says {spec.get('circuits')}")
    shots = float(j["shots"])
    seen = {}
    for c in circuits:
        for key in ("index", "kind", "name", "native", "isa", "counts", "measured"):
            if key not in c:
                raise ValueError(f"circuit {c.get('name')} lacks {key!r}")
        counts = c["counts"]
        if not counts:
            raise ValueError(f"circuit {c['name']}: empty counts")
        if any(len(k) != n or set(k) - {"0", "1"} for k in counts):
            raise ValueError(f"circuit {c['name']}: count keys are not {n}-bit strings")
        tot = float(sum(counts.values()))
        if abs(tot - shots) > 1e-6 * shots + 1e-9:
            raise ValueError(f"circuit {c['name']}: counts sum to {tot}, shots = {shots}")
        if c["kind"] == "measurement":
            seen.setdefault((c["family"], c["operator"]), []).append(int(c["group"]))
        elif c["kind"] == "calibration":
            if c.get("prepared") not in ("0" * n, "1" * n):
                raise ValueError(f"calibration circuit {c['name']}: prepared = {c.get('prepared')!r}")
        else:
            raise ValueError(f"circuit {c['name']}: kind {c['kind']!r}")
    for (fam, op), groups in seen.items():
        if groups != list(range(len(groups))):
            raise ValueError(f"{fam}/{op}: groups {groups} are not 0..{len(groups) - 1} in order")
    fams = {f for f, _ in seen}
    if fams != set(FAMILIES):
        raise ValueError(f"families {sorted(fams)} != {list(FAMILIES)}")
    if sum(1 for c in circuits if c["kind"] == "calibration") != 2:
        raise ValueError("need exactly two calibration circuits")
    return True


def save_record(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=1, default=_json_default)
    return path


def load_record(path, check_hash=True):
    with open(path) as fh:
        record = json.load(fh)
    validate_record(record, check_hash=check_hash)
    return record


def default_record_path(out_dir, record_or_info, suffix=".json"):
    r = record_or_info.info if isinstance(record_or_info, Submission) else record_or_info
    name = r["backend"]["name"]
    job_id = r["job"]["job_id"] if "job" in r else r["job_id"]
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(job_id))
    return Path(out_dir) / f"{name}_{safe}{suffix}"


def analysis_path(record_path):
    """``<record stem>.analysis.json`` next to the record."""
    rp = Path(record_path)
    return rp.with_name(rp.stem + ".analysis.json")


def save_analysis(analysis, path):
    """Persist an :func:`analyze` result (it carries ``record_sha256``, so the
    record it was derived from is pinned)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(analysis, fh, indent=1, default=_json_default)
    return path


def load_analysis(path):
    with open(path) as fh:
        an = json.load(fh)
    if an.get("format") != RECORD_FORMAT or "record_sha256" not in an or "verdict" not in an:
        raise ValueError(f"{path}: not a {RECORD_FORMAT} analysis")
    return an


def environment_info():
    env: Dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "scipy", "qiskit", "qiskit_aer", "qiskit_ibm_runtime", "vacuum"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", None)
        except Exception:
            env[mod] = None
    try:
        root = Path(__file__).resolve().parents[2]
        env["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                         cwd=root, check=True, timeout=10).stdout.strip()
        env["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                               text=True, cwd=root, check=True, timeout=10).stdout.strip())
    except Exception:
        env["git_head"] = None
    return env


# ---------------------------------------------------------------------------
# shot-free companions: noiseless statevector, and Aer at the backend's calibration
# ---------------------------------------------------------------------------


def _apply_assignment(p, mats):
    """Push a probability vector (bit k <- mats[k], A[measured, prepared])
    through per-bit readout assignment matrices."""
    p = np.asarray(p, dtype=float)
    n = int(round(math.log2(p.size)))
    t = p.reshape((2,) * n)
    for k, A in enumerate(mats):
        A = np.asarray(A, dtype=float)
        if np.allclose(A, np.eye(2)):
            continue
        ax = n - 1 - k
        t = np.moveaxis(np.tensordot(A, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def _readout_assignment_from_backend(backend):
    """{physical qubit: A[measured, prepared]} of ``NoiseModel.from_backend``
    (the readout errors the local sampler applies), read from its dict."""
    _, _, _, NoiseModel = _qk()
    nm = NoiseModel.from_backend(backend)
    out = {}
    for err in nm.to_dict().get("errors", []):
        if err.get("type") != "roerror":
            continue
        P = np.asarray(err["probabilities"], dtype=float)      # rows: prepared, cols: measured
        for qs in err.get("gate_qubits", []):
            if len(qs) == 1:
                out[int(qs[0])] = P.T
    return out


def exact_counts(job: FreeTierJob, shots, backend=None):
    """Shot-free expected counts (probability x shots, floats) of the batch:
    noiseless from ``Statevector`` on the logical circuits (``backend`` None)
    or at ``backend``'s calibration on its ISA circuits (Aer density matrix
    from ``AerSimulator.from_backend``, the backend's readout assignment
    applied analytically)."""
    transpile, Statevector, AerSimulator, _ = _qk()
    n = job.n_qubits
    vectors = []
    if backend is None:
        for qc, lab in zip(job.circuits, job.labels):
            if lab["measured"] != list(range(n)):
                raise ValueError("noiseless exact counts assume measured qubits 0..n-1 in order")
            vectors.append(np.asarray(Statevector(_strip(qc)).probabilities(), dtype=float))
    else:
        if job.isa is None or job.backend_name != str(backend.name):
            transpile_for_backend(job, backend)
        sim = AerSimulator.from_backend(backend, method="density_matrix")
        ro = _readout_assignment_from_backend(backend)
        for tqc in job.isa:
            order = pr._readout_qubits(tqc)
            body = _strip(tqc)
            body.save_probabilities(order, label="p")
            p = np.asarray(sim.run(body, shots=1).result().data(0)["p"], dtype=float)
            p = np.clip(p, 0.0, None)
            p = p / p.sum()
            vectors.append(_apply_assignment(p, [ro.get(q, np.eye(2)) for q in order]))
    out = []
    for p in vectors:
        out.append({format(k, f"0{n}b"): float(p[k] * shots) for k in range(2 ** n) if p[k] > 0.0})
    return out


def predict_on_backend(job: FreeTierJob, backend, shots, n_sigma=3.0, physical_qubits=None):
    """The survival analysis at ``backend``'s calibration: the shot-free
    ledger of the ISA batch (:func:`exact_counts`) through :func:`analyze`,
    with the ``survival`` row of ``vacuum.hardware.noise`` (signal, sigma at
    ``shots``, z, alive, shots for n_sigma)."""
    from vacuum.hardware import noise as nz

    if job.isa is None or job.backend_name != str(backend.name):
        transpile_for_backend(job, backend, physical_qubits)
    counts = exact_counts(job, shots, backend)
    binfo = backend_info(backend, job.physical_qubits)
    info = submission_info(job, binfo, shots, f"shot-free_{backend.name}",
                           "qiskit_aer.AerSimulator.from_backend(method='density_matrix') + analytic readout",
                           None, None, _utcnow())
    record = build_record(info, counts, {"shot_free": True})
    an = analyze(record, n_sigma=n_sigma, strict=False)
    v, s = an["E_B_info"], an["E_B_info_std_error"]
    z = v / s if s > 0 else math.inf
    an["survival"] = {"signal": v, "sigma": s, "z": z, "alive": z > n_sigma, "shots": int(shots),
                      "n_sigma": float(n_sigma),
                      "shots_for_n_sigma": nz.shots_for_n_sigma(v, s, shots, n_sigma)}
    an["record"] = record
    return an


# ---------------------------------------------------------------------------
# analysis through the standing audits
# ---------------------------------------------------------------------------


def _family_circuits(fam, counts, family):
    """[(basis dict, count dict)] of a family in batch order."""
    f = fam[family]
    out = []
    for opname, (groups, _c) in f["operators"].items():
        cs = counts[family][opname]
        if len(cs) != len(groups):
            raise ValueError(f"{family}/{opname}: {len(cs)} count dicts for {len(groups)} groups")
        for g, c in zip(groups, cs):
            out.append((dict(g.basis), c))
    return out


def _operator_from_counts(fam, counts, family, terms, shots, mitigation=None):
    """(value, standard error) of sum(terms) from a family's counts: every
    non-identity term is read from the first circuit of the family whose
    measurement basis covers it; the per-circuit value vectors give the
    exact multinomial variance (circuits independent)."""
    f = fam[family]
    measured = [int(q) for q in f["measured"]]
    circuits = _family_circuits(fam, counts, family)
    constant = sum(t.coeff.real for t in terms if t.is_identity())
    assigned: List[List[PauliTerm]] = [[] for _ in circuits]
    for t in terms:
        if t.is_identity():
            continue
        for k, (basis, _c) in enumerate(circuits):
            if all(basis.get(q, "Z") == ch for q, ch in t.paulis):
                assigned[k].append(t)
                break
        else:
            raise ValueError(f"term {t} is not measurable in family {family!r}")
    value, var = float(constant), 0.0
    for k, (basis, c) in enumerate(circuits):
        if not assigned[k]:
            continue
        g = MeasurementGroup(terms=assigned[k], basis=basis)
        m, v = group_statistics(counts_to_vector(c, len(measured)), g, measured, mitigation)
        value += m
        var += v
    return value, math.sqrt(var / float(shots))


def _closure_terms(chain, fam, site_A):
    """The operators of the closure, as PauliTerm lists: H (ground frame),
    H_rot (Alice's measurement frame, x_A -> diag(lambda), h_A's off-diagonal
    part in the sign basis dropped: zero by the measurement postulate),
    D_A, H_B (ground frame), H_B_rot, h_spectators (both frames equal)."""
    A = int(site_A)
    if chain.d != 2:
        raise NotImplementedError("the hardware closure assumes d = 2 (Alice's site is the sign qubit)")
    W = fam["meta"]["W_A"]
    lam = fam["meta"]["x_A_eigenvalues"]
    sign_qubit = fam["meta"]["sign_qubit"]
    N = chain.N
    H = vs.hamiltonian_pauli_terms(chain)
    # rotated frame
    rot: List[PauliTerm] = []
    dropped = []
    for i in range(N):
        ops = site_operators(chain.d, chain.omega[i])
        h_i = 0.5 * ops["p2"] + 0.5 * chain.K[i, i] * ops["x2"]
        if i == A:
            for t in _site_terms(W.T @ h_i @ W, chain, i):
                if any(q == sign_qubit and ch in ("X", "Y") for q, ch in t.paulis):
                    dropped.append(t)
                else:
                    rot.append(t)
        else:
            rot += _site_terms(h_i, chain, i)
    x_terms = {}
    for i in range(N):
        if i == A:
            x_terms[i] = _site_terms(np.diag(lam).astype(complex), chain, i)
        else:
            x_terms[i] = vs.site_pauli_terms(chain, i, "x")
    for i in range(N):
        for j in range(i + 1, N):
            if chain.K[i, j] != 0.0:
                for a in x_terms[i]:
                    for b in x_terms[j]:
                        rot.append(_multiply_disjoint(a, PauliTerm(b.paulis, b.coeff * chain.K[i, j])))
    spectators = []
    for i in range(N):
        if i in (A, int(fam["meta"]["site_B"])):
            continue
        ops = site_operators(chain.d, chain.omega[i])
        spectators += _site_terms(0.5 * ops["p2"] + 0.5 * chain.K[i, i] * ops["x2"], chain, i)
    return {
        "H": _merge(H), "H_rot": _merge(rot), "dropped_h_A_terms": dropped,
        "D_A": _terms_of(fam["ground"]["operators"]["D_A"]),
        "H_B": _terms_of(fam["ground"]["operators"]["H_B"]),
        "H_B_rot": _terms_of(fam["control"]["operators"]["H_B_rot"]),
        "h_spectators": _merge(spectators),
        "h_A_rot_diagonal_constant": float(sum(
            t.coeff.real for t in _site_terms(W.T @ (0.5 * site_operators(chain.d, chain.omega[A])["p2"]
                                                     + 0.5 * chain.K[A, A] * site_operators(chain.d, chain.omega[A])["x2"]) @ W,
                                              chain, A) if t.is_identity())),
    }


def _marginal(vec, measured, keep):
    """Marginal of a probability vector (bit k <- measured[k]) on ``keep``."""
    n = len(measured)
    t = np.asarray(vec, dtype=float).reshape((2,) * n)
    pos = {q: k for k, q in enumerate(measured)}
    drop = [n - 1 - pos[q] for q in measured if q not in keep]
    if drop:
        t = np.sum(t, axis=tuple(sorted(drop)))
    return t.reshape(-1)


def _no_signaling_marginals(fam, counts, chain, site_A, n_sigma):
    """Ground vs control: for every pair of circuits reading the non-Alice
    qubits in the same basis, the marginal on those qubits must agree —
    two-sample chi-square homogeneity, pass iff p >= p_min = 2 Phi(-n_sigma)."""
    A_q = set(chain.site_qubits(int(site_A)))
    measured = [int(q) for q in fam["ground"]["measured"]]
    others = [q for q in measured if q not in A_q]
    p_min = float(2.0 * _norm.sf(float(n_sigma)))
    g_circ = _family_circuits(fam, counts, "ground")
    c_circ = _family_circuits(fam, counts, "control")
    pairs = []
    for gi, (gb, gc) in enumerate(g_circ):
        for ci, (cb, cc) in enumerate(c_circ):
            if any(gb.get(q, "Z") != cb.get(q, "Z") for q in others):
                continue
            n1, n2 = float(sum(gc.values())), float(sum(cc.values()))
            p1 = _marginal(counts_to_vector(gc, len(measured)), measured, others)
            p2 = _marginal(counts_to_vector(cc, len(measured)), measured, others)
            c1, c2 = n1 * p1, n2 * p2
            pooled = (c1 + c2) / (n1 + n2)
            mask = pooled > 0
            chi2 = float(np.sum((c1[mask] - n1 * pooled[mask]) ** 2 / (n1 * pooled[mask])
                                + (c2[mask] - n2 * pooled[mask]) ** 2 / (n2 * pooled[mask])))
            dof = int(np.count_nonzero(mask)) - 1
            if dof <= 0:
                p_value, crit = 1.0, math.inf
            else:
                p_value = float(_chi2_dist.sf(chi2, dof))
                crit = float(_chi2_dist.isf(p_min, dof))
            ratio = chi2 / crit if math.isfinite(crit) and crit > 0 else 0.0
            pairs.append({"ground_circuit": gi, "control_circuit": ci,
                          "basis": {str(q): gb.get(q, "Z") for q in others}, "qubits": others,
                          "tv_distance": float(0.5 * np.sum(np.abs(p1 - p2))),
                          "chi2": chi2, "dof": dof, "p_value": p_value, "chi2_critical": crit,
                          "ratio": float(ratio), "passed": bool(ratio <= 1.0)})
    if not pairs:
        raise ValueError("no ground/control circuit pair reads the non-Alice qubits in a common basis")
    worst = max(p["ratio"] for p in pairs)
    return {"pairs": pairs, "worst_ratio": float(worst), "p_min": p_min, "n_sigma": float(n_sigma),
            "passed": bool(worst <= 1.0), "non_alice_qubits": others,
            "rule": "two-sample chi-square homogeneity of the marginal counts on the non-Alice qubits, "
                    "ground vs control; ratio = chi2 / chi2_crit(p_min) must be <= 1"}


def _signal_vs_ideal(value, sigma, ideal, n_sigma, tol_abs=1e-9):
    """One-sided consistency of the reported signal with its noiseless value.

    Every channel in the device model is unital-or-contractive on the
    reported estimator — gate depolarizing, thermal relaxation, readout
    assignment (and the tensored inverse of it, which is unbiased) — so a
    noisy run's E_B_info can sit anywhere BELOW the noiseless ideal, and the
    survival map is the statement of how far below.  It cannot sit
    *significantly above* it: a value exceeding the ideal by more than
    n_sigma standard errors is not a better device, it is a mis-shaped
    result — counts assigned to the wrong family or group, a circuit that is
    not the one analysed, a mitigation blow-up — and no verdict may be read
    from it.  The ideal is recomputed from the record's own VQE parameters
    (:func:`qet_dense`), never taken from the record's ``spec`` block, so
    editing the file's stated prediction cannot buy the check off.

    One-sided by construction: being below the ideal is the expected
    physics and is never a violation.
    """
    ideal, value = float(ideal), float(value)
    sigma = float(sigma)
    tol = float(n_sigma) * sigma if sigma > 0 else float(tol_abs)
    excess = value - ideal
    pull = excess / sigma if sigma > 0 else (0.0 if abs(excess) <= tol_abs else math.copysign(math.inf, excess))
    passed = excess <= tol
    return AuditResult(
        name="signal_vs_ideal",
        passed=bool(passed),
        worst_violation=0.0 if passed else float(excess - tol),
        details={"value": value, "std_error": sigma, "ideal": ideal, "excess": float(excess),
                 "pull": float(pull), "tol": tol, "n_sigma": float(n_sigma), "one_sided": True,
                 "ideal_source": "qet_dense on the record's own VQE parameters (not the spec block)",
                 "rule": "E_B_info - E_B_info(noiseless) <= n_sigma x standard error; noise contracts "
                         "the signal, so an excess above the ideal is a mis-shaped result, not a better device"})


def analyze(record, n_sigma=3.0, strict=False, mitigation="tensored"):
    """The audited analysis of a record (module docstring).

    Returns a dict with the headline numbers (E_A, E_B_ctrl, E_B_raw,
    E_B_info +- standard errors, mitigated and raw), the full measured
    energies of the ledger, the closure, the no-signaling test, the three
    audits (``audits``), the verdict (``verdict``) and the comparison with
    the spec's sourced-noise prediction.  ``strict=True`` raises
    :class:`AuditViolation` on the first failed audit.
    """
    validate_record(record)
    wp = record["working_point"]
    chain = vs.truncated_chain(int(wp["N"]), float(wp["m"]), int(wp["d"]), bc=wp["bc"])
    params = np.asarray(wp["vqe_params"], dtype=float)
    layers = int(wp["layers"])
    A, B = int(wp["site_A"]), int(wp["site_B"])
    theta = float(wp["theta"])
    state = vs.hea_statevector(params, chain.n_qubits, layers)
    dense = qet_dense(chain, state, A, B, theta)
    dense_consistency = max(abs(float(dense[k]) - float(wp["dense"][k])) for k in _DENSE_KEYS
                            if k in wp["dense"] and k not in ("ledger_defect", "no_signaling"))
    if dense_consistency > 1e-9:
        raise ValueError(f"the record's dense ledger differs from the one its VQE parameters give: {dense_consistency:.3e}")
    fam = qet_circuits(chain, params, layers, A, B, theta, form="deferred")
    n = chain.n_qubits
    shots = int(record["job"]["shots"])
    counts: Dict[str, Dict[str, list]] = {}
    cal = {}
    for c in record["circuits"]:
        if c["kind"] == "measurement":
            counts.setdefault(c["family"], {}).setdefault(c["operator"], []).append(c["counts"])
        else:
            cal[c["prepared"]] = c["counts"]
    mats = calibration_from_counts(cal["0" * n], cal["1" * n], n)
    led_mit = qet_from_counts(fam, counts, ideal=dense, shots=shots, calibration=mats)
    led_raw = qet_from_counts(fam, counts, ideal=dense, shots=shots, calibration=None)
    if mitigation not in ("tensored", "none"):
        raise ValueError("mitigation must be 'tensored' or 'none'")
    led = led_mit if mitigation == "tensored" else led_raw
    mit = [mats[q] for q in fam["ground"]["measured"]] if mitigation == "tensored" else None

    # -- the ledger's measured energies and its closure ----------------------
    T = _closure_terms(chain, fam, A)
    E_full = {
        "ground": _operator_from_counts(fam, counts, "ground", T["H"], shots, mit),
        "meas": _operator_from_counts(fam, counts, "control", T["H_rot"], shots, mit),
        "final": _operator_from_counts(fam, counts, "protocol", T["H_rot"], shots, mit),
        "wrong": _operator_from_counts(fam, counts, "wrong", T["H_rot"], shots, mit),
    }
    # defect = <H_rot - H_B_rot>_final - <H + D_A>_ground + <H_B_rot>_meas
    d_final = _operator_from_counts(fam, counts, "protocol", _merge(T["H_rot"] + _neg(T["H_B_rot"])), shots, mit)
    d_ground = _operator_from_counts(fam, counts, "ground", _merge(T["H"] + T["D_A"]), shots, mit)
    d_meas = _operator_from_counts(fam, counts, "control", T["H_B_rot"], shots, mit)
    defect = d_final[0] - d_ground[0] + d_meas[0]
    sigma_defect = math.sqrt(d_final[1] ** 2 + d_ground[1] ** 2 + d_meas[1] ** 2)
    spec_g = _operator_from_counts(fam, counts, "ground", T["h_spectators"], shots, mit)
    spec_f = _operator_from_counts(fam, counts, "protocol", T["h_spectators"], shots, mit)
    spectator = (spec_f[0] - spec_g[0], math.sqrt(spec_f[1] ** 2 + spec_g[1] ** 2))
    bob = (led["no_signaling_energy"], led["no_signaling_energy_std_error"] or 0.0)
    E_A, sA = led["E_A"], led["E_A_std_error"] or 0.0
    E_B, sB = led["E_B_ctrl"], led["E_B_ctrl_std_error"] or 0.0
    tol_closure = float(n_sigma) * sigma_defect
    closure_passed = abs(defect) <= tol_closure
    omega_A = float(chain.omega[A])
    ledger_audit = AuditResult(
        name="energy_ledger", passed=bool(closure_passed),
        worst_violation=0.0 if closure_passed else abs(defect) - tol_closure,
        details={
            "table": [
                {"kind": "snapshot", "label": "ground", "value": E_full["ground"][0], "std_error": E_full["ground"][1]},
                {"kind": "work", "label": "E_A = <D_A>_ground (Alice's site)", "value": E_A, "std_error": sA},
                {"kind": "snapshot", "label": "meas", "value": E_full["meas"][0], "std_error": E_full["meas"][1]},
                {"kind": "dissipation", "label": "E_B = <H_B>_meas - <H_B>_final (Bob's site)", "value": E_B, "std_error": sB},
                {"kind": "snapshot", "label": "final", "value": E_full["final"][0], "std_error": E_full["final"][1]},
            ],
            "delta_E": E_full["final"][0] - E_full["ground"][0],
            "work_in": E_A, "dissipated": E_B, "defect": defect, "sigma": sigma_defect,
            "tol": tol_closure, "n_sigma": float(n_sigma), "z": (defect / sigma_defect if sigma_defect > 0 else 0.0),
            "statistical": True,
            "spectator_residual": {"value": spectator[0], "std_error": spectator[1],
                                   "z": spectator[0] / spectator[1] if spectator[1] > 0 else 0.0,
                                   "meaning": "<h_spectator>_final - <h_spectator>_ground (untouched site)"},
            "bob_residual": {"value": bob[0], "std_error": bob[1], "z": bob[0] / bob[1] if bob[1] > 0 else 0.0,
                             "meaning": "<H_B>_meas - <H_B>_ground (no-signaling in energy)"},
            "h_A_after_measurement": {"value": omega_A, "measured": False,
                                      "meaning": "<h_A>_meas = <h_A>_final = omega_A by the measurement postulate "
                                                 "(d = 2: Alice's site is read in its sign basis; the off-diagonal "
                                                 f"part of h_A has zero expectation); {len(T['dropped_h_A_terms'])} term(s) dropped"},
            "first_snapshot": "ground", "last_snapshot": "final",
            "identity": "(E_final - E_ground) - (E_A - E_B) == 0 within n_sigma x propagated shot error",
            "mitigation": mitigation,
        })
    ns = _no_signaling_marginals(fam, counts, chain, A, n_sigma)
    ns_audit = AuditResult(name="no_signaling_marginals", passed=bool(ns["passed"]),
                           worst_violation=max(0.0, ns["worst_ratio"] - 1.0), details=ns)
    margin_sigma = math.sqrt(sA ** 2 + sB ** 2)
    hw = qet_audits({"E_A": E_A, "E_B": E_B, "ledger_defect": abs(defect), "no_signaling": ns["worst_ratio"]},
                    tol=float(n_sigma) * margin_sigma, ledger_tol=tol_closure, signaling_tol=1.0, strict=False)
    hw.details["tolerances"] = {"E_A_ge_E_B": f"{n_sigma} sigma of E_A - E_B ({margin_sigma:.3e})",
                                "ledger_closure": f"{n_sigma} sigma of the closure defect ({sigma_defect:.3e})",
                                "no_signaling": "chi2 / chi2_crit of the worst ground-vs-control marginal pair <= 1"}
    v, s = float(led["E_B_info"]), float(led["E_B_info_std_error"] or 0.0)
    ideal_audit = _signal_vs_ideal(v, s, dense["E_B_info"], n_sigma)
    audits = {"energy_ledger": _audit_dict(ledger_audit), "no_signaling_marginals": _audit_dict(ns_audit),
              "signal_vs_ideal": _audit_dict(ideal_audit), "hardware_qet": _audit_dict(hw)}
    all_passed = bool(ledger_audit.passed and ns_audit.passed and ideal_audit.passed and hw.passed)
    audits["all_passed"] = all_passed

    # -- the number tested ---------------------------------------------------
    z = v / s if s > 0 else (math.inf if v > 0 else -math.inf)
    spec = record["spec"]
    pred = float(spec["predicted_value_at_sourced_noise"])
    pred_sigma = float(spec["predicted_std_error_at_sourced_noise"]) * math.sqrt(
        float(spec["predicted_std_error_shots"]) / shots)
    b = record["backend"]
    where = f"{b['name']} ({'simulated: ' + b['kind'] if b['simulated'] else 'device'})"
    positive = z >= float(n_sigma)
    verdict_ = {
        "number_tested": spec["number_tested"],
        "E_B_info": v, "std_error": s, "z": z, "n_sigma": float(n_sigma),
        "positive_at_n_sigma": bool(positive),
        "audits_passed": all_passed,
        "claim_holds": bool(positive and all_passed),
        "mitigation": mitigation,
        "raw": {"E_B_info": float(led_raw["E_B_info"]), "std_error": float(led_raw["E_B_info_std_error"] or 0.0)},
        "backend": b["name"], "simulated": bool(b["simulated"]), "shots": shots,
        "statement": (f"E_B_info = {v:+.4f} +- {s:.4f} (z = {z:.2f}) on {where} at {shots} shots/circuit: "
                      + ("E_B_info > 0 at " if positive else "NOT positive at ") + f"{n_sigma:g} sigma; "
                      + ("audits passed" if all_passed
                         else "AUDIT FAILURE in " + ", ".join(k for k in ("energy_ledger", "no_signaling_marginals",
                                                                          "signal_vs_ideal", "hardware_qet")
                                                              if not audits[k]["passed"])
                              + " (an event, not a number)")),
    }
    analysis = {
        "format": RECORD_FORMAT, "record_sha256": record["sha256"], "backend": b["name"],
        "simulated": bool(b["simulated"]), "shots": shots, "n_sigma": float(n_sigma), "mitigation": mitigation,
        "E_A": E_A, "E_A_std_error": sA,
        "E_B_ctrl": E_B, "E_B_ctrl_std_error": sB,
        "E_B_raw": float(led["E_B_raw"]), "E_B_raw_std_error": float(led["E_B_raw_std_error"] or 0.0),
        "E_B_info": v, "E_B_info_std_error": s, "E_B_cut": float(led["E_B_cut"]),
        "HB_ground": float(led["HB_ground"]), "HB_meas": float(led["HB_meas"]),
        "HB_final": float(led["HB_final"]), "HB_wrong": float(led["HB_wrong"]),
        "no_signaling_energy": bob[0], "no_signaling_energy_std_error": bob[1],
        "z": {k: (float(led[k]) / float(led[k + "_std_error"]) if led[k + "_std_error"] else None)
              for k in ("E_A", "E_B_ctrl", "E_B_raw", "E_B_info")},
        "ledger": {"mitigated": _ledger_numbers(led_mit), "raw": _ledger_numbers(led_raw)},
        "readout_calibration": [np.asarray(m).tolist() for m in mats],
        "full_energies": {k: {"value": val, "std_error": sig} for k, (val, sig) in E_full.items()},
        "closure": {"defect": defect, "sigma": sigma_defect, "tol": tol_closure, "passed": bool(closure_passed)},
        "no_signaling": ns,
        "audits": audits,
        "dense": {k: float(dense[k]) for k in _DENSE_KEYS},
        "dense_consistency": float(dense_consistency),
        "pull_vs_dense": {k: ((float(led[k]) - float(dense[dk])) / float(led[k + "_std_error"]) if led[k + "_std_error"] else None)
                          for k, dk in (("E_A", "E_A"), ("E_B_ctrl", "E_B"), ("E_B_info", "E_B_info"), ("E_B_raw", "E_B_raw"))},
        "prediction": {"sourced_noise_value": pred, "sourced_noise_std_error_at_shots": pred_sigma,
                       "pull_vs_sourced_noise": (v - pred) / s if s > 0 else None,
                       "ideal_value": float(spec["ideal_value"]),
                       "pull_vs_ideal": (v - float(spec["ideal_value"])) / s if s > 0 else None,
                       "pull_vs_ideal_audited": ideal_audit.details["pull"],
                       "ideal_value_recomputed": float(dense["E_B_info"])},
        "verdict": verdict_,
    }
    if strict and not all_passed:
        for res in (ledger_audit, ns_audit, ideal_audit, hw):
            if not res.passed:
                raise AuditViolation(res)
    return _jsonable(analysis)


def _ledger_numbers(led):
    keys = ("E_A", "E_A_std_error", "HB_ground", "HB_meas", "HB_final", "HB_wrong", "E_B", "E_B_ctrl",
            "E_B_ctrl_std_error", "E_B_raw", "E_B_raw_std_error", "E_B_wrong", "E_B_info", "E_B_info_std_error",
            "E_B_cut", "no_signaling_energy", "no_signaling_energy_std_error")
    return {k: (None if led.get(k) is None else float(led[k])) for k in keys if k in led}


def verdict(analysis):
    """The verdict dict of an :func:`analyze` result (E_B_info > 0 at
    n_sigma AND every audit passed)."""
    return dict(analysis["verdict"])


# ---------------------------------------------------------------------------
# one-call driver, service helpers
# ---------------------------------------------------------------------------


def run_batch(backend, shots, sampler=None, seed=None, params=None, physical_qubits=None, out_dir=None,
              tags=None, n_sigma=3.0, strict=False, job=None, save_sidecar=True):
    """Build -> transpile -> submit -> wait -> collect -> persist -> analyze.
    Returns dict(job, submission, record, analysis, path, submission_path)."""
    if job is None:
        job = build_free_tier_job(params, backend=backend, physical_qubits=physical_qubits)
    elif job.isa is None or job.backend_name != str(backend.name):
        transpile_for_backend(job, backend, physical_qubits)
    sub = submit(job, backend, shots, sampler=sampler, seed=seed, tags=tags)
    sub_path = None
    if out_dir is not None and save_sidecar:
        sub_path = save_submission(sub, default_record_path(out_dir, sub, ".submitted.json"))
    record = collect(sub.runtime_job, sub)
    path = None
    if out_dir is not None:
        path = save_record(record, default_record_path(out_dir, record))
    analysis = analyze(record, n_sigma=n_sigma, strict=strict)
    return {"job": job, "submission": sub, "record": record, "analysis": analysis, "path": path,
            "submission_path": sub_path}


def fake_backend(name="FakeSherbrooke"):
    """A ``qiskit_ibm_runtime.fake_provider`` backend (a real, dated
    calibration snapshot; runs locally through Aer)."""
    from qiskit_ibm_runtime import fake_provider

    try:
        cls = getattr(fake_provider, name)
    except AttributeError as exc:
        raise ValueError(f"no fake backend {name!r} in qiskit_ibm_runtime.fake_provider") from exc
    return cls()


def runtime_service(token=None, instance=None, channel="ibm_quantum_platform"):
    """``QiskitRuntimeService`` from ``IBM_QUANTUM_TOKEN`` (and optional
    ``IBM_QUANTUM_INSTANCE``) or the saved account.  Never called on import."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = token or os.environ.get("IBM_QUANTUM_TOKEN")
    instance = instance or os.environ.get("IBM_QUANTUM_INSTANCE")
    if token:
        return QiskitRuntimeService(channel=channel, token=token, instance=instance)
    if instance:
        return QiskitRuntimeService(instance=instance)
    return QiskitRuntimeService()


def nz_shots(signal, sigma, shots_used, n_sigma=3.0):
    """Shots per circuit for ``signal`` to reach ``n_sigma`` (alias of
    :func:`vacuum.hardware.noise.shots_for_n_sigma`)."""
    from vacuum.hardware import noise as nz

    return nz.shots_for_n_sigma(signal, sigma, shots_used, n_sigma)


def least_busy_backend(service, min_qubits=None):
    """The least-busy operational real device with at least ``min_qubits``
    (default: the spec's 3)."""
    n = FREE_TIER_SPEC["qubits"] if min_qubits is None else int(min_qubits)
    return service.least_busy(min_num_qubits=n, operational=True, simulator=False)
