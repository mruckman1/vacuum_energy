"""Ikeda's IBM-hardware QET protocol, reproduced exactly — dense, no qiskit.

Transcribed from

    K. Ikeda, "Demonstration of Quantum Energy Teleportation on
    Superconducting Quantum Hardware", Phys. Rev. Applied 20, 024051 (2023)
    [arXiv:2301.02666v5]: Eqs. (1)-(14), Fig. 1, Appendix A Eqs. (A1)-(A6),
    Appendix B,

and — for the two details the paper's text delegates to its public code
(the readout estimators and the order of Bob's two controlled rotations) —
from the author's MIT-licensed repository cited by the paper as Ref. [16],
github.com/IKEDAKAZUKI/Quantum-Energy-Teleportation (``QET.py``:
``inject_energy_circuit`` / ``inject_energy_val``, ``QET_circuit_XX`` /
``QET_energy_XX``, ``QET_circuit_Z`` / ``QET_energy_Z``; the 2023 notebook
``Quantum_Energy_Teleportation.ipynb`` cells 19 and 23 carry the same
circuits, the first with ``c_if`` feed-forward, the second deferred).

The model is Hotta's minimal QET model, i.e. exactly :mod:`vacuum.qet.hotta`
with (A, B) = (qubit 0, qubit 1) and |0> = |+> (Z = +1), s = sqrt(h^2+k^2):

    H_tot = H_0 + H_1 + V                                     [Eq. (1)]
    H_n   = h Z_n + h^2/s,  n = 0, 1                          [Eq. (2)]
    V     = 2k X_0 X_1 + 2k^2/s                               [Eq. (3)]
    |g>   = (1/sqrt2) sqrt(1 - h/s) |00> - (1/sqrt2) sqrt(1 + h/s) |11>
                                                              [Eq. (4)]
    <g|H_tot|g> = <g|H_0|g> = <g|H_1|g> = <g|V|g> = 0         [Eq. (5)]

The protocol as it was run on the devices (Fig. 1):

  (A) ground-state preparation  |g> = CNOT (R_Y(2 theta) (x) I) |00>,
      theta = -arccos( (1/sqrt2) sqrt(1 - h/s) )              [Eq. (7)]
      (checked through CNOT(R_Y(2 theta) (x) I)|00> = cos(theta)|00>
      + sin(theta)|11>, Eq. (A4)), with R_Y(a) = exp(-i a Y/2) [Eq. (A2)].
      Alice's projective X_0 measurement P_0(mu) = (1 + mu X_0)/2 [Eq. (8)]
      is realised as an H on qubit 0 followed by a Z-basis readout,
      mu = (-1)^{b_0}; her deposited energy is <E_0> = h^2/s   [Eq. (9)].
  (B) Bob's conditional rotation
      U_1(mu) = cos(phi) I - i mu sin(phi) Y_1 = R_Y(2 mu phi) [Eq. (10)]
      cos 2phi = (h^2 + 2k^2)/D,  sin 2phi = h k/D,
      D = sqrt((h^2 + 2k^2)^2 + h^2 k^2)                      [Eqs. (11)-(12)]
      rho_QET = sum_mu U_1(mu) P_0(mu) |g><g| P_0(mu) U_1(mu)^dag  [Eq. (6)]
      <E_1> = Tr[rho_QET (H_1 + V)] = Tr[rho_QET H_tot] - <E_0>   [Eq. (13)]
            = -(1/s) [ h k sin 2phi - (h^2 + 2k^2)(1 - cos 2phi) ] [Eq. (14)]
      and Bob's device extracts E_B = -<E_1> (> 0).
  (C) the deferred-measurement form actually executed on hardware
      (Fig. 1(C), right; Sec. II.D): controlled-U_1(-1) on b_0 = 1
      (Lambda(U), Eq. (A3)) and anti-controlled U_1(+1) on b_0 = 0
      ((X (x) I) Lambda(U) (X (x) I), Appendix A.1), all readouts at the
      end. The two commute (Appendix B), so their order is immaterial; the
      order coded here is the author's (``QET_circuit_XX``).

Readout estimators (Appendix A.1 and the author's code):

    <Z_1>     = sum_b (1 - 2 b_1) counts_b / n_shots              [Eq. (A5)]
    <X_0 X_1> = sum_b (1 - 2 b_0)(1 - 2 b_1) counts_b / n_shots   [Eq. (A6)]
    <H_1> = h^2/s + h <Z_1>,  <V> = 2k^2/s + 2k <X_0 X_1>          (Eqs. (2)-(3))
    <E_0> = h^2/s + h <X_0>   (``inject_energy_val``: the X_0 term is
            identically zero on |g> and records the device bias on hardware)
    <E_1> = <H_1> + <V>                                       (Table I caption)

Because V and H_1 do not commute, <H_1> and <V> come from two separate
circuits (Sec. II.C): the "H_1" circuit reads Z_1 directly, the "V" circuit
adds an H on qubit 1 before the readout (Fig. 1(B), right; Eq. (A6)).

Bit strings follow the qiskit/OpenQASM convention "c1 c0" — the RIGHTMOST
character is classical bit 0 = qubit 0 — which is also the x-axis labelling
of Fig. 2(B). Dense states use the :mod:`vacuum.qet.hotta` basis
{|00>, |01>, |10>, |11>} with qubit 0 (Alice) the FIRST tensor factor. All
functions are pure over their arguments; the only I/O is the experiments
loader (:mod:`vacuum.experiments_io`), the ONE sanctioned source of
published numbers.
"""

from __future__ import annotations

import itertools
import math
import os
from typing import Any, Mapping

import numpy as np

from vacuum.audits import AuditResult, AuditViolation
from vacuum.qet.hotta import (
    bob_rotation,
    e_a_closed,
    e_b_closed,
    e_b_optimal,
    ground_state,
    hamiltonian,
    hamiltonian_terms,
    measurement_projector,
    theta_optimal,
    trace_distance,
)

__all__ = [
    "IKEDA_FIG2_FILE",
    "IKEDA_TABLE2_FILE",
    "OBSERVABLES",
    "FORMS",
    "BITSTRINGS",
    "PAPER_PAIRS",
    "ikeda_params",
    "prep_angle",
    "bob_angle",
    "e0_closed",
    "h1_closed",
    "v_closed",
    "e1_closed",
    "ikeda_circuit_spec",
    "terminal_measure_index",
    "spec_prefix",
    "gate_matrix",
    "simulate_spec",
    "counts_to_probabilities",
    "expectation_from_counts",
    "ikeda_ibm",
    "ikeda_no_signaling",
    "ikeda_sweep",
]

#: The shipped parameter files (experiments/ibm_qet/).
IKEDA_FIG2_FILE = "ibm_qet/ikeda_2023_fig2"
IKEDA_TABLE2_FILE = "ibm_qet/ikeda_2023_table2"

#: The three readout circuits of the paper (Sec. II.B-II.C, Fig. 1).
OBSERVABLES = ("E0", "H1", "V")
#: Feed-forward (Fig. 1(B), ``c_if``) or deferred measurement (Fig. 1(C)).
FORMS = ("deferred", "feedforward")
#: qiskit-ordered two-bit strings "c1c0".
BITSTRINGS = ("00", "01", "10", "11")
#: The (h, k) pairs of the paper's Tables I, II and IV.
PAPER_PAIRS = ((1.0, 0.1), (1.0, 0.2), (1.0, 0.5), (1.0, 1.0), (1.5, 1.0))

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)          # Eq. (A1)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)       # Eq. (A1)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)         # Eq. (A1)
_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)  # Eq. (A1)
_P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
_P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)


def _check_hk(h, k):
    h = float(h)
    k = float(k)
    if not (h > 0.0 and k > 0.0):
        raise ValueError(f"the minimal model needs h > 0 and k > 0, got h={h}, k={k}")
    return h, k


def _ry(alpha):
    """R_Y(alpha) = exp(-i alpha Y / 2)  [Eq. (A2)]."""
    c = math.cos(alpha / 2.0)
    s = math.sin(alpha / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


def ikeda_params(params=None):
    """Flatten a parameter source into the dict the protocol functions consume.

    Parameters
    ----------
    params : None | str | os.PathLike | Experiment | dict
        ``None`` loads the shipped ``experiments/ibm_qet/ikeda_2023_fig2.json``
        through :func:`vacuum.experiments_io.load_experiment`; a name or path
        loads that file; an already-loaded ``Experiment`` is flattened; a
        dict with at least ``'h'`` and ``'k'`` is passed through (copied).

    Returns
    -------
    dict
        ``'h'``, ``'k'`` (validated floats), every quantity of the file under
        its own name (verbatim value), ``'<name>_error'`` where the file
        carries an uncertainty, ``'_units'`` (name -> units string), the
        provenance strings (``'source'``, ``'location'``, ``'retrieved'``,
        ``'name'``, ``'path'``) and any top-level extras (``'device'``).
        No unit conversion: every quantity of the Ikeda files is already
        dimensionless, a count, or a device time/frequency used only by the
        qiskit noise model, which converts explicitly and cites each use.
    """
    if isinstance(params, Mapping) and "h" in params and "k" in params:
        out = dict(params)
        out["h"], out["k"] = _check_hk(out["h"], out["k"])
        return out
    from vacuum import experiments_io as xio

    if params is None:
        params = IKEDA_FIG2_FILE
    if isinstance(params, xio.Experiment):
        exp = params
    elif isinstance(params, (str, os.PathLike)):
        exp = xio.load_experiment(params)
    else:
        raise TypeError(
            "params must be None, a file name/path, an Experiment, or a dict "
            f"with 'h' and 'k'; got {type(params).__name__}"
        )
    out: dict[str, Any] = {
        "name": exp.name,
        "path": exp.path,
        "source": exp.source,
        "location": exp.location,
        "retrieved": exp.retrieved,
        "_units": {},
    }
    for qname, q in exp.quantities.items():
        out[qname] = q.value
        out["_units"][qname] = q.units
        if q.error is not None:
            out[qname + "_error"] = q.error
    for key, val in exp.extras.items():
        out[key] = val
    if "h" not in out or "k" not in out:
        raise ValueError(f"experiments file {exp.path} carries no 'h'/'k' quantities")
    out["h"], out["k"] = _check_hk(out["h"], out["k"])
    return out


# ---------------------------------------------------------------------------
# angles and closed forms
# ---------------------------------------------------------------------------


def prep_angle(h, k):
    """Ground-state preparation angle theta of Eq. (7).

        theta = -arccos( (1/sqrt2) sqrt(1 - h/sqrt(h^2 + k^2)) )  in (-pi/2, 0),

    so that CNOT (R_Y(2 theta) (x) I)|00> = cos(theta)|00> + sin(theta)|11>
    [Eq. (A4)] reproduces Eq. (4) with sin(theta) = -(1/sqrt2) sqrt(1 + h/s).
    The author's code writes the same angle as
    -arcsin((1/sqrt2) sqrt(1 + h/s)); the two agree because cos(theta) >= 0.
    """
    h, k = _check_hk(h, k)
    s = math.hypot(h, k)
    return -math.acos(math.sqrt((1.0 - h / s) / 2.0))


def bob_angle(h, k):
    """Bob's rotation half-angle phi of Eqs. (11)-(12).

        cos 2phi = (h^2 + 2k^2)/D,  sin 2phi = h k/D,
        D = sqrt((h^2 + 2k^2)^2 + h^2 k^2),

    i.e. phi = (1/2) atan2(h k, h^2 + 2k^2) in (0, pi/4) — the same angle as
    :func:`vacuum.qet.hotta.theta_optimal` (review Eqs. (9)-(10)).
    """
    h, k = _check_hk(h, k)
    return 0.5 * math.atan2(h * k, h * h + 2.0 * k * k)


def e0_closed(h, k):
    """Alice's deposited energy <E_0> = h^2/sqrt(h^2 + k^2)  [Eq. (9)]."""
    h, k = _check_hk(h, k)
    return h * h / math.hypot(h, k)


def _phi_or(h, k, phi):
    return bob_angle(h, k) if phi is None else float(phi)


def h1_closed(h, k, phi=None):
    """<H_1> = Tr[rho_QET H_1] for Bob's angle phi (default Eqs. (11)-(12)).

    Derived here from Eqs. (2), (4), (6), (10) — not a numbered equation of
    the paper: with |g> = a|00> + b|11>, a^2 - b^2 = -h/s, 2ab = -k/s, each
    branch mu leaves Bob in a|0> + mu b|1> and U_1(mu) = R_Y(2 mu phi) gives
    <Z_1> = -(h/s) cos 2phi + (k/s) sin 2phi, so

        <H_1> = (1/s) [ h^2 (1 - cos 2phi) + h k sin 2phi ].

    Its sum with :func:`v_closed` is Eq. (14) (checked in the tests).
    """
    h, k = _check_hk(h, k)
    s = math.hypot(h, k)
    t = 2.0 * _phi_or(h, k, phi)
    return (h * h * (1.0 - math.cos(t)) + h * k * math.sin(t)) / s


def v_closed(h, k, phi=None):
    """<V> = Tr[rho_QET V] for Bob's angle phi (default Eqs. (11)-(12)).

    Derived as in :func:`h1_closed`: <X_0 X_1> = -(h/s) sin 2phi
    - (k/s) cos 2phi, so

        <V> = (1/s) [ 2k^2 (1 - cos 2phi) - 2 h k sin 2phi ].
    """
    h, k = _check_hk(h, k)
    s = math.hypot(h, k)
    t = 2.0 * _phi_or(h, k, phi)
    return (2.0 * k * k * (1.0 - math.cos(t)) - 2.0 * h * k * math.sin(t)) / s


def e1_closed(h, k, phi=None):
    """<E_1> = <H_1> + <V> of Eq. (14):

        <E_1> = -(1/s) [ h k sin 2phi - (h^2 + 2k^2)(1 - cos 2phi) ],

    which is minus :func:`vacuum.qet.hotta.e_b_closed` (review Eq. (14)).
    """
    h, k = _check_hk(h, k)
    s = math.hypot(h, k)
    t = 2.0 * _phi_or(h, k, phi)
    return -(h * k * math.sin(t) - (h * h + 2.0 * k * k) * (1.0 - math.cos(t))) / s


# ---------------------------------------------------------------------------
# the gate list (backend-agnostic)
# ---------------------------------------------------------------------------


def _gate(name, qubits, params=None, cite="", **extra):
    g = {"name": name, "qubits": list(qubits), "cite": cite}
    if params is not None:
        g["params"] = [float(p) for p in params]
    g.update(extra)
    return g


def ikeda_circuit_spec(params=None, observable="V", form="deferred"):
    """Backend-agnostic gate list for one of the paper's readout circuits.

    Parameters
    ----------
    params : see :func:`ikeda_params`.
    observable : 'E0' | 'H1' | 'V'
        Which circuit: Alice's deposit readout (Fig. 1(A) with both qubits
        read out — the author's ``inject_energy_circuit``), Bob's Z_1 readout
        (Fig. 1(B) left / ``QET_circuit_Z``), or the X_0 X_1 readout
        (Fig. 1(B) right / ``QET_circuit_XX``).
    form : 'deferred' | 'feedforward'
        'deferred' (default) is the circuit run on the devices (Fig. 1(C),
        right): Bob's rotation as a controlled/anti-controlled pair and all
        readouts at the end. 'feedforward' is the textbook protocol of
        Fig. 1(B): a mid-circuit measurement of qubit 0 into classical bit 0
        and a rotation conditioned on that bit (the notebook's ``c_if``).

    Returns
    -------
    dict
        ``n_qubits``, ``n_clbits`` (both 2), ``h``, ``k``, ``theta``, ``phi``,
        ``observable``, ``form``, ``bit_order``, ``source`` and ``gates``: a
        list of dicts ``{name, qubits, [params], [clbits], [condition], cite}``
        with OpenQASM gate names (``ry``, ``cx``, ``h``, ``x``, ``cry``,
        ``measure``); ``condition = [clbit, value]`` means "apply only if
        classical bit ``clbit`` reads ``value``". Angles are in radians.
    """
    p = ikeda_params(params)
    h, k = p["h"], p["k"]
    if observable not in OBSERVABLES:
        raise ValueError(f"observable must be one of {OBSERVABLES}, got {observable!r}")
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}, got {form!r}")
    theta = prep_angle(h, k)
    phi = bob_angle(h, k)

    gates = [
        _gate("ry", [0], [2.0 * theta],
              "Eq. (7): R_Y(2 theta) on qubit 0, ground-state preparation, Fig. 1(A)"),
        _gate("cx", [0, 1],
              cite="Eq. (7): CNOT, control qubit 0 -> target qubit 1, Fig. 1(A)"),
        _gate("h", [0],
              cite="Eq. (8): X_0 -> Z_0 readout basis change for Alice's "
                   "projective measurement, Fig. 1(A)"),
    ]
    if observable == "E0":
        # Sec. II.B: "<E_0> can be calculated with the output bit-strings
        # 00, 01, 10, 11"; author's inject_energy_circuit reads both qubits.
        gates.append(_gate("measure", [0, 1], clbits=[0, 1],
                           cite="Sec. II.B / inject_energy_circuit: read out "
                                "both qubits, <E_0> = h^2/s + h <X_0>"))
    else:
        if form == "deferred":
            gates += [
                _gate("cry", [0, 1], [-2.0 * phi],
                      "Eq. (10) + Eq. (A3): controlled U_1(-1) = R_Y(-2 phi) on "
                      "qubit 1 when b_0 = 1 (mu = -1), Fig. 1(C) right"),
                _gate("x", [0], cite="Appendix A.1 anti-control identity "
                                     "(X (x) I) Lambda(U) (X (x) I), Fig. 1(C)"),
                _gate("cry", [0, 1], [2.0 * phi],
                      "Eq. (10) + Eq. (A3): anti-controlled U_1(+1) = R_Y(+2 phi) "
                      "on qubit 1 when b_0 = 0 (mu = +1), Fig. 1(C) right"),
                _gate("x", [0], cite="Appendix A.1 anti-control identity (closing X)"),
            ]
        else:
            gates += [
                _gate("measure", [0], clbits=[0],
                      cite="Eq. (8), Fig. 1(A)/(B): Alice's mid-circuit X_0 "
                           "readout, mu = (-1)^{b_0}"),
                _gate("ry", [1], [2.0 * phi], condition=[0, 0],
                      cite="Eq. (10), Fig. 1(B): U_1(+1) = R_Y(2 phi) if b_0 = 0"),
                _gate("ry", [1], [-2.0 * phi], condition=[0, 1],
                      cite="Eq. (10), Fig. 1(B): U_1(-1) = R_Y(-2 phi) if b_0 = 1"),
            ]
        if observable == "V":
            gates.append(_gate("h", [1],
                               cite="Fig. 1(B) right / Eq. (A6): X_1 -> Z_1 "
                                    "readout basis change for <X_0 X_1>"))
        if form == "deferred":
            gates.append(_gate("measure", [0, 1], clbits=[0, 1],
                               cite="Fig. 1(C): both readouts at the end "
                                    "(deferred measurement, Sec. II.D)"))
        else:
            gates.append(_gate("measure", [1], clbits=[1],
                               cite="Fig. 1(B): Bob's readout (Eq. (A5) for "
                                    "H_1, Eq. (A6) for V)"))
    return {
        "n_qubits": 2,
        "n_clbits": 2,
        "h": h,
        "k": k,
        "theta": theta,
        "phi": phi,
        "observable": observable,
        "form": form,
        "gates": gates,
        "bit_order": "c1c0 (qiskit/OpenQASM): rightmost character is clbit 0 = qubit 0",
        "source": "K. Ikeda, Phys. Rev. Applied 20, 024051 (2023), Fig. 1, "
                  "Eqs. (7)-(12), (A2)-(A6); Ref. [16] code",
    }


def terminal_measure_index(spec):
    """Index of the first gate of the trailing block of ``measure`` gates.

    Everything from this index on is readout; everything before it is the
    protocol proper (including a mid-circuit measurement in the feed-forward
    form). Equals ``len(gates)`` when the spec ends with a non-measure gate.
    """
    gates = spec["gates"]
    i = len(gates)
    while i > 0 and gates[i - 1]["name"] == "measure":
        i -= 1
    return i


def spec_prefix(spec, n_gates):
    """A copy of ``spec`` truncated to its first ``n_gates`` gates."""
    out = dict(spec)
    out["gates"] = [dict(g) for g in spec["gates"][:n_gates]]
    return out


# ---------------------------------------------------------------------------
# dense simulation of a gate list
# ---------------------------------------------------------------------------


def _embed1(U, q):
    """2x2 ``U`` on qubit ``q`` of two, qubit 0 the first tensor factor."""
    return np.kron(U, _I2) if q == 0 else np.kron(_I2, U)


def _controlled(U, control, target):
    """Lambda(U) = |0><0| (x) I + |1><1| (x) U  [Eq. (A3)], control/target as given."""
    if control == target or control not in (0, 1) or target not in (0, 1):
        raise ValueError(f"bad control/target ({control}, {target})")
    if control == 0:
        return np.kron(_P0, _I2) + np.kron(_P1, U)
    return np.kron(_I2, _P0) + np.kron(U, _P1)


def gate_matrix(gate):
    """The (4, 4) unitary of one non-measurement gate of a spec."""
    name = gate["name"]
    qs = gate["qubits"]
    if name == "ry":
        return _embed1(_ry(gate["params"][0]), qs[0])
    if name == "h":
        return _embed1(_H, qs[0])
    if name == "x":
        return _embed1(_X, qs[0])
    if name == "cx":
        return _controlled(_X, qs[0], qs[1])
    if name == "cry":
        return _controlled(_ry(gate["params"][0]), qs[0], qs[1])
    raise ValueError(f"unknown gate {name!r}")


def _measure_branches(branches, qubit, clbit):
    """Split every branch by a Z-basis readout of ``qubit`` into ``clbit``."""
    out = []
    for psi, cbits in branches:
        for b, proj in ((0, _P0), (1, _P1)):
            new = _embed1(proj, qubit) @ psi
            w = float(np.real(np.vdot(new, new)))
            if w == 0.0:
                continue
            cb = list(cbits)
            cb[clbit] = b
            out.append((new, tuple(cb)))
    return out


def simulate_spec(spec):
    """Exact dense run of a gate list (unnormalised branch bookkeeping).

    Mid-circuit measurements split the run into branches carrying their
    classical bits; conditioned gates act only on the branches whose bit
    matches; the terminal readout block is applied last. Branch weights are
    the squared norms of the unnormalised vectors, so ensemble quantities
    are plain sums over branches.

    Returns
    -------
    dict
        ``'rho_pre_readout'`` : (4, 4) ensemble state just before the
            terminal readout block (the state a statevector backend
            reports), dense hotta.py basis, qubit 0 first;
        ``'rho'`` : (4, 4) ensemble state after the readout (dephased);
        ``'probabilities'`` : {"c1c0": p} over the four readout outcomes
            (bits never measured are shown as '-');
        ``'branches'`` : list of (state, classical-bit tuple) after all gates;
        ``'n_branches_pre_readout'`` : how many classical branches the
            protocol proper produced (1 for the deferred form).
    """
    n_cl = int(spec["n_clbits"])
    psi0 = np.zeros(4, dtype=complex)
    psi0[0] = 1.0                                # |00>
    branches = [(psi0, tuple([None] * n_cl))]
    i_term = terminal_measure_index(spec)
    rho_pre = None
    n_pre = None
    for i, gate in enumerate(spec["gates"]):
        if i == i_term:
            rho_pre = sum(np.outer(psi, psi.conj()) for psi, _ in branches)
            n_pre = len(branches)
        if gate["name"] == "measure":
            for q, c in zip(gate["qubits"], gate["clbits"]):
                branches = _measure_branches(branches, q, c)
            continue
        U = gate_matrix(gate)
        cond = gate.get("condition")
        new = []
        for psi, cbits in branches:
            if cond is not None:
                cbit, val = int(cond[0]), int(cond[1])
                if cbits[cbit] is None:
                    raise ValueError(
                        f"gate {gate['name']} conditioned on unmeasured clbit {cbit}"
                    )
                if cbits[cbit] != val:
                    new.append((psi, cbits))
                    continue
            new.append((U @ psi, cbits))
        branches = new
    if rho_pre is None:                          # spec ends with a non-measure gate
        rho_pre = sum(np.outer(psi, psi.conj()) for psi, _ in branches)
        n_pre = len(branches)
    rho = sum(np.outer(psi, psi.conj()) for psi, _ in branches)
    probs: dict[str, float] = {}
    for psi, cbits in branches:
        key = "".join("-" if b is None else str(b) for b in reversed(cbits))
        probs[key] = probs.get(key, 0.0) + float(np.real(np.vdot(psi, psi)))
    return {
        "rho_pre_readout": rho_pre,
        "rho": rho,
        "probabilities": probs,
        "branches": branches,
        "n_branches_pre_readout": n_pre,
    }


# ---------------------------------------------------------------------------
# the paper's estimators
# ---------------------------------------------------------------------------


def counts_to_probabilities(counts):
    """Normalise a counts (or probabilities) dict; keys reduced to 'c1c0'.

    Returns (probabilities, total). Keys may carry register spaces
    (qiskit's ``'01 1'`` style); only the last two characters are used.
    """
    clean: dict[str, float] = {}
    for key, val in counts.items():
        k2 = str(key).replace(" ", "")[-2:]
        clean[k2] = clean.get(k2, 0.0) + float(val)
    total = sum(clean.values())
    if total <= 0.0:
        raise ValueError("counts sum to zero")
    return {k: v / total for k, v in clean.items()}, total


def expectation_from_counts(counts, h, k, observable, shots=None):
    """The paper's readout estimator for one circuit, from counts.

    Parameters
    ----------
    counts : {"c1c0": count or probability}
        qiskit-ordered bit strings (rightmost = qubit 0).
    h, k : float
    observable : 'E0' | 'H1' | 'V'
        'H1': <H_1> = h^2/s + h <Z_1>,     <Z_1> per Eq. (A5) from bit c1;
        'V' : <V>   = 2k^2/s + 2k <X_0 X_1>, per Eq. (A6) from (-1)^{c0+c1};
        'E0': <E_0> = h^2/s + h <X_0>,     <X_0> from bit c0 after Alice's H
              (author's ``inject_energy_val``).
    shots : int, optional
        Number of shots behind ``counts`` when they are probabilities (for
        the standard error); inferred from integer counts otherwise.

    Returns
    -------
    dict
        ``value``; ``pauli_mean`` (the estimated <Z_1>, <X_0 X_1> or <X_0>);
        ``constant`` (the offset of Eqs. (2)-(3)); ``coefficient`` (h or 2k);
        ``std_error`` = std of the per-shot value / sqrt(n) — the author's
        error bar — or None when n is unknown; ``n``.
    """
    h, k = _check_hk(h, k)
    s = math.hypot(h, k)
    probs, total = counts_to_probabilities(counts)
    if observable == "H1":
        coef, const = h, h * h / s

        def eig(key):                       # Eq. (A5): (1 - 2 b_1)
            return 1.0 - 2.0 * int(key[0])
    elif observable == "V":
        coef, const = 2.0 * k, 2.0 * k * k / s

        def eig(key):                       # Eq. (A6): (1 - 2 b_0)(1 - 2 b_1)
            return (1.0 - 2.0 * int(key[1])) * (1.0 - 2.0 * int(key[0]))
    elif observable == "E0":
        coef, const = h, h * h / s

        def eig(key):                       # inject_energy_val: (-1)^{b_0}
            return 1.0 - 2.0 * int(key[1])
    else:
        raise ValueError(f"observable must be one of {OBSERVABLES}, got {observable!r}")
    mean = 0.0
    mean2 = 0.0
    for key, p in probs.items():
        if "-" in key:
            raise ValueError(f"bit string {key!r} has an unmeasured bit")
        e = eig(key)
        mean += p * e
        mean2 += p * e * e
    var = max(mean2 - mean * mean, 0.0) * coef * coef
    n = None
    if shots is not None:
        n = float(shots)
    elif all(float(v) == int(float(v)) for v in counts.values()) and total > 1.0:
        n = total
    return {
        "value": const + coef * mean,
        "pauli_mean": mean,
        "constant": const,
        "coefficient": coef,
        "std_error": (math.sqrt(var / n) if n else None),
        "n": n,
    }


# ---------------------------------------------------------------------------
# the protocol ledger
# ---------------------------------------------------------------------------


def _ptrace_a(rho):
    r = np.asarray(rho, dtype=complex).reshape(2, 2, 2, 2)
    return np.einsum("ijik->jk", r)


def _energy(rho, H):
    return float(np.real(np.trace(rho @ H)))


def _dephase_x0(rho):
    """sum_mu P_0(mu) rho P_0(mu): drop coherences between Alice's X outcomes."""
    out = np.zeros_like(rho)
    for mu in (+1, -1):
        P = measurement_projector(mu)
        out = out + P @ rho @ P
    return out


def ikeda_ibm(params=None, form="deferred"):
    """Run the paper's three readout circuits exactly and return the ledger.

    Every energy is obtained the way the hardware obtains it — from the
    readout distribution of the corresponding circuit through the paper's
    estimators (:func:`expectation_from_counts`) — and then cross-checked in
    difference form against the physical states of Eq. (6) built with
    :mod:`vacuum.qet.hotta`'s projectors and rotation.

    Parameters
    ----------
    params : see :func:`ikeda_params` (default: the shipped Fig. 2 file,
        (h, k) = (1, 1) on ibm_cairo).
    form : 'deferred' | 'feedforward'
        Which circuit family (see :func:`ikeda_circuit_spec`); the two give
        identical readout statistics (Appendix B), which the tests assert.

    Returns
    -------
    dict
        ``E_A``  : <E_0> from the E0 circuit (Alice's deposit)            [Eq. (9)]
        ``E_B``  : -<E_1>, Bob's extracted energy                          [Eq. (13)]
        ``E0``, ``H1``, ``V``, ``E1`` : the paper's four reported quantities
        ``X0``, ``Z1``, ``X0X1`` : the estimated Pauli means
        ``E_ground``, ``E_meas``, ``E_final`` : Tr[rho H_tot] of |g>, of the
            post-measurement ensemble, and of rho_QET (Eq. (6))
        ``E_A_difference_form`` = E_meas - E_ground,
        ``E_B_difference_form`` = E_meas - E_final   (must equal E_A, E_B)
        ``closed_form`` : Eq. (9), Eq. (14) at phi, review Eq. (11), and the
            derived <H_1>, <V> forms; ``residuals`` : |ledger - closed form|
        ``paper`` : the file's printed analytical values when present
        ``circuit_state_check`` : max |X_0-dephased physical circuit state
            - rho_QET| — the circuit implements Eq. (6) exactly
        ``ground_state_check`` : max |circuit-prepared |g> - Eq. (4)|
        ``probabilities`` : {observable: {"c1c0": p}} (Fig. 2(B) ideal bars)
        ``specs``, ``rho_ground``, ``rho_QET``, ``theta``, ``phi``, ``h``,
        ``k``, ``form``, ``source``, ``location``, ``name``.
    """
    p = ikeda_params(params)
    h, k = p["h"], p["k"]
    theta = prep_angle(h, k)
    phi = bob_angle(h, k)
    specs = {obs: ikeda_circuit_spec(p, obs, form) for obs in OBSERVABLES}
    sims = {obs: simulate_spec(specs[obs]) for obs in OBSERVABLES}
    probs = {obs: dict(sims[obs]["probabilities"]) for obs in OBSERVABLES}
    est = {obs: expectation_from_counts(probs[obs], h, k, obs) for obs in OBSERVABLES}
    E0 = est["E0"]["value"]
    H1 = est["H1"]["value"]
    V = est["V"]["value"]
    E1 = H1 + V                                   # Eq. (13) / Table I caption

    # --- physical-state bookkeeping (difference form), hotta.py conventions
    H = hamiltonian(h, k)
    H_A, H_B, V_op = hamiltonian_terms(h, k)
    prep = spec_prefix(specs["E0"], 2)            # R_Y(2 theta), CNOT  [Eq. (7)]
    g_circ = simulate_spec(prep)["branches"][0][0]
    rho_g = np.outer(g_circ, g_circ.conj())
    E_ground = _energy(rho_g, H)
    rho_M = _dephase_x0(rho_g)                    # sum_mu P_0(mu)|g><g|P_0(mu)
    E_meas = _energy(rho_M, H)
    rho_QET = np.zeros((4, 4), dtype=complex)     # Eq. (6)
    for mu in (+1, -1):
        P = measurement_projector(mu)
        U = bob_rotation(mu, phi)                 # U_1(mu) = R_Y(2 mu phi), Eq. (10)
        rho_QET = rho_QET + U @ P @ rho_g @ P @ U.conj().T
    E_final = _energy(rho_QET, H)

    # the circuit's own pre-readout state, taken back to the physical frame
    # (undo Alice's readout H on qubit 0) and X_0-dephased, must be rho_QET
    Hq0 = _embed1(_H, 0)
    rho_circ = sims["H1"]["rho_pre_readout"]
    rho_phys = _dephase_x0(Hq0 @ rho_circ @ Hq0)
    circuit_state_check = float(np.max(np.abs(rho_phys - rho_QET)))
    ground_state_check = float(np.max(np.abs(g_circ - ground_state(h, k))))

    closed = {
        "E_A": e_a_closed(h, k),                          # Eq. (9) / review Eq. (8)
        "E_B": e_b_optimal(h, k),                         # review Eq. (11)
        "E_B_at_phi": e_b_closed(h, k, phi),              # review Eq. (14)
        "H1": h1_closed(h, k, phi),
        "V": v_closed(h, k, phi),
        "E1": e1_closed(h, k, phi),                       # Eq. (14)
        "theta_star": theta_optimal(h, k),                # review Eqs. (9)-(10)
    }
    ledger = {
        "E_A": E0,
        "E_B": -E1,
        "E0": E0,
        "H1": H1,
        "V": V,
        "E1": E1,
        "X0": est["E0"]["pauli_mean"],
        "Z1": est["H1"]["pauli_mean"],
        "X0X1": est["V"]["pauli_mean"],
        "E_ground": E_ground,
        "E_meas": E_meas,
        "E_final": E_final,
        "E_A_difference_form": E_meas - E_ground,
        "E_B_difference_form": E_meas - E_final,
        "H1_from_state": _energy(rho_QET, H_B),
        "V_from_state": _energy(rho_QET, V_op),
        "closed_form": closed,
        "residuals": {
            "E_A": abs(E0 - closed["E_A"]),
            "E_B": abs(-E1 - closed["E_B"]),
            "H1": abs(H1 - closed["H1"]),
            "V": abs(V - closed["V"]),
            "E1": abs(E1 - closed["E1"]),
            "phi_vs_theta_star": abs(phi - closed["theta_star"]),
        },
        "circuit_state_check": circuit_state_check,
        "ground_state_check": ground_state_check,
        "probabilities": probs,
        "specs": specs,
        "rho_ground": rho_g,
        "rho_QET": rho_QET,
        "theta": theta,
        "phi": phi,
        "h": h,
        "k": k,
        "form": form,
        "source": p.get("source"),
        "location": p.get("location"),
        "name": p.get("name"),
    }
    paper = {}
    for key, name in (("E0", "analytical_E0"), ("H1", "analytical_H1"),
                      ("V", "analytical_V"), ("E1", "analytical_E1")):
        if name in p:
            paper[key] = float(p[name])
    if paper:
        ledger["paper"] = paper
        ledger["paper_residuals"] = {key: abs(ledger[key] - val) for key, val in paper.items()}
    return ledger


def ikeda_no_signaling(params=None):
    """Bob's reduced state before the classical bit, with vs without Alice's
    measurement, both taken from the gate list (not the closed-form |g>).

    Returns dict with ``trace_distance`` (no-signaling demands 0 to roundoff),
    ``rho_B_unmeasured`` (after R_Y, CNOT) and ``rho_B_measured`` (after the
    readout H and the mid-circuit measurement of qubit 0).
    """
    spec = ikeda_circuit_spec(params, "H1", "feedforward")
    names = [g["name"] for g in spec["gates"]]
    i_meas = names.index("measure")               # Alice's mid-circuit readout
    rho_no = simulate_spec(spec_prefix(spec, 2))["rho"]
    rho_yes = simulate_spec(spec_prefix(spec, i_meas + 1))["rho"]
    # take the measured branches back to the physical frame (undo the H)
    Hq0 = _embed1(_H, 0)
    rho_yes = Hq0 @ rho_yes @ Hq0
    rb_no = _ptrace_a(rho_no)
    rb_yes = _ptrace_a(rho_yes)
    return {
        "trace_distance": trace_distance(rb_no, rb_yes),
        "rho_B_unmeasured": rb_no,
        "rho_B_measured": rb_yes,
    }


def ikeda_sweep(pairs=PAPER_PAIRS, tol=1e-12, strict=True, form="deferred"):
    """Standing audit: E_A >= E_B on every (h, k) configuration of the circuit.

    Runs :func:`ikeda_ibm` at each pair and demands E_A - E_B >= -tol; a
    violation is an anomaly-protocol event (raises :class:`AuditViolation`
    in strict mode). Mirrors :func:`vacuum.qet.hotta.hotta_sweep`.

    Returns
    -------
    AuditResult
        ``details['rows']``: per pair h, k, E_A, E_B, margin, and the
        closed-form residuals; ``min_margin``, ``argmin``, ``tol``.
    """
    rows = []
    for h, k in pairs:
        led = ikeda_ibm({"h": h, "k": k}, form=form)
        rows.append({
            "h": float(h), "k": float(k), "E_A": led["E_A"], "E_B": led["E_B"],
            "margin": led["E_A"] - led["E_B"], "residuals": led["residuals"],
        })
    margins = np.array([r["margin"] for r in rows])
    i_min = int(np.argmin(margins))
    min_margin = float(margins[i_min])
    passed = min_margin >= -tol
    result = AuditResult(
        name="ikeda_sweep",
        passed=passed,
        worst_violation=max(0.0, -min_margin),
        details={
            "rows": rows,
            "min_margin": min_margin,
            "argmin": (rows[i_min]["h"], rows[i_min]["k"]),
            "tol": float(tol),
            "form": form,
        },
    )
    if strict and not passed:
        raise AuditViolation(result)
    return result
