"""The two minimal protocols on the qubit-encoded vacuum (Layer 7, L6).

Both run twice — as exact dense linear algebra on the truncated model
(core-only, the arbiter) and as qiskit/Aer circuits (the ``.[ibm]`` extra;
noiseless Aer must reproduce the dense path to 1e-10, and the same circuits
run under the noise models of :mod:`vacuum.hardware.noise`).  Every reported
run carries the standing audits: energy-ledger closure (locality of the
bookkeeping), E_A >= E_B, and no-signaling.

(a) Quantum energy teleportation on the encoded field
-----------------------------------------------------
The field-theoretic QET protocol of Hotta, Phys. Rev. D 78, 045006 (2008)
[arXiv:0803.2272] and its harmonic-chain form, Nambu & Hotta, Phys. Rev.
A 82, 042329 (2010) [arXiv:1007.2234] — Alice measures the field amplitude
at her site, Bob applies a displacement conditioned on the classical
outcome — transcribed to the truncated chain of
:mod:`vacuum.hardware.vacuum_sim`:

  I.   Alice measures the SIGN of x_A (projectors P_+- onto the positive /
       negative eigenspaces of the truncated x_A; for d = 2 exactly Hotta's
       sigma_x measurement), injecting E_A = <D_A>_0 with
       D_A = sum_mu P_mu h_A P_mu - h_A a single-site operator (the bond
       terms commute with P_mu, so only the on-site energy is deposited).
  II.  The bit mu = +-1 travels to Bob.
  III. Bob applies the conditional displacement
       U_B(mu) = exp(-i mu theta p_B)  (a real orthogonal d x d matrix;
       for d = 2, R_Y(2 mu theta sqrt(omega_B/2))) and extracts
       E_B = <H>_meas - <H>_final = <H_B^touch>_meas - <H_B^touch>_final,
       H_B^touch the terms of H_d acting on Bob's site.

For N = 2, d = 2 this is Hotta's minimal model with h = omega/2,
k = 1/(4 omega) (:mod:`vacuum.qet.hotta`), whose closed forms
(review arXiv:1101.3954 Eqs. (8), (11), (14)) the dense path reproduces to
1e-12 (tests) — the anchor that ties this package to Layer 3.

Circuit realisation (Ikeda, PRApplied 20, 024051 (2023), Fig. 1(C)
deferred form, generalised): the VQE preparation, W_A^dag on Alice's site
(the eigenbasis of x_A, ordered so that the site's top qubit reads 0 for
x_A > 0, i.e. mu = +1), Bob's U_B(+theta) anti-controlled and U_B(-theta)
controlled on that qubit, then a Pauli-group readout of H_B^touch written
in Alice's rotated frame (x_A -> its eigenvalue diagonal, so every measured
term is diagonal on Alice's site and the deferred and feed-forward forms
agree).  The families are

  'ground'   : preparation only; reads D_A (-> E_A) and H_B^touch (-> the
               reference <H_B>_0),
  'control'  : preparation + W_A^dag, Bob idle (theta = 0): <H_B>_meas — on
               hardware the no-signaling reference (must equal <H_B>_0),
  'protocol' : the full circuit: <H_B>_final,
  'wrong'    : the full circuit with Bob's controls inverted (he acts on
               the wrong bit): <H_B>_wrong — the same gates, no usable
               information; averaged 50/50 with 'protocol' it is Hotta's
               cut channel (arXiv:1101.3954 Sec. 3),

giving three signals of decreasing noise sensitivity:
  E_B_raw  = <H_B>_0(ideal) - <H_B>_final      (Ikeda's raw definition),
  E_B_ctrl = <H_B>_meas - <H_B>_final          (Bob-idle control; the
             protocol circuit carries Bob's extra gates, so their noise
             heating competes with E_B),
  E_B_info = <H_B>_wrong - <H_B>_final         (equal-depth control: the
             energy the classical bit is worth; ideal value E_B - E_B_wrong
             = 2 (E_B - E_B_cut) > 0).
E_B_info > 0 is the "information does work" statement; E_B_ctrl > 0 the
teleportation statement proper.  The survival map reports all three.

(b) Minimal entanglement harvesting
-----------------------------------
Two ancilla qubits are Unruh–DeWitt detectors, H_d = (Omega/2)(1 - Z_d)
(ground state |0>), monopole-coupled to the encoded field at sites
j_A, j_B:  H_int(t) = lam chi(t) sum_d X_d x_{j_d}  — the two-level UDW
model of Pozas-Kerstjens & Martin-Martinez, PRD 92, 064042 (2015) with
sigma_x as the Schrodinger-picture monopole, on a smooth switching
profile chi from :mod:`vacuum.detectors.switching` (cos^2 or Gaussian;
sharp switching is banned there and here).  Time evolution is a
second-order (Strang) product formula over the qubit-wise commuting
groups of H(t_k) = H_field + H_det + H_int(t_k), t_k the step midpoints:
n_steps steps of e^{-i H_1 dt/2} ... e^{-i H_G dt} ... e^{-i H_1 dt/2},
each group exponential the exact product of its commuting term
exponentials.  The dense mirror (:func:`harvest_dense`, ``method='trotter'``)
applies the identical operator sequence in numpy; ``method='exact'`` is
the converged midpoint-rule evolution of the full H(t) and gives the
Trotter error.  The harvested log-negativity of the ancilla pair,
E_N = ln ||rho^{T_B}||_1 = ln(1 - 2 lambda_min) (nats), uses that a
two-qubit partial transpose has at most one negative eigenvalue
(Sanpera, Tarrach, Vidal, PRA 58, 826 (1998)); on Aer it comes from
two-qubit Pauli tomography (nine settings, linear inversion).

Gaussian comparison: :func:`harvest_gaussian_reference` runs the SAME K,
sites, gap and switching through ``vacuum.detectors.run_harvesting`` with
oscillator detectors at lam_osc = lam sqrt(2 Omega) — the leading-order
two-level <-> oscillator map derived and gate-tested in
tests/test_gate_crossvalidation.py (MAP-1).  Agreement can only be
qualitative: the field truncation vanishes as d grows, the detector
truncation does not, and both models are compared at leading order in lam.

Conventions: hbar = 1; energies in the chain's lattice units; negativities
in nats; qubit/dense ordering as in :mod:`vacuum.hardware.vacuum_sim`
(site 0 least significant, ancillas above the field register).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix

from vacuum.audits import AuditResult, AuditViolation
from vacuum.detectors.switching import switching as make_switching
from vacuum.hardware.vacuum_sim import (
    PAULIS,
    PauliTerm,
    TruncatedChain,
    _merge,
    _multiply_disjoint,
    _qiskit,
    _site_terms,
    commuting_groups,
    embed_site,
    hamiltonian_dense,
    hamiltonian_pauli_terms,
    hea_circuit,
    pauli_label,
    site_operators,
    site_pauli_terms,
    touched_hamiltonian_dense,
    touched_pauli_terms,
)
from vacuum.qet.ikeda_qiskit import run_on_backend

__all__ = [
    "IBM_BASIS_GATES",
    "linear_coupling_map",
    "alice_projectors",
    "alice_basis_change",
    "bob_displacement",
    "reduced_site_state",
    "trace_distance",
    "qet_dense",
    "qet_optimize_theta",
    "qet_audits",
    "MeasurementGroup",
    "measurement_groups",
    "group_statistics",
    "expectation_from_groups",
    "qet_circuits",
    "qet_exact_aer",
    "qet_sampled",
    "qet_from_counts",
    "readout_calibration_circuits",
    "calibration_from_counts",
    "apply_tensored_mitigation",
    "HarvestSpec",
    "harvest_spec",
    "harvest_terms",
    "trotter_schedule",
    "harvest_dense",
    "negativity_two_qubit",
    "partial_transpose_B",
    "harvest_gaussian_reference",
    "harvest_circuit",
    "harvest_coupling_map",
    "tomography_settings",
    "tomography_circuits",
    "rho_from_pauli_expectations",
    "harvest_exact_aer",
    "harvest_sampled",
    "harvest_from_counts",
    "run_on_backend",
]

#: ibm_cairo-style native basis (as in vacuum.qet.ikeda_qiskit).
IBM_BASIS_GATES = ("rz", "sx", "x", "cx")


def linear_coupling_map(n_qubits, order=None):
    """Bidirectional line through the register: (q_k, q_{k+1}) for the given
    ``order`` (default 0..n-1).  A device path; routing SWAPs are the
    transpiler's business."""
    order = list(range(int(n_qubits))) if order is None else [int(q) for q in order]
    edges = []
    for a, b in zip(order[:-1], order[1:]):
        edges.append([a, b])
        edges.append([b, a])
    return edges


# ---------------------------------------------------------------------------
# (a) QET — dense path
# ---------------------------------------------------------------------------


def alice_basis_change(chain: TruncatedChain, site):
    """(W, eigenvalues): W's columns are the eigenvectors of the truncated
    x_site, ordered positive eigenvalues first (descending), then negative
    (ascending magnitude), so that for the site's Fock index k the top bit
    of k is 0 exactly for x > 0.  Returns real orthogonal W (d, d) and the
    eigenvalues in that order.  Requires an even d (no zero eigenvalue)."""
    d = chain.d
    if d % 2:
        raise ValueError("the sign measurement needs an even d (x_d has a zero eigenvalue for odd d)")
    ops = site_operators(d, chain.omega[int(site)])
    w, U = np.linalg.eigh(ops["x"].real)
    pos = np.argsort(-w)[: d // 2]          # positive, largest first
    neg = np.argsort(w)[: d // 2][::-1]     # negative, smallest magnitude first
    order = np.concatenate([pos, neg])
    W = U[:, order]
    for k in range(d):                      # deterministic column signs
        j = int(np.argmax(np.abs(W[:, k])))
        if W[j, k] < 0:
            W[:, k] = -W[:, k]
    return W, w[order]


def alice_projectors(chain: TruncatedChain, site):
    """{+1: P_+, -1: P_-} — dense projectors onto x_site > 0 / < 0 (embedded)."""
    W, lam = alice_basis_change(chain, site)
    d = chain.d
    Pp = W[:, : d // 2] @ W[:, : d // 2].T
    Pm = W[:, d // 2:] @ W[:, d // 2:].T
    return {+1: embed_site(Pp, site, chain.N, d), -1: embed_site(Pm, site, chain.N, d)}


def bob_displacement_site(chain: TruncatedChain, site, mu, theta):
    """The (d, d) real orthogonal U_B(mu) = exp(-i mu theta p_site)."""
    from scipy.linalg import expm

    ops = site_operators(chain.d, chain.omega[int(site)])
    G = -1.0j * float(mu) * float(theta) * ops["p"]     # real antisymmetric
    U = expm(G)
    return U.real if np.max(np.abs(U.imag)) < 1e-13 else U


def bob_displacement(chain: TruncatedChain, site, mu, theta):
    """U_B(mu) embedded in the d^N space."""
    return embed_site(bob_displacement_site(chain, site, mu, theta), site, chain.N, chain.d)


def reduced_site_state(chain: TruncatedChain, rho, site):
    """Partial trace onto one site (d x d) of a dense state/density matrix."""
    N, d = chain.N, chain.d
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim == 1:
        rho = np.outer(rho, rho.conj())
    s = int(site)
    # index = n_0 + d n_1 + ... ; reshape to (site N-1, ..., site 0)
    t = rho.reshape((d,) * N + (d,) * N)
    ax = N - 1 - s
    keep = [ax]
    out = t
    # trace out all axes except `ax` (ket) and its bra partner
    for a in reversed(range(N)):
        if a == ax:
            continue
        out = np.trace(out, axis1=a, axis2=a + out.ndim // 2)
    return out


def trace_distance(rho, sigma):
    diff = np.asarray(rho, dtype=complex) - np.asarray(sigma, dtype=complex)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(0.5 * (diff + diff.conj().T)))))


def _energy(rho, H):
    return float(np.real(np.trace(rho @ H)))


def qet_dense(chain: TruncatedChain, psi0, site_A, site_B, theta, channel_open=True):
    """The QET ledger on the truncated chain, exact dense linear algebra.

    Parameters
    ----------
    chain : :class:`TruncatedChain`
    psi0 : (dim,) initial (ground) state — the ED state or a VQE state.
    site_A, site_B : Alice's and Bob's sites (distinct).
    theta : Bob's displacement amplitude.
    channel_open : False cuts the classical channel (Bob uses a fair coin).

    Returns
    -------
    dict : E_ground, E_A, E_B, E_meas, E_final, E_A_touch / E_B_touch
        (the same numbers from the touched operators, hardware style),
        HB_ground / HB_meas / HB_final / HB_wrong (<H_B^touch> at the
        stages; 'wrong' = Bob acting on the inverted bit), E_B_raw
        (= HB_ground - HB_final), E_B_wrong, E_B_info (= HB_wrong -
        HB_final, the equal-depth information signal), E_B_cut (Hotta's
        cut channel = the fair-coin average), ledger_defect (|(E_final -
        E_ground) - (E_A - E_B)| plus the locality residuals), outcomes
        {mu: p}, no_signaling (trace distance of Bob's reduced state before
        vs after Alice's measurement), rho_meas, rho_final, theta, sites.
    """
    A, B = int(site_A), int(site_B)
    if A == B:
        raise ValueError("Alice and Bob need distinct sites")
    H = hamiltonian_dense(chain)
    HA = touched_hamiltonian_dense(chain, A)
    HB = touched_hamiltonian_dense(chain, B)
    hA = embed_site(site_operators(chain.d, chain.omega[A])["h_site"], A, chain.N, chain.d)
    psi0 = np.asarray(psi0, dtype=complex)
    rho0 = np.outer(psi0, psi0.conj())
    E_ground = _energy(rho0, H)
    P = alice_projectors(chain, A)
    D_A = sum(P[mu] @ hA @ P[mu] for mu in P) - hA

    branches = {mu: P[mu] @ rho0 @ P[mu] for mu in (+1, -1)}
    outcomes = {mu: float(np.real(np.trace(branches[mu]))) for mu in branches}
    rho_meas = branches[+1] + branches[-1]
    E_meas = _energy(rho_meas, H)

    rho_final = np.zeros_like(rho_meas)
    if channel_open:
        for mu in (+1, -1):
            U = bob_displacement(chain, B, mu, theta)
            rho_final += U @ branches[mu] @ U.conj().T
    else:
        for mu, mu_p in itertools.product((+1, -1), repeat=2):
            U = bob_displacement(chain, B, mu_p, theta)
            rho_final += 0.5 * (U @ branches[mu] @ U.conj().T)
    E_final = _energy(rho_final, H)
    # the equal-depth control: Bob acts on the INVERTED bit (a fair-coin
    # average of this and the protocol is Hotta's cut channel)
    rho_wrong = np.zeros_like(rho_meas)
    for mu in (+1, -1):
        U = bob_displacement(chain, B, -mu, theta)
        rho_wrong += U @ branches[mu] @ U.conj().T
    E_final_wrong = _energy(rho_wrong, H)

    E_A = E_meas - E_ground
    E_B = E_meas - E_final
    E_A_touch = _energy(rho0, D_A)
    HB_ground, HB_meas, HB_final = _energy(rho0, HB), _energy(rho_meas, HB), _energy(rho_final, HB)
    HA_ground, HA_meas = _energy(rho0, HA), _energy(rho_meas, HA)
    E_B_touch = HB_meas - HB_final
    defect = max(
        abs((E_final - E_ground) - (E_A - E_B)),
        abs(E_A - E_A_touch),
        abs(E_A - (HA_meas - HA_ground)),
        abs(E_B - E_B_touch),
    )
    rhoB_before = reduced_site_state(chain, rho0, B)
    rhoB_after = reduced_site_state(chain, rho_meas, B)
    HB_wrong = _energy(rho_wrong, HB)
    return {
        "E_ground": E_ground, "E_A": E_A, "E_B": E_B, "E_meas": E_meas, "E_final": E_final,
        "E_A_touch": E_A_touch, "E_B_touch": E_B_touch,
        "HB_ground": HB_ground, "HB_meas": HB_meas, "HB_final": HB_final, "HB_wrong": HB_wrong,
        "E_B_raw": HB_ground - HB_final,
        "E_B_wrong": E_meas - E_final_wrong,
        "E_B_info": E_final_wrong - E_final,
        "E_B_cut": E_meas - 0.5 * (E_final + E_final_wrong),
        "ledger_defect": defect,
        "outcomes": outcomes,
        "no_signaling": trace_distance(rhoB_before, rhoB_after),
        "rho_meas": rho_meas, "rho_final": rho_final,
        "theta": float(theta), "site_A": A, "site_B": B, "channel_open": bool(channel_open),
    }


def qet_optimize_theta(chain, psi0, site_A, site_B, theta_max=None, xatol=1e-10):
    """Bounded scalar search of theta maximising E_B (never a closed form).
    Default bound: |theta| <= pi / sqrt(2 omega_B) (one full R_Y turn at d=2)."""
    if theta_max is None:
        theta_max = math.pi / math.sqrt(2.0 * chain.omega[int(site_B)])

    def neg(th):
        return -qet_dense(chain, psi0, site_A, site_B, th)["E_B"]

    best = None
    for lo, hi in ((0.0, theta_max), (-theta_max, 0.0)):
        res = minimize_scalar(neg, bounds=(lo, hi), method="bounded", options={"xatol": xatol})
        if best is None or res.fun < best.fun:
            best = res
    return float(best.x), -float(best.fun)


def qet_audits(ledger, tol=1e-9, ledger_tol=1e-9, signaling_tol=1e-10, strict=True):
    """Standing audits on a QET ledger (dense or exact-Aer): ledger closure,
    E_A >= E_B, no-signaling.  Returns an :class:`AuditResult`; strict mode
    raises :class:`AuditViolation`."""
    margin = ledger["E_A"] - ledger["E_B"]
    checks = {
        "ledger_closure": (ledger["ledger_defect"], ledger_tol),
        "E_A_ge_E_B": (max(0.0, -margin), tol),
        "no_signaling": (ledger["no_signaling"], signaling_tol),
    }
    worst = max(v / t for v, t in checks.values())
    passed = all(v <= t for v, t in checks.values())
    result = AuditResult(
        name="hardware_qet",
        passed=passed,
        worst_violation=0.0 if passed else worst,
        details={"checks": {k: {"value": v, "tol": t} for k, (v, t) in checks.items()},
                 "margin": margin, "E_A": ledger["E_A"], "E_B": ledger["E_B"]},
    )
    if strict and not passed:
        raise AuditViolation(result)
    return result


# ---------------------------------------------------------------------------
# Pauli-group measurement (shared by both protocols)
# ---------------------------------------------------------------------------


@dataclass
class MeasurementGroup:
    """A qubit-wise commuting set of Pauli terms read out in one basis.

    ``basis`` maps every involved qubit to the single-qubit basis ('X', 'Y'
    or 'Z') it is measured in; the per-shot value of the group is
    sum_t c_t prod_{q in t} (1 - 2 b_q).
    """

    terms: List[PauliTerm]
    basis: Dict[int, str]

    @property
    def qubits(self):
        return sorted(self.basis)


def measurement_groups(terms):
    """(groups, constant): greedy qubit-wise commuting partition of ``terms``
    (:func:`vacuum.hardware.vacuum_sim.commuting_groups`) plus the identity
    coefficient.  Coefficients must be real (Hermitian operator)."""
    constant = 0.0
    for t in terms:
        if abs(t.coeff.imag) > 1e-12:
            raise ValueError(f"non-Hermitian term {t}")
        if t.is_identity():
            constant += t.coeff.real
    groups = []
    for g in commuting_groups(terms):
        basis = {}
        for t in g:
            for qb, ch in t.paulis:
                basis[qb] = ch
        groups.append(MeasurementGroup(terms=list(g), basis=basis))
    return groups, float(constant)


def group_value_vector(group: MeasurementGroup, measured_qubits):
    """v[idx] = per-shot value of the group for outcome index idx = sum_k
    b_{measured_qubits[k]} 2^k (the classical register order)."""
    measured = [int(q) for q in measured_qubits]
    pos = {q: k for k, q in enumerate(measured)}
    n = len(measured)
    idx = np.arange(2 ** n)
    v = np.zeros(2 ** n)
    for t in group.terms:
        sign = np.ones(2 ** n)
        for qb, _ in t.paulis:
            if qb not in pos:
                raise ValueError(f"qubit {qb} of term {t} is not measured")
            bit = (idx >> pos[qb]) & 1
            sign = sign * (1.0 - 2.0 * bit)
        v += t.coeff.real * sign
    return v


def _mitigated_values(v, mats):
    """w = (A^{-1})^T v per bit: the per-shot value vector of the tensored-
    mitigated estimator (mean_p w = mean_{A^{-1} p} v)."""
    n = int(round(math.log2(v.size)))
    t = np.asarray(v, dtype=float).reshape((2,) * n)
    for k, A in enumerate(mats):
        MinvT = np.linalg.inv(np.asarray(A, dtype=float)).T
        ax = n - 1 - k
        t = np.moveaxis(np.tensordot(MinvT, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def group_statistics(probs, group: MeasurementGroup, measured_qubits, mitigation=None):
    """(mean, single-shot variance) of the group value under ``probs`` (a
    length-2^n probability vector over the measured register).  With
    ``mitigation`` (per-bit assignment matrices) the statistics are those of
    the tensored-mitigated estimator — the mean is the unmitigated mean of
    (A^{-1})^T v under the raw distribution, and the variance grows by the
    inverse (1 - 2p) factors, which is the price of mitigation."""
    v = group_value_vector(group, measured_qubits)
    if mitigation is not None:
        v = _mitigated_values(v, mitigation)
    p = np.asarray(probs, dtype=float)
    mean = float(p @ v)
    var = float(p @ (v * v) - mean * mean)
    return mean, max(var, 0.0)


def expectation_from_groups(prob_vectors, groups, constant, measured_qubits, shots=None,
                            mitigation=None):
    """Operator expectation from one probability vector per group.

    Returns dict ``value``, ``variance_single_shot`` (sum over groups),
    ``std_error`` (sqrt(var/shots) if shots given), ``group_means``.
    """
    means = []
    var = 0.0
    for p, g in zip(prob_vectors, groups):
        m, v = group_statistics(p, g, measured_qubits, mitigation)
        means.append(m)
        var += v
    value = float(constant + sum(means))
    return {
        "value": value,
        "variance_single_shot": var,
        "std_error": (math.sqrt(var / float(shots)) if shots else None),
        "group_means": means,
    }


def counts_to_vector(counts, n_bits):
    """{bitstring: count} (qiskit order, rightmost = clbit 0) -> normalised
    probability vector indexed by sum_k b_k 2^k."""
    v = np.zeros(2 ** int(n_bits))
    for key, c in counts.items():
        # qiskit prints the LAST-added register leftmost; the readout
        # register is always the last one added here (run_on_backend has
        # already stripped register spaces), so take the leading n_bits
        k = str(key).replace(" ", "")[:n_bits]
        v[int(k, 2)] += float(c)
    tot = v.sum()
    if tot <= 0:
        raise ValueError("counts sum to zero")
    return v / tot


def apply_readout_vector(probs, flips):
    """Push a probability vector through independent symmetric per-bit
    readout flips ``flips[k]`` (classical-bit order)."""
    p = np.asarray(probs, dtype=float)
    n = int(round(math.log2(p.size)))
    t = p.reshape((2,) * n)
    for k, f in enumerate(flips):
        f = float(f)
        if f == 0.0:
            continue
        M = np.array([[1.0 - f, f], [f, 1.0 - f]])
        ax = n - 1 - k                      # bit k is axis n-1-k (C order)
        t = np.moveaxis(np.tensordot(M, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


# ---------------------------------------------------------------------------
# circuits (qiskit) — shared helpers
# ---------------------------------------------------------------------------


def _prep_gate(chain: TruncatedChain, params, layers):
    """The bound VQE ansatz as a gate on the field qubits."""
    Q = _qiskit()
    qc, th = hea_circuit(chain.n_qubits, layers, name="vqe_prep")
    bound = qc.assign_parameters({th[k]: float(params[k]) for k in range(len(th))})
    return bound.to_gate(label="prep")


def _basis_change(qc, basis, qubits_map=None):
    """Append the readout basis changes for a group: X -> H, Y -> S^dag H."""
    for qb, ch in basis.items():
        if ch == "X":
            qc.h(qb)
        elif ch == "Y":
            qc.sdg(qb)
            qc.h(qb)


def measurement_circuits(body, groups, measured_qubits, name=None):
    """One circuit per group: ``body`` + basis changes + measurement of
    ``measured_qubits`` into a classical register of the same length (bit k
    <- measured_qubits[k])."""
    Q = _qiskit()
    out = []
    measured = [int(q) for q in measured_qubits]
    for i, g in enumerate(groups):
        qc = body.copy(name=f"{name or body.name}_g{i}")
        target = Q.ClassicalRegister(len(measured), f"m{len(qc.cregs)}")
        qc.add_register(target)
        _basis_change(qc, g.basis)
        qc.barrier()
        qc.measure(measured, [target[k] for k in range(len(measured))])
        out.append(qc)
    return out


def transpile_native(circuit, coupling_map, optimization_level=1, seed_transpiler=0):
    """Transpile to the ibm native basis on the given coupling map (identity
    initial layout so the noise model's per-qubit rows land on the intended
    qubits)."""
    Q = _qiskit()
    return Q.transpile(
        circuit,
        basis_gates=list(IBM_BASIS_GATES),
        coupling_map=coupling_map,
        initial_layout=list(range(circuit.num_qubits)),
        optimization_level=int(optimization_level),
        seed_transpiler=seed_transpiler,
    )


def _readout_qubits(tqc):
    """Physical qubits read into the LAST classical register of a transpiled
    measurement circuit, in bit order (robust to routing permutations)."""
    target = tqc.cregs[-1]
    pairs = []
    for inst in tqc.data:
        if inst.operation.name != "measure":
            continue
        cb = inst.clbits[0]
        for reg, idx in tqc.find_bit(cb).registers:
            if reg.name == target.name:      # the transpiler copies registers
                pairs.append((idx, tqc.find_bit(inst.qubits[0]).index))
    pairs.sort()
    return [q for _, q in pairs]


def _exact_probability_vectors(circuits, noise_model, coupling_map, optimization_level=1,
                               seed_transpiler=0):
    """Shot-free outcome distributions (before readout flips) of measurement
    circuits from Aer's density-matrix method.  Each circuit is transpiled
    to the native basis WITH its measurements (so routing permutations are
    followed), the terminal measurements are then replaced by
    ``save_probabilities`` on the physical qubits they read, in classical
    bit order.  Returns (vectors, transpiled circuits)."""
    Q = _qiskit()
    sim = Q.AerSimulator(method="density_matrix", noise_model=noise_model)
    vectors = []
    tqcs = []
    for qc in circuits:
        tqc = transpile_native(qc, coupling_map, optimization_level, seed_transpiler)
        order = _readout_qubits(tqc)
        body = tqc.copy()
        body.data = [inst for inst in tqc.data if inst.operation.name not in ("measure", "barrier")]
        if not order:
            raise RuntimeError("measurement circuit reads nothing into its last register")
        body.save_probabilities(order, label="p")
        res = sim.run(body, shots=1).result()
        vectors.append(np.asarray(res.data(0)["p"], dtype=float))
        tqcs.append(tqc)
    return vectors, tqcs


def _permute_density_matrix(rho, phys_of_virtual):
    """rho' whose qubit v is qubit phys_of_virtual[v] of rho."""
    n = len(phys_of_virtual)
    t = np.asarray(rho, dtype=complex).reshape((2,) * (2 * n))
    # axis of ket qubit q is n-1-q, of bra qubit q is 2n-1-q
    ket = [n - 1 - int(phys_of_virtual[v]) for v in range(n)][::-1]
    bra = [2 * n - 1 - int(phys_of_virtual[v]) for v in range(n)][::-1]
    return np.transpose(t, ket + bra).reshape(2 ** n, 2 ** n)


def _exact_density_matrix(body, noise_model, coupling_map, optimization_level=1, seed_transpiler=0):
    """Noisy density matrix of a body circuit in VIRTUAL qubit order."""
    Q = _qiskit()
    sim = Q.AerSimulator(method="density_matrix", noise_model=noise_model)
    tqc = transpile_native(body, coupling_map, optimization_level, seed_transpiler)
    tqc.save_density_matrix(label="rho")
    rho = np.asarray(sim.run(tqc, shots=1).result().data(0)["rho"])
    if tqc.layout is not None:
        rho = _permute_density_matrix(rho, tqc.layout.final_index_layout())
    return rho, tqc


def readout_calibration_circuits(n_qubits):
    """The two tensored-mitigation calibration circuits |0..0> and X^n|0..0>
    (every qubit measured into register 'm0')."""
    Q = _qiskit()
    out = []
    for ones in (False, True):
        qc = Q.QuantumCircuit(int(n_qubits), name="cal_ones" if ones else "cal_zeros")
        if ones:
            qc.x(range(int(n_qubits)))
        cr = Q.ClassicalRegister(int(n_qubits), "m0")
        qc.add_register(cr)
        qc.measure(range(int(n_qubits)), cr)
        out.append(qc)
    return out


def calibration_from_vectors(p_zeros, p_ones, n_qubits):
    """Per-qubit assignment matrices A_q[measured, prepared] from the two
    calibration distributions (tensored model: marginals per bit)."""
    mats = []
    idx = np.arange(2 ** int(n_qubits))
    for q in range(int(n_qubits)):
        bit = (idx >> q) & 1
        p10 = float(p_zeros[bit == 1].sum())          # read 1 | prepared 0
        p01 = float(p_ones[bit == 0].sum())           # read 0 | prepared 1
        mats.append(np.array([[1.0 - p10, p01], [p10, 1.0 - p01]]))
    return mats


def calibration_from_counts(counts_zeros, counts_ones, n_qubits):
    return calibration_from_vectors(counts_to_vector(counts_zeros, n_qubits),
                                    counts_to_vector(counts_ones, n_qubits), n_qubits)


def tensored_calibration(noise_model, readout_flips, coupling_map, optimization_level=1,
                         seed_transpiler=0):
    """Shot-free tensored readout calibration through the same noisy
    simulator (the X gates of the |1..1> circuit carry their gate noise, as
    on the device); returns per-qubit assignment matrices."""
    n = len(readout_flips)
    circs = readout_calibration_circuits(n)
    vecs, _ = _exact_probability_vectors(circs, noise_model, coupling_map, optimization_level,
                                         seed_transpiler)
    vecs = [apply_readout_vector(v, list(readout_flips)) for v in vecs]
    return calibration_from_vectors(vecs[0], vecs[1], n)


def apply_tensored_mitigation(probs, mats):
    """p_true = (A_0^{-1} x A_1^{-1} x ...) p_measured, bit k <- mats[k]."""
    p = np.asarray(probs, dtype=float)
    n = int(round(math.log2(p.size)))
    t = p.reshape((2,) * n)
    for k, A in enumerate(mats):
        Minv = np.linalg.inv(np.asarray(A, dtype=float))
        ax = n - 1 - k
        t = np.moveaxis(np.tensordot(Minv, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def _circuit_stats(tqc):
    ops = tqc.count_ops()
    return {"depth": int(tqc.depth()), "cx_count": int(ops.get("cx", 0)),
            "n_gates": int(sum(ops.values()))}


# ---------------------------------------------------------------------------
# (a) QET — circuits
# ---------------------------------------------------------------------------


def qet_circuits(chain: TruncatedChain, params, layers, site_A, site_B, theta, form="deferred"):
    """The three QET circuit families (module docstring).

    Returns dict family -> {'body': QuantumCircuit, 'operators': {name:
    (groups, constant)}, 'measured': qubit list}; plus 'meta' with the
    sign qubit, W_A, Bob's site unitaries and the rotated-frame operators.
    ``form='feedforward'`` builds the mid-circuit-measurement version of
    'protocol' (sampling only).
    """
    from qiskit.circuit.library import UnitaryGate

    Q = _qiskit()
    A, B = int(site_A), int(site_B)
    n = chain.n_qubits
    d = chain.d
    W, lam = alice_basis_change(chain, A)
    sign_qubit = chain.site_qubits(A)[-1]
    prep = _prep_gate(chain, params, layers)
    hA = site_operators(d, chain.omega[A])["h_site"]
    Pp = W[:, : d // 2] @ W[:, : d // 2].T
    Pm = W[:, d // 2:] @ W[:, d // 2:].T
    D_A = Pp @ hA @ Pp + Pm @ hA @ Pm - hA
    ops_DA = measurement_groups(_site_terms(D_A, chain, A))
    ops_HB = measurement_groups(touched_pauli_terms(chain, B))
    ops_HB_rot = measurement_groups(
        touched_pauli_terms(chain, B, x_override={A: np.diag(lam)})
    )
    U_plus = bob_displacement_site(chain, B, +1, theta)
    U_minus = bob_displacement_site(chain, B, -1, theta)

    def body(name):
        qc = Q.QuantumCircuit(n, name=name)
        qc.append(prep, list(range(n)))
        return qc

    def alice(qc):
        if d == 2:
            qc.h(sign_qubit)
        else:
            qc.append(UnitaryGate(W.T, label="WA_dag"), chain.site_qubits(A))

    def bob(qc, U, ctrl_state):
        bq = chain.site_qubits(B)
        if d == 2:
            # U = R_Y(2 mu theta sqrt(omega/2)) -> angle from the matrix
            ang = 2.0 * math.atan2(U[1, 0].real, U[0, 0].real)
            if ctrl_state == 0:
                qc.x(sign_qubit)
                qc.cry(ang, sign_qubit, bq[0])
                qc.x(sign_qubit)
            else:
                qc.cry(ang, sign_qubit, bq[0])
        else:
            g = UnitaryGate(U, label=f"UB_{'p' if ctrl_state == 0 else 'm'}").control(
                1, ctrl_state=str(ctrl_state))
            qc.append(g, [sign_qubit] + bq)

    fam = {}
    qc = body("qet_ground")
    fam["ground"] = {"body": qc, "operators": {"D_A": ops_DA, "H_B": ops_HB},
                     "measured": list(range(n))}
    qc = body("qet_control")
    alice(qc)
    fam["control"] = {"body": qc, "operators": {"H_B_rot": ops_HB_rot},
                      "measured": list(range(n))}
    qc = body("qet_protocol")
    alice(qc)
    if form == "deferred":
        bob(qc, U_minus, 1)          # mu = -1 when the sign qubit reads 1
        bob(qc, U_plus, 0)           # mu = +1 when it reads 0
    elif form == "feedforward":
        cr = Q.ClassicalRegister(1, "a")
        qc.add_register(cr)
        qc.measure(sign_qubit, cr[0])
        bq = chain.site_qubits(B)
        for val, U in ((0, U_plus), (1, U_minus)):
            with qc.if_test((cr[0], val)):
                if d == 2:
                    qc.ry(2.0 * math.atan2(U[1, 0].real, U[0, 0].real), bq[0])
                else:
                    qc.append(UnitaryGate(U), bq)
    else:
        raise ValueError("form must be 'deferred' or 'feedforward'")
    fam["protocol"] = {"body": qc, "operators": {"H_B_rot": ops_HB_rot},
                       "measured": list(range(n))}
    if form == "deferred":
        qc = body("qet_wrong")
        alice(qc)
        bob(qc, U_plus, 1)           # inverted bit: same gates, wrong information
        bob(qc, U_minus, 0)
        fam["wrong"] = {"body": qc, "operators": {"H_B_rot": ops_HB_rot},
                        "measured": list(range(n))}
    fam["meta"] = {"sign_qubit": sign_qubit, "W_A": W, "x_A_eigenvalues": lam,
                   "U_plus": U_plus, "U_minus": U_minus, "theta": float(theta),
                   "site_A": A, "site_B": B, "form": form, "n_qubits": n,
                   "coupling_map": linear_coupling_map(n)}
    return fam


def _qet_ledger_from_values(vals, HB_ground_ideal, shots=None):
    """Assemble the hardware-style ledger from operator expectation dicts."""
    E_A = vals["ground"]["D_A"]
    HB0 = vals["ground"]["H_B"]
    HBm = vals["control"]["H_B_rot"]
    HBf = vals["protocol"]["H_B_rot"]
    se = lambda x: (x["std_error"] or 0.0)
    E_B_ctrl = HBm["value"] - HBf["value"]
    E_B_raw = HB_ground_ideal - HBf["value"]
    out = {
        "E_A": E_A["value"],
        "E_A_std_error": E_A["std_error"],
        "HB_ground": HB0["value"], "HB_meas": HBm["value"], "HB_final": HBf["value"],
        "E_B": E_B_ctrl,
        "E_B_ctrl": E_B_ctrl,
        "E_B_ctrl_std_error": (math.sqrt(se(HBm) ** 2 + se(HBf) ** 2) if shots else None),
        "E_B_raw": E_B_raw,
        "E_B_raw_std_error": HBf["std_error"],
    }
    if "wrong" in vals:
        HBw = vals["wrong"]["H_B_rot"]
        out.update({
            "HB_wrong": HBw["value"],
            "E_B_wrong": HBm["value"] - HBw["value"],
            "E_B_info": HBw["value"] - HBf["value"],
            "E_B_info_std_error": (math.sqrt(se(HBw) ** 2 + se(HBf) ** 2) if shots else None),
            "E_B_cut": HBm["value"] - 0.5 * (HBf["value"] + HBw["value"]),
        })
    out.update({
        "no_signaling_energy": HBm["value"] - HB0["value"],
        "no_signaling_energy_std_error": (math.sqrt(se(HBm) ** 2 + se(HB0) ** 2) if shots else None),
        "values": vals,
        "shots": shots,
    })
    return out


def qet_exact_aer(chain, params, layers, site_A, site_B, theta, noise_model=None,
                  readout_flips=None, shots=10000, coupling_map=None, optimization_level=1,
                  seed_transpiler=0, ideal=None, mitigation="none"):
    """Shot-free QET ledger from Aer's density-matrix method.

    Gate noise through ``noise_model`` on the native-gate circuits; readout
    flips ``readout_flips`` (per qubit) applied analytically; ``shots``
    only sets the reported standard errors.  ``ideal`` (a dense ledger from
    :func:`qet_dense`) supplies HB_ground for E_B_raw; without it the
    family's own value is used.  ``mitigation='tensored'`` applies the
    per-qubit calibration-matrix readout mitigation (Ikeda 2023, Sec. II.D,
    tensored over the register; calibration from the two circuits |0..0>,
    |1..1> through the same noisy simulator) to every estimator, with the
    corresponding variance inflation in the standard errors.

    Returns the ledger of :func:`_qet_ledger_from_values` plus
    ``no_signaling`` (trace distance of Bob's reduced state, 'ground' vs
    'control', from the density matrices), ``circuits`` (per family/
    operator, transpiled), ``depth``/``cx_count`` of the deepest circuit,
    ``ledger_defect`` (= the no-signaling energy residual |<H_B>_meas -
    <H_B>_0|, the locality check available on hardware), ``mitigation``.
    """
    Q = _qiskit()
    fam = qet_circuits(chain, params, layers, site_A, site_B, theta, form="deferred")
    cmap = fam["meta"]["coupling_map"] if coupling_map is None else coupling_map
    n = chain.n_qubits
    flips = list(readout_flips) if readout_flips is not None else [0.0] * n
    mats = None
    if mitigation == "tensored":
        mats = tensored_calibration(noise_model, flips, cmap, optimization_level, seed_transpiler)
    elif mitigation != "none":
        raise ValueError("mitigation must be 'none' or 'tensored'")
    vals = {}
    circuits = {}
    stats = []
    for name in ("ground", "control", "protocol", "wrong"):
        f = fam[name]
        vals[name] = {}
        circuits[name] = {}
        for opname, (groups, const) in f["operators"].items():
            mcs = measurement_circuits(f["body"], groups, f["measured"], name=f"{name}_{opname}")
            vecs, tqcs = _exact_probability_vectors(mcs, noise_model, cmap, optimization_level,
                                                    seed_transpiler)
            vecs = [apply_readout_vector(v, [flips[q] for q in f["measured"]]) for v in vecs]
            mit = None if mats is None else [mats[q] for q in f["measured"]]
            vals[name][opname] = expectation_from_groups(vecs, groups, const, f["measured"], shots, mit)
            circuits[name][opname] = tqcs
            stats += [_circuit_stats(t) for t in tqcs]
    HB_ideal = ideal["HB_ground"] if ideal is not None else vals["ground"]["H_B"]["value"]
    led = _qet_ledger_from_values(vals, HB_ideal, shots)
    rhos = {}
    for name in ("ground", "control"):
        rho, _ = _exact_density_matrix(fam[name]["body"], noise_model, cmap, optimization_level,
                                       seed_transpiler)
        rhos[name] = reduced_site_state(chain, rho, site_B)
    led["no_signaling"] = trace_distance(rhos["ground"], rhos["control"])
    led["rho_B"] = rhos
    led["ledger_defect"] = abs(led["no_signaling_energy"])
    led["circuits"] = circuits
    led["depth"] = max(s["depth"] for s in stats)
    led["cx_count"] = max(s["cx_count"] for s in stats)
    led["circuit_stats"] = stats
    led["n_circuits"] = len(stats)
    led["meta"] = fam["meta"]
    led["mitigation"] = mitigation
    led["calibration"] = mats
    return led


def qet_sampled(chain, params, layers, site_A, site_B, theta, backend, shots, seed=None,
                sampler=None, form="deferred", ideal=None, optimization_level=1, mitigation="none"):
    """Run the QET circuit families on ``backend`` (Aer with a noise model, a
    fake backend, or — with ``sampler=SamplerV2(mode=backend)`` — a real
    device) through :func:`vacuum.qet.ikeda_qiskit.run_on_backend`, and
    build the ledger from the counts.  Never selects a device itself."""
    fam = qet_circuits(chain, params, layers, site_A, site_B, theta, form=form)
    counts = {}
    k = 0
    for name in [f for f in ("ground", "control", "protocol", "wrong") if f in fam]:
        f = fam[name]
        counts[name] = {}
        for opname, (groups, const) in f["operators"].items():
            mcs = measurement_circuits(f["body"], groups, f["measured"], name=f"{name}_{opname}")
            counts[name][opname] = [
                run_on_backend(qc, backend, shots, seed=(None if seed is None else seed + k + i),
                               sampler=sampler, optimization_level=optimization_level)
                for i, qc in enumerate(mcs)
            ]
            k += len(mcs)
    cal = None
    if mitigation == "tensored":
        c0, c1 = [run_on_backend(qc, backend, shots, seed=(None if seed is None else seed + 100 + i),
                                 sampler=sampler, optimization_level=optimization_level)
                  for i, qc in enumerate(readout_calibration_circuits(chain.n_qubits))]
        cal = calibration_from_counts(c0, c1, chain.n_qubits)
    return qet_from_counts(fam, counts, ideal=ideal, shots=shots, calibration=cal)


def qet_from_counts(fam, counts, ideal=None, shots=None, calibration=None):
    """Ledger from device/simulator counts (keys as produced by
    :func:`qet_sampled`; the readout register is the last one added, which
    qiskit prints as the LEFTMOST space-separated group)."""
    vals = {}
    for name in [f for f in ("ground", "control", "protocol", "wrong") if f in fam and f in counts]:
        f = fam[name]
        n_bits = len(f["measured"])
        vals[name] = {}
        for opname, (groups, const) in f["operators"].items():
            vecs = [counts_to_vector(c, n_bits) for c in counts[name][opname]]
            tot = shots if shots is not None else sum(counts[name][opname][0].values())
            mit = None if calibration is None else [calibration[q] for q in f["measured"]]
            vals[name][opname] = expectation_from_groups(vecs, groups, const, f["measured"], tot, mit)
    HB_ideal = ideal["HB_ground"] if ideal is not None else vals["ground"]["H_B"]["value"]
    led = _qet_ledger_from_values(vals, HB_ideal, shots)
    led["counts"] = counts
    led["meta"] = fam["meta"]
    return led


# ---------------------------------------------------------------------------
# (b) harvesting — model and dense path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarvestSpec:
    """Two qubit UDW detectors on the encoded field (module docstring).

    sites : (j_A, j_B) field sites the detectors couple to.
    gap : detector gap Omega (H_d = (Omega/2)(1 - Z_d)).
    lam : coupling strength (H_int = lam chi(t) X_d x_j).
    kind : switching kind ('cos2' | 'gaussian').
    width : the switching T parameter (:mod:`vacuum.detectors.switching`).
    window : total evolution window [0, window]; the profile is centred at
        window/2 (cos2: window = 2 width covers the exact support; a
        Gaussian is truncated at the window edges — declared, not hidden).
    n_steps : Trotter steps of the product formula.
    """

    sites: Tuple[int, int]
    gap: float
    lam: float
    kind: str
    width: float
    window: float
    n_steps: int

    @property
    def switching(self):
        return make_switching(self.kind, self.width, t0=0.5 * self.window)

    @property
    def dt(self):
        return self.window / self.n_steps

    def causal_contact(self):
        """True if the switching window exceeds the lattice light-crossing
        time between the sites (v_max = 1 on the chain): the detectors are
        then in causal contact and part of what they share is communication,
        not vacuum correlation (vacuum.detectors.communication)."""
        return self.window > abs(self.sites[0] - self.sites[1])


def harvest_spec(sites, gap, lam, kind="cos2", width=1.0, n_steps=4, window=None):
    if window is None:
        window = 2.0 * float(width) if kind == "cos2" else 6.0 * float(width)
    sites = (int(sites[0]), int(sites[1]))
    if sites[0] == sites[1]:
        raise ValueError("the two detectors need distinct sites")
    return HarvestSpec(sites=sites, gap=float(gap), lam=float(lam), kind=str(kind),
                       width=float(width), window=float(window), n_steps=int(n_steps))


def harvest_terms(chain: TruncatedChain, spec: HarvestSpec):
    """(static_terms, int_terms, constant): PauliTerms on n_field + 2 qubits.

    static = H_field (identity dropped into ``constant``) + sum_d
    (Omega/2)(-Z_d); int = sum_d X_d x_{j_d} (to be scaled by lam chi(t)).
    Ancilla A is qubit n_field, B is n_field + 1.
    """
    nf = chain.n_qubits
    static = []
    constant = 0.0
    for t in hamiltonian_pauli_terms(chain):
        if t.is_identity():
            constant += t.coeff.real
        else:
            static.append(t)
    for k in range(2):
        static.append(PauliTerm(paulis=((nf + k, "Z"),), coeff=-0.5 * spec.gap))
        constant += 0.5 * spec.gap
    interaction = []
    for k, j in enumerate(spec.sites):
        for t in site_pauli_terms(chain, j, "x"):
            interaction.append(_multiply_disjoint(PauliTerm(paulis=((nf + k, "X"),), coeff=1.0), t))
    return static, _merge(interaction), float(constant)


def trotter_schedule(spec: HarvestSpec):
    """Midpoint times, dt and chi(t_mid) of the n_steps Trotter steps."""
    dt = spec.dt
    t_mid = (np.arange(spec.n_steps) + 0.5) * dt
    chi = spec.switching(t_mid)
    return t_mid, dt, np.asarray(chi, dtype=float)


def _pauli_masks(term: PauliTerm, n):
    xmask = 0
    zmask = 0
    n_y = 0
    for qb, ch in term.paulis:
        if ch in ("X", "Y"):
            xmask |= 1 << qb
        if ch in ("Z", "Y"):
            zmask |= 1 << qb
        if ch == "Y":
            n_y += 1
    return xmask, zmask, n_y


def _popcount(arr):
    arr = arr.astype(np.uint64)
    c = np.zeros_like(arr)
    while np.any(arr):
        c += arr & np.uint64(1)
        arr = arr >> np.uint64(1)
    return c


def apply_pauli_string(psi, term: PauliTerm, n):
    """coeff * P |psi> for a Pauli string on n qubits (bit q of the index =
    qubit q).  Uses Y = i X Z: P = (i)^{n_Y} X^{xmask} Z^{zmask} with the
    Z applied first."""
    xmask, zmask, n_y = _pauli_masks(term, n)
    idx = np.arange(2 ** n, dtype=np.int64)
    phase = np.where(_popcount(idx & zmask) % 2 == 1, -1.0, 1.0) * (1.0j ** n_y)
    out = np.empty_like(psi)
    out[idx ^ xmask] = phase * psi[idx]
    return term.coeff * out


def apply_exp_pauli(psi, term: PauliTerm, angle, n):
    """exp(-i angle coeff P) |psi> = cos(a) psi - i sin(a) P psi, a = angle coeff."""
    a = float(angle) * term.coeff.real
    P = apply_pauli_string(psi, PauliTerm(term.paulis, 1.0), n)
    return math.cos(a) * psi - 1.0j * math.sin(a) * P


def _strang_step(psi, groups, dt, n):
    """One second-order step: groups 1..M-1 for dt/2, group M for dt, then
    groups M-1..1 for dt/2 (the exponential of each group is the exact
    product of its commuting term exponentials)."""
    M = len(groups)
    for g in groups[:-1]:
        for t in g:
            psi = apply_exp_pauli(psi, t, 0.5 * dt, n)
    for t in groups[-1]:
        psi = apply_exp_pauli(psi, t, dt, n)
    for g in reversed(groups[:-1]):
        for t in g:
            psi = apply_exp_pauli(psi, t, 0.5 * dt, n)
    return psi


def _step_groups(static, interaction, lam_chi):
    """Commuting groups of H(t_k) = static + lam chi_k * interaction."""
    terms = list(static) + [PauliTerm(t.paulis, t.coeff * lam_chi) for t in interaction]
    return commuting_groups(terms)


def _sparse_from_terms(terms, n):
    dim = 2 ** n
    idx = np.arange(dim, dtype=np.int64)
    rows = []
    cols = []
    vals = []
    for t in terms:
        xmask, zmask, n_y = _pauli_masks(t, n)
        phase = np.where(_popcount(idx & zmask) % 2 == 1, -1.0, 1.0) * (1.0j ** n_y)
        rows.append(idx ^ xmask)
        cols.append(idx)
        vals.append(t.coeff * phase)
    return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(dim, dim))


def reduced_pair_state(psi_full, n_field):
    """rho_AB of the two ancillas (A the FIRST tensor factor: index 2a + b)."""
    psi = np.asarray(psi_full, dtype=complex)
    nf = int(n_field)
    t = psi.reshape(2, 2, 2 ** nf)              # (b, a, field)
    rho = np.einsum("baf,dcf->abcd", t, t.conj())   # rho[a,b,a',b']
    return rho.reshape(4, 4)


def partial_transpose_B(rho):
    r = np.asarray(rho, dtype=complex).reshape(2, 2, 2, 2)   # a b a' b'
    return np.transpose(r, (0, 3, 2, 1)).reshape(4, 4)


def negativity_two_qubit(rho):
    """(lambda_min, E_N, negativity) of a two-qubit state.

    lambda_min is the smallest eigenvalue of rho^{T_B}; at most one is
    negative (Sanpera-Tarrach-Vidal 1998), so ||rho^{T_B}||_1 = 1 - 2
    lambda_min when lambda_min < 0, E_N = ln(1 - 2 lambda_min) (nats) and
    the negativity is max(0, -lambda_min).
    """
    pt = partial_transpose_B(rho)
    w = np.linalg.eigvalsh(0.5 * (pt + pt.conj().T))
    lmin = float(w[0])
    if lmin < 0.0:
        return lmin, float(math.log(1.0 - 2.0 * lmin)), -lmin
    return lmin, 0.0, 0.0


def harvest_dense(chain: TruncatedChain, psi0, spec: HarvestSpec, method="trotter",
                  exact_min_steps=64, exact_tol=1e-10, exact_max_doublings=8):
    """Dense harvesting on the truncated chain.

    method='trotter': exactly the circuit's operator sequence (Strang over
    the commuting groups, n_steps steps at the profile midpoints).
    method='exact': classical fourth-order Runge–Kutta integration of
    i d/dt psi = H(t) psi (norm renormalised at the end; the drift is
    reported implicitly through the gate), the step count doubled from
    ``exact_min_steps`` until lambda_min and rho_AB move by less than
    ``exact_tol`` between consecutive doublings.

    Returns dict: rho_AB (A first factor), lambda_min, E_N, negativity,
    p_exc (ancilla excitation probabilities), psi (final state), method,
    n_steps (used), converged / movement (exact), groups (trotter: the
    group sizes), causal_contact.
    """
    nf = chain.n_qubits
    n = nf + 2
    static, interaction, const = harvest_terms(chain, spec)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[: 2 ** nf] = np.asarray(psi0, dtype=complex)     # ancillas in |00>
    out = {"method": method, "causal_contact": spec.causal_contact(), "spec": spec}
    if method == "trotter":
        t_mid, dt, chi = trotter_schedule(spec)
        sizes = None
        for k in range(spec.n_steps):
            groups = _step_groups(static, interaction, spec.lam * chi[k])
            sizes = [len(g) for g in groups]
            psi = _strang_step(psi, groups, dt, n)
        out["n_steps"] = spec.n_steps
        out["groups"] = sizes
    elif method == "exact":
        H_s = _sparse_from_terms(static, n)
        H_i = _sparse_from_terms(interaction, n)
        sw = spec.switching

        def rhs(t, v):
            return -1.0j * (H_s @ v + (spec.lam * float(sw(t))) * (H_i @ v))

        prev = None
        n_steps = int(exact_min_steps)
        move = None
        for _ in range(int(exact_max_doublings) + 1):
            # classical fourth-order Runge–Kutta on i d/dt psi = H(t) psi
            phi = psi.copy()
            dt = spec.window / n_steps
            for k in range(n_steps):
                t_k = k * dt
                k1 = rhs(t_k, phi)
                k2 = rhs(t_k + 0.5 * dt, phi + 0.5 * dt * k1)
                k3 = rhs(t_k + 0.5 * dt, phi + 0.5 * dt * k2)
                k4 = rhs(t_k + dt, phi + dt * k3)
                phi = phi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            phi = phi / np.linalg.norm(phi)
            rho = reduced_pair_state(phi, nf)
            lmin = negativity_two_qubit(rho)[0]
            if prev is not None:
                move = max(abs(lmin - prev[0]), float(np.max(np.abs(rho - prev[1]))))
                if move < exact_tol:
                    out["converged"] = True
                    out["movement"] = move
                    break
            prev = (lmin, rho)
            n_steps *= 2
        else:
            out["converged"] = False
            out["movement"] = move
        psi = phi
        out["n_steps"] = n_steps // 2 if out.get("converged") else n_steps
    else:
        raise ValueError("method must be 'trotter' or 'exact'")
    rho = reduced_pair_state(psi, nf)
    lmin, E_N, neg = negativity_two_qubit(rho)
    probs = np.abs(psi) ** 2
    idx = np.arange(2 ** n)
    out.update({
        "rho_AB": rho, "lambda_min": lmin, "E_N": E_N, "negativity": neg, "psi": psi,
        "p_exc": (float(probs[(idx >> nf) & 1 == 1].sum()), float(probs[(idx >> (nf + 1)) & 1 == 1].sum())),
        "constant": const,
    })
    return out


def harvest_gaussian_reference(chain: TruncatedChain, spec: HarvestSpec, n_ts=None,
                               conv_tol=1e-9, **kwargs):
    """The same protocol with oscillator detectors through
    ``vacuum.detectors.run_harvesting`` (Gaussian, exact), lam_osc = lam
    sqrt(2 Omega) (MAP-1 of tests/test_gate_crossvalidation.py).  Returns
    dict with E_N, lam_osc, and the HarvestResult."""
    from vacuum.detectors.nonperturbative import run_harvesting

    lam_osc = spec.lam * math.sqrt(2.0 * spec.gap)
    sw = spec.switching
    if n_ts is None:
        n_ts = max(65, 16 * spec.n_steps + 1)
    ts = np.linspace(0.0, spec.window, int(n_ts))
    res = run_harvesting(
        chain.K, list(spec.sites), [spec.gap, spec.gap],
        lambda t: lam_osc * float(sw(t)), ts,
        dlambda_of_t=lambda t: lam_osc * float(sw.derivative(t)),
        conv_tol=conv_tol, **kwargs,
    )
    return {"E_N": res.E_N, "lam_osc": lam_osc, "result": res, "flagged": res.flagged}


# ---------------------------------------------------------------------------
# (b) harvesting — circuits
# ---------------------------------------------------------------------------


def harvest_coupling_map(chain: TruncatedChain, spec: HarvestSpec):
    """Field line plus each ancilla attached to its site's top qubit — a tree
    of degree <= 3, as available on heavy-hex devices, so no routing SWAPs
    are needed."""
    nf = chain.n_qubits
    edges = linear_coupling_map(nf)
    for k, j in enumerate(spec.sites):
        top = chain.site_qubits(j)[-1]
        edges += [[nf + k, top], [top, nf + k]]
    return edges


def harvest_circuit(chain: TruncatedChain, params, layers, spec: HarvestSpec):
    """Preparation + n_steps Strang steps as a ``QuantumCircuit`` on
    n_field + 2 qubits (ancillas A = n_field, B = n_field + 1), term
    exponentials via ``PauliEvolutionGate`` (exact for a single term)."""
    from qiskit.circuit.library import PauliEvolutionGate

    Q = _qiskit()
    nf = chain.n_qubits
    n = nf + 2
    static, interaction, const = harvest_terms(chain, spec)
    qc = Q.QuantumCircuit(n, name="harvest")
    qc.append(_prep_gate(chain, params, layers), list(range(nf)))
    t_mid, dt, chi = trotter_schedule(spec)

    def evolve(term, time):
        op = Q.SparsePauliOp.from_list([(pauli_label(term, n), complex(term.coeff.real))])
        qc.append(PauliEvolutionGate(op, time=float(time)), list(range(n)))

    for k in range(spec.n_steps):
        groups = _step_groups(static, interaction, spec.lam * chi[k])
        for g in groups[:-1]:
            for t in g:
                evolve(t, 0.5 * dt)
        for t in groups[-1]:
            evolve(t, dt)
        for g in reversed(groups[:-1]):
            for t in g:
                evolve(t, 0.5 * dt)
    return qc


def tomography_settings():
    """The nine two-qubit Pauli settings (P_A, P_B), P in X, Y, Z."""
    return [(a, b) for a in "XYZ" for b in "XYZ"]


def tomography_circuits(body, ancillas):
    """Nine circuits: body + basis change on the two ancillas + measurement
    of the ancillas into two classical bits (bit 0 = A, bit 1 = B)."""
    out = []
    qA, qB = int(ancillas[0]), int(ancillas[1])
    for a, b in tomography_settings():
        g = MeasurementGroup(terms=[PauliTerm(((qA, a), (qB, b)), 1.0)], basis={qA: a, qB: b})
        out += measurement_circuits(body, [g], [qA, qB], name=f"tomo_{a}{b}")
    return out


def rho_from_pauli_expectations(prob_vectors):
    """Linear-inversion two-qubit state from the nine setting distributions
    (length-4 vectors indexed a + 2b).  Single-qubit expectations are the
    average over the three settings that share the basis.  Returns rho with
    A the first tensor factor (index 2a + b)."""
    idx = np.arange(4)
    sa = 1.0 - 2.0 * (idx & 1)
    sb = 1.0 - 2.0 * ((idx >> 1) & 1)
    r = {("I", "I"): 1.0}
    singles_A = {c: [] for c in "XYZ"}
    singles_B = {c: [] for c in "XYZ"}
    for (a, b), p in zip(tomography_settings(), prob_vectors):
        p = np.asarray(p, dtype=float)
        r[(a, b)] = float(p @ (sa * sb))
        singles_A[a].append(float(p @ sa))
        singles_B[b].append(float(p @ sb))
    for c in "XYZ":
        r[(c, "I")] = float(np.mean(singles_A[c]))
        r[("I", c)] = float(np.mean(singles_B[c]))
    rho = np.zeros((4, 4), dtype=complex)
    for (a, b), val in r.items():
        rho += 0.25 * val * np.kron(PAULIS[a], PAULIS[b])
    return rho


def _bootstrap_lambda_min(prob_vectors, shots, n_boot=200, seed=0):
    rng = np.random.default_rng(int(seed))
    vals = np.empty(int(n_boot))
    shots = int(shots)
    for i in range(int(n_boot)):
        res = [rng.multinomial(shots, np.clip(p, 0.0, None) / np.sum(np.clip(p, 0.0, None))) / shots
               for p in prob_vectors]
        vals[i] = negativity_two_qubit(rho_from_pauli_expectations(res))[0]
    return float(np.std(vals, ddof=1)), vals


def _harvest_result(prob_vectors, shots, n_boot, seed, mitigation=None):
    raw = [np.asarray(p, dtype=float) for p in prob_vectors]
    mit = (lambda p: apply_tensored_mitigation(p, mitigation)) if mitigation is not None else (lambda p: p)
    rho = rho_from_pauli_expectations([mit(p) for p in raw])
    lmin, E_N, neg = negativity_two_qubit(rho)
    sigma, boot = (None, None)
    if shots:
        rng = np.random.default_rng(int(seed))
        vals = np.empty(int(n_boot))
        for i in range(int(n_boot)):
            res = [mit(rng.multinomial(int(shots), np.clip(p, 0.0, None) / np.sum(np.clip(p, 0.0, None))) / float(shots))
                   for p in raw]
            vals[i] = negativity_two_qubit(rho_from_pauli_expectations(res))[0]
        sigma, boot = float(np.std(vals, ddof=1)), vals
    return {
        "rho_AB": rho, "lambda_min": lmin, "E_N": E_N, "negativity": neg,
        "lambda_min_std_error": sigma,
        "z_score": ((-lmin / sigma) if sigma else None),
        "bootstrap": boot,
        "prob_vectors": raw,
        "shots": shots,
        "hermiticity_defect": float(np.max(np.abs(rho - rho.conj().T))),
        "trace": float(np.real(np.trace(rho))),
        "mitigated": mitigation is not None,
    }


def harvest_exact_aer(chain, params, layers, spec, noise_model=None, readout_flips=None,
                      shots=10000, coupling_map=None, optimization_level=1, seed_transpiler=0,
                      n_boot=200, seed=0, mitigation="none"):
    """Shot-free harvesting run on Aer (density matrix): tomography settings
    evaluated exactly, readout flips analytic, shot noise only in the
    bootstrap standard error of lambda_min; ``mitigation='tensored'`` as in
    :func:`qet_exact_aer` (calibration on the two ancilla bits).  Returns
    the dict of :func:`_harvest_result` plus circuit statistics and the
    transpiled circuits."""
    nf = chain.n_qubits
    n = nf + 2
    body = harvest_circuit(chain, params, layers, spec)
    cmap = coupling_map if coupling_map is not None else harvest_coupling_map(chain, spec)
    flips = list(readout_flips) if readout_flips is not None else [0.0] * n
    circs = tomography_circuits(body, [nf, nf + 1])
    vecs, tqcs = _exact_probability_vectors(circs, noise_model, cmap, optimization_level, seed_transpiler)
    vecs = [apply_readout_vector(v, [flips[nf], flips[nf + 1]]) for v in vecs]
    mats = None
    if mitigation == "tensored":
        full = tensored_calibration(noise_model, flips, cmap, optimization_level, seed_transpiler)
        mats = [full[nf], full[nf + 1]]
    elif mitigation != "none":
        raise ValueError("mitigation must be 'none' or 'tensored'")
    out = _harvest_result(vecs, shots, n_boot, seed, mats)
    stats = [_circuit_stats(t) for t in tqcs]
    out.update({"circuits": tqcs, "depth": max(s["depth"] for s in stats),
                "cx_count": max(s["cx_count"] for s in stats), "circuit_stats": stats,
                "n_circuits": len(tqcs), "spec": spec, "coupling_map": cmap,
                "mitigation": mitigation, "calibration": mats})
    return out


def harvest_sampled(chain, params, layers, spec, backend, shots, seed=None, sampler=None,
                    optimization_level=1, n_boot=200, mitigation="none"):
    """Run the nine tomography circuits on ``backend`` (Aer / fake / real via
    ``sampler``) through :func:`run_on_backend`; negativity from counts."""
    nf = chain.n_qubits
    body = harvest_circuit(chain, params, layers, spec)
    circs = tomography_circuits(body, [nf, nf + 1])
    counts = [run_on_backend(qc, backend, shots, seed=(None if seed is None else seed + i),
                             sampler=sampler, optimization_level=optimization_level)
              for i, qc in enumerate(circs)]
    cal = None
    if mitigation == "tensored":
        c0, c1 = [run_on_backend(qc, backend, shots, seed=(None if seed is None else seed + 100 + i),
                                 sampler=sampler, optimization_level=optimization_level)
                  for i, qc in enumerate(readout_calibration_circuits(nf + 2))]
        full = calibration_from_counts(c0, c1, nf + 2)
        cal = [full[nf], full[nf + 1]]
    return harvest_from_counts(counts, shots=shots, n_boot=n_boot, seed=seed or 0, calibration=cal)


def harvest_from_counts(counts, shots=None, n_boot=200, seed=0, calibration=None):
    """Negativity from nine counts dicts (setting order of
    :func:`tomography_settings`, two-bit keys, bit 0 = A); ``calibration``
    = the two ancillas' assignment matrices for tensored mitigation."""
    vecs = [counts_to_vector(c, 2) for c in counts]
    shots = shots if shots is not None else sum(counts[0].values())
    out = _harvest_result(vecs, shots, n_boot, seed, calibration)
    out["counts"] = counts
    return out
