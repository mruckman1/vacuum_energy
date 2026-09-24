"""Harvest-then-teleport: the composite ledger of Layer 7, leap L4.

PLAN.md, L4: "harvest entanglement from the vacuum, then spend it as the
channel for energy teleportation, and compute the full composite ledger —
joules of measurement energy in, ebits harvested, joules made remotely
available out."  This module closes that loop on the validated stack —
the Layer 2 nonperturbative harvesting engine (``vacuum.detectors``, 'xx'
coupling, run through ``vacuum.protocol.run_protocol`` so the switching
work sits in a closing ledger) feeding a Gaussian measurement-plus-
feedforward step — and it defines the second half honestly first.

The definition, stated before anything is computed
-------------------------------------------------
Hotta's quantum energy teleportation (QET) extracts energy from a state
that is *locally passive but globally correlated* — the GROUND state of an
interacting Hamiltonian: no local operation at B can lower <H>, yet with
Alice's measurement outcome Bob's conditional local operation can, and
Alice's measurement pays E_A >= E_B for it (M. Hotta, "Quantum energy
teleportation: an introductory review", arXiv:1101.3954, Secs. 2-3: the
passivity of the vacuum, Eq. (3), and E_A >= E_B "because the total energy
does not become negative", p. 15).  A pair of detectors that has just
harvested entanglement from the vacuum is NOT in such a state: the
switching excited them (the ledger books the switching work W_in > 0
against the field-plus-detector energy), so the detector pair is locally
ACTIVE — Bob can extract energy from his detector with no communication at
all, and a naive "QET on the harvested state" would report that local
ergotropy as if it had been teleported.  It has not.  What the harvested
entanglement can honestly be said to buy is the *classical-communication-
assisted surplus*: the energy Bob extracts from his detector, under his own
detector Hamiltonian H_B, given Alice's measurement outcome, MINUS what he
extracts with no communication.  That is the daemonic ergotropy of
G. Francica, J. Goold, F. Plastina, M. Paternostro, npj Quantum Inf. 3, 12
(2017) [arXiv:1608.00124] — ergotropy (Eq. (2)) being the maximal work a
cyclic unitary extracts (A. E. Allahverdyan, R. Balian, Th. M.
Nieuwenhuizen, Europhys. Lett. 67, 565 (2004)), the daemonic ergotropy
(Eq. (4)) the outcome-averaged conditional ergotropy after a measurement on
the correlated party, and the daemonic gain (Eq. (5)) their difference,
which is >= 0 and vanishes on product states (their Theorem 2 for pure
states).  For Gaussian states with Gaussian measurements on Alice and
Gaussian unitaries on Bob every piece is closed-form linear algebra on
conditional covariance matrices, the setting of K. H. Kua, A. Serafini,
M. G. Genoni, Quantum Sci. Technol. 11, 015014 (2025) [arXiv:2506.22288],
whose Eqs. (16), (33), (42), (50), (51) are transcribed and anchored below.

Two columns are therefore reported, and named for what they are:

1. **Daemonic surplus on the harvested state** (:func:`daemonic_surplus`)
   — the honest quantitative form of "spending harvested entanglement as
   the channel for remote energy access": no coupling between the
   detectors, Bob's Hamiltonian is his own oscillator, E_local is what he
   gets anyway, E_daemonic what he gets with Alice's outcome, and the
   surplus is the difference.
2. **True-QET reference through a Hotta-type coupling**
   (:func:`hotta_variant`) — after harvesting, a weak x_A x_B coupling g
   is switched on between the detectors; the JOINT GROUND STATE of
   H_g = H_A + H_B + g x_A x_B is the locally passive reference (its
   passivity is audited, and the no-communication extraction from it is
   zero — a test), and the same measurement + feedforward is run against
   it: Alice injects E_A, Bob's conditional operation extracts E_B <= E_A
   from H_B + g x_A x_B (Bob's local energy operator includes the coupling
   term, exactly as in the minimal model's H_B + V, arXiv:1101.3954
   Sec. 3).  Those reference numbers use the ground-state correlations of
   H_g, not the harvested entanglement; the harvested state's own
   extraction under H_g is reported next to them, with the coupling
   switch-on work booked.

Conventions: docs/API.md throughout — hbar = k_B = 1, V_vac = I/2,
nu >= 1/2, block quadrature ordering.  A two-detector block V_AB is
ordered R = (x_A, x_B, p_A, p_B); the "mode-block" form used in the
formulas below is V_AB = [[V_A, C], [C^T, V_B]] with V_A, V_B the 2x2
single-mode covariances in (x, p) and C = Cov(R_A, R_B).  Entropies and
negativities in nats unless a name says ``_bits``.

The formulas (all derived here; each one is an anchor test)
-----------------------------------------------------------
Gaussian measurement on A.  A pure single-mode Gaussian measurement is a
general-dyne detection with pointer covariance sigma_M, det sigma_M = 1/4;
the family sigma_M(s, theta) = S (I/2) S^T, S = D R(theta) diag(s^-1/2,
s^1/2), D = diag(Omega_A^-1/2, Omega_A^1/2), exhausts it: s = 1 is the
heterodyne (projection onto Alice's own oscillator coherent states, the
minimum-energy pure measurement), s -> infinity the homodyne of the
quadrature u_theta = (cos theta sqrt(Omega_A), sin theta / sqrt(Omega_A)).
Bob's conditional covariance is the Schur complement (G. Giedke and J. I.
Cirac, Phys. Rev. A 66, 032316 (2002), Eq. (15) and its pseudo-inverse
remark for the homodyne limit; Kua-Serafini-Genoni Eq. (16)):

    V_{B|A}(sigma_M) = V_B - C^T (V_A + sigma_M)^-1 C,                    (1)
    V_{B|A}(homodyne u) = V_B - (C^T u)(C^T u)^T / (u^T V_A u),           (2)

(2) being the s -> infinity limit of (1) (proved in the module notes of
:func:`conditional_covariance`); the conditional mean is
m_B(alpha) = C^T (V_A + sigma_M)^-1 alpha with the outcome alpha Gaussian
of covariance V_A + sigma_M.  The conditional covariance is outcome-
independent — the whole Gaussian daemon lives in that fact.

Single-mode ergotropy.  For H_B = (1/2) R_B^T H_BB R_B with
omega_B = sqrt(det H_BB) (= Omega_B for H_BB = diag(Omega_B^2, 1)), the
minimum of (1/2) Tr(H_BB S V S^T) over symplectic S is omega_B nu with
nu = sqrt(det V) (Williamson on both), attained by the thermal state of
H_B with that nu; a displaced state adds (1/2) m^T H_BB m, removed by the
inverse displacement.  Since a single-mode Gaussian state is a Gaussian
unitary applied to a thermal state, its passive state IS that thermal
state, so the Gaussian ergotropy equals the full ergotropy here
(Kua-Serafini-Genoni Eq. (33), E - 1/(2 mu); E. G. Brown, N. Friis,
M. Huber, New J. Phys. 18, 113028 (2016) for Gaussian passivity):

    E_local = (1/2) Tr(H_BB V_B) - omega_B nu_B.                           (3)

Daemonic ergotropy.  Conditioned on alpha, Bob holds (m_B(alpha),
V_{B|A}); his optimal Gaussian operation leaves omega_B nu_{B|A}.  The
outcome average of the removed mean energy is
E[(1/2) m_B^T H_BB m_B] = (1/2) Tr(H_BB (V_B - V_{B|A})) and adds to
(1/2) Tr(H_BB V_{B|A}) to give the unconditional energy, so
(Kua-Serafini-Genoni Eq. (42))

    E_daemonic = (1/2) Tr(H_BB V_B) - omega_B nu_{B|A},                   (4)
    surplus    = E_daemonic - E_local = omega_B (nu_B - nu_{B|A}) >= 0.    (5)

The surplus is measurement-induced purification of Bob's mode, priced at
his oscillator frequency.  Maximised over the pure Gaussian family (s,
theta) it is the Gaussian daemonic gain; it is a LOWER bound on the
unrestricted daemonic gain (non-Gaussian measurements may do better).

Closed-form anchor (the test to 1e-10).  For the two-mode squeezed thermal
state of squeezing r and thermal occupation N (per mode, before
squeezing), written in the detectors' own units,

    homodyne of any quadrature:  surplus = Omega_B (2N + 1) sinh^2 r,      (6)
    heterodyne:  surplus = Omega_B (2N+1)^2 sinh^2(2r) /
                                  [2 + 2 (2N+1) cosh(2r)],                  (7)

Kua-Serafini-Genoni Eqs. (51) and (50); at N = 0 (two-mode squeezed
vacuum, E_N = 2r) both give Omega_B sinh^2 r = Omega_B sinh^2(E_N / 2),
E_local = 0 (Bob is thermal in his own H_B) and E_daemonic = surplus.

The resource bound (Landauer / Szilard form).  The entropy Bob loses on
conditioning is the Holevo quantity of the outcome ensemble and cannot
exceed the mutual information (data processing under Alice's local
measurement): S(nu_B) - S(nu_{B|A}) <= I(A:B).  With S'(nu) =
ln((nu+1/2)/(nu-1/2)) decreasing, the mean-value theorem gives
nu_B - nu_{B|A} <= I(A:B) / S'(nu_B), i.e.

    surplus <= k T_B I(A:B),   k T_B := omega_B / ln((nu_B+1/2)/(nu_B-1/2)),  (8)

T_B being the temperature of Bob's local passive (thermal) state: at most
one k T_B per nat of correlation.  This is the monotone resource bound
verified numerically here.  It is NOT a bound in the log-negativity:
classically correlated separable states (E_N = 0) have a strictly positive
surplus (a test exhibits one), so no bound of the form (energy scale) x
f(E_N) with f(0) = 0 can hold for mixed states; for PURE two-mode states
the surplus equals Omega_B sinh^2(E_N/2) exactly, which is the only
E_N-monotone statement made.  Two comparison columns ride along and
NEITHER is a bound: the ergotropy of the joint two-mode state under
H_A + H_B, Gaussian (Brown-Friis-Huber, symplectic spectrum against
frequencies) and full (passive state by sorted Fock spectrum, Allahverdyan
et al., computed by truncation with the discarded tail reported).  The
joint Gaussian ergotropy can be zero while the surplus is positive (the
classically correlated state), and even the full unitary ergotropy is
exceeded by E_daemonic on weakly correlated, nearly pure, equal-frequency
pairs — the map's T_sw = 5 row (E_daemonic 1.9e-6 against 1.6e-6) and a
synthetic thermal-pair construction in the tests.  That is the Szilard
engine in its simplest form: a measurement is not a unitary, and the
information it returns (up to k T_B per nat) is a resource the global
unitary ergotropy does not count.  The Landauer bound (8) is the one that
holds; the ergotropy columns are reported so the reader can see how far
the daemon reaches past what any cyclic unitary on the pair could do.

The Hotta reference.  With H = H_A + H_B + g x_A x_B (g^2 < Omega_A^2
Omega_B^2 so H is bounded below) and Alice's measurement leaving A in the
re-prepared pointer state D(alpha)|sigma_M> (the Lueders update of the
rank-one general-dyne POVM): the unconditional post-measurement covariance
is V + diag(2 sigma_M, 0) — B's block and the A-B block are unchanged
(no-signaling, and <x_A x_B> is preserved on average) — so Alice injects

    E_A = Tr(H_AA sigma_M)     (= Omega_A for the heterodyne),               (9)

and Bob, holding the conditional product state D(alpha)|sigma_M> (x)
(m_B(alpha), V_{B|A}) and knowing alpha, minimises
(1/2) m^T H_BB m + g alpha_x m_x over his displacement and (1/2) Tr(H_BB
S V_{B|A} S^T) over his symplectic, extracting on average

    E_B = (1/2) Tr(H_BB V_B) + g C_xx - Omega_B nu_{B|A}
          + g^2 (V_A + sigma_M)_xx / (2 Omega_B^2)                           (10)
        = E_B^corr + E_B^inj,   E_B^inj := g^2 (sigma_M)_xx / (2 Omega_B^2),

from his local energy operator H_B + g x_A x_B.  E_B^inj is Alice's
pointer noise re-routed through the coupling by Bob's displacement — it
grows without bound with (sigma_M)_xx, as does E_A, and E_B <= E_A holds
throughout because the final state is physical; E_B^corr is the part
carried by the pre-measurement correlations (the x_A-homodyne value).  On
the ground state of H_g the no-communication extraction is zero (audited
passivity, and :func:`local_extraction_coupled` returns 0 — a test) and
E_B is the teleported energy.

Every reported protocol runs the standing audits: the passivity audit on
the claimed vacuum (``initial_state`` and the runner's
``claim_ground_entry``) and on the Hotta reference ground state; the
precision audit on the reported detector block; mandatory ledger closure
to 1e-9 with the switching work charged as the telescoped <dH/dt> sum;
the causality audit of the field over the protocol window; and the
no-signaling audit (:func:`no_signaling_audit`): Bob's unconditional state
reconstructed from conditional covariance plus outcome-averaged
displacement is independent of Alice's measurement choice to 1e-14.

Pure functions over arrays and frozen configs; no hidden state, no input
mutation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from vacuum.audits import AuditResult, AuditViolation, EnergyLedger, causality_audit, passivity_audit
from vacuum.audits.ledger import DEFAULT_CLOSURE_ATOL
from vacuum.core import (
    entropy_from_nu,
    ground_state_cov,
    harmonic_chain_K,
    mutual_information,
    symplectic_eigenvalues,
)
from vacuum.detectors import (
    NegativityResult,
    attach_detectors,
    detector_block,
    detector_occupations,
    harvested_log_negativity,
    initial_state,
    switching,
)
from vacuum.protocol import GeneratorStep, run_protocol

from .fock_conditioning import FockConditioningResult, fock_conditioning, optimal_fock_squeezing

__all__ = [
    "HarvestConfig",
    "HarvestOutcome",
    "EbitsResult",
    "Measurement",
    "SurplusResult",
    "CompositeRow",
    "field_coupling",
    "harvest",
    "ebits",
    "mode_blocks",
    "gaussian_measurement",
    "conditional_covariance",
    "local_ergotropy",
    "local_extraction_coupled",
    "landauer_bound",
    "optimal_measurement",
    "daemonic_surplus",
    "no_signaling_audit",
    "pair_hamiltonian",
    "hotta_variant",
    "joint_ergotropy",
    "two_mode_squeezed_thermal",
    "composite_ledger",
]

LN2 = math.log(2.0)
_A_IDX = (0, 2)  # (x_A, p_A) rows of a block-ordered two-mode covariance
_B_IDX = (1, 3)  # (x_B, p_B)


# --------------------------------------------------------------------------
# Two-mode block algebra
# --------------------------------------------------------------------------


def _check_two_mode(V):
    V = np.asarray(V, dtype=float)
    if V.shape != (4, 4):
        raise ValueError(f"expected a two-mode (4, 4) covariance, got {V.shape}")
    return 0.5 * (V + V.T)


def mode_blocks(V_AB):
    """(V_A, V_B, C) of a block-ordered two-mode covariance.

    V_A, V_B are the 2x2 single-mode covariances in (x, p); C = Cov(R_A,
    R_B) with rows (x_A, p_A) and columns (x_B, p_B).
    """
    V = _check_two_mode(V_AB)
    iA, iB = list(_A_IDX), list(_B_IDX)
    return V[np.ix_(iA, iA)].copy(), V[np.ix_(iB, iB)].copy(), V[np.ix_(iA, iB)].copy()


def _from_blocks(V_A, V_B, C):
    V = np.zeros((4, 4))
    iA, iB = list(_A_IDX), list(_B_IDX)
    V[np.ix_(iA, iA)] = V_A
    V[np.ix_(iB, iB)] = V_B
    V[np.ix_(iA, iB)] = C
    V[np.ix_(iB, iA)] = np.asarray(C).T
    return V


def _single_mode_form(H_B):
    """Normalise Bob's Hamiltonian spec to a 2x2 quadratic form H_BB."""
    H = np.asarray(H_B, dtype=float)
    if H.ndim == 0:
        gap = float(H)
        if gap <= 0.0:
            raise ValueError(f"detector gap must be > 0, got {gap}")
        return np.diag([gap * gap, 1.0])
    if H.shape != (2, 2):
        raise ValueError(f"H_B must be a gap or a (2, 2) quadratic form, got {H.shape}")
    H = 0.5 * (H + H.T)
    if np.linalg.det(H) <= 0.0 or H[0, 0] <= 0.0:
        raise ValueError("H_B must be positive definite (a bounded-below oscillator)")
    return H


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _mode_energy(H_BB, V_B, mean=None):
    """<H_B> = (1/2) Tr(H_BB V_B) + (1/2) m^T H_BB m."""
    E = 0.5 * float(np.trace(H_BB @ V_B))
    if mean is not None:
        m = np.asarray(mean, dtype=float)
        E += 0.5 * float(m @ H_BB @ m)
    return E


def _sym_nu(V2):
    """Symplectic eigenvalue of a single-mode covariance: sqrt(det V)."""
    d = float(np.linalg.det(np.asarray(V2, dtype=float)))
    return math.sqrt(max(d, 0.0))


# --------------------------------------------------------------------------
# Gaussian measurements on Alice and Bob's conditional state
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Measurement:
    """A pure single-mode Gaussian measurement on Alice.

    ``kind``: 'heterodyne' (s = 1), 'general' (finite s > 1) or 'homodyne'
    (s = infinity, quadrature angle ``theta``).  ``sigma`` is the pointer
    covariance in Alice's physical quadratures (None for homodyne);
    ``u`` the measured quadrature direction for homodyne (None otherwise);
    ``gap_A`` the oscillator frequency that fixes the heterodyne point.
    """

    kind: str
    s: float
    theta: float
    gap_A: float
    sigma: Optional[np.ndarray]
    u: Optional[np.ndarray]
    S_inv: Optional[np.ndarray] = None   # sigma = S (I/2) S^T; S^-1 in closed form

    def label(self):
        if self.kind == "heterodyne":
            return "heterodyne"
        if self.kind == "homodyne":
            return f"homodyne(theta={self.theta:.6g})"
        return f"general-dyne(s={self.s:.6g}, theta={self.theta:.6g})"


def gaussian_measurement(kind="heterodyne", s=1.0, theta=0.0, gap_A=1.0):
    """Build a pure Gaussian measurement of the (s, theta) family.

    sigma_M(s, theta) = S (I/2) S^T with S = D R(theta) diag(s^-1/2, s^1/2),
    D = diag(Omega_A^-1/2, Omega_A^1/2): s = 1 is the heterodyne in Alice's
    oscillator units (theta then irrelevant), s -> infinity the homodyne of
    u_theta = (cos theta sqrt(Omega_A), sin theta / sqrt(Omega_A)).
    ``kind='homodyne'`` returns the exact limit (u only).
    """
    gap_A = float(gap_A)
    if gap_A <= 0.0:
        raise ValueError(f"gap_A must be > 0, got {gap_A}")
    theta = float(theta)
    if kind == "homodyne":
        u = np.array([math.cos(theta) * math.sqrt(gap_A), math.sin(theta) / math.sqrt(gap_A)])
        return Measurement("homodyne", math.inf, theta, gap_A, None, u, None)
    if kind == "heterodyne":
        s = 1.0
    elif kind != "general":
        raise ValueError(f"kind must be 'heterodyne', 'general' or 'homodyne', got {kind!r}")
    s = float(s)
    if not s >= 1.0:
        raise ValueError(f"s must be >= 1 (s = 1 is the heterodyne), got {s}")
    if math.isinf(s):
        return gaussian_measurement("homodyne", theta=theta, gap_A=gap_A)
    D = np.diag([1.0 / math.sqrt(gap_A), math.sqrt(gap_A)])
    R = _rot(theta)
    S = D @ R @ np.diag([1.0 / math.sqrt(s), math.sqrt(s)])
    S_inv = np.diag([math.sqrt(s), 1.0 / math.sqrt(s)]) @ R.T @ np.diag([math.sqrt(gap_A), 1.0 / math.sqrt(gap_A)])
    sigma = 0.5 * (S @ S.T)
    return Measurement("heterodyne" if s == 1.0 else "general", s, theta, gap_A,
                       0.5 * (sigma + sigma.T), None, S_inv)


def _as_measurement(measurement, gap_A):
    if isinstance(measurement, Measurement):
        return measurement
    if isinstance(measurement, str):
        if measurement in ("heterodyne", "homodyne"):
            return gaussian_measurement(measurement, gap_A=gap_A)
        raise ValueError(
            f"measurement {measurement!r} unknown: pass 'heterodyne', 'homodyne', "
            "'optimal' (daemonic_surplus only) or a Measurement"
        )
    if isinstance(measurement, (tuple, list)) and len(measurement) in (2, 3):
        kind = measurement[0]
        if kind == "homodyne":
            return gaussian_measurement("homodyne", theta=measurement[1], gap_A=gap_A)
        if kind == "general":
            return gaussian_measurement("general", s=measurement[1],
                                        theta=measurement[2] if len(measurement) == 3 else 0.0,
                                        gap_A=gap_A)
    raise TypeError(f"cannot interpret measurement spec {measurement!r}")


def _inv2(M):
    """Explicit inverse of a symmetric 2x2 matrix."""
    a, b, d = M[0, 0], M[0, 1], M[1, 1]
    det = a * d - b * b
    if det <= 0.0:
        raise ValueError("2x2 matrix is not positive definite")
    return np.array([[d, -b], [-b, a]]) / det


def _conditional_pieces(V_A, V_B, C, meas):
    """(V_{B|A}, E[m_B m_B^T]) for a measurement on A.

    Finite pointer: with sigma_M = S (I/2) S^T the Schur complement is
    evaluated in the pointer frame, V_A~ = S^-1 V_A S^-T, C~ = S^-1 C,
    G~ = (V_A~ + I/2)^-1, so that C^T (V_A + sigma_M)^-1 C = C~^T G~ C~ with
    no ill-conditioned inverse even for a strongly squeezed pointer (s of
    order 1e4 costs four digits in the naive inverse and none here).  The
    outcome covariance is V_A~ + I/2 in that frame, so the averaged outer
    product of the conditional mean is C~^T G~ (V_A~ + I/2) G~ C~.  Homodyne
    of u: V_B - (C^T u)(C^T u)^T/(u^T V_A u) and the same rank-one term for
    the mean part.  The two pieces sum to V_B identically (no-signaling).
    """
    if meas.kind == "homodyne":
        u = meas.u
        den = float(u @ V_A @ u)
        if den <= 0.0:
            raise ValueError("homodyne quadrature has non-positive variance")
        w = C.T @ u
        mean_cov = np.outer(w, w) / den
    else:
        Si = meas.S_inv
        Vt = Si @ V_A @ Si.T
        Ct = Si @ C
        M = 0.5 * (Vt + Vt.T) + 0.5 * np.eye(2)
        Gt = _inv2(M)
        mean_cov = Ct.T @ Gt @ M @ Gt @ Ct
    V_cond = V_B - mean_cov
    return 0.5 * (V_cond + V_cond.T), 0.5 * (mean_cov + mean_cov.T)


def conditional_covariance(V_AB, measurement="heterodyne", gap_A=1.0):
    """Bob's conditional covariance after a Gaussian measurement on Alice.

    Eq. (1) for a finite pointer covariance, Eq. (2) for a homodyne; the
    latter is the s -> infinity limit of the former: with sigma_M = (1/2)
    [a a^T / s + s b b^T] (a = D R e_1, b = D R e_2), lim (V_A + s b b^T /
    2)^-1 = V_A^-1 - V_A^-1 b b^T V_A^-1 / (b^T V_A^-1 b), a rank-one
    matrix annihilating b, hence proportional to u u^T for the u
    orthogonal to b — which is u_theta — with the constant fixed by
    u^T(.)u to 1/(u^T V_A u) (2x2 adjugate identity).  The test suite
    pins the limit numerically.
    """
    V_A, V_B, C = mode_blocks(V_AB)
    meas = _as_measurement(measurement, gap_A)
    V_cond, _ = _conditional_pieces(V_A, V_B, C, meas)
    return V_cond


# --------------------------------------------------------------------------
# Ergotropies
# --------------------------------------------------------------------------


def local_ergotropy(V_B, H_B):
    """Eq. (3): Gaussian (= full) ergotropy of one zero-mean Gaussian mode.

    Returns (ergotropy, mean energy, omega_B nu_B, nu_B).
    """
    H = _single_mode_form(H_B)
    V = np.asarray(V_B, dtype=float)
    if V.shape != (2, 2):
        raise ValueError(f"V_B must be (2, 2), got {V.shape}")
    omega = math.sqrt(float(np.linalg.det(H)))
    nu = _sym_nu(V)
    E = _mode_energy(H, V)
    return E - omega * nu, E, omega * nu, nu


def _embed_bob_symplectic(x):
    """3-parameter Sp(2, R) on Bob's mode embedded in the 4x4 block ordering."""
    th, r, ph = x
    S2 = _rot(th) @ np.diag([math.exp(r), math.exp(-r)]) @ _rot(ph)
    S = np.eye(4)
    iB = list(_B_IDX)
    S[np.ix_(iB, iB)] = S2
    return S


def local_extraction_coupled(V_AB, H, n_starts=10, seed=0):
    """Energy Bob extracts with no communication under a coupled H (4x4).

    Minimises (1/2) Tr(H (S_B (+) I) V (S_B (+) I)^T) over Bob's symplectic
    group (three parameters; a displacement cannot help because <R_A> = 0
    makes its cross-term vanish and its own term is >= 0), from the
    identity plus deterministic seeded restarts, and returns
    (E0 - E_min, S_best, E0).  The identity is always a candidate, so the
    result is >= 0 exactly; on the ground state of H it is 0 (passivity).
    """
    V = _check_two_mode(V_AB)
    H = np.asarray(H, dtype=float)
    if H.shape != (4, 4):
        raise ValueError(f"H must be a (4, 4) quadratic form, got {H.shape}")
    H = 0.5 * (H + H.T)
    E0 = 0.5 * float(np.trace(H @ V))
    rng = np.random.default_rng(int(seed))

    def f(x):
        S = _embed_bob_symplectic(x)
        return 0.5 * float(np.trace(H @ S @ V @ S.T))

    best_val, best_x = E0, np.zeros(3)
    starts = [np.zeros(3)]
    for _ in range(int(n_starts)):
        starts.append(rng.normal(size=3) * np.array([1.5, 0.6, 1.5]))
    for x0 in starts:
        res = minimize(f, x0, method="BFGS", options={"gtol": 1e-14, "maxiter": 400})
        if res.fun < best_val:
            best_val, best_x = float(res.fun), np.asarray(res.x, dtype=float)
    return E0 - best_val, _embed_bob_symplectic(best_x), E0


def landauer_bound(V_AB, omega_B):
    """Eq. (8): k T_B I(A:B) with k T_B = omega_B / ln((nu_B+1/2)/(nu_B-1/2)).

    Returns (bound, I_AB nats, nu_B, kT_B); a pure Bob (nu_B = 1/2) is
    uncorrelated with A and the bound is 0.
    """
    V = _check_two_mode(V_AB)
    _, V_B, _ = mode_blocks(V)
    nu_B = _sym_nu(V_B)
    I_AB = float(mutual_information(V, [0], [1]))
    gap = nu_B - 0.5
    if gap <= 1e-15:
        return 0.0, I_AB, nu_B, 0.0
    kT = float(omega_B) / math.log((nu_B + 0.5) / gap)
    return kT * I_AB, I_AB, nu_B, kT


def _gain_of(V_A, V_B, C, omega_B, nu_B, meas):
    V_cond, _ = _conditional_pieces(V_A, V_B, C, meas)
    return omega_B * (nu_B - _sym_nu(V_cond))


def optimal_measurement(V_AB, omega_B=1.0, gap_A=1.0, n_theta=32, s_max_log2=14):
    """Maximise the surplus Eq. (5) over the pure Gaussian family (s, theta).

    Deterministic: a (theta, log2 s) grid including the heterodyne (s = 1)
    and every homodyne angle, then a bounded L-BFGS-B refinement in
    (theta, ln s) from the best grid point and a bounded scalar refinement
    of the homodyne angle; the better of the two is returned as a
    :class:`Measurement`.  Kua-Serafini-Genoni report the heterodyne
    optimal for phase-invariant (squeezed-thermal) states; harvested states
    are not phase-invariant, hence the search.
    """
    V_A, V_B, C = mode_blocks(V_AB)
    omega_B = float(omega_B)
    nu_B = _sym_nu(V_B)
    thetas = np.linspace(0.0, math.pi, int(n_theta), endpoint=False)
    s_grid = [2.0**k for k in range(0, int(s_max_log2) + 1)]

    # heterodyne (theta-independent) is the incumbent; a candidate replaces
    # it only by a resolvable margin, so a pure state (every measurement
    # ties at nu_{B|A} = 1/2) reports the minimum-energy measurement.
    het = gaussian_measurement("heterodyne", gap_A=gap_A)
    best = (_gain_of(V_A, V_B, C, omega_B, nu_B, het), het)
    tie = 1e-13 * max(1.0, omega_B * nu_B)
    for th in thetas:
        for s in s_grid[1:]:
            m = gaussian_measurement("general", s=s, theta=th, gap_A=gap_A)
            g = _gain_of(V_A, V_B, C, omega_B, nu_B, m)
            if g > best[0] + tie:
                best = (g, m)
        m = gaussian_measurement("homodyne", theta=th, gap_A=gap_A)
        g = _gain_of(V_A, V_B, C, omega_B, nu_B, m)
        if g > best[0] + tie:
            best = (g, m)

    # refinement of finite-s candidates in (theta, ln s)
    def neg_gain(x):
        th, y = x
        m = gaussian_measurement("general", s=math.exp(y), theta=th, gap_A=gap_A)
        return -_gain_of(V_A, V_B, C, omega_B, nu_B, m)

    m0 = best[1]
    th0 = m0.theta if m0.kind != "heterodyne" else 0.0
    y0 = 0.0 if math.isinf(m0.s) else math.log(m0.s)
    if math.isinf(m0.s):
        y0 = float(s_max_log2) * LN2
    res = minimize(neg_gain, np.array([th0, y0]), method="L-BFGS-B",
                   bounds=[(th0 - 0.5, th0 + 0.5), (0.0, (s_max_log2 + 4.0) * LN2)])
    if -res.fun > best[0] + tie:
        th, y = res.x
        best = (-float(res.fun), gaussian_measurement("general", s=math.exp(y), theta=th, gap_A=gap_A))

    # refinement of the homodyne angle
    def neg_hom(th):
        return -_gain_of(V_A, V_B, C, omega_B, nu_B,
                         gaussian_measurement("homodyne", theta=th, gap_A=gap_A))

    th_h = thetas[int(np.argmin([neg_hom(t) for t in thetas]))]
    res_h = minimize_scalar(neg_hom, bounds=(th_h - 0.2, th_h + 0.2), method="bounded",
                            options={"xatol": 1e-12})
    if -res_h.fun > best[0] + tie:
        best = (-float(res_h.fun), gaussian_measurement("homodyne", theta=float(res_h.x), gap_A=gap_A))
    return best[1], best[0]


@dataclass
class SurplusResult:
    """Output of :func:`daemonic_surplus` (Eqs. (3)-(5), (8), (9))."""

    E_B: float            # (1/2) Tr(H_BB V_B): Bob's mean energy
    E_local: float        # Eq. (3)
    E_daemonic: float     # Eq. (4)
    surplus: float        # Eq. (5)
    nu_B: float
    nu_B_cond: float
    omega_B: float
    measurement: Measurement
    E_A_meas: float       # Eq. (9): Alice's injection under the Lueders update; inf for homodyne
    landauer_bound: float
    I_AB: float           # nats
    kT_B: float
    V_B_cond: np.ndarray


def daemonic_surplus(V_AB, H_B, measurement="optimal", gap_A=None):
    """Column 1: E_local, E_daemonic and their difference on a two-mode state.

    Parameters
    ----------
    V_AB : (4, 4) block-ordered two-mode covariance (Alice = mode 0).
    H_B : float gap Omega_B (H_BB = diag(Omega_B^2, 1)) or a (2, 2) form.
    measurement : 'optimal' (search of :func:`optimal_measurement`),
        'heterodyne', 'homodyne' (theta = 0), ('homodyne', theta),
        ('general', s, theta) or a :class:`Measurement`.
    gap_A : Alice's oscillator frequency fixing the heterodyne point
        (default: Bob's omega_B).
    """
    V = _check_two_mode(V_AB)
    H_BB = _single_mode_form(H_B)
    omega_B = math.sqrt(float(np.linalg.det(H_BB)))
    gap_A = omega_B if gap_A is None else float(gap_A)
    V_A, V_B, C = mode_blocks(V)
    nu_B = _sym_nu(V_B)
    if isinstance(measurement, str) and measurement == "optimal":
        meas, _ = optimal_measurement(V, omega_B=omega_B, gap_A=gap_A)
    else:
        meas = _as_measurement(measurement, gap_A)
    V_cond, _ = _conditional_pieces(V_A, V_B, C, meas)
    nu_cond = _sym_nu(V_cond)
    E_B = _mode_energy(H_BB, V_B)
    E_local = E_B - omega_B * nu_B
    E_daemonic = E_B - omega_B * nu_cond
    H_AA = np.diag([gap_A * gap_A, 1.0])
    E_A_meas = math.inf if meas.kind == "homodyne" else float(np.trace(H_AA @ meas.sigma))
    bound, I_AB, _, kT = landauer_bound(V, omega_B)
    return SurplusResult(
        E_B=E_B, E_local=E_local, E_daemonic=E_daemonic, surplus=E_daemonic - E_local,
        nu_B=nu_B, nu_B_cond=nu_cond, omega_B=omega_B, measurement=meas,
        E_A_meas=E_A_meas, landauer_bound=bound, I_AB=I_AB, kT_B=kT, V_B_cond=V_cond,
    )


def no_signaling_audit(V_AB, measurements=None, gap_A=1.0, tol=1e-14, strict=True):
    """Standing audit: Bob's unconditional state is measurement-independent.

    For each measurement the unconditional covariance is REBUILT the long
    way — conditional covariance plus the outcome-averaged outer product of
    the conditional mean, E[m_B m_B^T] = C^T G Cov(alpha) G C, G =
    (V_A + sigma_M)^-1 (see :func:`_conditional_pieces`) — and the worst
    deviation across measurements, and from V_B itself, is the violation.  Default measurement set: heterodyne, homodyne at
    theta = 0, pi/4, pi/2, and a squeezed general-dyne.
    """
    V = _check_two_mode(V_AB)
    V_A, V_B, C = mode_blocks(V)
    if measurements is None:
        measurements = [
            gaussian_measurement("heterodyne", gap_A=gap_A),
            gaussian_measurement("homodyne", theta=0.0, gap_A=gap_A),
            gaussian_measurement("homodyne", theta=math.pi / 4, gap_A=gap_A),
            gaussian_measurement("homodyne", theta=math.pi / 2, gap_A=gap_A),
            gaussian_measurement("general", s=7.0, theta=0.3, gap_A=gap_A),
        ]
    recon = []
    labels = []
    for m in measurements:
        meas = _as_measurement(m, gap_A)
        V_cond, mean_cov = _conditional_pieces(V_A, V_B, C, meas)
        recon.append(V_cond + mean_cov)
        labels.append(meas.label())
    worst = 0.0
    for R in recon:
        worst = max(worst, float(np.max(np.abs(R - V_B))))
    for i in range(len(recon)):
        for j in range(i + 1, len(recon)):
            worst = max(worst, float(np.max(np.abs(recon[i] - recon[j]))))
    scale = max(1.0, float(np.max(np.abs(V_B))))
    passed = worst <= tol * scale
    result = AuditResult(
        name="no_signaling",
        passed=passed,
        worst_violation=worst if not passed else 0.0,
        details={"max_deviation": worst, "tol": float(tol), "scale": scale,
                 "measurements": labels, "V_B": V_B},
    )
    if strict and not passed:
        raise AuditViolation(result)
    return result


# --------------------------------------------------------------------------
# Joint ergotropies (comparison columns)
# --------------------------------------------------------------------------


def joint_ergotropy(V_AB, gaps, tail_tol=1e-14, n_max=4000):
    """Ergotropy of the two-mode state under H_A + H_B, two ways.

    ``gaussian``: E - sum_k nu_k^(desc) omega_k^(asc) (Brown-Friis-Huber
    NJP 18, 113028 (2016): the largest symplectic eigenvalue goes to the
    lowest frequency).  ``full``: E - sum_j lambda_j^(desc) eps_j^(asc)
    over the Fock spectrum (Allahverdyan-Balian-Nieuwenhuizen passive
    state) — the Gaussian state's spectrum is the product of two geometric
    distributions with nbar_k = nu_k - 1/2 in its Williamson basis, the
    levels are n_A Omega_A + n_B Omega_B + zero point; truncated at n_max
    per mode with the discarded probability ``tail`` reported (it bounds
    the error times the largest retained level).
    """
    V = _check_two_mode(V_AB)
    Om = np.asarray(gaps, dtype=float)
    if Om.shape != (2,) or np.any(Om <= 0.0):
        raise ValueError(f"gaps must be two positive frequencies, got {gaps}")
    H = np.diag([Om[0] ** 2, Om[1] ** 2, 1.0, 1.0])
    E = 0.5 * float(np.trace(H @ V))
    nus = np.sort(symplectic_eigenvalues(V))[::-1]
    nus = np.maximum(nus, 0.5)
    gaussian = E - float(np.sum(nus * np.sort(Om)))
    nbar = nus - 0.5
    pops = []
    for nb in nbar:
        if nb <= 0.0:
            pops.append(np.array([1.0]))
            continue
        q = nb / (nb + 1.0)
        n_cut = int(min(n_max, max(2, math.ceil(math.log(tail_tol) / math.log(q)) + 1)))
        pops.append((1.0 - q) * q ** np.arange(n_cut))
    lam = np.sort(np.outer(pops[0], pops[1]).ravel())[::-1]
    tail = 1.0 - float(np.sum(lam))
    # The K retained populations go on the K LOWEST levels of the FULL grid
    # n_A Omega_A + n_B Omega_B (not of the truncated population grid: a
    # nearly pure mode has one retained population but its excited levels
    # still exist and are cheap).  Levels below E* = sqrt(4 K Omega_A Omega_B)
    # number about E*^2/(2 Omega_A Omega_B) = 2K, so a grid reaching E* on
    # both axes contains the K lowest.
    K = lam.size
    E_star = math.sqrt(4.0 * K * Om[0] * Om[1]) + Om[0] + Om[1]
    LA = int(math.ceil(E_star / Om[0])) + 1
    LB = int(math.ceil(E_star / Om[1])) + 1
    nA = np.arange(LA)[:, None]
    nB = np.arange(LB)[None, :]
    eps = np.sort((Om[0] * (nA + 0.5) + Om[1] * (nB + 0.5)).ravel())[:K]
    if eps.size < K:  # pragma: no cover - grid sizing guarantees this never fires
        raise RuntimeError("passive-state level grid too small")
    E_pass = float(np.sum(lam * eps))
    return {"gaussian": gaussian, "full": E - E_pass, "E": E, "E_passive_full": E_pass,
            "E_passive_gaussian": E - gaussian, "nu": nus, "tail": tail,
            "truncation": (pops[0].size, pops[1].size)}


# --------------------------------------------------------------------------
# The Hotta reference: weak x_A x_B coupling, joint ground state
# --------------------------------------------------------------------------


def pair_hamiltonian(gaps, g):
    """(K_pair, H) for H = H_A + H_B + g x_A x_B in block ordering.

    K_pair = [[Omega_A^2, g], [g, Omega_B^2]]; H = diag(K_pair, I).  Requires
    g^2 < Omega_A^2 Omega_B^2 (K_pair positive definite, H bounded below).
    """
    Om = np.asarray(gaps, dtype=float)
    if Om.shape != (2,) or np.any(Om <= 0.0):
        raise ValueError(f"gaps must be two positive frequencies, got {gaps}")
    g = float(g)
    if not g * g < (Om[0] * Om[1]) ** 2:
        raise ValueError(f"coupling g={g} makes H unbounded below (need |g| < Omega_A Omega_B)")
    K = np.array([[Om[0] ** 2, g], [g, Om[1] ** 2]])
    H = np.zeros((4, 4))
    H[:2, :2] = K
    H[2:, 2:] = np.eye(2)
    return K, H


def _hotta_extraction(V, gaps, g, meas):
    """Eq. (9)-(10) on a state V under H_g; returns the dict of pieces."""
    Om = np.asarray(gaps, dtype=float)
    V_A, V_B, C = mode_blocks(V)
    H_AA = np.diag([Om[0] ** 2, 1.0])
    H_BB = np.diag([Om[1] ** 2, 1.0])
    V_cond, _ = _conditional_pieces(V_A, V_B, C, meas)
    nu_cond = _sym_nu(V_cond)
    E_A = float(np.trace(H_AA @ meas.sigma))
    E_B_local_op = _mode_energy(H_BB, V_B) + g * float(C[0, 0])   # <H_B + g x_A x_B>
    E_B_corr = E_B_local_op - Om[1] * nu_cond + g * g * float(V_A[0, 0]) / (2.0 * Om[1] ** 2)
    E_B_inj = g * g * float(meas.sigma[0, 0]) / (2.0 * Om[1] ** 2)
    return {"E_A": E_A, "E_B": E_B_corr + E_B_inj, "E_B_corr": E_B_corr, "E_B_inj": E_B_inj,
            "nu_B_cond": nu_cond, "local_energy_op": E_B_local_op, "C_xx": float(C[0, 0])}


def hotta_variant(V_AB, gaps, g, measurement="heterodyne", passivity_kwargs=None, strict=True):
    """Column 2: the true-QET reference and the harvested state under H_g.

    ``reference``: the ground state of H_g = H_A + H_B + g x_A x_B (passivity-
    audited), on which the no-communication extraction is zero and the
    measurement + feedforward gives Alice's injection E_A (Eq. (9)) and
    Bob's teleported E_B (Eq. (10)), with the Hotta inequality E_B <= E_A
    checked.  ``harvested``: the same three numbers on V_AB after the
    coupling is switched on (sudden-switch work W_couple = g <x_A x_B>
    booked), plus Bob's no-communication extraction under H_g
    (:func:`local_extraction_coupled`) and the surplus.  The measurement
    must have a finite pointer covariance (a homodyne injects infinite
    E_A); default heterodyne, E_A = Omega_A.
    """
    V = _check_two_mode(V_AB)
    Om = np.asarray(gaps, dtype=float)
    K_pair, H = pair_hamiltonian(Om, g)
    meas = _as_measurement(measurement, Om[0])
    if meas.kind == "homodyne":
        raise ValueError("the Hotta column needs a finite-strength measurement (E_A finite)")
    g = float(g)

    V_ref = ground_state_cov(K_pair)
    pas = passivity_audit(K_pair, V=V_ref, strict=strict, **(passivity_kwargs or {}))
    ref_local, _, _ = local_extraction_coupled(V_ref, H)
    ref = _hotta_extraction(V_ref, Om, g, meas)
    ref.update({"E_local": ref_local, "passivity": pas,
                "hotta_inequality": bool(ref["E_B"] <= ref["E_A"] + 1e-12),
                "E_ground": 0.5 * float(np.trace(H @ V_ref)), "V": V_ref})
    if strict and not ref["hotta_inequality"]:
        raise AuditViolation(AuditResult(
            name="hotta_inequality", passed=False,
            worst_violation=ref["E_B"] - ref["E_A"], details={"E_A": ref["E_A"], "E_B": ref["E_B"]}))

    H0 = np.diag([Om[0] ** 2, Om[1] ** 2, 1.0, 1.0])
    W_couple = 0.5 * float(np.trace((H - H0) @ V))
    har_local, S_best, _ = local_extraction_coupled(V, H)
    har = _hotta_extraction(V, Om, g, meas)
    har.update({"E_local": har_local, "surplus": har["E_B"] - har_local,
                "W_couple": W_couple, "S_local": S_best})
    return {"g": g, "K_pair": K_pair, "H": H, "measurement": meas,
            "reference": ref, "harvested": har}


# --------------------------------------------------------------------------
# Anchors: two-mode squeezed thermal states in the detectors' units
# --------------------------------------------------------------------------


def two_mode_squeezed_thermal(r, nbar=0.0, gaps=(1.0, 1.0)):
    """Two-mode squeezed thermal state, block ordering, in oscillator units.

    Dimensionless form (vacuum I/2): V_xx = (2N+1)/2 [[c, s], [s, c]],
    V_pp = (2N+1)/2 [[c, -s], [-s, c]], c = cosh 2r, s = sinh 2r; then
    rescaled by the local symplectic diag(Omega^-1/2) (x) diag(Omega^1/2)
    so that r = N = 0 is the detectors' ground state.  E_N = 2r at N = 0.
    """
    Om = np.asarray(gaps, dtype=float)
    c, s = math.cosh(2.0 * r), math.sinh(2.0 * r)
    w = 2.0 * float(nbar) + 1.0
    V = np.zeros((4, 4))
    V[:2, :2] = 0.5 * w * np.array([[c, s], [s, c]])
    V[2:, 2:] = 0.5 * w * np.array([[c, -s], [-s, c]])
    D = np.diag([1.0 / math.sqrt(Om[0]), 1.0 / math.sqrt(Om[1]), math.sqrt(Om[0]), math.sqrt(Om[1])])
    return D @ V @ D


# --------------------------------------------------------------------------
# Ebits
# --------------------------------------------------------------------------


@dataclass
class EbitsResult:
    """Entanglement columns of a two-mode Gaussian state.

    E_N : log-negativity in nats (mp-margin-safe, ``flagged`` carried);
    E_N_bits : the same in bits — an UPPER bound on distillable
        entanglement (Vidal-Werner PRA 65, 032314 (2002));
    hashing_bits : max(0, S_B - S_AB, S_A - S_AB) in bits — the coherent
        information, a rigorous LOWER bound on one-way distillable
        entanglement (Devetak-Winter, Proc. R. Soc. A 461, 207 (2005),
        the hashing inequality); the distillable-entanglement proxy;
    eof_sym_bits : entanglement of formation for SYMMETRIC states
        (det V_A = det V_B), Giedke-Wolf-Krueger-Werner-Cirac PRL 91,
        107901 (2003), Prop. 2 with Eq. (17): f(delta) = c_+ log2 c_+ -
        c_- log2 c_-, c_+- = (delta^-1/2 +- delta^1/2)^2/4, delta = 2 nu~_min
        = exp(-E_N) (their covariance is twice ours); NaN when the state is
        not symmetric;
    distillable : E_N > 0 — every NPT Gaussian state is distillable
        (Giedke-Duan-Cirac-Zoller, Quant. Inf. Comp. 1, 79 (2001));
    I_AB : mutual information (nats).
    """

    E_N: float
    E_N_bits: float
    hashing_bits: float
    eof_sym_bits: float
    symmetric: bool
    distillable: bool
    I_AB: float
    S_A: float
    S_B: float
    S_AB: float
    nu_A: float
    nu_B: float
    nu_pt_min: float
    flagged: bool
    regime: str
    neg: NegativityResult


def _eof_symmetric(delta):
    if delta >= 1.0:
        return 0.0
    cp = (delta ** -0.5 + delta ** 0.5) ** 2 / 4.0
    cm = (delta ** -0.5 - delta ** 0.5) ** 2 / 4.0
    out = cp * math.log2(cp)
    if cm > 0.0:
        out -= cm * math.log2(cm)
    return out


def ebits(V_AB, floor_tol=1e-10, dps=50, sym_tol=1e-9):
    """The ebits column: E_N (nats, bits), hashing lower bound, symmetric EoF."""
    V = _check_two_mode(V_AB)
    neg = harvested_log_negativity(V, floor_tol=floor_tol, dps=dps)
    V_A, V_B, _ = mode_blocks(V)
    nu_A, nu_B = _sym_nu(V_A), _sym_nu(V_B)
    nus = symplectic_eigenvalues(V)
    S_A = entropy_from_nu(np.array([nu_A]))
    S_B = entropy_from_nu(np.array([nu_B]))
    S_AB = entropy_from_nu(nus)
    I_AB = S_A + S_B - S_AB
    hashing = max(0.0, S_B - S_AB, S_A - S_AB) / LN2
    symmetric = abs(nu_A - nu_B) <= sym_tol * max(1.0, nu_A, nu_B)
    eof = _eof_symmetric(math.exp(-neg.E_N)) if symmetric else float("nan")
    return EbitsResult(
        E_N=float(neg.E_N), E_N_bits=float(neg.E_N) / LN2, hashing_bits=float(hashing),
        eof_sym_bits=float(eof), symmetric=bool(symmetric), distillable=bool(neg.E_N > 0.0),
        I_AB=float(I_AB), S_A=float(S_A), S_B=float(S_B), S_AB=float(S_AB),
        nu_A=nu_A, nu_B=nu_B, nu_pt_min=float(neg.min_nu_pt), flagged=bool(neg.flagged),
        regime=neg.regime, neg=neg,
    )


# --------------------------------------------------------------------------
# The harvest: vacuum.detectors 'xx' through vacuum.protocol.run_protocol
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HarvestConfig:
    """Frozen configuration of one harvesting protocol (lattice units).

    Two identical pointlike oscillator detectors of gap ``gap`` on a 1+1
    chain (``N_field`` sites, mass ``mass``, boundary ``bc``), placed
    ``separation`` sites apart and mirror-symmetric about the chain centre
    when ``site_A`` is None (needs N_field - 1 - separation even for exact
    symmetry), coupled through ``coupling`` — ``'xx'`` (amplitude, x_d x_i)
    or ``'xp'`` (derivative, x_d p_i = x_d d_t phi_i; ``attach_detectors``)
    — and switched with lam_d(t) = lam chi_d(t): cos^2 of half-width
    ``T_sw`` (support [0, 2 T_sw] for Alice) or Gaussian of width ``T_sw``
    (effective support [0, 17 T_sw]); Bob's profile is the same shape
    delayed by ``delay`` >= 0 (``delay = 0``: simultaneous switching;
    ``delay = separation``: full light contact on the unit-velocity
    lattice, the configuration where the derivative coupling harvests
    genuinely in causal contact, Teixido-Bonfill & Martin-Martinez, PRD
    110, 105016 (2024)).  The window is the union of both supports; the
    pair is *spacelike* iff every (t, t') pair of the two supports has
    |t - t'| < separation, i.e. iff ``duration`` < separation.  ``T`` is
    the field temperature.  Gate: ``n_segments0`` segments x ``substeps``
    exponential-midpoint substeps at level 0, doubled until every reported
    observable moves < ``conv_tol`` (spec M2.2), at most ``max_halvings``
    times.  The IR side: ``mass = 0`` is admissible on the Dirichlet chain
    (``harmonic_chain_K`` is positive definite there, no zero mode; the
    periodic massless chain is singular and rejected by ``vacuum.core``),
    so the m -> 0 limit is taken on Dirichlet chains of growing N.
    """

    N_field: int = 61
    mass: float = 0.0
    bc: str = "dirichlet"
    separation: int = 12
    site_A: Optional[int] = None
    gap: float = 1.0
    lam: float = 0.3
    switching: str = "cos2"
    T_sw: float = 3.0
    T: float = 0.0
    n_segments0: int = 16
    substeps: int = 4
    conv_tol: float = 1e-9
    max_halvings: int = 12
    ledger_tol: float = DEFAULT_CLOSURE_ATOL
    causality_buffer: float = 15.0
    coupling: str = "xx"
    delay: float = 0.0

    def __post_init__(self):
        if self.coupling not in ("xx", "xp"):
            raise ValueError(f"coupling must be 'xx' or 'xp', got {self.coupling!r}")
        if not float(self.delay) >= 0.0:
            raise ValueError(f"delay must be >= 0, got {self.delay}")
        if float(self.mass) < 0.0:
            raise ValueError(f"mass must be >= 0, got {self.mass}")
        if float(self.mass) == 0.0 and self.bc != "dirichlet":
            raise ValueError("mass = 0 needs Dirichlet boundaries (the periodic massless chain has a "
                             "zero mode; vacuum.core.ground_state_cov rejects it)")

    @property
    def sites(self) -> Tuple[int, int]:
        a = (self.N_field - 1 - self.separation) // 2 if self.site_A is None else int(self.site_A)
        b = a + int(self.separation)
        if not (0 <= a < b < self.N_field):
            raise ValueError(f"detector sites ({a}, {b}) outside the chain 0..{self.N_field - 1}")
        return a, b

    @property
    def mirror_symmetric(self) -> bool:
        a, b = self.sites
        return a + b == self.N_field - 1

    @property
    def t_center(self) -> float:
        return float(self.T_sw) if self.switching == "cos2" else 8.5 * float(self.T_sw)

    @property
    def t_center_B(self) -> float:
        return self.t_center + float(self.delay)

    @property
    def switchings(self):
        """(chi_A, chi_B): Bob's profile is Alice's delayed by ``delay``."""
        return (switching(self.switching, self.T_sw, self.t_center),
                switching(self.switching, self.T_sw, self.t_center_B))

    @property
    def window(self) -> Tuple[float, float]:
        chi_A, chi_B = self.switchings
        return (0.0, max(float(chi_A.support[1]), float(chi_B.support[1])))

    @property
    def light_contact(self) -> bool:
        return abs(float(self.delay) - float(self.separation)) < 1e-12

    @property
    def duration(self) -> float:
        return self.window[1]

    @property
    def spacelike(self) -> bool:
        return self.duration < self.separation

    @property
    def gaps(self) -> Tuple[float, float]:
        return (float(self.gap), float(self.gap))


def field_coupling(cfg):
    """The chain coupling matrix of ``cfg`` (vacuum.core.harmonic_chain_K)."""
    return harmonic_chain_K(int(cfg.N_field), float(cfg.mass), bc=cfg.bc)


@dataclass
class HarvestOutcome:
    """One audited harvest: the detector block, the ledger, the audits."""

    V_AB: np.ndarray
    V: np.ndarray
    W_in: float
    ledger: EnergyLedger
    ledger_result: AuditResult
    audits: Dict[str, Any]
    neg: NegativityResult
    E_N: float
    n_d: np.ndarray
    converged: bool
    halvings: int
    levels_run: List[int]
    n_segments: int
    dt_converged: float
    movement: Optional[float]
    work_movement: Optional[float]
    observables: Dict[str, float]
    causality: AuditResult
    passivity_initial: Optional[AuditResult]
    flagged: bool
    all_audits_passed: bool
    cfg: HarvestConfig
    K_tot0: np.ndarray
    elapsed_s: float


class _LevelSchedule:
    """dt-halving levels with a second-order jump to the predecessor of the
    predicted accepting level (the M2.7 schedule): acceptance is always a
    single-halving comparison at ``conv_tol``, never relaxed."""

    _MAX_JUMP = 10
    _MIN_PREDICT_LEVEL = 2

    def __init__(self, max_level):
        self.max_level = int(max_level)
        self._next = 0

    def __iter__(self):
        while self._next <= self.max_level:
            level = self._next
            self._next = level + 1
            yield level

    def jump(self, level, movement, tol):
        if level < self._MIN_PREDICT_LEVEL:
            return
        if not (movement > 0.0) or not (tol > 0.0) or movement <= tol:
            return
        need = int(math.ceil(math.log(movement / tol) / math.log(4.0)))
        target = level + min(max(need, 1), self._MAX_JUMP)
        self._next = max(self._next, min(target - 1, self.max_level - 1))


def _harvest_level(K, cfg, init, n_segments, gaps, sites, det_modes):
    t0, t1 = cfg.window
    chi_A, chi_B = cfg.switchings
    lam = float(cfg.lam)
    coupling = cfg.coupling

    def H_of_t(t):
        return attach_detectors(K, sites, gaps, (lam * chi_A(t), lam * chi_B(t)), coupling=coupling)

    ts = np.linspace(t0, t1, int(n_segments) + 1)
    steps = [
        GeneratorStep(H=H_of_t, duration=float(ts[i + 1] - ts[i]),
                      n_substeps=int(cfg.substeps), label=f"drive{i}")
        for i in range(int(n_segments))
    ]
    ledger = EnergyLedger(K=init.K_tot, tol=float(cfg.ledger_tol))
    res = run_protocol(
        init.V, init.K_tot, steps, ledger=ledger,
        claim_ground_entry=(float(cfg.T) == 0.0),
        report_modes=[det_modes],
        run_causality=False,   # detector-augmented ordering; the field's audit runs directly below
        ledger_tol=float(cfg.ledger_tol), strict=True,
    )
    N = K.shape[0]
    V_AB = detector_block(res.V, N)
    neg = harvested_log_negativity(V_AB)
    n_d = detector_occupations(res.V, N, np.asarray(gaps))
    V_A, V_B, C = mode_blocks(V_AB)
    het = gaussian_measurement("heterodyne", gap_A=gaps[0])
    V_cond, _ = _conditional_pieces(V_A, V_B, C, het)
    obs = {
        "E_N": float(neg.E_N),
        "margin": float(neg.margin_min),
        "n_d_A": float(n_d[0]),
        "n_d_B": float(n_d[1]),
        "nu_B": _sym_nu(V_B),
        "nu_B_cond_het": _sym_nu(V_cond),
    }
    return res, ledger, V_AB, neg, n_d, obs


def harvest(K, cfg):
    """Run the harvesting protocol of ``cfg`` on the field ``K``, audited.

    initial state (passivity-audited at T = 0, M2.2) -> ``run_protocol``
    over the switching window with the dt-halving gate on the reported
    physical observables (E_N, PT margin, both occupations, nu_B, the
    heterodyne nu_{B|A}) at ``conv_tol`` (spec M2.2's set plus the two
    symplectic eigenvalues the surplus is made of); the drive work's
    movement under the same halving is recorded as ``work_movement`` (the
    ledger closes on the work at every level regardless — closure tests the
    instrumentation, the gate the physics; same convention as
    papers/circuit-qed-noise-thresholds) -> detector block, mp-safe E_N,
    the closed ledger (its ``work`` entries are W_in, the switching work),
    the field's causality audit over the window, and the audit flags.
    """
    t_start = time.time()
    K = np.asarray(K, dtype=float)
    if K.shape != (int(cfg.N_field), int(cfg.N_field)):
        raise ValueError(f"K shape {K.shape} does not match cfg.N_field={cfg.N_field}")
    N = K.shape[0]
    sites = cfg.sites
    gaps = cfg.gaps
    det_modes = (N, N + 1)
    init = initial_state(K, sites, gaps, float(cfg.T), strict=True)

    schedule = _LevelSchedule(int(cfg.max_halvings))
    prev = None
    prev_work = None
    movement = None
    work_movement = None
    converged = False
    levels_run: List[int] = []
    out = None
    n_segments = int(cfg.n_segments0)
    for level in schedule:
        n_segments = int(cfg.n_segments0) * 2**level
        levels_run.append(level)
        out = _harvest_level(K, cfg, init, n_segments, gaps, sites, det_modes)
        obs = out[5]
        work = float(out[0].work_total)
        if prev is not None and levels_run[-2] == level - 1:
            movement = max(abs(obs[k] - prev[k]) for k in obs)
            work_movement = abs(work - prev_work)
            if movement < float(cfg.conv_tol):
                converged = True
                break
            schedule.jump(level, movement, float(cfg.conv_tol))
        prev = dict(obs)
        prev_work = work
    if not converged:
        raise RuntimeError(
            f"harvest: dt-halving gate did not reach {cfg.conv_tol:g} within "
            f"{cfg.max_halvings} halvings (levels {levels_run}, last movement {movement!r})"
        )
    res, ledger, V_AB, neg, n_d, obs = out

    caus = causality_audit(
        K, cfg.duration, v_max=1.0, buffer=float(cfg.causality_buffer),
        bc="open" if cfg.bc == "dirichlet" else "periodic", strict=True,
    )
    ledger_result = res.audits["ledger"]
    all_passed = bool(res.all_passed() and caus.passed
                      and (init.passivity is None or init.passivity.passed))
    return HarvestOutcome(
        V_AB=V_AB, V=res.V, W_in=float(res.work_total), ledger=ledger,
        ledger_result=ledger_result, audits=res.audits, neg=neg, E_N=float(neg.E_N),
        n_d=n_d, converged=converged, halvings=int(levels_run[-1]), levels_run=levels_run,
        n_segments=int(n_segments),
        dt_converged=float(cfg.duration / (n_segments * int(cfg.substeps))),
        movement=movement, work_movement=work_movement, observables=obs, causality=caus,
        passivity_initial=init.passivity,
        flagged=bool(res.flagged or neg.flagged), all_audits_passed=all_passed,
        cfg=cfg, K_tot0=init.K_tot, elapsed_s=time.time() - t_start,
    )


# --------------------------------------------------------------------------
# The composite ledger row
# --------------------------------------------------------------------------

ROW_COLUMNS = (
    # configuration
    "N_field", "mass", "separation", "site_A", "site_B", "gap", "lam", "switching", "T_sw",
    "duration", "spacelike", "T", "mirror_symmetric", "coupling", "delay", "light_contact",
    # stage 1: switching work -> ebits
    "W_in", "E_N", "E_N_bits", "hashing_bits", "eof_sym_bits", "I_AB", "distillable",
    "n_d_A", "n_d_B", "nu_A", "nu_B",
    "ebits_per_work",
    # stage 2: ebits -> energy at Bob (column 1, daemonic surplus, H_B = Bob's oscillator)
    "E_B_mean", "E_local", "E_daemonic", "surplus", "nu_B_cond", "measurement", "meas_s",
    "meas_theta", "E_A_meas",
    "surplus_het", "E_A_meas_het", "surplus_hom", "landauer_bound", "landauer_ratio",
    "joint_ergotropy_gaussian", "joint_ergotropy_full", "joint_tail",
    "exchange_rate", "work_per_ebit", "all_in_rate_het",
    # non-Gaussian measurements on Alice (Fock representation; general ergotropy at Bob)
    "E_local_fock", "gain_fock_pnr", "gain_fock_onoff", "gain_fock_pnr_sq", "fock_sq_s",
    "fock_sq_theta", "fock_over_gaussian", "fock_sq_over_gaussian", "fock_p_click", "fock_cutoff",
    "fock_movement", "fock_trace_deficit", "fock_converged", "fock_local_consistent",
    "surplus_best", "best_family", "exchange_rate_best",
    # column 2: the Hotta reference through a coupling g
    "g", "W_couple", "hg_E_local", "hg_E_daemonic", "hg_surplus",
    "ref_E_A", "ref_E_B", "ref_E_B_corr", "ref_E_B_inj", "ref_ratio", "ref_E_local",
    # audits and provenance
    "ledger_defect", "ledger_passed", "passivity_initial_passed", "passivity_entry_passed",
    "passivity_ref_passed", "precision_passed", "precision_flagged", "mp_margin_flagged",
    "causality_passed", "causality_max_outside", "no_signaling_defect", "no_signaling_passed",
    "hotta_inequality", "surplus_le_full_ergotropy", "surplus_le_landauer",
    "all_audits_passed", "converged", "halvings", "n_segments", "dt_converged", "movement",
    "work_movement", "elapsed_s",
)


@dataclass
class CompositeRow:
    """The full composite row plus the objects behind it."""

    row: Dict[str, Any]
    harvest: HarvestOutcome
    ebits: EbitsResult
    surplus: SurplusResult
    surplus_het: SurplusResult
    surplus_hom: SurplusResult
    hotta: Dict[str, Any]
    joint: Dict[str, Any]
    no_signaling: AuditResult
    fock: Optional[FockConditioningResult] = None
    fock_sq: Optional[Dict[str, Any]] = None

    def as_dict(self):
        return dict(self.row)


def composite_ledger(cfg, K=None, g=None, measurement="optimal", hotta_measurement="heterodyne",
                     no_signaling_tol=1e-14, fock=True, fock_kwargs=None, fock_sq_kwargs=None):
    """W_in -> ebits -> energy at Bob: the L4 row, every audit run.

    ``g`` defaults to 0.1 Omega_A Omega_B (a tenth of the stability limit).
    ``fock=True`` adds the non-Gaussian-measurement columns of
    :mod:`vacuum.composite.fock_conditioning` — photon-number-resolved and
    on/off conditioning on Alice with Bob's general ergotropy, plus the
    PNR gain in the best squeezed number basis — and the best-of-families
    surplus ``surplus_best`` / ``exchange_rate_best``; the Fock cutoff
    ladder is convergence-checked and its record travels with the row.
    """
    t_start = time.time()
    K = field_coupling(cfg) if K is None else np.asarray(K, dtype=float)
    out = harvest(K, cfg)
    V_AB = out.V_AB
    gaps = cfg.gaps
    eb = ebits(V_AB)
    sur = daemonic_surplus(V_AB, gaps[1], measurement=measurement, gap_A=gaps[0])
    sur_het = daemonic_surplus(V_AB, gaps[1], measurement="heterodyne", gap_A=gaps[0])
    sur_hom = daemonic_surplus(V_AB, gaps[1], measurement="homodyne", gap_A=gaps[0])
    ns = no_signaling_audit(
        V_AB, gap_A=gaps[0], tol=no_signaling_tol,
        measurements=[gaussian_measurement("heterodyne", gap_A=gaps[0]),
                      gaussian_measurement("homodyne", theta=0.0, gap_A=gaps[0]),
                      gaussian_measurement("homodyne", theta=math.pi / 2, gap_A=gaps[0]),
                      gaussian_measurement("general", s=9.0, theta=0.7, gap_A=gaps[0]),
                      sur.measurement],
    )
    je = joint_ergotropy(V_AB, gaps)
    g_use = 0.1 * gaps[0] * gaps[1] if g is None else float(g)
    hv = hotta_variant(V_AB, gaps, g_use, measurement=hotta_measurement)
    ref, har = hv["reference"], hv["harvested"]

    fc: Optional[FockConditioningResult] = None
    fsq: Optional[Dict[str, Any]] = None
    if fock:
        fc = fock_conditioning(V_AB, gaps, **(fock_kwargs or {}))
        fsq = optimal_fock_squeezing(V_AB, gaps, **(fock_sq_kwargs or {}))
    W_in = out.W_in
    a, b = cfg.sites
    prec_ok = all(d["result"].passed for d in out.audits["precision"])
    prec_flag = any(bool(d["result"].details.get("flagged", False)) for d in out.audits["precision"])
    pe = out.audits.get("passivity_entry")
    row: Dict[str, Any] = {
        "N_field": int(cfg.N_field), "mass": float(cfg.mass), "separation": int(cfg.separation),
        "site_A": int(a), "site_B": int(b), "gap": float(cfg.gap), "lam": float(cfg.lam),
        "switching": cfg.switching, "T_sw": float(cfg.T_sw), "duration": float(cfg.duration),
        "spacelike": bool(cfg.spacelike), "T": float(cfg.T),
        "mirror_symmetric": bool(cfg.mirror_symmetric), "coupling": cfg.coupling,
        "delay": float(cfg.delay), "light_contact": bool(cfg.light_contact),
        "W_in": W_in, "E_N": eb.E_N, "E_N_bits": eb.E_N_bits, "hashing_bits": eb.hashing_bits,
        "eof_sym_bits": eb.eof_sym_bits, "I_AB": eb.I_AB, "distillable": eb.distillable,
        "n_d_A": float(out.n_d[0]), "n_d_B": float(out.n_d[1]), "nu_A": eb.nu_A, "nu_B": eb.nu_B,
        "ebits_per_work": eb.E_N_bits / W_in if W_in > 0 else float("nan"),
        "E_B_mean": sur.E_B, "E_local": sur.E_local, "E_daemonic": sur.E_daemonic,
        "surplus": sur.surplus, "nu_B_cond": sur.nu_B_cond, "measurement": sur.measurement.label(),
        "meas_s": sur.measurement.s, "meas_theta": sur.measurement.theta, "E_A_meas": sur.E_A_meas,
        "surplus_het": sur_het.surplus, "E_A_meas_het": sur_het.E_A_meas,
        "surplus_hom": sur_hom.surplus,
        "landauer_bound": sur.landauer_bound,
        "landauer_ratio": sur.surplus / sur.landauer_bound if sur.landauer_bound > 0 else float("nan"),
        "joint_ergotropy_gaussian": je["gaussian"], "joint_ergotropy_full": je["full"],
        "joint_tail": je["tail"],
        "exchange_rate": sur.surplus / W_in if W_in > 0 else float("nan"),
        "work_per_ebit": sur.surplus / eb.E_N_bits if eb.E_N_bits > 0 else float("nan"),
        "all_in_rate_het": sur_het.surplus / (W_in + sur_het.E_A_meas),
        "g": g_use, "W_couple": har["W_couple"], "hg_E_local": har["E_local"],
        "hg_E_daemonic": har["E_B"], "hg_surplus": har["surplus"],
        "ref_E_A": ref["E_A"], "ref_E_B": ref["E_B"], "ref_E_B_corr": ref["E_B_corr"],
        "ref_E_B_inj": ref["E_B_inj"], "ref_ratio": ref["E_B"] / ref["E_A"],
        "ref_E_local": ref["E_local"],
        "ledger_defect": float(out.ledger_result.details["defect"]),
        "ledger_passed": bool(out.ledger_result.passed),
        "passivity_initial_passed": (None if out.passivity_initial is None
                                     else bool(out.passivity_initial.passed)),
        "passivity_entry_passed": None if pe is None else bool(pe.passed),
        "passivity_ref_passed": bool(ref["passivity"].passed),
        "precision_passed": bool(prec_ok), "precision_flagged": bool(prec_flag),
        "mp_margin_flagged": bool(out.neg.flagged),
        "causality_passed": bool(out.causality.passed),
        "causality_max_outside": float(out.causality.details["max_abs_commutator_outside"]),
        "no_signaling_defect": float(ns.details["max_deviation"]),
        "no_signaling_passed": bool(ns.passed),
        "hotta_inequality": bool(ref["hotta_inequality"]),
        "surplus_le_full_ergotropy": bool(sur.E_daemonic <= je["full"] + 1e-9),
        "surplus_le_landauer": bool(sur.surplus <= sur.landauer_bound + 1e-12),
        "converged": bool(out.converged), "halvings": int(out.halvings),
        "n_segments": int(out.n_segments), "dt_converged": float(out.dt_converged),
        "movement": float(out.movement) if out.movement is not None else float("nan"),
        "work_movement": (float(out.work_movement) if out.work_movement is not None
                          else float("nan")),
    }
    nan = float("nan")
    if fc is not None:
        g_pnr, g_off, g_sq = fc.gain_pnr, fc.gain_onoff, fsq["gain"]
        families = {"gaussian": sur.surplus, "pnr": g_pnr, "onoff": g_off, "pnr_squeezed": g_sq}
        best_family = max(families, key=families.get)
        gauss = sur.surplus
        row.update({
            "E_local_fock": fc.E_local_full, "gain_fock_pnr": g_pnr, "gain_fock_onoff": g_off,
            "gain_fock_pnr_sq": g_sq, "fock_sq_s": fsq["s"], "fock_sq_theta": fsq["theta"],
            "fock_over_gaussian": g_pnr / gauss if gauss > 0 else nan,
            "fock_sq_over_gaussian": g_sq / gauss if gauss > 0 else nan,
            "fock_p_click": fc.p_click, "fock_cutoff": int(fc.cutoff), "fock_movement": fc.movement,
            "fock_trace_deficit": fc.trace_deficit, "fock_converged": bool(fc.converged),
            "fock_local_consistent": bool(abs(fc.E_local_full - sur.E_local)
                                          <= 1e-8 * max(1.0, abs(sur.E_local))),
            "surplus_best": families[best_family], "best_family": best_family,
            "exchange_rate_best": families[best_family] / W_in if W_in > 0 else nan,
        })
    else:
        row.update({k: nan for k in ("E_local_fock", "gain_fock_pnr", "gain_fock_onoff",
                                     "gain_fock_pnr_sq", "fock_sq_s", "fock_sq_theta",
                                     "fock_over_gaussian", "fock_sq_over_gaussian", "fock_p_click",
                                     "fock_movement", "fock_trace_deficit")})
        row.update({"fock_cutoff": 0, "fock_converged": False, "fock_local_consistent": False,
                    "surplus_best": sur.surplus, "best_family": "gaussian",
                    "exchange_rate_best": row["exchange_rate"]})
    row["all_audits_passed"] = bool(
        out.all_audits_passed and ns.passed and ref["passivity"].passed and ref["hotta_inequality"]
        and row["surplus_le_landauer"]
    )
    row["elapsed_s"] = time.time() - t_start
    return CompositeRow(row=row, harvest=out, ebits=eb, surplus=sur, surplus_het=sur_het,
                        surplus_hom=sur_hom, hotta=hv, joint=je, no_signaling=ns, fock=fc,
                        fock_sq=fsq)
