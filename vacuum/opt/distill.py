"""Distillation back end: BBPSSW and DEJMPS recurrence on two-qubit states.

The vacuum-to-Bell-pair compiler (PLAN.md, Layer 6): feed a harvested
two-detector state through a standard recurrence distillation protocol and
compute the yield of usable Bell pairs, then divide by the switching work
the harvesting ledger charged — the exchange rate printed at the end.

Two protocols, transcribed with equation numbers (fetched 2026-09-02 from
the arXiv/ar5iv renderings; equation numbers as printed there):

BBPSSW — C. H. Bennett, G. Brassard, S. Popescu, B. Schumacher, J. A.
Smolin, W. K. Wootters, "Purification of noisy entanglement and faithful
teleportation via noisy channels", Phys. Rev. Lett. 76, 722 (1996)
[arXiv:quant-ph/9511027].  Werner state of fidelity F, their Eq. (4):

    W_F = F |Psi-><Psi-| + (1-F)/3 (|Psi+><Psi+| + |Phi+><Phi+| + |Phi-><Phi-|),

step A1 converts the mostly-Psi- Werner state to the analogous
mostly-Phi+ state by a unilateral sigma_y rotation, A2 is the bilateral
XOR (CNOT), A3 measures the target pair and keeps the source pair when the
outcomes are parallel; after a twirl back to Werner form the fidelity is
[their Eq. (7)]

    F' = [F^2 + (1/9)(1-F)^2] / [F^2 + (2/3) F (1-F) + (5/9)(1-F)^2],

with the denominator the probability that the pair passes the test.

DEJMPS — D. Deutsch, A. Ekert, R. Jozsa, C. Macchiavello, S. Popescu,
A. Sanpera, "Quantum privacy amplification and the security of quantum
cryptography over noisy channels", Phys. Rev. Lett. 77, 2818 (1996)
[arXiv:quant-ph/9604039].  Bell basis {phi+, psi-, psi+, phi-} with
|phi+-> = (|00> +- |11>)/sqrt2, |psi+-> = (|01> +- |10>)/sqrt2, and a
Bell-diagonal state with diagonal {A, B, C, D} in that order.  Alice
applies a pi/2 rotation about x to her qubits, Bob the inverse (-pi/2);
bilateral CNOT (one pair the controls, the other the targets); the pair
is kept iff the target measurement outcomes coincide.  One round maps
[their Eq. (7)]

    A' = (A^2 + B^2)/N,  B' = 2 C D / N,  C' = (C^2 + D^2)/N,  D' = 2 A B / N,
    N  = (A + B)^2 + (C + D)^2   (probability that the outcomes coincide).

Both recurrences are implemented twice: as the closed-form maps above and
as the CODED protocol on the dense 16x16 two-pair state (explicit local
rotations, bilateral CNOT, target measurement, post-selection, partial
trace).  The anchor test demands the two agree to 1e-12 — the recurrence
formulas are then a transcription check on the coded operations, not an
assumption.

Yield accounting.  A distillation tree of n rounds consumes 2^n input
pairs and returns one pair with probability prod_r p_r, so the expected
number of distilled pairs PER INPUT PAIR — i.e. per harvesting protocol
run, since one run yields one detector pair — is

    bell_pairs_per_run = prod_r p_r / 2^n.

From Gaussian detectors to qubits.  The harvesting engine's detectors are
oscillators; a two-qubit state is obtained by a STATED projection:

- :func:`bell_diagonal_from_pair_state` reads the second-order two-qubit
  UDW state (``vacuum.detectors.udw.UDWPairState``, PKMM PRD 92, 064042
  (2015), Eq. (13)) and twirls it to Bell-diagonal form after a local
  phase that aligns the pair term M with |phi+>;
- :func:`bell_diagonal_from_covariance` applies the leading-order
  oscillator <-> two-level map derived and gate-tested in
  tests/test_gate_crossvalidation.py (n_nu = P_nu, m = M, c = conj(C))
  to the detector-block covariance V_AB, then the same twirl.

The twirled coefficients are a bona fide probability vector even though
the second-order X-state itself is not PSD at that order (its |ee|
population is O(lambda^4)); the twirl discards exactly the non-PSD part.
Distillation needs A > 1/2 (a state with |phi+> fidelity <= 1/2 is not
distillable by these protocols), which for the harvested X-state is
A - 1/2 = |M| - (P_A + P_B)/2 = N^(2), PKMM's negativity estimator: the
compiler distills iff the detectors harvested.

Conventions: two-qubit basis {|00>, |01>, |10>, |11>} with qubit A the
first factor; hbar = 1.  All functions are pure over their arguments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BELL_ORDER",
    "bell_states",
    "bell_diagonal_coefficients",
    "twirl_to_bell_diagonal",
    "bell_diagonal_state",
    "werner_state",
    "bbpssw_recurrence",
    "dejmps_recurrence",
    "bbpssw_coded_round",
    "dejmps_coded_round",
    "DistillResult",
    "distill_yield",
    "oscillator_moments",
    "x_state_from_moments",
    "bell_diagonal_from_covariance",
    "bell_diagonal_from_pair_state",
    "ExchangeRate",
    "vacuum_to_bell_rate",
]

#: DEJMPS labeling: (A, B, C, D) multiply (phi+, psi-, psi+, phi-).
BELL_ORDER = ("phi+", "psi-", "psi+", "phi-")

_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


# --------------------------------------------------------------------------
# Bell basis and the twirl
# --------------------------------------------------------------------------


def bell_states():
    """(4, 4) complex array whose columns are |phi+>, |psi->, |psi+>, |phi-> (BELL_ORDER)."""
    s = 1.0 / math.sqrt(2.0)
    phi_p = s * np.array([1, 0, 0, 1], dtype=complex)
    phi_m = s * np.array([1, 0, 0, -1], dtype=complex)
    psi_p = s * np.array([0, 1, 1, 0], dtype=complex)
    psi_m = s * np.array([0, 1, -1, 0], dtype=complex)
    return np.stack([phi_p, psi_m, psi_p, phi_m], axis=1)


def bell_diagonal_coefficients(rho):
    """(A, B, C, D) = diagonal of rho in the Bell basis (BELL_ORDER)."""
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError(f"expected a (4, 4) two-qubit state, got {rho.shape}")
    Bm = bell_states()
    return np.real(np.einsum("ik,ij,jk->k", Bm.conj(), rho, Bm))


def bell_diagonal_state(coeffs):
    """Bell-diagonal density matrix sum_k coeffs[k] |bell_k><bell_k|."""
    c = np.asarray(coeffs, dtype=float)
    if c.shape != (4,):
        raise ValueError(f"need 4 Bell-diagonal coefficients, got shape {c.shape}")
    Bm = bell_states()
    return (Bm * c) @ Bm.conj().T


def twirl_to_bell_diagonal(rho):
    """Project rho onto its Bell-diagonal part (the random bilateral twirl)."""
    return bell_diagonal_state(bell_diagonal_coefficients(rho))


def werner_state(F, target="phi+"):
    """BBPSSW Eq. (4) Werner state of fidelity F with respect to ``target``.

    ``target='psi-'`` is the printed Eq. (4) (singlet weight F); 'phi+' is
    the mostly-phi+ form their step A1 produces, which is what the coded
    bilateral XOR round consumes.
    """
    F = float(F)
    if not 0.0 <= F <= 1.0:
        raise ValueError(f"fidelity must lie in [0, 1], got {F}")
    if target not in BELL_ORDER:
        raise ValueError(f"target must be one of {BELL_ORDER}, got {target!r}")
    c = np.full(4, (1.0 - F) / 3.0)
    c[BELL_ORDER.index(target)] = F
    return bell_diagonal_state(c)


# --------------------------------------------------------------------------
# The literature recurrences (closed forms)
# --------------------------------------------------------------------------


def bbpssw_recurrence(F):
    """BBPSSW Eq. (7): (F', p_pass) for a Werner input of fidelity F."""
    F = float(F)
    num = F * F + (1.0 - F) ** 2 / 9.0
    den = F * F + (2.0 / 3.0) * F * (1.0 - F) + (5.0 / 9.0) * (1.0 - F) ** 2
    return num / den, den


def dejmps_recurrence(A, B, C, D):
    """DEJMPS Eq. (7): (A', B', C', D', N) for a Bell-diagonal input (BELL_ORDER)."""
    A, B, C, D = (float(x) for x in (A, B, C, D))
    N = (A + B) ** 2 + (C + D) ** 2
    return (A * A + B * B) / N, 2.0 * C * D / N, (C * C + D * D) / N, 2.0 * A * B / N, N


# --------------------------------------------------------------------------
# The coded protocols (dense 16x16 two-pair algebra)
# --------------------------------------------------------------------------


def _op4(op, pos):
    """Embed a single-qubit operator on qubit ``pos`` of a 4-qubit register."""
    ops = [_I2] * 4
    ops[pos] = op
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def _cnot4(control, target):
    """CNOT on a 4-qubit register (qubit ordering A1, B1, A2, B2)."""
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    return _op4(P0, control) + _op4(P1, control) @ _op4(_SX, target)


def _two_pair_state(rho):
    """rho (x) rho on qubits (A1, B1, A2, B2)."""
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError(f"expected a (4, 4) two-qubit state, got {rho.shape}")
    return np.kron(rho, rho)


def _bilateral_cnot_and_select(rho16):
    """Bilateral CNOT (A1->A2, B1->B2), measure A2, B2 in z, keep coincident outcomes.

    Returns (rho_source (4x4, normalized), p_keep).
    """
    U = _cnot4(0, 2) @ _cnot4(1, 3)
    r = U @ rho16 @ U.conj().T
    P00 = _op4(np.array([[1, 0], [0, 0]], dtype=complex), 2) @ _op4(
        np.array([[1, 0], [0, 0]], dtype=complex), 3
    )
    P11 = _op4(np.array([[0, 0], [0, 1]], dtype=complex), 2) @ _op4(
        np.array([[0, 0], [0, 1]], dtype=complex), 3
    )
    kept = P00 @ r @ P00 + P11 @ r @ P11
    p = float(np.real(np.trace(kept)))
    # partial trace over the target pair (qubits 2, 3)
    k = kept.reshape(2, 2, 2, 2, 2, 2, 2, 2)  # (a1 b1 a2 b2 | a1' b1' a2' b2')
    src = np.einsum("abcdefcd->abef", k).reshape(4, 4)
    return src / p, p


def bbpssw_coded_round(rho):
    """One BBPSSW round on a (mostly-phi+) two-qubit state, explicitly.

    Bilateral XOR, target measurement, keep parallel outcomes, then the
    twirl back to Werner form (the coefficients other than the phi+
    fidelity equalized).  Returns (rho_out, F_out, p_keep).
    """
    src, p = _bilateral_cnot_and_select(_two_pair_state(rho))
    F = float(bell_diagonal_coefficients(src)[0])
    return werner_state(F, target="phi+"), F, p


def dejmps_coded_round(rho):
    """One DEJMPS round on a Bell-diagonal two-qubit state, explicitly.

    Alice: R_x(+pi/2) = exp(-i pi sigma_x / 4) on her qubits; Bob: the
    inverse R_x(-pi/2); bilateral CNOT; keep coincident target outcomes.
    Returns (rho_out, (A', B', C', D'), p_keep).
    """
    Ra = np.cos(np.pi / 4) * _I2 - 1.0j * np.sin(np.pi / 4) * _SX
    Rb = Ra.conj().T
    R = _op4(Ra, 0) @ _op4(Ra, 2) @ _op4(Rb, 1) @ _op4(Rb, 3)
    r = _two_pair_state(rho)
    r = R @ r @ R.conj().T
    src, p = _bilateral_cnot_and_select(r)
    coeffs = bell_diagonal_coefficients(src)
    return src, tuple(float(c) for c in coeffs), p


# --------------------------------------------------------------------------
# Yields
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DistillResult:
    """Yield of an n-round recurrence tree.

    Iterating yields ``(bell_pairs_per_run, fidelity)`` — the README's
    ``distill_yield(...) -> (bell_pairs_per_run, fidelity)`` signature,
    literally — while the record also carries the per-round provenance.
    """

    bell_pairs_per_run: float   # expected distilled pairs per input pair = prod p_r / 2^n
    fidelity: float             # phi+ fidelity after the last round
    success_probability: float  # prod_r p_r (one pair out of the 2^n-pair tree)
    n_rounds: int
    fidelities: Tuple[float, ...]        # F after each round (index 0 = input)
    pass_probabilities: Tuple[float, ...]
    coefficients: Tuple[Tuple[float, float, float, float], ...]  # (A,B,C,D) per stage
    protocol: str
    method: str

    def __iter__(self):
        yield self.bell_pairs_per_run
        yield self.fidelity


def _coerce_bell_diagonal(rho_or_coeffs, gaps=None):
    """Accept a (4,4) state, a 4-vector of Bell coefficients, or a (4,4) covariance."""
    arr = np.asarray(rho_or_coeffs)
    if arr.shape == (4,):
        c = np.asarray(arr, dtype=float)
    elif arr.shape == (4, 4) and np.iscomplexobj(arr):
        c = bell_diagonal_coefficients(arr)
    elif arr.shape == (4, 4) and np.allclose(arr, arr.T) and float(np.trace(arr)) > 1.5:
        # a real symmetric 4x4 with trace > 3/2 is a two-mode covariance
        # (a density matrix has trace 1); needs the gaps for the projection
        if gaps is None:
            raise ValueError("a two-mode covariance V_AB needs gaps=(Omega_A, Omega_B)")
        c = bell_diagonal_from_covariance(arr, gaps)
    elif arr.shape == (4, 4):
        c = bell_diagonal_coefficients(arr.astype(complex))
    else:
        raise ValueError(
            f"expected a (4,4) density matrix / covariance or 4 Bell coefficients, got {arr.shape}"
        )
    if np.any(c < -1e-12) or abs(np.sum(c) - 1.0) > 1e-9:
        raise ValueError(f"Bell-diagonal coefficients are not a probability vector: {c}")
    return np.clip(c, 0.0, None) / np.sum(c)


def distill_yield(rho_or_V_AB, protocol="DEJMPS", n_rounds=1, *, gaps=None,
                  method="recurrence", target_fidelity=None, max_rounds=64):
    """Bell-pair yield of ``n_rounds`` of recurrence distillation.

    Parameters
    ----------
    rho_or_V_AB : (4,4) density matrix, 4 Bell coefficients, or (4,4) covariance
        The input pair.  A density matrix is twirled to Bell-diagonal form
        first (both protocols act on the Bell-diagonal part); a covariance
        goes through :func:`bell_diagonal_from_covariance` (``gaps`` needed).
    protocol : {'DEJMPS', 'BBPSSW'}
    n_rounds : int
        Rounds of the recurrence tree (ignored when ``target_fidelity`` is
        given: rounds are then added until the phi+ fidelity reaches it,
        up to ``max_rounds``).
    method : {'recurrence', 'coded'}
        Closed-form maps (Eq. (7) of each paper) or the dense coded protocol.
    target_fidelity : float, optional
        Stop when F >= target (n_rounds then reports the rounds used).

    Returns
    -------
    DistillResult  — ``(bell_pairs_per_run, fidelity)`` on tuple-unpacking.
    """
    protocol = str(protocol).upper()
    if protocol not in ("DEJMPS", "BBPSSW"):
        raise ValueError(f"protocol must be 'DEJMPS' or 'BBPSSW', got {protocol!r}")
    if method not in ("recurrence", "coded"):
        raise ValueError(f"method must be 'recurrence' or 'coded', got {method!r}")
    c = _coerce_bell_diagonal(rho_or_V_AB, gaps=gaps)
    if protocol == "BBPSSW":
        # the Werner form: keep the phi+ fidelity, equalize the rest
        F = float(c[0])
        c = np.full(4, (1.0 - F) / 3.0)
        c[0] = F
    coeffs = [tuple(float(x) for x in c)]
    fids = [float(c[0])]
    probs: list = []
    rounds_target = int(n_rounds) if target_fidelity is None else int(max_rounds)
    if target_fidelity is not None and fids[0] >= float(target_fidelity):
        rounds_target = 0
    r = 0
    while r < rounds_target:
        A, B, C, D = coeffs[-1]
        if A <= 0.5 and target_fidelity is not None:
            break  # not distillable; more rounds only degrade
        if protocol == "DEJMPS":
            if method == "recurrence":
                A2, B2, C2, D2, p = dejmps_recurrence(A, B, C, D)
            else:
                _, (A2, B2, C2, D2), p = dejmps_coded_round(bell_diagonal_state([A, B, C, D]))
            coeffs.append((A2, B2, C2, D2))
        else:
            if method == "recurrence":
                F2, p = bbpssw_recurrence(A)
            else:
                _, F2, p = bbpssw_coded_round(werner_state(A, target="phi+"))
            coeffs.append((F2, (1 - F2) / 3, (1 - F2) / 3, (1 - F2) / 3))
        probs.append(float(p))
        fids.append(float(coeffs[-1][0]))
        r += 1
        if target_fidelity is not None and fids[-1] >= float(target_fidelity):
            break
    n = len(probs)
    succ = float(np.prod(probs)) if probs else 1.0
    # a pair whose phi+ fidelity is <= 1/2 is not a distilled Bell pair
    distilled = fids[-1] > 0.5
    return DistillResult(
        bell_pairs_per_run=(succ / (2.0 ** n)) if distilled else 0.0,
        fidelity=fids[-1],
        success_probability=succ,
        n_rounds=n,
        fidelities=tuple(fids),
        pass_probabilities=tuple(probs),
        coefficients=tuple(coeffs),
        protocol=protocol,
        method=method,
    )


# --------------------------------------------------------------------------
# Gaussian two-mode block -> two-qubit Bell-diagonal state
# --------------------------------------------------------------------------


def oscillator_moments(V_AB, gaps):
    """n_A, n_B, m = <a_A a_B>, c = <a_A^dag a_B> of a two-mode block (block ordering).

    a_nu = (sqrt(Omega_nu) x_nu + i p_nu / sqrt(Omega_nu)) / sqrt 2, mean-zero
    state; the inversion of the rescaled second moments is the one derived
    in tests/test_gate_crossvalidation.py (oscillator_moments).
    """
    g = np.asarray(gaps, dtype=float)
    if g.shape != (2,):
        raise ValueError(f"gaps must be a pair, got shape {g.shape}")
    V = np.asarray(V_AB, dtype=float)
    if V.shape != (4, 4):
        raise ValueError(f"V_AB must be a two-mode (4, 4) block, got {V.shape}")
    Dm = np.diag(np.concatenate([np.sqrt(g), 1.0 / np.sqrt(g)]))
    W = Dm @ V @ Dm
    xx, pp, xp = W[:2, :2], W[2:, 2:], W[:2, 2:]
    return {
        "n_A": 0.5 * (xx[0, 0] + pp[0, 0] - 1.0),
        "n_B": 0.5 * (xx[1, 1] + pp[1, 1] - 1.0),
        "m": complex(0.5 * (xx[0, 1] - pp[0, 1]), 0.5 * (xp[0, 1] + xp[1, 0])),
        "c": complex(0.5 * (xx[0, 1] + pp[0, 1]), 0.5 * (xp[0, 1] - xp[1, 0])),
    }


def x_state_from_moments(n_A, n_B, m, c):
    """Second-order two-qubit X-state (PKMM Eq. (13) layout) from the moments.

    Basis {|00>, |01>, |10>, |11>} = {gg, g_A e_B, e_A g_B, ee} with A first:
    rho_00 = 1 - n_A - n_B, rho_10,10 = n_A, rho_01,01 = n_B,
    rho_10,01 = C = conj(c), rho_11,00 = M = m, rho_11,11 = 0 at this order.
    """
    r = np.zeros((4, 4), dtype=complex)
    r[0, 0] = 1.0 - n_A - n_B
    r[2, 2] = n_A
    r[1, 1] = n_B
    r[2, 1] = np.conj(c)
    r[1, 2] = c
    r[3, 0] = m
    r[0, 3] = np.conj(m)
    return r


def _align_and_twirl(rho):
    """Local phase on B aligning M = rho_11,00 to |phi+>, then the Bell twirl."""
    rho = np.asarray(rho, dtype=complex)
    phase = np.angle(rho[3, 0])
    U = np.kron(_I2, np.diag([1.0, np.exp(-1.0j * phase)]))
    r = U @ rho @ U.conj().T
    c = bell_diagonal_coefficients(r)
    return np.clip(c, 0.0, None)


def bell_diagonal_from_covariance(V_AB, gaps):
    """Bell-diagonal (A, B, C, D) of the two-level projection of a detector block.

    Leading-order oscillator <-> qubit map (Gate A) into the X-state, local
    phase aligning the pair term with |phi+>, Bell twirl.  A - 1/2 equals
    PKMM's negativity estimator |m| - (n_A + n_B)/2.
    """
    mom = oscillator_moments(V_AB, gaps)
    return _align_and_twirl(x_state_from_moments(mom["n_A"], mom["n_B"], mom["m"], mom["c"]))


def bell_diagonal_from_pair_state(state):
    """Bell-diagonal (A, B, C, D) of a ``vacuum.detectors.udw.UDWPairState`` (its ``rho``)."""
    return _align_and_twirl(np.asarray(state.rho, dtype=complex))


# --------------------------------------------------------------------------
# The exchange rate
# --------------------------------------------------------------------------


class ExchangeRate(NamedTuple):
    rate: float                 # distilled Bell pairs (ebits) per unit switching work
    bell_pairs_per_run: float
    fidelity: float
    work: float
    n_rounds: int
    E_N: float
    bell_coefficients: Tuple[float, float, float, float]
    distill: DistillResult
    harvest: Any


def vacuum_to_bell_rate(protocol_cfg, *, protocol="DEJMPS", target_fidelity=0.99,
                        max_rounds=64, projection="covariance"):
    """Ebits of distilled Bell pairs per unit switching work (ledger work column).

    ``protocol_cfg`` is either a ``vacuum.detectors.HarvestResult`` (an
    already audited harvesting run) or a dict of keyword arguments for
    ``vacuum.detectors.run_harvesting`` (``K``, ``sites_or_profiles``,
    ``gaps``, ``lambda_of_t``, ``ts``, plus any of its options), which is
    then run — passivity audit, dt-halving gate, closing ledger and the
    mp-margin rule included.  The detector block is projected to a
    two-qubit Bell-diagonal state (``projection='covariance'``, the Gate-A
    leading-order map; or ``'pair_state'`` with a ``UDWPairState`` supplied
    under ``protocol_cfg['pair_state']``), distilled with ``protocol`` until
    ``target_fidelity`` (or ``max_rounds``), and the expected number of
    distilled pairs per harvesting run is divided by the run's drive work.

    A non-distillable state (phi+ fidelity <= 1/2, i.e. no harvested
    negativity at leading order) returns rate 0 with n_rounds 0.
    """
    from vacuum.detectors import run_harvesting  # numpy stack, read-only use

    if hasattr(protocol_cfg, "V_AB") and hasattr(protocol_cfg, "work"):
        harvest = protocol_cfg
        cfg: Dict[str, Any] = {}
    else:
        cfg = dict(protocol_cfg)
        cfg.pop("pair_state", None) if projection != "pair_state" else None
        pair_state = cfg.pop("pair_state", None)
        harvest = run_harvesting(**cfg)
        if projection == "pair_state":
            cfg["pair_state"] = pair_state
    gaps = np.asarray(getattr(harvest, "K_tot0")[-2:, -2:].diagonal(), dtype=float) ** 0.5
    if projection == "covariance":
        coeffs = bell_diagonal_from_covariance(harvest.V_AB, gaps)
    elif projection == "pair_state":
        ps = cfg.get("pair_state") if cfg else None
        if ps is None:
            raise ValueError("projection='pair_state' needs protocol_cfg['pair_state']")
        coeffs = bell_diagonal_from_pair_state(ps)
    else:
        raise ValueError(f"projection must be 'covariance' or 'pair_state', got {projection!r}")
    if coeffs[0] <= 0.5:
        d = DistillResult(0.0, float(coeffs[0]), 1.0, 0, (float(coeffs[0]),), (), (tuple(float(x) for x in coeffs),),
                          str(protocol).upper(), "recurrence")
    else:
        d = distill_yield(coeffs, protocol=protocol, target_fidelity=target_fidelity,
                          max_rounds=max_rounds)
    work = float(harvest.work)
    rate = d.bell_pairs_per_run / work if work > 0.0 else float("nan")
    return ExchangeRate(
        rate=rate, bell_pairs_per_run=d.bell_pairs_per_run, fidelity=d.fidelity, work=work,
        n_rounds=d.n_rounds, E_N=float(harvest.E_N), bell_coefficients=tuple(float(x) for x in coeffs),
        distill=d, harvest=harvest,
    )
