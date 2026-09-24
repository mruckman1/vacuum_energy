"""Hotta's minimal quantum energy teleportation (QET) model — exact, dim 4.

Two qubits A (Alice) and B (Bob), dense 4x4 linear algebra, no Gaussian
machinery. Everything is transcribed from

    M. Hotta, "Quantum energy teleportation: an introductory review",
    arXiv:1101.3954, Sec. 3 "Minimal QET Model", Eqs. (5)-(11) and (14),

with the equation number cited next to each expression below. The same model
is the one run on IBM superconducting hardware by K. Ikeda, Phys. Rev.
Applied 20, 024051 (2023) [arXiv:2301.02666] (cited for context; every
formula in this file is transcribed from the review, not from that paper).

The model (h > 0, k > 0 constants; |+-> denote sigma_z eigenstates with
eigenvalue +-1; s = sqrt(h^2 + k^2) throughout):

    H = H_A + H_B + V,
    H_A = h sigma_z^A + h^2/s          [Eq. (5)]
    H_B = h sigma_z^B + h^2/s          [Eq. (6)]
    V   = 2k sigma_x^A sigma_x^B + 2k^2/s   [Eq. (7)]

The constant offsets fix E_ground = <g|H|g> = 0 exactly, with ground state
(stated after Eq. (7) of the review)

    |g> = (1/sqrt(2)) sqrt(1 - h/s) |+>_A |+>_B
        - (1/sqrt(2)) sqrt(1 + h/s) |->_A |->_B.

Protocol (review Sec. 3, items I-III):
  I.   Alice measures sigma_x^A projectively, POVM projectors
       P_A(mu) = (1 + mu sigma_x^A)/2, mu = +-1, injecting
       E_A = h^2/s   [Eq. (8)].
  II.  Alice sends the classical bit mu to Bob.
  III. Bob applies the conditional rotation
       U_B(mu, theta) = I_B cos(theta) - i mu sigma_y^B sin(theta)
       and extracts, on ensemble average,
       E_B(theta) = (1/s) [ h k sin(2 theta)
                            - (h^2 + 2k^2)(1 - cos(2 theta)) ]   [Eq. (14)],
       maximized at cos(2 theta*) = (h^2+2k^2)/D, sin(2 theta*) = h k/D
       with D = sqrt((h^2+2k^2)^2 + h^2 k^2)   [Eqs. (9)-(10)], giving
       E_B_max = ((h^2+2k^2)/s) [ sqrt(1 + h^2 k^2/(h^2+2k^2)^2) - 1 ]
       [Eq. (11)].

Entanglement bookkeeping uses the Wootters concurrence
C(rho) = max(0, l1 - l2 - l3 - l4), l_i the descending square roots of the
eigenvalues of rho (sy x sy) rho* (sy x sy)
[W. K. Wootters, Phys. Rev. Lett. 80, 2245 (1998)].

Conventions: hbar = 1 (docs/API.md); tensor products are A (x) B (qubit A is
the first factor); basis ordering {|++>, |+->, |-+>, |-->}. All functions are
pure over their array arguments — no hidden state, no input mutation
(Layer-6 JAX-mirror discipline).
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize_scalar

from vacuum.audits import AuditResult, AuditViolation

__all__ = [
    "hamiltonian",
    "hamiltonian_terms",
    "ground_state",
    "measurement_projector",
    "bob_rotation",
    "e_a_closed",
    "e_b_closed",
    "theta_optimal",
    "e_b_optimal",
    "run_protocol",
    "optimize_e_b",
    "bob_reduced_state",
    "trace_distance",
    "concurrence",
    "hotta_sweep",
]

# Pauli matrices in the sigma_z eigenbasis |+> = (1,0), |-> = (0,1).
_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _check_hk(h, k):
    h = float(h)
    k = float(k)
    if not (h > 0.0 and k > 0.0):
        raise ValueError(f"the minimal model needs h > 0 and k > 0, got h={h}, k={k}")
    return h, k


def _check_mu(mu):
    mu = int(mu)
    if mu not in (+1, -1):
        raise ValueError(f"measurement outcome mu must be +1 or -1, got {mu}")
    return mu


def hamiltonian_terms(h, k):
    """The three pieces (H_A, H_B, V) of the minimal-model Hamiltonian.

    arXiv:1101.3954 Eqs. (5)-(7), each including its constant offset
    (h^2/s, h^2/s, 2k^2/s with s = sqrt(h^2+k^2)); the offsets sum to
    2(h^2+k^2)/s = 2s, exactly cancelling the ground energy -2s of the
    offset-free operator so that E_ground = 0.

    Returns
    -------
    (H_A, H_B, V) : three (4, 4) complex Hermitian ndarrays in the
        {|++>, |+->, |-+>, |-->} basis, A the first tensor factor.
    """
    h, k = _check_hk(h, k)
    s = np.hypot(h, k)
    I4 = np.eye(4, dtype=complex)
    H_A = h * np.kron(_SZ, _I2) + (h * h / s) * I4          # Eq. (5)
    H_B = h * np.kron(_I2, _SZ) + (h * h / s) * I4          # Eq. (6)
    V = 2.0 * k * np.kron(_SX, _SX) + (2.0 * k * k / s) * I4  # Eq. (7)
    return H_A, H_B, V


def hamiltonian(h, k):
    """Total H = H_A + H_B + V of arXiv:1101.3954 Eqs. (5)-(7), as (4, 4)."""
    H_A, H_B, V = hamiltonian_terms(h, k)
    return H_A + H_B + V


def ground_state(h, k):
    """Closed-form ground state |g> of the minimal model, E_g = 0.

    arXiv:1101.3954, the (unnumbered) expression following Eq. (7):

        |g> = (1/sqrt(2)) sqrt(1 - h/s) |++>
            - (1/sqrt(2)) sqrt(1 + h/s) |-->,   s = sqrt(h^2 + k^2).

    Returns a (4,) complex unit vector in the {|++>, |+->, |-+>, |-->} basis.
    """
    h, k = _check_hk(h, k)
    s = np.hypot(h, k)
    g = np.zeros(4, dtype=complex)
    g[0] = np.sqrt(1.0 - h / s) / np.sqrt(2.0)   # |++> amplitude
    g[3] = -np.sqrt(1.0 + h / s) / np.sqrt(2.0)  # |--> amplitude
    return g


def measurement_projector(mu):
    """Alice's POVM projector P_A(mu) = (1 + mu sigma_x^A)/2, embedded as 4x4.

    arXiv:1101.3954 Sec. 3, protocol item I (projective sigma_x measurement
    on A with outcome mu = +-1). Returns P_A(mu) (x) I_B.
    """
    mu = _check_mu(mu)
    return np.kron((_I2 + mu * _SX) / 2.0, _I2)


def bob_rotation(mu, theta):
    """Bob's conditional rotation U_B(mu, theta), embedded as 4x4.

    arXiv:1101.3954 Sec. 3, protocol item III:
        U_B(mu, theta) = I_B cos(theta) - i mu sigma_y^B sin(theta).
    Returns I_A (x) U_B(mu, theta); unitary for real theta.
    """
    mu = _check_mu(mu)
    theta = float(theta)
    U_B = np.cos(theta) * _I2 - 1.0j * mu * np.sin(theta) * _SY
    return np.kron(_I2, U_B)


# ---------------------------------------------------------------------------
# closed forms (the transcription anchors)
# ---------------------------------------------------------------------------


def e_a_closed(h, k):
    """Energy injected by Alice's measurement: E_A = h^2/sqrt(h^2+k^2).

    arXiv:1101.3954 Eq. (8).
    """
    h, k = _check_hk(h, k)
    return h * h / np.hypot(h, k)


def e_b_closed(h, k, theta):
    """Ensemble-average energy Bob extracts with angle theta.

    arXiv:1101.3954 Eq. (14):
        E_B(theta) = (1/s) [ h k sin(2 theta)
                             - (h^2 + 2 k^2)(1 - cos(2 theta)) ],
    s = sqrt(h^2 + k^2). Positive E_B means energy left the qubit system
    into Bob's device.
    """
    h, k = _check_hk(h, k)
    s = np.hypot(h, k)
    t2 = 2.0 * float(theta)
    return (h * k * np.sin(t2) - (h * h + 2.0 * k * k) * (1.0 - np.cos(t2))) / s


def theta_optimal(h, k):
    """The angle maximizing E_B(theta).

    From arXiv:1101.3954 Eqs. (9)-(10):
        cos(2 theta*) = (h^2 + 2k^2)/D,  sin(2 theta*) = h k/D,
        D = sqrt((h^2 + 2k^2)^2 + h^2 k^2),
    i.e. theta* = (1/2) atan2(h k, h^2 + 2k^2) in (0, pi/4).
    """
    h, k = _check_hk(h, k)
    return 0.5 * np.arctan2(h * k, h * h + 2.0 * k * k)


def e_b_optimal(h, k):
    """The theta-optimal teleported energy.

    arXiv:1101.3954 Eq. (11):
        E_B_max = ((h^2 + 2k^2)/s) [ sqrt(1 + h^2 k^2/(h^2 + 2k^2)^2) - 1 ],
    s = sqrt(h^2 + k^2). Equals e_b_closed(h, k, theta_optimal(h, k)).
    """
    h, k = _check_hk(h, k)
    s = np.hypot(h, k)
    q = h * h + 2.0 * k * k
    r = (h * k) / q
    return (q / s) * (np.sqrt(1.0 + r * r) - 1.0)


# ---------------------------------------------------------------------------
# the coded protocol (dense 4x4, difference-form energy bookkeeping)
# ---------------------------------------------------------------------------


def _energy(rho, H):
    """Re Tr(rho H) for Hermitian H (the real part strips roundoff phase)."""
    return float(np.real(np.trace(rho @ H)))


def run_protocol(h, k, theta, channel_open=True):
    """Run the full minimal QET protocol and return its energy ledger.

    Steps (arXiv:1101.3954 Sec. 3, items I-III): start in |g>; Alice measures
    sigma_x^A (projectors P_A(mu), Eq. above); the classical outcome mu
    travels to Bob; Bob applies U_B(mu, theta). All energies are computed in
    difference form from the dense 4x4 state, never from the closed forms.

    Parameters
    ----------
    h, k : float
        Model constants (> 0).
    theta : float
        Bob's rotation angle.
    channel_open : bool
        True (default): Bob conditions on Alice's actual outcome mu.
        False: the classical channel is cut — Bob receives an independent
        fair coin mu' instead, and the returned E_B is the average over
        both mu and mu'. This is the entanglement-as-fuel control: with no
        classical information the cross term of Eq. (14) averages away and
        no positive E_B survives at any theta.

    Returns
    -------
    dict with keys
        'E_ground'  : <g|H|g> (must be ~0 by the Eq. (5)-(7) offsets),
        'E_A'       : E_meas - E_ground (measurement-injected energy),
        'E_B'       : E_meas - E_final (energy extracted by Bob's rotation),
        'E_meas'    : ensemble energy after step I,
        'E_final'   : ensemble energy after step III,
        'rho_meas'  : (4,4) post-measurement ensemble state,
        'rho_final' : (4,4) post-rotation ensemble state,
        'outcomes'  : per-mu dict: probability p, normalized post-measurement
                      pure state 'state', per-outcome energy 'E_meas',
        'h', 'k', 'theta', 'channel_open' : the inputs, echoed.
    """
    h, k = _check_hk(h, k)
    theta = float(theta)
    H = hamiltonian(h, k)
    g = ground_state(h, k)
    rho_g = np.outer(g, g.conj())
    E_ground = _energy(rho_g, H)

    # step I: projective sigma_x^A measurement (unnormalized branches)
    outcomes = {}
    rho_meas = np.zeros((4, 4), dtype=complex)
    branches = {}
    for mu in (+1, -1):
        P = measurement_projector(mu)
        branch = P @ rho_g @ P               # weight p(mu) carried in the trace
        p = float(np.real(np.trace(branch)))
        branches[mu] = branch
        rho_meas = rho_meas + branch
        outcomes[mu] = {
            "p": p,
            "state": (P @ g) / np.sqrt(p),
            "E_meas": _energy(branch, H) / p,
        }
    E_meas = _energy(rho_meas, H)

    # steps II + III: classical bit to Bob, conditional rotation
    rho_final = np.zeros((4, 4), dtype=complex)
    if channel_open:
        for mu in (+1, -1):
            U = bob_rotation(mu, theta)
            rho_final = rho_final + U @ branches[mu] @ U.conj().T
    else:
        # channel cut: Bob's bit mu' is an independent fair coin
        for mu, mu_prime in itertools.product((+1, -1), repeat=2):
            U = bob_rotation(mu_prime, theta)
            rho_final = rho_final + 0.5 * (U @ branches[mu] @ U.conj().T)
    E_final = _energy(rho_final, H)

    return {
        "E_ground": E_ground,
        "E_A": E_meas - E_ground,
        "E_B": E_meas - E_final,
        "E_meas": E_meas,
        "E_final": E_final,
        "rho_meas": rho_meas,
        "rho_final": rho_final,
        "outcomes": outcomes,
        "h": h,
        "k": k,
        "theta": theta,
        "channel_open": bool(channel_open),
    }


def optimize_e_b(h, k, channel_open=True, xatol=1e-10):
    """Numerically sweep theta for the maximal extracted E_B.

    Independent of the closed forms: bounded scalar minimization of
    -E_B(theta) from :func:`run_protocol` over theta in (0, pi/2), which
    contains the analytic optimum theta* in (0, pi/4). Returns
    (theta_best, E_B_best).
    """
    h, k = _check_hk(h, k)

    def neg_e_b(theta):
        return -run_protocol(h, k, theta, channel_open=channel_open)["E_B"]

    res = minimize_scalar(
        neg_e_b, bounds=(0.0, np.pi / 2.0), method="bounded",
        options={"xatol": float(xatol)},
    )
    return float(res.x), -float(res.fun)


# ---------------------------------------------------------------------------
# no-signaling and entanglement bookkeeping
# ---------------------------------------------------------------------------


def _ptrace_a(rho):
    """Partial trace over qubit A of a (4,4) matrix in the A(x)B basis."""
    r = np.asarray(rho, dtype=complex).reshape(2, 2, 2, 2)
    return np.einsum("ijik->jk", r)


def bob_reduced_state(h, k, alice_measured):
    """Bob's reduced density matrix before he receives the classical bit.

    alice_measured=False: rho_B = Tr_A |g><g|.
    alice_measured=True:  rho_B = Tr_A sum_mu P_A(mu)|g><g|P_A(mu).

    No-signaling demands the two be identical (Alice's local measurement is
    invisible to Bob until the classical bit arrives) — the standing
    causality anchor of the milestone.
    """
    h, k = _check_hk(h, k)
    g = ground_state(h, k)
    rho_g = np.outer(g, g.conj())
    if not alice_measured:
        return _ptrace_a(rho_g)
    rho = np.zeros((4, 4), dtype=complex)
    for mu in (+1, -1):
        P = measurement_projector(mu)
        rho = rho + P @ rho_g @ P
    return _ptrace_a(rho)


def trace_distance(rho, sigma):
    """T(rho, sigma) = (1/2) ||rho - sigma||_1 for Hermitian matrices."""
    diff = np.asarray(rho, dtype=complex) - np.asarray(sigma, dtype=complex)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(diff))))


def concurrence(state_or_rho):
    """Wootters concurrence of a two-qubit state (vector or density matrix).

    C(rho) = max(0, l1 - l2 - l3 - l4), with l_i the descending square roots
    of the eigenvalues of rho rho~, rho~ = (sy x sy) rho* (sy x sy)
    [W. K. Wootters, Phys. Rev. Lett. 80, 2245 (1998)]. C = 0 iff separable;
    for a pure state a|++> + b|--> it reduces to 2|ab|.

    Numerics: a pure state (1D input) uses the exact pure-state form
    C = |<psi| sy x sy |psi*>| = 2 |det c| (c the 2x2 coefficient matrix,
    Wootters Eq. (2) with the spin-flipped state) — machine-exact, no
    eigensolve. A density matrix uses the Hermitian route: l_i are the
    eigenvalues of the PSD matrix sqrt(rho) rho~ sqrt(rho), taken through
    eigh with a clip of roundoff-negative eigenvalues before each sqrt.
    """
    arr = np.asarray(state_or_rho, dtype=complex)
    yy = np.kron(_SY, _SY)
    if arr.ndim == 1:
        if arr.shape != (4,):
            raise ValueError(f"expected a 2-qubit state, got shape {arr.shape}")
        # pure state: C = |<psi| (sy x sy) |psi*>|, exact
        return float(abs(arr.conj() @ yy @ arr.conj()))
    if arr.shape != (4, 4):
        raise ValueError(f"expected a 2-qubit state, got shape {arr.shape}")
    rho = arr
    rho_tilde = yy @ rho.conj() @ yy
    w, U = np.linalg.eigh((rho + rho.conj().T) / 2.0)
    sqrt_rho = (U * np.sqrt(np.clip(w, 0.0, None))) @ U.conj().T
    M = sqrt_rho @ rho_tilde @ sqrt_rho
    lam2 = np.linalg.eigvalsh((M + M.conj().T) / 2.0)
    lam = np.sqrt(np.clip(lam2, 0.0, None))
    lam = np.sort(lam)[::-1]
    return float(max(0.0, lam[0] - lam[1] - lam[2] - lam[3]))


# ---------------------------------------------------------------------------
# standing audit
# ---------------------------------------------------------------------------


def hotta_sweep(h_vals, k_vals, tol=1e-12, strict=True, xatol=1e-10):
    """Standing audit: E_A >= E_B(theta*) at every grid point.

    Runs the coded protocol with a numerical theta sweep (never the closed
    forms) at every (h, k) in the Cartesian grid and demands
    E_A - E_B_max >= -tol. QET never returns more energy than the
    measurement injected (arXiv:1101.3954 Sec. 3 discussion around
    Eqs. (8) and (11)); a violation is an anomaly-protocol event by
    definition, so strict mode raises :class:`AuditViolation`.

    Parameters
    ----------
    h_vals, k_vals : sequences of float (> 0)
        Grid axes.
    tol : float
        Roundoff allowance on the margin E_A - E_B (default 1e-12).
    strict : bool
        Raise on violation (default) instead of returning a failed result.
    xatol : float
        Angle tolerance for the per-point theta optimization.

    Returns
    -------
    AuditResult
        ``worst_violation`` = max(0, -min margin). ``details['rows']`` holds
        one dict per grid point: h, k, E_A, E_B (numerically optimal),
        theta (argmax), margin = E_A - E_B; plus 'min_margin', 'argmin'
        (the (h, k) achieving it), 'tol', and the grid axes.
    """
    h_vals = [float(h) for h in h_vals]
    k_vals = [float(k) for k in k_vals]
    rows = []
    for h, k in itertools.product(h_vals, k_vals):
        theta_best, e_b_best = optimize_e_b(h, k, xatol=xatol)
        e_a = run_protocol(h, k, theta_best)["E_A"]
        rows.append({
            "h": h, "k": k, "E_A": e_a, "E_B": e_b_best,
            "theta": theta_best, "margin": e_a - e_b_best,
        })
    margins = np.array([row["margin"] for row in rows])
    i_min = int(np.argmin(margins))
    min_margin = float(margins[i_min])
    passed = min_margin >= -tol
    result = AuditResult(
        name="hotta_sweep",
        passed=passed,
        worst_violation=max(0.0, -min_margin),
        details={
            "rows": rows,
            "min_margin": min_margin,
            "argmin": (rows[i_min]["h"], rows[i_min]["k"]),
            "h_vals": h_vals,
            "k_vals": k_vals,
            "tol": float(tol),
        },
    )
    if strict and not passed:
        raise AuditViolation(result)
    return result
