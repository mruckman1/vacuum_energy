"""QET on transverse-field Ising chains — exact diagonalization (M3.2).

Open chain of L qubits, transverse-field Ising model (TFIM)

    H = -J sum_{i=0}^{L-2} sigma^x_i sigma^x_{i+1} - g sum_{i=0}^{L-1} sigma^z_i,

quantum-critical at g = J [P. Pfeuty, Ann. Phys. 57, 79 (1970): the exact
solution of the 1d TFIM places the critical point at transverse field equal
to the Ising coupling]. Local energy density h_i with the bond terms split
symmetrically between their two sites,

    h_i = -g sigma^z_i - (J/2) [sigma^x_{i-1} sigma^x_i
                                + sigma^x_i sigma^x_{i+1}],

edge sites keeping only the bonds that exist, so that sum_i h_i = H exactly.
Every reported energy density is in DIFFERENCE FORM <h_i> - <h_i>_GS: the
QET dip is tiny against the extensive ground energy, and the difference form
is what survives roundoff (spec: docs/PLAN_LAYERS_2_3.md, M3.2).

Protocol — spin-chain QET after M. Hotta, "Quantum Energy Teleportation in
Spin Chain Systems", J. Phys. Soc. Jpn. 78, 034001 (2009) [arXiv:0803.0348],
with the conditional-rotation conventions of the minimal model
(vacuum.qet.hotta, arXiv:1101.3954):

  I.   Prepare the chain ground state |g>.
  II.  Alice measures sigma^x at site A projectively (Kraus projectors
       P_A(mu) = (1 + mu sigma^x_A)/2, renormalized branches). P_A(mu)
       commutes with every bond term, so the measurement injects energy
       only through the local field: E_A = g <sigma^z_A>_GS exactly (the
       ensemble dephases sigma^z_A to zero).
  III. The classical outcome mu travels to Bob at site B.
  IV.  Bob applies the conditional single-site rotation
       U_B(mu, theta) = I cos(theta) - i mu sigma^y_B sin(theta)
       with theta optimized numerically PER OUTCOME, and extracts
       E_B = sum_mu p(mu) [<H>_meas,mu - <H>_final,mu] >= 0.

Only the protocol *structure* is taken from the papers above; every number
this module produces is computed by its own sparse/dense linear algebra —
no closed-form values are transcribed.

Conventions: hbar = 1 (docs/API.md); site 0 is the leftmost tensor factor;
single-site basis ordering (|sigma^z=+1>, |sigma^z=-1>). All functions are
pure over their array arguments — no hidden state, no input mutation
(Layer-6 JAX-mirror discipline).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import eigsh, expm_multiply

__all__ = [
    "tfim_hamiltonian",
    "tfim_local_energy_ops",
    "tfim_ground_state",
    "energy_profile",
    "apply_site_gate",
    "measure_site_x",
    "bob_rotation_gate",
    "local_hamiltonian",
    "optimize_rotation",
    "rotation_optimum_closed",
    "run_chain_qet",
    "chain_distance_sweep",
    "evolve_dip_profiles",
    "reduced_density",
    "premeasurement_reduced_states",
]

# single-site matrices in the sigma_z eigenbasis |0> = |sz=+1>, |1> = |sz=-1>
_ID2 = np.eye(2)
_SX = np.array([[0.0, 1.0], [1.0, 0.0]])
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]])
# -i sigma_y = sigma_x sigma_z is real, so the whole protocol stays real
_MISY = np.array([[0.0, -1.0], [1.0, 0.0]])


def _check_L(L):
    L = int(L)
    if L < 3:
        raise ValueError(f"need at least 3 sites for a QET chain, got L={L}")
    return L


def _check_site(L, site, name="site"):
    site = int(site)
    if not 0 <= site < L:
        raise ValueError(f"{name}={site} outside chain of L={L} sites")
    return site


def _check_gJ(g, J):
    g = float(g)
    J = float(J)
    if not (g > 0.0 and J > 0.0):
        raise ValueError(f"need g > 0 and J > 0, got g={g}, J={J}")
    return g, J


def _kron_chain(L, ops):
    """I x ... x ops[i] x ... x I as a sparse CSR matrix (site 0 leftmost).

    ops: dict {site: (2, 2) ndarray}; sites not listed carry the identity.
    """
    out = sp.identity(1, format="csr")
    for i in range(L):
        out = sp.kron(out, sp.csr_matrix(ops.get(i, _ID2)), format="csr")
    return out


def tfim_hamiltonian(L, g, J=1.0):
    """Sparse H = -J sum sx_i sx_{i+1} - g sum sz_i on the open chain."""
    L = _check_L(L)
    g, J = _check_gJ(g, J)
    H = sp.csr_matrix((2**L, 2**L))
    for i in range(L - 1):
        H = H - J * _kron_chain(L, {i: _SX, i + 1: _SX})
    for i in range(L):
        H = H - g * _kron_chain(L, {i: _SZ})
    return H.tocsr()


def tfim_local_energy_ops(L, g, J=1.0):
    """The symmetric-bond-split energy densities h_i, i = 0..L-1 (sparse).

    h_i = -g sz_i - (J/2)(sx_{i-1} sx_i + sx_i sx_{i+1}), edge terms dropped;
    sum_i h_i = tfim_hamiltonian(L, g, J) exactly (tested).
    """
    L = _check_L(L)
    g, J = _check_gJ(g, J)
    bonds = [_kron_chain(L, {i: _SX, i + 1: _SX}) for i in range(L - 1)]
    h_ops = []
    for i in range(L):
        h = -g * _kron_chain(L, {i: _SZ})
        if i > 0:
            h = h - (J / 2.0) * bonds[i - 1]
        if i < L - 1:
            h = h - (J / 2.0) * bonds[i]
        h_ops.append(h.tocsr())
    return h_ops


def tfim_ground_state(L, g, J=1.0):
    """(E0, psi0) from sparse eigsh (k=1, 'SA'), deterministic start vector.

    The start vector is the uniform superposition, which has nonzero overlap
    with the ground state (all off-diagonal elements of H are <= 0, so the
    ground state is positive in this basis within its parity sector).
    Returns a real, normalized (2**L,) vector.
    """
    L = _check_L(L)
    g, J = _check_gJ(g, J)
    H = tfim_hamiltonian(L, g, J)
    dim = 2**L
    v0 = np.full(dim, 1.0 / np.sqrt(dim))
    vals, vecs = eigsh(H, k=1, which="SA", v0=v0)
    psi = np.real(vecs[:, 0])
    psi = psi / np.linalg.norm(psi)
    return float(vals[0]), psi


def _expect(op, psi):
    """Re <psi| op |psi> for a sparse Hermitian op (real part strips roundoff)."""
    return float(np.real(np.vdot(psi, op @ psi)))


def energy_profile(psi, h_ops):
    """<h_i> for each local energy density, as an (L,) float array."""
    return np.array([_expect(h, psi) for h in h_ops])


def apply_site_gate(psi, L, site, gate):
    """Apply a (2, 2) gate at `site` to a (2**L,) state vector. Pure.

    Site 0 is the leftmost tensor factor: index n has the bit for site i at
    position L-1-i, i.e. psi.reshape((2,)*L)[b_0, ..., b_{L-1}].
    """
    L = _check_L(L)
    site = _check_site(L, site)
    gate = np.asarray(gate)
    left = 2**site
    right = 2 ** (L - site - 1)
    psi3 = np.asarray(psi).reshape(left, 2, right)
    return np.einsum("ab,lbr->lar", gate, psi3).reshape(-1)


def measure_site_x(psi, L, site):
    """Projective sigma^x measurement at `site`: Kraus update + renormalize.

    P(mu) = (1 + mu sigma^x)/2, mu = +-1. Returns
    {mu: {'p': probability, 'state': normalized post-measurement vector}}.
    """
    out = {}
    for mu in (+1, -1):
        P = (_ID2 + mu * _SX) / 2.0
        branch = apply_site_gate(psi, L, site, P)
        p = float(np.real(np.vdot(branch, branch)))
        if p <= 0.0:
            raise ValueError(f"outcome mu={mu} has zero probability")
        out[mu] = {"p": p, "state": branch / np.sqrt(p)}
    return out


def bob_rotation_gate(mu, theta):
    """U_B(mu, theta) = I cos(theta) - i mu sigma^y sin(theta), real (2, 2).

    Same conditional-rotation convention as the minimal model
    (vacuum.qet.hotta.bob_rotation, arXiv:1101.3954 Sec. 3); -i sigma^y is
    the real matrix [[0, -1], [1, 0]].
    """
    mu = int(mu)
    if mu not in (+1, -1):
        raise ValueError(f"measurement outcome mu must be +1 or -1, got {mu}")
    theta = float(theta)
    return np.cos(theta) * _ID2 + mu * np.sin(theta) * _MISY


def local_hamiltonian(h_ops, B):
    """sum of h_j over j in {B-1, B, B+1} (clipped to the chain).

    Contains every term of H touched by a single-site operation at B, so
    energy differences under Bob's rotation computed against it equal the
    differences of the full <H> — with far better roundoff behavior.
    """
    L = len(h_ops)
    B = _check_site(L, B, "B")
    h_loc = h_ops[B].copy()
    if B > 0:
        h_loc = h_loc + h_ops[B - 1]
    if B < L - 1:
        h_loc = h_loc + h_ops[B + 1]
    return h_loc.tocsr()


def _extracted(psi_mu, L, B, h_loc, e_before, mu, theta):
    rotated = apply_site_gate(psi_mu, L, B, bob_rotation_gate(mu, theta))
    return e_before - _expect(h_loc, rotated)


def optimize_rotation(psi_mu, L, B, h_loc, mu, xatol=1e-10):
    """Numerically optimal (theta, extracted energy) for one branch.

    Bounded scalar maximization of the extracted energy over
    theta in (-pi/2, pi/2) — U_B has period pi up to a global sign, so this
    interval covers all distinct rotations. The extracted energy at the
    optimum is >= 0 (theta = 0 is 'do nothing').
    """
    e_before = _expect(h_loc, psi_mu)

    def neg(theta):
        return -_extracted(psi_mu, L, B, h_loc, e_before, mu, theta)

    res = minimize_scalar(
        neg, bounds=(-np.pi / 2.0, np.pi / 2.0), method="bounded",
        options={"xatol": float(xatol)},
    )
    return float(res.x), -float(res.fun)


def rotation_optimum_closed(psi_mu, L, B, h_loc, mu):
    """Closed-form (theta*, dE*) for one branch, via the trigonometric form.

    U_B(mu, theta) conjugates the (Pauli-linear) local Hamiltonian into
    dE(theta) = b (1 - cos 2theta) + c sin 2theta, so two evaluations at
    theta = +-pi/4 fix (b, c) and the maximum is b + sqrt(b^2 + c^2) at
    2 theta* = pi - atan2(c, b) (mod 2 pi, folded into (-pi/2, pi/2]).
    Cross-validates :func:`optimize_rotation` to machine precision.
    """
    e_before = _expect(h_loc, psi_mu)
    f_plus = _extracted(psi_mu, L, B, h_loc, e_before, mu, np.pi / 4.0)
    f_minus = _extracted(psi_mu, L, B, h_loc, e_before, mu, -np.pi / 4.0)
    b = 0.5 * (f_plus + f_minus)
    c = 0.5 * (f_plus - f_minus)
    theta_star = 0.5 * (np.pi - np.arctan2(c, b))
    if theta_star > np.pi / 2.0:
        theta_star -= np.pi
    return float(theta_star), float(b + np.hypot(b, c))


def _protocol_on_ground(L, g, J, A, B, E0, gs, h_ops, xatol,
                        theta_method="numeric"):
    """The QET protocol on a prebuilt ground state. Internal work-horse."""
    A = _check_site(L, A, "A")
    B = _check_site(L, B, "B")
    if A == B:
        raise ValueError("Alice and Bob must sit on different sites")
    if theta_method not in ("numeric", "closed"):
        raise ValueError(f"theta_method must be 'numeric' or 'closed', "
                         f"got {theta_method!r}")
    profile_gs = energy_profile(gs, h_ops)
    branches = measure_site_x(gs, L, A)
    h_loc = local_hamiltonian(h_ops, B)

    profile_meas = np.zeros(L)
    profile_final = np.zeros(L)
    outcomes = {}
    E_B = 0.0
    for mu, br in branches.items():
        p, psi_mu = br["p"], br["state"]
        if theta_method == "closed":
            theta, dE = rotation_optimum_closed(psi_mu, L, B, h_loc, mu)
        else:
            theta, dE = optimize_rotation(psi_mu, L, B, h_loc, mu, xatol=xatol)
        psi_final = apply_site_gate(psi_mu, L, B, bob_rotation_gate(mu, theta))
        profile_meas = profile_meas + p * energy_profile(psi_mu, h_ops)
        profile_final = profile_final + p * energy_profile(psi_final, h_ops)
        E_B += p * dE
        outcomes[mu] = {
            "p": p,
            "theta": theta,
            "dE": dE,
            "state_meas": psi_mu,
            "state_final": psi_final,
        }

    dip_meas = profile_meas - profile_gs    # difference form, always
    dip_final = profile_final - profile_gs
    E_A = float(np.sum(dip_meas))
    return {
        "L": L, "g": g, "J": J, "A": A, "B": B,
        "E_gs": E0,
        "profile_gs": profile_gs,
        "E_A": E_A,
        "E_B": float(E_B),
        "ratio": float(E_B) / E_A,
        "dip_meas": dip_meas,
        "dip_final": dip_final,
        "outcomes": outcomes,
    }


def run_chain_qet(L, g, A, B, J=1.0, xatol=1e-10, ground=None, h_ops=None,
                  theta_method="numeric"):
    """Run the full chain-QET protocol; return its ledger (see module doc).

    Parameters
    ----------
    L, g, J : chain size and couplings (g = J is criticality).
    A, B : Alice's and Bob's sites (0-based, distinct).
    xatol : angle tolerance of the per-outcome rotation optimization.
    ground : optional precomputed (E0, psi) from :func:`tfim_ground_state`.
    h_ops : optional precomputed :func:`tfim_local_energy_ops` output.
    theta_method : 'numeric' (default — the spec's per-outcome numerical
        optimization) or 'closed' (:func:`rotation_optimum_closed`, exact).
        The two agree to machine precision (tested); 'closed' is what the
        ED<->DMRG comparison uses so the residual measures the tensor
        network rather than two optimizers' xatol.

    Returns
    -------
    dict with the inputs echoed plus: 'E_gs', 'profile_gs' (ground <h_i>),
    'E_A' (measurement-injected, = sum of dip_meas), 'E_B' (ensemble energy
    Bob extracts), 'ratio' = E_B/E_A, 'dip_meas'/'dip_final' (<h_i> minus
    ground value after step II / step IV), and per-outcome 'outcomes'
    {mu: p, theta, dE, state_meas, state_final}.
    """
    L = _check_L(L)
    g, J = _check_gJ(g, J)
    if ground is None:
        ground = tfim_ground_state(L, g, J)
    E0, gs = ground
    if h_ops is None:
        h_ops = tfim_local_energy_ops(L, g, J)
    return _protocol_on_ground(L, g, J, A, B, E0, gs, h_ops, xatol,
                               theta_method=theta_method)


def chain_distance_sweep(L, g, A, B_list, J=1.0, xatol=1e-10,
                         theta_method="numeric"):
    """QET ledgers for one Alice and a list of Bob sites (shared ground state).

    Returns {'E_A', 'rows': [...]} where each row carries B, d = |A - B|,
    E_B, ratio = E_B/E_A, dip_min = min_i dip_final_i, and the per-outcome
    angles. The exchange-rate curve E_B_max/E_A vs distance is
    [row['ratio'] for row in rows].
    """
    L = _check_L(L)
    g, J = _check_gJ(g, J)
    ground = tfim_ground_state(L, g, J)
    h_ops = tfim_local_energy_ops(L, g, J)
    rows = []
    E_A = None
    for B in B_list:
        led = _protocol_on_ground(L, g, J, A, B, ground[0], ground[1], h_ops,
                                  xatol, theta_method=theta_method)
        E_A = led["E_A"]
        rows.append({
            "B": int(B),
            "d": abs(int(B) - int(A)),
            "E_A": led["E_A"],
            "E_B": led["E_B"],
            "ratio": led["ratio"],
            "dip_min": float(np.min(led["dip_final"])),
            "theta": {mu: o["theta"] for mu, o in led["outcomes"].items()},
        })
    return {"E_A": E_A, "rows": rows, "L": L, "g": g, "J": J, "A": int(A)}


def evolve_dip_profiles(ledger, ts):
    """Free evolution of the post-protocol dip profile: (len(ts), L) array.

    Each measurement branch's final pure state evolves under exp(-i H t)
    (sparse expm_multiply, stepping between the sorted times); row j is the
    ensemble <h_i>(ts[j]) - <h_i>_GS. Free evolution conserves the total,
    sum_i dip_i(t) = E_A - E_B for every t (tested to 1e-9).
    """
    ts = [float(t) for t in ts]
    if any(t2 < t1 for t1, t2 in zip(ts, ts[1:])):
        raise ValueError("ts must be sorted ascending")
    L, g, J = ledger["L"], ledger["g"], ledger["J"]
    H = tfim_hamiltonian(L, g, J)
    h_ops = tfim_local_energy_ops(L, g, J)
    profile_gs = ledger["profile_gs"]

    states = {mu: o["state_final"].astype(complex)
              for mu, o in ledger["outcomes"].items()}
    probs = {mu: o["p"] for mu, o in ledger["outcomes"].items()}
    out = np.zeros((len(ts), L))
    t_now = 0.0
    for j, t in enumerate(ts):
        dt = t - t_now
        if dt > 0.0:
            for mu in states:
                states[mu] = expm_multiply((-1j * dt) * H, states[mu])
            t_now = t
        prof = np.zeros(L)
        for mu in states:
            prof += probs[mu] * energy_profile(states[mu], h_ops)
        out[j] = prof - profile_gs
    return out


def reduced_density(psi, L, sites):
    """Reduced density matrix of the listed sites (any subset, 0-based).

    For an unnormalized input vector the result is unnormalized (trace =
    ||psi||^2) — exactly what the no-signaling ensemble sum needs.
    """
    L = _check_L(L)
    sites = [_check_site(L, s) for s in sites]
    if len(set(sites)) != len(sites):
        raise ValueError(f"duplicate sites in {sites}")
    arr = np.asarray(psi).reshape((2,) * L)
    arr = np.moveaxis(arr, sites, range(len(sites)))
    M = arr.reshape(2 ** len(sites), -1)
    return M @ M.conj().T


def premeasurement_reduced_states(psi, L, A, sites):
    """Bob-side reduced states with and without Alice's measurement.

    Returns (rho_no, rho_yes): the reduced state of `sites` from |psi> and
    from the post-measurement ensemble sum_mu P_A(mu)|psi><psi|P_A(mu),
    BEFORE any classical communication. No-signaling demands the two be
    identical (trace distance < 1e-14 in the ED anchor) — Alice's local
    measurement is invisible to Bob until the bit arrives.
    """
    rho_no = reduced_density(psi, L, sites)
    rho_yes = np.zeros_like(rho_no)
    for mu in (+1, -1):
        P = (_ID2 + mu * _SX) / 2.0
        branch = apply_site_gate(psi, L, A, P)   # unnormalized: weight in trace
        rho_yes = rho_yes + reduced_density(branch, L, sites)
    return rho_no, rho_yes
