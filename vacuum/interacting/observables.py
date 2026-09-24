"""Observables of the phi^4 MPS ground state: correlators, entropies,
negativity, mutual information, local energy density.

Everything is evaluated on the *truncated* local basis of
:mod:`vacuum.interacting.phi4` (projected operators P O P), so the lambda = 0
values converge to the exact Gaussian core (``vacuum.core``) exponentially
in n_max — the anchor the tests measure.

Two-point functions and the covariance
--------------------------------------
<phi_i phi_j> and <pi_i pi_j> for i != j from ``MPS.correlation_function``
on 'X' and 'Pim' (pi = i Pim, so <pi_i pi_j> = -<Pim_i Pim_j>); the
diagonals from the projected on-site operators 'Xsq', 'Psq' (the same
operators that define H_trunc). The symmetrized cross block
(1/2)<{phi_i, pi_j}> = i <X_i Pim_j> (i != j), i <XPim_i> (i = j) vanishes
identically for a real MPS (real antisymmetric operators) and is computed
in general. ``covariance_from_mps`` assembles the (2N, 2N) block-ordered
covariance of docs/API.md; at lambda = 0 it equals ``ground_state_cov(K)``,
and at lambda > 0 it is the second-moment matrix of a non-Gaussian state
(the *Gaussian proxy* below).

Entanglement
------------
- ``block_entropies``: S([0, l)) for l = 1..L-1 from the Schmidt spectrum on
  each bond (exact for the MPS; von Neumann, nats).
- ``reduced_density_matrix``: rho_A for a small, possibly non-contiguous site
  set A via ``MPS.get_rho_segment`` (intermediate sites traced out), returned
  as a dense (d^k, d^k) matrix with the sites in the order given. TeNPy
  returns the legs *interleaved* (p0, p0*, p1, ...) for non-contiguous
  segments; they are re-ordered by label here (regression-tested against
  the Gaussian core at lambda = 0).
- ``log_negativity_mps``: E_N(A:B) = ln || rho_AB^{T_B} ||_1 (nats; Vidal-
  Werner PRA 65, 032314 (2002)) from the exact partial transpose of the
  dense rho_AB — cost O(d^{3(|A|+|B|)}) for the Hermitian eigenproblem,
  practical for |A| + |B| <= 4 at d <= 9 (d^4 = 6561). No moment/replica
  approximation: this is the true negativity of the truncated-basis state.
- ``mutual_information_mps``: I(A:B) = S_A + S_B - S_AB from the same dense
  reduced states (|A| + |B| <= 4 at the default d).
- ``gaussian_proxy_negativity`` / ``gaussian_proxy_mutual_information``:
  the Gaussian formulas of ``vacuum.core`` applied to the MPS covariance —
  i.e. the negativity/MI of the Gaussian state sharing the interacting
  state's second moments. Exact at lambda = 0, a labelled proxy at
  lambda > 0 (it is neither an upper nor a lower bound on the true value),
  but it reaches block sizes (10+ sites) the dense route cannot.

Local energy density
--------------------
The PSD bond split of ``vacuum.inequalities.qei``:

    h_i = (1/2) pi_i^2 + (1/2) m^2 phi_i^2 + (lam/4!) phi_i^4
          + (1/4)(phi_{i+1} - phi_i)^2 + (1/4)(phi_i - phi_{i-1})^2,

wall bonds ('dirichlet') wholly on their end site, no wall bonds for
'open'; sum_i h_i = H exactly (tested against the MPO energy).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigvalsh as _eigvalsh

from vacuum.core import log_negativity as _gauss_log_negativity
from vacuum.core import mutual_information as _gauss_mutual_information

__all__ = [
    "two_point_functions",
    "covariance_from_mps",
    "block_entropies",
    "reduced_density_matrix",
    "entropy_of_rho",
    "log_negativity_mps",
    "mutual_information_mps",
    "gaussian_proxy_negativity",
    "gaussian_proxy_mutual_information",
    "energy_density_profile",
    "local_energy_density",
]


def two_point_functions(psi):
    """(X, P, C) with X_ij = <phi_i phi_j>, P_ij = <pi_i pi_j>,
    C_ij = (1/2)<{phi_i, pi_j}> (all (L, L) real arrays)."""
    L = psi.L
    X = np.real(np.asarray(psi.correlation_function("X", "X")))
    PP = np.real(np.asarray(psi.correlation_function("Pim", "Pim")))
    P = -PP
    XPim = np.asarray(psi.correlation_function("X", "Pim"))
    C = np.real(1j * XPim)
    xsq = np.real(np.asarray(psi.expectation_value("Xsq")))
    psq = np.real(np.asarray(psi.expectation_value("Psq")))
    xpim = np.asarray(psi.expectation_value("XPim"))
    idx = np.arange(L)
    X[idx, idx] = xsq
    P[idx, idx] = psq
    C[idx, idx] = np.real(1j * xpim)
    X = 0.5 * (X + X.T)
    P = 0.5 * (P + P.T)
    return X, P, C


def covariance_from_mps(psi):
    """(2L, 2L) block-ordered second-moment (covariance) matrix of the MPS.

    Mean-zero by the phi -> -phi symmetry of the ground state; general
    states get the raw second moments (the QEI family is symmetric too).
    """
    X, P, C = two_point_functions(psi)
    L = psi.L
    V = np.zeros((2 * L, 2 * L))
    V[:L, :L] = X
    V[L:, L:] = P
    V[:L, L:] = C
    V[L:, :L] = C.T
    return V


def block_entropies(psi):
    """S([0, l)) in nats for l = 1..L-1 (Schmidt spectrum on bond l)."""
    return np.asarray(psi.entanglement_entropy(), dtype=float)


def reduced_density_matrix(psi, sites):
    """Dense rho_A (d^k x d^k) of the listed sites, in the listed order.

    Sites must be distinct; they need not be contiguous (the sites in
    between are traced out). k = len(sites) is capped at 4 at the default
    d = 7..9 by memory, and TeNPy refuses k > 20 outright.
    """
    sites = [int(s) for s in sites]
    if len(set(sites)) != len(sites):
        raise ValueError("sites must be distinct")
    order = np.argsort(sites)
    sorted_sites = [sites[i] for i in order]
    rho = psi.get_rho_segment(sorted_sites)
    k = len(sites)
    # re-order legs by label: TeNPy interleaves p0, p0*, p1, ... for gaps
    labels = [f"p{j}" for j in range(k)] + [f"p{j}*" for j in range(k)]
    rho = rho.copy()
    rho.itranspose(labels)
    arr = rho.to_ndarray()
    d = arr.shape[0]
    # undo the sort so axis j corresponds to sites[j]
    inv = np.empty(k, dtype=int)
    inv[order] = np.arange(k)
    perm = list(inv) + [k + int(i) for i in inv]
    arr = np.transpose(arr, perm)
    M = arr.reshape(d**k, d**k)
    M = 0.5 * (M + M.conj().T)
    return M


def entropy_of_rho(rho, eig_floor=1e-300):
    """von Neumann entropy (nats) of a density matrix from eigvalsh."""
    w = _eigvalsh(rho, driver="ev")  # 'ev' is ~2x faster than numpy's default at d^4 = 6561
    w = w[w > eig_floor]
    return float(-np.sum(w * np.log(w)))


def log_negativity_mps(psi, sites_A, sites_B, return_spectrum=False):
    """E_N(A:B) = ln || rho_AB^{T_B} ||_1 (nats), exact partial transpose.

    The partial transpose is taken by swapping the B-side ket/bra indices
    of rho_AB reshaped as (dA, dB, dA, dB); the trace norm is the sum of
    |eigenvalues| of the Hermitian result. E_N = 0 exactly when the PT is
    PSD (a PPT state), which for 1 x 1 and 2 x 2 site blocks of the free
    chain happens already at separation 1 and 2 respectively (sudden death;
    Audenaert-Eisert-Plenio-Werner PRA 66, 042327 (2002)).
    """
    A = [int(s) for s in sites_A]
    B = [int(s) for s in sites_B]
    if set(A) & set(B):
        raise ValueError("sites_A and sites_B must be disjoint")
    rho = reduced_density_matrix(psi, A + B)
    d = psi.sites[0].dim
    dA, dB = d ** len(A), d ** len(B)
    T = rho.reshape(dA, dB, dA, dB)
    rho_pt = np.transpose(T, (0, 3, 2, 1)).reshape(dA * dB, dA * dB)
    ev = _eigvalsh(rho_pt, driver="ev")
    e_n = float(np.log(np.sum(np.abs(ev))))
    if e_n < 0.0:  # roundoff below ln(1) = 0 for an exactly PPT state
        e_n = 0.0 if e_n > -1e-12 else e_n
    if return_spectrum:
        return e_n, ev
    return e_n


def mutual_information_mps(psi, sites_A, sites_B):
    """I(A:B) = S_A + S_B - S_AB (nats) from dense reduced states."""
    A = [int(s) for s in sites_A]
    B = [int(s) for s in sites_B]
    if set(A) & set(B):
        raise ValueError("sites_A and sites_B must be disjoint")
    SA = entropy_of_rho(reduced_density_matrix(psi, A))
    SB = entropy_of_rho(reduced_density_matrix(psi, B))
    SAB = entropy_of_rho(reduced_density_matrix(psi, A + B))
    return SA + SB - SAB


def gaussian_proxy_negativity(V, sites_A, sites_B):
    """Negativity of the Gaussian state with covariance V (``vacuum.core``)."""
    return float(_gauss_log_negativity(V, list(sites_A), list(sites_B)))


def gaussian_proxy_mutual_information(V, sites_A, sites_B):
    """Mutual information of the Gaussian state with covariance V."""
    return float(_gauss_mutual_information(V, list(sites_A), list(sites_B)))


def _bond_energy(psi, i, xsq):
    """(1/4)<(phi_{i+1} - phi_i)^2> for bond (i, i+1)."""
    xx = float(np.real(psi.correlation_function("X", "X", sites1=[i], sites2=[i + 1])[0, 0]))
    return 0.25 * (xsq[i] + xsq[i + 1] - 2.0 * xx)


def energy_density_profile(psi, m, lam, bc="dirichlet"):
    """<h_i> for all sites (PSD split, module docstring); sums to <H>."""
    L = psi.L
    m = float(m)
    lam = float(lam)
    xsq = np.real(np.asarray(psi.expectation_value("Xsq")))
    psq = np.real(np.asarray(psi.expectation_value("Psq")))
    prof = 0.5 * psq + 0.5 * m * m * xsq
    if lam != 0.0:
        prof = prof + (lam / 24.0) * np.real(np.asarray(psi.expectation_value("X4")))
    bonds = np.array([_bond_energy(psi, i, xsq) for i in range(L - 1)])
    prof[:-1] += bonds
    prof[1:] += bonds
    if bc == "dirichlet":
        prof[0] += 0.5 * xsq[0]
        prof[L - 1] += 0.5 * xsq[L - 1]
    elif bc != "open":
        raise ValueError(f"bc must be 'dirichlet' or 'open', got {bc!r}")
    return prof


def local_energy_density(psi, site, m, lam, bc="dirichlet"):
    """<h_site> alone (three-site read; the hot path of the QEI evolution)."""
    L = psi.L
    i = int(site)
    m = float(m)
    lam = float(lam)
    lo, hi = max(0, i - 1), min(L - 1, i + 1)
    sites = list(range(lo, hi + 1))
    xsq_loc = np.real(np.asarray(psi.expectation_value("Xsq", sites=sites)))
    xsq = {s: float(v) for s, v in zip(sites, xsq_loc)}
    e = 0.5 * float(np.real(psi.expectation_value("Psq", sites=[i])[0])) + 0.5 * m * m * xsq[i]
    if lam != 0.0:
        e += (lam / 24.0) * float(np.real(psi.expectation_value("X4", sites=[i])[0]))
    for j in (i - 1, i):
        if 0 <= j < L - 1:
            xx = float(np.real(psi.correlation_function("X", "X", sites1=[j], sites2=[j + 1])[0, 0]))
            e += 0.25 * (xsq[j] + xsq[j + 1] - 2.0 * xx)
    if bc == "dirichlet":
        if i == 0:
            e += 0.5 * xsq[0]
        if i == L - 1:
            e += 0.5 * xsq[L - 1]
    elif bc != "open":
        raise ValueError(f"bc must be 'dirichlet' or 'open', got {bc!r}")
    return float(e)
