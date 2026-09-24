"""Non-Gaussian measurements on Alice: photon-number-resolved and on/off.

Companion of :mod:`vacuum.composite.harvest_teleport`, closing its open
item "pure Gaussian measurements only — the surplus is a LOWER bound on
the daemonic gain".  The daemonic gain of Francica-Goold-Plastina-
Paternostro (npj Quantum Inf. 3, 12 (2017), arXiv:1608.00124) is defined
over *projective* measurements on the ancilla, Eq. (5) there,
δW = max_{Π} W_{Π} − W, with W_{Π} the outcome-averaged conditional
ergotropy, Eq. (3), and W the unconditional ergotropy (Allahverdyan-
Balian-Nieuwenhuizen, Europhys. Lett. 67, 565 (2004), Eq. (2) of Francica
et al.); δW ≥ 0 by construction, and statistically independent S and A
give no gain ("we would have ρ_{S|a} = ρ_S for any set {Π_a}", Sec.
"Daemonic work and quantum correlations").  Bernards-Kleinmann-Gühne-
Paternostro (Entropy 21, 771 (2019), arXiv:1907.01970) extend the
definition to generalised measurements and exhibit a state on which a
POVM beats every projective measurement, so no closed family is known to
be optimal in general; the Gaussian general-dyne family of Kua-Serafini-
Genoni (Quantum Sci. Technol. 11, 015014 (2025), arXiv:2506.22288) is one
family, the photon-number (Fock) projection another, and neither
dominates the other a priori.  This module evaluates the second family
and the coarse on/off (click / no-click) POVM on the same two-mode
Gaussian states the composite harvests, so the two can be compared.

The conditional states on Bob are non-Gaussian (a Fock projection on
Alice does not preserve Gaussianity), so Bob's ergotropy is the general
finite-dimensional one — the passive-state construction, spectrum sorted
against the sorted level ladder — not the symplectic formula of the
Gaussian column.  Both columns are exact statements about the same state:
for a *Gaussian* measurement the conditional state is Gaussian and its
full ergotropy equals the Gaussian one (a single-mode Gaussian state is a
Gaussian unitary applied to a thermal state), so comparing "Gaussian
optimum" with "Fock-resolved" is a like-for-like comparison of two
measurement families under one ergotropy.

Fock representation of a Gaussian state (the tool)
--------------------------------------------------
For a zero-mean M-mode Gaussian state with covariance V (block
quadrature ordering, V_vac = I/2, docs/API.md) the coherent-state matrix
element factorises as ⟨α|ρ|β⟩ = e^{−(|α|²+|β|²)/2} F(α*, β) with F entire
in z = (α*_1..α*_M, β_1..β_M), and the Husimi function is the Gaussian
Q(α) = ⟨α|ρ|α⟩/π^M = exp(−½ ζ† σ_Q⁻¹ ζ) / (π^M √det σ_Q), ζ = (α, α*),
σ_Q = σ + I/2, σ = W V W† the covariance in the (a, a†) basis (W = [[I,
iI], [I, −iI]]/√2).  Reading F off the diagonal β = α and continuing (F
and the right-hand side are both entire in the independent variables
(α*, α), which span a totally real slice) gives

    F(z) = exp(½ zᵀ A z) / √det σ_Q,     A = X (I − σ_Q⁻¹),  X = [[0, I], [I, 0]],

with A complex-symmetric because of the (W, Y; Y*, W*) block structure
of σ_Q — the zero-mean case of N. Quesada, L. G. Helt, J. Izaac, J. M.
Arrazola, R. Shahrokhshahi, C. R. Myers, K. K. Sabapathy, Phys. Rev. A
100, 022341 (2019), arXiv:1905.07011, Eqs. (6)-(11) and (A1)-(A4)
(their σ_Q = σ + ½ I_{2ℓ}, A = X(I_{2ℓ} − σ_Q⁻¹), prefactor 1/√det σ_Q,
loop hafnian of the row/column-repeated matrix, which at zero mean is the
plain hafnian, i.e. the multidimensional Hermite polynomial of Dodonov-
Man'ko-Man'ko, their Ref. [64]).  Expanding F(z) = Σ_k c_k z^k and
differentiating, (k_i + 1) c_{k+e_i} = Σ_j A_ij c_{k−e_j}; with the
normalised coefficients d_k = √(k!) c_k the Fock matrix elements are
⟨m|ρ|n⟩ = d_{(m, n)} exactly, and the recursion

    d_{k+e_i} = (1/√(k_i+1)) Σ_j A_ij √k_j d_{k−e_j},   d_0 = 1/√det σ_Q,     (F1)

fills the truncated (n_c)^{2M} tensor axis by axis with vectorised
numpy (each entry is exact — truncation drops entries, it does not
perturb the retained ones, so the truncated matrix is a principal
submatrix of the true density matrix, hence positive semidefinite).
Anchors (tests): vacuum → |0⟩⟨0|; thermal n̄ → (1−q) qⁿ, q = n̄/(n̄+1);
two-mode squeezed vacuum → √(1−q) q^{n/2} δ_{mn} on |n, n⟩, q = tanh² r;
first and second moments ⟨a†a⟩, ⟨a a⟩, ⟨a_A a_B⟩, ⟨a_A a_B†⟩ of the
truncated matrix against the covariance to 1e-12.

Measurements and the columns
----------------------------
Alice's Fock basis is that of her own oscillator (a_A = (√Ω_A x_A +
i p_A/√Ω_A)/√2), Bob's likewise, so the covariance is first taken to the
detectors' dimensionless units by the local symplectic diag(√Ω, 1/√Ω);
the level ladder at Bob is ε_n = Ω_B (n + ½).

* photon-number-resolved (PNR): Π_k = |k⟩⟨k|_A, k < n_c; Bob's
  unnormalised conditional state is ρ_B^{(k)} = ⟨k|ρ|k⟩_A (the [k, :, k, :]
  slice of the Fock tensor), p_k = Tr ρ_B^{(k)};
* on/off: Π_off = |0⟩⟨0|_A, Π_on = I − Π_off; ρ_B^{on} = ρ_B − ρ_B^{(0)};
* the general ergotropy E(ρ) = Tr(Hρ) − Σ_j λ_j↓ ε_j↑ (Allahverdyan et
  al.), extended to unnormalised PSD ρ by homogeneity so that the
  outcome average is Σ_k E(ρ_B^{(k)}) with no division by p_k;
* E_local^full = E(ρ_B) with ρ_B = Tr_A ρ = Σ_k ρ_B^{(k)} — equal to the
  Gaussian E_local of the composite, Eq. (3) there, to the truncation
  error (a test at 1e-10);
* gains: gain_pnr = Σ_k E(ρ_B^{(k)}) − E(ρ_B), gain_onoff = E(ρ_B^{(0)}) +
  E(ρ_B − ρ_B^{(0)}) − E(ρ_B).  gain_pnr ≥ gain_onoff always (PNR refines
  on/off, and the ergotropy is convex — Tr(Hρ) is linear and the passive
  energy is a minimum of linear functionals, so it is concave; a test).

Cutoff convergence.  The tensor is built at n_c and at n_c + step until
every reported gain moves by less than ``tol`` (absolute, in energy
units); the accepted n_c, the last movement and the trace deficit
1 − Tr ρ_trunc are returned.  Because retained elements are exact, the
truncation error of any outcome average is bounded by the discarded
probability times the largest level considered.  The ladder *starts* at
a cutoff predicted from the state's own occupations
(:func:`predicted_cutoff`): each mode's Fock populations decay like
q^n with q = n̄/(n̄+1), so n_start ~ ln(tail)/ln(max q) reaches a given
tail.  The prediction only chooses where to start — the one-step
movement check at ``tol`` still runs and still decides, so a bad
prediction costs time, never accuracy.  Hot rows (n̄ of order 1)
therefore converge instead of stopping at the ceiling, and cold rows
(the great majority of a harvesting map, n̄ ~ 1e-2) skip wasted rungs.

Closed-form anchors (tests, to 1e-10).  On the two-mode squeezed vacuum
of squeezing r, |ψ⟩ = √(1−q) Σ q^{n/2} |n, n⟩, q = tanh² r: a PNR outcome k
leaves Bob in |k⟩ (ergotropy Ω_B k), so gain_pnr = Ω_B Σ (1−q) qⁿ n =
Ω_B q/(1−q) = Ω_B sinh² r — the same as every Gaussian measurement, as it
must be on a pure state (any rank-one measurement purifies Bob and
no-signaling fixes his mean energy); the click branch leaves Bob in the
shifted geometric mixture Σ_{n≥1} (1−q) q^{n−1} |n⟩⟨n|, already
energy-sorted, whose passive state is the same populations shifted down
one level: ergotropy exactly Ω_B, so gain_onoff = P(click) Ω_B =
Ω_B tanh² r < Ω_B sinh² r.  On any pure two-mode Gaussian state (local
squeezings on top of the TMSV, not phase-invariant) every rank-one
measurement, Gaussian or Fock, gives the same daemonic ergotropy
E_B − Ω_B/2 and hence the same gain Ω_B (ν_B − ½).

Pure functions over arrays; no hidden state, no input mutation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "complex_covariance",
    "predicted_cutoff",
    "hermite_matrix",
    "fock_tensor",
    "fock_density_matrix",
    "fock_moments",
    "ergotropy_finite",
    "FockConditioningResult",
    "fock_conditioning",
    "optimal_fock_squeezing",
    "tmsv_fock_gain_closed_form",
]

_LOG_HALF = math.log(0.5)


# --------------------------------------------------------------------------
# The Fock representation
# --------------------------------------------------------------------------


def _check_cov(V):
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] != V.shape[1] or V.shape[0] % 2:
        raise ValueError(f"V must be a (2M, 2M) covariance, got {V.shape}")
    return 0.5 * (V + V.T)


def complex_covariance(V):
    """σ = W V W†: the covariance in the (a_1..a_M, a†_1..a†_M) basis.

    a = (x + i p)/√2, so W = [[I, iI], [I, −iI]]/√2 is unitary and the
    vacuum I/2 maps to I/2.  σ is Hermitian with blocks (W, Y; Y*, W*),
    W = ⟨a a†⟩-like (Hermitian), Y = ⟨a a⟩-like (symmetric).
    """
    V = _check_cov(V)
    M = V.shape[0] // 2
    I = np.eye(M)
    W = np.block([[I, 1j * I], [I, -1j * I]]) / math.sqrt(2.0)
    sigma = W @ V @ W.conj().T
    return 0.5 * (sigma + sigma.conj().T)


def hermite_matrix(V):
    """(A, norm): A = X (I − σ_Q⁻¹) with σ_Q = σ + I/2, norm = 1/√det σ_Q.

    A is complex-symmetric (asserted to 1e-12 relative and symmetrised);
    norm is ⟨0|ρ|0⟩, the vacuum population.
    """
    sigma = complex_covariance(V)
    M = sigma.shape[0] // 2
    sigma_Q = sigma + 0.5 * np.eye(2 * M)
    inv = np.linalg.inv(sigma_Q)
    X = np.zeros((2 * M, 2 * M))
    X[:M, M:] = np.eye(M)
    X[M:, :M] = np.eye(M)
    A = X @ (np.eye(2 * M) - inv)
    asym = float(np.max(np.abs(A - A.T)))
    scale = max(1.0, float(np.max(np.abs(A))))
    if asym > 1e-10 * scale:
        raise ValueError(f"Hermite matrix is not symmetric (defect {asym:.3e}); V is not a valid covariance")
    A = 0.5 * (A + A.T)
    det = np.linalg.det(sigma_Q)
    det_r = float(np.real(det))
    if det_r <= 0.0 or abs(np.imag(det)) > 1e-10 * det_r:
        raise ValueError(f"det σ_Q = {det} is not real positive; V violates the uncertainty principle")
    return A, 1.0 / math.sqrt(det_r)


def _shift_scaled(X, axis):
    """Y[..., k, ...] = √k X[..., k−1, ...] along ``axis`` (zero at k = 0)."""
    n = X.shape[axis]
    Y = np.zeros_like(X)
    src = [slice(None)] * X.ndim
    dst = [slice(None)] * X.ndim
    src[axis] = slice(0, n - 1)
    dst[axis] = slice(1, n)
    shape = [1] * X.ndim
    shape[axis] = n - 1
    root = np.sqrt(np.arange(1, n, dtype=float)).reshape(shape)
    Y[tuple(dst)] = root * X[tuple(src)]
    return Y


def fock_tensor(V, cutoff):
    """The normalised Hermite coefficients d_k, Eq. (F1), on the (cutoff,)^{2M} grid.

    Index order (m_1..m_M, n_1..n_M): ⟨m|ρ|n⟩ = d[m_1, .., m_M, n_1, .., n_M].
    Every retained entry is exact; nothing outside the grid is used.
    """
    A, norm = hermite_matrix(V)
    L = A.shape[0]                      # 2M
    n_c = int(cutoff)
    if n_c < 1:
        raise ValueError(f"cutoff must be >= 1, got {cutoff}")
    D = np.zeros((n_c,) * L, dtype=complex)
    D[(0,) * L] = norm
    for i in range(L):
        # entries with axes > i at zero, axes < i arbitrary, axis i from 0 to n_c - 1
        head = (slice(None),) * i
        tail = (0,) * (L - 1 - i)
        prev = None                                  # slab at k_i - 1
        cur = D[head + (0,) + tail]                  # slab at k_i = 0 (shape (n_c,)*i)
        cur = np.array(cur, dtype=complex, copy=True)
        for k in range(n_c - 1):
            new = np.zeros_like(cur)
            if prev is not None:
                new += A[i, i] * math.sqrt(k) * prev
            for j in range(i):
                if A[i, j] != 0.0:
                    new += A[i, j] * _shift_scaled(cur, j)
            new /= math.sqrt(k + 1.0)
            D[head + (k + 1,) + tail] = new
            prev, cur = cur, new
    return D


def fock_density_matrix(V, cutoff):
    """⟨m|ρ|n⟩ as a (cutoff^M, cutoff^M) matrix, row-major multi-index (m_1..m_M)."""
    D = fock_tensor(V, cutoff)
    M = D.ndim // 2
    n_c = D.shape[0]
    return D.reshape(n_c**M, n_c**M)


def fock_moments(D):
    """First and second moments of the truncated Fock tensor (M ≤ 2).

    Returns a dict with ``trace``, ``n`` (per-mode ⟨a†a⟩), ``aa`` (per-mode
    ⟨a a⟩), and for M = 2 ``a1a2`` = ⟨a_1 a_2⟩ and ``a1a2d`` = ⟨a_1 a_2†⟩.
    Used by the anchors to pin the representation against the covariance.
    """
    M = D.ndim // 2
    n_c = D.shape[0]
    rho = D.reshape(n_c**M, n_c**M)
    idx = np.indices((n_c,) * M).reshape(M, -1)          # multi-index of each basis vector
    out: Dict[str, object] = {"trace": complex(np.trace(rho))}
    diag = np.real(np.diag(rho))
    out["n"] = [float(np.sum(diag * idx[s])) for s in range(M)]
    # <a_s a_s> = sum_n sqrt((n+1)(n+2)) <n+2| rho |n>  (a a |n+2> = sqrt((n+2)(n+1)) |n>)
    aa = []
    for s in range(M):
        tot = 0.0 + 0.0j
        for col in range(n_c**M):
            m = idx[:, col].copy()
            if m[s] + 2 < n_c:
                n = m.copy(); n[s] += 2
                row = int(np.ravel_multi_index(n, (n_c,) * M))
                tot += math.sqrt((m[s] + 1) * (m[s] + 2)) * rho[col, row]
        aa.append(complex(tot))
    out["aa"] = aa
    if M == 2:
        t12 = 0.0 + 0.0j
        t12d = 0.0 + 0.0j
        for col in range(n_c**2):
            m = idx[:, col].copy()
            if m[0] + 1 < n_c and m[1] + 1 < n_c:
                n = m.copy(); n[0] += 1; n[1] += 1
                row = int(np.ravel_multi_index(n, (n_c, n_c)))
                t12 += math.sqrt((m[0] + 1) * (m[1] + 1)) * rho[col, row]
            if m[0] + 1 < n_c and m[1] >= 1:
                n = m.copy(); n[0] += 1; n[1] -= 1
                row = int(np.ravel_multi_index(n, (n_c, n_c)))
                t12d += math.sqrt((m[0] + 1) * m[1]) * rho[col, row]
        out["a1a2"] = complex(t12)
        out["a1a2d"] = complex(t12d)
    return out


# --------------------------------------------------------------------------
# General ergotropy (passive-state construction)
# --------------------------------------------------------------------------


def ergotropy_finite(rho, energies, hermitian_tol=1e-10):
    """E(ρ) = Tr(Hρ) − Σ_j λ_j↓ ε_j↑ for H diagonal with levels ``energies``.

    Allahverdyan-Balian-Nieuwenhuizen (2004): the passive state carries
    the sorted spectrum on the sorted levels.  ρ may be unnormalised and
    PSD (the outcome-weighted conditional states); the result is then
    p × E(ρ/p) by homogeneity.  Returns (ergotropy, E, E_passive, min eig).
    """
    rho = np.asarray(rho, dtype=complex)
    eps = np.asarray(energies, dtype=float)
    d = rho.shape[0]
    if rho.shape != (d, d) or eps.shape != (d,):
        raise ValueError(f"rho {rho.shape} and energies {eps.shape} do not match")
    herm = float(np.max(np.abs(rho - rho.conj().T))) if d else 0.0
    scale = max(1.0, float(np.max(np.abs(rho)))) if d else 1.0
    if herm > hermitian_tol * scale:
        raise ValueError(f"rho is not Hermitian (defect {herm:.3e})")
    rho = 0.5 * (rho + rho.conj().T)
    E = float(np.real(np.sum(np.diag(rho) * eps)))
    lam = np.linalg.eigvalsh(rho)
    lam_desc = np.sort(lam)[::-1]
    eps_asc = np.sort(eps)
    E_pass = float(np.sum(lam_desc * eps_asc))
    return E - E_pass, E, E_pass, float(np.min(lam))


# --------------------------------------------------------------------------
# The conditioning columns
# --------------------------------------------------------------------------


@dataclass
class FockConditioningResult:
    """Non-Gaussian-measurement columns on a two-mode Gaussian state.

    Energies are in the units of Bob's H_B = Ω_B (a†a + ½); ``E_local_full``
    is Bob's unconditional ergotropy from the Fock spectrum (equals the
    Gaussian closed form to the truncation error), ``E_daemonic_pnr`` /
    ``E_daemonic_onoff`` the outcome-averaged conditional ergotropies,
    ``gain_*`` their excess over ``E_local_full``.  ``cutoff`` is the
    accepted Fock cutoff, ``movement`` the largest change of any gain
    under the last cutoff increase, ``trace_deficit`` 1 − Tr ρ_trunc,
    ``min_eig`` the most negative eigenvalue met in any conditional state
    (roundoff only: retained elements are exact), ``p_k`` the outcome
    distribution of the PNR measurement, ``no_signaling_defect`` the
    deviation of Σ_k ρ_B^{(k)} from the partial trace computed directly.
    """

    omega_B: float
    E_B: float
    E_local_full: float
    E_daemonic_pnr: float
    E_daemonic_onoff: float
    gain_pnr: float
    gain_onoff: float
    cutoff: int
    movement: float
    trace_deficit: float
    min_eig: float
    p_k: np.ndarray
    p_click: float
    no_signaling_defect: float
    cutoffs_tried: List[int]
    converged: bool


def predicted_cutoff(V_dimless, tail=1e-18, step=8, lo=16, hi=88):
    """Fock cutoff whose per-mode geometric tail falls below ``tail``.

    Populations of a thermal mode of mean occupation n̄ decay like q^n,
    q = n̄/(n̄+1); n > ln(tail)/ln(q) puts the tail below ``tail``.
    ``V_dimless`` is the covariance already in oscillator units, so
    n̄_i = (V_xx + V_pp)/2 − 1/2 per mode.  Rounded up to a multiple of
    ``step`` and clamped to [lo, hi].  A prediction only chooses the
    ladder's first rung; convergence is still decided by the movement
    check in :func:`fock_conditioning`.
    """
    V = _check_cov(V_dimless)
    M = V.shape[0] // 2
    n_req = float(lo)
    for i in range(M):
        nbar = 0.5 * (V[i, i] + V[M + i, M + i]) - 0.5
        if nbar <= 1e-12:
            continue
        q = nbar / (nbar + 1.0)
        n_req = max(n_req, math.log(tail) / math.log(q))
    n_c = int(step * math.ceil(n_req / step))
    return max(int(lo), min(int(hi), n_c))


def _to_oscillator_units(V, gaps):
    """Local symplectic diag(√Ω, 1/√Ω) taking each mode to its own a, a†."""
    Om = np.asarray(gaps, dtype=float)
    M = Om.size
    D = np.diag(np.r_[np.sqrt(Om), 1.0 / np.sqrt(Om)])
    return D @ V @ D


def _conditioning_at(V_dimless, omega_B, cutoff):
    D = fock_tensor(V_dimless, cutoff)
    n_c = D.shape[0]
    eps = omega_B * (np.arange(n_c) + 0.5)
    rho_B = np.einsum("kakb->ab", D)                     # partial trace over Alice
    E_loc, E_B, _, mn = ergotropy_finite(rho_B, eps)
    min_eig = mn
    E_pnr = 0.0
    p_k = np.zeros(n_c)
    acc = np.zeros_like(rho_B)
    rho_0 = None
    for k in range(n_c):
        r_k = D[k, :, k, :]
        p_k[k] = float(np.real(np.trace(r_k)))
        e_k, _, _, mn_k = ergotropy_finite(r_k, eps)
        E_pnr += e_k
        min_eig = min(min_eig, mn_k)
        acc += r_k
        if k == 0:
            rho_0 = r_k
    e_off, _, _, _ = ergotropy_finite(rho_0, eps)
    e_on, _, _, mn_on = ergotropy_finite(rho_B - rho_0, eps)
    min_eig = min(min_eig, mn_on)
    E_onoff = e_off + e_on
    ns_defect = float(np.max(np.abs(acc - rho_B)))
    trace = float(np.real(np.trace(rho_B)))
    return {
        "E_B": E_B, "E_local": E_loc, "E_pnr": E_pnr, "E_onoff": E_onoff,
        "gain_pnr": E_pnr - E_loc, "gain_onoff": E_onoff - E_loc,
        "trace_deficit": 1.0 - trace, "min_eig": min_eig, "p_k": p_k,
        "p_click": 1.0 - p_k[0], "ns_defect": ns_defect,
    }


def fock_conditioning(V_AB, gaps, cutoff0=None, step=8, max_cutoff=88, tol=1e-10):
    """PNR and on/off conditioning on Alice; Bob's general ergotropy.

    Parameters
    ----------
    V_AB : (4, 4) block-ordered two-mode covariance (Alice = mode 0), in
        the detectors' physical units (H_d = ½(p² + Ω² x²)).
    gaps : (Ω_A, Ω_B): the oscillators whose Fock bases are used.
    cutoff0, step, max_cutoff : the cutoff ladder; ``cutoff0=None`` starts
        it at :func:`predicted_cutoff` of the state.  Convergence is
        declared when every gain moves by less than ``tol`` under one
        step of ``step`` — the prediction never decides it.
    tol : absolute convergence tolerance on the gains (energy units).
    """
    V = _check_cov(V_AB)
    if V.shape != (4, 4):
        raise ValueError(f"expected a two-mode (4, 4) covariance, got {V.shape}")
    Om = np.asarray(gaps, dtype=float)
    if Om.shape != (2,) or np.any(Om <= 0.0):
        raise ValueError(f"gaps must be two positive frequencies, got {gaps}")
    Vd = _to_oscillator_units(V, Om)
    omega_B = float(Om[1])
    tried: List[int] = []
    prev = None
    movement = math.inf
    converged = False
    n_c = predicted_cutoff(Vd, step=step, hi=max_cutoff) if cutoff0 is None else int(cutoff0)
    res = None
    while True:
        tried.append(n_c)
        res = _conditioning_at(Vd, omega_B, n_c)
        if prev is not None:
            movement = max(abs(res["gain_pnr"] - prev["gain_pnr"]),
                           abs(res["gain_onoff"] - prev["gain_onoff"]),
                           abs(res["E_local"] - prev["E_local"]))
            if movement < tol:
                converged = True
                break
        if n_c + int(step) > int(max_cutoff):
            break
        prev = res
        n_c += int(step)
    return FockConditioningResult(
        omega_B=omega_B, E_B=res["E_B"], E_local_full=res["E_local"],
        E_daemonic_pnr=res["E_pnr"], E_daemonic_onoff=res["E_onoff"],
        gain_pnr=res["gain_pnr"], gain_onoff=res["gain_onoff"], cutoff=n_c,
        movement=float(movement), trace_deficit=res["trace_deficit"], min_eig=res["min_eig"],
        p_k=res["p_k"], p_click=res["p_click"], no_signaling_defect=res["ns_defect"],
        cutoffs_tried=tried, converged=converged,
    )


def _alice_squeezer(s, theta):
    """S⁻¹ of the general-dyne family in Alice's dimensionless units.

    The Gaussian family's pointer is σ_M = S (I/2) Sᵀ with S = R(θ)
    diag(s^-1/2, s^1/2) (physical-unit prefactor D removed here since the
    covariance is already in oscillator units); counting photons after
    applying S⁻¹ to Alice's mode is photon counting in the squeezed number
    basis S†|k⟩ — the Fock analogue of the (s, θ) general-dyne, s = 1
    being plain counting in her own oscillator basis.
    """
    c, sn = math.cos(theta), math.sin(theta)
    R = np.array([[c, -sn], [sn, c]])
    return np.diag([math.sqrt(s), 1.0 / math.sqrt(s)]) @ R.T


def optimal_fock_squeezing(V_AB, gaps, s_grid=(1.0, 2.0, 4.0), n_theta=4,
                           cutoff0=None, step=8, max_cutoff=88, tol=1e-10):
    """PNR gain maximised over a deterministic (s, θ) grid of squeezed number bases.

    A HYBRID family, and the docstring says so because the number matters:
    counting photons in the basis S†|k⟩ of an s-squeezed mode tends, as
    s → ∞, to a HOMODYNE of the corresponding quadrature — which is
    already in the Gaussian general-dyne family.  So this search does not
    isolate "non-Gaussian beats Gaussian"; the plain (s = 1) PNR and the
    on/off POVM of :func:`fock_conditioning` are the genuinely
    non-Gaussian columns, and this one is reported only as a diagnostic
    of how the two families interpolate.  Its truncation also gets harder
    exactly as it approaches that limit (squeezing raises Alice's
    occupation, so the required cutoff grows with s).

    Only bases that CONVERGE are accepted: a grid point whose predicted
    cutoff already exceeds ``max_cutoff`` is skipped without evaluation,
    and one whose movement check fails is rejected after it.  The
    returned ``gain`` is therefore always a converged number, falling
    back to the plain PNR gain (s = 1) when no squeezed basis converges.

    Returns a dict with ``gain``, ``s``, ``theta``, ``result`` (the
    :class:`FockConditioningResult` at the optimum), ``grid`` (accepted
    (s, θ, gain) triples, s = 1 first) and ``skipped`` / ``rejected``
    (the (s, θ) pairs not evaluated / not converged).
    """
    V = _check_cov(V_AB)
    Om = np.asarray(gaps, dtype=float)
    Vd = _to_oscillator_units(V, Om)
    omega_B = float(Om[1])
    thetas = np.linspace(0.0, math.pi, int(n_theta), endpoint=False)
    best = None
    grid = []
    skipped = []
    rejected = []
    for s in s_grid:
        for th in (thetas if s != 1.0 else np.array([0.0])):
            S4 = np.eye(4)
            Si = _alice_squeezer(float(s), float(th))
            S4[np.ix_([0, 2], [0, 2])] = Si
            Vs = S4 @ Vd @ S4.T
            # a basis whose required cutoff is already past the ceiling cannot
            # converge: skip it without paying for the tensor (the unclamped
            # prediction is the test)
            if cutoff0 is None and predicted_cutoff(Vs, step=step, hi=10**9) > int(max_cutoff):
                skipped.append((float(s), float(th)))
                continue
            res = fock_conditioning(Vs, (1.0, omega_B), cutoff0=cutoff0, step=step,
                                    max_cutoff=max_cutoff, tol=tol)
            if not res.converged:
                rejected.append((float(s), float(th)))
                continue
            grid.append((float(s), float(th), res.gain_pnr))
            if best is None or res.gain_pnr > best[2] + 1e-13 * max(1.0, abs(best[2])):
                best = (float(s), float(th), res.gain_pnr, res)
    if best is None:  # pragma: no cover - s = 1 converges whenever the row does
        raise RuntimeError("no Fock basis converged, not even the unsqueezed one")
    return {"gain": best[2], "s": best[0], "theta": best[1], "result": best[3], "grid": grid,
            "skipped": skipped, "rejected": rejected}


def tmsv_fock_gain_closed_form(r, omega_B=1.0):
    """(gain_pnr, gain_onoff) on the two-mode squeezed vacuum: Ω sinh² r, Ω tanh² r."""
    return omega_B * math.sinh(r) ** 2, omega_B * math.tanh(r) ** 2
