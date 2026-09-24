"""The delta-coupling no-go: the suite's designated exact null (M2.5).

Nonperturbative closed form for two Unruh-DeWitt detectors that couple to a
scalar field through *Dirac-delta* switching functions, transcribed from

    P. Simidzija and E. Martin-Martinez, "Non-perturbative analysis of
    entanglement harvesting from coherent field states", Phys. Rev. D 96,
    065008 (2017), arXiv:1707.00016 -- referred to as SMM17 below.

Its Theorem 3 is the statement this module exists to encode: *delta-coupled
detectors cannot harvest any entanglement from the field vacuum, nor from
any coherent field state.*  The negativity is not small, it is exactly
zero, for every choice of spatial smearings, coupling strengths, switching
strengths, switching times, energy gaps and coherent amplitude.  That makes
it the one place in the program where the right answer is a hard zero, so
it is the test that proves the engine can output zero (spec M2.5).

Provenance of the equation numbers
----------------------------------
The arXiv source tarball (arxiv.org/e-print/1707.00016) was fetched on
2026-09-01; every equation number cited below was obtained by counting the
numbered environments of ``main.tex`` and cross-checking the count against
the ar5iv rendering of the same source
(ar5iv.labs.arxiv.org/html/1707.00016), which carries the printed tags
(the two agree on the document total, 169 numbered equations, and on every
tag spot-checked).  The (label, printed number) pairs used here are:

    eq:field (1)        eq:alpha (3)         eq:H_I_A (6)      eq:m_A (7)
    eq:chi_delta_a (19) eq:Ya (21)           eq:c_a (32)       eq:FT (33)
    eq:fa (36)          eq:bnu (37)          eq:fa_final (39)
    eq:rho_a_final (40) eq:E_A_1 (41)        eq:E_A_2 (42)
    eq:chi_delta_nu (45) eq:U2 (47)          eq:Yb (49)
    eq:U2_v2 (50)       eq:X_jk (51)         eq:B2_2 (52)
    eq:rhoab (56)       eq:f2_def (57)       eq:c_b (58)
    eq:f2 (60)          eq:theta (61)        eq:fb (62)
    eq:fb_final (69)    eq:fp_final (70)     eq:fm_final (71)
    eq:W (73)           eq:Q (74)            eq:Qpt (77)       eq:neg (79)
    eq:theta_final (80) eq:omega (81)        eq:Q2 (84)        eq:Qpt2 (85)
    eq:e1 (86)          eq:e2 (87)           eq:e3 (88)        eq:e4 (89)

NOTE ON THE SPEC'S CITATION.  docs/PLAN_LAYERS_2_3.md, M2.5, attributes the
delta-coupled closed form to "PRD 98, 085007 (2018)" (arXiv:1809.05547).
That paper is Simidzija and Martin-Martinez, "Harvesting correlations from
thermal and squeezed coherent states"; its source was fetched and read, and
it contains no delta-coupling and no such closed form -- it is a purely
perturbative (second-order) treatment of thermal and squeezed coherent
states.  The delta-coupled nonperturbative closed form and the N = 0
theorem are in SMM17 (PRD 96, 065008 (2017), arXiv:1707.00016), which is
what is transcribed here; the general structural statement was later
proved in P. Simidzija, R. H. Jonsson and E. Martin-Martinez, "General
no-go theorem for entanglement extraction", Phys. Rev. D 97, 125002
(2018), arXiv:1803.11214.  Both journal references were verified against
the arXiv abstract pages on 2026-09-01.  Nothing here is transcribed from
PRD 98, 085007.

Setup
-----
Field phi(x, t) in n+1 flat dimensions [SMM17 Eq. (1)], two UDW detectors
nu in {A, B} with spatial profiles F_nu(x - x_nu), coupling strengths
lam_nu and *delta* switchings

    chi_nu(t) = eta_nu delta(t - t_nu)          [Eq. (45)]

(the symmetric limit of a symmetric switching of area eta_nu; SMM17
Sec. III).  With t_A <= t_B the time ordering in the Dyson unitary
collapses [Eq. (47)] and

    U_2 = exp(mu_B (x) Y_B) exp(mu_A (x) Y_A),
    Y_nu = -i lam_nu eta_nu int d^n x F_nu(x - x_nu) phi(x, t_nu)
                                                [Eqs. (21), (49)]

with Y_nu anti-Hermitian.  Because m_nu^2 = 1 this is [Eq. (50)]

    U_2 = 1(x)1(x)X_(++) + m_A(x)1(x)X_(+-)
        + 1(x)m_B(x)X_(-+) + m_A(x)m_B(x)X_(--),
    X_(jk) = (1/4)(e^{Y_B} + j e^{-Y_B})(e^{Y_A} + k e^{-Y_A})  [Eq. (51)],

and the two-detector density matrix in the gap-free basis
{|g~_A g~_B>, |g~_A e~_B>, |e~_A g~_B>, |e~_A e~_B>} [Eq. (52)] has entries
f_2^{(jklm)} = <alpha| X_(jk)^dag X_(lm) |alpha> [Eqs. (56), (57)], given
in closed form by Eq. (60).  Since the basis (52) absorbs the free phases
e^{i Omega_nu t_nu}, *the whole state is independent of the detector gaps*
(SMM17 Sec. IV) -- delta coupling leaves no room for free dynamics.

Everything reduces to five real/complex scalars.  Writing [Eq. (37)]

    beta_nu(k) = -2 i lam_nu eta_nu Ftilde_nu(k)* e^{i(|k| t_nu - k.x_nu)}
                 / sqrt(2 |k|),

so that e^{2 Y_nu}|0> = |beta_nu> is a coherent state, the closed forms are

    f_nu   = <0|e^{2Y_nu}|0> = exp(-(1/2) int d^n k |beta_nu|^2)   [(39), (69)]
    theta  = (i/4) int d^n k (beta_A^* beta_B - c.c.)              [(80)]
    omega  = -(1/2) int d^n k (beta_A^* beta_B + c.c.)             [(81)]
    C_nu   = the coherent-amplitude shift of Eqs. (32), (58)
    f_p = f_A f_B e^{omega - 2 i theta},  f_m = f_A f_B e^{-omega + 2 i theta}
                                                            [(70), (71)]

and every one of them is a *pulled-back smeared Wightman function* of
milestone M2.1.  With W_munu(dt) = <0|Phi_mu(t) Phi_nu(t')|0>, dt = t - t'
(the M2.1 kernel convention: ``LatticeWightman.smeared`` and
``ContinuumWightman3p1`` both return exactly this object; the continuum
normalization was checked against SMM17's Eq. (1) field convention and
Eq. (33) Fourier transform), the identifications are

    int d^n k |beta_nu|^2      = 4 lam_nu^2 eta_nu^2 W_nunu(0),
    int d^n k beta_A^* beta_B  = 4 lam_A lam_B eta_A eta_B W_AB(t_A - t_B),

hence

    f_nu  = exp(-2 lam_nu^2 eta_nu^2 W_nunu(0)),
    omega = -4 lam_A lam_B eta_A eta_B Re W_AB(t_A - t_B),
    theta = -2 lam_A lam_B eta_A eta_B Im W_AB(t_A - t_B),
    C_nu  = -lam_nu eta_nu <Phi_nu(t_nu)>_state,

the last being Eq. (32) rewritten: C_nu is minus the coupling times the
*classical* smeared field amplitude the coherent state carries at the
detector.  It enters rho_AB only through the unitary W of Eq. (73), which
is why the spectrum cannot depend on it (Theorem 2).

The null
--------
The partial transpose over A has the four eigenvalues [Eqs. (86)-(89)]

    E_1,2 = (1/8)[2 - 2 cosh(omega) f_A f_B
                  +/- sqrt(4|f_A e^{i theta} - f_B e^{-i theta}|^2
                           + 4 sinh^2(omega) f_A^2 f_B^2)],
    E_3,4 = (1/8)[2 + 2 cosh(omega) f_A f_B
                  +/- sqrt(4|f_A e^{i theta} + f_B e^{-i theta}|^2
                           + 4 sinh^2(omega) f_A^2 f_B^2)]

(SMM17 write (e^w +/- e^-w) where this module writes 2 cosh / 2 sinh).
They are invariant under omega -> -omega, which is exactly the map taking
Q^{t_A} [Eq. (85)] to Q [Eq. (84)]; so the partial transpose has the same
spectrum as rho_AB itself, which is a density matrix, so every E_i >= 0 and

    N[rho_AB] = sum_i max(0, -E_i) = 0                    [Eq. (79)]

identically -- Theorem 3.  Perturbatively the same statement is the
Cauchy-Schwarz inequality |W_AB(dt)|^2 <= W_AA(0) W_BB(0) on the smeared
Wightman function (expand E_2 to O(lam^4): the harvesting condition
|M| > P becomes |W_AB|^2 > W_AA W_BB), which is why the null is *marginal*
at weak coupling and needs the stabilized evaluation below.

Numerical hazards (logged, per the spec's discipline)
-----------------------------------------------------
E_2 is a difference of two O(lam^2) quantities whose leading terms cancel,
so the naive Eqs. (86)-(89) lose all significant digits as lam -> 0 and
can return a spuriously negative eigenvalue (the radicand itself goes
negative in float64 below lam ~ 1e-4).  Two rewrites fix it exactly:

    4|f_A e^{i th} -/+ f_B e^{-i th}|^2 = 4[(f_A -/+ f_B)^2 +/- 4 f_A f_B sin^2(th)]
                                              (identity, no cancellation),
    1 - cosh(om) f_A f_B = -expm1(-s) - 2 sinh^2(om/2) e^{-s},  s = a_A + a_B,

with a_nu = 2 lam_nu^2 eta_nu^2 W_nunu(0) = -ln f_nu carried *unexponentiated*
from the caller.  With those, the worst float64 eigenvalue over 21000
randomized draws on the 24-site chain of tests/test_nogo_communication.py,
spanning lam in [1e-8, 1] and dt in [-8, 8], is -1.3e-17 -- pure roundoff on
a quantity whose true value is ~1e-16, so N < 1e-16 everywhere, inside the
spec's 1e-14 gate with two orders of margin.  (Without the rewrites the same
sweep raises a domain error below lam ~ 1e-4: the radicand itself goes
negative.)  A ``dps`` escalation to mpmath is available
(:func:`pt_eigenvalues`, ``dps=...``) for the deep perturbative corner,
mirroring the ``vacuum.core.precision`` ladder.

Pure-functional over arrays throughout (frozen records, no mutation of
inputs) -- pre-positioning for the Layer 6 JAX mirror.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

__all__ = [
    "DeltaCoupledPair",
    "delta_coupled_pair",
    "delta_coupled_overlaps",
    "delta_coupled_rho",
    "partial_transpose_A",
    "pt_eigenvalues",
    "delta_coupled_negativity",
    "single_detector_rho",
    "single_detector_eigenvalues",
    "coherent_shift",
    "lattice_coherent_shifts",
    "wightman_triple",
    "SIMIDZIJA_BASIS",
]

#: Basis of SMM17 Eq. (52), in which every matrix here is expressed.  The
#: tildes absorb the free phases e^{i Omega_nu t_nu} [Eqs. (53), (54)], so
#: nothing in this module depends on the detector gaps.
SIMIDZIJA_BASIS = ("g~_A g~_B", "g~_A e~_B", "e~_A g~_B", "e~_A e~_B")

# (j, k) index patterns of Eq. (60), ordered to match SIMIDZIJA_BASIS: entry
# r of this tuple is (l, m) for the ket of row r and (j, k) for the bra of
# column r (see the module docstring / Eq. (56)).
_JK = ((+1, +1), (-1, +1), (+1, -1), (-1, -1))

# Swap permutation of the two qubits in SIMIDZIJA_BASIS: |g e> <-> |e g>.
_SWAP = (0, 2, 1, 3)


def _pair(value, name):
    """Normalize a scalar-or-(A, B) argument to two floats."""
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    if arr.size == 2:
        return float(arr[0]), float(arr[1])
    raise ValueError(f"{name} must be a scalar or a pair, got size {arr.size}")


# ---------------------------------------------------------------------------
# Kernel plumbing: the five scalars, from any M2.1 backend
# ---------------------------------------------------------------------------

class _ProfileHolder:
    """Minimal ``.F`` carrier so raw profiles reach :func:`wightman_triple`."""

    __slots__ = ("F",)

    def __init__(self, F):
        self.F = None if F is None else np.asarray(F, dtype=float)


def wightman_triple(kernel, F_A=None, F_B=None):
    """Resolve (W_AA, W_BB, W_AB) scalar kernels for a delta-coupled pair.

    Same dispatch as :func:`vacuum.detectors.udw.pair_kernels` (which this
    delegates to) but taking raw smearing profiles instead of
    :class:`~vacuum.detectors.udw.UDWDetector` objects, because a
    delta-coupled detector has no switching *function* -- only a switching
    *strength* eta_nu [SMM17 Eq. (45)].

    - :class:`~vacuum.detectors.kernels.LatticeWightman`: ``F_A``/``F_B``
      are the lattice profiles.
    - :class:`~vacuum.detectors.kernels.ContinuumWightman3p1`: the A-B
      *cross* kernel (its ``distance`` is the separation); both profiles
      must be None (the smearing is the kernel's sigma).
    - a 3-sequence ``(W_AA, W_BB, W_AB)``: used verbatim.
    """
    from .udw import pair_kernels  # local: keeps this module import-light

    return pair_kernels(kernel, _ProfileHolder(F_A), _ProfileHolder(F_B))


def delta_coupled_overlaps(W_AA0, W_BB0, W_AB, lam=1.0, eta=1.0):
    """The SMM17 scalars (a_A, a_B, omega, theta) from Wightman values.

    Parameters
    ----------
    W_AA0, W_BB0 : float
        Coincidence-limit smeared auto-correlations W_nunu(0) (real and
        positive: they are ``int d^n k |Ftilde_nu(k)|^2 / (2|k|)``).
    W_AB : complex
        Cross kernel at ``dt = t_A - t_B`` (the M2.1 sign convention,
        W(dt) = <Phi(t) Phi(t')>, dt = t - t').
    lam, eta : float or (A, B) pair
        Coupling strengths lam_nu and switching areas eta_nu of
        chi_nu(t) = eta_nu delta(t - t_nu) [Eq. (45)].

    Returns
    -------
    (a_A, a_B, omega, theta) : floats
        a_nu = 2 lam_nu^2 eta_nu^2 W_nunu(0) = -ln f_nu, i.e. the
        *unexponentiated* exponent of Eqs. (39)/(69) -- keeping it in this
        form is what makes the null evaluable at weak coupling (module
        docstring, "Numerical hazards").  omega is Eq. (81) and theta is
        Eq. (80), both via
        ``int d^n k beta_A^* beta_B = 4 lam_A lam_B eta_A eta_B W_AB``.
    """
    lam_A, lam_B = _pair(lam, "lam")
    eta_A, eta_B = _pair(eta, "eta")
    W_AA0 = float(np.real(W_AA0))
    W_BB0 = float(np.real(W_BB0))
    if W_AA0 < 0.0 or W_BB0 < 0.0:
        raise ValueError(
            "W_AA(0), W_BB(0) must be >= 0 (they are int |Ftilde|^2/2|k|); "
            f"got {W_AA0:.3e}, {W_BB0:.3e}"
        )
    a_A = 2.0 * lam_A * lam_A * eta_A * eta_A * W_AA0
    a_B = 2.0 * lam_B * lam_B * eta_B * eta_B * W_BB0
    J = 4.0 * lam_A * lam_B * eta_A * eta_B * complex(W_AB)
    omega = -J.real                     # Eq. (81)
    theta = -0.5 * J.imag               # Eq. (80)
    return a_A, a_B, omega, theta


def coherent_shift(lam, eta, phi_mean):
    """C_nu = -lam_nu eta_nu <Phi_nu(t_nu)> -- SMM17 Eq. (32) (and (58)).

    Eq. (32) reads

        C_A = -lam_A eta_A int d^n k (Ftilde_A(k) alpha(k)
                                      e^{-i(|k| t_A - k.x_A)} + c.c.)
                          / sqrt(2|k|),

    and the bracket is exactly <alpha| Phi_A(t_A) |alpha>, the *classical*
    smeared field amplitude the coherent state |alpha(k)> [Eq. (3)] carries
    at detector A's location and switching instant.  Equivalently
    D^dag Y_nu D = Y_nu + i C_nu, which is the only way the coherent
    amplitude enters the evolved state at all.  C_nu is real; the vacuum
    (alpha = 0) gives C_nu = 0.
    """
    return -float(lam) * float(eta) * float(phi_mean)


def lattice_coherent_shifts(K, profiles, x0, p0, times, lam=1.0, eta=1.0):
    """(C_A, C_B) for a displaced-vacuum (coherent) state of a lattice field.

    A coherent state of the lattice field H = (1/2)(p.p + x.K.x) is the
    vacuum displaced by a *classical* solution (xbar(t), pbar(t)) of the
    same quadratic dynamics, so <Phi_nu(t)> = F_nu . xbar(t) with
    xbar(t) = [S(t) (x0, p0)]_x and S the exact symplectic propagator
    ``vacuum.core.dynamics.chain_propagator``.  :func:`coherent_shift`
    then gives C_nu.

    Parameters
    ----------
    K : (N, N) array          lattice coupling matrix
    profiles : (F_A, F_B)     smearing profiles, each (N,)
    x0, p0 : (N,) arrays      classical initial displacement of the coherent state
    times : (t_A, t_B)        the two delta-switching instants
    lam, eta : float or pair  as in :func:`delta_coupled_overlaps`
    """
    from ..core.dynamics import chain_propagator  # local: avoid import cycles

    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    F_A, F_B = (np.asarray(p, dtype=float) for p in profiles)
    z0 = np.concatenate([np.asarray(x0, dtype=float), np.asarray(p0, dtype=float)])
    if z0.shape != (2 * N,):
        raise ValueError(f"x0, p0 must each have shape ({N},)")
    lam_A, lam_B = _pair(lam, "lam")
    eta_A, eta_B = _pair(eta, "eta")
    t_A, t_B = (float(t) for t in times)
    xbar_A = (chain_propagator(K, t_A) @ z0)[:N]
    xbar_B = (chain_propagator(K, t_B) @ z0)[:N]
    return (
        coherent_shift(lam_A, eta_A, float(F_A @ xbar_A)),
        coherent_shift(lam_B, eta_B, float(F_B @ xbar_B)),
    )


# ---------------------------------------------------------------------------
# The closed-form state and its spectra
# ---------------------------------------------------------------------------

def _f2(j, k, l, m, f_A, f_B, f_p, f_m, theta, C_A, C_B):
    """One matrix element f_2^{(jklm)} of SMM17 Eq. (60).

    Transcribed term by term; f_A, f_B are real [(39), (69)] but the
    conjugations of Eq. (60) are kept explicit so the transcription can be
    read against the paper.
    """
    e = np.exp
    tt = 2j * theta
    ca = 2j * C_A
    cb = 2j * C_B
    return (1.0 / 16.0) * (
        1 + j * l + k * m + j * k * l * m
        + l * np.conj(f_B) * (k * m * e(-tt) + e(tt)) * e(-cb)
        + m * np.conj(f_A) * (1 + j * l) * e(-ca)
        + j * f_B * (e(-tt) + k * m * e(tt)) * e(cb)
        + k * f_A * (1 + j * l) * e(ca)
        + j * k * f_p * e(tt) * e(ca) * e(cb)
        + m * l * np.conj(f_p) * e(-tt) * e(-ca) * e(-cb)
        + j * m * f_m * e(-tt) * e(-ca) * e(cb)
        + k * l * np.conj(f_m) * e(tt) * e(ca) * e(-cb)
    )


def delta_coupled_rho(f_A, f_B, omega, theta, C_A=0.0, C_B=0.0):
    """The evolved two-detector state rho_AB, SMM17 Eqs. (56) + (60).

    Returned in the basis :data:`SIMIDZIJA_BASIS` = Eq. (52), with detector
    A the *earlier* one (the derivation of Eq. (47) assumes t_A <= t_B);
    :func:`delta_coupled_pair` handles relabelling for the general case.

    f_p and f_m come from Eqs. (70), (71):
    f_p = f_A f_B e^{omega - 2 i theta}, f_m = f_A f_B e^{-omega + 2 i theta}.
    """
    f_A = float(f_A)
    f_B = float(f_B)
    omega = float(omega)
    theta = float(theta)
    f_p = f_A * f_B * np.exp(omega - 2j * theta)     # Eq. (70)
    f_m = f_A * f_B * np.exp(-omega + 2j * theta)    # Eq. (71)
    rho = np.empty((4, 4), dtype=complex)
    for r, (l, m) in enumerate(_JK):
        for c, (j, k) in enumerate(_JK):
            rho[r, c] = _f2(j, k, l, m, f_A, f_B, f_p, f_m, theta, C_A, C_B)
    return rho


def partial_transpose_A(rho):
    """Partial transpose over the first qubit of a 4x4 two-qubit matrix.

    (SMM17 Eq. (75) writes the same permutation out entry by entry.)
    """
    rho = np.asarray(rho)
    if rho.shape != (4, 4):
        raise ValueError(f"rho must be 4x4, got {rho.shape}")
    return rho.reshape(2, 2, 2, 2).transpose(2, 1, 0, 3).reshape(4, 4)


def _pt_eigenvalues_float(a_A, a_B, omega, theta):
    """Eqs. (86)-(89), stabilized (module docstring, "Numerical hazards")."""
    f_A = math.exp(-a_A)
    f_B = math.exp(-a_B)
    P = f_A * f_B                     # = exp(-(a_A + a_B))
    s = a_A + a_B
    sinh_om = math.sinh(omega)
    st2 = math.sin(theta) ** 2
    # 4|f_A e^{i th} -/+ f_B e^{-i th}|^2 = 4[(f_A -/+ f_B)^2 +/- 4 f_A f_B sin^2 th].
    # Both radicands are manifestly >= 0 -- rad_p >= (f_A - f_B)^2 because
    # sin^2 <= 1 -- so the clamps below only ever absorb roundoff, never a
    # genuine sign change (which would be a Theorem 3 violation, not a
    # numerical one, and would show up as a nonzero negativity).
    rad_m = (f_A - f_B) ** 2 + 4.0 * P * st2 + sinh_om * sinh_om * P * P
    rad_p = (f_A + f_B) ** 2 - 4.0 * P * st2 + sinh_om * sinh_om * P * P
    r_m = 2.0 * math.sqrt(max(rad_m, 0.0))
    r_p = 2.0 * math.sqrt(max(rad_p, 0.0))
    # 2 - 2 cosh(om) P  =  2[-expm1(-s) - 2 sinh^2(om/2) e^{-s}]   (no cancellation)
    head_minus = 2.0 * (-math.expm1(-s) - 2.0 * math.sinh(0.5 * omega) ** 2 * P)
    head_plus = 2.0 + 2.0 * math.cosh(omega) * P
    return (
        (head_minus + r_m) / 8.0,     # Eq. (86)
        (head_minus - r_m) / 8.0,     # Eq. (87)
        (head_plus + r_p) / 8.0,      # Eq. (88)
        (head_plus - r_p) / 8.0,      # Eq. (89)
    )


def _pt_eigenvalues_mp(a_A, a_B, omega, theta, dps):
    """mpmath evaluation of Eqs. (86)-(89) (precision-ladder escalation)."""
    import mpmath as mp

    with mp.workdps(int(dps)):
        a_A = mp.mpf(a_A)
        a_B = mp.mpf(a_B)
        om = mp.mpf(omega)
        th = mp.mpf(theta)
        f_A = mp.e ** (-a_A)
        f_B = mp.e ** (-a_B)
        P = f_A * f_B
        sh = mp.sinh(om)
        r_m = 2 * mp.sqrt(
            (f_A - f_B) ** 2 + 4 * P * mp.sin(th) ** 2 + sh**2 * P**2
        )
        r_p = 2 * mp.sqrt(
            (f_A + f_B) ** 2 - 4 * P * mp.sin(th) ** 2 + sh**2 * P**2
        )
        hm = 2 - 2 * mp.cosh(om) * P
        hp = 2 + 2 * mp.cosh(om) * P
        return tuple(
            float(x) for x in ((hm + r_m) / 8, (hm - r_m) / 8, (hp + r_p) / 8, (hp - r_p) / 8)
        )


def pt_eigenvalues(a_A, a_B, omega, theta, dps=None):
    """The four eigenvalues of rho_AB^{t_A}, SMM17 Eqs. (86)-(89).

    Parameters
    ----------
    a_A, a_B : float
        -ln f_nu = 2 lam_nu^2 eta_nu^2 W_nunu(0) (see
        :func:`delta_coupled_overlaps`).  Passing the exponent rather than
        f_nu itself is what keeps the O(lam^4) null resolvable.
    omega, theta : float
        Eqs. (81), (80).
    dps : int, optional
        Escalate to mpmath at this precision instead of float64.  The
        float64 path is already good to ~1e-16 absolute on the smallest
        eigenvalue (module docstring); ``dps`` is the audit hammer for the
        deep perturbative corner, mirroring ``vacuum.core.precision``.

    Returns
    -------
    (E_1, E_2, E_3, E_4) : floats, in the paper's order.  E_2 [Eq. (87)] is
    the only one that could conceivably be negative; Theorem 3 says it is
    not.
    """
    a_A = float(a_A)
    a_B = float(a_B)
    if a_A < 0.0 or a_B < 0.0:
        raise ValueError(
            f"a_nu = -ln f_nu must be >= 0 (f_nu <= 1 by Eq. (39)); "
            f"got {a_A:.3e}, {a_B:.3e}"
        )
    if dps is None:
        return _pt_eigenvalues_float(a_A, a_B, float(omega), float(theta))
    return _pt_eigenvalues_mp(a_A, a_B, float(omega), float(theta), dps)


def delta_coupled_negativity(a_A, a_B, omega, theta, dps=None):
    """N[rho_AB] = sum_i max(0, -E_i) -- SMM17 Eq. (79); Theorem 3 says 0.

    This is *the* exact null of the anchor suite: it is zero for every
    admissible argument, for the vacuum and for every coherent field state
    (the coherent amplitude does not even appear -- Theorem 2).
    """
    return float(sum(max(0.0, -E) for E in pt_eigenvalues(a_A, a_B, omega, theta, dps)))


def single_detector_rho(f_A, C_A=0.0):
    """One-detector evolved state, SMM17 Eq. (40), in the basis {g~_A, e~_A}.

        rho_A = (1/2) [[1 + f_A cos 2C_A,  -i f_A sin 2C_A],
                       [ i f_A sin 2C_A,   1 - f_A cos 2C_A]]

    Its eigenvalues (1 +/- f_A)/2 [Eqs. (41), (42)] are independent of the
    coherent amplitude -- Theorem 1.
    """
    f_A = float(f_A)
    c = f_A * math.cos(2.0 * float(C_A))
    s = f_A * math.sin(2.0 * float(C_A))
    return 0.5 * np.array([[1.0 + c, -1j * s], [1j * s, 1.0 - c]], dtype=complex)


def single_detector_eigenvalues(f_A):
    """(E_A1, E_A2) = ((1 + f_A)/2, (1 - f_A)/2) -- SMM17 Eqs. (41), (42)."""
    f_A = float(f_A)
    return 0.5 * (1.0 + f_A), 0.5 * (1.0 - f_A)


# ---------------------------------------------------------------------------
# The user-facing record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeltaCoupledPair:
    """Nonperturbative delta-coupled two-detector state (SMM17).

    All fields are in the *user's* A/B labelling.  ``swapped`` records
    whether t_B < t_A, in which case the Eq. (47) time ordering puts B
    first and the internal construction relabels before building rho (the
    returned ``rho`` is permuted back, so it is always in
    :data:`SIMIDZIJA_BASIS` for the user's A and B).

    Attributes
    ----------
    a_A, a_B : float      -ln f_nu = 2 lam_nu^2 eta_nu^2 W_nunu(0)
    omega, theta : float  Eqs. (81), (80), in the user's labelling
    C_A, C_B : float      coherent shifts, Eq. (32); 0 for the vacuum
    lam_A, lam_B, eta_A, eta_B, t_A, t_B : the construction parameters
    W_AA0, W_BB0, W_AB : the pulled-back Wightman values behind them
    meta : dict           provenance (backend, swap flag), pure data
    """

    a_A: float
    a_B: float
    omega: float
    theta: float
    C_A: float = 0.0
    C_B: float = 0.0
    lam_A: float = 1.0
    lam_B: float = 1.0
    eta_A: float = 1.0
    eta_B: float = 1.0
    t_A: float = 0.0
    t_B: float = 0.0
    W_AA0: float = 0.0
    W_BB0: float = 0.0
    W_AB: complex = 0.0
    swapped: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def f_A(self):
        """f_A = exp(-a_A) [SMM17 Eq. (39)]."""
        return math.exp(-self.a_A)

    @property
    def f_B(self):
        """f_B = exp(-a_B) [SMM17 Eq. (69)]."""
        return math.exp(-self.a_B)

    @property
    def rho(self):
        """rho_AB in :data:`SIMIDZIJA_BASIS` for the user's (A, B) labels."""
        if not self.swapped:
            return delta_coupled_rho(
                self.f_A, self.f_B, self.omega, self.theta, self.C_A, self.C_B
            )
        # t_B < t_A: build with B as the earlier detector (Eq. (47)), which
        # maps (f_A, f_B, omega, theta, C_A, C_B) -> (f_B, f_A, omega,
        # -theta, C_B, C_A) since theta = -(1/2) Im int beta_A^* beta_B is
        # odd under the relabelling, then undo the qubit swap.
        rho = delta_coupled_rho(
            self.f_B, self.f_A, self.omega, -self.theta, self.C_B, self.C_A
        )
        p = np.array(_SWAP)
        return rho[np.ix_(p, p)]

    @property
    def rho_pt(self):
        """Partial transpose of :attr:`rho` over detector A [Eq. (75)]."""
        return partial_transpose_A(self.rho)

    @property
    def pt_eigenvalues(self):
        """Analytic Eqs. (86)-(89) (invariant under the A/B relabelling)."""
        return pt_eigenvalues(self.a_A, self.a_B, self.omega, self.theta)

    @property
    def eigenvalues(self):
        """Spectrum of rho_AB.  Equal to :attr:`pt_eigenvalues` (SMM17: the
        Eq. (86)-(89) expressions are invariant under omega -> -omega, which
        is the map Q^{t_A} -> Q)."""
        return self.pt_eigenvalues

    @property
    def negativity(self):
        """N[rho_AB] [Eq. (79)] -- exactly 0 by Theorem 3."""
        return delta_coupled_negativity(self.a_A, self.a_B, self.omega, self.theta)

    @property
    def log_negativity(self):
        """E_N = ln(1 + 2N) in nats (docs/API.md) -- exactly 0 by Theorem 3."""
        return math.log1p(2.0 * self.negativity)

    def negativity_precise(self, dps=60):
        """:attr:`negativity` re-evaluated in mpmath at ``dps`` digits."""
        return delta_coupled_negativity(
            self.a_A, self.a_B, self.omega, self.theta, dps=dps
        )

    @property
    def excitation_probabilities(self):
        """(P_A, P_B) = (1 - f_nu cos 2C_nu)/2, from Eq. (40) per detector.

        These are *not* small: a delta coupling is nonperturbative.  They
        are what the negativity has to beat, and never does.
        """
        return (
            0.5 * (1.0 - self.f_A * math.cos(2.0 * self.C_A)),
            0.5 * (1.0 - self.f_B * math.cos(2.0 * self.C_B)),
        )


def delta_coupled_pair(
    kernel,
    F_A=None,
    F_B=None,
    lam=1.0,
    eta=1.0,
    times=(0.0, 0.0),
    coherent=(0.0, 0.0),
) -> DeltaCoupledPair:
    """Build the exact delta-coupled two-detector state (SMM17, Theorems 2-3).

    Parameters
    ----------
    kernel : LatticeWightman | ContinuumWightman3p1 | (W_AA, W_BB, W_AB)
        Any M2.1 backend; see :func:`wightman_triple`.
    F_A, F_B : (N,) arrays, optional
        Lattice smearing profiles (lattice backend only).
    lam, eta : float or (A, B) pair
        Coupling strengths and switching areas of
        chi_nu(t) = eta_nu delta(t - t_nu) [Eq. (45)].
    times : (t_A, t_B)
        The two switching instants.  Either order is accepted; the Eq. (47)
        ordering is applied internally and recorded in ``swapped``.
    coherent : (C_A, C_B)
        The coherent-amplitude shifts of Eq. (32) -- 0 for the vacuum.
        :func:`coherent_shift` / :func:`lattice_coherent_shifts` build them
        from a field state.  They cannot change any spectrum (Theorem 2),
        which is itself a test.

    Notes
    -----
    The detector energy gaps are deliberately absent: with delta switching
    the evolved state is *gap-independent* in the basis of Eq. (52) (SMM17
    Sec. IV) -- the interaction is so intense that the free dynamics never
    happen.  That is one of the two physical readings SMM17 offers for why
    the harvesting null holds.
    """
    ker_AA, ker_BB, ker_AB = wightman_triple(kernel, F_A, F_B)
    t_A, t_B = (float(t) for t in times)
    lam_A, lam_B = _pair(lam, "lam")
    eta_A, eta_B = _pair(eta, "eta")
    C_A, C_B = (float(c) for c in coherent)

    W_AA0 = complex(ker_AA(0.0))
    W_BB0 = complex(ker_BB(0.0))
    W_AB = complex(ker_AB(t_A - t_B))       # M2.1 convention: W(t - t')
    a_A, a_B, omega, theta = delta_coupled_overlaps(
        W_AA0.real, W_BB0.real, W_AB, (lam_A, lam_B), (eta_A, eta_B)
    )
    return DeltaCoupledPair(
        a_A=a_A,
        a_B=a_B,
        omega=omega,
        theta=theta,
        C_A=C_A,
        C_B=C_B,
        lam_A=lam_A,
        lam_B=lam_B,
        eta_A=eta_A,
        eta_B=eta_B,
        t_A=t_A,
        t_B=t_B,
        W_AA0=W_AA0.real,
        W_BB0=W_BB0.real,
        W_AB=W_AB,
        swapped=bool(t_B < t_A),
        meta={
            "source": "Simidzija & Martin-Martinez, PRD 96, 065008 (2017), "
            "arXiv:1707.00016, Eqs. (45)-(89)",
            "backend": type(kernel).__name__,
            "gap_independent": True,
        },
    )
