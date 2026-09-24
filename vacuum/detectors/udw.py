"""Perturbative UDW engine (Layer 2, milestone M2.3).

Second-order two-detector Unruh-DeWitt physics, transcribed from

    A. Pozas-Kerstjens and E. Martin-Martinez, "Harvesting correlations
    from the quantum vacuum", Phys. Rev. D 92, 064042 (2015),
    arXiv:1506.03081 -- referred to as PKMM below.

Provenance of the equation numbers
----------------------------------
The arXiv source tarball (arxiv.org/e-print/1506.03081) was fetched on
2026-09-01 and every equation cited below was read off the LaTeX source
(``main.tex``) with its number taken from the ar5iv rendering of the same
source (ar5iv.labs.arxiv.org/html/1506.03081), which carries the printed
equation tags.  The pairs (label in the source, printed number) used here
are:

    eq:hamiltonian (1)   eq:moment (2)     eq:field (3)     eq:evo (6)
    eq:seconddens (13)   basis (14)        Lmunu (15)       M (16)
    Lmunotint (17)       Mnotint (18)      smear F (19)     Ftilde (20)
    point (21)           gauss (22)        E(kappa,gamma) (28)
    LAAGS3D (29)         LABGS3D (30)      MGS3D (31)
    MGS3Dnonover (32)    M_coinc (37)      E_1 (66)
    N^(2) (67)           negat2equal (68)

The anchor plots of PKMM Fig. 2 are the arbiter of convention slips, so
each transcription carries its equation number next to the expression
(spec M2.3 discipline), and tests/test_udw.py pins the engine against
Eqs. (29)-(32)/(37) *and* against separation/gap death lines digitized
out of the Fig. 2 vector artwork.

Physics
-------
Two two-level detectors A, B, gaps Omega_nu, couplings lam_nu, switching
functions chi_nu(t), spatial profiles F_nu, coupled to the field vacuum
through the UDW Hamiltonian [PKMM Eq. (1)]

    H_I(t) = sum_nu lam_nu chi_nu(t) mu_nu(t) int d^n x F_nu(x - x_nu) phi(x, t),
    mu_nu(t) = sigma^+_nu e^{i Omega_nu t} + sigma^-_nu e^{-i Omega_nu t}   [Eq. (2)].

To second order in lam the joint detector density matrix, in the basis

    {|g_A g_B>, |e_A g_B>, |g_A e_B>, |e_A e_B>}   [PKMM Eq. (14)],

is [PKMM Eq. (13)]

    rho_AB = [[1 - L_AA - L_BB, 0,    0,    M*],
              [0,               L_AA, L_AB, 0 ],
              [0,               L_BA, L_BB, 0 ],
              [M,               0,    0,    0 ]]  + O(lam^4),

with P_A = L_AA, P_B = L_BB, C = L_AB (L_BA = C*).  The plan's
{|gg>, |ge>, |eg>, |ee>} matrix is the same state with the two middle
basis vectors swapped, so its (2,2) entry is P_B and its (3,2) entry is C.

Write Phi_nu(t) for the smeared field operator of detector nu and

    W_{mu nu}(t - t') = <0| Phi_mu(t) Phi_nu(t') |0>

for the pulled-back Wightman kernel of milestone M2.1 (an
``ExpSumKernel``: the continuum 3+1 backend, or ``LatticeWightman.smeared``).
Both spacetimes give a *symmetric* kernel, W_AB = W_BA -- on the lattice
because W(dt) = U diag(e^{-i omega dt}/2 omega) U^T is a symmetric matrix,
in 3+1 because the angular integral turns e^{i k.(x_A - x_B)} into
sinc(k d).  Then the matrix elements of Eq. (13) are, from Eqs. (15)-(18),

    P_nu = lam_nu^2 int dt int dt' chi_nu(t) chi_nu(t')
                     e^{-i Omega_nu (t - t')} W_nunu(t - t'),

    C    = lam_A lam_B int dt int dt' chi_A(t) chi_B(t')
                     e^{i Omega_A t} e^{-i Omega_B t'} W_AB(t' - t),

    M    = -lam_A lam_B int dt int_{-inf}^{t} dt' W_AB(t - t')
              [ chi_A(t) chi_B(t') e^{i(Omega_A t + Omega_B t')}
              + chi_B(t) chi_A(t') e^{i(Omega_B t + Omega_A t')} ].

Equivalence with PKMM's momentum-space form is immediate on inserting the
mode sum W_AB(dt) = sum_k c_k e^{-i w_k dt}: the M2.1 kernels are exactly
the |F~(k)|^2 e^{i k.(x_A - x_B)} / 2|k| integrands of Eqs. (17)-(18)
after the angular integral (see vacuum/detectors/kernels.py).  M
integrates over the time-ordered triangle t' < t [Eq. (18)]; per the M2.1
hazard note it is evaluated on a triangular rule whose inner integral
ends exactly at the diagonal (cumulative composite Gauss-Legendre;
cross-validated against ``kernels.triangle_rule`` in tests/test_udw.py).

Pair negativity to second order [PKMM Eqs. (66)-(67); spec M2.3]:

    E_1 = (1/2)[P_A + P_B - sqrt((P_A - P_B)^2 + 4 |M|^2)]      [Eq. (66)]
    N^(2) = -E_1,   N = max(0, N^(2))                            [Eq. (67)]
        = max(0, sqrt(|M|^2 + (P_A - P_B)^2/4) - (P_A + P_B)/2),
    E_N = ln(1 + 2N)   (nats; the two-qubit trace norm of the partial
                        transpose of Eq. (13) is 1 + 2N).

For identical detectors this reduces to N^(2) = |M| - P [PKMM Eq. (68)].

PKMM closed forms (3+1D, Gaussian switching + Gaussian smearing)
----------------------------------------------------------------
PKMM's Gaussian switching is chi(t) = e^{-(t - t_nu)^2 / T^2}
[Eq. (22)] -- NOTE this is *not* this repo's ``switching('gaussian', T)``
convention e^{-(t - t0)^2 / (2 T^2)}: PKMM's T maps to a Switching width
T/sqrt(2) (``PKMM_SWITCHING_WIDTH_FACTOR``).  The smearing
F(x) = e^{-x^2/sigma^2} / (pi^{3/2} sigma^3) [Eq. (19), F~ of Eq. (20)]
matches the M2.1 continuum backend's convention exactly.  With the
dimensionless variables alpha = Omega T, beta = d/T, gamma = Delta/T,
delta = sigma/T [PKMM Table I], the closed forms Eq. (29) [L_AA],
Eq. (30) [L_AB], Eq. (31) [|M| as a 1D kappa-integral], Eq. (32)
[|M_non|, non-overlapping switchings] and Eq. (37) [M_coinc, pointlike
and gamma = 0] are implemented below as ``pkmm_*`` reference functions.

All of them are stabilized through the Faddeeva function
w(z) = e^{-z^2} erfc(-i z) (``scipy.special.wofz``): erfc(z) = e^{-z^2} w(i z),
and w is evaluated only in the closed upper half-plane where |w| <= 1, the
reflection w(z) = 2 e^{-z^2} - w(-z) being used to fold the lower half-plane
back up with the growing Gaussians cancelled analytically.  No intermediate
overflow occurs at any parameter these functions accept.

Everything here is pure-functional over arrays (frozen inputs, no
mutation) -- pre-positioning for the Layer 6 JAX mirror.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import dawsn, erfcx, wofz

from .kernels import (
    ContinuumWightman3p1,
    LatticeWightman,
    derivative_kernel,
    detector_response,
    gl_panels,
    wightman_continuum_3p1,
)
from .switching import switching

__all__ = [
    "COUPLINGS",
    "UDWDetector",
    "UDWPairState",
    "pair_kernels",
    "transition_probability",
    "cross_term",
    "pair_term",
    "pair_negativity",
    "pair_log_negativity",
    "udw_pair_state",
    "PKMM_SWITCHING_WIDTH_FACTOR",
    "PKMM_NONOVERLAP_GAMMA",
    "pkmm_L_AA",
    "pkmm_L_AB",
    "pkmm_M_abs",
    "pkmm_M_abs_nonoverlap",
    "pkmm_M_abs_coincident_pointlike",
    "pkmm_negativity_estimator",
    "pkmm_detectors",
    "pkmm_pair_state",
]

_SQRT2 = math.sqrt(2.0)
_SQRT_PI = math.sqrt(math.pi)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

#: PKMM's Gaussian switching chi = e^{-(t - t0)^2 / T^2} [Eq. (22)] equals
#: this repo's switching('gaussian', T * PKMM_SWITCHING_WIDTH_FACTOR, t0),
#: because Switching('gaussian', Ts) is e^{-(t - t0)^2 / (2 Ts^2)}.
PKMM_SWITCHING_WIDTH_FACTOR = 1.0 / _SQRT2

#: PKMM's non-overlap threshold for the two Gaussian switchings: their
#: standard deviation is s_T = T/sqrt(2), and the paper takes
#: Delta = t_B - t_A >= 7 T / sqrt(2) (seven standard deviations, overlap
#: suppressed by e^{-49/2} ~ 1e-11) as the regime where Eq. (32) applies
#: (text between Eqs. (31) and (32)).
PKMM_NONOVERLAP_GAMMA = 7.0 / _SQRT2

# Read-only cache of Gauss-Legendre nodes/weights on [-1, 1].  Memoized
# pure data (the arrays are frozen), not hidden mutable state.
_GL_CACHE: Dict[int, Any] = {}


def _readonly(a, dtype=float):
    out = np.array(a, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


def _leggauss_cached(n_gl):
    n_gl = int(n_gl)
    if n_gl < 2:
        raise ValueError(f"n_gl must be >= 2, got {n_gl}")
    if n_gl not in _GL_CACHE:
        x, w = leggauss(n_gl)
        _GL_CACHE[n_gl] = (_readonly(x), _readonly(w))
    return _GL_CACHE[n_gl]


# ---------------------------------------------------------------------------
# Detector spec and kernel dispatch
# ---------------------------------------------------------------------------

class UDWDetector:
    """Frozen spec of one UDW detector: switching profile + lattice smearing.

    Parameters
    ----------
    chi : Switching (or any callable with ``.support`` and ``.width``)
        The switching function chi_nu(t) [PKMM Eq. (1)].
    F : optional (N,) profile (ndarray or vacuum.detectors.Smearing)
        Lattice spatial smearing.  Required for the lattice backend
        (:class:`~vacuum.detectors.kernels.LatticeWightman`); must be
        omitted for the continuum backend, whose Gaussian smearing is
        baked into the kernel's sigma.
    """

    __slots__ = ("chi", "F")

    def __init__(self, chi, F=None):
        if not callable(chi) or not hasattr(chi, "support") or not hasattr(chi, "width"):
            raise TypeError(
                "chi must be a Switching-like callable with .support and .width "
                "(use vacuum.detectors.switching(...))"
            )
        object.__setattr__(self, "chi", chi)
        object.__setattr__(self, "F", None if F is None else _readonly(np.asarray(F)))

    def __setattr__(self, name, value):  # frozen
        raise AttributeError("UDWDetector objects are immutable")

    def __repr__(self):
        shape = "None" if self.F is None else f"F{self.F.shape}"
        return f"UDWDetector(chi={self.chi!r}, {shape})"


def _pair_params(lam, gap):
    """Normalize scalar-or-pair coupling and gap arguments -> 4 floats."""
    lam_arr = np.atleast_1d(np.asarray(lam, dtype=float))
    gap_arr = np.atleast_1d(np.asarray(gap, dtype=float))
    if lam_arr.size == 1:
        lam_A = lam_B = float(lam_arr[0])
    elif lam_arr.size == 2:
        lam_A, lam_B = float(lam_arr[0]), float(lam_arr[1])
    else:
        raise ValueError(f"lam must be a scalar or a pair, got size {lam_arr.size}")
    if gap_arr.size == 1:
        gap_A = gap_B = float(gap_arr[0])
    elif gap_arr.size == 2:
        gap_A, gap_B = float(gap_arr[0]), float(gap_arr[1])
    else:
        raise ValueError(f"gap must be a scalar or a pair, got size {gap_arr.size}")
    return lam_A, lam_B, gap_A, gap_B


#: Field-detector couplings the perturbative engine knows.  'xx' is the
#: amplitude (monopole-phi) coupling of PKMM Eq. (1); 'xp' is the derivative
#: coupling to d_t phi (Teixido-Bonfill & Martin-Martinez, PRD 110, 105016
#: (2024), Eq. (2)), for which every matrix element below is obtained by
#: replacing the Wightman kernel W by d_t d_t' W (see
#: ``vacuum.detectors.udw_derivative``).
COUPLINGS = ("xx", "xp")


def _resolve_pair_kernels(kernel, det_A, det_B):
    """Resolve (W_AA, W_BB, W_AB) amplitude-coupling kernels for a pair.

    Accepted ``kernel`` values (spec M2.3: both M2.1 backends plug in):

    - :class:`~vacuum.detectors.kernels.LatticeWightman`: ``det_A.F`` /
      ``det_B.F`` are the smearing profiles; returns
      ``kernel.smeared(F_A)``, ``kernel.smeared(F_B)``,
      ``kernel.smeared(F_A, F_B)``.  This is the native spacetime of the
      circuit-QED digital twin (a microwave waveguide is a 1+1 field).
    - :class:`~vacuum.detectors.kernels.ContinuumWightman3p1`: interpreted
      as the A-B *cross* kernel (its ``distance`` is the detector
      separation); the two auto-correlation kernels are rebuilt at
      distance 0 with the same sigma / kmax / panel / n_gl.  Detectors
      must not carry F (the smearing is the kernel's sigma).
    - a 3-sequence ``(W_AA, W_BB, W_AB)`` of ExpSumKernel-like objects
      (each with ``.freqs``/``.amps``): used verbatim.  This is the escape
      hatch M2.5's communication split uses -- the imaginary
      (commutator) part of W is itself an exponential sum on the doubled
      frequency set {+w_k, -w_k}.
    """
    if isinstance(kernel, LatticeWightman):
        if det_A.F is None or det_B.F is None:
            raise ValueError(
                "lattice backend: both detectors need a smearing profile F "
                "(vacuum.detectors.smearing(...) or a plain (N,) array)"
            )
        F_A = np.asarray(det_A.F, dtype=float)
        F_B = np.asarray(det_B.F, dtype=float)
        return kernel.smeared(F_A), kernel.smeared(F_B), kernel.smeared(F_A, F_B)
    if isinstance(kernel, ContinuumWightman3p1):
        if det_A.F is not None or det_B.F is not None:
            raise ValueError(
                "continuum backend: smearing is the kernel's sigma; "
                "detectors must not carry a lattice profile F"
            )
        auto = ContinuumWightman3p1(
            kernel.sigma, 0.0, kmax=kernel.kmax, panel=kernel.panel, n_gl=kernel.n_gl
        )
        return auto, auto, kernel
    try:
        k3 = tuple(kernel)
    except TypeError:
        k3 = None
    if k3 is not None and len(k3) == 3 and all(
        hasattr(k, "freqs") and hasattr(k, "amps") for k in k3
    ):
        return k3
    raise TypeError(
        "kernel must be a LatticeWightman, a ContinuumWightman3p1 (the A-B "
        "cross kernel), or a (W_AA, W_BB, W_AB) triple of ExpSumKernel-like "
        f"objects; got {type(kernel).__name__}"
    )


def pair_kernels(kernel, det_A, det_B, coupling="xx"):
    """Resolve (W_AA, W_BB, W_AB) scalar kernels for a detector pair.

    Accepted ``kernel`` values (spec M2.3: both M2.1 backends plug in) —
    see :func:`_resolve_pair_kernels` for the three forms (a
    :class:`~vacuum.detectors.kernels.LatticeWightman` with per-detector
    smearing profiles, a :class:`~vacuum.detectors.kernels.ContinuumWightman3p1`
    cross kernel, or a verbatim ``(W_AA, W_BB, W_AB)`` triple).

    ``coupling='xx'`` returns the amplitude-coupling kernels as resolved;
    ``coupling='xp'`` returns their derivative kernels d_t d_t' W
    (:func:`vacuum.detectors.kernels.derivative_kernel`), the pulled-back
    pi-pi correlators of the derivative-coupled detector.  A verbatim
    triple is treated as *amplitude* kernels in both cases — pass
    ``coupling='xx'`` with an already-differentiated triple to use it as is.
    """
    if coupling not in COUPLINGS:
        raise ValueError(f"coupling must be one of {COUPLINGS}, got {coupling!r}")
    triple = _resolve_pair_kernels(kernel, det_A, det_B)
    if coupling == "xx":
        return triple
    return tuple(derivative_kernel(k) for k in triple)


# ---------------------------------------------------------------------------
# Convergence gates (M2.1 hazard notes, applied to every quadrature here)
# ---------------------------------------------------------------------------

def _initial_panel(gaps, widths, panel0=None):
    """Panel width min(1/max|Omega|, min switching width)/8 (M2.1 hazard note).

    The kernel's own top frequency w_max is deliberately *not* folded in:
    the hazard note fixes this rule, and the halving gate below is what
    certifies the result (empirically the rule is already conservative --
    the continuum engine reproduces PKMM Eqs. (29)-(31) to 1e-15 from it).
    Callers who want a finer start -- a kernel with a very high w_max, say
    -- pass ``panel0`` explicitly.
    """
    if panel0 is not None:
        panel0 = float(panel0)
        if not panel0 > 0.0:
            raise ValueError(f"panel0 must be > 0, got {panel0}")
        return panel0
    om = max(abs(float(g)) for g in gaps)
    width = min(float(w) for w in widths)
    return (min(1.0 / om, width) / 8.0) if om > 0.0 else width / 8.0


def _doubling_gate(evaluate, h0, rtol, atol, max_doublings, what):
    """Halve the time-panel width until successive quadratures agree.

    Returns (value, h, n_halvings).  Raises RuntimeError if the doubling
    test does not converge -- silence is never an option here (spec).
    """
    if int(max_doublings) < 2:
        raise ValueError(f"max_doublings must be >= 2, got {max_doublings}")
    prev = None
    val = None
    last_change = None  # recorded BEFORE prev is overwritten, else it is always 0
    h = float(h0)
    for i in range(int(max_doublings)):
        val = evaluate(h)
        if prev is not None:
            last_change = abs(val - prev)
            if last_change <= max(rtol * abs(val), atol):
                return val, h, i
        prev = val
        h *= 0.5
    raise RuntimeError(
        f"{what}: time-quadrature doubling did not converge after "
        f"{max_doublings} halvings (last change "
        f"{float('nan') if last_change is None else last_change:.3e}, "
        f"|value| {abs(val):.3e}, target "
        f"{max(rtol * abs(val), atol):.3e}; an analytically-zero integral "
        f"needs an absolute scale — pass atol)"
    )


def _kernel_gate(kernel, evaluate, val, rtol, atol, max_doublings, what):
    """Double the kernel's own k-quadrature until the observable stops moving.

    Only the continuum backend carries ``with_refined``; lattice kernels
    are exact normal-mode sums with nothing to refine, so this is a no-op
    there.  Returns (value, n_refinements, n_modes).
    """
    if not hasattr(kernel, "with_refined"):
        return val, 0, int(np.asarray(kernel.freqs).size)
    ker = kernel
    for i in range(int(max_doublings)):
        ker = ker.with_refined(2)
        v2 = evaluate(ker)
        done = abs(v2 - val) <= max(rtol * abs(v2), atol)
        val = v2
        if done:
            return val, i + 1, int(ker.freqs.size)
    raise RuntimeError(
        f"{what}: kernel k-refinement did not converge after "
        f"{max_doublings} doublings"
    )


# ---------------------------------------------------------------------------
# The three matrix elements: P_nu, C, M
# ---------------------------------------------------------------------------

def transition_probability(kernel_auto, chi, gap, lam=1.0, **response_kwargs):
    """Single-detector transition probability P_nu = L_nunu [PKMM Eqs. (15), (17)].

    P_nu = lam^2 int dt dt' chi(t) chi(t') e^{-i Omega (t - t')} W(t - t')

    for the detector's *auto-correlation* kernel W.  The doubly-gated
    quadrature (time-panel halving, then kernel k-refinement) is delegated
    to :func:`vacuum.detectors.kernels.detector_response`, whose closed
    form ``response_gaussian_closed_form`` is analytically the same object
    as PKMM Eq. (29) under the parameter map
    (T_PKMM = sqrt(2) T_switching, sigma = delta T_PKMM) -- both identities
    are pinned in tests/test_udw.py.
    """
    lam = float(lam)
    return lam * lam * detector_response(kernel_auto, chi, float(gap), **response_kwargs)


def _time_vector(freqs, Omega, t, g, chunk):
    """A_k = sum_i g_i e^{i (Omega + w_k) t_i} for all kernel modes k."""
    out = np.empty(freqs.size, dtype=complex)
    for start in range(0, freqs.size, chunk):
        f = freqs[start : start + chunk]
        out[start : start + chunk] = (
            np.exp(1j * (Omega + f)[:, None] * t[None, :]) @ g
        )
    return out


def _cross_quad(freqs, amps, chi_A, chi_B, gap_A, gap_B, winA, winB, h, n_gl, chunk):
    """One evaluation of C/(lam_A lam_B) at fixed panel width h.

    From PKMM Eqs. (15) + (17), with the kernel mode sum
    W_AB(dt) = sum_k c_k e^{-i w_k dt}:

        C = sum_k c_k A_k conj(B_k),
        A_k = int dt chi_A(t) e^{i (w_k + Omega_A) t},
        B_k = int dt chi_B(t) e^{i (w_k + Omega_B) t},

    which is bitwise the same quadrature as the O(n_t^2) double sum, at
    O(n_k n_t) cost.
    """
    tA, wA = gl_panels(winA[0], winA[1], h, n_gl=n_gl)
    tB, wB = gl_panels(winB[0], winB[1], h, n_gl=n_gl)
    gA = wA * np.asarray(chi_A(tA), dtype=float)
    gB = wB * np.asarray(chi_B(tB), dtype=float)
    A = _time_vector(freqs, gap_A, tA, gA, chunk)
    B = _time_vector(freqs, gap_B, tB, gB, chunk)
    return complex(np.sum(amps * A * np.conj(B)))


def cross_term(
    kernel_AB,
    chi_A,
    chi_B,
    gap_A,
    gap_B,
    lam_A=1.0,
    lam_B=1.0,
    rtol=1e-10,
    atol=1e-18,
    n_gl=16,
    max_doublings=12,
    refine_kernel=True,
    window=None,
    panel0=None,
    chunk=64,
    return_meta=False,
):
    """Cross term C = L_AB [PKMM Eqs. (15), (17); element (2,3) of Eq. (13)].

    C = lam_A lam_B int dt int dt' chi_A(t) chi_B(t')
                     e^{i Omega_A t} e^{-i Omega_B t'} W_AB(t' - t)

    Doubling convergence per the M2.1 hazard note: Gauss-Legendre panels
    in time of initial width min(1/max|Omega|, min switching width)/8,
    halved until successive values agree to rtol/atol; then, for kernels
    that carry their own k-quadrature (continuum backend), that quadrature
    is doubled under the same criterion.  ``window`` overrides the two
    switching supports as ``((aA, bA), (aB, bB))``; ``panel0`` overrides the
    starting panel width (the gate then only has to confirm it).
    """
    freqs, amps = kernel_AB.freqs, kernel_AB.amps
    gap_A, gap_B = float(gap_A), float(gap_B)
    if window is not None:
        (aA, bA), (aB, bB) = window
        winA, winB = (float(aA), float(bA)), (float(aB), float(bB))
    else:
        winA = (float(chi_A.support[0]), float(chi_A.support[1]))
        winB = (float(chi_B.support[0]), float(chi_B.support[1]))
    h0 = _initial_panel((gap_A, gap_B), (chi_A.width, chi_B.width), panel0)

    val, h, n_halv = _doubling_gate(
        lambda h: _cross_quad(
            freqs, amps, chi_A, chi_B, gap_A, gap_B, winA, winB, h, n_gl, chunk
        ),
        h0, rtol, atol, max_doublings, "cross_term",
    )
    n_ref, n_modes = 0, int(np.asarray(freqs).size)
    if refine_kernel:
        val, n_ref, n_modes = _kernel_gate(
            kernel_AB,
            lambda ker: _cross_quad(
                ker.freqs, ker.amps, chi_A, chi_B, gap_A, gap_B, winA, winB,
                h, n_gl, chunk,
            ),
            val, rtol, atol, max_doublings, "cross_term",
        )
    out = float(lam_A) * float(lam_B) * val
    if not return_meta:
        return out
    return out, {
        "panel": h,
        "halvings": n_halv,
        "kernel_refinements": n_ref,
        "kernel_modes": n_modes,
    }


def _pair_quad(
    freqs, amps, chi_A, chi_B, gap_A, gap_B, a, b, h, n_gl, n_gl_inner, chunk
):
    """One evaluation of M/(-lam_A lam_B) at fixed outer panel width h.

    Time-ordered double integral over the triangle t' < t of
    W_AB(t - t') [chi_A(t) chi_B(t') e^{i(Omega_A t + Omega_B t')} + (A<->B)]
    [PKMM Eq. (18)], with W_AB(dt) = sum_k c_k e^{-i w_k dt}:

        M/(-lam_A lam_B) = sum_k c_k [ sum_i v1_i e^{-i w_k t_i} I1_k(t_i)
                                     + sum_i v2_i e^{-i w_k t_i} I2_k(t_i) ],
        I_k(t_i) = int_a^{t_i} dt' u(t') e^{i w_k t'}.

    Triangular rule per the M2.1 hazard note (never a masked square): the
    outer variable t gets GL panels on [a, b]; the inner cumulative
    integral is a *composite* GL rule on the segments between consecutive
    outer nodes, so every inner rule ends exactly at the diagonal t' = t_i
    and no order of accuracy is lost there.  Cross-validated against
    kernels.triangle_rule in tests/test_udw.py.
    """
    t, w = gl_panels(a, b, h, n_gl=n_gl)
    m = t.size
    # inner GL nodes on the segments [a, t_0], [t_0, t_1], ..., [t_{m-2}, t_{m-1}]
    edges = np.concatenate(([a], t))
    xg, wg = _leggauss_cached(n_gl_inner)
    mid = 0.5 * (edges[1:] + edges[:-1])
    half = 0.5 * (edges[1:] - edges[:-1])
    tp = (mid[:, None] + half[:, None] * xg[None, :]).ravel()      # (m * n_in,)
    wp = (half[:, None] * wg[None, :]).ravel()

    chiA_t = np.asarray(chi_A(t), dtype=float)
    chiB_t = np.asarray(chi_B(t), dtype=float)
    chiA_tp = np.asarray(chi_A(tp), dtype=float)
    chiB_tp = np.asarray(chi_B(tp), dtype=float)

    # inner weights u, outer weights v; term1 = (A outer, B inner), term2 swapped
    u1 = wp * chiB_tp * np.exp(1j * gap_B * tp)
    u2 = wp * chiA_tp * np.exp(1j * gap_A * tp)
    v1 = w * chiA_t * np.exp(1j * gap_A * t)
    v2 = w * chiB_t * np.exp(1j * gap_B * t)

    acc = 0.0 + 0.0j
    for start in range(0, freqs.size, chunk):
        f = freqs[start : start + chunk]
        c = amps[start : start + chunk]
        Zin = np.exp(1j * f[:, None] * tp[None, :])        # e^{+i w_k t'}
        Zout = np.exp(-1j * f[:, None] * t[None, :])       # e^{-i w_k t}
        # cumulative inner integrals I(t_i) = int_a^{t_i} dt' u(t') e^{i w t'}
        I1 = np.cumsum((Zin * u1[None, :]).reshape(-1, m, n_gl_inner).sum(axis=2), axis=1)
        I2 = np.cumsum((Zin * u2[None, :]).reshape(-1, m, n_gl_inner).sum(axis=2), axis=1)
        T1 = np.sum(Zout * v1[None, :] * I1, axis=1)
        T2 = np.sum(Zout * v2[None, :] * I2, axis=1)
        acc += np.sum(c * (T1 + T2))
    return acc


def pair_term(
    kernel_AB,
    chi_A,
    chi_B,
    gap_A,
    gap_B,
    lam_A=1.0,
    lam_B=1.0,
    rtol=1e-8,
    atol=1e-18,
    n_gl=16,
    n_gl_inner=4,
    max_doublings=10,
    refine_kernel=True,
    window=None,
    panel0=None,
    chunk=48,
    return_meta=False,
):
    """Pair term M [PKMM Eqs. (16), (18); elements (1,4)/(4,1) of Eq. (13)].

    M = -lam_A lam_B int dt int_{-inf}^{t} dt' W_AB(t - t')
          [chi_A(t) chi_B(t') e^{i(Omega_A t + Omega_B t')} + (A<->B)],

    on the union of the two switching supports (override with ``window =
    (a, b)``), with the outer-panel halving gate (starting width
    overridable with ``panel0``) and, for the continuum backend, the
    kernel k-refinement gate.  ``n_gl_inner`` is the order of
    the composite rule used for the cumulative inner integral; the inner
    segments are sub-panel sized, so 4 is ample and the outer halving gate
    also shrinks them, which makes the doubling test cover both.
    """
    freqs, amps = kernel_AB.freqs, kernel_AB.amps
    gap_A, gap_B = float(gap_A), float(gap_B)
    if window is not None:
        a, b = float(window[0]), float(window[1])
    else:
        aA, bA = chi_A.support
        aB, bB = chi_B.support
        a, b = min(float(aA), float(aB)), max(float(bA), float(bB))
    h0 = _initial_panel((gap_A, gap_B), (chi_A.width, chi_B.width), panel0)

    val, h, n_halv = _doubling_gate(
        lambda h: _pair_quad(
            freqs, amps, chi_A, chi_B, gap_A, gap_B, a, b, h, n_gl, n_gl_inner, chunk
        ),
        h0, rtol, atol, max_doublings, "pair_term",
    )
    n_ref, n_modes = 0, int(np.asarray(freqs).size)
    if refine_kernel:
        val, n_ref, n_modes = _kernel_gate(
            kernel_AB,
            lambda ker: _pair_quad(
                ker.freqs, ker.amps, chi_A, chi_B, gap_A, gap_B, a, b, h, n_gl,
                n_gl_inner, chunk,
            ),
            val, rtol, atol, max_doublings, "pair_term",
        )
    out = -float(lam_A) * float(lam_B) * val
    if not return_meta:
        return out
    return out, {
        "panel": h,
        "halvings": n_halv,
        "kernel_refinements": n_ref,
        "kernel_modes": n_modes,
        "n_gl_inner": int(n_gl_inner),
    }


# ---------------------------------------------------------------------------
# Pair state, negativity
# ---------------------------------------------------------------------------

def pair_negativity(P_A, P_B, M):
    """Second-order pair negativity N [PKMM Eqs. (66)-(67); spec M2.3].

    N = max(0, sqrt(|M|^2 + (P_A - P_B)^2/4) - (P_A + P_B)/2),

    i.e. N = max(0, -E_1) with E_1 the single eigenvalue of the partial
    transpose of PKMM Eq. (13) that can be negative at this order
    [Eq. (66)].  For P_A = P_B this is max(0, |M| - P) [Eq. (68)].
    """
    P_A, P_B = float(P_A), float(P_B)
    return max(0.0, math.hypot(abs(M), 0.5 * (P_A - P_B)) - 0.5 * (P_A + P_B))


def pair_log_negativity(P_A, P_B, M):
    """E_N = ln(1 + 2N) in nats (spec M2.3; ||rho^Gamma||_1 = 1 + 2N here)."""
    return math.log1p(2.0 * pair_negativity(P_A, P_B, M))


@dataclass(frozen=True)
class UDWPairState:
    """Second-order two-detector state [PKMM Eq. (13)] plus metadata.

    ``P_A``, ``P_B`` are the transition probabilities (L_AA, L_BB), ``C``
    the cross term L_AB, ``M`` the pair term; ``rho`` assembles the 4x4
    matrix in the PKMM Eq. (14) basis
    {|g_A g_B>, |e_A g_B>, |g_A e_B>, |e_A e_B>}.  ``meta`` records
    quadrature provenance (pure data, no behavior) so every sweep row can
    carry the settings at which it converged (plan, "pre-positioning").

    ``comm`` carries the M2.5 communication split
    (:class:`vacuum.detectors.communication.CommunicationSplit`) when one
    has been computed for this state, and is ``None`` otherwise; build a
    state with it filled through
    :func:`vacuum.detectors.communication.pair_state_with_split`.  It
    supplies the (M_vac, M_comm, |M_comm|/|M|) entries of :attr:`row`, the
    permanent results-table columns of spec M2.5.
    """

    P_A: float
    P_B: float
    C: complex
    M: complex
    lam_A: float
    lam_B: float
    gap_A: float
    gap_B: float
    meta: Dict[str, Any] = field(default_factory=dict)
    comm: Any = None

    @property
    def row(self):
        """One results-table row: the permanent Layer 2 columns (spec M2.5).

        Beyond the matrix elements the schema mandates ``E_N``, ``M_vac``,
        ``M_comm`` and ``|M_comm| / |M|``; the three communication columns
        are ``None`` until a
        :class:`~vacuum.detectors.communication.CommunicationSplit` has
        been attached to ``comm`` (see
        :func:`vacuum.detectors.communication.pair_state_with_split`), so a
        row can never silently claim a split it did not compute.  Plain
        data only -- this is what a sweep writes to disk.
        """
        cols = {
            "P_A": self.P_A,
            "P_B": self.P_B,
            "C": self.C,
            "M": self.M,
            "E_N": self.log_negativity,
            "negativity": self.negativity,
            "M_vac": None,
            "M_comm": None,
            "comm_fraction": None,
            "lam_A": self.lam_A,
            "lam_B": self.lam_B,
            "gap_A": self.gap_A,
            "gap_B": self.gap_B,
            "backend": self.meta.get("backend"),
            "perturbative_scale": self.meta.get("perturbative_scale"),
            "perturbative_ok": self.meta.get("perturbative_ok"),
        }
        if self.comm is not None:
            cols["M_vac"] = self.comm.M_vac
            cols["M_comm"] = self.comm.M_comm
            cols["comm_fraction"] = self.comm.comm_fraction
        return cols

    @property
    def rho(self):
        """4x4 density matrix, PKMM Eq. (13) in the Eq. (14) basis."""
        r = np.zeros((4, 4), dtype=complex)
        r[0, 0] = 1.0 - self.P_A - self.P_B
        r[1, 1] = self.P_A
        r[2, 2] = self.P_B
        r[1, 2] = self.C
        r[2, 1] = np.conj(self.C)
        r[0, 3] = np.conj(self.M)
        r[3, 0] = self.M
        return r

    @property
    def negativity_estimator(self):
        """N^(2) = -E_1, without the max(0, .) clamp [PKMM Eq. (67)]."""
        return math.hypot(abs(self.M), 0.5 * (self.P_A - self.P_B)) - 0.5 * (
            self.P_A + self.P_B
        )

    @property
    def negativity(self):
        """N = max(0, N^(2)) [PKMM Eq. (67)]."""
        return pair_negativity(self.P_A, self.P_B, self.M)

    @property
    def log_negativity(self):
        """E_N = ln(1 + 2N), nats (spec M2.3)."""
        return math.log1p(2.0 * self.negativity)


def udw_pair_state(kernel, det_A, det_B, lam, gap, coupling="xx", **quad_kwargs):
    """Second-order two-detector UDW state from either M2.1 kernel backend.

    Parameters
    ----------
    kernel : LatticeWightman | ContinuumWightman3p1 | (W_AA, W_BB, W_AB)
        See :func:`pair_kernels`.  The same code runs continuum 3+1 (the
        PKMM anchors) and lattice 1+1 (the circuit-QED digital twin).
    det_A, det_B : UDWDetector
        Switching (and, on the lattice backend, smearing) of each detector.
    lam, gap : float or (A, B) pair
        Coupling strengths lam_nu and gaps Omega_nu.
    coupling : {'xx', 'xp'}
        'xx': the amplitude coupling of PKMM Eq. (1).  'xp': the derivative
        coupling to d_t phi — identical second-order algebra with W replaced
        by d_t d_t' W (``vacuum.detectors.udw_derivative``); the resulting
        state is what ``attach_detectors(..., coupling='xp')`` reproduces
        nonperturbatively (tests/test_gate_xp.py).
    quad_kwargs :
        Forwarded to the quadratures: ``rtol``, ``atol``, ``n_gl``,
        ``max_doublings``, ``refine_kernel`` (all three integrals), plus
        ``chunk`` (C and M) and ``n_gl_inner`` (M).

    Returns
    -------
    UDWPairState
        with P_A, P_B [Eqs. (15), (17)], C = L_AB [Eq. (30) in 3+1
        Gaussian-Gaussian], M [Eqs. (16), (18)], the rho of Eq. (13), and
        the negativity of Eqs. (66)-(67).
    """
    lam_A, lam_B, gap_A, gap_B = _pair_params(lam, gap)
    ker_AA, ker_BB, ker_AB = pair_kernels(kernel, det_A, det_B, coupling=coupling)

    resp_keys = ("rtol", "atol", "n_gl", "max_doublings", "refine_kernel")
    unknown = set(quad_kwargs) - set(resp_keys) - {"chunk", "n_gl_inner"}
    if unknown:
        raise TypeError(f"unknown quadrature options: {sorted(unknown)}")
    resp_kwargs = {k: v for k, v in quad_kwargs.items() if k in resp_keys}
    cross_kwargs = dict(resp_kwargs)
    if "chunk" in quad_kwargs:
        cross_kwargs["chunk"] = quad_kwargs["chunk"]
    pair_kwargs = dict(cross_kwargs)
    if "n_gl_inner" in quad_kwargs:
        pair_kwargs["n_gl_inner"] = quad_kwargs["n_gl_inner"]

    P_A = transition_probability(ker_AA, det_A.chi, gap_A, lam_A, **resp_kwargs)
    P_B = transition_probability(ker_BB, det_B.chi, gap_B, lam_B, **resp_kwargs)
    C, meta_C = cross_term(
        ker_AB, det_A.chi, det_B.chi, gap_A, gap_B, lam_A, lam_B,
        return_meta=True, **cross_kwargs,
    )
    M, meta_M = pair_term(
        ker_AB, det_A.chi, det_B.chi, gap_A, gap_B, lam_A, lam_B,
        return_meta=True, **pair_kwargs,
    )

    scale = float(max(P_A, P_B, abs(C), abs(M)))
    meta = {
        "backend": type(kernel).__name__,
        "coupling": coupling,
        # Second-order perturbation theory is trustworthy only while every
        # matrix element is small; carried into result rows per spec M2.2/M2.3.
        "perturbative_scale": scale,
        "perturbative_ok": bool(scale < 0.1),
        "cross": meta_C,
        "pair": meta_M,
    }
    return UDWPairState(
        P_A=P_A, P_B=P_B, C=C, M=M,
        lam_A=lam_A, lam_B=lam_B, gap_A=gap_A, gap_B=gap_B, meta=meta,
    )


# ---------------------------------------------------------------------------
# PKMM closed forms (3+1D, Gaussian switching chi = e^{-(t - t_nu)^2/T^2}
# [Eq. (22)], Gaussian smearing [Eq. (19)]; dimensionless alpha = Omega T,
# beta = d/T, gamma = Delta/T, delta = sigma/T [PKMM Table I]).  All values
# are the L / M of Eq. (13) *including* the lam^2 factor, in PKMM units
# (T = 1).
# ---------------------------------------------------------------------------

def pkmm_L_AA(alpha, delta, lam=1.0):
    """L_AA closed form, PKMM Eq. (29).

        L_AA = lam^2 e^{-alpha^2/2} / (8 pi (1 + delta^2))
               * [2 - sqrt(2 pi) (alpha/s) e^{alpha^2/(2 s^2)} erfc(alpha/(sqrt2 s))],
        s = sqrt(1 + delta^2).

    The paper's e^{alpha^2/(2(1+delta^2))} erfc(...) product is evaluated
    as the scaled complementary error function erfcx(x) = e^{x^2} erfc(x)
    with x = alpha/(sqrt2 s), which is exactly that product and is
    overflow-safe for either sign of alpha.  Analytically identical to
    ``kernels.response_gaussian_closed_form(Omega=alpha, T=1/sqrt2,
    sigma=delta)`` (pinned to 1e-12 in tests/test_udw.py).
    """
    alpha = float(alpha)
    delta = float(delta)
    if delta < 0.0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    s2 = 1.0 + delta * delta
    s = math.sqrt(s2)
    bracket = 2.0 - _SQRT_2PI * (alpha / s) * erfcx(alpha / (_SQRT2 * s))
    return float(lam) ** 2 * math.exp(-0.5 * alpha * alpha) / (8.0 * math.pi * s2) * bracket


def pkmm_L_AB(alpha, beta, gamma, delta, lam=1.0):
    """Cross term L_AB closed form, PKMM Eq. (30).

        L_AB = i lam^2 e^{-alpha^2/2} e^{-i alpha gamma} / (8 sqrt(2 pi) beta s)
               * [ e^{-u^2} erfc(i u) - e^{-v^2} erfc(-i v) ],
        u = (beta + gamma - i alpha)/(sqrt2 s),
        v = (beta - gamma + i alpha)/(sqrt2 s),   s = sqrt(1 + delta^2).

    With w(z) = e^{-z^2} erfc(-i z) (Faddeeva), e^{-u^2} erfc(i u) = w(-u)
    and e^{-v^2} erfc(-i v) = w(v), so the bracket is w(-u) - w(v).  For
    alpha >= 0 both arguments sit in the closed upper half-plane where
    |w| <= 1; for alpha < 0 the reflection w(z) = 2 e^{-z^2} - w(-z) folds
    them back up, and the paper's overall e^{-alpha^2/2} is multiplied into
    the reflected Gaussians analytically (the residual exponent is
    -alpha^2 delta^2/(2 s^2) - (beta -/+ gamma)^2/(2 s^2) <= 0), so nothing
    overflows at either sign.

    Convention: switching centers t_A = 0, t_B = gamma (PKMM Sec. II.A.1);
    L_BA = conj(L_AB).
    """
    alpha, beta, gamma, delta = map(float, (alpha, beta, gamma, delta))
    if not beta > 0.0:
        raise ValueError(f"beta must be > 0 (Eq. (30) carries a 1/beta), got {beta}")
    if delta < 0.0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    s = math.sqrt(1.0 + delta * delta)
    u = (beta + gamma - 1j * alpha) / (_SQRT2 * s)
    v = (beta - gamma + 1j * alpha) / (_SQRT2 * s)
    g = math.exp(-0.5 * alpha * alpha)
    if alpha >= 0.0:
        bracket = g * (wofz(-u) - wofz(v))
    else:  # reflect w into the upper half-plane, Gaussians folded in
        bracket = 2.0 * (
            np.exp(-0.5 * alpha * alpha - u * u) - np.exp(-0.5 * alpha * alpha - v * v)
        ) - g * (wofz(u) - wofz(-v))
    pref = 1j * float(lam) ** 2 * np.exp(-1j * alpha * gamma) / (
        8.0 * _SQRT_2PI * beta * s
    )
    return complex(pref * bracket)


def _E_damped(kappa, g, delta):
    """e^{-(1 + delta^2) kappa^2 / 2} * E(kappa, g), with E of PKMM Eq. (28).

        E(kappa, g) = e^{i g kappa} [1 - erf((g + i kappa)/sqrt2)]
                    = e^{(kappa^2 - g^2)/2} w((-kappa + i g)/sqrt2)

    via erfc(z) = e^{-z^2} w(i z).  For g < 0 the Faddeeva argument drops
    into the lower half-plane, so w(z) = 2 e^{-z^2} - w(-z) is applied
    first; both branches evaluate w only at Im >= 0 (where |w| <= 1), and
    the growing e^{kappa^2/2} cancels *analytically* against the Gaussian
    damping -- no overflow at any kappa.
    """
    kappa = np.asarray(kappa, dtype=float)
    g = float(g)
    d2 = float(delta) ** 2
    if g >= 0.0:
        return np.exp(-0.5 * d2 * kappa**2 - 0.5 * g * g) * wofz(
            (-kappa + 1j * g) / _SQRT2
        )
    return 2.0 * np.exp(1j * kappa * g - 0.5 * (1.0 + d2) * kappa**2) - np.exp(
        -0.5 * d2 * kappa**2 - 0.5 * g * g
    ) * wofz((kappa - 1j * g) / _SQRT2)


def pkmm_M_abs(alpha, beta, gamma, delta, lam=1.0, rtol=1e-10, max_doublings=12):
    """|M| via PKMM Eq. (31), with E of Eq. (28).

        |M| = lam^2 e^{-alpha^2/2} / (8 pi beta)
              * | int_0^inf dk sin(beta k) e^{-(1 + delta^2) k^2 / 2}
                  [E(k, gamma) + E(k, -gamma)] |,

    the k-integral on Gauss-Legendre panels under a doubling gate.  Note
    that alpha enters Eq. (31) only through the prefactor.

    ``delta`` must be > 0: the E-terms decay only like 1/k, so at
    delta = 0 the integral is merely *conditionally* convergent and a
    panelled cutoff rule is not a legitimate discretization of it.  For
    delta > 0 the extra e^{-delta^2 k^2/2} makes it absolutely convergent
    and the cutoff k_max = 8.5/delta + 5 puts the integrand below 1e-16 of
    its peak.  The pointlike limit has its own closed forms: Eq. (32) for
    non-overlapping switchings (:func:`pkmm_M_abs_nonoverlap`, which is
    also exact at delta = 0) and Eq. (37) for gamma = 0
    (:func:`pkmm_M_abs_coincident_pointlike`).
    """
    alpha, beta, gamma, delta = map(float, (alpha, beta, gamma, delta))
    if not beta > 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if not delta > 0.0:
        raise ValueError(
            "delta must be > 0: at delta = 0 the Eq. (31) integrand decays "
            "only like 1/k (conditional convergence).  Use Eq. (32) "
            "(pkmm_M_abs_nonoverlap) or Eq. (37) "
            "(pkmm_M_abs_coincident_pointlike) for the pointlike limit."
        )
    kmax = 8.5 / delta + 5.0
    panel0 = min(1.0 / beta, 1.0 / max(abs(gamma), 1.0), 1.0) / 8.0

    def quad(panel):
        k, w = gl_panels(0.0, kmax, panel, n_gl=16)
        f = np.sin(beta * k) * (
            _E_damped(k, gamma, delta) + _E_damped(k, -gamma, delta)
        )
        return complex(np.sum(w * f))

    val, _, _ = _doubling_gate(
        quad, panel0, rtol, 0.0, max_doublings, "pkmm_M_abs"
    )
    return (
        float(lam) ** 2
        * math.exp(-0.5 * alpha * alpha)
        / (8.0 * math.pi * beta)
        * abs(val)
    )


def pkmm_M_abs_nonoverlap(alpha, beta, gamma, delta, lam=1.0):
    """|M_non| closed form, PKMM Eq. (32).

    Valid when the two switching Gaussians do not overlap, which PKMM take
    to be |gamma| >= 7/sqrt(2) (``PKMM_NONOVERLAP_GAMMA``; the overlap is
    then suppressed by e^{-49/2} ~ 1e-11):

        |M_non| = lam^2 e^{-alpha^2/2} / (8 sqrt(2 pi) beta s)
                  * | e^{-x^2}[1 + erf(i x)] - e^{-y^2}[1 - erf(i y)] |,
        x = (beta - gamma)/(sqrt2 s), y = (beta + gamma)/(sqrt2 s).

    Both arguments are real, and e^{-x^2}[1 + erf(i x)] = w(x),
    e^{-y^2}[1 - erf(i y)] = w(-y), so the bracket is |w(x) - w(-y)|.
    Exact at delta = 0 as well (unlike the Eq. (31) quadrature).
    """
    alpha, beta, gamma, delta = map(float, (alpha, beta, gamma, delta))
    if not beta > 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if delta < 0.0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    if abs(gamma) < PKMM_NONOVERLAP_GAMMA:
        raise ValueError(
            f"Eq. (32) needs non-overlapping switchings, "
            f"|gamma| >= 7/sqrt(2) ~ {PKMM_NONOVERLAP_GAMMA:.4f}, got {gamma}"
        )
    s = math.sqrt(1.0 + delta * delta)
    x = (beta - gamma) / (_SQRT2 * s)
    y = (beta + gamma) / (_SQRT2 * s)
    pref = float(lam) ** 2 * math.exp(-0.5 * alpha * alpha) / (
        8.0 * _SQRT_2PI * beta * s
    )
    return pref * abs(wofz(x) - wofz(-y))


def pkmm_M_abs_coincident_pointlike(alpha, beta, lam=1.0):
    """|M_coinc| from PKMM Eq. (37): pointlike (delta -> 0) and gamma = 0.

        M_coinc = lam^2 e^{-(alpha^2 + beta^2)/2} / (4 sqrt(2 pi) beta)
                  * [erfi(beta/sqrt2) - i].

    erfi overflows for beta >~ 37 while the product does not, so the real
    part is evaluated as e^{-beta^2/2} erfi(beta/sqrt2)
    = (2/sqrt(pi)) D(beta/sqrt2) with D the Dawson function
    (erfi(x) = (2/sqrt(pi)) e^{x^2} D(x), and (beta/sqrt2)^2 = beta^2/2).
    """
    alpha, beta = float(alpha), float(beta)
    if not beta > 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    pref = float(lam) ** 2 * math.exp(-0.5 * alpha * alpha) / (
        4.0 * _SQRT_2PI * beta
    )
    real = 2.0 / _SQRT_PI * float(dawsn(beta / _SQRT2))
    imag = math.exp(-0.5 * beta * beta)
    return pref * math.hypot(real, imag)


def pkmm_negativity_estimator(alpha, beta, gamma, delta, lam=1.0, **m_kwargs):
    """N^(2) = |M| - L_AA for identical detectors [PKMM Eq. (68)].

    May be negative (no harvesting); the negativity itself is
    N = max(0, N^(2)) [Eq. (67)].  This is the quantity mapped in PKMM
    Fig. 2 (panels a.1/a.2, b.1/b.2 for 3+1D Gaussian switching with
    Gaussian / pointlike smearing), and its zero set is the harvesting
    "death line" the anchors in tests/test_udw.py reproduce.
    """
    return pkmm_M_abs(alpha, beta, gamma, delta, lam=lam, **m_kwargs) - pkmm_L_AA(
        alpha, delta, lam=lam
    )


def pkmm_detectors(gamma):
    """The two PKMM Gaussian-switched detectors, T = 1, t_A = 0, t_B = gamma.

    chi_nu(t) = e^{-(t - t_nu)^2} [Eq. (22) at T = 1], which in this repo's
    convention is ``switching('gaussian', 1/sqrt(2), t_nu)``.
    """
    chi_A = switching("gaussian", PKMM_SWITCHING_WIDTH_FACTOR, 0.0)
    chi_B = switching("gaussian", PKMM_SWITCHING_WIDTH_FACTOR, float(gamma))
    return UDWDetector(chi_A), UDWDetector(chi_B)


def pkmm_pair_state(alpha, beta, gamma, delta, lam=1.0, kmax=None, panel=None,
                    **quad_kwargs):
    """Run the *engine* on the PKMM 3+1 Gaussian-Gaussian setup (T = 1).

    Builds the continuum kernel (sigma = delta, distance = beta), the two
    Gaussian switchings of Eq. (22) with t_A = 0, t_B = gamma, gap
    Omega = alpha, and calls :func:`udw_pair_state`.  The ``pkmm_*`` closed
    forms above are the anchors this must reproduce (tests/test_udw.py
    pins P_A/P_B against Eq. (29), C against Eq. (30) and |M| against
    Eq. (31), all to <= 1e-10 relative).
    """
    kernel = wightman_continuum_3p1(
        float(delta), distance=float(beta), kmax=kmax, panel=panel
    )
    det_A, det_B = pkmm_detectors(gamma)
    return udw_pair_state(kernel, det_A, det_B, lam, float(alpha), **quad_kwargs)
