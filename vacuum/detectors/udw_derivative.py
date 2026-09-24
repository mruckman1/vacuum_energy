"""Derivative-coupled (d_t phi) UDW detectors: the perturbative side of 'xp'.

Source of the claim this module exists to reproduce
-----------------------------------------------------
PLAN.md (Layer 2 discovery targets) attributes the "derivative-coupling
result" — that even causally connected detectors draw their entanglement
primarily from field correlations rather than from communication — to the
Waterloo 2025/26 circuit-QED work.  Tracing it (2026-09-01):

* The result is **established** in

      A. Teixido-Bonfill and E. Martin-Martinez, "Derivative coupling
      enables genuine entanglement harvesting in causal communication",
      Phys. Rev. D 110, 105016 (2024), arXiv:2406.14637   [TB-MM24],

  whose abstract states that "particle detectors coupled to a massless
  quantum field through its derivative can genuinely harvest entanglement
  from the field even when they are in causal contact in flat spacetime"
  and that "the harvested entanglement peaks at full light contact"; the
  conclusion (Sec. V) states that at full light contact "the source of that
  entanglement is genuine harvesting and not communication".  The model is
  TB-MM24 Eq. (2),

      H^nu(t) = lam_nu chi_nu(t) mu_nu(t) int d^n x F_nu(x) d_t phi(t, x),

  i.e. the detector monopole couples to the **time derivative** d_t phi
  (n+1 dimensions; results shown for 1+1 and 3+1).  The second-order state
  is Eq. (8) with pair term M [Eq. (10)] and negativity
  N = max(|M| - L, 0) [Eq. (12)]; the pair term splits as M = M^+ + M^-
  [Eq. (16)] into the anti-commutator (field-correlation, "genuine
  harvesting") part M^+ and the commutator (communication) part M^-, the
  Tjoa-Martin-Martinez split this repo implements in
  ``vacuum.detectors.communication``.  Switching and smearing are Gaussian,
  chi(t) = e^{-t^2/T^2}, F(x) = e^{-x^2/sigma^2}/(sqrt(pi) sigma)^n
  [Eq. (24)]; Fig. 2 uses Omega T = 4 and |x_Delta|/T = 5 with the light
  cone marked.  In 1+1 the commutator part |M^-| cancels exactly at the
  light cone [Eq. (27)] while |M| peaks there.

* The result is **restated** for the circuit-QED proposal in

      A. Teixido-Bonfill, X. Dai, A. Lupascu, E. Martin-Martinez, "Towards
      an experimental implementation of entanglement harvesting in
      superconducting circuits: effect of detector gap variation on
      entanglement harvesting", arXiv:2505.01516 (Phys. Rev. A, 2026)
      [TB-DLM26],

  abstract: "Notably, our analysis shows that (due to the derivative
  coupling nature of the model) even for causally connected detectors, the
  entanglement primarily originates from the field's correlations."  Its
  interaction, Eq. (31), H_int(t) = -(phi_0/l_0) gamma chi(t) mu(t)
  d_x Phi_C(t, x_d), couples the flux qubit to the **spatial** derivative
  of the 1+1 transmission-line flux field (inductive coupling, derived from
  the lumped circuit in Eqs. (19)-(22)); the split is Eq. (63),
  M = M^+ + M^-, with Eq. (64) the harvested fraction.

Both are "derivative couplings"; on a 1+1 massless field the two
derivatives share what matters for the claim (no coupling to the k = 0
mode, a commutator supported on the light-cone edges rather than filling
the cone).  This module and ``attach_detectors(..., coupling='xp')``
implement the **d_t phi model of TB-MM24 Eq. (2)** — the one the plan's
'xp' flag names: on the lattice p_i = d_t phi_i for the free field, so the
detector position couples to the field momentum.  The d_x Phi model of
TB-DLM26 Eq. (31) needs no new machinery on the lattice: it is the
amplitude ('xx') coupling with the finite-difference stencil profile
:func:`spatial_derivative_profile` (F = e_{j+1} - e_j), and is offered here
for completeness and used as a third column in tests/test_xp_coupling.py.

Provenance note: the equation numbers above were read off the arXiv HTML
renderings (arxiv.org/html/2406.14637, arxiv.org/html/2505.01516) on
2026-09-01, not off the LaTeX sources; the sentences quoted are verbatim
from the abstracts/conclusion as rendered there.

The transcription
-----------------
With H_I(t) = lam chi(t) mu(t) Pi(t), Pi = int F d_t phi = F^T p for the
free lattice field, every second-order matrix element of the PKMM state
[``vacuum.detectors.udw``, PKMM Eqs. (13)-(18)] is obtained by replacing
the pulled-back Wightman kernel W(t - t') = <0|Phi(t) Phi(t')|0> by

    W^{pi pi}(t - t') = <0| Pi(t) Pi(t') |0> = d_t d_t' W(t - t') = -W''(dt).

For an exponential sum W(dt) = sum_k c_k e^{-i w_k dt} this is
sum_k c_k w_k^2 e^{-i w_k dt}: on the lattice
U diag(omega e^{-i omega dt}/2) U^T sandwiched by the profiles
(``LatticeWightman.momentum``), in the continuum 3+1 backend an extra k^2
in the radial integrand — :class:`vacuum.detectors.kernels.DerivativeKernel`.
The equal-time real part is <p_i p_j> = (K^{1/2}/2)_ij, the p-p block of
the ground-state covariance (tested).

Nothing else changes: the same P_nu, C, M quadratures, the same negativity
Eqs. (66)-(67), the same commutator/anti-commutator split, the same
oscillator <-> two-level map lam^UDW = lam^osc/sqrt(2 Omega) (Gate A,
tests/test_gate_crossvalidation.py, step 1 — the detector side is
untouched by which field operator it couples to).

Single-detector exact mode sum (the 1e-12 cross-check)
------------------------------------------------------
P/lam^2 = int dt dt' chi(t) chi(t') e^{-i Omega (t - t')} W(t - t') with
W = sum_k c_k e^{-i w_k dt} factorizes exactly as

    P/lam^2 = sum_k c_k |chihat(Omega + w_k)|^2,
    chihat(u) = int chi(t) e^{-i u t} dt,

so with the *analytic* Fourier transform of the switching function
(:func:`switching_fourier`: sqrt(2 pi) T e^{-T^2 u^2/2} for the Gaussian,
T sinc(uT/pi) + (T/2)[sinc(uT/pi - 1) + sinc(uT/pi + 1)] for cos^2, both
elementary) the response is a finite sum with no quadrature at all —
:func:`response_mode_sum`.  It applies verbatim to the derivative kernel
(c_k -> c_k w_k^2) and is the independent check the doubling-gated
``detector_response`` is pinned against.

Everything here is pure-functional over frozen arrays (docs/API.md
conventions: hbar = 1, V_vac = I/2, block ordering, nats).
"""

from __future__ import annotations

import math

import numpy as np

from .communication import communication_split, pair_state_with_split
from .kernels import derivative_kernel
from .udw import UDWDetector, pair_kernels, udw_pair_state

__all__ = [
    "derivative_pair_kernels",
    "udw_pair_state_derivative",
    "communication_split_derivative",
    "pair_state_with_split_derivative",
    "switching_fourier",
    "response_mode_sum",
    "spatial_derivative_profile",
]


def derivative_pair_kernels(kernel, det_A, det_B):
    """(W^{pipi}_AA, W^{pipi}_BB, W^{pipi}_AB): the d_t d_t' W kernels of a pair.

    ``pair_kernels(kernel, det_A, det_B, coupling='xp')``; see
    :func:`vacuum.detectors.udw.pair_kernels` for the accepted kernel forms.
    """
    return pair_kernels(kernel, det_A, det_B, coupling="xp")


def udw_pair_state_derivative(kernel, det_A, det_B, lam, gap, **quad_kwargs):
    """Second-order pair state of two derivative-coupled (d_t phi) detectors.

    TB-MM24 Eq. (8) with the matrix elements of PKMM Eqs. (15)-(18) built on
    W^{pipi} = d_t d_t' W; equals
    ``udw_pair_state(kernel, det_A, det_B, lam, gap, coupling='xp', ...)``.
    """
    return udw_pair_state(kernel, det_A, det_B, lam, gap, coupling="xp", **quad_kwargs)


def communication_split_derivative(kernel, det_A, det_B, lam=1.0, gap=0.0, **kwargs):
    """M = M^+ + M^- for the derivative coupling [TB-MM24 Eq. (16)].

    ``communication_split(..., coupling='xp')``: the commutator half of
    d_t d_t' W is d_t d_t' of the Pauli-Jordan function — state-independent
    and causal, exactly as for the amplitude coupling.
    """
    return communication_split(kernel, det_A, det_B, lam, gap, coupling="xp", **kwargs)


def pair_state_with_split_derivative(kernel, det_A, det_B, lam, gap, **quad_kwargs):
    """Derivative-coupled pair state with the M2.5 split columns filled."""
    return pair_state_with_split(
        kernel, det_A, det_B, lam, gap, coupling="xp", **quad_kwargs
    )


def switching_fourier(chi, u):
    """Analytic Fourier transform chihat(u) = int chi(t) e^{-i u t} dt.

    For the two M2.1 switching kinds (``vacuum.detectors.switching``):

    - gaussian, chi = e^{-(t - t0)^2/(2 T^2)}:
          chihat(u) = sqrt(2 pi) T e^{-T^2 u^2 / 2} e^{-i u t0};
    - cos2, chi = cos^2(pi (t - t0)/(2 T)) on |t - t0| <= T
      = (1 + cos(pi (t - t0)/T))/2:
          chihat(u) = e^{-i u t0} [ T sinc(uT/pi)
                                    + (T/2)(sinc(uT/pi - 1) + sinc(uT/pi + 1)) ],
      with numpy's sinc(x) = sin(pi x)/(pi x), i.e. the three terms are
      sin(uT)/u, sin((u - pi/T) T)/(u - pi/T) and sin((u + pi/T) T)/(u + pi/T)
      (each an elementary integral of e^{-i u t} times 1/2, cos/4, cos/4 over
      [-T, T]); the removable poles at u = 0, +-pi/T are handled by sinc.

    Elementary; both forms are verified against direct quadrature in
    tests/test_xp_coupling.py.  ``u`` may be a scalar or an array.
    """
    u = np.asarray(u, dtype=float)
    T = float(chi.width)
    t0 = float(chi.t0)
    phase = np.exp(-1j * u * t0)
    if chi.kind == "gaussian":
        mag = math.sqrt(2.0 * math.pi) * T * np.exp(-0.5 * (T * u) ** 2)
    elif chi.kind == "cos2":
        x = u * T / math.pi
        mag = T * np.sinc(x) + 0.5 * T * (np.sinc(x - 1.0) + np.sinc(x + 1.0))
    else:  # pragma: no cover - Switching only builds the two kinds
        raise ValueError(f"no analytic transform for switching kind {chi.kind!r}")
    out = phase * mag
    return complex(out) if out.ndim == 0 else out


def response_mode_sum(kernel, chi, Omega, lam=1.0):
    """Exact single-detector response P = lam^2 sum_k c_k |chihat(Omega + w_k)|^2.

    No quadrature: the double time integral of the response factorizes
    per mode of the exponential-sum kernel and the switching transform is
    analytic (:func:`switching_fourier`).  Works for any
    :class:`~vacuum.detectors.kernels.ExpSumKernel` — amplitude kernels
    and their :func:`~vacuum.detectors.kernels.derivative_kernel` alike —
    and is the independent reference ``detector_response`` is cross-checked
    against for the derivative coupling (spec: 1e-12).

    For the continuum backend the mode sum is the kernel's own k-quadrature,
    so the result is exact *for that discretization* (the k-refinement gate
    of ``detector_response`` is what certifies the continuum value).
    """
    freqs = np.asarray(kernel.freqs, dtype=float)
    amps = np.asarray(kernel.amps, dtype=complex)
    ft = switching_fourier(chi, float(Omega) + freqs)
    val = np.sum(amps * (ft * np.conj(ft)))
    return float(lam) ** 2 * float(np.real(val))


def spatial_derivative_profile(N, site, bc="open"):
    """Lattice stencil for a d_x Phi coupling: F = e_{site+1} - e_{site}.

    The TB-DLM26 (arXiv:2505.01516) Eq. (31) interaction couples the flux
    qubit to the *spatial* derivative of the transmission-line field; on a
    unit-spacing lattice d_x phi at the bond (site, site + 1) is
    phi_{site+1} - phi_site, an *amplitude* ('xx') coupling with this
    (unnormalized, sum F = 0) profile — it needs none of the 'xp'
    machinery.  With ``bc='periodic'`` the bond wraps at the last site.
    """
    N = int(N)
    j = int(site)
    if bc not in ("open", "periodic"):
        raise ValueError(f"bc must be 'open' or 'periodic', got {bc!r}")
    if bc == "open" and not 0 <= j < N - 1:
        raise ValueError(f"bond ({j}, {j + 1}) is outside the open lattice 0..{N - 1}")
    if bc == "periodic" and not 0 <= j < N:
        raise ValueError(f"site {j} outside the lattice 0..{N - 1}")
    F = np.zeros(N)
    F[j] = -1.0
    F[(j + 1) % N] = 1.0
    return F


def derivative_detector(chi, F=None):
    """Alias of :class:`~vacuum.detectors.udw.UDWDetector` (same object).

    A derivative-coupled detector is specified by exactly the same data
    (switching, and on the lattice its smearing); the coupling is a
    property of the *pair state*, chosen at ``udw_pair_state(...,
    coupling='xp')`` time.  Kept so call sites can say what they mean.
    """
    return UDWDetector(chi, F)


__all__.append("derivative_detector")
