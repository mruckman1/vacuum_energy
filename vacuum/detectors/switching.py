"""Switching functions for detector protocols (Layer 2, milestone M2.1).

``switching(kind, T, t0)`` builds smooth switching profiles chi(t) with an
analytic-derivative handle.  Two kinds (spec M2.1):

- ``'gaussian'``: chi(t) = exp(-(t - t0)^2 / (2 T^2)), peak 1 at t0, width
  parameter T.  Infinite tails; ``support`` reports the effective support
  (t0 -/+ 8.5 T), where chi ~ 2e-16 — below double-precision resolution of
  the peak — which is the quadrature window used by
  ``vacuum.detectors.kernels.detector_response``.
- ``'cos2'``: chi(t) = cos^2(pi (t - t0) / (2 T)) for |t - t0| <= T, else 0.
  Compactly supported on [t0 - T, t0 + T] and C^1 (chi and chi' both vanish
  at the edges) — the profile for strict spacelike-separation arguments
  (M2.5's communication split needs exactly-compact supports).

Sharp (top-hat) switching is deliberately NOT offered: the spec forbids it
(its UV behavior invalidates every convergence guarantee here); the delta
limit exists only inside the Simidzija closed form (M2.5).

Pure functions of their arguments, no hidden state: a :class:`Switching`
is a frozen (kind, width, t0) triple whose ``__call__`` and ``derivative``
return plain ndarrays — the analytic-derivative handle required by the QEI
module (M3.4) and the Layer 6 JAX mirror.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Switching", "switching", "GAUSSIAN_SUPPORT_SIGMAS"]

# Effective support half-width of the Gaussian profile, in units of T.
# exp(-8.5^2/2) ~ 2.1e-16: tail truncation is below double roundoff of the
# unit peak, so quadrature windows built from .support incur ~1e-16
# relative error — far inside every 1e-8 anchor tolerance.
GAUSSIAN_SUPPORT_SIGMAS = 8.5

_KINDS = ("gaussian", "cos2")


class Switching:
    """Frozen switching profile; see module docstring.

    Attributes
    ----------
    kind : str            'gaussian' or 'cos2'
    width : float         the T parameter
    t0 : float            center of the switching window
    support : (float, float)
        Exact support for 'cos2'; effective (8.5 T) support for 'gaussian'.
    """

    __slots__ = ("kind", "width", "t0", "support")

    def __init__(self, kind, T, t0=0.0):
        if kind not in _KINDS:
            raise ValueError(
                f"unknown switching kind {kind!r}: choose from {_KINDS}. "
                "Sharp/top-hat switching is deliberately not offered (spec "
                "M2.1); the delta-switching limit lives only inside the "
                "Simidzija closed form (M2.5)."
            )
        T = float(T)
        if not T > 0.0:
            raise ValueError(f"switching width T must be > 0, got {T}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "width", T)
        object.__setattr__(self, "t0", float(t0))
        if kind == "gaussian":
            half = GAUSSIAN_SUPPORT_SIGMAS * T
        else:  # cos2: exact compact support
            half = T
        object.__setattr__(self, "support", (self.t0 - half, self.t0 + half))

    def __setattr__(self, name, value):  # frozen
        raise AttributeError("Switching objects are immutable")

    def __call__(self, t):
        """chi(t); scalar in -> float out, ndarray in -> ndarray out."""
        u = np.asarray(t, dtype=float) - self.t0
        if self.kind == "gaussian":
            out = np.exp(-0.5 * (u / self.width) ** 2)
        else:
            inside = np.abs(u) <= self.width
            out = np.where(
                inside, np.cos(0.5 * np.pi * u / self.width) ** 2, 0.0
            )
        return float(out) if out.ndim == 0 else out

    def derivative(self, t):
        """Analytic dchi/dt; same shape convention as __call__.

        gaussian: chi'(t) = -((t - t0)/T^2) chi(t)
        cos2:     chi'(t) = -(pi / 2 T) sin(pi (t - t0) / T) inside, 0 outside
                  (continuous everywhere: sin(+-pi) = 0 at the edges).
        """
        u = np.asarray(t, dtype=float) - self.t0
        if self.kind == "gaussian":
            out = -(u / self.width**2) * np.exp(-0.5 * (u / self.width) ** 2)
        else:
            inside = np.abs(u) <= self.width
            out = np.where(
                inside,
                -(0.5 * np.pi / self.width) * np.sin(np.pi * u / self.width),
                0.0,
            )
        return float(out) if out.ndim == 0 else out

    def __repr__(self):
        return f"Switching(kind={self.kind!r}, T={self.width}, t0={self.t0})"


def switching(kind, T, t0=0.0):
    """Build a switching profile: ``switching('gaussian'|'cos2', T, t0)``.

    Returns a :class:`Switching` — callable chi(t) returning plain arrays,
    with the analytic derivative available as ``.derivative(t)``.
    """
    return Switching(kind, T, t0)
