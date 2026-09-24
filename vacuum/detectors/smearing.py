"""Lattice spatial smearing profiles (Layer 2, milestone M2.1).

``smearing(kind, center, width, N)`` returns a normalized lattice profile
F (shape (N,), sum F = 1 — the spec's normalization) plus analytic
derivative handles with respect to the continuous parameters (center,
width): the differentiation-ready data the QEI f-family optimizer (M3.4)
and the Layer 6 JAX mirror consume.

Kinds
-----
- ``'gaussian'``: g_i = exp(-(i - center)^2 / (2 width^2))
- ``'cos2'``:     g_i = cos^2(pi (i - center) / (2 width)) for
  |i - center| <= width, else 0 (compact support, C^1 in the parameters)
- ``'point'``:    one-hot at round(center) — the pointlike lattice detector
  (a *spatial* delta is legitimate on the lattice, unlike sharp *time*
  switching, which is banned).  ``width`` is ignored; the parameter
  derivatives are identically zero (the profile is piecewise constant in
  center), documented rather than differentiable.

All kinds are normalized F = g / sum(g); the derivative of the normalized
profile follows by the quotient rule
dF = (dg - F * sum(dg)) / sum(g), applied analytically.

Boundary conditions: ``bc='open'`` (default) measures displacement i - center
directly; ``bc='periodic'`` uses the minimal image on the N-ring (derivative
formulas hold except exactly at the wrap discontinuity |delta| = N/2).

Pure functions of their arguments: a :class:`Smearing` is a frozen record of
read-only arrays; nothing is mutated after construction.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Smearing", "smearing"]

_KINDS = ("gaussian", "cos2", "point")


def _readonly(a):
    out = np.array(a, dtype=float, copy=True)
    out.flags.writeable = False
    return out


class Smearing:
    """Frozen normalized lattice smearing profile; see module docstring.

    Attributes
    ----------
    F : (N,) ndarray, read-only, sum(F) = 1
    dF_dcenter, dF_dwidth : (N,) ndarrays, read-only
        Analytic derivatives of the *normalized* profile with respect to
        center and width (zeros for kind='point').
    kind, center, width, N, bc : the construction parameters.

    ``np.asarray(smearing_obj)`` yields F, so profiles drop directly into
    F^T W(dt) F' sandwiches and ``LatticeWightman.smeared``.
    """

    __slots__ = ("kind", "center", "width", "N", "bc", "F", "dF_dcenter", "dF_dwidth")

    def __init__(self, kind, center, width, N, bc="open"):
        if kind not in _KINDS:
            raise ValueError(f"unknown smearing kind {kind!r}: choose from {_KINDS}")
        if bc not in ("open", "periodic"):
            raise ValueError(f"bc must be 'open' or 'periodic', got {bc!r}")
        N = int(N)
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        center = float(center)
        width = float(width)

        idx = np.arange(N, dtype=float)
        delta = idx - center
        if bc == "periodic":
            delta = (delta + 0.5 * N) % N - 0.5 * N  # minimal image on the ring

        if kind == "point":
            j = int(round(center))
            if bc == "periodic":
                j %= N
            if not 0 <= j < N:
                raise ValueError(
                    f"point smearing center {center} rounds to site {j}, "
                    f"outside the lattice 0..{N - 1}"
                )
            F = np.zeros(N)
            F[j] = 1.0
            dg_dc = np.zeros(N)
            dg_dw = np.zeros(N)
            g, s = F, 1.0
        else:
            if not width > 0.0:
                raise ValueError(f"width must be > 0 for kind {kind!r}, got {width}")
            if kind == "gaussian":
                g = np.exp(-0.5 * (delta / width) ** 2)
                dg_dc = (delta / width**2) * g          # d/dc: d(delta)/dc = -1
                dg_dw = (delta**2 / width**3) * g
            else:  # cos2, compact support |delta| <= width, C^1 at the edges
                inside = np.abs(delta) <= width
                g = np.where(inside, np.cos(0.5 * np.pi * delta / width) ** 2, 0.0)
                sin_term = np.where(inside, np.sin(np.pi * delta / width), 0.0)
                dg_dc = (0.5 * np.pi / width) * sin_term
                dg_dw = (0.5 * np.pi / width**2) * delta * sin_term
            s = float(np.sum(g))
            if s <= 0.0:
                raise ValueError(
                    f"smearing profile has zero weight on the lattice "
                    f"(kind={kind!r}, center={center}, width={width}, N={N})"
                )
            F = g / s

        # Quotient rule on F = g / sum(g), applied to the analytic dg's.
        dF_dc = (dg_dc - F * np.sum(dg_dc)) / s
        dF_dw = (dg_dw - F * np.sum(dg_dw)) / s

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "N", N)
        object.__setattr__(self, "bc", bc)
        object.__setattr__(self, "F", _readonly(F))
        object.__setattr__(self, "dF_dcenter", _readonly(dF_dc))
        object.__setattr__(self, "dF_dwidth", _readonly(dF_dw))

    def __setattr__(self, name, value):  # frozen
        raise AttributeError("Smearing objects are immutable")

    def __array__(self, dtype=None, copy=None):
        if dtype is None:
            return self.F if not copy else self.F.copy()
        return self.F.astype(dtype)

    def __len__(self):
        return self.N

    def __repr__(self):
        return (
            f"Smearing(kind={self.kind!r}, center={self.center}, "
            f"width={self.width}, N={self.N}, bc={self.bc!r})"
        )


def smearing(kind, center, width, N, bc="open"):
    """Build a normalized lattice smearing profile (sum F = 1).

    Returns a :class:`Smearing`: ``.F`` is the plain profile array,
    ``.dF_dcenter`` / ``.dF_dwidth`` are the analytic-derivative handles.
    """
    return Smearing(kind, center, width, N, bc=bc)
