"""Periodic drives of the lattice field: K(t) for time-modulated vacua (Layer 5).

A Floquet drive is a periodic coupling matrix K(t) for the quadratic
Hamiltonian H(t) = (1/2)(p.p + x.K(t).x) of docs/API.md (hbar = 1, block
quadrature ordering, V_vac = I/2).  The :class:`Drive` object built by
:func:`modulated_K` is a *pure callable* ``t -> K(t)`` carrying its own
metadata (period, depth, profile, kind), so it drops unchanged into
``vacuum.protocol.GeneratorStep`` (a time-dependent generator is "a periodic
callable — nothing else changes", vacuum/protocol.py) and into the fast
monodromy path of :mod:`vacuum.floquet.monodromy`.

Modulation kinds
----------------
``'coupling'``   K(t) = K0 (1 + depth f(omega_mod t)).  Every normal mode k
                 of K0 obeys its own Mathieu-type equation
                 x'' + omega_k^2 (1 + depth f) x = 0 — the photonic-time-crystal
                 model (uniform modulation of the medium; E. Lustig, O. Segal,
                 S. Saha, C. Fruhling, V. M. Shalaev, A. Boltasseva, M. Segev,
                 Opt. Express 31, 9165 (2023), Sec. 2).
``'mass'``       K(t) = K0 + mass_sq depth f(omega_mod t) I: mass-only
                 modulation m^2(t) = m^2 (1 + depth f); per mode the relative
                 depth is depth m^2/omega_k^2.
``'pattern'``    K(t) = K0 + depth f(omega_mod t) P for a user-supplied
                 symmetric pattern P (or a length-N vector = diag pattern):
                 spatially structured modulation; modes couple.
``'boundary'``   K(t) = K0 + [1/L_eff(t) - 1/L0] e_s e_s^T with
                 L_eff(t) = L0 (1 + depth f(omega_mod t)): harmonic modulation
                 of the *effective length* of a mirror behind site s — the
                 lattice transcription of the SQUID boundary condition of
                 J. R. Johansson, G. Johansson, C. M. Wilson, F. Nori, Phys.
                 Rev. A 82, 052509 (2010), Eqs. (10) and (17)
                 (Phi(0) + L_eff dPhi/dx = 0, L_eff = (Phi_0/2pi)^2/(E_J L_0)),
                 used by Wilson et al., Nature 479, 376 (2011).  Build K0 with
                 :func:`dce_chain_K` so it already carries the static spring
                 1/L0 at the boundary site.

Profiles
--------
``'cos'`` (default), ``'sin'``, ``'triangle'``, ``'square'`` (piecewise
constant, +1 on the first half period, -1 on the second), a custom
callable ``f(theta)`` of the phase theta = omega_mod t, or an explicit
piecewise-constant tuple ``((value, fraction), ...)`` whose fractions sum
to 1.  Piecewise-constant profiles carry ``Drive.pieces`` so the monodromy
can be assembled from *exact* static exponentials (no discretisation error).

Everything is pure-functional over arrays: no hidden state, no input
mutation (Layer 6 JAX-mirror pre-positioning).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "PROFILES",
    "KINDS",
    "Drive",
    "modulated_K",
    "dce_chain_K",
    "profile_function",
]

_TWO_PI = 2.0 * np.pi


def _cos(theta):
    return np.cos(theta)


def _sin(theta):
    return np.sin(theta)


def _square(theta):
    th = np.mod(theta, _TWO_PI)
    return np.where(th < np.pi, 1.0, -1.0)


def _triangle(theta):
    th = np.mod(theta, _TWO_PI) / _TWO_PI
    return 1.0 - 4.0 * np.abs(th - 0.5)


PROFILES = {
    "cos": _cos,
    "sin": _sin,
    "square": _square,
    "triangle": _triangle,
}

#: piecewise-constant structure of the named profiles: ((value, fraction), ...)
_PIECES = {
    "square": ((1.0, 0.5), (-1.0, 0.5)),
}

KINDS = ("coupling", "mass", "pattern", "boundary")


def profile_function(profile):
    """Resolve a profile spec to ``(name, f(theta), pieces)``.

    ``pieces`` is the piecewise-constant description ``((value, fraction),
    ...)`` when the profile is piecewise constant, else None.
    """
    if isinstance(profile, str):
        if profile not in PROFILES:
            raise ValueError(
                f"unknown profile {profile!r}; choose from {tuple(PROFILES)} "
                "or pass a callable f(theta) / a ((value, fraction), ...) tuple"
            )
        return profile, PROFILES[profile], _PIECES.get(profile)
    if callable(profile):
        return "custom", profile, None
    # explicit piecewise-constant profile
    pieces = tuple((float(v), float(w)) for v, w in profile)
    if not pieces:
        raise ValueError("piecewise profile needs at least one (value, fraction)")
    if any(w <= 0.0 for _, w in pieces):
        raise ValueError("piece fractions must be positive")
    if abs(sum(w for _, w in pieces) - 1.0) > 1e-12:
        raise ValueError("piece fractions must sum to 1")
    edges = np.cumsum([w for _, w in pieces])
    values = np.array([v for v, _ in pieces])

    def f(theta, _edges=edges, _values=values):
        th = np.mod(np.asarray(theta, dtype=float), _TWO_PI) / _TWO_PI
        idx = np.minimum(np.searchsorted(_edges, th, side="right"), len(_values) - 1)
        return _values[idx]

    return "pieces", f, pieces


@dataclass(frozen=True)
class Drive:
    """A periodic coupling-matrix drive K(t) (see module docstring).

    Callable: ``drive(t) -> K(t)`` (N, N).  ``drive.period`` is 2 pi /
    omega_mod; ``drive.diagonal`` says whether K(t) shares the eigenvectors
    of K0 for all t (kinds 'coupling' and 'mass'), in which case
    :meth:`mode_freq_sq` gives the per-normal-mode frequency-squared
    omega_k^2(t) and the monodromy factorises into 2 x 2 blocks.
    """

    K0: np.ndarray
    depth: float
    omega_mod: float
    profile: str
    kind: str
    f: Callable = field(repr=False, compare=False)
    mass_sq: Optional[float] = None
    pattern: Optional[np.ndarray] = field(default=None, repr=False)
    L0: Optional[float] = None
    site: Optional[int] = None
    pieces: Optional[Tuple[Tuple[float, float], ...]] = None

    # -- metadata ---------------------------------------------------------

    @property
    def N(self) -> int:
        return int(self.K0.shape[0])

    @property
    def period(self) -> float:
        return _TWO_PI / self.omega_mod

    @property
    def diagonal(self) -> bool:
        """True iff K(t) is diagonal in the normal modes of K0 for all t."""
        return self.kind in ("coupling", "mass")

    @property
    def piecewise_constant(self) -> bool:
        return self.pieces is not None

    # -- the drive --------------------------------------------------------

    def profile_value(self, t):
        """f(omega_mod t)."""
        return self.f(self.omega_mod * np.asarray(t, dtype=float))

    def boundary_spring(self, t):
        """1/L_eff(t) for kind 'boundary' (the time-dependent boundary spring)."""
        if self.kind != "boundary":
            raise ValueError("boundary_spring is defined for kind='boundary' only")
        return 1.0 / (self.L0 * (1.0 + self.depth * self.profile_value(t)))

    def __call__(self, t):
        """K(t), a fresh (N, N) array."""
        fval = float(self.profile_value(float(t)))
        if self.kind == "coupling":
            return self.K0 * (1.0 + self.depth * fval)
        if self.kind == "mass":
            return self.K0 + (self.mass_sq * self.depth * fval) * np.eye(self.N)
        if self.kind == "pattern":
            return self.K0 + (self.depth * fval) * self.pattern
        # boundary
        K = np.array(self.K0, dtype=float, copy=True)
        K[self.site, self.site] += (
            1.0 / (self.L0 * (1.0 + self.depth * fval)) - 1.0 / self.L0
        )
        return K

    def mode_freq_sq(self, omega_k_sq, t):
        """Per-normal-mode omega_k^2(t) for diagonal kinds (vectorised).

        ``omega_k_sq`` and ``t`` broadcast against each other.
        """
        if not self.diagonal:
            raise ValueError(
                f"kind {self.kind!r} does not preserve the normal modes of K0; "
                "use the full monodromy"
            )
        w2 = np.asarray(omega_k_sq, dtype=float)
        fval = self.profile_value(t)
        if self.kind == "coupling":
            return w2 * (1.0 + self.depth * fval)
        return w2 + self.mass_sq * self.depth * fval

    def with_depth(self, depth):
        """A copy with a different depth (pure)."""
        return modulated_K(
            self.K0,
            depth,
            self.omega_mod,
            profile=self.pieces if self.profile == "pieces" else (
                self.f if self.profile == "custom" else self.profile
            ),
            kind=self.kind,
            mass_sq=self.mass_sq,
            pattern=self.pattern,
            L0=self.L0,
            site=self.site,
        )

    def with_omega(self, omega_mod):
        """A copy with a different modulation frequency (pure)."""
        return modulated_K(
            self.K0,
            self.depth,
            omega_mod,
            profile=self.pieces if self.profile == "pieces" else (
                self.f if self.profile == "custom" else self.profile
            ),
            kind=self.kind,
            mass_sq=self.mass_sq,
            pattern=self.pattern,
            L0=self.L0,
            site=self.site,
        )


def modulated_K(
    K0,
    depth,
    omega_mod,
    profile="cos",
    *,
    kind="coupling",
    mass_sq=None,
    pattern=None,
    L0=None,
    site=0,
):
    """Build a periodic drive K(t) (README API: ``modulated_K``).

    Parameters
    ----------
    K0 : (N, N) ndarray, symmetric
        Undriven coupling matrix (the drive averages to K0 for zero-mean
        profiles; for kind 'boundary' K0 must already contain the static
        spring 1/L0 at `site`, see :func:`dce_chain_K`).
    depth : float >= 0
        Modulation depth (relative for 'coupling', 'mass' and 'boundary';
        absolute prefactor of `pattern` for 'pattern').
    omega_mod : float > 0
        Modulation angular frequency; period 2 pi / omega_mod.
    profile : str, callable, or ((value, fraction), ...)
        See module docstring.
    kind : {'coupling', 'mass', 'pattern', 'boundary'}
    mass_sq : float, required for kind 'mass'
        The m^2 whose relative modulation is `depth`.
    pattern : (N, N) symmetric or (N,) array, required for kind 'pattern'
    L0 : float > 0, required for kind 'boundary'
        Static effective length (lattice units) of the mirror behind `site`.
    site : int
        Boundary site for kind 'boundary' (default 0).

    Returns
    -------
    Drive
    """
    K0 = np.array(K0, dtype=float, copy=True)
    if K0.ndim != 2 or K0.shape[0] != K0.shape[1]:
        raise ValueError(f"K0 must be square, got shape {K0.shape}")
    K0 = 0.5 * (K0 + K0.T)
    N = K0.shape[0]
    depth = float(depth)
    if depth < 0.0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    omega_mod = float(omega_mod)
    if not omega_mod > 0.0:
        raise ValueError(f"omega_mod must be > 0, got {omega_mod}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    name, f, pieces = profile_function(profile)

    mass_sq_v = None
    pattern_v = None
    L0_v = None
    site_v = None
    if kind == "mass":
        if mass_sq is None:
            raise ValueError("kind='mass' needs mass_sq (the m^2 being modulated)")
        mass_sq_v = float(mass_sq)
        if mass_sq_v <= 0.0:
            raise ValueError(f"mass_sq must be > 0, got {mass_sq_v}")
    elif kind == "pattern":
        if pattern is None:
            raise ValueError("kind='pattern' needs a pattern P")
        P = np.asarray(pattern, dtype=float)
        if P.shape == (N,):
            P = np.diag(P)
        if P.shape != (N, N):
            raise ValueError(f"pattern shape {P.shape} incompatible with N={N}")
        pattern_v = 0.5 * (P + P.T)
    elif kind == "boundary":
        if L0 is None or not float(L0) > 0.0:
            raise ValueError("kind='boundary' needs an effective length L0 > 0")
        if depth >= 1.0:
            raise ValueError(
                f"kind='boundary' needs depth < 1 (L_eff(t) > 0), got {depth}"
            )
        L0_v = float(L0)
        site_v = int(site)
        if not 0 <= site_v < N:
            raise ValueError(f"site {site_v} outside 0..{N - 1}")
    return Drive(
        K0=K0,
        depth=depth,
        omega_mod=omega_mod,
        profile=name,
        kind=kind,
        f=f,
        mass_sq=mass_sq_v,
        pattern=pattern_v,
        L0=L0_v,
        site=site_v,
        pieces=pieces,
    )


def dce_chain_K(N, L_eff, m=0.0):
    """Coupling matrix of a half-open chain terminated by a mirror at -L_eff.

    Sites 0..N-1 with unit spacing and unit signal speed.  The far end
    (site N-1) is Dirichlet (phi_N = 0); the near end (site 0) is an open
    end with a spring 1/L_eff to ground, K_00 = 1 + 1/L_eff.  In the static
    limit the force balance at site 0, phi_0 / L_eff = phi_1 - phi_0, is the
    lattice form of the mixed boundary condition Phi(0) + L_eff dPhi/dx = 0
    of Johansson et al., PRA 82, 052509 (2010), Eq. (10) with the SQUID
    capacitance dropped (their Eq. (16)-(17) regime): the field extrapolates
    to zero at x = -L_eff, an effective mirror behind the boundary.  The
    exact lattice reflection phase is tan(k L) = sin k / (1/L_eff - 1 + cos k),
    i.e. L -> L_eff as k -> 0.

    A mass term m^2 (default 0) may be added as an IR regulator; K is
    positive definite for any L_eff > 0 even at m = 0 thanks to the far
    Dirichlet wall.
    """
    N = int(N)
    if N < 2:
        raise ValueError("need N >= 2")
    L_eff = float(L_eff)
    if not L_eff > 0.0:
        raise ValueError(f"L_eff must be > 0, got {L_eff}")
    K = np.zeros((N, N))
    np.fill_diagonal(K, 2.0 + float(m) ** 2)
    idx = np.arange(N - 1)
    K[idx, idx + 1] = -1.0
    K[idx + 1, idx] = -1.0
    K[0, 0] = 1.0 + 1.0 / L_eff + float(m) ** 2  # open end + boundary spring
    return K
