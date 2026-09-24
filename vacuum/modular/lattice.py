"""Closed-form free-fermion correlation matrices, in float64 or mpmath.

Why this module exists
----------------------
The single-particle entanglement Hamiltonian h = ln((1 - C_A)/C_A) is
dominated, in real space, by the entanglement modes whose occupations zeta
are exponentially close to 0 or 1: for an interval of L sites in the critical
chain the largest entanglement energy is eps_max ~ 1.7627 L (Eisler-Peschel
2017, Eq. (54)), i.e. 1 - zeta ~ e^{-1.76 L}, which is below the float64
floor already for L >~ 20 (ibid., Sec. 6: "this can be done with normal
numerics for up to N = 20 sites, larger systems require special routines").
Beyond that, the matrix elements of h in the interior of the interval are
wrong at O(1) in float64 (measured in tests/test_modular_lab.py: 26 % at the
centre for L = 40, while the edge bonds are still right to 1e-4 because the
high-lying modes are localized in the middle). The remedy is Eisler-Peschel's:
build C_A *exactly* from a closed form and diagonalize it in arbitrary
precision. This module provides the closed forms; the mp diagonalization lives
in vacuum.modular.fermionic.

Closed forms (site labels are integers; C_mn = <c_m^dag c_n>)
-------------------------------------------------------------
infinite_chain_C   Eisler-Peschel 2017 Eq. (3) (= Eisler-Tonni-Peschel 2022
                   Eq. (17)): C_mn = sin(k_F (m-n))/(pi (m-n)), C_mm = k_F/pi,
                   for H = -(1/2) sum (c_m^dag c_{m+1} + h.c.) on the infinite
                   chain (hopping t = 1/2, v_F = sin k_F; half filling k_F = pi/2).
ring_C             Half-filled antiperiodic ring of N = 0 mod 4 sites (the
                   degeneracy-free choice of vacuum.fermions.tight_binding_C),
                   occupied momenta k = +-(2n+1)pi/N, n = 0..N/4-1:
                       C_mn = sin(pi r/2) / (N sin(pi r/N)),  r = m - n,  C_mm = 1/2.
                   Reduces to infinite_chain_C as N -> inf.
massive_chain_C    Staggered mass (the lattice Dirac mass):
                       H = -(1/2) sum (c_j^dag c_{j+1} + h.c.) + m sum (-1)^j c_j^dag c_j.
                   With T the hopping matrix and S = diag((-1)^j) one has
                   T S = -S T, hence (T + mS)^2 = T^2 + m^2, and the half-filled
                   ground-state projector is C = (1 - (T + mS)(T^2 + m^2)^{-1/2})/2.
                   Writing F = (T^2 + m^2)^{-1/2}, F_mn = f(m - n) with
                       f(r) = (1/pi) int_0^pi cos(k r) / sqrt(cos^2 k + m^2) dk
                   (f(r) = 0 for odd r by k -> pi - k), the closed form is
                       C_mn = (1/2) delta_mn + (1/4)[f(r-1) + f(r+1)] - (m/2)(-1)^m f(r).
                   Dispersion +-sqrt(cos^2 k + m^2): gap 2m, unit velocity at
                   the Fermi points as m -> 0, and correlation length
                   xi = 1/asinh(m) (the integrand's nearest singularities sit at
                   k = pi/2 +- i asinh(m)). The integral is done by the
                   trapezoidal rule, which converges exponentially for a
                   periodic analytic integrand, with the point count set by the
                   requested digits and the strip half-width asinh(m).

All three are cross-checked in tests/test_modular_lab.py: ring_C against the
validated vacuum.fermions.tight_binding_C (3e-15), massive_chain_C against a
brute-force diagonalization of a 600-site ring (1e-15), infinite_chain_C as the
N -> inf limit of ring_C.

References
----------
- V. Eisler, I. Peschel, J. Phys. A 50, 284003 (2017), arXiv:1703.08126.
- V. Eisler, E. Tonni, I. Peschel, J. Stat. Mech. (2022) 083101, arXiv:2204.03966.
- I. Peschel, V. Eisler, J. Phys. A 42, 504003 (2009) (correlation-matrix review).
"""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np

__all__ = [
    "infinite_chain_C",
    "ring_C",
    "massive_chain_C",
    "correlation_length",
    "staggered_mass_single_particle",
    "to_mp",
    "to_float",
]

_LN10 = math.log(10.0)


def to_mp(C, dps):
    """Return C as an mp.matrix (a float64 array is converted entry by entry).

    Converting a float64 array adds *no* information: the entries carry ~16
    digits, so eigenvalues within 1e-16 of 0 or 1 remain unresolved. Use the
    closed-form builders with ``dps`` set for genuine high precision.
    """
    with mp.workdps(int(dps)):
        if isinstance(C, mp.matrix):
            return C
        C = np.asarray(C, dtype=float)
        n, m = C.shape
        M = mp.matrix(n, m)
        for i in range(n):
            for j in range(m):
                M[i, j] = mp.mpf(float(C[i, j]))
        return M


def to_float(M):
    """Return M as a float64 ndarray (mp.matrix or array-like)."""
    if isinstance(M, mp.matrix):
        return np.array(M.tolist(), dtype=float)
    return np.asarray(M, dtype=float)


def _sites(sites):
    s = np.asarray(list(sites), dtype=int)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("sites must be a non-empty 1-D sequence of integers")
    return s


def infinite_chain_C(sites, k_F=None, dps=None):
    """C_mn = sin(k_F (m-n))/(pi (m-n)), C_mm = k_F/pi (Eisler-Peschel 2017, Eq. (3)).

    Parameters
    ----------
    sites : sequence of int
        Site labels (need not be contiguous); the result is indexed in this order.
    k_F : float or None
        Fermi momentum; None means exactly half filling, k_F = pi/2.
    dps : int or None
        None -> float64 ndarray; int -> mp.matrix at that many decimal digits.
    """
    s = _sites(sites)
    n = s.size
    r = s[:, None] - s[None, :]
    if dps is None:
        kF = np.pi / 2 if k_F is None else float(k_F)
        C = np.empty((n, n))
        off = r != 0
        C[off] = np.sin(kF * r[off]) / (np.pi * r[off])
        C[~off] = kF / np.pi
        return 0.5 * (C + C.T)
    with mp.workdps(int(dps)):
        kF = mp.pi / 2 if k_F is None else mp.mpf(k_F)
        M = mp.matrix(n, n)
        cache = {}
        for a in range(n):
            for b in range(a, n):
                key = int(abs(r[a, b]))  # sin(kF r)/(pi r) is even in r
                if key not in cache:
                    cache[key] = kF / mp.pi if key == 0 else mp.sin(kF * key) / (mp.pi * key)
                M[a, b] = cache[key]
                M[b, a] = cache[key]
        return M


def ring_C(N, sites, dps=None):
    """Half-filled antiperiodic ring of N = 0 mod 4 sites: C_mn = sin(pi r/2)/(N sin(pi r/N)).

    Same ground state as ``vacuum.fermions.tight_binding_C(N, 0.5, 'antiperiodic')``
    (the Fermi level is gapped for N = 0 mod 4), but in closed form so it can be
    evaluated at arbitrary precision.
    """
    N = int(N)
    if N < 4 or N % 4 != 0:
        raise ValueError(f"ring_C needs N = 0 mod 4 (unique half-filled ground state), got N={N}")
    s = _sites(sites)
    n = s.size
    r = s[:, None] - s[None, :]
    if dps is None:
        C = np.empty((n, n))
        off = r != 0
        C[off] = np.sin(np.pi * r[off] / 2) / (N * np.sin(np.pi * r[off] / N))
        C[~off] = 0.5
        return 0.5 * (C + C.T)
    with mp.workdps(int(dps)):
        M = mp.matrix(n, n)
        cache = {}
        sgn = (0, 1, 0, -1)  # sin(pi k/2) for integer k
        for a in range(n):
            for b in range(a, n):
                key = int(abs(r[a, b]))
                if key not in cache:
                    if key == 0:
                        cache[key] = mp.mpf(1) / 2
                    elif sgn[key % 4] == 0:
                        cache[key] = mp.mpf(0)
                    else:
                        cache[key] = sgn[key % 4] / (N * mp.sin(mp.pi * key / N))
                M[a, b] = cache[key]
                M[b, a] = cache[key]
        return M


def correlation_length(m):
    """xi = 1/asinh(m) for the staggered-mass chain (singularity of 1/sqrt(cos^2 k + m^2))."""
    m = float(m)
    if m <= 0.0:
        return np.inf
    return 1.0 / math.asinh(m)


def _mass_f_table(m, rmax, digits, dps=None):
    """f(r) = (1/pi) int_0^pi cos(k r)/sqrt(cos^2 k + m^2) dk for r = 0..rmax.

    Trapezoidal rule on [0, pi) with M points: the integrand is pi-periodic and
    analytic in the strip |Im k| < asinh(m), so the error is ~exp(-2 M asinh(m)).
    """
    a = math.asinh(float(m))
    Mpts = max(64, int(math.ceil((digits * _LN10 + 20.0) / (2.0 * a))))
    # Half-step-shifted grid k_j = pi (j + 1/2)/M on [0, pi): the integrand is
    # pi-periodic for even r (exponential convergence) and anti-periodic for odd r,
    # where the integral vanishes identically; the shifted grid makes that
    # cancellation exact under j -> M-1-j, and we set the odd entries to 0 anyway.
    if dps is None:
        k = np.pi * (np.arange(Mpts) + 0.5) / Mpts
        g = 1.0 / np.sqrt(np.cos(k) ** 2 + float(m) ** 2)
        rs = np.arange(rmax + 1)
        f = (np.cos(np.outer(rs, k)) * g).sum(axis=1) / Mpts
        f[1::2] = 0.0
        return f
    with mp.workdps(int(dps)):
        mm = mp.mpf(m)
        f = [mp.mpf(0)] * (rmax + 1)
        for j in range(Mpts):
            c = mp.cos(mp.pi * (j + mp.mpf(1) / 2) / Mpts)
            g = 1 / mp.sqrt(c * c + mm * mm)
            # Chebyshev recurrence cos(k r) = T_r(cos k)
            t_prev, t_cur = mp.mpf(1), c
            f[0] += g
            if rmax >= 1:
                f[1] += g * c
            for r in range(2, rmax + 1):
                t_prev, t_cur = t_cur, 2 * c * t_cur - t_prev
                f[r] += g * t_cur
        f = [v / Mpts for v in f]
        for r in range(1, rmax + 1, 2):
            f[r] = mp.mpf(0)
        return f


def massive_chain_C(sites, m, dps=None):
    """Ground-state correlation matrix of the staggered-mass chain (see module docstring).

    H = -(1/2) sum_j (c_j^dag c_{j+1} + h.c.) + m sum_j (-1)^j c_j^dag c_j, half filling.
    The parity (-1)^m uses the *site labels*, so non-contiguous site sets are consistent.
    m = 0 is delegated to :func:`infinite_chain_C`.
    """
    m = float(m)
    if m < 0.0:
        raise ValueError("mass must be >= 0")
    s = _sites(sites)
    if m == 0.0:
        return infinite_chain_C(s, dps=dps)
    n = s.size
    r = s[:, None] - s[None, :]
    rmax = int(np.max(np.abs(r))) + 1
    if dps is None:
        f = _mass_f_table(m, rmax, digits=17, dps=None)
        C = np.empty((n, n))
        for a in range(n):
            for b in range(n):
                rr = int(r[a, b])
                val = 0.25 * (f[abs(rr - 1)] + f[abs(rr + 1)]) - 0.5 * m * ((-1) ** int(s[a])) * f[abs(rr)]
                if rr == 0:
                    val += 0.5
                C[a, b] = val
        return 0.5 * (C + C.T)
    with mp.workdps(int(dps)):
        f = _mass_f_table(m, rmax, digits=int(dps), dps=int(dps))
        mm = mp.mpf(m)
        half = mp.mpf(1) / 2
        M = mp.matrix(n, n)
        for a in range(n):
            for b in range(n):
                rr = int(r[a, b])
                val = (f[abs(rr - 1)] + f[abs(rr + 1)]) / 4 - (mm / 2) * ((-1) ** int(s[a])) * f[abs(rr)]
                if rr == 0:
                    val += half
                M[a, b] = val
        return M


def staggered_mass_single_particle(N, m, bc="antiperiodic"):
    """Single-particle matrix T + m S of the staggered-mass ring/chain (float64, brute-force checks).

    T has -1/2 on nearest-neighbour bonds (boundary bond +1/2 for 'antiperiodic',
    -1/2 for 'periodic', absent for 'open'); S = diag((-1)^j).
    """
    N = int(N)
    T = np.zeros((N, N))
    idx = np.arange(N - 1)
    T[idx, idx + 1] = -0.5
    T[idx + 1, idx] = -0.5
    if bc == "antiperiodic":
        T[0, N - 1] += 0.5
        T[N - 1, 0] += 0.5
    elif bc == "periodic":
        T[0, N - 1] += -0.5
        T[N - 1, 0] += -0.5
    elif bc != "open":
        raise ValueError(f"bc must be 'antiperiodic', 'periodic' or 'open', got {bc!r}")
    return T + float(m) * np.diag([(-1.0) ** j for j in range(N)])
