"""Casini-Huerta exact modular Hamiltonian of the massless free fermion on n intervals.

Transcribed from H. Casini, M. Huerta, "Reduced density matrix and internal
dynamics for multicomponent regions", Class. Quantum Grav. 26, 185005 (2009),
arXiv:0903.5284 [hep-th] (referred to as CH below; equation numbers are theirs).
For each chirality the reduced state of V = U_i (a_i, b_i) is rho = c e^{-H}
(Eq. (2)) with H = int Psi^dag(x) H(x, y) Psi(y) (Eqs. (28)-(29)), where the
chiral correlator kernel is

    D(x, y) = (1/2) delta(x - y) - (i/2pi) 1/(x - y)                        (30)

(principal value), H = -log(D^{-1} - 1) (29), and the resolvent (32) of the
Cauchy kernel on multicomponent sets (Muskhelishvili) gives, with

    z(x) = log( - prod_i (x - a_i) / prod_i (x - b_i) )                      (33)

(monotonic from -inf to +inf inside every interval), the closed form

    H(x, y) = -2 pi i  delta(z(x) - z(y)) / (x - y)  =  H_loc + H_noloc.  (44), (45)

The solution y = x of z(x) = z(y) is the LOCAL term,

    H_loc = pi i ( 2 (dz/dx)^{-1} d_x + d/dx (dz/dx)^{-1} ) delta(x - y),     (46)

i.e. H_loc = 2 pi int dx (z'(x))^{-1} T(x): a Bisognano-Wichmann form with the
local inverse temperature beta(x) = 2 pi / z'(x) (for one interval
beta = 2 pi (x - a)(b - x)/(b - a); for a half line 2 pi x). The other n - 1
solutions x_l(z(x)), one in each other interval, give the NONLOCAL term

    H_noloc = -2 pi i sum_{l, x_l(z(x)) != x} (1/(x - y)) (dz/dy)^{-1} delta(y - x_l(z(x))),   (48)

which couples each point to exactly one conjugate point per other interval.
For two intervals the conjugate point is the Mobius map

    x_1(z) = [(b_1 b_2 - a_1 a_2) x_2 + (b_1 + b_2) a_1 a_2 - (a_1 + a_2) b_1 b_2]
             / [(b_1 + b_2 - a_1 - a_2) x_2 + a_1 a_2 - b_1 b_2].                        (47)

Modular flow (Sec. 3.3): z moves at constant speed, z(tau) = z_0 + 2 pi tau
(55); for one interval the point transformation is

    u(tau) = [b (u_0 - a) + e^{-2 pi tau} a (b - u_0)] / [(u_0 - a) + e^{-2 pi tau} (b - u_0)];   (53)

for two intervals the fields on the two related trajectories rotate into each
other by the angle (59), which in the translation-invariant form of
Eisler-Tonni-Peschel 2022 Eqs. (8)-(9) (x_c = x_0 - R^2/(x - x_0)) reads

    theta(tau) = arctan((x_1(tau) - x_0)/R) - arctan((x_1(0) - x_0)/R),
    x_0 = (b_1 b_2 - a_1 a_2)/(b_1 + b_2 - a_1 - a_2),
    R^2 = (b_1 - a_1)(b_2 - a_2)(b_2 - a_1)(a_2 - b_1)/(b_1 + b_2 - a_1 - a_2)^2,

so that a narrow right-moving wavepacket started at x_1(0) in A_1 carries the
weight sin^2 theta(tau) in A_2 at modular time tau (|theta| never reaches pi/2:
the "teleportation" is never complete). The massless entropy is CH Eq. (63);
for a Dirac fermion (two chiralities) the coefficient is 1/3, and the mutual
information of two intervals is the cutoff-free cross ratio
I(A_1 : A_2) = (1/3) log[(a_2 - a_1)(b_2 - b_1)/((a_2 - b_1)(b_2 - a_1))].

Lattice-to-continuum dictionary (stated once here, tested in tests/test_modular_lab.py)
-------------------------------------------------------------------------------------
The lattice model is H = -t sum (c_i^dag c_{i+1} + h.c.), t = 1/2, at half
filling (k_F = pi/2, v_F = 2 t sin k_F = 1), whose low-energy content is a
Dirac fermion c_j ~ e^{i k_F j} psi_R(j) + e^{-i k_F j} psi_L(j)
(Eisler-Tonni-Peschel 2022, Eq. (21)). CH is a continuum result, so:

 (D1) geometry: site j occupies the cell (j, j+1), i.e. sits at x_j = j + 1/2;
      a block of sites {a, ..., b-1} is the interval (a, b); bonds (j, j+1) sit
      at x = j + 1.
 (D2) chirality gauge: the right-mover kernel is H_R(i, j) = e^{i k_F (j - i)} h_ij
      (``to_chiral_gauge``); the left movers are its complex conjugate. The
      lattice correlator e^{i k_F (m-n)} C_mn -> (1/2) delta + i/(2 pi (x - y))
      is the complex conjugate of CH's D (30), so H_R = conj(H_CH)
      = +2 pi i delta(z(x) - z(y))/(x - y). Its local part sampled on
      nearest-neighbour bonds is H_R(i, i+1) = -pi i / z'(x_{i+1}), hence
      h_{i,i+1} = -beta(x)/2 = -pi/z'(x): the modular Hamiltonian is
      beta(x) times the lattice energy density (hopping -1/2), as
      Bisognano-Wichmann says. Row-integrating the nonlocal part gives
      sum_j e^{i k_F (j - i)} h_ij -> 2 pi i K(x_i - x_l)/z'(x_l), K(r) = 1/r,
      i.e. the alternating row sum
          S(x_i) = sum_{j in A_l} sin(k_F (j - i)) h_ij -> 2 pi / ((x_i - x_l) z'(x_l)),
      which is exactly Eisler-Tonni-Peschel 2022 Eq. (32) and their bilocal
      weight 2 pi beta~(x) = 2 pi beta(x_c)/(x - x_c) (their Eq. (11)).
 (D3) the local weight must be *resummed* over the odd-range hoppings of h
      (Eisler-Tonni-Peschel Eq. (31); Eisler-Peschel 2017 Eq. (22)):
      beta(x) = -(2/v_F) sum_{r odd} r sin(k_F r) h_{i,i+r} at the bond
      midpoint x = i + (r+1)/2. The nearest-neighbour term alone carries the
      lattice correction of Eisler-Peschel 2017 Eq. (27) (8 % at the centre of
      one interval, ~16 % for two).
 (D4) comparisons use test functions smooth on the lattice scale (they project
      out the 2 k_F components, i.e. the psi_R^dag psi_L cross terms).
 (D5) modular time: e^{-i tau h} moves right movers with z increasing
      (towards b_l), left movers with z decreasing.
 (D6) a ring of N sites (antiperiodic, N = 0 mod 4) is the CFT on a circle:
      x - y -> (N/pi) sin(pi (x - y)/N) in K and in z (the conformal map
      w = tan(pi x/N) makes the circle kernel unitarily equivalent to the line
      kernel, and z_circle(x) = z_line(w(x)) up to a constant).

References: CH 2009 as above; V. Eisler, E. Tonni, I. Peschel, J. Stat. Mech.
(2022) 083101, arXiv:2204.03966 (the lattice/continuum comparison for two
intervals, whose Eqs. (31)-(32) coincide with (D3) and (D2)); J. J. Bisognano,
E. H. Wichmann, J. Math. Phys. 17, 303 (1976).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

__all__ = [
    "site_positions",
    "interval_of_sites",
    "intervals_from_parts",
    "ch_z",
    "ch_zprime",
    "ch_beta",
    "which_interval",
    "ch_conjugate_points",
    "ch_mobius_conjugate",
    "ch_nonlocal_weight",
    "ch_bilinear",
    "ch_trajectory",
    "ch_single_interval_flow",
    "ch_mixing_angle",
    "ch_mutual_information",
    "to_chiral_gauge",
    "from_chiral_gauge",
    "casini_huerta_exact",
]


# --------------------------------------------------------------------------- #
# Geometry dictionary (D1)
# --------------------------------------------------------------------------- #
def site_positions(sites):
    """Continuum position of each site: x_j = j + 1/2 (site j occupies the cell (j, j+1))."""
    return np.asarray(list(sites), dtype=float) + 0.5


def interval_of_sites(A):
    """(a, b) for a contiguous block of sites {a, ..., b-1}: the interval (a, b)."""
    A = np.asarray(list(A), dtype=int)
    if A.size == 0 or np.any(np.diff(A) != 1):
        raise ValueError("A must be a non-empty contiguous, increasing block of sites")
    return float(A[0]), float(A[-1] + 1)


def intervals_from_parts(parts):
    """[(a_i, b_i)] for a list of contiguous site blocks (ordered left to right)."""
    iv = [interval_of_sites(A) for A in parts]
    for (a1, b1), (a2, b2) in zip(iv[:-1], iv[1:]):
        if not b1 < a2:
            raise ValueError("parts must be disjoint and ordered left to right with a gap between them")
    return iv


# --------------------------------------------------------------------------- #
# The function z(x) of Eq. (33) and its derivative
# --------------------------------------------------------------------------- #
def _kernel(r, N=None):
    """The Cauchy kernel K(r) = 1/r on the line; (pi/N)/sin(pi r/N) on a circle of N sites."""
    if N is None:
        return 1.0 / r
    return (np.pi / N) / np.sin(np.pi * r / N)


def ch_z(x, intervals, N=None):
    """z(x) = log(-prod(x - a_i)/prod(x - b_i)) (CH Eq. (33)); real for x inside the union.

    Evaluated as sum log|x - a_i| - sum log|x - b_i| (the sign inside the log is
    positive throughout the union, so this is exact there). N = ring size
    replaces x - c by (N/pi) sin(pi (x - c)/N) (dictionary D6).
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for a, b in intervals:
        if N is None:
            out = out + np.log(np.abs(x - a)) - np.log(np.abs(x - b))
        else:
            out = out + np.log(np.abs(np.sin(np.pi * (x - a) / N))) - np.log(np.abs(np.sin(np.pi * (x - b) / N)))
    return out


def ch_zprime(x, intervals, N=None):
    """dz/dx = sum_i [1/(x - a_i) - 1/(x - b_i)] > 0 inside the union (circle: cotangents)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for a, b in intervals:
        if N is None:
            out = out + 1.0 / (x - a) - 1.0 / (x - b)
        else:
            out = out + (np.pi / N) * (1.0 / np.tan(np.pi * (x - a) / N) - 1.0 / np.tan(np.pi * (x - b) / N))
    return out


def ch_beta(x, intervals, N=None):
    """Local inverse temperature beta(x) = 2 pi / z'(x) of the local term, CH Eq. (46).

    One interval: 2 pi (x - a)(b - x)/(b - a); half line: 2 pi x (Bisognano-Wichmann).
    """
    return 2.0 * np.pi / ch_zprime(x, intervals, N)


def which_interval(x, intervals):
    """Index of the interval containing x, or None."""
    for k, (a, b) in enumerate(intervals):
        if a < x < b:
            return k
    return None


def _root_in_interval(target, a, b, intervals, N):
    """Unique y in (a, b) with z(y) = target (z is monotonic from -inf to +inf on (a, b))."""
    d = 1e-13 * (b - a)
    f = lambda y: float(ch_z(y, intervals, N)) - target
    lo, hi = a + d, b - d
    flo, fhi = f(lo), f(hi)
    if flo >= 0.0:
        return lo
    if fhi <= 0.0:
        return hi
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200)


def ch_conjugate_points(x, intervals, N=None):
    """{l: x_l} — the conjugate point of x in every *other* interval l (solutions of z(x_l) = z(x))."""
    x = float(x)
    k = which_interval(x, intervals)
    if k is None:
        raise ValueError(f"x = {x} is not inside any interval")
    zx = float(ch_z(x, intervals, N))
    out = {}
    for l, (a, b) in enumerate(intervals):
        if l == k:
            continue
        out[l] = _root_in_interval(zx, a, b, intervals, N)
    return out


def ch_mobius_conjugate(x, A1, A2):
    """Two-interval conjugate point in closed form, CH Eq. (47) (a Mobius map sending a_1 -> a_2, b_1 -> b_2)."""
    a1, b1 = A1
    a2, b2 = A2
    x = np.asarray(x, dtype=float)
    num = (b1 * b2 - a1 * a2) * x + (b1 + b2) * a1 * a2 - (a1 + a2) * b1 * b2
    den = (b1 + b2 - a1 - a2) * x + a1 * a2 - b1 * b2
    return num / den


def ch_nonlocal_weight(x, intervals, l, N=None):
    """Row-integrated nonlocal coupling of x to its conjugate point in interval l.

    int dy H_R(x, y) over y near x_l = 2 pi i K(x - x_l)/z'(x_l), CH Eq. (48) in the
    right-mover gauge (D2); this function returns the real coefficient
    2 pi K(x - x_l)/z'(x_l) = 2 pi beta~(x) of Eisler-Tonni-Peschel 2022 Eq. (11),
    which the lattice alternating row sum S(x) (their Eq. (32)) converges to.
    """
    xl = ch_conjugate_points(x, intervals, N)[l]
    return 2.0 * np.pi * _kernel(float(x) - xl, N) / float(ch_zprime(xl, intervals, N))


def ch_bilinear(intervals, l_from, l_to, g, f, N=None):
    """Smeared bilocal matrix element  sum_{x in I_from, y in I_to} g(x) f(y) H_R(x, y)

    = 2 pi i int_{I_from} dx g(x) f(x_l(x)) K(x - x_l(x))/z'(x_l(x))  (the y integral
    of CH Eq. (48) done with the delta function). Complex; purely imaginary.
    """
    a, b = intervals[l_from]

    def integrand(x):
        xl = ch_conjugate_points(x, intervals, N)[l_to]
        return g(x) * f(xl) * _kernel(x - xl, N) / float(ch_zprime(xl, intervals, N))

    val, _ = quad(integrand, a, b, limit=200)
    return 2j * np.pi * val


def ch_trajectory(x0, tau, intervals, l, N=None):
    """Modular-flow trajectory in interval l: z(x_l(tau)) = z(x0) + 2 pi tau (CH Eq. (55)).

    Right movers (dictionary D5) move with z increasing under e^{-i tau h}, tau > 0.
    """
    a, b = intervals[l]
    target = float(ch_z(x0, intervals, N)) + 2.0 * np.pi * float(tau)
    return _root_in_interval(target, a, b, intervals, N)


def ch_single_interval_flow(x0, tau, a, b):
    """CH Eq. (53): the Mobius flow of a point of a single interval (a, b) under modular time tau."""
    e = np.exp(-2.0 * np.pi * np.asarray(tau, dtype=float))
    return (b * (x0 - a) + e * a * (b - x0)) / ((x0 - a) + e * (b - x0))


def ch_mixing_angle(x0, tau, A1, A2):
    """Rotation angle theta(tau) between the fields on the two related trajectories, CH Eq. (59).

    Translation-invariant form with x_0 and R of Eisler-Tonni-Peschel 2022 Eq. (9).
    A narrow right-moving packet started at x0 in A1 has weight sin^2 theta(tau) in A2.
    """
    a1, b1 = A1
    a2, b2 = A2
    s = b1 + b2 - a1 - a2
    xc0 = (b1 * b2 - a1 * a2) / s
    R = np.sqrt((b1 - a1) * (b2 - a2) * (b2 - a1) * (a2 - b1)) / s
    x1 = ch_trajectory(x0, tau, [A1, A2], 0)
    return np.arctan((x1 - xc0) / R) - np.arctan((x0 - xc0) / R)


def ch_mutual_information(A1, A2):
    """I(A1:A2) = (1/3) log[(a2-a1)(b2-b1)/((a2-b1)(b2-a1))] for the massless Dirac fermion (CH Eq. (63), both chiralities)."""
    a1, b1 = A1
    a2, b2 = A2
    return np.log((a2 - a1) * (b2 - b1) / ((a2 - b1) * (b2 - a1))) / 3.0


# --------------------------------------------------------------------------- #
# Dictionary (D2): chirality gauge
# --------------------------------------------------------------------------- #
def _phase(sites, k_F):
    s = np.asarray(list(sites), dtype=float)
    kF = np.pi / 2 if k_F is None else float(k_F)
    return np.exp(1j * kF * (s[None, :] - s[:, None]))  # e^{i k_F (s_j - s_i)}


def to_chiral_gauge(h, sites, k_F=None):
    """Right-mover kernel on the lattice: H_R(i, j) = e^{i k_F (s_j - s_i)} h_ij."""
    return np.asarray(h) * _phase(sites, k_F)


def from_chiral_gauge(H, sites, k_F=None):
    """Inverse of to_chiral_gauge: h_ij = e^{-i k_F (s_j - s_i)} H_R(i, j) (real for a consistent H_R)."""
    return np.asarray(H) * np.conj(_phase(sites, k_F))


# --------------------------------------------------------------------------- #
# The lattice-sampled CH kernel
# --------------------------------------------------------------------------- #
def casini_huerta_exact(A1, A2, N=None, k_F=None, from_side=0):
    """The CH two-interval kernel sampled on the lattice, in the right-mover gauge.

    Returns the complex Hermitian matrix H_R on the sites A1 + A2 (in that order):

    - local part on nearest-neighbour bonds, H_R(i, i+1) = -pi i / z'(x_{i+1}),
      H_R(i+1, i) = +pi i / z'(x_{i+1})  (CH Eq. (46) with dictionary D1-D2), so
      that ``from_chiral_gauge`` gives h_{i,i+1} = -beta(x)/2;
    - nonlocal part: for each site i of the interval ``from_side`` (0 = A1), the
      weight 2 pi i K(x_i - x_l)/z'(x_l) (CH Eq. (48) integrated over the cell)
      is placed by linear interpolation on the two sites of the other interval
      bracketing the conjugate point x_l, restricted at half filling (k_F = None)
      to sites with (j - i) odd — the only ones the particle-hole-symmetric
      lattice h can populate. The opposite block is the Hermitian conjugate, so
      the row sums on the ``from_side`` interval reproduce
      ``ch_nonlocal_weight`` exactly, while on the other interval they are the
      column sums of a delta sampled on a parity grid and fluctuate at O(1)
      from row to row (the kernel is a distribution; its Jacobian is only
      reproduced on average).

    This is a *sampling*: element by element it need not match the lattice h
    (which carries lattice-scale 2 k_F structure), but its smeared bilinears do
    (tests/test_modular_lab.py). Compare with the lattice h through
    ``vacuum.modular.bilocal_weight``, ``vacuum.modular.chiral_bilinear`` or
    ``from_chiral_gauge``.
    """
    A1 = [int(s) for s in A1]
    A2 = [int(s) for s in A2]
    parts = [A1, A2]
    intervals = intervals_from_parts(parts)
    sites = A1 + A2
    n = len(sites)
    pos = {s: k for k, s in enumerate(sites)}
    x = site_positions(sites)
    H = np.zeros((n, n), dtype=complex)
    half_filling = k_F is None

    # local part
    for A in parts:
        for s in A[:-1]:
            xb = float(s + 1)
            w = np.pi / float(ch_zprime(xb, intervals, N))
            H[pos[s], pos[s + 1]] += -1j * w
            H[pos[s + 1], pos[s]] += +1j * w

    # nonlocal part, sampled row-exactly from one side, completed by Hermiticity
    Hn = np.zeros((n, n), dtype=complex)
    for k, A in enumerate(parts):
        if k != int(from_side):
            continue
        l = 1 - k
        B = parts[l]
        for s in A:
            i = pos[s]
            xl = ch_conjugate_points(x[i], intervals, N)[l]
            w = 2j * np.pi * _kernel(x[i] - xl, N) / float(ch_zprime(xl, intervals, N))
            t = xl - 0.5  # site coordinate of the conjugate point
            j_lo = int(np.floor(t))
            step = 1
            if half_filling:
                step = 2
                if (j_lo - s) % 2 == 0:
                    j_lo -= 1
            j_hi = j_lo + step
            w_hi = (t - j_lo) / step
            w_lo = 1.0 - w_hi
            cand = [(j_lo, w_lo), (j_hi, w_hi)]
            inside = [(j, wt) for j, wt in cand if j in pos and j in B]
            if not inside:
                # conjugate point too close to the edge for the parity grid: nearest valid site
                jbest = min(B, key=lambda j: abs(j - t) + (0 if (not half_filling or (j - s) % 2 == 1) else 1e3))
                inside = [(jbest, 1.0)]
            tot = sum(wt for _, wt in inside)
            for j, wt in inside:
                Hn[i, pos[j]] += w * wt / tot
    Hn = Hn + Hn.conj().T
    return H + Hn
