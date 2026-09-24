"""The modular-Hamiltonian laboratory on free-fermion chains (Peschel-Eisler methods).

For a Gaussian fermionic state the reduced density matrix of a set of sites A
is rho_A = exp(-sum_ij h_ij c_i^dag c_j)/Z with h = ln((1 - C_A)/C_A) (Peschel
2003; Peschel-Eisler 2009). This module turns that formula into instruments:

interval_modular / two_interval_modular / multi_interval_modular
    h for one, two or several blocks. Two precision paths (see
    vacuum.modular.lattice for why): float64 through the validated
    vacuum.fermions.entanglement_hamiltonian, trustworthy only while no
    occupation is clipped (|eps| < ~32, i.e. <~ 18 sites at criticality), and
    an mpmath path (Jacobi/QR diagonalization of the closed-form C_A at `dps`
    digits) for larger blocks. Both flag insufficient precision LOUDLY with a
    ModularPrecisionWarning — never silently — in keeping with the repository's
    precision audit (docs/API.md).
bw_profile
    The local inverse-temperature profile beta(x). Three extractions:
    'nn' (nearest-neighbour hopping only, beta = -h_{i,i+1}/t; this is the
    quantity Eisler-Peschel 2017 Eq. (27) describes, with its persistent lattice
    correction), 'resummed' (Eisler-Peschel 2017 Eq. (22) / Eisler-Tonni-Peschel
    2022 Eq. (31): all odd-range hoppings resummed at their midpoints,
    beta(x) = -(2/v_F) sum_r r sin(k_F r) h_{i,i+r}, which converges to the
    conformal / Bisognano-Wichmann profile), and 'row' (their Eq. (33), any filling).
bilocal_weight
    The alternating row sums S(x_i) = sum_j sin(k_F (j - i)) h_ij between two
    blocks (Eisler-Tonni-Peschel 2022 Eq. (32)), the lattice observable of the
    Casini-Huerta nonlocal term (vacuum.modular.casini_huerta, dictionary D2).
chiral_bilinear
    Smeared matrix elements sum g(x_i) f(x_j) e^{i k_F (j - i)} h_ij between
    right-moving test envelopes (dictionary D4).
nonlocal_decay, locality_score
    Diagnostics of the inter-block couplings and of how banded h is.
modular_flow, modular_flow_unitary, flow_wavefunction, wavepacket
    The modular flow e^{-i s h} on correlation matrices and single-particle
    wavefunctions; [h, C_A] = 0 makes C_A invariant, any other correlation
    matrix is rotated isospectrally (unitarity anchor).
eisler_peschel_eq26 / eq27 / eq54
    The analytic lattice results used as anchors.

Conventions: C_ij = <c_i^dag c_j> real symmetric (time-reversal-symmetric
chains); entropies in nats; hopping t = 1/2 and k_F = pi/2 unless stated.
"""

from __future__ import annotations

import math
import warnings

import mpmath as mp
import numpy as np

from vacuum.fermions.correlation import entanglement_hamiltonian as _eh_float64

from .lattice import to_float, to_mp

__all__ = [
    "ModularPrecisionWarning",
    "entanglement_spectrum",
    "required_dps",
    "interval_modular",
    "two_interval_modular",
    "multi_interval_modular",
    "entropy_from_h",
    "bw_profile",
    "bilocal_weight",
    "chiral_bilinear",
    "nonlocal_decay",
    "locality_score",
    "modular_flow_unitary",
    "modular_flow",
    "flow_wavefunction",
    "wavepacket",
    "part_weights",
    "part_centroids",
    "eisler_peschel_eq26",
    "eisler_peschel_eq27",
    "eisler_peschel_eq54",
]

_LN10 = math.log(10.0)
_ZETA_CLIP_FLOAT = 1e-14  # must match vacuum.fermions.correlation._ZETA_CLIP
_FLOAT_EPS_MAX = -math.log(_ZETA_CLIP_FLOAT)  # ~32.2: largest |eps| float64 can represent
_MARGIN_CHECK = 15  # digits of margin required beyond eps_max/ln(10)


class ModularPrecisionWarning(UserWarning):
    """The entanglement Hamiltonian was computed with insufficient precision (clipped occupations)."""


def required_dps(eps_max, margin=25):
    """Decimal digits needed to resolve occupations 1 - zeta ~ e^{-eps_max}, with margin."""
    return int(math.ceil(float(eps_max) / _LN10)) + int(margin)


def _mp_submatrix(C, idx):
    n = len(idx)
    M = mp.matrix(n, n)
    for a in range(n):
        for b in range(n):
            M[a, b] = C[idx[a], idx[b]]
    return M


def entanglement_spectrum(C_A, dps=None):
    """Single-particle entanglement energies eps_k = ln((1 - zeta_k)/zeta_k) and modes.

    Returns (eps, phi, info): eps (n,) float64 ascending in zeta order of the
    solver, phi (n, n) float64 with modes as columns, info = {'dps',
    'eps_max', 'n_clipped', 'digits_needed', 'sufficient'}. In the mp path the
    occupations are clipped at 10^-(dps-5) and every clip is counted; a
    nonzero count means the precision was insufficient for this block.
    """
    if dps is None:
        C = to_float(C_A)
        C = 0.5 * (C + C.T)
        zeta, U = np.linalg.eigh(C)
        n_clipped = int(np.sum((zeta < _ZETA_CLIP_FLOAT) | (zeta > 1.0 - _ZETA_CLIP_FLOAT)))
        z = np.clip(zeta, _ZETA_CLIP_FLOAT, 1.0 - _ZETA_CLIP_FLOAT)
        eps = np.log1p(-z) - np.log(z)
        eps_max = float(np.max(np.abs(eps)))
        info = {
            "dps": None,
            "eps_max": eps_max,
            "n_clipped": n_clipped,
            "digits_needed": eps_max / _LN10 + _MARGIN_CHECK,
            "sufficient": n_clipped == 0,
        }
        return eps, U, info
    dps = int(dps)
    with mp.workdps(dps):
        M = C_A if isinstance(C_A, mp.matrix) else to_mp(C_A, dps)
        n = M.rows
        E, Q = mp.eigsy(M)
        floor = mp.mpf(10) ** (-(dps - 5))
        eps = np.empty(n)
        n_clipped = 0
        for k in range(n):
            z = E[k]
            if z <= floor or z >= 1 - floor:
                n_clipped += 1
                z = min(max(z, floor), 1 - floor)
            eps[k] = float(mp.log((1 - z) / z))
        phi = np.array(Q.tolist(), dtype=float)
    eps_max = float(np.max(np.abs(eps)))
    info = {
        "dps": dps,
        "eps_max": eps_max,
        "n_clipped": n_clipped,
        "digits_needed": eps_max / _LN10 + _MARGIN_CHECK,
        "sufficient": (n_clipped == 0) and (eps_max / _LN10 + _MARGIN_CHECK <= dps),
    }
    return eps, phi, info


def interval_modular(C, sites, dps=None, return_info=False):
    """Entanglement (modular) Hamiltonian h = ln((1 - C_A)/C_A) of the block `sites`.

    Parameters
    ----------
    C : ndarray or mp.matrix
        Correlation matrix C_ij = <c_i^dag c_j> over some site labelling; `sites`
        index its rows/columns. For the mp path pass a closed-form mp.matrix
        (vacuum.modular.lattice); a float64 array converted to mp carries only
        ~16 digits and is flagged if that is not enough.
    sites : sequence of int
        0-based rows of C forming the block (any order; h is indexed in this order).
    dps : int or None
        None -> float64 via vacuum.fermions.entanglement_hamiltonian (valid while
        no occupation is clipped, |eps| < ~32); int -> mpmath at `dps` digits.
    return_info : bool
        If True return (h, info) with the precision diagnostics of
        :func:`entanglement_spectrum`.

    Warns
    -----
    ModularPrecisionWarning if the precision was insufficient (occupations
    clipped, or digits < eps_max/ln10 + 15). The returned h is then the
    regularized operator, wrong at O(1) in the interior — see lattice.py.
    """
    sites = [int(s) for s in sites]
    if dps is None:
        C_A = to_float(C)[np.ix_(sites, sites)]
        h = _eh_float64(C_A)  # the validated Layer-1 formula
        _, _, info = entanglement_spectrum(C_A, None)
        if not info["sufficient"]:
            warnings.warn(
                f"float64 entanglement Hamiltonian: {info['n_clipped']} occupation(s) clipped "
                f"at 1e-14 (block of {len(sites)} sites, eps_max would exceed {_FLOAT_EPS_MAX:.1f}); "
                f"h is wrong at O(1) in the interior. Use the mp path (dps ~ {info['digits_needed']:.0f}+).",
                ModularPrecisionWarning,
                stacklevel=2,
            )
        return (h, info) if return_info else h
    dps = int(dps)
    if isinstance(C, mp.matrix):
        C_A = _mp_submatrix(C, sites)
        input_float64 = False
    else:
        C_A = to_mp(np.asarray(C, dtype=float)[np.ix_(sites, sites)], dps)
        input_float64 = True
    eps, phi, info = entanglement_spectrum(C_A, dps)
    info["input_float64"] = input_float64
    if input_float64 and info["eps_max"] > 30.0:
        info["sufficient"] = False
        warnings.warn(
            f"mp path on a float64 correlation matrix: eps_max = {info['eps_max']:.1f} exceeds what "
            "16-digit input can resolve (~34); build C in closed form at high precision "
            "(vacuum.modular.lattice).",
            ModularPrecisionWarning,
            stacklevel=2,
        )
    elif not info["sufficient"]:
        warnings.warn(
            f"mp entanglement Hamiltonian at dps={dps}: {info['n_clipped']} clipped occupation(s), "
            f"eps_max = {info['eps_max']:.1f} needs ~{info['digits_needed']:.0f} digits.",
            ModularPrecisionWarning,
            stacklevel=2,
        )
    h = (phi * eps) @ phi.T
    h = 0.5 * (h + h.T)
    return (h, info) if return_info else h


def two_interval_modular(C, A1, A2, dps=None, return_info=False):
    """h on the union A1 + A2 (indexed A1 first, then A2)."""
    return interval_modular(C, list(A1) + list(A2), dps=dps, return_info=return_info)


def multi_interval_modular(C, parts, dps=None, return_info=False):
    """h on the union of the blocks in `parts` (concatenated in order)."""
    sites = [int(s) for A in parts for s in A]
    return interval_modular(C, sites, dps=dps, return_info=return_info)


def entropy_from_h(h):
    """Free-fermion entropy (nats) from h: S = sum_k [ln(1 + e^{-eps_k}) + eps_k/(e^{eps_k} + 1)]."""
    eps = np.linalg.eigvalsh(np.asarray(h, dtype=float))
    return float(np.sum(np.logaddexp(0.0, -eps) + eps / (np.exp(eps) + 1.0)))


# --------------------------------------------------------------------------- #
# Bisognano-Wichmann profile
# --------------------------------------------------------------------------- #
def bw_profile(h, hopping=0.5, k_F=None, mode="nn", offset=0):
    """Local inverse temperature beta(x) from the hopping structure of h.

    K = sum_ij h_ij c_i^dag c_j is compared with int beta(x) T_00(x) dx where
    T_00 is the density of the *physical* lattice Hamiltonian -t sum (c^dag c + h.c.)
    (Fermi velocity v_F = 2 t sin k_F). Bisognano-Wichmann / CFT predict
    beta(x) = 2 pi dist(x, boundary) near an edge and 2 pi x(L - x)/L for an
    interval (Casini-Huerta beta = 2 pi/z' in general).

    mode='nn'        beta = -h_{i,i+1}/t at the bond positions x = offset + 1..n-1.
                     Carries the lattice correction of Eisler-Peschel 2017 Eq. (27).
    mode='resummed'  beta(x) = -(2/v_F) sum_{r odd} r sin(k_F r) h_{i,i+r} at the bond
                     midpoint x = i + (r+1)/2 (Eisler-Tonni-Peschel 2022 Eq. (31));
                     half filling only (even ranges vanish there).
    mode='row'       beta(x_i) = -(1/v_F) sum_j (j-i) sin(k_F (j-i)) h_ij at the site
                     positions x = offset + i + 1/2 (their Eq. (33), any filling).

    `h` must be the block of a single contiguous interval whose first site is `offset`.
    Returns (x, beta).
    """
    h = np.asarray(h, dtype=float)
    n = h.shape[0]
    kF = np.pi / 2 if k_F is None else float(k_F)
    vF = 2.0 * float(hopping) * np.sin(kF)
    if mode == "nn":
        x = float(offset) + np.arange(1, n, dtype=float)
        return x, -np.diag(h, 1) / float(hopping)
    if mode == "resummed":
        if k_F is not None and abs(kF - np.pi / 2) > 1e-12:
            raise ValueError("mode='resummed' (perpendicular sums) is defined at half filling only; use mode='row'")
        x = float(offset) + np.arange(1, n, dtype=float)
        beta = np.zeros(n - 1)
        for r in range(1, n, 2):
            w = -(2.0 / vF) * r * np.sin(kF * r)
            d = np.diag(h, r)
            mid = np.arange(d.size) + (r + 1) // 2  # bond index (1-based position - 1)
            np.add.at(beta, mid - 1, w * d)
        return x, beta
    if mode == "row":
        x = float(offset) + np.arange(n, dtype=float) + 0.5
        r = np.arange(n)[None, :] - np.arange(n)[:, None]
        beta = -(1.0 / vF) * np.sum(r * np.sin(kF * r) * h, axis=1)
        return x, beta
    raise ValueError(f"mode must be 'nn', 'resummed' or 'row', got {mode!r}")


# --------------------------------------------------------------------------- #
# Nonlocal (bilocal) structure
# --------------------------------------------------------------------------- #
def bilocal_weight(h, sites, rows, cols, k_F=None):
    """Alternating row sums S(x_i) = sum_{j in cols} sin(k_F (s_j - s_i)) h_ij for i in rows.

    Eisler-Tonni-Peschel 2022 Eq. (32) (k_F = pi/2: (-1)^{(j-i-1)/2} on odd j - i);
    converges to the Casini-Huerta weight 2 pi/((x_i - x_l) z'(x_l))
    (vacuum.modular.casini_huerta.ch_nonlocal_weight) — dictionary D2.
    """
    h = np.asarray(h, dtype=float)
    s = np.asarray(list(sites), dtype=float)
    kF = np.pi / 2 if k_F is None else float(k_F)
    rows = np.asarray(list(rows), dtype=int)
    cols = np.asarray(list(cols), dtype=int)
    ph = np.sin(kF * (s[cols][None, :] - s[rows][:, None]))
    return np.sum(ph * h[np.ix_(rows, cols)], axis=1)


def chiral_bilinear(h, sites, g, f, rows, cols, k_F=None):
    """sum_{i in rows, j in cols} g(x_i) f(x_j) e^{i k_F (s_j - s_i)} h_ij  (x = s + 1/2).

    The smeared right-mover matrix element (dictionary D2, D4); compare with
    vacuum.modular.casini_huerta.ch_bilinear.
    """
    h = np.asarray(h, dtype=float)
    s = np.asarray(list(sites), dtype=float)
    kF = np.pi / 2 if k_F is None else float(k_F)
    rows = np.asarray(list(rows), dtype=int)
    cols = np.asarray(list(cols), dtype=int)
    gi = np.asarray([g(v) for v in s[rows] + 0.5], dtype=float)
    fj = np.asarray([f(v) for v in s[cols] + 0.5], dtype=float)
    ph = np.exp(1j * kF * (s[cols][None, :] - s[rows][:, None]))
    return complex(np.sum(gi[:, None] * fj[None, :] * ph * h[np.ix_(rows, cols)]))


def nonlocal_decay(h, sites, parts, k_F=None):
    """Inter-block couplings of h for the partition `parts` (index lists into h).

    Returns a dict with
      'offblock_fraction' : ||off-diagonal blocks||_F^2 / ||h||_F^2,
      'pairs' : list of dicts, one per pair p < q, with keys
          'p', 'q', 'gap' (minimal site distance |s_j - s_i| between the blocks,
          i.e. number of empty sites + 1), 'fro', 'max_abs',
          'sum_abs', 'distance' (sorted lattice distances |s_j - s_i| occurring in
          the block), 'profile' (max |h_ij| at each of those distances),
          'row_weight_pq' / 'row_weight_qp' (alternating row sums, bilocal_weight).
    """
    h = np.asarray(h, dtype=float)
    s = np.asarray(list(sites), dtype=float)
    parts = [np.asarray(list(P), dtype=int) for P in parts]
    tot = float(np.sum(h ** 2))
    off = 0.0
    pairs = []
    for p in range(len(parts)):
        for q in range(p + 1, len(parts)):
            blk = h[np.ix_(parts[p], parts[q])]
            dist = np.abs(s[parts[q]][None, :] - s[parts[p]][:, None])
            ds = np.unique(dist)
            prof = np.array([np.max(np.abs(blk[dist == d])) for d in ds])
            off += 2.0 * float(np.sum(blk ** 2))
            pairs.append(
                {
                    "p": p,
                    "q": q,
                    "gap": float(np.min(dist)),
                    "fro": float(np.linalg.norm(blk)),
                    "max_abs": float(np.max(np.abs(blk))),
                    "sum_abs": float(np.sum(np.abs(blk))),
                    "distance": ds,
                    "profile": prof,
                    "row_weight_pq": bilocal_weight(h, s, parts[p], parts[q], k_F),
                    "row_weight_qp": bilocal_weight(h, s, parts[q], parts[p], k_F),
                }
            )
    return {"offblock_fraction": off / tot if tot > 0 else 0.0, "pairs": pairs}


def locality_score(M, bandwidth=1, block=1):
    """Fraction of ||M||_F^2 within |i - j| <= bandwidth (in mode index).

    block=1: M is indexed by sites (fermionic h). block=2: M is a (2N, 2N)
    quadratic form in block ordering R = (x_1..x_N, p_1..p_N) (bosonic G) and
    the mode index is i mod N. 1.0 means strictly banded; a tridiagonal
    (Bisognano-Wichmann-local) h scores 1.0 at bandwidth 1.
    """
    M = np.asarray(M)
    n = M.shape[0]
    if block == 2:
        idx = np.arange(n) % (n // 2)
    else:
        idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :])
    w = np.abs(M) ** 2
    tot = float(np.sum(w))
    if tot == 0.0:
        return 1.0
    return float(np.sum(w[dist <= bandwidth]) / tot)


# --------------------------------------------------------------------------- #
# Modular flow
# --------------------------------------------------------------------------- #
def modular_flow_unitary(h, s):
    """U(s) = exp(-i s h) via the eigendecomposition of the real symmetric h."""
    h = np.asarray(h, dtype=float)
    e, Q = np.linalg.eigh(0.5 * (h + h.T))
    return (Q * np.exp(-1j * float(s) * e)) @ Q.T


def modular_flow(h, C_A, s):
    """Flow of a correlation matrix under the modular unitary: C_s = U C_A U^dag, U = e^{-i s h}.

    C_A itself is invariant ([h, C_A] = 0); any other correlation matrix is
    rotated isospectrally (the spectrum of C is preserved to roundoff).
    """
    U = modular_flow_unitary(h, s)
    return U @ np.asarray(C_A) @ U.conj().T


def flow_wavefunction(h, psi, s):
    """Single-particle wavefunction under the modular flow: psi_s = e^{-i s h} psi (dictionary D5)."""
    return modular_flow_unitary(h, s) @ np.asarray(psi)


def wavepacket(sites, x0, sigma, k_F=None, chirality=+1):
    """Normalized Gaussian packet e^{i chirality k_F s_j} exp(-(x_j - x0)^2/(4 sigma^2)), x_j = s_j + 1/2.

    chirality +1: right mover (moves towards larger x under e^{-i s h}, s > 0); -1: left mover.
    """
    s = np.asarray(list(sites), dtype=float)
    kF = np.pi / 2 if k_F is None else float(k_F)
    psi = np.exp(1j * chirality * kF * s) * np.exp(-((s + 0.5 - x0) ** 2) / (4.0 * sigma ** 2))
    return psi / np.linalg.norm(psi)


def part_weights(psi, parts):
    """Probability sum_{j in part} |psi_j|^2 for each part (index lists)."""
    w = np.abs(np.asarray(psi)) ** 2
    return np.array([float(np.sum(w[np.asarray(list(P), dtype=int)])) for P in parts])


def part_centroids(psi, sites, parts):
    """|psi|^2-weighted centroid of x = s + 1/2 within each part (nan if the part is empty of weight)."""
    w = np.abs(np.asarray(psi)) ** 2
    x = np.asarray(list(sites), dtype=float) + 0.5
    out = []
    for P in parts:
        P = np.asarray(list(P), dtype=int)
        tot = float(np.sum(w[P]))
        out.append(float(np.sum(w[P] * x[P]) / tot) if tot > 1e-300 else np.nan)
    return np.array(out)


# --------------------------------------------------------------------------- #
# Eisler-Peschel analytic lattice results (anchors)
# --------------------------------------------------------------------------- #
def _dfact(n):
    """Double factorial with (-1)!! = 0!! = 1."""
    out = 1
    while n > 1:
        out *= n
        n -= 2
    return out


def eisler_peschel_eq26(x, p):
    """Eisler-Peschel 2017 Eq. (26): scaled hopping h_{i,i+2p+1} of the half-filled interval.

    h = pi (2p-1)!! (4p-1)!! / (2^p p! (2p+1)!) z^{2p+1} 3F2(p+1/4, p+1/2, p+3/4; p+1, 2p+2; (4z)^2),
    z = x(1 - x), x the scaled midpoint of the hop. The lattice value is L * h.
    """
    x = float(x)
    z = x * (1.0 - x)
    pref = np.pi * _dfact(2 * p - 1) * _dfact(4 * p - 1) / (2 ** p * math.factorial(p) * math.factorial(2 * p + 1))
    F = mp.hyp3f2(p + 0.25, p + 0.5, p + 0.75, p + 1, 2 * p + 2, (4.0 * z) ** 2)
    return float(pref * z ** (2 * p + 1) * F)


def eisler_peschel_eq27(x):
    """Eisler-Peschel 2017 Eq. (27): h_{i,i+1} = pi x(1-x) 3F2(1/4,1/2,3/4; 1,2; [4x(1-x)]^2) (scaled by L)."""
    return eisler_peschel_eq26(x, 0)


def eisler_peschel_eq54(L):
    """Eisler-Peschel 2017 Eq. (54): eps_max = 1.7627 L - (1/2) ln L - 2.1320 at half filling."""
    L = float(L)
    return 1.7627 * L - 0.5 * math.log(L) - 2.1320
