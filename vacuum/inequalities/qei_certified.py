"""Computer-assisted certification of the 4d Gaussian-sampled QEI constant.

The object certified here is the ratio

    R := C_4d(gaussian) / C_FE,     C_FE = 1/(16 pi^2)

of :func:`vacuum.inequalities.qei.worldline_momentum_minimize` (``d=4``,
``operator='T00'``, Gaussian ``f``), archived numerically as
``0.4574904`` in ``papers/qei-2d-sharp-constant/data/s6b_momentum_4d_worldline.json``.
The derivation and the error budget are written out in
``papers/qei-2d-sharp-constant/draft/proof_4d_gaussian_constant.md``.

Summary of the exact formulation proved there (tau0 = 1, t0 = 0, f^2 smearing,
Fhat(nu) = int f(t)^2 e^{i nu t} dt = sqrt(pi) e^{-nu^2/4}):

    inf_states int f^2 <:T_00:(t,0)> dt  =  E_Q / (2 pi^2),

    E_Q = inf spec[ a^dag K a + (1/2)(a^dag K_- a^dag + h.c.) ] on L^2(0, oo),
    K(w, w')   = (w w')^{3/2} Fhat(w - w'),
    K_-(w, w') = (w w')^{3/2} Fhat(w + w'),

so that, with int f''^2 = 3 sqrt(pi)/4,

    R = -32 E_Q / (3 sqrt(pi)).

Two facts drive the certification:

* **Lower bound on R (rigorous, computer-assisted).**  Any normalisable state
  gives an upper bound on the infimum.  We exhibit an explicit finitely-many-mode
  squeezed state supported on the span of phi_n(w) = w^{n+3/2} e^{-w^2/4},
  n < N, and evaluate its energy in interval arithmetic.  Every matrix element
  is a convergent series of Gamma functions with a proved geometric tail bound
  (see :func:`phi_kernel_elements`), so the whole evaluation is two-sided.
* **Upper bound on R (rigorous, cited).**  Fewster & Eveson, PRD 58, 084010
  (1998), Eq. (40), is the theorem ``R <= 1``.  We do **not** improve on it
  here: a tight rigorous lower bound on E_Q is open (the problem is critical,
  ||K^{-1/2} K_- K^{-1/2}|| = 1 and Tr(K_- K^{-1} K_-) = +infinity, so every
  standard perturbative lower bound diverges).  See the draft's "What is not
  proved" section.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mpmath as mp
from mpmath import iv

__all__ = [
    "CERTIFIED_PINNED_RATIO",
    "FEWSTER_EVESON_RATIO_UPPER_BOUND",
    "CertifiedEnclosure",
    "KernelElements",
    "certified_enclosure",
    "certified_lower_bound",
    "phi_gram",
    "phi_kernel_elements",
    "ratio_from_energy",
    "trial_state_covariance",
]

#: The value archived by the momentum-space route at its largest grid
#: (``s6b_momentum_4d_worldline.json::C_4d_over_FE``); the MANIFEST headline
#: rounds it to 0.4574904 with a stated momentum-grid uncertainty of 2e-6.
CERTIFIED_PINNED_RATIO = 0.45749073394875556

#: Fewster & Eveson, PRD 58, 084010 (1998), Eq. (40): the bound they prove is
#: int f^2 <:T_00:> dt >= -(1/(16 pi^2)) int f''^2, i.e. C_sharp(f) <= C_FE for
#: every admissible f.  That is the only rigorous upper bound we have here.
FEWSTER_EVESON_RATIO_UPPER_BOUND = 1.0


def ratio_from_energy(e_q):
    """R = -32 E_Q / (3 sqrt(pi)) for the tau0 = 1 Gaussian (see module docstring).

    E_Q is the infimum of the *radial unit problem* (kernels K, K_- above); the
    physical infimum is E_Q/(2 pi^2) and the Fewster-Eveson denominator is
    (1/(16 pi^2)) * 3 sqrt(pi)/4.  The ratio is invariant under tau0 (the
    dilation w -> w/tau0 maps the kernel to tau0^{-3} times itself, and
    int f''^2 scales the same way).
    """
    return -32 * e_q / (3 * mp.sqrt(mp.pi))


# --------------------------------------------------------------------------
# matrix elements in the basis phi_n(w) = w^{n+3/2} e^{-w^2/4}
# --------------------------------------------------------------------------


@dataclass
class KernelElements:
    """Interval enclosures of the Gram and kernel matrices in the phi basis."""

    n_basis: int
    n_terms: int
    dps: int
    gram: list = field(repr=False)          # list[list[iv.mpf]]
    a_elem: list = field(repr=False)        # <phi_m, K phi_n>
    b_elem: list = field(repr=False)        # <phi_m, K_- phi_n>
    max_rel_width: float = 0.0   # relative half-width; may underflow float to 0


def _gamma_half_table(jmax, dps):
    """gh[j] = Gamma(j/2) for j = 1 .. jmax, at the working precision."""
    mp.mp.dps = dps
    return [mp.mpf(0)] + [mp.gamma(mp.mpf(j) / 2) for j in range(1, jmax + 1)]


def phi_gram(n_basis, dps=200):
    """<phi_m, phi_n> = 2^{(m+n)/2 + 1} Gamma((m+n+4)/2), exactly.

    phi_n(w) = w^{n + 3/2} e^{-w^2/4} on (0, oo); substituting u = w^2/2 in
    int_0^oo w^{m+n+3} e^{-w^2/2} dw gives the closed form directly.
    """
    mp.mp.dps = dps
    gh = _gamma_half_table(2 * n_basis + 6, dps)
    sq2 = mp.sqrt(2)
    pw = [sq2 ** i for i in range(2 * n_basis + 6)]
    G = [[mp.mpf(0)] * n_basis for _ in range(n_basis)]
    for m in range(n_basis):
        for n in range(n_basis):
            G[m][n] = pw[m + n + 2] * gh[m + n + 4]
    return G


def _series_terms_needed(n_basis, dps):
    """Smallest K with a proved tail ratio q <= 0.6 and tail below 10^-(dps-8).

    t_k = Gamma((m+k+4)/2) Gamma((n+k+4)/2) / k!  with m, n < n_basis.  By
    Wendel's inequality Gamma(x + 1/2) <= sqrt(x) Gamma(x) (x > 0),

        t_{k+1}/t_k <= sqrt((m+k+4)(n+k+4)) / (2(k+1)) <= (N+k+3)/(2(k+1)),

    which decreases in k, so q := (N+K+3)/(2(K+1)) bounds every later ratio and
    the tail beyond K is at most t_K q/(1 - q).
    """
    mp.mp.dps = dps + 20
    N = int(n_basis)
    m = n = N - 1
    K = max(8, int(math.ceil(1.25 * N + 4)))
    target = mp.mpf(10) ** (-(dps - 8))
    lt0 = mp.loggamma(mp.mpf(m + 4) / 2) + mp.loggamma(mp.mpf(n + 4) / 2)
    while K <= 40000:
        q = mp.mpf(N + K + 3) / (2 * (K + 1))
        if q < 1:
            lt = (mp.loggamma(mp.mpf(m + K + 4) / 2)
                  + mp.loggamma(mp.mpf(n + K + 4) / 2) - mp.loggamma(K + 1))
            if mp.exp(lt - lt0) * q / (1 - q) < target:
                mp.mp.dps = dps
                return int(K)
        K = int(K * 1.35) + 8
    raise RuntimeError("series tail never reached the target")


def _iv_around(mid, abs_err, pad):
    """Interval [mid - e, mid + e] with e = abs_err + pad*|mid| (pad >= 64 ulp)."""
    e = abs_err + abs(mid) * pad
    return iv.mpf([mid - e, mid + e])


def phi_kernel_elements(n_basis, dps=200, n_terms=None):
    """Interval enclosures of <phi_m, K phi_n> and <phi_m, K_- phi_n>.

    With Fhat(nu) = sqrt(pi) e^{-nu^2/4} and phi_n(w) = w^{n+3/2} e^{-w^2/4},

        <phi_m, K phi_n>   = sqrt(pi) 2^{(m+n)/2+2} sum_k t_k / 1,
        <phi_m, K_- phi_n> = sqrt(pi) 2^{(m+n)/2+2} sum_k (-1)^k t_k,
        t_k = Gamma((m+k+4)/2) Gamma((n+k+4)/2) / k!,

    from expanding e^{+- w w'/2} in the identity
    Fhat(w -+ w') = sqrt(pi) e^{-w^2/4} e^{-w'^2/4} e^{+- w w'/2}.  The returned
    intervals contain the exact values: the series tail is bounded by the
    geometric estimate of :func:`_series_terms_needed` and the accumulated
    mpmath rounding error by 4(K+3) 2^{-prec} sum_k |t_k|.

    """
    N = int(n_basis)
    mp.mp.dps = dps
    iv.dps = dps
    K = int(n_terms) if n_terms is not None else _series_terms_needed(N, dps)
    gh = _gamma_half_table(2 * N + K + 12, dps)
    inv = [mp.mpf(1)]
    for k in range(1, K + 2):
        inv.append(inv[-1] / k)
    sq2 = mp.sqrt(2)
    pw = [sq2 ** i for i in range(2 * N + 8)]
    sp = mp.sqrt(mp.pi)
    q = mp.mpf(N + K + 3) / (2 * (K + 1))
    if q >= 1:
        raise ValueError("n_terms too small for a proved tail bound")
    tail_factor = q / (1 - q)
    round_factor = 4 * (K + 3) * mp.mpf(2) ** (-mp.mp.prec)
    pad = 64 * mp.mpf(2) ** (-mp.mp.prec)

    A = [[None] * N for _ in range(N)]
    B = [[None] * N for _ in range(N)]
    max_w = mp.mpf(0)
    for m in range(N):
        for n in range(m, N):
            sA = mp.mpf(0)
            sB = mp.mpf(0)
            for k in range(K + 1):
                t = gh[m + k + 4] * gh[n + k + 4] * inv[k]
                sA += t
                sB += -t if (k & 1) else t
            tK = gh[m + K + 4] * gh[n + K + 4] * inv[K]
            err = tK * tail_factor + round_factor * sA
            c = sp * pw[m + n + 4]
            ea = _iv_around(c * sA, c * err, pad)
            eb = _iv_around(c * sB, c * err, pad)
            A[m][n] = A[n][m] = ea
            B[m][n] = B[n][m] = eb
            w = (err / abs(sA)) if sA != 0 else mp.mpf(0)
            max_w = max(max_w, w)
    G_mid = phi_gram(N, dps)
    G = [[_iv_around(G_mid[m][n], mp.mpf(0), pad) for n in range(N)]
         for m in range(N)]
    return KernelElements(n_basis=N, n_terms=K, dps=dps, gram=G, a_elem=A,
                          b_elem=B, max_rel_width=float(max_w))


# --------------------------------------------------------------------------
# interval linear algebra (only what the certificate needs)
# --------------------------------------------------------------------------


def _chol_iv(M, n):
    """Interval Cholesky of a symmetric interval matrix.

    Returns the lower factor as list[list[iv.mpf]].  Raises ``ValueError`` if
    any pivot interval fails to be strictly positive -- in which case the
    certificate is simply not granted (the routine never *asserts* positivity
    it has not verified).  A successful run proves that every symmetric matrix
    contained in ``M`` is positive definite, because the pivots are the
    leading-minor ratios and the arithmetic is outward-rounded.
    """
    L = [[iv.mpf(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = M[i][j]
            for k in range(j):
                s = s - L[i][k] * L[j][k]
            if i == j:
                if not (s.a > 0):
                    raise ValueError(f"interval Cholesky pivot {i} not > 0: {s}")
                L[i][i] = iv.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _tri_inv_iv(L, n):
    """Interval inverse of a lower-triangular interval matrix with positive diagonal."""
    X = [[iv.mpf(0)] * n for _ in range(n)]
    for j in range(n):
        X[j][j] = iv.mpf(1) / L[j][j]
        for i in range(j + 1, n):
            s = iv.mpf(0)
            for k in range(j, i):
                s = s - L[i][k] * X[k][j]
            X[i][j] = s / L[i][i]
    return X


def _congruence_iv(Li, A, n):
    """Li * A * Li^T for lower-triangular Li (interval)."""
    T = [[iv.mpf(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = iv.mpf(0)
            for k in range(i + 1):
                s = s + Li[i][k] * A[k][j]
            T[i][j] = s
    R = [[iv.mpf(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = iv.mpf(0)
            for k in range(j + 1):
                s = s + T[i][k] * Li[j][k]
            R[i][j] = s
    return R


def _mid(x):
    """Midpoint of an mpmath interval, as an ``mp.mpf``."""
    return (mp.mpf(x.a) + mp.mpf(x.b)) / 2


# --------------------------------------------------------------------------
# the trial state
# --------------------------------------------------------------------------


def _symfun(Mat, fun):
    E, Q = mp.eigsy(Mat)
    n = Mat.rows
    D = mp.matrix(n, n)
    for i in range(n):
        D[i, i] = fun(E[i])
    return Q * D * Q.T


def trial_state_covariance(Ah, Bh):
    """(N, M) of the Gaussian ground state of the N-mode problem (Ah, Bh).

    ``Ah``, ``Bh`` are real symmetric ``mp.matrix`` in an orthonormal basis.
    Writing H = a^dag Ah a + (1/2)(a^dag Bh a^dag + h.c.) in quadratures gives
    H = (1/2) x.(Ah+Bh).x + (1/2) p.(Ah-Bh).p - (1/2) Tr Ah, whose ground state
    has <xx> = (1/2) P^{1/2}(P^{1/2} X P^{1/2})^{-1/2} P^{1/2} and
    <pp> = (1/2) P^{-1/2}(P^{1/2} X P^{1/2})^{1/2} P^{-1/2} with P = Ah - Bh,
    X = Ah + Bh; then N = (<xx> + <pp> - I)/2 and M = (<xx> - <pp>)/2.

    Returned only as a *trial*: nothing downstream trusts that it is optimal,
    only that the pair it returns satisfies the state-positivity condition,
    which is verified independently in interval arithmetic.
    """
    n = Ah.rows
    P = Ah - Bh
    X = Ah + Bh
    Ps = _symfun(P, lambda x: mp.sqrt(max(x, mp.mpf(0))))
    Pi = _symfun(P, lambda x: 1 / mp.sqrt(x))
    C = Ps * X * Ps
    C = (C + C.T) / 2
    W = _symfun(C, lambda x: mp.sqrt(max(x, mp.mpf(0))))
    Wi = _symfun(C, lambda x: 1 / mp.sqrt(x))
    xx = (Ps * Wi * Ps) / 2
    pp = (Pi * W * Pi) / 2
    I = mp.eye(n)
    Nh = (xx + pp - I) / 2
    Mh = (xx - pp) / 2
    Nh = (Nh + Nh.T) / 2
    Mh = (Mh + Mh.T) / 2
    return Nh, Mh


# --------------------------------------------------------------------------
# the certificate
# --------------------------------------------------------------------------


@dataclass
class CertifiedEnclosure:
    """Two-sided enclosure of C_4d(gaussian)/C_FE with its provenance."""

    lo: float
    hi: float
    lo_method: str
    hi_method: str
    n_basis: int
    dps: int
    n_terms: int
    energy_upper: float
    energy_interval: tuple
    delta: float
    energy_width_log10: float
    element_rel_width_log10: float
    kernel_scale: float
    b_scale: float
    gram_cond_log10: float
    max_element_rel_width: float
    seconds: float = 0.0

    def contains(self, x, tol=0.0):
        return self.lo - tol <= x <= self.hi + tol


def certified_lower_bound(n_basis=24, dps=None, n_terms=None, delta=None,
                          kernel_scale=1.0, b_scale=1.0, elements=None):
    """Rigorous lower bound on R = C_4d(gaussian)/C_FE from an exhibited state.

    The state is the Gaussian (squeezed) state whose one-particle covariance is
    ``(N + delta I, M)`` in the orthonormal basis obtained by Cholesky
    orthonormalisation of ``{phi_0 .. phi_{n_basis-1}}``, tensored with the
    vacuum on the orthogonal complement, and used identically in the l = 0
    sector and in each of the three l = 1 sectors.  Two things are then
    *verified*, not assumed:

    1. ``[[N, M], [M, I + N]] >= 0`` -- the Robertson-Schroedinger condition, so
       the pair really is the covariance of a state (see the draft, Lemma 3);
    2. the energy ``Tr(A N) + Tr(B M)`` lies in the returned interval.

    Because any state's energy is an upper bound for the infimum, the upper
    endpoint of that interval is a rigorous upper bound for E_Q and hence
    ``-32 E_Q/(3 sqrt(pi))`` is bounded below.

    ``kernel_scale`` (both kernels) and ``b_scale`` (the pair-creation kernel
    only) are the deliberate mutations used by the regression tests: the trial
    state is always built from the *unmutated* matrices, so a mutated run is an
    honest bound for the mutated problem.
    """
    import time

    t0 = time.time()
    N = int(n_basis)
    if dps is None:
        dps = int(60 + 9.6 * N)
    if delta is None:
        delta = mp.mpf(10) ** (-(dps // 3))
    mp.mp.dps = dps
    iv.dps = dps
    ke = elements if elements is not None else phi_kernel_elements(N, dps=dps,
                                                                  n_terms=n_terms)
    if ke.n_basis != N:
        raise ValueError("supplied elements have the wrong basis size")

    L = _chol_iv(ke.gram, N)
    Li = _tri_inv_iv(L, N)
    Ahat = _congruence_iv(Li, ke.a_elem, N)
    Bhat = _congruence_iv(Li, ke.b_elem, N)
    diag = [_mid(L[i][i]) for i in range(N)]
    gcond = float(2 * mp.log10(max(diag) / min(diag)))

    Am = mp.matrix(N, N)
    Bm = mp.matrix(N, N)
    for i in range(N):
        for j in range(N):
            Am[i, j] = _mid(Ahat[i][j])
            Bm[i, j] = _mid(Bhat[i][j])
    Am = (Am + Am.T) / 2
    Bm = (Bm + Bm.T) / 2
    Nh, Mh = trial_state_covariance(Am, Bm)

    dl = mp.mpf(delta)
    for i in range(N):
        Nh[i, i] = Nh[i, i] + dl

    # (1) verify the covariance is a state: [[N, M],[M, I + N]] >= 0
    blk = [[iv.mpf(0)] * (2 * N) for _ in range(2 * N)]
    for i in range(N):
        for j in range(N):
            blk[i][j] = iv.mpf(Nh[i, j])
            blk[i][N + j] = iv.mpf(Mh[i, j])
            blk[N + i][j] = iv.mpf(Mh[i, j])
            blk[N + i][N + j] = iv.mpf(Nh[i, j] + (1 if i == j else 0))
    _chol_iv(blk, 2 * N)   # raises if the certificate is not granted

    # (2) the energy of that state, in interval arithmetic
    ks = iv.mpf(str(kernel_scale))
    bs = ks * iv.mpf(str(b_scale))
    e = iv.mpf(0)
    for i in range(N):
        for j in range(N):
            e = e + ks * Ahat[i][j] * iv.mpf(Nh[j, i]) + bs * Bhat[i][j] * iv.mpf(Mh[j, i])
    e_lo, e_hi = mp.mpf(e.a), mp.mpf(e.b)
    r = ratio_from_energy(iv.mpf([e_lo, e_hi]))
    lo = float(mp.mpf(r.a))
    ewidth = e_hi - e_lo
    ewlog = float(mp.log10(ewidth / abs(e_lo))) if ewidth > 0 else -mp.inf
    elog = (float(mp.log10(mp.mpf(ke.max_rel_width)))
            if ke.max_rel_width > 0 else float("-inf"))
    return CertifiedEnclosure(
        lo=lo, hi=FEWSTER_EVESON_RATIO_UPPER_BOUND,
        lo_method=("exhibited %d-mode squeezed state, interval arithmetic "
                   "(dps=%d, %d series terms)" % (N, dps, ke.n_terms)),
        hi_method="Fewster-Eveson, PRD 58, 084010 (1998), Eq. (40): R <= 1",
        n_basis=N, dps=int(dps), n_terms=int(ke.n_terms),
        energy_upper=float(e_hi), energy_interval=(float(e_lo), float(e_hi)),
        delta=float(delta), energy_width_log10=float(ewlog),
        element_rel_width_log10=elog, kernel_scale=float(kernel_scale),
        b_scale=float(b_scale), gram_cond_log10=gcond,
        max_element_rel_width=ke.max_rel_width, seconds=time.time() - t0)


def certified_enclosure(n_basis=24, **kw):
    """[lo, hi] for C_4d(gaussian)/C_FE.  Thin alias of :func:`certified_lower_bound`.

    The upper endpoint is the Fewster-Eveson theorem, not a computation: the
    width of the enclosure is dominated by the *missing* rigorous lower bound on
    the energy, which is the open problem recorded in the draft.
    """
    return certified_lower_bound(n_basis=n_basis, **kw)
