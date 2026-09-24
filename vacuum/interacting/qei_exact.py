"""The exact lattice QEI infimum of the interacting chain (Layer 7, L5).

The quantity and why it is exact
--------------------------------
With f a sampling function (``vacuum.inequalities.qei.FHandle``, weight
rho = f^2) and h_x the PSD local energy density of site x
(:mod:`vacuum.interacting.observables`), the time-smeared energy along the
static worldline of x is the Hermitian operator

    O_f = int dt f(t)^2 h_x(t),      h_x(t) = U(t)^dag h_x U(t),  U = e^{-iHt},

and E_f(psi) = <psi|O_f|psi> - <Omega|O_f|Omega>. Its infimum over ALL
states of the (truncated-basis) chain is therefore

    E_min = lambda_min(O_f) - <Omega|O_f|Omega>,

the ground energy of O_f taken as a Hamiltonian, normal-ordered against the
interacting vacuum Omega (an eigenstate of H, so <Omega|h_x(t)|Omega> is
t-independent and the subtraction is ||f||^2 <Omega|h_x|Omega>). At
lambda = 0 this is exactly the Williamson problem of
``vacuum.inequalities.qei.qei_minimize`` (O_f is then a quadratic form whose
ground state is the Gaussian "vacuum of O_f"); at lambda > 0 no closed form
exists, and the number produced here replaces the *variational upper bound*
of :mod:`vacuum.interacting.qei_mps` by the infimum itself, up to three
measured approximations: the local Fock cutoff n_max, the Trotter step of
the Heisenberg evolution, and the bond-dimension truncation of the operator
MPO. The DMRG ground state of O_f *is* the extremal state.

Construction (operator-space TEBD, then DMRG)
---------------------------------------------
h_x is a 3-site operator; vectorized (row index p, column index q per site,
|h>> = sum h_{p..,q..} |p..><q..|) it is an MPS on the doubled local space
C^d x C^d.  The Heisenberg step h -> u^dag h u for a two-site Trotter gate
u = exp(-i delta H_bond) is the linear map |h>> -> (u^dag (x) u^T)|h>>,
i.e. u^dag contracted into the row legs and u into the column legs of the
two-site block; so the same two-site TEBD that evolves states evolves the
operator, with the very same bond Hamiltonians ``model.H_bond`` and the same
Suzuki-Trotter sequence (TeNPy's ``TEBDEngine.suzuki_trotter_decomposition``)
as the Schroedinger route of ``qei_mps.smeared_energy_mps`` — applied in
reverse order, U^dag h U = g_1^dag ... g_n^dag h g_n ... g_1.  The operator
MPS is never renormalized, so the truncated operator is the orthogonal
projection of the exact one and the accumulated discarded weight is reported
as ``trunc_op``.

Truncation norm (the knob that makes this affordable).  The plain
Hilbert-Schmidt (Frobenius) norm is the wrong criterion here: the matrix
elements of pi^2, phi^2 and phi^4 between HIGH-occupation states are the
largest and the least relevant, because the ground state of O_f is a mildly
squeezed, low-occupation state (the vacuum carries nbar ~ 0.05 per site), so
Frobenius truncation spends bond dimension on matrix elements no low-energy
state ever sees.  The two-site block is therefore weighted by G = w (x) w
with w_n = weight_decay^(n/2) (:func:`occupation_weights`) before the SVD, so
that the truncation minimizes the error the low-energy sector actually sees;
the weight is a positive on-site *diagonal product* operator, so applying and
undoing it is exact and local, the retained subspace is exact, and only the
choice of discarded direction changes.  ``weight_decay = 1`` recovers plain
Frobenius truncation and is the control.  ``trunc_op`` is reported in that
weighted norm.

Measured at the paper point (m = 0.5, tau0 = 0.75, L = 16, n_max = 6), the
lambda = 0 relative error against the exact Gaussian infimum:

    weight_decay   chi_op = 8   chi_op = 16   chi_op = 24
    1.0 (plain)      -1.75e-1     -3.21e-2          --
    0.5              -4.17e-2     -6.58e-3          --
    0.25             -1.83e-2     -1.27e-2      -1.69e-2

Two things to read off.  (i) The weighting is worth roughly a factor of two
in chi_op, and at chi_op = 8 a factor of ten in the error -- without it this
study is unaffordable.  (ii) The weight is itself a bias: at decay = 0.25 the
error stops improving with chi_op (it is -1.27e-2 at 16 and -1.69e-2 at 24
with an unchanged trunc_op = 1.9e-4), because the discarded high-occupation
content is no longer negligible for the genuinely squeezed extremal state.
**decay = 0.5 is the better operating point** -- it reaches -6.6e-3 at
chi_op = 16, the best of the scan -- and decay should be treated as a
convergence axis alongside chi_op, not as a fixed constant.  The default 0.25
is the value the archived run used; it is safe (it over-controls truncation)
but it costs about a factor of two in the calibration band.

Signs (why the lambda = 0 anchor is a calibration, not a formality).  The
operator truncation LOWERS the computed infimum -- discarding operator weight
can manufacture states below the true minimum, and at chi_op = 8 unweighted
it puts the lambda = 0 answer 17% BELOW the exact free infimum -- while the
Fock cutoff n_max RAISES it (Rayleigh-Ritz on nested subspaces).  The two
dominant errors therefore pull in opposite directions, so the residual at
lambda = 0, where one Williamson problem gives the answer exactly, is a
genuine measurement of the total systematic, and it is quoted as a band on
the lambda > 0 numbers.

Gates act only on bonds inside a light-cone window |j - x| <= ceil(t) +
window_pad (lattice group velocity <= 1): outside it the operator is the
identity and u^dag 1 u = 1 exactly, so the restriction only drops the
Lieb-Robinson skin of the window edge; window_pad=None uses the whole chain
and the notebook measures the pad-independence.

Quadrature.  h(-t) = h(t)^T for the real symmetric H and h_x (the
Trotterized U is a symmetric matrix too), so O_f = Q + Q^T with
Q = int_0^T f^2 h(t) dt, and the time grid is the TEBD grid t_k = k dt,
k = 0..n, n dt = t_hi.  Q is accumulated by the trapezoid rule (weight 1/2
at t = 0 and t = T).  For a Gaussian f the integrand has all derivatives
~e^{-T^2/tau0^2} at the ends and the trapezoid rule is spectrally accurate
(Euler-Maclaurin; aliasing at frequency 2 pi/dt against the ~2 omega_max
content of h(t) is e^{-(2 pi/dt - 2 omega_max)^2 tau0^2/4} ~ e^{-100} at
dt = 0.25); the quadrature and Trotter errors are measured together by
the dt-halving change of E_min, and separately at lambda = 0 by the
Gaussian anchor.  The tails beyond t_hi are dropped exactly as
``sampling_operator`` drops them (same FHandle).

Real symmetric MPO.  Q is complex (h(t) is Hermitian, not real); Q + Q^T is
real symmetric.  A complex MPS with tensors T_j is turned into a real MPS
of twice the bond dimension by the ring isomorphism z -> [[Re z, -Im z],
[Im z, Re z]] applied blockwise (boundary rows [Re, -Im] and [Re; Im]),
which represents Re(T_1 ... T_L) exactly; the transpose is the p <-> q
swap.  The result is compressed (real SVD, lossless threshold), converted
to a TeNPy MPO with trivial IdL/IdR, and handed to ``TwoSiteDMRGEngine``
warm-started from Omega; a real MPO and a real start keep the DMRG real,
so the extremal state can be fed straight back to the Schroedinger-picture
``smeared_energy_mps`` — the independent implementation of the same
number, which is the cross-check the notebook records at every lambda.

Errors reported (``SmearedOperator`` / ``QEIExactResult``)
----------------------------------------------------------
trunc_op (operator TEBD, accumulated relative discarded weight),
trunc_acc (accumulator compression), trunc_real (final real compression),
the largest bond dimensions reached, the DMRG record (sweeps, Delta E,
truncation, chi), the MPO variance <O^2> - <O>^2 of the extremal state (an
eigenstate has zero), the drift of <Omega|h(t)|Omega> along the trajectory
(zero for an exact eigenstate under exact evolution), and the static
reference ||f||^2 <Omega|h_x|Omega> beside the MPO reference
<Omega|O_f|Omega> (they differ by the Trotter drift the subtraction
cancels).  n_max, dt and chi remain explicit convergence axes.

TeNPy is imported lazily via :func:`vacuum.interacting.phi4._tenpy`; the
operator MPS itself is plain numpy.  Functions are pure over their MPS
arguments.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np

from vacuum.core import chain_propagator, ground_state_cov
from vacuum.inequalities.qei import (
    FHandle,
    _gauss_legendre_grid,
    chain_energy_density_form,
    embed_extremal,
    qei_bound_2d,
    qei_minimize,
    sampling_operator,
)

from .phi4 import (
    GroundState,
    _tadpole_D,
    _tenpy,
    free_coupling_matrix,
    hartree_mass_squared,
    local_boson_ops,
)

__all__ = [
    "occupation_weights",
    "local_energy_operator_dense",
    "local_energy_infimum_dense",
    "SmearedOperator",
    "heisenberg_smeared_operator",
    "QEIExactResult",
    "qei_exact_mps",
    "free_exact_infimum",
    "first_order_infimum_slope",
]


# --------------------------------------------------------------------------
# Dense local energy density (the t = 0 operator)
# --------------------------------------------------------------------------


def _model_params(model):
    o = model.options
    return {"L": int(model.lat.N_sites), "m": float(o["m"]), "lam": float(o["lam"]),
            "n_max": int(o["n_max"]), "omega": float(o["omega"]), "bc": str(o["bc"])}


def local_energy_operator_dense(n_max, omega, m, lam, site, L, bc="dirichlet"):
    """Dense h_site on its support sites, in the projected local basis.

    Returns (sites, H) with `sites` the ascending support (site-1, site,
    site+1 clipped to the chain) and H the (d^k, d^k) real symmetric matrix
    of h_site = (1/2) Psq + (1/2) m^2 Xsq + (lam/4!) X4 + (1/4) sum over
    adjacent bonds (Xsq_j + Xsq_k - 2 X_j X_k) [+ (1/2) Xsq per Dirichlet
    wall bond], i.e. exactly the operator ``observables.local_energy_density``
    measures (same projected 'Xsq', 'Psq', 'X4', 'X').
    """
    ops = local_boson_ops(n_max, omega)
    d = n_max + 1
    L = int(L)
    x = int(site)
    if not 0 <= x < L:
        raise ValueError(f"site must be in [0, {L}), got {x}")
    if bc not in ("dirichlet", "open"):
        raise ValueError(f"bc must be 'dirichlet' or 'open', got {bc!r}")
    lo, hi = max(0, x - 1), min(L - 1, x + 1)
    sites = list(range(lo, hi + 1))
    k = len(sites)
    eye = np.eye(d)

    def embed(mats):
        out = np.array([[1.0]])
        for j in sites:
            out = np.kron(out, mats.get(j, eye))
        return out

    H = embed({x: 0.5 * ops["Psq"] + 0.5 * m * m * ops["Xsq"] + (float(lam) / 24.0) * ops["X4"]})
    for j in (x - 1, x):
        if 0 <= j < L - 1:
            H += 0.25 * (embed({j: ops["Xsq"]}) + embed({j + 1: ops["Xsq"]})
                         - 2.0 * embed({j: ops["X"], j + 1: ops["X"]}))
    if bc == "dirichlet":
        if x == 0:
            H += 0.5 * embed({0: ops["Xsq"]})
        if x == L - 1:
            H += 0.5 * embed({L - 1: ops["Xsq"]})
    return sites, 0.5 * (H + H.T)


def local_energy_infimum_dense(n_max, omega, m, lam, site, L, bc="dirichlet"):
    """lambda_min(h_site) over the truncated chain (h_site acts on <= 3 sites).

    The tiny-tau limit of the smeared infimum: O_f / ||f||^2 -> h_site as
    tau0 -> 0, so E_min / ||f||^2 -> lambda_min(h_site) - <Omega|h_site|Omega>.
    """
    _, H = local_energy_operator_dense(n_max, omega, m, lam, site, L, bc)
    return float(np.linalg.eigvalsh(H)[0])


# --------------------------------------------------------------------------
# Operator MPS in the doubled space (plain numpy)
# --------------------------------------------------------------------------


def _truncated_svd(M, chi_max, svd_min, sketch=None, n_random=16, power=1, rng=None,
                   dense_limit=600):
    """SVD of M with truncation; sketched (randomized, one power iteration)
    beyond `dense_limit` rows/cols.

    Keeps s_i >= svd_min * s_0, at most chi_max values.  Returns
    (U, s, Vh, discarded, total) with `discarded` = ||M||_F^2 - sum_kept s^2
    computed exactly (so a sketch can only over-report the discarded
    weight, never hide it) and `total` = ||M||_F^2.
    """
    total = float(np.sum(np.abs(M) ** 2))
    n, p = M.shape
    if min(n, p) <= dense_limit or chi_max + n_random >= min(n, p):
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
    else:
        rng = np.random.default_rng(0) if rng is None else rng
        cols = []
        if sketch is not None and sketch.shape[0] == p:
            cols.append(sketch)
        cols.append(rng.standard_normal((p, n_random)).astype(M.dtype))
        Om = np.concatenate(cols, axis=1)
        k = min(chi_max + n_random, Om.shape[1], min(n, p))
        Om = Om[:, :k] if Om.shape[1] > k else Om
        Y = M @ Om
        Q, _ = np.linalg.qr(Y)
        for _ in range(int(power)):
            Z = M.conj().T @ Q
            Qz, _ = np.linalg.qr(Z)
            Y = M @ Qz
            Q, _ = np.linalg.qr(Y)
        B = Q.conj().T @ M
        Ub, s, Vh = np.linalg.svd(B, full_matrices=False)
        U = Q @ Ub
    if s.size == 0 or s[0] <= 0.0:
        keep = 1
    else:
        keep = int(np.sum(s >= svd_min * s[0]))
        keep = max(1, min(keep, int(chi_max)))
    U, s, Vh = U[:, :keep], s[:keep], Vh[:keep, :]
    discarded = max(0.0, total - float(np.sum(s * s)))
    return U, s, Vh, discarded, total


class _OpMPS:
    """Unnormalized MPS on the doubled space, tensors (a, p, q, b), B-form
    with Schmidt vectors S[j] on the bond left of site j (S[0] = S[L] = [1])."""

    def __init__(self, tensors, S=None):
        self.T = [np.asarray(t) for t in tensors]
        self.L = len(self.T)
        if S is None:
            S = [np.ones(1)] * (self.L + 1)
        self.S = [np.asarray(s, dtype=float) for s in S]

    @property
    def chi(self):
        return [t.shape[0] for t in self.T[1:]]

    @property
    def d(self):
        return self.T[0].shape[1]

    def norm2(self):
        # B-form: the left S carries the norm (bond 0 has S = [1] and T[0] is
        # right-isometric only up to truncation; contract explicitly)
        E = np.ones((1, 1))
        for t in self.T:
            E = np.einsum("ab,apqc,bpqd->cd", E, t, t.conj())
        return float(np.real(E[0, 0]))

    def copy(self):
        return _OpMPS([t.copy() for t in self.T], [s.copy() for s in self.S])

    def scale(self, alpha):
        out = self.copy()
        out.T[0] = out.T[0] * alpha
        out.S = [s * abs(alpha) for s in out.S]
        out.S[0] = np.ones(1)
        return out

    def transpose(self):
        """|h^T>>: swap p and q on every site."""
        return _OpMPS([np.transpose(t, (0, 2, 1, 3)) for t in self.T], self.S)

    def to_dense(self):
        """Dense (d^L, d^L) matrix (tests only)."""
        d = self.d
        M = self.T[0]
        for t in self.T[1:]:
            M = np.tensordot(M, t, axes=([M.ndim - 1], [0]))
        # legs: a, p1, q1, p2, q2, ..., b
        M = M.reshape(M.shape[1:-1])
        perm = list(range(0, 2 * self.L, 2)) + list(range(1, 2 * self.L, 2))
        return np.transpose(M, perm).reshape(d ** self.L, d ** self.L)

    # -- canonical forms -------------------------------------------------
    def canonicalize(self, chi_max=None, svd_min=0.0):
        """Right-canonical B-form with Schmidt vectors; optional truncation.

        Sweeps: (1) right-to-left LQ (right-canonical), (2) left-to-right
        SVD with truncation (left-canonical A-form, truncated ranks),
        (3) right-to-left SVD (B-form + S, no truncation).  Returns the
        relative discarded weight sum_discarded s^2 / ||psi||^2 of sweep (2).
        """
        L = self.L
        T = self.T
        for j in range(L - 1, 0, -1):
            a, d, _, b = T[j].shape
            M = T[j].reshape(a, d * d * b)
            Q, R = np.linalg.qr(M.T)        # M^T = Q R -> M = R^T Q^T
            T[j] = (Q.T).reshape(-1, d, d, b)
            T[j - 1] = np.tensordot(T[j - 1], R.T, axes=([3], [0]))
        total2 = float(np.sum(np.abs(T[0]) ** 2))
        discarded = 0.0
        for j in range(L - 1):
            a, d, _, b = T[j].shape
            M = T[j].reshape(a * d * d, b)
            U, s, Vh = np.linalg.svd(M, full_matrices=False)
            keep = s.size
            if s.size and s[0] > 0.0:
                keep = max(1, int(np.sum(s >= svd_min * s[0])))
            if chi_max is not None:
                keep = max(1, min(keep, int(chi_max)))
            discarded += float(np.sum(s[keep:] ** 2))
            U, s, Vh = U[:, :keep], s[:keep], Vh[:keep, :]
            T[j] = U.reshape(a, d, d, keep)
            T[j + 1] = np.tensordot((s[:, None] * Vh), T[j + 1], axes=([1], [0]))
        S = [np.ones(1)] * (L + 1)
        for j in range(L - 1, 0, -1):
            a, d, _, b = T[j].shape
            M = T[j].reshape(a, d * d * b)
            U, s, Vh = np.linalg.svd(M, full_matrices=False)
            T[j] = Vh.reshape(-1, d, d, b)
            T[j - 1] = np.tensordot(T[j - 1], U * s[None, :], axes=([3], [0]))
            S[j] = s
        S[0] = np.ones(1)
        S[L] = np.ones(1)
        self.T = T
        self.S = S
        return discarded / total2 if total2 > 0 else 0.0

    def add(self, other, alpha=1.0, beta=1.0):
        """alpha * self + beta * other (block-diagonal direct sum; not canonical)."""
        L = self.L
        out = []
        for j in range(L):
            A, B = self.T[j], other.T[j]
            if j == 0:
                A = A * alpha
                B = B * beta
            a1, d, _, b1 = A.shape
            a2, _, _, b2 = B.shape
            dt = np.result_type(A, B)
            if L == 1:
                out.append(A + B)
                continue
            if j == 0:
                t = np.zeros((1, d, d, b1 + b2), dtype=dt)
                t[:, :, :, :b1] = A
                t[:, :, :, b1:] = B
            elif j == L - 1:
                t = np.zeros((a1 + a2, d, d, 1), dtype=dt)
                t[:a1] = A
                t[a1:] = B
            else:
                t = np.zeros((a1 + a2, d, d, b1 + b2), dtype=dt)
                t[:a1, :, :, :b1] = A
                t[a1:, :, :, b1:] = B
            out.append(t)
        return _OpMPS(out)

    # -- the two-site Heisenberg update ----------------------------------
    def update_bond(self, j, u, chi_max, svd_min, rng=None, w=None, dense_limit=600):
        """Sites (j, j+1): h -> u^dag h u on that pair, SVD-truncate.

        u: dense (d^2, d^2) Schroedinger gate on (p_j, p_{j+1}); w: optional
        (d,) positive on-site weight defining the norm the truncation
        minimizes (see :func:`occupation_weights`); dense_limit: blocks with
        more rows AND columns than this are SVD'd by the randomized sketch
        of :func:`_truncated_svd` (chi_max + 16 columns, one power
        iteration) -- pass ``np.inf`` for the exact dense SVD, which is the
        control that isolates the sketch's error.  Returns (discarded,
        total): the discarded weight of the two-site block and its total
        weight, in that norm, absolute and NOT renormalized.
        """
        A, B = self.T[j], self.T[j + 1]
        a, d, _, _ = A.shape
        b = B.shape[3]
        d2 = d * d
        # C[a, p, q, p', q', b] without the left S
        C = np.tensordot(A, B, axes=([3], [0]))               # a p q p' q' b
        C = np.transpose(C, (0, 1, 3, 2, 4, 5)).reshape(a, d2, d2, b)  # a (pp') (qq') b
        C = np.tensordot(u.conj(), C, axes=([0], [1]))       # (pp')_new a (qq') b   [u^dag on rows]
        C = np.tensordot(C, u, axes=([2], [0]))               # (pp') a b (qq')_new   [u on cols]
        C = np.transpose(C, (1, 0, 3, 2))                    # a (pp') (qq') b
        C = C.reshape(a, d, d, d, d, b)                       # a p p' q q' b
        C = np.transpose(C, (0, 1, 3, 2, 4, 5)).reshape(a * d2, d2 * b)  # (a p q), (p' q' b)
        theta = (self.S[j][:, None] * C.reshape(a, d2 * d2 * b)).reshape(a * d2, d2 * b)
        if w is None:
            sketch = B.reshape(B.shape[0], d2 * b).conj().T   # the current right basis
            U, s, Vh, disc, total = _truncated_svd(theta, chi_max, svd_min, sketch=sketch, rng=rng,
                                                   dense_limit=dense_limit)
            chi_new = s.size
            self.T[j + 1] = Vh.reshape(chi_new, d, d, b)
            self.T[j] = (C @ Vh.conj().T).reshape(a, d, d, chi_new)
            self.S[j + 1] = s
            return disc, total
        # Weighted norm: G = w (x) w on each site's (p, q) pair is a positive
        # diagonal product operator, so theta -> G theta G is exact and local.
        # The SVD then minimizes || G (theta - theta_trunc) G ||_F, i.e. the
        # error the LOW-occupation sector actually sees; the right isometry it
        # returns is un-weighted back before it is stored, so the retained
        # subspace is exact and only the discarded direction is chosen
        # differently.  Reported disc/total are in this weighted norm.
        gw = np.outer(w, w).ravel()                            # (d2,) weight on (p, q)
        gl = np.repeat(gw[None, :], a, axis=0).ravel()          # (a d2,)
        gr = np.repeat(gw[:, None], b, axis=1).ravel()          # (d2 b,)
        theta_w = (theta * gl[:, None]) * gr[None, :]
        _U, _s, Vh_w, disc, total = _truncated_svd(theta_w, chi_max, svd_min, rng=rng,
                                                   dense_limit=dense_limit)
        # The kept RIGHT SUBSPACE is what the weighted SVD chooses; un-weight it
        # and re-orthonormalize (the un-weighted rows are no longer isometric).
        Vh = Vh_w / gr[None, :]
        Q, _R = np.linalg.qr(Vh.conj().T)                      # span(Q) = span(Vh^dag)
        chi_new = Q.shape[1]
        self.T[j + 1] = Q.conj().T.reshape(chi_new, d, d, b)
        self.T[j] = (C @ Q).reshape(a, d, d, chi_new)
        # true Schmidt values of the projected block (theta Q is (a d2, chi_new))
        self.S[j + 1] = np.linalg.svd(theta @ Q, compute_uv=False)
        return disc, total


def occupation_weights(n_max, decay):
    """w_n = decay^(n/2) on the local Fock basis (the truncation norm's weight).

    The Frobenius norm treats every matrix element of h(t) alike, but the
    matrix elements of pi^2, phi^2 and phi^4 between high-occupation states
    are the largest and the *least* relevant: the ground state of O_f is a
    mildly squeezed, low-occupation state (the interacting vacuum has
    nbar ~ 0.05 per site).  Truncating h(t) in the plain Frobenius norm
    therefore spends bond dimension on matrix elements no low-energy state
    ever sees.  Weighting the two-site block by G = w (x) w with a decaying
    w before the SVD makes the truncation minimize the error the low-energy
    sector sees instead; the weight is a positive on-site *diagonal product*
    operator, so applying and undoing it is exact and local, and only the
    choice of discarded direction changes.  ``decay = 1`` recovers the plain
    Frobenius truncation.  The value is a reported convergence axis: the
    lambda = 0 anchor and the chi_op ladder measure what it buys.
    """
    decay = float(decay)
    if not 0.0 < decay <= 1.0:
        raise ValueError(f"decay must be in (0, 1], got {decay}")
    return decay ** (0.5 * np.arange(int(n_max) + 1, dtype=float))


def _identity_site(d, dtype=float):
    """The identity operator on one site (unnormalized: h_3 (x) 1 (x) ... exactly)."""
    return np.eye(d).reshape(1, d, d, 1).astype(dtype)


def _operator_mps_from_dense(sites, H, L, d):
    """h_3 (x) identity elsewhere as an unnormalized :class:`_OpMPS`."""
    k = len(sites)
    T = H.reshape([d] * (2 * k))
    # (p1..pk, q1..qk) -> (p1 q1 p2 q2 ...)
    perm = []
    for i in range(k):
        perm += [i, k + i]
    T = np.transpose(T, perm).reshape(1, *([d * d] * k), 1)
    tensors = {}
    rest = T
    for i, site in enumerate(sites):
        a = rest.shape[0]
        if i == k - 1:
            tensors[site] = rest.reshape(a, d, d, 1)
            break
        M = rest.reshape(a * d * d, -1)
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        keep = int(np.sum(s > 1e-14 * s[0]))
        tensors[site] = U[:, :keep].reshape(a, d, d, keep)
        rest = (s[:keep, None] * Vh[:keep]).reshape(keep, *([d * d] * (k - i - 1)), 1)
    out = []
    for j in range(L):
        out.append(tensors[j] if j in tensors else _identity_site(d))
    return _OpMPS(out)


# --------------------------------------------------------------------------
# Heisenberg TEBD of h_x and the accumulated O_f
# --------------------------------------------------------------------------


def _bond_hamiltonians(model):
    """Dense (d^2, d^2) H_bond[i] acting on sites (i-1, i), i = 1..L-1."""
    L = int(model.lat.N_sites)
    out = {}
    for i in range(1, L):
        Hb = model.H_bond[i]
        Hb = Hb.copy()
        Hb.itranspose(["p0", "p1", "p0*", "p1*"])
        arr = np.real(Hb.to_ndarray())
        d = arr.shape[0]
        out[i] = 0.5 * (arr.reshape(d * d, d * d) + arr.reshape(d * d, d * d).T)
    return out


def _gate(Hb, delta):
    """exp(-i delta H_bond) for a real symmetric bond Hamiltonian (eigh)."""
    w, V = np.linalg.eigh(Hb)
    return (V * np.exp(-1j * float(delta) * w)[None, :]) @ V.T


def _time_grid(f: FHandle, dt):
    if abs(f.t_lo + f.t_hi) > 1e-12 * max(1.0, abs(f.t_hi)):
        raise ValueError("sampling function must be centred at t = 0 (t_lo = -t_hi)")
    n = max(2, int(math.ceil(f.t_hi / float(dt) - 1e-9)))
    dt_eff = f.t_hi / n
    times = dt_eff * np.arange(n + 1)
    w = np.full(n + 1, dt_eff)
    w[0] *= 0.5
    w[-1] *= 0.5
    return times, w, dt_eff


class SmearedOperator(NamedTuple):
    """O_f as a real symmetric MPO on the chain, with its error record."""

    mpo: object             # TeNPy MPO (real), IdL/IdR trivial
    tensors: list           # real numpy tensors (a, p, q, b), O_f = scale * prod
    scale: float
    times: np.ndarray       # TEBD grid t_k >= 0
    weights: np.ndarray     # trapezoid weights on [0, T] (Q = sum w_k f_k^2 h_k)
    trunc_op: float         # accumulated relative discarded weight of h(t)
    trunc_acc: float        # accumulated relative discarded weight of the accumulator
    trunc_real: float       # relative discarded weight of the final real compression
    chi_op: int             # largest bond dimension of h(t)
    chi_acc: int            # largest bond dimension of the accumulator
    chi_mpo: int            # bond dimension of the final MPO
    h_ref_trace: np.ndarray  # <psi_ref| h(t_k) |psi_ref> (empty if no psi_ref)
    norm_ratio: float       # ||h(T)||_F / ||h(0)||_F (1 - truncation loss)
    meta: dict


def _mpo_from_tensors(sites, tensors, scale=1.0):
    t = _tenpy()
    Ws = []
    L = len(tensors)
    for j, T in enumerate(tensors):
        W = np.asarray(T)
        if j == 0:
            W = W * scale
        # (a, p, q, b) -> (wL, wR, p, p*) with p* = q (matrix element <p|O|q>)
        W = np.transpose(W, (0, 3, 1, 2))
        Ws.append(t.npc.Array.from_ndarray_trivial(np.ascontiguousarray(W),
                                                   labels=["wL", "wR", "p", "p*"]))
    from tenpy.networks.mpo import MPO
    return MPO(sites, Ws, bc="finite", IdL=[0] + [None] * L,
               IdR=[None] * L + [tensors[-1].shape[3] - 1])


def _realify(Q: _OpMPS):
    """Real MPS of Re(prod T_j) via z -> [[Re, -Im], [Im, Re]] blockwise."""
    L = Q.L
    out = []
    for j, T in enumerate(Q.T):
        R, I = np.real(T), np.imag(T)
        a, d, _, b = T.shape
        if L == 1:
            out.append(R)
            continue
        if j == 0:
            t = np.concatenate([R, -I], axis=3)
        elif j == L - 1:
            t = np.concatenate([R, I], axis=0)
        else:
            t = np.zeros((2 * a, d, d, 2 * b))
            t[:a, :, :, :b] = R
            t[:a, :, :, b:] = -I
            t[a:, :, :, :b] = I
            t[a:, :, :, b:] = R
        out.append(t)
    return _OpMPS(out)


def heisenberg_smeared_operator(model, site, f: FHandle, *, dt=0.25, order=4,
                                chi_op=24, svd_min_op=1e-8, chi_acc=None, svd_min_acc=1e-10,
                                window_pad=2, psi_ref=None, seed=0, svd_min_real=1e-9,
                                chi_mpo=None, weight_decay=0.25, svd_dense_limit=600):
    """Build O_f = int f^2 h_site(t) dt as a real symmetric MPO (module docstring).

    model: the TeNPy phi^4 model (``phi4_model`` / ``GroundState.model``);
    dt: target TEBD step (adjusted so that n dt = f.t_hi exactly); order:
    Suzuki-Trotter order (1, 2, 4 as in TeNPy); chi_op / svd_min_op: the
    truncation of the evolving operator (svd_min relative to the largest
    singular value); chi_acc / svd_min_acc: of the accumulator (default
    chi_acc = 2 chi_op); window_pad: light-cone window half-width
    ceil(t) + pad (None: whole chain); psi_ref: optional MPS to record
    <psi_ref|h(t)|psi_ref> along the trajectory; chi_mpo / svd_min_real: the
    final compression of the real symmetric O_f.  Q + Q^T is built at
    8 chi_op before it is compressed and the DMRG cost is linear in the MPO
    bond dimension, so this is the knob that makes the DMRG affordable; its
    discarded weight is reported as ``trunc_real`` and its effect on E_min
    is measured by varying it.  svd_dense_limit: two-site blocks larger than
    this (rows and columns) are truncated by the randomized sketch of
    :func:`_truncated_svd`; ``np.inf`` forces the exact dense SVD (a
    convergence control -- the sketch keeps a slightly suboptimal subspace
    when the block's spectrum decays slowly).
    """
    t = _tenpy()
    prm = _model_params(model)
    L, m, lam, n_max, omega, bc = (prm["L"], prm["m"], prm["lam"], prm["n_max"],
                                   prm["omega"], prm["bc"])
    x = int(site)
    d = n_max + 1
    if chi_acc is None:
        chi_acc = 2 * int(chi_op)
    rng = np.random.default_rng(int(seed))
    w_occ = None if weight_decay >= 1.0 else occupation_weights(n_max, weight_decay)
    sites_h, H3 = local_energy_operator_dense(n_max, omega, m, lam, x, L, bc)
    h = _operator_mps_from_dense(sites_h, H3, L, d)
    scale = math.sqrt(h.norm2())
    h = h.scale(1.0 / scale)
    h.canonicalize()
    h.T = [tt.astype(complex) for tt in h.T]

    Hbonds = _bond_hamiltonians(model)
    fracs = t.tebd.TEBDEngine.suzuki_trotter_time_steps(order)
    times, w, dt_eff = _time_grid(f, dt)
    layers = t.tebd.TEBDEngine.suzuki_trotter_decomposition(order, 1)[::-1]  # Heisenberg: reversed
    gates = {}
    for k_dt, _parity in set(layers):
        delta = fracs[k_dt] * dt_eff
        gates[k_dt] = {i: _gate(Hb, delta) for i, Hb in Hbonds.items()}
    f2 = np.asarray(f.f(times), float) ** 2

    def ref_expect(op: _OpMPS):
        if psi_ref is None:
            return None
        mpo = _mpo_from_tensors(psi_ref.sites, op.T, scale)
        return float(np.real(mpo.expectation_value(psi_ref)))

    acc = h.scale(w[0] * f2[0])
    href = [ref_expect(h)]
    disc_op = 0.0
    disc_acc = 0.0
    chi_op_max = max(h.chi) if h.chi else 1
    chi_acc_max = max(acc.chi) if acc.chi else 1
    for k in range(1, times.size):
        t_end = times[k]
        if window_pad is None:
            lo, hi = 0, L - 1
        else:
            R = int(math.ceil(t_end - 1e-9)) + int(window_pad)
            lo, hi = max(0, x - R), min(L - 1, x + R)
        for k_dt, parity in layers:
            for i in range(1, L):          # bond i acts on sites (i-1, i)
                if i % 2 != parity:
                    continue
                if i - 1 < lo or i > hi:
                    continue
                disc, tot = h.update_bond(i - 1, gates[k_dt][i], chi_op, svd_min_op,
                                          rng=rng, w=w_occ, dense_limit=svd_dense_limit)
                if tot > 0.0:
                    disc_op += disc / tot
        chi_op_max = max(chi_op_max, max(h.chi))
        acc = acc.add(h, 1.0, w[k] * f2[k])
        disc_acc += acc.canonicalize(chi_max=chi_acc, svd_min=svd_min_acc)
        chi_acc_max = max(chi_acc_max, max(acc.chi))
        href.append(ref_expect(h))
    norm_ratio = math.sqrt(h.norm2())
    # O_f = Q + Q^T, real: realify each and add, compress losslessly
    O = _realify(acc).add(_realify(acc.transpose()), 1.0, 1.0)
    disc_real = O.canonicalize(chi_max=chi_mpo, svd_min=svd_min_real)
    O.T = [np.real(tt) for tt in O.T]
    sites = model.lat.mps_sites()
    mpo = _mpo_from_tensors(sites, O.T, scale)
    meta = {"site": x, "L": L, "m": m, "lam": lam, "n_max": n_max, "omega": omega, "bc": bc,
            "dt": dt_eff, "order": order, "chi_op_max_allowed": int(chi_op), "svd_min_op": svd_min_op,
            "chi_acc_max_allowed": int(chi_acc), "svd_min_acc": svd_min_acc, "window_pad": window_pad,
            "chi_mpo_max_allowed": (None if chi_mpo is None else int(chi_mpo)),
            "svd_min_real": float(svd_min_real), "weight_decay": float(weight_decay),
            "svd_dense_limit": float(svd_dense_limit),
            "f": f.label, "t_hi": float(f.t_hi), "n_steps": int(times.size - 1)}
    return SmearedOperator(mpo=mpo, tensors=O.T, scale=scale, times=times, weights=w,
                           trunc_op=float(disc_op), trunc_acc=float(disc_acc),
                           trunc_real=float(disc_real), chi_op=int(chi_op_max),
                           chi_acc=int(chi_acc_max), chi_mpo=int(max(O.chi)),
                           h_ref_trace=np.array([v for v in href if v is not None], float),
                           norm_ratio=float(norm_ratio), meta=meta)


# --------------------------------------------------------------------------
# DMRG on O_f: the exact infimum and the extremal state
# --------------------------------------------------------------------------


class QEIExactResult(NamedTuple):
    E_min: float            # lambda_min(O_f) - <Omega|O_f|Omega>: the infimum
    E_star: float           # lambda_min(O_f) (DMRG ground energy of the MPO)
    E_ref: float            # <Omega|O_f|Omega>
    E_ref_static: float     # ||f||^2 <Omega|h_x|Omega>
    psi_star: object        # the extremal MPS (real)
    op: SmearedOperator
    variance: float         # <O_f^2> - <O_f>^2 in psi_star
    E_free_exact: float     # Gaussian exact infimum, same f/m/lattice (Williamson)
    E_hartree_exact: float  # ... at the Hartree mass
    mu_H2: float
    bound_2d: float
    dmrg: dict
    meta: dict


def free_exact_infimum(L, site, f: FHandle, m, bc="dirichlet"):
    """The free lattice infimum of ``vacuum.inequalities.qei`` (one Williamson problem)."""
    op = sampling_operator(int(L), int(site), f, m=float(m), bc=bc)
    return qei_minimize(op).e_min


def qei_exact_mps(gs: GroundState, site, f: FHandle, *, dt=0.25, order=4, chi_op=24,
                  svd_min_op=1e-8, chi_acc=None, svd_min_acc=1e-10, window_pad=2,
                  chi_mpo=None, svd_min_real=1e-9, weight_decay=0.25, chi_dmrg=None, max_sweeps=30, min_sweeps=4, max_E_err=1e-12,
                  svd_min_dmrg=1e-13, variance=True, seed=0, op: Optional[SmearedOperator] = None,
                  psi0=None, svd_dense_limit=600):
    """The exact lattice QEI infimum at the state's (m, lambda, n_max) (module docstring).

    gs: DMRG ground state (``phi4_ground_mps``); site: the worldline; f: the
    sampling function.  The MPO is built by :func:`heisenberg_smeared_operator`
    (its keyword arguments are passed through) unless a prebuilt `op` is
    given (chi_dmrg-doubling reuses one MPO); the DMRG on it is warm-started
    from psi0 (default gs.psi) at chi_dmrg (default: gs.info['chi_max']).
    Returns :class:`QEIExactResult`; every approximation is in `op` and
    `dmrg` (which also carries the wall times of the two phases).
    """
    import time as _time
    t = _tenpy()
    L, m, lam, bc = gs.info["L"], gs.info["m"], gs.info["lam"], gs.info["bc"]
    x = int(site)
    if chi_dmrg is None:
        chi_dmrg = int(gs.info["chi_max"])
    t0 = _time.time()
    if op is None:
        op = heisenberg_smeared_operator(gs.model, x, f, dt=dt, order=order, chi_op=chi_op,
                                         svd_min_op=svd_min_op, chi_acc=chi_acc,
                                         svd_min_acc=svd_min_acc, window_pad=window_pad,
                                         chi_mpo=chi_mpo, svd_min_real=svd_min_real,
                                         weight_decay=weight_decay, psi_ref=gs.psi, seed=seed,
                                         svd_dense_limit=svd_dense_limit)
    elif op.meta["site"] != x or op.meta["L"] != L or op.meta["lam"] != lam or op.meta["m"] != m \
            or op.meta["n_max"] != gs.info["n_max"] or op.meta["f"] != f.label:
        raise ValueError("the prebuilt SmearedOperator does not match (site, L, m, lam, n_max, f)")
    t_op = _time.time() - t0
    mpo = op.mpo
    E_ref = float(np.real(mpo.expectation_value(gs.psi)))
    f2_norm = 2.0 * float(np.sum(op.weights * np.asarray(f.f(op.times), float) ** 2))
    E_ref_static = f2_norm * float(op.h_ref_trace[0])
    from tenpy.models.lattice import Chain
    from tenpy.models.model import MPOModel
    sites = gs.model.lat.mps_sites()
    lat = Chain(L, sites[0], bc="open", bc_MPS="finite")
    omodel = MPOModel(lat, mpo)
    psi = (gs.psi if psi0 is None else psi0).copy()
    t0 = _time.time()
    options = {
        "trunc_params": {"chi_max": int(chi_dmrg), "svd_min": float(svd_min_dmrg)},
        "max_E_err": float(max_E_err), "max_S_err": 1e-8,
        "max_sweeps": int(max_sweeps), "min_sweeps": int(min_sweeps), "mixer": False,
        "lanczos_params": {"N_max": 60, "P_tol": 1e-16, "E_tol": 1e-16},
    }
    eng = t.dmrg.TwoSiteDMRGEngine(psi, omodel, options)
    E_star, psi = eng.run()
    psi.canonical_form()   # O_f is the identity outside the light cone: that tail is a
                           # flat direction the sweeps leave untouched (and un-canonical)
    E_star = float(np.real(mpo.expectation_value(psi)))
    stats = getattr(eng, "sweep_stats", None) or {}
    delta_E = None
    if stats.get("Delta_E"):
        delta_E = float(abs(stats["Delta_E"][-1]))
    max_trunc = float(np.max(stats["max_trunc_err"])) if stats.get("max_trunc_err") else 0.0
    dmrg = {"sweeps": int(getattr(eng, "sweeps", 0)), "delta_E": delta_E, "max_trunc_err": max_trunc,
            "chi_max": int(chi_dmrg), "chi": int(max(psi.chi)) if len(psi.chi) else 1,
            "converged": (delta_E is not None and delta_E <= float(max_E_err)),
            "mpo_chi": int(op.chi_mpo), "real": not np.iscomplexobj(psi.get_B(0).to_ndarray()),
            "time_operator_s": t_op, "time_dmrg_s": _time.time() - t0}
    t0 = _time.time()
    var = float(np.real(mpo.variance(psi))) if variance else float("nan")
    dmrg["time_variance_s"] = _time.time() - t0
    E_free = free_exact_infimum(L, x, f, m, bc)
    mu_H2 = hartree_mass_squared(L, m, lam, bc)
    E_H = free_exact_infimum(L, x, f, math.sqrt(mu_H2), bc)
    meta = {"site": x, "L": L, "m": m, "lam": lam, "bc": bc, "n_max": gs.info["n_max"],
            "f": f.label, "f2_norm": f2_norm, **{k: v for k, v in op.meta.items() if k not in ("site", "L", "m", "lam", "bc", "n_max", "f")}}
    return QEIExactResult(E_min=float(E_star - E_ref), E_star=float(E_star), E_ref=E_ref,
                          E_ref_static=float(E_ref_static), psi_star=psi, op=op, variance=var,
                          E_free_exact=float(E_free), E_hartree_exact=float(E_H),
                          mu_H2=float(mu_H2), bound_2d=float(qei_bound_2d(f)), dmrg=dmrg, meta=meta)


# --------------------------------------------------------------------------
# The O(lambda) slope of the infimum (the small-lambda anchor; pure numpy)
# --------------------------------------------------------------------------


def first_order_infimum_slope(L, site, f: FHandle, m, bc="dirichlet", *, n_panels_t=12,
                              nodes_t=10, nodes_s=24, eps_fd=1e-3):
    """d E_min / d lambda at lambda = 0, exactly, from the free extremal state.

    Hellmann-Feynman on the lowest eigenvalue of O_f(lambda), evaluated in
    the free extremal state psi* (the Gaussian vacuum of O_f, exhibited by
    ``qei_minimize``), with the Duhamel derivative of the Heisenberg operator

        d/dlambda [U^dag h_x U] |_0 = v_x(t) + i int_0^t ds [V(s), h_x(t)],

    V = sum_i phi_i^4 / 24, v_x = phi_x^4 / 24, X(t) = e^{iH_0 t} X e^{-iH_0 t}.
    In a Gaussian state Wick's theorem closes both terms in the covariance
    V* of psi* and the free propagator S(t) (phi_i(s) = c_i(s)^T R with
    c_i(s) the i-th row of S(s); h_x(t) = (1/2) R^T h_t R, h_t = S(t)^T h S(t)):

        <v_x(t)>              = (1/8) (c_x(t)^T V* c_x(t))^2,
        i <[phi_i(s)^4/24, h_x(t)]> = -(1/2) (c_i^T Omega h_t V* c_i)(c_i^T V* c_i),

    the second line being the mean-field identity [phi^4, q] -> 6 <phi^2> [phi^2, q]
    that holds for commutators with a quadratic form in any Gaussian state.
    The reference <Omega_lambda|O_f|Omega_lambda> = ||f||^2 <h_x>_lambda is
    differentiated through the O(lambda) tadpole covariance of ``phi4``
    (exact through first order) plus the direct quartic expectation:
    d<h_x>/dlambda = (1/2) Tr(h dV_lambda/dlambda) + X_x^2 / 8.

    Returns a dict: 'E_free' (the lambda = 0 infimum), 'dE_min' (the slope of
    the infimum), 'dE_star', 'dE_ref' (its two pieces), 'dE_hartree' (the
    slope of the free infimum at the Hartree mass, by central differences),
    and 'dR' = d/dlambda [1 - E_min/E_hartree] at 0 = (dE_hartree - dE_min) /
    E_free -- the O(lambda) coefficient of the residual beyond mass
    renormalization.  dR = 0 would mean the Hartree reference absorbs the
    whole first-order effect; dR > 0 is a first-order 'quartic tax'.
    Quadrature: composite Gauss-Legendre in t over the handle's support and
    in s over [0, t] (both spectrally accurate for the Gaussian weight).
    """
    L = int(L)
    x = int(site)
    m = float(m)
    K = free_coupling_matrix(L, m, bc)
    N2 = 2 * L
    Om = np.zeros((N2, N2))
    Om[:L, L:] = np.eye(L)
    Om[L:, :L] = -np.eye(L)
    op = sampling_operator(L, x, f, m=m, bc=bc)
    res = qei_minimize(op)
    V_star = embed_extremal(res, op)
    h = chain_energy_density_form(L, x, m, bc)
    V0 = ground_state_cov(K)

    ts, wts = _gauss_legendre_grid(f.t_lo, f.t_hi, n_panels_t, nodes_t)
    f2 = np.asarray(f.f(ts), float) ** 2
    xs_gl, ws_gl = np.polynomial.legendre.leggauss(int(nodes_s))
    dA = 0.0
    dB = 0.0
    for t, wt, f2t in zip(ts, wts, f2):
        if f2t < 1e-300:
            continue
        St = chain_propagator(K, float(t))
        cx = St[x]
        dA += wt * f2t * 0.125 * float(cx @ V_star @ cx) ** 2
        h_t = St.T @ h @ St
        A_t = Om @ h_t @ V_star            # c^T (Omega h_t V*) c
        # inner integral over s in [0, t] (signed)
        ss = 0.5 * t * (xs_gl + 1.0)
        ws_s = 0.5 * t * ws_gl
        inner = 0.0
        for s_, w_s in zip(ss, ws_s):
            Ss = chain_propagator(K, float(s_))
            C = Ss[:L]                       # rows c_i(s), i = 0..L-1
            q1 = np.einsum("ik,kl,il->i", C, A_t, C)
            q2 = np.einsum("ik,kl,il->i", C, V_star, C)
            inner += w_s * float(np.sum(-0.5 * q1 * q2))
        dB += wt * f2t * inner
    dE_star = dA + dB

    # reference: ||f||^2 d<h_x>_lambda / dlambda at 0
    f2_norm = float(np.sum(wts * f2))
    D = _tadpole_D(K)
    Vp = ground_state_cov(K + eps_fd * D)
    Vm = ground_state_cov(K - eps_fd * D)
    dV = (Vp - Vm) / (2.0 * eps_fd)
    X_x = float(V0[x, x])
    dE_ref = f2_norm * (0.5 * float(np.einsum("ij,ji->", h, dV)) + X_x * X_x / 8.0)
    dE_min = dE_star - dE_ref

    E_free = float(res.e_min)
    mu_p = math.sqrt(hartree_mass_squared(L, m, eps_fd, bc))
    E_Hp = free_exact_infimum(L, x, f, mu_p, bc)
    # mu_H^2 = m^2 - eps X/2 + ... is not a valid mass for lambda < 0 in general;
    # use a one-sided second-order stencil on the positive side instead
    mu_pp = math.sqrt(hartree_mass_squared(L, m, 2.0 * eps_fd, bc))
    E_Hpp = free_exact_infimum(L, x, f, mu_pp, bc)
    dE_hartree = (-3.0 * E_free + 4.0 * E_Hp - E_Hpp) / (2.0 * eps_fd)
    dR = (dE_hartree - dE_min) / E_free
    return {"E_free": E_free, "dE_min": float(dE_min), "dE_star": float(dE_star),
            "dE_star_quartic": float(dA), "dE_star_dynamic": float(dB),
            "dE_ref": float(dE_ref), "dE_hartree": float(dE_hartree), "dR": float(dR),
            "f2_norm": f2_norm, "extremal_gap": float(res.gap), "L": L, "site": x, "m": m,
            "f": f.label}

