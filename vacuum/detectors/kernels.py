"""Wightman kernels for detector physics (Layer 2, milestone M2.1).

Two backends, because the anchors live in different spacetimes
(docs/PLAN_LAYERS_2_3.md, M2.1):

**Lattice backend (native).** For the lattice field H = (1/2)(p.p + x.K.x)
with K = U diag(omega^2) U^T (eigh cached at construction), the vacuum
Wightman function of the position operators is

    W_ij(t, t') = <0| x_i(t) x_j(t') |0> = sum_k U_ik U_jk e^{-i omega_k (t-t')} / (2 omega_k),

a function of dt = t - t' only.  Its real part is the (symmetrized) Hadamard
function; its imaginary part is state-independent and proportional to the
Pauli-Jordan commutator: with the spectral propagator S(t) of
``vacuum.core.dynamics.chain_propagator``,

    [x_i(dt), x_j(0)] = i (S(dt) Omega)_{ij}  =>  2 Im W(dt) = (S(dt) Omega)_{xx block},

which is exactly the object the causality audit checks (cross-test in
tests/test_kernels.py).  Smeared two-point kernels are F^T W(dt) F' for
spatial profiles F, F' (see :meth:`LatticeWightman.smeared`).

**Continuum 3+1 massless backend (for the PKMM anchor).**  Massless scalar
vacuum in 3+1 Minkowski, static detectors, Gaussian spatial smearing
F(x) = exp(-|x|^2/sigma^2) / (pi^{3/2} sigma^3) (the convention of
Pozas-Kerstjens and Martin-Martinez, PRD 92, 064042 (2015); Fourier
transform Ftilde(k) = exp(-k^2 sigma^2 / 4)).  Pulling the mode expansion
of the massless Wightman function (standard flat-space QFT; e.g. Birrell &
Davies, "Quantum Fields in Curved Space", CUP 1982, Ch. 3) back to two
static smeared detectors separated by d gives, after the angular integral
int dOmega_k e^{i k.d} = 4 pi sin(kd)/(kd), the absolutely convergent 1D
radial integral

    W(dt; d, sigma) = (1/4pi^2) int_0^inf dk k e^{-k^2 sigma^2/2}
                                          * sinc(kd) * e^{-i k dt},

with sinc(kd) = sin(kd)/(kd) -> 1 as d -> 0.  The Gaussian smearing factor
|Ftilde|^2 = e^{-k^2 sigma^2/2} supplies UV convergence — no i-epsilon
prescription anywhere.  Pointlike (sigma = 0) physics is only exposed
through the sigma -> 0 closed-form cross-check
(:func:`response_gaussian_closed_form`); production code always smears.

**Quadrature hazards (logged per spec M2.1).**  Oscillatory e^{i Omega t}
factors are integrated on Gauss-Legendre panels sized
min(1/Omega, switching width)/8 with a doubling convergence test
(:func:`detector_response`).  The time-ordered theta(t - t') kernel of the
pair term integrates over a triangle: use the triangular tensor-product
rule :func:`triangle_rule`, never a masked square grid (masking costs an
order of accuracy at the diagonal).

Everything here is pure-functional over arrays: kernel objects are frozen
containers of precomputed spectra whose methods are pure functions of their
arguments (pre-positioning for the Layer 6 JAX mirror).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import erfcx

__all__ = [
    "LatticeWightman",
    "ExpSumKernel",
    "DerivativeKernel",
    "RefinableDerivativeKernel",
    "derivative_kernel",
    "ContinuumWightman3p1",
    "wightman_lattice",
    "wightman_continuum_3p1",
    "gl_panels",
    "triangle_rule",
    "detector_response",
    "response_gaussian_closed_form",
]

_FOUR_PI2 = 4.0 * np.pi**2

# Cache of Gauss-Legendre nodes/weights on [-1, 1], keyed by order.  The
# cached arrays are read-only, so this is a memoization of pure data, not
# hidden mutable state.
_GL_CACHE: dict = {}


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
# Quadrature toolkit (hazard-note rules, spec M2.1)
# ---------------------------------------------------------------------------

def gl_panels(a, b, max_panel, n_gl=16):
    """Gauss-Legendre panel quadrature nodes/weights on [a, b].

    The interval is split into equal panels of width <= ``max_panel`` and an
    ``n_gl``-point Gauss-Legendre rule is mapped onto each panel.  This is
    the rule prescribed by the M2.1 hazard note for oscillatory e^{i Omega t}
    integrands: size the panels to min(1/Omega, switching width)/8 and halve
    ``max_panel`` for the doubling convergence test.

    Returns
    -------
    (nodes, weights) : 1D float ndarrays, sum(weights) = b - a.
    """
    a = float(a)
    b = float(b)
    if not b > a:
        raise ValueError(f"need b > a, got a={a}, b={b}")
    max_panel = float(max_panel)
    if not max_panel > 0.0:
        raise ValueError(f"max_panel must be > 0, got {max_panel}")
    x, w = _leggauss_cached(n_gl)
    n_panels = max(1, int(math.ceil((b - a) / max_panel - 1e-12)))
    edges = np.linspace(a, b, n_panels + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    half = 0.5 * (edges[1:] - edges[:-1])
    nodes = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    return nodes, weights


def triangle_rule(a, b, max_panel, n_gl=16):
    """Tensor-product Gauss-Legendre rule on the triangle a <= t' <= t <= b.

    For the time-ordered theta(t - t') kernel: the outer variable t gets GL
    panels on [a, b]; for each outer node t_i the inner variable t' gets GL
    panels on [a, t_i].  Because every inner rule ends exactly at the
    diagonal t' = t, no accuracy is lost there — the M2.1 hazard note
    forbids the alternative (masking a square grid), which degrades the
    order of accuracy at the diagonal.

    Returns
    -------
    (t, tp, w) : 1D ndarrays with sum over nodes of w * f(t, tp)
        approximating int_a^b dt int_a^t dt' f(t, t').
    """
    t_out, w_out = gl_panels(a, b, max_panel, n_gl=n_gl)
    ts, tps, ws = [], [], []
    for t_i, w_i in zip(t_out, w_out):
        if not t_i > a:  # pragma: no cover - GL nodes are interior
            continue
        tp_i, wp_i = gl_panels(a, t_i, max_panel, n_gl=n_gl)
        ts.append(np.full(tp_i.size, t_i))
        tps.append(tp_i)
        ws.append(w_i * wp_i)
    return np.concatenate(ts), np.concatenate(tps), np.concatenate(ws)


# ---------------------------------------------------------------------------
# Lattice backend
# ---------------------------------------------------------------------------

class LatticeWightman:
    """Vacuum Wightman kernel of the lattice field H = (1/2)(p.p + x.K.x).

    Frozen container: the eigendecomposition K = U diag(omega^2) U^T is
    computed once at construction (the "cached eigh" of the spec) and every
    method is a pure function of dt.

    Calling the object returns the complex (N, N) matrix

        W(dt) = U diag(e^{-i omega dt} / (2 omega)) U^T,

    i.e. W_ij(t, t') for dt = t - t'.  ``real_part`` (Hadamard) and
    ``imag_part`` (proportional to the state-independent Pauli-Jordan
    commutator: [x_i(dt), x_j(0)] = 2 i Im W_ij(dt)) are separately
    accessible; ``d2`` is the analytic second dt-derivative, so equal-time
    momentum moments are <p_i p_j> = -Re d2(0) = (K^{1/2}/2)_ij.
    """

    __slots__ = ("omega", "U", "N")

    def __init__(self, K):
        K = np.asarray(K, dtype=float)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError(f"K must be square, got shape {K.shape}")
        w2, U = np.linalg.eigh(0.5 * (K + K.T))
        # Relative threshold: a true zero mode of the massless periodic
        # chain comes back from eigh as O(1e-16) roundoff, not exactly 0.
        if np.min(w2) <= 1e-12 * max(1.0, float(np.max(np.abs(w2)))):
            raise ValueError(
                "K must be positive definite for the Wightman kernel "
                f"(zero/negative mode): min eig {np.min(w2):.3e}"
            )
        self.omega = _readonly(np.sqrt(w2))
        self.U = _readonly(U)
        self.N = int(K.shape[0])

    def __call__(self, dt):
        """Complex Wightman matrix W(dt), shape (N, N), symmetric."""
        f = np.exp(-1j * self.omega * float(dt)) / (2.0 * self.omega)
        return (self.U * f) @ self.U.T

    def real_part(self, dt):
        """Re W(dt) = U diag(cos(omega dt)/(2 omega)) U^T (Hadamard part)."""
        f = np.cos(self.omega * float(dt)) / (2.0 * self.omega)
        return (self.U * f) @ self.U.T

    def imag_part(self, dt):
        """Im W(dt) = -U diag(sin(omega dt)/(2 omega)) U^T.

        State-independent (spectral data only); equals (S(dt) Omega)_{xx}/2
        with the causality module's propagator S — the Pauli-Jordan
        commutator function of the lattice field.
        """
        f = np.sin(self.omega * float(dt)) / (2.0 * self.omega)
        return -(self.U * f) @ self.U.T

    def d2(self, dt):
        """Second derivative d^2 W / d(dt)^2 = -U diag(omega e^{-i omega dt}/2) U^T."""
        f = -0.5 * self.omega * np.exp(-1j * self.omega * float(dt))
        return (self.U * f) @ self.U.T

    def momentum(self, dt):
        """Momentum-momentum Wightman matrix <0| p_i(t) p_j(t') |0>, dt = t - t'.

        p_i = d_t x_i for the free lattice field, so
        <p_i(t) p_j(t')> = d_t d_t' W_ij(t - t') = -W''(dt)
                         = U diag(omega e^{-i omega dt} / 2) U^T,
        the pulled-back kernel of the derivative-coupled detector
        (``vacuum.detectors.udw_derivative``).  Equal-time real part is
        <p_i p_j> = (K^{1/2}/2)_ij, the p-p block of the ground-state
        covariance.
        """
        return -self.d2(dt)

    def smeared(self, F, Fprime=None):
        """Smeared two-point kernel F^T W(dt) F' as an :class:`ExpSumKernel`.

        W_FF'(dt) = sum_k c_k e^{-i omega_k dt} with
        c_k = (U^T F)_k (U^T F')_k / (2 omega_k).  ``Fprime=None`` means
        F' = F (single-detector auto-correlation kernel).  Accepts plain
        arrays or :class:`vacuum.detectors.smearing.Smearing` objects.
        """
        F = np.asarray(F, dtype=float)
        if F.shape != (self.N,):
            raise ValueError(f"F must have shape ({self.N},), got {F.shape}")
        a = self.U.T @ F
        if Fprime is None:
            b = a
        else:
            Fprime = np.asarray(Fprime, dtype=float)
            if Fprime.shape != (self.N,):
                raise ValueError(
                    f"Fprime must have shape ({self.N},), got {Fprime.shape}"
                )
            b = self.U.T @ Fprime
        return ExpSumKernel(self.omega, a * b / (2.0 * self.omega))


def wightman_lattice(K):
    """Lattice Wightman kernel: ``wightman_lattice(K)(dt)`` -> complex (N, N).

    The eigendecomposition of K is cached inside the returned
    :class:`LatticeWightman`; construct once per K and reuse.
    """
    return LatticeWightman(K)


# ---------------------------------------------------------------------------
# Exponential-sum kernels (common form of both smeared backends)
# ---------------------------------------------------------------------------

class ExpSumKernel:
    """Scalar kernel W(dt) = sum_k amps_k e^{-i freqs_k dt}.

    Common representation of every smeared two-point kernel in this module:
    lattice smeared kernels carry the exact normal-mode sum, the continuum
    backend carries its radial k-quadrature nodes.  ``freqs`` and ``amps``
    are exposed read-only so response integrals can factorize over modes
    (see :func:`detector_response`).
    """

    __slots__ = ("freqs", "amps")

    def __init__(self, freqs, amps):
        freqs = np.asarray(freqs, dtype=float)
        amps = np.asarray(amps, dtype=complex)
        if freqs.ndim != 1 or freqs.shape != amps.shape:
            raise ValueError("freqs and amps must be matching 1D arrays")
        self.freqs = _readonly(freqs)
        self.amps = _readonly(amps, dtype=complex)

    def _sum(self, dt, coeff):
        dt_arr = np.asarray(dt, dtype=float)
        phase = np.exp(-1j * dt_arr[..., None] * self.freqs)
        out = phase @ coeff
        return complex(out) if dt_arr.ndim == 0 else out

    def __call__(self, dt):
        """Kernel value(s) at dt (scalar or ndarray; complex out)."""
        return self._sum(dt, self.amps)

    def real_part(self, dt):
        return np.real(self.__call__(dt))

    def imag_part(self, dt):
        return np.imag(self.__call__(dt))

    def d2(self, dt):
        """Second derivative d^2 W / d(dt)^2 = -sum_k amps_k freqs_k^2 e^{-i freqs_k dt}."""
        return self._sum(dt, -self.amps * self.freqs**2)


class DerivativeKernel(ExpSumKernel):
    """d_t d_t' W — the momentum (pi-pi) kernel of an exponential-sum kernel.

    For W(dt) = sum_k c_k e^{-i w_k dt} (dt = t - t'),

        <pi(t) pi(t')> = d_t d_t' W(t - t') = -W''(dt)
                       = sum_k c_k w_k^2 e^{-i w_k dt},

    i.e. the same exponential sum with amplitudes c_k w_k^2.  This is the
    pulled-back kernel of a detector coupled to the *time derivative* of
    the field (pi = d_t phi for the free field; Teixido-Bonfill &
    Martin-Martinez, PRD 110, 105016 (2024), Eq. (2) — see
    ``vacuum.detectors.udw_derivative``):

    - lattice backend, ``LatticeWightman.smeared(F, F')`` -> the smeared
      form of U diag(omega e^{-i omega dt}/2) U^T (``LatticeWightman.momentum``);
    - continuum 3+1 backend -> the extra k^2 factor in the radial integrand,
      (1/4pi^2) int dk k^3 e^{-k^2 sigma^2/2} sinc(kd) e^{-i k dt}
      (the Gaussian smearing still supplies absolute convergence).

    The base kernel is kept (read-only) for provenance.  Use
    :func:`derivative_kernel` to build one: it returns
    :class:`RefinableDerivativeKernel` when the base carries its own
    k-quadrature, so the M2.3 kernel-refinement gate stays meaningful.
    """

    __slots__ = ("base",)

    def __init__(self, base):
        freqs = np.asarray(base.freqs, dtype=float)
        amps = np.asarray(base.amps, dtype=complex)
        super().__init__(freqs, amps * freqs**2)
        object.__setattr__(self, "base", base)

    def __repr__(self):
        return f"DerivativeKernel(base={type(self.base).__name__})"


class RefinableDerivativeKernel(DerivativeKernel):
    """:class:`DerivativeKernel` over a base with its own quadrature.

    Forwards ``with_refined`` / ``with_kmax`` to the base, so the
    k-refinement and cutoff-independence gates of the continuum backend act
    on the derivative kernel exactly as on the base.
    """

    __slots__ = ()

    def with_refined(self, factor=2):
        return RefinableDerivativeKernel(self.base.with_refined(factor))

    def with_kmax(self, kmax):
        return RefinableDerivativeKernel(self.base.with_kmax(kmax))


def derivative_kernel(kernel):
    """d_t d_t' W of an exponential-sum kernel (see :class:`DerivativeKernel`).

    Returns a :class:`RefinableDerivativeKernel` when ``kernel`` has
    ``with_refined`` (continuum backend), else a plain
    :class:`DerivativeKernel` (lattice: exact normal-mode sum, nothing to
    refine).
    """
    if not (hasattr(kernel, "freqs") and hasattr(kernel, "amps")):
        raise TypeError(
            "derivative_kernel needs an ExpSumKernel-like object with "
            f".freqs/.amps, got {type(kernel).__name__}"
        )
    if hasattr(kernel, "with_refined"):
        return RefinableDerivativeKernel(kernel)
    return DerivativeKernel(kernel)


class ContinuumWightman3p1(ExpSumKernel):
    """Gaussian-smeared massless 3+1 Wightman kernel (radial k-quadrature).

    W(dt) = (1/4pi^2) int_0^kmax dk k e^{-k^2 sigma^2/2} sinc(k d) e^{-i k dt}

    discretized on Gauss-Legendre panels in k (absolutely convergent — the
    e^{-k^2 sigma^2/2} smearing factor is the UV regulator, no i-epsilon).
    The quadrature nodes/weights become ``freqs``/``amps`` of the parent
    :class:`ExpSumKernel`, so evaluation and response integrals share one
    code path with the lattice backend.

    ``kmax`` defaults to the point where the smearing factor reaches
    ~1e-26 (kmax = sqrt(120)/sigma); cutoff-independence around that tail
    is a permanent test (vary via :meth:`with_kmax`, observables must move
    < 1e-8).  Panel width defaults to (1/sigma)/8, the spectral width of
    the smearing Gaussian over 8; refine via :meth:`with_refined`.
    """

    __slots__ = ("sigma", "distance", "kmax", "panel", "n_gl")

    def __init__(self, sigma, distance=0.0, kmax=None, panel=None, n_gl=16):
        sigma = float(sigma)
        if not sigma > 0.0:
            raise ValueError(
                "sigma must be > 0: the continuum backend always smears; "
                "pointlike physics is exposed only through the sigma -> 0 "
                "closed-form cross-check (response_gaussian_closed_form)"
            )
        distance = float(distance)
        if distance < 0.0:
            raise ValueError(f"distance must be >= 0, got {distance}")
        if kmax is None:
            # e^{-kmax^2 sigma^2 / 2} = e^{-60} ~ 8.8e-27: far below every
            # 1e-8 target, yet finite so cutoff-independence is testable.
            kmax = math.sqrt(120.0) / sigma
        kmax = float(kmax)
        if not kmax > 0.0:
            raise ValueError(f"kmax must be > 0, got {kmax}")
        if panel is None:
            # Resolve the two intrinsic k-scales: the smearing Gaussian
            # (width 1/sigma in k) and, for separated detectors, sin(kd)
            # (period 2 pi / d).  Oscillation from e^{-i k dt} at large |dt|
            # is handled by with_refined / the response doubling test.
            panel = min(1.0 / sigma, np.inf if distance == 0.0 else 1.0 / distance) / 8.0
        panel = float(panel)

        k, w = gl_panels(0.0, kmax, panel, n_gl=n_gl)
        rho = k * np.exp(-0.5 * (k * sigma) ** 2) / _FOUR_PI2
        if distance > 0.0:
            # sinc(kd) = sin(kd)/(kd); np.sinc(x) = sin(pi x)/(pi x)
            rho = rho * np.sinc(k * distance / np.pi)
        super().__init__(k, w * rho)
        self.sigma = sigma
        self.distance = distance
        self.kmax = kmax
        self.panel = panel
        self.n_gl = int(n_gl)

    def with_kmax(self, kmax):
        """New kernel with a different hard k-cutoff (cutoff-independence tests)."""
        return ContinuumWightman3p1(
            self.sigma, self.distance, kmax=kmax, panel=self.panel, n_gl=self.n_gl
        )

    def with_refined(self, factor=2):
        """New kernel with k-panels finer by ``factor`` (doubling convergence test)."""
        return ContinuumWightman3p1(
            self.sigma,
            self.distance,
            kmax=self.kmax,
            panel=self.panel / float(factor),
            n_gl=self.n_gl,
        )


def wightman_continuum_3p1(sigma, distance=0.0, kmax=None, panel=None, n_gl=16):
    """Continuum 3+1 massless Gaussian-smeared Wightman kernel.

    See :class:`ContinuumWightman3p1`.  ``sigma`` is the PKMM smearing width
    (F(x) = exp(-|x|^2/sigma^2)/(pi^{3/2} sigma^3)); ``distance`` separates
    the two detector centers (0 for a single detector's auto-correlation).
    """
    return ContinuumWightman3p1(sigma, distance=distance, kmax=kmax, panel=panel, n_gl=n_gl)


# ---------------------------------------------------------------------------
# Detector response (single detector, second order in the coupling)
# ---------------------------------------------------------------------------

def _response_quad(freqs, amps, chi, Omega, a, b, panel, n_gl, chunk=1024):
    """One evaluation of the response double integral at fixed panel width.

    P = int dt int dt' chi(t) chi(t') e^{-i Omega (t - t')} W(t - t') on the
    GL panel nodes.  Because W(dt) = sum_k c_k e^{-i w_k dt}, the double sum
    factorizes exactly per mode:

        P = sum_k c_k |A_k|^2,   A_k = sum_i w_i chi(t_i) e^{-i (Omega + w_k) t_i},

    bitwise the same quadrature as the O(n^2) double sum, at O(n_k n_t) cost.
    """
    t, w = gl_panels(a, b, panel, n_gl=n_gl)
    g = w * np.asarray(chi(t), dtype=float)
    acc = 0.0 + 0.0j
    for start in range(0, freqs.size, chunk):
        f = freqs[start : start + chunk]
        A = np.exp(-1j * (Omega + f)[:, None] * t[None, :]) @ g
        acc += np.sum(amps[start : start + chunk] * (A * np.conj(A)))
    return acc


def detector_response(
    kernel,
    chi,
    Omega,
    rtol=1e-10,
    atol=1e-18,
    window=None,
    n_gl=16,
    max_doublings=12,
    refine_kernel=True,
):
    """Single-detector response P/lambda^2 (second order in the coupling).

    P / lambda^2 = int dt int dt' chi(t) chi(t') e^{-i Omega (t-t')} W(t-t')

    for the *auto-correlation* kernel W of one detector (an
    :class:`ExpSumKernel`: continuum backend, or ``LatticeWightman.smeared(F)``
    with F' = F) and switching function ``chi``
    (:class:`vacuum.detectors.switching.Switching`).  This is the standard
    UDW vacuum excitation probability per unit coupling squared (Unruh 1976;
    DeWitt 1979; smeared form as in PKMM, PRD 92, 064042 (2015)).

    Quadrature per the M2.1 hazard note: Gauss-Legendre panels in time,
    initial width min(1/|Omega|, chi.width)/8, halved until successive
    values agree to ``rtol``/``atol`` (the doubling test).  If the kernel
    supports ``with_refined`` (continuum backend), its k-quadrature is then
    doubled under the same criterion, so the returned value is converged in
    both quadratures.  Raises RuntimeError if either fails to converge.

    ``window`` defaults to ``chi.support`` (exact for cos2; the 8.5-sigma
    effective support for the Gaussian, truncation error ~1e-16 relative).
    """
    freqs = getattr(kernel, "freqs", None)
    amps = getattr(kernel, "amps", None)
    if freqs is None or amps is None:
        raise TypeError(
            "detector_response needs a smeared scalar kernel with .freqs/.amps "
            "(ExpSumKernel); for the lattice backend pass "
            "wightman_lattice(K).smeared(F)"
        )
    Omega = float(Omega)
    a, b = window if window is not None else chi.support
    width = float(chi.width)
    h = min(1.0 / abs(Omega), width) / 8.0 if Omega != 0.0 else width / 8.0

    prev = None
    P = None
    for _ in range(max_doublings):
        P = _response_quad(freqs, amps, chi, Omega, a, b, h, n_gl)
        if prev is not None and abs(P - prev) <= max(rtol * abs(P), atol):
            break
        prev = P
        h *= 0.5
    else:
        raise RuntimeError(
            f"time-quadrature doubling did not converge after {max_doublings} "
            f"doublings (last change {abs(P - prev):.3e})"
        )

    if refine_kernel and hasattr(kernel, "with_refined"):
        ker = kernel
        for _ in range(max_doublings):
            ker = ker.with_refined(2)
            P2 = _response_quad(ker.freqs, ker.amps, chi, Omega, a, b, h, n_gl)
            if abs(P2 - P) <= max(rtol * abs(P2), atol):
                P = P2
                break
            P = P2
        else:
            raise RuntimeError(
                f"kernel k-refinement did not converge after {max_doublings} doublings"
            )

    return float(np.real(P))


def response_gaussian_closed_form(Omega, T, sigma=0.0):
    """Closed-form P/lambda^2: static UDW detector, Gaussian switching, 3+1 vacuum.

    Setup: massless scalar vacuum in 3+1 Minkowski; static detector of gap
    Omega, Gaussian switching chi(t) = exp(-(t - t0)^2 / (2 T^2)) (this
    module's ``switching('gaussian', T, t0)``), Gaussian spatial smearing of
    width sigma (PKMM convention; sigma = 0 is the pointlike limit).

    Derivation (self-contained, each step elementary; kept here per the
    cite-or-derive rule):

        P/lambda^2 = int dt dt' chi(t) chi(t') e^{-i Omega (t-t')} W(t-t')
                   = int_0^inf dk (k/4pi^2) e^{-k^2 sigma^2/2} |chihat(Omega+k)|^2

    (Fubini, absolutely convergent for sigma > 0; for sigma = 0 the time
    integrals are done first and the k-integral below still converges
    because |chihat|^2 is Gaussian).  With
    chihat(u) = int chi(t) e^{-i u t} dt = sqrt(2 pi) T e^{-T^2 u^2/2 - i u t0},
    |chihat(u)|^2 = 2 pi T^2 e^{-T^2 u^2} (t0 drops out), so

        P/lambda^2 = (T^2 / 2 pi) int_0^inf dk k e^{-alpha k^2 - b k - Omega^2 T^2},
        alpha = T^2 + sigma^2/2,   b = 2 Omega T^2.

    The Gaussian integral int_0^inf k e^{-alpha k^2 - b k} dk
    = 1/(2 alpha) - (b / 4 alpha) sqrt(pi/alpha) e^{b^2/4alpha} erfc(b/2 sqrt(alpha))
    (differentiate int_0^inf e^{-alpha k^2 - b k} dk w.r.t. b), giving

        P/lambda^2 = (T^2 / 2 pi) e^{-Omega^2 T^2}
                     [ 1/(2 alpha)
                       - (Omega T^2 / 2 alpha) sqrt(pi/alpha) erfcx(Omega T^2 / sqrt(alpha)) ],

    evaluated with the scaled complementary error function
    erfcx(x) = e^{x^2} erfc(x) (overflow-safe).  At sigma = 0 this reduces to
    the textbook pointlike result for Gaussian switching,

        P/lambda^2 = (1/4pi) [ e^{-Omega^2 T^2} - sqrt(pi) Omega T erfc(Omega T) ],

    the standard finite-time inertial-detector response in 3+1 Minkowski
    vacuum (cf. Sriramkumar & Padmanabhan, Class. Quantum Grav. 13, 2061
    (1996), for Gaussian-windowed inertial detector response; form as used
    throughout the entanglement-harvesting literature).  Both branches are
    verified against independent numerical quadrature in
    tests/test_kernels.py.
    """
    Omega = float(Omega)
    T = float(T)
    sigma = float(sigma)
    if not T > 0.0:
        raise ValueError(f"T must be > 0, got {T}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    alpha = T * T + 0.5 * sigma * sigma
    z = Omega * T * T / math.sqrt(alpha)
    bracket = 1.0 / (2.0 * alpha) - (Omega * T * T / (2.0 * alpha)) * math.sqrt(
        np.pi / alpha
    ) * erfcx(z)
    return (T * T / (2.0 * np.pi)) * math.exp(-((Omega * T) ** 2)) * bracket
