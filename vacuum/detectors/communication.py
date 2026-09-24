"""Harvesting vs. communication: the commutator/anti-commutator split (M2.5).

Two detectors that are in causal contact can end up entangled for two very
different reasons: they can *harvest* correlations that already exist in
the field state, or they can simply *signal* to each other through the
field.  The estimator that separates them, and the one this module
implements, is

    E. Tjoa and E. Martin-Martinez, "When entanglement harvesting is not
    really harvesting", Phys. Rev. D 104, 125005 (2021), arXiv:2109.11561
    -- referred to as TMM21 below.

Provenance of the equation numbers
----------------------------------
The arXiv source tarball (arxiv.org/e-print/2109.11561) was fetched on
2026-09-01; the printed equation numbers below were obtained by counting
the numbered environments of ``EH-comm.tex`` and cross-checked against the
ar5iv rendering (ar5iv.labs.arxiv.org/html/2109.11561), which carries the
printed tags -- both give 94 numbered equations and agree on every tag
spot-checked.  The (label, printed number) pairs used here are:

    eq: final-detector-matrix (14)   eq: Lij (15)      eq: M-nonloc (16)
    eq: commutator-anticomm (22)     eq: anti-comm (23)  eq: comm (24)
    eq: M-splitting (31)             eq: Mplus (32)    eq: Mmin (33)
    eq: communication-estimator (39)

The physics
-----------
TMM21 Eq. (22) splits the Wightman function into its real and imaginary
parts,

    W(x, x') = (1/2)[ C^+(x, x') + C^-(x, x') ],
    C^+ = <{phi(x), phi(x')}>   (anti-commutator, Hadamard)   [Eq. (23)],
    C^- = <[phi(x), phi(x')]>   (commutator, Pauli-Jordan)    [Eq. (24)],

so that (1/2)C^+ = Re W and (1/2)C^- = i Im W.  The two halves behave
completely differently:

- **C^- is state-independent.**  Any entanglement it produces would be
  produced just as well by a field state with *no correlations at all*,
  so it cannot be called harvesting.  It is also exactly the object that
  carries leading-order signalling between the detectors (it is a Green's
  function of the Klein-Gordon equation) and it vanishes identically for
  spacelike-separated arguments.
- **C^+ is state-dependent**, has support at spacelike separation, and is
  the only surviving contribution when the detectors cannot communicate.
  Genuine vacuum entanglement harvesting must come from it.

The pair term M [TMM21 Eq. (16) = PKMM Eq. (18) = this repo's
``vacuum.detectors.udw.pair_term``] is *linear* in the Wightman kernel, so
the split passes straight through it [Eq. (31)]:

    M = M^+ + M^-,      M^+ from (1/2)C^+ = Re W    [Eq. (32)],
                        M^- from (1/2)C^- = i Im W  [Eq. (33)].

In this module's names, ``M_vac = M^+`` (genuine vacuum correlation) and
``M_comm = M^-`` (field-mediated signalling), which is the naming the spec
(M2.5) fixes: "once with the full W in the time-ordered kernel, once with
only its imaginary (commutator/retarded) part; M_comm is the second,
M_vac = M - M_comm".  Note that "only its imaginary part" means the
imaginary *term* i Im W, not the real number Im W -- that is what makes
M_vac = M - M_comm equal to the anti-commutator contribution rather than
some mixture, and it is TMM21's Eq. (31) exactly.

Implementation
--------------
Both halves of W are exponential sums, which is all
``vacuum.detectors.udw.pair_term`` needs.  For W(dt) = sum_k c_k e^{-i w_k dt}
(the common M2.1 :class:`~vacuum.detectors.kernels.ExpSumKernel` form):

    i Im W(dt) = (W(dt) - conj(W(dt)))/2
               = sum_k [ (c_k/2) e^{-i w_k dt} + (-conj(c_k)/2) e^{-i(-w_k) dt} ],
    Re W(dt)   = sum_k [ (c_k/2) e^{-i w_k dt} + ( conj(c_k)/2) e^{-i(-w_k) dt} ],

i.e. the same kernel on the doubled frequency set {+w_k, -w_k}, exactly as
the M2.3 docstring anticipated ("the imaginary (commutator) part of W is
itself an exponential sum on the doubled frequency set").  The derived
kernels forward ``with_refined`` to their base when the base has one
(continuum backend), so ``pair_term``'s k-refinement gate keeps working on
the split pieces.

Numerical hazard (logged).  ``pair_term``'s doubling gate stops when
successive quadratures agree to ``max(rtol*|value|, atol)``.  For
spacelike-separated compact switchings the commutator integral is
analytically zero and numerically a cancellation residue of order 1e-19,
so a *relative* criterion can never be met and the gate would run out of
halvings.  :func:`communication_split` therefore gives the commutator run
an absolute tolerance tied to the scale of the full M,
``comm_atol = max(atol, rtol * |M|)``: M_vac = M - M_comm only ever needs
both pieces to the same *absolute* accuracy.  Pass ``comm_atol``
explicitly to tighten it (the spacelike causality test does).

Everything is pure-functional over frozen arrays -- no mutation, no hidden
state (Layer 6 JAX mirror pre-positioning).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict

import numpy as np

from .kernels import ExpSumKernel
from .udw import (
    pair_kernels,
    pair_negativity,
    pair_term,
    udw_pair_state,
)

__all__ = [
    "CommunicationSplit",
    "WightmanPart",
    "RefinableWightmanPart",
    "commutator_kernel",
    "hadamard_kernel",
    "commutator_term",
    "communication_split",
    "communication_estimator",
    "pair_state_with_split",
    "communication_columns",
]


# ---------------------------------------------------------------------------
# The two halves of W as exponential-sum kernels
# ---------------------------------------------------------------------------

class WightmanPart(ExpSumKernel):
    """The commutator or anti-commutator half of a Wightman kernel.

    ``part='commutator'`` is i Im W = (1/2) C^- [TMM21 Eq. (24)];
    ``part='hadamard'`` is Re W = (1/2) C^+ [Eq. (23)].  Both are built on
    the doubled frequency set {+w_k, -w_k} of the base kernel (module
    docstring), so they are ordinary
    :class:`~vacuum.detectors.kernels.ExpSumKernel` objects that drop into
    every M2.3 quadrature unchanged.

    The base kernel is kept (read-only) for provenance;
    :class:`RefinableWightmanPart` additionally forwards ``with_refined``.
    """

    __slots__ = ("base", "part")

    _PARTS = ("commutator", "hadamard")

    def __init__(self, base, part):
        if part not in self._PARTS:
            raise ValueError(f"part must be one of {self._PARTS}, got {part!r}")
        freqs = np.asarray(base.freqs, dtype=float)
        amps = np.asarray(base.amps, dtype=complex)
        sign = -1.0 if part == "commutator" else +1.0
        super().__init__(
            np.concatenate([freqs, -freqs]),
            np.concatenate([0.5 * amps, sign * 0.5 * np.conj(amps)]),
        )
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "part", part)

    def __repr__(self):
        return f"WightmanPart({self.part!r}, base={type(self.base).__name__})"


class RefinableWightmanPart(WightmanPart):
    """:class:`WightmanPart` over a base kernel that carries its own quadrature.

    Forwarding ``with_refined`` keeps ``pair_term``'s kernel-refinement gate
    meaningful for the continuum 3+1 backend: the split is taken *after*
    refinement, so both halves are converged in k as well as in time.
    """

    __slots__ = ()

    def with_refined(self, factor=2):
        return RefinableWightmanPart(self.base.with_refined(factor), self.part)

    def with_kmax(self, kmax):
        return RefinableWightmanPart(self.base.with_kmax(kmax), self.part)


def _part_kernel(kernel, part):
    if hasattr(kernel, "with_refined"):
        return RefinableWightmanPart(kernel, part)
    return WightmanPart(kernel, part)


def commutator_kernel(kernel):
    """i Im W = (1/2) C^- of a Wightman kernel [TMM21 Eqs. (22), (24)].

    State-independent (it is fixed by the field's canonical commutation
    relations alone) and supported only on and inside the light cone, which
    is what makes the M-integral built on it the *signalling* part of M.
    """
    return _part_kernel(kernel, "commutator")


def hadamard_kernel(kernel):
    """Re W = (1/2) C^+ of a Wightman kernel [TMM21 Eqs. (22), (23)].

    The state-dependent half: the genuine field correlations, the only ones
    two spacelike-separated detectors can possibly harvest.
    """
    return _part_kernel(kernel, "hadamard")


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------

def commutator_term(
    kernel_AB, chi_A, chi_B, gap_A, gap_B, lam_A=1.0, lam_B=1.0, **quad_kwargs
):
    """M^- : the pair term with the Wightman kernel replaced by i Im W.

    TMM21 Eq. (33).  A thin wrapper on
    :func:`vacuum.detectors.udw.pair_term` -- the M-integral is linear in
    the kernel, so nothing but the kernel changes.
    """
    return pair_term(
        commutator_kernel(kernel_AB),
        chi_A,
        chi_B,
        gap_A,
        gap_B,
        lam_A,
        lam_B,
        **quad_kwargs,
    )


@dataclass(frozen=True)
class CommunicationSplit:
    """M = M_vac + M_comm, with the columns every Layer 2 result row carries.

    Unpacks as ``(M_vac, M_comm)`` so it satisfies the spec's
    ``communication_split(...) -> (M_vac, M_comm)`` signature literally
    while still carrying M, the fraction and the quadrature provenance.

    Attributes
    ----------
    M : complex        full pair term [TMM21 Eq. (16)]
    M_vac : complex    M^+, anti-commutator / genuine harvesting [Eq. (32)]
    M_comm : complex   M^-, commutator / field-mediated signalling [Eq. (33)]
    meta : dict        quadrature settings at convergence, per-piece
    """

    M: complex
    M_vac: complex
    M_comm: complex
    meta: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        yield self.M_vac
        yield self.M_comm

    @property
    def comm_fraction(self):
        """|M_comm| / |M| -- the permanent results-table column (spec M2.5).

        0.0 when M itself vanishes (nothing to attribute).  Note this is a
        ratio of *matrix elements*; TMM21's estimator I [Eq. (39)] is a
        ratio of negativities and lives in :func:`communication_estimator`.
        """
        denom = abs(self.M)
        return 0.0 if denom == 0.0 else abs(self.M_comm) / denom

    @property
    def vac_fraction(self):
        """|M_vac| / |M| (the two fractions need not sum to 1: M^+ and M^-
        are complex and add as vectors, not magnitudes -- TMM21 call the
        contributions sub-additive)."""
        denom = abs(self.M)
        return 0.0 if denom == 0.0 else abs(self.M_vac) / denom

    @property
    def columns(self):
        """The three M-columns as a plain dict (see :func:`communication_columns`)."""
        return {
            "M_vac": self.M_vac,
            "M_comm": self.M_comm,
            "comm_fraction": self.comm_fraction,
        }


def communication_split(
    kernel,
    det_A,
    det_B,
    lam=1.0,
    gap=0.0,
    M=None,
    comm_atol=None,
    cross_check=False,
    coupling="xx",
    **quad_kwargs,
):
    """Split the pair term into harvesting and communication (TMM21 Eq. (31)).

    Runs the M-integral twice: once with the full Wightman kernel and once
    with only its imaginary (commutator) term i Im W.  ``M_comm`` is the
    second; ``M_vac = M - M_comm`` is then the anti-commutator (Hadamard)
    contribution, i.e. the part that can be attributed to pre-existing
    field correlations.

    Parameters
    ----------
    kernel, det_A, det_B, lam, gap
        Exactly as in :func:`vacuum.detectors.udw.udw_pair_state`; the
        cross kernel is resolved by
        :func:`vacuum.detectors.udw.pair_kernels`, so both M2.1 backends
        (lattice 1+1, continuum 3+1) plug in.
    M : complex, optional
        A previously converged full pair term for these arguments.  Given,
        the full M-integral is not rerun (this is how
        :func:`pair_state_with_split` avoids computing M twice).
    comm_atol : float, optional
        Absolute tolerance for the commutator quadrature.  Defaults to
        ``max(atol, rtol * |M|)`` -- see the module docstring's hazard
        note: for spacelike separation M_comm is analytically zero and a
        purely relative criterion cannot converge.
    cross_check : bool
        Also integrate the Hadamard kernel directly and record
        ``|M_vac_direct - (M - M_comm)|`` in ``meta['cross_check']``.
        Costs a third M-integral; off by default.
    coupling : {'xx', 'xp'}
        Amplitude ('xx') or derivative ('xp') coupling: the latter runs the
        identical split on the pi-pi kernel d_t d_t' W (its commutator half
        is d_t d_t' of the Pauli-Jordan function — still state-independent,
        still supported on and inside the light cone).  This is the split
        of Teixido-Bonfill & Martin-Martinez, PRD 110, 105016 (2024),
        Eq. (16) [M = M^+ + M^-], and of arXiv:2505.01516 Eq. (63); see
        ``vacuum.detectors.udw_derivative``.
    quad_kwargs
        Forwarded to :func:`vacuum.detectors.udw.pair_term` (``rtol``,
        ``atol``, ``n_gl``, ``n_gl_inner``, ``max_doublings``,
        ``refine_kernel``, ``window``, ``panel0``, ``chunk``).

    Returns
    -------
    CommunicationSplit
        Unpacks as ``(M_vac, M_comm)``.
    """
    from .udw import _pair_params  # normalization shared with the M2.3 engine

    lam_A, lam_B, gap_A, gap_B = _pair_params(lam, gap)
    _, _, ker_AB = pair_kernels(kernel, det_A, det_B, coupling=coupling)

    rtol = float(quad_kwargs.get("rtol", 1e-8))
    atol = float(quad_kwargs.get("atol", 1e-18))

    meta: Dict[str, Any] = {
        "source": "Tjoa & Martin-Martinez, PRD 104, 125005 (2021), "
        "arXiv:2109.11561, Eqs. (22)-(33)",
        "backend": type(kernel).__name__,
        "coupling": coupling,
    }

    if M is None:
        M, meta_full = pair_term(
            ker_AB, det_A.chi, det_B.chi, gap_A, gap_B, lam_A, lam_B,
            return_meta=True, **quad_kwargs,
        )
        meta["full"] = meta_full
    else:
        M = complex(M)
        meta["full"] = "supplied"

    if comm_atol is None:
        comm_atol = max(atol, rtol * abs(M))
    comm_kwargs = dict(quad_kwargs)
    comm_kwargs["atol"] = float(comm_atol)
    M_comm, meta_comm = pair_term(
        commutator_kernel(ker_AB), det_A.chi, det_B.chi, gap_A, gap_B,
        lam_A, lam_B, return_meta=True, **comm_kwargs,
    )
    meta["commutator"] = meta_comm
    meta["comm_atol"] = float(comm_atol)

    M_vac = M - M_comm

    if cross_check:
        M_vac_direct, meta_had = pair_term(
            hadamard_kernel(ker_AB), det_A.chi, det_B.chi, gap_A, gap_B,
            lam_A, lam_B, return_meta=True, **quad_kwargs,
        )
        meta["hadamard"] = meta_had
        meta["cross_check"] = abs(M_vac_direct - M_vac)
        meta["M_vac_direct"] = M_vac_direct

    return CommunicationSplit(M=M, M_vac=M_vac, M_comm=M_comm, meta=meta)


def communication_estimator(P_A, P_B, M_vac, M_comm):
    """TMM21's harvested / communication-assisted negativities and estimator.

    N^{+/-} = max(0, sqrt(|M^{+/-}|^2 + (P_A - P_B)^2/4) - (P_A + P_B)/2),
    which is PKMM Eqs. (66)-(67) with M replaced by M^{+/-} and reduces to
    TMM21's ``max(0, |M^{+/-}| - L_jj)`` for identical detectors, and

        I[rho_AB] = N^- / N   if N > 0, else 0                 [Eq. (39)].

    I ~ 1 means essentially all of the entanglement came from signalling
    (not harvesting); I = 0 with N > 0 means all of it was harvested.

    Returns
    -------
    dict with keys ``N``, ``N_vac``, ``N_comm``, ``estimator``.
    """
    M = complex(M_vac) + complex(M_comm)
    N = pair_negativity(P_A, P_B, M)
    N_vac = pair_negativity(P_A, P_B, M_vac)
    N_comm = pair_negativity(P_A, P_B, M_comm)
    return {
        "N": N,
        "N_vac": N_vac,
        "N_comm": N_comm,
        "estimator": (N_comm / N) if N > 0.0 else 0.0,
    }


def pair_state_with_split(kernel, det_A, det_B, lam, gap, coupling="xx", **quad_kwargs):
    """:func:`~vacuum.detectors.udw.udw_pair_state` with the M2.5 columns filled.

    Computes the second-order pair state once, then reruns only the
    commutator M-integral, and attaches the resulting
    :class:`CommunicationSplit` to the state's ``comm`` field.  From here
    on every harvested E_N can be reported next to its (M_vac, M_comm,
    |M_comm|/|M|) provenance, which is what the spec's permanent
    results-table schema requires -- see ``UDWPairState.row``.

    ``quad_kwargs`` are the options
    :func:`~vacuum.detectors.udw.udw_pair_state` accepts (``rtol``,
    ``atol``, ``n_gl``, ``n_gl_inner``, ``max_doublings``,
    ``refine_kernel``, ``chunk``) plus this module's ``comm_atol`` and
    ``cross_check``; ``window``/``panel0`` are per-integral overrides and
    only reach :func:`communication_split` when it is called directly.
    """
    split_keys = ("comm_atol", "cross_check")
    split_only = {k: quad_kwargs.pop(k) for k in split_keys if k in quad_kwargs}
    state = udw_pair_state(
        kernel, det_A, det_B, lam, gap, coupling=coupling, **quad_kwargs
    )
    split = communication_split(
        kernel, det_A, det_B, lam, gap, M=state.M, coupling=coupling,
        **split_only, **quad_kwargs,
    )
    return replace(state, comm=split)


def communication_columns(state):
    """The permanent M2.5 results-table columns of a state or a split.

    ``(E_N, M_vac, M_comm, |M_comm|/|M|)`` -- E_N only when the argument
    carries a negativity (a :class:`~vacuum.detectors.udw.UDWPairState`).
    """
    if isinstance(state, CommunicationSplit):
        return dict(state.columns)
    cols = {"E_N": float(state.log_negativity)}
    comm = getattr(state, "comm", None)
    if comm is None:
        cols.update({"M_vac": None, "M_comm": None, "comm_fraction": None})
    else:
        cols.update(comm.columns)
    return cols
