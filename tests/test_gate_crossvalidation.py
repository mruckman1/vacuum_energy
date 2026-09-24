"""GATE A — perturbative UDW vs nonperturbative Gaussian harvesting (M2.4).

This is the load-bearing cross-validation of Layer 2
(docs/PLAN_LAYERS_2_3.md, M2.4).  The *same* physical setup is computed
two ways:

* **nonperturbative** (M2.2, ``vacuum.detectors.nonperturbative``):
  harmonic-oscillator detectors appended to the lattice field's coupling
  matrix, evolved exactly (to Magnus-truncation error, gated by
  dt-halving) as a Gaussian state, negativity from the partial-transpose
  symplectic spectrum;
* **perturbative** (M2.3, ``vacuum.detectors.udw``): the second-order
  two-*qubit* UDW state of Pozas-Kerstjens & Martin-Martinez, PRD 92,
  064042 (2015) [PKMM], driven by the *same* lattice Wightman kernel
  (M2.1 ``LatticeWightman.smeared``), same switching, same smearing.

The two engines share no code past ``vacuum.core``: one cannot have a
convention error (it is an exact covariance evolution of a manifestly
correct quadratic form), the other is an independent transcription of a
published second-order expansion.  Requirement (spec M2.4): E_N agreement
with relative deviation bounded by c*lambda^2 over a lambda sweep — fit
the deviation's lambda-scaling and assert the exponent >= 2 with
R^2 > 0.999.


The two-level <-> oscillator leading-order mapping
==================================================

The two models are genuinely different (an oscillator has |2>, |3>, ...
and a qubit does not), so the correspondence has to be *derived*, not
assumed.  Everything below is self-contained and elementary; no published
equation number is claimed for the derivation itself (repo cite-or-derive
discipline).  Conventions are docs/API.md: hbar = 1, V_vac = I/2,
R = (x_1..x_N, p_1..p_N), nats.

**1. The two Hamiltonians.**  ``attach_detectors`` puts the coupling in
the off-diagonal block of K_tot, i.e. H = (1/2)(p.p + x^T K_tot x)
contains, for detector nu,

    H_int^osc(t) = lambda_nu(t) x_nu  Phi_nu,    Phi_nu = sum_i F_nu,i x_i,

with a detector block Omega_nu^2, i.e. H_nu = (1/2)(p_nu^2 + Omega_nu^2 x_nu^2).
The UDW model [PKMM Eq. (1)-(2)] uses instead

    H_int^UDW(t) = lambda_nu^UDW chi_nu(t) mu_nu(t) Phi_nu(t),
    mu_nu(t) = sigma^+_nu e^{+i Omega_nu t} + sigma^-_nu e^{-i Omega_nu t}.

In the interaction picture the oscillator monopole is

    x_nu(t) = (a_nu e^{-i Omega_nu t} + a_nu^dag e^{+i Omega_nu t}) / sqrt(2 Omega_nu),

which is exactly mu_nu(t)/sqrt(2 Omega_nu) under a_nu <-> sigma^-_nu.  Hence
the **coupling map**

    lambda_nu^UDW = lambda_nu^osc / sqrt(2 Omega_nu)        (MAP-1)

— note the per-detector Omega_nu, which setup 'B' below (unequal gaps)
tests independently of any overall normalization.  No rotating-wave
approximation is made on either side: both monopoles carry e^{+i Omega t}
*and* e^{-i Omega t}.

**2. Second-order moments.**  With |psi> = T exp(-i int H_int) |0,0,vac>
expanded to second order, the detector-pair moments of the oscillator
model are, term by term, the PKMM matrix elements of Eq. (13):

    n_nu   = <a_nu^dag a_nu>  = P_nu = L_nunu   [PKMM Eqs. (15), (17)]
    m      = <a_A a_B>        = M             [PKMM Eqs. (16), (18)]
    c      = <a_A^dag a_B>    = conj(C) = conj(L_AB)  [PKMM Eqs. (15), (17)]

after (MAP-1).  (n_nu: the first-order one-excitation amplitude squared,
with W(t'-t) = conj(W(t-t')) turning the norm into PKMM's P_nu.  m: only
the a_A^dag a_B^dag component of |psi^(2)> contributes, which is the
time-ordered triangle integral of Eq. (18) verbatim.  c: the overlap of
the two first-order branches, which is Eq. (17) conjugated — only |c| is
compared here, so the conjugation is immaterial.)

The oscillator model additionally carries **local squeezing**
s_nu = <a_nu a_nu> (the a_nu^dag a_nu^dag component of |psi^(2)>), which
has *no two-level counterpart* — sigma^+ sigma^+ = 0.  It is not small:
in the setups below |s_nu| is 15-40x larger than n_nu.  Step 4 shows why
it nevertheless drops out.

**3. Negativity of a near-vacuum two-mode Gaussian state.**  Rescale each
detector mode by the local symplectic X = sqrt(Omega) x, P = p/sqrt(Omega)
(so a = (X + iP)/sqrt(2), vacuum covariance I/2; local symplectics change
neither symplectic eigenvalues nor E_N).  Writing V = I/2 + delta with
delta = O(lambda^2), the symplectic eigenvalues near the vacuum floor are
nu_k = 1/2 + eig(Q) with, by first-order degenerate perturbation theory on
i*Omega*V about i*Omega/2 (whose +1/2 right/left eigenvectors are
(u, -i u) and (a, i a)/2),

    Q = (1/2) [ (delta_xx + delta_pp) - i (delta_xp - delta_xp^T) ].

Substituting the moment parametrization of delta and partial-transposing
on B (P_B -> -P_B) gives

    Q^Gamma = [[ n_A, m ], [ conj(m), n_B ]],

so that

    nu~_min = 1/2 + (n_A + n_B)/2 - sqrt(|m|^2 + (n_A - n_B)^2 / 4),
    E_N     = -ln(2 nu~_min)
            = ln(1 + 2 [sqrt(|m|^2 + (n_A-n_B)^2/4) - (n_A+n_B)/2]),

which under step 2 is *identically* PKMM Eqs. (66)-(67) — the
``pair_negativity`` / ``pair_log_negativity`` of ``vacuum.detectors.udw``.
(Simon's PT criterion, PRL 84, 2726 (2000); E_N per Vidal & Werner,
PRA 65, 032314 (2002), in nats, docs/API.md.)

**4. Why the oscillator's extra structure cancels.**  In Q the local
squeezing s_nu enters delta_xx and delta_pp with opposite signs
(<X_nu^2> = n + 1/2 + Re s, <P_nu^2> = n + 1/2 - Re s), so it cancels in
delta_xx + delta_pp; its Im part sits on the *diagonal* of delta_xp, which
the antisymmetrizer delta_xp - delta_xp^T kills.  The exchange term c
survives in Q but is removed by the partial transpose (it is |m|, not |c|,
that appears in Q^Gamma) — mirroring the fact that C does not enter PKMM
Eq. (67) either.  So the oscillator and two-level detectors agree at
leading order *despite* differing at the same order in observables that
are not the negativity: the correspondence is a statement about E_N, and
this file checks it as one.

**5. Order of the residual.**  Both E_N's are O(lambda^2); the first
neglected terms are O(lambda^4) on both sides, so the *relative* deviation
is O(lambda^2) — the spec's c*lambda^2 law.  Both engines report E_N in the
resummed ln(1 + 2N) / -ln(2 nu~) form, so the ln nonlinearity itself
cancels between them and does not pollute the exponent.

**6. Picture and phase.**  The nonperturbative covariance is a
Schroedinger-picture object at the end of the protocol; the UDW state is
an interaction-picture object anchored at t = 0.  They differ by free
detector rotation a_nu -> a_nu e^{-i Omega_nu t}, a local symplectic:
n_nu, |m|, |c| and E_N are all invariant under it, which is why only
moduli are compared for m and c.


What a failure means
====================
A slip in ``udw.py`` (a kernel-argument sign, a missing sqrt(2 Omega), a
mis-ordered triangle) or in ``nonperturbative.py`` breaks the lambda^2 law
outright — the deviation goes to O(1) and the exponent collapses toward 0.
Four such slips are exhibited as canaries at the end of this file, so the
gate's teeth are themselves regression-tested.  (The suite was mutation-
tested against ten injected slips in ``udw.py``, ``kernels.py`` and
``nonperturbative.py`` — dropped Eq. (18) bracket half, negated gap phases
singly and jointly, conjugated Wightman kernel in M and in C, kernel
amplitude 1/omega instead of 1/2omega, response-phase sign, doubled
``attach_detectors`` off-diagonal block, Omega instead of Omega^2 on the
detector diagonal, and a shifted ``detector_occupations`` — and every one
of them turns the suite red.)

One of them dictates the choice of setup 'C'.  Only |M| enters the
negativity [PKMM Eq. (67)], and |M| is *invariant* under negating both
detector gap phases in the time-ordered integrand of Eq. (18) whenever the
configuration has a time-reflection symmetry — i.e. whenever the two
switchings share a reflection axis and Omega_A = Omega_B.  (PKMM's own
closed form Eq. (31) shows the same blindness: alpha = Omega T enters only
through the even prefactor e^{-alpha^2/2}.)  Setup 'C' therefore carries
unequal gaps *and* displaced switchings; there the slip moves |M| by more
than a factor of two and the gate fails loudly, as
``test_canary_pair_term_gap_phase_sign`` pins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pytest
from scipy.linalg import expm

from vacuum.core import Omega as Omega_form
from vacuum.core import harmonic_chain_K
from vacuum.detectors import (
    ExpSumKernel,
    UDWDetector,
    detector_occupations,
    pair_kernels,
    pair_term,
    run_harvesting,
    smearing,
    switching,
    udw_pair_state,
    wightman_lattice,
)

# --------------------------------------------------------------------------
# Sweep configuration
# --------------------------------------------------------------------------

#: The lambda sweep (geometric, factor 6 end to end).  Lower end: the
#: deviation must stay far above the numerics floor set by CONV_REL_TOL.
#: Upper end: every matrix element of the perturbative state must remain
#: small (asserted through UDWPairState.meta['perturbative_ok']).
LAMBDAS: Tuple[float, ...] = (0.07, 0.095, 0.13, 0.175, 0.24, 0.32, 0.42)

#: dt-halving gate tolerance for the nonperturbative runs, as a *fraction*
#: of the perturbative E_N at that lambda.  A relative (not absolute) gate
#: is what keeps the Magnus discretization error a lambda-independent
#: relative bias, which would otherwise tilt the fitted exponent: with
#: conv_tol ~ 2e-6 * E_N the residual bias is ~1e-6 relative, i.e. ~0.1% of
#: the smallest deviation measured here (~1e-3).
CONV_REL_TOL = 2.0e-6

#: Spec M2.4 requirement.
GATE_EXPONENT = 2.0
GATE_R2 = 0.999

#: Bound asserted on (matrix-element discrepancy)/lambda^2 for the
#: individual matrix elements (the spec's "bounded by c*lambda^2", with c
#: stated).  Largest value measured across the three setups is 1.21
#: (n_B vs P_B in setup 'C'); see test_matrix_elements_agree_at_leading_order
#: for the per-element normalization.
MATRIX_ELEMENT_C = 2.5


@dataclass(frozen=True)
class GateSetup:
    """One lattice setup, computed both ways.  Frozen data, no behavior."""

    name: str
    N: int
    mass: float
    bc: str
    profiles: Tuple[np.ndarray, np.ndarray]
    gaps: Tuple[float, float]
    centers: Tuple[float, float]
    width: float
    note: str

    @property
    def chis(self):
        return tuple(switching("cos2", self.width, t0) for t0 in self.centers)


def _point(N, site):
    F = np.zeros(N)
    F[site] = 1.0
    return F


def _setups() -> Dict[str, GateSetup]:
    """Three structurally distinct setups (spec M2.4: parameterized).

    'cos2' switching throughout: it has *exact* compact support, so
    lambda(t) vanishes identically at both ends of the evolution grid and
    the nonperturbative initial state (``initial_state`` at lambda = 0) is
    the true state at t0 — no switching-tail mismatch enters the
    comparison.
    """
    out = {}
    out["A"] = GateSetup(
        name="A",
        N=20,
        mass=0.6,
        bc="dirichlet",
        profiles=(_point(20, 8), _point(20, 9)),
        gaps=(2.0, 2.0),
        centers=(0.0, 0.0),
        width=2.0,
        note="pointlike, dirichlet, equal gaps, coincident switching",
    )
    out["B"] = GateSetup(
        name="B",
        N=16,
        mass=0.8,
        bc="periodic",
        profiles=(
            np.asarray(smearing("gaussian", 5.0, 1.0, 16, bc="periodic")),
            np.asarray(smearing("gaussian", 8.0, 1.0, 16, bc="periodic")),
        ),
        gaps=(2.5, 2.0),
        centers=(0.0, 0.0),
        width=2.0,
        # Unequal gaps make lambda^UDW_A != lambda^UDW_B under (MAP-1)
        # while the oscillator coupling is the same number for both
        # detectors: this tests the 1/sqrt(2 Omega_nu) factor's *Omega
        # dependence*, not just an overall normalization.  It also turns on
        # the (n_A - n_B)/2 term of the negativity formula.
        note="gaussian-smeared, periodic, unequal gaps, coincident switching",
    )
    out["C"] = GateSetup(
        name="C",
        N=20,
        mass=0.6,
        bc="dirichlet",
        profiles=(_point(20, 8), _point(20, 9)),
        gaps=(2.5, 2.0),
        centers=(0.0, 0.7),
        width=2.0,
        # Unequal gaps AND displaced switchings *together*.  Neither alone
        # suffices: PKMM Eq. (18)'s bracket is invariant under
        # (t, t') -> (t_A + t_B - t', t_A + t_B - t) combined with
        # Omega_nu -> -Omega_nu whenever the two switchings share a
        # reflection axis and Omega_A = Omega_B, so |M| — the only thing
        # the negativity sees — cannot detect a sign slip on the pair
        # term's gap phases in a symmetric configuration.  (PKMM's own
        # closed form Eq. (31) is likewise even in alpha = Omega T, so the
        # M2.3 anchors cannot see it either.)  This setup breaks that
        # symmetry; see test_canary_pair_term_gap_phase_sign.
        note="pointlike, unequal gaps AND displaced switchings",
    )
    return out


SETUPS = _setups()


# --------------------------------------------------------------------------
# Helpers (pure functions over arrays)
# --------------------------------------------------------------------------


def oscillator_moments(V_AB, gaps):
    """Complex second moments of a two-mode Gaussian block, zero mean.

    Returns ``dict(n_A, n_B, m, c, s_A, s_B)`` with

        n_nu = <a_nu^dag a_nu>,  m = <a_A a_B>,
        c    = <a_A^dag a_B>,    s_nu = <a_nu a_nu>,

    for a_nu = (sqrt(Omega_nu) x_nu + i p_nu / sqrt(Omega_nu)) / sqrt(2).
    The rescaling X = sqrt(Omega) x, P = p/sqrt(Omega) is the local
    symplectic diag(G, G^-1) of the module docstring (step 3); in the
    rescaled block (docs/API.md ordering (X_A, X_B, P_A, P_B)),

        <X_nu^2> = n + 1/2 + Re s,   <P_nu^2>   = n + 1/2 - Re s,
        <{X_nu,P_nu}>/2 = Im s,
        <X_A X_B> = Re m + Re c,     <P_A P_B>  = -Re m + Re c,
        <X_A P_B> = Im m + Im c,     <P_A X_B>  =  Im m - Im c,

    which inverts to the expressions below.  Pure function; V_AB is not
    mutated.
    """
    g = np.asarray(gaps, dtype=float)
    if g.shape != (2,):
        raise ValueError(f"gaps must be a pair, got shape {g.shape}")
    V_AB = np.asarray(V_AB, dtype=float)
    if V_AB.shape != (4, 4):
        raise ValueError(f"V_AB must be a two-mode (4, 4) block, got {V_AB.shape}")
    D = np.diag(np.concatenate([np.sqrt(g), 1.0 / np.sqrt(g)]))
    W = D @ V_AB @ D
    xx, pp, xp = W[:2, :2], W[2:, 2:], W[:2, 2:]
    return {
        "n_A": 0.5 * (xx[0, 0] + pp[0, 0] - 1.0),
        "n_B": 0.5 * (xx[1, 1] + pp[1, 1] - 1.0),
        "m": complex(0.5 * (xx[0, 1] - pp[0, 1]), 0.5 * (xp[0, 1] + xp[1, 0])),
        "c": complex(0.5 * (xx[0, 1] + pp[0, 1]), 0.5 * (xp[0, 1] - xp[1, 0])),
        "s_A": complex(0.5 * (xx[0, 0] - pp[0, 0]), xp[0, 0]),
        "s_B": complex(0.5 * (xx[1, 1] - pp[1, 1]), xp[1, 1]),
    }


def power_fit(lams, devs):
    """Least-squares fit of log(dev) = exponent*log(lambda) + log(prefactor).

    Returns (exponent, R^2, prefactor).  R^2 = 1 - SS_res/SS_tot in the
    log-log variables, which is the quantity the spec's "R^2 > 0.999"
    refers to (a power law is a straight line there).
    """
    x = np.log(np.asarray(lams, dtype=float))
    y = np.log(np.asarray(devs, dtype=float))
    if x.size < 3:
        raise ValueError("need >= 3 sweep points for a meaningful R^2")
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(coef[0]), float(1.0 - np.sum(resid**2) / ss_tot), float(np.exp(coef[1]))


def _pert_state(setup, lam, kernel=None, lam_map=True):
    """Perturbative UDW pair state at oscillator coupling `lam` (MAP-1)."""
    chi_A, chi_B = setup.chis
    det_A = UDWDetector(chi_A, setup.profiles[0])
    det_B = UDWDetector(chi_B, setup.profiles[1])
    if lam_map:
        lam_udw = tuple(lam / math.sqrt(2.0 * g) for g in setup.gaps)
    else:  # canary: the sqrt(2 Omega) of (MAP-1) omitted
        lam_udw = (lam, lam)
    if kernel is None:
        kernel = wightman_lattice(harmonic_chain_K(setup.N, setup.mass, bc=setup.bc))
    return udw_pair_state(
        kernel, det_A, det_B, lam_udw, setup.gaps, rtol=1e-11, max_doublings=16
    )


@dataclass(frozen=True)
class SweepData:
    """One setup's paired sweep: perturbative states + audited harvests."""

    setup: GateSetup
    lams: np.ndarray
    pert: Tuple[Any, ...]
    harv: Tuple[Any, ...]
    mom: Tuple[Dict[str, Any], ...]

    def deviation(self, np_values, pt_values):
        """|X_np - X_pt| / |X_pt| elementwise, as an ndarray."""
        a = np.asarray(np_values, dtype=float)
        b = np.asarray(pt_values, dtype=float)
        return np.abs(a - b) / np.abs(b)

    @property
    def E_N_pert(self):
        return np.array([s.log_negativity for s in self.pert])

    @property
    def E_N_nonpert(self):
        return np.array([h.E_N for h in self.harv])

    @property
    def E_N_deviation(self):
        return self.deviation(self.E_N_nonpert, self.E_N_pert)


def _run_sweep(setup: GateSetup) -> SweepData:
    """Run both engines on `setup` across LAMBDAS.

    The nonperturbative side goes through ``run_harvesting`` — the audited
    M2.2 stack (mandatory passivity audit on the claimed T = 0 ground
    state, dt-halving convergence gate, closing EnergyLedger, mp-margin
    rule on E_N) — so every point of this gate is produced by exactly the
    code path a Layer 2 sweep row would use, with no bypass.
    """
    K = harmonic_chain_K(setup.N, setup.mass, bc=setup.bc)
    kernel = wightman_lattice(K)
    chi_A, chi_B = setup.chis
    profiles = [np.asarray(F, dtype=float) for F in setup.profiles]
    t_start = min(chi_A.support[0], chi_B.support[0])
    t_end = max(chi_A.support[1], chi_B.support[1])
    ts = np.array([t_start, t_end])

    pert, harv, mom = [], [], []
    for lam in LAMBDAS:
        st = _pert_state(setup, lam, kernel=kernel)

        def lambda_of_t(t, _l=lam):
            return np.array([_l * chi_A(t), _l * chi_B(t)])

        def dlambda_of_t(t, _l=lam):
            return np.array([_l * chi_A.derivative(t), _l * chi_B.derivative(t)])

        hr = run_harvesting(
            K,
            profiles,
            setup.gaps,
            lambda_of_t,
            ts,
            dlambda_of_t=dlambda_of_t,
            conv_tol=CONV_REL_TOL * st.log_negativity,
            max_halvings=18,
        )
        pert.append(st)
        harv.append(hr)
        mom.append(oscillator_moments(hr.V_AB, setup.gaps))
    return SweepData(
        setup=setup,
        lams=np.array(LAMBDAS, dtype=float),
        pert=tuple(pert),
        harv=tuple(harv),
        mom=tuple(mom),
    )


@pytest.fixture(scope="module")
def sweeps():
    """Module-scoped cache: the three paired sweeps (~25 s total)."""
    return {name: _run_sweep(setup) for name, setup in SETUPS.items()}


NAMES = tuple(SETUPS)


# --------------------------------------------------------------------------
# The gate itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_gate_a_deviation_exponent(sweeps, name):
    """GATE A (spec M2.4): deviation exponent >= 2 in lambda, R^2 > 0.999.

    Nothing in M2.5+ is admissible unless this passes: it validates the
    PKMM transcription against machinery that cannot carry a convention
    error, and the exact engine against an independent derivation.
    """
    sw = sweeps[name]
    dev = sw.E_N_deviation
    assert np.all(dev > 0.0), f"[{name}] zero deviation — engines are not independent"
    exponent, r2, c = power_fit(sw.lams, dev)
    assert exponent >= GATE_EXPONENT, (
        f"[{name}: {sw.setup.note}] Gate A FAILED: relative-deviation exponent "
        f"{exponent:.6f} < {GATE_EXPONENT} (R^2 = {r2:.8f}, c = {c:.4f}); "
        f"deviations {np.array2string(dev, precision=4)} at lambda "
        f"{np.array2string(sw.lams, precision=3)}"
    )
    assert r2 > GATE_R2, (
        f"[{name}: {sw.setup.note}] Gate A FAILED: power-law R^2 {r2:.8f} <= "
        f"{GATE_R2} (exponent {exponent:.6f}); the deviation is not a clean "
        f"power law — deviations {np.array2string(dev, precision=4)}"
    )


@pytest.mark.parametrize("name", NAMES)
def test_gate_a_deviation_bounded_by_c_lambda_squared(sweeps, name):
    """The spec's other half: the deviation is *bounded* by c*lambda^2.

    Asserting the exponent alone would pass on a deviation that is a clean
    power law but drifting; requiring dev/lambda^2 to stay inside a narrow
    band pins the prefactor too.  (Measured spread: 1.012-1.026.)
    """
    sw = sweeps[name]
    ratio = sw.E_N_deviation / sw.lams**2
    spread = float(np.max(ratio) / np.min(ratio))
    assert spread < 1.15, (
        f"[{name}] dev/lambda^2 spans {np.min(ratio):.4f}-{np.max(ratio):.4f} "
        f"(spread {spread:.4f}): the c*lambda^2 bound is drifting"
    )
    assert float(np.max(ratio)) < 3.0, (
        f"[{name}] c = max(dev/lambda^2) = {np.max(ratio):.4f} is not O(1)"
    )


@pytest.mark.parametrize("name", NAMES)
def test_pair_term_deviation_exponent(sweeps, name):
    """|m| = |<a_A a_B>| vs |M| [PKMM Eq. (18)] — the sharpest single check.

    M is the only matrix element that carries the time-ordered triangle,
    the kernel's imaginary part and both detectors' phases at once, so it
    is where a convention slip in ``udw.py`` would land.  Its relative
    deviation must obey the same lambda^2 law as E_N.
    """
    sw = sweeps[name]
    dev = sw.deviation(
        [abs(mm["m"]) for mm in sw.mom], [abs(st.M) for st in sw.pert]
    )
    exponent, r2, c = power_fit(sw.lams, dev)
    assert exponent >= 1.95, (
        f"[{name}] |m| vs |M|: deviation exponent {exponent:.6f} < 1.95 "
        f"(R^2 = {r2:.8f}, c = {c:.4f})"
    )
    assert r2 > GATE_R2, f"[{name}] |m| vs |M|: R^2 {r2:.8f} <= {GATE_R2}"


@pytest.mark.parametrize("name", NAMES)
def test_matrix_elements_agree_at_leading_order(sweeps, name):
    """n_nu <-> P_nu, |m| <-> |M|, |c| <-> |C| (docstring step 2).

    E_N could in principle agree by cancellation between wrong pieces;
    these four checks make the mapping itself the assertion.  Each
    discrepancy is required to be bounded by MATRIX_ELEMENT_C * lambda^2
    across the whole sweep, measured against that element's own natural
    scale:

    * n_nu and |m| are measured against themselves (ordinary relative
      deviation) — each is the dominant element of its own kind;
    * |c| is measured against sqrt(P_A P_B), the Cauchy-Schwarz bound on
      an exchange amplitude (|C| <= sqrt(P_A P_B) exactly, and likewise
      |c| <= sqrt(n_A n_B)).  |C| itself may sit far *below* that bound
      when the two detectors' response channels overlap poorly — in setup
      'C' it is 0.138 of it — while the O(lambda^4) correction to the
      exchange term is still set by the state's leading scale.  Dividing
      by |C| there would report a large number for a small absolute
      discrepancy; dividing by the Cauchy-Schwarz scale is the
      normalization that does not depend on an accidental near-zero.
    """
    sw = sweeps[name]
    cs_scale = np.array(
        [math.sqrt(st.P_A * st.P_B) for st in sw.pert], dtype=float
    )
    checks = {
        "n_A vs P_A": (
            [mm["n_A"] for mm in sw.mom],
            [st.P_A for st in sw.pert],
            None,
        ),
        "n_B vs P_B": (
            [mm["n_B"] for mm in sw.mom],
            [st.P_B for st in sw.pert],
            None,
        ),
        "|m| vs |M|": (
            [abs(mm["m"]) for mm in sw.mom],
            [abs(st.M) for st in sw.pert],
            None,
        ),
        "|c| vs |C| (per sqrt(P_A P_B))": (
            [abs(mm["c"]) for mm in sw.mom],
            [abs(st.C) for st in sw.pert],
            cs_scale,
        ),
    }
    for label, (np_vals, pt_vals, scale) in checks.items():
        diff = np.abs(np.asarray(np_vals, float) - np.asarray(pt_vals, float))
        dev = diff / (np.abs(np.asarray(pt_vals, float)) if scale is None else scale)
        ratio = dev / sw.lams**2
        assert float(np.max(ratio)) < MATRIX_ELEMENT_C, (
            f"[{name}] {label}: max(dev/lambda^2) = {np.max(ratio):.4f} exceeds "
            f"{MATRIX_ELEMENT_C}; deviations {np.array2string(dev, precision=4)}"
        )
        # ... and the mapping must actually converge, not merely stay bounded.
        assert dev[0] < 0.02, (
            f"[{name}] {label}: deviation {dev[0]:.4e} at the smallest lambda = "
            f"{sw.lams[0]}; the leading-order mapping does not hold"
        )


@pytest.mark.parametrize("name", NAMES)
def test_occupations_match_moment_extraction(sweeps, name):
    """``detector_occupations`` and :func:`oscillator_moments` agree exactly.

    Guards the helper this file's mapping checks are built on against a
    private convention drift: n_nu computed from the (2N_tot) covariance by
    ``vacuum.detectors.nonperturbative`` must equal n_nu computed from the
    reduced two-mode block here.

    Both routes form n = (Omega<x^2> + <p^2>/Omega - 1)/2 from covariance
    entries of order 1/2, so the identity holds only to the cancellation
    floor of that subtraction — ~1e-16 absolute, which at n ~ 1e-5 is ~1e-11
    *relative*.  The absolute tolerance is therefore the meaningful one.
    """
    sw = sweeps[name]
    gaps = np.asarray(sw.setup.gaps, dtype=float)
    for lam, hr, mm in zip(sw.lams, sw.harv, sw.mom):
        n_d = detector_occupations(hr.V, sw.setup.N, gaps)
        assert n_d[0] == pytest.approx(mm["n_A"], rel=1e-9, abs=1e-14), (
            f"[{name}] lambda={lam}: n_A mismatch {n_d[0]:.12e} vs {mm['n_A']:.12e}"
        )
        assert n_d[1] == pytest.approx(mm["n_B"], rel=1e-9, abs=1e-14), (
            f"[{name}] lambda={lam}: n_B mismatch {n_d[1]:.12e} vs {mm['n_B']:.12e}"
        )


def test_moment_extraction_against_complex_second_moments():
    """:func:`oscillator_moments` vs <R_i R_j> = V_ij + (i/2) Omega_ij.

    Independent route to the same complex moments (the standard Gaussian
    identity, docs/API.md ordering), checked on a random physical two-mode
    state built by squeezing + rotating + mixing the vacuum — so the helper
    is not validated only where it is used.
    """
    rng = np.random.default_rng(20260901)
    n = 2
    Om = Omega_form(n)
    for _ in range(8):
        # random symplectic: exp(Omega H) with H a random symmetric form
        H = rng.normal(size=(2 * n, 2 * n))
        H = 0.5 * (H + H.T)
        S = expm(Om @ H)
        V = 0.5 * S @ S.T
        gaps = (1.0, 1.0)  # already in (X, P) units
        mom = oscillator_moments(V, gaps)
        RR = V + 0.5j * Om  # <R_i R_j>; indices x_A, x_B, p_A, p_B
        m_ref = 0.5 * (RR[0, 1] + 1j * RR[0, 3] + 1j * RR[2, 1] - RR[2, 3])
        c_ref = 0.5 * (RR[0, 1] + 1j * RR[0, 3] - 1j * RR[2, 1] + RR[2, 3])
        nA_ref = 0.5 * (RR[0, 0] + 1j * RR[0, 2] - 1j * RR[2, 0] + RR[2, 2])
        assert mom["m"] == pytest.approx(m_ref, rel=1e-11, abs=1e-13)
        assert mom["c"] == pytest.approx(c_ref, rel=1e-11, abs=1e-13)
        assert mom["n_A"] == pytest.approx(nA_ref.real, rel=1e-11, abs=1e-13)


@pytest.mark.parametrize("name", NAMES)
def test_gaussian_negativity_reduces_to_pkmm_form(sweeps, name):
    """The Q^Gamma identity of docstring step 3, on the exact covariances.

    nu~_min = 1/2 + (n_A+n_B)/2 - sqrt(|m|^2 + (n_A-n_B)^2/4) is the
    first-order partial-transpose eigenvalue; feeding it the *oscillator*
    moments must reproduce the exact covariance E_N to O(lambda^2)
    relative.  This is the structural reason the two engines can agree,
    tested as such — and it is what shows that the local squeezing s_nu
    (measured here, 15-40x larger than n_nu, with no two-level
    counterpart) genuinely drops out.
    """
    sw = sweeps[name]
    for lam, hr, mm in zip(sw.lams, sw.harv, sw.mom):
        assert abs(mm["s_A"]) > 5.0 * mm["n_A"], (
            f"[{name}] lambda={lam}: |s_A| = {abs(mm['s_A']):.4e} is not the "
            f"large oscillator-only term this test claims to neutralize"
        )
        nu_first_order = (
            0.5
            + 0.5 * (mm["n_A"] + mm["n_B"])
            - math.hypot(abs(mm["m"]), 0.5 * (mm["n_A"] - mm["n_B"]))
        )
        E_N_first_order = -math.log(2.0 * nu_first_order)
        rel = abs(E_N_first_order - hr.E_N) / hr.E_N
        assert rel < 0.5 * lam**2, (
            f"[{name}] lambda={lam}: Q^Gamma prediction {E_N_first_order:.10e} vs "
            f"exact {hr.E_N:.10e} — relative {rel:.4e} is not O(lambda^2)"
        )


@pytest.mark.parametrize("name", NAMES)
def test_every_gate_point_is_audited(sweeps, name):
    """Gate B discipline: no gate point may bypass the standing audits.

    Each sweep row must carry a passed passivity audit on its claimed
    T = 0 ground state, a closed EnergyLedger, a converged dt-halving gate
    with the converged dt recorded, a precision (mp-margin) flag, and a
    perturbative state whose matrix elements are actually small.
    """
    sw = sweeps[name]
    for lam, hr, st in zip(sw.lams, sw.harv, sw.pert):
        assert hr.passivity is not None and hr.passivity.passed, (
            f"[{name}] lambda={lam}: passivity audit on the initial ground state "
            f"did not pass"
        )
        assert hr.ledger_result.passed, (
            f"[{name}] lambda={lam}: energy ledger did not close "
            f"({hr.ledger_result.details})"
        )
        assert hr.converged and hr.dt_converged > 0.0, (
            f"[{name}] lambda={lam}: dt-halving gate did not converge "
            f"(dt {hr.dt_converged:.3e}, halvings {hr.halvings})"
        )
        assert isinstance(hr.flagged, bool) and hr.neg.regime in ("float64", "mpmath")
        # E_N is comfortably above the vacuum floor: this gate measures a
        # live negativity, not a near-death one (that regime is M2.2's
        # mp-margin test, not a place to fit a power law).
        assert hr.neg.margin_min < -1e-7, (
            f"[{name}] lambda={lam}: PT margin {hr.neg.margin_min:.3e} is at the "
            f"vacuum floor — no harvested entanglement to compare"
        )
        assert st.meta["perturbative_ok"], (
            f"[{name}] lambda={lam}: perturbative scale "
            f"{st.meta['perturbative_scale']:.3e} is outside the second-order "
            f"regime; the gate would be comparing against an invalid expansion"
        )


# --------------------------------------------------------------------------
# Canaries: the gate must be able to *fail*
# --------------------------------------------------------------------------


def test_canary_missing_oscillator_to_qubit_normalization(sweeps):
    """Drop the 1/sqrt(2 Omega) of (MAP-1): the lambda^2 law must collapse.

    This is the exact slip the mapping is there to prevent, so it is
    regression-tested: with lambda^UDW = lambda^osc the perturbative E_N is
    wrong by the constant factor 2*Omega and the relative deviation becomes
    O(1) and lambda-independent (exponent -> 0).
    """
    sw = sweeps["A"]
    setup = sw.setup
    K = harmonic_chain_K(setup.N, setup.mass, bc=setup.bc)
    kernel = wightman_lattice(K)
    bad = np.array(
        [
            _pert_state(setup, lam, kernel=kernel, lam_map=False).log_negativity
            for lam in sw.lams
        ]
    )
    good = sw.E_N_pert
    # E_N ~ 2N with N ~ lambda_UDW^2, so the whole curve is scaled by 2*Omega
    # (up to the ln(1 + 2N) resummation, which bends it by ~1% at the top).
    scale = bad / good
    assert np.allclose(scale, 2.0 * setup.gaps[0], rtol=1e-2), (
        f"canary setup wrong: E_N ratio {scale} is not 2*Omega = "
        f"{2.0 * setup.gaps[0]}"
    )
    dev = sw.deviation(sw.E_N_nonpert, bad)
    exponent, r2, _ = power_fit(sw.lams, dev)
    assert np.all(dev > 0.5), (
        f"canary: dropping sqrt(2 Omega) left deviations {dev} below 0.5 — the "
        f"gate would not notice the slip"
    )
    assert exponent < 0.1, (
        f"canary: dropping sqrt(2 Omega) still yields exponent {exponent:.4f} "
        f"(R^2 {r2:.6f}); the gate's exponent test has no teeth"
    )


def test_canary_conjugated_wightman_kernel(sweeps):
    """Flip Im W (i.e. W -> W(-dt)): the harvesting signal must be destroyed.

    A sign slip on the kernel's imaginary part — the classic UDW
    convention error, and the one M2.1 separates out explicitly — swaps the
    detector's excitation and de-excitation channels.  Here P_A jumps by
    more than an order of magnitude and the negativity dies, so the gate
    cannot be passed with a conjugated kernel.
    """
    sw = sweeps["A"]
    setup = sw.setup
    kernel = wightman_lattice(harmonic_chain_K(setup.N, setup.mass, bc=setup.bc))
    chi_A, chi_B = setup.chis
    det_A = UDWDetector(chi_A, setup.profiles[0])
    det_B = UDWDetector(chi_B, setup.profiles[1])
    # W(dt) = sum_k c_k e^{-i w_k dt} with real c_k, so conj(W)(dt) = W(-dt)
    # is the same exponential sum on the negated frequency set.
    flipped = tuple(
        ExpSumKernel(-np.asarray(k.freqs), np.asarray(k.amps))
        for k in pair_kernels(kernel, det_A, det_B)
    )
    lam = sw.lams[0]
    ref = sw.pert[0]
    bad = _pert_state(setup, lam, kernel=flipped)
    assert bad.P_A > 10.0 * ref.P_A, (
        f"canary: conjugating W changed P_A only from {ref.P_A:.4e} to "
        f"{bad.P_A:.4e}"
    )
    assert bad.negativity_estimator < 0.0 <= bad.log_negativity, (
        f"canary: conjugating W left N^(2) = {bad.negativity_estimator:.4e} "
        f"positive; harvesting survived a kernel sign slip"
    )
    assert sw.E_N_nonpert[0] > 0.0, "nonperturbative reference lost its negativity"


def _abs_M(setup, gap_signs=(1.0, 1.0)):
    """|M| from ``pair_term`` on `setup`, with the gap phases optionally negated."""
    chi_A, chi_B = setup.chis
    kernel = wightman_lattice(harmonic_chain_K(setup.N, setup.mass, bc=setup.bc))
    ker_AB = pair_kernels(
        kernel, UDWDetector(chi_A, setup.profiles[0]), UDWDetector(chi_B, setup.profiles[1])
    )[2]
    gA = gap_signs[0] * setup.gaps[0]
    gB = gap_signs[1] * setup.gaps[1]
    return abs(
        pair_term(ker_AB, chi_A, chi_B, gA, gB, 1.0, 1.0, rtol=1e-11, max_doublings=16)
    )


def test_canary_pair_term_gap_phase_sign(sweeps):
    """Why setup 'C' must be asymmetric: |M| can be blind to a gap-sign slip.

    Negating both gap phases inside PKMM Eq. (18) — the slip that would
    follow from writing mu(t) with sigma^+ and sigma^- exchanged, i.e. from
    starting the detectors in |e> rather than |g> — leaves |M| *exactly*
    invariant whenever the configuration is time-reflection symmetric.
    Setups 'A' (equal gaps, coincident switchings) and 'B' (unequal gaps,
    but coincident switchings, so a common reflection axis survives) both
    show this: the reflection (t, t') -> (t_A + t_B - t', t_A + t_B - t)
    maps the bracket of Eq. (18) onto its gap-negated self up to a global
    phase, and only |M| is observable.

    Setup 'C' carries unequal gaps *and* displaced switchings, killing the
    symmetry: there the slip more than doubles |M|, which is what makes the
    gate able to see it at all.  This test locks that property in — if a
    future edit makes 'C' symmetric again, this fails rather than silently
    reopening the blind spot.
    """
    for name in ("A", "B"):
        setup = SETUPS[name]
        m_ok = _abs_M(setup)
        m_flip = _abs_M(setup, gap_signs=(-1.0, -1.0))
        assert m_flip == pytest.approx(m_ok, rel=1e-9), (
            f"[{name}] |M| is expected to be blind to a both-gap sign flip in a "
            f"time-reflection-symmetric configuration: {m_ok:.10e} vs "
            f"{m_flip:.10e} — the documented symmetry no longer holds"
        )

    setup = SETUPS["C"]
    m_ok = _abs_M(setup)
    m_flip = _abs_M(setup, gap_signs=(-1.0, -1.0))
    assert m_flip / m_ok > 2.0, (
        f"[C] setup has lost its asymmetry: the both-gap sign flip moves |M| "
        f"only from {m_ok:.4e} to {m_flip:.4e} (ratio {m_flip / m_ok:.4f}); the "
        f"gate would no longer detect that slip"
    )
    # ... and with that |M| the leading-order mapping to the exact
    # |m| = |<a_A a_B>| is destroyed at the smallest lambda of the sweep.
    sw = sweeps["C"]
    lam = float(sw.lams[0])
    scale = lam**2 / (2.0 * math.sqrt(setup.gaps[0] * setup.gaps[1]))
    m_exact = abs(sw.mom[0]["m"])
    good = abs(m_ok * scale - m_exact) / m_exact
    bad = abs(m_flip * scale - m_exact) / m_exact
    assert good < 0.02 < 0.5 < bad, (
        f"[C] canary inconclusive: correct |M| deviates {good:.3e} from the "
        f"exact |m|, flipped |M| deviates {bad:.3e}"
    )


def test_canary_single_gap_phase_sign(sweeps):
    """Negating only Omega_A in the pair term is visible in *every* setup.

    The companion to the test above: the asymmetric slip has no reflection
    symmetry to hide behind, so all three setups catch it (|M| moves by
    more than an order of magnitude).  Kept so that the symmetric-blindness
    result above is not mistaken for a general weakness of the gate.
    """
    for name in NAMES:
        setup = SETUPS[name]
        m_ok = _abs_M(setup)
        m_flip = _abs_M(setup, gap_signs=(-1.0, 1.0))
        assert m_flip / m_ok > 2.0, (
            f"[{name}] negating Omega_A alone moved |M| only from {m_ok:.4e} to "
            f"{m_flip:.4e} (ratio {m_flip / m_ok:.4f})"
        )
