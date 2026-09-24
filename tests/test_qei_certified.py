"""S8 anchors — the certified enclosure of C_4d(gaussian)/C_FE.

What these pin (draft/proof_4d_gaussian_constant.md, section 5):

(i)   the enclosure [lo, 1] returned by
      :func:`vacuum.inequalities.qei_certified.certified_lower_bound` contains
      the archived momentum-space value 0.45749073 of
      ``papers/qei-2d-sharp-constant/data/s6b_momentum_4d_worldline.json``
      (MANIFEST headline 0.4574904), and `lo` is close to it, not trivially
      small;
(ii)  a REAL mutation: the pair-creation kernel scaled by 1 + 1e-4 moves the
      rigorous lower bound ABOVE the archived value, so the mutated enclosure
      EXCLUDES it.  A second mutation (the verifier fed a covariance that is
      not a state) must be rejected by the interval Cholesky;
(iii) the formulation: the Gamma-series matrix elements of the phi basis agree
      with a direct 2D quadrature of the kernel that
      ``vacuum.inequalities.qei`` itself builds (``fhat_of(gaussian_f(1.0))``).

The whole module runs in well under 60 s.  The long ladder (n_basis up to 48)
lives in ``papers/qei-2d-sharp-constant/notebook_certified.py``; set
``QEI_CERT_SLOW=1`` to run its top rung here too.
"""

import math
import os

import mpmath as mp
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from vacuum.inequalities.qei import fhat_of, gaussian_f
from vacuum.inequalities.qei_certified import (
    CERTIFIED_PINNED_RATIO,
    FEWSTER_EVESON_RATIO_UPPER_BOUND,
    certified_enclosure,
    certified_lower_bound,
    phi_gram,
    phi_kernel_elements,
    ratio_from_energy,
)

# MANIFEST headline and the momentum-grid uncertainty it is quoted with.
MANIFEST_HEADLINE = 0.4574904
MOMENTUM_GRID_UNCERTAINTY = 2e-6

N_FAST = 20
DPS_FAST = 252


@pytest.fixture(scope="module")
def elements():
    return phi_kernel_elements(N_FAST, dps=DPS_FAST)


@pytest.fixture(scope="module")
def base(elements):
    return certified_lower_bound(N_FAST, dps=DPS_FAST, elements=elements)


def test_ratio_from_energy_matches_the_archived_conversion():
    """R = -32 E_Q/(3 sqrt(pi)) reproduces the archived (e_Q, ratio) pair."""
    # s6b_momentum_4d_worldline.json, the n = 400 row
    e_q = -0.0760200519912855
    ratio = 0.4574903623919162
    mp.mp.dps = 30
    got = float(ratio_from_energy(mp.mpf(e_q)))
    assert abs(got - ratio) < 1e-13, (got, ratio)


def test_phi_gram_closed_form():
    """<phi_m, phi_n> = 2^{(m+n)/2+1} Gamma((m+n+4)/2) against direct quadrature."""
    G = phi_gram(5, dps=40)
    x, w = leggauss(3000)
    om = 0.5 * 40.0 * (x + 1.0)
    wo = 0.5 * 40.0 * w
    for m in range(5):
        for n in range(5):
            num = float(np.sum(wo * om ** (m + n + 3) * np.exp(-om ** 2 / 2)))
            assert abs(num / float(G[m][n]) - 1.0) < 1e-12


def test_elements_match_the_qei_module_kernel():
    """(iii) the Gamma series IS the kernel that vacuum.inequalities.qei builds."""
    n_check, omega_max, n_nodes = 5, 40.0, 1600
    fh = fhat_of(gaussian_f(1.0))
    x, w = leggauss(n_nodes)
    om = 0.5 * omega_max * (x + 1.0)
    wo = 0.5 * omega_max * w
    base_k = np.outer(om, om) ** 1.5
    Kp = base_k * np.real(fh(om[:, None] - om[None, :]))
    Km = base_k * np.real(fh(om[:, None] + om[None, :]))
    ke = phi_kernel_elements(n_check, dps=60)
    worst = 0.0
    for m in range(n_check):
        pm = wo * om ** (m + 1.5) * np.exp(-om ** 2 / 4)
        for n in range(n_check):
            pn = wo * om ** (n + 1.5) * np.exp(-om ** 2 / 4)
            worst = max(
                worst,
                abs(float(pm @ Kp @ pn) / float(mp.mpf(ke.a_elem[m][n].a)) - 1.0),
                abs(float(pm @ Km @ pn) / float(mp.mpf(ke.b_elem[m][n].a)) - 1.0),
            )
    # the quadrature, not the series, is the loose side here
    assert worst < 5e-11, worst


def test_enclosure_contains_the_archived_value(base):
    """(i) [lo, hi] brackets both the archived value and the MANIFEST headline."""
    assert base.hi == FEWSTER_EVESON_RATIO_UPPER_BOUND
    assert base.lo < CERTIFIED_PINNED_RATIO < base.hi
    assert base.lo < MANIFEST_HEADLINE < base.hi
    assert base.contains(CERTIFIED_PINNED_RATIO)
    assert base.contains(MANIFEST_HEADLINE)
    # the bound must be tight, not trivially small: at n_basis = 20 the
    # measured deficit is 4.81e-5, so 1e-4 is a ceiling a wrong number fails
    gap = CERTIFIED_PINNED_RATIO - base.lo
    assert 0.0 < gap < 1.0e-4, gap
    assert base.lo > 0.45744, base.lo
    # the energy enclosure is a genuine two-sided interval and it is narrow
    lo_e, hi_e = base.energy_interval
    assert lo_e <= hi_e
    assert hi_e - lo_e < 1e-30, (lo_e, hi_e)
    assert base.max_element_rel_width < 1e-100


def test_ladder_increases_and_never_crosses_the_archived_value(elements):
    """Galerkin monotonicity: more modes -> a better (never worse) lower bound."""
    los = []
    for N, dps in ((10, 156), (14, 194)):
        los.append(certified_lower_bound(N, dps=dps).lo)
    los.append(certified_lower_bound(N_FAST, dps=DPS_FAST, elements=elements).lo)
    assert los == sorted(los), los
    for x in los:
        assert x < CERTIFIED_PINNED_RATIO, x


def test_mutated_kernel_enclosure_excludes_the_archived_value(elements, base):
    """(ii) the mutation test: B -> (1 + 1e-4) B must exclude 0.45749073."""
    up = certified_lower_bound(N_FAST, dps=DPS_FAST, elements=elements, b_scale=1.0001)
    dn = certified_lower_bound(N_FAST, dps=DPS_FAST, elements=elements, b_scale=0.9999)
    # the unmutated bound does NOT exclude it (the test can fail in both directions)
    assert base.lo < CERTIFIED_PINNED_RATIO
    # the mutated one does, with room to spare
    assert up.lo > CERTIFIED_PINNED_RATIO + MOMENTUM_GRID_UNCERTAINTY, up.lo
    assert up.lo - CERTIFIED_PINNED_RATIO > 3e-5, up.lo
    # and the mutation is signed, not a symmetric blow-up
    assert dn.lo < base.lo < up.lo, (dn.lo, base.lo, up.lo)


def test_state_positivity_is_actually_verified():
    """The interval Cholesky must REJECT a covariance that is not a state."""
    from mpmath import iv

    from vacuum.inequalities.qei_certified import _chol_iv

    iv.dps = 40
    # for one mode the condition is M^2 <= N(1 + N); at N = 0.05 the pure-state
    # boundary is |M| = sqrt(0.0525) = 0.22913..., so 0.2 is a state and 0.3 is
    # not and must be rejected.
    for m_val, ok in ((0.2, True), (0.3, False)):
        blk = [[iv.mpf(0.05), iv.mpf(m_val)], [iv.mpf(m_val), iv.mpf(1.05)]]
        if ok:
            _chol_iv(blk, 2)
        else:
            with pytest.raises(ValueError):
                _chol_iv(blk, 2)


def test_certified_enclosure_alias(elements):
    r = certified_enclosure(N_FAST, dps=DPS_FAST, elements=elements)
    assert r.lo == certified_lower_bound(N_FAST, dps=DPS_FAST, elements=elements).lo
    assert "Fewster" in r.hi_method
    assert "interval arithmetic" in r.lo_method


@pytest.mark.skipif(os.environ.get("QEI_CERT_SLOW") != "1",
                    reason="set QEI_CERT_SLOW=1 for the n_basis = 40 rung (~2 min)")
def test_slow_top_rung_is_within_1e_6_of_the_archived_value():
    r = certified_lower_bound(40)
    assert r.lo < CERTIFIED_PINNED_RATIO
    assert CERTIFIED_PINNED_RATIO - r.lo < 1.0e-6, r.lo


# --------------------------------------------------------------------------
# the archive: every number the MANIFEST quotes from S8 is pinned here
# --------------------------------------------------------------------------

_S8_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "papers", "qei-2d-sharp-constant", "data",
                        "s8_certified_enclosure.json")


@pytest.fixture(scope="module")
def s8():
    import json

    if not os.path.exists(_S8_PATH):
        pytest.skip("S8 archive not present (run notebook_certified.py)")
    with open(_S8_PATH) as fh:
        return json.load(fh)


def test_s8_archive_pins_the_enclosure(s8):
    """The MANIFEST headline row, to the digits it prints."""
    assert s8["hi"] == 1.0
    assert abs(s8["lo"] - 0.457490709365) < 5e-13, s8["lo"]
    assert abs(s8["gap_lo_to_archived"] - 2.458e-08) < 2e-11, s8["gap_lo_to_archived"]
    assert s8["archived_momentum_value"] == CERTIFIED_PINNED_RATIO
    # the lower endpoint is a LOWER bound: it must sit below the archived value
    assert s8["lo"] < CERTIFIED_PINNED_RATIO
    # ... and inside the momentum route's own stated uncertainty band
    assert s8["lo"] > MANIFEST_HEADLINE - MOMENTUM_GRID_UNCERTAINTY


def test_s8_archive_ladder_is_monotone_and_converging(s8):
    rows = sorted(s8["ladder"], key=lambda r: r["n_basis"])
    assert [r["n_basis"] for r in rows] == [16, 24, 32, 40, 48, 56, 64]
    los = [r["ratio_lower_bound"] for r in rows]
    assert los == sorted(los)
    assert all(x < CERTIFIED_PINNED_RATIO for x in los)
    gaps = [r["gap_to_archived"] for r in rows]
    assert abs(gaps[0] - 1.523e-04) < 1e-7, gaps[0]
    assert abs(gaps[-1] - 2.458e-08) < 2e-11, gaps[-1]
    # the interval arithmetic contributes nothing: widths are astronomically
    # below the Galerkin gap at every rung
    for r in rows:
        assert r["energy_width_log10"] < -150.0, r


def test_s8_archive_formulation_and_mutations(s8):
    cc = s8["element_crosscheck"]
    assert cc["max_rel_disagreement_A"] < 5e-12
    assert cc["max_rel_disagreement_B"] < 5e-12
    assert abs(cc["max_rel_disagreement_A"] - 8.364e-13) < 5e-15
    muts = {m["mutation"]: m for m in s8["mutations"]}
    assert muts["b_scale=1+1e-4"]["excludes_archived"] is True
    assert abs(muts["b_scale=1+1e-4"]["ratio_lower_bound"] - 0.457628840433) < 5e-12
    assert muts["kernel_scale=1+1e-4"]["excludes_archived"] is True
    assert muts["b_scale=1-1e-4"]["excludes_archived"] is False


def test_s8_archive_powers_stormer_is_labelled_not_certified(s8):
    ps = s8["powers_stormer_estimate_NOT_CERTIFIED"]
    assert "NOT CERTIFIED" in ps["STATUS"]
    assert abs(ps["powers_stormer_ratio_upper_estimate"] - 0.576051) < 1e-6
    # the PROVED trace-norm bound must sit above the measured trace norm
    assert ps["trace_norm_K_minus_proved_bound"] > ps["trace_norm_K_minus_measured"]
    assert abs(ps["trace_norm_K_minus_proved_bound"] - 1.2278213378537361) < 1e-12
    assert abs(ps["trace_norm_K_minus_measured"] - 1.21135) < 1e-5
    # and the rigorous coarsening really is weaker than Fewster-Eveson's 1
    assert ps["coarsened_rigorous_ratio_upper_bound_from_trace_norm"] > 1.0
    # the Nystrom control reproduces the qei route's own archived n = 400 row
    # (s6b_momentum_4d_worldline.json); the 2.5e-12 residual is the float64
    # Riccati/LAPACK path, not a convention difference
    assert abs(ps["ratio_nystrom"] - 0.4574903623919162) < 1e-10
