"""Layer 5 / S8: the harvest of the convergence plane, the tau0 >= 1 rows, lambda = 0.1
and the ED anchor (``papers/interacting-vacua-first-numbers/data/s8_*.json``).

Every test reads an archived S8 record and pins the headline numbers the MANIFEST quotes
from it.  FIX ROUND 2 -- the tolerance rationale, stated as measured: the pins are tight
(1e-6 relative) because each is on an ARCHIVED number in a finished job file, which never
changes, NOT because the route reproduces to that precision.  The route's own same-spec
reproducibility is 0.0 % to 5.12 % relative, measured (L 16, n_max 8, chi_op 32, w 0.5,
lambda = 0: E_min -2.2652090e-02 at 2 BLAS threads against -2.2697211e-02 at 4, a 0.199 %
spread; worst archived same-spec spread 5.120 % at L 5, n_max 6, chi_op 32, w 0.5,
lambda = 4, 1 vs 2 threads -- and exactly 0.0 at L 16, n_max 6, chi_op 16, both routes,
so the floor is spec-dependent and is not a property of "the route").  That is why every knob penalty quoted
here is a BAND-AWARE factor -- the worse leg's smallest distance from truth over the better
leg's largest, across archived thread counts -- and never a raw ratio of two small numbers.
Each test skips cleanly if the record or the row is not on the tree (the S8
pool runs for days; rows are pinned as they land).  What is asserted beyond the pins is
the VERDICT LOGIC: which readings of the small-lambda deficit lambda = 0.1 excludes, which
tau0 rows are usable and why the others are not, and that the archived "resolved" flag of
the plane is what its own rows give under the stated criterion.
"""

import json
import pathlib

import numpy as np
import pytest

_DATA = pathlib.Path(__file__).resolve().parents[1] / "papers" / "interacting-vacua-first-numbers" / "data"


def _load(name):
    pth = _DATA / name
    if not pth.exists():
        pytest.skip(f"{name} is not on this tree")
    with open(pth) as fh:
        return json.load(fh)


def _close(x, y, rel):
    return abs(x - y) <= rel * max(abs(y), 1e-300)


# --------------------------------------------------------------------------
# lambda = 0.1 at the sweep's settings: the "next decisive cheap point"
# --------------------------------------------------------------------------

def test_lambda_0p1_is_the_first_order_slope_not_a_floor():
    """At the sweep's settings (n_max 8, chi_op 32, weight 0.5) the control-normalized
    residual at lambda = 0.1 reads 0.741 %, against the exact first-order prediction
    0.1 x 0.0809 = 0.809 % -- 0.35 bands (band 0.19 %) below it.  The alternative reading
    of the small-lambda deficit -- a lambda-independent cutoff floor at the ~1.4 % the
    lambda = 0.25 and 0.5 points sit at -- would have put it 3.3 bands ABOVE the
    prediction; it is excluded.  Measured 2026-09-11 (803 s, 2 threads)."""
    rec = _load("s8_lam0.1.json")
    row = rec["row"]
    assert row is not None, "the lambda = 0.1 row is not in the archive"
    assert row["lam"] == 0.1 and row["n_max"] == 8 and row["chi_op_allowed"] == 32 and row["weight_decay"] == 0.5
    assert _close(row["E_min"], -0.02157277132576929, 1e-6)
    assert _close(row["ratio_exact_over_hartree_exact"], 0.9944970631301456, 1e-6)
    assert _close(row["ratio_exact_over_free_exact"], 0.9522867704792837, 1e-6)
    assert row["ref_consistency"] < 5e-5, "the reference inconsistency must be at the lambda = 0 level"
    assert _close(rec["control_lam0_s5h"], 1.001922889968071, 1e-6)
    band = rec["band_lam0_s5h"]
    assert 0.0018 < band < 0.0020
    r = rec["residual_ctrl_normalized"]
    assert abs(r - 0.00741157519433655) < 2e-6
    pred = rec["predicted_residual_O1"]
    assert abs(pred - 0.00809088379025304) < 1e-8
    # the verdict: within one band of the exact first-order slope ...
    assert abs(r - pred) < band, f"lambda = 0.1 sits {(r - pred) / band:+.2f} bands from the O(lambda) prediction"
    assert abs(rec["bands_from_prediction"] - (r - pred) / band) < 1e-9
    # ... and more than three bands from the ~1.4 % floor the lambda = 0.25 / 0.5 points read
    floor = 0.0136
    assert (floor - r) > 3.0 * band, "a lambda-independent floor at the lambda = 0.25 level is not excluded"
    # the second-order reading (c2 ~ -0.10 from the lambda = 0.25 / 0.5 points) is not
    # separable from pure first order at this lambda: both are within one band
    assert abs(r - (pred - 0.10 * 0.1 ** 2)) < band


# --------------------------------------------------------------------------
# the tau0 >= 1 rows at chi_op 32 (n_max 8, weight 0.5): the controls and the gates
# --------------------------------------------------------------------------

_TAU0_PINS_CHIOP32 = {
    # (tau0, lam) -> (E_min, exact/E_H)   measured 2026-09-11/12, 2 threads
    (1.0, 0.0): (-0.01432501605077019, 1.0361415243928451),
    (1.0, 4.0): (-0.003958064870701605, 1.5813964609220081),
    (1.5, 0.0): (-0.005510591470120119, 1.253060528088746),
    (1.5, 4.0): (-0.004750395138244823, 26.757648835768723),
}


def _tau_rows(rec, axis, chi_op):
    return {(q["tau0"], q["lam"]): q for q in rec["rows"] if q["axis"] == axis and int(q["chi_op"]) == chi_op}


def test_tau0_rows_at_chiop32_are_pinned_and_not_usable():
    """At the S8 recipe (n_max 8, weight 0.5, chi_op 32, chi_acc 64, chi_mpo 128, chi_dmrg 16)
    the lambda = 0 control reads 1.036 at tau0 = 1.0 and 1.253 at tau0 = 1.5, against 1.085 and
    1.410 for the archived s7d rows (n_max 6, weight 0.25, chi_op 16, chi_acc 32, chi_mpo 64,
    chi_dmrg 12) -- six knobs apart, so the change is NOT attributable to chi_op alone; the
    split is measured by the chi_op-16 isolation rows and pinned in the test below.  The 1.036
    passes the 5 % control gate of s7s, but the
    lambda = 4 row it would normalize has a reference inconsistency |E_ref - E_ref_static|
    of 38 % of its own E_min (1.3-5 % at the resolved lambda <= 1 sweep points), so no
    ratio is quoted from tau0 >= 1 at chi_op 32 and the exponent statement stays restricted
    to tau0 = 0.375-0.75."""
    rec = _load("s8_tau0_rows.json")
    rows = _tau_rows(rec, "tau0", 32)
    for (tau0, lam), (e_min, ratio) in _TAU0_PINS_CHIOP32.items():
        if (tau0, lam) not in rows:
            pytest.skip(f"tau0 {tau0} lam {lam} chi_op 32 has not landed")
        q = rows[(tau0, lam)]
        assert _close(q["E_min"], e_min, 1e-6), f"tau0 {tau0} lam {lam}: E_min {q['E_min']!r}"
        assert _close(q["ratio_exact_over_hartree_exact"], ratio, 1e-6)
        assert q["L"] == 16 and q["m"] == 0.5
    # the controls, and why the rows are not usable
    assert abs(rows[(1.0, 0.0)]["ratio_exact_over_hartree_exact"] - 1.0) < 0.05
    assert abs(rows[(1.5, 0.0)]["ratio_exact_over_hartree_exact"] - 1.0) > 0.20
    assert rows[(1.0, 4.0)]["refcons_over_E_min"] > 0.30
    assert rows[(1.5, 4.0)]["refcons_over_E_min"] > 1.0 and rows[(1.5, 0.0)]["refcons_over_E_min"] > 1.0
    # the S8 recipe reads closer to 1 than the archived s7d recipe, in both rows (a
    # comparison of two recipes, not of chi_op: see test_chiop16_isolation_splits_the_knobs)
    arch = rec["archived_controls_chiop16"]
    assert abs(rows[(1.0, 0.0)]["ratio_exact_over_hartree_exact"] - 1.0) < abs(arch["tau0=1.0"] - 1.0)
    assert abs(rows[(1.5, 0.0)]["ratio_exact_over_hartree_exact"] - 1.0) < abs(arch["tau0=1.5"] - 1.0)
    fit = rec["scaling_refit"]["by_chi_op"]["chi_op=32"]
    assert fit["gates"] == {"control": 0.05, "refcons_over_E_min": 0.05}
    assert fit["usable_tau0"] == [] and set(fit["failed_tau0_refcons"]) == {"1", "1.5"}
    assert fit["residual_hartree_normalized"]["exponent_of_inverse_tau0"] is None
    assert fit["residual_keff_normalized"]["exponent_of_inverse_tau0"] is None


_ISO_PINS = {
    # tau0 -> (E_min, lambda = 0 control) at the S8 recipe run AT chi_op 16
    # (n_max 8, weight 0.5, chi_acc 32, chi_mpo 64, chi_dmrg 16, sketched SVD, s7d's dt)
    1.0: (-1.42925271e-02, 1.03379157),
    1.5: (-5.88299749e-03, 1.33774241),
}


def test_chiop16_isolation_splits_the_knobs():
    """FIX ROUND 1.  The archived tau0 >= 1 controls (1.085 / 1.410) are s7d rows at
    n_max 6, chi_op 16, chi_acc 32, chi_mpo 64, chi_dmrg 12, weight 0.25; the S8 rows are
    at n_max 8, chi_op 32, chi_acc 64, chi_mpo 128, chi_dmrg 16, weight 0.5.  Six knobs,
    so the improvement of the control was NOT a chi_op effect and the earlier "only chi_op
    changed (16 -> 32/48)" is retracted.  These rows run the S8 recipe AT chi_op 16 and
    measure the split: at tau0 = 1.0 the (n_max, weight, chi_dmrg) step carries all of it
    (-0.0512) and chi_op 16 -> 32 moves the control 0.0023 the WRONG way; at tau0 = 1.5 the
    two steps are comparable (-0.0723 and -0.0847) and chi_op carries the larger half."""
    rec = _load("s8_tau0_rows.json")
    iso = rec.get("chiop16_isolation")
    if not iso or not iso["rows"]:
        pytest.skip("no chi_op 16 isolation row has landed")
    rows = {round(float(r["tau0"]), 6): r for r in iso["rows"] if r["lam"] == 0.0}
    for tau0, (e_min, ctrl) in _ISO_PINS.items():
        if tau0 not in rows:
            pytest.skip(f"the chi_op 16 isolation row at tau0 {tau0} has not landed")
        q = rows[tau0]
        assert _close(q["E_min"], e_min, 1e-6), f"tau0 {tau0}: E_min {q['E_min']!r}"
        assert _close(q["control_lam0"], ctrl, 1e-6), f"tau0 {tau0}: control {q['control_lam0']!r}"
        assert (q["n_max"], q["chi_op"], q["weight_decay"], q["chi_acc"], q["chi_mpo"], q["dmrg_chi"]) \
            == (8, 16, 0.5, 32, 64, 16)
        assert q["L"] == 16 and q["m"] == 0.5
    # the split is arithmetic on the three controls, and it adds up
    for key, sp in iso["split"].items():
        if sp["s8_chiop32_nmax8_w0.5_chidmrg16"] is None:
            continue
        total = sp["s8_chiop32_nmax8_w0.5_chidmrg16"] - sp["s7d_chiop16_nmax6_w0.25_chidmrg12"]
        assert abs(sp["step_nmax_weight_chidmrg"] + sp["step_chi_op_16_to_32"] - total) < 1e-12, key
    s1 = iso["split"]["tau0=1"]
    # tau0 = 1.0: chi_op carried none of the improvement -- it moved the control AWAY from 1
    assert s1["step_nmax_weight_chidmrg"] < -0.04
    assert 0.0 < s1["step_chi_op_16_to_32"] < 0.005
    assert abs(s1["iso_chiop16_nmax8_w0.5_chidmrg16"] - 1.0) < abs(s1["s8_chiop32_nmax8_w0.5_chidmrg16"] - 1.0)
    s15 = iso["split"]["tau0=1.5"]
    # tau0 = 1.5: both steps help, and chi_op carries the larger half
    assert s15["step_nmax_weight_chidmrg"] < -0.05 and s15["step_chi_op_16_to_32"] < -0.05
    assert abs(s15["step_chi_op_16_to_32"]) > abs(s15["step_nmax_weight_chidmrg"])
    # neither isolation row is usable in its own right: both lose >= 12 % of the operator norm
    for q in rows.values():
        assert q["norm_ratio"] < 0.89


_TAU0_PINS_CHIOP48 = {
    # (tau0, lam) -> (E_min, exact/E_H): filled as the chi_op 48 rows land.
    # The first one (landed 13:46 PDT 2026-09-12, 2696 s): its lambda = 0 control was still
    # running, and the row is unusable on its own terms -- see the assertion below.
    (1.5, 4.0): (-0.0019248737173302999, 10.842275954449343),
    (1.5, 0.0): (-0.0053538588617185745, 1.217420897367203),
    (1.0, 4.0): (-0.004070078105677499, 1.6261499804205164),
    # FIX ROUND 2: the tau0 = 1.0 lambda = 0 control landed 14:12 PDT 2026-09-12 (2222 s),
    # completing the tau0 = 1.0 chi_op sequence.  Its own refcons is 7.65 % of |E_min|, i.e.
    # it now FAILS the 5 % gate that the chi_op 32 row (2.69 %) passed.
    (1.0, 0.0): (-0.014356626253559046, 1.038427919297251),
}


def test_tau0_rows_at_chiop48_are_pinned():
    rec = _load("s8_tau0_rows.json")
    rows = _tau_rows(rec, "tau0", 48)
    if not rows:
        pytest.skip("no chi_op 48 tau0 row has landed")
    # FIX ROUND 2: `continue`, not `skip` -- a skip on the first un-landed key used to drop
    # every pin after it.
    n_pinned = 0
    for (tau0, lam), (e_min, ratio) in _TAU0_PINS_CHIOP48.items():
        if (tau0, lam) not in rows:
            continue
        q = rows[(tau0, lam)]
        assert _close(q["E_min"], e_min, 1e-6), f"tau0 {tau0} lam {lam}: E_min {q['E_min']!r}"
        assert _close(q["ratio_exact_over_hartree_exact"], ratio, 1e-6)
        n_pinned += 1
    assert n_pinned >= 1, "no pinned chi_op 48 tau0 row has landed"
    # the tau0 = 1.5 lambda = 4 row at chi_op 48 is unusable by the block's own gate: its
    # reference inconsistency is 654 % of its own |E_min| and it loses 20 % of the operator norm
    if (1.5, 4.0) in rows:
        q = rows[(1.5, 4.0)]
        assert q["refcons_over_E_min"] > 5.0 and q["norm_ratio"] < 0.81
    # the chi_op sequence of the tau0 = 1.5 control at FIXED n_max 8 / weight 0.5 / chi_dmrg 16
    # (the isolation row, chi_op 32, chi_op 48): 1.33774 -> 1.25306 -> 1.21742, steps -0.0847
    # then -0.0356.  It improves monotonically with chi_op and is still 21.7 % off 1, so chi_op
    # is not closing this row at any cost the plan contemplates.
    iso = rec.get("chiop16_isolation") or {}
    c16 = {round(float(r["tau0"]), 6): r["control_lam0"] for r in iso.get("rows", []) if r["lam"] == 0.0}
    c32 = _tau_rows(rec, "tau0", 32)
    if (1.5, 0.0) in rows and (1.5, 0.0) in c32 and 1.5 in c16:
        seq = [c16[1.5], c32[(1.5, 0.0)]["ratio_exact_over_hartree_exact"],
               rows[(1.5, 0.0)]["ratio_exact_over_hartree_exact"]]
        assert seq[0] > seq[1] > seq[2] > 1.2, f"the chi_op sequence at tau0 1.5 is {seq}"
        assert abs(seq[2] - 1.2174209) < 1e-6
        assert abs(seq[2] - 1.0) > 0.20, "still a fifth off 1 at chi_op 48"
        assert abs(seq[1] - seq[2]) < abs(seq[0] - seq[1]), "the chi_op steps are shrinking"
        assert rows[(1.5, 0.0)]["refcons_over_E_min"] > 0.5, "the control itself fails the refcons gate"
    # FIX ROUND 2.  The tau0 = 1.0 control's chi_op sequence at the SAME fixed n_max 8 /
    # weight 0.5 / chi_dmrg 16: 1.033792 (chi_op 16) -> 1.036142 (32) -> 1.038428 (48).
    # It is monotone and it moves AWAY from 1 at every step (+0.00235, +0.00229), while its
    # reference inconsistency goes 9.8 % -> 2.7 % -> 7.65 % of |E_min| (gate 5 %).  So at
    # tau0 = 1.0 chi_op neither closes the control nor keeps the refcons gate: the opposite
    # of the tau0 = 1.5 row, where chi_op improves the control monotonically.
    if (1.0, 0.0) in rows and (1.0, 0.0) in c32 and 1.0 in c16:
        seq1 = [c16[1.0], c32[(1.0, 0.0)]["ratio_exact_over_hartree_exact"],
                rows[(1.0, 0.0)]["ratio_exact_over_hartree_exact"]]
        assert seq1[0] < seq1[1] < seq1[2], f"the chi_op sequence at tau0 1.0 is {seq1}"
        assert all(x > 1.0 for x in seq1), seq1
        assert abs(seq1[2] - 1.0384279) < 1e-6
        steps = [seq1[1] - seq1[0], seq1[2] - seq1[1]]
        assert steps[0] > 0 and steps[1] > 0, steps
        assert abs(steps[0] - 0.00234995) < 1e-6 and abs(steps[1] - 0.00228639) < 1e-6, steps
        assert abs(rows[(1.0, 0.0)]["refcons_over_E_min"] - 0.07653625) < 1e-6
        assert rows[(1.0, 0.0)]["refcons_over_E_min"] > 0.05 > c32[(1.0, 0.0)]["refcons_over_E_min"], \
            "the chi_op 48 control crosses back over the 5 % refcons gate the chi_op 32 one passed"
    fit = rec["scaling_refit"]["by_chi_op"].get("chi_op=48")
    assert fit is not None
    # every usable row must have passed BOTH gates (the archive's own bookkeeping)
    for r in fit["rows"]:
        assert r["usable"] == (r["control_ok"] and r["refcons_ok"])
    if fit["residual_hartree_normalized"].get("exponent_of_inverse_tau0") is not None:
        assert fit["n_usable"] > 0 and fit["residual_hartree_normalized"]["n_points_used"] >= 4


_CHIOP_SEQ_PINS = {
    # FIX ROUND 2.  `s8_tau0_rows.json::control_chi_op_sequence` -- the lambda = 0 control and
    # its reference inconsistency against chi_op at FIXED n_max 8 / weight 0.5 / chi_dmrg 16
    # (chi_op 16 is the tau0_iso row, 32 and 48 are the pool rows).  tau0 = 1.0 moves AWAY
    # from 1 at every step; tau0 = 1.5 moves toward it and is still 21.7 % off.  Neither
    # tau0 >= 1 control holds the 5 % refcons gate as chi_op is raised: at tau0 = 1.5 the
    # refcons goes 2.3 % -> 117 % -> 64 %, non-monotone and never usable above chi_op 16.
    "tau0=1": {"chi_op": [16, 32, 48],
               "control_lam0": [1.0337915706092045, 1.0361415243928451, 1.038427919297251],
               "refcons": [0.09808325026306924, 0.026942112334095606, 0.07653625265241559],
               "toward_1": False, "refcons_ok": [False, True, False]},
    "tau0=1.5": {"chi_op": [16, 32, 48],
                 "control_lam0": [1.3377424145720471, 1.253060528088746, 1.217420897367203],
                 "refcons": [0.022630281400394, 1.1721334207419396, 0.639761056879877],
                 "toward_1": True, "refcons_ok": [True, False, False]},
}


def test_control_chi_op_sequences_are_pinned():
    """FIX ROUND 2.  The chi_op sequence of the lambda = 0 control at fixed n_max 8 / weight
    0.5 / chi_dmrg 16 is now a field of the archive (`control_chi_op_sequence`) instead of a
    number reassembled by hand, and both tau0 >= 1 sequences are pinned with their reference
    inconsistencies.  The verdict the MANIFEST states rests on two facts here: at tau0 = 1.0
    raising chi_op moves the control monotonically AWAY from 1 (+0.00235, +0.00229), and at
    neither tau0 does the control hold the 5 % refcons gate above chi_op 16."""
    rec = _load("s8_tau0_rows.json")
    seqs = rec.get("control_chi_op_sequence")
    assert seqs is not None, "the harvest no longer writes control_chi_op_sequence"
    n = 0
    for key, want in _CHIOP_SEQ_PINS.items():
        got = seqs.get(key)
        if got is None:
            continue
        assert got["chi_op"] == want["chi_op"], (key, got["chi_op"])
        for a, b in zip(got["control_lam0"], want["control_lam0"]):
            assert abs(a - b) <= 1e-9, (key, a, b)
        for a, b in zip(got["refcons_over_E_min"], want["refcons"]):
            assert abs(a - b) <= 1e-9, (key, a, b)
        assert got["toward_1"] is want["toward_1"], (key, got["toward_1"])
        assert got["refcons_ok"] == want["refcons_ok"], (key, got["refcons_ok"])
        assert got["refcons_gate"] == 0.05
        n += 1
    assert n >= 1, "no pinned chi_op sequence is in the archive"
    if "tau0=1" in seqs:
        st = seqs["tau0=1"]["steps"]
        assert all(x > 0 for x in st), st            # away from 1, at every step
        assert all(c > 1.0 for c in seqs["tau0=1"]["control_lam0"])
    if "tau0=1.5" in seqs:
        st = seqs["tau0=1.5"]["steps"]
        assert all(x < 0 for x in st) and abs(st[1]) < abs(st[0]), st
        assert seqs["tau0=1.5"]["control_lam0"][-1] > 1.20, "still a fifth off 1 at chi_op 48"


_AHALF_PINS = {
    # chi_op -> (E_min at lam 1, exact/E_H at lam 1, control at lam 0): filled as the rows land.
    # LANDED 2026-09-13 (the lam = 0 control finished at 00:30 local, 1957 s; the lam = 1 row
    # 14837 s), at L 32, m 0.25, tau0 1.5, n_max 8, chi_op 32, weight 0.5, dt 0.375, order 4,
    # n_steps 16.  The PAIR EXISTS AND IS NOT USABLE -- see the assertions below.
    32: (-0.0077595670293511, 2.03833904983217, 1.1865002227506591),
}


def test_a_half_rows_are_pinned_with_their_controls():
    rec = _load("s8_tau0_rows.json")
    rows = {int(q["chi_op"]): q for q in rec["rows"] if q["axis"] == "a_half" and q["lam"] == 1.0 and q["control_lam0"] is not None}
    if not rows:
        pytest.skip("no a = 1/2 pair has landed")
    for chi, (e_min, ratio, ctrl) in _AHALF_PINS.items():
        if chi not in rows:
            pytest.skip(f"a = 1/2 at chi_op {chi} has not landed")
        q = rows[chi]
        assert q["L"] == 32 and q["m"] == 0.25 and q["tau0"] == 1.5
        assert _close(q["E_min"], e_min, 1e-6) and _close(q["ratio_exact_over_hartree_exact"], ratio, 1e-6)
        assert _close(q["control_lam0"], ctrl, 1e-6)
    for r in rec["scaling_refit"]["a_half"]["rows"]:
        assert r["usable"] == (r["control_ok"] and r["ref_consistency_over_E_min"] <= 0.05
                               and r["control_refcons_over_E_min"] <= 0.05)
    # THE SUB-ITEM CLOSES NEGATIVELY AT chi_op 32, and these are the numbers that close it.
    # The pair has landed, so the skip above no longer fires; what it reports is that the
    # a = 1/2 point is NOT usable at these knobs -- the same verdict tau0 >= 1 got, and for
    # the same two reasons: the lambda = 0 control is far from 1 and the reference
    # inconsistency is far above the harvest's own 5 % gate.  If a future run makes this row
    # usable, item 5's a = 1/2 paragraph must be rewritten, not this assertion loosened.
    r32 = next((r for r in rec["scaling_refit"]["a_half"]["rows"] if int(r["chi_op"]) == 32), None)
    if r32 is not None:
        assert r32["usable"] is False and r32["control_ok"] is False
        assert _close(r32["control_lam0"], 1.1865002227506591, 1e-9), r32["control_lam0"]
        assert _close(r32["ratio_exact_over_hartree"], 2.03833904983217, 1e-9)
        assert _close(r32["ratio_hartree_normalized"], 1.7179424080567773, 1e-9)
        # 18.65 % off 1 at lambda = 0, and 175.3 % / 20.0 % reference inconsistency
        assert abs(r32["control_lam0"] - 1.0) > 0.15, r32["control_lam0"]
        assert _close(r32["ref_consistency_over_E_min"], 1.7531990543469176, 1e-9)
        assert _close(r32["control_refcons_over_E_min"], 0.20014788661498115, 1e-9)
        assert r32["ref_consistency_over_E_min"] > 0.05 and r32["control_refcons_over_E_min"] > 0.05


# --------------------------------------------------------------------------
# the (n_max, chi_op) plane at lambda 0 / 2 / 4: pins and the verdict logic
# --------------------------------------------------------------------------

_PLANE_PINS = {
    # (n_max, chi_op, lam) -> (E_min, exact/E_H): filled as the rows land (weight 0.5, exact SVD)
    # The FIRST plane point, landed 13:05-13:16 PDT 2026-09-12 (2.7 h each at 2 threads).
    (8, 32, 0.0): (-0.02316592345734858, 1.0226132795472942),
    (8, 32, 4.0): (-0.007219076229307397, 0.9597907409840675),
}


def _plane(rec):
    return {(int(q["n_max"]), int(q["chi_op"]), float(q["lam"])): q for q in rec["rows"]}


def test_plane_rows_are_pinned():
    rec = _load("s8_convergence_plane.json")
    rows = _plane(rec)
    if not rows:
        pytest.skip("no plane row has landed")
    n = 0
    for key, (e_min, ratio) in _PLANE_PINS.items():
        if key not in rows:
            continue
        q = rows[key]
        assert _close(q["E_min"], e_min, 1e-6), f"{key}: E_min {q['E_min']!r}"
        assert _close(q["ratio_exact_over_hartree_exact"], ratio, 1e-6), f"{key}: ratio {q['ratio_exact_over_hartree_exact']!r}"
        assert q["chi_mpo"] <= 4 * key[1] and q["dmrg_chi"] == 16
        n += 1
    if n == 0:
        pytest.skip("none of the pinned plane rows has landed")


def test_first_plane_point_is_not_usable_and_does_not_close_item_4():
    """FIX ROUND 1.  The first plane point (n_max 8, chi_op 32, weight 0.5, EXACT SVD) reads
    a control-normalized residual of +6.14 % against E_H (+10.2 % against K_eff) at lambda = 4.
    It closes nothing: (i) its own lambda = 0 control is 1.0226, i.e. 2.3 % off 1 and WORSE
    than the sweep's sketched-SVD control (1.00192), (ii) the lambda = 4 row's reference
    inconsistency is 56.5 % of its own |E_min| -- eleven times the 5 % gate the tau0 rows are
    held to -- and (iii) one setting gives neither spread, so the verdict is unresolved."""
    rec = _load("s8_convergence_plane.json")
    rows = _plane(rec)
    if (8, 32, 4.0) not in rows or (8, 32, 0.0) not in rows:
        pytest.skip("the first plane point has not landed")
    ctrl, row = rows[(8, 32, 0.0)], rows[(8, 32, 4.0)]
    assert abs(ctrl["ratio_exact_over_hartree_exact"] - 1.0226133) < 1e-6
    assert abs(ctrl["ratio_exact_over_hartree_exact"] - 1.0) > 0.0019, "the control is off 1 by more than the sweep's band"
    assert abs(row["residual_ctrl_normalized"] - 0.0614333) < 1e-6
    assert abs(row["residual_keff_ctrl_normalized"] - 0.1020069) < 1e-6
    assert abs(row["refcons_over_E_min"] - 0.5651385) < 1e-6 and row["refcons_ok"] is False
    assert row["refcons_over_E_min"] > 10 * 0.05, "the row fails the tau0 block's 5 % gate by an order of magnitude"
    assert ctrl["refcons_ok"] is True and ctrl["refcons_over_E_min"] < 0.02
    v = rec["verdict"]["lam=4"]
    assert v["resolved"] is False and v["n_settings"] == 1
    assert v["spread_two_largest_n_max"] is None and v["spread_chi32_vs_64"] is None
    assert [(q["n_max"], q["chi_op"]) for q in v["rows_failing_refcons_gate"]] == [(8, 32)]


def test_plane_verdict_is_what_its_rows_give():
    """The archived 'resolved' flag per lambda is recomputed here from the archived rows
    under the stated criterion (both spreads exist and max(spread) <= 3 x the sweep's
    lambda = 0 band); the control-normalized residual is 1 - ratio / control; the K_eff
    column is the Hartree one divided by the cutoff-free factor of s7b."""
    rec = _load("s8_convergence_plane.json")
    rows = _plane(rec)
    if not rows:
        pytest.skip("no plane row has landed")
    band = rec["band_lam0_sweep"]
    assert 0.0018 < band < 0.0020
    for key, v in rec["verdict"].items():
        lam = float(key.split("=")[1])
        sub = {(n, c): q for (n, c, l), q in rows.items() if l == lam and q["residual_ctrl_normalized"] is not None}
        for (n, c), q in sub.items():
            ctrl = rows[(n, c, 0.0)]
            assert abs(q["control_lam0"] - ctrl["ratio_exact_over_hartree_exact"]) < 1e-12
            assert abs(q["residual_ctrl_normalized"] - (1.0 - q["ratio_exact_over_hartree_exact"] / q["control_lam0"])) < 1e-12
            if q.get("E_keff_over_E_hartree"):
                assert abs(q["ratio_exact_over_keff"] - q["ratio_exact_over_hartree_exact"] / q["E_keff_over_E_hartree"]) < 1e-12
        nm32 = sorted(n for (n, c) in sub if c == 32)
        top2 = nm32[-2:]
        spread_n = (max(sub[(n, 32)]["residual_ctrl_normalized"] for n in top2)
                    - min(sub[(n, 32)]["residual_ctrl_normalized"] for n in top2)) if len(top2) == 2 else None
        pairs = [abs(sub[(n, 32)]["residual_ctrl_normalized"] - sub[(n, 64)]["residual_ctrl_normalized"]) for n in nm32 if (n, 64) in sub]
        spread_c = max(pairs) if pairs else None
        assert v["spread_two_largest_n_max"] == spread_n or abs(v["spread_two_largest_n_max"] - spread_n) < 1e-12
        assert (v["spread_chi32_vs_64"] is None and spread_c is None) or abs(v["spread_chi32_vs_64"] - spread_c) < 1e-12
        expected = spread_n is not None and spread_c is not None and max(spread_n, spread_c) <= 3.0 * band
        assert v["resolved"] == expected, f"{key}: archived resolved={v['resolved']} vs recomputed {expected}"
        vals = [q["residual_ctrl_normalized"] for q in sub.values()]
        assert v["bracket_hartree"] == [min(vals), max(vals)]


# --------------------------------------------------------------------------
# the ED anchor: MPO vs matrix-free ED at the same (L, n_max, lambda, tau0, grid)
# --------------------------------------------------------------------------

_PAIR_PINS = {
    # (knobs, L, n_max, lam) -> (rel dev MPO vs ED on the same trapezoid grid, tolerance):
    # filled as the pairs land.  "sweep_knobs" = chi_op 32, weight 0.5, sketched SVD (the
    # settings every quoted S8 ratio is at); "generous" = chi_op 64, weight 1, exact SVD,
    # no window, chi_dmrg 32.  The tolerance is 1e-6 absolute on a deterministic number; a
    # wrong knob moves E_min by >= 1e-3.
    ("sweep_knobs", 5, 6, 0.0): (+0.006356045, 1e-6),
    ("sweep_knobs", 5, 6, 4.0): (-0.119118554, 1e-6),
    ("sweep_knobs", 6, 5, 0.0): (+0.003355926, 1e-6),
    ("sweep_knobs", 6, 5, 4.0): (+0.033669878, 1e-6),
    ("generous", 6, 5, 0.0): (-0.001647010, 1e-6),
    ("generous", 6, 5, 4.0): (-0.015169710, 1e-6),
    # L = 5, n_max 8 (the sweep's own cutoff), lambda = 0: the plane's knobs (chi_op 32,
    # EXACT SVD) beat the sweep's (sketched) by 3.5x on the same point
    ("plane_knobs", 5, 8, 0.0): (-0.003891669, 1e-6),
    ("sweep_knobs", 5, 8, 0.0): (+0.013798158, 1e-6),
    # FIX ROUND 2.  The same point at lambda = 4 -- the two largest deviations in the whole
    # archive, and the ones the MANIFEST bullet and the numbers table are about.  Round 1
    # quoted only the n_max 5 and 6 lambda = 4 pairs and called the effect "3-12 %"; that
    # sentence is retracted (MANIFEST no-longer-claims item 8).  MPO E_min -1.3060750e-02
    # (sweep) / -1.2451097e-02 (plane) against the exact -2.1561127e-02.
    ("sweep_knobs", 5, 8, 4.0): (+0.394245467, 1e-6),
    ("plane_knobs", 5, 8, 4.0): (+0.422521056, 1e-6),
    # FIX ROUND 2, the chi_op ladder at the paper's own cutoff (L 5, n_max 8, weight 0.5,
    # sketched SVD, chi_dmrg 16 -- the sweep's other knobs held fixed).  Rows are added here
    # as they land; chi_op 64 (lam 4 and 0) and 96 (lam 4) were still running at 15:15 PDT.
    ("chiop48_n8", 5, 8, 4.0): (+0.148838336, 1e-6),
    ("chiop64_n8", 5, 8, 4.0): (+0.136695496, 1e-6),
    ("chiop64_dmrg32_n8", 5, 8, 4.0): (+0.136694436, 1e-6),
    ("chiop96_n8", 5, 8, 4.0): (-0.045009691, 1e-6),
    ("chiop64_w1_n8", 5, 8, 4.0): (-0.011876990, 1e-6),
    ("chiop64_n8", 5, 8, 0.0): (+0.012082481, 1e-6),
}

# FIX ROUND 2.  The worst measured MPO-vs-ED deviation per knob set, pinned so it cannot
# silently shrink: these are the numbers the MANIFEST bullet "What the sweep's knobs cost
# at lambda = 4" and the Status section quote.  Tolerance 1e-6 absolute on deterministic
# numbers.  A knob set is checked only once its pairs have landed.
_WORST_BY_KNOBS_PINS = {
    "sweep_knobs": (0.394245467, 1e-6),
    "chiop48_n8": (0.148838336, 1e-6),
    "chiop64_n8": (0.136695496, 1e-6),
    "chiop64_dmrg32_n8": (0.136694436, 1e-6),
    "chiop96_n8": (0.045009691, 1e-6),
    "chiop64_w1_n8": (0.011876990, 1e-6),
    "plane_knobs": (0.422521056, 1e-6),
    "archived": (0.035347680, 1e-6),
    # 2026-09-13: the pool finished, landing anchor_generous_L5_nmax8_lam0/lam4 and
    # taking this knob set from 12 pairs to 14; the worst moved 0.015169706 ->
    # 0.018027347 (n_max range now [2, 8]).  The pin follows the completed archive.
    "generous": (0.018027347, 1e-6),
}

# FIX ROUND 2.  The sweep knobs' worst |rel dev| per n_max: the lambda = 4 error GROWS with
# the Fock cutoff at fixed chi_op 32, and the largest value is at n_max 8, the cutoff every
# quoted S8 lambda = 4 ratio and the first plane point are taken at.
_SWEEP_WORST_BY_NMAX_PINS = {
    "5": (0.033669878, 1e-6),
    "6": (0.119118554, 1e-6),
    "8": (0.394245467, 1e-6),
}


def test_mpo_vs_ed_pairs_are_pinned_and_the_cross_validation_levels_hold():
    rec = _load("s8_ed_anchor.json")
    pairs = {(p["knobs"], p["L"], p["n_max"], p["lam"]): p for p in rec["pairs_mpo_vs_ed"]}
    if not pairs:
        pytest.skip("no MPO-vs-ED pair has landed")
    n = 0
    for key, (dev, tol) in _PAIR_PINS.items():
        if key not in pairs:
            continue
        assert abs(pairs[key]["rel_dev_mpo_vs_ed_same_grid"] - dev) <= tol, f"{key}: {pairs[key]['rel_dev_mpo_vs_ed_same_grid']!r}"
        n += 1
    if n == 0:
        pytest.skip("none of the pinned pairs has landed")
    xv = rec["cross_validation_by_knobs"]
    for knobs, e in xv.items():
        assert e["worst_abs_rel_dev"] == max(abs(p["rel_dev"]) for p in e["pairs"])
        assert e["n_max_range"] == [min(p["n_max"] for p in e["pairs"]), max(p["n_max"] for p in e["pairs"])]
    # FIX ROUND 2.  The level assertions, made non-vacuous.  Round 1's only level check was
    # `worst < 2e-4 or n_max_range[1] > 3`, whose second branch is true on this tree, so it
    # asserted nothing.  Pin each knob set's worst deviation instead.
    n_lvl = 0
    for knobs, (worst, tol) in _WORST_BY_KNOBS_PINS.items():
        if knobs not in xv:
            continue
        assert abs(xv[knobs]["worst_abs_rel_dev"] - worst) <= tol, (knobs, xv[knobs]["worst_abs_rel_dev"])
        assert abs(rec["worst_rel_dev_by_knobs"][knobs] - worst) <= tol, (knobs, rec["worst_rel_dev_by_knobs"][knobs])
        n_lvl += 1
    assert n_lvl >= 1, "no pinned knob set has landed"
    # the generous knobs reproduce ED to <= 2e-4 at n_max <= 3 (s5i), and to 1.5e-2 at n_max 5
    if "generous" in xv:
        g = xv["generous"]
        assert max(abs(p["rel_dev"]) for p in g["pairs"] if p["n_max"] <= 3) < 2e-4


def test_sweep_knob_pairs_are_an_order_of_magnitude_worse_at_lambda_4():
    """FIX ROUND 1, docstring corrected in FIX ROUND 2.  The knobs every quoted S8 ratio is
    at (chi_op 32, weight 0.5, sketched SVD) reproduce exact ED to 3.4e-3 / 6.4e-3 / 1.4e-2
    at lambda = 0 (L = 6 n_max 5, L = 5 n_max 6, L = 5 n_max 8) but only to +3.4e-2 /
    -1.2e-1 / +3.9e-1 at lambda = 4 on the SAME (L, n_max, grid).  That is the size of the
    MPO truncation error on the interacting side at these knobs, and it is why item 4 (the
    lambda >= 2 magnitude) is not decided by rows at these knobs.  The generous knobs at
    L = 6, n_max 5 sit between: -1.6e-3 (lambda 0), -1.5e-2 (lambda 4).  The growth of the
    lambda = 4 error with n_max is pinned separately, below."""
    rec = _load("s8_ed_anchor.json")
    pairs = {(p["knobs"], p["L"], p["n_max"], p["lam"]): p for p in rec["pairs_mpo_vs_ed"]}
    need = [("sweep_knobs", 6, 5, 0.0), ("sweep_knobs", 6, 5, 4.0)]
    if any(k not in pairs for k in need):
        pytest.skip("the sweep-knob lambda 0 / 4 pair at L = 6, n_max 5 has not landed")
    d0 = abs(pairs[("sweep_knobs", 6, 5, 0.0)]["rel_dev_mpo_vs_ed_same_grid"])
    d4 = abs(pairs[("sweep_knobs", 6, 5, 4.0)]["rel_dev_mpo_vs_ed_same_grid"])
    assert d0 < 5e-3 < d4, (d0, d4)
    assert d4 / d0 > 5.0
    if ("sweep_knobs", 5, 6, 4.0) in pairs:
        assert abs(pairs[("sweep_knobs", 5, 6, 4.0)]["rel_dev_mpo_vs_ed_same_grid"]) > 0.10
    if ("generous", 6, 5, 4.0) in pairs:
        # generous beats the sweep's knobs at lambda = 4 on the same point, but not by a decade
        assert abs(pairs[("generous", 6, 5, 4.0)]["rel_dev_mpo_vs_ed_same_grid"]) < d4


def test_chi_op_ladder_at_nmax8_and_what_it_says_about_the_refcons_gate():
    """FIX ROUND 2.  Seven MPO rows at L = 5, n_max 8, tau0 0.75, lambda = 4 -- the paper's
    own Fock cutoff, where the exact answer E_min = -2.1561127e-02 exists -- walking one
    knob at a time off the sweep's settings (weight 0.5, sketched SVD, chi_dmrg 16).  Four
    things are measured and pinned here, each at this one point:

    (1) chi_dmrg is NOT the limiter: 16 -> 32 at chi_op 64 moves the deviation by 1.1e-5
        relative (+13.6695 % -> +13.6694 %).
    (2) The WEIGHTING is: weight_decay 0.5 -> 1 at chi_op 64 takes +13.67 % -> -1.19 %.
    (3) chi_op alone helps but is non-monotone and changes sign: +39.42 / +14.88 / +13.670 /
        -4.50 % at chi_op 32 / 48 / 64 / 96.  |error| falls throughout but the route crosses
        the exact answer between 64 and 96, so no chi_op here is a one-sided approximation.
    (4) The harvest's 5 % reference-consistency gate does not track lambda = 4 accuracy and
        is anti-correlated with it across this set: the best refcons (1.21 %, chi_op 64) is
        13.7 % wrong, and the most accurate row (1.19 %, weight 1) has refcons 36.06 %.  The
        gate is necessary, not sufficient, and not monotone in the error -- so no lambda >= 2
        magnitude may be quoted on the strength of passing it.

    This is the measured reason item 4 stays open even when the plane's rows start passing
    their gates.  Rows are skipped individually until they land."""
    rec = _load("s8_ed_anchor.json")
    pairs = {(p["knobs"], p["L"], p["n_max"], p["lam"]): p for p in rec["pairs_mpo_vs_ed"]}

    def g(knobs, lam=4.0):
        return pairs.get((knobs, 5, 8, lam))

    if g("chiop48_n8") is None and g("chiop64_n8") is None:
        pytest.skip("no chi_op ladder row at L = 5, n_max 8 has landed")
    # the pinned values themselves live in _PAIR_PINS (checked by the pairs test); here the
    # RELATIONS between the rows are asserted, which is what the MANIFEST bullet claims.
    q48, q64, q64d, q96, qw1 = (g("chiop48_n8"), g("chiop64_n8"), g("chiop64_dmrg32_n8"),
                                g("chiop96_n8"), g("chiop64_w1_n8"))
    q32 = g("sweep_knobs")

    def rc(q):
        return abs(q["ref_consistency_mpo"]) / abs(q["E_min_mpo"])

    # (1) chi_dmrg 16 vs 32 at chi_op 64: the same answer to 1e-4 relative
    if q64 is not None and q64d is not None:
        d, dd = q64["rel_dev_mpo_vs_ed_same_grid"], q64d["rel_dev_mpo_vs_ed_same_grid"]
        assert abs(d - dd) < 1e-4, (d, dd)
        assert q64d["weight_decay"] == 0.5 and q64d["chi_op"] == 64
    # (2) weight 0.5 -> 1 at chi_op 64 is worth a factor >= 5 in the error
    if q64 is not None and qw1 is not None:
        assert qw1["weight_decay"] == 1.0 and qw1["chi_op"] == 64
        assert abs(qw1["rel_dev_mpo_vs_ed_same_grid"]) < 0.02 < abs(q64["rel_dev_mpo_vs_ed_same_grid"])
        assert abs(q64["rel_dev_mpo_vs_ed_same_grid"]) / abs(qw1["rel_dev_mpo_vs_ed_same_grid"]) > 5.0
    # (3) the chi_op ladder: |error| falls, and it changes sign between 64 and 96
    if all(q is not None for q in (q32, q48, q64, q96)):
        devs = [q["rel_dev_mpo_vs_ed_same_grid"] for q in (q32, q48, q64, q96)]
        mags = [abs(x) for x in devs]
        assert mags == sorted(mags, reverse=True), mags
        assert devs[2] > 0 > devs[3], devs           # the sign change
        assert mags[0] / mags[1] > 2.0               # 32 -> 48 is the big step
        assert mags[1] - mags[2] < 0.02              # 48 -> 64 buys almost nothing
    # (4) the gate is not monotone in the error: the best refcons is the 13.7 % row, and the
    #     most accurate row is seven gates out
    if q64 is not None and qw1 is not None:
        assert rc(q64) < 0.05, rc(q64)                          # passes the gate
        assert abs(q64["rel_dev_mpo_vs_ed_same_grid"]) > 0.12   # and is 13.7 % wrong
        assert rc(qw1) > 0.35, rc(qw1)                          # fails the gate sevenfold
        assert abs(qw1["rel_dev_mpo_vs_ed_same_grid"]) < 0.02   # and is the most accurate
        assert rc(q64) < rc(qw1) and abs(q64["rel_dev_mpo_vs_ed_same_grid"]) > abs(qw1["rel_dev_mpo_vs_ed_same_grid"])
    if q96 is not None and q48 is not None:
        # the refcons even goes back UP (2.66 % -> 6.56 %) as the error falls 14.9 % -> 4.5 %
        assert rc(q96) > rc(q48) and abs(q96["rel_dev_mpo_vs_ed_same_grid"]) < abs(q48["rel_dev_mpo_vs_ed_same_grid"])
    # the lambda = 0 control at the same chi_op 64 / weight 0.5 knobs is a factor >= 5 better
    c0 = g("chiop64_n8", 0.0)
    if c0 is not None and q64 is not None:
        assert abs(c0["rel_dev_mpo_vs_ed_same_grid"]) < 0.02
        assert abs(q64["rel_dev_mpo_vs_ed_same_grid"]) / abs(c0["rel_dev_mpo_vs_ed_same_grid"]) > 5.0


def test_lambda4_mpo_error_grows_with_nmax_at_the_sweep_knobs():
    """FIX ROUND 2 (the verifier finding).  Round 1's MANIFEST and draft said the lambda = 4
    operator truncation at the sweep's knobs "is a 3-12 % effect wherever an exact answer
    exists".  That is false on this tree: the pair at L = 5, n_max 8 -- the sweep's OWN Fock
    cutoff, the cutoff every quoted S8 lambda = 4 ratio and the first plane point are taken
    at -- deviates by +39.4 %, and the plane's knobs (exact SVD) at the same point by
    +42.3 %.  The measured statement is that the lambda = 4 error GROWS with n_max at fixed
    chi_op 32 (3.4 % -> 11.9 % -> 39.4 % at n_max 5 / 6 / 8) while the lambda = 0 error at
    the same points stays at 0.3-1.4 %.  This test fails if any of that is weakened."""
    rec = _load("s8_ed_anchor.json")
    xv = rec["cross_validation_by_knobs"]
    if "sweep_knobs" not in xv:
        pytest.skip("no sweep-knob pair has landed")
    wbn = xv["sweep_knobs"]["worst_by_n_max"]
    seen = [k for k in _SWEEP_WORST_BY_NMAX_PINS if k in wbn]
    assert seen, "no pinned sweep-knob n_max has landed"
    for k in seen:
        want, tol = _SWEEP_WORST_BY_NMAX_PINS[k]
        assert abs(wbn[k] - want) <= tol, (k, wbn[k])
    # monotone growth in n_max over whatever has landed, and a factor >= 3 from n_max 5 to 8
    got = [wbn[k] for k in sorted(seen, key=int)]
    assert got == sorted(got), got
    if {"5", "8"} <= set(seen):
        assert wbn["8"] / wbn["5"] > 3.0, (wbn["5"], wbn["8"])
        assert wbn["8"] > 0.35, wbn["8"]
    pairs = {(p["knobs"], p["L"], p["n_max"], p["lam"]): p for p in rec["pairs_mpo_vs_ed"]}
    # at the SAME point the lambda = 0 error is 1.4 %: the two lambdas are a factor ~29 apart
    k0, k4 = ("sweep_knobs", 5, 8, 0.0), ("sweep_knobs", 5, 8, 4.0)
    if k0 in pairs and k4 in pairs:
        r0 = abs(pairs[k0]["rel_dev_mpo_vs_ed_same_grid"])
        r4 = abs(pairs[k4]["rel_dev_mpo_vs_ed_same_grid"])
        assert r0 < 0.02 < r4, (r0, r4)
        assert r4 / r0 > 20.0, r4 / r0
        # the MPO understates the magnitude of the lambda = 4 infimum by 1.6508x
        # (the plane knobs, exact SVD, by 1.7317x -- checked below)
        f = abs(pairs[k4]["E_min_ed_same_grid"]) / abs(pairs[k4]["E_min_mpo"])
        assert abs(f - 1.650834) < 1e-5, f
    kp = ("plane_knobs", 5, 8, 4.0)
    if kp in pairs and k4 in pairs:
        # the EXACT SVD makes the lambda = 4 disagreement larger, not smaller
        assert pairs[kp]["rel_dev_mpo_vs_ed_same_grid"] > pairs[k4]["rel_dev_mpo_vs_ed_same_grid"] > 0.35
        fp = abs(pairs[kp]["E_min_ed_same_grid"]) / abs(pairs[kp]["E_min_mpo"])
        assert abs(fp - 1.731665) < 1e-5, fp


# --------------------------------------------------------------------------
# the measured-dispersion reference at lambda = 2 (s7a had lambda 0 / 1 / 4 only)
# --------------------------------------------------------------------------

def test_keff_reference_at_lambda_2_is_pinned_and_between_its_neighbours():
    """The s7a pipeline verbatim (n_max 6, T = 10, dt 0.1, chi 32, r_max 4) at lambda = 2:
    E_free(K_eff)/E_H = 1.02104, between the cutoff-free s7b factors 1.0090 (lambda 1) and
    1.0452 (lambda 4); the quasiparticle is sharp (weight >= 0.9992, gamma/gap <= 2e-4, gap
    check 2e-4) and the band nearest-neighbour (J_1 = 0.99967, |J_{r>=2}| < 5e-5); mu_eff^2
    sits 2.1 % below the Hartree value.  Measured 2026-09-12 (242 s, 2 threads)."""
    pth = _DATA / "jobs" / "s8_keff_lam2.json"
    if not pth.exists():
        pytest.skip("the lambda = 2 keff row is not on this tree")
    with open(pth) as fh:
        r = json.load(fh)["row"]
    assert r["lam"] == 2.0 and r["L"] == 16 and r["n_max"] == 6 and r["tau0"] == 0.75
    assert _close(r["E_keff_over_E_hartree"], 1.0210388039135745, 1e-5)
    assert _close(r["E_keff"], -0.012122547930615069, 1e-5)
    assert _close(r["E_hartree_exact"], -0.01187275927628817, 1e-9)
    assert _close(r["mu2_eff"], 0.5995449260866912, 1e-4) and _close(r["mu_H2"], 0.6124520305136425, 1e-9)
    assert abs(r["J_eff"][0] - 0.99967) < 1e-4 and max(abs(j) for j in r["J_eff"][1:]) < 5e-5
    assert r["keff_misfit_max"] < 2e-4 and r["weight_min"] > 0.999 and r["gamma_over_gap_max"] < 3e-4
    assert abs(r["gap_check_dev"]) < 5e-4
    assert 1.0090 < r["E_keff_over_E_hartree"] < 1.0452, "the lambda = 2 factor must sit between the lambda 1 and 4 ones"
    rec = _load("s8_convergence_plane.json")
    assert _close(rec["keff_factor_by_lambda"]["lam=2"], r["E_keff_over_E_hartree"], 1e-12)
    assert _close(rec["keff_factor_by_lambda"]["lam=4"], 1.045182470073791, 1e-9)


# --------------------------------------------------------------------------
# S8 TRIAGE (2026-09-12): why the weight-0.5 exact-SVD plane was retired
# --------------------------------------------------------------------------

def _triage():
    return _load("s8_weight_triage.json")


def _pair(pairs, **sel):
    hit = [p for p in pairs if all(abs(p[k] - v) < 1e-12 if isinstance(v, float) else p[k] == v
                                   for k, v in sel.items())]
    if not hit:
        pytest.skip(f"no archived pair for {sel}")
    assert len(hit) == 1, f"{sel} is not a unique pair: {len(hit)}"
    return hit[0]


def _banded(rec, axis, tag_better, tag_worse):
    """One row of ``banded_penalties_l16_lam0``: the L = 16, lambda = 0 knob penalty with
    the route's own same-spec thread scatter attached to it."""
    hit = [z for z in rec.get("banded_penalties_l16_lam0", [])
           if z["axis"] == axis and z["tag_better"] == tag_better and z["tag_worse"] == tag_worse]
    if not hit:
        pytest.skip(f"no banded penalty row for {axis} {tag_better} vs {tag_worse}")
    assert len(hit) == 1
    return hit[0]


def test_exact_svd_is_a_cost_and_not_an_accuracy_gain():
    """The retired plane's distinguishing knob is ``svd_dense_limit = "inf"`` (an exact
    dense SVD in the operator truncation instead of the randomized sketch), and it is where
    the plane's cost lived.  At IDENTICAL (L, n_max, lambda, chi_op, chi_acc, chi_mpo,
    weight, dt, order, chi_dmrg, window_pad) it costs 3.0-22x and it does not improve the
    answer at either place where the truth is known:

      * L = 16, n_max 8, chi_op 32, weight 0.5, lambda = 0 -- the ONLY exactly-known number
        at the paper's own L, where exact/E_free must be 1 -- goes 1.001923 (sketched at 4
        BLAS threads) / 0.999931 (sketched at 2, the thread-matched leg) -> 1.022613 (exact
        SVD, 2 threads, 10074 s vs 1353 s, 7.4x the wall).  FIX ROUND 1: the penalty that
        may be QUOTED is the band-aware one, >= 11.8x, i.e. the exact SVD's 2.261 %
        distance from 1 over the LARGEST distance the sketched leg shows across archived
        thread counts (1.923e-3).  The thread-matched ratio is 328x, but its denominator
        (6.9e-5) is 29x below the 0.199 % same-spec thread scatter archived at this very
        spec, so the sketched leg is simply NOT RESOLVED from 1 and no ratio built on it
        alone is reproducible.  The exact-SVD leg is resolved: 2.261 % is 11x the band.
      * L = 5, n_max 8, chi_op 32, weight 0.5, lambda = 4 -- the only exactly-known lambda = 4
        number at the paper's Fock cutoff, exact E_min -2.1561126717e-02 -- goes +39.425 %
        (171 s) -> +42.252 % (3127 s): further from the exact value for 18.3x the wall.

    The exact SVD wins one of the four archived comparisons (L = 5, n_max 8, lambda = 0:
    +1.380 % -> -0.389 %) and loses the two that decide item 4.

    FIX ROUND 2.  Tolerance 1e-6 relative on every pinned value because each pin is on an
    ARCHIVED number in a finished job file, not on a re-run -- NOT because the route is
    reproducible to 1e-6.  It is not: the same spec gives -2.2652090e-02 at 2 BLAS threads
    and -2.2697211e-02 at 4 (0.199 %), and up to 5.120 % elsewhere in the archive.  That measured band
    is why the QUOTED penalty is the band-aware >= 11.8x and not the 328x raw ratio.
    NOTE the exact-SVD leg's OWN band is unmeasured (one archived thread count); 11.4x is
    against the SKETCHED leg's band -- see the assertions below."""
    rec = _triage()
    pairs = rec["svd_mode_pairs"]

    p = _pair(pairs, tag_sketched="plane2_nmax8_chiop32_w0.5_lam0",
              tag_exact_svd="plane_nmax8_chiop32_lam0")
    assert p["threads"] == "2", "the pair must be thread-matched: the route is thread-sensitive"
    assert _close(p["ratio_free_sketched"], 0.9999311124645787, 1e-6)
    assert _close(p["ratio_free_exact_svd"], 1.0226132795472942, 1e-6)
    assert _close(p["cost_ratio"], 7.44353429163232, 1e-4)
    # FIX ROUND 1.  The band-aware factor is the one that may be quoted.
    b = _banded(rec, "svd", "plane2_nmax8_chiop32_w0.5_lam0", "plane_nmax8_chiop32_lam0")
    assert _close(b["err_better_max"], 0.0019228899680709954, 1e-6)     # sketched, 4 threads
    assert _close(b["err_worse_min"], 0.022613279547294196, 1e-6)       # exact SVD
    assert _close(b["same_spec_band_better"], 0.001991914721588167, 1e-6)
    assert _close(b["factor_conservative"], 11.760048636573513, 1e-4)
    assert b["factor_conservative"] > 10.0, b["factor_conservative"]
    # the sketched leg's distance from 1 is INSIDE its own thread band: it is not resolved
    # from 1, which is why the 328x thread-matched ratio must not be quoted
    assert b["better_leg_resolved_above_its_band"] is False
    assert _close(b["factor_at_matched_threads"], 328.2637331846018, 1e-4)
    assert b["factor_at_matched_threads"] > 25.0 * b["factor_conservative"], (
        "the thread-matched ratio is an artefact of an unresolved denominator")
    # FIX ROUND 2 (verifier finding, upheld).  The exact-SVD leg's OWN band is UNMEASURED:
    # one archived thread count.  So "11.4x its own band, resolved" is NOT a statement this
    # archive supports, and the MANIFEST no longer makes it.  What IS measured: the exact
    # SVD's distance from 1 is 11.35x the SKETCHED leg's same-spec band at the same
    # (L, n_max, chi_op, lambda).  ASSUMPTION, unproved here: the exact-SVD route's thread
    # scatter is no larger than the sketched route's.  These three pins exist so that the
    # words "its own band" can never be re-derived from this archive; if a second thread
    # count of the exact-SVD spec ever lands they FAIL ON PURPOSE and must be updated to
    # the measured band, with a MANIFEST note.
    assert b["same_spec_band_worse"] is None, "the exact-SVD leg's own band is unmeasured"
    assert b["worse_leg_resolved_above_its_band"] is None
    assert b["n_threads_archived_worse"] == 1
    assert b["err_worse_min"] > 10.0 * b["same_spec_band_better"]   # vs the SKETCHED band
    assert _close(b["err_worse_min"] / b["same_spec_band_better"], 11.352533972571115, 1e-4)

    p = _pair(pairs, tag_sketched="anchor_sweepknobs_L5_nmax8_lam4",
              tag_exact_svd="anchor_planeknobs32_L5_nmax8_lam4")
    assert _close(p["E_min_exact_ed"], -0.021561126716953, 1e-9)
    assert _close(p["rel_dev_sketched"], 0.39424546741441363, 1e-6)
    assert _close(p["rel_dev_exact_svd"], 0.4225210557161616, 1e-6)
    assert 18.0 < p["cost_ratio"] < 19.0, p["cost_ratio"]
    # the exact SVD is FARTHER from the exact lambda = 4 value, not closer
    assert abs(p["rel_dev_exact_svd"]) > abs(p["rel_dev_sketched"])

    # the one comparison the exact SVD wins, pinned so the statement stays honest
    p = _pair(pairs, tag_sketched="anchor_sweepknobs_L5_nmax8_lam0",
              tag_exact_svd="anchor_planeknobs32_L5_nmax8_lam0")
    assert _close(p["rel_dev_sketched"], 0.013798157801797762, 1e-6)
    assert _close(p["rel_dev_exact_svd"], -0.0038916689012000287, 1e-6)
    assert abs(p["rel_dev_exact_svd"]) < abs(p["rel_dev_sketched"])

    # every archived same-knob pair costs at least 1.7x, and the plane's settings 3x
    assert min(p["cost_ratio"] for p in pairs) > 1.7
    assert all(p["cost_ratio"] > 3.0 for p in pairs if p["n_max"] >= 8 or p["chi_op_requested"] >= 32)


def test_weight_1_is_cheap_and_the_L_ranking_of_the_weight_axis_reverses():
    """Item B of the triage: what weight_decay = 1 COSTS, measured at the paper's own
    L = 16 at identical knobs, and what it does to the one exactly-known number there.

    COST.  Weight 1 costs 1.00-1.26x wherever chi_op saturates (n_max 8 chi_op 32: 2973 ->
    3367 s; n_max 6 chi_op 32: 593 -> 746 s; n_max 6 chi_op 32 with the EXACT SVD: 4686 ->
    4678 s, i.e. free, because there the dense SVD dominates and does not see the weight).
    It is not the expensive option the plan feared.

    ACCURACY.  At L <= 6 weight 1 is the better knob at lambda = 4 (L = 5, n_max 8,
    chi_op 64: +13.670 % -> -1.188 % against the exact ED value).  At L = 16, where
    exact/E_free must be 1 at lambda = 0, the SAME change goes the other way and by more:
    1.001923 -> 1.081776 at n_max 8, chi_op 32, i.e. 0.19 % -> 8.18 % off.  FIX ROUND 1:
    the factor that may be QUOTED is the band-aware >= 42.5x -- the weight-1 leg's SMALLEST
    distance from 1 over the weight-0.5 leg's LARGEST, across every archived thread count.
    The thread-matched pair gives 1190x, but its denominator (6.9e-5) is 29x below the
    0.199 % same-spec thread scatter at that spec, so the weight-0.5 leg is not resolved
    from 1 at all and 1190x is not reproducible.  The weight-1 leg IS resolved: 8.18 %
    against its own 0.029 % band.  The same sign holds at n_max 6 (chi_op 32: 1.011667 ->
    1.035981).  So the candidate's standing ASSUMPTION -- that the L = 5 knob ranking
    carries to L = 16 -- is measured FALSE on the weight axis at lambda = 0, and no single
    weight can be fixed for the plane.  This is the reason plane2 runs weight as an AXIS
    (0.5 and 1.0 both) instead of choosing one."""
    rec = _triage()
    cp = rec["l16_weight_cost_pairs"]

    # the thread-matched pair (both 2 threads, both from plane2)
    p = _pair(cp, tag_lo="plane2_nmax8_chiop32_w0.5_lam0", tag_hi="plane2_nmax8_chiop32_w1_lam0")
    assert p["threads"] == "2"
    assert 1.05 < p["cost_ratio_hi_over_lo"] < 1.30, p["cost_ratio_hi_over_lo"]
    assert _close(p["ratio_free_lo"], 0.9999311124645787, 1e-6)
    assert _close(p["ratio_free_hi"], 1.0820856372481684, 1e-6)
    assert p["chi_op_reached_lo"] == p["chi_op_reached_hi"] == 32
    # FIX ROUND 1: the band-aware factor, not the 1190x thread-matched one
    b = _banded(rec, "weight_decay", "plane2_nmax8_chiop32_w0.5_lam0",
                "plane2_nmax8_chiop32_w1_lam0")
    assert _close(b["err_better_max"], 0.0019228899680709954, 1e-6)   # weight 0.5, 4 threads
    assert _close(b["err_worse_min"], 0.08177591721378796, 1e-6)      # weight 1.0, 4 threads
    assert _close(b["same_spec_band_better"], 0.001991914721588167, 1e-6)
    assert _close(b["same_spec_band_worse"], 0.0002869740930777923, 1e-6)
    assert _close(b["factor_conservative"], 42.52761134108153, 1e-4)
    assert b["factor_conservative"] > 20.0, b["factor_conservative"]
    # weight 0.5's distance from 1 is inside its own thread band (not resolved); weight 1's
    # is 285x its own band (resolved), so the DIRECTION is solid and the 1190x is not
    assert b["better_leg_resolved_above_its_band"] is False
    assert b["worse_leg_resolved_above_its_band"] is True
    assert _close(b["factor_at_matched_threads"], 1191.5995887279182, 1e-4)
    assert b["factor_at_matched_threads"] > 25.0 * b["factor_conservative"]
    # FIX ROUND 2 (verifier finding, upheld).  The 4-thread pair is NOT "band-clearing on
    # both legs": its thread-matched ratio coincides with the band-aware lower bound only
    # because the 4-thread leg IS the larger of the weight-0.5 leg's two archived distances
    # from 1.  That leg is unresolved from 1 at BOTH thread counts, so 42.5276x is quotable
    # as a LOWER BOUND on the weight penalty -- never as a resolved ratio.
    b4 = _banded(rec, "weight_decay", "sweep_lam0", "lam0_best_nmax8_chiop32_wd1")
    assert _close(b4["factor_at_matched_threads"], 42.52761134108153, 1e-4)
    assert b4["better_leg_resolved_above_its_band"] is False   # the denominator leg
    assert b4["err_better_max"] < b4["same_spec_band_better"]  # strictly inside its band
    assert b4["worse_leg_resolved_above_its_band"] is True     # only the numerator clears
    # the coincidence itself, pinned: matched-thread ratio == band-aware lower bound here
    assert _close(b4["factor_conservative"], b4["factor_at_matched_threads"], 1e-9)
    # the 4-thread pair of the same comparison, kept because the archive quotes the
    # sweep's 1.001923 control: same sign, same size
    q = _pair(cp, tag_lo="sweep_lam0", tag_hi="lam0_best_nmax8_chiop32_wd1")
    assert _close(q["ratio_free_lo"], 1.001922889968071, 1e-6)
    assert _close(q["ratio_free_hi"], 1.081775917213788, 1e-6)

    p = _pair(cp, tag_lo="lam0_chiop32_wd0.5", tag_hi="lam0_chiop32_wd1")
    assert _close(p["ratio_free_lo"], 1.0116673712521354, 1e-6)
    assert _close(p["ratio_free_hi"], 1.0359806612796647, 1e-6)
    assert 1.1 < p["cost_ratio_hi_over_lo"] < 1.4

    # with the exact SVD the weight is free: the dense SVD dominates the cost
    p = _pair(cp, tag_lo="lam0_exactsvd_chiop32_wd0.5", tag_hi="lam0_exactsvd_chiop32_wd1")
    assert 0.95 < p["cost_ratio_hi_over_lo"] < 1.05, p["cost_ratio_hi_over_lo"]

    # and the L <= 6 ladder that says the opposite, at lambda = 4
    lad = rec["weight_ladders_multi"].get("L5/n8/lam4/chi_op64/chi_dmrg16/sketched/thr2")
    if lad is None:
        pytest.skip("the L = 5 n_max 8 chi_op 64 weight ladder is not on this tree")
    by_w = {row[0]: row[1] for row in lad}
    assert _close(by_w[0.5], 0.13669549608662926, 1e-6), by_w
    assert _close(by_w[1.0], -0.011876989769345665, 1e-6), by_w
    assert abs(by_w[0.5]) / abs(by_w[1.0]) > 10.0


def test_the_plane_is_retired_and_plane2_is_what_is_queued():
    """The triage decision, as the plan file records it.  The queued plane (18 remaining
    jobs, weight 0.5, exact SVD) is gone from the default part set; plane2 replaces it with
    the same (n_max, chi_op) grid at lambda 0/2/4, sketched SVD, and weight_decay as an
    explicit axis at 0.5 AND 1.0, and every plane2 column has an exact-ED-anchored weight
    ladder row in part 'wt' at the same chi_op and weight."""
    plan = _load("s8_plan.json")
    jobs = plan["jobs"]
    parts = {j["part"] for j in jobs}
    assert "plane" not in parts, "the retired plane is still being queued"
    assert {"plane2", "wt"} <= parts

    p2 = [j for j in jobs if j["part"] == "plane2"]
    assert len(p2) == 36
    assert {j["kw"]["weight_decay"] for j in p2} == {0.5, 1.0}
    assert {j["n_max"] for j in p2} == {8, 10, 12}
    assert {j["kw"]["chi_op"] for j in p2} == {32, 64}
    assert {j["lam"] for j in p2} == {0.0, 2.0, 4.0}
    # plane2 is sketched throughout: that is the entire cost saving
    assert all("svd_dense_limit" not in j["kw"] for j in p2)
    # and it is cheaper than the plane it replaces by more than a factor 8 in the same model
    assert sum(j["est_wall_s"] for j in p2) / 3600.0 < 50.0

    # FIX ROUND 1: the saving must be quoted like for like.  The first pass calibrated the
    # RETIRED plane (x2.33, from its two landed exact-SVD jobs) and quoted its REPLACEMENT
    # at nominal, which flattered the saving to ~19x.  The wall model under-estimates the
    # L = 16 plane CLASS generally -- the landed sketched plane2 jobs run x1.68 median --
    # so the exact-SVD-specific excess is only x1.39 on top, and the honest saving is ~12x.
    cal = _triage()["wall_model_calibration"]
    pj = cal["per_job"]
    # the landed jobs on both sides, pinned exactly (a finished job file never changes)
    for tag, t_s, e_s in (("plane_nmax8_chiop32_lam0", 10074.165363311768, 4248.675),
                          ("plane_nmax8_chiop32_lam4", 9694.36689901352, 4248.675),
                          ("plane2_nmax8_chiop32_w0.5_lam0", 1353.4115607738495, 900.0),
                          ("plane2_nmax8_chiop32_w0.5_lam2", 1512.7329609394073, 900.0),
                          ("plane2_nmax8_chiop32_w0.5_lam4", 1509.0339291095734, 900.0),
                          ("plane2_nmax8_chiop32_w1_lam0", 1548.9580450057983, 900.0),
                          ("plane2_nmax8_chiop32_w1_lam2", 1155.5469498634338, 900.0),
                          ("plane2_nmax8_chiop32_w1_lam4", 1692.4009990692139, 900.0),
                          ("plane2_nmax8_chiop64_w0.5_lam0", 2298.0485949516296, 2545.584412271571)):
        if tag not in pj:
            pytest.skip(f"{tag} has not landed on this tree")
        assert _close(pj[tag]["time_s"], t_s, 1e-6), tag
        assert _close(pj[tag]["est_wall_s"], e_s, 1e-6), tag
    # the frozen fix-round-1 tag set: the live per_part medians keep moving as the pool
    # lands more plane2 rows, so the QUOTED comparison is recomputed over fixed tags
    f1 = cal["fix_round_1"]
    assert f1["plane"]["n"] == 2 and f1["plane2"]["n"] == 7
    assert set(f1["tags"]["plane2"]) <= set(pj) and set(f1["tags"]["plane"]) <= set(pj)
    # the plane's exact-SVD rows run 2.28x / 2.37x; the sketched plane2 rows 0.90-1.88x,
    # median 1.68 -- so the model understates this whole class, not just the exact SVD
    assert _close(f1["plane"]["median"], 2.3264349782373666, 1e-6)
    assert _close(f1["plane2"]["median"], 1.6767043656773037, 1e-6)
    assert _close(f1["exact_svd_excess_over_sketched_plane_class"], 1.3875045749628048, 1e-6)
    assert 1.3 < f1["exact_svd_excess_over_sketched_plane_class"] < 1.5
    # calibrated on BOTH sides: 368 h nominal -> 856 h for the retired plane, 43.6 h
    # nominal -> 73 h for plane2, a saving of ~12x and not the ~19x that 856/43.6 implies
    assert _close(f1["plane_retired_remaining_calibrated_h"], 856.1280719913509, 1e-6)
    assert _close(f1["plane2_nominal_h"], 43.556017148775645, 1e-6)
    assert _close(f1["plane2_calibrated_h"], 73.03056410486764, 1e-6)
    assert _close(f1["saving_factor_like_for_like"], 11.7228736007297, 1e-6)
    assert 11.0 < f1["saving_factor_like_for_like"] < 13.0
    # the saving quoted at nominal-vs-calibrated would have been ~19x: at least 1.5x larger
    assert f1["saving_factor_like_for_like"] < 0.7 * (
        f1["plane_retired_remaining_calibrated_h"] / f1["plane2_nominal_h"])
    # and most of plane2's estimate is wall-model EXTRAPOLATION: no sketched job above
    # n_max 8 has ever been timed on this tree, and n_max 10 + 12 are 87 % of the estimate
    assert _close(f1["plane2_hours_that_are_wall_model_extrapolation"], 37.813376461656354, 1e-6)
    assert f1["plane2_hours_that_are_wall_model_extrapolation"] > 0.85 * f1["plane2_nominal_h"]
    assert max(j["n_max"] for j in jobs
               if j["part"] == "plane2" and j["tag"] in f1["tags"]["plane2"]) == 8

    wt = [j for j in jobs if j["part"] == "wt"]
    assert len(wt) == 72
    assert {j["L"] for j in wt} == {5, 6} and {(j["L"], j["n_max"]) for j in wt} == {(5, 8), (6, 5), (5, 6)}
    assert {j["kw"]["weight_decay"] for j in wt} == {0.25, 0.5, 0.75, 1.0}
    assert {j["lam"] for j in wt} == {0.0, 2.0, 4.0}
    # every (chi_op, weight) column of plane2 is anchored at all three ED points
    cols = {(j["kw"]["chi_op"], j["kw"]["weight_decay"], j["lam"]) for j in p2}
    anch = {(j["kw"]["chi_op"], j["kw"]["weight_decay"], j["lam"], j["L"], j["n_max"]) for j in wt}
    for chi, w, lam in cols:
        for pt in ((5, 8), (6, 5), (5, 6)):
            assert (chi, w, lam) + pt in anch, f"plane2 column {(chi, w, lam)} has no anchor at {pt}"
    # the lambda = 2 exact references the anchors need
    ed2 = [j for j in jobs if j["part"] == "ed" and j["lam"] == 2.0]
    assert {(j["L"], j["n_max"]) for j in ed2} == {(5, 6), (6, 5), (5, 8)}


# the four-rung weight ladder at the one point where an exact lambda = 4 answer exists at
# the paper's own Fock cutoff, all four rungs run at 1 BLAS thread, sketched SVD, chi_op 64,
# chi_dmrg 16, seed 0; exact (matrix-free Lanczos) E_min = -2.1561126717e-02
_W_LADDER_L5N8_LAM4_THR1 = {0.25: 0.3861339199196462, 0.5: 0.15717644493670926,
                            0.75: -0.11834189969508263, 1.0: 0.004632221615357096}
# the same ladder at lambda = 2 and lambda = 0, same knobs, same 1 BLAS thread.  Exact
# (matrix-free Lanczos, tau0 0.75, the MPO route's own trapezoid grid):
#   lambda = 2  -9.388966078354377e-03    lambda = 0  -1.8462671543e-02
_W_LADDER_L5N8_THR1 = {
    2.0: {0.25: 0.014354008227815197, 0.5: 0.007529539138597015,
          0.75: -0.6333091756800348, 1.0: -0.008452609894830694},
    0.0: {0.5: 0.01268003914462411, 0.75: -0.01056096183821696, 1.0: -0.002722759840421597},
}


def test_the_weight_axis_is_non_monotone_like_chi_op():
    """S8 TRIAGE.  Yesterday's reading -- "weight_decay is the limiter at lambda >= 2" --
    rested on two rungs (0.5 and 1.0) at one point.  The full four-rung ladder at that
    point changes what can be said: the deviation from the exact value runs

        w 0.25  +38.613 %    w 0.5  +15.718 %    w 0.75  -11.834 %    w 1.0  +0.463 %

    with a sign change between 0.5 and 0.75 and BACK between 0.75 and 1.0.  That is the
    same behaviour the chi_op axis has (+39.4 % -> -4.5 % with a sign change between 64 and
    96), so no weight is a one-sided approximation and w = 1 being closest here is a
    statement about an endpoint, not about a convergence direction.  This is why plane2
    runs weight_decay as an AXIS instead of choosing a value.

    Tolerance 1e-5 relative on each rung: the rungs differ from each other by >= 1e-2 and
    the route's own run-to-run spread at these knobs is ~2e-2 (pinned in the test below),
    so 1e-5 catches a changed knob and does not catch a re-run."""
    rec = _triage()
    lad = rec["weight_ladders_multi"].get("L5/n8/lam4/chi_op64/chi_dmrg16/sketched/thr1")
    if lad is None:
        pytest.skip("the 1-thread weight ladder at L 5 n_max 8 chi_op 64 lambda = 4 is not on this tree")
    by_w = {row[0]: row[1] for row in lad}
    for w, dev in _W_LADDER_L5N8_LAM4_THR1.items():
        if w not in by_w:
            pytest.skip(f"weight {w} rung has not landed")
        assert _close(by_w[w], dev, 1e-5), f"w {w}: {by_w[w]!r}"
    # the verdict logic: non-monotone, with a sign change in the interior
    devs = [by_w[w] for w in sorted(by_w)]
    assert devs[0] > devs[1] > 0 > devs[2] and devs[3] > devs[2], devs
    # w = 1 is the closest rung, but by less than the interior swing
    assert abs(by_w[1.0]) == min(abs(v) for v in by_w.values())
    assert abs(by_w[0.75]) > 10 * abs(by_w[1.0])


def test_the_route_is_not_reproducible_across_blas_threads_at_lambda_4():
    """S8 TRIAGE, and a correction to this file's own stated tolerance rationale.  Two
    jobs whose specs are identical except for the worker's BLAS thread count -- the
    randomized sketch is seeded at 0 in both -- give different answers at lambda = 4:

        L 5, n_max 8, chi_op 64, weight 0.5, sketched:  -1.8613818e-02 (2 thr)
                                                        -1.8172225e-02 (1 thr)   2.430 %
        L 5, n_max 8, chi_op 64, weight 1.0, sketched:  -2.1817208e-02 (2 thr)
                                                        -2.1461251e-02 (1 thr)   1.659 %

    The truncation decisions amplify the non-associativity of the threaded reductions (the
    operator MPO reaches chi_mpo 240 on one run and 244 on the other).  So this file's old
    rationale, "the MPO route is deterministic at a fixed seed and BLAS threading moves it
    by < 1e-8", is false: there is a 1.7-2.4 % floor under every lambda = 4 magnitude at
    these knobs, larger than the weight-1 advantage (0.46-1.19 %) it would be used to
    establish.  FIX ROUND 2: it is not true at lambda = 0 either, only smaller -- the same
    spec at L = 16, n_max 8, chi_op 32, w 0.5, lambda = 0 moves 0.199 % between 2 and 4
    threads (asserted below), which is 2e7 times the claimed 1e-8 and is exactly the band
    that makes the first triage pass's 328x / 1190x unquotable.  Nothing in this candidate quotes a lambda = 4 magnitude, so no archived
    number is retracted -- but any future one must carry this floor."""
    rec = _triage()
    reps = rec.get("repeat_runs_same_spec")
    if not reps:
        pytest.skip("no same-spec repeat group is on this tree")
    got = {(z["L"], z["n_max"], z["lam"], z["chi_op_requested"], z["weight_decay"]): z for z in reps}
    for w, spread in ((0.5, 0.02430039920816168), (1.0, 0.0165860416051393)):
        z = got.get((5, 8, 4.0, 64, w))
        if z is None:
            pytest.skip(f"the weight {w} repeat group has not landed")
        assert _close(z["rel_spread_E_min"], spread, 1e-5), (w, z["rel_spread_E_min"])
        assert {r["threads"] for r in z["runs"]} == {"1", "2"}
    # at weight 0.5 the two runs kept a different operator bond (244 vs 240); at weight 1
    # both saturate the chi_mpo cap of 256 and the difference is in WHICH directions were
    # kept, not how many
    assert {r["chi_mpo_reached"] for r in got[(5, 8, 4.0, 64, 0.5)]["runs"]} == {240, 244}
    assert {r["chi_mpo_reached"] for r in got[(5, 8, 4.0, 64, 1.0)]["runs"]} == {256}
    # the floor is bigger than the weight-1 advantage it would certify
    z1 = got[(5, 8, 4.0, 64, 1.0)]
    assert z1["rel_spread_E_min"] > max(abs(r["rel_dev_vs_ed"]) for r in z1["runs"])
    # and the floor is not confined to L = 5: the sweep's OWN lambda = 2 row at L = 16
    # (n_max 8, chi_op 32, weight 0.5, sketched) moves 2.031 % between 2 and 4 threads,
    # -1.1347773e-02 vs -1.1578241e-02, while its lambda = 0 control moves 0.199 % and its
    # lambda = 4 row 0.142 %.  So the "0.19 % lambda = 0 control at the sweep's settings"
    # that this candidate quotes is the size of the route's own thread scatter.
    l16 = {(z["lam"], z["weight_decay"]): z for z in reps
           if z["L"] == 16 and z["n_max"] == 8 and z["chi_op_requested"] == 32}
    if (2.0, 0.5) in l16:
        assert _close(l16[(2.0, 0.5)]["rel_spread_E_min"], 0.020309498370027646, 1e-5)
        assert _close(l16[(0.0, 0.5)]["rel_spread_E_min"], 0.001991914721588167, 1e-5)
        assert _close(l16[(4.0, 0.5)]["rel_spread_E_min"], 0.001423160575388614, 1e-5)
        assert {r["threads"] for r in l16[(2.0, 0.5)]["runs"]} == {"2", "4"}


def test_the_weighting_is_not_the_limiter_at_lambda_2():
    """S8 TRIAGE, and the correction the triage was run to find.  The standing reading was
    "weight_decay, not chi_op, is the limiter at lambda >= 2", inferred from two rungs at
    lambda = 4.  It is not a lambda >= 2 statement.  At the SAME point and knobs (L = 5,
    n_max 8, chi_op 64, chi_dmrg 16, sketched SVD, 1 BLAS thread), against the exact
    matrix-free Lanczos values -9.388966078354377e-03 (lambda = 2, fine-grid quadrature
    check 8.6e-7) and -1.8462671543e-02 (lambda = 0):

        lambda = 2:  w 0.25 +1.435 %   w 0.5  +0.753 %   w 0.75  -63.331 %   w 1.0  -0.845 %
        lambda = 0:                     w 0.5  +1.268 %   w 0.75   -1.056 %   w 1.0  -0.272 %
        lambda = 4:  w 0.5 +15.718 %   w 0.75  -11.834 %   w 1.0  +0.463 %  (w 0.25 +38.613 %)

    At lambda = 2 BOTH ends of the weight axis are already inside 1 % of the exact value --
    the weighting costs nothing there -- while at lambda = 4 weight 0.5 is 15.7 % out.  The
    weight effect is a lambda = 4 effect at this point, not a lambda >= 2 effect.  The
    w = 0.75 rung at lambda = 2 is a -63 % outlier: the axis is not merely non-monotone, it
    has an interior catastrophe, and that rung is the one row the 5 % reference-consistency
    gate does catch (refcons/|E_min| 115.7 %) -- while the same gate rejects the ACCURATE
    w = 1 row (44.8 % against a 0.85 % error), so it still certifies nothing."""
    rec = _triage()
    for lam, pins in _W_LADDER_L5N8_THR1.items():
        key = f"L5/n8/lam{lam:g}/chi_op64/chi_dmrg16/sketched/thr1"
        lad = rec["weight_ladders_multi"].get(key)
        if lad is None:
            pytest.skip(f"{key} is not on this tree")
        by_w = {row[0]: row[1] for row in lad}
        for w, dev in pins.items():
            assert w in by_w, f"{key}: weight {w} has not landed"
            assert _close(by_w[w], dev, 1e-5), f"{key} w {w}: {by_w[w]!r}"
    # the verdict: at lambda = 2 both ends are inside 1 %, at lambda = 4 weight 0.5 is not
    l2, l4 = _W_LADDER_L5N8_THR1[2.0], _W_LADDER_L5N8_LAM4_THR1
    assert abs(l2[0.5]) < 0.01 and abs(l2[1.0]) < 0.01 and abs(l2[0.25]) < 0.02
    assert abs(l4[0.5]) > 0.15 and abs(l4[0.5]) / abs(l2[0.5]) > 15.0
    # the interior catastrophe at lambda = 2, and the gate that half-works
    assert l2[0.75] < -0.5
    rows = {(r["lam"], r["weight_decay"]): r for r in rec["rows"]
            if r["L"] == 5 and r["n_max"] == 8 and r["chi_op_requested"] == 64
            and r["svd"] == "sketched" and r["threads"] == "1"}
    assert rows[(2.0, 0.75)]["refcons_over_E_min"] > 1.0        # caught
    assert rows[(2.0, 1.0)]["refcons_over_E_min"] > 0.05        # rejected, but accurate to 0.85 %
    assert abs(rows[(2.0, 1.0)]["rel_dev_vs_ed"]) < 0.01


def test_the_thread_scatter_is_a_lambda_4_effect_only():
    """The 1.7-2.4 % same-spec spread of the test above is not a property of the route at
    every lambda: at lambda = 0, the identical comparison (L 5, n_max 8, chi_op 64,
    weight 0.5, sketched; -1.8239597e-02 on 2 threads vs -1.8228564e-02 on 1) has a spread
    of 0.061 %, forty times smaller.  So the lambda = 0 control does not see the floor that
    limits lambda = 4 -- the same shape as the rest of this candidate's lambda = 0 / lambda = 4
    story, and one more reason the control certifies nothing about the magnitude."""
    rec = _triage()
    reps = {(z["lam"], z["weight_decay"]): z for z in rec.get("repeat_runs_same_spec", [])
            if z["L"] == 5 and z["n_max"] == 8 and z["chi_op_requested"] == 64}
    if (0.0, 0.5) not in reps or (4.0, 0.5) not in reps:
        pytest.skip("the lambda = 0 / lambda = 4 repeat pair is not both on this tree")
    assert _close(reps[(0.0, 0.5)]["rel_spread_E_min"], 0.000605232481348943, 1e-5)
    assert _close(reps[(4.0, 0.5)]["rel_spread_E_min"], 0.02430039920816168, 1e-5)
    assert reps[(4.0, 0.5)]["rel_spread_E_min"] / reps[(0.0, 0.5)]["rel_spread_E_min"] > 30.0


def test_the_exact_svd_route_is_not_the_noisier_route_where_both_bands_are_measured():
    """FIX ROUND 2, verifier finding (a).  The band-aware exact-SVD penalty at L = 16,
    n_max 8, chi_op 32, lambda = 0 (>= 11.8x) is quoted against the SKETCHED leg's 0.199 %
    same-spec band, because the exact-SVD leg of that spec has only ONE archived thread
    count.  The ASSUMPTION that closes the gap -- the exact-SVD route's same-spec scatter is
    no larger than the sketched route's -- is measured here, at the cheapest exact-SVD spec
    at the paper's own L: L 16, n_max 6, chi_op 16, chi_acc 32, chi_mpo 64, weight 0.5,
    tau0 0.75, m 0.5, lambda 0, chi_dmrg 16, each route run at 2 AND at 4 BLAS threads
    (jobs svdband_{sketched,exactsvd}_chiop16_wd0.5_thr4, this round).  Both routes come
    back BIT-IDENTICAL across the thread counts:

        sketched   E_min -2.2800211601990195e-02 at 2 threads (438.4 s) and at 4 (159.1 s)
        exact SVD  E_min -2.2865136075353676e-02 at 2 threads (767.8 s) and at 4 (353.9 s)

    Both same-spec bands are exactly 0.0, so the exact SVD is NOT the noisier route where
    both are measured, and the assumption is supported at this spec.

    What this does NOT establish, stated so no one reads more into it:
      * nothing about the n_max 8 / chi_op 32 spec itself, whose exact-SVD leg still has one
        archived thread count (the pins in test_exact_svd_is_a_cost_and_not_an_accuracy_gain
        assert that null band on purpose);
      * that the 0.199 % spread at n_max 8 / chi_op 32 is a THREAD effect.  Here the thread
        count alone changes nothing whatsoever, so that spread is an unexplained same-spec
        difference between two archived jobs that differ in thread count AND in when they
        were run.  It stays a conservative noise floor, and "thread band" stays an
        ASSUMPTION until one spec is run twice at the SAME thread count."""
    rec = _triage()
    reps = {z["svd"]: z for z in rec.get("repeat_runs_same_spec", [])
            if z["L"] == 16 and z["n_max"] == 6 and z["lam"] == 0.0
            and z["chi_op_requested"] == 16 and z["weight_decay"] == 0.5}
    if {"sketched", "exact"} - set(reps):
        pytest.skip("the chi_op 16 two-thread-count pair is not both on this tree")
    for mode, e_min, times in (("sketched", -0.022800211601990195, {438.4338638782501, 159.05424284934998}),
                               ("exact", -0.022865136075353676, {767.7991161346436, 353.88435196876526})):
        z = reps[mode]
        assert {r["threads"] for r in z["runs"]} == {"2", "4"}, z["runs"]
        # bit-identical, not merely close: the spread is exactly zero
        assert z["rel_spread_E_min"] == 0.0, (mode, z["rel_spread_E_min"])
        assert {r["E_min"] for r in z["runs"]} == {e_min}, (mode, z["runs"])
        assert {round(r["time_s"], 6) for r in z["runs"]} == {round(t, 6) for t in times}
    # the assumption, as an inequality on the two measured bands
    assert reps["exact"]["rel_spread_E_min"] <= reps["sketched"]["rel_spread_E_min"]
    # and the archive's own band fields for that comparison, both measured, both resolved
    b = _banded(rec, "svd", "lam0_chiop16_wd0.5", "lam0_exactsvd_chiop16_wd0.5")
    assert b["same_spec_band_better"] == 0.0 and b["same_spec_band_worse"] == 0.0
    assert b["n_threads_archived_better"] == b["n_threads_archived_worse"] == 2
    assert b["better_leg_resolved_above_its_band"] is True
    assert b["worse_leg_resolved_above_its_band"] is True
    assert _close(b["factor_conservative"], 1.4429846009180611, 1e-6)
    # the exact SVD still costs more at BOTH thread counts at this spec (1.75x at 2, 2.22x
    # at 4); wall under pool contention is an upper bound, so these are indicative only
    pairs = {p["threads"]: p for p in rec["svd_mode_pairs"]
             if p["L"] == 16 and p["n_max"] == 6 and p["chi_op_requested"] == 16
             and p["weight_decay"] == 0.5}
    assert _close(pairs["2"]["cost_ratio"], 1.751231324476012, 1e-6)
    assert _close(pairs["4"]["cost_ratio"], 2.2249287138096077, 1e-6)

    # FIX ROUND 2, the other end of the same statement: the archive's WORST same-spec
    # spread, now that the wt grid is complete.  Pinned on the group (a finished job file
    # never changes) and as a lower bound on the archive-wide field, which may only grow as
    # the pools land more repeats -- if it grows, the MANIFEST row must be restated.
    worst = next((z for z in rec["repeat_runs_same_spec"]
                  if z["L"] == 5 and z["n_max"] == 6 and z["lam"] == 4.0
                  and z["chi_op_requested"] == 32 and z["weight_decay"] == 0.5
                  and z["svd"] == "sketched"), None)
    if worst is not None:
        assert _close(worst["rel_spread_E_min"], 0.05120232863008305, 1e-6)
        assert {r["threads"] for r in worst["runs"]} == {"1", "2"}
        assert {r["E_min"] for r in worst["runs"]} == {-0.010049170457017143,
                                                       -0.010563711385217056}
        assert rec["worst_rel_spread_same_spec"] >= worst["rel_spread_E_min"] - 1e-12
        # 26x the 0.199 % band the L = 16 factors are quoted against: the floor is
        # spec-dependent, and 0.0 at the chi_op 16 spec measured above
        assert worst["rel_spread_E_min"] > 25.0 * 0.001991914721588167


# ---------------------------------------------------------------------------
# MANIFEST item 4 (unit 2 of the S8 triage, 2026-09-13): the lambda >= 2 verdict.
#
# WHAT IS PINNED AND WHY IT IS SAFE AS THE POOL LANDS MORE ROWS.  Two kinds of number
# live in data/s8_convergence_plane2.json.  (a) PER-SETTING values -- E_min, the two
# ratios, the control-normalized residual, the ED-anchor maximum, the reference
# inconsistency -- are functions of finished job records only, so they never change and
# are pinned to 1e-9 relative.  (b) AGGREGATES -- the bracket, the systematic, the
# qualified counts -- move as the plane fills, so they are RECOMPUTED from the record's
# own rows inside the test and the stored value is asserted equal to the recomputation.
# That pins the verdict LOGIC without going stale.  The one-sided claims ("not resolved",
# "systematic >= the archived weight-pair spread") are stable because the weight pair that
# carries the systematic is already landed and immutable: a spread that is a max over
# pairs cannot shrink when pairs are added.
# ---------------------------------------------------------------------------

_P2 = "s8_convergence_plane2.json"


def _p2_frozen(rec, key):
    """The frozen unit-2 verdict the MANIFEST quotes, or None.

    The live `verdict` moves as the pool lands plane2 rows; `unit2_snapshot.verdict` is
    computed over a tag list frozen when the prose was written (rows whose own lambda = 0
    control is also in the list), so every number the MANIFEST quotes stays reproducible."""
    return (rec.get("unit2_snapshot") or {}).get("verdict", {}).get(key)



def _p2_rows(rec):
    return {(q["n_max"], q["chi_op"], q["weight_decay"], q["lam"]): q for q in rec["rows"]}


def test_item4_plane2_rows_are_pinned_at_their_knobs():
    """Every landed (n_max, chi_op, weight, lambda) setting the MANIFEST quotes, exactly.

    These are archived job numbers: E_min never changes, so neither does any ratio built
    from it.  Rows that have not landed are skipped one by one, never as a group."""
    rec = _load(_P2)
    by = _p2_rows(rec)
    # (n_max, chi_op, w, lam): E_min, exact/E_H, exact/K_eff, R_ctrl, R_keff, anchor, refcons
    want = {
        (8, 32, 0.5, 2.0): (-0.011347772988336446, 0.9557822848307718, 0.9360881106254936,
                            0.04415186914725666, 0.06384740012912304, 0.02330351842109992, 0.20299),
        (8, 32, 0.5, 4.0): (-0.006653794896693599, 0.8846354479991766, 0.8463933077032189,
                            0.11530360744674428, 0.15354838233098655, 0.39424546741441363, 0.05271),
        (8, 32, 1.0, 2.0): (-0.013078182880103428, 1.1015285137821909, 1.0788311957979506,
                            -0.017967285832743096, 0.0030082285502357475, 0.18214065906271934, 1.98908),
        (8, 32, 1.0, 4.0): (-0.010934557025711111, 1.4537714046339547, 1.39092593519227,
                            -0.3434892628563455, -0.2854112093570549, 0.26424333785866316, 2.57288),
        (8, 64, 0.5, 2.0): (-0.011482240958593404, 0.9671080404641326, 0.9471804957434243,
                            0.032173442280004516, 0.0521157922594323, 0.015514328088979068, 0.10959),
        (8, 64, 0.5, 4.0): (-0.0063786924734494654, 0.8480600261217806, 0.8113990144342036,
                            0.15130990387850818, 0.18799815302913248, 0.3448863999185752, 0.07244),
        (8, 64, 1.0, 2.0): (-0.012195538237004366, 1.0271865160578837, 1.0060210367331246,
                            -0.00985492032073032, 0.010953436392404425, 0.021384171291304554, 0.15914),
        (8, 64, 1.0, 4.0): (-0.0077830531522122826, 1.0347726100679913, 0.9900401505920159,
                            -0.0173130150701859, 0.026664678945139242, 0.016715069210005765, 1.13835),
        (10, 32, 0.5, 2.0): (-0.012147128751772818, 1.0231091584609662, 1.0020276942849342,
                            -0.010556318051525793, 0.01026649116749534, 0.02330351842109992, 0.26315),
        (10, 32, 0.5, 4.0): (-0.006792320133510499, 0.9030526575515186, 0.8640143548214716,
                            0.10802717278507756, 0.14658650259227168, 0.39424546741441363, 0.27699),
        (10, 32, 1.0, 4.0): (-0.011362415475387522, 1.510656048237533, 1.4453514974575483,
                            -0.30746251343730235, -0.2509418698392387, 0.26424333785866316, 1.09627),
    }
    seen = 0
    for key, (e, rh, rk, R, Rk, anc, refc) in want.items():
        if key not in by:
            continue
        q = by[key]
        if q["residual_ctrl_normalized"] is None:
            # landed without its own lambda = 0 control yet; the ratios are still pinnable
            assert _close(q["E_min"], e, 1e-9), (key, q["E_min"])
            assert _close(q["ratio_exact_over_hartree_exact"], rh, 1e-9), (key,)
            continue
        seen += 1
        assert _close(q["E_min"], e, 1e-9), (key, q["E_min"])
        assert _close(q["ratio_exact_over_hartree_exact"], rh, 1e-9), (key, q["ratio_exact_over_hartree_exact"])
        assert _close(q["ratio_exact_over_keff"], rk, 1e-9), (key, q["ratio_exact_over_keff"])
        assert _close(q["residual_ctrl_normalized"], R, 1e-9), (key, q["residual_ctrl_normalized"])
        assert _close(q["residual_keff_ctrl_normalized"], Rk, 1e-9), (key, q["residual_keff_ctrl_normalized"])
        assert _close(q["anchor_max_abs_rel_dev_vs_ed"], anc, 1e-9), (key, q["anchor_max_abs_rel_dev_vs_ed"])
        assert _close(q["refcons_over_E_min"], refc, 1e-4), (key, q["refcons_over_E_min"])
        # MEASURED, and the reason the 5 % gate certifies nothing here: EVERY lambda >= 2
        # row fails it, including the two the ED anchor likes best
        assert not q["refcons_ok"], key
        assert q["refcons_over_E_min"] > 0.05, key
    if seen == 0:
        pytest.skip("no pinned plane2 setting has landed")
    # the lambda = 0 controls the residuals are normalized by
    for key, ctrl in {(8, 32, 0.5, 0.0): 0.9999311124645787, (8, 32, 1.0, 0.0): 1.0820863588765437, (8, 64, 0.5, 0.0): 0.9992575970868628, (8, 64, 1.0, 0.0): 1.0171624610509882, (10, 32, 0.5, 0.0): 1.012421712857769, (10, 32, 1.0, 0.0): 1.1554106008485379}.items():
        if key in by:
            assert _close(by[key]["control_lam0"], ctrl, 1e-9), (key, by[key]["control_lam0"])


def test_item4_is_bracketed_and_not_resolved_at_lambda_2_and_4():
    """The verdict itself, and the logic that produced it, recomputed from the rows.

    If either `resolved` ever flips to True, item 4 has CLOSED and this test must be
    rewritten with the MANIFEST -- it is not to be loosened to make it pass."""
    rec = _load(_P2)
    band = rec["band_lam0_sweep"]
    assert _close(band, 0.0019228899680709954, 1e-12)
    assert _close(rec["anchor_gate"], 0.02, 1e-12) and _close(rec["resolve_bands"], 3.0, 1e-12)
    got = 0
    for key, weight_pair_spread in (("lam=2", 0.06211915497999976), ("lam=4", 0.4587928703030898)):
        v = rec["verdict"].get(key)
        if v is None:
            continue
        got += 1
        lam = float(key.split("=")[1])
        rows = [q for q in rec["rows"] if q["lam"] == lam and q["residual_ctrl_normalized"] is not None]
        allh = [q["residual_ctrl_normalized"] for q in rows]
        allk = [q["residual_keff_ctrl_normalized"] for q in rows
                if q["residual_keff_ctrl_normalized"] is not None]
        # the stored bracket IS min/max over the record's own rows (logic, not a literal)
        assert _close(v["bracket_hartree"][0], min(allh), 1e-12)
        assert _close(v["bracket_hartree"][1], max(allh), 1e-12)
        assert _close(v["bracket_keff"][0], min(allk), 1e-12)
        assert _close(v["bracket_keff"][1], max(allk), 1e-12)
        assert v["n_settings"] == len(rows)
        # the systematic is a MAX over axis pairs, so it can never fall below the weight
        # pair that is already landed and immutable
        assert v["systematic"] >= weight_pair_spread - 1e-12, (key, v["systematic"])
        assert _close(v["systematic_in_bands"], v["systematic"] / band, 1e-12)
        # and the verdict follows its own stated criterion
        assert v["resolved"] == bool(v["systematic"] <= 3.0 * band and v["n_anchor_qualified"] >= 2)
        assert v["resolved"] is False, (key, "item 4 has closed -- update the MANIFEST")
        assert v["systematic_in_bands"] > 10.0, (key, v["systematic_in_bands"])
    if got == 0:
        pytest.skip("no lambda > 0 plane2 verdict is on this tree")
    # the headline brackets as the MANIFEST quotes them, while these settings are the ones
    # on the tree (each endpoint is an archived per-setting value pinned above)
    v2, v4 = _p2_frozen(rec, "lam=2"), _p2_frozen(rec, "lam=4")
    if v2 is not None:
        # 5, not 4: fix round 1 added plane2_nmax10_chiop32_w0.5_lam2 to the frozen tag
        # list (it was on disk at the freeze commit and its own lambda = 0 control was
        # already frozen).  Its residual -1.056 % sits inside the bracket, so no endpoint
        # and no systematic moved; what it added is the lambda = 2 n_max axis.
        assert v2["n_settings"] == 5, v2["n_settings"]
        assert v2["resolved"] is False
        assert _close(v2["bracket_hartree"][0], -0.017967285832743096, 1e-9)
        assert _close(v2["bracket_hartree"][1], 0.04415186914725666, 1e-9)
        assert _close(v2["bracket_keff"][0], 0.0030082285502357475, 1e-9)
        assert _close(v2["bracket_keff"][1], 0.06384740012912304, 1e-9)
        assert _close(v2["systematic"], 0.06211915497999976, 1e-9)
        assert _close(v2["systematic_in_bands"], 32.305101181799, 1e-9)
    if v4 is not None:
        assert v4["n_settings"] == 5, v4["n_settings"]
        assert v4["resolved"] is False
        assert _close(v4["bracket_hartree"][0], -0.3434892628563455, 1e-9)
        assert _close(v4["bracket_hartree"][1], 0.15130990387850818, 1e-9)
        assert _close(v4["bracket_keff"][0], -0.2854112093570549, 1e-9)
        assert _close(v4["bracket_keff"][1], 0.18799815302913248, 1e-9)
        assert _close(v4["systematic"], 0.4587928703030898, 1e-9)
        assert _close(v4["systematic_in_bands"], 238.59548800046088, 1e-9)
        # the lambda = 4 bracket spans BOTH SIGNS -- the sharpest statement of why no
        # magnitude is quotable there
        assert v4["bracket_hartree"][0] < 0.0 < v4["bracket_hartree"][1]


def test_the_weight_axis_carries_the_item4_systematic_not_chi_op_and_not_n_max():
    """Which knob the lambda >= 2 systematic is made of, at both lambdas."""
    rec = _load(_P2)
    want = {"lam=2": {"weight_0.5_vs_1": 0.06211915497999976, "chi_op_32_vs_64": 0.011978426867252145,
                      "n_max_8_vs_10": 0.054708187198782454, "same_spec": 0.01941279605554776},
            "lam=4": {"weight_0.5_vs_1": 0.4587928703030898, "chi_op_32_vs_64": 0.3261762477861596,
                      "n_max_8_vs_10": 0.00727643466166672, "same_spec": 0.00125727572183062}}
    got = 0
    for key, w in want.items():
        v = _p2_frozen(rec, key)
        if v is None:
            continue
        got += 1
        for axis, val in w.items():
            if axis == "same_spec":
                if v["same_spec_spread_in_residual"] is None:
                    continue
                assert _close(v["same_spec_spread_in_residual"], val, 1e-9), (key, axis)
                continue
            sp = v["spreads"].get(axis)
            if sp is None:
                continue
            assert _close(sp["spread"], val, 1e-9), (key, axis, sp["spread"])
        # the ranking the MANIFEST states.  The weight axis is the largest at BOTH
        # lambdas; the rest of the ranking is lambda-dependent and is asserted per lambda,
        # never as "n_max is smallest" in general -- at lambda = 2 n_max is SECOND.
        sw, sc = v["spreads"].get("weight_0.5_vs_1"), v["spreads"].get("chi_op_32_vs_64")
        assert sw is not None and sc is not None, (key, "an axis pair vanished from the record")
        assert sw["spread"] > sc["spread"], key
        # NOT "if sn": at both lambdas the frozen snapshot HAS the n_max pair, and the
        # earlier guard let lambda = 2 pass vacuously while the record carried None.
        sn = v["spreads"].get("n_max_8_vs_10")
        assert sn is not None, (key, "the n_max 8 -> 10 pair is missing from the frozen "
                                     "snapshot; it is what the lambda = 2 ranking rests on")
        assert sw["spread"] > sn["spread"], key
        if key == "lam=2":
            # MEASURED, and the correction fix round 1 made: at lambda = 2 the n_max axis is
            # the SECOND largest, 28.5 bands, not the smallest
            assert sn["spread"] > sc["spread"], (key, sn["spread"], sc["spread"])
            assert sn["spread"] / rec["band_lam0_sweep"] > 25.0, sn["spread"]
        if key == "lam=4":
            # and at lambda = 4 it IS the smallest -- still 3.8 bands, above the criterion
            assert sn["spread"] < sc["spread"], (key, sn["spread"], sc["spread"])
            assert 3.0 < sn["spread"] / rec["band_lam0_sweep"] < 4.0, sn["spread"]
    if got == 0:
        pytest.skip("no plane2 verdict is on this tree")
    # THE BAND COUNTS THE MANIFEST'S "what would close it" PARAGRAPH QUOTES.  That paragraph
    # said, until fix round 2, that "the only n_max spread the frozen snapshot measures" was
    # the lambda = 4 one; the frozen snapshot measures the axis at BOTH lambdas and both are
    # above the 3-band criterion, which is what makes the closure statement honest.
    band = rec["band_lam0_sweep"]
    want_bands = {"lam=2": 28.451023255201964, "lam=4": 3.784113902765998}
    for key, nb in want_bands.items():
        v = _p2_frozen(rec, key)
        if v is None:
            continue
        sn = v["spreads"]["n_max_8_vs_10"]
        assert _close(sn["spread"] / band, nb, 1e-9), (key, sn["spread"] / band)
        assert sn["spread"] / band > 3.0, (key, "below the 3-band criterion?")
    # the live tree's same axis (a different weight column) is larger at both lambdas, so
    # nothing here depends on reading the live record as if it were the frozen one
    # 2026-09-13: READ THE PAIR THE VERDICT CHOSE.  Until the pool finished, the live
    # n_max pair happened to sit at chi_op 32, weight 1, and the tags were hard-coded here.
    # With all 36 rows landed the verdict picks chi_op 64 at lambda = 4, so a hard-coded tag
    # tested a pair the record was not using.  The pair is taken from the record itself now;
    # the identity being checked (the spread IS the two rows' residual difference) is the same.
    for key, lam in (("lam=2", 2.0), ("lam=4", 4.0)):
        by = _p2_rows(rec)
        lv = rec["verdict"].get(key, {}).get("spreads", {}).get("n_max_8_vs_10")
        if lv is None or not lv.get("pair"):
            continue
        (n_lo, c_lo, w_lo), (n_hi, c_hi, w_hi) = lv["pair"]
        lo, hi = by.get((n_lo, c_lo, w_lo, lam)), by.get((n_hi, c_hi, w_hi, lam))
        if lo is None or hi is None:
            continue
        assert _close(lv["spread"], abs(lo["residual_ctrl_normalized"] - hi["residual_ctrl_normalized"]), 1e-12)
        assert lv["spread"] > _p2_frozen(rec, key)["spreads"]["n_max_8_vs_10"]["spread"], key
    v2 = _p2_frozen(rec, "lam=2")
    if v2 is not None and v2["same_spec_spread_in_residual"] is not None:
        # MEASURED, and the reason the chi_op axis is not the limiter at lambda = 2: the
        # route's own same-spec floor at the identical spec (1.941 pp, 10.1 bands) is
        # LARGER than the whole chi_op 32 -> 64 move (1.198 pp)
        assert v2["same_spec_spread_in_residual"] > v2["spreads"]["chi_op_32_vs_64"]["spread"]
        assert v2["same_spec_spread_in_residual"] / rec["band_lam0_sweep"] > 10.0


def test_the_ed_anchor_gate_clears_one_setting_per_lambda_and_they_are_different_settings():
    """The only accuracy evidence at lambda >= 2, and what it does and does not select.

    Each entry is max |rel_dev_vs_ed| over every anchored point at that
    (chi_op, weight, lambda, SVD mode) at L <= 6 -- archived job numbers, so immutable
    unless a new anchored point lands, and a max can only grow."""
    rec = _load(_P2)
    tab = rec["anchor_table"]
    want = {"chi_op32_w0.5_lam2_sketched": (0.02330351842109992, 3),
            "chi_op32_w1_lam2_sketched": (0.18214065906271934, 3),
            "chi_op64_w0.5_lam2_sketched": (0.015514328088979068, 3),
            "chi_op64_w1_lam2_sketched": (0.021384171291304554, 3),
            "chi_op32_w0.5_lam4_sketched": (0.39424546741441363, 7),
            "chi_op32_w1_lam4_sketched": (0.26424333785866316, 3),
            "chi_op64_w0.5_lam4_sketched": (0.3448863999185752, 5),
            "chi_op64_w1_lam4_sketched": (0.016715069210005765, 4)}
    got = 0
    for k, (mx, n) in want.items():
        if k not in tab:
            continue
        got += 1
        assert _close(tab[k]["max_abs_rel_dev_vs_ed"], mx, 1e-9), (k, tab[k]["max_abs_rel_dev_vs_ed"])
        assert tab[k]["n_points"] >= n, (k, tab[k]["n_points"])
        assert _close(max(abs(p["rel_dev_vs_ed"]) for p in tab[k]["points"]), mx, 1e-12), k
    if got == 0:
        pytest.skip("no plane2 anchor table is on this tree")
    # the weighting finding is a THREE-point statement at lambda = 4, not a one-point one:
    # (chi_op 64, weight 1) is inside 2 % at L 5 n_max 6, L 5 n_max 8 and L 6 n_max 5
    k = "chi_op64_w1_lam4_sketched"
    if k in tab:
        pts = {(p["L"], p["n_max"]): p for p in tab[k]["points"]}
        assert {(5, 6), (5, 8), (6, 5)} <= set(pts), sorted(pts)
        assert max(abs(p["rel_dev_vs_ed"]) for p in tab[k]["points"]) < 0.02
        for other in ("chi_op64_w0.5_lam4_sketched", "chi_op32_w0.5_lam4_sketched",
                      "chi_op32_w1_lam4_sketched"):
            if other in tab:
                assert tab[other]["max_abs_rel_dev_vs_ed"] > 0.1, other
    # and it selects ONE setting per lambda, and a different one at each
    qual = {}
    for key in ("lam=2", "lam=4"):
        v = _p2_frozen(rec, key)
        if v is None:
            continue
        assert v["n_anchor_qualified"] == len(v["anchor_qualified"])
        qual[key] = {(q["chi_op"], q["weight_decay"]) for q in v["anchor_qualified"]}
    if len(qual) == 2 and all(len(z) == 1 for z in qual.values()):
        assert qual["lam=2"] != qual["lam=4"], qual
        assert qual["lam=2"] == {(64, 0.5)} and qual["lam=4"] == {(64, 1.0)}, qual
        v2, v4 = _p2_frozen(rec, "lam=2"), _p2_frozen(rec, "lam=4")
        assert _close(v2["anchor_qualified"][0]["residual_ctrl_normalized"], 0.032173442280004516, 1e-9)
        assert _close(v2["anchor_qualified"][0]["residual_keff_ctrl_normalized"], 0.0521157922594323, 1e-9)
        assert _close(v4["anchor_qualified"][0]["residual_ctrl_normalized"], -0.0173130150701859, 1e-9)
        assert _close(v4["anchor_qualified"][0]["residual_keff_ctrl_normalized"], 0.026664678945139242, 1e-9)


def test_the_item4_verdict_does_not_depend_on_the_anchor_gates_value():
    """The gate is the one knob of the verdict that could have been tuned to an answer.

    The (chi_op 64, weight 1) column at lambda = 2 misses the 2 % gate by 0.14 pp, so the
    sensitivity is recorded and asserted: at NO gate from 1 % to 5 % does either lambda
    resolve."""
    rec = _load(_P2)
    got = 0
    for key in ("lam=2", "lam=4"):
        v = rec["verdict"].get(key)
        if v is None or "anchor_gate_sensitivity" not in v:
            continue
        got += 1
        rows = [q for q in rec["rows"] if q["lam"] == float(key.split("=")[1])
                and q["residual_ctrl_normalized"] is not None]
        for g in v["anchor_gate_sensitivity"]:
            # recomputed, not a literal
            vals = [q["residual_ctrl_normalized"] for q in rows
                    if q["anchor_max_abs_rel_dev_vs_ed"] is not None
                    and q["anchor_max_abs_rel_dev_vs_ed"] <= g["gate"]]
            assert g["n_qualified"] == len(vals), (key, g["gate"])
            assert g["resolved_at_this_gate"] is False, (key, g["gate"])
    if got == 0:
        pytest.skip("no anchor-gate sensitivity is on this tree")
    v2 = _p2_frozen(rec, "lam=2")
    if v2 is not None:
        counts = {g["gate"]: g["n_qualified"] for g in v2["anchor_gate_sensitivity"]}
        assert counts == {0.01: 0, 0.02: 1, 0.025: 4, 0.03: 4, 0.05: 4}, counts
        wide = next(g for g in v2["anchor_gate_sensitivity"] if g["gate"] == 0.025)
        # admitting the near-miss column does NOT close it: four settings, 28.5 bands
        # (four, not three, since fix round 1 added the (n_max 10, chi_op 32, w 0.5) row,
        # whose 2.330 % anchor also clears a 2.5 % gate and whose -1.056 % is the new low)
        assert _close(wide["bracket_hartree"][0], -0.010556318051525793, 1e-9)
        assert _close(wide["bracket_hartree"][1], 0.04415186914725666, 1e-9)
        assert _close(wide["spread_in_bands"], 28.451023255201964, 1e-9), wide["spread_in_bands"]
        assert wide["spread_in_bands"] > 3.0
    v4 = _p2_frozen(rec, "lam=4")
    if v4 is not None:
        counts = {g["gate"]: g["n_qualified"] for g in v4["anchor_gate_sensitivity"]}
        assert counts == {0.01: 0, 0.02: 1, 0.025: 1, 0.03: 1, 0.05: 1}, counts


def test_the_lambda_ge_2_sign_survives_more_of_the_plane_than_its_magnitude():
    """Against K_eff the residual is positive nearly everywhere; against E_H it is not.

    This is the weaker claim item 4 CAN support, and it is recorded separately from the
    magnitude so the two are never conflated."""
    rec = _load(_P2)
    got = 0
    for key, (posh, posk, n) in (("lam=2", (2, 5, 5)), ("lam=4", (3, 4, 5))):
        v = rec["verdict"].get(key)
        if v is None or "n_positive_keff" not in v:
            continue
        got += 1
        lam = float(key.split("=")[1])
        rows = [q for q in rec["rows"] if q["lam"] == lam and q["residual_ctrl_normalized"] is not None]
        assert v["n_positive_hartree"] == sum(1 for q in rows if q["residual_ctrl_normalized"] > 0)
        assert v["n_positive_keff"] == sum(1 for q in rows
                                           if q["residual_keff_ctrl_normalized"] is not None
                                           and q["residual_keff_ctrl_normalized"] > 0)
        f = _p2_frozen(rec, key)
        if f is None or f["n_settings"] != n:
            continue
        assert (f["n_positive_hartree"], f["n_positive_keff"]) == (posh, posk), (key, v["n_positive_hartree"], v["n_positive_keff"])
        # the K_eff sign is stronger than the E_H sign, at both lambdas
        assert v["n_positive_keff"] >= v["n_positive_hartree"], key
    if got == 0:
        pytest.skip("no plane2 sign record is on this tree")
    v4 = _p2_frozen(rec, "lam=4")
    if v4 is not None and v4["n_settings"] == 5:
        # THE SUPERLATIVE IS SCOPED TO THE FROZEN SNAPSHOT.  The lambda = 4 K_eff-negative
        # setting is the worst-diagnosed row OF THE 15 FROZEN TAGS -- it is NOT the worst
        # row on the tree.  `refcons_over_E_min > 2.5` alone cannot tell the two apart,
        # because the live worst (plane2_nmax10_chiop32_w1_lam2, 370.2 %, itself the extra
        # lambda = 2 sign exception) also clears 2.5; so both halves are pinned here, and
        # the MANIFEST sentence that once said "the tree's worst" now says "the frozen
        # snapshot's worst ... not the tree's worst, which is 370.2 %".
        assert len(v4["negative_keff_settings"]) == 1
        bad = v4["negative_keff_settings"][0]
        assert (bad["chi_op"], bad["weight_decay"]) == (32, 1.0)
        assert bad["refcons_over_E_min"] > 2.5 and bad["anchor_max_abs_rel_dev_vs_ed"] > 0.25
        assert _close(bad["refcons_over_E_min"], 2.5728792918907546, 1e-9)
        frozen_tags = set((rec.get("unit2_snapshot") or {}).get("tags") or [])
        assert len(frozen_tags) == 15, len(frozen_tags)
        fz = [q["refcons_over_E_min"] for q in rec["rows"]
              if q["tag"] in frozen_tags and q.get("refcons_over_E_min") is not None]
        assert len(fz) == 15 and _close(max(fz), bad["refcons_over_E_min"], 1e-9), max(fz)
        # ... and the tree carries a strictly worse-diagnosed row than the frozen worst,
        # which is exactly why the superlative may not be written tree-wide.
        allr = [(q["refcons_over_E_min"], q["tag"]) for q in rec["rows"]
                if q.get("refcons_over_E_min") is not None]
        worst_live, worst_tag = max(allr)
        assert worst_live > max(fz), (worst_live, max(fz))
        assert worst_tag not in frozen_tags, worst_tag
        if worst_tag == "plane2_nmax10_chiop32_w1_lam2":
            assert _close(worst_live, 3.7019471751341095, 1e-9), worst_live
    # THE SCOPE IS PART OF THE CLAIM.  "positive at every landed setting at lambda = 2" is
    # FALSE on the live tree and the docs may not say it unscoped: the archived row
    # (n_max 10, chi_op 32, weight 1, lambda 2) reads R_keff -19.637 %.  Whenever that row
    # is on the tree the live lambda = 2 K_eff sign count must be strictly below the number
    # of K_eff settings, which is what forces the docs' "5 of 6 live" wording.
    by = _p2_rows(rec)
    neg2 = by.get((10, 32, 1.0, 2.0))
    if neg2 is not None and neg2["residual_keff_ctrl_normalized"] is not None:
        assert _close(neg2["residual_keff_ctrl_normalized"], -0.19637378380409154, 1e-9)
        lv = rec["verdict"]["lam=2"]
        assert lv["n_positive_keff"] < lv["n_keff"], (lv["n_positive_keff"], lv["n_keff"])
        assert any((z["n_max"], z["chi_op"], z["weight_decay"]) == (10, 32, 1.0)
                   for z in lv["negative_keff_settings"])


def test_the_item4_bracket_lies_below_first_order_and_its_c2_misses_the_small_lambda_one():
    """Confrontation with the exact O(lambda) slope and with the small-lambda curvature.

    One statement is one-sided and stable as the plane fills: every landed setting lies
    BELOW the first-order prediction at lambda 2 and 4.  The other is NOT: "the implied
    second-order coefficient overlaps no lambda <= 1 value" held on the frozen unit-2
    snapshot and was falsified within the hour by (n_max 10, chi_op 32, weight 1) at
    lambda = 2 (R_ctrl -22.154 %, its own lambda = 0 control 15.5 % off 1, refcons 370 % of
    |E_min|, ED anchor 18.2 % -- it fails every gate and still widens the bracket).  So the
    no-overlap claim is asserted on the FROZEN verdict only, which is where the MANIFEST
    quotes it, and the MANIFEST records the falsification beside it."""
    rec = _load(_P2)
    assert _close(rec["pt_slope_dR_dlambda_at_0"], 0.0809088379025304, 1e-12)
    small = rec["implied_second_order_coefficient_small_lambda"]
    want_small = {"lam=0.1": -0.06793085959164896, "lam=0.25": -0.10609516448081435,
                  "lam=0.5": -0.10281188877061964, "lam=1": -0.05532495395511877}
    for k, v in want_small.items():
        if k in small:
            assert _close(small[k], v, 1e-9), (k, small[k])
    # "about -0.10" is the lambda = 0.25 and 0.5 pair ALONE: the four lambda <= 1 values
    # span a factor 1.9, so the small-lambda curvature is itself not a single number
    have = [small[k] for k in want_small if k in small]
    if len(have) == 4:
        assert _close(max(abs(z) for z in have) / min(abs(z) for z in have), 1.917672892541074, 1e-12)
    got = 0
    for key, pred in (("lam=2", 0.1618176758050608), ("lam=4", 0.3236353516101216)):
        v = rec["verdict"].get(key)
        if v is None or v.get("implied_second_order_coefficient") is None:
            continue
        got += 1
        lam = float(key.split("=")[1])
        assert _close(v["predicted_residual_O1"], pred, 1e-12), key
        assert _close(v["predicted_residual_O1"], rec["pt_slope_dR_dlambda_at_0"] * lam, 1e-12)
        # EVERY landed setting is below the first-order prediction, at both lambdas
        assert v["bracket_hartree"][1] < pred, (key, v["bracket_hartree"][1], pred)
        c2 = v["implied_second_order_coefficient"]["bracket"]
        assert _close(c2[0], (v["bracket_hartree"][0] - rec["pt_slope_dR_dlambda_at_0"] * lam) / lam ** 2, 1e-12)
        assert _close(c2[1], (v["bracket_hartree"][1] - rec["pt_slope_dR_dlambda_at_0"] * lam) / lam ** 2, 1e-12)
        # NOT asserted on the live bracket.  The no-overlap statement held on the frozen
        # snapshot and was FALSIFIED by a row that landed minutes later: (n_max 10, chi_op 32,
        # weight 1) at lambda = 2 reads R_ctrl -22.154 %, which drags the live c2 bracket to
        # -0.09584 and swallows the lambda = 0.1 and lambda = 1 values (-0.0679, -0.0553).
        # (FIX ROUND 1: the first writing of this comment said "the lambda = 0.25 / 0.5
        # values".  Those are -0.1061 and -0.1028, BELOW -0.09584, hence still outside; the
        # falsification stands, the identification was wrong.  Pinned just below.)  It is
        # asserted on the frozen verdict below, where it is what the MANIFEST quotes, and the
        # MANIFEST records the falsification.  See test docstring.
    if got == 0:
        pytest.skip("no plane2 PT confrontation is on this tree")
    v2, v4 = _p2_frozen(rec, "lam=2"), _p2_frozen(rec, "lam=4")
    for v_ in (v2, v4):
        if v_ is not None and len(have) == 4:
            c2_ = v_["implied_second_order_coefficient"]["bracket"]
            assert not any(c2_[0] <= z <= c2_[1] for z in have), (c2_, have)
    # and the live brackets, whatever they now are, still lie below the first-order prediction
    for key_, pred_ in (("lam=2", 0.1618176758050608), ("lam=4", 0.3236353516101216)):
        lv = rec["verdict"].get(key_)
        if lv is not None:
            assert lv["bracket_hartree"][1] < pred_, (key_, lv["bracket_hartree"][1])
    if v2 is not None:
        assert [round(z, 10) for z in v2["implied_second_order_coefficient"]["bracket"]] == \
            [-0.0449462404, -0.0294164517], v2["implied_second_order_coefficient"]["bracket"]
    if v4 is not None:
        assert [round(z, 10) for z in v4["implied_second_order_coefficient"]["bracket"]] == \
            [-0.0416952884, -0.0107703405], v4["implied_second_order_coefficient"]["bracket"]
    # WHICH small-lambda c2 values the falsifying row actually swallows.  This is built from
    # ONE archived residual, so it is immutable: c2 = (R - slope*lam)/lam**2 at lam = 2 for
    # the (n_max 10, chi_op 32, weight 1) row.  -0.0679 (lam 0.1) and -0.0553 (lam 1) are
    # inside [c2_row, -0.0294164517]; -0.1061 (lam 0.25) and -0.1028 (lam 0.5) are below it
    # and are NOT.  The MANIFEST and this file both said 0.25/0.5 before fix round 1.
    by = _p2_rows(rec)
    q = by.get((10, 32, 1.0, 2.0))
    if q is not None and q["residual_ctrl_normalized"] is not None and len(have) == 4:
        assert _close(q["residual_ctrl_normalized"], -0.22154405724888693, 1e-9)
        c2_row = (q["residual_ctrl_normalized"] - rec["pt_slope_dR_dlambda_at_0"] * 2.0) / 4.0
        assert _close(c2_row, -0.09584043326348693, 1e-9), c2_row
        hi = -0.029416451664451032
        inside = {k for k, z in want_small.items() if c2_row <= z <= hi}
        assert inside == {"lam=0.1", "lam=1"}, inside
        assert small["lam=0.25"] < c2_row and small["lam=0.5"] < c2_row
    # the lambda = 0.1 point constrains the SLOPE and cannot pin c2: its deficit is 0.35
    # bands, inside the noise
    l01 = _load("s8_lam0.1.json")
    assert _close(l01["residual_ctrl_normalized"], 0.00741157519433655, 1e-9)
    assert _close(l01["predicted_residual_O1"], 0.00809088379025304, 1e-9)
    assert abs(l01["bands_from_prediction"]) < 1.0, l01["bands_from_prediction"]


def test_what_closing_item4_would_cost_is_a_measured_wall_not_a_model_literal():
    """The closure estimate, at the plane2 calibration measured on this tree."""
    rec = _load(_P2)
    # the FROZEN snapshot's closure cost: its calibration is the median over the snapshot's
    # own jobs, so the hours do not drift as the pool lands rows
    cc = (rec.get("unit2_snapshot") or {}).get("closure_cost")
    if not cc:
        pytest.skip("no frozen plane2 closure cost is on this tree")
    for k, v in cc.items():
        assert v["n_jobs_unlanded"] == len(v["tags"]), k
        assert _close(v["calibrated_h"], v["nominal_h"] * v["plane2_calibration_measured"], 1e-12), k
        # the column costed out is one the ED anchor actually clears
        assert k in ("chi_op64_w0.5", "chi_op64_w1"), k
    if set(cc) == {"chi_op64_w0.5", "chi_op64_w1"}:
        # the calibration is a MEDIAN over the frozen snapshot's own 2-thread plane2 jobs,
        # so adding the 15th tag in fix round 1 moved it: 1.6767x (n = 14) -> 1.5038x
        # (n = 15), and 46.841 h -> 42.010 h.  That sensitivity is itself the point: a wall
        # calibration over 15 jobs is determined to about 10 %.
        assert _close(cc["chi_op64_w1"]["plane2_calibration_measured"], 1.503790623082055, 1e-9)
        assert cc["chi_op64_w1"]["n_calibration_jobs"] == 15
        for k in cc:
            assert _close(cc[k]["nominal_h"], 13.968188002204492, 1e-9), k
            assert _close(cc[k]["calibrated_h"], 21.00523013916238, 1e-9), k
        tot_n = sum(v["n_jobs_unlanded"] for v in cc.values())
        assert _close(sum(v["nominal_h"] for v in cc.values()), 27.936376004408984, 1e-9)
        assert _close(sum(v["calibrated_h"] for v in cc.values()), 42.01046027832476, 1e-9)
        assert tot_n == 12, tot_n
    # and it is stated as a TEST of resolution, not a delivery of it: EVERY n_max spread the
    # frozen snapshot measures -- at BOTH lambdas, not just lambda = 4 -- is already above the
    # 3-band criterion (28.451 bands at lambda = 2, 3.784 at lambda = 4)
    for key in ("lam=2", "lam=4"):
        v = _p2_frozen(rec, key)
        if v is not None and v["spreads"].get("n_max_8_vs_10"):
            assert v["spreads"]["n_max_8_vs_10"]["spread"] / rec["band_lam0_sweep"] > 3.0, key
