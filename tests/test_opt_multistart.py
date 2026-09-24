"""Layer-6 global search (vacuum.opt.multistart): starts, basins, the raised gate,
the equal-work round trip, and the nonperturbative communication decomposition.

Anchors, each a permanent regression test:

1. NONPERTURBATIVE SPLIT = TMM21 SPLIT AT LEADING ORDER.  On the Gate-A
   lattice (N = 20, m = 0.6, gap 2, cos^2 half-width 2, sites 7/11) the
   field-state / detector-zero-point decomposition of <a_A a_B>
   (``nonperturbative_split``) converges to the perturbative engine's
   |M_comm|/|M| as lambda^2: |f_np - f_pert| measured 2.6e-4, 2.9e-5,
   2.3e-6 at lam_osc = 0.3, 0.1, 0.03 (asserted < 1e-3, < 1e-4, < 1e-5),
   additivity M_full = M_field + M_signal to 1e-10, |M_field| -> |M_vac|
   and |M_signal| -> |M_comm| to 1 % at lam_osc = 0.03.
2. SPACELIKE ZERO.  In a spacelike layout (sites 4/16, separation 12 >
   window 4) the signal part vanishes to lattice leakage: f_np < 1e-8.
3. OBJECTIVE ROUTE + CHANNELS.  ``split_from_objective`` reproduces the
   direct call on the same theta (1e-12) and stays exactly additive with
   loss/dephasing channels on (the channel-injected term is booked).
4. MULTISTART MECHANICS.  Starts are the declared mix (published, extra,
   CEM best/mean, random at the cycled scales); sign-flipped waveforms
   fall in one basin; every round-tripped row carries the gate level and a
   movement below the declared tolerance, equal work to the budget within
   the work tolerance, the check dict and the twin diagnostics; the
   sibling composite carries the sibling's proxy and the main quantity.
5. SPEC ROUTE.  ``multistart_from_spec`` on plain data reproduces
   ``multistart`` on the built objective (same trials, same best).

Requires JAX (importorskip); ~90 s.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from vacuum.core import harmonic_chain_K  # noqa: E402
from vacuum.detectors import UDWDetector, imperfection_channel, pair_state_with_split, switching, wightman_lattice  # noqa: E402
from vacuum.opt.multistart import (  # noqa: E402
    cluster_basins,
    lam_max_index,
    make_starts,
    multistart,
    multistart_from_spec,
    nonperturbative_split,
    roundtrip_equal_work,
    split_from_objective,
    waveform_samples,
    with_signaling_sibling,
)
from vacuum.opt.objectives import make_objective  # noqa: E402
from vacuum.opt.optimize import constraints, make_value  # noqa: E402

N, MASS = 20, 0.6
K = harmonic_chain_K(N, MASS, bc="dirichlet")
GAP = 2.0
TW = 2.0
CHI = switching("cos2", TW, 0.0)
TS = np.array([-TW, TW])


def _pert_split(sites, lam):
    F = np.eye(N)
    st = pair_state_with_split(wightman_lattice(K), UDWDetector(CHI, F[sites[0]]), UDWDetector(CHI, F[sites[1]]),
                               lam / math.sqrt(2 * GAP), GAP, rtol=1e-10, max_doublings=14, comm_atol=1e-18)
    return st


def _np_split(sites, lam, splits=1024, channels=()):
    return nonperturbative_split(K, list(sites), lambda t: np.array([GAP, GAP]), lambda t: lam * CHI(t) * np.ones(2), TS, splits,
                                 channels=channels)


# ---- 1. leading-order agreement with the TMM21 split -------------------------

def test_nonperturbative_split_converges_to_tmm21():
    st = _pert_split((7, 11), 0.1)
    f_pert = st.comm.comm_fraction
    lu2 = (0.1 / math.sqrt(2 * GAP)) ** 2
    devs = []
    for lam, tol in ((0.3, 1e-3), (0.1, 1e-4), (0.03, 1e-5)):
        d = _np_split((7, 11), lam)
        assert d["additivity"] < 1e-10
        dev = abs(d["f_np"] - f_pert)
        devs.append(dev)
        assert dev < tol, f"lam {lam}: f_np {d['f_np']:.6f} vs perturbative {f_pert:.6f} (|d| = {dev:.2e})"
        if lam == 0.03:
            s = (lam / math.sqrt(2 * GAP)) ** 2 / lu2
            assert abs(abs(d["M_field"]) / s / abs(st.comm.M_vac) - 1.0) < 1e-2
            assert abs(abs(d["M_signal"]) / s / abs(st.comm.M_comm) - 1.0) < 1e-2
    assert devs[0] > devs[1] > devs[2]  # lambda^2 scaling
    print(f"\nf_np - f_pert at lam 0.3/0.1/0.03: {devs[0]:.2e} / {devs[1]:.2e} / {devs[2]:.2e} (f_pert {f_pert:.5f})")


# ---- 2. spacelike zero ----------------------------------------------------------

def test_spacelike_signal_part_vanishes():
    d = _np_split((4, 16), 0.2)
    assert abs(d["M_full"]) > 0.0
    assert d["f_np"] < 1e-8, f"spacelike layout: f_np = {d['f_np']:.2e}"
    assert d["additivity"] < 1e-8


# ---- 3. objective route, channels -----------------------------------------------

def test_split_from_objective_matches_direct_and_is_additive_with_channels():
    obj = make_objective("negativity", K=K, detectors=[{"site": 7}, {"site": 11}], gaps=(GAP, GAP), window=(0.0, TW), basis="fourier",
                         n_coeff=3, lam_max=0.2, free=("lam", "lam_max"), splits=64, quantity="surrogate")
    d1 = split_from_objective(obj, obj.theta0, K, splits=256)
    d2 = _np_split((7, 11), 0.2, splits=256)
    assert abs(d1["M_full"] - d2["M_full"]) < 1e-12 and abs(d1["f_np"] - d2["f_np"]) < 1e-12
    noise = dict(kappa=0.02, nbar=0.01, gamma_phi=0.01)
    objn = make_objective("negativity", K=K, detectors=[{"site": 7}, {"site": 11}], gaps=(GAP, GAP), window=(0.0, TW), basis="fourier",
                          n_coeff=3, lam_max=0.2, free=("lam", "lam_max"), splits=64, quantity="surrogate", T=0.05, noise=noise)
    dn = split_from_objective(objn, objn.theta0, K, splits=256)
    assert dn["n_channels"] == 1 and dn["additivity"] < 1e-10
    assert abs(dn["M_signal_channel"]) > 0.0 and abs(dn["M_signal"] - dn["M_signal_det"] - dn["M_signal_channel"]) < 1e-15
    print(f"\nnoiseless f_np {d1['f_np']:.5f}; with channels f_np {dn['f_np']:.5f} (channel term {abs(dn['M_signal_channel']) / abs(dn['M_full']):.2e})")


# ---- 4. multistart mechanics ----------------------------------------------------

@pytest.fixture(scope="module")
def small_objective():
    obj = make_objective("negativity", K=K, detectors=[{"site": 7}, {"site": 11}], gaps=(GAP, GAP), window=(0.0, TW), basis="fourier",
                         n_coeff=3, lam_max=0.2, free=("lam", "lam_max"), splits=128, quantity="surrogate", conv_tol=1e-8, max_halvings=12)
    base = obj.roundtrip(obj.theta0)
    _, aux = obj.quantity_aux(jnp.asarray(obj.theta0))
    return obj, base, float(aux["work"]) / base["work"]


def test_starts_and_basins(small_objective):
    obj, base, bias = small_objective
    C = constraints(energy_budget=base["work"] * bias, signaling_bound=base["comm_fraction"], weight=1e4, signaling_weight=1e3)
    extra = obj.theta0 + np.array([0.0, 0.01, 0.0, 0.0])
    starts, n_evals = make_starts(obj, 7, (3, 4), value_fn=make_value(obj, C), extra_starts=[("extra", extra)],
                                  n_cem=1, cem_pop=6, cem_iter=2, random_scales=(0.1, 0.3))
    names = [s[0] for s in starts]
    assert names[:2] == ["published", "extra"] and names[2:4] == ["cem3:best", "cem3:mean"]
    assert names[4:] == ["random0:s0.1", "random1:s0.3", "random2:s0.1"] and n_evals == 12
    assert np.array_equal(starts[0][1], obj.theta0) and all(s[1].shape == obj.theta0.shape for s in starts)
    assert all(s[1][lam_max_index(obj)] == obj.theta0[-1] for s in starts)  # lam_max is never perturbed
    assert all(s[1][0] == obj.theta0[0] for s in starts)  # nor the cos^2 weight a_0
    # explicit seeds: the same composition reproduces the same random draws
    starts2, _ = make_starts(obj, 7, (3, 4), value_fn=make_value(obj, C), extra_starts=[("extra", extra)],
                             n_cem=1, cem_pop=6, cem_iter=2, random_scales=(0.1, 0.3))
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(starts, starts2))
    # basins: sign flip and scale are one basin; a different shape is another
    w = waveform_samples(obj, obj.theta0)
    w_flip = waveform_samples(obj, obj.theta0 * np.array([-1.0, 1.0, 1.0, 1.0]))
    assert np.max(np.abs(w - w_flip)) == 0.0
    other = waveform_samples(obj, np.array([1.0, 0.8, -0.5, 0.2]))
    labels = cluster_basins([w, w_flip, 1.02 * w, other], [1.0, 0.9, 0.8, 2.0], tol=0.05)
    assert labels.tolist() == [1, 1, 1, 0]


def test_multistart_roundtrips_and_gate(small_objective):
    obj, base, bias = small_objective
    C = constraints(energy_budget=base["work"] * bias, signaling_bound=base["comm_fraction"] * 0.999, weight=1e4, signaling_weight=1e3)
    res = multistart(obj, C, 4, (0,), budget=base["work"], comm_bound=base["comm_fraction"], baseline_E_N=base["E_N"], n_cem=0,
                     maxiter=8, restarts=0, n_polish=1, polish_maxiter=6, polish_restarts=0, current_best=base["E_N"],
                     conv_tol=1e-8, max_halvings=12, max_roundtrips=2)
    d = res.as_dict()
    assert d["stats"]["n_trials"] == 5 and d["starts_run"] == 4 and d["n_evals"] > 0 and not d["time_limited"]
    assert d["stats"]["n_basins_all"] >= 1 and 1 <= d["stats"]["n_roundtrips"] <= 2
    for row in d["roundtrips"]:
        assert row["converged"] and row["movement"] < 1e-8 and row["halvings"] <= 12 and row["max_halvings"] == 12
        assert abs(row["work"] / base["work"] - 1.0) < 1e-4  # equal work finished on the gated ledger
        assert set(row["check"]) >= {"energy_budget", "signaling_bound", "audits", "admissible"}
        assert abs(row["E_N_jax"] - row["E_N"]) < 1e-3 * max(row["E_N"], 1e-6) + 1e-8  # twin vs gated numpy
        assert abs(row["comm_proxy"] - row["comm_fraction"]) < 1e-6  # JAX split vs numpy split
        assert row["V_AB"].shape == (4, 4) and "basin" in row and "gain" in row
    assert d["gate"]["max_level_used"] <= 12 and d["gate"]["max_movement"] < 1e-8 and d["gate"]["all_converged"]
    assert isinstance(d["beats_current_best"], bool)
    if d["best"] is not None:
        assert d["best"]["admissible"] and d["best"]["E_N"] > base["E_N"]
    # direct round trip of the published protocol reproduces the baseline row
    row0 = roundtrip_equal_work(obj, obj.theta0, base["work"], comm_bound=base["comm_fraction"], conv_tol=1e-8, max_halvings=12,
                                baseline_E_N=base["E_N"])
    assert abs(row0["E_N"] - base["E_N"]) < 1e-12 and row0["gated_reprojections"] == 0 and row0["admissible"]


def test_signaling_sibling_composite():
    main = make_objective("negativity", K=K, detectors=[{"site": 7}, {"site": 11}], gaps=(GAP, GAP), window=(0.0, TW), basis="fourier",
                          n_coeff=3, lam_max=0.2, free=("lam", "lam_max"), splits=64, quantity="surrogate",
                          gap_basis="cos2", n_gap_coeff=1, gap_coeffs=[-0.05])
    sib = make_objective("negativity", K=K, detectors=[{"site": 7}, {"site": 11}], gaps=(GAP, GAP), window=(0.0, TW), basis="fourier",
                         n_coeff=3, lam_max=0.2, free=("lam", "lam_max"), splits=64, quantity="surrogate")
    assert main.cfg["signaling_proxy"] is False and sib.cfg["signaling_proxy"] is True
    comp = with_signaling_sibling(main, sib)
    q, aux = comp.quantity_aux(jnp.asarray(comp.theta0))
    qm, auxm = main.quantity_aux(jnp.asarray(main.theta0))
    _, auxs = sib.quantity_aux(jnp.asarray(sib.theta0))
    assert float(q) == float(qm) and float(aux["work"]) == float(auxm["work"])
    assert float(aux["comm_fraction"]) == float(auxs["comm_fraction"]) and comp.cfg["signaling_proxy"] == "sibling"
    g = jax.grad(lambda th: comp.quantity_aux(th)[1]["comm_fraction"])(jnp.asarray(comp.theta0))
    assert np.all(np.isfinite(np.asarray(g)))


# ---- 5. spec route -------------------------------------------------------------

def test_multistart_from_spec_matches_direct(small_objective):
    obj, base, bias = small_objective
    cfg = dict(N=N, mass=MASS, bc="dirichlet", detectors=[{"site": 7}, {"site": 11}], gaps=(GAP, GAP), window=(0.0, TW), basis="fourier",
               n_coeff=3, lam_max=0.2, free=("lam", "lam_max"), splits=128, quantity="surrogate", conv_tol=1e-8, max_halvings=12)
    ckw = dict(energy_budget=base["work"] * bias, signaling_bound=base["comm_fraction"], weight=1e4, signaling_weight=1e3)
    opts = dict(budget=base["work"], comm_bound=base["comm_fraction"], baseline_E_N=base["E_N"], n_cem=0, maxiter=4, restarts=0,
                n_polish=0, polish_maxiter=0, conv_tol=1e-8, max_halvings=12, max_roundtrips=1)
    label, d = multistart_from_spec(dict(label="x", objective=cfg, constraints=ckw, n_starts=2, seeds=(5,),
                                         options=dict(opts, extra_starts={"e": (obj.theta0 + 0.02).tolist()})))
    res = multistart(obj, constraints(**ckw), 2, (5,), extra_starts=[("e", obj.theta0 + 0.02)], **opts)
    assert label == "x" and [t["start"] for t in d["trials"]] == [t["start"] for t in res.trials] == ["published", "e"]
    assert all(abs(a["E_N_jax"] - b["E_N_jax"]) < 1e-12 for a, b in zip(d["trials"], res.trials))
    assert abs(d["roundtrips"][0]["E_N"] - res.roundtrips[0]["E_N"]) < 1e-12
