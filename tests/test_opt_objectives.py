"""Layer-6 anchors for vacuum.opt.objectives + vacuum.opt.optimize.

Mandatory before any improvement claim (vacuum/opt/README.md):

1. PKMM RE-DISCOVERY.  The optimizer, from a generic start, re-finds the
   hand-tuned structure recorded in tests/test_udw.py on PKMM's own
   closed forms (PRD 92, 064042 (2015), Eqs. (29)-(31), (68); Fig. 2 panel
   b.1, Delta/T = 3, pointlike sigma/T = 0.02):
   (a) the interior gap optimum Omega T = alpha* at fixed separation
       d/T = 6 (a separation beyond the small-gap death line, where
       harvesting needs a gap window) — Nelder-Mead from alpha0 = 9 lands
       on the bracketed scalar optimum to 1e-4 (recovered alpha* = 4.7740);
   (b) the recorded death-line point (Omega T = 9.1, d/T = 9.803 digitized
       from the vector artwork) — minimizing N^(2)^2 over d/T from d/T = 5
       recovers it to the digitization tolerance (2%; measured 0.07%).
   A gradient method started on the exponentially flat large-gap plateau
   (alpha0 = 9, |dN2/dalpha| ~ 1e-21) stops where it stands — recorded,
   because that is exactly the case the derivative-free driver exists for.
2. LATTICE RE-DISCOVERY.  On the Gate-A lattice at lightlike separation,
   the JAX nonperturbative objective's gap optimum agrees with the
   perturbative PKMM-formula engine's (scipy bounded search on
   ``udw_pair_state``) to O(lambda^2) (measured 1.3e-3 at lambda = 0.1).
3. HOTTA.  ``qet_energy`` (dense 4x4 in JAX) equals the closed form Eq. (14)
   to 1e-14, and the optimizer recovers theta* = (1/2) atan2(hk, h^2+2k^2)
   [Eqs. (9)-(10)] to 1e-8 from a generic start (L-BFGS: 1e-13; Adam:
   1e-9); the round trip runs the coded protocol and the E_A >= E_B audit.
4. QEI.  ``qei_ratio`` (JAX assembly on the fixed grid/support of the numpy
   ``optimize_sampling``) agrees with the numpy ratio to 1e-10 (measured
   1e-12), its gradient with finite differences, and the ascent recovers
   the 2d test's interior optimum (tests/test_qei.py: tau0* ~ 5.5 interior,
   ratio in (0.98, 1)) — the same optimum ``optimize_sampling`` finds.
5. ROUND TRIP + SIGNALING PROXY.  The negativity objective's JAX quantity,
   work and second-order communication fraction agree with the numpy
   stack (gated E_N to the declared resolution, |M_comm|/|M| to 1e-8);
   the constraint penalties and checks behave.

Requires JAX (importorskip); ~70 s.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from scipy.optimize import brentq, minimize_scalar  # noqa: E402

from vacuum.core import harmonic_chain_K  # noqa: E402
from vacuum.detectors import UDWDetector, pkmm_negativity_estimator, switching, udw_pair_state, wightman_lattice  # noqa: E402
from vacuum.opt.objectives import make_objective  # noqa: E402
from vacuum.opt.optimize import constraints, optimize, project_energy_budget  # noqa: E402
from vacuum.qet import hotta  # noqa: E402


# ---- 1. PKMM re-discovery ------------------------------------------------------

def test_pkmm_interior_gap_optimum_rediscovered():
    obj = make_objective("pkmm_negativity", free=("alpha",), beta=6.0, gamma=3.0, delta=0.02)
    theta, hist = optimize(obj, None, np.array([9.0]), method="nelder-mead", bounds=obj.bounds, xatol=1e-6, fatol=1e-14)
    ref = minimize_scalar(lambda a: -pkmm_negativity_estimator(a, 6.0, 3.0, 0.02), bounds=(3.0, 12.0),
                          method="bounded", options={"xatol": 1e-8})
    assert abs(theta[0] - ref.x) < 1e-4, f"alpha* {theta[0]!r} vs bracketed {ref.x!r}"
    assert abs(theta[0] - 4.774) < 5e-3, f"recovered alpha* = {theta[0]:.4f} (recorded 4.7740)"
    assert obj.quantity(theta) > 0.0 and hist[-1]["grad_mode"] == "none"
    # the recorded death line at this d/T (digitized d*(4.6) = 6.009) sits below alpha*
    assert pkmm_negativity_estimator(4.6, 6.0, 3.0, 0.02) < obj.quantity(theta)
    # a gradient method from the flat plateau does not move (documented)
    th_l, h_l = optimize(obj, None, np.array([9.0]), method="lbfgs", bounds=obj.bounds, maxiter=50)
    assert h_l[-1]["grad_mode"] == "fd" and abs(th_l[0] - 9.0) < 1e-6
    print(f"\nPKMM alpha* = {theta[0]:.6f} (N2 = {obj.quantity(theta):.3e}); L-BFGS from 9.0 stays at {th_l[0]:.6f}")


def test_pkmm_death_line_point_rediscovered():
    obj = make_objective("pkmm_negativity", free=("beta",), alpha=9.1, gamma=3.0, delta=0.02)
    theta, hist = optimize(lambda th: obj.quantity(th) ** 2, None, np.array([5.0]), method="nelder-mead",
                           bounds=((0.05, 25.0),), xatol=1e-7, fatol=1e-30)
    root = brentq(lambda b: pkmm_negativity_estimator(9.1, b, 3.0, 0.02), 5.0, 15.0, xtol=1e-9)
    assert abs(theta[0] - root) < 1e-4
    rel = abs(theta[0] / 9.803 - 1.0)
    assert rel < 0.02, f"death line d*/T = {theta[0]:.4f} vs digitized 9.803 (rel {rel:.4f})"
    print(f"\nPKMM death line at Omega T = 9.1: d*/T = {theta[0]:.5f} (digitized 9.803, rel {rel:.2e})")


# ---- 2. lattice re-discovery --------------------------------------------------

def test_lattice_gap_optimum_matches_perturbative_engine():
    N = 20
    K = harmonic_chain_K(N, 0.6, bc="dirichlet")
    kernel = wightman_lattice(K)
    F = np.eye(N)
    lam = 0.1

    def pert(gap):
        st = udw_pair_state(kernel, UDWDetector(switching("cos2", 2.0, 0.0), F[7]),
                            UDWDetector(switching("cos2", 2.0, 0.0), F[11]), lam / math.sqrt(2 * gap), gap,
                            rtol=1e-10, max_doublings=14)
        return st.log_negativity

    ref = minimize_scalar(lambda g: -pert(g), bounds=(1.6, 3.2), method="bounded", options={"xatol": 1e-6})
    obj = make_objective("negativity", K=K, detectors=[{"site": 7}, {"site": 11}], gaps=(2.0, 2.0), window=(0.0, 2.0),
                         basis="cos2", n_coeff=1, lam_max=lam, free=("gap",), splits=256, quantity="surrogate")
    f = lambda th: -obj.quantity_aux(jnp.stack([th[0], th[0]]))[0]
    theta, hist = optimize(f, None, np.array([2.8]), method="lbfgs", bounds=((1.6, 3.2),), maxiter=60)
    assert hist[-1]["grad_mode"] == "jax"
    dev = abs(theta[0] - ref.x)
    assert dev < 5e-3, f"JAX gap* {theta[0]:.5f} vs perturbative {ref.x:.5f} (|d| = {dev:.2e}, lambda^2 = {lam**2})"
    rt = obj.roundtrip(np.array([theta[0], theta[0]]))
    assert rt["audits_passed"] and rt["E_N"] > 0.0
    assert abs(rt["E_N"] - rt["E_N_pert"]) < 3.0 * lam ** 2 * rt["E_N_pert"]  # Gate-A c lambda^2 bound
    print(f"\nlattice gap*: JAX {theta[0]:.5f} vs PKMM-formula engine {ref.x:.5f}; E_N {rt['E_N']:.4e} "
          f"(pert {rt['E_N_pert']:.4e}), comm fraction {rt['comm_fraction']:.3f}")


# ---- 3. Hotta -----------------------------------------------------------------

@pytest.mark.parametrize("h, k", [(1.0, 0.5), (0.3, 2.5), (5.0, 0.1)])
def test_hotta_angle_recovered(h, k):
    obj = make_objective("qet_energy", h=h, k=k)
    for th in (0.05, 0.3, 1.0):
        assert abs(float(obj.quantity([th])) - hotta.e_b_closed(h, k, th)) < 5e-14  # float64 roundoff on an O(1) energy
    th_star = hotta.theta_optimal(h, k)
    th_l, hist_l = optimize(obj, None, np.array([0.6]), method="lbfgs", maxiter=200)
    assert abs(th_l[0] - th_star) < 1e-8, f"L-BFGS theta* {th_l[0]!r} vs closed form {th_star!r}"
    th_a, _ = optimize(obj, None, np.array([0.6]), method="adam", n_iter=600, lr=0.02)
    assert abs(th_a[0] - th_star) < 1e-8, f"Adam theta* {th_a[0]!r} vs closed form {th_star!r}"
    rt = obj.roundtrip(th_l)
    assert rt["audits_passed"] and abs(rt["E_B"] - hotta.e_b_optimal(h, k)) < 1e-12
    assert rt["E_A"] >= rt["E_B"]


# ---- 4. QEI -------------------------------------------------------------------

@pytest.fixture(scope="module")
def qei_obj():
    return make_objective("qei_ratio", N=161, site=80, family="gaussian", bounds=(2.5, 7.0))


def test_qei_objective_agrees_with_numpy_and_fd(qei_obj):
    obj = qei_obj
    q = jax.jit(lambda th: obj.quantity_aux(th)[0])
    for th in (3.0, 5.5):
        r = float(q(jnp.array([th])))
        rt = obj.roundtrip([th], weight_floor=0.0)
        assert abs(r - rt["ratio"]) < 1e-10, f"theta={th}: jax {r!r} vs numpy {rt['ratio']!r}"
        assert rt["audits_passed"], rt["audits"]
        assert 0.97 < r < 1.0
    g = float(jax.grad(lambda th: obj.quantity_aux(th)[0])(jnp.array([3.0]))[0])
    h = 1e-3
    c = lambda e: (obj.roundtrip([3.0 + e], weight_floor=0.0)["ratio"] - obj.roundtrip([3.0 - e], weight_floor=0.0)["ratio"]) / (2 * e)
    ref = (4 * c(h / 2) - c(h)) / 3
    assert abs(g - ref) / abs(ref) < 1e-6, f"dratio/dtau0 jax {g!r} vs FD {ref!r}"


def test_qei_ascent_recovers_interior_optimum(qei_obj):
    from vacuum.inequalities.qei import gaussian_family, optimize_sampling

    obj = qei_obj
    theta, hist = optimize(obj, None, np.array([3.0]), method="lbfgs", bounds=obj.bounds, maxiter=100)
    ref = optimize_sampling(161, 80, gaussian_family(), [3.0], bounds=[(2.5, 7.0)], m=0.0, bc="dirichlet")
    assert 2.5 + 1e-3 < theta[0] < 7.0 - 1e-3, f"tau0* = {theta[0]:.4f} stuck at a bound"
    assert abs(theta[0] - ref.theta[0]) < 5e-2, f"JAX tau0* {theta[0]:.4f} vs numpy optimize_sampling {ref.theta[0]:.4f}"
    rt = obj.roundtrip(theta, weight_floor=0.0)
    assert 0.98 < rt["ratio"] < 1.0 and rt["ratio"] >= obj.roundtrip([3.0], weight_floor=0.0)["ratio"]
    assert abs(rt["ratio"] - ref.ratio) < 1e-6, f"ratio at the optimum: JAX round trip {rt['ratio']!r} vs numpy {ref.ratio!r}"
    print(f"\nQEI ascent: tau0* = {theta[0]:.4f} (numpy {ref.theta[0]:.4f}), ratio {rt['ratio']:.6f} (numpy {ref.ratio:.6f})")


# ---- 5. round trip + constraints ------------------------------------------

def test_negativity_roundtrip_and_signaling_proxy():
    obj = make_objective("negativity", N=20, mass=0.6, bc="dirichlet", detectors=[{"site": 8}, {"site": 9}],
                         gaps=(2.0, 2.0), window=(0.0, 2.0), basis="fourier", n_coeff=5, lam_max=0.24,
                         free=("lam", "lam_max", "gap"), splits=256, quantity="surrogate")
    th = obj.theta0.copy()
    th[1], th[3] = 0.3, -0.2  # a non-cos2 shape
    q, aux = obj.quantity_aux(jnp.asarray(th))
    rt = obj.roundtrip(th)
    assert rt["audits_passed"] and rt["converged"]
    assert abs(float(aux["E_N"]) - rt["E_N"]) < 1e-5 * max(1.0, 1.0)  # fixed 256 substeps vs the gated run
    assert abs(float(aux["work"]) - rt["work"]) < 1e-5
    assert abs(float(aux["comm_fraction"]) - rt["comm_fraction"]) < 1e-8, (float(aux["comm_fraction"]), rt["comm_fraction"])
    assert abs(rt["E_N"] - rt["E_N_pert"]) < 0.1 * rt["E_N"]
    # constraints: penalty and check
    C = constraints(energy_budget=0.5 * float(aux["work"]), signaling_bound=0.5 * rt["comm_fraction"], weight=10.0)
    pen = float(C.penalty(th, aux))
    assert pen > 0.0
    chk = C.check({"work": rt["work"], "comm_fraction": rt["comm_fraction"], "audits_passed": True})
    assert chk["energy_budget"] is False and chk["signaling_bound"] is False and chk["admissible"] is False
    C2 = constraints(energy_budget=2 * rt["work"], signaling_bound=1.0)
    assert float(C2.penalty(th, aux)) == 0.0 and C2.check({"work": rt["work"], "comm_fraction": rt["comm_fraction"], "audits_passed": True})["admissible"]
    # the energy-budget projection hits the budget on the twin
    th_p = project_energy_budget(obj, th, 0.5 * float(aux["work"]), index=5)
    assert abs(float(obj.quantity_aux(jnp.asarray(th_p))[1]["work"]) - 0.5 * float(aux["work"])) < 1e-10
    # unpack / names
    p = obj.unpack(th)
    assert set(p) >= {"lam", "lam_max", "gap"} and len(obj.names) == th.size
    with pytest.raises(ValueError):
        make_objective("negativity", N=20, mass=0.6, detectors=[{"site": 8}, {"site": 9}], gaps=(2.0, 2.0),
                       window=(0.0, 2.0), free=("bogus",))
    with pytest.raises(NotImplementedError):
        make_objective("gain_per_loss")
