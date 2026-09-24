"""Layer-6 anchor: vacuum.opt.gjax_detectors mirrors the detector stack and is differentiable.

1. FORWARD AGREEMENT — on the three Gate-A setups of
   tests/test_gate_crossvalidation.py (pointlike/smeared, dirichlet/periodic,
   equal/unequal gaps, coincident/displaced cos^2 switchings) the twin of
   ``protocol_evolve_fixed`` reproduces the numpy stepper at the same
   substeps to 1e-10 (measured 2.4e-14) in V, drive work and E_N; the
   noisy + thermal path (Strang-interleaved ``imperfection_channel``,
   thermal field) likewise; smearing profiles and their analytic parameter
   derivatives to 1e-12; every ``with_imperfections`` axis exactly.
2. GRADIENTS — jax.grad of E_N and of the drive work through a FULL
   harvesting protocol (initial state -> midpoint/Strang evolution ->
   detector block -> partial-transpose spectrum) with respect to the
   switching-waveform samples, both detector gaps, the smearing centre and
   width and the coupling amplitude, against Richardson-extrapolated
   central finite differences of the numpy stepper: relative error < 1e-6
   (measured <= 4e-9), noiseless and under loss + dephasing on a thermal
   field.  The dead-region surrogate keeps a finite gradient that matches
   the finite difference of the numpy partial-transpose minimum eigenvalue.
3. jit == eager.

Requires JAX (importorskip); ~60 s.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from vacuum.core import harmonic_chain_K, partial_transpose, symplectic_eigenvalues  # noqa: E402
from vacuum.detectors import (  # noqa: E402
    attach_detectors,
    detector_block,
    harvested_log_negativity,
    imperfection_channel,
    initial_state,
    protocol_evolve_fixed,
    run_harvesting,
    smearing,
    switching,
    with_imperfections,
)
from vacuum.opt import gjax_detectors as D  # noqa: E402

FORWARD_TOL = 1e-10
GRAD_TOL = 1e-6


def _setup(name):
    if name == "A":
        return dict(N=20, m=0.6, bc="dirichlet", prof=[8, 9], gaps=(2.0, 2.0), centers=(0.0, 0.0), T=2.0)
    if name == "B":
        return dict(N=16, m=0.8, bc="periodic",
                    prof=[np.asarray(smearing("gaussian", 5.0, 1.0, 16, bc="periodic")),
                          np.asarray(smearing("gaussian", 8.0, 1.0, 16, bc="periodic"))],
                    gaps=(2.5, 2.0), centers=(0.0, 0.0), T=2.0)
    return dict(N=20, m=0.6, bc="dirichlet", prof=[8, 9], gaps=(2.5, 2.0), centers=(0.0, 0.7), T=2.0)


def _cos2_j(t, T, c):
    u = t - c
    return jnp.where(jnp.abs(u) <= T, jnp.cos(0.5 * jnp.pi * u / T) ** 2, 0.0)


def fd(f, h):
    c = lambda e: (f(e) - f(-e)) / (2.0 * e)
    return (4.0 * c(h / 2.0) - c(h)) / 3.0


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_forward_agreement_gate_a_setups(name):
    s = _setup(name)
    K = harmonic_chain_K(s["N"], s["m"], bc=s["bc"])
    lam = 0.24
    chis = [switching("cos2", s["T"], t0) for t0 in s["centers"]]
    ts = np.array([min(c.support[0] for c in chis), max(c.support[1] for c in chis)])
    splits = 256
    init = initial_state(K, s["prof"], s["gaps"], 0.0)
    r = protocol_evolve_fixed(init.V, lambda t: attach_detectors(K, s["prof"], s["gaps"], np.array([lam * c(t) for c in chis])),
                              ts, splits=splits)
    E_np = harvested_log_negativity(detector_block(r.V, s["N"])).E_N
    F = D.profile_matrix(s["N"], s["prof"])
    out = D.run_harvesting(K, F, lambda t: jnp.asarray(s["gaps"]),
                           lambda t: jnp.array([lam * _cos2_j(t, s["T"], c) for c in s["centers"]]), ts, splits=splits)
    dV = float(np.max(np.abs(np.asarray(out.V) - r.V)))
    dW = abs(float(out.work) - r.work)
    dE = abs(float(out.E_N) - E_np)
    assert dV < FORWARD_TOL and dW < FORWARD_TOL and dE < FORWARD_TOL, f"[{name}] dV={dV:.2e} dW={dW:.2e} dE={dE:.2e}"
    assert np.max(np.abs(np.asarray(D.initial_state(K, F, jnp.asarray(s["gaps"]))) - init.V)) == 0.0
    assert E_np > 1e-5 and abs(float(out.surrogate) - float(out.E_N)) < 1e-14  # alive: surrogate == E_N
    print(f"\n[{name}] E_N = {E_np:.6e}: |dV| {dV:.1e}, |dW| {dW:.1e}, |dE_N| {dE:.1e} (fixed 256 substeps)")


def test_forward_agreement_noisy_thermal():
    s = _setup("A")
    N = s["N"]
    K = harmonic_chain_K(N, s["m"], bc=s["bc"])
    det = (N, N + 1)
    lam, T, ts, splits = 0.3, 0.4, np.array([-2.0, 2.0]), 128
    chi = switching("cos2", 2.0, 0.0)
    init = initial_state(K, s["prof"], s["gaps"], T)
    ch = imperfection_channel(det, kappa=0.05, nbar=0.02, gamma_phi=0.01, gaps=s["gaps"])
    r = protocol_evolve_fixed(init.V, lambda t: attach_detectors(K, s["prof"], s["gaps"], lam * chi(t)), ts, splits=splits,
                              channels=(ch,))
    F = D.profile_matrix(N, s["prof"])
    chj = D.rate_channel(N + 2, det, kappa=0.05, nbar=0.02, gamma_phi=0.01, gaps=jnp.asarray(s["gaps"]))
    out = D.run_harvesting(K, F, lambda t: jnp.asarray(s["gaps"]), lambda t: lam * _cos2_j(t, 2.0, 0.0), ts, splits=splits,
                           T=T, channels=(chj,))
    assert float(np.max(np.abs(np.asarray(out.V) - r.V))) < FORWARD_TOL
    assert abs(float(out.work) - r.work) < FORWARD_TOL
    assert abs(float(out.dissipated) - r.dissipated) < FORWARD_TOL
    for axis in ("isotropic", "p", "x"):
        Vn = with_imperfections(init.V, det, eta=0.9, nbar=0.1, dephasing=0.05, axis=axis, gaps=s["gaps"])
        Vj = D.with_imperfections(init.V, det, eta=0.9, nbar=0.1, dephasing=0.05, axis=axis, gaps=jnp.asarray(s["gaps"]))
        assert float(np.max(np.abs(np.asarray(Vj) - Vn))) < 1e-14, axis


def test_smearing_profiles_and_derivatives():
    for kind, width in (("gaussian", 1.2), ("cos2", 2.2)):
        for bc in ("open", "periodic"):
            S = smearing(kind, 5.3, width, 16, bc=bc)
            fn = D.gaussian_profile if kind == "gaussian" else D.cos2_profile
            assert float(np.max(np.abs(np.asarray(fn(16, 5.3, width, bc=bc)) - S.F))) < 1e-12
            dc = np.asarray(jax.jacfwd(lambda c: fn(16, c, width, bc=bc))(5.3))
            dw = np.asarray(jax.jacfwd(lambda w: fn(16, 5.3, w, bc=bc))(width))
            assert float(np.max(np.abs(dc - S.dF_dcenter))) < 1e-12, (kind, bc)
            assert float(np.max(np.abs(dw - S.dF_dwidth))) < 1e-12, (kind, bc)
    F = D.profile_matrix(16, [3, np.asarray(smearing("gaussian", 5.3, 1.2, 16))])
    assert F.shape == (16, 2) and float(F[3, 0]) == 1.0 and abs(float(jnp.sum(F[:, 1])) - 1.0) < 1e-14
    with pytest.raises(ValueError):
        D.profile_matrix(16, [np.ones(15)])


# ---- gradients through a full harvesting protocol ----------------------------

N_G = 20
K_G = harmonic_chain_K(N_G, 0.6, bc="dirichlet")
TS_G = np.array([-2.0, 2.0])
SPLITS_G = 128
KNOTS = np.linspace(-2.0, 2.0, 13)
N_A = 11
NAMES = [f"a{k}" for k in range(N_A)] + ["gapA", "gapB", "centerB", "widthB", "lam_max"]


def _unpack(th):
    return th[:N_A], th[N_A:N_A + 2], th[N_A + 2], th[N_A + 3], th[N_A + 4]


def _np_run(th, noisy, T):
    a, g, cB, wB, lm = _unpack(th)
    samples = np.concatenate([[0.0], np.asarray(a), [0.0]])
    prof = [8, smearing("gaussian", float(cB), float(wB), N_G).F]
    init = initial_state(K_G, prof, tuple(float(x) for x in g), T)
    ch = (imperfection_channel((N_G, N_G + 1), kappa=0.0002, nbar=0.01, gamma_phi=1e-05,
                               gaps=tuple(float(x) for x in g)),) if noisy else ()
    r = protocol_evolve_fixed(init.V, lambda t: attach_detectors(K_G, prof, g, float(lm) * np.interp(t, KNOTS, samples)),
                              TS_G, splits=SPLITS_G, channels=ch)
    V_AB = detector_block(r.V, N_G)
    nu_min = float(np.min(symplectic_eigenvalues(partial_transpose(V_AB, [1]))))
    return harvested_log_negativity(V_AB).E_N, r.work, -math.log(2.0 * nu_min)


def _jax_run(th, noisy, T):
    a, g, cB, wB, lm = _unpack(th)
    samples = jnp.concatenate([jnp.zeros(1), a, jnp.zeros(1)])
    F = jnp.stack([D.point_profile(N_G, 8), D.gaussian_profile(N_G, cB, wB)], axis=1)
    ch = (D.rate_channel(N_G + 2, (N_G, N_G + 1), kappa=0.0002, nbar=0.01, gamma_phi=1e-05, gaps=g),) if noisy else ()
    return D.run_harvesting(K_G, F, lambda t: g, lambda t: lm * jnp.interp(t, jnp.asarray(KNOTS), samples) * jnp.ones(2),
                            TS_G, splits=SPLITS_G, T=T, channels=ch)


THETA_G = np.concatenate([0.35 * np.sin(np.pi * (KNOTS[1:-1] + 2) / 4) ** 2 * (1 + 0.2 * np.cos(3 * KNOTS[1:-1])),
                          [2.0, 2.3], [10.4], [1.1], [1.0]])


@pytest.mark.parametrize("noisy, T", [(False, 0.0), (True, 0.05)])
def test_gradients_through_full_protocol(noisy, T):
    E0, W0, _ = _np_run(THETA_G, noisy, T)
    out = _jax_run(THETA_G, noisy, T)
    assert E0 > 1e-5, f"fixture must be alive: E_N = {E0:.3e}"
    assert abs(float(out.E_N) - E0) < FORWARD_TOL and abs(float(out.work) - W0) < FORWARD_TOL
    gE = np.asarray(jax.grad(lambda th: _jax_run(th, noisy, T).E_N)(jnp.asarray(THETA_G)))
    gW = np.asarray(jax.grad(lambda th: _jax_run(th, noisy, T).work)(jnp.asarray(THETA_G)))
    worst = {}
    # every parameter block: 3 waveform samples, both gaps, centre, width, coupling
    for i in (0, 5, 10, 11, 12, 13, 14, 15):
        e = np.zeros_like(THETA_G)
        e[i] = 1.0
        rE = fd(lambda s: _np_run(THETA_G + s * e, noisy, T)[0], 1e-3)
        rW = fd(lambda s: _np_run(THETA_G + s * e, noisy, T)[1], 1e-3)
        relE = abs(gE[i] - rE) / abs(rE)
        relW = abs(gW[i] - rW) / abs(rW)
        worst[NAMES[i]] = (relE, relW)
        assert relE < GRAD_TOL, f"dE_N/d{NAMES[i]}: jax {gE[i]!r} vs FD {rE!r} (rel {relE:.2e})"
        assert relW < GRAD_TOL, f"dW/d{NAMES[i]}: jax {gW[i]!r} vs FD {rW!r} (rel {relW:.2e})"
    print(f"\n[noisy={noisy}, T={T}] E_N = {E0:.4e}; worst rel gradient error: "
          f"E_N {max(v[0] for v in worst.values()):.1e}, work {max(v[1] for v in worst.values()):.1e}")


def test_dead_region_surrogate_gradient():
    """Far-apart detectors: E_N = 0 with zero gradient; the surrogate carries one."""
    th = THETA_G.copy()
    th[N_A + 2] = 16.0  # move detector B far from A (site 8): sudden death
    E0, _, S0 = _np_run(th, False, 0.0)
    assert E0 == 0.0 and S0 < 0.0
    out = _jax_run(th, False, 0.0)
    assert float(out.E_N) == 0.0 and abs(float(out.surrogate) - S0) < 1e-12
    gE = np.asarray(jax.grad(lambda t: _jax_run(t, False, 0.0).E_N)(jnp.asarray(th)))
    gS = np.asarray(jax.grad(lambda t: _jax_run(t, False, 0.0).surrogate)(jnp.asarray(th)))
    assert np.all(gE == 0.0)
    for i in (5, 11, 12, 15):
        e = np.zeros_like(th)
        e[i] = 1.0
        r = fd(lambda s: _np_run(th + s * e, False, 0.0)[2], 1e-3)
        assert abs(gS[i] - r) / max(abs(r), 1e-12) < GRAD_TOL, f"surrogate d/d{NAMES[i]}: {gS[i]!r} vs FD {r!r}"
    assert np.max(np.abs(gS)) > 0.0


def test_jit_matches_eager():
    f = lambda th: _jax_run(th, False, 0.0).E_N
    assert abs(float(jax.jit(f)(THETA_G)) - float(f(THETA_G))) < 1e-13
    g = jax.jit(jax.grad(f))(THETA_G)
    assert float(np.max(np.abs(np.asarray(g) - np.asarray(jax.grad(f)(THETA_G))))) < 1e-12
    grid = D.evolution_grid(np.array([0.0, 1.0, 3.0]), 2)
    assert np.allclose(grid[0], [0.25, 0.75, 1.5, 2.5]) and np.allclose(grid[1], [0.5, 0.5, 1.0, 1.0])
    with pytest.raises(ValueError):
        D.evolution_grid(np.array([1.0, 0.5]), 1)
