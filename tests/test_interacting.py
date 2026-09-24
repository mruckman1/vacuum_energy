"""Layer 7 / L5 anchors: the lattice lambda-phi^4 chain (vacuum.interacting).

Four anchors, in order (task spec):

1. lambda = 0 reproduces the exact Gaussian core: DMRG energy, <phi phi>,
   <pi pi>, block entropies, disjoint-block negativity and mutual information
   converge to ``vacuum.core`` values exponentially in the local cutoff
   n_max, reaching <= 1e-6 (measured at L = 12, m = 0.5: dE 5e-10, dV 1e-8,
   dS 2e-9, dE_N(1+1) 1.4e-7 and dE_N(2+2 adjacent) ~6e-7 at n_max = 8;
   dE_N(2+2, separation 1) 2e-7 and dMI 2e-7 at n_max = 6 — the separated
   blocks converge fastest, the adjacent 2+2 blocks slowest, ~5.5x/level). The DMRG energy is also a Rayleigh-Ritz upper bound,
   non-increasing in n_max (projected operators, phi4.py docstring).
2. Small lambda: the ground-energy shift matches Rayleigh-Schroedinger
   theory to second order (residual scales as lambda^3; E_1, E_2 also
   verified against dense ED on two sites), and the two-point functions
   match the O(lambda) tadpole (Hartree) chain K + lambda diag(<phi^2>/2)
   with an O(lambda^2) residual.
3. chi-doubling moves every reported number by < 1% (measured: < 1e-7) and
   truncation errors are tracked in the info record.
4. Passivity: random single- and two-site unitaries never lower <H> on the
   interacting ground state; the audit fires on an active (squeezed) state.

Plus the QEI pipeline anchor: at lambda = 0 the TEBD-evolved smeared energy
of a locally squeezed state equals the exact Gaussian smeared energy of
``vacuum.inequalities.qei`` (deviation dominated by the local-basis
truncation of the squeezer: 7.6e-5 at n_max = 6, < 1e-5 at n_max = 8 for
tau0 = 0.6; Trotter/chi/dt errors < 1e-7 at order 4, Simpson on the dt =
0.2 grid < 1e-8 against the Gauss-Legendre route).

Chain parameters (documented tuning): m = 0.5 gives a correlation length
~2 sites (mutual information alive over the block separations probed) with
a local occupation nbar ~ 0.054 (fast n_max convergence); lambda = 1 for
the interacting checks is lambda/mu_H^2 ~ 2.7, far below the critical ~65
(phi4.py, CRITICAL_RATIO_LAMBDA_OVER_MU2). Sizes are chosen for the
~90 s file budget; the L >= 32 sweeps live in the papers notebook.

TeNPy (``.[tensor]`` extra) is importorskip'd; the numpy-only checks run
regardless. Conventions: hbar = 1, block ordering, nats (docs/API.md).
"""

import logging
import warnings

import numpy as np
import pytest

from vacuum.core import (
    entropy,
    ground_state_cov,
    log_negativity,
    mean_energy,
    mutual_information,
    reduce,
)
from vacuum.interacting.phi4 import (
    CRITICAL_RATIO_LAMBDA_OVER_MU2,
    first_order_covariance,
    free_coupling_matrix,
    hartree_mass_squared,
    local_boson_ops,
    oscillator_frequency,
    perturbative_energy,
)

MASS = 0.5
L_FREE = 12
NMAX_SWEEP = (2, 3, 4, 6, 8)
NMAX_DENSE_4SITE = 6  # 2+2-site reduced states: d^4 = 2401 at n_max = 6 (1 s eig)


def _tenpy_modules():
    pytest.importorskip("tenpy")
    logging.getLogger("tenpy").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message="unit_cell_width")
    import vacuum.interacting as vi
    return vi


# --------------------------------------------------------------------------
# numpy-only checks (no TeNPy)
# --------------------------------------------------------------------------


def test_local_ops_are_projections_not_products():
    """P O P: Xsq equals X @ X except the top diagonal; [X, Pim] = 1 below it."""
    n_max, omega = 6, 1.3
    ops = local_boson_ops(n_max, omega)
    d = n_max + 1
    diff = ops["Xsq"] - ops["X"] @ ops["X"]
    off = diff.copy()
    off[d - 1, d - 1] = 0.0
    assert np.max(np.abs(off)) < 1e-14, "phi^2 projection differs from the product off the top level"
    assert diff[d - 1, d - 1] > 0.0, "the projected phi^2 must exceed the truncated product at the top"
    comm = ops["X"] @ ops["Pim"] - ops["Pim"] @ ops["X"]  # = -i [phi, pi] = 1 (below the cutoff)
    assert np.allclose(comm[: d - 1, : d - 1], np.eye(d - 1), atol=1e-13), "[phi, pi] != i below the cutoff"
    assert np.allclose(comm[d - 1, d - 1], 1.0 - d, atol=1e-12), "top-level commutator defect is -(n_max+1)"
    for k in ("X", "Xsq", "Psq", "X4"):
        assert np.allclose(ops[k], ops[k].T), f"{k} must be symmetric"
    assert np.min(np.linalg.eigvalsh(ops["Psq"])) > -1e-13, "P pi^2 P must be PSD"
    assert np.allclose(ops["Pim"], -ops["Pim"].T), "Pim must be antisymmetric"


def test_basis_frequency_minimizes_local_occupation():
    """omega* = sqrt(<pi^2>/<phi^2>) minimizes nbar(omega) of the free local state."""
    N = 16
    K = free_coupling_matrix(N, MASS)
    V = ground_state_cov(K)
    c = N // 2
    x2, p2 = V[c, c], V[N + c, N + c]
    om_star = oscillator_frequency(N, MASS)

    def nbar(om):
        return 0.5 * (om * x2 + p2 / om - 1.0)

    grid = om_star * np.exp(np.linspace(-1.0, 1.0, 201))
    assert nbar(om_star) <= np.min(nbar(grid)) + 1e-12, "omega* is not the occupation minimum"
    assert abs(nbar(om_star) - (np.sqrt(x2 * p2) - 0.5)) < 1e-12, "nbar_min != nu_loc - 1/2"


def test_perturbation_theory_against_dense_ed_two_sites():
    """E_1 exact and E_2 exact on two oscillators (dense Fock ED, n_max = 39)."""
    N, m = 2, 0.7
    K = free_coupling_matrix(N, m, "open")
    om = oscillator_frequency(N, m, "open")
    D = 40
    ops = local_boson_ops(D - 1, om)
    I = np.eye(D)
    H0 = (0.5 * (np.kron(ops["Psq"], I) + np.kron(I, ops["Psq"]))
          + 0.5 * K[0, 0] * np.kron(ops["Xsq"], I) + 0.5 * K[1, 1] * np.kron(I, ops["Xsq"])
          + K[0, 1] * np.kron(ops["X"], ops["X"]))
    V4 = (np.kron(ops["X4"], I) + np.kron(I, ops["X4"])) / 24.0
    pt = perturbative_energy(N, m, 0.0, "open")
    E = {lam: float(np.linalg.eigvalsh(H0 + lam * V4)[0]) for lam in (0.0, 0.02, 0.04, 0.08)}
    assert abs(E[0.0] - pt["E0"]) < 1e-10, f"zero-point energy mismatch {E[0.0]} vs {pt['E0']}"
    ratios = []
    for lam in (0.02, 0.04, 0.08):
        r1 = E[lam] - (pt["E0"] + lam * pt["E1"])
        r2 = r1 - lam * lam * pt["E2"]
        # first-order residual IS lambda^2 E_2 (to ~1%): E_1 exact, E_2 right
        assert abs(r1 - lam * lam * pt["E2"]) < 0.05 * abs(lam * lam * pt["E2"]), (
            f"lam={lam}: first-order residual {r1:.3e} vs lam^2 E2 {lam * lam * pt['E2']:.3e}")
        ratios.append(r2 / lam**3)
    ratios = np.asarray(ratios)
    assert np.max(np.abs(ratios - ratios[0])) < 0.1 * abs(ratios[0]), (
        f"second-order residual is not O(lambda^3): resid/lam^3 = {ratios}")


# --------------------------------------------------------------------------
# TeNPy fixtures (module scoped: every DMRG run is done once)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def free_chain():
    K = free_coupling_matrix(L_FREE, MASS)
    V = ground_state_cov(K)
    return K, V


@pytest.fixture(scope="module")
def free_ground_states():
    vi = _tenpy_modules()
    return {n: vi.phi4_ground_mps(L_FREE, MASS, 0.0, n_max=n, chi_max=32) for n in NMAX_SWEEP}


@pytest.fixture(scope="module")
def lam1_ground_states():
    """lambda = 1 at chi = 16 and chi = 32 (warm-started): the chi-doubling pair."""
    vi = _tenpy_modules()
    g16 = vi.phi4_ground_mps(L_FREE, MASS, 1.0, n_max=6, chi_max=16)
    g32 = vi.phi4_ground_mps(L_FREE, MASS, 1.0, n_max=6, chi_max=32, psi0=g16.psi)
    return g16, g32


def _blocks(c):
    return {
        "EN_1+1_d0": ([c - 1], [c]),
        "EN_2+2_d0": ([c - 2, c - 1], [c, c + 1]),
        "EN_2+2_d1": ([c - 2, c - 1], [c + 1, c + 2]),
        "MI_1+1_d3": ([c - 2], [c + 1]),
        "MI_2+2_d1": ([c - 2, c - 1], [c + 1, c + 2]),
    }


# --------------------------------------------------------------------------
# Anchor 1: lambda = 0 vs the Gaussian core
# --------------------------------------------------------------------------


def test_free_anchor_rayleigh_ritz_monotone(free_chain, free_ground_states):
    """E_DMRG(n_max) >= E_exact and non-increasing in n_max (projected H)."""
    K, V = free_chain
    E_ex = mean_energy(V, K)
    Es = [free_ground_states[n].E0 for n in NMAX_SWEEP]
    for n, E in zip(NMAX_SWEEP, Es):
        assert E >= E_ex - 1e-12, f"n_max={n}: DMRG energy {E} below the exact {E_ex}"
    for (n1, e1), (n2, e2) in zip(zip(NMAX_SWEEP, Es), zip(NMAX_SWEEP[1:], Es[1:])):
        assert e2 <= e1 + 1e-12, f"energy rose from n_max={n1} ({e1}) to {n2} ({e2})"
    print("\nRayleigh-Ritz: E(n_max) - E_exact =",
          ", ".join(f"{n}: {E - E_ex:.2e}" for n, E in zip(NMAX_SWEEP, Es)))


def test_free_anchor_convergence_in_nmax(free_chain, free_ground_states):
    """THE lambda = 0 ANCHOR: every observable converges to the core to <= 1e-6."""
    vi = _tenpy_modules()
    K, V = free_chain
    c = L_FREE // 2
    E_ex = mean_energy(V, K)
    S_ex = np.array([entropy(reduce(V, range(l))) for l in range(1, L_FREE)])
    blocks = _blocks(c)
    exact = {}
    for name, (A, B) in blocks.items():
        exact[name] = mutual_information(V, A, B) if name.startswith("MI") else log_negativity(V, A, B)

    devs = {}
    for n in NMAX_SWEEP:
        gs = free_ground_states[n]
        Vm = vi.covariance_from_mps(gs.psi)
        d = {"E": abs(gs.E0 - E_ex),
             "V": float(np.max(np.abs(Vm - V))),
             "S": float(np.max(np.abs(vi.block_entropies(gs.psi) - S_ex)))}
        for name, (A, B) in blocks.items():
            # 4-site dense states cost d^4: only the slowest-converging one
            # (adjacent 2+2 blocks, ~5.5x per level) is pushed to n_max = 8
            if len(A) + len(B) > 2 and n > NMAX_DENSE_4SITE and name != "EN_2+2_d0":
                continue
            val = (vi.mutual_information_mps(gs.psi, A, B) if name.startswith("MI")
                   else vi.log_negativity_mps(gs.psi, A, B))
            d[name] = abs(val - exact[name])
        devs[n] = d
    print("\nlambda = 0 anchor, |MPS - Gaussian core| vs n_max:")
    for n in NMAX_SWEEP:
        print(f"  n_max={n}: " + ", ".join(f"{k}={v:.1e}" for k, v in devs[n].items()))

    # exponential convergence: every deviation shrinks along the sweep
    for key in devs[NMAX_SWEEP[0]]:
        seq = [devs[n][key] for n in NMAX_SWEEP if key in devs[n]]
        for a, b in zip(seq, seq[1:]):
            assert b < a or b < 1e-9, f"{key} did not decrease along n_max: {seq}"
    # the target: <= 1e-6 at the largest cutoff each observable was evaluated at
    final = {key: [devs[n][key] for n in NMAX_SWEEP if key in devs[n]][-1] for key in devs[NMAX_SWEEP[0]]}
    for key, val in final.items():
        assert val <= 1e-6, f"{key} converged only to {val:.2e} > 1e-6 (n_max sweep {NMAX_SWEEP})"
    # truncation errors are tracked and negligible for this gapped chain
    for n in NMAX_SWEEP:
        assert free_ground_states[n].info["max_trunc_err"] < 1e-12
        assert free_ground_states[n].info["converged"]


def test_free_anchor_sudden_death_reproduced(free_chain, free_ground_states):
    """Small-block sudden death: E_N is exactly zero one gap beyond adjacency."""
    vi = _tenpy_modules()
    _, V = free_chain
    c = L_FREE // 2
    # float64 Gaussian routes return roundoff (~1e-14) rather than an exact 0
    # at death (tests/test_negativity_death.py docstring), and the truncated-
    # basis MPS state is not exactly Gaussian, so its dead-pair E_N is a small
    # n_max-truncation residual (measured 7.6e-9 at n_max = 6 for 1+1 blocks,
    # shrinking with n_max). Live values one step closer are 1e-2..1e-1, so
    # 1e-6 separates dead from alive by four orders of magnitude.
    for A, B in (([c - 1], [c + 1]), ([c - 2, c - 1], [c + 2, c + 3])):
        ex = log_negativity(V, A, B)
        assert ex < 1e-10, f"Gaussian core: blocks {A},{B} expected dead, E_N = {ex}"
        dead = [vi.log_negativity_mps(free_ground_states[n].psi, A, B)
                for n in (NMAX_DENSE_4SITE, 8) if len(A) + len(B) == 2 or n <= NMAX_DENSE_4SITE]
        assert max(dead) < 1e-6, f"MPS: blocks {A},{B} expected dead, E_N = {dead}"
        if len(dead) == 2:  # the residual is truncation: it shrinks with n_max
            assert dead[1] < dead[0], f"dead-pair residual not shrinking with n_max: {dead}"
        print(f"\nsudden death {A}|{B}: Gaussian {ex:.1e}, MPS residual {dead}")


# --------------------------------------------------------------------------
# Anchor 2: small lambda vs perturbation theory
# --------------------------------------------------------------------------


def test_perturbation_theory_dmrg():
    """Energy residual after 2nd order is O(lambda^3); covariance matches the
    O(lambda) tadpole chain with an O(lambda^2) residual."""
    vi = _tenpy_modules()
    L = 10
    lams = (0.1, 0.2)
    V0 = ground_state_cov(free_coupling_matrix(L, MASS))
    resid2 = {}
    cov_dev = {}
    cov_shift = {}
    for lam in lams:
        gs = vi.phi4_ground_mps(L, MASS, lam, n_max=8, chi_max=32)
        pt = perturbative_energy(L, MASS, lam)
        resid1 = gs.E0 - (pt["E0"] + lam * pt["E1"])
        resid2[lam] = gs.E0 - pt["total"]
        assert abs(resid1 - lam * lam * pt["E2"]) < 0.1 * abs(lam * lam * pt["E2"]), (
            f"lam={lam}: first-order residual {resid1:.3e} vs lam^2 E2 {lam**2 * pt['E2']:.3e}")
        V1 = first_order_covariance(L, MASS, lam)
        Vm = vi.covariance_from_mps(gs.psi)
        cov_dev[lam] = float(np.max(np.abs(Vm - V1)))
        cov_shift[lam] = float(np.max(np.abs(V1 - V0)))
        assert gs.info["max_trunc_err"] < 1e-12
    r_e = resid2[0.2] / resid2[0.1]
    r_c = cov_dev[0.2] / cov_dev[0.1]
    print(f"\nPT anchor: resid2/lam^3 = {resid2[0.1] / 0.1**3:.4f}, {resid2[0.2] / 0.2**3:.4f} "
          f"(ratio {r_e:.2f}, target ~8); covariance O(lam^2) residual ratio {r_c:.2f} (target ~4); "
          f"first order captures {1 - cov_dev[0.1] / cov_shift[0.1]:.3%} of the shift at lam=0.1")
    assert 6.0 <= r_e <= 10.0, f"second-order residual not O(lambda^3): ratio {r_e}"
    assert abs(resid2[0.1]) < 0.05 * 0.1**3 * L, f"second-order residual too large: {resid2[0.1]}"
    assert 2.5 <= r_c <= 5.5, f"covariance residual not O(lambda^2): ratio {r_c}"
    assert cov_dev[0.1] < 0.05 * cov_shift[0.1], (
        f"tadpole chain misses the O(lambda) covariance shift: dev {cov_dev[0.1]:.2e} vs shift {cov_shift[0.1]:.2e}")


def test_hartree_distance_from_criticality():
    """The interacting runs sit in the symmetric phase, far from lambda/mu_H^2 ~ 65."""
    mu2 = hartree_mass_squared(L_FREE, MASS, 1.0)
    assert mu2 > MASS**2, "the tadpole must raise the mass"
    ratio = 1.0 / mu2
    assert ratio < 0.1 * CRITICAL_RATIO_LAMBDA_OVER_MU2, f"lambda/mu_H^2 = {ratio} too close to critical"


# --------------------------------------------------------------------------
# Anchor 3: chi-doubling and truncation tracking; energy-density closure
# --------------------------------------------------------------------------


def test_chi_doubling_moves_nothing(lam1_ground_states):
    vi = _tenpy_modules()
    g16, g32 = lam1_ground_states
    c = L_FREE // 2
    assert g32.info["chi_max"] == 2 * g16.info["chi_max"]
    for g in (g16, g32):
        assert "max_trunc_err" in g.info and g.info["max_trunc_err"] < 1e-8
    numbers = {"E0": (g16.E0, g32.E0)}
    S16, S32 = vi.block_entropies(g16.psi), vi.block_entropies(g32.psi)
    numbers["S_half"] = (S16[c - 1], S32[c - 1])
    for name, (A, B) in _blocks(c).items():
        fn = vi.mutual_information_mps if name.startswith("MI") else vi.log_negativity_mps
        numbers[name] = (fn(g16.psi, A, B), fn(g32.psi, A, B))
    worst = max(vi.relative_change(a, b) for a, b in numbers.values())
    print("\nchi-doubling (16 -> 32) relative changes: "
          + ", ".join(f"{k}={vi.relative_change(a, b):.1e}" for k, (a, b) in numbers.items()))
    assert worst < 1e-2, f"chi-doubling moved a reported number by {worst:.2e} >= 1%"


def test_energy_density_sums_to_hamiltonian(lam1_ground_states):
    vi = _tenpy_modules()
    g, _ = lam1_ground_states
    prof = vi.energy_density_profile(g.psi, MASS, 1.0)
    E = vi.mps_energy(g.psi, g.model)
    assert abs(prof.sum() - E) < 1e-10, f"sum h_i = {prof.sum()} != <H> = {E}"
    assert abs(vi.local_energy_density(g.psi, L_FREE // 2, MASS, 1.0) - prof[L_FREE // 2]) < 1e-12


# --------------------------------------------------------------------------
# Anchor 4: passivity of the interacting ground state
# --------------------------------------------------------------------------


def test_passivity_audit_interacting_ground_state(lam1_ground_states):
    vi = _tenpy_modules()
    _, g = lam1_ground_states
    worst = 0.0
    for strength in (0.05, 0.3, 1.0):
        aud = vi.passivity_audit_mps(g.psi, g.model, n_samples=8, max_sites=2,
                                     strength=strength, seed=int(100 * strength), tol=1e-8)
        assert aud["passed"], f"passivity violated at strength {strength}: worst dE = {aud['worst']:.3e}"
        worst = min(worst, aud["worst"])
    print(f"\npassivity: worst energy change over 24 random local unitaries = {worst:+.3e} (must be >= -1e-8)")
    # the audit must be able to fire: an active state with the inverse squeeze in the sampler
    from vacuum.interacting.qei_mps import _squeezer_ops
    fam = vi.squeeze_family(L_FREE // 2, 0, L_FREE)
    active = vi.apply_squeeze_family(g.psi, fam, [0.3])
    d = g.psi.sites[0].dim

    def sampler(rng, k):
        if k == 0:
            return L_FREE // 2, 1, _squeezer_ops(g.psi.sites[L_FREE // 2], r_single=-0.3).to_ndarray()
        return int(rng.integers(0, L_FREE)), 1, vi.random_local_unitary(rng, d, 1, 0.3)

    canary = vi.passivity_audit_mps(active, g.model, n_samples=3, sampler=sampler)
    assert not canary["passed"] and canary["worst"] < -1e-3, "audit failed to fire on an active state"


# --------------------------------------------------------------------------
# QEI pipeline anchor (lambda = 0): TEBD smeared energy == Gaussian smeared energy
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qei_setup():
    vi = _tenpy_modules()
    from vacuum.inequalities.qei import gaussian_f, qei_minimize, sampling_operator
    L, tau0 = 12, 0.6   # light cone (speed <= 1) from the window stays clear of the walls for t <= 4 tau0
    c = L // 2
    f = gaussian_f(tau0, n_sigmas=4.0)
    op = sampling_operator(L, c, f, m=MASS, bc="dirichlet")
    K = free_coupling_matrix(L, MASS)
    fam = vi.squeeze_family(c, 1, L)
    theta, E_fam, _ = vi.optimize_free_family(fam, K, op)
    gs = vi.phi4_ground_mps(L, MASS, 0.0, n_max=8, chi_max=16)
    return {"L": L, "c": c, "f": f, "op": op, "K": K, "fam": fam, "theta": theta,
            "E_fam": E_fam, "E_exact": qei_minimize(op).e_min, "gs": gs}


def test_qei_free_anchor_tebd_matches_gaussian(qei_setup):
    vi = _tenpy_modules()
    s = qei_setup
    V0 = ground_state_cov(s["K"])
    E_gauss = vi.free_family_energy(s["fam"], s["theta"], V0, s["op"])
    assert E_gauss < 0.0 and E_gauss >= s["E_exact"] - 1e-12, (
        f"family energy {E_gauss} must be negative and above the exact infimum {s['E_exact']}")
    psi_s = vi.apply_squeeze_family(s["gs"].psi, s["fam"], s["theta"])
    out = vi.smeared_energy_mps(psi_s, s["gs"].model, s["c"], s["f"], MASS, 0.0, dt=0.2, order=4,
                                chi_max=16, psi_ref=s["gs"].psi, reference="evolved")
    dev = out["E_f"] - E_gauss
    print(f"\nQEI anchor: TEBD E_f = {out['E_f']:.6e} vs Gaussian {E_gauss:.6e} (dev {dev:+.2e}); "
          f"exact free infimum {s['E_exact']:.6e}; family reaches {E_gauss / s['E_exact']:.3f} of it; "
          f"TEBD trunc {out['trunc_err']:.1e}, chi {out['chi']}, {out['times'].size - 1} steps")
    assert abs(dev) < 1e-5, f"TEBD smeared energy off the Gaussian value by {dev:.2e}"
    assert abs(dev) < 1e-3 * abs(E_gauss), "relative deviation above 0.1%"
    assert out["trunc_err"] < 1e-6
    # the reference trace of an eigenstate must not drift
    assert np.max(np.abs(out["h_ref"] - out["h_ref"][0])) < 1e-7


def test_qei_variational_result_structure(qei_setup):
    """qei_variational_mps at lambda = 0: the reported number is an evaluated
    state's energy, sits at or below the family optimum (s = 1 is on the grid),
    and carries chi / n_max / dt / truncation."""
    vi = _tenpy_modules()
    s = qei_setup
    res = vi.qei_variational_mps(s["gs"], s["c"], s["f"], half_width=1, s_grid=(0.9, 1.0, 1.1),
                                 dt=0.2, order=4, chi_max=16, shape_mass="bare", refine=False)
    assert res.E_var == float(np.min(res.E_grid))
    assert res.s_star in set(res.s_grid.tolist())
    assert res.E_var <= s["E_fam"] + 1e-5, f"E_var {res.E_var} above the free family optimum {s['E_fam']}"
    assert res.E_var >= res.E_free_exact - 1e-4, "variational number below the exact free infimum at lambda = 0"
    assert abs(res.E_free_exact - s["E_exact"]) < 1e-12
    assert abs(res.mu_H2 - MASS**2) < 1e-12 and abs(res.E_hartree_exact - res.E_free_exact) < 1e-12
    # bookkeeping: chi / n_max / dt carried; TEBD truncation accumulated over
    # the reference + grid trajectories at chi = 16 is ~1e-6 (tracked, gate 1e-5)
    assert res.meta["n_max"] == 8 and res.chi <= 16 and res.dt == 0.2 and res.trunc_err < 1e-5
    print(f"\nvariational (lambda=0): E_var = {res.E_var:.6e} at s = {res.s_star}, grid {res.E_grid}, "
          f"free family optimum {s['E_fam']:.6e}, exact {res.E_free_exact:.6e}, 2d bound {res.bound_2d:.4e}")
