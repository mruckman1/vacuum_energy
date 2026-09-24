"""Layer 7 / L5 follow-up: the sharpest free reference and the decomposition
of the extremal state (``vacuum.interacting.qei_reference``).

Anchors:

1. **K_eff's local density closes and is PSD.**  The range-r bond split of
   ``keff_density_terms`` sums to diag(K_eff, I) exactly for a multi-range
   coupling (interior bonds, wall images, self-image bonds), every h_x is
   PSD, and at nearest-neighbour range it *is* the density of
   ``vacuum.inequalities.qei`` on ``harmonic_chain_K``.
2. **The K_eff Williamson infimum reproduces the archived free numbers.**
   With mu^2 = m^2, J = (1,) it equals ``free_exact_infimum`` to machine
   precision at the paper point, and with the Hartree mass it equals the
   Hartree reference — so any difference between E_free(K_eff) and E_H at
   lambda > 0 comes from omega(k), not from the pipeline.
3. **The dispersion pipeline is calibrated at lambda = 0.**  The TEBD
   correlator + harmonic inversion returns the exact free dispersion on
   every standing wave, with unit spectral weight per mode, and its lowest
   frequency equals the excited-state DMRG gap.
4. **The four pieces sum to the infimum** (the key regression).  On the
   extremal state of a lambda > 0 chain the Heisenberg-picture pieces sum to
   E_min up to the operator truncation, the Schroedinger-picture pieces sum
   to the Schroedinger re-measurement, and the two pictures agree piece by
   piece; at lambda = 0 the pieces agree with the Gaussian pieces of the
   Williamson extremal state.

Sizes keep the file near ~30 s uncontended (L <= 6, n_max <= 4, tau0 <= 0.25).
TeNPy is importorskip'd; the numpy-only anchors run without it.
"""

import logging
import math
import warnings

import numpy as np
import pytest

from vacuum.core import harmonic_chain_K
from vacuum.inequalities.qei import (
    _assemble_pullback,
    _terms_to_dense,
    chain_energy_density_form,
    gaussian_f,
    qei_minimize,
    sampling_operator,
)
from vacuum.interacting.qei_exact import free_exact_infimum, local_energy_operator_dense
from vacuum.interacting import qei_reference as qr

MASS = 0.5


def _vi():
    pytest.importorskip("tenpy")
    logging.getLogger("tenpy").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="unit_cell_width")
    warnings.filterwarnings("ignore", message="unused options")
    import vacuum.interacting as vi
    return vi


# --------------------------------------------------------------------------
# 1. K_eff and its PSD local density (numpy)
# --------------------------------------------------------------------------


def test_keff_density_closes_and_is_psd():
    L, mu2, J = 9, 0.3, np.array([1.0, 0.2, 0.05, 0.01])
    K = qr.keff_from_couplings(L, mu2, J)
    assert np.min(np.linalg.eigvalsh(K)) > 0.0
    # the sine modes diagonalize it with the prescribed cosine series
    k, _ = qr.standing_wave_modes(L)
    w2 = mu2 + sum(Jr * (2 - 2 * np.cos(r * k)) for r, Jr in enumerate(J, start=1))
    assert np.allclose(qr.free_dispersion(K), w2, atol=1e-12)
    total = np.zeros((2 * L, 2 * L))
    for x in range(L):
        h = _terms_to_dense(L, qr.keff_density_terms(L, x, mu2, J))
        assert np.min(np.linalg.eigvalsh(h)) > -1e-12, f"h_{x} is not PSD"
        total += h
    target = np.zeros((2 * L, 2 * L))
    target[:L, :L] = K
    target[L:, L:] = np.eye(L)
    assert np.max(np.abs(total - target)) < 1e-12, "sum_x h_x != diag(K_eff, I)"
    # nearest-neighbour range: the archived density and coupling matrix exactly
    assert np.allclose(qr.keff_from_couplings(L, MASS ** 2, [1.0]), harmonic_chain_K(L, MASS, "dirichlet"))
    for x in (0, 4, L - 1):
        assert np.allclose(_terms_to_dense(L, qr.keff_density_terms(L, x, MASS ** 2, [1.0])),
                           chain_energy_density_form(L, x, MASS, "dirichlet"))


def test_free_infimum_keff_reproduces_the_archived_free_numbers():
    L, x = 16, 8
    f = gaussian_f(0.75, n_sigmas=4.0)
    E_free = free_exact_infimum(L, x, f, MASS)
    res, op = qr.free_infimum_keff(L, x, f, MASS ** 2, [1.0])
    assert abs(res.e_min - E_free) < 1e-12 * abs(E_free)
    mu_H2 = 0.9087036615929707   # the archived lambda = 4 Hartree mass (s5b)
    E_H = free_exact_infimum(L, x, f, math.sqrt(mu_H2))
    res_H, _ = qr.free_infimum_keff(L, x, f, mu_H2, [1.0])
    assert abs(res_H.e_min - E_H) < 1e-12 * abs(E_H)
    # a longer-range coupling changes the number (the reference is not the mass alone)
    res_2, _ = qr.free_infimum_keff(L, x, f, mu_H2, [1.0, 0.05])
    assert abs(res_2.e_min - E_H) > 1e-4 * abs(E_H)
    assert res_2.gap < 1e-6 * abs(res_2.e_min)


def test_energy_pieces_sum_to_the_local_density():
    for (x, L) in ((4, 9), (0, 5), (4, 5)):
        sites, pcs = qr.energy_pieces_dense(4, 1.1, MASS, 4.0, x, L)
        sites2, H3 = local_energy_operator_dense(4, 1.1, MASS, 4.0, x, L)
        assert sites == sites2
        assert np.max(np.abs(sum(pcs.values()) - H3)) < 1e-12
        for name, H in pcs.items():
            assert np.min(np.linalg.eigvalsh(H)) > -1e-12, f"{name} piece is not PSD"


def test_harmonic_inversion_recovers_the_poles():
    dt = 0.1
    t = dt * np.arange(80)
    om = np.array([0.7, 1.3, 2.1])
    A = np.array([1.0, 0.3, 0.05])
    s = sum(a * np.exp(-1j * w * t) for a, w in zip(A, om))
    hi = qr.harmonic_inversion(s, dt, sv_tol=1e-8)
    assert hi["n_poles"] == 3
    order = np.argsort(hi["omega"])
    assert np.allclose(hi["omega"][order], om, atol=1e-8)
    assert np.allclose(np.abs(hi["A"][order]), A, atol=1e-8)
    assert np.max(np.abs(hi["gamma"])) < 1e-8 and hi["resid"] < 1e-8
    # the frequency-domain width of a single harmonic is the window's own
    om0, fw, fw0 = qr.peak_width(np.exp(-1j * 0.9 * t), t)
    assert abs(om0 - 0.9) < 2e-3 and abs(fw / fw0 - 1.0) < 1e-2


# --------------------------------------------------------------------------
# 2. The dispersion pipeline at lambda = 0 (tenpy)
# --------------------------------------------------------------------------


def test_tebd_dispersion_reproduces_the_free_chain_and_the_dmrg_gap():
    vi = _vi()
    gs = vi.phi4_ground_mps(6, MASS, 0.0, n_max=4, chi_max=16)
    d = qr.interacting_dispersion_tebd(gs, T=5.0, dt=0.1, chi_max=16)
    dev = np.max(np.abs(d.omega - d.omega_free))
    print(f"\nlambda = 0 dispersion: max |omega - omega_free| = {dev:.2e}, weights {np.min(d.weight):.4f}, "
          f"gamma max {np.max(np.abs(d.gamma)):.1e}, resid max {np.max(d.resid):.1e}")
    assert dev < 2e-3, "the TEBD correlator does not reproduce the free dispersion"
    assert np.min(d.weight) > 0.999, "a free mode must carry all of its projected weight"
    assert np.max(np.abs(d.gamma)) < 1e-3
    assert np.max(d.fwhm / d.fwhm_window) < 1.02
    # t = 0 is the equal-time correlator
    assert np.allclose(d.C[:, 0].real, gs.psi.correlation_function("X", "X")[:, 3], atol=1e-10)
    gap, info = qr.excitation_gap_dmrg(gs, chi_max=16)
    assert info["overlap_with_ground"] < 1e-6
    assert abs(gap - d.omega[0]) < 2e-3, f"DMRG gap {gap} vs omega_1 {d.omega[0]}"
    mu2, J, fit = qr.fit_dispersion_couplings(d.k, d.omega ** 2, r_max=2)
    assert abs(mu2 - MASS ** 2) < 5e-3 and abs(J[0] - 1.0) < 5e-3 and fit["misfit_max"] < 2e-3


# --------------------------------------------------------------------------
# 3. The decomposition on the extremal state (tenpy; the key regression)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extremal():
    vi = _vi()
    from vacuum.interacting.qei_exact import qei_exact_mps
    L, x, tau0 = 6, 3, 0.25
    f = gaussian_f(tau0, n_sigmas=4.0)
    # order-2 Trotter and chi_op = 12 keep the fixture cheap; the two pictures
    # share the step, so the piece-by-piece comparison is like for like
    kw = dict(dt=0.125, order=2, chi_op=12, svd_min_op=1e-12, chi_acc=24, svd_min_acc=1e-12,
              window_pad=2, svd_min_real=1e-8, chi_mpo=32, weight_decay=0.25)
    out = {}
    for lam in (0.0, 1.0):
        gs = vi.phi4_ground_mps(L, MASS, lam, n_max=4, chi_max=16)
        res = qei_exact_mps(gs, x, f, **kw, chi_dmrg=8, max_sweeps=8, max_E_err=1e-10)
        dec = qr.decompose_smeared_energy(res, gs, f, **kw)
        sch = qr.schroedinger_pieces(res.psi_star, gs.model, x, f, gs.psi, dt=0.125, order=2, chi_max=16)
        out[lam] = {"gs": gs, "res": res, "dec": dec, "sch": sch}
    return {"L": L, "x": x, "f": f, "runs": out}


def test_pieces_sum_to_the_infimum(extremal):
    """The free audit: E_kin + E_grad + E_mass + E_quartic = E_min."""
    for lam, run in extremal["runs"].items():
        res, dec, sch = run["res"], run["dec"], run["sch"]
        E = abs(res.E_min)
        print(f"\nlambda = {lam}: E_min {res.E_min:.6e}; Heisenberg pieces "
              f"{ {k: round(v, 6) for k, v in dec['pieces'].items()} } sum {dec['sum']:.6e} "
              f"(audit {dec['audit_rel']:+.2e}); Schroedinger sum {sch['sum']:.6e}")
        assert abs(dec["audit"]) < 5e-3 * E, f"pieces do not sum to the infimum: {dec['audit_rel']:+.2e}"
        assert abs(sch["sum"] - res.E_min) < 1e-2 * E
        for name in qr.PIECES:
            assert abs(dec["pieces"][name] - sch[name]) < 5e-3 * E, name
        # the two negative-energy carriers are the kinetic and gradient pieces;
        # the extremal state is pi-squeezed at the sampling point
        assert dec["pieces"]["kinetic"] < 0.0 and dec["pieces"]["gradient"] < 0.0
        assert dec["pi2_star"] < dec["pi2_vac"]
        if lam == 0.0:
            assert dec["pieces"]["quartic"] == 0.0 and dec["trunc_op"]["quartic"] == 0.0
        else:
            assert dec["pieces"]["quartic"] != 0.0


def test_pieces_at_lambda_zero_match_the_gaussian_pieces(extremal):
    """At lambda = 0 every piece is a quadratic form, so on the extremal MPS it
    must equal (1/2) Tr(O_a (V_psi* - V_Omega)) with O_a the pulled-back piece
    of ``vacuum.inequalities.qei`` and V the state's own covariance.

    The comparison is made on the MPS state's covariance, not on the Williamson
    V*: the pieces are first-order sensitive to the state where the total is
    only second-order (the total is stationary at the minimum), so the Fock
    cutoff redistributes weight between kinetic and gradient pieces at the
    percent level while the sum moves far less -- that sensitivity is itself
    a finding (see the draft), and this anchor isolates the operator pipeline
    from it."""
    from vacuum.interacting import covariance_from_mps

    L, x, f = extremal["L"], extremal["x"], extremal["f"]
    run = extremal["runs"][0.0]
    K = harmonic_chain_K(L, MASS, "dirichlet")
    ei, ev = np.array([], dtype=int), np.array([])
    terms = {
        "kinetic": [(1.0, ei, ev, np.array([x]), np.array([1.0]))],
        "mass": [(MASS ** 2, np.array([x]), np.array([1.0]), ei, ev)],
        "gradient": [(0.5, np.array([x - 1, x]), np.array([-1.0, 1.0]), ei, ev),
                     (0.5, np.array([x, x + 1]), np.array([-1.0, 1.0]), ei, ev)],
    }
    ops = {}
    for name, tt in terms.items():
        O, _, _, _, _, _ = _assemble_pullback(K, tt, f.f, f.t_lo, f.t_hi, np.arange(L), None, 10,
                                              weight_floor=0.0)
        ops[name] = O
    full = sampling_operator(L, x, f, m=MASS, bc="dirichlet", weight_floor=0.0)
    assert np.allclose(sum(ops.values()), full.O, atol=1e-10)
    V = covariance_from_mps(run["res"].psi_star)
    V0 = covariance_from_mps(run["gs"].psi)
    gauss = {name: 0.5 * float(np.einsum("ij,ji->", O, V - V0)) for name, O in ops.items()}
    E = abs(run["res"].E_min)
    print(f"\nGaussian pieces on the MPS covariance { {k: round(v, 6) for k, v in gauss.items()} } "
          f"sum {sum(gauss.values()):.6e} vs MPO pieces { {k: round(v, 6) for k, v in run['dec']['pieces'].items()} }")
    for name, val in gauss.items():
        dev = abs(run["dec"]["pieces"][name] - val)
        assert dev < 2.5e-2 * E, f"{name}: MPO {run['dec']['pieces'][name]:.4e} vs Gaussian {val:.4e}"
    # and the Williamson infimum itself is what the truncated chain approaches from above
    assert qei_minimize(full).e_min < run["res"].E_min < 0.0
