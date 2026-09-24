"""QET on TFIM chains via MPS/DMRG — the ``.[tensor]`` extra rung of M3.2.

Same protocol as :mod:`vacuum.qet.chains` (measure sigma^x at A with Kraus
projectors + renormalize, classical bit, conditional single-site rotation at
B optimized per outcome; spin-chain QET after M. Hotta, J. Phys. Soc. Jpn.
78, 034001 (2009) [arXiv:0803.0348]), implemented as MPS operations:
project, canonicalize, track truncation error, and gate every claim on
bond-dimension convergence (chi-doubling moves the dip < 1%). ED validates
DMRG at L = 14 before any large-L number is admissible (spec
docs/PLAN_LAYERS_2_3.md, M3.2; tests/test_qet_chains.py).

TeNPy (validated here against tenpy 1.1.1) is imported lazily inside
:func:`_tenpy`, so importing this module never requires the extra; the
functions raise a guided ImportError instead. The model is
``tenpy.models.tf_ising.TFIChain`` whose documented convention

    H = -J sum_i sigma^x_i sigma^x_{i+1} - g sum_i sigma^z_i

matches ``chains.tfim_hamiltonian`` exactly (the L = 14 ED<->DMRG anchor is
the arbiter of any convention slip). ``conserve=None`` throughout: Alice's
sigma^x measurement flips the Z2 parity, so a parity-conserving MPS could
not represent the branches.

Energy densities use the same symmetric bond split as the ED module and are
always reported in difference form <h_i> - <h_i>_GS.

Functions are pure over their MPS arguments: every operation copies the
input state before acting (TeNPy MPS methods mutate in place).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = [
    "tfim_ground_mps",
    "mps_energy_profile",
    "mps_measure_site_x",
    "mps_apply_rotation",
    "mps_local_energy",
    "mps_optimize_rotation",
    "mps_rotation_optimum_closed",
    "run_chain_qet_mps",
]


def _tenpy():
    """Lazy, guarded TeNPy import (the ``.[tensor]`` extra)."""
    try:
        import tenpy.linalg.np_conserved as npc
        from tenpy.algorithms import dmrg
        from tenpy.models.tf_ising import TFIChain
        from tenpy.networks.mps import MPS
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise ImportError(
            "vacuum.qet.dmrg needs TeNPy — install the '.[tensor]' extra "
            "(pip install physics-tenpy); the ED path in vacuum.qet.chains "
            "is the core-only fallback"
        ) from exc
    return SimpleNamespace(npc=npc, dmrg=dmrg, TFIChain=TFIChain, MPS=MPS)


def _check_gJ(g, J):
    g = float(g)
    J = float(J)
    if not (g > 0.0 and J > 0.0):
        raise ValueError(f"need g > 0 and J > 0, got g={g}, J={J}")
    return g, J


def tfim_ground_mps(L, g, J=1.0, chi_max=64, max_sweeps=24, min_sweeps=4,
                    max_E_err=1e-13, svd_min=1e-13, mixer=False, psi0=None,
                    lanczos_tight=True):
    """DMRG ground state of the open TFIM chain: (E0, psi, info).

    Parameters
    ----------
    L, g, J : chain size and couplings (matching chains.tfim_hamiltonian).
    chi_max : bond-dimension cap; chi_max >= 2**(L//2) makes the MPS exact.
    max_sweeps, min_sweeps, max_E_err : TwoSiteDMRGEngine sweep control
        (stop when the per-sweep energy change drops below max_E_err).
    svd_min : discard singular values below this in truncation.
    mixer : enable the density-matrix mixer (recommended for large L).
    psi0 : optional MPS to warm-start from (copied, never mutated) — used
        for chi-doubling continuations.
    lanczos_tight : push the site-eigenproblem Lanczos tolerances to
        machine precision (needed for the 1e-8 ED<->DMRG dip validation).

    Returns
    -------
    (E0, psi, info): ground energy, the optimized finite MPS (canonical,
    conserve=None), and a convergence record with keys 'sweeps', 'delta_E'
    (last per-sweep energy change, None if unavailable), 'max_trunc_err',
    'chi_max', 'converged'.
    """
    t = _tenpy()
    L = int(L)
    g, J = _check_gJ(g, J)
    model = t.TFIChain({"L": L, "J": J, "g": g,
                        "bc_MPS": "finite", "conserve": None})
    if psi0 is not None:
        psi = psi0.copy()
    else:
        psi = t.MPS.from_product_state(model.lat.mps_sites(), ["up"] * L,
                                       bc="finite")
    options = {
        "trunc_params": {"chi_max": int(chi_max), "svd_min": float(svd_min)},
        "max_E_err": float(max_E_err),
        "max_S_err": 1e-8,
        "max_sweeps": int(max_sweeps),
        "min_sweeps": int(min_sweeps),
        "mixer": bool(mixer),
    }
    if lanczos_tight:
        options["lanczos_params"] = {"N_max": 60, "P_tol": 1e-16, "E_tol": 1e-16}
    eng = t.dmrg.TwoSiteDMRGEngine(psi, model, options)
    E0, psi = eng.run()

    # convergence record from the engine's sweep statistics (tenpy 1.1.1)
    delta_E = None
    max_trunc = 0.0
    stats = getattr(eng, "sweep_stats", None) or {}
    if stats.get("Delta_E"):
        delta_E = float(abs(stats["Delta_E"][-1]))
    elif len(stats.get("E", [])) >= 2:
        delta_E = float(abs(stats["E"][-1] - stats["E"][-2]))
    if stats.get("max_trunc_err"):
        max_trunc = float(np.max(stats["max_trunc_err"]))
    info = {
        "sweeps": int(getattr(eng, "sweeps", 0)),
        "delta_E": delta_E,
        "max_trunc_err": max_trunc,
        "chi_max": int(chi_max),
        "converged": (delta_E is not None and delta_E <= float(max_E_err)),
    }
    return float(E0), psi, info


def _site_op(psi, i, name):
    return psi.sites[i].get_op(name)


def _projector_op(psi, site, mu):
    """(Id + mu Sigmax)/2 as an npc array on the given site."""
    return (_site_op(psi, site, "Id") + float(mu) * _site_op(psi, site, "Sigmax")) * 0.5


def _rotation_op(psi, site, mu, theta):
    """U_B(mu, theta) = cos(theta) Id + mu sin(theta) (Sigmax Sigmaz).

    Sigmax Sigmaz = [[0, -1], [1, 0]] = -i sigma^y (real), so this is the
    same conditional rotation as chains.bob_rotation_gate.
    """
    t = _tenpy()
    misy = t.npc.tensordot(_site_op(psi, site, "Sigmax"),
                           _site_op(psi, site, "Sigmaz"),
                           axes=[["p*"], ["p"]])
    return (_site_op(psi, site, "Id") * np.cos(float(theta))
            + misy * (float(mu) * np.sin(float(theta))))


def mps_energy_profile(psi, g, J=1.0):
    """<h_i> with the symmetric bond split, as an (L,) float array.

    h_i = -g <sz_i> - (J/2)(<sx_{i-1} sx_i> + <sx_i sx_{i+1}>), matching
    chains.tfim_local_energy_ops; sums to <H> exactly.
    """
    g, J = _check_gJ(g, J)
    L = psi.L
    sz = np.real(np.asarray(psi.expectation_value("Sigmaz")))
    xx = np.array([
        float(np.real(psi.expectation_value_multi_sites(["Sigmax", "Sigmax"], i)))
        for i in range(L - 1)
    ])
    prof = -g * sz
    prof[:-1] += -(J / 2.0) * xx
    prof[1:] += -(J / 2.0) * xx
    return prof


def mps_measure_site_x(psi, site):
    """Projective sigma^x measurement at `site` on an MPS: Kraus + renormalize.

    p(mu) = (1 + mu <Sigmax_site>)/2 from the canonical input state; each
    branch applies the projector (non-unitary single-site op), then restores
    canonical form with renormalization — the MPS transcription of
    chains.measure_site_x. Returns {mu: {'p': p, 'psi': branch MPS}}.
    """
    site = int(site)
    ex = float(np.real(psi.expectation_value("Sigmax", sites=[site])[0]))
    out = {}
    for mu in (+1, -1):
        p = 0.5 * (1.0 + mu * ex)
        if p <= 0.0:
            raise ValueError(f"outcome mu={mu} has zero probability")
        branch = psi.copy()
        branch.apply_local_op(site, _projector_op(branch, site, mu),
                              unitary=False, renormalize=True)
        branch.canonical_form()   # idempotent safety: projection broke canonicity
        out[mu] = {"p": p, "psi": branch}
    return out


def mps_apply_rotation(psi, site, mu, theta):
    """Copy of `psi` with U_B(mu, theta) applied at `site` (unitary, exact)."""
    rotated = psi.copy()
    rotated.apply_local_op(int(site), _rotation_op(rotated, int(site), mu, theta),
                           unitary=True)
    return rotated


def mps_local_energy(psi, B, g, J=1.0):
    """The <H>-terms touched by a single-site operation at B.

    -g <sz_B> - J <sx_{B-1} sx_B> - J <sx_B sx_{B+1}> (existing bonds only):
    differences of this under Bob's rotation equal differences of the full
    <H>, with far better roundoff behavior.
    """
    g, J = _check_gJ(g, J)
    B = int(B)
    L = psi.L
    e = -g * float(np.real(psi.expectation_value("Sigmaz", sites=[B])[0]))
    for i in (B - 1, B):
        if 0 <= i < L - 1:
            e += -J * float(np.real(
                psi.expectation_value_multi_sites(["Sigmax", "Sigmax"], i)))
    return e


def mps_optimize_rotation(psi_mu, B, g, J=1.0, mu=+1, xatol=1e-10):
    """Numerically optimal (theta, extracted energy) for one MPS branch.

    Same bounded scalar maximization over theta in (-pi/2, pi/2) as
    chains.optimize_rotation, with MPS energy evaluations.
    """
    e_before = mps_local_energy(psi_mu, B, g, J)

    def neg(theta):
        rotated = mps_apply_rotation(psi_mu, B, mu, theta)
        return -(e_before - mps_local_energy(rotated, B, g, J))

    res = minimize_scalar(
        neg, bounds=(-np.pi / 2.0, np.pi / 2.0), method="bounded",
        options={"xatol": float(xatol)},
    )
    return float(res.x), -float(res.fun)


def mps_rotation_optimum_closed(psi_mu, B, g, J=1.0, mu=+1):
    """Closed-form (theta*, dE*) for one MPS branch — the MPS mirror of
    :func:`vacuum.qet.chains.rotation_optimum_closed`.

    U_B(mu, theta) is a rotation by 2 theta about y at site B, so the local
    energy it can shed is dE(theta) = b (1 - cos 2theta) + c sin 2theta.
    Two evaluations at theta = +-pi/4 fix (b, c); the maximum is
    b + sqrt(b^2 + c^2) at 2 theta* = pi - atan2(c, b), folded into
    (-pi/2, pi/2]. Two MPS energy evaluations instead of the ~30 the bounded
    scalar search costs, and — being exact rather than tolerance-limited —
    this is the path that makes the ED<->DMRG dip comparison measure the
    tensor network rather than two independent optimizers' xatol.
    """
    e_before = mps_local_energy(psi_mu, B, g, J)

    def dE(theta):
        rotated = mps_apply_rotation(psi_mu, B, mu, theta)
        return e_before - mps_local_energy(rotated, B, g, J)

    f_plus, f_minus = dE(np.pi / 4.0), dE(-np.pi / 4.0)
    b = 0.5 * (f_plus + f_minus)
    c = 0.5 * (f_plus - f_minus)
    theta_star = 0.5 * (np.pi - np.arctan2(c, b))
    if theta_star > np.pi / 2.0:
        theta_star -= np.pi
    return float(theta_star), float(b + np.hypot(b, c))


def run_chain_qet_mps(L, g, A, B, J=1.0, chi_max=64, ground=None,
                      dmrg_options=None, xatol=1e-10, theta_method="numeric"):
    """The chain-QET protocol on MPS; ledger mirrors chains.run_chain_qet.

    Parameters
    ----------
    L, g, J, A, B, xatol : as in chains.run_chain_qet.
    chi_max : bond-dimension cap for a freshly built ground state.
    ground : optional precomputed (E0, psi, info) from tfim_ground_mps —
        pass it to reuse one ground state across many (A, B) runs or to
        warm-start chi-doubling studies.
    dmrg_options : extra keyword arguments for tfim_ground_mps.
    theta_method : 'numeric' (default, the spec's per-outcome numerical
        optimization, via :func:`mps_optimize_rotation`) or 'closed'
        (:func:`mps_rotation_optimum_closed`, exact and ~15x cheaper).
        The two agree to machine precision — that agreement is itself a
        tested anchor.

    Returns
    -------
    dict with the chains.run_chain_qet keys ('E_gs', 'profile_gs', 'E_A',
    'E_B', 'ratio', 'dip_meas', 'dip_final', 'outcomes' — the per-outcome
    states are MPS objects under key 'psi_meas'/'psi_final') plus
    'convergence': the tfim_ground_mps info record.
    """
    L = int(L)
    g, J = _check_gJ(g, J)
    A = int(A)
    B = int(B)
    if A == B:
        raise ValueError("Alice and Bob must sit on different sites")
    if theta_method not in ("numeric", "closed"):
        raise ValueError(f"theta_method must be 'numeric' or 'closed', "
                         f"got {theta_method!r}")
    if ground is None:
        ground = tfim_ground_mps(L, g, J, chi_max=chi_max,
                                 **(dmrg_options or {}))
    E0, psi_gs, info = ground

    profile_gs = mps_energy_profile(psi_gs, g, J)
    branches = mps_measure_site_x(psi_gs, A)

    profile_meas = np.zeros(L)
    profile_final = np.zeros(L)
    outcomes = {}
    E_B = 0.0
    for mu, br in branches.items():
        p, psi_mu = br["p"], br["psi"]
        if theta_method == "closed":
            theta, dE = mps_rotation_optimum_closed(psi_mu, B, g, J, mu=mu)
        else:
            theta, dE = mps_optimize_rotation(psi_mu, B, g, J, mu=mu,
                                              xatol=xatol)
        psi_final = mps_apply_rotation(psi_mu, B, mu, theta)
        profile_meas = profile_meas + p * mps_energy_profile(psi_mu, g, J)
        profile_final = profile_final + p * mps_energy_profile(psi_final, g, J)
        E_B += p * dE
        outcomes[mu] = {
            "p": p,
            "theta": theta,
            "dE": dE,
            "psi_meas": psi_mu,
            "psi_final": psi_final,
        }

    dip_meas = profile_meas - profile_gs    # difference form, always
    dip_final = profile_final - profile_gs
    E_A = float(np.sum(dip_meas))
    return {
        "L": L, "g": g, "J": J, "A": A, "B": B,
        "E_gs": float(E0),
        "profile_gs": profile_gs,
        "E_A": E_A,
        "E_B": float(E_B),
        "ratio": float(E_B) / E_A,
        "dip_meas": dip_meas,
        "dip_final": dip_final,
        "outcomes": outcomes,
        "convergence": info,
    }
