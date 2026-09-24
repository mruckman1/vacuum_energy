"""The sharpest free reference for the interacting QEI infimum (Layer 7, L5 follow-up).

Why a second reference
----------------------
``qei_exact`` compares the exact lattice infimum E_min(lambda) of the smeared
energy with the free chain's Williamson infimum at the self-consistent
Hartree mass, E_H.  That reference matches the interacting theory at one
point of the Brillouin zone (the gap) and nowhere else.  Interactions do not
only shift the gap: they renormalize the whole single-particle dispersion
omega(k), and the free QEI infimum depends on the full omega(k), so the
Hartree line is not the sharpest quadratic reference.  This module builds
the sharpest one and decomposes the extremal state against it.

1.  **The interacting dispersion** (:func:`interacting_dispersion_tebd`).
    The retarded-like correlator C(j, t) = <Omega| phi_j(t) phi_x(0) |Omega>
    is obtained by TEBD of phi_x|Omega> under the phi^4 Hamiltonian and
    projected on the standing waves of the Dirichlet chain,
    v_n(j) = sqrt(2/(L+1)) sin(k_n (j+1)), k_n = n pi/(L+1) (the exact
    normal modes of the free chain, so at lambda = 0 every projected signal
    is a single harmonic v_n(x) e^{-i omega_n t}/(2 omega_n)).  Each
    projected signal is harmonically inverted by the matrix-pencil method
    (:func:`harmonic_inversion`), which returns complex poles
    z_p = exp(-i (omega_p - i gamma_p) dt) with amplitudes A_p: the
    quasiparticle is the dominant pole, its **weight fraction**
    |A_dom| / sum |A_p| is the fraction of the projected spectral weight it
    carries, and gamma_dom is its decay rate.  On a finite chain the spectrum
    is discrete, so "the one-particle peak is sharp at k" means: one pole
    carries the weight (fraction -> 1) and does not decay over the trajectory
    (gamma T << 1); the windowed Fourier transform's FWHM relative to the
    window's own response (:func:`peak_width`) says the same thing in the
    frequency domain.  The lowest frequency is cross-checked against an
    independent excited-state DMRG (:func:`excitation_gap_dmrg`; the model
    is shifted so both energies are negative, TeNPy issue 329).  Resolution:
    the lambda = 0 run against the exact free dispersion measures the
    combined Trotter/truncation/Fock error of the whole pipeline.

2.  **K_eff** (:func:`fit_dispersion_couplings`, :func:`keff_from_couplings`).
    The inverse (sine) Fourier transform of omega(k)^2 on the chain's own
    lattice is the coupling matrix K_eff = sum_n omega_n^2 v_n v_n^T.  Written
    as the cosine series omega^2(k) = mu^2 + sum_r J_r (2 - 2 cos r k) it is
    K_eff = mu^2 I + sum_r J_r M_r with M_r the range-r Dirichlet Laplacian
    (M_1 is ``harmonic_chain_K`` at m = 0), which is positive definite and
    admits a PSD *local* energy density whenever mu^2, J_r >= 0 — the
    generalization of the half/half bond split of ``vacuum.inequalities.qei``
    to range-r bonds, with the odd-image wall bonds wholly on their site
    (:func:`keff_density_terms`; sum_x h_x = diag(K_eff, I) exactly, tested).
    The couplings are fitted by non-negative least squares so that the
    density is PSD by construction; the misfit of the fitted omega(k) is
    reported, and the unconstrained cosine coefficients beside it.

3.  **The free infimum at K_eff** (:func:`free_infimum_keff`): the same
    Williamson problem as ``qei_minimize`` with the same sampling function,
    lattice, site and cone, only the pullback S(t) and the density split
    taken from K_eff.  With mu^2 = m^2, J = (1,) it reproduces
    ``free_exact_infimum`` to machine precision (tested).

4.  **Decomposition on the extremal state** (:func:`decompose_smeared_energy`).
    h_x = (1/2) pi^2 + (1/2) m^2 phi^2 + (grad phi)^2-bonds + (lambda/4!) phi^4
    is split into its four pieces and each is evolved by the *same*
    operator-space TEBD as ``qei_exact.heisenberg_smeared_operator`` into its
    own MPO O_a; E_a = <psi*|O_a|psi*> - <Omega|O_a|Omega> is the piece of
    the smeared energy the extremal state carries, normal-ordered against the
    interacting vacuum.  Linearity says sum_a E_a = E_min; the pieces are
    truncated separately so the identity holds up to the operator
    truncation, and that deviation is the audit reported.  The same four
    pieces are also measured in the Schroedinger picture along one TEBD
    trajectory of psi* and Omega (:func:`schroedinger_pieces`), where the
    sum equals the Schroedinger re-measurement of E_min identically.

TeNPy is imported lazily; the K_eff / Williamson pieces are pure numpy.
Functions are pure over their MPS arguments.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np
from scipy.optimize import nnls

from vacuum.inequalities.qei import (
    FHandle,
    SamplingOp,
    _assemble_pullback,
    _cone_support,
    qei_minimize,
)

from .phi4 import (
    GroundState,
    _tenpy,
    free_coupling_matrix,
    hartree_coupling_matrix,
    local_boson_ops,
    phi4_model_class,
)
from .qei_exact import (
    QEIExactResult,
    SmearedOperator,
    _bond_hamiltonians,
    _gate,
    _model_params,
    _mpo_from_tensors,
    _operator_mps_from_dense,
    _realify,
    _time_grid,
    local_energy_operator_dense,
    occupation_weights,
)

__all__ = [
    "standing_wave_modes",
    "free_dispersion",
    "harmonic_inversion",
    "peak_width",
    "DispersionResult",
    "interacting_dispersion_tebd",
    "excitation_gap_dmrg",
    "fit_dispersion_couplings",
    "cosine_coefficients",
    "keff_from_couplings",
    "keff_density_terms",
    "free_infimum_keff",
    "energy_pieces_dense",
    "heisenberg_smeared_operator_dense",
    "decompose_smeared_energy",
    "schroedinger_pieces",
]


# --------------------------------------------------------------------------
# Standing waves of the Dirichlet chain and dispersions
# --------------------------------------------------------------------------


def standing_wave_modes(L):
    """(k, V): k_n = n pi/(L+1), V[:, n] = sqrt(2/(L+1)) sin(k_n (j+1)), n = 1..L.

    The exact orthonormal eigenvectors of every Dirichlet-chain coupling matrix
    of the form mu^2 I + sum_r J_r M_r (``keff_from_couplings``).
    """
    L = int(L)
    n = np.arange(1, L + 1)
    k = n * math.pi / (L + 1)
    j = np.arange(L)
    V = math.sqrt(2.0 / (L + 1)) * np.sin(np.outer(j + 1, k))
    return k, V


def free_dispersion(K):
    """omega_n^2 = v_n^T K v_n on the standing waves (exact eigenvalues for a
    Dirichlet-chain K; the sine-basis diagonal otherwise)."""
    K = np.asarray(K, dtype=float)
    _, V = standing_wave_modes(K.shape[0])
    return np.einsum("jn,jk,kn->n", V, K, V)


# --------------------------------------------------------------------------
# Harmonic inversion (matrix pencil)
# --------------------------------------------------------------------------


def harmonic_inversion(signal, dt, *, max_poles=6, sv_tol=1e-4, n_poles=None):
    """Poles and amplitudes of s_k = sum_p A_p z_p^k (Hua-Sarkar matrix pencil).

    Returns a dict: 'z' (complex poles), 'omega' = -arg(z)/dt (so that a
    ground-state correlator e^{-i omega t} has omega > 0), 'gamma' =
    -ln|z|/dt (decay rate), 'A' (complex amplitudes, least squares), 'resid'
    (relative fit residual), 'n_poles'.  The model order is the number of
    Hankel singular values above sv_tol * s_0, capped at max_poles, unless
    n_poles is given.
    """
    s = np.asarray(signal, dtype=complex).ravel()
    N = s.size
    if N < 6:
        raise ValueError("need at least 6 samples")
    dt = float(dt)
    Lp = N // 2
    Y = np.array([s[i:i + Lp + 1] for i in range(N - Lp)])  # (N-Lp, Lp+1) Hankel
    U, sv, Vh = np.linalg.svd(Y, full_matrices=False)
    if n_poles is None:
        M = int(np.sum(sv >= sv_tol * sv[0]))
        M = max(1, min(M, int(max_poles), Lp))
    else:
        M = max(1, min(int(n_poles), Lp))
    # rows of Vh span the row space of Y, i.e. the (un-conjugated) Vandermonde
    # columns z_p^j: the shift-invariance pencil on them has eigenvalues z_p
    # (using Vh^dagger instead would return the conjugate poles)
    V = Vh[:M, :].T
    V1, V2 = V[:-1, :], V[1:, :]
    z = np.linalg.eigvals(np.linalg.pinv(V1) @ V2)
    Z = z[None, :] ** np.arange(N)[:, None]
    A, *_ = np.linalg.lstsq(Z, s, rcond=None)
    resid = float(np.linalg.norm(Z @ A - s) / max(np.linalg.norm(s), 1e-300))
    omega = -np.angle(z) / dt
    gamma = -np.log(np.maximum(np.abs(z), 1e-300)) / dt
    return {"z": z, "omega": omega, "gamma": gamma, "A": A, "resid": resid, "n_poles": M}


def peak_width(signal, times, omega_grid=None, *, window_frac=0.5):
    """FWHM of |int C(t) w(t) e^{i omega t} dt| around its maximum, and the FWHM
    of the same transform of a pure e^{-i omega_0 t} (the window's own response).

    w(t) = exp(-(t / (window_frac T))^2) with T the trajectory length.  The
    ratio FWHM / FWHM_window -> 1 for a resolution-limited (sharp) peak.
    Returns (omega_peak, fwhm, fwhm_window).
    """
    s = np.asarray(signal, dtype=complex).ravel()
    t = np.asarray(times, dtype=float).ravel()
    T = float(t[-1])
    w = np.exp(-(t / (window_frac * T)) ** 2)
    if omega_grid is None:
        omega_grid = np.linspace(0.0, math.pi / (t[1] - t[0]), 20001)
    ph = np.exp(1j * np.outer(omega_grid, t))
    spec = np.abs(ph @ (s * w))
    i0 = int(np.argmax(spec))
    om0 = float(omega_grid[i0])

    def _fwhm(sp):
        half = 0.5 * float(np.max(sp))
        i = int(np.argmax(sp))
        lo = i
        while lo > 0 and sp[lo] > half:
            lo -= 1
        hi = i
        while hi < sp.size - 1 and sp[hi] > half:
            hi += 1
        return float(omega_grid[hi] - omega_grid[lo])

    ref = np.abs(ph @ (np.exp(-1j * om0 * t) * w))
    return om0, _fwhm(spec), _fwhm(ref)


# --------------------------------------------------------------------------
# The interacting dispersion by TEBD correlators
# --------------------------------------------------------------------------


class DispersionResult(NamedTuple):
    k: np.ndarray            # standing-wave momenta k_n
    omega: np.ndarray        # dominant-pole frequency per mode
    gamma: np.ndarray        # its decay rate
    weight: np.ndarray       # its fraction of the projected spectral weight
    amplitude: np.ndarray    # |A_dom|
    n_poles: np.ndarray
    resid: np.ndarray        # matrix-pencil fit residual per mode
    fwhm: np.ndarray         # windowed-FT FWHM per mode
    fwhm_window: np.ndarray  # the window's own FWHM (resolution floor)
    poles: list              # per mode: dict of all poles
    times: np.ndarray
    C: np.ndarray            # (L, nt) complex correlator <phi_j(t) phi_x(0)>
    omega_free: np.ndarray   # exact free dispersion at the bare mass
    omega_hartree: np.ndarray  # ... at the Hartree mass
    E0: float
    trunc_err: float
    chi: int
    meta: dict


def _apply_phi(psi, j):
    out = psi.copy()
    out.apply_local_op(int(j), "X", unitary=False, renormalize=False)
    return out


def interacting_dispersion_tebd(gs: GroundState, site=None, *, T=12.0, dt=0.1, order=4,
                                chi_max=48, svd_min=1e-10, max_poles=6, sv_tol=1e-4):
    """omega(k) of the interacting chain from TEBD correlators (module docstring).

    gs: DMRG ground state; site: the excited site x (default: middle); the
    trajectory runs to T in steps of dt (order-`order` Suzuki-Trotter, bond
    dimension chi_max).  Returns :class:`DispersionResult`.
    """
    t = _tenpy()
    L, m, lam, bc = gs.info["L"], gs.info["m"], gs.info["lam"], gs.info["bc"]
    x = L // 2 if site is None else int(site)
    n_steps = int(round(float(T) / float(dt)))
    times = float(dt) * np.arange(n_steps + 1)
    k, V = standing_wave_modes(L)
    bras = [_apply_phi(gs.psi, j) for j in range(L)]
    psi = _apply_phi(gs.psi, x)
    eng = t.tebd.TEBDEngine(psi, gs.model, {
        "order": int(order), "dt": float(dt), "N_steps": 1,
        "trunc_params": {"chi_max": int(chi_max), "svd_min": float(svd_min)},
    })
    C = np.empty((L, n_steps + 1), dtype=complex)
    E0 = float(gs.E0)
    for s in range(n_steps + 1):
        if s > 0:
            eng.run()
        ph = np.exp(1j * E0 * times[s])
        for j in range(L):
            C[j, s] = ph * bras[j].overlap(eng.psi)
    trunc = float(eng.trunc_err.eps)
    chi = int(max(eng.psi.chi))
    Cn = V.T @ C                                   # (L, nt) projected signals
    omega = np.empty(L)
    gamma = np.empty(L)
    weight = np.empty(L)
    amp = np.empty(L)
    npl = np.empty(L, dtype=int)
    resid = np.empty(L)
    fw = np.empty(L)
    fw0 = np.empty(L)
    poles = []
    for n in range(L):
        hi = harmonic_inversion(Cn[n], dt, max_poles=max_poles, sv_tol=sv_tol)
        pos = np.flatnonzero(hi["omega"] > 0.0)
        if pos.size == 0:
            pos = np.arange(hi["omega"].size)
        absA = np.abs(hi["A"][pos])
        i = int(pos[np.argmax(absA)])
        omega[n] = hi["omega"][i]
        gamma[n] = hi["gamma"][i]
        amp[n] = abs(hi["A"][i])
        weight[n] = abs(hi["A"][i]) / max(float(np.sum(absA)), 1e-300)
        npl[n] = hi["n_poles"]
        resid[n] = hi["resid"]
        _, fw[n], fw0[n] = peak_width(Cn[n], times)
        poles.append({key: hi[key] for key in ("omega", "gamma", "A", "z")})
    K0 = free_coupling_matrix(L, m, bc)
    KH = hartree_coupling_matrix(L, m, lam, bc)
    meta = {"L": L, "m": m, "lam": lam, "bc": bc, "n_max": gs.info["n_max"], "site": x,
            "T": float(T), "dt": float(dt), "order": int(order), "chi_max": int(chi_max),
            "n_steps": n_steps}
    return DispersionResult(k=k, omega=omega, gamma=gamma, weight=weight, amplitude=amp,
                            n_poles=npl, resid=resid, fwhm=fw, fwhm_window=fw0, poles=poles,
                            times=times, C=C, omega_free=np.sqrt(free_dispersion(K0)),
                            omega_hartree=np.sqrt(free_dispersion(KH)), E0=E0,
                            trunc_err=trunc, chi=chi, meta=meta)


def _shifted_model(gs: GroundState, shift_per_site):
    """The phi^4 model minus `shift_per_site` * Id on every site (a constant)."""
    base = phi4_model_class()

    class _Shifted(base):
        def init_terms(self, model_params):
            super().init_terms(model_params)
            self.add_onsite(-float(model_params.get("shift", 0.0, float)), 0, "Id")

    _Shifted.__module__ = __name__
    prm = _model_params(gs.model)
    return _Shifted({**prm, "bc_MPS": "finite", "shift": float(shift_per_site)})


def excitation_gap_dmrg(gs: GroundState, *, chi_max=None, max_sweeps=20, max_E_err=1e-10):
    """E_1 - E_0 by DMRG orthogonal to the ground state (independent of TEBD).

    The model is shifted by a constant so that both energies are negative
    (TeNPy's ``orthogonal_to`` needs the target below zero).  Returns
    (gap, info).
    """
    t = _tenpy()
    L = gs.info["L"]
    chi_max = int(gs.info["chi_max"] if chi_max is None else chi_max)
    shift = float(gs.E0) / L + 1.0
    model = _shifted_model(gs, shift)
    E0s = float(np.real(model.H_MPO.expectation_value(gs.psi)))
    psi = gs.psi.copy()
    psi.perturb({"N_steps": 1, "trunc_params": {"chi_max": chi_max}}, close_1=False)
    psi.canonical_form()
    options = {
        "trunc_params": {"chi_max": chi_max, "svd_min": 1e-13},
        "max_E_err": float(max_E_err), "max_S_err": 1e-8,
        "max_sweeps": int(max_sweeps), "min_sweeps": 4, "mixer": True,
        "lanczos_params": {"N_max": 60, "P_tol": 1e-16, "E_tol": 1e-16},
    }
    # TeNPy 1.x takes the states to project out as a constructor keyword, not an option
    eng = t.dmrg.TwoSiteDMRGEngine(psi, model, options, orthogonal_to=[gs.psi])
    E1s, psi1 = eng.run()
    if not (E0s < 0.0 and E1s < 0.0):
        raise RuntimeError(f"shifted energies must be negative: E0 {E0s}, E1 {E1s}")
    ov = abs(gs.psi.overlap(psi1))
    return float(E1s - E0s), {"E0_shifted": E0s, "E1_shifted": float(E1s), "overlap_with_ground": float(ov),
                              "sweeps": int(eng.sweeps), "chi": int(max(psi1.chi))}


# --------------------------------------------------------------------------
# K_eff: the inverse Fourier transform of omega(k)^2 and its PSD density
# --------------------------------------------------------------------------


def cosine_coefficients(k, omega2):
    """Unconstrained (mu^2, J_1..J_{L-1}) with omega^2(k_n) = mu^2 + sum_r J_r (2 - 2 cos r k_n)
    exactly (square linear solve on the L standing waves)."""
    k = np.asarray(k, float).ravel()
    w2 = np.asarray(omega2, float).ravel()
    L = k.size
    B = np.column_stack([np.ones(L)] + [2.0 - 2.0 * np.cos(r * k) for r in range(1, L)])
    c = np.linalg.solve(B, w2)
    return float(c[0]), c[1:].copy()


def fit_dispersion_couplings(k, omega2, r_max=4, *, weights=None):
    """Non-negative least squares for (mu^2, J_1..J_{r_max}) (module docstring).

    Returns (mu2, J, info) with info['omega2_fit'], info['misfit_max'] (max
    |omega_fit - omega| over the modes), info['misfit_rms'].
    """
    k = np.asarray(k, float).ravel()
    w2 = np.asarray(omega2, float).ravel()
    r_max = int(r_max)
    B = np.column_stack([np.ones(k.size)] + [2.0 - 2.0 * np.cos(r * k) for r in range(1, r_max + 1)])
    wt = np.ones(k.size) if weights is None else np.sqrt(np.asarray(weights, float).ravel())
    c, rnorm = nnls(B * wt[:, None], w2 * wt, maxiter=10000)
    w2_fit = B @ c
    om = np.sqrt(np.maximum(w2, 0.0))
    om_fit = np.sqrt(np.maximum(w2_fit, 0.0))
    info = {"omega2_fit": w2_fit, "misfit_max": float(np.max(np.abs(om_fit - om))),
            "misfit_rms": float(np.sqrt(np.mean((om_fit - om) ** 2))), "rnorm": float(rnorm)}
    return float(c[0]), c[1:].copy(), info


def keff_from_couplings(L, mu2, J):
    """K_eff = mu^2 I + sum_r J_r M_r, M_r = sum_n (2 - 2 cos r k_n) v_n v_n^T.

    With mu2 = m^2 and J = (1,) this is ``harmonic_chain_K(L, m, 'dirichlet')``.
    """
    L = int(L)
    k, V = standing_wave_modes(L)
    w2 = float(mu2) * np.ones(L)
    for r, Jr in enumerate(np.asarray(J, float).ravel(), start=1):
        w2 += float(Jr) * (2.0 - 2.0 * np.cos(r * k))
    K = (V * w2[None, :]) @ V.T
    return 0.5 * (K + K.T)


def keff_density_terms(L, site, mu2, J):
    """PSD rank-1 terms of h_site for K_eff (the ``qei._chain_density_terms`` convention).

    h = sum coeff u u^T, energy = (1/2) R^T h R.  Interior range-r bonds
    (J_r/2)(phi_i - phi_{i+r})^2 are split half/half between their two
    sites; a bond crossing a wall meets the odd image of its partner
    (phi_{-1-a} = -phi_{a-1}, phi_{L+a} = -phi_{L-a}) — if the image is the
    wall itself (phi = 0) the term (J_r/2) phi_i^2 is wholly on site i, if it
    is site i itself the self-image bond gives J_r phi_i^2, and otherwise the
    bond and its mirror give (J_r/2)(phi_i + phi_j')^2 split half/half.
    sum_site h_site = diag(K_eff, I) exactly (tested).
    """
    L = int(L)
    x = int(site)
    if not 0 <= x < L:
        raise ValueError(f"site must be in [0, {L}), got {x}")
    J = np.asarray(J, float).ravel()
    if J.size > L - 1:
        raise ValueError(f"at most L - 1 = {L - 1} couplings, got {J.size}")
    ei = np.array([], dtype=int)
    ev = np.array([])
    terms = [(1.0, ei, ev, np.array([x]), np.array([1.0]))]
    if mu2 != 0.0:
        terms.append((float(mu2), np.array([x]), np.array([1.0]), ei, ev))

    def wall(Jr, jp, wall_site):
        if jp == wall_site:
            terms.append((Jr, np.array([x]), np.array([1.0]), ei, ev))
        elif jp == x:
            terms.append((2.0 * Jr, np.array([x]), np.array([1.0]), ei, ev))
        else:
            terms.append((0.5 * Jr, np.array([x, jp]), np.array([1.0, 1.0]), ei, ev))

    for r, Jr in enumerate(J, start=1):
        Jr = float(Jr)
        if Jr == 0.0:
            continue
        for partner in (x - r, x + r):
            if 0 <= partner < L:
                terms.append((0.5 * Jr, np.array([x, partner]), np.array([1.0, -1.0]), ei, ev))
            elif partner < 0:
                wall(Jr, -2 - partner, -1)
            else:
                wall(Jr, 2 * L - partner, L)
    return terms


def free_infimum_keff(L, site, f: FHandle, mu2, J, *, pad=8, nodes_per_panel=10,
                      weight_floor=1e-9, panel_width=None, eps_reg=1e-8):
    """The free Williamson infimum of the chain K_eff(mu2, J) with the same sampling.

    Same cone, quadrature and trim as ``qei.sampling_operator``; only the
    coupling matrix and its PSD density split are K_eff's.  Returns
    (QEIResult, SamplingOp).
    """
    L = int(L)
    x = int(site)
    K = keff_from_couplings(L, mu2, J)
    terms = keff_density_terms(L, x, mu2, J)
    t_lo, t_hi = float(f.t_lo), float(f.t_hi)
    t_max = max(abs(t_lo), abs(t_hi))
    support = _cone_support(L, x, int(math.ceil(t_max)) + int(pad), "dirichlet")
    O, dO, V_vac_sub, support, n_nodes, pw = _assemble_pullback(
        K, terms, f.f, t_lo, t_hi, support, panel_width, nodes_per_panel, weight_floor=weight_floor)
    meta = {"mu2": float(mu2), "J": np.asarray(J, float).tolist(), "bc": "dirichlet", "pad": int(pad),
            "panel_width": float(pw), "nodes_per_panel": int(nodes_per_panel), "t_lo": t_lo,
            "t_hi": t_hi, "weight_floor": float(weight_floor), "label": f.label}
    op = SamplingOp(O=O, support=support, V_vac_sub=V_vac_sub, N=L, site=x, t_max=t_max,
                    n_nodes=int(n_nodes), dO=dO, meta=meta)
    return qei_minimize(op, eps_reg=eps_reg), op


# --------------------------------------------------------------------------
# Decomposition of the smeared energy on the extremal state
# --------------------------------------------------------------------------

PIECES = ("kinetic", "gradient", "mass", "quartic")


def energy_pieces_dense(n_max, omega, m, lam, site, L, bc="dirichlet"):
    """The four pieces of h_site as dense matrices on its support (sites, {name: H}).

    kinetic = (1/2) Psq, mass = (1/2) m^2 Xsq, quartic = (lam/4!) X4 on the
    site; gradient = the PSD bond terms (1/4)(Xsq_j + Xsq_k - 2 X_j X_k) of the
    adjacent bonds plus Dirichlet wall terms.  Their sum is
    ``qei_exact.local_energy_operator_dense`` (tested).
    """
    ops = local_boson_ops(n_max, omega)
    d = n_max + 1
    L = int(L)
    x = int(site)
    lo, hi = max(0, x - 1), min(L - 1, x + 1)
    sites = list(range(lo, hi + 1))
    eye = np.eye(d)

    def embed(mats):
        out = np.array([[1.0]])
        for j in sites:
            out = np.kron(out, mats.get(j, eye))
        return out

    kin = 0.5 * embed({x: ops["Psq"]})
    mass = 0.5 * float(m) ** 2 * embed({x: ops["Xsq"]})
    quart = (float(lam) / 24.0) * embed({x: ops["X4"]})
    grad = np.zeros_like(kin)
    for j in (x - 1, x):
        if 0 <= j < L - 1:
            grad += 0.25 * (embed({j: ops["Xsq"]}) + embed({j + 1: ops["Xsq"]})
                            - 2.0 * embed({j: ops["X"], j + 1: ops["X"]}))
    if bc == "dirichlet":
        if x == 0:
            grad += 0.5 * embed({0: ops["Xsq"]})
        if x == L - 1:
            grad += 0.5 * embed({L - 1: ops["Xsq"]})
    pieces = {"kinetic": kin, "gradient": grad, "mass": mass, "quartic": quart}
    return sites, {k: 0.5 * (v + v.T) for k, v in pieces.items()}


def heisenberg_smeared_operator_dense(model, site, sites_h, H3, f: FHandle, *, dt=0.25, order=4,
                                      chi_op=24, svd_min_op=1e-8, chi_acc=None, svd_min_acc=1e-10,
                                      window_pad=2, psi_ref=None, seed=0, svd_min_real=1e-9,
                                      chi_mpo=None, weight_decay=0.25):
    """O = int f^2 h(t) dt for an arbitrary dense h on `sites_h` (same machinery
    as ``qei_exact.heisenberg_smeared_operator``, which fixes h = h_site).

    Every knob has the same meaning as there; pass the identical values to
    evolve a piece of h_site under the identical truncation policy.  A zero
    operator (e.g. the quartic piece at lambda = 0) returns None.
    """
    t = _tenpy()
    prm = _model_params(model)
    L, n_max = prm["L"], prm["n_max"]
    x = int(site)
    d = n_max + 1
    if chi_acc is None:
        chi_acc = 2 * int(chi_op)
    rng = np.random.default_rng(int(seed))
    w_occ = None if weight_decay >= 1.0 else occupation_weights(n_max, weight_decay)
    H3 = np.asarray(H3, float)
    if float(np.max(np.abs(H3))) == 0.0:
        return None
    h = _operator_mps_from_dense(list(sites_h), H3, L, d)
    scale = math.sqrt(h.norm2())
    h = h.scale(1.0 / scale)
    h.canonicalize()
    h.T = [tt.astype(complex) for tt in h.T]
    Hbonds = _bond_hamiltonians(model)
    fracs = t.tebd.TEBDEngine.suzuki_trotter_time_steps(order)
    times, w, dt_eff = _time_grid(f, dt)
    layers = t.tebd.TEBDEngine.suzuki_trotter_decomposition(order, 1)[::-1]
    gates = {}
    for k_dt, _parity in set(layers):
        delta = fracs[k_dt] * dt_eff
        gates[k_dt] = {i: _gate(Hb, delta) for i, Hb in Hbonds.items()}
    f2 = np.asarray(f.f(times), float) ** 2

    def ref_expect(op):
        if psi_ref is None:
            return None
        return float(np.real(_mpo_from_tensors(psi_ref.sites, op.T, scale).expectation_value(psi_ref)))

    acc = h.scale(w[0] * f2[0])
    href = [ref_expect(h)]
    disc_op = 0.0
    disc_acc = 0.0
    chi_op_max = max(h.chi) if h.chi else 1
    chi_acc_max = max(acc.chi) if acc.chi else 1
    for k in range(1, times.size):
        t_end = times[k]
        if window_pad is None:
            lo, hi = 0, L - 1
        else:
            R = int(math.ceil(t_end - 1e-9)) + int(window_pad)
            lo, hi = max(0, x - R), min(L - 1, x + R)
        for k_dt, parity in layers:
            for i in range(1, L):
                if i % 2 != parity or i - 1 < lo or i > hi:
                    continue
                disc, tot = h.update_bond(i - 1, gates[k_dt][i], chi_op, svd_min_op, rng=rng, w=w_occ)
                if tot > 0.0:
                    disc_op += disc / tot
        chi_op_max = max(chi_op_max, max(h.chi))
        acc = acc.add(h, 1.0, w[k] * f2[k])
        disc_acc += acc.canonicalize(chi_max=chi_acc, svd_min=svd_min_acc)
        chi_acc_max = max(chi_acc_max, max(acc.chi))
        href.append(ref_expect(h))
    norm_ratio = math.sqrt(h.norm2())
    O = _realify(acc).add(_realify(acc.transpose()), 1.0, 1.0)
    disc_real = O.canonicalize(chi_max=chi_mpo, svd_min=svd_min_real)
    O.T = [np.real(tt) for tt in O.T]
    mpo = _mpo_from_tensors(model.lat.mps_sites(), O.T, scale)
    meta = {"site": x, "L": L, "m": prm["m"], "lam": prm["lam"], "n_max": n_max, "omega": prm["omega"],
            "bc": prm["bc"], "dt": dt_eff, "order": order, "chi_op_max_allowed": int(chi_op),
            "svd_min_op": svd_min_op, "chi_acc_max_allowed": int(chi_acc), "svd_min_acc": svd_min_acc,
            "window_pad": window_pad, "chi_mpo_max_allowed": (None if chi_mpo is None else int(chi_mpo)),
            "svd_min_real": float(svd_min_real), "weight_decay": float(weight_decay), "f": f.label,
            "t_hi": float(f.t_hi), "n_steps": int(times.size - 1)}
    return SmearedOperator(mpo=mpo, tensors=O.T, scale=scale, times=times, weights=w,
                           trunc_op=float(disc_op), trunc_acc=float(disc_acc), trunc_real=float(disc_real),
                           chi_op=int(chi_op_max), chi_acc=int(chi_acc_max), chi_mpo=int(max(O.chi)),
                           h_ref_trace=np.array([v for v in href if v is not None], float),
                           norm_ratio=float(norm_ratio), meta=meta)


def decompose_smeared_energy(res: QEIExactResult, gs: GroundState, f: FHandle, **heisenberg_kw):
    """E_a = <psi*|O_a|psi*> - <Omega|O_a|Omega> for the four pieces (module docstring).

    `heisenberg_kw` must be the operator knobs the infimum was built with
    (dt, order, chi_op, svd_min_op, chi_acc, svd_min_acc, window_pad,
    svd_min_real, chi_mpo, weight_decay, seed).  Returns a dict with the
    pieces, 'sum', 'E_min', 'audit' = sum - E_min, 'audit_rel', the per-piece
    truncation records, and the equal-time squeeze <phi_x^2>, <pi_x^2>,
    <phi_x^4> of psi* relative to Omega.
    """
    prm = _model_params(gs.model)
    x = int(res.meta["site"])
    sites_h, pieces = energy_pieces_dense(prm["n_max"], prm["omega"], prm["m"], prm["lam"], x,
                                          prm["L"], prm["bc"])
    out = {"pieces": {}, "ref": {}, "star": {}, "trunc_op": {}, "norm_ratio": {}, "chi_mpo": {}}
    total = 0.0
    for name in PIECES:
        op = heisenberg_smeared_operator_dense(gs.model, x, sites_h, pieces[name], f,
                                               psi_ref=gs.psi, **heisenberg_kw)
        if op is None:
            e_star = e_ref = 0.0
            out["trunc_op"][name] = 0.0
            out["norm_ratio"][name] = 1.0
            out["chi_mpo"][name] = 0
        else:
            e_star = float(np.real(op.mpo.expectation_value(res.psi_star)))
            e_ref = float(np.real(op.mpo.expectation_value(gs.psi)))
            out["trunc_op"][name] = op.trunc_op
            out["norm_ratio"][name] = op.norm_ratio
            out["chi_mpo"][name] = op.chi_mpo
        out["pieces"][name] = e_star - e_ref
        out["star"][name] = e_star
        out["ref"][name] = e_ref
        total += e_star - e_ref
    out["sum"] = total
    out["E_min"] = float(res.E_min)
    out["audit"] = total - float(res.E_min)
    out["audit_rel"] = (total - float(res.E_min)) / abs(float(res.E_min))
    for key, opname in (("phi2", "Xsq"), ("pi2", "Psq"), ("phi4", "X4")):
        s = float(np.real(res.psi_star.expectation_value(opname, sites=[x])[0]))
        v = float(np.real(gs.psi.expectation_value(opname, sites=[x])[0]))
        out[f"{key}_star"] = s
        out[f"{key}_vac"] = v
    return out


def schroedinger_pieces(psi, model, site, f: FHandle, psi_ref, *, dt=0.25, order=4, chi_max=32,
                        svd_min=1e-12):
    """The four pieces along one TEBD trajectory of psi and of psi_ref (Schroedinger picture).

    E_a = 2 sum_k w_k f(t_k)^2 [<h_a>_psi(t_k) - <h_a>_ref(t_k)] with the
    trapezoid weights of ``qei_exact._time_grid`` (real states: h(-t) = h(t)),
    so sum_a E_a is the Schroedinger re-measurement of the smeared energy.
    Returns a dict of pieces plus 'sum', 'trunc_err', 'chi'.
    """
    t = _tenpy()
    prm = _model_params(model)
    x = int(site)
    L, m, lam, bc = prm["L"], prm["m"], prm["lam"], prm["bc"]
    times, w, dt_eff = _time_grid(f, dt)
    f2 = np.asarray(f.f(times), float) ** 2

    def measure(state):
        e = {}
        e["kinetic"] = 0.5 * float(np.real(state.expectation_value("Psq", sites=[x])[0]))
        xsq = {j: float(np.real(state.expectation_value("Xsq", sites=[j])[0]))
               for j in range(max(0, x - 1), min(L - 1, x + 1) + 1)}
        e["mass"] = 0.5 * m * m * xsq[x]
        e["quartic"] = (lam / 24.0) * float(np.real(state.expectation_value("X4", sites=[x])[0])) if lam else 0.0
        g = 0.0
        for j in (x - 1, x):
            if 0 <= j < L - 1:
                xx = float(np.real(state.correlation_function("X", "X", sites1=[j], sites2=[j + 1])[0, 0]))
                g += 0.25 * (xsq[j] + xsq[j + 1] - 2.0 * xx)
        if bc == "dirichlet":
            if x == 0:
                g += 0.5 * xsq[0]
            if x == L - 1:
                g += 0.5 * xsq[L - 1]
        e["gradient"] = g
        return e

    def trace(state):
        work = state.copy()
        eng = t.tebd.TEBDEngine(work, model, {
            "order": int(order), "dt": float(dt_eff), "N_steps": 1,
            "trunc_params": {"chi_max": int(chi_max), "svd_min": float(svd_min)},
        })
        rows = [measure(work)]
        for _ in range(1, times.size):
            eng.run()
            rows.append(measure(eng.psi))
        return rows, float(eng.trunc_err.eps), int(max(eng.psi.chi))

    rows, trunc, chi = trace(psi)
    rows_ref, trunc_ref, _ = trace(psi_ref)
    out = {}
    for name in PIECES:
        h = np.array([r[name] for r in rows])
        h0 = np.array([r[name] for r in rows_ref])
        out[name] = 2.0 * float(np.sum(w * f2 * (h - h0)))
    out["sum"] = float(sum(out[n] for n in PIECES))
    out["trunc_err"] = max(trunc, trunc_ref)
    out["chi"] = chi
    return out
