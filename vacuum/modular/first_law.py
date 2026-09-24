"""Numerical check of the entanglement first law, delta S = delta <K>.

For a small perturbation of a global Gaussian ground state, the change in the
entanglement entropy of a region A equals the change in the modular energy of
A computed with the *unperturbed* entanglement Hamiltonian K_A:

    delta S(A) = delta <K_A> = (1/2) Tr[ G (V_A' - V_A) ],

where rho_A ~ exp(-(1/2) R^T G R) is the unperturbed reduced state (G from
``vacuum.core.entanglement_hamiltonian`` via Williamson). This is the lattice
Gaussian form of the first law of entanglement entropy: at first order in the
state perturbation, delta S = delta <K> holds for *any* global perturbation,
because S(rho_A) and Tr[rho_A K_A] have the same first variation
(dS = Tr[d rho_A K_A]; equivalently for Gaussian states dS/dV_A = G/2).

The two sides are computed by deliberately independent routes:

- LEFT: entropy differences from the symplectic eigenvalues of the perturbed
  reduced covariance (``vacuum.core.entropy``);
- RIGHT: (1/2) Tr[G dV_A] with the G-matrix built ONCE from the unperturbed
  V_A (``vacuum.core.entanglement_hamiltonian`` -> ``modular_energy``).

Agreement validates the entanglement-Hamiltonian construction that the
Layer-4 modular program depends on (PLAN.md, Layers 1 and 4).

Anchor references
-----------------
- D. D. Blanco, H. Casini, L.-Y. Hung, R. C. Myers, JHEP 08 (2013) 060,
  "Relative entropy and holography": delta S = delta <K> as the first law of
  entanglement entropy (positivity of relative entropy at first order).
- T. Faulkner, M. Guica, T. Hartman, R. C. Myers, M. Van Raamsdonk,
  JHEP 03 (2014) 051, "Gravitation from entanglement in holographic CFTs":
  the entanglement first law as the engine of the geometry-from-entanglement
  program (the form delta <K> = (1/2) Tr[G dV_A] is its lattice avatar).
- A. Botero, B. Reznik, Phys. Rev. A 70, 052329 (2004); J. Eisert, M. Cramer,
  M. B. Plenio, Rev. Mod. Phys. 82, 277 (2010): Gaussian entanglement
  Hamiltonians via the Williamson decomposition.
- M. Srednicki, Phys. Rev. Lett. 71, 666 (1993): ground-state covariance of
  H = (1/2)(p.p + x.K.x) used to build the perturbed global states.

Conventions: hbar = 1, vacuum nu = 1/2, block quadrature ordering
R = (x_1..x_N, p_1..p_N), entropies in nats (docs/API.md).
"""

from __future__ import annotations

import numpy as np

from vacuum.core import (
    entanglement_hamiltonian,
    entropy,
    ground_state_cov,
    modular_energy,
    reduce,
    symplectic_inverse,
    williamson,
)
from vacuum.core.precision import entropy_precise

__all__ = ["first_law_check"]

# Must match the default clamp in vacuum.core.entanglement_hamiltonian so the
# per-mode diagnostic decomposition reproduces the G used on the right side.
_NU_FLOOR = 0.5 * (1.0 + 1e-14)


def first_law_check(K, modes, perturb, eps, floor_tol=1e-8, use_precise=False):
    """Check the entanglement first law delta S = delta <K> for region `modes`.

    The global state is the ground state of H = (1/2)(p.p + x^T K x); the
    perturbed global states are the ground states of ``perturb(K, e)``. Both
    sides of the first law are evaluated as derivatives d/de at e = 0 using
    central differences at +-e (killing all odd orders, leaving O(e^2)),
    Richardson-extrapolated over the two step sizes ``eps`` and ``eps/2`` to
    cancel the O(e^2) term as well, leaving O(e^4) truncation error.

    Parameters
    ----------
    K : (N, N) ndarray
        Unperturbed coupling matrix (symmetric positive definite).
    modes : sequence of int
        0-based mode indices of region A.
    perturb : callable
        ``perturb(K, e) -> K_e``, smooth in the scalar ``e`` with
        ``perturb(K, 0) == K`` (checked). Any global perturbation is legal —
        the first law does not care where the perturbation lives.
    eps : float or (float, float)
        Base step. A scalar uses the pair (eps, eps/2); an explicit pair
        (e_coarse, e_fine) with e_fine < e_coarse is Richardson-combined with
        the general weights (e_c^2 D_f - e_f^2 D_c) / (e_c^2 - e_f^2).
    floor_tol : float
        Modes of the unperturbed V_A with nu - 1/2 < floor_tol are labelled
        vacuum-floor modes in the diagnostics (they receive the clamp inside
        ``entanglement_hamiltonian``; their contribution to delta<K> is
        reported so callers can *verify* it is negligible).
    use_precise : bool
        If True, the entropy side uses ``vacuum.core.precision.entropy_precise``
        (automatic float64 -> mpmath escalation near the vacuum floor).

    Returns
    -------
    dict with keys
        dS : float            — d S(A)/de at e = 0 (entropy route, nats).
        dK_modular : float    — d <K_A>/de at e = 0 (modular route, nats).
        rel_err : float       — |dS - dK_modular| / |dS|.
        dS_pair, dK_pair : (float, float) — un-extrapolated (coarse, fine)
            central differences, to expose the Richardson gain.
        nu_A : (L,) ndarray   — unperturbed symplectic eigenvalues, descending.
        mode_contributions : (L,) ndarray — per-Williamson-mode contribution
            to dK_modular, (1/2) g(nu_k) d(S^{-1} V_A S^{-T})_kk/de; sums to
            dK_modular up to roundoff.
        floor_mask : (L,) bool ndarray — modes with nu - 1/2 < floor_tol.
        dK_floor_modes : float — total contribution of the floor modes.
        eps : (float, float)  — the (coarse, fine) steps actually used.
    """
    K = np.asarray(K, dtype=float)
    modes = list(modes)

    K0 = np.asarray(perturb(K, 0.0), dtype=float)
    if not np.allclose(K0, K, rtol=0.0, atol=1e-13 * max(1.0, np.abs(K).max())):
        raise ValueError("perturb(K, 0) must return K unchanged")

    if np.isscalar(eps):
        e_c, e_f = float(eps), 0.5 * float(eps)
    else:
        e_c, e_f = (float(e) for e in eps)
        if not 0.0 < e_f < e_c:
            raise ValueError(f"need 0 < e_fine < e_coarse, got ({e_c}, {e_f})")

    S_of = entropy_precise if use_precise else entropy

    # Unperturbed region state and its entanglement Hamiltonian — built ONCE.
    V_A = reduce(ground_state_cov(K), modes)
    G = entanglement_hamiltonian(V_A)

    def central(e):
        """Central-difference derivatives at step e; O(e^2) truncation."""
        V_p = reduce(ground_state_cov(perturb(K, +e)), modes)
        V_m = reduce(ground_state_cov(perturb(K, -e)), modes)
        dS = (S_of(V_p) - S_of(V_m)) / (2.0 * e)
        dV = (V_p - V_m) / (2.0 * e)
        # Difference form (1/2) Tr[G (V_p - V_m)]: no large-constant
        # cancellation inside the trace.
        dK = modular_energy(G, dV)
        return dS, dK, dV

    dS_c, dK_c, dV_c = central(e_c)
    dS_f, dK_f, dV_f = central(e_f)

    # Richardson: D(e) = D + a e^2 + O(e^4)  =>  weights kill the e^2 term.
    w = e_c**2 / (e_c**2 - e_f**2)
    dS = w * dS_f + (1.0 - w) * dS_c
    dK = w * dK_f + (1.0 - w) * dK_c
    dV = w * dV_f + (1.0 - w) * dV_c

    denom = abs(dS)
    if denom > 0.0:
        rel_err = abs(dS - dK) / denom
    else:
        rel_err = 0.0 if dK == 0.0 else np.inf

    # --- Diagnostics: per-mode decomposition of dK in the Williamson basis.
    # G = S^{-T} diag(g, g) S^{-1}  =>  (1/2) Tr[G dV] splits mode by mode as
    # (1/2) g_k (W_kk + W_{L+k, L+k}) with W = S^{-1} dV S^{-T}.
    S_w, nu = williamson(V_A)
    Sinv = symplectic_inverse(S_w)
    W = Sinv @ dV @ Sinv.T
    L = len(modes)
    nu_cl = np.maximum(nu, _NU_FLOOR)
    g = np.log((nu_cl + 0.5) / (nu_cl - 0.5))
    d = np.diag(W)
    mode_contrib = 0.5 * g * (d[:L] + d[L:])
    floor_mask = (nu - 0.5) < floor_tol

    return {
        "dS": float(dS),
        "dK_modular": float(dK),
        "rel_err": float(rel_err),
        "dS_pair": (float(dS_c), float(dS_f)),
        "dK_pair": (float(dK_c), float(dK_f)),
        "nu_A": nu,
        "mode_contributions": mode_contrib,
        "floor_mask": floor_mask,
        "dK_floor_modes": float(np.sum(mode_contrib[floor_mask])),
        "eps": (e_c, e_f),
    }
