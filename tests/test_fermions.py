"""Validation anchors for vacuum.fermions.correlation (Peschel machinery).

1. Exact-diagonalization cross-check: many-body ground state of the N=8 open
   hopping chain (occupation basis, Jordan-Wigner signs), reduced density
   matrix of the first 4 sites, von Neumann entropy vs. block_entropy from
   the correlation matrix (Peschel, J. Phys. A 36, L205 (2003)).
2. Central charge of the critical chain: S(l) = (c/3) ln[(N/pi) sin(pi l/N)]
   + c1 with c = 1 (Vidal-Latorre-Rico-Kitaev PRL 90, 227902 (2003);
   Calabrese-Cardy J. Stat. Mech. P06002 (2004)), fitting the standard
   parity-oscillating finite-size correction ~ (-1)^l (Calabrese-Essler-
   Fagotti-type term at k_F = pi/2).
3. entanglement_hamiltonian consistency: entropy recomputed from the
   single-particle entanglement energies eps_k via the free-fermion formula
   S = sum_k [ln(1 + e^{-eps_k}) + eps_k/(e^{eps_k} + 1)].

All entropies in nats (docs/API.md).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from vacuum.fermions import block_entropy, entanglement_hamiltonian, tight_binding_C

# ---------------------------------------------------------------------------
# Module-scoped fixtures (cache the expensive builds)
# ---------------------------------------------------------------------------

N_CRIT = 400  # antiperiodic bc with N = 0 mod 4: unique half-filled ground state


@pytest.fixture(scope="module")
def critical_C():
    """Half-filled critical chain, antiperiodic bc (no Fermi-level degeneracy)."""
    return tight_binding_C(N_CRIT, filling=0.5, bc="antiperiodic")


@pytest.fixture(scope="module")
def open8_C():
    """Half-filled N=8 open chain (spectrum -cos(q pi/9): unique ground state)."""
    return tight_binding_C(8, filling=0.5, bc="open")


# ---------------------------------------------------------------------------
# 1. Exact-diagonalization cross-check
# ---------------------------------------------------------------------------


def _popcount(x):
    return bin(x).count("1")


def _hop_element(b, i, j):
    """Apply c_i^dag c_j to occupation-basis state b (bit i = site i).

    Returns (b_new, sign) with the Jordan-Wigner fermionic sign
    (-1)^{# occupied sites below the operator site}, or None if annihilated.
    """
    if not (b >> j) & 1:
        return None
    sign = -1 if _popcount(b & ((1 << j) - 1)) & 1 else 1
    b1 = b & ~(1 << j)
    if (b1 >> i) & 1:
        return None
    if _popcount(b1 & ((1 << i) - 1)) & 1:
        sign = -sign
    return b1 | (1 << i), sign


def _many_body_ground_state_entropy(N, n_block):
    """Entropy of the first n_block sites from full exact diagonalization.

    Builds H = -(1/2) sum_{i=0}^{N-2} (c_i^dag c_{i+1} + h.c.) in the
    2^N-dimensional occupation basis, restricts to the half-filling sector,
    finds the (unique) sector ground state, forms the reduced density matrix
    of sites 0..n_block-1 by partial trace and returns its von Neumann
    entropy in nats. Independent of the correlation-matrix machinery.
    """
    dim = 1 << N
    H = np.zeros((dim, dim))
    for b in range(dim):
        for a in range(N - 1):
            for (i, j) in ((a, a + 1), (a + 1, a)):
                out = _hop_element(b, i, j)
                if out is not None:
                    b1, sign = out
                    H[b1, b] += -0.5 * sign

    assert np.allclose(H, H.T), "many-body Hamiltonian must be symmetric"

    sector = np.array([b for b in range(dim) if _popcount(b) == N // 2])
    H_sec = H[np.ix_(sector, sector)]
    w, v = np.linalg.eigh(H_sec)
    assert w[1] - w[0] > 1e-10, "half-filling sector ground state must be unique"

    psi = np.zeros(dim)
    psi[sector] = v[:, 0]

    # Partial trace over sites n_block..N-1. Basis index b = b_A + 2^n_block b_B
    # (bit i = site i), so reshape row-major gives M[b_B, b_A] = psi[b].
    dA = 1 << n_block
    M = psi.reshape(dim // dA, dA)
    rho_A = M.T @ M
    p = np.linalg.eigvalsh(rho_A)
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log(p)))


def test_exact_diagonalization_cross_check(open8_C):
    """Peschel correlation-matrix entropy vs. brute-force 2^8 ED, N=8 open chain."""
    S_ed = _many_body_ground_state_entropy(N=8, n_block=4)
    S_corr = block_entropy(open8_C, range(4))
    assert abs(S_corr - S_ed) < 1e-8, (
        f"correlation-matrix entropy {S_corr:.12f} != ED entropy {S_ed:.12f} "
        f"(diff {S_corr - S_ed:.3e})"
    )
    # sanity: a half-filled ground state has Tr C = N/2 and 0 <= C <= 1
    assert abs(np.trace(open8_C) - 4.0) < 1e-12
    zeta = np.linalg.eigvalsh(open8_C)
    assert zeta.min() > -1e-12 and zeta.max() < 1 + 1e-12


# ---------------------------------------------------------------------------
# 2. Critical chain central charge (Calabrese-Cardy)
# ---------------------------------------------------------------------------


def test_central_charge_critical_chain(critical_C):
    """Fit S(l) = (c/3) ln W(l) + c1 + b (-1)^l / W(l), W = (N/pi) sin(pi l/N).

    The (-1)^l term is the standard k_F = pi/2 parity oscillation of the
    half-filled chain (Calabrese-Essler-Fagotti, PRL 104, 095701 (2010));
    including it in the fit handles both parities of l at once.
    """
    N = N_CRIT
    ells = np.arange(40, N // 2 + 1)
    S = np.array([block_entropy(critical_C, range(l)) for l in ells])

    W = (N / np.pi) * np.sin(np.pi * ells / N)
    X = np.column_stack([np.log(W), np.ones_like(W), (-1.0) ** ells / W])
    beta, *_ = np.linalg.lstsq(X, S, rcond=None)
    c = 3.0 * beta[0]

    resid = S - X @ beta
    assert abs(c - 1.0) < 0.02, (
        f"measured central charge c = {c:.6f} (expected 1 within 0.02); "
        f"const = {beta[1]:.6f}, osc amp = {beta[2]:.2e}, "
        f"max |resid| = {np.max(np.abs(resid)):.2e}"
    )
    # the fit itself must be good, not just the slope
    assert np.max(np.abs(resid)) < 5e-3, (
        f"poor Calabrese-Cardy fit: max |resid| = {np.max(np.abs(resid)):.2e}"
    )
    # the non-universal constant is also exactly known for the half-filled XX
    # chain: c1 = ln(2)/3 + Upsilon_1 with Upsilon_1 = 0.4950179... (Jin-Korepin,
    # J. Stat. Phys. 116, 79 (2004))
    c1_exact = np.log(2.0) / 3.0 + 0.4950179
    assert abs(beta[1] - c1_exact) < 1e-3, (
        f"fitted constant {beta[1]:.6f} vs Jin-Korepin c1 = {c1_exact:.6f}"
    )


# ---------------------------------------------------------------------------
# 3. Entanglement-Hamiltonian consistency
# ---------------------------------------------------------------------------


def _entropy_from_eps(eps):
    """Free-fermion entropy from single-particle entanglement energies.

    S = sum_k [ln(1 + e^{-eps_k}) + eps_k / (e^{eps_k} + 1)], evaluated
    stably: logaddexp for the first term, eps * expit(-eps) for the second.
    """
    return float(np.sum(np.logaddexp(0.0, -eps) + eps * expit(-eps)))


@pytest.mark.parametrize("l", [4, 20, 40])
def test_entanglement_hamiltonian_consistency(critical_C, open8_C, l):
    """S from eigenvalues of h = ln((1-C_A)/C_A) matches block_entropy to 1e-9."""
    C = open8_C if l == 4 else critical_C
    sites = range(l)
    C_A = np.asarray(C)[np.ix_(sites, sites)]

    h = entanglement_hamiltonian(C_A)
    assert np.allclose(h, h.T, atol=1e-10), "h must be hermitian"

    eps = np.linalg.eigvalsh(h)
    S_h = _entropy_from_eps(eps)
    S_c = block_entropy(C, sites)
    assert abs(S_h - S_c) < 1e-9, (
        f"l={l}: free-fermion-formula entropy {S_h:.15f} != "
        f"block_entropy {S_c:.15f} (diff {S_h - S_c:.3e})"
    )


# ---------------------------------------------------------------------------
# Deliberate degeneracy handling
# ---------------------------------------------------------------------------


def test_degenerate_fermi_level_raises():
    """Periodic N=8 at half filling has zero modes at k = +/- pi/2: refuse it."""
    with pytest.raises(ValueError, match="degenerate"):
        tight_binding_C(8, filling=0.5, bc="periodic")
    # ...while periodic N = 2 mod 4 is fine (gapped Fermi level).
    C = tight_binding_C(10, filling=0.5, bc="periodic")
    assert abs(np.trace(C) - 5.0) < 1e-12
