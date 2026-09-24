"""Layer-1 validation anchor: the entanglement first law delta S = delta <K>.

For small perturbations of the global ground state of a harmonic chain, the
change in entanglement entropy of an interval equals the change in modular
energy computed with the UNPERTURBED interval's entanglement Hamiltonian
(Blanco-Casini-Hung-Myers JHEP 08 (2013) 060; Faulkner-Guica-Hartman-Myers-
Van Raamsdonk JHEP 03 (2014) 051). The two sides are computed by independent
routes (entropy differences vs (1/2) Tr[G dV_A] with G from Williamson), so
agreement validates vacuum.core.entanglement_hamiltonian — the construction
Layer 4 depends on.

Setup: N = 120 periodic chain, near-critical mass m = 0.005 (correlation
length 1/m = 200 >> N, so the interval carries genuine long-range
entanglement), interval of 24 sites. Perturbations: (a) a global mass shift
m^2 -> m^2 + eps; (b) a single-site mass bump outside and inside the
interval — the first law holds for ANY global perturbation.

Anchor tolerance: |dS - dK|/|dS| < 1e-3 after central differences +
Richardson extrapolation. Measured values (recorded in the assertion
messages) come in around 1e-10 — seven orders inside the anchor.
"""

import numpy as np
import pytest

from vacuum.core import (
    ground_state_cov,
    harmonic_chain_K,
    reduce,
    symplectic_eigenvalues,
)
from vacuum.modular import first_law_check

# --- configuration -----------------------------------------------------------
N = 120                          # chain length, PBC
M = 0.005                        # near-critical mass (window 1e-3 .. 0.1)
REGION = list(range(48, 72))     # 24-site interval, centered
SITE_OUT = 28                    # bump site 20 sites outside the interval
SITE_IN = 60                     # bump site in the middle of the interval
EPS_GLOBAL = 1e-7                # step for m^2 -> m^2 + eps (must be << m^2)
EPS_BUMP = 1e-5                  # step for single-site bumps

ANCHOR_TOL = 1e-3                # THE first-law anchor
FLOOR_FRAC_TOL = 1e-6            # clamped-mode contribution must sit far
                                 # below the anchor tolerance (measured ~1e-8)


def _global_mass_shift(K, e):
    """m^2 -> m^2 + e on every site: K -> K + e I."""
    return K + e * np.eye(K.shape[0])


def _site_bump(j):
    """Single-site mass bump: K_jj -> K_jj + e."""

    def perturb(K, e):
        Kp = K.copy()
        Kp[j, j] += e
        return Kp

    return perturb


# --- module-scoped fixtures (lattice build + checks cached once) -------------
@pytest.fixture(scope="module")
def chain_K():
    return harmonic_chain_K(N, M, bc="periodic")


@pytest.fixture(scope="module")
def region_nu(chain_K):
    return symplectic_eigenvalues(reduce(ground_state_cov(chain_K), REGION))


@pytest.fixture(scope="module")
def results(chain_K):
    """All three first-law checks, computed once for the whole module."""
    return {
        "global": first_law_check(chain_K, REGION, _global_mass_shift, EPS_GLOBAL),
        "bump_out": first_law_check(chain_K, REGION, _site_bump(SITE_OUT), EPS_BUMP),
        "bump_in": first_law_check(chain_K, REGION, _site_bump(SITE_IN), EPS_BUMP),
    }


# --- conditioning of the reduced state ---------------------------------------
def test_region_spectrum_conditioning(region_nu):
    """Several symplectic eigenvalues sit well above the vacuum floor.

    The interval entanglement spectrum decays roughly exponentially, so
    'several well above 1/2' means: the top two are O(1)-separated from the
    floor, and at least 3 (resp. 5) resolve the floor by 1e-2 (resp. 1e-4) —
    enough structure for a well-conditioned G on the modes that matter
    (measured spectrum: nu - 1/2 = 0.80, 0.16, 1.5e-2, 2.1e-3, 1.9e-4, ...).
    The deep tail sits at the floor and exercises the clamp (verified
    negligible in test_floor_mode_contribution_negligible, not assumed).
    """
    nu = region_nu
    assert nu[0] > 1.0, f"top symplectic eigenvalue {nu[0]:.4f} not well above 1/2"
    assert nu[1] > 0.6, f"second symplectic eigenvalue {nu[1]:.4f} not well above 1/2"
    n_1e2 = int(np.sum(nu - 0.5 > 1e-2))
    n_1e4 = int(np.sum(nu - 0.5 > 1e-4))
    assert n_1e2 >= 3, f"only {n_1e2} eigenvalues with nu - 1/2 > 1e-2 (want >= 3)"
    assert n_1e4 >= 5, f"only {n_1e4} eigenvalues with nu - 1/2 > 1e-4 (want >= 5)"
    # The clamp in entanglement_hamiltonian is actually exercised:
    n_floor = int(np.sum(nu - 0.5 < 1e-8))
    assert n_floor >= 1, "no vacuum-floor modes present; clamp untested"


# --- THE anchor: delta S = delta <K> -----------------------------------------
def test_first_law_global_mass_shift(results):
    r = results["global"]
    assert r["rel_err"] < ANCHOR_TOL, (
        f"first law violated for global mass shift: dS={r['dS']:.9e}, "
        f"dK={r['dK_modular']:.9e}, rel_err={r['rel_err']:.3e}"
    )
    # Physics sanity: raising the mass gaps the chain and cuts entanglement.
    assert r["dS"] < 0.0, f"dS/d(m^2) = {r['dS']:.3e} should be negative"
    # Record the achieved precision (typically ~1e-10, seven orders inside).
    assert r["rel_err"] < 1e-6, (
        f"achieved rel_err {r['rel_err']:.3e} regressed far above the "
        f"~1e-10 this configuration delivers"
    )


def test_first_law_site_bump_outside(results):
    r = results["bump_out"]
    assert r["rel_err"] < ANCHOR_TOL, (
        f"first law violated for site bump OUTSIDE region (site {SITE_OUT}): "
        f"dS={r['dS']:.9e}, dK={r['dK_modular']:.9e}, rel_err={r['rel_err']:.3e}"
    )
    assert r["rel_err"] < 1e-6, (
        f"achieved rel_err {r['rel_err']:.3e} regressed far above ~1e-10"
    )


def test_first_law_site_bump_inside(results):
    r = results["bump_in"]
    assert r["rel_err"] < ANCHOR_TOL, (
        f"first law violated for site bump INSIDE region (site {SITE_IN}): "
        f"dS={r['dS']:.9e}, dK={r['dK_modular']:.9e}, rel_err={r['rel_err']:.3e}"
    )
    assert r["rel_err"] < 1e-6, (
        f"achieved rel_err {r['rel_err']:.3e} regressed far above ~1e-10"
    )


# --- clamp verification: floor modes contribute negligibly -------------------
@pytest.mark.parametrize("key", ["global", "bump_out", "bump_in"])
def test_floor_mode_contribution_negligible(results, key):
    """Modes clamped at nu ~ 1/2 in G contribute negligibly to delta<K>.

    Verified, not assumed: the per-mode Williamson decomposition of dK is
    summed over the floor modes (nu - 1/2 < 1e-8) and compared to |dS|.
    """
    r = results[key]
    # The decomposition is exact: per-mode contributions resum to dK.
    gap = abs(np.sum(r["mode_contributions"]) - r["dK_modular"])
    scale = max(1.0, abs(r["dK_modular"]))
    assert gap < 1e-8 * scale, (
        f"[{key}] mode decomposition does not resum to dK: gap {gap:.3e}"
    )
    # Floor modes exist and their total contribution is far below the anchor.
    n_floor = int(np.sum(r["floor_mask"]))
    assert n_floor >= 1, f"[{key}] no floor modes flagged"
    floor_frac = abs(r["dK_floor_modes"]) / abs(r["dS"])
    assert floor_frac < FLOOR_FRAC_TOL, (
        f"[{key}] clamped floor modes contribute |dK_floor|/|dS| = "
        f"{floor_frac:.3e} (n_floor={n_floor}) — not negligible vs 1e-3 anchor"
    )


# --- API guard ---------------------------------------------------------------
def test_perturb_identity_guard(chain_K):
    """perturb(K, 0) != K must raise: the check would be meaningless."""
    with pytest.raises(ValueError, match="perturb"):
        first_law_check(
            chain_K, REGION, lambda K, e: K + (e + 1e-3) * np.eye(N), 1e-5
        )
