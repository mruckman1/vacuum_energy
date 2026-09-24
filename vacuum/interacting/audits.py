"""Standing audits on the interacting MPS ground state (Layer 0 discipline).

Passivity (Pusz-Woronowicz, Commun. Math. Phys. 58, 273 (1978)): the ground
state of any Hamiltonian is passive, so no unitary — in particular no
*local* unitary — can lower <H>. The Gaussian audit in
``vacuum.audits.passivity`` throws random local symplectics at a covariance
matrix; the non-Gaussian analogue here throws random single-site and
two-site unitaries U = exp(-i G) (G a random Hermitian matrix on the local
truncated space, GUE-scaled by `strength`) at the DMRG ground state and
records the worst energy change <H>_{U psi} - <H>_psi. Every change must be
>= -tol, where tol absorbs (i) the DMRG's own distance above the true
ground energy of H_trunc (a local unitary keeps the MPS exactly
representable, so E(U psi) >= E_true >= E_DMRG - delta with delta ~ the
truncation error) and (ii) float roundoff. As in the Gaussian audit, an
audit that can never fire is untrustworthy: ``passivity_audit_mps`` run on
an *active* state (e.g. the vacuum after a local squeeze, with the sampler
seeded to include the inverse squeeze) must report the violation, and the
tests check that too.

Chi-doubling: ``relative_change`` is the one-line gate every reported
number goes through (|a - b| / max(|a|, |b|, floor) < 1%).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .phi4 import _tenpy, mps_energy

__all__ = ["random_local_unitary", "apply_local_unitary", "passivity_audit_mps",
           "relative_change"]


def random_local_unitary(rng, d, n_sites=1, strength=1.0, real=False):
    """U = exp(-i G) on n_sites adjacent sites, G random Hermitian (GUE-like).

    real=True draws a real antisymmetric generator instead (U real
    orthogonal), the subgroup that keeps a real MPS real.
    """
    D = int(d) ** int(n_sites)
    if real:
        A = rng.standard_normal((D, D))
        G = (A - A.T) / np.sqrt(2.0 * D)
        return expm(float(strength) * G)
    A = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
    G = (A + A.conj().T) / np.sqrt(4.0 * D)
    return expm(-1j * float(strength) * G)


def apply_local_unitary(psi, site, U, n_sites=1):
    """Copy of `psi` with the dense (d^n x d^n) unitary applied on sites
    site..site+n_sites-1 (n_sites in {1, 2})."""
    t = _tenpy()
    out = psi.copy()
    site = int(site)
    d = psi.sites[site].dim
    leg = psi.sites[site].leg
    if n_sites == 1:
        op = t.npc.Array.from_ndarray(np.asarray(U).reshape(d, d), [leg, leg.conj()],
                                      labels=["p", "p*"])
    elif n_sites == 2:
        leg2 = psi.sites[site + 1].leg
        op = t.npc.Array.from_ndarray(np.asarray(U).reshape(d, d, d, d),
                                      [leg, leg2, leg.conj(), leg2.conj()],
                                      labels=["p0", "p1", "p0*", "p1*"])
    else:
        raise ValueError("n_sites must be 1 or 2")
    out.apply_local_op(site, op, unitary=True)
    return out


def passivity_audit_mps(psi, model, n_samples=16, max_sites=2, strength=1.0,
                        seed=0, tol=1e-8, sampler=None):
    """Random local unitaries never lower <H> on a ground state.

    Returns a dict: 'passed', 'worst' (most negative energy change),
    'changes' (all sampled changes), 'E_ref', 'tol'. `sampler(rng, k)` may
    replace the random draw (returns (site, n_sites, U)); the default draws
    a site uniformly and n_sites in {1..max_sites}.
    """
    rng = np.random.default_rng(seed)
    L = psi.L
    d = psi.sites[0].dim
    E_ref = mps_energy(psi, model)
    changes = []
    for k in range(int(n_samples)):
        if sampler is not None:
            site, n_sites, U = sampler(rng, k)
        else:
            n_sites = int(rng.integers(1, int(max_sites) + 1))
            site = int(rng.integers(0, L - n_sites + 1))
            U = random_local_unitary(rng, d, n_sites=n_sites, strength=strength)
        E = mps_energy(apply_local_unitary(psi, site, U, n_sites=n_sites), model)
        changes.append(E - E_ref)
    changes = np.asarray(changes)
    worst = float(np.min(changes)) if changes.size else 0.0
    return {"passed": bool(worst >= -float(tol)), "worst": worst,
            "changes": changes, "E_ref": E_ref, "tol": float(tol)}


def relative_change(a, b, floor=1e-12):
    """|a - b| / max(|a|, |b|, floor): the chi-doubling gate."""
    a = float(a)
    b = float(b)
    return abs(a - b) / max(abs(a), abs(b), float(floor))
