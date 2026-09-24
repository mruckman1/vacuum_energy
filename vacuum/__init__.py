"""The Vacuum Program — a covariance-matrix engine for vacuum entanglement.

Subpackages, by layer (see ``PLAN.md`` and ``docs/PLAN_LAYERS_2_3.md``):

===================== ===========================================================
:mod:`vacuum.core`     Gaussian covariance stack: symplectic algebra, Williamson,
                       channels, harmonic-chain and Srednicki radial models,
                       the precision-escalation ladder (Layer 0-1)
:mod:`vacuum.fermions` tight-binding correlation matrices, Peschel entanglement
                       Hamiltonians (Layer 1)
:mod:`vacuum.audits`   the standing audits — energy ledger, passivity, causality,
                       precision — run on every protocol in every layer (Layer 0)
:mod:`vacuum.protocol` the one protocol runner every Layer 2/3 pipeline goes
                       through, so the audit discipline is structural (M-I.4)
:mod:`vacuum.detectors`  UDW harvesting: kernels, perturbative and nonperturbative
                       engines, the Simidzija null, the M_vac/M_comm split,
                       imperfections (Layer 2)
:mod:`vacuum.qet`      quantum energy teleportation and strong local passivity
                       (Layer 3)
:mod:`vacuum.inequalities`  the QEI stress-tester and work statistics (Layer 3)
:mod:`vacuum.modular`  numerical modular Hamiltonians (Layer 4)
:mod:`vacuum.geometry` MERA, mutual-information geometry, HaPPY code, the
                       failure map and the toy-Jacobson residual (Layer 4, L2)
:mod:`vacuum.floquet`  time-modulated vacua: Floquet engine, Mathieu chart,
                       DCE, pump-enhanced harvesting (Layer 5)
:mod:`vacuum.opt`      the differentiable vacuum: JAX mirror, objectives,
                       optimizers, distillation, inverse design (Layer 6)
:mod:`vacuum.composite`  harvest-then-teleport composite ledger (Layer 7, L4)
:mod:`vacuum.interacting`  lambda-phi^4 vacua via DMRG/TEBD (Layer 7, L5)
:mod:`vacuum.hardware` qubit-encoded vacuum on Aer, noise survival (Layer 7, L6)
:mod:`vacuum.experiments_io`  the ``experiments/`` loader: provenance enforcement
                       and the ONE place published units become hbar = 1
                       lattice units (M-I.3)
===================== ===========================================================

Conventions in force everywhere (``docs/API.md``, binding): hbar = 1; vacuum
covariance V_vac = I/2 with symplectic eigenvalues nu >= 1/2; block quadrature
ordering R = (x_1..x_N, p_1..p_N) with Omega = [[0, I], [-I, 0]]; all entropies
and negativities in nats.

Subpackages resolve lazily on attribute access (PEP 562), so ``import vacuum``
stays cheap and importing one layer never drags in the others::

    import vacuum
    K = vacuum.core.harmonic_chain_K(64, m=1e-3)

Explicit imports (``from vacuum.core import harmonic_chain_K``) work as usual
and remain the house style.
"""

__version__ = "0.1.0"

#: Subpackages and modules served by the lazy ``__getattr__`` below.
_SUBMODULES = frozenset({
    "audits",
    "composite",
    "core",
    "detectors",
    "experiments_io",
    "fermions",
    "floquet",
    "geometry",
    "hardware",
    "inequalities",
    "interacting",
    "modular",
    "opt",
    "protocol",
    "qet",
})

__all__ = ["__version__", *sorted(_SUBMODULES)]


def __getattr__(name):
    """PEP 562 lazy subpackage import."""
    if name in _SUBMODULES:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)
