"""Fermionic correlation-matrix methods (Peschel) — Layer 1 of the Vacuum Program.

Free-fermion ground states, block entanglement entropies and single-particle
entanglement Hamiltonians from two-point correlation matrices. See
vacuum.fermions.correlation and docs/API.md.
"""

from vacuum.fermions.correlation import (
    block_entropy,
    entanglement_hamiltonian,
    tight_binding_C,
)

__all__ = [
    "tight_binding_C",
    "block_entropy",
    "entanglement_hamiltonian",
]
