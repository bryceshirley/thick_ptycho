"""
Finite-difference implementation of the Plane Wave Expansion (PWE) solver.
Contains boundary condition handling and test utilities.
"""

from .pwe_finite_differences import PWEFiniteDifferences
from .boundary_conditions import BoundaryConditions

__all__ = [
    "PWEFiniteDifferences",
    "BoundaryConditions",
]
