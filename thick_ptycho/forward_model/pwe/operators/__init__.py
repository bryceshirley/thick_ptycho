"""
Finite-difference implementation of the Plane Wave Expansion (PWE) solver.
Contains boundary condition handling and test utilities.
"""

from .finite_differences.pwe_forward_model import PWEForwardModel
from .finite_differences.boundary_conditions import BoundaryConditions

__all__ = [
    "PWEForwardModel",
    "BoundaryConditions",
]
