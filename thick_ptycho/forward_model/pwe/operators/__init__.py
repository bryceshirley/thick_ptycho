"""
Finite-difference implementation of the Plane Wave Expansion (PWE) solver.
Contains boundary condition handling and test utilities.
"""

from .finite_differences.forward_model import PWEForwardModel
from .utils import BoundaryType

__all__ = [
    "PWEForwardModel",
    "BoundaryType",
]
