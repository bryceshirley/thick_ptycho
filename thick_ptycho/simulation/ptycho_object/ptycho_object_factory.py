"""
Factory for creating PtychoObject1D or PtychoObject2D based on simulation space.
"""
from .ptycho_object_1d import PtychoObject1D
from .ptycho_object_2d import PtychoObject2D


class PtychoObjectFactory:
    """Factory that returns the appropriate ptychographic object type."""

    @staticmethod
    def create(simulation_space):
        if hasattr(simulation_space, "y"):
            return PtychoObject2D(simulation_space)
        return PtychoObject1D(simulation_space)
