"""
Defines the physical space and scan path for a ptychographic simulation.
"""

from .simulation_space_1d import SimulationSpace1D
from .simulation_space_2d import SimulationSpace2D
from .config import SimulationConfig

def create_simulation_space(config: SimulationConfig):
    """Factory function to create the appropriate simulation space class."""
    dim = len(config.continuous_dimensions) - 1
    cls = SimulationSpace1D if dim == 1 else SimulationSpace2D
    return cls(**vars(config))
