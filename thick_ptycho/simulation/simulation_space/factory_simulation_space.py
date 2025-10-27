"""
Defines the physical space and scan path for a ptychographic simulation.
"""

from .simulation_space_1d import SimulationSpace1D
from .simulation_space_2d import SimulationSpace2D
from ..config import SimulationConfig

def create_simulation_space(config: SimulationConfig):
    """Factory function to create the appropriate simulation space class."""
    dim = len(config.continuous_dimensions) - 1
    cls = SimulationSpace1D if dim == 1 else SimulationSpace2D
    return cls(
        continuous_dimensions=config.continuous_dimensions,
        discrete_dimensions=config.discrete_dimensions,
        probe_dimensions=config.probe_dimensions,
        scan_points=config.scan_points,
        step_size=config.step_size,
        bc_type=config.bc_type.value,
        probe_type=config.probe_type.value,
        wave_number=config.wave_number,
        probe_diameter_scale=config.probe_diameter_scale,
        probe_focus=config.probe_focus,
        probe_angles=config.probe_angles,
        tomographic_projection_90_degree=config.tomographic_projection_90_degree,
        thin_sample=config.thin_sample,
        n_medium=config.n_medium,
        results_dir=config.results_dir,
        use_logging=config.use_logging,
        verbose=config.verbose,
    )
