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
        wave_length=config.wave_length,
        probe_diameter=config.probe_diameter,
        continuous_dimensions=config.continuous_dimensions,
        probe_type=config.probe_type.value,
        probe_focus=config.probe_focus,
        probe_angles=config.probe_angles,
        scan_points=config.scan_points,
        step_size_px=config.step_size_px,
        pad_factor=config.pad_factor,
        solve_reduced_domain=config.solve_reduced_domain,
        points_per_wavelength=config.points_per_wavelength,
        nz=config.nz,
        tomographic_projection_90_degree=config.tomographic_projection_90_degree,
        medium=config.medium,
        results_dir=config.results_dir,
        use_logging=config.use_logging,
        verbose=config.verbose,
    )
