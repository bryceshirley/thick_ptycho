"""
Defines the physical space and scan path for a ptychographic simulation.
"""

import dataclasses
from .simulation_space_1d import SimulationSpace1D
from .simulation_space_2d import SimulationSpace2D
from ..config import SimulationConfig

def create_simulation_space(config: SimulationConfig):
    """Factory function to create the appropriate simulation space class."""
    cls = SimulationSpace1D if config.spatial_limits.y is None else SimulationSpace2D
    return cls(wave_length=config.wave_length,
        probe_diameter=config.probe_diameter,
        spatial_limits=config.spatial_limits,
        probe_type=config.probe_type,
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
        scan_path=config.scan_path,
        results_dir=config.results_dir,
        use_logging=config.use_logging,
        verbose=config.verbose)

