"""
Defines the physical space and scan path for a ptychographic simulation.
"""

from ..config import SimulationConfig
from .simulation_space_2d import SimulationSpace2D
from .simulation_space_3d import SimulationSpace3D


def create_simulation_space(config: SimulationConfig):
    """Factory function to create the appropriate simulation space class."""
    cls = SimulationSpace2D if config.spatial_limits.y is None else SimulationSpace3D
    return cls(
        wave_length=config.probe_config.wave_length,
        probe_diameter=config.probe_config.diameter,
        probe_type=config.probe_config.type,
        probe_focus=config.probe_config.focus,
        probe_angles=config.probe_config.tilts,
        spatial_limits=config.spatial_limits,
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
        verbose=config.verbose,
        exact_ref_coeff=config.exact_ref_coeff,
    )
