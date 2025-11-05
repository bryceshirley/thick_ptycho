"""
Configuration definitions for ptychographic simulations.

This module provides typed configuration structures and enumerations
used throughout the simulation package, including boundary conditions,
probe field types, and general simulation parameters.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

from enum import Enum

class BoundaryType(Enum):
    """
    Supported boundary condition types for the simulation domain.
    Not required for Multislice Solver.
    """
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    IMPEDANCE = "impedance"
    IMPEDANCE2 = "impedance2"

    @classmethod
    def list(cls):
        """Return a list of all boundary condition names."""
        return [bc.value for bc in cls]

class ProbeType(Enum):
    """Enumeration of supported probe field types."""
    CONSTANT = "constant"
    GAUSSIAN = "gaussian"
    SINUSOIDAL = "sinusoidal"
    COMPLEX_EXP = "complex_exp"
    DIRICHLET_TEST = "dirichlet_test"
    NEUMANN_TEST = "neumann_test"
    AIRY_DISK = "airy_disk"
    DISK = "disk"
    BLURRED_DISK = "blurred_disk"

    @classmethod
    def list(cls):
        """Return a list of all probe type names."""
        return [p.value for p in cls]

@dataclass
class SimulationConfig:
    """
    Configuration object for defining the simulation space and probe parameters.

    Attributes
    ----------
    continuous_dimensions : tuple
        Coordinate arrays defining the simulation domain.
    discrete_dimensions : tuple of int
        Number of discrete grid points in each spatial dimension.
    probe_dimensions : int
        Dimensions of the probe region pixels.
    scan_points : int
        Number of scan positions in one dimension.
    step_size : float
        Step size between scan positions.
    bc_type : BoundaryType
        Boundary condition type for the simulation domain.
        Not required for Multislice Solver.
    probe_type : ProbeType
        Type of probe field to use.
    wave_number : float
        Optical wavenumber (2π / λ).
    probe_diameter_scale : float, optional
        Diameter of the probe aperture.
    probe_focus : float, optional
        Focal length of the probe.
    probe_angles : Tuple[float, ...]
        Tuple of probe tilt angle(s) in degrees.
    tomographic_projection_90_degree : bool, optional
        Whether to use 90-degree tomographic projection. 
        Only possible for 2D simulations when nx == nz.
    thin_sample : bool, default=False
        Whether to use a thin-sample approximation.
    n_medium : float, default=1.0
        Refractive index of the propagation medium.
    results_dir : str, optional
        Output directory for simulation results.
    use_logging : bool, default=True
        Whether to enable logging.
    verbose : bool, default=False
        Whether to print additional debug information.
    """
    continuous_dimensions: Tuple
    discrete_dimensions: Tuple[int, ...]
    probe_dimensions: int
    scan_points: int
    step_size: float
    wave_number: float
    bc_type: BoundaryType = BoundaryType.IMPEDANCE
    probe_type: ProbeType = ProbeType.AIRY_DISK
    probe_diameter_scale: Optional[float] = None
    probe_focus: Optional[float] = 0.0
    probe_angles: Tuple[float, ...] = (0.0,)
    tomographic_projection_90_degree: Optional[bool] = False
    thin_sample: bool = False
    n_medium: float = 1.0
    results_dir: Optional[str] = None
    use_logging: bool = True
    verbose: bool = False