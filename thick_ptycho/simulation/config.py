"""
Configuration definitions for ptychographic simulations.

This module provides typed configuration structures and enumerations
used throughout the simulation package, including probe field types,
simulation parameters, and logging options.
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import warnings

from enum import Enum

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

@dataclass(frozen=True,slots=True)
class SimulationConfig:
    """
    Configuration object defining the spatial discretization, probe geometry,
    and optical parameters used for ptychographic simulation.

    Parameters
    ----------
    probe_type : ProbeType, default=ProbeType.AIRY_DISK
        Selects the illumination probe field model.

    wave_length : float
        Illumination wavelength in meters.

    probe_diameter : float
        Effective diameter or support size of the probe in meters (before
        discretization into pixel units).

    probe_focus : float, optional
        Focal length used for curved probe illumination. ``0.0`` corresponds
        to a planar probe.

    probe_angles : tuple of float, default=(0.0,)
        List of probe tilt angles in degrees. Multiple angles enables
        tilt-series or tomographic ptychography.

    scan_points : int
        Number of discrete probe scan positions in the lateral direction.

    step_size_px : float
        Step size between probe positions, expressed in pixels.

    pad_factor : float, default=1.1
        Controls padding beyond the region required to contain all scan
        positions. Must satisfy ``pad_factor >= 1.0``.

        Let ``min_nx`` denote the minimum required discrete domain width
        to place all probe positions without clipping::

            min_nx = (scan_points - 1) * step_size_px + probe_diameter_px

        The total simulation domain width (including padding) is::

            Nx = int(pad_factor * min_nx)

        When ``pad_factor > 1.0``, padding is symmetric::

            Ne = Nx - min_nx              # Total padding width
            pL = pR = Ne / 2              # Left and right padding

    solve_reduced_domain : bool, default=False
        If ``True``, the simulation or reconstruction is performed only on the
        reduced domain of width ``min_nx`` that directly contains the scanned
        region. This reduces memory and computation but removes the padding
        region ``Ne`` and may introduce boundary effects. If ``False``,
        the full padded domain of width ``Nx`` is used.

    points_per_wavelength : int, default=8
        Number of grid points per wavelength along the propagation axis.
        Determines axial discretization::

            dz = wave_length / points_per_wavelength
            nz = int((zlims[1] - zlims[0]) / dz)

    nz : int, optional
        Explicit override for the number of discretization points along z.
        If provided, it replaces the computed value.

    continuous_dimensions : Tuple[Tuple[float, float], ...]
        xlims, zlims or xlims, ylims, zlims
        Continuous spatial extent of the domain in meters.

    tomographic_projection_90_degree : bool, default=False
        Enables the 90° projection transform used in 2D multislice tomography.
        Valid only when ``Nx == nz``.

    medium : float, default=1.0
        Background refractive index of the propagation medium.

    results_dir : str, optional
        Output directory for storing intermediate and final results.

    use_logging : bool, default=True
        Enable runtime logging and progress reporting.

    verbose : bool, default=False
        Enable runtime printing and progress reporting.
    """
    wave_length: float
    probe_diameter: float
    continuous_dimensions: Tuple[Tuple[float, float], ...]
    scan_points: int = 1 # number of scan points in one dimension greater than 0
    step_size_px: int = 1 # in pixels greater than 0 for more than 1 scan point
    probe_type: ProbeType = ProbeType.AIRY_DISK
    probe_focus: Optional[float] = 0.0
    probe_angles: Tuple[float, ...] = (0.0,)
    pad_factor: float = 1.1 # Must be >= 1.0
    solve_reduced_domain: bool = False
    points_per_wavelength: Optional[int] = 8
    nz: Optional[int] = None 
    tomographic_projection_90_degree: Optional[bool] = False
    medium: float = 1.0 # 1.0 for free space
    results_dir: Optional[str] = None
    use_logging: bool = True
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # If only 1 scan point, reduced-domain setting has no effect → warn, don't fail.
        if self.scan_points == 1 and self.solve_reduced_domain:
            warnings.warn(
                "solve_reduced_domain=True has no effect when scan_points=1; using full domain.",
                RuntimeWarning,
            )
        if not isinstance(self.continuous_dimensions, tuple) or len(self.continuous_dimensions) < 2:
            raise ValueError("continuous_dimensions must be a tuple of at least two tuples (xlims, zlims).")
        if self.probe_type not in ProbeType:
            raise ValueError(f"Invalid probe type: {self.probe_type}. Must be one of {ProbeType.list()}.")
        if self.pad_factor < 1.0:
            raise ValueError("pad_factor must be >= 1.0")
        if self.scan_points < 1:
            raise ValueError("scan_points must be greater than 0")
        if self.step_size_px < 1 and self.scan_points > 1:
            raise ValueError("step_size_px must be greater than 0 for multiple scan points")
        if self.points_per_wavelength < 1:
            raise ValueError("points_per_wavelength must be >= 1")
        if len(self.continuous_dimensions) != 2 and len(self.continuous_dimensions) != 3:
            raise ValueError("continuous_dimensions must have 2 or 3 dimensions.")
        if self.probe_diameter <= 0:
            raise ValueError("probe_diameter must be positive.")
        for i, (start, end) in enumerate(self.continuous_dimensions):
            if start >= end:
                raise ValueError(f"Invalid range for dimension {i}: start >= end.")
            if i < len(self.continuous_dimensions) - 1:  # only check spatial dimensions
                if self.probe_diameter > end - start:
                    raise ValueError(
                        f"Probe diameter must be smaller than the range of dimension {i}."
                    )
        if self.nz is not None and self.nz <= 0:
            raise ValueError("nz must be a positive integer if specified.")
        if self.results_dir is not None and not isinstance(self.results_dir, str):
            raise ValueError("results_dir must be a string if specified.")
        if self.use_logging and not isinstance(self.use_logging, bool):
            raise ValueError("use_logging must be a boolean value.")
        if self.verbose and not isinstance(self.verbose, bool):
            raise ValueError("verbose must be a boolean value.")
        if self.tomographic_projection_90_degree and len(self.continuous_dimensions) != 2:
            raise ValueError("tomographic_projection_90_degree is only valid for 2D simulations.")
        if self.wave_length <= 0:
            raise ValueError("wave_length must be a positive value.")
        if self.points_per_wavelength is None and self.nz is None:
            raise ValueError("Either points_per_wavelength or nz must be specified.")
        if self.points_per_wavelength is not None and self.points_per_wavelength <= 0:
            raise ValueError("points_per_wavelength must be a positive if specified.")