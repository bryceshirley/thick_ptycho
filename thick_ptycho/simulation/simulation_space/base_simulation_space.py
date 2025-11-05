from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import List, Tuple, Union, Optional
import numpy as np

from thick_ptycho.utils.io import setup_log


@dataclass
class ScanFrame:
    """Represents one scan frame in a ptychographic scan."""
    probe_centre_continuous: Union[float, Tuple[float, float]]
    probe_centre_discrete: Union[int, Tuple[int, int]]
    sub_limits_continuous: Union[
        Tuple[float, float],
        Tuple[Tuple[float, float], Tuple[float, float]]
    ]
    sub_limits_discrete: Union[
        Tuple[int, int],
        Tuple[Tuple[int, int], Tuple[int, int]]
    ]


class BaseSimulationSpace(ABC):
    """Abstract base defining shared simulation-space logic."""

    def __init__(
        self,
        continuous_dimensions: List[Tuple[float, float]],
        discrete_dimensions: List[int],
        probe_dimensions: List[int],
        scan_points: int,
        step_size: float,
        bc_type: str,
        probe_type: str,
        wave_number: float,
        probe_diameter_scale: Optional[float] = None,
        probe_focus: Optional[float] = None,
        probe_angles: Optional[Tuple[float, ...]] = (0.0,),
        tomographic_projection_90_degree: bool = False,
        thin_sample: bool = False,
        n_medium: Union[float, complex] = 1.0,
        results_dir: Optional[str] = None,
        use_logging: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the simulation space.

        Parameters
        ----------
        continuous_dimensions : list of tuple
            Continuous spatial limits in each dimension, e.g. [(x_min, x_max), (z_min, z_max)].
        discrete_dimensions : list of int
            Number of discrete points (pixels) in each dimension.
        probe_dimensions : list of int
            Probe shape in pixels.
        scan_points : int
            Number of scan points along one axis.
        step_size : float
            Step size between scan points (pixels).
        bc_type : str
            Boundary condition type ('dirichlet', 'neumann', etc.).
        probe_type : str
            Type of probe (e.g. Gaussian, plane wave).
        wave_number : float
            Wavenumber (1/nm).
        probe_diameter_scale : float, optional
            Fraction of total x-range representing probe diameter.
        probe_focus : float, optional
            Probe focal distance.
        probe_angles : tuple of float, optional
            Incident probe angles.
        tomographic_projection_90_degree : bool, default=False
            If True, adds an orthogonal projection (requires cubic grid).
        thin_sample : bool, default=False
            If True, assumes thin-sample propagation.
        n_medium : float or complex, default=1.0
            Background refractive index.
        results_dir : str, optional
            Directory for logging and results.
        use_logging : bool, default=True
            Enable file logging.
        verbose : bool, default=False
            Enable console verbosity.
        """

        # ------------------------------------------------------------------
        # 1. Validation and core dimension setup
        # ------------------------------------------------------------------
        self._validate_dimensions(continuous_dimensions, discrete_dimensions)
        self.continuous_dimensions = continuous_dimensions
        self.discrete_dimensions = discrete_dimensions
        self.thin_sample = thin_sample

        # Effective (in-plane) dimensions
        self.effective_dimensions = (
            probe_dimensions if thin_sample else discrete_dimensions[:-1]
        )

        # ------------------------------------------------------------------
        # 2. Probe configuration
        # ------------------------------------------------------------------
        self.probe_dimensions = probe_dimensions
        self.probe_type = probe_type
        self.probe_focus = probe_focus
        self.probe_angles = probe_angles
        self.num_angles = len(self.probe_angles)

        self._setup_probe_diameter(probe_diameter_scale)

        # ------------------------------------------------------------------
        # 3. Scanning setup
        # ------------------------------------------------------------------
        self.scan_points = scan_points
        self.step_size = step_size
        self.total_step = (self.scan_points - 1) * self.step_size + self.probe_dimensions[0]
        
        # ------------------------------------------------------------------
        # 4. Tomography configuration
        # ------------------------------------------------------------------
        self.num_projections = self._determine_num_projections(
            discrete_dimensions, tomographic_projection_90_degree
        )

        # ------------------------------------------------------------------
        # 5. Physical constants
        # ------------------------------------------------------------------
        self.k = float(wave_number)
        self.wavelength = 2 * np.pi / self.k
        self.n_medium = complex(n_medium)

        # ------------------------------------------------------------------
        # 6. Spatial grid setup (z-axis)
        # ------------------------------------------------------------------
        self._setup_z_dimension()

        # ------------------------------------------------------------------
        # 7. Logging and results management
        # ------------------------------------------------------------------
        self.results_dir = results_dir
        self._log = setup_log(
            results_dir,
            log_file_name="simulation_space_log.txt",
            use_logging=use_logging,
            verbose=verbose,
        )
        self.verbose = verbose

        # ------------------------------------------------------------------
        # 8. Boundary conditions and scan frame info
        # ------------------------------------------------------------------
        self.bc_type = bc_type.lower()
        self._scan_frame_info: List[ScanFrame] = []

        self.total_scans = self.num_angles * self.scan_points * self.num_projections


    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------

    def join_results_dir(self, filename: str) -> str:
        """Join filename with results directory."""
                
        if self.results_dir is None:
            raise ValueError("results_dir is not set.")
        return os.path.join(self.results_dir, filename)

    def _validate_dimensions(
        self,
        continuous_dimensions: List[Tuple[float, float]],
        discrete_dimensions: List[int],
    ) -> None:
        """Validate consistency between continuous and discrete dimensions."""
        if len(continuous_dimensions) != len(discrete_dimensions):
            raise ValueError(
                "continuous_dimensions and discrete_dimensions must have the same length."
            )
        for i, (start, end) in enumerate(continuous_dimensions):
            if start >= end:
                raise ValueError(f"Invalid range for dimension {i}: start >= end.")

    def _setup_probe_diameter(self, probe_diameter_scale: Optional[float]) -> None:
        """Compute probe diameter in discrete and continuous units."""
        nx = self.discrete_dimensions[0]
        x_start, x_end = self.continuous_dimensions[0]
        x_range = x_end - x_start
        dx = x_range / nx

        if probe_diameter_scale is None:
            self.probe_diameter_discrete = self.probe_dimensions[0]
            self.probe_diameter_continuous = self.probe_diameter_discrete * dx
        else:
            if not (0.0 < probe_diameter_scale < 1.0):
                raise ValueError("probe_diameter_scale must be between 0 and 1.")
            self.probe_diameter_continuous = probe_diameter_scale * x_range
            self.probe_diameter_discrete = int(probe_diameter_scale * nx)

    def _determine_num_projections(
        self, discrete_dimensions: List[int], tomo_flag: bool
    ) -> int:
        """Determine number of tomographic projections."""
        if not tomo_flag:
            return 1
        if len(set(discrete_dimensions)) == 1:
            return 2
        print(
            "Warning: 90° tomographic projection requires cubic dimensions; "
            "proceeding with num_projections=1."
        )
        return 1

    def _setup_z_dimension(self) -> None:
        """Initialize z-axis grid and spacing."""
        self.nz = self.discrete_dimensions[-1]
        z_min, z_max = self.continuous_dimensions[-1]
        self.z = np.linspace(z_min, z_max, self.nz)
        self.dz = self.z[1] - self.z[0]
        self.zlims = (z_min, z_max)

    # ----------------------------------------------------------------------
    # Abstract and utility methods
    # ----------------------------------------------------------------------

    @property
    def scan_frame_info(self) -> List[ScanFrame]:
        """List of scan frame objects corresponding to each scan position."""
        return self._scan_frame_info

    @abstractmethod
    def summarize(self) -> None:
        """Summarize simulation parameters and key geometry."""
        raise NotImplementedError

    @abstractmethod
    def _generate_scan_frames(self) -> List[ScanFrame]:
        """Generate scan frame information."""
        raise NotImplementedError

    def update_nz(self, nz: int) -> None:
        """Update the number of z-slices and recompute z-axis spacing."""
        dims = list(self.discrete_dimensions)
        dims[-1] = nz
        self.discrete_dimensions = tuple(dims)
        self._setup_z_dimension()
        self.summarize()
