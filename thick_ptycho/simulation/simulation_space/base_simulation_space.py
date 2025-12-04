"""
Discretization of the Simulation Domain (1D Example)
===================================================

This section explains how the discrete simulation grid is constructed
from the scanning parameters. The same principles generalize to 2D and 3D.

Overview
--------
A 1D sample domain is discretized into Nx points with spacing ``dx``.
A probe scans the sample at scan_points (or in diagram ``n``) positions, 
separated by step_size_px (or in diagram ``s``) pixels.
To ensure the full probe footprint fits inside the simulated region at
each position, the domain must be padded on both sides.

Diagram
-------
    x = 0                                                                  x = Nx * dx
     ____________________________________________________________________________
     |<-- pL -->|<--  s  -->|<--  s  -->|   ...  |<-- s -->|<-- s -->|<-- pR -->|
                      c₁          c₂                  cₙ₋₁       cₙ              
                      |<--  s  -->|                                             
                 |<-- pL -->|<--  s  -->|<--  pR  -->|                          
     |<------------  Ne  --------------->|                                       
                |<------------------------ min_nx ------------------>|           
     |<------------------------------        Nx       -------------------------->|
Where:
    cᵢ  : Scan point centers (pixel indices)
    s   : Step size between consecutive scan points (pixels)
    n   : Number of scan positions
    min_nx : Minimum required simulation width to contain all scan positions
    Nx  : Total number of grid points in the simulation domain (including padding)
    pL, pR : Padding widths on the left and right sides (pL = pR)
    Ne  : Effective padding region width (pL + pR)

Key Quantities
--------------
step_size_px : int
    Step size between scan positions, measured in pixels.

scan_points : int
    Number of probe positions along the scan line.

min_nx : int
    Minimum required simulation width to contain all scan positions.
    Computed as::
        min_nx = scan_points * step_size_px

pad_factor : float, >= 1.0
    Controls how much total padding to add around the scanned region.
    A value of ``1.0`` means no padding; larger values expand the domain.

Nx : int
    Total number of grid points in the simulation domain, including padding.
    Computed as::

        Nx = int(pad_factor * min_nx)

Ne : int
    Effective . Used when solving only a reduced "effective" domain instead 
    of the full padded space::
        padding = Nx - min_nx = (pad_factor - 1) * Nx
        Ne = step_size + padding

dx : float
    Spatial step in meters (physical pixel size).

Notes
-----
- A larger `pad_factor` reduces boundary artifacts and improves numerical stability,
  but increases memory and computational cost.
- Increasing `step_size_px` increases the resolution of the scan.
- When performing iterative reconstruction methods (e.g., PWE or MS),
  the effective domain width ``Ne`` may be used to accelerate computation considering
  a smaller region of interest.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from thick_ptycho.simulation.config import ProbeType
from thick_ptycho.simulation.scan_frame import (Limits, ScanFrame,
                                                ScanPath)
from thick_ptycho.utils.io import setup_log


class BaseSimulationSpace(ABC):
    """Abstract base defining shared simulation-space logic."""

    def __init__(
        self,
        wave_length: float,
        probe_diameter: float,
        spatial_limits: Limits,
        probe_type: ProbeType = ProbeType.AIRY_DISK,
        probe_focus: Optional[float] = 0.0,
        probe_angles: Tuple[float, ...] = (0.0,),
        scan_points: int = 1, 
        step_size_px: int = 10, 
        pad_factor: float = 1, 
        solve_reduced_domain: bool = False,
        points_per_wavelength: int = 8,
        nz: Optional[int] = None,
        tomographic_projection_90_degree: Optional[bool] = False,
        medium: float = 1.0,  # 1.0 for free space
        results_dir: Optional[str] = None,
        scan_path: Optional[ScanPath] = None,
        use_logging: bool = True,
        verbose: bool = False
    ) -> None:
        """
        Initialize the base simulation space.
        Parameters
        ----------
        wave_length : float
            Wavelength of the illuminating wave in meters.
        probe_diameter : float
            Diameter of the probe in meters.
        spatial_limits : Limits
            Continuous spatial extent of the domain in meters.
        probe_type : string, optional
            Type of probe illumination.
        probe_focus : float, optional
            Focal position of the probe along z (meters).
        probe_angles : tuple of float, optional
            Angles of incidence for the probe (degrees).
        scan_points : int, optional
            Number of scan points in one dimension greater than 0.
        step_size_px : int, optional
            Step size between scan points in pixels greater than 0 for more than 1 scan point.
        pad_factor : float, optional
            Padding factor >= 1.0 to expand the simulation domain.
        solve_reduced_domain : bool, optional
            Whether to solve only the reduced effective domain.
        points_per_wavelength : int, optional
            Number of grid points per wavelength along the propagation axis.
        nz : int, optional
            Explicit override for the number of discretization points along z.
        tomographic_projection_90_degree : bool, optional
            Enables the 90° projection transform used in 2D multislice tomography.
        medium : float, optional
            Background refractive index of the propagation medium.
        results_dir : str, optional
            Output directory for storing intermediate and final results.
        use_logging : bool, optional
            Enable runtime logging and progress reporting.
        verbose : bool, optional
            Print additional debugging and diagnostic information.
        """
        self.wave_length = wave_length
        self.spatial_limits = spatial_limits

        # Scan setup
        self.scan_points = scan_points
        self.step_size = int(step_size_px)
        self.pad_factor = pad_factor
        self.solve_reduced_domain = solve_reduced_domain
        self._scan_frame_info: List[ScanFrame] = []
        self.tomographic_projection_90_degree = tomographic_projection_90_degree

        # Configure simulation geometry
        self._configure_z_axis(points_per_wavelength, nz)
        self._configure_domain()
        self._configure_probe(probe_diameter, probe_type.value, probe_focus, probe_angles)
        self.shape = None  # To be defined in subclass

        # Physical medium
        self.n_medium = complex(medium)

        # Logging
        self.results_dir = results_dir
        self._log = setup_log(
            results_dir,
            log_file_name="simulation_space_log.txt",
            use_logging=use_logging,
            verbose=verbose,
        )
        self.verbose = verbose


    def _configure_z_axis(self, points_per_wavelength: int, nz: Optional[int]) -> None:
        """Set up the propagation axis."""
        if nz is None:
            z_range = self.spatial_limits.z[1] - self.spatial_limits.z[0]
            self.dz = self.wave_length / points_per_wavelength
            self.nz = int(z_range / self.dz)
        else:
            self.nz = nz

        self.z = np.linspace(*self.spatial_limits.z, self.nz)
        self.dz = self.z[1] - self.z[0]


    def _configure_domain(self) -> None:
        """Compute domain width, apply padding, enforce symmetry, set coordinates."""

        self.min_nx = self.scan_points * self.step_size
        self.nx = int(self.pad_factor * self.min_nx)
        self.pad_discrete = self.nx - self.min_nx

        # Ensure symmetric padding (even)
        if self.pad_discrete % 2 != 0:
            self.nx += 1
            self.pad_discrete = self.nx - self.min_nx

        # Ensure scan symmetry
        desired_parity = 1 if (self.step_size % 2 == 0) else 0
        if self.nx % 2 != desired_parity:
            self.nx += 1
            self.pad_discrete = self.nx - self.min_nx

        self.edge_margin = self.pad_discrete // 2

        # Effective region Ne
        if self.solve_reduced_domain and self.scan_points > 1:
            self.effective_nx = self.pad_discrete + self.step_size - 1
        else:
            self.effective_nx = self.nx

        # Coordinates
        self.x = np.linspace(self.spatial_limits.x[0], 
                             self.spatial_limits.x[1], 
                             self.nx)
        self.dx = self.x[1] - self.x[0]

    def _configure_probe(
        self,
        probe_diameter: float,
        probe_type: str,
        probe_focus: float,
        probe_angles: Tuple[float, ...],
    ) -> None:
        """Store probe configuration & derived quantities."""
        if probe_diameter is not None:
            self.probe_diameter = probe_diameter
            self.probe_diameter_pixels = int(probe_diameter / self.dx)
        else:
            self.probe_diameter = None
        self.probe_type = probe_type
        self.probe_focus = probe_focus
        self.probe_angles = probe_angles
        self.num_angles = len(probe_angles)

        self.k = 2 * np.pi / self.wave_length


    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------

    def join_results_dir(self, filename: str) -> str:
        """Join filename with results directory."""
                
        if self.results_dir is None:
            raise ValueError("results_dir is not set.")
        return os.path.join(self.results_dir, filename)

    def _determine_num_projections(
        self, tomo_flag: bool
    ) -> int:
        """Determine number of tomographic projections."""
        if not tomo_flag:
            num_projections = 1
        elif len(self.shape) == 2:
            # Override nz for 1D case
            self.nz = self.nx
            self._log(f"Overriding nz to match nx(={self.nx}) for 1D case due to 90° tomographic projection.")
            num_projections = 2
        else:
            num_projections = 1
            self._log(
                "Warning: 90° tomographic projection requires cubic dimensions; "
                "proceeding with num_projections=1."
            )
        self.num_projections = num_projections
        self.total_scans = self.num_angles * self.scan_points * self.num_projections


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
    
    # ----------------------------------------------------------------------
    # Object contribution methods
    # ----------------------------------------------------------------------
    
    @abstractmethod
    def _get_reduced_sample(self, *args, **kwargs):
        """Retrieve the reduced object for propagation."""
        pass

    def create_object_contribution(self,n=None, grad=False, scan_index=0):
        """
        Create the field of object slices in free space.

        Works for 2D (x, z) and 3D (x, y, z) refractive index fields.

        Parameters
        ----------
        n : ndarray, optional
            Refractive index field. If None, uses self.refractive_index.
        grad : bool, optional
            If True, compute gradient coefficient (linear in n).
            If False, compute propagation coefficient (quadratic in n).
        scan_index : int, optional
            Index for sub-sampling info in scan_frame_info.

        Returns
        -------
        object_steps : ndarray
            Complex-valued slices between adjacent z-layers.
            Shape is same as n except the last axis (z) is reduced by 1.
        """

        # Use the true refractive index field if not provided
        if n is None:
            n = self.refractive_index_empty

        # Restrict to thin sample region if requested
        if self.solve_reduced_domain:
            n = self._get_reduced_sample(n, scan_index)

        # Compute coefficient
        if grad:
            coefficient = (self.k / 1j) * n
        else:
            coefficient = (self.k / 2j) * (n**2 - 1)

        # Compute all half time-step slices along the z-axis
        object_steps = (self.dz / 2) * (coefficient[..., :-1] + coefficient[..., 1:]) / 2

        return object_steps