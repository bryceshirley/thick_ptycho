from typing import List, Tuple, Union
from thick_ptycho.utils.utils import setup_log
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union

@dataclass
class ScanFrame:
    """Represents one scan frame in a ptychographic scan."""
    probe_centre_continuous: Union[float, Tuple[float, float]]
    probe_centre_discrete: Union[int, Tuple[int, int]]
    sub_dimensions: Union[Tuple[np.ndarray], Tuple[np.ndarray, np.ndarray]]
    sub_limits: Union[int, Tuple[int, int]]

class BaseSimulationSpace(ABC):
    """Abstract base defining shared simulation-space logic."""

    def __init__(
            self,
            continuous_dimensions, discrete_dimensions, probe_dimensions,
            scan_points, step_size, bc_type, probe_type, wave_number,
            probe_diameter=None, probe_focus=None, probe_angle_list=None,
            tomographic_projection_90_degree: bool = False,
            thin_sample: bool = False, n_medium=1.0, results_dir=None, 
            use_logging=True, verbose=False):
        """
        Initialize the 1D sample space.

        Parameters:
        continuous_dimensions (list): Sample space limits in continuous units (x, z) or (x,y,z).
        discrete_dimensions (list): Sample space dimensions in pixels (nx, nz) or (nx, ny, nz).
        detector_dimensions (list): Detector shape in pixels.
        scan_points (int): Number of scan points.
        bc_type (str): Boundary condition type (e.g., 'dirichlet', 'neumann', 'impedance').
        wave_number (float): Wavenumber in 1/nm.
        probe_diameter (float, optional): Probe diameter in continuous units. Default is 12.
        """
        # Dimensions Limits and sizes
        self.continuous_dimensions = continuous_dimensions
        self.discrete_dimensions = discrete_dimensions

        # Determine number of tomographic projections
        self.num_projections = 1

        if tomographic_projection_90_degree:
            # Check if all discrete dimensions are equal
            if len(set(discrete_dimensions)) == 1:
                self.num_projections = 2
            else:
                print(
                    "Warning: 90° tomographic projection requires all discrete dimensions "
                    "to be equal. Proceeding with num_projections=1."
                )


        # Thin sample mode
        self.thin_sample = thin_sample

        # Setup logging
        self._log = setup_log(results_dir, log_file_name="simulation_space_log.txt",
                               use_logging=use_logging, verbose=verbose)
        
        # Probe shape (pixels)
        self.probe_dimensions = probe_dimensions

        # Step size for scanning (pixels)
        self.step_size = step_size

        # Number of scan points along one axis
        self.scan_points = scan_points

        # Probe Type
        self.probe_type = probe_type
        self.probe_angles_list = probe_angle_list if probe_angle_list is not None else [0.0]
        self.probe_focus = probe_focus


        # Boundary condition type (lowercase)
        self.bc_type = bc_type.lower()

        # Probe diameter in continuous units and discrete pixels
        if probe_diameter is None:
            self.probe_diameter = probe_dimensions[0]
        else:
            self.probe_diameter = probe_diameter
        self.probe_diameter_continuous = self.probe_diameter * self.dx
        # Wavenumber
        self.k = wave_number
        self.wavelength = 2 * np.pi / self.k

        # Initialize the refractive index field
        self.n_medium = complex(n_medium)  # Ensure n_medium is complex

        # Generate scan frame information
        self._scan_frame_info: List[ScanFrame] = []

        # Use Sub-sampling for thin sample mode
        if self.thin_sample:
            self.slice_dimensions = probe_dimensions
        else:
            self.slice_dimensions = discrete_dimensions[:-1]

    @property
    def scan_frame_info(self) -> List[ScanFrame]:
        """
        List[ScanFrame]: List of ScanFrame objects with scan frame data.
        Each ScanFrame in the list corresponds to a scan.
        - 'probe_centre_continuous': probe_centre_continuous,
        - 'probe_centre_discrete': probe_centre_discrete,
        - 'sub_dimensions': (sub_x, sub_y) or (sub_x,) for 1D,
        """
        return self._scan_frame_info

    @abstractmethod
    def summarize(self):
        raise NotImplementedError
    
    @abstractmethod
    def _generate_scan_frames(self) -> List[ScanFrame]:
        raise NotImplementedError