
from typing import List
from .base_simulation_space import BaseSimulationSpace, ScanFrame
import numpy as np


class SimulationSpace1D(BaseSimulationSpace):
    """1D simulation domain setup."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Continuous sample space limits (x, z)
        self.xlims, self.zlims = self.continuous_dimensions

        self.nx, self.nz = self.discrete_dimensions

        # Define grid
        nx_with_bc = self.nx + 2 if self.bc_type == "dirichlet" else self.nx
        self.x = np.linspace(self.xlims[0], self.xlims[1], nx_with_bc)
        self.z = np.linspace(self.zlims[0], self.zlims[1], self.nz)
        self.dx = self.x[1] - self.x[0]
        self.dz = self.z[1] - self.z[0]

        # Initialize the refractive index field
        self.n_true = np.ones((self.nx, self.nz), dtype=complex)*self.n_medium

        # Total number of probes
        self.num_probes = self.scan_points

        # Generate scan frame information
        self._scan_frame_info: List[ScanFrame] = self._generate_scan_frames()

    def summarize(self):
        """ 
        Print a summary of the sample space and scan parameters.
        """
        # Print summary of the scan
        continuous_stepsize = self.step_size * self.dx  

        # Overlap in meters
        overlap = self.probe_diameter * self.dx - continuous_stepsize

        self._log("=== Scan Summary (Continuous) ===")
        self._log(f"  Sample space (x-range): {self.xlims[1] - self.xlims[0]:.3e} m")
        self._log(f"  Sample space (z-range): {self.zlims[1] - self.zlims[0]:.3e} m")
        self._log(f"  Probe diameter:         {self.probe_diameter * self.dx:.3e} m")
        self._log(f"  Number of scan points:  {self.scan_points}")
        self._log(f"  Steps in z:             {self.nz}")
        self._log(f"  Detector Pixels:        {self.nx}")

        if self.scan_points > 1:
            self._log(f"  Max Overlap:            {overlap:.3e} m")
            self._log(f"  Percentage Overlap:     {overlap / (self.probe_diameter * self.dx) * 100:.2f}%\n")

    def _generate_scan_frames(self) -> List[ScanFrame]:
        """
        Generate detector frames along a serpentine scan path.

        Returns:
            List[ScanFrame]: List of ScanFrame objects with detector frame data.
            Each dictionary in the list corresponds to a scan.
            'probe_centre_continuous': probe_centre_continuous,
            'probe_centre_discrete': probe_centre_discrete,
            'sub_sample_slices': sample_slices,
            'sub_dimensions': (sub_x, sub_y)
        """
        # Total distance between probe centres
        total_step = (self.scan_points - 1) * self.step_size


        # Distance from the edge to the first probe centre
        edge_margin = (self.nx - total_step) // 2

        # Validate parameters
        assert self.nx > total_step, "nx must be greater than the total scan steps. Reduce scan points or steps size, or increase nx."
        assert edge_margin >= self.probe_dimensions[
            0] // 2, "Probe shape is too large for the dimensions of the sample space. Reduce probe shape or increase nx."

        # Generate x for the scan path
        centre_x = np.floor(np.linspace(
            edge_margin, self.nx - edge_margin, self.scan_points)).astype(int)

        # --- Construct detector frames ---
        frames = []

        for k in range(self.num_probes):
            # Centre of each probe
            cx = centre_x[k]

            if self.bc_type == "dirichlet" and self.num_probes == 1:
                cx += 1
            probe_centre_discrete = (cx)

            # Continuous Centre of each probe
            probe_centre_continuous = (self.x[cx])

            # Boundaries of each probe
            x_min = int(cx - self.probe_dimensions[0] / 2)
            x_max = int(cx + self.probe_dimensions[0] / 2)

            # Continuous Boundaries of each probe
            if self.bc_type == "dirichlet":
                x_min -= 1
                x_max += 1

            sub_x = self.x[x_min:x_max]

            frames.append(ScanFrame(
                probe_centre_continuous= probe_centre_continuous,
                probe_centre_discrete= probe_centre_discrete,
                sub_dimensions= (sub_x,),
                sub_limits= x_min
            ))
        return frames