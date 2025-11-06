
from typing import List

from thick_ptycho.utils.visualisations import Visualisation
from .base_simulation_space import BaseSimulationSpace, ScanFrame
import numpy as np


class SimulationSpace1D(BaseSimulationSpace):
    """1D simulation domain setup."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # ------------------------------------------------------------------
        # 1. Geometry setup
        # ------------------------------------------------------------------
        self.xlims, self.zlims = self.continuous_dimensions
        self.discrete_dimensions = (self.nx, self.nz)
        self.dimension = 1

        # Effective width
        self.block_size = self.effective_dimensions
        self.effective_nx = self.effective_dimensions 

        # ------------------------------------------------------------------
        # 2. Spatial grid setup
        # ------------------------------------------------------------------
        self.x = np.linspace(self.xlims[0], self.xlims[1], self.nx)

        # ------------------------------------------------------------------
        # 3. Scan setup
        # ------------------------------------------------------------------
        self.num_probes = self.scan_points
        self._scan_frame_info = self._generate_scan_frames()

        # ------------------------------------------------------------------
        # 4. Visualization utility
        # ------------------------------------------------------------------
        self.viewer = Visualisation(self, results_dir=self.results_dir)

    def summarize(self):
        """ 
        Print a summary of the sample space and scan parameters.
        """
        # Print summary of the scan
        continuous_stepsize = self.step_size * self.dx  

        # Overlap in meters
        overlap = self.probe_diameter_continuous - continuous_stepsize

        self._log("=== Scan Summary (Continuous) ===")
        self._log(f"  Sample space (x-range): {self.xlims[1] - self.xlims[0]:.3e} m")
        self._log(f"  Sample space (z-range): {self.zlims[1] - self.zlims[0]:.3e} m")
        self._log(f"  Sample Pixels:          {self.nx}")
        self._log(f"  Number of scan points:  {self.scan_points}")
        self._log(f"  Steps in z:             {self.nz}")
        if self.solve_reduced_domain:
            self._log(f"  Solve reduced domain:   {self.effective_nx} px")
        self._log(f"  Probe diameter:         {self.probe_diameter_continuous:.3e} m")
        self._log(f"  Probe Pixels:          {int(self.probe_diameter_pixels)} px")

        if self.scan_points > 1:
            self._log(f"  Max Overlap:            {overlap:.3e} m")
            self._log(f"  Percentage Overlap:     {overlap / (self.probe_diameter_continuous) * 100:.2f}%\n")

    def _generate_scan_frames(self) -> List[ScanFrame]:
        """Generate detector frames along a serpentine scan path."""
        # --- Corrected start and stop for linspace ---
        start = self.edge_margin
        stop = self.nx - self.edge_margin - 1
        centres_x = np.floor(np.linspace(start, stop, self.scan_points)).astype(int)

        # --- Construct frames ---
        frames: List[ScanFrame] = []
        for cx in centres_x:
            x_min = int(cx - self.edge_margin)
            x_max = int(cx + self.edge_margin)

            frames.append(
                ScanFrame(
                    probe_centre_continuous=self.x[cx],
                    probe_centre_discrete=cx,
                    sub_limits_continuous=(self.x[x_min], self.x[x_max]),
                    sub_limits_discrete=(x_min, x_max),
                )
            )
        return frames
