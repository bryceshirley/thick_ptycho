
from typing import List

from thick_ptycho.utils.visualisations import Visualisation
from .base_simulation_space import BaseSimulationSpace, ScanFrame
import numpy as np
from thick_ptycho.simulation.scan_frame import ScanFrame, Point, Limits

class SimulationSpace1D(BaseSimulationSpace):
    """1D simulation domain setup."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # ------------------------------------------------------------------
        # 1. Geometry setup
        # ------------------------------------------------------------------
        self.shape = (self.nx, self.nz)
        self.dimension = 1

        # Effective width
        self.block_size = self.effective_dimensions
        self.effective_nx = self.effective_dimensions 

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

        self._log("=== Scan Summary (Continuous) ===")
        self._log(f"  Sample space (x-range): {self.spatial_limits.x[1] - self.spatial_limits.x[0]:.3e} m")
        self._log(f"  Sample space (z-range): {self.spatial_limits.z[1] - self.spatial_limits.z[0]:.3e} m")
        self._log(f"  Sample Pixels:          {self.nx}")
        self._log(f"  Number of scan points:  {self.scan_points}")
        self._log(f"  Steps in z:             {self.nz}")
        if self.solve_reduced_domain:
            self._log(f"  Solve reduced domain:   {self.effective_nx} px")
        if self.probe_diameter is not None:
            overlap = self.probe_diameter - continuous_stepsize
            self._log(f"  Probe diameter:         {self.probe_diameter:.3e} m")
            self._log(f"  Probe Pixels:          {int(self.probe_diameter_pixels)} px")
            if self.scan_points > 1:
                self._log(f"  Max Overlap:            {overlap:.3e} m")
                self._log(f"  Percentage Overlap:     {overlap / (self.probe_diameter) * 100:.2f}%\n")

    def _generate_scan_frames(self) -> List[ScanFrame]:
        """Generate detector frames along a symmetric scan grid."""

        # ---- Determine window width (Effective N) ----
        W = self.step_size + self.pad_discrete

        # ---- Single Scan ----
        if self.scan_points == 1:
            cx = self.nx // 2
            scan_frame = ScanFrame(
                        probe_centre_continuous=Point(x=self.x[cx]),
                        probe_centre_discrete=Point(x=cx))
            if self.solve_reduced_domain:
                scan_frame.set_reduced_limits_continuous(Limits(x=(self.x[0], self.x[self.nx - 1])))
                scan_frame.set_reduced_limits_discrete(Limits(x=(0, self.nx - 1)))

            return [scan_frame]

        # ---- Multi-scan Case ----
        half = W // 2
        mid = (self.nx - 1) / 2.0

        centres_x = [
            int(round(mid + (i - (self.scan_points - 1) / 2) * self.step_size))
            for i in range(self.scan_points)
        ]

        frames = []
        for cx in centres_x:
            xmin = cx - half
            xmax = xmin + W - 1

            scan_frame = ScanFrame(
                        probe_centre_continuous=Point(x=self.x[cx]),
                        probe_centre_discrete=Point(x=cx))
            if self.solve_reduced_domain:
                scan_frame.set_reduced_limits_continuous(Limits(x=(self.x[xmin], self.x[xmax])))
                scan_frame.set_reduced_limits_discrete(Limits(x=(xmin, xmax)))
    
            frames.append(scan_frame)

        return frames
