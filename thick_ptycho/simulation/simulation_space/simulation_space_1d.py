
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
        self.nx, self.nz = self.discrete_dimensions
        self.block_size = self.nx
        self.dimension = 1

        # self.nz = self.discrete_dimensions[1]
        # pad_factor = 1.1

        # probe_width = self.probe_dimensions[0]

        # self.min_nx = (self.scan_points - 1) * self.step_size + probe_width
        # self.nx = int(self.min_nx * pad_factor)

        # self.discrete_dimensions = (self.nx, self.nz)

        # Effective width
        self.effective_nx = (
            self.probe_dimensions[0] if self.thin_sample else self.nx
        )

        # ------------------------------------------------------------------
        # 2. Spatial grid setup
        # ------------------------------------------------------------------
        nx_with_bc = self.nx + 2 if self.bc_type == "dirichlet" else self.nx
        self.x = np.linspace(self.xlims[0], self.xlims[1], nx_with_bc)
        self.dx = self.x[1] - self.x[0]

        # ------------------------------------------------------------------
        # 3. Scan setup
        # ------------------------------------------------------------------
        self.num_probes = self.scan_points
        self._scan_frame_info = self._generate_scan_frames()

        # ------------------------------------------------------------------
        # 4. Visualization utility
        # ------------------------------------------------------------------
        self.viewer = Visualisation(self, results_dir=self.results_dir)

    # def validate_parameters(self) -> int:
    #     """
    #     Validate simulation space parameters and compute edge margin for scanning.

    #     Returns:
    #         int: Edge margin in discrete units.
    #     """
    #     # ------------------------------------------------------------------
    #     # 9. Validation
    #     # -------------------------------------------------------------------
    #     nx = self.discrete_dimensions[0]
    #     total_distance = (self.scan_points-1) * self.step_size + self.probe_dimensions[0]
    #     edge_margin = (nx - total_distance) // 2

    #     # if nx <= total_distance:
    #     #     raise ValueError(
    #     #         f"nx must be greater than the total scan steps ({total_distance}). "
    #     #         "Reduce scan_points or step_size, or increase nx."
    #     #     )
    #     return edge_margin

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
        self._log(f"  Probe diameter:         {self.probe_diameter_continuous:.3e} m")
        self._log(f"  Number of scan points:  {self.scan_points}")
        self._log(f"  Steps in z:             {self.nz}")
        self._log(f"  Detector Pixels:        {self.nx}")

        if self.scan_points > 1:
            self._log(f"  Max Overlap:            {overlap:.3e} m")
            self._log(f"  Percentage Overlap:     {overlap / (self.probe_diameter_continuous * self.dx) * 100:.2f}%\n")

    # def _generate_scan_frames(self) -> List[ScanFrame]:
    #     """
    #     Generate detector frames along a serpentine scan path.

    #     Returns:
    #         List[ScanFrame]: List of ScanFrame objects with detector frame data.
    #         Each dictionary in the list corresponds to a scan.
    #     """

    #     # Generate x for the scan path
    #     print("Theoretical scan dimensions:")
    #     nx = self.discrete_dimensions[0]
    #     total_distance = (self.scan_points-1) * self.step_size + self.probe_dimensions[0]
    #     edge_margin = (nx - total_distance) // 2
    #     print(f"Total scan distance (in discrete units): {total_distance}")
    #     print(f"Edge margin from final probe centre to boundary (in discrete units): {edge_margin}\n")


    #     print("Actual scan dimensions:")
    #     centres_x = np.floor(np.linspace(
    #         self.edge_margin, self.nx - self.edge_margin, self.scan_points)).astype(int)
    #     print(f"total x points in simulation space: {len(self.x)}")
    #     print(f"final scan centres (discrete units): {centres_x[-1]}")
    #     print(f"probe half-width (discrete units): {self.probe_dimensions[0] // 2}")
    #     print(f"max x index + half probe width: {centres_x[-1] + self.probe_dimensions[0] // 2}")

    #     assert centres_x[-1] + self.probe_dimensions[0] // 2 >= len(self.x), "First/last probe goes out of bounds."
    #     # --- Construct detector frames ---
    #     frames: List[ScanFrame] = []

    #     for cx in centres_x:
    #         if self.bc_type == "dirichlet" and self.num_probes == 1:
    #             cx += 1

    #         # Boundaries of each probe
    #         x_min = int(cx - self.probe_dimensions[0] / 2)
    #         x_max = int(cx + self.probe_dimensions[0] / 2)

    #         if self.bc_type == "dirichlet":
    #             x_min = max(x_min - 1, 0)
    #             x_max = min(x_max + 1, self.nx - 1)

    #         frames.append(ScanFrame(
    #             probe_centre_continuous= self.x[cx],
    #             probe_centre_discrete= cx,
    #             sub_limits_continuous= (self.x[x_min], self.x[x_max]),
    #             sub_limits_discrete= (x_min, x_max)
    #         ))
    #     return frames
    def _generate_scan_frames(self) -> List[ScanFrame]:
        """Generate detector frames along a serpentine scan path."""
        nx = self.discrete_dimensions[0]
        probe_width = self.probe_dimensions[0]
        half_probe = probe_width // 2

        total_distance = (self.scan_points - 1) * self.step_size + probe_width
        edge_margin = int((nx - total_distance) // 2)

        print("Theoretical scan dimensions:")
        print(f"  Total scan distance: {total_distance}")
        print(f"  Edge margin: {edge_margin}")

        # --- Corrected start and stop for linspace ---
        start = edge_margin + half_probe
        stop = nx - edge_margin - half_probe - 1

        centres_x = np.floor(np.linspace(start, stop, self.scan_points)).astype(int)

        print(f"  Start index: {start}")
        print(f"  Stop index: {stop}")
        print(f"  Final centre: {centres_x[-1]}")
        print(f"  Probe half-width: {half_probe}")
        print(f"  Last + half-width: {centres_x[-1] + half_probe}")
        print(f"  Grid length: {len(self.x)}")

        # --- Assert the probes fit inside the grid ---
        assert centres_x[0] - half_probe >= 0, "First probe out of bounds."
        assert centres_x[-1] + half_probe <= nx, "Last probe out of bounds."

        # --- Construct frames ---
        frames: List[ScanFrame] = []
        for cx in centres_x:
            x_min = int(cx - half_probe)
            x_max = int(cx + half_probe)

            frames.append(
                ScanFrame(
                    probe_centre_continuous=self.x[cx],
                    probe_centre_discrete=cx,
                    sub_limits_continuous=(self.x[x_min], self.x[x_max]),
                    sub_limits_discrete=(x_min, x_max),
                )
            )
        return frames
