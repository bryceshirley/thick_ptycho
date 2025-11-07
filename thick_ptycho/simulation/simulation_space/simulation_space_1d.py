
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

    # def _generate_scan_frames(self) -> List[ScanFrame]:
    #     """Generate detector frames along a serpentine scan path."""

    #     # Compute scan centres along 1D axis
    #     centres_x = [
    #         self.effective_dimensions // 2 + i * self.step_size
    #         for i in range(self.scan_points)
    #     ]

    #     frames: List[ScanFrame] = []
    #     half = self.effective_dimensions // 2

    #     for cx in centres_x:
    #         cx = int(cx)
    #         x_min = max(cx - half, 0)
    #         x_max = min(cx + half, self.nx - 1)

            

    #         frames.append(
    #             ScanFrame(
    #                 probe_centre_continuous=self.x[cx],
    #                 probe_centre_discrete=cx,
    #                 reduced_limits_continuous=(self.x[x_min], self.x[x_max]),
    #                 reduced_limits_discrete=(x_min, x_max),
    #             )
    #         )

    #     return frames
    # def _generate_scan_frames(self) -> List[ScanFrame]:
    #     """
    #     Generate uniformly spaced, symmetric scan frames.

    #     reduced_limits_discrete is stored as (start, end_exclusive).
    #     When effective_nx == nx, the window covers the entire domain.
    #     """

    #     frames: List[ScanFrame] = []

    #     # Width of the effective window
    #     W = self.effective_nx
    #     half = W // 2

    #     # ---- Compute symmetric scan centres ----
    #     mid = (self.nx - 1) / 2
    #     centres_x = [
    #         int(round(mid + (i - (self.scan_points - 1) / 2) * self.step_size))
    #         for i in range(self.scan_points)
    #     ]

    #     # ---- Case A: No reduction -> full domain windows ----
    #     if W == self.nx:
    #         for cx in centres_x:
    #             frames.append(
    #                 ScanFrame(
    #                     probe_centre_continuous=self.x[cx],
    #                     probe_centre_discrete=cx,
    #                     reduced_limits_continuous=(self.x[0], self.x[self.nx - 1]),
    #                     reduced_limits_discrete=(0, self.nx),   # EXCLUSIVE end
    #                 )
    #             )
    #         return frames

    #     # ---- Case B: Reduced domain window sliding ----
    #     for cx in centres_x:
    #         cx = int(cx)

    #         # Compute inclusive bounds
    #         x_min = cx - half
    #         x_max = cx + half   # inclusive

    #         # Shift window if out of bounds
    #         if x_min < 0:
    #             shift = -x_min
    #             x_min += shift
    #             x_max += shift

    #         if x_max >= self.nx:
    #             shift = x_max - (self.nx - 1)
    #             x_min -= shift
    #             x_max -= shift

    #         # Convert to exclusive end index for discrete storage
    #         frames.append(
    #             ScanFrame(
    #                 probe_centre_continuous=self.x[cx],
    #                 probe_centre_discrete=cx,
    #                 reduced_limits_continuous=(self.x[x_min], self.x[x_max]),
    #                 reduced_limits_discrete=(x_min, x_max + 1),  # EXCLUSIVE
    #             )
    #         )

    #     return frames

    # def _generate_scan_frames(self) -> List[ScanFrame]:
    #     """Generate detector frames along a symmetric scan grid."""

    #     # Effective width (Ne) = number of samples in the window
    #     W = self.effective_dimensions
    #     half = W // 2

    #     mid = (self.nx - 1) / 2.0
    #     centres_x = [int(round(mid + (i - (self.scan_points - 1) / 2) * self.step_size))
    #                 for i in range(self.scan_points)]

    #     frames = []

    #     for cx in centres_x:
    #         # Exclusive index window [xmin, xmax)
    #         xmin = cx - half
    #         xmax = xmin + W

    #         # Shift to stay in bounds
    #         if xmin < 0:
    #             xmax -= xmin  # shift right
    #             xmin = 0

    #         if xmax > self.nx:
    #             shift = xmax - self.nx
    #             xmin -= shift
    #             xmax -= shift

    #         frames.append(
    #             ScanFrame(
    #                 probe_centre_continuous=self.x[cx],
    #                 probe_centre_discrete=cx,
    #                 reduced_limits_continuous=(self.x[xmin], self.x[xmax - 1]),
    #                 reduced_limits_discrete=(xmin, xmax),
    #             )
    #         )

    #     return frames
    # def _generate_scan_frames(self) -> List[ScanFrame]:
    #     """Generate detector frames along a symmetric scan grid."""

    #     if self.scan_points == 1:
    #         # Single scan point covers full effective domain
    #         return [
    #             ScanFrame(
    #                 probe_centre_continuous=self.x[self.nx // 2],
    #                 probe_centre_discrete=self.nx // 2,
    #                 reduced_limits_continuous=(self.x[0], self.x[self.nx - 1]),
    #                 reduced_limits_discrete=(0, self.nx-1),
    #             )
    #         ]
    #     W = self.effective_dimensions

    #     # Split half-width symmetrically (handles even & odd W properly)
    #     half_left = (W - 1) // 2
    #     half_right = W - 1 - half_left

    #     mid = (self.nx - 1) / 2.0
    #     centres_x = [int(round(mid + (i - (self.scan_points - 1) / 2) * self.step_size))
    #                 for i in range(self.scan_points)]

    #     frames = []

    #     for cx in centres_x:
    #         xmin = cx - half_left
    #         xmax = cx + half_right + 1  # exclusive end index

    #         # Shift if needed to stay in bounds
    #         if xmin < 0:
    #             xmax -= xmin
    #             xmin = 0

    #         if xmax > self.nx:
    #             shift = xmax - self.nx
    #             xmin -= shift
    #             xmax -= shift

    #         frames.append(
    #             ScanFrame(
    #                 probe_centre_continuous=self.x[cx],
    #                 probe_centre_discrete=cx,
    #                 reduced_limits_continuous=(self.x[xmin], self.x[xmax-1]),
    #                 reduced_limits_discrete=(xmin, xmax-1),
    #             )
    #         )

    #     return frames
    # def _generate_scan_frames(self) -> List[ScanFrame]:
    #     """Generate detector frames along a symmetric scan grid."""

    #     # --- Determine window width ---
    #     if self.scan_points == 1:
    #         # Single scan → full domain
    #         cx = self.nx // 2
    #         return [
    #             ScanFrame(
    #                 probe_centre_continuous=self.x[cx],
    #                 probe_centre_discrete=cx,
    #                 reduced_limits_continuous=(self.x[0], self.x[self.nx - 1]),
    #                 reduced_limits_discrete=(0, self.nx - 1),
    #             )
    #         ]

    #     # Multi-scan case:
    #     if self.solve_reduced_domain:
    #         W = self.step_size + self.pad_discrete  # Ne (expected in tests)
    #     else:
    #         W = self.nx  # full domain

    #     half = W // 2

    #     # Compute symmetric centre positions
    #     mid = (self.nx - 1) / 2.0
    #     centres_x = [
    #         int(round(mid + (i - (self.scan_points - 1) / 2) * self.step_size))
    #         for i in range(self.scan_points)
    #     ]

    #     frames = []
    #     for cx in centres_x:
    #         xmin = cx - half
    #         xmax = xmin + W  # exclusive

    #         # Clamp window to bounds while preserving width
    #         if xmin < 0:
    #             xmax -= xmin
    #             xmin = 0
    #         if xmax > self.nx:
    #             shift = xmax - self.nx
    #             xmin -= shift
    #             xmax -= shift

    #         # Store inclusive bounds
    #         xmax_inclusive = xmax - 1

    #         frames.append(
    #             ScanFrame(
    #                 probe_centre_continuous=self.x[cx],
    #                 probe_centre_discrete=cx,
    #                 reduced_limits_continuous=(self.x[xmin], self.x[xmax_inclusive]),
    #                 reduced_limits_discrete=(xmin, xmax_inclusive),
    #             )
    #         )

    #     return frames





    # def _generate_scan_frames(self) -> List[ScanFrame]:
    #     """Generate detector frames along a serpentine scan path."""

    #     # Compute scan centres (old method)
    #     centres_x = [
    #         self.effective_nx // 2 + i * (self.step_size)
    #         for i in range(self.scan_points)
    #     ]


    #     frames: List[ScanFrame] = []

    #     for cx in centres_x:
    #         # Old (incorrect-by-one) window calculation
    #         x_min = int(cx - self.effective_nx // 2)
    #         x_max = int(cx + self.effective_nx // 2)

    #         frames.append(
    #             ScanFrame(
    #                 probe_centre_continuous=self.x[int(cx)],
    #                 probe_centre_discrete=int(cx),
    #                 reduced_limits_continuous=(self.x[x_min], self.x[x_max]),
    #                 reduced_limits_discrete=(x_min, x_max),
    #             )
    #         )

    #     return frames


    def _generate_scan_frames(self) -> List[ScanFrame]:
        """Generate detector frames along a symmetric scan grid."""

        # ---- Determine window width: always Ne ----
        W = self.step_size + self.pad_discrete

        # ---- Special Case: Single Scan ----
        if self.scan_points == 1:
            cx = self.nx // 2
            return [
                ScanFrame(
                    probe_centre_continuous=self.x[cx],
                    probe_centre_discrete=cx,
                    reduced_limits_continuous=(self.x[0], self.x[self.nx - 1]),
                    reduced_limits_discrete=(0, self.nx - 1),
                )
            ]

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
            xmax = xmin + W  # exclusive

            # Clamp to bounds while preserving width
            if xmin < 0:
                xmax -= xmin
                xmin = 0
            if xmax > self.nx:
                shift = xmax - self.nx
                xmin -= shift
                xmax -= shift

            xmax_inc = xmax - 1  # inclusive right endpoint

            frames.append(
                ScanFrame(
                    probe_centre_continuous=self.x[cx],
                    probe_centre_discrete=cx,
                    reduced_limits_continuous=(self.x[xmin], self.x[xmax_inc]),
                    reduced_limits_discrete=(xmin, xmax_inc),
                )
            )

        return frames
