

from typing import List

from thick_ptycho.utils.visualisations import Visualisation2D
from .base_simulation_space import BaseSimulationSpace, ScanFrame
from matplotlib import pyplot as plt
import numpy as np
from thick_ptycho.simulation.scan_frame import ScanFrame, Point, Limits

class SimulationSpace2D(BaseSimulationSpace):
    """2D simulation domain setup."""

    def __init__(self,scan_path: str = "serpentine",
                 **kwargs):
        super().__init__(**kwargs)
        # ------------------------------------------------------------------
        # 1. Geometry setup
        # ------------------------------------------------------------------
        self.ny = self.nx  # Assume square grid for 2D
        self.dimension = 2

        # Effective resolution
        self.effective_nx = self.effective_dimensions
        self.effective_ny = self.effective_dimensions
        self.block_size = self.effective_nx * self.effective_ny

        # ------------------------------------------------------------------
        # 2. Spatial grid setup
        # ------------------------------------------------------------------
        self.x = np.linspace(*self.spatial_limits.x, self.nx)
        self.y = np.linspace(*self.spatial_limits.y, self.ny)
        self.z = np.linspace(*self.spatial_limits.z, self.nz)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

        # ------------------------------------------------------------------
        # 4. Scan setup
        # ------------------------------------------------------------------
        self.num_probes = self.scan_points**2
        self.scan_path = scan_path
        self._scan_frame_info = self._generate_scan_frames()

        # ------------------------------------------------------------------
        # 5. Visualization utility
        # ------------------------------------------------------------------
        self.viewer = Visualisation2D(self, results_dir=self.results_dir)

    def summarize(self):
        """ 
        Print a summary of the sample space and scan parameters.
        """
        # Print summary of the scan
        continuous_stepsize = self.step_size * self.dx  

        self._log("Summary of the scan (continuous):")
        self._log(f"    Sample space x: {self.spatial_limits.x[1] - self.self.spatial_limits.x[0]} m")
        self._log(f"    Sample space y: {self.spatial_limits.y[1] - self.spatial_limits.y[0]} m")
        self._log(f"    Sample space z: {self.spatial_limits.z[1] - self.spatial_limits.z[0]} m")
        self._log(f"    Probe diameter: {self.probe_diameter:.3e} m")
        self._log(f"    Probe Pixels: {self.probe_diameter_pixels:.3e} px")
        self._log(f"    Number of scan points: {self.num_probes}")
        if self.probe_diameter is not None:
            overlap = self.probe_diameter - continuous_stepsize
            self._log(f"  Probe diameter:         {self.probe_diameter:.3e} m")
            self._log(f"  Probe Pixels:          {int(self.probe_diameter_pixels)} px")
            if self.scan_points > 1:
                self._log(f"  Max Overlap:            {overlap:.3e} m")
                self._log(f"  Percentage Overlap:     {overlap / (self.probe_diameter) * 100:.2f}%\n")

        self.plot_scan_path()

    def plot_scan_path(self):
        """
        Plot the 2D scan path with probe areas.
        """
        # Plot the scan path with flipped axes
        centre_x, centre_y = self._scan_frame_info[0].probe_centre_discrete.to_tuple()

        plt.figure(figsize=(6, 6))
        plt.plot(centre_y, centre_x, marker='o', linestyle='-', markersize=2)
        plt.title("2D Discrete Scan Path")
        plt.xlabel("Ny")
        plt.ylabel("Nx")
        plt.xlim((0, self.ny))
        plt.ylim((0, self.nx))
        # Draw a box around the first probe area with faint fill and no outline
        y_min = int(centre_y - self.edge_margin)
        y_max = int(centre_y + self.edge_margin)
        x_min = int(centre_x - self.edge_margin)
        x_max = int(centre_x + self.edge_margin)
        rect1 = plt.Rectangle(
            (y_min, x_min), y_max - y_min, x_max - x_min,
            linewidth=0, edgecolor='none', facecolor='red', alpha=0.2, label='First Probe Area'
        )
        plt.gca().add_patch(rect1)
        if self.probe_type == "disk":
            circ1 = plt.Circle(
                (centre_y, centre_x),
                radius=self.probe_diameter / 2,
                color='red', fill=False, alpha=0.2, label='First Probe'
            )
            plt.gca().add_patch(circ1)

        if self.scan_points > 1:
            centre_x, centre_y = self._scan_frame_info[1].probe_centre_discrete.to_tuple()
            # Draw a box around the second probe area with faint fill and no outline
            y_min = int(centre_y - self.edge_margin)
            y_max = int(centre_y + self.edge_margin)
            x_min = int(centre_x - self.edge_margin)
            x_max = int(centre_x + self.edge_margin)
            rect2 = plt.Rectangle(
                (y_min, x_min), y_max - y_min, x_max - x_min,
                linewidth=0, edgecolor='none', facecolor='green', alpha=0.2, label='Second Probe Area'
            )
            plt.gca().add_patch(rect2)
            if self.probe_type == "disk":
                circ2 = plt.Circle(
                    (centre_y, centre_x),
                    radius=self.probe_diameter / 2,
                    color='green', fill=False, alpha=0.2, label='Second Probe')
                plt.gca().add_patch(circ2)
        plt.legend()
        plt.grid()
        plt.show()

    def _generate_scan_frames(self) -> List[ScanFrame]:
        """
        Generate detector frames along a serpentine scan path.

        Returns:
            List[ScanFrame]: List of ScanFrame objects with detector frame data.
        """
        # ---- Determine window width (Effective N) ----
        W = self.step_size + self.pad_discrete

        if self.scan_points == 1:
            cx = self.nx // 2
            cy = self.ny // 2
            scan_frame = ScanFrame(
                        probe_centre_continuous=Point(x=self.x[cx], y=self.y[cy]),
                        probe_centre_discrete=Point(x=cx, y=cy))
            if self.solve_reduced_domain:
                scan_frame.set_reduced_limits_continuous(Limits(
                    x=(self.x[0], self.x[-1]),
                    y=(self.y[0], self.y[-1])
                ))
                scan_frame.set_reduced_limits_discrete(Limits(
                    x=(0, self.nx - 1),
                    y=(0, self.ny - 1)
                ))

            return [scan_frame]
        
        # ---- Multi-scan Case ----
        half = W // 2
        mid_x = (self.nx - 1) / 2.0
        mid_y = (self.ny - 1) / 2.0

        # Central coordinates for scan points
        x_coords = [
            int(round(mid_x + (i - (self.scan_points - 1) / 2) * self.step_size))
            for i in range(self.scan_points)
        ]
        y_coords = [
            int(round(mid_y + (i - (self.scan_points - 1) / 2) * self.step_size))
            for i in range(self.scan_points)
        ]

        # Serpentine scan path
        if self.scan_path == "serpentine":
            centre_x, centre_y = self.serpentine_order_scan(x_coords, y_coords)
        elif self.scan_path == "spiral":
            centre_x, centre_y = self.spiral_order_scan(x_coords, y_coords)
        else:
            raise ValueError(f"Unknown scan path: {self.scan_path}")

        # --- Construct detector frames ---
        frames: List[ScanFrame] = []
        for cx, cy in zip(centre_x, centre_y):
            xmin = cx - half
            xmax = xmin + W  - 1
            ymin = cy - half
            ymax = ymin + W  - 1

            scan_frame = ScanFrame(
                        probe_centre_continuous=Point(x=self.x[cx], y=self.y[cy]),
                        probe_centre_discrete=Point(x=cx, y=cy))
            if self.solve_reduced_domain:
                scan_frame.set_reduced_limits_continuous(Limits(
                    x=(self.x[xmin], self.x[xmax]),
                    y=(self.y[ymin], self.y[ymax])
                ))
                scan_frame.set_reduced_limits_discrete(Limits(
                    x=(xmin, xmax),
                    y=(ymin, ymax)
                ))
    
            frames.append(scan_frame)

    # ------ Define different scan path patterns -----
    def serpentine_order_scan(self, x_coords: List[int], y_coords: List[int]) -> List[ScanFrame]:
        """Reorder frames in a serpentine pattern."""
        centre_x = []
        centre_y = []
        x_coords = np.flip(x_coords)
        for x in x_coords:
            for y in y_coords:
                centre_x.append(x)
                centre_y.append(y)
            y_coords = np.flip(y_coords)
        return centre_x, centre_y

    def spiral_order_scan(self, x_coords: List[int], y_coords: List[int]) -> List[ScanFrame]:
        """Reorder frames in a spiral pattern."""
        centre_x = []
        centre_y = []
        left, right = 0, len(x_coords) - 1
        top, bottom = 0, len(y_coords) - 1
        while left <= right and top <= bottom:
            for i in range(left, right + 1):
                centre_x.append(x_coords[i])
                centre_y.append(y_coords[top])
            top += 1
            for i in range(top, bottom + 1):
                centre_x.append(x_coords[right])
                centre_y.append(y_coords[i])
            right -= 1
            if top <= bottom:
                for i in range(right, left - 1, -1):
                    centre_x.append(x_coords[i])
                    centre_y.append(y_coords[bottom])
                bottom -= 1
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    centre_x.append(x_coords[left])
                    centre_y.append(y_coords[i])
                left += 1
        return centre_x, centre_y
        # # Generate x and y positions for the scan path
        # x_positions = np.floor(np.linspace(
        #     self.edge_margin, self.nx - self.edge_margin, self.scan_points)).astype(int)
        # y_positions = np.floor(np.linspace(
        #     self.edge_margin, self.ny - self.edge_margin, self.scan_points)).astype(int)

        # centres_x = []
        # centres_y = []
        # x_positions = np.flip(x_positions)
        # for x in x_positions:
        #     for y in y_positions:
        #         centres_x.append(x)
        #         centres_y.append(y)
        #     y_positions = np.flip(y_positions)

        # # --- Construct detector frames ---
        # frames: List[ScanFrame] = []
        # for cx, cy in zip(centres_x, centres_y):
        #     # Boundaries of each probe
        #     x_min, x_max = int(cx - self.edge_margin), int(cx + self.edge_margin)
        #     y_min, y_max = int(cy - self.edge_margin), int(cy + self.edge_margin)

        #     frames.append(ScanFrame(
        #         probe_centre_continuous= (self.x[cx], self.y[cy]),
        #         probe_centre_discrete= (cx, cy),
        #         effective_limits_continuous= (self.x[x_min], self.x[x_max], self.y[y_min], self.y[y_max]),
        #         effective_limits_discrete= (x_min, x_max, y_min, y_max)
        #     ))

        # return frames