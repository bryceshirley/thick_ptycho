

from typing import List

from thick_ptycho.utils.visualisations import Visualisation2D
from .base_simulation_space import BaseSimulationSpace, ScanFrame
from matplotlib import pyplot as plt
import numpy as np


class SimulationSpace2D(BaseSimulationSpace):
    """2D simulation domain setup."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ------------------------------------------------------------------
        # 1. Geometry setup
        # ------------------------------------------------------------------
        self.xlims, self.ylims, self.zlims = self.continuous_dimensions
        self.ny = self.nx  # Assume square grid for 2D
        self.dimension = 2

        # Effective resolution
        self.effective_nx = self.effective_dimensions
        self.effective_ny = self.effective_dimensions
        self.block_size = self.effective_nx * self.effective_ny

        # ------------------------------------------------------------------
        # 2. Spatial grid setup
        # ------------------------------------------------------------------
        self.x = np.linspace(*self.xlims, self.nx)
        self.y = np.linspace(*self.ylims, self.ny)
        self.z = np.linspace(*self.zlims, self.nz)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

        # ------------------------------------------------------------------
        # 4. Scan setup
        # ------------------------------------------------------------------
        self.num_probes = self.scan_points**2
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

        # Overlap in meters
        overlap = self.probe_diameter_continuous - continuous_stepsize

        self._log("Summary of the scan (continuous):")
        self._log(f"    Sample space x: {self.xlims[1] - self.xlims[0]} m")
        self._log(f"    Sample space y: {self.ylims[1] - self.ylims[0]} m")
        self._log(f"    Sample space z: {self.zlims[1] - self.zlims[0]} m")
        self._log(f"    Probe diameter: {self.probe_diameter_continuous:.3e} m")
        self._log(f"    Probe Pixels: {self.probe_diameter_pixels:.3e} px")
        self._log(f"    Number of scan points: {self.num_probes}")
        if self.scan_points > 1:
            self._log(f"    Max Overlap: {overlap:.2f} m \n")

        self.plot_scan_path()

    def plot_scan_path(self):
        """
        Plot the 2D scan path with probe areas.
        """
        # Plot the scan path with flipped axes
        centre_y = self._scan_frame_info[0].probe_centre_discrete[0]
        centre_x = self._scan_frame_info[0].probe_centre_discrete[1]

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
            centre_y = self._scan_frame_info[1].probe_centre_discrete[0]
            centre_x = self._scan_frame_info[1].probe_centre_discrete[1]
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
        # Generate x and y positions for the scan path
        x_positions = np.floor(np.linspace(
            self.edge_margin, self.nx - self.edge_margin, self.scan_points)).astype(int)
        y_positions = np.floor(np.linspace(
            self.edge_margin, self.ny - self.edge_margin, self.scan_points)).astype(int)

        centres_x = []
        centres_y = []
        x_positions = np.flip(x_positions)
        for x in x_positions:
            for y in y_positions:
                centres_x.append(x)
                centres_y.append(y)
            y_positions = np.flip(y_positions)

        # --- Construct detector frames ---
        frames: List[ScanFrame] = []
        for cx, cy in zip(centres_x, centres_y):
            # Boundaries of each probe
            x_min, x_max = int(cx - self.edge_margin), int(cx + self.edge_margin)
            y_min, y_max = int(cy - self.edge_margin), int(cy + self.edge_margin)

            frames.append(ScanFrame(
                probe_centre_continuous= (self.x[cx], self.y[cy]),
                probe_centre_discrete= (cx, cy),
                effective_limits_continuous= (self.x[x_min], self.x[x_max], self.y[y_min], self.y[y_max]),
                effective_limits_discrete= (x_min, x_max, y_min, y_max)
            ))

        return frames