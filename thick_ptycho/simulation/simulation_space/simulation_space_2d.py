

from typing import List

from thick_ptycho.thick_ptycho.utils.visualisations import Visualisation2D
from .base_simulation_space import BaseSimulationSpace, ScanFrame
from matplotlib import pyplot as plt
import numpy as np


class SimulationSpace2D(BaseSimulationSpace):
    """2D simulation domain setup."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xlims, self.ylims, self.zlims = self.continuous_dimensions
        self.nx, self.ny, self.nz = self.discrete_dimensions

        # Define grid
        self.x = np.linspace(self.xlims[0], self.xlims[1], self.nx)
        self.y = np.linspace(self.ylims[0], self.ylims[1], self.ny)
        self.z = np.linspace(self.zlims[0], self.zlims[1], self.nz)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

        # Image dimension
        self.dimension = 2

        # Initialize the refractive index field
        self.n_true = np.ones((self.nx, self.ny, self.nz), dtype=complex)*self.n_medium

        # Total number of probes (scan_points squared)
        self.num_probes = self.scan_points**2

        # Generate scan frame information
        self._detector_frame_info: List[ScanFrame] = self._generate_scan_frames()

        # Visualization utility
        self.viewer = Visualisation2D(self, results_dir=self.results_dir)

    def summarize(self):
        """ 
        Print a summary of the sample space and scan parameters.
        """
        # Print summary of the scan
        continuous_stepsize = (
            self.xlims[1] - self.xlims[0]) * (self.step_size / self.nx)
        overlap = self.probe_diameter * self.dx - continuous_stepsize
        self._log("Summary of the scan (continuous):")
        self._log(f"    Sample space x: {self.xlims[1] - self.xlims[0]} m")
        self._log(f"    Sample space y: {self.ylims[1] - self.ylims[0]} m")
        self._log(f"    Sample space z: {self.zlims[1] - self.zlims[0]} m")
        self._log(f"    Probe Diameter: {self.probe_diameter*self.dx:.2f} m")
        self._log(f"    Number of scan points: {self.num_probes}")
        if self.scan_points > 1:
            self._log(f"    Max Overlap: {overlap:.2f} m \n")

        # Plot the scan path with flipped axes
        centre_y = self._detector_frame_info[0].probe_centre_discrete[0]
        centre_x = self._detector_frame_info[0].probe_centre_discrete[1]
        plt.figure(figsize=(6, 6))
        plt.plot(centre_y, centre_x, marker='o', linestyle='-', markersize=2)
        plt.title("2D Discrete Scan Path")
        plt.xlabel("Ny")
        plt.ylabel("Nx")
        plt.xlim((0, self.ny))
        plt.ylim((0, self.nx))
        # Draw a box around the first probe area with faint fill and no outline
        y_min = int(centre_y - self.probe_dimensions[1] / 2)
        y_max = int(centre_y + self.probe_dimensions[1] / 2)
        x_min = int(centre_x - self.probe_dimensions[0] / 2)
        x_max = int(centre_x + self.probe_dimensions[0] / 2)
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
            centre_y = self._detector_frame_info[1].probe_centre_discrete[0]
            centre_x = self._detector_frame_info[1].probe_centre_discrete[1]
            # Draw a box around the second probe area with faint fill and no outline
            y_min = int(centre_y - self.probe_dimensions[1] / 2)
            y_max = int(centre_y + self.probe_dimensions[1] / 2)
            x_min = int(centre_x - self.probe_dimensions[0] / 2)
            x_max = int(centre_x + self.probe_dimensions[0] / 2)
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
    
    def _generate_scan_frames(self) -> List[Dict[str, Any]]:
        """
        Generate detector frames along a serpentine scan path.

        Returns:
            List[Dict[str, Any]]: List of dictionaries with detector frame data.
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
        assert edge_margin >= self.probe_dimensions[
            1] // 2, "Probe shape is too large for the dimensions of the sample space. Reduce probe shape or increase ny."

        # Generate x and y positions for the scan path
        x_positions = np.floor(np.linspace(
            edge_margin, self.nx - edge_margin, self.scan_points)).astype(int)
        y_positions = np.floor(np.linspace(
            edge_margin, self.ny - edge_margin, self.scan_points)).astype(int)

        centre_x = []
        centre_y = []
        x_positions = np.flip(x_positions)
        for x in x_positions:
            for y in y_positions:
                centre_x.append(x)
                centre_y.append(y)
            y_positions = np.flip(y_positions)

        # --- Construct detector frames ---
        frames = []

        for k in range(self.num_probes):
            # Centre of each probe
            cx = centre_x[k]
            cy = centre_y[k]
            if self.bc_type == "dirichlet" and self.num_probes == 1:
                cx += 1
                cy += 1
            probe_centre_discrete = (cx, cy)

            # Continuous Centre of each probe
            probe_centre_continuous = (self.x[cx], self.y[cy])

            # Boundaries of each probe
            x_min = int(cx - self.probe_dimensions[0] / 2)
            x_max = int(cx + self.probe_dimensions[0] / 2)
            y_min = int(cy - self.probe_dimensions[1] / 2)
            y_max = int(cy + self.probe_dimensions[1] / 2)

            # Continuous Boundaries of each probe
            if self.bc_type == "dirichlet":
                x_min -= 1
                x_max += 1
                y_min -= 1
                y_max += 1
            sub_x = self.x[x_min:x_max]
            sub_y = self.y[y_min:y_max]

            frames.append(ScanFrame(
                probe_centre_continuous= probe_centre_continuous,
                probe_centre_discrete= probe_centre_discrete,
                sub_dimensions= (sub_x, sub_y),
                sub_limits= (x_min, y_min)
            ))

        return frames