

from typing import List

from thick_ptycho.utils.visualisations import Visualisation2D
from .base_simulation_space import BaseSimulationSpace, ScanFrame
from matplotlib import pyplot as plt
import numpy as np
from thick_ptycho.simulation.scan_frame import ScanFrame, Point, Limits, ScanPath


class SimulationSpace2D(BaseSimulationSpace):
    """2D simulation domain setup."""

    def __init__(self,scan_path: ScanPath = ScanPath.SERPENTINE, **kwargs):
        super().__init__(**kwargs)
        # ------------------------------------------------------------------
        # 1. Geometry setup
        # ------------------------------------------------------------------
        self.ny = self.nx  # Assume square grid for 2D
        self.dimension = 2
        self.shape = (self.nx,
                      self.ny,
                      self.nz)
        # Empty refractive index field representing the background medium
        self.refractive_index_empty = np.ones(self.shape, dtype=complex) * self.n_medium


        # Effective resolution
        self.effective_nx = self.effective_nx
        self.effective_ny = self.effective_nx
        self.effective_shape = (self.effective_nx, self.effective_ny, self.nz)
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
        self.scan_path = scan_path.value if scan_path is not None else ScanPath.SERPENTINE.value
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
        self._log(f"    Sample space x: {self.spatial_limits.x[1] - self.spatial_limits.x[0]} m")
        self._log(f"    Sample space y: {self.spatial_limits.y[1] - self.spatial_limits.y[0]} m")
        self._log(f"    Sample space z: {self.spatial_limits.z[1] - self.spatial_limits.z[0]} m")
        self._log(f"    Sample Pixels: nx={self.nx},ny={self.ny},nz={self.nz}")
        self._log(f"    Number of scan points: {self.num_probes}")
        if self.probe_diameter is not None:
            overlap = self.probe_diameter - continuous_stepsize
            self._log(f"    Probe diameter: {self.probe_diameter:.3e} m")
            self._log(f"    Probe Pixels: {int(self.probe_diameter_pixels)} px")
            if self.scan_points > 1:
                self._log(f"    Max Overlap: {overlap:.3e} m")
                self._log(f"    Percentage Overlap: {overlap / (self.probe_diameter) * 100:.2f}%\n")

        self.plot_scan_path()

    def plot_scan_path(self):
        """
        Plot the 2D scan path showing all probe centres,
        and highlight the first two probe areas (rectangles + circles).
        """
        plt.figure(figsize=(6, 6))
        ax = plt.gca()

        # --- Plot all probe centres ---
        all_centres = np.array([
            f.probe_centre_discrete.as_tuple() for f in self._scan_frame_info
        ])
        xs, ys = all_centres[:, 0], all_centres[:, 1]
        ax.plot(ys, xs, 'o-', color='blue', markersize=3, linewidth=0.5, label='Probe Centres')

        plt.title("2D Discrete Scan Path")
        plt.xlabel("Ny")
        plt.ylabel("Nx")
        plt.xlim((0, self.ny))
        plt.ylim((0, self.nx))

        # --- Highlight first two probe regions ---
        colors = ['red', 'green']
        labels = ['First', 'Second']

        for scan in range(min(2, len(self._scan_frame_info))):
            cx, cy = self._scan_frame_info[scan].probe_centre_discrete.as_tuple()

            # Rectangle around probe area
            W = self.step_size + self.pad_discrete
            half = W // 2
            x_min = int(cx - half)
            y_min = int(cy - half)
            rect = plt.Rectangle(
                (y_min, x_min), W-1, W-1,
                linewidth=0, edgecolor='none',
                facecolor=colors[scan], alpha=0.2,
                label=f'{labels[scan]} Probe Area'
            )
            ax.add_patch(rect)

            # Circle for probe footprint
            if self.probe_diameter is not None:
                circ = plt.Circle(
                    (cy, cx),
                    radius=self.probe_diameter_pixels // 2,
                    color=colors[scan], fill=False, alpha=0.4,
                    label=f'{labels[scan]} Probe'
                )
                ax.add_patch(circ)

        plt.legend()
        plt.grid(True)
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
                    y=(self.y[0], self.y[-1]),
                    units="meters"
                ))
                scan_frame.set_reduced_limits_discrete(Limits(
                    x=(0, self.nx - 1),
                    y=(0, self.ny - 1),
                    units="pixels"
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
        elif self.scan_path == "raster":
            centre_x, centre_y = self.raster_order_scan(x_coords, y_coords)
        else:
            raise ValueError(f"Unknown scan path: {self.scan_path}")

        # --- Construct detector frames ---
        frames: List[ScanFrame] = []
        for cx, cy in zip(centre_x, centre_y):
            xmin = max(cx - half, 0)
            xmax = min(xmin + W - 1,(self.nx - 1))
            ymin = max(cy - half, 0)
            ymax = min(ymin + W  - 1,(self.ny - 1))


            scan_frame = ScanFrame(
                        probe_centre_continuous=Point(x=self.x[cx], y=self.y[cy]),
                        probe_centre_discrete=Point(x=cx, y=cy))
            if self.solve_reduced_domain:
                scan_frame.set_reduced_limits_continuous(Limits(
                    x=(self.x[xmin], self.x[xmax]),
                    y=(self.y[ymin], self.y[ymax]),
                    units="meters"
                ))
                scan_frame.set_reduced_limits_discrete(Limits(
                    x=(xmin, xmax),
                    y=(ymin, ymax),
                    units="pixels"
                ))
    
            frames.append(scan_frame)
        return frames

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
    
    def raster_order_scan(self, x_coords: List[int], y_coords: List[int]) -> List[ScanFrame]:
        """
        Standard raster scan (row-major):
        y varies outer (rows), x varies inner (columns), always left → right.
        """
        centre_x = []
        centre_y = []
        for y in y_coords:        # row-by-row
            for x in x_coords:    # always left → right
                centre_x.append(x)
                centre_y.append(y)
        return centre_x, centre_y
    
    def _get_reduced_sample(self,n,scan_index):
        """Retrieve the object slices for propagation."""
        x_min,x_max = self._scan_frame_info[scan_index].reduced_limits_discrete.x
        y_min, y_max = self._scan_frame_info[scan_index].reduced_limits_discrete.y    
        return n[
                    x_min:x_max+1,
                    y_min:y_max+1,
                    :
                ]
