import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Sequence, Tuple, Union

try:
    from ipywidgets import interact, IntSlider
    import ipywidgets as widgets
    _HAS_WIDGETS = True
except Exception:
    _HAS_WIDGETS = False

import os
import numpy as np
import matplotlib.pyplot as plt

class Visualisation:
    """Common utilities for visualisation classes."""
    def __init__(self, simulation_space, results_dir=None):
        self.simulation_space = simulation_space
        self.results_dir = os.fspath(results_dir) if results_dir else None
        if self.results_dir:
            os.makedirs(self.results_dir, exist_ok=True)

    @staticmethod
    def phase(arr, tol=5e-12):
        ph = np.angle(arr)
        mag = np.abs(arr)
        return np.where(mag < tol, 0.0, ph)

    def _save_if_needed(self, fig, filename):
        if self.results_dir:
            path = os.path.join(self.results_dir, filename)
            fig.savefig(path, bbox_inches="tight")

    def _view_and_title(self, solution, view: str, title: str) -> Tuple[str, str]:
        """Return modified title and solution based on view type."""
        assert view in ("phase_amp", "real_imag"), "view must be 'phase_amp' or 'real_imag'"
        if view == "phase_amp":
            data1 = self.phase(solution)
            data2 = np.abs(solution)
            title1 = title + " Phase"
            title2 = title + " Amplitude"
        else:
            data1 = solution.real
            data2 = solution.imag
            title1 = title + " Real"
            title2 = title + " Imaginary"
        return data1, data2, title1, title2

    def plot_two_panels(self, data, view="phase_amp",title="",
                        xlabel=None, ylabel=None,
                        filename=None, extent=None):
        """Return figure and axes for phase/amp or real/imag view.
        Parameters
        ----------
        data : np.ndarray
            2D array to plot.
        view : {'phase_amp', 'real_imag'}
            Determines whether to plot (phase, amplitude) or (real, imaginary).
        filename : str, optional
            If provided, saves the figure to this file. 
        extent : list, optional for continuous axes
            Extent for imshow [xmin, xmax, ymin, ymax].
        """
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        data1, data2, title1, title2 = self._view_and_title(data, view, title)

        im0 = axs[0].imshow(data1, cmap="viridis", origin="lower")
        im1 = axs[1].imshow(data2, cmap="viridis", origin="lower")
        fig.colorbar(im0, ax=axs[0], extent=extent)
        fig.colorbar(im1, ax=axs[1], extent=extent)

        axs[0].set_title(title1)
        axs[1].set_title(title2)
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel)


        fig.tight_layout()
        if filename:
            self._save_if_needed(fig, filename)
        return fig, axs

    def plot_residual(self, residuals, title="Residual History", loglog=True, filename=None):
        """Plot residual history over iterations.
        Paramaters
        ----------
        residuals : Sequence[float]
            List of residual values.
        title : str
            Plot title.
        loglog : bool
            Whether to use log-log scale.
        filename : str, optional
            If provided, saves the figure to this file. 
        """
        r = np.asarray(residuals)
        if r.size == 0:
            raise ValueError("Empty residual list")
        x = np.arange(1, len(r) + 1)
        fig, ax = plt.subplots()
        if loglog:
            ax.loglog(x, r, marker="x")
        else:
            ax.plot(x, r, marker="x")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMSE")
        ax.set_title(title)
        ax.grid(True, which="both", ls=":")
        fig.tight_layout()
        if filename:
            self._save_if_needed(fig, filename)
        return fig, ax
    

class Visualisation2D(Visualisation):
    """
    3D visualisation tools for multi-probe ptychography simulations.

    Includes:
      - plot_grid(): display all probe slices as a grid.
      - build_slider_widget(): interactive z-slice visualisation.
    """

    def __init__(self, simulation_space, results_dir: Optional[str] = None):
        """
        Parameters
        ----------
        simulation_space : object
            Must provide attributes x, y, z, num_probes, scan_points, bc_type, and nz.
        results_dir : str, optional
            Directory to save plots. If None, plots are not saved automatically.
        """
        super().__init__(simulation_space, results_dir)
        self.x_lims = simulation_space.continuous_dimensions[0]
        self.y_lims = simulation_space.continuous_dimensions[1]
        self.z = simulation_space.z

    # ------------------------------------------------------------------
    # Grid plot for multiple probes
    # ------------------------------------------------------------------
    def plot_grid(self, solution: np.ndarray, *,
                  view: str = "phase_amp",
                  title: str = "",
                  z_step: int = 0,
                  filename: Optional[str] = None,
                  axis_ticks: str = "real") -> Tuple[plt.Figure, np.ndarray, plt.Figure, np.ndarray]:
        """
        Display a grid of probe slices (phase+amplitude or real+imag).
        Intended for multi-probe simulations.

        Parameters
        ----------
        solution : np.ndarray
            4D array [num_probes, x, y, z].
        view : {'phase_amp', 'real_imag'}
            Determines whether to plot (phase, amplitude) or (real, imaginary).
        z_step : int
            Index of z-slice to display.
        filename : str, optional
            Base filename for saved figures.
        axis_ticks : {'real', 'pixels'}
            Whether to label axes with real units or pixel indices.

        Returns
        -------
        (fig1, axes1, fig2, axes2)
        """
        assert solution.ndim == 4, "solution must be a 4D array [num_probes, x, y, z]"
        assert 0 <= z_step < solution.shape[3], "z_step out of bounds"
        assert title is not None or isinstance(title, str), "title must be a string or None"
        assert axis_ticks in ("real", "pixels"), "axis_ticks must be 'real' or 'pixels'"
        extent = [self.x_lims[0], self.x_lims[1], self.y_lims[0], self.y_lims[1]] if axis_ticks == "real" else None

        rows = cols = int(self.simulation_space.scan_points)
        fig1, axes1 = plt.subplots(rows, cols, figsize=(15, 12), squeeze=False)
        fig2, axes2 = plt.subplots(rows, cols, figsize=(15, 12), squeeze=False)

        data1, data2, t1, t2 = self._view_and_title(solution[:, :, :, z_step],
                                                     view, title)
        t1 += f"\n at z={self.z[z_step]:.2f}" if axis_ticks == "real" else f"\n at z-step = {z_step}"
        t2 += f"\n at z={self.z[z_step]:.2f}" if axis_ticks == "real" else f"\n at z-slice = {z_step}"

        im1 = im2 = None
        num_probes = solution.shape[0]
        for idx in range(num_probes):
            r, c = divmod(idx, cols)
            if r % 2:  # zig-zag order for scanning
                c = cols - 1 - c
            im1 = axes1[r, c].imshow(data1[idx], origin="lower", cmap="viridis", extent=extent)
            im2 = axes2[r, c].imshow(data2[idx], origin="lower", cmap="viridis", extent=extent)
            axes1[r, c].set_title(f"Scan {idx}")
            axes2[r, c].set_title(f"Scan {idx}")

        if im1 is not None:
            fig1.colorbar(im1, ax=axes1, shrink=0.6)
        if im2 is not None:
            fig2.colorbar(im2, ax=axes2, shrink=0.6)

        fig1.suptitle(f"{t1} Grid", fontsize=14)
        fig2.suptitle(f"{t2} Grid", fontsize=14)
        fig1.tight_layout(); fig2.tight_layout()

        self._save_if_needed(fig1, filename or f"grid_{t1.lower()}.png")
        self._save_if_needed(fig2, filename or f"grid_{t2.lower()}.png")

        return fig1, axes1, fig2, axes2

    # ------------------------------------------------------------------
    # Interactive z-slice widget
    # ------------------------------------------------------------------
    def build_slider_widget(self, solution: np.ndarray, *,
                            view: str = "phase_amp",
                            mode: str = "forward",
                            title: Optional[str] = None,
                            axis_ticks: str = "real"):
        """
        Interactive slider for visualising z-slices of a 3D field.

        Parameters
        ----------
        solution : np.ndarray
            3D field array (x, y, z).
        view : {'phase_amp', 'real_imag'}
            Display type.
        mode : {'forward', 'reverse'}
            Determines direction of z traversal.
        title : str, optional
            Custom plot title.

        axis_ticks : {'real', 'pixels'}
            Whether to label axes with real units or pixel indices.
        """
        assert solution.ndim == 3, "solution must be a 3D array"
        assert title is not None or isinstance(title, str), "title must be a string or None"
        assert mode in ("forward", "reverse"), "mode must be 'forward' or 'reverse'"
        assert axis_ticks in ("real", "pixels"), "axis_ticks must be 'real' or 'pixels'"

        extent = [self.x_lims[0], self.x_lims[1], self.y_lims[0], self.y_lims[1]] if axis_ticks == "real" else None
        if title in (None, ""):
            title = f" ({self.simulation_space.bc_type}, {mode})"

        len_z = solution.shape[2]

        data1, data2, title1, title2 = self._view_and_title(solution,
                                                     view, title)

        vmin1, vmax1 = np.min(data1), np.max(data1)
        vmin2, vmax2 = np.min(data2), np.max(data2)

        def update(frame):
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            z_frame = self.z[-(frame + 1)] if mode == "reverse" else self.z[frame]
            z_title = f"\n at z={self.z[z_frame]:.2f}" if axis_ticks == "real" else f"\n at z-step = {z_frame}"
            im0 = axs[0].imshow(data1[:, :, frame], cmap='viridis', origin='lower',
                                 extent=extent,
                                 vmin=vmin1, vmax=vmax1)
            axs[0].set_title(f'{title1}{z_title}')
            axs[0].set_xlabel('X'); axs[0].set_ylabel('Y')
            fig.colorbar(im0, ax=axs[0])

            im1 = axs[1].imshow(data2[:, :, frame], cmap='viridis', origin='lower',
                                 extent=extent,
                                 vmin=vmin2, vmax=vmax2)
            axs[1].set_title(f'{title2}{z_title}')
            axs[1].set_xlabel('X'); axs[1].set_ylabel('Y')
            fig.colorbar(im1, ax=axs[1])

            fig.tight_layout()
            plt.show()

        interact(update, frame=IntSlider(min=0, max=len_z - 1, step=1, value=0, description="slice"))