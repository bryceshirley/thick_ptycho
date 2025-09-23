import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Sequence, Tuple, Union

try:
    import ipywidgets as widgets
    from ipywidgets import IntSlider, VBox, HBox, interactive_output
    _HAS_WIDGETS = True
except Exception:
    _HAS_WIDGETS = False


class Visualisation:
    """Visualisation for results (single, grid, slider)."""

    def __init__(self, sample_space, results_dir: Optional[Union[str, os.PathLike]] = None):
        """
        Parameters
        ----------
        sample_space : object
            Provides grid info (x, y, z, num_probes, dimension, etc.).
        results_dir : str or PathLike, optional
            Directory where plots are saved. If None, nothing is saved.
        """
        self.sample_space = sample_space
        self.bc_type = getattr(sample_space, "bc_type", "unknown")
        self.num_probes = getattr(sample_space, "num_probes", 1)
        self.x = getattr(sample_space, "x", np.arange(1))
        self.z = getattr(sample_space, "z", np.arange(1))
        self.y = getattr(sample_space, "y", None) if getattr(sample_space, "dimension", 1) == 2 else None
        self.dimension = getattr(sample_space, "dimension", 1)
        self.scan_points = getattr(sample_space, "scan_points", max(1, int(np.sqrt(self.num_probes))))

        self.results_dir = os.fspath(results_dir) if results_dir else None
        if self.results_dir:
            os.makedirs(self.results_dir, exist_ok=True)

        self._figs = []
        self._widgets = []

    def plot_auto(self, solution: np.ndarray, *,
                  view: str = "phase_amp", layout: str = "single",
                  time: str = "final", probe_index: Optional[int] = None,
                  title_prefix: str = "",
                  axes: Optional[Sequence] = None,
                  filename: Optional[str] = None):
        """Automatically choose plotting style."""
        if probe_index is None:
            probe_index = int(self.num_probes * 0.5)

        if layout == "slider":
            return self.build_slider_widget(solution, view=view, time=time, probe_index=probe_index)

        if self.dimension == 1:
            if solution.ndim == 3:
                sel = 0 if self.num_probes == 1 else probe_index
                sol = solution[sel, :, :]
            else:
                sol = solution
            return self.plot_single(sol, view=view, time=time,title_prefix=title_prefix,
                                    axes=axes, filename=filename)

        if layout == "grid" and self.num_probes > 1:
            return self.plot_grid(solution, view=view, time=time, filename=filename)

        if solution.ndim == 4:  # (probes, x, y, z)
            sol = solution[0, ...]
        else:
            sol = solution
        return self.plot_single(sol, view=view, time=time,
                                axes=axes, filename=filename)

    def plot_single(self, solution: np.ndarray, *,
                view: str = "phase_amp", time: str = "final",
                axes: Optional[Sequence] = None,
                filename: Optional[str] = None,
                title_prefix: Optional[str] = None,
                title_left: Optional[str] = None,
                title_right: Optional[str] = None,
                xlabel_left: Optional[str] = None,
                ylabel_left: Optional[str] = None,
                xlabel_right: Optional[str] = None,
                ylabel_right: Optional[str] = None
                ) -> Tuple[plt.Figure, Sequence]:
        """Two panels: Phase+Amplitude or Real+Imag.

        Overrides:
        - title_left/title_right
        - xlabel_left/ylabel_left, xlabel_right/ylabel_right
        """
        if self.dimension == 2:
            idx = 0 if time == "initial" else -1
            to_plot = solution[:, :, idx]
            dim1, dim2 = "X", "Y"
        else:
            to_plot = solution
            dim1, dim2 = "Z", "X"

        if axes is not None and len(axes) == 2:
            fig = axes[0].figure
            ax0, ax1 = axes
        else:
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

        if view == "phase_amp":
            p1, p2 = self.phase(to_plot), np.abs(to_plot)
            t1_default, t2_default = "Phase", "Amplitude"
        else:
            p1, p2 = to_plot.real, to_plot.imag
            t1_default, t2_default = "Real", "Imaginary"

        if title_prefix:
            t1_default = title_prefix + t1_default
            t2_default = title_prefix + t2_default

        # Left panel
        im0 = ax0.imshow(p1, cmap="viridis", origin="lower")
        fig.colorbar(im0, ax=ax0)
        ax0.set_title(title_left if title_left is not None else t1_default)
        ax0.set_xlabel(xlabel_left if xlabel_left is not None else dim1)
        ax0.set_ylabel(ylabel_left if ylabel_left is not None else dim2)

        # Right panel
        im1 = ax1.imshow(p2, cmap="viridis", origin="lower")
        fig.colorbar(im1, ax=ax1)
        ax1.set_title(title_right if title_right is not None else t2_default)
        ax1.set_xlabel(xlabel_right if xlabel_right is not None else dim1)
        ax1.set_ylabel(ylabel_right if ylabel_right is not None else dim2)

        fig.tight_layout()
        self._save_if_needed(fig, filename or f"{t1_default.lower()}_{t2_default.lower()}.png")
        self._figs.append(fig)
        return fig, (ax0, ax1)

    def plot_grid(self, solution: np.ndarray, *,
                  view: str = "phase_amp", time: str = "final",
                  filename: Optional[str] = None):
        """Grid for multi-probe data."""
        rows = cols = int(self.scan_points)
        fig1, axes1 = plt.subplots(rows, cols, figsize=(15, 12), squeeze=False)
        fig2, axes2 = plt.subplots(rows, cols, figsize=(15, 12), squeeze=False)

        k = 0 if time == "initial" else -1

        if view == "phase_amp":
            data1, data2 = self.phase(solution[:, :, :, k]), np.abs(solution[:, :, :, k])
            t1, t2 = "Phase", "Amplitude"
        else:
            data1, data2 = solution[:, :, :, k].real, solution[:, :, :, k].imag
            t1, t2 = "Real", "Imaginary"

        im1 = im2 = None
        for idx in range(self.num_probes):
            r, c = divmod(idx, cols)
            if r % 2:  # zig-zag order
                c = cols - 1 - c
            im1 = axes1[r, c].imshow(data1[idx], origin="lower", cmap="viridis")
            axes1[r, c].set_title(f"Scan {idx}")
            im2 = axes2[r, c].imshow(data2[idx], origin="lower", cmap="viridis")
            axes2[r, c].set_title(f"Scan {idx}")

        if im1 is not None: fig1.colorbar(im1, ax=axes1)
        if im2 is not None: fig2.colorbar(im2, ax=axes2)
        fig1.tight_layout(); fig2.tight_layout()

        self._save_if_needed(fig1, filename or f"grid_{t1.lower()}.png")
        self._save_if_needed(fig2, filename or f"grid_{t2.lower()}.png")
        self._figs += [fig1, fig2]
        return fig1, axes1, fig2, axes2

    def build_slider_widget(self, solution: np.ndarray, *,
                            view: str = "phase_amp"):
        """Interactive slice slider (Jupyter)."""
        if not _HAS_WIDGETS:
            raise RuntimeError("ipywidgets not available")

        nz = getattr(self.sample_space, "nz", solution.shape[-1])
        if view == "phase_amp":
            d1, d2 = self.phase(solution), np.abs(solution)
            t1, t2 = "Phase", "Amplitude"
        else:
            d1, d2 = solution.real, solution.imag
            t1, t2 = "Real", "Imaginary"

        extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()] if self.dimension == 2 and self.y is not None else None
        slider = IntSlider(min=0, max=nz - 1, step=1, value=0, description="slice")
        out1, out2 = widgets.Output(), widgets.Output()

        def _update(frame):
            with out1:
                out1.clear_output(wait=True)
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                im = ax.imshow(d1[:, :, frame], cmap="viridis", origin="lower", extent=extent)
                fig.colorbar(im, ax=ax); ax.set_title(f"{t1} z={frame}")
                fig.tight_layout(); #self._figs.append(fig)
                fig.show()
            with out2:
                out2.clear_output(wait=True)
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                im = ax.imshow(d2[:, :, frame], cmap="viridis", origin="lower", extent=extent)
                fig.colorbar(im, ax=ax); ax.set_title(f"{t2} z={frame}")
                fig.tight_layout(); #self._figs.append(fig)
                fig.show()

        interactive_output(_update, {"frame": slider})
        box = VBox([slider, HBox([out1, out2])])
        #self._widgets.append(box)
        return box

    def show(self, clear_queue: bool = True):
        """Display queued figures and widgets."""
        for fig in self._figs:
            fig.show()
        if self._widgets:
            try:
                from IPython.display import display
                for w in self._widgets:
                    display(w)
            except Exception:
                pass
        if clear_queue:
            self._figs.clear(); self._widgets.clear()

    @staticmethod
    def phase(arr: np.ndarray, tol: float = 5e-3) -> np.ndarray:
        """Phase with near-zero magnitude masked to 0."""
        ph = np.angle(arr)
        mag = np.abs(arr)
        return np.where(mag < tol, 0.0, ph)

    def _save_if_needed(self, fig: plt.Figure, filename: str):
        """Save figure if results_dir is set."""
        if not self.results_dir:
            return
        path = os.path.join(self.results_dir, filename)
        fig.savefig(path, bbox_inches="tight")

    def plot_refractive_index(self, n: Optional[np.ndarray] = None, *,
                          title: str = "Refractive Index",
                          filename: Optional[str] = None):
        """
        Plot phase and amplitude of a refractive index field.
        """
        n_true = getattr(self.sample_space, "n_true", None)
        if n is None:
            if n_true is None:
                raise ValueError("No refractive index provided and sample_space.n_true missing")
            n = n_true

        # scaling from true object if available
        if n_true is not None:
            vmin_phase, vmax_phase = np.min(np.angle(n_true)), np.max(np.angle(n_true))
            vmin_amp, vmax_amp = np.min(np.abs(n_true)), np.max(np.abs(n_true))
        else:
            vmin_phase, vmax_phase = np.min(np.angle(n)), np.max(np.angle(n))
            vmin_amp, vmax_amp = np.min(np.abs(n)), np.max(np.abs(n))

        phase = np.angle(n)
        amplitude = np.abs(n)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        im0 = axs[0].imshow(phase, origin="lower", aspect="auto",
                            cmap="viridis", vmin=vmin_phase, vmax=vmax_phase)
        axs[0].set_title(f"Phase of {title}")
        axs[0].set_xlabel("z (pixels)"); axs[0].set_ylabel("x (pixels)")
        fig.colorbar(im0, ax=axs[0], label="Phase (radians)")

        im1 = axs[1].imshow(amplitude, origin="lower", aspect="auto",
                            cmap="viridis", vmin=vmin_amp, vmax=vmax_amp)
        axs[1].set_title(f"Amplitude of {title}")
        axs[1].set_xlabel("z (pixels)"); axs[1].set_ylabel("x (pixels)")
        fig.colorbar(im1, ax=axs[1], label="Amplitude")

        fig.tight_layout()
        self._save_if_needed(fig, filename or f"{title.replace(' ', '_').lower()}.png")
        self._figs.append(fig)
        return fig, axs
    
    def plot_residual(self, residual_history, *,
                  title: str = "Residual History of Least Squares Objective Function",
                  loglog: bool = True,
                  ax=None,
                  filename: str = "residual_history.png"):
        """
        Plot residual history (e.g., RMSE per iteration).
        """
        r = np.asarray(residual_history, dtype=float)
        if r.size == 0:
            raise ValueError("residual_history is empty")

        x = np.arange(1, r.size + 1)
        # avoid log(0) if there are zeros
        r_plot = np.maximum(r, np.finfo(float).tiny)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.figure

        if loglog:
            ax.loglog(x, r_plot, marker="x", linestyle="-", markersize=3)
        else:
            ax.plot(x, r, marker="x", linestyle="-", markersize=3)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMSE")
        ax.set_title(title)
        ax.grid(True, which="both", ls=":")

        fig.tight_layout()
        self._save_if_needed(fig, filename)
        self._figs.append(fig)
        return fig, ax

