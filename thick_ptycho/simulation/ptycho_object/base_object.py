"""
Base class for PtychoObject1D and PtychoObject2D defining common logic.
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class BasePtychoObject(ABC):
    """
    Abstract base class for ptychographic objects in a simulation space.
    """

    def __init__(self, simulation_space):
        self.simulation_space = simulation_space
        self.objects = []
        self.n_true = np.ones(simulation_space.discrete_dimensions, dtype=complex) * simulation_space.n_medium

    @abstractmethod
    def add_shape(self, *args, **kwargs):
        """Add a geometric shape to the ptychographic object."""
        pass

    @abstractmethod
    def build_field(self):
        """Construct the total refractive index field."""
        pass

    # ---------------------------
    # Utility visualization
    # ---------------------------

    def plot_refractive_index(self, z_slice: int = None):
        """
        Visualize the refractive index field.
        For 1D: plots n(x)
        For 2D: plots n(x, y) at a given z-slice.
        """
        n_real = np.real(self.n_true)

        if len(self.n_true.shape) == 2:
            # 1D: shape (nx, nz)
            plt.figure(figsize=(6, 3))
            plt.imshow(n_real.T, extent=[*self.simulation_space.xlims, *self.simulation_space.zlims],
                       aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label="Re(n)")
            plt.title("1D Refractive Index Field (x-z plane)")
            plt.xlabel("x")
            plt.ylabel("z")
            plt.show()

        elif len(self.n_true.shape) == 3:
            # 2D: shape (nx, ny, nz)
            if z_slice is None:
                z_slice = self.n_true.shape[2] // 2
            plt.figure(figsize=(5, 5))
            plt.imshow(n_real[:, :, z_slice], cmap='viridis', origin='lower')
            plt.title(f"2D Refractive Index Field (z={z_slice})")
            plt.colorbar(label="Re(n)")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
