"""
Base class for PtychoObject1D and PtychoObject2D defining common logic.
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from .shapes import OpticalShape

class BasePtychoObject(ABC):
    """
    Abstract base class for ptychographic objects in a simulation space.
    """

    def __init__(self, simulation_space):
        self.simulation_space = simulation_space
        self.objects = []
        self.n_true = np.ones(simulation_space.discrete_dimensions, dtype=complex) * simulation_space.n_medium

    @abstractmethod
    def get_sub_sample(self, *args, **kwargs):
        """Retrieve the object slices for propagation."""
        pass

    def build_field(self):
        """Generate total refractive index field."""
        for obj in self.objects:
            self.n_true += obj.get_refractive_index_field()
        # return self.n_true
    
    def add_object(self, shape: str, refractive_index: complex, side_length: float, 
                   centre: tuple, depth: float, gaussian_blur=None):
        """
        Add an optical object to the simulation.

        Parameters:
        shape (str): Shape of the object.
        refractive_index (float): Refractive index of the object.
        side_length (float): Side length of the object.
        centre (tuple): Centre of the object.
        depth (float): Depth of the object.
        """
        self.objects.append(
            OpticalShape(
                centre,
                shape,
                refractive_index,
                side_length,
                depth,
                gaussian_blur,
                self.simulation_space))

    def load_sample_object_from_file(self, filepath: str, real_perturbation=1e-4, imaginary_perturbation=1e-6):
        """
        Load a refractive index field from a .npy file.

        Parameters
        ----------
        filepath : str
            Path to the .npy file containing the refractive index field.
        real_perturbation : float
            Magnitude of random perturbation added to the real part.
        imaginary_perturbation : float
            Magnitude of random perturbation added to the imaginary part.
        """
        loaded_n = np.load(filepath)
        if loaded_n.shape != self.n_true.shape:
            raise ValueError("Loaded sample shape does not match simulation space shape.")

        # Normalize the refractive index field and rescale it
        n_true = (loaded_n - np.mean(loaded_n)) / np.std(loaded_n) + 1
        self.n_true = self.simulation_space.n_medium - (real_perturbation * n_true) - (imaginary_perturbation * 1j * n_true)

    def create_object_contribution(self,n=None, grad=False, scan_index=0):
        """
        Create the field of object slices in free space.

        Works for 2D (x, z) and 3D (x, y, z) refractive index fields.

        Parameters
        ----------
        n : ndarray, optional
            Refractive index field. If None, uses self.n_true.
        grad : bool, optional
            If True, compute gradient coefficient (linear in n).
            If False, compute propagation coefficient (quadratic in n).
        scan_index : int, optional
            Index for sub-sampling info in detector_frame_info.

        Returns
        -------
        object_slices : ndarray
            Complex-valued slices between adjacent z-layers.
            Shape is same as n except the last axis (z) is reduced by 1.
        """

        # Use the true refractive index field if not provided
        if n is None:
            n = self.n_true

        # Restrict to thin sample region if requested
        if self.simulation_space.thin_sample:
            sub_limits = self.simulation_space.detector_frame_info[scan_index]['sub_limits']
            n = self.get_sub_sample(n, sub_limits)

        # Compute coefficient
        if grad:
            coefficient = (self.k / 1j) * n
        else:
            coefficient = (self.k / 2j) * (n**2 - 1)

        # Compute all half time-step slices along the z-axis
        object_slices = (self.dz / 2) * (coefficient[..., :-1] + coefficient[..., 1:]) / 2

        return object_slices

    # def plot_refractive_index(self, z_slice: int = None):
    #     """
    #     Visualize the refractive index field.
    #     For 1D: plots n(x)
    #     For 2D: plots n(x, y) at a given z-slice.
    #     """
    #     n_real = np.real(self.n_true)

    #     if len(self.n_true.shape) == 2:
    #         # 1D: shape (nx, nz)
    #         plt.figure(figsize=(6, 3))
    #         plt.imshow(n_real.T, extent=[*self.simulation_space.xlims, *self.simulation_space.zlims],
    #                    aspect='auto', origin='lower', cmap='viridis')
    #         plt.colorbar(label="Re(n)")
    #         plt.title("1D Refractive Index Field (x-z plane)")
    #         plt.xlabel("x")
    #         plt.ylabel("z")
    #         plt.show()

    #     elif len(self.n_true.shape) == 3:
    #         # 2D: shape (nx, ny, nz)
    #         if z_slice is None:
    #             z_slice = self.n_true.shape[2] // 2
    #         plt.figure(figsize=(5, 5))
    #         plt.imshow(n_real[:, :, z_slice], cmap='viridis', origin='lower')
    #         plt.title(f"2D Refractive Index Field (z={z_slice})")
    #         plt.colorbar(label="Re(n)")
    #         plt.xlabel("x")
    #         plt.ylabel("y")
    #         plt.show()
