"""
Base class for PtychoObject1D and PtychoObject2D defining common logic.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from .shapes import OpticalShape
import tifffile as tiff
from scipy.ndimage import zoom
from PIL import Image


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

    # Method to save n_true to npz file
    def save_n_true(self, file_path: str):
        """
        Save the refractive index field to a .npy file.

        Parameters:
        filename (str): Path to the output .npy file.
        """
        self.simulation_space._log(f"Saving refractive index field to {file_path}")
        np.save(file_path, self.n_true)

    def add_object(self, shape: str, refractive_index: complex, side_length_factor: float,
                   centre_factor: tuple, depth_factor: float, gaussian_blur=None):
        """
        Add an optical object to the simulation.

        Parameters:
        shape (str): Shape of the object.
        refractive_index (float): Refractive index of the object.
        side_length_factor (float): Side length of the object.
        centre_factor (tuple): Centre of the object as a fraction of simulation space.
        depth_factor (float): Depth of the object.
        """
        assert 0.0 <= centre_factor[0] <= 1.0, "Centre x-factor must be in [0, 1]."
        assert 0.0 <= centre_factor[1] <= 1.0, "Centre z-factor must be in [0, 1]."
        assert 0.0 <= side_length_factor <= 1.0, "Side length factor must be in (0, 1]."
        assert 0.0 <= depth_factor <= 1.0, "Depth factor must be in (0, 1]."
        self.objects.append(
            OpticalShape(
                centre_factor,
                shape,
                refractive_index,
                side_length_factor,
                depth_factor,
                gaussian_blur,
                self.simulation_space))
        


    def load_image(self, file_path: str, real_perturbation=1e-4, imaginary_perturbation=1e-6):
        """
        Load a refractive index field from a TIFF image and rescale to (nx, nz).

        The TIFF is assumed to encode structural contrast; we transform it into:
            n(x,z) = n_medium - real_perturbation * scaled_image
                                - 1j * imaginary_perturbation * scaled_image
        """

        # Load and force grayscale (if TIFF has multiple channels)
        img = Image.open(file_path).convert('L')

        # Target resolution
        nx = self.simulation_space.nx     # number of x-pixels (height)
        nz = self.simulation_space.nz     # number of slices (width)

        # PIL resize expects (width, height)
        img_resized = img.resize((nz, nx), resample=Image.BILINEAR)

        # Convert to float array
        img_array = np.array(img_resized).astype(np.float32)

        img_array = np.flipud(img_array)  # Flip vertically to match coordinate system

        # Normalize image → zero mean, unit variance
        img_norm = (img_array - img_array.mean()) / (img_array.std() + 1e-12)

        # Construct refractive index field
        self.n_true = (
            self.simulation_space.n_medium
            - real_perturbation * img_norm
            - 1j * imaginary_perturbation * img_norm
        )
        return self.n_true

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
            Index for sub-sampling info in scan_frame_info.

        Returns
        -------
        object_steps : ndarray
            Complex-valued slices between adjacent z-layers.
            Shape is same as n except the last axis (z) is reduced by 1.
        """

        # Use the true refractive index field if not provided
        if n is None:
            n = self.n_true

        # Restrict to thin sample region if requested
        if self.simulation_space.solve_reduced_domain:
            sub_limits = self.simulation_space.scan_frame_info[scan_index].sub_limits_discrete
            n = self.get_sub_sample(n, sub_limits)

        # Compute coefficient
        if grad:
            coefficient = (self.simulation_space.k / 1j) * n
        else:
            coefficient = (self.simulation_space.k / 2j) * (n**2 - 1)

        # Compute all half time-step slices along the z-axis
        object_steps = (self.simulation_space.dz / 2) * (coefficient[..., :-1] + coefficient[..., 1:]) / 2

        return object_steps
