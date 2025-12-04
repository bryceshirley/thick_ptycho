"""
Base class for PtychoObject1D and PtychoObject2D defining common logic.
"""

from abc import ABC

import numpy as np
from PIL import Image

from .shapes import OpticalShape


class BasePtychoObject(ABC):
    """
    Abstract base class for ptychographic objects in a simulation space.
    """

    def __init__(self, simulation_space):
        self.simulation_space = simulation_space
        self.optical_shapes = []
        self.refractive_index = simulation_space.refractive_index_empty

    def build_field(self):
        """Generate total refractive index field."""
        for shape in self.optical_shapes:
            self.refractive_index += shape.get_refractive_index_field()
        # return self.refractive_index

    # Method to save refractive_index to npz file
    def save_refractive_index(self, file_path: str):
        """
        Save the refractive index field to a .npy file.

        Parameters:
        filename (str): Path to the output .npy file.
        """
        self.simulation_space._log(f"Saving refractive index field to {file_path}")
        np.save(file_path, self.refractive_index)

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
        self.optical_shapes.append(
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
        self.refractive_index = (
            self.simulation_space.n_medium
            - real_perturbation * img_norm
            - 1j * imaginary_perturbation * img_norm
        )
        return self.refractive_index

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
        if loaded_n.shape != self.refractive_index.shape:
            raise ValueError("Loaded sample shape does not match simulation space shape.")

        # Normalize the refractive index field and rescale it
        refractive_index = (loaded_n - np.mean(loaded_n)) / np.std(loaded_n) + 1
        self.refractive_index = self.simulation_space.n_medium - (real_perturbation * refractive_index) - (imaginary_perturbation * 1j * refractive_index)
