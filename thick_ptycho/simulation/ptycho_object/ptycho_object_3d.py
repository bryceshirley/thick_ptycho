"""
3d implementation of a ptychographic sample object.
"""

import numpy as np
from skimage import data, transform

from .base_object import BasePtychoObject


class PtychoObject3D(BasePtychoObject):
    """3D ptychographic object in x–y–z space."""

    def __init__(self, simulation_space):
        super().__init__(simulation_space)

    def load_cameraman(
        self, real_perturbation=1e-4, imaginary_perturbation=1e-6
    ):  # TODO: correct for non free space medium
        """
        Load the cameraman image as the refractive index field and resize to sample space dimensions.

        Sets self.refractive_index to shape (self.nx, self.ny, self.nz).
        """

        # Load cameraman image (assumed grayscale)
        # Load built-in grayscale cameraman image (uint8 -> float32)
        refractive_index_3d = data.camera().astype(np.float32)

        # Flip the image vertically
        refractive_index_3d = np.flipud(refractive_index_3d)

        # Normalize to [0, 1] range
        refractive_index_3d -= refractive_index_3d.min()  # Ensure zero baseline
        refractive_index_3d /= refractive_index_3d.max()  # Normalize to unit peak

        refractive_index_3d = transform.resize(
            refractive_index_3d, (self.nx, self.ny), preserve_range=True
        )

        # Expand to 3D by repeating along z
        refractive_index = np.repeat(
            refractive_index_3d[:, :, np.newaxis], self.nz, axis=2
        )

        delta = real_perturbation * refractive_index
        beta = imaginary_perturbation * refractive_index

        # Set refractive index field
        self.refractive_index = self.n_medium - delta - beta * 1j
