"""
2D implementation of a ptychographic sample object.
"""

import numpy as np
from .base_object import BasePtychoObject
from skimage import data, transform

class PtychoObject2D(BasePtychoObject):
    """2D ptychographic object in x–y–z space."""

    def __init__(self, simulation_space):
        super().__init__(simulation_space)

    def get_reduced_sample(self,n,scan_index):
        """Retrieve the object slices for propagation."""
        x_min,x_max = self.simulation_space.scan_frame_info[scan_index].reduced_limits_discrete.x
        y_min, y_max = self.simulation_space.scan_frame_info[scan_index].reduced_limits_discrete.y    
        return n[
                    x_min:x_max+1,
                    y_min:y_max+1,
                    :
                ]

    def load_cameraman(self, real_perturbation=1e-4, imaginary_perturbation=1e-6):  # TODO: correct for non free space medium
        """
        Load the cameraman image as the refractive index field and resize to sample space dimensions.

        Sets self.n_true to shape (self.nx, self.ny, self.nz).
        """

        # Load cameraman image (assumed grayscale)
        # Load built-in grayscale cameraman image (uint8 -> float32)
        n_true_2d = data.camera().astype(np.float32)

        # Flip the image vertically
        n_true_2d = np.flipud(n_true_2d)

        
        # Normalize to [0, 1] range
        n_true_2d -= n_true_2d.min()  # Ensure zero baseline
        n_true_2d /= n_true_2d.max()  # Normalize to unit peak

        n_true_2d = transform.resize(n_true_2d, (self.nx, self.ny), preserve_range=True)

        # Expand to 3D by repeating along z
        n_true = np.repeat(n_true_2d[:, :, np.newaxis], self.nz, axis=2)

        delta = (real_perturbation * n_true)
        beta = (imaginary_perturbation * n_true)

        # Set refractive index field
        self.n_true = self.n_medium - delta - beta * 1j
    
