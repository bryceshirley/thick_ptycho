"""
1D implementation of a ptychographic sample object.
"""

import numpy as np
from .base_object import BasePtychoObject
from typing import Tuple
import cv2
from skimage import data

class PtychoObject1D(BasePtychoObject):
    """1D ptychographic object in x–z space."""

    def __init__(self, simulation_space):
        super().__init__(simulation_space)

    # Resize complex array from np file
    def load_and_resize_refractive_index(self, file_path: str) -> np.ndarray:
        """ 
        Resize a complex-valued 2D array to the target shape using OpenCV.
        Parameters:
        n : np.ndarray
            Complex-valued input array to be resized.
        """
        print(f"Original refractive index shape: {self.n_true.shape}")
        n = np.load(file_path)
        print(f"Loaded refractive index shape: {n.shape}")
        target_shape = (self.simulation_space.nz, self.simulation_space.nx)
        real_resized = cv2.resize(np.real(n), target_shape, interpolation=cv2.INTER_LINEAR)
        imag_resized = cv2.resize(np.imag(n), target_shape, interpolation=cv2.INTER_LINEAR)
        self.n_true = real_resized + 1j * imag_resized
        print(f"Loaded refractive index new shape: {self.n_true.shape}")

    # From from skimage import data create refractive index from phantom image
    def create_refractive_index_of_phantom(self, real_perturbation=1e-4, imaginary_perturbation=1e-6) -> np.ndarray:
        """Create a refractive index field from built-in phantom image in skimage data.

        Parameters
        ----------
        real_perturbation : float
            Magnitude of perturbation for the real part of the refractive index.
        imaginary_perturbation : float
            Magnitude of perturbation for the imaginary part of the refractive index.

        Returns
        -------
        n_true : np.ndarray
            The constructed refractive index field.
        
        """
        # Load built-in phantom image
        img = data.shepp_logan_phantom()

        # Target resolution
        nx = self.simulation_space.nx     # number of x-pixels (height)
        nz = self.simulation_space.nz     # number of slices (width)

        pad_x = int(0.5 * nx)  # 25% of nx total padding
        pad_z = int(0.5 * nz)  # 25% of nz total padding

        # The region for the resized phantom:
        nx_inner = nx - pad_x   # new height
        nz_inner = nz - pad_z   # new width

        # Resize phantom to the inner region
        img_resized = cv2.resize(img, (nx_inner,nz_inner), interpolation=cv2.INTER_LINEAR)

        # Rotate if required
        img_resized = np.rot90(img_resized, k=1)

        # Create full-sized padded phantom
        full_img = np.zeros((nx, nz), dtype=img_resized.dtype)

        # Compute symmetric pad offsets
        pad_top = (nx - nx_inner) // 2
        pad_left = (nz - nz_inner) // 2

        # Insert the resized phantom into the padded canvas
        full_img[pad_top:pad_top + nx_inner,
                pad_left:pad_left + nz_inner] = img_resized


        # Normalize image → zero mean, unit variance
        img_norm = (img_resized - np.mean(img_resized)) / (np.std(img_resized) + 1e-12)

        # Construct refractive index field
        self.n_true = (
            self.simulation_space.n_medium
            - real_perturbation * img_norm
            - 1j * imaginary_perturbation * img_norm
        )

        
        return self.n_true



    def get_reduced_sample(self,n,scan_index):
        """Retrieve the object slices for propagation."""
        x_min, x_max = self.simulation_space.scan_frame_info[scan_index].reduced_limits_discrete.x
        return n[
            x_min:x_max+1,
            :
        ]