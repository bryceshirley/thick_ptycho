"""
1D implementation of a ptychographic sample object.
"""

import numpy as np
from .base_object import BasePtychoObject
from typing import Tuple
import cv2

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

    def get_sub_sample(self,n,sub_limits):
        """Retrieve the object slices for propagation."""
        x_min,x_max = sub_limits
        return n[
            x_min:x_max,
            :
        ]