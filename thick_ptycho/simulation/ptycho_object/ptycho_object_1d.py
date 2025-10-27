"""
1D implementation of a ptychographic sample object.
"""

import numpy as np
from .base_object import BasePtychoObject


class PtychoObject1D(BasePtychoObject):
    """1D ptychographic object in x–z space."""

    def __init__(self, simulation_space):
        super().__init__(simulation_space)

    def get_sub_sample(self,n,sub_limits):
        """Retrieve the object slices for propagation."""
        x_min,x_max = sub_limits
        return n[
            x_min:x_max,
            :
        ]