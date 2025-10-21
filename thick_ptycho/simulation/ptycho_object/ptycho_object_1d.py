"""
1D implementation of a ptychographic sample object.
"""

import numpy as np
from .base_object import BasePtychoObject
from .shapes import OpticalObject


class PtychoObject1D(BasePtychoObject):
    """1D ptychographic object in x–z space."""

    def add_shape(self, shape, refractive_index, side_length, centre, depth, gaussian_blur=None):
        """
        Add a 1D shape defined by fractional parameters (0–1).

        centre: (x_frac, z_frac)
        side_length: fraction of total x-range
        depth: fraction of total z-range
        """
        sim = self.simulation_space
        xlims, zlims = sim.xlims, sim.zlims

        # Convert fractions to absolute coordinates
        abs_centre = (
            xlims[0] + centre[0] * (xlims[1] - xlims[0]),
            zlims[0] + centre[1] * (zlims[1] - zlims[0])
        )
        abs_side = side_length * (xlims[1] - xlims[0])
        abs_depth = depth * (zlims[1] - zlims[0])

        # Clamp to fit within the domain
        abs_centre = self._clamp(abs_centre, abs_side, abs_depth, xlims, zlims)

        obj = OpticalObject(abs_centre, shape, refractive_index,
                            abs_side, abs_depth, sim.nx, sim.nz,
                            sim.x, sim.z, gaussian_blur)
        self.objects.append(obj)

    def build_field(self):
        """Generate total refractive index field."""
        for obj in self.objects:
            self.n_true += obj.get_refractive_index_field()
        return self.n_true

    def _clamp(self, centre, side, depth, xlims, zlims):
        cx, cz = centre
        cx = np.clip(cx, xlims[0] + side/2, xlims[1] - side/2)
        cz = np.clip(cz, zlims[0] + depth/2, zlims[1] - depth/2)
        return cx, cz
