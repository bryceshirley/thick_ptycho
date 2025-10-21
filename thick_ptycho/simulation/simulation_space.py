"""
Defines the physical space and scan path for a ptychographic simulation.
"""
from typing import List, Dict, Tuple, Any
import numpy as np

class SimulationFactory:
    """Factory for creating 1D or 2D simulation spaces."""
    @staticmethod
    def create(continuous_dimensions, *args, **kwargs):
        dim = len(continuous_dimensions) - 1
        if dim == 1:
            return SimulationSpace1D(continuous_dimensions, *args, **kwargs)
        elif dim == 2:
            return SimulationSpace2D(continuous_dimensions, *args, **kwargs)
        raise ValueError(f"Unsupported dimension: {dim}")


class BaseSimulationSpace:
    """Abstract base defining shared simulation-space logic."""

    def __init__(self, continuous_dimensions, discrete_dimensions, wave_number, bc_type, n_medium=1.0):
        self.continuous_dimensions = continuous_dimensions
        self.discrete_dimensions = discrete_dimensions
        self.bc_type = bc_type.lower()
        self.k = wave_number
        self.wavelength = 2 * np.pi / wave_number
        self.n_medium = complex(n_medium)
        self.objects = []  # later filled by PtychoObject

    def _validate_bc(self):
        if self.bc_type not in ["dirichlet", "neumann", "periodic"]:
            raise ValueError(f"Unsupported BC: {self.bc_type}")

    def get_refractive_index_field(self):
        raise NotImplementedError

    def summarize(self):
        raise NotImplementedError


class SimulationSpace1D(BaseSimulationSpace):
    """1D simulation domain setup."""

    def __init__(self, continuous_dimensions, discrete_dimensions, wave_number, bc_type, **kwargs):
        super().__init__(continuous_dimensions, discrete_dimensions, wave_number, bc_type, **kwargs)
        self.xlims, self.zlims = continuous_dimensions
        self.nx, self.nz = discrete_dimensions

        # Define grid
        self.x = np.linspace(self.xlims[0], self.xlims[1], self.nx)
        self.z = np.linspace(self.zlims[0], self.zlims[1], self.nz)
        self.dx = self.x[1] - self.x[0]
        self.dz = self.z[1] - self.z[0]

    def summarize(self):
        print(f"1D Space: nx={self.nx}, nz={self.nz}, wavelength={self.wavelength:.3e} m")


class SimulationSpace2D(BaseSimulationSpace):
    """2D simulation domain setup."""

    def __init__(self, continuous_dimensions, discrete_dimensions, wave_number, bc_type, **kwargs):
        super().__init__(continuous_dimensions, discrete_dimensions, wave_number, bc_type, **kwargs)
        self.xlims, self.ylims, self.zlims = continuous_dimensions
        self.nx, self.ny, self.nz = discrete_dimensions

        # Define grid
        self.x = np.linspace(self.xlims[0], self.xlims[1], self.nx)
        self.y = np.linspace(self.ylims[0], self.ylims[1], self.ny)
        self.z = np.linspace(self.zlims[0], self.zlims[1], self.nz)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

    def summarize(self):
        print(f"2D Space: nx={self.nx}, ny={self.ny}, nz={self.nz}, λ={self.wavelength:.3e} m")
