import numpy as np
from scipy.special import j1

import scipy.sparse as sp


def u0_nm_neumann(n, m):
    """Returns the exact solution for a given n and m."""
    return lambda x, y: np.cos(n * np.pi * x) * np.cos(m * np.pi * y)


def u0_nm_dirichlet(n, m):
    """Returns the exact solution for a given n and m."""
    return lambda x, y: np.sin(n * np.pi * x) * np.sin(m * np.pi * y)


class InitialConditions:
    """
    Handles initial conditions in 1D or 2D and sets up the system matrices.
    """

    def __init__(self, sample_space, thin_sample: bool = False):
        self.probe_type = sample_space.probe_type
        self.sample_space = sample_space
        self.bc_type = sample_space.bc_type
        self.k = sample_space.k  # wave number
        self.probe_diameter = sample_space.probe_diameter * self.sample_space.dx
        self.dimension = sample_space.dimension
        self.thin_sample = thin_sample

        # For thin samples, use sub-sampling
        if thin_sample:
            self.nx = self.sample_space.sub_nx
            if self.sample_space.dimension == 2:
                self.ny = self.sample_space.sub_ny
        else:
            self.nx = self.sample_space.nx
            if self.sample_space.dimension == 2:
                self.ny = self.sample_space.ny

        if self.sample_space.dimension == 1:
            self.block_size = self.nx
        else:
            self.block_size = self.nx * self.ny

    def apply_initial_condition(self, scan, probe_type=None):
        """Apply initial condition to the solution grid, excluding boundaries if dirichlet."""
        if probe_type is None:
            probe_type = self.probe_type

        if self.dimension == 1:
            if self.thin_sample:
                x = self.sample_space.detector_frame_info[scan]["sub_dimensions"][0]
            else:
                x = self.sample_space.x

            initial_solution = np.zeros(
                (self.nx), dtype=complex)

            if self.bc_type == "dirichlet":
                x_interior = x[1:-1]
            else:
                x_interior = x

            for i, xval in enumerate(x_interior):
                initial_solution[i] = self.get_initial_condition_value(
                    xval, scan=scan, probe_type=probe_type)
            return initial_solution

        elif self.dimension == 2:
            if self.thin_sample:
                x, y = self.sample_space.detector_frame_info[scan]["sub_dimensions"]
            else:
                x, y = self.sample_space.x, self.sample_space.y

            initial_solution = np.zeros((self.nx, self.ny), dtype=complex)

            if self.bc_type == "dirichlet":
                x_interior = x[1:-1]
                y_interior = y[1:-1]
            else:
                x_interior = x
                y_interior = y

            for i, xval in enumerate(x_interior):
                for j, yval in enumerate(y_interior):
                    initial_solution[i, j] = self.get_initial_condition_value(
                        xval, yval, scan=scan, probe_type=probe_type)
            return initial_solution
        else:
            raise ValueError(
                "Unsupported dimension: {}".format(self.dimension))

    def get_initial_condition_value(self, x, y=None, scan=0, probe_type=None):
        """
        Choose which initial condition to use, options are:

        1. Constant x = 1
        2. Gaussian centered at (c_x, c_y) with standard deviation sd
        3. Single sinusoidal bump
        4. Complex exponential with real part from 3
        5. Sum of several sinusoids to check with exact solution for dirichlet
        6. Sum of several cosinusoids to check with exact solution for neumann
        7. Airy disk
        8. Disk

        Returns:
            initial condition value at (x, y)
        """
        if self.dimension == 1:
            c_x = self.sample_space.detector_frame_info[scan]["probe_centre_continuous"]
        else:
            c_x, c_y = self.sample_space.detector_frame_info[scan]["probe_centre_continuous"]

        if probe_type is None:
            probe_type = self.probe_type

        if probe_type == "constant":
            return 0
        elif probe_type == "gaussian":
            sd =  1
            if self.dimension == 1:
                dx = (x - c_x) / sd
                return 1 / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (dx ** 2))
            else:
                dx = (x - c_x) / sd
                dy = (y - c_y) / sd
                norm = 1 / (2 * np.pi * sd ** 2)
                return norm * np.exp(-0.5 * (dx ** 2 + dy ** 2))
        elif probe_type == "sinusoidal":
            if self.dimension == 1:
                return np.sin(np.pi * x)
            else:
                return np.sin(np.pi * x) * np.sin(np.pi * y)
        elif probe_type == "complex_exp":
            if self.dimension == 1:
                return -1j * np.exp(1j * np.pi * x)
            else:
                return np.exp(1j * np.pi * (x + y))
        elif probe_type == "dirichlet_test":
            if self.dimension == 1:
                return (u0_nm_dirichlet(1, 1)(x, 0.5) +
                        0.5 * u0_nm_dirichlet(5, 5)(x, 0.5) +
                        0.2 * u0_nm_dirichlet(9, 9)(x, 0.5))
            else:
                return (u0_nm_dirichlet(1, 1)(x, y) +
                        0.5 * u0_nm_dirichlet(5, 5)(x, y) +
                        0.2 * u0_nm_dirichlet(9, 9)(x, y))
        elif probe_type == "neumann_test":
            if self.dimension == 1:
                return (u0_nm_neumann(1, 1)(x, 0) +
                        0.5 * u0_nm_neumann(2, 2)(x, 0) +
                        0.2 * u0_nm_neumann(5, 5)(x, 0))
            else:
                return (u0_nm_neumann(1, 1)(x, y) +
                        0.5 * u0_nm_neumann(2, 2)(x, y) +
                        0.2 * u0_nm_neumann(5, 5)(x, y))
        elif probe_type == "airy_disk":
            if self.dimension == 1:
                r = np.abs(x - c_x)
                kr = self.k * r / 10
                airy = np.where(r != 0, np.divide(
                    (2 * j1(kr))**2, (kr)**2, where=kr != 0), 1.0)
                return airy
            else:
                r = np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2)
                kr = self.k * r
                airy = np.ones_like(r, dtype=float) if isinstance(
                    r, np.ndarray) else 1.0
                mask = r != 0
                if isinstance(r, np.ndarray):
                    airy[mask] = ((2 * j1(kr[mask])) / (kr[mask])) ** 2
                elif r != 0:
                    airy = ((2 * j1(kr)) / (kr)) ** 2
                return airy
        elif probe_type == "disk":
            if self.dimension == 1:
                r = np.abs(x - c_x)
                return np.where(r <= self.probe_diameter / 2, 1, 0)
            else:
                r = np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2)
                return np.where(r <= self.probe_diameter / 2, 1, 0)
        else:
            raise ValueError("Not a valid initial condition type")

    def return_probes(self, probe_type: str = None):
        """
        Returns the initial condition for the probe for all scans.
        """
        if probe_type is None:
            probe_type = self.probe_type

        probes = np.zeros(
            (self.sample_space.num_probes, self.block_size), dtype=complex)

        for scan_index in range(self.sample_space.num_probes):
            probes[scan_index, :] = self.apply_initial_condition(
                scan_index,
                probe_type=probe_type).flatten()

        return probes
