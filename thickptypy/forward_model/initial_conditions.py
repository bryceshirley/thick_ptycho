import numpy as np
from scipy.special import j1


def u0_nm_neumann(n, m):
    """Returns the exact solution for a given n and m."""
    return lambda x, y: np.cos(n * np.pi * x) * np.cos(m * np.pi * y)


def u0_nm_dirichlet(n, m):
    """Returns the exact solution for a given n and m."""
    return lambda x, y: np.sin(n * np.pi * x) * np.sin(m * np.pi * y)


def initial_condition_setup(sample_space):
    """
    Interface class to create either 1D or 2D initial conditions.
    """
    if sample_space.dimension == 1:
        return InitialConditions1D(sample_space)
    elif sample_space.dimension == 2:
        return InitialConditions2D(sample_space)
    else:
        raise ValueError("Unsupported dimension: {}".format(sample_space.dimension))


class InitialConditions1D:
    """
    Handles initial conditions in 1D and sets up the system matrices.
    """

    def __init__(self, sample_space):
        self.ic_type = sample_space.probe_type
        self.sample_space = sample_space
        self.bc_type = sample_space.bc_type
        self.k = sample_space.k  # wave number
        self.probe_diameter = sample_space.probe_diameter * self.sample_space.dx

        # Define initial condition function
        self.initial_condition_func = self._define_incidence()

    def get_initial_condition_value(self, x, scan=0):
        """Return the initial condition value at point (x)."""
        cx = self.sample_space.detector_frame_info[scan]["probe_centre_continuous"]
        return self.initial_condition_func(x, cx)

    def apply_initial_condition(self, scan, thin_sample):
        """Apply initial condition to the solution grid excluding boundaries
        if dirichlet (bc_type == 1)."""
        if thin_sample:
            x = self.sample_space.detector_frame_info[scan]["sub_dimensions"][0]
            initial_solution = np.zeros((self.sample_space.sub_nx),
                                        dtype=complex)
        else:
            x = self.sample_space.x
            initial_solution = np.zeros((self.sample_space.nx),
                                        dtype=complex)

        if self.bc_type == "dirichlet":
            x_interior = x[1:-1]
        else:
            x_interior = x

        for i, x in enumerate(x_interior):
            initial_solution[i] = self.get_initial_condition_value(x, scan)

        return initial_solution

    def _define_incidence(self):
        """
        Choose which initial condition to use, options are:

        1. Constant x = 1 ("const1")
        2. Gaussian centered at (x, y) = (0.5, 0.5) with standard deviation sd ("gaussian")
        3. Single sinusoidal bump ("sinusoidal")
        4. Complex exponential with real part from 3 ("complex_exp")
        5. Sum of several sinusoids to check with exact solution for dirichlet ("dirichlet_sum")
        6. Sum of several cosinusoids to check with exact solution for neumann and impedance ("neumann_sum")
        7. Airy disk ("airy_disk")
        8. Disk ("disk")

        Choices:
            "const1", "gaussian", "sinusoidal", "complex_exp", "dirichlet_sum", "neumann_sum", "airy_disk", "disk"

        Returns:
            incidence (function): A function that takes two arguments (x, y) and returns the initial condition value at that point.

        Raises:
            ValueError: If `self.initial_condition` is not one of the valid options.
        """
        if self.ic_type == "constant":
            def incidence(x, c_x):
                return 1
        elif self.ic_type == "gaussian":
            def incidence(x, c_x):
                sd = 0.05
                dx = (x - c_x) / sd
                return 1 / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (dx ** 2))
        elif self.ic_type == "sinusoidal":
            def incidence(x, c_x):
                return np.sin(np.pi * x)
        elif self.ic_type == "complex_exp":
            def incidence(x, c_x):
                return -1j * np.exp(1j * np.pi * x)
        elif self.ic_type == "dirichlet_test":
            def incidence(x, c_x):
                return (u0_nm_dirichlet(1, 1)(x, 0.5) +
                        0.5 * u0_nm_dirichlet(5, 5)(x, 0.5) +
                        0.2 * u0_nm_dirichlet(9, 9)(x, 0.5))
        elif self.ic_type == "neumann_test":
            def incidence(x, c_x):
                return (u0_nm_neumann(1, 1)(x, 0) +
                        0.5 * u0_nm_neumann(2, 2)(x, 0) +
                        0.2 * u0_nm_neumann(5, 5)(x, 0))
        elif self.ic_type == "airy_disk":
            def incidence(x, c_x):
                r = np.abs(x - c_x)
                kr = self.k * r
                # Compute the Airy disk
                return np.where(r != 0, np.divide(
                    (2 * j1(kr))**2, (kr)**2, where=kr != 0), 1)
        elif self.ic_type == "disk":
            def incidence(x, c_x):
                r = np.abs(x - c_x)
                return np.where(r <= self.probe_diameter / 2, 1, 0)
        else:
            raise ValueError("not a valid initial condition type")
        return incidence


class InitialConditions2D:
    """
    Handles initial conditions in 2D and sets up the system matrices.
    """

    def __init__(self, sample_space):
        self.ic_type = sample_space.probe_type
        self.sample_space = sample_space
        self.bc_type = sample_space.bc_type
        self.k = sample_space.k  # wave number
        self.probe_diameter = sample_space.probe_diameter * self.sample_space.dx

        # Define initial condition function
        self.initial_condition_func = self._define_incidence()

    def get_initial_condition_value(self, x, y, scan=0):
        """Return the initial condition value at point (x,y)."""
        cx, cy = self.sample_space.detector_frame_info[scan]["probe_centre_continuous"]
        return self.initial_condition_func(x, y, cx, cy)

    def apply_initial_condition(self, scan, thin_sample):
        """Apply initial condition to the solution grid excluding boundaries
        if dirichlet (bc_type == 1)."""
        if thin_sample:
            x, y = self.sample_space.detector_frame_info[scan]["sub_dimensions"]
            initial_solution = np.zeros((self.sample_space.sub_nx,
                                         self.sample_space.sub_ny),
                                        dtype=complex)
        else:
            x, y = self.sample_space.x, self.sample_space.y
            initial_solution = np.zeros((self.sample_space.nx,
                                         self.sample_space.ny),
                                        dtype=complex)

        if self.bc_type == "dirichlet":
            x_interior = x[1:-1]
            y_interior = y[1:-1]
        else:
            x_interior = x
            y_interior = y

        for i, x in enumerate(x_interior):
            for j, y in enumerate(y_interior):
                initial_solution[i, j] = self.get_initial_condition_value(x,
                                                                          y,
                                                                          scan)
        return initial_solution

    def _define_incidence(self):
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
            incidence (function): A function that takes (x, y, c_x, c_y) and 
            returns the initial condition value.

        Raises:
            ValueError: If `self.ic_type` is not one of the valid options.
        """
        if self.ic_type == "constant":
            def incidence(x, y, c_x, c_y):
                return 1
        elif self.ic_type == "gaussian":
            def incidence(x, y, c_x, c_y):
                sd = 0.05
                dx = (x - c_x) / sd
                dy = (y - c_y) / sd
                norm = 1 / (2 * np.pi * sd ** 2)
                return norm * np.exp(-0.5 * (dx ** 2 + dy ** 2))
        elif self.ic_type == "sinusoidal":
            def incidence(x, y, c_x, c_y):
                return np.sin(np.pi * x) * np.sin(np.pi * y)
        elif self.ic_type == "complex_exp":
            def incidence(x, y, c_x, c_y):
                return np.exp(1j * np.pi * (x + y))
        elif self.ic_type == "dirichlet_test":
            def incidence(x, y, c_x, c_y):
                return (u0_nm_dirichlet(1, 1)(x, y) +
                        0.5 * u0_nm_dirichlet(5, 5)(x, y) +
                        0.2 * u0_nm_dirichlet(9, 9)(x, y))
        elif self.ic_type == "neumann_test":
            def incidence(x, y, c_x, c_y):
                return (u0_nm_neumann(1, 1)(x, y) +
                        0.5 * u0_nm_neumann(2, 2)(x, y) +
                        0.2 * u0_nm_neumann(5, 5)(x, y))
        elif self.ic_type == "airy_disk":
            def incidence(x, y, c_x, c_y):
                r = np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2)
                kr = self.k * r
                # Avoid division by zero at r=0
                airy = np.ones_like(r, dtype=float)
                mask = r != 0
                airy[mask] = ((2 * j1(kr[mask])) / (kr[mask])) ** 2
                return airy
        elif self.ic_type == "disk":
            def incidence(x, y, c_x, c_y):
                r = np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2)
                return np.where(r <= self.probe_diameter / 2, 1, 0)
        else:
            raise ValueError("ic should be 1, 2, 3, 4, 5, 6, 7 or 8")
        return incidence
