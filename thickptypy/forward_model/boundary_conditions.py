import numpy as np
import scipy.sparse as sp


def u_nm(a, n, m):
    """Returns the exact solution for a given n and m."""
    return lambda x, y, z: np.exp(-a * ((n**2) + (m**2)) * (np.pi**2)
                                  * z) * np.cos(n * np.pi * x) * np.cos(m * np.pi * y)


def u_exact_neuman(a):
    return lambda x, y, z: u_nm(a, 1, 1)(
        x, y, z) + 0.5 * u_nm(a, 2, 2)(x, y, z) + 0.2 * u_nm(a, 5, 5)(x, y, z)


class BoundaryConditions:
    """
    Handles boundary conditions and sets up the system matrices.
    """

    def __init__(self, sample_space, initial_condition, thin_sample=False,
                 scan_index=0):
        self.sample_space = sample_space
        self.initial_condition = initial_condition
        self.bc_type = sample_space.bc_type
        self.ic_type = initial_condition.ic_type
        self.dz = sample_space.dz
        self.k = sample_space.k
        self.a = 1j / (2 * self.k)
        self.scan_index = scan_index
        self.thin_sample = thin_sample

        # If thin_sample use sub_dimensions
        if thin_sample:
            self.x = sample_space.detector_frame_info[scan_index]['sub_dimensions'][0]
            self.nx = sample_space.sub_nx
            if self.sample_space.dimension == 2:
                self.ny = self.sample_space.sub_ny
        else:
            self.x = sample_space.x
            self.nx = sample_space.nx
            if self.sample_space.dimension == 2:
                self.ny = self.sample_space.ny
        
        if sample_space.dimension == 2:
            if thin_sample:
                self.y = sample_space.detector_frame_info[scan_index]['sub_dimensions'][1]
                self.ny = sample_space.sub_ny
                self.dy = self.y[1] - self.y[0]
            else:
                self.y = sample_space.y
                self.ny = sample_space.ny
                self.dy = sample_space.dy
        
        self.dx = sample_space.dx

        r_x = (0.5 * sample_space.dz / self.dx**2)
        self.mu_x = self.a * r_x

        if hasattr(sample_space, 'y'):
            r_y = (0.5 * sample_space.dz / self.dy**2)
            self.mu_y = self.a * r_y
        else:
            self.mu_y = 0

        if sample_space.bc_type == "impedance":  # Impedance
            self.beta_x = 2 * self.mu_x * self.dx * 1j * self.k

            if hasattr(sample_space, 'y'):
                self.beta_y = 2 * self.mu_y * self.dy * 1j * self.k
            # else:
            #     self.beta_x *= 2 # TODO: Check this

    def get_matrices_1d_system(self):
        """Set up matrices based on the boundary condition type."""
        Kx = self._create_1D_dirichlet()

        if self.bc_type in ["neumann", "impedance"]:  # Neumann or Impedance
            Kx = self._apply_1D_neumann(Kx)

        if self.bc_type == "impedance":  # Impedance
            Kx = self._apply_1D_impedance(Kx)

        Ix = sp.eye(self.nx)
        Ax = Ix - Kx
        Bx = Ix + Kx

        return Ax.tocsr(), Bx.tocsr()

    def get_initial_boundary_conditions_system(self):
        
        if self.sample_space.dimension == 1:
            return self.get_initial_boundary_conditions_1d_system()
        elif self.sample_space.dimension == 2:
            return self.get_initial_boundary_conditions_2d_system()
        else:
            raise ValueError("Unsupported dimension")
        
    def get_exact_boundary_conditions_system(self, z):
        """Get the exact boundary conditions for the system."""
        if self.sample_space.dimension == 1:
            return self.get_exact_boundary_conditions_1d_test(z)
        elif self.sample_space.dimension == 2:
            return self.get_exact_boundary_conditions_2d_test(z)
        else:
            raise ValueError("Unsupported dimension")
    
    def get_matrix_system(self):
        """Get the system matrices based on the boundary condition type."""
        if self.sample_space.dimension == 1:
            return self.get_matrices_1d_system()
        elif self.sample_space.dimension == 2:
            return self.get_matrices_2d_system()
        else:
            raise ValueError("Unsupported dimension")

    def get_initial_boundary_conditions_1d_system(self):
        """Apply the initial condition to the boundaries."""
        ubc = np.zeros((self.nx), dtype=complex)

        # Neumann Boundary Conditions
        if self.bc_type == "neumann":
            return ubc.flatten()

        # Dirichlet Boundary Conditions
        if self.bc_type == "dirichlet":
            # Left boundary
            ubc[0] += 2 * self.mu_x * \
                self.initial_condition.get_initial_condition_value(self.x[0],
                                                                   self.scan_index)
            # Right boundary
            ubc[-1] += 2 * self.mu_x * \
                self.initial_condition.get_initial_condition_value(self.x[-1],
                                                                   self.scan_index)
            return ubc.flatten()

        # Impedance Boundary Conditions
        elif self.bc_type == "impedance":
            ubc[0] -= 2 * self.beta_x * \
                self.initial_condition.get_initial_condition_value(self.x[0],self.scan_index)
            ubc[-1] -= 2 * self.beta_x * \
                self.initial_condition.get_initial_condition_value(self.x[-1],self.scan_index)
            return ubc.flatten()

    def get_matrices_2d_system(self):
        """Set up matrices based on the boundary condition type."""
        Ax, Bx = self.get_matrices_1d_system()

        # Apply Dirchlet boundary conditions
        Axy, Bxy = self._create_2D_dirichlet(Ax, Bx)

        if self.bc_type == "dirichlet":
            return Axy, Bxy

        Axy, Bxy = self._apply_2D_neumann(Axy, Bxy)

        if self.bc_type == "neumann":
            return Axy, Bxy

        Axy, Bxy = self._apply_2D_impedance(Axy, Bxy)

        if self.bc_type == "impedance":
            return Axy, Bxy
        else:
            raise ValueError(
                "Invalid boundary condition type. Please choose from dirichlet, neumann or dirichlet.")

    def get_initial_boundary_conditions_2d_system(self):
        """Apply the initial condition to the boundaries."""
        ubc = np.zeros((self.nx, self.ny), dtype=complex)

        # Neumann Boundary Conditions
        if self.bc_type == "neumann":
            return ubc.flatten()

        if self.bc_type == "dirichlet":
            x_interior = self.x[1:-1]
            y_interior = self.y[1:-1]
        else:
            x_interior = self.x
            y_interior = self.y

        # Dirichlet Boundary Conditions
        if self.bc_type == "dirichlet":
            ubc[0, :] += 2 * self.mu_x * self.initial_condition.get_initial_condition_value(
                self.x[0], y_interior, self.scan_index)  # Top boundary
            ubc[-1, :] += 2 * self.mu_x * self.initial_condition.get_initial_condition_value(
                self.x[-1], y_interior, self.scan_index)  # Bottom boundary
            ubc[:, 0] += 2 * self.mu_y * self.initial_condition.get_initial_condition_value(
                x_interior, self.y[0], self.scan_index)  # Left boundary
            ubc[:, -1] += 2 * self.mu_y * self.initial_condition.get_initial_condition_value(
                x_interior, self.y[-1],self.scan_index)  # Right boundary
            return ubc.flatten()

        # Impedance Boundary Conditions
        elif self.bc_type == "impedance":
            ubc[0, :] -= 4 * self.mu_x * self.dx * \
                (1j * self.k *
                 self.initial_condition.get_initial_condition_value(self.x[0], y_interior,self.scan_index))
            ubc[-1, :] -= 4 * self.mu_x * self.dx * \
                (1j * self.k *
                 self.initial_condition.get_initial_condition_value(self.x[-1], y_interior,self.scan_index))
            ubc[:, 0] -= 4 * self.mu_y * self.dy * \
                (1j * self.k *
                 self.initial_condition.get_initial_condition_value(x_interior, self.y[0],self.scan_index))
            ubc[:, -1] -= 4 * self.mu_y * self.dy * \
                (1j * self.k *
                 self.initial_condition.get_initial_condition_value(x_interior, self.y[-1], self.scan_index))
            return ubc.flatten()
        else:
            raise ValueError(
                "Invalid boundary condition type. Please choose from 1, 2 or 3.")

    def _create_1D_dirichlet(self):
        """Create 1D Dirichlet boundary condition matrix."""
        e = np.ones(self.nx, dtype=complex)

        return sp.diags([self.mu_x * e, -2 * (self.mu_x + self.mu_y)
                        * e, self.mu_x * e], [-1, 0, 1], shape=(self.nx, self.nx))

    def _apply_1D_neumann(self, K):
        """Modify matrix K for 1D Neumann boundary conditions."""
        K = K.tolil()
        K[0, 1] *= 2
        K[-1, -2] *= 2
        return K.tocsr()

    def _apply_1D_impedance(self, K):
        """Modify matrix K for 1D Impedance boundary conditions."""
        K = K.tolil()
        K[0, 0] += self.beta_x
        K[-1, -1] += self.beta_x
        return K.tocsr()

    def _create_2D_dirichlet(self, Ax, Bx):
        """Apply Dirichlet boundary conditions in 2D."""
        Ix = sp.eye(self.nx)
        Iy = sp.eye(self.ny)
        Axy = sp.kron(
            Iy, Ax) + sp.kron(sp.diags([1, 1], [-1, 1], shape=(self.ny, self.ny)), -self.mu_y * Ix)
        Bxy = sp.kron(
            Iy, Bx) + sp.kron(sp.diags([1, 1], [-1, 1], shape=(self.ny, self.ny)), self.mu_y * Ix)
        return Axy, Bxy

    def _apply_2D_neumann(self, Axy, Bxy):
        """Modify matrices Axy, Bxy for 2D Neumann boundary conditions."""
        Axy = Axy.tolil()
        Bxy = Bxy.tolil()
        Axy[:self.nx, self.nx:self.nx * 2] *= 2
        Axy[-self.nx:, -self.nx * 2:-self.nx] *= 2
        Bxy[:self.nx, self.nx:self.nx * 2] *= 2
        Bxy[-self.nx:, -self.nx * 2:-self.nx] *= 2
        return Axy.tocsr(), Bxy.tocsr()

    def _apply_2D_impedance(self, Axy, Bxy):
        """Modify matrices Axy, Bxy for 2D Impedance boundary conditions."""
        Ix = sp.eye(self.nx)
        Axy = Axy.tolil()
        Bxy = Bxy.tolil()
        Axy[:self.nx, :self.nx] -= self.beta_y * Ix
        Axy[-self.nx:, -self.nx:] -= self.beta_y * Ix
        Bxy[:self.nx, :self.nx] += self.beta_y * Ix
        Bxy[-self.nx:, -self.nx:] += self.beta_y * Ix
        return Axy.tocsr(), Bxy.tocsr()

    def get_exact_boundary_conditions_2d_test(self, z):
        """Apply the initial condition to the boundaries."""
        ubc = np.zeros((self.nx, self.ny), dtype=complex)

        x_interior = self.x
        y_interior = self.y
        # print(f"z: {z}")

        # if z < 0.05:
        #     u_intial = self.get_initial_boundary_conditions_2D_system()
        #     frobenius_norm = np.linalg.norm(u_intial.reshape((self.nx, self.ny)), 'fro')
        #     print(f"Frobenius norm of the initial boundary conditions: {frobenius_norm}")

        # Impedance Boundary Conditions
        ubc[0, :] -= 4 * self.mu_x * self.dx * \
            (1j * self.k * u_exact_neuman(self.a)(self.x[0], y_interior, z))
        ubc[-1, :] -= 4 * self.mu_x * self.dx * \
            (1j * self.k * u_exact_neuman(self.a)(self.x[-1], y_interior, z))
        ubc[:, 0] -= 4 * self.mu_y * self.dy * \
            (1j * self.k * u_exact_neuman(self.a)(x_interior, self.y[0], z))
        ubc[:, -1] -= 4 * self.mu_y * self.dy * \
            (1j * self.k * u_exact_neuman(self.a)(x_interior, self.y[-1], z))
        # frobenius_norm = np.linalg.norm(ubc, 'fro')
        # print(f"Frobenius norm of the boundary conditions: {frobenius_norm}")

        return ubc.flatten()

    def get_exact_boundary_conditions_1d_test(self, z):
        """Apply the initial condition to the boundaries."""
        ubc = np.zeros((self.nx), dtype=complex)

        # Impedance Boundary Conditions
        ubc[0] -= 4 * self.mu_x * self.dx * \
            (1j * self.k * u_exact_neuman(self.a)(self.x[0], 0, z))
        ubc[-1] -= 4 * self.mu_x * self.dx * \
            (1j * self.k * u_exact_neuman(self.a)(self.x[-1], 0, z))

        return ubc.flatten()
