import numpy as np
import scipy.sparse as sp


class BoundaryConditions:
    """
    Handles boundary conditions and sets up the system matrices.
    """

    def __init__(self, simulation_space, probe=None):
        self.simulation_space = simulation_space
        self.probe = probe
        self.bc_type = simulation_space.bc_type
        self.probe_type = simulation_space.probe_type
        self.dz = simulation_space.dz
        self.k = simulation_space.k
        self.a = 1j / (2 * self.k)

        if simulation_space.dimension == 2:
            self.nx, self.ny = simulation_space.effective_dimensions
            self.dy = simulation_space.dy
        else:
            self.nx = simulation_space.effective_dimensions[0]
        self.dx = simulation_space.dx

        r_x = (0.5 * simulation_space.dz / self.dx**2)
        self.mu_x = self.a * r_x

        if simulation_space.dimension == 2:
            r_y = (0.5 * simulation_space.dz / self.dy**2)
            self.mu_y = self.a * r_y
        else:
            self.mu_y = 0

        if simulation_space.bc_type == "impedance":
            self.beta_x = self.mu_x * 2 * 1j * self.k * self.dx
            self.beta_y = self.mu_y * 2 * 1j * self.k * self.dy if simulation_space.dimension == 2 else 0

        elif simulation_space.bc_type == "impedance2":
            # include the 1st-order part too
            self.beta_x = self.mu_x * 2 * 1j * self.k * self.dx
            self.beta_y = self.mu_y * 2 * 1j * self.k * self.dy if simulation_space.dimension == 2 else 0
            # 2nd-order coefficient (tangential curvature)
            self.gamma = - 1 / (2j * self.k)


    def get_initial_boundary_conditions_system(self):
        if self.simulation_space.dimension == 1:
            return self.get_initial_boundary_conditions_1d_system()
        elif self.simulation_space.dimension == 2:
            return self.get_initial_boundary_conditions_2d_system()
        else:
            raise ValueError("Unsupported dimension")

    def get_matrix_system(self):
        """Get the system matrices based on the boundary condition type."""
        if self.simulation_space.dimension == 1:
            return self.get_matrices_1d_system()
        elif self.simulation_space.dimension == 2:
            return self.get_matrices_2d_system()
        else:
            raise ValueError("Unsupported dimension")

    def report_bc_effect(self):
        """Utility: returns (A_row0, A_rowN, B_row0, B_rowN) dense for inspection."""
        Ax, Bx = self.get_matrices_1d_system()
        return Ax.getrow(0).toarray(), Ax.getrow(-1).toarray(), Bx.getrow(0).toarray(), Bx.getrow(-1).toarray()


    def get_initial_boundary_conditions_1d_system(self):
        """RHS contributions (zero for Neumann & Impedance under this formulation)."""
        ubc = np.zeros(self.nx, dtype=complex)

        if self.probe is None:
            return ubc.flatten()
        if self.bc_type == "dirichlet":
            ubc[1]  += 2 * self.mu_x * self.probe[0]
            ubc[-2] += 2 * self.mu_x * self.probe[-1]

        elif self.bc_type == "impedance":
            ubc[0]  -= 2 * self.beta_x * self.probe[0]
            ubc[-1] -= 2 * self.beta_x * self.probe[-1]

        return ubc.flatten()
        
    def get_initial_boundary_conditions_2d_system(self):
        """Apply the initial condition to the boundaries."""
        ubc = np.zeros((self.nx, self.ny), dtype=complex)

        if self.probe is None:
            return ubc.flatten()
        # Dirichlet Boundary Conditions
        if self.bc_type == "dirichlet":
            ubc[1, :] += 2 * self.mu_x * self.probe[0, :]  # Top boundary
            ubc[-2, :] += 2 * self.mu_x * self.probe[-1, :]  # Bottom boundary
            ubc[:, 1] += 2 * self.mu_y * self.probe[:, 0]  # Left boundary
            ubc[:, -2] += 2 * self.mu_y * self.probe[:, -1]  # Right boundary

        # Impedance Boundary Conditions
        elif self.bc_type == "impedance":
            ubc[1, :] -= 2 * self.beta_x * self.probe[0, :]
            ubc[-2, :] -= 2 * self.beta_x * self.probe[-1, :]
            ubc[:, 1] -= 2 * self.beta_y * self.probe[:, 0]
            ubc[:, -2] -= 2 * self.beta_y * self.probe[:, -1]
            return ubc.flatten()

        return ubc.flatten()

    # Create Matrices for 1D Boundary Conditions
    def _create_1D_dirichlet(self):
        e = np.ones(self.nx, dtype=complex)
        K = sp.diags(
            [self.mu_x * e, -2 * (self.mu_x + self.mu_y) * e, self.mu_x * e],
            offsets=[-1, 0, 1], shape=(self.nx, self.nx), dtype=complex
        ).tolil()  # Convert once to LIL for efficient row assignment

        # if self.bc_type == "dirichlet":
        #     K[0, :] = 0
        #     K[0, 0] = 1
        #     K[-1, :] = 0
        #     K[-1, -1] = 1

        return K.tocsr()  # Convert back to CSR for efficient arithmetic


    def _apply_1D_neumann(self, K):
        """Modify matrix K for 1D Neumann boundary conditions."""
        K = K.tolil()
        K[0, 1] *= 2
        K[-1, -2] *= 2
        return K.tocsr()

    def _apply_1D_impedance(self, K):
        """
        Minimal Robin: modify diagonal only (no prior Neumann scaling).
        K holds scaled second-difference; adding beta_x shifts boundary rows.
        """
        K = K.tolil()
        K[0, 0] += self.beta_x
        K[-1, -1] += self.beta_x
        return K.tocsr()
    
    def _apply_1D_impedance2(self, K):
        K = K.tolil()

        gamma = -1 / (2j * self.k)

        K[0, 0] += self.beta_x + gamma / (self.dz**2)
        K[0, 1] += -gamma / (self.dz**2)

        K[-1, -1] += self.beta_x + gamma / (self.dz**2)
        K[-1, -2] += -gamma / (self.dz**2)

        return K.tocsr()




    def get_matrices_1d_system(self):
        Kx = self._create_1D_dirichlet()

        if self.bc_type in ["neumann", "impedance", "impedance2"]:
            Kx = self._apply_1D_neumann(Kx)
        if self.bc_type == "impedance":
            Kx = self._apply_1D_impedance(Kx)
        elif self.bc_type == "impedance2":
            Kx = self._apply_1D_impedance2(Kx)

        Ix = sp.eye(self.nx)
        Ax, Bx = Ix - Kx, Ix + Kx
        return Ax.tocsr(), Bx.tocsr()

    
    # --- 2D builders ----
    def _create_2D_dirichlet(self, Ax, Bx):
        """Apply Dirichlet boundary conditions in 2D."""
        Ix = sp.eye(self.nx)
        Iy = sp.eye(self.ny)
        Axy = sp.kron(
            Iy, Ax) + sp.kron(sp.diags([1, 1], [-1, 1], shape=(self.ny, self.ny)), -self.mu_y * Ix).tolil()
        Bxy = sp.kron(
            Iy, Bx) + sp.kron(sp.diags([1, 1], [-1, 1], shape=(self.ny, self.ny)), self.mu_y * Ix).tolil()
        if self.bc_type == "dirichlet":
            Axy[:self.nx, :self.nx] = Ix
            Axy[-self.nx:, -self.nx:] = Ix
            Bxy[:self.nx, :self.nx] = Ix
            Bxy[-self.nx:, -self.nx:] = Ix
        return Axy.tocsr(), Bxy.tocsr()

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
        # x is the fast index in kron(Iy, Ax)
        Axy = Axy.tolil(); Bxy = Bxy.tolil()

        # --- y-boundaries (first/last block of size nx): add ±beta_y on diag
        Sy = sp.diags(([1] + [0]*(self.ny-2) + [1]), 0, shape=(self.ny, self.ny))
        Ix = sp.eye(self.nx)
        Dy = sp.kron(Sy, Ix)  # picks y=0 and y=ny-1 rows/cols

        Axy -= self.beta_y * Dy
        Bxy += self.beta_y * Dy

        # --- x-boundaries (columns x=0 and x=nx-1 across all y)
        sel_x = np.zeros(self.nx); sel_x[0] = 1; sel_x[-1] = 1
        Sx = sp.diags(sel_x, 0, shape=(self.nx, self.nx))
        Ix = Sx  # reuse name
        DyI = sp.kron(sp.eye(self.ny), Sx)

        Axy -= self.beta_x * DyI
        Bxy += self.beta_x * DyI

        return Axy.tocsr(), Bxy.tocsr()

    
    def _apply_2D_impedance2(self, Axy, Bxy):
        # start from first-order Robin on all boundaries
        Axy, Bxy = self._apply_2D_impedance(Axy, Bxy)

        Axy = Axy.tolil()

        # Tangential Laplacians
        Lx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(self.nx, self.nx)) / (self.dx**2)
        Ly = sp.diags([1, -2, 1], [-1, 0, 1], shape=(self.ny, self.ny)) / (self.dy**2)

        # Selectors for boundaries
        Sy = sp.diags(([1] + [0]*(self.ny-2) + [1]), 0, shape=(self.ny, self.ny))   # y=0, y=ny-1
        sel_x = np.zeros(self.nx); sel_x[0] = 1; sel_x[-1] = 1
        Sx = sp.diags(sel_x, 0, shape=(self.nx, self.nx))                             # x=0, x=nx-1

        # y-boundaries → tangential is x: add gamma * (Sy ⊗ Lx)
        Axy += self.gamma * sp.kron(Sy, Lx)

        # x-boundaries → tangential is y: add gamma * (Ly ⊗ Sx)
        Axy += self.gamma * sp.kron(Ly, Sx)

        return Axy.tocsr(), Bxy.tocsr()


    
    def get_matrices_2d_system(self):
        """Set up matrices based on the boundary condition type."""
        Ax, Bx = self.get_matrices_1d_system()

        # Apply dirichlet boundary conditions
        Axy, Bxy = self._create_2D_dirichlet(Ax, Bx)

        if self.bc_type == "dirichlet":
            return Axy, Bxy

        Axy, Bxy = self._apply_2D_neumann(Axy, Bxy)

        if self.bc_type == "neumann":
            return Axy, Bxy

        Axy, Bxy = self._apply_2D_impedance(Axy, Bxy)

        if self.bc_type == "impedance":
            return Axy, Bxy
        
        elif self.bc_type == "impedance2":
            return self._apply_2D_impedance2(Axy, Bxy)
        else:
            raise ValueError(
                "Invalid boundary condition type. Please choose from dirichlet, neumann or dirichlet.")
    
 