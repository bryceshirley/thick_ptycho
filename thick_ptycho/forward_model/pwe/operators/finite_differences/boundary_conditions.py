import scipy.sparse as sp


class BoundaryConditions:
    """
    Handles boundary conditions.
    """

    def __init__(self, simulation_space, bc_type="impedance"):
        self.simulation_space = simulation_space
        self.bc_type = bc_type
        self.probe_type = simulation_space.probe_type
        self.dz = simulation_space.dz
        self.k = simulation_space.k
        self.a = 1j / (2 * self.k)

        if self.simulation_space.dimension == 2:
            self.nx = simulation_space.effective_shape[0]
        else:
            self.dy = simulation_space.dy
            self.nx, self.ny = (
                simulation_space.effective_shape[0],
                simulation_space.effective_shape[1],
            )
        self.block_size = self.simulation_space.block_size

        self.dx = simulation_space.dx

        r_x = 0.5 * simulation_space.dz / self.dx**2
        self.mu_x = self.a * r_x

        if simulation_space.dimension == 3:
            r_y = 0.5 * simulation_space.dz / self.dy**2
            self.mu_y = self.a * r_y
        else:
            self.mu_y = 0

        if self.bc_type == "impedance":
            self.beta_x = self.mu_x * 2 * 1j * self.k * self.dx
            self.beta_y = (
                self.mu_y * 2 * 1j * self.k * self.dy
                if simulation_space.dimension == 3
                else 0
            )

        elif self.bc_type == "impedance2":
            # include the 1st-order part too
            self.beta_x = self.mu_x * 2 * 1j * self.k * self.dx
            self.beta_y = (
                self.mu_y * 2 * 1j * self.k * self.dy
                if simulation_space.dimension == 3
                else 0
            )
            # 2nd-order coefficient (tangential curvature)
            self.gamma = -1 / (2j * self.k)

    def _apply_1D_dirichlet(self, K):
        """Modify matrix K for 1D Dirichlet boundary conditions."""
        K = K.tolil()
        # Zero out off-diagonal entries at boundaries
        K[0, 1] = 0
        K[-1, -2] = 0
        # Set diagonal entries at boundaries to 0
        K[0, 0] = 0
        K[-1, -1] = 0
        return K.tocsr()

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

        K[0, 0] += self.beta_x + self.gamma / (self.dz**2)
        K[0, 1] += -self.gamma / (self.dz**2)

        K[-1, -1] += self.beta_x + self.gamma / (self.dz**2)
        K[-1, -2] += -self.gamma / (self.dz**2)

        return K.tocsr()

    def _apply_2D_dirichlet(self, Axy, Bxy):
        """Modify matrices Axy, Bxy for 2D Dirichlet boundary conditions."""
        Ix = sp.eye(self.nx)
        Axy[: self.nx, : self.nx] = Ix
        Axy[-self.nx :, -self.nx :] = Ix
        Bxy[: self.nx, : self.nx] = Ix
        Bxy[-self.nx :, -self.nx :] = Ix

        # Zero out coupling terms at boundaries
        Axy[: self.nx, self.nx : self.nx * 2] *= 0
        Axy[-self.nx :, -self.nx * 2 : -self.nx] *= 0
        Bxy[: self.nx, self.nx : self.nx * 2] *= 0
        Bxy[-self.nx :, -self.nx * 2 : -self.nx] *= 0
        return Axy.tocsr(), Bxy.tocsr()

    def _apply_2D_neumann(self, Axy, Bxy):
        """Modify matrices Axy, Bxy for 2D Neumann boundary conditions."""
        Axy = Axy.tolil()
        Bxy = Bxy.tolil()
        Axy[: self.nx, self.nx : self.nx * 2] *= 2
        Axy[-self.nx :, -self.nx * 2 : -self.nx] *= 2
        Bxy[: self.nx, self.nx : self.nx * 2] *= 2
        Bxy[-self.nx :, -self.nx * 2 : -self.nx] *= 2
        return Axy.tocsr(), Bxy.tocsr()

    def _apply_2D_impedance(self, Axy, Bxy):
        # x is the fast index in kron(Iy, Ax)
        Ix = sp.eye(self.nx, dtype=complex)
        Axy = Axy.tolil()
        Bxy = Bxy.tolil()
        Axy[: self.nx, : self.nx] -= self.beta_y * Ix
        Axy[-self.nx :, -self.nx :] -= self.beta_y * Ix
        Bxy[: self.nx, : self.nx] += self.beta_y * Ix
        Bxy[-self.nx :, -self.nx :] += self.beta_y * Ix
        return Axy.tocsr(), Bxy.tocsr()
