import numpy as np
import scipy.sparse as sp

from .boundary_conditions import BoundaryConditions

class OperatorMatrices(BoundaryConditions):
    """
    Handles boundary conditions and sets up the system matrices.
    """

    def __init__(self, simulation_space, bc_type="impedance"):
        super().__init__(simulation_space, bc_type=bc_type)
        self._probe = None

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

    # --- 2D builders ----
    def _create_1D_laplacian(self):
        e = np.ones(self.nx, dtype=complex)
        K = sp.diags(
            [self.mu_x * e, -2 * (self.mu_x + self.mu_y) * e, self.mu_x * e],
            offsets=[-1, 0, 1], shape=(self.nx, self.nx), dtype=complex
        ).tolil()  # Convert once to LIL for efficient row assignment

        return K.tocsr()  # Convert back to CSR for efficient arithmetic

    def get_matrices_1d_system(self):
        Kx = self._create_1D_laplacian()

        Kx = self._apply_boundary_conditions_1D(Kx)

        Ix = sp.eye(self.nx)
        Ax, Bx = Ix - Kx, Ix + Kx
        return Ax.tocsr(), Bx.tocsr()

    def _apply_boundary_conditions_1D(self, Kx):
        if self.bc_type == "dirichlet":
            Kx = self._apply_1D_dirichlet(Kx)
        if self.bc_type in ["neumann", "impedance", "impedance2"]:
            Kx = self._apply_1D_neumann(Kx)
        if self.bc_type == "impedance":
            Kx = self._apply_1D_impedance(Kx)
        elif self.bc_type == "impedance2":
            Kx = self._apply_1D_impedance2(Kx)

    
    # --- 2D builders ----
    def _create_2D_laplacian(self, Ax, Bx):
        """
        Create the 2D Laplacian operator.
        A is n+1 z step
        B is n z step
        Parameters:
         Ax, Bx: 1D system matrices with bcs applied.
        Returns:
         Axy, Bxy: 2D system matrices before BCs are applied.
        """
        Ix = sp.eye(self.nx)
        Iy = sp.eye(self.ny)
        Axy = sp.kron(
            Iy, Ax) + sp.kron(sp.diags([1, 1], [-1, 1], shape=(self.ny, self.ny)), -self.mu_y * Ix).tolil()
        Bxy = sp.kron(
            Iy, Bx) + sp.kron(sp.diags([1, 1], [-1, 1], shape=(self.ny, self.ny)), self.mu_y * Ix).tolil()
        return Axy.tocsr(), Bxy.tocsr()
    
    def get_matrices_2d_system(self):
        """Set up matrices based on the boundary condition type."""
        # Get 1D system matrices first with BCs applied
        Ax, Bx = self.get_matrices_1d_system()

        # Create 2D Laplacian
        Axy, Bxy = self._create_2D_laplacian(Ax, Bx)
        
        # Apply 2D boundary conditions
        if self.bc_type == "dirichlet":
            return self._apply_2D_dirichlet(Axy, Bxy)

        Axy, Bxy = self._apply_2D_neumann(Axy, Bxy)

        if self.bc_type == "neumann":
            return Axy, Bxy

        if self.bc_type == "impedance":
            return self._apply_2D_impedance(Axy, Bxy)
        
        elif self.bc_type == "impedance2":
            return self._apply_2D_impedance2(Axy, Bxy)
        else:
            raise ValueError(
                "Invalid boundary condition type. Please choose from dirichlet, neumann or dirichlet.")

    # ---- Non-Zero Boundary Probe Handling ----
    @property
    def probe(self):
        """Get the probe field."""
        return self._probe

    @probe.setter
    def probe(self, probe: np.ndarray):
        """Set the probe field for boundary condition calculations."""
        self._probe = probe

    def reset_probe(self):
        """Reset the probe field."""
        self._probe = None

    def get_probe_boundary_conditions_system(self):
        """Get the probe/initial RHS contributions based on the dimension."""
        if self.simulation_space.dimension == 1:
            return self.get_probe_boundary_conditions_1d_system()
        elif self.simulation_space.dimension == 2:
            return self.get_probe_boundary_conditions_2d_system()
        else:
            raise ValueError("Unsupported dimension")

    def get_probe_boundary_conditions_1d_system(self):
        """Get probe/initial RHS contributions (zero for Neumann & Impedance under this formulation)."""
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
        
    def get_probe_boundary_conditions_2d_system(self):
        """Apply the probe/initial condition to the boundaries."""
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
    
