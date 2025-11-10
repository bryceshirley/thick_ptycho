import numpy as np
from thick_ptycho.forward_model.pwe.operators.finite_differences.boundary_conditions import BoundaryConditions


class BoundaryConditionsTest(BoundaryConditions):
    """
    Test-only class providing exact analytic Neumann / impedance boundary
    conditions for validation of finite difference solvers.

    Implements exact boundary conditions and right-hand sides based on
    the analytic solutions described in test formulations of the heat /
    diffusion equation with Neumann-type boundary conditions.
    """

    def __init__(self, simulation_space):
        super().__init__(simulation_space, bc_type="impedance")
        self.x = np.linspace(
            simulation_space.xlims[0],
            simulation_space.xlims[1],
            simulation_space.effective_nx,
        )
        if simulation_space.dimension == 2:
            self.y = np.linspace(
                simulation_space.ylims[0],
                simulation_space.ylims[1],
                simulation_space.effective_ny,
            )

    # ------------------------------------------------------------------
    # Analytic reference solutions (Neumann exact forms)
    # ------------------------------------------------------------------
    @staticmethod
    def u_nm(a, n, m):
        """Return analytic 2D mode solution u_{n,m}(x,y,z)."""
        return lambda x, y, z: np.exp(-a * ((n ** 2 + m ** 2) * np.pi ** 2) * z) * np.cos(n * np.pi * x) * np.cos(m * np.pi * y)

    def u_exact_neuman_1d(self):
        """Exact analytic 1D solution (sum of modes)."""
        return lambda x, z: (
            self.u_nm(self.a, 1, 0)(x, 0, z)
            + 0.5 * self.u_nm(self.a, 2, 0)(x, 0, z)
            + 0.2 * self.u_nm(self.a, 5, 0)(x, 0, z)
        )

    def u_exact_neuman_2d(self):
        """Exact analytic 2D solution (sum of modes)."""
        return lambda x, y, z: (
            self.u_nm(self.a, 1, 1)(x, y, z)
            + 0.5 * self.u_nm(self.a, 2, 2)(x, y, z)
            + 0.2 * self.u_nm(self.a, 5, 5)(x, y, z)
        )

    # ------------------------------------------------------------------
    # Boundary condition generators
    # ------------------------------------------------------------------
    def get_exact_boundary_conditions_system(self, z):
        """Dispatch boundary condition generator by simulation dimension."""
        if self.simulation_space.dimension == 1:
            return self.get_exact_boundary_conditions_1d_test(z)
        elif self.simulation_space.dimension == 2:
            return self.get_exact_boundary_conditions_2d_test(z)
        else:
            raise ValueError("Unsupported simulation dimension for BC test")

    def get_exact_boundary_conditions_1d_test(self, z):
        """Compute impedance BCs in 1D using analytic Neumann solution."""
        ubc = np.zeros(self.nx, dtype=complex)
        ue = self.u_exact_neuman_1d()
        ubc[0] -= 2 * self.beta_x * ue(self.x[0], z)
        ubc[-1] -= 2 * self.beta_x * ue(self.x[-1], z)
        return ubc.flatten()

    def get_exact_boundary_conditions_2d_test(self, z):
        """Compute impedance BCs in 2D using analytic Neumann solution."""
        ubc = np.zeros((self.nx, self.ny), dtype=complex)
        ue = self.u_exact_neuman_2d()

        # Impedance-type Neumann corrections at boundaries
        ubc[0, :] -= 2 * self.beta_x * ue(self.x[0], self.y, z)
        ubc[-1, :] -= 2 * self.beta_x * ue(self.x[-1], self.y, z)
        ubc[:, 0] -= 2 * self.beta_y * ue(self.x, self.y[0], z)
        ubc[:, -1] -= 2 * self.beta_y * ue(self.x, self.y[-1], z)
        return ubc.flatten()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def test_exact_impedance_forward_model_rhs(self):
        """
        Build RHS using the average of exact boundary conditions between steps.
        Useful for validating impedance BC implementation.
        """
        z = self.simulation_space.z
        nz = self.simulation_space.nz
        out = []
        for j in range(1, nz):
            b0 = self.get_exact_boundary_conditions_system(z[j - 1])
            b1 = self.get_exact_boundary_conditions_system(z[j])
            out.append(0.5 * (b0 + b1))
        return np.concatenate(out, axis=0)

    def test_exact_impedance_rhs_step(self, j: int):
        """Average exact boundary conditions across a single slab interface j-1→j."""
        z = self.simulation_space.z
        b_old = self.get_exact_boundary_conditions_system(z[j - 1])
        b_new = self.get_exact_boundary_conditions_system(z[j])
        return 0.5 * (b_old + b_new)
