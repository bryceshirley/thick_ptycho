# def u_nm(a, n, m):
#     """Returns the exact solution for a given n and m."""
#     return lambda x, y, z: np.exp(-a * ((n**2) + (m**2)) * (np.pi**2)
#                                   * z) * np.cos(n * np.pi * x) * np.cos(m * np.pi * y)

# def u_exact_neuman_1d(a):
#     return lambda x, z: u_nm(a, 1, 0)(
#         x, 0, z) + 0.5 * u_nm(a, 2, 0)(x, 0, z) + 0.2 * u_nm(a, 5, 0)(x, 0, z)


# def u_exact_neuman_2d(a):
#     return lambda x, y, z: u_nm(a, 1, 1)(
#         x, y, z) + 0.5 * u_nm(a, 2, 2)(x, y, z) + 0.2 * u_nm(a, 5, 5)(x, y, z)

        # def get_exact_boundary_conditions_system(self, z):
    #     """Get the exact boundary conditions for the system."""
    #     if self.simulation_space.dimension == 1:
    #         return self.get_exact_boundary_conditions_1d_test(z)
    #     elif self.simulation_space.dimension == 2:
    #         return self.get_exact_boundary_conditions_2d_test(z)
    #     else:
    #         raise ValueError("Unsupported dimension")


# Test Methods for Impedance Boundary Conditions

    # def test_exact_impedance_forward_model_rhs(self,probe):
    #     """
    #     Build RHS using the average of exact boundary conditions between slices.

    #     Useful for validating impedance BC implementation.
    #     """
    #     bcs = BoundaryConditions(self.simulation_space, probe)
    #     z = self.simulation_space.z
    #     nz = self.simulation_space.nz
    #     out = []
    #     for j in range(1, nz):
    #         b0 = bcs.get_exact_boundary_conditions_system(z[j - 1])
    #         b1 = bcs.get_exact_boundary_conditions_system(z[j])
    #         out.append(0.5 * (b0 + b1))
    #     return np.concatenate(out, axis=0)

    # def test_exact_impedance_rhs_slice(self, j: int, probe: np.ndarray):
    #     """
    #     Average exact boundary conditions across a single slab interface j-1→j.
    #     """
    #     bcs = BoundaryConditions(self.simulation_space, probe)
    #     b_old = bcs.get_exact_boundary_conditions_system(self.simulation_space.z[j - 1])
    #     b_new = bcs.get_exact_boundary_conditions_system(self.simulation_space.z[j])
    #     return (b_old + b_new) / 2

    # def get_exact_boundary_conditions_2d_test(self, z):
    #     """Apply the initial condition to the boundaries."""
    #     ubc = np.zeros((self.nx, self.ny), dtype=complex)

    #     # Impedance Boundary Conditions
    #     ue = u_exact_neuman_2d(self.a)
    #     ubc[0, :] -= 2 * self.beta_x * ue(self.x[0], self.y, z)
    #     ubc[-1, :] -= 2 * self.beta_x * ue(self.x[-1], self.y, z)
    #     ubc[:, 0] -= 2 * self.beta_y * ue(self.x, self.y[0], z)
    #     ubc[:, -1] -= 2 * self.beta_y * ue(self.x, self.y[-1], z)

    #     return ubc.flatten()

    # def get_exact_boundary_conditions_1d_test(self, z):
    #     """Apply the initial condition to the boundaries."""
    #     ubc = np.zeros((self.nx), dtype=complex)

    #     # Impedance Boundary Conditions
    #     ue = u_exact_neuman_1d(self.a)
    #     ubc[0] -= 2 * self.beta_x * ue(self.x[0], z)
    #     ubc[-1] -= 2 * self.beta_x * ue(self.x[-1], z)

    #     return ubc.flatten()
 