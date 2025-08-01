import numpy as np
import scipy.sparse as sp

from .boundary_conditions import BoundaryConditions

from typing import Optional


class LinearSystemSetup:
    """
    Base class for linear system setup, providing methods to create
    forward models and handle boundary conditions.
    """

    def __init__(self, sample_space, initial_condition, thin_sample,
                 full_system):
        self.sample_space = sample_space
        self.initial_condition = initial_condition
        self.full_system = full_system
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

        self._probe_angle = 0.0

        if not self.thin_sample and self.full_system:
             self.A_slice, self.B_slice, self.b_slice = self.create_system_slice()

        # Preallocate probes for speed
        self.b0 = None
        if self.sample_space.dimension == 1:
            self.probes = self.initial_condition.return_probes()
            if not self.thin_sample and self.full_system:
                # Batch multiply B_slice @ probe.T -> (block_size, num_probes)
                b0_mat = self.B_slice @ self.probes.T
                # Transpose and add b_slice
                self.b0 = b0_mat.T + self.b_slice

    @property
    def probe_angle(self):
        """Get the probe angle in radians."""
        return self._probe_angle

    @probe_angle.setter
    def probe_angle(self, value):
        """Set the probe angle in radians."""
        self._probe_angle = value

    def setup_homogeneous_forward_model_lhs(self, scan_index: Optional[int] = 0):
        """Create Free-Space Forward Model Left-Hand Side (LHS) Matrix."""
        if self.thin_sample or not self.full_system:
            self.A_slice, self.B_slice, _ = self.create_system_slice(scan_index=scan_index)

        # Homogeneous forward model
        A_homogeneous = (
            sp.kron(sp.eye(self.sample_space.nz - 1), self.A_slice)
            - sp.kron(
                sp.diags([1], [-1], shape=(self.sample_space.nz - 1,
                                           self.sample_space.nz - 1)),
                self.B_slice
            )
        )
        return A_homogeneous

    def setup_homogeneous_forward_model_rhs(self, scan_index: Optional[int] = 0):
        """Create Free-Space Forward Model Right-Hand Side (RHS) Vector."""
        if self.thin_sample or not self.full_system:
            _, _, self.b_slice = self.create_system_slice(scan_index=scan_index)
        # Homogeneous forward model
        b_homogeneous = np.tile(self.b_slice, self.sample_space.nz - 1)
        return b_homogeneous
    
    def test_exact_impedance_forward_model_rhs(self):
        """Test Impedance Boundary Conditions in Free-Space Forward Model
        Right-Hand Side (RHS) Vector."""
        bcs = BoundaryConditions(
            self.sample_space, self.initial_condition,
            thin_sample=self.thin_sample
        )
        z = self.sample_space.z
        nz = self.sample_space.nz

        # Compute the average boundary condition between adjacent z slices
        for j in range(1, nz):
            b0 = bcs.get_exact_boundary_conditions_system(z[j - 1])
            b1 = bcs.get_exact_boundary_conditions_system(z[j])
            b_avg = 0.5 * (b0 + b1)
            if j == 1:
                exact_b_homogeneous = b_avg
            else:
                exact_b_homogeneous = np.concatenate((exact_b_homogeneous,
                                                      b_avg))

        return exact_b_homogeneous

    def test_exact_impedance_rhs_slice(self, j: int):
        """Test if the exact impedance boundary conditions are satisfied."""
        bcs = BoundaryConditions(self.sample_space, self.initial_condition,
                                 thin_sample=self.thin_sample)
        b_old = bcs.get_exact_boundary_conditions_system(
                    self.sample_space.z[j-1])
        b_new = bcs.get_exact_boundary_conditions_system(
            self.sample_space.z[j])
        return (b_old + b_new) / 2

    # Foward Model Contribution from Inhomogeneous Field
    def setup_inhomogeneous_forward_model(
            self, n=None, grad=False, scan_index: Optional[int] = 0):
        """
        Compute the inhomogeneous matrix.

        Parameters:
        n (ndarray, optional): Refractive index distribution.
        Defaults to None. grad (bool, optional): If True, compute the gradient
        of the object. Defaults to False.
        """
        # Create the inhomogeneous forward matrix contribution
        # For thin samples, use sub-sampling
        object_slices = self.sample_space.create_sample_slices(
            self.thin_sample, n=n, grad=grad, scan_index=scan_index)

        # Create the diagonal matrix for the inhomogeneous term
        return sp.diags(object_slices.T.flatten(
        )) + sp.diags([object_slices[..., 1:].T.flatten()], [-self.block_size])

    def probe_contribution(self, scan_index: Optional[int] = 0,
                           probes: Optional[np.ndarray] = None):
        """Compute the probe contribution to the forward model."""
        if self.thin_sample:
            _, self.B_slice, self.b_slice = self.create_system_slice(scan_index=scan_index)

        # Apply initial condition if provided
        if probes is not None:
            probe = probes[scan_index, :]
        elif self.sample_space.dimension > 1:
            probe = self.initial_condition.apply_initial_condition(
                scan_index)
        else:
            probe = self.probes[scan_index, :]

        # Apply probe angle (linear phase shift)
        if self._probe_angle != 0.0:
            n = np.arange(len(probe))
            linear_phase = np.exp(1j * self._probe_angle * n)
            probe *= linear_phase


        if self.sample_space.dimension == 1 and not self.thin_sample and self.full_system and probes is None:
            b0 = self.b0[scan_index, :]
        else:
            b0 = self.B_slice @ probe.flatten() + self.b_slice

        probe_contribution = np.concatenate(
            (b0, np.zeros(self.block_size * (self.sample_space.nz - 2))))

        return probe_contribution, probe

    def create_system_slice(self, scan_index: Optional[int] = 0):
        """Create the linear system based on boundary conditions."""
        bcs = BoundaryConditions(self.sample_space, self.initial_condition,
                                 thin_sample=self.thin_sample,
                                 scan_index=scan_index)
        # 2D system
        A_slice, B_slice = bcs.get_matrix_system()
        b_slice = bcs.get_initial_boundary_conditions_system()

        return A_slice, B_slice, b_slice
