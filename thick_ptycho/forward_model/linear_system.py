import numpy as np
import scipy.sparse as sp

from scipy.special import j1
from astropy.convolution import AiryDisk2DKernel


def u0_nm_neumann(n, m):
    """Returns the exact solution for a given n and m."""
    return lambda x, y: np.cos(n * np.pi * x) * np.cos(m * np.pi * y)


def u0_nm_dirichlet(n, m):
    """Returns the exact solution for a given n and m."""
    return lambda x, y: np.sin(n * np.pi * x) * np.sin(m * np.pi * y)

from .boundary_conditions import BoundaryConditions


from typing import Optional, Tuple, Union


class LinearSystemSetup:
    """
    Base class for linear system setup, providing methods to create
    forward models and handle boundary conditions.
    """

    def __init__(self, sample_space, thin_sample,
                 full_system,
                 probe_angles_list=[0.0]):
        self.sample_space = sample_space
        self.full_system = full_system
        self.thin_sample = thin_sample
        self.signal_strength = 1.0  # Default signal strength

        # For thin samples, use sub-sampling
        if thin_sample:
            self.nx = self.sample_space.sub_nx
            if self.sample_space.dimension == 2:
                self.ny = self.sample_space.sub_ny
        else:
            self.nx = self.sample_space.nx
            if self.sample_space.dimension == 2:
                self.ny = self.sample_space.ny

        if not self.thin_sample and self.full_system:
             self.A_slice, self.B_slice, self.b_slice = self.create_system_slice()

            
        num_angles = len(probe_angles_list)

        if self.sample_space.dimension == 1:
            self.block_size = self.nx
            self.probes = np.zeros((num_angles, self.sample_space.num_probes, self.nx), dtype=complex)
        else:
            self.block_size = self.nx * self.ny
            self.probes = np.zeros((num_angles, self.sample_space.num_probes, self.nx, self.ny), dtype=complex)

        # Precompute probes for all scans
        #print(probe_angles_list)
        for angle_index in range(num_angles):
            for scan_index in range(self.sample_space.num_probes):
                self.probes[angle_index,scan_index, ...] = self.apply_initial_condition(scan_index,
                                                                                        probe_angle=probe_angles_list[angle_index])
        flat_shape = (num_angles * self.sample_space.num_probes,) + self.probes.shape[2:]
        self.probes = self.probes.reshape(flat_shape)

        # Preallocate Compute b0 for speed
        self.b0 = None
        if self.sample_space.dimension == 1:
            if not self.thin_sample and self.full_system:
                # Batch multiply B_slice @ probe.T -> (block_size, num_probes)
                b0_mat = self.B_slice @ self.probes.T
                # Transpose and add b_slice
                self.b0 = b0_mat.T + self.b_slice


    def setup_homogeneous_forward_model_lhs(self, scan_index: Optional[int] = 0,
                                            angle_index: Optional[int] = 0):
        """Create Free-Space Forward Model Left-Hand Side (LHS) Matrix."""
        if self.thin_sample or not self.full_system:
            self.A_slice, self.B_slice, _ = self.create_system_slice(scan_index=scan_index,angle_index=angle_index)

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

    def setup_homogeneous_forward_model_rhs(self, scan_index: Optional[int] = 0,
                                            angle_index: Optional[int] = 0):

        """Create Free-Space Forward Model Right-Hand Side (RHS) Vector."""
        if self.thin_sample or not self.full_system:
            _, _, self.b_slice = self.create_system_slice(scan_index=scan_index, angle_index=angle_index)
        # Homogeneous forward model
        b_homogeneous = np.tile(self.b_slice, self.sample_space.nz - 1)
        return b_homogeneous
    
    def test_exact_impedance_forward_model_rhs(self):
        """Test Impedance Boundary Conditions in Free-Space Forward Model
        Right-Hand Side (RHS) Vector."""
        bcs = BoundaryConditions(
            self.sample_space, self.probes[0, ...],
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
        bcs = BoundaryConditions(self.sample_space, self.probes[0, ...],
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

    def probe_contribution(self, scan_index: Optional[int] = 0, angle_index: Optional[int] = 0,
                           probes: Optional[np.ndarray] = None):
        """Compute the probe contribution to the forward model."""
        probe_index = angle_index*self.sample_space.num_probes + scan_index
        if self.thin_sample:
            _, self.B_slice, self.b_slice = self.create_system_slice(scan_index=scan_index)

        # Apply initial condition if provided
        if probes is not None:
            probe = probes[probe_index, ...].flatten()
        else:
            probe = self.probes[probe_index, ...].flatten()


        if self.sample_space.dimension == 1 and not self.thin_sample and self.full_system and probes is None:
            b0 = self.b0[scan_index, :]
        else:
            b0 = self.B_slice @ probe.flatten() + self.b_slice

        probe_contribution = np.concatenate(
            (b0, np.zeros(self.block_size * (self.sample_space.nz - 2))))

        return probe_contribution, probe

    def create_system_slice(self, scan_index: Optional[int] = 0, angle_index: Optional[int] = 0):
        """Always rebuild (per-scan) matrices; no caching to avoid stale BC data."""
        probe_index = angle_index*self.sample_space.num_probes + scan_index
        bcs = BoundaryConditions(self.sample_space,
                                 self.probes[probe_index, ...],
                                 thin_sample=self.thin_sample,
                                 scan_index=scan_index)
        A_slice, B_slice = bcs.get_matrix_system()
        b_slice = bcs.get_initial_boundary_conditions_system()
        return A_slice, B_slice, b_slice
    
    def apply_initial_condition(self, scan, probe_type=None,probe_focus=None,
                            probe_angle=None):
        """
        Apply initial condition to the solution grid, excluding boundaries if dirichlet.
        Returns a matrix (1D or 2D) of initial condition values using meshgrid.
        """
        if probe_type is None:
            probe_type = self.sample_space.probe_type
        if probe_focus is None:
            probe_focus = self.sample_space.probe_focus
        if probe_angle is None:
            probe_angle_val = self.sample_space.probe_angle
        else:
            probe_angle_val = probe_angle
        
        radius = self.sample_space.probe_diameter_continuous / 2
        radius_discrete = self.sample_space.probe_diameter / 2

        # Get grid points
        if self.sample_space.dimension == 1:
            if self.thin_sample:
                x = self.sample_space.detector_frame_info[scan]["sub_dimensions"][0]
            else:
                x = self.sample_space.x

            if self.sample_space.bc_type == "dirichlet":
                x_interior = x[1:-1]
            else:
                x_interior = x

            x_mesh = x_interior

            if isinstance(probe_angle_val, (int, float)):
                angle_x = float(probe_angle_val)
            else:
                raise ValueError("In 1D, probe_angle must be a single float.")
            angle_y = 0.0  # unused in 1D
        elif self.sample_space.dimension == 2:
            if self.thin_sample:
                x, y = self.sample_space.detector_frame_info[scan]["sub_dimensions"]
            else:
                x, y = self.sample_space.x, self.sample_space.y

            if self.sample_space.bc_type == "dirichlet":
                x_interior = x[1:-1]
                y_interior = y[1:-1]
            else:
                x_interior = x
                y_interior = y

            x_mesh, y_mesh = np.meshgrid(x_interior, y_interior, indexing='ij')

            if isinstance(probe_angle_val, (tuple, list)) and len(probe_angle_val) == 2:
                angle_x, angle_y = probe_angle_val
            elif isinstance(probe_angle_val, (int, float)):
                # allow a float -> only x tilt
                angle_x, angle_y = float(probe_angle_val), 0.0
            else:
                raise ValueError("In 2D, probe_angle must be a float or (angle_x, angle_y).")
        else:
            raise ValueError("Unsupported dimension: {}".format(self.sample_space.dimension))

        # Allocate
        if self.sample_space.dimension == 1:
            initial_shape = x_mesh.shape
        else:
            initial_shape = x_mesh.shape
        initial_solution = np.zeros(initial_shape, dtype=complex)

        # Get probe centre
        if self.sample_space.dimension == 1:
            c_x = self.sample_space.detector_frame_info[scan]["probe_centre_continuous"]
        else:
            c_x, c_y = self.sample_space.detector_frame_info[scan]["probe_centre_continuous"]

        # Probes:
        if probe_type == "constant":
            initial_solution[:] = 1.0

        elif probe_type == "gaussian":
            sd = max(radius / 2.0, 1e-12)
            if self.sample_space.dimension == 1:
                dx = (x_mesh - c_x) / sd
                initial_solution = np.exp(-0.5 * dx**2)
            else:
                dx = (x_mesh - c_x) / sd
                dy = (y_mesh - c_y) / sd
                initial_solution = np.exp(-0.5 * (dx**2 + dy**2))

        elif probe_type == "sinusoidal":
            if self.sample_space.dimension == 1:
                initial_solution = np.sin(np.pi * x_mesh)
            else:
                initial_solution = np.sin(np.pi * x_mesh) * np.sin(np.pi * y_mesh)

        elif probe_type == "complex_exp":
            if self.sample_space.dimension == 1:
                initial_solution = -1j * np.exp(1j * np.pi * x_mesh)
            else:
                initial_solution = np.exp(1j * np.pi * (x_mesh + y_mesh))

        elif probe_type == "dirichlet_test":
            if self.sample_space.dimension == 1:
                initial_solution = (
                    np.sin(1 * np.pi * x_mesh) * np.sin(1 * np.pi * 0.5) +
                    0.5 * np.sin(5 * np.pi * x_mesh) * np.sin(5 * np.pi * 0.5) +
                    0.2 * np.sin(9 * np.pi * x_mesh) * np.sin(9 * np.pi * 0.5)
                )
            else:
                initial_solution = (
                    np.sin(1 * np.pi * x_mesh) * np.sin(1 * np.pi * y_mesh) +
                    0.5 * np.sin(5 * np.pi * x_mesh) * np.sin(5 * np.pi * y_mesh) +
                    0.2 * np.sin(9 * np.pi * x_mesh) * np.sin(9 * np.pi * y_mesh)
                )

        elif probe_type == "neumann_test":
            if self.sample_space.dimension == 1:
                initial_solution = (
                    np.cos(1 * np.pi * x_mesh) * np.cos(1 * np.pi * 0) +
                    0.5 * np.cos(2 * np.pi * x_mesh) * np.cos(2 * np.pi * 0) +
                    0.2 * np.cos(5 * np.pi * x_mesh) * np.cos(5 * np.pi * 0)
                )
            else:
                initial_solution = (
                    np.cos(1 * np.pi * x_mesh) * np.cos(1 * np.pi * y_mesh) +
                    0.5 * np.cos(2 * np.pi * x_mesh) * np.cos(2 * np.pi * y_mesh) +
                    0.2 * np.cos(5 * np.pi * x_mesh) * np.cos(5 * np.pi * y_mesh)
                )

        elif probe_type == "airy_disk":
            from scipy.special import j1
            if self.sample_space.dimension == 1:
                r = np.abs(x_mesh - c_x)
                scaled_r = np.pi * r / max(radius, 1e-12)
                amp = np.ones_like(r)
                m = scaled_r != 0
                amp[m] = (2 * j1(scaled_r[m]) / scaled_r[m])**2
            else:
                r = np.sqrt((x_mesh - c_x) ** 2 + (y_mesh - c_y) ** 2)
                scaled_r = np.pi * r / max(radius, 1e-12)
                amp = np.ones_like(r)
                m = scaled_r != 0
                amp[m] = (2 * j1(scaled_r[m]) / scaled_r[m])**2
            initial_solution = amp
        elif probe_type == "disk":
            if self.sample_space.dimension == 1:
                r = np.abs(x_mesh - c_x)
            else:
                r = np.sqrt((x_mesh - c_x) ** 2 + (y_mesh - c_y) ** 2)
            amp = np.where(r <= radius, 1.0, 0)
            initial_solution = amp
        elif probe_type == "blurred_disk":
            if self.sample_space.dimension == 1:
                r = np.abs(x_mesh - c_x)
                pix_area = self.sample_space.dx
                area = (2 * radius)
            else:
                r = np.hypot(x_mesh - c_x, y_mesh - c_y)
                pix_area = self.sample_space.dx * self.sample_space.dy
                area = (np.pi * radius**2)

            # Amplitude term: smooth disk with cosine edge
            portion_blur = self._disk_blur  # 50% of radius
            inner = radius - portion_blur * radius  # Inner radius for smooth edge
            t = np.clip((r - inner) / max(portion_blur * radius, 1e-12), 0.0, 1.0)  # 0..1 in the rim
            rim = portion_blur * (1 + np.cos(np.pi * t))                       # 1→0 smoothly
            amp = np.where(r <= inner, 1.0, np.where(r >= radius, 0.0, rim))

            # Optional: keep ∑|amp|^2 ≈ area of ideal disk so brightness stays comparable
            target = area / pix_area
            s = np.sqrt(target / (amp**2).sum())
            amp *= s
            initial_solution = amp


        else:
            raise ValueError("Not a valid initial condition type")
        
        # #quadratic phase (focus)
        k = self.sample_space.k
        # Quadratic phase (focus)
        if probe_focus is not None and np.isfinite(probe_focus) and probe_focus != 0.0:
            if self.sample_space.dimension == 1:
                phase_focus = np.exp(-1j * k * (x_mesh - c_x)**2 / (2.0 * probe_focus))
            else:
                r2 = (x_mesh - c_x)**2 + (y_mesh - c_y)**2
                phase_focus = np.exp(-1j * k * r2 / (2.0 * probe_focus))
            initial_solution = initial_solution * phase_focus

        # Linear tilt phase (consistent with same focus)
        if self.sample_space.dimension == 1:
            if angle_x != 0.0:
                phase_tilt = np.exp(1j * k * (x_mesh - c_x) * np.sin(angle_x))
                initial_solution = initial_solution * phase_tilt
        elif self.sample_space.dimension == 2:
            if angle_x != 0.0 or angle_y != 0.0:
                phase_tilt = np.exp(1j * (
                    k * (x_mesh - c_x) * np.sin(angle_x) +
                    k * (y_mesh - c_y) * np.sin(angle_y)
                ))
                initial_solution = initial_solution * phase_tilt

        return initial_solution

        
        # max_val = np.max(np.abs(initial_solution))
        # if max_val == 0:
        #     return initial_solution
        # return self.signal_strength * initial_solution / max_val
