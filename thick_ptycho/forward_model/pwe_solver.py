import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from thick_ptycho.forward_model.base_solver import BaseForwardModel
from thick_ptycho.forward_model.PWE.linear_system import ForwardModelPWE


class ForwardModelPWEIterative(BaseForwardModel):
    """Iterative LU-based slice-by-slice propagation solver."""

    def __init__(self, sample_space, thin_sample=True, probe_angles_list=None, **kwargs):
        super().__init__(sample_space, "pwe_iterative", thin_sample, probe_angles_list, **kwargs)
        self.linear_system = ForwardModelPWE(sample_space, thin_sample, full_system=False)
        self.block_size = self.linear_system.block_size

    def _solve_single_probe(self, angle_idx, probe_idx, n=None, reverse=False, adjoint=False, **kwargs):
        probe = self.probes[angle_idx, probe_idx, ...]
        A, B, b = self.linear_system.create_system_slice(probe, scan_index=probe_idx)
        nz = self.sample_space.nz
        slices = self.sample_space.create_sample_slices(self.thin_sample, n=n, scan_index=probe_idx).reshape(-1, nz - 1)
        u = np.zeros((self.block_size, nz), dtype=complex)
        u[:, 0] = probe.flatten()

        for j in range(1, nz):
            C = sp.diags(slices[:, j - 1])
            A_mod, B_mod = A - C, B + C
            rhs = B_mod @ u[:, j - 1] + b
            u[:, j] = spla.spsolve(A_mod, rhs)
        return u


class ForwardModelPWEFull(BaseForwardModel):
    """Full-system PWE solver using a single block-tridiagonal system."""

    def __init__(self, sample_space, thin_sample=False, probe_angles_list=None, **kwargs):
        super().__init__(sample_space, "pwe_full", thin_sample, probe_angles_list, **kwargs)
        self.linear_system = ForwardModelPWE(sample_space, thin_sample, full_system=True)
        self.block_size = self.linear_system.block_size

    def _solve_single_probe(self, angle_idx, probe_idx, n=None, test_impedance=False, **kwargs):
        probe = self.probes[angle_idx, probe_idx, ...]
        A = self.linear_system.setup_homogeneous_forward_model_lhs(scan_index=probe_idx, angle_index=angle_idx)
        b = (
            self.linear_system.test_exact_impedance_forward_model_rhs(probe)
            if test_impedance
            else self.linear_system.setup_homogeneous_forward_model_rhs(self.probes, probe_idx, angle_idx)
        )
        b += self.linear_system.probe_contribution(probe_idx, angle_idx, self.probes)
        Ck = self.linear_system.setup_inhomogeneous_forward_model(n=n, scan_index=probe_idx)
        sol = spla.spsolve(A - Ck, b)
        return sol.reshape((self.block_size, self.sample_space.nz - 1))
