import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional, Tuple

from thick_ptycho.thick_ptycho.forward_model.base import BaseForwardModel
from thick_ptycho.forward_model.pwe_finite_differences import PWEFiniteDifferences


class ForwardModelPWEIterative(BaseForwardModel):
    """Iterative LU-based slice-by-slice propagation solver."""

    def __init__(self, simulation_space, **kwargs):
        super().__init__(simulation_space, "pwe_iterative", **kwargs)
        self.pwe_finite_differences = PWEFiniteDifferences(simulation_space, **kwargs)
        self.block_size = self.pwe_finite_differences.block_size


    def prepare_solver(self, n: Optional[np.ndarray] = None, iterative_lu=None, 
                       adjoint=False, reverse=False):
        """Optional pre-solve. Override in subclasses."""
        if not self.thin_sample:
            if iterative_lu is None:
                iterative_lu = self.construct_iterative_lu(
                    n=n, reverse=reverse, adjoint=adjoint)
    
    def construct_iterative_lu(self, n: Optional[np.ndarray] = None, adjoint: bool = False,
                            reverse: bool = False) -> Tuple[Optional[spla.SuperLU], Optional[np.ndarray]]:
        """Compute the LU of the PWE blocks."""
        # Precompute C, A_mod, B_mod, LU factorizations
        A_lu_list = []
        B_with_object_list = []
        object_slices = self.sample_space.create_sample_slices(
                            self.thin_sample,
                            n=n).reshape(-1, self.nz - 1)

        # Create Linear System and Apply Boundary Conditions
        A, B, b = self.pwe_finite_differences.create_system_slice()

        if reverse:
            A, B = B, A
            b = - b

        # Iterate over the z dimension
        for j in range(1, self.nz):
            if reverse:
                C = - sp.diags(object_slices[:, -j])
            elif adjoint:
                C = sp.diags(object_slices[:, -j])
            else:
                C = sp.diags(object_slices[:, j - 1])

            A_with_object = A - C  # LHS Matrix
            B_with_object = B + C  # RHS Matrix

            if adjoint:
                A_with_object, B_with_object = A_with_object.conj().T, B_with_object.conj().T

            A_lu = spla.splu(A_with_object.tocsc())
            A_lu_list.append(A_lu)
            B_with_object_list.append(B_with_object)

        return (A_lu_list, B_with_object_list, b)

    def _solve_single_probe(self, angle_idx, probe_idx, n=None, reverse=False, adjoint=False, **kwargs):
        probe = self.probes[angle_idx, probe_idx, ...]

        A, B, b = self.pwe_finite_differences.create_system_slice(probe, scan_index=probe_idx)
        nz = self.simulation_space.nz
        slices = self.simulation_space.create_sample_slices(self.thin_sample, n=n, scan_index=probe_idx).reshape(-1, nz - 1)
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

    def __init__(self, simulation_space):
        super().__init__(simulation_space, "pwe_full")
        self.pwe_finite_differences = PWEFiniteDifferences(simulation_space, full_system=True)
        self.block_size = self.pwe_finite_differences.block_size
        self.use_pit = False  # Placeholder for PiT usage

        self.b0 = self.pwe_finite_differences.precompute_b0(self.probes)

    def prepare_solver(self, n: Optional[np.ndarray] = None, lu=None, test_impedance=False, **kwargs):
        """Optional pre-solve. Override in subclasses."""
        if not self.thin_sample:
            if lu is None:
                lu = spla.splu(self.pwe_finite_differences.return_forward_model_matrix(n=n))
            self.b_homogeneous = self.pwe_finite_differences.setup_homogeneous_forward_model_rhs()
        elif test_impedance:
            self.b_homogeneous = (
                self.pwe_finite_differences.test_exact_impedance_forward_model_rhs()
            )
        else:
            self.b_homogeneous = None
        return lu, self.b_homogeneous
        
    # -------------------------------------------------------------------------
    # Main probe solve
    # -------------------------------------------------------------------------
    def _solve_single_probe(self, probe, n=None, test_impedance=False, reverse=False, adjoint=False, **kwargs):
        """
        Solve for a single probe's field with optional PiT preconditioning.
        """
        if reverse:
            raise ValueError(
                "Reverse propagation is not supported in the full system solver. "
                "Please use the iterative solver for reverse propagation.")

        elif self.b_homogeneous is None:
            self.b_homogeneous = (
                self.pwe_finite_differences.setup_homogeneous_forward_model_rhs(
                    probe)
            )

        A = self.pwe_finite_differences.setup_homogeneous_forward_model_lhs(
            probe=probe
        )

        b = (
            self.linear_system.test_exact_impedance_forward_model_rhs(probe)
            if test_impedance
            else self.linear_system.setup_homogeneous_forward_model_rhs(
                self.probes, probe_idx, angle_idx
            )
        )
        b += self.linear_system.probe_contribution(probe_idx, angle_idx)
        Ck = self.linear_system.setup_inhomogeneous_forward_model(n=n, scan_index=probe_idx)

        # Full operator
        M = A - Ck

        if self.use_pit:
            sol = self._solve_with_parallel_time_preconditioner(M, b)
        else:
            sol = spla.spsolve(M, b)

        return sol.reshape((self.block_size, self.simulation_space.nz - 1))

    def _init_parallel_time_preconditioner(self):
        pass  # Implementation of PiT preconditioner initialization
    def _solve_with_parallel_time_preconditioner(self, M, b):
        pass  # Implementation of PiT preconditioned solve
