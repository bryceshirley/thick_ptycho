from dataclasses import dataclass, fields
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional, Tuple

from thick_ptycho.forward_model.pwe.solvers.base_solver import BasePWESolver
from thick_ptycho.forward_model.pwe.operators import BoundaryType
from thick_ptycho.forward_model.pwe.operators.finite_differences.boundary_condition_test import BoundaryConditionsTest

@dataclass
class PWEIterativeLUSolverCache():
    """Cache structure for storing precomputed variables in the PWE Iterative LU solver.
     cached_n_id: Optional[int] = None
    """
    cached_n_id: Optional[int] = None
    A_lu_list: Optional[Tuple[spla.SuperLU]] = None
    B_list: Optional[Tuple[np.ndarray]] = None
    b: Optional[np.ndarray] = None

    def reset(self, n_id: Optional[int] = None):
        """Reset cached variables."""
        # Reinitialize all cached variables to None
        for f in fields(self):
            setattr(self, f.name, None)
    
        # Update cached n id
        object.__setattr__(self, 'cached_n_id', n_id)


class PWEIterativeLUSolver(BasePWESolver):
    """Iterative LU-based slice-by-slice propagation solver."""
    solver_cache_class = PWEIterativeLUSolverCache
    def __init__(self, simulation_space, ptycho_object, ptycho_probes,
                 bc_type: BoundaryType = BoundaryType.IMPEDANCE,
                 results_dir="", use_logging=False, verbose=False, 
                 log=None, test_bcs: BoundaryConditionsTest = None):
        super().__init__(
            simulation_space,
            ptycho_object,
            ptycho_probes,
            bc_type=bc_type,
            results_dir=results_dir,
            use_logging=use_logging,
            verbose=verbose,
            log=log,
            test_bcs=test_bcs
        )

    # ------------------------------------------------------------------
    # LU Factorization construction
    # ------------------------------------------------------------------
    def _construct_solve_cache(self, n: Optional[np.ndarray] = None, 
                               mode: str = "forward",
                               scan_idx: Optional[int] = 0,
                               proj_idx: Optional[int] = 0) -> None:
        """
        Compute the LU of the PWE blocks Updates Projection Cache.

        Parameters
        ----------
        n : ndarray, optional
            Refractive index field. If None, uses self.ptycho_object.n_true.
        mode : {'forward', 'adjoint', 'reverse'}
            Propagation mode.
        scan_idx : int
            Index of the probe position.
        proj_idx : int
            Index of the projection (for tomographic scans).
        """
        # Create Linear System and Apply Boundary Conditions
        #A, B, b = self.pwe_finite_differences.generate_zstep_matrices()
        A, B, b = self.pwe_finite_differences._get_or_generate_step_matrices()

        if mode == "reverse":
            A, B, b = B, A, -b

        A_lu_list, B_with_object_list = self._build_lu_factors(A, B, mode, n, 
                                                     scan_idx=scan_idx)
        
        # Cache the results
        self.projection_cache[proj_idx].modes[mode].A_lu_list = A_lu_list
        self.projection_cache[proj_idx].modes[mode].B_list = B_with_object_list
        self.projection_cache[proj_idx].modes[mode].b = b

    def _build_lu_factors(self, A, B, mode, n, 
                          scan_idx: Optional[int] = 0):
       
        A_lu_list, B_with_object_list = [], []

        object_contribution = self.ptycho_object.create_object_contribution(
                            n=n, scan_index=scan_idx).reshape(-1, self.nz - 1)

        # Iterate over the z dimension
        for j in range(1, self.nz):
            if mode == "reverse":
                C = - sp.diags(object_contribution[:, -j])
            elif mode == "adjoint":
                C = sp.diags(object_contribution[:, -j])
            else:
                C = sp.diags(object_contribution[:, j - 1])

            A_with_object = A - C  # LHS Matrix
            B_with_object = B + C  # RHS Matrix

            if mode == "adjoint":
                A_with_object, B_with_object = A_with_object.conj().T, B_with_object.conj().T

            A_lu = spla.splu(A_with_object.tocsc())
            A_lu_list.append(A_lu)
            B_with_object_list.append(B_with_object)
        return A_lu_list, B_with_object_list

    # ------------------------------------------------------------------
    # Main probe solve
    # ------------------------------------------------------------------
    def _solve_single_probe_impl(self,
            scan_idx: int=0,
            proj_idx: int=0,
            probe: Optional[np.ndarray] = None,
            mode: str = "forward",
            rhs_block: Optional[np.ndarray] = None,
            ) -> np.ndarray:
        """
        Perform forward (or backward) propagation through `time-steps` in z.
        for a single probe position, using precomputed LU factorizations.

        Parameters
        ----------
        probe : ndarray
            The probe field to propagate.
        scan_idx : int
            Index of the probe position.
        n : ndarray, optional
            Refractive index field. If None, uses self.ptycho_object.n_true.
        mode : {'forward', 'adjoint', 'reverse', 'forward_rotated', 'adjoint_rotated', 'reverse_rotated'}
            Propagation mode.
        rhs_block : ndarray, optional
            If provided, use this as the RHS block instead of the default only use
            if presolve_setup was called.
        initial_condition : ndarray, optional
            Initial probe condition to use instead of default.

        Returns
        -------
        u : ndarray
            Wavefield propagated through all slices for the given probe.
            Shape: (block_size, nz)
        """
        # Select initial probe condition
        if probe is None:
            probe = np.zeros((self.block_size,), dtype=complex)
            
        u = np.zeros((self.block_size, self.nz), dtype=complex)
        u[:, 0] = probe.flatten()


        # Select (and/or construct LUs)
        A_lu_list = self.projection_cache[proj_idx].modes[mode].A_lu_list
        B_list = self.projection_cache[proj_idx].modes[mode].B_list
        b = self.projection_cache[proj_idx].modes[mode].b

        # Solve the system z-step by z-step
        for j in range(1, self.nz):
            if rhs_block is not None:
                b = rhs_block[:, -j] if mode == "adjoint" else rhs_block[:, j - 1]

            if self.test_bcs is not None:
                b = self.test_bcs.test_exact_impedance_rhs_step(j)

            rhs_matrix = B_list[j - 1] @ u[:, j - 1] + b
            u[:, j] = A_lu_list[j - 1].solve(rhs_matrix)

        if mode == "adjoint":
            u = np.flip(u, axis=1)
        return u