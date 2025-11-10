import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional, Tuple

from thick_ptycho.forward_model.pwe.solvers.base_solver import BasePWESolver
from thick_ptycho.forward_model.pwe.operators import BoundaryType
from thick_ptycho.forward_model.pwe.operators.finite_differences.boundary_condition_test import BoundaryConditionsTest


class PWEIterativeLUSolver(BasePWESolver):
    """Iterative LU-based slice-by-slice propagation solver."""

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
        )

        # LU caches for different propagation modes
        self.lu_cache = {"forward": None, "adjoint": None, "reverse": None,
                         "forward_rotated": None, "adjoint_rotated": None, "reverse_rotated": None}
        self._cached_n_id = None

        # Solver type (for logging purposes)
        self.solver_type = "Iterative PWE Full Solver"

        # For testing purposes
        self.test_bcs = test_bcs

    # ------------------------------------------------------------------
    # LU setup
    # ------------------------------------------------------------------
    def presolve_setup(self, n: Optional[np.ndarray] = None, mode: str = "forward"):
        """Precompute LU factorizations for a given propagation mode."""
        assert not self.simulation_space.solve_reduced_domain, \
            "presolve_setup does not support thin samples."
        self._get_or_construct_lu(n=n, mode=mode)

    def _get_or_construct_lu(self, n: Optional[np.ndarray] = None,
                       mode: str = "forward", scan_idx: Optional[int] = 0):
        """ Retrieve or build LU factorizations for the specified mode.
        Parameters
        ----------
        n : ndarray, optional
            Refractive index field. If None, uses self.ptycho_object.n_true.
        mode : {'forward', 'adjoint', 'reverse'}
            Propagation mode.

        Returns
        -------
        (A_lu_list, B_list, b)
            Precomputed LU factorizations, RHS matrices, and right-hand side vector.
        """
        assert mode in {"forward", "adjoint", "reverse", "forward_rotated", "adjoint_rotated", "reverse_rotated"}, f"Invalid mode: {mode}"
        #assert not self.simulation_space.solve_reduced_domain, "presolve_setup does not support thin samples."
        if n is None:
            n = self.ptycho_object.n_true
        if mode in {"forward_rotated", "adjoint_rotated", "reverse_rotated"}:
            n = self.rotate_n(n)

        # Reset LU cache if refractive index changed
        n_id = id(n)
        if self._cached_n_id != n_id:
            self.lu_cache = {"forward": None, "adjoint": None, "reverse": None,
                             "forward_rotated": None, "adjoint_rotated": None, "reverse_rotated": None}
            self._cached_n_id = n_id

        if self.lu_cache[mode] is None:
            self.lu_cache[mode] = self.construct_iterative_lu(n=n, mode=mode, scan_idx=scan_idx)

        return self.lu_cache[mode]
    
    def reset_cache(self):
        """Reset the LU cache."""
        self.lu_cache = {"forward": None, "adjoint": None, "reverse": None,
                         "forward_rotated": None, "adjoint_rotated": None, "reverse_rotated": None}
        self._cached_n_id = None

    def construct_iterative_lu(self, n: Optional[np.ndarray] = None, 
                               mode: str = "forward",
                               scan_idx: Optional[int] = 0) -> Tuple[Optional[spla.SuperLU], Optional[np.ndarray]]:
        """
        Compute the LU of the PWE blocks.

        Parameters
        ----------
        n : ndarray, optional
            Refractive index field. If None, uses self.ptycho_object.n_true.
        mode : {'forward', 'adjoint', 'reverse'}
            Propagation mode.
         
        Return
        ------
        iterative_lu : Tuple[[spla.SuperLU], [np.ndarray], [np.ndarray]]
            Tuple of lists of LU factorizations and modified B matrices for each slice and the RHS vector b.
        """
        assert mode in {"forward", "adjoint", "reverse", "forward_rotated", "adjoint_rotated", "reverse_rotated"}, f"Invalid mode: {mode}"

        # Precompute C, A_mod, B_mod, LU factorizations

        # Create Linear System and Apply Boundary Conditions
        A, B, b = self.pwe_finite_differences.generate_zstep_matrices()

        if mode == "reverse":
            A, B, b = B, A, -b

        A_lu_list, B_with_object_list = self._build_lu_factors(A, B, mode, n, 
                                                               scan_idx=scan_idx)

        return (A_lu_list, B_with_object_list, b)

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
    def _solve_single_probe(self,
            scan_idx: int=0,
            probe: Optional[np.ndarray] = None,
            n: Optional[np.ndarray] = None,
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
        # if proj_idx == 1 and mode in {"forward", "adjoint", "reverse"}:
        #     mode = mode + "_rotated"
        assert mode in {"forward", "adjoint", "reverse", "forward_rotated", "adjoint_rotated", "reverse_rotated"}, f"Invalid mode: {mode}"

        # # Select initial probe condition
        # if probe is None
        #     probe = initial_condition[angle_idx, scan_idx, :]
        # else:
        #     probe = self.probes[angle_idx, scan_idx, :]
        if probe is None:
            probe = np.zeros((self.block_size,), dtype=complex)
            
        u = np.zeros((self.block_size, self.nz), dtype=complex)
        u[:, 0] = probe.flatten()

        # if rhs_block is not None:
        #     rhs_block = rhs_block.reshape(self.nz - 1, self.block_size).T

        # ------------------------------------------------------
        # Select or construct LUs (always via construct_iterative_lu)
        # ------------------------------------------------------
        A_lu_list, B_list, b = self._get_or_construct_lu(n=n, mode=mode, scan_idx=scan_idx)

        # ------------------------------------------------------
        # Main solve loop (reuse LU always)
        # ------------------------------------------------------
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
    
    def _solve_probes_batch(self, probes, n=None, mode="forward", rhs_block=None):
        P = probes.shape[0]   # now P = num_angles * num_probes
        u = np.zeros((P, self.block_size, self.nz), dtype=complex)
        u[:, :, 0] = probes

        A_lu_list, B_list, b_global = self._get_or_construct_lu(n=n, mode=mode)

        if rhs_block is not None:
            rhs_block = rhs_block.reshape(P, self.block_size, self.nz - 1)

        for j in range(1, self.nz):
            b = rhs_block[:, :, -j] if (rhs_block is not None and mode == "adjoint") \
                                    else rhs_block[:, :, j-1] if rhs_block is not None \
                                    else b_global
            if test_impedance:
                    b = BoundaryConditionsTest.test_exact_impedance_rhs_step(j)

            rhs = (B_list[j-1] @ u[:, :, j-1].T).T + b

            u[:, :, j] = A_lu_list[j-1].solve(rhs.T).T

        if mode == "adjoint":
            u = np.flip(u, axis=2)

        return u