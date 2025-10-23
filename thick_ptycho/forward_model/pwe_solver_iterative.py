import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional, Tuple

from thick_ptycho.thick_ptycho.forward_model.base_pwe_solver import BaseForwardModelPWE


class ForwardModelPWEIterative(BaseForwardModelPWE):
    """Iterative LU-based slice-by-slice propagation solver."""

    def __init__(self, simulation_space, ptycho_object, ptycho_probes,
                 results_dir="", use_logging=False, verbose=True, log=None):
        super().__init__(
            simulation_space,
            ptycho_object,
            ptycho_probes,
            results_dir=results_dir,
            use_logging=use_logging,
            verbose=verbose,
            log=log,
        )

        # LU caches for different propagation modes
        self.lu_cache = {"forward": None, "adjoint": None, "reverse": None}
        self._cached_n_id = None

    # ------------------------------------------------------------------
    # LU setup
    # ------------------------------------------------------------------
    def presolve_setup(self, n: Optional[np.ndarray] = None, mode: str = "forward"):
        """Precompute LU factorizations for a given propagation mode."""
        assert not self.simulation_space.thin_sample, \
            "presolve_setup does not support thin samples."
        self._get_or_construct_lu(n=n, mode=mode)

    def _get_or_construct_lu(self, n: Optional[np.ndarray] = None,
                       mode: str = "forward"):
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
        assert mode in {"forward", "adjoint", "reverse"}, f"Invalid mode: {mode}"
        assert not self.simulation_space.thin_sample, "presolve_setup does not support thin samples."

        # Reset LU cache if refractive index changed
        n_id = id(n)
        if self._cached_n_id != n_id:
            self.lu_cache = {"forward": None, "adjoint": None, "reverse": None}
            self._cached_n_id = n_id

        if self.lu_cache[mode] is None:
            self.lu_cache[mode] = self.construct_iterative_lu(n=n, mode=mode)

        return self.lu_cache[mode]

    def construct_iterative_lu(self, n: Optional[np.ndarray] = None, mode: str = "forward") -> Tuple[Optional[spla.SuperLU], Optional[np.ndarray]]:
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
        assert mode in {"forward", "adjoint", "reverse"}, f"Invalid mode: {mode}"

        # Precompute C, A_mod, B_mod, LU factorizations
        A_lu_list, B_with_object_list = [], []
        object_contribution = self.ptycho_object.create_object_contribution(
                            n=n).reshape(-1, self.nz - 1)

        # Create Linear System and Apply Boundary Conditions
        A, B, b = self.pwe_finite_differences.generate_zstep_matrices()

        if mode == "reverse":
            A, B, b = B, A, -b

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

        return (A_lu_list, B_with_object_list, b)

    # ------------------------------------------------------------------
    # Main probe solve
    # ------------------------------------------------------------------
    def _solve_single_probe(self,
            angle_idx: int,
            scan_idx: int,
            n: Optional[np.ndarray] = None,
            mode: str = "forward",
            rhs_block: Optional[np.ndarray] = None,
            initial_condition: Optional[np.ndarray] = None
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
        mode : {'forward', 'adjoint', 'reverse'}
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
        assert mode in {"forward", "adjoint", "reverse"}, f"Invalid mode: {mode}"

        # Select initial probe condition
        if initial_condition is not None:
            probe = initial_condition[angle_idx, scan_idx, :]
        else:
            probe = self.probes[angle_idx, scan_idx, :]
            
        u = np.zeros((self.block_size, self.nz), dtype=complex)
        u[:, 0] = probe.flatten()

        if rhs_block is not None:
            rhs_block = rhs_block.reshape(self.nz - 1, self.block_size).T

        # ------------------------------------------------------
        # Select or construct LUs (always via construct_iterative_lu)
        # ------------------------------------------------------
        A_lu_list, B_list, b = self._get_or_construct_lu(n=n, mode=mode)

        # ------------------------------------------------------
        # Main solve loop (reuse LU always)
        # ------------------------------------------------------
        for j in range(1, self.nz):
            # if test_impedance:
            #     b = self.linear_system.test_exact_impedance_rhs_slice(j)
            if rhs_block is not None:
                b = rhs_block[:, -j] if mode == "adjoint" else rhs_block[:, j - 1]

            rhs_matrix = B_list[j - 1] @ u[:, j - 1] + b
            u[:, j] = A_lu_list[j - 1].solve(rhs_matrix)

        if mode == "adjoint":
            u = np.flip(u, axis=1)
        return u