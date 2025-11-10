
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional
import time


from thick_ptycho.forward_model.pwe.solvers.base_solver import BasePWESolver
from thick_ptycho.forward_model.pwe.operators import BoundaryType
from thick_ptycho.forward_model.pwe.operators.finite_differences.boundary_condition_test import BoundaryConditionsTest

class PWEFullLUSolver(BasePWESolver):
    """Full-system PWE solver using a single block-tridiagonal system."""

    def __init__(self, simulation_space, ptycho_object, ptycho_probes,
                 bc_type: BoundaryType = BoundaryType.IMPEDANCE,
                 results_dir="", use_logging=False, verbose=False, log=None,
                 test_bcs: BoundaryConditionsTest = None):
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

        # Cached LU systems and RHS for different modes
        # Cache LU factorizations for rotated and non-rotated modes
        self.lu_cache = {
            "projection_0": None,
            "projection_1": None,  # Rotated
        }
        self.b_cache = None
        self._cached_n_id = None

        # Precompute B0 term if applicable
        self.pwe_finite_differences.full_system = True
        self.b0 = self.pwe_finite_differences.precompute_b0(self.probes)

        # Solver type (for logging purposes)
        self.solver_type = "Block PWE Full Solver with LU Decomposition"

        # For testing purposes
        self.test_bcs = test_bcs

    def _get_or_construct_lu(self, n: Optional[np.ndarray] = None, mode: str = "forward"):
        """
        Retrieve or construct LU factorization for given mode.
        Caches LU and RHS to avoid recomputation.
        Parameters
        ----------
        n : ndarray, optional
            Refractive index field. If None, uses self.ptycho_object.n_true.
        mode : {'forward', 'adjoint','forward_rotated', 'adjoint_rotated'}
            Propagation mode.
        """
        assert mode in {"forward", "adjoint", "forward_rotated", "adjoint_rotated"}, f"Invalid mode: {mode!r}"
        if n is None:
            n = self.ptycho_object.n_true

        # Determine projection key and possibly rotate n
        if mode in {"forward_rotated", "adjoint_rotated"}:
            n = self.rotate_n(n)
            projection_key = "projection_1"
        else:
            projection_key = "projection_0"

        # Reset LU cache if refractive index changed
        n_id = id(n)
        if self._cached_n_id != n_id:
            # Reinitialize caches if refractive index changed
            self.lu_cache[projection_key] = None
            self.b_cache = None
            self._cached_n_id = n_id

        # Build if missing
        if self.lu_cache[projection_key] is None:  # Same for both modes
            A = self.pwe_finite_differences.return_forward_model_matrix(n=n)
            if not sp.isspmatrix_csc(A):
                A = A.tocsc()

            lu = spla.splu(A)

            self.lu_cache[projection_key] = lu

        # Build if missing
        if self.b_cache is None:  # Same for both modes
            b_h = self.pwe_finite_differences.setup_homogeneous_forward_model_rhs()
            self.b_cache = b_h

        return self.lu_cache, self.b_cache

    def reset_cache(self):
        self.lu_cache = {
            "projection_0": None,
            "projection_1": None,  # Rotated
        }
        self.b_cache = None
        self._cached_n_id = None


    def prepare_solver(self, n: Optional[np.ndarray] = None, mode: str = "forward"):
        """
        Precompute LU decomposition and homogeneous RHS for given mode.

        Parameters
        ----------
        n : ndarray, optional
            Refractive index field. If None, uses default.
        mode : {'forward', 'adjoint','forward_rotated','adjoint_rotated'}
            Propagation mode.
        """
        assert not getattr(self, "solve_reduced_domain", False), \
            "Full-system solver does not support thin-sample approximation."
        assert mode in {"forward", "adjoint","forward_rotated","adjoint_rotated"}, f"Invalid mode: {mode!r}"
        
        return self._get_or_construct_lu(n=n, mode=mode)

        
    # -------------------------------------------------------------------------
    # Main probe solve
    # -------------------------------------------------------------------------
    def _solve_single_probe(
        self,
        scan_idx: int=0,
        probe: Optional[np.ndarray] = None,
        n: Optional[np.ndarray] = None,
        mode: str = "forward",
        rhs_block: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve for a single probe's field using the full block-tridiagonal system.

        Parameters
        ----------
        angle_idx : int
            Illumination angle index.
        scan_idx : int
            Probe position index.
        n : ndarray, optional
            Refractive index field.
        mode : {'forward', 'adjoint','forward_rotated','adjoint_rotated'}
            Propagation mode. Reverse is not supported.
        rhs_block : ndarray, optional
            Optional RHS vector for reusing precomputed blocks.
        initial_condition : ndarray, optional
            Initial probe condition to use instead of default.

        Returns
        -------
        u : ndarray
            Complex propagated field, shape (block_size, nz).
        """

        assert mode in {"forward", "adjoint","forward_rotated","adjoint_rotated"}, \
            f"Invalid mode '{mode}'. Reverse propagation is unsupported in the full solver."


        self._log("Retrieving LU decomposition and setting up system...")
        time_start = time.time()
        # Retrieve or construct LU and homogeneous RHS
        lu, b_homogeneous = self._get_or_construct_lu(n=n, mode=mode)

        # Construct right-hand side
        if rhs_block is not None:
            b = rhs_block
        else:
            probe_contribution = self.pwe_finite_differences.probe_contribution(
                scan_index=scan_idx,
                probe=probe
            )
            b = b_homogeneous + probe_contribution
        time_end = time.time()
        self._log(f"LU retrieval and setup time: {time_end - time_start:.2f} seconds.\n")

        self._log("Solving with direct LU solver...")
        time_start = time.time()
        # Determine projection key and possibly rotate n
        if mode in {"forward_rotated", "adjoint_rotated"}:
            projection_key = "projection_1"
        else:
            projection_key = "projection_0"
        if mode == "adjoint":
            u = lu[projection_key].solve(b, trans="H")
        else:
            u = lu[projection_key].solve(b)
        time_end = time.time()
        self._log(f"Direct LU solve time: {time_end - time_start:.2f} seconds.\n")


        # Reshape and concatenate with initial condition
        u = u.reshape(self.nz - 1, self.block_size).T
        initial = probe.reshape(self.block_size, 1)
        return np.concatenate([initial, u], axis=1)





