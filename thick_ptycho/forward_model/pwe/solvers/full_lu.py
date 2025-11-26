
from dataclasses import dataclass, fields
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional
import time


from thick_ptycho.forward_model.pwe.solvers.base_solver import BasePWESolver
from thick_ptycho.forward_model.pwe.operators import BoundaryType
from thick_ptycho.forward_model.pwe.operators.finite_differences.boundary_condition_test import BoundaryConditionsTest

@dataclass
class PWEFullLUSolverCache():
    """
    Cache structure for storing precomputed variables.
    Parameters
    ----------
    lu : spla.SuperLU, optional
        LU factorization of the full PWE system.
    b : ndarray, optional
        Right-hand side vector for homogeneous propagation.
    """
    cached_n: Optional[np.ndarray] = None
    lu: Optional[spla.SuperLU] = None
    b: Optional[np.ndarray] = None

    def reset(self, n: Optional[np.ndarray] = None):
        """Reset cached variables."""
        # Reinitialize all cached variables to None
        for f in fields(self):
            setattr(self, f.name, None)
    
        # Update cached n
        self.cached_n = n

# ------------------------------------------------------------------
#  Full-system PWE Solver
# ------------------------------------------------------------------
class PWEFullLUSolver(BasePWESolver):
    """Full-system PWE solver using a single block-tridiagonal system."""

    solver_cache_class = PWEFullLUSolverCache
    def __init__(self, simulation_space, ptycho_probes,
                 bc_type: BoundaryType = BoundaryType.IMPEDANCE,
                 results_dir="", use_logging=False, verbose=False, log=None,
                 test_bcs: BoundaryConditionsTest = None):
        super().__init__(
            simulation_space,
            ptycho_probes,
            bc_type=bc_type,
            results_dir=results_dir,
            use_logging=use_logging,
            verbose=verbose,
            log=log,
            test_bcs=test_bcs
        )
        self.b0 = self.pwe_finite_differences.precompute_b0(self.probes)

    def _construct_solve_cache(self, n: Optional[np.ndarray] = None, 
                               mode: str = "forward",
                               scan_idx: Optional[int] = 0,
                               proj_idx: Optional[int] = 0) -> None:
        """
        Retrieve or construct LU factorization for given mode.
        Caches LU and RHS to avoid recomputation.
        Parameters
        ----------
        n : ndarray, optional
            Refractive index field.
        mode : {'forward', 'adjoint','forward_rotated', 'adjoint_rotated'}
            Propagation mode.
        """
        A = self.pwe_finite_differences.return_forward_model_matrix(n=n, 
                                                                    scan_index=scan_idx
                                                                    ).tocsc()
        self.projection_cache[proj_idx].modes[mode].lu = spla.splu(A)
        self.projection_cache[proj_idx].modes[mode].cached_n = n


        if self.test_bcs is None:
            self.projection_cache[proj_idx].modes[mode].b = self.pwe_finite_differences.setup_homogeneous_forward_model_rhs()
        else:
            self.projection_cache[proj_idx].modes[mode].b = self.test_bcs.test_exact_impedance_forward_model_rhs()

        
    # -------------------------------------------------------------------------
    # Main probe solve
    # -------------------------------------------------------------------------
    def _solve_single_probe_impl(
        self,
        scan_idx: int=0,
        proj_idx: int=0,
        probe: Optional[np.ndarray] = None,
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

        self._log("Retrieving LU decomposition and setting up system...")
        time_start = time.time()
        # Retrieve or construct LU and homogeneous RHS
        lu = self.projection_cache[proj_idx].modes[mode].lu
        b_homogeneous = self.projection_cache[proj_idx].modes[mode].b

        # Build RHS
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
        if mode == "adjoint":
            u = lu.solve(b, trans="H")
        else:
            u = lu.solve(b)
        time_end = time.time()
        self._log(f"Direct LU solve time: {time_end - time_start:.2f} seconds.\n")


        # Reshape and concatenate with initial condition
        u = u.reshape(self.nz - 1, self.block_size).T
        initial = probe.reshape(self.block_size, 1)
        return np.concatenate([initial, u], axis=1)





