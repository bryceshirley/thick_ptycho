import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional, Tuple

from thick_ptycho.thick_ptycho.forward_model.base_solver import BaseForwardModel
from thick_ptycho.forward_model.pwe_finite_differences import PWEFiniteDifferences


class ForwardModelPWEFull(BaseForwardModel):
    """Full-system PWE solver using a single block-tridiagonal system."""

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
        self.pwe_finite_differences = PWEFiniteDifferences(simulation_space, full_system=True)
        self.block_size = self.pwe_finite_differences.block_size
        self.use_pit = False  # Placeholder for PiT usage

        # Cached LU systems and RHS for different modes
        self.lu_cache = None
        self.b_cache = None
        self._cached_n_id = None

        # Precompute B0 term if applicable
        self.b0 = self.pwe_finite_differences.precompute_b0(self.probes)

    def _get_or_construct_lu(self, n: Optional[np.ndarray] = None, mode: str = "forward"):
        """
        Retrieve or construct LU factorization for given mode.
        Caches LU and RHS to avoid recomputation.
        """
        assert mode in {"forward", "adjoint"}, f"Invalid mode: {mode!r}"

        # Reset LU cache if refractive index changed
        n_id = id(n)
        if self._cached_n_id != n_id:
            # Reinitialize caches if refractive index changed
            self.lu_cache = None
            self.b_cache = None
            self._cached_n_id = n_id

        # Build if missing
        if self.lu_cache is None: # Same for both modes
            A = self.pwe_finite_differences.return_forward_model_matrix(n=n)
            if not sp.isspmatrix_csc(A):
                A = A.tocsc()

            lu = spla.splu(A)

            self.lu_cache = lu
        
        # Build if missing
        if self.b_cache is None: # Same for both modes
            b_h = self.pwe_finite_differences.setup_homogeneous_forward_model_rhs()
            self.b_cache = b_h

        return self.lu_cache, self.b_cache

    def prepare_solver(self, n: Optional[np.ndarray] = None, mode: str = "forward"):
        """
        Precompute LU decomposition and homogeneous RHS for given mode.

        Parameters
        ----------
        n : ndarray, optional
            Refractive index field. If None, uses default.
        mode : {'forward', 'adjoint'}
            Propagation mode.
        """
        assert not getattr(self, "thin_sample", False), \
            "Full-system solver does not support thin-sample approximation."
        assert mode in {"forward", "adjoint"}, f"Invalid mode: {mode!r}"
        return self._get_or_construct_lu(n=n, mode=mode)

        
    # -------------------------------------------------------------------------
    # Main probe solve
    # -------------------------------------------------------------------------
    def _solve_single_probe(
        self,
        angle_idx: int,
        scan_idx: int,
        n: Optional[np.ndarray] = None,
        mode: str = "forward",
        rhs_block: Optional[np.ndarray] = None,
        initial_condition: Optional[np.ndarray] = None
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
        mode : {'forward', 'adjoint'}
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
        assert mode in {"forward", "adjoint"}, \
            f"Invalid mode '{mode}'. Reverse propagation is unsupported in the full solver."
        
        # Select initial probe condition
        if initial_condition is not None:
            probe = initial_condition[angle_idx, scan_idx, :]
        else:
            probe = self.probes[angle_idx, scan_idx, :]

        # Retrieve or construct LU and homogeneous RHS
        lu, b_homogeneous = self._get_or_construct_lu(n=n, mode=mode)

        # Construct right-hand side
        if rhs_block is not None:
            b = rhs_block
        else:
            probe_contribution = self.pwe_finite_differences.probe_contribution(
                scan_index=scan_idx,
                angle_index=angle_idx,
                probe=probe,
            )
            b = b_homogeneous + probe_contribution

        # Solve the global system
        if mode == "adjoint":
            u = lu.solve(b, trans="H")
        else:
            u = lu.solve(b)

        # Reshape and concatenate with initial condition
        u = u.reshape(self.nz - 1, self.block_size).T
        initial = probe.reshape(self.block_size, 1)
        return np.concatenate([initial, u], axis=1)
    
    def get_gradient(self,nk: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the forward model with respect to the refractive index.

        Parameters
        ----------
        nk : ndarray
            Current estimate of the refractive index field.

        Returns
        -------
        gradient : ndarray
            Gradient of the forward model with respect to nk.
        """
        return self.pwe_finite_differences.setup_inhomogeneous_forward_model(
             n=nk, grad=True)

    # -------------------------------------------------------------------------
    # PiT Preconditioner (placeholder)
    # -------------------------------------------------------------------------
    def _init_parallel_time_preconditioner(self):
        """Initialize Parallel-in-Time preconditioner (not yet implemented)."""
        pass

    def _solve_with_parallel_time_preconditioner(self, M, b):
        """Solve using PiT preconditioner (not yet implemented)."""
        pass

