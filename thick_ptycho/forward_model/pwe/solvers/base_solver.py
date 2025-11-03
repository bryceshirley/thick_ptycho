import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional, Tuple

from thick_ptycho.forward_model.base.base_solver import BaseForwardModelSolver
from thick_ptycho.thick_ptycho.forward_model.pwe.operators import PWEFiniteDifferences


class BasePWESolver(BaseForwardModelSolver):
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
        self.pwe_finite_differences = PWEFiniteDifferences(simulation_space, ptycho_object)
        self.block_size = self.pwe_finite_differences.block_size

    # ------------------------------------------------------------------
    # LU setup
    # ------------------------------------------------------------------
    def presolve_setup(self, n: Optional[np.ndarray] = None, mode: str = "forward"):
        """Precompute LU factorizations for a given propagation mode."""
        assert mode in ["forward", "adjoint", "reverse"], \
            f"Unsupported mode: {mode}"
        assert not self.simulation_space.thin_sample, \
            "presolve_setup does not support thin samples."
        self._get_or_construct_lu(n=n, mode=mode)

    def _get_or_construct_lu(self, n: Optional[np.ndarray] = None,
                       mode: str = "forward"):
        """Override in subclasses."""
        raise NotImplementedError

    def get_gradient(self,n: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the forward model with respect to the refractive index.

        Parameters
        ----------
        n : ndarray
            Current estimate of the refractive index field.

        Returns
        -------
        gradient : ndarray
            Gradient of the forward model with respect to n.
        """
        return self.pwe_finite_differences.setup_inhomogeneous_forward_model(
             n=n, grad=True)