from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import numpy as np

from thick_ptycho.forward_model.base.base_solver import BaseForwardModelSolver
from thick_ptycho.forward_model.pwe.operators import BoundaryType
from thick_ptycho.forward_model.pwe.operators.finite_differences import PWEForwardModel
from thick_ptycho.forward_model.pwe.operators.finite_differences.boundary_condition_test import (
    BoundaryConditionsTest,
)


@dataclass
class ProjectionCache:
    modes: dict[str, type]

    def __init__(self, cache_cls):
        self.modes = {m: cache_cls() for m in ("forward", "adjoint", "reverse")}

    def reset(self, cache_cls):
        self.modes = {m: cache_cls() for m in ("forward", "adjoint", "reverse")}

    def reset_mode(self, mode: str):
        self.modes[mode] = type(self.modes[mode])()


# ------------------------------------------------------------------
#  Base PWE Solver
# ------------------------------------------------------------------
class BasePWESolver(BaseForwardModelSolver, ABC):
    """Iterative LU-based slice-by-slice propagation solver."""

    solver_cache_class: type = None

    def __init__(
        self,
        simulation_space,
        ptycho_probes,
        bc_type: BoundaryType = BoundaryType.IMPEDANCE,
        results_dir="",
        use_logging=False,
        verbose=True,
        log=None,
        test_bcs: BoundaryConditionsTest = None,
    ):
        super().__init__(
            simulation_space,
            ptycho_probes,
            results_dir=results_dir,
            use_logging=use_logging,
            verbose=verbose,
            log=log,
        )
        # For testing purposes
        self.test_bcs = test_bcs

        self.bc_type = bc_type
        self.pwe_finite_differences = PWEForwardModel(simulation_space, bc_type=bc_type)
        self.block_size = self.pwe_finite_differences.block_size

        # Projection handling (for tomographic cases)
        self.create_projection_cache()

    # ------------------------------------------------------------------
    #  Cache Management and Projection Handling
    # ------------------------------------------------------------------
    @abstractmethod
    def _solve_single_probe_impl(self, scan_idx, proj_idx, probe, mode, rhs_block):
        """Subclasses must implement the actual single-probe solver."""
        raise NotImplementedError

    def create_projection_cache(self):
        """Create projection cache for multiple projections."""
        self.projection_cache = []
        for _ in range(self.num_projections):
            self.projection_cache.append(ProjectionCache(self.solver_cache_class))

    @abstractmethod
    def _construct_solve_cache(
        self, n: np.ndarray, mode: str = "forward", scan_idx: int = 0, proj_idx: int = 0
    ):
        """
        Subclasses must implement this to perform the expensive computation
        (e.g., building LU factors, PiT operators)
        """
        raise NotImplementedError

    def _get_or_construct_cache(
        self, n: Optional[np.ndarray], mode: str, proj_idx: int = 0, scan_idx: int = 0
    ):
        """
        Generic function to check the cache, reset it if 'n' has changed,
        call the specific construction logic, and return the populated cache instance.
        """
        assert mode in self.projection_cache[proj_idx].modes, f"Invalid mode: {mode!r}"

        # Potentially rotate object 'n'
        n = self.get_projected_obj(n=n, proj_idx=proj_idx)

        cache = self.projection_cache[proj_idx].modes[mode]

        # True if ANY field except cached_n is None
        uninitialized = any(
            getattr(cache, f.name) is None
            for f in fields(cache)
            if f.name != "cached_n"
        )

        # Rebuild cache if missing values
        if (
            cache.cached_n is None
            or uninitialized
            or self.simulation_space.solve_reduced_domain
        ):
            # Build solver matrices
            self._construct_solve_cache(
                n=n, mode=mode, proj_idx=proj_idx, scan_idx=scan_idx
            )

    def prepare_solver_caches(
        self, n: np.ndarray, modes: Tuple[str] = ("forward", "adjoint")
    ):
        """Prepare solver caches for all projections & modes used."""
        for proj_idx in range(self.num_projections):
            for mode in modes:
                self._construct_solve_cache(n=n, proj_idx=proj_idx, mode=mode)

    def get_projected_obj(
        self, n: np.ndarray, mode: str = "forward", proj_idx: int = 0
    ) -> Optional[np.ndarray]:
        """Get cached projection matrix for given projection index.
        Also handles rotation of n for different projections."""
        # Rotate n if needed
        if self.simulation_space.num_projections == 1 or proj_idx == 0:
            return n
        elif proj_idx == 1:
            return np.rot90(n, k=1)
        else:
            raise ValueError(f"Invalid projection index: {proj_idx}")

    def reset_cache(self):
        """Reset all cached variables (e.g., LU factors)."""
        for i in range(self.num_projections):
            self.projection_cache[i].reset(cache_cls=self.solver_cache_class)

    def _solve_single_probe(
        self,
        scan_idx: int = 0,
        proj_idx: int = 0,
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

        # Select (and/or construct) cache (e.g., LU factors or PiT Preconditioners)
        self._get_or_construct_cache(
            n=n, mode=mode, scan_idx=scan_idx, proj_idx=proj_idx
        )

        # Actual solving logic to be implemented in subclasses
        return self._solve_single_probe_impl(
            scan_idx=scan_idx,
            proj_idx=proj_idx,
            probe=probe,
            mode=mode,
            rhs_block=rhs_block,
        )

    @abstractmethod
    def _solve_single_probe_impl(
        self,
        scan_idx: int = 0,
        proj_idx: int = 0,
        probe: Optional[np.ndarray] = None,
        mode: str = "forward",
        rhs_block: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Override in subclasses."""
        raise NotImplementedError

    def get_gradient(self, n: np.ndarray) -> np.ndarray:
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
            n=n, grad=True
        )
