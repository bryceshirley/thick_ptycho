import time
from dataclasses import dataclass, fields
from typing import Optional


import numpy as np
import scipy.sparse.linalg as spla

from thick_ptycho.forward_model.pwe.operators import BoundaryType
from thick_ptycho.forward_model.pwe.operators.finite_differences.boundary_condition_test import (
    BoundaryConditionsTest,
)
from thick_ptycho.forward_model.pwe.solvers.base_solver import BasePWESolver
from thick_ptycho.forward_model.pwe.solvers.utils._pint_utils import (
    _pintobj_matvec_exact,
    AbstractPiTPreconditioner,
)


class PiTPreconditioner(AbstractPiTPreconditioner):
    """
    Implements the alpha-Block circulant PiT preconditioner.
    """

    def __init__(self, A, B, N, L, alpha, _log=None):
        super().__init__(A, B, N, L, alpha, _log=_log)

    def apply(self, v):
        def apply_prec(v):
            return self._apply_prec(v).reshape(self.L * self.N)

        linear_operator = spla.LinearOperator(
            shape=(self.L * self.N, self.L * self.N),
            dtype=self.dtype,
            matvec=apply_prec,
        )
        return linear_operator.matvec(v)


@dataclass
class PWEFullPinTSolverCache:
    """
    Cache structure for storing precomputed variables.
    Parameters
    ----------
    ARop : LinearOperator, optional
        Right-preconditioned operator A@R.
    Mop : LinearOperator, optional
        PiT preconditioner operator.
    b : ndarray, optional
        Right-hand side vector.
    """

    cached_n: Optional[np.ndarray] = None
    ARop: Optional[spla.LinearOperator] = None
    b: Optional[np.ndarray] = None
    u0_cache: Optional[np.ndarray] = None

    def reset(self, n: Optional[np.ndarray] = None):
        """Reset cached variables."""
        # Reinitialize all cached variables to None
        for f in fields(self):
            setattr(self, f.name, None)
        # Update cached n
        self.cached_n = n


# ------------------------------------------------------------------
#  Full-system PWE Solver with PiT Preconditioning
# ------------------------------------------------------------------
class PWEFullPinTSolver(BasePWESolver):
    """Full-system PWE solver using a single block-tridiagonal system."""

    solver_cache_class = PWEFullPinTSolverCache

    def __init__(
        self,
        simulation_space,
        ptycho_probes,
        bc_type: BoundaryType = BoundaryType.IMPEDANCE,
        results_dir="",
        use_logging=False,
        verbose=False,
        log=None,
        alpha=1e-2,
        atol=1e-8,
        test_bcs: BoundaryConditionsTest = None,
    ):
        super().__init__(
            simulation_space,
            ptycho_probes,
            bc_type=bc_type,
            results_dir=results_dir,
            use_logging=use_logging,
            verbose=verbose,
            log=log,
            test_bcs=test_bcs,
        )
        self.b0 = self.pwe_finite_differences.precompute_b0(self.probes)

        self.atol = atol

        (
            self.A_step,
            self.B_step,
            _,
        ) = self.pwe_finite_differences._get_or_generate_step_matrices()

        self.preconditioner = PiTPreconditioner(
            A=self.A_step,
            B=self.B_step,
            N=self.block_size,
            L=self.nz - 1,
            alpha=alpha,
            _log=self._log,
        )
        self.preconditioner.factorize_blocks()

    # ------------------------------------------------------------------
    # PinT Preconditioner construction
    # ------------------------------------------------------------------
    def _construct_solve_cache(
        self,
        n: Optional[np.ndarray] = None,
        mode: str = "forward",
        scan_idx: Optional[int] = 0,
        proj_idx: Optional[int] = 0,
    ) -> None:
        """
        Compute the PiT preconditioner for the full system.

        Parameters
        ----------
        n : ndarray, optional
            Refractive index field. If None.
        mode : {'forward', 'adjoint', 'reverse'}
            Propagation mode.
        scan_idx : int
            Index of the probe position.
        proj_idx : int
            Index of the projection (for tomographic scans).
        """
        L = self.nz - 1

        # Assemble system matrices
        C = self.simulation_space.create_object_contribution(
            n=n, scan_index=scan_idx
        ).reshape(-1, self.nz - 1)

        # Update PiT preconditioner with current C
        # self.preconditioner.update(C)

        # Adjoint modes (if you compare those): conjugate-transpose blocks + reverse z + conjugate C
        A_csr = self.A_step.tocsr()
        B_csr = self.B_step.tocsr()
        if mode == "adjoint":
            A_csr = A_csr.conj().T
            B_csr = B_csr.conj().T
            C = np.flip(C.conj(), axis=1)

        Aop = spla.LinearOperator(
            (self.block_size * L, self.block_size * L),
            matvec=lambda x: _pintobj_matvec_exact(A_csr, B_csr, C, L, x),
            dtype=np.complex128,
        )

        def _matvec_A_right(y):
            return Aop.matvec(self.preconditioner.apply(y))

        # Right-preconditioned operator A@R
        ARop = spla.LinearOperator(
            Aop.shape, dtype=np.complex128, matvec=_matvec_A_right
        )

        # Cache A, R, and A@R
        self.projection_cache[proj_idx].modes[mode].ARop = ARop
        self.projection_cache[proj_idx].modes[mode].cached_n = n

        # Construct RHS if needed (mirror LU b logic)
        if (
            self.projection_cache[proj_idx].modes[mode].b is None
            and self.test_bcs is None
        ):
            self.projection_cache[proj_idx].modes[
                mode
            ].b = self.pwe_finite_differences.setup_homogeneous_forward_model_rhs()
        elif self.test_bcs is not None:
            self.projection_cache[proj_idx].modes[
                mode
            ].b = self.test_bcs.test_exact_impedance_forward_model_rhs()

    # -------------------------------------------------------------------------
    # Main probe solve
    # -------------------------------------------------------------------------
    def _solve_single_probe_impl(
        self,
        scan_idx: int = 0,
        proj_idx: int = 0,
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

        # Solve the global system
        self._log("Retrieving PiT preconditioner and setting up system...")

        # Build RHS
        if rhs_block is not None:
            b = rhs_block
        else:
            probe_contribution = self.pwe_finite_differences.probe_contribution(
                scan_index=scan_idx, probe=probe
            )
            b = self.projection_cache[proj_idx].modes[mode].b + probe_contribution

        ARop = self.projection_cache[proj_idx].modes[mode].ARop
        self._log("Solving with PiT-preconditioned GMRES...", flush=True)

        residuals = []

        def gmres_callback(rn):
            residuals.append(rn)
            self._log(f"  Iter {len(residuals):3d} | Precond residual: {rn:.3e}")

        # Right-preconditioned GMRES solve
        t0 = time.perf_counter()
        y, info = spla.gmres(
            ARop,
            b.astype(np.complex128, copy=False),
            atol=self.atol,
            x0=self.projection_cache[proj_idx].modes[mode].u0_cache,
            callback=gmres_callback,
            callback_type="pr_norm",
        )
        t1 = time.perf_counter()
        # Recover solution in original variables: x = R y
        u = self.preconditioner.apply(y)
        self.projection_cache[proj_idx].modes[mode].u0_cache = y

        self._log(f"Time with PiT preconditioner: {t1 - t0:.2f} seconds.", flush=True)
        if info == 0:
            self._log(f"GMRES converged in {len(residuals)} iterations.\n", flush=True)
        else:
            self._log(
                f"GMRES stopped early (info={info}) after {len(residuals)} iterations.\n",
                flush=True,
            )

        # Reshape and concatenate with initial condition
        u = u.reshape(self.nz - 1, self.block_size).T
        initial = probe.reshape(self.block_size, 1)
        return np.concatenate([initial, u], axis=1)
