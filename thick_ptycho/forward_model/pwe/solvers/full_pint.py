import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.fft import fft, ifft
from scipy.sparse.linalg import LinearOperator, splu

from thick_ptycho.forward_model.pwe.operators import BoundaryType
from thick_ptycho.forward_model.pwe.operators.finite_differences.boundary_condition_test import (
    BoundaryConditionsTest,
)
from thick_ptycho.forward_model.pwe.solvers.base_solver import BasePWESolver
from thick_ptycho.forward_model.pwe.solvers.utils._pint_utils import (
    _init_worker,
    _pintobj_matvec_exact,
    _solve_block,
)

mp.set_start_method("spawn", force=True)


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
    ARop: Optional[LinearOperator] = None
    Mop: Optional[LinearOperator] = None
    b: Optional[np.ndarray] = None

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
        num_workers=8,
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

        # Get number of workers for PiT preconditioner
        # based on available CPU cores
        self._log(f"Available CPU cores: {os.cpu_count()}")
        self.num_workers = min(num_workers, os.cpu_count())
        self._log(f"Using {self.num_workers} workers for PiT preconditioner.")

        self.atol = atol

        # PiT preconditioner parameter
        self.alpha = alpha

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
        # A_step, B_step, _ = self.pwe_finite_differences.generate_zstep_matrices()
        A_step, B_step, _ = self.pwe_finite_differences._get_or_generate_step_matrices()
        L = self.nz - 1

        # Convert to CSR now for fast SpMV
        A_csr = A_step.tocsr()
        B_csr = B_step.tocsr()

        C = self.simulation_space.create_object_contribution(
            n=n, scan_index=scan_idx
        ).reshape(-1, self.nz - 1)

        # M_prec = self._make_pit_preconditioner(A_step, B_step, C)
        M_prec = self._make_pit_preconditioner_multi_workers(A_step, B_step, C)

        # Adjoint modes (if you compare those): conjugate-transpose blocks + reverse z + conjugate C
        if mode in {"adjoint", "adjoint_rotated"}:
            A_csr = A_csr.conj().T
            B_csr = B_csr.conj().T
            C = np.flip(C.conj(), axis=1)

        Aop = spla.LinearOperator(
            (self.block_size * L, self.block_size * L),
            matvec=lambda x: _pintobj_matvec_exact(A_csr, B_csr, C, L, x),
            dtype=np.complex128,
        )

        def _matvec_A_right(y):
            return Aop.matvec(M_prec.matvec(y))

        # Right-preconditioned operator A@R
        ARop = spla.LinearOperator(
            Aop.shape, dtype=np.complex128, matvec=_matvec_A_right
        )

        # Cache A, R, and A@R
        self.projection_cache[proj_idx].modes[mode].ARop = ARop
        self.projection_cache[proj_idx].modes[mode].M_prec = M_prec
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
        time_start = time.time()
        # Build RHS
        if rhs_block is not None:
            b = rhs_block
        else:
            probe_contribution = self.pwe_finite_differences.probe_contribution(
                scan_index=scan_idx, probe=probe
            )
            b = self.projection_cache[proj_idx].modes[mode].b + probe_contribution

        ARop = self.projection_cache[proj_idx].modes[mode].ARop
        M_prec = self.projection_cache[proj_idx].modes[mode].M_prec

        time_end = time.time()
        self._log(
            f"PiT preconditioner retrieval and setup time: {time_end - time_start:.2f} seconds.\n"
        )

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
            callback=gmres_callback,
            callback_type="pr_norm",
        )
        t1 = time.perf_counter()
        # Recover solution in original variables: x = R y
        u = M_prec @ y

        # A = self.pwe_finite_differences.return_forward_model_matrix(n=n)

        # t0 = time.perf_counter()
        # u, info = spla.gmres(
        #     A,
        #     b.astype(np.complex128, copy=False),
        #     M=M_prec,
        #     atol=self.atol,
        #     callback=gmres_callback,
        #     callback_type='pr_norm'
        # )
        # t1 = time.perf_counter()

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

    def _make_pit_preconditioner(self, A, B, C):
        """
        alpha-Block circulant PiT preconditioner with element-wise spatial averaging of C.
        """
        dtype = np.complex128
        alpha = self.alpha
        Nx, L = C.shape
        alpha_root = alpha ** (1.0 / L)

        # --- Compute spatial mean C_bar over z/time (Nx-vector) ---
        C_avg = np.mean(C, axis=1)  # (Nx,)
        C_bar = sp.diags(C_avg, 0, format="csr")  # diag(C_bar)

        # --- Build averaged blocks ---
        A = (A - C_bar).tocsr()
        B = (B + C_bar).tocsr()

        N = A.shape[0]
        alpha_root = alpha ** (1.0 / L)

        js = np.arange(L)
        omegas = np.exp(2j * np.pi * js / L) * alpha_root

        # Pre-factorize blocks (A - ζ_j B)
        lus = [splu((A - (z * B)).astype(dtype)) for z in omegas]

        gamma = (alpha ** (np.arange(L) / L)).astype(dtype)

        def apply_prec(v):
            # reshape (L,N)
            X = np.asarray(v, dtype=dtype).reshape(L, N)

            X = gamma[:, None] * X
            X_hat = ifft(X, axis=0, norm="ortho")

            # Solve each block
            for j in range(L):
                X_hat[j, :] = lus[j].solve(X_hat[j, :])

            Y = fft(X_hat, axis=0, norm="ortho")
            Y = Y / gamma[:, None]

            return Y.reshape(L * N)

        return LinearOperator(shape=(L * N, L * N), dtype=dtype, matvec=apply_prec)

    def _make_pit_preconditioner_multi_workers(self, A, B, C):
        """
        alpha-Block circulant PiT preconditioner with element-wise spatial averaging of C.
        Multi-worker version using ProcessPoolExecutor.
        """
        dtype = np.complex128
        alpha = self.alpha
        Nx, L = C.shape

        # --- Compute spatial mean C_bar over z/time (Nx-vector) ---
        C_avg = np.mean(C, axis=1)  # (Nx,)
        C_bar = sp.diags(C_avg, 0, format="csr")  # diag(C_bar)

        # --- Build averaged blocks ---
        A_bar = (A - C_bar).tocsr()
        B_bar = (B + C_bar).tocsr()

        alpha_root = alpha ** (1.0 / L)
        js = np.arange(L)
        omegas = alpha_root * np.exp(2j * np.pi * js / L)

        gamma = (alpha ** (np.arange(L) / L)).astype(dtype)

        # --- Prepare worker pool ---
        executor = ProcessPoolExecutor(max_workers=self.num_workers)
        executor._initializer = _init_worker
        executor._initargs = (A_bar, B_bar, omegas)
        self._log(f"Creating the Pit preconditioner with {self.num_workers} workers.")

        def apply_prec(v):
            # Reshape v into L blocks of size Nx
            V = np.asarray(v, dtype=dtype).reshape(L, Nx).T  # (Nx, L)

            V_hat = np.fft.ifft(gamma[None, :] * V, axis=1, norm="ortho")

            # Solve each block in parallel
            X_hat = list(
                executor.map(_solve_block, ((j, V_hat[:, j]) for j in range(L)))
            )
            X_hat = np.column_stack(X_hat)  # (Nx, L)

            Y = np.fft.fft(X_hat, axis=1, norm="ortho") / gamma[None, :]

            return Y.T.reshape(L * Nx)

        return spla.LinearOperator((Nx * L, Nx * L), dtype=dtype, matvec=apply_prec)
