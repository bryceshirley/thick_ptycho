import time
import numpy as np
from dataclasses import dataclass, fields
from typing import Optional

# PETSc imports
try:
    import petsc4py

    petsc4py.init([])
    from petsc4py import PETSc
except ImportError:
    print("PETSc4Py is not installed. PWEPetscFullPinTSolver will not work without it.")

from thick_ptycho.forward_model.pwe.operators import BoundaryType
from thick_ptycho.forward_model.pwe.operators.finite_differences.boundary_condition_test import (
    BoundaryConditionsTest,
)
from thick_ptycho.forward_model.pwe.solvers.base_solver import BasePWESolver

from thick_ptycho.forward_model.pwe.solvers.utils._pint_utils import (
    _pintobj_matvec_exact,
    AbstractPiTPreconditioner,
)

# ------------------------------------------------------------------
#  PETSc Shell Contexts
# ------------------------------------------------------------------


class PWEGlobalOperatorShell:
    """
    Context for the PETSc MatShell.
    Wraps the matrix-vector multiplication of the full space-time system.
    """

    def __init__(self, A_csr, B_csr, C, L):
        self.A_csr = A_csr
        self.B_csr = B_csr
        self.C = C
        self.L = L
        self.Nx = A_csr.shape[0]

    def mult(self, mat, X, Y):
        # Access raw arrays from PETSc vectors
        # Note: We use read-only for X, write-only for Y
        x_arr = X.array_r
        y_arr = Y.array

        # Call the numpy/scipy logic
        res = _pintobj_matvec_exact(self.A_csr, self.B_csr, self.C, self.L, x_arr)

        # Copy result back to PETSc vector
        y_arr[:] = res


class PiTPreconditionerShell(AbstractPiTPreconditioner):
    """
    Context for the PETSc PCShell.
    Implements the alpha-Block circulant PiT preconditioner.
    """

    def __init__(self, A, B, N, L, alpha, _log=None):
        super().__init__(A, B, N, L, alpha, _log=_log)

    # In PiTPreconditionerShell.apply:
    def apply(self, pc, X, Y):
        x_arr = X.array_r
        y_arr = Y.array

        # Convert x_arr to c++ array
        v_c_order = np.asarray(x_arr, dtype=self.dtype).copy(order="C")

        # 2. Apply preconditioner using the C-ordered data
        Y_res = self._apply_prec(v_c_order)

        # 3. Flatten back (Y_res is (L, N) C-order -> F-order vector)
        y_arr[:] = Y_res.T.reshape(self.L * self.N, order="F")


# ------------------------------------------------------------------
#  Main Solver Class
# ------------------------------------------------------------------


@dataclass
class PWEFullPinTSolverCache:
    """
    Cache structure for storing precomputed PETSc objects.
    """

    cached_n: Optional[np.ndarray] = None
    # Store PETSc Mat and PC contexts/shells
    A_shell_ctx: Optional[PWEGlobalOperatorShell] = None
    u0_cache: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None

    def reset(self, n: Optional[np.ndarray] = None):
        for f in fields(self):
            setattr(self, f.name, None)
        self.cached_n = n


class PWEPetscFullPinTSolver(BasePWESolver):
    """Full-system PWE solver using PETSc GMRES with Python Shell Preconditioning."""

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
        self.alpha = alpha

        # Verify complex numbers support
        scalar_type = PETSc.ScalarType
        if not np.issubdtype(scalar_type, np.complexfloating):
            self._log(
                "WARNING: PETSc is not configured with complex scalars. This solver requires complex numbers."
            )

        (
            self.A_step,
            self.B_step,
            _,
        ) = self.pwe_finite_differences._get_or_generate_step_matrices()

        self.preconditioner = {"forward": None, "adjoint": None}
        self.preconditioner["forward"] = PiTPreconditionerShell(
            A=self.A_step,
            B=self.B_step,
            N=self.block_size,
            L=self.nz - 1,
            alpha=alpha,
            _log=self._log,
        )
        self.preconditioner["adjoint"] = PiTPreconditionerShell(
            A=self.A_step.conj().T,
            B=self.B_step.conj().T,
            N=self.block_size,
            L=self.nz - 1,
            alpha=alpha,
            _log=self._log,
        )

    def _construct_solve_cache(
        self,
        n: Optional[np.ndarray] = None,
        mode: str = "forward",
        scan_idx: Optional[int] = 0,
        proj_idx: Optional[int] = 0,
    ) -> None:
        """
        Prepare the PETSc Shell Contexts.
        """

        L = self.nz - 1
        A_csr = self.A_step.tocsr()
        B_csr = self.B_step.tocsr()

        C = self.simulation_space.create_object_contribution(
            n=n, scan_index=scan_idx
        ).reshape(-1, self.nz - 1)

        # Adjoint logic
        if mode in {"adjoint", "adjoint_rotated"}:
            A_csr = A_csr.conj().T
            B_csr = B_csr.conj().T
            C = np.flip(C.conj(), axis=1)

        # Create the Operator Shell Context
        A_ctx = PWEGlobalOperatorShell(A_csr, B_csr, C, L)

        # Store in cache
        self.projection_cache[proj_idx].modes[mode].A_shell_ctx = A_ctx

        # Construct RHS (numpy)
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

    def _solve_single_probe_impl(
        self,
        scan_idx: int = 0,
        proj_idx: int = 0,
        probe: Optional[np.ndarray] = None,
        mode: str = "forward",
        rhs_block: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._log("Setting up PETSc system...")
        time_start_setup = time.perf_counter()
        cache = self.projection_cache[proj_idx].modes[mode]

        if probe is None:
            probe = np.zeros((self.block_size,), dtype=complex)

        # 1. Prepare RHS
        if rhs_block is not None:
            if mode == "adjoint":
                b_numpy = np.flip(rhs_block, axis=1)
            else:
                # If not adjoint, use the block as is.
                b_numpy = rhs_block
        else:
            probe_contribution = self.pwe_finite_differences.probe_contribution(
                scan_index=scan_idx, probe=probe
            )
            b_numpy = cache.b + probe_contribution

        # 2. Setup PETSc Dimensions
        Nx = cache.A_shell_ctx.Nx
        L = cache.A_shell_ctx.L
        N_total = Nx * L

        # 3. Create PETSc Vectors
        x_sol = PETSc.Vec().createSeq(N_total, comm=PETSc.COMM_SELF)
        b_vec = PETSc.Vec().createSeq(N_total, comm=PETSc.COMM_SELF)
        b_vec.setArray(b_numpy.astype(np.complex128, copy=False))

        # --- INITIAL GUESS ---
        if cache.u0_cache is not None:
            x_sol.setArray(cache.u0_cache.astype(np.complex128, copy=False))
            x0_is_nonzero = True
        else:
            x_sol.set(0)
            x0_is_nonzero = False

        # 4. Create PETSc Matrix (Shell)
        A_mat = PETSc.Mat().createPython(
            [N_total, N_total], context=cache.A_shell_ctx, comm=PETSc.COMM_SELF
        )
        A_mat.setUp()

        # 5. Create KSP (Solver)
        ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp.setOperators(A_mat)
        ksp.setType(PETSc.KSP.Type.GMRES)
        ksp.setInitialGuessNonzero(x0_is_nonzero)

        # 6. Create PC (Preconditioner)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        # ksp.setType(PETSc.KSP.Type.GMRES)

        pc.setPythonContext(self.preconditioner[mode])
        pc.setReusePreconditioner(True)

        ksp.setPCSide(PETSc.PC.Side.RIGHT)
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)

        # Set Tolerances
        ksp.setConvergenceHistory(True)
        ksp.setTolerances(atol=self.atol, rtol=1e-30, max_it=20)
        ksp.setFromOptions()
        time_end_setup = time.perf_counter()
        self._log(
            f"PETSc system setup time: {time_end_setup - time_start_setup:.2f} seconds."
        )

        # 7. Solve
        self._log("Solving with PETSc GMRES...", flush=True)

        # Optional: Custom Monitor for logging
        def monitor_standard(ksp, it, rnorm):
            # rnorm is the *currently tracked norm* (NORM_NONE, Unpreconditioned)
            self._log(f"  Iter {it:3d} | Residual: {rnorm:.3e}")

        ksp.setMonitor(monitor_standard)

        t0 = time.perf_counter()
        ksp.solve(b_vec, x_sol)
        t1 = time.perf_counter()
        self._log(f"PETSc GMRES solve time: {t1 - t0:.2f} s")

        # Check convergence
        iters = ksp.getIterationNumber()
        reason = ksp.getConvergedReason()
        self._log(
            f"Time: {t1 - t0:.2f} s. Iters: {iters}. Reason: {reason}", flush=True
        )

        # 9. Retrieve Solution
        u_flat = x_sol.array  # Auxiliary solution y

        # Cache u_flat_y (the auxiliary solution y) for the next KSP solve as x0
        self.projection_cache[proj_idx].modes[mode].u0_cache = u_flat

        # Reshape and concatenate with initial condition
        u = u_flat.reshape(self.nz - 1, self.block_size).T
        initial = probe.reshape(self.block_size, 1)

        # Cleanup PETSc objects manually (good practice in loops)
        ksp.destroy()
        A_mat.destroy()
        x_sol.destroy()
        b_vec.destroy()

        u = np.concatenate([initial, u], axis=1)

        if mode == "adjoint":
            u = np.flip(u, axis=1)

        return u
