import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.fft import fft, ifft
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

# ------------------------------------------------------------------
#  Math Helpers (Reused from your code)
# ------------------------------------------------------------------


def _pintobj_matvec_exact(A_csr, B_csr, C, L, v):
    """
    Computes (A_hat @ v) with blocks.
    Preserved exactly from original scipy implementation logic.
    """
    Nx = A_csr.shape[0]

    # IMPORTANT: columns are z-slices => Fortran order
    # Note: v coming from PETSc will be a flat buffer.
    V = np.reshape(v, (Nx, L), order="F")

    # Diagonal blocks: Ai[j] V[:, j] = (A - diag(C[:,j])) V[:,j]
    U = (A_csr @ V) - (C * V)

    # Subdiagonal blocks: j>=1, add -Bi[j] V[:, j-1] = -(B + diag(C[:,j])) V[:, j-1]
    U[:, 1:] += -(B_csr @ V[:, :-1] + C[:, 1:] * V[:, :-1])

    # Flatten back in the SAME ordering
    return np.reshape(U, (Nx * L,), order="F")


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


class PiTPreconditionerShell:
    """
    Context for the PETSc PCShell.
    Implements the alpha-Block circulant PiT preconditioner.
    """

    def __init__(self, A, B, C, alpha, _log=None):
        self.A = A
        self.B = B
        self.C = C
        self.alpha = alpha
        self.Nx = C.shape[0]
        self.L = C.shape[1]

        self._log = _log if _log is not None else print

        # --- Initialization Logic (Moved from setup) ---
        dtype = np.complex128

        # 1. Compute spatial mean C_bar over z/time
        C_avg = np.mean(self.C, axis=1)
        C_bar = sp.diags(C_avg, 0, format="csr")

        # 2. Build averaged blocks
        A_bar = (self.A - C_bar).tocsr()
        B_bar = (self.B + C_bar).tocsr()

        alpha_root = self.alpha ** (1.0 / self.L)
        js = np.arange(self.L)

        # 3. Roots of unity scaled by alpha
        omegas = alpha_root * np.exp(2j * np.pi * js / self.L)

        # 4. Pre-factorize blocks (IMMEDIATELY populates self.lus)
        time_start = time.time()
        self.lus = [
            spla.splu((A_bar - (z * B_bar)).astype(dtype).tocsc()) for z in omegas
        ]
        time_end = time.time()
        self._log(f"PiT Preconditioner setup time: {time_end - time_start:.2f} s")

        # 5. Calculate Gamma
        self.gamma = (self.alpha ** (np.arange(self.L) / self.L)).astype(dtype)

    def apply(self, pc, X, Y):
        """Apply the preconditioner: M^-1 * x"""
        x_arr = X.array_r
        y_arr = Y.array

        dtype = np.complex128

        # 1. Reshape v into L blocks
        V = np.asarray(x_arr, dtype=dtype).reshape(self.Nx, self.L, order="F")

        # 2. Scale by Gamma and IFFT
        V_scaled = V * self.gamma[None, :]
        V_hat = ifft(V_scaled, axis=1, norm="ortho")

        # 3. Block Solve
        X_hat = np.zeros_like(V_hat)
        for j in range(self.L):
            # This will now work because self.lus is populated in __init__
            X_hat[:, j] = self.lus[j].solve(V_hat[:, j])

        # 4. FFT and Inverse Scale
        Y_res = fft(X_hat, axis=1, norm="ortho")
        Y_res = Y_res / self.gamma[None, :]

        # 5. Flatten back
        y_arr[:] = Y_res.reshape((self.Nx * self.L), order="F")


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
    M_shell_ctx: Optional[PiTPreconditionerShell] = None
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
        A_step, B_step, _ = self.pwe_finite_differences._get_or_generate_step_matrices()
        L = self.nz - 1
        A_csr = A_step.tocsr()
        B_csr = B_step.tocsr()

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

        # Create the Preconditioner Shell Context
        M_ctx = PiTPreconditionerShell(A_step, B_step, C, self.alpha, _log=self._log)

        # Store in cache
        cache = self.projection_cache[proj_idx].modes[mode]
        cache.A_shell_ctx = A_ctx
        cache.M_shell_ctx = M_ctx
        cache.cached_n = n

        # Construct RHS (numpy)
        if cache.b is None and self.test_bcs is None:
            cache.b = self.pwe_finite_differences.setup_homogeneous_forward_model_rhs()
        elif self.test_bcs is not None:
            cache.b = self.test_bcs.test_exact_impedance_forward_model_rhs()

    def _solve_single_probe_impl(
        self,
        scan_idx: int = 0,
        proj_idx: int = 0,
        probe: Optional[np.ndarray] = None,
        mode: str = "forward",
        rhs_block: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._log("Setting up PETSc system...")

        cache = self.projection_cache[proj_idx].modes[mode]

        # 1. Prepare RHS
        if rhs_block is not None:
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
        # Assuming sequential run or simple MPI distribution.
        # For advanced MPI usage, one might need `comm=PETSc.COMM_WORLD`
        x_sol = PETSc.Vec().createSeq(N_total, comm=PETSc.COMM_SELF)
        b_vec = PETSc.Vec().createSeq(N_total, comm=PETSc.COMM_SELF)

        # Fill RHS vector
        # Ensure complex128 and alignment
        b_vec.setArray(b_numpy.astype(np.complex128, copy=False))

        # 4. Create PETSc Matrix (Shell)
        A_mat = PETSc.Mat().createPython(
            [N_total, N_total], context=cache.A_shell_ctx, comm=PETSc.COMM_SELF
        )
        A_mat.setUp()

        # 5. Create KSP (Solver)
        ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp.setOperators(A_mat)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Configure GMRES (Restart, etc)
        # ksp.setGMRESRestart(30) # Default is usually 30

        # 6. Create PC (Preconditioner)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(cache.M_shell_ctx)

        # Set Right Preconditioning as in your original code
        ksp.setPCSide(PETSc.PC.Side.RIGHT)

        # Set Tolerances
        ksp.setTolerances(atol=self.atol, rtol=1e-30, max_it=1000)

        # Allow command line overrides (e.g., -ksp_monitor)
        ksp.setFromOptions()

        # 7. Solve
        self._log("Solving with PETSc GMRES...", flush=True)

        # Optional: Custom Monitor for logging
        def monitor(ksp, it, rnorm):
            self._log(f"  Iter {it:3d} | Residual: {rnorm:.3e}")

        ksp.setMonitor(monitor)

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

        # 8. Retrieve Solution
        u_flat = x_sol.array

        # Reshape and concatenate with initial condition (same as original)
        u = u_flat.reshape(self.nz - 1, self.block_size).T
        initial = probe.reshape(self.block_size, 1)

        # Cleanup PETSc objects manually (good practice in loops)
        ksp.destroy()
        A_mat.destroy()
        x_sol.destroy()
        b_vec.destroy()

        return np.concatenate([initial, u], axis=1)
