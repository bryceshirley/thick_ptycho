import time
from typing import Optional

import numpy as np
import scipy.sparse as sp
# PETSc / petsc4py
from petsc4py import PETSc
from scipy.sparse.linalg import splu

from thick_ptycho.forward_model.pwe.solvers.base_solver import BasePWESolver
from thick_ptycho.forward_model.pwe.utils._pint_utils import \
    _pintobj_matvec_exact


# ---------------------------
# Helper: PETSc MatShell for A
# ---------------------------
class _AOperatorShell:
    __slots__ = ("A_csr", "B_csr", "C", "L", "N", "dtype")

    def __init__(self, A_csr, B_csr, C, L, N, dtype=np.complex128):
        self.A_csr = A_csr
        self.B_csr = B_csr
        self.C = C
        self.L = L
        self.N = N
        self.dtype = dtype


    def mult(self, A: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec):
        # x, y are PETSc vectors of length L*N
        x_arr = x.getArray(readonly=True)
        out = _pintobj_matvec_exact(self.A_csr, self.B_csr, self.C, self.L, x_arr)
        y_arr = y.getArray()
        y_arr[...] = out
        y.restoreArray()
        x.restoreArray()
    # def mult(self, A: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec):
    #     # Use your exact block-structure matvec with Fortran ordering:
    #     x_arr = x.getArray(readonly=True)
    #     out = _pintobj_matvec_exact(
    #         self.A_csr,
    #         self.B_csr,
    #         self.C,
    #         self.L,
    #         x_arr
    #     )
    #     y_arr = y.getArray()
    #     y_arr[...] = out
    #     y.restoreArray()
    #     x.restoreArray()


# -------------------------------------------
# Helper: PETSc PCShell for PiT preconditioner
# -------------------------------------------
class _PiTPreconditioner:
    """PiT right preconditioner R for alpha-block circulant scheme.

    Implements y <- R x.

    Notes
    -----
    This version uses an *averaged* object contribution C_bar over z, building
    A_bar = A - diag(C_bar), B_bar = B + diag(C_bar). We do the FFT-based
    block-diagonalization in Python/NumPy and solve each (A_bar - z B_bar)
    system via sparse LU factors precomputed once at construction.

    The PC is intended to be used as a **right** preconditioner with KSP GMRES
    (i.e., A*R is the operator actually iterated on), and the physical field is
    recovered as x = R y after the KSP solve.
    """

    def __init__(self, A_step: sp.csr_matrix, B_step: sp.csr_matrix, C: np.ndarray, alpha: float):
        assert sp.isspmatrix(A_step) and sp.isspmatrix(B_step)
        self.dtype = np.complex128
        self.alpha = float(alpha)

        # Shapes
        self.N = A_step.shape[0]
        self.L = int(C.shape[1])

        # Average C over z and build A_bar, B_bar
        C_avg = np.mean(C, axis=1)  # (N,)
        C_bar = sp.diags(C_avg, 0, format='csr')

        self.A_bar = (A_step - C_bar).tocsr().astype(self.dtype)
        self.B_bar = (B_step + C_bar).tocsr().astype(self.dtype)

        # Frequencies and gamma scalings
        js = np.arange(self.L)
        alpha_root = self.alpha ** (1.0 / self.L)
        self.omegas = alpha_root * np.exp(2j * np.pi * js / self.L)
        self.gamma = (self.alpha ** (js / self.L)).astype(self.dtype)

        # Pre-factorize blocks: LU of (A_bar - z B_bar)
        # We keep SciPy LUs but could be replaced by PETSc factorizations if matrices are large
        self._lus = [splu((self.A_bar - (z * self.B_bar)).astype(self.dtype)) for z in self.omegas]

    # ---- PETSc PCShell hooks ----
    def setUp(self, pc: PETSc.PC):  # optional hook
        return

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec):
        """y <- R x (right preconditioner application)."""
        xin = x.getArray(readonly=True)  # (L*N,)
        out = self._apply_numpy(xin)
        y_arr = y.getArray()
        y_arr[...] = out
        y.restoreArray()
        x.restoreArray()

    # ---- Convenience: numpy application for post-solve recovery ----
    def apply_numpy(self, x_np: np.ndarray) -> np.ndarray:
        return self._apply_numpy(x_np)

    def _apply_numpy(self, v: np.ndarray) -> np.ndarray:
        N, L = self.N, self.L
        V = np.asarray(v, dtype=self.dtype).reshape(L, N)  # (L, N)
        # Move to (N, L) for axis-1 FFTs with gamma scaling
        V = V.T  # (N, L)
        V_hat = np.fft.ifft(self.gamma[None, :] * V, axis=1, norm='ortho')
        # Solve each block system
        for j in range(L):
            V_hat[:, j] = self._lus[j].solve(V_hat[:, j])
        Y = np.fft.fft(V_hat, axis=1, norm='ortho') / self.gamma[None, :]
        # Back to (L, N) then ravel
        Y = Y.T
        return Y.reshape(L * N)


class PWEFullPinTSolverPETSc(BasePWESolver):
    """Full-system PWE solver using PETSc GMRES + PiT right preconditioner.

    This is a petsc4py-based rewrite of the SciPy GMRES version. The global
    block-tridiagonal system is applied through a PETSc MatShell, while the
    PiT preconditioner is provided via a PETSc PCShell used on the **right**.

    Key differences vs SciPy version
    --------------------------------
    - Uses PETSc's KSP (GMRES) for the Krylov solve.
    - Right preconditioning with a PCShell that implements the alpha-block
      circulant PiT operator R. Solution is recovered as x = R y.
    - Still relies on SciPy sparse LUs inside the preconditioner for each
      block (A_bar - z B_bar). For very large problems, these could be
      replaced with PETSc factorizations or iterative inner solves.
    """

    def __init__(self, simulation_space, ptycho_probes,
                 results_dir="", use_logging=False, verbose=False, log=None,
                 alpha: float = 1e-6, atol: float = 1e-6):
        super().__init__(
            simulation_space,
            ptycho_probes,
            results_dir=results_dir,
            use_logging=use_logging,
            verbose=verbose,
            log=log,
        )

        # Cache for PiT components per projection (0: nominal, 1: rotated)
        self.pit_cache = {"projection_0": None, "projection_1": None}
        self._cached_n_id = None
        self.b_cache = None
        self.alpha = float(alpha)

        # Precompute B0 if applicable
        self.pwe_finite_differences.full_system = True
        self.b0 = self.pwe_finite_differences.precompute_b0(self.probes)

        self.solver_type = "PETSc Block PWE Full Solver"
        self.atol = float(atol)

    # ------------------------------
    # Preconditioner / operator prep
    # ------------------------------
    def _get_or_construct_pit(self, n: Optional[np.ndarray] = None, mode: str = "forward"):
        assert mode in {"forward", "adjoint", "forward_rotated", "adjoint_rotated"}
        if n is None:
            n = self.simulation_space.refractive_index_empty

        if mode in {"forward_rotated", "adjoint_rotated"}:
            n = self.rotate_n(n)
            projection_key = "projection_1"
        else:
            projection_key = "projection_0"

        # Reset cache if the object ndarray changed
        n_id = id(n)
        if self._cached_n_id != n_id:
            self.pit_cache = {"projection_0": None, "projection_1": None}
            self._cached_n_id = n_id
        print(f"Preparing PiT components for mode '{mode}'...", flush=True)
        if self.pit_cache[projection_key] is None:
            print("Constructing PiT operator components...", flush=True)
            A_step, B_step, _ = self.pwe_finite_differences.generate_zstep_matrices()
            Nx = self.block_size
            L = self.nz - 1

            # Object contribution (Nx, L)
            C = self.simulation_space.create_object_contribution(n=n).reshape(-1, L).astype(np.complex128)

            A_csr = A_step.tocsr()  # ensure A_step is already complex128 upstream
            B_csr = B_step.tocsr()
            C_eff = C.reshape(-1, L)  # C should also already be complex128


            # Adjoint handling: conjugate-transpose blocks + reverse z + conjugate C
            if mode in {"adjoint", "adjoint_rotated"}:
                A_eff = A_csr.conj().T
                B_eff = B_csr.conj().T
                C_eff = np.flip(C.conj(), axis=1).reshape(-1, L) 
            else:
                A_eff = A_csr
                B_eff = B_csr
                C_eff = C.reshape(-1, L)  # C should also already be complex128


            # Build MatShell for A and PCShell for R
            size = Nx * L
            A_shell = PETSc.Mat().create()
            A_shell.setSizes([[size, size], [size, size]])
            A_shell.setType(PETSc.Mat.Type.SHELL)

            print("Setting up PETSc MatShell for A...", flush=True)
            ctx = _AOperatorShell(A_eff, B_eff, np.asfortranarray(C_eff), L=L, N=Nx)
            print("Binding PETSc MatShell context...", flush=True)
            A_shell = PETSc.Mat().createPython([size, size], context=ctx, comm=PETSc.COMM_SELF)
            A_shell.setUp()
            print("MatShell ready.", flush=True)

            # Right preconditioner context R
            print("Setting up PETSc PCShell for PiT preconditioner...", flush=True)
            pc_ctx = _PiTPreconditioner(A_step=A_csr, B_step=B_csr, C=C, alpha=self.alpha)

            self.pit_cache[projection_key] = (A_shell, pc_ctx)

        if self.b_cache is None:
            self.b_cache = self.pwe_finite_differences.setup_homogeneous_forward_model_rhs()

        return self.pit_cache, self.b_cache

    def prepare_solver(self, n: Optional[np.ndarray] = None, mode: str = "forward"):
        #assert not getattr(self, "solve_reduced_domain", False), "Full-system solver does not support thin-sample approximation."
        assert mode in {"forward", "adjoint", "forward_rotated", "adjoint_rotated"}
        return self._get_or_construct_pit(n=n, mode=mode)

    # ------------------
    # Single-probe solve
    # ------------------
    def _solve_single_probe(
        self,
        scan_idx: int = 0,
        probe: Optional[np.ndarray] = None,
        n: Optional[np.ndarray] = None,
        mode: str = "forward",
        rhs_block: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve for a single probe field with PETSc GMRES + right PC.

        Parameters
        ----------
        scan_idx : int
            Probe position index.
        probe : ndarray, optional
            Initial probe (complex) of shape (block_size,).
        n : ndarray, optional
            Refractive index field.
        mode : {"forward","adjoint","forward_rotated","adjoint_rotated"}
            Propagation mode.
        rhs_block : ndarray, optional
            Optional RHS to reuse across solves.

        Returns
        -------
        u : ndarray, shape (block_size, nz)
            Complex propagated field including initial slice.
        """
        assert mode in {"forward", "adjoint", "forward_rotated", "adjoint_rotated"}

        print("Setting up PETSc operator and PiT preconditioner...", flush=True)
        t_setup0 = time.perf_counter()
        pit_cache, b_hom = self._get_or_construct_pit(n=n, mode=mode)

        projection_key = "projection_1" if mode in {"forward_rotated", "adjoint_rotated"} else "projection_0"
        A_shell, pc_ctx = pit_cache[projection_key]

        if rhs_block is not None:
            b_np = rhs_block
        else:
            probe_contrib = self.pwe_finite_differences.probe_contribution(scan_index=scan_idx, probe=probe)
            b_np = b_hom + probe_contrib

        b_np = b_np.astype(np.complex128)
        size = b_np.size
        t_setup1 = time.perf_counter()
        print(f"Setup time: {t_setup1 - t_setup0:.2f}s", flush=True)

        # PETSc vectors
        b = PETSc.Vec().createSeq(size, comm=PETSc.COMM_SELF)
        b.setArray(np.asarray(b_np, dtype=np.complex128))
        y = PETSc.Vec().createSeq(size, comm=PETSc.COMM_SELF)
        y.set(0)

        # KSP GMRES with right preconditioning
        ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp.setOperators(A_shell)
        ksp.setType(PETSc.KSP.Type.GMRES)
        ksp.setTolerances(atol=self.atol)
        ksp.setPCSide(PETSc.KSP.PCSide.RIGHT)

        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(pc_ctx)  # binds PCShell.apply

        def monitor(ksp_obj, its, rnorm):
            print(f"  Iter {its:3d} | Precond residual: {rnorm:.3e}", flush=True)
        ksp.setMonitor(monitor)

        t0 = time.perf_counter()
        ksp.solve(b, y)
        t1 = time.perf_counter()

        its = ksp.getIterationNumber()
        reason = ksp.getConvergedReason()
        print(f"PETSc GMRES time: {t1 - t0:.2f}s | iters={its} | reason={reason}", flush=True)

        # Recover original variable: x = R y
        y_np = y.getArray(readonly=True).copy()
        x_np = pc_ctx.apply_numpy(y_np)

        # Reshape and concatenate initial condition
        u = x_np.reshape(self.nz - 1, self.block_size).T
        initial = probe.reshape(self.block_size, 1)
        return np.concatenate([initial, u], axis=1)
