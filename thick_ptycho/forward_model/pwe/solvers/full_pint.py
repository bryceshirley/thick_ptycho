
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional, Tuple
from scipy.fft import fft, ifft
import time
from concurrent.futures import ProcessPoolExecutor
from scipy.sparse.linalg import splu, LinearOperator

from thick_ptycho.forward_model.pwe.solvers.base_solver import BasePWESolver
from thick_ptycho.forward_model.pwe.utils._pint_utils import _init_worker, _solve_block, _pintobj_matvec_exact


class PWEFullPinTSolver(BasePWESolver):
    """Full-system PWE solver using a single block-tridiagonal system."""

    def __init__(self, simulation_space, ptycho_object, ptycho_probes,
                 results_dir="", use_logging=False, verbose=False, log=None,
                alpha=1e-2, num_workers=8, atol=1e-6):
        super().__init__(
            simulation_space,
            ptycho_object,
            ptycho_probes,
            results_dir=results_dir,
            use_logging=use_logging,
            verbose=verbose,
            log=log,
        )

        # Cache for PiT preconditioner components
        self.pit_cache = {
            "projection_0": None,
            "projection_1": None,  # Rotated
        }
        self._cached_n_id = None
        self.b_cache = None
        self.alpha = alpha
        # Precompute B0 term if applicable
        self.pwe_finite_differences.full_system = True
        self.b0 = self.pwe_finite_differences.precompute_b0(self.probes)

        # Solver type (for logging purposes)
        self.solver_type = "Block PWE Full Solver"


        # Get number of workers for PiT preconditioner
        # based on available CPU cores
        print(f"Available CPU cores: {os.cpu_count()}")
        self.num_workers = min(num_workers, os.cpu_count())
        print(f"Using {self.num_workers} workers for PiT preconditioner.")

        self.atol = atol
    
    def _get_or_construct_pit(self, n: Optional[np.ndarray] = None, mode: str = "forward"):
        """
        Retrieve or construct the PiT preconditioner for the given mode.
        Caches FFT frequencies, (A_step, B_step), and LinearOperator M_prec.
        """
        assert mode in {"forward", "adjoint", "forward_rotated", "adjoint_rotated"}

        if n is None:
            n = self.ptycho_object.n_true

        # Determine projection key and possibly rotate n
        if mode in {"forward_rotated", "adjoint_rotated"}:
            n = self.rotate_n(n)
            projection_key = "projection_1"
        else:
            projection_key = "projection_0"

        # If n changed, reset PiT cache
        n_id = id(n)
        if self._cached_n_id != n_id:
            self.pit_cache = {"projection_0": None, "projection_1": None}
            self._cached_n_id = n_id

        # Build preconditioner if missing
        if self.pit_cache[projection_key] is None:
            A_step, B_step, _ = self.pwe_finite_differences.generate_zstep_matrices()
            # Object contribution, shape (Nx, L)
            Nx = self.block_size
            L  = self.nz - 1
            
            # Convert to CSR now for fast SpMV
            A_csr = A_step.tocsr()
            B_csr = B_step.tocsr()
            
            C = self.ptycho_object.create_object_contribution(n=n).reshape(-1, self.nz - 1)

            #M_prec = self._make_pit_preconditioner(A_step, B_step, [sp.diags(d) for d in C_diags])
            M_prec = self._make_pit_preconditioner_multi_workers(A_step, B_step, C)

            # Adjoint modes (if you compare those): conjugate-transpose blocks + reverse z + conjugate C
            if mode in {"adjoint", "adjoint_rotated"}:
                A_csr = A_csr.conj().T
                B_csr = B_csr.conj().T
                C     = np.flip(C.conj(), axis=1)

            Aop = spla.LinearOperator(
                (self.block_size*L, self.block_size*L),
                matvec=lambda x: _pintobj_matvec_exact(A_csr, B_csr, C, L, x),
                dtype=np.complex128,
            )

            def _matvec_A_right(y):
                # y  →  R y  →  A (R y)
                return Aop.matvec(M_prec.matvec(y))

            ARop = spla.LinearOperator(Aop.shape, dtype=np.complex128, matvec=_matvec_A_right)

            # Cache A, R, and A@R
            self.pit_cache[projection_key] = (Aop, M_prec, ARop)

        # Construct RHS if needed (mirror LU b_cache logic)
        if self.b_cache is None:
            self.b_cache = self.pwe_finite_differences.setup_homogeneous_forward_model_rhs()

        return self.pit_cache, self.b_cache


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
        assert not getattr(self, "thin_sample", False), \
            "Full-system solver does not support thin-sample approximation."
        assert mode in {"forward", "adjoint","forward_rotated","adjoint_rotated"}, f"Invalid mode: {mode!r}"
    
        return self._get_or_construct_pit(n=n, mode=mode)

        
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


        # Solve the global system
        self._log("Retrieving PiT preconditioner and setting up system...")
        time_start = time.time()
        pit_cache, b_homogeneous = self._get_or_construct_pit(n=n, mode=mode)

        if rhs_block is not None:
            b = rhs_block
        else:
            probe_contribution = self.pwe_finite_differences.probe_contribution(
                scan_index=scan_idx,
                probe=probe
            )
            b = b_homogeneous + probe_contribution

        if mode in {"forward_rotated", "adjoint_rotated"}:
            projection_key = "projection_1"
        else:
            projection_key = "projection_0"

        (Aop, M_prec, ARop) = pit_cache[projection_key]

        time_end = time.time()
        self._log(f"PiT preconditioner retrieval and setup time: {time_end - time_start:.2f} seconds.\n")


        print("Solving with PiT-preconditioned GMRES...",flush=True)

        residuals = []
        def gmres_callback(rn):
            residuals.append(rn)
            print(f"  Iter {len(residuals):3d} | Precond residual: {rn:.3e}")

        t0 = time.perf_counter()
        y, info = spla.gmres(
            ARop,
            b.astype(np.complex128, copy=False),
            atol=self.atol,
            callback=gmres_callback,
            callback_type='pr_norm'
        )
        t1 = time.perf_counter()

        # Recover solution in original variables: x = R y
        u = M_prec @ y



        print(f"Time with PiT preconditioner: {t1 - t0:.2f} seconds.",flush=True)
        if info == 0:
            print(f"GMRES converged in {len(residuals)} iterations.\n",flush=True)
        else:
            print(f"GMRES stopped early (info={info}) after {len(residuals)} iterations.\n",flush=True)



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
        C_avg = np.mean(C, axis=1)              # (Nx,)
        C_bar = sp.diags(C_avg, 0, format='csr')  # diag(C_bar)

        # --- Build averaged blocks ---
        A = (A - C_bar).tocsr()
        B  = (B + C_bar).tocsr()

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
            X_hat = ifft(X, axis=0, norm='ortho')

            # Solve each block
            for j in range(L):
                X_hat[j, :] = lus[j].solve(X_hat[j, :])

            Y = fft(X_hat, axis=0, norm='ortho')
            Y = Y / gamma[:, None]

            return Y.reshape(L * N)

        return LinearOperator(
            shape=(L * N, L * N),
            dtype=dtype,
            matvec=apply_prec
        )

    def _make_pit_preconditioner_multi_workers(self, A, B, C):
        """
        alpha-Block circulant PiT preconditioner with element-wise spatial averaging of C.
        Multi-worker version using ProcessPoolExecutor.
        """
        dtype = np.complex128
        alpha = self.alpha
        Nx, L = C.shape

        # --- Compute spatial mean C_bar over z/time (Nx-vector) ---
        C_avg = np.mean(C, axis=1)              # (Nx,)
        C_bar = sp.diags(C_avg, 0, format='csr')  # diag(C_bar)

        # --- Build averaged blocks ---
        A_bar = (A - C_bar).tocsr()
        B_bar  = (B + C_bar).tocsr()

        alpha_root = alpha ** (1.0 / L)
        js = np.arange(L)
        omegas = alpha_root * np.exp(2j * np.pi * js / L)

        gamma = (alpha ** (np.arange(L) / L)).astype(dtype)

        # --- Prepare worker pool ---
        executor = ProcessPoolExecutor(
            max_workers=self.num_workers
        )
        executor._initializer = _init_worker
        executor._initargs = (A_bar, B_bar, omegas)
        self._log(f"Creating the Pit preconditioner with {self.num_workers} workers.")

        def apply_prec(v):
            # Reshape v into L blocks of size Nx
            V = np.asarray(v, dtype=dtype).reshape(L, Nx).T   # (Nx, L)


            V_hat = np.fft.ifft(gamma[None, :] * V, axis=1, norm='ortho')

            # Solve each block in parallel
            X_hat = list(executor.map(
                _solve_block,
                ((j, V_hat[:, j]) for j in range(L))
            ))
            X_hat = np.column_stack(X_hat)  # (Nx, L)


            Y = np.fft.fft(X_hat, axis=1, norm='ortho') / gamma[None, :]

            return Y.T.reshape(L * Nx)

        return spla.LinearOperator(
            (Nx * L, Nx * L),
            dtype=dtype,
            matvec=apply_prec
        )








