import numpy as np
import scipy.sparse.linalg as spla
import time
import scipy.sparse as sp
from abc import ABC, abstractmethod

from scipy.fft import fft, ifft


# ---- Global block data for worker processes ----
_WORKER_A_BAR = None
_WORKER_B_BAR = None
_WORKER_OMEGAS = None


def _init_worker(A_bar, B_bar, omegas):
    global _WORKER_A_BAR, _WORKER_B_BAR, _WORKER_OMEGAS
    _WORKER_A_BAR, _WORKER_B_BAR, _WORKER_OMEGAS = A_bar, B_bar, omegas


def _solve_block(j_vhatj):
    j, v_hat_j = j_vhatj
    A_j = _WORKER_A_BAR - _WORKER_OMEGAS[j] * _WORKER_B_BAR
    return spla.spsolve(A_j, v_hat_j)


def _pintobj_matvec_exact(A_csr, B_csr, C, L, v):
    """
    Computes (A_hat @ v) with blocks:
      Ai[j] = A - diag(C[:, j])
      Bi[j] = B + diag(C[:, j])
    and subdiagonal is -Bi[j] at block-row j (j>=1).
    Shapes:
      A_csr, B_csr: (Nx, Nx)
      C: (Nx, L)
      v: (Nx*L,)
    """
    Nx = A_csr.shape[0]

    # IMPORTANT: columns are z-slices ⇒ Fortran order
    V = np.reshape(v, (Nx, L), order="F")

    # Diagonal blocks: Ai[j] V[:, j] = (A - diag(C[:,j])) V[:,j]
    U = (A_csr @ V) - (C * V)

    # Subdiagonal blocks: j>=1, add -Bi[j] V[:, j-1] = -(B + diag(C[:,j])) V[:, j-1]
    U[:, 1:] += -(B_csr @ V[:, :-1] + C[:, 1:] * V[:, :-1])

    # Flatten back in the SAME ordering
    return np.reshape(U, (Nx * L,), order="F")


class AbstractPiTPreconditioner(ABC):
    """
    Implements the alpha-Block circulant PiT preconditioner.
    """

    def __init__(self, A, B, N, L, alpha, _log=None):
        self.A = A
        self.B = B
        self.A_bar = A.tocsr()
        self.B_bar = B.tocsr()
        self.alpha = alpha
        self.N, self.L = N, L
        self.dtype = np.complex128

        self._log = _log if _log is not None else print

        self.setup()

    def _apply_prec(self, v):
        # reshape (L,N)
        X = np.asarray(v, dtype=self.dtype).reshape(self.L, self.N)

        X = self.gamma[:, None] * X
        X_hat = ifft(X, axis=0, norm="ortho")

        # Solve each block
        X_hat = self._factorized_solve(X_hat)

        Y = fft(X_hat, axis=0, norm="ortho")
        Y = Y / self.gamma[:, None]
        return Y

    def _factorized_solve(self, X_hat):
        if self.lus is None:
            raise RuntimeError("PiT Preconditioner blocks have not been factorized.")
        for j in range(self.L):
            X_hat[j, :] = self.lus[j].solve(X_hat[j, :])
        return X_hat

    def setup(self):
        """Setup method required by PETSc PCShell interface. No-op since we do setup in __init__."""
        alpha_root = self.alpha ** (1.0 / self.L)

        js = np.arange(self.L)
        self.omegas = np.exp(2j * np.pi * js / self.L) * alpha_root

        # Rescale Matrix gamma
        self.gamma = (self.alpha ** (np.arange(self.L) / self.L)).astype(self.dtype)

    def update(self, C):
        """Setup the PiT preconditioner with given C matrix."""
        # --- Compute spatial mean C_bar over z/time (Nx-vector) ---
        C_avg = np.mean(C, axis=1)
        C_bar = sp.diags(C_avg, 0, format="csr")  # diag(C_bar)

        # --- Build averaged blocks ---
        self.A_bar = (self.A - C_bar).tocsr()
        self.B_bar = (self.B + C_bar).tocsr()

        # Pre-factorize blocks
        self.factorize_blocks()

    def factorize_blocks(self):
        """Pre-factorize the blocks of the PiT preconditioner."""
        self._log("Pre-factorizing PiT preconditioner blocks...")
        time_start = time.perf_counter()
        self.lus = [
            spla.splu((self.A_bar - (z * self.B_bar)).astype(self.dtype))
            for z in self.omegas
        ]
        time_end = time.perf_counter()
        self._log(
            f"PiT preconditioner block factorization time: {time_end - time_start:.2f} seconds.\n"
        )

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass
