import numpy as np
import scipy.sparse.linalg as spla
import time
import scipy.sparse as sp
from abc import ABC, abstractmethod

from scipy.fft import fft, ifft


# ---- Global block data for worker processes (Unchanged) ----
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


def _pintobj_matvec_exact(A, B, C, L, v, mode="forward"):
    """
    Computes (A_hat @ v) or (A_hat^H @ v)
    (Unchanged, but included for completeness)
    """
    Nx = A.shape[0]
    V = np.reshape(v, (Nx, L), order="F")
    U = np.zeros_like(V)

    if mode == "forward":
        # Diagonal
        U[:, :] += (A @ V) - (C * V)
        # Subdiagonal
        U[:, 1:] += -(B @ V[:, :-1] + C[:, 1:] * V[:, :-1])

    elif mode == "adjoint":
        A_H = A.conj().T
        B_H = B.conj().T
        Cc = C.conj()

        # Diagonal
        U[:, :] += (A_H @ V) - (Cc * V)
        # Superdiagonal: contribution from next slice
        U[:, :-1] += -(B_H @ V[:, 1:] + Cc[:, 1:] * V[:, 1:])
    else:
        raise ValueError("mode must be 'forward' or 'adjoint'")

    return np.reshape(U, (Nx * L,), order="F")


class AbstractPiTPreconditioner(ABC):
    """
    Implements the alpha-Block circulant PiT preconditioner.
    """

    def __init__(self, A, B, N, L, alpha, _log=None, mode="forward"):
        self.A = A
        self.B = B
        # A_bar, B_bar will be set in update()
        self.alpha = alpha
        self.N, self.L = N, L
        self.dtype = np.complex128
        self.lus = None  # will hold the factorized blocks
        self._log = _log if _log is not None else print
        self.mode = mode

        # Store A and B for easy access to their adjoints in factorize_blocks
        self._A_orig = A.tocsr()
        self._B_orig = B.tocsr()
        self.A_bar = (self._A_orig).tocsr()
        self.B_bar = (self._B_orig).tocsr()
        self.setup()

    def _apply_prec(self, v):
        """
        Applies the preconditioner P_alpha^{-1} (mode='forward') or (P_alpha^{-1})^H (mode='adjoint').
        """
        X = np.asarray(v, dtype=self.dtype).reshape(self.L, self.N)

        # Determine the scaling factor and transform type based on mode
        if self.mode == "forward":
            # P_alpha^{-1} = (Gamma_alpha^{-1} F) [Block Diagonal Inverse] (F* Gamma_alpha)
            # Apply (F* Gamma_alpha) (Conjugate Transpose DFT * Gamma)
            X = self.gamma[:, None] * X
            # DFT: F* (ifft with 'ortho' norm)
            X_hat = ifft(X, axis=0, norm="ortho")
            # Apply Block Diagonal Inverse
            X_hat = self._factorized_solve(X_hat)
            # Inverse DFT: F (fft with 'ortho' norm)
            Y = fft(X_hat, axis=0, norm="ortho")
            # Apply (Gamma_alpha^{-1})
            Y = Y / self.gamma[:, None]

        elif self.mode == "adjoint":
            # (P_alpha^{-1})^H = (Gamma_alpha F) [Block Diagonal Inverse of Adjoint] (F* Gamma_alpha^{-1})
            # Apply (F* Gamma_alpha^{-1}) (Conjugate Transpose DFT * Gamma inverse)
            X = X / self.gamma[:, None]
            # DFT: F* (ifft with 'ortho' norm)
            X_hat = ifft(X, axis=0, norm="ortho")
            # Apply Block Diagonal Inverse (uses lus of A^H - omega* B^H)
            X_hat = self._factorized_solve(X_hat)
            # Inverse DFT: F (fft with 'ortho' norm)
            Y = fft(X_hat, axis=0, norm="ortho")
            # Apply (Gamma_alpha)
            Y = self.gamma[:, None] * Y

        else:
            raise ValueError("mode must be 'forward' or 'adjoint'")

        # Reshape to (L*N,)
        return Y.ravel()

    def _factorized_solve(self, X_hat):
        """
        Applies the inverse of the central block-diagonal term.
        In 'forward' mode, this is (A - omega B)^{-1}.
        In 'adjoint' mode, this is (A^H - omega* B^H)^{-1} which is the inverse of the blocks
        computed in factorize_blocks for the adjoint problem.
        """
        if self.lus is None:
            self.factorize_blocks()

        # X_hat has shape (L, N)
        for j in range(self.L):
            # The solve method automatically handles the sparse LU factors
            X_hat[j, :] = self.lus[j].solve(X_hat[j, :])

        return X_hat

    def setup(self):
        """Setup method required by PETSc PCShell interface. No-op since we do setup in __init__."""
        alpha_root = self.alpha ** (1.0 / self.L)

        js = np.arange(self.L)
        # Omegas are the scaling of the roots of unity: Lambda_alpha = alpha^{1/L} Lambda
        self.omegas = np.exp(2j * np.pi * js / self.L) * alpha_root

        # Rescale Matrix gamma: Gamma_alpha. This is real-valued.
        # Gamma_alpha has (alpha^{j/L}) on its diagonal.
        self.gamma = (self.alpha ** (np.arange(self.L) / self.L)).astype(self.dtype)

    def update(self, C):
        """Setup the PiT preconditioner with given C matrix."""
        # --- Compute spatial mean C_bar over z/time (Nx-vector) ---
        C_avg = np.mean(C, axis=1)
        C_bar = sp.diags(C_avg, 0, format="csr")  # diag(C_bar)

        # --- Build averaged blocks: A_bar = A - C_bar, B_bar = B + C_bar ---
        self.A_bar = (self._A_orig - C_bar).tocsr()
        self.B_bar = (self._B_orig + C_bar).tocsr()

        # Pre-factorize blocks
        self.factorize_blocks()

    def factorize_blocks(self):
        """Pre-factorize the blocks of the PiT preconditioner."""
        self._log(
            f"Pre-factorizing PiT preconditioner blocks for mode '{self.mode}'..."
        )
        time_start = time.perf_counter()

        if self.mode == "forward":
            # Blocks are A_bar - omega_j * B_bar
            self.lus = [
                spla.splu((self.A_bar - (z * self.B_bar)).astype(self.dtype).tocsc())
                for z in self.omegas
            ]

        elif self.mode == "adjoint":
            # Blocks are (A_bar - omega_j * B_bar)^H = A_bar^H - omega_j* * B_bar^H
            # We need the conjugate of omega_j
            omegas_conj = self.omegas.conj()
            # We need the conjugate transpose of the averaged blocks
            A_bar_H = self.A_bar.conj().T.tocsr()
            B_bar_H = self.B_bar.conj().T.tocsr()

            self.lus = [
                spla.splu(((A_bar_H - (z_conj * B_bar_H)).astype(self.dtype)).tocsc())
                for z_conj in omegas_conj
            ]

        else:
            raise ValueError("mode must be 'forward' or 'adjoint'")

        time_end = time.perf_counter()
        self._log(
            f"PiT preconditioner block factorization time: {time_end - time_start:.2f} seconds.\n"
        )

    @abstractmethod
    def apply(self, *args, **kwargs):
        # Implementation of apply() is not provided but is required by AbstractPiTPreconditioner
        pass
