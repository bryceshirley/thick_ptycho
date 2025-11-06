from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.sparse.linalg as spla

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
    V = np.reshape(v, (Nx, L), order='F')

    # Diagonal blocks: Ai[j] V[:, j] = (A - diag(C[:,j])) V[:,j]
    U = (A_csr @ V) - (C * V)

    # Subdiagonal blocks: j>=1, add -Bi[j] V[:, j-1] = -(B + diag(C[:,j])) V[:, j-1]
    U[:, 1:] += -(B_csr @ V[:, :-1] + C[:, 1:] * V[:, :-1])

    # Flatten back in the SAME ordering
    return np.reshape(U, (Nx * L,), order='F')