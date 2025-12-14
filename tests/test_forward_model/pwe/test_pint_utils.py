import numpy as np
import scipy.sparse as sp
import pytest

from thick_ptycho.forward_model.pwe.solvers.utils._pint_utils import (
    _pintobj_matvec_exact,
)  # replace with the actual import


@pytest.mark.parametrize("Nx,L", [(5, 4), (3, 6)])
def test_adjoint_property(Nx, L):
    rng = np.random.default_rng(42)

    # Random sparse matrices A and B
    density = 0.5
    A = sp.random(
        Nx, Nx, density=density, format="csr", dtype=np.complex128, random_state=rng
    )
    B = sp.random(
        Nx, Nx, density=density, format="csr", dtype=np.complex128, random_state=rng
    )

    # Random dense object contribution
    C = rng.random((Nx, L)) + 1j * rng.random((Nx, L))

    # Random test vectors
    v = rng.random(Nx * L) + 1j * rng.random(Nx * L)
    w = rng.random(Nx * L) + 1j * rng.random(Nx * L)

    # Apply forward and adjoint
    Av = _pintobj_matvec_exact(A, B, C, L, w, mode="forward")
    Atu = _pintobj_matvec_exact(A, B, C, L, v, mode="adjoint")

    # Compute inner products
    ip1 = np.vdot(v, Av)  # <v, A w>
    ip2 = np.vdot(Atu, w)  # <A* v, w>

    # They should be equal up to numerical precision
    np.testing.assert_allclose(ip1, ip2, rtol=1e-12, atol=1e-12)
