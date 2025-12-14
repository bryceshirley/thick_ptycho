import numpy as np
import scipy.sparse as sp
import pytest

from petsc4py import PETSc

from thick_ptycho.forward_model.pwe.solvers.utils._pint_utils import (
    _pintobj_matvec_exact,
)
from thick_ptycho.forward_model.pwe.solvers.full_pint import (
    PiTPreconditioner,
)
from thick_ptycho.forward_model.pwe.solvers.full_pit_petsc import (
    PiTPreconditionerShell,
)


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


@pytest.mark.parametrize("Nx,L", [(5, 4), (3, 6)])
def test_prec_adjoint_property(Nx, L):
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

    # Forward preconditioner
    prec_fwd = PiTPreconditioner(A, B, Nx, L, alpha=1e-3, mode="forward")
    prec_fwd.update(C)
    Av = prec_fwd.apply(w)

    # Adjoint preconditioner
    prec_adj = PiTPreconditioner(A, B, Nx, L, alpha=1e-3, mode="adjoint")
    prec_adj.update(C)
    Atu = prec_adj.apply(v)

    # Inner products
    ip1 = np.vdot(v, Av)  # <v, P w>
    ip2 = np.vdot(Atu, w)  # <P* v, w>

    # They should be equal up to numerical precision
    np.testing.assert_allclose(ip1, ip2, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("Nx,L", [(5, 4), (3, 6)])
def test_petsc_prec_adjoint_property(Nx, L):
    rng = np.random.default_rng(42)

    # Structured matrices for stability (Hermitian)
    A = sp.diags(np.arange(1, Nx + 1), 0, format="csr", dtype=np.complex128)
    B = sp.eye(Nx, format="csr", dtype=np.complex128)

    # Random dense object contribution
    C = rng.random((Nx, L)) + 1j * rng.random((Nx, L))

    # Random test vectors
    v = rng.random(Nx * L) + 1j * rng.random(Nx * L)
    w = rng.random(Nx * L) + 1j * rng.random(Nx * L)

    # Forward PETSc preconditioner
    prec_fwd = PiTPreconditionerShell(A, B, Nx, L, alpha=0.9, mode="forward")
    prec_fwd.update(C)

    X_w = PETSc.Vec().createWithArray(w.copy(), comm=PETSc.COMM_SELF)
    Y_fwd = PETSc.Vec().createSeq(Nx * L, comm=PETSc.COMM_SELF)

    prec_fwd.apply(None, X_w, Y_fwd)
    Av = Y_fwd.array.copy()

    # Adjoint PETSc preconditioner
    prec_adj = PiTPreconditionerShell(A, B, Nx, L, alpha=0.9, mode="adjoint")
    prec_adj.update(C)

    X_v = PETSc.Vec().createWithArray(v.copy(), comm=PETSc.COMM_SELF)
    Y_adj = PETSc.Vec().createSeq(Nx * L, comm=PETSc.COMM_SELF)

    prec_adj.apply(None, X_v, Y_adj)
    Atu = Y_adj.array.copy()

    # Inner products
    ip1 = np.vdot(v, Av)  # <v, P w>
    ip2 = np.vdot(Atu, w)  # <P* v, w>

    # They should match up to numerical precision
    np.testing.assert_allclose(ip1, ip2, rtol=1e-10, atol=1e-10)
