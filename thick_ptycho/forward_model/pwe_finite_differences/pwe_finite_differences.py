import numpy as np
import scipy.sparse as sp
from scipy.special import jv
from typing import Optional, Tuple, Union
import scipy.sparse.linalg as spla
from .boundary_conditions import BoundaryConditions


class PWEFiniteDifferences:
    """
    Paraxial wave finite-difference forward model linear system setup.

    This class now delegates all probe construction and phase handling to
    `Probes`. The external behavior and array shapes remain compatible with
    previous code, including `BoundaryConditions`.

    Parameters
    ----------
    simulation_space : object
        Geometry/discretization descriptor (see `Probes` docstring).
    thin_sample : bool
        If True, use sub-sampled grids per scan from `detector_frame_info`.
    full_system : bool, default False
        If True and not thin_sample, prebuild a system slice used for repeated
        scan indices (performance path in 1D).
    """

    def __init__(self, simulation_space, full_system: bool = False):
        self.simulation_space = simulation_space
        self.full_system = full_system
        self.thin_sample = simulation_space.thin_sample
        self.signal_strength = 1.0  # reserved for future scaling

        # Grid sizes (respect thin sample sub-sampling)
        if self.thin_sample:
            self.slice_dimensions = simulation_space.probe_dimensions
        else:
            self.slice_dimensions = simulation_space.discrete_dimensions[:-1]

        if self.simulation_space.dimension == 1:
            self.nx = self.slice_dimensions[0]
        else:
            self.nx, self.ny = self.slice_dimensions

        if not self.thin_sample and self.full_system:
            self.A_slice, self.B_slice, self.b_slice = self.create_system_slice()

        self.block_size = self.nx if self.simulation_space.dimension == 1 else self.nx * self.ny

        self.b0 = None

    def precompute_b0(self, probes: np.ndarray):
        """Precompute b0 (fast path for 1D, full system)"""
        assert (
            self.simulation_space.dimension == 1
            and (not self.thin_sample)
            and self.full_system
        ), "precompute_b0 fast path requires 1D, not thin_sample and full_system"
        num_angles = len(self.simulation_space.probe_angles_list)
        num_probes = self.simulation_space.num_probes
        flat_shape = (num_angles * num_probes,) + probes.shape[2:]
        probes = probes.reshape(flat_shape)
        self.b0 = None
        b0_mat = self.B_slice @ probes.T  # (block_size, num_probes_total)
        self.b0 = b0_mat.T + self.b_slice

    # ------------------------- forward model ---------------------------

    def setup_homogeneous_forward_model_lhs(self, probe: np.ndarray):
        """
        Construct the homogeneous (free-space) LHS block-tridiagonal system.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix of size ( (nz-1)*block_size , (nz-1)*block_size ).
        """
        if self.thin_sample or not self.full_system:
            self.A_slice, self.B_slice, _ = self.create_system_slice(probe=probe)

        A_homogeneous = (
            sp.kron(sp.eye(self.simulation_space.nz - 1, format="csr"), self.A_slice, format="csr")
            - sp.kron(
                sp.diags([1], [-1], shape=(self.simulation_space.nz - 1, self.simulation_space.nz - 1), format="csr"),
                self.B_slice,
                format="csr",
            )
        )
        return A_homogeneous

    def setup_homogeneous_forward_model_rhs(self, probe: np.ndarray):
        """
        Construct the homogeneous (free-space) RHS vector (stacked b-slices).

        Returns
        -------
        ndarray (complex)
            Vector of length (nz-1)*block_size.
        """
        if self.thin_sample or not self.full_system:
            _, _, self.b_slice = self.create_system_slice(probe=probe)
        return np.tile(self.b_slice, self.simulation_space.nz - 1)

    def setup_inhomogeneous_forward_model(self, n=None, grad: bool = False, scan_index: int = 0):
        """
        Create the diagonal inhomogeneous forward model operator.

        Parameters
        ----------
        n : ndarray, optional
            Refractive index distribution on the (x[,y], z) grid.
        grad : bool, default False
            If True, build a gradient-operator version (as used by your code).
        scan_index : int, default 0
            Probe index for thin-sample sub-slicing.

        Returns
        -------
        scipy.sparse.dia_matrix
            Diagonal matrix with the object slices and a subdiagonal coupling.
        """
        object_slices = self.simulation_space.create_sample_slices(
            self.thin_sample, n=n, grad=grad, scan_index=scan_index
        )
        return sp.diags(object_slices.T.flatten()) + sp.diags(
            [object_slices[..., 1:].T.flatten()], offsets=[-self.block_size]
        )

    def probe_contribution(
        self,
        probe: np.ndarray,
        scan_index: int = 0,
    ):
        """
        Compute the contribution of the probe boundary at z = z0.

        Parameters
        ----------
        probe: ndarray, optional
            If provided, use this probe stack (flattened index is angle*num_probes + scan).

        Returns
        -------
        (b0_stacked, probe_vector) : (ndarray, ndarray)
            b0_stacked : vector of length (nz-1)*block_size with b0 then zeros.
            probe_vector : flattened probe field at z0 of length block_size.
        """
        if self.b0 is not None:
            b0 = self.b0[scan_index, :]
        else:
            if self.thin_sample:
                _, self.B_slice, self.b_slice = self.create_system_slice(probe=probe)
            probe = probe.flatten()
            b0 = self.B_slice @ probe + self.b_slice

        return np.concatenate((b0, np.zeros(self.block_size * (self.simulation_space.nz - 2))))

    def create_system_slice(self, probe: np.ndarray):
        """
        Build per-scan boundary-condition matrices (no caching).

        Returns
        -------
        (A_slice, B_slice, b_slice) : tuple
            A_slice : LHS block for slice coupling
            B_slice : LHS block coupling to previous axial slice
            b_slice : RHS boundary term at z0 for this probe
        """
        bcs = BoundaryConditions(self.simulation_space, probe)
        A_slice, B_slice = bcs.get_matrix_system()
        b_slice = bcs.get_initial_boundary_conditions_system()
        return A_slice, B_slice, b_slice
        

    def return_forward_model_matrix(self, scan_index: int = 0,
                                    n: np.ndarray = None) -> np.ndarray:
        """Return the forward model matrix for a given scan index."""
        # Create the inhomogeneous forward model matrix
        A_homogeneous = (
            self.setup_homogeneous_forward_model_lhs(
                scan_index=scan_index)
        )

        Ck = self.setup_inhomogeneous_forward_model(
            n=n, scan_index=scan_index)

        return (A_homogeneous - Ck).tocsc()  # Convert to Compressed Sparse Column format for efficiency
