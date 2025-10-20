import numpy as np
import scipy.sparse as sp
from scipy.special import jv
from typing import Optional, Tuple, Union

from .boundary_conditions import BoundaryConditions
from ..probes import Probes


class ForwardModelPWE:
    """
    Paraxial wave finite-difference forward model linear system setup.

    This class now delegates all probe construction and phase handling to
    `Probes`. The external behavior and array shapes remain compatible with
    previous code, including `BoundaryConditions`.

    Parameters
    ----------
    sample_space : object
        Geometry/discretization descriptor (see `Probes` docstring).
    thin_sample : bool
        If True, use sub-sampled grids per scan from `detector_frame_info`.
    full_system : bool, default False
        If True and not thin_sample, prebuild a system slice used for repeated
        scan indices (performance path in 1D).
    """

    def __init__(self, sample_space, thin_sample,
                 full_system: bool = False):
        self.sample_space = sample_space
        self.full_system = full_system
        self.thin_sample = thin_sample
        self.signal_strength = 1.0  # reserved for future scaling

        # Grid sizes (respect thin sample sub-sampling)
        if thin_sample:
            self.nx = self.sample_space.sub_nx
            if self.sample_space.dimension == 2:
                self.ny = self.sample_space.sub_ny
        else:
            self.nx = self.sample_space.nx
            if self.sample_space.dimension == 2:
                self.ny = self.sample_space.ny

        if not self.thin_sample and self.full_system:
            self.A_slice, self.B_slice, self.b_slice = self.create_system_slice()

        self.block_size = self.nx if self.sample_space.dimension == 1 else self.nx * self.ny

        self.b0 = None
    
    def precompute_b0(self,probes: np.ndarray):
        """Precompute b0 (fast path for 1D, full system)"""
        assert (
            self.sample_space.dimension == 1
            and (not self.thin_sample)
            and self.full_system
        ), "precompute_b0 fast path requires 1D, not thin_sample and full_system"
        num_angles = len(self.sample_space.probe_angles_list)
        num_probes = self.sample_space.num_probes
        flat_shape = (num_angles * num_probes,) + probes.shape[2:]
        probes = probes.reshape(flat_shape)
        self.b0 = None
        b0_mat = self.B_slice @ probes.T  # (block_size, num_probes_total)
        self.b0 = b0_mat.T + self.b_slice

    # ------------------------- forward model ---------------------------

    def setup_homogeneous_forward_model_lhs(self, scan_index: int = 0, angle_index: int = 0):
        """
        Construct the homogeneous (free-space) LHS block-tridiagonal system.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix of size ( (nz-1)*block_size , (nz-1)*block_size ).
        """
        if self.thin_sample or not self.full_system:
            self.A_slice, self.B_slice, _ = self.create_system_slice(scan_index=scan_index, angle_index=angle_index)

        A_homogeneous = (
            sp.kron(sp.eye(self.sample_space.nz - 1, format="csr"), self.A_slice, format="csr")
            - sp.kron(
                sp.diags([1], [-1], shape=(self.sample_space.nz - 1, self.sample_space.nz - 1), format="csr"),
                self.B_slice,
                format="csr",
            )
        )
        return A_homogeneous

    def setup_homogeneous_forward_model_rhs(self, probes: np.ndarray, scan_index: int = 0, angle_index: int = 0):
        """
        Construct the homogeneous (free-space) RHS vector (stacked b-slices).

        Returns
        -------
        ndarray (complex)
            Vector of length (nz-1)*block_size.
        """
        probe_index = angle_index * self.sample_space.num_probes + scan_index
        if self.thin_sample or not self.full_system:
            _, _, self.b_slice = self.create_system_slice(probe=probes[probe_index, ...], scan_index=scan_index)
        return np.tile(self.b_slice, self.sample_space.nz - 1)

    def test_exact_impedance_forward_model_rhs(self,probe):
        """
        Build RHS using the average of exact boundary conditions between slices.

        Useful for validating impedance BC implementation.
        """
        bcs = BoundaryConditions(self.sample_space, probe, thin_sample=self.thin_sample)
        z = self.sample_space.z
        nz = self.sample_space.nz
        out = []
        for j in range(1, nz):
            b0 = bcs.get_exact_boundary_conditions_system(z[j - 1])
            b1 = bcs.get_exact_boundary_conditions_system(z[j])
            out.append(0.5 * (b0 + b1))
        return np.concatenate(out, axis=0)

    def test_exact_impedance_rhs_slice(self, j: int, probe: np.ndarray):
        """
        Average exact boundary conditions across a single slab interface j-1→j.
        """
        bcs = BoundaryConditions(self.sample_space, probe, thin_sample=self.thin_sample)
        b_old = bcs.get_exact_boundary_conditions_system(self.sample_space.z[j - 1])
        b_new = bcs.get_exact_boundary_conditions_system(self.sample_space.z[j])
        return (b_old + b_new) / 2

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
        object_slices = self.sample_space.create_sample_slices(
            self.thin_sample, n=n, grad=grad, scan_index=scan_index
        )
        return sp.diags(object_slices.T.flatten()) + sp.diags(
            [object_slices[..., 1:].T.flatten()], offsets=[-self.block_size]
        )

    def probe_contribution(
        self,
        scan_index: int = 0,
        angle_index: int = 0,
        probes: Optional[np.ndarray] = None,
    ):
        """
        Compute the contribution of the probe boundary at z = z0.

        Parameters
        ----------
        scan_index : int
            Which scan (position) in the ptychographic grid.
        angle_index : int
            Which angle index from `probe_angles_list`.
        probes : ndarray, optional
            If provided, use this probe stack (flattened index is angle*num_probes + scan).

        Returns
        -------
        (b0_stacked, probe_vector) : (ndarray, ndarray)
            b0_stacked : vector of length (nz-1)*block_size with b0 then zeros.
            probe_vector : flattened probe field at z0 of length block_size.
        """
        probe_index = angle_index * self.sample_space.num_probes + scan_index
        if self.thin_sample:
            _, self.B_slice, self.b_slice = self.create_system_slice(probe=probes[probe_index, ...], scan_index=scan_index)

        probe = probes[probe_index, ...].flatten()

        if self.sample_space.dimension == 1 and (not self.thin_sample) and self.full_system and (probes is None):
            b0 = self.b0[scan_index, :]
        else:
            b0 = self.B_slice @ probe + self.b_slice

        probe_contrib = np.concatenate((b0, np.zeros(self.block_size * (self.sample_space.nz - 2))))
        return probe_contrib

    def create_system_slice(self, probe: np.ndarray, scan_index: int = 0):# angle_index: int = 0):
        """
        Build per-scan boundary-condition matrices (no caching).

        Returns
        -------
        (A_slice, B_slice, b_slice) : tuple
            A_slice : LHS block for slice coupling
            B_slice : LHS block coupling to previous axial slice
            b_slice : RHS boundary term at z0 for this probe
        """
        #probe_index = angle_index * self.sample_space.num_probes + scan_index
        bcs = BoundaryConditions(
            self.sample_space,
            probe,
            thin_sample=self.thin_sample,
            scan_index=scan_index,
        )
        A_slice, B_slice = bcs.get_matrix_system()
        b_slice = bcs.get_initial_boundary_conditions_system()
        return A_slice, B_slice, b_slice
