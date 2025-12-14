from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from .operator_matrices import OperatorMatrices


class PWEForwardModel:
    """
    Paraxial wave finite-difference forward model linear system setup.

    This class now delegates all probe construction and phase handling to
    `Probes`. The external behavior and array shapes remain compatible with
    previous code, including `BoundaryConditions`.

    Parameters
    ----------
    simulation_space : object
        Geometry/discretization descriptor (see `Probes` docstring).
    solve_reduced_domain : bool
        If True, use sub-sampled grids per scan from `detector_frame_info`.
    """

    def __init__(self, simulation_space, bc_type: str = "impedance"):
        self.simulation_space = simulation_space
        self.solve_reduced_domain = simulation_space.solve_reduced_domain
        self.signal_strength = 1.0  # reserved for future scaling

        # Boundary conditions operator
        self.differiential_operator_matrices = OperatorMatrices(
            self.simulation_space, bc_type=bc_type
        )
        self.nx = self.differiential_operator_matrices.nx
        self.block_size = self.differiential_operator_matrices.block_size

        if self.simulation_space.dimension == 3:
            self.ny = self.differiential_operator_matrices.ny

        # Caching control
        self.enable_caching = not self.solve_reduced_domain
        self._cache = {"AB_step": None, "b_step": None} if self.enable_caching else None

        # Optional precomputed b0 (used in homogeneous probes)
        self.b0 = None

    def _get_or_generate_step_matrices(self, probe: Optional[np.ndarray] = None):
        """
        Retrieve step matrices from cache or regenerate them.

        If caching is disabled, this always regenerates.
        """
        if self.enable_caching and self._cache["AB_step"] is not None:
            A_step, B_step = self._cache["AB_step"]
            b_step = self._cache["b_step"]
            return A_step, B_step, b_step

        # Otherwise, generate fresh matrices
        A_step, B_step, b_step = self.generate_zstep_matrices(probe)

        # Cache if allowed
        if self.enable_caching:
            self._cache["AB_step"] = (A_step, B_step)
            self._cache["b_step"] = b_step

        return A_step, B_step, b_step

    def reset_cache(self):
        """Clear cached matrices if caching is enabled."""
        if self.enable_caching:
            self._cache = {"AB_step": None, "b_step": None}

    def generate_zstep_matrices(
        self, probe: Optional[np.ndarray] = None
    ) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
        """
        Build per-scan boundary-condition matrices (no caching).
        """
        self.differiential_operator_matrices.probe = probe
        A_step, B_step = self.differiential_operator_matrices.get_matrix_system()
        b_step = (
            self.differiential_operator_matrices.get_probe_boundary_conditions_system()
        )
        self.differiential_operator_matrices.reset_probe()
        return A_step, B_step, b_step

    # Precompute b0 for all probes
    def precompute_b0(self, probes: np.ndarray):
        """Precompute b0 for all probes when caching is enabled."""
        if (
            not self.enable_caching
            or self.simulation_space.dimension != 1
            or self.solve_reduced_domain
        ):
            # No precomputation in reduced domain
            return

        _, B_step, b_step = self._get_or_generate_step_matrices()
        num_angles = len(self.simulation_space.probe_angles)
        num_probes = self.simulation_space.num_probes
        flat_shape = (num_angles * num_probes,) + probes.shape[2:]
        probes = probes.reshape(flat_shape)

        b0_mat = B_step @ probes.T  # (block_size, num_probes_total)
        self.b0 = b0_mat.T + b_step

    # ------------------------- forward model ---------------------------
    def setup_homogeneous_forward_model_lhs(self, probe: Optional[np.ndarray] = None):
        """
        Construct the homogeneous (free-space) LHS block-tridiagonal system.
        """
        A_step, B_step, _ = self._get_or_generate_step_matrices(probe)
        # A_step, B_step, _ = self.generate_zstep_matrices()
        nz = self.simulation_space.nz
        return sp.kron(sp.eye(nz - 1, format="csr"), A_step, format="csr") - sp.kron(
            sp.diags([1], [-1], shape=(nz - 1, nz - 1), format="csr"),
            B_step,
            format="csr",
        )

    def setup_homogeneous_forward_model_rhs(self, probe: Optional[np.ndarray] = None):
        """
        Construct the homogeneous (free-space) RHS vector (stacked b-steps).
        """
        _, _, b_step = self._get_or_generate_step_matrices(probe)
        return np.tile(b_step, self.simulation_space.nz - 1)

    def setup_inhomogeneous_forward_model(
        self, n=None, grad: bool = False, scan_index: int = 0
    ):
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
            Diagonal matrix with the object effective contribution and a subdiagonal coupling.
        """
        effective_object = self.simulation_space.create_object_contribution(
            n=n, grad=grad, scan_index=scan_index
        )
        return sp.diags(effective_object.T.flatten()) + sp.diags(
            [effective_object[..., 1:].T.flatten()], offsets=[-self.block_size]
        )

    def probe_contribution(
        self,
        probe: Optional[np.ndarray] = None,
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
            _, B_step, b_step = self._get_or_generate_step_matrices(probe)

            if probe is None:
                probe_vec = np.zeros(self.block_size)
            else:
                probe_vec = probe.flatten()

            b0 = B_step @ probe_vec + b_step

        return np.concatenate(
            (
                b0,
                np.zeros(
                    self.block_size * (self.simulation_space.nz - 2),
                ),
            )
        )

    def return_forward_model_matrix(
        self,
        probe: Optional[np.ndarray] = None,
        n: np.ndarray = None,
        scan_index: int = 0,
    ) -> np.ndarray:
        """Return the forward model matrix for a given scan index."""
        # Create the inhomogeneous forward model matrix
        A_homogeneous = self.setup_homogeneous_forward_model_lhs(probe=probe)

        Ck = self.setup_inhomogeneous_forward_model(n=n, scan_index=scan_index)

        return (
            A_homogeneous - Ck
        ).tocsc()  # Convert to Compressed Sparse Column format for efficiency
