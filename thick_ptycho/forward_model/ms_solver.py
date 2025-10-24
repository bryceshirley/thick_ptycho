from typing import Optional
import numpy as np
from thick_ptycho.forward_model.base_solver import BaseForwardModel


class ForwardModelMS(BaseForwardModel):
    """Angular spectrum multislice forward/backward propagation."""

    def __init__(self, simulation_space, ptycho_object, ptycho_probes,
                 results_dir="", use_logging=False, verbose=True, log=None):
        super().__init__(
            simulation_space,
            ptycho_object,
            ptycho_probes,
            results_dir=results_dir,
            use_logging=use_logging,
            verbose=verbose,
            log=log,
        )

        assert simulation_space.dimension == 1, "ForwardModelMS only supports 1D samples."

       # Precompute angular spectrum kernels for forward/backward propagation
        self._initialize_propagation_kernels()

        # Solver type (for logging purposes)
        self.solver_type = "Multislice Solver"

    def _initialize_propagation_kernels(self, remove_global_phase=True):
        """Precompute the forward angular spectrum kernel (H_forward)."""
        fx = np.fft.fftfreq(self.simulation_space.nx, d=self.dx)
        kz = np.sqrt(np.clip(
            self.simulation_space.k**2 - (2 * np.pi * fx)**2, 0, None
        ))

        self.H_forward = np.exp(1j * kz * self.simulation_space.dz)

        # Remove global phase factor for stability if requested
        if remove_global_phase:
            self.H_forward *= np.exp(-1j * self.simulation_space.k *
                                     self.simulation_space.dz)

    def _propagate_between_slices(self, psi, mode="forward"):
        """Propagate the wavefield ψ between adjacent slices using ASM."""
        H = self.H_forward if mode == "forward" or mode == "forward_rotated" else np.conj(self.H_forward)
        return np.fft.ifft(np.fft.fft(psi) * H)
    
    def _object_transmission_function(self, n_slice):
        """Compute the object transmission function for a given slice."""
        return np.exp(
            1j * self.simulation_space.k *
            (n_slice - self.simulation_space.n_medium) *
            self.simulation_space.dz # Using dz here is tecnically wrong. Slice thickness is difficult to recover.
        )

    def _solve_single_probe(self, proj_idx: int, angle_idx: int, scan_idx: int,
                            n=None, mode="forward", 
                            initial_condition: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Perform forward (or backward) propagation through multiple slices
        for a single probe position, following Eq. (2) in Maiden et al. (2012).

        Parameters
        ----------
        angle_idx : int
            Index of the illumination angle.
        scan_idx : int
            Index of the probe position.
        n : ndarray, optional
            Refractive index distribution. If None, uses self.simulation_space.n_true.
        mode : {'forward', 'reverse'}, optional
            If 'reverse', performs backward propagation (adjoint operation).
            If 'forward', performs forward propagation.
        **kwargs : dict, optional
            Additional arguments (not used here).

        Returns
        -------
        wavefield_through_slices : ndarray
            Complex-valued wavefield at each slice along z. Shape (nx, nz).
        """
        if proj_idx == 1 and mode in {"forward", "adjoint"}:
            mode = mode + "_rotated"
        assert mode in {"forward", "adjoint", "forward_rotated", "adjoint_rotated"}, f"Invalid mode: {mode}"
        # Object refractive index distribution
        refractive_index = n if n is not None else self.simulation_space.n_true

        if mode in {"forward_rotated", "adjoint_rotated"}:
            refractive_index = self.rotate_n(refractive_index)

        # Select initial probe condition
        if initial_condition is not None:
            probe = initial_condition[angle_idx, scan_idx, :]
        else:
            probe = self.probes[angle_idx, scan_idx, :]

        # Initial probe field at the entrance plane
        psi_incident = probe.copy()
        u = np.empty((self.simulation_space.nx, self.simulation_space.nz), dtype=complex)
        u[:, 0] = psi_incident

        # Forward propagation through slices
        for z in range(self.simulation_space.nz - 1):
            # Transmission through current slice
            # Reference:
            #   F. Wittwer, J. Hagemann, D. Brückner, S. Flenner, and C. G. Schroer,
            #   "Phase retrieval framework for direct reconstruction of the projected refractive index
            #   applied to ptychography and holography," *Optica*, vol. 9, no. 3, pp. 288–297, 2022.
            #   DOI: https://doi.org/10.1364/OPTICA.447021
            O_z = self._object_transmission_function(n_slice=refractive_index[:, z])

            # Compute exit wave and propagate to next slice
            psi_exit = psi_incident * O_z

            # Propagate to next slice
            psi_incident = self._propagate_between_slices(psi_exit, mode=mode)
            u[:, z + 1] = psi_incident
        return u