import numpy as np
from thick_ptycho.thick_ptycho.forward_model.base import BaseForwardModel


class ForwardModelMS(BaseForwardModel):
    """Angular spectrum multislice forward/backward propagation."""

    def __init__(self, simulation_space, ptycho_object, ptycho_probes,
                  remove_global_phase=True, **kwargs):
        super().__init__(simulation_space, ptycho_object, ptycho_probes, "multislice", **kwargs)
        self.ptycho_object = ptycho_object
        self.probes = ptycho_probes
        self.simulation_space = simulation_space
        self.remove_global_phase = remove_global_phase
        assert simulation_space.dimension == 1, "ForwardModelMS only supports 1D samples."

       # Precompute angular spectrum kernels for forward/backward propagation
        self._initialize_propagation_kernels()

    def _initialize_propagation_kernels(self):
        """Precompute the forward angular spectrum kernel (H_forward)."""
        fx = np.fft.fftfreq(self.simulation_space.nx, d=self.dx)
        kz = np.sqrt(np.clip(
            self.simulation_space.k**2 - (2 * np.pi * fx)**2, 0, None
        ))

        self.H_forward = np.exp(1j * kz * self.simulation_space.dz)
        if self.remove_global_phase:
            self.H_forward *= np.exp(-1j * self.simulation_space.k *
                                     self.simulation_space.dz)

    def _propagate_between_slices(self, psi, backward=False):
        """Propagate the wavefield ψ between adjacent slices using ASM."""
        H = self.H_forward if not backward else np.conj(self.H_forward)
        return np.fft.ifft(np.fft.fft(psi) * H)

    def _solve_single_probe(self, angle_idx, probe_idx,
                            n=None, backpropagate=False):
        """
        Perform forward (or backward) propagation through multiple slices
        for a single probe position, following Eq. (2) in Maiden et al. (2012).
        """
        # Object refractive index distribution
        refractive_index = n if n is not None else self.simulation_space.n_true

        # Initial probe field at the entrance plane
        psi_incident = self.probes[angle_idx, probe_idx, :].copy()

        wavefield_through_slices = np.empty(
            (self.simulation_space.nx, self.simulation_space.nz), dtype=complex
        )
        wavefield_through_slices[:, 0] = psi_incident

        # Forward propagation through slices
        for z in range(self.simulation_space.nz - 1):
            # Transmission through current slice
            slice_transmission = np.exp(
                1j * self.simulation_space.k *
                (refractive_index[:, z] - self.simulation_space.n_medium) *
                self.simulation_space.dz
            )
            
            psi_exit = psi_incident * slice_transmission
            psi_incident = self._propagate_between_slices(
                psi_exit, backward=backpropagate
            )
            wavefield_through_slices[:, z + 1] = psi_incident
        return wavefield_through_slices