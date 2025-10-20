import numpy as np
from thick_ptycho.forward_model.base_solver import BaseForwardModel


class ForwardModelMS(BaseForwardModel):
    """Angular spectrum multislice forward/backward propagation."""

    def __init__(self, sample_space, pad_factor=1.0, use_padding=False, remove_global_phase=True, **kwargs):
        super().__init__(sample_space, "multislice", thin_sample=False, **kwargs)
        self.k = sample_space.k
        self.dx = sample_space.dx
        self.dz = sample_space.dz
        self.n_medium = sample_space.n_medium
        self.pad_factor = pad_factor
        self.use_padding = use_padding
        self.remove_global_phase = remove_global_phase
        self._kernel_cache = {}

    def _get_kernel(self, dz, nx_eff=None):
        nx_eff = nx_eff or self.sample_space.nx
        key = (dz, nx_eff)
        if key not in self._kernel_cache:
            fx = np.fft.fftfreq(nx_eff, d=self.dx)
            kz = np.sqrt(np.clip(self.k ** 2 - (2 * np.pi * fx) ** 2, 0, None))
            H = np.exp(1j * kz * dz)
            if self.remove_global_phase:
                H *= np.exp(-1j * self.k * dz)
            self._kernel_cache[key] = H
        return self._kernel_cache[key]

    def _propagate(self, psi, dz):
        H = self._get_kernel(dz, psi.size)
        return np.fft.ifft(np.fft.fft(psi) * H)

    def _solve_single_probe(self, angle_idx, probe_idx, n=None, backpropagate=False, **kwargs):
        n = n if n is not None else self.sample_space.n_true
        assert n.shape == (self.sample_space.nx, self.sample_space.nz)
        psi = self.probes[angle_idx, probe_idx, :].copy()
        nz = self.sample_space.nz
        psi_stack = np.empty((self.sample_space.nx, nz), dtype=complex)
        psi_stack[:, 0] = psi

        for z in range(nz - 1):
            T = np.exp(1j * self.k * (n[:, z] - self.n_medium) * self.dz)
            dz_eff = -self.dz if backpropagate else self.dz
            psi = self._propagate(psi * T, dz_eff)
            psi_stack[:, z + 1] = psi
        return psi_stack
