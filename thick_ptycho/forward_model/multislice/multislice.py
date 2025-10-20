import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from thick_ptycho.forward_model.probes import Probes


class ForwardModelMultiSlice:
    """
    1D Angular Spectrum Multi-slice Forward and Inverse (3PIE-style) Model.
    Implements forward angular spectrum propagation and inverse reconstruction
    per Maiden, Humphry, and Rodenburg (2012).
    """

    def __init__(self,
                 sample_space,
                 pad_factor: float = 1.0,
                 use_padding: bool = False,
                 dtype=np.complex64,
                 remove_global_phase: bool = True,
                 normalize_probes: bool = True):

        assert sample_space.dimension == 1, "MultiSliceForwardModel expects 1D SampleSpace."

        self.sample_space = sample_space
        self.k = sample_space.k
        self.wavelength = 2 * np.pi / self.k
        self.nx = sample_space.nx
        self.nz = sample_space.nz
        self.dx = sample_space.dx
        self.dz = sample_space.dz
        self.n_medium = sample_space.n_medium
        self.dtype = dtype
        self.remove_global_phase = remove_global_phase
        self.normalize_probes = normalize_probes

        # --- Object field
        self.n_field = sample_space.n_true  # (nx, nz)

        # --- Detector info
        self._detector_info = sample_space.detector_frame_info
        self._probe_half_width = sample_space.probe_dimensions[0] // 2

        # Probe generator
        self.probe_builder = Probes(self.sample_space, thin_sample=False)[0] # (num_probes, nx)

        # Precompute probes for all scans × angles
        self.probes = self.probe_builder.build_probes()
        self.num_probes = len(self.probes)

        # --- Minimal cache (disabled unless needed)
        self.use_padding = use_padding
        self.pad_factor = pad_factor
        self._kernel_cache = {}

    # --------------------------------------------------------------------------
    # Angular Spectrum Propagation
    # --------------------------------------------------------------------------
    def _get_kernel(self, dz: float, nx_eff: Optional[int] = None):
        """Angular spectrum kernel for forward propagation."""
        nx_eff = nx_eff or self.nx
        key = (dz, nx_eff)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        fx = np.fft.fftfreq(nx_eff, d=self.dx)
        kx = 2 * np.pi * fx
        inside = (self.k ** 2 - kx ** 2)
        kz = np.sqrt(np.clip(inside, 0.0, None))
        H = np.exp(1j * kz * dz)
        if self.remove_global_phase:
            H *= np.exp(-1j * self.k * dz)

        self._kernel_cache[key] = H.astype(self.dtype)
        return self._kernel_cache[key]

    def _propagate(self, psi, dz):
        """Forward propagation through background medium."""
        H = self._get_kernel(dz, psi.size)
        Psi = np.fft.fft(psi)
        Psi *= H
        return np.fft.ifft(Psi)

    def _backpropagate(self, psi, dz):
        """Inverse propagation (negative dz)."""
        return self._propagate(psi, -dz)

    def forward(self, n: Optional[np.ndarray] = None, probe_idx: Optional[int] = None):
        """
        Forward angular spectrum multislice propagation.

        Returns
        -------
        exit_fields : np.ndarray
            Complex detector-plane field for each probe (num_probes, nx).
        psi_slices_all : list of list of np.ndarray
            psi_slices_all[p][z] gives the complex field for probe p at slice index z.
            Shape of psi_slices_all[p][z] = (nx,)
        probe_stack : np.ndarray
            Stack of input probes (num_probes, nx).
        """
        n = n if n is not None else self.n_field
        assert n.shape == (self.nx, self.nz)

        # Choose subset of probes
        probes = self.probes if probe_idx is None else [self.probes[probe_idx]]

        if probe_idx is not None:
            num_probes = 1
        else:
            num_probes = self.num_probes
        psi_slices = np.empty((num_probes, self.nx, self.nz), dtype=self.dtype)

        for p, psi0 in enumerate(probes):
            psi_i = psi0.copy()
            psi_slices[p, :, 0] = psi_i.copy()  # field at first slice plane

            # Propagate through all slices
            for z in range(self.nz - 1):
                Tz = np.exp(1j * self.k * (n[:, z] - self.n_medium) * self.dz)
                psi_i = self._propagate(psi_i * Tz, self.dz)
                psi_slices[p, :, z + 1] = psi_i.copy()

        return psi_slices


    @staticmethod
    def _update_object(nz, Sz, dpsi, alpha, eps=1e-8):
        # PIE-style normalized gradient step using sensitivity Sz
        denom = np.max(np.abs(Sz)**2) + eps
        return nz + alpha * np.conj(Sz) / denom * dpsi
    
    def _apply_modulus_constraint(self, Psi, meas_amp, gamma=1.0):
    # gamma in (0, 1] for relaxation; 1.0 = hard constraint
        amp = np.abs(Psi) + 1e-12
        Psi_target = meas_amp * (Psi / amp)
        return (1 - gamma) * Psi + gamma * Psi_target