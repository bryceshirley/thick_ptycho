from typing import Optional

import numpy as np

from thick_ptycho.forward_model.base.base_solver import BaseForwardModelSolver


class MSForwardModelSolver(BaseForwardModelSolver):
    """Angular spectrum multislice forward/backward propagation."""

    def __init__(
        self,
        simulation_space,
        ptycho_probes,
        results_dir="",
        use_logging=False,
        verbose=False,
        log=None,
    ):
        super().__init__(
            simulation_space,
            ptycho_probes,
            results_dir=results_dir,
            use_logging=use_logging,
            verbose=verbose,
            log=log,
        )

        assert (
            self.simulation_space.dimension == 2
        ), "ForwardModelMS only supports 2D samples."

        self.k = self.simulation_space.k  # Wave number
        self.dx = self.simulation_space.dx  # Pixel size in x
        self.nx = self.simulation_space.nx  # Number of pixels in x
        self.nz = self.simulation_space.nz  # Number of slices in z
        self.dz = self.simulation_space.dz  # Distance between slices
        self.n_medium = (
            self.simulation_space.n_medium
        )  # Refractive index of surrounding medium

        # Precompute angular spectrum kernels for forward/backward propagation
        self._kernel_cache = {}

    def reset_cache(self):
        self._kernel_cache = {}

    def _get_propagation_kernels(
        self, dz: float, nx_eff: Optional[int] = None, remove_global_phase=True
    ) -> np.ndarray:
        """Precompute the forward angular spectrum kernel (H_forward)."""
        nx = self.nx if nx_eff is None else nx_eff
        dz = self.dz if dz is None else dz
        key = (dz, nx_eff)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        fx = np.fft.fftfreq(nx, d=self.dx)
        kx = 2 * np.pi * fx  # Spatial frequency in x

        fx = np.fft.fftfreq(nx, d=self.dx)
        kx = 2 * np.pi * fx  # Spatial frequency in x
        inside = self.k**2 - kx**2
        kz = np.sqrt(np.clip(inside, 0.0, None))

        H = np.exp(1j * kz * dz)

        # Remove global phase factor for stability if requested
        if remove_global_phase:
            H *= np.exp(-1j * self.k * dz)

        self._kernel_cache[key] = H.astype(np.complex128)
        return self._kernel_cache[key]

    def _propagate(self, psi, dz):
        """Propagate the wavefield between adjacent slices using ASM."""
        Psi = np.fft.fft(psi)
        Psi *= self._get_propagation_kernels(dz, psi.size)
        return np.fft.ifft(Psi)

    # def _backpropagate(self, psi, dz):
    #     """Inverse propagation (negative dz)."""
    #     return self._propagate(psi, -dz)
    def _backpropagate(self, psi, dz):
        Psi = np.fft.fft(psi)
        H = self._get_propagation_kernels(dz, psi.size)
        Psi *= np.conj(H)  # adjoint of forward operator with same dz
        return np.fft.ifft(Psi)

    def _object_transmission_function(self, n_slice):
        """Compute the object transmission function for a given slice."""
        return np.exp(1j * self.k * (n_slice - self.n_medium) * self.dz)

    def _solve_single_probe(
        self,
        scan_idx: int = 0,
        n: Optional[np.ndarray] = None,
        obj: Optional[np.ndarray] = None,
        mode="forward",
        probe: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
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
        mode : {'forward', 'reverse', 'forward_rotated', 'reverse_rotated'}, optional
            If 'reverse', performs backward propagation (adjoint operation).
            If 'forward', performs forward propagation.
        **kwargs : dict, optional
            Additional arguments (not used here).

        Returns
        -------
        wavefield_through_slices : ndarray
            Complex-valued wavefield at each slice along z. Shape (nx, nz).
        """
        assert mode in {
            "forward",
            "backward",
            "forward_rotated",
            "reverse_rotated",
        }, f"Invalid mode: {mode}"

        if mode in {"forward_rotated", "reverse_rotated"}:
            n = self.rotate_n(n)

        # Object refractive index distribution
        if n is None:
            n = self.simulation_space.refractive_index_empty

        # Select initial probe condition
        if probe is None:
            probe = np.zeros((self.nx,), dtype=complex)

        # Initial probe field at the entrance plane
        psi_incident = probe.copy()
        u = np.empty((self.nx, self.nz), dtype=complex)
        u[:, 0] = psi_incident.copy()

        # Forward propagation through slices
        for z in range(self.nz - 1):
            # Transmission through current slice
            # Reference:
            #   F. Wittwer, J. Hagemann, D. Brückner, S. Flenner, and C. G. Schroer,
            #   "Phase retrieval framework for direct reconstruction of the projected refractive index
            #   applied to ptychography and holography," *Optica*, vol. 9, no. 3, pp. 288–297, 2022.
            #   DOI: https://doi.org/10.1364/OPTICA.447021
            if obj is None:
                obj_slice = self._object_transmission_function(n_slice=n[:, z])
            else:
                obj_slice = obj[:, z]

            # Compute exit wave and propagate to next slice
            psi_exit = psi_incident * obj_slice

            # Propagate to next slice
            psi_incident = self._propagate(psi_exit, self.dz)
            u[:, z + 1] = psi_incident.copy()
        return u
