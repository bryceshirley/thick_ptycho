import numpy as np
from typing import Optional, List
from thick_ptycho.forward_model.multislice import ForwardModelMS
from thick_ptycho.thick_ptycho.simulation.ptycho_object import SampleSpace
from thick_ptycho.utils.utils import setup_log


from thick_ptycho.reconstruction.base import ReconstructorBase
from thick_ptycho.forward_model.multislice import ForwardModelMS


class ReconstructorMS(ReconstructorBase):
    """
    Multislice 3PIE-style reconstruction for thick-sample ptychography.

    Extends ReconstructorBase to implement the 3PIE iterative algorithm,
    using the ForwardModelMS for multislice propagation and object updates.
    """

    def __init__(
        self,
        sample_space,
        data,
        data_is_intensity,
        results_dir=None,
        use_logging=True,
        verbose=False,
        use_gs_init=True,
        gs_iters=10,
    ):
        super().__init__(
            sample_space=sample_space,
            data=data,
            data_is_intensity=data_is_intensity,
            results_dir=results_dir,
            use_logging=use_logging,
            verbose=verbose,
            use_gs_init=use_gs_init,
            gs_iters=gs_iters,
            log_file_name="multislice_reconstruction_log.txt",
        )

        self.forward_model = ForwardModelMS(sample_space)
        self.k = sample_space.k
        self.nz = sample_space.nz
        self.num_probes = sample_space.num_probes
        self.dz = sample_space.dz
        self.n_medium = sample_space.n_medium
        self._log("Initialized Multislice 3PIE Reconstructor.")

    # -------------------------------------------------------------------------
    # Main reconstruction method
    # -------------------------------------------------------------------------

    def reconstruct(
        self,
        epochs: int = 10,
        alpha_obj: float = 1.0,
    ):
        """
        Run multislice 3PIE reconstruction over multiple epochs.

        Parameters
        ----------
        epochs : int, default=10
            Number of full reconstruction iterations.
        alpha_obj : float, default=1.0
            Step size for object (refractive index) updates.

        Returns
        -------
        n_est : np.ndarray
            Reconstructed refractive index distribution.
        loss_history : list[float]
            Mean loss value per epoch.
        """
        n_est = self.n_medium * np.ones((self.sample_space.nx, self.nz), dtype=complex)
        loss_history = []

        for it in range(epochs):
            loss_total = 0.0

            for p in range(self.num_probes):
                # Forward propagate probe p through current object estimate
                psi_slices = self.forward_model.forward(n_est, probe_idx=p)[0]  # shape (nx, nz)

                # Measured field (Fourier domain)
                Psi_det = np.fft.fft(psi_slices[:, -1])
                amp_meas = np.sqrt(np.abs(self.data[p])) if self.data_is_intensity else np.abs(self.data[p])

                # Enforce measured modulus constraint
                Psi_corr = self._apply_modulus_constraint(Psi_det, amp_meas)

                # Compute reconstruction loss (mean squared amplitude error)
                loss_total += np.mean((np.abs(Psi_det) - amp_meas) ** 2)

                # Inverse FFT → corrected field at detector plane
                psi_corr = np.fft.ifft(Psi_corr)

                # Backpropagate corrections slice-by-slice
                psi_back = psi_corr
                for z in reversed(range(self.nz - 1)):
                    # Propagate one slice backward
                    psi_back = self._backpropagate(psi_back, self.dz)

                    # Compute current slice transmission
                    Tz = np.exp(1j * self.k * (n_est[:, z] - self.n_medium) * self.dz)
                    psi_back *= np.conj(Tz)

                    # Local difference
                    dpsi_z = psi_back - psi_slices[:, z]

                    # Sensitivity of slice transmission
                    Sz = 1j * self.k * self.dz * Tz * psi_slices[:, z]

                    # Update refractive index (PIE-style)
                    n_est[:, z] = self._update_object(n_est[:, z], Sz, dpsi_z, alpha_obj / (it + 1))

                    # Mild relaxation of slice field
                    psi_slices[:, z] += 0.25 * dpsi_z

            loss_history.append(loss_total / self.num_probes)
            if self.verbose:
                self._log(f"[Iter {it+1:03d}] Mean Loss = {loss_total/self.num_probes:.3e}")

        return n_est, loss_history

    # -------------------------------------------------------------------------
    # Internal numerical routines
    # -------------------------------------------------------------------------

    def _apply_modulus_constraint(self, Psi, amp_meas, gamma=1.0):
        """
        Apply modulus constraint in Fourier domain (detector plane).

        Psi' = (1 - γ) * Psi + γ * amp_meas * exp(1j * angle(Psi))

        Parameters
        ----------
        Psi : np.ndarray
            Current detector-plane complex field.
        amp_meas : np.ndarray
            Measured amplitude (sqrt of intensity).
        gamma : float, default=1.0
            Relaxation parameter for modulus constraint.

        Returns
        -------
        np.ndarray
            Updated complex field with measured modulus enforced.
        """
        return (1 - gamma) * Psi + gamma * amp_meas * np.exp(1j * np.angle(Psi))

    def _update_object(self, n_slice, Sz, dpsi_z, alpha):
        """
        Perform gradient-style update of the refractive index slice.

        n_new = n_old + α * conj(Sz) * dpsi_z / |Sz|²

        Parameters
        ----------
        n_slice : np.ndarray
            Current slice refractive index estimate.
        Sz : np.ndarray
            Sensitivity term (∂ψ/∂n).
        dpsi_z : np.ndarray
            Slice residual (backpropagated correction).
        alpha : float
            Step size for update.

        Returns
        -------
        np.ndarray
            Updated refractive index slice.
        """
        denom = np.maximum(np.abs(Sz) ** 2, 1e-12)
        return n_slice + alpha * (np.conj(Sz) * dpsi_z / denom)

    def _backpropagate(self, psi, dz):
        """
        Backward Fresnel propagation for one slice.

        Parameters
        ----------
        psi : np.ndarray
            Complex field at current slice plane.
        dz : float
            Propagation step (m).

        Returns
        -------
        np.ndarray
            Backward-propagated complex field.
        """
        return self.forward_model.propagate(psi, -dz)

    # -------------------------------------------------------------------------
    # Optional: Gerchberg–Saxton initialization
    # -------------------------------------------------------------------------

    def _gerchberg_saxton_init(self, intensities, iters=10):
        """
        Perform Gerchberg–Saxton (GS) phase retrieval to estimate
        complex exit waves from intensities.

        Parameters
        ----------
        intensities : np.ndarray
            Far-field intensities (|FFT(exit)|²) for each probe.
        iters : int, default=10
            Number of GS iterations.

        Returns
        -------
        np.ndarray
            Complex-valued exit waves with recovered phase.
        """
        sqrtI = np.sqrt(intensities)
        phi = np.exp(1j * 2 * np.pi * np.random.rand(*sqrtI.shape))
        est = sqrtI * phi

        for _ in range(iters):
            psi = np.fft.ifft(est)
            est = sqrtI * np.exp(1j * np.angle(np.fft.fft(psi)))

        return np.fft.ifft(est)




# def reconstruct(self, data, epochs=10, alpha_obj=1.0, verbose=True):
#         """
#         Multi-probe multislice reconstruction using 3PIE updates.
#         """
#         n_est = self.n_medium*np.ones_like(self.n_field)
#         loss_history = []

#         for it in range(epochs):
#             loss_total = 0
#             for p in range(self.num_probes):
#                 psi_slices = self.forward(n_est, probe_idx=p)[0]  # (nx, nz)

#                 Psi_det  = np.fft.fft(psi_slices[:, -1])
#                 amp_meas = np.sqrt(data[p])
#                 Psi_corr = self._apply_modulus_constraint(Psi_det, amp_meas, gamma=1.0)
#                 psi_corr = np.fft.ifft(Psi_corr)

#                 loss_total += np.mean((np.abs(Psi_det) - amp_meas)**2)

#                 # Backpropagate correction
#                 psi_back = psi_corr
#                 for z in reversed(range(self.nz - 1)):
#                     # Adj. propagation then adj. multiply (your code here is correct)
#                     psi_back = self._backpropagate(psi_back, self.dz)
#                     Tz = np.exp(1j * self.k * (n_est[:, z] - self.n_medium) * self.dz)
#                     psi_back *= np.conj(Tz)

#                     # Residual at slice plane
#                     dpsi_z = psi_back - psi_slices[:, z]

#                     # >>> Sensitivity for n at this slice <<<
#                     Sz = 1j * self.k * self.dz * Tz * psi_slices[:, z]   # <-- crucial

#                     # Update object (small step; do NOT decay too fast)
#                     n_est[:, z] = self._update_object(n_est[:, z], Sz, dpsi_z, alpha_obj/(it+1))

#                     # (Optional) update stored slice field to keep consistency / damping
#                     psi_slices[:, z] = psi_slices[:, z] + 0.25 * dpsi_z  # mild relaxation

#             loss_history.append(loss_total / self.num_probes)
#             if verbose:
#                 print(f"[Iter {it+1:03d}]  Mean Loss = {loss_total/self.num_probes:.3e}")

#         return n_est, loss_history



