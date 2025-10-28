import numpy as np
from matplotlib import pyplot as plt

from thick_ptycho.reconstruction.base_reconstructor import ReconstructorBase
from thick_ptycho.forward_model.ms_solver import ForwardModelMS

class ReconstructorMS(ReconstructorBase):
    """
    Multislice 3PIE-style reconstruction for thick-sample ptychography.

    Extends ReconstructorBase to implement the 3PIE iterative algorithm,
    using the ForwardModelMS for multislice propagation and object updates.
    """

    def __init__(
        self,
        simulation_space,
        data,
        phase_retrieval=True,
        results_dir=None,
        use_logging=True,
        verbose=False,
    ):
        super().__init__(
            simulation_space=simulation_space,
            data=data,
            phase_retrieval=phase_retrieval,
            results_dir=results_dir,
            use_logging=use_logging,
            verbose=verbose,
            log_file_name="multislice_reconstruction_log.txt",
        )
        # Create ptychographic object and probes
        self.ms = ForwardModelMS(self.simulation_space, self.ptycho_object, 
                                            self.ptycho_probes,
                                            results_dir=self._results_dir,
                                            use_logging=use_logging,
                                            verbose=self.verbose,
                                            log=self._log)
        self.k = self.simulation_space.k
        self.nz = self.simulation_space.nz
        self.num_probes = self.simulation_space.num_probes
        self.dz = self.simulation_space.dz
        self.n_medium = self.simulation_space.n_medium
        self._log("Initialized Multislice 3PIE Reconstructor.")

    # -------------------------------------------------------------------------
    # Main reconstruction method
    # -------------------------------------------------------------------------

    def reconstruct(
        self,
        max_iters: int = 10,
        alpha_obj: float = 1.0,
    ):
        """
        Run multislice 3PIE reconstruction over multiple epochs.

        Parameters
        ----------
        max_iters : int, default=10
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
        n_est = self.n_medium * np.ones((self.simulation_space.nx, self.nz), dtype=complex)
        loss_history = []

        for it in range(max_iters):
            loss_total = 0.0

            for p in range(self.num_probes):
                # Forward propagate probe p through current object estimate
                psi_slices = self.ms._solve_single_probe(n=n_est, scan_idx=p)  # shape (nx, nz)
                # self.simulation_space.viewer.plot_two_panels(psi_slices,
                #                         view="phase_amp", 
                #                         title="Wavefield Solution Multislice",
                #                         xlabel="z (m)",
                #                         ylabel="x (m)")
    
                # Measured field (Fourier domain)
                Psi_det = np.fft.fft(psi_slices[:, -1], axis=-1)
                #print("max |Psi_det|:", np.max(np.abs(Psi_det)))
                # plt.plot(self.simulation_space.x, np.abs(Psi_det), label=f'probe {p}')
                # plt.show()

                amp_meas = np.sqrt(self.data[p, :])
                #print("max amp_meas:", np.max(amp_meas))

                # Enforce measured modulus constraint
                Psi_corr = self._apply_modulus_constraint(Psi_det, amp_meas)
                # Inverse FFT → corrected field at detector plane
                psi_corr = np.fft.ifft(Psi_corr)

                # Compute reconstruction loss (mean squared amplitude error)
                loss_total += np.mean((np.abs(Psi_det) - amp_meas) ** 2)
                #print(f"Probe {p}: Loss = {np.mean((np.abs(Psi_det) - amp_meas) ** 2):.3e}")

                # Inverse FFT → corrected field at detector plane
                psi_corr = np.fft.ifft(Psi_corr)

                # Backpropagate corrections slice-by-slice
                psi_back = psi_corr
                for z in reversed(range(self.nz - 1)):
                    # Propagate one slice backward
                    psi_back = self.ms._backpropagate(psi_back, self.dz)

                    # Compute current slice transmission
                    O_z = np.exp(1j * self.k * (n_est[:, z] - self.n_medium) * self.dz)
                    psi_back *= np.conj(O_z)

                    # Residual at slice plane
                    dpsi_z = psi_back - psi_slices[:, z]

                    # >>> Sensitivity for n at this slice <<<
                    Sz = 1j * self.k * self.dz * O_z * psi_slices[:, z]   # <-- crucial

                    # Update object (small step; do NOT decay too fast)
                    n_est[:, z] = self._update_object(n_est[:, z], Sz, dpsi_z, alpha_obj/(it+1))

                    # # (Optional) update stored slice field to keep consistency / damping
                    psi_slices[:, z] = psi_slices[:, z] + 0.25 * dpsi_z  # mild relaxation

            loss_history.append(loss_total / self.num_probes)
            if self.verbose:
                self._log(f"[Iter {it+1:03d}] Mean Loss = {loss_total/self.num_probes:.3e}")

        return n_est, loss_history

    # -------------------------------------------------------------------------
    # Internal numerical routines
    # -------------------------------------------------------------------------

    def _apply_modulus_constraint(self, Psi, meas_amp):
        amp = np.abs(Psi)
        Psi_target = meas_amp * (Psi / amp)
        return  Psi_target


    def _update_object(self, Sz, dpsi, alpha, eps=1e-8):
        # PIE-style normalized gradient step using sensitivity Sz
        denom = np.max(np.abs(Sz)**2) + eps
        return self.nz + alpha * np.conj(Sz) / denom * dpsi




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



