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
        use_logging=False,
        verbose=True,
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

        probes = self.ptycho_probes[0, :]  # shape (num_probes, nx)

        for it in range(max_iters):
            loss_total = 0.0

            for p in range(self.num_probes):
                # Forward propagate probe p through current object estimate
                psi_slices = self.ms._solve_single_probe(n=n_est, scan_idx=p,
                                                         probe=probes[p])  # shape (nx, nz)
                # Compute loss at detector plane
                Psi_det = np.fft.fft(psi_slices[:, -1])
                target_amp = np.sqrt(self.data[p, :])
                loss_total += np.sum((np.abs(Psi_det) - np.abs(target_amp))**2) / (np.sum(np.abs(target_amp)**2) + 1e-12)
                
                 # Inverse FFT → corrected field at detector plane
                psi_corr = self._apply_modulus_constraint(psi_slices[:, -1], self.data[p, :])
    
                # Backpropagate corrections slice-by-slice
                psi_back = psi_corr
                for z in reversed(range(self.nz - 1)):
                    # Propagate one slice backward
                    psi_back = self.ms._backpropagate(psi_back, self.dz)

                    # Compute current slice transmission
                    O_z = self.ms._object_transmission_function(n_est[:, z])
                    psi_back *= np.conj(O_z)

                    # Residual at slice plane
                    dpsi_z = psi_back - psi_slices[:, z]

                    # Update object (small step; do NOT decay too fast)
                    n_est[:, z] = self._update_object_n(n_est[:, z], psi_slices[:, z], dpsi_z, alpha_obj/(it+1))

                    # # (Optional) update stored slice field to keep consistency / damping
                    psi_slices[:, z] = psi_slices[:, z] + 0.25 * dpsi_z  # mild relaxation

            loss_history.append(loss_total / self.num_probes)
            if self.verbose:
                self._log(f"[Iter {it+1:03d}] Mean Loss = {loss_total/self.num_probes:}")

        return n_est, loss_history

    # -------------------------------------------------------------------------
    # Internal numerical routines
    # -------------------------------------------------------------------------

    def _apply_modulus_constraint(self, Psi, detector_data):
        if self.phase_retrieval:
            # Forward FFT
            fmodel = np.fft.fft(Psi, axis=-1)

            # Avoid division by zero
            magnitude = np.abs(fmodel)
            magnitude[magnitude == 0] = 1.0
            phase = fmodel / magnitude

            # Select measured amplitude data
            target_amplitude = np.sqrt(detector_data)

            # Replace amplitude while keeping phase (vectorized)
            fmodel_updated = target_amplitude * phase

            # Inverse FFT to get updated exit waves
            emodel = np.fft.ifft(fmodel_updated, axis=-1)

            return emodel
        else:
            return Psi - detector_data


    # def _update_object(f, g, dpsi, alpha, eps=1e-12):
    #     # Standard 3PIE object update: normalized by |Sz|^2 locally
    #     denom = np.maximum(np.max(np.abs(g) ** 2), eps)
    #     return f + alpha * (np.conj(g) * dpsi) / denom
    def _update_object_n(self, n_slice, psi_i, dpsi, alpha_obj, eps=1e-12):
        """
        Update refractive index slice directly (3PIE update in n-space).
        Based on O = exp(i k (n - n_med) dz) and U-update rule in O-space.
        """

        # Gradient term (Maiden 2012 Eq. 3 numerators)
        grad = np.conj(psi_i) * dpsi     # same structure as O-update

        # Global normalization (stability recommended in paper)
        scale = np.max(np.abs(psi_i)**2) + eps

        # Refractive index update derived from: dO = i k dz O * dn
        dOz = (alpha_obj ) * (grad / scale)

        # Apply update
        n_slice = n_slice + dOz/ (1j * self.k * self.dz)

        return n_slice



