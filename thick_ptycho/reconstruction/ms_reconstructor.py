import numpy as np

from thick_ptycho.forward_model.multislice.ms_solver import MSForwardModelSolver
from thick_ptycho.reconstruction.base_reconstructor import ReconstructorBase


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
        verbose: bool = None,
        method: str = "ASM",
        transform: str = "FFT",
        pruning_fraction: float = 0.0,
        strang_splitting: bool = False
    ):
        super().__init__(
            simulation_space=simulation_space,
            data=data,
            phase_retrieval=phase_retrieval,
            verbose=verbose,
        )
        # Create ptychographic object and probes
        self.ms = MSForwardModelSolver(self.simulation_space, self.ptycho_probes,method=method,
                                       transform=transform, pruning_fraction=pruning_fraction,
                                       strang_splitting=strang_splitting)
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
        simulate_refractive_index: bool = True,
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
        obj_est = np.ones((self.simulation_space.nx, self.nz), dtype=complex)
        loss_history = []

        probes = self.ptycho_probes[0, :]  # shape (num_probes, nx)
        tmp_free_space_field = np.ones(self.ms.nx_eff, dtype=complex) # Free space field placeholder
        empty_space_px = self.simulation_space.empty_space_px
        v_slice = slice(empty_space_px, -empty_space_px) if empty_space_px > 0 else slice(None)

        for it in range(max_iters):
            loss_total = 0.0

            for idx, (p_idx, a_idx, s_idx) in enumerate(self.triplets):
                # Forward propagate probe p through current object estimate
                if self.simulation_space.solve_reduced_domain:
                    x_min, x_max = self.simulation_space._scan_frame_info[
                        s_idx
                    ].reduced_limits_discrete.x
                else:
                    x_min, x_max = 0, self.simulation_space.nx

                psi_slices,psi_end = self.ms._solve_single_probe(
                    obj=obj_est[x_min:x_max, :], scan_idx=s_idx, probe=probes[s_idx]
                )  # shape (nx, nz)

                # Compute loss at detector plane
                Psi_det = np.fft.fft(psi_slices[:, -1])
                target_amp = np.sqrt(self.data[s_idx, :])
                loss_total += np.sum((np.abs(Psi_det[v_slice]) - np.abs(target_amp[v_slice])) ** 2) / (
                    np.sum(np.abs(target_amp[v_slice]) ** 2) + 1e-12
                )

                # Inverse FFT → corrected field at detector plane
                tmp_free_space_field[v_slice] = self._apply_modulus_constraint(
                    psi_slices[v_slice, -1], self.data[s_idx, :][v_slice]
                )

                # # Backpropagate corrections slice-by-slice
                # psi_back = psi_corr
                # for z in reversed(range(self.nz)):
                #     # Propagate one slice backward
                #     psi_back = self.ms._backpropagate(psi_back, self.dz)

                #     # Compute current slice transmission
                #     obj_slice = obj_est[:, z]
                #     psi_back *= np.conj(obj_slice)

                #     # Residual at slice plane
                #     dpsi_z = psi_back - psi_slices[:, z]

                #     # Update object (small step; do NOT decay too fast)
                #     obj_est[:, z] = self._update_object_obj(
                #         obj_est[:, z],
                #         psi_slices[:, z],
                #         dpsi_z,
                #         alpha_obj / (it + 1),
                #     )
                self._apply_backprop_updates(
                    psi_slices,
                    tmp_free_space_field,
                    obj_est,
                    alpha_obj,
                    it,
                    x_min=x_min,
                    x_max=x_max,
                    v_slice=v_slice,
                )

            loss_history.append(loss_total / self.num_probes)
            if self.verbose:
                self._log(
                    f"[Iter {it + 1:03d}] Mean Loss = {loss_total / self.num_probes:}"
                )
        if simulate_refractive_index:
            return self.ms.object_to_refractive_index(obj_est), loss_history
        else:
            return obj_est, loss_history

    def _apply_backprop_updates(
            self,
            psi_slices: np.ndarray,
            tmp_free_space_field: np.ndarray,
            obj_est: np.ndarray,
            alpha_obj: float,
            it: int,
            x_min: int,
            x_max: int,
            v_slice: slice = slice(None),
        ):
        """
        Backpropagate detector-plane correction psi_corr through all slices
        and update the refractive index estimate n_est in place.

        Parameters
        ----------
        psi_slices : array, shape (nx, nz)
            Forward propagated probe at every slice.
        tmp_free_space_field : array, shape (nx,)
            Corrected field at detector plane (after modulus constraint).
        n_est : array, shape (nx, nz)
            Current refractive index estimate updated in place.
        alpha_obj : float
            Step size for slice updates.
        it : int
            Current iteration index (0-based).
        """
        psi_back = tmp_free_space_field

        # Loop through slices from bottom (detector-side) → top (entrance)
        for z in reversed(range(self.nz)):
            # Backpropagate one slice
            psi_back = self.ms._backpropagate(psi_back, self.dz)

            # Slice transmission
            tmp_free_space_field[v_slice] = obj_est[x_min:x_max, z]
            psi_back *= np.conj(tmp_free_space_field)

            # Compute slice residual
            dpsi_z = psi_back - psi_slices[:, z]

            # Update object slice
            obj_est[x_min:x_max, z] = self._update_object_obj(
                            tmp_free_space_field,
                            psi_slices[:, z],
                            dpsi_z,
                            alpha_obj / (it + 1),
                        )[v_slice]

    # -------------------------------------------------------------------------
    # Internal numerical routines
    # -------------------------------------------------------------------------

    def _apply_modulus_constraint(self, Psi, detector_data):
        if self.phase_retrieval:
            # Forward FFT
            fmodel = np.fft.fft(Psi, axis=-1)
            phase = fmodel / (np.abs(fmodel) + 1e-12)
            f_updated = np.sqrt(detector_data) * phase
            return np.fft.ifft(f_updated)

            # # Forward FFT

            # fmodel = np.fft.fft(Psi, axis=-1)
            # # Avoid division by zero
            # magnitude = np.abs(fmodel)
            # magnitude[magnitude == 0] = 1.0
            # phase = fmodel / magnitude

            # # Select measured amplitude data
            # target_amplitude = np.sqrt(detector_data)

            # # Replace amplitude while keeping phase (vectorized)
            # fmodel_updated = target_amplitude * phase

            # # Inverse FFT to get updated exit waves
            # emodel = np.fft.ifft(fmodel_updated, axis=-1)

            # return emodel
        else:
            return detector_data

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
        grad = np.conj(psi_i) * dpsi  # same structure as O-update

        # Global normalization (stability recommended in paper)
        scale = np.max(np.abs(psi_i) ** 2) + eps

        # Refractive index update derived from: dO = i k dz O * dn
        dobj_slice = (alpha_obj) * (grad / scale)

        # Apply update
        n_slice = n_slice + dobj_slice / (1j * self.k * self.dz)

        return n_slice

    def _update_object_obj(self, obj_slice, psi_i, dpsi, alpha_obj, eps=1e-12):
        """
        Update object slice directly (3PIE update in object space).
        Based on O = exp(i k (n - n_med) dz) and U-update rule in O-space.
        """

        # Gradient term (Maiden 2012 Eq. 3 numerators)
        grad = np.conj(psi_i) * dpsi  # same structure as O-update

        # Global normalization (stability recommended in paper)
        scale = np.max(np.abs(psi_i) ** 2) + eps

        # Refractive index update derived from: dO = i k dz O * dn
        dobj_slice = (alpha_obj) * (grad / scale)

        # Apply update
        obj_slice = obj_slice + dobj_slice
        return obj_slice
