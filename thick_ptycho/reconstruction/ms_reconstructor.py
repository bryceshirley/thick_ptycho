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
    ):
        super().__init__(
            simulation_space=simulation_space,
            data=data,
            phase_retrieval=phase_retrieval,
        )
        # Create ptychographic object and probes
        self.ms = MSForwardModelSolver(self.simulation_space, self.ptycho_probes)
        self.k = self.simulation_space.k
        self.nz = self.simulation_space.nz
        self.num_probes = self.simulation_space.num_probes
        self.probe_angles = self.simulation_space.probe_angles
        self.num_projections = self.simulation_space.num_projections
        self.num_angles = len(self.probe_angles)
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
        refractive_index: np.ndarray = None,
    ):
        """
        Run multislice 3PIE reconstruction over multiple epochs.
        """
        if refractive_index is None:
            n_est = self.n_medium * np.ones(
                (self.simulation_space.nx, self.nz), dtype=complex
            )

        loss_history = []
        probes = self.ptycho_probes  # shape (num_probes, nx)

        for it in range(max_iters):
            loss_total = 0.0

            for p_idx, a_idx, s_idx in self.triplets:
                x_min, x_max = self.simulation_space._scan_frame_info[
                    s_idx
                ].reduced_limits_discrete.x

                # ---- Forward pass ----
                psi_slices = self.ms._solve_single_probe(
                    probe_idx=p_idx,
                    n=n_est[x_min:x_max, :],
                    probe=probes[a_idx, s_idx, :],
                )

                # Detector intensity & loss
                Psi_det = np.fft.fft(psi_slices[:, -1])
                target_amp = np.sqrt(self.data[p_idx, a_idx, s_idx])
                loss_total += np.sum((np.abs(Psi_det) - target_amp) ** 2) / (
                    np.sum(target_amp**2) + 1e-12
                )

                # ---- Modulus constraint ----
                psi_corr = self._apply_modulus_constraint(
                    psi_slices[:, -1], self.data[p_idx, a_idx, s_idx]
                )

                # ---- Backpropagation & update (delegated!) ----
                self._apply_backprop_updates(
                    psi_slices,
                    psi_corr,
                    n_est,
                    alpha_obj,
                    it,
                    x_min=x_min,
                    x_max=x_max,
                )

            # Epoch loss
            mean_loss = loss_total / self.num_probes
            loss_history.append(mean_loss)

            if self.verbose:
                self._log(f"[Iter {it+1:03d}] Mean Loss = {mean_loss:.6f}")

        return n_est, loss_history

    def _apply_backprop_updates(
        self,
        psi_slices: np.ndarray,
        psi_corr: np.ndarray,
        n_est: np.ndarray,
        alpha_obj: float,
        iter_idx: int,
        x_min: int,
        x_max: int,
    ):
        """
        Backpropagate detector-plane correction psi_corr through all slices
        and update the refractive index estimate n_est in place.

        Parameters
        ----------
        psi_slices : array, shape (nx, nz)
            Forward propagated probe at every slice.
        psi_corr : array, shape (nx,)
            Corrected field at detector plane (after modulus constraint).
        n_est : array, shape (nx, nz)
            Current refractive index estimate updated in place.
        alpha_obj : float
            Step size for slice updates.
        iter_idx : int
            Current iteration index (0-based).
        """
        psi_back = psi_corr

        # Loop through slices from bottom (detector-side) → top (entrance)
        for z in reversed(range(self.nz)):
            # Backpropagate one slice
            psi_back = self.ms._backpropagate(psi_back, self.dz)

            # Slice transmission
            obj_slice = self.ms._object_transmission_function(n_est[x_min:x_max, z])
            psi_back *= np.conj(obj_slice)

            # Compute slice residual
            dpsi_z = psi_back - psi_slices[:, z]

            # Update object slice
            n_est[x_min:x_max, z] = self._update_object_n(
                n_est[x_min:x_max, z],
                psi_slices[:, z],
                dpsi_z,
                alpha_obj / (iter_idx + 1),
            )

    def low_pass(self, x, sigma=1.0):
        X = np.fft.fft(x, axis=0)
        nx = x.shape[0]
        freqs = np.fft.fftfreq(nx)
        H = np.exp(-(freqs**2) / (2 * sigma**2))
        return np.fft.ifft(X * H[:, None], axis=0)

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
        else:
            return detector_data

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
