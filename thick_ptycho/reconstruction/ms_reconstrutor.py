def reconstruct(self, data, epochs=10, alpha_obj=1.0, verbose=True):
        """
        Multi-probe multislice reconstruction using 3PIE updates.
        """
        n_est = self.n_medium*np.ones_like(self.n_field)
        loss_history = []

        for it in range(epochs):
            loss_total = 0
            for p in range(self.num_probes):
                psi_slices = self.forward(n_est, probe_idx=p)[0]  # (nx, nz)

                Psi_det  = np.fft.fft(psi_slices[:, -1])
                amp_meas = np.sqrt(data[p])
                Psi_corr = self._apply_modulus_constraint(Psi_det, amp_meas, gamma=1.0)
                psi_corr = np.fft.ifft(Psi_corr)

                loss_total += np.mean((np.abs(Psi_det) - amp_meas)**2)

                # Backpropagate correction
                psi_back = psi_corr
                for z in reversed(range(self.nz - 1)):
                    # Adj. propagation then adj. multiply (your code here is correct)
                    psi_back = self._backpropagate(psi_back, self.dz)
                    Tz = np.exp(1j * self.k * (n_est[:, z] - self.n_medium) * self.dz)
                    psi_back *= np.conj(Tz)

                    # Residual at slice plane
                    dpsi_z = psi_back - psi_slices[:, z]

                    # >>> Sensitivity for n at this slice <<<
                    Sz = 1j * self.k * self.dz * Tz * psi_slices[:, z]   # <-- crucial

                    # Update object (small step; do NOT decay too fast)
                    n_est[:, z] = self._update_object(n_est[:, z], Sz, dpsi_z, alpha_obj/(it+1))

                    # (Optional) update stored slice field to keep consistency / damping
                    psi_slices[:, z] = psi_slices[:, z] + 0.25 * dpsi_z  # mild relaxation

            loss_history.append(loss_total / self.num_probes)
            if verbose:
                print(f"[Iter {it+1:03d}]  Mean Loss = {loss_total/self.num_probes:.3e}")

        return n_est, loss_history



