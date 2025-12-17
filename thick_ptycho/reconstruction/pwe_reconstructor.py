import time
from typing import Optional

import numpy as np
from numpy.fft import fft2, fftshift, ifft2

from thick_ptycho.forward_model.pwe.solvers import (
    PWEIterativeLUSolver,
    PWEPetscFullPinTSolver,
)
from .base_reconstructor import ReconstructorBase


class ReconstructorPWE(ReconstructorBase):
    """
    Nonlinear Conjugate Gradient (NLCG) reconstruction for PWE-based
    thick-sample ptychography.

    This class minimizes the least-squares loss:

        L(n) = ½ || F(n) - data ||²₂,

    where F(n) is the simulated exit wave (or its Fourier intensity)
    produced by the forward model. Supports both full complex field
    and intensity-only data (phase retrieval mode).

    Optionally, for intensity-only data, the class can perform an
    **initial Gerchberg–Saxton (GS)** phase retrieval step to estimate
    the complex exit waves before nonlinear optimization. This step
    alternates between the object and detector planes to impose both
    intensity and support constraints, yielding an approximate phase
    that often improves convergence of the subsequent NLCG stage.
    """

    def __init__(
        self,
        simulation_space,
        data,
        phase_retrieval=True,
        solver_type="iterative",
        bc_type="impedance",
        verbose: bool = None,
        alpha: float = 1e-8,
        atol: float = 5e-3,
    ):
        super().__init__(
            simulation_space=simulation_space,
            data=data,
            phase_retrieval=phase_retrieval,
            verbose=verbose,
        )
        # Store number of tomographic projections
        if solver_type not in {"full", "iterative"}:
            raise ValueError(f"Invalid solver_type: {solver_type!r}")

        # Forward model selection
        if solver_type == "iterative":
            self.forward_model = PWEIterativeLUSolver(
                simulation_space,
                self.ptycho_probes,
                bc_type=bc_type,
                results_dir=simulation_space.results_dir,
                use_logging=simulation_space.use_logging,
                verbose=simulation_space.verbose,
            )
        if solver_type == "full":
            self.forward_model = PWEPetscFullPinTSolver(
                simulation_space,
                self.ptycho_probes,
                bc_type=bc_type,
                results_dir=simulation_space.results_dir,
                use_logging=False,
                verbose=False,
                alpha=alpha,
                atol=atol,
            )
            self.forward_model.preconditioner["forward"].factorize_blocks()
            self.forward_model.preconditioner["adjoint"].factorize_blocks()
        self._log("Initializing Least Squares Solver...")

        if self.num_projections > 1 and simulation_space.solve_reduced_domain:
            raise NotImplementedError(
                "Tomographic projections with reduced domain is not supported."
            )

    def compute_forward_model(self, probes: Optional[np.ndarray] = None):
        """Compute the forward model for the current object and gradient."""
        # Prepare solver caches for all projections & modes
        # for when nk does not change between iterations
        if not self.simulation_space.solve_reduced_domain:
            self.forward_model.prepare_solver_caches(
                n=self.nk, modes=("forward", "adjoint")
            )
        uk = self.convert_to_vector_form(
            self.forward_model.solve(n=self.nk, probes=probes)
        )

        return uk

    def compute_grad_least_squares(self, exit_wave_error, uk, grad_A):
        """Compute the gradient of least squares problem."""

        # For speed: avoid recomputing sub-block size & flatten operation repeatedly
        projection_size = self.nx * (self.nz - 1)

        grad_real = np.zeros(projection_size, dtype=float)
        grad_imag = np.zeros(projection_size, dtype=float)

        idx = 0
        for idx, (proj_idx, angle_idx, scan_idx) in enumerate(self.triplets):
            # 1) Solve adjoint model
            backprop = self.forward_model._solve_single_probe(
                scan_idx=scan_idx,
                proj_idx=proj_idx,
                probe=None,
                rhs_block=exit_wave_error[proj_idx, angle_idx, scan_idx],
                mode="adjoint",
                n=self.nk,
            )

            # 2) Flatten the spatial field block
            backprop = backprop[:, :-1].T.ravel()
            # import matplotlib.pyplot as plt
            # plt.imshow(np.abs(backprop).reshape((self.nz-1, self.simulation_space.effective_nx)).T);plt.colorbar;plt.title("adjoint solution"); plt.savefig("adjoint_amplitude.png")

            # 3) Compute contribution of current frame
            g_base = grad_A @ uk[idx]

            # 4) Accumulate gradient (real and imaginary parts separately)
            grad_update_re = (g_base.conj() * backprop).real
            grad_update_im = ((1j * g_base).conj() * backprop).real

            # 5) Inverse-rotate the gradient update if proj_idx > 0
            if proj_idx == 1 and not self.simulation_space.solve_reduced_domain:
                grad_update_re = self.rotate_back(grad_update_re)
                grad_update_im = self.rotate_back(grad_update_im)
            # plt.imshow(np.abs(grad_update_re).reshape((self.nz-1, self.simulation_space.effective_nx)).T);plt.colorbar();plt.title("gradient update"); plt.savefig("grad_amplitude.png"); plt.close()
            # 6) Accumulate gradient
            grad_real = self.accumulate_gradient(grad_update_re, scan_idx, grad_real)
            grad_imag = self.accumulate_gradient(grad_update_im, scan_idx, grad_imag)

        return grad_real, grad_imag

    def accumulate_gradient(self, grad_update, scan_idx, grad_total):
        """
        Accumulate the gradient update into the total gradient array.

        Parameters:
        grad_update (ndarray): The gradient update for the current frame.
        scan_idx (int): Index of the current scan position.
        grad_total (ndarray): The total gradient array to accumulate into.
        """
        if self.simulation_space.solve_reduced_domain:
            x_min, x_max = self.simulation_space._scan_frame_info[
                scan_idx
            ].reduced_limits_discrete.x
            grad_total = grad_total.reshape((self.nz - 1, self.nx)).T
            grad_total[x_min:x_max, :] -= grad_update.reshape(
                self.nz - 1, self.simulation_space.effective_nx
            ).T
            grad_total = grad_total.T.ravel()
        else:
            grad_total -= grad_update
        return grad_total

    def compute_alphak(self, u, grad_A, grad_E, d):
        """
        Compute alpha_k calculation.

        Parameters:
        u (ndarray): Current solution vector, shape (num_probes, nx * (nz - 1)).
        A (sp.spmatrix): Sparse matrix for the linear system.
        grad_A (sp.spmatrix): Gradient of A wrt n.
        d (ndarray): Descent direction vector.

        Returns:
        float: The computed denominator value.
        """
        denominator = 0.0

        for idx, (proj_idx, angle_idx, scan_idx) in enumerate(self.triplets):
            # 1) Perturbation term: (grad_A @ u[idx])
            perturb = grad_A @ u[idx]

            # 2) Retrieve probe field once per loop
            probe = self.ptycho_probes[angle_idx, scan_idx]

            # 3) Solve forward model for delta_u
            delta_u = self.forward_model._solve_single_probe(
                scan_idx=scan_idx,
                proj_idx=proj_idx,
                probe=probe,
                rhs_block=perturb.reshape(
                    self.nz - 1, self.simulation_space.effective_nx
                ).T,
                mode="forward",
                n=self.nk,
            )

            # (nz, nx) field → drop the first z-plane → flatten
            if proj_idx == 1 and not self.simulation_space.solve_reduced_domain:
                # Rotate back if proj_idx > 0
                delta_u = np.rot90(delta_u, k=-1)

            # (nz, nx) field → drop the first z-plane → flatten
            delta_u = delta_u[:, 1:].T.ravel()

            # 4) Only use last block_size elements
            delta_p_i = -delta_u[-self.block_size :] @ d[-self.block_size :]

            # 5) accumulate squared contribution
            denominator += np.linalg.norm(delta_p_i) ** 2

        # Compute the numerator
        numerator = np.vdot(d, grad_E)

        # Compute the step size
        alphak = -(numerator / denominator) if denominator > 0 else 0.0
        return alphak

    def compute_betak(self, grad_E, grad_E_old):
        """Compute the beta_k value for the conjugate gradient method."""
        grad_diff = grad_E - grad_E_old
        grad_E_norm_sq = np.vdot(grad_E_old, grad_E_old)
        betak_pr = np.vdot(grad_E, grad_diff) / grad_E_norm_sq
        betak_fr = np.vdot(grad_E, grad_E) / grad_E_norm_sq
        betak = min(betak_fr, max(betak_pr, 0))
        return betak

    def reconstruct(
        self,
        max_iters=10,
        tol=1e-8,
        n_initial=None,
        fixed_step_size=None,
        tv_lambda_amp=0.0,
        low_pass_sigma_phase=0.0,
    ):
        """Solve the least squares problem using conjugate gradient method with optional L1/L2/TV regularization.
        Parameters
        ----------
        max_iters : int
            Maximum number of iterations.
        tol : float
            Tolerance for convergence.
        n_initial : ndarray, optional
            Initial guess for the refractive index. If None, uses uniform medium.
        fixed_step_size : float, optional
            If provided, uses a fixed step size for updates.
        solve_probe : bool
            If True, alternates between updating object and probes.
        lambda_amp : float
            Regularization parameter for amplitude TV denoising.
        sigma_phase : float
            Standard deviation for Gaussian low-pass filtering of phase.
        Returns
        -------
        n_reconstructed : ndarray
            Reconstructed refractive index.
        uk : ndarray
            Simulated exit waves for the reconstructed object.
        residual : list
            List of residuals at each iteration."""

        # Initialize the fixed step size
        if fixed_step_size is not None:
            alpha0 = fixed_step_size

        # Initialize the refractive index
        if n_initial is not None:
            n0 = n_initial
        else:
            n0 = (
                np.ones(self.simulation_space.shape, dtype=complex)
                * self.simulation_space.n_medium
            )

        # Initialize residual
        residual = []
        self.nk = n0
        gradient_update = np.zeros((self.nx, self.nz), dtype=complex)

        # Define probe
        probes = self.forward_model.probes  # Initial Simulated Probes

        # Precompute gradient operator
        grad_Ak = -self.forward_model.get_gradient(n=self.nk)

        for i in range(max_iters):
            time_start = time.time()
            self._log(f"Iteration {i + 1}/{max_iters}")
            # Compute the Forward Model
            uk = self.compute_forward_model(probes=probes)

            if self.simulation_space.exact_ref_coeff:
                # Update gradient operator if nk has changed
                grad_Ak = -self.forward_model.get_gradient(n=self.nk)

            exit_wave_error = self.get_error_in_exit_wave(uk)
            rel_rmse = np.sqrt(np.mean(np.abs(exit_wave_error[..., -1]) ** 2))
            residual.append(rel_rmse)
            # Check for convergence
            if residual[i] < tol:
                self._log(f"Converged in {i + 1} iterations.")
                break

            # Output the current iteration information
            self._log(f"    RMSE: {residual[i]}")

            # Compute the gradient of the least squares problem
            (
                grad_least_squares_real,
                grad_least_squares_imag,
            ) = self.compute_grad_least_squares(exit_wave_error, uk, grad_Ak)

            # Set direction for first iteration Conjugate Gradient
            if i == 0:
                dk_im = -grad_least_squares_imag
                grad_least_squares_imag_old = grad_least_squares_imag
                dk_re = -grad_least_squares_real
                grad_least_squares_real_old = grad_least_squares_real

            # Compute the step size
            if fixed_step_size is not None:
                alphak_re = -alpha0 / (i + 1)
                alphak_im = -alpha0 / (i + 1)
            else:
                alphak_re = self.compute_alphak(
                    uk, grad_Ak, grad_least_squares_real, dk_re
                )
                alphak_im = self.compute_alphak(
                    uk, 1j * grad_Ak, grad_least_squares_imag, dk_im
                )

            # Compute Product of step size and direction
            alphakdk_re = (alphak_re * dk_re).reshape((self.nz - 1, self.nx)).T
            alphakdk_im = (alphak_im * dk_im).reshape((self.nz - 1, self.nx)).T
            gradient_update[:, 1:] = alphakdk_re + 1j * alphakdk_im

            # Update the current estimate of the refractive index of the object
            self.nk += gradient_update

            # Compute beta using Polak-Ribière and Fletcher-Reeves
            betak_re = self.compute_betak(
                grad_least_squares_real, grad_least_squares_real_old
            )
            betak_im = self.compute_betak(
                grad_least_squares_imag, grad_least_squares_imag_old
            )

            # Update direction
            dk_re = -grad_least_squares_real + betak_re * dk_re
            grad_least_squares_real_old = grad_least_squares_real
            dk_im = -grad_least_squares_imag + betak_im * dk_im
            grad_least_squares_imag_old = grad_least_squares_imag

            # Apply TV and low-pass filtering
            if tv_lambda_amp > 0.0 or low_pass_sigma_phase > 0.0:
                self.nk = self.global_object_update(
                    self.nk, tv_lambda_amp, low_pass_sigma_phase
                )

            time_end = time.time()
            self._log(
                f"    Iteration {i + 1} took {time_end - time_start:.2f} seconds."
            )

        return self.nk, self.convert_to_tensor_form(uk), residual

    def tv_denoise_amp(self, u, lam=0.01, n_iter=30, eps=1e-8):
        """
        Chambolle-TV on amplitude only.
        """
        px = np.zeros_like(u)
        py = np.zeros_like(u)
        tau = 0.25

        for _ in range(n_iter):
            u_xy = u - lam * (np.roll(px, 1, 1) - px + np.roll(py, 1, 0) - py)
            dx = np.roll(u_xy, -1, axis=1) - u_xy
            dy = np.roll(u_xy, -1, axis=0) - u_xy
            denom = 1.0 + np.sqrt(dx * dx + dy * dy + eps) / lam
            px = (px + tau * dx) / denom
            py = (py + tau * dy) / denom

        return u - lam * (np.roll(px, 1, 1) - px + np.roll(py, 1, 0) - py)

    def gaussian_lowpass_phase(self, phi, sigma=0.08):
        """
        Low-pass filter only the phase to remove high-frequency noise,
        while preserving large-scale structure (critical for thick samples).
        """
        ny, nx = phi.shape
        qx = np.linspace(-0.5, 0.5, nx, endpoint=False)
        qy = np.linspace(-0.5, 0.5, ny, endpoint=False)
        Qx, Qy = np.meshgrid(qx, qy)
        W = np.exp(-(Qx**2 + Qy**2) / (2 * sigma**2))

        return np.real(ifft2(fft2(phi) * fftshift(W)))

    def global_object_update(self, transmission_obj, lam_amp, sigma_phase):
        """
        Apply amplitude TV + phase low-pass to the complex object transmission_obj.
        """

        # Separate amplitude & phase
        A = np.abs(transmission_obj)
        Phi = np.angle(transmission_obj)

        # Denoise amplitude using TV
        if lam_amp > 0.0:
            A = self.tv_denoise_amp(A, lam=lam_amp)

        # Smooth phase using Gaussian low-pass
        if sigma_phase > 0.0:
            Phi = self.gaussian_lowpass_phase(Phi, sigma=sigma_phase)

        # Return updated complex object
        return A * np.exp(1j * Phi)
