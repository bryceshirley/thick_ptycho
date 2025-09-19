import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import logging

from thick_ptycho.forward_model.solver import ForwardModel
from thick_ptycho.sample_space.sample_space import SampleSpace
from thick_ptycho.utils.visualisations import Visualisation
from thick_ptycho.utils.utils import setup_log
import time

from typing import Optional, List
from scipy.ndimage import gaussian_filter1d


class LeastSquaresSolver:
    """
    Class to solve the least squares problem for the thick sample ptychography.
    This class handles the forward model, computes the gradient, and solves
    the least squares problem using a non-linear conjugate gradient method.

    Parameters
    ----------
    sample_space : SampleSpace
    full_system_solver : bool, default True
        Use full system solver; if False, use iterative variant.
    probe_angles_list : list[float], default [0.0]
        Probe angles.
    logger : logging.Logger | None, default None
        If provided, messages are sent to this logger.
        If None and use_logging=True, a logger is created.
    use_logging : bool, default True
        If True, use the Python logging system.
    verbose : bool, default False
        If True, also echo messages with self._log(). This mirrors your previous behaviour.
        You can set use_logging=False and verbose=True to only print if you prefer.
    log_level : int, default logging.INFO
        Minimum level for emitted log messages when creating an internal logger.
    """

    def __init__(self, 
                 sample_space: SampleSpace, 
                 full_system_solver: bool = True,
                 probe_angles_list: Optional[List[float]] = [0.0],
                 rotate90=False,
                 *, # all arguments after the * must be passed as keyword arguments
                 results_dir: str = None,
                 use_logging: bool = True,
                 verbose: bool = False,
                 ):
        # ---- logging/verbosity setup ----
        self._log = setup_log(results_dir,log_file_name="reconstruction_log.txt",
                               use_logging=use_logging, verbose=verbose)
        self._results_dir = results_dir

        self._log("Initializing Least Squares Solver...")

        self.nx = sample_space.nx
        self.nz = sample_space.nz
        self.sample_space = sample_space
        self.wave_number = sample_space.k
        self.num_probes = sample_space.num_probes
        self.num_angles = len(probe_angles_list)
        self.probe_angles_list = probe_angles_list
        self.true_probe_type = sample_space.probe_type  # True Probe Type is contained in sample_space

        # Currently, only thick sample mode is supported
        self.thin_sample = False

        self.full_system = full_system_solver

        # Set up forward model, linear system and visualisation objects
        self.forward_model = ForwardModel(sample_space,
                                          full_system_solver=self.full_system,
                                          thin_sample=self.thin_sample,
                                          probe_angles_list=probe_angles_list)
        self.linear_system = self.forward_model.linear_system
        self.visualisation = Visualisation(sample_space, results_dir=results_dir)

        # Set block size (number of pixels in the exit wave)
        self.block_size = self.linear_system.block_size

        # Initialize LU decomposition handles
        self.lu = None
        self.iterative_lu = None
        self.adjoint_iterative_lu = None

        # Initialize the Forward Model matrix
        self.Ak = None

        # Check if Sample Can be Rotated
        self.rotate90 = False
        if self.nx == self.nz:
            self.rotate90 = rotate90
        elif rotate90:
            self._log("90 degree rotation not possible since nx != nz")

        # Set up
        self.setup()

    
    def setup(self):
        # --- Precompute the “true” forward solution (no plots here) ---
        self._log("Solving the true forward problem to generate the dataset...")
        start_time = time.time()
        u_true = self.forward_model.solve()
        self.u_true = self.convert_to_block_form(u_true)
        self.probes_true = self.forward_model.linear_system.probes
        self.n_true = self.sample_space.n_true
        self.true_exit_waves = self.u_true[:, -self.block_size:]
        self.visualize_data()

        # If square grid, also precompute the 90 degree rotation case once
        self.u_true_rot = None
        self.true_exit_waves_rot = None
        if self.rotate90:
            self.n_true_rot = np.rot90(self.n_true, k=1)
            self.u_true_rot, _ = self.compute_forward_model(self.n_true_rot)
            self.true_exit_waves_rot = self.u_true_rot[:, -self.block_size:]
            self.visualize_data(rotate=True)

        end_time = time.time()
        self._log(f"True Forward Solution computed in {end_time - start_time:.2f} seconds.")

        self._log(f"Angle {self.probe_angles_list[0]}")
        self.visualisation.plot_auto(u_true[0], view="phase_amp", layout="single")
        if self.num_angles > 1:
            self._log(f"Angle {self.probe_angles_list[-1]}")
            self.visualisation.plot_auto(u_true[-1], view="phase_amp", layout="single")

    def visualize_data(self, rotate: bool = False) -> None:
        """
        Visualize FFT intensities, phases, and amplitudes of the exit waves, plus
        differences versus a homogeneous medium. Optionally uses the precomputed
        rotated forward model (if available).
        """

        # Select exit waves according to the orientation
        if rotate and (self.true_exit_waves_rot is not None):
            exit_waves = self.true_exit_waves_rot
            n_for_homog_shape = self.n_true_rot.shape
            title_prefix = "(rotated)"

            n_true = self.n_true_rot
        else:
            if rotate:
                self._log("Warning: rotate=True requested, but rotated forward model "
                      "was not precomputed (requires nx == nz). Using non-rotated data.")
            exit_waves = self.true_exit_waves
            n_for_homog_shape = self.n_true.shape
            title_prefix = ""
            n_true = self.n_true


        self._log("Plot True Object")
        self.visualisation.plot_single(n_true, view="phase_amp", time="final",
                                       filename=f"{'rot_' if title_prefix else ''}true_object.png")

        # Compute homogeneous forward solution (same shape/orientation as selected case)
        n_homogeneous = np.ones(n_for_homog_shape, dtype=complex) * self.sample_space.n_medium
        if title_prefix:
            # If we're visualizing the rotated case, make sure we pass the rotated
            # n to the forward model (keep other settings identical).
            u_homogeneous = self.convert_to_block_form(self.forward_model.solve(n=n_homogeneous))
        else:
            u_homogeneous = self.convert_to_block_form(self.forward_model.solve(n=n_homogeneous))

        exit_waves_homogeneous = u_homogeneous[:, -self.block_size:]
        diff_exit_waves = exit_waves_homogeneous - exit_waves

        # ---------- FFT-squared intensities ----------
        data = np.zeros((self.num_probes * self.num_angles, self.block_size))
        diff_data = np.zeros_like(data)

        for i in range(self.num_probes * self.num_angles):
            exit_wave_fft = np.fft.fftshift(np.fft.fft(exit_waves[i, :]))
            data[i, :] = np.abs(exit_wave_fft) ** 2

            diff_exit_wave_fft = np.fft.fftshift(np.fft.fft(diff_exit_waves[i, :]))
            diff_data[i, :] = np.abs(diff_exit_wave_fft) ** 2

        if self._results_dir:
            fig = plt.figure(figsize=(8, 4))
            plt.imshow(data, cmap='viridis', origin='lower')
            plt.colorbar(label='Intensity')
            plt.title(f'Exit Wave Squared FFT Intensity {title_prefix}'.strip())
            plt.xlabel('x'); plt.ylabel('Image #'); plt.tight_layout()
            fig.savefig(os.path.join(self._results_dir, f'fft_intensity{ "_rot" if title_prefix else ""}.png'),
                        bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8, 4))
            plt.imshow(diff_data, cmap='viridis', origin='lower')
            plt.colorbar(label='Intensity')
            plt.title(f'Differences in Exit Waves {title_prefix}:\nFar Field Intensity'.strip())
            plt.xlabel('x'); plt.ylabel('Image #'); plt.tight_layout()
            fig.savefig(os.path.join(self._results_dir, f'fft_intensity_diff{ "_rot" if title_prefix else ""}.png'),
                        bbox_inches="tight")
            plt.close(fig)

        # ---------- Phase & Amplitude (and differences) ----------
        self.visualisation.plot_single(
                exit_waves, view="phase_amp", time="final",
                filename=f"exit_phase_amp{ '_rot' if title_prefix else ''}.png",
                title_left=f"Exit Wave Phase {title_prefix}".strip(),
                title_right=f"Exit Wave Amplitude {title_prefix}".strip(),
                xlabel_left="x",  ylabel_left="Image #",
                xlabel_right="x", ylabel_right="Image #",
            )

        self.visualisation.plot_single(
                diff_exit_waves, view="phase_amp", time="final",
                filename=f"exit_phase_amp_diff{ '_rot' if title_prefix else ''}.png",
                title_left=f"Phase Differences in Exit Waves {title_prefix}:\nHomogeneous vs. Inhomogeneous Media".strip(),
                title_right=f"Amplitude Differences in Exit Waves {title_prefix}:\nHomogeneous vs. Inhomogeneous Media".strip(),
                xlabel_left="x",  ylabel_left="Image #",
                xlabel_right="x", ylabel_right="Image #",
            )




    def compute_forward_model(self, nk, probes: Optional[np.ndarray] = None):
        """Compute the forward model for the current object and gradient."""
        # TODO: Initial condition (probe) should update this

        if not self.thin_sample and self.full_system:
            self.Ak = self.forward_model.return_forward_model_matrix(n=nk)
            self.lu = spla.splu(self.Ak)

        if not self.thin_sample and not self.full_system:
            self.iterative_lu = self.forward_model.construct_iterative_lu(n=nk)
            self.adjoint_iterative_lu = self.forward_model.construct_iterative_lu(
                n=nk, adjoint=True)

        grad_Ak = - self.linear_system.setup_inhomogeneous_forward_model(
             n=nk, grad=True)

        uk = self.convert_to_block_form(self.forward_model.solve(
            n=nk, lu=self.lu,
            iterative_lu=self.iterative_lu,
            initial_condition=probes))

        return uk, grad_Ak

    def compute_grad_least_squares(self, uk, grad_A,rotate=False):
        """Compute the gradient of least squares problem."""
        grad_real = np.zeros(self.nx * (self.nz - 1), dtype=float)
        grad_imag = np.zeros(self.nx * (self.nz - 1), dtype=float)

        # Preallocate zero vector
        exit_wave_error = self.compute_error_in_exit_wave(uk,rotate=rotate)

        for i in range(self.num_probes):
            if self.lu is not None:
                error_backpropagation = self.lu.solve(exit_wave_error[i, :], trans='H')
            elif self.iterative_lu is not None:
                error_backpropagation = self.forward_model._solve_single_probe_iteratively(
                    initial_condition=0,
                    iterative_lu=self.adjoint_iterative_lu,
                    adjoint=True,
                    b_block=exit_wave_error[i, :])
                error_backpropagation = error_backpropagation[:, :-1].transpose().ravel()
            else:
                error_backpropagation = spla.spsolve(self.Ak.conj().T, exit_wave_error[i, :])

            g_base = (grad_A @ uk[i, :])
            grad_real -= (g_base.conj() * error_backpropagation).real
            grad_imag -= ((1j * g_base).conj() * error_backpropagation).real

        return grad_real, grad_imag

    def compute_error_in_exit_wave(self, uk,rotate=False):
        """
        Compute the error in the exit wave.

        Parameters:
        uk (ndarray): Current solution vector, shape (num_probes, nx * (nz-1)).

        Returns:
        ndarray: The exit wave error.
        """
        exit_wave_error = np.zeros_like(uk, dtype=complex)
        exit_waves = uk[:, -self.block_size:]
        if rotate:
            exit_wave_error[:, -self.block_size:] = (
            exit_waves - self.true_exit_waves_rot)
        else:
            exit_wave_error[:, -self.block_size:] = (
                exit_waves - self.true_exit_waves)

        return exit_wave_error

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

        for i in range(self.num_probes):
            # Compute the perturbation for each probe
            perturbation = grad_A @ u[i, :]
            if self.lu is not None:
                delta_u = self.lu.solve(perturbation)
            elif self.iterative_lu is not None:
                delta_u = self.forward_model._solve_single_probe_iteratively(
                    initial_condition=0,
                    iterative_lu=self.iterative_lu,
                    b_block=perturbation)
                delta_u = delta_u[:, 1:].transpose().flatten()
            else:
                delta_u = spla.spsolve(self.Ak, perturbation)

            # Only use last block_size elements (final slice)
            delta_p_i = - delta_u[-self.block_size:] @ d[-self.block_size:]

            # Accumulate squared norm
            denominator += np.linalg.norm(delta_p_i)**2

        # Compute the numerator
        numerator = np.vdot(d, grad_E)

        # Compute the step size
        alphak = - (numerator / denominator) if denominator > 0 else 0.0
        return alphak

    def compute_betak(self, grad_E, grad_E_old):
        """Compute the beta_k value for the conjugate gradient method."""
        grad_diff = grad_E - grad_E_old
        grad_E_norm_sq = np.vdot(grad_E_old, grad_E_old)
        betak_pr = np.vdot(grad_E, grad_diff) / grad_E_norm_sq
        betak_fr = np.vdot(grad_E, grad_E) / grad_E_norm_sq
        betak = min(betak_fr, max(betak_pr, 0))
        return betak

    def convert_to_block_form(self, u):
        """`
        Convert the input array to block form.

        Parameters:
        u (ndarray): Input array to be converted. shape: (ang_number, num_probes, nx, nz)

        Returns:
        ndarray: Block-formatted array. (ang_number*num_probes, nx*nz)
        """
        # 2. Remove initial condition
        u = u[:, :, :, 1:]  # shape: (ang_number, num_probes, block_size, nz - 1)

        # 3. Transpose axes
        u = u.transpose(0, 1, 3, 2) # shape: (ang_number, num_probes, nz - 1, block_size)

        # 4. Flatten last two dims
        # shape: (num_angles*num_probes, block_size * (nz - 1))
        u = u.reshape(self.num_angles*self.num_probes, -1)

        return u

    def convert_from_block_form(self, u):
        """
        Reverse the block flattening process.

        Parameters:
        u (ndarray): Flattened array of shape (num_angles, num_probes, block_size * (nz - 1))

        Returns:
        ndarray: Unflattened array of shape (num_angles, num_probes, block_size, nz - 1)
        """
        # Step 1: Reshape to (num_probes, nz - 1, block_size)
        reshaped = u.reshape(self.num_angles, self.num_probes, self.nz-1, self.block_size)

        # Step 2: Transpose to (num_probes, num_probes, nx, nz - 1)
        return reshaped.transpose(0, 1, 3, 2)

    def solve(
            self,
            n_initial=None,
            max_iters=10,
            tol=1e-8,
            plot_object=False,
            plot_forward=False,
            plot_gradient=False,
            plot_reverse=False,
            fixed_step_size=None,
            verbose=True,
            solve_probe=False,
            sparsity_lambda=0.0,
            low_pass_filter=0.0):
        """Solve the least squares problem using conjugate gradient method with optional L1/L2/TV regularization."""

        # Initialize the fixed step size
        if fixed_step_size is not None:
            alpha0 = fixed_step_size

        # Initialize the refractive index
        if n_initial is not None:
            n0 = n_initial
        else:
            n0 = np.ones((self.block_size, self.nz), dtype=complex)*self.sample_space.n_medium

        # Output True Object and Forward Solution if requested
        if plot_object:
            self._log("True Object")
            self.visualisation.plot_refractive_index()

        if plot_forward:
            self._log("True Forward Solution")
            utrue_unblocked = self.convert_from_block_form(self.u_true)
            self.visualisation.plot_auto(utrue_unblocked[0], view="phase_amp", layout="single")
            self.visualisation.plot_auto(utrue_unblocked[int(self.num_angles/2)], view="phase_amp", layout="single")
            self.visualisation.plot_auto(utrue_unblocked[-1], view="phase_amp", layout="single")

        # Initialize residual
        residual = []
        nk = n0

        # Define probe
        if solve_probe:
            probesk = self.forward_model.initial_condition.return_probes(
                 probe_type="gaussian")
            # probesk = self.solve_for_probes(nk, plot_reverse)
            probesk_old = probesk.copy()
            
        else:
            probesk = None#self.probes_true

        for i in range(max_iters):
            time_start = time.time()
            # Compute the Forward Model
            uk, grad_Ak = self.compute_forward_model(nk, probes=probesk)

            grad_least_squares_real, grad_least_squares_imag = (
                self.compute_grad_least_squares(uk, grad_Ak)
            )

            # Compute RMSE of the current refractive index estimate
            error = self.true_exit_waves - uk[:, -self.block_size:]
            rel_rmse = np.sqrt(np.mean(np.abs(error) ** 2))

            residual.append(rel_rmse)
            # Check for convergence
            if residual[i] < tol:
                self._log(f"Converged in {i + 1} iterations.")
                break

            # Output the current iteration information
            if verbose:
                self._log(f"Iteration {i + 1}/{max_iters}")
                self._log(f"    RMSE: {residual[i]}")

            # Set direction for first iteration Conjugate Gradient
            if i == 0:
                dk_im = - grad_least_squares_imag
                grad_least_squares_imag_old = grad_least_squares_imag
                dk_re = - grad_least_squares_real
                grad_least_squares_real_old = grad_least_squares_real

            # Compute the step size
            if fixed_step_size is not None:
                alphak_re = -alpha0 / (i + 1)
                alphak_im = -alpha0 / (i + 1)
            else:
                alphak_re = self.compute_alphak(
                    uk, grad_Ak, grad_least_squares_real, dk_re)
                alphak_im = self.compute_alphak(
                    uk, 1j * grad_Ak, grad_least_squares_imag, dk_im)

            # Compute Product of step size and direction
            alphakdk_re = (alphak_re * dk_re).reshape((self.nz - 1, self.nx)).T
            alphakdk_im = (alphak_im * dk_im).reshape((self.nz - 1, self.nx)).T

            gradient_update = np.zeros((self.nx, self.nz), dtype=complex)
            gradient_update[:, 1:] = alphakdk_re + 1j * alphakdk_im

            # Get min and max values from the true sample space for color scaling
            if plot_gradient:
                self.visualisation.plot_single(
                    gradient_update, view="phase_amp", time="final",
                    filename=f"gradient_update.png",
                    title_left=f"gradient_update Phase",
                    title_right=f"gradient_update Amplitude",
                )


            # Combine with rotated object gradient
            if self.rotate90:
                uk_rot, grad_Ak_rot = self.compute_forward_model(np.rot90(nk, k=1), probes=probesk)

                # Compute the gradient of the least squares problem
                grad_least_squares_real_rot, grad_least_squares_imag_rot = (
                    self.compute_grad_least_squares(uk_rot, grad_Ak_rot,rotate=True)
                )

                if i == 0:
                    dk_im_rot = - grad_least_squares_imag_rot
                    grad_least_squares_imag_old_rot = grad_least_squares_imag_rot
                    dk_re_rot = - grad_least_squares_real_rot
                    grad_least_squares_real_old_rot = grad_least_squares_real_rot

                # Compute the step size
                if fixed_step_size is not None:
                    alphak_re_rot = -alpha0 / (i + 1)
                    alphak_im_rot = -alpha0 / (i + 1)
                else:
                    alphak_re_rot = self.compute_alphak(
                        uk_rot, grad_Ak_rot, grad_least_squares_real_rot, dk_re_rot)
                    alphak_im_rot = self.compute_alphak(
                        uk_rot, 1j * grad_Ak_rot, grad_least_squares_imag_rot, dk_im_rot)

                # Compute Product of step size and direction
                alphakdk_re_rot = (alphak_re_rot * dk_re_rot).reshape((self.nx - 1, self.nz)).T
                alphakdk_im_rot = (alphak_im_rot * dk_im_rot).reshape((self.nx - 1, self.nz)).T

                gradient_update_rot = np.zeros((self.nz, self.nx), dtype=complex)
                gradient_update_rot[:, 1:] = alphakdk_re_rot + 1j * alphakdk_im_rot

                # Get min and max values from the true sample space for color scaling
                if plot_gradient:
                    self.visualisation.plot_single(
                        gradient_update_rot, view="phase_amp", time="final",
                        filename=f"gradient_update_rot.png",
                        title_left=f"gradient_update_rot Phase",
                        title_right=f"gradient_update_rot Amplitude",
                    )

                gradient_update += np.rot90(gradient_update_rot, k=-1)


            # Update the current estimate of the refractive index of the object
            nk += gradient_update



            # Apply a low pass filter to nk in the z direction
            if low_pass_filter > 0.0:
                nk[:, 1:] = gaussian_filter1d(nk[:, 1:], sigma=low_pass_filter, axis=1)

            # Compute beta using Polak-Ribière and Fletcher-Reeves
            betak_re = self.compute_betak(grad_least_squares_real,
                                          grad_least_squares_real_old)
            betak_im = self.compute_betak(grad_least_squares_imag,
                                          grad_least_squares_imag_old)

            # Update direction
            dk_re = -grad_least_squares_real + betak_re * dk_re
            grad_least_squares_real_old = grad_least_squares_real
            dk_im = -grad_least_squares_imag + betak_im * dk_im
            grad_least_squares_imag_old = grad_least_squares_imag

            if self.rotate90:
                # Compute beta using Polak-Ribière and Fletcher-Reeves
                betak_re_rot = self.compute_betak(grad_least_squares_real_rot,
                                            grad_least_squares_real_old_rot)
                betak_im_rot = self.compute_betak(grad_least_squares_imag_rot,
                                            grad_least_squares_imag_old_rot)

                # Update direction
                dk_re_rot = -grad_least_squares_real_rot + betak_re_rot * dk_re_rot
                grad_least_squares_real_old_rot = grad_least_squares_real_rot
                dk_im_rot = -grad_least_squares_imag_rot + betak_im_rot * dk_im_rot
                grad_least_squares_imag_old_rot = grad_least_squares_imag_rot

            
            # Nesterov accelerated gradient descent update
            # if i == 0:
            #     v_prev = np.zeros_like(nk[:, 1:], dtype=complex)
            #     yk = nk[:, 1:]
            #     momentum = 0.9
            # else:
            #     momentum = 0.9
            #     yk = nk[:, 1:] + momentum * (nk[:, 1:] - nk_prev[:, 1:])

            # gradient = (grad_least_squares_real + 1j * grad_least_squares_imag).reshape((self.nz - 1, self.nx)).T
            # step_size = -alpha0 / (i + 1) #if fixed_step_size is not None else -1e-2  # fallback step size

            # v = momentum * v_prev + step_size * gradient
            # nk_prev = nk.copy()
            # nk[:, 1:] = yk + v
            # v_prev = v


            # Add sparsity regularization (L1) to the gradient
            if sparsity_lambda > 0.0:
                # Apply soft thresholding to real and imaginary parts separately
                nk_real = nk[:, 1:].real - self.sample_space.n_medium
                nk_imag = nk[:, 1:].imag
                nk[:, 1:] = self.sample_space.n_medium + self.soft_threshold(nk_real,
                                                    sparsity_lambda,
                                                    alphak_re) \
                    + 1j * self.soft_threshold(nk_imag,
                                               sparsity_lambda,
                                               alphak_im)
            
            # Compute source by solving the reverse problem
            if solve_probe:
                gamma = 0.5 
                probesk = (1 - gamma) * probesk_old + gamma * self.solve_for_probes(nk, plot_reverse)
                probesk_old = probesk.copy()
            
            time_end = time.time()
            if verbose:
                self._log(
                    f"    Iteration {i + 1} took {time_end - time_start:.2f} seconds.")

        return nk, self.convert_from_block_form(uk), residual

    def solve_for_probes(self, n, plot_reverse: Optional[bool] = False):
        """
        Solve the forward model for the probes in reversed time.
        """
        if not self.thin_sample and not self.full_system:
            iterative_lu_reverse = self.forward_model.construct_iterative_lu(
                        n=n, reverse=True)
            uk_reverse = self.forward_model.solve(
                n=n, iterative_lu=iterative_lu_reverse,
                initial_condition=self.true_exit_waves
                )
        else:
            # Solve the forward model in reversed time
            uk_reverse = self.forward_model.solve(
                n=n, reverse=True,
                initial_condition=self.true_exit_waves
                )
            
        if plot_reverse:
            self._log("    Reverse Solution for Reconstructed Object")
            # self.visualisation.plot(uk_reverse,
            #                         probe_index=-1)
            self.visualisation.plot(uk_reverse)
            # self.visualisation.plot(uk_reverse,
            #                         probe_index=0)
    
        # Convert to block form
        uk_reverse = self.convert_to_block_form(uk_reverse)

        # Select probes
        return uk_reverse[:, -self.block_size:]

    def soft_threshold(self, x, lam, step):
        """Soft thresholding operator for ISTA."""
        return np.sign(x) * np.maximum(np.abs(x) - lam * step, 0.0)
