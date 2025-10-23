import numpy as np
import scipy.sparse.linalg as spla

import time
from typing import Optional, List

from thick_ptycho.thick_ptycho.forward_model.pwe_solver_iterative import ForwardModelPWEIterative
from thick_ptycho.thick_ptycho.forward_model.pwe_solver_full import ForwardModelPWEFull
from thick_ptycho.thick_ptycho.simulation import simulation_space
from thick_ptycho.thick_ptycho.simulation import ptycho_object
from thick_ptycho.thick_ptycho.simulation.ptycho_object import SampleSpace
from thick_ptycho.utils.utils import setup_log
from thick_ptycho.utils.visualisations import Visualisation

from .base import ReconstructorBase

# Todo make pwe solvers allow for 90 degree rotations


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
        simulation_space: simulation_space,
        ptycho_object: ptycho_object,
        ptycho_probes: np.ndarray,
        results_dir=None,
        use_logging=True,
        verbose=False,
        solver_type: str = "iterative",
        rotated_90: bool = False,
    ):
        super().__init__(
            simulation_space=simulation_space,
            ptycho_object=ptycho_object,
            ptycho_probes=ptycho_probes,
            results_dir=results_dir,
            use_logging=use_logging,
            verbose=verbose,
        )
        # Store number of tomographic rotations
        if rotated_90:
            self.num_rotations = 2
        else:
            self.num_rotations = 1

        assert solver_type in {"full", "iterative"}, f"Invalid solver_type: {solver_type!r}"
        # Forward model selection
        SolverClass = ForwardModelPWEFull if solver_type == "full" else ForwardModelPWEIterative
        self.forward_model = SolverClass(simulation_space, ptycho_object, ptycho_probes,rotated_90=rotated_90,
                                        results_dir=results_dir,
                                        use_logging=use_logging,
                                        verbose=verbose)
        # Set block size (number of pixels in the exit wave)
        self.block_size = self.forward_model.block_size


        # ---- logging/verbosity setup ----
        self._log = setup_log(results_dir,log_file_name="reconstruction_log.txt",
                               use_logging=use_logging, verbose=verbose)
        self._results_dir = results_dir
        self._log("Initializing Least Squares Solver...")
        self.visualisation = Visualisation(simulation_space, results_dir=results_dir)
        
        self.data = None
        self.phase_retrieval = False


    def compute_forward_model(self, nk, probes: Optional[np.ndarray] = None):
        """Compute the forward model for the current object and gradient."""

        # Prepare the solver with the current refractive index
        self.forward_model.prepare_solver(n=nk)
        self.forward_model.prepare_solver(n=nk, mode='adjoint')

        # Compute Gradient of Ak wrt n
        grad_Ak = - self.forward_model.get_gradient(n=nk)

        uk = self.convert_to_block_form(self.forward_model.solve(
            n=nk,
            initial_condition=probes))

        return uk, grad_Ak

    def compute_grad_least_squares(self, uk, grad_A):
        """Compute the gradient of least squares problem."""
        grad_real = np.zeros(self.nx * (self.nz - 1), dtype=float)
        grad_imag = np.zeros(self.nx * (self.nz - 1), dtype=float)

        # Compute exit wave error
        exit_wave_error = self.apply_exit_wave_constraint(uk)


        for i in range(self.num_probes*self.num_probe_angles*self.num_rotations):
            error_backpropagation = self.forward_model._solve_single_probe(
                    rhs_block=exit_wave_error[i, :], mode='adjoint') 
            error_backpropagation = error_backpropagation[:, :-1].transpose().ravel()
    
            g_base = (grad_A @ uk[i, :])
            grad_real -= (g_base.conj() * error_backpropagation).real
            grad_imag -= ((1j * g_base).conj() * error_backpropagation).real

        return grad_real, grad_imag


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

        for i in range(self.num_probes*self.num_angles*self.num_rotations):
            # Compute the perturbation for each probe
            perturbation = grad_A @ u[i, :]

            delta_u = self.forward_model._solve_single_probe(
                rhs_block=perturbation)
            delta_u = delta_u[:, 1:].transpose().flatten()

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

    def convert_from_block_form(self,u):
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

    def convert_to_block_form(self, u):
        """
        Convert the input array to block form.

        Parameters:
        u (ndarray): Input array to be converted. shape: (ang_number, num_probes, nx, nz)

        Returns:
        ndarray: Block-formatted array. (ang_number*num_probes, nx*nz)
        """
        # 2. Remove initial condition
        dims = u.shape
        num_angles,num_probes = dims[0], dims[1]
        u = u[:, :, :, 1:]  # shape: (num_angles, num_probes, block_size, nz - 1)

        # 3. Transpose axes
        u = u.transpose(0, 1, 3, 2) # shape: (ang_number, num_probes, nz - 1, block_size)

        # 4. Flatten last two dims
        # shape: (num_angles*num_probes, block_size * (nz - 1))
        u = u.reshape(num_angles*num_probes, -1)

        return u

    def reconstruct(
            self,
            data: np.ndarray,
            phase_retrieval: bool = False,
            n_initial=None,
            max_iters=10,
            tol=1e-8,
            plot_forward=False,
            plot_gradient=False,
            plot_reverse=False,
            fixed_step_size=None,
            verbose=True,
            solve_probe=False,
            sparsity_lambda=0.0):
        """Solve the least squares problem using conjugate gradient method with optional L1/L2/TV regularization."""
        # Store data and phase retrieval mode
        self.data = data
        self.phase_retrieval = phase_retrieval

        # Initialize the fixed step size
        if fixed_step_size is not None:
            alpha0 = fixed_step_size

        # Initialize the refractive index
        if n_initial is not None:
            n0 = n_initial
        else:
            n0 = np.ones((self.block_size, self.nz), dtype=complex)*self.simulation_space.n_medium

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
            probesk = self.forward_model.probes # Initial Simulated Probes
            probesk_old = probesk.copy()
            
        else:
            probesk = None

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

            # Update the current estimate of the refractive index of the object
            nk += gradient_update

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

            # Add sparsity regularization (L1) to the gradient
            if sparsity_lambda > 0.0:
                # Apply soft thresholding to real and imaginary parts separately
                nk_real = nk[:, 1:].real - self.simulation_space.n_medium
                nk_imag = nk[:, 1:].imag
                nk[:, 1:] = self.simulation_space.n_medium + self.soft_threshold(nk_real,
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
                
            self.visualisation.plot_refractive_index(
                nk, title=f"Reconstructed Sample Space")
    

        return nk, self.convert_from_block_form(uk), residual

    def solve_for_probes(self, n, plot_reverse: Optional[bool] = False):
        """
        Solve the forward model for the probes in reversed time.
        """
        assert not self.phase_retrieval, "Phase retrieval mode not supported for probe solving."
        self.forward_model.prepare_solver(n=n, mode='reverse')

        uk_reverse = self.forward_model.solve(n=n,mode='reverse',
                initial_condition=self.data)
            
        if plot_reverse:
            self._log("    Reverse Solution for Reconstructed Object")
            self.visualisation.plot_auto(uk_reverse[int(self.num_angles/2)], view="phase_amp", layout="single")
    
        # Convert to block form
        uk_reverse = self.convert_to_block_form(uk_reverse)

        # Select probes
        return uk_reverse[:, -self.block_size:]

    def soft_threshold(self, x, lam, step):
        """Soft thresholding operator for ISTA."""
        return np.sign(x) * np.maximum(np.abs(x) - lam * step, 0.0)
    

    # def setup(self):
    #     # --- Precompute the “true” forward solution (no plots here) ---
    #     self._log("Solving the true forward problem to generate the dataset...")
    #     start_time = time.time()
    #     u_true = self.forward_model.solve()
    #     self.u_true = self.convert_to_block_form(u_true)
    #     self.data = None
    #     self.probes_true = self.forward_model.pwe_finite_differences.probes
    #     self.n_true = self.simulation_space.n_true
    #     self.true_exit_waves = self.u_true[:, -self.block_size:]
    #     self.visualize_data()

    #     # If square grid, also precompute the 90 degree rotation case once
    #     self.u_true_rot = None
    #     self.true_exit_waves_rot = None
    #     self.data_rot = None
    #     if self.rotate90:
    #         self.n_true_rot = np.rot90(self.n_true, k=1)
    #         self.u_true_rot, _ = self.compute_forward_model(self.n_true_rot)
    #         self.true_exit_waves_rot = self.u_true_rot[:, -self.block_size:]
    #         self.visualize_data(rotate=True)

    #     end_time = time.time()
    #     self._log(f"True Forward Solution computed in {end_time - start_time:.2f} seconds.")

    #     self._log(f"Angle {self.probe_angles_list[0]}")
    #     self.visualisation.plot_auto(u_true[0], view="phase_amp", layout="single")
    #     if self.num_angles > 1:
    #         self._log(f"Angle {self.probe_angles_list[-1]}")
    #         self.visualisation.plot_auto(u_true[-1], view="phase_amp", layout="single")

    # def visualize_data(self, rotate: bool = False) -> None:
    #     """
    #     Visualize FFT intensities, phases, and amplitudes of the exit waves, plus
    #     differences versus a homogeneous medium. Optionally uses the precomputed
    #     rotated forward model (if available).
    #     """

    #     # Select exit waves according to the orientation
    #     if rotate and (self.true_exit_waves_rot is not None):
    #         exit_waves = self.true_exit_waves_rot
    #         n_for_homog_shape = self.n_true_rot.shape
    #         title_prefix = "(rotated)"

    #         n_true = self.n_true_rot
    #     else:
    #         if rotate:
    #             self._log("Warning: rotate=True requested, but rotated forward model "
    #                   "was not precomputed (requires nx == nz). Using non-rotated data.")
    #         exit_waves = self.true_exit_waves
    #         n_for_homog_shape = self.n_true.shape
    #         title_prefix = ""
    #         n_true = self.n_true


    #     self._log("Plot True Object")
    #     self.visualisation.plot_single(n_true, view="phase_amp", time="final",
    #                                    filename=f"{'rot_' if title_prefix else ''}true_object.png")

    #     # Compute homogeneous forward solution (same shape/orientation as selected case)
    #     n_homogeneous = np.ones(n_for_homog_shape, dtype=complex) * self.simulation_space.n_medium
    #     if title_prefix:
    #         # If we're visualizing the rotated case, make sure we pass the rotated
    #         # n to the forward model (keep other settings identical).
    #         u_homogeneous = self.convert_to_block_form(self.forward_model.solve(n=n_homogeneous))
    #     else:
    #         u_homogeneous = self.convert_to_block_form(self.forward_model.solve(n=n_homogeneous))

    #     exit_waves_homogeneous = u_homogeneous[:, -self.block_size:]
    #     diff_exit_waves = exit_waves_homogeneous - exit_waves

    #     # ---------- FFT-squared intensities ----------
    #     data = np.zeros((self.num_probes * self.num_angles, self.block_size))
    #     diff_data = np.zeros_like(data)

    #     for i in range(self.num_probes * self.num_angles):
    #         data[i, :] = np.square(np.abs(np.fft.fft(exit_waves[i, :])))

    #         if self.poisson_noise:
    #             data[i, :] = np.random.poisson(data[i, :])

    #         diff_exit_wave_fft = np.fft.fft(diff_exit_waves[i, :])
    #         diff_data[i, :] = np.square(np.abs(diff_exit_wave_fft))

    #     if rotate:
    #         self.data_rot = data
    #     else:
    #         self.data = data

    #     if self._results_dir:
    #         fig = plt.figure(figsize=(8, 4))
    #         plt.imshow(data, cmap='viridis', origin='lower')
    #         plt.colorbar(label='Intensity')
    #         plt.title(f'Exit Wave Squared FFT Intensity {title_prefix}'.strip())
    #         plt.xlabel('x'); plt.ylabel('Image #'); plt.tight_layout()
    #         fig.savefig(os.path.join(self._results_dir, f'true_fft_intensity{ "_rot" if title_prefix else ""}.png'),
    #                     bbox_inches="tight")
    #         plt.close(fig)

    #         fig = plt.figure(figsize=(8, 4))
    #         plt.imshow(diff_data, cmap='viridis', origin='lower')
    #         plt.colorbar(label='Intensity')
    #         plt.title(f'Differences in Exit Waves {title_prefix}:\nFar Field Intensity'.strip())
    #         plt.xlabel('x'); plt.ylabel('Image #'); plt.tight_layout()
    #         fig.savefig(os.path.join(self._results_dir, f'true_fft_intensity_diff{ "_rot" if title_prefix else ""}.png'),
    #                     bbox_inches="tight")
    #         plt.close(fig)

    #     # ---------- Phase & Amplitude (and differences) ----------
    #     self.visualisation.plot_single(
    #             exit_waves, view="phase_amp", time="final",
    #             filename=f"exit_phase_amp{ '_rot' if title_prefix else ''}.png",
    #             title_left=f"Exit Wave Phase {title_prefix}".strip(),
    #             title_right=f"Exit Wave Amplitude {title_prefix}".strip(),
    #             xlabel_left="x",  ylabel_left="Image #",
    #             xlabel_right="x", ylabel_right="Image #",
    #         )

    #     self.visualisation.plot_single(
    #             diff_exit_waves, view="phase_amp", time="final",
    #             filename=f"exit_phase_amp_diff{ '_rot' if title_prefix else ''}.png",
    #             title_left=f"Phase Differences in Exit Waves {title_prefix}:\nHomogeneous vs. Inhomogeneous Media".strip(),
    #             title_right=f"Amplitude Differences in Exit Waves {title_prefix}:\nHomogeneous vs. Inhomogeneous Media".strip(),
    #             xlabel_left="x",  ylabel_left="Image #",
    #             xlabel_right="x", ylabel_right="Image #",
    #         )

