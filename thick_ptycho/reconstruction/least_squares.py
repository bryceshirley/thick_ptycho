import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from thick_ptycho.forward_model.solver import ForwardModel
from thick_ptycho.sample_space.sample_space import SampleSpace
from thick_ptycho.utils.visualisations import Visualisation
import time

from typing import Optional
from scipy.ndimage import gaussian_filter1d


class LeastSquaresSolver:
    """
    Class to solve the least squares problem for the paraxial wave equation.
    This class handles the forward model, computes the gradient, and solves
    the least squares problem using the conjugate gradient method.
    """

    def __init__(self, sample_space: SampleSpace, full_system_solver: bool = True):
        print("Initializing Least Squares Solver...")
        self.nx = sample_space.nx
        self.nz = sample_space.nz
        self.sample_space = sample_space
        self.wave_number = sample_space.k
        self.num_probes = sample_space.num_probes
        self.true_probe_type = sample_space.probe_type  # True Probe Type is contained in sample_space

        # Currently, only thick sample mode is supported
        self.thin_sample = False

        self.full_system = full_system_solver

        # Set up forward model, linear system and visualisation objects
        self.forward_model = ForwardModel(sample_space,
                                          full_system_solver=self.full_system,
                                          thin_sample=self.thin_sample)
        self.linear_system = self.forward_model.linear_system
        self.visualisation = Visualisation(sample_space)

        # Set block size (number of pixels in the exit wave)
        self.block_size = self.linear_system.block_size

        # Probe angle for steering the probe
        self._probe_angle = 0.0

        # Initialize LU decomposition
        self.lu = None
        self.iterative_lu = None
        self.adjoint_iterative_lu = None

        # Initialize the Forward Model matrix
        self.Ak = None

        # Define True Forward Solution
        print("Solving the true forward problem once to generate the dataset...")
        start_time = time.time()
        self.u_true = self.convert_to_block_form(self.forward_model.solve())
        self.probes_true = self.forward_model.linear_system.probes
        self.n_true = self.sample_space.n_true
        end_time = time.time()
        print(
            f"True Forward Solution computed in {end_time - start_time:.2f} seconds.")
        self.true_exit_waves = self.u_true[:, -self.block_size:]

        data = np.zeros((self.num_probes, self.block_size))


         # Create homogeneous forward model solution
        n_homogeneous = np.ones_like(self.n_true, dtype=complex)*self.sample_space.n_medium
        u_homogeneous = self.convert_to_block_form(self.forward_model.solve(n=n_homogeneous))
        exit_waves_homogeneous = u_homogeneous[:, -self.block_size:]
        diff_exit_waves = exit_waves_homogeneous - self.true_exit_waves

        diff_data = np.zeros((self.num_probes, self.block_size))

        for i in range(self.num_probes):
            # Extract the exit wave for each probe
            # Compute the squared FFT (intensity) of the exit wave for each probe
            exit_wave_fft = np.fft.fftshift(np.fft.fft(self.true_exit_waves[i, :]))
            data[i,:] = np.abs(exit_wave_fft) ** 2
            

            diff_exit_wave_fft = np.fft.fftshift(np.fft.fft(diff_exit_waves[i, :]))
            diff_data[i,:] = np.abs(diff_exit_wave_fft) ** 2

        plt.figure(figsize=(8, 4))
        plt.imshow(data, cmap='viridis', origin='lower')
        plt.colorbar(label='Intensity')
        plt.title('Exit Wave Squared FFT Intensity')
        plt.xlabel('x')
        plt.ylabel('Image #')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.imshow(diff_data, cmap='viridis', origin='lower')
        plt.colorbar(label='Intensity')
        plt.title(f'Differences in Exit Waves:\nFar Field Intensity (squared fourier transform)')
        plt.xlabel('x')
        plt.ylabel('Image #')
        plt.tight_layout()
        plt.show()


        phase = self.visualisation.compute_phase(self.true_exit_waves)
        amplitude = np.abs(self.true_exit_waves)
        diff_phase = self.visualisation.compute_phase(diff_exit_waves)
        diff_amplitude = np.abs(diff_exit_waves)

        # Mask low-magnitude values
        threshold = 1e-3  # Adjust based on your data
        phase[amplitude < threshold] = 0.0  # or np.nan for masking
        diff_phase[diff_amplitude < threshold] = 0.0  # or np.nan for masking

        fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        im0 = axs[0].imshow(phase, cmap='viridis', origin='lower')
        plt.colorbar(im0, ax=axs[0], label='Phase')
        axs[0].set_title('Exit Wave Phase')
        axs[0].set_ylabel('Image #')

        im1 = axs[1].imshow(amplitude, cmap='viridis', origin='lower')
        plt.colorbar(im1, ax=axs[1], label='Amplitude')
        axs[1].set_title('Exit Wave Amplitude')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('Image #')

        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        im0 = axs[0].imshow(diff_phase, cmap='viridis', origin='lower')
        plt.colorbar(im0, ax=axs[0], label='Phase')
        axs[0].set_title(f'Phase Differences in Exit Waves:\nHomogeneous vs. Inhomogeneous Media')
        axs[0].set_ylabel('Image #')

        im1 = axs[1].imshow(diff_amplitude, cmap='viridis', origin='lower')
        plt.colorbar(im1, ax=axs[1], label='Amplitude')
        axs[1].set_title(f'Amplitude Differences in Exit Waves:\nHomogeneous vs. Inhomogeneous Media')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('Image #')

        plt.tight_layout()
        plt.show()




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

    def compute_grad_least_squares(self, uk, grad_A):
        """Compute the gradient of least squares problem."""
        grad_real = np.zeros(self.nx * (self.nz - 1), dtype=float)
        grad_imag = np.zeros(self.nx * (self.nz - 1), dtype=float)

        # Preallocate zero vector
        exit_wave_error = self.compute_error_in_exit_wave(uk)

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

    def compute_error_in_exit_wave(self, uk):
        """
        Compute the error in the exit wave.

        Parameters:
        uk (ndarray): Current solution vector, shape (num_probes, nx * (nz-1)).

        Returns:
        ndarray: The exit wave error.
        """
        exit_wave_error = np.zeros_like(uk, dtype=complex)
        exit_waves = uk[:, -self.block_size:]
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
        u (ndarray): Input array to be converted. shape: (num_probes, nx, nz)

        Returns:
        ndarray: Block-formatted array.
        """
        # 2. Remove initial condition
        u = u[:, :, 1:]  # shape: (num_probes, block_size, nz - 1)

        # 3. Transpose axes
        u = u.transpose(0, 2, 1) # shape: (num_probes, nz - 1, block_size)

        # 4. Flatten last two dims
        # shape: (num_probes, block_size * (nz - 1))
        u = u.reshape(self.num_probes, -1)

        return u

    def convert_from_block_form(self, u):
        """
        Reverse the block flattening process.

        Parameters:
        u (ndarray): Flattened array of shape (num_probes, block_size * (nz - 1))

        Returns:
        ndarray: Unflattened array of shape (num_probes, block_size, nz - 1)
        """
        # Step 1: Reshape to (num_probes, nz - 1, block_size)
        reshaped = u.reshape(self.num_probes, self.nz-1, self.block_size)

        # Step 2: Transpose to (num_probes, nx, nz - 1)
        return reshaped.transpose(0, 2, 1)

    def solve(
            self,
            n_initial=None,
            max_iters=10,
            tol=1e-8,
            plot_object=False,
            plot_forward=False,
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


        true_phase = self.visualisation.compute_phase(self.sample_space.n_true)
        true_amplitude = np.abs(self.sample_space.n_true)
        vmin_phase = np.min(true_phase)
        vmax_phase = np.max(true_phase)
        vmin_amp = np.min(true_amplitude)
        vmax_amp = np.max(true_amplitude)

        # Initialize the refractive index
        if n_initial is not None:
            n0 = n_initial
        else:
            n0 = np.ones((self.block_size, self.nz), dtype=complex)*self.sample_space.n_medium

        # Output True Object and Forward Solution if requested
        if plot_object:
            print("True Object")
            self.visualisation.plot(self.n_true,
                                    title='True Object')
        if plot_forward:
            print("True Forward Solution")
            self.visualisation.plot(self.convert_from_block_form(self.u_true),
                                    probe_index=-1)
            self.visualisation.plot(self.convert_from_block_form(self.u_true))
            self.visualisation.plot(self.convert_from_block_form(self.u_true),
                                    probe_index=0)

        # Initialize residual
        residual = []
        nk = n0
        rel_rmse_denominator = np.sqrt(np.mean(np.abs(n0) ** 2))

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

            # Compute the gradient of the least squares problem
            grad_least_squares_real, grad_least_squares_imag = (
                self.compute_grad_least_squares(uk, grad_Ak)
            )
            # Compute RMSE of the current refractive index estimate
            error = self.true_exit_waves - uk[:, -self.block_size:]
            rel_rmse = np.sqrt(np.mean(np.abs(error) ** 2))

            residual.append(rel_rmse)
            # Check for convergence
            if residual[i] < tol:
                print(f"Converged in {i + 1} iterations.")
                break

            # Output the current iteration information
            if verbose:
                print(f"Iteration {i + 1}/{max_iters}")
                print(f"    RMSE: {residual[i]}")
            if i == 0 or (i + 1) % 5 == 0:
                if plot_object:
                    print("    Reconstructed Object")
                    fig, axs = plt.subplots(1, 2, figsize=(16, 5))

                    # Get min and max values from the true sample space for color scaling
                    phase = self.visualisation.compute_phase(nk)
                    amplitude = np.abs(nk)
                    axs[0].set_title("Phase of Reconstructed Sample Space")
                    im0 = axs[0].imshow(phase, origin='lower', aspect='auto', cmap='viridis', vmin=vmin_phase, vmax=vmax_phase)
                    axs[0].set_xlabel('z (pixels)')
                    axs[0].set_ylabel('x (pixels)')
                    fig.colorbar(im0, ax=axs[0], label='Phase (radians)')

                    # Amplitude subplot
                    axs[1].set_title("Amplitude of Reconstructed Sample Space")
                    im1 = axs[1].imshow(amplitude, origin='lower', aspect='auto', cmap='viridis', vmin=vmin_amp, vmax=vmax_amp)
                    axs[1].set_xlabel('z (pixels)')
                    axs[1].set_ylabel('x (pixels)')
                    fig.colorbar(im1, ax=axs[1], label='Amplitude')
                if plot_forward:
                    print("    Forward Solution for Reconstructed Object")
                    self.visualisation.plot(self.convert_from_block_form(uk),
                                            probe_index=-1)
                    self.visualisation.plot(self.convert_from_block_form(uk))
                    self.visualisation.plot(self.convert_from_block_form(uk),
                                            probe_index=0)

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

            # Update the current estimate of the refractive index of the object
            nk[:, 1:] = nk[:, 1:] + alphakdk_re + 1j * alphakdk_im

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
                print(
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
            print("    Reverse Solution for Reconstructed Object")
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
