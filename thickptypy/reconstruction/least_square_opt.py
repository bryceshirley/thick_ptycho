import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from thickptypy.forward_model.solver import ForwardModel
from thickptypy.sample_space.sample_space import SampleSpace
from thickptypy.utils.visualisations import Visualisation
import time


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

        # Initialize LU decomposition for the full system solver
        self.lu = None
        self.Ak = None

        # Define True Forward Solution
        print("Solving the true forward problem once to generate the dataset...")
        start_time = time.time()
        self.u_true = self.convert_to_block_form(self.forward_model.solve())
        end_time = time.time()
        print(
            f"True Forward Solution computed in {end_time - start_time:.2f} seconds.")
        self.true_exit_waves = self.u_true[:, -self.block_size:]
        

    def compute_forward_model(self, nk):
        """Compute the forward model for the current object and gradient."""
        self.Ak = self.forward_model.return_forward_model_matrix(n=nk)
        
        if not self.thin_sample and self.full_system:
            self.lu = spla.splu(self.Ak)


        grad_Ak = - self.linear_system.setup_inhomogeneous_forward_model(
            n=nk, grad=True)
        if self.lu is not None:
            uk = self.convert_to_block_form(self.forward_model.solve(n=nk, lu=self.lu))
        else:
            uk = self.convert_to_block_form(self.forward_model.solve(n=nk))

        return uk, grad_Ak

    def compute_grad_least_squares(self, uk, grad_A):
        """Compute the gradient of least squares problem using efficient linear algebra."""
        # Compute the exit wave error for all probes
        exit_wave_error = self.compute_error_in_exit_wave(uk)  # shape: (num_probes, nx*(nz-1))

        # Preallocate arrays for error backpropagation
        if self.lu is not None:
            # Use LU solve for all probes at once (if possible)
            # Stack errors as columns for batch solve
            error_backprop = np.column_stack([
                self.lu.solve(exit_wave_error[i, :], trans='H')
                for i in range(self.num_probes)
            ])
        else:
            # Use spsolve for each probe
            error_backprop = np.column_stack([
                spla.spsolve(self.Ak.conj().T, exit_wave_error[i, :])
                for i in range(self.num_probes)
            ])
        # error_backprop shape: (nx*(nz-1), num_probes)

        # Compute grad_A @ uk for all probes at once
        grad_A_uk = grad_A @ uk.T  # shape: (nx*(nz-1), num_probes)

        # Compute the real and imaginary gradients using vectorized operations
        # Wirtinger derivative: real part and imaginary part
        grad_real = -np.real(np.sum(np.conj(grad_A_uk) * error_backprop, axis=1))
        grad_imag = -np.real(np.sum(np.conj(1j * grad_A_uk) * error_backprop, axis=1))

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
        u (ndarray): Current solution vector, shape (num_probes, nx * (nz-1)).
        A (sp.spmatrix): Sparse matrix for the linear system.
        grad_A (sp.spmatrix): Gradient of A wrt n.
        d (ndarray): Descent direction vector.

        Returns:
        float: The computed denominator value.
        """
        denominator = 0.0

        for i in range(self.num_probes):
            # Compute the perturbation for each probe
            if self.lu is not None:
                delta_u = self.lu.solve(grad_A @ u[i, :])
            else:
                delta_u = spla.spsolve(self.Ak, grad_A @ u[i, :])

            # Only use last block_size elements (final slice)
            delta_p_i = - delta_u[-self.block_size:] @ d[-self.block_size:]

            # Accumulate squared norm
            denominator += np.linalg.norm(delta_p_i)**2

        # Compute the numerator
        numerator = np.vdot(d, grad_E)

        # Compute the step size
        alphak = - (numerator / denominator) if denominator != 0 else 0.0
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
        """
        Convert the input array to block form.

        Parameters:
        u (ndarray): Input array to be converted. shape: (num_probes, nx, nz)

        Returns:
        ndarray: Block-formatted array.
        """
        # 2. Remove initial condition
        u = u[:, :, 1:]                       # shape: (num_probes, nx, nz - 1)

        # 3. Transpose axes
        u = u.transpose(0, 2, 1)              # shape: (num_probes, nz - 1, nx)

        # 4. Flatten last two dims
        # shape: (num_probes, nx * (nz - 1))
        u = u.reshape(self.num_probes, -1)

        return u

    def convert_from_block_form(self, u):
        """
        Reverse the block flattening process.

        Parameters:
        u (ndarray): Flattened array of shape (num_probes, nx * (nz - 1))

        Returns:
        ndarray: Unflattened array of shape (num_probes, nx, nz - 1)
        """
        # Step 1: Reshape to (num_probes, nz - 1, nx)
        reshaped = u.reshape(self.num_probes, self.nz-1, self.nx)

        # Step 2: Transpose to (num_probes, nx, nz - 1)
        return reshaped.transpose(0, 2, 1)

    def solve(
            self,
            n_initial=None,
            max_iters=10,
            tol=1e-8,
            plot_object=False,
            plot_forward=False,
            fixed_step_size=None,
            verbose=True,
            sparsity_lambda=0.0,
            l2_lambda=0.0,
            tv_lambda=0.0):
        """Solve the least squares problem using conjugate gradient method with optional L1/L2/TV regularization."""

        # Initialize the fixed step size
        if fixed_step_size is not None:
            alpha0 = fixed_step_size

        # Initialize the refractive index
        if n_initial is not None:
            nk = n_initial
        else:
            nk = np.ones((self.nx, self.nz), dtype=complex)

        # Output True Object and Forward Solution if requested
        if plot_object:
            print("True Object")
            self.visualisation.plot(self.sample_space.n_true,
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

        for i in range(max_iters):
            time_start = time.time()
            # Compute RMSE of the current refractive index estimate
            residual.append(np.sqrt(np.mean(np.abs(
                nk - self.sample_space.n_true) ** 2)))
            # Check for convergence
            if residual[i] < tol:
                print(f"Converged in {i + 1} iterations.")
                break

            # Compute the Forward Model and Gradient of Least Squares
            uk, grad_Ak = self.compute_forward_model(nk)
            grad_least_squares_real, grad_least_squares_imag = (
                self.compute_grad_least_squares(uk, grad_Ak)
            )

            # Output the current iteration information
            if verbose:
                print(f"Iteration {i + 1}/{max_iters}")
                print(f"    RMSE: {residual[i]}")
            if (i + 1) % 5 == 0 or i == 0:
                if plot_object:
                    print("    Reconstructed Object")
                    self.visualisation.plot(
                        nk,
                        title=f"Reconstructed Object (Iteration {i + 1})"
                    )
                if plot_forward:
                    print("    Forward Solution for Reconstructed Object")
                    self.visualisation.plot(self.convert_from_block_form(uk),
                                            probe_index=-1)
                    self.visualisation.plot(self.convert_from_block_form(uk))
                    self.visualisation.plot(self.convert_from_block_form(uk),
                                            probe_index=0)
                    
            # Add sparsity regularization (L1) to the gradient
            if sparsity_lambda > 0.0:
                # Only regularize the unknown region (exclude first column)
                grad_least_squares_real += sparsity_lambda * np.sign(nk[:, 1:].real).T.reshape((-1,))
                grad_least_squares_imag += sparsity_lambda * np.sign(nk[:, 1:].imag).T.reshape((-1,))

            # Add L2 regularization to the gradient
            if l2_lambda > 0.0:
                grad_least_squares_real += (2 * l2_lambda * nk[:, 1:].real).T.reshape((-1,))
                grad_least_squares_imag += (2 * l2_lambda * nk[:, 1:].imag).T.reshape((-1,))

            # Add TV regularization to the gradient
            if tv_lambda > 0.0:
                tv_grad_real = self.tv_grad(nk[:, 1:].real).T.reshape((-1,))
                tv_grad_imag = self.tv_grad(nk[:, 1:].imag).T.reshape((-1,))
                grad_least_squares_real += tv_lambda * tv_grad_real
                grad_least_squares_imag += tv_lambda * tv_grad_imag

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
            time_end = time.time()
            if verbose:
                print(
                    f"    Iteration {i + 1} took {time_end - time_start:.2f} seconds.")
        return nk, self.convert_from_block_form(uk), residual
    
    def tv_grad(self, x):
            """Compute isotropic TV gradient for 2D array x."""
            grad = np.zeros_like(x)
            # Pad for boundary conditions
            x_pad = np.pad(x, ((0, 1), (0, 1)), mode='edge')
            dx = x_pad[:-1, 1:] - x_pad[:-1, :-1]
            dy = x_pad[1:, :-1] - x_pad[:-1, :-1]
            norm = np.sqrt(dx**2 + dy**2 + 1e-12)
            # Compute divergence
            dx_div = dx / norm
            dy_div = dy / norm
            grad[:-1, :] += dx_div[:-1, :]
            grad[:, :-1] += dy_div[:, :-1]
            grad[:-1, :] -= dx_div[1:, :]
            grad[:, :-1] -= dy_div[:, 1:]
            return grad
