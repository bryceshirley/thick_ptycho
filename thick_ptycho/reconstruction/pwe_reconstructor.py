# import numpy as np
# import scipy.sparse.linalg as spla

# import time
# from typing import Optional, List

# from thick_ptycho.forward_model.pwe.solvers import PWEIterativeLUSolver
# from thick_ptycho.forward_model.pwe.solvers import PWEFullPinTSolver
# from thick_ptycho.forward_model.pwe.solvers import PWEFullLUSolver
# from thick_ptycho.forward_model.pwe.operators.finite_differences import PWEForwardModel


# from .base_reconstructor import ReconstructorBase

import numpy as np
import scipy.sparse.linalg as spla

import time
from typing import Optional, List

from thick_ptycho.forward_model.pwe.solvers import PWEIterativeLUSolver
from thick_ptycho.forward_model.pwe.solvers import PWEFullPinTSolver
from thick_ptycho.forward_model.pwe.solvers import PWEFullLUSolver
from thick_ptycho.forward_model.pwe.operators.finite_differences import PWEForwardModel

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
        results_dir=None,
        use_logging=False,
        verbose=False,
        solver_type="iterative",
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
        # Store number of tomographic projections

        assert solver_type in {"full", "iterative"}, f"Invalid solver_type: {solver_type!r}"
        # Forward model selection
        SolverClass = PWEFullLUSolver if solver_type == "full" else PWEIterativeLUSolver
        self.forward_model = SolverClass(simulation_space, 
                                         self.ptycho_object,
                                         self.ptycho_probes,
                                        results_dir=results_dir)

        self._results_dir = results_dir
        self._log("Initializing Least Squares Solver...")



    def compute_forward_model(self, nk, probes: Optional[np.ndarray] = None):
        """Compute the forward model for the current object and gradient."""

        # Prepare the solver with the current refractive index
        self.forward_model.presolve_setup(n=nk)
        self.forward_model.presolve_setup(n=nk, mode='adjoint')

        # Compute Gradient of Ak wrt n
        grad_Ak = - self.forward_model.get_gradient(n=nk)

        uk = self.convert_to_vector_form(self.forward_model.solve(
            n=nk,
            probes=probes))


        return uk, grad_Ak
    
    def compute_grad_least_squares(self, exit_wave_error, uk, grad_A):
        """Compute the gradient of least squares problem."""

        # For speed: avoid recomputing sub-block size & flatten operation repeatedly
        projection_size = self.nx * (self.nz - 1)

        grad_real = np.zeros(projection_size, dtype=float)
        grad_imag = np.zeros(projection_size, dtype=float)
        
        idx = 0
        for proj_idx in range(self.num_projections):
            temp_mode = "adjoint_rotated" if proj_idx == 1 else "adjoint"
    
            for angle_idx in range(self.num_angles):
                for scan_idx in range(self.num_probes):

                    # 1) Solve adjoint model
                    backprop = self.forward_model._solve_single_probe(
                    scan_idx=scan_idx,
                    probe=None,
                    rhs_block=exit_wave_error[proj_idx, angle_idx, scan_idx],
                    mode=temp_mode
                    )

                    # 2) Flatten the spatial field block
                    backprop = backprop[:, :-1].T.ravel()

                    
                    # 3) Compute contribution of current frame
                    g_base = (grad_A @ uk[idx])

                    # 4) Accumulate gradient (real and imaginary parts separately)
                    grad_real -= (g_base.conj() * backprop).real
                    grad_imag -= ((1j * g_base).conj() * backprop).real

                    idx += 1

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
        projection_size = self.nx * (self.nz - 1)

        idx = 0
        for proj_idx in range(self.num_projections):
            temp_mode = "forward_rotated" if proj_idx == 1 else "forward"

            for angle_idx in range(self.num_angles):
                for scan_idx in range(self.num_probes):

                    # 1) Perturbation term: (grad_A @ u[idx])
                    perturb = grad_A @ u[idx]

                    # 2) Retrieve probe field once per loop
                    probe = self.ptycho_probes[angle_idx, scan_idx]

                    # 3) Solve forward model for delta_u
                    delta_u = self.forward_model._solve_single_probe(
                        scan_idx=scan_idx,
                        probe=probe,
                        rhs_block=perturb.reshape(self.nz - 1, self.nx).T,
                        mode=temp_mode
                    )

                    # (nz, nx) field → drop the first z-plane → flatten
                    delta_u = delta_u[:, 1:].T.ravel()

                    # 4) Only use last block_size elements
                    delta_p_i = - delta_u[-self.block_size:] @ d[-self.block_size:]


                    # 5) accumulate squared contribution
                    denominator += np.linalg.norm(delta_p_i)**2

                    idx += 1

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

    def reconstruct(
            self,
            max_iters=10,
            tol=1e-8,
            n_initial=None,
            fixed_step_size=None,
            solve_probe=False,
            sparsity_lambda=0.0):
        """Solve the least squares problem using conjugate gradient method with optional L1/L2/TV regularization."""

        # Initialize the fixed step size
        if fixed_step_size is not None:
            alpha0 = fixed_step_size

        # Initialize the refractive index
        if n_initial is not None:
            n0 = n_initial
        else:
            n0 = np.ones((self.block_size, self.simulation_space.nz), dtype=complex)*self.simulation_space.n_medium

        # Initialize residual
        residual = []
        nk = n0
        gradient_update = np.zeros((self.nx, self.nz), dtype=complex)

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

            exit_wave_error = self.get_error_in_exit_wave(uk)
            rel_rmse = np.sqrt(np.mean(np.abs(exit_wave_error[...,-1]) ** 2))
            residual.append(rel_rmse)
            # Check for convergence
            if residual[i] < tol:
                self._log(f"Converged in {i + 1} iterations.")
                break

            # Output the current iteration information
            if self.verbose:
                self._log(f"Iteration {i + 1}/{max_iters}")
                self._log(f"    RMSE: {residual[i]}")

            # Compute the gradient of the least squares problem
            grad_least_squares_real, grad_least_squares_imag = (
                self.compute_grad_least_squares(exit_wave_error, uk, grad_Ak)
            )

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
            gradient_update[:, 1:] = alphakdk_re + 1j * alphakdk_im

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
                probesk = (1 - gamma) * probesk_old + gamma * self.solve_for_probes(nk)
                probesk_old = probesk.copy()
            
            time_end = time.time()
            if self.verbose:
                self._log(
                    f"    Iteration {i + 1} took {time_end - time_start:.2f} seconds.")

        return nk, self.convert_to_tensor_form(uk), residual

    def solve_for_probes(self, n, plot_reverse: Optional[bool] = False):
        """
        Solve the forward model for the probes in reversed time.
        """
        assert not self.phase_retrieval, "Phase retrieval mode not supported for probe solving."
        self.forward_model.prepare_solver(n=n, mode='reverse')

        uk_reverse = self.forward_model.solve(n=n,mode='reverse',
                probes=self.data)
            
        if plot_reverse:
            self._log("    Reverse Solution for Reconstructed Object")
            self.simulation_space.viewer.plot_two_panels(uk_reverse[int(self.num_angles/2)], view="phase_amp")
    
        # Convert to block form
        uk_reverse = self.convert_to_vector_form(uk_reverse)

        # Select probes
        return uk_reverse[:, -self.block_size:]

    def soft_threshold(self, x, lam, step):
        """Soft thresholding operator for ISTA."""
        return np.sign(x) * np.maximum(np.abs(x) - lam * step, 0.0)
    


# class ReconstructorPWE(ReconstructorBase):
#     """
#     Nonlinear Conjugate Gradient (NLCG) reconstruction for PWE-based
#     thick-sample ptychography.

#     This class minimizes the least-squares loss:

#         L(n) = ½ || F(n) - data ||²₂,

#     where F(n) is the simulated exit wave (or its Fourier intensity)
#     produced by the forward model. Supports both full complex field
#     and intensity-only data (phase retrieval mode).

#     Optionally, for intensity-only data, the class can perform an
#     **initial Gerchberg–Saxton (GS)** phase retrieval step to estimate
#     the complex exit waves before nonlinear optimization. This step
#     alternates between the object and detector planes to impose both
#     intensity and support constraints, yielding an approximate phase
#     that often improves convergence of the subsequent NLCG stage.
#     """

#     def __init__(
#         self,
#         simulation_space,
#         data,
#         phase_retrieval=True,
#         results_dir=None,
#         use_logging=False,
#         verbose=False,
#         solver_type="iterative",
#     ):
#         super().__init__(
#             simulation_space=simulation_space,
#             data=data,
#             phase_retrieval=phase_retrieval,
#             results_dir=results_dir,
#             use_logging=use_logging,
#             verbose=verbose,
#             log_file_name="multislice_reconstruction_log.txt",
#         )
#         # Store number of tomographic projections

#         assert solver_type in {"full", "iterative"}, f"Invalid solver_type: {solver_type!r}"
#         # Forward model selection
#         SolverClass = PWEFullPinTSolver if solver_type == "full" else PWEIterativeLUSolver
#         self.forward_model = SolverClass(simulation_space, 
#                                          self.ptycho_object,
#                                          self.ptycho_probes,
#                                         results_dir=results_dir)

#         self._results_dir = results_dir
#         self._log("Initializing Least Squares Solver...")



#     def construct_forward_model(self, nk):
#         """Compute the forward model for the current object and gradient."""

#         # Prepare the solver with the current refractive index
#         self.forward_model.presolve_setup(n=nk)
#         self.forward_model.presolve_setup(n=nk, mode='adjoint')

    
#     def compute_forward_model(self, nk, probes: Optional[np.ndarray] = None):
#         """Compute the forward model for the current object and gradient."""

#         # Compute Gradient of Ak wrt n
#         self.grad_Ak = - self.forward_model.get_gradient(n=nk)

#         uk = np.zeros((self.total_projection_scans, self.projection_size), dtype=complex)
        
#         idx = 0
#         for angle_idx in range(self.num_angles):
#             for scan_idx in range(self.num_probes):
#                 # 1) Solve forward model
#                 uk[idx,:] = self.forward_model._solve_single_probe(
#                 scan_idx=scan_idx, n=nk,
#                 probe=probes[angle_idx, scan_idx] if probes is not None else None,
#                 mode="forward"
#                 )[:,1:].flatten()

#                 idx += 1

#         # Compute Gradient of Ak wrt n
#         grad_Ak = - self.forward_model.get_gradient(n=nk)
#         return uk, grad_Ak 

#     def compute_grad_least_squares(self, exit_wave_error, uk, grad_A):
#         """Compute the gradient of least squares problem."""

#         # For speed: avoid recomputing sub-block size & flatten operation repeatedly
#         grad_real = np.zeros(self.projection_size, dtype=float)
#         grad_imag = np.zeros(self.projection_size, dtype=float)
        
#         idx = 0
#         for angle_idx in range(self.num_angles):
#             for scan_idx in range(self.num_probes):

#                 # 1) Solve adjoint model
#                 backprop = self.forward_model._solve_single_probe(
#                 scan_idx=scan_idx,
#                 probe=None,
#                 rhs_block=exit_wave_error[angle_idx, scan_idx],
#                 mode="adjoint"
#                 )

#                 # 2) Flatten the spatial field block
#                 backprop = backprop[:, :-1].T.ravel()

                
#                 # 3) Compute contribution of current frame
#                 g_base = (grad_A @ uk[idx])

#                 # 4) Accumulate gradient (real and imaginary parts separately)
#                 grad_real -= (g_base.conj() * backprop).real
#                 grad_imag -= ((1j * g_base).conj() * backprop).real

#                 idx += 1

#         return grad_real, grad_imag


#     def compute_alphak(self, u, grad_A, grad_E, d):
#         """
#         Compute alpha_k calculation.

#         Parameters:
#         u (ndarray): Current solution vector, shape (num_probes, nx * (nz - 1)).
#         A (sp.spmatrix): Sparse matrix for the linear system.
#         grad_A (sp.spmatrix): Gradient of A wrt n.
#         d (ndarray): Descent direction vector.

#         Returns:
#         float: The computed denominator value.
#         """
#         denominator = 0.0

#         idx = 0
#         for angle_idx in range(self.num_angles):
#             for scan_idx in range(self.num_probes):

#                 # 1) Perturbation term: (grad_A @ u[idx])
#                 perturb = grad_A @ u[idx]

#                 # 2) Retrieve probe field once per loop
#                 probe = self.ptycho_probes[angle_idx, scan_idx]

#                 # 3) Solve forward model for delta_u
#                 delta_u = self.forward_model._solve_single_probe(
#                     scan_idx=scan_idx,
#                     probe=probe,
#                     rhs_block=perturb.reshape(self.nz - 1, self.nx).T,
#                     mode="forward"
#                 )

#                 # (nz, nx) field → drop the first z-plane → flatten
#                 delta_u = delta_u[:, 1:].T.ravel()

#                 # 4) Only use last block_size elements
#                 delta_p_i = - delta_u[-self.block_size:] @ d[-self.block_size:]


#                 # 5) accumulate squared contribution
#                 denominator += np.linalg.norm(delta_p_i)**2

#                 idx += 1

#         # Compute the numerator
#         numerator = np.vdot(d, grad_E)

#         # Compute the step size
#         alphak = - (numerator / denominator) if denominator > 0 else 0.0
#         return alphak

#     def compute_betak(self, grad_E, grad_E_old):
#         """Compute the beta_k value for the conjugate gradient method."""
#         grad_diff = grad_E - grad_E_old
#         grad_E_norm_sq = np.vdot(grad_E_old, grad_E_old)
#         betak_pr = np.vdot(grad_E, grad_diff) / grad_E_norm_sq
#         betak_fr = np.vdot(grad_E, grad_E) / grad_E_norm_sq
#         betak = min(betak_fr, max(betak_pr, 0))
#         return betak

#     def reconstruct(
#             self,
#             max_iters=10,
#             tol=1e-8,
#             n_initial=None,
#             fixed_step_size=None,
#             solve_probe=False,
#             sparsity_lambda=0.0):
#         """Solve the least squares problem using conjugate gradient method with optional L1/L2/TV regularization."""

#         # Initialize the fixed step size
#         if fixed_step_size is not None:
#             alpha0 = fixed_step_size

#         # Initialize the refractive index
#         if n_initial is not None:
#             n0 = n_initial
#         else:
#             n0 = np.ones((self.block_size, self.simulation_space.nz), dtype=complex)*self.simulation_space.n_medium

#         # Initialize residual
#         residual = []
#         nk = n0
#         gradient_update = np.zeros((self.nx, self.nz), dtype=complex)

#         # Define probe
#         if solve_probe:
#             probesk = self.forward_model.probes # Initial Simulated Probes
#             probesk_old = probesk.copy()
            
#         else:
#             probesk = None

#         # Conjugate Gradient direction info
#         direction_info_rot = None
#         direction_info = None

#         for i in range(max_iters):
#             time_start = time.time()

#             if fixed_step_size is not None:
#                 alpha = alpha0 / (i + 1)
#             else:
#                 alpha = None
#             # Construct forward model
#             self.construct_forward_model(nk=nk)

#             # Compute gradient update
#             projection_gradient_update, direction_info, step_sizes, error = self.compute_grad_update(
#                 nk, alpha=alpha, probesk=probesk, direction_info=direction_info)
#             gradient_update[:, 1:] = projection_gradient_update

#             # If tomographic, add rotated gradient contribution
#             if self.num_projections == 2:
#                 # Construct forward model
#                 self.construct_forward_model(nk=np.rot90(nk, k=1))

#                 # Compute gradient update for rotated angle
#                 gradient_update_rot, direction_info_rot, _, error_rot = self.compute_grad_update(
#                     np.rot90(nk, k=1), alpha=alpha, probesk=probesk, direction_info=direction_info_rot)

#                 gradient_update_rot = np.rot90(gradient_update_rot, k=-1)
#                 gradient_update[:, 1:] += gradient_update_rot

#                 error += error_rot

#             # Update the current estimate of the refractive index of the object
#             nk += gradient_update
       
#             # Compute and store residual norm
#             rel_rmse = np.sqrt(np.mean(np.abs(error) ** 2))
#             residual.append(rel_rmse)
#             # Check for convergence
#             if residual[i] < tol:
#                 self._log(f"Converged in {i + 1} iterations.")
#                 break

#             # Output the current iteration information
#             if self.verbose:
#                 self._log(f"Iteration {i + 1}/{max_iters}")
#                 self._log(f"    RMSE: {residual[i]}")
            
#             # Compute source by solving the reverse problem
#             if solve_probe:
#                 gamma = 0.5 
#                 probesk = (1 - gamma) * probesk_old + gamma * self.solve_for_probes(nk)
#                 probesk_old = probesk.copy()

#             # Add sparsity regularization (L1) to the gradient
#             if sparsity_lambda > 0.0:
#                 # Apply soft thresholding to real and imaginary parts separately
#                 nk_real = nk[:, 1:].real - self.simulation_space.n_medium
#                 nk_imag = nk[:, 1:].imag
#                 nk[:, 1:] = self.simulation_space.n_medium + self.soft_threshold(nk_real,
#                                                     sparsity_lambda,
#                                                     step_sizes[0]) \
#                     + 1j * self.soft_threshold(nk_imag,
#                                                 sparsity_lambda,
#                                                 step_sizes[1])

#             time_end = time.time()
#             if self.verbose:
#                 self._log(
#                     f"    Iteration {i + 1} took {time_end - time_start:.2f} seconds.")

#         return nk, residual

#     def solve_for_probes(self, n, plot_reverse: Optional[bool] = False):
#         """
#         Solve the forward model for the probes in reversed time.
#         """
#         assert not self.phase_retrieval, "Phase retrieval mode not supported for probe solving."
#         self.forward_model.prepare_solver(n=n, mode='reverse')

#         uk_reverse = self.forward_model.solve(n=n,mode='reverse',
#                 probes=self.data[0,...])
            
#         if plot_reverse:
#             self._log("    Reverse Solution for Reconstructed Object")
#             self.simulation_space.viewer.plot_two_panels(uk_reverse[int(self.num_angles/2)], view="phase_amp")
    
#         # Convert to block form
#         uk_reverse = self.convert_to_vector_form(uk_reverse)

#         # Select probes
#         return uk_reverse[:, -self.block_size:]

#     def soft_threshold(self, x, lam, step):
#         """Soft thresholding operator for ISTA."""
#         return np.sign(x) * np.maximum(np.abs(x) - lam * step, 0.0)

#     def compute_grad_update(self, nk, alpha: Optional[float]=None, 
#                             probesk: Optional[np.ndarray] = None,
#                             direction_info: Optional[tuple] = None):
#         """Compute the gradient update for the refractive index."""
#         # Compute the Forward Model
#         uk, grad_Ak = self.compute_forward_model(nk, probes=probesk)

#         exit_wave_error = self.apply_exit_wave_constraint(uk)

#         # Compute the gradient of the least squares problem
#         grad_least_squares_real, grad_least_squares_imag = (
#             self.compute_grad_least_squares(exit_wave_error, uk, grad_Ak)
#         )

#         # Set direction for first iteration Conjugate Gradient
#         if direction_info is None:
#             dk_im = - grad_least_squares_imag
#             grad_least_squares_imag_old = grad_least_squares_imag
#             dk_re = - grad_least_squares_real
#             grad_least_squares_real_old = grad_least_squares_real
#         else:
#             (grad_least_squares_real_old,
#              grad_least_squares_imag_old,
#              dk_re,
#              dk_im) = direction_info

#         # Compute the step size
#         if alpha is not None:
#             alphak_re = -alpha
#             alphak_im = -alpha
#         else:
#             alphak_re = self.compute_alphak(
#                 uk, grad_Ak, grad_least_squares_real, dk_re)
#             alphak_im = self.compute_alphak(
#                 uk, 1j * grad_Ak, grad_least_squares_imag, dk_im)

#         # Compute Product of step size and direction
#         alphakdk_re = (alphak_re * dk_re).reshape((self.nz - 1, self.nx)).T
#         alphakdk_im = (alphak_im * dk_im).reshape((self.nz - 1, self.nx)).T

#         # Compute beta using Polak-Ribière and Fletcher-Reeves
#         betak_re = self.compute_betak(grad_least_squares_real,
#                                         grad_least_squares_real_old)
#         betak_im = self.compute_betak(grad_least_squares_imag,
#                                         grad_least_squares_imag_old)

#         # Update direction
#         dk_re = -grad_least_squares_real + betak_re * dk_re
#         grad_least_squares_real_old = grad_least_squares_real
#         dk_im = -grad_least_squares_imag + betak_im * dk_im
#         grad_least_squares_imag_old = grad_least_squares_imag

#         # Outputs 
#         gradient_update = alphakdk_re + 1j * alphakdk_im
#         direction_info = (grad_least_squares_real_old, grad_least_squares_imag_old, dk_re, dk_im)
#         step_sizes = (alphak_re, alphak_im)
#         return gradient_update, direction_info, step_sizes, exit_wave_error[...,-1]