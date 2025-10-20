# Import the paraxial solver
import matplotlib.pyplot as plt
import numpy as np
import pytest
from thick_ptycho.sample_space.sample_space import SampleSpace
from thick_ptycho.forward_model.solver import ForwardModel
<<<<<<< HEAD

def u_nm(a, n):
    """Returns the exact solution for a given n in 1D."""
    return lambda x, z: np.exp(-a * (n**2) * (np.pi**2) * z) * np.cos(n * np.pi * x)
=======
from thick_ptycho.utils.visualisations import Visualisation
>>>>>>> main


def compute_error(nx, nz, thin_sample, full_system_solver):
    """Computes the Frobenius norm of the error between the exact and computed solutions in 1D."""
    bc_type = "neumann"  # Neumann boundary condition
    probe_type = "neumann_test"  # Sum of cosines
    wave_number = 100

    # Set continuous space limits
    xlims = [0, 1]
    zlims = [0, 2]

    continuous_dimensions = [
        xlims,
        zlims
    ]

    # Set detector shape, probe radius, step size, and no of scan points in each coordinate (All in pixels)
    probe_dimensions = [nx]
    discrete_dimensions = [
        nx,
        nz
    ]
    scan_points = 1
    step_size = 0

    sample_space = SampleSpace(
        continuous_dimensions,
        discrete_dimensions,
        probe_dimensions,
        scan_points,
        step_size,
        bc_type,
        probe_type,
        wave_number,
    )

    forward_model = ForwardModel(sample_space,
                                 thin_sample=thin_sample,
                                 full_system_solver=full_system_solver)

    # Solve the experiment
    solution = forward_model.solve()[0, :, :]

    a = 1j / (2 * wave_number)

    # Define grid points
    x = np.linspace(xlims[0], xlims[1], nx)
    z = np.linspace(zlims[0], zlims[1], nz)
    X, Z = np.meshgrid(x, z, indexing='ij')

    # Compute the exact solution
    exact_solution = u_nm(a, 1)(X, Z) + 0.5 * u_nm(a, 2)(X, Z) + 0.2 * u_nm(a, 5)(X, Z)

    # Compute the relative RMSE
    return exact_solution, solution

@pytest.mark.parametrize("thin_sample", [True, False], ids=["Thin", "Thick"])
@pytest.mark.parametrize("full_system_solver", [True, False], ids=["AllAtOnce", "Iterations"])
def test_error(thin_sample, full_system_solver, request):
    """Test that the error norm decreases as the grid resolution increases."""

    nx_values = [16, 32, 64, 128, 256, 512, 1024]
    nz_values = []
    # rmse_errors = []
    inf_norms = []
    bc_type = "Neumann 2D"

    print(f"\n=== CONVERGENCE STUDY: {bc_type.upper()} BOUNDARY CONDITIONS ===")
    print(f"thin_sample: {thin_sample}, full_system_solver: {full_system_solver}\n")
    for i, nx in enumerate(nx_values):
        nz = nx
        nz_values.append(nz)

        print(f"Computing solution for nx={nx}, nz={nz}")
        exact_solution, solution = compute_error(nx, nz, thin_sample, full_system_solver)
        error = solution - exact_solution
        error_norm = np.max(np.abs(error))

        # rmse = np.sqrt(np.mean(np.abs(error)**2))
        # error_norm = np.linalg.norm(error) / np.linalg.norm(exact_solution)

        # Store errors for plotting
        # rmse_errors.append(rmse)
        inf_norms.append(error_norm)
    
    # Print convergence rates
    print(f"\nConvergence Analysis ({bc_type} BC):")
    print("nx\tnz\tRate\tInfinity Norm")
    print("-" * 40)
    for i in range(len(nx_values)):
        if i > 0:
            ratio = inf_norms[i-1] / inf_norms[i]
            rate = np.log2(ratio)
            print(f"{nx_values[i]}\t{nz_values[i]}\t{rate:.2f}\t{inf_norms[i]:.6f}")
        else:
            print(f"{nx_values[i]}\t{nz_values[i]}\t-\t{inf_norms[i]:.6f}")
    
    # Check if plotting is enabled
    if request.config.getoption("--plot"):
        # Convergence analysis plots
        plt.figure(figsize=(6, 5))
        plt.loglog(nx_values, inf_norms, 'bo-', linewidth=2, markersize=8, 
            label=r'Infinity Norm ($L^\infty$)')
        plt.xlabel(r'Grid Points ($n_x = n_z$)')
        plt.ylabel(r'Infinity Norm Error ($L^\infty$)')
        plt.title('Convergence Study: '+ r'$L^\infty$'+ f' Error vs Grid Resolution\n({bc_type} BC)')
        plt.grid(True, alpha=0.3)

        # Add theoretical convergence lines for reference
        dx_values = [1.0/(nx-1) for nx in nx_values]
        # Plot theoretical convergence rate reference line (slope = 2)
        theoretical_line = np.array(dx_values) ** 2 * inf_norms[-1] / dx_values[-1] ** 2
        plt.loglog(nx_values, theoretical_line, 'r--', alpha=0.7, 
               label='Theoretical convergence rate = 2')
        plt.legend()

        # plt.subplot(1, 2, 2)
        # plt.loglog(nx_values, rel_l2_norms, 'ro-', linewidth=2, markersize=8, 
        #         label=f'L2 norm ({bc_type} BC)')
        # plt.xlabel('Number of x grid points (nx=nz)')
        # plt.ylabel('L2 Norm Error')
        # plt.title(f'Convergence Study: Rel L2 Norm vs Grid Resolution ({bc_type} BC)')
        # plt.grid(True, alpha=0.3)
        # plt.loglog(nx_values, np.array(dx_values)**2 * rel_l2_norms[0] / dx_values[0]**2, 
        #         'r--', alpha=0.7, label='O(dx² + dz²) reference')
        # plt.legend()

        plt.tight_layout()
        plt.show()
    
    if request.config.getoption("--plot_error"):
        # Plot the error for each grid size
        plt.figure(figsize=(15, 8))

        # Compute vmin/vmax for real and imaginary parts for consistent color mapping
        real_min = min(np.real(solution).min(), np.real(exact_solution).min())
        real_max = max(np.real(solution).max(), np.real(exact_solution).max())
        imag_min = min(np.imag(solution).min(), np.imag(exact_solution).min())
        imag_max = max(np.imag(solution).max(), np.imag(exact_solution).max())

        # Top row: Real parts
        plt.subplot(2, 3, 1)
        plt.imshow(np.real(solution), aspect='auto', cmap='viridis', vmin=real_min, vmax=real_max)
        plt.colorbar(label='Re(Numerical)')
        plt.title(f'Numerical Solution (Real) - {bc_type} BC')
        plt.xlabel('z')
        plt.ylabel('x')

        plt.subplot(2, 3, 2)
        plt.imshow(np.real(exact_solution), aspect='auto', cmap='viridis', vmin=real_min, vmax=real_max)
        plt.colorbar(label='Re(Exact)')
        plt.title(f'Exact Solution (Real) - {bc_type} BC')
        plt.xlabel('z')
        plt.ylabel('x')

        plt.subplot(2, 3, 3)
        plt.imshow(np.real(error), aspect='auto', cmap='viridis')
        plt.colorbar(label='Re(Error)')
        plt.title(f'Error (Real) - {bc_type} BC')
        plt.xlabel('z')
        plt.ylabel('x')

        # Bottom row: Imaginary parts
        plt.subplot(2, 3, 4)
        plt.imshow(np.imag(solution), aspect='auto', cmap='viridis', vmin=imag_min, vmax=imag_max)
        plt.colorbar(label='Im(Numerical)')
        plt.title(f'Numerical Solution (Imag) - {bc_type} BC')
        plt.xlabel('z')
        plt.ylabel('x')

        plt.subplot(2, 3, 5)
        plt.imshow(np.imag(exact_solution), aspect='auto', cmap='viridis', vmin=imag_min, vmax=imag_max)
        plt.colorbar(label='Im(Exact)')
        plt.title(f'Exact Solution (Imag) - {bc_type} BC')
        plt.xlabel('z')
        plt.ylabel('x')

        plt.subplot(2, 3, 6)
        plt.imshow(np.imag(error), aspect='auto', cmap='viridis')
        plt.colorbar(label='Im(Error)')
        plt.title(f'Error (Imag) - {bc_type} BC')
        plt.xlabel('z')
        plt.ylabel('x')

        plt.suptitle(f'{bc_type} Grid {nx}x{nz}, thin_sample: {thin_sample}, full_system_solver: {full_system_solver}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # Assert that the error decreases
    for i in range(1, len(inf_norms)):
        assert inf_norms[i] < inf_norms[i - 1], f"Error norm did not decrease: {inf_norms[i]} >= {inf_norms[i - 1]}"
