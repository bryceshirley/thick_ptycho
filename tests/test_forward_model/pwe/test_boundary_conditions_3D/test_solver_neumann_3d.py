import matplotlib.pyplot as plt
import numpy as np
import pytest
from thick_ptycho.simulation.ptycho_object import SampleSpace
from thick_ptycho.forward_model.base_solver import ForwardModel
from thick_ptycho.utils.visualisations import Visualisation


def u_nm(a, n, m):
    """Returns the exact solution for a given n and m."""
    return lambda x, y, z: np.exp(-a*((n**2)+(m**2))*(np.pi**2)*z)*np.cos(n*np.pi*x)*np.cos(m*np.pi*y)


def compute_error(nx, ny, nz, thin_sample, full_system_solver):
    """Computes the Frobenius norm of the error between the exact and computed solutions."""
    bc_type = "neumann"
    probe_type = "neumann_test"  # Sum of sinusoids
    wave_number = 100

    # Set continuous space limits
    xlims = [0, 1]
    ylims = [0, 1]
    zlims = [0, 2]

    continuous_dimensions = [
        xlims,
        ylims,
        zlims
    ]

    # Set detector shape, probe radius, step size, and no of scan points in each coordinate (All in pixels)
    probe_dimensions = [nx, ny]
    nx, ny = probe_dimensions          # Number of pixels in x and y directions
    propagation_slices = nz  # Number of z slices
    discrete_dimensions = [
        nx,
        ny,
        propagation_slices
    ]
    scan_points = 1
    step_size = 0

    # Create a sample space object
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

    # Create a forward model object
    forward_model = ForwardModel(sample_space,
                                 thin_sample=thin_sample,
                                 full_system_solver=full_system_solver)

    # Solve the experiment
    solution = forward_model.solve()[0, :, :, :]

    a = 1j / (2 * wave_number)

    # Define grid points
    x = np.linspace(xlims[0], xlims[1], nx)
    y = np.linspace(ylims[0], ylims[1], ny)
    z = np.linspace(zlims[0], zlims[1], nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Compute the exact solution
    exact_solution = u_nm(a, 1, 1)(X, Y, Z) + 0.5*u_nm(a, 2, 2)(X, Y, Z) + \
        0.2*u_nm(a, 5, 5)(X, Y, Z)

    # Compute the relative RMSE
    return exact_solution, solution


@pytest.mark.parametrize("thin_sample", [True, False], ids=["Thin", "Thick"])
@pytest.mark.parametrize("full_system_solver", [True, False], ids=["AllAtOnce", "Iterations"])
def test_error(thin_sample, full_system_solver, request):
    """
    Test that the error norm decreases as the grid resolution increases for a 2D problem.
    Here the spatial domain is discretized with nx, ny, and nz grid points.
    For plotting error, show imshow of the final z-slice (xy plane).
    """
    nx_values = [8, 16, 32, 64, 128]
    ny_values = nx_values  # Square grid in xy plane
    nz_values = []
    # rmse_errors = []
    inf_norms = []
    bc_type = "Neumann 3D"

    print(
        f"\n=== CONVERGENCE STUDY: {bc_type.upper()} BOUNDARY CONDITIONS ===")
    print(f"thin_sample: {thin_sample}, full_system_solver: {full_system_solver}\n")

    for i, nx in enumerate(nx_values):
        ny = nx  # square grid in xy plane
        nz = nx  # Keep nz proportional to nx
        nz_values.append(nz)

        print(f"Computing solution for nx={nx}, ny={ny}, nz={nz}")
        # You need to provide compute_error_2d to return (exact_solution, solution)
        # each of shape (nx, ny, nz)
        exact_solution, solution = compute_error(
            nx, ny, nz, thin_sample, full_system_solver)
        error = solution - exact_solution

        error_norm = np.max(np.abs(error))

        # rmse = np.sqrt(np.mean(np.abs(error)**2))
        # error_norm = np.linalg.norm(error) / np.linalg.norm(exact_solution)

        # Store errors for plotting
        # rmse_errors.append(rmse)
        inf_norms.append(error_norm)

    # Print convergence rates
    print(f"\nConvergence Analysis ({bc_type} BC):")
    print("nx\tny\tnz\tRate\tInfinity Norm")
    print("-" * 60)
    for i in range(len(nx_values)):
        if i > 0:
            ratio = inf_norms[i-1] / inf_norms[i]
            rate = np.log2(ratio)
            print(f"{nx_values[i]}\t{ny_values[i]}\t{nz_values[i]}\t{rate:.2f}\t{inf_norms[i]:.6f}")
        else:
            print(f"{nx_values[i]}\t{ny_values[i]}\t{nz_values[i]}\t-\t{inf_norms[i]:.6f}")

    # Check if plotting is enabled
    if request.config.getoption("--plot"):
        # Convergence analysis plots
        plt.figure(figsize=(6, 5))
        plt.loglog(nx_values, inf_norms, 'bo-', linewidth=2, markersize=8, 
            label=r'Infinity Norm ($L^\infty$)')
        plt.xlabel(r'Grid Points ($n_x = n_y = n_z$)')
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
        # Plot only the final z-slice of the last computed solution and error
        final_z = solution.shape[2] - 1  # last z index
        sol_slice = solution[:, :, final_z]
        exact_slice = exact_solution[:, :, final_z]
        error_slice = sol_slice - exact_slice

        # Setup consistent color mapping
        real_min = min(np.real(sol_slice).min(), np.real(exact_slice).min())
        real_max = max(np.real(sol_slice).max(), np.real(exact_slice).max())
        imag_min = min(np.imag(sol_slice).min(), np.imag(exact_slice).min())
        imag_max = max(np.imag(sol_slice).max(), np.imag(exact_slice).max())

        plt.figure(figsize=(15, 6))

        # Real parts
        plt.subplot(2, 3, 1)
        plt.imshow(np.real(sol_slice), cmap='viridis', vmin=real_min, vmax=real_max)
        plt.colorbar(label='Re(Numerical)')
        plt.title(f'Numerical Solution (Real) - {bc_type} BC')
        plt.xlabel('y')
        plt.ylabel('x')

        plt.subplot(2, 3, 2)
        plt.imshow(np.real(exact_slice), cmap='viridis', vmin=real_min, vmax=real_max)
        plt.colorbar(label='Re(Exact)')
        plt.title(f'Exact Solution (Real) - {bc_type} BC')
        plt.xlabel('y')
        plt.ylabel('x')

        plt.subplot(2, 3, 3)
        plt.imshow(np.real(error_slice), cmap='viridis')
        plt.colorbar(label='Re(Error)')
        plt.title(f'Error (Real) - {bc_type} BC')
        plt.xlabel('y')
        plt.ylabel('x')

        # Imaginary parts
        plt.subplot(2, 3, 4)
        plt.imshow(np.imag(sol_slice), cmap='viridis', vmin=imag_min, vmax=imag_max)
        plt.colorbar(label='Im(Numerical)')
        plt.title(f'Numerical Solution (Imag) - {bc_type} BC')
        plt.xlabel('y')
        plt.ylabel('x')

        plt.subplot(2, 3, 5)
        plt.imshow(np.imag(exact_slice), cmap='viridis', vmin=imag_min, vmax=imag_max)
        plt.colorbar(label='Im(Exact)')
        plt.title(f'Exact Solution (Imag) - {bc_type} BC')
        plt.xlabel('y')
        plt.ylabel('x')

        plt.subplot(2, 3, 6)
        plt.imshow(np.imag(error_slice), cmap='viridis')
        plt.colorbar(label='Im(Error)')
        plt.title(f'Error (Imag) - {bc_type} BC')
        plt.xlabel('y')
        plt.ylabel('x')

        plt.suptitle(f'{bc_type} Grid {nx}x{ny}x{nz} (final z-slice), thin_sample: {thin_sample}, full_system_solver: {full_system_solver}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# Assert error decreases
    for i in range(1, len(inf_norms)):
        assert inf_norms[i] < inf_norms[i -
                                            1], f"Error norm did not decrease: {inf_norms[i]} >= {inf_norms[i - 1]}"

