import matplotlib.pyplot as plt
import numpy as np
import pytest
from thick-ptycho.sample_space.sample_space import SampleSpace
from thick-ptycho.forward_model.solver import ForwardModel
from thick-ptycho.utils.visualisations import Visualisation


def u_nm(a, n, m):
    """Returns the exact solution for a given n and m."""
    return lambda x, y, z: np.exp(-a*((n**2)+(m**2))*(np.pi**2)*z)*np.cos(n*np.pi*x)*np.cos(m*np.pi*y)


def compute_error_norm(nx, ny, nz, thin_sample, full_system):
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

    # Create a visualisation object
    visualisation = Visualisation(sample_space=sample_space)

    # Create a forward model object
    forward_model = ForwardModel(sample_space,
                                 thin_sample=thin_sample,
                                 full_system_solver=full_system)

    # Solve the experiment
    solution = forward_model.solve(test_impedance=True)[0, :, :, :]

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
    return exact_solution, solution, visualisation


@pytest.mark.parametrize("thin_sample", [True, False])
@pytest.mark.parametrize("full_system", [True, False])
def test_error(thin_sample, full_system, request):
    """Test that the error norm decreases as the grid resolution increases."""
    nz = 8
    grid_sizes = [8, 16, 32, 64]
    errors = []

    print(f"\nNeumann Boundary Condition Forward Refinement Test, thin_sample: {thin_sample}, full_system: {full_system}\n")
    for i, nx in enumerate(grid_sizes):
        ny = nx
        exact_solution, solution, visualisation = compute_error_norm(nx,
                                                                     ny,
                                                                     nz,
                                                                     thin_sample,
                                                                     full_system)
        error = solution - exact_solution
        mse = np.mean(np.abs(error)**2)
        rmse = np.sqrt(mse)
        norm_UE = np.sqrt(np.mean(np.abs(exact_solution)**2))
        relative_rmse = rmse / norm_UE

        # Store errors for plotting
        errors.append(relative_rmse)

        print(f"Grid size: {nx}x{ny}x{nz}, Error norm: {relative_rmse:.6f}")

        if request.config.getoption("--plot_error"):
            visualisation.plot(solution=error, plot_phase=False)

    # Check if plotting is enabled
    if request.config.getoption("--plot"):
        # Plot the convergence
        plt.figure()
        plt.loglog(grid_sizes, errors, marker='o', label='Error Norm')
        plt.xlabel('Grid Size (nx=ny)')
        plt.ylabel('Error Norm')
        plt.title('Convergence Plot,thin_sample: {}, full_system: {}'.format(thin_sample, full_system))
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()

    # Assert that the error decreases
    for i in range(1, len(errors)):
        assert errors[i] < errors[i -
                                  1], f"Error norm did not decrease: {errors[i]} >= {errors[i - 1]}"
