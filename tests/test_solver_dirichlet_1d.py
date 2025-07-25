# Import the paraxial solver
import matplotlib.pyplot as plt
import numpy as np
import pytest
from thick-ptycho.sample_space.sample_space import SampleSpace
from thick-ptycho.forward_model.solver import ForwardModel


def u_nm(a, n):
    """Returns the exact solution for a given n in 1D."""
    return lambda x, z: np.exp(-a * (n**2) * (np.pi**2) * z) * np.sin(n * np.pi * x)

def compute_error(nx, nz, thin_sample, full_system):
    """Computes the Frobenius norm of the error between the exact and computed solutions in 1D."""
    bc_type = "dirichlet"
    probe_type = "dirichlet_test"  # Sum of sinusoids
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
                                 full_system_solver=full_system)

    # Solve the experiment
    solution = forward_model.solve()[0, :, :]

    a = 1j / (2 * wave_number)

    # Define grid points
    x = np.linspace(xlims[0], xlims[1], nx+2)
    z = np.linspace(zlims[0], zlims[1], nz)
    X, Z = np.meshgrid(x, z, indexing='ij')

    # Compute the exact solution
    exact_solution = u_nm(a, 1)(X, Z) + 0.5 * u_nm(a, 2)(X, Z) + 0.2 * u_nm(a, 5)(X, Z)

    # Compute the relative RMSE
    return exact_solution, solution

@pytest.mark.parametrize("thin_sample", [True, False], ids=["Thin", "Full"])
@pytest.mark.parametrize("full_system", [True, False], ids=["AllAtOnce", "Iterations"])
def test_error(thin_sample, full_system, request):
    """Test that the error norm decreases as the grid resolution increases."""
    nz = 8
    grid_sizes = [8, 16, 32, 64, 128]
    errors = []

    print(f"\nDirichlet Boundary Condition Forward Refinement Test, thin_sample: {thin_sample}, full_system: {full_system}\n")
    for i, nx in enumerate(grid_sizes):
        nz = int(nx/2)

        exact_solution, solution = compute_error(nx, nz, thin_sample, full_system)
        error = solution - exact_solution
        mse = np.mean(np.abs(error)**2)
        rmse = np.sqrt(mse)
        norm_UE = np.sqrt(np.mean(np.abs(exact_solution)**2))
        relative_rmse = rmse / norm_UE

        # Store errors for plotting
        errors.append(relative_rmse)

        print(f"Grid size: {nx}x{nz}, Error norm: {relative_rmse:.6f}")

    # Check if plotting is enabled
    if request.config.getoption("--plot"):
        # Plot the convergence
        plt.figure()
        plt.loglog(grid_sizes, errors, marker='o', label='Error Norm')
        plt.xlabel('Grid Size nx')
        plt.ylabel('Error Norm')
        plt.title('Convergence Plot,thin_sample: {}, full_system: {}'.format(thin_sample, full_system))
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()

    # Assert that the error decreases
    for i in range(1, len(errors)):
        assert errors[i] < errors[i - 1], f"Error norm did not decrease: {errors[i]} >= {errors[i - 1]}"
