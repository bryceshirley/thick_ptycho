import matplotlib.pyplot as plt
import numpy as np
import pytest
from thickptypy.sample_space.sample_space import SampleSpace
from thickptypy.forward_model.solver import ForwardModel
from thickptypy.utils.visualisations import Visualisation

def u_nm(a,n, m):
    """Returns the exact solution for a given n and m."""
    return lambda x,y,z: np.exp(-a*((n**2)+(m**2))*(np.pi**2)*z)*np.cos(n*np.pi*x)*np.cos(m*np.pi*y) 


def compute_error(nx, ny, nz, thin_sample, full_system):
    """Computes the Frobenius norm of the error between the exact and computed solutions."""
    bc_type = "impedance"
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
    return exact_solution, solution


@pytest.mark.parametrize("thin_sample", [True, False], ids=["Thin", "Full"])
@pytest.mark.parametrize("full_system", [True, False], ids=["AllAtOnce", "Iterations"])
def test_error(thin_sample, full_system, request):
    """
    Test that the error norm decreases as the grid resolution increases for a 2D problem.
    Here the spatial domain is discretized with nx, ny, and nz grid points.
    For plotting error, show imshow of the final z-slice (xy plane).
    """
    nx_values = [8, 16, 32, 64, 128]
    ny_values = nx_values  # For simplicity, keep square xy grid
    nz_values = []
    
    rmse_errors = []
    rel_l2_norms = []
    bc_type = "Impedance 3D"

    print(f"\n=== CONVERGENCE STUDY: {bc_type.upper()} BOUNDARY CONDITIONS ===")
    print(f"thin_sample: {thin_sample}, full_system: {full_system}\n")

    for i, nx in enumerate(nx_values):
        ny = nx  # square grid in xy plane
        nz = nx  # Keep nz proportional to nx
        nz_values.append(nz)

        print(f"Computing solution for nx={nx}, ny={ny}, nz={nz}")
        # You need to provide compute_error_2d to return (exact_solution, solution)
        # each of shape (nx, ny, nz)
        exact_solution, solution = compute_error(nx, ny, nz, thin_sample, full_system)
        error = solution - exact_solution

        # RMSE over entire 3D volume
        rmse = np.sqrt(np.mean(np.abs(error)**2))
        error_norm = np.linalg.norm(error.ravel()) / np.linalg.norm(exact_solution.ravel())

        rmse_errors.append(rmse)
        rel_l2_norms.append(error_norm)

    # Print convergence rates
    print(f"\nConvergence Analysis ({bc_type} BC):")
    print("nx\tny\tnz\tRMSE\t\tRatio\tRate\tRelative L2 Norm")
    print("-" * 80)
    for i, (nx, ny, nz, rmse, l2) in enumerate(zip(nx_values, ny_values, nz_values, rmse_errors, rel_l2_norms)):
        if i > 0:
            ratio = rmse_errors[i-1] / rmse
            rate = np.log2(ratio)
            print(f"{nx}\t{ny}\t{nz}\t{rmse:.6e}\t{ratio:.2f}\t{rate:.2f}\t{l2:.15f}")
        else:
            print(f"{nx}\t{ny}\t{nz}\t{rmse:.6e}\t-\t-\t{l2:.15f}")

    # Check if plotting is enabled
    if request.config.getoption("--plot"):
        # Convergence analysis plots
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.loglog(nx_values, rmse_errors, 'bo-', linewidth=2, markersize=8, 
                label=f'RMSE ({bc_type} BC)')
        plt.xlabel('Number of x grid points (nx=ny=nz)')
        plt.ylabel('RMSE Error')
        plt.title(f'Convergence Study: RMSE vs Grid Resolution ({bc_type} BC)')
        plt.grid(True, alpha=0.3)

        # Compute dz values (assuming nz = nx)
        dz_values = [1.0 / (nz - 1) for nz in nz_values]  # nz = nx for this test
        plt.loglog(nx_values, (np.array(dz_values)) * rmse_errors[0] / dz_values[0], 
                'r--', alpha=0.7, label='O(dx² + dy² + dz) = O(dz) reference')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.loglog(nx_values, rel_l2_norms, 'ro-', linewidth=2, markersize=8, 
                label=f'L2 norm ({bc_type} BC)')
        plt.xlabel('Number of x grid points (nx=ny=nz)')
        plt.ylabel('L2 Norm Error')
        plt.title(f'Convergence Study: Rel L2 Norm vs Grid Resolution ({bc_type} BC)')
        plt.grid(True, alpha=0.3)
        plt.loglog(nx_values, (np.array(dz_values)) * rmse_errors[0] / dz_values[0], 
                'r--', alpha=0.7, label='O(dx² + dy² + dz) = O(dz) reference')
        plt.legend()

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

        plt.suptitle(f'{bc_type} Grid {nx}x{ny}x{nz} (final z-slice), thin_sample: {thin_sample}, full_system: {full_system}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # Assert error decreases
    for i in range(1, len(rmse_errors)):
        assert rmse_errors[i] < rmse_errors[i - 1], f"Error norm did not decrease: {rmse_errors[i]} >= {rmse_errors[i - 1]}"