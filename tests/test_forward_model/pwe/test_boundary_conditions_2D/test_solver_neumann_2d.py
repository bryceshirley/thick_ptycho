# Import the paraxial solver
import matplotlib.pyplot as plt
import numpy as np
import pytest
from thick_ptycho.simulation.config import SimulationConfig, ProbeType
from thick_ptycho.simulation.simulation_space import create_simulation_space
from thick_ptycho.simulation.ptycho_object import create_ptycho_object
from thick_ptycho.simulation.ptycho_probe import create_ptycho_probes
from thick_ptycho.forward_model import PWEIterativeLUSolver, PWEFullPinTSolver, PWEFullLUSolver
import os

from thick_ptycho.forward_model.pwe.operators.utils import BoundaryType


def u_nm(n, k):
    a = 1j / (2 * k)
    """Returns the exact solution for a given n in 1D."""
    return lambda x, z: np.exp(-a * (n**2) * (np.pi**2) * z) * np.cos(n * np.pi * x)


def compute_error(nx, nz,Solver):
    """Computes the Frobenius norm of the error between the exact and computed solutions in 1D."""
    k = 100
    wave_length = 2 * np.pi / k

    # Set continuous space limits
    xlims = [0, 1]
    zlims = [0, 2]

    sim_config = SimulationConfig(
        probe_type="neumann_test",      # triggers sinusoidal probe generation
        wave_length=wave_length,
        step_size_px=nx,
        nz = nz,
        continuous_dimensions=(xlims, zlims),
    )

    simulation_space = create_simulation_space(sim_config)
    obj = create_ptycho_object(simulation_space)
    probes = create_ptycho_probes(simulation_space)

    solver = Solver(simulation_space, obj, probes, bc_type="neumann")
    solution = solver.solve().squeeze()                        # shape: (scan, x, z)

    # Define grid points
    x = np.linspace(xlims[0], xlims[1], simulation_space.nx)
    z = np.linspace(zlims[0], zlims[1], simulation_space.nz)
    X, Z = np.meshgrid(x, z, indexing='ij')

    # Compute the exact solution
    exact_solution = u_nm(1,k)(X, Z) + 0.5 * u_nm(2, k)(X, Z) + 0.2 * u_nm(5, k)(X, Z)

    # Compute the relative RMSE
    return exact_solution, solution

@pytest.mark.parametrize("Solver", [PWEIterativeLUSolver, 
                                    PWEFullPinTSolver,
                                    #PWEFullLUSolver
                                    ], ids=["SolverIterative", 
                                            "SolverFullPinT", 
                                            #"SolverFullLU"
                                            ])
def test_error(Solver, request):
    """Test that the error norm decreases as the grid resolution increases."""

    nx_values = [16, 32, 64, 128, 256, 512]
    nz_values = []
    # rmse_errors = []
    inf_norms = []
    bc_type = "Neumann 2D"

    print(f"\n=== CONVERGENCE STUDY: {bc_type.upper()} BOUNDARY CONDITIONS ===")
    print(f"Solver: {Solver}\n")
    for i, nx in enumerate(nx_values):
        nz = nx
        nz_values.append(nz)

        print(f"Computing solution for nx={nx}, nz={nz}")
        exact_solution, solution = compute_error(nx, nz, Solver)
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
        plt.title('Convergence Study: '+ r'$L^\infty$'+ f' Error vs Grid Resolution\n({bc_type} BC) Solver: {Solver.__name__}')
        plt.grid(True, alpha=0.3)

        # Add theoretical convergence lines for reference
        dx_values = [1.0/(nx-1) for nx in nx_values]
        # Plot theoretical convergence rate reference line (slope = 2)
        theoretical_line = np.array(dx_values) ** 2 * inf_norms[-1] / dx_values[-1] ** 2
        plt.loglog(nx_values, theoretical_line, 'r--', alpha=0.7, 
               label='Theoretical convergence rate = 2')
        plt.legend()

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

        plt.suptitle(f'{bc_type} Grid {nx}x{nz}, Solver: {Solver.__name__}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # Assert that the error decreases
    for i in range(1, len(inf_norms)):
        assert inf_norms[i] < inf_norms[i - 1], f"Error norm did not decrease: {inf_norms[i]} >= {inf_norms[i - 1]}"


    # Compute observed convergence rates
    observed_rates = []
    for i in range(1, len(inf_norms)):
        ratio = inf_norms[i-1] / inf_norms[i]
        rate = np.log2(ratio)
        observed_rates.append(rate)

    avg_rate = np.mean(observed_rates[-3:])  # focus on fine-grid regime

    print(f"\nExpected convergence rate ≈ 2, observed average rate = {avg_rate:.2f}")

    # Assert second order convergence within tolerance
    assert 1.5 <= avg_rate <= 2.5, \
        f"Expected ~second-order convergence, but observed rate was {avg_rate:.3f}"
