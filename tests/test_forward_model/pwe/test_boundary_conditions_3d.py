import matplotlib.pyplot as plt
import numpy as np
import pytest

from thick_ptycho.simulation.config import ProbeConfig, SimulationConfig, ProbeType
from thick_ptycho.simulation.simulation_space import create_simulation_space
from thick_ptycho.simulation.ptycho_object import create_ptycho_object
from thick_ptycho.simulation.ptycho_probe import create_ptycho_probes
from thick_ptycho.forward_model import (
    PWEIterativeLUSolver, PWEFullPinTSolver, PWEFullLUSolver
)
from thick_ptycho.simulation.scan_frame import Limits
from thick_ptycho.forward_model.pwe.operators.utils import BoundaryType
from thick_ptycho.forward_model.pwe.operators.finite_differences.boundary_condition_test import BoundaryConditionsTest


# ----------------------------------------------------------------------------------------
#  Analytical 3D exact solutions
# ----------------------------------------------------------------------------------------

def _u_nm_neumann(n, m, k):
    """Returns the exact solution for a given n and m."""
    a = 1j / (2 * k)
    return lambda x, y, z: np.exp(-a*((n**2)+(m**2))*(np.pi**2)*z)*np.cos(n*np.pi*x)*np.cos(m*np.pi*y)


def _u_nm_dirichlet(n, m, k):
    """Returns the exact solution for a given n and m."""
    a = 1j / (2 * k)
    return lambda x, y, z: np.exp(-a*((n**2)+(m**2))*(np.pi**2)*z)*np.sin(n*np.pi*x)*np.sin(m*np.pi*y)


def get_exact_solution(bc_type, k, X, Y, Z):
    """Return the analytical solution corresponding to the boundary type."""
    if bc_type in (BoundaryType.IMPEDANCE, BoundaryType.NEUMANN):
        return (
            _u_nm_neumann(1, 1, k)(X, Y, Z) 
            + 0.5*_u_nm_neumann(2, 2, k)(X, Y, Z)
            + 0.2*_u_nm_neumann(5, 5, k)(X, Y, Z)
        )
    elif bc_type == BoundaryType.DIRICHLET:
        exact_solution = (
            _u_nm_dirichlet(1, 1, k)(X, Y, Z)
            + 0.5 * _u_nm_dirichlet(5, 5, k)(X, Y, Z)
            + 0.2 * _u_nm_dirichlet(9, 9, k)(X, Y, Z)
        )
        # Stricly enforce the boundary conditions
        exact_solution[:, 0, :] = 0
        exact_solution[:, -1, :] = 0
        exact_solution[0, :, :] = 0
        exact_solution[-1, :, :] = 0
        return exact_solution
    raise ValueError(f"Unsupported BC type: {bc_type}")


# ----------------------------------------------------------------------------------------
#  Utility selection helpers
# ----------------------------------------------------------------------------------------

def select_probe_type(bc_type):
    if bc_type in (BoundaryType.IMPEDANCE, BoundaryType.NEUMANN):
        return ProbeType.NEUMANN_TEST
    elif bc_type == BoundaryType.DIRICHLET:
        return ProbeType.DIRICHLET_TEST
    raise ValueError(f"Unsupported BC type: {bc_type}")


# ----------------------------------------------------------------------------------------
#  Core solver setup
# ----------------------------------------------------------------------------------------

def compute_exact_and_numerical(nx, nz, Solver, bc_type):
    """
    Compute both analytical and numerical solutions for given grid size.
    Returns (exact_solution, numerical_solution).
    """
    k = 100
    wavelength = 2 * np.pi / k
    limits = Limits(x=(0, 1),y=(0,1), z=(0, 2), units="meters")

    probe_config = ProbeConfig(
        type=select_probe_type(bc_type),
        wave_length=wavelength,
    )

    sim_cfg = SimulationConfig(
        probe_config=probe_config,
        step_size_px=nx,
        nz=nz,
        spatial_limits=limits,
        solve_reduced_domain=False,
    )

    sim_space = create_simulation_space(sim_cfg)
    obj = create_ptycho_object(sim_space)
    probes = create_ptycho_probes(sim_space)

    test_bcs = BoundaryConditionsTest(sim_space) if bc_type == BoundaryType.IMPEDANCE else None

    solver = Solver(
        sim_space, obj, probes,
        bc_type=bc_type.value,
        test_bcs=test_bcs
    )
    solution = solver.solve().squeeze()

    # Define coordinate grid
    x = np.linspace(*limits.x, sim_space.nx)
    y = np.linspace(*limits.y, sim_space.ny)
    z = np.linspace(*limits.z, sim_space.nz)
    X, Y, Z = np.meshgrid(x,y, z, indexing='ij')

    exact_solution = get_exact_solution(bc_type, k, X, Y, Z)
    return exact_solution, solution


# ----------------------------------------------------------------------------------------
#  Plotting utilities
# ----------------------------------------------------------------------------------------

def plot_convergence(nx_values, inf_norms, bc_type, Solver):
    """Plot L-inf norm convergence curve."""
    plt.figure(figsize=(6, 5))
    plt.loglog(nx_values, inf_norms, 'bo-', linewidth=2, markersize=8, label=r'$L^\infty$ norm')
    plt.xlabel(r'Grid points ($n_x = n_z$)')
    plt.ylabel(r'$L^\infty$ Error')
    plt.title(f'Convergence Study ({bc_type}) — {Solver.__name__}')
    plt.grid(True, alpha=0.3)

    dx_values = [1.0 / (nx - 1) for nx in nx_values]
    theoretical_line = np.array(dx_values)**2 * inf_norms[-1] / dx_values[-1]**2
    plt.loglog(nx_values, theoretical_line, 'r--', alpha=0.7, label='Theoretical slope = 2')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_exit_waves(exact, numerical, error, nx, nz, bc_type, Solver):
    """Visualize real/imag parts of exact, numerical, and error fields."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    titles = ['Numerical', 'Exact', 'Error']

    def _plot(ax, data, title, label, vmin=None, vmax=None):
        im = ax.imshow(data, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, label=label)

    real_min = min(np.real(numerical).min(), np.real(exact).min())
    real_max = max(np.real(numerical).max(), np.real(exact).max())
    imag_min = min(np.imag(numerical).min(), np.imag(exact).min())
    imag_max = max(np.imag(numerical).max(), np.imag(exact).max())

    for j, (data_real, data_imag) in enumerate([(np.real(numerical), np.imag(numerical)),
                                                (np.real(exact), np.imag(exact)),
                                                (np.real(error), np.imag(error))]):
        if j == 2:
            _plot(axes[0, j], data_real, f"{titles[j]} (Real)", 'Re')
            _plot(axes[1, j], data_imag, f"{titles[j]} (Imag)", 'Im')
        else:
            _plot(axes[0, j], data_real, f"{titles[j]} (Real)", 'Re', vmin=real_min, vmax=real_max)
            _plot(axes[1, j], data_imag, f"{titles[j]} (Imag)", 'Im', vmin=imag_min, vmax=imag_max)

    fig.suptitle(f"{bc_type} BC — Grid {nx}x{nx}×{nz}, {Solver.__name__}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ----------------------------------------------------------------------------------------
#  Main test
# ----------------------------------------------------------------------------------------

# dont test
#@pytest.mark.skip(reason="Tests are too slow for regular CI runs.")
@pytest.mark.parametrize("bc_type", [
    BoundaryType.NEUMANN,
    BoundaryType.IMPEDANCE,
    BoundaryType.DIRICHLET
], ids=["Neumann", "Impedance", "Dirichlet"])
@pytest.mark.parametrize("Solver", [
    PWEIterativeLUSolver,
    PWEFullPinTSolver,
    PWEFullLUSolver
], ids=["Iterative", "FullPinT", "FullLU"])
def test_error_convergence(Solver, bc_type, request):
    """Validate second-order convergence for each solver and boundary type."""
    nx_values = [8, 16, 32]
    if bc_type == BoundaryType.DIRICHLET:
        nx_values.append(64)  # Finer grid for Dirichlet due to higher errors
        if Solver != PWEIterativeLUSolver:
            # Skip solvers that are too slow on fine grids
            pytest.mark.skip(reason="Solver too slow for fine grids in CI.")

    inf_norms = []
    observed_rates = []

    print(f"\n=== {Solver.__name__} — {bc_type} BC ===")

    print("\nGrid\tInf-Norm\tRate")
    for i, nx in enumerate(nx_values):
        nz = nx
        exact, numerical = compute_exact_and_numerical(nx, nz, Solver, bc_type)
        error = numerical - exact
        inf_norm = np.max(np.abs(error))
        inf_norms.append(inf_norm)
        if i == 0:
            print(f"{nx}\t{inf_norm:.3e}\t-")
        else:
            rate = np.log2(inf_norms[i - 1] / inf_norm)
            print(f"{nx}\t{inf_norm:.3e}\t{rate:.2f}")
            observed_rates.append(rate)

    # Optional plotting
    if request.config.getoption("--plot"):
        plot_convergence(nx_values, inf_norms, bc_type, Solver)

    if request.config.getoption("--plot_error"):
        plot_exit_waves(exact[:,:,-1], numerical[:,:,-1], error[:,:,-1], nx, nz, bc_type, Solver)

    # Assertions: monotonic decrease & second-order rate
    assert all(inf_norms[i] < inf_norms[i - 1] for i in range(1, len(inf_norms))), \
        f"Error did not decrease monotonically: {inf_norms}"
    assert 1.5 <= observed_rates[-1] <= 2.5, \
        f"Expected ~2nd order convergence, got {observed_rates[-1]:.2f}"