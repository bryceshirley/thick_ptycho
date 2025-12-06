import matplotlib.pyplot as plt
import numpy as np
import pytest

from thick_ptycho.forward_model import (
    PWEFullLUSolver,
    PWEFullPinTSolver,
    PWEIterativeLUSolver,
)
from thick_ptycho.forward_model.pwe.operators.finite_differences.boundary_condition_test import (
    BoundaryConditionsTest,
)
from thick_ptycho.forward_model.pwe.operators.utils import BoundaryType
from thick_ptycho.simulation.config import ProbeConfig, ProbeType, SimulationConfig
from thick_ptycho.simulation.ptycho_probe import create_ptycho_probes
from thick_ptycho.simulation.simulation_space import create_simulation_space

# ----------------------------------------------------------------------------------------
#  Analytical 1D exact solutions
# ----------------------------------------------------------------------------------------


def _u_nm_neumann(n, k):
    a = 1j / (2 * k)
    return lambda x, z: np.exp(-a * (n**2) * (np.pi**2) * z) * np.cos(n * np.pi * x)


def _u_nm_dirichlet(n, k):
    a = 1j / (2 * k)
    return lambda x, z: np.exp(-a * (n**2) * (np.pi**2) * z) * np.sin(n * np.pi * x)


def get_exact_solution(bc_type, k, X, Z):
    """Return the analytical solution corresponding to the boundary type."""
    if bc_type in (BoundaryType.IMPEDANCE, BoundaryType.NEUMANN):
        return (
            _u_nm_neumann(1, k)(X, Z)
            + 0.5 * _u_nm_neumann(2, k)(X, Z)
            + 0.2 * _u_nm_neumann(5, k)(X, Z)
        )
    elif bc_type == BoundaryType.DIRICHLET:
        return (
            _u_nm_dirichlet(1, k)(X, Z)
            + 0.5 * _u_nm_dirichlet(5, k)(X, Z)
            + 0.2 * _u_nm_dirichlet(9, k)(X, Z)
        )
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


def compute_exact_and_numerical(nx, nz, Solver, bc_type, limits):
    """
    Compute both analytical and numerical solutions for given grid size.
    Returns (exact_solution, numerical_solution).
    """
    k = 100
    wavelength = 2 * np.pi / k

    probe_config = ProbeConfig(
        type=select_probe_type(bc_type),
        wave_length=wavelength,
    )

    sim_cfg = SimulationConfig(
        probe_config=probe_config,
        step_size_px=nx,
        nz=nz,
        spatial_limits=limits,
    )

    sim_space = create_simulation_space(sim_cfg)
    probes = create_ptycho_probes(sim_space)

    test_bcs = (
        BoundaryConditionsTest(sim_space) if bc_type == BoundaryType.IMPEDANCE else None
    )

    solver = Solver(sim_space, probes, bc_type=bc_type.value, test_bcs=test_bcs)
    numerical_solution = solver.solve().squeeze()
    # pdb.set_trace()

    # Define coordinate grid
    x = np.linspace(*limits.x, sim_space.nx)
    z = np.linspace(*limits.z, sim_space.nz)
    X, Z = np.meshgrid(x, z, indexing="ij")

    exact_solution = get_exact_solution(bc_type, k, X, Z)
    return exact_solution, numerical_solution


# ----------------------------------------------------------------------------------------
#  Plotting utilities
# ----------------------------------------------------------------------------------------


def plot_convergence(nx_values, inf_norms, bc_type, Solver):
    """Plot L-inf norm convergence curve."""
    plt.figure(figsize=(6, 5))
    plt.loglog(
        nx_values,
        inf_norms,
        "bo-",
        linewidth=2,
        markersize=8,
        label=r"Infinity Norm ($L^\infty$)",
    )
    plt.xlabel(r"Grid Points ($n_x = n_y = n_z$)")
    plt.ylabel(r"Infinity Norm Error ($L^\infty$)")
    plt.title(
        "Convergence Study: "
        + r"$L^\infty$"
        + f" Error vs Grid Resolution\n({bc_type} BC)- {Solver.__name__}"
    )
    plt.grid(True, alpha=0.3)

    # Add theoretical convergence lines for reference
    dx_values = [1.0 / (nx - 1) for nx in nx_values]
    # Plot theoretical convergence rate reference line (slope = 2)
    theoretical_line = np.array(dx_values) ** 2 * inf_norms[-1] / dx_values[-1] ** 2
    plt.loglog(
        nx_values,
        theoretical_line,
        "r--",
        alpha=0.7,
        label="Theoretical convergence rate = 2",
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_maps(exact, numerical, error, bc_type, Solver):
    """Visualize real/imag parts of exact, numerical, and error fields."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    titles = ["Numerical", "Exact", "Error"]

    def _plot(ax, data, title, label, vmin=None, vmax=None):
        im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("z")
        ax.set_ylabel("x")
        fig.colorbar(im, ax=ax, label=label)

    real_min = min(np.real(exact).min(), np.real(exact).min())
    real_max = max(np.real(numerical).max(), np.real(exact).max())
    imag_min = min(np.imag(numerical).min(), np.imag(exact).min())
    imag_max = max(np.imag(numerical).max(), np.imag(exact).max())

    for j, (data_real, data_imag) in enumerate(
        [
            (np.real(numerical), np.imag(numerical)),
            (np.real(exact), np.imag(exact)),
            (np.real(error), np.imag(error)),
        ]
    ):
        if titles[j] == "Error":
            _plot(axes[0, j], data_real, f"{titles[j]} (Real)", "Re")
            _plot(axes[1, j], data_imag, f"{titles[j]} (Imag)", "Im")
        else:
            _plot(
                axes[0, j], data_real, f"{titles[j]} (Real)", "Re", real_min, real_max
            )
            _plot(
                axes[1, j], data_imag, f"{titles[j]} (Imag)", "Im", imag_min, imag_max
            )

    fig.suptitle(f"{bc_type} BC — Grid {error.shape}, {Solver.__name__}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_probe_and_ew(exact, numerical, error):
    """Plot probe intensity and exit wave intensity."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))
    axes = axes.flatten()  # Convert to 1D array for easy indexing

    nx = exact.shape[0]

    def _plot(ax, data, title, idx=0):
        ax.plot(np.arange(nx), np.abs(data)[:, idx])
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("Amplitude")

    # Plot Exit Wave Amplitudes
    _plot(axes[0], numerical, "Numerical Exit Wave Amplitude", idx=-1)
    _plot(axes[1], exact, "Exact Exit Wave Amplitude", idx=-1)
    _plot(axes[2], error, "Error Amplitude", idx=-1)

    # Plot Probe Amplitudes
    _plot(axes[3], numerical, "Numerical Probe Amplitude", idx=0)
    _plot(axes[4], exact, "Exact Probe Amplitude", idx=0)
    _plot(axes[5], error, "Error Amplitude", idx=0)

    fig.suptitle("Probe and Exit Wave Amplitudes", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ----------------------------------------------------------------------------------------
#  Main test
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bc_type",
    [BoundaryType.NEUMANN, BoundaryType.IMPEDANCE, BoundaryType.DIRICHLET],
    ids=["Neumann", "Impedance", "Dirichlet"],
)
@pytest.mark.parametrize(
    "Solver",
    [PWEIterativeLUSolver, PWEFullPinTSolver, PWEFullLUSolver],
    ids=["Iterative", "FullPinT", "FullLU"],
)
def test_error_convergence(Solver, bc_type, request, limits_2d):
    """Validate second-order convergence for each solver and boundary type."""
    nx_values = [16, 32, 64]
    inf_norms = []

    print(f"\n=== {Solver.__name__} — {bc_type} BC ===")

    print("\nGrid\tInf-Norm\tRate")
    observed_rates = []
    for i, nx in enumerate(nx_values):
        nz = nx
        exact, numerical = compute_exact_and_numerical(
            nx, nz, Solver, bc_type, limits_2d
        )
        error = numerical - exact
        inf_norm = np.max(np.abs(error))
        inf_norms.append(inf_norm)

        if i == 0:
            print(f"{nx}\t{inf_norm:.3e}\t-")
        else:
            rate = np.log2(inf_norms[i - 1] / inf_norm)
            observed_rates.append(rate)
            print(f"{nx}\t{inf_norm:.3e}\t{rate:.2f}")

    # Optional plotting
    if request.config.getoption("--plot"):
        plot_convergence(nx_values, inf_norms, bc_type, Solver)

    if request.config.getoption("--plot_error"):
        plot_error_maps(exact, numerical, error, bc_type, Solver)

    if request.config.getoption("--plot_probe_and_ew"):
        plot_probe_and_ew(exact, numerical, error)

    # Assertions: monotonic decrease & second-order rate
    assert all(inf_norms[i] < inf_norms[i - 1] for i in range(1, len(inf_norms))), (
        f"Error did not decrease monotonically: {inf_norms}"
    )
    avg_rate = np.mean(observed_rates[-3:])
    assert 1.5 <= avg_rate <= 2.5, (
        f"Expected ~2nd order convergence, got {avg_rate:.2f}"
    )
