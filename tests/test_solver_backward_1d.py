import numpy as np
import pytest
from thick-ptycho.sample_space.sample_space import SampleSpace
from thick-ptycho.forward_model.solver import ForwardModel


def run_solver_with_objects(boundary_condition, probe_type, thin_sample, full_system):
    bc_type = boundary_condition
    nz = 2**4
    nx = 32
    wave_number = 100

    # Set continuous space limits
    xlims = [0, 1]
    zlims = [0, 1]
    continuous_dimensions = [xlims, zlims]

    # Set detector shape, probe radius, step size, and no of scan points in each coordinate (All in pixels)
    propagation_slices = nz
    probe_dimensions = [nx]
    probe_diameter = int(nx * 0.8)
    scan_points = 1
    step_size = 0
    discrete_dimensions = [
        nx,
        propagation_slices
    ]

    sample_space = SampleSpace(
        # sample space dimensions in nanometers (x, z) or (x, y, z)
        continuous_dimensions,
        # sample space dimensions in pixels (nx, nz) or (nx, ny, nz)
        discrete_dimensions,
        # shape of the detector in pixels - subset of (nx) or (nx, ny)
        probe_dimensions,
        # number of ptychography scan points or sqrt(scan_points) for square scan
        scan_points,
        step_size,
        # boundary condition type (impedance, dirichlet, neumann)
        bc_type,
        probe_type,
        wave_number,            # wavenumber in 1/nm
        probe_diameter=probe_diameter
    )

    delta = 1e-4
    beta = 1e-6j
    refractive_index1 = - delta + beta
    refractive_index2 = - 0.9 * delta + beta
    guassian_blur = 0
    sample_space.add_object('rectangle', refractive_index1, side_length=0.2*xlims[1],
                            centre=(xlims[1]*0.7, zlims[1]*0.5),
                            depth=zlims[1]*0.5, guassian_blur=guassian_blur)
    sample_space.add_object('triangle', refractive_index2, side_length=0.2*xlims[1],
                            centre=(xlims[1]*0.3, zlims[1]*0.5),
                            depth=zlims[1]*0.5, guassian_blur=guassian_blur)

    sample_space.generate_sample_space()

    forward_model = ForwardModel(sample_space,
                                 full_system_solver=full_system,
                                 thin_sample=thin_sample)

    forward_solution = forward_model.solve()
    initial_condition = forward_solution[..., -1].copy()
    backward_solution = forward_model.solve(reverse=True,
                                            initial_condition=initial_condition)

    # Compute RMSE between forward and backward solutions
    rmse = np.sqrt(
        np.mean(np.abs(forward_solution[..., 0] - backward_solution[..., -1]) ** 2))

    print(
        f'\nBackward Boundary Condition: {boundary_condition}\nError Norm: {rmse}\n')
    assert rmse < 1e-8, f"Error norm too high: {rmse}"

@pytest.mark.parametrize("thin_sample", [True, False], ids=["Thin", "Full"])
@pytest.mark.parametrize("full_system", [True, False], ids=["AllAtOnce", "Iterations"])
@pytest.mark.parametrize("boundary_condition", ["dirichlet", "neumann", "impedance"])
@pytest.mark.parametrize("initial_condition", ["constant", "gaussian", "sinusoidal",
                                               "complex_exp", "dirichlet_test",
                                               "neumann_test", "airy_disk", "disk"])
def test_solver_with_objects(boundary_condition, initial_condition,
                             thin_sample,
                             full_system):
    """Test the solver with various boundary conditions and initial conditions."""
    run_solver_with_objects(boundary_condition, initial_condition, thin_sample,
                            full_system)
