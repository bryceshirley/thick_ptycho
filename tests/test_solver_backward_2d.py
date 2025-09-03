import numpy as np
import pytest
from thick_ptycho.sample_space.sample_space import SampleSpace
from thick_ptycho.forward_model.solver import ForwardModel


def run_solver_with_objects(boundary_condition, initial_condtion):
    bc_type = boundary_condition
    probe_type = initial_condtion

    wave_number = 2e-1

    # Continuous space limits (micrometers)
    xlims = [0, 300]                    # X limits in micrometers
    ylims = [0, 300]                    # Y limits in micrometers
    zlims = [0, 10]                    # Z limits in micrometers
    continuous_dimensions = [
        xlims,
        ylims,
        zlims
        ]                               # Continuous dimensions in micrometers

    # Discrete space parameters
    propagation_slices = 30              # Number of z slices
    probe_dimensions = [32, 32]         # Detector shape can be different from discrete_dimensions
    probe_diameter = 12            # Diameter of the probe in micrometers
    scan_points = 1                     # Single probe
    step_size = 0                      # Step size in pixels

    nx, ny = probe_dimensions          # Number of pixels in x and y directions
    discrete_dimensions = [
        nx,
        ny,
        propagation_slices
        ]     

    sample_space = SampleSpace(
        continuous_dimensions, # sample space dimensions in nanometers (x, z) or (x, y, z)
        discrete_dimensions,   # sample space dimensions in pixels (nx, nz) or (nx, ny, nz)
        probe_dimensions,        # shape of the detector in pixels - subset of (nx) or (nx, ny)
        scan_points,           # number of ptychography scan points or sqrt(scan_points) for square scan
        step_size,
        bc_type,               # boundary condition type (impedance, dirichlet, neumann)
        probe_type,
        wave_number,            # wavenumber in 1/nm
        probe_diameter=probe_diameter
    )

    delta = 1e-4
    beta = 1e-6j
    refractive_index1 = 1 - delta + beta
    refractive_index2 = 1 - 0.9 * delta + beta
    guassian_blur = 0
    sample_space.add_object('prism', refractive_index1,
                            side_length=0.4*xlims[1],
                            centre=(xlims[1]*0.7, xlims[1]*0.7, zlims[1]*0.5),
                            depth=zlims[1]*0.8, guassian_blur=guassian_blur)
    sample_space.add_object('cuboid', refractive_index2,
                            side_length=0.2*xlims[1],
                            centre=(xlims[1]*0.3, xlims[1]*0.3, zlims[1]*0.5),
                            depth=zlims[1]*0.8, guassian_blur=guassian_blur)

    sample_space.generate_sample_space()

    forward_model = ForwardModel(sample_space,
                                 full_system_solver=False,
                                 thin_sample=True)

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


@pytest.mark.parametrize("thin_sample", [True, False])
@pytest.mark.parametrize("boundary_condition", ["dirichlet", "neumann", "impedance"])
@pytest.mark.parametrize("initial_condition", ["constant", "gaussian", "sinusoidal",
                                               "complex_exp", "dirichlet_test",
                                               "neumann_test", "airy_disk", "disk"])
def test_solver_with_objects(boundary_condition, initial_condition):
    run_solver_with_objects(boundary_condition, initial_condition)
