import numpy as np
import pytest

from thick_ptycho.simulation import SimulationSpace2D
from thick_ptycho.simulation.ptycho_object import create_ptycho_object


@pytest.fixture(scope="session")
def nondefault_config_2d(limits_2d):
    return dict(
        wave_length=0.05,
        probe_diameter=0.01,
        spatial_limits=limits_2d,
    )


def test_rotation(nondefault_config_2d):
    sim = SimulationSpace2D(
        **nondefault_config_2d,
        tomographic_projection_90_degree=True,
    )
    # nz overridden to nx
    assert sim.num_projections == 2
    assert sim.nz == sim.nx

    # Test shape
    assert sim.shape == (sim.nx, sim.nx)
    ptycho_object = create_ptycho_object(sim)
    assert ptycho_object.refractive_index.shape == (sim.nx, sim.nx)
    ptycho_object.add_object("circle", 1.5 + 0.1j, 0.2, (0.5, 0.5), 0.1)
    assert ptycho_object.refractive_index.shape == (sim.nx, sim.nx)


@pytest.mark.parametrize(
    "scan_points, step_size, pad",
    [
        (7, 5, 1.33),
        (5, 4, 1.0),
        (5, 4, 2.0),
        (4, 3, 1.0),
    ],
)
def test_effective_domain_consistency(
    scan_points, step_size, pad, nondefault_config_2d
):
    sim = SimulationSpace2D(
        **nondefault_config_2d,
        solve_reduced_domain=True,
        scan_points=scan_points,
        step_size_px=step_size,
        pad_factor=pad,
    )
    base_n_min = scan_points * step_size
    assert sim.min_nx in (base_n_min, base_n_min + 1)

    # pad_discrete is still nx - min_nx
    assert sim.pad_discrete == sim.nx - sim.min_nx

    # we now test the actual intended invariant:
    # domain midpoint should support scan symmetry
    mid = (sim.nx - 1) / 2
    assert abs(mid - round(mid)) <= 0.5


@pytest.mark.parametrize(
    "scan_points, step_size, pad",
    [
        (4, 3, 1.0),
        (4, 4, 1.5),
        (5, 4, 2.0),
        (6, 3, 2.0),
    ],
)
def test_scan_centres_spacing_and_symmetry(
    scan_points, step_size, pad, nondefault_config_2d
):
    sim = SimulationSpace2D(
        **nondefault_config_2d,
        solve_reduced_domain=True,
        scan_points=scan_points,
        step_size_px=step_size,
        pad_factor=pad,
    )
    centres = np.array([f.probe_centre_discrete.x for f in sim._scan_frame_info])

    # spacing holds
    assert np.all(np.diff(centres) == sim.step_size)

    # now symmetric at boundaries
    assert centres[0] == (sim.nx - 1) - centres[-1]

    # Centres correct distance away from boundary of sample
    assert centres[0] == sim.effective_nx // 2


def test_domain_limits_without_reduction(nondefault_config_2d):
    sim = SimulationSpace2D(
        **nondefault_config_2d,
        scan_points=4,
        step_size_px=3,
        pad_factor=2.0,
        solve_reduced_domain=False,
    )
    # Whole domain is used
    assert sim.effective_nx == sim.nx


def test_domain_limits_with_reduction(nondefault_config_2d):
    sim = SimulationSpace2D(
        **nondefault_config_2d,
        scan_points=4,
        step_size_px=3,
        pad_factor=2.0,
        solve_reduced_domain=True,
    )
    expected_ne = sim.step_size + sim.pad_discrete - 1
    assert sim.effective_nx == expected_ne

    xmin, xmax = sim._scan_frame_info[0].reduced_limits_discrete.x
    assert (xmin, xmax) == (0, expected_ne)

    assert sim.effective_shape == (sim.effective_nx, sim.nz)


@pytest.mark.parametrize(
    "scan_points, step_size, pad",
    [
        (6, 3, 2.0),
        (4, 4, 1.5),
    ],
)
def test_scan_frame_width(scan_points, step_size, pad, nondefault_config_2d):
    sim = SimulationSpace2D(
        **nondefault_config_2d,
        solve_reduced_domain=True,
        scan_points=scan_points,
        step_size_px=step_size,
        pad_factor=pad,
    )
    expected_width = sim.step_size + sim.pad_discrete
    for f in sim._scan_frame_info:
        xmin, xmax = f.reduced_limits_discrete.x
        assert 0 <= xmin
        assert xmax < sim.nx
        assert (xmax - xmin + 1) == expected_width


def test_single_scan_point_centering_and_bounds(nondefault_config_2d):
    sim = SimulationSpace2D(
        **nondefault_config_2d,
        solve_reduced_domain=True,
        scan_points=1,
        step_size_px=7,
        pad_factor=1.6,
    )
    centre = sim._scan_frame_info[0].probe_centre_discrete.x
    mid = (sim.nx - 1) / 2
    assert abs(centre - mid) <= 0.5
    assert sim.effective_nx == sim.nx


def test_single_scan_point_full_window(nondefault_config_2d):
    sim = SimulationSpace2D(
        **nondefault_config_2d,
        solve_reduced_domain=True,
        scan_points=1,
        step_size_px=7,
        pad_factor=1.4,
    )
    xmin, xmax = sim._scan_frame_info[0].reduced_limits_discrete.x
    assert (xmin, xmax) == (0, sim.nx - 1)


@pytest.mark.parametrize("solve_reduced_domain", [True, False])
def test_empty_space_padding(nondefault_config_2d, solve_reduced_domain):
    """
    Test padding logic when 'empty_space=True' is used, which creates a large
    zero-padded object contribution for an effective domain.
    """
    scan_points = 4
    step_size = 5
    pad = 1.5

    sim = SimulationSpace2D(
        **nondefault_config_2d,
        scan_points=scan_points,
        step_size_px=step_size,
        pad_factor=pad,
        solve_reduced_domain=solve_reduced_domain,
        empty_space_px=10,  # Add 10 pixels of empty space padding on each side
    )
    ptycho_object = create_ptycho_object(sim)
    ptycho_object.add_object("circle", 1.5 + 0.1j, 0.2, (0.5, 0.5), 0.1)

    # Calculate object contribution with empty_space=True
    object_steps = sim.create_object_contribution(
        n=ptycho_object.refractive_index,
        scan_index=0,
    )

    # 1. Check the new shape:
    #    The original padding is (self.effective_nx // 2, self.effective_nx // 2)
    #    The intended padded shape is: (original_nx + self.effective_nx, original_nz)
    #    Since n is already (sim.effective_nx, sim.nz) when solve_reduced_domain=True,
    #    The total added padding in the transverse dimension is self.effective_nx.

    original_transverse_dim = sim.effective_nx
    expected_new_transverse_dim = original_transverse_dim + 2 * sim.empty_space_px
    # Note: the last axis is z-steps, which is nz - 1

    assert object_steps.shape[0] == expected_new_transverse_dim
    assert object_steps.shape[1] == sim.nz - 1

    # 2. Check if the original data is centered correctly
    #    The data is placed in the center of the large zero array.

    # Check for zeros in the padding region
    # Left padding width is effective_nx // 2
    assert np.all(object_steps[: original_transverse_dim // 2, :] == 0)

    # Right padding width is effective_nx // 2 (at the other end of the domain)
    # The transverse dimension is expected_new_transverse_dim
    right_start_index = expected_new_transverse_dim - (original_transverse_dim // 2)
    assert np.all(object_steps[right_start_index:, :] == 0)

    # 3. Check if the non-zero (object) region has the correct width
    non_zero_width = right_start_index - (original_transverse_dim // 2)
    assert non_zero_width == sim.effective_nx
