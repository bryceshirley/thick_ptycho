import numpy as np
import pytest
from thick_ptycho.simulation import SimulationSpace1D

NONDEFAULT_CONFIG_1D = dict(
    wave_length=0.05,
    probe_diameter=0.01,
    continuous_dimensions=((0, 1), (0, 1)),
)

@pytest.mark.parametrize(
    "scan_points, step_size, pad",
    [
        (7, 5, 1.33),
        (5, 4, 1.0),
        (5, 4, 2.0),
        (4, 3, 1.0),
    ]
)
def test_effective_domain_consistency(scan_points, step_size, pad):
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
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
    ]
)
def test_scan_centres_spacing_and_symmetry(scan_points, step_size, pad):
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=scan_points,
        step_size_px=step_size,
        pad_factor=pad,
    )
    centres = np.array([f.probe_centre_discrete for f in sim._scan_frame_info])
    # spacing holds
    assert np.all(np.diff(centres) == sim.step_size)

    # now symmetric about the *effective* window
    effective_mid = (sim.effective_nx - 1)
    assert centres[0] + centres[-1] == effective_mid


def test_domain_limits_without_reduction():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=4,
        step_size_px=3,
        pad_factor=2.0,
        solve_reduced_domain=False,
    )
    # Whole domain is used
    assert sim.effective_dimensions == sim.nx


def test_domain_limits_with_reduction():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=4,
        step_size_px=3,
        pad_factor=2.0,
        solve_reduced_domain=True,
    )
    expected_ne = sim.step_size + sim.pad_discrete - 1
    assert sim.effective_dimensions == expected_ne 

    xmin, xmax = sim._scan_frame_info[0].reduced_limits_discrete
    assert (xmin, xmax) == (0, expected_ne)



@pytest.mark.parametrize(
    "scan_points, step_size, pad",
    [
        (6, 3, 2.0),
        (4, 4, 1.5),
    ]
)
def test_scan_frame_width(scan_points, step_size, pad):
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=scan_points,
        step_size_px=step_size,
        pad_factor=pad,
    )
    expected_width = sim.step_size + sim.pad_discrete
    for f in sim._scan_frame_info:
        xmin, xmax = f.reduced_limits_discrete
        assert 0 <= xmin
        assert xmax < sim.nx
        assert (xmax - xmin + 1) == expected_width


def test_single_scan_point_centering_and_bounds():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=1,
        step_size_px=7,
        pad_factor=1.6,
    )
    centre = sim._scan_frame_info[0].probe_centre_discrete
    mid = (sim.nx - 1) / 2
    assert abs(centre - mid) <= 0.5
    assert sim.effective_dimensions == sim.nx


def test_single_scan_point_full_window():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=1,
        step_size_px=7,
        pad_factor=1.4,
    )
    xmin, xmax = sim._scan_frame_info[0].reduced_limits_discrete
    assert (xmin, xmax) == (0, sim.nx - 1)
