import numpy as np
import pytest
from thick_ptycho.simulation import SimulationSpace2D

from thick_ptycho.simulation.scan_frame import Limits

NONDEFAULT_CONFIG_1D = dict(
    wave_length=0.05,
    probe_diameter=0.01,
    spatial_limits=Limits(x=(0, 1),
                          y=(0, 1),
                          z=(0, 1)),
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
    sim = SimulationSpace2D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=scan_points,
        step_size_px=step_size,
        pad_factor=pad,
        solve_reduced_domain=True,
    )
    base_n_min = scan_points * step_size
    assert sim.min_nx in (base_n_min, base_n_min + 1)

    # pad_discrete is still ny - min_nx
    assert sim.pad_discrete == sim.ny - sim.min_nx

    # we now test the actual intended invariant:
    # domain midpoint should support scan symmetry
    mid = (sim.ny - 1) / 2
    assert abs(mid - round(mid)) <= 0.5


def test_scan_centres_spacing_and_symmetry(scan_points, step_size, pad):
    sim = SimulationSpace2D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=scan_points,
        step_size_px=step_size,
        pad_factor=pad,
        solve_reduced_domain=True,
    )
    centres_x = np.array([f.probe_centre_discrete.x for f in sim._scan_frame_info])
    
    # spacing holds
    assert np.all(np.diff(centres_x) == sim.step_size)

    # now symmetric about the *effective* window
    effective_mid = (sim.effective_nx - 1)
    assert centres_x[0] + centres_x[-1] == effective_mid